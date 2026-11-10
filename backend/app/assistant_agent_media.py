"""Bounded ID-only media tools for the trusted assistant supervisor.

No model command/path access, provider calls, Run mutation or version publication.
Candidate blobs may be retained explicitly for the authenticated preview route.
"""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from functools import wraps
from fractions import Fraction
import hashlib

from pathlib import Path
import tempfile
import time
from uuid import uuid4

from . import assistant_media as media
from .assistant_images import read_jpeg_dimensions
from .assistant_access import AuthorizedArtifactCatalog, canonical_uuid
from .assistant_contracts import EditPlan
from .assistant_executor import render_local_candidate
from .assistant_models import AssistantArtifact
from .assistant_service import AssistantAdmissionService, AssistantCapabilityError
from .assistant_store import get_version_artifact, verify_artifact
from .assistant_subtitles import get_subtitle_context


class AgentToolUnavailable(media.AssistantMediaError):
    """Known local capability/resource limitation; safe to report and continue."""


def _bounded_tool(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        previous = self._operation_deadline
        self._operation_deadline = time.monotonic() + min(600, max(0, self.budget.deadline-time.monotonic()))
        try:
            return method(self, *args, **kwargs)
        finally:
            self._operation_deadline = previous
    return invoke


class AgentMediaService:
    MAX_FRAME_BYTES = 524288
    MAX_CALL_BYTES = 2 * 1024 * 1024
    MAX_TOTAL_FRAMES = 32
    MAX_TOTAL_FRAME_BYTES = 8 * 1024 * 1024
    MAX_SCRATCH_BYTES = 512 * 1024 * 1024

    def __init__(self, session_factory, *, storage, policy, target_version_id,
                 asr=None, timeout_sec=300, cancelled=None):
        if type(timeout_sec) not in (int, float) or not 0 < timeout_sec <= 1200:
            raise ValueError("invalid agent media timeout")
        self.factory, self.storage, self.policy = session_factory, storage, policy
        self.target = canonical_uuid(target_version_id)
        self.asr = asr
        self.budget = media._budget(timeout_sec, cancelled)
        self._temporary = tempfile.TemporaryDirectory(prefix="agent-", dir=storage.root / "staging")
        self.work = Path(self._temporary.name)
        self.frames, self.candidates = {}, {}
        self.total_frame_bytes = 0
        self._closed = False
        self._last_scan = 0.0
        self._operation_deadline = None

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._temporary.cleanup()
        self._closed = True

    def check(self):
        if self._closed:
            raise media.AssistantMediaError("agent media tools are closed")
        self.budget.check()
        now = time.monotonic()
        if self._operation_deadline is not None and now >= self._operation_deadline:
            raise media.AssistantMediaTimeout("agent media operation exceeded time budget")
        if now - self._last_scan >= .1:
            self._last_scan = now
            total = sum(p.stat().st_size for p in self.work.rglob("*") if p.is_file())
            if total > self.MAX_SCRATCH_BYTES:
                raise media.AssistantMediaError("agent scratch byte budget exceeded")
        return False

    def __call__(self):
        return self.check()

    def _remaining(self):
        self.check()
        return max(.000001, min(self.budget.deadline, self._operation_deadline or self.budget.deadline) - time.monotonic())

    def _catalog(self, session):
        return AuthorizedArtifactCatalog(session, policy=self.policy,
            is_available=lambda a: bool(verify_artifact(a, storage=self.storage, cancelled=self)))

    def _resolve(self, asset_id, version_id):
        self.check()
        with self.factory() as session:
            descriptor = self._catalog(session).resolve_editable_reference(target_version_id=self.target,
                asset_id=asset_id, version_id=version_id)
            path = verify_artifact(session.get(AssistantArtifact, descriptor.asset_id),
                storage=self.storage, cancelled=self)
        return descriptor, path

    def _candidate(self, candidate_id):
        candidate_id = canonical_uuid(candidate_id)
        if candidate_id not in self.candidates:
            raise media.AssistantMediaError("unknown candidate")
        item = self.candidates[candidate_id]
        for asset, version in item["references"]:
            self._resolve(asset, version)
        self.check()
        if media._hash(item["path"], self.budget) != item["evidence"]["sha256"]:
            raise media.AssistantMediaError("candidate integrity failure")
        return item

    def _source(self, *, asset_id=None, version_id=None, candidate_id=None):
        if candidate_id is not None:
            if asset_id is not None or version_id is not None:
                raise ValueError("select exactly one media source")
            candidate = self._candidate(candidate_id)
            return candidate["path"], candidate["evidence"]["sha256"], {"candidate_id": candidate_id}
        if asset_id is None or version_id is None:
            raise ValueError("source IDs are required")
        descriptor, path = self._resolve(asset_id, version_id)
        if descriptor.media_kind not in ("video", "audio"):
            raise media.AssistantMediaError("unsupported source kind")
        return path, descriptor.sha256, {"asset_id": descriptor.asset_id, "version_id": descriptor.version_id}

    @_bounded_tool
    def search_sources(self, query="", *, after_asset_id=None, limit=20):
        if not isinstance(query, str) or len(query) > 200 or type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("invalid source query")
        # The trusted importer only snapshots this Run's allowlisted ORIGINAL.
        from .assistant_agent_sources import import_original_source, SourceCapabilityUnavailable
        with self.factory.begin() as session:
            target = self._catalog(session).require_editable_target(self.target)
            original_source = None
            limitations = []
            try:
                # Nested transaction rolls back only an unavailable import, not
                # any unrelated source artifacts in the caller's transaction.
                with session.begin_nested():
                    original_source = import_original_source(session, run_id=target.run_id, base_version_id=self.target,
                        storage=self.storage, policy=self.policy, cancelled=self, timeout_sec=min(60,self._remaining()))
            except SourceCapabilityUnavailable:
                limitations.append("original_source_unavailable")
            if original_source is None and not limitations:
                limitations.append("original_source_unavailable")
        if "original" in query.casefold() and original_source is None:
            raise AgentToolUnavailable("original source unavailable for local inspection")
        if "original" in query.casefold() and original_source is not None:
            return {"sources":[original_source.model_dump(mode="json")],"next_after_asset_id":None}
        with self.factory() as session:
            page = self._catalog(session).list_editable_artifacts(target_version_id=self.target,
                after_asset_id=after_asset_id, limit=limit)
            records = []
            for descriptor in page.artifacts:
                if descriptor.media_kind not in ("video", "audio"):
                    continue
                path = verify_artifact(session.get(AssistantArtifact, descriptor.asset_id), storage=self.storage, cancelled=self)
                if descriptor.media_kind == "video":
                    info = media.probe_trusted_media(path, timeout_sec=self._remaining(), cancelled=self)
                    duration, has_audio, fps = info.video_duration_sec, info.has_audio, info.fps
                else:
                    from .assistant_audio import probe_trusted_audio
                    info = probe_trusted_audio(path, timeout_sec=self._remaining(), cancelled=self)
                    duration, has_audio, fps = info.duration_sec, True, None
                artifact = session.get(AssistantArtifact, descriptor.asset_id)
                original = artifact.metadata_json.get("source_role") == "original_project_source"
                label = "Original project source video" if original else f"{descriptor.media_kind} {descriptor.asset_id[:8]}"
                # Query is a bounded inert label filter, never a SQL/path query.
                if query.strip() and not any(word.casefold() in label.casefold() for word in query.split()):
                    continue
                verify_artifact(artifact, storage=self.storage, cancelled=self)
                records.append({**{key: getattr(descriptor,key) for key in ("asset_id","version_id","run_id","project_id","media_kind")},
                    "duration_sec":duration,"has_audio":has_audio,"fps":fps,"label":label})
        self.check()
        return {"sources": records, "next_after_asset_id": page.next_after_asset_id, "limitations":limitations}

    @_bounded_tool
    def inspect_frames(self, *, timestamps_sec, asset_id=None, version_id=None, candidate_id=None):
        if not isinstance(timestamps_sec, (tuple,list)) or not 1 <= len(timestamps_sec) <= 8:
            raise ValueError("one to eight frame timestamps required")
        if len(self.frames) + len(timestamps_sec) > self.MAX_TOTAL_FRAMES:
            raise AgentToolUnavailable("turn frame budget exceeded")
        times = [media._number(t,"timestamp_sec") for t in timestamps_sec]
        path, source_hash, identity = self._source(asset_id=asset_id, version_id=version_id,candidate_id=candidate_id)
        info = media.probe_trusted_media(path, timeout_sec=self._remaining(), cancelled=self)
        if info.video_duration_sec > 120 or max(info.width,info.height)>1920 or info.width*info.height>2073600:
            raise AgentToolUnavailable("frame source exceeds inspection resource limits")
        if any(t < 0 or t >= info.video_duration_sec for t in times):
            raise ValueError("frame timestamp outside source")
        fps = Fraction(info.fps)
        records, call_bytes = [], 0
        # ffprobe rounds stream duration decimals. The normalized CFR probe
        # proves an integral frame count, so round the count and exclude its end.
        last_frame = max(0, round(Fraction(str(info.video_duration_sec)) * fps) - 1)
        for requested in times:
            position = Fraction(str(requested)) * fps
            nearest = round(position)
            # Canonical decimal representations of n/fps can land just below n
            # (7.083333 * 24 = 169.999992). Snap only within 1e-5 of a FRAME;
            # all other positions keep floor semantics, including real gaps.
            frame_number = nearest if abs(position-nearest) <= Fraction(1,100000) else int(position)
            frame_number = min(last_frame,frame_number)
            actual = float(frame_number / fps)
            frame_id = str(uuid4())
            output = self.work / f"{frame_id}.jpg"
            ratio = min(1, 768/max(info.width,info.height))
            width, height = max(2, int(info.width*ratio)//2*2), max(2,int(info.height*ratio)//2*2)
            media._run([media.FFMPEG,"-v","error","-nostdin","-n","-noautorotate","-ss",f"{max(0,actual-.000001):.9f}","-i",str(path),
                "-frames:v","1","-vf",f"scale={width}:{height}","-q:v","3","-threads","1","-fs",str(self.MAX_FRAME_BYTES+1),str(output)],media._budget(self._remaining(),self),inputs=(path,),outputs=(output,))
            data = output.read_bytes()
            call_bytes += len(data)
            if (not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9") or len(data)>self.MAX_FRAME_BYTES
                    or call_bytes>self.MAX_CALL_BYTES or self.total_frame_bytes+call_bytes>self.MAX_TOTAL_FRAME_BYTES):
                raise AgentToolUnavailable("frame image byte budget exceeded")
            if read_jpeg_dimensions(data) != (width, height):
                raise media.AssistantMediaError("invalid frame dimensions")
            # Validate entropy/pixels only inside the same mandatory sandbox.
            media._run([media.FFMPEG, "-v", "error", "-xerror", "-nostdin",
                "-i", str(output), "-frames:v", "1", "-threads", "1", "-f", "null", "-"],
                media._budget(self._remaining(), self), inputs=(output,))
            self.check()
            evidence = {"frame_id":frame_id,**identity,"timestamp_sec":actual,"sha256":hashlib.sha256(data).hexdigest(),
                "source_sha256":source_hash,"size_bytes":len(data),"mime_type":"image/jpeg","width":width,"height":height}
            self.frames[frame_id] = {"path":output,"evidence":evidence,"identity":identity}
            records.append(evidence)
        # Recheck both authority and source bytes after decoding.
        self._source(**identity)
        self.total_frame_bytes += call_bytes
        return {"frames":records}

    def read_frame(self, frame_id):
        self.check()
        item = self.frames.get(canonical_uuid(frame_id))
        if item is None:
            raise media.AssistantMediaError("unknown frame")
        self._source(**item["identity"])
        data = item["path"].read_bytes()
        if hashlib.sha256(data).hexdigest() != item["evidence"]["sha256"]:
            raise media.AssistantMediaError("frame integrity failure")
        return data

    @_bounded_tool
    def transcribe_window(self, *, start_sec, end_sec, asset_id=None, version_id=None, candidate_id=None, language=None):
        if self.asr is None:
            raise AgentToolUnavailable("offline speech inspection unavailable")
        start, end = media._number(start_sec,"start_sec"), media._number(end_sec,"end_sec")
        if not 0 <= start < end or end-start > 60:
            raise ValueError("speech window must be within sixty seconds")
        path, source_hash, identity = self._source(asset_id=asset_id,version_id=version_id,candidate_id=candidate_id)
        if candidate_id is not None:
            # The credential-free ASR socket accepts immutable object keys only,
            # never scratch paths. Pinning a blob creates no creative version.
            pinned = self.storage.pin(path,cancelled=self)
            if pinned.sha256 != source_hash:
                raise media.AssistantMediaError("candidate changed before speech inspection")
            path = self.storage.verify(pinned,cancelled=self)
        from .assistant_audio import probe_trusted_audio
        audio_info = probe_trusted_audio(path,timeout_sec=min(30,self._remaining()),cancelled=self)
        # Normalized video allows its AAC track to end <=50ms before picture.
        # Clamp only that proven tiny tail; transcript evidence retains the
        # actual audio extent rather than claiming recognition of missing time.
        if audio_info.duration_sec < end <= audio_info.duration_sec + .050001:
            end = audio_info.duration_sec
        if not start < end <= audio_info.duration_sec:
            raise AgentToolUnavailable("speech window exceeds actual soundtrack extent")
        transcript = self.asr.transcribe(path,start_sec=start,end_sec=end,language=language,
            timeout_sec=min(120,self._remaining()),cancelled=self)
        self._source(**identity)
        words = [w.model_dump(mode="json") for w in transcript.words[:400]]
        common = {"source_start_sec":transcript.source_start_sec,"source_end_sec":transcript.source_end_sec,
            "source_duration_sec":transcript.source_duration_sec,"language":transcript.language,
            "language_probability":transcript.language_probability,"words":words,
            "coverage_complete":start == 0 and end == transcript.source_duration_sec,
            "words_truncated":len(transcript.words)>400,"timing_precision":"estimated_word",
            "pronunciation_verified":False,"speaker_verified":False,"review_required":True}
        if candidate_id is not None:
            return {"candidate_speech":[{**identity,"source_sha256":source_hash,**common}]}
        from .assistant_planner import PlannerSpeechEvidence
        return {"speech":[PlannerSpeechEvidence.model_validate({**identity,**common}).model_dump(mode="json")]}

    # --- AICP-345: read-only QA tools ------------------------------------

    def _qa_input(self, ref):
        ref = dict(ref)
        path, sha, identity = self._source(**ref)
        if "asset_id" in identity:
            with self.factory() as session:
                descriptor = self._catalog(session).resolve_editable_reference(target_version_id=self.target,
                    asset_id=identity["asset_id"], version_id=identity["version_id"])
                if descriptor.media_kind != "video":
                    raise media.AssistantMediaError("QA requires a video input")
                artifact = session.get(AssistantArtifact, identity["asset_id"])
                original = (artifact.metadata_json or {}).get("source_role") == "original_project_source"
        else:
            original = False
        return {"path": str(path), "identity": identity, "original": original,
                "ref": {**identity, "sha256": sha}}

    def _qa_recheck(self, *inputs):
        # Authority and bytes are rechecked after the (long) decode, as for frames.
        for item in inputs:
            self._source(**item["identity"])
        self.check()

    def _qa_run(self, function, *args, **kwargs):
        from . import qa_tools
        from .assistant_qa import ConfinedQAMedia, QAInputUnavailable
        backend = ConfinedQAMedia(remaining=self._remaining, check=self.check)
        try:
            with qa_tools.use_media(backend):
                return function(*args, **kwargs)
        except QAInputUnavailable as exc:
            raise AgentToolUnavailable(str(exc)) from None
        except qa_tools.QAToolError:
            # Bad window/geometry for this input; the decoder text never leaves.
            raise AgentToolUnavailable("QA tool could not measure this input") from None

    def _caption_config(self):
        from . import qa_tools
        from .models import VideoProject
        with self.factory() as session:
            target = self._catalog(session).require_editable_target(self.target)
            project = session.get(VideoProject, target.project_id)
            stored = project.caption_detector if project is not None else None
        try:
            return qa_tools.caption_config(stored), bool(stored)
        except qa_tools.QAToolError:
            raise AgentToolUnavailable("project caption detector config is invalid") from None

    @_bounded_tool
    def detect_captions(self, *, target, calibration, start_sec=None, duration_sec=None,
                        calibration_start_sec=None, calibration_duration_sec=None):
        from . import qa_tools
        from .assistant_qa import caption_evidence
        result = self._qa_input(target)
        source = self._qa_input(calibration)
        if not source["original"]:
            raise AgentToolUnavailable("calibrate on the Original project source video")
        config, project_config = self._caption_config()
        scan = self._qa_run(qa_tools.detect_captions, result["path"], calibrate=source["path"],
            calibrate_start=calibration_start_sec, calibrate_duration=calibration_duration_sec,
            start=start_sec, duration=duration_sec, config=config)
        self._qa_recheck(result, source)
        return caption_evidence(scan, target_ref=result["ref"], calibration_ref=source["ref"],
            start_sec=start_sec, duration_sec=duration_sec, calibration_start_sec=calibration_start_sec,
            calibration_duration_sec=calibration_duration_sec, project_config=project_config)

    @_bounded_tool
    def check_alignment(self, *, source, result, mode="timeline", source_start_sec=None, duration_sec=None):
        from . import qa_tools
        from .assistant_qa import alignment_evidence, match_evidence
        src, res = self._qa_input(source), self._qa_input(result)
        if mode == "timeline":
            out = self._qa_run(qa_tools.alignment_check, src["path"], res["path"], duration=duration_sec)
            self._qa_recheck(src, res)
            return alignment_evidence(out, source_ref=src["ref"], result_ref=res["ref"])
        if mode != "window":
            raise ValueError("unknown alignment mode")
        out = self._qa_run(qa_tools.match_times, src["path"], res["path"],
            source_start=source_start_sec, duration=duration_sec)
        self._qa_recheck(src, res)
        return self._qa_wrap(match_evidence, out, source_ref=src["ref"], result_ref=res["ref"])

    def _qa_wrap(self, builder, *args, **kwargs):
        from .assistant_qa import QAInputUnavailable
        try:
            return builder(*args, **kwargs)
        except QAInputUnavailable as exc:
            raise AgentToolUnavailable(str(exc)) from None

    @_bounded_tool
    def compare_frames(self, *, source, result, windows, fractions=(0.1, 0.5, 0.9), every_sec=None,
                       time_map="offset"):
        from . import qa_tools
        from .assistant_qa import MAX_COMPARE_SAMPLES, compare_evidence
        src, res = self._qa_input(source), self._qa_input(result)
        fractions = None if every_sec is not None else tuple(fractions)
        spec, count = [], 0
        for index, window in enumerate(windows):
            window = dict(window)
            item = {"source_start": window["source_start_sec"], "source_end": window["source_end_sec"],
                    "label": window.get("label") or f"w{index}"}
            if window.get("result_start_sec") is not None:
                item["result_start"] = window["result_start_sec"]
            if window.get("result_end_sec") is not None:
                item["result_end"] = window["result_end_sec"]
            count += len(qa_tools._sample_times(item["source_start"], item["source_end"],
                                                fractions=fractions, every_sec=every_sec))
            spec.append(item)
        if count > MAX_COMPARE_SAMPLES:
            raise AgentToolUnavailable("comparison requests more than 48 frame pairs")
        out = self._qa_run(qa_tools.comparison_sheet, src["path"], res["path"], None, windows=spec,
            fractions=fractions, every_sec=every_sec, time_map=time_map)
        self._qa_recheck(src, res)
        return compare_evidence(out, source_ref=src["ref"], result_ref=res["ref"], time_map=time_map)

    @_bounded_tool
    def script_captions(self, *, library_donors=()):
        from . import assistant_script_captions as script
        from .assistant_qa import ConfinedQAMedia
        from .models import Run
        donors = {}
        for donor in library_donors:
            donor = dict(donor)
            donors[int(donor["plan_index"])] = canonical_uuid(donor["run_segment_id"])
        with self.factory() as session:
            target = self._catalog(session).require_editable_target(self.target)
            selected = get_version_artifact(session, target.version_id)
            backend = ConfinedQAMedia(remaining=self._remaining, check=self.check)
            try:
                evidence = script.resolve(session, run=session.get(Run, target.run_id), library_donors=donors,
                    data_root=self.storage.data_root, check=self.check,
                    probe=lambda path: SimpleNamespace(duration_sec=backend.container_duration(path)))
            except script.ScriptCaptionsRefused as exc:
                raise AgentToolUnavailable(str(exc)) from None
            if selected is None or selected.sha256 != evidence["final_sha256"]:
                raise AgentToolUnavailable("the selected version is not the run's current result; "
                                           "script captions apply to the run result only")
        self.check()
        return evidence

    @_bounded_tool
    def render_candidate(self, plan):
        if len(self.candidates) >= 3:
            raise AgentToolUnavailable("candidate render budget exceeded")
        plan = EditPlan.model_validate(plan.model_dump(mode="json") if isinstance(plan,EditPlan) else plan)
        if plan.base_version_id != self.target:
            raise ValueError("candidate must use pinned base")
        service = AssistantAdmissionService(storage=self.storage,policy=self.policy,enabled=True)
        with self.factory() as session:
            try:
                snapshot = service._validate(session,plan,cancelled=self,timeout_sec=self._remaining())
            except AssistantCapabilityError as error:
                raise AgentToolUnavailable("candidate operation is unsupported by local renderer") from error
            base = get_version_artifact(session,self.target)
            base_path = verify_artifact(base,storage=self.storage,cancelled=self)
            base_id = base.id
            info = snapshot["base"]["media"]
            if info["video_duration_sec"] > 120 or max(info["width"],info["height"])>1920 or info["width"]*info["height"]>2073600:
                raise AgentToolUnavailable("candidate source exceeds inspection resource limits")
            subtitle_status = base.metadata_json.get("subtitle_status","unknown")
            track = get_subtitle_context(session,self.target,storage=self.storage,cancelled=self)
            clean_artifact = get_version_artifact(session,self.target,role='clean') if track else None
            clean_path = verify_artifact(clean_artifact,storage=self.storage,cancelled=self) if clean_artifact else None
            exact_audio_timing = clean_artifact.metadata_json.get('exact_audio_timing',False) if clean_artifact else False
        references = [(base_id,self.target)]
        donors = {}
        for op in plan.operations:
            ref = getattr(op,"source",None)
            if ref:
                descriptor, donor = self._resolve(ref.asset_id,ref.version_id)
                references.append((descriptor.asset_id,descriptor.version_id))
                donors[ref.asset_id] = donor
        candidate_id = str(uuid4())
        work = self.work/candidate_id
        work.mkdir()
        result, _, track, _ = render_local_candidate(plan,base_path,donors,work,self,self._remaining(),track=track,
            clean_path=clean_path,exact_audio_timing=exact_audio_timing)
        for asset, version in references:
            self._resolve(asset,version)
        output = Path(result.output_path)
        self._last_scan = 0
        self.check()
        from .assistant_agent_contracts import plan_fingerprint
        from .assistant_contracts import ExtendEndingOperation, ending_bounds
        if isinstance(plan.operations[0],ExtendEndingOperation):
            _, expected_end = ending_bounds(plan.operations[0],info['video_duration_sec'],info['fps'])
            actual = result.metadata  # Trusted primitive (and caption reburn) validated the actual output.
            if (abs(actual.video_duration_sec-expected_end)>1e-6 or actual.video_duration_sec>120
                    or not actual.has_audio or (actual.width,actual.height,actual.fps,actual.sample_aspect_ratio) !=
                    (info['width'],info['height'],info['fps'],info['sample_aspect_ratio'])):
                raise media.AssistantMediaError('ending candidate changed its verified output profile')
        evidence = {"candidate_id":candidate_id,"base_version_id":self.target,"plan":plan.model_dump(mode="json"),
            "sha256":media._hash(output,self.budget),"plan_sha256":plan_fingerprint(plan),"size_bytes":output.stat().st_size,
            "duration_sec":result.metadata.video_duration_sec,"has_audio":result.metadata.has_audio,
            "subtitle_status":"editable_burned" if track else subtitle_status,
            "mixed_audio_changed":any(op.kind in ("audio_splice","full_track_audio","extend_ending") for op in plan.operations),"review_required":True}
        self.candidates[candidate_id] = {"path":output,"evidence":evidence,"references":tuple(set(references))}
        return {"candidate":evidence}

    @_bounded_tool
    def persist_candidate(self, candidate_id):
        """Backend-only blob metadata; NEVER put storage_key into model context."""
        candidate = self._candidate(candidate_id)
        pinned = self.storage.pin(candidate["path"], cancelled=self)
        self.check()
        if pinned.sha256 != candidate["evidence"]["sha256"]:
            raise media.AssistantMediaError("retained candidate integrity failure")
        self.storage.verify(pinned,cancelled=self)
        self._candidate(candidate_id)
        return asdict(pinned)
