"""Trusted local-edit worker; no model, provider, Drive or legacy Run mutation.

The queue adapter supplies a session factory and server policy. Every attempt
gets an isolated directory; reclaimed phases rerender locally, with publication
guarded by the current SQL lease. Orphan files never become selected results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import tempfile
import time

from . import assistant_audio as audio, assistant_media as media, assistant_subtitles as subtitles
from .assistant_access import AuthorizedArtifactCatalog
from .assistant_contracts import EditPlan, validate_local_plan_capability
from .assistant_jobs import (AssistantJob, LostLease, claim_job, renew_lease, advance_phase,
                             publish_job, fail_job, finalize_cancel)
from .assistant_service import AssistantAdmissionService
from .assistant_store import (ArtifactInput, get_version_artifact, verify_artifact,
                              append_message)
from .assistant_models import AssistantArtifact, AssistantVersion


@dataclass(frozen=True)
class ExecutionResult:
    job_id: str
    status: str
    version_id: str | None = None
    explanation: str | None = None


def _now():
    return datetime.now(timezone.utc)


class _Guard:
    def __init__(self, factory, lease, lease_sec, timeout_sec):
        self.factory, self.lease, self.lease_sec = factory, lease, lease_sec
        self.checked, self.renewed = 0.0, time.monotonic()
        self.budget = media._budget(timeout_sec,None)

    def __call__(self):
        self.budget.check()
        now = time.monotonic()
        if now-self.checked < 0.25:
            return False
        self.checked = now
        with self.factory() as session:
            job = session.get(AssistantJob, self.lease.job_id, populate_existing=True)
            if job is None or job.cancel_requested_at is not None:
                return True
            if (job.lease_token, job.lease_generation, job.status) != (self.lease.token, self.lease.generation, "running"):
                raise LostLease("media execution lost its lease")
            expires=job.lease_expires_at
            if expires is None or (expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc))<=_now():
                raise LostLease("media execution lease expired")
            if now-self.renewed >= self.lease_sec/4:
                self.lease = renew_lease(session, self.lease, now=_now(), lease_sec=self.lease_sec)
                session.commit()
                self.renewed = now
        return False

    def check(self):
        if self():
            raise media.AssistantMediaCancelled("edit cancelled")

    def phase(self, phase):
        self.check()
        with self.factory() as session:
            self.lease = advance_phase(session, self.lease, phase=phase, now=_now(), lease_sec=self.lease_sec)
            session.commit()


def _render(operation, src, donors, dst, guard, timeout):
    args = {"timeout_sec": timeout, "cancelled": guard}
    kind = operation.kind
    if kind == "trim":
        return media.trim_video(src, dst, start_sec=operation.start_sec, end_sec=operation.end_sec, **args)
    if kind == "cut":
        return media.cut_section(src, dst, start_sec=operation.start_sec, end_sec=operation.end_sec, **args)
    if kind == "image_fade":
        return media.image_fades(src,dst,fades=((operation.start_sec,operation.end_sec,operation.direction),),**args)
    if kind == "replace_footage":
        return media.replace_footage(src, donors[operation.source.asset_id], dst,
                                    start_sec=operation.start_sec, end_sec=operation.end_sec,
                                    donor_start_sec=operation.donor_start_sec, **args)
    if kind == "audio_splice":
        return audio.splice_audio(src, donors[operation.source.asset_id], dst,
                                  start_sec=operation.start_sec, end_sec=operation.end_sec,
                                  donor_start_sec=operation.donor_start_sec,
                                  donor_duration_sec=operation.donor_duration_sec,
                                  crossfade_sec=operation.crossfade_sec, gain_db=operation.gain_db, **args)
    if kind == "full_track_audio":
        return audio.replace_full_track(src, donors[operation.source.asset_id], dst,
                                        donor_start_sec=operation.donor_start_sec, gain_db=operation.gain_db, **args)
    if kind == 'extend_ending':
        from .assistant_ending import extend_ending
        return extend_ending(src,donors[operation.source.asset_id],dst,start_sec=operation.start_sec,
            end_sec=operation.end_sec,donor_start_sec=operation.donor_start_sec,visual_kind=operation.visual.kind,
            freeze_timestamp_sec=getattr(operation.visual,'timestamp_sec',None),fade_out_sec=operation.fade_out_sec,**args)
    raise media.AssistantMediaError("unsupported trusted edit")


def _render_plan(plan, src, donors, dst, guard, timeout):
    """Stage all selected edits; coordinates stay on the unchanged base timeline."""
    validate_local_plan_capability(plan)
    current, results, fits = src, [], []
    operations = [op for op in plan.operations if op.kind not in ("image_fade", "subtitle_edit")]
    fades = tuple((op.start_sec,op.end_sec,op.direction) for op in plan.operations if op.kind == "image_fade")
    for index, operation in enumerate(operations):
        guard.check()
        target = dst.parent / f"{dst.stem}-step-{index}.mp4"
        result = _render(operation,current,donors,target,guard,timeout)
        results.append(result)
        if getattr(result,"audio_fit",None) is not None:
            fits.append(result.audio_fit)
        current = result.output_path
    if fades:
        result = media.image_fades(current,dst,fades=fades,timeout_sec=timeout,cancelled=guard)
        results.append(result)
    elif results:
        result = results[-1]
    else:
        return None, []
    # Persist original provenance, not temporary intermediate source identities.
    originals = {str(Path(src)), *(str(Path(path)) for path in donors.values())}
    sources = {item.path:item for rendered in results for item in rendered.sources if item.path in originals}
    result = replace(result, sources=tuple(sources.values()),
        operation=plan.operations[0].kind if len(plan.operations)==1 else "composite",
        audio_policy="replace_mixed_audio" if any(op.kind in ("audio_splice","full_track_audio") for op in operations) else result.audio_policy)
    return result, fits


def _ending_prefix_track(track, result):
    """Persist an ASS-safe prefix bound, including through later child renders."""
    clock = Fraction(result.ending_timing.start_frame,1)/Fraction(result.metadata.fps)
    retained = subtitles.remap_track(track,result.base_timeline)
    # ASS centisecond timestamps round to nearest. A frame boundary such as
    # 62/30 rounds UP to2.07, leaking text onto the first replaced frame.
    # Store the conservative representable end rather than a render-only mask.
    ticks = (clock*100).numerator//(clock*100).denominator
    bound = ticks/100
    cues = []
    for cue in retained.cues:
        end = min(cue.end_sec,bound)
        if round(end*100) <= round(cue.start_sec*100):
            raise media.AssistantMediaError('ending boundary leaves subtitle fragment shorter than ASS resolution')
        cues.append(cue.model_copy(update={'end_sec':end}))
    return subtitles.SubtitleTrack(cues=tuple(cues),style=retained.style)


def render_local_candidate(plan, base_path, donors, work, guard, timeout_sec, *, track=None, clean_path=None,
                           exact_audio_timing: bool = False):
    """Render verified inputs without jobs, publication or version selection.

    Shared with bounded agent verification so editable subtitles and mixed-audio
    semantics stay identical to the accepted edit worker. Caller owns authority,
    source integrity, destination isolation and the total resource guard.
    """
    if type(exact_audio_timing) is not bool:
        raise media.AssistantMediaError('clean audio timing profile must be a boolean')
    operation = plan.operations[0]
    picture_ops = [op for op in plan.operations if op.kind not in ("audio_splice", "full_track_audio", "subtitle_edit")]
    subtitle_ops = [op for op in plan.operations if op.kind == "subtitle_edit"]
    args = {"timeout_sec": timeout_sec, "cancelled": guard}
    clean_input = None
    audio_fits = []
    if track is not None and not picture_ops and not subtitle_ops:
        # Audio-only edits retain rendered picture packets exactly and
        # update the clean counterpart for future editable captions.
        result, audio_fits = _render_plan(plan,base_path,donors,work/"full.mp4",guard,timeout_sec)
        clean_result, _ = _render_plan(plan,clean_path,donors,work/"clean.mp4",guard,timeout_sec)
        clean_input = ArtifactInput(clean_result.output_path,metadata={"duration_sec":clean_result.metadata.video_duration_sec})
    elif track is not None:
        clean_result, audio_fits = _render_plan(plan,clean_path,donors,work/"clean.mp4",guard,timeout_sec)
        rendered_clean = clean_result.output_path if clean_result else clean_path
        if clean_result:
            track = (_ending_prefix_track(track,clean_result) if operation.kind == 'extend_ending'
                else subtitles.remap_track(track,clean_result.base_timeline))
        for subtitle_op in subtitle_ops:
            track = subtitles.edit_track(track,subtitle_op,duration_sec=media.probe_trusted_media(rendered_clean,**args).video_duration_sec)
        # A subtitle-only child reuses the same clean bytes and must retain
        # their MP4 movie clock. Other picture/audio edits create a new clean
        # render on their existing default clock, so do not inherit the flag.
        exact_for_burn = operation.kind == 'extend_ending' or (clean_result is None and exact_audio_timing)
        result = subtitles.burn_track(rendered_clean,work/"full.mp4",track=track,clean_source_verified=True,
            **args,**({'exact_audio_timing':True} if exact_for_burn else {}))
        if clean_result:
            result = replace(result,sources=clean_result.sources,audio_policy=clean_result.audio_policy,
                operation="composite" if len(plan.operations)>1 else operation.kind)
            if operation.kind == 'extend_ending':
                from .assistant_ending import validate_ending_output
                validate_ending_output(result.output_path,clean_result.ending_timing,**args)
                result = replace(clean_result,output_path=result.output_path,output_sha256=result.output_sha256,
                    metadata=result.metadata,subtitle_effect=result.subtitle_effect)
        clean_input = ArtifactInput(rendered_clean,metadata={"duration_sec":result.metadata.video_duration_sec,
            **({'exact_audio_timing':True} if exact_for_burn else {})})
    else:
        if subtitle_ops:
            raise media.AssistantMediaError("subtitle edit requires verified editable clean source")
        result, audio_fits = _render_plan(plan,base_path,donors,work/"full.mp4",guard,timeout_sec)
    return result, clean_input, track, audio_fits


def _metadata(result, *, subtitle_status, explanation):
    metadata = {**asdict(result.metadata), "duration_sec": result.metadata.video_duration_sec,
            "subtitle_status": subtitle_status, "operation": result.operation,
            "audio_policy": result.audio_policy, "explanation": explanation,
            "source_hashes": [p.sha256 for p in result.sources]}
    if getattr(result,'ending_timing',None) is not None:
        metadata['ending_timing'] = asdict(result.ending_timing)
    return metadata


def _audio_comparison(base, after, dst, *, start, end, info, timeout, guard):
    ratio=min(1,640/max(info.width,info.height))
    width,height=(max(2,round(size*ratio/2)*2) for size in (info.width,info.height))
    parts = []
    for i in (0,1):
        parts += [f"[{i}:v:0]trim=start={start:.9f}:end={end:.9f},setpts=PTS-STARTPTS,scale={width}:{height},setsar={info.sample_aspect_ratio.replace(':','/')}[v{i}]",
                  f"[{i}:a:0]atrim=start={start:.9f}:end={end:.9f},asetpts=PTS-STARTPTS[a{i}]"]
    parts.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
    return media._render("audio_comparison", (Path(base),Path(after)), media._destination(dst), replace(info,width=width,height=height),
                         ";".join(parts), 2*(end-start), (), "comparison_reencode",
                         "First half before; second half after, same timeline window.", media._budget(timeout,guard))


def execute_assistant_job(
    session_factory, job_id: str, *, owner: str, storage, policy,
    enabled: bool = False, lease_sec: int = 120, timeout_sec: float = 600,
) -> ExecutionResult:
    """Claim, resolve, render, inspect and atomically publish an accepted job.

    Feature enablement and policy are server configuration. Cancellation and
    lease heartbeats run during hashing/probes/FFmpeg through the media callback.
    No browser/model input can select a path, command, provider or destination.
    """
    if enabled is not True:
        raise ValueError("assistant executor is disabled")
    if type(lease_sec) is not int or not 30 <= lease_sec <= 3600:
        raise ValueError("executor lease must be within [30,3600] seconds")
    timeout_sec = media._number(timeout_sec, "timeout_sec")
    if not 0 < timeout_sec <= 600:
        raise ValueError("executor timeout must be within (0,600] seconds")
    service = AssistantAdmissionService(storage=storage, policy=policy, enabled=True)
    with session_factory() as session:
        lease = claim_job(session, job_id, owner=owner, now=_now(), lease_sec=lease_sec)
        session.commit()
    if lease is None:
        return ExecutionResult(job_id, "not_claimed")
    guard = _Guard(session_factory, lease, lease_sec, timeout_sec)
    try:
        guard.check()
        with session_factory() as session:
            service.revalidate_job(session, job_id, cancelled=guard, timeout_sec=max(.000001,guard.budget.deadline-time.monotonic()))
            job = session.get(AssistantJob, job_id)
            plan = EditPlan.model_validate(job.accepted_plan)
            version_id, actor = job.pending_version_id, job.actor
            base = get_version_artifact(session, plan.base_version_id)
            base_path = verify_artifact(base, storage=storage,cancelled=guard)
            subtitle_status = base.metadata_json.get("subtitle_status", "unknown")
            track = subtitles.get_subtitle_context(session, plan.base_version_id, storage=storage,cancelled=guard)
            clean_artifact = get_version_artifact(session,plan.base_version_id,role='clean') if track else None
            clean_path = verify_artifact(clean_artifact, storage=storage,cancelled=guard) if clean_artifact else None
            exact_audio_timing = clean_artifact.metadata_json.get('exact_audio_timing',False) if clean_artifact else False
            catalog = AuthorizedArtifactCatalog(session, policy=policy,
                        is_available=lambda a: bool(verify_artifact(a, storage=storage,cancelled=guard)))
            donors = {}
            for op in plan.operations:
                ref = getattr(op, "source", None)
                if ref and ref.asset_id not in donors:
                    catalog.resolve_editable_reference(target_version_id=plan.base_version_id, asset_id=ref.asset_id, version_id=ref.version_id)
                    donors[ref.asset_id] = verify_artifact(session.get(AssistantArtifact, ref.asset_id), storage=storage,cancelled=guard)
        guard.check()
        # A reclaimed attempt may start in any phase. Local rendering is pure;
        # recomputing in a fresh directory is safe and does not replay spending.
        if guard.lease.phase == "resolve":
            guard.phase("render")
        with tempfile.TemporaryDirectory(prefix="edit-", dir=storage.root / "staging") as temporary:
            work = Path(temporary)
            operation = plan.operations[0]
            audio_ops = [op for op in plan.operations if op.kind in ("audio_splice","full_track_audio","extend_ending")]
            ending_extension = operation.kind == 'extend_ending'
            picture_ops = [op for op in plan.operations if op.kind not in ("audio_splice","full_track_audio","subtitle_edit")]
            subtitle_ops = [op for op in plan.operations if op.kind == "subtitle_edit"]
            subtitle_intervals = []
            if track is not None:
                for subtitle_op in subtitle_ops:
                    changed = {change.cue_id for change in subtitle_op.changes}
                    subtitle_intervals.extend((cue.start_sec,cue.end_sec) for cue in track.cues
                        if subtitle_op.style is not None or cue.cue_id in changed)
                    subtitle_intervals.extend((change.start_sec,change.end_sec) for change in subtitle_op.changes
                        if change.action == "update")
            args = {"timeout_sec": timeout_sec, "cancelled": guard}
            result, clean_input, track, audio_fits = render_local_candidate(
                plan, base_path, donors, work, guard, timeout_sec, track=track, clean_path=clean_path,
                exact_audio_timing=exact_audio_timing)
            audio_fit = audio_fits[0] if len(audio_fits)==1 else None
            explanations = []
            for selected in plan.operations:
                kind = selected.kind
                if kind in ("audio_splice","full_track_audio"):
                    a,b = getattr(selected,"start_sec",0),getattr(selected,"end_sec",result.metadata.video_duration_sec)
                    explanations.append(f"Replaced mixed audio [{a:g}, {b:g}) seconds from the selected source starting at {selected.donor_start_sec:g}s, gain {selected.gain_db:g}dB, crossfade {getattr(selected,'crossfade_sec',0):g}s; music/background also change. Review pronunciation, level, seam and lip sync by listening; no transcript-only pronunciation claim.")
                elif kind == 'extend_ending':
                    explanations.append(f"Replaced the ending from {selected.start_sec:g}s and extended output to {selected.end_sec:g}s using {selected.visual.kind}; donor mixed audio starts at {selected.donor_start_sec:g}s and plays at 1x. Music/background also change. Retained editable captions are clipped to the prefix, with no new CTA captions; frozen burned text may remain. Review the seam, complete speech and frozen picture by watching and listening.")
                    explanations.append('Caption clipping can omit less than 0.01s at the boundary because ASS end times are rounded down to centiseconds; collapsed fragments are refused.')
                    explanations.append('Preview shows only the revised ending; the original has no picture or audio beyond its own end.')
                elif kind == "trim":
                    explanations.append(f"Kept [{selected.start_sec:g}, {selected.end_sec:g}) seconds; picture, audio and editable cue times move together.")
                elif kind == "cut":
                    explanations.append(f"Removed [{selected.start_sec:g}, {selected.end_sec:g}) seconds; later picture/audio and editable cues move earlier.")
                elif kind == "image_fade":
                    explanations.append(f"Faded picture {selected.direction} over [{selected.start_sec:g}, {selected.end_sec:g}) seconds; original picture resumes outside this interval, retaining base audio.")
                elif kind == "replace_footage":
                    explanations.append(f"Replaced only the requested picture interval [{selected.start_sec:g}, {selected.end_sec:g}); base audio timing remains. Review content, text and lip sync.")
                else:
                    explanations.append("Rendered the revised complete cue track from verified clean picture; audio timing remains.")
            for fit in audio_fits:
                explanations.append(f"Fitted {fit.donor_duration_sec:g}s of donor audio into {fit.target_duration_sec:g}s at tempo ratio {fit.speed_ratio:g}; padded {fit.pad_samples} and trimmed {fit.trim_samples} near-silent residual samples at {fit.sample_rate}Hz. Cadence changes require listening review.")
            if picture_ops or subtitle_ops:
                explanations.append("Picture was re-encoded at the source frame rate and geometry.")
            elif audio_ops:
                explanations.append("Picture packets were copied.")
            if audio_ops or operation.kind in ("trim","cut"):
                explanations.append("The soundtrack was re-encoded to AAC.")
            elif result.metadata.has_audio:
                explanations.append("Original audio packets were copied.")
            if track is None and subtitle_status == "unknown":
                explanations.append("Existing burned text is picture content; editable caption lineage is unavailable.")
            explanation = " ".join(explanations)
            if guard.lease.phase == "render":
                guard.phase("inspect")
            info = media.probe_trusted_media(result.output_path, **args)
            fps = float(Fraction(info.fps))
            intervals = []
            for selected in plan.operations:
                if selected.kind == "subtitle_edit":
                    intervals.extend(subtitle_intervals)
                elif selected.kind == "trim":
                    intervals.append((max(0,info.video_duration_sec-4),info.video_duration_sec))
                elif selected.kind == "cut":
                    intervals.append((min(selected.start_sec,info.video_duration_sec),min(selected.start_sec+1,info.video_duration_sec)))
                else:
                    intervals.append((getattr(selected,"start_sec",0),getattr(selected,"end_sec",info.video_duration_sec)))
            affected_start = min((a for a,b in intervals),default=0)
            affected_end = max((b for a,b in intervals),default=info.video_duration_sec)
            start = max(0,int((affected_start-1)*fps)/fps)
            requested_end = min(info.video_duration_sec,affected_end+1)
            end = round(min(requested_end,start+30)*fps)/fps
            if ending_extension:
                # The sole <=10s ending always fits this after-only excerpt,
                # including its whole extension and the old source boundary.
                end = info.video_duration_sec
                start = max(0,int((operation.start_sec-1)*fps)/fps)
            preview_truncated = end+1e-6 < requested_end
            if preview_truncated:
                explanation += f" Preview covers [{start:g}, {end:g}) seconds; use the full result to review remaining affected intervals through {affected_end:g}s."
            if audio_ops and not ending_extension:
                preview = _audio_comparison(base_path,result.output_path,work/"comparison.mp4",
                    start=start,end=end,info=info,timeout=timeout_sec,guard=guard)
            else:
                preview = media.trim_video(result.output_path,work/"preview.mp4",start_sec=start,end_sec=end,**args)
            full_metadata = _metadata(result, subtitle_status="editable_burned" if track else subtitle_status, explanation=explanation)
            full_metadata["operations"] = [op.model_dump(mode="json") for op in plan.operations]
            if audio_fits:
                full_metadata["audio_fits"] = [asdict(fit) for fit in audio_fits]
            if audio_fit is not None:
                full_metadata["audio_fit"]=asdict(audio_fit)
            preview_metadata = _metadata(preview, subtitle_status="preview", explanation=explanation)
            preview_metadata.update({"source_start_sec": start, "source_end_sec": end, "preview_mode": "affected_window"})
            preview_metadata.update({"affected_start_sec":affected_start,"affected_end_sec":affected_end,
                "affected_intervals":[list(span) for span in intervals],"preview_truncated":preview_truncated})
            if audio_ops and not ending_extension:
                preview_metadata.update({"preview_mode":"before_after", "sequence":["before","after"],
                    "switch_sec":end-start,"source_window_limit_sec":30,"preview_duration_limit_sec":60,
                    "preview_quality":"proxy_max_dimension_640","full_quality_preserved":True})
            caption_input = None
            if track is not None:
                path = work/"captions.json"
                path.write_text(track.model_dump_json(), encoding="utf-8")
                caption_input = ArtifactInput(path, "subtitle", {"schema_version": 1, "cue_count": len(track.cues)})
            if guard.lease.phase == "inspect":
                guard.phase("publish")
            guard.check()
            with session_factory() as session:
                # Recheck mutable authorization and every immutable source right
                # before the SQL publication fence, not just when queued.
                service.revalidate_job(session, job_id, cancelled=guard, timeout_sec=max(.000001,guard.budget.deadline-time.monotonic()))
                guard.budget.check()
                job = publish_job(session, guard.lease,
                    full=ArtifactInput(result.output_path, metadata=full_metadata),
                    preview=ArtifactInput(preview.output_path, metadata=preview_metadata),
                    clean=clean_input, captions=caption_input, storage=storage, now=_now())
                version = session.get(AssistantVersion, version_id)
                append_message(session, version.conversation_id, idempotency_key="execution:"+job_id,
                    actor=actor, role="assistant", content=explanation, base_version_id=plan.base_version_id,
                    context_snapshot={"job_id": job_id, "version_id": version_id, "status": "succeeded"})
                guard.budget.check()  # Slow artifact pinning must not outlive the total budget.
                session.commit()
            return ExecutionResult(job_id, "succeeded", version_id, explanation)
    except LostLease:
        return ExecutionResult(job_id, "lost_lease")
    except Exception as exc:
        # Catalog availability deliberately redacts callback exceptions. Retain
        # the deadline outcome even if it expired inside an availability hash.
        try:
            guard.budget.check()
        except media.AssistantMediaTimeout as timeout:
            exc = timeout
        with session_factory() as session:
            job = session.get(AssistantJob, job_id, populate_existing=True)
            try:
                if job and job.cancel_requested_at is not None:
                    finalize_cancel(session, job_id, now=_now(), lease=guard.lease)
                    status = "cancelled"
                else:
                    error_code = ("audio_fit_requires_review" if isinstance(exc, audio.AudioFitRejected)
                        else "local_edit_timeout" if isinstance(exc, media.AssistantMediaTimeout) else "local_edit_failed")
                    fail_job(session, guard.lease, error_code=error_code, now=_now())
                    status = "failed"
                session.commit()
            except LostLease:
                session.rollback()
                status = "lost_lease"
        return ExecutionResult(job_id, status)
