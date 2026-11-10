"""Import a currently delivered legacy result without trusting cached duration.

Legacy files and subtitle options are mutable. Copy bytes under the Run lock,
probe that copy and freeze it; do not infer editable captions from mutable rows.
"""
from pathlib import Path
import tempfile
from sqlalchemy import select

from .assistant_access import ProjectAllowlist
from .assistant_jobs import _atomic
from .assistant_media import probe_trusted_media, EditingPreparationRequired
from .assistant_models import AssistantConversation, AssistantVersion
from .assistant_store import (ArtifactInput, AssistantStoreError, get_or_create_conversation,
                              import_run_version, _lock)


def import_legacy_run(session, run, *, actor, storage, policy: ProjectAllowlist,
                      idempotency_key="legacy-import-v1", timeout_sec=30, cancelled=None):
    import time
    from .assistant_media import _budget
    if type(timeout_sec) not in (int, float) or not 0 < timeout_sec <= 30:
        raise AssistantStoreError("legacy import probe deadline must be within (0,30] seconds")
    budget = _budget(timeout_sec, cancelled)
    budget.check()
    policy.require_project(run.project_id)
    with _atomic(session):
        conversation = get_or_create_conversation(session, run_id=run.id, project_id=run.project_id)
        _lock(session, conversation.id)
        session.refresh(run, with_for_update=True)
        policy.require_project(run.project_id)
        existing = session.scalar(select(AssistantVersion).join(
            AssistantConversation, AssistantConversation.id == AssistantVersion.conversation_id,
        ).where(AssistantConversation.run_id == run.id,
                AssistantVersion.idempotency_key == idempotency_key))
        if existing is not None:
            # The store verifies this is the same kind of import. Reopening
            # does not follow a newly replaced or deleted legacy file pointer.
            return import_run_version(session, run, actor=actor, storage=storage,
                                      idempotency_key=idempotency_key)
        if getattr(run.status, "value", run.status) != "done" or not run.result_local_path:
            raise AssistantStoreError("only a completed local result can be imported")
        budget.check()
        pinned = storage.pin(run.result_local_path)
        budget.check()
        path = storage.verify(pinned)
        budget.check()
        preparation = None
        try:
            info = probe_trusted_media(path, timeout_sec=max(0.1, budget.deadline - time.monotonic()), cancelled=cancelled)
        except EditingPreparationRequired as error:
            info, preparation = error.info, error.preparation
        budget.check()
        full = ArtifactInput(path, "video", {
            "duration_sec": info.video_duration_sec, "has_audio": info.has_audio, "fps": info.fps,
            "preview_mode": "full_as_preview", "subtitle_status": "unknown",
            **({"needs_editing_preparation": True, "editing_preparation": preparation} if preparation else {}),
        })
        return import_run_version(session, run, actor=actor, storage=storage,
                                  idempotency_key=idempotency_key, full=full)


def import_legacy_caption(session, run, caption_id, *, actor, storage, policy,
                          timeout_sec=240):
    """Accept old caption lineage only if reproducing its decoded result exactly.

    Old rows have neither immutable source hashes nor immutable rendered cue
    JSON. Current options/transcripts are therefore never evidence. A changed
    source, missing ASS, unsupported style, or different historical renderer
    refuses this import rather than guessing away old burned subtitles.
    """
    from . import assistant_media as safe_media, media as legacy_media
    from .assistant_store import import_derived_media_version
    from .assistant_subtitles import parse_legacy_ass, validate_duration
    from .models import RunCaption, RunTailEdit

    policy.require_project(run.project_id)
    budget = safe_media._budget(timeout_sec, None)
    with _atomic(session):
        import_legacy_run(session, run, actor=actor, storage=storage, policy=policy)
        caption = session.scalar(select(RunCaption).where(
            RunCaption.id == caption_id, RunCaption.run_id == run.id).with_for_update())
        if caption is None or getattr(caption.status, "value", caption.status) != "done" or not caption.output_path:
            raise AssistantStoreError("completed caption output unavailable")
        if caption.source_type == "run":
            source_path = run.result_local_path
        elif caption.source_type == "tail_edit":
            tail = session.scalar(select(RunTailEdit).where(
                RunTailEdit.id == caption.source_tail_edit_id, RunTailEdit.run_id == run.id).with_for_update())
            if tail is None or getattr(tail.status, "value", tail.status) != "done":
                raise AssistantStoreError("caption clean source unavailable")
            source_path = tail.output_path
        else:
            raise AssistantStoreError("unsupported caption source lineage")
        if not source_path:
            raise AssistantStoreError("caption clean source unavailable")
        output_path = storage._source(caption.output_path)
        ass_path = storage._source(output_path.parent / "captions.ass")
        if ass_path.stat().st_size > 4*1024*1024:
            raise AssistantStoreError("caption track exceeds import limit")
        pinned_full, pinned_clean, pinned_ass = (storage.pin(path) for path in (output_path, source_path, ass_path))
        full_path, clean_path, original_ass = (storage.verify(item) for item in (pinned_full, pinned_clean, pinned_ass))
        full_info, clean_info = (safe_media._probe(path, budget) for path in (full_path, clean_path))
        if ((full_info.width, full_info.height, full_info.fps, full_info.video_duration_sec, full_info.has_audio)
                != (clean_info.width, clean_info.height, clean_info.fps, clean_info.video_duration_sec, clean_info.has_audio)):
            raise AssistantStoreError("caption output and candidate clean source do not align")
        track = parse_legacy_ass(original_ass.read_text(encoding="utf-8"), width=full_info.width, height=full_info.height)
        validate_duration(track, full_info.video_duration_sec)
        with tempfile.TemporaryDirectory(prefix="caption-proof-", dir=storage.root/"staging") as temporary:
            work = Path(temporary)
            ass_copy = work/"captions.ass"
            ass_copy.write_bytes(original_ass.read_bytes())
            proof = work/"proof.mp4"
            argv = ["ffmpeg", "-v", "error", "-n", "-threads", legacy_media._FFMPEG_THREADS,
                    "-i", str(clean_path), "-vf", "subtitles="+legacy_media._escape_filter_path(str(ass_copy))]
            argv += legacy_media.delivery_video_opts(clean_info.video_duration_sec,
                audio_bitrate_bps=legacy_media._STITCH_AUDIO_BPS if clean_info.has_audio else 0)
            argv += ["-c:a", "copy"] if clean_info.has_audio else ["-an"]
            safe_media._run(argv+["-movflags", "+faststart", str(proof)], budget,
                inputs=(clean_path, ass_copy),outputs=(proof,))
            def digest(path, stream):
                args = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:"+stream]
                if stream == "a:0": args += ["-c:a", "copy"]
                return safe_media._run(args+["-f", "hash", "-hash", "sha256", "-"], budget,inputs=(path,)).strip()
            for stream in (["v:0", "a:0"] if full_info.has_audio else ["v:0"]):
                if digest(proof, stream) != digest(full_path, stream):
                    raise AssistantStoreError("saved captions cannot be reproduced from the available source; matching clean source required")
            track_path = work/"captions.json"
            track_path.write_text(track.model_dump_json(), encoding="utf-8")
            budget.check()
            metadata = {"duration_sec": full_info.video_duration_sec, "has_audio": full_info.has_audio, "fps": full_info.fps}
            return import_derived_media_version(session, run, actor=actor, storage=storage,
                idempotency_key="caption-import:"+caption.id,
                provenance={"caption_id": caption.id, "source_type": caption.source_type,
                    "source_tail_edit_id": caption.source_tail_edit_id, "proof": "decoded_video_and_audio_match",
                    "original_ass_sha256": pinned_ass.sha256},
                full=ArtifactInput(full_path, metadata={**metadata, "subtitle_status":"editable_burned", "preview_mode":"full_as_preview"}),
                clean=ArtifactInput(clean_path, metadata={**metadata,"subtitle_status":"clean_for_imported_captions"}),
                captions=ArtifactInput(track_path, "subtitle", {"schema_version":1,"cue_count":len(track.cues)}))
