"""
pipeline_tail.py — process_tail_edit(edit_id) for the "Change ending" tool.

Trims a done Run's stitched final video to ``[0, cut_sec)`` and optionally
appends a stitch-library clip, producing a brand-new video delivered to
Google Drive. No AI generation involved — pure ffmpeg (media.cut_clip +
media.stitch). This lets an operator swap the outro of an already-delivered
video without re-running the AI pipeline (and without spending AI credits).

The video actually read is resolved by :func:`_resolve_source`, not always
``run.result_local_path`` — a tail edit can instead be cut from a captioned
derived creative (``source_type == "caption"``), so both directions of
docs/burned-captions.md §3.7 (chaining derived creatives) can compose. Either
way, ``edit.run_id`` stays the *parent Run* (provenance for the project id
and the output filename), never the video being read.

process_tail_edit(edit_id, *, gdrive=None)
    Transitions: queued -> processing -> delivering -> done  (or -> failed).
    A non-queued starting status is treated as an orphan resume (a prior
    worker attempt was interrupted) and reset to queued first, mirroring
    pipeline_merge.process_merge's orphan-resume handling.

This module intentionally mirrors pipeline_merge.py's shape (single ffmpeg
pass + Drive delivery, duplicated retry/error-truncation helpers in miniature)
rather than sharing code with it — the entities and validation differ enough
that a shared abstraction would be net-negative for a two-function pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from . import media as media_mod
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .models import LibraryClip, Run, RunCaption, RunTailEdit
from .pipeline_common import TAIL_EDIT_MIN_CUT_MARGIN_SEC as _MIN_CUT_MARGIN_SEC
from .state_machine import CaptionStatus, TailEditStatus, transition
from .storage import tail_edit_body_path, tail_edit_dir

log = logging.getLogger(__name__)

# Delivery (Google Drive upload) retry tuning — mirrors pipeline_merge's
# MERGE_DELIVER_ATTEMPTS / MERGE_DELIVER_BACKOFF_SEC so a transient Drive
# error self-heals the same way a merge's delivery does.
TAIL_EDIT_DELIVER_ATTEMPTS = int(os.getenv("TAIL_EDIT_DELIVER_ATTEMPTS", "3"))
TAIL_EDIT_DELIVER_BACKOFF_SEC = float(os.getenv("TAIL_EDIT_DELIVER_BACKOFF_SEC", "10"))

# Cap how much of an exception message is persisted to RunTailEdit.error_message.
_ERROR_MESSAGE_MAX_LEN = 2000


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


def _truncate_error(exc: Exception) -> str:
    """Render *exc* as a bounded-length string safe to store in error_message."""
    msg = str(exc)
    if len(msg) > _ERROR_MESSAGE_MAX_LEN:
        msg = msg[:_ERROR_MESSAGE_MAX_LEN] + "... (truncated)"
    return msg


# Chars kept as-is in a filesystem-safe name; any run of other characters
# collapses to a single underscore.
_UNSAFE_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_name_component(name: str) -> str:
    """Make *name* safe to embed in a filename: keep alnum/dash/underscore,
    collapse any other run of characters to a single underscore, trim
    leading/trailing underscores."""
    return _UNSAFE_NAME_CHARS_RE.sub("_", name).strip("_")


def _resolve_source(edit: RunTailEdit, run: Run, session) -> str:
    """Return the absolute path to the video this tail edit trims.

    ``edit.run_id`` (== ``run.id``) is always the parent Run — the
    provenance behind the output filename, not necessarily the video being
    read. ``edit.source_type`` says which file that is:
      "run"     (default) -> ``run.result_local_path``, byte-for-byte the v1
                behaviour.
      "caption" -> the referenced RunCaption's ``output_path`` (a captioned
                body; the new tail this job appends is NOT itself captioned
                — see docs/burned-captions.md §3.7).
    Mirrors overlay_pipeline._resolve_source. Raises ValueError — the same
    shape as the existing "run has no result video on disk" failure, so the
    caller's except block fails the job with a clear error_message rather
    than raising unhandled — when the referenced caption is missing, is not
    ``done``, or has no file on disk.
    """
    if edit.source_type == "caption":
        if not edit.source_caption_id:
            raise ValueError(
                "tail edit source_type='caption' but source_caption_id is unset"
            )
        caption = session.get(RunCaption, edit.source_caption_id)
        if caption is None:
            raise ValueError(f"source caption {edit.source_caption_id!r} not found")
        status_val = (
            caption.status.value if hasattr(caption.status, "value")
            else str(caption.status)
        )
        if status_val != CaptionStatus.done.value:
            raise ValueError(
                f"source caption {edit.source_caption_id!r} is not done "
                f"(status={status_val})"
            )
        path = caption.output_path
        if not path or not os.path.exists(path):
            raise ValueError("source caption has no output video on disk")
        return path

    # source_type == "run" (default) — byte-for-byte the v1 behaviour.
    path = run.result_local_path
    if not path or not os.path.exists(path):
        raise ValueError(f"run {edit.run_id!r} has no result video on disk")
    return path


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError as exc:
        log.warning("could not remove tail-edit intermediate %s: %s", path, exc)


def _output_filename(
    run_id: str, clip: Optional[LibraryClip], *, captioned: bool = False
) -> str:
    """Output/delivery filename for a tail edit: ``{run_id}_{clip_name}.mp4``,
    or ``{run_id}_trim.mp4`` when no library clip is appended (trim-only).

    When this tail edit's source is a caption job (``captioned=True``), an
    extra ``captioned_`` segment is inserted so a chained job's filename
    cannot collide with an unchained one's: ``{run_id}_captioned_{clip_name}
    .mp4`` — or ``{run_id}_captioned_trim.mp4`` for the trim-only variant,
    which keeps the same ``_trim`` suffix convention as the unchained case.
    """
    if clip is not None:
        clip_part = _sanitize_name_component(clip.name) or "clip"
    else:
        clip_part = "trim"
    if captioned:
        return f"{run_id}_captioned_{clip_part}.mp4"
    return f"{run_id}_{clip_part}.mp4"


def _deliver_with_retry(
    gdrive: GDriveClient, path: str, folder_id: str, edit_id: str, filename: str
) -> dict:
    """Upload the tail-edited video to Drive, retrying the whole upload on failure.

    Raises the last exception if every attempt fails (the caller then marks the
    tail edit failed — but the output file stays on disk, so a manual retry
    re-delivers without re-encoding).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, TAIL_EDIT_DELIVER_ATTEMPTS + 1):
        try:
            return gdrive.upload_file(path, folder_id, filename)
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Drive upload attempt %d/%d failed for tail edit %s: %s",
                attempt, TAIL_EDIT_DELIVER_ATTEMPTS, edit_id, exc,
            )
            if attempt < TAIL_EDIT_DELIVER_ATTEMPTS:
                time.sleep(TAIL_EDIT_DELIVER_BACKOFF_SEC * attempt)
    raise last_exc  # type: ignore[misc]


def process_tail_edit(edit_id: str, *, gdrive: Optional[GDriveClient] = None) -> None:
    """
    Trim a done Run's final video and optionally append a library clip.

    Transitions
    -----------
    queued -> processing -> delivering -> done
    (on error: -> failed, error_message set, exception re-raised)

    Validates before encoding (fails the tail edit, does not raise unhandled):
    - the resolved source (the Run's result, or — when source_type ==
      "caption" — the referenced RunCaption's output, per _resolve_source)
      has a video on disk;
    - 0 < cut_sec < the resolved source's probed duration (with a small
      margin) — probed from the SAME file that gets cut, never the Run's
      result when a caption source is chained in, since that source can be a
      different (shorter or longer) duration;
    - if library_clip_id is set, the LibraryClip row exists and its file is
      on disk.
    """
    t_start = time.monotonic()
    log.info("process_tail_edit start: edit_id=%s", edit_id)

    with get_session() as session:
        edit: RunTailEdit = session.get(RunTailEdit, edit_id)
        if edit is None:
            raise ValueError(f"RunTailEdit not found: {edit_id}")

        try:
            # Orphan-resume: any non-queued starting status means a prior
            # attempt was interrupted mid-flight — reset to queued before
            # advancing (mirrors pipeline_merge.process_merge).
            if edit.status != TailEditStatus.queued:
                log.info(
                    "process_tail_edit: edit %s is in status=%r (not queued) — "
                    "resetting to queued for orphan resume",
                    edit_id, edit.status,
                )
                try:
                    transition(edit, TailEditStatus.queued)
                except Exception:
                    # delivering lacks a direct ->queued edge; go via failed.
                    transition(edit, TailEditStatus.failed)
                    transition(edit, TailEditStatus.queued)
                session.commit()

            transition(edit, TailEditStatus.processing)
            session.commit()

            run: Run = session.get(Run, edit.run_id)
            if run is None:
                raise ValueError(f"Run not found: {edit.run_id}")

            final_path = _resolve_source(edit, run, session)

            # Probe the RESOLVED source, never the Run's result — a chained
            # caption source can be a different duration, and both the
            # cut_sec bounds check below and the cut/stitch calls further
            # down must all agree on the file that is actually being cut.
            info = media_mod.probe(final_path)
            if not (_MIN_CUT_MARGIN_SEC < edit.cut_sec < info.duration_sec):
                raise ValueError(
                    f"cut_sec ({edit.cut_sec}) must be within "
                    f"({_MIN_CUT_MARGIN_SEC}, {info.duration_sec}) — the "
                    "resolved source video is that long"
                )

            clip: Optional[LibraryClip] = None
            if edit.library_clip_id:
                clip = session.get(LibraryClip, edit.library_clip_id)
                if clip is None:
                    raise ValueError(
                        f"library clip {edit.library_clip_id!r} not found"
                    )
                if not clip.local_path or not os.path.exists(clip.local_path):
                    raise ValueError(
                        f"library clip {edit.library_clip_id!r} has no file on disk"
                    )

            # ---------------------------------------------------------------
            # Cut + stitch
            # ---------------------------------------------------------------
            t_encode_start = time.monotonic()
            out_dir = tail_edit_dir(edit.id, run.project_id)

            target_w, target_h, fps = media_mod.get_default_target(info)

            body_path = tail_edit_body_path(edit.id, run.project_id)
            media_mod.cut_clip(
                final_path, 0, edit.cut_sec, body_path, fast_intermediate=True
            )

            clips = [body_path] if clip is None else [body_path, clip.local_path]

            out_filename = _output_filename(
                run.id, clip, captioned=(edit.source_type == "caption")
            )
            output_path = os.path.join(out_dir, out_filename)
            media_mod.stitch(
                clips,
                audio_source=final_path,
                dst=output_path,
                width=target_w,
                height=target_h,
                fps=fps,
                audio_mode="seedance",
            )
            # body.mp4 is a pure intermediate, re-cut from the run's final on
            # every (re)run of the edit — drop it now rather than keep a
            # ~250 MB copy of the final per edit on disk.
            _remove_quietly(body_path)
            # Normally a no-op (stitch already caps the output); this catches a
            # body cut from a pre-cap run result that the cap could not reach.
            # Never fail the edit over it — an oversized deliverable still beats
            # no deliverable.
            try:
                media_mod.enforce_size_limit(output_path)
            except Exception as exc:
                log.warning(
                    "edit_id=%s: could not shrink %s to the delivery size cap: %s",
                    edit_id, output_path, exc,
                )
            edit.output_path = output_path
            session.commit()
            log.info(
                "edit_id=%s encode_phase=%.1fs -> %s",
                edit_id, time.monotonic() - t_encode_start, output_path,
            )

            # ---------------------------------------------------------------
            # Deliver
            # ---------------------------------------------------------------
            t_deliver_start = time.monotonic()
            transition(edit, TailEditStatus.delivering)
            session.commit()
            folder_id = run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id:
                if gdrive is None:
                    gdrive = _default_gdrive()
                result = _deliver_with_retry(
                    gdrive, output_path, folder_id, edit_id, out_filename
                )
                edit.gdrive_file_id = result.get("id")
                log.info(
                    "edit_id=%s deliver_phase=%.1fs gdrive_file=%s",
                    edit_id, time.monotonic() - t_deliver_start, edit.gdrive_file_id,
                )
                session.flush()

            # Clear any stale failure message from a prior attempt.
            edit.error_message = None
            transition(edit, TailEditStatus.done)
            session.commit()
            log.info(
                "process_tail_edit done: edit_id=%s total=%.1fs",
                edit_id, time.monotonic() - t_start,
            )

        except Exception as exc:
            log.exception("process_tail_edit failed: edit_id=%s", edit_id)
            edit = session.get(RunTailEdit, edit_id)
            if edit is not None:
                if edit.status not in (TailEditStatus.done, TailEditStatus.failed):
                    try:
                        transition(edit, TailEditStatus.failed)
                    except Exception:
                        edit.status = TailEditStatus.failed
                edit.error_message = _truncate_error(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise
