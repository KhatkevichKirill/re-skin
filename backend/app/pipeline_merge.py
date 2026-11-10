"""
pipeline_merge.py — process_merge(merge_id) for the PIP-insert merge feature.

Composites a completed child Run's result video back onto a target project's
source video at the placement/time-window recorded on an InsertMerge row (see
docs/pip-insert.md, "Merge" section).

process_merge(merge_id, *, gdrive=None)
    Transitions: queued -> compositing -> delivering -> done  (or -> failed).
    A non-queued starting status is treated as an orphan resume (a prior
    worker attempt was interrupted) and reset to queued first, mirroring
    pipeline_v2._build_submit_state's orphan-resume handling.

This module intentionally shares no code with pipeline_v2.py (kept untouched
per the PIP-insert milestone plan): the state machines, entities and Drive
delivery targets differ enough that a shared abstraction would be net-negative
for a single-ffmpeg-pass job. The delivery-retry and phase-timing conventions
are duplicated in miniature (see _deliver_with_retry below) for consistency
with process_run's observability.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from . import media as media_mod
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .models import InsertMerge, Run, VideoProject
from .state_machine import MergeStatus, RunStatus, transition
from .storage import merge_dir

log = logging.getLogger(__name__)

# Delivery (Google Drive upload) retry tuning — mirrors pipeline_v2's
# RUN_DELIVER_ATTEMPTS / RUN_DELIVER_BACKOFF_SEC so a transient Drive error
# self-heals the same way a run's delivery does.
MERGE_DELIVER_ATTEMPTS = int(os.getenv("MERGE_DELIVER_ATTEMPTS", "3"))
MERGE_DELIVER_BACKOFF_SEC = float(os.getenv("MERGE_DELIVER_BACKOFF_SEC", "10"))

# Cap how much of an exception message is persisted to InsertMerge.error_message.
_ERROR_MESSAGE_MAX_LEN = 2000


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


def _truncate_error(exc: Exception) -> str:
    """Render *exc* as a bounded-length string safe to store in error_message."""
    msg = str(exc)
    if len(msg) > _ERROR_MESSAGE_MAX_LEN:
        msg = msg[:_ERROR_MESSAGE_MAX_LEN] + "... (truncated)"
    return msg


def _deliver_with_retry(
    gdrive: GDriveClient, path: str, folder_id: str, merge_id: str
) -> dict:
    """Upload the merged video to Drive, retrying the whole upload on failure.

    Raises the last exception if every attempt fails (the caller then marks the
    merge failed — but merged.mp4 stays on disk, so a manual retry re-delivers
    without re-compositing).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MERGE_DELIVER_ATTEMPTS + 1):
        try:
            return gdrive.upload_file(path, folder_id, f"reskin_merge_{merge_id}.mp4")
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Drive upload attempt %d/%d failed for merge %s: %s",
                attempt, MERGE_DELIVER_ATTEMPTS, merge_id, exc,
            )
            if attempt < MERGE_DELIVER_ATTEMPTS:
                time.sleep(MERGE_DELIVER_BACKOFF_SEC * attempt)
    raise last_exc  # type: ignore[misc]


def process_merge(merge_id: str, *, gdrive: Optional[GDriveClient] = None) -> None:
    """
    Composite a child Run's result back onto the target project's source video.

    Transitions
    -----------
    queued -> compositing -> delivering -> done
    (on error: -> failed, error_message set, exception re-raised)

    Validates before compositing (fails the merge, does not raise unhandled):
    - the child Run is status=done with a result video on disk;
    - the target VideoProject has a source video on disk and probed
      width/height/fps.
    """
    t_start = time.monotonic()
    log.info("process_merge start: merge_id=%s", merge_id)

    with get_session() as session:
        merge: InsertMerge = session.get(InsertMerge, merge_id)
        if merge is None:
            raise ValueError(f"InsertMerge not found: {merge_id}")

        try:
            # Orphan-resume: any non-queued starting status means a prior
            # attempt was interrupted mid-flight — reset to queued before
            # advancing (mirrors pipeline_v2._build_submit_state).
            if merge.status != MergeStatus.queued:
                log.info(
                    "process_merge: merge %s is in status=%r (not queued) — "
                    "resetting to queued for orphan resume",
                    merge_id, merge.status,
                )
                try:
                    transition(merge, MergeStatus.queued)
                except Exception:
                    # delivering lacks a direct ->queued edge; go via failed.
                    transition(merge, MergeStatus.failed)
                    transition(merge, MergeStatus.queued)
                session.commit()

            transition(merge, MergeStatus.compositing)
            session.commit()

            target: VideoProject = session.get(VideoProject, merge.target_project_id)
            if target is None:
                raise ValueError(
                    f"VideoProject not found: {merge.target_project_id}"
                )
            child_run: Run = session.get(Run, merge.child_run_id)
            if child_run is None:
                raise ValueError(f"Run not found: {merge.child_run_id}")

            if child_run.status != RunStatus.done:
                raise ValueError(
                    f"child run {merge.child_run_id!r} is not done "
                    f"(status={child_run.status!r})"
                )
            insert_path = child_run.result_local_path
            if not insert_path or not os.path.exists(insert_path):
                raise ValueError(
                    f"child run {merge.child_run_id!r} has no result video on disk"
                )

            source = target.source_local_path
            if not source or not os.path.exists(source):
                raise ValueError(
                    f"target project {merge.target_project_id!r} has no source "
                    "video on disk"
                )
            if target.width is None or target.height is None or target.fps is None:
                raise ValueError(
                    f"target project {merge.target_project_id!r} has not been "
                    "probed (no width/height/fps)"
                )

            # ---------------------------------------------------------------
            # Composite
            # ---------------------------------------------------------------
            t_composite_start = time.monotonic()
            out_dir = merge_dir(merge_id, merge.target_project_id)

            info = media_mod.probe(source)
            _target_w, _target_h, fps = media_mod.get_default_target(info)

            mask_png: Optional[str] = None
            if merge.corner_radius_px > 0:
                mask_png = os.path.join(out_dir, "mask.png")
                media_mod.make_rounded_mask(
                    merge.width, merge.height, merge.corner_radius_px, mask_png
                )

            merged_path = os.path.join(out_dir, "merged.mp4")
            media_mod.overlay_insert(
                base=source,
                insert=insert_path,
                dst=merged_path,
                x=merge.x,
                y=merge.y,
                start_sec=merge.start_sec,
                end_sec=merge.end_sec,
                width=merge.width,
                height=merge.height,
                fps=fps,
                mask_png=mask_png,
            )
            # Normally a no-op (overlay_insert already caps the composite);
            # this catches a base video the cap could not reach. Never fail the
            # merge over it — an oversized deliverable still beats none.
            try:
                media_mod.enforce_size_limit(merged_path)
            except Exception as exc:
                log.warning(
                    "merge_id=%s: could not shrink %s to the delivery size "
                    "cap: %s", merge_id, merged_path, exc,
                )
            merge.output_path = merged_path
            session.commit()
            log.info(
                "merge_id=%s composite_phase=%.1fs -> %s",
                merge_id, time.monotonic() - t_composite_start, merged_path,
            )

            # ---------------------------------------------------------------
            # Deliver
            # ---------------------------------------------------------------
            t_deliver_start = time.monotonic()
            transition(merge, MergeStatus.delivering)
            session.commit()
            folder_id = settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id:
                if gdrive is None:
                    gdrive = _default_gdrive()
                result = _deliver_with_retry(gdrive, merged_path, folder_id, merge_id)
                file_id = result.get("id")
                merge.gdrive_url = result.get("webViewLink") or (
                    f"https://drive.google.com/file/d/{file_id}/view"
                    if file_id else None
                )
                log.info(
                    "merge_id=%s deliver_phase=%.1fs gdrive_file=%s",
                    merge_id, time.monotonic() - t_deliver_start, file_id,
                )
                session.flush()

            # Clear any stale failure message from a prior attempt.
            merge.error_message = None
            transition(merge, MergeStatus.done)
            session.commit()
            log.info(
                "process_merge done: merge_id=%s total=%.1fs",
                merge_id, time.monotonic() - t_start,
            )

        except Exception as exc:
            log.exception("process_merge failed: merge_id=%s", merge_id)
            merge = session.get(InsertMerge, merge_id)
            if merge is not None:
                if merge.status not in (MergeStatus.done, MergeStatus.failed):
                    try:
                        transition(merge, MergeStatus.failed)
                    except Exception:
                        merge.status = MergeStatus.failed
                merge.error_message = _truncate_error(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise
