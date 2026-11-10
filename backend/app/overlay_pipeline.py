"""
overlay_pipeline.py — Worker entry point for the standalone overlay-burn feature.

process_overlay_job(job_id) drives one OverlayJob through:
    queued -> processing -> [delivering] -> done   (or -> failed)

It resolves the source video (a completed Run's result, one of that run's
derived creatives — a tail edit's or a caption job's output — or a
directly-uploaded file), probes it, builds the OverlaySpec list from the stored
JSON, burns the PNGs via :func:`overlay.burn_overlays`, and optionally delivers
to Drive.

Shares NO state with the run pipeline — it never imports or mutates Run rows
beyond reading a referenced run's ``result_local_path``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .media import MediaError, enforce_size_limit, probe
from .models import OverlayJob, Run, RunCaption, RunTailEdit
from .overlay import OverlaySpec, burn_overlays
from .state_machine import (
    CaptionStatus,
    OverlayStatus,
    RunStatus,
    TailEditStatus,
    transition,
)
from .storage import overlay_job_dir, overlay_output_dir, resolve_overlay_asset

log = logging.getLogger(__name__)


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


def _resolve_source(job: OverlayJob, session) -> str:
    """Return the absolute path to the source video, validating availability."""
    if job.source_type == "run":
        if not job.source_run_id:
            raise MediaError("overlay job source_type='run' but source_run_id is unset")
        run = session.get(Run, job.source_run_id)
        if run is None:
            raise MediaError(f"source run {job.source_run_id!r} not found")
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        if status_val != RunStatus.done.value:
            raise MediaError(
                f"source run {job.source_run_id!r} is not done (status={status_val})"
            )
        path = run.result_local_path
        if not path or not os.path.exists(path):
            raise MediaError("source run has no result video on disk")
        return path

    if job.source_type == "tail_edit":
        if not job.source_tail_edit_id:
            raise MediaError(
                "overlay job source_type='tail_edit' but source_tail_edit_id is unset"
            )
        edit = session.get(RunTailEdit, job.source_tail_edit_id)
        if edit is None:
            raise MediaError(f"source tail edit {job.source_tail_edit_id!r} not found")
        status_val = edit.status.value if hasattr(edit.status, "value") else str(edit.status)
        if status_val != TailEditStatus.done.value:
            raise MediaError(
                f"source tail edit {job.source_tail_edit_id!r} is not done (status={status_val})"
            )
        path = edit.output_path
        if not path or not os.path.exists(path):
            raise MediaError("source tail edit has no output video on disk")
        return path

    if job.source_type == "caption":
        if not job.source_caption_id:
            raise MediaError(
                "overlay job source_type='caption' but source_caption_id is unset"
            )
        caption = session.get(RunCaption, job.source_caption_id)
        if caption is None:
            raise MediaError(f"source caption {job.source_caption_id!r} not found")
        status_val = (
            caption.status.value if hasattr(caption.status, "value") else str(caption.status)
        )
        if status_val != CaptionStatus.done.value:
            raise MediaError(
                f"source caption {job.source_caption_id!r} is not done (status={status_val})"
            )
        path = caption.output_path
        if not path or not os.path.exists(path):
            raise MediaError("source caption has no output video on disk")
        return path

    # source_type == "upload"
    path = job.source_local_path
    if not path or not os.path.exists(path):
        raise MediaError("uploaded source video not found on disk")
    return path


def _build_specs(job: OverlayJob, src_dur: float) -> List[OverlaySpec]:
    """Build validated OverlaySpec objects from the stored overlays JSON."""
    specs: List[OverlaySpec] = []
    for i, ov in enumerate(job.overlays or []):
        rel = ov.get("asset_path")
        if not rel:
            raise MediaError(f"overlay[{i}] has no asset_path")
        # AICP-160: contained in the job dir (no absolute path, no `..`, no
        # symlink out) -- the path goes straight into ffmpeg `-i`.
        abs_path = resolve_overlay_asset(job.id, rel)
        if abs_path is None:
            raise MediaError(f"overlay[{i}] asset not found: {rel}")

        nx, ny = float(ov["nx"]), float(ov["ny"])
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            raise MediaError(f"overlay[{i}] nx/ny out of [0,1] range")

        start = float(ov["start_sec"])
        end = float(ov["end_sec"])
        if end <= start:
            raise MediaError(f"overlay[{i}] end_sec must be > start_sec")
        # Clamp the visible window to the source duration (burn_overlays also
        # clamps, but persisting the clamp keeps the API response honest).
        end = min(end, src_dur)

        sw = ov.get("sw")
        specs.append(
            OverlaySpec(
                asset_path=abs_path,
                nx=nx,
                ny=ny,
                start_sec=start,
                end_sec=end,
                sw=float(sw) if sw else None,
                fade_sec=float(ov.get("fade_sec") or 0.0),
            )
        )
    if not specs:
        raise MediaError("overlay job has no overlays to burn")
    return specs


def process_overlay_job(
    job_id: str, *, gdrive: Optional[GDriveClient] = None
) -> None:
    """queued -> processing -> [delivering] -> done."""
    with get_session() as session:
        job = session.get(OverlayJob, job_id)
        if job is None:
            raise ValueError(f"OverlayJob {job_id!r} not found")

        try:
            transition(job, OverlayStatus.processing)
            session.commit()

            # 1. Resolve + probe the source video.
            src = _resolve_source(job, session)
            info = probe(src)
            job.source_local_path = src
            job.src_width = info.width
            job.src_height = info.height
            job.src_duration_sec = info.duration_sec
            session.commit()

            # 2. Build + validate the placement specs.
            specs = _build_specs(job, info.duration_sec)

            # 3. Burn.
            out_path = os.path.join(overlay_output_dir(job_id), "result.mp4")
            burn_overlays(src, out_path, specs)
            # Normally a no-op (burn_overlays already caps the render); this
            # catches a source the cap could not reach. Never fail the job over
            # it — an oversized deliverable still beats none.
            try:
                enforce_size_limit(out_path)
            except Exception as exc:
                log.warning(
                    "overlay job %s: could not shrink %s to the delivery size "
                    "cap: %s", job_id, out_path, exc,
                )
            job.result_local_path = out_path
            session.commit()

            # 4. Deliver to Drive. Overlay results are ALWAYS auto-uploaded (as
            # long as a target folder is configured), regardless of the job's
            # deliver_to_drive flag — that flag is legacy; older jobs created
            # before auto-delivery still upload on (re-)render.
            folder_id = (
                settings.OVERLAY_GDRIVE_FOLDER_ID
                or settings.GDRIVE_DEFAULT_FOLDER_ID
            )
            if folder_id:
                transition(job, OverlayStatus.delivering)
                session.commit()
                if gdrive is None:
                    gdrive = _default_gdrive()
                label = (
                    job.source_run_id
                    or job.source_tail_edit_id
                    or job.source_caption_id
                    or "upload"
                )[:8]
                name = f"overlay_{label}_{job_id[:8]}.mp4"
                result = gdrive.upload_file(out_path, folder_id=folder_id, name=name)
                job.gdrive_file_id = result.get("id")
                session.flush()
                log.info(
                    "overlay job %s delivered to Drive file=%s", job_id, result.get("id")
                )

            # 5. Done — clear any stale error from a prior failed attempt.
            job.error = None
            transition(job, OverlayStatus.done)
            session.commit()
            log.info("process_overlay_job done: job_id=%s", job_id)

        except Exception as exc:
            log.exception("process_overlay_job failed: job_id=%s", job_id)
            job.error = str(exc)
            if job.status not in (OverlayStatus.done, OverlayStatus.failed):
                try:
                    transition(job, OverlayStatus.failed)
                except Exception:
                    job.status = OverlayStatus.failed
            try:
                session.commit()
            except Exception:
                pass
            raise
