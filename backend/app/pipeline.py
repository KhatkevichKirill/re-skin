"""
pipeline.py — End-to-end job processing orchestration.

Public functions
----------------
analyze_job(job_id, *, detector=None)
    Download/locate the source, probe it, propose segments, persist to DB.
    Transitions: created → analyzing → review.

process_job(job_id, *, kie=None, gdrive=None)
    For each swap segment: cut → upload → create_task → poll → download.
    Then stitch everything together and deliver to Google Drive.
    Transitions: queued → processing → stitching → delivering → done.

resolve_reference_urls(urls_or_paths, kie, *, gdrive=None) -> list[str]
    Convert a mix of http URLs / Drive links / local paths to public URLs
    that Seedance can fetch.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

from . import media as media_mod
from . import face as face_mod
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .kie_client import KieClient, KieTaskFailed
from .models import Job, Segment
from .state_machine import JobStatus, SegmentStatus, transition
from .storage import clips_dir, results_dir, source_path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Aspect ratios accepted by Seedance.
_SEEDANCE_ASPECTS = {"1:1", "4:3", "16:9", "9:16", "21:9", "adaptive"}


def _map_aspect(raw: Optional[str]) -> str:
    """
    Return a Seedance-compatible aspect-ratio string.

    If *raw* (e.g. ``"16:9"``) is in the allowed set, return it directly.
    Otherwise return ``"adaptive"``.
    """
    if raw and raw in _SEEDANCE_ASPECTS:
        return raw
    return "adaptive"


# Seedance rejects reference videos shorter than 1.8s; pad swap clips to this floor.
MIN_SWAP_VIDEO_SEC = 2.0


def _clamp_duration(start: float, end: float) -> int:
    """Round the clip duration to an integer in [4, 15]."""
    dur = int(round(end - start))
    return max(4, min(15, dur))


def _default_kie() -> KieClient:
    return KieClient()


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


# ---------------------------------------------------------------------------
# resolve_reference_urls
# ---------------------------------------------------------------------------


def resolve_reference_urls(
    urls_or_paths: list[str],
    kie: KieClient,
    *,
    gdrive: Optional[GDriveClient] = None,
) -> list[str]:
    """
    Convert a mixed list of URLs / Drive links / local paths to public URLs.

    Rules (applied per item):
    1. Starts with ``http://`` or ``https://`` and is NOT a recognisable Drive
       share link → use as-is.
    2. Google Drive share link (contains ``drive.google.com`` or ``docs.google.com``)
       → download via *gdrive* to a temp file, then upload to kie → public URL.
    3. Anything else → treat as a local file path, upload to kie → public URL.

    Parameters
    ----------
    urls_or_paths:
        Input items to resolve.
    kie:
        KieClient for uploading local/Drive-downloaded files.
    gdrive:
        GDriveClient for downloading Drive links.  Created lazily if None and
        a Drive link is encountered.
    """
    resolved: list[str] = []
    for item in urls_or_paths:
        item = item.strip()

        is_http = item.startswith("http://") or item.startswith("https://")
        is_drive = is_http and (
            "drive.google.com" in item or "docs.google.com" in item
        )

        if is_http and not is_drive:
            # Already a public URL.
            resolved.append(item)
            continue

        if is_drive:
            # Download from Drive then upload to kie.
            if gdrive is None:
                gdrive = _default_gdrive()
            import tempfile
            suffix = ".jpg"  # assume image; kie doesn't care much
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
            try:
                gdrive.download_file(item, tmp_path)
                url = kie.upload_file(tmp_path, "charswap/refs")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            resolved.append(url)
            continue

        # Local path.
        url = kie.upload_file(item, "charswap/refs")
        resolved.append(url)

    return resolved


# ---------------------------------------------------------------------------
# analyze_job
# ---------------------------------------------------------------------------


def analyze_job(job_id: str, *, detector=None) -> None:
    """
    Probe the source video and propose segments.

    Transitions
    -----------
    created → analyzing → review
    (on error: → failed, error_message set, exception re-raised)
    """
    log.info("analyze_job start: job_id=%s", job_id)

    with get_session() as session:
        job: Job = session.get(Job, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        try:
            # created → analyzing
            transition(job, JobStatus.analyzing)
            session.flush()

            # ------------------------------------------------------------------
            # Resolve local source path
            # ------------------------------------------------------------------
            if job.source_type == "gdrive":
                gdrive = _default_gdrive()
                local = source_path(job_id, "mp4")
                log.info("Downloading gdrive source %s → %s", job.source_ref, local)
                gdrive.download_file(job.source_ref, local)
                job.source_local_path = local
                session.flush()
            else:
                local = job.source_local_path
                if not local or not os.path.exists(local):
                    raise FileNotFoundError(
                        f"Source file not found at {local!r} for job {job_id}"
                    )

            # ------------------------------------------------------------------
            # Probe
            # ------------------------------------------------------------------
            info = media_mod.probe(local)
            job.duration_sec = info.duration_sec
            job.width = info.width
            job.height = info.height
            job.fps = info.fps
            job.aspect_ratio = info.aspect_ratio
            session.flush()

            # ------------------------------------------------------------------
            # Propose segments
            # ------------------------------------------------------------------
            proposed = face_mod.propose_segments(
                local,
                duration_sec=info.duration_sec,
                max_segment_sec=float(settings.SEGMENT_MAX_SECONDS),
                detector=detector,
            )
            log.info("propose_segments returned %d segments", len(proposed))

            for idx, ps in enumerate(proposed):
                status = (
                    SegmentStatus.pending
                    if ps.action == "swap"
                    else SegmentStatus.skipped
                )
                seg = Segment(
                    job_id=job_id,
                    index=idx,
                    start_sec=ps.start_sec,
                    end_sec=ps.end_sec,
                    has_face=ps.has_face,
                    action=ps.action,
                    status=status,
                )
                session.add(seg)

            # analyzing → review
            transition(job, JobStatus.review)
            log.info("analyze_job done: job_id=%s, segments=%d", job_id, len(proposed))

        except Exception as exc:
            log.exception("analyze_job failed for job_id=%s", job_id)
            # Only transition to failed if we haven't already reached review.
            if job.status not in (JobStatus.review, JobStatus.failed):
                try:
                    transition(job, JobStatus.failed)
                except Exception:
                    job.status = JobStatus.failed
            job.error_message = str(exc)
            # Commit the failed state before get_session()'s rollback fires.
            try:
                session.commit()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# process_job
# ---------------------------------------------------------------------------


def process_job(
    job_id: str,
    *,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
) -> None:
    """
    Process all segments and stitch the final video.

    Transitions
    -----------
    queued → processing → stitching → delivering → done
    (on error: segment → failed; job → failed, error_message set, exception re-raised)
    """
    log.info("process_job start: job_id=%s", job_id)

    # Lazy-create real clients so tests can inject fakes.
    if kie is None:
        kie = _default_kie()

    with get_session() as session:
        job: Job = session.get(Job, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        try:
            # queued → processing
            transition(job, JobStatus.processing)
            session.flush()

            source = job.source_local_path
            if not source or not os.path.exists(source):
                raise FileNotFoundError(
                    f"Source file not found at {source!r} for job {job_id}"
                )

            # Determine target dimensions.
            info = media_mod.probe(source)
            width, height, fps = media_mod.get_default_target(info)
            duration_sec = info.duration_sec

            # Refresh segments (ordered by index via the relationship).
            segments: list[Segment] = list(job.segments)
            clip_paths: list[str] = []

            # ------------------------------------------------------------------
            # Process each segment in order
            # ------------------------------------------------------------------
            c_dir = clips_dir(job_id)
            r_dir = results_dir(job_id)

            for seg in segments:
                clip_dst = os.path.join(c_dir, f"clip_{seg.index:04d}.mp4")

                if seg.action == "keep":
                    # Cut the untouched gap from source.
                    media_mod.cut_clip(source, seg.start_sec, seg.end_sec, clip_dst)
                    seg.local_clip_path = clip_dst
                    clip_paths.append(clip_dst)
                    continue

                # action == "swap"
                # Resume-awareness: if already completed AND result exists, skip.
                if (
                    seg.status == SegmentStatus.completed
                    and seg.local_result_path
                    and os.path.exists(seg.local_result_path)
                ):
                    log.info(
                        "Segment %s already completed, skipping re-processing", seg.id
                    )
                    clip_paths.append(seg.local_result_path)
                    continue

                # Retry-awareness: a swap segment left in any non-pending state by a
                # previous interrupted/failed run (failed/uploading/submitted/generating)
                # must be reset to pending so _process_swap_segment can restart cleanly
                # (the state machine only allows forward moves out of `pending`).
                if seg.status != SegmentStatus.pending:
                    log.info(
                        "Resetting segment %s from %s to pending for reprocessing",
                        seg.id, seg.status,
                    )
                    seg.status = SegmentStatus.pending
                    seg.error_message = None
                    seg.seedance_task_id = None
                    seg.seedance_result_url = None
                    session.flush()

                try:
                    _process_swap_segment(
                        seg=seg,
                        source=source,
                        duration_sec=duration_sec,
                        c_dir=c_dir,
                        r_dir=r_dir,
                        clip_dst=clip_dst,
                        job=job,
                        kie=kie,
                        gdrive=gdrive,
                        session=session,
                    )
                    clip_paths.append(seg.local_result_path)
                except Exception as exc:
                    log.exception("Segment %s failed", seg.id)
                    if seg.status != SegmentStatus.failed:
                        try:
                            transition(seg, SegmentStatus.failed)
                        except Exception:
                            seg.status = SegmentStatus.failed
                    seg.error_message = str(exc)
                    # Abort job.
                    if job.status not in (JobStatus.failed,):
                        try:
                            transition(job, JobStatus.failed)
                        except Exception:
                            job.status = JobStatus.failed
                    job.error_message = f"Segment {seg.index} failed: {exc}"
                    # Commit the error state before get_session()'s rollback fires.
                    try:
                        session.commit()
                    except Exception:
                        pass
                    raise

            # ------------------------------------------------------------------
            # Stitch
            # ------------------------------------------------------------------
            transition(job, JobStatus.stitching)
            session.flush()

            final_dst = os.path.join(r_dir, "final.mp4")
            log.info(
                "Stitching %d clips → %s (%dx%d @ %.2ffps)",
                len(clip_paths),
                final_dst,
                width,
                height,
                fps,
            )
            media_mod.stitch(
                clip_paths,
                audio_source=source,
                dst=final_dst,
                width=width,
                height=height,
                fps=fps,
            )
            job.result_local_path = final_dst
            session.flush()

            # ------------------------------------------------------------------
            # Deliver
            # ------------------------------------------------------------------
            transition(job, JobStatus.delivering)
            session.flush()

            folder_id = job.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id:
                if gdrive is None:
                    gdrive = _default_gdrive()
                upload_name = f"reskin_{job_id}.mp4"
                result = gdrive.upload_file(final_dst, folder_id, upload_name)
                job.result_gdrive_file_id = result.get("id")
                log.info(
                    "Uploaded final video to Drive: %s", result.get("webViewLink")
                )
                session.flush()

            transition(job, JobStatus.done)
            log.info("process_job done: job_id=%s", job_id)

        except Exception as exc:
            # Only set failed if not already in a terminal state set by
            # the inner segment failure path.
            if job.status not in (JobStatus.done, JobStatus.failed):
                try:
                    transition(job, JobStatus.failed)
                except Exception:
                    job.status = JobStatus.failed
            if not job.error_message:
                job.error_message = str(exc)
            # Commit error state before get_session()'s rollback fires.
            # The inner segment failure handler already committed; this
            # handles other outer failures (stitch, deliver, etc.).
            try:
                session.commit()
            except Exception:
                pass
            raise


def _process_swap_segment(
    *,
    seg: Segment,
    source: str,
    duration_sec: float,
    c_dir: str,
    r_dir: str,
    clip_dst: str,
    job: Job,
    kie: KieClient,
    gdrive: Optional[GDriveClient],
    session,
) -> None:
    """Handle a single 'swap' segment end-to-end (cut → upload → task → download)."""

    # Apply pre/post roll to the cut boundaries.
    raw_start = seg.start_sec - seg.pre_roll_sec
    raw_end = seg.end_sec + seg.post_roll_sec
    clip_start = max(0.0, raw_start)
    clip_end = min(duration_sec, raw_end)

    # Ensure the clip meets Seedance's minimum reference-video duration (>=1.8s).
    # Pad forward first, then backward if we're near the end of the video.
    if clip_end - clip_start < MIN_SWAP_VIDEO_SEC:
        clip_end = min(duration_sec, clip_start + MIN_SWAP_VIDEO_SEC)
        if clip_end - clip_start < MIN_SWAP_VIDEO_SEC:
            clip_start = max(0.0, clip_end - MIN_SWAP_VIDEO_SEC)

    # 1. Cut
    media_mod.cut_clip(source, clip_start, clip_end, clip_dst)
    seg.local_clip_path = clip_dst
    session.flush()

    # 2. Upload clip to kie
    transition(seg, SegmentStatus.uploading)
    session.flush()

    clip_url = kie.upload_file(clip_dst, "charswap/segments")
    seg.kie_upload_url = clip_url
    session.flush()

    # 3. Resolve reference image URLs
    raw_refs = seg.reference_image_urls_override or job.default_reference_image_urls or []
    ref_urls = resolve_reference_urls(list(raw_refs), kie, gdrive=gdrive)

    # 4. Create task
    transition(seg, SegmentStatus.submitted)
    session.flush()

    aspect = _map_aspect(job.aspect_ratio)
    clip_duration = _clamp_duration(clip_start, clip_end)
    prompt = seg.prompt_override or job.default_prompt or ""

    task_id = kie.create_task(
        prompt=prompt,
        reference_image_urls=ref_urls,
        reference_video_urls=[clip_url],
        resolution=job.resolution or settings.DEFAULT_RESOLUTION,
        aspect_ratio=aspect,
        duration=clip_duration,
    )
    seg.seedance_task_id = task_id
    session.flush()

    # 5. Poll
    transition(seg, SegmentStatus.generating)
    session.flush()

    result_url = kie.poll_task(task_id)
    seg.seedance_result_url = result_url
    session.flush()

    # 6. Download result
    result_dst = os.path.join(r_dir, f"result_{seg.index:04d}.mp4")
    kie.download_result(result_url, result_dst)
    seg.local_result_path = result_dst

    # 7. Mark complete
    transition(seg, SegmentStatus.completed)
    session.flush()
    log.info("Segment %s completed → %s", seg.id, result_dst)
