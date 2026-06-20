"""
pipeline_v2.py — End-to-end processing for the v2 Project → Runs model.

Public functions
----------------
analyze_project(project_id, *, detector=None)
    Download/locate the source, probe it, propose segments, persist to DB.
    Transitions: created → analyzing → ready.

process_run(run_id, *, kie=None, gdrive=None)
    For each swap SegmentDef: cut → upload → create_task → poll → download.
    Then stitch everything together and deliver to Google Drive.
    Transitions: queued → processing → stitching → delivering → done.

Shared helpers (resolve_reference_urls, _map_aspect, _clamp_duration,
MIN_SWAP_VIDEO_SEC) are imported from pipeline.py — single source of truth.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from . import media as media_mod
from . import face as face_mod
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .kie_client import KieClient
from .models import Run, RunSegment, SegmentDef, VideoProject
from .pipeline import (
    MIN_SWAP_VIDEO_SEC,
    _clamp_duration,
    _map_aspect,
    resolve_reference_urls,
)
from .state_machine import ProjectStatus, RunStatus, SegmentStatus, transition
from .storage import (
    project_source_path,
    run_clips_dir,
    run_results_dir,
)

log = logging.getLogger(__name__)


def _default_kie() -> KieClient:
    return KieClient()


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


# ---------------------------------------------------------------------------
# analyze_project
# ---------------------------------------------------------------------------


def analyze_project(project_id: str, *, detector=None) -> None:
    """
    Probe the source video and propose segments for a VideoProject.

    Transitions
    -----------
    created → analyzing → ready
    (on error: → failed, error_message set, exception re-raised)
    """
    log.info("analyze_project start: project_id=%s", project_id)

    with get_session() as session:
        project: VideoProject = session.get(VideoProject, project_id)
        if project is None:
            raise ValueError(f"VideoProject not found: {project_id}")

        try:
            # created → analyzing (commit so readers see live state)
            transition(project, ProjectStatus.analyzing)
            session.commit()

            # ------------------------------------------------------------------
            # Resolve local source path
            # ------------------------------------------------------------------
            if project.source_type == "gdrive":
                gdrive = _default_gdrive()
                local = project_source_path(project_id, "mp4")
                log.info(
                    "Downloading gdrive source %s → %s", project.source_ref, local
                )
                gdrive.download_file(project.source_ref, local)
                project.source_local_path = local
                session.commit()
            else:
                local = project.source_local_path
                if not local or not os.path.exists(local):
                    raise FileNotFoundError(
                        f"Source file not found at {local!r} for project {project_id}"
                    )

            # ------------------------------------------------------------------
            # Probe
            # ------------------------------------------------------------------
            info = media_mod.probe(local)
            project.duration_sec = info.duration_sec
            project.width = info.width
            project.height = info.height
            project.fps = info.fps
            project.aspect_ratio = info.aspect_ratio
            session.commit()

            # ------------------------------------------------------------------
            # Propose segments — create SegmentDef rows
            # ------------------------------------------------------------------
            proposed = face_mod.propose_segments(
                local,
                duration_sec=info.duration_sec,
                max_segment_sec=float(settings.SEGMENT_MAX_SECONDS),
                detector=detector,
            )
            log.info("propose_segments returned %d segments", len(proposed))

            for idx, ps in enumerate(proposed):
                sd = SegmentDef(
                    project_id=project_id,
                    index=idx,
                    start_sec=ps.start_sec,
                    end_sec=ps.end_sec,
                    has_face=ps.has_face,
                    action=ps.action,
                )
                session.add(sd)

            # analyzing → ready (commit)
            transition(project, ProjectStatus.ready)
            session.commit()
            log.info(
                "analyze_project done: project_id=%s, segments=%d",
                project_id,
                len(proposed),
            )

        except Exception as exc:
            log.exception("analyze_project failed for project_id=%s", project_id)
            # Only transition to failed if we haven't already reached ready.
            if project.status not in (ProjectStatus.ready, ProjectStatus.failed):
                try:
                    transition(project, ProjectStatus.failed)
                except Exception:
                    project.status = ProjectStatus.failed
            project.error_message = str(exc)
            # Commit the failed state before get_session()'s rollback fires.
            try:
                session.commit()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# process_run
# ---------------------------------------------------------------------------


def process_run(
    run_id: str,
    *,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
) -> None:
    """
    Process all swap segments for a Run and stitch the final video.

    Transitions
    -----------
    queued → processing → stitching → delivering → done
    (on error: RunSegment → failed; Run → failed, error_message set, exception re-raised)
    """
    log.info("process_run start: run_id=%s", run_id)

    if kie is None:
        kie = _default_kie()

    with get_session() as session:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        project: VideoProject = session.get(VideoProject, run.project_id)
        if project is None:
            raise ValueError(f"VideoProject not found: {run.project_id}")

        try:
            # queued → processing (commit so readers see live state)
            transition(run, RunStatus.processing)
            session.commit()

            source = project.source_local_path
            if not source or not os.path.exists(source):
                raise FileNotFoundError(
                    f"Source file not found at {source!r} for project {run.project_id}"
                )

            # Determine target dimensions from the project source.
            info = media_mod.probe(source)
            width, height, fps = media_mod.get_default_target(info)
            duration_sec = info.duration_sec

            # Load SegmentDefs ordered by index.
            seg_defs: list[SegmentDef] = list(project.segments)
            if not seg_defs:
                raise ValueError(
                    f"Project {run.project_id} has no segments — run analyze_project first"
                )

            # ------------------------------------------------------------------
            # Ensure RunSegment rows exist for every swap SegmentDef (idempotent).
            # ------------------------------------------------------------------
            existing_rs: dict[str, RunSegment] = {
                rs.segment_def_id: rs for rs in run.run_segments
            }
            for sd in seg_defs:
                if sd.action == "swap" and sd.id not in existing_rs:
                    rs = RunSegment(
                        run_id=run_id,
                        segment_def_id=sd.id,
                        index=sd.index,
                        status=SegmentStatus.pending,
                    )
                    session.add(rs)
            session.flush()

            # Reload run_segments after potential inserts.
            session.refresh(run)

            # Build a segment_def_id → RunSegment map.
            rs_map: dict[str, RunSegment] = {
                rs.segment_def_id: rs for rs in run.run_segments
            }

            c_dir = run_clips_dir(run_id, run.project_id)
            r_dir = run_results_dir(run_id, run.project_id)

            clip_paths: list[str] = []

            # ------------------------------------------------------------------
            # Process each SegmentDef in order
            # ------------------------------------------------------------------
            for sd in seg_defs:
                clip_dst = os.path.join(c_dir, f"clip_{sd.index:04d}.mp4")

                if sd.action == "keep":
                    media_mod.cut_clip(source, sd.start_sec, sd.end_sec, clip_dst)
                    clip_paths.append(clip_dst)
                    continue

                # action == "swap"
                rs = rs_map[sd.id]

                # Resume-awareness: if already completed AND result exists, skip.
                if (
                    rs.status == SegmentStatus.completed
                    and rs.local_result_path
                    and os.path.exists(rs.local_result_path)
                ):
                    log.info(
                        "RunSegment %s (index %d) already completed, skipping",
                        rs.id,
                        rs.index,
                    )
                    clip_paths.append(rs.local_result_path)
                    continue

                # Retry-awareness: reset non-pending, non-completed RunSegment to pending.
                if rs.status != SegmentStatus.pending:
                    log.info(
                        "Resetting RunSegment %s from %s to pending for reprocessing",
                        rs.id,
                        rs.status,
                    )
                    rs.status = SegmentStatus.pending
                    rs.error_message = None
                    rs.seedance_task_id = None
                    rs.seedance_result_url = None
                    session.flush()

                try:
                    _process_run_swap_segment(
                        rs=rs,
                        sd=sd,
                        run=run,
                        source=source,
                        duration_sec=duration_sec,
                        c_dir=c_dir,
                        r_dir=r_dir,
                        clip_dst=clip_dst,
                        kie=kie,
                        gdrive=gdrive,
                        session=session,
                    )
                    clip_paths.append(rs.local_result_path)
                except Exception as exc:
                    log.exception("RunSegment %s (index %d) failed", rs.id, rs.index)
                    if rs.status != SegmentStatus.failed:
                        try:
                            transition(rs, SegmentStatus.failed)
                        except Exception:
                            rs.status = SegmentStatus.failed
                    rs.error_message = str(exc)
                    # Abort run.
                    if run.status not in (RunStatus.failed,):
                        try:
                            transition(run, RunStatus.failed)
                        except Exception:
                            run.status = RunStatus.failed
                    run.error_message = f"Segment {sd.index} failed: {exc}"
                    # Commit the error state before get_session()'s rollback fires.
                    try:
                        session.commit()
                    except Exception:
                        pass
                    raise

            # ------------------------------------------------------------------
            # Stitch
            # ------------------------------------------------------------------
            transition(run, RunStatus.stitching)
            session.commit()  # publish status before the (slow) stitch

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
            run.result_local_path = final_dst
            session.flush()

            # ------------------------------------------------------------------
            # Deliver
            # ------------------------------------------------------------------
            transition(run, RunStatus.delivering)
            session.commit()  # publish status before the (slow) Drive upload

            folder_id = run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id:
                if gdrive is None:
                    gdrive = _default_gdrive()
                upload_name = f"reskin_run_{run_id}.mp4"
                result = gdrive.upload_file(final_dst, folder_id, upload_name)
                run.result_gdrive_file_id = result.get("id")
                log.info(
                    "Uploaded final video to Drive: %s", result.get("webViewLink")
                )
                session.flush()

            transition(run, RunStatus.done)
            session.commit()
            log.info("process_run done: run_id=%s", run_id)

        except Exception as exc:
            # Only set failed if not already in a terminal state set by the inner
            # segment failure path.
            if run.status not in (RunStatus.done, RunStatus.failed):
                try:
                    transition(run, RunStatus.failed)
                except Exception:
                    run.status = RunStatus.failed
            if not run.error_message:
                run.error_message = str(exc)
            # Commit error state before get_session()'s rollback fires.
            try:
                session.commit()
            except Exception:
                pass
            raise


def _process_run_swap_segment(
    *,
    rs: RunSegment,
    sd: SegmentDef,
    run: Run,
    source: str,
    duration_sec: float,
    c_dir: str,
    r_dir: str,
    clip_dst: str,
    kie: KieClient,
    gdrive: Optional[GDriveClient],
    session,
) -> None:
    """Handle a single 'swap' RunSegment end-to-end (cut → upload → task → download)."""

    # Apply pre/post roll to the cut boundaries.
    raw_start = sd.start_sec - sd.pre_roll_sec
    raw_end = sd.end_sec + sd.post_roll_sec
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
    rs.local_clip_path = clip_dst
    session.flush()

    # 2. Upload clip to kie
    transition(rs, SegmentStatus.uploading)
    session.commit()  # publish live RunSegment status

    clip_url = kie.upload_file(clip_dst, "charswap/segments")
    rs.kie_upload_url = clip_url
    session.flush()

    # 3. Resolve reference image URLs (from the Run, not per-segment overrides)
    raw_refs = run.reference_image_urls or []
    ref_urls = resolve_reference_urls(list(raw_refs), kie, gdrive=gdrive)

    # 4. Create task
    transition(rs, SegmentStatus.submitted)
    session.commit()  # publish live RunSegment status

    aspect = _map_aspect(run.project.aspect_ratio)
    clip_duration = _clamp_duration(clip_start, clip_end)
    prompt = run.prompt or ""

    task_id = kie.create_task(
        prompt=prompt,
        reference_image_urls=ref_urls,
        reference_video_urls=[clip_url],
        resolution=run.resolution or settings.DEFAULT_RESOLUTION,
        aspect_ratio=aspect,
        duration=clip_duration,
    )
    rs.seedance_task_id = task_id
    session.flush()

    # 5. Poll
    transition(rs, SegmentStatus.generating)
    session.commit()  # publish live RunSegment status

    result_url = kie.poll_task(task_id)
    rs.seedance_result_url = result_url
    session.flush()

    # 6. Download result
    result_dst = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
    kie.download_result(result_url, result_dst)
    rs.local_result_path = result_dst

    # 7. Mark complete
    transition(rs, SegmentStatus.completed)
    session.commit()  # publish completion so progress count advances live
    log.info("RunSegment %s (index %d) completed → %s", rs.id, rs.index, result_dst)
