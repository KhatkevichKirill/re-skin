"""
pipeline_v2.py — End-to-end processing for the v2 Project → Runs model.

Public functions
----------------
analyze_project(project_id, *, detector=None)
    Download/locate the source, probe it, propose segments, persist to DB.
    Transitions: created → analyzing → ready.

process_run(run_id, *, kie=None, gdrive=None)
    Submit ALL swap segments to Seedance in parallel, then poll every task
    concurrently (round-robin), downloading each result as it lands. A task
    with no result within RUN_SKIP_TIMEOUT_SEC (default 2h) — or one that
    fails — is skipped: that segment falls back to the original (un-swapped)
    clip so one stuck segment never blocks or fails the whole run.
    Then stitch everything together and deliver to Google Drive.
    Transitions: queued → processing → stitching → delivering → done.

Shared helpers (resolve_reference_urls, _map_aspect, _clamp_duration,
MIN_SWAP_VIDEO_SEC) are imported from pipeline.py — single source of truth.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    _map_omni_aspect,
    _omni_resolution,
    _snap_omni_duration,
    resolve_reference_urls,
)
from .state_machine import ProjectStatus, RunStatus, SegmentStatus, transition
from .storage import (
    project_source_path,
    run_clips_dir,
    run_results_dir,
)

log = logging.getLogger(__name__)

# Parallel-submit tuning (env-overridable).
# Seedance tasks are submitted all at once and polled concurrently; a task that
# yields no result within RUN_SKIP_TIMEOUT_SEC is skipped (original clip used).
RUN_SKIP_TIMEOUT_SEC = float(os.getenv("RUN_SKIP_TIMEOUT_SEC", "7200"))  # 2 hours
RUN_POLL_INTERVAL_SEC = float(os.getenv("RUN_POLL_INTERVAL_SEC", "15"))

# Delivery (Google Drive upload) retry tuning. The upload itself is chunked with
# per-chunk retries; this is a whole-upload retry so a fully-failed delivery
# self-heals without re-generating or re-stitching the video.
RUN_DELIVER_ATTEMPTS = int(os.getenv("RUN_DELIVER_ATTEMPTS", "3"))
RUN_DELIVER_BACKOFF_SEC = float(os.getenv("RUN_DELIVER_BACKOFF_SEC", "10"))

# Stitch-phase parallelism: number of concurrent cut_clip calls for keep/fallback
# segments during assembly.  Bounded so we don't saturate the CPU — each ffmpeg
# process already uses FFMPEG_THREADS threads internally.  Default 2 is safe
# even on a single worker; set higher only if worker has spare CPU headroom.
# Set to 1 to disable parallelism (equivalent to the old serial loop).
STITCH_CUT_CONCURRENCY = int(os.getenv("STITCH_CUT_CONCURRENCY", "2"))

# Submit-phase parallelism: max concurrent cut/upload/create-task calls during
# the submit phase.  Each thread opens its own DB session.  Default 2 keeps I/O
# pressure low on a single-core VPS; raise to 4-6 if upload bandwidth allows.
SUBMIT_CONCURRENCY = int(os.getenv("SUBMIT_CONCURRENCY", "2"))


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

            # Resolve local source path.
            if project.source_type == "gdrive":
                gdrive = _default_gdrive()
                local = project_source_path(project_id, "mp4")
                log.info("Downloading gdrive source %s → %s", project.source_ref, local)
                gdrive.download_file(project.source_ref, local)
                project.source_local_path = local
                session.commit()
            else:
                local = project.source_local_path
                if not local or not os.path.exists(local):
                    raise FileNotFoundError(
                        f"Source file not found at {local!r} for project {project_id}"
                    )

            # Probe.
            info = media_mod.probe(local)
            project.duration_sec = info.duration_sec
            project.width = info.width
            project.height = info.height
            project.fps = info.fps
            project.aspect_ratio = info.aspect_ratio
            session.commit()

            # Propose segments — create SegmentDef rows.
            proposed = face_mod.propose_segments(
                local,
                duration_sec=info.duration_sec,
                max_segment_sec=float(settings.SEGMENT_MAX_SECONDS),
                detector=detector,
            )
            log.info("propose_segments returned %d segments", len(proposed))

            # Idempotent re-analyze: clear any existing SegmentDefs first so a
            # re-run replaces the segmentation instead of duplicating it.
            # (Cascades to RunSegments via FK — re-analysis invalidates prior runs.)
            for old in list(project.segments):
                session.delete(old)
            session.flush()

            for idx, ps in enumerate(proposed):
                session.add(
                    SegmentDef(
                        project_id=project_id,
                        index=idx,
                        start_sec=ps.start_sec,
                        end_sec=ps.end_sec,
                        has_face=ps.has_face,
                        action=ps.action,
                    )
                )

            transition(project, ProjectStatus.ready)
            session.commit()
            log.info(
                "analyze_project done: project_id=%s, segments=%d",
                project_id, len(proposed),
            )

        except Exception as exc:
            log.exception("analyze_project failed for project_id=%s", project_id)
            if project.status not in (ProjectStatus.ready, ProjectStatus.failed):
                try:
                    transition(project, ProjectStatus.failed)
                except Exception:
                    project.status = ProjectStatus.failed
            project.error_message = str(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# process_run — parallel submit + concurrent poll
# ---------------------------------------------------------------------------


def _clip_bounds(sd: SegmentDef, duration_sec: float) -> tuple[float, float]:
    """Cut bounds for a swap segment: sd range ± rolls, padded to Seedance's min."""
    start = max(0.0, sd.start_sec - sd.pre_roll_sec)
    end = min(duration_sec, sd.end_sec + sd.post_roll_sec)
    if end - start < MIN_SWAP_VIDEO_SEC:
        end = min(duration_sec, start + MIN_SWAP_VIDEO_SEC)
        if end - start < MIN_SWAP_VIDEO_SEC:
            start = max(0.0, end - MIN_SWAP_VIDEO_SEC)
    return start, end


def _parse_result_url(data: dict) -> Optional[str]:
    """Extract resultUrls[0] from a recordInfo 'data' dict (resultJson is a string)."""
    raw = data.get("resultJson") or "{}"
    try:
        urls = json.loads(raw).get("resultUrls") or []
    except (ValueError, TypeError):
        urls = []
    return urls[0] if urls else None


def _submit_swap_segment_isolated(
    *,
    rs_id: str,
    sd_start_sec: float,
    sd_end_sec: float,
    sd_pre_roll_sec: float,
    sd_post_roll_sec: float,
    sd_index: int,
    run_model: str,
    run_prompt: Optional[str],
    run_resolution: Optional[str],
    project_aspect_ratio: Optional[str],
    project_width: Optional[int],
    project_height: Optional[int],
    source: str,
    duration_sec: float,
    clip_dst: str,
    ref_urls: list,
    prompt_override: Optional[str],
    kie: KieClient,
) -> str:
    """Cut, upload, then create_task for one swap segment.

    Opens its own DB session so this function is safe to call from a thread pool
    (no shared SQLAlchemy Session across threads).  Returns the Seedance task_id.
    Logs per-segment timing for observability.
    """
    t0 = time.monotonic()

    # Recompute clip bounds from primitive data (mirrors _clip_bounds logic).
    clip_start = max(0.0, sd_start_sec - sd_pre_roll_sec)
    clip_end = min(duration_sec, sd_end_sec + sd_post_roll_sec)
    if clip_end - clip_start < MIN_SWAP_VIDEO_SEC:
        clip_end = min(duration_sec, clip_start + MIN_SWAP_VIDEO_SEC)
        if clip_end - clip_start < MIN_SWAP_VIDEO_SEC:
            clip_start = max(0.0, clip_end - MIN_SWAP_VIDEO_SEC)

    media_mod.cut_clip(source, clip_start, clip_end, clip_dst)
    t_cut = time.monotonic()

    with get_session() as session:
        rs = session.get(RunSegment, rs_id)
        rs.local_clip_path = clip_dst
        transition(rs, SegmentStatus.uploading)
        session.commit()

    clip_url = kie.upload_file(clip_dst, "charswap/segments")
    t_upload = time.monotonic()

    with get_session() as session:
        rs = session.get(RunSegment, rs_id)
        rs.kie_upload_url = clip_url
        transition(rs, SegmentStatus.submitted)
        session.commit()

    effective_prompt = prompt_override if prompt_override else (run_prompt or "")

    if run_model == "gemini-omni":
        trim_end = round(min(clip_end - clip_start, 10.0), 2)
        task_id = kie.create_omni_task(
            prompt=effective_prompt,
            image_urls=ref_urls,
            video_url=clip_url,
            video_start=0,
            video_end=trim_end,
            resolution=_omni_resolution(run_resolution),
            aspect_ratio=_map_omni_aspect(project_aspect_ratio, project_width, project_height),
            duration=_snap_omni_duration(clip_start, clip_end),
        )
    else:
        aspect = _map_aspect(project_aspect_ratio)
        clip_duration = _clamp_duration(clip_start, clip_end)
        task_id = kie.create_task(
            prompt=effective_prompt,
            reference_image_urls=ref_urls,
            reference_video_urls=[clip_url],
            resolution=run_resolution or settings.DEFAULT_RESOLUTION,
            aspect_ratio=aspect,
            duration=clip_duration,
        )

    with get_session() as session:
        rs = session.get(RunSegment, rs_id)
        rs.seedance_task_id = task_id
        transition(rs, SegmentStatus.generating)
        session.commit()

    t_done = time.monotonic()
    log.info(
        "segment idx=%d submitted task=%s cut=%.1fs upload=%.1fs create=%.1fs total=%.1fs",
        sd_index, task_id,
        t_cut - t0,
        t_upload - t_cut,
        t_done - t_upload,
        t_done - t0,
    )
    return task_id


def _deliver_with_retry(gdrive: GDriveClient, path: str, folder_id: str, run_id: str) -> dict:
    """Upload the final video to Drive, retrying the whole upload on failure.

    Raises the last exception if every attempt fails (the caller then marks the
    run failed — but the final video stays on disk, so a manual Retry re-delivers
    without re-stitching).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, RUN_DELIVER_ATTEMPTS + 1):
        try:
            return gdrive.upload_file(path, folder_id, f"reskin_run_{run_id}.mp4")
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Drive upload attempt %d/%d failed for run %s: %s",
                attempt, RUN_DELIVER_ATTEMPTS, run_id, exc,
            )
            if attempt < RUN_DELIVER_ATTEMPTS:
                time.sleep(RUN_DELIVER_BACKOFF_SEC * attempt)
    raise last_exc  # type: ignore[misc]


def process_run(
    run_id: str,
    *,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
) -> None:
    """
    Process all swap segments for a Run (concurrent submit + concurrent poll) and
    stitch the final video.

    Transitions: queued → processing → stitching → delivering → done.
    A swap segment that fails or times out is skipped (original clip used); only
    fatal errors (missing source, stitch failure) fail the whole run.
    Phase durations are logged for observability.
    """
    t_run_start = time.monotonic()
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
            # queued → processing: normal first-time path.
            # Any other active state (processing/stitching/delivering) means this
            # is an orphan resume — reset to queued first, then advance.
            # The processing → queued edge was added in state_machine.py (TR5b).
            if run.status != RunStatus.queued:
                log.info(
                    "process_run: run %s is in status=%r (not queued) — "
                    "resetting to queued for orphan resume",
                    run_id, run.status,
                )
                try:
                    transition(run, RunStatus.queued)
                except Exception:
                    # Fallback for stitching/delivering which lack a direct →queued
                    # edge: go via failed first.
                    transition(run, RunStatus.failed)
                    transition(run, RunStatus.queued)
                session.commit()
            transition(run, RunStatus.processing)
            session.commit()

            source = project.source_local_path
            if not source or not os.path.exists(source):
                raise FileNotFoundError(
                    f"Source file not found at {source!r} for project {run.project_id}"
                )

            info = media_mod.probe(source)
            width, height, fps = media_mod.get_default_target(info)
            duration_sec = info.duration_sec

            seg_defs: list[SegmentDef] = list(project.segments)
            if not seg_defs:
                raise ValueError(
                    f"Project {run.project_id} has no segments — run analyze_project first"
                )

            # Ensure a RunSegment row exists for every swap SegmentDef (idempotent).
            existing_rs = {rs.segment_def_id: rs for rs in run.run_segments}
            for sd in seg_defs:
                if sd.action == "swap" and sd.id not in existing_rs:
                    session.add(
                        RunSegment(
                            run_id=run_id, segment_def_id=sd.id,
                            index=sd.index, status=SegmentStatus.pending,
                        )
                    )
            session.flush()
            session.refresh(run)
            rs_map = {rs.segment_def_id: rs for rs in run.run_segments}

            c_dir = run_clips_dir(run_id, run.project_id)
            r_dir = run_results_dir(run_id, run.project_id)

            # ---------------------------------------------------------------
            # Reference resolution
            # ---------------------------------------------------------------
            t_ref_start = time.monotonic()
            run_ref_urls = resolve_reference_urls(
                list(run.reference_image_urls or []), kie, gdrive=gdrive
            )
            # Cache per-segment override resolutions to avoid duplicate uploads.
            _override_ref_cache: dict[str, list[str]] = {}
            log.info(
                "run_id=%s ref_resolution=%.1fs refs=%d",
                run_id, time.monotonic() - t_ref_start, len(run_ref_urls),
            )

            # ---------------------------------------------------------------
            # Submit phase — prepare work items (serial) then submit concurrently.
            # ---------------------------------------------------------------
            pending: dict[str, dict] = {}  # task_id -> {rs_id, index, deadline}
            submit_work: list[dict] = []   # segments queued for concurrent submit

            for sd in seg_defs:
                if sd.action != "swap":
                    continue
                rs = rs_map[sd.id]

                # Resume: already completed with a real result → don't resubmit.
                if (
                    rs.status == SegmentStatus.completed
                    and rs.local_result_path
                    and os.path.exists(rs.local_result_path)
                ):
                    log.info("RunSegment %s (idx %d) already completed, skipping submit",
                             rs.id, rs.index)
                    continue

                # Resume / no-rebill: segment not yet completed but has a
                # seedance_task_id (worker crashed during the poll loop).  Check
                # kie.ai first — if the task already succeeded we can download the
                # result without resubmitting, saving a Seedance credit.
                if (
                    rs.status != SegmentStatus.pending
                    and rs.seedance_task_id
                ):
                    task_id = rs.seedance_task_id
                    log.info(
                        "RunSegment %s (idx %d) has existing task_id=%s (status=%s) "
                        "— checking kie.ai before resubmitting",
                        rs.id, rs.index, task_id, rs.status,
                    )
                    try:
                        data = kie.get_task(task_id)
                        state = (data.get("state") or "").lower()
                    except Exception as exc:
                        log.warning(
                            "get_task(%s) failed during resume check: %s — will resubmit",
                            task_id, exc,
                        )
                        state = "unknown"

                    if state == "success":
                        url = _parse_result_url(data)
                        if url:
                            result_dst = os.path.join(
                                r_dir, f"result_{sd.index:04d}.mp4"
                            )
                            try:
                                kie.download_result(url, result_dst)
                                rs.seedance_result_url = url
                                rs.local_result_path = result_dst
                                try:
                                    transition(rs, SegmentStatus.completed)
                                except Exception:
                                    rs.status = SegmentStatus.completed
                                session.commit()
                                log.info(
                                    "RunSegment %s (idx %d) task %s was already "
                                    "success — recovered without rebilling",
                                    rs.id, rs.index, task_id,
                                )
                                continue
                            except Exception as exc:
                                log.warning(
                                    "download_result failed for task %s: %s "
                                    "— will resubmit",
                                    task_id, exc,
                                )
                        else:
                            log.warning(
                                "task %s success but no result url — will resubmit",
                                task_id,
                            )
                    elif state not in ("fail",):
                        # Still in-progress or unknown — if we got here via a
                        # restart the task may still be running on kie.ai.
                        # We re-add it to the pending poll set to avoid
                        # re-submitting a task that Seedance is already processing.
                        log.info(
                            "RunSegment %s (idx %d) task %s state=%r — "
                            "resuming poll without resubmitting",
                            rs.id, rs.index, task_id, state,
                        )
                        # Ensure segment is in generating state for the poll loop.
                        if rs.status != SegmentStatus.generating:
                            try:
                                # generating requires submitted→generating path;
                                # force-set status directly since we're resuming.
                                rs.status = SegmentStatus.generating
                            except Exception:
                                pass
                            session.flush()
                        pending[task_id] = {
                            "rs_id": rs.id,
                            "index": sd.index,
                            "deadline": time.monotonic() + RUN_SKIP_TIMEOUT_SEC,
                        }
                        continue
                    # state == "fail" or download failed → fall through to reset+resubmit

                # Retry: reset an interrupted RunSegment before resubmitting.
                if rs.status != SegmentStatus.pending:
                    rs.status = SegmentStatus.pending
                    rs.error_message = None
                    rs.seedance_task_id = None
                    rs.seedance_result_url = None
                    session.flush()

                # Resolve effective refs for this segment (override takes priority).
                if rs.reference_image_urls_override:
                    cache_key = rs.id
                    if cache_key not in _override_ref_cache:
                        _override_ref_cache[cache_key] = resolve_reference_urls(
                            list(rs.reference_image_urls_override), kie, gdrive=gdrive
                        )
                    effective_ref_urls = _override_ref_cache[cache_key]
                else:
                    effective_ref_urls = run_ref_urls

                # Queue work item for concurrent submit (primitive data only —
                # no ORM objects, safe to pass across thread boundaries).
                submit_work.append({
                    "rs_id": rs.id,
                    "sd_start_sec": sd.start_sec,
                    "sd_end_sec": sd.end_sec,
                    "sd_pre_roll_sec": sd.pre_roll_sec,
                    "sd_post_roll_sec": sd.post_roll_sec,
                    "sd_index": sd.index,
                    "run_model": run.model or "seedance",
                    "run_prompt": run.prompt,
                    "run_resolution": run.resolution,
                    "project_aspect_ratio": project.aspect_ratio,
                    "project_width": project.width,
                    "project_height": project.height,
                    "source": source,
                    "duration_sec": duration_sec,
                    "clip_dst": os.path.join(c_dir, f"clip_{sd.index:04d}.mp4"),
                    "ref_urls": list(effective_ref_urls),
                    "prompt_override": rs.prompt_override,
                })

            # Commit so that newly created RunSegment rows are visible to the
            # independent DB sessions opened by each submit thread.
            # (flush() only writes within the current transaction; other sessions
            # cannot see uncommitted rows.)
            session.commit()

            # Concurrent submit via thread pool.
            t_submit_start = time.monotonic()
            if submit_work:
                log.info(
                    "run_id=%s submit_phase: submitting %d segment(s) "
                    "concurrency=%d",
                    run_id, len(submit_work), SUBMIT_CONCURRENCY,
                )
                with ThreadPoolExecutor(max_workers=SUBMIT_CONCURRENCY) as pool:
                    submit_futures = [
                        (work, pool.submit(
                            _submit_swap_segment_isolated, **work, kie=kie
                        ))
                        for work in submit_work
                    ]
                # Collect results; re-raise on first failure (preserves existing
                # serial semantics: any submit failure fails the whole run).
                for work, fut in submit_futures:
                    task_id = fut.result()
                    pending[task_id] = {
                        "rs_id": work["rs_id"],
                        "index": work["sd_index"],
                        "deadline": time.monotonic() + RUN_SKIP_TIMEOUT_SEC,
                    }

            log.info(
                "run_id=%s submit_phase_total=%.1fs segments_submitted=%d",
                run_id, time.monotonic() - t_submit_start, len(submit_work),
            )

            # Did we (re)submit anything this run? If not, and a final video
            # already exists, this is a delivery-only retry → skip the re-stitch.
            did_submit = bool(pending)
            log.info("Submitted %d swap task(s) to Seedance for run %s",
                     len(pending), run_id)

            # ---------------------------------------------------------------
            # Poll phase — round-robin over all pending tasks, act per task.
            # ---------------------------------------------------------------
            t_poll_start = time.monotonic()
            while pending:
                for task_id in list(pending):
                    meta = pending[task_id]
                    try:
                        data = kie.get_task(task_id)
                    except Exception as exc:  # transient — retry next round
                        log.warning("get_task(%s) transient error: %s", task_id, exc)
                        continue
                    state = (data.get("state") or "").lower()
                    rs = session.get(RunSegment, meta["rs_id"])

                    if state == "success":
                        url = _parse_result_url(data)
                        if not url:
                            _skip_segment(rs, "success but no result url", session)
                            del pending[task_id]
                            continue
                        result_dst = os.path.join(r_dir, f"result_{meta['index']:04d}.mp4")
                        kie.download_result(url, result_dst)
                        rs.seedance_result_url = url
                        rs.local_result_path = result_dst
                        transition(rs, SegmentStatus.completed)
                        session.commit()
                        log.info("RunSegment idx %d completed", meta["index"])
                        del pending[task_id]
                    elif state == "fail":
                        msg = data.get("failMsg") or data.get("failCode") or "unknown"
                        _skip_segment(rs, f"Seedance failed: {msg}", session)
                        log.warning("task %s (seg %d) failed: %s — using original clip",
                                    task_id, meta["index"], msg)
                        del pending[task_id]
                    elif time.monotonic() > meta["deadline"]:
                        _skip_segment(
                            rs,
                            f"timed out after {RUN_SKIP_TIMEOUT_SEC:.0f}s (last state={state!r})",
                            session,
                        )
                        log.warning("task %s (seg %d) timed out — using original clip",
                                    task_id, meta["index"])
                        del pending[task_id]
                    # else: still waiting/queuing/generating → leave pending
                if pending:
                    time.sleep(RUN_POLL_INTERVAL_SEC)

            log.info(
                "run_id=%s poll_phase_total=%.1fs",
                run_id, time.monotonic() - t_poll_start,
            )

            # ---------------------------------------------------------------
            # Stitch — unless this is a delivery-only retry: when nothing was
            # (re)processed this run AND a final video already exists on disk,
            # reuse it and skip the expensive re-encode (e.g. a Retry after the
            # Drive upload timed out). We still pass through the `stitching`
            # state so the run's state-machine path stays valid.
            # ---------------------------------------------------------------
            t_stitch_start = time.monotonic()
            transition(run, RunStatus.stitching)
            session.commit()
            final_dst = os.path.join(r_dir, "final.mp4")
            reuse_final = (
                not did_submit
                and run.result_local_path
                and os.path.exists(run.result_local_path)
            )
            if reuse_final:
                final_dst = run.result_local_path
                log.info(
                    "No segments reprocessed and final video exists — skipping "
                    "re-stitch (delivery-only retry): %s", final_dst,
                )
            else:
                # Assemble clips in order. Non-completed swaps fall back to the
                # original (un-swapped) clip so the timeline stays intact.
                #
                # PARALLELISM: keep-segment cuts and fallback-original cuts are
                # independent ffmpeg calls with no shared state.  We run them
                # concurrently (up to STITCH_CUT_CONCURRENCY) using a thread pool,
                # then collect results in the original seg_defs order so the stitch
                # list is always correctly ordered.
                #
                # Each worker is a closure that either (a) cuts and returns a path,
                # or (b) returns the already-computed result path.  Futures are
                # stored in order so we can re-raise any exception on the main thread
                # without losing the ordering guarantee.

                def _cut_or_lookup(sd: SegmentDef) -> str:
                    """Return the clip path for this segment (cut if needed)."""
                    if sd.action == "keep":
                        keep_dst = os.path.join(c_dir, f"clip_{sd.index:04d}.mp4")
                        media_mod.cut_clip(source, sd.start_sec, sd.end_sec, keep_dst)
                        return keep_dst
                    # swap segment
                    rs = session.get(RunSegment, rs_map[sd.id].id)
                    if (
                        rs.status == SegmentStatus.completed
                        and rs.local_result_path
                        and os.path.exists(rs.local_result_path)
                    ):
                        return rs.local_result_path
                    # fallback: use original (un-swapped) clip
                    log.warning(
                        "Segment %d not swapped (status=%s) — using original clip",
                        sd.index, rs.status,
                    )
                    orig_dst = os.path.join(c_dir, f"orig_{sd.index:04d}.mp4")
                    media_mod.cut_clip(source, sd.start_sec, sd.end_sec, orig_dst)
                    return orig_dst

                # Determine which segments need ffmpeg work (cuts) vs which are
                # already-available result files.  Only segments that require a
                # cut_clip call benefit from concurrency; already-available results
                # are returned immediately.
                #
                # We submit ALL seg_defs (keeps + fallbacks) to the pool and track
                # them by insertion order using an ordered list of (index, future).
                ordered_futures: list[tuple[int, object]] = []  # (sd.index, Future)
                with ThreadPoolExecutor(max_workers=STITCH_CUT_CONCURRENCY) as pool:
                    for sd in seg_defs:
                        fut = pool.submit(_cut_or_lookup, sd)
                        ordered_futures.append((sd.index, fut))

                # Collect results in seg_defs order.  If any future raised an
                # exception, it will re-raise here, marking the run failed (outer
                # except clause handles that).
                clip_paths: list[str] = []
                for _idx, fut in ordered_futures:
                    clip_paths.append(fut.result())  # type: ignore[union-attr]

                log.info("Stitching %d clips → %s (%dx%d @ %.2ffps)",
                         len(clip_paths), final_dst, width, height, fps)
                audio_mode = run.audio_mode if run.audio_mode else "original"
                media_mod.stitch(
                    clip_paths, audio_source=source, dst=final_dst,
                    width=width, height=height, fps=fps,
                    audio_mode=audio_mode,
                )
                run.result_local_path = final_dst
            session.flush()

            log.info(
                "run_id=%s stitch_phase_total=%.1fs reuse=%s",
                run_id, time.monotonic() - t_stitch_start, reuse_final,
            )

            # ---------------------------------------------------------------
            # Deliver
            # ---------------------------------------------------------------
            t_deliver_start = time.monotonic()
            transition(run, RunStatus.delivering)
            session.commit()
            folder_id = run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id:
                if gdrive is None:
                    gdrive = _default_gdrive()
                result = _deliver_with_retry(gdrive, final_dst, folder_id, run_id)
                run.result_gdrive_file_id = result.get("id")
                log.info(
                    "run_id=%s deliver_phase=%.1fs gdrive_file=%s",
                    run_id, time.monotonic() - t_deliver_start, result.get("id"),
                )
                session.flush()

            transition(run, RunStatus.done)
            session.commit()
            t_total = time.monotonic() - t_run_start
            log.info(
                "process_run done: run_id=%s total=%.1fs",
                run_id, t_total,
            )

        except Exception as exc:
            if run.status not in (RunStatus.done, RunStatus.failed):
                try:
                    transition(run, RunStatus.failed)
                except Exception:
                    run.status = RunStatus.failed
            if not run.error_message:
                run.error_message = str(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise


def _skip_segment(rs: RunSegment, reason: str, session) -> None:
    """Mark a RunSegment failed (skipped); the run falls back to the original clip."""
    rs.error_message = reason
    if rs.status != SegmentStatus.failed:
        try:
            transition(rs, SegmentStatus.failed)
        except Exception:
            rs.status = SegmentStatus.failed
    session.commit()


def _submit_swap_segment(
    *,
    rs: RunSegment,
    sd: SegmentDef,
    run: Run,
    project: VideoProject,
    source: str,
    duration_sec: float,
    clip_dst: str,
    ref_urls: list[str],
    kie: KieClient,
    session,
) -> str:
    """Cut → upload → create_task for one swap segment. Returns the Seedance task id."""
    clip_start, clip_end = _clip_bounds(sd, duration_sec)

    media_mod.cut_clip(source, clip_start, clip_end, clip_dst)
    rs.local_clip_path = clip_dst
    transition(rs, SegmentStatus.uploading)
    session.commit()

    clip_url = kie.upload_file(clip_dst, "charswap/segments")
    rs.kie_upload_url = clip_url
    transition(rs, SegmentStatus.submitted)
    session.commit()

    effective_prompt = rs.prompt_override if rs.prompt_override else (run.prompt or "")

    if run.model == "gemini-omni":
        # Gemini takes the clip via video_list (trim <= 10s) and a fixed-set
        # output duration; segments longer than 10s are truncated to 10s.
        trim_end = round(min(clip_end - clip_start, 10.0), 2)
        task_id = kie.create_omni_task(
            prompt=effective_prompt,
            image_urls=ref_urls,
            video_url=clip_url,
            video_start=0,
            video_end=trim_end,
            resolution=_omni_resolution(run.resolution),
            aspect_ratio=_map_omni_aspect(
                project.aspect_ratio, project.width, project.height
            ),
            duration=_snap_omni_duration(clip_start, clip_end),
        )
    else:
        aspect = _map_aspect(project.aspect_ratio)
        clip_duration = _clamp_duration(clip_start, clip_end)
        task_id = kie.create_task(
            prompt=effective_prompt,
            reference_image_urls=ref_urls,
            reference_video_urls=[clip_url],
            resolution=run.resolution or settings.DEFAULT_RESOLUTION,
            aspect_ratio=aspect,
            duration=clip_duration,
        )
    rs.seedance_task_id = task_id
    transition(rs, SegmentStatus.generating)
    session.commit()
    return task_id
