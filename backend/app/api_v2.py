"""
v2 REST API router for re-skin.

Mounted at /api/v2 from app/main.py.

Adds VideoProject (video + segmentation) and Run (one character per project)
endpoints. v1 /api/jobs endpoints are left completely untouched.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import Run, RunSegment, SegmentDef, VideoProject
from .schemas_v2 import (
    NewSegmentDef,
    ProjectCreateResponse,
    ProjectListItem,
    ProjectResponse,
    ProjectUpdate,
    RunCreateResponse,
    RunListItem,
    RunResponse,
    RunSegmentResponse,
    SegmentDefResponse,
    SegmentsUpdateRequest,
)
from .state_machine import InvalidTransition, ProjectStatus, RunStatus, SegmentStatus, transition
from .storage import project_dir, project_source_path, run_dir
from .tasks import enqueue_analyze_project, enqueue_process_run

log = logging.getLogger(__name__)

router = APIRouter(tags=["v2"])

_VALID_RESOLUTIONS = {"480p", "720p", "1080p", "4k"}
_VALID_AUDIO_MODES = {"original", "seedance"}
_VALID_MODELS = {"seedance", "gemini-omni"}
# Run states that mean a worker may still be touching the run's files — block
# deletion while in any of these (delete would race the worker / rmtree live files).
_ACTIVE_RUN_STATUSES = {
    RunStatus.queued,
    RunStatus.processing,
    RunStatus.stitching,
    RunStatus.delivering,
}
# Allowed resolutions per model (each backend supports a different set).
_MODEL_RESOLUTIONS = {
    "seedance": {"480p", "720p", "1080p"},
    "gemini-omni": {"720p", "1080p", "4k"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_project_or_404(project_id: str, db: Session) -> VideoProject:
    project = db.get(VideoProject, project_id)
    if project is None:
        raise HTTPException(
            status_code=404, detail=f"Project {project_id!r} not found"
        )
    return project


def _get_run_or_404(run_id: str, db: Session) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return run


def _safe_filename(name: str) -> str:
    """Sanitize an uploaded filename: keep only the basename, strip leading dots."""
    base = os.path.basename(name.replace("\\", "/"))
    base = base.lstrip(".")
    return base or "upload"


def _save_upload(upload: UploadFile, dest: str, max_bytes: int | None = None) -> None:
    """Write an UploadFile to *dest* using streaming 1-MiB chunks.

    Raises HTTPException(413) when *max_bytes* is set and the upload exceeds it.
    The partial file is removed before raising.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _CHUNK = 1 << 20  # 1 MiB
    written = 0
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = upload.file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    fh.close()
                    os.unlink(dest)
                    limit_mb = max_bytes >> 20
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload too large: limit is {limit_mb} MiB",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception:
        if os.path.exists(dest):
            os.unlink(dest)
        raise


def _normalize_partition(segments: list, duration: float, db: Session) -> None:
    """Normalize *segments* into a contiguous partition of [0, duration].

    Forgiving cursor-walk: ends are the authoritative boundaries; starts are
    derived from the running cursor. A segment whose duration collapses to <= 0
    (e.g. its neighbour was extended over it) is DROPPED (deleted) rather than
    rejected — so "shrink a segment to zero" behaves like deleting it, and the
    partition stays contiguous. The first kept segment starts at 0 and the last
    is extended to *duration* for full coverage. Indices are reassigned.

    Raises HTTPException(400) only if every segment would be dropped.
    """
    if not segments:
        return

    ordered = sorted(segments, key=lambda s: s.start_sec)
    EPS = 1e-6
    cursor = 0.0
    kept: list = []
    for seg in ordered:
        seg.start_sec = cursor
        end = min(seg.end_sec, duration)
        if end - cursor > EPS:
            seg.end_sec = end
            kept.append(seg)
            cursor = end
        else:
            # Collapsed (zero/negative duration) → drop it.
            db.delete(seg)

    if not kept:
        raise HTTPException(
            status_code=400,
            detail="No segments with positive duration remain after edits.",
        )

    # Ensure full coverage of [0, duration].
    kept[-1].end_sec = duration
    for i, seg in enumerate(kept):
        seg.index = i


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/projects",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectCreateResponse,
)
def create_project(
    video_file: Optional[UploadFile] = File(None),
    gdrive_link: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> ProjectCreateResponse:
    """Create a new VideoProject.

    Exactly one of *video_file* or *gdrive_link* must be provided.
    Analysis is enqueued immediately (ffprobe + segment proposal).
    """
    has_file = video_file is not None and video_file.filename
    has_link = gdrive_link is not None and gdrive_link.strip()

    if not has_file and not has_link:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of video_file or gdrive_link",
        )
    if has_file and has_link:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of video_file or gdrive_link, not both",
        )

    project_id = str(uuid.uuid4())

    max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
    if has_file:
        filename = _safe_filename(video_file.filename or "source.mp4")
        ext = os.path.splitext(filename)[-1].lstrip(".") or "mp4"
        src_path = project_source_path(project_id, ext)
        _save_upload(video_file, src_path, max_bytes=max_bytes)

        project = VideoProject(
            id=project_id,
            source_type="upload",
            source_ref=filename,
            source_local_path=src_path,
            status=ProjectStatus.created,
        )
    else:
        link = gdrive_link.strip()
        project = VideoProject(
            id=project_id,
            source_type="gdrive",
            source_ref=link,
            status=ProjectStatus.created,
        )

    db.add(project)
    db.commit()

    # Enqueue analysis — import at module level so monkeypatch targets app.api_v2.*
    enqueue_analyze_project(project_id)

    log.info("Created project %s source_type=%s", project_id, project.source_type)
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)
    return ProjectCreateResponse(project_id=project_id, status=status_val)


@router.get("/projects", response_model=list[ProjectListItem])
def list_projects(db: Session = Depends(get_db)) -> list:
    """Return all projects, newest first."""
    projects = (
        db.execute(select(VideoProject).order_by(desc(VideoProject.created_at)))
        .scalars()
        .all()
    )
    return [ProjectListItem.model_validate(p) for p in projects]


@router.get("/projects/{pid}", response_model=ProjectResponse)
def get_project(pid: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """Return full project details."""
    project = _get_project_or_404(pid, db)
    return ProjectResponse.model_validate(project)


@router.patch("/projects/{pid}", response_model=ProjectResponse)
def update_project(
    pid: str, body: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """Update editable project settings (currently just the display name)."""
    project = _get_project_or_404(pid, db)
    if body.name is not None:
        name = body.name.strip()
        project.name = name[:255] or None
    db.commit()
    db.refresh(project)
    log.info("Updated project %s name=%r", pid, project.name)
    return ProjectResponse.model_validate(project)


@router.delete("/projects/{pid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(pid: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a project: DB rows (cascades to segments/runs) + disk.

    Blocked (409) while the project is analyzing or any of its runs is active, so
    we never remove files a worker is still using.
    """
    project = _get_project_or_404(pid, db)
    if project.status == ProjectStatus.analyzing:
        raise HTTPException(
            status_code=409, detail="Cannot delete a project while it is analyzing"
        )
    active = db.execute(
        select(Run.id).where(
            Run.project_id == pid, Run.status.in_(_ACTIVE_RUN_STATUSES)
        )
    ).first()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a project while one of its runs is active",
        )

    db.delete(project)  # cascades to SegmentDef / Run / RunSegment
    db.commit()
    shutil.rmtree(project_dir(pid), ignore_errors=True)
    log.info("Deleted project %s (db + disk)", pid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/projects/{pid}/segments", response_model=list[SegmentDefResponse])
def get_project_segments(pid: str, db: Session = Depends(get_db)) -> list:
    """Return all SegmentDefs for a project, ordered by index."""
    _get_project_or_404(pid, db)
    segments = (
        db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == pid)
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    )
    return [SegmentDefResponse.model_validate(s) for s in segments]


@router.patch("/projects/{pid}/segments", response_model=list[SegmentDefResponse])
def update_project_segments(
    pid: str,
    body: SegmentsUpdateRequest,
    db: Session = Depends(get_db),
) -> list:
    """Edit SegmentDefs while project is in ready status."""
    project = _get_project_or_404(pid, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit segments: project status is {project.status!r}, "
                "expected 'ready'"
            ),
        )

    # Load all current segments into a dict by id
    segments: dict[str, SegmentDef] = {
        s.id: s
        for s in db.execute(
            select(SegmentDef).where(SegmentDef.project_id == pid)
        )
        .scalars()
        .all()
    }

    # Apply updates
    for upd in body.updates:
        seg = segments.get(upd.id)
        if seg is None:
            raise HTTPException(
                status_code=404, detail=f"SegmentDef {upd.id!r} not found"
            )
        for field, val in upd.model_dump(exclude={"id"}, exclude_none=True).items():
            setattr(seg, field, val)

    # Apply deletes
    for seg_id in body.deletes:
        seg = segments.pop(seg_id, None)
        if seg is None:
            raise HTTPException(
                status_code=404, detail=f"SegmentDef {seg_id!r} not found"
            )
        db.delete(seg)

    # Apply creates
    for new_seg in body.creates:
        seg = SegmentDef(
            id=str(uuid.uuid4()),
            project_id=pid,
            index=0,  # will be renumbered
            **new_seg.model_dump(),
        )
        db.add(seg)
        db.flush()
        segments[seg.id] = seg

    # Normalize to a contiguous partition [0, duration]
    remaining = list(segments.values())
    duration = project.duration_sec
    if duration is None:
        # Fallback: just renumber without partition enforcement
        ordered = sorted(remaining, key=lambda s: s.start_sec)
        for i, seg in enumerate(ordered):
            seg.index = i
    else:
        _normalize_partition(remaining, duration, db)

    db.commit()

    updated = (
        db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == pid)
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    )
    return [SegmentDefResponse.model_validate(s) for s in updated]


@router.get("/projects/{pid}/frame")
def get_project_frame(
    pid: str,
    t: float = Query(0.0, description="Timestamp in seconds"),
    db: Session = Depends(get_db),
) -> Response:
    """Extract a single JPEG frame from the project's source video at time *t* seconds.

    Cached as ``frames/frame_<t_ms>.jpg`` inside the project directory.
    Returns 404 if the project or its source video file is not found on disk.
    """
    import subprocess
    import tempfile

    project = _get_project_or_404(pid, db)

    src = project.source_local_path
    if not src or not os.path.exists(src):
        raise HTTPException(
            status_code=404, detail="Source video not available on disk"
        )

    pdir = project_dir(pid)
    frames_dir = os.path.join(pdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    t_ms = int(t * 1000)
    cache_path = os.path.join(frames_dir, f"frame_{t_ms}.jpg")

    if not os.path.exists(cache_path):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=frames_dir)
        os.close(tmp_fd)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(t),
                    "-i", src,
                    "-frames:v", "1",
                    "-q:v", "5",
                    "-f", "image2",
                    tmp_path,
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=500,
                    detail="ffmpeg failed to extract frame",
                )
            os.rename(tmp_path, cache_path)
        except subprocess.TimeoutExpired:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail="ffmpeg timed out")

    with open(cache_path, "rb") as fh:
        data = fh.read()

    return Response(content=data, media_type="image/jpeg")


@router.get("/projects/{pid}/source")
def get_project_source(pid: str, db: Session = Depends(get_db)):
    """Stream the project's original source video (for in-page review/seeking).

    FileResponse handles HTTP Range requests, so the <video> element can scrub.
    404 if the project or its source file is not on disk.
    """
    project = _get_project_or_404(pid, db)
    src = project.source_local_path
    if not src or not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Source video not available on disk")
    return FileResponse(src, media_type="video/mp4", filename="source.mp4")


@router.get("/projects/{pid}/runs", response_model=list[RunListItem])
def list_project_runs(pid: str, db: Session = Depends(get_db)) -> list:
    """Return all runs for a project."""
    _get_project_or_404(pid, db)
    runs = (
        db.execute(
            select(Run)
            .where(Run.project_id == pid)
            .order_by(desc(Run.created_at))
        )
        .scalars()
        .all()
    )
    items = []
    for run in runs:
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        items.append(
            RunListItem(
                id=run.id,
                name=run.name,
                status=status_val,
                created_at=run.created_at,
                result_available=(
                    status_val == "done"
                    and bool(run.result_local_path)
                    and os.path.exists(run.result_local_path)
                ),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{pid}/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=RunCreateResponse,
)
def create_run(
    pid: str,
    prompt: str = Form(...),
    name: Optional[str] = Form(None),
    model: str = Form("seedance"),
    resolution: str = Form(settings.DEFAULT_RESOLUTION),
    audio_mode: str = Form("original"),
    gdrive_folder_id: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    segment_prompts: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunCreateResponse:
    """Create a new Run under a project (one character attempt).

    The project must be in *ready* status. Exactly one character prompt is
    required. Reference images (files + URLs) are capped at MAX_REFERENCE_IMAGES.

    *segment_prompts* is an optional JSON object ``{segment_def_id: extra_text}``
    of per-segment additions: the extra text is appended to the run prompt for
    that swap segment on the very first run (blank/absent → uses the run prompt).
    """
    project = _get_project_or_404(pid, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot create run: project status is {project.status!r}, "
                "expected 'ready'"
            ),
        )

    # Validate model
    if model not in _VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {sorted(_VALID_MODELS)}",
        )

    # Validate resolution (must be allowed for the chosen model)
    allowed_res = _MODEL_RESOLUTIONS[model]
    if resolution not in allowed_res:
        raise HTTPException(
            status_code=400,
            detail=(
                f"resolution {resolution!r} not allowed for model {model!r}; "
                f"choose one of {sorted(allowed_res)}"
            ),
        )

    # Validate audio_mode
    if audio_mode not in _VALID_AUDIO_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"audio_mode must be one of {sorted(_VALID_AUDIO_MODES)}",
        )

    # Validate reference image count
    ref_files = reference_files or []
    ref_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    total_refs = len(ref_files) + len(ref_urls)
    if total_refs > settings.MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many reference images: got {total_refs}, "
                f"max {settings.MAX_REFERENCE_IMAGES}"
            ),
        )

    run_id = str(uuid.uuid4())

    resolved_folder_id = gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID or None

    run = Run(
        id=run_id,
        project_id=pid,
        name=name,
        prompt=prompt,
        model=model,
        resolution=resolution,
        audio_mode=audio_mode,
        gdrive_folder_id=resolved_folder_id,
        status=RunStatus.created,
        reference_image_urls=[],
    )
    db.add(run)
    db.flush()  # persist id before saving reference files

    # Save reference files into project/run dir
    max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
    saved_ref_paths: list[str] = []
    refs_dir = os.path.join(project_dir(pid), "runs", run_id, "references")
    os.makedirs(refs_dir, exist_ok=True)
    for rf in ref_files:
        safe = _safe_filename(rf.filename or f"ref_{len(saved_ref_paths)}.jpg")
        dest = os.path.join(refs_dir, safe)
        _save_upload(rf, dest, max_bytes=max_bytes)
        saved_ref_paths.append(dest)

    run.reference_image_urls = saved_ref_paths + list(ref_urls)

    # Optional per-segment prompt additions: pre-create RunSegments carrying a
    # prompt_override (= run prompt + the extra text) so the FIRST run already
    # submits tailored prompts. process_run is idempotent and reuses these.
    if segment_prompts:
        try:
            seg_map = json.loads(segment_prompts)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="segment_prompts must be valid JSON"
            ) from exc
        if isinstance(seg_map, dict) and seg_map:
            swap_defs = {
                s.id: s
                for s in db.execute(
                    select(SegmentDef).where(
                        SegmentDef.project_id == pid, SegmentDef.action == "swap"
                    )
                )
                .scalars()
                .all()
            }
            for sd_id, extra in seg_map.items():
                text = extra.strip() if isinstance(extra, str) else ""
                sd = swap_defs.get(sd_id)
                if text and sd is not None:
                    db.add(
                        RunSegment(
                            run_id=run_id,
                            segment_def_id=sd_id,
                            index=sd.index,
                            status=SegmentStatus.pending,
                            prompt_override=f"{prompt.rstrip()}\n{text}",
                        )
                    )

    # Transition created → queued
    try:
        transition(run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()

    enqueue_process_run(run_id)

    log.info("Created run %s for project %s", run_id, pid)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    return RunCreateResponse(run_id=run_id, status=status_val)


@router.get("/runs/{rid}", response_model=RunResponse)
def get_run(rid: str, db: Session = Depends(get_db)) -> RunResponse:
    """Return full run details."""
    run = _get_run_or_404(rid, db)
    return RunResponse.model_validate(run)


@router.delete("/runs/{rid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(rid: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a run: DB rows (cascades to RunSegments) + disk.

    Blocked (409) while the run is active (queued/processing/stitching/delivering).
    """
    run = _get_run_or_404(rid, db)
    if run.status in _ACTIVE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a run while it is {run.status.value!r}",
        )
    project_id = run.project_id
    db.delete(run)  # cascades to RunSegment
    db.commit()
    shutil.rmtree(run_dir(rid, project_id), ignore_errors=True)
    log.info("Deleted run %s (db + disk)", rid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/runs/{rid}/segments", response_model=list[RunSegmentResponse])
def get_run_segments(rid: str, db: Session = Depends(get_db)) -> list:
    """Return RunSegments for a run (progress display)."""
    _get_run_or_404(rid, db)
    segments = (
        db.execute(
            select(RunSegment)
            .where(RunSegment.run_id == rid)
            .order_by(RunSegment.index)
        )
        .scalars()
        .all()
    )
    return [RunSegmentResponse.model_validate(s) for s in segments]


@router.get("/runs/{rid}/result/info")
def get_run_result_info(rid: str, db: Session = Depends(get_db)) -> dict:
    """Return result metadata without downloading the file."""
    run = _get_run_or_404(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Run not done yet: status is {run.status!r}",
        )
    gdrive_link = (
        f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
        if run.result_gdrive_file_id
        else None
    )
    return {
        "run_id": rid,
        "result_local_path": run.result_local_path,
        "result_gdrive_file_id": run.result_gdrive_file_id,
        "result_gdrive_link": gdrive_link,
    }


@router.get("/runs/{rid}/result")
def download_run_result(rid: str, db: Session = Depends(get_db)):
    """Download the final video file for a run."""
    run = _get_run_or_404(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Run not done yet: status is {run.status!r}",
        )
    path = run.result_local_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    filename = os.path.basename(path)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "X-GDrive-File-Id": run.result_gdrive_file_id or "",
        },
    )


def _get_run_segment_or_404(rsid: str, run_id: str, db: Session) -> RunSegment:
    rs = db.get(RunSegment, rsid)
    if rs is None or rs.run_id != run_id:
        raise HTTPException(
            status_code=404, detail=f"RunSegment {rsid!r} not found in run {run_id!r}"
        )
    return rs


def _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls):
    """Set a RunSegment's prompt/reference overrides (shared by PATCH and rerun).

    Empty prompt clears the prompt override; empty reference set clears the ref
    override (both fall back to run-level values).
    """
    ref_files = reference_files or []
    parsed_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    total_refs = len(ref_files) + len(parsed_urls)
    if total_refs > settings.MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many reference images: got {total_refs}, "
                f"max {settings.MAX_REFERENCE_IMAGES}"
            ),
        )

    rs.prompt_override = (prompt.strip() or None) if prompt is not None else None

    if total_refs == 0:
        rs.reference_image_urls_override = None
    else:
        refs_dir = os.path.join(
            project_dir(run.project_id), "runs", rid, "segment_refs", rsid
        )
        os.makedirs(refs_dir, exist_ok=True)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
        saved_paths: list[str] = []
        for rf in ref_files:
            safe = _safe_filename(rf.filename or f"ref_{len(saved_paths)}.jpg")
            dest = os.path.join(refs_dir, safe)
            _save_upload(rf, dest, max_bytes=max_bytes)
            saved_paths.append(dest)
        rs.reference_image_urls_override = saved_paths + parsed_urls


@router.patch(
    "/runs/{rid}/segments/{rsid}",
    response_model=RunSegmentResponse,
)
def patch_run_segment(
    rid: str,
    rsid: str,
    prompt: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunSegmentResponse:
    """Override prompt and/or reference images for an individual RunSegment.

    Only allowed when the run is in done or failed status. Empty prompt clears the
    override (falls back to run-level prompt). Empty reference list clears the
    override (falls back to run-level references).
    """
    run = _get_run_or_404(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in ("done", "failed"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit segment: run status is {run.status!r}; "
                "expected 'done' or 'failed'"
            ),
        )

    rs = _get_run_segment_or_404(rsid, rid, db)

    _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls)

    db.commit()
    db.refresh(rs)
    log.info("Patched RunSegment %s (run %s): prompt_override=%r refs_override=%r",
             rsid, rid, rs.prompt_override, rs.reference_image_urls_override)
    return RunSegmentResponse.model_validate(rs)


@router.post("/runs/{rid}/segments/{rsid}/rerun", response_model=RunResponse)
def rerun_segment(
    rid: str,
    rsid: str,
    prompt: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunResponse:
    """Apply the (optional) prompt/reference override, reset one RunSegment to
    pending, and re-queue the run — atomically, so the re-run always uses the
    prompt sent with THIS request (no separate save needed).

    The run must be in done or failed status. Other completed RunSegments are
    skipped by process_run (resumability); only this segment is reprocessed and
    the final video is re-stitched. If no prompt field is sent at all, the
    existing override is left untouched.
    """
    run = _get_run_or_404(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in ("done", "failed"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot re-run segment: run status is {run.status!r}; "
                "expected 'done' or 'failed'"
            ),
        )

    rs = _get_run_segment_or_404(rsid, rid, db)

    # Apply the prompt/reference sent with this request so the re-run uses
    # exactly what's on screen. Only when a prompt field is present (a form was
    # sent) — otherwise leave any previously-saved override untouched.
    if prompt is not None or reference_files or reference_urls:
        _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls)

    # Reset this RunSegment to pending
    rs.status = SegmentStatus.pending
    rs.error_message = None
    rs.seedance_task_id = None
    rs.seedance_result_url = None
    rs.local_result_path = None

    # Transition run → queued (done→queued or failed→queued both allowed now)
    try:
        transition(run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    enqueue_process_run(rid)

    log.info("Rerunning segment %s in run %s", rsid, rid)
    return RunResponse.model_validate(run)


@router.post("/runs/{rid}/retry", response_model=RunResponse)
def retry_run(rid: str, db: Session = Depends(get_db)) -> RunResponse:
    """Re-enqueue a failed or stuck run.

    Accepted statuses
    -----------------
    - ``failed``     — normal retry after a processing error.
    - ``queued``     — orphaned run stuck in queue with no live RQ job.
    - ``processing`` — orphaned run whose worker crashed mid-flight.
    - ``stitching``  — orphaned run whose worker crashed during stitch.
    - ``delivering`` — orphaned run whose worker crashed during GDrive upload.

    Safety note
    -----------
    This endpoint does NOT check whether the run is genuinely idle vs. actively
    being worked on by another worker right now.  Do NOT call it on a run that
    is legitimately in-progress: re-enqueuing a live run would double-process it.
    The startup reconciliation routine (``recovery.py``) uses a queue-idle guard
    to avoid this; a human operator must exercise the same judgement when calling
    this endpoint manually.

    For automatic safe recovery, rely on the startup reconciliation.  Use this
    endpoint for manual intervention only.
    """
    run = _get_run_or_404(rid, db)

    _RETRYABLE = {
        RunStatus.failed,
        RunStatus.queued,
        RunStatus.processing,
        RunStatus.stitching,
        RunStatus.delivering,
    }

    if run.status not in _RETRYABLE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot retry: run status is {run.status!r}. "
                f"Retryable statuses: {sorted(s.value for s in _RETRYABLE)}"
            ),
        )

    original_status = run.status

    # For already-queued runs, skip the transition (already queued).
    if run.status != RunStatus.queued:
        try:
            transition(run, RunStatus.queued)
        except InvalidTransition:
            # stitching/delivering → queued lacks a direct edge; go via failed.
            try:
                transition(run, RunStatus.failed)
                transition(run, RunStatus.queued)
            except InvalidTransition as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    enqueue_process_run(rid)

    log.info("Retrying run %s (was %s)", rid, original_status)
    return RunResponse.model_validate(run)


def _copy_reference_files(items: list, dest_dir: str) -> list:
    """Clone a run's reference list for a copied run.

    Local file paths are copied into *dest_dir* (so the copy is self-contained and
    survives deleting the source run); http(s) URLs and missing/odd entries are
    passed through unchanged.
    """
    out: list = []
    made = False
    for ref in items or []:
        if isinstance(ref, str) and (ref.startswith("http://") or ref.startswith("https://")):
            out.append(ref)
            continue
        if isinstance(ref, str) and os.path.exists(ref):
            if not made:
                os.makedirs(dest_dir, exist_ok=True)
                made = True
            dst = os.path.join(dest_dir, os.path.basename(ref))
            shutil.copy2(ref, dst)
            out.append(dst)
        else:
            out.append(ref)  # best-effort: keep whatever it was
    return out


@router.post(
    "/runs/{rid}/copy",
    status_code=status.HTTP_201_CREATED,
    response_model=RunCreateResponse,
)
def copy_run(
    rid: str,
    resolution: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunCreateResponse:
    """Duplicate a run — optionally at a new resolution and/or with a new
    reference photo — and enqueue it.

    Clones everything that defines the result — prompt, model, audio mode, Drive
    folder, and any per-segment prompt overrides. Two things can be changed:

    * **resolution** — defaults to the source run's resolution if omitted; pass a
      different one to promote a 480p test to production.
    * **reference photo** — pass *reference_files* and/or *reference_urls* to swap
      the character to a new face of the same type. This is the "project as a
      template" workflow: tune a run once, then re-run it with a new person's
      photo. When new references are supplied they REPLACE the photo everywhere —
      both the run-level references AND any per-segment reference overrides are
      dropped so every swap segment uses the new photo (per-segment *prompt*
      tweaks are still carried over). When omitted, the source run's references
      (run-level and per-segment) are cloned unchanged.
    """
    src = _get_run_or_404(rid, db)
    project = _get_project_or_404(src.project_id, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot copy run: project status is {project.status!r}, expected 'ready'",
        )

    # Resolution: default to the source run's when not changing it.
    resolution = (resolution or src.resolution).strip()
    allowed = _MODEL_RESOLUTIONS[src.model]
    if resolution not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"resolution {resolution!r} not allowed for model {src.model!r}; "
                f"choose one of {sorted(allowed)}"
            ),
        )

    # New reference photo (optional). When provided it replaces the photo
    # everywhere; when absent we clone the source run's references.
    new_ref_files = reference_files or []
    new_ref_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    has_new_refs = bool(new_ref_files or new_ref_urls)
    if has_new_refs:
        total_refs = len(new_ref_files) + len(new_ref_urls)
        if total_refs > settings.MAX_REFERENCE_IMAGES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Too many reference images: got {total_refs}, "
                    f"max {settings.MAX_REFERENCE_IMAGES}"
                ),
            )

    new_id = str(uuid.uuid4())
    if name and name.strip():
        new_name = name.strip()
    else:
        suffix = "new ref" if has_new_refs else resolution
        new_name = f"{src.name or 'run'} · {suffix}"
    new_name = new_name[:255]
    new_run = Run(
        id=new_id,
        project_id=src.project_id,
        name=new_name,
        prompt=src.prompt,
        model=src.model,
        resolution=resolution,
        audio_mode=src.audio_mode,
        gdrive_folder_id=src.gdrive_folder_id,
        status=RunStatus.created,
        reference_image_urls=[],
    )
    db.add(new_run)
    db.flush()

    refs_dir = os.path.join(project_dir(src.project_id), "runs", new_id, "references")
    if has_new_refs:
        # Use the supplied photo as the new run-level reference.
        _max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
        saved_ref_paths: list[str] = []
        if new_ref_files:
            os.makedirs(refs_dir, exist_ok=True)
            for rf in new_ref_files:
                safe = _safe_filename(rf.filename or f"ref_{len(saved_ref_paths)}.jpg")
                dest = os.path.join(refs_dir, safe)
                _save_upload(rf, dest, max_bytes=_max_bytes)
                saved_ref_paths.append(dest)
        new_run.reference_image_urls = saved_ref_paths + new_ref_urls
    else:
        # Clone run-level reference images into the new run's own dir.
        new_run.reference_image_urls = _copy_reference_files(
            list(src.reference_image_urls or []), refs_dir
        )

    # Clone per-segment overrides so the copy reproduces the same tuned result.
    # process_run is idempotent: it reuses these RunSegments (by segment_def_id)
    # instead of creating fresh ones, so the overrides take effect.
    #
    # When a new photo is supplied we DROP per-segment reference overrides so the
    # new run-level photo is used in every segment (the "replace everywhere"
    # behaviour); per-segment prompt tweaks are still carried over.
    for src_rs in src.run_segments:
        if has_new_refs:
            if not src_rs.prompt_override:
                # Only a photo override here → dropped; process_run will create a
                # fresh pending segment that inherits the new run-level photo.
                continue
            override_refs = None
        else:
            if not (src_rs.prompt_override or src_rs.reference_image_urls_override):
                continue
            override_refs = None
            if src_rs.reference_image_urls_override:
                new_rs_id_refs = str(uuid.uuid4())
                seg_dir = os.path.join(
                    project_dir(src.project_id), "runs", new_id, "segment_refs",
                    new_rs_id_refs,
                )
                override_refs = _copy_reference_files(
                    list(src_rs.reference_image_urls_override), seg_dir
                )
        db.add(
            RunSegment(
                id=str(uuid.uuid4()),
                run_id=new_id,
                segment_def_id=src_rs.segment_def_id,
                index=src_rs.index,
                status=SegmentStatus.pending,
                prompt_override=src_rs.prompt_override,
                reference_image_urls_override=override_refs,
            )
        )

    try:
        transition(new_run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    enqueue_process_run(new_id)

    log.info(
        "Copied run %s → %s (resolution=%s, new_refs=%s)",
        rid, new_id, resolution, has_new_refs,
    )
    status_val = new_run.status.value if hasattr(new_run.status, "value") else str(new_run.status)
    return RunCreateResponse(run_id=new_id, status=status_val)
