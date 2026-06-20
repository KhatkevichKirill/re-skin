"""
REST API router for re-skin.

Mounted at /api from app/main.py.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import Job, Segment
from .schemas import (
    JobCreateResponse,
    JobListItem,
    JobResponse,
    SegmentResponse,
    SegmentsUpdateRequest,
)
from .state_machine import InvalidTransition, JobStatus, SegmentStatus, transition
from .storage import job_dir, source_path
from .tasks import enqueue_analyze, enqueue_process

log = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])

_VALID_RESOLUTIONS = {"480p", "720p", "1080p"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_job_or_404(job_id: str, db: Session) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


def _save_upload(upload: UploadFile, dest: str) -> None:
    """Write an UploadFile to a local path."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(upload.file.read())


def _renumber_segments(segments: list) -> None:
    """Re-assign index values sorted by start_sec (in-place)."""
    ordered = sorted(segments, key=lambda s: s.start_sec)
    for i, seg in enumerate(ordered):
        seg.index = i


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/jobs", status_code=status.HTTP_201_CREATED, response_model=JobCreateResponse)
def create_job(
    prompt: str = Form(...),
    video_file: Optional[UploadFile] = File(None),
    gdrive_link: Optional[str] = Form(None),
    resolution: str = Form(settings.DEFAULT_RESOLUTION),
    gdrive_folder_id: Optional[str] = Form(None),
    reference_files: Optional[List[UploadFile]] = File(None),
    reference_urls: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> JobCreateResponse:
    """Create a new re-skin job.

    Exactly one of *video_file* or *gdrive_link* must be provided.
    """
    # Validate source: exactly one of video_file or gdrive_link
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

    # Validate resolution
    if resolution not in _VALID_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"resolution must be one of {sorted(_VALID_RESOLUTIONS)}",
        )

    # Validate reference image count
    # reference_urls is a comma-separated string (handles HTMX / multipart limitations)
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

    job_id = str(uuid.uuid4())
    jdir = job_dir(job_id)

    # Resolve gdrive_folder_id default
    resolved_folder_id = gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID or None

    # Handle source
    if has_file:
        filename = video_file.filename or "source.mp4"
        ext = os.path.splitext(filename)[-1].lstrip(".") or "mp4"
        src_path = source_path(job_id, ext)
        _save_upload(video_file, src_path)

        job = Job(
            id=job_id,
            source_type="upload",
            source_ref=filename,
            source_local_path=src_path,
            default_prompt=prompt,
            resolution=resolution,
            gdrive_folder_id=resolved_folder_id,
            status=JobStatus.created,
        )
    else:
        link = gdrive_link.strip()
        job = Job(
            id=job_id,
            source_type="gdrive",
            source_ref=link,
            default_prompt=prompt,
            resolution=resolution,
            gdrive_folder_id=resolved_folder_id,
            status=JobStatus.created,
        )

    db.add(job)
    db.flush()  # persist id before saving reference files

    # Save reference files into the job dir
    saved_ref_paths: list[str] = []
    refs_dir = os.path.join(jdir, "references")
    os.makedirs(refs_dir, exist_ok=True)
    for rf in ref_files:
        dest = os.path.join(refs_dir, rf.filename or f"ref_{len(saved_ref_paths)}.jpg")
        _save_upload(rf, dest)
        saved_ref_paths.append(dest)

    # Combine saved paths + provided URLs
    all_refs = saved_ref_paths + list(ref_urls)
    job.default_reference_image_urls = all_refs

    db.commit()

    # Enqueue analysis (module-level names so tests can monkeypatch app.api.enqueue_analyze)
    enqueue_analyze(job_id)

    log.info("Created job %s, source_type=%s", job_id, job.source_type)
    status_val = job.status.value if hasattr(job.status, "value") else str(job.status)
    return JobCreateResponse(job_id=job_id, status=status_val)


@router.get("/jobs", response_model=list[JobListItem])
def list_jobs(db: Session = Depends(get_db)) -> list:
    """Return all jobs, newest first."""
    from sqlalchemy import select, desc

    jobs = db.execute(select(Job).order_by(desc(Job.created_at))).scalars().all()
    return [JobListItem.model_validate(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    """Return full job details."""
    job = _get_job_or_404(job_id, db)
    return JobResponse.model_validate(job)


@router.get("/jobs/{job_id}/segments", response_model=list[SegmentResponse])
def get_segments(job_id: str, db: Session = Depends(get_db)) -> list:
    """Return all segments for a job, ordered by index."""
    from sqlalchemy import select

    _get_job_or_404(job_id, db)
    segments = (
        db.execute(select(Segment).where(Segment.job_id == job_id).order_by(Segment.index))
        .scalars()
        .all()
    )
    return [SegmentResponse.model_validate(s) for s in segments]


@router.patch("/jobs/{job_id}/segments", response_model=list[SegmentResponse])
def update_segments(
    job_id: str,
    body: SegmentsUpdateRequest,
    db: Session = Depends(get_db),
) -> list:
    """Edit segments while job is in review status."""
    from sqlalchemy import select

    job = _get_job_or_404(job_id, db)
    if job.status != JobStatus.review:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot edit segments: job status is {job.status!r}, expected 'review'",
        )

    # Load all current segments into a dict by id
    segments: dict[str, Segment] = {
        s.id: s
        for s in db.execute(select(Segment).where(Segment.job_id == job_id)).scalars().all()
    }

    # Apply updates
    for upd in body.updates:
        seg = segments.get(upd.id)
        if seg is None:
            raise HTTPException(status_code=404, detail=f"Segment {upd.id!r} not found")
        for field, val in upd.model_dump(exclude={"id"}, exclude_none=True).items():
            setattr(seg, field, val)

    # Apply deletes
    for seg_id in body.deletes:
        seg = segments.pop(seg_id, None)
        if seg is None:
            raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")
        db.delete(seg)

    # Apply creates
    for new_seg in body.creates:
        seg = Segment(
            id=str(uuid.uuid4()),
            job_id=job_id,
            index=0,  # will be renumbered
            status=SegmentStatus.pending,
            **new_seg.model_dump(),
        )
        db.add(seg)
        db.flush()
        segments[seg.id] = seg

    # Re-number by start_sec
    remaining = list(segments.values())
    _renumber_segments(remaining)

    db.commit()

    # Return fresh ordered list
    updated = (
        db.execute(select(Segment).where(Segment.job_id == job_id).order_by(Segment.index))
        .scalars()
        .all()
    )
    return [SegmentResponse.model_validate(s) for s in updated]


@router.post("/jobs/{job_id}/submit", response_model=JobResponse)
def submit_job(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    """Submit a reviewed job for processing."""
    job = _get_job_or_404(job_id, db)
    if job.status != JobStatus.review:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit: job status is {job.status!r}, expected 'review'",
        )
    try:
        transition(job, JobStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    enqueue_process(job_id)

    log.info("Submitted job %s for processing", job_id)
    return JobResponse.model_validate(job)


@router.get("/jobs/{job_id}/result/info")
def get_result_info(job_id: str, db: Session = Depends(get_db)) -> dict:
    """Return result metadata (paths, gdrive link) without downloading the file."""
    job = _get_job_or_404(job_id, db)
    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job not done yet: status is {job.status!r}",
        )
    gdrive_link = (
        f"https://drive.google.com/file/d/{job.result_gdrive_file_id}/view"
        if job.result_gdrive_file_id
        else None
    )
    return {
        "job_id": job_id,
        "result_local_path": job.result_local_path,
        "result_gdrive_file_id": job.result_gdrive_file_id,
        "result_gdrive_link": gdrive_link,
    }


@router.get("/jobs/{job_id}/result")
def download_result(job_id: str, db: Session = Depends(get_db)):
    """Download the final video file."""
    job = _get_job_or_404(job_id, db)
    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job not done yet: status is {job.status!r}",
        )
    path = job.result_local_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    filename = os.path.basename(path)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "X-GDrive-File-Id": job.result_gdrive_file_id or "",
        },
    )


@router.post("/jobs/{job_id}/retry", response_model=JobResponse)
def retry_job(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    """Re-enqueue a failed job."""
    job = _get_job_or_404(job_id, db)
    if job.status != JobStatus.failed:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot retry: job status is {job.status!r}, expected 'failed'",
        )
    try:
        transition(job, JobStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    enqueue_process(job_id)

    log.info("Retrying job %s", job_id)
    return JobResponse.model_validate(job)
