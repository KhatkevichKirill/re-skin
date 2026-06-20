"""
Web UI router — server-rendered Jinja2 + HTMX pages.

Mounted at / (no prefix) from app/main.py.
All routes here return HTML; JSON lives under /api.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import Job, Segment
from .state_machine import JobStatus

log = logging.getLogger(__name__)

router = APIRouter(tags=["web"])

# Templates are relative to this file's directory
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_job_or_404(job_id: str, db: Session) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


def _get_segments(job_id: str, db: Session) -> list[Segment]:
    return (
        db.execute(
            select(Segment).where(Segment.job_id == job_id).order_by(Segment.index)
        )
        .scalars()
        .all()
    )


def _gdrive_link(job: Job) -> str | None:
    if job.result_gdrive_file_id:
        return f"https://drive.google.com/file/d/{job.result_gdrive_file_id}/view"
    return None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    """Jobs dashboard with a new-job form and list of all jobs."""
    jobs = db.execute(select(Job).order_by(desc(Job.created_at))).scalars().all()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "jobs": jobs,
            "default_resolution": settings.DEFAULT_RESOLUTION,
            "max_refs": settings.MAX_REFERENCE_IMAGES,
            "gdrive_folder_id": settings.GDRIVE_DEFAULT_FOLDER_ID or "",
        },
    )


# ---------------------------------------------------------------------------
# Job detail
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(job_id: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    """Full job detail page."""
    job = _get_job_or_404(job_id, db)
    segments = _get_segments(job_id, db)
    return templates.TemplateResponse(
        "job_detail.html",
        {
            "request": request,
            "job": job,
            "segments": segments,
            "gdrive_link": _gdrive_link(job),
        },
    )


# ---------------------------------------------------------------------------
# HTMX polling fragment — status-dependent content only
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/status-fragment", response_class=HTMLResponse)
def job_status_fragment(
    job_id: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Returns the inner status-content partial for HTMX polling."""
    job = _get_job_or_404(job_id, db)
    segments = _get_segments(job_id, db)
    return templates.TemplateResponse(
        "partials/job_status_content.html",
        {
            "request": request,
            "job": job,
            "segments": segments,
            "gdrive_link": _gdrive_link(job),
        },
    )


# ---------------------------------------------------------------------------
# Retry redirect (form POST from the failed-job view)
# ---------------------------------------------------------------------------


@router.post("/jobs/{job_id}/retry", response_class=HTMLResponse)
def retry_redirect(job_id: str, db: Session = Depends(get_db)) -> RedirectResponse:
    """
    Web-form retry — delegates to the API endpoint via redirect
    so the operator doesn't have to deal with JSON.

    We call the API logic directly rather than doing an internal HTTP round-trip.
    """
    from .state_machine import InvalidTransition, transition

    import app.api as api_module

    job = _get_job_or_404(job_id, db)
    if job.status != JobStatus.failed:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot retry: job status is {job.status!r}",
        )
    try:
        transition(job, JobStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    api_module.enqueue_process(job_id)

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)
