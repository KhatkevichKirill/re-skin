"""
web_overlay.py — Operator web UI for the standalone overlay-burn feature.

Mounted at ``/v2/overlays`` (see main.py). HTMX + Jinja2, consistent with
web_v2.py. Two pages:

  /v2/overlays/            dashboard: pick a completed run as source + list jobs
  /v2/overlays/{job_id}    editor: upload PNGs, drag-position them on the video
                           preview, set time windows, render, download result

The drag layer maps CSS percentages 1:1 onto the normalized coords the backend
expects — no server-side frame extraction needed.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from . import drive_sa
from . import overlay_runs
from .accel import accel_or_file_response
from .db import get_db, get_session
from .models import LibraryClip, OverlayJob, Run, RunCaption, RunTailEdit, VideoProject
from .public import (
    make_caption_result_token,
    make_result_token,
    make_tail_edit_result_token,
)
from .state_machine import CaptionStatus, RunStatus, TailEditStatus
from .storage import overlay_job_dir

log = logging.getLogger(__name__)

router = APIRouter(tags=["web_overlay"])

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates", "overlay")
# Search path also covers templates/ (parent) so `{% include "site_header.html" %}`
# in base.html resolves to the one shared header partial used by every area
# (v2 + overlay) — see web_v2.py, which adds the same fallback dir.
_SHARED_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=[_TEMPLATES_DIR, _SHARED_TEMPLATES_DIR])
drive_sa.register(templates.env)  # AICP-274 Drive sharing notice


def _status_val(entity) -> str:
    s = entity.status
    return s.value if hasattr(s, "value") else str(s)


def _get_job_or_404(job_id: str, db: Session) -> OverlayJob:
    job = db.get(OverlayJob, job_id)
    # Soft-deleted jobs are hidden (the row survives only for DB-level lookups).
    if job is None or job.deleted_at is not None:
        raise HTTPException(status_code=404, detail=f"Overlay job {job_id!r} not found")
    return job


def _source_preview_url(job: OverlayJob, db: Session) -> str | None:
    """Public, token-signed URL the editor's <video> can stream/seek.

    None when public links are disabled (no PUBLIC_LINK_SECRET): the editor
    falls back to numeric-only placement.
    """
    if job.source_type == "run" and job.source_run_id:
        tok = make_result_token(job.source_run_id)
        return f"/public/runs/{job.source_run_id}/result?token={tok}" if tok else None
    if job.source_type == "tail_edit" and job.source_tail_edit_id:
        tok = make_tail_edit_result_token(job.source_tail_edit_id)
        return (
            f"/public/tail-edits/{job.source_tail_edit_id}/result?token={tok}"
            if tok
            else None
        )
    if job.source_type == "caption" and job.source_caption_id:
        tok = make_caption_result_token(job.source_caption_id)
        return (
            f"/public/captions/{job.source_caption_id}/result?token={tok}"
            if tok
            else None
        )
    # Uploaded sources: served by the API download of the source is not exposed;
    # fall back to None (the editor shows numeric-only placement).
    return None


def _completed_runs_for_project(pid: str, db: Session) -> list[Run]:
    """Completed runs of a project whose result file is present on disk."""
    runs = (
        db.execute(
            select(Run)
            .where(Run.project_id == pid, Run.status == RunStatus.done)
            .order_by(desc(Run.created_at))
        )
        .scalars()
        .all()
    )
    return [r for r in runs if r.result_local_path and os.path.exists(r.result_local_path)]


def _completed_tail_edits_for_project(pid: str, db: Session) -> list[RunTailEdit]:
    """Done tail edits (via their parent run) of a project whose output file
    is present on disk."""
    edits = (
        db.execute(
            select(RunTailEdit)
            .join(Run, RunTailEdit.run_id == Run.id)
            .where(Run.project_id == pid, RunTailEdit.status == TailEditStatus.done)
            .order_by(desc(RunTailEdit.created_at))
        )
        .scalars()
        .all()
    )
    return [e for e in edits if e.output_path and os.path.exists(e.output_path)]


def _completed_captions_for_project(pid: str, db: Session) -> list[RunCaption]:
    """Done caption jobs (via their parent run) of a project whose burned
    output file is present on disk — the caption-source counterpart of
    _completed_tail_edits_for_project."""
    captions = (
        db.execute(
            select(RunCaption)
            .join(Run, RunCaption.run_id == Run.id)
            .where(Run.project_id == pid, RunCaption.status == CaptionStatus.done)
            .order_by(desc(RunCaption.created_at))
        )
        .scalars()
        .all()
    )
    return [c for c in captions if c.output_path and os.path.exists(c.output_path)]


@router.get("/", response_class=HTMLResponse)
def overlay_dashboard(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    """Pick a source video (projects → expand to completed runs) + list jobs."""
    # Projects that have at least one completed run — the source-picker pivots
    # on projects (like the All Projects table) so runs are grouped, not a flat
    # list of hundreds. The runs themselves load lazily per project.
    project_ids = set(
        db.execute(
            select(Run.project_id)
            .where(Run.status == RunStatus.done)
            .distinct()
        )
        .scalars()
        .all()
    )
    # A project may have no done Run of its own (e.g. it was superseded) but
    # still have a done derived creative off an older run — include those
    # projects too so those videos stay selectable as overlay sources.
    project_ids |= set(
        db.execute(
            select(Run.project_id)
            .join(RunTailEdit, RunTailEdit.run_id == Run.id)
            .where(RunTailEdit.status == TailEditStatus.done)
            .distinct()
        )
        .scalars()
        .all()
    )
    project_ids |= set(
        db.execute(
            select(Run.project_id)
            .join(RunCaption, RunCaption.run_id == Run.id)
            .where(RunCaption.status == CaptionStatus.done)
            .distinct()
        )
        .scalars()
        .all()
    )
    projects = (
        db.execute(
            select(VideoProject)
            .where(VideoProject.id.in_(project_ids))
            .order_by(desc(VideoProject.created_at))
        )
        .scalars()
        .all()
    ) if project_ids else []
    project_rows = [
        {
            "id": p.id,
            "name": p.name,
            # source_type/source_run_id feed the shared source_cell macro
            # (templates/project_source_cell.html), which shows the KIND of
            # source and keeps source_ref in the hover title. They must be
            # carried explicitly: these are plain dicts, and a missing key is
            # Undefined in Jinja — falsy, so every project would read "file".
            "source_type": p.source_type,
            "source_run_id": p.source_run_id,
            "source_ref": p.source_ref,
            "created_at": p.created_at,
        }
        for p in projects
    ]

    jobs = (
        db.execute(
            select(OverlayJob)
            .where(OverlayJob.deleted_at.is_(None))
            .order_by(desc(OverlayJob.created_at))
            .limit(100)
        )
        .scalars()
        .all()
    )
    job_rows = [
        {"id": j.id, "status": _status_val(j), "source_run_id": j.source_run_id,
         "created_at": j.created_at, "n_overlays": len(j.overlays or []),
         "gdrive_file_id": j.gdrive_file_id}
        for j in jobs
    ]
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"request": request, "projects": project_rows, "jobs": job_rows},
    )


@router.get("/projects/{pid}/sources-fragment", response_class=HTMLResponse)
def project_sources_fragment(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX fragment: a project's completed runs, each with its derived
    creatives nested underneath as a pivot.

    A derived creative — a tail edit or a caption job — belongs to exactly one
    run via its ``run_id`` (a real FK), so both kinds nest under the run they
    came from and carry a ``kind`` the template turns into the source_type the
    create call needs.
    """
    project = db.get(VideoProject, pid)
    runs = []
    # Keyed by run_id so derived creatives can be attached below in one pass,
    # but built from a list first to preserve _completed_runs_for_project's
    # newest-first ordering for the template.
    run_by_id: dict[str, dict] = {}
    for run in _completed_runs_for_project(pid, db):
        row = {
            "run_id": run.id,
            "run_name": run.name or f"(unnamed) {run.id[:8]}",
            "created_at": run.created_at,
            "width": project.width if project else None,
            "height": project.height if project else None,
            "duration_sec": project.duration_sec if project else None,
            "derived": [],
        }
        runs.append(row)
        run_by_id[run.id] = row

    tail_edits = _completed_tail_edits_for_project(pid, db)
    captions = _completed_captions_for_project(pid, db)
    # library_clip_id is a soft reference (no FK/relationship — mirrors
    # web_v2._tail_clip_names); look up names for the clips still on file.
    clip_ids = {e.library_clip_id for e in tail_edits if e.library_clip_id}
    clip_names = (
        {
            c.id: c.name
            for c in db.execute(
                select(LibraryClip).where(LibraryClip.id.in_(clip_ids))
            ).scalars()
        }
        if clip_ids
        else {}
    )

    derived = [
        {
            "kind": "tail_edit",
            "source_id": e.id,
            "run_id": e.run_id,
            "label": (
                clip_names.get(e.library_clip_id) or os.path.basename(e.output_path)
            ),
            "created_at": e.created_at,
        }
        for e in tail_edits
    ] + [
        {
            "kind": "caption",
            "source_id": c.id,
            "run_id": c.run_id,
            "label": f"Captions ({c.language or 'auto'})",
            "created_at": c.created_at,
        }
        for c in captions
    ]

    # Derived creatives whose parent run isn't in `runs` — because the run
    # itself isn't done, or its result file is missing (see
    # overlay_dashboard's comment on why such projects are still listed).
    # They must stay selectable, so they're rendered in a trailing "orphaned"
    # group rather than dropped; we still need each such run's name for that
    # group even though the run row itself was excluded above, so fetch it
    # directly.
    orphan_run_ids = {d["run_id"] for d in derived if d["run_id"] not in run_by_id}
    orphan_run_names = (
        {
            r.id: r.name or f"(unnamed) {r.id[:8]}"
            for r in db.execute(select(Run).where(Run.id.in_(orphan_run_ids))).scalars()
        }
        if orphan_run_ids
        else {}
    )

    orphan_rows = []
    for row in derived:
        parent = run_by_id.get(row["run_id"])
        if parent is not None:
            parent["derived"].append(row)
        else:
            orphan_rows.append(
                {**row, "run_name": orphan_run_names.get(row["run_id"], row["run_id"][:8])}
            )
    for run_row in runs:
        run_row["derived"].sort(key=lambda d: d["created_at"], reverse=True)
    orphan_rows.sort(key=lambda d: d["created_at"], reverse=True)

    return templates.TemplateResponse(
        request,
        "partials/source_runs_list.html",
        {"request": request, "runs": runs, "orphan_derived": orphan_rows},
    )


@router.get("/{job_id}", response_class=HTMLResponse)
def overlay_editor(job_id: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    """Editor: upload PNGs, drag-position them, set times, render, download."""
    job = _get_job_or_404(job_id, db)
    # Existing assets on disk (so a reloaded editor re-renders saved placements).
    assets = []
    adir = os.path.join(overlay_job_dir(job_id), "assets")
    if os.path.isdir(adir):
        assets = sorted(
            os.path.join("assets", f) for f in os.listdir(adir)
            if f.lower().endswith(".png")
        )
    # Overlays started from a run page get a way back to it (the run's own
    # "Overlays" card lists this job and its exports) — resolved through the
    # tail edit for a tail-edit source, same link overlay_runs.parent_run makes.
    parent = overlay_runs.parent_run(job, db)
    return templates.TemplateResponse(
        request,
        "editor.html",
        {
            "request": request,
            "job": job,
            "parent_run_id": parent.id if parent else None,
            "status_val": _status_val(job),
            "preview_url": _source_preview_url(job, db),
            "overlays": job.overlays or [],
            "assets": assets,
            "duration_sec": job.src_duration_sec,
            "result_available": bool(
                job.result_local_path and os.path.exists(job.result_local_path)
            ),
            "exported": job.final_creatives_exported_at is not None,
        },
    )


@router.get("/{job_id}/asset")
def overlay_asset(
    job_id: str,
    path: str = Query(..., description="Relative asset path, e.g. assets/card_01.png"),
) -> FileResponse:
    """Serve an uploaded PNG asset so the editor can preview it (operator-only).

    Fix #5: when NGINX_ACCEL_ENABLED, returns X-Accel-Redirect so nginx
    serves the bytes directly instead of streaming through Python.
    Fix #6: DB session is scoped to a plain `with` block so it's committed and
    closed before the response body is sent.
    """
    with get_session() as db:
        _get_job_or_404(job_id, db)
        # Guard against path traversal: resolve and confirm it stays inside the job dir.
        root = os.path.realpath(overlay_job_dir(job_id))
        target = os.path.realpath(os.path.join(root, path))
        if not target.startswith(root + os.sep) or not os.path.exists(target):
            raise HTTPException(status_code=404, detail="asset not found")

    return accel_or_file_response(target, media_type="image/png")


@router.get("/{job_id}/status-fragment", response_class=HTMLResponse)
def overlay_status_fragment(job_id: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    """HTMX polling fragment for the render status + result."""
    job = _get_job_or_404(job_id, db)
    return templates.TemplateResponse(
        request,
        "partials/status_content.html",
        {
            "request": request,
            "job": job,
            "status_val": _status_val(job),
            "result_available": bool(
                job.result_local_path and os.path.exists(job.result_local_path)
            ),
            "exported": job.final_creatives_exported_at is not None,
        },
    )
