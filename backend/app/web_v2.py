"""
v2 Web UI router — server-rendered Jinja2 + HTMX pages.

Mounted at /v2 from app/main.py.
Provides: Projects dashboard, Project detail (segment editor + runs panel),
Run detail (progress polling + result preview).

v1 web.py is left completely untouched.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from . import ai_models
from .api_v2 import _MAX_BATCH_COPY_RUNS
from .config import settings
from .db import get_db
from .models import Run, RunSegment, SegmentDef, VideoProject
from sqlalchemy.orm import selectinload
from .pipeline_v2 import effective_segment_cap_sec
from .public import make_result_token, make_source_token
from .state_machine import ProjectStatus, RunStatus, SegmentStatus

log = logging.getLogger(__name__)

router = APIRouter(tags=["web_v2"])

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates", "v2")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

# Every template that names a model renders it through this filter rather than a
# ternary chain, so a new entry in app/ai_models.py is labelled everywhere at
# once.
templates.env.filters["model_label"] = ai_models.label_for


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


def _new_run_defaults(project: VideoProject) -> dict:
    """New Run form pre-fill: the AI model registry the `<select>` and JS use.

    The ``model*`` keys all come from app/ai_models.py so the model dropdown, the
    model→resolution options map and the per-model clip-length/audio limits derive
    from one table instead of hand-kept copies in the template.

    Shared by the full page render and the HTMX status fragment, which both render
    partials/project_status_content.html — keep them in one helper so the two can
    never drift.
    """
    # The project's segmentation cap decides which models can actually generate
    # its segments. Pre-selecting one that cannot is a trap: over-long segments
    # are skipped and the run delivers original footage where a swap was paid
    # for. So when the registry default can't cover the cap, fall back to the
    # first model that can — registry order is cheapest-first within a family, so
    # "first compatible" is also the least-surprising spend.
    #
    # This only ever *widens* what the operator sees pre-filled — every model
    # stays selectable, and `models_over_cap` lets the template mark the ones
    # that would fall back rather than hiding them.
    cap = effective_segment_cap_sec(project.max_segment_sec)
    compatible = ai_models.models_supporting_segment_len(cap)

    default_model = ai_models.DEFAULT_MODEL
    if compatible and default_model not in compatible:
        default_model = compatible[0]
        log.info(
            "project %s cap=%.1fs excludes the default model %r — New Run form "
            "defaults to %r instead",
            project.id, cap, ai_models.DEFAULT_MODEL, default_model,
        )

    return {
        "default_model": default_model,
        # Effective segmentation cap and the models that cannot honour it, so the
        # <option> list can label them instead of silently offering a run that
        # delivers un-swapped footage.
        "segment_cap_sec": cap,
        "models_over_cap": ai_models.models_excluded_by_segment_len(cap),
        # {model: [[value, label], ...]} — rebuilds the resolution dropdown.
        "model_resolutions_json": ai_models.resolution_choices_json(),
        # {model: {max_clip_sec, produces_audio, label}} — drives the "segments
        # too long for this model" warning and the audio lock.
        "model_limits_json": ai_models.model_limits_json(),
        # The specs themselves, to loop over when building the <option> list.
        "ai_models": ai_models.AI_MODELS,
    }


def _get_run_or_404(run_id: str, db: Session) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return run


def _gdrive_link_for_run(run: Run) -> str | None:
    if run.result_gdrive_file_id:
        return f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
    return None


def _result_version_for_run(run: Run) -> str:
    """Return a stable cache-buster for the current on-disk result file."""
    path = run.result_local_path
    if path and os.path.exists(path):
        stat = os.stat(path)
        return f"{int(stat.st_mtime)}-{stat.st_size}"
    updated = run.updated_at
    if updated is not None:
        return str(int(updated.timestamp()))
    return "0"


def _get_segments(project_id: str, db: Session) -> list[SegmentDef]:
    return (
        db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == project_id)
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    )


def _get_runs(project_id: str, db: Session) -> list[Run]:
    return (
        db.execute(
            select(Run)
            .where(Run.project_id == project_id)
            .order_by(desc(Run.created_at))
        )
        .scalars()
        .all()
    )


def _get_run_segments(run_id: str, db: Session) -> list[RunSegment]:
    return (
        db.execute(
            select(RunSegment)
            .where(RunSegment.run_id == run_id)
            .order_by(RunSegment.index)
            .options(selectinload(RunSegment.segment_def))
        )
        .scalars()
        .all()
    )


_DEFAULT_PROMPT = (
    "Replace the main person in the reference video with the person shown in the "
    "reference image. Keep their face and identity consistent with the reference image "
    "throughout. Change only the character — keep everything else exactly the same: "
    "the phone or tablet screen and its contents, all on-screen text and captions, "
    "the background, lighting, framing, and the original motion and lip movements."
)


# ---------------------------------------------------------------------------
# Projects dashboard — GET /v2/
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def projects_dashboard(
    request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """v2 dashboard: list of VideoProjects + new-project form."""
    projects = (
        db.execute(select(VideoProject).order_by(desc(VideoProject.created_at)))
        .scalars()
        .all()
    )
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "projects": projects,
        },
    )


# ---------------------------------------------------------------------------
# Dashboard runs pivot — GET /v2/projects/{pid}/runs-fragment
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/runs-fragment", response_class=HTMLResponse)
def project_runs_fragment(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX fragment: the Existing-runs list for a project, shown inline in the
    dashboard pivot so finished videos are reachable without opening the project."""
    _get_project_or_404(pid, db)
    runs = _get_runs(pid, db)
    rows = []
    for run in runs:
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        rows.append(
            {
                "id": run.id,
                "name": run.name,
                "status": status_val,
                "model": run.model,
                "resolution": run.resolution,
                "created_at": run.created_at,
                "result_available": (
                    status_val == "done"
                    and bool(run.result_local_path)
                    and os.path.exists(run.result_local_path)
                ),
            }
        )
    return templates.TemplateResponse(
        "partials/project_runs_list.html",
        {"request": request, "project_id": pid, "runs": rows},
    )


# ---------------------------------------------------------------------------
# Project page — GET /v2/projects/{pid}
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}", response_class=HTMLResponse)
def project_detail(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Full project detail page."""
    project = _get_project_or_404(pid, db)
    segments = _get_segments(pid, db)
    runs = _get_runs(pid, db)
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)
    return templates.TemplateResponse(
        "project_detail.html",
        {
            "request": request,
            "project": project,
            "status_val": status_val,
            "segments": segments,
            "runs": runs,
            "default_resolution": settings.DEFAULT_RESOLUTION,
            "source_public_token": make_source_token(pid),
            "max_refs": settings.MAX_REFERENCE_IMAGES,
            "gdrive_folder_id": settings.GDRIVE_DEFAULT_FOLDER_ID or "",
            "default_prompt": _DEFAULT_PROMPT,
            **_new_run_defaults(project),
        },
    )


# ---------------------------------------------------------------------------
# Project status fragment — GET /v2/projects/{pid}/status-fragment
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/status-fragment", response_class=HTMLResponse)
def project_status_fragment(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment for project status-dependent content."""
    project = _get_project_or_404(pid, db)
    segments = _get_segments(pid, db)
    runs = _get_runs(pid, db)
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)
    return templates.TemplateResponse(
        "partials/project_status_content.html",
        {
            "request": request,
            "project": project,
            "status_val": status_val,
            "segments": segments,
            "runs": runs,
            "default_resolution": settings.DEFAULT_RESOLUTION,
            "source_public_token": make_source_token(pid),
            "max_refs": settings.MAX_REFERENCE_IMAGES,
            "gdrive_folder_id": settings.GDRIVE_DEFAULT_FOLDER_ID or "",
            "default_prompt": _DEFAULT_PROMPT,
            **_new_run_defaults(project),
        },
    )


# ---------------------------------------------------------------------------
# Run page — GET /v2/runs/{rid}
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}", response_class=HTMLResponse)
def run_detail(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Full run detail page."""
    run = _get_run_or_404(rid, db)
    run_segments = _get_run_segments(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    gdrive_link = _gdrive_link_for_run(run)

    # Compute progress for display
    total_swap = len(run_segments)
    completed = sum(
        1 for rs in run_segments
        if (rs.status.value if hasattr(rs.status, "value") else str(rs.status)) == "completed"
    )

    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run": run,
            "project_id": run.project_id,
            "status_val": status_val,
            "run_segments": run_segments,
            "gdrive_link": gdrive_link,
            "total_swap": total_swap,
            "completed": completed,
            "result_public_token": make_result_token(rid),
            "result_version": _result_version_for_run(run),
            "max_copy_runs": _MAX_BATCH_COPY_RUNS,
            # The copy control's resolution list is built from the run's model's
            # own supported set, so it can never offer a tier create_run rejects.
            "ai_models": ai_models.AI_MODELS,
        },
    )


# ---------------------------------------------------------------------------
# Run status fragment — GET /v2/runs/{rid}/status-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/status-fragment", response_class=HTMLResponse)
def run_status_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment for run status-dependent content."""
    run = _get_run_or_404(rid, db)
    run_segments = _get_run_segments(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    gdrive_link = _gdrive_link_for_run(run)

    total_swap = len(run_segments)
    completed = sum(
        1 for rs in run_segments
        if (rs.status.value if hasattr(rs.status, "value") else str(rs.status)) == "completed"
    )

    # Find current generating segment
    generating_seg = None
    for rs in run_segments:
        rs_status = rs.status.value if hasattr(rs.status, "value") else str(rs.status)
        if rs_status == "generating":
            generating_seg = rs
            break

    return templates.TemplateResponse(
        "partials/run_status_content.html",
        {
            "request": request,
            "run": run,
            "project_id": run.project_id,
            "status_val": status_val,
            "run_segments": run_segments,
            "gdrive_link": gdrive_link,
            "total_swap": total_swap,
            "completed": completed,
            "generating_seg": generating_seg,
            "result_public_token": make_result_token(rid),
            "result_version": _result_version_for_run(run),
        },
    )
