"""
HTTP surface of the reframe workflow (AICP-251, docs/reframe-design.md §5-§9).

JSON/action endpoints under ``/api/v2`` and the project-page fragment under
``/v2``. Every write takes the project row lock first (the provider layer's
consumer contract), then the job row, then the render row.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import languages
from . import pipeline_provider as pp
from . import pipeline_reframe as prf
from . import reframe as rf
from . import storage
from .config import settings
from .db import get_db
from .ops_gate import GATED
from .models import LibraryClip, VideoProject
from .project_types import REFRAME
from .provider_models import ProviderTask
from .reframe_models import REFRAME_RENDER_ACTIVE_STATUSES, ReframeJob, ReframeRender
from .state_machine import ProjectStatus, ProviderTaskStatus

log = logging.getLogger(__name__)

router = APIRouter(tags=["reframe"])
web_router = APIRouter(tags=["reframe-web"])

_ACTIVE_RENDER = tuple(s.value for s in REFRAME_RENDER_ACTIVE_STATUSES)
_ACTIONS = ("adopt", "abandon", "close", "resume")


# ---------------------------------------------------------------------------
# Locking helpers
# ---------------------------------------------------------------------------


def _lock_project(db: Session, pid: str) -> VideoProject:
    project = db.execute(
        select(VideoProject).where(VideoProject.id == pid).with_for_update()
    ).scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _job_or_404(db: Session, jid: str) -> ReframeJob:
    job = db.get(ReframeJob, jid)
    if job is None:
        raise HTTPException(status_code=404, detail="reframe job not found")
    return job


def _lock_job(db: Session, jid: str) -> tuple[VideoProject, ReframeJob]:
    """Project row, then job row (the design's lock order)."""
    job = _job_or_404(db, jid)
    project = _lock_project(db, job.project_id)
    job = db.execute(
        select(ReframeJob).where(ReframeJob.id == jid).with_for_update()
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="reframe job not found")
    return project, job


def _task(db: Session, job: ReframeJob) -> Optional[ProviderTask]:
    return db.get(ProviderTask, job.provider_task_id) if job.provider_task_id else None


def fal_gate(pid: str, project_type: str | None) -> tuple[int, str] | None:
    """The admission checks create repeats before encoding (design §5 step 3)."""
    if not (settings.FAL_ENABLED and settings.FAL_API_KEY):
        return 403, "Fal is disabled on this server (FAL_ENABLED / FAL_API_KEY)"
    if not pp.project_allowed(pid, project_type):
        return 403, "this project is not in FAL_PROJECT_IDS (and is not a reframe project)"
    if not pp.fal_worker_alive():
        return 503, ("no fal-worker is running (deploy with `--profile fal`); "
                     "nothing was encoded or reserved")
    return None


# ---------------------------------------------------------------------------
# Create a generation
# ---------------------------------------------------------------------------


@router.post("/projects/{pid}/reframe-jobs", status_code=status.HTTP_201_CREATED, dependencies=[GATED])
def create_reframe_job(
    pid: str,
    window_start_sec: float = Form(...),
    window_end_sec: float = Form(...),
    resolution: str = Form("768p"),
    target_aspect: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    from .api_v2 import _budget_precheck

    project = _lock_project(db, pid)
    if project.project_type != REFRAME:
        raise HTTPException(status_code=409, detail="not a reframe project")
    if project.status != ProjectStatus.ready:
        raise HTTPException(status_code=409, detail="the project is not ready")
    if db.execute(select(ReframeJob.id).where(
            ReframeJob.project_id == pid, ReframeJob.status == "preparing")).first():
        raise HTTPException(status_code=409, detail="a generation is already being prepared")
    if pp.active_task_for_project(db, pid) is not None:
        raise HTTPException(
            status_code=409,
            detail="a Fal generation of this project is still active (one at a time; "
                   "settle a held or unknown task with its actions first)",
        )
    aspect = target_aspect or rf.target_aspect_for(project.width, project.height)
    try:
        rf.validate_generation(
            window_start=window_start_sec, window_end=window_end_sec, aspect=aspect or "",
            resolution=resolution, source_duration=project.duration_sec,
            source_width=project.width, source_height=project.height)
    except rf.ReframeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    gate = fal_gate(pid, project.project_type)
    if gate:
        raise HTTPException(status_code=gate[0], detail=gate[1])
    base, reserved = rf.estimate_usd(resolution, window_end_sec - window_start_sec)
    _budget_precheck(db, "reframe", needed=reserved / settings.KIE_USD_PER_CREDIT)

    job = ReframeJob(
        id=str(uuid.uuid4()), project_id=pid, status="preparing", target_aspect=aspect,
        resolution=resolution, window_start_sec=window_start_sec,
        window_end_sec=window_end_sec, estimated_usd_base=base,
        estimated_usd_reserved=reserved,
    )
    db.add(job)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="a generation is already being prepared") from None
    os.makedirs(storage.reframe_job_dir(job.id, pid), exist_ok=True)
    db.commit()
    try:
        from .tasks import enqueue_reframe_prepare

        enqueue_reframe_prepare(job.id)
    except Exception:
        log.exception("reframe: enqueue of prepare %s failed; recovery re-enqueues it", job.id)
    log.info("reframe: job %s created for project %s (%s %s, %.2f-%.2f s, ~%.2f USD)",
             job.id, pid, aspect, resolution, window_start_sec, window_end_sec, base)
    return {"id": job.id, "status": job.status, "estimated_usd_base": base,
            "estimated_usd_reserved": reserved}


# ---------------------------------------------------------------------------
# Operator actions on a job's task (design §5)
# ---------------------------------------------------------------------------


@router.post("/reframe-jobs/{jid}/task-actions/{action}")
def reframe_task_action(
    jid: str,
    action: str,
    note: str = Form(...),
    request_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    if action not in _ACTIONS:
        raise HTTPException(status_code=404, detail="unknown action")
    note = (note or "").strip()
    if not note:
        raise HTTPException(status_code=400, detail="a note is required")
    job = _job_or_404(db, jid)
    task_id = job.provider_task_id  # never from the request
    if not task_id:
        raise HTTPException(status_code=409, detail="this job has no Fal task")
    try:
        if action == "adopt":
            task = pp.adopt(db, task_id, request_id or "", actor="dashboard", note=note)
        else:
            task = getattr(pp, action)(db, task_id, actor="dashboard", note=note)
    except pp.ActionRefused as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
    log.info("reframe: operator %s on task %s of job %s", action, task_id, jid)
    return {"task_id": task_id, "status": task.status if task else None}


# ---------------------------------------------------------------------------
# Renders (design §6)
# ---------------------------------------------------------------------------


def cta_matches(width: Optional[int], height: Optional[int], aspect: str, resolution: str) -> bool:
    """Within 1 % of the target aspect (9:16 or 16:9), or of the model's
    native output ratio (768x1344 is 1.6 % off 9:16); the stitch scales the
    clip to the output size."""
    if not width or not height:
        return False
    a, b = (int(x) for x in aspect.split(":"))
    w, h = rf.output_size(aspect, resolution)
    r = width / height
    return any(abs(r - t) <= 0.01 * t for t in (a / b, w / h))


def _snapshot_cta(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
        return
    except FileNotFoundError:
        raise
    except OSError:
        pass
    tmp = f"{dst}.{uuid.uuid4().hex[:8]}.part"
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@router.post("/reframe-jobs/{jid}/renders", status_code=status.HTTP_201_CREATED, dependencies=[GATED])
def create_reframe_render(
    jid: str,
    picture_end_sec: float = Form(...),
    fade_sec: float = Form(rf.DEFAULT_FADE_SEC),
    audio_end_sec: Optional[float] = Form(None),
    black_tail_sec: float = Form(rf.DEFAULT_BLACK_TAIL_SEC),
    library_clip_id: Optional[str] = Form(None),
    language: str = Form(languages.DEFAULT_LANGUAGE),
    db: Session = Depends(get_db),
) -> dict:
    _project, job = _lock_job(db, jid)
    task = _task(db, job)
    if job.status != "admitted" or task is None or task.status != ProviderTaskStatus.done.value:
        raise HTTPException(status_code=409, detail="the generation is not done")
    if job.qa_status != "passed":
        raise HTTPException(status_code=409, detail=f"job QA is {job.qa_status}, not passed")
    raw = task.result_path or task.destination_path
    if not raw or not os.path.isfile(raw):
        raise HTTPException(status_code=409, detail="the raw output is missing on disk")
    if language not in languages.LANGUAGES:
        raise HTTPException(status_code=400, detail="unknown language")
    times = prf.job_times(job)
    ending = rf.Ending(picture_end_sec, fade_sec, audio_end_sec if times.has_audio else None,
                       black_tail_sec)
    try:
        rf.validate_ending(times, ending)
    except rf.ReframeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    clip = None
    if library_clip_id:
        clip = db.get(LibraryClip, library_clip_id)
        if clip is None or not clip.local_path or not os.path.isfile(clip.local_path):
            raise HTTPException(status_code=400, detail="CTA clip not found")
        if not cta_matches(clip.width, clip.height, job.target_aspect, job.resolution):
            raise HTTPException(status_code=400,
                                detail=f"the CTA clip is not {job.target_aspect}")
    render = ReframeRender(
        id=str(uuid.uuid4()), job_id=job.id, status="queued",
        picture_end_sec=picture_end_sec, fade_sec=fade_sec,
        audio_end_sec=ending.audio_end, black_tail_sec=black_tail_sec,
        library_clip_id=clip.id if clip else None, language=language,
    )
    rdir = storage.reframe_render_dir(render.id, job.id, job.project_id)
    os.makedirs(rdir, exist_ok=True)
    if clip is not None:
        cta = os.path.join(rdir, "cta.mp4")
        try:
            _snapshot_cta(clip.local_path, cta)
        except FileNotFoundError:
            db.rollback()
            shutil.rmtree(rdir, ignore_errors=True)
            raise HTTPException(status_code=409, detail="the CTA clip is no longer available") from None
        render.cta_path = cta
    db.add(render)
    db.commit()
    try:
        from .tasks import enqueue_reframe_render

        enqueue_reframe_render(render.id)
    except Exception:
        log.exception("reframe: enqueue of render %s failed", render.id)
    return {"id": render.id, "status": render.status}


@router.post("/reframe-renders/{rid}/retry", dependencies=[GATED])
def retry_reframe_render(rid: str, db: Session = Depends(get_db)) -> dict:
    render = db.get(ReframeRender, rid)
    if render is None:
        raise HTTPException(status_code=404, detail="render not found")
    _project, job = _lock_job(db, render.job_id)
    render = db.execute(select(ReframeRender).where(ReframeRender.id == rid)
                        .with_for_update()).scalar_one_or_none()
    if render is None or render.status != "failed":
        raise HTTPException(status_code=409, detail="only a failed render can be retried")
    if not os.path.isdir(storage.reframe_render_dir(rid, job.id, job.project_id)):
        raise HTTPException(status_code=409, detail="the render's files are gone; create a new one")
    render.status = "queued"
    render.claim_owner = None
    render.error_message = None
    render.updated_at = prf._now()
    db.commit()
    from .tasks import enqueue_reframe_render

    enqueue_reframe_render(rid)
    return {"id": rid, "status": "queued"}


# ---------------------------------------------------------------------------
# Deletion (design §7)
# ---------------------------------------------------------------------------


@router.delete("/reframe-jobs/{jid}")
def delete_reframe_job(jid: str, db: Session = Depends(get_db)) -> Response:
    _project, job = _lock_job(db, jid)
    why = prf.job_busy(db, job)
    if why:
        raise HTTPException(status_code=409, detail=f"Cannot delete the job: {why}")
    jdir = storage.reframe_job_dir(job.id, job.project_id)
    db.delete(job)  # renders cascade; the provider task row stays
    db.commit()
    shutil.rmtree(jdir, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/reframe-renders/{rid}")
def delete_reframe_render(rid: str, db: Session = Depends(get_db)) -> Response:
    render = db.get(ReframeRender, rid)
    if render is None:
        raise HTTPException(status_code=404, detail="render not found")
    _project, job = _lock_job(db, render.job_id)
    render = db.execute(select(ReframeRender).where(ReframeRender.id == rid)
                        .with_for_update()).scalar_one_or_none()
    if render is None:
        raise HTTPException(status_code=404, detail="render not found")
    if render.status in _ACTIVE_RENDER:
        raise HTTPException(status_code=409, detail="the render is still running")
    rdir = storage.reframe_render_dir(rid, job.id, job.project_id)
    db.delete(render)
    db.commit()
    shutil.rmtree(rdir, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _file(path: Optional[str], media_type: str, filename: Optional[str] = None) -> FileResponse:
    real = storage.path_within(path, storage.data_dir_root()) if path else None
    if not real or not os.path.isfile(real):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(real, media_type=media_type, filename=filename)


@router.get("/reframe-jobs/{jid}/raw")
def reframe_job_raw(jid: str, db: Session = Depends(get_db)) -> FileResponse:
    job = _job_or_404(db, jid)
    task = _task(db, job)
    if task is None or task.status != ProviderTaskStatus.done.value:
        raise HTTPException(status_code=404, detail="no raw output yet")
    return _file(task.result_path or task.destination_path, "video/mp4")


@router.get("/reframe-jobs/{jid}/sheet")
def reframe_job_sheet(jid: str, db: Session = Depends(get_db)) -> FileResponse:
    job = _job_or_404(db, jid)
    return _file(os.path.join(storage.reframe_job_dir(job.id, job.project_id), "sheet.jpg"),
                 "image/jpeg")


@router.get("/reframe-jobs/{jid}/audio")
def reframe_job_audio(jid: str, db: Session = Depends(get_db)) -> FileResponse:
    job = _job_or_404(db, jid)
    return _file(job.audio_path, "audio/flac")


@router.get("/reframe-renders/{rid}/result")
def reframe_render_result(rid: str, db: Session = Depends(get_db)) -> FileResponse:
    render = db.get(ReframeRender, rid)
    if render is None or render.status != "done":
        raise HTTPException(status_code=404, detail="no result")
    return _file(render.result_local_path, "video/mp4",
                 prf.output_filename(render.job_id, render.id))


# ---------------------------------------------------------------------------
# Project-page fragment (design §9)
# ---------------------------------------------------------------------------

_TASK_LABELS = {
    "queued": "Sending to Fal", "submit_started": "Sending to Fal",
    "accepted_unrecorded": "Sending to Fal", "submitted": "Generating",
    "fetching": "Downloading", "cancelling": "Waiting for Fal to confirm",
    "needs_reconciliation": "Waiting for Fal to confirm",
    "closing": "Waiting for Fal to confirm",
    "submit_unknown": "Unknown — may have been billed", "done": "Ready",
    "refused": "Refused by Fal", "failed": "Failed", "abandoned": "Abandoned",
}


def task_actions_for(task: Optional[ProviderTask]) -> list[str]:
    """The buttons shown for a task (the layer's functions stay the authority)."""
    if task is None:
        return []
    st = task.status
    if st == "submit_unknown":
        return ["adopt", "abandon", "close"]
    if st == "refused" and task.reservation_held:
        return ["abandon", "close"]
    if st == "needs_reconciliation":
        return ["resume", "close"]
    return []


def _kick_qa(job_id: str) -> None:
    """Enqueue job QA at most once a minute per job (the claim makes a
    duplicate harmless; this only avoids a queue full of them)."""
    try:
        from .tasks import _get_redis, enqueue_reframe_qa

        if _get_redis().set(f"reframe:qa-kick:{job_id}", "1", nx=True, ex=60):
            enqueue_reframe_qa(job_id)
    except Exception:
        log.exception("reframe: could not enqueue QA for job %s", job_id)


def reframe_context(db: Session, project: VideoProject) -> dict:
    jobs = db.execute(select(ReframeJob).where(ReframeJob.project_id == project.id)
                      .order_by(desc(ReframeJob.created_at))).scalars().all()
    rows = []
    active = False
    for job in jobs:
        task = _task(db, job)
        renders = db.execute(select(ReframeRender).where(ReframeRender.job_id == job.id)
                             .order_by(desc(ReframeRender.created_at))).scalars().all()
        st = task.status if task else None
        if job.status == "preparing" or (st and st not in ("done", "refused", "failed",
                                                             "abandoned", "submit_unknown")):
            active = True
        if st == "done" and job.qa_status in ("pending", "running"):
            active = True
            if job.qa_status == "pending":
                _kick_qa(job.id)
        if any(r.status in _ACTIVE_RENDER for r in renders):
            active = True
        defaults = None
        if st == "done" and job.qa_status == "passed":
            defaults = rf.default_ending(prf.job_times(job))
        rows.append({"job": job, "task": task, "renders": renders,
                     "task_label": _TASK_LABELS.get(st or "", ""),
                     "actions": task_actions_for(task), "defaults": defaults})
    aspect = rf.target_aspect_for(project.width, project.height)
    dur = project.duration_sec or 0.0
    default_end = round(min(dur, settings.REFRAME_MAX_WINDOW_SEC), 2)
    gate = fal_gate(project.id, project.project_type) if project.status == ProjectStatus.ready else None
    clips = db.execute(select(LibraryClip).order_by(desc(LibraryClip.created_at)).limit(200)
                       ).scalars().all()
    return {
        "project": project, "rows": rows, "reframe_active": active,
        "implied_aspect": aspect, "default_window_end": default_end,
        "max_window_sec": settings.REFRAME_MAX_WINDOW_SEC,
        "fal_gate": gate, "tariff": {"480p": 0.05, "768p": 0.08},
        "library_clips": clips, "languages": languages.LANGUAGES,
        "estimate": {r: rf.estimate_usd(r, max(default_end, rf.MIN_WINDOW_SEC))
                     for r in rf.RESOLUTIONS},
        "now": datetime.utcnow(),
    }


@web_router.get("/projects/{pid}/reframe-fragment", response_class=HTMLResponse)
def reframe_fragment(pid: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    from .web_v2 import templates

    project = db.get(VideoProject, pid)
    if project is None or project.project_type != REFRAME:
        raise HTTPException(status_code=404, detail="not a reframe project")
    ctx = reframe_context(db, project)
    ctx["request"] = request
    return templates.TemplateResponse(request, "partials/project_reframe_card.html", ctx)
