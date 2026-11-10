"""Segment caption fill API (AICP-358, docs/segment-caption-fill-design.md §3).

* ``POST /runs/{rid}/segments/{rsid}/fill/preview`` — free: freeze the base,
  queue ``segment_fill_prepare`` (mask + VOID inputs + preview sheet), 202.
* ``POST /runs/{rid}/segments/fill/{fid}/start`` — paid: the operator's
  approval of the estimate becomes a Fal provider task.
* ``POST /runs/{rid}/segments/fill/{fid}/apply-source`` — free: put a source
  preview that found no caption in as the blocked segment's result (as is).
* ``POST /runs/{rid}/segments/fill/{fid}/cancel`` — drop a ``ready`` preview.

``basis`` (preview body): ``model_result`` (default — a done run's model
output is rebuilt) or ``source`` (a swap segment the video provider refused
with a content filter, ``app.content_block``, on an incomplete run).
* ``GET  /runs/{rid}/segments/fill`` and ``.../{fid}/preview.jpg`` — status.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import pipeline_segment_fill as psf
from .db import get_db
from .ops_gate import GATED
from .models import Run, RunSegment, VideoProject
from .project_types import SUBTITLE_REMOVAL
from .provider_models import ProviderTask
from .segment_fill_models import FillStatus, RunSegmentFillBatch, RunSegmentFillJob
from .state_machine import SegmentStatus

log = logging.getLogger(__name__)

router = APIRouter(tags=["segment-fill"])


class FillPreviewRequest(BaseModel):
    mask_box: Optional[list[float]] = None
    prompt: Optional[str] = None
    basis: Literal["model_result", "source"] = "model_result"
    # Batch fills (2026-11-03): extra glyph-mask dilation in px (0..24), and
    # the applied composite fill this preview retries with a grown mask.
    mask_grow_px: Optional[int] = None
    retry_of: Optional[str] = None


class FillStartRequest(BaseModel):
    approved_usd: float
    discard_audio_patch: bool = False


class BatchPreviewRequest(BaseModel):
    run_segment_ids: list[str]
    mask_grow_px: Optional[int] = None
    basis: Literal["model_result", "source"] = "model_result"


class BatchStartRequest(BaseModel):
    fill_ids: list[str]
    approved_usd: float
    discard_audio_patch: bool = False


def _status(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def job_view(job: RunSegmentFillJob, run: Optional[Run], db: Session) -> dict:
    task = db.get(ProviderTask, job.provider_task_id) if job.provider_task_id else None
    media = job.media or {}
    return {
        "id": job.id, "run_id": job.run_id, "run_segment_id": job.run_segment_id,
        "status": job.status, "model": job.model, "mask_box": job.mask_box,
        "basis": (job.base or {}).get("basis") or psf.BASIS_MODEL,
        "source_as_is": bool(media.get("source_as_is")),
        "source_as_is_reason": media.get("source_as_is_reason"),
        "box_source": job.box_source, "prompt": job.prompt or psf.DEFAULT_PROMPT,
        "frames": media.get("frames"), "frames_masked": media.get("frames_masked"),
        "void_frames": media.get("void_frames"), "estimated_usd": job.estimated_usd,
        "approved_usd": job.approved_usd,
        "preview_url": (f"/api/v2/runs/{job.run_id}/segments/fill/{job.id}/preview.jpg"
                        if job.preview_path else None),
        "provider_task_id": job.provider_task_id,
        "provider_status": task.status if task is not None else None,
        "provider_error": (task.error_message if task is not None else None),
        "delivery": psf.delivery_state(job, run, db.get(RunSegment, job.run_segment_id), db=db),
        "error_code": job.error_code, "error_message": job.error_message,
        "batch_id": job.batch_id, "retry_of": job.retry_of, "mask_grow_px": job.mask_grow_px,
        "segment_bound": psf.base_bind(job) == psf.BIND_SEGMENT,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "applied_at": job.applied_at.isoformat() if job.applied_at else None,
    }


def batch_view(batch: RunSegmentFillBatch, db: Session) -> dict:
    members = db.execute(select(RunSegmentFillJob).where(
        RunSegmentFillJob.batch_id == batch.id).order_by(RunSegmentFillJob.id)).scalars().all()
    return {
        "id": batch.id, "run_id": batch.run_id, "status": batch.status,
        "authorization": batch.authorization_kind, "approved_usd": batch.approved_usd,
        "reserved_usd": batch.reserved_usd, "error_code": batch.error_code,
        "error_message": batch.error_message, "summary": batch.summary,
        "members": [{"id": m.id, "run_segment_id": m.run_segment_id, "status": m.status,
                     "error_code": m.error_code} for m in members],
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "applied_at": batch.applied_at.isoformat() if batch.applied_at else None,
    }


def _load(db: Session, rid: str, rsid: Optional[str] = None):
    run = db.get(Run, rid)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    project = db.get(VideoProject, run.project_id)
    rs = None
    if rsid is not None:
        rs = db.get(RunSegment, rsid)
        if rs is None or rs.run_id != rid:
            raise HTTPException(status_code=404, detail="segment not found in this run")
    return run, project, rs


def _require_fillable(run: Run, project: VideoProject, rs: RunSegment,
                      basis: str = psf.BASIS_MODEL) -> None:
    from . import content_block
    from .api_v2 import _run_segment_defs_for
    from .pipeline_v2 import _active_swap_segment_ids

    if basis not in psf.BASES:
        raise HTTPException(status_code=422, detail=f"unknown fill basis {basis!r}")
    if project is None or project.project_type != SUBTITLE_REMOVAL:
        raise HTTPException(status_code=422, detail="caption fill is for subtitle_removal projects")
    if basis == psf.BASIS_SOURCE:
        if _status(run.status) not in psf.SOURCE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=(
                f"the run is {_status(run.status)}: a fill from the source is for a segment "
                "the provider refused on an incomplete/failed run"))
    elif _status(run.status) != "done":
        raise HTTPException(status_code=409, detail=f"the run is {_status(run.status)}, not done")
    if run.audio_mode != "original":
        raise HTTPException(status_code=422,
                            detail="caption fill needs a run with audio_mode=original")
    if rs.segment_def_id not in _active_swap_segment_ids(run.stitch_plan, _run_segment_defs_for(run)):
        raise HTTPException(status_code=422, detail="not an active swap segment of this run")
    if basis == psf.BASIS_SOURCE:
        from .project_types import is_void_source_run

        if is_void_source_run(run):
            # VOID from the source as the run's method (design §7.2): every
            # active swap segment without a finished result is filled from
            # the source -- a model fallback that failed goes back to it too.
            if rs.status == SegmentStatus.completed and not rs.source_fallback \
                    and rs.local_result_path and os.path.exists(rs.local_result_path):
                raise HTTPException(status_code=409, detail=(
                    "the segment already has a result; rebuild it from that result instead"))
            return
        if content_block.segment_block(rs) is None:
            raise HTTPException(status_code=409, detail=(
                "a fill from the source needs a failed segment the provider refused with a "
                "content filter (safety / sensitive content / copyright); this one is "
                f"{_status(rs.status)}: {(rs.error_message or 'no error')[:200]}"))
        return
    if rs.status != SegmentStatus.completed or rs.source_fallback \
            or not rs.local_result_path or not os.path.exists(rs.local_result_path):
        raise HTTPException(status_code=409, detail="the segment has no finished model result")


def _new_preview(db: Session, run: Run, project: VideoProject, rs: RunSegment,
                 body: FillPreviewRequest) -> RunSegmentFillJob:
    """Validate and ADD one preview row (the caller commits, then enqueues)."""
    _require_fillable(run, project, rs, body.basis)
    box = None
    if body.mask_box is not None:
        try:
            box = psf.validate_box(body.mask_box)
        except psf.FillError as exc:
            raise HTTPException(status_code=422, detail=exc.message) from None
    grow = body.mask_grow_px
    if grow is not None and (isinstance(grow, bool) or not 0 <= int(grow) <= psf.MAX_GROW_PX):
        raise HTTPException(status_code=422,
                            detail=f"mask_grow_px must be 0..{psf.MAX_GROW_PX}")
    retry_of = body.retry_of
    if retry_of is not None:
        parent = db.get(RunSegmentFillJob, retry_of)
        if parent is None or parent.run_id != run.id or parent.run_segment_id != rs.id \
                or parent.status != FillStatus.applied.value:
            raise HTTPException(status_code=422, detail=(
                "retry_of must be an applied fill of this segment"))
    prompt = (body.prompt or "").strip()[:2000]
    # A model-result preview is bound to its SEGMENT (batch fills, 2026-11-03):
    # another segment's fill re-stitching the final does not make it stale.
    bind = psf.BIND_SEGMENT if body.basis == psf.BASIS_MODEL else None
    job = RunSegmentFillJob(
        run_id=run.id, run_segment_id=rs.id, project_id=project.id,
        status=FillStatus.preparing.value, model="void", prompt=prompt,
        mask_box=box, box_source="request" if box else None,
        mask_grow_px=int(grow) if grow else None, retry_of=retry_of,
        base=psf.capture_base(db, run, rs, project, body.basis, bind=bind),
    )
    db.add(job)
    return job


def _enqueue_prepares(ids) -> None:
    from .tasks import enqueue_segment_fill_prepare

    for jid in ids:
        try:
            enqueue_segment_fill_prepare(jid)
        except Exception as exc:  # the recovery tick re-enqueues a never-claimed row
            log.error("segment_fill: enqueue of prepare %s failed: %s", jid, exc)


@router.post("/runs/{rid}/segments/{rsid}/fill/preview", status_code=status.HTTP_202_ACCEPTED, dependencies=[GATED])
def create_fill_preview(rid: str, rsid: str, body: FillPreviewRequest,
                        db: Session = Depends(get_db)) -> dict:
    run, project, rs = _load(db, rid, rsid)
    job = _new_preview(db, run, project, rs, body)
    db.commit()
    _enqueue_prepares([job.id])
    return job_view(job, run, db)


@router.post("/runs/{rid}/segments/fill/batch/preview", status_code=status.HTTP_202_ACCEPTED, dependencies=[GATED])
def create_batch_previews(rid: str, body: BatchPreviewRequest,
                          db: Session = Depends(get_db)) -> list[dict]:
    """Free: one preview per listed segment (all or none), prepared in
    parallel on the run workers. Each is bound to its own segment, so the
    ready ones can be started together (``.../fill/batch/start``)."""
    run, project, _ = _load(db, rid)
    ids = list(dict.fromkeys(body.run_segment_ids or []))
    cap = int(getattr(psf.settings, "SEGMENT_FILL_BATCH_MAX", 12))
    if not ids or len(ids) > cap:
        raise HTTPException(status_code=422, detail=f"select 1..{cap} segments")
    jobs = []
    for rsid in ids:
        rs = db.get(RunSegment, rsid)
        if rs is None or rs.run_id != rid:
            raise HTTPException(status_code=404, detail=f"segment {rsid} not found in this run")
        jobs.append(_new_preview(db, run, project, rs, FillPreviewRequest(
            mask_grow_px=body.mask_grow_px, basis=body.basis)))
    db.commit()
    _enqueue_prepares([j.id for j in jobs])
    return [job_view(j, run, db) for j in jobs]


@router.post("/runs/{rid}/segments/fill/batch/start", dependencies=[GATED])
def start_fill_batch(rid: str, body: BatchStartRequest, db: Session = Depends(get_db)) -> dict:
    """The operator's ONE paid start over several ready previews
    (``psf.admit_batch_start``): every member's provider task is admitted and
    reserved in one transaction; one promote, one re-stitch, one in-place
    re-delivery when they are done. A stale member is canceled (409)."""
    if not math.isfinite(body.approved_usd) or body.approved_usd < 0:
        raise HTTPException(status_code=422, detail="approved_usd must be a finite amount ≥ 0")
    run, _, _ = _load(db, rid)
    try:
        batch, members = psf.admit_batch_start(
            db, run, body.fill_ids, approved_usd=body.approved_usd,
            discard_audio_patch=body.discard_audio_patch)
    except psf.StalePreview as exc:
        db.rollback()
        _cancel_stale(db, exc.fill_id, exc.part)
        db.commit()
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    _enqueue_fal(members)
    log.info("segment_fill batch %s: admitted %d member(s)", batch.id, len(members))
    return batch_view(batch, db)


def _enqueue_fal(members) -> None:
    from .tasks import enqueue_fal_job

    for job, task, created in members:
        if not created:
            continue
        try:
            enqueue_fal_job("fal_submit", task.id)
        except Exception as exc:  # the Fal ticker recovers a queued task
            log.error("segment_fill %s: enqueue of fal_submit for task %s failed: %s",
                      job.id, task.id, exc)


def _cancel_stale(db: Session, fid: Optional[str], part: str) -> None:
    if not fid:
        return
    job = db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == fid)
                     .with_for_update()).scalar_one_or_none()
    if job is not None and job.status == FillStatus.ready.value:
        job.status, job.error_code = FillStatus.canceled.value, "stale_preview"
        job.error_message = f"{part} changed since the preview"


@router.get("/runs/{rid}/segments/fill/batches")
def list_fill_batches(rid: str, db: Session = Depends(get_db)) -> list[dict]:
    _load(db, rid)
    rows = db.execute(select(RunSegmentFillBatch).where(RunSegmentFillBatch.run_id == rid)
                      .order_by(desc(RunSegmentFillBatch.created_at))).scalars().all()
    return [batch_view(b, db) for b in rows]


@router.post("/runs/{rid}/segments/fill/{fid}/start", dependencies=[GATED])
def start_fill(rid: str, fid: str, body: FillStartRequest, db: Session = Depends(get_db)) -> dict:
    """The operator's paid start: ``psf.admit_start`` (single-commit service,
    AICP-364) + this endpoint's own commit / rollback / enqueue."""
    if not math.isfinite(body.approved_usd) or body.approved_usd < 0:
        raise HTTPException(status_code=422, detail="approved_usd must be a finite amount ≥ 0")
    run, _, _ = _load(db, rid)
    try:
        job, task, created = psf.admit_start(db, run, fid, approved_usd=body.approved_usd,
                                             discard_audio_patch=body.discard_audio_patch)
    except psf.StalePreview as exc:
        # Unchanged operator behaviour: the stale preview is canceled for good.
        db.rollback()
        _cancel_stale(db, exc.fill_id or fid, exc.part)
        db.commit()
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except HTTPException:
        db.rollback()
        raise
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="another fill is already active on this run")
    _enqueue_fal([(job, task, created)])
    log.info("segment_fill %s: admitted as provider task %s (created=%s)", fid, task.id, created)
    return job_view(job, run, db)


@router.post("/runs/{rid}/segments/fill/{fid}/apply-source", dependencies=[GATED])
def apply_source_fill(rid: str, fid: str, db: Session = Depends(get_db)) -> dict:
    """The operator's FREE "use the source clip as is" for a source preview
    that found no caption (``psf.apply_source_as_is``; it commits)."""
    run, _, _ = _load(db, rid)
    try:
        job = psf.apply_source_as_is(db, run, fid)
    except psf.StalePreview as exc:
        job = db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == fid)
                         .with_for_update()).scalar_one_or_none()
        if job is not None and job.status == FillStatus.ready.value:
            job.status, job.error_code = FillStatus.canceled.value, "stale_preview"
            job.error_message = f"{exc.part} changed since the preview"
        db.commit()
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except HTTPException:
        db.rollback()
        raise
    db.refresh(run)
    return job_view(job, run, db)


@router.post("/runs/{rid}/segments/fill/{fid}/cancel")
def cancel_fill(rid: str, fid: str, db: Session = Depends(get_db)) -> dict:
    run, _, _ = _load(db, rid)
    job = db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == fid)
                     .with_for_update()).scalar_one_or_none()
    if job is None or job.run_id != rid:
        raise HTTPException(status_code=404, detail="fill job not found in this run")
    if job.status != FillStatus.ready.value:
        raise HTTPException(status_code=409, detail=f"only a ready preview can be canceled "
                                                    f"(this one is {job.status})")
    job.status, job.error_code = FillStatus.canceled.value, "operator_canceled"
    db.commit()
    return job_view(job, run, db)


@router.get("/runs/{rid}/segments/fill")
def list_fills(rid: str, db: Session = Depends(get_db)) -> list[dict]:
    run, _, _ = _load(db, rid)
    rows = db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.run_id == rid)
                      .order_by(desc(RunSegmentFillJob.created_at))).scalars().all()
    return [job_view(j, run, db) for j in rows]


@router.get("/runs/{rid}/segments/fill/{fid}/preview.jpg")
def fill_preview_image(rid: str, fid: str, db: Session = Depends(get_db)):
    job = db.get(RunSegmentFillJob, fid)
    if job is None or job.run_id != rid or not job.preview_path \
            or not os.path.isfile(job.preview_path):
        raise HTTPException(status_code=404, detail="no preview")
    return FileResponse(job.preview_path, media_type="image/jpeg")
