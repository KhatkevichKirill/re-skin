"""Operator resolution of parked paid submit attempts (AICP-351, design §5.1).

* ``GET  /api/v2/runs/{rid}/paid-attempts`` — the run's open attempts.
* ``POST /api/v2/runs/{rid}/paid-attempts/{aid}/resubmit`` (GATED) — the
  operator checked the kie.ai dashboard and accepts a possible second charge:
  CAS ``parked -> resubmitted`` (audited) and the owner's own relaunch in the
  SAME transaction (a refused relaunch rolls the CAS back).
* ``POST /api/v2/runs/{rid}/paid-attempts/{aid}/abandon`` (ungated, so it stays
  usable during the §6 rollback barrier) — CAS ``parked -> abandoned``.

Both refuse while the attempt's original sender is not proven stopped (see
:func:`app.paid_attempts._resolvable`). Both take the Run-row lock first, the
lock the other routes that touch the same owner take (lock order Run ->
attempt). CLI: ``python -m app.paid_attempts resubmit|abandon``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session

from . import paid_attempts
from .db import get_db
from .models import Run
from .ops_gate import GATED
from .paid_attempts_models import (
    OWNER_RUN_PREPROCESS,
    OWNER_RUN_SEGMENT,
    OWNER_SEGMENT_IMAGE_JOB,
    PaidSubmitAttempt,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["paid-attempts"])


def _http(exc: paid_attempts.ActionRefused) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def _lock_run(db: Session, rid: str) -> Run:
    run = db.get(Run, rid, with_for_update=True, populate_existing=True)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _attempt_of_run(db: Session, rid: str, aid: str) -> PaidSubmitAttempt:
    row = db.get(PaidSubmitAttempt, aid)
    if row is None or row.run_id != rid:
        raise HTTPException(status_code=404, detail="paid attempt not found on this run")
    return row


def relaunch(db: Session, row: PaidSubmitAttempt) -> None:
    """The owner kind's own gated relaunch, in the caller's transaction (the
    route functions it calls commit + enqueue themselves; a lost enqueue
    leaves a queued run that recovery re-enqueues, or a queued image job the
    derived reaper fails — retryable with the normal button)."""
    from . import api_v2

    if row.owner_kind == OWNER_RUN_SEGMENT:
        from .models import RunSegment

        rs = db.get(RunSegment, row.owner_id)
        if rs is None:
            raise paid_attempts.ActionRefused(409, "the segment no longer exists; abandon instead")
        run = api_v2._get_run_or_404(rs.run_id, db)
        api_v2._admit_segment_rerun(run, db)
        api_v2._check_segment_rerun_conflicts(run.id, db)
        api_v2._require_audio_patch_discard_ack(run, db, False)
        api_v2._budget_precheck(db, "resubmit parked segment",
                                needed=api_v2._segment_rerun_need(db, run, [rs]))
        api_v2._reset_segments_and_requeue(run, [rs], db, kind="resubmit parked segment")
        return
    if row.owner_kind == OWNER_RUN_PREPROCESS:
        rid, _, idx = row.owner_id.rpartition(":")
        run = api_v2._get_run_or_404(rid, db)
        if (run.preprocess_mode or "per_image") == "all_as_refs":
            api_v2.regenerate_preprocess(rid, db=db, preprocess_image_prompts=None,
                                         reference_files=[], reference_urls=None,
                                         preprocess_reference_file=None,
                                         preprocess_reference_url=None)
        else:
            api_v2.regenerate_preprocess_ref(rid, int(idx), db=db, prompt=None)
        return
    if row.owner_kind == OWNER_SEGMENT_IMAGE_JOB:
        from .models import RunSegmentImageJob

        job = db.get(RunSegmentImageJob, row.owner_id)
        if job is None:
            raise paid_attempts.ActionRefused(409, "the image job no longer exists; abandon instead")
        api_v2.regenerate_segment_image(job.run_id, job.run_segment_id, prompt=job.prompt,
                                        character_file=None, character_url=None,
                                        frame_sec=job.frame_sec, db=db)
        return
    raise paid_attempts.ActionRefused(409, f"no relaunch for {row.owner_kind}")


@router.get("/runs/{rid}/paid-attempts")
def list_paid_attempts(rid: str, db: Session = Depends(get_db)) -> list[dict]:
    return [paid_attempts.describe(r) for r in paid_attempts.run_open_attempts(db, rid)]


@router.post("/runs/{rid}/paid-attempts/{aid}/resubmit", dependencies=[GATED])
def resubmit_paid_attempt(
    rid: str, aid: str,
    confirm_checked_provider: bool = Form(False),
    actor: str = Form("dashboard"),
    note: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    if not confirm_checked_provider:
        raise HTTPException(status_code=400, detail=(
            "Confirm you checked the kie.ai dashboard first: the parked call may already be "
            "running and billed, and a resubmit can be charged a second time"))
    _lock_run(db, rid)
    _attempt_of_run(db, rid, aid)
    try:
        row = paid_attempts.resubmit(db, aid, actor=actor or "dashboard",
                                     note=note or "resubmit from the run page")
        relaunch(db, row)
    except paid_attempts.ActionRefused as exc:
        db.rollback()
        raise _http(exc) from None
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    log.warning("paid attempt %s resubmitted by %s (run %s)", aid, actor, rid)
    return paid_attempts.describe(db.get(PaidSubmitAttempt, aid))


@router.post("/runs/{rid}/paid-attempts/{aid}/abandon")
def abandon_paid_attempt(
    rid: str, aid: str,
    actor: str = Form("dashboard"),
    note: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    _lock_run(db, rid)
    _attempt_of_run(db, rid, aid)
    try:
        row = paid_attempts.abandon(db, aid, actor=actor or "dashboard",
                                    note=note or "abandoned from the run page")
    except paid_attempts.ActionRefused as exc:
        db.rollback()
        raise _http(exc) from None
    if row.owner_kind == OWNER_RUN_SEGMENT:
        from .models import RunSegment

        rs = db.get(RunSegment, row.owner_id)
        if rs is not None and rs.error_message and "parked" in rs.error_message:
            rs.error_message = "parked submit abandoned — re-run the segment if needed"
    db.commit()
    log.warning("paid attempt %s abandoned by %s (run %s)", aid, actor, rid)
    return paid_attempts.describe(row)
