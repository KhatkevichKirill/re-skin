"""Run-level endpoints of an images-only (rebuild) run — design §6.5–6.6, §10.

* ``POST /runs/{rid}/rebuild/segments/{rsid}/use-attempt/{aid}`` — "use
  attempt N": re-select an older succeeded attempt of a chunk and re-stitch
  (replaces AICP-342's take restore for this mode; free).
* ``POST /runs/{rid}/rebuild/attempts/{aid}/character-check`` — the
  operator's "Run character check" after the automatic one ended
  ``unavailable``: a NEW, separately reserved call (slot >= 2).
* ``GET /runs/{rid}/rebuild/attempts/{aid}/{raw|fitted|sheet}`` — the raw
  take, the fitted clip and the contact sheet (immutable files).

Re-rolling a chunk is the existing segment rerun route
(``POST /runs/{rid}/segments/{rsid}/rerun``), which branches on the run's
generation mode; Retry Run likewise.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import pipeline_rebuild
from . import rebuild_paid_call as pc
from . import spend as spend_mod
from .accel import accel_or_file_response
from .db import get_db, get_session
from .ops_gate import GATED
from .models import Run, RunSegment
from .rebuild_models import PAID_CALL_LIVE_STATUSES, RebuildChunkAttempt, RebuildLlmCall
from .schemas_v2 import RunResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["rebuild-runs"])

_FILES = {"raw": ("raw_path", "video/mp4"), "fitted": ("fitted_path", "video/mp4"),
          "sheet": (None, "image/jpeg")}


def _images_only_run_or_404(rid: str, db: Session) -> Run:
    run = db.get(Run, rid)
    if run is None or not spend_mod.is_images_only_run(run):
        raise HTTPException(status_code=404, detail=f"images-only run {rid!r} not found")
    return run


@router.post("/runs/{rid}/rebuild/segments/{rsid}/use-attempt/{aid}",
             response_model=RunResponse, dependencies=[GATED])
def use_attempt(
    rid: str,
    rsid: str,
    aid: str,
    discard_audio_patch: bool = Form(False),
    db: Session = Depends(get_db),
) -> RunResponse:
    """Make attempt *aid* the chunk's result again and re-stitch the run (the
    delivered final is replaced in place, as for a rerun)."""
    from .api_v2 import (
        _admit_segment_rerun,
        _check_segment_rerun_conflicts,
        _invalidate_final_and_requeue,
        _require_audio_patch_discard_ack,
    )

    run = _images_only_run_or_404(rid, db)
    _admit_segment_rerun(run, db)
    rs = db.get(RunSegment, rsid)
    if rs is None or rs.run_id != rid:
        raise HTTPException(status_code=404, detail="segment not found in this run")
    attempt = db.get(RebuildChunkAttempt, aid)
    if attempt is None:
        raise HTTPException(status_code=404, detail="attempt not found")
    _check_segment_rerun_conflicts(rid, db)
    _require_audio_patch_discard_ack(run, db, discard_audio_patch)
    try:
        pipeline_rebuild.use_attempt(db, run, rs, attempt)
    except pipeline_rebuild.RepairRefused as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    _invalidate_final_and_requeue(run, db, kind="use attempt")
    log.info("Run %s chunk %s now uses attempt %s (#%s)", rid, rs.index, aid, attempt.attempt_no)
    return RunResponse.model_validate(run)


@router.post("/runs/{rid}/rebuild/attempts/{aid}/character-check", status_code=202, dependencies=[GATED])
def run_character_check(rid: str, aid: str, db: Session = Depends(get_db)) -> dict:
    """A new character check of a succeeded attempt (its automatic check
    ended ``unavailable``): one more paid call, reserved here."""
    from .api_v2 import _require_rebuild_enabled_for_repair, _spend_reserve

    run = _images_only_run_or_404(rid, db)
    _require_rebuild_enabled_for_repair(run)
    db.refresh(run, with_for_update=True)
    attempt = db.get(RebuildChunkAttempt, aid)
    if attempt is None or attempt.run_id != rid:
        raise HTTPException(status_code=404, detail="attempt not found")
    if attempt.status != "succeeded" or not (attempt.qa or {}).get("contact_sheet"):
        raise HTTPException(status_code=409, detail="the attempt has no contact sheet to check")
    live = db.execute(select(func.count()).select_from(RebuildLlmCall).where(
        RebuildLlmCall.attempt_id == aid, RebuildLlmCall.kind == "character_check",
        RebuildLlmCall.status.in_(PAID_CALL_LIVE_STATUSES))).scalar()
    if live:
        raise HTTPException(status_code=409, detail="a character check of this attempt is running")
    # Slots 0/1 are the automatic check and its fallback; an operator's
    # check takes the next slot >= 2 (each slot used once, by index).
    used = db.execute(select(func.max(RebuildLlmCall.slot)).where(
        RebuildLlmCall.attempt_id == aid, RebuildLlmCall.kind == "character_check")).scalar()
    slot = max(int(used or 0) + 1, max(spend_mod.CHECK_AUTO_SLOTS) + 1)
    _spend_reserve(spend_mod.reserve_computed, db, spend_mod.run_scope(rid),
                   lambda _s: spend_mod.llm_credits(), kind="character check", run_id=rid)
    call = RebuildLlmCall(project_id=run.project_id, attempt_id=aid, kind="character_check",
                          slot=slot, model=pipeline_rebuild.check_models()[0],
                          status="queued", estimate_credits=spend_mod.llm_credits())
    db.add(call)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409,
                            detail="a character check of this attempt is running") from exc
    qa = dict(attempt.qa or {})
    qa.pop("character_check", None)
    qa["flags"] = pipeline_rebuild.qa_flags(qa)
    attempt.qa = qa
    db.commit()
    pc.enqueue_step(pipeline_rebuild.LLM_TABLE, call.id, 1, pc.STEP_CLAIM)
    return {"call_id": call.id, "slot": slot}


@router.get("/runs/{rid}/rebuild/attempts/{aid}/{kind}")
def download_attempt_file(rid: str, aid: str, kind: str):
    """The raw take, the fitted clip or the contact sheet of one attempt."""
    if kind not in _FILES:
        raise HTTPException(status_code=404, detail="unknown file")
    with get_session() as db:
        run = _images_only_run_or_404(rid, db)
        a = db.get(RebuildChunkAttempt, aid)
        if a is None or a.run_id != run.id:
            raise HTTPException(status_code=404, detail="attempt not found")
        col, media_type = _FILES[kind]
        path = getattr(a, col) if col else (a.qa or {}).get("contact_sheet")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="the file is not on disk")
    return accel_or_file_response(
        path, media_type=media_type,
        extra_headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )
