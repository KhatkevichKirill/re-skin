"""Run-page context of an images-only (rebuild) run — design §10 "Run page".

Per chunk: its status, request vs chunk length, the rendered prompt, every
attempt (status, cost, links to the raw take, the fitted clip and the contact
sheet, QA flags, the character check), the re-roll and "use attempt N"
controls with their cost. Read-only: nothing here writes.
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import select

from . import ai_models
from . import spend as spend_mod
from .rebuild_models import (
    PAID_CALL_LIVE_STATUSES,
    PC_SENDING,
    PC_SUCCEEDED,
    PC_UNKNOWN,
    RebuildChunkAttempt,
    RebuildLlmCall,
)

# Run statuses in which a chunk may be re-rolled / an attempt re-selected.
IDLE_RUN_STATUSES = ("done", "failed", "incomplete")

_ATTEMPT_LABELS = {
    "queued": "queued", "claimed": "preparing", "sending": "sending",
    "submitted": "generating", "fetching": "downloading", "succeeded": "succeeded",
    "failed": "failed", "unknown": "unknown — may have billed",
}


def _status(value) -> str:
    return value.value if hasattr(value, "value") else str(value or "")


def _file_url(run_id: str, attempt_id: str, kind: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (f"/api/v2/runs/{run_id}/rebuild/attempts/{attempt_id}/{kind}"
            f"?v={int(st.st_mtime)}-{st.st_size}")


def _check_state(attempt, calls: list) -> dict:
    """What the run page says about an attempt's character check."""
    qa = attempt.qa or {}
    cc = qa.get("character_check")
    if isinstance(cc, dict):
        return {"state": "flagged" if cc.get("flagged") else "ok", "result": cc}
    if cc == "unavailable":
        return {"state": "unavailable", "result": None}
    if any(c.status in PAID_CALL_LIVE_STATUSES for c in calls):
        return {"state": "running", "result": None}
    if attempt.status == PC_SUCCEEDED:
        return {"state": "pending", "result": None}
    return {"state": "none", "result": None}


def chunk_cards(run, db) -> list[dict]:
    """One dict per chunk of the run's snapshot, in timeline order."""
    from .config import settings
    from .models import RunSegment

    snap = run.rebuild_snapshot or {}
    rs_by_def = {rs.segment_def_id: rs for rs in db.execute(
        select(RunSegment).where(RunSegment.run_id == run.id)).scalars()}
    attempts: dict[str, list] = {}
    for a in db.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == run.id)
            .order_by(RebuildChunkAttempt.attempt_no.desc())).scalars():
        attempts.setdefault(a.run_segment_id, []).append(a)
    ids = [a.id for lst in attempts.values() for a in lst]
    calls: dict[str, list] = {}
    if ids:
        for c in db.execute(select(RebuildLlmCall).where(
                RebuildLlmCall.attempt_id.in_(ids),
                RebuildLlmCall.kind == "character_check")
                .order_by(RebuildLlmCall.slot)).scalars():
            calls.setdefault(c.attempt_id, []).append(c)
    run_idle = _status(run.status) in IDLE_RUN_STATUSES
    any_live = any(a.status in PAID_CALL_LIVE_STATUSES
                   for lst in attempts.values() for a in lst)
    enabled = bool(settings.REBUILD_ENABLED)
    cards = []
    for ch in sorted(snap.get("chunks") or [], key=lambda c: int(c.get("index", 0))):
        rs = rs_by_def.get(ch.get("segment_def_id"))
        rows = []
        for a in attempts.get(rs.id, []) if rs is not None else []:
            qa = a.qa or {}
            rows.append({
                "id": a.id, "n": a.attempt_no, "status": a.status,
                "label": _ATTEMPT_LABELS.get(a.status, a.status),
                "live": a.status in PAID_CALL_LIVE_STATUSES,
                "waiting_sender": a.status == PC_SENDING and a.not_live_since is not None,
                "may_have_billed": a.status == PC_UNKNOWN,
                "selected": bool(a.selected), "error": a.error,
                "estimate": a.estimate_credits, "cost": a.cost_credits,
                "request_sec": a.request_sec, "chunk_frames": a.chunk_frames,
                "raw_url": _file_url(run.id, a.id, "raw", a.raw_path),
                "fitted_url": _file_url(run.id, a.id, "fitted", a.fitted_path),
                "sheet_url": _file_url(run.id, a.id, "sheet", qa.get("contact_sheet")),
                "qa": qa, "flags": list(qa.get("flags") or []),
                "cut_timing": qa.get("cut_timing"), "duration": qa.get("duration"),
                "text_band": qa.get("text_band"),
                "check": _check_state(a, calls.get(a.id, [])),
                "usable": (a.status == PC_SUCCEEDED and not a.selected and a.fitted_path
                           and os.path.exists(a.fitted_path)),
            })
        selected = next((r for r in rows if r["selected"]), None)
        # What the books hold for the chunk: the provider's cost when known,
        # else the estimate drawn at `sending` (nothing before it).
        spent = sum(float(r["cost"]) if r["cost"] is not None else
                    (0.0 if r["status"] in ("queued", "claimed") else float(r["estimate"] or 0))
                    for r in rows)
        reroll_cost = spend_mod.images_only_video_credits(
            run.model, run.resolution, int(ch.get("request_sec") or 0)) + 2 * spend_mod.llm_credits()
        cards.append({
            "index": int(ch.get("index", 0)),
            "segment_def_id": ch.get("segment_def_id"),
            "rs": rs, "rs_id": rs.id if rs is not None else None,
            "status": _status(rs.status) if rs is not None else "not started",
            "error": rs.error_message if rs is not None else None,
            "start_sec": float(ch.get("start_sec") or 0.0),
            "end_sec": float(ch.get("end_sec") or 0.0),
            "chunk_sec": float(ch.get("end_sec") or 0.0) - float(ch.get("start_sec") or 0.0),
            "request_sec": int(ch.get("request_sec") or 0),
            "shots": len(ch.get("shots") or []),
            "sheet_key": ch.get("sheet_key"),
            "prompt": ch.get("prompt"),
            "prompt_override": rs.prompt_override if rs is not None else None,
            "attempts": rows, "selected": selected,
            "flags": list(selected["flags"]) if selected else [],
            "spent": round(spent, 2), "reroll_cost": reroll_cost,
            "can_reroll": enabled and run_idle and not any_live and rs is not None,
            "can_use": run_idle and not any_live,
        })
    return cards


def run_context(run, db) -> dict:
    """``rebuild_*`` keys of the run page / status fragment (empty for v2v)."""
    from .config import settings

    if not spend_mod.is_images_only_run(run):
        return {"rebuild_chunks": None}
    cards = chunk_cards(run, db)
    snap = run.rebuild_snapshot or {}
    return {
        "rebuild_chunks": cards,
        "rebuild_enabled": bool(settings.REBUILD_ENABLED),
        "rebuild_model_label": ai_models.spec_for(run.model).label,
        "rebuild_resolution": run.resolution,
        "rebuild_spent": round(sum(c["spent"] for c in cards), 2),
        "rebuild_fps_out": snap.get("fps_out"),
        "rebuild_any_live": any(r["live"] for c in cards for r in c["attempts"]),
        "check_cost": spend_mod.llm_credits(),
    }
