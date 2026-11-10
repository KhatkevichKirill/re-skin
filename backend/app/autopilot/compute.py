"""Free compute effects (docs/autopilot-design.md §5.5).

``run(effect_id)`` is the RQ job on the ``autopilot_compute`` queue (served by
the run workers). It claims the effect (autopilot row lock first, then the
effect — the lock order of §5.2), runs the registered function into a
per-attempt directory and publishes the result only while its claim is still
the effect's. A duplicate attempt is waste, never harm.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import timedelta
from typing import Callable, Optional

from sqlalchemy import select

from . import registry
from ..models_autopilot import AutopilotEffect, AutopilotJournal, ProjectAutopilot, utcnow

log = logging.getLogger(__name__)

CLAIM_GRACE_SEC = 300


def attempt_dir(project_id: str, effect_id: str, token: str) -> str:
    from ..storage import project_dir

    return os.path.join(project_dir(project_id), "autopilot", "compute", effect_id, token)


def _lock(db, effect_id: str):
    eff = db.get(AutopilotEffect, effect_id)
    if eff is None:
        return None, None
    ap = db.execute(
        select(ProjectAutopilot).where(ProjectAutopilot.id == eff.autopilot_id).with_for_update()
    ).scalar_one_or_none()
    db.refresh(eff, with_for_update=True)
    return ap, eff


def run(effect_id: str, *, session_factory: Optional[Callable] = None) -> Optional[str]:
    """RQ entry point. Returns the final state it wrote (or None)."""
    if session_factory is None:
        from ..db import SessionLocal as session_factory  # noqa: N813
    token = uuid.uuid4().hex
    db = session_factory()
    try:
        ap, eff = _lock(db, effect_id)
        if eff is None or eff.kind != "compute" or ap is None:
            db.rollback()
            return None
        now = utcnow()
        claimable = eff.state == "queued" or (
            eff.state == "running" and eff.claim_expires_at is not None
            and eff.claim_expires_at < now
        )
        func = registry.get_compute(eff.handler, eff.handler_version)
        if not claimable or func is None:
            if func is None:
                log.warning("autopilot compute %s: %s v%s is not registered in this "
                            "image — not claimed", effect_id, eff.handler, eff.handler_version)
            db.rollback()
            return None
        timeout = int((eff.params or {}).get("timeout_sec") or 1800)
        eff.state, eff.claim_token = "running", token
        eff.attempts = (eff.attempts or 0) + 1
        eff.claim_expires_at = now + timedelta(seconds=timeout + CLAIM_GRACE_SEC)
        eff.not_live_since = None
        eff.updated_at = now
        ap.version += 1
        project_id = ap.project_id
        args = dict((eff.params or {}).get("args") or {})
        db.commit()
    finally:
        db.close()

    out = attempt_dir(project_id, effect_id, token)
    os.makedirs(out, exist_ok=True)
    state, result, error = "succeeded", None, None
    try:
        result = dict(func(args, out) or {})
        result["out_dir"] = out
    except Exception as exc:  # the step decides what a failed compute means
        log.exception("autopilot compute %s failed", effect_id)
        state, error = "failed", f"{type(exc).__name__}: {exc}"[:2000]

    db = session_factory()
    try:
        ap, eff = _lock(db, effect_id)
        if eff is None or eff.claim_token != token or eff.state != "running":
            db.rollback()
            return None  # another attempt owns it now
        eff.state, eff.result, eff.error = state, result, error
        eff.updated_at = utcnow()
        ap.version += 1
        db.add(AutopilotJournal(
            autopilot_id=ap.id, step_id=eff.step_id, kind="observed", effect_id=eff.id,
            message=f"compute {eff.name} ({eff.handler}) {state}", at=utcnow(),
            data={"error": error} if error else None,
        ))
        db.commit()
        return state
    finally:
        db.close()
