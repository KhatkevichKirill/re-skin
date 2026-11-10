"""The deploy hold (docs/autopilot-design.md §2.8, §9).

``hold_on`` flips the singleton ``autopilot_gate`` row with an UPDATE. Every
admitting autopilot transaction holds that row ``FOR SHARE`` until its commit,
so on Postgres the UPDATE waits for every admission in progress, and every
later one sees ``held``: when ``hold_on`` returns, no autopilot admission is in
progress and none can start.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from ..models_autopilot import AutopilotGate, utcnow


def _row(db, *, lock: bool) -> AutopilotGate:
    q = select(AutopilotGate).where(AutopilotGate.id == 1)
    if lock:
        q = q.with_for_update()
    g = db.execute(q).scalar_one_or_none()
    if g is None:
        g = AutopilotGate(id=1, held=False, generation=0)
        db.add(g)
        db.flush()
    return g


def hold_on(db, *, by: str = "operator", note: Optional[str] = None) -> AutopilotGate:
    g = _row(db, lock=True)
    if not g.held:
        g.held, g.held_since, g.held_by, g.note = True, utcnow(), by[:128], note
        g.generation = (g.generation or 0) + 1
    db.commit()
    return g


def hold_off(db) -> AutopilotGate:
    g = _row(db, lock=True)
    g.held, g.held_since, g.held_by, g.note = False, None, None, None
    db.commit()
    try:
        from .core import wake

        wake()
    except Exception:
        pass
    return g


def status(db) -> dict:
    g = db.get(AutopilotGate, 1)
    if g is None:
        return {"held": False, "generation": 0}
    return {"held": bool(g.held), "generation": g.generation,
            "held_since": g.held_since.isoformat() if g.held_since else None,
            "held_by": g.held_by, "note": g.note}
