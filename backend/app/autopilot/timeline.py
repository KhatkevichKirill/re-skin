"""The journal as a timeline (AICP-377, ``docs/autopilot-operator-ux.md`` §6).

Newest first, filtered SERVER-SIDE (default: the important kinds) and paged
with a cursor on ``autopilot_journal.id``, so an important line older than
the newest page is still reachable. Read-only.

On a Run page the lines are those ``run_view.run_journal_scope`` gives: the
run's own effects and approvals exactly, plus its steps' step-only lines
inside the run's time window. Journal rows carry no step generation, so those
step-only lines are marked ``during`` ("during this run"), not claimed as
exact (``docs/autopilot-run-view.md``).
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ..models_autopilot import AutopilotJournal

# Shown by default. Hidden: "decision" (its latest "waiting:" line is the
# status line), "observed" (admissions / reconciles), "step_started".
IMPORTANT_KINDS = (
    "created", "approval_requested", "approval_answered", "launch", "budget", "paused",
    "resumed", "stopped", "failed", "completed", "reopened", "refused", "warning", "comment",
)

PAGE = 30
MAX_PAGE = 200


def page(db: Session, autopilot_id: str, *, important: bool = True,
         before: Optional[int] = None, limit: int = PAGE, scope=None,
         mark_during: bool = False) -> dict:
    """``{"entries", "next_before"}``: one page, newest first.

    ``scope`` is an extra SQL condition (the Run page's); with ``mark_during``
    a line that is in it only by its step and time window (no effect, no
    approval) is marked ``during``."""
    limit = max(1, min(int(limit or PAGE), MAX_PAGE))
    conds = [AutopilotJournal.autopilot_id == autopilot_id]
    if scope is not None:
        conds.append(scope)
    if important:
        conds.append(AutopilotJournal.kind.in_(IMPORTANT_KINDS))
    if before:
        conds.append(AutopilotJournal.id < int(before))
    rows = db.execute(select(AutopilotJournal).where(and_(*conds))
                      .order_by(AutopilotJournal.id.desc()).limit(limit + 1)).scalars().all()
    more = len(rows) > limit
    rows = rows[:limit]
    entries = []
    for j in rows:
        e = {"id": j.id, "at": j.at.isoformat(), "kind": j.kind, "step_id": j.step_id,
             "message": j.message, "cost_credits": (j.cost_mcredits / 1000.0) if j.cost_mcredits else None,
             "during": False}
        if mark_during:
            e["during"] = j.effect_id is None and j.approval_id is None
        entries.append(e)
    return {"entries": entries, "next_before": rows[-1].id if (more and rows) else None}
