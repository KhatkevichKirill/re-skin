"""The "Waiting for you" inbox and header badge (AICP-375).

Design: ``docs/autopilot-operator-ux.md`` §3. Read-only. The count is
:func:`links.answerable` — the same rule the cards answer by — so the badge
never counts a question the cards would refuse (a stopped, failed, completed or
replaced attempt's question).

* **Waiting for you · N** — answerable questions (a paused attempt's too: the
  operator answers, then resumes).
* **N paused** — latest attempts paused by the SYSTEM (status_reason not
  "paused by …"); a separate marker, never added to N.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import VideoProject
from ..models_autopilot import AutopilotJournal, ProjectAutopilot, utcnow
from . import links

ENDED_WINDOW = timedelta(days=7)
OPERATOR_PAUSE = "paused by "
OPERATOR_STOP = "stopped by "


def _latest_attempts(db: Session, *statuses: str):
    """Each project's latest attempt, when its status is one of ``statuses``."""
    return (select(ProjectAutopilot)
            .where(ProjectAutopilot.status.in_(statuses),
                   ProjectAutopilot.id == links._latest_attempt_id_of(ProjectAutopilot.project_id)))


def _system_paused(ap: ProjectAutopilot) -> bool:
    return not (ap.status_reason or "").startswith(OPERATOR_PAUSE)


@dataclass(frozen=True)
class Badge:
    waiting: int
    paused: int


def badge(db: Session) -> Badge:
    paused = [ap for ap in db.execute(_latest_attempts(db, "paused")).scalars() if _system_paused(ap)]
    return Badge(waiting=len(links.answerable(db)), paused=len(paused))


def _waited(delta: timedelta) -> str:
    mins = max(0, int(delta.total_seconds() // 60))
    if mins < 60:
        return f"{mins} min"
    hours, mins = divmod(mins, 60)
    if hours < 48:
        return f"{hours} h {mins:02d} min"
    return f"{hours // 24} days"


def _cost_mcredits(a) -> tuple[int, bool]:
    """``(mcredits, is_upper_bound)``: the dearest of the approval's own cost
    and its options' costs (a "Regenerate" or "Retry" option can cost more than
    the proposal itself); an upper bound whenever an option is the dearest."""
    own = int(a.cost_mcredits or 0)
    opts = [int(o.get("cost_mcredits") or 0) for o in (a.options or []) if isinstance(o, dict)]
    top = max(opts) if opts else 0
    return (top, True) if top > own else (own, False)


def _paused_since(db: Session, aps) -> dict:
    """When each attempt's CURRENT pause began: the first "paused" journal line
    carrying its status_reason (journal messages are cut at 4000 characters)
    after its latest "resumed" line. Not ``updated_at`` (moves on every answer),
    and not a later "paused" line a manual paid action writes while the attempt
    is already paused — even an identical one."""
    out = {}
    for ap in aps:
        resumed = db.execute(select(func.max(AutopilotJournal.at)).where(
            AutopilotJournal.autopilot_id == ap.id, AutopilotJournal.kind == "resumed")).scalar_one_or_none()
        q = select(func.min(AutopilotJournal.at)).where(
            AutopilotJournal.autopilot_id == ap.id, AutopilotJournal.kind == "paused",
            AutopilotJournal.message == (ap.status_reason or "")[:4000])
        if resumed is not None:
            q = q.where(AutopilotJournal.at > resumed)
        out[ap.id] = db.execute(q).scalar_one_or_none()
    return out


def _names(db: Session, project_ids) -> dict:
    ids = list(set(project_ids))
    if not ids:
        return {}
    return {pid: (name or f"project {pid[:8]}") for pid, name in db.execute(
        select(VideoProject.id, VideoProject.name).where(VideoProject.id.in_(ids)))}


def view(db: Session, now: Optional[datetime] = None) -> dict:
    """The sections of ``/v2/assistant``."""
    from .. import spend
    from . import notify, run_view

    now = now or utcnow()
    pairs = links.answerable(db)
    paused_aps = list(db.execute(_latest_attempts(db, "paused")
                                 .order_by(ProjectAutopilot.updated_at)).scalars())
    ended_aps = [ap for ap in db.execute(
        _latest_attempts(db, "failed", "stopped")
        .where(ProjectAutopilot.finished_at >= now - ENDED_WINDOW)
        .order_by(ProjectAutopilot.finished_at.desc())).scalars()
        if not (ap.status == "stopped" and (ap.status_reason or "").startswith(OPERATOR_STOP))]
    failing = notify.failing(db)
    names = _names(db, [ap.project_id for _, ap in pairs] + [ap.project_id for ap in paused_aps]
                   + [ap.project_id for ap in ended_aps] + [n.project_id for n in failing if n.project_id])

    to_answer = []
    prs: dict = {}
    for a, ap in pairs:
        pr = prs.get(ap.project_id) or prs.setdefault(ap.project_id, run_view.project_runs(db, ap.project_id))
        pl = links.place(db, a, project_id=ap.project_id, pr=pr, latest=ap)
        mc, upto = _cost_mcredits(a)
        credits = mc / 1000.0
        to_answer.append({
            "id": a.id, "kind": a.kind, "title": links.approval_title(a.question, a.proposal),
            "project_id": ap.project_id, "project": names.get(ap.project_id, ap.project_id[:8]),
            "where": pl.where, "path": pl.path, "created_at": a.created_at,
            "waited": _waited(now - a.created_at),
            "cost": ({"credits": credits, "usd": round(spend.usd(credits), 2), "upto": upto}
                     if credits else None),
            "paused": ap.status == "paused",
        })

    def _attempt(ap: ProjectAutopilot, at: Optional[datetime]) -> dict:
        return {"project_id": ap.project_id, "project": names.get(ap.project_id, ap.project_id[:8]),
                "status": ap.status, "reason": ap.status_reason or "", "flow": ap.flow_id,
                "at": at, "ago": _waited(now - at) if at else "",
                "path": f"/v2/projects/{ap.project_id}"}

    since = _paused_since(db, paused_aps)
    paused = [dict(_attempt(ap, since.get(ap.id) or ap.updated_at), by_operator=not _system_paused(ap))
              for ap in paused_aps]
    paused.sort(key=lambda r: r["by_operator"])          # system pauses first
    return {
        "to_answer": to_answer,
        "paused": paused,
        "ended": [_attempt(ap, ap.finished_at) for ap in ended_aps],
        "failing": [{"key": n.key, "kind": n.kind, "project_id": n.project_id,
                     "project": names.get(n.project_id, "") if n.project_id else "",
                     "attempts": n.attempts, "error": n.last_error or "",
                     "path": f"/v2/projects/{n.project_id}" if n.project_id else None}
                    for n in failing],
        "notifications": notify.config_problem() or "on",
        "badge": Badge(waiting=len(to_answer), paused=sum(1 for p in paused if not p["by_operator"])),
    }
