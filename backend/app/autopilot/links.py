"""Shared pieces of the "by Assistant" operator UX (AICP-373).

Design: ``docs/autopilot-operator-ux.md`` §2.2 (links), §3 (answerable), §4.2
(titles). Read-only. Slack messages (AICP-374), the "Waiting for you" inbox
(AICP-375) and the cards (AICP-376) all have to agree on four things, so they
are computed here and nowhere else:

* **Where a question lives and how to link to it** — :func:`place`. The page is
  the AICP-369 placement (``run_view.approval_runs``): one of the assistant's
  runs, else the project. The link goes to that run only while the approval's
  attempt is the project's latest one (the run card answers nothing for a
  replaced attempt); otherwise, and when the run is gone, to the project page.
  Every link carries ``#ap-<approval id>``, the id of the approval's element on
  the card it lands on.
* **Its title** — :func:`approval_title`: ``proposal["_title"]`` once flows
  set it (AICP-376), else the first sentence of the question, at most
  :data:`TITLE_MAX` characters.
* **Whether it can still be answered** — :func:`answerable`: ``pending``
  approvals of each project's LATEST attempt while that attempt is live
  (active or paused). The same rule the cards use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from ..config import settings
from ..models import Run
from ..models_autopilot import LIVE_AUTOPILOT_STATUSES, AutopilotApproval, ProjectAutopilot

TITLE_MAX = 80
CONSEQUENCE_MAX = 120
TITLE_KEY = "_title"          # reserved proposal key (doc §4.2); flows may not set it themselves
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


def clip_text(text: str, limit: int = TITLE_MAX) -> str:
    """Whitespace collapsed, cut at a word to ``limit`` characters ("…")."""
    return _clip(" ".join((text or "").split()), limit) if text else ""


def _clip(text: str, limit: int = TITLE_MAX) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-—–") + "…"


def approval_title(question: Optional[str], proposal=None) -> str:
    """The approval's short title: the stored one, else the question's first
    sentence (whitespace collapsed), at most ``TITLE_MAX`` characters."""
    stored = proposal.get(TITLE_KEY) if isinstance(proposal, dict) else None
    if isinstance(stored, str) and stored.strip():
        return _clip(" ".join(stored.split()))
    q = " ".join((question or "").split())
    if not q:
        return "Question"
    return _clip(_SENTENCE_END.split(q, 1)[0])


# ---------------------------------------------------------------------------
# The latest attempt and answerable approvals
# ---------------------------------------------------------------------------


def latest_attempt(db: Session, project_id: str) -> Optional[ProjectAutopilot]:
    """The project's latest autopilot attempt (what the project card shows)."""
    return db.execute(select(ProjectAutopilot).where(ProjectAutopilot.project_id == project_id)
                      .order_by(ProjectAutopilot.created_at.desc(), ProjectAutopilot.id.desc())
                      .limit(1)).scalar_one_or_none()


def _latest_attempt_id_of(project_id_col):
    other = aliased(ProjectAutopilot)
    return (select(other.id).where(other.project_id == project_id_col)
            .order_by(other.created_at.desc(), other.id.desc()).limit(1)
            .correlate(ProjectAutopilot).scalar_subquery())


def answerable(db: Session, *, project_id: Optional[str] = None
               ) -> list[tuple[AutopilotApproval, ProjectAutopilot]]:
    """``(approval, attempt)`` for every approval the operator can answer now,
    oldest first: ``pending``, of its project's latest attempt, that attempt
    live. A paused attempt's approvals are included (answer, then Resume);
    callers that count "paused" apart read ``attempt.status``."""
    q = (select(AutopilotApproval, ProjectAutopilot)
         .join(ProjectAutopilot, ProjectAutopilot.id == AutopilotApproval.autopilot_id)
         .where(AutopilotApproval.status == "pending",
                ProjectAutopilot.status.in_(LIVE_AUTOPILOT_STATUSES),
                ProjectAutopilot.id == _latest_attempt_id_of(ProjectAutopilot.project_id)))
    if project_id is not None:
        q = q.where(ProjectAutopilot.project_id == project_id)
    q = q.order_by(AutopilotApproval.created_at, AutopilotApproval.id)
    return [(a, ap) for a, ap in db.execute(q).all()]


def is_answerable(a: AutopilotApproval, latest: Optional[ProjectAutopilot]) -> bool:
    return (a.status == "pending" and latest is not None and latest.id == a.autopilot_id
            and latest.status in LIVE_AUTOPILOT_STATUSES)


# ---------------------------------------------------------------------------
# Placement and links
# ---------------------------------------------------------------------------


def anchor(approval_id: str) -> str:
    """The element id of an approval on its card (and the links' fragment)."""
    return f"ap-{approval_id}"


def absolute(path: str) -> Optional[str]:
    """``PUBLIC_BASE_URL`` + path, or None when no public base is configured."""
    base = (settings.PUBLIC_BASE_URL or "").rstrip("/")
    return f"{base}{path}" if base else None


@dataclass(frozen=True)
class Place:
    approval_id: str
    project_id: str
    run_id: Optional[str]           # the run whose page shows it (AICP-369); None = project queue
    candidates: tuple               # runs it concerns when it is in the project queue
    reason: str
    short_id: Optional[str]         # of run_id
    run_label: Optional[str]        # the assistant's label of run_id ("main", "final ES" …)
    latest: bool                    # its attempt is the project's latest one
    open: bool                      # answerable now (see answerable())
    path: str                       # where the operator answers it, with #ap-<id>

    @property
    def url(self) -> Optional[str]:
        return absolute(self.path)

    @property
    def where(self) -> str:
        """``run <short id> (<label>)`` or ``project`` — for messages and rows."""
        if self.run_id and self.latest:
            label = f" ({self.run_label})" if self.run_label else ""
            return f"run {self.short_id or self.run_id[:8]}{label}"
        return "project"


def place(db: Session, a: AutopilotApproval, *, project_id: Optional[str] = None,
          pr=None, latest: Optional[ProjectAutopilot] = None, runs: Optional[dict] = None) -> Place:
    """Where approval ``a`` is shown and the link to it.

    ``pr`` (``run_view.project_runs``), ``latest`` (:func:`latest_attempt`) and
    ``runs`` (``{run_id: Run}``) may be passed by a caller placing several
    approvals of one project; they are read here otherwise."""
    from . import run_view

    if project_id is None:
        ap = db.get(ProjectAutopilot, a.autopilot_id)
        project_id = ap.project_id
    if pr is None:
        pr = run_view.project_runs(db, project_id)
    if latest is None:
        latest = latest_attempt(db, project_id)
    run_id, cands, reason = run_view.approval_runs(a, pr)
    short_id = label = None
    if run_id:
        r = (runs or {}).get(run_id) or db.get(Run, run_id)
        if r is None:   # deleted since project_runs read it
            run_id, reason = None, "its run was deleted"
        else:
            short_id = r.short_id
            ar = pr.by_run.get(run_id)
            label = ar.label if ar else None
    is_latest = latest is not None and latest.id == a.autopilot_id
    if run_id and is_latest:
        path = f"/v2/runs/{run_id}#{anchor(a.id)}"
    else:
        path = f"/v2/projects/{project_id}#{anchor(a.id)}"
    return Place(approval_id=a.id, project_id=project_id, run_id=run_id,
                 candidates=tuple(cands), reason=reason, short_id=short_id, run_label=label,
                 latest=is_latest, open=is_answerable(a, latest), path=path)
