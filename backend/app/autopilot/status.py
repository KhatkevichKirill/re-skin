"""What the autopilot is doing, in one line, a stepper, a budget bar and a pause
banner (AICP-377, ``docs/autopilot-operator-ux.md`` §5).

Read-only presentation: everything here is derived from the rows the core
already writes (``project_autopilots.status/status_reason``, the step rows'
``status/reason``, the pending approvals, ``core.budget_view``) and nothing
changes an autopilot. The journal timeline is ``timeline.py``.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models_autopilot import AutopilotJournal

# Short names of the flows' steps (subtitle_removal.py / localisation.py /
# localisation_finish.py step ids, the part before ":<language>").
STEP_LABELS = {
    "segmentation": "Segmentation", "launch": "Launch", "clean": "Cleaning", "handoff": "Handoff",
    "intake": "Intake", "transcribe": "Transcribe", "segments": "Segments",
    "breakdown": "Breakdown", "translate": "Translate", "prompts": "Prompts", "probe": "Probe",
    "rest": "Rest", "qa": "QA", "captions": "Captions", "tail": "Tail", "cleanup": "Cleanup",
}

# What a stepper cell says about a step's state.
_STATE = {"completed": "done", "active": "active", "failed": "failed", "skipped": "skipped",
          "pending": "pending"}


def step_label(step_id: str, *, with_language: bool = True) -> str:
    base, _, lang = (step_id or "").partition(":")
    label = STEP_LABELS.get(base, base.replace("_", " ").capitalize() or step_id)
    return f"{label} ({lang.upper()})" if lang and with_language else label


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _step_reason(reason: Optional[str]) -> str:
    r = (reason or "").strip()
    if r.startswith("refused: "):
        return "refused, retrying later: " + r[len("refused: "):]
    if r.startswith("error: "):
        return "error, retrying later: " + r[len("error: "):]
    return r


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------


def status_line(status: str, reason: Optional[str], steps: Iterable[dict], pending: int) -> dict:
    """``{"tone", "text"}``: the one line that says what the attempt is doing.

    ``steps`` are ``core.step_dict`` rows; ``pending`` is the number of pending
    approvals of this attempt."""
    reason = (reason or "").strip()
    if status == "completed":
        return {"tone": "done", "text": "Completed" + (f": {reason}" if reason else "")}
    if status == "stopped":
        return {"tone": "ended", "text": "Stopped" + (f": {reason}" if reason else "")}
    if status == "failed":
        return {"tone": "ended", "text": "Failed" + (f": {reason}" if reason else "")}
    if status == "paused":
        return {"tone": "paused", "text": "Paused" + (f": {reason}" if reason else "")}
    if pending:
        return {"tone": "you", "text": f"Waiting for you: {_plural(pending, 'question')}"}
    active = [s for s in steps if s.get("status") == "active"]
    if not active:
        return {"tone": "work", "text": "Working: deciding the next step"}
    parts = []
    for s in active[:3]:
        r = _step_reason(s.get("reason"))
        parts.append(f"{step_label(s['step_id'])} — {r}" if r else step_label(s["step_id"]))
    more = f" (+{len(active) - 3} more)" if len(active) > 3 else ""
    return {"tone": "work", "text": "; ".join(parts) + more}


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


def stepper(spec_ids: Optional[list], steps: Iterable[dict], waiting_steps: Iterable[str],
            *, only: Optional[Iterable[str]] = None) -> list[dict]:
    """Rows of cells, in the flow's order: the steps without a language in one
    row, then one row per language (localisation: Translate → … → Cleanup).

    ``spec_ids`` is the flow's declared order (``flow.steps(...)``); without it
    (flow no longer deployed) the step rows' own start order is used.
    ``waiting_steps`` are the step ids with a pending approval. ``only`` keeps
    just those step ids (the Run page: the steps that work on that run)."""
    rows_by_id = {s["step_id"]: s for s in steps}
    order = list(spec_ids) if spec_ids else list(rows_by_id)
    order += [sid for sid in rows_by_id if sid not in order]       # a step the spec no longer has
    if only is not None:
        keep = set(only)
        order = [sid for sid in order if sid in keep]
    waiting = set(waiting_steps)
    groups: dict = {}
    for sid in order:
        _base, _, lang = sid.partition(":")
        s = rows_by_id.get(sid)
        state = _STATE.get(s["status"], s["status"]) if s else "pending"
        if sid in waiting and state == "active":
            state = "you"
        cell = {"step_id": sid, "label": step_label(sid, with_language=False), "state": state,
                "generation": (s or {}).get("generation") or 1,
                "reason": _step_reason((s or {}).get("reason"))}
        groups.setdefault(lang, []).append(cell)
    out = []
    if "" in groups:
        out.append({"language": None, "cells": groups.pop("")})
    for lang, cells in groups.items():
        out.append({"language": lang.upper(), "cells": cells})
    return out


def spec_ids(flow_id: str, flow_version: int, project, params: Optional[dict]) -> Optional[list]:
    """The flow's step ids in declared order, or None when the flow is gone."""
    from . import registry

    flow = registry.get_flow(flow_id, flow_version)
    if flow is None:
        return None
    try:
        return [s.id for s in flow.steps(project, params or {})]
    except Exception:                       # bad params: show the rows as they are
        return None


# ---------------------------------------------------------------------------
# Budget bar
# ---------------------------------------------------------------------------


def budget_bar(b: Optional[dict]) -> Optional[dict]:
    """Widths (in %) of committed, the approved mark and the tolerance band.

    The scale is the larger of committed and approved × (1 + tolerance), so the
    band always fits and an overrun shows as a full bar past the band."""
    if not b or not b.get("approved"):
        return None
    approved = float(b["approved"]["credits"] or 0)
    committed = float(b["committed"]["credits"] or 0)
    tol = float(b.get("tolerance") or 0)
    ceiling = approved * (1 + tol)
    scale = max(ceiling, committed)
    if scale <= 0:
        return None

    def pct(x):
        return round(100.0 * max(0.0, x) / scale, 1)

    tone = "over" if committed > ceiling else ("band" if committed > approved else "ok")
    return {"committed_pct": pct(committed), "approved_pct": pct(approved),
            "ceiling_pct": pct(ceiling), "tone": tone}


# ---------------------------------------------------------------------------
# Paused banner
# ---------------------------------------------------------------------------

# (pattern on status_reason, cause, fix, resume_hint, show_lines)
_PAUSES = [
    (r"^paused by ", "You paused it.", "Resume when you want it to continue.", "", False),
    (r"^manual paid action: ", "A paid action was started by hand on this project.",
     "Resume to continue.", "It re-checks the runs first and may ask again.", False),
    (r" refused \d+ times: ", "The same action was refused again and again.",
     "Fix what the refusal says (lines below), then Resume.", "", True),
    (r"^step \S+ failed \d+ times in a row: ", "A step kept failing.",
     "Read the errors below; Resume retries the step.", "", True),
    (r"^step \S+ reopened more than \d+ times", "A step was redone too many times.",
     "Check why its inputs keep changing, then Resume.", "", True),
    (r"^ambiguous launch of ", "A launch has an unconfirmed outcome.",
     "Check the project's runs for a duplicate, then Resume.", "", False),
    (r"admission committed without its effect row", "A launch was not recorded properly.",
     "Check the project's runs before resuming.", "", False),
    (r"^analysis failed", "The project analysis failed.", "Re-analyze the project, then Resume.",
     "", False),
    (r"^the project has been \S+ for too long", "The project analysis seems lost.",
     "Re-analyze the project, then Resume.", "", False),
    (r"^run \S+ has not moved for ", "A run has not moved for hours.",
     "Check the run; Resume when it moves again.", "", False),
    (r"^handler \S+ v\d+ is not deployed", "This build cannot finish a launch it started.",
     "Deploy a build with that handler, then Resume.", "", False),
]


def pause_banner(db: Session, ap_id: str, status: str, reason: Optional[str], *,
                 flow_deployed: bool) -> Optional[dict]:
    """``{"cause", "fix", "hint", "resume", "lines"}`` for a paused attempt, else None.

    ``resume`` is False when Resume cannot help (the flow is not in this
    build: the next tick pauses it again)."""
    if status != "paused":
        return None
    raw = reason or ""                       # as journalled (core.journal cuts at 4000, no strip)
    reason = raw.strip()
    if not flow_deployed:
        return {"cause": "This attempt's flow is not in the deployed build.",
                "fix": "Resume would pause again: Stop it, then Start a new autopilot.",
                "hint": "", "resume": False, "lines": [], "reason": reason}
    cause, fix, hint, show = "The assistant paused itself.", "Resume to continue.", "", True
    for pat, c, f, h, s in _PAUSES:
        if re.search(pat, reason):
            cause, fix, hint, show = c, f, h, s
            break
    lines = _pause_lines(db, ap_id, raw) if show else []
    return {"cause": cause, "fix": fix, "hint": hint, "resume": True, "lines": lines,
            "reason": reason}


def _pause_lines(db: Session, ap_id: str, reason: str, limit: int = 3) -> list[dict]:
    """The refusals / errors of the step THIS pause names: the newest ``paused``
    line whose message is the current reason (journal messages are cut at 4000).
    A later pause line (a manual paid action while already paused) is not it."""
    p = db.execute(select(AutopilotJournal).where(
        AutopilotJournal.autopilot_id == ap_id, AutopilotJournal.kind == "paused",
        AutopilotJournal.message == reason[:4000])
        .order_by(AutopilotJournal.id.desc()).limit(1)).scalar_one_or_none()
    if p is None or not p.step_id:
        return []
    rows = db.execute(select(AutopilotJournal).where(
        AutopilotJournal.autopilot_id == ap_id, AutopilotJournal.step_id == p.step_id,
        AutopilotJournal.kind.in_(("refused", "warning")), AutopilotJournal.id < p.id)
        .order_by(AutopilotJournal.id.desc()).limit(limit)).scalars().all()
    return [{"at": j.at.isoformat(), "kind": j.kind, "message": j.message} for j in rows]


# ---------------------------------------------------------------------------
# Stop confirmation (§5)
# ---------------------------------------------------------------------------

STOP_CONFIRM = (
    "Stops this attempt. Work already running (renders, fills) continues and is still billed. "
    "Start a new autopilot later creates a new attempt with the same parameters; the project's "
    "spend so far still counts, and it decides again from the project's current state — "
    "answers given to this attempt are not carried over.")


# ---------------------------------------------------------------------------
# Everything a card needs
# ---------------------------------------------------------------------------


def summary(db: Session, ap, project, *, steps: list, pending_steps: Iterable[str], pending: int,
            budget: dict, only: Optional[Iterable[str]] = None) -> dict:
    """The §5 block of a card for attempt ``ap`` (a ProjectAutopilot row)."""
    from . import registry

    deployed = registry.get_flow(ap.flow_id, ap.flow_version) is not None
    ids = spec_ids(ap.flow_id, ap.flow_version, project, ap.params) if deployed else None
    return {
        "line": status_line(ap.status, ap.status_reason, steps, pending),
        "stepper": stepper(ids, steps, pending_steps, only=only),
        "bar": budget_bar(budget),
        "banner": pause_banner(db, ap.id, ap.status, ap.status_reason, flow_deployed=deployed),
        "stop_confirm": STOP_CONFIRM,
    }
