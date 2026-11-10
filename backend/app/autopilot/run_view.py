"""Where the assistant's work is shown (AICP-369, docs/autopilot-run-view.md).

Read-only. Nothing here changes an autopilot, a step, an effect or an approval;
it only answers three display questions from the records the core already
writes:

* **Which runs are the assistant's?** A run is the assistant's when one of the
  project's autopilot attempts recorded it in ``created_refs["runs"]`` of a
  run-creating action (``create_run`` / ``localisation_create_run``). The
  handlers write that ref in the same commit that creates the run (actions.py
  ``CreateRun.execute``), so the record is the admission fact itself — no
  column, no backfill.
* **Where does a pending approval belong?** On one run's page when it resolves
  to exactly one of the assistant's runs (its version bindings / run-produced
  artifacts, else the run its step works on); otherwise in the project-level
  queue. The project page lists EVERY pending approval anyway (unresolved ones
  in full, run ones as a link), so nothing is ever only reachable from a run.
* **What does a run page show?** The pending approvals that resolve to it, the
  steps that work on it, the project budget, and the journal entries of its
  effects / approvals / steps.

A run that is not recorded is ``manual`` — unless a run-creating launch of the
same project is still ``launching`` or ended ``unknown`` without refs, in
which case runs created after it are ``unconfirmed`` (never guessed manual).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from ..models import Run, VideoProject
from ..models_autopilot import (
    LIVE_AUTOPILOT_STATUSES,
    AutopilotApproval,
    AutopilotEffect,
    AutopilotJournal,
    AutopilotStep,
    ProjectAutopilot,
)

RUN_CREATING_HANDLERS = ("create_run", "localisation_create_run")

# Steps that work on the run a creating step made, per flow: the creating
# step itself plus the ones that read its run from that step's output
# (subtitle_removal.py _run_id; localisation.py rest -> localisation_finish.py).
_SUBTITLE_STEPS = ("launch", "clean", "handoff")
_FINAL_STEP_PREFIXES = ("rest", "qa", "captions", "tail", "cleanup")


@dataclass(frozen=True)
class AssistantRun:
    run_id: str
    autopilot_id: str
    flow_id: str
    step_id: str
    step_generation: int
    effect_id: str
    role: str                       # main | probe | final | <creating step id>
    language: Optional[str]
    current: bool                   # created by the step's CURRENT generation
    steps: tuple = ()               # step ids that work on this run

    @property
    def label(self) -> str:
        parts = [self.role]
        if self.language:
            parts.append(self.language.upper())
        if not self.current:
            parts.append("replaced")
        return " ".join(parts)


@dataclass
class ProjectRuns:
    by_run: dict = field(default_factory=dict)          # run_id -> AssistantRun
    unconfirmed_since: Optional[object] = None          # datetime of the oldest ambiguous launch
    latest_autopilot_id: Optional[str] = None
    latest_live: bool = False

    def kind_of(self, run) -> str:
        """``assistant`` | ``unconfirmed`` | ``manual`` for a Run of this project."""
        if run.id in self.by_run:
            return "assistant"
        if self.unconfirmed_since is not None and run.created_at is not None \
                and run.created_at >= self.unconfirmed_since:
            # the launch protocol writes the effect row BEFORE the run exists
            return "unconfirmed"
        return "manual"


def _role(flow_id: str, step_id: str) -> tuple[str, Optional[str], tuple]:
    base, _, lang = step_id.partition(":")
    if flow_id == "subtitle_removal" and step_id == "launch":
        return "main", None, _SUBTITLE_STEPS
    if base == "probe" and lang:
        return "probe", lang, (step_id,)
    if base == "rest" and lang:
        return "final", lang, tuple(f"{p}:{lang}" for p in _FINAL_STEP_PREFIXES)
    return step_id, (lang or None), (step_id,)


def project_runs(db: Session, project_id: str) -> ProjectRuns:
    """Every assistant-created run of the project, over ALL autopilot attempts."""
    out = ProjectRuns()
    aps = db.execute(select(ProjectAutopilot).where(ProjectAutopilot.project_id == project_id)
                     .order_by(ProjectAutopilot.created_at.desc(), ProjectAutopilot.id.desc())
                     ).scalars().all()
    if not aps:
        return out
    out.latest_autopilot_id = aps[0].id
    out.latest_live = aps[0].status in LIVE_AUTOPILOT_STATUSES
    flows = {a.id: a.flow_id for a in aps}
    gens = {(s.autopilot_id, s.step_id): s.generation for s in db.execute(
        select(AutopilotStep).where(AutopilotStep.autopilot_id.in_(list(flows)))).scalars()}
    effects = db.execute(select(AutopilotEffect).where(
        AutopilotEffect.autopilot_id.in_(list(flows)),
        AutopilotEffect.handler.in_(RUN_CREATING_HANDLERS))
        .order_by(AutopilotEffect.created_at)).scalars().all()
    candidates = []
    for e in effects:
        rids = [r for r in ((e.created_refs or {}).get("runs") or []) if isinstance(r, str)]
        if not rids:
            if e.state in ("launching", "unknown"):
                if out.unconfirmed_since is None or e.created_at < out.unconfirmed_since:
                    out.unconfirmed_since = e.created_at
            continue
        for rid in rids:
            candidates.append((rid, e))
    if not candidates:
        return out
    owned = {rid for (rid,) in db.execute(select(Run.id).where(
        Run.id.in_([c[0] for c in candidates]), Run.project_id == project_id))}
    for rid, e in candidates:
        if rid not in owned or rid in out.by_run:
            continue  # deleted / another project's / already recorded (first record wins)
        flow_id = flows[e.autopilot_id]
        role, lang, steps = _role(flow_id, e.step_id)
        current = (e.autopilot_id == out.latest_autopilot_id
                   and gens.get((e.autopilot_id, e.step_id), e.step_generation) == e.step_generation)
        out.by_run[rid] = AssistantRun(
            run_id=rid, autopilot_id=e.autopilot_id, flow_id=flow_id, step_id=e.step_id,
            step_generation=e.step_generation, effect_id=e.id, role=role, language=lang,
            current=current, steps=steps)
    # Of two runs with the same role in one attempt, only the newest is current.
    seen = {}
    for rid, ar in list(out.by_run.items()):  # insertion order = effect creation order
        key = (ar.autopilot_id, ar.role, ar.language)
        prev = seen.get(key)
        if prev is not None and out.by_run[prev].current and ar.current:
            out.by_run[prev] = _replace(out.by_run[prev], current=False)
        seen[key] = rid
    return out


def _replace(ar: AssistantRun, **kw) -> AssistantRun:
    d = {k: getattr(ar, k) for k in ar.__dataclass_fields__}
    d.update(kw)
    return AssistantRun(**d)


def _step_owner(pr: ProjectRuns, autopilot_id: str, step_id: str) -> Optional[str]:
    """The CURRENT assistant run whose steps include ``step_id`` in this attempt."""
    hits = [ar.run_id for ar in pr.by_run.values()
            if ar.autopilot_id == autopilot_id and ar.current and step_id in ar.steps]
    return hits[0] if len(hits) == 1 else None


def approval_runs(a: AutopilotApproval, pr: ProjectRuns) -> tuple[Optional[str], list, str]:
    """``(run_id or None, candidate run ids, reason)`` for one approval.

    ``run_id`` set = show it on that run's page. ``None`` = the project queue
    (budget, pre-run work, nothing or several runs to point at)."""
    if a.kind == "budget":
        return None, [], "the budget is project-wide"
    mine = {rid for rid, ar in pr.by_run.items() if ar.autopilot_id == a.autopilot_id}
    bound = set()
    for k in (a.bound_versions or {}):
        if isinstance(k, str) and k.startswith("run:"):
            bound.add(k[4:])
    for art in a.artifacts or []:
        prod = art.get("producer") or ""
        if prod.startswith("run:"):
            bound.add(prod[4:])
    bound &= mine
    owner = _step_owner(pr, a.autopilot_id, a.step_id)
    if len(bound) == 1:
        return next(iter(bound)), sorted(bound), "bound to the run"
    if len(bound) > 1:
        if owner in bound:
            return owner, sorted(bound), "bound to several runs; its step works on this one"
        return None, sorted(bound), "bound to several runs"
    if owner:
        return owner, [owner], "its step works on this run"
    return None, [], "not about a run yet"


def _ap_header(db: Session, ap: ProjectAutopilot) -> dict:
    from . import core

    return {"id": ap.id, "project_id": ap.project_id, "flow_id": ap.flow_id,
            "flow_version": ap.flow_version, "status": ap.status,
            "status_reason": ap.status_reason, "created_at": ap.created_at.isoformat(),
            "budget": core.budget_view(db, ap)}


def project_card_extras(db: Session, project_id: str, view: Optional[dict]) -> dict:
    """What the project card adds to ``core.view``: the assistant's runs, where
    each pending approval is shown, and the §5 status block (``ux``, AICP-377)."""
    pr = project_runs(db, project_id)
    runs = {r.id: r for r in db.execute(select(Run).where(
        Run.id.in_(list(pr.by_run)))).scalars()} if pr.by_run else {}
    placement = {}
    if view and view.get("approvals"):
        from .links import latest_attempt, place

        latest = latest_attempt(db, project_id)
        rows = {a.id: a for a in db.execute(select(AutopilotApproval).where(
            AutopilotApproval.id.in_([x["id"] for x in view["approvals"]]))).scalars()}
        for x in view["approvals"]:
            a = rows.get(x["id"])
            if a is None:
                continue
            pl = place(db, a, project_id=project_id, pr=pr, latest=latest, runs=runs)
            placement[x["id"]] = {"run_id": pl.run_id, "candidates": list(pl.candidates),
                                  "reason": pl.reason, "short_id": pl.short_id, "path": pl.path}
    listed = []
    for rid, ar in sorted(pr.by_run.items(), key=lambda kv: runs[kv[0]].created_at):
        r = runs[rid]
        listed.append({"run_id": rid, "short_id": r.short_id, "name": r.name,
                       "status": r.status.value if hasattr(r.status, "value") else str(r.status),
                       "label": ar.label, "current": ar.current,
                       "this_attempt": ar.autopilot_id == pr.latest_autopilot_id})
    ux = None
    if view:
        from . import status as ap_status

        ap = db.get(ProjectAutopilot, view["id"])
        ux = ap_status.summary(
            db, ap, db.get(VideoProject, project_id), steps=view.get("steps") or [],
            pending_steps=[a["step_id"] for a in view.get("approvals") or []],
            pending=len(view.get("approvals") or []), budget=view.get("budget"))
    return {"assistant_runs": listed, "placement": placement, "ux": ux}


def run_labels(db: Session, project, runs) -> dict:
    """``{run_id: {"kind", "label"}}`` for the project's runs table (assistant projects)."""
    if getattr(project, "mode", None) != "assistant" or not runs:
        return {}
    pr = project_runs(db, project.id)
    out = {}
    for r in runs:
        kind = pr.kind_of(r)
        ar = pr.by_run.get(r.id)
        out[r.id] = {"kind": kind, "label": ar.label if ar else kind}
    return out


def run_journal_scope(db: Session, run, pr: Optional[ProjectRuns] = None):
    """``(autopilot_id, SQL condition)`` of the journal lines about this run, or
    ``None`` for a run that is not the assistant's.

    The run's effects and approvals exactly; step-only lines of its steps from
    the run's creation until a replacement of the same role (journal rows have
    no generation, so the window is what separates them — the timeline marks
    them "during this run")."""
    from .links import latest_attempt, place

    pr = pr or project_runs(db, run.project_id)
    ar = pr.by_run.get(run.id)
    if ar is None:
        return None
    latest = latest_attempt(db, run.project_id)
    runs = {run.id: run}
    effect_ids = [e.id for e in db.execute(select(AutopilotEffect).where(
        AutopilotEffect.autopilot_id == ar.autopilot_id,
        AutopilotEffect.step_id.in_(list(ar.steps)))).scalars()
        if run.id in _effect_runs(e)]
    approval_ids = [a.id for a in db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == ar.autopilot_id,
        AutopilotApproval.step_id.in_(list(ar.steps)))).scalars()
        if place(db, a, project_id=run.project_id, pr=pr, latest=latest, runs=runs).run_id == run.id
        or f"run:{run.id}" in (a.bound_versions or {})]
    creator = db.get(AutopilotEffect, ar.effect_id)
    since = creator.created_at if creator is not None else run.created_at
    until = None
    for other in pr.by_run.values():
        if other.run_id != run.id and other.autopilot_id == ar.autopilot_id \
                and other.role == ar.role and other.language == ar.language:
            o = db.get(AutopilotEffect, other.effect_id)
            if o is not None and o.created_at > since and (until is None or o.created_at < until):
                until = o.created_at
    step_only = and_(AutopilotJournal.step_id.in_(list(ar.steps)),
                     AutopilotJournal.effect_id.is_(None), AutopilotJournal.approval_id.is_(None),
                     AutopilotJournal.at >= since)
    if until is not None:
        step_only = and_(step_only, AutopilotJournal.at < until)
    conds = [step_only]
    if effect_ids:
        conds.append(AutopilotJournal.effect_id.in_(effect_ids))
    if approval_ids:
        conds.append(AutopilotJournal.approval_id.in_(approval_ids))
    return ar.autopilot_id, or_(*conds)


def run_card(db: Session, run) -> dict:
    """The Run page's assistant card. ``kind`` = assistant | unconfirmed | manual."""
    from . import core

    pr = project_runs(db, run.project_id)
    kind = pr.kind_of(run)
    ar = pr.by_run.get(run.id)
    out = {"kind": kind, "run_id": run.id, "project_id": run.project_id,
           "latest_live": pr.latest_live, "ap": None}
    if ar is None:
        return out
    ap = db.get(ProjectAutopilot, ar.autopilot_id)
    if ap is None:
        return out
    # Answers: whenever the core accepts them (the latest attempt, any status, as
    # on the project card). Pause/Resume/Stop: only while that attempt is live.
    can_answer = ap.id == pr.latest_autopilot_id
    controls = can_answer and ap.status in LIVE_AUTOPILOT_STATUSES
    pending = db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.status == "pending")
        .order_by(AutopilotApproval.created_at)).scalars().all()
    from .links import latest_attempt, place

    latest = latest_attempt(db, run.project_id)
    runs = {run.id: run}

    def _here(a):
        return place(db, a, project_id=run.project_id, pr=pr, latest=latest, runs=runs)

    mine, elsewhere = [], 0
    for a in pending:
        pl = _here(a)
        if pl.run_id == run.id:
            d = core._card(core.approval_dict(a, run.project_id), ap.flow_id)
            d["reason"] = pl.reason
            mine.append(d)
        else:
            elsewhere += 1
    answered = []
    for a in core.reopenable(db, ap.id):
        if _here(a).run_id == run.id:
            answered.append(dict(core.approval_dict(a, run.project_id),
                                 chosen_option=a.chosen_option))
    steps = db.execute(select(AutopilotStep).where(
        AutopilotStep.autopilot_id == ap.id, AutopilotStep.step_id.in_(list(ar.steps)))
        .order_by(AutopilotStep.started_at)).scalars().all()
    from . import status as ap_status

    header = _ap_header(db, ap)
    step_rows = [core.step_dict(s) for s in steps]
    # The status line is the attempt's (all its steps); the stepper keeps this run's.
    all_steps = [core.step_dict(s) for s in db.execute(select(AutopilotStep).where(
        AutopilotStep.autopilot_id == ap.id).order_by(AutopilotStep.started_at)).scalars()]
    header["ux"] = ap_status.summary(
        db, ap, db.get(VideoProject, run.project_id), steps=all_steps,
        pending_steps=[a.step_id for a in pending], pending=len(pending),
        budget=header["budget"], only=ar.steps)
    related = []
    for other in pr.by_run.values():
        if other.run_id != run.id and other.autopilot_id == ap.id and \
                (other.language == ar.language or not ar.language):
            r = db.get(Run, other.run_id)
            if r is not None:
                related.append({"run_id": r.id, "short_id": r.short_id, "label": other.label})
    out.update({
        "ap": header, "role": ar.label, "current": ar.current,
        "controls": controls, "can_answer": can_answer, "approvals": mine, "elsewhere": elsewhere,
        "answered": answered,
        "steps": step_rows, "related": related,
    })
    return out


def _effect_runs(e: AutopilotEffect) -> set:
    rids = set((e.created_refs or {}).get("runs") or []) | set((e.targets or {}).get("runs") or [])
    params = e.params or {}
    if isinstance(params.get("run_id"), str):
        rids.add(params["run_id"])
    return rids
