"""The autopilot engine (docs/autopilot-design.md §3, §5, §6).

One tick of one autopilot (:func:`tick_autopilot`):

A. reconcile (any status, observation only) — manual paid actions (§5.7),
   answered budget approvals, ``launching`` effects, compute liveness,
   judgement polls, spend-scope registration, ``depends_on`` reopening, stall
   and project checks;
B. stop unless ``active``;
C. evaluate ``flow.steps``; rows for ready steps;
D. per ready step: PLAN in a database-enforced read-only session, then APPLY
   in one transaction under gate → project → autopilot (→ runs → approvals)
   locks with a ``version`` compare-and-set; one admitted launch ends the tick;
E. completion.

The launch protocol (§5.4): the effect row with its UNIQUE idempotency key is
flushed by the core before any handler code and committed by the handler's
single admission commit, together with the domain rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from fastapi import HTTPException
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Run, RunSegment, RunSegmentTake, SegmentDef, VideoProject
from ..models_autopilot import (
    EFFECT_TERMINAL_STATES,
    LIVE_AUTOPILOT_STATUSES,
    AutopilotApproval,
    AutopilotEffect,
    AutopilotGate,
    AutopilotJournal,
    AutopilotStep,
    ProjectAutopilot,
    ProjectManualSpendEvent,
    utcnow,
)
from . import actions as _actions  # noqa: F401 — registers the core's handlers
from . import registry, spend_view
from .contract import (
    APPROVAL_KINDS,
    NAME_RE,
    OUTPUT_MAX_BYTES,
    ActionInvalid,
    ActionRefused,
    ApprovalView,
    ArtifactSpec,
    Ask,
    BudgetLine,
    BudgetView,
    Complete,
    Compute,
    ContractViolation,
    EffectView,
    Fail,
    Judge,
    Launch,
    RefusalView,
    RunSegmentView,
    RunView,
    Wait,
)

log = logging.getLogger(__name__)

TERMINAL_RUN_STATUSES = ("done", "failed", "incomplete")
LEASE_SEC = 90
COMPUTE_NOT_LIVE_GRACE_SEC = 120
COMPUTE_MAX_ATTEMPTS = 3
MAX_STEP_ERRORS = 5
MAX_REFUSALS = 8
MAX_REOPENS = 5
CORE_SEP = "#"  # core-created approval names carry it; flow names cannot (NAME_RE)


def _status(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def _mc(credits: float) -> int:
    return int(round(float(credits or 0.0) * 1000))


def _cr(mcredits: Optional[int]) -> Optional[float]:
    return None if mcredits is None else mcredits / 1000.0


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        from datetime import timezone

        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ---------------------------------------------------------------------------
# Injectable side effects (tests replace them)
# ---------------------------------------------------------------------------


@dataclass
class Hooks:
    enqueue_compute: Optional[Callable[[str, str], None]] = None
    compute_is_live: Optional[Callable[[AutopilotEffect], bool]] = None
    port: Any = None
    now: Optional[Callable[[], datetime]] = None

    def clock(self) -> datetime:
        return self.now() if self.now else utcnow()

    def get_port(self):
        if self.port is not None:
            return self.port
        from .judgement import get_port

        return get_port()

    def enqueue(self, effect_id: str, job_id: str, queue: Optional[str] = None) -> None:
        if self.enqueue_compute is not None:
            return self.enqueue_compute(effect_id, job_id)
        from ..tasks import enqueue_autopilot_compute

        enqueue_autopilot_compute(effect_id, job_id, queue=queue)

    def is_live(self, eff: AutopilotEffect) -> bool:
        if self.compute_is_live is not None:
            return self.compute_is_live(eff)
        from ..tasks import _get_redis, rq_job_is_live

        # AICP-389: a compute lives on its registered queue (brand_asr for
        # the brand check), not always on autopilot_compute.
        queue = registry.compute_queue(eff.handler, eff.handler_version)
        return rq_job_is_live(_get_redis(), queue, eff.rq_job_id)


# ---------------------------------------------------------------------------
# Small DB helpers
# ---------------------------------------------------------------------------


def journal(db: Session, ap: ProjectAutopilot, kind: str, message: str, *,
            step_id: Optional[str] = None, data: Optional[dict] = None,
            cost_mcredits: Optional[int] = None, effect_id: Optional[str] = None,
            approval_id: Optional[str] = None, now: Optional[datetime] = None) -> None:
    db.add(AutopilotJournal(
        autopilot_id=ap.id, at=now or utcnow(), step_id=step_id, kind=kind,
        message=message[:4000], data=data, cost_mcredits=cost_mcredits,
        effect_id=effect_id, approval_id=approval_id,
    ))


def _touch(ap: ProjectAutopilot, now: Optional[datetime] = None) -> None:
    ap.version = (ap.version or 0) + 1
    ap.updated_at = now or utcnow()


class _HeldGate:
    """What :func:`_gate` returns: the autopilot deploy hold OR the global
    deploy admission gate (AICP-350). Read-only view — never an ORM row, so
    reporting ``held`` for the global gate can never write ``autopilot_gate``."""

    __slots__ = ("held", "generation", "row")

    def __init__(self, row: AutopilotGate, ops_paused: bool):
        self.row = row
        self.held = bool(row.held) or ops_paused
        self.generation = row.generation


def _gate(db: Session, *, share: bool) -> _HeldGate:
    # AICP-350: the global admission gate's share lock FIRST (the same lock
    # order as every gated API route), then the autopilot's own hold row. A
    # deploy's begin.sql therefore waits for every autopilot admission in
    # flight too, and every later one reads `held`.
    from .. import ops_gate

    ops = ops_gate.read_gate(db, share=share)
    q = select(AutopilotGate).where(AutopilotGate.id == 1)
    q = q.with_for_update(read=True) if share else q.with_for_update()
    gate = db.execute(q).scalar_one_or_none()
    if gate is None:  # the migration inserts it; SQLite test schemas may not
        gate = AutopilotGate(id=1, held=False, generation=0)
        db.add(gate)
        db.flush()
    return _HeldGate(gate, ops.paused)


def _lock_project(db: Session, project_id: str) -> Optional[VideoProject]:
    return db.execute(
        select(VideoProject).where(VideoProject.id == project_id).with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


def _lock_ap(db: Session, ap_id: str) -> Optional[ProjectAutopilot]:
    return db.execute(
        select(ProjectAutopilot).where(ProjectAutopilot.id == ap_id).with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


def _lock_runs(db: Session, run_ids) -> dict[str, Run]:
    ids = sorted(set(run_ids))
    if not ids:
        return {}
    rows = db.execute(
        select(Run).where(Run.id.in_(ids)).order_by(Run.id).with_for_update()
        .execution_options(populate_existing=True)
    ).scalars().all()
    return {r.id: r for r in rows}


def _set_status(db: Session, ap: ProjectAutopilot, status: str, reason: str, *,
                kind: str, now: datetime, step_id: Optional[str] = None) -> None:
    ap.status, ap.status_reason = status, reason
    if status in ("completed", "stopped", "failed"):
        ap.finished_at = now
    _touch(ap, now)
    journal(db, ap, kind, reason, step_id=step_id, now=now)


# ---------------------------------------------------------------------------
# Versions (§4.3, §4.6)
# ---------------------------------------------------------------------------


def _file_sig(path: Optional[str]):
    if not path:
        return None
    try:
        st = os.stat(path)
        return [path, st.st_size, st.st_mtime_ns]
    except OSError:
        return [path, None, None]


def run_version(db: Session, run_id: str) -> str:
    run = db.get(Run, run_id)
    if run is None:
        return "gone"
    segs = db.execute(
        select(RunSegment).where(RunSegment.run_id == run_id).order_by(RunSegment.index)
    ).scalars().all()
    return _sha({
        "status": _status(run.status), "prompt": run.prompt, "model": run.model,
        "resolution": run.resolution, "refs": run.reference_image_urls,
        "audio_refs": run.reference_audio_urls, "plan": run.stitch_plan,
        "result": _file_sig(run.result_local_path), "drive": run.result_gdrive_file_id,
        "segments": [[
            rs.id, _status(rs.status), rs.prompt_override, rs.reference_image_urls_override,
            rs.generated_reference_urls, rs.generated_image_job_id,
            _file_sig(rs.local_result_path), rs.seedance_task_id,
        ] for rs in segs],
    })


def segments_version(db: Session, project_id: str) -> str:
    rows = db.execute(
        select(SegmentDef.id, SegmentDef.index, SegmentDef.action, SegmentDef.start_sec,
               SegmentDef.end_sec, SegmentDef.pre_roll_sec, SegmentDef.post_roll_sec)
        .where(SegmentDef.project_id == project_id).order_by(SegmentDef.index)
    ).all()
    return _sha([list(r) for r in rows])


def detector_version(db: Session, project_id: str) -> str:
    row = db.execute(select(VideoProject.caption_detector).where(
        VideoProject.id == project_id)).first()
    return "gone" if row is None else _sha(row[0])


def transcript_version(db: Session, project_id: str) -> str:
    row = db.execute(select(VideoProject.transcript, VideoProject.transcript_status).where(
        VideoProject.id == project_id)).first()
    return "gone" if row is None else _sha([row[0], row[1]])


def _step_version(db: Session, ap_id: str, step_id: str) -> str:
    row = db.execute(select(AutopilotStep).where(
        AutopilotStep.autopilot_id == ap_id, AutopilotStep.step_id == step_id)).scalar_one_or_none()
    return "absent" if row is None else f"{row.generation}:{row.status}"


def _effect_version(db: Session, ap_id: str, effect_id: str) -> str:
    eff = db.get(AutopilotEffect, effect_id)
    if eff is None or eff.autopilot_id != ap_id:
        return "absent"
    return f"{eff.state}:{eff.claim_token}"


def key_version(db: Session, ap: ProjectAutopilot, key: str) -> str:
    kind, _, ident = key.partition(":")
    if kind == "run":
        run = db.get(Run, ident)
        if run is not None and run.project_id != ap.project_id:
            raise ContractViolation(f"version key {key}: run of another project")
        return run_version(db, ident)
    if kind == "project_segments":
        if ident != ap.project_id:
            raise ContractViolation(f"version key {key}: another project")
        return segments_version(db, ident)
    if kind == "project_transcript":
        if ident != ap.project_id:
            raise ContractViolation(f"version key {key}: another project")
        return transcript_version(db, ident)
    if kind == "project_detector":
        if ident != ap.project_id:
            raise ContractViolation(f"version key {key}: another project")
        return detector_version(db, ident)
    if kind == "step":
        return _step_version(db, ap.id, ident)
    if kind == "effect":
        return _effect_version(db, ap.id, ident)
    raise ContractViolation(f"unknown version key {key!r}")


def _run_ids_of_keys(keys) -> set[str]:
    return {k.partition(":")[2] for k in keys if k.startswith("run:")}


# ---------------------------------------------------------------------------
# Context handed to advance() / estimate()
# ---------------------------------------------------------------------------


def _approval_view(a: AutopilotApproval) -> ApprovalView:
    return ApprovalView(
        id=a.id, name=a.name, generation=a.generation, kind=a.kind, status=a.status,
        decision=a.answered_decision, chosen_option=a.chosen_option,
        comments=list(a.comments or []), answer_comment=a.answer_comment,
        answered_at=a.answered_at,
    )


def _effect_view(e: AutopilotEffect) -> EffectView:
    return EffectView(
        id=e.id, name=e.name, kind=e.kind, handler=e.handler, state=e.state,
        params=dict(e.params or {}),
        targets=dict(e.targets or {}), created_refs=dict(e.created_refs or {}),
        result=e.result, error=e.error, created_at=e.created_at,
    )


def _latest_approval(db: Session, ap_id: str, step_id: str, step_gen: int,
                     name: str) -> Optional[AutopilotApproval]:
    return db.execute(
        select(AutopilotApproval).where(
            AutopilotApproval.autopilot_id == ap_id, AutopilotApproval.step_id == step_id,
            AutopilotApproval.step_generation == step_gen, AutopilotApproval.name == name,
        ).order_by(AutopilotApproval.generation.desc()).limit(1)
    ).scalar_one_or_none()


class Context:
    """Implements contract.StepContext over a (read-only) session."""

    def __init__(self, db: Session, ap: ProjectAutopilot, project: VideoProject,
                 step_row: AutopilotStep, *, now: datetime, hold: bool):
        self.db = db
        self._ap = ap
        self.project = project
        self.now = now
        self.params = dict(ap.params or {})
        self.step_id = step_row.step_id
        self.step_generation = step_row.generation
        self._row = step_row
        self.hold = hold
        self.seen_versions: dict[str, str] = {}
        f = spend_view.figures_mcredits(db, ap.project_id)
        self.budget = BudgetView(
            approved_credits=_cr(ap.approved_budget_mcredits),
            committed_credits=f["committed"] / 1000.0,
            consumed_credits=f["consumed"] / 1000.0,
            settled_credits=f["settled"] / 1000.0,
            tolerance=float(ap.tolerance),
        )

    def output(self, step_id):
        row = self.db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == self._ap.id,
            AutopilotStep.step_id == step_id)).scalar_one_or_none()
        if row is None or row.status != "completed":
            return None
        return dict(row.output or {})

    def _effects(self):
        return self.db.execute(select(AutopilotEffect).where(
            AutopilotEffect.autopilot_id == self._ap.id,
            AutopilotEffect.step_id == self.step_id,
            AutopilotEffect.step_generation == self.step_generation,
        ).order_by(AutopilotEffect.created_at)).scalars().all()

    def effect(self, name):
        for e in self._effects():
            if e.name == name:
                return _effect_view(e)
        return None

    def effects(self, prefix=""):
        return [_effect_view(e) for e in self._effects() if e.name.startswith(prefix)]

    def next_round(self, prefix):
        used = {e.name for e in self._effects()}
        used |= {n for (n,) in self.db.execute(select(AutopilotApproval.name).where(
            AutopilotApproval.autopilot_id == self._ap.id,
            AutopilotApproval.step_id == self.step_id,
            AutopilotApproval.step_generation == self.step_generation))}
        n = 1
        while f"{prefix}:r{n}" in used or any(u.startswith(f"{prefix}:r{n}{CORE_SEP}") for u in used):
            n += 1
        return f"{prefix}:r{n}"

    def approval(self, name):
        a = _latest_approval(self.db, self._ap.id, self.step_id, self.step_generation, name)
        if a is None or a.status == "superseded":
            return None
        return _approval_view(a)

    def refusal(self, name):
        lr = self._row.last_refusal or {}
        if lr.get("name") != name:
            return None
        return RefusalView(name=name, status=int(lr.get("status", 0)), detail=lr.get("detail", ""),
                           count=int(self._row.refusals or 0),
                           at=datetime.fromisoformat(lr["at"]) if lr.get("at") else self.now)

    def run(self, run_id):
        run = self.db.get(Run, run_id)
        if run is None or run.project_id != self.project.id:
            return None
        segs = self.db.execute(select(RunSegment).where(RunSegment.run_id == run_id)
                               .order_by(RunSegment.index)).scalars().all()
        takes = dict(self.db.execute(
            select(RunSegmentTake.run_segment_id, func.count()).where(
                RunSegmentTake.run_id == run_id).group_by(RunSegmentTake.run_segment_id)).all())
        st = _status(run.status)
        return RunView(
            id=run.id, status=st, is_terminal=st in TERMINAL_RUN_STATUSES,
            result_local_path=run.result_local_path,
            result_gdrive_file_id=run.result_gdrive_file_id,
            segments=[RunSegmentView(
                id=rs.id, index=rs.index, status=_status(rs.status),
                local_result_path=rs.local_result_path, cost_credits=rs.cost_credits,
                take_count=int(takes.get(rs.id, 0))) for rs in segs],
            short_id=run.short_id, updated_at=run.updated_at,
        )

    def run_version(self, run_id):
        v = key_version(self.db, self._ap, f"run:{run_id}")
        self.seen_versions[f"run:{run_id}"] = v
        return v

    def segments_version(self):
        return self.version(f"project_segments:{self.project.id}")

    def transcript_version(self):
        return self.version(f"project_transcript:{self.project.id}")

    def version(self, key):
        """Any version key of §4.3 (run:, project_segments:, project_transcript:,
        step:, effect:) as the core will re-check it under its lock."""
        v = key_version(self.db, self._ap, key)
        self.seen_versions[key] = v
        return v

    def fills(self, run_id):
        """Read-only snapshot of a run's segment caption fills (AICP-358) for a
        planner: newest first, DB fields only (no ffprobe, no hashing)."""
        from ..provider_models import ProviderTask
        from ..segment_fill_models import RunSegmentFillJob

        run = self.db.get(Run, run_id)
        if run is None or run.project_id != self.project.id:
            return []
        out = []
        for j in self.db.execute(select(RunSegmentFillJob).where(
                RunSegmentFillJob.run_id == run_id)
                .order_by(RunSegmentFillJob.created_at.desc())).scalars():
            task = self.db.get(ProviderTask, j.provider_task_id) if j.provider_task_id else None
            out.append({
                "id": j.id, "run_segment_id": j.run_segment_id, "status": j.status,
                "mask_box": j.mask_box, "estimated_usd": j.estimated_usd,
                "approved_usd": j.approved_usd, "preview_path": j.preview_path,
                "composite_path": j.composite_path, "provider_task_id": j.provider_task_id,
                "provider_status": task.status if task is not None else None,
                "error_code": j.error_code, "error_message": j.error_message,
                "created_at": j.created_at, "applied_at": j.applied_at,
            })
        return out

    def previous_outputs(self):
        """Outputs of this step's earlier generations, oldest first
        ([{"generation": n, "output": {...}}]); empty on generation 1."""
        return list(self._row.previous_outputs or [])

    def artifact(self, path, *, label, kind, run_id=None, effect_id=None):
        return ArtifactSpec(path=path, label=label, kind=kind, run_id=run_id, effect_id=effect_id)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def create_autopilot(db: Session, project: VideoProject, flow, params: dict, *,
                     now: Optional[datetime] = None) -> ProjectAutopilot:
    """Add (no commit) the autopilot of a new assistant project."""
    now = now or utcnow()
    ap = ProjectAutopilot(
        id=str(uuid.uuid4()), project_id=project.id, flow_id=flow.id,
        flow_version=int(flow.version), status="active", params=params or {},
        tolerance=float(settings.AUTOPILOT_DEFAULT_TOLERANCE), version=0, outstanding=0,
        created_at=now, updated_at=now,
    )
    db.add(ap)
    db.flush()  # the journal row's FK needs the autopilot row first
    journal(db, ap, "created", f"autopilot created: flow {flow.id} v{flow.version}",
            data={"params": params}, now=now)
    return ap


# ---------------------------------------------------------------------------
# Planning (read-only session, §4.5)
# ---------------------------------------------------------------------------


def _deny(*_a, **_k):
    raise ContractViolation("the planning session is read-only (flush/commit refused)")


class _ReadOnly:
    """A planning session whose transaction is read-only in the database
    (Postgres ``SET TRANSACTION READ ONLY``; SQLite ``PRAGMA query_only`` on
    the session's own connection, reset on that same connection before it goes
    back to the pool), with flush/commit refused."""

    def __init__(self, session_factory):
        self.sf = session_factory
        self.db: Optional[Session] = None
        self.conn = None
        self._sqlite = False

    def __enter__(self) -> Session:
        db = self.sf()
        self.conn = db.connection()
        self._sqlite = self.conn.dialect.name == "sqlite"
        if self._sqlite:
            self.conn.exec_driver_sql("PRAGMA query_only=ON")
        else:
            self.conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        db.flush = _deny  # type: ignore[method-assign]
        db.commit = _deny  # type: ignore[method-assign]
        self.db = db
        return db

    def __exit__(self, *exc):
        try:
            if self._sqlite:
                try:
                    self.conn.exec_driver_sql("PRAGMA query_only=OFF")
                except Exception:
                    self.conn.invalidate()  # never return a read-only connection
        finally:
            try:
                self.db.rollback()
            finally:
                self.db.close()
        return False


@dataclass
class Plan:
    version: int
    step_gen: int
    decision: Any
    params: Optional[dict] = None
    targets: Optional[dict] = None
    target_versions: dict = field(default_factory=dict)
    prediction: Optional[list] = None
    spec: Any = None   # the StepSpec / step list evaluated with the planned params
    specs: Any = None


def _needs_closure(specs_by_id: dict, step_id: str) -> list[str]:
    out, stack = [], list(specs_by_id[step_id].needs) if step_id in specs_by_id else []
    while stack:
        s = stack.pop()
        if s in out:
            continue
        out.append(s)
        if s in specs_by_id:
            stack.extend(specs_by_id[s].needs)
    return out


def _dependents_closure(specs: list, step_id: str) -> list[str]:
    out, frontier = [], [step_id]
    while frontier:
        cur = frontier.pop()
        for s in specs:
            if cur in s.needs and s.id not in out:
                out.append(s.id)
                frontier.append(s.id)
    return out


def _plan(session_factory, ap_id: str, spec, specs: list, flow, *, now: datetime,
          hold: bool) -> Plan:
    with _ReadOnly(session_factory) as db:
        ap = db.get(ProjectAutopilot, ap_id)
        project = db.get(VideoProject, ap.project_id)
        # The step list is a function of the params read IN THIS session (the
        # version the CAS will check), never of what the tick saw earlier.
        specs = list(flow.steps(project, dict(ap.params or {})))
        spec = next((s for s in specs if s.id == spec.id), None)
        if spec is None:
            return Plan(version=-1, step_gen=-1, decision=None)
        row = db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap_id, AutopilotStep.step_id == spec.id)).scalar_one()
        ctx = Context(db, ap, project, row, now=now, hold=hold)
        decision = spec.advance(ctx)
        plan = Plan(version=ap.version, step_gen=row.generation, decision=decision,
                    spec=spec, specs=specs)
        if isinstance(decision, Launch):
            handler = registry.get_handler(decision.handler, flow.handlers.get(decision.handler)
                                           if getattr(flow, "handlers", None) else None)
            if handler is None:
                raise ContractViolation(f"unknown handler {decision.handler!r}")
            plan.params = handler.validate(db, project, dict(decision.params))
            plan.targets = handler.targets(db, project, plan.params)
            for rid in handler.target_run_ids(plan.targets):
                plan.target_versions[f"run:{rid}"] = run_version(db, rid)
            if handler.paid and ap.approved_budget_mcredits is None:
                plan.prediction = _predict(db, ap, project, specs, flow, now=now, hold=hold)
        return plan


def _predict(db, ap, project, specs, flow, *, now, hold) -> list[dict]:
    lines = []
    for s in specs:
        if not s.paid or s.estimate is None:
            continue
        row = db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap.id, AutopilotStep.step_id == s.id)).scalar_one_or_none()
        if row is not None and row.status in ("completed", "skipped"):
            continue
        fake = row or AutopilotStep(step_id=s.id, generation=1, status="pending")
        ctx = Context(db, ap, project, fake, now=now, hold=hold)
        lines.append({"step_id": s.id, "label": s.title, "credits": float(s.estimate(ctx) or 0.0)})
    extra = getattr(flow, "budget_lines", None)
    if extra is not None and specs:
        fake = AutopilotStep(step_id=specs[0].id, generation=1, status="pending")
        for bl in extra(Context(db, ap, project, fake, now=now, hold=hold)) or ():
            if isinstance(bl, BudgetLine):
                lines.append({"step_id": bl.step_id, "label": bl.label, "credits": float(bl.credits)})
    return lines


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


class _Discard(Exception):
    """Abandon the decision being applied; the next tick replans."""


def _artifact_store(project_id: str) -> str:
    from ..storage import project_dir

    return os.path.join(project_dir(project_id), "autopilot", "artifacts")


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_artifact(project_id: str, src: str) -> tuple[str, int, str]:
    store = _artifact_store(project_id)
    os.makedirs(store, exist_ok=True)
    tmp = os.path.join(store, f".tmp-{uuid.uuid4().hex}")
    h = hashlib.sha256()
    size = 0
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            h.update(chunk)
            fout.write(chunk)
            size += len(chunk)
    digest = h.hexdigest()
    ext = os.path.splitext(src)[1].lstrip(".").lower()[:8] or "bin"
    dest = os.path.join(store, f"{digest}.{ext}")
    if os.path.exists(dest):
        os.unlink(tmp)
    else:
        os.replace(tmp, dest)
    return digest, size, dest


def _revalidate(db: Session, ap: ProjectAutopilot, a: AutopilotApproval) -> bool:
    for key, v in (a.bound_versions or {}).items():
        if key_version(db, ap, key) != v:
            return False
    for art in a.artifacts or []:
        src = art.get("source_path")
        if not src or not os.path.exists(src) or _hash_file(src) != art.get("sha256"):
            return False
    return True


def _supersede(db, ap, a, reason, now):
    a.status = "superseded"
    journal(db, ap, "approval_answered", f"approval {a.name} superseded: {reason}",
            step_id=a.step_id, approval_id=a.id, now=now)


def _revalidate_named(db, ap, row, names, *, now) -> list[AutopilotApproval]:
    """Lock and revalidate the approvals a decision consumes, BEFORE anything
    else of the decision is written: a stale one is superseded (committed on
    its own) and the decision discarded."""
    out = []
    for name in names:
        a = _latest_approval(db, ap.id, row.step_id, row.generation, name)
        if a is None or a.answered_decision is None or a.status not in ("approved", "rejected"):
            raise ContractViolation(f"consumes {name!r}: no answered, unconsumed approval")
        db.refresh(a, with_for_update=True)
        if not _revalidate(db, ap, a):
            db.rollback()  # nothing of this decision may survive the commit below
            a = db.get(AutopilotApproval, a.id)
            ap2 = _lock_ap(db, ap.id)
            _supersede(db, ap2, a, "what it was about changed since it was answered", now)
            _touch(ap2, now)
            db.commit()
            raise _Discard()
        out.append(a)
    return out


def _sweep_pending(db, ap, row, retire, *, now) -> int:
    """Supersede this step generation's PENDING approvals that can no longer be
    acted on (journalled, in the decision's transaction):

    * every one whose bound versions / artifacts fail :func:`_revalidate`
      (answering it would only be refused as stale at consumption);
    * every flow approval named in *retire* (the flow knows its result moved
      on), validated like ``Ask.replaces``.

    Each row is locked (FOR UPDATE) and its status re-read, so an answer that
    raced the planning is never lost: a row answered meanwhile is left alone.
    Returns how many were superseded."""
    names = set()
    for rname in retire or ():
        if not isinstance(rname, str) or not NAME_RE.match(rname) or CORE_SEP in rname:
            raise ContractViolation(f"cannot retire {rname!r}")
        names.add(rname)
    rows = db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.step_id == row.step_id,
        AutopilotApproval.step_generation == row.generation,
        AutopilotApproval.status == "pending").order_by(AutopilotApproval.id)).scalars().all()
    n = 0
    for a in rows:
        latest = _latest_approval(db, ap.id, row.step_id, row.generation, a.name)
        if latest is None or latest.id != a.id:
            continue
        db.refresh(a, with_for_update=True)
        if a.status != "pending" or a.answered_decision is not None:
            continue
        if a.name in names:
            _supersede(db, ap, a, "no longer asked: what it was about moved on", now)
            n += 1
        elif not _revalidate(db, ap, a):
            _supersede(db, ap, a, "no longer current: what it was about changed", now)
            n += 1
    if n:
        _touch(ap, now)
    return n


def _mark_consumed(approvals, *, now, effect_id=None) -> None:
    for a in approvals:
        a.status, a.consumed_at, a.consumed_by_effect_id = "consumed", now, effect_id


def _consume(db, ap, row, names, *, now, effect_id=None) -> list[AutopilotApproval]:
    out = _revalidate_named(db, ap, row, names, now=now)
    _mark_consumed(out, now=now, effect_id=effect_id)
    return out


def _check_unconsumed(db, ap, row, consumes, *, exclude_ids=()):
    pending = db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.step_id == row.step_id,
        AutopilotApproval.step_generation == row.generation,
        AutopilotApproval.status.in_(("approved", "rejected")))).scalars().all()
    for a in pending:
        if CORE_SEP in a.name or a.id in exclude_ids or a.name in consumes:
            continue
        raise ContractViolation(
            f"step {row.step_id} acted without consuming the answered approval {a.name!r}")


def _new_approval(db, ap, row, *, name, kind, question, proposal, subject="", options=None,
                  artifacts=None, action=None, bound=None, cost_mc=0, max_cost_mc=None,
                  committed_mc=None, budget_after_mc=None, now) -> AutopilotApproval:
    latest = _latest_approval(db, ap.id, row.step_id, row.generation, name)
    gen = 1 if latest is None else latest.generation + 1
    a = AutopilotApproval(
        autopilot_id=ap.id, step_id=row.step_id, step_generation=row.generation, name=name,
        generation=gen, dedupe_key=f"{ap.id}:{row.step_id}:g{row.generation}:{name}:{gen}",
        kind=kind, subject=subject or "", status="pending", question=question,
        proposal=proposal or {}, options=options, artifacts=artifacts, action=action,
        action_sha256=(action or {}).get("params_sha256"), bound_versions=bound or {},
        cost_mcredits=int(cost_mc or 0), max_cost_mcredits=max_cost_mc,
        committed_at_ask_mcredits=committed_mc, budget_after_mcredits=budget_after_mc,
        comments=[], created_at=now,
    )
    db.add(a)
    db.flush()
    journal(db, ap, "approval_requested", f"{kind}: {question}", step_id=row.step_id,
            approval_id=a.id, cost_mcredits=int(cost_mc or 0), now=now,
            data={"subject": subject} if subject else None)
    return a


# ---------------------------------------------------------------------------
# APPLY (§5.3–5.4)
# ---------------------------------------------------------------------------


@dataclass
class _Applied:
    outcome: str  # launched | applied | waiting | discarded | refused | error
    effect_key: Optional[str] = None


def _reopen(db, ap, specs, step_id, reason, *, now) -> None:
    rows = {r.step_id: r for r in db.execute(select(AutopilotStep).where(
        AutopilotStep.autopilot_id == ap.id)).scalars()}
    targets = [step_id] + _dependents_closure(specs, step_id)
    for sid in targets:
        r = rows.get(sid)
        if r is None or r.status in ("pending", "skipped"):
            continue
        for a in db.execute(select(AutopilotApproval).where(
                AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.step_id == sid,
                AutopilotApproval.status.in_(("pending", "approved", "rejected")))).scalars():
            a.status = "superseded"
        prev = list(r.previous_outputs or [])
        if r.output is not None:
            prev.append({"generation": r.generation, "output": r.output})
        r.previous_outputs = prev[-10:]
        r.output, r.depends_on = None, None
        r.generation += 1
        r.reopen_count = (r.reopen_count or 0) + 1
        r.status = "active" if sid == step_id else "pending"
        r.completed_at, r.next_check_at, r.updated_at = None, None, now
        r.refusals, r.errors, r.last_refusal = 0, 0, None
        journal(db, ap, "reopened", f"step {sid} reopened: {reason}", step_id=sid, now=now)
        if r.reopen_count > MAX_REOPENS and ap.status == "active":
            _set_status(db, ap, "paused", f"step {sid} reopened more than {MAX_REOPENS} times",
                        kind="paused", now=now, step_id=sid)
    _touch(ap, now)


def _verify_needs(db, ap, specs_by_id, specs, spec, *, now) -> Optional[str]:
    """Re-verify depends_on of the transitive needs; returns the changed step id."""
    for sid in _needs_closure(specs_by_id, spec.id):
        r = db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap.id, AutopilotStep.step_id == sid)).scalar_one_or_none()
        if r is None or r.status != "completed":
            return sid
        for key, v in (r.depends_on or {}).items():
            if key_version(db, ap, key) != v:
                return sid
    return None


def _apply(session_factory, ap_id, spec, specs, flow, plan: Plan, hooks: Hooks) -> _Applied:
    now = hooks.clock()
    d = plan.decision
    if plan.spec is not None:
        spec, specs = plan.spec, plan.specs
    if d is None:
        return _Applied("discarded")
    specs_by_id = {s.id: s for s in specs}
    db = session_factory()
    try:
        gate = _gate(db, share=True)
        ap0 = db.get(ProjectAutopilot, ap_id)
        if ap0 is None:
            return _Applied("discarded")
        project = _lock_project(db, ap0.project_id)
        ap = _lock_ap(db, ap_id)
        if project is None or ap is None or gate.held or ap.version != plan.version \
                or ap.status != "active":
            db.rollback()
            return _Applied("discarded")
        row = db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap.id, AutopilotStep.step_id == spec.id)
            .with_for_update()).scalar_one()
        if row.generation != plan.step_gen or row.status != "active":
            db.rollback()
            return _Applied("discarded")

        # --- the version keys this decision relies on --------------------
        explicit: dict[str, str] = {}
        if isinstance(d, Launch):
            explicit.update(plan.target_versions)
            explicit.update(dict(d.expect))
        if isinstance(d, Complete):
            explicit.update(dict(d.depends_on))
        if isinstance(d, Ask):
            explicit.update(dict(d.about))
        need_rows = []
        for sid in _needs_closure(specs_by_id, spec.id):
            r = db.execute(select(AutopilotStep).where(
                AutopilotStep.autopilot_id == ap.id, AutopilotStep.step_id == sid)).scalar_one_or_none()
            if r is not None:
                need_rows.append(r)
        need_keys = {k for r in need_rows for k in (r.depends_on or {})}
        art_runs = {a.run_id for a in getattr(d, "artifacts", ()) or () if a.run_id}
        # the rows the consumed approvals are bound to are locked too, so the
        # revalidation at consumption cannot race a manual edit
        bound_keys: set[str] = set()
        for name in getattr(d, "consumes", ()) or ():
            a = _latest_approval(db, ap.id, row.step_id, row.generation, name)
            if a is not None:
                bound_keys |= set((a.bound_versions or {}).keys())
        runs = _lock_runs(db, _run_ids_of_keys(set(explicit) | need_keys | bound_keys) | art_runs)

        changed = _verify_needs(db, ap, specs_by_id, specs, spec, now=now)
        if changed is not None:
            r = next((x for x in need_rows if x.step_id == changed), None)
            if r is not None and r.status == "completed":
                _reopen(db, ap, specs, changed, "a dependency changed", now=now)
                db.commit()
            else:
                db.rollback()
            return _Applied("discarded")
        for key, v in explicit.items():
            if key_version(db, ap, key) != v:
                db.rollback()
                return _Applied("discarded")
        if isinstance(d, (Complete, Ask)):
            for key in list(explicit) + [f"run:{r}" for r in art_runs]:
                if key.startswith("run:"):
                    run = runs.get(key[4:])
                    if run is None or _status(run.status) not in TERMINAL_RUN_STATUSES:
                        raise ContractViolation(
                            f"{type(d).__name__} names the non-terminal run {key[4:]}")

        if isinstance(d, (Wait, Ask, Launch, Compute)):
            # Stale cards (2026-11-03): pending approvals of this step whose
            # bound versions / artifacts no longer revalidate, and the ones the
            # flow retires, are superseded in the decision's transaction (only
            # PENDING ones: what a Launch consumes is answered, never touched).
            _sweep_pending(db, ap, row, getattr(d, "retire", ()) or (), now=now)
        if isinstance(d, Wait):
            return _apply_wait(db, ap, row, d, now=now)
        if isinstance(d, Complete):
            return _apply_complete(db, ap, row, d, now=now)
        if isinstance(d, Fail):
            _check_unconsumed(db, ap, row, d.consumes)
            _consume(db, ap, row, d.consumes, now=now)
            row.status, row.reason, row.updated_at = "failed", d.reason, now
            _set_status(db, ap, "failed", f"step {row.step_id} failed: {d.reason}",
                        kind="failed", now=now, step_id=row.step_id)
            db.commit()
            return _Applied("applied")
        if isinstance(d, Ask):
            return _apply_ask(db, ap, project, row, spec, d, now=now)
        if isinstance(d, Compute):
            return _apply_compute(db, ap, row, d, hooks, now=now)
        if isinstance(d, Judge):
            return _apply_judge(db, ap, row, d, now=now)
        if isinstance(d, Launch):
            return _apply_launch(db, session_factory, ap, project, row, spec, flow, d, plan,
                                 hooks, now=now)
        raise ContractViolation(f"advance returned {type(d).__name__}, not a Decision")
    except _Discard:
        return _Applied("discarded")
    finally:
        db.close()


def _apply_wait(db, ap, row, d: Wait, *, now) -> _Applied:
    grace = timedelta(seconds=settings.AUTOPILOT_REJECT_GRACE_SEC)
    rejected = db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.step_id == row.step_id,
        AutopilotApproval.step_generation == row.generation,
        AutopilotApproval.status == "rejected")).scalars().all()
    for a in rejected:
        if a.answered_at is not None and now - a.answered_at > grace:
            a.status, a.consumed_at = "consumed", now
            _set_status(db, ap, "stopped",
                        f"the operator rejected {a.kind} ({a.name})"
                        + (f": {a.answer_comment}" if a.answer_comment else ""),
                        kind="stopped", now=now, step_id=row.step_id)
            db.commit()
            return _Applied("applied")
    if row.reason != d.reason:
        journal(db, ap, "decision", f"waiting: {d.reason}", step_id=row.step_id, now=now)
    row.reason = d.reason
    row.next_check_at = now + timedelta(seconds=max(1, int(d.recheck_sec)))
    row.refusals, row.errors, row.updated_at = 0, 0, now
    _touch(ap, now)
    db.commit()
    return _Applied("waiting")


def _apply_complete(db, ap, row, d: Complete, *, now) -> _Applied:
    _check_unconsumed(db, ap, row, d.consumes)
    _consume(db, ap, row, d.consumes, now=now)
    # A completed step is never advanced again (a reopen is a new generation):
    # an approval of it still pending could only be answered into the void.
    for a in db.execute(select(AutopilotApproval).where(
            AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.step_id == row.step_id,
            AutopilotApproval.step_generation == row.generation,
            AutopilotApproval.status == "pending")).scalars().all():
        _supersede(db, ap, a, f"step {row.step_id} completed", now)
    out = dict(d.output)
    if len(json.dumps(out, default=str).encode()) > OUTPUT_MAX_BYTES:
        raise ContractViolation(f"step {row.step_id}: output larger than 64 KiB")
    row.status, row.output, row.depends_on = "completed", out, dict(d.depends_on)
    row.reason, row.completed_at, row.updated_at = d.reason or "completed", now, now
    _touch(ap, now)
    journal(db, ap, "completed", f"step {row.step_id} completed" + (f": {d.reason}" if d.reason else ""),
            step_id=row.step_id, now=now)
    db.commit()
    return _Applied("applied")


_HANDLER_WORDS = {"create_run": "a new run", "rerun_segments": "rerun segments",
                  "segment_fill_start": "a VOID fill", "segment_fill_batch_start": "VOID fills",
                  "retry_run": "retry the run", "transcribe_project": "transcribe the source",
                  "localisation_prompt": "translate the script",
                  "localisation_create_run": "a localised run",
                  "localisation_create_caption": "burn captions"}


def _core_title(kind: str, handler: str, est_mc: int, after_mc: int, budget_mc: int) -> str:
    """Card title of a core regeneration / overrun approval (doc §4.2)."""
    what = _HANDLER_WORDS.get(handler, handler.replace("_", " "))
    if kind == "regeneration":
        return f"Regenerate: {what} (~{est_mc / 1000:.0f} credits)"
    return (f"Over budget: {what} (~{est_mc / 1000:.0f} credits, "
            f"{after_mc / 1000:.0f} of {budget_mc / 1000:.0f} after)")


def _apply_ask(db, ap, project, row, spec, d: Ask, *, now) -> _Applied:
    if not NAME_RE.match(d.name):
        raise ContractViolation(f"invalid approval name {d.name!r}")
    if d.kind not in APPROVAL_KINDS or d.kind not in spec.approvals:
        raise ContractViolation(f"step {spec.id} may not ask {d.kind!r}")
    if d.kind == "overrun":
        raise ContractViolation("flows never ask an overrun approval (the core does)")
    if d.kind == "regeneration" and not d.options:
        raise ContractViolation("a flow asks regeneration only with options")
    from .links import CONSEQUENCE_MAX, TITLE_KEY, clip_text
    if TITLE_KEY in (d.proposal or {}):
        raise ContractViolation(f"proposal key {TITLE_KEY!r} is reserved: set Ask.title instead")
    latest = _latest_approval(db, ap.id, row.step_id, row.generation, d.name)
    if latest is not None and latest.status != "superseded":
        db.rollback()  # already asked (pending / answered): nothing to write
        return _Applied("waiting")
    # Approvals this one replaces: locked (approvals, lock 4 -- after the run
    # locks _apply took) and re-read here, so an answer racing the planning is
    # never lost: an answered one discards this Ask; a concurrent answer()
    # blocks on the row lock and then finds it superseded (409).
    replaced = []
    for rname in dict.fromkeys(getattr(d, "replaces", ()) or ()):
        if not isinstance(rname, str) or not NAME_RE.match(rname) or rname == d.name:
            raise ContractViolation(f"Ask {d.name!r} cannot replace {rname!r}")
        old = _latest_approval(db, ap.id, row.step_id, row.generation, rname)
        if old is None:
            continue
        db.refresh(old, with_for_update=True)
        if old.status in ("approved", "rejected"):
            db.rollback()
            return _Applied("discarded")
        if old.status == "pending":
            replaced.append(old)
    bound = {k: key_version(db, ap, k) for k in d.about}
    artifacts = []
    for art in d.artifacts:
        if art.run_id:
            key = f"run:{art.run_id}"
        elif art.effect_id:
            eff = db.get(AutopilotEffect, art.effect_id)
            if eff is None or eff.autopilot_id != ap.id or eff.state != "succeeded":
                raise ContractViolation("artifact effect is not a succeeded compute of this autopilot")
            out_dir = os.path.realpath((eff.result or {}).get("out_dir") or "/nonexistent")
            if not os.path.realpath(art.path).startswith(out_dir + os.sep):
                raise ContractViolation("artifact path is outside its compute attempt dir")
            key = f"effect:{art.effect_id}"
        else:
            raise ContractViolation("an artifact needs a producer (run_id or effect_id)")
        if not os.path.isfile(art.path):
            db.rollback()
            return _Applied("waiting")  # retryable: the source is not there (yet)
        bound[key] = key_version(db, ap, key)
        digest, size, _dest = _copy_artifact(project.id, art.path)
        artifacts.append({"label": art.label, "kind": art.kind, "producer": key,
                          "source_path": art.path, "sha256": digest, "size": size})
    options = []
    committed_mc = spend_view.figures_mcredits(db, ap.project_id)["committed"]
    for o in d.options:
        entry = {"id": o.id, "label": o.label, "action": None, "cost_mcredits": 0}
        consequence = clip_text(getattr(o, "consequence", "") or "", CONSEQUENCE_MAX)
        if consequence:
            entry["consequence"] = consequence   # presentation only (AICP-376)
        if o.handler:
            h = registry.get_handler(o.handler)
            if h is None:
                raise ContractViolation(f"option {o.id}: unknown handler {o.handler!r}")
            p = h.validate(db, project, dict(o.params or {}))
            est = _mc(h.estimate(db, project, p))
            tg = h.targets(db, project, p)
            for rid in h.target_run_ids(tg):
                bound[f"run:{rid}"] = run_version(db, rid)
            entry["action"] = {"handler": h.name, "handler_version": h.version, "params": p,
                               "params_sha256": _sha([h.name, h.version, p])}
            entry["cost_mcredits"] = est
        options.append(entry)
    proposal = dict(d.proposal)
    title = clip_text(getattr(d, "title", "") or "")
    if title:
        proposal[TITLE_KEY] = title             # presentation only (AICP-376, doc §4.2)
    _new_approval(db, ap, row, name=d.name, kind=d.kind, question=d.question,
                  proposal=proposal, subject=d.subject, options=options or None,
                  artifacts=artifacts or None, bound=bound, cost_mc=_mc(d.cost_credits),
                  committed_mc=committed_mc, now=now)
    for old in replaced:
        _supersede(db, ap, old, f"replaced by {d.name}", now)
    row.reason, row.updated_at = f"waiting for the operator: {d.question}", now
    _touch(ap, now)
    db.commit()
    return _Applied("applied")


def _apply_compute(db, ap, row, d: Compute, hooks, *, now) -> _Applied:
    if not NAME_RE.match(d.name):
        raise ContractViolation(f"invalid effect name {d.name!r}")
    version = registry.compute_version(d.func)
    if version is None:
        raise ContractViolation(f"unknown compute function {d.func!r}")
    key = f"{ap.id}:{row.step_id}:g{row.generation}:{d.name}"
    existing = db.execute(select(AutopilotEffect).where(
        AutopilotEffect.idempotency_key == key)).scalar_one_or_none()
    if existing is not None:
        db.rollback()
        return _Applied("waiting")
    params = {"args": dict(d.args), "timeout_sec": int(d.timeout_sec)}
    eff_id = str(uuid.uuid4())
    eff = AutopilotEffect(
        id=eff_id, rq_job_id=f"autopilot-compute-{eff_id}", autopilot_id=ap.id, step_id=row.step_id, step_generation=row.generation, name=d.name,
        idempotency_key=key, kind="compute", handler=d.func, handler_version=version,
        params=params, params_sha256=_sha([d.func, version, params]), paid=False,
        estimate_mcredits=0, state="queued", created_at=now, updated_at=now,
    )
    db.add(eff)
    ap.outstanding = (ap.outstanding or 0) + 1
    row.reason, row.updated_at = d.reason, now
    journal(db, ap, "decision", f"compute {d.name} ({d.func}): {d.reason}", step_id=row.step_id,
            effect_id=eff.id, now=now)
    _touch(ap, now)
    db.commit()
    try:
        hooks.enqueue(eff.id, eff.rq_job_id, queue=registry.compute_queue(d.func, version))
    except Exception:
        log.exception("autopilot: enqueue of compute %s failed (recovery re-enqueues)", eff.id)
    return _Applied("applied")


def _apply_judge(db, ap, row, d: Judge, *, now) -> _Applied:
    kind = registry.get_judgement(d.kind)
    if kind is None:
        raise ContractViolation(f"unknown judgement kind {d.kind!r}")
    try:
        payload = kind.input_model.model_validate(dict(d.payload)).model_dump(mode="json")
    except Exception as exc:
        raise ContractViolation(f"judgement {d.kind} payload invalid: {exc}") from None
    key = f"{ap.id}:{row.step_id}:g{row.generation}:{d.name}"
    if db.execute(select(AutopilotEffect.id).where(
            AutopilotEffect.idempotency_key == key)).scalar_one_or_none():
        db.rollback()
        return _Applied("waiting")
    eff = AutopilotEffect(
        autopilot_id=ap.id, step_id=row.step_id, step_generation=row.generation, name=d.name,
        idempotency_key=key, kind="judgement", handler=d.kind, handler_version=kind.version,
        params=payload, params_sha256=_sha(payload), paid=False, estimate_mcredits=0,
        state="prepared", created_at=now, updated_at=now,
    )
    db.add(eff)
    ap.outstanding = (ap.outstanding or 0) + 1
    journal(db, ap, "decision", f"judgement {d.name} ({d.kind}): {d.reason}", step_id=row.step_id,
            now=now)
    _touch(ap, now)
    db.commit()
    return _Applied("applied")


def _apply_launch(db, session_factory, ap, project, row, spec, flow, d: Launch, plan: Plan,
                  hooks: Hooks, *, now) -> _Applied:
    if not NAME_RE.match(d.name):
        raise ContractViolation(f"invalid effect name {d.name!r}")
    handler = registry.get_handler(d.handler, (getattr(flow, "handlers", None) or {}).get(d.handler))
    if handler is None:
        raise ContractViolation(f"unknown handler {d.handler!r}")
    if handler.paid and not spec.paid:
        raise ContractViolation(f"step {spec.id} is free but launched the paid handler {handler.name}")
    params = plan.params
    targets = plan.targets or {}
    key = f"{ap.id}:{row.step_id}:g{row.generation}:{d.name}"
    if db.execute(select(AutopilotEffect.id).where(
            AutopilotEffect.idempotency_key == key)).scalar_one_or_none():
        db.rollback()
        return _Applied("waiting")  # already admitted; the step observes it
    est = handler.estimate(db, project, params)
    est_mc = _mc(est)
    sha = _sha([handler.name, handler.version, params])
    tol = float(ap.tolerance)
    committed = spend_view.figures_mcredits(db, ap.project_id)["committed"]

    consumed = []
    authorizing: Optional[AutopilotApproval] = None
    _check_unconsumed(db, ap, row, d.consumes)
    # Revalidate what the launch consumes BEFORE any budget change is made, so a
    # stale approval can never leave a raised budget behind.
    listed = _revalidate_named(db, ap, row, d.consumes, now=now) if d.consumes else []
    if d.consumes:
        for name in d.consumes:
            a = _latest_approval(db, ap.id, row.step_id, row.generation, name)
            if a is not None and a.chosen_option and a.status == "approved":
                opt = next((o for o in (a.options or []) if o["id"] == a.chosen_option), None)
                if opt and opt.get("action") and opt["action"].get("params_sha256") == sha:
                    authorizing = a

    if handler.paid:
        # 1. no budget yet -> the core-generated budget approval (§6.2)
        if ap.approved_budget_mcredits is None:
            name = f"{d.name}{CORE_SEP}budget"
            latest = _latest_approval(db, ap.id, row.step_id, row.generation, name)
            if latest is not None and latest.status == "pending":
                db.rollback()
                return _Applied("waiting")
            lines = [{"step_id": None, "label": "already committed", "credits": committed / 1000.0}]
            lines += plan.prediction or []
            total = int(sum(_mc(x["credits"]) for x in lines))
            total = ((total + 999) // 1000) * 1000
            _new_approval(db, ap, row, name=name, kind="budget",
                          question=f"Approve the predicted project budget of {total / 1000:.0f} credits",
                          proposal={"lines": lines, "total_credits": total / 1000.0,
                                    "total_usd": round(_usd(total / 1000.0), 2),
                                    "tolerance": tol,
                                    "_title": f"Approve the budget: {total / 1000:.0f} credits"},
                          cost_mc=total, committed_mc=committed, now=now)
            row.reason = "waiting for the budget approval"
            _touch(ap, now)
            db.commit()
            return _Applied("waiting")
        regen = bool(d.regeneration or handler.always_regeneration)
        B = int(ap.approved_budget_mcredits)
        if regen or committed + est_mc > B * (1 + tol):
            kind = "regeneration" if regen else "overrun"
            a = authorizing
            if a is None:
                name = f"{d.name}{CORE_SEP}{kind}"
                a = _latest_approval(db, ap.id, row.step_id, row.generation, name)
                if a is not None and a.action_sha256 != sha and a.status in ("pending", "approved"):
                    _supersede(db, ap, a, "the action changed", now)
                    a = None
                if a is None or a.status == "superseded":
                    after = max(B + est_mc, committed + est_mc) if regen else committed + est_mc
                    bound = dict(plan.target_versions)
                    _new_approval(
                        db, ap, row, name=name, kind=kind,
                        question=(f"{'Regenerate' if regen else 'Overrun'}: {handler.name} "
                                  f"for ~{est_mc / 1000:.0f} credits ({d.reason})"),
                        proposal={"handler": handler.name, "params": params,
                                  "estimate_credits": est_mc / 1000.0,
                                  "committed_credits": committed / 1000.0,
                                  "approved_budget_credits": B / 1000.0,
                                  "budget_after_credits": after / 1000.0,
                                  "_title": _core_title(kind, handler.name, est_mc, after, B)},
                        action={"handler": handler.name, "handler_version": handler.version,
                                "params": params, "params_sha256": sha},
                        bound=bound, cost_mc=est_mc, max_cost_mc=((est_mc + 999) // 1000) * 1000,
                        committed_mc=committed, budget_after_mc=after, now=now)
                    row.reason = f"waiting for the operator: {kind} of {d.name}"
                    _touch(ap, now)
                    db.commit()
                    return _Applied("waiting")
                if a.status == "pending":
                    db.rollback()
                    return _Applied("waiting")
                if a.status == "rejected":
                    a.status, a.consumed_at = "consumed", now
                    _set_status(db, ap, "stopped",
                                f"the operator rejected the {kind} of {d.name}"
                                + (f": {a.answer_comment}" if a.answer_comment else ""),
                                kind="stopped", now=now, step_id=row.step_id)
                    db.commit()
                    return _Applied("applied")
                if a.status != "approved":
                    db.rollback()
                    return _Applied("waiting")
                db.refresh(a, with_for_update=True)
                ok = (a.max_cost_mcredits is None or est_mc <= a.max_cost_mcredits) \
                    and (a.budget_after_mcredits is None
                         or committed + est_mc <= a.budget_after_mcredits * (1 + tol)) \
                    and _revalidate(db, ap, a)
                if not ok:
                    _supersede(db, ap, a, "its cost, the budget or its targets changed", now)
                    _touch(ap, now)
                    db.commit()
                    return _Applied("discarded")
                after = a.budget_after_mcredits or B
                ap.approved_budget_mcredits = max(B, int(after))
                consumed.append(a)
            else:
                opt = next(o for o in a.options if o["id"] == a.chosen_option)
                c_ask = (committed if a.committed_at_ask_mcredits is None
                         else a.committed_at_ask_mcredits)
                after = max(B + opt["cost_mcredits"], c_ask + opt["cost_mcredits"])
                max_cost = ((int(opt["cost_mcredits"]) + 999) // 1000) * 1000
                if est_mc > max_cost or committed + est_mc > after * (1 + tol):
                    _supersede(db, ap, a, "its cost or the budget changed", now)
                    _touch(ap, now)
                    db.commit()
                    return _Applied("discarded")
                ap.approved_budget_mcredits = max(B, int(after))
            journal(db, ap, "budget", f"{kind} approved: budget now "
                    f"{ap.approved_budget_mcredits / 1000:.0f} credits", step_id=row.step_id, now=now)

    # consume the flow-named approvals (revalidated)
    consumed += [a for a in listed if a not in consumed]

    # --- the launch protocol (§5.4) ------------------------------------------
    eff = AutopilotEffect(
        id=str(uuid.uuid4()), autopilot_id=ap.id, step_id=row.step_id,
        step_generation=row.generation, name=d.name, idempotency_key=key, kind="action",
        handler=handler.name, handler_version=handler.version, params=params,
        params_sha256=sha, paid=bool(handler.paid), estimate_mcredits=est_mc,
        targets=targets, target_versions=dict(plan.target_versions), created_refs={},
        state="launching", created_at=now, updated_at=now,
    )
    for a in consumed:
        a.status, a.consumed_at, a.consumed_by_effect_id = "consumed", now, eff.id
    eff.approval_id = consumed[0].id if consumed else None
    db.add(eff)
    spend_view.register_scopes(db, ap.project_id, handler.spend_scopes(params, targets),
                               autopilot_id=ap.id, source="admission")
    ap.outstanding = (ap.outstanding or 0) + 1
    row.reason, row.updated_at = d.reason, now
    _touch(ap, now)
    journal(db, ap, "launch", f"launching {d.name} ({handler.name}): {d.reason}",
            step_id=row.step_id, effect_id=eff.id, cost_mcredits=est_mc, now=now,
            data={"params": params})

    ap_id, step_id = ap.id, row.step_id  # survive the rollback below
    error: Optional[BaseException] = None
    try:
        db.flush()  # controlled flush: a concurrent duplicate collides HERE
        with spend_view.autopilot_admission():
            result = handler.execute(db, project, params, eff)
        db.commit()  # no-op when the endpoint already committed
    except HTTPException as exc:
        error = ActionRefused(exc.status_code, str(exc.detail)) if 400 <= exc.status_code < 500 \
            else exc
    except BaseException as exc:  # noqa: BLE001 — classified below, never by its text
        error = exc
    if error is not None:
        db.rollback()
        db.close()
        return _after_failed_launch(session_factory, ap_id, step_id, d.name, key, error,
                                    hooks)
    project_id = ap.project_id
    db.close()
    # T2: bookkeeping in a new transaction
    try:
        db2 = session_factory()
        try:
            _lock_project(db2, project_id)  # lock order: project, then autopilot
            ap2 = _lock_ap(db2, ap_id)
            e2 = db2.execute(select(AutopilotEffect).where(
                AutopilotEffect.idempotency_key == key)).scalar_one_or_none()
            if e2 is None:
                # The admission committed WITHOUT its effect row: the handler
                # broke rule (g) (a rollback inside execute dropped it). Never
                # guess: stop launching and say so.
                log.error("autopilot %s: %s admitted but its effect row is missing", ap_id, key)
                _set_status(db2, ap2, "paused",
                            f"{d.name}: the admission committed without its effect row "
                            "(handler contract (g) broken) — check the project before resuming",
                            kind="paused", now=hooks.clock(), step_id=step_id)
                db2.commit()
            elif e2.state == "launching":
                merged = dict(e2.result or {})  # a handler may have written result itself
                merged.update(result or {})
                e2.state, e2.result, e2.updated_at = "launched", merged, hooks.clock()
                ap2.outstanding = max(0, (ap2.outstanding or 1) - 1)
                _touch(ap2)
                journal(db2, ap2, "observed", f"{d.name} admitted", step_id=step_id,
                        effect_id=e2.id, data=result)
                db2.commit()
        finally:
            db2.close()
    except Exception:
        log.exception("autopilot %s: post-launch bookkeeping failed (reconciled next tick)", ap_id)
    return _Applied("launched", key)


def _usd(credits: float) -> float:
    from .. import spend

    return spend.usd(credits)


def _after_failed_launch(session_factory, ap_id, step_id, name, key, error, hooks) -> _Applied:
    now = hooks.clock()
    db = session_factory()
    try:
        present = db.execute(select(AutopilotEffect.id).where(
            AutopilotEffect.idempotency_key == key)).scalar_one_or_none()
        if present is not None:
            return _Applied("discarded")  # reconciled in step A of the next tick
        ap0 = db.get(ProjectAutopilot, ap_id)
        _lock_project(db, ap0.project_id)
        ap = _lock_ap(db, ap_id)
        row = db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap_id, AutopilotStep.step_id == step_id)).scalar_one()
        if isinstance(error, ContractViolation):
            row.status, row.reason = "failed", str(error)
            _set_status(db, ap, "failed", f"contract violation in step {step_id}: {error}",
                        kind="failed", now=now, step_id=step_id)
            db.commit()
            return _Applied("error")
        if isinstance(error, ActionRefused):
            detail = f"{error.status}: {error.detail}"[:1000]
            prev = row.last_refusal or {}
            row.refusals = (row.refusals or 0) + 1
            row.last_refusal = {"name": name, "status": error.status, "detail": error.detail[:1000],
                                "at": now.isoformat()}
            base = 60 * (2 ** min(row.refusals - 1, 5))
            if error.status == 429:
                base = max(base, 600)
            row.next_check_at = now + timedelta(seconds=min(base, 1800))
            row.reason = f"refused: {detail}"
            if prev.get("detail") != error.detail[:1000]:
                journal(db, ap, "refused", f"{name} refused: {detail}", step_id=step_id, now=now)
            if row.refusals >= MAX_REFUSALS and ap.status == "active":
                _set_status(db, ap, "paused", f"{name} refused {row.refusals} times: {detail}",
                            kind="paused", now=now, step_id=step_id)
            _touch(ap, now)
            db.commit()
            return _Applied("refused")
        _step_error(db, ap, row, error, now=now)
        db.commit()
        return _Applied("error")
    finally:
        db.close()


def _step_error(db, ap, row, error, *, now) -> None:
    row.errors = (row.errors or 0) + 1
    row.next_check_at = now + timedelta(seconds=min(30 * (2 ** (row.errors - 1)), 900))
    msg = f"{type(error).__name__}: {error}"[:1000]
    row.reason = f"error: {msg}"
    journal(db, ap, "warning", f"step {row.step_id} error: {msg}", step_id=row.step_id, now=now)
    if row.errors >= MAX_STEP_ERRORS and ap.status == "active":
        _set_status(db, ap, "paused", f"step {row.step_id} failed {row.errors} times in a row: {msg}",
                    kind="paused", now=now, step_id=row.step_id)
    _touch(ap, now)


# ---------------------------------------------------------------------------
# Step A — reconcile (any status, observation only)
# ---------------------------------------------------------------------------


def _reconcile(session_factory, ap_id: str, hooks: Hooks) -> list[tuple[str, str]]:
    """Returns compute (effect_id, job_id) pairs to re-enqueue after commit."""
    now = hooks.clock()
    requeue: list[tuple[str, str]] = []
    db = session_factory()
    try:
        ap0 = db.get(ProjectAutopilot, ap_id)
        if ap0 is None:
            return []
        project = _lock_project(db, ap0.project_id)
        ap = _lock_ap(db, ap_id)
        if project is None or ap is None:
            db.rollback()
            return []
        live = ap.status in LIVE_AUTOPILOT_STATUSES
        flow = registry.get_flow(ap.flow_id, ap.flow_version)

        # §5.7 manual paid actions
        events = db.execute(select(ProjectManualSpendEvent).where(
            ProjectManualSpendEvent.project_id == ap.project_id,
            ProjectManualSpendEvent.seen_at.is_(None),
            ProjectManualSpendEvent.at >= ap.created_at)).scalars().all()
        for ev in events:
            ev.seen_at = now
            if ap.status in LIVE_AUTOPILOT_STATUSES:
                msg = (f"manual paid action: {ev.kind} ({ev.scope_key}, ~{ev.mcredits / 1000:.0f} "
                       "credits) — Resume to continue")
                if ap.status == "active":
                    ap.status, ap.status_reason = "paused", msg
                journal(db, ap, "paused", msg, now=now, data={"scope": ev.scope_key})

        # answered budget approvals (their bound runs locked once, sorted, §5.2)
        answered_budgets = db.execute(select(AutopilotApproval).where(
            AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.kind == "budget",
            AutopilotApproval.status.in_(("approved", "rejected")))).scalars().all()
        _lock_runs(db, {rid for a in answered_budgets if a.answered_decision == "approve"
                        for rid in _run_ids_of_keys((a.bound_versions or {}).keys())})
        step_rows = {r.step_id: r for r in db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap.id)).scalars()}
        # the step list as the CURRENT (locked) params define it, not as the
        # stored step rows last saw it
        current_ids = ({s.id for s in flow.steps(project, dict(ap.params or {}))}
                       if flow is not None and answered_budgets else set())
        for a in (answered_budgets if flow is not None else []):  # missing flow: keep answers
            srow = step_rows.get(a.step_id)
            if srow is None or srow.generation != a.step_generation or srow.status != "active" \
                    or a.step_id not in current_ids:
                _supersede(db, ap, a, "its step was skipped or reopened", now)
                continue
            if a.answered_decision == "approve" and not _revalidate(db, ap, a):
                _supersede(db, ap, a, "what it was about changed since it was answered", now)
                continue
            a.status, a.consumed_at = "consumed", now
            if a.answered_decision == "approve":
                total = _mc(float((a.proposal or {}).get("total_credits") or 0.0))
                ap.approved_budget_mcredits = total
                journal(db, ap, "budget", f"budget approved: {total / 1000:.0f} credits",
                        approval_id=a.id, now=now)
            elif live:
                _set_status(db, ap, "stopped", "the operator rejected the budget"
                            + (f": {a.answer_comment}" if a.answer_comment else ""),
                            kind="stopped", now=now)
                live = False

        # effects
        effects = db.execute(select(AutopilotEffect).where(
            AutopilotEffect.autopilot_id == ap.id,
            AutopilotEffect.state.notin_(EFFECT_TERMINAL_STATES))).scalars().all()
        for eff in effects:
            if eff.kind == "action" and eff.state == "launching":
                h = registry.get_handler(eff.handler, eff.handler_version)
                if h is None:
                    if ap.status == "active":
                        _set_status(db, ap, "paused", f"handler {eff.handler} v{eff.handler_version} "
                                    "is not deployed; outstanding effect left untouched",
                                    kind="paused", now=now)
                    continue
                verdict = h.reconcile(db, eff)
                eff.state, eff.updated_at = verdict, now
                journal(db, ap, "observed", f"{eff.name}: reconciled as {verdict}",
                        step_id=eff.step_id, effect_id=eff.id, now=now)
                if verdict == "not_launched" and eff.approval_id:
                    a = db.get(AutopilotApproval, eff.approval_id)
                    if a is not None and a.status == "consumed":
                        a.status, a.consumed_at, a.consumed_by_effect_id = "approved", None, None
                if verdict == "unknown" and ap.status == "active":
                    _set_status(db, ap, "paused", f"ambiguous launch of {eff.name}; check "
                                f"{eff.targets or eff.created_refs}", kind="paused", now=now)
            elif eff.kind == "compute":
                if registry.get_compute(eff.handler, eff.handler_version) is None:
                    continue
                try:
                    alive = hooks.is_live(eff)
                except Exception:
                    log.exception("autopilot: compute liveness check failed")
                    continue
                if alive:
                    eff.not_live_since = None
                    continue
                if eff.state == "running" and eff.claim_expires_at and eff.claim_expires_at > now:
                    pass  # within its claim: a slow-to-register worker; wait for the grace
                if eff.not_live_since is None:
                    eff.not_live_since = now
                elif (now - eff.not_live_since).total_seconds() >= COMPUTE_NOT_LIVE_GRACE_SEC:
                    if (eff.attempts or 0) >= COMPUTE_MAX_ATTEMPTS \
                            or (eff.requeues or 0) >= COMPUTE_MAX_ATTEMPTS:
                        eff.state, eff.error = "failed", "the compute job was lost too often"
                        journal(db, ap, "observed", f"compute {eff.name} failed: lost",
                                step_id=eff.step_id, effect_id=eff.id, now=now)
                    else:
                        if eff.state == "running":
                            eff.state, eff.claim_token = "queued", None
                        else:  # lost before any worker claimed it (claims count attempts)
                            eff.requeues = (eff.requeues or 0) + 1
                        eff.not_live_since = None
                        requeue.append((eff.id, eff.rq_job_id,
                                        registry.compute_queue(eff.handler, eff.handler_version)))
                        journal(db, ap, "observed", f"compute {eff.name} re-enqueued (job lost)",
                                step_id=eff.step_id, effect_id=eff.id, now=now)
                eff.updated_at = now
            elif eff.kind == "judgement" and eff.state in ("sending", "submitted"):
                _poll_judgement(db, ap, eff, hooks, now=now)

        # spend attribution
        spend_view.register_scopes(db, ap.project_id, spend_view.project_scopes(db, ap.project_id),
                                   autopilot_id=ap.id, source="scan")

        if live and flow is not None:
            specs = list(flow.steps(project, dict(ap.params or {})))
            # depends_on reopening
            for r in db.execute(select(AutopilotStep).where(
                    AutopilotStep.autopilot_id == ap.id,
                    AutopilotStep.status == "completed")).scalars().all():
                for key, v in (r.depends_on or {}).items():
                    try:
                        cur = key_version(db, ap, key)
                    except ContractViolation:
                        cur = None
                    if cur != v:
                        _reopen(db, ap, specs, r.step_id, f"{key} changed", now=now)
                        break
            # project usable?
            pst = _status(project.status)
            created = _naive(project.created_at) or now
            if pst == "failed" and ap.status == "active":
                _set_status(db, ap, "paused", "analysis failed: re-analyze the project, then Resume",
                            kind="paused", now=now)
            elif pst in ("created", "analyzing") and ap.status == "active" and \
                    (now - created).total_seconds() > settings.AUTOPILOT_STALL_SEC:
                _set_status(db, ap, "paused", f"the project has been {pst} for too long "
                            "(lost analysis?): re-analyze, then Resume", kind="paused", now=now)
            # stall watchdog
            for eff in db.execute(select(AutopilotEffect).where(
                    AutopilotEffect.autopilot_id == ap.id, AutopilotEffect.kind == "action",
                    AutopilotEffect.state == "launched")).scalars():
                run_ids = list((eff.targets or {}).get("runs", [])) + \
                    list((eff.created_refs or {}).get("runs", []))
                for rid in run_ids:
                    run = db.get(Run, rid)
                    if run is None or _status(run.status) in TERMINAL_RUN_STATUSES:
                        continue
                    upd = _naive(run.updated_at)
                    if upd and (now - upd).total_seconds() > settings.AUTOPILOT_STALL_SEC \
                            and ap.status == "active":
                        _set_status(db, ap, "paused", f"run {rid} has not moved for "
                                    f"{settings.AUTOPILOT_STALL_SEC // 3600} h", kind="paused", now=now)
        elif live and flow is None and ap.status == "active":
            _set_status(db, ap, "paused", f"flow {ap.flow_id} v{ap.flow_version} is no longer deployed",
                        kind="paused", now=now)

        ap.outstanding = db.execute(select(func.count()).select_from(AutopilotEffect).where(
            AutopilotEffect.autopilot_id == ap.id,
            AutopilotEffect.state.notin_(EFFECT_TERMINAL_STATES))).scalar_one()
        _touch(ap, now)
        db.commit()
    finally:
        db.close()
    return requeue


def _poll_judgement(db, ap, eff, hooks, *, now) -> None:
    port = hooks.get_port()
    try:
        st = port.poll(eff.id)
    except Exception:
        log.exception("autopilot: judgement poll failed")
        return
    if st.state == "succeeded":
        kind = registry.get_judgement(eff.handler, eff.handler_version)
        try:
            result = kind.output_model.model_validate(st.result or {}).model_dump(mode="json")
            eff.state, eff.result = "succeeded", result
        except Exception as exc:
            eff.state, eff.error = "failed", f"invalid judgement result: {exc}"[:2000]
    elif st.state == "failed":
        eff.state, eff.error = "failed", (st.error or "failed")[:2000]
    elif st.state == "unknown_to_runtime" and eff.state == "submitted":
        try:
            port.fence(eff.id, eff.params_sha256)
        except Exception:
            log.exception("autopilot: judgement fence failed")
            return
        eff.state, eff.error = "unknown", "the runtime lost the invocation (fenced)"
    else:
        return
    eff.updated_at = now
    journal(db, ap, "observed", f"judgement {eff.name}: {eff.state}", step_id=eff.step_id,
            effect_id=eff.id, now=now)


def _send_judgements(session_factory, ap_id: str, hooks: Hooks) -> None:
    """New work: only for an active autopilot with the gate open (§8)."""
    port = hooks.get_port()
    db = session_factory()
    try:
        ids = [i for (i,) in db.execute(select(AutopilotEffect.id).where(
            AutopilotEffect.autopilot_id == ap_id, AutopilotEffect.kind == "judgement",
            AutopilotEffect.state.in_(("prepared", "sending"))))]
    finally:
        db.close()
    if not ids:
        return
    if not port.available():
        return
    for eid in ids:
        db = session_factory()
        try:
            gate = _gate(db, share=True)
            ap0 = db.get(ProjectAutopilot, ap_id)
            _lock_project(db, ap0.project_id)
            ap = _lock_ap(db, ap_id)
            eff = db.get(AutopilotEffect, eid)
            if gate.held or ap.status != "active" or eff.state not in ("prepared", "sending"):
                db.rollback()
                continue
            if eff.state == "prepared":
                eff.state, eff.updated_at = "sending", hooks.clock()
                _touch(ap)
                db.commit()
                gate = _gate(db, share=True)  # tx B: the send holds the gate share
                _lock_project(db, ap0.project_id)
                ap = _lock_ap(db, ap_id)
                eff = db.get(AutopilotEffect, eid)
                if gate.held or ap.status != "active" or eff.state != "sending":
                    db.rollback()
                    continue
            try:
                port.submit(eff.id, eff.handler, dict(eff.params or {}), eff.params_sha256)
            except Exception:
                log.exception("autopilot: judgement submit of %s failed (stays sending)", eid)
                db.rollback()
                continue
            ap = _lock_ap(db, ap_id)
            eff.state, eff.updated_at = "submitted", hooks.clock()
            _touch(ap)
            db.commit()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------


def _take_lease(session_factory, ap_id: str, token: str, now: datetime) -> bool:
    db = session_factory()
    try:
        res = db.execute(update(ProjectAutopilot).where(
            ProjectAutopilot.id == ap_id,
            or_(ProjectAutopilot.lease_token.is_(None), ProjectAutopilot.lease_expires_at < now),
        ).values(lease_token=token, lease_expires_at=now + timedelta(seconds=LEASE_SEC)))
        db.commit()
        return res.rowcount == 1
    finally:
        db.close()


def _release_lease(session_factory, ap_id: str, token: str) -> None:
    db = session_factory()
    try:
        db.execute(update(ProjectAutopilot).where(
            ProjectAutopilot.id == ap_id, ProjectAutopilot.lease_token == token,
        ).values(lease_token=None, lease_expires_at=None))
        db.commit()
    finally:
        db.close()


def _fail_contract(session_factory, ap_id, step_id, exc, hooks) -> None:
    now = hooks.clock()
    db = session_factory()
    try:
        ap0 = db.get(ProjectAutopilot, ap_id)
        _lock_project(db, ap0.project_id)
        ap = _lock_ap(db, ap_id)
        row = db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap_id, AutopilotStep.step_id == step_id)).scalar_one_or_none()
        if row is not None:
            row.status, row.reason = "failed", str(exc)[:2000]
        _set_status(db, ap, "failed", f"contract violation in step {step_id}: {exc}",
                    kind="failed", now=now, step_id=step_id)
        db.commit()
    finally:
        db.close()


def _record_step_error(session_factory, ap_id, step_id, exc, hooks) -> None:
    now = hooks.clock()
    db = session_factory()
    try:
        ap0 = db.get(ProjectAutopilot, ap_id)
        _lock_project(db, ap0.project_id)
        ap = _lock_ap(db, ap_id)
        row = db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == ap_id, AutopilotStep.step_id == step_id)).scalar_one()
        _step_error(db, ap, row, exc, now=now)
        db.commit()
    finally:
        db.close()


def _gate_held(session_factory) -> bool:
    """The autopilot hold or the global deploy admission gate (AICP-350)."""
    from .. import ops_gate

    db = session_factory()
    try:
        g = db.get(AutopilotGate, 1)
        return bool(g and g.held) or ops_gate.read_gate(db).paused
    finally:
        db.close()


def tick_autopilot(ap_id: str, *, session_factory=None, hooks: Optional[Hooks] = None) -> dict:
    """One tick of one autopilot. Never raises (errors are journaled/logged)."""
    if session_factory is None:
        from ..db import SessionLocal as session_factory  # noqa: N813
    hooks = hooks or Hooks()
    report: dict = {"launched": [], "applied": 0}
    token = uuid.uuid4().hex
    if not _take_lease(session_factory, ap_id, token, hooks.clock()):
        report["skipped"] = "leased"
        return report
    try:
        requeue = _reconcile(session_factory, ap_id, hooks)
        for eid, job_id, queue in requeue:
            try:
                hooks.enqueue(eid, job_id, queue=queue)
            except Exception:
                log.exception("autopilot: re-enqueue of compute %s failed", eid)

        db = session_factory()
        try:
            ap = db.get(ProjectAutopilot, ap_id)
            if ap is None or ap.status != "active":
                report["status"] = ap.status if ap else "gone"
                return report
            flow = registry.get_flow(ap.flow_id, ap.flow_version)
            project = db.get(VideoProject, ap.project_id)
            if flow is None or project is None:
                return report
            specs = list(flow.steps(project, dict(ap.params or {})))
            rows = {r.step_id: r for r in db.execute(select(AutopilotStep).where(
                AutopilotStep.autopilot_id == ap.id)).scalars()}
            now = hooks.clock()
            changed = False
            spec_ids = {s.id for s in specs}
            for sid, r in rows.items():
                if sid not in spec_ids and r.status in ("pending", "active"):
                    r.status, r.updated_at = "skipped", now
                    for a in db.execute(select(AutopilotApproval).where(
                            AutopilotApproval.autopilot_id == ap.id,
                            AutopilotApproval.step_id == sid,
                            AutopilotApproval.status.in_(("pending", "approved", "rejected")))).scalars():
                        a.status = "superseded"
                    journal(db, ap, "decision", f"step {sid} skipped (no longer in the flow)",
                            step_id=sid, now=now)
                    changed = True
            for s in specs:
                r = rows.get(s.id)
                if r is not None and r.status == "skipped":
                    # the params brought the step back: it runs again, in a new generation
                    r.status, r.generation, r.updated_at = "pending", r.generation + 1, now
                    changed = True
                ready = all(rows.get(n) is not None and rows[n].status == "completed"
                            for n in s.needs)
                if r is None and ready:
                    r = AutopilotStep(autopilot_id=ap.id, step_id=s.id, generation=1,
                                      status="active", started_at=now, updated_at=now,
                                      refusals=0, errors=0, reopen_count=0)
                    db.add(r)
                    rows[s.id] = r
                    journal(db, ap, "step_started", f"step {s.id} started: {s.title}",
                            step_id=s.id, now=now)
                    changed = True
                elif r is not None and r.status == "pending" and ready:
                    r.status, r.started_at, r.updated_at = "active", now, now
                    changed = True
            if changed:
                _lock_project(db, ap.project_id)
                locked = _lock_ap(db, ap.id)
                _touch(locked, now)
                db.commit()
            ready_specs = [s for s in specs if rows.get(s.id) is not None
                           and rows[s.id].status == "active"]
            done = all(rows.get(s.id) is not None and rows[s.id].status == "completed"
                       for s in specs)
            next_checks = {s.id: rows[s.id].next_check_at for s in ready_specs}
        finally:
            db.close()

        hold = _gate_held(session_factory)
        for s in ready_specs:
            if hold:
                break
            nc = next_checks.get(s.id)
            if nc is not None and nc > hooks.clock():
                continue
            try:
                plan = _plan(session_factory, ap_id, s, specs, flow, now=hooks.clock(), hold=hold)
                res = _apply(session_factory, ap_id, s, specs, flow, plan, hooks)
            except ContractViolation as exc:
                _fail_contract(session_factory, ap_id, s.id, exc, hooks)
                break
            except Exception as exc:  # advance() or an apply bug: backoff on the step
                log.exception("autopilot %s: step %s raised", ap_id, s.id)
                _record_step_error(session_factory, ap_id, s.id, exc, hooks)
                continue
            report["applied"] += 1
            if res.outcome == "launched":
                report["launched"].append(res.effect_key)
                break

        _send_judgements(session_factory, ap_id, hooks)

        if done:
            db = session_factory()
            try:
                ap0 = db.get(ProjectAutopilot, ap_id)
                _lock_project(db, ap0.project_id)
                ap = _lock_ap(db, ap_id)
                n = db.execute(select(func.count()).select_from(AutopilotEffect).where(
                    AutopilotEffect.autopilot_id == ap.id,
                    AutopilotEffect.state.notin_(EFFECT_TERMINAL_STATES))).scalar_one()
                # recheck with the params as they are NOW, under the lock
                proj_now = db.get(VideoProject, ap.project_id)
                specs_now = list(flow.steps(proj_now, dict(ap.params or {})))
                rows_now = {r.step_id: r.status for r in db.execute(select(AutopilotStep).where(
                    AutopilotStep.autopilot_id == ap.id)).scalars()}
                all_done = all(rows_now.get(sp.id) == "completed" for sp in specs_now)
                if all_done:
                    # every completion must still hold, under the locks of what it relied on
                    done_rows = db.execute(select(AutopilotStep).where(
                        AutopilotStep.autopilot_id == ap.id,
                        AutopilotStep.status == "completed")).scalars().all()
                    _lock_runs(db, {rid for r in done_rows
                                    for rid in _run_ids_of_keys((r.depends_on or {}).keys())})
                    for r in done_rows:
                        stale = [k for k, v in (r.depends_on or {}).items()
                                 if key_version(db, ap, k) != v]
                        if stale:
                            _reopen(db, ap, specs_now, r.step_id, f"{stale[0]} changed",
                                    now=hooks.clock())
                            all_done = False
                            break
                if ap.status == "active" and n == 0 and all_done:
                    _set_status(db, ap, "completed", "all steps completed", kind="completed",
                                now=hooks.clock())
                db.commit()  # completion, or the reopen found above
            finally:
                db.close()
    except Exception:
        log.exception("autopilot %s: tick failed", ap_id)
    finally:
        _release_lease(session_factory, ap_id, token)
    return report


def due_autopilot_ids(session_factory) -> list[str]:
    db = session_factory()
    try:
        return [i for (i,) in db.execute(select(ProjectAutopilot.id).where(or_(
            ProjectAutopilot.status.in_(LIVE_AUTOPILOT_STATUSES),
            ProjectAutopilot.outstanding > 0)).order_by(ProjectAutopilot.updated_at))]
    finally:
        db.close()


def run_pass(*, session_factory=None, hooks: Optional[Hooks] = None) -> int:
    """One pass over every autopilot that needs ticking. Returns how many ticked."""
    if not settings.AUTOPILOT_ENABLED:
        return 0
    if session_factory is None:
        from ..db import SessionLocal as session_factory  # noqa: N813
    n = 0
    for ap_id in due_autopilot_ids(session_factory):
        tick_autopilot(ap_id, session_factory=session_factory, hooks=hooks)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Operator actions (API / card)
# ---------------------------------------------------------------------------


class OperatorError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail
        super().__init__(detail)


def _live_or_latest(db: Session, project_id: str) -> Optional[ProjectAutopilot]:
    from .links import latest_attempt  # one rule for the cards, links and inbox (AICP-373)

    return latest_attempt(db, project_id)


def _operator_ap(db: Session, project_id: str,
                 expected_id: Optional[str] = None) -> ProjectAutopilot:
    # Lock order: project, then autopilot (as everywhere in this module). The
    # project lock serializes with start_new, which creates a successor under
    # it, so the attempt read below stays the latest until this commits.
    _lock_project(db, project_id)
    ap = _live_or_latest(db, project_id)
    if ap is None:
        db.rollback()
        raise OperatorError(404, "this project has no autopilot")
    locked = _lock_ap(db, ap.id)
    # AICP-369: a page that showed one autopilot attempt (a Run page, a stale
    # project card) must never act on a newer attempt of the same project.
    if expected_id and locked.id != expected_id:
        db.rollback()
        raise OperatorError(409, "this autopilot attempt is no longer the project's current one; "
                                 "reload the page")
    return locked


def set_switch(db: Session, project_id: str, action: str, *, reason: str = "",
               by: str = "operator", expected_id: Optional[str] = None) -> ProjectAutopilot:
    now = utcnow()
    ap = _operator_ap(db, project_id, expected_id)
    if action == "pause":
        if ap.status != "active":
            raise OperatorError(409, f"the autopilot is {ap.status}, not active")
        _set_status(db, ap, "paused", f"paused by {by}" + (f": {reason}" if reason else ""),
                    kind="paused", now=now)
    elif action == "resume":
        if ap.status != "paused":
            raise OperatorError(409, f"the autopilot is {ap.status}, not paused")
        _set_status(db, ap, "active", f"resumed by {by}", kind="resumed", now=now)
        ap.status_reason = None
        for r in db.execute(select(AutopilotStep).where(AutopilotStep.autopilot_id == ap.id,
                                                        AutopilotStep.status == "active")).scalars():
            r.errors, r.refusals, r.next_check_at = 0, 0, None
    elif action == "stop":
        if ap.status not in LIVE_AUTOPILOT_STATUSES:
            raise OperatorError(409, f"the autopilot is already {ap.status}")
        _set_status(db, ap, "stopped", f"stopped by {by}" + (f": {reason}" if reason else ""),
                    kind="stopped", now=now)
    else:
        raise OperatorError(400, f"unknown action {action!r}")
    db.commit()
    wake()
    return ap


def answer(db: Session, project_id: str, approval_id: str, decision: str, *,
           option: Optional[str] = None, comment: Optional[str] = None,
           by: str = "operator", expected_id: Optional[str] = None) -> AutopilotApproval:
    if decision not in ("approve", "reject"):
        raise OperatorError(400, "decision must be approve or reject")
    ap = _operator_ap(db, project_id, expected_id)
    a = db.execute(select(AutopilotApproval).where(AutopilotApproval.id == approval_id)
                   .with_for_update()).scalar_one_or_none()
    if a is None or a.autopilot_id != ap.id:
        raise OperatorError(404, "approval not found on this project's autopilot")
    if a.answered_decision is not None:
        if a.answered_decision == decision and (a.chosen_option or None) == (option or None):
            db.rollback()
            return a  # idempotent repeat
        raise OperatorError(409, f"already answered: {a.answered_decision}")
    if a.status != "pending":
        raise OperatorError(409, f"the approval is {a.status}")
    if decision == "approve" and a.options:
        if not option or option not in {o["id"] for o in a.options}:
            raise OperatorError(422, "pick one of the options")
    now = utcnow()
    a.status = "approved" if decision == "approve" else "rejected"
    a.answered_decision, a.chosen_option = decision, (option if decision == "approve" else None)
    a.answer_comment = (comment or "").strip()[:4000] or None
    a.answered_at, a.answered_by = now, by[:128]
    _touch(ap, now)
    journal(db, ap, "approval_answered", f"{a.kind} {a.name}: {decision}"
            + (f" ({option})" if option else "") + (f" — {a.answer_comment}" if a.answer_comment else ""),
            step_id=a.step_id, approval_id=a.id, now=now)
    db.commit()
    wake()
    return a


def _reopenable(db: Session, a: AutopilotApproval) -> bool:
    """An answered flow question whose chosen option launched nothing (a
    choice such as "ship it as it is" / "I fix it by hand"), in the CURRENT
    generation of its active step, and the latest generation of its name.
    Core approvals (budget / regeneration / overrun) and answers that
    authorised a launch never qualify (2026-11-02, "Ask again")."""
    if CORE_SEP in (a.name or "") or a.answered_decision != "approve" \
            or a.status not in ("approved", "consumed") or not a.chosen_option:
        return False
    opt = next((o for o in (a.options or []) if o.get("id") == a.chosen_option), None)
    if opt is None or opt.get("action"):
        return False
    row = db.execute(select(AutopilotStep).where(
        AutopilotStep.autopilot_id == a.autopilot_id,
        AutopilotStep.step_id == a.step_id)).scalar_one_or_none()
    if row is None or row.status != "active" or row.generation != a.step_generation:
        return False
    latest = _latest_approval(db, a.autopilot_id, a.step_id, a.step_generation, a.name)
    return latest is not None and latest.id == a.id


def reopenable(db: Session, ap_id: str) -> list[AutopilotApproval]:
    """The answers the operator may take back on the card ("Ask again")."""
    rows = db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == ap_id,
        AutopilotApproval.status.in_(("approved", "consumed")),
        AutopilotApproval.answered_decision == "approve")
        .order_by(AutopilotApproval.answered_at)).scalars().all()
    return [a for a in rows if _reopenable(db, a)]


def reopen(db: Session, project_id: str, approval_id: str, *, by: str = "operator",
           expected_id: Optional[str] = None) -> AutopilotApproval:
    """"Ask again": take back an answered choice so the flow asks the question
    afresh (the same name, a new generation) with its current proposal. The
    answer stays in the record (superseded, journalled); the plan in flight is
    discarded (the autopilot version moves). Only :func:`_reopenable` answers:
    a choice that launched nothing, in the step that is still running."""
    ap = _operator_ap(db, project_id, expected_id)
    a = db.execute(select(AutopilotApproval).where(AutopilotApproval.id == approval_id)
                   .with_for_update()).scalar_one_or_none()
    if a is None or a.autopilot_id != ap.id:
        raise OperatorError(404, "approval not found on this project's autopilot")
    if a.status == "superseded":
        db.rollback()
        return a  # idempotent repeat
    if not _reopenable(db, a):
        raise OperatorError(409, "only an answered choice that started nothing, in the step that "
                                 "is still running, can be asked again")
    now = utcnow()
    _supersede(db, ap, a, f"asked again by {by[:64]} (was: {a.chosen_option})", now)
    _touch(ap, now)
    db.commit()
    wake()
    return a


def comment(db: Session, project_id: str, approval_id: str, text: str, *,
            by: str = "operator") -> AutopilotApproval:
    text = (text or "").strip()
    if not text:
        raise OperatorError(400, "empty comment")
    ap = _operator_ap(db, project_id)
    a = db.execute(select(AutopilotApproval).where(AutopilotApproval.id == approval_id)
                   .with_for_update()).scalar_one_or_none()
    if a is None or a.autopilot_id != ap.id:
        raise OperatorError(404, "approval not found on this project's autopilot")
    now = utcnow()
    a.comments = list(a.comments or []) + [{"at": now.isoformat(), "by": by, "text": text[:4000]}]
    _touch(ap, now)
    journal(db, ap, "comment", f"comment on {a.name}: {text[:500]}", step_id=a.step_id,
            approval_id=a.id, now=now)
    db.commit()
    return a


def patch(db: Session, project_id: str, *, tolerance: Optional[float] = None,
          params: Optional[dict] = None) -> ProjectAutopilot:
    ap = _operator_ap(db, project_id)
    now = utcnow()
    if tolerance is not None:
        if not (0 <= float(tolerance) <= 0.5):
            raise OperatorError(422, "tolerance must be within 0 .. 0.5")
        ap.tolerance = float(tolerance)
        journal(db, ap, "budget", f"tolerance set to {float(tolerance):.0%}", now=now)
    if params is not None:
        paid = db.execute(select(func.count()).select_from(AutopilotEffect).where(
            AutopilotEffect.autopilot_id == ap.id, AutopilotEffect.paid.is_(True))).scalar_one()
        if paid:
            raise OperatorError(409, "params can no longer change: a paid step was launched")
        flow = registry.get_flow(ap.flow_id, ap.flow_version)
        project = db.get(VideoProject, ap.project_id)
        from .contract import FlowParamsError

        try:
            ap.params = flow.validate_params(project, dict(params))
        except FlowParamsError as exc:
            raise OperatorError(422, str(exc)) from None
        journal(db, ap, "decision", "params changed by the operator", data={"params": ap.params},
                now=now)
    _touch(ap, now)
    db.commit()
    wake()
    return ap


def start_new(db: Session, project_id: str) -> ProjectAutopilot:
    project = _lock_project(db, project_id)
    if project is None:
        raise OperatorError(404, "project not found")
    if project.mode != "assistant":
        raise OperatorError(409, "a manual project has no autopilot (no switching in v1)")
    prev = _live_or_latest(db, project_id)
    if prev is not None and prev.status in LIVE_AUTOPILOT_STATUSES:
        raise OperatorError(409, f"the autopilot is still {prev.status}")
    flow = registry.flow_for_project_type(project.project_type)
    if flow is None or not settings.AUTOPILOT_ENABLED:
        raise OperatorError(422, "no autopilot flow is available for this project type")
    ap = create_autopilot(db, project, flow, dict(prev.params or {}) if prev else {})
    db.commit()
    wake()
    return ap


def wake() -> None:
    try:
        from ..tasks import _get_redis

        _get_redis().set("reskin:autopilot:wake", "1", ex=60)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# View for the API and the card
# ---------------------------------------------------------------------------


def money(mc: int) -> dict:
    from .. import spend

    return {"credits": mc / 1000.0, "usd": round(spend.usd(mc / 1000.0), 2)}


def budget_view(db: Session, ap: ProjectAutopilot) -> dict:
    """The budget block of the card (project-wide figures, AICP-369 reuses it)."""
    f = spend_view.figures_mcredits(db, ap.project_id)
    approved = _cr(ap.approved_budget_mcredits)
    return {
        "approved": money(ap.approved_budget_mcredits) if approved is not None else None,
        "committed": money(f["committed"]), "consumed": money(f["consumed"]),
        "settled": money(f["settled"]), "tolerance": ap.tolerance,
        "headroom": (money(int(ap.approved_budget_mcredits * (1 + ap.tolerance)) - f["committed"])
                     if approved is not None else None),
        "note": "manual spend is counted from the project's rows as last scanned",
    }


def _card(d: dict, flow_id: Optional[str]) -> dict:
    from .card import present

    return present(d, flow_id=flow_id)


def approval_dict(a: AutopilotApproval, project_id: str) -> dict:
    from .links import anchor, approval_title

    return {
        "id": a.id, "autopilot_id": a.autopilot_id, "name": a.name, "kind": a.kind,
        "step_id": a.step_id, "step_generation": a.step_generation, "subject": a.subject,
        "question": a.question, "title": approval_title(a.question, a.proposal),
        "anchor": anchor(a.id), "proposal": a.proposal, "options": a.options,
        "artifacts": [{"label": x["label"], "kind": x["kind"], "sha256": x["sha256"],
                       "url": f"/api/v2/projects/{project_id}/autopilot/artifacts/{x['sha256']}"}
                      for x in (a.artifacts or [])],
        "cost": money(a.cost_mcredits or 0), "comments": a.comments,
        "created_at": a.created_at.isoformat(),
    }


def journal_dict(j: AutopilotJournal) -> dict:
    return {"id": j.id, "at": j.at.isoformat(), "step_id": j.step_id, "kind": j.kind,
            "message": j.message, "cost_credits": _cr(j.cost_mcredits)}


def step_dict(s: AutopilotStep) -> dict:
    return {"step_id": s.step_id, "status": s.status, "generation": s.generation,
            "reason": s.reason, "output": s.output}


def view(db: Session, project_id: str, *, journal_before: Optional[int] = None,
         journal_limit: int = 50) -> Optional[dict]:
    ap = _live_or_latest(db, project_id)
    if ap is None:
        return None
    steps = db.execute(select(AutopilotStep).where(AutopilotStep.autopilot_id == ap.id)
                       .order_by(AutopilotStep.started_at)).scalars().all()
    approvals = db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.status == "pending")
        .order_by(AutopilotApproval.created_at)).scalars().all()
    jq = select(AutopilotJournal).where(AutopilotJournal.autopilot_id == ap.id)
    if journal_before:
        jq = jq.where(AutopilotJournal.id < journal_before)
    rows = db.execute(jq.order_by(AutopilotJournal.id.desc()).limit(journal_limit)).scalars().all()
    return {
        "id": ap.id, "project_id": project_id, "flow_id": ap.flow_id,
        "flow_version": ap.flow_version, "status": ap.status, "status_reason": ap.status_reason,
        "params": ap.params, "created_at": ap.created_at.isoformat(),
        "budget": budget_view(db, ap),
        "steps": [step_dict(s) for s in steps],
        "approvals": [_card(approval_dict(a, project_id), ap.flow_id) for a in approvals],
        "answered": [dict(approval_dict(a, project_id), chosen_option=a.chosen_option,
                          answered_by=a.answered_by,
                          answered_at=a.answered_at.isoformat() if a.answered_at else None)
                     for a in reopenable(db, ap.id)],
        "journal": [journal_dict(j) for j in rows],
    }


def artifact_path(db: Session, project_id: str, sha256: str) -> Optional[str]:
    """The immutable artifact copy, only if an approval of this project names it."""
    if not sha256 or not all(c in "0123456789abcdef" for c in sha256) or len(sha256) != 64:
        return None
    ap_ids = [i for (i,) in db.execute(select(ProjectAutopilot.id).where(
        ProjectAutopilot.project_id == project_id))]
    if not ap_ids:
        return None
    for a in db.execute(select(AutopilotApproval).where(
            AutopilotApproval.autopilot_id.in_(ap_ids),
            AutopilotApproval.artifacts.isnot(None))).scalars():
        for art in a.artifacts or []:
            if art.get("sha256") == sha256:
                store = _artifact_store(project_id)
                for name in os.listdir(store) if os.path.isdir(store) else []:
                    if name.startswith(sha256 + "."):
                        return os.path.join(store, name)
    return None
