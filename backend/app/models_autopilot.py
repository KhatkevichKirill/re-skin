"""ORM models of the "by Assistant" autopilot (AICP-363, docs/autopilot-design.md §2).

* ``project_autopilots`` — one record per autopilot attempt on a project.
* ``autopilot_steps`` — per-step state (generation, output, depends_on).
* ``autopilot_effects`` — every action / compute job / judgement a step causes;
  its UNIQUE ``idempotency_key`` is the double-launch guard (§5.4).
* ``autopilot_approvals`` — operator decisions (budget, regeneration, …).
* ``project_spend_scopes`` — project-owned, append-only spend attribution (§6.1).
* ``autopilot_journal`` — append-only "what and why".
* ``autopilot_gate`` — singleton deploy hold (§9).
* ``project_manual_spend_events`` — manual paid admissions on assistant
  projects, which pause the autopilot (§5.7).

Status columns are VARCHAR + CHECK (never a Postgres enum). Times are naive
UTC. Imported at the end of ``models.py`` so ``Base.metadata`` and Alembic see
the tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_uuid() -> str:
    return str(uuid.uuid4())


AUTOPILOT_STATUSES = ("active", "paused", "completed", "stopped", "failed")
LIVE_AUTOPILOT_STATUSES = ("active", "paused")
STEP_STATUSES = ("pending", "active", "completed", "failed", "skipped")
EFFECT_KINDS = ("action", "compute", "judgement")
EFFECT_STATES = (
    # action
    "launching", "launched", "not_launched", "unknown",
    # compute
    "queued", "running", "succeeded", "failed",
    # judgement (plus succeeded / failed / unknown)
    "prepared", "sending", "submitted",
)
EFFECT_TERMINAL_STATES = ("launched", "not_launched", "unknown", "succeeded", "failed")
APPROVAL_KINDS = (
    "budget", "regeneration", "overrun", "listening_probe", "listening_final", "handoff",
)
APPROVAL_STATUSES = ("pending", "approved", "rejected", "superseded", "consumed")

_LIVE_WHERE = "status IN ('active', 'paused')"


def _in(col: str, values) -> str:
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class ProjectAutopilot(Base):
    __tablename__ = "project_autopilots"
    __table_args__ = (
        CheckConstraint(_in("status", AUTOPILOT_STATUSES), name="ck_project_autopilots_status"),
        CheckConstraint("tolerance >= 0 AND tolerance <= 0.5", name="ck_project_autopilots_tolerance"),
        Index(
            "uq_project_autopilots_live", "project_id", unique=True,
            postgresql_where=text(_LIVE_WHERE), sqlite_where=text(_LIVE_WHERE),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    flow_id: Mapped[str] = mapped_column(String(64), nullable=False)
    flow_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    status_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    approved_budget_mcredits: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    tolerance: Mapped[float] = mapped_column(Float, nullable=False, default=0.10)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    outstanding: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_check_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class AutopilotStep(Base):
    __tablename__ = "autopilot_steps"
    __table_args__ = (
        UniqueConstraint("autopilot_id", "step_id", name="uq_autopilot_steps_step"),
        CheckConstraint(_in("status", STEP_STATUSES), name="ck_autopilot_steps_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    autopilot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    step_id: Mapped[str] = mapped_column(String(96), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    depends_on: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    previous_outputs: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    next_check_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    refusals: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_refusal: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    reopen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)


class AutopilotEffect(Base):
    __tablename__ = "autopilot_effects"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_autopilot_effects_key"),
        CheckConstraint(_in("kind", EFFECT_KINDS), name="ck_autopilot_effects_kind"),
        CheckConstraint(_in("state", EFFECT_STATES), name="ck_autopilot_effects_state"),
        Index("ix_autopilot_effects_autopilot_state", "autopilot_id", "state"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    autopilot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False,
    )
    step_id: Mapped[str] = mapped_column(String(96), nullable=False)
    step_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    name: Mapped[str] = mapped_column(String(96), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    handler: Mapped[str] = mapped_column(String(96), nullable=False)
    handler_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    params_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    paid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    estimate_mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    targets: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    target_versions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_refs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    approval_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    claim_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # compute: re-enqueues of a job lost BEFORE any worker claimed it
    requeues: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rq_job_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    not_live_since: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)


class AutopilotApproval(Base):
    __tablename__ = "autopilot_approvals"
    __table_args__ = (
        UniqueConstraint("dedupe_key", name="uq_autopilot_approvals_dedupe"),
        CheckConstraint(_in("kind", APPROVAL_KINDS), name="ck_autopilot_approvals_kind"),
        CheckConstraint(_in("status", APPROVAL_STATUSES), name="ck_autopilot_approvals_status"),
        Index("ix_autopilot_approvals_autopilot_status", "autopilot_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    autopilot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False,
    )
    step_id: Mapped[str] = mapped_column(String(96), nullable=False)
    step_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    dedupe_key: Mapped[str] = mapped_column(String(400), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    subject: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    question: Mapped[str] = mapped_column(Text, nullable=False, default="")
    proposal: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    options: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    chosen_option: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    artifacts: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # {handler, handler_version, params, params_sha256} for action-bound approvals
    action: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    action_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    bound_versions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    cost_mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    max_cost_mcredits: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    committed_at_ask_mcredits: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    budget_after_mcredits: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    comments: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    answer_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    answered_decision: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    answered_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    answered_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    consumed_by_effect_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    consumed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)


class ProjectSpendScope(Base):
    """Project-owned, append-only spend attribution (no FK: it is history)."""

    __tablename__ = "project_spend_scopes"

    project_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    registered_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    registered_by_autopilot_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="scan")


class AutopilotJournal(Base):
    __tablename__ = "autopilot_journal"
    __table_args__ = (Index("ix_autopilot_journal_autopilot_id", "autopilot_id", "id"),)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True,
    )
    autopilot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False,
    )
    at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    step_id: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    cost_mcredits: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    effect_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    approval_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)


class AutopilotGate(Base):
    """Singleton (id = 1), created by the migration (and lazily in tests)."""

    __tablename__ = "autopilot_gate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    held: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    held_since: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    held_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ProjectManualSpendEvent(Base):
    __tablename__ = "project_manual_spend_events"
    __table_args__ = (Index("ix_project_manual_spend_events_project", "project_id", "seen_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    scope_key: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    seen_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


NOTIFICATION_KINDS = ("approval", "pause", "end", "marker")


class AutopilotNotification(Base):
    """One chat message the notifier keeps in step with an approval or an
    attempt (AICP-374, docs/autopilot-operator-ux.md §2). ``key`` is
    ``approval:<id>``, ``pause:<journal id>``, ``end:<autopilot id>`` or the
    singleton ``notifier:since`` marker. No FK: the message outlives the rows it
    describes (a deleted project still gets its message closed)."""

    __tablename__ = "autopilot_notifications"
    __table_args__ = (
        UniqueConstraint("key", name="uq_autopilot_notifications_key"),
        CheckConstraint(_in("kind", NOTIFICATION_KINDS), name="ck_autopilot_notifications_kind"),
        Index("ix_autopilot_notifications_open", "closed_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    project_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    autopilot_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    approval_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    journal_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    posted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    state: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    state_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    gone: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    last_reminder_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    claim_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    next_try_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
