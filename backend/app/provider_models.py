"""ORM models of the second-provider layer (Fal, AICP-312 Stage 1 / AICP-330).

Two tables, see docs/fal-provider-design.md §4:

* ``provider_tasks`` — one row per provider request (one POST at most). Owned
  by a consumer through a SOFT reference (``owner_kind``/``owner_id``, no FK),
  so a consumer's hard delete can never remove the record of a possibly billed
  call. Every status change goes through ``pipeline_provider._cas`` (a guarded
  Core UPDATE that also bumps ``version_id``); the ORM mapper's
  ``version_id_col`` makes any ORM write of a stale copy fail instead of
  silently overwriting a committed status.
* ``provider_task_actions`` — the insert-only audit of operator actions
  (``adopt``, ``abandon``, ``close``, ``resume``), written in the same
  transaction as the action's CAS. ORM listeners below and triggers in the
  migration reject UPDATE and DELETE on both dialects.

Times are naive UTC (as every other table here). Imported at the end of
``models.py`` so ``Base.metadata`` and Alembic see the tables.
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
    event,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .state_machine import PROVIDER_TASK_ACTIVE_WHERE, PROVIDER_TASK_STATUS_CHECK


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class ProviderTask(Base):
    __tablename__ = "provider_tasks"
    __table_args__ = (
        CheckConstraint(PROVIDER_TASK_STATUS_CHECK, name="ck_provider_tasks_status"),
        UniqueConstraint("provider", "operation_key", name="uq_provider_tasks_operation"),
        UniqueConstraint("provider", "request_id", name="uq_provider_tasks_request"),
        # A destination belongs to exactly one task, ever: an existing file at
        # it is therefore THIS task's verified output (fetch crash recovery).
        UniqueConstraint("destination_path", name="uq_provider_tasks_destination"),
        # One ACTIVE task per owner slot, across providers (design §4, rule
        # 4.1). The WHERE text is state_machine.PROVIDER_TASK_ACTIVE_WHERE,
        # byte-identical in the migration.
        Index(
            "uq_provider_tasks_active_per_owner_slot",
            "owner_kind", "owner_id", "owner_slot",
            unique=True,
            postgresql_where=text(PROVIDER_TASK_ACTIVE_WHERE),
            sqlite_where=text(PROVIDER_TASK_ACTIVE_WHERE),
        ),
        Index("ix_provider_tasks_status", "status"),
        Index("ix_provider_tasks_owner_project", "owner_project_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    # Spend kind (``fal_<family>``), also the reservation's kind.
    kind: Mapped[str] = mapped_column(String(64), nullable=False)

    owner_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_slot: Mapped[str] = mapped_column(
        String(64), nullable=False, default="main", server_default=text("'main'")
    )
    owner_project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    # Set for a run-scoped owner: admission locked that run row, and
    # delete_run refuses while an active task names it (AICP-330 M2).
    owner_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    # realpath under DATA_DIR, frozen at admission; the ONLY place a result is
    # ever written (via a claim-scoped staging file beside it).
    destination_path: Mapped[str] = mapped_column(Text, nullable=False)
    operation_key: Mapped[str] = mapped_column(String(255), nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    claim_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    submit_started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    status_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_requested_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    # Durable cancel delivery (AICP-330 M3): set only when Fal answered the PUT
    # with 2xx or "already completed"; anything else is retried.
    cancel_delivered_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    cancel_attempted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    reservation_held: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    draw_pending: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    # The credits reserved at admission (estimated_usd / KIE_USD_PER_CREDIT at
    # that moment). Every draw for the task uses THIS amount, never a
    # recomputation, so a later KIE_USD_PER_CREDIT change cannot split the books.
    reserved_mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fetch_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    # NULL until the FIRST fal_fetch attempt actually starts (AICP-330 B1): a
    # result waiting in a backed-up queue does not age.
    fetch_deadline_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    # When a poll first saw COMPLETED (entry to `fetching`): the watchdog
    # alerts while the result is still unread half-way to its retention.
    completed_seen_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    # Consecutive transient poll failures (backoff).
    poll_failures: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )

    request_spec: Mapped[dict] = mapped_column(JSON, nullable=False)
    input_refs: Mapped[list] = mapped_column(JSON, nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    output_sec: Mapped[float] = mapped_column(Float, nullable=False)
    resolution: Mapped[str] = mapped_column(String(16), nullable=False)
    estimated_usd: Mapped[float] = mapped_column(Float, nullable=False)

    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Whitelisted, URL-free fields of Fal's result body (placement, seed,
    # frame_count, fps, video size/type) for the consumer's QA (AICP-251 Q1).
    # Output media URLs are never stored.
    result_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())
    version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __mapper_args__ = {"version_id_col": version_id, "version_id_generator": False}

    def __repr__(self) -> str:
        return f"<ProviderTask id={self.id!r} {self.provider}:{self.status}>"


class ProviderTaskAction(Base):
    __tablename__ = "provider_task_actions"
    __table_args__ = (
        CheckConstraint(
            "action IN ('adopt', 'abandon', 'close', 'resume')",
            name="ck_provider_task_actions_action",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("provider_tasks.id", ondelete="RESTRICT"),
        nullable=False, index=True,
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())
    from_status: Mapped[str] = mapped_column(String(32), nullable=False)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class ProviderTaskActionImmutable(Exception):
    pass


def _reject_update(mapper, connection, target):  # noqa: ARG001
    raise ProviderTaskActionImmutable("provider_task_actions is insert-only")


def _reject_delete(mapper, connection, target):  # noqa: ARG001
    raise ProviderTaskActionImmutable("provider_task_actions is insert-only")


event.listen(ProviderTaskAction, "before_update", _reject_update)
event.listen(ProviderTaskAction, "before_delete", _reject_delete)
