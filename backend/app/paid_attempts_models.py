"""Paid submit attempts (AICP-351 = AICP-344 stage 2,
docs/assistant-drain-check-design.md §5).

* ``paid_submit_attempts`` — ONE row per attempt at a paid provider call,
  committed BEFORE the call (the receipt belongs to the attempt, never to the
  owner row: during an in-poll resubmit ``RunSegment.seedance_task_id`` still
  holds the PREVIOUS task until the new one commits, so "the owner has a task
  id" proves nothing). ``status`` is a varchar (no Postgres enum, no
  ``ALTER TYPE``):

  - video / image (§5.1): ``started`` -> ``accepted`` (the provider task id is
    on the row) | ``failed_unbilled`` (a refusal that proves nothing was
    billed) | ``parked`` (the outcome is unknown: never resubmitted blind) ->
    ``resubmitted`` | ``abandoned`` (the operator's audited resolution).
  - LLM-only jobs (§5.2): ``started`` -> ``accepted`` | ``failed`` (a definite
    failure; may have billed) | ``unknown`` (an unknown outcome whose sender is
    stopped: its own worker saw it, or the sender is PROVEN gone) |
    ``superseded`` (the lease ran out with no proof: a newer attempt may run,
    and this one is STILL counted active) -> ``stopped`` (acknowledged by its
    own worker, proven gone, or host-attested).

* ``paid_submit_attempt_actions`` — the insert-only audit of operator actions
  (``resubmit``, ``abandon``, ``release_sender``), written in the transaction of
  the action's CAS (triggers in the migration reject UPDATE/DELETE).

Imported at the end of ``models.py`` so ``Base.metadata`` and Alembic see the
tables. The migration ``x3b5d7f9a1c4`` is a fixed copy of this schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

# -- statuses ---------------------------------------------------------------
STARTED = "started"
ACCEPTED = "accepted"
FAILED_UNBILLED = "failed_unbilled"
PARKED = "parked"
RESUBMITTED = "resubmitted"
ABANDONED = "abandoned"
FAILED = "failed"
UNKNOWN = "unknown"
SUPERSEDED = "superseded"
STOPPED = "stopped"

STATUSES = (STARTED, ACCEPTED, FAILED_UNBILLED, PARKED, RESUBMITTED, ABANDONED,
            FAILED, UNKNOWN, SUPERSEDED, STOPPED)
#: At most one of these per owner (partial unique index).
OPEN_STATUSES = (STARTED, PARKED)
#: Counted ``active`` by the drain (a sender may still be live).
ACTIVE_STATUSES = (STARTED, SUPERSEDED)
#: Counted ``parked`` by the drain (waits for the operator).
PARKED_STATUSES = (PARKED,)

# -- owners -----------------------------------------------------------------
OWNER_RUN_SEGMENT = "run_segment"
OWNER_RUN_PREPROCESS = "run_preprocess"
OWNER_SEGMENT_IMAGE_JOB = "segment_image_job"
OWNER_LLM_TRANSLATE = "llm_translate"
OWNER_LLM_TRANSCRIBE = "llm_transcribe"
OWNER_LLM_CAPTION_ASR = "llm_caption_asr"
VIDEO_IMAGE_OWNERS = (OWNER_RUN_SEGMENT, OWNER_RUN_PREPROCESS, OWNER_SEGMENT_IMAGE_JOB)
LLM_OWNERS = (OWNER_LLM_TRANSLATE, OWNER_LLM_TRANSCRIBE, OWNER_LLM_CAPTION_ASR)
OWNER_KINDS = VIDEO_IMAGE_OWNERS + LLM_OWNERS

KINDS = ("video", "image", "llm")
ACTIONS = ("resubmit", "abandon", "release_sender")


def _in(col: str, values) -> str:
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


OPEN_WHERE = _in("status", OPEN_STATUSES)
STATUS_CHECK = _in("status", sorted(STATUSES))
OWNER_CHECK = _in("owner_kind", sorted(OWNER_KINDS))
KIND_CHECK = _in("kind", KINDS)
ACTION_CHECK = _in("action", ACTIONS)


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Naive UTC, like every other timestamp column here."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PaidSubmitAttempt(Base):
    __tablename__ = "paid_submit_attempts"
    __table_args__ = (
        CheckConstraint(STATUS_CHECK, name="ck_paid_submit_attempts_status"),
        CheckConstraint(OWNER_CHECK, name="ck_paid_submit_attempts_owner_kind"),
        CheckConstraint(KIND_CHECK, name="ck_paid_submit_attempts_kind"),
        UniqueConstraint("owner_kind", "owner_id", "attempt_no",
                         name="uq_paid_submit_attempts_owner_attempt"),
        Index("uq_paid_submit_attempts_open", "owner_kind", "owner_id", unique=True,
              postgresql_where=text(OPEN_WHERE), sqlite_where=text(OPEN_WHERE)),
        Index("ix_paid_submit_attempts_run", "run_id"),
        Index("ix_paid_submit_attempts_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(80), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    # The run the owner belongs to (run page lookup). No FK: the audit of a
    # billed call must survive a run delete.
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)
    provider: Mapped[str] = mapped_column(String(8), nullable=False, default="kie")
    request_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # The holder's fence when the attempt began: the per-run Redis lock token
    # (run phases) or the job's claim token (LLM jobs).
    lease_generation: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=STARTED)
    provider_task_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    predecessor_task_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    # LLM attempts (§5.2): calls claimed so far (one send per call number), the
    # bound, the lease (RQ job_timeout + 300 s) and the sender identity the
    # reaper's proof reads (rebuild_paid_call.sender_proof).
    calls_sent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_calls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    sender_host: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sender_boot: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sender_job: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    # Set when the sender can no longer send for this attempt: its own write
    # after the POST returned, sender proof (rebuild_paid_call.sender_proof) or
    # host attestation. Operator resolution of a parked attempt needs it.
    sender_released_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    sender_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PaidSubmitAttemptAction(Base):
    __tablename__ = "paid_submit_attempt_actions"
    __table_args__ = (
        CheckConstraint(ACTION_CHECK, name="ck_paid_submit_attempt_actions_action"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("paid_submit_attempts.id", ondelete="RESTRICT"),
        nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)
    from_status: Mapped[str] = mapped_column(String(16), nullable=False)
    to_status: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
