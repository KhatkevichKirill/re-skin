"""Durable SQL-only assistant execution lifecycle (AICP-266).

All functions operate in the CALLER'S transaction and never commit, enqueue,
contact providers, or render. Authorization, capability admission, pinned donor
validation and resource policy must run before accepting a plan. Callers must
commit a paid submission permit BEFORE any future network call. A lost receipt
is UNKNOWN, never permission to submit again. PostgreSQL race/recovery tests,
outbox dispatch, provider reconciliation and runtime integration remain separate
work; SQLite tests establish logical invariants only.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, and_, event, inspect, or_, select, update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column

from .assistant_contracts import EditPlan
from .db import Base

_IDENTIFIER = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL = ("succeeded", "failed", "cancelled")
_NEXT_PHASE = {"resolve": "render", "render": "inspect", "inspect": "publish"}


class JobConflict(ValueError):
    """An idempotency key, lease, state transition or request is inconsistent."""


class LostLease(JobConflict):
    """The work is expired, cancelled, terminal or fenced by newer work."""


class BlindRetryProhibited(JobConflict):
    """A provider call may already exist; reconcile instead of submitting."""


class AssistantJob(Base):
    __tablename__ = "assistant_jobs"
    __table_args__ = (
        UniqueConstraint("request_scope", "idempotency_key", name="uq_assistant_job_request"),
        UniqueConstraint("pending_version_id", name="uq_assistant_job_pending_version"),
        CheckConstraint("status IN ('queued','running','cancel_requested','succeeded','failed','cancelled')", name="ck_assistant_job_status"),
        CheckConstraint("phase IN ('resolve','render','inspect','publish')", name="ck_assistant_job_phase"),
        CheckConstraint("lease_generation >= 0 AND claim_count >= 0 AND max_claims BETWEEN 1 AND 10", name="ck_assistant_job_counters"),
        CheckConstraint("((status IN ('running','cancel_requested')) AND lease_owner IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL) OR ((status NOT IN ('running','cancel_requested')) AND lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)", name="ck_assistant_job_lease"),
        Index("ix_assistant_jobs_recovery", "status", "lease_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_scope: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    pinned_base_version_id: Mapped[str] = mapped_column(ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False)
    pending_version_id: Mapped[str] = mapped_column(ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False)
    accepted_plan: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    phase: Mapped[str] = mapped_column(String(24), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token: Mapped[str | None] = mapped_column(String(36))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_claims: Mapped[int] = mapped_column(Integer, nullable=False)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AssistantJobOutbox(Base):
    __tablename__ = "assistant_job_outbox"
    __table_args__ = (
        UniqueConstraint("job_id", "event_key", name="uq_assistant_job_outbox_event"),
        Index("ix_assistant_job_outbox_pending", "dispatched_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("assistant_jobs.id", ondelete="RESTRICT"), nullable=False)
    event_key: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AssistantPaidCallIntent(Base):
    __tablename__ = "assistant_paid_call_intents"
    __table_args__ = (
        UniqueConstraint("job_id", "logical_key", name="uq_assistant_paid_call_logical"),
        UniqueConstraint("job_id", "provider", "request_sha256", name="uq_assistant_paid_call_fingerprint"),
        UniqueConstraint("provider", "provider_task_id", name="uq_assistant_paid_call_receipt"),
        UniqueConstraint("provider_request_key", name="uq_assistant_paid_call_request_key"),
        CheckConstraint("status IN ('prepared','submitting','submitted','unknown','completed','failed')", name="ck_assistant_paid_call_status"),
        CheckConstraint("(status = 'prepared' AND submission_token IS NULL) OR (status <> 'prepared' AND submission_token IS NOT NULL)", name="ck_assistant_paid_call_permit"),
        Index("ix_assistant_paid_call_ambiguity", "status", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("assistant_jobs.id", ondelete="RESTRICT"), nullable=False)
    logical_key: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_request_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    submission_token: Mapped[str | None] = mapped_column(String(36))
    provider_task_id: Mapped[str | None] = mapped_column(String(255))
    ambiguity_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


@dataclass(frozen=True)
class JobLease:
    job_id: str
    owner: str
    token: str
    generation: int
    phase: str
    expires_at: datetime


@dataclass(frozen=True)
class SubmissionPermit:
    intent_id: str
    token: str
    provider_request_key: str


_IMMUTABLE_FIELDS = {
    AssistantJob: {"id", "request_scope", "idempotency_key", "request_sha256", "actor",
                   "pinned_base_version_id", "pending_version_id", "accepted_plan", "max_claims", "created_at"},
    AssistantJobOutbox: {"id", "job_id", "event_key", "kind", "payload", "created_at"},
    AssistantPaidCallIntent: {"id", "job_id", "logical_key", "provider", "request_sha256", "provider_request_key", "created_at"},
}


def _preserve_accepted_fields(mapper, connection, target):
    for field in _IMMUTABLE_FIELDS[type(target)]:
        if inspect(target).attrs[field].history.has_changes():
            raise JobConflict(f"immutable assistant job field: {field}")


for _model in _IMMUTABLE_FIELDS:
    event.listen(_model, "before_update", _preserve_accepted_fields)


def _id() -> str:
    return str(uuid.uuid4())


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise JobConflict(f"{label} must be 1-128 ASCII identifier characters")
    return value


def _now(now: datetime) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise JobConflict("now must be a timezone-aware datetime")
    return now.astimezone(timezone.utc)


def _monotonic() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class _OperationClock:
    """Advance the caller's UTC clock across SQL waits and synchronous work."""

    origin: datetime
    started: float

    @classmethod
    def start(cls, now: datetime) -> _OperationClock:
        return cls(_now(now), _monotonic())

    def now(self) -> datetime:
        return self.origin + timedelta(seconds=max(0, _monotonic() - self.started))

    def require_live(self, deadline: datetime | None) -> datetime:
        # SQLite drops timezone information; these columns always store UTC.
        if deadline is None:
            raise LostLease("assistant job has no active lease")
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        current = self.now()
        if deadline <= current:
            raise LostLease("assistant job lease expired while waiting or processing")
        return current


def _expires(now: datetime, lease_sec: int) -> datetime:
    if type(lease_sec) is not int or not 1 <= lease_sec <= 300:
        raise JobConflict("lease_sec must be an integer within 1..300")
    return now + timedelta(seconds=lease_sec)


def _get(session: Session, job_id: str) -> AssistantJob:
    job = session.get(AssistantJob, job_id, populate_existing=True)
    if job is None:
        raise JobConflict("assistant job does not exist")
    return job


@contextmanager
def _atomic(session: Session):
    # sqlite3 legacy transaction mode does not start a real transaction for a
    # SELECT or SAVEPOINT. Without this, releasing the first savepoint commits
    # despite a later caller rollback. PostgreSQL already has a real BEGIN.
    connection = session.connection()
    if connection.dialect.name == "sqlite" and not connection.connection.driver_connection.in_transaction:
        connection.exec_driver_sql("BEGIN")
    with session.begin_nested():
        yield


def _event(session: Session, job: AssistantJob, key: str, kind: str, now: datetime) -> None:
    session.add(AssistantJobOutbox(
        id=_id(), job_id=job.id, event_key=key, kind=kind,
        payload={"job_id": job.id, "version_id": job.pending_version_id, "status": job.status},
        created_at=now,
    ))


def accept_job(
    session: Session, plan: EditPlan, *, request_scope: str,
    idempotency_key: str, actor: str, now: datetime, max_claims: int = 5,
) -> AssistantJob:
    """Reserve a child version + job + accepted outbox event atomically.

    A duplicate scoped key with identical plan/actor/policy returns the original
    job. Changing those fields conflicts, even after termination. Base is never
    resolved to latest here. Trusted callers must admit and authorize sources.
    """
    from .assistant_store import reserve_version

    now = _now(now)
    request_scope = _identifier(request_scope, "request_scope")
    idempotency_key = _identifier(idempotency_key, "idempotency_key")
    actor = _identifier(actor, "actor")
    if type(max_claims) is not int or not 1 <= max_claims <= 10:
        raise JobConflict("max_claims must be an integer within 1..10")
    # Revalidate even a model_copy/model_construct instance made by trusted code.
    accepted = EditPlan.model_validate(plan.model_dump(mode="json"))
    payload = accepted.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(
        {"plan": payload, "actor": actor, "max_claims": max_claims},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    predicate = and_(AssistantJob.request_scope == request_scope, AssistantJob.idempotency_key == idempotency_key)
    existing = session.scalar(select(AssistantJob).where(predicate).execution_options(populate_existing=True))
    if existing:
        if existing.request_sha256 != digest:
            raise JobConflict("idempotency key was already used for another request")
        return existing
    try:
        with _atomic(session):
            version_key = "job:" + hashlib.sha256((request_scope + "\0" + idempotency_key).encode()).hexdigest()
            version = reserve_version(session, accepted, idempotency_key=version_key, actor=actor)
            job = AssistantJob(
                id=_id(), request_scope=request_scope, idempotency_key=idempotency_key,
                request_sha256=digest, actor=actor, pinned_base_version_id=accepted.base_version_id,
                pending_version_id=version.id, accepted_plan=payload, status="queued", phase="resolve",
                lease_generation=0, claim_count=0, max_claims=max_claims, created_at=now, updated_at=now,
            )
            session.add(job)
            session.flush()
            _event(session, job, "accepted", "job.accepted", now)
            session.flush()
        return job
    except IntegrityError:
        # The unique key is the arbiter when concurrent acceptors both miss the
        # initial read. The savepoint rolls back any new reservation/outbox too.
        existing = session.scalar(select(AssistantJob).where(predicate).execution_options(populate_existing=True))
        if existing is None:
            raise
        if existing.request_sha256 != digest:
            raise JobConflict("idempotency key was already used for another request")
        return existing


def _owned(lease: JobLease, now: datetime, *, phase: str | None = None):
    return and_(
        AssistantJob.id == lease.job_id, AssistantJob.status == "running",
        AssistantJob.cancel_requested_at.is_(None), AssistantJob.lease_owner == lease.owner,
        AssistantJob.lease_token == lease.token, AssistantJob.lease_generation == lease.generation,
        AssistantJob.phase == (phase or lease.phase), AssistantJob.lease_expires_at > now,
    )


def _lock_owned(session: Session, lease: JobLease, clock: _OperationClock, *, phase: str | None = None) -> AssistantJob:
    # UPDATE is both a compare-and-set and a row lock held through the caller's
    # transaction. Publication/cancellation serialize on this SAME job row.
    now = clock.now()
    result = session.execute(update(AssistantJob).where(_owned(lease, now, phase=phase)).values(updated_at=now).execution_options(synchronize_session=False))
    if result.rowcount != 1:
        raise LostLease("assistant job lease expired, was cancelled or was fenced")
    job = _get(session, lease.job_id)
    # PostgreSQL re-evaluates the WHERE predicate after a row-lock wait, but
    # its bound timestamp is still the pre-wait value. Recheck only AFTER the
    # lock is ours, before renewing a lease or permitting external side effects.
    job.updated_at = clock.require_live(job.lease_expires_at)
    return job


def _clear_lease(job: AssistantJob) -> None:
    job.lease_owner = job.lease_token = job.lease_expires_at = None


def _terminal(session: Session, job: AssistantJob, status: str, now: datetime, error_code: str | None = None) -> None:
    from .assistant_store import mark_version_terminal

    if status != "succeeded":
        mark_version_terminal(session, job.pending_version_id, status=status)
        session.execute(update(AssistantPaidCallIntent).where(
            AssistantPaidCallIntent.job_id == job.id, AssistantPaidCallIntent.status == "submitting",
        ).values(status="unknown", ambiguity_reason="job ended while submission was outstanding", updated_at=now).execution_options(synchronize_session=False))
    job.status, job.finished_at, job.updated_at, job.error_code = status, now, now, error_code
    _clear_lease(job)
    session.flush()
    _event(session, job, "terminal", "job." + status, now)
    session.flush()


def claim_job(session: Session, job_id: str, *, owner: str, now: datetime, lease_sec: int = 60) -> JobLease | None:
    """Claim queued/expired work; no terminal job can be replayed.

    Reclaim preserves phase checkpoints. An interrupted submission becomes
    unknown; reclaim never grants another provider submission for that intent.
    Exhausted bounded claims fail both job and reserved version atomically.
    """
    clock, owner = _OperationClock.start(now), _identifier(owner, "owner")
    now = clock.now()
    _expires(now, lease_sec)
    predicate = and_(AssistantJob.id == job_id, AssistantJob.cancel_requested_at.is_(None), or_(
        AssistantJob.status == "queued",
        and_(AssistantJob.status == "running", AssistantJob.lease_expires_at <= now),
    ))
    with _atomic(session):
        result = session.execute(update(AssistantJob).where(predicate).values(updated_at=now).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            return None
        job = _get(session, job_id)
        session.execute(update(AssistantPaidCallIntent).where(
            AssistantPaidCallIntent.job_id == job_id, AssistantPaidCallIntent.status == "submitting",
        ).values(status="unknown", ambiguity_reason="lease lost while submission was outstanding", updated_at=now).execution_options(synchronize_session=False))
        now = clock.now()
        if job.claim_count >= job.max_claims:
            _terminal(session, job, "failed", now, "claim_limit_exhausted")
            return None
        expires = _expires(now, lease_sec)
        job.status, job.lease_owner, job.lease_token = "running", owner, _id()
        job.lease_expires_at, job.updated_at = expires, now
        job.lease_generation += 1
        job.claim_count += 1
        session.flush()
        clock.require_live(expires)
        return JobLease(job.id, owner, job.lease_token, job.lease_generation, job.phase, expires)


def renew_lease(session: Session, lease: JobLease, *, now: datetime, lease_sec: int = 60) -> JobLease:
    clock = _OperationClock.start(now)
    _expires(clock.now(), lease_sec)
    with _atomic(session):
        job = _lock_owned(session, lease, clock)
        original_deadline = job.lease_expires_at
        expires = _expires(clock.require_live(original_deadline), lease_sec)
        job.lease_expires_at = expires
        session.flush()
        clock.require_live(original_deadline)
        clock.require_live(expires)
    return JobLease(job.id, lease.owner, lease.token, lease.generation, job.phase, expires)


def advance_phase(session: Session, lease: JobLease, *, phase: str, now: datetime, lease_sec: int = 60) -> JobLease:
    """Start exactly the next phase, rotating the fence against late work."""
    if _NEXT_PHASE.get(lease.phase) != phase:
        raise JobConflict("phase must advance resolve -> render -> inspect -> publish")
    clock = _OperationClock.start(now)
    _expires(clock.now(), lease_sec)
    with _atomic(session):
        job = _lock_owned(session, lease, clock)
        original_deadline = job.lease_expires_at
        expires = _expires(clock.require_live(original_deadline), lease_sec)
        job.phase, job.lease_token, job.lease_expires_at = phase, _id(), expires
        job.lease_generation += 1
        session.flush()
        clock.require_live(original_deadline)
        clock.require_live(expires)
        return JobLease(job.id, lease.owner, job.lease_token, job.lease_generation, phase, expires)


def request_cancel(session: Session, job_id: str, *, now: datetime) -> AssistantJob:
    """Queued work cancels immediately; running work is fenced from publication."""
    clock = _OperationClock.start(now)
    now = clock.now()
    with _atomic(session):
        session.execute(update(AssistantJob).where(
            AssistantJob.id == job_id, AssistantJob.status.notin_(_TERMINAL),
        ).values(updated_at=now).execution_options(synchronize_session=False))
        job = _get(session, job_id)
        if job.status in _TERMINAL or job.cancel_requested_at is not None:
            return job
        now = clock.now()
        job.cancel_requested_at = now
        if job.status == "queued":
            _terminal(session, job, "cancelled", now)
        else:
            job.status = "cancel_requested"
            session.flush()
            _event(session, job, "cancel_requested", "job.cancel_requested", now)
            session.flush()
        return job


def finalize_cancel(session: Session, job_id: str, *, now: datetime, lease: JobLease | None = None) -> AssistantJob:
    """Current worker acknowledges cancellation; a reaper may finalize after expiry."""
    clock = _OperationClock.start(now)
    now = clock.now()
    allowed = AssistantJob.lease_expires_at <= now
    if lease is not None:
        if lease.job_id != job_id:
            raise LostLease("cancellation lease belongs to another job")
        allowed = and_(AssistantJob.lease_owner == lease.owner, AssistantJob.lease_token == lease.token,
                       AssistantJob.lease_generation == lease.generation, AssistantJob.lease_expires_at > now)
    with _atomic(session):
        result = session.execute(update(AssistantJob).where(
            AssistantJob.id == job_id, AssistantJob.status == "cancel_requested", allowed,
        ).values(updated_at=now).execution_options(synchronize_session=False))
        job = _get(session, job_id)
        if job.status == "cancelled":
            return job
        if result.rowcount != 1:
            raise LostLease("cancellation requires the current worker or an expired lease")
        original_deadline = job.lease_expires_at
        now = clock.require_live(original_deadline) if lease is not None else clock.now()
        _terminal(session, job, "cancelled", now)
        if lease is not None:
            clock.require_live(original_deadline)
        return job


def fail_job(session: Session, lease: JobLease, *, error_code: str, now: datetime) -> AssistantJob:
    clock, error_code = _OperationClock.start(now), _identifier(error_code, "error_code")
    with _atomic(session):
        job = _lock_owned(session, lease, clock)
        original_deadline = job.lease_expires_at
        _terminal(session, job, "failed", clock.require_live(original_deadline), error_code)
        clock.require_live(original_deadline)
        return job


def publish_job(session: Session, lease: JobLease, *, full: Any, preview: Any, storage: Any, now: datetime,
                clean: Any = None, captions: Any = None) -> AssistantJob:
    """Fence cancellation/late work and publish version + terminal event atomically.

    Artifact storage can leave unreferenced immutable objects after rollback;
    DB state/publication rolls back together. No upload/provider call occurs.
    """
    from .assistant_store import publish_version

    clock = _OperationClock.start(now)
    with _atomic(session):
        job = _lock_owned(session, lease, clock, phase="publish")
        original_deadline = job.lease_expires_at
        outstanding = session.scalar(select(AssistantPaidCallIntent.id).where(
            AssistantPaidCallIntent.job_id == job.id,
            AssistantPaidCallIntent.status.in_(("submitting", "unknown", "submitted")),
        ).limit(1))
        if outstanding:
            raise JobConflict("publication requires resolution of outstanding paid calls")
        publish_version(session, job.pending_version_id, full=full, preview=preview, storage=storage,
                        clean=clean, captions=captions)
        # Pinning can be slow. An entry-time check alone must not allow an
        # already-expired lease to publish after synchronous artifact work.
        completed_at = clock.require_live(original_deadline)
        _terminal(session, job, "succeeded", completed_at)
        clock.require_live(original_deadline)
        return job


def pending_outbox(session: Session, *, limit: int = 100) -> list[AssistantJobOutbox]:
    """At-least-once dispatcher input. Ack only AFTER a durable external enqueue."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise JobConflict("outbox limit must be within 1..1000")
    return list(session.scalars(select(AssistantJobOutbox).where(
        AssistantJobOutbox.dispatched_at.is_(None),
    ).order_by(AssistantJobOutbox.created_at, AssistantJobOutbox.id).limit(limit)))


def acknowledge_outbox(session: Session, event_id: str, *, now: datetime) -> None:
    result = session.execute(update(AssistantJobOutbox).where(
        AssistantJobOutbox.id == event_id, AssistantJobOutbox.dispatched_at.is_(None),
    ).values(dispatched_at=_now(now)).execution_options(synchronize_session=False))
    if result.rowcount != 1 and session.get(AssistantJobOutbox, event_id) is None:
        raise JobConflict("outbox event does not exist")


def prepare_paid_call(
    session: Session, lease: JobLease, *, logical_key: str, provider: str,
    request_sha256: str, now: datetime,
) -> AssistantPaidCallIntent:
    """Persist a logical call; the fingerprint contains no credentials/payload."""
    clock = _OperationClock.start(now)
    logical_key, provider = _identifier(logical_key, "logical_key"), _identifier(provider, "provider")
    if not isinstance(request_sha256, str) or not _DIGEST.fullmatch(request_sha256):
        raise JobConflict("request_sha256 must be a lowercase SHA-256 digest")
    with _atomic(session):
        job = _lock_owned(session, lease, clock)
        original_deadline = job.lease_expires_at
        intent = session.scalar(select(AssistantPaidCallIntent).where(
            AssistantPaidCallIntent.job_id == lease.job_id, or_(
                AssistantPaidCallIntent.logical_key == logical_key,
                and_(AssistantPaidCallIntent.provider == provider, AssistantPaidCallIntent.request_sha256 == request_sha256),
            ),
        ).execution_options(populate_existing=True))
        if intent:
            if (intent.logical_key, intent.provider, intent.request_sha256) != (logical_key, provider, request_sha256):
                raise JobConflict("paid call logical key already represents another request")
            clock.require_live(original_deadline)
            return intent
        now = clock.require_live(original_deadline)
        intent = AssistantPaidCallIntent(
            id=_id(), job_id=lease.job_id, logical_key=logical_key, provider=provider,
            request_sha256=request_sha256, provider_request_key="assistant-" + _id(),
            status="prepared", created_at=now, updated_at=now,
        )
        session.add(intent)
        session.flush()
        clock.require_live(original_deadline)
        return intent


def begin_paid_submission(session: Session, lease: JobLease, intent_id: str, *, now: datetime) -> SubmissionPermit:
    """Single submission permit. Caller MUST commit it before doing network I/O."""
    clock = _OperationClock.start(now)
    with _atomic(session):
        job = _lock_owned(session, lease, clock)
        original_deadline = job.lease_expires_at
        now = clock.require_live(original_deadline)
        token = _id()
        result = session.execute(update(AssistantPaidCallIntent).where(
            AssistantPaidCallIntent.id == intent_id, AssistantPaidCallIntent.job_id == lease.job_id,
            AssistantPaidCallIntent.status == "prepared",
        ).values(status="submitting", submission_token=token, updated_at=now).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            raise BlindRetryProhibited("intent is not unsubmitted; reconcile its existing provider operation")
        intent = session.get(AssistantPaidCallIntent, intent_id, populate_existing=True)
        clock.require_live(original_deadline)
        return SubmissionPermit(intent.id, token, intent.provider_request_key)


def record_submission(
    session: Session, permit: SubmissionPermit, *, provider_task_id: str | None, now: datetime,
) -> AssistantPaidCallIntent:
    """Record a late receipt/ambiguity even after cancellation or lease loss.

    Permit binds the one original submission. It never authorizes another call.
    A known task cannot be replaced or erased by a late timeout notification.
    """
    now = _now(now)
    if provider_task_id is not None and (
        not isinstance(provider_task_id, str) or not 1 <= len(provider_task_id) <= 255
        or any(ord(c) < 33 or ord(c) > 126 for c in provider_task_id)
    ):
        raise JobConflict("provider_task_id must be a nonempty printable ASCII receipt")
    with _atomic(session):
        result = session.execute(update(AssistantPaidCallIntent).where(
            AssistantPaidCallIntent.id == permit.intent_id,
            AssistantPaidCallIntent.submission_token == permit.token,
            AssistantPaidCallIntent.provider_request_key == permit.provider_request_key,
        ).values(updated_at=now).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            raise JobConflict("unknown provider submission permit")
        intent = session.get(AssistantPaidCallIntent, permit.intent_id, populate_existing=True)
        if intent.provider_task_id:
            if provider_task_id is not None and intent.provider_task_id != provider_task_id:
                raise JobConflict("submission already has a different provider receipt")
            return intent
        if intent.status not in ("submitting", "unknown"):
            raise JobConflict("submission cannot accept a receipt in its current state")
        if provider_task_id is None:
            intent.status, intent.ambiguity_reason = "unknown", "submission outcome is unknown; reconciliation required"
        else:
            intent.status, intent.provider_task_id, intent.ambiguity_reason = "submitted", provider_task_id, None
        session.flush()
        return intent


def complete_paid_call(session: Session, permit: SubmissionPermit, *, succeeded: bool, now: datetime) -> AssistantPaidCallIntent:
    """Persist a definitive provider outcome; failed intents also cannot resubmit."""
    if type(succeeded) is not bool:
        raise JobConflict("succeeded must be a boolean")
    now = _now(now)
    with _atomic(session):
        result = session.execute(update(AssistantPaidCallIntent).where(
            AssistantPaidCallIntent.id == permit.intent_id, AssistantPaidCallIntent.submission_token == permit.token,
            AssistantPaidCallIntent.provider_request_key == permit.provider_request_key,
        ).values(updated_at=now).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            raise JobConflict("unknown provider submission permit")
        intent = session.get(AssistantPaidCallIntent, permit.intent_id, populate_existing=True)
        terminal = "completed" if succeeded else "failed"
        if intent.status == terminal:
            return intent
        if intent.status != "submitted" or not intent.provider_task_id:
            raise BlindRetryProhibited("provider outcome requires a known submitted task")
        intent.status = terminal
        session.flush()
        return intent
