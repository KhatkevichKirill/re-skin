"""Paid submit attempts: the receipt before every paid provider call (AICP-351
= AICP-344 stage 2, docs/assistant-drain-check-design.md §5).

The window this closes: a worker that dies between kie.ai accepting a paid
``createTask`` and committing the task id used to resubmit on recovery and bill
twice. Now each paid video/image call first commits a ``started`` row in
``paid_submit_attempts`` (with an estimated draw keyed ``attempt:<id>``), then
makes ONE POST (``KieClient.create_*(..., once=True)``), then records the
outcome on that row in the same transaction as the owner row. Recovery parks
any ``started`` attempt that has no task id of its own — whatever the owner row
holds (during an in-poll resubmit ``RunSegment.seedance_task_id`` still holds
the PREVIOUS task) — and never resubmits it. The operator resolves a parked
attempt (resubmit / abandon, audited in ``paid_submit_attempt_actions``), and
only once its sender is RELEASED: its own post-POST write, sender proof
(``rebuild_paid_call.sender_proof``: boot id changed, job finished / raised /
horse reaped) or host attestation (``scripts/paid_attempt_release_sender.sh``).
Elapsed time is never proof.

LLM-only jobs (§5.2) use :class:`LlmLedger`: one attempt row per run of the
job's LLM phase (at most :data:`LLM_MAX_ATTEMPTS`), every call claimed by a CAS
on the row (``calls_sent = n-1 -> n``) in the same transaction as its draw keyed
``llm:<attempt_id>:<n>`` — committed BEFORE the call, replacing the post-call
draw — and sent with one POST. A ``superseded`` attempt (lease expired with no
proof) stays counted ``active`` until its own worker acknowledges or its sender
is proven gone (§9 item 2).

CLI (inside the api container)::

    python -m app.paid_attempts list [--all]
    python -m app.paid_attempts resubmit|abandon <attempt id> --actor <who> --note <what you saw>
    python -m app.paid_attempts release-sender <attempt id> --attest <container id>:<state>
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import spend as spend_mod
from .paid_attempts_models import (
    ABANDONED,
    ACCEPTED,
    ACTIVE_STATUSES,
    FAILED,
    FAILED_UNBILLED,
    LLM_OWNERS,
    OPEN_STATUSES,
    PARKED,
    RESUBMITTED,
    STARTED,
    STOPPED,
    SUPERSEDED,
    UNKNOWN,
    VIDEO_IMAGE_OWNERS,
    PaidSubmitAttempt,
    PaidSubmitAttemptAction,
    utcnow,
)

log = logging.getLogger(__name__)

#: At most this many LLM attempts per job: the first + 2 automatic retries (§5.2).
LLM_MAX_ATTEMPTS = 3
#: Model fallback chains are capped at this length, so the admission reservation
#: (attempts x calls x chain) is a real bound.
LLM_MAX_CHAIN = 4
#: Logical LLM calls per attempt: translation = main + 1:1 re-ask + shorten pass.
TRANSLATE_CALLS_PER_ATTEMPT = 3
TRANSCRIBE_CALLS_PER_ATTEMPT = 1


class AttemptBlocked(RuntimeError):
    """No paid call may be made: the lease is lost, the attempt is no longer
    ``started``, or another attempt of the owner is open."""


class LedgerFenced(RuntimeError):
    """An LLM attempt may make no further call (superseded, revoked, or its
    bound is spent). Never a provider error: the job stops without writing."""


class ActionRefused(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ---------------------------------------------------------------------------
# Lease + sender identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sender:
    host: Optional[str] = None
    boot: Optional[str] = None
    job: Optional[str] = None


@dataclass
class SubmitLease:
    """What a paid-call site carries from the RQ main thread: the fence check
    (the per-run Redis lock refresh, by token) and the sender identity.
    ``rq.get_current_job()`` is thread-local, so the identity is captured here,
    not inside the submit thread pool."""

    check: Callable[[], bool]
    token: Optional[str] = None
    sender: Sender = Sender()

    def ok(self) -> bool:
        try:
            return bool(self.check())
        except Exception:  # a Redis error proves nothing: do not send
            log.exception("paid_attempts: lease check failed")
            return False


def current_sender(job_id: Optional[str] = None) -> Sender:
    """This process's identity for the sender proof (empty outside an RQ job
    or without a registered boot id: such an attempt is released only by its
    own write or by attestation)."""
    try:
        from .rebuild_paid_call import current_boot_id, current_hostname

        if job_id is None:
            from rq import get_current_job

            job = get_current_job()
            job_id = job.id if job is not None else None
        return Sender(host=current_hostname(), boot=current_boot_id(), job=job_id)
    except Exception:
        return Sender()


def run_lease(conn, lock_key: str, lock_token: str) -> SubmitLease:
    """The lease of a run phase holding the per-run Redis lock."""
    from .tasks import _refresh_lock

    return SubmitLease(check=lambda: _refresh_lock(conn, lock_key, lock_token),
                       token=lock_token, sender=current_sender())


def request_sha256(request: dict) -> str:
    blob = json.dumps(request, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def attempt_call_key(attempt_id: str) -> str:
    return f"attempt:{attempt_id}"


def llm_call_key(attempt_id: str, call_no: int) -> str:
    return f"llm:{attempt_id}:{int(call_no)}"


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def open_attempt(db: Session, owner_kind: str, owner_id: str, *,
                 for_update: bool = False) -> Optional[PaidSubmitAttempt]:
    q = select(PaidSubmitAttempt).where(
        PaidSubmitAttempt.owner_kind == owner_kind,
        PaidSubmitAttempt.owner_id == str(owner_id),
        PaidSubmitAttempt.status.in_(OPEN_STATUSES),
    )
    if for_update:
        q = q.with_for_update()
    return db.execute(q.execution_options(populate_existing=True)).scalars().first()


def last_task_id(db: Session, owner_kind: str, owner_id: str) -> Optional[str]:
    return db.execute(
        select(PaidSubmitAttempt.provider_task_id).where(
            PaidSubmitAttempt.owner_kind == owner_kind,
            PaidSubmitAttempt.owner_id == str(owner_id),
            PaidSubmitAttempt.provider_task_id.is_not(None),
        ).order_by(PaidSubmitAttempt.attempt_no.desc())
    ).scalars().first()


def latest_attempt(db: Session, owner_kind: str, owner_id: str) -> Optional[PaidSubmitAttempt]:
    """The owner's highest-numbered attempt, or None."""
    return db.execute(
        select(PaidSubmitAttempt).where(
            PaidSubmitAttempt.owner_kind == owner_kind,
            PaidSubmitAttempt.owner_id == str(owner_id),
        ).order_by(PaidSubmitAttempt.attempt_no.desc())
    ).scalars().first()


#: How an LLM attempt ended that §5.2 may retry: the outcome is unknown.
RETRYABLE_LLM_STATUSES = (UNKNOWN, SUPERSEDED, STOPPED)


def attempt_count(db: Session, owner_kind: str, owner_id: str) -> int:
    return int(db.execute(
        select(func.count()).select_from(PaidSubmitAttempt).where(
            PaidSubmitAttempt.owner_kind == owner_kind,
            PaidSubmitAttempt.owner_id == str(owner_id),
        )
    ).scalar_one())


def live_attempts(db: Session, owner_kind: str, owner_id: str) -> int:
    """Attempts whose sender may still send (``started`` / ``superseded``)."""
    return int(db.execute(
        select(func.count()).select_from(PaidSubmitAttempt).where(
            PaidSubmitAttempt.owner_kind == owner_kind,
            PaidSubmitAttempt.owner_id == str(owner_id),
            PaidSubmitAttempt.status.in_(ACTIVE_STATUSES),
        )
    ).scalar_one())


def run_open_attempts(db: Session, run_id: str) -> list[PaidSubmitAttempt]:
    """Open (started / parked) video/image attempts of a run — the run page's
    "paid submit outcome unknown" card and the routes' 409 gates."""
    return list(db.execute(
        select(PaidSubmitAttempt).where(
            PaidSubmitAttempt.run_id == run_id,
            PaidSubmitAttempt.owner_kind.in_(VIDEO_IMAGE_OWNERS),
            PaidSubmitAttempt.status.in_(OPEN_STATUSES),
        ).order_by(PaidSubmitAttempt.started_at)
    ).scalars())


def owner_open(db: Session, owner_kind: str, owner_id: str) -> bool:
    return open_attempt(db, owner_kind, owner_id) is not None


def _next_attempt_no(db: Session, owner_kind: str, owner_id: str) -> int:
    cur = db.execute(
        select(func.max(PaidSubmitAttempt.attempt_no)).where(
            PaidSubmitAttempt.owner_kind == owner_kind,
            PaidSubmitAttempt.owner_id == str(owner_id),
        )
    ).scalar_one()
    return int(cur or 0) + 1


def _default_sessions():
    from .db import get_session

    return get_session


# ---------------------------------------------------------------------------
# §5.1 video / image attempts
# ---------------------------------------------------------------------------


def begin(
    owner_kind: str,
    owner_id: str,
    *,
    kind: str,
    run_id: Optional[str],
    request: dict,
    credits: float,
    scope_key: str,
    draw_kind: str,
    predecessor_task_id: Optional[str] = None,
    lease: Optional[SubmitLease] = None,
    fence: Optional[Callable[[Session], bool]] = None,
    sessions=None,
) -> str:
    """Commit a ``started`` attempt + its estimated draw BEFORE the paid call.

    *fence* runs FIRST in the same transaction: it locks the OWNER row and
    says whether this sender may still submit for it (e.g. the segment is
    still ``submitted`` with no task id, the image job still ``queued``). So a
    sender that paused before this point cannot submit for an owner a
    recovery/reaper has since moved on (lock order owner -> attempt).

    Raises :class:`AttemptBlocked` (send nothing) when the lease is lost, the
    fence refuses, or another attempt of the owner is open. Downstream of an
    admitted run: the admission gate does not apply."""
    sessions = sessions or _default_sessions()
    if lease is not None and not lease.ok():
        raise AttemptBlocked(f"{owner_kind} {owner_id}: lease lost before the paid call")
    sender = lease.sender if lease is not None else current_sender()
    now = utcnow()
    try:
        with sessions() as db:
            if fence is not None and not fence(db):
                db.rollback()
                raise AttemptBlocked(f"{owner_kind} {owner_id}: the owner moved on — not sent")
            if open_attempt(db, owner_kind, owner_id) is not None:
                raise AttemptBlocked(f"{owner_kind} {owner_id}: another attempt is open")
            row = PaidSubmitAttempt(
                owner_kind=owner_kind, owner_id=str(owner_id),
                attempt_no=_next_attempt_no(db, owner_kind, owner_id), run_id=run_id,
                kind=kind, provider="kie", request_sha256=request_sha256(request),
                lease_generation=(lease.token if lease is not None else None),
                status=STARTED, predecessor_task_id=predecessor_task_id,
                started_at=now, updated_at=now,
                sender_host=sender.host, sender_boot=sender.boot, sender_job=sender.job,
            )
            db.add(row)
            db.flush()
            spend_mod.draw_in_session(db, scope_key, attempt_call_key(row.id), credits,
                                      kind=draw_kind, provider="kie")
            attempt_id = row.id
            db.commit()
    except IntegrityError as exc:
        raise AttemptBlocked(f"{owner_kind} {owner_id}: another attempt is open ({exc.orig})") from exc
    log.info("paid attempt %s started (%s %s, predecessor=%s)", attempt_id, owner_kind,
             owner_id, predecessor_task_id)
    return attempt_id


def pre_send_check(attempt_id: str, lease: Optional[SubmitLease], *, scope_key: Optional[str] = None,
                   kind: str = "attempt", sessions=None) -> None:
    """Re-check the lease and the row right before the POST. On a refusal the
    sender PROVES it never sent: the attempt is closed ``failed_unbilled``
    (from ``started`` or from a recovery's ``parked``), released, and
    :class:`AttemptBlocked` is raised."""
    sessions = sessions or _default_sessions()
    lease_ok = lease is None or lease.ok()
    with sessions() as db:
        row = db.get(PaidSubmitAttempt, attempt_id, with_for_update=True, populate_existing=True)
        if row is not None and row.status == STARTED and lease_ok:
            db.rollback()
            return
        reason = "lease lost" if not lease_ok else f"status {getattr(row, 'status', None)}"
        if row is not None and row.status in (STARTED, PARKED):
            now = utcnow()
            row.status, row.finished_at, row.updated_at = FAILED_UNBILLED, now, now
            row.error = f"not sent: {reason} before the paid call"
            row.sender_released_at = row.sender_released_at or now
            row.sender_note = "the sender stopped before its POST"
            if scope_key:
                _move_attempt_draw(db, attempt_id, outcome="unbilled", task_id=None,
                                   credits=0.0, scope_key=scope_key, kind=kind)
            db.commit()
        else:
            db.rollback()
    raise AttemptBlocked(f"attempt {attempt_id}: {reason} — not sent")


def classify_send_error(exc: BaseException) -> str:
    """``unbilled`` only when the send is PROVEN not to have billed: a
    single-send refusal, or a client-side ValueError/TypeError raised before
    any POST (validation). Everything else is ``unknown``."""
    from .kie_client import KieTaskRefused

    if isinstance(exc, KieTaskRefused):
        return "unbilled"
    if isinstance(exc, (ValueError, TypeError)):
        return "unbilled"
    return "unknown"


def record_outcome(
    db: Session,
    attempt_id: str,
    *,
    outcome: str,
    task_id: Optional[str] = None,
    error: Optional[str] = None,
    credits: float = 0.0,
    scope_key: Optional[str] = None,
    draw_kind: str = "attempt",
) -> bool:
    """Record the POST's outcome on the attempt in the CALLER's transaction
    (the caller commits it together with the owner row). Releases the sender.

    *outcome*: ``accepted`` | ``unbilled`` | ``unknown``. Returns True when the
    caller may write the owner row (an ``accepted`` attempt that was still
    ``started``); False when a recovery parked it meanwhile — then the task id
    is kept on the parked row as a receipt and the owner row is left alone.

    Spend moves in the same transaction (a savepoint, so an accounting error
    never loses the receipt — it then over-, never under-counts): accepted ->
    the usual draw keyed by the task id at *credits* and the attempt's
    estimate settled to 0; unbilled -> the estimate settled to 0; unknown ->
    the estimate stays."""
    row = db.get(PaidSubmitAttempt, attempt_id, with_for_update=True, populate_existing=True)
    if row is None:
        log.error("paid attempt %s vanished before its outcome (%s)", attempt_id, outcome)
        return False
    now = utcnow()
    row.updated_at = now
    row.sender_released_at = row.sender_released_at or now
    row.sender_note = row.sender_note or f"sender recorded its outcome ({outcome})"
    if error:
        row.error = str(error)[:4000]
    if task_id:
        row.provider_task_id = task_id
    owner_ok = False
    if row.status == STARTED:
        if outcome == "accepted":
            row.status, row.finished_at = ACCEPTED, now
            owner_ok = True
        elif outcome == "unbilled":
            row.status, row.finished_at = FAILED_UNBILLED, now
        else:
            row.status = PARKED
    elif row.status == PARKED and outcome == "unbilled":
        row.status, row.finished_at = FAILED_UNBILLED, now
    elif row.status == PARKED and outcome == "accepted":
        row.error = (f"late receipt: provider accepted task {task_id} after recovery parked "
                     "this attempt — the owner was NOT updated; check the provider dashboard")
    if scope_key and outcome in ("accepted", "unbilled"):
        _move_attempt_draw(db, attempt_id, outcome=outcome, task_id=task_id,
                           credits=credits, scope_key=scope_key, kind=draw_kind)
    return owner_ok


def _move_attempt_draw(db: Session, attempt_id: str, *, outcome: str, task_id: Optional[str],
                       credits: float, scope_key: str, kind: str) -> None:
    try:
        with db.begin_nested():
            if outcome == "accepted" and task_id:
                spend_mod.draw_in_session(db, scope_key, task_id, credits, kind=kind,
                                          provider="kie")
            spend_mod.settle_in_session(db, scope_key, attempt_call_key(attempt_id), 0.0,
                                        kind=kind, scope_lock=False)
    except Exception:
        log.exception("kie spend: ACCOUNTING FAILURE moving the draw of attempt %s (%s, task "
                      "%s); the estimate stays on the day", attempt_id, outcome, task_id)


def close_after_error(attempt_id: str, exc: BaseException, *, scope_key: str, kind: str,
                      sessions=None) -> str:
    """The POST raised: record ``failed_unbilled`` or ``parked`` (own session).
    Returns the outcome (``unbilled`` | ``unknown``)."""
    sessions = sessions or _default_sessions()
    outcome = classify_send_error(exc)
    with sessions() as db:
        record_outcome(db, attempt_id, outcome=outcome, error=f"{type(exc).__name__}: {exc}",
                       scope_key=scope_key, draw_kind=kind)
        db.commit()
    if outcome == "unknown":
        log.error("paid attempt %s PARKED: submit outcome unknown (%s) — never resubmitted "
                  "blind; the operator resolves it on the run page", attempt_id, exc)
    return outcome


def park_started(db: Session, row: PaidSubmitAttempt, *, reason: str,
                 evidence: Optional[Callable] = None) -> bool:
    """Recovery's rule (caller's transaction): a ``started`` attempt without its
    own task id is parked, never resubmitted. Its sender is released only on
    proof. Returns True when it parked the row."""
    if row.status != STARTED:
        return False
    now = utcnow()
    row.status, row.updated_at = PARKED, now
    row.error = (row.error or "") + (f"; {reason}" if row.error else reason)
    _maybe_release(row, evidence, now)
    log.error("paid attempt %s (%s %s) PARKED by recovery: %s", row.id, row.owner_kind,
              row.owner_id, reason)
    return True


def _maybe_release(row: PaidSubmitAttempt, evidence: Optional[Callable], now) -> bool:
    if row.sender_released_at is not None or evidence is None:
        return row.sender_released_at is not None
    try:
        proof = evidence(row)
    except Exception:
        log.exception("paid attempt %s: sender evidence unreadable", row.id)
        proof = None
    if proof:
        row.sender_released_at, row.sender_note = now, f"proof: {proof}"
        return True
    return False


def default_evidence():
    """The sender proof over Redis (rebuild_paid_call.default_evidence), or None
    when Redis is unavailable."""
    try:
        from .rebuild_paid_call import default_evidence as _ev
        from .tasks import _get_redis

        return _ev(_get_redis())
    except Exception:
        log.warning("paid_attempts: sender evidence unavailable", exc_info=True)
        return None


def park_owner(owner_kind: str, owner_id: str, *, reason: str, sessions=None,
               evidence: Optional[Callable] = None) -> Optional[PaidSubmitAttempt]:
    """Own session: park the owner's ``started`` attempt (if any) and return the
    open attempt (``parked``), or None when the owner has none."""
    sessions = sessions or _default_sessions()
    with sessions() as db:
        row = open_attempt(db, owner_kind, owner_id, for_update=True)
        if row is None:
            db.rollback()
            return None
        if row.status == STARTED:
            park_started(db, row, reason=reason, evidence=evidence)
        db.commit()
        db.expunge(row)
        return row


# ---------------------------------------------------------------------------
# Operator actions (audited; like pipeline_provider close|resume)
# ---------------------------------------------------------------------------


def _audit(db: Session, row: PaidSubmitAttempt, action: str, actor: str, to: str,
           note: Optional[str]) -> None:
    db.add(PaidSubmitAttemptAction(attempt_id=row.id, action=action, actor=actor or "?",
                                   from_status=row.status, to_status=to, note=note))
    db.flush()


def _resolvable(db: Session, attempt_id: str) -> PaidSubmitAttempt:
    row = db.get(PaidSubmitAttempt, attempt_id, with_for_update=True, populate_existing=True)
    if row is None:
        raise ActionRefused(404, "paid attempt not found")
    if row.status != PARKED:
        raise ActionRefused(409, f"attempt is {row.status}, not parked")
    if row.sender_released_at is None:
        raise ActionRefused(409, (
            "the original sender of this attempt is not proven stopped and may still "
            "send — wait for the worker to finish or restart, or (if its container is "
            f"gone) run scripts/paid_attempt_release_sender.sh {row.id} on the host"))
    return row


def resubmit(db: Session, attempt_id: str, *, actor: str, note: Optional[str]) -> PaidSubmitAttempt:
    """CAS ``parked -> resubmitted`` + audit in the caller's transaction (no
    commit): the caller relaunches the owner in the SAME transaction, so a
    failed relaunch rolls the CAS back."""
    row = _resolvable(db, attempt_id)
    _audit(db, row, "resubmit", actor, RESUBMITTED, note)
    now = utcnow()
    row.status, row.finished_at, row.updated_at = RESUBMITTED, now, now
    db.flush()
    return row


def abandon(db: Session, attempt_id: str, *, actor: str, note: Optional[str]) -> PaidSubmitAttempt:
    """CAS ``parked -> abandoned`` + audit (no relaunch). Ungated (§6 rollback
    barrier). The estimate stays on the day: the call may have billed."""
    row = _resolvable(db, attempt_id)
    _audit(db, row, "abandon", actor, ABANDONED, note)
    now = utcnow()
    row.status, row.finished_at, row.updated_at = ABANDONED, now, now
    db.flush()
    return row


def release_sender(db: Session, attempt_id: str, *, attest: str, actor: str = "host-attestation") -> PaidSubmitAttempt:
    """Host attestation that the sender container no longer runs
    (``<container id>:<state>`` from ``docker ps -a --no-trunc``): releases a
    parked/started/superseded attempt's sender. A ``started`` row is parked
    (its outcome is unknown); ``superseded`` -> ``stopped``."""
    row = db.get(PaidSubmitAttempt, attempt_id, with_for_update=True, populate_existing=True)
    if row is None:
        raise ActionRefused(404, "paid attempt not found")
    if row.sender_released_at is not None and row.status not in (STARTED, SUPERSEDED):
        raise ActionRefused(409, "the sender is already released")
    if row.status not in (STARTED, PARKED, SUPERSEDED):
        raise ActionRefused(409, f"attempt is {row.status}; nothing to release")
    container, _, state = (attest or "").partition(":")
    host = row.sender_host or ""
    if not host or not container or not container.startswith(host) or state not in ("absent", "exited"):
        raise ActionRefused(409, f"attestation {attest!r} does not name the sender host {host!r} "
                                 "as absent/exited")
    now = utcnow()
    if row.status == STARTED:
        to = PARKED if row.owner_kind in VIDEO_IMAGE_OWNERS else UNKNOWN
    elif row.status == SUPERSEDED:
        to = STOPPED
    else:
        to = row.status
    _audit(db, row, "release_sender", actor, to, f"attested {attest}")
    row.status, row.updated_at = to, now
    if to in (STOPPED, UNKNOWN):
        row.finished_at = now
    row.sender_released_at, row.sender_note = now, f"host attestation {attest}"
    db.flush()
    return row


# ---------------------------------------------------------------------------
# §5.2 LLM attempts
# ---------------------------------------------------------------------------


def llm_lease_sec(job_timeout_sec: float) -> int:
    """At least the RQ job_timeout + 300 s (tasks._RUN_LOCK_TTL_SEC pattern)."""
    return int(job_timeout_sec) + 300


def begin_llm(
    owner_kind: str,
    owner_id: str,
    *,
    max_calls: int,
    lease_sec: int,
    fence_token: Optional[str],
    run_id: Optional[str] = None,
    request: Optional[dict] = None,
    sender: Optional[Sender] = None,
    fence: Optional[Callable[[Session], bool]] = None,
    sessions=None,
) -> Optional[str]:
    """Commit a ``started`` LLM attempt; None when the bound
    (:data:`LLM_MAX_ATTEMPTS`) is spent, another attempt is open, or the job's
    *fence* (it LOCKS the owner row first — lock order owner -> attempt) says
    this worker no longer owns the job. Raises :class:`LedgerFenced` for the
    last case."""
    assert owner_kind in LLM_OWNERS
    sessions = sessions or _default_sessions()
    sender = sender or current_sender()
    now = utcnow()
    try:
        with sessions() as db:
            if fence is not None and not fence(db):
                db.rollback()
                raise LedgerFenced(f"{owner_kind} {owner_id}: the job is no longer ours")
            if attempt_count(db, owner_kind, owner_id) >= LLM_MAX_ATTEMPTS:
                return None
            if open_attempt(db, owner_kind, owner_id) is not None:
                return None
            row = PaidSubmitAttempt(
                owner_kind=owner_kind, owner_id=str(owner_id),
                attempt_no=_next_attempt_no(db, owner_kind, owner_id), run_id=run_id,
                kind="llm", provider="kie", request_sha256=request_sha256(request or {}),
                lease_generation=fence_token, status=STARTED, started_at=now, updated_at=now,
                calls_sent=0, max_calls=int(max_calls),
                lease_expires_at=now + timedelta(seconds=int(lease_sec)),
                sender_host=sender.host, sender_boot=sender.boot, sender_job=sender.job,
            )
            db.add(row)
            db.commit()
            return row.id
    except IntegrityError:
        return None


class LlmLedger:
    """The per-attempt call permit of an LLM-only job (§5.2).

    :meth:`claim` = one transaction: lock the attempt row, CAS ``status =
    'started' AND calls_sent = n-1 AND n <= max_calls`` (+ the job's own
    *fence*), set ``calls_sent = n`` and draw ``llm:<attempt_id>:<n>`` at the
    estimate; commit. Only then the caller makes ONE POST. A failed CAS raises
    :class:`LedgerFenced` and nothing is sent. The attempt id is minted by
    this process and never shared, so ``n`` is this attempt's own sequence."""

    def __init__(self, attempt_id: str, *, scope_key: str, kind: str,
                 fence: Optional[Callable[[Session], bool]] = None, sessions=None):
        self.attempt_id = attempt_id
        self.scope_key = scope_key
        self.kind = kind
        self.fence = fence
        self.sessions = sessions or _default_sessions()
        self.calls = 0

    def claim(self) -> int:
        n = self.calls + 1
        with self.sessions() as db:
            # Lock order owner -> attempt (the fence locks the owner row).
            fenced = self.fence is not None and not self.fence(db)
            row = db.get(PaidSubmitAttempt, self.attempt_id, with_for_update=True,
                         populate_existing=True)
            ok = (not fenced and row is not None and row.status == STARTED
                  and int(row.calls_sent or 0) == n - 1
                  and n <= int(row.max_calls or 0))
            if not ok:
                db.rollback()
                raise LedgerFenced(
                    f"LLM attempt {self.attempt_id}: call {n} refused "
                    f"(status={getattr(row, 'status', None)}, calls_sent="
                    f"{getattr(row, 'calls_sent', None)}, max={getattr(row, 'max_calls', None)})")
            row.calls_sent, row.updated_at = n, utcnow()
            db.flush()
            spend_mod.draw_in_session(db, self.scope_key, llm_call_key(self.attempt_id, n),
                                      spend_mod.llm_credits(), kind=self.kind, provider="kie")
            db.commit()
        self.calls = n
        return n

    def before_post(self, n: int) -> None:
        """Right before the POST: a superseded/revoked attempt does not send the
        call it claimed (its draw is settled to 0 — proven unsent)."""
        with self.sessions() as db:
            row = db.get(PaidSubmitAttempt, self.attempt_id, populate_existing=True)
            status = getattr(row, "status", None)
            fenced = self.fence is not None and not self.fence(db)
        if status == STARTED and not fenced:
            return
        self.settle(n, 0.0)
        raise LedgerFenced(f"LLM attempt {self.attempt_id}: {status} before call {n} — not sent")

    def settle(self, n: int, credits: Optional[float]) -> None:
        if credits is None:
            return
        spend_mod.settle_draw(llm_call_key(self.attempt_id, n), credits,
                              scope_key=self.scope_key, kind=self.kind, sessions=self.sessions)


def finish_llm(db: Session, attempt_id: str, outcome: str, *, error: Optional[str] = None) -> bool:
    """The worker's own acknowledgement after its LLM phase's last I/O returned
    (caller's transaction): ``started -> accepted|failed|unknown`` (True for
    an accepted CAS), ``superseded -> stopped``. Releases the sender."""
    row = db.get(PaidSubmitAttempt, attempt_id, with_for_update=True, populate_existing=True)
    if row is None:
        return False
    now = utcnow()
    row.updated_at = now
    if error:
        row.error = str(error)[:4000]
    if row.status == STARTED:
        row.status = {"accepted": ACCEPTED, "failed": FAILED}.get(outcome, UNKNOWN)
        row.finished_at = now
        row.sender_released_at = row.sender_released_at or now
        row.sender_note = row.sender_note or f"sender finished ({outcome})"
        return row.status == ACCEPTED
    if row.status == SUPERSEDED:
        row.status, row.finished_at = STOPPED, now
        row.sender_released_at = row.sender_released_at or now
        row.sender_note = "the superseded sender acknowledged it stopped"
    return False


def finish_llm_own_session(attempt_id: str, outcome: str, *, error: Optional[str] = None,
                           sessions=None) -> None:
    sessions = sessions or _default_sessions()
    try:
        with sessions() as db:
            finish_llm(db, attempt_id, outcome, error=error)
            db.commit()
    except Exception:
        log.exception("LLM attempt %s: could not record its end (%s)", attempt_id, outcome)


def reap_llm_attempt(db: Session, row: PaidSubmitAttempt, *, now: datetime,
                     evidence: Optional[Callable]) -> Optional[str]:
    """The job reaper's decision on a dead job's ``started`` LLM attempt
    (caller's transaction). Returns ``unknown`` (sender proven gone),
    ``superseded`` (lease expired, no proof: STILL counted active) or None
    (leave it: the job stays running)."""
    if row.status != STARTED:
        return row.status
    if _maybe_release(row, evidence, now):
        row.status, row.finished_at, row.updated_at = UNKNOWN, now, now
        return UNKNOWN
    if row.lease_expires_at is not None and now >= row.lease_expires_at:
        row.status, row.updated_at = SUPERSEDED, now
        return SUPERSEDED
    return None


def release_llm_scope_if_idle(db: Session, owner_kind: str, owner_id: str, scope_key: str,
                              *, kind: str) -> bool:
    """Shrink *scope_key*'s reservation to what it drew once no attempt of the
    owner can still send (§5.2: the reservation covers every potentially live
    attempt). Caller's transaction; returns True when released."""
    from .paid_attempts_models import OWNER_LLM_CAPTION_ASR

    if owner_kind == OWNER_LLM_CAPTION_ASR:
        # The caption scope spans every epoch (operator retries) of the caption.
        cid = owner_id.split(":", 1)[0]
        live = int(db.execute(select(func.count()).select_from(PaidSubmitAttempt).where(
            PaidSubmitAttempt.owner_kind == owner_kind,
            PaidSubmitAttempt.owner_id.like(f"{cid}:%"),
            PaidSubmitAttempt.status.in_(ACTIVE_STATUSES),
        )).scalar_one())
    else:
        live = live_attempts(db, owner_kind, owner_id)
    if live:
        return False
    spend_mod.reserve_computed(db, scope_key, lambda _db: 0.0, kind=kind, release_unused=True)
    return True


# ---------------------------------------------------------------------------
# Reconcile-tick sweep
# ---------------------------------------------------------------------------


def sweep(*, sessions=None, evidence: Optional[Callable] = None,
          started_park_after_sec: Optional[float] = None, now=None) -> dict:
    """One pass from the reconcile tick (never raises):

    * a parked attempt whose sender is not released: re-check the proof;
    * a ``superseded`` LLM attempt (its job may already be terminal): proof ->
      ``stopped`` and its scope released when idle;
    * a video/image ``started`` attempt older than the run-lock TTL is parked
      (age is NOT proof: its sender stays unreleased).
    """
    sessions = sessions or _default_sessions()
    evidence = evidence if evidence is not None else default_evidence()
    now = now or utcnow()
    out = {"released": 0, "stopped": 0, "parked": 0}
    if started_park_after_sec is None:
        try:
            from .tasks import _RUN_LOCK_TTL_SEC

            started_park_after_sec = float(_RUN_LOCK_TTL_SEC)
        except Exception:
            started_park_after_sec = 3 * 3600.0
    try:
        with sessions() as db:
            ids = list(db.execute(select(PaidSubmitAttempt.id).where(
                ((PaidSubmitAttempt.status == PARKED) & PaidSubmitAttempt.sender_released_at.is_(None))
                | (PaidSubmitAttempt.status == SUPERSEDED)
                | ((PaidSubmitAttempt.status == STARTED)
                   & PaidSubmitAttempt.owner_kind.in_(VIDEO_IMAGE_OWNERS)
                   & (PaidSubmitAttempt.started_at < now - timedelta(seconds=started_park_after_sec)))
            )).scalars())
    except Exception:
        log.exception("paid_attempts.sweep: query failed")
        return out
    for aid in ids:
        try:
            with sessions() as db:
                peek = db.get(PaidSubmitAttempt, aid)
                if peek is None:
                    continue
                # Lock order owner -> attempt (as the job workers and reapers).
                owner_terminal = (lock_llm_owner(db, peek.owner_kind, peek.owner_id)
                                  if peek.owner_kind in LLM_OWNERS else None)
                row = db.get(PaidSubmitAttempt, aid, with_for_update=True, populate_existing=True)
                if row.status == STARTED and row.owner_kind in VIDEO_IMAGE_OWNERS:
                    if park_started(db, row, reason="started attempt outlived the run-lock TTL",
                                    evidence=evidence):
                        out["parked"] += 1
                elif row.status == PARKED and row.sender_released_at is None:
                    if _maybe_release(row, evidence, now):
                        row.updated_at = now
                        out["released"] += 1
                elif row.status == SUPERSEDED:
                    if _maybe_release(row, evidence, now):
                        row.status, row.finished_at, row.updated_at = STOPPED, now, now
                        out["stopped"] += 1
                        scope = llm_scope_for(row)
                        if scope and owner_terminal:
                            db.flush()
                            release_llm_scope_if_idle(db, row.owner_kind, row.owner_id, scope,
                                                      kind=row.owner_kind)
                db.commit()
        except Exception:
            log.exception("paid_attempts.sweep: attempt %s", aid)
    if any(out.values()):
        log.warning("paid_attempts.sweep: %s", out)
    return out


def caption_owner_id(caption_id: str, epoch: Optional[int]) -> str:
    return f"{caption_id}:{int(epoch or 0)}"


def lock_llm_owner(db: Session, owner_kind: str, owner_id: str) -> bool:
    """Lock an LLM attempt's owner row (FOR UPDATE) and return whether it is
    terminal (no new attempt can start). A vanished owner is terminal."""
    from .paid_attempts_models import (OWNER_LLM_CAPTION_ASR, OWNER_LLM_TRANSCRIBE,
                                       OWNER_LLM_TRANSLATE)

    if owner_kind in (OWNER_LLM_TRANSCRIBE, OWNER_LLM_TRANSLATE):
        from .localisation_jobs_models import (LIVE_JOB_STATUSES, LocalisationPromptJob,
                                               ProjectTranscription)

        model = ProjectTranscription if owner_kind == OWNER_LLM_TRANSCRIBE else LocalisationPromptJob
        job = db.execute(select(model).where(model.id == owner_id).with_for_update()
                         .execution_options(populate_existing=True)).scalar_one_or_none()
        return job is None or job.status not in LIVE_JOB_STATUSES
    if owner_kind == OWNER_LLM_CAPTION_ASR:
        from .models import RunCaption
        from .state_machine import CaptionStatus

        cid = owner_id.split(":", 1)[0]
        cap = db.execute(select(RunCaption).where(RunCaption.id == cid).with_for_update()
                         .execution_options(populate_existing=True)).scalar_one_or_none()
        return cap is None or cap.status in (CaptionStatus.done, CaptionStatus.failed)
    return True


def llm_scope_for(row: PaidSubmitAttempt) -> Optional[str]:
    from .paid_attempts_models import (OWNER_LLM_CAPTION_ASR, OWNER_LLM_TRANSCRIBE,
                                       OWNER_LLM_TRANSLATE)

    if row.owner_kind == OWNER_LLM_TRANSCRIBE:
        return spend_mod.transcribe_job_scope(row.owner_id)
    if row.owner_kind == OWNER_LLM_TRANSLATE:
        return spend_mod.localise_job_scope(row.owner_id)
    if row.owner_kind == OWNER_LLM_CAPTION_ASR:
        return spend_mod.caption_scope(row.owner_id.split(":", 1)[0])
    return None


def llm_reservation_credits(calls_per_attempt: int, chain_len: int) -> float:
    """Admission's worst case (§5.2): attempts x calls x chain x llm_credits()."""
    chain = max(1, min(int(chain_len), LLM_MAX_CHAIN))
    return LLM_MAX_ATTEMPTS * int(calls_per_attempt) * chain * spend_mod.llm_credits()


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------


def describe(row: PaidSubmitAttempt) -> dict:
    return {
        "id": row.id, "owner_kind": row.owner_kind, "owner_id": row.owner_id,
        "attempt_no": row.attempt_no, "status": row.status, "kind": row.kind,
        "started_at": row.started_at, "predecessor_task_id": row.predecessor_task_id,
        "provider_task_id": row.provider_task_id, "error": row.error,
        "sender_released": row.sender_released_at is not None,
        "sender_note": row.sender_note, "sender_host": row.sender_host,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse

    p = argparse.ArgumentParser(prog="python -m app.paid_attempts",
                                description="Paid submit attempts (AICP-351)")
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="open / active attempts")
    ls.add_argument("--all", action="store_true")
    for name in ("resubmit", "abandon"):
        sp = sub.add_parser(name)
        sp.add_argument("attempt_id")
        sp.add_argument("--actor", required=True)
        sp.add_argument("--note", required=True, help="what you saw on the kie.ai dashboard")
    rel = sub.add_parser("release-sender")
    rel.add_argument("attempt_id")
    rel.add_argument("--attest", required=True)
    args = p.parse_args(argv)
    from .db import get_session

    with get_session() as db:
        if args.cmd == "list":
            q = select(PaidSubmitAttempt).order_by(PaidSubmitAttempt.started_at)
            if not args.all:
                q = q.where(PaidSubmitAttempt.status.in_(OPEN_STATUSES + (SUPERSEDED,)))
            for r in db.execute(q).scalars():
                print(f"{r.id} {r.status:16} {r.owner_kind}:{r.owner_id} #{r.attempt_no} "
                      f"run={r.run_id or '-'} task={r.provider_task_id or '-'} "
                      f"pred={r.predecessor_task_id or '-'} released={r.sender_released_at is not None} "
                      f"host={r.sender_host or '-'} started={r.started_at:%Y-%m-%d %H:%M}")
            return 0
        try:
            if args.cmd == "release-sender":
                row = release_sender(db, args.attempt_id, attest=args.attest)
            elif args.cmd == "abandon":
                row = abandon(db, args.attempt_id, actor=args.actor, note=args.note)
            else:
                from .api_paid_attempts import relaunch

                row = db.get(PaidSubmitAttempt, args.attempt_id)
                if row is not None and row.run_id:
                    from .models import Run

                    db.get(Run, row.run_id, with_for_update=True)  # lock order Run -> attempt
                row = resubmit(db, args.attempt_id, actor=args.actor, note=args.note)
                relaunch(db, row)
            db.commit()
        except ActionRefused as exc:
            db.rollback()
            print(f"refused ({exc.status_code}): {exc.detail}")
            return 1
        except Exception as exc:  # an HTTPException from the owner's relaunch
            db.rollback()
            print(f"refused: {getattr(exc, 'detail', exc)}")
            return 1
        print(f"{row.id} is now {row.status}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
