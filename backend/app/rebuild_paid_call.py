"""The paid-call protocol of the rebuild feature (design §7.2) — shared primitives.

Every paid provider call of the rebuild feature (a GPT Image keyframe, a chunk's
video task, a Gemini call) is ONE row of a paid-call table
(``rebuild_models.PAID_CALL_MODELS``) and AT MOST ONE provider request. This
module owns the row state machine, so the storyboard/keyframe jobs (wave 2a)
and the images-only run pipeline (wave 2b) only supply what is specific to
them (inputs, the payload, the download/fit) and never re-implement fencing,
accounting or the reaper:

States::

    queued -> claimed -> sending -> submitted -> fetching -> succeeded
                            |            |            |
                            +-> failed   +-> failed   +-> failed (download gave up)
                            +-> unknown  (send outcome unknown; FINAL)

* ``claim`` (:func:`claim`): CAS ``queued -> claimed`` fenced by
  ``dispatch_gen``, with a fresh ``claim_token``. Free work (frames, uploads)
  happens while ``claimed``.
* ``send`` (:func:`begin_send` + commit, then :func:`send_window_open`, the
  provider call, then :func:`record_send_outcome`): one transaction CASes
  ``claimed -> sending`` fenced by ``claim_token`` + ``dispatch_gen``, records
  the request, the sender identity (container hostname, the worker process's
  boot id, the RQ job id), ``send_deadline`` and the spend draw (keyed
  ``rebuild:<row id>``, scope-locked) — so a paid call can never be missing from
  the books. After the commit the caller makes ONE call with
  ``KieClient.create_task_once`` / ``chat_completion_once`` (never the retrying
  methods) unless :func:`send_window_open` says too much time has passed.
* ``poll`` / ``fetch`` (:func:`record_poll`, :func:`complete_fetch`,
  :func:`fail_fetch`): idempotent; a failed download retries the download only
  (bounded), never the create.

Reaper (:func:`reap`, from ``tasks.run_reconcile_tick``): a non-terminal row
whose step jobs have not been live for :data:`NOT_LIVE_GRACE_SEC` is re-driven
— ``queued``/``claimed`` -> new ``dispatch_gen``, back to ``queued``, re-enqueue
``claim`` (nothing was sent: the send step needs the fenced CAS the bump
revokes); ``submitted``/``fetching`` -> new ``dispatch_gen``, re-enqueue
``poll``/``fetch``; ``sending`` -> ``unknown`` ONLY when the sender is PROVEN
terminated (:func:`sender_proof`): the boot id registered for its host changed
(the container restarted), or its RQ job was ended by its own worker
(``finished``; ``failed`` with rq 1.15's work-horse reap record or the job's
own traceback). An abandoned-job cleanup ("AbandonedJobError") proves nothing —
a frozen container looks exactly like that. A recreated container (new
hostname) is released only by the host attestation
(``scripts/rebuild_release_sender.sh`` -> ``python -m app.rebuild_paid_call
release-sender <row id> --attest <container id>:<state>``).

``failed`` and ``unknown`` are terminal and nothing ever re-sends a row: the
operator's regenerate / rerun admits a NEW row with its own reservation.

Step jobs: one RQ job per step on the ``process`` queue, id
``rb:<table>:<row id>:<dispatch_gen>:<step>[:<seq>]`` (poll/fetch carry a
sequence so a step can re-schedule itself); the generic entry point is
:func:`run_step`, which hands off to the table's handler registered in
:data:`STEP_HANDLERS` (wave 2 fills it). Liveness is "any job with the row's
``rb:<table>:<id>:<gen>:`` prefix is queued, scheduled or the current job of a
live worker".
"""

from __future__ import annotations

import logging
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import select, update

from . import spend as spend_mod
from .rebuild_models import (
    PAID_CALL_LIVE_STATUSES,
    PAID_CALL_MODELS,
    PC_CLAIMED,
    PC_FAILED,
    PC_FETCHING,
    PC_QUEUED,
    PC_SENDING,
    PC_SUBMITTED,
    PC_SUCCEEDED,
    PC_UNKNOWN,
    ProjectStoryboard,
    RebuildChunkAttempt,
    RebuildKeyframe,
    RebuildLlmCall,
)

log = logging.getLogger(__name__)

# --- Timing (design §7.2) -----------------------------------------------------

# A sender skips its call when more than SEND_WINDOW_SEC - SEND_WINDOW_MARGIN_SEC
# passed since its `sending` commit (cheap insurance, NOT a proof of anything).
SEND_WINDOW_SEC = 30
SEND_WINDOW_MARGIN_SEC = 5
SEND_HTTP_TIMEOUT_SEC = 60
# A row whose step jobs are not live for this long is re-driven by the reaper.
NOT_LIVE_GRACE_SEC = 120
# Poll intervals.
POLL_DELAY_IMAGE_SEC = 10
POLL_DELAY_VIDEO_SEC = 20
POLL_DELAY_LLM_SEC = 0
# Download attempts before a fetch gives up (the task id stays on the row).
FETCH_MAX_ATTEMPTS = 5
# Step jobs keep their RQ hash (and result) this long: the sender proof reads
# the send job's final status, which rq's default 500s result_ttl would lose.
STEP_JOB_TTL_SEC = 7 * 24 * 3600
STEP_JOB_TIMEOUT_SEC = {"claim": 900, "send": SEND_HTTP_TIMEOUT_SEC + 120,
                        "poll": 180, "fetch": 1800}
# Per-table overrides: a Gemini call carrying a video proxy answers in minutes,
# not seconds (rebuild_storyboard.LLM_TIMEOUT_SEC, at most 300 s), so its send
# job gets that much plus the same 120 s margin (wave 2a).
STEP_JOB_TIMEOUT_OVERRIDES = {"rebuild_llm_calls": {"send": 300 + 120}}


def step_job_timeout(table: str, step: str) -> int:
    return STEP_JOB_TIMEOUT_OVERRIDES.get(table, {}).get(step, STEP_JOB_TIMEOUT_SEC[step])

STEP_CLAIM = "claim"
STEP_SEND = "send"
STEP_POLL = "poll"
STEP_FETCH = "fetch"
STEPS = (STEP_CLAIM, STEP_SEND, STEP_POLL, STEP_FETCH)

# The step that drives a row out of each live status (what the reaper enqueues).
NEXT_STEP = {
    PC_QUEUED: STEP_CLAIM,
    PC_CLAIMED: STEP_SEND,
    PC_SUBMITTED: STEP_POLL,
    PC_FETCHING: STEP_FETCH,
}

BOOT_KEY_PREFIX = "rebuild:boot:"

UNKNOWN_MESSAGE = (
    "the provider may have billed this attempt (task id unknown); regenerating "
    "bills again"
)
WAITING_SENDER_MESSAGE = "waiting for the sending worker"
CANCELLED_MESSAGE = "cancelled before send"
NOT_SENT_MESSAGE = "not sent: the send window elapsed before the call"

# Draw kinds (kie_spend_draws.kind) per table.
DRAW_KIND = {
    "rebuild_keyframes": "rebuild keyframe",
    "rebuild_chunk_attempts": "rebuild chunk",
    "rebuild_llm_calls": "rebuild llm",
}

# Table -> dotted path of its step handler ``handler(row_id, gen, step, job_id)``.
# None = no driver deployed yet (wave 1): the reaper then only resolves
# `sending` rows (which needs no handler) and leaves the others alone.
STEP_HANDLERS: dict[str, Optional[str]] = {
    "rebuild_keyframes": "app.rebuild_keyframes.step",          # wave 2a
    # Wave 2b: the images-only run pipeline (claim -> send -> poll -> fetch+fit).
    "rebuild_chunk_attempts": "app.pipeline_rebuild.chunk_attempt_step",
    # Dispatches by kind (global / shot -> storyboard; character_check -> QA).
    "rebuild_llm_calls": "app.rebuild_llm_steps.handle",         # wave 2a
}

MODEL_BY_TABLE = {m.__tablename__: m for m in PAID_CALL_MODELS}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Step job ids
# ---------------------------------------------------------------------------


def step_job_prefix(table: str, row_id: str, gen: int) -> str:
    return f"rb:{table}:{row_id}:{int(gen)}:"


def step_job_id(table: str, row_id: str, gen: int, step: str, seq: Optional[str] = None) -> str:
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r}")
    jid = f"{step_job_prefix(table, row_id, gen)}{step}"
    return f"{jid}:{seq}" if seq else jid


def parse_step_job_id(job_id: str) -> Optional[tuple[str, str, int, str]]:
    """``(table, row_id, gen, step)`` of a step job id, else None."""
    parts = (job_id or "").split(":")
    if len(parts) < 5 or parts[0] != "rb":
        return None
    try:
        return parts[1], parts[2], int(parts[3]), parts[4]
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Sender identity (boot ids)
# ---------------------------------------------------------------------------

_BOOT_ID: Optional[str] = None


@dataclass(frozen=True)
class SenderIdentity:
    host: str
    boot: str
    job: str


class SenderUnavailable(RuntimeError):
    """This process has no registered boot id: it must not send."""


def current_hostname() -> str:
    """The container hostname (= its container id under Docker)."""
    return socket.gethostname()


def register_boot(conn, *, hostname: Optional[str] = None) -> str:
    """Register a NEW boot id for this host (call once, at worker start-up,
    BEFORE any work-horse is forked so every horse inherits it).

    ``rebuild:boot:<hostname> = <uuid>``, no TTL: a frozen container keeps its
    old value and is never mistaken for dead; a restarted one (same hostname)
    writes a new value, which proves every process of the old boot is gone.
    """
    global _BOOT_ID
    boot = uuid.uuid4().hex
    conn.set(BOOT_KEY_PREFIX + (hostname or current_hostname()), boot)
    _BOOT_ID = boot
    log.info("rebuild: registered boot id %s for host %s", boot, hostname or current_hostname())
    return boot


def current_boot_id() -> Optional[str]:
    return _BOOT_ID


def sender_identity(job_id: Optional[str]) -> SenderIdentity:
    """The identity the send step records. Raises :class:`SenderUnavailable`
    in a process without a registered boot id (never registers lazily: a
    work-horse writing a new boot id would make every in-flight sender of its
    host look dead)."""
    boot = current_boot_id()
    if not boot:
        raise SenderUnavailable("no rebuild boot id registered in this process")
    if not job_id:
        raise SenderUnavailable("the send step must run as an RQ job")
    return SenderIdentity(host=current_hostname(), boot=boot, job=job_id)


# ---------------------------------------------------------------------------
# CAS core
# ---------------------------------------------------------------------------


def cas(
    db,
    model,
    row_id: str,
    *,
    expect_status,
    values: dict,
    expect_gen: Optional[int] = None,
    expect_token: Optional[str] = None,
    now: Optional[datetime] = None,
) -> bool:
    """Guarded UPDATE of one paid-call row. True when exactly one row matched.

    *expect_status* is a status or a tuple of statuses; *expect_gen* /
    *expect_token* add the dispatch-generation / claim-token fences.
    ``updated_at`` is always set. Never commits.
    """
    statuses = (expect_status,) if isinstance(expect_status, str) else tuple(expect_status)
    conds = [model.id == row_id, model.status.in_(statuses)]
    if expect_gen is not None:
        conds.append(model.dispatch_gen == int(expect_gen))
    if expect_token is not None:
        conds.append(model.claim_token == expect_token)
    vals = dict(values)
    vals.setdefault("updated_at", now or _utcnow())
    res = db.execute(
        update(model).where(*conds).values(**vals)
        .execution_options(synchronize_session=False)
    )
    return res.rowcount == 1


def _get(db, model, row_id: str):
    return db.execute(
        select(model).where(model.id == row_id).execution_options(populate_existing=True)
    ).scalar_one_or_none()


def draw_kind(model) -> str:
    return DRAW_KIND[model.__tablename__]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def claim(db, model, row_id: str, gen: int, *, now: Optional[datetime] = None) -> Optional[str]:
    """``queued -> claimed`` fenced by *gen*; returns the new claim token or
    None (a stale or duplicate delivery claims nothing)."""
    token = uuid.uuid4().hex
    ok = cas(db, model, row_id, expect_status=PC_QUEUED, expect_gen=gen,
             values={"status": PC_CLAIMED, "claim_token": token, "not_live_since": None},
             now=now)
    return token if ok else None


def begin_send(
    db,
    model,
    row_id: str,
    *,
    gen: int,
    token: str,
    request: dict,
    sender: SenderIdentity,
    scope_key: str,
    now: Optional[datetime] = None,
) -> bool:
    """``claimed -> sending`` fenced by *token* + *gen*, recording the request,
    the sender identity, ``send_deadline`` and the spend draw at the row's
    ``estimate_credits`` — all in the caller's transaction, which the caller
    commits BEFORE the provider call. Lock order: the paid-call row (this
    UPDATE) -> the scope lock -> the day lock (``spend.draw_in_session``).
    Returns False when fenced out (nothing written)."""
    now = now or _utcnow()
    ok = cas(db, model, row_id, expect_status=PC_CLAIMED, expect_gen=gen, expect_token=token,
             values={
                 "status": PC_SENDING, "request": request,
                 "sender_host": sender.host, "sender_boot": sender.boot,
                 "sender_job": sender.job, "sender_note": None,
                 "send_deadline": now + timedelta(seconds=SEND_WINDOW_SEC),
                 "not_live_since": None, "error": None,
             }, now=now)
    if not ok:
        return False
    row = _get(db, model, row_id)
    spend_mod.draw_in_session(
        db, scope_key, spend_mod.rebuild_call_key(row_id), float(row.estimate_credits or 0.0),
        kind=draw_kind(model), provider="kie",
        now=now.replace(tzinfo=timezone.utc), scope_lock=True,
    )
    return True


def send_window_open(committed_at_monotonic: float, *, clock: Callable[[], float] = time.monotonic) -> bool:
    """Whether the sender may still call after its ``sending`` commit at
    *committed_at_monotonic* (``time.monotonic()``). Insurance only."""
    return clock() - committed_at_monotonic <= SEND_WINDOW_SEC - SEND_WINDOW_MARGIN_SEC


def record_send_outcome(
    db,
    model,
    row_id: str,
    *,
    gen: int,
    token: str,
    outcome,
    scope_key: str,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Record what the ONE provider call did (*outcome*: a
    ``kie_client.SendOutcome``, or None when the call was skipped because
    :func:`send_window_open` said no). Returns the new status, or None when
    fenced out. In the caller's transaction (commit after).

    * accepted -> ``submitted`` with the task id; an LLM call (no task id)
      -> ``succeeded`` with the response body, its draw settled to the
      body's ``credits_consumed`` (AICP-382; left at the estimate when the
      body carries no usable figure);
    * refused -> ``failed``, draw settled to 0;
    * unknown -> ``unknown``, draw stays at the estimate;
    * skipped -> ``failed`` ("not sent"), draw settled to 0 (nothing left the
      process; design §7.2 says ``unknown`` — ``failed`` is the truthful state
      and frees the operator from a "may have billed" warning).
    """
    from .kie_client import OUTCOME_ACCEPTED, OUTCOME_REFUSED, chat_credits_consumed

    now = now or _utcnow()
    call_key = spend_mod.rebuild_call_key(row_id)
    fence = dict(expect_status=PC_SENDING, expect_gen=gen, expect_token=token, now=now)
    is_llm = model is RebuildLlmCall
    if outcome is None:
        if not cas(db, model, row_id, values={
            "status": PC_FAILED, "error": NOT_SENT_MESSAGE, "finished_at": now,
            "claim_token": None, "cost_credits": 0.0,
        }, **fence):
            return None
        spend_mod.settle_in_session(db, scope_key, call_key, 0.0, kind=draw_kind(model),
                                    now=now.replace(tzinfo=timezone.utc))
        return PC_FAILED
    if outcome.kind == OUTCOME_ACCEPTED:
        if is_llm:
            # AICP-382: settle to kie.ai's own ``credits_consumed`` (in the
            # same transaction as the terminal state); with no usable figure
            # the draw stays at the estimate, as before.
            cost = chat_credits_consumed(outcome.body)
            ok = cas(db, model, row_id, values={
                "status": PC_SUCCEEDED, "response": outcome.body, "finished_at": now,
                "claim_token": None, "cost_credits": cost,
            }, **fence)
            if not ok:
                return None
            if cost is not None:
                spend_mod.settle_in_session(db, scope_key, call_key, cost,
                                            kind=draw_kind(model),
                                            now=now.replace(tzinfo=timezone.utc))
            else:
                log.warning("%s %s: accepted LLM response carries no usable "
                            "credits_consumed; its draw stays at the estimate",
                            model.__tablename__, row_id)
            return PC_SUCCEEDED
        if not outcome.task_id:
            raise ValueError("an accepted createTask outcome must carry a task id")
        ok = cas(db, model, row_id, values={"status": PC_SUBMITTED, "task_id": outcome.task_id},
                 **fence)
        return PC_SUBMITTED if ok else None
    if outcome.kind == OUTCOME_REFUSED:
        if not cas(db, model, row_id, values={
            "status": PC_FAILED, "error": (outcome.error or "refused")[:2000],
            "finished_at": now, "claim_token": None, "cost_credits": 0.0,
        }, **fence):
            return None
        spend_mod.settle_in_session(db, scope_key, call_key, 0.0, kind=draw_kind(model),
                                    now=now.replace(tzinfo=timezone.utc))
        return PC_FAILED
    ok = cas(db, model, row_id, values={
        "status": PC_UNKNOWN,
        "error": f"{UNKNOWN_MESSAGE} ({outcome.error or 'no complete response'})"[:2000],
        "finished_at": now, "claim_token": None,
    }, **fence)
    return PC_UNKNOWN if ok else None


def record_poll(
    db,
    model,
    row_id: str,
    *,
    gen: int,
    state: str,
    scope_key: str,
    credits_consumed: Optional[float] = None,
    error: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Record one ``recordInfo`` observation of a ``submitted`` row.

    *state*: ``"success"`` -> ``fetching``; ``"fail"`` -> ``failed`` with the
    draw settled to *credits_consumed* (left at the estimate when the provider
    reports none);
    anything else -> no change (re-schedule the poll). Fenced by *gen*.
    """
    now = now or _utcnow()
    if state == "success":
        vals = {"status": PC_FETCHING}
        if credits_consumed is not None:
            vals["cost_credits"] = float(credits_consumed)
        ok = cas(db, model, row_id, expect_status=PC_SUBMITTED, expect_gen=gen, values=vals, now=now)
        return PC_FETCHING if ok else None
    if state == "fail":
        # Settled to the provider's reported cost; with no report the draw
        # stays at the estimate (conservative: a failed task may still have
        # been billed — Terra code gate round 2).
        cost = None if credits_consumed is None else float(credits_consumed)
        ok = cas(db, model, row_id, expect_status=PC_SUBMITTED, expect_gen=gen, values={
            "status": PC_FAILED, "error": (error or "provider task failed")[:2000],
            "finished_at": now, "cost_credits": cost, "claim_token": None,
        }, now=now)
        if not ok:
            return None
        if cost is not None:
            spend_mod.settle_in_session(db, scope_key, spend_mod.rebuild_call_key(row_id), cost,
                                        kind=draw_kind(model),
                                        now=now.replace(tzinfo=timezone.utc))
        return PC_FAILED
    return PC_SUBMITTED


def complete_fetch(
    db,
    model,
    row_id: str,
    *,
    gen: int,
    values: dict,
    scope_key: str,
    now: Optional[datetime] = None,
) -> bool:
    """``fetching -> succeeded`` with the result *values* (paths, sha, size,
    ...), settling the draw to the row's ``cost_credits`` when known (else the
    estimate stays). Fenced by *gen*. The caller adds its own same-transaction
    effects (auto-select a keyframe; select an attempt + complete its segment)."""
    now = now or _utcnow()
    vals = dict(values)
    vals.update({"status": PC_SUCCEEDED, "finished_at": now, "claim_token": None})
    if not cas(db, model, row_id, expect_status=PC_FETCHING, expect_gen=gen, values=vals, now=now):
        return False
    row = _get(db, model, row_id)
    if row.cost_credits is not None:
        spend_mod.settle_in_session(db, scope_key, spend_mod.rebuild_call_key(row_id),
                                    float(row.cost_credits), kind=draw_kind(model),
                                    now=now.replace(tzinfo=timezone.utc))
    return True


def fail_fetch(
    db,
    model,
    row_id: str,
    *,
    gen: int,
    error: str,
    scope_key: str,
    max_attempts: int = FETCH_MAX_ATTEMPTS,
    final: bool = False,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Count one failed download (or fit). Returns ``"retry"`` (re-schedule
    the fetch — never a new create), ``"failed"`` once *max_attempts* is
    reached or *final* is set (e.g. "result too short"), or None when fenced
    out. A failed fetch of a task the provider completed was billed: the draw
    is settled to ``cost_credits`` when known, else it stays at the estimate.
    """
    now = now or _utcnow()
    row = _get(db, model, row_id)
    if row is None or row.status != PC_FETCHING or int(row.dispatch_gen) != int(gen):
        return None
    attempts = int(row.fetch_attempts or 0) + 1
    if final or attempts >= max_attempts:
        msg = error if final else (
            f"result could not be downloaded; task {row.task_id} ({error})"
        )
        if not cas(db, model, row_id, expect_status=PC_FETCHING, expect_gen=gen, values={
            "status": PC_FAILED, "error": msg[:2000], "fetch_attempts": attempts,
            "finished_at": now, "claim_token": None,
        }, now=now):
            return None
        if row.cost_credits is not None:
            spend_mod.settle_in_session(db, scope_key, spend_mod.rebuild_call_key(row_id),
                                        float(row.cost_credits), kind=draw_kind(model),
                                        now=now.replace(tzinfo=timezone.utc))
        return "failed"
    if not cas(db, model, row_id, expect_status=PC_FETCHING, expect_gen=gen,
               values={"fetch_attempts": attempts, "error": error[:2000]}, now=now):
        return None
    return "retry"


def cancel_unsent(db, model, *conditions, now: Optional[datetime] = None) -> int:
    """CAS every ``queued``/``claimed`` row matching *conditions* to
    ``failed`` ("cancelled before send"). Nothing was sent from those states
    and the cancel revokes the fenced ``claimed -> sending`` CAS (the token is
    cleared). Used when a storyboard is interrupted (design §5.2), in the same
    transaction. Rows already ``sending`` keep their protocol life."""
    now = now or _utcnow()
    res = db.execute(
        update(model).where(model.status.in_((PC_QUEUED, PC_CLAIMED)), *conditions)
        .values(status=PC_FAILED, error=CANCELLED_MESSAGE, claim_token=None,
                finished_at=now, updated_at=now)
        .execution_options(synchronize_session=False)
    )
    return int(res.rowcount or 0)


# ---------------------------------------------------------------------------
# Sender proof (reaper, design §7.2)
# ---------------------------------------------------------------------------

_HORSE_REAPED_MARKERS = (
    "Work-horse terminated unexpectedly",      # rq 1.15 Worker.monitor_work_horse
    "Job stopped by user, work-horse terminated",
)
_NO_PROOF_MARKERS = ("AbandonedJobError", "Moved to FailedJobRegistry")


def sender_proof(
    *,
    sender_boot: Optional[str],
    registered_boot: Optional[str],
    job_status: Optional[str],
    job_exc: Optional[str],
) -> Optional[str]:
    """Why the sender of a ``sending`` row is PROVEN terminated, else None.

    * the boot id registered for its host differs from the one it sent under
      (the container restarted: every process of the old boot is gone);
    * its RQ job ``finished``;
    * its RQ job ``failed`` with the record its OWN worker writes after
      ``waitpid`` (rq 1.15: "Work-horse terminated unexpectedly; waitpid
      returned ...", or a deliberate stop), or with the job's own traceback
      (the job function exited by raising — a death-penalty timeout included).

    A failure written by ``StartedJobRegistry.cleanup`` ("due to
    AbandonedJobError") proves nothing and is refused; so is a missing boot
    key, a missing job hash, and every other state.
    """
    if sender_boot and registered_boot and registered_boot != sender_boot:
        return f"sender container restarted (boot {sender_boot} -> {registered_boot})"
    status = (job_status or "").lower()
    if status == "finished":
        return "sender job finished"
    if status in ("failed", "stopped"):
        exc = job_exc or ""
        if any(m in exc for m in _NO_PROOF_MARKERS):
            return None
        if any(m in exc for m in _HORSE_REAPED_MARKERS):
            return "sender work-horse reaped by its own worker"
        if exc.lstrip().startswith("Traceback"):
            return "sender job raised"
    return None


def _job_evidence(conn, job_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """``(status, exc_string)`` of an RQ job, (None, None) when unknown."""
    if not job_id:
        return None, None
    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return None, None
    try:
        status = job.get_status()
        status = getattr(status, "value", status)
    except Exception:
        status = None
    texts = []
    try:
        result = job.latest_result()
        if result is not None and getattr(result, "exc_string", None):
            texts.append(result.exc_string)
    except Exception:
        pass
    if getattr(job, "_exc_info", None):
        texts.append(str(job._exc_info))
    return (str(status) if status is not None else None), "\n".join(texts) or None


def default_evidence(conn) -> Callable:
    """The reaper's evidence reader over Redis (boot key + the send job)."""

    def evidence(row) -> Optional[str]:
        registered = conn.get(BOOT_KEY_PREFIX + row.sender_host) if row.sender_host else None
        if isinstance(registered, bytes):
            registered = registered.decode()
        status, exc = _job_evidence(conn, row.sender_job)
        return sender_proof(sender_boot=row.sender_boot, registered_boot=registered,
                            job_status=status, job_exc=exc)

    return evidence


# ---------------------------------------------------------------------------
# Liveness + enqueue
# ---------------------------------------------------------------------------


def step_is_live(conn, queue: str, prefix: str) -> bool:
    """Any job whose id starts with *prefix* is queued on *queue*, in its
    scheduled registry, or the current job of a worker with a live heartbeat."""
    from rq.registry import ScheduledJobRegistry

    from .tasks import _decode, _get_queue, _live_worker_current_jobs

    try:
        if any(_decode(j).startswith(prefix) for j in _get_queue(queue).get_job_ids()):
            return True
    except Exception:
        log.warning("step_is_live: queue read failed — treating as live", exc_info=True)
        return True
    try:
        if any(_decode(j).startswith(prefix)
               for j in ScheduledJobRegistry(queue, connection=conn).get_job_ids()):
            return True
    except Exception:
        log.warning("step_is_live: scheduled registry read failed — treating as live",
                    exc_info=True)
        return True
    return any(j.startswith(prefix) for j in _live_worker_current_jobs(conn))


def enqueue_step(table: str, row_id: str, gen: int, step: str, *, delay_sec: float = 0,
                 seq: Optional[str] = None) -> str:
    """Push one step job (after the commit that made it due). Best effort: a
    lost enqueue is re-sent by the reaper. Returns the job id."""
    from .tasks import QUEUE_PROCESS, _get_queue

    if step in (STEP_POLL, STEP_FETCH) and seq is None:
        seq = uuid.uuid4().hex[:8]
    job_id = step_job_id(table, row_id, gen, step, seq)
    kwargs = dict(job_id=job_id, job_timeout=step_job_timeout(table, step),
                  result_ttl=STEP_JOB_TTL_SEC, failure_ttl=STEP_JOB_TTL_SEC)
    q = _get_queue(QUEUE_PROCESS)
    args = (table, row_id, int(gen), step)
    if delay_sec and delay_sec > 0:
        q.enqueue_in(timedelta(seconds=float(delay_sec)), "app.rebuild_paid_call.run_step",
                     *args, **kwargs)
    else:
        q.enqueue("app.rebuild_paid_call.run_step", *args, **kwargs)
    return job_id


def run_step(table: str, row_id: str, gen: int, step: str):
    """RQ entry point of every step job: hands off to the table's handler
    (``STEP_HANDLERS[table]``, a dotted path ``module.func`` called as
    ``func(row_id, gen, step, job_id)``). With no handler deployed the job
    logs and does nothing (the row stays for the reaper)."""
    from rq import get_current_job

    path = STEP_HANDLERS.get(table)
    job = get_current_job()
    job_id = job.id if job is not None else None
    if not path:
        log.error("rebuild step %s for %s/%s g%s: no handler deployed", step, table, row_id, gen)
        return None
    import importlib

    mod_name, func_name = path.rsplit(".", 1)
    func = getattr(importlib.import_module(mod_name), func_name)
    return func(row_id, int(gen), step, job_id)


# ---------------------------------------------------------------------------
# Reaper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReaperAction:
    kind: str                # "none" | "observe" | "clear" | "requeue" | "unknown" | "wait"
    step: Optional[str] = None
    reason: str = ""


def reaper_decision(
    *,
    status: str,
    step_live: bool,
    not_live_since: Optional[datetime],
    now: datetime,
    proof: Optional[str],
    has_handler: bool,
    grace_sec: float = NOT_LIVE_GRACE_SEC,
) -> ReaperAction:
    """Pure decision for one live row (design §7.2)."""
    if status not in PAID_CALL_LIVE_STATUSES:
        return ReaperAction("none")
    if step_live:
        return ReaperAction("clear" if not_live_since is not None else "none")
    if not_live_since is None:
        return ReaperAction("observe")
    if (now - not_live_since).total_seconds() < grace_sec:
        return ReaperAction("none")
    if status == PC_SENDING:
        if proof:
            return ReaperAction("unknown", reason=proof)
        return ReaperAction("wait", reason=WAITING_SENDER_MESSAGE)
    if not has_handler:
        return ReaperAction("none", reason="no step handler deployed")
    if status in (PC_QUEUED, PC_CLAIMED):
        return ReaperAction("requeue", step=STEP_CLAIM)
    return ReaperAction("requeue", step=NEXT_STEP[status])


def reap(
    *,
    session_factory: Optional[Callable] = None,
    conn=None,
    is_live: Optional[Callable[[str, str], bool]] = None,
    evidence: Optional[Callable] = None,
    enqueue: Optional[Callable] = None,
    now: Optional[datetime] = None,
    models=PAID_CALL_MODELS,
) -> dict:
    """One reaper pass over every live paid-call row. Returns counters.

    Every transition is a CAS on the row's current (``status``,
    ``dispatch_gen``), so a slow step job finds its fence gone and stops.
    """
    from .tasks import QUEUE_PROCESS

    if session_factory is None:
        from .db import SessionLocal as session_factory  # noqa: N813
    if conn is None and (is_live is None or evidence is None):
        from .tasks import _get_redis

        conn = _get_redis()
    is_live = is_live or (lambda queue, prefix: step_is_live(conn, queue, prefix))
    evidence = evidence or default_evidence(conn)
    enqueue = enqueue or (lambda table, row_id, gen, step: enqueue_step(table, row_id, gen, step))
    now = now or _utcnow()
    out = {"observed": 0, "requeued": 0, "unknown": 0, "waiting": 0}
    for model in models:
        table = model.__tablename__
        has_handler = bool(STEP_HANDLERS.get(table))
        db = session_factory()
        try:
            ids = [i for (i,) in db.execute(
                select(model.id).where(model.status.in_(PAID_CALL_LIVE_STATUSES)))]
        finally:
            db.close()
        for row_id in ids:
            db = session_factory()
            to_enqueue = None
            try:
                row = db.execute(select(model).where(model.id == row_id)
                                 .with_for_update(skip_locked=True)).scalar_one_or_none()
                if row is None or row.status not in PAID_CALL_LIVE_STATUSES:
                    db.rollback()
                    continue
                gen = int(row.dispatch_gen)
                live = is_live(QUEUE_PROCESS, step_job_prefix(table, row.id, gen))
                proof = None
                if (not live and row.status == PC_SENDING and row.not_live_since is not None
                        and (now - row.not_live_since).total_seconds() >= NOT_LIVE_GRACE_SEC):
                    proof = evidence(row)
                action = reaper_decision(status=row.status, step_live=live,
                                         not_live_since=row.not_live_since, now=now,
                                         proof=proof, has_handler=has_handler)
                fence = dict(expect_status=row.status, expect_gen=gen, now=now)
                if action.kind == "clear":
                    cas(db, model, row.id, values={"not_live_since": None}, **fence)
                elif action.kind == "observe":
                    cas(db, model, row.id, values={"not_live_since": now}, **fence)
                    out["observed"] += 1
                elif action.kind == "unknown":
                    if cas(db, model, row.id, values={
                        "status": PC_UNKNOWN, "claim_token": None, "finished_at": now,
                        "sender_note": f"released by the reaper: {action.reason}",
                        "error": UNKNOWN_MESSAGE,
                    }, **fence):
                        out["unknown"] += 1
                        log.warning("%s %s: sending -> unknown (%s)", table, row.id, action.reason)
                elif action.kind == "wait":
                    out["waiting"] += 1
                elif action.kind == "requeue":
                    new_gen = gen + 1
                    vals = {"dispatch_gen": new_gen, "not_live_since": None}
                    if row.status in (PC_QUEUED, PC_CLAIMED):
                        vals.update(status=PC_QUEUED, claim_token=None)
                    if cas(db, model, row.id, values=vals, **fence):
                        to_enqueue = (table, row.id, new_gen, action.step)
                        out["requeued"] += 1
                db.commit()
            except Exception:
                db.rollback()
                log.exception("rebuild reaper: %s %s failed — continuing", table, row_id)
                to_enqueue = None
            finally:
                db.close()
            if to_enqueue is not None:
                try:
                    enqueue(*to_enqueue)
                except Exception:
                    log.exception("rebuild reaper: enqueue of %s failed — next tick retries",
                                  to_enqueue)
    return out


# ---------------------------------------------------------------------------
# Host attestation (recreated container) — design §7.2
# ---------------------------------------------------------------------------


def release_sender(db, row_id: str, *, attest: str, now: Optional[datetime] = None) -> str:
    """Move a ``sending`` row to ``unknown`` on a HOST attestation that its
    sender container no longer runs (``<container id>:<state>``, produced by
    ``scripts/rebuild_release_sender.sh`` from ``docker ps -a --no-trunc``).

    The attested id must name the row's ``sender_host`` (the container id, a
    prefix match either way) and the state must be ``absent`` or ``exited``.
    Returns the table name. Raises ``ValueError`` otherwise. Commits nothing.
    """
    container, _, state = (attest or "").partition(":")
    if state not in ("absent", "exited") or not container:
        raise ValueError("attestation must be '<container id>:absent|exited'")
    now = now or _utcnow()
    for model in PAID_CALL_MODELS:
        row = db.execute(select(model).where(model.id == row_id).with_for_update()) \
            .scalar_one_or_none()
        if row is None:
            continue
        if row.status != PC_SENDING:
            raise ValueError(f"{model.__tablename__} {row_id} is {row.status}, not sending")
        host = row.sender_host or ""
        if not host or not (container.startswith(host) or host.startswith(container)):
            raise ValueError(
                f"attested container {container!r} is not the sender host {host!r}"
            )
        if not cas(db, model, row_id, expect_status=PC_SENDING,
                   expect_gen=int(row.dispatch_gen), values={
                       "status": PC_UNKNOWN, "claim_token": None, "finished_at": now,
                       "sender_note": f"released by host attestation {attest} at {now.isoformat()}",
                       "error": UNKNOWN_MESSAGE,
                   }, now=now):
            raise ValueError(f"{row_id}: concurrent change, try again")
        return model.__tablename__
    raise ValueError(f"no rebuild paid-call row {row_id}")


# ---------------------------------------------------------------------------
# Live-row guards (deletion, re-analysis, admissions — design §5.1.7, §7.4)
# ---------------------------------------------------------------------------


def project_live_rows(db, project_id: str) -> dict:
    """Counts of the project's live rebuild rows (storyboards, keyframes, LLM
    calls, chunk attempts of its runs). All zero for a non-rebuild project."""
    from sqlalchemy import func

    from .models import Run

    def count(stmt) -> int:
        return int(db.execute(stmt).scalar() or 0)

    return {
        "storyboards": count(select(func.count()).select_from(ProjectStoryboard).where(
            ProjectStoryboard.project_id == project_id,
            ProjectStoryboard.status.in_(("queued", "running")))),
        "keyframes": count(select(func.count()).select_from(RebuildKeyframe).where(
            RebuildKeyframe.project_id == project_id,
            RebuildKeyframe.status.in_(PAID_CALL_LIVE_STATUSES))),
        "llm_calls": count(select(func.count()).select_from(RebuildLlmCall).where(
            RebuildLlmCall.project_id == project_id,
            RebuildLlmCall.status.in_(PAID_CALL_LIVE_STATUSES))),
        "attempts": count(select(func.count()).select_from(RebuildChunkAttempt).join(
            Run, Run.id == RebuildChunkAttempt.run_id).where(
            Run.project_id == project_id,
            RebuildChunkAttempt.status.in_(PAID_CALL_LIVE_STATUSES))),
    }


def run_live_attempts(db, run_id: str) -> int:
    """Live chunk attempts of the run PLUS live character-check calls of its
    attempts (a check of a succeeded attempt runs after the run may be done;
    deleting the run would cascade away a call that can still bill)."""
    from sqlalchemy import func

    attempts = int(db.execute(select(func.count()).select_from(RebuildChunkAttempt).where(
        RebuildChunkAttempt.run_id == run_id,
        RebuildChunkAttempt.status.in_(PAID_CALL_LIVE_STATUSES))).scalar() or 0)
    checks = int(db.execute(select(func.count()).select_from(RebuildLlmCall).join(
        RebuildChunkAttempt, RebuildChunkAttempt.id == RebuildLlmCall.attempt_id).where(
        RebuildChunkAttempt.run_id == run_id,
        RebuildLlmCall.status.in_(PAID_CALL_LIVE_STATUSES))).scalar() or 0)
    return attempts + checks


def live_rows_message(counts: dict) -> Optional[str]:
    """A 409 detail naming the live rows, or None when there are none."""
    parts = [f"{n} {k.replace('_', ' ')}" for k, n in counts.items() if n]
    if not parts:
        return None
    return ("rebuild work is still in flight (" + ", ".join(parts) + "); wait for it to "
            "finish — a row that is 'sending' waits for its worker")


# ---------------------------------------------------------------------------
# CLI: python -m app.rebuild_paid_call release-sender <row id> --attest <id>:<state>
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m app.rebuild_paid_call")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rel = sub.add_parser("release-sender", help="sending -> unknown on a host attestation")
    rel.add_argument("row_id")
    rel.add_argument("--attest", required=True)
    args = parser.parse_args(argv)
    from .db import SessionLocal

    db = SessionLocal()
    try:
        table = release_sender(db, args.row_id, attest=args.attest)
        db.commit()
    except ValueError as exc:
        db.rollback()
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()
    print(f"released {table} {args.row_id}: sending -> unknown")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
