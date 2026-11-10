"""Background translation job behind ``POST /projects/{pid}/localisation-prompt``.

AICP-338. The endpoint used to translate inside the request, behind nginx's
300 s ``proxy_read_timeout`` and a 90 s per-call translate timeout. A whole-video
localisation (project 46cc6443, 79 s, 23 lines into 7 segments) needs 180-280 s
for ONE translate call, so the UI and the API answered 502. The request now only
validates, reserves the spend estimate and enqueues; this module runs the
translation on the dedicated ``localise`` queue, and the UI polls
``GET /projects/{pid}/localisation-prompt/jobs/{job_id}`` for the answer.

What did NOT change:

* **No database writes.** The answer is the same
  :class:`~app.schemas_v2.LocalisationPromptResponse` as before. It is stored
  only as the RQ job's return value (Redis, ``LOCALISATION_PROMPT_RESULT_TTL_SEC``).
  The operator still reviews it before launching a run, and the run they submit
  stays the only record of what was used. A lost or expired job means "press
  Translate again", never a corrupted row.
* **The translation itself.** The job calls the same
  ``api_v2._compute_localisation_prompt`` code path the request used to run:
  frames, ``localisation.translate_lines``, ``build_prompt``, the per-segment
  split and the warnings. ``localisation.py`` is untouched.
* **Synchronous errors stay synchronous.** A bad language/hook (400), a missing
  project (404), a transcript that is not ready (409), a hook with no speech
  (422) and an over-budget day (429) are still answered by the POST itself.
  Only the model's own failure (the old 502) now arrives through the poll.

Inputs are SNAPSHOT at admission (the transcript lines, the resolved hook, the
hook's swap segments), so the job translates exactly what the operator asked
for, not whatever the project looks like when a worker picks it up. A
fingerprint of the project state is stored with the job, and the poll refuses
a result whose project has since changed (re-transcribed, transcript edited,
re-analyzed, segments edited): those per-segment prompts would be keyed to
segments that no longer exist.

Dedupe: identical requests (same project, language, mode, hook override and
project fingerprint) share one job while it is queued or running. The second
click reserves nothing and returns the first job's id. Admission is serialised
per dedupe key with a short Redis lock.

Interruption (a deploy recreates the worker mid-call): the RQ job stays
``started`` with a heartbeat that stops moving. The poll reports it as
``failed`` / interrupted once the heartbeat is older than
:data:`_HEARTBEAT_STALE_SEC`. It is NEVER replayed automatically: the upstream
call may already have been billed, and repeating it is the operator's call (the
same reasoning recovery.py applies to derived creatives). Nothing in the DB
points at the job, so there is nothing for recovery to reap.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from .config import settings

log = logging.getLogger(__name__)

# Redis key prefixes. The pointer maps a dedupe key to the job id currently
# serving it; the admit lock serialises two clicks on the same key.
POINTER_PREFIX = "reskin:localise-prompt:active:"
LOCK_PREFIX = "reskin:localise-prompt:admit:"
JOB_ID_PREFIX = "localise-prompt-"

# RQ (1.15) refreshes a started job's heartbeat every job_monitoring_interval
# (30 s) from the worker's parent process, even while the work-horse is blocked
# in the HTTP call. Five missed beats is a dead worker, not a slow model.
_HEARTBEAT_STALE_SEC = 150

# An IDLE RQ worker heartbeats only after each blocking dequeue (worker TTL
# 420 s), so liveness for admission uses that TTL plus a margin — the same
# rule as pipeline_provider._FAL_WORKER_FRESH_SEC.
_WORKER_FRESH_SEC = 420 + 60

# KieClient retries a transient failure (a timeout included) 4 times.
_KIE_TRANSPORT_ATTEMPTS = 4

# Lease of the per-key admission lock. Long enough to outlive a reserve that
# waits on the spend day lock; a request that dies holding it only delays the
# next identical click (409 "being started") until the lease runs out.
_ADMIT_LOCK_SEC = 120

# A pointer with no RQ job behind it is "being started" (not stale) for this
# long after it was published — covers an enqueue that stalls on Redis.
_PENDING_POINTER_GRACE_SEC = 300

_CAS_DELETE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)

_ACTIVE_RQ_STATUSES = {"queued", "started", "deferred", "scheduled"}


class AdmissionError(Exception):
    """The request cannot be admitted right now (mapped to an HTTP status)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


def call_timeout_sec() -> float:
    """Per-call translate timeout inside the job (never below the request's)."""
    return max(
        float(settings.LOCALISATION_TRANSLATE_JOB_CALL_TIMEOUT_SEC),
        float(settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC),
    )


def translate_budgets() -> dict:
    """Keyword overrides for localisation.translate_lines inside the job.

    The request-path defaults were sized for nginx's 300 s budget; a
    whole-video call of ~280 s would otherwise get neither the 1:1 re-ask nor
    the shorten pass. Passed as arguments, never by mutating localisation's
    module constants, so nothing outside the job changes."""
    return {
        "retry_budget_sec": float(settings.LOCALISATION_JOB_TRANSLATE_RETRY_BUDGET_SEC),
        "shorten_admit_sec": float(settings.LOCALISATION_JOB_SHORTEN_ADMIT_SEC),
        "shorten_timeout_sec": float(settings.LOCALISATION_JOB_SHORTEN_TIMEOUT_SEC),
    }


def _translate_model_count() -> int:
    models = [settings.LOCALISATION_TRANSLATE_MODEL]
    for m in settings.LOCALISATION_TRANSLATE_FALLBACK_MODELS.split(","):
        m = m.strip()
        if m and m not in models:
            models.append(m)
    return len(models)


def job_timeout_sec() -> int:
    """RQ job_timeout: the configured value, floored at the worst case of
    translate_lines under the job's budgets.

    Two paths bound it. The translate path: every model hop and the first
    attempt are admitted only within the retry budget, then the last call can
    burn all its transport attempts at :func:`call_timeout_sec`. The shorten
    path: it starts only within the shorten admission, then burns its own
    transport attempts at the shorten timeout. Plus a margin for frames,
    client backoff and the prompt build."""
    b = translate_budgets()
    translate_path = (
        _translate_model_count() * b["retry_budget_sec"]
        + _KIE_TRANSPORT_ATTEMPTS * call_timeout_sec()
    )
    shorten_path = (
        b["shorten_admit_sec"]
        + _KIE_TRANSPORT_ATTEMPTS * b["shorten_timeout_sec"]
    )
    floor = max(translate_path, shorten_path) + 300
    return int(max(settings.LOCALISATION_PROMPT_JOB_TIMEOUT_SEC, math.ceil(floor)))


def queue_ttl_sec() -> int:
    return max(60, int(settings.LOCALISATION_PROMPT_QUEUE_TTL_SEC))


def result_ttl_sec() -> int:
    return max(60, int(settings.LOCALISATION_PROMPT_RESULT_TTL_SEC))


@contextmanager
def _job_translate_timeout():
    """Raise localisation's per-call translate timeout for this job only.

    localisation.translate_lines reads ``settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC``
    at call time and takes no timeout argument (and localisation.py is owned
    elsewhere), so the job swaps the setting for its own duration. The run
    workers fork one work-horse per job (rq.Worker), so the change never leaks
    into another job or into the API process; the ``finally`` restores it
    anyway, for tests and for any non-forking worker class."""
    previous = settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC
    settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC = call_timeout_sec()
    try:
        yield
    finally:
        settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC = previous


# ---------------------------------------------------------------------------
# Fingerprint + dedupe key
# ---------------------------------------------------------------------------


def project_fingerprint(project, swap_segments: list) -> str:
    """Hash of everything the answer depends on that lives in the project.

    The transcript (lines, context, status), the stored hook intent, the
    localisation scope and probed duration (AICP-340), the source file, and
    the id + bounds of every swap SegmentDef. Any of those changing
    makes an answer computed before the change stale."""
    payload = {
        "transcript": project.transcript,
        "transcript_status": project.transcript_status,
        # AICP-340: in whole-video scope the window is the probed duration
        # (api_v2._intended_hook_sec) and hook_sec is not read, so editing it
        # must not make a running whole-video answer stale. The scope only
        # changes together with new SegmentDefs, which already change this
        # hash; it is included so the dependency is explicit.
        "hook_sec": (
            None
            if getattr(project, "localisation_scope", None) == "whole"
            and isinstance(project.duration_sec, (int, float))
            and project.duration_sec > 0
            # (without a probed duration the window falls back to hook_sec)
            else project.hook_sec
        ),
        "localisation_scope": getattr(project, "localisation_scope", None),
        "duration_sec": project.duration_sec,
        "source": project.source_local_path,
        "segments": [
            [
                str(s.id), round(float(s.start_sec), 4), round(float(s.end_sec), 4),
                round(float(getattr(s, "pre_roll_sec", None) or 0.0), 4),
                round(float(getattr(s, "post_roll_sec", None) or 0.0), 4),
            ]
            for s in swap_segments
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dedupe_key(
    project_id: str,
    language: str,
    swap_character: bool,
    hook_override: Optional[float],
    fingerprint: str,
    extra: Optional[dict] = None,
) -> str:
    """*extra*: any further optional request fields (snapshot["request_params"]),
    so two requests differing only in one of them never share a job."""
    raw = json.dumps(
        [project_id, language, bool(swap_character), hook_override, fingerprint,
         extra or {}],
        default=str, sort_keys=True,
    )
    return f"{project_id}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


# ---------------------------------------------------------------------------
# The job itself (runs in the localise-worker)
# ---------------------------------------------------------------------------


def run(snapshot: dict) -> dict:
    """Translate one admitted request. Returns a plain-JSON envelope.

    ``{"ok": True, "result": <LocalisationPromptResponse as JSON>}`` or
    ``{"ok": False, "status_code": 502, "detail": "..."}`` for a model failure
    (the old synchronous 502). Anything else raises, and RQ records the job as
    failed — the poll reports that as a crash without leaking the traceback."""
    from . import localisation
    from .api_v2 import _compute_localisation_prompt

    with _job_translate_timeout():
        try:
            response = _compute_localisation_prompt(
                snapshot, translate_budgets=translate_budgets()
            )
        except localisation.LocalisationError as exc:
            log.warning(
                "localisation prompt job for project %s failed: %s",
                snapshot.get("project_id"), exc,
            )
            return {"ok": False, "status_code": 502, "detail": str(exc)}
    return {"ok": True, "result": response.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Admission (API side)
# ---------------------------------------------------------------------------


def worker_alive(conn) -> bool:
    """True when some RQ worker serving the localise queue heartbeated recently."""
    from rq import Worker

    from .tasks import QUEUE_LOCALISE

    cutoff = datetime.utcnow() - timedelta(seconds=_WORKER_FRESH_SEC)
    for w in Worker.all(connection=conn):
        hb = getattr(w, "last_heartbeat", None)
        if QUEUE_LOCALISE in w.queue_names() and hb is not None and hb >= cutoff:
            return True
    return False


def _fetch(conn, job_id: str):
    from rq.exceptions import NoSuchJobError
    from rq.job import Job

    try:
        return Job.fetch(job_id, connection=conn)
    except NoSuchJobError:
        return None


def _decode(value) -> Optional[str]:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _pointer_ttl_full() -> int:
    return queue_ttl_sec() + job_timeout_sec() + 60


def _pointer_age_sec(conn, key: str) -> float:
    """Seconds since the pointer was published (from its remaining TTL)."""
    ttl = conn.ttl(POINTER_PREFIX + key)
    if ttl is None or ttl < 0:
        return float("inf")
    return max(0.0, _pointer_ttl_full() - float(ttl))


def _heartbeat_stale(job) -> bool:
    hb = job.last_heartbeat or job.started_at
    if hb is None:
        return False
    return datetime.utcnow() - hb > timedelta(seconds=_HEARTBEAT_STALE_SEC)


def _is_active(job) -> bool:
    status = _status_value(job)
    if status not in _ACTIVE_RQ_STATUSES:
        return False
    return not (status == "started" and _heartbeat_stale(job))


def _status_value(job) -> str:
    status = job.get_status(refresh=True)
    return getattr(status, "value", status) or ""


def admit(
    conn,
    *,
    key: str,
    reserve: Callable[[], None],
    enqueue: Callable[[str], None],
) -> tuple[str, bool]:
    """Return ``(job_id, created)`` for *key*.

    An active job already serving the key is returned with ``created=False``
    and nothing is reserved. Otherwise *reserve* (the spend reservation, which
    commits and may raise a 429) runs, then *enqueue(job_id)*, then the pointer
    is published. All of it under a per-key Redis lock, so two clicks cannot
    both reserve and enqueue.

    Crash boundaries, in order: dying after *reserve* leaves a reservation
    without a job (it only lowers today's remaining budget by the estimate, as
    an abandoned synchronous request did before); dying after *enqueue* but
    before the pointer leaves a running job a second click will not find, so it
    starts another (one extra translation, never a lost one). The pointer TTL
    covers the queue wait AND the run, and a stale pointer is harmless because
    the job behind it is re-checked."""
    from redis.exceptions import LockError

    lock = conn.lock(LOCK_PREFIX + key, timeout=_ADMIT_LOCK_SEC, blocking_timeout=10)
    if not lock.acquire():
        raise AdmissionError(
            409,
            "Another Translate with the same settings is being started right "
            "now — wait a moment and try again",
        )
    try:
        current = conn.get(POINTER_PREFIX + key)
        if current is not None:
            current_id = current.decode() if isinstance(current, bytes) else str(current)
            job = _fetch(conn, current_id)
            if job is not None and _is_active(job):
                return current_id, False
            if job is None and _pointer_age_sec(conn, key) < _PENDING_POINTER_GRACE_SEC:
                # Published moments ago by an admission whose enqueue has not
                # landed yet (pointer first, then enqueue): it is being started.
                raise AdmissionError(
                    409,
                    "Another Translate with the same settings is being started "
                    "right now — wait a moment and try again",
                )
        reserve()
        # The reserve can block on the spend day lock. If that outlived the
        # lock lease, another admission may be past its own pointer check, so
        # this one must not enqueue at all: it joins a job that is already
        # active, or refuses (the reservation just made is wasted — an
        # estimate of cents, never a second paid call).
        if not lock.owned():
            current = conn.get(POINTER_PREFIX + key)
            if current is not None:
                current_id = current.decode() if isinstance(current, bytes) else str(current)
                job = _fetch(conn, current_id)
                if job is not None and _is_active(job):
                    return current_id, False
            raise AdmissionError(
                409,
                "Starting the translation took too long and was abandoned — "
                "press Translate again",
            )
        # Pointer first, then the enqueue: a competing admission that sees
        # the pointer while the job is not yet in Redis finds no job behind
        # it and is still serialised by the lock we hold.
        job_id = JOB_ID_PREFIX + uuid.uuid4().hex
        conn.set(POINTER_PREFIX + key, job_id, ex=_pointer_ttl_full())
        try:
            enqueue(job_id)
        except Exception:
            # Compare-and-delete in one step: never withdraw a newer
            # admission's pointer.
            conn.eval(_CAS_DELETE_LUA, 1, POINTER_PREFIX + key, job_id)
            raise
        return job_id, True
    finally:
        try:
            lock.release()
        except LockError:
            # The lock lease expired under a slow reserve/enqueue; the
            # next admission simply re-checks the pointer.
            pass


# ---------------------------------------------------------------------------
# Poll (API side)
# ---------------------------------------------------------------------------


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def job_state(conn, job_id: str, project_id: str) -> Optional[dict]:
    """The poll's view of one job, or None when it is not this project's.

    Returns ``{"status": queued|running|done|failed|expired, "result"?,
    "error"?: {"status_code", "detail"}, "fingerprint", "enqueued_at",
    "started_at", "ended_at"}``. ``expired`` means Redis no longer has the job
    (result TTL passed, the queue-wait TTL passed, or Redis was recreated)."""
    if not job_id.startswith(JOB_ID_PREFIX):
        return None
    job = _fetch(conn, job_id)
    if job is None:
        return {"status": "expired"}
    meta = job.meta or {}
    if meta.get("project_id") != project_id:
        return None

    state: dict[str, Any] = {
        "fingerprint": meta.get("fingerprint"),
        "language": meta.get("language"),
        "swap_character": meta.get("swap_character"),
        "enqueued_at": _iso(job.enqueued_at),
        "started_at": _iso(job.started_at),
        "ended_at": _iso(job.ended_at),
    }
    status = _status_value(job)
    if status in ("queued", "deferred", "scheduled"):
        state["status"] = "queued"
    elif status == "started":
        if _heartbeat_stale(job):
            state["status"] = "failed"
            state["error"] = {
                "status_code": 503,
                "detail": (
                    "The translation was interrupted (the worker restarted, "
                    "e.g. during a deploy). Nothing was saved — press "
                    "Translate again."
                ),
            }
        else:
            state["status"] = "running"
    elif status == "finished":
        envelope = job.return_value()
        if isinstance(envelope, dict) and envelope.get("ok"):
            state["status"] = "done"
            state["result"] = envelope.get("result")
        elif isinstance(envelope, dict):
            state["status"] = "failed"
            state["error"] = {
                "status_code": int(envelope.get("status_code") or 502),
                "detail": str(envelope.get("detail") or "translation failed"),
            }
        else:
            state["status"] = "failed"
            state["error"] = {
                "status_code": 500,
                "detail": "The translation job returned no answer — press Translate again.",
            }
    else:
        # failed / stopped / canceled. exc_info is a traceback; only its
        # classification reaches the client.
        exc = job.exc_info or ""
        if "JobTimeoutException" in exc:
            detail = (
                f"The translation did not finish within {job.timeout}s and was "
                "stopped. Press Translate again, or write the prompt by hand."
            )
            code = 504
        elif "AbandonedJobError" in exc or "abandoned" in exc.lower():
            detail = (
                "The translation was interrupted (the worker restarted, e.g. "
                "during a deploy). Nothing was saved — press Translate again."
            )
            code = 503
        else:
            detail = (
                "The translation job crashed (see the localise-worker log). "
                "Press Translate again, or write the prompt by hand."
            )
            code = 500
        state["status"] = "failed"
        state["error"] = {"status_code": code, "detail": detail}
    return state
