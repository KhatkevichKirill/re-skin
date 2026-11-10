"""
tasks.py — RQ job enqueue helpers and worker-callable functions.

Enqueue helpers (v2)
--------------------
enqueue_analyze_project(project_id)   Push run_analyze_project onto the "analyze" RQ queue.
enqueue_process_run(run_id)           Push run_submit_run onto the "process" RQ queue.
enqueue_transcribe_project(pid)       Push run_transcribe_project onto the "process" queue.

RQ-callable targets (v2)
------------------------
run_analyze_project(project_id)    Call pipeline_v2.analyze_project with real clients,
                                   then queue transcription for localisation projects.
run_transcribe_project(project_id) Localisation side channel: transcribe the source.
run_submit_run(run_id)             Submit phase: cut + upload + create AI tasks.
run_poll_run(run_id)               Poll phase: one round of status checks; re-enqueues
                                   itself if tasks are still pending.
run_finalize_run(run_id)           Finalize phase: stitch + deliver.

Overlay, merge, tail-edit, caption, segment-image processing
--------------------------------------------------------------
run_process_overlay(overlay_job_id)      PNG-burn overlay job.
run_process_merge(merge_id)              PIP-insert merge.
run_process_tail_edit(edit_id)           Tail edit ("Change ending").
run_process_caption(caption_id)          Burn captions (transcribe + render + encode).
run_process_segment_image(job_id)        Per-segment GPT Image 2 regeneration — submit phase.
run_process_segment_image_poll(job_id)   Per-segment GPT Image 2 regeneration — poll phase.
run_process_range_attempt(attempt_id)        Run-local segment range regeneration — submit phase.
run_process_range_attempt_poll(attempt_id)   Run-local segment range regeneration — poll phase
                                              (also stitches/delivers/promotes on success).

Final Creatives sheet export
----------------------------
run_export_run_to_sheet(run_id)           Append run to Final Creatives tab.
run_export_overlay_to_sheet(overlay_id)  Append overlay to Final Creatives tab.
run_export_tail_edit_to_sheet(edit_id)   Append tail edit to Final Creatives tab.
run_export_caption_to_sheet(caption_id)  Append caption to Final Creatives tab.

Queue layout
------------
"overlay"  — PNG-burn jobs (fast, interactive, highest priority).
"analyze"  — InsightFace segmentation jobs (minutes; CPU-heavy).
"process"  — AI swap runs (submit/poll/finalize; long but mostly idle-waiting).

Workers listen on [overlay, analyze, process] in that priority order, so short
interactive jobs are always picked up first.

"fal"      — Fal provider tasks (submit/poll/fetch/cancel/close), served only by
             the fal-worker container; see app/pipeline_provider.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid as _uuid_mod
from datetime import timedelta
from typing import Optional

from redis import Redis
from rq import Callback, Queue, Retry

from . import languages
from .config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Queue names
# ---------------------------------------------------------------------------

QUEUE_OVERLAY = "overlay"
QUEUE_ANALYZE = "analyze"
QUEUE_PROCESS = "process"
# Fal provider tasks (AICP-330 B1): served ONLY by the dedicated fal-worker
# container (worker/fal_worker.py), never by the run workers, so a Fal result
# never waits behind a long run job and Fal work never starves run work.
QUEUE_FAL = "fal"
# Localisation translate jobs (AICP-338): served ONLY by the dedicated
# localise-worker container (worker/localise_worker.py). A whole-video
# translation holds its worker for minutes on one LLM call; on the shared run
# workers two of them would stall every run's submit/poll/stitch, and a
# hook-only translation would queue behind a stitch.
QUEUE_LOCALISE = "localise"
# Free autopilot compute jobs (AICP-363, docs/autopilot-design.md §5.5): QA
# scans and similar CPU work a flow step asks for. Served by the run workers
# (after analyze, before process) — they have the memory and the pre-warmed
# InsightFace; the autopilot ticker itself never runs heavy work.
QUEUE_AUTOPILOT_COMPUTE = "autopilot_compute"
# AICP-389: the brand check (local word ASR), served ONLY by the brand-asr
# container (worker/brand_asr_worker.py), one job at a time on the host.
QUEUE_BRAND_ASR = "brand_asr"
AUTOPILOT_COMPUTE_JOB_TIMEOUT = int(os.getenv("AUTOPILOT_COMPUTE_JOB_TIMEOUT", "3600"))

# Periodic orphaned-run reconciliation ("reaper"). Boot-time reconciliation can
# miss a run orphaned moments earlier — a just-killed worker still has a fresh
# heartbeat, so its abandoned jobs briefly look live (see app/recovery.py). This
# tick re-runs reconciliation after those signals expire, so recovery is
# self-healing. See enqueue_reconcile_tick / run_reconcile_tick below.
RECONCILE_TICK_INTERVAL_SEC = int(os.getenv("RECONCILE_TICK_INTERVAL_SEC", "180"))
# Lease key bounding the reaper to a single chain across N workers / restarts.
_RECONCILE_CHAIN_KEY = "reskin:reconcile-tick:chain"

# Periodic run-automation tick — launches scheduled batches from the persona
# spreadsheet and fires batch-completion webhooks (see app/automation.py).
# Same single-chain lease pattern as the reconcile reaper above.
AUTOMATION_TICK_INTERVAL_SEC = int(os.getenv("AUTOMATION_TICK_INTERVAL_SEC", "300"))
_AUTOMATION_CHAIN_KEY = "reskin:automation-tick:chain"

# Periodic disk cleanup of regenerable media intermediates (see
# app/disk_cleanup.py). Same single-chain lease pattern as the reaper.
DISK_CLEANUP_TICK_INTERVAL_SEC = int(os.getenv("DISK_CLEANUP_TICK_INTERVAL_SEC", "21600"))
_DISK_CLEANUP_CHAIN_KEY = "reskin:disk-cleanup-tick:chain"

# ---------------------------------------------------------------------------
# Module-level Redis connection + Queue cache (Problem 4 fix)
# ---------------------------------------------------------------------------

_redis_conn: Redis | None = None
_queues: dict[str, Queue] = {}


def _get_redis() -> Redis:
    """Return a cached Redis connection (created once at module level)."""
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = Redis.from_url(settings.REDIS_URL)
    return _redis_conn


def _get_queue(name: str = QUEUE_PROCESS) -> Queue:
    """Return a cached Queue object for the given queue name."""
    global _queues
    if name not in _queues:
        _queues[name] = Queue(name, connection=_get_redis())
    return _queues[name]


# ---------------------------------------------------------------------------
# Job timeouts
# ---------------------------------------------------------------------------

# RQ's default job_timeout is 180s, far too short: analyze runs InsightFace over
# the whole video, and process runs several Seedance jobs sequentially (each can
# take minutes). Without these, long jobs are killed mid-flight with
# "Task exceeded maximum timeout value (180 seconds)" while the Seedance task
# keeps running on kie.ai. Override generously (seconds); tunable via env.
ANALYZE_JOB_TIMEOUT = int(os.getenv("ANALYZE_JOB_TIMEOUT", "1800"))      # 30 min

# Submit phase: cut + upload + create AI tasks (no sleep; bounded by network I/O).
SUBMIT_JOB_TIMEOUT = int(os.getenv("SUBMIT_JOB_TIMEOUT", "1800"))        # 30 min

# Poll phase: a single round of status checks (no sleep; re-enqueues itself).
# Each round does at most N get_task calls + N download_result calls.
POLL_JOB_TIMEOUT = int(os.getenv("POLL_JOB_TIMEOUT", "900"))             # 15 min

# Finalize phase: stitch (ffmpeg) + Drive upload.
FINALIZE_JOB_TIMEOUT = int(os.getenv("FINALIZE_JOB_TIMEOUT", "3600"))   # 1 hour

# Legacy combined timeout (kept so _RUN_LOCK_TTL_SEC is large enough to
# cover the entire lifecycle). Also used by recovery.py/api_v2.py if they
# reference PROCESS_JOB_TIMEOUT directly.
PROCESS_JOB_TIMEOUT = int(os.getenv("PROCESS_JOB_TIMEOUT", "10800"))    # 3 hours

OVERLAY_JOB_TIMEOUT = int(os.getenv("OVERLAY_JOB_TIMEOUT", "3600"))     # 1 hour

# PIP-insert merge: one ffmpeg overlay pass (conform + optional alphamerge +
# overlay, base audio stream-copied) over the target video, plus a Drive
# upload — the same shape as a run's finalize phase (stitch + deliver), so it
# gets the same generous timeout.
MERGE_JOB_TIMEOUT = int(os.getenv("MERGE_JOB_TIMEOUT", "3600"))         # 1 hour

# Tail edit ("Change ending"): a cut_clip pass + a stitch pass (re-encode),
# plus a Drive upload — the same shape as a merge, so it gets the same
# generous timeout.
TAIL_EDIT_JOB_TIMEOUT = int(os.getenv("TAIL_EDIT_JOB_TIMEOUT", "3600"))  # 1 hour

# Localisation transcription: one upload of the source to kie.ai plus one
# video-model request. The request itself is already bounded by
# settings.LOCALISATION_TIMEOUT_SEC (600s by default); this leaves room for the
# upload of a large source on top of it without ever being the thing that kills
# a call the provider is still answering.
TRANSCRIBE_JOB_TIMEOUT = int(os.getenv("TRANSCRIBE_JOB_TIMEOUT", "1800"))  # 30 min

# Burned captions: a transcription of the DELIVERED result (the same kie.ai
# round trip TRANSCRIBE_JOB_TIMEOUT covers) followed by a full-length libass
# burn — a re-encode of the whole video, not a trim — and a Drive upload. It is
# the only job that pays for a model call AND an encode, so it gets its own
# knob rather than borrowing the tail edit's: the two move for different
# reasons, and tuning one should never silently retune the other.
CAPTION_JOB_TIMEOUT = int(os.getenv("CAPTION_JOB_TIMEOUT", "5400"))  # 90 min

# Audio patch ("Fix audio from another run"): one ffmpeg pass that STREAM-COPIES
# the picture and re-encodes only the soundtrack — seconds, not minutes, even on
# a 40s 1080x1920 delivery — plus a Drive content replacement of the same order
# as any other delivery. Sized off the upload, not the encode, which is why it
# lands near the tail edit's rather than the caption's.
AUDIO_PATCH_JOB_TIMEOUT = int(os.getenv("AUDIO_PATCH_JOB_TIMEOUT", "3600"))  # 1 hour

# ---------------------------------------------------------------------------
# Per-run processing lock
# ---------------------------------------------------------------------------

# Per-run processing lock: prevents two RQ jobs from processing the same run
# at the same time (e.g. a duplicate enqueue from a race in recovery or retry).
# TTL is slightly larger than the max job timeout so the lock auto-expires if the
# holder crashes without releasing it.  Token-based release prevents a timed-out
# lock from being deleted by a different job.
#
# Env: PROCESS_JOB_TIMEOUT controls the overall lifecycle timeout; the lock TTL
# is larger by 5 minutes to give finalize a chance to run. Default 10800s (3h).
_RUN_LOCK_TTL_SEC = PROCESS_JOB_TIMEOUT + 300


def _kcap_begin() -> None:
    """Knowledge capture (AICP-299 §1.3): record this job's deadline first, so
    a capture call never risks the RQ alarm. A no-op with capture off."""
    try:
        from .knowledge import capture

        capture.begin_job()
    except Exception:
        pass


def run_lock_key(run_id: str) -> str:
    """The per-run processing lock key.

    A function rather than an inline f-string in each caller because
    ``app.recovery`` also has to clear this key when it reclaims an orphaned run
    — and a recovery that builds the key itself would silently stop working the
    day this format changed.
    """
    return f"reskin:run:lock:{run_id}"

# Lua script for atomic token-checked lock release.
# Compares the stored token with ARGV[1]; only DELetes if they match.
# Prevents a lock held by job A from being deleted by a later job B
# after job A's lock TTL expired and B acquired the lock.
_RELEASE_LOCK_LUA = """\
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""

# Token-checked TTL refresh. The poll chain can outlive the initial lock TTL
# (each task timeout is 2h and failed tasks are resubmitted), so every poll
# round re-arms the TTL. Only the owner may refresh; a lost lock means another
# job (e.g. recovery) has taken over the run and this chain must stop.
#
# AICP-360: a successful refresh also re-stamps the lock's GENERATION
# (KEYS[2] = run_lock_gen_key) with a fresh random nonce (ARGV[3]). Every job
# that inherits a lock (poll, finalize, the submit/preprocess handoffs)
# refreshes BEFORE it does any work, so "token and generation unchanged" proves
# no job has started working under this lock since they were read. The
# recovery reaper relies on exactly that to take over a stale lock with a
# compare-and-set (claim_stale_run_lock) instead of deleting it blindly. A
# random nonce rather than a counter: a counter key that expired and was
# re-INCRed could come back to the value the reaper read (ABA); a fresh uuid
# cannot.
_REFRESH_LOCK_LUA = """\
if redis.call("GET", KEYS[1]) == ARGV[1] then
    redis.call("EXPIRE", KEYS[1], ARGV[2])
    redis.call("SET", KEYS[2], ARGV[3], "EX", ARGV[2])
    return 1
else
    return 0
end
"""

# AICP-360: the reaper's takeover of a lock it judged stale. Succeeds only if
# the lock token AND its generation are still exactly what the reaper read
# BEFORE its liveness scan, then installs the reaper's own token as the new
# owner. Absence is explicit: ARGV[1]/ARGV[3] are "1" when the key was present
# (with its value in ARGV[2]/ARGV[4]) and "0" when it was absent. See
# recovery._recover_orphaned_run for why that ordering makes the takeover safe
# against a job that was requeued (and started) after the reaper's snapshot.
_CLAIM_STALE_LOCK_LUA = """\
local function same(key, present, value)
    local cur = redis.call("GET", key)
    if present == "0" then
        return cur == false
    end
    return cur == value
end
if not same(KEYS[1], ARGV[1], ARGV[2]) then
    return 0
end
if not same(KEYS[2], ARGV[3], ARGV[4]) then
    return 0
end
redis.call("SET", KEYS[1], ARGV[5], "EX", ARGV[6])
redis.call("SET", KEYS[2], ARGV[7], "EX", ARGV[6])
return 1
"""


def run_lock_gen_key(lock_key: str) -> str:
    """The generation nonce that belongs to a per-run lock key (AICP-360).

    Derived from the LOCK KEY (not the run id) because poll/finalize jobs carry
    ``lock_key`` in their kwargs, and a job enqueued before a deploy must keep
    re-stamping the same generation the new reaper reads.
    """
    return f"{lock_key}:gen"


def _as_str(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def read_run_lock_states(
    conn: Redis, run_ids: list[str]
) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """``{run_id: (lock token, generation)}`` in one atomic MGET (None = absent).

    Raises on a Redis error: the caller must treat an unreadable lock as "do not
    touch this run", never as "no lock".
    """
    keys: list[str] = []
    for rid in run_ids:
        lk = run_lock_key(rid)
        keys += [lk, run_lock_gen_key(lk)]
    if not keys:
        return {}
    values = list(conn.mget(keys))
    if len(values) != len(keys):
        raise RuntimeError(f"MGET returned {len(values)} value(s) for {len(keys)} key(s)")
    return {
        rid: (_as_str(values[2 * i]), _as_str(values[2 * i + 1]))
        for i, rid in enumerate(run_ids)
    }


def claim_stale_run_lock(
    conn: Redis, run_id: str, observed: tuple[Optional[str], Optional[str]]
) -> Optional[str]:
    """Take over *run_id*'s lock if it is still exactly *observed* (AICP-360).

    Returns the new owner token on success, None if the lock or its generation
    moved since *observed* was read (some job acquired or refreshed it — leave
    the run alone). The claim gets a full lifecycle TTL because it is handed to
    the recovered job as ITS lock (``enqueue_process_run(..., lock_token=)``),
    never released and re-acquired. Raises on a Redis error.
    """
    lock_key = run_lock_key(run_id)
    token = _uuid_mod.uuid4().hex
    observed_token, observed_gen = observed
    claimed = conn.eval(
        _CLAIM_STALE_LOCK_LUA, 2, lock_key, run_lock_gen_key(lock_key),
        "0" if observed_token is None else "1", observed_token or "",
        "0" if observed_gen is None else "1", observed_gen or "",
        token, _RUN_LOCK_TTL_SEC, _uuid_mod.uuid4().hex,
    )
    return token if claimed == 1 else None

# ---------------------------------------------------------------------------
# Per-run poll-state Redis key
# ---------------------------------------------------------------------------

# Redis hash where pending-task state is stored between poll invocations.
# Key: reskin:run:poll:{run_id}   (JSON; TTL = _POLL_STATE_TTL_SEC)
# Value: JSON-encoded dict mapping task_id → {rs_id, index, deadline, attempt, recipe}
#
# Using wall-clock epoch for deadlines (not monotonic) so state survives across
# process boundaries.
#
# TTL is 2× RUN_SKIP_TIMEOUT (default 2h each → 4h) plus buffer.
_POLL_STATE_TTL_SEC = int(os.getenv("RUN_SKIP_TIMEOUT_SEC", "7200")) * 2 + 3600


def _poll_state_key(run_id: str) -> str:
    return f"reskin:run:poll:{run_id}"


def _load_poll_state(conn: Redis, run_id: str) -> dict:
    """Load the pending-task map from Redis.  Returns {} if missing."""
    raw = conn.get(_poll_state_key(run_id))
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _save_poll_state(conn: Redis, run_id: str, state: dict) -> None:
    """Persist the pending-task map to Redis with TTL."""
    conn.setex(_poll_state_key(run_id), _POLL_STATE_TTL_SEC, json.dumps(state))


def _delete_poll_state(conn: Redis, run_id: str) -> None:
    """Remove the pending-task map from Redis (called on completion/failure)."""
    conn.delete(_poll_state_key(run_id))


# ---------------------------------------------------------------------------
# Poll interval
# ---------------------------------------------------------------------------

RUN_POLL_INTERVAL_SEC = float(os.getenv("RUN_POLL_INTERVAL_SEC", "15"))

# ---------------------------------------------------------------------------
# RQ-callable targets — analyze
# ---------------------------------------------------------------------------


def run_analyze_project(project_id: str, analysis_token: Optional[str] = None) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_v2.analyze_project`.

    Imports pipeline_v2 lazily so the worker process doesn't need GPU/insightface
    at import time (only at execution time).

    For a ``localisation`` project this also queues the transcription job — but
    only once analysis has RETURNED, and outside its transaction. That ordering
    is the whole point (docs/localisation.md §8): transcription talks to a
    third-party LLM, and a Gemini outage inside analyze_project's try block
    would mark the project failed and destroy a perfectly good segmentation.
    Here the worst case is a project that is ready with no transcript, which the
    operator works around by pasting a script by hand.
    """
    from .pipeline_v2 import analyze_project

    log.info("run_analyze_project: project_id=%s", project_id)
    if analysis_token:
        analyze_project(project_id, analysis_token=analysis_token)
    else:
        analyze_project(project_id)
    _enqueue_transcribe_after_analyze(project_id)


def _enqueue_transcribe_after_analyze(project_id: str) -> None:
    """Queue transcription for a just-analysed ``localisation`` project.

    Best-effort by construction: any failure here (DB hiccup, Redis down) is
    logged and swallowed, because analysis has already succeeded by this point
    and raising would mark the whole analyze job failed over a side channel.
    """
    from . import spend as spend_mod
    from .db import get_session
    from .models import VideoProject
    from .project_types import LOCALISATION

    try:
        with get_session() as session:
            project = session.get(VideoProject, project_id)
            if project is None:
                log.warning(
                    "_enqueue_transcribe_after_analyze: project %s not found",
                    project_id,
                )
                return
            if project.project_type != LOCALISATION:
                return
            # AICP-363 §6.5: an assistant project's autopilot launches its
            # transcription inside the approved budget; nothing paid starts
            # here on its own.
            if getattr(project, "mode", "manual") == "assistant":
                log.info(
                    "_enqueue_transcribe_after_analyze: project %s is in assistant "
                    "mode — transcription is left to its autopilot", project_id,
                )
                return
            # AICP-365: the durable admission (app/localisation_jobs.py) writes
            # "pending", the job row and the reservation in one commit; the RQ
            # job is only a doorbell the reaper re-sends if it is lost. A
            # transcription already live (the operator pressed the button)
            # wins: nothing is queued twice.
            from fastapi import HTTPException

            from . import localisation_jobs

            try:
                job = localisation_jobs.admit_transcription(session, project, origin="analysis")
            except HTTPException as exc:
                session.rollback()
                if exc.status_code == 429:
                    project = session.get(VideoProject, project_id)
                    project.transcript_status = "failed"
                    project.transcript_error = str(exc.detail)
                    session.commit()
                log.warning(
                    "_enqueue_transcribe_after_analyze: project %s — %s",
                    project_id, exc.detail,
                )
                return
            session.commit()

        localisation_jobs.enqueue(job)
    except Exception:
        log.exception(
            "_enqueue_transcribe_after_analyze: could not queue transcription "
            "for project %s — analysis is unaffected", project_id,
        )


def run_transcribe_project(project_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_v2.transcribe_project`.

    Lazily imported like the other pipeline wrappers. No per-project lock: the
    long-running single-entity jobs on this queue (merge, tail edit, overlay)
    don't take one either, and transcription has nothing to corrupt if it runs
    twice — each attempt overwrites the same three columns with its own result.
    The cost of a duplicate is one wasted LLM call, not a double AI submission,
    which is what the per-run lock exists to prevent.
    """
    from .pipeline_v2 import transcribe_project

    log.info("run_transcribe_project: project_id=%s", project_id)
    transcribe_project(project_id)


def run_localisation_prompt(snapshot: dict) -> dict:
    """RQ-callable wrapper for :func:`localisation_prompt_job.run` (AICP-338).

    The return value IS the result: the poll endpoint reads it back with
    ``job.return_value()``. No lock, no DB write — see the module docstring."""
    from .localisation_prompt_job import run

    log.info(
        "run_localisation_prompt: project_id=%s language=%s",
        snapshot.get("project_id"), snapshot.get("target_language"),
    )
    return run(snapshot)


def enqueue_localisation_prompt(
    job_id: str, snapshot: dict, *, meta: dict
) -> None:
    """Push :func:`run_localisation_prompt` onto the "localise" queue.

    *job_id* is chosen by the caller (localisation_prompt_job.admit) so the
    dedupe pointer can name it. ``ttl`` bounds the queue wait; the results
    live ``LOCALISATION_PROMPT_RESULT_TTL_SEC``."""
    from . import localisation_prompt_job as lpj

    q = _get_queue(QUEUE_LOCALISE)
    job = q.enqueue(
        "app.tasks.run_localisation_prompt",
        snapshot,
        job_id=job_id,
        job_timeout=lpj.job_timeout_sec(),
        ttl=lpj.queue_ttl_sec(),
        result_ttl=lpj.result_ttl_sec(),
        failure_ttl=lpj.result_ttl_sec(),
        meta=meta,
    )
    log.info(
        "Enqueued localisation_prompt for project_id=%s -> rq_job=%s",
        meta.get("project_id"), job.id,
    )


# ---------------------------------------------------------------------------
# RQ-callable targets — process (split submit/poll/finalize)
# ---------------------------------------------------------------------------


def run_submit_run(run_id: str, *, lock_key: str | None = None, lock_token: str | None = None) -> None:
    """
    RQ-callable submit-phase wrapper.

    Acquires the per-run Redis lock, calls pipeline_v2.submit_run (which
    submits AI tasks and returns a pending-task map), persists that map to Redis,
    then enqueues run_poll_run to begin polling.

    The lock is held for the full lifecycle (submit → poll → finalize).  It is
    released only by run_finalize_run or on unrecoverable error.

    When invoked from the preprocess chain (``lock_key``/``lock_token`` already
    provided), the existing lock is refreshed rather than re-acquired, so the
    chain holds the lock continuously from preprocessing through finalize.
    """
    _kcap_begin()
    from .gdrive_client import GDriveClient
    from .pipeline_v2 import submit_run

    conn = _get_redis()

    if lock_key is None or lock_token is None:
        lock_key = run_lock_key(run_id)
        lock_token = _uuid_mod.uuid4().hex
        acquired = conn.set(lock_key, lock_token, nx=True, ex=_RUN_LOCK_TTL_SEC)
        if not acquired:
            log.warning(
                "run_submit_run: run %s is already being processed (lock held) - "
                "skipping this duplicate job to avoid double AI submission",
                run_id,
            )
            return
    else:
        # Handed off from the preprocess chain — refresh the lock we already
        # own. If it was lost (TTL expired / recovery re-took it), abort.
        if not _refresh_lock(conn, lock_key, lock_token):
            log.warning(
                "run_submit_run: run %s lock lost during preprocess handoff "
                "— aborting", run_id,
            )
            return

    log.info("run_submit_run: run_id=%s (lock acquired)", run_id)
    if _is_images_only_run(run_id):
        # Images-only (rebuild) run — docs/rebuild-from-reference-design.md
        # §6.4: admit chunk attempts, then hand the SAME lock to the observer
        # chain. The v2v code below never sees such a run.
        _submit_images_only_run(conn, run_id, lock_key, lock_token)
        return
    try:
        pending = submit_run(run_id, gdrive=GDriveClient())

        if pending is None:
            # submit_run returns None when the run is already done or in a
            # terminal state — nothing more to do.
            log.info("run_submit_run: run_id=%s nothing to do (run terminal)", run_id)
            _release_lock(conn, lock_key, lock_token)
            return

        if pending == {}:
            # No tasks were submitted (all segments already completed) →
            # go straight to finalize.
            log.info("run_submit_run: run_id=%s no pending tasks — going to finalize", run_id)
            _delete_poll_state(conn, run_id)
            _enqueue_finalize(run_id, lock_key=lock_key, lock_token=lock_token)
            return

        # Persist the pending-task map so poll invocations can access it.
        _save_poll_state(conn, run_id, pending)

        # Enqueue the first poll round (runs after RUN_POLL_INTERVAL_SEC delay).
        _enqueue_poll(
            run_id,
            lock_key=lock_key,
            lock_token=lock_token,
            delay_sec=RUN_POLL_INTERVAL_SEC,
        )

    except Exception:
        # On unrecoverable submit error, release the lock so a retry can re-acquire it.
        _release_lock(conn, lock_key, lock_token)
        raise


# ---------------------------------------------------------------------------
# RQ-callable targets — preprocess (GPT Image 2, optional first phase)
# ---------------------------------------------------------------------------


def run_preprocess_refs(
    run_id: str, *, lock_key: str | None = None, lock_token: str | None = None
) -> None:
    """
    RQ-callable preprocess-phase wrapper (GPT Image 2 image-to-image).

    Acquires the per-run Redis lock, calls
    ``pipeline_v2.preprocess_reference_images`` to create one kie.ai task per
    ref-with-a-prompt, then either:
      - enqueues ``run_preprocess_poll`` (tasks still pending), or
      - transitions ``preprocessing → queued`` and enqueues ``run_submit_run``
        (no tasks / all pass-through), handing the lock off to submit.

    The run is expected to already be in ``preprocessing`` (the API transitions
    ``created → preprocessing`` before enqueueing this job).

    ``lock_key``/``lock_token`` given (recovery handoff, AICP-360): the lock is
    already ours — refresh it instead of acquiring, and abort if it was lost.
    """
    _kcap_begin()
    from .gdrive_client import GDriveClient
    from .pipeline_v2 import preprocess_reference_images
    from .state_machine import RunStatus, transition
    from .db import get_session
    from .models import Run

    conn = _get_redis()

    if lock_key is None or lock_token is None:
        lock_key = run_lock_key(run_id)
        lock_token = _uuid_mod.uuid4().hex
        acquired = conn.set(lock_key, lock_token, nx=True, ex=_RUN_LOCK_TTL_SEC)
        if not acquired:
            log.warning(
                "run_preprocess_refs: run %s is already being processed (lock held) "
                "— skipping duplicate", run_id,
            )
            return
    elif not _refresh_lock(conn, lock_key, lock_token):
        log.warning(
            "run_preprocess_refs: run %s lock lost before the recovery handoff "
            "ran — aborting", run_id,
        )
        return

    log.info("run_preprocess_refs: run_id=%s (lock acquired)", run_id)
    try:
        pending = preprocess_reference_images(run_id, gdrive=GDriveClient())

        if not pending:
            # No tasks to poll (preprocessing disabled, no refs, all
            # pass-through, or no trims). R2-new: PAUSE for human review instead
            # of auto-starting Seedance — the operator still confirms before
            # any AI video credit is spent. Auto-approve copies skip that pause
            # and go straight to submit (see _finish_preprocessing).
            log.info(
                "run_preprocess_refs: run_id=%s no pending preprocess tasks — "
                "finishing preprocess phase", run_id,
            )
            _finish_preprocessing(
                run_id, conn=conn, lock_key=lock_key, lock_token=lock_token
            )
            return

        _save_preprocess_poll_state(conn, run_id, pending)
        _enqueue_preprocess_poll(
            run_id, lock_key=lock_key, lock_token=lock_token,
            delay_sec=RUN_POLL_INTERVAL_SEC,
        )

    except Exception as exc:
        message = f"GPT Image preprocessing failed to start: {exc}"
        # Same rule as the poll phase (_should_park_failed_preprocess): don't
        # take an operator-driven run down when it already has a reference set.
        # A single-image regen that cannot even create its task leaves an empty
        # slot the operator can retry from the review card; failing here would
        # strand the run — `failed → queued` is the only edge out of `failed`,
        # and it submits.
        #
        # AICP-360: the run-state write happens BEFORE the lock is released, so
        # whoever takes the lock next never races this job's last write.
        try:
            if _should_park_failed_preprocess(run_id):
                log.warning(
                    "run_preprocess_refs: run_id=%s failed to start but the run has a "
                    "reference set — parking at preprocess_review", run_id,
                )
                _transition_preprocessing_to_review(run_id, error_message=message)
            else:
                _fail_preprocessing_run(run_id, error_message=message)
        finally:
            _release_lock(conn, lock_key, lock_token)
        raise


def run_preprocess_poll(run_id: str, *, lock_key: str, lock_token: str) -> None:
    """
    RQ-callable preprocess poll-phase wrapper.

    Performs ONE round of GPT Image 2 task checks, downloads results into
    ``Run.preprocessed_reference_urls``. On task failure (R2) the run is
    transitioned to ``failed`` with ``error_message`` set (no fallback to
    originals, no submit). Otherwise re-enqueues itself (tasks still
    pending) or transitions ``preprocessing → queued`` and enqueues
    ``run_submit_run`` (handing the lock off to submit).
    """
    _kcap_begin()
    from .pipeline_v2 import poll_preprocess_once

    conn = _get_redis()

    if not _refresh_lock(conn, lock_key, lock_token):
        log.warning(
            "run_preprocess_poll: run %s lock lost — aborting poll chain", run_id
        )
        return

    pending = _load_preprocess_poll_state(conn, run_id)
    if not pending:
        # No poll state — all tasks already terminal in a prior round. R2-new:
        # PAUSE for human review instead of auto-starting Seedance (auto-approve
        # copies submit straight away — see _finish_preprocessing).
        log.info(
            "run_preprocess_poll: run_id=%s no poll state — finishing preprocess "
            "phase", run_id,
        )
        _finish_preprocessing(
            run_id, conn=conn, lock_key=lock_key, lock_token=lock_token
        )
        return

    log.info("run_preprocess_poll: run_id=%s pending_tasks=%d", run_id, len(pending))

    try:
        result = poll_preprocess_once(run_id, pending=pending)
    except Exception as exc:
        # AICP-360: fail the run while still holding the lock, then release.
        try:
            _fail_preprocessing_run(
                run_id,
                error_message=f"GPT Image preprocessing failed: {exc}",
            )
        finally:
            _release_lock(conn, lock_key, lock_token)
        raise

    failed_list = result.get("failed") or []
    still_pending = result.get("pending") or {}

    if failed_list:
        # R2: a preprocess task failed → no submit. Which end state depends on
        # what the run has to show for the phase:
        #   * no reference set yet, or other tasks still in flight → fail the
        #     run, as R2 always did (mid-flight there is nothing stable to park
        #     on);
        #   * this was the last round, the run already holds a reference set
        #     and a human is driving it → park at preprocess_review with the
        #     error instead (see _should_park_failed_preprocess). That is the
        #     "regenerate just this image" case: failing would throw away the
        #     images the operator kept — and worse, `failed → queued` is the
        #     only edge out of `failed` and it SUBMITS, so a run with an empty
        #     slot would crash the submit and land right back in `failed`. The
        #     operator sees the error on the review card and can re-roll the one
        #     image; approve_preprocess refuses to start while a slot is empty.
        reasons = "; ".join(f.get("reason", "unknown") for f in failed_list)
        message = f"GPT Image preprocessing failed: {reasons}"
        # AICP-360: every write (poll state, run status) happens while this job
        # still owns the lock; the release is last.
        try:
            _delete_preprocess_poll_state(conn, run_id)
            if not still_pending and _should_park_failed_preprocess(run_id):
                log.warning(
                    "run_preprocess_poll: run_id=%s task(s) failed but the run has a "
                    "reference set — parking at preprocess_review",
                    run_id,
                )
                _transition_preprocessing_to_review(run_id, error_message=message)
                return
            _fail_preprocessing_run(run_id, error_message=message)
            return
        finally:
            _release_lock(conn, lock_key, lock_token)

    if still_pending:
        _save_preprocess_poll_state(conn, run_id, still_pending)
        _enqueue_preprocess_poll(
            run_id, lock_key=lock_key, lock_token=lock_token,
            delay_sec=RUN_POLL_INTERVAL_SEC,
        )
    else:
        # All tasks succeeded — R2-new: PAUSE for human review (do NOT enqueue
        # submit). The approve endpoint re-acquires the lock and starts submit.
        # Auto-approve copies submit here instead, keeping the lock.
        _finish_preprocessing(
            run_id, conn=conn, lock_key=lock_key, lock_token=lock_token
        )


def _transition_preprocessing_to_queued(run_id: str) -> bool:
    """Move a run from preprocessing → queued (idempotent; logs on non-fatal).

    Returns True when the run is in ``queued`` afterwards — the caller must
    check it before handing off to submit, because ``submit_run`` transitions
    ``queued → processing`` and would fail the run outright from any other
    status.
    """
    from .db import get_session
    from .models import Run
    from .state_machine import RunStatus, transition

    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            log.warning("_transition_preprocessing_to_queued: run %s not found", run_id)
            return False
        if run.status == RunStatus.preprocessing:
            transition(run, RunStatus.queued)
            session.commit()
            return True
        if run.status == RunStatus.queued:
            return True
        log.warning(
            "_transition_preprocessing_to_queued: run %s in unexpected status "
            "%r — leaving as-is", run_id, run.status,
        )
        return False


def _preprocess_auto_approve(run_id: str) -> bool:
    """True when this run's GPT Image 2 results need no human sign-off.

    Set by "Copy run · re-run GPT Image 2" and by an automation configured the
    same way: those clone their frames and image prompt from a run whose recipe
    an operator already approved, and change only the character photo.
    """
    from .db import get_session
    from .models import Run

    with get_session() as session:
        run = session.get(Run, run_id)
        return bool(run is not None and run.preprocess_auto_approve)


def _should_park_failed_preprocess(run_id: str) -> bool:
    """Where a failed GPT Image 2 task lands: back at review, or ``failed``.

    Park when BOTH hold:

    * **The run already holds a reference set** — at least one entry in
      ``preprocessed_reference_urls``. Deliberately not the stricter "a
      *generated* image survives": regenerating the only generated image is
      precisely where that test flips to False, and failing there is the worst
      outcome available, because ``failed → queued`` is the only edge out of
      ``failed`` and it SUBMITS — feeding a list with an empty slot into
      ``resolve_reference_urls``, which crashes and fails the run again, with no
      way back to review. An empty list means the phase never produced anything
      and fails as it always did.
    * **A human is driving** — i.e. NOT ``preprocess_auto_approve``. An
      unattended copy or automation batch keeps R2's fail-fast, and must: a run
      parked at ``preprocess_review`` is not in ``automation._TERMINAL_RUN_
      STATUSES``, so it would block its batch — and every later batch of that
      automation — until someone noticed. Failing is what the batch reports
      today and what its webhook expects.
    """
    from .db import get_session
    from .models import Run

    try:
        with get_session() as session:
            run = session.get(Run, run_id)
            if run is None:
                return False
            if run.preprocess_auto_approve:
                return False
            return any(run.preprocessed_reference_urls or [])
    except Exception:
        # Can't tell → fall back to the historical behaviour (fail the run).
        # The caller releases the lock right after this decision (AICP-360:
        # still holding it while deciding), so an exception escaping here would
        # leave the run parked in `preprocessing` with no error at all.
        log.exception(
            "_should_park_failed_preprocess: could not inspect run %s", run_id
        )
        return False


def _finish_preprocessing(
    run_id: str, *, conn: Redis, lock_key: str, lock_token: str
) -> None:
    """End the GPT Image 2 phase — pause for review, or auto-approve and submit.

    Default path: transition to ``preprocess_review`` and RELEASE the lock; the
    approve endpoint re-acquires it when the operator clicks through.

    Auto-approve path: take the ``preprocessing → queued`` edge and hand the
    still-held lock straight to submit, so the chain runs unbroken from
    preprocessing through finalize (same handoff the legacy no-review path
    used). If the transition did not land the run in ``queued`` we release the
    lock and stop rather than submit from an unexpected status.
    """
    _delete_preprocess_poll_state(conn, run_id)

    if _preprocess_auto_approve(run_id):
        if _transition_preprocessing_to_queued(run_id):
            log.info(
                "_finish_preprocessing: run %s auto-approved — enqueueing submit",
                run_id,
            )
            _enqueue_submit(run_id, lock_key=lock_key, lock_token=lock_token)
            return
        log.warning(
            "_finish_preprocessing: run %s auto-approve could not reach 'queued' "
            "— releasing lock without submitting", run_id,
        )
        _release_lock(conn, lock_key, lock_token)
        return

    _transition_preprocessing_to_review(run_id)
    _release_lock(conn, lock_key, lock_token)


def _transition_preprocessing_to_review(
    run_id: str, *, error_message: Optional[str] = None
) -> None:
    """R2-new: move a run from preprocessing → preprocess_review (pause for
    human approval of the GPT Image 2 results before Seedance starts).

    *error_message* is set when the phase reached review with a task that
    failed (a single-image regen that did not come back), so the review card can
    say what went wrong; it is cleared on a clean pass.

    Idempotent; logs on non-fatal mismatches.
    """
    from .db import get_session
    from .models import Run
    from .state_machine import RunStatus, transition

    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            log.warning("_transition_preprocessing_to_review: run %s not found", run_id)
            return
        if run.status == RunStatus.preprocessing:
            run.error_message = error_message
            # Belt: nothing should reach review with auto-approve still set —
            # _finish_preprocessing checks it first, and the failure paths
            # refuse to park an unattended run. If one ever did, the card's
            # promise ("will not start until you approve") would break the
            # moment the operator hit Regenerate, which hands the run back to
            # _finish_preprocessing → straight to submit.
            if run.preprocess_auto_approve:
                log.info(
                    "_transition_preprocessing_to_review: run %s parked for a "
                    "human — clearing preprocess_auto_approve", run_id,
                )
                run.preprocess_auto_approve = False
            transition(run, RunStatus.preprocess_review)
            session.commit()
        elif run.status != RunStatus.preprocess_review:
            log.warning(
                "_transition_preprocessing_to_review: run %s in unexpected status "
                "%r — leaving as-is", run_id, run.status,
            )


def _fail_preprocessing_run(run_id: str, error_message: Optional[str] = None) -> None:
    """Mark a preprocessing run failed (best-effort; logs on non-fatal).

    When *error_message* is provided it is written to ``run.error_message``
    before the transition so the run-detail UI can surface why preprocessing
    failed (R2).
    """
    from .db import get_session
    from .models import Run
    from .state_machine import RunStatus, transition

    try:
        with get_session() as session:
            run = session.get(Run, run_id)
            if run and run.status == RunStatus.preprocessing:
                if error_message:
                    run.error_message = error_message
                transition(run, RunStatus.failed)
                session.commit()
    except Exception:
        log.exception("_fail_preprocessing_run: failed to mark run %s failed", run_id)


def _enqueue_submit(run_id: str, *, lock_key: str, lock_token: str) -> None:
    """Enqueue run_submit_run with an existing lock (preprocess → submit handoff)."""
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_submit_run",
        run_id,
        lock_key=lock_key,
        lock_token=lock_token,
        job_timeout=SUBMIT_JOB_TIMEOUT,
    )
    log.info("run_submit_run enqueued for run_id=%s → rq_job=%s", run_id, job.id)


# ---------------------------------------------------------------------------
# Per-run preprocess poll-state Redis key
# ---------------------------------------------------------------------------


def _preprocess_poll_state_key(run_id: str) -> str:
    return f"reskin:run:preprocess:poll:{run_id}"


def _load_preprocess_poll_state(conn: Redis, run_id: str) -> dict:
    raw = conn.get(_preprocess_poll_state_key(run_id))
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _save_preprocess_poll_state(conn: Redis, run_id: str, state: dict) -> None:
    conn.setex(_preprocess_poll_state_key(run_id), _POLL_STATE_TTL_SEC, json.dumps(state))


def _delete_preprocess_poll_state(conn: Redis, run_id: str) -> None:
    conn.delete(_preprocess_poll_state_key(run_id))


def _enqueue_preprocess_poll(
    run_id: str, *, lock_key: str, lock_token: str, delay_sec: float,
) -> None:
    """Schedule run_preprocess_poll to execute after delay_sec."""
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue_in(
        timedelta(seconds=delay_sec),
        "app.tasks.run_preprocess_poll",
        run_id,
        lock_key=lock_key,
        lock_token=lock_token,
        job_timeout=POLL_JOB_TIMEOUT,
    )
    log.info(
        "run_preprocess_poll scheduled for run_id=%s in %.0fs → rq_job=%s",
        run_id, delay_sec, job.id,
    )


def enqueue_preprocess_run(run_id: str, *, lock_token: Optional[str] = None) -> None:
    """Push :func:`run_preprocess_refs` onto the RQ "process" queue.

    Called by api_v2.py when a Run is created with ``preprocess_enabled=True``.
    Mirrors :func:`enqueue_process_run` (the non-preprocess entry point).

    *lock_token*: recovery only (AICP-360) — hand the job a per-run lock the
    caller already owns (``claim_stale_run_lock``) instead of letting it
    acquire one.
    """
    q = _get_queue(QUEUE_PROCESS)
    kwargs = {}
    if lock_token is not None:
        kwargs = {"lock_key": run_lock_key(run_id), "lock_token": lock_token}
    job = q.enqueue(
        "app.tasks.run_preprocess_refs", run_id, job_timeout=SUBMIT_JOB_TIMEOUT,
        **kwargs,
    )
    log.info("Enqueued preprocess_refs for run_id=%s → rq_job=%s", run_id, job.id)


def _is_images_only_run(run_id: str) -> bool:
    from .db import get_session
    from .models import Run

    try:
        with get_session() as session:
            run = session.get(Run, run_id)
            return bool(run is not None and run.generation_mode == "images_only")
    except Exception:  # defence only: never break a v2v poll round over it
        log.warning("_is_images_only_run(%s) failed — treating as v2v", run_id, exc_info=True)
        return False


def _submit_images_only_run(conn: Redis, run_id: str, lock_key: str, lock_token: str) -> None:
    """The images-only branch of :func:`run_submit_run` (lock already held)."""
    from .pipeline_rebuild import submit_images_only

    try:
        drivable = submit_images_only(run_id)
    except Exception:
        _release_lock(conn, lock_key, lock_token)
        raise
    if not drivable:
        _release_lock(conn, lock_key, lock_token)
        return
    _enqueue_rebuild_observe(run_id, lock_key=lock_key, lock_token=lock_token,
                             delay_sec=RUN_POLL_INTERVAL_SEC)


def run_rebuild_observe(run_id: str, *, lock_key: str, lock_token: str) -> None:
    """RQ-callable observer of an images-only run (design §6.4).

    Holds the per-run lock exactly like ``run_poll_run``: refreshes it first
    and stops if it lost it. Pure observation of the chunk attempts (their own
    step jobs drive them): every chunk completed -> ``run_finalize_run`` with
    the lock handed over; a chunk failed and nothing live -> the run is
    ``incomplete`` and the lock released; otherwise re-schedule itself.
    """
    _kcap_begin()
    from .pipeline_rebuild import (
        OBSERVE_FINALIZE,
        OBSERVE_WAIT,
        observe_run,
    )

    conn = _get_redis()
    if not _refresh_lock(conn, lock_key, lock_token):
        log.warning("run_rebuild_observe: run %s lock lost — stopping this chain", run_id)
        return
    try:
        verdict = observe_run(run_id)
    except Exception:
        _release_lock(conn, lock_key, lock_token)
        raise
    if verdict == OBSERVE_FINALIZE:
        _enqueue_finalize(run_id, lock_key=lock_key, lock_token=lock_token)
    elif verdict == OBSERVE_WAIT:
        _enqueue_rebuild_observe(run_id, lock_key=lock_key, lock_token=lock_token,
                                 delay_sec=RUN_POLL_INTERVAL_SEC)
    else:
        _release_lock(conn, lock_key, lock_token)


def _enqueue_rebuild_observe(run_id: str, *, lock_key: str, lock_token: str,
                             delay_sec: float) -> None:
    """Schedule :func:`run_rebuild_observe` (run id as the first argument, so
    recovery's liveness scan sees the run as live while it is scheduled)."""
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue_in(
        timedelta(seconds=delay_sec), "app.tasks.run_rebuild_observe", run_id,
        lock_key=lock_key, lock_token=lock_token, job_timeout=POLL_JOB_TIMEOUT,
    )
    log.info("run_rebuild_observe scheduled for run_id=%s in %.0fs -> rq_job=%s",
             run_id, delay_sec, job.id)


def run_poll_run(run_id: str, *, lock_key: str, lock_token: str) -> None:
    """
    RQ-callable poll-phase wrapper.

    Performs ONE round of status checks on all pending AI tasks.  For each
    completed task, downloads the result and marks the segment done.  For each
    failed task, retries or marks the segment failed.

    If tasks are still pending after this round, re-enqueues itself via
    ``queue.enqueue_in`` to run after RUN_POLL_INTERVAL_SEC.

    When all tasks are terminal (completed or failed), enqueues run_finalize_run.
    """
    _kcap_begin()
    from .pipeline_v2 import poll_run_once

    conn = _get_redis()

    if not _refresh_lock(conn, lock_key, lock_token):
        # Lock expired or was taken over (e.g. recovery re-enqueued the run).
        # Stop this chain — the new owner drives the run from here.
        log.warning(
            "run_poll_run: run %s lock lost — aborting this poll chain", run_id
        )
        return

    if _is_images_only_run(run_id):
        # Defence in depth (docs/rebuild-from-reference-design.md §6.4): the
        # v2v poll chain never drives an images-only run. No state change.
        log.error("run_poll_run refused images-only run %s", run_id)
        return

    pending = _load_poll_state(conn, run_id)

    if not pending:
        # No pending state — either all tasks finished or state expired.
        # Proceed to finalize (finalize will detect any missing segments).
        log.info("run_poll_run: run_id=%s no poll state — going to finalize", run_id)
        _delete_poll_state(conn, run_id)
        _enqueue_finalize(run_id, lock_key=lock_key, lock_token=lock_token)
        return

    log.info("run_poll_run: run_id=%s pending_tasks=%d", run_id, len(pending))

    try:
        still_pending = poll_run_once(run_id, pending=pending)
    except Exception:
        # On unexpected error, release the lock and propagate.
        _release_lock(conn, lock_key, lock_token)
        raise

    if still_pending:
        # Persist updated state and re-enqueue after the poll interval.
        _save_poll_state(conn, run_id, still_pending)
        _enqueue_poll(
            run_id,
            lock_key=lock_key,
            lock_token=lock_token,
            delay_sec=RUN_POLL_INTERVAL_SEC,
        )
    else:
        # All tasks are terminal — clean up state and finalize.
        _delete_poll_state(conn, run_id)
        _enqueue_finalize(run_id, lock_key=lock_key, lock_token=lock_token)


def run_finalize_run(run_id: str, *, lock_key: str, lock_token: str) -> None:
    """
    RQ-callable finalize-phase wrapper.

    Calls pipeline_v2.finalize_run (completeness gate + stitch + deliver), then
    releases the per-run Redis lock.
    """
    _kcap_begin()
    from .gdrive_client import GDriveClient
    from .pipeline_v2 import finalize_run

    conn = _get_redis()
    log.info("run_finalize_run: run_id=%s", run_id)
    # AICP-342: a finalize whose lock was taken over or expired belongs to a
    # lifecycle that is over (recovery takes a lock over only for a run it
    # judged orphaned — AICP-360 — and a new lifecycle may own the run now). Do nothing — not
    # even a state transition — rather than stitch, fail, or re-deliver over
    # the owner. Since a re-delivery now replaces the content behind the
    # exported Drive link, the same check is repeated under the Run-row lock
    # right before that update (owner_check below).
    if not _refresh_lock(conn, lock_key, lock_token):
        log.warning(
            "run_finalize_run: run_id=%s no longer owns its processing lock — "
            "superseded, skipping", run_id,
        )
        return
    link_failure = None
    try:
        # AICP-342: an in-place re-delivery replaces the bytes behind the
        # run's exported Drive link, so it is only done while this job still
        # owns the run's processing lock (checked under the Run-row lock).
        published = finalize_run(
            run_id, gdrive=GDriveClient(), publication_id=_stitch_publication_id(run_id),
            owner_check=lambda: _refresh_lock(conn, lock_key, lock_token),
        )
        if published is not None:
            # Knowledge publication link (i) (AICP-299 §3.2): the last statement
            # after all committed work; syscalls only, never raises.
            from .knowledge import capture

            link_failure = capture.link_publication(published)
    finally:
        _release_lock(conn, lock_key, lock_token)
    if link_failure is not None:
        # The durable failure row, only after the lock is released.
        from .knowledge import capture

        capture.record_link_failure(link_failure)


def _stitch_publication_id(run_id: str) -> Optional[str]:
    """``stitch:<run>:<RQ job id>`` for this finalize invocation (None → minted)."""
    try:
        from rq import get_current_job

        job = get_current_job()
        return f"stitch:{run_id}:{job.id}" if job is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Legacy wrapper — kept for backward compatibility with existing callers
# (api_v2.py calls enqueue_process_run; the lock contract is unchanged).
# ---------------------------------------------------------------------------


def run_process_run(run_id: str) -> None:
    """
    Legacy monolithic RQ-callable wrapper (kept for backward compat).

    New code should use run_submit_run, but this wrapper is kept in case any
    queue-introspection code references the old function string.  It delegates
    to run_submit_run.
    """
    run_submit_run(run_id)


# ---------------------------------------------------------------------------
# Helper: enqueue poll / finalize
# ---------------------------------------------------------------------------


def _enqueue_poll(
    run_id: str,
    *,
    lock_key: str,
    lock_token: str,
    delay_sec: float,
) -> None:
    """Schedule run_poll_run to execute after delay_sec."""
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue_in(
        timedelta(seconds=delay_sec),
        "app.tasks.run_poll_run",
        run_id,
        lock_key=lock_key,
        lock_token=lock_token,
        job_timeout=POLL_JOB_TIMEOUT,
    )
    log.info(
        "run_poll_run scheduled for run_id=%s in %.0fs → rq_job=%s",
        run_id, delay_sec, job.id,
    )


def _enqueue_finalize(run_id: str, *, lock_key: str, lock_token: str) -> None:
    """Enqueue run_finalize_run immediately."""
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_finalize_run",
        run_id,
        lock_key=lock_key,
        lock_token=lock_token,
        job_timeout=FINALIZE_JOB_TIMEOUT,
    )
    log.info(
        "run_finalize_run enqueued for run_id=%s → rq_job=%s", run_id, job.id
    )


def _refresh_lock(conn: Redis, lock_key: str, lock_token: str) -> bool:
    """Re-arm the lock TTL (and re-stamp its generation) if we still own it.

    Returns False if the lock was lost.
    """
    return bool(conn.eval(
        _REFRESH_LOCK_LUA, 2, lock_key, run_lock_gen_key(lock_key),
        lock_token, _RUN_LOCK_TTL_SEC, _uuid_mod.uuid4().hex,
    ))


def _release_lock(conn: Redis, lock_key: str, lock_token: str) -> None:
    """Atomically release the per-run processing lock."""
    released = conn.eval(_RELEASE_LOCK_LUA, 1, lock_key, lock_token)
    if released:
        log.debug("run lock released: %s", lock_key)
    else:
        log.debug("run lock not released (token mismatch or already expired): %s", lock_key)


# ---------------------------------------------------------------------------
# v2 Enqueue helpers (public API — called by api_v2.py and recovery.py)
# ---------------------------------------------------------------------------


def enqueue_analyze_project(project_id: str, *, analysis_token: Optional[str] = None) -> None:
    """
    Push :func:`run_analyze_project` onto the RQ "analyze" queue.

    Parameters
    ----------
    project_id:
        The UUID of the VideoProject to analyse.
    analysis_token:
        Rebuild re-analysis only: the ``rebuild_analysis_token`` its admission
        wrote; the job runs only while the project still carries it. Omitted
        for every other enqueue (the job's arguments stay what they were).
    """
    q = _get_queue(QUEUE_ANALYZE)
    if analysis_token:
        # Passed as a plain keyword: rq hands unknown keywords to the job
        # function (mixing positional args with `kwargs=` is rejected).
        job = q.enqueue(
            "app.tasks.run_analyze_project", project_id,
            analysis_token=analysis_token, job_timeout=ANALYZE_JOB_TIMEOUT,
        )
    else:
        job = q.enqueue(
            "app.tasks.run_analyze_project", project_id, job_timeout=ANALYZE_JOB_TIMEOUT
        )
    log.info(
        "Enqueued analyze_project for project_id=%s → rq_job=%s", project_id, job.id
    )


def enqueue_transcribe_project(project_id: str) -> None:
    """
    Push :func:`run_transcribe_project` onto the RQ "process" queue.

    The "process" tier, not "analyze": transcription burns no CPU, it is one
    upload plus one LLM request that spends nearly all its time waiting — the
    same "long-ish, mostly I/O" shape as merges and tail edits. Keeping it off
    the analyze queue also keeps it from delaying the CPU-bound segmentation of
    the next upload.

    Callers that want the UI to show the queued state set
    ``transcript_status="pending"`` before calling (see
    :func:`_enqueue_transcribe_after_analyze` and ``POST /transcribe``); this
    helper stays a pure queue push, like every other enqueue_* in this module.

    Parameters
    ----------
    project_id:
        The UUID of the VideoProject to transcribe.
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_transcribe_project",
        project_id,
        job_timeout=TRANSCRIBE_JOB_TIMEOUT,
    )
    log.info(
        "Enqueued transcribe_project for project_id=%s → rq_job=%s", project_id, job.id
    )


def enqueue_process_run(run_id: str, *, lock_token: Optional[str] = None) -> None:
    """
    Push :func:`run_submit_run` onto the RQ "process" queue.

    This is the single public entry point for starting a run; it submits to the
    process queue where workers pick it up.  api_v2.py and recovery.py call this.

    Parameters
    ----------
    run_id:
        The UUID of the Run to process.
    lock_token:
        Recovery only (AICP-360): a per-run lock the caller already owns
        (``claim_stale_run_lock``), handed to the job the same way the
        preprocess chain hands its lock to submit — so there is never a moment
        between the reaper's claim and the recovered job where the run is
        unlocked.
    """
    q = _get_queue(QUEUE_PROCESS)
    kwargs = {}
    if lock_token is not None:
        kwargs = {"lock_key": run_lock_key(run_id), "lock_token": lock_token}
    job = q.enqueue(
        "app.tasks.run_submit_run", run_id, job_timeout=SUBMIT_JOB_TIMEOUT,
        **kwargs,
    )
    log.info("Enqueued process_run (submit) for run_id=%s → rq_job=%s", run_id, job.id)


# ---------------------------------------------------------------------------
# Periodic orphaned-run reconciliation ("reaper")
# ---------------------------------------------------------------------------


def enqueue_reconcile_tick(*, is_continuation: bool = False) -> None:
    """(Re)schedule the periodic reconciliation tick, bounded to one chain.

    A Redis lease (`_RECONCILE_CHAIN_KEY`, TTL = 3× the interval) ensures exactly
    one tick chain exists no matter how many workers boot or how often they
    restart:

    - Boot (``is_continuation=False``): only schedules a tick if no chain is
      already live (``SET NX``). With two workers booting together, the second's
      NX fails, so a single chain is created. A restart while a chain is still
      live is likewise a no-op — the scheduled tick survives the restart in
      Redis and the running scheduler will fire it.
    - Continuation (``is_continuation=True``, called from ``run_reconcile_tick``
      itself): refreshes the lease and schedules the next tick unconditionally,
      keeping the single chain alive.

    Each tick uses a fresh auto-generated job id (never a fixed one) so a tick
    never collides with the RQ result hash of the invocation that scheduled it.
    If a tick is killed mid-run (worker restart during the tick), the lease
    expires within 3× the interval and the next boot restarts the chain.
    """
    conn = _get_redis()
    try:
        if is_continuation:
            conn.set(_RECONCILE_CHAIN_KEY, "1", ex=RECONCILE_TICK_INTERVAL_SEC * 3)
        else:
            acquired = conn.set(
                _RECONCILE_CHAIN_KEY, "1", nx=True, ex=RECONCILE_TICK_INTERVAL_SEC * 3
            )
            if not acquired:
                log.info(
                    "reconcile tick: a chain is already active — not starting another"
                )
                return
        q = _get_queue(QUEUE_PROCESS)
        job = q.enqueue_in(
            timedelta(seconds=RECONCILE_TICK_INTERVAL_SEC),
            "app.tasks.run_reconcile_tick",
            job_timeout=300,
        )
        log.info(
            "reconcile tick scheduled in %ds → rq_job=%s",
            RECONCILE_TICK_INTERVAL_SEC, job.id,
        )
    except Exception as exc:  # never let scheduling failure break the caller
        log.warning("reconcile tick: failed to schedule: %s", exc)


def run_reconcile_tick() -> None:
    """Periodic reaper: re-run orphaned-run reconciliation, then reschedule.

    Reschedules the next tick in a ``finally`` block so a single reconciliation
    failure never breaks the chain.
    """
    from .recovery import reconcile_orphaned_runs

    try:
        reconcile_orphaned_runs(_get_redis())
    except Exception:
        log.exception("run_reconcile_tick: reconciliation raised — continuing")
    try:
        # AICP-365: re-send lost transcription / translate doorbells, interrupt
        # jobs whose worker died mid-call (never replayed).
        from . import localisation_jobs

        localisation_jobs.reap()
    except Exception:
        log.exception("run_reconcile_tick: localisation job reaper raised — continuing")
    try:
        # Rebuild paid-call rows (docs/rebuild-from-reference-design.md §7.2):
        # re-drive rows whose step job died, release a `sending` row only on a
        # proof that its sender is gone. A no-op while no rebuild row is live.
        from . import rebuild_paid_call

        rebuild_paid_call.reap()
    except Exception:
        log.exception("run_reconcile_tick: rebuild paid-call reaper raised — continuing")
    try:
        # Rebuild storyboards (design §5.2): re-send a lost first job, interrupt
        # a storyboard whose advance chain died (its unsent calls cancelled in
        # the same transaction; sent calls keep their own life above).
        from . import rebuild_storyboard

        rebuild_storyboard.reap()
    except Exception:
        log.exception("run_reconcile_tick: rebuild storyboard reaper raised — continuing")
    try:
        # Rebuild QA (design §6.5): re-enqueue the free QA of a succeeded
        # chunk attempt whose QA job was lost; admit a missing automatic
        # character check. Never re-sends a paid call.
        from . import pipeline_rebuild

        pipeline_rebuild.reap_qa()
    except Exception:
        log.exception("run_reconcile_tick: rebuild QA reaper raised — continuing")
    finally:
        enqueue_reconcile_tick(is_continuation=True)


def knowledge_import_tick() -> Optional[dict]:
    """One bounded knowledge import pass (AICP-297, design §6.3).

    A no-op unless ``KNOWLEDGE_CAPTURE_ENABLED``. Its own job, never part of
    ``run_reconcile_tick``; nothing schedules it yet. Holds a Redis lease and
    the per-source ``kb_import_cursors`` leases, and handles at most
    ``TICK_LIMIT`` source rows.
    """
    if not settings.KNOWLEDGE_CAPTURE_ENABLED:
        return None
    # The historical sources and the other reconcile steps read every project, so
    # they run only by explicit opt-in (AICP-298); otherwise just the C2 drain of
    # the capture queue, which only in-scope capture fills.
    drain_only = not getattr(settings, "KNOWLEDGE_IMPORT_TICK", False)
    from .db import SessionLocal
    from .knowledge.importer import run_tick

    try:
        report = run_tick(SessionLocal, _get_redis(), drain_only=drain_only)
    except Exception:
        log.exception("knowledge_import_tick: import raised")
        return None
    log.info("knowledge_import_tick: outcome=%s batch=%s created=%s", report.get("outcome") or report.get("skipped"),
             report.get("batch_id"), report.get("created"))
    return report


# ---------------------------------------------------------------------------
# Knowledge tick (AICP-299 §6.2): enqueued per slot by the supervised ticker
# (worker/knowledge_ticker.py) onto the "knowledge" queue, which only the slim
# knowledge-worker listens on. No chain, no RQ scheduler.
# ---------------------------------------------------------------------------

QUEUE_KNOWLEDGE = "knowledge"
KNOWLEDGE_TICK_INTERVAL_SEC = int(os.getenv("KNOWLEDGE_TICK_INTERVAL_SEC", "300"))
KNOWLEDGE_TICK_TIMEOUT = 900  # importer LEASE_TTL_SEC; its TICK_MAX_SEC is 600
KNOWLEDGE_TICK_JOB_PREFIX = "knowledge-tick-"
FINALS_SCAN_BUDGET_SEC = 240
_KNOWLEDGE_TICK_LOCK = "reskin:knowledge-tick:running"
_KNOWLEDGE_TICK_LAST_OK = "reskin:knowledge-tick:last_ok"
_KNOWLEDGE_SLOT_KEY = "reskin:knowledge-tick:slot:"

_TICK_LOCK_CAS = """\
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  return 1
end
return 0
"""
_TICK_LOCK_RELEASE = """\
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _live_worker_current_jobs(conn: Redis) -> set:
    """Job ids that a worker with a live heartbeat key reports as current.

    Positive evidence of a running horse (design §6.2): the work-horse sets the
    worker's current job before the job function runs; it is cleared only after
    the job finished or the parent reaped a killed horse. Registry expiry and
    cleanup are deliberately not consulted.
    """
    from rq import Worker

    out = set()
    for w in Worker.all(connection=conn):  # only workers whose key (heartbeat) still exists
        try:
            job_id = w.get_current_job_id()
        except Exception:
            job_id = None
        if job_id:
            out.add(_decode(job_id))
    return out


def knowledge_job_is_live(conn: Redis, job_id: Optional[str]) -> bool:
    """Queued on ``knowledge``, or the current job of a live worker."""
    if not job_id:
        return False
    job_id = _decode(job_id)
    if job_id in {_decode(j) for j in _get_queue(QUEUE_KNOWLEDGE).get_job_ids()}:
        return True
    return job_id in _live_worker_current_jobs(conn)


def rq_job_is_live(conn: Redis, queue: str, job_id: Optional[str]) -> bool:
    """Pending in *queue*, in its scheduled registry, or the current job of a
    worker with a live heartbeat (``knowledge_job_is_live`` generalised)."""
    from rq.registry import ScheduledJobRegistry

    if not job_id:
        return False
    job_id = _decode(job_id)
    if job_id in {_decode(j) for j in _get_queue(queue).get_job_ids()}:
        return True
    try:
        if job_id in {_decode(j) for j in ScheduledJobRegistry(queue, connection=conn).get_job_ids()}:
            return True
    except Exception:
        log.warning("rq_job_is_live: scheduled registry read failed", exc_info=True)
    return job_id in _live_worker_current_jobs(conn)


def _enqueue_fixed_id(queue: str, func: str, args: tuple, job_id: str, timeout: int) -> None:
    from rq.job import Job

    conn = _get_redis()
    try:
        Job.fetch(job_id, connection=conn).delete()
    except Exception:
        pass  # no stale job hash under this id
    _get_queue(queue).enqueue(func, *args, job_id=job_id, job_timeout=timeout)


def enqueue_transcription_job(job_id: str, dispatch_gen: int, rq_job_id: str) -> None:
    """Doorbell of a durable transcription (AICP-365, app/localisation_jobs.py)."""
    _enqueue_fixed_id(QUEUE_PROCESS, "app.localisation_jobs.run_transcription_job",
                      (job_id, dispatch_gen), rq_job_id, TRANSCRIBE_JOB_TIMEOUT)
    log.info("Enqueued transcription job %s (rq_job=%s)", job_id, rq_job_id)


def enqueue_translate_job(job_id: str, dispatch_gen: int, rq_job_id: str) -> None:
    """Doorbell of a durable translate job, on the localise worker's queue."""
    from .localisation_prompt_job import job_timeout_sec

    _enqueue_fixed_id(QUEUE_LOCALISE, "app.localisation_jobs.run_translate_job",
                      (job_id, dispatch_gen), rq_job_id, job_timeout_sec())
    log.info("Enqueued translate job %s (rq_job=%s)", job_id, rq_job_id)


def enqueue_autopilot_compute(effect_id: str, job_id: str, queue: Optional[str] = None) -> None:
    """Enqueue (or re-enqueue) one autopilot compute effect under its fixed id.

    *queue* is the compute function's registered queue (AICP-389: the brand
    check runs on ``brand_asr``); default ``autopilot_compute``."""
    from rq.job import Job

    conn = _get_redis()
    try:
        Job.fetch(job_id, connection=conn).delete()
    except Exception:
        pass  # no stale job hash
    _get_queue(queue or QUEUE_AUTOPILOT_COMPUTE).enqueue(
        "app.autopilot.compute.run", effect_id, job_id=job_id,
        job_timeout=AUTOPILOT_COMPUTE_JOB_TIMEOUT,
    )


def knowledge_tick_is_live(conn: Redis) -> bool:
    """A tick is queued, or is some live worker's current job (never overlap)."""
    queued = [_decode(j) for j in _get_queue(QUEUE_KNOWLEDGE).get_job_ids()]
    if any(j.startswith(KNOWLEDGE_TICK_JOB_PREFIX) for j in queued):
        return True
    return any(j.startswith(KNOWLEDGE_TICK_JOB_PREFIX) for j in _live_worker_current_jobs(conn))


def enqueue_knowledge_tick_for_slot(conn: Redis, slot: int, ticker_id: str) -> bool:
    """Admission: at most one tick per slot (SET NX, never deleted early)."""
    from .knowledge import capture

    if not settings.KNOWLEDGE_CAPTURE_ENABLED or not capture.capture_scope().valid:
        return False
    if knowledge_tick_is_live(conn):
        log.info("knowledge_tick_slot_skipped slot=%s reason=tick_live", slot)
        return False
    if not conn.set(f"{_KNOWLEDGE_SLOT_KEY}{slot}", ticker_id, nx=True, ex=2 * KNOWLEDGE_TICK_INTERVAL_SEC):
        return False
    _get_queue(QUEUE_KNOWLEDGE).enqueue(
        "app.tasks.run_knowledge_tick", job_id=f"{KNOWLEDGE_TICK_JOB_PREFIX}{slot}",
        job_timeout=KNOWLEDGE_TICK_TIMEOUT,
    )
    return True


def acquire_knowledge_tick_lock(conn: Redis, me: str) -> bool:
    """Execution-time mutual exclusion; a holder is replaced only when it is
    provably not running (not queued, not any live worker's current job)."""
    if conn.set(_KNOWLEDGE_TICK_LOCK, me, nx=True, ex=86400):
        return True
    holder = _decode(conn.get(_KNOWLEDGE_TICK_LOCK))
    if holder is None:
        return bool(conn.set(_KNOWLEDGE_TICK_LOCK, me, nx=True, ex=86400))
    if holder == me or knowledge_job_is_live(conn, holder):
        return False
    return bool(conn.eval(_TICK_LOCK_CAS, 1, _KNOWLEDGE_TICK_LOCK, holder, me, 86400))


def release_knowledge_tick_lock(conn: Redis, me: str) -> None:
    try:
        conn.eval(_TICK_LOCK_RELEASE, 1, _KNOWLEDGE_TICK_LOCK, me)
    except Exception:
        log.warning("knowledge tick: lock release failed (it is replaced by liveness)", exc_info=True)


def run_knowledge_tick() -> Optional[dict]:
    """One knowledge pass: capture scope row, finals scan, importer (C2 + reconcile)."""
    from rq import get_current_job
    from rq.timeouts import BaseTimeoutException

    if not settings.KNOWLEDGE_CAPTURE_ENABLED:
        return None
    conn = _get_redis()
    job = get_current_job()
    me = job.id if job is not None else f"local-{_uuid_mod.uuid4().hex}"
    if not acquire_knowledge_tick_lock(conn, me):
        log.info("knowledge_tick_skipped_running")
        return None
    _kcap_begin()
    report: dict = {}
    try:
        from .db import SessionLocal
        from .knowledge import capture, capture_exec, finals_scan

        session = SessionLocal()
        try:
            capture.record_capture_scope(session, recorded_by=f"knowledge_tick:{me}")
            session.commit()
        finally:
            session.close()
        report["finals"] = finals_scan.run(SessionLocal, deadline=time.monotonic() + FINALS_SCAN_BUDGET_SEC)
        report["import"] = knowledge_import_tick()
        conn.set(_KNOWLEDGE_TICK_LAST_OK, json.dumps({
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "capture_health": capture_exec.read_health_keys(conn)}, default=str), ex=86400)
    except BaseTimeoutException:
        raise  # RQ's timeout is never swallowed (the job is marked failed)
    except Exception:
        log.exception("run_knowledge_tick")
    finally:
        release_knowledge_tick_lock(conn, me)
    log.info("run_knowledge_tick: %s", json.dumps(report, default=str)[:2000])
    return report


def enqueue_automation_tick(*, is_continuation: bool = False) -> None:
    """(Re)schedule the periodic run-automation tick, bounded to one chain.

    Identical lease semantics to :func:`enqueue_reconcile_tick`: a boot call
    only starts a chain when none is live (SET NX); each tick refreshes the
    lease and schedules its successor. If a tick dies mid-run, the lease
    expires within 3× the interval and the next worker boot restarts the chain.
    """
    conn = _get_redis()
    try:
        if is_continuation:
            conn.set(_AUTOMATION_CHAIN_KEY, "1", ex=AUTOMATION_TICK_INTERVAL_SEC * 3)
        else:
            acquired = conn.set(
                _AUTOMATION_CHAIN_KEY, "1", nx=True, ex=AUTOMATION_TICK_INTERVAL_SEC * 3
            )
            if not acquired:
                log.info(
                    "automation tick: a chain is already active — not starting another"
                )
                return
        q = _get_queue(QUEUE_PROCESS)
        job = q.enqueue_in(
            timedelta(seconds=AUTOMATION_TICK_INTERVAL_SEC),
            "app.tasks.run_automation_tick",
            job_timeout=600,
        )
        log.info(
            "automation tick scheduled in %ds → rq_job=%s",
            AUTOMATION_TICK_INTERVAL_SEC, job.id,
        )
    except Exception as exc:  # never let scheduling failure break the caller
        log.warning("automation tick: failed to schedule: %s", exc)


def run_automation_tick() -> None:
    """Periodic automation pass: webhooks for finished batches, then launch the
    next batches within each automation's daily quota. Reschedules itself in a
    ``finally`` block so one failed pass never breaks the chain.
    """
    from .automation import run_tick
    from .db import get_session
    from .models import Run
    from .state_machine import RunStatus

    try:
        with get_session() as session:
            new_run_ids = run_tick(session)
            # An automation with preprocess_refs launches its copies into
            # `preprocessing`, which starts on the GPT Image 2 chain instead of
            # submit. Read the statuses back inside the session that created
            # them so the entry point matches (mirrors _enqueue_copied_run).
            statuses = {
                rid: (session.get(Run, rid).status if session.get(Run, rid) else None)
                for rid in new_run_ids
            }
        for rid in new_run_ids:
            if statuses.get(rid) == RunStatus.preprocessing:
                enqueue_preprocess_run(rid)
            else:
                enqueue_process_run(rid)
    except Exception:
        log.exception("run_automation_tick: tick raised — continuing")
    finally:
        enqueue_automation_tick(is_continuation=True)


def enqueue_disk_cleanup_tick(*, is_continuation: bool = False) -> None:
    """(Re)schedule the periodic disk-cleanup tick, bounded to one chain.

    Identical lease semantics to :func:`enqueue_reconcile_tick`.
    """
    conn = _get_redis()
    try:
        if is_continuation:
            conn.set(_DISK_CLEANUP_CHAIN_KEY, "1", ex=DISK_CLEANUP_TICK_INTERVAL_SEC * 3)
        else:
            acquired = conn.set(
                _DISK_CLEANUP_CHAIN_KEY, "1", nx=True, ex=DISK_CLEANUP_TICK_INTERVAL_SEC * 3
            )
            if not acquired:
                log.info(
                    "disk cleanup tick: a chain is already active — not starting another"
                )
                return
        q = _get_queue(QUEUE_PROCESS)
        job = q.enqueue_in(
            timedelta(seconds=DISK_CLEANUP_TICK_INTERVAL_SEC),
            "app.tasks.run_disk_cleanup_tick",
            job_timeout=600,
        )
        log.info(
            "disk cleanup tick scheduled in %ds → rq_job=%s",
            DISK_CLEANUP_TICK_INTERVAL_SEC, job.id,
        )
    except Exception as exc:  # never let scheduling failure break the caller
        log.warning("disk cleanup tick: failed to schedule: %s", exc)


def run_disk_cleanup_tick() -> None:
    """Periodic disk cleanup (app/disk_cleanup.py), then reschedule in a
    ``finally`` block so one failed pass never breaks the chain."""
    from .db import get_session
    from .disk_cleanup import sweep

    try:
        with get_session() as session:
            sweep(session)
    except Exception:
        log.exception("run_disk_cleanup_tick: sweep raised — continuing")
    try:
        # AICP-350: Idempotency-Key retention (30 days, design §3).
        from .ops_gate import purge_idempotency_keys

        with get_session() as session:
            purged = purge_idempotency_keys(session)
        if purged:
            log.info("run_disk_cleanup_tick: purged %d expired idempotency key(s)", purged)
    except Exception:
        log.exception("run_disk_cleanup_tick: idempotency-key purge raised — continuing")
    finally:
        enqueue_disk_cleanup_tick(is_continuation=True)


# ---------------------------------------------------------------------------
# Overlay feature (standalone PNG-burn) — RQ target + enqueue helper
# ---------------------------------------------------------------------------

# Overlay is a single ffmpeg re-encode plus an optional Drive upload — far
# shorter than a generation run, but reuse the same generous timeout so a large
# source video on the memory-capped worker is never killed mid-encode.


def run_process_overlay(overlay_job_id: str) -> None:
    """
    RQ-callable wrapper for :func:`overlay_pipeline.process_overlay_job`.

    Creates a real :class:`GDriveClient` (used only when the job opts into Drive
    delivery). Imports the pipeline lazily to match the v1/v2 task wrappers.
    """
    from .gdrive_client import GDriveClient
    from .overlay_pipeline import process_overlay_job

    log.info("run_process_overlay: overlay_job_id=%s", overlay_job_id)
    process_overlay_job(overlay_job_id, gdrive=GDriveClient())


def run_process_merge(merge_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_merge.process_merge`.

    Creates a real :class:`GDriveClient`. Imports the pipeline lazily to match
    the other RQ target wrappers in this module.
    """
    from .gdrive_client import GDriveClient
    from .pipeline_merge import process_merge

    log.info("run_process_merge: merge_id=%s", merge_id)
    process_merge(merge_id, gdrive=GDriveClient())


def enqueue_process_merge(merge_id: str) -> None:
    """Push :func:`run_process_merge` onto the RQ "process" queue.

    Same queue as run processing (compositing is a single ffmpeg pass, cheaper
    than a full run, but belongs to the same "long-ish, mostly I/O" priority
    tier — behind the interactive overlay/analyze queues).
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_process_merge", merge_id, job_timeout=MERGE_JOB_TIMEOUT
    )
    log.info(
        "Enqueued process_merge for merge_id=%s → rq_job=%s", merge_id, job.id
    )


def run_process_tail_edit(edit_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_tail.process_tail_edit`.

    Creates a real :class:`GDriveClient`. Imports the pipeline lazily to match
    the other RQ target wrappers in this module.
    """
    from .gdrive_client import GDriveClient
    from .pipeline_tail import process_tail_edit

    log.info("run_process_tail_edit: edit_id=%s", edit_id)
    process_tail_edit(edit_id, gdrive=GDriveClient())


def enqueue_process_tail_edit(edit_id: str) -> None:
    """Push :func:`run_process_tail_edit` onto the RQ "process" queue.

    Same queue as run processing and merge compositing (a cut + stitch pass,
    cheaper than a full run, but belongs to the same "long-ish, mostly I/O"
    priority tier — behind the interactive overlay/analyze queues).
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_process_tail_edit", edit_id, job_timeout=TAIL_EDIT_JOB_TIMEOUT
    )
    log.info(
        "Enqueued process_tail_edit for edit_id=%s → rq_job=%s", edit_id, job.id
    )


def run_process_audio_patch(patch_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_audio_patch.process_audio_patch`.

    Creates a real :class:`GDriveClient`. Imports the pipeline lazily to match
    the other RQ target wrappers in this module.
    """
    _kcap_begin()
    from .gdrive_client import GDriveClient
    from .pipeline_audio_patch import process_audio_patch

    log.info("run_process_audio_patch: patch_id=%s", patch_id)
    published = process_audio_patch(patch_id, gdrive=GDriveClient())
    if published is not None:
        # Knowledge publication link (AICP-299 §4): the last statement after the
        # promotion's commit; from the descriptor, no DB read; never raises.
        from .knowledge import capture

        capture.record_link_failure(capture.link_publication(published))


def enqueue_process_audio_patch(patch_id: str) -> None:
    """Push :func:`run_process_audio_patch` onto the RQ "process" queue.

    Same queue as tail editing and caption burning. The job is far shorter than
    either (the picture is stream-copied), but it ends by REPLACING a delivered
    creative's bytes on Drive, so it belongs behind the interactive queues with
    the rest of the delivery work rather than ahead of it.
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_process_audio_patch", patch_id,
        job_timeout=AUDIO_PATCH_JOB_TIMEOUT,
    )
    log.info(
        "Enqueued process_audio_patch for patch_id=%s → rq_job=%s", patch_id, job.id
    )


def run_process_caption(caption_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_captions.process_caption`.

    Creates a real :class:`GDriveClient`. Imports the pipeline lazily to match
    the other RQ target wrappers in this module.
    """
    from .gdrive_client import GDriveClient
    from .pipeline_captions import process_caption

    log.info("run_process_caption: caption_id=%s", caption_id)
    process_caption(caption_id, gdrive=GDriveClient())


def enqueue_process_caption(caption_id: str) -> None:
    """Push :func:`run_process_caption` onto the RQ "process" queue.

    Same queue as tail editing (a transcribe + burn + stitch pass, plus a Drive
    upload — the same shape as a tail edit, so it gets the same generous timeout).
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_process_caption", caption_id, job_timeout=CAPTION_JOB_TIMEOUT
    )
    log.info(
        "Enqueued process_caption for caption_id=%s → rq_job=%s", caption_id, job.id
    )


def run_process_segment_image(job_id: str) -> None:
    """
    RQ-callable submit-phase wrapper for one per-segment GPT Image 2 attempt
    (see :mod:`app.pipeline_segment_image`).

    No per-run Redis lock: serialisation is per-SEGMENT, enforced by
    ``api_v2``'s 409 gate (only one active job per RunSegment) rather than the
    run-wide lock — see ``pipeline_segment_image``'s module docstring for why
    taking that lock here would be worse than the race it prevents. Re-enqueues
    the poll phase while a kie.ai task is outstanding; a job already terminal
    (e.g. a duplicate enqueue after it already finished) is a no-op.
    """
    _kcap_begin()
    from .gdrive_client import GDriveClient
    from .pipeline_segment_image import submit_segment_image_job

    log.info("run_process_segment_image: job_id=%s", job_id)
    task_id = submit_segment_image_job(job_id, gdrive=GDriveClient())
    if task_id is None:
        return
    _enqueue_segment_image_poll(job_id, delay_sec=RUN_POLL_INTERVAL_SEC)


def run_process_segment_image_poll(job_id: str) -> None:
    """
    RQ-callable poll-phase wrapper: one round of status-check on a single
    job's kie.ai task, never sleeping. Re-enqueues itself while the result is
    still pending; stops once the job reaches `done` or `failed`.
    """
    _kcap_begin()
    from .pipeline_segment_image import poll_segment_image_job_once

    result = poll_segment_image_job_once(job_id)
    if result == "pending":
        _enqueue_segment_image_poll(job_id, delay_sec=RUN_POLL_INTERVAL_SEC)


def _enqueue_segment_image_poll(job_id: str, *, delay_sec: float) -> None:
    """Schedule run_process_segment_image_poll to execute after delay_sec.

    ``args[0]`` is the job id, never the run id: recovery._live_run_ids treats
    every queued job's first positional arg as an entity id across all
    queues, and this attempt's liveness must be judged against ITS OWN row —
    not the (possibly already `done`) parent run's, which would make a `done`
    run generating a still read as having a live job.
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue_in(
        timedelta(seconds=delay_sec),
        "app.tasks.run_process_segment_image_poll",
        job_id,
        job_timeout=POLL_JOB_TIMEOUT,
    )
    log.info(
        "run_process_segment_image_poll scheduled for job_id=%s in %.0fs → rq_job=%s",
        job_id, delay_sec, job.id,
    )


def enqueue_process_segment_image(job_id: str) -> None:
    """Push :func:`run_process_segment_image` onto the RQ "process" queue.

    QUEUE_PROCESS, not QUEUE_OVERLAY: this blocks on a kie.ai poll chain, the
    same shape as run-level GPT Image 2 preprocessing
    (``enqueue_preprocess_run``) — "overlay" is the fast interactive queue for
    PNG burns and sheet exports, and workers listen [overlay, analyze,
    process], so a blocking kie poll parked there could starve them.
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_process_segment_image", job_id, job_timeout=SUBMIT_JOB_TIMEOUT
    )
    log.info(
        "Enqueued process_segment_image for job_id=%s → rq_job=%s", job_id, job.id
    )


# Run-local segment range regeneration: a cut + video-model submission, then
# (once the model succeeds) a re-cut of the adjacent keep neighbour(s), a
# single-segment stitch and a Drive upload, all inside the poll call that
# observes the model task succeed — the same "the whole finalize phase runs
# synchronously once its gate passes" shape as pipeline_v2.finalize_run, just
# scoped to one candidate. Generous timeout for that reason: unlike segment-
# image regeneration's poll (a single image download), a range attempt's poll
# can also do a full stitch + Drive upload on its last round.
RANGE_ATTEMPT_POLL_JOB_TIMEOUT = int(os.getenv("RANGE_ATTEMPT_POLL_JOB_TIMEOUT", "3600"))


def run_process_range_attempt(attempt_id: str) -> None:
    """
    RQ-callable submit-phase wrapper for one run-local segment range
    regeneration attempt (see :mod:`app.pipeline_range_attempt`).

    No per-run Redis lock — same reasoning as segment-image regeneration:
    serialisation is enforced by api_v2's one-active-attempt-per-run 409 gate.
    Re-enqueues the poll phase while a kie.ai task is outstanding; an attempt
    already terminal (e.g. a duplicate enqueue after it already finished) is a
    no-op.
    """
    _kcap_begin()
    from .pipeline_range_attempt import submit_range_attempt

    log.info("run_process_range_attempt: attempt_id=%s", attempt_id)
    task_id = submit_range_attempt(attempt_id)
    if task_id is None:
        return
    _enqueue_range_attempt_poll(attempt_id, delay_sec=RUN_POLL_INTERVAL_SEC)


def run_process_range_attempt_poll(attempt_id: str) -> None:
    """
    RQ-callable poll-phase wrapper: one round of status-check on a single
    attempt's kie.ai task. On success this round also stitches, delivers and
    PROMOTES (see pipeline_range_attempt.poll_range_attempt_once). Re-enqueues
    itself while the result is still pending; stops once the attempt reaches
    `done` or `failed`.
    """
    _kcap_begin()
    from .pipeline_range_attempt import poll_range_attempt_once_published

    result, published = poll_range_attempt_once_published(attempt_id)
    if result == "pending":
        _enqueue_range_attempt_poll(attempt_id, delay_sec=RUN_POLL_INTERVAL_SEC)
    if published is not None:
        # Knowledge publication link (AICP-299 §4), last; never raises.
        from .knowledge import capture

        capture.record_link_failure(capture.link_publication(published))


def _enqueue_range_attempt_poll(attempt_id: str, *, delay_sec: float) -> None:
    """Schedule run_process_range_attempt_poll to execute after delay_sec.

    ``args[0]`` is the attempt id, never the run id — same reasoning as
    ``_enqueue_segment_image_poll``: recovery._live_run_ids must judge this
    attempt's liveness against ITS OWN row, not the (possibly already `done`)
    parent run's.
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue_in(
        timedelta(seconds=delay_sec),
        "app.tasks.run_process_range_attempt_poll",
        attempt_id,
        job_timeout=RANGE_ATTEMPT_POLL_JOB_TIMEOUT,
    )
    log.info(
        "run_process_range_attempt_poll scheduled for attempt_id=%s in %.0fs → rq_job=%s",
        attempt_id, delay_sec, job.id,
    )


def enqueue_process_range_attempt(attempt_id: str) -> None:
    """Push :func:`run_process_range_attempt` onto the RQ "process" queue.

    QUEUE_PROCESS, not QUEUE_OVERLAY — same reasoning as
    ``enqueue_process_segment_image``: this blocks on a kie.ai poll chain.
    """
    q = _get_queue(QUEUE_PROCESS)
    job = q.enqueue(
        "app.tasks.run_process_range_attempt", attempt_id,
        job_timeout=RANGE_ATTEMPT_POLL_JOB_TIMEOUT,
    )
    log.info(
        "Enqueued process_range_attempt for attempt_id=%s → rq_job=%s",
        attempt_id, job.id,
    )


def enqueue_process_overlay(overlay_job_id: str) -> None:
    """Push :func:`run_process_overlay` onto the RQ "overlay" queue."""
    q = _get_queue(QUEUE_OVERLAY)
    job = q.enqueue(
        "app.tasks.run_process_overlay", overlay_job_id, job_timeout=OVERLAY_JOB_TIMEOUT
    )
    log.info(
        "Enqueued process_overlay for overlay_job_id=%s → rq_job=%s",
        overlay_job_id, job.id,
    )


# ---------------------------------------------------------------------------
# Reframe (AICP-251) — prepare, QA and render on the "process" queue
# ---------------------------------------------------------------------------
# Each job takes its entity id first (recovery's _live_run_ids keys on
# args[0]). The paid Fal work itself runs on the layer's own "fal" queue.


def run_reframe_prepare(job_id: str) -> None:
    from .pipeline_reframe import reframe_prepare

    reframe_prepare(job_id)


def run_reframe_qa(job_id: str) -> None:
    from .pipeline_reframe import reframe_qa

    reframe_qa(job_id)


def run_reframe_render(render_id: str) -> None:
    from .pipeline_reframe import reframe_render

    reframe_render(render_id)


def run_segment_fill_prepare(job_id: str) -> None:
    from .pipeline_segment_fill import segment_fill_prepare

    segment_fill_prepare(job_id)


def run_segment_fill_composite(job_id: str) -> None:
    from .pipeline_segment_fill import segment_fill_composite

    segment_fill_composite(job_id)


def run_segment_fill_batch_promote(batch_id: str) -> None:
    from .pipeline_segment_fill import promote_batch

    promote_batch(batch_id)


# AICP-368: the caption-placement detector scan (app.caption_placement). On the
# analyze queue: it is CPU work (a streamed 2 fps decode of the source), the
# same shape as analysis, and never waits on a provider.
CAPTION_PLACEMENT_SCAN_TIMEOUT = int(os.getenv("CAPTION_PLACEMENT_SCAN_TIMEOUT", "900"))


def run_caption_placement_scan(project_id: str, rev: int) -> None:
    from .caption_placement import scan_and_store

    scan_and_store(project_id, int(rev))


UNCOUNTED_JOB_DEFER_SEC = 120


def defer_uncounted_job(queue_name: str, func: str, *args, job_timeout: int) -> None:
    """Re-run a Drive re-link / caption-placement scan job later because the
    deploy admission gate is paused (ops_gate.job_barrier). A fresh
    RQ job id: the job is idempotent (re-link: row status + run-row lock +
    reserved Drive id; scan: rev compare-and-set)."""
    from datetime import timedelta

    job = _get_queue(queue_name).enqueue_in(
        timedelta(seconds=UNCOUNTED_JOB_DEFER_SEC), func, *args, job_timeout=job_timeout)
    log.info("deploy gate paused: %s%s deferred %ss → rq_job=%s",
             func, args, UNCOUNTED_JOB_DEFER_SEC, job.id)


def enqueue_caption_placement_scan(project_id: str, rev: int) -> None:
    job = _get_queue(QUEUE_ANALYZE).enqueue(
        "app.tasks.run_caption_placement_scan", project_id, int(rev),
        job_timeout=CAPTION_PLACEMENT_SCAN_TIMEOUT,
    )
    log.info("Enqueued caption_placement_scan for project %s rev %s → rq_job=%s",
             project_id, rev, job.id)


def enqueue_segment_fill_prepare(job_id: str) -> None:
    job = _get_queue(QUEUE_PROCESS).enqueue(
        "app.tasks.run_segment_fill_prepare", job_id, job_timeout=settings.SEGMENT_FILL_JOB_TIMEOUT
    )
    log.info("Enqueued segment_fill_prepare for fill %s → rq_job=%s", job_id, job.id)


def enqueue_segment_fill_composite(job_id: str) -> None:
    job = _get_queue(QUEUE_PROCESS).enqueue(
        "app.tasks.run_segment_fill_composite", job_id, job_timeout=settings.SEGMENT_FILL_JOB_TIMEOUT
    )
    log.info("Enqueued segment_fill_composite for fill %s → rq_job=%s", job_id, job.id)


def enqueue_segment_fill_batch_promote(batch_id: str) -> None:
    job = _get_queue(QUEUE_PROCESS).enqueue(
        "app.tasks.run_segment_fill_batch_promote", batch_id,
        job_timeout=settings.SEGMENT_FILL_JOB_TIMEOUT
    )
    log.info("Enqueued segment_fill_batch_promote for batch %s → rq_job=%s", batch_id, job.id)


def enqueue_reframe_prepare(job_id: str) -> None:
    job = _get_queue(QUEUE_PROCESS).enqueue(
        "app.tasks.run_reframe_prepare", job_id, job_timeout=settings.REFRAME_PREP_JOB_TIMEOUT
    )
    log.info("Enqueued reframe_prepare for job %s → rq_job=%s", job_id, job.id)


def enqueue_reframe_qa(job_id: str) -> None:
    job = _get_queue(QUEUE_PROCESS).enqueue(
        "app.tasks.run_reframe_qa", job_id, job_timeout=settings.REFRAME_QA_JOB_TIMEOUT
    )
    log.info("Enqueued reframe_qa for job %s → rq_job=%s", job_id, job.id)


def enqueue_reframe_render(render_id: str) -> None:
    job = _get_queue(QUEUE_PROCESS).enqueue(
        "app.tasks.run_reframe_render", render_id, job_timeout=settings.REFRAME_RENDER_JOB_TIMEOUT
    )
    log.info("Enqueued reframe_render for render %s → rq_job=%s", render_id, job.id)


# ---------------------------------------------------------------------------
# Final Creatives sheet export — RQ tasks (#23)
# ---------------------------------------------------------------------------
# Moving the 3-round-trip Sheets I/O (get_header + column_values + append_row)
# out of the request path onto the overlay RQ queue (fast, interactive jobs).
# The endpoints optimistically set *_exported_at before enqueueing so the DB
# fast-path ("already exported — skip sheet") and UI state work correctly even
# before the job runs.  The job itself performs the dedupe + append; if the row
# already exists it logs and exits cleanly.


def run_export_run_to_sheet(run_id: str) -> None:
    """RQ-callable: append a Final Creatives row for a completed Run.

    Performs the 3-round-trip Sheets flow (get_header → column_values dedupe →
    append_row) in the worker so the API response is not blocked by network I/O.
    """
    from datetime import datetime, timezone

    from .db import SessionLocal
    from .models import Run
    from .sheets_client import (
        SheetsClient,
        SheetsError,
        build_final_creatives_row,
        build_idea,
    )
    from .config import settings
    from datetime import date

    _PIPELINE_TYPE = "re-skin-v2"
    _RUN_ON_META = "need_to_rename"

    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if run is None:
            log.warning("run_export_run_to_sheet: run %s not found", run_id)
            return
        if not run.result_gdrive_file_id:
            log.warning("run_export_run_to_sheet: run %s has no Drive result", run_id)
            return

        project = run.project
        # 8-char prefix + language suffix (``1736426f-ja``) — see Run.short_id.
        job_id = run.short_id
        creative_link = (
            f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
        )
        fields = {
            "Date": date.today().isoformat(),
            "Job ID": job_id,
            "Input Link": project.source_ref if project else "",
            "Creative Link": creative_link,
            "Pipeline Type": _PIPELINE_TYPE,
            "Run on Meta": _RUN_ON_META,
            "Idea": build_idea(
                project.name if project else None,
                run.name,
                run.model,
                project_id=project.id if project else "",
                run_id=run.id,
            ),
        }

    sid = settings.FINAL_CREATIVES_SPREADSHEET_ID
    tab = settings.FINAL_CREATIVES_TAB
    client = SheetsClient()
    try:
        header = client.get_header(sid, tab)
        if not header:
            raise SheetsError(f"tab {tab!r} has no header row")
        norm = {h.strip().lower(): i for i, h in enumerate(header)}
        job_idx = norm.get("job id")
        if job_idx is not None:
            existing = client.column_values(sid, tab, job_idx, last_n=500)
            if any(v.strip() == job_id for v in existing):
                log.info(
                    "run_export_run_to_sheet: run %s already in sheet — skipping append",
                    run_id,
                )
                return
        row = build_final_creatives_row(header, fields=fields)
        client.append_row(sid, tab, row)
        log.info("run_export_run_to_sheet: appended row for run %s", run_id)
    except SheetsError as exc:
        log.error("run_export_run_to_sheet: Sheets error for run %s: %s", run_id, exc)
        raise


def run_export_overlay_to_sheet(overlay_job_id: str) -> None:
    """RQ-callable: append a Final Creatives row for a completed OverlayJob."""
    from datetime import date

    from . import overlay_runs
    from .db import SessionLocal
    from .models import OverlayJob, Run, RunCaption, RunTailEdit
    from .sheets_client import (
        SheetsClient,
        SheetsError,
        build_final_creatives_row,
        build_idea,
    )
    from .config import settings

    _PIPELINE_TYPE = "re-skin-overlay"
    _RUN_ON_META = "need_to_rename"

    with SessionLocal() as db:
        job = db.get(OverlayJob, overlay_job_id)
        if job is None:
            log.warning("run_export_overlay_to_sheet: overlay %s not found", overlay_job_id)
            return
        if not job.gdrive_file_id:
            log.warning(
                "run_export_overlay_to_sheet: overlay %s has no Drive result", overlay_job_id
            )
            return

        creative_link = f"https://drive.google.com/file/d/{job.gdrive_file_id}/view"
        input_link = ""
        # Overlays carry the Idea of the run they were built from — directly
        # (source_run_id) or through the tail edit that produced their input.
        parent_run = None
        if job.source_run_id:
            run = db.get(Run, job.source_run_id)
            parent_run = run
            if run and run.result_gdrive_file_id:
                input_link = (
                    f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
                )
            else:
                input_link = job.source_run_id
        elif job.source_tail_edit_id:
            edit = db.get(RunTailEdit, job.source_tail_edit_id)
            parent_run = edit.run if edit else None
            if edit and edit.gdrive_file_id:
                input_link = (
                    f"https://drive.google.com/file/d/{edit.gdrive_file_id}/view"
                )
            else:
                input_link = job.source_tail_edit_id
        elif job.source_caption_id:
            caption = db.get(RunCaption, job.source_caption_id)
            parent_run = caption.run if caption else None
            if caption and caption.gdrive_file_id:
                input_link = (
                    f"https://drive.google.com/file/d/{caption.gdrive_file_id}/view"
                )
            else:
                input_link = job.source_caption_id
        project = parent_run.project if parent_run else None
        # The PARENT RUN's id — an overlay is that run's creative re-dressed,
        # so it is tracked under the same id, told apart from the run's own row
        # by Pipeline Type. Several overlays on one run each get the parent id
        # plus their own prefix. See overlay_runs.sheet_job_id.
        job_short = overlay_runs.sheet_job_id(
            job,
            parent_run,
            sibling_count=(
                overlay_runs.count_overlay_jobs_for_run(parent_run.id, db)
                if parent_run
                else 0
            ),
        )
        fields = {
            "Date": date.today().isoformat(),
            "Job ID": job_short,
            "Input Link": input_link,
            "Creative Link": creative_link,
            "Pipeline Type": _PIPELINE_TYPE,
            "Run on Meta": _RUN_ON_META,
            "Idea": build_idea(
                project.name if project else None,
                parent_run.name if parent_run else None,
                parent_run.model if parent_run else None,
                project_id=project.id if project else "",
                run_id=parent_run.id if parent_run else job.id,
            ),
        }

    sid = settings.FINAL_CREATIVES_SPREADSHEET_ID
    tab = settings.FINAL_CREATIVES_TAB
    client = SheetsClient()
    try:
        header = client.get_header(sid, tab)
        if not header:
            raise SheetsError(f"tab {tab!r} has no header row")
        norm = {h.strip().lower(): i for i, h in enumerate(header)}
        # Dedupe on Creative Link, NOT Job ID like the other exports: this
        # overlay's Job ID is deliberately its parent run's, so the run's own
        # row (and a sibling overlay's) would look like a duplicate and the
        # append would be silently skipped. The Drive link is unique per
        # rendered creative, so it answers the question the dedupe is actually
        # asking — "did I already write THIS video?" — which is what makes a
        # retried RQ job idempotent. Falls back to the Job ID scan on a sheet
        # with no Creative Link column.
        dedupe_idx, dedupe_val = norm.get("creative link"), creative_link
        if dedupe_idx is None:
            dedupe_idx, dedupe_val = norm.get("job id"), job_short
        if dedupe_idx is not None:
            existing = client.column_values(sid, tab, dedupe_idx, last_n=500)
            if any(v.strip() == dedupe_val for v in existing):
                log.info(
                    "run_export_overlay_to_sheet: overlay %s already in sheet — skipping",
                    overlay_job_id,
                )
                return
        row = build_final_creatives_row(header, fields=fields)
        client.append_row(sid, tab, row)
        log.info(
            "run_export_overlay_to_sheet: appended row for overlay %s", overlay_job_id
        )
    except SheetsError as exc:
        log.error(
            "run_export_overlay_to_sheet: Sheets error for overlay %s: %s",
            overlay_job_id, exc,
        )
        raise


def run_export_tail_edit_to_sheet(edit_id: str) -> None:
    """RQ-callable: append a Final Creatives row for a completed RunTailEdit.

    Mirrors ``run_export_overlay_to_sheet`` (the existing precedent for a
    derived creative): Job ID is the tail edit's OWN 8-char prefix (the parent
    run's prefix would collide with the run's own row — and with sibling tail
    edits — in the sheet dedupe) suffixed with the PARENT RUN's language;
    Input Link is the parent run's creative Drive link; Creative Link is the
    tail edit's own Drive result.
    """
    from datetime import date

    from .db import SessionLocal
    from .models import RunTailEdit
    from .sheets_client import (
        SheetsClient,
        SheetsError,
        build_final_creatives_row,
        build_idea,
    )
    from .config import settings

    _PIPELINE_TYPE = "re-skin-tail-edit"
    _RUN_ON_META = "need_to_rename"

    with SessionLocal() as db:
        edit = db.get(RunTailEdit, edit_id)
        if edit is None:
            log.warning("run_export_tail_edit_to_sheet: tail edit %s not found", edit_id)
            return
        if not edit.gdrive_file_id:
            log.warning(
                "run_export_tail_edit_to_sheet: tail edit %s has no Drive result", edit_id
            )
            return

        run = edit.run
        # Own prefix, parent run's language — same reasoning as the overlay
        # export above.
        job_id = languages.job_id_for(edit.id, run.language if run else None)
        creative_link = f"https://drive.google.com/file/d/{edit.gdrive_file_id}/view"
        if run and run.result_gdrive_file_id:
            input_link = (
                f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
            )
        else:
            input_link = edit.run_id
        # Idea intentionally repeats the PARENT run's value: a tail edit is the
        # same creative idea, re-cut. Sharing the key groups them downstream.
        project = run.project if run else None
        fields = {
            "Date": date.today().isoformat(),
            "Job ID": job_id,
            "Input Link": input_link,
            "Creative Link": creative_link,
            "Pipeline Type": _PIPELINE_TYPE,
            "Run on Meta": _RUN_ON_META,
            "Idea": build_idea(
                project.name if project else None,
                run.name if run else None,
                run.model if run else None,
                project_id=project.id if project else "",
                run_id=edit.run_id,
            ),
        }

    sid = settings.FINAL_CREATIVES_SPREADSHEET_ID
    tab = settings.FINAL_CREATIVES_TAB
    client = SheetsClient()
    try:
        header = client.get_header(sid, tab)
        if not header:
            raise SheetsError(f"tab {tab!r} has no header row")
        norm = {h.strip().lower(): i for i, h in enumerate(header)}
        job_idx = norm.get("job id")
        if job_idx is not None:
            existing = client.column_values(sid, tab, job_idx, last_n=500)
            if any(v.strip() == job_id for v in existing):
                log.info(
                    "run_export_tail_edit_to_sheet: tail edit %s already in sheet — "
                    "skipping append",
                    edit_id,
                )
                return
        row = build_final_creatives_row(header, fields=fields)
        client.append_row(sid, tab, row)
        log.info("run_export_tail_edit_to_sheet: appended row for tail edit %s", edit_id)
    except SheetsError as exc:
        log.error(
            "run_export_tail_edit_to_sheet: Sheets error for tail edit %s: %s", edit_id, exc
        )
        raise


def run_export_caption_to_sheet(caption_id: str) -> None:
    """RQ-callable: append a Final Creatives row for a completed RunCaption.

    Mirrors ``run_export_tail_edit_to_sheet``: Job ID is the caption's OWN
    8-char prefix suffixed with the PARENT RUN's language; Input Link is the
    parent run's creative Drive link; Creative Link is the caption's own Drive
    result. Idea repeats the parent run's value (a captioned cut is the same
    creative idea).
    """
    from datetime import date

    from .db import SessionLocal
    from .models import RunCaption
    from .sheets_client import (
        SheetsClient,
        SheetsError,
        build_final_creatives_row,
        build_idea,
    )
    from .config import settings

    _PIPELINE_TYPE = "re-skin-captions"
    _RUN_ON_META = "need_to_rename"

    with SessionLocal() as db:
        caption = db.get(RunCaption, caption_id)
        if caption is None:
            log.warning("run_export_caption_to_sheet: caption %s not found", caption_id)
            return
        if not caption.gdrive_file_id:
            log.warning(
                "run_export_caption_to_sheet: caption %s has no Drive result", caption_id
            )
            return

        run = caption.run
        # Own prefix, parent run's language — same reasoning as the tail edit
        # export above.
        job_id = languages.job_id_for(caption.id, run.language if run else None)
        creative_link = f"https://drive.google.com/file/d/{caption.gdrive_file_id}/view"
        if run and run.result_gdrive_file_id:
            input_link = (
                f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
            )
        else:
            input_link = caption.run_id
        # Idea intentionally repeats the PARENT run's value: a captioned cut is the
        # same creative idea. Sharing the key groups them downstream.
        project = run.project if run else None
        fields = {
            "Date": date.today().isoformat(),
            "Job ID": job_id,
            "Input Link": input_link,
            "Creative Link": creative_link,
            "Pipeline Type": _PIPELINE_TYPE,
            "Run on Meta": _RUN_ON_META,
            "Idea": build_idea(
                project.name if project else None,
                run.name if run else None,
                run.model if run else None,
                project_id=project.id if project else "",
                run_id=caption.run_id,
            ),
        }

    sid = settings.FINAL_CREATIVES_SPREADSHEET_ID
    tab = settings.FINAL_CREATIVES_TAB
    client = SheetsClient()
    try:
        header = client.get_header(sid, tab)
        if not header:
            raise SheetsError(f"tab {tab!r} has no header row")
        norm = {h.strip().lower(): i for i, h in enumerate(header)}
        job_idx = norm.get("job id")
        if job_idx is not None:
            existing = client.column_values(sid, tab, job_idx, last_n=500)
            if any(v.strip() == job_id for v in existing):
                log.info(
                    "run_export_caption_to_sheet: caption %s already in sheet — "
                    "skipping append",
                    caption_id,
                )
                return
        row = build_final_creatives_row(header, fields=fields)
        client.append_row(sid, tab, row)
        log.info("run_export_caption_to_sheet: appended row for caption %s", caption_id)
    except SheetsError as exc:
        log.error(
            "run_export_caption_to_sheet: Sheets error for caption %s: %s", caption_id, exc
        )
        raise


# Short timeout: only Sheets I/O (3 HTTP round-trips, no heavy computation).
_EXPORT_JOB_TIMEOUT = int(os.getenv("EXPORT_JOB_TIMEOUT", "120"))  # 2 min

# Sheets is a third-party API that returns transient 5xx/429s under load, and a
# single one used to destroy an export: on 2026-08-18 a 503 on the header read
# killed run 59f29a5c's Final Creatives job, the row was never written, and the
# UI kept claiming it had been sent (the stamp is written before the job runs —
# see :func:`clear_export_stamp_on_final_failure`).  So retry with a widening
# gap and let a Google blip heal itself.
#
# Retrying cannot double-write: every export dedupes against the sheet before
# appending (Job ID, or Creative Link for overlays, whose Job ID is deliberately
# their parent run's), which also covers the nastier case where the append
# succeeded server-side but its response was lost.
#
# ``len(intervals) == max`` on purpose — RQ indexes the list by
# ``len(intervals) - retries_left`` (:meth:`rq.job.Job.get_retry_interval`), so
# these are consumed in order: 30s, then 2min, then 5min.  4 attempts total
# (the first one plus 3 retries) spanning ~7.5 min.  The delays need the RQ
# scheduler, which worker/worker.py starts (``work(with_scheduler=True)``).
_EXPORT_RETRY_MAX = 3
_EXPORT_RETRY_INTERVALS = [30, 120, 300]


def clear_export_stamp_on_final_failure(job, connection, exc_type, exc_value, traceback) -> None:
    """RQ ``on_failure`` hook: undo the optimistic export stamp when a job dies.

    The export endpoints stamp the entity (``final_creatives_exported_at``)
    and commit BEFORE enqueuing the job, so that the UI reflects the click
    immediately and a double-click is rejected while the job is still
    queued.  The cost is that a job which never succeeds
    leaves the UI asserting a sheet row that does not exist, with nothing to
    correct it — the operator sees "sent" forever and the creative silently
    never reaches the spreadsheet.  Clearing the stamp on the last attempt puts
    the button back so they can send it again.

    Only the FINAL attempt may clear it.  RQ runs this callback after *every*
    failed attempt (``Worker.perform_job``), before ``handle_job_failure``
    decides whether to retry and before ``Job.retry`` decrements
    ``retries_left`` — so a truthy ``retries_left`` here means another attempt
    is still coming and the stamp must stay put.  ``0`` (retries exhausted) or
    ``None`` (no ``Retry`` configured) means this was the last one.

    Which field to clear travels in ``job.meta`` rather than being derived from
    the task name, so renaming a task cannot silently disable the repair.

    Never raises: RQ re-raises a failing callback and would overwrite the real
    exception in the failed registry with ours, hiding why the export died.
    """
    if job.retries_left:
        return

    meta = job.meta or {}
    model_name = meta.get("export_model")
    field = meta.get("export_stamp")
    entity_id = job.args[0] if job.args else None
    if not (model_name and field and entity_id):
        return

    try:
        from . import models as models_mod
        from .db import SessionLocal

        model = getattr(models_mod, model_name, None)
        if model is None:
            log.error(
                "clear_export_stamp: unknown model %r in job %s meta", model_name, job.id
            )
            return

        with SessionLocal() as db:
            entity = db.get(model, entity_id)
            if entity is None:
                return
            if getattr(entity, field, None) is None:
                return  # already cleared (manual fix, or a second failed job)
            setattr(entity, field, None)
            db.commit()
    except Exception:  # noqa: BLE001 — a broken repair must not mask the cause
        log.exception(
            "clear_export_stamp: failed to clear %s.%s for %s",
            model_name, field, entity_id,
        )
        return

    log.error(
        "Export %s gave up on %s %s after %s attempts — cleared %s.%s so the "
        "operator can send it again. Last error: %s: %s",
        job.func_name, model_name, entity_id, _EXPORT_RETRY_MAX + 1,
        model_name, field,
        getattr(exc_type, "__name__", exc_type), exc_value,
    )


class ExportEnqueueError(Exception):
    """Raised when an export job could not be handed to RQ at all.

    Distinct from a job that ran and failed (that path is covered by
    :func:`clear_export_stamp_on_final_failure`): here nothing was ever queued,
    so no callback exists to repair anything. ``main.py`` maps this to a 503 so
    the operator gets a real message instead of a bare 500.
    """


def enqueue_export_or_rollback(enqueue, entity_id: str, entity, field: str, db) -> None:
    """Enqueue an export, undoing the optimistic stamp if the enqueue itself fails.

    The export endpoints stamp *entity*.*field* and commit BEFORE queuing the
    job, so the UI reflects the click immediately.  If the enqueue then throws —
    Redis down or unreachable — the stamp is already durable and there is no job
    in existence to carry :func:`clear_export_stamp_on_final_failure`, so the UI
    would claim a spreadsheet row forever with nothing able to correct it.  This
    is the one gap the retry/callback pair cannot close from the worker side, so
    it is closed here at the point of the write.

    Rolling back is safe: the enqueue either queued the job or it did not, and
    if it somehow did (the job landed but the call raised afterwards), the
    export's own Job-ID dedupe keeps the retry from writing a second row.

    Raises :class:`ExportEnqueueError` after a successful rollback.  If the
    rollback ITSELF fails, that database error surfaces instead (with the
    enqueue failure chained onto it as ``__context__``) — deliberately NOT
    ``ExportEnqueueError``, because the stamp is still set and telling the
    operator to "try again" would be a second lie on top of the first.
    """
    try:
        enqueue(entity_id)
    except Exception as exc:
        try:
            setattr(entity, field, None)
            db.commit()
        except Exception:
            log.exception(
                "Export enqueue for %s failed AND the %s rollback failed too — the "
                "stamp is stuck set with no job queued",
                entity_id, field,
            )
            raise
        log.exception(
            "Could not queue the export for %s — rolled %s back so the operator "
            "can try again",
            entity_id, field,
        )
        raise ExportEnqueueError(
            "Could not queue the export — the job queue is unreachable. "
            "Nothing was sent; try again."
        ) from exc


def _enqueue_export(
    func_name: str, entity_id: str, *, model: str, stamp: str
) -> str:
    """Enqueue one sheet-export job on the "overlay" (fast) queue.

    Every export shares the same shape: short timeout, retry the transient
    Sheets failures, and on permanent failure clear the optimistic *stamp*
    (an attribute of *model*) that the endpoint wrote before enqueuing.
    Returns the RQ job id for the caller's log line.
    """
    q = _get_queue(QUEUE_OVERLAY)
    job = q.enqueue(
        func_name,
        entity_id,
        job_timeout=_EXPORT_JOB_TIMEOUT,
        retry=Retry(max=_EXPORT_RETRY_MAX, interval=_EXPORT_RETRY_INTERVALS),
        on_failure=Callback(clear_export_stamp_on_final_failure),
        meta={"export_model": model, "export_stamp": stamp},
    )
    return job.id


def enqueue_export_run_to_sheet(run_id: str) -> None:
    """Push :func:`run_export_run_to_sheet` onto the RQ "overlay" (fast) queue."""
    job_id = _enqueue_export(
        "app.tasks.run_export_run_to_sheet",
        run_id,
        model="Run",
        stamp="final_creatives_exported_at",
    )
    log.info(
        "Enqueued export_run_to_sheet for run_id=%s → rq_job=%s", run_id, job_id
    )


def enqueue_export_overlay_to_sheet(overlay_job_id: str) -> None:
    """Push :func:`run_export_overlay_to_sheet` onto the RQ "overlay" (fast) queue."""
    job_id = _enqueue_export(
        "app.tasks.run_export_overlay_to_sheet",
        overlay_job_id,
        model="OverlayJob",
        stamp="final_creatives_exported_at",
    )
    log.info(
        "Enqueued export_overlay_to_sheet for overlay_job_id=%s → rq_job=%s",
        overlay_job_id, job_id,
    )


def enqueue_export_tail_edit_to_sheet(edit_id: str) -> None:
    """Push :func:`run_export_tail_edit_to_sheet` onto the RQ "overlay" (fast) queue."""
    job_id = _enqueue_export(
        "app.tasks.run_export_tail_edit_to_sheet",
        edit_id,
        model="RunTailEdit",
        stamp="final_creatives_exported_at",
    )
    log.info(
        "Enqueued export_tail_edit_to_sheet for edit_id=%s → rq_job=%s", edit_id, job_id
    )


def enqueue_export_caption_to_sheet(caption_id: str) -> None:
    """Push :func:`run_export_caption_to_sheet` onto the RQ "overlay" (fast) queue."""
    job_id = _enqueue_export(
        "app.tasks.run_export_caption_to_sheet",
        caption_id,
        model="RunCaption",
        stamp="final_creatives_exported_at",
    )
    log.info(
        "Enqueued export_caption_to_sheet for caption_id=%s → rq_job=%s", caption_id, job_id
    )


# ---------------------------------------------------------------------------
# Fal provider tasks (AICP-312 Stage 1 / AICP-330) — see app/pipeline_provider.py
# ---------------------------------------------------------------------------

FAL_JOBS = ("fal_submit", "fal_poll", "fal_fetch", "fal_cancel", "fal_close")
# The plain status poll holds no claim, so it needs no long timeout.
FAL_POLL_JOB_TIMEOUT = 300


def enqueue_fal_job(fn: str, task_id: str, *, delay_sec: int = 0) -> None:
    """Enqueue one Fal job on QUEUE_FAL, task id first (recovery's
    _live_run_ids keys on args[0]). Every claim-holding job takes its RQ
    timeout from FAL_JOB_TIMEOUT, which the claim lease outlasts."""
    if fn not in FAL_JOBS:
        raise ValueError(f"unknown Fal job {fn!r}")
    timeout = FAL_POLL_JOB_TIMEOUT if fn == "fal_poll" else settings.FAL_JOB_TIMEOUT
    q = _get_queue(QUEUE_FAL)
    target = f"app.pipeline_provider.{fn}"
    if delay_sec > 0:
        job = q.enqueue_in(timedelta(seconds=delay_sec), target, task_id, job_timeout=timeout)
    else:
        job = q.enqueue(target, task_id, job_timeout=timeout)
    log.info("fal: enqueued %s(%s) in %ss -> rq_job=%s", fn, task_id, delay_sec, job.id)


# ---------------------------------------------------------------------------
# AICP-348 — re-link a run whose delivered Drive file was deleted
# ---------------------------------------------------------------------------

DRIVE_RELINK_JOB_TIMEOUT = 1800


def drive_relink_job_id(relink_id: str) -> str:
    """Deterministic RQ job id, so admission can tell a live job from a dead one."""
    return f"drive-relink-{relink_id}"


def run_drive_relink(relink_id: str) -> None:
    """RQ-callable wrapper for :func:`app.drive_relink.process_relink`."""
    from .drive_relink import process_relink

    log.info("run_drive_relink: relink_id=%s", relink_id)
    status = process_relink(relink_id)
    log.info("run_drive_relink: relink_id=%s -> %s", relink_id, status)


def drive_relink_job_live(relink_id: str) -> bool:
    """True while this re-link's RQ job is queued, scheduled or running.

    Raises on a Redis error: the caller must not read "could not tell" as
    "dead" and start a second job for the same row.
    """
    from rq.exceptions import NoSuchJobError
    from rq.job import Job

    try:
        job = Job.fetch(drive_relink_job_id(relink_id), connection=_get_redis())
    except NoSuchJobError:
        return False
    status = job.get_status(refresh=True)
    status = getattr(status, "value", status)
    return status in ("queued", "started", "deferred", "scheduled")


def enqueue_drive_relink(relink_id: str) -> None:
    """Push :func:`run_drive_relink` onto the "overlay" (fast) queue."""
    q = _get_queue(QUEUE_OVERLAY)
    job = q.enqueue(
        "app.tasks.run_drive_relink", relink_id,
        job_id=drive_relink_job_id(relink_id),
        job_timeout=DRIVE_RELINK_JOB_TIMEOUT,
    )
    log.info("Enqueued drive relink %s → rq_job=%s", relink_id, job.id)
