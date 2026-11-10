"""
recovery.py — Startup orphaned-run reconciliation (TR5b).

Called ONCE by the worker bootstrap (worker/worker.py) BEFORE the worker begins
listening for new jobs.  It finds runs that are stuck in an active state with no
live RQ job and re-enqueues them so they proceed automatically.

Race-safety design (per-run liveness)
-------------------------------------
With N parallel workers, a run may appear "stuck" in `processing` while it is
actually being worked on by another worker right now.  We must NOT re-enqueue
such a run (that would double-process it).

We decide **per run**, not with a single global gate:

    A run is reconciled only if it has NO genuinely-live RQ job — i.e. no job
    that is pending, scheduled, or started on a worker whose heartbeat is fresh.

See :func:`_live_run_ids`.  This replaces the earlier global "is the whole queue
idle" gate, which had a fatal flaw exposed in production on 2026-07-10:

- The old gate skipped ALL reconciliation if the StartedJobRegistry was
  non-empty.  But when a redeploy kills workers mid-job, RQ does NOT remove the
  in-flight jobs from the StartedJobRegistry immediately — they linger there as
  "started" until their heartbeat-TTL lapses and RQ reaps them as
  AbandonedJobError (which can be many minutes later).  So at the exact moment
  the restarted worker booted, its own just-abandoned jobs made the registry
  look "busy", the global gate tripped, and the three orphaned runs were skipped
  and never re-enqueued (reconciliation runs only once, at boot).
- Two workers booting together each saw the other's started jobs and BOTH
  skipped ("mutual skip"), so nobody reconciled.

Per-run liveness fixes both: an abandoned started job whose owning worker is
dead is NOT counted as live, so the run it references is correctly reconciled;
and each run is judged on its own evidence, so unrelated in-flight work never
blocks recovery of a genuine orphan.

Boot + periodic reaper
----------------------
Reconciliation runs at worker boot AND on a periodic tick (see
``app.tasks.run_reconcile_tick``).  The boot pass alone is not enough: a worker
killed moments before boot still has a fresh last-heartbeat, so its abandoned
`started` jobs briefly look live and their runs are (correctly) left alone at
boot — but nothing would re-check them.  The periodic reaper re-runs after those
stale signals expire (dead worker drops out of ``Worker.all`` / heartbeat goes
stale), so recovery is self-healing within a couple of minutes without ever
risking a double-process.

Requeued after the snapshot (AICP-360)
--------------------------------------
The liveness snapshot is taken once, before the slow parts of the pass. A run
requeued after it (API retry/rerun, a segment-fill promote, a preprocess
approval, an unattended launch) has a job the snapshot never saw; the old pass
judged it orphaned, deleted its per-run lock and enqueued a second submit, so
two workers could process — and bill — the same run. The run pass now never
deletes a run lock: it reads each candidate's (lock token, lock generation),
rescans liveness fail-closed, and takes the lock over only by compare-and-set
on what it read, then hands that lock to the job it enqueues. See
:func:`_recover_orphaned_run` for the argument.

Per-run re-poll before resubmit
---------------------------------
For a run segment stuck in `generating` with a `seedance_task_id`, we call
kie.ai's get_task() endpoint before re-enqueuing.  If that task is already
`success`, we download the result and mark the segment `completed` — avoiding a
redundant Seedance submission and the associated billing cost.

Active states covered
---------------------
- `queued`      — job was never picked up (or lost before pickup)
- `processing`  — worker died during Seedance submit/poll loop
- `stitching`   — worker died during ffmpeg stitch
- `delivering`  — worker died during GDrive upload
  (projects in `analyzing` are also covered — see analyze reconciliation below)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Optional

from redis import Redis
from rq import Queue, Worker
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import ScheduledJobRegistry, StartedJobRegistry
from sqlalchemy import select

from .db import get_session
from .kie_client import KieClient
from .models import (
    InsertMerge,
    OverlayJob,
    RunAudioPatch,
    Run,
    RunCaption,
    RunSegment,
    RunSegmentImageJob,
    RunSegmentRangeAttempt,
    RunTailEdit,
    VideoProject,
)
from .state_machine import (
    CaptionStatus,
    MergeStatus,
    OverlayStatus,
    ProjectStatus,
    RANGE_ATTEMPT_ACTIVE_STATUSES,
    RangeAttemptStatus,
    RunStatus,
    SegmentImageStatus,
    SegmentStatus,
    AudioPatchStatus,
    TailEditStatus,
    transition,
)
from .storage import run_results_dir
from .tasks import (
    _enqueue_range_attempt_poll,
    _refresh_lock,
    _release_lock,
    claim_stale_run_lock,
    enqueue_preprocess_run,
    enqueue_process_run,
    read_run_lock_states,
    run_lock_key,
)

log = logging.getLogger(__name__)

# Active run states — any run in these states should have a live RQ job.
_ACTIVE_RUN_STATUSES = {
    RunStatus.preprocessing,
    RunStatus.queued,
    RunStatus.processing,
    RunStatus.stitching,
    RunStatus.delivering,
}

# Active project states — any project in these states should have a live RQ job.
_ACTIVE_PROJECT_STATUSES = {ProjectStatus.analyzing}


# A `started` job's owning worker is treated as alive only if it is registered
# AND heartbeated within this window.  A busy worker (the only kind that owns a
# started job) heartbeats every ~job_monitoring_interval (30s in RQ 1.15), so
# 150s is a ~5x safety margin: a live worker is never falsely judged dead (which
# would risk double-processing), while a worker killed by a redeploy/OOM drops
# below the threshold within ~150s so its abandoned jobs stop masking the
# orphaned run they reference — and the periodic reaper then recovers it.
_WORKER_HEARTBEAT_FRESH_SEC = 150


def _run_id_from_job(job: Job, *, strict: bool = False) -> Optional[str]:
    """Return the entity id (run_id / project_id / overlay id) an RQ job targets.

    Every queue entrypoint (run_submit_run, run_poll_run, run_finalize_run,
    run_analyze_project, run_process_overlay) takes that id as its first
    positional argument, so ``job.args[0]`` identifies the entity the job works
    on.  Returns None for jobs with no positional args (e.g. the reconcile tick).
    *strict* (AICP-360): a job whose args cannot be decoded raises instead of
    reading as "targets nothing" — it may well be a live job of the run.
    """
    try:
        args = job.args
    except Exception:
        if strict:
            raise
        return None
    if args and isinstance(args[0], str):
        return args[0]
    return None


def _live_worker_names(redis_conn: Redis, *, strict: bool = False) -> set[str]:
    """Names of workers that are registered AND heartbeated recently.

    Membership in ``Worker.all()`` alone is not proof of liveness right after a
    restart: a worker killed by a redeploy stops heartbeating but its registry
    key lingers until the ~420s worker-TTL expires.  Requiring a fresh heartbeat
    makes a just-killed worker count as dead within _WORKER_HEARTBEAT_FRESH_SEC.

    *strict* (AICP-360): raise instead of answering "no live workers" when the
    worker list cannot be read — that answer would make every started job look
    abandoned.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=_WORKER_HEARTBEAT_FRESH_SEC)
    live: set[str] = set()
    try:
        workers = Worker.all(connection=redis_conn)
    except Exception as exc:  # pragma: no cover - defensive
        if strict:
            raise
        log.warning("Worker.all() failed during liveness check: %s", exc)
        return live
    for w in workers:
        hb = getattr(w, "last_heartbeat", None)
        if hb is not None and hb >= cutoff:
            live.add(w.name)
    return live


def _live_run_ids(
    redis_conn: Redis, queue_names: Optional[list[str]] = None, *, strict: bool = False,
) -> set[str]:
    """Entity ids that currently have a *genuinely live* RQ job.

    An id is live if it has a job that is:
      * pending in a queue (any worker will pick it up), OR
      * scheduled (the RQScheduler will move it to a queue when due — e.g. the
        15s poll re-schedule), OR
      * started on a worker whose heartbeat is fresh (see _live_worker_names).

    A started job whose owning worker is dead (redeploy/OOM) is NOT live — it is
    abandoned, and the run it references is a genuine orphan.  This is the crux
    of the fix: the previous global "is any job in flight" gate counted an
    abandoned-but-not-yet-reaped started job as activity and skipped ALL
    reconciliation, stranding the orphaned runs (prod incident 2026-07-10).

    Checks "overlay", "analyze", "process" and the legacy "default" queue, or
    only *queue_names* when given (the Fal recovery pass passes ["fal"]).

    *strict* (AICP-360) makes the scan fail closed: any read error raises
    instead of silently dropping the ids it could not read. The run pass uses
    it for the scan that authorises taking over a run's lock, where a dropped
    id would mean stealing a live job's lock.
    """
    from .tasks import QUEUE_OVERLAY, QUEUE_ANALYZE, QUEUE_PROCESS

    if queue_names is None:
        queue_names = [QUEUE_OVERLAY, QUEUE_ANALYZE, QUEUE_PROCESS, "default"]
    live_workers = (
        _live_worker_names(redis_conn, strict=True) if strict
        else _live_worker_names(redis_conn)
    )
    live_ids: set[str] = set()

    def _collect(job_ids, *, require_live_worker: bool) -> None:
        job_ids = list(job_ids)
        if not job_ids:
            return
        # Job.fetch_many pipelines all HGETALLs into one Redis round-trip
        # (vs. one Job.fetch() round-trip per id) and returns None in place
        # of any job hash that vanished between listing and fetch
        # (finished/expired) rather than raising — same case the old
        # per-id try/except was guarding.
        try:
            jobs = Job.fetch_many(job_ids, connection=redis_conn)
        except Exception as exc:
            # A whole-batch failure (e.g. a Redis hiccup) must not crash the
            # reconcile tick — degrade to per-id fetch for this batch so a
            # transient pipeline error doesn't drop the whole queue's ids.
            log.warning(
                "Job.fetch_many failed for %d id(s), falling back to per-id "
                "fetch: %s", len(job_ids), exc,
            )
            jobs = []
            for jid in job_ids:
                try:
                    jobs.append(Job.fetch(jid, connection=redis_conn))
                except NoSuchJobError:
                    jobs.append(None)  # genuinely gone (finished/expired)
                except Exception:
                    if strict:
                        raise
                    jobs.append(None)
        for job in jobs:
            if job is None:
                # Job hash vanished between listing and fetch (finished/expired).
                continue
            if require_live_worker and getattr(job, "worker_name", None) not in live_workers:
                continue
            rid = _run_id_from_job(job, strict=strict)
            if rid:
                live_ids.add(rid)

    for name in queue_names:
        q = Queue(name, connection=redis_conn)
        _collect(q.job_ids, require_live_worker=False)              # pending
        sched = ScheduledJobRegistry(name, connection=redis_conn)
        _collect(sched.get_job_ids(), require_live_worker=False)    # scheduled
        started = StartedJobRegistry(name, connection=redis_conn)
        try:
            started.cleanup()  # reap past-TTL abandoned jobs → FailedJobRegistry
        except Exception:  # pragma: no cover - best-effort
            pass
        _collect(started.get_job_ids(), require_live_worker=True)   # started

    log.info("reconcile_orphaned_runs: %d entity id(s) have a live job", len(live_ids))
    return live_ids


# ---------------------------------------------------------------------------
# Derived-creative reaping (captions, tail edits, overlays, PIP merges)
# ---------------------------------------------------------------------------
#
# Runs and projects are *resumed* above. Derived creatives are instead marked
# `failed`, deliberately:
#
# - Resuming a caption would re-bill a gemini-2.5-pro transcription and deliver
#   a second file to Drive without anyone asking. That is real money and a real
#   artifact, so the operator decides — same reasoning that leaves an orphaned
#   `analyzing` project at `failed` rather than auto-re-analyzing.
# - Every one of these already has a retry path (`failed → queued`), so a
#   reaped job is one click from running again, with its inputs intact.
#
# Without this, a job interrupted mid-flight keeps its active status forever:
# nothing else ever revisits the row, so the UI shows a spinner that will never
# resolve. That is exactly what happened to caption fbd2d5fd on 2026-08-11 —
# a deploy recreated the workers ~1 minute into its transcription, and it sat
# in `transcribing` until reaped by hand.


class _DerivedCreativeSpec(NamedTuple):
    """How to find and fail one kind of derived-creative job."""
    model: type
    label: str
    active: set          # statuses that imply "a worker should be on this"
    failed_status: object
    error_attr: str      # OverlayJob calls it `error`; the rest `error_message`


# `OverlayStatus.created` is deliberately absent: it is the pre-queue state of a
# job the operator is still composing in the editor. It has no RQ job by design,
# so reaping it would delete in-progress work rather than recover it.
_DERIVED_CREATIVE_SPECS: tuple[_DerivedCreativeSpec, ...] = (
    _DerivedCreativeSpec(
        RunCaption, "caption",
        {CaptionStatus.queued, CaptionStatus.transcribing,
         CaptionStatus.burning, CaptionStatus.delivering},
        CaptionStatus.failed, "error_message",
    ),
    _DerivedCreativeSpec(
        RunTailEdit, "tail edit",
        {TailEditStatus.queued, TailEditStatus.processing,
         TailEditStatus.delivering},
        TailEditStatus.failed, "error_message",
    ),
    # Audio patch (app/pipeline_audio_patch.py). Not a derived creative — it
    # repairs the run's own result — but the stuck-forever failure mode is
    # identical, and here it also holds the DB-level one-active-patch-per-run
    # index, so an unreaped row blocks the operator from ever retrying the
    # correction. `done` is terminal and deliberately absent: a promoted patch
    # must never be reopened.
    _DerivedCreativeSpec(
        RunAudioPatch, "audio patch",
        {AudioPatchStatus.queued, AudioPatchStatus.processing,
         AudioPatchStatus.delivering},
        AudioPatchStatus.failed, "error_message",
    ),
    _DerivedCreativeSpec(
        InsertMerge, "merge",
        {MergeStatus.queued, MergeStatus.compositing, MergeStatus.delivering},
        MergeStatus.failed, "error_message",
    ),
    _DerivedCreativeSpec(
        OverlayJob, "overlay",
        {OverlayStatus.queued, OverlayStatus.processing,
         OverlayStatus.delivering},
        OverlayStatus.failed, "error",
    ),
    # Per-segment GPT Image 2 regeneration (app/pipeline_segment_image.py). A
    # derived INPUT job, not a derived creative (no Drive upload, no sheet
    # row), but the failure mode is identical: a worker that dies after
    # kie.ai has already billed the task leaves the row stuck `generating`
    # forever, the UI spinning, and — because api_v2's 409 gate blocks
    # rerun_segment/patch_run_segment while a job is active on that segment —
    # the operator unable to re-run the segment at all. Reaping unblocks that
    # gate the same way it unblocks a caption's Retry button.
    _DerivedCreativeSpec(
        RunSegmentImageJob, "segment image job",
        {SegmentImageStatus.queued, SegmentImageStatus.generating},
        SegmentImageStatus.failed, "error_message",
    ),
    # Run-local segment range regeneration (app/pipeline_range_attempt.py). A
    # candidate revision, not a derived creative (no Drive-permanent output of
    # its own — a promoted one just becomes the Run's own result), but the
    # failure mode is identical: a worker killed mid-attempt leaves the row
    # stuck in an active status forever, the UI spinning, and — because
    # api_v2's 409 gate blocks a second attempt, Copy, and Automation while
    # one is active on a run — the operator locked out of all three until it
    # is reaped. Reaping never touches the Run/RunSegment: a failed attempt
    # was never promoted, so the run's last successful timeline/result is
    # already exactly what it should be (see RunSegmentRangeAttempt's
    # docstring) — this only unblocks the gate, the same as every other row
    # in this table.
    _DerivedCreativeSpec(
        RunSegmentRangeAttempt, "range attempt",
        RANGE_ATTEMPT_ACTIVE_STATUSES,
        RangeAttemptStatus.failed, "error_message",
    ),
    # Reframe renders (AICP-251) are NOT here: they are reaped by a CAS on the
    # observed status and claim in _reconcile_reframe_jobs, so a render that
    # finished between this pass's read and its commit is never overwritten.
)

# A job is only reaped once it has been untouched for this long. The API commits
# the row *before* it enqueues the RQ job (see create_caption), so for a moment a
# perfectly healthy job is `queued` in Postgres with nothing in Redis yet — and
# `live_ids` is computed before the slow kie.ai pre-poll, so it can be minutes
# stale by the time we get here. Without this window either gap would let the
# reaper fail a job that was about to run normally.
_DERIVED_ORPHAN_GRACE_SEC = 300

_DERIVED_ORPHAN_ERROR = (
    "Interrupted by a worker restart — the job was in flight when its worker "
    "was killed (deploy, crash or OOM) and its RQ job died with it. Nothing was "
    "delivered. Press Retry to run it again."
)


def _orphan_error_message(spec: "_DerivedCreativeSpec", row) -> str:
    """The message to stamp on one orphaned derived-creative row.

    Almost always the generic ``_DERIVED_ORPHAN_ERROR`` above — "Press
    Retry" is safe advice for every kind in this table except one:
    ``RunSegmentRangeAttempt``'s ``submit_billed_at`` can be set (a worker
    committed it right before, or right after, kie.ai's paid ``createTask``
    call) with no ``seedance_task_id`` ever persisted, meaning the crash
    this reaper is cleaning up may have happened in the exact window where
    a billed generation could already exist server-side. This reaper's own
    grace window (``_DERIVED_ORPHAN_GRACE_SEC``, 300s) is SHORTER than
    ``pipeline_range_attempt._CLAIM_LEASE_SEC`` (900s), so for a genuinely
    dead worker THIS reaper — not that module's own expired-claim check —
    is what actually reaps the row in production; it must carry the same
    ambiguous-billing warning that check would have used, or the operator
    sees a plain "Press Retry" and risks a second billed generation (found
    by the mandatory Cursor CLI review gate). No idempotency key or
    lookup-by-reference exists on kie.ai's side to resolve this any more
    precisely than flagging it for manual reconciliation — see
    ``pipeline_range_attempt._ambiguous_submit_error`` and
    ``docs/run-local-segment-range-regeneration.md``.
    """
    if (
        getattr(row, "submit_billed_at", None) is not None
        and not getattr(row, "seedance_task_id", None)
    ):
        from .pipeline_range_attempt import _ambiguous_submit_error

        return _ambiguous_submit_error(row.submit_billed_at)
    return _DERIVED_ORPHAN_ERROR


def _as_naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalise a timestamp so it can be compared with ``datetime.utcnow()``.

    ``models._utcnow`` is timezone-**aware** (``datetime.now(timezone.utc)``)
    while this module compares against naive ``datetime.utcnow()``. Postgres
    hands back naive values from its ``TIMESTAMP WITHOUT TIME ZONE`` columns, but
    a freshly-defaulted in-session object is still aware — and comparing the two
    raises TypeError. Normalise rather than assume either shape.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _reap_orphaned_derived_creatives(live_ids: set[str]) -> dict[str, int]:
    """Mark derived-creative jobs with no live RQ job as `failed` — with ONE
    exception: a `RunSegmentRangeAttempt` whose submission already fully
    succeeded (`seedance_task_id` persisted) is RECOVERED instead of failed
    (see the loop body below and docs/run-local-segment-range-regeneration.md
    §5) — kie.ai's paid generation already happened, so finalizing it
    (download/recut/stitch/deliver/promote) makes no further paid call, and
    failing it would discard an already-paid-for result for nothing (found
    by an independent audit).

    *live_ids* comes from :func:`_live_run_ids` — every queue entrypoint takes
    its entity id as ``args[0]``, so caption/tail-edit/merge/overlay jobs are
    already covered by that sweep (they ride the "process" and "overlay" queues).
    A job that is still pending in Redis counts as live and is left alone, which
    is what keeps a legitimately-`queued` job from being reaped — and, for a
    range attempt specifically, is what stops recovery from ever racing a
    genuinely live RQ worker: its id is never reaped OR recovered while a
    live job for it exists.

    Returns ``{label: count}`` for the jobs actually FAILED (the range
    attempts recovered instead are not counted here — see the log line
    "recovering N orphaned range attempt(s)").
    """
    cutoff = datetime.utcnow() - timedelta(seconds=_DERIVED_ORPHAN_GRACE_SEC)
    reaped: dict[str, int] = {}
    to_resume: list[str] = []

    with get_session() as session:
        for spec in _DERIVED_CREATIVE_SPECS:
            stmt = select(spec.model).where(
                spec.model.status.in_([s.value for s in spec.active])
            )
            try:
                rows = list(session.execute(stmt).scalars())
            except Exception as exc:  # pragma: no cover - defensive
                log.error("reap_derived: query for %s failed: %s", spec.label, exc)
                continue

            for row in rows:
                if row.id in live_ids:
                    continue
                # Soft-deleted overlay jobs are not the operator's problem.
                if getattr(row, "deleted_at", None) is not None:
                    continue
                updated = _as_naive_utc(getattr(row, "updated_at", None))
                if updated is not None and updated > cutoff:
                    continue

                if spec.model is RunSegmentRangeAttempt and getattr(row, "seedance_task_id", None):
                    # Submission already fully succeeded — no paid call is
                    # needed to finish this attempt. Clear the dead claim
                    # TRANSACTIONALLY (in this SAME commit) rather than
                    # leaving it to expire on its own: this reaper's grace
                    # window (5 min) is far shorter than
                    # pipeline_range_attempt._CLAIM_LEASE_SEC (job-timeout-
                    # sized, ~1 hour), so a freshly-enqueued recovery poll
                    # would otherwise see the dead worker's claim as still
                    # "live" and just report "pending" again for up to that
                    # full hour.
                    #
                    # Force status back to `generating` regardless of
                    # whether the crash happened there, or later in
                    # `stitching`/`delivering` — poll_range_attempt_once's
                    # terminal-state handler unconditionally re-runs the
                    # WHOLE download -> recut -> stitch -> deliver ->
                    # promote sequence from scratch (fresh claim, fresh
                    # claim-scoped paths) every time it is invoked; it has
                    # no notion of "resume from stitching" and its own
                    # `transition(attempt, stitching)` call would raise
                    # InvalidTransition if the row were still sitting at
                    # `stitching`/`delivering` (no self-loop edge exists).
                    # Bypassing `transition()` here (not calling it) matches
                    # this table's existing precedent for a forced value
                    # (see the OverlayStatus comment below) — `generating`
                    # is always a valid resting state for a row with a
                    # persisted task id.
                    row.status = RangeAttemptStatus.generating
                    row.claim_owner = None
                    row.claim_expires_at = None
                    to_resume.append(row.id)
                    log.warning(
                        "reap_derived: range attempt %s was orphaned "
                        "(status=%s) with a persisted seedance_task_id — "
                        "claim cleared, recovering finalization instead of "
                        "failing it (no new paid call)", row.id, row.status,
                    )
                    continue

                # Capture BEFORE the transition: the status it was stuck in is
                # the whole diagnostic (`transcribing` says the model call died,
                # `delivering` says the Drive upload did). Logging row.status
                # afterwards just prints "failed" for every row.
                was = row.status
                try:
                    # Most active→failed edges exist, but not all: OverlayStatus
                    # has no `queued → failed`. recovery.py's existing precedent
                    # for a missing edge is to force the value rather than let a
                    # guard strand the row in a state nothing will revisit.
                    try:
                        transition(row, spec.failed_status)
                    except Exception:
                        row.status = spec.failed_status
                    setattr(row, spec.error_attr, _orphan_error_message(spec, row))
                    reaped[spec.label] = reaped.get(spec.label, 0) + 1
                    log.warning(
                        "reap_derived: %s %s was orphaned (status=%s) — marked failed",
                        spec.label, row.id, was,
                    )
                except Exception as exc:
                    log.error(
                        "reap_derived: failed to reap %s %s: %s",
                        spec.label, row.id, exc,
                    )

        try:
            session.commit()
        except Exception as exc:  # pragma: no cover - defensive
            log.error("reap_derived: commit failed: %s", exc)
            session.rollback()
            return {}

    if reaped:
        log.warning(
            "reap_derived: reaped %d orphaned derived-creative job(s): %s",
            sum(reaped.values()), reaped,
        )
    else:
        log.info("reap_derived: no orphaned derived-creative jobs found")

    # Enqueue recovery polls ONLY after the claim-clear above has actually
    # committed — never before, so a commit failure (return {} above) never
    # leaves an enqueue pointing at a claim that was never really cleared.
    # Exactly one poll per recovered attempt; `delay_sec=0` since kie.ai
    # already reported (or will report) a real, already-billed task — there
    # is nothing left to wait out.
    if to_resume:
        log.warning(
            "reap_derived: recovering %d orphaned range attempt(s) for "
            "finalize resume (claim cleared, no new paid call): %s",
            len(to_resume), to_resume,
        )
        for attempt_id in to_resume:
            try:
                _enqueue_range_attempt_poll(attempt_id, delay_sec=0)
            except Exception as exc:
                log.error(
                    "reap_derived: failed to enqueue recovery poll for "
                    "range attempt %s: %s", attempt_id, exc,
                )
    return reaped


class _OwnershipLost(Exception):
    """The pass lease or the run's claim was lost mid-recovery (AICP-360)."""


def _discard(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - best-effort cleanup
        log.warning("Could not remove %s: %s", path, exc)


def _repoll_generating_segments(
    run_id: str, redis_conn: Redis, *, still_owner=None,
) -> None:
    """
    For a run in `processing`, check each `generating` segment's existing
    seedance_task_id against kie.ai before re-enqueuing.

    If the task is already `success` on kie.ai, download its result and mark the
    segment `completed` — so the resumed process_run skips resubmission and avoids
    a redundant Seedance billing.

    This is a best-effort pass: any kie.ai error (network, auth, etc.) is caught
    and logged; the segment stays in its current state and will be re-processed
    normally by process_run's retry reset path.
    """
    with get_session() as session:
        run: Optional[Run] = session.get(Run, run_id)
        if run is None:
            return
        if getattr(run, "generation_mode", "v2v") == "images_only":
            # Images-only runs never hold a seedance_task_id: their provider
            # work is in rebuild_chunk_attempts, re-driven by the rebuild
            # reaper (docs/rebuild-from-reference-design.md §7.3).
            log.error("_repoll_generating_segments refused images-only run %s", run_id)
            return

        generating_segs = [
            rs for rs in run.run_segments
            if rs.status == SegmentStatus.generating and rs.seedance_task_id
        ]
        if not generating_segs:
            return

        log.info(
            "Pre-poll pass for run %s: %d segment(s) in generating with task_id",
            run_id, len(generating_segs),
        )

        try:
            kie = KieClient()
        except Exception as exc:
            log.warning("Cannot create KieClient for pre-poll: %s", exc)
            return

        r_dir = run_results_dir(run_id, run.project_id)
        os.makedirs(r_dir, exist_ok=True)

        def _check_owner(what: str) -> None:
            # AICP-360: re-checked before each provider call and before each
            # write, so a pass that lost its lease or its claim to another pass
            # stops instead of writing segment state beside it.
            if still_owner is not None and not still_owner():
                raise _OwnershipLost(f"run {run_id}: ownership lost before {what}")

        for rs in generating_segs:
            task_id = rs.seedance_task_id
            _check_owner(f"get_task({task_id})")
            try:
                data = kie.get_task(task_id)
            except Exception as exc:
                log.warning("get_task(%s) failed during pre-poll: %s — will resubmit", task_id, exc)
                continue

            state = (data.get("state") or "").lower()
            if state != "success":
                log.info(
                    "Segment %s task %s state=%r — will resubmit (not yet success)",
                    rs.id, task_id, state,
                )
                continue

            # Task is success — extract url and download result to avoid rebilling.
            raw = data.get("resultJson") or "{}"
            try:
                urls = json.loads(raw).get("resultUrls") or []
            except (ValueError, TypeError):
                urls = []

            url = urls[0] if urls else None
            if not url:
                log.warning(
                    "Segment %s task %s success but no result url — will resubmit",
                    rs.id, task_id,
                )
                continue

            result_dst = os.path.join(r_dir, f"result_{rs.index:04d}.mp4")
            _check_owner(f"download of task {task_id}")
            # AICP-360: download to a path private to THIS attempt (its own
            # .part too) and move it into place only after a fresh ownership
            # check. A download still running when this pass lost the run can
            # then never truncate or replace the file a new owner is writing.
            private_dst = f"{result_dst}.recovery-{uuid.uuid4().hex[:12]}"
            try:
                kie.download_result(url, private_dst)
            except Exception as exc:
                log.warning(
                    "download_result failed for task %s: %s — will resubmit", task_id, exc
                )
                _discard(private_dst)
                continue
            try:
                _check_owner(f"completing segment {rs.id}")
            except BaseException:   # lost, or the check itself failed (Redis)
                _discard(private_dst)
                raise
            os.replace(private_dst, result_dst)

            rs.seedance_result_url = url
            rs.local_result_path = result_dst
            try:
                transition(rs, SegmentStatus.completed)
            except Exception:
                rs.status = SegmentStatus.completed
            session.commit()
            # AICP-155: the real cost replaces the submit-time estimate (and a
            # task submitted before the ledger existed gets recorded at all).
            from . import spend as spend_mod
            from .pipeline_v2 import _parse_cost_credits

            spend_mod.settle_draw(
                task_id, _parse_cost_credits(data),
                scope_key=spend_mod.run_scope(run_id), kind="video",
                sessions=get_session,
            )
            # Knowledge capture B3 (AICP-299), after the commit; never raises.
            # Skipped if this pass no longer owns the run (AICP-360): a new
            # owner may already have replaced the file at result_dst.
            from .knowledge import capture as kcap

            if still_owner is None or still_owner():
                kcap.output(task_id=task_id, path=result_dst, run_id=run_id, rs_id=rs.id,
                            payload={"url": url, "recovered": True}, segment_version=True,
                            site="B3")
            log.info(
                "Pre-poll: segment %s (idx %d) task %s recovered as completed — no rebill",
                rs.id, rs.index, task_id,
            )


# Distributed lock so that with N workers cold-starting simultaneously, only ONE
# performs reconciliation. Without it, all replicas pass the queue-idle gate at the
# same instant and each re-enqueues the same orphaned runs (observed in production
# 2026-06-24: two workers double-enqueued two runs on a Postgres cutover). TTL is a
# deadlock guard if the holder dies mid-reconcile; it only needs to span the boot skew.
_RECONCILE_LOCK_KEY = "reskin:reconcile:lock"
_RECONCILE_LOCK_TTL_SEC = 120
_RECONCILE_LOCK_HEARTBEAT_SEC = 30

# AICP-360: the pass lock carries an owner token, so a pass whose lock expired
# can neither re-arm nor delete the lock of the pass that took over — it learns
# it lost the lock (refresh returns 0) and stops reconciling runs.
_REFRESH_PASS_LOCK_LUA = """\
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("EXPIRE", KEYS[1], ARGV[2])
else
    return 0
end
"""


def _reconcile_reframe_jobs(live_ids: set[str]) -> dict[str, int]:
    """Reframe jobs (docs/reframe-design.md §7). Never fails a job on a
    snapshot: prepare is free and claimed, so the default is to re-enqueue.

    * ``preparing`` with no live RQ job, untouched for the grace window and no
      live claim -> another ``reframe_prepare``; once 3 claimed attempts are
      used up and the last claim has expired, a CAS under the project lock
      (re-deciding against the row) ends it ``prepare_failed(prepare_crashed)``.
    * QA ``pending`` on a done task, or ``running`` with an expired claim, and
      no live job -> another ``reframe_qa`` (free and idempotent).

    Provider tasks are never touched here: the layer recovers them.
    """
    from .pipeline_reframe import MAX_PREP_ATTEMPTS, fail_crashed_prepare, reap_orphaned_render
    from .reframe_models import ReframeRender
    from .provider_models import ProviderTask
    from .reframe_models import ReframeJob
    from .tasks import enqueue_reframe_prepare, enqueue_reframe_qa

    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=_DERIVED_ORPHAN_GRACE_SEC)
    to_prepare: list[str] = []
    to_qa: list[str] = []
    failed: list[str] = []
    renders_failed: list[str] = []
    try:
        with get_session() as session:
            rows = session.execute(
                select(ReframeJob).where(ReframeJob.status == "preparing")
            ).scalars().all()
            for job in rows:
                if job.id in live_ids:
                    continue
                upd = _as_naive_utc(job.updated_at)
                if upd is not None and upd > cutoff:
                    continue
                exp = _as_naive_utc(job.prep_claim_expires_at)
                if job.prep_claim_owner and exp is not None and exp >= now:
                    continue
                if job.prep_attempts >= MAX_PREP_ATTEMPTS:
                    if fail_crashed_prepare(session, job.id, job.project_id, now):
                        failed.append(job.id)
                else:
                    to_prepare.append(job.id)

            qa_rows = session.execute(
                select(ReframeJob, ProviderTask.status)
                .join(ProviderTask, ProviderTask.id == ReframeJob.provider_task_id)
                .where(ReframeJob.status == "admitted",
                       ReframeJob.qa_status.in_(("pending", "running")))
            ).all()
            for job, task_status in qa_rows:
                if job.id in live_ids or task_status != "done":
                    continue
                if job.qa_status == "running":
                    exp = _as_naive_utc(job.qa_claim_expires_at)
                    if exp is not None and exp >= now:
                        continue
                to_qa.append(job.id)

            # Renders: the tail-edit outcome (failed, Retry), but by a CAS on
            # what was observed. A render reaped in `delivering` may already
            # have uploaded a Drive file; Retry uploads a new one.
            renders = session.execute(
                select(ReframeRender.id, ReframeRender.status, ReframeRender.claim_owner,
                       ReframeRender.updated_at)
                .where(ReframeRender.status.in_(("queued", "processing", "delivering")))
            ).all()
            for rid, st, claim, observed in renders:
                if rid in live_ids:
                    continue
                upd = _as_naive_utc(observed)
                if upd is not None and upd > cutoff:
                    continue
                if reap_orphaned_render(session, rid, st, claim, observed, now):
                    renders_failed.append(rid)
    except Exception as exc:  # pragma: no cover - defensive
        log.error("reconcile_reframe: pass failed: %s", exc)
        return {}
    for jid in to_prepare:
        try:
            enqueue_reframe_prepare(jid)
        except Exception as exc:
            log.error("reconcile_reframe: re-enqueue of prepare %s failed: %s", jid, exc)
    for jid in to_qa:
        try:
            enqueue_reframe_qa(jid)
        except Exception as exc:
            log.error("reconcile_reframe: re-enqueue of QA %s failed: %s", jid, exc)
    if to_prepare or to_qa or failed or renders_failed:
        log.warning("reconcile_reframe: prepare re-enqueued %s, QA re-enqueued %s, "
                    "prepare_crashed %s, renders failed %s", to_prepare, to_qa, failed,
                    renders_failed)
    return {"prepare": len(to_prepare), "qa": len(to_qa), "failed": len(failed),
            "renders_failed": len(renders_failed)}


def reconcile_orphaned_runs(redis_conn: Redis) -> None:
    """
    Public entry point — called once on worker startup before the worker starts
    listening, and by the periodic reconcile tick. Acquires a Redis lock so
    exactly one pass reconciles at a time (prevents duplicate re-enqueue across
    simultaneous boots), then delegates to :func:`_reconcile_orphaned_runs_impl`.
    """
    token = uuid.uuid4().hex
    acquired = redis_conn.set(
        _RECONCILE_LOCK_KEY, token, nx=True, ex=_RECONCILE_LOCK_TTL_SEC
    )
    if not acquired:
        log.info(
            "reconcile_orphaned_runs: another worker holds the reconciliation "
            "lock — skipping (avoids duplicate re-enqueue across simultaneous "
            "worker boots)."
        )
        return

    lost = threading.Event()
    stop = threading.Event()

    def _rearm() -> bool:
        """True only if the pass lock was just re-armed by its owner token.

        A token mismatch means another pass holds the lock: final. A Redis
        error is not final for the heartbeat (it retries on its next beat,
        well inside the TTL) but is a "no" for the caller's checkpoint — this
        pass then stops touching the run rather than guess.
        """
        if lost.is_set():
            return False
        try:
            ok = bool(redis_conn.eval(
                _REFRESH_PASS_LOCK_LUA, 1, _RECONCILE_LOCK_KEY, token,
                _RECONCILE_LOCK_TTL_SEC,
            ))
        except Exception as exc:
            log.warning("reconcile_orphaned_runs: could not re-arm the pass lock: %s", exc)
            return False
        if not ok:
            lost.set()
        return ok

    # AICP-360: keep the pass lock alive for the WHOLE pass, not just between
    # runs. A run's kie.ai pre-poll can outlast the 120s TTL; if the lease
    # lapsed there, a second pass could take the lock, see this pass's per-run
    # claim as just another stale lock (the pass is no RQ job of that run) and
    # take it over while this pass is still writing segment state. The
    # heartbeat dies with the process, so a killed pass still frees the lock
    # within one TTL.
    def _heartbeat() -> None:
        while not stop.wait(_RECONCILE_LOCK_HEARTBEAT_SEC):
            _rearm()
            if lost.is_set():
                return

    beat = threading.Thread(target=_heartbeat, name="reconcile-pass-lock", daemon=True)
    beat.start()
    try:
        _reconcile_orphaned_runs_impl(redis_conn, still_own_pass=_rearm)
    finally:
        stop.set()
        beat.join(timeout=5)
        try:
            _release_lock(redis_conn, _RECONCILE_LOCK_KEY, token)
        except Exception:  # pragma: no cover - best-effort lock release
            pass


def _recover_orphaned_run(
    redis_conn: Redis, run_id: str, observed: tuple[Optional[str], Optional[str]],
    *, still_own_pass=lambda: True,
) -> Optional[str]:
    """Take one orphaned run over and re-enqueue it (AICP-360).

    Returns the status it was re-enqueued from (``"queued"`` /
    ``"preprocessing"``), or None if the run was left alone.

    *observed* is the run's ``(lock token, lock generation)`` read BEFORE the
    caller's fresh liveness scan found no live job for it. The takeover is a
    compare-and-set on both (:func:`app.tasks.claim_stale_run_lock`), which is
    what closes the snapshot→scan race:

    * a job that was executing when the scan ran is in StartedJobRegistry on a
      live worker, so the scan saw it and we never get here;
    * a job that started executing after *observed* was read either acquired
      the lock (new token) or inherited it and refreshed it first (generation
      re-stamped — every poll/finalize/handoff job refreshes before any work), so
      the claim fails and we leave the run alone;
    * a job that starts after the claim finds the lock taken (``SET NX`` fails,
      it skips) or its refresh fails (it aborts) before doing any work.

    A stale lock of a killed worker is never refreshed again, so its token and
    generation stay put and the claim replaces it — the 2026-08-11 fix (a dead
    holder's lock defeating recovery for 3h) is kept without ever deleting a
    live owner's lock.

    Everything below — the status reset, the kie.ai re-poll — runs while we own
    the lock, and the lock is then HANDED to the re-enqueued job
    (``lock_token=``) rather than released, so no job can slip in between. If
    this process dies after the claim, the claim is just another stale lock:
    the next tick finds the run active with no live job and claims it again.

    *still_own_pass* (the reconcile pass lease) is re-checked together with the
    claim at every step that writes — the status reset, each provider call and
    write of the pre-poll, the enqueue — so a pass that lost its lease stops
    before a pass that took over can start on the same run.
    """
    lock_key = run_lock_key(run_id)

    def _still_owner() -> bool:
        return bool(still_own_pass()) and _refresh_lock(redis_conn, lock_key, claim)

    try:
        claim = claim_stale_run_lock(redis_conn, run_id, observed)
    except Exception as exc:
        log.error(
            "reconcile_orphaned_runs: could not claim the lock of run %s (%s) — "
            "leaving it for the next tick", run_id, exc,
        )
        return None
    if claim is None:
        log.warning(
            "reconcile_orphaned_runs: run %s's lock changed since it was read "
            "(a job acquired or refreshed it — e.g. the run was requeued after "
            "the liveness snapshot) — leaving it alone", run_id,
        )
        return None

    handed_off = False
    try:
        with get_session() as session:
            run: Optional[Run] = session.get(Run, run_id, with_for_update=True)
            # Checked UNDER the Run-row lock, not before it: a pass that stalled
            # waiting for the row (or anywhere before this) and lost the run
            # meanwhile must not reset a run a new owner is already driving.
            if not _still_owner():
                log.error(
                    "reconcile_orphaned_runs: lost the pass lease or the claim on "
                    "run %s before recovering it — leaving it", run_id,
                )
                return None
            if run is None or run.status not in _ACTIVE_RUN_STATUSES:
                log.info(
                    "reconcile_orphaned_runs: run %s is no longer active (%s) — "
                    "nothing to recover", run_id, getattr(run, "status", None),
                )
                return None
            log.info("Reconciling orphaned run %s (status=%s)", run_id, run.status)
            # A run stuck in `preprocessing` resumes the GPT Image 2 chain
            # (preprocess_reference_images is resume-aware: a ref whose image is
            # already downloaded is skipped outright, and one with a task_id but
            # no image yet is re-polled rather than recreated, so nothing is
            # rebilled). It stays in `preprocessing`.
            #
            # Every other active state is reset to `queued`:
            #   queued     → queued  no-op; already there
            #   processing → queued  direct edge
            #   stitching/delivering → failed → queued (no direct edge)
            # process_run is idempotent: it skips already-completed segments and
            # reuses an existing final.mp4 (delivery-only retry).
            if run.status not in (RunStatus.preprocessing, RunStatus.queued):
                try:
                    transition(run, RunStatus.queued)
                except Exception:
                    transition(run, RunStatus.failed)
                    transition(run, RunStatus.queued)
            status = run.status
            session.commit()

        if status != RunStatus.preprocessing:
            # The slow kie.ai pre-poll writes segment state: it re-checks the
            # pass lease and re-arms the claim (re-stamping its generation, so
            # a concurrent pass that read it earlier cannot claim it over us)
            # before every provider call and every write.
            try:
                _repoll_generating_segments(run_id, redis_conn, still_owner=_still_owner)
            except _OwnershipLost as exc:
                log.error("reconcile_orphaned_runs: %s — not re-enqueueing", exc)
                return None

        if not _still_owner():
            log.error(
                "reconcile_orphaned_runs: lost the pass lease or the claim on run %s "
                "before the re-enqueue — not re-enqueueing", run_id,
            )
            return None
        # From here on the claim is never released by us, even if the enqueue
        # raises: an enqueue error does not prove the job was not queued (the
        # write can land and the reply be lost), and releasing would let a
        # second owner in beside it. A claim nobody uses is just a stale lock
        # — the next tick takes it over with a NEW token, which the possibly
        # queued job then fails to refresh, so it aborts.
        handed_off = True
        if status == RunStatus.preprocessing:
            enqueue_preprocess_run(run_id, lock_token=claim)
        else:
            enqueue_process_run(run_id, lock_token=claim)
        log.info(
            "reconcile_orphaned_runs: re-enqueued run %s (%s) with its lock handed "
            "over", run_id, status.value,
        )
        return status.value
    except Exception as exc:
        log.error("reconcile_orphaned_runs: failed to recover run %s: %s", run_id, exc)
        return None
    finally:
        if not handed_off:
            try:
                _release_lock(redis_conn, lock_key, claim)
            except Exception:  # pragma: no cover - a leaked claim is re-claimed next tick
                pass


def _reconcile_orphaned_runs_impl(
    redis_conn: Redis, *, still_own_pass=lambda: True,
) -> None:
    """
    Core reconciliation logic (runs under the lock held by the caller).

    1. Compute the set of entity ids with a genuinely-live RQ job.
    2. Query DB for runs in active states; skip any whose id is live.
    3. For the rest, read their lock state, take a FRESH fail-closed liveness
       scan, and take over each still-orphaned run's lock by compare-and-set
       (AICP-360, see :func:`_recover_orphaned_run`).
    4. Under that lock: reset the run to `queued`, pre-poll kie.ai to recover
       already-finished tasks, and re-enqueue it with the lock handed over.
    5. Also reconcile projects stuck in `analyzing` (same per-entity liveness).

    *still_own_pass* re-arms the caller's pass lock; False means another pass
    took over and this one stops reconciling runs.
    """
    log.info("reconcile_orphaned_runs: computing live-job set")
    live_ids = _live_run_ids(redis_conn)

    # -----------------------------------------------------------------
    # Reap orphaned derived creatives (captions, tail edits, overlays, merges)
    # -----------------------------------------------------------------
    # Done FIRST, while live_ids is freshest: the run pass below does serial
    # kie.ai get_task() calls that can take minutes, and judging a job against a
    # minutes-old liveness snapshot is how you fail one that started meanwhile.
    _reap_orphaned_derived_creatives(live_ids)
    _reconcile_reframe_jobs(live_ids)

    # -----------------------------------------------------------------
    # Reconcile orphaned Runs
    # -----------------------------------------------------------------
    orphaned_run_ids: list[str] = []

    with get_session() as session:
        active_ids: list[str] = list(session.execute(
            select(Run.id).where(Run.status.in_([s.value for s in _ACTIVE_RUN_STATUSES]))
        ).scalars())

    # A run that is active in the DB but has a live job is NOT orphaned —
    # another worker is handling it right now. Leave it alone.
    candidates = [rid for rid in active_ids if rid not in live_ids]
    skipped_live_run_ids = [rid for rid in active_ids if rid in live_ids]

    if not candidates:
        log.info("reconcile_orphaned_runs: no orphaned runs found")
    else:
        # AICP-360: `live_ids` is only a snapshot. A run requeued after it (API
        # retry/rerun, segment-fill promote, an approval, an autopilot launch)
        # has a job the snapshot never saw. Read every candidate's lock state
        # FIRST, then rescan liveness, then claim by compare-and-set on what
        # was read — see _recover_orphaned_run for why this order is the fix.
        try:
            observed = read_run_lock_states(redis_conn, candidates)
            fresh_live = _live_run_ids(redis_conn, strict=True)
        except Exception as exc:
            log.error(
                "reconcile_orphaned_runs: could not read lock state / liveness for "
                "%d candidate run(s) (%s) — touching none of them this tick",
                len(candidates), exc,
            )
            candidates, observed, fresh_live = [], {}, set()

        requeued_meanwhile = [rid for rid in candidates if rid in fresh_live]
        if requeued_meanwhile:
            log.info(
                "reconcile_orphaned_runs: %d run(s) gained a live job after the "
                "snapshot — leaving alone: %s",
                len(requeued_meanwhile), requeued_meanwhile,
            )
        orphans = [rid for rid in candidates if rid not in fresh_live]
        if orphans:
            log.warning(
                "reconcile_orphaned_runs: found %d orphaned run(s): %s",
                len(orphans), orphans,
            )
        for run_id in orphans:
            # Re-arm (by owner token) the pass lock before each run: a pre-poll
            # is serial kie.ai get_task() calls, and with many orphaned runs the
            # total can exceed _RECONCILE_LOCK_TTL_SEC. Losing it means another
            # pass is running — stop rather than race it.
            if not still_own_pass():
                log.error(
                    "reconcile_orphaned_runs: lost the reconcile pass lock — "
                    "leaving the remaining runs to the pass that holds it"
                )
                break
            if _recover_orphaned_run(
                redis_conn, run_id, observed[run_id], still_own_pass=still_own_pass,
            ):
                orphaned_run_ids.append(run_id)

    if skipped_live_run_ids:
        log.info(
            "reconcile_orphaned_runs: %d active run(s) have a live job — leaving "
            "alone: %s",
            len(skipped_live_run_ids), skipped_live_run_ids,
        )

    # -----------------------------------------------------------------
    # Reconcile orphaned VideoProjects (stuck in `analyzing`)
    # -----------------------------------------------------------------
    orphaned_project_ids: list[str] = []

    # AICP-350: every analysis admission now commits `analyzing` BEFORE it
    # enqueues the job (so the deploy drain counts it), and `live_ids` above is
    # older than the run pass (minutes of kie.ai pre-polls). A project touched
    # inside the grace window is left alone — a just-admitted analysis is
    # committed but maybe not yet in Redis, exactly the gap the derived-
    # creative reaper's grace covers — and the remaining candidates are judged
    # on a FRESH fail-closed liveness scan (taken only when there are any).
    project_cutoff = datetime.utcnow() - timedelta(seconds=_DERIVED_ORPHAN_GRACE_SEC)

    with get_session() as session:
        stmt = select(VideoProject).where(
            VideoProject.status.in_([s.value for s in _ACTIVE_PROJECT_STATUSES])
        )
        active_projects: list[VideoProject] = list(session.execute(stmt).scalars())

        # A project analyzing with a live analyze job is not orphaned; nor is
        # one admitted inside the grace window.
        orphaned_projects = [
            p for p in active_projects
            if p.id not in live_ids
            and not ((_as_naive_utc(p.updated_at) or project_cutoff) > project_cutoff)
        ]
        if orphaned_projects:
            try:
                fresh_project_live = _live_run_ids(redis_conn, strict=True)
            except Exception as exc:  # fail closed: a failed scan reaps nothing
                log.warning(
                    "reconcile_orphaned_runs: fresh project liveness scan failed "
                    "(no project reaped this pass): %s", exc,
                )
                fresh_project_live = None
            orphaned_projects = [] if fresh_project_live is None else [
                p for p in orphaned_projects if p.id not in fresh_project_live
            ]

        if not orphaned_projects:
            log.info("reconcile_orphaned_runs: no orphaned projects found")
        else:
            log.warning(
                "reconcile_orphaned_runs: found %d orphaned project(s): %s",
                len(orphaned_projects),
                [p.id for p in orphaned_projects],
            )
            for project in orphaned_projects:
                log.info(
                    "Reconciling orphaned project %s (status=%s)",
                    project.id, project.status,
                )
                # analyzing → failed → analyzing is the clean retry path.
                # (created → analyzing is valid; analyzing → analyzing is not.)
                try:
                    # Re-check under the row lock, for EVERY project type
                    # (AICP-350): the liveness snapshot and the select above
                    # are older than this write — the analysis may have
                    # finished and a NEW one been admitted (fresh `analyzing`,
                    # inside the grace window) meanwhile; a rebuild project may
                    # also have new work admitted against it (design §5.1.8).
                    session.refresh(project, with_for_update=True)
                    if project.status != ProjectStatus.analyzing or (
                        (_as_naive_utc(project.updated_at) or project_cutoff) > project_cutoff
                    ):
                        session.rollback()
                        continue
                    transition(project, ProjectStatus.failed)
                    transition(project, ProjectStatus.analyzing)
                    # analyzing is a transient state — reset back to created so the
                    # project can be re-analyzed cleanly via normal trigger path.
                    # Actually: we set it to failed so the operator can re-trigger
                    # analysis from the UI — we don't auto-re-analyze because that
                    # requires InsightFace/GPU and a re-analyze is more expensive.
                    # Undo the second transition: leave at failed so operator retries.
                    project.status = ProjectStatus.failed
                    project.error_message = (
                        (project.error_message or "")
                        + " [recovered from orphaned analyzing state on worker restart]"
                    ).strip()
                    # AICP-350: a localisation analysis promised a transcript
                    # (`pending`, counted by the deploy drain) that will now
                    # never start; settle it unless a transcription is live.
                    from .pipeline_v2 import settle_unstarted_transcript

                    settle_unstarted_transcript(session, project)
                    session.commit()
                    orphaned_project_ids.append(project.id)
                except Exception as exc:
                    log.error(
                        "Failed to reset orphaned project %s: %s", project.id, exc
                    )

    if orphaned_project_ids:
        log.warning(
            "reconcile_orphaned_runs: %d orphaned project(s) reset to failed "
            "(operator must re-trigger analysis): %s",
            len(orphaned_project_ids),
            orphaned_project_ids,
        )

    log.info(
        "reconcile_orphaned_runs: complete — %d run(s) re-enqueued, %d project(s) "
        "reset to failed",
        len(orphaned_run_ids),
        len(orphaned_project_ids),
    )

    # AICP-358: LAST, after the run scan. A composite enqueued here promotes
    # and requeues its run with a process job this pass's live_ids snapshot
    # never saw; running before the scan would let it reap that live run.
    try:
        from .pipeline_segment_fill import reconcile_fill_jobs

        reconcile_fill_jobs(live_ids)
    except Exception as exc:  # pragma: no cover - defensive, never block run recovery
        log.error("reconcile_fill: pass failed: %s", exc)
