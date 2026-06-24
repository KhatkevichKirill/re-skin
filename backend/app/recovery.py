"""
recovery.py — Startup orphaned-run reconciliation (TR5b).

Called ONCE by the worker bootstrap (worker/worker.py) BEFORE the worker begins
listening for new jobs.  It finds runs that are stuck in an active state with no
live RQ job and re-enqueues them so they proceed automatically.

Race-safety design
------------------
With N parallel workers, a run may appear "stuck" in `processing` while it is
actually being worked on by another worker right now.  We must NOT re-enqueue
such a run (that would double-process it).

The approach used here is a **safe-start window**:

    Only reconcile when the entire "default" RQ queue+started-registry is empty
    at the moment this worker boots.

Rationale:
- Each worker calls reconcile_orphaned_runs() once at startup, BEFORE it begins
  consuming jobs.
- A run genuinely active on another worker will have its RQ job visible in the
  StartedJobRegistry on Redis.
- If the StartedJobRegistry is non-empty, at least one worker is mid-job → we
  skip reconciliation entirely (the other worker is healthy; runs will drain
  normally).  A worker that boots while siblings are busy is not a cold-start
  scenario and does not need to reconcile.
- If both the queue AND the started-registry are empty, the system is at rest
  (no jobs in-flight on any worker).  Any DB run still in an active state is
  genuinely orphaned (its worker died before completing), and it is safe to
  re-enqueue it.

The two-condition check (queue empty AND started-registry empty) is the key
guard.  A run stuck in `processing` while another worker is busy is left alone;
it will be recovered when the queue next goes idle (next worker restart after
all other workers have also gone idle), or by a manual retry.

This is conservative: it trades the edge case of "orphan not recovered until
the queue fully drains" for the hard requirement of "never double-process".  In
practice, worker crashes happen during redeploys or OOM kills, both of which
drain/kill all workers simultaneously, so the queue IS empty at the restart
moment.

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
from typing import Optional

from redis import Redis
from rq import Queue
from rq.registry import StartedJobRegistry
from sqlalchemy import select

from .db import get_session
from .kie_client import KieClient
from .models import Run, RunSegment, VideoProject
from .state_machine import ProjectStatus, RunStatus, SegmentStatus, transition
from .storage import run_results_dir
from .tasks import enqueue_process_run

log = logging.getLogger(__name__)

# Active run states — any run in these states should have a live RQ job.
_ACTIVE_RUN_STATUSES = {
    RunStatus.queued,
    RunStatus.processing,
    RunStatus.stitching,
    RunStatus.delivering,
}

# Active project states — any project in these states should have a live RQ job.
_ACTIVE_PROJECT_STATUSES = {ProjectStatus.analyzing}


def _queue_is_idle(redis_conn: Redis) -> bool:
    """
    Return True iff the default RQ queue AND the started-job registry are both
    empty, meaning no job is in-flight or waiting on any worker right now.

    This is the safety gate: we only reconcile on a fully idle queue to avoid
    double-processing a run that is actively being worked on by another worker.
    """
    q = Queue("default", connection=redis_conn)
    pending_count = len(q)

    reg = StartedJobRegistry("default", connection=redis_conn)
    started_ids = reg.get_job_ids()

    log.debug(
        "Queue idle check: pending=%d started=%d", pending_count, len(started_ids)
    )
    return pending_count == 0 and len(started_ids) == 0


def _repoll_generating_segments(run_id: str, redis_conn: Redis) -> None:
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

        for rs in generating_segs:
            task_id = rs.seedance_task_id
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
            try:
                kie.download_result(url, result_dst)
            except Exception as exc:
                log.warning(
                    "download_result failed for task %s: %s — will resubmit", task_id, exc
                )
                continue

            rs.seedance_result_url = url
            rs.local_result_path = result_dst
            try:
                transition(rs, SegmentStatus.completed)
            except Exception:
                rs.status = SegmentStatus.completed
            session.commit()
            log.info(
                "Pre-poll: segment %s (idx %d) task %s recovered as completed — no rebill",
                rs.id, rs.index, task_id,
            )


def reconcile_orphaned_runs(redis_conn: Redis) -> None:
    """
    Main entry point — called once on worker startup before the worker starts
    listening for new jobs.

    1. Safety gate: if the RQ queue+started-registry is NOT empty, skip
       reconciliation entirely (another worker is active; not a cold start).
    2. Query DB for runs in active states.
    3. For processing runs with generating segments, pre-poll kie.ai to recover
       already-finished tasks without resubmitting.
    4. Reset each orphaned run to `queued` and re-enqueue it.
    5. Also reconcile projects stuck in `analyzing`.
    """
    log.info("reconcile_orphaned_runs: starting safety check")

    if not _queue_is_idle(redis_conn):
        log.info(
            "reconcile_orphaned_runs: queue is NOT idle — another worker is active. "
            "Skipping reconciliation to avoid double-processing."
        )
        return

    log.info("reconcile_orphaned_runs: queue is idle — checking DB for orphaned runs")

    # -----------------------------------------------------------------
    # Reconcile orphaned Runs
    # -----------------------------------------------------------------
    orphaned_run_ids: list[str] = []

    with get_session() as session:
        stmt = select(Run).where(Run.status.in_([s.value for s in _ACTIVE_RUN_STATUSES]))
        orphaned_runs: list[Run] = list(session.execute(stmt).scalars())

        if not orphaned_runs:
            log.info("reconcile_orphaned_runs: no orphaned runs found")
        else:
            log.warning(
                "reconcile_orphaned_runs: found %d orphaned run(s): %s",
                len(orphaned_runs),
                [r.id for r in orphaned_runs],
            )

            for run in orphaned_runs:
                log.info(
                    "Reconciling orphaned run %s (status=%s)", run.id, run.status
                )
                # Reset active run to queued.  All active states → queued are
                # valid:
                #   queued     → queued  would be a no-op; already there
                #   processing → queued  newly added transition edge
                #   stitching  → queued  (via failed intermediate if needed)
                #   delivering → queued
                # For stitching/delivering, process_run is idempotent: it skips
                # already-completed segments and reuses an existing final.mp4
                # (delivery-only retry) so re-processing from queued is safe.
                try:
                    if run.status == RunStatus.queued:
                        # Already queued — just re-enqueue (no state change needed).
                        pass
                    else:
                        # Reset to queued via the new processing→queued edge, or
                        # via failed→queued for stitching/delivering (which lack a
                        # direct →queued edge).
                        try:
                            transition(run, RunStatus.queued)
                        except Exception:
                            # Fallback: force-set without transition guard
                            # (stitching/delivering don't have a direct →queued edge;
                            # go via failed then queued).
                            transition(run, RunStatus.failed)
                            transition(run, RunStatus.queued)
                    session.commit()
                    orphaned_run_ids.append(run.id)
                except Exception as exc:
                    log.error(
                        "Failed to reset orphaned run %s to queued: %s", run.id, exc
                    )

    # Pre-poll generating segments for recovered runs to avoid rebilling.
    # Done OUTSIDE the session above (each call opens its own session) so that
    # each run's pre-poll operates on a fresh committed state.
    for run_id in orphaned_run_ids:
        _repoll_generating_segments(run_id, redis_conn)

    # Re-enqueue all recovered runs.
    if orphaned_run_ids:
        for run_id in orphaned_run_ids:
            try:
                enqueue_process_run(run_id)
                log.info("reconcile_orphaned_runs: re-enqueued run %s", run_id)
            except Exception as exc:
                log.error(
                    "reconcile_orphaned_runs: failed to re-enqueue run %s: %s",
                    run_id, exc,
                )

    # -----------------------------------------------------------------
    # Reconcile orphaned VideoProjects (stuck in `analyzing`)
    # -----------------------------------------------------------------
    orphaned_project_ids: list[str] = []

    with get_session() as session:
        stmt = select(VideoProject).where(
            VideoProject.status.in_([s.value for s in _ACTIVE_PROJECT_STATUSES])
        )
        orphaned_projects: list[VideoProject] = list(session.execute(stmt).scalars())

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
