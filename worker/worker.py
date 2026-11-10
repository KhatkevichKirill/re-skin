"""
Minimal RQ worker bootstrap.
Connects to Redis and listens on the "overlay", "analyze", "autopilot_compute"
and "process" queues in that priority order — so short interactive jobs (overlay) and segmentation
jobs (analyze) are always picked up before long AI generation runs (process).

The container WORKDIR is /app and PYTHONPATH=/app, so imports of app.* resolve
correctly without any sys.path manipulation here.

Startup sequence
----------------
1. Connect to Redis.
2. Run reconcile_orphaned_runs() — find any runs stuck in an active DB state
   with no live RQ job (from a previous worker crash) and re-enqueue them.
   See app/recovery.py for the race-safety reasoning (only reconciles when all
   queues+started-registries are empty so we never double-process).
3. Pre-warm the InsightFace buffalo_l detector so forked work-horses inherit
   the loaded model via copy-on-write (avoids ~10-30s reload per analyze job).
4. Start the RQ worker with the scheduler enabled (required for enqueue_in /
   enqueue_at to actually execute scheduled jobs) and begin consuming jobs.
"""

import os
import logging
import re

from redis import Redis
from rq import Worker, Queue
from rq.logutils import setup_loghandlers

# Load .env in local dev; no-op in Docker where env vars are injected.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure root logger so all app.* loggers emit to stdout at INFO.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# RQ's own log handler (outputs job lifecycle events).
setup_loghandlers("INFO")

# Get Redis URL from environment.
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Connect to Redis.
redis_conn = Redis.from_url(redis_url)


def _mask_redis_url(url: str) -> str:
    """Replace password in redis://:pass@host or redis://user:pass@host with ***."""
    return re.sub(r"(?<=://)[^@]*:[^@]*(?=@)", "***:***", url)


if __name__ == "__main__":
    logger.info("Starting RQ worker, connecting to %s", _mask_redis_url(redis_url))

    # Queue priority order: overlay (fastest / interactive) → analyze (CPU but finite)
    # → process (long-running AI jobs; mostly idle-waiting between poll rounds).
    from app.tasks import (
        QUEUE_ANALYZE, QUEUE_AUTOPILOT_COMPUTE, QUEUE_OVERLAY, QUEUE_PROCESS,
    )

    # autopilot_compute (AICP-363): free QA scans an autopilot step asks for;
    # after analyze, before the long process jobs.
    queue_names = [QUEUE_OVERLAY, QUEUE_ANALYZE, QUEUE_AUTOPILOT_COMPUTE, QUEUE_PROCESS]
    logger.info("Listening on queues (priority order): %s", queue_names)

    queues = [Queue(name, connection=redis_conn) for name in queue_names]

    # AICP-300 §9 step 6: the capture flags this process will run with.
    try:
        from app.knowledge import capture

        logger.info(capture.boot_line("worker"))
    except Exception:
        logger.exception("capture boot line failed — worker boot continues")

    # Rebuild paid-call protocol (docs/rebuild-from-reference-design.md §7.2):
    # register a NEW boot id for this container BEFORE any work-horse is
    # forked, so every horse inherits it and a send step can record which
    # boot it ran under. A restarted container writes a new id, which is what
    # lets the reaper prove an old sender is gone. Never fatal: without a boot
    # id a send step refuses to send (it never sends unaccounted).
    try:
        from app.rebuild_paid_call import register_boot
        register_boot(redis_conn)
    except Exception:
        logger.exception("rebuild boot id registration failed — rebuild sends will refuse")

    # --- Startup reconciliation (TR5b) ---
    # Must run BEFORE worker.work() so we never race with our own job dispatch.
    logger.info("Running startup orphaned-run reconciliation ...")
    try:
        from app.recovery import reconcile_orphaned_runs
        reconcile_orphaned_runs(redis_conn)
    except Exception:
        logger.exception(
            "reconcile_orphaned_runs raised an unexpected exception — "
            "continuing worker startup (manual recovery may be needed)"
        )
    # Start (or join) the periodic reconciliation reaper. The boot pass above
    # can miss a run orphaned moments earlier (a just-killed worker still has a
    # fresh heartbeat, so its abandoned jobs briefly look live); the reaper
    # re-checks after those signals expire. Bounded to one chain via a Redis
    # lease, so booting a second worker / restarting does not multiply ticks.
    try:
        from app.tasks import enqueue_reconcile_tick
        enqueue_reconcile_tick()
    except Exception:
        logger.exception("failed to start reconcile reaper — boot reconcile still ran")
    # Start (or join) the periodic run-automation tick (scheduled batch
    # generation from the persona spreadsheet — see app/automation.py). Same
    # single-chain Redis lease pattern as the reaper.
    try:
        from app.tasks import enqueue_automation_tick
        enqueue_automation_tick()
    except Exception:
        logger.exception("failed to start automation tick — worker boot continues")
    # Start (or join) the periodic disk cleanup of regenerable media
    # intermediates (see app/disk_cleanup.py). Same single-chain lease.
    try:
        from app.tasks import enqueue_disk_cleanup_tick
        enqueue_disk_cleanup_tick()
    except Exception:
        logger.exception("failed to start disk cleanup tick — worker boot continues")
    # --- End reconciliation ---

    # --- Problem 3: Pre-warm InsightFace detector ---
    # RQ forks a work-horse per job.  By loading the detector here (before
    # worker.work()), forked children inherit the loaded model in memory via
    # copy-on-write, avoiding the ~10-30s reload and ~2GB RSS spike per job.
    #
    # The api container also imports app modules but does NOT execute this block
    # (it runs uvicorn, not worker.py), so the model is never loaded in the API
    # process.
    #
    # Guard: if model files are missing (fresh install, CI environment without
    # GPU/models), log a warning and continue — the work-horse will fall back to
    # the normal lazy-load path (analyze_project's propose_segments call).
    logger.info("Pre-warming InsightFace buffalo_l detector ...")
    try:
        from app.face import _build_detector
        _build_detector()
        logger.info("InsightFace detector pre-warmed successfully.")
    except Exception:
        logger.warning(
            "InsightFace detector pre-warm failed — workers will lazy-load it "
            "on first analyze job (this is normal in CI or if model files are "
            "missing).",
            exc_info=True,
        )
    # --- End pre-warm ---

    worker = Worker(queues, connection=redis_conn)
    # with_scheduler=True: starts the RQScheduler in a sibling process so that
    # jobs enqueued via queue.enqueue_in() (the poll re-schedule mechanism) are
    # moved from the ScheduledJobRegistry to the run queue when their time comes.
    worker.work(with_scheduler=True)
