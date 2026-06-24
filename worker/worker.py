"""
Minimal RQ worker bootstrap.
Connects to Redis and listens on the "default" queue.

The container WORKDIR is /app and PYTHONPATH=/app, so imports of app.* resolve
correctly without any sys.path manipulation here.

Startup sequence
----------------
1. Connect to Redis.
2. Run reconcile_orphaned_runs() — find any runs stuck in an active DB state
   with no live RQ job (from a previous worker crash) and re-enqueue them.
   See app/recovery.py for the race-safety reasoning (only reconciles when the
   queue+started-registry are both empty so we never double-process).
3. Start the RQ worker and begin consuming jobs.
"""

import os
import logging

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

# Create the default queue object.
queue = Queue(connection=redis_conn)

if __name__ == "__main__":
    logger.info("Starting RQ worker, connecting to %s", redis_url)
    logger.info("Listening on queue: default")

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
    # --- End reconciliation ---

    worker = Worker([queue], connection=redis_conn)
    worker.work(with_scheduler=False)
