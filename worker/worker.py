"""
Minimal RQ worker bootstrap.
Connects to Redis and listens on the "default" queue.

The container WORKDIR is /app and PYTHONPATH=/app, so imports of app.* resolve
correctly without any sys.path manipulation here.
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

    worker = Worker([queue], connection=redis_conn)
    worker.work(with_scheduler=False)
