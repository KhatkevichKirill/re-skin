"""Slim RQ worker for the "fal" queue only (AICP-330 B1).

Started by ``worker.fal_supervisor`` (PID 1 of the fal-worker container). It
serves Fal provider tasks and nothing else, so a Fal result is never queued
behind a long run job and Fal work can never starve run work. Unlike
``worker.worker`` it imports no InsightFace and runs no run reconciliation;
it DOES run RQ's scheduler for its own queue, because polls and fetch retries
are delayed with ``enqueue_in``.
"""
from __future__ import annotations

import logging
import os
import sys

from redis import Redis
from rq import Queue, Worker
from rq.logutils import setup_loghandlers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("fal_worker")


def main() -> int:
    setup_loghandlers("INFO")
    from app.pipeline_provider import validate_settings
    from app.tasks import QUEUE_FAL

    errors = validate_settings()
    if errors:
        for e in errors:
            logger.error("fal worker: invalid settings: %s", e)
        return 2
    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    Worker([Queue(QUEUE_FAL, connection=conn)], connection=conn).work(with_scheduler=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
