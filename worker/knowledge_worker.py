"""Slim RQ worker for the "knowledge" queue only (AICP-299 §6.1).

Started by ``worker.knowledge_supervisor`` (PID 1 of the knowledge-worker
container). Unlike ``worker.worker`` it never imports or pre-warms InsightFace
or any ML dependency, runs no boot run-reconciliation, starts no reconcile or
automation tick, and runs **without** RQ's scheduler: nothing on this queue is
delayed — the supervised ticker enqueues each tick when it is due.
"""
from __future__ import annotations

import logging
import os

from redis import Redis
from rq import Queue, Worker
from rq.logutils import setup_loghandlers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("knowledge_worker")

QUEUE = "knowledge"


def main() -> None:
    setup_loghandlers("INFO")
    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    try:
        from app.knowledge import capture

        logger.info(capture.boot_line("knowledge-worker"))
    except Exception:
        logger.exception("knowledge worker: capture boot line failed")
    Worker([Queue(QUEUE, connection=conn)], connection=conn).work()  # no scheduler (§6.2)


if __name__ == "__main__":
    main()
