"""Slim RQ worker for the "localise" queue only (AICP-338).

PID 1 of the localise-worker container. It serves the background translation
behind ``POST /projects/{pid}/localisation-prompt`` and nothing else: a
whole-video translation holds its worker for minutes on one LLM call, which on
the shared run workers would stall every run's submit/poll/stitch, while a
hook-only translation would queue behind a stitch.

Unlike ``worker.worker`` it imports no InsightFace, runs no run
reconciliation, starts no ticks and runs WITHOUT RQ's scheduler (nothing on
this queue is delayed). A job killed by a restart is never replayed: the poll
reports it as interrupted and the operator presses Translate again (see
app/localisation_prompt_job.py). ``restart: unless-stopped`` brings the
container back if the worker process exits.
"""
from __future__ import annotations

import logging
import os

from redis import Redis
from rq import Queue, Worker
from rq.logutils import setup_loghandlers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("localise_worker")


def main() -> None:
    setup_loghandlers("INFO")
    from app.tasks import QUEUE_LOCALISE

    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    logger.info("localise worker: listening on %r (no scheduler)", QUEUE_LOCALISE)
    # AICP-351 §5.2: the translate jobs' LLM attempts record this container's
    # boot id as their sender identity (rebuild_paid_call.sender_proof), so a
    # restarted container proves its old senders gone. Registered BEFORE any
    # work-horse is forked; never fatal (an attempt without one is released
    # only by its own write or by host attestation).
    try:
        from app.rebuild_paid_call import register_boot

        register_boot(conn)
    except Exception:
        logger.exception("boot id registration failed — attempts will need attestation")
    Worker([Queue(QUEUE_LOCALISE, connection=conn)], connection=conn).work()


if __name__ == "__main__":
    main()
