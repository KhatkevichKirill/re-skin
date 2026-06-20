"""
tasks.py — RQ job enqueue helpers and worker-callable functions.

Enqueue helpers
---------------
enqueue_analyze(job_id)   Push analyze_job onto the "default" RQ queue.
enqueue_process(job_id)   Push process_job onto the "default" RQ queue.

RQ-callable targets (importable as ``app.tasks.run_analyze`` / ``app.tasks.run_process``)
------------------------------------------------------------------------------------------
run_analyze(job_id)   Instantiate real clients and call pipeline.analyze_job.
run_process(job_id)   Instantiate real clients and call pipeline.process_job.
"""

from __future__ import annotations

import logging

from redis import Redis
from rq import Queue

from .config import settings

log = logging.getLogger(__name__)

_DEFAULT_QUEUE = "default"


def _get_queue() -> Queue:
    """Create a Redis connection and return the RQ default queue."""
    conn = Redis.from_url(settings.REDIS_URL)
    return Queue(_DEFAULT_QUEUE, connection=conn)


# ---------------------------------------------------------------------------
# RQ-callable targets
# ---------------------------------------------------------------------------


def run_analyze(job_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline.analyze_job`.

    Imports pipeline lazily so that the worker process doesn't need to have
    GPU/insightface available at import time (only at execution time).
    """
    from .pipeline import analyze_job

    log.info("run_analyze: job_id=%s", job_id)
    analyze_job(job_id)


def run_process(job_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline.process_job`.

    Creates real :class:`KieClient` and :class:`GDriveClient` instances using
    the configured API keys / service-account file.
    """
    from .kie_client import KieClient
    from .gdrive_client import GDriveClient
    from .pipeline import process_job

    log.info("run_process: job_id=%s", job_id)
    process_job(job_id, kie=KieClient(), gdrive=GDriveClient())


# ---------------------------------------------------------------------------
# Enqueue helpers
# ---------------------------------------------------------------------------


def enqueue_analyze(job_id: str) -> None:
    """
    Push :func:`run_analyze` onto the RQ default queue.

    Parameters
    ----------
    job_id:
        The UUID of the job to analyse.
    """
    q = _get_queue()
    job = q.enqueue("app.tasks.run_analyze", job_id)
    log.info("Enqueued analyze for job_id=%s → rq_job=%s", job_id, job.id)


def enqueue_process(job_id: str) -> None:
    """
    Push :func:`run_process` onto the RQ default queue.

    Parameters
    ----------
    job_id:
        The UUID of the job to process.
    """
    q = _get_queue()
    job = q.enqueue("app.tasks.run_process", job_id)
    log.info("Enqueued process for job_id=%s → rq_job=%s", job_id, job.id)
