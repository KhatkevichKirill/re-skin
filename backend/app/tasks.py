"""
tasks.py — RQ job enqueue helpers and worker-callable functions.

Enqueue helpers (v1)
--------------------
enqueue_analyze(job_id)   Push analyze_job onto the "default" RQ queue.
enqueue_process(job_id)   Push process_job onto the "default" RQ queue.

RQ-callable targets (v1, importable as ``app.tasks.run_analyze`` / ``app.tasks.run_process``)
----------------------------------------------------------------------------------------------
run_analyze(job_id)   Instantiate real clients and call pipeline.analyze_job.
run_process(job_id)   Instantiate real clients and call pipeline.process_job.

Enqueue helpers (v2)
--------------------
enqueue_analyze_project(project_id)   Push run_analyze_project onto the "default" RQ queue.
enqueue_process_run(run_id)           Push run_process_run onto the "default" RQ queue.

RQ-callable targets (v2)
------------------------
run_analyze_project(project_id)   Call pipeline_v2.analyze_project with real clients.
run_process_run(run_id)           Call pipeline_v2.process_run with real clients.
"""

from __future__ import annotations

import logging
import os

from redis import Redis
from rq import Queue

from .config import settings

log = logging.getLogger(__name__)

_DEFAULT_QUEUE = "default"

# RQ's default job_timeout is 180s, far too short: analyze runs InsightFace over
# the whole video, and process runs several Seedance jobs sequentially (each can
# take minutes). Without these, long jobs are killed mid-flight with
# "Task exceeded maximum timeout value (180 seconds)" while the Seedance task
# keeps running on kie.ai. Override generously (seconds); tunable via env.
ANALYZE_JOB_TIMEOUT = int(os.getenv("ANALYZE_JOB_TIMEOUT", "1800"))      # 30 min
PROCESS_JOB_TIMEOUT = int(os.getenv("PROCESS_JOB_TIMEOUT", "10800"))     # 3 hours


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
    job = q.enqueue("app.tasks.run_analyze", job_id, job_timeout=ANALYZE_JOB_TIMEOUT)
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
    job = q.enqueue("app.tasks.run_process", job_id, job_timeout=PROCESS_JOB_TIMEOUT)
    log.info("Enqueued process for job_id=%s → rq_job=%s", job_id, job.id)


# ---------------------------------------------------------------------------
# v2 RQ-callable targets
# ---------------------------------------------------------------------------


def run_analyze_project(project_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_v2.analyze_project`.

    Imports pipeline_v2 lazily so the worker process doesn't need GPU/insightface
    at import time (only at execution time).
    """
    from .pipeline_v2 import analyze_project

    log.info("run_analyze_project: project_id=%s", project_id)
    analyze_project(project_id)


def run_process_run(run_id: str) -> None:
    """
    RQ-callable wrapper for :func:`pipeline_v2.process_run`.

    Creates real :class:`KieClient` and :class:`GDriveClient` instances using
    the configured API keys / service-account file.
    """
    from .kie_client import KieClient
    from .gdrive_client import GDriveClient
    from .pipeline_v2 import process_run

    log.info("run_process_run: run_id=%s", run_id)
    process_run(run_id, kie=KieClient(), gdrive=GDriveClient())


# ---------------------------------------------------------------------------
# v2 Enqueue helpers
# ---------------------------------------------------------------------------


def enqueue_analyze_project(project_id: str) -> None:
    """
    Push :func:`run_analyze_project` onto the RQ default queue.

    Parameters
    ----------
    project_id:
        The UUID of the VideoProject to analyse.
    """
    q = _get_queue()
    job = q.enqueue(
        "app.tasks.run_analyze_project", project_id, job_timeout=ANALYZE_JOB_TIMEOUT
    )
    log.info(
        "Enqueued analyze_project for project_id=%s → rq_job=%s", project_id, job.id
    )


def enqueue_process_run(run_id: str) -> None:
    """
    Push :func:`run_process_run` onto the RQ default queue.

    Parameters
    ----------
    run_id:
        The UUID of the Run to process.
    """
    q = _get_queue()
    job = q.enqueue(
        "app.tasks.run_process_run", run_id, job_timeout=PROCESS_JOB_TIMEOUT
    )
    log.info("Enqueued process_run for run_id=%s → rq_job=%s", run_id, job.id)
