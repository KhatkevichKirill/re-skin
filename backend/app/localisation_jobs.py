"""Durable transcription and translate jobs (AICP-365, docs/autopilot-localisation-flow.md §3).

Both used to be "commit a status + a reservation, then push an RQ job no
recovery covers": a lost enqueue left ``pending`` forever, a worker killed
mid-call left ``running`` forever, and a translate result lived only in Redis
(a deploy kills it without replay). Here each admitted job is a database row
(app/localisation_jobs_models.py) and RQ is only the doorbell:

* **Admission** (``admit_transcription`` / ``admit_translate``) writes the row,
  the visible status and the spend reservation in the CALLER's transaction and
  never commits, flushes only through ``spend.reserve`` and never rolls back —
  so the autopilot's handlers can call it inside their single admission commit
  (contract rules (a), (b), (g)) and the endpoints commit it themselves. The
  caller enqueues AFTER its commit (:func:`enqueue`).
* **Dispatch generation.** The RQ id is ``<prefix><row id>-g<dispatch_gen>``.
  A worker claims only ``status='queued' AND dispatch_gen=<its gen>``, so a
  duplicate or stale delivery claims nothing; the database claim is
  authoritative.
* **Fencing.** Every write of a worker after its claim requires
  ``status='running' AND claim_token=<its token>``; the project row is locked
  first (the ``project_transcript:`` version key's fence, core §4.3).
* **Reaper** (:func:`reap`, from the run workers' 180 s reconcile tick): a
  ``queued`` row whose RQ job has not been live for :data:`NOT_LIVE_GRACE_SEC`
  gets a new generation and is re-enqueued (nothing was sent before a claim);
  a ``running`` row whose job has not been live as long is ``interrupted`` and
  its claim revoked in the same statement. An interrupted call may already have
  been billed, so NOTHING resubmits it: the operator presses the button again,
  or the autopilot asks a regeneration approval.

The operator's Translate button keeps its Redis-only job
(localisation_prompt_job.py); only the autopilot uses the durable translate
row. Every transcription (button, after-analysis hand-off, autopilot) goes
through :func:`admit_transcription` (operator decision 2026-10-31: a second
click while one is live answers 409).
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from typing import Callable, Optional

from fastapi import HTTPException
from sqlalchemy import select

from . import spend as spend_mod
from .localisation_jobs_models import (
    LIVE_JOB_STATUSES,
    LocalisationPromptJob,
    ProjectTranscription,
)
from .models import VideoProject
from .models_autopilot import utcnow

log = logging.getLogger(__name__)

TRANSCRIBE_PREFIX = "transcribe-db-"
TRANSLATE_PREFIX = "localise-prompt-db-"
NOT_LIVE_GRACE_SEC = 120

TRANSCRIPT_PENDING = "pending"
TRANSCRIPT_RUNNING = "running"
TRANSCRIPT_FAILED = "failed"

INTERRUPTED_MESSAGE = (
    "interrupted: the worker stopped while the call was in flight (e.g. a deploy). "
    "The call may already have been billed — run it again only if you need it."
)


def rq_job_id(job) -> str:
    prefix = TRANSCRIBE_PREFIX if isinstance(job, ProjectTranscription) else TRANSLATE_PREFIX
    return f"{prefix}{job.id}-g{job.dispatch_gen}"


def _budget_refusal(exc) -> HTTPException:
    return HTTPException(status_code=429, detail=getattr(exc, "message", str(exc)))


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def live_transcription(db, project_id: str) -> Optional[ProjectTranscription]:
    return db.execute(select(ProjectTranscription).where(
        ProjectTranscription.project_id == project_id,
        ProjectTranscription.status.in_(LIVE_JOB_STATUSES),
    )).scalars().first()


def latest_transcription(db, project_id: str) -> Optional[ProjectTranscription]:
    return db.execute(select(ProjectTranscription).where(
        ProjectTranscription.project_id == project_id,
    ).order_by(ProjectTranscription.created_at.desc())).scalars().first()


def admit_transcription(db, project: VideoProject, *, origin: str) -> ProjectTranscription:
    """Admit one transcription of *project* in the caller's transaction (no commit).

    Takes the project row lock (re-entrant when the caller holds it). Raises
    HTTPException 409 without a source file or while another transcription of
    the project is live, 429 when today's kie.ai budget cannot cover it.
    """
    db.refresh(project, with_for_update=True)
    if not project.source_local_path:
        raise HTTPException(
            status_code=409,
            detail="Cannot transcribe: this project has no source video on disk",
        )
    live = live_transcription(db, project.id)
    if live is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A transcription of this project is already {live.status} (started "
                f"{live.created_at:%Y-%m-%d %H:%M} UTC by {live.origin}) — wait for it to "
                "finish"
            ),
        )
    now = utcnow()
    job = ProjectTranscription(
        id=str(uuid.uuid4()), project_id=project.id, origin=origin, status="queued",
        dispatch_gen=1, attempts=0, created_at=now, updated_at=now,
    )
    job.rq_job_id = rq_job_id(job)
    project.transcript_status = TRANSCRIPT_PENDING
    project.transcript_error = None
    db.add(job)
    try:
        spend_mod.reserve(
            db, spend_mod.transcribe_scope(project.id), spend_mod.llm_credits(),
            kind="transcription",
        )
    except spend_mod.BudgetExceeded as exc:
        raise _budget_refusal(exc) from exc
    return job


def _claim(db, model, job_id: str, gen: int, token: str):
    job = db.execute(select(model).where(model.id == job_id).with_for_update()).scalar_one_or_none()
    if job is None or job.status != "queued" or int(job.dispatch_gen) != int(gen):
        return None
    now = utcnow()
    job.status, job.claim_token = "running", token
    job.attempts = (job.attempts or 0) + 1
    job.started_at = job.started_at or now
    job.not_live_since, job.updated_at = None, now
    return job


def _fenced(db, model, job_id: str, token: str):
    job = db.execute(select(model).where(model.id == job_id).with_for_update()).scalar_one_or_none()
    if job is None or job.status != "running" or job.claim_token != token:
        return None
    return job


def run_transcription_job(job_id: str, gen: int, *, kie=None,
                          session_factory: Optional[Callable] = None) -> Optional[str]:
    """RQ entry point (``process`` queue). Returns the state it wrote, or None
    when it claimed nothing / was fenced out."""
    from . import localisation as localisation_mod

    sf = session_factory or _session_factory()
    token = uuid.uuid4().hex
    db = sf()
    try:
        row = db.get(ProjectTranscription, job_id)
        if row is None:
            return None
        project = db.execute(select(VideoProject).where(
            VideoProject.id == row.project_id).with_for_update()).scalar_one_or_none()
        job = _claim(db, ProjectTranscription, job_id, gen, token)
        if job is None:
            db.rollback()
            log.info("transcription job %s g%s: nothing to claim", job_id, gen)
            return None
        if project is None:
            job.status, job.error, job.finished_at = "failed", "project deleted", utcnow()
            job.claim_token = None
            db.commit()
            return "failed"
        local = project.source_local_path
        project_id = project.id
        import os

        if not local or not os.path.exists(local):
            msg = f"source file not found at {local!r}"
            job.status, job.error, job.finished_at = "failed", msg, utcnow()
            job.claim_token = None
            project.transcript_status, project.transcript_error = TRANSCRIPT_FAILED, msg
            db.commit()
            return "failed"
        project.transcript_status, project.transcript_error = TRANSCRIPT_RUNNING, None
        db.commit()
    finally:
        db.close()

    transcript, error = None, None
    try:
        transcript = localisation_mod.transcribe_video(
            local, kie=kie, spend_scope=spend_mod.transcribe_scope(project_id),
        )
    except localisation_mod.LocalisationError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 — recorded, as pipeline_v2.transcribe_project does
        log.exception("transcription job %s: unexpected error", job_id)
        error = f"{type(exc).__name__}: {exc}"

    db = sf()
    try:
        project = db.execute(select(VideoProject).where(
            VideoProject.id == project_id).with_for_update()).scalar_one_or_none()
        job = _fenced(db, ProjectTranscription, job_id, token)
        if job is None:
            db.rollback()
            log.warning("transcription job %s: claim revoked meanwhile — result discarded", job_id)
            return None
        now = utcnow()
        job.claim_token, job.finished_at, job.updated_at = None, now, now
        if error is not None:
            job.status, job.error = "failed", error[:4000]
            if project is not None:
                # A failed attempt keeps an earlier good transcript in place.
                project.transcript_status, project.transcript_error = TRANSCRIPT_FAILED, error
        else:
            job.status = "succeeded"
            if project is not None:
                lines = transcript.get("lines") or []
                project.transcript = transcript
                project.transcript_status = "ready" if lines else "empty"
                project.transcript_error = None
        db.commit()
        return job.status
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Translate (the autopilot's durable localisation-prompt job)
# ---------------------------------------------------------------------------


def admit_translate(db, project: VideoProject, *, language: str, swap_character: bool = False,
                    model: Optional[str] = None) -> LocalisationPromptJob:
    """Admit one translate job in the caller's transaction (no commit).

    The refusals of ``POST /localisation-prompt`` (400 bad language/model,
    409 transcript not ready, 422 no speech, 429 budget) come from the same
    ``api_v2._prepare_localisation_snapshot``; a missing localise worker is a
    409 here (retryable), so nothing is reserved for a job nobody would run.
    """
    from . import localisation_prompt_job as lpj
    from .api_v2 import _prepare_localisation_snapshot

    request_params = {"model": model} if model else {}
    snapshot, project, swap_segments = _prepare_localisation_snapshot(
        project.id, language, swap_character, None, db, request_params=request_params,
    )
    if not _localise_worker_alive():
        raise HTTPException(
            status_code=409,
            detail="No localise worker is running, so a translation would never start",
        )
    now = utcnow()
    job = LocalisationPromptJob(
        id=str(uuid.uuid4()), project_id=project.id, language=snapshot["target_language"],
        request={"language": snapshot["target_language"], "swap_character": bool(swap_character),
                 "model": model},
        snapshot=snapshot, fingerprint=lpj.project_fingerprint(project, swap_segments),
        status="queued", dispatch_gen=1, attempts=0, created_at=now, updated_at=now,
    )
    job.rq_job_id = rq_job_id(job)
    db.add(job)
    try:
        # Three calls' worth, the most translate_lines can bill (as the button).
        spend_mod.reserve(
            db, spend_mod.localise_scope(project.id), 3 * spend_mod.llm_credits(),
            kind="localisation prompt",
        )
    except spend_mod.BudgetExceeded as exc:
        raise _budget_refusal(exc) from exc
    return job


def _localise_worker_alive() -> bool:
    from . import localisation_prompt_job as lpj
    from .tasks import _get_redis

    try:
        return lpj.worker_alive(_get_redis())
    except Exception:  # Redis unreachable: refuse (retryable), never guess
        log.warning("localise worker liveness unknown (Redis error)", exc_info=True)
        return False


def run_translate_job(job_id: str, gen: int, *,
                      session_factory: Optional[Callable] = None) -> Optional[str]:
    """RQ entry point (``localise`` queue). Same budgets and timeout as the
    button's job (localisation_prompt_job.run)."""
    from . import localisation as localisation_mod
    from . import localisation_prompt_job as lpj
    from .api_v2 import _compute_localisation_prompt

    sf = session_factory or _session_factory()
    token = uuid.uuid4().hex
    db = sf()
    try:
        job = _claim(db, LocalisationPromptJob, job_id, gen, token)
        if job is None:
            db.rollback()
            return None
        snapshot = dict(job.snapshot or {})
        db.commit()
    finally:
        db.close()

    result, error = None, None
    with lpj._job_translate_timeout():
        try:
            response = _compute_localisation_prompt(
                snapshot, translate_budgets=lpj.translate_budgets())
            result = response.model_dump(mode="json")
        except localisation_mod.LocalisationError as exc:
            error = {"status_code": 502, "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.exception("translate job %s crashed", job_id)
            error = {"status_code": 500, "detail": f"{type(exc).__name__}: {exc}"[:2000]}

    db = sf()
    try:
        job = _fenced(db, LocalisationPromptJob, job_id, token)
        if job is None:
            db.rollback()
            log.warning("translate job %s: claim revoked meanwhile — result discarded", job_id)
            return None
        now = utcnow()
        job.claim_token, job.finished_at, job.updated_at = None, now, now
        if error is None:
            job.status, job.result = "succeeded", result
        else:
            job.status, job.error_detail, job.error = "failed", error, error["detail"]
        db.commit()
        return job.status
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Enqueue + reaper
# ---------------------------------------------------------------------------


def enqueue(job) -> None:
    """Push the doorbell for *job* (after the admission commit). Best effort:
    a lost enqueue is re-sent by the reaper."""
    from . import tasks

    try:
        if isinstance(job, ProjectTranscription):
            tasks.enqueue_transcription_job(job.id, int(job.dispatch_gen), job.rq_job_id)
        else:
            tasks.enqueue_translate_job(job.id, int(job.dispatch_gen), job.rq_job_id)
    except Exception:
        log.exception("enqueue of %s failed — the reaper re-sends it", job.rq_job_id)


def _session_factory():
    from .db import SessionLocal

    return SessionLocal


def _default_is_live(queue: str, job_id: Optional[str]) -> bool:
    from .tasks import _get_redis, rq_job_is_live

    return rq_job_is_live(_get_redis(), queue, job_id)


def reap(*, session_factory: Optional[Callable] = None,
         is_live: Optional[Callable[[str, Optional[str]], bool]] = None,
         enqueue_fn: Optional[Callable] = None, now=None) -> dict:
    """One reaper pass over every live job row. Returns counters."""
    from .tasks import QUEUE_LOCALISE, QUEUE_PROCESS

    sf = session_factory or _session_factory()
    is_live = is_live or _default_is_live
    enqueue_fn = enqueue_fn or enqueue
    now = now or utcnow()
    grace = timedelta(seconds=NOT_LIVE_GRACE_SEC)
    out = {"requeued": 0, "interrupted": 0, "observed": 0}
    for model, queue in ((ProjectTranscription, QUEUE_PROCESS),
                         (LocalisationPromptJob, QUEUE_LOCALISE)):
        db = sf()
        try:
            ids = [i for (i,) in db.execute(select(model.id).where(
                model.status.in_(LIVE_JOB_STATUSES)))]
        finally:
            db.close()
        for job_id in ids:
            db = sf()
            requeue = None
            try:
                row = db.get(model, job_id)
                if row is None:
                    continue
                project = None
                if model is ProjectTranscription:
                    # Lock order: project, then job (as admission and the worker).
                    project = db.execute(select(VideoProject).where(
                        VideoProject.id == row.project_id).with_for_update()).scalar_one_or_none()
                job = db.execute(select(model).where(model.id == job_id).with_for_update(
                    skip_locked=True)).scalar_one_or_none()
                if job is None or job.status not in LIVE_JOB_STATUSES:
                    db.rollback()
                    continue
                if is_live(queue, job.rq_job_id):
                    if job.not_live_since is not None:
                        job.not_live_since = None
                        db.commit()
                    else:
                        db.rollback()
                    continue
                if job.not_live_since is None:
                    job.not_live_since, job.updated_at = now, now
                    db.commit()
                    out["observed"] += 1
                    continue
                if now - job.not_live_since < grace:
                    db.rollback()
                    continue
                if job.status == "queued":
                    job.dispatch_gen = int(job.dispatch_gen) + 1
                    job.rq_job_id = rq_job_id(job)
                    job.not_live_since, job.updated_at = None, now
                    db.commit()
                    requeue = job
                    out["requeued"] += 1
                else:  # running: the worker died mid-call — revoke, never replay
                    job.status, job.claim_token = "interrupted", None
                    job.error, job.finished_at, job.updated_at = INTERRUPTED_MESSAGE, now, now
                    if model is LocalisationPromptJob:
                        job.error_detail = {"status_code": 503, "detail": INTERRUPTED_MESSAGE}
                    if project is not None:
                        project.transcript_status = TRANSCRIPT_FAILED
                        project.transcript_error = INTERRUPTED_MESSAGE
                    db.commit()
                    out["interrupted"] += 1
                    log.warning("%s %s interrupted (worker gone mid-call)", model.__tablename__,
                                job_id)
            finally:
                db.close()
            if requeue is not None:
                enqueue_fn(requeue)
    return out
