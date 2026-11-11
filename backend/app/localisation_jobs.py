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

from . import paid_attempts
from . import spend as spend_mod
from .paid_attempts_models import OWNER_LLM_TRANSCRIBE, OWNER_LLM_TRANSLATE
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
        # AICP-351 §5.2: the worst case of the bounded retry (3 attempts x 1
        # call x the capped model chain), on the job's own scope; the job's
        # terminal write releases what it did not draw.
        spend_mod.reserve(
            db, spend_mod.transcribe_job_scope(job.id),
            paid_attempts.llm_reservation_credits(
                paid_attempts.TRANSCRIBE_CALLS_PER_ATTEMPT, _transcribe_chain_len()),
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


def _transcribe_chain_len() -> int:
    from . import localisation

    return localisation.transcribe_chain_len()


def _translate_chain_len() -> int:
    from . import localisation

    return localisation.translate_chain_len()


def transcription_job_timeout_sec() -> int:
    """RQ job_timeout of a durable transcription: room for the in-process
    retries of §5.2 (3 attempts x the chain x the per-call timeout)."""
    from .config import settings
    from .tasks import TRANSCRIBE_JOB_TIMEOUT

    worst = (paid_attempts.LLM_MAX_ATTEMPTS * _transcribe_chain_len()
             * float(settings.LOCALISATION_TIMEOUT_SEC) + 600)
    return int(max(TRANSCRIBE_JOB_TIMEOUT, worst))


def translate_job_timeout_sec() -> int:
    """RQ job_timeout of a durable translate job: 3 attempts of the button
    job's worst case."""
    from . import localisation_prompt_job as lpj

    return int(paid_attempts.LLM_MAX_ATTEMPTS * lpj.job_timeout_sec())


def _sessions(sf):
    """A ``get_session``-style context manager over *sf* (paid_attempts and
    spend take one)."""
    from contextlib import contextmanager

    @contextmanager
    def cm():
        db = sf()
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    return cm


def _job_fence(model, job_id: str, token: str):
    """True while this worker still owns the job (locks the job row: lock
    order job -> attempt)."""

    def fence(db) -> bool:
        job = db.execute(select(model).where(model.id == job_id).with_for_update()
                         .execution_options(populate_existing=True)).scalar_one_or_none()
        return job is not None and job.status == "running" and job.claim_token == token

    return fence


def _llm_attempts(*, owner_kind: str, model=None, job_id: str, token: str, sf=None,
                  max_calls: int, lease_sec: int, scope: str, kind: str, call,
                  sessions=None, fence=None, run_id: Optional[str] = None):
    """Run *call(ledger)* as up to LLM_MAX_ATTEMPTS attempts (AICP-351 §5.2):
    only an UNKNOWN outcome is retried, each try is a new attempt row, and
    each attempt is acknowledged after its last I/O returned.

    Returns ``(result, attempt_id, error, fenced)``: ``result`` with the still-
    ``started`` attempt to accept in the job's final write; or ``error``
    (definite failure, or the bound spent); ``fenced`` = the job is no longer
    ours (write nothing)."""
    from . import localisation as localisation_mod

    sessions = sessions or _sessions(sf)
    fence = fence or _job_fence(model, job_id, token)
    last_unknown = None
    while True:
        try:
            aid = paid_attempts.begin_llm(
                owner_kind, job_id, max_calls=max_calls, lease_sec=lease_sec,
                fence_token=token, fence=fence, sessions=sessions, run_id=run_id)
        except paid_attempts.LedgerFenced:
            return None, None, None, True
        if aid is None:
            with sessions() as db:
                n = paid_attempts.attempt_count(db, owner_kind, job_id)
            return None, None, (
                f"interrupted: {n} attempt(s) ended with an unknown outcome — not retried "
                f"automatically again (AICP-351); the calls may have been billed. Run it again "
                f"only if you need it. Last: {last_unknown}"), False
        ledger = paid_attempts.LlmLedger(aid, scope_key=scope, kind=kind, fence=fence,
                                         sessions=sessions)
        try:
            result = call(ledger)
        except localisation_mod.LocalisationUnknownOutcome as exc:
            last_unknown = str(exc)
            paid_attempts.finish_llm_own_session(aid, "unknown", error=str(exc), sessions=sessions)
            log.warning("%s %s attempt %s: unknown outcome (%s) — retrying within the bound",
                        owner_kind, job_id, aid, exc)
            continue
        except paid_attempts.LedgerFenced as exc:
            paid_attempts.finish_llm_own_session(aid, "unknown", error=str(exc), sessions=sessions)
            log.warning("%s %s attempt %s fenced: %s", owner_kind, job_id, aid, exc)
            return None, None, None, True
        except localisation_mod.LocalisationError as exc:
            paid_attempts.finish_llm_own_session(aid, "failed", error=str(exc), sessions=sessions)
            return None, None, str(exc), False
        except Exception as exc:  # noqa: BLE001 — recorded on the job
            log.exception("%s %s attempt %s: unexpected error", owner_kind, job_id, aid)
            paid_attempts.finish_llm_own_session(aid, "failed", error=str(exc), sessions=sessions)
            return None, None, f"{type(exc).__name__}: {exc}", False
        return result, aid, None, False


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

    scope = spend_mod.transcribe_job_scope(job_id)
    transcript, aid, error, fenced = _llm_attempts(
        owner_kind=OWNER_LLM_TRANSCRIBE, model=ProjectTranscription, job_id=job_id, token=token,
        sf=sf, max_calls=paid_attempts.TRANSCRIBE_CALLS_PER_ATTEMPT * _transcribe_chain_len(),
        lease_sec=paid_attempts.llm_lease_sec(transcription_job_timeout_sec()),
        scope=scope, kind="transcribe",
        call=lambda ledger: localisation_mod.transcribe_video(local, kie=kie, ledger=ledger),
    )
    if fenced:
        log.warning("transcription job %s: claim revoked meanwhile — nothing written", job_id)
        return None

    db = sf()
    try:
        project = db.execute(select(VideoProject).where(
            VideoProject.id == project_id).with_for_update()).scalar_one_or_none()
        job = _fenced(db, ProjectTranscription, job_id, token)
        if job is None:
            db.rollback()
            if aid:
                paid_attempts.finish_llm_own_session(aid, "accepted", error="claim revoked",
                                                     sessions=_sessions(sf))
            log.warning("transcription job %s: claim revoked meanwhile — result discarded", job_id)
            return None
        if aid and not paid_attempts.finish_llm(db, aid, "accepted"):
            # A reaper superseded the attempt: its answer is not this job's.
            # finish_llm acknowledged it (superseded -> stopped): commit that
            # alone, so the sender stops counting as active (§9 item 2).
            db.commit()
            log.warning("transcription job %s: attempt %s superseded — result discarded",
                        job_id, aid)
            return None
        now = utcnow()
        job.claim_token, job.finished_at, job.updated_at = None, now, now
        if error is not None and error.startswith("interrupted:"):
            job.status, job.error = "interrupted", error[:4000]
            if project is not None:
                project.transcript_status, project.transcript_error = TRANSCRIPT_FAILED, error
        elif error is not None:
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
        db.flush()
        paid_attempts.release_llm_scope_if_idle(db, OWNER_LLM_TRANSCRIBE, job_id, scope,
                                                kind="transcription")
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
        # AICP-351 §5.2: 3 attempts x 3 calls (main, 1:1 re-ask, shorten) x the
        # capped model chain, on the job's own scope (released at its end).
        spend_mod.reserve(
            db, spend_mod.localise_job_scope(job.id),
            paid_attempts.llm_reservation_credits(
                paid_attempts.TRANSLATE_CALLS_PER_ATTEMPT, _translate_chain_len()),
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

    scope = spend_mod.localise_job_scope(job_id)

    def _call(ledger):
        response = _compute_localisation_prompt(
            snapshot, translate_budgets=lpj.translate_budgets(), ledger=ledger)
        return response.model_dump(mode="json")

    with lpj._job_translate_timeout():
        result, aid, err, fenced = _llm_attempts(
            owner_kind=OWNER_LLM_TRANSLATE, model=LocalisationPromptJob, job_id=job_id,
            token=token, sf=sf,
            max_calls=paid_attempts.TRANSLATE_CALLS_PER_ATTEMPT * _translate_chain_len(),
            lease_sec=paid_attempts.llm_lease_sec(translate_job_timeout_sec()),
            scope=scope, kind="translate", call=_call,
        )
    if fenced:
        log.warning("translate job %s: claim revoked meanwhile — nothing written", job_id)
        return None
    error = None
    if err is not None:
        error = {"status_code": 503 if err.startswith("interrupted:") else 502, "detail": err}

    db = sf()
    try:
        job = _fenced(db, LocalisationPromptJob, job_id, token)
        if job is None:
            db.rollback()
            if aid:
                paid_attempts.finish_llm_own_session(aid, "accepted", error="claim revoked",
                                                     sessions=_sessions(sf))
            log.warning("translate job %s: claim revoked meanwhile — result discarded", job_id)
            return None
        if aid and not paid_attempts.finish_llm(db, aid, "accepted"):
            db.commit()  # the acknowledgement only (superseded -> stopped)
            log.warning("translate job %s: attempt %s superseded — result discarded", job_id, aid)
            return None
        now = utcnow()
        job.claim_token, job.finished_at, job.updated_at = None, now, now
        if error is None:
            job.status, job.result = "succeeded", result
        elif error["status_code"] == 503:
            job.status, job.error_detail, job.error = "interrupted", error, error["detail"]
        else:
            job.status, job.error_detail, job.error = "failed", error, error["detail"]
        db.flush()
        paid_attempts.release_llm_scope_if_idle(db, OWNER_LLM_TRANSLATE, job_id, scope,
                                                kind="localisation prompt")
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


LOST_ANSWER_MESSAGE = (
    "interrupted: the model answered (billed) but the worker stopped before saving it — "
    "not re-run automatically; run it again only if you need it."
)


def _end_after_definite(db, model, job, project, last, owner_kind, *, now, out):
    """End a dead job whose latest attempt has a DEFINITE outcome (caller's
    transaction): ``failed`` -> the job fails with that error; ``accepted`` ->
    the answer is lost, the job is interrupted. Nothing is re-sent."""
    from .paid_attempts_models import FAILED

    if last.status == FAILED:
        msg = (last.error or "the model call failed")[:4000]
        job.status, job.error = "failed", msg
        if model is LocalisationPromptJob:
            job.error_detail = {"status_code": 502, "detail": msg}
        if project is not None:
            project.transcript_status, project.transcript_error = TRANSCRIPT_FAILED, msg
        key = "failed"
    else:
        job.status, job.error = "interrupted", LOST_ANSWER_MESSAGE
        if model is LocalisationPromptJob:
            job.error_detail = {"status_code": 503, "detail": LOST_ANSWER_MESSAGE}
        if project is not None:
            project.transcript_status = TRANSCRIPT_FAILED
            project.transcript_error = LOST_ANSWER_MESSAGE
        key = "interrupted"
    job.claim_token, job.finished_at, job.updated_at = None, now, now
    db.flush()
    scope = (spend_mod.transcribe_job_scope(job.id) if model is ProjectTranscription
             else spend_mod.localise_job_scope(job.id))
    paid_attempts.release_llm_scope_if_idle(db, owner_kind, job.id, scope, kind=owner_kind)
    db.commit()
    out[key] = out.get(key, 0) + 1
    log.warning("%s %s: worker gone after a definite outcome (%s) — ended %s, not re-billed",
                model.__tablename__, job.id, last.status, job.status)
    return None


def _reap_running(db, model, job, project, *, now, evidence, out):
    """A ``running`` job whose RQ job is not live (caller's transaction; the
    project and job rows are locked). Returns the job to re-enqueue, or None."""
    # running: the worker died mid-call. AICP-351 §5.2: its open
    # attempt decides — sender PROVEN gone -> `unknown`; lease
    # expired without proof -> `superseded` (still counted active
    # until acknowledged / proven); neither -> leave it running.
    owner_kind = (OWNER_LLM_TRANSCRIBE if model is ProjectTranscription
                  else OWNER_LLM_TRANSLATE)
    att = paid_attempts.open_attempt(db, owner_kind, job.id, for_update=True)
    if att is not None:
        decision = paid_attempts.reap_llm_attempt(db, att, now=now, evidence=evidence)
        if decision is None:
            db.rollback()
            out["waiting"] = out.get("waiting", 0) + 1
            return None
        db.flush()  # the latest-attempt read below must see this decision
    tries = paid_attempts.attempt_count(db, owner_kind, job.id)
    last = paid_attempts.latest_attempt(db, owner_kind, job.id)
    # Only an UNKNOWN outcome is retried (§5.2). The latest attempt decides:
    # * unknown / superseded / stopped -> retry within the bound;
    # * failed (a definite outcome the worker committed before dying) -> the
    #   job ends failed with that error — never re-billed;
    # * accepted (billed, answer received, the job row never written) -> the
    #   job ends interrupted with that reason — never re-billed;
    # * none (the worker died before its first call: nothing was sent) ->
    #   retried like an unknown outcome, bounded by the dispatch count.
    if last is not None and last.status not in paid_attempts.RETRYABLE_LLM_STATUSES:
        return _end_after_definite(db, model, job, project, last, owner_kind,
                                   now=now, out=out)
    if (int(job.attempts or 0) < paid_attempts.LLM_MAX_ATTEMPTS
            and tries < paid_attempts.LLM_MAX_ATTEMPTS):
        # Bounded automatic retry: a new dispatch generation, the
        # dead worker's claim revoked in the same statement.
        job.status, job.claim_token = "queued", None
        job.dispatch_gen = int(job.dispatch_gen) + 1
        job.rq_job_id = rq_job_id(job)
        job.not_live_since, job.updated_at = None, now
        job.error = (f"worker gone mid-call — retried automatically "
                     f"(attempt {tries + 1} of {paid_attempts.LLM_MAX_ATTEMPTS})")
        if project is not None:
            project.transcript_status = TRANSCRIPT_PENDING
        db.commit()
        out["retried"] = out.get("retried", 0) + 1
        log.warning("%s %s: worker gone mid-call — bounded retry (AICP-351)",
                    model.__tablename__, job.id)
        return job
    job.status, job.claim_token = "interrupted", None
    job.error, job.finished_at, job.updated_at = INTERRUPTED_MESSAGE, now, now
    if model is LocalisationPromptJob:
        job.error_detail = {"status_code": 503, "detail": INTERRUPTED_MESSAGE}
    if project is not None:
        project.transcript_status = TRANSCRIPT_FAILED
        project.transcript_error = INTERRUPTED_MESSAGE
    db.flush()
    scope = (spend_mod.transcribe_job_scope(job.id) if model is ProjectTranscription
             else spend_mod.localise_job_scope(job.id))
    paid_attempts.release_llm_scope_if_idle(db, owner_kind, job.id, scope,
                                            kind=owner_kind)
    db.commit()
    out["interrupted"] += 1
    log.warning("%s %s interrupted (worker gone mid-call)", model.__tablename__,
                job.id)
    return None


def reap(*, session_factory: Optional[Callable] = None,
         is_live: Optional[Callable[[str, Optional[str]], bool]] = None,
         enqueue_fn: Optional[Callable] = None, now=None,
         evidence: Optional[Callable] = None) -> dict:
    """One reaper pass over every live job row. Returns counters.

    *evidence*: the paid-attempt sender proof (default: Redis,
    ``rebuild_paid_call.default_evidence``)."""
    from .tasks import QUEUE_LOCALISE, QUEUE_PROCESS

    if evidence is None:
        evidence = paid_attempts.default_evidence()
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
                else:
                    requeue = _reap_running(db, model, job, project, now=now,
                                            evidence=evidence, out=out)
            finally:
                db.close()
            if requeue is not None:
                enqueue_fn(requeue)
    return out
