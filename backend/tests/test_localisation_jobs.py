"""AICP-365: durable transcription / translate jobs (app/localisation_jobs.py).

Dispatch generations, claim fencing, the reaper (lost doorbell -> re-sent;
dead worker -> interrupted, claim revoked, never replayed), the after-analysis
hand-off through the same admission, and Run.mute_source frozen at admission.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import localisation, localisation_jobs
from app.db import Base
from app.localisation_jobs_models import LocalisationPromptJob, ProjectTranscription
from app.models import VideoProject
from app.models_autopilot import utcnow
from app.state_machine import ProjectStatus


@pytest.fixture(scope="module")
def SF():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    Base.metadata.drop_all(eng)


def _project(SF, tmp_path, **kw):
    src = tmp_path / f"src-{os.urandom(3).hex()}.mp4"
    src.write_bytes(b"x")
    db = SF()
    p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                     status=ProjectStatus.ready, project_type="localisation", **kw)
    db.add(p)
    db.commit()
    pid = p.id
    db.close()
    return pid


def _admit(SF, pid, origin="manual"):
    db = SF()
    try:
        job = localisation_jobs.admit_transcription(db, db.get(VideoProject, pid), origin=origin)
        db.commit()
        return job
    finally:
        db.close()


def _get(SF, model, jid):
    db = SF()
    try:
        return db.get(model, jid)
    finally:
        db.close()


@pytest.fixture()
def transcribe_ok(monkeypatch):
    calls = []

    def fake(path, kie=None, spend_scope=None, **kw):
        calls.append(path)
        return {"schema_version": 3, "source_language": "en",
                "lines": [{"id": 1, "start": 0.0, "end": 1.0, "speaker": "a", "text": "hi"}]}

    monkeypatch.setattr(localisation, "transcribe_video", fake)
    return calls


def test_admission_is_one_live_job_per_project(SF, tmp_path):
    pid = _project(SF, tmp_path)
    job = _admit(SF, pid)
    assert (job.status, job.dispatch_gen, job.rq_job_id) == (
        "queued", 1, f"transcribe-db-{job.id}-g1")
    assert _get(SF, VideoProject, pid).transcript_status == "pending"
    with pytest.raises(HTTPException) as exc:
        _admit(SF, pid, origin="autopilot")
    assert exc.value.status_code == 409 and "already queued" in exc.value.detail


def test_no_source_is_409_and_writes_nothing(SF, tmp_path):
    pid = _project(SF, tmp_path)
    db = SF()
    p = db.get(VideoProject, pid)
    p.source_local_path = None
    db.commit()
    db.close()
    with pytest.raises(HTTPException) as exc:
        _admit(SF, pid)
    assert exc.value.status_code == 409


def test_worker_claims_once_and_writes_under_its_claim(SF, tmp_path, transcribe_ok):
    pid = _project(SF, tmp_path)
    job = _admit(SF, pid)
    assert localisation_jobs.run_transcription_job(job.id, 1, session_factory=SF) == "succeeded"
    # A duplicate delivery of the same generation claims nothing, calls nothing.
    assert localisation_jobs.run_transcription_job(job.id, 1, session_factory=SF) is None
    assert len(transcribe_ok) == 1
    p = _get(SF, VideoProject, pid)
    assert p.transcript_status == "ready" and p.transcript["schema_version"] == 3
    row = _get(SF, ProjectTranscription, job.id)
    assert row.status == "succeeded" and row.claim_token is None and row.attempts == 1


def test_stale_generation_claims_nothing(SF, tmp_path, transcribe_ok):
    pid = _project(SF, tmp_path)
    job = _admit(SF, pid)
    db = SF()
    db.get(ProjectTranscription, job.id).dispatch_gen = 2  # the reaper re-sent it
    db.commit()
    db.close()
    assert localisation_jobs.run_transcription_job(job.id, 1, session_factory=SF) is None
    assert transcribe_ok == []
    assert localisation_jobs.run_transcription_job(job.id, 2, session_factory=SF) == "succeeded"


def test_failure_keeps_the_previous_transcript(SF, tmp_path, monkeypatch):
    pid = _project(SF, tmp_path, transcript={"lines": [{"id": 1}]}, transcript_status="ready")
    job = _admit(SF, pid)

    def boom(*a, **k):
        raise localisation.LocalisationError("down")

    monkeypatch.setattr(localisation, "transcribe_video", boom)
    assert localisation_jobs.run_transcription_job(job.id, 1, session_factory=SF) == "failed"
    p = _get(SF, VideoProject, pid)
    assert p.transcript_status == "failed" and p.transcript_error == "down"
    assert p.transcript == {"lines": [{"id": 1}]}


def test_reaper_resends_a_lost_doorbell(SF, tmp_path):
    pid = _project(SF, tmp_path)
    job = _admit(SF, pid)
    sent = []
    t0 = utcnow()
    kw = dict(session_factory=SF, is_live=lambda q, j: False, enqueue_fn=sent.append)
    assert localisation_jobs.reap(now=t0, **kw)["observed"] >= 1
    assert sent == []  # first sighting only starts the grace
    out = localisation_jobs.reap(now=t0 + timedelta(seconds=121), **kw)
    assert out["requeued"] >= 1
    row = _get(SF, ProjectTranscription, job.id)
    assert (row.status, row.dispatch_gen, row.rq_job_id) == ("queued", 2,
                                                             f"transcribe-db-{job.id}-g2")
    assert any(j.id == job.id for j in sent)


def test_a_live_job_is_left_alone(SF, tmp_path):
    pid = _project(SF, tmp_path)
    job = _admit(SF, pid)
    t0 = utcnow()
    for dt in (0, 200, 400):
        localisation_jobs.reap(session_factory=SF, is_live=lambda q, j: True,
                               enqueue_fn=lambda j: None, now=t0 + timedelta(seconds=dt))
    row = _get(SF, ProjectTranscription, job.id)
    assert row.status == "queued" and row.dispatch_gen == 1


def test_dead_worker_is_retried_and_its_late_write_discarded(SF, tmp_path, monkeypatch):
    """AICP-351 §5.2: a worker proven gone mid-call is retried automatically (a
    new dispatch generation, its claim revoked) — no longer interrupted at once —
    and its late answer never lands. The 3-attempt bound and the final
    `interrupted` state are covered in tests/test_paid_attempts.py."""
    pid = _project(SF, tmp_path)
    job = _admit(SF, pid)
    state = {}

    def slow(path, kie=None, spend_scope=None, **kw):
        # While "the model thinks", the worker is proven dead (its container
        # restarted) and the reaper takes the job back.
        t0 = utcnow()
        kwr = dict(session_factory=SF, is_live=lambda q, j: False, enqueue_fn=lambda j: None,
                   evidence=lambda row: "sender container restarted")
        localisation_jobs.reap(now=t0, **kwr)
        state["out"] = localisation_jobs.reap(now=t0 + timedelta(seconds=121), **kwr)
        return {"schema_version": 3, "lines": [{"id": 1, "start": 0, "end": 1, "text": "late"}]}

    monkeypatch.setattr(localisation, "transcribe_video", slow)
    assert localisation_jobs.run_transcription_job(job.id, 1, session_factory=SF) is None
    assert state["out"]["retried"] >= 1
    row = _get(SF, ProjectTranscription, job.id)
    assert (row.status, row.claim_token, row.dispatch_gen) == ("queued", None, 2)
    p = _get(SF, VideoProject, pid)
    assert p.transcript_status == "pending"
    assert p.transcript is None  # the late answer never landed


def test_dead_worker_without_proof_is_left_running(SF, tmp_path, monkeypatch):
    """No proof the sender is gone and its lease has not run out: the reaper
    waits (the job stays running and counted active)."""
    pid = _project(SF, tmp_path)
    job = _admit(SF, pid)
    state = {}

    def slow(path, kie=None, spend_scope=None, **kw):
        t0 = utcnow()
        kwr = dict(session_factory=SF, is_live=lambda q, j: False, enqueue_fn=lambda j: None,
                   evidence=lambda row: None)
        localisation_jobs.reap(now=t0, **kwr)
        state["out"] = localisation_jobs.reap(now=t0 + timedelta(seconds=121), **kwr)
        return {"schema_version": 3, "lines": [{"id": 1, "start": 0, "end": 1, "text": "ok"}]}

    monkeypatch.setattr(localisation, "transcribe_video", slow)
    assert localisation_jobs.run_transcription_job(job.id, 1, session_factory=SF) == "succeeded"
    assert state["out"].get("waiting") >= 1


def test_translate_job_fenced_and_reaped(SF, tmp_path, monkeypatch):
    pid = _project(SF, tmp_path)
    db = SF()
    job = LocalisationPromptJob(project_id=pid, language="es", request={}, snapshot={"x": 1},
                                fingerprint="f", status="queued", dispatch_gen=1, attempts=0)
    db.add(job)
    db.flush()
    job.rq_job_id = localisation_jobs.rq_job_id(job)
    db.commit()
    jid = job.id
    db.close()
    assert job.rq_job_id == f"localise-prompt-db-{jid}-g1"

    import app.api_v2 as api_v2_module

    monkeypatch.setattr(api_v2_module, "_compute_localisation_prompt",
                        lambda snap, translate_budgets=None, **_kw: SimpleNamespace(
                            model_dump=lambda mode=None: {"lines": [], "warnings": [], "snap": snap}))
    assert localisation_jobs.run_translate_job(jid, 2, session_factory=SF) is None  # wrong gen
    assert localisation_jobs.run_translate_job(jid, 1, session_factory=SF) == "succeeded"
    row = _get(SF, LocalisationPromptJob, jid)
    assert row.result["snap"] == {"x": 1} and row.claim_token is None


def test_translate_failure_is_a_502_record(SF, tmp_path, monkeypatch):
    pid = _project(SF, tmp_path)
    db = SF()
    job = LocalisationPromptJob(project_id=pid, language="pt", request={}, snapshot={},
                                fingerprint="f", status="queued", dispatch_gen=1, attempts=0)
    db.add(job)
    db.commit()
    jid = job.id
    db.close()
    import app.api_v2 as api_v2_module

    def boom(snap, translate_budgets=None, **_kw):
        raise localisation.LocalisationError("model refused")

    monkeypatch.setattr(api_v2_module, "_compute_localisation_prompt", boom)
    assert localisation_jobs.run_translate_job(jid, 1, session_factory=SF) == "failed"
    row = _get(SF, LocalisationPromptJob, jid)
    assert row.error_detail == {"status_code": 502, "detail": "model refused"}


def test_after_analysis_handoff_uses_the_durable_admission(SF, tmp_path, monkeypatch):
    from app import tasks
    import app.db as app_db

    @contextmanager
    def _gs():
        s = SF()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr(app_db, "get_session", _gs)
    sent = []
    monkeypatch.setattr(localisation_jobs, "enqueue", sent.append)
    pid = _project(SF, tmp_path)
    tasks._enqueue_transcribe_after_analyze(pid)
    assert [j.origin for j in sent] == ["analysis"]
    # A transcription already live (the operator's click) wins: nothing twice.
    tasks._enqueue_transcribe_after_analyze(pid)
    assert len(sent) == 1
    db = SF()
    assert len(db.execute(select(ProjectTranscription).where(
        ProjectTranscription.project_id == pid)).scalars().all()) == 1
    db.close()


def test_effective_mute_source():
    from app.pipeline_v2 import effective_mute_source

    on, off = SimpleNamespace(mute_source=True), SimpleNamespace(mute_source=False)
    assert effective_mute_source(SimpleNamespace(mute_source=None), on) is True
    assert effective_mute_source(SimpleNamespace(mute_source=None), off) is False
    assert effective_mute_source(SimpleNamespace(mute_source=True), off) is True
    assert effective_mute_source(SimpleNamespace(mute_source=False), on) is False
