"""
PostgreSQL proofs for the reframe workflow (AICP-251, docs/reframe-design.md
§12): the races SQLite cannot show because it serializes every writer.

* The `preparing` reaper's failure CAS against admission, in both commit
  orders: exactly one of them wins.
* A QA claim against a job deletion: one wins, and a deleted job's QA claim
  changes zero rows.
* Two creates of a generation at once: one 201, one 409 (partial unique index).

Opt-in like test_pipeline_provider_pg.py: RESKIN_TEST_PG_URL must name a
DISPOSABLE database (this drops and recreates the public schema).
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = [
    pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)"),
    pytest.mark.spend_cap,
]

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_reframe_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture(scope="module")
def engine():
    from app.db import Base
    import app.models  # noqa: F401

    eng = create_engine(_PG_URL, pool_size=20, max_overflow=0)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def source():
    from reframe_media import make_source

    return make_source(os.path.join(tempfile.mkdtemp(), "src.mp4"), sec=4.0)


@pytest.fixture()
def env(engine, source, monkeypatch):
    from app import db as db_mod
    from app import pipeline_provider as pp
    from app import storage, tasks
    from app.config import settings
    from app.models import VideoProject
    from app.state_machine import ProjectStatus

    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_mod, "SessionLocal", factory)
    monkeypatch.setattr(tasks, "enqueue_reframe_prepare", lambda j: None)
    monkeypatch.setattr(tasks, "enqueue_fal_job", lambda *a, **k: None)
    monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)
    monkeypatch.setattr(settings, "FAL_ENABLED", True)
    monkeypatch.setattr(settings, "FAL_API_KEY", "test-key")
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 250.0)

    with db_mod.get_session() as s:
        p = VideoProject(source_type="upload", source_ref="s.mp4", project_type="reframe",
                         status=ProjectStatus.ready, width=1280, height=720, duration_sec=4.0,
                         fps=30.0)
        s.add(p)
        s.flush()
        pdir = os.path.join(storage.data_dir_root(), "projects", p.id)
        os.makedirs(pdir, exist_ok=True)
        dst = os.path.join(pdir, "source.mp4")
        with open(source, "rb") as a, open(dst, "wb") as b:
            b.write(a.read())
        p.source_local_path = dst
        pid = p.id
    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", pid)

    class E:
        pass

    e = E()
    e.pid, e.src, e.factory = pid, dst, factory
    yield e


def _new_job(e, *, attempts=3, token="tok", expires_in=3600.0) -> str:
    from app import storage
    from app.db import get_session
    from app.reframe_models import ReframeJob

    with get_session() as s:
        j = ReframeJob(project_id=e.pid, status="preparing", target_aspect="9:16",
                       resolution="480p", window_start_sec=0.0, window_end_sec=3.0,
                       estimated_usd_base=0.15, estimated_usd_reserved=0.3,
                       prep_attempts=attempts, prep_claim_owner=token,
                       prep_claim_expires_at=_utcnow() + timedelta(seconds=expires_in))
        s.add(j)
        s.flush()
        jid = j.id
    os.makedirs(storage.reframe_job_dir(jid, e.pid), exist_ok=True)
    return jid


def _admit_now(e, jid, token="tok"):
    from app import pipeline_reframe as prf
    from app import storage

    jdir = storage.reframe_job_dir(jid, e.pid)
    staging = os.path.join(jdir, f"prep-{token}")
    os.makedirs(staging, exist_ok=True)
    prf._prepare_and_admit(jid, token, e.pid, e.src, 30.0, 0.0, 3.0, "9:16", "480p",
                           jdir, staging)


def _status(jid):
    from app.db import get_session
    from app.reframe_models import ReframeJob

    with get_session() as s:
        j = s.get(ReframeJob, jid)
        return (j.status, j.provider_task_id) if j else None


def test_reaper_cas_after_admission_changes_nothing(env, monkeypatch):
    """Admission holds the project lock (slow destination check); the reaper,
    judging the claim expired, waits on that lock and then finds `admitted`."""
    from app import pipeline_provider as pp
    from app import pipeline_reframe as prf
    from app.db import get_session

    jid = _new_job(env)
    real = pp._check_destination
    entered = threading.Event()

    def slow(*a, **k):
        entered.set()
        time.sleep(0.5)
        return real(*a, **k)

    monkeypatch.setattr(pp, "_check_destination", slow)
    out = {}

    def reaper():
        entered.wait(60)
        with get_session() as s:
            out["reaped"] = prf.fail_crashed_prepare(
                s, jid, env.pid, _utcnow() + timedelta(days=1))

    t = threading.Thread(target=reaper)
    t.start()
    _admit_now(env, jid)
    t.join(60)
    st, task = _status(jid)
    assert out["reaped"] is False
    assert st == "admitted" and task is not None


def test_admission_after_reaper_changes_nothing(env):
    from app import pipeline_reframe as prf
    from app.db import get_session
    from app.provider_models import ProviderTask
    from sqlalchemy import select

    jid = _new_job(env, expires_in=-1)
    with get_session() as s:
        assert prf.fail_crashed_prepare(s, jid, env.pid, _utcnow()) is True
    _admit_now(env, jid)
    assert _status(jid)[0] == "prepare_failed"
    with get_session() as s:
        assert s.execute(select(ProviderTask).where(ProviderTask.owner_id == jid)).first() is None


def test_qa_claim_and_job_delete_one_wins(env, monkeypatch):
    from fastapi import HTTPException

    from app import api_reframe
    from app import pipeline_reframe as prf
    from app.db import get_session

    jid = _new_job(env, attempts=1)
    _admit_now(env, jid)
    with get_session() as s:
        s.execute(text("UPDATE provider_tasks SET status='done', reservation_held=false "
                       "WHERE owner_id=:j"), {"j": jid})
    out = {}
    barrier = threading.Barrier(2)

    def delete():
        barrier.wait()
        try:
            with get_session() as s:
                api_reframe.delete_reframe_job(jid, s)
            out["delete"] = 204
        except HTTPException as exc:
            out["delete"] = exc.status_code

    def qa():
        barrier.wait()
        prf.reframe_qa(jid)  # the raw file does not exist: QA ends `failed` if it claims

    ts = [threading.Thread(target=delete), threading.Thread(target=qa)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    st = _status(jid)
    if out["delete"] == 204:
        assert st is None
    else:
        assert out["delete"] == 409 or st is not None


def test_two_creates_one_wins(env):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    out = []
    barrier = threading.Barrier(2)

    def create():
        barrier.wait()
        r = client.post(f"/api/v2/projects/{env.pid}/reframe-jobs",
                        data={"window_start_sec": 0, "window_end_sec": 3, "resolution": "480p"})
        out.append(r.status_code)

    ts = [threading.Thread(target=create) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert sorted(out) == [201, 409], out
