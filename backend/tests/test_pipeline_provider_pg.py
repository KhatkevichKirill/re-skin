"""
PostgreSQL proofs for the Fal provider layer (AICP-330): what SQLite cannot
show because it serializes every writer through one database lock.

* B2 — the live migration: rows written before it, an OLD writer (that does not
  name ``provider``) after it, then the new spend code; and the ALTER failing
  cleanly on its lock_timeout instead of queueing behind a long transaction.
* M4 — close and backfill on the same row, concurrently: no deadlock, one draw.
* The insert-only triggers and the one-active-task partial index on PG.
* Concurrent admissions of one operation (one row, one reservation) and an
  admission racing ``delete_project`` (never both).

Opt-in like test_spend_pg_concurrency.py: RESKIN_TEST_PG_URL must name a
DISPOSABLE database (this creates and drops every table), e.g.

    docker run --rm -d --name reskin-fal-pgtest -e POSTGRES_PASSWORD=t \\
        -p 127.0.0.1:15511:5432 postgres:16-alpine
    RESKIN_TEST_PG_URL=postgresql://postgres:t@127.0.0.1:15511/postgres \\
        pytest tests/test_pipeline_provider_pg.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = [
    pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)"),
    pytest.mark.spend_cap,
]

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_fal_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, inspect, select, text  # noqa: E402
from sqlalchemy.exc import DBAPIError, IntegrityError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


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


def _migration():
    from tests.test_pipeline_provider import _migration as m

    return m()


@pytest.fixture()
def env(engine, monkeypatch):
    from app import pipeline_provider as pp
    from app.config import settings
    from app.models import Run, VideoProject
    from app.state_machine import ProjectStatus, RunStatus
    from tests.test_pipeline_provider import FakeClient

    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

    @contextmanager
    def sessions():
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    client = FakeClient()
    monkeypatch.setattr(pp, "_sessions", sessions)
    monkeypatch.setattr(pp, "_client", lambda: client)
    monkeypatch.setattr(pp, "_enqueue", lambda *a, **k: None)
    monkeypatch.setattr(pp, "_probe_duration", lambda path: 5.0)
    monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)
    monkeypatch.setattr(settings, "FAL_ENABLED", True)
    monkeypatch.setattr(settings, "FAL_API_KEY", "test-key")
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 250.0)
    with sessions() as s:
        p = VideoProject(source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready)
        s.add(p)
        s.flush()
        r = Run(project_id=p.id, prompt="x", model="seedance", resolution="720p", status=RunStatus.done)
        s.add(r)
        s.flush()
        pid, rid = p.id, r.id
    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", pid)
    from app import storage

    proj = os.path.join(storage.data_dir_root(), "projects", pid)
    os.makedirs(os.path.join(proj, "reframe"), exist_ok=True)
    src = os.path.join(proj, "source.mp4")
    with open(src, "wb") as fh:
        fh.write(b"source-bytes")

    class E:
        pass

    e = E()
    e.factory, e.sessions, e.client, e.pid, e.rid, e.proj, e.src = factory, sessions, client, pid, rid, proj, src
    yield e


def _admit(e, n, op=None):
    from app import pipeline_provider as pp

    with e.sessions() as s:
        task, created = pp.admit(
            s, endpoint_id="minimax/h3-max/reframe-video", params={"aspect_ratio": "9:16"},
            inputs=[pp.TaskInput("video_url", e.src, "video/mp4")], output_sec=5.0,
            resolution="768p", owner_kind="reframe_job", owner_id=f"pg{n}",
            owner_project_id=e.pid, destination_path=os.path.join(e.proj, "reframe", f"pg{n}.mp4"),
            operation_key=op or f"reframe_job:pg{n}:main:1")
        return task.id, created


def test_b2_live_migration_with_old_writers_and_lock_timeout(engine):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from app import spend
    from app.models import KieSpendDraw

    mig = _migration()
    # Back to the l2b8d4f0a316 shape of these tables (no provider_tasks rows yet).
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM kie_spend_draws; DELETE FROM kie_spend_reservations"))
        with Operations.context(MigrationContext.configure(conn)):
            mig.downgrade()
        conn.execute(text("INSERT INTO kie_spend_reservations (id, scope_key, day, kind, "
                          "reserved_mcredits, consumed_mcredits, charged_mcredits, created_at, "
                          "updated_at) VALUES ('r0','run:old',current_date,'run_create',1000,0,1000,"
                          "now(),now())"))
    # A long transaction holding a lock on the table: the migration must fail
    # on its lock_timeout, not queue (and block every writer behind it).
    blocker = engine.connect()
    btx = blocker.begin()
    blocker.execute(text("LOCK TABLE kie_spend_draws IN ROW EXCLUSIVE MODE"))
    t0 = time.monotonic()
    with pytest.raises(DBAPIError, match="lock timeout"):
        with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
            mig.upgrade()
    assert time.monotonic() - t0 < 30
    btx.rollback()
    blocker.close()
    with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
        mig.upgrade()
    with engine.begin() as conn:
        # the old code's insert, after the migration
        conn.execute(text("INSERT INTO kie_spend_draws (id, call_key, reservation_id, kind, "
                          "estimated_mcredits, mcredits, created_at) VALUES "
                          "('d0','old-task','r0','video',5,5,now())"))
        assert conn.execute(text("SELECT provider FROM kie_spend_draws WHERE id='d0'")).scalar() == "kie"
        assert conn.execute(text("SELECT provider FROM kie_spend_reservations WHERE id='r0'")).scalar() == "kie"
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def sessions():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    with sessions() as s:
        spend.reserve(s, "run:new", 100, kind="run_create")
        s.commit()
    spend.record_draw("run:new", 40, kind="video", call_key="kie-pg-1", sessions=sessions)
    spend.settle_draw("kie-pg-1", 30, scope_key="run:new", kind="video", sessions=sessions)
    with sessions() as s:
        d = s.execute(select(KieSpendDraw).where(KieSpendDraw.call_key == "kie-pg-1")).scalar_one()
        assert (d.provider, d.mcredits) == ("kie", 30000)


def test_pg_triggers_and_active_index(env):
    from app import pipeline_provider as pp

    tid, _ = _admit(env, 100)
    with env.sessions() as s:
        pp.abandon(s, tid, actor="ops", note="never sent")
    with env.factory() as s:
        with pytest.raises(DBAPIError, match="insert-only"):
            s.execute(text("UPDATE provider_task_actions SET note='x'"))
        s.rollback()
        with pytest.raises(DBAPIError, match="insert-only"):
            s.execute(text("DELETE FROM provider_task_actions"))
        s.rollback()
    # the index: a held refused row still occupies the slot
    tid2, _ = _admit(env, 101, op="reframe_job:slot:main:1")
    with env.factory() as s:
        s.execute(text("UPDATE provider_tasks SET status='refused', reservation_held=true, "
                       "owner_id='slot' WHERE id=:i"), {"i": tid2})
        s.commit()
    with pytest.raises(pp.AdmissionRefused) as exc:
        with env.sessions() as s:
            pp.admit(s, endpoint_id="minimax/h3-max/reframe-video", params={},
                     inputs=[pp.TaskInput("video_url", env.src, "video/mp4")], output_sec=5.0,
                     resolution="768p", owner_kind="reframe_job", owner_id="slot",
                     owner_project_id=env.pid,
                     destination_path=os.path.join(env.proj, "reframe", "slot2.mp4"),
                     operation_key="reframe_job:slot:main:2")
    assert exc.value.status_code == 409


def test_m4_close_and_backfill_on_one_row_do_not_deadlock(env, monkeypatch):
    from app import pipeline_provider as pp
    from app import spend
    from app.fal_client import SubmitOutcome

    real = spend.draw_in_session

    def slow_draw(*a, **k):
        time.sleep(0.3)  # widen the window between the row lock and the day lock
        return real(*a, **k)

    for n in range(5):
        tid, _ = _admit(env, 200 + n)
        env.client.submit_outcome = SubmitOutcome("unknown", None, error="timeout")
        pp.fal_submit(tid)
        with env.sessions() as s:
            pp.close(s, tid, actor="ops", note="unsure")
        # the race: the close's last transaction vs a backfill of the same row
        with env.factory() as s:
            s.execute(text("UPDATE provider_tasks SET draw_pending=true WHERE id=:i"), {"i": tid})
            s.commit()
        monkeypatch.setattr(spend, "draw_in_session", slow_draw)
        errors = []
        barrier = threading.Barrier(2)

        def run(fn):
            try:
                barrier.wait()
                fn(tid)
            except Exception as exc:  # pragma: no cover - the failure we look for
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(pp.fal_close,)),
                   threading.Thread(target=run, args=(pp._backfill,))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        monkeypatch.setattr(spend, "draw_in_session", real)
        assert not errors and not any(t.is_alive() for t in threads)
        with env.sessions() as s:
            from app.models import KieSpendDraw
            from app.provider_models import ProviderTask

            draws = s.execute(select(KieSpendDraw).where(
                KieSpendDraw.call_key == spend.provider_task_call_key(tid))).scalars().all()
            row = s.get(ProviderTask, tid)
            assert len(draws) == 1
            assert (row.status, row.draw_pending, row.reservation_held) == ("failed", False, False)


def test_concurrent_admissions_of_one_operation(env):
    from app import pipeline_provider as pp
    from app.models import KieSpendReservation
    from app.provider_models import ProviderTask

    results, errors = [], []
    barrier = threading.Barrier(6)

    def run():
        try:
            barrier.wait()
            results.append(_admit(env, 300, op="reframe_job:pg300:main:1"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    assert len({tid for tid, _ in results}) == 1 and sum(c for _, c in results) == 1
    with env.sessions() as s:
        tid = results[0][0]
        assert s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == f"fal:{tid}")).scalars().all().__len__() == 1


def test_admission_racing_delete_project_never_both(env, monkeypatch):
    from fastapi import HTTPException

    from app import api_v2
    from app import pipeline_provider as pp

    real = pp._check_destination

    def slow(*a, **k):
        time.sleep(0.3)  # admission holds the project lock here
        return real(*a, **k)

    monkeypatch.setattr(pp, "_check_destination", slow)
    out = {}
    barrier = threading.Barrier(2)

    def admit():
        barrier.wait()
        try:
            out["admit"] = _admit(env, 400)
        except pp.AdmissionRefused as exc:
            out["admit"] = exc.status_code

    def delete():
        barrier.wait()
        time.sleep(0.05)
        try:
            with env.sessions() as s:
                api_v2.delete_project(env.pid, s)
            out["delete"] = 204
        except HTTPException as exc:
            out["delete"] = exc.status_code

    ts = [threading.Thread(target=admit), threading.Thread(target=delete)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    admitted = isinstance(out.get("admit"), tuple)
    assert admitted != (out.get("delete") == 204), out
