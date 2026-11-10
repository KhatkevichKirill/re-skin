"""Rebuild wave 2b on PostgreSQL (opt-in, RESKIN_TEST_PG_URL — a DISPOSABLE
server, never production), e.g.

    docker run --rm -d --name reskin-rebuild-pgtest-2b -e POSTGRES_PASSWORD=t \\
        -p 127.0.0.1:15513:5432 postgres:15
    RESKIN_TEST_PG_URL=postgresql://postgres:t@127.0.0.1:15513/postgres \\
        pytest tests/test_rebuild_run_pg.py

* concurrent ``submit_images_only`` entries (a launch racing a recovery
  re-entry) admit exactly one attempt per chunk — the run row lock plus the
  live-attempt index, never a second paid row;
* the whole launch (JSON snapshot, run-owned copies, reservation under the
  scope lock) and a Retry's reset + re-baselined reservation on Postgres;
* concurrent Retry admissions of one incomplete run: one wins, the other sees
  ``queued`` and is refused, and the reservation is counted once.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = [
    pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)"),
    pytest.mark.spend_cap,
]

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_rebuild2b_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _db_url(name: str) -> str:
    from sqlalchemy.engine import make_url

    return make_url(_PG_URL).set(database=name).render_as_string(hide_password=False)


def _admin(sql: str) -> None:
    from sqlalchemy import create_engine, text

    eng = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with eng.connect() as c:
        c.execute(text(sql))
    eng.dispose()


@pytest.fixture(scope="module")
def SF():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.models  # noqa: F401  (every table, rebuild ones included)
    from app.db import Base

    name = f"rebuild2b_pg_{uuid.uuid4().hex[:8]}"
    _admin(f'DROP DATABASE IF EXISTS "{name}"')
    _admin(f'CREATE DATABASE "{name}"')
    eng = create_engine(_db_url(name), pool_size=20, max_overflow=0)
    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    eng.dispose()
    _admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(autouse=True)
def env(SF, monkeypatch):
    from app import pipeline_rebuild as pr
    from app import rebuild_paid_call as pc
    from app.config import settings

    @contextmanager
    def sess():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr(pr, "get_session", sess)
    steps = []
    lock = threading.Lock()

    def record(table, row_id, gen, step, delay_sec=0, seq=None):
        with lock:
            steps.append((table, row_id, gen, step))

    monkeypatch.setattr(pc, "enqueue_step", record)
    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0)
    monkeypatch.setattr(settings, "REBUILD_ENABLED", True)
    return steps


def _run_with_chunks(SF, n=2, status="queued"):
    from app.models import Run, RunSegment, SegmentDef, VideoProject
    from app.state_machine import ProjectStatus, SegmentStatus

    s = SF()
    p = VideoProject(source_type="upload", source_ref="x", status=ProjectStatus.ready,
                     project_type="rebuild")
    s.add(p)
    s.flush()
    defs = []
    for i in range(n):
        sd = SegmentDef(project_id=p.id, index=i, start_sec=i * 10, end_sec=(i + 1) * 10,
                        action="swap")
        s.add(sd)
        defs.append(sd)
    s.flush()
    run = Run(project_id=p.id, model="kling-4-0", resolution="720p", status=status,
              generation_mode="images_only",
              segment_snapshot=[{"id": d.id, "index": d.index, "action": "swap",
                                 "start_sec": d.start_sec, "end_sec": d.end_sec} for d in defs],
              rebuild_snapshot={"fps_out": 30.0, "chunks": [
                  {"segment_def_id": d.id, "index": d.index, "request_sec": 10,
                   "chunk_frames": 300} for d in defs]})
    s.add(run)
    s.flush()
    for d in defs:
        s.add(RunSegment(run_id=run.id, segment_def_id=d.id, index=d.index,
                         status=SegmentStatus.pending))
    s.commit()
    rid = run.id
    s.close()
    return rid


def test_concurrent_submit_admits_one_attempt_per_chunk(SF, env):
    from sqlalchemy import select

    from app import pipeline_rebuild as pr
    from app.models import Run, RunSegment
    from app.rebuild_models import RebuildChunkAttempt
    from app.state_machine import RunStatus, SegmentStatus

    rid = _run_with_chunks(SF, n=3)
    n = 6
    barrier = threading.Barrier(n)
    errors = []

    def entry():
        try:
            barrier.wait()
            pr.submit_images_only(rid)
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))

    ts = [threading.Thread(target=entry) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == []
    with SF() as s:
        attempts = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == rid)).scalars().all()
        assert len(attempts) == 3 and {a.attempt_no for a in attempts} == {1}
        assert s.get(Run, rid).status == RunStatus.processing
        assert {rs.status for rs in s.execute(select(RunSegment).where(
            RunSegment.run_id == rid)).scalars()} == {SegmentStatus.uploading}
    assert sorted(st[3] for st in env) == ["claim"] * 3


def _png(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=blue:s=90x160:d=1", "-frames:v", "1", path], check=True)
    return path


def test_launch_and_retry_on_postgres(SF, env, tmp_path):
    from sqlalchemy import select

    from app import pipeline_rebuild as pr
    from app import rebuild_launch, spend
    from app.models import KieSpendReservation, Run, RunSegment, SegmentDef, VideoProject
    from app.rebuild_models import (
        ProjectStoryboard,
        RebuildChunkAttempt,
        RebuildKeyframe,
        RebuildShot,
    )
    from app.state_machine import ProjectStatus, RunStatus, SegmentStatus, transition

    png = _png(str(tmp_path / "k.png"))
    s = SF()
    p = VideoProject(source_type="upload", source_ref="x", status=ProjectStatus.ready,
                     project_type="rebuild", width=90, height=160, fps=30.0, duration_sec=8.0)
    s.add(p)
    s.flush()
    shots = []
    for i, (a, b) in enumerate([(0, 3), (3, 8)]):
        sh = RebuildShot(project_id=p.id, index=i, generation=1, start_sec=a, end_sec=b,
                         frame_sec=(a + b) / 2, characters=["C1"], sheet_key="main",
                         action=f"a{i}")
        s.add(sh)
        shots.append(sh)
    s.add(SegmentDef(project_id=p.id, index=0, start_sec=0, end_sec=8, action="swap"))
    s.add(ProjectStoryboard(project_id=p.id, status="succeeded", shots_generation=1,
                            applied_at=datetime(2026, 11, 3),
                            result={"style": "cartoon", "characters": [{"id": "C1"}]}))
    s.flush()
    for sh in shots:
        dst = str(tmp_path / f"{sh.id}.png")
        shutil.copyfile(png, dst)
        s.add(RebuildKeyframe(project_id=p.id, batch_id="b", shot_id=sh.id, status="succeeded",
                              selected=True, image_path=dst))
    s.add(RebuildKeyframe(project_id=p.id, batch_id="b", sheet_key="main", status="succeeded",
                          selected=True, image_path=png))
    s.commit()
    run, _w = rebuild_launch.launch_run(s, p, rebuild_launch.LaunchRequest(model="kling-4-0"))
    s.commit()
    rid = run.id
    s.close()
    with SF() as s:
        run = s.get(Run, rid)
        ch = run.rebuild_snapshot["chunks"][0]
        assert ch["request_sec"] == 8 and ch["chunk_frames"] == 240
        res = s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend.run_scope(rid))).scalar_one()
        assert res.reserved_mcredits == int((8 * 28.5 + 40) * 1000)
    # The attempt fails (refused, settled to 0): Retry resets + re-baselines.
    pr.submit_images_only(rid)
    with SF() as s:
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == rid)).scalar_one()
        a.status, a.cost_credits = "failed", 0.0
        rs = s.get(RunSegment, a.run_segment_id)
        rs.status = SegmentStatus.failed
        transition(s.get(Run, rid), RunStatus.incomplete)
        s.commit()
    with SF() as s:
        run = s.get(Run, rid)
        s.refresh(run, with_for_update=True)
        pr.reset_for_retry(s, run)
        transition(run, RunStatus.queued)
        spend.reserve_run(s, run, kind="retry run", release_unused=True)
        s.commit()
        res = s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend.run_scope(rid))).scalar_one()
        assert res.reserved_mcredits == int((8 * 28.5 + 40) * 1000)
        assert s.execute(select(RunSegment.status).where(
            RunSegment.run_id == rid)).scalar() == SegmentStatus.pending


def test_concurrent_reroll_admissions_count_once(SF, env):
    """Two re-roll / Retry admissions of one idle run race under the run row
    lock: the loser sees the winner's committed state (no double reset, one
    reservation)."""
    from sqlalchemy import select

    from app import pipeline_rebuild as pr
    from app import spend
    from app.models import KieSpendReservation, Run, RunSegment
    from app.state_machine import RunStatus, SegmentStatus

    rid = _run_with_chunks(SF, n=1, status="incomplete")
    with SF() as s:
        rs = s.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalar_one()
        rs.status = SegmentStatus.failed
        s.commit()
    barrier = threading.Barrier(2)
    outcomes = []

    def admit():
        s = SF()
        try:
            barrier.wait()
            run = s.get(Run, rid)
            s.refresh(run, with_for_update=True)
            if run.status != RunStatus.incomplete:
                outcomes.append("refused")
                s.rollback()
                return
            pr.reset_for_retry(s, run)
            run.status = RunStatus.queued
            spend.reserve_run(s, run, kind="retry run", release_unused=True)
            s.commit()
            outcomes.append("admitted")
        finally:
            s.close()

    ts = [threading.Thread(target=admit) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(outcomes) == ["admitted", "refused"]
    with SF() as s:
        res = s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend.run_scope(rid))).scalar_one()
        assert res.reserved_mcredits == int((10 * 28.5 + 40) * 1000)
