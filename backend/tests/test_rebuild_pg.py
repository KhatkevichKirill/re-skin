"""Rebuild wave 1 on PostgreSQL — production's database (design §4, §7.2, §7.3, §14).

Opt-in like tests/test_spend_pg_concurrency.py: set RESKIN_TEST_PG_URL to a
DISPOSABLE Postgres server (never production — this creates and drops
databases), e.g.

    docker run --rm -d --name reskin-rebuild-pgtest -e POSTGRES_PASSWORD=t \\
        -p 127.0.0.1:15511:5432 postgres:15
    RESKIN_TEST_PG_URL=postgresql://postgres:t@127.0.0.1:15511/postgres \\
        pytest tests/test_rebuild_pg.py

* the migration rehearsal: the whole Alembic chain to t4d9a2c6e815 (main's head), a
  representative v2v row, ``upgrade head`` (the new enum labels unused in the
  same transaction), rebuild rows incl. an ``unknown`` attempt, the downgrade
  refused, then allowed once the feature is empty; the ORM vs the migrated
  schema for every rebuild table;
* the partial unique indexes under true concurrency (one live attempt per
  chunk, one selected keyframe per shot);
* the scope lock (round 9): a draw that would pick yesterday's allowance
  waits on the scope lock and lands on today's re-baselined reservation;
* reserve_computed interleavings (round 8): a draw already committed is seen
  as drawn; one waiting on the lock is counted as undrawn and lands inside
  the reservation without a second charge.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = [
    pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)"),
    pytest.mark.spend_cap,
]

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_rebuild_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKEND = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DAY1 = datetime(2026, 11, 2, 23, 30, tzinfo=timezone.utc)
DAY2 = datetime(2026, 11, 3, 0, 30, tzinfo=timezone.utc)


def _db_url(name: str) -> str:
    from sqlalchemy.engine import make_url

    return make_url(_PG_URL).set(database=name).render_as_string(hide_password=False)


def _create_db(name: str):
    from sqlalchemy import create_engine, text

    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        c.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()


def _drop_db(name: str):
    from sqlalchemy import create_engine, text

    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


# ---------------------------------------------------------------------------
# Migration rehearsal
# ---------------------------------------------------------------------------


def test_migration_rehearsal_on_postgres(monkeypatch):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine, text

    from app.config import settings
    from app.db import Base

    name = f"rebuild_mig_{uuid.uuid4().hex[:8]}"
    _create_db(name)
    url = _db_url(name)
    monkeypatch.setattr(settings, "DATABASE_URL", url)  # alembic/env.py reads it
    cfg = Config(os.path.join(BACKEND, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(BACKEND, "alembic"))
    eng = create_engine(url)
    try:
        command.upgrade(cfg, "t4d9a2c6e815")
        with eng.begin() as c:
            c.execute(text(
                "INSERT INTO video_projects (id, project_type, source_type, source_ref, status, "
                "mute_source, mode) VALUES ('p1', 'face_swap', 'upload', 'x', "
                "'ready', false, 'manual')"))
            c.execute(text(
                "INSERT INTO runs (id, project_id, prompt, reference_image_urls, model, "
                "resolution, audio_mode, status, preprocess_enabled, preprocess_auto_approve) "
                "VALUES ('r1', 'p1', '', '[]', 'seedance', '720p', 'original', 'done', false, "
                "false)"))
        command.upgrade(cfg, "head")
        with eng.begin() as c:
            assert c.execute(text("SELECT generation_mode FROM runs WHERE id='r1'")).scalar() == "v2v"
            labels = {r[0] for r in c.execute(text(
                "SELECT enumlabel FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid "
                "WHERE t.typname IN ('project_type_enum', 'run_model_enum')"))}
            assert {"rebuild", "kling-4-0"} <= labels
        # The ORM matches the migrated schema for everything this feature added.
        with eng.connect() as c:
            diffs = compare_metadata(MigrationContext.configure(c), Base.metadata)
        ours = ("rebuild_", "project_storyboards")

        def touches(d):
            flat = repr(d)
            return any(t in flat for t in ours) or "generation_mode" in flat \
                or "rebuild_snapshot" in flat
        assert [d for d in diffs if touches(d)] == []
        # Representative rows: a rebuild project, an images-only run, an
        # `unknown` attempt (a possibly billed task).
        with eng.begin() as c:
            c.execute(text(
                "INSERT INTO video_projects (id, project_type, source_type, source_ref, status, "
                "mute_source, mode) VALUES ('p2', 'rebuild', 'upload', 'x', "
                "'ready', false, 'manual')"))
            c.execute(text(
                "INSERT INTO runs (id, project_id, prompt, reference_image_urls, model, "
                "resolution, audio_mode, status, preprocess_enabled, preprocess_auto_approve, "
                "generation_mode, rebuild_snapshot) VALUES ('r2', 'p2', '', '[]', 'kling-4-0', "
                "'720p', 'original', 'incomplete', false, false, 'images_only', '{}')"))
            c.execute(text(
                "INSERT INTO run_segments (id, run_id, segment_def_id, index, status, "
                "source_fallback) VALUES ('rs2', 'r2', 'sd2', 0, 'failed', false)"))
            c.execute(text(
                "INSERT INTO rebuild_chunk_attempts (id, run_id, run_segment_id, attempt_no, "
                "model, resolution, request_sec, chunk_frames, selected, status, dispatch_gen, "
                "estimate_credits, fetch_attempts, created_at, updated_at) VALUES ('a1', 'r2', "
                "'rs2', 1, 'kling-4-0', '720p', 15, 441, false, 'unknown', 1, 427.5, 0, now(), "
                "now())"))
        with pytest.raises(RuntimeError, match="refusing to downgrade"):
            command.downgrade(cfg, "t4d9a2c6e815")
        with eng.begin() as c:
            assert c.execute(text("SELECT count(*) FROM rebuild_chunk_attempts")).scalar() == 1
            c.execute(text("DELETE FROM runs WHERE id='r2'"))  # cascades the attempt
            c.execute(text("DELETE FROM video_projects WHERE id='p2'"))
        command.downgrade(cfg, "t4d9a2c6e815")
        with eng.begin() as c:
            assert c.execute(text(
                "SELECT count(*) FROM information_schema.tables WHERE table_name LIKE 'rebuild_%'"
            )).scalar() == 0
            assert c.execute(text("SELECT count(*) FROM runs")).scalar() == 1
        command.upgrade(cfg, "head")  # labels already present: IF NOT EXISTS
    finally:
        eng.dispose()
        _drop_db(name)


# ---------------------------------------------------------------------------
# A full schema for the concurrency tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def SF():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db import Base

    name = f"rebuild_pg_{uuid.uuid4().hex[:8]}"
    _create_db(name)
    eng = create_engine(_db_url(name), pool_size=20, max_overflow=0)
    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    eng.dispose()
    _drop_db(name)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0)


def _chunk_run(SF):
    from app.models import Run, RunSegment, SegmentDef, VideoProject
    from app.state_machine import ProjectStatus, RunStatus, SegmentStatus

    s = SF()
    p = VideoProject(source_type="upload", source_ref="x", status=ProjectStatus.ready,
                     project_type="rebuild")
    s.add(p)
    s.flush()
    sd = SegmentDef(project_id=p.id, index=0, start_sec=0, end_sec=10, action="swap")
    s.add(sd)
    s.flush()
    run = Run(project_id=p.id, model="kling-4-0", resolution="720p", status=RunStatus.processing,
              generation_mode="images_only",
              segment_snapshot=[{"id": sd.id, "index": 0, "action": "swap", "start_sec": 0,
                                 "end_sec": 10}],
              rebuild_snapshot={"chunks": [{"segment_def_id": sd.id, "request_sec": 10}]})
    s.add(run)
    s.flush()
    rs = RunSegment(run_id=run.id, segment_def_id=sd.id, index=0, status=SegmentStatus.pending)
    s.add(rs)
    s.commit()
    ids = (p.id, run.id, rs.id, sd.id)
    s.close()
    return ids


def test_only_one_live_attempt_per_chunk_under_concurrency(SF):
    from sqlalchemy.exc import IntegrityError

    from app.rebuild_models import RebuildChunkAttempt

    _pid, rid, rsid, _sd = _chunk_run(SF)
    n = 8
    barrier = threading.Barrier(n)
    ok, dup, other = [], [], []

    def admit(i):
        s = SF()
        try:
            barrier.wait()
            s.add(RebuildChunkAttempt(run_id=rid, run_segment_id=rsid, attempt_no=i + 1,
                                      model="kling-4-0", resolution="720p", request_sec=10,
                                      chunk_frames=300, status="queued"))
            s.commit()
            ok.append(i)
        except IntegrityError:
            s.rollback()
            dup.append(i)
        except Exception as exc:  # pragma: no cover
            s.rollback()
            other.append(repr(exc))
        finally:
            s.close()

    ts = [threading.Thread(target=admit, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert other == [] and len(ok) == 1 and len(dup) == n - 1


def test_keyframe_selection_is_unique_under_concurrent_select(SF):
    """Auto-select racing a manual select: both take the project row lock and
    do unselect-slot + select-row in one transaction, so they serialise and
    exactly one keyframe ends up selected; without the lock the partial unique
    index still refuses a second selected row."""
    from sqlalchemy import select, update
    from sqlalchemy.exc import IntegrityError

    from app.models import VideoProject
    from app.rebuild_models import RebuildKeyframe, RebuildShot

    pid, _rid, _rs, _sd = _chunk_run(SF)
    s = SF()
    shot = RebuildShot(project_id=pid, index=0, generation=1, start_sec=0, end_sec=1,
                       frame_sec=0.5)
    s.add(shot)
    s.flush()
    kfs = [RebuildKeyframe(project_id=pid, batch_id="b", shot_id=shot.id, status="succeeded")
           for _ in range(6)]
    s.add_all(kfs)
    s.commit()
    shot_id, kf_ids = shot.id, [k.id for k in kfs]
    s.close()
    barrier = threading.Barrier(len(kf_ids))
    errors = []

    def select_locked(kid):
        s = SF()
        try:
            barrier.wait()
            s.execute(select(VideoProject).where(VideoProject.id == pid).with_for_update())
            s.execute(update(RebuildKeyframe).where(RebuildKeyframe.shot_id == shot_id,
                                                    RebuildKeyframe.owner_run_id.is_(None))
                      .values(selected=False))
            s.execute(update(RebuildKeyframe).where(RebuildKeyframe.id == kid)
                      .values(selected=True))
            s.commit()
        except Exception as exc:  # pragma: no cover
            s.rollback()
            errors.append(repr(exc))
        finally:
            s.close()

    ts = [threading.Thread(target=select_locked, args=(k,)) for k in kf_ids]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == []
    s = SF()
    assert s.execute(select(RebuildKeyframe.id).where(
        RebuildKeyframe.shot_id == shot_id, RebuildKeyframe.selected.is_(True))).all().__len__() == 1
    # Without the lock: a second selected row is refused by the index.
    with pytest.raises(IntegrityError):
        s.execute(update(RebuildKeyframe).where(RebuildKeyframe.id.in_(kf_ids))
                  .values(selected=True))
        s.commit()
    s.rollback()
    s.close()


# ---------------------------------------------------------------------------
# Scope lock (design §7.3, round 9) and reserve_computed (round 8)
# ---------------------------------------------------------------------------


def _reservations(SF, scope):
    from sqlalchemy import select

    from app.models import KieSpendReservation

    s = SF()
    try:
        return {r.day.isoformat(): (r.reserved_mcredits // 1000, r.consumed_mcredits // 1000)
                for r in s.execute(select(KieSpendReservation).where(
                    KieSpendReservation.scope_key == scope)).scalars()}
    finally:
        s.close()


@pytest.mark.parametrize("scope_lock", [True, False])
def test_draw_waits_on_the_scope_lock_while_a_two_day_rebaseline_runs(SF, scope_lock):
    """Yesterday's reservation (1000, nothing drawn) is re-baselined onto today
    (300) by a Retry's reserve_computed(release_unused) while a sender draws
    200. With the scope lock the draw waits and lands on TODAY; without it
    (the v2v draw path) it picks yesterday before the re-baseline commits and
    lands as overdraft on a released allowance — the race round 9 found."""
    from app import spend

    scope = f"run:{uuid.uuid4()}"
    s = SF()
    spend.reserve_computed(s, scope, lambda db: 1000.0, kind="launch", now=DAY1)
    s.commit()
    s.close()
    in_estimate, release = threading.Event(), threading.Event()
    errors = []

    def rebaseline():
        s = SF()
        try:
            def est(db):
                in_estimate.set()
                release.wait(10)
                return 300.0

            spend.reserve_computed(s, scope, est, kind="retry", release_unused=True, now=DAY2)
            s.commit()
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))
            s.rollback()
        finally:
            s.close()

    def draw():
        s = SF()
        try:
            spend.draw_in_session(s, scope, f"rebuild:{uuid.uuid4()}", 200.0,
                                  kind="rebuild chunk", provider="kie", now=DAY2,
                                  scope_lock=scope_lock)
            s.commit()
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))
            s.rollback()
        finally:
            s.close()

    a = threading.Thread(target=rebaseline)
    a.start()
    assert in_estimate.wait(10)
    b = threading.Thread(target=draw)
    b.start()
    time.sleep(1.0)  # let the draw choose its day / block
    release.set()
    a.join(10)
    b.join(10)
    assert errors == []
    got = _reservations(SF, scope)
    if scope_lock:
        assert got == {"2026-11-02": (0, 0), "2026-11-03": (300, 200)}
    else:
        assert got["2026-11-02"] == (0, 200)  # drawn onto the released allowance


def test_reserve_computed_interleavings_with_a_sending_transition(SF):
    """Round 8: (a) a check call already `sending` (committed draw) counts 0 in
    the estimate; (b) a call still `claimed` in our read whose sender is
    waiting on the scope lock is counted as undrawn, and its draw lands inside
    the reservation after our commit without a second charge."""
    from sqlalchemy import select

    from app import rebuild_paid_call as pc
    from app import spend
    from app.models import KieSpendDay
    from app.rebuild_models import RebuildChunkAttempt, RebuildLlmCall
    from app.state_machine import SegmentStatus

    pid, rid, rsid, _sd = _chunk_run(SF)
    s = SF()
    from app.models import Run, RunSegment

    s.get(RunSegment, rsid).status = SegmentStatus.completed
    att = RebuildChunkAttempt(run_id=rid, run_segment_id=rsid, attempt_no=1, model="kling-4-0",
                              resolution="720p", request_sec=10, chunk_frames=300,
                              status="succeeded", selected=True)
    s.add(att)
    s.flush()
    call = RebuildLlmCall(project_id=pid, attempt_id=att.id, kind="character_check", slot=0,
                          model="gemini-3.1-pro", status="queued", estimate_credits=20.0)
    s.add(call)
    s.commit()
    token = pc.claim(s, RebuildLlmCall, call.id, gen=1)
    s.commit()
    run = s.get(Run, rid)
    scope = spend.run_scope(rid)
    s.close()

    # (b) the reserver holds the locks; the sender's CAS commits nothing yet.
    in_estimate, release = threading.Event(), threading.Event()
    seen, errors = [], []

    def reserver():
        db = SF()
        try:
            def est(sess):
                v = spend.estimate_images_only_run_credits(sess, run)
                seen.append(v)
                in_estimate.set()
                release.wait(10)
                return v

            spend.reserve_computed(db, scope, est, kind="retry", release_unused=True, now=DAY2)
            db.commit()
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))
            db.rollback()
        finally:
            db.close()

    def sender():
        db = SF()
        try:
            ok = pc.begin_send(db, RebuildLlmCall, call.id, gen=1, token=token, request={"x": 1},
                               sender=pc.SenderIdentity("h", "b", "j"), scope_key=scope,
                               now=DAY2.replace(tzinfo=None))
            assert ok
            db.commit()
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))
            db.rollback()
        finally:
            db.close()

    a = threading.Thread(target=reserver)
    a.start()
    assert in_estimate.wait(10)
    b = threading.Thread(target=sender)
    b.start()
    time.sleep(1.0)
    release.set()
    a.join(10)
    b.join(10)
    assert errors == []
    assert seen == [40.0]  # slot 0 still `claimed` in our read: both slots undrawn
    assert _reservations(SF, scope) == {"2026-11-03": (40, 20)}
    s = SF()
    day = s.execute(select(KieSpendDay).where(KieSpendDay.day == DAY2.date())).scalar_one()
    assert day.charged_mcredits >= 40000  # charged = max(reserved, consumed): no double charge
    # (a) now the call is `sending` with its draw committed: it counts 0, the
    # fallback slot still counts.
    assert spend.estimate_images_only_run_credits(s, s.get(Run, rid)) == 20.0
    s.close()


# ---------------------------------------------------------------------------
# Wave 2a: admissions under the project row lock, selection races
# ---------------------------------------------------------------------------


def _rebuild_project(SF):
    from app import rebuild_analysis
    from app.models import VideoProject
    from app.state_machine import ProjectStatus

    fd, src = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    s = SF()
    p = VideoProject(source_type="upload", source_ref="x", status=ProjectStatus.analyzing,
                     project_type="rebuild", source_local_path=src, width=720, height=1280,
                     duration_sec=20.0, fps=30.0)
    s.add(p)
    s.flush()
    rebuild_analysis.publish_analysis(s, p, [(0, 3), (3, 6), (6, 10), (10, 14), (14, 20)])
    p.status = ProjectStatus.ready
    s.commit()
    pid = p.id
    s.close()
    return pid


def test_concurrent_storyboard_admissions_admit_exactly_one(SF):
    from fastapi import HTTPException

    from app import rebuild_storyboard as sbmod
    from app.models import VideoProject

    pid = _rebuild_project(SF)
    n = 6
    barrier = threading.Barrier(n)
    ok, codes, other = [], [], []

    def admit(i):
        s = SF()
        try:
            p = s.get(VideoProject, pid)
            barrier.wait()
            sbmod.admit_storyboard(s, p)
            s.commit()
            ok.append(i)
        except HTTPException as exc:
            s.rollback()
            codes.append(exc.status_code)
        except Exception as exc:  # pragma: no cover
            s.rollback()
            other.append(repr(exc))
        finally:
            s.close()

    ts = [threading.Thread(target=admit, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert other == [] and len(ok) == 1 and codes == [409] * (n - 1)


def _reanalysis_admission(s, pid):
    """What api_v2.reanalyze_project does for a rebuild project, minus the
    enqueue: project lock, live-row guard, `analyzing` in one transaction."""
    from app.api_v2 import _rebuild_live_rows_guard
    from app.models import VideoProject
    from app.state_machine import ProjectStatus

    p = s.get(VideoProject, pid)
    s.refresh(p, with_for_update=True)
    _rebuild_live_rows_guard(s, p, "re-analyze the project")
    p.status = ProjectStatus.analyzing
    s.commit()


def test_storyboard_admission_vs_reanalysis_in_both_orders(SF):
    from fastapi import HTTPException

    from app import rebuild_storyboard as sbmod
    from app.models import VideoProject
    from app.state_machine import ProjectStatus

    # Admission first: the re-analysis sees the live storyboard and refuses.
    pid = _rebuild_project(SF)
    s = SF()
    sbmod.admit_storyboard(s, s.get(VideoProject, pid))
    s.commit()
    s.close()
    s = SF()
    with pytest.raises(HTTPException) as exc:
        _reanalysis_admission(s, pid)
    assert exc.value.status_code == 409
    s.rollback()
    s.close()

    # Re-analysis first, holding the project lock while the admission waits on
    # it: the admission then reads `analyzing` and refuses.
    pid = _rebuild_project(SF)
    locked, done = threading.Event(), threading.Event()
    result = {}

    def reanalyze():
        s = SF()
        try:
            p = s.get(VideoProject, pid)
            s.refresh(p, with_for_update=True)
            locked.set()
            time.sleep(0.5)
            p.status = ProjectStatus.analyzing
            s.commit()
        finally:
            s.close()

    def admit():
        locked.wait()
        s = SF()
        try:
            sbmod.admit_storyboard(s, s.get(VideoProject, pid))
            s.commit()
            result["code"] = 202
        except HTTPException as e:
            s.rollback()
            result["code"], result["detail"] = e.status_code, e.detail
        finally:
            s.close()
            done.set()

    ta, tb = threading.Thread(target=reanalyze), threading.Thread(target=admit)
    ta.start(), tb.start()
    ta.join(), tb.join()
    assert result["code"] == 409 and "not ready" in result["detail"]
    s = SF()
    assert sbmod.live_storyboard(s, pid) is None
    s.close()


def test_manual_select_racing_auto_select_leaves_one_selected(SF):
    from datetime import datetime as _dt

    from sqlalchemy import select

    from app import rebuild_keyframes as kfmod
    from app.models import VideoProject
    from app.rebuild_models import RebuildKeyframe, RebuildShot

    pid = _rebuild_project(SF)
    s = SF()
    shot = s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)).scalars().first()
    t0 = _dt(2026, 11, 3, 10, 0, 0)
    kfs = [RebuildKeyframe(project_id=pid, batch_id="b", shot_id=shot.id, status="succeeded",
                           image_path=f"/x/{i}.png", finished_at=t0, updated_at=t0,
                           created_at=t0.replace(minute=i))
           for i in range(8)]
    s.add_all(kfs)
    s.commit()
    ids, shot_id = [k.id for k in kfs], shot.id
    s.close()
    barrier = threading.Barrier(len(ids))
    errors = []

    def manual(kid):
        s = SF()
        try:
            barrier.wait()
            kfmod.select_keyframe(s, s.get(VideoProject, pid), kid)
            s.commit()
        except Exception as exc:  # pragma: no cover
            s.rollback()
            errors.append(repr(exc))
        finally:
            s.close()

    def auto(kid):
        s = SF()
        try:
            barrier.wait()
            s.execute(select(VideoProject).where(VideoProject.id == pid).with_for_update())
            kf = s.get(RebuildKeyframe, kid)
            kfmod._auto_select(s, kf, _dt.utcnow())
            s.commit()
        except Exception as exc:  # pragma: no cover
            s.rollback()
            errors.append(repr(exc))
        finally:
            s.close()

    ts = [threading.Thread(target=(manual if i % 2 else auto), args=(k,))
          for i, k in enumerate(ids)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == []
    s = SF()
    sel = s.execute(select(RebuildKeyframe.id).where(
        RebuildKeyframe.shot_id == shot_id, RebuildKeyframe.selected.is_(True))).all()
    assert len(sel) == 1
    s.close()
