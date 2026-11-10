"""
Tests for the daily kie.ai spend cap (app/spend.py, AICP-155).

Covers the ledger itself on its own SQLite file: under cap / over cap,
idempotent draws and settlement, overdraft, release on retry, UTC day rollover,
and concurrent reservers (SQLite serializes them through the write lock taken by
the day-row UPDATE; the PostgreSQL row-lock proof lives in
test_spend_pg_concurrency.py). End-to-end API/automation/pipeline paths are in
test_api_v2.py, test_automation.py and test_pipeline_v2.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app import spend
from app.config import settings
from app.db import Base
from app.models import (
    KieSpendDay,
    KieSpendDraw,
    KieSpendReservation,
    Run,
    RunSegment,
    SegmentDef,
    VideoProject,
)
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus

pytestmark = pytest.mark.spend_cap

DAY1 = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
DAY2 = datetime(2026, 10, 17, 0, 0, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def engine():
    path = tempfile.mkstemp(suffix=".db")[1]
    eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def Session(engine):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    yield factory
    s = factory()
    for model in (KieSpendDraw, KieSpendReservation, KieSpendDay, RunSegment, Run, SegmentDef, VideoProject):
        s.query(model).delete()
    s.commit()
    s.close()


@pytest.fixture()
def sessions(Session):
    @contextmanager
    def _get_session():
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return _get_session


@pytest.fixture()
def cap(monkeypatch):
    """Set the cap in USD; 1 credit = $0.005, so $1 = 200 credits."""
    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)

    def _set(usd):
        monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", usd)

    _set(250.0)
    return _set


def _day_total(Session, when=DAY1) -> float:
    s = Session()
    try:
        return spend.today_summary(s, when)["charged_credits"]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Config / estimates
# ---------------------------------------------------------------------------


def test_default_cap_is_250_usd_and_zero_disables(monkeypatch):
    import subprocess

    env = {k: v for k, v in os.environ.items()
           if k not in ("KIE_DAILY_BUDGET_USD", "KIE_USD_PER_CREDIT")}
    out = subprocess.run(
        [sys.executable, "-c",
         "from app.config import Settings as S; "
         "print(S.KIE_DAILY_BUDGET_USD, S.KIE_USD_PER_CREDIT)"],
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env,
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert out == ["250.0", "0.005"]

    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0)
    assert spend.cap_usd() is None and spend.cap_credits() is None
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 250.0)
    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    assert spend.cap_credits() == pytest.approx(50_000)


def test_video_estimate_uses_model_rate_clamped_to_clip_bounds():
    # 2.5 at 720p: 80 credits per clip-second, ceil'd.
    assert spend.video_credits("seedance-2-5", "720p", 10.2) == 11 * 80
    # Under the model's minimum clip -> billed as the minimum (4s on 2.5).
    assert spend.video_credits("seedance-2-5", "720p", 1.0) == 4 * 80
    # Over the 2.0 ceiling (15s) -> capped.
    assert spend.video_credits("seedance", "720p", 40.0) == 15 * 52
    # A resolution the model lacks is coerced like the pipeline does.
    assert spend.video_credits("seedance-fast", "1080p", 5.0) == 5 * 45


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_under_cap_reserve_is_admitted(Session, cap):
    cap(10.0)  # 2000 credits
    s = Session()
    delta = spend.reserve(s, "run:a", 1500, kind="new run", now=DAY1)
    s.commit()
    s.close()
    assert delta == pytest.approx(1500)
    assert _day_total(Session) == pytest.approx(1500)


def test_over_cap_reserve_is_refused_and_leaves_no_trace(Session, cap):
    cap(10.0)  # 2000 credits
    s = Session()
    spend.reserve(s, "run:a", 1500, kind="new run", now=DAY1)
    s.commit()
    with pytest.raises(spend.BudgetExceeded) as excinfo:
        spend.reserve(s, "run:b", 600, kind="copy run", now=DAY1)
    s.rollback()
    s.close()
    msg = excinfo.value.message
    assert "Daily kie.ai budget reached" in msg
    assert "$7.50 of $10.00" in msg and "$3.00" in msg and "copy run" in msg
    assert _day_total(Session) == pytest.approx(1500)
    s = Session()
    assert s.execute(
        select(KieSpendReservation).where(KieSpendReservation.scope_key == "run:b")
    ).first() is None
    s.close()


def test_exactly_at_cap_is_admitted_one_more_credit_is_not(Session, cap):
    cap(10.0)
    s = Session()
    spend.reserve(s, "run:a", 2000, kind="new run", now=DAY1)
    s.commit()
    with pytest.raises(spend.BudgetExceeded):
        spend.reserve(s, "run:b", 1, kind="new run", now=DAY1)
    s.rollback()
    # A zero-cost action (delivery-only retry) is never refused.
    spend.reserve(s, "run:a", 0, kind="retry run", now=DAY1)
    s.commit()
    s.close()


def test_disabled_cap_admits_everything_but_still_counts(Session, cap):
    cap(0.0)
    s = Session()
    spend.reserve(s, "run:a", 10_000_000, kind="new run", now=DAY1)
    s.commit()
    s.close()
    assert _day_total(Session) == pytest.approx(10_000_000)


def test_release_unused_rebaselines_instead_of_double_counting(Session, cap, sessions):
    s = Session()
    spend.reserve(s, "run:a", 1000, kind="new run", now=DAY1)
    s.commit()
    s.close()
    # 4 of 10 segments were submitted, then the run failed.
    for i in range(4):
        spend.record_draw("run:a", 100, kind="video", call_key=f"t{i}", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(1000)
    s = Session()
    # Retry re-estimates the 6 unsubmitted segments: total stays 1000, not 1600.
    spend.reserve(s, "run:a", 600, kind="retry run", release_unused=True, now=DAY1)
    s.commit()
    # Without release (an additive action) it would stack.
    spend.reserve(s, "run:a", 6, kind="regenerate one GPT Image", now=DAY1)
    s.commit()
    s.close()
    assert _day_total(Session) == pytest.approx(1006)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_draw_within_reservation_does_not_add_and_is_idempotent(Session, cap, sessions):
    s = Session()
    spend.reserve(s, "run:a", 800, kind="new run", now=DAY1)
    s.commit()
    s.close()
    spend.record_draw("run:a", 800, kind="video", call_key="task-1", sessions=sessions, now=DAY1)
    spend.record_draw("run:a", 800, kind="video", call_key="task-1", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(800)
    s = Session()
    assert len(s.execute(select(KieSpendDraw)).all()) == 1
    s.close()


def test_overdraft_draw_is_never_refused_and_is_counted(Session, cap, sessions):
    cap(5.0)  # 1000 credits
    s = Session()
    spend.reserve(s, "run:a", 1000, kind="new run", now=DAY1)
    s.commit()
    s.close()
    # The in-flight run resubmits a failed segment: over the cap, still recorded.
    spend.record_draw("run:a", 1000, kind="video", call_key="t1", sessions=sessions, now=DAY1)
    spend.record_draw("run:a", 400, kind="video", call_key="t1-retry", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(1400)
    # ...and new admissions now see the overdraft.
    s = Session()
    with pytest.raises(spend.BudgetExceeded):
        spend.reserve(s, "run:b", 1, kind="new run", now=DAY1)
    s.rollback()
    s.close()


def test_unreserved_draw_lands_on_today_as_overdraft(Session, cap, sessions):
    spend.record_draw(None, 20, kind="transcribe", sessions=sessions, now=DAY1)
    spend.record_draw("run:pre-deploy", 700, kind="video", call_key="old", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(720)


def test_settle_replaces_estimate_with_actual_both_ways(Session, cap, sessions):
    s = Session()
    spend.reserve(s, "run:a", 800, kind="new run", now=DAY1)
    s.commit()
    s.close()
    spend.record_draw("run:a", 800, kind="video", call_key="t1", sessions=sessions, now=DAY1)
    spend.settle_draw("t1", 1100, scope_key="run:a", kind="video", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(1100)
    # Settling twice is a no-op; a lower actual falls back to the reservation.
    spend.settle_draw("t1", 1100, scope_key="run:a", kind="video", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(1100)
    spend.settle_draw("t1", 650, scope_key="run:a", kind="video", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(800)
    # Unknown cost (None) changes nothing.
    spend.settle_draw("t1", None, scope_key="run:a", kind="video", sessions=sessions, now=DAY1)
    assert _day_total(Session) == pytest.approx(800)


def test_draw_never_raises_on_accounting_failure(caplog):
    @contextmanager
    def broken():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    spend.record_draw("run:a", 10, kind="video", call_key="x", sessions=broken)
    spend.settle_draw("x", 10, scope_key="run:a", kind="video", sessions=broken)
    assert "ACCOUNTING FAILURE" in caplog.text


# ---------------------------------------------------------------------------
# Day rollover (UTC)
# ---------------------------------------------------------------------------


def test_day_rollover_resets_the_budget_and_keeps_late_draws_on_admission_day(Session, cap, sessions):
    cap(5.0)  # 1000 credits
    s = Session()
    spend.reserve(s, "run:a", 1000, kind="new run", now=DAY1)
    s.commit()
    with pytest.raises(spend.BudgetExceeded):
        spend.reserve(s, "run:b", 100, kind="new run", now=DAY1)
    s.rollback()
    # 00:00:01 UTC next day: a fresh budget.
    spend.reserve(s, "run:b", 900, kind="new run", now=DAY2)
    s.commit()
    s.close()
    # run:a (admitted yesterday) keeps billing after midnight: its draws stay
    # on yesterday's books and do not eat today's budget.
    spend.record_draw("run:a", 1000, kind="video", call_key="a1", sessions=sessions, now=DAY2)
    spend.record_draw("run:a", 300, kind="video", call_key="a2", sessions=sessions, now=DAY2)
    assert _day_total(Session, DAY1) == pytest.approx(1300)
    assert _day_total(Session, DAY2) == pytest.approx(900)


def test_utc_boundary_not_local_time():
    # 23:30 at UTC-5 is 04:30 UTC the next day.
    local = datetime(2026, 10, 16, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert spend.utc_day(local).isoformat() == "2026-10-17"


def test_next_day_retry_releases_yesterdays_unused_part(Session, cap, sessions):
    s = Session()
    spend.reserve(s, "run:a", 1000, kind="new run", now=DAY1)
    s.commit()
    s.close()
    spend.record_draw("run:a", 400, kind="video", call_key="a1", sessions=sessions, now=DAY1)
    s = Session()
    spend.reserve(s, "run:a", 600, kind="retry run", release_unused=True, now=DAY2)
    s.commit()
    s.close()
    assert _day_total(Session, DAY1) == pytest.approx(400)
    assert _day_total(Session, DAY2) == pytest.approx(600)
    # The retry's draw goes to today's reservation, not yesterday's.
    spend.record_draw("run:a", 600, kind="video", call_key="a2", sessions=sessions, now=DAY2)
    assert _day_total(Session, DAY1) == pytest.approx(400)
    assert _day_total(Session, DAY2) == pytest.approx(600)


def test_video_draw_prefers_the_reservation_that_reserved_it(Session, cap, sessions):
    s = Session()
    spend.reserve(s, "run:a", 1000, kind="new run", now=DAY1)  # video, at launch
    s.commit()
    spend.reserve(s, "run:a", 6, kind="regenerate one GPT Image", now=DAY2)
    s.commit()
    s.close()
    spend.record_draw("run:a", 6, kind="image", call_key="img", sessions=sessions, now=DAY2)
    spend.record_draw("run:a", 1000, kind="video", call_key="vid", sessions=sessions, now=DAY2)
    assert _day_total(Session, DAY1) == pytest.approx(1000)
    assert _day_total(Session, DAY2) == pytest.approx(6)


# ---------------------------------------------------------------------------
# Concurrency (SQLite: serialized by the write lock the day UPDATE takes)
# ---------------------------------------------------------------------------


def test_concurrent_reservers_never_overshoot(Session, cap):
    cap(5.0)  # 1000 credits; 12 racers x 100 -> exactly 10 fit
    barrier = threading.Barrier(12)
    admitted, refused, errors = [], [], []

    def racer(i):
        s = Session()
        try:
            barrier.wait()
            spend.reserve(s, f"run:{i}", 100, kind="new run", now=DAY1)
            s.commit()
            admitted.append(i)
        except spend.BudgetExceeded:
            s.rollback()
            refused.append(i)
        except Exception as exc:  # pragma: no cover - surfaced below
            s.rollback()
            errors.append(exc)
        finally:
            s.close()

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(admitted) == 10 and len(refused) == 2
    assert _day_total(Session) == pytest.approx(1000)


# ---------------------------------------------------------------------------
# Whole-run estimates
# ---------------------------------------------------------------------------


def _project_with_segments(s, durations, actions=None):
    p = VideoProject(source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready)
    s.add(p)
    s.flush()
    t = 0.0
    defs = []
    for i, d in enumerate(durations):
        sd = SegmentDef(
            project_id=p.id, index=i, start_sec=t, end_sec=t + d, has_face=True,
            action=(actions[i] if actions else "swap"),
        )
        s.add(sd)
        defs.append(sd)
        t += d
    s.commit()
    return p, defs


def test_run_estimate_counts_only_unfinished_active_swaps(Session):
    s = Session()
    p, defs = _project_with_segments(s, [5.0, 3.0, 8.0, 10.0], ["swap", "keep", "swap", "swap"])
    run = Run(project_id=p.id, prompt="x", model="seedance-2-5", resolution="720p",
              status=RunStatus.created, reference_image_urls=[])
    s.add(run)
    s.commit()
    full = spend.estimate_run_credits(s, run)
    assert full == (5 + 8 + 10) * 80
    done_file = tempfile.mkstemp(suffix=".mp4")[1]
    s.add(RunSegment(run_id=run.id, segment_def_id=defs[0].id, index=0,
                     status=SegmentStatus.completed, local_result_path=done_file))
    s.add(RunSegment(run_id=run.id, segment_def_id=defs[2].id, index=2, status=SegmentStatus.generating,
                     seedance_task_id="live"))
    s.add(RunSegment(run_id=run.id, segment_def_id=defs[3].id, index=3, status=SegmentStatus.failed))
    s.commit()
    assert spend.estimate_run_credits(s, run) == 10 * 80
    # A completed segment whose result file vanished will be regenerated.
    os.remove(done_file)
    assert spend.estimate_run_credits(s, run) == (5 + 10) * 80
    assert spend.estimate_run_credits(s, run, fresh=True) == full
    # A custom stitch plan that leaves segment 3 out never submits it.
    run.stitch_plan = [{"kind": "segment", "segment_def_id": defs[0].id}]
    assert spend.estimate_run_credits(s, run, fresh=True) == 5 * 80
    s.close()


def test_run_estimate_includes_preprocess_images(Session):
    s = Session()
    p, _ = _project_with_segments(s, [4.0])
    run = Run(project_id=p.id, prompt="x", model="seedance", resolution="720p",
              status=RunStatus.preprocessing, reference_image_urls=["c.jpg", "t1.jpg", "t2.jpg", "t3.jpg"],
              preprocess_enabled=True, preprocess_mode="per_image",
              preprocess_image_prompts=["", "p1", "", "p3"],
              preprocessed_reference_urls=[None, None, None, "done.jpg"])
    s.add(run)
    s.commit()
    # 1 image (t1; t2 pass-through, t3 already done) + 4s at 52/s
    assert spend.estimate_run_credits(s, run) == 6 + 4 * 52
    assert spend.preprocess_image_count(run, fresh=True) == 2
    s.close()


def test_submitted_row_without_a_task_id_still_counts_as_unpaid(Session):
    """createTask failed after upload: the row says `submitted` but kie.ai never
    took a task, so a retry WILL bill it — the estimate must include it."""
    s = Session()
    p, defs = _project_with_segments(s, [5.0])
    run = Run(project_id=p.id, prompt="x", model="seedance", resolution="720p",
              status=RunStatus.failed, reference_image_urls=[])
    s.add(run)
    s.commit()
    rs = RunSegment(run_id=run.id, segment_def_id=defs[0].id, index=0,
                    status=SegmentStatus.submitted, seedance_task_id=None)
    s.add(rs)
    s.commit()
    assert spend.estimate_run_credits(s, run) == 5 * 52
    rs.seedance_task_id = "real-task"
    s.commit()
    assert spend.estimate_run_credits(s, run) == 0
    s.close()
