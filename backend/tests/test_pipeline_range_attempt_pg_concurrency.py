"""
True-concurrency proof for the independent audit's blocking finding #2:
"one-active-attempt and promotion are not concurrency-safe".

SQLite (used by the rest of the test suite) cannot prove this — it has no
real row-level locking, so a `SELECT ... FOR UPDATE` is a silent no-op there
and any race test against it can only prove the LOGIC is right when steps
happen to be sequenced favourably, never that two genuinely simultaneous
transactions are actually serialized. This module runs the same code paths
against a real PostgreSQL 15 instance — this feature's production database —
using real threads and a barrier to force true simultaneity, not a sequenced
approximation.

Opt-in and skipped by default: set RESKIN_TEST_PG_URL to a disposable
Postgres 15 database before running (e.g.
``postgresql://reskin:reskin@localhost:15432/reskin``, matching the
disposable container recipe used to verify migration 73df0ff3f344). Never
point this at the production database — it creates and drops real tables.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    not _PG_URL,
    reason=(
        "RESKIN_TEST_PG_URL not set — this module needs a real PostgreSQL 15 "
        "instance to prove row-level locking; SQLite cannot (see module "
        "docstring). Skipped by default so the normal suite needs no Docker."
    ),
)

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_range_attempt_pg_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if _PG_URL:
    from sqlalchemy import create_engine
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    from app import media as media_mod
    from app.db import Base
    from app.models import Run, RunSegment, RunSegmentRangeAttempt, SegmentDef, VideoProject
    from app.state_machine import ProjectStatus, RangeAttemptStatus, RunStatus, SegmentStatus
    from app.storage import keep_clip_path, range_attempt_dir
    import app.pipeline_range_attempt as pra_mod
    from app.pipeline_range_attempt import (
        create_range_attempt,
        poll_range_attempt_once,
        submit_range_attempt,
    )
    from tests.test_pipeline_v2 import FakeGDriveClient, FakeKieClient

    class _PausingKieClient(FakeKieClient):
        """FakeKieClient whose upload_file/download_result can be held open
        on a threading.Event, so a test can prove a losing racer has
        ALREADY resolved (bailed via the claim check) before the claim
        winner is allowed to proceed into its own paid/side-effecting call
        — genuine temporal overlap with the moment of contention, not just
        a fortunate ordering. Defined inside this guard (not at module
        scope) because it subclasses FakeKieClient, which is itself only
        importable when _PG_URL is set — the plain test functions below can
        reference FakeKieClient lazily (inside a function body, evaluated
        only if the module's skipif lets the test run), but a class
        statement resolves its base class immediately at import time."""

        def __init__(self, *a, pause_event: threading.Event, release_event: threading.Event, **kw):
            super().__init__(*a, **kw)
            self._pause_event = pause_event
            self._release_event = release_event
            self.download_calls_detailed: list[dict] = []

        def upload_file(self, *a, **kw):
            self._pause_event.set()
            assert self._release_event.wait(timeout=15), "test never released the paused winner"
            return super().upload_file(*a, **kw)

        def download_result(self, url: str, dst_path: str) -> None:
            self.download_calls_detailed.append({"url": url, "dst_path": dst_path})
            self._pause_event.set()
            assert self._release_event.wait(timeout=15), "test never released the paused winner"
            return super().download_result(url, dst_path)

_VIDEO_DURATION = 25.0


@pytest.fixture(scope="module")
def pg_engine():
    engine = create_engine(_PG_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(autouse=True)
def patch_get_session(pg_engine, monkeypatch):
    """Point pipeline_range_attempt.get_session (used internally by
    submit_range_attempt/poll_range_attempt_once) at pg_engine — otherwise
    those calls would open sessions against whatever DATABASE_URL the app
    was imported with, not this module's Postgres engine, and every
    attempt/run lookup inside them would 404 against the wrong database."""
    from contextlib import contextmanager

    from sqlalchemy.orm import sessionmaker as _sessionmaker

    _TestSession = _sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)

    @contextmanager
    def _test_get_session():
        sess = _TestSession()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    monkeypatch.setattr(pra_mod, "get_session", _test_get_session)


@pytest.fixture(scope="module")
def synthetic_video():
    path = os.path.join(_TMP_DIR, "source.mp4")
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={int(_VIDEO_DURATION)}:r=25",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={int(_VIDEO_DURATION)}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            "-c:a", "aac", path,
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return path


def _make_pipeline_run(session, synthetic_video):
    project = VideoProject(
        source_type="upload", source_ref="src.mp4", source_local_path=synthetic_video,
        duration_sec=_VIDEO_DURATION, width=320, height=240, fps=25.0,
        aspect_ratio="4:3", status=ProjectStatus.ready,
    )
    session.add(project)
    session.flush()

    sd_keep0 = SegmentDef(project_id=project.id, index=0, action="keep", start_sec=0.0, end_sec=9.0)
    sd_swap1 = SegmentDef(project_id=project.id, index=1, action="swap", start_sec=9.0, end_sec=15.0)
    sd_keep2 = SegmentDef(project_id=project.id, index=2, action="keep", start_sec=15.0, end_sec=25.0)
    session.add_all([sd_keep0, sd_swap1, sd_keep2])
    session.flush()

    run = Run(
        project_id=project.id, prompt="a person", model="seedance", resolution="480p",
        audio_mode="original", status=RunStatus.done,
        segment_snapshot=[
            {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 9.0},
            {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 9.0, "end_sec": 15.0},
            {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 15.0, "end_sec": 25.0},
        ],
    )
    session.add(run)
    session.flush()

    rs = RunSegment(
        run_id=run.id, segment_def_id=sd_swap1.id, index=1,
        status=SegmentStatus.completed,
        local_result_path=os.path.join(_TMP_DIR, f"orig_result_{run.id}.mp4"),
        seedance_task_id="orig-task-1", cost_credits=10.0,
    )
    session.add(rs)
    session.flush()
    shutil.copy2(synthetic_video, rs.local_result_path)
    run.result_local_path = os.path.join(_TMP_DIR, f"orig_final_{run.id}.mp4")
    shutil.copy2(synthetic_video, run.result_local_path)
    session.commit()
    return project, run, sd_keep0, sd_swap1, sd_keep2, rs


def test_two_concurrent_create_requests_only_one_wins_and_only_one_kie_call_happens(
    pg_engine, synthetic_video,
):
    """Mandatory concurrency tests #1-5: race two create requests for the
    same run, assert exactly one succeeds, exactly one active row exists,
    and — driving the winner with a fake Kie client — exactly one external
    create-task call happens (the loser never got far enough to make one)."""
    Session = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    setup = Session()
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(setup, synthetic_video)
    run_id, rs_id, project_id = run.id, rs.id, project.id
    setup.close()

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def _attempt(name: str, start_sec: float, end_sec: float) -> None:
        session = Session()
        try:
            r = session.get(Run, run_id)
            seg = session.get(RunSegment, rs_id)
            proj = session.get(VideoProject, project_id)
            barrier.wait()
            try:
                attempt = create_range_attempt(
                    run=r, rs=seg, project=proj,
                    start_sec=start_sec, end_sec=end_sec,
                    prompt_override=None, reference_image_urls_override=None,
                    attempt_id=str(uuid.uuid4()), db=session,
                )
                session.commit()
                results[name] = ("ok", attempt.id)
            except IntegrityError as exc:
                session.rollback()
                results[name] = ("integrity_error", str(exc))
        finally:
            session.close()

    t1 = threading.Thread(target=_attempt, args=("A", 8.0, 15.0))
    t2 = threading.Thread(target=_attempt, args=("B", 9.0, 16.0))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    outcomes = [results["A"][0], results["B"][0]]
    assert sorted(outcomes) == ["integrity_error", "ok"], results

    verify = Session()
    active_rows = (
        verify.query(RunSegmentRangeAttempt)
        .filter(RunSegmentRangeAttempt.run_id == run_id)
        .all()
    )
    assert len(active_rows) == 1
    winner_attempt_id = active_rows[0].id
    verify.close()

    # Drive the winner through submit — exactly one external create_task call.
    kie = FakeKieClient(synthetic_video)
    task_id = submit_range_attempt(winner_attempt_id, kie=kie)
    assert task_id is not None
    assert len(kie.create_task_records) == 1


def test_two_concurrent_promotions_of_the_same_attempt_serialize_on_the_row_lock(
    pg_engine, synthetic_video,
):
    """Mandatory concurrency tests #6-8: race two promotion transactions
    built from the same base snapshot using two separate sessions/
    connections and a barrier. Exactly one must commit as promoted; the
    other must fail without modifying the winner. Re-read all final state
    from a THIRD fresh session — never against a cached ORM object."""
    Session = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    setup = Session()
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(setup, synthetic_video)
    run_id, rs_id, project_id = run.id, rs.id, project.id
    sd_swap1_id = sd_swap1.id

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=7.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=setup,
    )
    setup.commit()
    attempt_id = attempt.id
    setup.close()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt_id, kie=kie)

    from app.pipeline_range_attempt import poll_range_attempt_once

    # Drive through generate/stitch/deliver but suppress the real _promote
    # so both threads below find the SAME attempt sitting ready to promote —
    # exactly the state a duplicate poll enqueue would find it in.
    real_promote = pra_mod._promote
    pra_mod._promote = lambda *a, **k: None
    try:
        assert poll_range_attempt_once(attempt_id, kie=kie, gdrive=FakeGDriveClient()) == "done"
    finally:
        pra_mod._promote = real_promote

    verify = Session()
    assert verify.get(RunSegmentRangeAttempt, attempt_id).promoted_at is None
    verify.close()

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def _promote_race(name: str) -> None:
        session = Session()
        try:
            r = session.get(Run, run_id)
            seg = session.get(RunSegment, rs_id)
            a = session.get(RunSegmentRangeAttempt, attempt_id)
            barrier.wait()
            try:
                real_promote(session, r, seg, a)
                session.commit()
                results[name] = ("ok", None)
            except Exception as exc:
                session.rollback()
                results[name] = ("failed", str(exc))
        finally:
            session.close()

    t1 = threading.Thread(target=_promote_race, args=("A",))
    t2 = threading.Thread(target=_promote_race, args=("B",))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    outcomes = [results["A"][0], results["B"][0]]
    assert sorted(outcomes) == ["failed", "ok"], results
    loser = "A" if results["A"][0] == "failed" else "B"
    assert "active timeline changed" in results[loser][1]

    # Third, fresh session — never the racer sessions' own (possibly stale)
    # in-memory objects.
    verify = Session()
    final_run = verify.get(Run, run_id)
    final_rs = verify.get(RunSegment, rs_id)
    final_attempt = verify.get(RunSegmentRangeAttempt, attempt_id)
    verify.close()
    assert final_attempt.promoted_at is not None
    by_id = {e["id"]: e for e in final_run.segment_snapshot}
    assert by_id[sd_swap1_id]["start_sec"] == pytest.approx(7.0)
    assert final_rs.local_result_path == final_attempt.local_result_path


# ---------------------------------------------------------------------------
# Full-worker duplicate-delivery races (hardening follow-up): two COMPLETE
# submit_range_attempt / poll_range_attempt_once calls for the SAME attempt,
# genuinely overlapping via real threads + a barrier + a controlled pause,
# proving the claim closes the race BEFORE any shared-file/external side
# effect — not just that the DB commit ordering comes out right afterwards.
# ---------------------------------------------------------------------------


def test_full_duplicate_submission_only_one_worker_cuts_uploads_and_creates_a_task(
    pg_engine, synthetic_video, monkeypatch,
):
    """Mandatory test A: two COMPLETE submit_range_attempt() workers race the
    SAME queued attempt from two independent sessions/threads. Exactly one
    must cut+upload+create the kie.ai task; the other must exit benignly
    before touching any file or external client."""
    Session = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    setup = Session()
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(setup, synthetic_video)
    run_id, rs_id, project_id = run.id, rs.id, project.id

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=setup,
    )
    setup.commit()
    attempt_id = attempt.id
    setup.close()

    cut_calls: list = []
    real_cut_clip = pra_mod.media_mod.cut_clip

    def _counting_cut_clip(*a, **kw):
        cut_calls.append((a, kw))
        return real_cut_clip(*a, **kw)

    monkeypatch.setattr(pra_mod.media_mod, "cut_clip", _counting_cut_clip)

    pause_event = threading.Event()
    release_event = threading.Event()
    kie = _PausingKieClient(synthetic_video, pause_event=pause_event, release_event=release_event)

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def _submit(name: str) -> None:
        barrier.wait()
        results[name] = submit_range_attempt(attempt_id, kie=kie)

    t1 = threading.Thread(target=_submit, args=("A",))
    t2 = threading.Thread(target=_submit, args=("B",))
    t1.start()
    t2.start()
    # The eventual claim winner blocks inside upload_file; give it (and, more
    # importantly, the loser — which never reaches upload_file at all) time
    # to fully resolve before releasing the winner to finish its paid call.
    assert pause_event.wait(timeout=20), "neither worker reached the paid call in time"
    time.sleep(0.5)
    release_event.set()
    t1.join(timeout=30)
    t2.join(timeout=30)

    task_ids = [results["A"], results["B"]]
    winners = [t for t in task_ids if t]
    losers = [t for t in task_ids if not t]
    assert len(winners) == 1, results
    assert len(losers) == 1, results
    winner_task_id = winners[0]

    # Exactly one cut, one upload, one createTask call — the loser touched none.
    assert len(cut_calls) == 1
    assert len(kie.upload_calls) == 1
    assert len(kie.create_task_records) == 1

    # Re-read from a THIRD fresh session — never a racer's own in-memory copy.
    verify = Session()
    final = verify.get(RunSegmentRangeAttempt, attempt_id)
    assert final.status == RangeAttemptStatus.generating
    assert final.seedance_task_id == winner_task_id
    assert final.error_message is None
    assert final.claim_owner is None  # released once the submit phase concluded
    # The uploaded clip is complete and matches the requested bounds exactly
    # (not truncated/corrupted by a concurrent writer to the same path).
    info = media_mod.probe(final.local_clip_path)
    assert info.duration_sec == pytest.approx(7.0, abs=0.3)
    verify.close()


def test_full_duplicate_poll_only_one_worker_finalizes_and_promotes(
    pg_engine, synthetic_video, monkeypatch,
):
    """Mandatory test B: two COMPLETE poll_range_attempt_once() workers race
    the SAME generating attempt (fake kie.ai task already reporting
    success) from two independent sessions/threads. Exactly one must
    download+recut+stitch+deliver+promote; the other must report `pending`
    without touching the winner's files or the Run."""
    Session = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    setup = Session()
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(setup, synthetic_video)
    run_id, rs_id, project_id = run.id, rs.id, project.id
    sd_swap1_id = sd_swap1.id

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=setup,
    )
    setup.commit()
    attempt_id = attempt.id
    setup.close()

    plain_kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt_id, kie=plain_kie)

    pause_event = threading.Event()
    release_event = threading.Event()
    kie = _PausingKieClient(synthetic_video, pause_event=pause_event, release_event=release_event)
    # Reuse the SAME fake task id/records the plain submit already created —
    # the pausing client only needs to answer get_task/download_result for
    # polling, so seed it with the already-created task bookkeeping.
    kie.create_task_calls = list(plain_kie.create_task_calls)
    kie.create_task_records = list(plain_kie.create_task_records)

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}
    gdrive_clients: dict[str, FakeGDriveClient] = {}

    def _poll(name: str) -> None:
        gd = FakeGDriveClient()
        gdrive_clients[name] = gd
        barrier.wait()
        results[name] = poll_range_attempt_once(attempt_id, kie=kie, gdrive=gd)

    t1 = threading.Thread(target=_poll, args=("A",))
    t2 = threading.Thread(target=_poll, args=("B",))
    t1.start()
    t2.start()
    assert pause_event.wait(timeout=20), "neither worker reached download_result in time"
    time.sleep(0.5)
    release_event.set()
    t1.join(timeout=30)
    t2.join(timeout=30)

    outcomes = sorted([results["A"], results["B"]])
    assert outcomes == ["done", "pending"], results
    winner_name = "A" if results["A"] == "done" else "B"
    loser_name = "B" if winner_name == "A" else "A"

    # Exactly one download call, into a claim-scoped (never the bare shared)
    # path — proves no two callers could ever race the same `.part` file.
    assert len(kie.download_calls_detailed) == 1
    downloaded_path = kie.download_calls_detailed[0]["dst_path"]
    assert "claims" in downloaded_path.split(os.sep), downloaded_path

    winner_gdrive = gdrive_clients[winner_name]
    loser_gdrive = gdrive_clients[loser_name]
    assert len(winner_gdrive.upload_calls) == 1
    assert len(loser_gdrive.upload_calls) == 0

    # Re-read from a THIRD fresh session.
    verify = Session()
    final_attempt = verify.get(RunSegmentRangeAttempt, attempt_id)
    final_run = verify.get(Run, run_id)
    final_rs = verify.get(RunSegment, rs_id)
    assert final_attempt.status == RangeAttemptStatus.done
    assert final_attempt.promoted_at is not None
    assert final_attempt.error_message is None
    by_id = {e["id"]: e for e in final_run.segment_snapshot}
    assert by_id[sd_swap1_id]["start_sec"] == pytest.approx(8.0)
    assert final_rs.local_result_path == final_attempt.local_result_path
    probe_target = final_attempt.final_local_path or final_attempt.local_result_path
    info = media_mod.probe(probe_target)
    assert info.duration_sec > 0
    verify.close()


# ---------------------------------------------------------------------------
# Follow-up hardening (2026-08-30, independent audit) — Finding 1: a live
# finalizer must retain ownership for its COMPLETE phase, never lose its
# claim to a duplicate poll chain while it is genuinely still alive. Proved
# here with a DELIBERATELY shortened lease (simulating the exact
# misconfiguration the audit found — the original 900s lease vs the 3600s
# job timeout) to exercise the explicit _assert_still_owner ownership
# re-check as a SECOND, independent line of defense: even when a duplicate
# DOES manage to acquire a claim the lease let expire out from under a still
# -alive worker, the critical safety properties (exactly one Drive upload,
# exactly one promotion, the loser never fails or corrupts the winner) must
# still hold.
# ---------------------------------------------------------------------------


def test_full_duplicate_poll_survives_a_shorter_than_finalize_lease(
    pg_engine, synthetic_video, monkeypatch,
):
    """Mandatory regression for Finding 1: run a full finalizer whose first
    download deliberately outlasts an artificially shortened claim lease;
    start a duplicate poll chain while the original worker is still alive.
    Even though the duplicate manages to acquire the (test-shortened)
    expired claim, exactly one worker's Drive upload and promotion actually
    take effect, verified from a third independent session, and the loser
    never fails the attempt."""
    Session = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    setup = Session()
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(setup, synthetic_video)
    run_id, rs_id = run.id, rs.id
    sd_swap1_id = sd_swap1.id

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=setup,
    )
    setup.commit()
    attempt_id = attempt.id
    setup.close()

    plain_kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt_id, kie=plain_kie)

    # Deliberately shorten the lease well below what this test's own
    # finalize round will take — simulating the exact misconfiguration the
    # audit found (a lease shorter than a real finalize round). Bypasses
    # the production floor (max(..., job_timeout + 60)) on purpose, since
    # this test exists specifically to prove the SECOND line of defense
    # (_assert_still_owner) holds even if the first (lease sizing) did not.
    monkeypatch.setattr(pra_mod, "_CLAIM_LEASE_SEC", 2)

    a_started_download = threading.Event()
    release_a = threading.Event()
    first_call_lock = threading.Lock()
    first_call_done = {"value": False}

    class _SlowFirstDownloadKie(FakeKieClient):
        def download_result(self, url: str, dst_path: str) -> None:
            with first_call_lock:
                is_first = not first_call_done["value"]
                first_call_done["value"] = True
            if is_first:
                a_started_download.set()
                assert release_a.wait(timeout=30), "test never released the paused worker"
            return super().download_result(url, dst_path)

    kie = _SlowFirstDownloadKie(synthetic_video)
    kie.create_task_calls = list(plain_kie.create_task_calls)
    kie.create_task_records = list(plain_kie.create_task_records)

    results: dict[str, object] = {}
    gdrive_clients: dict[str, FakeGDriveClient] = {}

    def _poll(name: str) -> None:
        gd = FakeGDriveClient()
        gdrive_clients[name] = gd
        results[name] = poll_range_attempt_once(attempt_id, kie=kie, gdrive=gd)

    t_a = threading.Thread(target=_poll, args=("A",))
    t_a.start()
    assert a_started_download.wait(timeout=20), "A never reached download_result"
    # Let the (deliberately short) lease genuinely expire — real wall-clock
    # sleep, since claim_expires_at is a real timestamp — while A is still
    # paused mid-download, i.e. still alive and still working.
    time.sleep(4)

    t_b = threading.Thread(target=_poll, args=("B",))
    t_b.start()
    t_b.join(timeout=30)
    release_a.set()
    t_a.join(timeout=30)

    # The loser must never report (or cause) a false failure — "pending" or
    # a genuine "done" (if it happened to observe the row after the winner
    # already finished) are the only acceptable outcomes.
    assert results["B"] == "done", results
    assert results["A"] != "failed", results

    # The critical safety property this test exists to prove: regardless of
    # how the race unfolded, exactly ONE Drive upload and ONE promotion
    # actually took effect.
    total_uploads = sum(len(gd.upload_calls) for gd in gdrive_clients.values())
    assert total_uploads == 1, gdrive_clients

    # Re-read from a THIRD fresh session — never a racer's own in-memory copy.
    verify = Session()
    final_attempt = verify.get(RunSegmentRangeAttempt, attempt_id)
    final_run = verify.get(Run, run_id)
    final_rs = verify.get(RunSegment, rs_id)
    assert final_attempt.status == RangeAttemptStatus.done
    assert final_attempt.promoted_at is not None
    assert final_attempt.error_message is None
    by_id = {e["id"]: e for e in final_run.segment_snapshot}
    assert by_id[sd_swap1_id]["start_sec"] == pytest.approx(8.0)
    assert final_rs.local_result_path == final_attempt.local_result_path
    verify.close()


def test_full_duplicate_poll_never_double_writes_the_shared_keep_cache(
    pg_engine, synthetic_video,
):
    """Mandatory regression for Finding 2: a stale claim's own (claim-
    scoped, already-cut) keep-cache copy must never publish over a newer
    owner's — proven by directly racing two _resolve_keep_clip calls (one
    holding the row's CURRENT claim_owner, one holding a stale token that
    does not) against the SAME missing shared cache entry, with real
    threads and a barrier forcing genuine ffmpeg-level overlap."""
    Session = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    setup = Session()
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(setup, synthetic_video)
    run_id, project_id, rs_id = run.id, project.id, rs.id

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=setup,
    )
    setup.commit()
    attempt_id = attempt.id

    # Give the row a REAL, current claim — the "current owner" thread
    # presents this token; the "stale" thread presents a DIFFERENT one that
    # does not match the row at all.
    from datetime import datetime, timedelta, timezone

    current_token = "current-owner-token"
    attempt.claim_owner = current_token
    attempt.claim_expires_at = datetime.now(timezone.utc) + timedelta(seconds=300)
    setup.commit()

    # Extract everything needed BEFORE closing this session — accessing a
    # detached instance's attributes afterwards raises DetachedInstanceError.
    keep_entry = {
        "id": sd_keep2.id, "index": sd_keep2.index,
        "start_sec": sd_keep2.start_sec, "end_sec": sd_keep2.end_sec,
    }
    setup.close()

    cache_dst = keep_clip_path(
        run_id, project_id, keep_entry["index"], keep_entry["start_sec"], keep_entry["end_sec"],
    )
    assert not os.path.exists(cache_dst)

    attempt_dir = range_attempt_dir(attempt_id, rs_id, run_id, project_id)
    current_claim_dir = pra_mod._claim_dir(attempt_dir, current_token)
    stale_claim_dir = pra_mod._claim_dir(attempt_dir, "stale-dead-token")

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def _resolve(name: str, token: str, claim_dir: str) -> None:
        session = Session()
        try:
            barrier.wait()
            try:
                path = pra_mod._resolve_keep_clip(
                    keep_entry, synthetic_video, run_id, project_id,
                    session=session, attempt_id=attempt_id,
                    claim_token=token, claim_dir=claim_dir,
                )
                results[name] = ("ok", path)
            except pra_mod._ClaimLost as exc:
                results[name] = ("claim_lost", str(exc))
        finally:
            session.close()

    t_current = threading.Thread(target=_resolve, args=("current", current_token, current_claim_dir))
    t_stale = threading.Thread(target=_resolve, args=("stale", "stale-dead-token", stale_claim_dir))
    t_current.start()
    t_stale.start()
    t_current.join(timeout=30)
    t_stale.join(timeout=30)

    assert results["current"][0] == "ok", results
    assert results["stale"][0] == "claim_lost", results
    assert results["current"][1] == cache_dst

    # The shared cache is complete and valid — never a partial/corrupt
    # write from the stale caller's (never-published) temp copy.
    assert os.path.exists(cache_dst)
    assert os.path.getsize(cache_dst) > 0
    info = media_mod.probe(cache_dst)
    expected_duration = keep_entry["end_sec"] - keep_entry["start_sec"]
    assert info.duration_sec == pytest.approx(expected_duration, abs=0.3)

    # The winner's own staging file was renamed away by the publish, never
    # left behind.
    staged = pra_mod._staging_path(current_claim_dir, cache_dst)
    assert not os.path.exists(staged)
