"""
Tests for app/recovery.py — startup orphaned-run reconciliation (TR5b).

These tests use an in-memory SQLite DB, a fake Redis/RQ setup, and mocked
external clients (KieClient).  No ffmpeg, no GPU, no real network calls needed.

Coverage
--------
1. reconcile_orphaned_runs skips when the queue is NOT idle (safety gate).
2. reconcile_orphaned_runs re-enqueues genuinely orphaned runs on an idle queue.
3. reconcile_orphaned_runs covers all active statuses: queued/processing/stitching/delivering.
4. _repoll_generating_segments recovers a `success` task without resubmitting.
5. process_run no longer raises InvalidTransition when a run is already in `processing`
   (orphan resume).
6. process_run re-polls existing seedance_task_id before resubmitting (no-rebill path).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, PropertyMock

import fakeredis
import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_recovery_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus, InvalidTransition


# ---------------------------------------------------------------------------
# Shared DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_engine():
    db_path = os.path.join(_TMP_DIR, "recovery_test.db")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture(autouse=True)
def patch_get_session(db_engine, monkeypatch):
    """
    Patch get_session so that pipeline_v2 and recovery use the test DB engine.
    """
    _TestSession = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

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

    import app.pipeline_v2 as pv2
    import app.recovery as rec
    monkeypatch.setattr(pv2, "get_session", _test_get_session)
    monkeypatch.setattr(rec, "get_session", _test_get_session)


@pytest.fixture(autouse=True)
def cleanup_tmp():
    yield
    # Nothing to clean during test; module-level fixture cleans at the end.


@pytest.fixture(scope="module", autouse=True)
def _cleanup_module():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(db_session, *, status=ProjectStatus.ready) -> str:
    """Create a minimal VideoProject row and return its id."""
    project = VideoProject(
        id=str(uuid.uuid4()),
        source_type="upload",
        source_ref="/nonexistent/source.mp4",
        source_local_path="/nonexistent/source.mp4",
        status=status,
    )
    db_session.add(project)
    db_session.commit()
    return project.id


def _make_run(db_session, project_id: str, status: RunStatus) -> str:
    """Create a Run in the given status and return its id."""
    run = Run(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name="Test Run",
        prompt="swap",
        reference_image_urls=[],
        resolution="480p",
        status=status,
    )
    db_session.add(run)
    db_session.commit()
    return run.id


def _make_run_with_generating_segment(
    db_session, project_id: str, task_id: str
) -> tuple[str, str]:
    """
    Create a Run in `processing` with a SegmentDef and RunSegment in `generating`
    (with a seedance_task_id set).  Returns (run_id, run_segment_id).
    """
    # Need a SegmentDef first
    sd = SegmentDef(
        id=str(uuid.uuid4()),
        project_id=project_id,
        index=0,
        start_sec=0.0,
        end_sec=5.0,
        has_face=True,
        action="swap",
    )
    db_session.add(sd)
    db_session.flush()

    run_id = _make_run(db_session, project_id, RunStatus.processing)

    rs = RunSegment(
        id=str(uuid.uuid4()),
        run_id=run_id,
        segment_def_id=sd.id,
        index=0,
        status=SegmentStatus.generating,
        seedance_task_id=task_id,
    )
    db_session.add(rs)
    db_session.commit()
    return run_id, rs.id


def _fresh_redis():
    """A Lua-capable fake Redis with its own private server (no cross-test state)."""
    return fakeredis.FakeRedis(server=fakeredis.FakeServer())


def _make_redis_mock(*, queue_len: int = 0, started_ids: list | None = None):
    """
    Legacy helper retained for signature compatibility. Current tests patch
    recovery._live_run_ids directly rather than mocking rq internals here.
    """
    return started_ids or []


# ---------------------------------------------------------------------------
# 1. Safety gate: skip reconciliation when queue is NOT idle
# ---------------------------------------------------------------------------


class TestSafetyGate:
    def test_skip_when_run_has_a_live_job(self, db_session):
        """A run whose id has a live job must NOT be touched (still being worked)."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.processing)

        # This run has a genuinely-live job → leave it alone. (The module-scoped
        # DB may hold orphans from earlier tests; assert on THIS run specifically.)
        with patch.object(rec, "_live_run_ids", return_value={run_id}):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    rec.reconcile_orphaned_runs(_fresh_redis())
                    called_ids = {c.args[0] for c in mock_enq.call_args_list}
                    assert run_id not in called_ids

        # Run must remain in processing (untouched).
        db_session.expire_all()
        run = db_session.get(Run, run_id)
        assert run.status == RunStatus.processing

    def test_queued_run_with_live_job_untouched(self, db_session):
        """A queued run with a pending/live job must be left for that job to run."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.queued)

        with patch.object(rec, "_live_run_ids", return_value={run_id}):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    rec.reconcile_orphaned_runs(_fresh_redis())
                    called_ids = {c.args[0] for c in mock_enq.call_args_list}
                    assert run_id not in called_ids

        db_session.expire_all()
        run = db_session.get(Run, run_id)
        assert run.status == RunStatus.queued  # still queued, untouched

    def test_orphan_recovered_even_when_another_run_is_live(self, db_session, db_engine):
        """Regression for the 2026-07-10 incident: an unrelated live job must NOT
        block recovery of a genuinely orphaned run. The old global idle-gate
        skipped ALL reconciliation if *any* job was in flight; per-run liveness
        recovers the orphan while leaving the live run alone."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        live_run = _make_run(db_session, project_id, RunStatus.processing)
        orphan_run = _make_run(db_session, project_id, RunStatus.processing)

        # Only `live_run` has a live job; `orphan_run`'s worker died.
        with patch.object(rec, "_live_run_ids", return_value={live_run}):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    rec.reconcile_orphaned_runs(_fresh_redis())
                    called_ids = {c.args[0] for c in mock_enq.call_args_list}
                    assert orphan_run in called_ids   # orphan recovered
                    assert live_run not in called_ids  # live run left alone

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            assert s.get(Run, orphan_run).status == RunStatus.queued
            assert s.get(Run, live_run).status == RunStatus.processing

    def test_skip_when_reconcile_lock_held_by_another_worker(self, db_session):
        """With N workers cold-starting at once, only the lock holder reconciles.

        A worker that fails to acquire the Redis lock (set NX returns falsy) must
        skip entirely — no DB mutation, no re-enqueue — so the same orphan is not
        enqueued twice. Regression test for the 2026-06-24 double-enqueue race.
        """
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.processing)

        # Lock NOT acquired (another worker holds it): redis.set(nx=True) -> None.
        mock_redis = MagicMock()
        mock_redis.set.return_value = None

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch("app.recovery.enqueue_process_run") as mock_enq:
                rec.reconcile_orphaned_runs(mock_redis)
                mock_enq.assert_not_called()

        # Lock was attempted with NX + a TTL; not released (we never held it).
        mock_redis.set.assert_called_once()
        assert mock_redis.set.call_args.kwargs.get("nx") is True
        mock_redis.delete.assert_not_called()

        # Run untouched — the lock holder will handle it.
        db_session.expire_all()
        run = db_session.get(Run, run_id)
        assert run.status == RunStatus.processing


# ---------------------------------------------------------------------------
# 2. Orphan detection and re-enqueue on idle queue
# ---------------------------------------------------------------------------


class TestOrphanDetectionAndReenqueue:
    def test_queued_orphan_is_reenqueued(self, db_session, db_engine):
        """A run stuck in `queued` with an idle queue is re-enqueued."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.queued)

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = _fresh_redis()
                    rec.reconcile_orphaned_runs(mock_redis)
                    # The specific run must have been enqueued (may be alongside others).
                    called_ids = {call.args[0] for call in mock_enq.call_args_list}
                    assert run_id in called_ids

        # queued run stays queued (was already queued; no state change needed)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.queued

    def test_processing_orphan_reset_to_queued_and_reenqueued(self, db_session, db_engine):
        """A run stuck in `processing` is reset to queued and re-enqueued."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.processing)

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = _fresh_redis()
                    rec.reconcile_orphaned_runs(mock_redis)
                    called_ids = {call.args[0] for call in mock_enq.call_args_list}
                    assert run_id in called_ids

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.queued

    def test_stitching_orphan_reset_and_reenqueued(self, db_session, db_engine):
        """A run stuck in `stitching` is reset to queued and re-enqueued."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.stitching)

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = _fresh_redis()
                    rec.reconcile_orphaned_runs(mock_redis)
                    called_ids = {call.args[0] for call in mock_enq.call_args_list}
                    assert run_id in called_ids

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.queued

    def test_delivering_orphan_reset_and_reenqueued(self, db_session, db_engine):
        """A run stuck in `delivering` is reset to queued and re-enqueued."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.delivering)

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = _fresh_redis()
                    rec.reconcile_orphaned_runs(mock_redis)
                    called_ids = {call.args[0] for call in mock_enq.call_args_list}
                    assert run_id in called_ids

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.queued

    def test_done_and_failed_runs_not_touched(self, db_session, db_engine):
        """Completed (done/failed) runs are not reconciled."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        done_id = _make_run(db_session, project_id, RunStatus.done)
        failed_id = _make_run(db_session, project_id, RunStatus.failed)

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = _fresh_redis()
                    rec.reconcile_orphaned_runs(mock_redis)
                    # enqueue_process_run must not be called with done/failed run ids
                    called_ids = [call.args[0] for call in mock_enq.call_args_list]
                    assert done_id not in called_ids
                    assert failed_id not in called_ids

    def test_multiple_orphans_all_reenqueued(self, db_session, db_engine):
        """Multiple orphaned runs are all recovered in one reconciliation pass."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        ids = [
            _make_run(db_session, project_id, RunStatus.queued),
            _make_run(db_session, project_id, RunStatus.processing),
        ]

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = _fresh_redis()
                    rec.reconcile_orphaned_runs(mock_redis)
                    called_ids = {call.args[0] for call in mock_enq.call_args_list}
                    for rid in ids:
                        assert rid in called_ids

    def test_preprocessing_orphan_reenqueued_via_preprocess_entrypoint(
        self, db_session, db_engine
    ):
        """A run stuck in `preprocessing` is re-enqueued via the preprocess entry
        point (run_preprocess_refs), NOT the normal submit path, and is left in
        `preprocessing` so the resume-aware preprocess function picks up where it
        left off (no rebill)."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.preprocessing)

        with patch.object(rec, "_live_run_ids", return_value=set()):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_proc_enq:
                    with patch("app.recovery.enqueue_preprocess_run") as mock_pp_enq:
                        rec.reconcile_orphaned_runs(_fresh_redis())
                        # Preprocess entry point called for the preprocessing run.
                        pp_ids = {c.args[0] for c in mock_pp_enq.call_args_list}
                        proc_ids = {c.args[0] for c in mock_proc_enq.call_args_list}
                        assert run_id in pp_ids
                        assert run_id not in proc_ids

        # Run stays in `preprocessing` (not reset to queued).
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            assert s.get(Run, run_id).status == RunStatus.preprocessing


# ---------------------------------------------------------------------------
# 3. _repoll_generating_segments — avoid rebilling on resume
# ---------------------------------------------------------------------------


class TestRepollGeneratingSegments:
    def test_success_task_recovered_without_rebill(self, db_session, db_engine):
        """
        A generating segment with a task_id that is already `success` on kie.ai
        should be marked completed without re-submitting.
        """
        import app.recovery as rec

        project_id = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, rs_id = _make_run_with_generating_segment(
            db_session, project_id, task_id
        )

        # Fake KieClient that returns success for the task.
        fake_url = f"https://fake-kie.example.com/results/{task_id}.mp4"
        fake_data = {
            "state": "success",
            "resultJson": json.dumps({"resultUrls": [fake_url]}),
        }

        # Create a fake synthetic file so download_result has something to copy.
        fake_src = os.path.join(_TMP_DIR, "fake_result.mp4")
        with open(fake_src, "wb") as f:
            f.write(b"fake-video-content")

        def _fake_get_task(tid):
            return fake_data

        def _fake_download(url, dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(fake_src, dst)

        fake_kie = MagicMock()
        fake_kie.get_task.side_effect = _fake_get_task
        fake_kie.download_result.side_effect = _fake_download

        with patch("app.recovery.KieClient", return_value=fake_kie):
            rec._repoll_generating_segments(run_id, MagicMock())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.get(RunSegment, rs_id)
            assert rs.status == SegmentStatus.completed, (
                f"Expected completed, got {rs.status}"
            )
            assert rs.seedance_result_url == fake_url
            assert rs.local_result_path is not None

        # get_task called exactly once with the right task id
        fake_kie.get_task.assert_called_once_with(task_id)
        fake_kie.download_result.assert_called_once()

    def test_in_progress_task_not_touched(self, db_session, db_engine):
        """A task that is still in-progress (not success/fail) should not be reset."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, rs_id = _make_run_with_generating_segment(
            db_session, project_id, task_id
        )

        fake_kie = MagicMock()
        fake_kie.get_task.return_value = {"state": "processing"}

        with patch("app.recovery.KieClient", return_value=fake_kie):
            rec._repoll_generating_segments(run_id, MagicMock())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.get(RunSegment, rs_id)
            # Should remain generating — not completed, not reset.
            assert rs.status == SegmentStatus.generating

    def test_failed_task_not_marked_completed(self, db_session, db_engine):
        """A failed kie.ai task should NOT be marked completed (will be resubmitted)."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, rs_id = _make_run_with_generating_segment(
            db_session, project_id, task_id
        )

        fake_kie = MagicMock()
        fake_kie.get_task.return_value = {"state": "fail", "failMsg": "timeout"}

        with patch("app.recovery.KieClient", return_value=fake_kie):
            rec._repoll_generating_segments(run_id, MagicMock())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.get(RunSegment, rs_id)
            assert rs.status == SegmentStatus.generating  # untouched; will resubmit

    def test_kie_error_is_swallowed(self, db_session, db_engine):
        """A network error during get_task must not crash the recovery routine."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, rs_id = _make_run_with_generating_segment(
            db_session, project_id, task_id
        )

        fake_kie = MagicMock()
        fake_kie.get_task.side_effect = RuntimeError("network error")

        with patch("app.recovery.KieClient", return_value=fake_kie):
            # Must not raise.
            rec._repoll_generating_segments(run_id, MagicMock())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.get(RunSegment, rs_id)
            assert rs.status == SegmentStatus.generating  # unchanged


# ---------------------------------------------------------------------------
# 4. State-machine: processing → queued transition (no InvalidTransition)
# ---------------------------------------------------------------------------


class TestStateMachineProcessingResume:
    def test_processing_to_queued_valid(self):
        """processing → queued must be a valid transition (TR5b fix)."""
        from app.state_machine import can_transition, RunStatus
        assert can_transition(RunStatus.processing, RunStatus.queued), (
            "processing → queued must be valid after TR5b fix"
        )

    def test_processing_to_queued_does_not_raise(self):
        """transition() must not raise InvalidTransition for processing → queued."""
        from app.state_machine import transition, RunStatus, InvalidTransition

        class _Entity:
            status = RunStatus.processing
            updated_at = None

        e = _Entity()
        transition(e, RunStatus.queued)
        assert e.status == RunStatus.queued

    def test_processing_to_processing_still_invalid(self):
        """processing → processing must still be invalid (no self-loops)."""
        from app.state_machine import can_transition, RunStatus
        assert not can_transition(RunStatus.processing, RunStatus.processing)

    def test_existing_valid_transitions_unchanged(self):
        """All previously valid transitions must still be valid."""
        from app.state_machine import can_transition, RunStatus
        pairs = [
            (RunStatus.created, RunStatus.queued),
            (RunStatus.queued, RunStatus.processing),
            (RunStatus.processing, RunStatus.stitching),
            (RunStatus.processing, RunStatus.failed),
            (RunStatus.stitching, RunStatus.delivering),
            (RunStatus.stitching, RunStatus.failed),
            (RunStatus.delivering, RunStatus.done),
            (RunStatus.delivering, RunStatus.failed),
            (RunStatus.done, RunStatus.queued),
            (RunStatus.failed, RunStatus.queued),
        ]
        for src, dst in pairs:
            assert can_transition(src, dst), (
                f"{src.value} → {dst.value} must still be valid"
            )


# ---------------------------------------------------------------------------
# 5. process_run orphan resume: no InvalidTransition on processing state
# ---------------------------------------------------------------------------


class TestProcessRunOrphanResume:
    """
    Verify that process_run handles a run already in `processing` (the orphan
    resume case) without raising InvalidTransition.

    These tests use a fake source file and fake KieClient — no ffmpeg or GPU.
    They are integration-style: they call process_run directly and verify the
    run reaches `done`.
    """

    def _setup_ready_project_with_segments(self, db_session, db_engine) -> str:
        """
        Create a VideoProject in `ready` state with two SegmentDefs (1 swap, 1 keep).
        Returns project_id.
        """
        import app.pipeline_v2 as pv2

        # Create a minimal fake source video file (just bytes; media_mod is mocked).
        src = os.path.join(_TMP_DIR, f"src_{uuid.uuid4().hex[:8]}.mp4")
        with open(src, "wb") as f:
            f.write(b"fake-video")

        project = VideoProject(
            id=str(uuid.uuid4()),
            source_type="upload",
            source_ref=src,
            source_local_path=src,
            status=ProjectStatus.ready,
        )
        db_session.add(project)
        db_session.commit()

        sd_swap = SegmentDef(
            id=str(uuid.uuid4()),
            project_id=project.id,
            index=0,
            start_sec=0.0,
            end_sec=5.0,
            has_face=True,
            action="swap",
        )
        sd_keep = SegmentDef(
            id=str(uuid.uuid4()),
            project_id=project.id,
            index=1,
            start_sec=5.0,
            end_sec=10.0,
            has_face=False,
            action="keep",
        )
        db_session.add(sd_swap)
        db_session.add(sd_keep)
        db_session.commit()
        return project.id

    def _run_process_run_with_mocks(
        self, run_id: str, *, monkeypatch, db_engine
    ) -> None:
        """
        Run process_run with all external calls mocked (media, kie, gdrive).
        """
        import app.pipeline_v2 as pv2
        import json, shutil

        fake_result_path = os.path.join(_TMP_DIR, "fake_result.mp4")
        with open(fake_result_path, "wb") as f:
            f.write(b"fake-video-result")

        # Mock media probing and ffmpeg operations.
        class _FakeVideoInfo:
            duration_sec = 10.0
            width = 320
            height = 240
            fps = 25.0
            aspect_ratio = "4:3"

        def _fake_probe(path):
            return _FakeVideoInfo()

        def _fake_get_default_target(info):
            return (320, 240, 25.0)

        def _fake_cut_clip(src, start, end, dst, **kw):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(b"fake-clip")

        def _fake_stitch(clips, *, audio_source, dst, **kw):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(b"fake-stitched")

        monkeypatch.setattr(pv2.media_mod, "probe", _fake_probe)
        monkeypatch.setattr(pv2.media_mod, "get_default_target", _fake_get_default_target)
        monkeypatch.setattr(pv2.media_mod, "cut_clip", _fake_cut_clip)
        monkeypatch.setattr(pv2.media_mod, "stitch", _fake_stitch)

        # Fake KieClient.
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        class _FakeKie:
            def upload_file(self, path, upload_path="", **kw):
                return f"https://fake-kie.example.com/{os.path.basename(path)}"

            def create_task(self, **kw):
                return task_id

            def get_task(self, tid, **kw):
                url = f"https://fake-kie.example.com/results/{tid}.mp4"
                return {
                    "state": "success",
                    "resultJson": json.dumps({"resultUrls": [url]}),
                }

            def download_result(self, url, dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(fake_result_path, dst)

        class _FakeGDrive:
            def upload_file(self, path, folder_id=None, name=None, **kw):
                return {"id": "gdrive-fake", "webViewLink": "https://drive.google.com/x"}

        pv2.process_run(run_id, kie=_FakeKie(), gdrive=_FakeGDrive())

    def test_orphan_resume_from_processing_no_invalid_transition(
        self, db_session, db_engine, monkeypatch
    ):
        """
        process_run called on a run already in `processing` must not raise
        InvalidTransition — it should reset to queued first, then re-process.
        """
        project_id = self._setup_ready_project_with_segments(db_session, db_engine)

        # Create the run directly in `processing` (simulating an orphaned crash state).
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Orphaned Run",
            prompt="swap",
            reference_image_urls=[],
            resolution="480p",
            status=RunStatus.processing,
        )
        db_session.add(run)
        db_session.commit()
        run_id = run.id

        # Must not raise.
        self._run_process_run_with_mocks(run_id, monkeypatch=monkeypatch, db_engine=db_engine)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            r = s.get(Run, run_id)
            assert r.status == RunStatus.done, (
                f"Expected done after orphan resume, got {r.status}"
            )

    def test_normal_queued_start_still_works(
        self, db_session, db_engine, monkeypatch
    ):
        """
        Regression: the normal queued→processing path must not be broken by the
        orphan-resume fix.
        """
        project_id = self._setup_ready_project_with_segments(db_session, db_engine)

        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Normal Run",
            prompt="swap",
            reference_image_urls=[],
            resolution="480p",
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()
        run_id = run.id

        self._run_process_run_with_mocks(run_id, monkeypatch=monkeypatch, db_engine=db_engine)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            r = s.get(Run, run_id)
            assert r.status == RunStatus.done


# ---------------------------------------------------------------------------
# 6. process_run no-rebill: re-poll existing task_id before resubmit
# ---------------------------------------------------------------------------


class TestProcessRunNoBillOnResume:
    """
    Verify that process_run checks kie.ai for a `generating` segment's existing
    task_id before resubmitting — and skips resubmission when task is success.
    """

    def test_existing_success_task_not_resubmitted(
        self, db_session, db_engine, monkeypatch
    ):
        """
        A RunSegment in `generating` with a task_id that is already `success`
        on kie.ai must be marked completed WITHOUT calling create_task again.
        """
        import app.pipeline_v2 as pv2

        # Minimal project + run setup.
        src = os.path.join(_TMP_DIR, f"src_{uuid.uuid4().hex[:8]}.mp4")
        with open(src, "wb") as f:
            f.write(b"fake-video")

        project = VideoProject(
            id=str(uuid.uuid4()),
            source_type="upload",
            source_ref=src,
            source_local_path=src,
            status=ProjectStatus.ready,
        )
        db_session.add(project)
        db_session.commit()

        sd = SegmentDef(
            id=str(uuid.uuid4()),
            project_id=project.id,
            index=0,
            start_sec=0.0,
            end_sec=5.0,
            has_face=True,
            action="swap",
        )
        db_session.add(sd)
        db_session.commit()

        run = Run(
            id=str(uuid.uuid4()),
            project_id=project.id,
            name="Resume Run",
            prompt="swap",
            reference_image_urls=[],
            resolution="480p",
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()

        # Pre-create a RunSegment in `generating` with an existing task_id.
        existing_task_id = f"existing-task-{uuid.uuid4().hex[:8]}"
        rs = RunSegment(
            id=str(uuid.uuid4()),
            run_id=run.id,
            segment_def_id=sd.id,
            index=0,
            status=SegmentStatus.generating,
            seedance_task_id=existing_task_id,
        )
        db_session.add(rs)
        db_session.commit()

        # Track which task IDs were passed to create_task.
        created_task_ids: list[str] = []
        get_task_calls: list[str] = []
        fake_result = os.path.join(_TMP_DIR, "fake_result2.mp4")
        with open(fake_result, "wb") as f:
            f.write(b"fake-video-result")

        fake_result_url = f"https://fake-kie.example.com/results/{existing_task_id}.mp4"

        class _FakeKie:
            def upload_file(self, path, upload_path="", **kw):
                return f"https://fake-kie.example.com/{os.path.basename(path)}"

            def create_task(self, **kw):
                new_id = f"new-task-{uuid.uuid4().hex[:8]}"
                created_task_ids.append(new_id)
                return new_id

            def get_task(self, tid, **kw):
                get_task_calls.append(tid)
                if tid == existing_task_id:
                    # Existing task is already success.
                    return {
                        "state": "success",
                        "resultJson": json.dumps({"resultUrls": [fake_result_url]}),
                    }
                # Any new task also succeeds.
                url = f"https://fake-kie.example.com/results/{tid}.mp4"
                return {
                    "state": "success",
                    "resultJson": json.dumps({"resultUrls": [url]}),
                }

            def download_result(self, url, dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(fake_result, dst)

        class _FakeGDrive:
            def upload_file(self, path, folder_id=None, name=None, **kw):
                return {"id": "gdrive-fake", "webViewLink": "https://drive.google.com/x"}

        class _FakeVideoInfo:
            duration_sec = 10.0
            width = 320
            height = 240
            fps = 25.0
            aspect_ratio = "4:3"

        def _fake_probe(path):
            return _FakeVideoInfo()

        def _fake_get_default_target(info):
            return (320, 240, 25.0)

        def _fake_cut_clip(src, start, end, dst, **kw):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(b"fake-clip")

        def _fake_stitch(clips, *, audio_source, dst, **kw):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(b"fake-stitched")

        monkeypatch.setattr(pv2.media_mod, "probe", _fake_probe)
        monkeypatch.setattr(pv2.media_mod, "get_default_target", _fake_get_default_target)
        monkeypatch.setattr(pv2.media_mod, "cut_clip", _fake_cut_clip)
        monkeypatch.setattr(pv2.media_mod, "stitch", _fake_stitch)

        pv2.process_run(run.id, kie=_FakeKie(), gdrive=_FakeGDrive())

        # create_task must NOT have been called (existing task was already success).
        assert created_task_ids == [], (
            f"create_task should not have been called, but was called with: {created_task_ids}"
        )

        # get_task must have been called with the existing task_id.
        assert existing_task_id in get_task_calls, (
            f"get_task should have been called with {existing_task_id}, got {get_task_calls}"
        )

        # RunSegment must be completed.
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs_fresh = s.get(RunSegment, rs.id)
            assert rs_fresh.status == SegmentStatus.completed
            assert rs_fresh.seedance_result_url == fake_result_url

        # Run must be done.
        with Session() as s:
            r = s.get(Run, run.id)
            assert r.status == RunStatus.done


# ---------------------------------------------------------------------------
# 7. Per-run liveness — the core of the 2026-07-10 fix
# ---------------------------------------------------------------------------


class TestLiveWorkerNames:
    def test_stale_heartbeat_worker_excluded(self):
        """A worker still registered but heartbeated long ago (killed by a
        redeploy) must NOT count as live — otherwise its abandoned started jobs
        would keep masking the orphaned run they reference."""
        import app.recovery as rec
        from datetime import datetime, timedelta

        now = datetime.utcnow()

        fresh = MagicMock()
        fresh.name = "w_fresh"
        fresh.last_heartbeat = now  # heartbeat right now

        stale = MagicMock()
        stale.name = "w_stale"
        stale.last_heartbeat = now - timedelta(
            seconds=rec._WORKER_HEARTBEAT_FRESH_SEC + 60
        )

        with patch.object(rec, "Worker") as MockWorker:
            MockWorker.all.return_value = [fresh, stale]
            names = rec._live_worker_names(MagicMock())

        assert names == {"w_fresh"}

    def test_worker_all_failure_returns_empty(self):
        """A Redis hiccup in Worker.all() must degrade to 'no live workers'
        (so started jobs are treated as abandoned) rather than raising."""
        import app.recovery as rec

        with patch.object(rec, "Worker") as MockWorker:
            MockWorker.all.side_effect = RuntimeError("redis down")
            assert rec._live_worker_names(MagicMock()) == set()


class TestLiveRunIds:
    def test_abandoned_started_job_is_not_live(self):
        """Regression for the incident: a started job whose owning worker is dead
        must NOT appear in the live set, while pending, scheduled, and
        live-worker-owned started jobs do."""
        import app.recovery as rec

        # job_id → (args, worker_name)
        jobs = {
            "j_pending":   (("R_pending",), None),
            "j_sched":     (("R_sched",), None),
            "j_start_live": (("R_live",), "w_live"),
            "j_start_dead": (("R_abandoned",), "w_dead"),  # worker killed
        }

        def _fake_fetch_many(job_ids, connection=None):
            result = []
            for jid in job_ids:
                args, wname = jobs[jid]
                j = MagicMock()
                j.args = args
                j.worker_name = wname
                result.append(j)
            return result

        fake_q = MagicMock()
        fake_q.job_ids = ["j_pending"]
        fake_sched = MagicMock()
        fake_sched.get_job_ids.return_value = ["j_sched"]
        fake_started = MagicMock()
        fake_started.get_job_ids.return_value = ["j_start_live", "j_start_dead"]

        with patch.object(rec, "_live_worker_names", return_value={"w_live"}), \
             patch.object(rec, "Queue", return_value=fake_q), \
             patch.object(rec, "ScheduledJobRegistry", return_value=fake_sched), \
             patch.object(rec, "StartedJobRegistry", return_value=fake_started), \
             patch.object(rec, "Job") as MockJob:
            MockJob.fetch_many.side_effect = _fake_fetch_many
            live = rec._live_run_ids(MagicMock())

        assert "R_pending" in live
        assert "R_sched" in live
        assert "R_live" in live
        assert "R_abandoned" not in live  # the whole point

    def test_vanished_job_is_skipped(self):
        """A job id that disappears between listing and fetch is skipped, not fatal.

        Job.fetch_many represents a vanished job hash as a None entry in the
        returned list (rather than raising), so this simulates that contract.
        """
        import app.recovery as rec

        fake_q = MagicMock()
        fake_q.job_ids = ["gone"]
        fake_sched = MagicMock()
        fake_sched.get_job_ids.return_value = []
        fake_started = MagicMock()
        fake_started.get_job_ids.return_value = []

        with patch.object(rec, "_live_worker_names", return_value=set()), \
             patch.object(rec, "Queue", return_value=fake_q), \
             patch.object(rec, "ScheduledJobRegistry", return_value=fake_sched), \
             patch.object(rec, "StartedJobRegistry", return_value=fake_started), \
             patch.object(rec, "Job") as MockJob:
            MockJob.fetch_many.side_effect = lambda job_ids, connection=None: [None for _ in job_ids]
            live = rec._live_run_ids(MagicMock())

        assert live == set()

    def test_fetch_many_batch_failure_falls_back_per_id(self):
        """If the whole fetch_many pipeline call raises (e.g. a Redis hiccup),
        the batch must not crash the reconcile tick — it degrades to a per-id
        Job.fetch fallback for that batch."""
        import app.recovery as rec

        def _fake_fetch(jid, connection=None):
            if jid == "j_ok":
                j = MagicMock()
                j.args = ("R_ok",)
                j.worker_name = None
                return j
            raise KeyError(jid)

        fake_q = MagicMock()
        fake_q.job_ids = ["j_ok", "j_gone"]
        fake_sched = MagicMock()
        fake_sched.get_job_ids.return_value = []
        fake_started = MagicMock()
        fake_started.get_job_ids.return_value = []

        with patch.object(rec, "_live_worker_names", return_value=set()), \
             patch.object(rec, "Queue", return_value=fake_q), \
             patch.object(rec, "ScheduledJobRegistry", return_value=fake_sched), \
             patch.object(rec, "StartedJobRegistry", return_value=fake_started), \
             patch.object(rec, "Job") as MockJob:
            MockJob.fetch_many.side_effect = RuntimeError("redis pipeline error")
            MockJob.fetch.side_effect = _fake_fetch
            live = rec._live_run_ids(MagicMock())

        assert "R_ok" in live


# ---------------------------------------------------------------------------
# 7. Reaping orphaned derived creatives (captions, tail edits, overlays, merges)
# ---------------------------------------------------------------------------


class TestReapOrphanedDerivedCreatives:
    """A derived-creative job interrupted mid-flight must not stay 'active'.

    Runs are resumed; these are failed instead, because resuming a caption
    re-bills a transcription and delivers an unasked-for file to Drive. Before
    this reaper nothing ever revisited the row, so the UI showed a spinner that
    could never resolve — caption fbd2d5fd sat in `transcribing` after a deploy
    killed its worker on 2026-08-11.
    """

    @staticmethod
    def _age(db_session, row, seconds: float):
        """Backdate updated_at past the grace window (the default is 'just now',
        which the reaper deliberately protects)."""
        from datetime import datetime, timedelta
        row.updated_at = datetime.utcnow() - timedelta(seconds=seconds)
        db_session.commit()

    def _caption(self, db_session, status, *, aged=True):
        from app.models import RunCaption
        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        cap = RunCaption(id=str(uuid.uuid4()), run_id=rid, status=status)
        db_session.add(cap)
        db_session.commit()
        if aged:
            self._age(db_session, cap, 600)
        return cap

    def test_orphaned_caption_in_transcribing_is_retried(self, db_session, monkeypatch):
        """AICP-351 §5.2: the ASR of an orphaned caption is re-dispatched (bounded)
        instead of failed; its dispatch fence is revoked."""
        import app.recovery as rec
        import app.tasks as tasks_mod
        from app.state_machine import CaptionStatus

        from app.paid_attempts_models import PaidSubmitAttempt

        sent = []
        monkeypatch.setattr(tasks_mod, "enqueue_process_caption",
                            lambda cid, dispatch=None: sent.append((cid, dispatch)))
        cap = self._caption(db_session, CaptionStatus.transcribing)
        cap.asr_claim = "old-token"
        # Its first ASR attempt ended with an unknown outcome.
        db_session.add(PaidSubmitAttempt(owner_kind="llm_caption_asr", owner_id=f"{cap.id}:0",
                                         attempt_no=1, kind="llm", status="unknown"))
        db_session.commit()
        self._age(db_session, cap, 600)

        reaped = rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(cap)
        assert reaped == {} and len(sent) == 1 and sent[0][0] == cap.id
        assert cap.status == CaptionStatus.transcribing
        assert cap.asr_claim == f"dispatch:{sent[0][1]}"      # the old worker is fenced out

    def test_caption_that_died_before_any_asr_call_is_failed(self, db_session):
        import app.recovery as rec
        from app.state_machine import CaptionStatus

        cap = self._caption(db_session, CaptionStatus.transcribing)
        assert rec._reap_orphaned_derived_creatives(set()) == {"caption": 1}
        db_session.refresh(cap)
        assert cap.status == CaptionStatus.failed

    def test_orphaned_caption_is_failed_with_a_readable_reason(self, db_session):
        import app.recovery as rec
        from app.paid_attempts_models import PaidSubmitAttempt
        from app.state_machine import CaptionStatus

        cap = self._caption(db_session, CaptionStatus.transcribing)
        # The automatic bound of this epoch is spent (AICP-351): reaped as before.
        for n in (1, 2, 3):
            db_session.add(PaidSubmitAttempt(owner_kind="llm_caption_asr",
                                             owner_id=f"{cap.id}:0", attempt_no=n,
                                             kind="llm", status="unknown"))
        db_session.commit()

        reaped = rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(cap)
        assert cap.status == CaptionStatus.failed
        assert reaped == {"caption": 1}
        # The operator has to understand this without reading the source.
        assert "restart" in cap.error_message.lower()
        assert "retry" in cap.error_message.lower()
        assert "nothing was delivered" in cap.error_message.lower()

    def test_caption_with_a_live_job_is_left_alone(self, db_session):
        """Another worker is on it right now — reaping would be a false positive."""
        import app.recovery as rec
        from app.state_machine import CaptionStatus

        cap = self._caption(db_session, CaptionStatus.burning)

        rec._reap_orphaned_derived_creatives({cap.id})

        db_session.refresh(cap)
        assert cap.status == CaptionStatus.burning
        assert cap.error_message is None

    def test_freshly_queued_caption_is_left_alone(self, db_session):
        """The API commits the row BEFORE enqueuing the RQ job, so a healthy job
        is briefly `queued` in Postgres with nothing in Redis. The grace window
        is what stops the reaper failing a job that is about to run."""
        import app.recovery as rec
        from app.state_machine import CaptionStatus

        cap = self._caption(db_session, CaptionStatus.queued, aged=False)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(cap)
        assert cap.status == CaptionStatus.queued
        assert cap.error_message is None

    def test_long_orphaned_queued_caption_is_reaped(self, db_session):
        """Past the grace window with nothing in Redis, `queued` is a genuine
        orphan — the job was lost before any worker picked it up."""
        import app.recovery as rec
        from app.state_machine import CaptionStatus

        cap = self._caption(db_session, CaptionStatus.queued)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(cap)
        assert cap.status == CaptionStatus.failed

    def test_terminal_captions_are_never_touched(self, db_session):
        import app.recovery as rec
        from app.state_machine import CaptionStatus

        done = self._caption(db_session, CaptionStatus.done)
        failed = self._caption(db_session, CaptionStatus.failed)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(done)
        db_session.refresh(failed)
        assert done.status == CaptionStatus.done
        assert done.error_message is None
        assert failed.status == CaptionStatus.failed

    def test_orphaned_tail_edit_is_failed(self, db_session):
        import app.recovery as rec
        from app.models import RunTailEdit
        from app.state_machine import TailEditStatus

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        edit = RunTailEdit(
            id=str(uuid.uuid4()), run_id=rid, cut_sec=19.0,
            status=TailEditStatus.processing,
        )
        db_session.add(edit)
        db_session.commit()
        self._age(db_session, edit, 600)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert "restart" in edit.error_message.lower()

    def test_orphaned_audio_patch_is_failed(self, db_session):
        """An unreaped audio patch is worse than an unreaped tail edit: it also
        holds uq_run_audio_patches_active_per_run, so it blocks the operator
        from ever retrying the correction."""
        import app.recovery as rec
        from app.models import RunAudioPatch
        from app.state_machine import AudioPatchStatus

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        patch = RunAudioPatch(
            id=str(uuid.uuid4()), run_id=rid, donor_run_id=str(uuid.uuid4()),
            target_start_sec=3.0, target_end_sec=4.0, donor_start_sec=1.0,
            crossfade_sec=0.06, base_result_path="/tmp/final.mp4",
            status=AudioPatchStatus.delivering,
        )
        db_session.add(patch)
        db_session.commit()
        self._age(db_session, patch, 600)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(patch)
        assert patch.status == AudioPatchStatus.failed
        assert "restart" in patch.error_message.lower()

    def test_promoted_audio_patch_is_never_touched(self, db_session):
        """`done` is terminal here and means the run's result has already been
        re-pointed — reopening it would be reopening a delivered change."""
        import app.recovery as rec
        from app.models import RunAudioPatch
        from app.state_machine import AudioPatchStatus

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        patch = RunAudioPatch(
            id=str(uuid.uuid4()), run_id=rid, donor_run_id=str(uuid.uuid4()),
            target_start_sec=3.0, target_end_sec=4.0, donor_start_sec=1.0,
            crossfade_sec=0.06, base_result_path="/tmp/final.mp4",
            status=AudioPatchStatus.done,
        )
        db_session.add(patch)
        db_session.commit()
        self._age(db_session, patch, 600)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(patch)
        assert patch.status == AudioPatchStatus.done
        assert patch.error_message is None

    def test_orphaned_merge_is_failed(self, db_session):
        import app.recovery as rec
        from app.models import InsertMerge
        from app.state_machine import MergeStatus

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        merge = InsertMerge(
            id=str(uuid.uuid4()), target_project_id=pid, child_run_id=rid,
            x=0, y=0, width=100, height=100, start_sec=0.0, end_sec=5.0,
            status=MergeStatus.compositing,
        )
        db_session.add(merge)
        db_session.commit()
        self._age(db_session, merge, 600)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed

    def test_queued_overlay_is_failed_despite_no_direct_edge(self, db_session):
        """OverlayStatus has no `queued -> failed` edge. The reaper must still
        land it in failed rather than let the guard strand it in a state nothing
        will ever revisit."""
        import app.recovery as rec
        from app.models import OverlayJob
        from app.state_machine import OverlayStatus

        job = OverlayJob(
            id=str(uuid.uuid4()), source_type="upload",
            status=OverlayStatus.queued, overlays=[],
        )
        db_session.add(job)
        db_session.commit()
        self._age(db_session, job, 600)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(job)
        assert job.status == OverlayStatus.failed
        # OverlayJob spells it `error`, not `error_message`.
        assert "restart" in job.error.lower()

    def test_created_overlay_is_never_reaped(self, db_session):
        """`created` is the pre-queue state of a job the operator is still
        composing in the editor. It has no RQ job BY DESIGN — reaping it would
        destroy in-progress work rather than recover it."""
        import app.recovery as rec
        from app.models import OverlayJob
        from app.state_machine import OverlayStatus

        job = OverlayJob(
            id=str(uuid.uuid4()), source_type="upload",
            status=OverlayStatus.created, overlays=[],
        )
        db_session.add(job)
        db_session.commit()
        self._age(db_session, job, 6000)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(job)
        assert job.status == OverlayStatus.created
        assert job.error is None

    def test_soft_deleted_overlay_is_skipped(self, db_session):
        import app.recovery as rec
        from datetime import datetime
        from app.models import OverlayJob
        from app.state_machine import OverlayStatus

        job = OverlayJob(
            id=str(uuid.uuid4()), source_type="upload",
            status=OverlayStatus.processing, overlays=[],
            deleted_at=datetime.utcnow(),
        )
        db_session.add(job)
        db_session.commit()
        self._age(db_session, job, 600)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(job)
        assert job.status == OverlayStatus.processing

    def test_aware_updated_at_does_not_raise(self, db_session):
        """models._utcnow is timezone-AWARE while the reaper compares against
        naive datetime.utcnow(); mixing them raises TypeError. Postgres returns
        naive values but a defaulted in-session object is aware, so the
        normaliser has to handle both."""
        import app.recovery as rec
        from datetime import datetime, timedelta, timezone
        from app.state_machine import CaptionStatus

        cap = self._caption(db_session, CaptionStatus.burning, aged=False)
        cap.updated_at = datetime.now(timezone.utc) - timedelta(seconds=600)
        db_session.commit()

        reaped = rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(cap)
        assert cap.status == CaptionStatus.failed
        assert reaped == {"caption": 1}

    def test_reconcile_runs_the_reaper(self, db_session):
        """The reaper must be wired into the boot + periodic reconcile pass, not
        just callable on its own."""
        import app.recovery as rec
        from app.state_machine import CaptionStatus

        cap = self._caption(db_session, CaptionStatus.delivering)

        with patch.object(rec, "_live_run_ids", return_value=set()):
            rec._reconcile_orphaned_runs_impl(_fresh_redis())

        db_session.refresh(cap)
        assert cap.status == CaptionStatus.failed


# ---------------------------------------------------------------------------
# 7b. Per-segment GPT Image 2 regeneration jobs — reaping + resume-safe retry
# ---------------------------------------------------------------------------


class TestSegmentImageJobReaping:
    """RunSegmentImageJob is registered in _DERIVED_CREATIVE_SPECS with the
    same grace window as captions/tail edits/overlays/merges — a worker that
    dies after create_image_task has already billed kie.ai and left the row
    stuck `generating`; without reaping, api_v2's 409 gate would leave the
    operator unable to re-run the segment forever."""

    @staticmethod
    def _age(db_session, row, seconds: float):
        from datetime import datetime, timedelta
        row.updated_at = datetime.utcnow() - timedelta(seconds=seconds)
        db_session.commit()

    def _job(self, db_session, status, *, aged=True):
        from app.models import RunSegmentImageJob

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        sd = SegmentDef(
            id=str(uuid.uuid4()), project_id=pid, index=0,
            start_sec=0.0, end_sec=10.0, has_face=True, action="swap",
        )
        db_session.add(sd)
        rs = RunSegment(
            id=str(uuid.uuid4()), run_id=rid, segment_def_id=sd.id, index=0,
            status=SegmentStatus.completed,
        )
        db_session.add(rs)
        db_session.commit()
        job = RunSegmentImageJob(
            id=str(uuid.uuid4()), run_id=rid, run_segment_id=rs.id,
            prompt="a prompt", frame_sec=5.0, status=status,
        )
        db_session.add(job)
        db_session.commit()
        if aged:
            self._age(db_session, job, 600)
        return job

    def test_orphaned_generating_job_is_failed_with_a_readable_reason(self, db_session):
        import app.recovery as rec
        from app.state_machine import SegmentImageStatus

        job = self._job(db_session, SegmentImageStatus.generating)

        reaped = rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(job)
        assert job.status == SegmentImageStatus.failed
        assert reaped == {"segment image job": 1}
        assert "restart" in job.error_message.lower()
        assert "retry" in job.error_message.lower()

    def test_live_job_is_left_alone(self, db_session):
        import app.recovery as rec
        from app.state_machine import SegmentImageStatus

        job = self._job(db_session, SegmentImageStatus.generating)

        rec._reap_orphaned_derived_creatives({job.id})

        db_session.refresh(job)
        assert job.status == SegmentImageStatus.generating
        assert job.error_message is None

    def test_freshly_queued_job_within_grace_window_is_left_alone(self, db_session):
        import app.recovery as rec
        from app.state_machine import SegmentImageStatus

        job = self._job(db_session, SegmentImageStatus.queued, aged=False)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(job)
        assert job.status == SegmentImageStatus.queued

    def test_terminal_jobs_are_never_touched(self, db_session):
        import app.recovery as rec
        from app.state_machine import SegmentImageStatus

        done = self._job(db_session, SegmentImageStatus.done)
        failed = self._job(db_session, SegmentImageStatus.failed)

        rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(done)
        db_session.refresh(failed)
        assert done.status == SegmentImageStatus.done
        assert failed.status == SegmentImageStatus.failed

    def test_job_liveness_is_judged_by_its_own_id_not_the_parent_run(self, db_session):
        """args[0] for the segment-image RQ entrypoints (enqueue_process_
        segment_image / _enqueue_segment_image_poll) is the JOB id, never the
        run id. A job keyed off its run's id would let a `done` run's
        unrelated liveness (or lack of it) decide this job's fate instead of
        its own."""
        import app.recovery as rec
        from app.state_machine import SegmentImageStatus

        job = self._job(db_session, SegmentImageStatus.generating)

        # The parent RUN's id is "live" (e.g. some unrelated activity), but
        # the JOB's own id is not — it must still be reaped.
        rec._reap_orphaned_derived_creatives({job.run_id})

        db_session.refresh(job)
        assert job.status == SegmentImageStatus.failed


class TestRangeAttemptReaping:
    """RunSegmentRangeAttempt is registered in _DERIVED_CREATIVE_SPECS with
    the same grace window as every other derived creative — but unlike
    them, one of its fields (submit_billed_at) can flag that the worker
    that died may have already made kie.ai's ONE PAID call for this
    attempt. This reaper's own grace window (300s) is shorter than
    pipeline_range_attempt._CLAIM_LEASE_SEC (900s), so in production THIS
    reaper — not that module's own expired-claim check — is what actually
    resolves a fully-dead submit-phase worker; it must carry the same
    ambiguous-billing warning that check would use (found by the mandatory
    Cursor CLI review gate: before this fix, the reaper always won that
    race and always stamped the generic "Press Retry" message, silently
    discarding the ambiguous-billing flag)."""

    @staticmethod
    def _age(db_session, row, seconds: float):
        from datetime import datetime, timedelta
        row.updated_at = datetime.utcnow() - timedelta(seconds=seconds)
        db_session.commit()

    def _attempt(self, db_session, status, *, submit_billed_at=None, seedance_task_id=None, aged=True):
        from app.models import RunSegmentRangeAttempt

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        sd = SegmentDef(
            id=str(uuid.uuid4()), project_id=pid, index=0,
            start_sec=0.0, end_sec=10.0, has_face=True, action="swap",
        )
        db_session.add(sd)
        rs = RunSegment(
            id=str(uuid.uuid4()), run_id=rid, segment_def_id=sd.id, index=0,
            status=SegmentStatus.completed,
        )
        db_session.add(rs)
        db_session.commit()
        attempt = RunSegmentRangeAttempt(
            id=str(uuid.uuid4()), run_id=rid, run_segment_id=rs.id,
            requested_start_sec=0.0, requested_end_sec=8.0,
            base_segment_snapshot=[], candidate_segment_snapshot=[],
            status=status, submit_billed_at=submit_billed_at,
            seedance_task_id=seedance_task_id,
        )
        db_session.add(attempt)
        db_session.commit()
        if aged:
            self._age(db_session, attempt, 600)
        return attempt

    def test_orphaned_attempt_without_billing_gets_the_generic_message(self, db_session):
        import app.recovery as rec
        from app.state_machine import RangeAttemptStatus

        attempt = self._attempt(db_session, RangeAttemptStatus.queued)

        reaped = rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(attempt)
        assert attempt.status == RangeAttemptStatus.failed
        assert reaped == {"range attempt": 1}
        assert "restart" in attempt.error_message.lower()
        assert "AMBIGUOUS" not in attempt.error_message

    def test_orphaned_attempt_with_ambiguous_billing_gets_the_ambiguous_message(
        self, db_session,
    ):
        """The crash-recovery scenario the review gate found unhandled: a
        worker committed submit_billed_at (about to call, or just called,
        kie.ai's paid createTask) and then died before persisting a
        seedance_task_id. The reaper must flag this explicitly, never with
        the generic "Press Retry" text — that would invite exactly the
        blind retry this whole mechanism exists to prevent."""
        import app.recovery as rec
        from datetime import datetime
        from app.state_machine import RangeAttemptStatus

        attempt = self._attempt(
            db_session, RangeAttemptStatus.generating,
            submit_billed_at=datetime.utcnow(),
        )

        reaped = rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(attempt)
        assert attempt.status == RangeAttemptStatus.failed
        assert reaped == {"range attempt": 1}
        assert "AMBIGUOUS" in attempt.error_message
        assert "idempotency" in attempt.error_message.lower()

    def test_orphaned_attempt_with_billing_AND_a_persisted_task_id_is_not_ambiguous(
        self, db_session,
    ):
        """submit_billed_at survives forever once set (it is never
        cleared), but a task id WAS persisted before the crash — e.g. a
        later phase (stitch/deliver) died instead. That is an ordinary
        orphan, not an ambiguous one — and, per Finding 3
        (2026-08-30 follow-up), a task id being persisted also means it is
        now RECOVERED rather than failed: finalizing makes no further paid
        call. See TestRangeAttemptFinalizeRecovery for the dedicated
        coverage; this test only needs to keep asserting "not ambiguous"."""
        import app.recovery as rec
        from datetime import datetime
        from app.state_machine import RangeAttemptStatus
        from unittest.mock import patch as _patch

        attempt = self._attempt(
            db_session, RangeAttemptStatus.stitching,
            submit_billed_at=datetime.utcnow(), seedance_task_id="real-task-1",
        )

        with _patch("app.recovery._enqueue_range_attempt_poll"):
            rec._reap_orphaned_derived_creatives(set())

        db_session.refresh(attempt)
        assert attempt.status == RangeAttemptStatus.generating  # recovered, not failed
        assert attempt.error_message is None
        assert attempt.claim_owner is None

    def test_live_attempt_is_left_alone(self, db_session):
        import app.recovery as rec
        from datetime import datetime
        from app.state_machine import RangeAttemptStatus

        attempt = self._attempt(
            db_session, RangeAttemptStatus.generating,
            submit_billed_at=datetime.utcnow(),
        )

        rec._reap_orphaned_derived_creatives({attempt.id})

        db_session.refresh(attempt)
        assert attempt.status == RangeAttemptStatus.generating
        assert attempt.error_message is None


class TestRangeAttemptFinalizeRecovery:
    """Mandatory regression for Finding 3 (independent audit, 2026-08-30):
    a RunSegmentRangeAttempt whose submission already fully succeeded
    (seedance_task_id persisted) must be RECOVERED — claim cleared, exactly
    one finalize poll enqueued — not failed, since finalizing it makes no
    further paid kie.ai call. The ambiguous-billing case (submit_billed_at
    set, no task id) keeps the EXISTING manual-only handling unchanged."""

    @staticmethod
    def _age(db_session, row, seconds: float):
        from datetime import datetime, timedelta
        row.updated_at = datetime.utcnow() - timedelta(seconds=seconds)
        db_session.commit()

    def _attempt(
        self, db_session, status, *,
        submit_billed_at=None, seedance_task_id=None,
        claim_owner=None, claim_expires_at=None, aged=True,
    ):
        from app.models import RunSegmentRangeAttempt

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        sd = SegmentDef(
            id=str(uuid.uuid4()), project_id=pid, index=0,
            start_sec=0.0, end_sec=10.0, has_face=True, action="swap",
        )
        db_session.add(sd)
        rs = RunSegment(
            id=str(uuid.uuid4()), run_id=rid, segment_def_id=sd.id, index=0,
            status=SegmentStatus.completed,
        )
        db_session.add(rs)
        db_session.commit()
        attempt = RunSegmentRangeAttempt(
            id=str(uuid.uuid4()), run_id=rid, run_segment_id=rs.id,
            requested_start_sec=0.0, requested_end_sec=8.0,
            base_segment_snapshot=[], candidate_segment_snapshot=[],
            status=status, submit_billed_at=submit_billed_at,
            seedance_task_id=seedance_task_id,
            claim_owner=claim_owner, claim_expires_at=claim_expires_at,
        )
        db_session.add(attempt)
        db_session.commit()
        if aged:
            self._age(db_session, attempt, 600)
        return attempt

    def test_crashed_finalizer_with_persisted_task_id_is_recovered_not_failed(
        self, db_session,
    ):
        import app.recovery as rec
        from datetime import datetime, timedelta, timezone
        from app.state_machine import RangeAttemptStatus

        dead_claim_expiry = datetime.now(timezone.utc) + timedelta(seconds=3000)
        attempt = self._attempt(
            db_session, RangeAttemptStatus.stitching,
            seedance_task_id="real-billed-task-1",
            claim_owner="dead-worker-token", claim_expires_at=dead_claim_expiry,
        )

        # db_engine is module-scoped (shared across every test in this
        # file), and a "recovered" range attempt is deliberately left in an
        # ACTIVE status (never terminal) — so an EARLIER test's own aged
        # row can still be sitting there and get swept up in the SAME
        # reap call. Assert on THIS attempt specifically (assert_any_call,
        # not assert_called_once_with; a per-row status check, not the
        # aggregate `reaped` dict), never on the call being the only one.
        with patch("app.recovery._enqueue_range_attempt_poll") as mock_enq:
            rec._reap_orphaned_derived_creatives(set())
            mock_enq.assert_any_call(attempt.id, delay_sec=0)

        # NOT failed — recovered. Status is reset to `generating`
        # regardless of which active status it crashed in
        # (poll_range_attempt_once's terminal-state handler always re-runs
        # the whole download->recut->stitch->deliver->promote sequence from
        # scratch, and has no "resume from stitching" edge).
        db_session.refresh(attempt)
        assert attempt.status == RangeAttemptStatus.generating
        assert attempt.error_message is None
        # The dead claim is cleared TRANSACTIONALLY, in this same commit —
        # not left to expire on its own (which would take ~an hour, per
        # pipeline_range_attempt._CLAIM_LEASE_SEC being job-timeout-sized).
        assert attempt.claim_owner is None
        assert attempt.claim_expires_at is None

    def test_ambiguous_billing_window_is_not_recovered_automatically(self, db_session):
        """The OTHER crash window (submit_billed_at set, no task id) keeps
        its existing manual-only handling — recovery must never conflate
        it with the safe-to-resume case just because both involve a dead
        claim."""
        import app.recovery as rec
        from datetime import datetime
        from app.state_machine import RangeAttemptStatus

        attempt = self._attempt(
            db_session, RangeAttemptStatus.generating,
            submit_billed_at=datetime.utcnow(),
            seedance_task_id=None,
        )

        with patch("app.recovery._enqueue_range_attempt_poll") as mock_enq:
            reaped = rec._reap_orphaned_derived_creatives(set())
            # THIS row must never be enqueued for recovery — other, unrelated
            # leftover rows from earlier tests in this module-scoped DB may
            # legitimately be enqueued in the same call, so check by id, not
            # "not called at all".
            called_ids = [c.args[0] for c in mock_enq.call_args_list]
            assert attempt.id not in called_ids

        assert reaped.get("range attempt", 0) >= 1
        db_session.refresh(attempt)
        assert attempt.status == RangeAttemptStatus.failed
        assert "AMBIGUOUS" in attempt.error_message

    def test_live_range_attempt_is_never_recovered_or_reaped(self, db_session):
        """Prevent recovery from racing a genuinely live RQ worker — the
        SAME live_ids liveness gate every other row in this table already
        uses."""
        import app.recovery as rec
        from datetime import datetime, timedelta, timezone
        from app.state_machine import RangeAttemptStatus

        attempt = self._attempt(
            db_session, RangeAttemptStatus.stitching,
            seedance_task_id="real-task-1",
            claim_owner="live-worker-token",
            claim_expires_at=datetime.now(timezone.utc) + timedelta(seconds=3000),
        )

        with patch("app.recovery._enqueue_range_attempt_poll") as mock_enq:
            rec._reap_orphaned_derived_creatives({attempt.id})
            called_ids = [c.args[0] for c in mock_enq.call_args_list]
            assert attempt.id not in called_ids

        db_session.refresh(attempt)
        assert attempt.status == RangeAttemptStatus.stitching
        assert attempt.claim_owner == "live-worker-token"  # untouched


class TestSegmentImageJobResumeSafeRetry:
    """submit_segment_image_job must never create a second kie.ai task for a
    job that already has one — the same resume rule
    pipeline_v2.preprocess_reference_images follows, NOT OverlayJob's
    "just run it again"."""

    @pytest.fixture(autouse=True)
    def _patch_psi_session(self, db_engine, monkeypatch):
        _TestSession = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

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

        import app.pipeline_segment_image as psi
        monkeypatch.setattr(psi, "get_session", _test_get_session)

    def _job_with_segment(self, db_session, *, tmp_path):
        from app.models import RunSegment, RunSegmentImageJob, SegmentDef
        from app.state_machine import SegmentImageStatus

        source = tmp_path / "source.mp4"
        source.write_bytes(b"\x00fake")
        pid = _make_project(db_session)
        project = db_session.get(VideoProject, pid)
        project.source_local_path = str(source)
        project.duration_sec = 30.0
        sd = SegmentDef(
            id=str(uuid.uuid4()), project_id=pid, index=0,
            start_sec=0.0, end_sec=10.0, has_face=True, action="swap",
        )
        db_session.add(sd)
        rid = _make_run(db_session, pid, RunStatus.done)
        run = db_session.get(Run, rid)
        run.reference_image_urls = ["/data/fake/character.jpg"]
        rs = RunSegment(
            id=str(uuid.uuid4()), run_id=rid, segment_def_id=sd.id, index=0,
            status=SegmentStatus.completed,
        )
        db_session.add(rs)
        db_session.commit()
        job = RunSegmentImageJob(
            id=str(uuid.uuid4()), run_id=rid, run_segment_id=rs.id,
            prompt="a prompt", input_paths=["/data/fake/character.jpg"],
            frame_sec=5.0, status=SegmentImageStatus.queued,
        )
        db_session.add(job)
        db_session.commit()
        return job.id

    def test_crash_after_accept_parks_and_never_recreates(
        self, db_session, tmp_path, monkeypatch
    ):
        """AICP-351 §5.1 restart test: kie.ai accepted the image task, the worker
        died before kie_task_id committed. The next delivery of the same job
        PARKS the attempt (job failed with the parked message) and makes no
        second createTask."""
        import app.pipeline_segment_image as psi
        from app import paid_attempts
        from app.models import RunSegmentImageJob
        from app.paid_attempts_models import PaidSubmitAttempt
        from app.state_machine import SegmentImageStatus

        class _Crash(BaseException):
            pass

        job_id = self._job_with_segment(db_session, tmp_path=tmp_path)
        fake_kie = MagicMock()
        fake_kie.create_image_task.side_effect = _Crash()

        def _fake_extract(project, t):
            f = os.path.join(str(tmp_path), "frame.jpg")
            with open(f, "wb") as fh:
                fh.write(b"\x00")
            return f, t

        monkeypatch.setattr(psi, "extract_project_frame", _fake_extract)
        monkeypatch.setattr(psi, "resolve_reference_urls", lambda paths, kie, gdrive=None: list(paths))
        monkeypatch.setattr(paid_attempts, "default_evidence", lambda: (lambda row: None))

        with pytest.raises(_Crash):
            psi.submit_segment_image_job(job_id, kie=fake_kie)
        assert fake_kie.create_image_task.call_count == 1
        assert fake_kie.create_image_task.call_args.kwargs.get("once") is True

        fresh = MagicMock()
        assert psi.submit_segment_image_job(job_id, kie=fresh) is None
        assert fresh.create_image_task.call_count == 0
        db_session.expire_all()
        job = db_session.get(RunSegmentImageJob, job_id)
        assert job.status == SegmentImageStatus.failed and "parked" in job.error_message
        row = db_session.query(PaidSubmitAttempt).filter_by(owner_id=job_id).one()
        # No proof the dead worker is gone: parked but NOT released (counted
        # active, the operator cannot resolve it until proof/attestation).
        assert row.status == "parked" and row.sender_released_at is None

    def test_reaped_before_the_receipt_sends_nothing(self, db_session, tmp_path, monkeypatch):
        """AICP-351 (Terra gate r1): a worker that paused before committing its
        attempt finds the job reaped (failed) and sends NOTHING — so an operator
        regeneration started meanwhile is the only image submit."""
        import app.pipeline_segment_image as psi
        from app.models import RunSegmentImageJob
        from app.paid_attempts_models import PaidSubmitAttempt
        from app.state_machine import SegmentImageStatus

        job_id = self._job_with_segment(db_session, tmp_path=tmp_path)
        fake_kie = MagicMock()

        def _fake_extract(project, t):
            f = os.path.join(str(tmp_path), "frame.jpg")
            with open(f, "wb") as fh:
                fh.write(b"\x00")
            return f, t

        def _reaped_during_the_pause(paths, kie, gdrive=None):
            job = db_session.get(RunSegmentImageJob, job_id)
            job.status = SegmentImageStatus.failed
            job.error_message = "worker was killed"
            db_session.commit()
            return list(paths)

        monkeypatch.setattr(psi, "extract_project_frame", _fake_extract)
        monkeypatch.setattr(psi, "resolve_reference_urls", _reaped_during_the_pause)
        with pytest.raises(Exception):
            psi.submit_segment_image_job(job_id, kie=fake_kie)
        assert fake_kie.create_image_task.call_count == 0
        assert db_session.query(PaidSubmitAttempt).filter_by(owner_id=job_id).count() == 0

    def test_retry_resumes_existing_task_and_creates_no_second_one(
        self, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_segment_image as psi
        from app.models import RunSegmentImageJob

        job_id = self._job_with_segment(db_session, tmp_path=tmp_path)

        fake_kie = MagicMock()
        fake_kie.create_image_task.return_value = "kie-task-1"

        def _fake_extract(project, t):
            f = os.path.join(str(tmp_path), "frame.jpg")
            with open(f, "wb") as fh:
                fh.write(b"\x00")
            return f, t

        monkeypatch.setattr(psi, "extract_project_frame", _fake_extract)
        monkeypatch.setattr(psi, "resolve_reference_urls", lambda paths, kie, gdrive=None: list(paths))

        # First call: no kie_task_id yet -> creates one.
        task_id_1 = psi.submit_segment_image_job(job_id, kie=fake_kie)
        assert task_id_1 == "kie-task-1"
        assert fake_kie.create_image_task.call_count == 1

        # Simulate a worker restart: call submit again on the SAME job, which
        # now has kie_task_id set. Must resume, not recreate (no rebill).
        task_id_2 = psi.submit_segment_image_job(job_id, kie=fake_kie)
        assert task_id_2 == "kie-task-1"
        assert fake_kie.create_image_task.call_count == 1  # still just one

        job = db_session.get(RunSegmentImageJob, job_id)
        assert job.kie_task_id == "kie-task-1"


class TestSegmentImageSubmitFailureIsRecorded:
    """Cursor-gate catch: a submit-phase exception (frame extraction, ref
    resolution, kie.ai) used to propagate with NO try/except, leaving the job
    stuck `queued` with no error_message. That kept api_v2's 409 gates closed
    (patch_run_segment / rerun_segment / regenerate_segment_image / retry_run)
    until the recovery reaper's 300s grace window expired — for a failure
    that in most cases happened in the first second. Mirrors
    overlay_pipeline.process_overlay_job's try/except-then-raise shape."""

    @pytest.fixture(autouse=True)
    def _patch_psi_session(self, db_engine, monkeypatch):
        _TestSession = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

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

        import app.pipeline_segment_image as psi
        monkeypatch.setattr(psi, "get_session", _test_get_session)

    def _job_with_segment(self, db_session, *, tmp_path):
        from app.models import RunSegment, RunSegmentImageJob, SegmentDef
        from app.state_machine import SegmentImageStatus

        source = tmp_path / "source.mp4"
        source.write_bytes(b"\x00fake")
        pid = _make_project(db_session)
        project = db_session.get(VideoProject, pid)
        project.source_local_path = str(source)
        project.duration_sec = 30.0
        sd = SegmentDef(
            id=str(uuid.uuid4()), project_id=pid, index=0,
            start_sec=0.0, end_sec=10.0, has_face=True, action="swap",
        )
        db_session.add(sd)
        rid = _make_run(db_session, pid, RunStatus.done)
        run = db_session.get(Run, rid)
        run.reference_image_urls = ["/data/fake/character.jpg"]
        rs = RunSegment(
            id=str(uuid.uuid4()), run_id=rid, segment_def_id=sd.id, index=0,
            status=SegmentStatus.completed,
        )
        db_session.add(rs)
        db_session.commit()
        job = RunSegmentImageJob(
            id=str(uuid.uuid4()), run_id=rid, run_segment_id=rs.id,
            prompt="a prompt", input_paths=["/data/fake/character.jpg"],
            frame_sec=5.0, status=SegmentImageStatus.queued,
        )
        db_session.add(job)
        db_session.commit()
        return job.id

    def test_frame_extraction_failure_marks_job_failed(
        self, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_segment_image as psi
        from app.models import RunSegmentImageJob
        from app.state_machine import SegmentImageStatus

        job_id = self._job_with_segment(db_session, tmp_path=tmp_path)

        def _boom(project, t):
            raise FileNotFoundError("source video not on disk")

        monkeypatch.setattr(psi, "extract_project_frame", _boom)

        with pytest.raises(FileNotFoundError):
            psi.submit_segment_image_job(job_id, kie=MagicMock())

        job = db_session.get(RunSegmentImageJob, job_id)
        assert job.status == SegmentImageStatus.failed
        assert "source video not on disk" in job.error_message

    def test_create_image_task_failure_marks_job_failed(
        self, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_segment_image as psi
        from app.models import RunSegmentImageJob
        from app.state_machine import SegmentImageStatus

        job_id = self._job_with_segment(db_session, tmp_path=tmp_path)

        def _fake_extract(project, t):
            f = os.path.join(str(tmp_path), "frame.jpg")
            with open(f, "wb") as fh:
                fh.write(b"\x00")
            return f, t

        monkeypatch.setattr(psi, "extract_project_frame", _fake_extract)
        monkeypatch.setattr(
            psi, "resolve_reference_urls", lambda paths, kie, gdrive=None: list(paths)
        )
        fake_kie = MagicMock()
        fake_kie.create_image_task.side_effect = RuntimeError("kie.ai 500")

        with pytest.raises(RuntimeError):
            psi.submit_segment_image_job(job_id, kie=fake_kie)

        job = db_session.get(RunSegmentImageJob, job_id)
        assert job.status == SegmentImageStatus.failed
        assert "kie.ai 500" in job.error_message


# ---------------------------------------------------------------------------
# 8. Stale per-run lock must not survive recovery
# ---------------------------------------------------------------------------




def _status(db_engine, rid):
    Session = sessionmaker(bind=db_engine)
    with Session() as s:
        return s.get(Run, rid).status


def _set_status(db_engine, rid, status):
    Session = sessionmaker(bind=db_engine)
    with Session() as s:
        s.get(Run, rid).status = status
        s.commit()


def _get(conn, key):
    v = conn.get(key)
    return v.decode() if isinstance(v, bytes) else v


class TestStaleRunLockTakenOver:
    """Recovery was defeated by its own lock (prod 2026-08-11, run b6265eec).

    The per-run lock is released by token, so a worker killed mid-run never
    releases it and it lingers for PROCESS_JOB_TIMEOUT + 300 (3h by default).
    Recovery must still get the run going again — since AICP-360 by taking the
    stale lock over with a compare-and-set and HANDING it to the re-enqueued
    job, never by deleting it blindly.
    """

    def test_dead_holders_lock_is_replaced_and_handed_to_the_new_job(self, db_session, db_engine):
        import app.recovery as rec
        from app import tasks
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        key = run_lock_key(rid)
        conn.set(key, "dead-holder", ex=10800)
        assert tasks._refresh_lock(conn, key, "dead-holder")  # it had refreshed once

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        calls = [c for c in mock_enqueue.call_args_list if c.args[0] == rid]
        assert len(calls) == 1
        handed = calls[0].kwargs["lock_token"]
        assert handed and handed != "dead-holder"
        assert _get(conn, key) == handed           # the new job owns the lock
        assert conn.ttl(key) > 10000              # with a full lifecycle TTL
        assert _status(db_engine, rid) == RunStatus.queued

    def test_lock_is_owned_by_the_handoff_token_at_enqueue_time(self, db_session):
        """No unlocked moment between the claim and the recovered job."""
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        conn.set(run_lock_key(rid), "dead-holder")
        seen = {}

        def _enqueue(run_id, *, lock_token=None):
            if run_id == rid:
                seen["lock"] = _get(conn, run_lock_key(rid))
                seen["token"] = lock_token

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run", side_effect=_enqueue):
            rec._reconcile_orphaned_runs_impl(conn)

        assert seen["lock"] == seen["token"] is not None

    def test_legacy_lock_without_a_generation_is_recovered(self, db_session):
        """A lock written before this deploy has no generation key."""
        import app.recovery as rec
        from app.tasks import run_lock_key, run_lock_gen_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        conn.set(run_lock_key(rid), "old-code-holder")
        assert conn.get(run_lock_gen_key(run_lock_key(rid))) is None

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert rid in [c.args[0] for c in mock_enqueue.call_args_list]
        assert _get(conn, run_lock_key(rid)) != "old-code-holder"

    def test_handoff_job_runs_and_the_dead_chain_cannot_resume(self, db_session, monkeypatch):
        """End to end with the real task wrappers: the recovered submit refreshes
        the handed-over lock and proceeds; a poll job of the dead chain (still
        carrying the old token) aborts before polling anything."""
        import app.recovery as rec
        from app import tasks
        from app.tasks import run_lock_key
        import app.pipeline_v2 as pipeline_v2
        import app.gdrive_client as gdrive_client

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        conn.set(run_lock_key(rid), "dead-holder")

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)
        call = [c for c in mock_enqueue.call_args_list if c.args[0] == rid][0]

        monkeypatch.setattr(tasks, "_redis_conn", conn)
        monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: MagicMock()})
        monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
        submits, polls = [], []
        monkeypatch.setattr(pipeline_v2, "submit_run", lambda *a, **k: submits.append(a) or None)
        monkeypatch.setattr(pipeline_v2, "poll_run_once",
                            lambda *a, **k: polls.append(a) or {})

        tasks.run_poll_run(rid, lock_key=run_lock_key(rid), lock_token="dead-holder")
        assert polls == []
        tasks.run_submit_run(rid, lock_key=run_lock_key(rid),
                             lock_token=call.kwargs["lock_token"])
        assert len(submits) == 1

    def test_live_run_keeps_its_lock(self, db_session):
        """A run with a live job is being worked on RIGHT NOW."""
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        conn.set(run_lock_key(rid), "live-holder")

        with patch.object(rec, "_live_run_ids", return_value={rid}), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert _get(conn, run_lock_key(rid)) == "live-holder"
        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]

    def test_unreadable_lock_state_touches_nothing(self, db_session, db_engine):
        """Fail closed: a lock we cannot read is not "no lock"."""
        import app.recovery as rec

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "read_run_lock_states", side_effect=RuntimeError("redis down")), \
             patch.object(rec, "_repoll_generating_segments") as mock_repoll, \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]
        mock_repoll.assert_not_called()
        assert _status(db_engine, rid) == RunStatus.processing

    def test_failed_enqueue_keeps_the_claim_and_next_tick_recovers(self, db_session, db_engine):
        """An enqueue error is ambiguous (the job may be queued and the reply
        lost), so the claim is NOT released beside it; the next tick takes the
        unused claim over with a NEW token, which a phantom job could not
        refresh."""
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        conn.set(run_lock_key(rid), "dead-holder")

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run",
                          side_effect=RuntimeError("enqueue")) as failing:
            rec._reconcile_orphaned_runs_impl(conn)
        first = [c for c in failing.call_args_list if c.args[0] == rid][0].kwargs["lock_token"]
        assert _get(conn, run_lock_key(rid)) == first
        assert _status(db_engine, rid) == RunStatus.queued

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)
        second = [c for c in mock_enqueue.call_args_list if c.args[0] == rid][0].kwargs["lock_token"]
        assert second != first and _get(conn, run_lock_key(rid)) == second

    def test_claim_left_by_a_dead_reaper_is_recovered_next_tick(self, db_session):
        """A reaper killed after its claim and before its enqueue leaves the
        claim behind as an ordinary stale lock — no 3h stall."""
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.queued)
        conn = _fresh_redis()

        class _ReaperKilled(BaseException):
            pass

        def _killed_on_ours(run_id, redis_conn, **kw):
            if run_id == rid:
                raise _ReaperKilled()

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments", side_effect=_killed_on_ours), \
             patch.object(rec, "_release_lock"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            with pytest.raises(_ReaperKilled):
                rec._reconcile_orphaned_runs_impl(conn)
        leaked = _get(conn, run_lock_key(rid))
        assert leaked  # the dead reaper's claim

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)
        call = [c for c in mock_enqueue.call_args_list if c.args[0] == rid]
        assert len(call) == 1 and call[0].kwargs["lock_token"] != leaked

    def test_preprocessing_orphan_gets_the_lock_handed_to_preprocess(self, db_session, db_engine):
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.preprocessing)
        conn = _fresh_redis()

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_proc, \
             patch.object(rec, "enqueue_preprocess_run") as mock_pp:
            rec._reconcile_orphaned_runs_impl(conn)

        assert rid not in [c.args[0] for c in mock_proc.call_args_list]
        call = [c for c in mock_pp.call_args_list if c.args[0] == rid][0]
        assert _get(conn, run_lock_key(rid)) == call.kwargs["lock_token"]
        assert _status(db_engine, rid) == RunStatus.preprocessing

    def test_tasks_and_recovery_agree_on_the_key(self):
        """If tasks.py changes the key format, recovery must follow it."""
        import inspect
        import app.tasks as tasks_mod

        src = inspect.getsource(tasks_mod)
        assert 'lock_key = f"reskin:run:lock:{run_id}"' not in src
        assert src.count("run_lock_key(run_id)") >= 2


# ---------------------------------------------------------------------------
# 9. AICP-360: a run requeued between the liveness snapshot and the run scan
# ---------------------------------------------------------------------------


class TestRequeuedAfterSnapshot:
    """Deterministic replays of the interleavings of AICP-360.

    ``_live_run_ids`` is called twice per pass: the snapshot (first) and the
    fail-closed rescan right before the claims (``strict=True``). Each test
    scripts what the rest of the system does between those points, with the
    REAL lock scripts on a Lua-capable fake Redis.
    """

    @staticmethod
    def _liveness(script):
        """A _live_run_ids stand-in: call i runs script[i](), returns its set."""
        calls = []

        def _live(redis_conn, queue_names=None, *, strict=False):
            i = len(calls)
            calls.append(strict)
            return script[i]() if i < len(script) else set()

        _live.calls = calls
        return _live

    def test_requeued_and_started_job_keeps_its_lock(self, db_session, db_engine):
        """The reported race: the run is requeued after the snapshot and its job
        J starts (acquires the lock, moves the run to processing) before the
        run scan. The old pass deleted J's lock and enqueued a second submit."""
        import app.recovery as rec
        from app.tasks import run_lock_key, _RUN_LOCK_TTL_SEC

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        conn = _fresh_redis()

        def _api_requeue_and_worker_starts(_live_ids):
            _set_status(db_engine, rid, RunStatus.queued)            # API commit
            conn.set(run_lock_key(rid), "J", nx=True, ex=_RUN_LOCK_TTL_SEC)  # J acquires
            _set_status(db_engine, rid, RunStatus.processing)        # submit_run starts

        live = self._liveness([lambda: set(), lambda: {rid}])  # J now in started
        with patch.object(rec, "_live_run_ids", side_effect=live), \
             patch.object(rec, "_reap_orphaned_derived_creatives",
                          side_effect=_api_requeue_and_worker_starts), \
             patch.object(rec, "_repoll_generating_segments") as mock_repoll, \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert live.calls == [False, True]
        assert _get(conn, run_lock_key(rid)) == "J"
        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]
        assert rid not in [c.args[0] for c in mock_repoll.call_args_list]
        assert _status(db_engine, rid) == RunStatus.processing

    def test_job_that_acquires_after_the_lock_read_is_not_stolen(self, db_session, db_engine):
        """J is in RQ's dequeue gap (popped, not yet in StartedJobRegistry) during
        the rescan, so the rescan misses it — but it acquires the lock before
        the claim, so the compare-and-set on the token fails."""
        import app.recovery as rec
        from app.tasks import run_lock_key, _RUN_LOCK_TTL_SEC

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.queued)
        conn = _fresh_redis()

        def _gap_then_acquire():
            conn.set(run_lock_key(rid), "J", nx=True, ex=_RUN_LOCK_TTL_SEC)
            _set_status(db_engine, rid, RunStatus.processing)
            return set()

        live = self._liveness([lambda: set(), _gap_then_acquire])
        with patch.object(rec, "_live_run_ids", side_effect=live), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert _get(conn, run_lock_key(rid)) == "J"
        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]
        assert _status(db_engine, rid) == RunStatus.processing

    def test_successor_that_refreshes_after_the_lock_read_is_not_stolen(
        self, db_session, db_engine
    ):
        """A poll/finalize job inheriting the chain's lock keeps the SAME token;
        only the generation shows it started working. Without the generation
        this claim would succeed under a running poll job."""
        import app.recovery as rec
        from app import tasks
        from app.tasks import run_lock_key, run_lock_gen_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        key = run_lock_key(rid)
        conn.set(key, "chain", ex=10800)

        gen_before = _get(conn, run_lock_gen_key(key))

        def _successor_starts():
            assert tasks._refresh_lock(conn, key, "chain")
            return set()

        live = self._liveness([lambda: set(), _successor_starts])
        with patch.object(rec, "_live_run_ids", side_effect=live), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert _get(conn, key) == "chain"
        assert _get(conn, run_lock_gen_key(key)) not in (None, gen_before)
        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]
        assert _status(db_engine, rid) == RunStatus.processing

    def test_requeued_job_that_runs_during_recovery_skips_and_one_owner_remains(
        self, db_session, monkeypatch
    ):
        """The requeue lands after the claim: J runs while the reaper is in its
        kie.ai pre-poll. J finds the lock taken and skips; the reaper's handoff
        job is the only one that submits."""
        import app.recovery as rec
        from app import tasks
        from app.tasks import run_lock_key
        import app.pipeline_v2 as pipeline_v2
        import app.gdrive_client as gdrive_client

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        monkeypatch.setattr(tasks, "_redis_conn", conn)
        monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: MagicMock()})
        monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
        submits = []
        monkeypatch.setattr(pipeline_v2, "submit_run",
                            lambda run_id, **k: submits.append(run_id) or None)

        def _j_runs_now(run_id, redis_conn, **kw):
            if run_id == rid:
                tasks.run_submit_run(rid)   # the requeued job, no token

        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments", side_effect=_j_runs_now), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert submits == []   # J skipped: the reaper owned the run
        call = [c for c in mock_enqueue.call_args_list if c.args[0] == rid][0]
        tasks.run_submit_run(rid, lock_key=run_lock_key(rid),
                             lock_token=call.kwargs["lock_token"])
        assert submits == [rid]

    def test_run_that_finished_before_the_claim_is_left_alone(self, db_session, db_engine):
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()

        def _finishes():
            _set_status(db_engine, rid, RunStatus.done)
            return set()

        live = self._liveness([lambda: set(), _finishes])
        with patch.object(rec, "_live_run_ids", side_effect=live), \
             patch.object(rec, "_repoll_generating_segments") as mock_repoll, \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]
        assert rid not in [c.args[0] for c in mock_repoll.call_args_list]
        assert _status(db_engine, rid) == RunStatus.done
        assert conn.get(run_lock_key(rid)) is None   # our claim was released

    def test_rescan_failure_touches_nothing(self, db_session, db_engine):
        import app.recovery as rec

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()

        def _boom():
            raise RuntimeError("Worker.all failed")

        live = self._liveness([lambda: set(), _boom])
        with patch.object(rec, "_live_run_ids", side_effect=live), \
             patch.object(rec, "_repoll_generating_segments") as mock_repoll, \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]
        mock_repoll.assert_not_called()
        assert _status(db_engine, rid) == RunStatus.processing

    def test_rescan_runs_only_when_there_are_candidates(self, db_session):
        import app.recovery as rec

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()

        Session = sessionmaker(bind=db_session.get_bind())
        with Session() as s:
            active = [r.id for r in s.query(Run).filter(Run.status.in_(
                [st.value for st in rec._ACTIVE_RUN_STATUSES]))]
        live = self._liveness([lambda: set(active)])
        with patch.object(rec, "_live_run_ids", side_effect=live), \
             patch.object(rec, "enqueue_process_run"):
            rec._reconcile_orphaned_runs_impl(conn)
        assert live.calls == [False]
        assert rid in active


class TestReconcilePassLock:
    def test_a_pass_cannot_delete_the_lock_of_the_pass_that_took_over(self):
        import app.recovery as rec

        conn = _fresh_redis()

        def _expired_and_taken_over(redis_conn, *, still_own_pass):
            conn.set(rec._RECONCILE_LOCK_KEY, "pass-B", ex=120)
            assert still_own_pass() is False

        with patch.object(rec, "_reconcile_orphaned_runs_impl",
                          side_effect=_expired_and_taken_over):
            rec.reconcile_orphaned_runs(conn)

        assert _get(conn, rec._RECONCILE_LOCK_KEY) == "pass-B"

    def test_own_pass_lock_is_rearmed_and_released(self):
        import app.recovery as rec

        conn = _fresh_redis()
        seen = []

        def _impl(redis_conn, *, still_own_pass):
            seen.append(still_own_pass())

        with patch.object(rec, "_reconcile_orphaned_runs_impl", side_effect=_impl):
            rec.reconcile_orphaned_runs(conn)
        assert seen == [True]
        assert conn.get(rec._RECONCILE_LOCK_KEY) is None

    def test_lost_pass_lock_stops_the_run_pass(self, db_session):
        import app.recovery as rec

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        with patch.object(rec, "_live_run_ids", return_value=set()), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn, still_own_pass=lambda: False)
        mock_enqueue.assert_not_called()


class TestStrictLiveness:
    def test_worker_list_failure_raises_in_strict_mode(self):
        import app.recovery as rec

        with patch.object(rec.Worker, "all", side_effect=RuntimeError("down")):
            assert rec._live_worker_names(MagicMock()) == set()
            with pytest.raises(RuntimeError):
                rec._live_worker_names(MagicMock(), strict=True)

    def test_unreadable_job_raises_in_strict_mode_but_a_gone_job_does_not(self):
        import app.recovery as rec
        from rq.exceptions import NoSuchJobError

        fake_queue = MagicMock()
        fake_queue.job_ids = ["gone", "unreadable"]
        empty_reg = MagicMock()
        empty_reg.get_job_ids.return_value = []

        def _fetch(jid, connection=None):
            if jid == "gone":
                raise NoSuchJobError(jid)
            raise RuntimeError("redis hiccup")

        with patch.object(rec, "_live_worker_names", return_value=set()), \
             patch.object(rec, "Queue", return_value=fake_queue), \
             patch.object(rec, "ScheduledJobRegistry", return_value=empty_reg), \
             patch.object(rec, "StartedJobRegistry", return_value=empty_reg), \
             patch.object(rec.Job, "fetch_many", side_effect=RuntimeError("batch")), \
             patch.object(rec.Job, "fetch", side_effect=_fetch):
            assert rec._live_run_ids(MagicMock()) == set()
            with pytest.raises(RuntimeError):
                rec._live_run_ids(MagicMock(), strict=True)

        def _fetch_gone(jid, connection=None):
            raise NoSuchJobError(jid)

        fake_queue.job_ids = ["gone"]
        with patch.object(rec, "_live_worker_names", return_value=set()), \
             patch.object(rec, "Queue", return_value=fake_queue), \
             patch.object(rec, "ScheduledJobRegistry", return_value=empty_reg), \
             patch.object(rec, "StartedJobRegistry", return_value=empty_reg), \
             patch.object(rec.Job, "fetch_many", side_effect=RuntimeError("batch")), \
             patch.object(rec.Job, "fetch", side_effect=_fetch_gone):
            assert rec._live_run_ids(MagicMock(), strict=True) == set()


class TestRealRqLiveness:
    """The rescan against REAL rq structures on fake Redis: a job enqueued by
    the normal API helper after the snapshot is seen and the run is left alone."""

    def test_job_enqueued_after_the_snapshot_is_seen_by_the_rescan(
        self, db_session, db_engine, monkeypatch
    ):
        import app.recovery as rec
        from app import tasks
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.done)
        conn = _fresh_redis()
        monkeypatch.setattr(tasks, "_redis_conn", conn)
        monkeypatch.setattr(tasks, "_queues", {})

        def _api_requeue(_live_ids):
            _set_status(db_engine, rid, RunStatus.queued)
            tasks.enqueue_process_run(rid)

        with patch.object(rec, "_reap_orphaned_derived_creatives", side_effect=_api_requeue), \
             patch.object(rec, "_reconcile_reframe_jobs"), \
             patch.object(rec, "_repoll_generating_segments"), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            rec._reconcile_orphaned_runs_impl(conn)

        assert rid not in [c.args[0] for c in mock_enqueue.call_args_list]
        assert conn.get(run_lock_key(rid)) is None
        assert _status(db_engine, rid) == RunStatus.queued


class TestLockProtocol:
    """tasks.py side of AICP-360: generation bump, the claim CAS, handoffs."""

    def test_refresh_bumps_the_generation_only_for_the_owner(self):
        from app import tasks
        from app.tasks import run_lock_gen_key

        conn = _fresh_redis()
        key = "reskin:run:lock:r1"
        conn.set(key, "tok")
        assert tasks._refresh_lock(conn, key, "tok")
        g1 = _get(conn, run_lock_gen_key(key))
        assert tasks._refresh_lock(conn, key, "tok")
        g2 = _get(conn, run_lock_gen_key(key))
        assert g1 and g2 and g1 != g2                 # a fresh nonce every time
        assert conn.ttl(run_lock_gen_key(key)) == conn.ttl(key) > 0
        assert not tasks._refresh_lock(conn, key, "other")
        assert _get(conn, run_lock_gen_key(key)) == g2

    def test_an_expired_generation_cannot_come_back_to_the_value_read(self):
        """ABA: a counter that expired and was re-INCRed could return to the
        value the reaper read; a nonce cannot."""
        from app import tasks
        from app.tasks import claim_stale_run_lock, read_run_lock_states, run_lock_key, run_lock_gen_key

        conn = _fresh_redis()
        key = run_lock_key("r-aba")
        conn.set(key, "J")
        conn.set(run_lock_gen_key(key), "1")          # leftover of a past lifecycle
        observed = read_run_lock_states(conn, ["r-aba"])["r-aba"]
        assert observed == ("J", "1")
        conn.delete(run_lock_gen_key(key))            # it expires...
        assert tasks._refresh_lock(conn, key, "J")    # ...and J's successor starts
        assert claim_stale_run_lock(conn, "r-aba", observed) is None
        assert _get(conn, key) == "J"

    def test_claim_is_a_compare_and_set_on_token_and_generation(self):
        from app import tasks
        from app.tasks import claim_stale_run_lock, read_run_lock_states, run_lock_key

        conn = _fresh_redis()
        key = run_lock_key("r2")
        assert read_run_lock_states(conn, ["r2"]) == {"r2": (None, None)}

        # absent → absent: claimable
        claim = claim_stale_run_lock(conn, "r2", (None, None))
        assert claim and _get(conn, key) == claim

        observed = read_run_lock_states(conn, ["r2"])
        assert observed["r2"][0] == claim and observed["r2"][1]  # claim stamps a nonce
        assert tasks._refresh_lock(conn, key, claim)           # generation moves
        assert claim_stale_run_lock(conn, "r2", observed["r2"]) is None
        assert _get(conn, key) == claim

        observed = read_run_lock_states(conn, ["r2"])
        conn.set(key, "someone-else")                           # token moves
        assert claim_stale_run_lock(conn, "r2", observed["r2"]) is None
        assert _get(conn, key) == "someone-else"

        observed = read_run_lock_states(conn, ["r2"])           # unchanged → claim
        second = claim_stale_run_lock(conn, "r2", observed["r2"])
        assert second and _get(conn, key) == second

        observed = (None, read_run_lock_states(conn, ["r2"])["r2"][1])
        assert claim_stale_run_lock(conn, "r2", observed) is None  # present ≠ absent
        conn.set(key, "")                                           # "" is not absent
        observed = (None, read_run_lock_states(conn, ["r2"])["r2"][1])
        assert claim_stale_run_lock(conn, "r2", observed) is None

    def test_enqueue_helpers_hand_the_lock_over_only_when_asked(self, monkeypatch):
        from app import tasks

        q = MagicMock()
        q.enqueue.return_value = MagicMock(id="j")
        monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: q})

        tasks.enqueue_process_run("r3")
        assert "lock_token" not in q.enqueue.call_args.kwargs
        tasks.enqueue_process_run("r3", lock_token="claim")
        kw = q.enqueue.call_args.kwargs
        assert q.enqueue.call_args.args == ("app.tasks.run_submit_run", "r3")
        assert kw["lock_key"] == tasks.run_lock_key("r3") and kw["lock_token"] == "claim"

        tasks.enqueue_preprocess_run("r3", lock_token="claim")
        kw = q.enqueue.call_args.kwargs
        assert q.enqueue.call_args.args == ("app.tasks.run_preprocess_refs", "r3")
        assert kw["lock_key"] == tasks.run_lock_key("r3") and kw["lock_token"] == "claim"

    def test_preprocess_handoff_refreshes_instead_of_acquiring(self, monkeypatch):
        from app import tasks
        import app.pipeline_v2 as pipeline_v2
        import app.gdrive_client as gdrive_client

        conn = _fresh_redis()
        monkeypatch.setattr(tasks, "_redis_conn", conn)
        monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
        calls = []
        monkeypatch.setattr(pipeline_v2, "preprocess_reference_images",
                            lambda run_id, **k: calls.append(run_id) or {})
        monkeypatch.setattr(tasks, "_finish_preprocessing", lambda *a, **k: None)
        key = tasks.run_lock_key("r4")

        tasks.run_preprocess_refs("r4", lock_key=key, lock_token="lost")
        assert calls == []                                       # not ours → abort

        conn.set(key, "claim")
        tasks.run_preprocess_refs("r4", lock_key=key, lock_token="claim")
        assert calls == ["r4"]
        assert _get(conn, tasks.run_lock_gen_key(key))   # the handoff refreshed

        calls.clear()
        tasks.run_preprocess_refs("r4")                          # plain job: lock held
        assert calls == []

    def test_preprocess_poll_failure_is_written_before_the_lock_is_released(
        self, monkeypatch
    ):
        from app import tasks
        import app.pipeline_v2 as pipeline_v2

        conn = _fresh_redis()
        monkeypatch.setattr(tasks, "_redis_conn", conn)
        key = tasks.run_lock_key("r5")
        conn.set(key, "tok")
        conn.set(tasks._preprocess_poll_state_key("r5"), json.dumps({"t": {}}))
        monkeypatch.setattr(pipeline_v2, "poll_preprocess_once",
                            lambda run_id, pending: {"failed": [{"reason": "x"}],
                                                     "pending": {}})
        monkeypatch.setattr(tasks, "_should_park_failed_preprocess", lambda rid: False)
        held = []
        monkeypatch.setattr(tasks, "_fail_preprocessing_run",
                            lambda rid, error_message=None: held.append(_get(conn, key)))

        tasks.run_preprocess_poll("r5", lock_key=key, lock_token="tok")

        assert held == ["tok"]
        assert conn.get(key) is None

    def test_preprocess_poll_exception_fails_the_run_before_releasing(self, monkeypatch):
        from app import tasks
        import app.pipeline_v2 as pipeline_v2

        conn = _fresh_redis()
        monkeypatch.setattr(tasks, "_redis_conn", conn)
        key = tasks.run_lock_key("r6")
        conn.set(key, "tok")
        conn.set(tasks._preprocess_poll_state_key("r6"), json.dumps({"t": {}}))

        def _boom(run_id, pending):
            raise RuntimeError("kie down")

        monkeypatch.setattr(pipeline_v2, "poll_preprocess_once", _boom)
        held = []
        monkeypatch.setattr(tasks, "_fail_preprocessing_run",
                            lambda rid, error_message=None: held.append(_get(conn, key)))

        with pytest.raises(RuntimeError):
            tasks.run_preprocess_poll("r6", lock_key=key, lock_token="tok")
        assert held == ["tok"]
        assert conn.get(key) is None

    def test_preprocess_start_failure_is_written_before_the_lock_is_released(
        self, monkeypatch
    ):
        from app import tasks
        import app.pipeline_v2 as pipeline_v2
        import app.gdrive_client as gdrive_client

        conn = _fresh_redis()
        monkeypatch.setattr(tasks, "_redis_conn", conn)
        monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())

        def _boom(run_id, **k):
            raise RuntimeError("cannot create task")

        monkeypatch.setattr(pipeline_v2, "preprocess_reference_images", _boom)
        monkeypatch.setattr(tasks, "_should_park_failed_preprocess", lambda rid: False)
        key = tasks.run_lock_key("r7")
        held = []
        monkeypatch.setattr(tasks, "_fail_preprocessing_run",
                            lambda rid, error_message=None: held.append(_get(conn, key)))

        with pytest.raises(RuntimeError):
            tasks.run_preprocess_refs("r7")
        assert len(held) == 1 and held[0]
        assert conn.get(key) is None


class TestStrictJobDecoding:
    def test_undecodable_args_raise_only_in_strict_mode(self):
        import app.recovery as rec

        class _Bad:
            @property
            def args(self):
                raise ValueError("cannot unpickle")

        assert rec._run_id_from_job(_Bad()) is None
        with pytest.raises(ValueError):
            rec._run_id_from_job(_Bad(), strict=True)


class TestPassLockHeartbeat:
    def test_heartbeat_keeps_the_pass_lock_alive_during_a_long_pass(self, monkeypatch):
        import time as _time
        import app.recovery as rec

        conn = _fresh_redis()
        monkeypatch.setattr(rec, "_RECONCILE_LOCK_HEARTBEAT_SEC", 0.05)
        ttls = []

        def _slow(redis_conn, *, still_own_pass):
            conn.expire(rec._RECONCILE_LOCK_KEY, 1)     # nearly lapsed
            _time.sleep(0.3)
            ttls.append(conn.ttl(rec._RECONCILE_LOCK_KEY))

        with patch.object(rec, "_reconcile_orphaned_runs_impl", side_effect=_slow):
            rec.reconcile_orphaned_runs(conn)
        assert ttls and ttls[0] > 1                     # re-armed by the heartbeat
        assert conn.get(rec._RECONCILE_LOCK_KEY) is None


class TestOwnershipCheckpoints:
    """Gate R1 (Terra): a pass that lost its lease must stop mid-recovery."""

    def _success_kie(self, task_id):
        fake_kie = MagicMock()
        fake_kie.get_task.return_value = {
            "state": "success",
            "resultJson": json.dumps({"resultUrls": [f"https://x/{task_id}.mp4"]}),
        }
        fake_kie.download_result.side_effect = lambda url, dst: (
            os.makedirs(os.path.dirname(dst), exist_ok=True),
            open(dst, "wb").write(b"v"),
        )
        return fake_kie

    @pytest.mark.parametrize("lose_after", [0, 1, 2])
    def test_repoll_stops_before_the_next_call_or_write(self, db_session, db_engine, lose_after):
        """Checkpoints: before get_task (0), before the download (1), before the
        segment write (2)."""
        import app.recovery as rec

        pid = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, rs_id = _make_run_with_generating_segment(db_session, pid, task_id)
        fake_kie = self._success_kie(task_id)
        checks = []

        def _owner():
            checks.append(1)
            return len(checks) <= lose_after

        with patch("app.recovery.KieClient", return_value=fake_kie):
            with pytest.raises(rec._OwnershipLost):
                rec._repoll_generating_segments(run_id, MagicMock(), still_owner=_owner)

        assert fake_kie.get_task.call_count == (0 if lose_after == 0 else 1)
        assert fake_kie.download_result.call_count == (1 if lose_after == 2 else 0)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            assert s.get(RunSegment, rs_id).status == SegmentStatus.generating

    def test_lost_lease_during_the_prepoll_means_no_enqueue_and_no_write(
        self, db_session, db_engine
    ):
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, rs_id = _make_run_with_generating_segment(db_session, pid, task_id)
        conn = _fresh_redis()
        fake_kie = self._success_kie(task_id)
        lease = {"ok": True}

        def _get_task_then_lease_lapses(tid):
            lease["ok"] = False          # heartbeat failed; another pass may start
            return fake_kie.get_task.return_value

        fake_kie.get_task.side_effect = _get_task_then_lease_lapses
        with patch("app.recovery.KieClient", return_value=fake_kie), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            out = rec._recover_orphaned_run(
                conn, run_id, (None, None), still_own_pass=lambda: lease["ok"],
            )

        assert out is None
        mock_enqueue.assert_not_called()
        fake_kie.download_result.assert_not_called()
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            assert s.get(RunSegment, rs_id).status == SegmentStatus.generating
        assert conn.get(run_lock_key(run_id)) is None   # our unused claim released

    def test_claim_taken_over_mid_prepoll_stops_this_pass(self, db_session):
        import app.recovery as rec
        from app.tasks import run_lock_key

        pid = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, _ = _make_run_with_generating_segment(db_session, pid, task_id)
        conn = _fresh_redis()
        fake_kie = self._success_kie(task_id)

        def _other_pass_claims(tid):
            conn.set(run_lock_key(run_id), "other-pass")
            return fake_kie.get_task.return_value

        fake_kie.get_task.side_effect = _other_pass_claims
        with patch("app.recovery.KieClient", return_value=fake_kie), \
             patch.object(rec, "enqueue_process_run") as mock_enqueue:
            assert rec._recover_orphaned_run(conn, run_id, (None, None)) is None
        mock_enqueue.assert_not_called()
        fake_kie.download_result.assert_not_called()
        assert _get(conn, run_lock_key(run_id)) == "other-pass"

    def test_transient_heartbeat_error_is_retried_not_final(self, monkeypatch):
        import time as _time
        import app.recovery as rec

        conn = _fresh_redis()
        monkeypatch.setattr(rec, "_RECONCILE_LOCK_HEARTBEAT_SEC", 0.05)
        real_eval = conn.eval
        fails = {"n": 2}

        def _flaky_eval(*a, **k):
            if a and a[0] == rec._REFRESH_PASS_LOCK_LUA and fails["n"] > 0:
                fails["n"] -= 1
                raise ConnectionError("blip")
            return real_eval(*a, **k)

        monkeypatch.setattr(conn, "eval", _flaky_eval)
        seen = []

        def _impl(redis_conn, *, still_own_pass):
            _time.sleep(0.4)            # the heartbeat hits both blips, then recovers
            seen.append(still_own_pass())

        with patch.object(rec, "_reconcile_orphaned_runs_impl", side_effect=_impl):
            rec.reconcile_orphaned_runs(conn)
        assert fails["n"] == 0 and seen == [True]


class TestGateR2:
    def test_ownership_lost_while_waiting_for_the_row_lock_means_no_reset(
        self, db_session, db_engine
    ):
        """The pass lost the run between the claim and the Run-row lock: the
        check under the row lock must stop it before it resets a live run."""
        import app.recovery as rec

        pid = _make_project(db_session)
        rid = _make_run(db_session, pid, RunStatus.processing)
        conn = _fresh_redis()
        lease = iter([False])

        with patch.object(rec, "enqueue_process_run") as mock_enqueue, \
             patch.object(rec, "_repoll_generating_segments") as mock_repoll:
            out = rec._recover_orphaned_run(
                conn, rid, (None, None), still_own_pass=lambda: next(lease, False),
            )
        assert out is None
        mock_enqueue.assert_not_called()
        mock_repoll.assert_not_called()
        assert _status(db_engine, rid) == RunStatus.processing

    def test_prepoll_download_never_touches_the_shared_result_path_until_owned(
        self, db_session, db_engine
    ):
        import app.recovery as rec

        pid = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, rs_id = _make_run_with_generating_segment(db_session, pid, task_id)
        fake_kie = MagicMock()
        fake_kie.get_task.return_value = {
            "state": "success",
            "resultJson": json.dumps({"resultUrls": ["https://x/r.mp4"]}),
        }
        targets = []

        def _download(url, dst):
            targets.append(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(b"ours")

        fake_kie.download_result.side_effect = _download
        owner = iter([True, True, False])   # lost after the download

        with patch("app.recovery.KieClient", return_value=fake_kie):
            with pytest.raises(rec._OwnershipLost):
                rec._repoll_generating_segments(
                    run_id, MagicMock(), still_owner=lambda: next(owner, False))

        assert len(targets) == 1 and ".recovery-" in targets[0]
        assert not os.path.exists(targets[0])                     # discarded
        assert not os.path.exists(targets[0].split(".recovery-")[0])  # never placed

        # Owned throughout: the private file is moved into place.
        with patch("app.recovery.KieClient", return_value=fake_kie):
            rec._repoll_generating_segments(run_id, MagicMock(), still_owner=lambda: True)
        final = targets[-1].split(".recovery-")[0]
        assert open(final, "rb").read() == b"ours" and not os.path.exists(targets[-1])
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.get(RunSegment, rs_id)
            assert rs.status == SegmentStatus.completed and rs.local_result_path == final


class TestGateR3:
    def test_failed_ownership_check_after_download_discards_the_private_file(
        self, db_session
    ):
        import app.recovery as rec

        pid = _make_project(db_session)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        run_id, _ = _make_run_with_generating_segment(db_session, pid, task_id)
        fake_kie = MagicMock()
        fake_kie.get_task.return_value = {
            "state": "success",
            "resultJson": json.dumps({"resultUrls": ["https://x/r.mp4"]}),
        }
        targets = []

        def _download(url, dst):
            targets.append(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(b"v")

        fake_kie.download_result.side_effect = _download
        calls = {"n": 0}

        def _owner():
            calls["n"] += 1
            if calls["n"] >= 3:
                raise ConnectionError("redis down")
            return True

        with patch("app.recovery.KieClient", return_value=fake_kie):
            with pytest.raises(ConnectionError):
                rec._repoll_generating_segments(run_id, MagicMock(), still_owner=_owner)
        assert targets and not os.path.exists(targets[0])
