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


def _make_redis_mock(*, queue_len: int = 0, started_ids: list | None = None):
    """
    Build a fake Redis connection and fake RQ Queue + StartedJobRegistry.
    Returns the redis mock; patch it into recovery._queue_is_idle.
    """
    # We patch _queue_is_idle directly rather than mocking all of rq internals.
    return started_ids or []


# ---------------------------------------------------------------------------
# 1. Safety gate: skip reconciliation when queue is NOT idle
# ---------------------------------------------------------------------------


class TestSafetyGate:
    def test_skip_when_queue_has_pending_jobs(self, db_session):
        """reconcile_orphaned_runs must NOT touch runs when the queue is not idle."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.processing)

        enqueue_calls = []

        def _fake_idle(redis_conn):
            return False  # queue is NOT idle

        with patch.object(rec, "_queue_is_idle", _fake_idle):
            with patch("app.recovery.enqueue_process_run") as mock_enq:
                mock_redis = MagicMock()
                rec.reconcile_orphaned_runs(mock_redis)
                mock_enq.assert_not_called()

        # Run must remain in processing (untouched).
        db_session.expire_all()
        run = db_session.get(Run, run_id)
        assert run.status == RunStatus.processing

    def test_skip_when_started_registry_non_empty(self, db_session):
        """Even if queue is empty, non-empty started-registry means a job is live."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.queued)

        def _fake_idle(redis_conn):
            return False  # started registry is non-empty

        with patch.object(rec, "_queue_is_idle", _fake_idle):
            with patch("app.recovery.enqueue_process_run") as mock_enq:
                mock_redis = MagicMock()
                rec.reconcile_orphaned_runs(mock_redis)
                mock_enq.assert_not_called()

        db_session.expire_all()
        run = db_session.get(Run, run_id)
        assert run.status == RunStatus.queued  # still queued, untouched


# ---------------------------------------------------------------------------
# 2. Orphan detection and re-enqueue on idle queue
# ---------------------------------------------------------------------------


class TestOrphanDetectionAndReenqueue:
    def test_queued_orphan_is_reenqueued(self, db_session, db_engine):
        """A run stuck in `queued` with an idle queue is re-enqueued."""
        import app.recovery as rec

        project_id = _make_project(db_session)
        run_id = _make_run(db_session, project_id, RunStatus.queued)

        with patch.object(rec, "_queue_is_idle", return_value=True):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = MagicMock()
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

        with patch.object(rec, "_queue_is_idle", return_value=True):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = MagicMock()
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

        with patch.object(rec, "_queue_is_idle", return_value=True):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = MagicMock()
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

        with patch.object(rec, "_queue_is_idle", return_value=True):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = MagicMock()
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

        with patch.object(rec, "_queue_is_idle", return_value=True):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = MagicMock()
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

        with patch.object(rec, "_queue_is_idle", return_value=True):
            with patch.object(rec, "_repoll_generating_segments"):
                with patch("app.recovery.enqueue_process_run") as mock_enq:
                    mock_redis = MagicMock()
                    rec.reconcile_orphaned_runs(mock_redis)
                    called_ids = {call.args[0] for call in mock_enq.call_args_list}
                    for rid in ids:
                        assert rid in called_ids


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
