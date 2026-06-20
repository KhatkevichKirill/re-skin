"""
Tests for app/pipeline.py — analyze_job and process_job.

Strategy
--------
* Own SQLite DB engine created per-session (bypasses whatever DATABASE_URL
  other test modules set).  The pipeline's get_session() is monkeypatched in
  every test to use this engine so there is no cross-module pollution.
* Tiny synthetic video created via ffmpeg (no GPU required).
* face.propose_segments is monkeypatched to return a fixed 3-segment partition
  (keep 0-3s, swap 3-7s, keep 7-10s) so tests are fast and deterministic.
* Fake KieClient and GDriveClient — no real network calls.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager

import pytest

# ---------------------------------------------------------------------------
# Prevent accidental real kie/gdrive calls, and set a data dir.
# ---------------------------------------------------------------------------

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_")
os.environ["DATA_DIR"] = os.path.join(_TMP_DIR, "data")
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

# Add backend to path so imports work when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# App imports
# ---------------------------------------------------------------------------

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Job, Segment
from app.state_machine import JobStatus, SegmentStatus
from app.kie_client import KieTaskFailed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Duration of the synthetic source video (seconds).
_VIDEO_DURATION = 10.0


@pytest.fixture(scope="session")
def synthetic_video():
    """
    Create a 10-second synthetic mp4 using ffmpeg lavfi sources.
    Returns the path; cleaned up at session end.
    """
    path = os.path.join(_TMP_DIR, "source.mp4")
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={int(_VIDEO_DURATION)}:r=25",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={int(_VIDEO_DURATION)}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            "-c:a", "aac",
            path,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"ffmpeg failed:\n{r.stderr}"
    yield path
    # Cleanup handled by _TMP_DIR removal in session teardown.


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    """
    Create a file-based SQLite engine isolated from the rest of the test suite.
    We use a file (not :memory:) so that multiple sessions opened by pipeline
    functions see the same data.
    """
    db_path = os.path.join(_TMP_DIR, "pipeline_test.db")
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
    """Per-test database session; does NOT auto-rollback because pipeline functions
    open their own sessions and we need data to persist between them."""
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture(autouse=True)
def patch_get_session(db_engine, monkeypatch):
    """
    Replace app.pipeline's get_session with one that uses the test engine.

    This is the key isolation fixture: the pipeline calls get_session() which
    would otherwise use whatever engine app.db was initialised with (potentially
    an in-memory DB from another test module).  We replace it here so all
    pipeline DB operations go to our isolated file-based engine.
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

    import app.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "get_session", _test_get_session)


# ---------------------------------------------------------------------------
# Fixed segment partition returned by the monkeypatched propose_segments.
# Partition: [0, 3) keep | [3, 7) swap | [7, 10) keep
# ---------------------------------------------------------------------------

from app.face import ProposedSegment as _PS

_FIXED_SEGMENTS = [
    _PS(start_sec=0.0, end_sec=3.0, has_face=False, action="keep"),
    _PS(start_sec=3.0, end_sec=7.0, has_face=True, action="swap"),
    _PS(start_sec=7.0, end_sec=10.0, has_face=False, action="keep"),
]


@pytest.fixture()
def patch_propose(monkeypatch):
    """Monkeypatch face.propose_segments to return the fixed partition."""
    import app.pipeline as pipeline_mod
    import app.face as face_mod

    monkeypatch.setattr(face_mod, "propose_segments", lambda *a, **kw: _FIXED_SEGMENTS)
    # Also patch through pipeline module's reference.
    monkeypatch.setattr(pipeline_mod.face_mod, "propose_segments", lambda *a, **kw: _FIXED_SEGMENTS)
    return _FIXED_SEGMENTS


# ---------------------------------------------------------------------------
# Fake external clients
# ---------------------------------------------------------------------------


class FakeKieClient:
    """
    Fake KieClient.

    - upload_file  → returns a fake URL, records calls.
    - create_task  → returns a fake task ID.
    - poll_task    → returns a fake result URL (or raises KieTaskFailed if
                     the segment_index_to_fail is set and matches).
    - download_result → copies the synthetic video to dst so it is a real file.
    """

    def __init__(self, synthetic_video_path: str, fail_task_id: str | None = None):
        self._src = synthetic_video_path
        self._fail_task_id = fail_task_id
        self.upload_calls: list[str] = []
        self.create_task_calls: list[str] = []
        self.poll_calls: list[str] = []

    def upload_file(self, local_path: str, upload_path: str = "charswap", **kw) -> str:
        self.upload_calls.append(local_path)
        return f"https://fake-kie.example.com/files/{os.path.basename(local_path)}"

    def create_task(self, *, prompt, reference_image_urls, reference_video_urls,
                    resolution, aspect_ratio, duration) -> str:
        task_id = f"fake-task-{uuid.uuid4().hex[:8]}"
        self.create_task_calls.append(task_id)
        return task_id

    def poll_task(self, task_id: str, **kw) -> str:
        self.poll_calls.append(task_id)
        if self._fail_task_id and task_id == self._fail_task_id:
            raise KieTaskFailed("injected failure")
        return f"https://fake-kie.example.com/results/{task_id}.mp4"

    def download_result(self, url: str, dst_path: str) -> None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(self._src, dst_path)


class FakeGDriveClient:
    """Fake GDriveClient — records calls, returns predictable data."""

    def __init__(self):
        self.upload_calls: list[dict] = []
        self.download_calls: list[dict] = []

    def download_file(self, link_or_id: str, dst_path: str) -> str:
        self.download_calls.append({"link": link_or_id, "dst": dst_path})
        # Write a placeholder so the caller doesn't crash on missing file.
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as fh:
            fh.write(b"fake-gdrive-content")
        return dst_path

    def upload_file(self, local_path: str, folder_id: str | None = None,
                    name: str | None = None, **kw) -> dict:
        self.upload_calls.append({"local_path": local_path, "folder_id": folder_id, "name": name})
        file_id = f"gdrive-{uuid.uuid4().hex[:8]}"
        return {"id": file_id, "webViewLink": f"https://drive.google.com/file/{file_id}"}


# ---------------------------------------------------------------------------
# Helper: create a minimal Job row and return its id.
# ---------------------------------------------------------------------------


def _create_job(
    db_session,
    synthetic_video: str,
    *,
    source_type: str = "upload",
    status: JobStatus = JobStatus.created,
    gdrive_folder_id: str | None = None,
    default_reference_image_urls: list | None = None,
) -> str:
    job = Job(
        id=str(uuid.uuid4()),
        source_type=source_type,
        source_ref=synthetic_video if source_type == "upload" else "https://drive.google.com/file/d/fake123/view",
        source_local_path=synthetic_video if source_type == "upload" else None,
        default_prompt="Replace the character",
        default_reference_image_urls=default_reference_image_urls or [],
        resolution="480p",
        status=status,
        gdrive_folder_id=gdrive_folder_id,
    )
    db_session.add(job)
    db_session.commit()
    return job.id


# ---------------------------------------------------------------------------
# Tests: analyze_job
# ---------------------------------------------------------------------------


class TestAnalyzeJob:
    def test_analyze_happy_path(self, db_engine, db_session, synthetic_video, patch_propose):
        """analyze_job: persists probe fields + expected segments; job ends in review."""
        from app.pipeline import analyze_job

        job_id = _create_job(db_session, synthetic_video)

        analyze_job(job_id)

        # Verify inside a fresh session to avoid ORM cache.
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            job = s.get(Job, job_id)
            assert job.status == JobStatus.review
            assert job.duration_sec is not None
            assert job.duration_sec == pytest.approx(_VIDEO_DURATION, abs=1.0)
            assert job.width == 320
            assert job.height == 240
            assert job.fps is not None
            assert job.aspect_ratio == "4:3"

            segs = list(job.segments)
            assert len(segs) == 3

            # index 0 — keep
            assert segs[0].action == "keep"
            assert segs[0].has_face is False
            assert segs[0].status == SegmentStatus.skipped

            # index 1 — swap
            assert segs[1].action == "swap"
            assert segs[1].has_face is True
            assert segs[1].status == SegmentStatus.pending

            # index 2 — keep
            assert segs[2].action == "keep"
            assert segs[2].status == SegmentStatus.skipped

    def test_analyze_transitions_to_failed_on_error(self, db_session, synthetic_video, monkeypatch):
        """If probe raises, job transitions to failed and error_message is set."""
        from app.pipeline import analyze_job
        import app.pipeline as pipeline_mod

        job_id = _create_job(db_session, synthetic_video)

        def _bad_probe(path):
            raise RuntimeError("simulated probe failure")

        monkeypatch.setattr(pipeline_mod.media_mod, "probe", _bad_probe)

        with pytest.raises(RuntimeError, match="simulated probe failure"):
            analyze_job(job_id)

        db_session.expire_all()
        job = db_session.get(Job, job_id)
        assert job.status == JobStatus.failed
        assert "simulated probe failure" in job.error_message


# ---------------------------------------------------------------------------
# Tests: process_job
# ---------------------------------------------------------------------------


class TestProcessJob:
    def _setup_job_after_analysis(self, db_session, db_engine, synthetic_video, patch_propose):
        """
        Run analyze_job so the job has real segments, then manually set it to queued.
        Returns job_id.
        """
        from app.pipeline import analyze_job
        from app.state_machine import transition as sm_transition

        job_id = _create_job(db_session, synthetic_video)
        analyze_job(job_id)

        # Simulate operator submit: review → queued.
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            job = s.get(Job, job_id)
            sm_transition(job, JobStatus.queued)
            s.commit()

        return job_id

    def test_process_happy_path(self, db_engine, db_session, synthetic_video, patch_propose):
        """All swap segments reach completed; job ends in done; stitch produces a file."""
        from app.pipeline import process_job

        job_id = self._setup_job_after_analysis(db_session, db_engine, synthetic_video, patch_propose)
        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_job(job_id, kie=fake_kie, gdrive=fake_gdrive)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            job = s.get(Job, job_id)
            assert job.status == JobStatus.done, f"Expected done, got {job.status}"
            assert job.result_local_path is not None
            assert os.path.exists(job.result_local_path)

            segs = list(job.segments)
            swap_segs = [sg for sg in segs if sg.action == "swap"]
            assert all(sg.status == SegmentStatus.completed for sg in swap_segs)
            for sg in swap_segs:
                assert sg.local_result_path is not None
                assert os.path.exists(sg.local_result_path)

        # upload_file is called once per swap segment (clip upload) + no ref images
        # in this test (no default_reference_image_urls set).
        assert len(fake_kie.upload_calls) == 1  # one swap segment → one clip upload
        assert len(fake_kie.create_task_calls) == 1
        assert len(fake_kie.poll_calls) == 1

    def test_process_gdrive_upload_called_when_folder_set(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """When gdrive_folder_id is set on the job, upload is called after stitching."""
        from app.pipeline import process_job

        # Create job with a gdrive folder id.
        from app.state_machine import transition as sm_transition

        job_id = _create_job(
            db_session, synthetic_video, gdrive_folder_id="folder-abc123"
        )
        from app.pipeline import analyze_job
        analyze_job(job_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            job = s.get(Job, job_id)
            sm_transition(job, JobStatus.queued)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_job(job_id, kie=fake_kie, gdrive=fake_gdrive)

        assert len(fake_gdrive.upload_calls) == 1
        call = fake_gdrive.upload_calls[0]
        assert call["folder_id"] == "folder-abc123"

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            job = s.get(Job, job_id)
            assert job.result_gdrive_file_id is not None

    def test_resumability_skips_completed_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        Pre-mark the swap segment as completed with a real result file.
        process_job must NOT call upload_file or create_task for it.
        """
        from app.pipeline import process_job, analyze_job
        from app.state_machine import transition as sm_transition

        job_id = _create_job(db_session, synthetic_video)
        analyze_job(job_id)

        # Create a fake result file and mark the swap segment completed.
        from app.storage import results_dir as get_results_dir
        r_dir = get_results_dir(job_id)
        fake_result = os.path.join(r_dir, "result_0001.mp4")
        shutil.copy2(synthetic_video, fake_result)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            job = s.get(Job, job_id)
            swap_seg = next(sg for sg in job.segments if sg.action == "swap")
            # Manually force to completed state (bypass state machine for test setup).
            swap_seg.status = SegmentStatus.completed
            swap_seg.local_result_path = fake_result
            swap_seg.seedance_task_id = "pre-existing-task"
            swap_seg.seedance_result_url = "https://fake/pre-existing.mp4"
            sm_transition(job, JobStatus.queued)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_job(job_id, kie=fake_kie, gdrive=fake_gdrive)

        # The swap segment was already done — no kie calls for it.
        assert len(fake_kie.upload_calls) == 0, (
            f"Expected 0 upload calls (resume), got {fake_kie.upload_calls}"
        )
        assert len(fake_kie.create_task_calls) == 0, (
            f"Expected 0 create_task calls (resume), got {fake_kie.create_task_calls}"
        )
        assert len(fake_kie.poll_calls) == 0

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            job = s.get(Job, job_id)
            assert job.status == JobStatus.done

    def test_segment_failure_drives_job_to_failed(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        When poll_task raises KieTaskFailed, the segment and job both end in failed;
        segment.error_message is stored.
        """
        from app.pipeline import process_job, analyze_job
        from app.state_machine import transition as sm_transition

        job_id = _create_job(db_session, synthetic_video)
        analyze_job(job_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            job = s.get(Job, job_id)
            sm_transition(job, JobStatus.queued)
            s.commit()

        # We need the fake client to fail on the task that gets created.
        # The fail_task_id must match what create_task returns; we intercept via
        # a subclass that captures and then fails the first task.
        class FailOnPollKieClient(FakeKieClient):
            def __init__(self, src):
                super().__init__(src)
                self._first_task: str | None = None

            def create_task(self, **kw) -> str:
                task_id = super().create_task(**kw)
                if self._first_task is None:
                    self._first_task = task_id
                return task_id

            def poll_task(self, task_id: str, **kw) -> str:
                super_result = None
                self.poll_calls.append(task_id)
                if task_id == self._first_task:
                    raise KieTaskFailed("injected failure for test")
                return f"https://fake/results/{task_id}.mp4"

        fake_kie = FailOnPollKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        with pytest.raises(KieTaskFailed):
            process_job(job_id, kie=fake_kie, gdrive=fake_gdrive)

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            job = s.get(Job, job_id)
            assert job.status == JobStatus.failed
            assert job.error_message is not None

            swap_seg = next(sg for sg in job.segments if sg.action == "swap")
            assert swap_seg.status == SegmentStatus.failed
            assert swap_seg.error_message is not None
            assert "injected failure" in swap_seg.error_message
