"""
Tests for app/pipeline_v2.py — analyze_project and process_run.

Strategy
--------
* Own SQLite DB engine (file-based for cross-session visibility) isolated from
  other test modules. pipeline_v2's get_session() is monkeypatched to use this
  engine.
* Tiny synthetic video created via ffmpeg (no GPU required).
* face.propose_segments is monkeypatched to return a fixed 3-segment partition:
    swap  0–9s  | keep  9–15s | swap  15–25s
  over a 25-second synthetic video.
* Fake KieClient and GDriveClient — no real network calls.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager

import pytest

# ---------------------------------------------------------------------------
# Prevent accidental real kie/gdrive calls; set a data dir for v2.
# ---------------------------------------------------------------------------

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_v2_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# App imports
# ---------------------------------------------------------------------------

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus
from app.kie_client import KieTaskFailed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Duration of the synthetic source video (seconds).
_VIDEO_DURATION = 25.0


@pytest.fixture(scope="session")
def synthetic_video():
    """
    Create a 25-second synthetic mp4 using ffmpeg lavfi sources.
    Returns the path; cleaned up at session end.
    """
    path = os.path.join(_TMP_DIR, "source_v2.mp4")
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"color=c=red:s=320x240:d={int(_VIDEO_DURATION)}:r=25",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={int(_VIDEO_DURATION)}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            "-c:a", "aac",
            path,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"ffmpeg failed:\n{r.stderr}"
    yield path


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    """
    Create a file-based SQLite engine isolated from the rest of the test suite.
    File-based (not :memory:) so multiple sessions see the same data.
    """
    db_path = os.path.join(_TMP_DIR, "pipeline_v2_test.db")
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
    """Per-test database session; does NOT auto-rollback because pipeline
    functions open their own sessions and we need data to persist."""
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture(autouse=True)
def patch_get_session(db_engine, monkeypatch):
    """
    Replace app.pipeline_v2's get_session with one that uses the test engine.
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

    import app.pipeline_v2 as pipeline_v2_mod
    monkeypatch.setattr(pipeline_v2_mod, "get_session", _test_get_session)


# ---------------------------------------------------------------------------
# Fixed segment partition returned by the monkeypatched propose_segments.
# Partition: [0, 9) swap | [9, 15) keep | [15, 25) swap
# ---------------------------------------------------------------------------

from app.face import ProposedSegment as _PS

_FIXED_SEGMENTS = [
    _PS(start_sec=0.0, end_sec=9.0, has_face=True, action="swap"),
    _PS(start_sec=9.0, end_sec=15.0, has_face=False, action="keep"),
    _PS(start_sec=15.0, end_sec=25.0, has_face=True, action="swap"),
]


@pytest.fixture()
def patch_propose(monkeypatch):
    """Monkeypatch face.propose_segments to return the fixed partition."""
    import app.pipeline_v2 as pipeline_v2_mod
    import app.face as face_mod

    monkeypatch.setattr(face_mod, "propose_segments", lambda *a, **kw: _FIXED_SEGMENTS)
    monkeypatch.setattr(
        pipeline_v2_mod.face_mod, "propose_segments", lambda *a, **kw: _FIXED_SEGMENTS
    )
    return _FIXED_SEGMENTS


# ---------------------------------------------------------------------------
# Fake external clients
# ---------------------------------------------------------------------------


class FakeKieClient:
    """
    Fake KieClient.

    - upload_file  → returns a fake URL, records calls.
    - create_task  → returns a fake task ID; records prompt + reference_image_urls.
    - poll_task    → returns a fake result URL (raises KieTaskFailed if the
                     task_id matches fail_task_id).
    - download_result → copies the synthetic video to dst so it is a real file.
    """

    def __init__(self, synthetic_video_path: str, fail_task_id: str | None = None):
        self._src = synthetic_video_path
        self._fail_task_id = fail_task_id
        self.upload_calls: list[str] = []
        self.create_task_calls: list[str] = []
        # Each entry: {"task_id": str, "prompt": str, "reference_image_urls": list}
        self.create_task_records: list[dict] = []
        # Gemini Omni task records: {"task_id", "prompt", "image_urls", "video_url",
        #   "video_start", "video_end", "resolution", "aspect_ratio", "duration"}
        self.create_omni_records: list[dict] = []
        self.poll_calls: list[str] = []

    def upload_file(self, local_path: str, upload_path: str = "charswap", **kw) -> str:
        self.upload_calls.append(local_path)
        return f"https://fake-kie.example.com/files/{os.path.basename(local_path)}"

    def create_task(
        self,
        *,
        prompt,
        reference_image_urls,
        reference_video_urls,
        resolution,
        aspect_ratio,
        duration,
    ) -> str:
        task_id = f"fake-task-{uuid.uuid4().hex[:8]}"
        self.create_task_calls.append(task_id)
        self.create_task_records.append({
            "task_id": task_id,
            "prompt": prompt,
            "reference_image_urls": list(reference_image_urls),
        })
        return task_id

    def create_omni_task(
        self,
        *,
        prompt,
        image_urls,
        video_url,
        video_start,
        video_end,
        resolution,
        aspect_ratio,
        duration,
        seed=None,
    ) -> str:
        task_id = f"fake-omni-{uuid.uuid4().hex[:8]}"
        self.create_task_calls.append(task_id)
        self.create_omni_records.append({
            "task_id": task_id,
            "prompt": prompt,
            "image_urls": list(image_urls),
            "video_url": video_url,
            "video_start": video_start,
            "video_end": video_end,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        })
        return task_id

    def poll_task(self, task_id: str, **kw) -> str:
        self.poll_calls.append(task_id)
        if self._fail_task_id and task_id == self._fail_task_id:
            raise KieTaskFailed("injected failure")
        return f"https://fake-kie.example.com/results/{task_id}.mp4"

    def get_task(self, task_id: str, **kw) -> dict:
        # The v2 poll loop uses get_task; return a terminal recordInfo 'data' dict.
        self.poll_calls.append(task_id)
        if self._fail_task_id and task_id == self._fail_task_id:
            return {"state": "fail", "failMsg": "injected failure"}
        url = f"https://fake-kie.example.com/results/{task_id}.mp4"
        return {"state": "success", "resultJson": json.dumps({"resultUrls": [url]})}

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
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as fh:
            fh.write(b"fake-gdrive-content")
        return dst_path

    def upload_file(
        self,
        local_path: str,
        folder_id: str | None = None,
        name: str | None = None,
        **kw,
    ) -> dict:
        self.upload_calls.append(
            {"local_path": local_path, "folder_id": folder_id, "name": name}
        )
        file_id = f"gdrive-{uuid.uuid4().hex[:8]}"
        return {
            "id": file_id,
            "webViewLink": f"https://drive.google.com/file/{file_id}",
        }


def test_poll_pending_tasks_does_not_hold_session_while_sleeping(monkeypatch, tmp_path):
    """The poll wait loop must not keep a DB session/transaction open."""
    import app.pipeline_v2 as pv2

    active_sessions = {"count": 0}
    slept = {"called": False}

    class _FakeRunSegment:
        def __init__(self):
            self.status = SegmentStatus.generating
            self.seedance_result_url = None
            self.local_result_path = None

    fake_rs = _FakeRunSegment()

    class _FakeSession:
        def get(self, _model, _id):
            return fake_rs

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    @contextmanager
    def fake_get_session():
        active_sessions["count"] += 1
        try:
            yield _FakeSession()
        finally:
            active_sessions["count"] -= 1

    class _SlowThenDoneKie:
        def __init__(self):
            self.calls = 0

        def get_task(self, _task_id):
            self.calls += 1
            if self.calls == 1:
                return {"state": "processing"}
            return {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://fake/result.mp4"]}),
            }

        def download_result(self, _url, dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, "wb") as fh:
                fh.write(b"result")

    def fake_sleep(_seconds):
        slept["called"] = True
        assert active_sessions["count"] == 0

    monkeypatch.setattr(pv2, "get_session", fake_get_session)
    monkeypatch.setattr(pv2.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        pv2, "transition", lambda obj, status: setattr(obj, "status", status)
    )

    pending = {
        "task-1": {
            "rs_id": "rs-1",
            "index": 0,
            "deadline": pv2.time.monotonic() + 60,
        }
    }

    pv2._poll_pending_tasks(
        pending=pending,
        kie=_SlowThenDoneKie(),
        results_dir=str(tmp_path),
    )

    assert slept["called"] is True
    assert pending == {}
    assert fake_rs.status == SegmentStatus.completed
    assert fake_rs.seedance_result_url == "https://fake/result.mp4"


# ---------------------------------------------------------------------------
# Helper: create a VideoProject row and return its id.
# ---------------------------------------------------------------------------


def _create_project(
    db_session,
    synthetic_video: str,
    *,
    source_type: str = "upload",
    status: ProjectStatus = ProjectStatus.created,
) -> str:
    project = VideoProject(
        id=str(uuid.uuid4()),
        source_type=source_type,
        source_ref=(
            synthetic_video
            if source_type == "upload"
            else "https://drive.google.com/file/d/fake123/view"
        ),
        source_local_path=synthetic_video if source_type == "upload" else None,
        status=status,
    )
    db_session.add(project)
    db_session.commit()
    return project.id


def _create_run(
    db_session,
    project_id: str,
    *,
    status: RunStatus = RunStatus.queued,
    gdrive_folder_id: str | None = None,
    reference_image_urls: list | None = None,
    prompt: str = "Replace the character",
) -> str:
    run = Run(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name="Test Run",
        prompt=prompt,
        reference_image_urls=reference_image_urls or [],
        resolution="480p",
        status=status,
        gdrive_folder_id=gdrive_folder_id,
    )
    db_session.add(run)
    db_session.commit()
    return run.id


# ---------------------------------------------------------------------------
# Tests: analyze_project
# ---------------------------------------------------------------------------


class TestAnalyzeProject:
    def test_analyze_happy_path(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """analyze_project: persists probe fields + SegmentDefs; project ends ready."""
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)

        analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.status == ProjectStatus.ready
            assert project.duration_sec is not None
            assert project.duration_sec == pytest.approx(_VIDEO_DURATION, abs=1.0)
            assert project.width == 320
            assert project.height == 240
            assert project.fps is not None
            assert project.aspect_ratio == "4:3"

            seg_defs = list(project.segments)
            assert len(seg_defs) == 3

            # index 0 — swap
            assert seg_defs[0].action == "swap"
            assert seg_defs[0].has_face is True
            assert seg_defs[0].start_sec == pytest.approx(0.0)
            assert seg_defs[0].end_sec == pytest.approx(9.0)

            # index 1 — keep
            assert seg_defs[1].action == "keep"
            assert seg_defs[1].has_face is False

            # index 2 — swap
            assert seg_defs[2].action == "swap"
            assert seg_defs[2].has_face is True

    def test_analyze_transitions_to_failed_on_probe_error(
        self, db_session, synthetic_video, monkeypatch
    ):
        """If probe raises, project transitions to failed and error_message is set."""
        from app.pipeline_v2 import analyze_project
        import app.pipeline_v2 as pipeline_v2_mod

        project_id = _create_project(db_session, synthetic_video)

        def _bad_probe(path):
            raise RuntimeError("simulated probe failure")

        monkeypatch.setattr(pipeline_v2_mod.media_mod, "probe", _bad_probe)

        with pytest.raises(RuntimeError, match="simulated probe failure"):
            analyze_project(project_id)

        db_session.expire_all()
        project = db_session.get(VideoProject, project_id)
        assert project.status == ProjectStatus.failed
        assert "simulated probe failure" in project.error_message

    def test_analyze_source_not_found_raises(self, db_session, db_engine):
        """Missing local file raises FileNotFoundError; project ends failed."""
        from app.pipeline_v2 import analyze_project

        project = VideoProject(
            id=str(uuid.uuid4()),
            source_type="upload",
            source_ref="/nonexistent/video.mp4",
            source_local_path="/nonexistent/video.mp4",
            status=ProjectStatus.created,
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        with pytest.raises(FileNotFoundError):
            analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            p = s.get(VideoProject, project_id)
            assert p.status == ProjectStatus.failed


# ---------------------------------------------------------------------------
# Tests: process_run
# ---------------------------------------------------------------------------


class TestProcessRun:
    def _setup_project_and_run(
        self,
        db_session,
        db_engine,
        synthetic_video,
        patch_propose,
        *,
        gdrive_folder_id: str | None = None,
        reference_image_urls: list | None = None,
    ):
        """
        Run analyze_project to get a ready project, then create a queued run.
        Returns (project_id, run_id).
        """
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)

        run_id = _create_run(
            db_session,
            project_id,
            gdrive_folder_id=gdrive_folder_id,
            reference_image_urls=reference_image_urls,
        )
        return project_id, run_id

    def test_process_happy_path(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """All swap RunSegments reach completed; run ends done; stitch produces a file."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup_project_and_run(
            db_session, db_engine, synthetic_video, patch_propose
        )
        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done, f"Expected done, got {run.status}"
            assert run.result_local_path is not None
            assert os.path.exists(run.result_local_path)

            run_segs = list(run.run_segments)
            assert len(run_segs) == 2  # two swap segments in our fixed partition
            assert all(rs.status == SegmentStatus.completed for rs in run_segs)
            for rs in run_segs:
                assert rs.local_result_path is not None
                assert os.path.exists(rs.local_result_path)

        # Two swap segments → two clip uploads + no ref images → 2 total upload calls
        assert len(fake_kie.upload_calls) == 2
        assert len(fake_kie.create_task_calls) == 2
        assert len(fake_kie.poll_calls) == 2

    def test_threaded_submit_sees_new_run_segments_without_ffmpeg(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Fresh RunSegments are committed before submit threads open sessions."""
        from app.media import MediaInfo
        import app.pipeline_v2 as pipeline_v2_mod
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")

        project_id = _create_project(
            db_session,
            str(source),
            status=ProjectStatus.ready,
        )
        segs = [
            SegmentDef(
                project_id=project_id,
                index=0,
                start_sec=0.0,
                end_sec=5.0,
                has_face=True,
                action="swap",
            ),
            SegmentDef(
                project_id=project_id,
                index=1,
                start_sec=5.0,
                end_sec=10.0,
                has_face=True,
                action="swap",
            ),
        ]
        db_session.add_all(segs)
        run_id = _create_run(db_session, project_id)

        monkeypatch.setattr(
            pipeline_v2_mod.media_mod,
            "probe",
            lambda _path: MediaInfo(
                duration_sec=10.0,
                width=320,
                height=240,
                fps=25.0,
                aspect_ratio="4:3",
                has_audio=True,
            ),
        )
        monkeypatch.setattr(
            pipeline_v2_mod.media_mod,
            "cut_clip",
            lambda _src, _start, _end, dst, **_kw: (os.makedirs(os.path.dirname(dst), exist_ok=True), open(dst, "wb").write(b"clip")),
        )
        monkeypatch.setattr(
            pipeline_v2_mod.media_mod,
            "stitch",
            lambda _clips, audio_source, dst, **_kw: (os.makedirs(os.path.dirname(dst), exist_ok=True), open(dst, "wb").write(b"final")),
        )

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            run_segments = (
                s.query(RunSegment)
                .filter(RunSegment.run_id == run_id)
                .order_by(RunSegment.index)
                .all()
            )
            assert len(run_segments) == 2
            assert all(rs.status == SegmentStatus.completed for rs in run_segments)

        assert len(fake_kie.create_task_calls) == 2

    def test_process_gdrive_upload_called_when_folder_set(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """When gdrive_folder_id is set on the run, upload is called after stitching."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup_project_and_run(
            db_session,
            db_engine,
            synthetic_video,
            patch_propose,
            gdrive_folder_id="folder-abc123",
        )
        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        assert len(fake_gdrive.upload_calls) == 1
        call = fake_gdrive.upload_calls[0]
        assert call["folder_id"] == "folder-abc123"

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.result_gdrive_file_id is not None
            assert run.status == RunStatus.done

    def test_resumability_skips_completed_run_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        Pre-mark one swap RunSegment as completed with a real result file.
        process_run must NOT call upload_file or create_task for it.
        """
        from app.pipeline_v2 import analyze_project, process_run
        from app.storage import run_results_dir

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        # Load the project's segment defs and manually create RunSegments.
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            seg_defs = list(project.segments)
            swap_defs = [sd for sd in seg_defs if sd.action == "swap"]

            # Create RunSegments for all swap defs.
            for sd in swap_defs:
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                )
                s.add(rs)
            s.commit()

        # Now pre-mark the FIRST swap RunSegment as completed with a real file.
        r_dir = run_results_dir(run_id, project_id)
        with Session() as s:
            run = s.get(Run, run_id)
            run_segs = sorted(run.run_segments, key=lambda rs: rs.index)
            first_rs = run_segs[0]

            fake_result = os.path.join(r_dir, f"result_{first_rs.index:04d}.mp4")
            shutil.copy2(synthetic_video, fake_result)

            first_rs.status = SegmentStatus.completed
            first_rs.local_result_path = fake_result
            first_rs.seedance_task_id = "pre-existing-task"
            first_rs.seedance_result_url = "https://fake/pre-existing.mp4"
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        # Only the second swap segment was actually processed.
        assert len(fake_kie.upload_calls) == 1, (
            f"Expected 1 upload call (only the non-resumed segment), "
            f"got {fake_kie.upload_calls}"
        )
        assert len(fake_kie.create_task_calls) == 1
        assert len(fake_kie.poll_calls) == 1

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done

    def test_retry_resets_failed_run_segment_and_completes(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        Regression: a RunSegment left in `failed` by a prior run must be reset
        to `pending` on retry — not crash with 'Invalid transition: failed → uploading'.
        """
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            seg_defs = list(project.segments)
            swap_defs = [sd for sd in seg_defs if sd.action == "swap"]

            # Create RunSegments and force the first one to failed.
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.failed if i == 0 else SegmentStatus.pending,
                    error_message="old failure" if i == 0 else None,
                    seedance_task_id="stale-task" if i == 0 else None,
                )
                s.add(rs)
            s.commit()

        # The stale task reports `fail` so the resume check resubmits the
        # segment (exercising the failed→pending reset) rather than recovering it
        # via the no-rebill path. Its resubmitted task (a new id) then succeeds.
        fake_kie = FakeKieClient(synthetic_video, fail_task_id="stale-task")
        fake_gdrive = FakeGDriveClient()

        # Must not raise InvalidTransition.
        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            for rs in run.run_segments:
                assert rs.status == SegmentStatus.completed
                assert rs.error_message is None

        # Both segments were processed (the failed one was reset and re-processed).
        assert len(fake_kie.create_task_calls) == 2

    def test_progress_is_committed_during_processing(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        process_run must COMMIT intermediate state so a separate reader (the API/UI)
        sees live progress — not stay 'queued / 0 done' until the whole run finishes.
        """
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        observed = {}

        class ObservingKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                # Read from a SEPARATE session/connection mid-processing.
                with sessionmaker(bind=db_engine)() as s2:
                    r = s2.get(Run, run_id)
                    observed["run_status"] = r.status
                    observed["generating"] = [
                        rs.index
                        for rs in r.run_segments
                        if rs.status == SegmentStatus.generating
                    ]
                return super().get_task(task_id, **kw)

        process_run(
            run_id, kie=ObservingKie(synthetic_video), gdrive=FakeGDriveClient()
        )

        # While the first segment was generating, a separate session saw the run
        # already in 'processing' and at least one RunSegment marked 'generating'.
        assert observed.get("run_status") == RunStatus.processing
        assert len(observed.get("generating", [])) >= 1

    def test_segment_permanent_failure_marks_run_incomplete(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        When ONE swap segment fails on every attempt (after retries), the run is
        marked `incomplete` and NOT stitched — we never mix a generated video
        with the original clip. The other segment still completes, its result is
        kept for the eventual re-stitch, and result_local_path stays None.
        """
        from app.pipeline_v2 import analyze_project, process_run
        import app.pipeline_v2 as pipeline_v2_mod

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        class FailSegmentZeroKie(FakeKieClient):
            """Every task generated from segment 0's clip (clip_0000) fails on
            every attempt; all other segments succeed."""

            def __init__(self, src):
                super().__init__(src)
                self._fail_tasks: set[str] = set()

            def create_task(self, *, reference_video_urls, **kw) -> str:
                task_id = super().create_task(
                    reference_video_urls=reference_video_urls, **kw
                )
                if reference_video_urls and "clip_0000" in reference_video_urls[0]:
                    self._fail_tasks.add(task_id)
                return task_id

            def get_task(self, task_id: str, **kw) -> dict:
                self.poll_calls.append(task_id)
                if task_id in self._fail_tasks:
                    return {"state": "fail", "failMsg": "injected failure for test"}
                url = f"https://fake/results/{task_id}.mp4"
                return {"state": "success",
                        "resultJson": json.dumps({"resultUrls": [url]})}

        fake_kie = FailSegmentZeroKie(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.incomplete
            assert run.result_local_path is None  # NOT stitched
            assert "did not complete" in (run.error_message or "")

            failed = [rs for rs in run.run_segments if rs.status == SegmentStatus.failed]
            completed = [rs for rs in run.run_segments if rs.status == SegmentStatus.completed]
            assert len(failed) == 1
            assert len(completed) == 1
            # The completed segment's result is preserved for the re-stitch.
            assert completed[0].local_result_path is not None

        # Segment 0 was retried up to the configured attempt cap before failing.
        assert len(fake_kie._fail_tasks) == pipeline_v2_mod.RUN_TASK_MAX_ATTEMPTS

    def test_transient_failure_retried_then_run_completes(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A segment whose FIRST task fails transiently is resubmitted; when the
        retry succeeds the run completes normally (done) with a stitched video."""
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        class FailFirstTaskKie(FakeKieClient):
            """Only the very first task id fails; its resubmission (a new id)
            succeeds — exercising the retry-recovery path."""

            def __init__(self, src):
                super().__init__(src)
                self._first_task: str | None = None

            def create_task(self, **kw) -> str:
                task_id = super().create_task(**kw)
                if self._first_task is None:
                    self._first_task = task_id
                return task_id

            def get_task(self, task_id: str, **kw) -> dict:
                self.poll_calls.append(task_id)
                if task_id == self._first_task:
                    return {"state": "fail", "failMsg": "Internal Error, Please try again later."}
                url = f"https://fake/results/{task_id}.mp4"
                return {"state": "success",
                        "resultJson": json.dumps({"resultUrls": [url]})}

        fake_kie = FailFirstTaskKie(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_local_path and os.path.exists(run.result_local_path)
            assert all(
                rs.status == SegmentStatus.completed for rs in run.run_segments
            )
        # One extra create_task beyond the 2 segments = the single retry.
        assert len(fake_kie.create_task_calls) == 3

    def test_run_segments_created_idempotently(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        If RunSegments already exist (e.g. from a prior call), process_run must not
        create duplicates — the ensure-RunSegment step is idempotent.
        """
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        # Pre-create RunSegments manually (simulating a prior incomplete call).
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            for sd in project.segments:
                if sd.action == "swap":
                    rs = RunSegment(
                        run_id=run_id,
                        segment_def_id=sd.id,
                        index=sd.index,
                        status=SegmentStatus.pending,
                    )
                    s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            # Should still be exactly 2 RunSegments (not 4).
            assert len(run.run_segments) == 2

    def test_process_run_missing_project_not_found(self, db_session):
        """process_run raises ValueError for a non-existent run."""
        from app.pipeline_v2 import process_run

        with pytest.raises(ValueError, match="Run not found"):
            process_run(str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Tests: TR6 per-segment override
# ---------------------------------------------------------------------------


class TestSegmentOverride:
    """
    Tests for per-segment prompt_override and reference_image_urls_override.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, prompt="run-level prompt")
        return project_id, run_id

    def test_prompt_override_used_for_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A RunSegment with prompt_override causes create_task to use that prompt."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        Session = sessionmaker(bind=db_engine)
        # Pre-create RunSegments; set prompt_override on the first one.
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = [sd for sd in project.segments if sd.action == "swap"]
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                    prompt_override="override prompt for seg 0" if i == 0 else None,
                )
                s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        from app.pipeline_v2 import process_run
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Check that two create_task calls were made and the first used the override.
        assert len(fake_kie.create_task_records) == 2
        prompts = {r["prompt"] for r in fake_kie.create_task_records}
        assert "override prompt for seg 0" in prompts
        assert "run-level prompt" in prompts

    def test_reference_override_used_for_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A RunSegment with reference_image_urls_override uses those URLs for create_task."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        Session = sessionmaker(bind=db_engine)
        override_refs = ["https://override-ref.example.com/img.jpg"]
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = [sd for sd in project.segments if sd.action == "swap"]
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                    reference_image_urls_override=override_refs if i == 0 else None,
                )
                s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        ref_sets = [frozenset(r["reference_image_urls"]) for r in fake_kie.create_task_records]
        # One record should include the override ref
        assert any("override-ref.example.com" in u for refs in ref_sets for u in refs)

    def test_single_segment_rerun_scenario(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        Simulate the full per-segment re-run flow:
        1. Pre-complete BOTH swap RunSegments (run is effectively done).
        2. Reset ONE to pending with a prompt_override.
        3. Call process_run → only that one is reprocessed (1 create_task call).
        4. Run ends done.
        """
        from app.pipeline_v2 import analyze_project, process_run
        from app.storage import run_results_dir

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, prompt="original prompt")

        r_dir = run_results_dir(run_id, project_id)

        Session = sessionmaker(bind=db_engine)

        # Pre-create RunSegments and mark both as completed.
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = sorted(
                [sd for sd in project.segments if sd.action == "swap"],
                key=lambda sd: sd.index,
            )
            rs_ids = []
            for sd in swap_defs:
                fake_result = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
                shutil.copy2(synthetic_video, fake_result)
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.completed,
                    local_result_path=fake_result,
                    seedance_task_id=f"pre-task-{sd.index}",
                    seedance_result_url=f"https://fake/pre-{sd.index}.mp4",
                )
                s.add(rs)
                s.flush()
                rs_ids.append(rs.id)
            s.commit()

        # Now reset the FIRST RunSegment to pending with an override prompt.
        with Session() as s:
            rs0 = s.get(RunSegment, rs_ids[0])
            rs0.status = SegmentStatus.pending
            rs0.error_message = None
            rs0.seedance_task_id = None
            rs0.seedance_result_url = None
            rs0.local_result_path = None
            rs0.prompt_override = "per-segment override prompt"
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Only ONE create_task call (the second was already completed, skipped).
        assert len(fake_kie.create_task_calls) == 1, (
            f"Expected 1 create_task (only the reset segment), "
            f"got {len(fake_kie.create_task_calls)}"
        )
        assert fake_kie.create_task_records[0]["prompt"] == "per-segment override prompt"

        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: model routing (Seedance vs Gemini Omni)
# ---------------------------------------------------------------------------


class TestModelRouting:
    """process_run must route to the kie method matching run.model."""

    def _setup_ready_project(self, db_session, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        return project_id

    def test_seedance_routes_to_create_task(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A default (seedance) run uses create_task, not create_omni_task."""
        from app.pipeline_v2 import process_run

        project_id = self._setup_ready_project(db_session, synthetic_video, patch_propose)
        run_id = _create_run(db_session, project_id)  # model defaults to seedance

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        assert len(fake_kie.create_omni_records) == 0

    def test_gemini_routes_to_create_omni_task(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A gemini-omni run uses create_omni_task with snapped duration + video trim."""
        from app.pipeline_v2 import process_run

        project_id = self._setup_ready_project(db_session, synthetic_video, patch_propose)

        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Gemini Run",
            prompt="Replace the character",
            reference_image_urls=[],
            model="gemini-omni",
            resolution="720p",
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run.id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Two swap segments → two omni calls, no seedance calls.
        assert len(fake_kie.create_omni_records) == 2
        assert len(fake_kie.create_task_records) == 0

        for rec in fake_kie.create_omni_records:
            assert rec["resolution"] == "720p"
            # 4:3 synthetic video (320x240) → landscape aspect for Gemini.
            assert rec["aspect_ratio"] == "16:9"
            assert rec["duration"] in (4, 6, 8, 10)
            assert rec["video_start"] == 0
            # Trim never exceeds Gemini's 10s cap.
            assert 0 < rec["video_end"] <= 10.0

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            r = s.get(Run, run.id)
            assert r.status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: Gemini Omni video-only clips + 10s limit
# ---------------------------------------------------------------------------


class TestGeminiOmniAudioAndLimit:
    """Gemini Omni clips are uploaded video-only (it fails on an audio track),
    capped at 10s, and always stitched with the original source audio.

    Media (probe / cut_clip / stitch) is fully mocked so these run without
    ffmpeg.
    """

    @staticmethod
    def _mock_media(monkeypatch, *, duration_sec, cut_calls, stitch_calls):
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(
                duration_sec=duration_sec, width=320, height=240,
                fps=25.0, aspect_ratio="4:3", has_audio=True,
            ),
        )

        def spy_cut_clip(_src, _start, _end, dst, *, include_audio=True):
            cut_calls.append({"dst": dst, "include_audio": include_audio})
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"clip")

        monkeypatch.setattr(pv2.media_mod, "cut_clip", spy_cut_clip)

        def spy_stitch(_clips, audio_source, dst, **kw):
            stitch_calls.append(kw)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"final")

        monkeypatch.setattr(pv2.media_mod, "stitch", spy_stitch)

    @staticmethod
    def _make_run(db_session, project_id, *, model, audio_mode="original"):
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="t",
            prompt="Replace the character",
            reference_image_urls=[],
            model=model,
            resolution="720p" if model == "gemini-omni" else "480p",
            audio_mode=audio_mode,
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def _two_swaps(self, db_session, project_id):
        # Two adjacent swap segments, no keep gap → every cut_clip call during
        # the run is a swap-submit cut.
        db_session.add_all([
            SegmentDef(project_id=project_id, index=0, start_sec=0.0, end_sec=5.0,
                       has_face=True, action="swap"),
            SegmentDef(project_id=project_id, index=1, start_sec=5.0, end_sec=10.0,
                       has_face=True, action="swap"),
        ])
        db_session.commit()

    def test_gemini_clip_sent_without_audio(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Every clip uploaded to Gemini Omni is cut with include_audio=False."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        self._two_swaps(db_session, project_id)
        run_id = self._make_run(db_session, project_id, model="gemini-omni")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_omni_records) == 2
        assert cut_calls, "expected at least one clip cut"
        assert all(c["include_audio"] is False for c in cut_calls), (
            f"Gemini clips must be video-only: {cut_calls}"
        )
        # Gemini always overlays the original source audio.
        assert stitch_calls and stitch_calls[0].get("audio_mode") == "original"

    def test_seedance_clip_keeps_audio(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Seedance clips are still cut WITH their audio track."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        self._two_swaps(db_session, project_id)
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        assert cut_calls and all(c["include_audio"] is True for c in cut_calls)

    def test_gemini_forces_original_audio_over_requested_seedance(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A Gemini run requesting audio_mode='seedance' still stitches 'original'."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        self._two_swaps(db_session, project_id)
        run_id = self._make_run(
            db_session, project_id, model="gemini-omni", audio_mode="seedance"
        )

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert stitch_calls and stitch_calls[0].get("audio_mode") == "original"

    def test_gemini_segment_over_10s_is_skipped(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A >10s swap segment is skipped for Gemini (create_omni_task never
        called). With no completed swap, the run is marked `incomplete` and the
        video is NOT stitched."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        db_session.add(
            SegmentDef(project_id=project_id, index=0, start_sec=0.0, end_sec=12.0,
                       has_face=True, action="swap")
        )
        db_session.commit()
        run_id = self._make_run(db_session, project_id, model="gemini-omni")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=12.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # No Gemini task created for the over-long segment.
        assert len(fake_kie.create_omni_records) == 0
        # Nothing stitched (mix of original + nothing is never produced).
        assert stitch_calls == []
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.incomplete
            assert run.result_local_path is None
            rs = (
                s.query(RunSegment)
                .filter(RunSegment.run_id == run_id)
                .one()
            )
            assert rs.status == SegmentStatus.failed
            assert "limit" in (rs.error_message or "")


class TestAnalyzeSegmentCap:
    """analyze_project caps segmentation at the most restrictive model limit."""

    def test_analyze_caps_segment_at_omni_limit(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """propose_segments is invoked with max_segment_sec == min(cfg, 10)."""
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source))

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(duration_sec=30.0, width=320, height=240,
                                 fps=25.0, aspect_ratio="4:3", has_audio=True),
        )
        captured: dict = {}

        def spy_propose(_path, **kw):
            captured.update(kw)
            return []

        monkeypatch.setattr(pv2.face_mod, "propose_segments", spy_propose)

        analyze_project(project_id)

        from app.config import settings
        expected = min(float(settings.SEGMENT_MAX_SECONDS), 10.0)
        assert captured.get("max_segment_sec") == expected
        assert captured["max_segment_sec"] <= 10.0


# ---------------------------------------------------------------------------
# Tests: TR7 audio_mode forwarding
# ---------------------------------------------------------------------------


class TestAudioModeForwarding:
    """
    Verify that process_run passes run.audio_mode to media_mod.stitch.
    Uses monkeypatch/spy on media_mod.stitch to capture kwargs without
    actually running ffmpeg for the stitch step.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose,
               audio_mode: str):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)

        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Audio Mode Test",
            prompt="Replace the character",
            reference_image_urls=[],
            resolution="480p",
            audio_mode=audio_mode,
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()
        return project_id, run.id

    def test_audio_mode_seedance_forwarded_to_stitch(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """process_run passes audio_mode='seedance' to media_mod.stitch."""
        from app.pipeline_v2 import process_run
        import app.pipeline_v2 as pipeline_v2_mod

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose, "seedance"
        )

        stitch_calls: list[dict] = []
        real_stitch = pipeline_v2_mod.media_mod.stitch

        def spy_stitch(*args, **kwargs):
            stitch_calls.append({"args": args, "kwargs": kwargs})
            return real_stitch(*args, **kwargs)

        monkeypatch.setattr(pipeline_v2_mod.media_mod, "stitch", spy_stitch)

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(stitch_calls) == 1, "stitch should be called exactly once"
        call = stitch_calls[0]
        assert call["kwargs"].get("audio_mode") == "seedance", (
            f"Expected audio_mode='seedance', got {call['kwargs'].get('audio_mode')!r}"
        )

    def test_audio_mode_original_forwarded_to_stitch(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """process_run passes audio_mode='original' to media_mod.stitch."""
        from app.pipeline_v2 import process_run
        import app.pipeline_v2 as pipeline_v2_mod

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose, "original"
        )

        stitch_calls: list[dict] = []
        real_stitch = pipeline_v2_mod.media_mod.stitch

        def spy_stitch(*args, **kwargs):
            stitch_calls.append({"args": args, "kwargs": kwargs})
            return real_stitch(*args, **kwargs)

        monkeypatch.setattr(pipeline_v2_mod.media_mod, "stitch", spy_stitch)

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(stitch_calls) == 1
        call = stitch_calls[0]
        assert call["kwargs"].get("audio_mode") == "original", (
            f"Expected audio_mode='original', got {call['kwargs'].get('audio_mode')!r}"
        )


# ---------------------------------------------------------------------------
# Tests: delivery retry + delivery-only retry (skip re-stitch)
# ---------------------------------------------------------------------------


class TestDeliveryRetryAndReuse:
    def test_delivery_retries_then_succeeds(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """A transient Drive-upload failure is retried within the same run."""
        from app.pipeline_v2 import analyze_project, process_run
        import app.pipeline_v2 as pv2

        # No real backoff sleeps.
        monkeypatch.setattr(pv2.time, "sleep", lambda *a, **k: None)
        # Skip the real ffmpeg stitch for speed (we only care about delivery).
        monkeypatch.setattr(pv2.media_mod, "stitch", lambda *a, **k: None)

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, gdrive_folder_id="folder-x")

        class FlakyGDrive(FakeGDriveClient):
            def __init__(self, fail_times):
                super().__init__()
                self._left = fail_times

            def upload_file(self, *a, **k):
                if self._left > 0:
                    self._left -= 1
                    raise RuntimeError("transient upload fail")
                return super().upload_file(*a, **k)

        gd = FlakyGDrive(2)  # fail twice, succeed on the 3rd (RUN_DELIVER_ATTEMPTS=3)
        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gd)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_gdrive_file_id is not None
        assert len(gd.upload_calls) == 1  # only the successful attempt recorded

    def test_delivery_only_retry_skips_stitch(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """Retry after a delivery failure: all segments completed + final exists →
        the run re-delivers WITHOUT re-stitching."""
        from app.pipeline_v2 import analyze_project, process_run
        from app.storage import run_results_dir
        import app.pipeline_v2 as pv2

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, gdrive_folder_id="folder-x")

        Session = sessionmaker(bind=db_engine)
        r_dir = run_results_dir(run_id, project_id)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            for sd in [d for d in project.segments if d.action == "swap"]:
                res = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
                shutil.copy2(synthetic_video, res)
                s.add(RunSegment(
                    run_id=run_id, segment_def_id=sd.id, index=sd.index,
                    status=SegmentStatus.completed, local_result_path=res,
                ))
            run = s.get(Run, run_id)
            final = os.path.join(r_dir, "final.mp4")
            shutil.copy2(synthetic_video, final)
            run.result_local_path = final
            s.commit()

        stitch_calls: list = []
        monkeypatch.setattr(pv2.media_mod, "stitch", lambda *a, **k: stitch_calls.append(1))

        gd = FakeGDriveClient()
        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gd)

        assert stitch_calls == []          # re-stitch skipped
        assert len(gd.upload_calls) == 1   # but it WAS re-delivered
        with Session() as s:
            assert s.get(Run, run_id).status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: parallel stitch cut_clip (STITCH_CUT_CONCURRENCY)
# ---------------------------------------------------------------------------


class TestStitchCutConcurrency:
    """
    Verify that the parallel cut_clip loop in the stitch phase:
    1. Produces clips in the correct segment order (not first-finished order).
    2. Is controlled by STITCH_CUT_CONCURRENCY env var.
    3. A cut_clip failure propagates and marks the run failed.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose):
        """Set up a ready project + queued run."""
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, prompt="concurrency test")
        return project_id, run_id

    def test_stitch_clip_order_preserved_with_concurrency(
        self,
        db_engine,
        db_session,
        synthetic_video,
        patch_propose,
        monkeypatch,
    ):
        """
        Even with STITCH_CUT_CONCURRENCY=2, clips must appear in ascending
        segment-index order in the stitch call — not in completion order.
        """
        import app.pipeline_v2 as pv2

        monkeypatch.setenv("STITCH_CUT_CONCURRENCY", "2")
        # Reload the constant so the monkeypatched env takes effect.
        monkeypatch.setattr(pv2, "STITCH_CUT_CONCURRENCY", 2)

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose
        )

        stitch_calls: list[list[str]] = []
        real_stitch = pv2.media_mod.stitch

        def spy_stitch(clip_paths, **kwargs):
            stitch_calls.append(list(clip_paths))
            return real_stitch(clip_paths, **kwargs)

        monkeypatch.setattr(pv2.media_mod, "stitch", spy_stitch)

        fake_kie = FakeKieClient(synthetic_video)
        from app.pipeline_v2 import process_run

        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(stitch_calls) == 1, "stitch should be called exactly once"
        clips = stitch_calls[0]

        # Verify ordering: segment indices appear in ascending order.
        # The fixed partition has 3 segments: swap(0), keep(1), swap(2).
        # The stitch list should be [result_0000, clip_0001, result_0002]
        # (or orig_0000/orig_0002 for fallbacks — doesn't matter, the point
        # is they're in index order).
        assert len(clips) == 3, f"Expected 3 clips, got {len(clips)}: {clips}"

        # Check filenames contain ascending indices.
        import re
        indices_in_filenames = [
            int(m.group(1))
            for path in clips
            if (m := re.search(r"(\d{4})\.mp4$", path))
        ]
        assert indices_in_filenames == sorted(indices_in_filenames), (
            f"Clips not in ascending index order: {clips}"
        )

        with sessionmaker(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done

    def test_stitch_cut_error_propagates(
        self,
        db_engine,
        db_session,
        synthetic_video,
        patch_propose,
        monkeypatch,
    ):
        """
        If a cut_clip call raises during the stitch phase, the exception must
        propagate (run ends failed), not be silently swallowed.
        """
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(pv2, "STITCH_CUT_CONCURRENCY", 2)

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose
        )

        call_count = {"n": 0}
        real_cut_clip = pv2.media_mod.cut_clip

        def failing_cut_clip(src, start, end, dst, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated cut_clip failure in stitch")
            return real_cut_clip(src, start, end, dst, **kw)

        monkeypatch.setattr(pv2.media_mod, "cut_clip", failing_cut_clip)

        fake_kie = FakeKieClient(synthetic_video)
        from app.pipeline_v2 import process_run

        with pytest.raises(RuntimeError, match="simulated cut_clip failure"):
            process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        with sessionmaker(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.failed

    def test_stitch_concurrency_1_behaves_identically(
        self,
        db_engine,
        db_session,
        synthetic_video,
        patch_propose,
        monkeypatch,
    ):
        """
        STITCH_CUT_CONCURRENCY=1 (serial) and =2 (parallel) must both produce
        a completed run — verify the serial path still works.
        """
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(pv2, "STITCH_CUT_CONCURRENCY", 1)

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose
        )

        from app.pipeline_v2 import process_run

        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=FakeGDriveClient())

        with sessionmaker(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: parallel stitch ordering guarantee (no ffmpeg/DB required)
# ---------------------------------------------------------------------------


class TestParallelStitchOrdering:
    """
    Pure-logic tests verifying the ThreadPoolExecutor ordering guarantee in
    _cut_or_lookup / ordered_futures.  No ffmpeg, no DB, no pipeline_v2 imports —
    tests the Python-level contract: futures collected in insertion order, results
    retrieved in that order, regardless of completion order.
    """

    def test_futures_collected_in_insertion_order(self):
        """
        When N futures are submitted to a ThreadPoolExecutor in a specific order
        and may complete in any order, collecting via ordered list (not as_completed)
        must return results in submission order.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        # Simulate segments with varying execution times to exercise out-of-order
        # completion:  seg 0 sleeps longer than seg 1.
        segments = [
            {"index": 0, "delay": 0.05, "result": "clip_0000.mp4"},
            {"index": 1, "delay": 0.01, "result": "clip_0001.mp4"},
            {"index": 2, "delay": 0.03, "result": "clip_0002.mp4"},
        ]

        def _work(seg):
            time.sleep(seg["delay"])
            return seg["result"]

        ordered_futures = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for seg in segments:
                fut = pool.submit(_work, seg)
                ordered_futures.append((seg["index"], fut))

        results = [fut.result() for _idx, fut in ordered_futures]

        # Must be in submission order, not completion order.
        assert results == ["clip_0000.mp4", "clip_0001.mp4", "clip_0002.mp4"]

    def test_future_exception_propagates_on_result(self):
        """
        If one of the futures raises, calling .result() must re-raise the exception.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _bad_work():
            raise ValueError("simulated cut error")

        def _good_work(val):
            return val

        ordered_futures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            ordered_futures.append(pool.submit(_bad_work))
            ordered_futures.append(pool.submit(_good_work, "ok"))

        with pytest.raises(ValueError, match="simulated cut error"):
            for fut in ordered_futures:
                fut.result()

    def test_concurrency_1_produces_same_order(self):
        """
        STITCH_CUT_CONCURRENCY=1 (max_workers=1) still produces results in
        insertion order — the pool serializes execution but order is preserved.
        """
        from concurrent.futures import ThreadPoolExecutor

        inputs = [f"clip_{i:04d}.mp4" for i in range(5)]
        ordered_futures = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            for item in inputs:
                fut = pool.submit(lambda x: x, item)
                ordered_futures.append(fut)

        results = [fut.result() for fut in ordered_futures]
        assert results == inputs
