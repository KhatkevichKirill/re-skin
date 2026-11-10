"""
Tests for the v2 REST API (app/api_v2.py).

Strategy
--------
- Temp SQLite DB per test session (DATABASE_URL set before importing app).
- enqueue_analyze_project and enqueue_process_run are monkeypatched to spies.
- A TestClient wraps the FastAPI app for request-level tests.
- ffmpeg is used for the frame endpoint test (generates a tiny synthetic video).
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager

# Must set DATABASE_URL and DATA_DIR before any app import
_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
_data_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _data_tmp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import ai_models, project_types
from app.config import settings
from app.db import Base, get_db
from app.models import (
    LibraryClip,
    VoiceClip,
    OverlayJob,
    Run,
    RunCaption,
    RunSegment,
    RunSegmentImageJob,
    RunSegmentRangeAttempt,
    RunTailEdit,
    SegmentDef,
    VideoProject,
)
from app.schemas_v2 import ProjectUpdate
from app.state_machine import (
    CaptionStatus,
    OverlayStatus,
    ProjectStatus,
    RangeAttemptStatus,
    RunStatus,
    SegmentImageStatus,
    SegmentStatus,
    TailEditStatus,
)
import app.api_v2 as api_v2_module


@pytest.fixture()
def tmp_path():
    """Per-test scratch dir INSIDE the storage root, shadowing pytest's own.

    Every endpoint that serves or uploads a stored path now refuses anything
    outside DATA_DIR (AICP-148), and real stored paths always live there — so
    fixture files must too, or every "serves the file" test would 404.
    """
    import pathlib

    from app import storage

    root = storage.data_dir_root()
    os.makedirs(root, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix="tmp-path-", dir=root))


# ---------------------------------------------------------------------------
# Session-scoped engine + schema creation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(
        f"sqlite:///{_db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(eng, "connect")
    def set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="session")
def SessionFactory(engine):
    return sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )


# ---------------------------------------------------------------------------
# Per-test DB session + TestClient
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session(SessionFactory):
    """Yield a session; rollback after the test so state doesn't leak."""
    session = SessionFactory()
    yield session
    session.rollback()
    session.close()


class _InlineLocaliseQueue:
    """Stands in for Redis + the localise-worker (AICP-338) — the suite has no
    Redis. Admission reserves and "enqueues" exactly like the real
    localisation_prompt_job.admit (minus the lock), and the enqueue runs the
    REAL job body (localisation_prompt_job.run) inline, so a POST followed by
    a poll exercises the production code path end to end. ``hold=True`` keeps
    jobs queued instead, for dedupe and polling tests."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.pointers: dict[str, str] = {}
        self.reserves = 0
        self.hold = False

    def admit(self, *, key, reserve, enqueue):
        current = self.pointers.get(key)
        if current and self.jobs.get(current, {}).get("status") in ("queued", "running"):
            return current, False
        reserve()
        self.reserves += 1
        job_id = "localise-prompt-" + uuid.uuid4().hex
        enqueue(job_id)
        self.pointers[key] = job_id
        return job_id, True

    def enqueue(self, job_id, snapshot, *, meta):
        from app import localisation_prompt_job

        job = {"meta": meta, "snapshot": snapshot, "status": "queued"}
        self.jobs[job_id] = job
        if not self.hold:
            self.run(job_id)

    def run(self, job_id):
        from app import localisation_prompt_job

        job = self.jobs[job_id]
        envelope = localisation_prompt_job.run(job["snapshot"])
        if envelope.get("ok"):
            job.update(status="done", result=envelope["result"])
        else:
            job.update(
                status="failed",
                error={"status_code": envelope["status_code"], "detail": envelope["detail"]},
            )

    def state(self, job_id, pid):
        job = self.jobs.get(job_id)
        if job is None:
            return {"status": "expired"} if job_id.startswith("localise-prompt-") else None
        if job["meta"]["project_id"] != pid:
            return None
        out = {
            "status": job["status"],
            "fingerprint": job["meta"]["fingerprint"],
            "language": job["meta"]["language"],
            "swap_character": job["meta"]["swap_character"],
        }
        for k in ("result", "error"):
            if k in job:
                out[k] = job[k]
        return out


def _install_inline_localise_queue(monkeypatch):
    import app.api_v2 as api_v2_module

    queue = _InlineLocaliseQueue()
    monkeypatch.setattr(api_v2_module, "_localise_admit", queue.admit)
    monkeypatch.setattr(api_v2_module, "_localise_state", queue.state)
    monkeypatch.setattr(api_v2_module, "enqueue_localisation_prompt", queue.enqueue)
    api_v2_module._test_localise_queue = queue
    return queue


@pytest.fixture()
def localise_queue():
    """The _InlineLocaliseQueue the ``client`` fixture installed."""
    import app.api_v2 as api_v2_module

    return api_v2_module._test_localise_queue


class _PolledTranslation:
    """A finished translate job, shaped like the old synchronous response.

    ``status_code`` is the job's ``error.status_code`` (or 200 when done), so
    every pre-AICP-338 assertion about the endpoint's answer still means what
    it meant."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _translate_via_job(client, pid, payload):
    """POST /localisation-prompt, then poll its job once (the inline queue has
    already run it). A synchronous refusal comes back as the POST response."""
    resp = client.post(f"/api/v2/projects/{pid}/localisation-prompt", data=payload)
    if resp.status_code != 202:
        return resp
    job_id = resp.json()["job_id"]
    poll = client.get(f"/api/v2/projects/{pid}/localisation-prompt/jobs/{job_id}")
    assert poll.status_code == 200, poll.text
    assert poll.headers.get("cache-control") == "no-store"
    body = poll.json()
    if body["status"] == "done":
        return _PolledTranslation(200, body["result"])
    assert body["status"] == "failed", body
    return _PolledTranslation(body["error"]["status_code"], {"detail": body["error"]["detail"]})


@pytest.fixture()
def client(engine, SessionFactory, monkeypatch):
    """TestClient with DB dependency overridden and task queues mocked."""
    import app.api_v2 as api_v2_module
    import app.public as public_module

    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project", lambda pid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_preprocess_run", lambda rid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_export_run_to_sheet", lambda rid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_tail_edit", lambda eid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_segment_image", lambda jid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_transcribe_project", lambda pid: None)
    # AICP-365: POST /transcribe rings the durable job's doorbell instead.
    monkeypatch.setattr(api_v2_module.localisation_jobs, "enqueue", lambda job: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_range_attempt", lambda aid: None)
    _install_inline_localise_queue(monkeypatch)

    from app.main import app

    def override_get_db():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    # Some endpoints (file-streaming ones) manage their own session via
    # `with get_session() as db:` instead of the overridden get_db dependency
    # (see B11). app.db's module-level get_session() is bound to whichever
    # test module's DATABASE_URL happened to be in effect when app.db was
    # first imported in this test process, which is NOT necessarily this
    # file's own temp DB when multiple test files run together. Patch the
    # name in each module that imports it so those endpoints use this test's
    # engine too (mirrors patch_get_session in test_pipeline_v2.py).
    @contextmanager
    def _test_get_session():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(api_v2_module, "get_session", _test_get_session)
    monkeypatch.setattr(public_module, "get_session", _test_get_session)

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Spy fixture: track enqueue calls
# ---------------------------------------------------------------------------


@pytest.fixture()
def enqueue_spy(monkeypatch):
    """Returns dicts that record calls to enqueue functions."""
    import app.api_v2 as api_v2_module

    calls: dict[str, list[str]] = {
        "analyze_project": [], "process_run": [], "preprocess_run": [],
        "transcribe_project": [], "segment_image": [], "range_attempt": [],
    }

    monkeypatch.setattr(
        api_v2_module,
        "enqueue_analyze_project",
        lambda pid: calls["analyze_project"].append(pid),
    )
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_process_run",
        lambda rid: calls["process_run"].append(rid),
    )
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_preprocess_run",
        lambda rid: calls["preprocess_run"].append(rid),
    )
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_transcribe_project",
        lambda pid: calls["transcribe_project"].append(pid),
    )
    # AICP-365: the durable transcription admission's doorbell.
    monkeypatch.setattr(
        api_v2_module.localisation_jobs,
        "enqueue",
        lambda job: calls["transcribe_project"].append(job.project_id),
    )
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_process_segment_image",
        lambda jid: calls["segment_image"].append(jid),
    )
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_process_range_attempt",
        lambda aid: calls["range_attempt"].append(aid),
    )
    return calls


@pytest.fixture()
def spy_client(engine, SessionFactory, enqueue_spy):
    """TestClient that also tracks enqueue calls."""
    from app.main import app
    from app.db import get_db

    def override_get_db():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, enqueue_spy

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_video_bytes() -> bytes:
    """Return a few bytes that pass as a video upload (storage test only)."""
    return b"\x00\x01\x02\x03" * 16


class _UploadStub:
    def __init__(self, data: bytes):
        self.file = io.BytesIO(data)


def test_safe_filename_strips_path_and_leading_dots():
    assert api_v2_module._safe_filename("../../evil.mp4") == "evil.mp4"
    assert api_v2_module._safe_filename("..\\..\\evil.jpg") == "evil.jpg"
    assert api_v2_module._safe_filename(".hidden") == "hidden"
    assert api_v2_module._safe_filename("...") == "upload"


def test_save_upload_streams_and_rejects_oversize(tmp_path):
    ok_path = tmp_path / "ok.bin"
    api_v2_module._save_upload(_UploadStub(b"abcd"), str(ok_path), max_bytes=4)
    assert ok_path.read_bytes() == b"abcd"

    too_large = tmp_path / "too-large.bin"
    with pytest.raises(Exception) as excinfo:
        api_v2_module._save_upload(_UploadStub(b"abcde"), str(too_large), max_bytes=4)
    assert getattr(excinfo.value, "status_code", None) == 413
    assert not too_large.exists()


def _make_project(session, **kwargs) -> VideoProject:
    defaults = dict(
        source_type="upload",
        source_ref="test.mp4",
        status=ProjectStatus.created,
    )
    defaults.update(kwargs)
    p = VideoProject(**defaults)
    session.add(p)
    session.commit()
    return p


def _staged_frame(project_id: str, name: str) -> str:
    """Path of a frame staged off a project's source video.

    With GPT Image 2 preprocessing on, the API only accepts trims from here
    (``_assert_trims_are_source_frames``) — an outside URL or an upload is a
    400. The file need not exist: create_run copies a staged frame only when it
    is on disk and otherwise stores the path as given.
    """
    from app.storage import project_dir

    return os.path.join(project_dir(project_id), "staged_frames", name)


def _make_segment_def(session, project_id: str, index: int, **kwargs) -> SegmentDef:
    defaults = dict(
        project_id=project_id,
        index=index,
        start_sec=float(index * 5),
        end_sec=float(index * 5 + 5),
        has_face=True,
        action="swap",
    )
    defaults.update(kwargs)
    s = SegmentDef(**defaults)
    session.add(s)
    session.commit()
    return s


def _make_run_segment(session, run_id: str, segment_def_id: str, index: int = 0, **kwargs) -> RunSegment:
    defaults = dict(
        run_id=run_id,
        segment_def_id=segment_def_id,
        index=index,
        status=SegmentStatus.pending,
    )
    defaults.update(kwargs)
    rs = RunSegment(**defaults)
    session.add(rs)
    session.commit()
    return rs


def _make_run(session, project_id: str, **kwargs) -> Run:
    defaults = dict(
        project_id=project_id,
        prompt="swap character",
        resolution="720p",
        status=RunStatus.created,
        reference_image_urls=[],
    )
    defaults.update(kwargs)
    r = Run(**defaults)
    session.add(r)
    session.commit()
    return r


def _make_ffmpeg_video(path: str) -> None:
    """Generate a tiny 1-second H.264 video at *path* using ffmpeg."""
    _make_ffmpeg_video_with_duration(path, 1)


def _make_ffmpeg_video_with_duration(path: str, duration_sec: float) -> None:
    """Like `_make_ffmpeg_video`, with a caller-chosen duration — needed to
    build a real, shorter-than-another-real-video source for a tail-edit
    bounds test (create_tail_edit ffprobes a caption source, so a fake byte
    string won't do)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=blue:size=64x64:rate=10:duration={duration_sec}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            path,
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# POST /api/v2/projects
# ---------------------------------------------------------------------------


class TestCreateProject:
    def test_upload_video_returns_201(self, spy_client, SessionFactory):
        client, spy = spy_client
        response = client.post(
            "/api/v2/projects",
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201
        body = response.json()
        assert "project_id" in body
        assert body["status"] == "analyzing"  # AICP-350: counted at admission
        # enqueue_analyze_project called exactly once with the project id
        assert spy["analyze_project"] == [body["project_id"]]

        # project row exists in DB
        session = SessionFactory()
        project = session.get(VideoProject, body["project_id"])
        session.close()
        assert project is not None
        assert project.source_type == "upload"

    def test_gdrive_link_variant(self, spy_client):
        client, spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view"},
        )
        assert response.status_code == 201
        body = response.json()
        assert "project_id" in body
        assert spy["analyze_project"] == [body["project_id"]]

    def test_neither_source_is_400(self, client):
        response = client.post("/api/v2/projects")
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"].lower()

    def test_both_sources_is_400(self, client):
        response = client.post(
            "/api/v2/projects",
            data={"gdrive_link": "https://drive.google.com/file/d/X/view"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /api/v2/projects  and  GET /api/v2/projects/{id}
# ---------------------------------------------------------------------------


class TestGetProject:
    def test_get_existing_project_200(self, client, db_session):
        project = _make_project(db_session)
        response = client.get(f"/api/v2/projects/{project.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == project.id
        assert body["status"] == "created"

    def test_get_missing_project_404(self, client):
        response = client.get("/api/v2/projects/does-not-exist-id")
        assert response.status_code == 404

    def test_list_projects_returns_array(self, client, db_session):
        _make_project(db_session)
        response = client.get("/api/v2/projects")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert "id" in body[0]
        assert "status" in body[0]


# ---------------------------------------------------------------------------
# GET /api/v2/projects/{pid}/segments  and  PATCH
# ---------------------------------------------------------------------------


class TestProjectSegments:
    def test_get_segments_returns_list(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0)
        _make_segment_def(db_session, project.id, 1)

        response = client.get(f"/api/v2/projects/{project.id}/segments")
        assert response.status_code == 200
        segs = response.json()
        assert len(segs) == 2
        assert segs[0]["index"] < segs[1]["index"]

    def test_get_segments_404_for_missing_project(self, client):
        response = client.get("/api/v2/projects/no-such-project/segments")
        assert response.status_code == 404

    def test_patch_segments_edits_field(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0, action="keep")

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap", "end_sec": 7.5}]},
        )
        assert response.status_code == 200
        segs = response.json()
        assert len(segs) == 1
        assert segs[0]["action"] == "swap"
        assert segs[0]["end_sec"] == pytest.approx(7.5)

    def test_patch_segments_renumbers_by_start_sec(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        s0 = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=5.0)
        s1 = _make_segment_def(db_session, project.id, 1, start_sec=5.0, end_sec=10.0)

        # Move s1 to start before s0
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s1.id, "start_sec": -1.0, "end_sec": 4.0}]},
        )
        assert response.status_code == 200
        segs = response.json()
        assert segs[0]["id"] == s1.id
        assert segs[0]["index"] == 0
        assert segs[1]["id"] == s0.id
        assert segs[1]["index"] == 1

    def test_patch_on_non_ready_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.analyzing)
        seg = _make_segment_def(db_session, project.id, 0)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap"}]},
        )
        assert response.status_code == 409

    def test_patch_on_created_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        seg = _make_segment_def(db_session, project.id, 0)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap"}]},
        )
        assert response.status_code == 409


class TestSegmentEditBlockedWhileRunActive:
    """Editing segments under an in-flight run cascade-deletes its RunSegments.

    Regression guard for run fdc49979 (2026-08-12): a segment edit made four
    minutes into a run killed the poll loop mid-download and left the run
    permanently at `incomplete`, its Seedance credits already spent.
    """

    @pytest.mark.parametrize(
        "status",
        [
            RunStatus.queued,
            RunStatus.processing,
            RunStatus.stitching,
            RunStatus.delivering,
        ],
    )
    def test_patch_blocked_for_each_active_status(
        self, client, db_session, status
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0)
        _make_run(db_session, project.id, status=status)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "keep"}]},
        )
        assert response.status_code == 409
        assert "in flight" in response.json()["detail"]

    def test_deletes_blocked_too(self, client, db_session):
        """The delete path is the one that actually cascades."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        s0 = _make_segment_def(db_session, project.id, 0)
        s1 = _make_segment_def(db_session, project.id, 1)
        _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"deletes": [s1.id]},
        )
        assert response.status_code == 409
        # Nothing was applied.
        db_session.expire_all()
        assert db_session.get(SegmentDef, s0.id) is not None
        assert db_session.get(SegmentDef, s1.id) is not None

    def test_detail_names_the_blocking_run(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0)
        _make_run(
            db_session, project.id, name="Q-00965", status=RunStatus.processing
        )

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "keep"}]},
        )
        assert response.status_code == 409
        assert "Q-00965" in response.json()["detail"]

    def test_unnamed_blocking_run_falls_back_to_short_id(
        self, client, db_session
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "keep"}]},
        )
        assert response.status_code == 409
        assert run.id[:8] in response.json()["detail"]

    @pytest.mark.parametrize(
        "status",
        [
            RunStatus.created,
            RunStatus.done,
            RunStatus.failed,
            RunStatus.incomplete,
        ],
    )
    def test_terminal_runs_do_not_block(self, client, db_session, status):
        """Nothing is processing these, so editing stays available."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0)
        _make_run(db_session, project.id, status=status)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "keep"}]},
        )
        assert response.status_code == 200

    def test_active_run_on_another_project_does_not_block(
        self, client, db_session
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0)
        other = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(db_session, other.id, status=RunStatus.processing)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "keep"}]},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TR8: linked/contiguous segment boundaries
# ---------------------------------------------------------------------------


class TestLinkedBoundaries:
    """Tests for TR8: _normalize_partition and contiguous partition invariants."""

    def _make_contiguous_project(self, db_session, duration=30.0, n=3):
        """Create a ready project with n contiguous segments covering [0, duration]."""
        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            duration_sec=duration,
        )
        step = duration / n
        segs = []
        for i in range(n):
            seg = _make_segment_def(
                db_session,
                project.id,
                index=i,
                start_sec=round(i * step, 6),
                end_sec=round((i + 1) * step, 6),
            )
            segs.append(seg)
        return project, segs

    def test_lengthen_seg0_shifts_seg1_start(self, client, db_session):
        """Lengthening seg0's end by +1.0 s → seg1.start shifts by +1.0 s,
        seg1 is 1.0 s shorter, all later boundaries unchanged, total = [0, 30]."""
        project, segs = self._make_contiguous_project(db_session, duration=30.0, n=3)
        # segs: [0..10], [10..20], [20..30]
        s0, s1, s2 = segs

        new_s0_end = 11.0  # extended by 1.0
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s0.id, "end_sec": new_s0_end}]},
        )
        assert response.status_code == 200, response.json()
        result = response.json()

        # Sort by index for deterministic access
        result.sort(key=lambda s: s["index"])
        r0, r1, r2 = result

        assert r0["start_sec"] == pytest.approx(0.0)
        assert r0["end_sec"] == pytest.approx(11.0)

        # seg1 start must equal seg0 new end
        assert r1["start_sec"] == pytest.approx(11.0)
        # seg1 end is unchanged (only the one boundary moved)
        assert r1["end_sec"] == pytest.approx(20.0)

        # seg2 boundaries completely unchanged
        assert r2["start_sec"] == pytest.approx(20.0)
        assert r2["end_sec"] == pytest.approx(30.0)

        # Contiguity invariant
        assert r0["end_sec"] == pytest.approx(r1["start_sec"])
        assert r1["end_sec"] == pytest.approx(r2["start_sec"])
        assert r0["start_sec"] == pytest.approx(0.0)
        assert r2["end_sec"] == pytest.approx(30.0)

        # Indices are 0..n-1
        assert [s["index"] for s in result] == [0, 1, 2]

    def test_contiguity_invariant_after_edit(self, client, db_session):
        """For any edit, seg[i].end == seg[i+1].start for all i,
        seg[0].start == 0, seg[-1].end == duration."""
        project, segs = self._make_contiguous_project(db_session, duration=20.0, n=4)
        s0, s1, s2, s3 = segs

        # Edit middle boundary
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s1.id, "end_sec": 8.5}]},
        )
        assert response.status_code == 200
        result = sorted(response.json(), key=lambda s: s["index"])

        assert result[0]["start_sec"] == pytest.approx(0.0)
        assert result[-1]["end_sec"] == pytest.approx(20.0)
        for i in range(len(result) - 1):
            assert result[i]["end_sec"] == pytest.approx(result[i + 1]["start_sec"]), \
                f"Contiguity broken between index {i} and {i+1}"

    def test_collapsing_a_segment_drops_it(self, client, db_session):
        """Collapsing a segment (end <= start) DROPS it (200), leaving a
        contiguous partition — this is how the editor 'deletes' a keep."""
        project, segs = self._make_contiguous_project(db_session, duration=30.0, n=3)
        s0, s1, s2 = segs

        # Collapse s0 (end=0). It should be dropped, not rejected.
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s0.id, "end_sec": 0.0}]},
        )
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2  # s0 dropped
        # Contiguous coverage of [0, 30]
        assert result[0]["start_sec"] == pytest.approx(0.0)
        assert result[-1]["end_sec"] == pytest.approx(30.0)
        for a, b in zip(result, result[1:]):
            assert a["end_sec"] == pytest.approx(b["start_sec"])

    def test_end_beyond_duration_is_clamped(self, client, db_session):
        """An end_sec beyond project duration is clamped to duration (200);
        segments pushed past the end collapse and are dropped."""
        project, segs = self._make_contiguous_project(db_session, duration=30.0, n=3)
        s0, s1, s2 = segs

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s1.id, "end_sec": 35.0}]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result[-1]["end_sec"] == pytest.approx(30.0)
        for a, b in zip(result, result[1:]):
            assert a["end_sec"] == pytest.approx(b["start_sec"])

    def test_patch_still_409_when_not_ready(self, client, db_session):
        """PATCH on a non-ready project is still 409."""
        project = _make_project(
            db_session,
            status=ProjectStatus.analyzing,
            duration_sec=30.0,
        )
        seg = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "end_sec": 12.0}]},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# TR8: _normalize_partition unit tests (pure helper)
# ---------------------------------------------------------------------------


class TestNormalizePartition:
    """Direct unit tests for the _normalize_partition helper."""

    def setup_method(self):
        """Import the helper fresh each test."""
        import app.api_v2 as m
        self.normalize = m._normalize_partition

        class FakeDB:
            def __init__(self):
                self.deleted = []
            def delete(self, x):
                self.deleted.append(x)

        self.db = FakeDB()

    def _fake_seg(self, idx, start, end, seg_id=None):
        """Minimal object with the fields _normalize_partition reads/writes."""
        class FakeSeg:
            pass
        s = FakeSeg()
        s.id = seg_id or f"seg-{idx}"
        s.index = idx
        s.start_sec = float(start)
        s.end_sec = float(end)
        return s

    def test_contiguous_input_unchanged(self):
        segs = [
            self._fake_seg(0, 0, 10),
            self._fake_seg(1, 10, 20),
            self._fake_seg(2, 20, 30),
        ]
        self.normalize(segs, 30.0, self.db)
        assert self.db.deleted == []
        assert segs[0].start_sec == pytest.approx(0.0)
        assert segs[0].end_sec == pytest.approx(10.0)
        assert segs[1].start_sec == pytest.approx(10.0)
        assert segs[1].end_sec == pytest.approx(20.0)
        assert segs[2].start_sec == pytest.approx(20.0)
        assert segs[2].end_sec == pytest.approx(30.0)

    def test_derives_starts_from_ends(self):
        """Starts are derived from the running cursor; only ends matter."""
        segs = [
            self._fake_seg(0, 0, 10),
            self._fake_seg(1, 9999, 20),  # stale start_sec; end=20 is the boundary
        ]
        self.normalize(segs, 30.0, self.db)
        assert segs[0].start_sec == pytest.approx(0.0)
        assert segs[0].end_sec == pytest.approx(10.0)
        assert segs[1].start_sec == pytest.approx(10.0)
        assert segs[1].end_sec == pytest.approx(30.0)  # last extended to duration

    def test_indices_reassigned_zero_based(self):
        segs = [
            self._fake_seg(5, 0, 10),
            self._fake_seg(7, 10, 20),
            self._fake_seg(3, 20, 30),
        ]
        self.normalize(segs, 30.0, self.db)
        indices = sorted(s.index for s in segs)
        assert indices == [0, 1, 2]

    def test_empty_list_is_noop(self):
        self.normalize([], 30.0, self.db)  # should not raise

    def test_collapsed_segment_is_dropped_not_rejected(self):
        """A zero-duration segment is DROPPED (deleted), not a 400 — and the
        remaining segments stay contiguous. This is the 'delete the keep by
        collapsing it' behaviour the editor relies on."""
        zero = self._fake_seg(1, 6.0, 6.0, seg_id="keep")   # collapsed
        s0 = self._fake_seg(0, 0, 6.0, seg_id="a")
        s2 = self._fake_seg(2, 6.0, 30.0, seg_id="b")
        segs = [s0, zero, s2]
        self.normalize(segs, 30.0, self.db)
        assert self.db.deleted == [zero]            # the collapsed one dropped
        assert s0.start_sec == pytest.approx(0.0) and s0.end_sec == pytest.approx(6.0)
        assert s2.start_sec == pytest.approx(6.0) and s2.end_sec == pytest.approx(30.0)
        assert s0.index == 0 and s2.index == 1      # reindexed over the gap

    def test_negative_duration_segment_dropped(self):
        """A neighbour extended over a segment (start>end) drops that segment."""
        s0 = self._fake_seg(0, 0, 6.5, seg_id="a")
        bad = self._fake_seg(1, 6.5, 6.0, seg_id="keep")  # end<start after edit
        s2 = self._fake_seg(2, 6.0, 30.0, seg_id="b")
        segs = [s0, bad, s2]
        self.normalize(segs, 30.0, self.db)
        assert bad in self.db.deleted
        assert s0.end_sec == pytest.approx(6.5)
        assert s2.start_sec == pytest.approx(6.5) and s2.end_sec == pytest.approx(30.0)

    def test_end_beyond_duration_is_clamped(self):
        """An end beyond duration is clamped; segments past it are dropped."""
        s0 = self._fake_seg(0, 0, 50, seg_id="a")   # end > duration 30
        s1 = self._fake_seg(1, 60, 80, seg_id="b")
        segs = [s0, s1]
        self.normalize(segs, 30.0, self.db)
        assert s0.start_sec == pytest.approx(0.0) and s0.end_sec == pytest.approx(30.0)
        assert s1 in self.db.deleted

    def test_all_collapsed_raises_400(self):
        from fastapi import HTTPException
        segs = [self._fake_seg(0, 0, 0)]
        with pytest.raises(HTTPException) as exc_info:
            self.normalize(segs, 30.0, self.db)
        assert exc_info.value.status_code == 400

    def test_single_segment_spans_full_duration(self):
        segs = [self._fake_seg(0, 5, 25)]  # start/end arbitrary; will be pinned
        self.normalize(segs, 30.0, self.db)
        assert segs[0].start_sec == pytest.approx(0.0)
        assert segs[0].end_sec == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# POST /api/v2/projects/{pid}/runs
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_create_run_on_ready_project_returns_201(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap to redhead", "resolution": "720p"},
        )
        assert response.status_code == 201
        body = response.json()
        assert "run_id" in body
        assert body["status"] == "queued"
        assert spy["process_run"] == [body["run_id"]]

        # Verify Run in DB
        session = SessionFactory()
        run = session.get(Run, body["run_id"])
        session.close()
        assert run is not None
        assert run.project_id == project.id
        assert run.prompt == "swap to redhead"
        # status should be queued after create_run transitions it
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        assert status_val == "queued"

    def test_create_run_on_non_ready_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 409

    def test_create_run_on_analyzing_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.analyzing)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 409

    def test_too_many_reference_images_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "reference_urls": (
                    "https://example.com/ref1.jpg,"
                    "https://example.com/ref2.jpg,"
                    "https://example.com/ref3.jpg,"
                    "https://example.com/ref4.jpg,"
                    "https://example.com/ref5.jpg,"
                    "https://example.com/ref6.jpg"
                ),
            },
        )
        assert response.status_code == 400
        assert "too many reference" in response.json()["detail"].lower()

    def test_bad_resolution_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "4k"},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    def test_create_run_missing_project_is_404(self, client):
        response = client.post(
            "/api/v2/projects/no-such-project/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 404

    def test_create_run_with_name_stored(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap character", "name": "Redhead woman"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run.name == "Redhead woman"

    def test_create_run_without_resolution_uses_default(self, spy_client, SessionFactory):
        """AICP-391: a launch that sends no resolution (API, assistant) gets
        settings.DEFAULT_RESOLUTION — 720p unless the env overrides it."""
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap character"},
        )
        assert response.status_code == 201
        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        session.close()
        assert run.resolution == settings.DEFAULT_RESOLUTION

    def test_create_run_with_reference_urls(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "reference_urls": (
                    "https://example.com/ref1.jpg,"
                    "https://example.com/ref2.jpg"
                ),
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert len(run.reference_image_urls) == 2

    def test_create_run_preprocess_enters_preprocessing(
        self, spy_client, SessionFactory
    ):
        """preprocess_enabled=True → status=preprocessing + enqueue_preprocess_run.

        R1-new: prompts[0] is ALWAYS "" (char ref pass-through). The textarea
        lines correspond to TRIMS (indices 1..N-1); the API prepends "" for
        index 0. The character ref is an explicit, required field
        (preprocess_reference_url here)."""
        import json as _json

        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": ",".join([
                    _staged_frame(project.id, "trim1.jpg"),
                    _staged_frame(project.id, "trim2.jpg"),
                ]),
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(
                    ["make a redshirt", "make a blueshirt"]
                ),
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        assert response.json()["status"] == "preprocessing"
        # preprocess entry point called, normal submit NOT called.
        assert spy["preprocess_run"] == [run_id]
        assert spy["process_run"] == []

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run.preprocess_enabled is True
        # 1 char ref + 2 trims; 2 trim prompts → ["", "make a redshirt", "make a blueshirt"].
        assert run.preprocess_image_prompts == ["", "make a redshirt", "make a blueshirt"]
        # reference_image_urls[0] = char ref, [1:] = trims.
        assert run.reference_image_urls[0] == "https://x.test/char.jpg"
        assert run.reference_image_urls[1:] == [
            _staged_frame(project.id, "trim1.jpg"),
            _staged_frame(project.id, "trim2.jpg"),
        ]
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        assert status_val == "preprocessing"

    def test_create_run_preprocess_prompts_fewer_than_refs_are_padded(
        self, client, db_session
    ):
        """R1-new: fewer trim prompts than trims → padded with pass-through
        ("") instead of a 400 (lenient). A single trim prompt with multiple
        trims is reused for every trim (per_image mode). prompts[0] is always
        "" (char ref)."""
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": ",".join([
                    _staged_frame(project.id, "t1.jpg"),
                    _staged_frame(project.id, "t2.jpg"),
                    _staged_frame(project.id, "t3.jpg"),
                ]),
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(["only one", "two"]),
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        session = db_session
        run = session.get(Run, run_id)
        # 1 char ref + 3 trims; 2 trim prompts + 1 padding pass-through.
        assert run.preprocess_image_prompts == ["", "only one", "two", ""]
        # Default mode is per_image.
        assert run.preprocess_mode == "per_image"

    def test_create_run_preprocess_single_prompt_reused_for_all_trims(
        self, client, db_session
    ):
        """R1-new: a single trim prompt with multiple trims is reused for every
        trim (per_image mode). prompts[0] is always "" (char ref)."""
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": ",".join([
                    _staged_frame(project.id, "t1.jpg"),
                    _staged_frame(project.id, "t2.jpg"),
                ]),
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(["only one"]),
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        run = db_session.get(Run, run_id)
        # 1 char ref + 2 trims; 1 trim prompt reused for both trims.
        assert run.preprocess_image_prompts == ["", "only one", "only one"]

    def test_create_run_preprocess_all_as_refs_takes_first_prompt(
        self, client, db_session
    ):
        """R1-new: all_as_refs mode stores a length-1 prompts list (the single
        trim prompt); the pipeline reads prompts[0] as the single prompt."""
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": _staged_frame(project.id, "t1.jpg"),
                "preprocess_enabled": "true",
                "preprocess_mode": "all_as_refs",
                "preprocess_image_prompts": _json.dumps(["combine", "ignored"]),
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        run = db_session.get(Run, run_id)
        assert run.preprocess_mode == "all_as_refs"
        # R1-new: length-1 list — the single trim prompt.
        assert run.preprocess_image_prompts == ["combine"]

    def test_create_run_preprocess_invalid_mode_is_400(self, client, db_session):
        """preprocess_mode not in allowed set → 400."""
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": "https://x.test/a.jpg",
                "preprocess_enabled": "true",
                "preprocess_mode": "bogus",
                "preprocess_image_prompts": _json.dumps(["x"]),
            },
        )
        assert response.status_code == 400
        assert "preprocess_mode must be one of" in response.json()["detail"]

    def test_create_run_preprocess_invalid_json_is_400(self, client, db_session):
        """preprocess_image_prompts not valid JSON → 400."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": "https://x.test/a.jpg",
                "preprocess_enabled": "true",
                "preprocess_image_prompts": "not-json",
            },
        )
        assert response.status_code == 400
        assert "preprocess_image_prompts must be valid JSON" in response.json()["detail"]

    def test_create_run_preprocess_prompts_not_list_is_400(self, client, db_session):
        """preprocess_image_prompts JSON but not a list → 400."""
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": "https://x.test/a.jpg",
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps({"a": 1}),
            },
        )
        assert response.status_code == 400
        assert "must be a JSON list" in response.json()["detail"]

    def test_create_run_preprocess_with_char_ref_file_and_trims(
        self, spy_client, SessionFactory
    ):
        """Explicit Character reference photo (file upload) + 2 trims →
        reference_image_urls[0] = saved char ref path, [1:] = trims;
        preprocess_image_prompts = ["", trim1, trim2]; status preprocessing."""
        import json as _json
        import os

        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "reference_urls": ",".join([
                    _staged_frame(project.id, "trim1.jpg"),
                    _staged_frame(project.id, "trim2.jpg"),
                ]),
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(["p1", "p2"]),
            },
            files={
                "preprocess_reference_file": ("char.jpg", io.BytesIO(b"char-bytes"), "image/jpeg"),
            },
        )
        assert response.status_code == 201, response.text
        run_id = response.json()["run_id"]
        assert response.json()["status"] == "preprocessing"
        assert spy["preprocess_run"] == [run_id]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run.preprocess_image_prompts == ["", "p1", "p2"]
        # reference_image_urls[0] is the saved char ref (a local path in the
        # run's references dir); [1:] are the trims.
        char_ref = run.reference_image_urls[0]
        assert os.sep + "references" + os.sep in char_ref
        assert char_ref.endswith("char.jpg")
        assert run.reference_image_urls[1:] == [
            _staged_frame(project.id, "trim1.jpg"),
            _staged_frame(project.id, "trim2.jpg"),
        ]
        assert os.path.exists(char_ref)

    def test_create_run_preprocess_missing_char_ref_is_400(self, client, db_session):
        """preprocess_enabled=True with NO character reference photo (no file,
        no url) → 400 with the character-ref message."""
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "reference_urls": "https://x.test/trim1.jpg,https://x.test/trim2.jpg",
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(["p1", "p2"]),
            },
        )
        assert response.status_code == 400
        assert "Character reference photo" in response.json()["detail"]

    def test_create_run_preprocess_char_ref_only_no_trims_is_allowed(
        self, spy_client, SessionFactory
    ):
        """Edge case: preprocessing enabled with a char ref and 0 trims is
        allowed (pipeline no-ops on len(refs)<=1). Prompts = [""] (just the
        char ref slot)."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "preprocess_enabled": "true",
            },
        )
        assert response.status_code == 201, response.text
        run_id = response.json()["run_id"]
        assert response.json()["status"] == "preprocessing"
        assert spy["preprocess_run"] == [run_id]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run.reference_image_urls == ["https://x.test/char.jpg"]
        assert run.preprocess_image_prompts == [""]

    def test_create_run_preprocess_off_uses_submit_path(self, spy_client, SessionFactory):
        """preprocess_enabled absent/false → status=queued + enqueue_process_run."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run_id]
        assert spy["preprocess_run"] == []


# ---------------------------------------------------------------------------
# Model selection (Seedance vs Gemini Omni) on Run creation
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Tests for the per-run model field and its model-specific resolution rules."""

    def test_create_run_with_gemini_model_persists(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "gemini-omni", "resolution": "1080p"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run.model == "gemini-omni"
        assert run.resolution == "1080p"

        # Visible in GET /api/v2/runs/{rid}
        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.json()["model"] == "gemini-omni"

    def test_create_run_default_model_is_seedance(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.json()["model"] == "seedance"

    def test_invalid_model_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "midjourney"},
        )
        assert response.status_code == 400
        assert "model" in response.json()["detail"].lower()

    def test_gemini_allows_4k(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "gemini-omni", "resolution": "4k"},
        )
        assert response.status_code == 201

    def test_gemini_rejects_480p(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "gemini-omni", "resolution": "480p"},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    def test_seedance_rejects_4k(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "seedance", "resolution": "4k"},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    @pytest.mark.parametrize(
        "variant", ["seedance-fast", "seedance-mini", "seedance-2-5"]
    )
    @pytest.mark.parametrize("resolution", ["480p", "720p"])
    def test_seedance_variant_accepts_480p_and_720p(
        self, spy_client, SessionFactory, variant, resolution
    ):
        """Every Seedance variant accepts the two shared low tiers.

        Seedance 2.5 is in this group too, and since kie.ai added 1080p on
        2026-08-18 it also accepts that — covered separately by
        test_seedance_2_5_accepts_1080p.
        """
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": variant, "resolution": resolution},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.json()["model"] == variant
        assert get_resp.json()["resolution"] == resolution

    @pytest.mark.parametrize(
        "variant,resolution",
        [
            ("seedance-fast", "1080p"),
            ("seedance-fast", "4k"),
            ("seedance-mini", "1080p"),
            ("seedance-mini", "4k"),
            # 2.5 gained 1080p on 2026-08-18 but still has no 4k tier.
            ("seedance-2-5", "4k"),
        ],
    )
    def test_seedance_variant_rejects_unsupported_resolution(
        self, client, db_session, variant, resolution
    ):
        """The Seedance variants reject the tiers above their own ceiling."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": variant, "resolution": resolution},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    def test_seedance_2_5_accepts_1080p(self, spy_client, SessionFactory):
        """2.5 does 1080p since kie.ai added the tier on 2026-08-18.

        It is the regression guard for the resolution that used to be coerced
        down: a 1080p run on 2.5 must persist as 1080p, not 720p.
        """
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "model": "seedance-2-5",
                "resolution": "1080p",
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        assert client.get(f"/api/v2/runs/{run_id}").json()["resolution"] == "1080p"


# ---------------------------------------------------------------------------
# TR7: audio_mode field on Run creation and RunResponse
# ---------------------------------------------------------------------------


class TestAudioMode:
    """Tests for TR7: per-run audio_mode field."""

    def test_create_run_with_audio_mode_seedance_persists(
        self, spy_client, SessionFactory
    ):
        """Creating a run with audio_mode=seedance persists that value and shows in RunResponse."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "audio_mode": "seedance"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        # Verify persisted in DB.
        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run is not None
        audio_mode_val = run.audio_mode.value if hasattr(run.audio_mode, "value") else str(run.audio_mode)
        assert audio_mode_val == "seedance"

        # Verify visible in GET /api/v2/runs/{rid}
        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["audio_mode"] == "seedance"

    def test_create_run_default_audio_mode_is_original(
        self, spy_client, SessionFactory
    ):
        """When audio_mode is not provided, it defaults to 'original'."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["audio_mode"] == "original"

    def test_create_run_invalid_audio_mode_is_400(self, client, db_session):
        """Passing an unrecognised audio_mode value returns HTTP 400."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "audio_mode": "dolby"},
        )
        assert response.status_code == 400
        assert "audio_mode" in response.json()["detail"].lower()

    def test_run_response_includes_audio_mode_field(self, client, db_session):
        """GET /api/v2/runs/{rid} returns audio_mode in the response body."""
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, audio_mode="original")

        response = client.get(f"/api/v2/runs/{run.id}")
        assert response.status_code == 200
        body = response.json()
        assert "audio_mode" in body
        assert body["audio_mode"] in ("original", "seedance")


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}  and  GET /api/v2/projects/{pid}/runs
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_get_existing_run_200(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id)

        response = client.get(f"/api/v2/runs/{run.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == run.id
        assert body["project_id"] == project.id

    def test_get_missing_run_404(self, client):
        response = client.get("/api/v2/runs/does-not-exist-id")
        assert response.status_code == 404

    def test_list_project_runs(self, client, db_session):
        project = _make_project(db_session)
        _make_run(db_session, project.id)
        _make_run(db_session, project.id)

        response = client.get(f"/api/v2/projects/{project.id}/runs")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 2

    def test_list_project_runs_404_missing_project(self, client):
        response = client.get("/api/v2/projects/no-such/runs")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}/result
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_result_on_non_done_run_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.get(f"/api/v2/runs/{run.id}/result")
        assert response.status_code == 409

    def test_result_on_queued_run_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.queued)

        response = client.get(f"/api/v2/runs/{run.id}/result")
        assert response.status_code == 409

    def test_result_on_done_run_returns_file(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")

        project = _make_project(db_session)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

        response = client.get(f"/api/v2/runs/{run.id}/result")
        assert response.status_code == 200
        assert response.content == b"\x00VIDEO\xff"
        # Cache-Control changed from no-store to a private cacheable policy
        # (URLs carry a ?v=mtime-size buster so immutable is safe here).
        assert "private" in response.headers["cache-control"]
        assert "no-store" not in response.headers["cache-control"]

    def test_result_info_on_done_run(self, client, db_session, tmp_path):
        result_file = tmp_path / "output.mp4"
        result_file.write_bytes(b"\x00\xff")

        project = _make_project(db_session)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_gdrive_file_id="GDRIVE_RUN_123",
        )

        response = client.get(f"/api/v2/runs/{run.id}/result/info")
        assert response.status_code == 200
        body = response.json()
        assert body["result_gdrive_file_id"] == "GDRIVE_RUN_123"
        assert "drive.google.com" in body["result_gdrive_link"]

    def test_result_info_on_non_done_run_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.get(f"/api/v2/runs/{run.id}/result/info")
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}/segments/{rsid}/result
# ---------------------------------------------------------------------------


class TestRunSegmentResult:
    def _seg_with_clip(self, session, tmp_path, *, run_status=RunStatus.done, **rs_kwargs):
        clip = tmp_path / "result_0000.mp4"
        clip.write_bytes(b"\x00SEGMENT\xff")
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=run_status)
        rs = _make_run_segment(
            session, run.id, sd.id,
            status=SegmentStatus.completed,
            local_result_path=str(clip),
            **rs_kwargs,
        )
        return run, rs

    def test_returns_the_generated_clip(self, client, db_session, tmp_path):
        run, rs = self._seg_with_clip(db_session, tmp_path)

        response = client.get(f"/api/v2/runs/{run.id}/segments/{rs.id}/result")
        assert response.status_code == 200
        assert response.content == b"\x00SEGMENT\xff"
        # Inline by default: a plain link opens the clip instead of saving it.
        assert "attachment" not in response.headers.get("content-disposition", "")

    def test_download_flag_attaches_with_the_sheet_job_id(
        self, client, db_session, tmp_path
    ):
        run, rs = self._seg_with_clip(db_session, tmp_path)
        db_session.get(Run, run.id).language = "ja"
        db_session.commit()

        response = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/result", params={"download": 1}
        )
        assert response.status_code == 200
        disposition = response.headers["content-disposition"]
        assert "attachment" in disposition
        assert f"{run.id[:8]}-ja_seg00.mp4" in disposition

    def test_available_while_the_run_is_still_processing(
        self, client, db_session, tmp_path
    ):
        """No run-status gate: a landed segment is watchable mid-run, which is
        the whole point — spotting a bad generation before the rest are paid
        for."""
        run, rs = self._seg_with_clip(
            db_session, tmp_path, run_status=RunStatus.processing
        )

        response = client.get(f"/api/v2/runs/{run.id}/segments/{rs.id}/result")
        assert response.status_code == 200

    def test_404_when_no_clip_on_disk(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.generating
        )

        response = client.get(f"/api/v2/runs/{run.id}/segments/{rs.id}/result")
        assert response.status_code == 404

    def test_404_when_the_recorded_path_is_gone(self, client, db_session, tmp_path):
        run, rs = self._seg_with_clip(db_session, tmp_path)
        os.remove(str(tmp_path / "result_0000.mp4"))

        response = client.get(f"/api/v2/runs/{run.id}/segments/{rs.id}/result")
        assert response.status_code == 404

    def test_404_for_a_segment_of_another_run(self, client, db_session, tmp_path):
        run, rs = self._seg_with_clip(db_session, tmp_path)
        other = _make_run(db_session, run.project_id, status=RunStatus.done)

        response = client.get(f"/api/v2/runs/{other.id}/segments/{rs.id}/result")
        assert response.status_code == 404

    def test_404_for_unknown_run(self, client, db_session, tmp_path):
        _, rs = self._seg_with_clip(db_session, tmp_path)

        response = client.get(f"/api/v2/runs/no-such-run/segments/{rs.id}/result")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v2/projects/{pid}/frame
# ---------------------------------------------------------------------------


class TestProjectFrame:
    def test_frame_endpoint_returns_jpeg(self, client, db_session, tmp_path):
        """Generate a real 1-second video with ffmpeg and verify the frame endpoint."""
        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)

        project = _make_project(
            db_session,
            source_local_path=video_path,
            status=ProjectStatus.ready,
        )

        response = client.get(f"/api/v2/projects/{project.id}/frame?t=0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 100  # non-empty JPEG

    def test_frame_endpoint_clamps_t_beyond_duration(self, client, db_session, tmp_path):
        """t past EOF is clamped to duration - 0.1 — ffmpeg would otherwise exit
        rc=0 having encoded nothing and a 0-byte 'jpeg' would be cached forever."""
        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)  # 1-second video

        project = _make_project(
            db_session,
            source_local_path=video_path,
            status=ProjectStatus.ready,
            duration_sec=1.0,
        )

        response = client.get(f"/api/v2/projects/{project.id}/frame?t=999")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 100  # real frame, not an empty cache file

    def test_frame_endpoint_clamps_negative_t(self, client, db_session, tmp_path):
        """Negative t is clamped to 0 (avoids odd negative cache keys)."""
        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)

        project = _make_project(
            db_session,
            source_local_path=video_path,
            status=ProjectStatus.ready,
            duration_sec=1.0,
        )

        response = client.get(f"/api/v2/projects/{project.id}/frame?t=-5")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 100

    def test_frame_endpoint_404_missing_source(self, client, db_session):
        """Project exists but source file not on disk → 404."""
        project = _make_project(
            db_session,
            source_local_path="/tmp/nonexistent_source_xyz.mp4",
            status=ProjectStatus.ready,
        )
        response = client.get(f"/api/v2/projects/{project.id}/frame?t=0")
        assert response.status_code == 404

    def test_frame_endpoint_404_no_source(self, client, db_session):
        """Project with no source_local_path → 404."""
        project = _make_project(db_session)
        response = client.get(f"/api/v2/projects/{project.id}/frame?t=0")
        assert response.status_code == 404

    def test_source_endpoint_returns_video(self, client, db_session, tmp_path):
        """The source endpoint streams the original video file."""
        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)
        project = _make_project(
            db_session, source_local_path=video_path, status=ProjectStatus.ready
        )
        response = client.get(f"/api/v2/projects/{project.id}/source")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert len(response.content) > 100

    def test_source_endpoint_404_missing(self, client, db_session):
        """No source file on disk → 404."""
        project = _make_project(
            db_session, source_local_path="/tmp/nope_xyz.mp4", status=ProjectStatus.ready
        )
        response = client.get(f"/api/v2/projects/{project.id}/source")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v2/projects/{pid}/stage-frame
# ---------------------------------------------------------------------------


class TestProjectStageFrame:
    def test_stage_frame_returns_path_and_saves_file(self, client, db_session, tmp_path):
        """stage-frame extracts a frame, copies it into staged_frames/, returns JSON."""
        import os

        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)
        project = _make_project(
            db_session, source_local_path=video_path, status=ProjectStatus.ready,
            duration_sec=1.0,
        )

        response = client.post(f"/api/v2/projects/{project.id}/stage-frame?t=0")
        assert response.status_code == 200
        body = response.json()
        assert "path" in body
        assert os.path.exists(body["path"])
        assert body["path"].endswith(".jpg")
        assert body["t"] == 0.0
        # The staged file lives under the project's staged_frames dir.
        assert os.sep + "staged_frames" + os.sep in body["path"]

    def test_stage_frame_name_carries_the_timestamp(self, client, db_session, tmp_path):
        """The staged filename IS where the trim's timestamp lives — the run's
        reference list keeps no other record of it, and the Reference images
        block reads it back off the name (storage.staged_frame_timestamp)."""
        from app.storage import staged_frame_timestamp

        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)
        project = _make_project(
            db_session, source_local_path=video_path, status=ProjectStatus.ready,
            duration_sec=1.0,
        )

        response = client.post(f"/api/v2/projects/{project.id}/stage-frame?t=0.4")
        assert response.status_code == 200
        body = response.json()
        assert body["t"] == 0.4
        assert staged_frame_timestamp(body["path"]) == 0.4

    def test_stage_frame_name_uses_the_clamped_timestamp(
        self, client, db_session, tmp_path
    ):
        """A timestamp past the end of the video is clamped to a real frame, so
        the staged name (and the label the UI shows) must say where the frame is
        actually from — not the moment that was asked for."""
        from app.storage import staged_frame_timestamp

        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)  # 1-second video
        project = _make_project(
            db_session, source_local_path=video_path, status=ProjectStatus.ready,
            duration_sec=1.0,
        )

        response = client.post(f"/api/v2/projects/{project.id}/stage-frame?t=999")
        assert response.status_code == 200
        body = response.json()
        assert body["t"] == 0.9  # duration - 0.1, quantised
        assert staged_frame_timestamp(body["path"]) == 0.9

    def test_stage_frame_404_missing_source(self, client, db_session):
        """No source on disk → 404."""
        project = _make_project(
            db_session, source_local_path="/tmp/nope_xyz.mp4", status=ProjectStatus.ready
        )
        response = client.post(f"/api/v2/projects/{project.id}/stage-frame?t=0")
        assert response.status_code == 404

    def test_stage_frame_each_call_unique(self, client, db_session, tmp_path):
        """Repeated calls at the same timestamp produce distinct files (uuid suffix)."""
        import os

        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)
        project = _make_project(
            db_session, source_local_path=video_path, status=ProjectStatus.ready,
            duration_sec=1.0,
        )

        paths = []
        for _ in range(2):
            resp = client.post(f"/api/v2/projects/{project.id}/stage-frame?t=0")
            assert resp.status_code == 200
            paths.append(resp.json()["path"])
        assert paths[0] != paths[1]
        assert all(os.path.exists(p) for p in paths)

    def test_create_run_copies_staged_frame_into_run_refs(
        self, spy_client, SessionFactory, tmp_path
    ):
        """A staged-frame path submitted as a reference_url is copied into the
        run's own references dir (durability) and the run stores the copy."""
        import os

        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(
            session, source_local_path=video_path, status=ProjectStatus.ready,
            duration_sec=1.0,
        )
        session.close()

        # Stage a frame.
        resp = client.post(f"/api/v2/projects/{project.id}/stage-frame?t=0")
        assert resp.status_code == 200
        staged_path = resp.json()["path"]
        assert os.path.exists(staged_path)

        # Create a run using the staged path as a reference URL.
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "reference_urls": staged_path},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert len(run.reference_image_urls) == 1
        stored = run.reference_image_urls[0]
        # The stored path is the run's own copy, not the staged path.
        assert stored != staged_path
        assert os.path.exists(stored)
        assert os.sep + "references" + os.sep in stored
        # Original staged file is untouched.
        assert os.path.exists(staged_path)


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}/segments
# ---------------------------------------------------------------------------


class TestRunSegments:
    def test_get_run_segments_empty(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id)

        response = client.get(f"/api/v2/runs/{run.id}/segments")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_run_segments_404_missing_run(self, client):
        response = client.get("/api/v2/runs/no-such-run/segments")
        assert response.status_code == 404

    def test_get_run_segments_with_data(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg_def = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        # Insert a RunSegment directly
        rs = RunSegment(
            run_id=run.id,
            segment_def_id=seg_def.id,
            index=0,
            status=SegmentStatus.pending,
        )
        db_session.add(rs)
        db_session.commit()

        response = client.get(f"/api/v2/runs/{run.id}/segments")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["run_id"] == run.id
        assert body[0]["segment_def_id"] == seg_def.id


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}/reference-image  and  /preprocessed-image  (R3)
# ---------------------------------------------------------------------------


class TestRunReferenceImages:
    def test_reference_image_404_missing_run(self, client):
        response = client.get("/api/v2/runs/no-such-run/reference-image")
        assert response.status_code == 404

    def test_reference_image_404_no_refs(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id)
        response = client.get(f"/api/v2/runs/{run.id}/reference-image?index=0")
        assert response.status_code == 404

    def test_reference_image_404_index_out_of_range(self, client, db_session, tmp_path):
        project = _make_project(db_session)
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG-ish bytes
        run = _make_run(db_session, project.id, reference_image_urls=[str(img)])
        response = client.get(f"/api/v2/runs/{run.id}/reference-image?index=5")
        assert response.status_code == 404
        assert "out of range" in response.json()["detail"]

    def test_reference_image_serves_file(self, client, db_session, tmp_path):
        project = _make_project(db_session)
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        run = _make_run(db_session, project.id, reference_image_urls=[str(img)])
        response = client.get(f"/api/v2/runs/{run.id}/reference-image?index=0")
        assert response.status_code == 200
        assert response.content.startswith(b"\xff\xd8\xff\xe0")

    def test_preprocessed_image_404_missing_run(self, client):
        response = client.get("/api/v2/runs/no-such-run/preprocessed-image")
        assert response.status_code == 404

    def test_preprocessed_image_404_no_pp(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id)
        response = client.get(f"/api/v2/runs/{run.id}/preprocessed-image?index=0")
        assert response.status_code == 404

    def test_preprocessed_image_serves_file(self, client, db_session, tmp_path):
        project = _make_project(db_session)
        img = tmp_path / "pp.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake-png")
        run = _make_run(
            db_session, project.id, preprocessed_reference_urls=[str(img)],
        )
        response = client.get(f"/api/v2/runs/{run.id}/preprocessed-image?index=0")
        assert response.status_code == 200
        assert response.content.startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# R2-new: POST /api/v2/runs/{rid}/approve-preprocess  and  /regenerate-preprocess
# ---------------------------------------------------------------------------


class TestApprovePreprocess:
    """R2-new: approving preprocessed refs transitions preprocess_review →
    queued and enqueues run_submit_run."""

    def test_approve_transitions_to_queued_and_enqueues_submit(
        self, spy_client, SessionFactory
    ):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = _make_run(
            session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            reference_image_urls=["https://x.test/a.jpg", "https://x.test/b.jpg"],
            preprocessed_reference_urls=["https://pp.test/a.png", "https://pp.test/b.png"],
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/approve-preprocess")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert spy["process_run"] == [run.id]

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        assert run.status == RunStatus.queued

    def test_approve_wrong_status_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(
            db_session, project.id, status=RunStatus.preprocessing,
            preprocess_enabled=True,
        )
        response = client.post(f"/api/v2/runs/{run.id}/approve-preprocess")
        assert response.status_code == 409
        assert "preprocess_review" in response.json()["detail"]

    def test_approve_missing_run_is_404(self, client):
        response = client.post("/api/v2/runs/no-such-run/approve-preprocess")
        assert response.status_code == 404

    def test_approve_with_an_empty_result_slot_is_400(self, client, db_session):
        """A hole means one image is missing — a single-image regen that failed
        or never came back. The submit phase feeds this list straight to the
        model, so it must not start half-built."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            reference_image_urls=[
                "https://x.test/a.jpg", "https://x.test/b.jpg",
                "https://x.test/c.jpg",
            ],
            preprocessed_reference_urls=["https://pp.test/a.png", None,
                                         "https://pp.test/c.png"],
        )

        response = client.post(f"/api/v2/runs/{run.id}/approve-preprocess")
        assert response.status_code == 400
        assert "#1" in response.json()["detail"]
        db_session.refresh(run)
        assert run.status == RunStatus.preprocess_review

    def test_approve_with_no_preprocess_results_at_all_is_allowed(
        self, spy_client, SessionFactory
    ):
        """An EMPTY list is not a hole: preprocessing produced nothing (every
        ref passed through), and submit falls back to the originals."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = _make_run(
            session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            reference_image_urls=["https://x.test/a.jpg"],
            preprocessed_reference_urls=None,
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/approve-preprocess")
        assert response.status_code == 200, response.text
        assert spy["process_run"] == [run.id]


class TestRegeneratePreprocess:
    """R2-new: regenerating transitions preprocess_review → preprocessing and
    enqueues run_preprocess_refs; old preprocessed images are deleted."""

    def test_regenerate_with_new_prompts_transitions_to_preprocessing(
        self, spy_client, SessionFactory
    ):
        import json as _json

        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = _make_run(
            session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=[
                "https://x.test/user.jpg", "https://x.test/trim1.jpg",
                "https://x.test/trim2.jpg",
            ],
            preprocess_image_prompts=["", "old prompt 1", "old prompt 2"],
        )
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={
                "preprocess_image_prompts": _json.dumps(["new prompt 1", "new prompt 2"]),
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "preprocessing"
        assert spy["preprocess_run"] == [run.id]

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        assert run.status == RunStatus.preprocessing
        # R1-new prompt model re-applied: ["", "new prompt 1", "new prompt 2"].
        assert run.preprocess_image_prompts == ["", "new prompt 1", "new prompt 2"]
        # Old preprocessed state is reset.
        assert run.preprocessed_reference_urls is None
        assert run.preprocess_task_ids is None
        # Refs unchanged (only prompts changed).
        assert run.reference_image_urls == [
            "https://x.test/user.jpg", "https://x.test/trim1.jpg",
            "https://x.test/trim2.jpg",
        ]

    def test_regenerate_voids_auto_approve(self, spy_client, SessionFactory):
        """An auto-approve run only reaches review when a failure parked it
        there. Regenerating from that card is manual control, and the card
        promises the run will not start until the operator approves."""
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = _make_run(
            session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            preprocess_auto_approve=True,
            reference_image_urls=["https://x.test/user.jpg", "https://x.test/t1.jpg"],
            preprocess_image_prompts=["", "old prompt"],
        )
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={"preprocess_image_prompts": '["new prompt"]'},
        )
        assert response.status_code == 200, response.text

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        assert run.preprocess_auto_approve is False

    def test_regenerate_with_new_char_ref_file_replaces_index_0(
        self, spy_client, SessionFactory
    ):
        """A new preprocess_reference_file replaces reference_image_urls[0]
        (the character ref); existing trims at [1:] are preserved."""
        import os

        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = _make_run(
            session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=[
                "https://x.test/old-char.jpg", "https://x.test/trim1.jpg",
                "https://x.test/trim2.jpg",
            ],
            preprocess_image_prompts=["", "old prompt 1", "old prompt 2"],
        )
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={},
            files={
                "preprocess_reference_file": ("new-char.jpg", io.BytesIO(b"new-char-bytes"), "image/jpeg"),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "preprocessing"
        assert spy["preprocess_run"] == [run.id]

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        # Index 0 replaced with the saved new char ref; trims kept.
        new_char = run.reference_image_urls[0]
        assert new_char != "https://x.test/old-char.jpg"
        assert os.sep + "references" + os.sep in new_char
        assert new_char.endswith("new-char.jpg")
        assert os.path.exists(new_char)
        assert run.reference_image_urls[1:] == [
            "https://x.test/trim1.jpg", "https://x.test/trim2.jpg",
        ]
        # Prompts re-applied from existing (R1-new model): ["", "old prompt 1", "old prompt 2"].
        assert run.preprocess_image_prompts == ["", "old prompt 1", "old prompt 2"]

    def test_regenerate_wrong_status_is_409(self, client, db_session):
        import json as _json

        project = _make_project(db_session)
        run = _make_run(
            db_session, project.id, status=RunStatus.queued,
            preprocess_enabled=True,
        )
        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={"preprocess_image_prompts": _json.dumps(["x"])},
        )
        assert response.status_code == 409
        assert "preprocess_review" in response.json()["detail"]

    def test_regenerate_missing_run_is_404(self, client):
        response = client.post("/api/v2/runs/no-such-run/regenerate-preprocess")
        assert response.status_code == 404

    def test_regenerate_invalid_json_is_400(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(
            db_session, project.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            reference_image_urls=["https://x.test/a.jpg", "https://x.test/b.jpg"],
        )
        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={"preprocess_image_prompts": "not-json"},
        )
        assert response.status_code == 400
        assert "must be valid JSON" in response.json()["detail"]

    def test_regenerate_no_refs_is_400(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(
            db_session, project.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True, reference_image_urls=[],
        )
        response = client.post(f"/api/v2/runs/{run.id}/regenerate-preprocess")
        assert response.status_code == 400


class TestDeletePreprocessRef:
    """DELETE /api/v2/runs/{rid}/preprocess-refs/{index} — drop ONE reference
    (and its GPT Image 2 result) from a run parked at preprocess_review."""

    def _review_run(self, session, project_id, **kwargs):
        """A run shaped like the accident this endpoint exists for: the
        character photo landed in the trims field a second time, so index 1 is
        a junk generation of the character photo from itself."""
        defaults = dict(
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=[
                "/d/char.png", "/d/char.png", "/d/f0.jpg", "/d/f95.jpg",
            ],
            preprocess_image_prompts=["", "pose it", "pose it", "pose it"],
            preprocessed_reference_urls=[
                "/d/char.png", "/d/pp/ref_0001.png", "/d/pp/ref_0002.png",
                "/d/pp/ref_0003.png",
            ],
            preprocess_task_ids=[None, "task1", "task2", "task3"],
        )
        defaults.update(kwargs)
        return _make_run(session, project_id, **defaults)

    def test_delete_pops_the_same_index_out_of_all_four_lists(
        self, client, db_session
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(db_session, project.id)

        response = client.delete(f"/api/v2/runs/{run.id}/preprocess-refs/1")
        assert response.status_code == 200, response.text

        db_session.refresh(run)
        # The lists are positional and parallel — index 1 leaves all four, and
        # what used to be index 2 keeps its prompt, its result and its task id.
        assert run.reference_image_urls == ["/d/char.png", "/d/f0.jpg", "/d/f95.jpg"]
        assert run.preprocess_image_prompts == ["", "pose it", "pose it"]
        assert run.preprocessed_reference_urls == [
            "/d/char.png", "/d/pp/ref_0002.png", "/d/pp/ref_0003.png",
        ]
        assert run.preprocess_task_ids == [None, "task2", "task3"]
        # Still under review — deleting a ref does not start the run.
        assert run.status == RunStatus.preprocess_review

    def test_delete_index_0_is_400(self, client, db_session):
        """Index 0 is the character photo — context for every GPT Image 2 task."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(db_session, project.id)

        response = client.delete(f"/api/v2/runs/{run.id}/preprocess-refs/0")
        assert response.status_code == 400
        assert "character photo" in response.json()["detail"]
        db_session.refresh(run)
        assert len(run.reference_image_urls) == 4

    def test_delete_out_of_range_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(db_session, project.id)

        response = client.delete(f"/api/v2/runs/{run.id}/preprocess-refs/9")
        assert response.status_code == 400
        assert "out of range" in response.json()["detail"]

    def test_delete_wrong_status_is_409(self, client, db_session):
        """Only preprocess_review — no worker may be holding the run."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(
            db_session, project.id, status=RunStatus.preprocessing
        )

        response = client.delete(f"/api/v2/runs/{run.id}/preprocess-refs/1")
        assert response.status_code == 409
        assert "preprocess_review" in response.json()["detail"]

    def test_delete_all_as_refs_is_409(self, client, db_session):
        """One combined image covers every ref — per-index deletion is meaningless."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(
            db_session, project.id,
            preprocess_mode="all_as_refs",
            preprocess_image_prompts=["combine them"],
            preprocessed_reference_urls=["/d/pp/ref_0000.png"],
        )

        response = client.delete(f"/api/v2/runs/{run.id}/preprocess-refs/1")
        assert response.status_code == 409
        assert "all_as_refs" in response.json()["detail"]

    def test_delete_missing_run_is_404(self, client):
        response = client.delete("/api/v2/runs/no-such-run/preprocess-refs/1")
        assert response.status_code == 404

    def test_delete_unlinks_the_generated_image(self, client, db_session):
        from app.storage import project_dir

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
        )
        pp_dir = os.path.join(
            project_dir(project.id), "runs", run.id, "preprocessed"
        )
        os.makedirs(pp_dir, exist_ok=True)
        doomed = os.path.join(pp_dir, "ref_0001.png")
        kept = os.path.join(pp_dir, "ref_0002.png")
        for f in (doomed, kept):
            with open(f, "wb") as fh:
                fh.write(b"png")
        run.reference_image_urls = ["/d/char.png", "/d/dup.png", "/d/f0.jpg"]
        run.preprocess_image_prompts = ["", "pose it", "pose it"]
        run.preprocessed_reference_urls = ["/d/char.png", doomed, kept]
        run.preprocess_task_ids = [None, "task1", "task2"]
        db_session.commit()

        response = client.delete(f"/api/v2/runs/{run.id}/preprocess-refs/1")
        assert response.status_code == 200, response.text
        assert not os.path.exists(doomed)
        assert os.path.exists(kept)

    def test_delete_keeps_a_pass_through_file_shared_with_another_ref(
        self, client, db_session
    ):
        """The duplicate-character-photo case: index 1 passed through, so its
        entry is the SAME path index 0 uses. Unlinking it would take out the
        character photo, so the file must survive the delete."""
        from app.storage import project_dir

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
        )
        refs_dir = os.path.join(
            project_dir(project.id), "runs", run.id, "references"
        )
        os.makedirs(refs_dir, exist_ok=True)
        char = os.path.join(refs_dir, "image.png")
        with open(char, "wb") as fh:
            fh.write(b"png")
        run.reference_image_urls = [char, char, "/d/f0.jpg"]
        run.preprocess_image_prompts = ["", "", "pose it"]
        run.preprocessed_reference_urls = [char, char, "/d/pp/ref_0002.png"]
        run.preprocess_task_ids = [None, None, "task2"]
        db_session.commit()

        response = client.delete(f"/api/v2/runs/{run.id}/preprocess-refs/1")
        assert response.status_code == 200, response.text
        assert os.path.exists(char)
        db_session.refresh(run)
        assert run.reference_image_urls == [char, "/d/f0.jpg"]
        assert run.preprocessed_reference_urls == [char, "/d/pp/ref_0002.png"]


class TestRegeneratePreprocessRef:
    """POST /api/v2/runs/{rid}/regenerate-preprocess-ref/{index} — redo ONE
    generated image and keep the others (and their billing)."""

    def _review_run(self, session, project_id, **kwargs):
        defaults = dict(
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=["/d/char.png", "/d/f0.jpg", "/d/f95.jpg"],
            preprocess_image_prompts=["", "pose it", "pose it"],
            preprocessed_reference_urls=[
                "/d/char.png", "/d/pp/ref_0001.png", "/d/pp/ref_0002.png",
            ],
            preprocess_task_ids=[None, "task1", "task2"],
        )
        defaults.update(kwargs)
        return _make_run(session, project_id, **defaults)

    def test_regenerate_one_clears_only_that_index(
        self, spy_client, SessionFactory
    ):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = self._review_run(session, project.id)
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1",
            data={"prompt": "pose it, but smiling"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "preprocessing"
        assert spy["preprocess_run"] == [run.id]

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        # Index 1 is empty and prompt-updated; index 2 keeps its image AND its
        # task id, which is what stops the phase from re-billing it.
        assert run.preprocess_image_prompts == ["", "pose it, but smiling", "pose it"]
        assert run.preprocessed_reference_urls == [
            "/d/char.png", None, "/d/pp/ref_0002.png",
        ]
        assert run.preprocess_task_ids == [None, None, "task2"]
        assert run.reference_image_urls == ["/d/char.png", "/d/f0.jpg", "/d/f95.jpg"]
        assert run.status == RunStatus.preprocessing

    def test_regenerate_one_without_a_prompt_rerolls_the_stored_one(
        self, spy_client, SessionFactory
    ):
        """GPT Image 2 is stochastic — "this one came out bad" needs no new
        prompt, just another go at the same one."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = self._review_run(session, project.id)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/2")
        assert response.status_code == 200, response.text

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        assert run.preprocess_image_prompts == ["", "pose it", "pose it"]
        assert run.preprocessed_reference_urls == [
            "/d/char.png", "/d/pp/ref_0001.png", None,
        ]
        assert run.preprocess_task_ids == [None, "task1", None]

    def test_regenerate_a_pass_through_ref_needs_a_prompt(self, client, db_session):
        """A trim that was passed through has no prompt of its own, so there is
        nothing to re-roll — regenerating it means writing one."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(
            db_session, project.id,
            preprocess_image_prompts=["", "", "pose it"],
            preprocessed_reference_urls=[
                "/d/char.png", "/d/f0.jpg", "/d/pp/ref_0002.png",
            ],
            preprocess_task_ids=[None, None, "task2"],
        )

        response = client.post(f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1")
        assert response.status_code == 400
        assert "no GPT Image prompt" in response.json()["detail"]
        db_session.refresh(run)
        assert run.status == RunStatus.preprocess_review

        # ...and with a prompt it goes through.
        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1",
            data={"prompt": "make it a close-up"},
        )
        assert response.status_code == 200, response.text
        db_session.refresh(run)
        assert run.preprocess_image_prompts == ["", "make it a close-up", "pose it"]

    def test_regenerate_index_0_is_400(self, client, db_session):
        """The character photo is context for every task, never a generation."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(db_session, project.id)

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/0",
            data={"prompt": "whatever"},
        )
        assert response.status_code == 400
        assert "character photo" in response.json()["detail"]
        db_session.refresh(run)
        assert run.status == RunStatus.preprocess_review

    def test_regenerate_out_of_range_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(db_session, project.id)

        response = client.post(f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/9")
        assert response.status_code == 400
        assert "out of range" in response.json()["detail"]

    def test_regenerate_wrong_status_is_409(self, client, db_session):
        """Only at preprocess_review — anywhere else a worker may hold the run."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(
            db_session, project.id, status=RunStatus.preprocessing
        )

        response = client.post(f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1")
        assert response.status_code == 409
        assert "preprocess_review" in response.json()["detail"]

    def test_regenerate_all_as_refs_is_409(self, client, db_session):
        """One image covers every ref there, so "just this one" is the full
        Regenerate."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = self._review_run(
            db_session, project.id,
            preprocess_mode="all_as_refs",
            preprocess_image_prompts=["combine them"],
            preprocessed_reference_urls=["/d/pp/ref_0000.png"],
        )

        response = client.post(f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1")
        assert response.status_code == 409
        assert "all_as_refs" in response.json()["detail"]

    def test_regenerate_missing_run_is_404(self, client):
        response = client.post(
            "/api/v2/runs/no-such-run/regenerate-preprocess-ref/1"
        )
        assert response.status_code == 404

    def test_regenerate_unlinks_the_old_image_and_keeps_the_others(
        self, spy_client, SessionFactory
    ):
        from app.storage import project_dir

        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = self._review_run(session, project.id)
        pp_dir = os.path.join(
            project_dir(project.id), "runs", run.id, "preprocessed"
        )
        os.makedirs(pp_dir, exist_ok=True)
        doomed = os.path.join(pp_dir, "ref_0001.png")
        kept = os.path.join(pp_dir, "ref_0002.png")
        for f in (doomed, kept):
            with open(f, "wb") as fh:
                fh.write(b"png")
        run = session.get(Run, run.id)
        run.preprocessed_reference_urls = ["/d/char.png", doomed, kept]
        session.commit()
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1",
            data={"prompt": "again please"},
        )
        assert response.status_code == 200, response.text
        assert not os.path.exists(doomed)
        assert os.path.exists(kept)

    def test_regenerate_keeps_an_original_that_was_passed_through(
        self, spy_client, SessionFactory
    ):
        """A pass-through slot holds the ORIGINAL frame — the input the new task
        transforms. Unlinking it would destroy the source image."""
        from app.storage import project_dir

        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = self._review_run(session, project.id)
        refs_dir = os.path.join(
            project_dir(project.id), "runs", run.id, "references"
        )
        os.makedirs(refs_dir, exist_ok=True)
        frame = os.path.join(refs_dir, "f0.jpg")
        with open(frame, "wb") as fh:
            fh.write(b"jpg")
        run = session.get(Run, run.id)
        run.reference_image_urls = ["/d/char.png", frame, "/d/f95.jpg"]
        run.preprocess_image_prompts = ["", "", "pose it"]
        run.preprocessed_reference_urls = [
            "/d/char.png", frame, "/d/pp/ref_0002.png",
        ]
        run.preprocess_task_ids = [None, None, "task2"]
        session.commit()
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1",
            data={"prompt": "now transform it"},
        )
        assert response.status_code == 200, response.text
        assert os.path.exists(frame)

    def test_regenerate_one_voids_auto_approve(self, spy_client, SessionFactory):
        """Hand-picking an image to redo happens AT the review gate: the run
        must pause there again instead of shipping to the video model the moment
        the new image lands."""
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = self._review_run(session, project.id, preprocess_auto_approve=True)
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess-ref/1",
            data={"prompt": "again"},
        )
        assert response.status_code == 200, response.text

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        assert run.preprocess_auto_approve is False


class TestPreprocessTrimsMustBeSourceFrames:
    """With GPT Image 2 preprocessing on, the reference field is the TRIMS and
    only takes frames staged off this project's source video. A stray image
    there burns a generation AND ships a junk reference to the AI video model
    for every segment."""

    def test_uploaded_trim_file_is_400(self, client, db_session):
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(["pose it"]),
            },
            files={
                "reference_files": ("trim.jpg", io.BytesIO(b"trim"), "image/jpeg"),
            },
        )
        assert response.status_code == 400
        assert "frames from the source video" in response.json()["detail"]

    def test_outside_trim_url_is_400(self, client, db_session):
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": "https://x.test/some-avatar.png",
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(["pose it"]),
            },
        )
        assert response.status_code == 400
        assert "frames from the source video" in response.json()["detail"]

    def test_another_projects_staged_frame_is_400(self, client, db_session):
        """Staged frames are per-project; a path from a different project is
        still not a frame of THIS video."""
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        other = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "preprocess_reference_url": "https://x.test/char.jpg",
                "reference_urls": _staged_frame(other.id, "t1.jpg"),
                "preprocess_enabled": "true",
                "preprocess_image_prompts": _json.dumps(["pose it"]),
            },
        )
        assert response.status_code == 400

    def test_without_preprocessing_any_reference_is_still_accepted(
        self, client, db_session
    ):
        """The rule is scoped to preprocessing — an ordinary run still takes
        arbitrary reference images."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "resolution": "720p",
                "reference_urls": "https://x.test/anything.png",
            },
        )
        assert response.status_code == 201, response.text

    def test_regenerate_with_outside_trim_url_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=["/d/char.png", "/d/f0.jpg"],
            preprocess_image_prompts=["", "pose it"],
        )
        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={"reference_urls": "https://x.test/some-avatar.png"},
        )
        assert response.status_code == 400
        assert "frames from the source video" in response.json()["detail"]
        db_session.refresh(run)
        # Rejected before anything was touched.
        assert run.status == RunStatus.preprocess_review
        assert run.reference_image_urls == ["/d/char.png", "/d/f0.jpg"]

    def test_regenerate_with_staged_frames_replaces_the_trims(
        self, spy_client, SessionFactory
    ):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        run = _make_run(
            session, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=["/d/char.png", "/d/old.jpg"],
            preprocess_image_prompts=["", "pose it"],
        )
        session.close()

        staged = _staged_frame(project.id, "frame_100_abc.jpg")
        response = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={"reference_urls": staged},
        )
        assert response.status_code == 200, response.text

        session = SessionFactory()
        run = session.get(Run, run.id)
        session.close()
        assert run.reference_image_urls == ["/d/char.png", staged]
        assert run.status == RunStatus.preprocessing


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/retry
# ---------------------------------------------------------------------------


class TestRetryRun:
    def test_retry_failed_run_transitions_to_queued(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.failed)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert spy["process_run"] == [run.id]

    def test_retry_non_retryable_run_is_409(self, client, db_session):
        """done status is not retryable via /retry (use /runs/{rid}/segments/{rsid}/rerun for that)."""
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 409

    def test_retry_processing_run_succeeds(self, spy_client, SessionFactory):
        """TR5b: /retry must accept a run stuck in processing (orphan resume)."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.processing)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert run.id in spy["process_run"]

    def test_retry_incomplete_run_succeeds(self, spy_client, SessionFactory):
        """/retry must accept an `incomplete` run (re-send the failed segments)."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.incomplete)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert run.id in spy["process_run"]

    def test_retry_incomplete_run_409_while_segment_image_job_active(
        self, spy_client, SessionFactory
    ):
        """Cursor-gate catch: /retry re-enqueues run_submit_run, which reads
        references ONCE — the same hazard rerun_segment/patch_run_segment
        already guard against. Retry must refuse the same way while any
        segment's GPT Image 2 regeneration is still in flight."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.incomplete)
        rs = _make_run_segment(
            session, run.id, sd.id, status=SegmentStatus.failed,
        )
        session.add(
            RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
                status=SegmentImageStatus.generating,
            )
        )
        session.commit()
        rid = run.id
        session.close()

        response = client.post(f"/api/v2/runs/{rid}/retry")
        assert response.status_code == 409
        assert "gpt image" in response.json()["detail"].lower()
        assert rid not in spy["process_run"]

    def test_retry_failed_run_409_while_segment_image_job_active(
        self, spy_client, SessionFactory
    ):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.failed)
        rs = _make_run_segment(
            session, run.id, sd.id, status=SegmentStatus.completed,
        )
        session.add(
            RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
                status=SegmentImageStatus.queued,
            )
        )
        session.commit()
        rid = run.id
        session.close()

        response = client.post(f"/api/v2/runs/{rid}/retry")
        assert response.status_code == 409

    def test_retry_queued_run_succeeds(self, spy_client, SessionFactory):
        """TR5b: /retry must accept a run stuck in queued (orphan resume)."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.queued)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert run.id in spy["process_run"]

    def test_retry_missing_run_is_404(self, client):
        response = client.post("/api/v2/runs/no-such-run/retry")
        assert response.status_code == 404

    # --- preprocess holes: retry must re-enter preprocessing, not submit ---

    def test_retry_failed_run_with_preprocess_hole_reenters_preprocessing(
        self, spy_client, SessionFactory
    ):
        """A missing GPT Image 2 image routes Retry back to the preprocess phase.

        Regression for run ce9b5bfd: failed → queued goes straight to submit,
        which cannot upload the None and fails the run again — forever.
        """
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(
            session, project.id, status=RunStatus.failed,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=["/photo.png", "/t1.jpg", "/t2.jpg"],
            preprocess_image_prompts=["", "pose like this", "pose like this"],
            preprocessed_reference_urls=["/photo.png", None, "/pp/ref_0002.png"],
            preprocess_task_ids=[None, "task-1", "task-2"],
            error_message="GPT Image 2 preprocessing failed: download died",
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "preprocessing"
        # The old error is cleared, and submit is NOT enqueued.
        assert body["error_message"] is None
        assert spy["preprocess_run"] == [run.id]
        assert spy["process_run"] == []

    def test_retry_with_hole_but_billed_segment_uses_submit_path(
        self, spy_client, SessionFactory
    ):
        """Once Seedance was billed, the failure is a VIDEO failure.

        Whatever the reference list looks like, such a run must retry through
        `queued` rather than being shoved back into GPT Image 2.
        """
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        sd = _make_segment_def(session, project.id, index=0, action="swap")
        run = _make_run(
            session, project.id, status=RunStatus.failed,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=["/photo.png", "/t1.jpg"],
            preprocess_image_prompts=["", "pose like this"],
            preprocessed_reference_urls=["/photo.png", None],
        )
        _make_run_segment(
            session, run.id, sd.id, index=0, seedance_task_id="already-billed",
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]
        assert spy["preprocess_run"] == []

    def test_retry_all_as_refs_single_result_is_not_a_hole(
        self, spy_client, SessionFactory
    ):
        """all_as_refs collapses every trim into ONE image by design.

        Its one-element result list is shorter than reference_image_urls, which
        a naive length check would read as a hole and route into GPT Image 2.
        """
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(
            session, project.id, status=RunStatus.failed,
            preprocess_enabled=True,
            preprocess_mode="all_as_refs",
            reference_image_urls=["/photo.png", "/t1.jpg", "/t2.jpg"],
            preprocess_image_prompts=["combine these"],
            preprocessed_reference_urls=["/pp/ref_0000.png"],
            preprocess_task_ids=["task-1", None, None],
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]
        assert spy["preprocess_run"] == []

    def test_retry_all_as_refs_with_dropped_task_id_is_still_a_hole(
        self, spy_client, SessionFactory
    ):
        """A gone result URL drops the task id — that must not read as "nothing owed".

        The poll phase clears preprocess_task_ids when a result URL 404s, so a
        predicate keyed off the task id would send this run to submit with an
        all-None reference list: the very dead end this change closes.
        """
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(
            session, project.id, status=RunStatus.failed,
            preprocess_enabled=True,
            preprocess_mode="all_as_refs",
            reference_image_urls=["/photo.png", "/t1.jpg", "/t2.jpg"],
            preprocess_image_prompts=["combine these"],
            preprocessed_reference_urls=[None, None, None],
            preprocess_task_ids=[None, None, None],
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "preprocessing"
        assert spy["preprocess_run"] == [run.id]
        assert spy["process_run"] == []

    def test_retry_all_as_refs_passthrough_is_not_a_hole(
        self, spy_client, SessionFactory
    ):
        """An empty all_as_refs prompt generates nothing and owes nothing."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(
            session, project.id, status=RunStatus.failed,
            preprocess_enabled=True,
            preprocess_mode="all_as_refs",
            reference_image_urls=["/photo.png", "/t1.jpg"],
            preprocess_image_prompts=[""],
            preprocessed_reference_urls=[],
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]
        assert spy["preprocess_run"] == []

    def test_retry_passthrough_preprocess_run_is_not_a_hole(
        self, spy_client, SessionFactory
    ):
        """A trim with no prompt is passed through — it owes no generated image.

        Such a run legitimately has an empty preprocessed list; a later Seedance
        failure must still retry through submit.
        """
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(
            session, project.id, status=RunStatus.failed,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=["/photo.png", "/t1.jpg"],
            preprocess_image_prompts=["", ""],
            preprocessed_reference_urls=[],
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]
        assert spy["preprocess_run"] == []


# ---------------------------------------------------------------------------
# PATCH /api/v2/runs/{rid}/segments/{rsid}
# ---------------------------------------------------------------------------


class TestPatchRunSegment:
    def test_patch_sets_prompt_override(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "per-segment override"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prompt_override"] == "per-segment override"

    def test_patch_while_processing_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.generating)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "override"},
        )
        assert response.status_code == 409

    def test_patch_on_incomplete_run_sets_prompt_override(self, client, db_session):
        """An `incomplete` run is exactly when a prompt needs rewording (a swap
        segment failed, e.g. Seedance's "output may contain sensitive
        information"), so Save override must work there — it mirrors the
        statuses /rerun accepts."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.incomplete)
        rs = _make_run_segment(
            db_session, run.id, sd.id,
            status=SegmentStatus.failed,
            error_message=(
                "failed after 3 attempt(s): The request failed because the "
                "output video may contain sensitive information."
            ),
        )

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "reworded prompt"},
        )
        assert response.status_code == 200
        assert response.json()["prompt_override"] == "reworded prompt"

    def test_patch_empty_prompt_clears_override(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id,
            status=SegmentStatus.completed,
            prompt_override="old override",
        )

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": ""},
        )
        assert response.status_code == 200
        assert response.json()["prompt_override"] is None

    def test_patch_too_many_refs_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.failed)
        rs = _make_run_segment(db_session, run.id, sd.id)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"reference_urls": "https://a.com/1.jpg,https://a.com/2.jpg,https://a.com/3.jpg,https://a.com/4.jpg,https://a.com/5.jpg,https://a.com/6.jpg"},
        )
        assert response.status_code == 400
        assert "too many" in response.json()["detail"].lower()

    def test_patch_missing_segment_is_404(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/no-such-seg",
            data={"prompt": "override"},
        )
        assert response.status_code == 404

    def test_patch_sets_reference_urls_override(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"reference_urls": "https://example.com/ref1.jpg"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["reference_image_urls_override"] is not None
        assert len(body["reference_image_urls_override"]) == 1


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/segments/{rsid}/rerun
# ---------------------------------------------------------------------------


class TestRerunSegment:
    def test_rerun_resets_segment_and_enqueues(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(
            session, project.id, status=RunStatus.done,
            result_local_path="/data/fake/final.mp4",
        )
        rs = _make_run_segment(
            session, run.id, sd.id,
            status=SegmentStatus.completed,
            seedance_task_id="old-task-id",
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        # enqueue_process_run should have been called
        assert spy["process_run"] == [run.id]

        # Verify the RunSegment was reset
        session2 = SessionFactory()
        rs_fetched = session2.get(RunSegment, rs.id)
        run_fetched = session2.get(Run, run.id)
        session2.close()
        assert rs_fetched.status == SegmentStatus.pending
        assert rs_fetched.seedance_task_id is None
        # The stitched final contains the OLD segment result — a rerun must
        # invalidate it so finalize re-stitches instead of re-delivering it.
        assert run_fetched.result_local_path is None

    def test_rerun_failed_segment_from_incomplete_run(self, spy_client, SessionFactory):
        """An `incomplete` run can re-run its failed segment; run → queued."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.incomplete)
        rs = _make_run_segment(
            session, run.id, sd.id,
            status=SegmentStatus.failed,
            seedance_task_id="failed-task-id",
            error_message="failed after 3 attempt(s): Internal Error",
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]

        session2 = SessionFactory()
        rs_fetched = session2.get(RunSegment, rs.id)
        session2.close()
        assert rs_fetched.status == SegmentStatus.pending
        assert rs_fetched.error_message is None

    def test_rerun_with_prompt_applies_override(self, spy_client, SessionFactory):
        """Re-run carrying a prompt persists it as the segment override atomically."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            session, run.id, sd.id, status=SegmentStatus.completed,
        )
        rs_id = rs.id
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs_id}/rerun",
            data={"prompt": "make the character a red panda"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]

        session2 = SessionFactory()
        rs_fetched = session2.get(RunSegment, rs_id)
        session2.close()
        assert rs_fetched.prompt_override == "make the character a red panda"
        assert rs_fetched.status == SegmentStatus.pending

    def test_rerun_on_failed_run_also_works(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.failed)
        rs = _make_run_segment(session, run.id, sd.id, status=SegmentStatus.failed)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    def test_rerun_on_processing_run_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.generating)

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 409

    def test_rerun_missing_segment_is_404(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.post(f"/api/v2/runs/{run.id}/segments/no-such-seg/rerun")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/segments/rerun — several segments, one re-stitch
# (AICP-341)
# ---------------------------------------------------------------------------


class TestRerunSegments:
    def _world(self, session, *, status=RunStatus.done, n=3, **run_kw):
        project = _make_project(session, status=ProjectStatus.ready)
        defs = [_make_segment_def(session, project.id, i) for i in range(n)]
        run = _make_run(
            session, project.id, status=status,
            result_local_path="/data/fake/final.mp4", result_duration_sec=15.0,
            **run_kw,
        )
        segs = [
            _make_run_segment(
                session, run.id, sd.id, index=sd.index,
                status=SegmentStatus.completed,
                seedance_task_id=f"old-{sd.index}",
                prompt_override=f"saved {sd.index}",
            )
            for sd in defs
        ]
        return project, run, segs

    def test_resets_every_listed_segment_and_enqueues_once(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s)
        rid, ids = run.id, [rs.id for rs in segs]
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0], "prompt": "new prompt 0"},
            {"run_segment_id": ids[2], "prompt": ""},
        ]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "queued"
        # ONE process_run for the whole list — one submit, one re-stitch.
        assert spy["process_run"] == [rid]

        s = SessionFactory()
        rs0, rs1, rs2 = (s.get(RunSegment, i) for i in ids)
        run2 = s.get(Run, rid)
        assert rs0.status == SegmentStatus.pending and rs0.seedance_task_id is None
        assert rs0.prompt_override == "new prompt 0"
        assert rs2.status == SegmentStatus.pending and rs2.prompt_override is None
        # Unlisted segment untouched.
        assert rs1.status == SegmentStatus.completed
        assert rs1.seedance_task_id == "old-1" and rs1.prompt_override == "saved 1"
        assert run2.result_local_path is None
        assert run2.result_duration_sec is None
        s.close()

    def test_omitted_prompt_keeps_the_saved_override(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=2)
        rid, ids = run.id, [rs.id for rs in segs]
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0]}, {"run_segment_id": ids[1], "prompt": None},
        ]})
        assert resp.status_code == 200, resp.text
        s = SessionFactory()
        assert [s.get(RunSegment, i).prompt_override for i in ids] == ["saved 0", "saved 1"]
        assert [s.get(RunSegment, i).status for i in ids] == [SegmentStatus.pending] * 2
        s.close()

    @pytest.mark.parametrize("payload_kind", ["empty", "duplicate"])
    def test_bad_list_is_400_and_changes_nothing(self, spy_client, SessionFactory, payload_kind):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=2)
        rid, ids = run.id, [rs.id for rs in segs]
        s.close()
        items = [] if payload_kind == "empty" else [
            {"run_segment_id": ids[0], "prompt": "x"}, {"run_segment_id": ids[0]},
        ]
        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": items})
        assert resp.status_code == 400, resp.text
        s = SessionFactory()
        assert s.get(Run, rid).status == RunStatus.done
        assert s.get(RunSegment, ids[0]).prompt_override == "saved 0"
        s.close()
        assert spy["process_run"] == []

    def test_a_segment_of_another_run_is_404_and_changes_nothing(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        _, other, other_segs = self._world(s, n=1)
        rid, mine, foreign = run.id, segs[0].id, other_segs[0].id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": mine, "prompt": "changed"}, {"run_segment_id": foreign},
        ]})
        assert resp.status_code == 404, resp.text
        s = SessionFactory()
        assert s.get(RunSegment, mine).status == SegmentStatus.completed
        assert s.get(RunSegment, mine).prompt_override == "saved 0"
        assert s.get(RunSegment, foreign).status == SegmentStatus.completed
        assert s.get(Run, rid).status == RunStatus.done
        s.close()
        assert spy["process_run"] == []

    def test_processing_run_is_409(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, status=RunStatus.processing, n=1)
        rid, sid = run.id, segs[0].id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": sid},
        ]})
        assert resp.status_code == 409, resp.text
        assert spy["process_run"] == []

    def test_a_segment_excluded_by_the_stitch_plan_is_400(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=2)
        run.stitch_plan = [{"kind": "segment", "segment_def_id": segs[0].segment_def_id}]
        s.commit()
        rid, ids = run.id, [rs.id for rs in segs]
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0]}, {"run_segment_id": ids[1]},
        ]})
        assert resp.status_code == 400, resp.text
        assert ids[1] in resp.json()["detail"]
        s = SessionFactory()
        assert s.get(RunSegment, ids[0]).status == SegmentStatus.completed
        s.close()
        assert spy["process_run"] == []

    def test_blocked_while_an_audio_patch_is_in_flight(self, spy_client, SessionFactory):
        from app.models import RunAudioPatch
        from app.state_machine import AudioPatchStatus

        client, spy = spy_client
        s = SessionFactory()
        project, run, segs = self._world(s, n=1)
        donor = _make_run(s, project.id, status=RunStatus.done)
        s.add(RunAudioPatch(
            run_id=run.id, donor_run_id=donor.id, target_start_sec=1.0,
            target_end_sec=2.0, donor_start_sec=1.0, crossfade_sec=0.03,
            base_result_path="/data/fake/final.mp4", status=AudioPatchStatus.queued,
        ))
        s.commit()
        rid, sid = run.id, segs[0].id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": sid},
        ]})
        assert resp.status_code == 409, resp.text
        assert spy["process_run"] == []


# ---------------------------------------------------------------------------
# Segment takes (AICP-342): rerun preserves the replaced output, restore puts
# a previous one back
# ---------------------------------------------------------------------------


class TestSegmentTakes:
    def _world(self, session, *, n=2, status=RunStatus.done):
        from app.storage import run_results_dir

        project = _make_project(session, status=ProjectStatus.ready)
        defs = [_make_segment_def(session, project.id, i) for i in range(n)]
        run = _make_run(
            session, project.id, status=status,
            result_local_path="/data/fake/final.mp4", result_gdrive_file_id="DRIVE1",
            segment_snapshot=[
                {"id": sd.id, "index": sd.index, "action": "swap",
                 "start_sec": sd.start_sec, "end_sec": sd.end_sec} for sd in defs
            ],
        )
        r_dir = run_results_dir(run.id, project.id)
        segs = []
        for sd in defs:
            path = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
            with open(path, "wb") as fh:
                fh.write(f"round0-{sd.index}".encode())
            segs.append(_make_run_segment(
                session, run.id, sd.id, index=sd.index,
                status=SegmentStatus.completed, local_result_path=path,
                seedance_task_id=f"task0-{sd.index}", cost_credits=100.0,
                prompt_override=f"round0 prompt {sd.index}",
            ))
        return project, run, segs

    def _takes(self, session, rsid):
        from app.models import RunSegmentTake
        return session.execute(
            select(RunSegmentTake).where(RunSegmentTake.run_segment_id == rsid)
        ).scalars().all()

    def _complete(self, session, rsid, content: bytes, task: str):
        """What the pipeline does when the regeneration lands."""
        from app.storage import run_results_dir

        rs = session.get(RunSegment, rsid)
        run = session.get(Run, rs.run_id)
        path = os.path.join(run_results_dir(run.id, run.project_id), f"result_{rs.index:04d}.mp4")
        with open(path, "wb") as fh:
            fh.write(content)
        rs.local_result_path = path
        rs.seedance_task_id = task
        rs.status = SegmentStatus.completed
        run.status = RunStatus.done
        run.result_local_path = "/data/fake/final2.mp4"
        session.commit()

    def test_rerun_preserves_the_replaced_output_with_its_recipe(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s)
        rid, ids, orig = run.id, [rs.id for rs in segs], segs[0].local_result_path
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0], "prompt": "round1 prompt"},
            {"run_segment_id": ids[1]},
        ]})
        assert resp.status_code == 200, resp.text

        s = SessionFactory()
        for i, sid in enumerate(ids):
            (take,) = self._takes(s, sid)
            assert open(take.result_path, "rb").read() == f"round0-{i}".encode()
            assert take.result_path != orig and "segment_takes" in take.result_path
            assert take.seedance_task_id == f"task0-{i}"
            assert take.prompt_override == f"round0 prompt {i}"  # NOT the new one
            assert (take.start_sec, take.end_sec) == (5.0 * i, 5.0 * i + 5.0)
        assert s.get(RunSegment, ids[0]).prompt_override == "round1 prompt"
        s.close()

    def test_single_rerun_preserves_the_take_before_applying_the_new_prompt(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        rid, sid = run.id, segs[0].id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/{sid}/rerun", data={"prompt": "new"})
        assert resp.status_code == 200, resp.text
        s = SessionFactory()
        (take,) = self._takes(s, sid)
        assert take.prompt_override == "round0 prompt 0"
        s.close()

    def test_restore_puts_the_take_back_and_preserves_the_current_one(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        import app.api_v2 as api_v2_module

        @contextmanager
        def _test_get_session():  # the video endpoint opens its own session
            session = SessionFactory()
            try:
                yield session
            finally:
                session.close()

        monkeypatch.setattr(api_v2_module, "get_session", _test_get_session)
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s)
        rid, ids = run.id, [rs.id for rs in segs]
        s.close()
        assert client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0]}]}).status_code == 200
        s = SessionFactory()
        self._complete(s, ids[0], b"round1-0", "task1-0")
        (take0,) = self._takes(s, ids[0])
        take0_id = take0.id
        s.close()

        listing = client.get(f"/api/v2/runs/{rid}/segments/{ids[0]}/takes").json()
        assert [t["id"] for t in listing] == [take0_id] and listing[0]["is_current"] is False

        resp = client.post(f"/api/v2/runs/{rid}/segments/restore", json={"takes": [take0_id]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "queued"
        assert spy["process_run"] == [rid, rid]

        s = SessionFactory()
        rs0 = s.get(RunSegment, ids[0])
        run2 = s.get(Run, rid)
        assert rs0.status == SegmentStatus.completed
        assert open(rs0.local_result_path, "rb").read() == b"round0-0"
        assert rs0.seedance_task_id == "task0-0"
        assert run2.result_local_path is None and run2.result_gdrive_file_id == "DRIVE1"
        takes = {t.seedance_task_id: t for t in self._takes(s, ids[0])}
        assert set(takes) == {"task0-0", "task1-0"}  # round 1 preserved too
        assert open(takes["task1-0"].result_path, "rb").read() == b"round1-0"
        # Restore is free: nothing left to generate.
        from app import spend
        assert spend.estimate_run_credits(s, run2, include_preprocess=False) == 0
        s.close()

        listing = client.get(f"/api/v2/runs/{rid}/segments/{ids[0]}/takes").json()
        assert {t["id"]: t["is_current"] for t in listing}[take0_id] is True
        video = client.get(f"/api/v2/runs/{rid}/segments/{ids[0]}/takes/{take0_id}/video")
        assert video.status_code == 200

        # Rerunning the restored segment does not duplicate its take.
        s = SessionFactory()
        r = s.get(Run, rid)
        r.status = RunStatus.done
        s.commit()
        s.close()
        assert client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0]}]}).status_code == 200
        s = SessionFactory()
        assert len(self._takes(s, ids[0])) == 2
        s.close()

    def test_restore_brings_back_the_takes_overrides(self, spy_client, SessionFactory):
        """Operator 2026-10-30: the take's recipe comes back with it, including a
        recorded "no override" (which clears the field), so Copy/Automation
        reproduce the restored output."""
        from app.storage import data_dir_root

        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s)
        ref = os.path.join(data_dir_root(), "fake-refs", f"ref-{run.id}.jpg")
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        open(ref, "wb").write(b"ref0")
        segs[0].reference_image_urls_override = [ref, "https://x.example/a.jpg"]
        segs[0].generated_reference_urls = ["/g/char.jpg", "/g/out.png"]
        segs[0].generated_image_job_id = "job-0"
        segs[1].prompt_override = None  # round 0 of seg 1 used the run prompt
        s.commit()
        rid, ids = run.id, [rs.id for rs in segs]
        s.close()

        assert client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0], "prompt": "round1 prompt"},
            {"run_segment_id": ids[1], "prompt": "round1 prompt b"},
        ]}).status_code == 200
        s = SessionFactory()
        rs0 = s.get(RunSegment, ids[0])
        rs0.reference_image_urls_override = ["https://x.example/new.jpg"]
        rs0.generated_reference_urls = None
        rs0.generated_image_job_id = None
        open(ref, "wb").write(b"overwritten by a later PATCH")
        self._complete(s, ids[0], b"round1-0", "task1-0")
        self._complete(s, ids[1], b"round1-1", "task1-1")
        (t0,) = self._takes(s, ids[0])
        (t1,) = self._takes(s, ids[1])
        take_ids = [t0.id, t1.id]
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/segments/restore", json={"takes": take_ids})
        assert resp.status_code == 200, resp.text
        s = SessionFactory()
        rs0, rs1 = s.get(RunSegment, ids[0]), s.get(RunSegment, ids[1])
        assert rs0.prompt_override == "round0 prompt 0"
        local, url = rs0.reference_image_urls_override
        assert url == "https://x.example/a.jpg"
        # A copy in the segment's own refs dir, made from the take's copy —
        # never a path into the take (the retention sweep deletes those).
        assert local != ref and "segment_takes" not in local and "segment_refs" in local
        assert open(local, "rb").read() == b"ref0"
        assert rs0.generated_reference_urls == ["/g/char.jpg", "/g/out.png"]
        assert rs0.generated_image_job_id == "job-0"
        assert rs1.prompt_override is None  # recorded "none" cleared it
        assert rs1.reference_image_urls_override is None
        # The round-1 takes recorded round 1's recipe.
        r1 = {t.seedance_task_id: t for t in self._takes(s, ids[1])}["task1-1"]
        assert r1.prompt_override == "round1 prompt b"
        s.close()

    def test_archive_keeps_two_local_refs_with_the_same_basename_apart(self, spy_client, SessionFactory):
        from app.storage import data_dir_root

        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        base = os.path.join(data_dir_root(), "fake-refs", run.id)
        a, b = os.path.join(base, "x", "ref.jpg"), os.path.join(base, "y", "ref.jpg")
        for path, body in ((a, b"A"), (b, b"B")):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(body)
        segs[0].reference_image_urls_override = [a, b]
        s.commit()
        rid, sid = run.id, segs[0].id
        s.close()
        assert client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": sid}]}).status_code == 200
        s = SessionFactory()
        (take,) = self._takes(s, sid)
        ra, rb = take.reference_image_urls_override
        assert ra != rb
        assert (open(ra, "rb").read(), open(rb, "rb").read()) == (b"A", b"B")
        assert ra.endswith(".jpg") and len(os.path.basename(ra)) < 25
        s.close()

    def test_restore_refuses_a_purged_take(self, spy_client, SessionFactory):
        from datetime import datetime, timezone
        from app.models import RunSegmentTake

        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        take = RunSegmentTake(
            run_id=run.id, run_segment_id=segs[0].id, segment_def_id=segs[0].segment_def_id,
            index=0, start_sec=0.0, end_sec=5.0, result_path="/gone/take.mp4",
            file_purged_at=datetime.now(timezone.utc),
        )
        s.add(take)
        s.commit()
        rid, sid, tid = run.id, segs[0].id, take.id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/restore", json={"takes": [tid]})
        assert resp.status_code == 409, resp.text
        assert "purged" in resp.json()["detail"]
        listing = client.get(f"/api/v2/runs/{rid}/segments/{sid}/takes").json()
        assert listing[0]["file_purged"] is True
        assert spy["process_run"] == []

    def test_restore_refuses_when_another_segment_would_be_billed(self, spy_client, SessionFactory):
        from app.models import RunSegmentTake

        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, status=RunStatus.incomplete)
        take = RunSegmentTake(
            run_id=run.id, run_segment_id=segs[0].id, segment_def_id=segs[0].segment_def_id,
            index=0, start_sec=0.0, end_sec=5.0, result_path=segs[0].local_result_path,
            seedance_task_id="task0-0",
        )
        s.add(take)
        segs[1].status = SegmentStatus.failed
        segs[1].local_result_path = None
        s.commit()
        rid, tid = run.id, take.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/segments/restore", json={"takes": [tid]})
        assert resp.status_code == 409, resp.text
        assert "[1]" in resp.json()["detail"]
        assert spy["process_run"] == []

    def test_restore_refuses_a_take_whose_bounds_no_longer_fit(self, spy_client, SessionFactory):
        from app.models import RunSegmentTake

        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        path = segs[0].local_result_path + ".old"
        open(path, "wb").write(b"old")
        take = RunSegmentTake(
            run_id=run.id, run_segment_id=segs[0].id, segment_def_id=segs[0].segment_def_id,
            index=0, start_sec=0.0, end_sec=3.5, result_path=path,
        )
        s.add(take)
        s.commit()
        rid, tid = run.id, take.id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/restore", json={"takes": [tid]})
        assert resp.status_code == 409, resp.text
        assert "no longer fits" in resp.json()["detail"]

    def test_restore_refuses_a_take_whose_file_is_gone(self, spy_client, SessionFactory):
        from app.models import RunSegmentTake

        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        take = RunSegmentTake(
            run_id=run.id, run_segment_id=segs[0].id, segment_def_id=segs[0].segment_def_id,
            index=0, start_sec=0.0, end_sec=5.0, result_path="/nonexistent/take.mp4",
        )
        s.add(take)
        s.commit()
        rid, tid = run.id, take.id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/restore", json={"takes": [tid]})
        assert resp.status_code == 409, resp.text

    def test_restore_rejects_foreign_and_duplicate_segment_takes(self, spy_client, SessionFactory):
        from app.models import RunSegmentTake

        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        _, other, other_segs = self._world(s, n=1)
        mk = lambda r, rs: RunSegmentTake(  # noqa: E731
            run_id=r.id, run_segment_id=rs.id, segment_def_id=rs.segment_def_id,
            index=0, start_sec=0.0, end_sec=5.0, result_path=rs.local_result_path,
        )
        a, b, foreign = mk(run, segs[0]), mk(run, segs[0]), mk(other, other_segs[0])
        s.add_all([a, b, foreign])
        s.commit()
        rid, ids = run.id, (a.id, b.id, foreign.id)
        s.close()
        assert client.post(f"/api/v2/runs/{rid}/segments/restore",
                           json={"takes": [ids[2]]}).status_code == 404
        assert client.post(f"/api/v2/runs/{rid}/segments/restore",
                           json={"takes": [ids[0], ids[1]]}).status_code == 400
        assert spy["process_run"] == []

    def _patched(self, s, run):
        from app.models import RunAudioPatch
        from app.state_machine import AudioPatchStatus

        donor = _make_run(s, run.project_id, status=RunStatus.done)
        s.add(RunAudioPatch(
            run_id=run.id, donor_run_id=donor.id, target_start_sec=1.0,
            target_end_sec=2.0, donor_start_sec=1.0, crossfade_sec=0.03,
            base_result_path="/data/fake/final.mp4", output_path="/data/fake/patched.mp4",
            status=AudioPatchStatus.done,
        ))
        run.result_local_path = "/data/fake/patched.mp4"
        s.commit()

    def test_rerun_of_an_audio_patched_result_needs_an_explicit_discard(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _, run, segs = self._world(s, n=1)
        self._patched(s, run)
        rid, sid = run.id, segs[0].id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun",
                           json={"segments": [{"run_segment_id": sid}]})
        assert resp.status_code == 409, resp.text
        assert "discard_audio_patch" in resp.json()["detail"]
        resp = client.post(f"/api/v2/runs/{rid}/segments/{sid}/rerun")
        assert resp.status_code == 409, resp.text
        assert spy["process_run"] == []

        resp = client.post(f"/api/v2/runs/{rid}/segments/{sid}/rerun",
                           data={"discard_audio_patch": "true"})
        assert resp.status_code == 200, resp.text
        assert spy["process_run"] == [rid]


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/segments/{rsid}/range-rerun — run-local segment
# range regeneration
# ---------------------------------------------------------------------------


def _make_run_with_snapshot(session, *, stitch_plan=None, model="seedance", rs_status=SegmentStatus.completed):
    """A done Run over keep[0,5) | swap[5,10) | keep[10,20), frozen into
    Run.segment_snapshot exactly as a real first submit would freeze it."""
    project = _make_project(session, status=ProjectStatus.ready, duration_sec=20.0)
    sd_keep0 = _make_segment_def(session, project.id, 0, action="keep", start_sec=0.0, end_sec=5.0)
    sd_swap1 = _make_segment_def(session, project.id, 1, action="swap", start_sec=5.0, end_sec=10.0)
    sd_keep2 = _make_segment_def(session, project.id, 2, action="keep", start_sec=10.0, end_sec=20.0)
    run = _make_run(
        session, project.id, status=RunStatus.done, model=model, stitch_plan=stitch_plan,
        result_local_path="/data/fake/final.mp4",
        segment_snapshot=[
            {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 5.0},
            {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 10.0},
            {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
        ],
    )
    rs = _make_run_segment(
        session, run.id, sd_swap1.id, index=1, status=rs_status,
        local_result_path="/data/fake/result_0001.mp4",
    )
    return project, run, sd_keep0, sd_swap1, sd_keep2, rs


class TestCreateRangeRerun:
    def test_audio_patched_result_needs_an_explicit_discard(self, spy_client, SessionFactory):
        """AICP-349: a promoted range attempt replaces the content behind the
        run's Drive link, so it drops a promoted audio fix from the exported
        link exactly like a rerun — same acknowledgement, nothing created or
        enqueued without it."""
        from app.models import RunAudioPatch, RunSegmentRangeAttempt
        from app.state_machine import AudioPatchStatus

        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        donor = _make_run(s, project.id, status=RunStatus.done)
        s.add(RunAudioPatch(
            run_id=run.id, donor_run_id=donor.id, target_start_sec=1.0,
            target_end_sec=2.0, donor_start_sec=1.0, crossfade_sec=0.03,
            base_result_path="/data/fake/final.mp4", output_path="/data/fake/patched.mp4",
            status=AudioPatchStatus.done,
        ))
        run.result_local_path = "/data/fake/patched.mp4"
        s.commit()
        rid, rsid = run.id, rs.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert resp.status_code == 409, resp.text
        assert "discard_audio_patch" in resp.json()["detail"]
        assert spy["range_attempt"] == []
        s = SessionFactory()
        assert s.query(RunSegmentRangeAttempt).filter_by(run_id=rid).count() == 0
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0", "discard_audio_patch": "true"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["attempt_id"] in spy["range_attempt"]

    def test_valid_request_creates_attempt_and_enqueues(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        rid, rsid = run.id, rs.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["status"] == "queued"
        assert body["requested_start_sec"] == pytest.approx(4.6)
        assert body["requested_end_sec"] == pytest.approx(10.0)
        assert body["attempt_id"] in spy["range_attempt"]

        # The active Run/RunSegment must NOT be touched yet.
        s = SessionFactory()
        run_fetched = s.get(Run, rid)
        rs_fetched = s.get(RunSegment, rsid)
        s.close()
        assert run_fetched.status == RunStatus.done
        assert run_fetched.result_local_path == "/data/fake/final.mp4"
        assert rs_fetched.local_result_path == "/data/fake/result_0001.mp4"

    def test_missing_timing_fields_is_422(self, client, db_session):
        """This endpoint always requires start_sec/end_sec — the legacy
        omitted-timing-fields-preserve-behaviour rule belongs to .../rerun
        only, not to this one."""
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun")
        assert response.status_code == 422

    def test_locked_boundary_is_400_with_readable_message(self, client, db_session):
        """swap -> swap: both neighbours are swap, so no boundary can move."""
        project = _make_project(db_session, status=ProjectStatus.ready, duration_sec=20.0)
        sd0 = _make_segment_def(db_session, project.id, 0, action="swap", start_sec=0.0, end_sec=5.0)
        sd1 = _make_segment_def(db_session, project.id, 1, action="swap", start_sec=5.0, end_sec=10.0)
        run = _make_run(
            db_session, project.id, status=RunStatus.done,
            segment_snapshot=[
                {"id": sd0.id, "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 5.0},
                {"id": sd1.id, "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 10.0},
            ],
        )
        rs = _make_run_segment(db_session, run.id, sd1.id, index=1, status=SegmentStatus.completed)

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "4.0", "end_sec": "10.0"},
        )
        assert resp.status_code == 400
        assert "not supported" in resp.json()["detail"]

    def test_model_duration_violation_is_400(self, client, db_session):
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "1.0", "end_sec": "17.0"},  # 16s > seedance 15s max
        )
        assert resp.status_code == 400
        assert "maximum" in resp.json()["detail"]

    def test_out_of_source_duration_is_400(self, client, db_session):
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "5.0", "end_sec": "25.0"},
        )
        assert resp.status_code == 400

    def test_reversed_range_is_400(self, client, db_session):
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "9.0", "end_sec": "6.0"},
        )
        assert resp.status_code == 400

    def test_non_finite_range_is_400(self, client, db_session):
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "nan", "end_sec": "10.0"},
        )
        assert resp.status_code in (400, 422)

    def test_non_done_run_is_409(self, client, db_session):
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        run.status = RunStatus.incomplete
        db_session.commit()
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert resp.status_code == 409
        assert "done" in resp.json()["detail"]

    def test_legacy_snapshotless_run_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready, duration_sec=20.0)
        sd = _make_segment_def(db_session, project.id, 0, action="swap", start_sec=0.0, end_sec=5.0)
        run = _make_run(db_session, project.id, status=RunStatus.done, segment_snapshot=None)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "1.0", "end_sec": "4.0"},
        )
        assert resp.status_code == 409
        assert "predates" in resp.json()["detail"]

    def test_missing_run_or_segment_is_404(self, client, db_session):
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        resp = client.post(
            f"/api/v2/runs/no-such-run/segments/{rs.id}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert resp.status_code == 404
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/no-such-seg/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert resp.status_code == 404

    def test_concurrent_attempt_on_same_run_is_409(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        rid, rsid = run.id, rs.id
        s.close()

        first = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert first.status_code == 201

        second = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.0", "end_sec": "10.0"},
        )
        assert second.status_code == 409
        assert "in progress" in second.json()["detail"]

    def test_legacy_rerun_is_blocked_while_range_attempt_in_flight(self, spy_client, SessionFactory):
        """A different (or the same) segment's legacy rerun must not race a
        pending range-regeneration promotion on the same run."""
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        rid, rsid = run.id, rs.id
        s.close()

        created = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert created.status_code == 201

        legacy = client.post(f"/api/v2/runs/{rid}/segments/{rsid}/rerun")
        assert legacy.status_code == 409
        assert "in progress" in legacy.json()["detail"]

    def test_copy_is_blocked_while_range_attempt_in_flight(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        rid, rsid = run.id, rs.id
        s.close()

        created = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert created.status_code == 201

        copy_resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert copy_resp.status_code == 409
        assert "in progress" in copy_resp.json()["detail"]
        assert spy["process_run"] == []

    def test_db_level_race_loser_gets_409_and_its_own_files_are_cleaned_up(
        self, client, db_session, monkeypatch,
    ):
        """Regression for the independent audit's blocking finding #2's API
        half: the app-level `_active_range_attempt` check is a non-locking
        SELECT and cannot itself stop a genuine race — the real backstop is
        the DB partial unique index. Simulate the race window deterministically
        (monkeypatch the app-level check to say "clear" once, as it would for
        real during the actual race window) so the SECOND request's
        create_range_attempt reaches the real DB constraint and raises
        IntegrityError; assert the API turns that into the same 409 AND
        cleans up the losing attempt's own uploaded reference file — never
        the winner's."""
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        rid, rsid = run.id, rs.id

        first = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert first.status_code == 201

        # Force the app-level check to (wrongly, as it would during a real
        # race) report no active attempt for exactly the next call.
        real_active_check = api_v2_module._active_range_attempt
        calls = {"n": 0}

        def _lie_once(rid_, db_):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return real_active_check(rid_, db_)

        monkeypatch.setattr(api_v2_module, "_active_range_attempt", _lie_once)

        second = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "5.0", "end_sec": "11.0"},
            files={"reference_files": ("ref.jpg", b"fake-image-bytes", "image/jpeg")},
        )
        assert second.status_code == 409, second.text
        assert "in progress" in second.json()["detail"]

        # Exactly one active attempt exists — the winner from `first`.
        rows = (
            db_session.query(RunSegmentRangeAttempt)
            .filter(RunSegmentRangeAttempt.run_id == rid)
            .all()
        )
        assert len(rows) == 1
        assert rows[0].id == first.json()["attempt_id"]

        # The loser's own directory (namespaced by its own fresh attempt_id,
        # which never made it into a committed row) must not linger — and
        # nothing but the winner's own (never independently asserted here,
        # since the winner's request carried no reference file) may remain
        # under this segment's range_attempts directory.
        from app.storage import range_attempt_dir
        project_root = os.path.dirname(
            range_attempt_dir(first.json()["attempt_id"], rsid, rid, project.id)
        )
        leftover_dirs = [
            d for d in os.listdir(project_root)
            if d != first.json()["attempt_id"]
        ] if os.path.isdir(project_root) else []
        assert leftover_dirs == [], f"loser's attempt directory was not cleaned up: {leftover_dirs}"

    def test_list_range_attempts_returns_history_newest_first(self, client, db_session):
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(db_session)
        first = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0"},
        )
        assert first.status_code == 201

        resp = client.get(f"/api/v2/runs/{run.id}/segments/{rs.id}/range-attempts")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["id"] == first.json()["attempt_id"]
        assert body[0]["requested_start_sec"] == pytest.approx(4.6)


class TestCopyInheritsPromotedRange:
    """_build_copied_run must deep-copy the source Run's latest ACTIVE
    segment_snapshot — not the live Project's SegmentDefs, and not a failed
    candidate's never-promoted snapshot. These simulate the state a Run is
    left in immediately after pipeline_range_attempt._promote() (exercised
    directly, with stubbed kie/gdrive, in test_pipeline_range_attempt.py) —
    here the concern is purely what _build_copied_run reads from that Run."""

    def _promoted_snapshot(self, sd_keep0, sd_swap1, sd_keep2):
        # A promoted range that moved swap1's start from 5.0 -> 4.6, exactly
        # like a real range-rerun would leave it.
        return [
            {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 4.6},
            {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 4.6, "end_sec": 10.0},
            {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
        ]

    def test_copy_receives_the_promoted_snapshot_not_project_boundaries(
        self, spy_client, SessionFactory
    ):
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        promoted = self._promoted_snapshot(sd_keep0, sd_swap1, sd_keep2)
        run.segment_snapshot = promoted
        rs.prompt_override = "promoted candidate prompt"
        s.commit()
        rid = run.id
        # Mutate the LIVE Project SegmentDefs to something else entirely —
        # proving the copy does not re-derive from these.
        sd_swap1.start_sec = 5.0
        sd_swap1.end_sec = 10.0
        sd_keep0.end_sec = 5.0
        s.commit()
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        overrides = {rs2.segment_def_id: rs2.prompt_override for rs2 in nr.run_segments}
        s.close()
        assert nr.segment_snapshot == promoted
        by_id = {e["id"]: e for e in nr.segment_snapshot}
        assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(4.6)
        assert by_id[sd_keep0.id]["end_sec"] == pytest.approx(4.6)
        # Not the live (mutated) Project boundaries.
        assert by_id[sd_swap1.id]["start_sec"] != pytest.approx(5.0)
        # Per-segment prompt override carried over alongside the range.
        assert overrides.get(sd_swap1.id) == "promoted candidate prompt"

    def test_copy_deep_copies_not_shares_the_snapshot(self, spy_client, SessionFactory):
        """Mutating the copy's snapshot after the fact must not affect the
        source Run's — proves a deep copy, not a shared mutable reference."""
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        promoted = self._promoted_snapshot(sd_keep0, sd_swap1, sd_keep2)
        run.segment_snapshot = promoted
        s.commit()
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        nr.segment_snapshot[0]["end_sec"] = 999.0
        s.commit()
        s.close()

        s = SessionFactory()
        src_reloaded = s.get(Run, rid)
        s.close()
        assert src_reloaded.segment_snapshot[0]["end_sec"] == pytest.approx(4.6)

    def test_copy_of_a_copy_inherits_the_same_promoted_range(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        promoted = self._promoted_snapshot(sd_keep0, sd_swap1, sd_keep2)
        run.segment_snapshot = promoted
        s.commit()
        rid = run.id
        s.close()

        first_copy = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert first_copy.status_code == 201
        copy_id = first_copy.json()["run_id"]

        second_copy = client.post(f"/api/v2/runs/{copy_id}/copy", data={})
        assert second_copy.status_code == 201
        copy_of_copy_id = second_copy.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, copy_of_copy_id)
        s.close()
        assert nr.segment_snapshot == promoted

    def test_failed_candidate_is_not_inherited_by_copy(self, spy_client, SessionFactory):
        """A failed RunSegmentRangeAttempt row (never promoted) must have no
        effect on what Copy reads — the Run's own segment_snapshot (the last
        successfully promoted / original one) is untouched by _fail_attempt,
        and that is exactly what _build_copied_run must clone."""
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(s)
        original_snapshot = list(run.segment_snapshot)
        never_promoted = self._promoted_snapshot(sd_keep0, sd_swap1, sd_keep2)
        failed_attempt = RunSegmentRangeAttempt(
            id="failed-attempt-1",
            run_id=run.id,
            run_segment_id=rs.id,
            requested_start_sec=4.6,
            requested_end_sec=10.0,
            base_segment_snapshot=original_snapshot,
            candidate_segment_snapshot=never_promoted,
            status=RangeAttemptStatus.failed,
            error_message="injected failure — never promoted",
        )
        s.add(failed_attempt)
        s.commit()
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        s.close()
        assert nr.segment_snapshot == original_snapshot
        by_id = {e["id"]: e for e in nr.segment_snapshot}
        # The FAILED candidate's 4.6 boundary must NOT have leaked in.
        assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(5.0)


class TestSegmentDefDeletionSurvival:
    """Regression for the independent audit's blocking finding #1:
    RunSegment.segment_def_id is deliberately not a foreign key into
    segment_defs any more (see its docstring in models.py) — a completed
    Run's own RunSegments, frozen segment_snapshot and
    RunSegmentRangeAttempt history must all survive deletion of the live
    SegmentDef they were originally created against, through the REAL
    supported application path (PATCH /projects/{pid}/segments — the same
    delete this endpoint always offered, and the one pipeline_v2.
    analyze_project performs wholesale on every re-analysis), not merely a
    hand-rolled DB delete standing in for it."""

    def _make_run_with_promoted_history(self, session):
        import uuid
        from datetime import datetime, timezone

        project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_run_with_snapshot(session)
        promoted = [
            {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 4.6},
            {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 4.6, "end_sec": 10.0},
            {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
        ]
        attempt = RunSegmentRangeAttempt(
            id=str(uuid.uuid4()),
            run_id=run.id,
            run_segment_id=rs.id,
            requested_start_sec=4.6,
            requested_end_sec=10.0,
            base_segment_snapshot=list(run.segment_snapshot),
            candidate_segment_snapshot=promoted,
            status=RangeAttemptStatus.done,
            promoted_at=datetime.now(timezone.utc),
        )
        run.segment_snapshot = promoted
        session.add(attempt)
        session.commit()
        return project, run, sd_keep0, sd_swap1, sd_keep2, rs, attempt

    def test_deleting_the_live_segment_def_does_not_touch_the_run_segment_or_its_history(
        self, client, SessionFactory,
    ):
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs, attempt = (
            self._make_run_with_promoted_history(s)
        )
        pid, rid, rsid, attempt_id = project.id, run.id, rs.id, attempt.id
        target_segment_def_id = sd_swap1.id  # the exact live row rs.segment_def_id names
        snapshot_before = list(run.segment_snapshot)
        s.close()

        # Delete the live SegmentDef this RunSegment/attempt were created
        # against — through the real bulk segment-edit endpoint.
        resp = client.patch(
            f"/api/v2/projects/{pid}/segments",
            json={"deletes": [target_segment_def_id]},
        )
        assert resp.status_code == 200, resp.text

        s = SessionFactory()
        # The live def is really gone.
        assert s.get(SegmentDef, target_segment_def_id) is None
        # ...but the historical rows are completely untouched.
        reloaded_rs = s.get(RunSegment, rsid)
        reloaded_run = s.get(Run, rid)
        reloaded_attempt = s.get(RunSegmentRangeAttempt, attempt_id)
        s.close()
        assert reloaded_rs is not None
        assert reloaded_rs.segment_def_id == target_segment_def_id
        assert reloaded_rs.local_result_path == "/data/fake/result_0001.mp4"
        assert reloaded_attempt is not None
        assert reloaded_attempt.status == RangeAttemptStatus.done
        assert reloaded_run.segment_snapshot == snapshot_before

    def test_another_range_regeneration_resolves_from_the_snapshot_after_live_def_deletion(
        self, client, SessionFactory,
    ):
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs, attempt = (
            self._make_run_with_promoted_history(s)
        )
        pid, rid, rsid = project.id, run.id, rs.id
        target_segment_def_id = sd_swap1.id
        s.close()

        resp = client.patch(
            f"/api/v2/projects/{pid}/segments",
            json={"deletes": [target_segment_def_id]},
        )
        assert resp.status_code == 200, resp.text

        # A NEW range regeneration from this historical Run must still work —
        # it resolves the target/neighbour bounds entirely from the Run's own
        # frozen snapshot, never from the (now-deleted) live SegmentDef row.
        resp = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/range-rerun",
            data={"start_sec": "4.0", "end_sec": "10.0"},
        )
        assert resp.status_code == 201, resp.text

    def test_copy_after_live_def_deletion_does_not_500_and_inherits_the_snapshot(
        self, spy_client, SessionFactory,
    ):
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs, attempt = (
            self._make_run_with_promoted_history(s)
        )
        pid, rid = project.id, run.id
        target_segment_def_id = sd_swap1.id
        promoted_snapshot = list(run.segment_snapshot)
        s.close()

        resp = client.patch(
            f"/api/v2/projects/{pid}/segments",
            json={"deletes": [target_segment_def_id]},
        )
        assert resp.status_code == 200, resp.text

        # Before the fix, inserting a fresh RunSegment naming a segment_def_id
        # whose live row is gone violated the FK — this Copy would 500.
        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        s.close()
        assert nr.segment_snapshot == promoted_snapshot

    def test_batch_copy_and_copy_of_copy_after_live_def_deletion_both_inherit_the_snapshot(
        self, spy_client, SessionFactory,
    ):
        client, spy = spy_client
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs, attempt = (
            self._make_run_with_promoted_history(s)
        )
        pid, rid = project.id, run.id
        target_segment_def_id = sd_swap1.id
        promoted_snapshot = list(run.segment_snapshot)
        s.close()

        resp = client.patch(
            f"/api/v2/projects/{pid}/segments",
            json={"deletes": [target_segment_def_id]},
        )
        assert resp.status_code == 200, resp.text

        batch_resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={"runs[0][resolution]": "480p", "runs[1][resolution]": "720p"},
        )
        assert batch_resp.status_code == 201, batch_resp.text
        batch_ids = [r["run_id"] for r in batch_resp.json()["runs"]]

        copy_resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert copy_resp.status_code == 201, copy_resp.text
        copy_id = copy_resp.json()["run_id"]
        copy_of_copy_resp = client.post(f"/api/v2/runs/{copy_id}/copy", data={})
        assert copy_of_copy_resp.status_code == 201, copy_of_copy_resp.text
        copy_of_copy_id = copy_of_copy_resp.json()["run_id"]

        s = SessionFactory()
        for run_id in [*batch_ids, copy_id, copy_of_copy_id]:
            r = s.get(Run, run_id)
            assert r.segment_snapshot == promoted_snapshot, run_id
        s.close()

    def test_deleting_the_run_still_cascades_run_segments_and_range_attempts(
        self, SessionFactory,
    ):
        """Sanity check: RunSegment/RunSegmentRangeAttempt still cascade when
        the RUN itself is intentionally deleted — only the (now-removed)
        SegmentDef-triggered cascade is gone."""
        s = SessionFactory()
        project, run, sd_keep0, sd_swap1, sd_keep2, rs, attempt = (
            self._make_run_with_promoted_history(s)
        )
        rid, rsid, attempt_id = run.id, rs.id, attempt.id
        s.delete(run)
        s.commit()
        s.close()

        # A fresh session, not one whose identity map might still hold a
        # pre-delete copy of `attempt` (this session never explicitly
        # session.delete()'d it — RunSegmentRangeAttempt has no ORM-level
        # cascade relationship from Run, only the DB-level FK does the work)
        # — Session.get() checks the identity map before the DB, so reading
        # back on the SAME session could return a stale cached object even
        # though the underlying row is genuinely gone.
        verify = SessionFactory()
        assert verify.get(RunSegment, rsid) is None
        assert verify.get(RunSegmentRangeAttempt, attempt_id) is None
        verify.close()
        s.close()

    def test_legacy_snapshotless_run_segment_survives_and_falls_back_safely(
        self, SessionFactory,
    ):
        """Mandatory check #9: a Run that predates Run.segment_snapshot
        (segment_snapshot is None) must keep its EXISTING safe fallback
        behaviour — no bounds guessed after its live SegmentDef disappears —
        while ALSO now surviving the deletion itself (before the fix, the
        live def's cascade would have destroyed this RunSegment outright,
        which is a different, worse failure than "no bounds available")."""
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready, duration_sec=20.0)
        sd_swap = _make_segment_def(s, project.id, 0, action="swap", start_sec=0.0, end_sec=5.0)
        run = _make_run(s, project.id, status=RunStatus.done, segment_snapshot=None)
        rs = _make_run_segment(s, run.id, sd_swap.id, index=0, status=SegmentStatus.completed)
        rid, rsid, sd_id = run.id, rs.id, sd_swap.id
        s.close()

        s = SessionFactory()
        live_run = s.get(Run, rid)
        live_rs = s.get(RunSegment, rsid)
        assert live_run.segment_snapshot is None
        # Before deletion: the legacy fallback resolves real bounds from the
        # still-live SegmentDef.
        window = api_v2_module._segment_launch_window(live_run, live_rs)
        assert window == (pytest.approx(0.0), pytest.approx(5.0))
        s.close()

        s = SessionFactory()
        s.delete(s.get(SegmentDef, sd_id))
        s.commit()
        s.close()

        s = SessionFactory()
        assert s.get(SegmentDef, sd_id) is None
        reloaded_rs = s.get(RunSegment, rsid)
        reloaded_run = s.get(Run, rid)
        assert reloaded_rs is not None, "RunSegment must survive the live def's deletion"
        # No bounds can be resolved any more (no snapshot, no live def) — the
        # existing, documented safe behaviour: None, not a guess, not a crash.
        window = api_v2_module._segment_launch_window(reloaded_run, reloaded_rs)
        assert window == (None, None)
        s.close()


# ---------------------------------------------------------------------------
# PATCH /api/v2/projects/{pid} — editable project name
# ---------------------------------------------------------------------------


class TestProjectName:
    def test_patch_name_persists_and_shows_in_get(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)

        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "  Acme promo  "})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Acme promo"  # trimmed

        get_resp = client.get(f"/api/v2/projects/{project.id}")
        assert get_resp.json()["name"] == "Acme promo"

    def test_patch_empty_name_clears_to_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready, name="old name")
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "   "})
        assert resp.status_code == 200
        assert resp.json()["name"] is None

    def test_patch_missing_project_is_404(self, client):
        resp = client.patch("/api/v2/projects/no-such-project", json={"name": "x"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# "Remove original audio" — VideoProject.mute_source
# ---------------------------------------------------------------------------


class TestProjectMuteSource:
    def test_defaults_to_off(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        assert client.get(f"/api/v2/projects/{project.id}").json()["mute_source"] is False

    def test_create_with_mute_source(self, spy_client, SessionFactory):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"mute_source": "true"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201
        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.mute_source is True

    def test_create_gdrive_with_mute_source(self, spy_client, SessionFactory):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={
                "gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view",
                "mute_source": "true",
            },
        )
        assert response.status_code == 201
        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.mute_source is True

    def test_patch_toggles_both_ways(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)

        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"mute_source": True}
        )
        assert resp.status_code == 200
        assert resp.json()["mute_source"] is True
        assert client.get(f"/api/v2/projects/{project.id}").json()["mute_source"] is True

        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"mute_source": False}
        )
        assert resp.status_code == 200
        assert resp.json()["mute_source"] is False

    def test_patch_name_alone_leaves_mute_source_untouched(self, client, db_session):
        """Omitted fields are not reset — the two settings are independent."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        client.patch(f"/api/v2/projects/{project.id}", json={"mute_source": True})

        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "Promo"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Promo"
        assert resp.json()["mute_source"] is True


# ---------------------------------------------------------------------------
# Per-segment GPT Image 2 regeneration
# ---------------------------------------------------------------------------


def _make_preprocess_ready_run(session, project_id, **kwargs) -> Run:
    """A `done` run with preprocessing on, ready for a segment-image regen."""
    defaults = dict(
        resolution="480p",
        status=RunStatus.done,
        preprocess_enabled=True,
        preprocess_mode="per_image",
        reference_image_urls=["/data/fake/character.jpg", "/data/fake/trim.jpg"],
        preprocess_image_prompts=["", "put her in this shot"],
    )
    defaults.update(kwargs)
    return _make_run(session, project_id, **defaults)


class TestRegenerateSegmentImage:
    def _setup(self, db_session, *, run_kwargs=None, window=(10.0, 20.0)):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(
            db_session, project.id, 0, start_sec=window[0], end_sec=window[1],
        )
        run = _make_preprocess_ready_run(
            db_session, project.id,
            segment_snapshot=[
                {"id": sd.id, "index": 0, "action": "swap",
                 "start_sec": window[0], "end_sec": window[1]},
            ],
            **(run_kwargs or {}),
        )
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.completed,
        )
        return project, run, sd, rs

    def test_creates_job_with_defaults(self, client, db_session):
        _p, run, _sd, rs = self._setup(db_session)

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image",
            data={},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "queued"

        job = db_session.get(RunSegmentImageJob, body["job_id"])
        assert job.run_id == run.id
        assert job.run_segment_id == rs.id
        # Default prompt: per_image, first non-empty of prompts[1:].
        assert job.prompt == "put her in this shot"
        # Default character photo: run.reference_image_urls[0].
        assert job.input_paths == ["/data/fake/character.jpg"]
        # Default frame_sec: the segment window's midpoint, not window_start.
        assert job.frame_sec == 15.0
        assert job.status == SegmentImageStatus.queued

    def test_custom_prompt_character_url_and_frame_sec(self, client, db_session):
        _p, run, _sd, rs = self._setup(db_session)

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image",
            data={
                "prompt": "a custom prompt",
                "character_url": "https://example.com/other.jpg",
                "frame_sec": "12.5",
            },
        )
        assert resp.status_code == 200, resp.text
        job = db_session.get(RunSegmentImageJob, resp.json()["job_id"])
        assert job.prompt == "a custom prompt"
        assert job.input_paths == ["https://example.com/other.jpg"]
        assert job.frame_sec == 12.5

    def test_409_when_run_status_not_editable(self, client, db_session):
        _p, run, _sd, rs = self._setup(
            db_session, run_kwargs={"status": RunStatus.processing},
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image", data={},
        )
        assert resp.status_code == 409

    def test_409_when_preprocess_disabled(self, client, db_session):
        _p, run, _sd, rs = self._setup(
            db_session, run_kwargs={"preprocess_enabled": False},
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image", data={},
        )
        assert resp.status_code == 409
        assert "preprocessing" in resp.json()["detail"].lower()

    def test_409_when_a_job_is_already_active(self, client, db_session):
        _p, run, _sd, rs = self._setup(db_session)
        db_session.add(
            RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=15.0,
                status=SegmentImageStatus.generating,
            )
        )
        db_session.commit()

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image", data={},
        )
        assert resp.status_code == 409
        assert "in progress" in resp.json()["detail"].lower()

    def test_a_done_or_failed_prior_job_does_not_block_a_new_one(self, client, db_session):
        """Only queued/generating block — done/failed are terminal history."""
        _p, run, _sd, rs = self._setup(db_session)
        db_session.add(
            RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=15.0,
                status=SegmentImageStatus.failed, error_message="boom",
            )
        )
        db_session.commit()

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image", data={},
        )
        assert resp.status_code == 200, resp.text

    def test_400_when_no_prompt_available(self, client, db_session):
        _p, run, _sd, rs = self._setup(
            db_session, run_kwargs={"preprocess_image_prompts": None},
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image", data={},
        )
        assert resp.status_code == 400
        assert "prompt" in resp.json()["detail"].lower()

    def test_400_when_frame_sec_outside_window(self, client, db_session):
        _p, run, _sd, rs = self._setup(db_session, window=(10.0, 20.0))
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image",
            data={"frame_sec": "999"},
        )
        assert resp.status_code == 400
        assert "window" in resp.json()["detail"].lower()

    def test_400_when_no_character_photo_available(self, client, db_session):
        _p, run, _sd, rs = self._setup(
            db_session, run_kwargs={"reference_image_urls": []},
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image", data={},
        )
        assert resp.status_code == 400

    def test_all_as_refs_mode_uses_the_single_prompt(self, client, db_session):
        _p, run, _sd, rs = self._setup(
            db_session,
            run_kwargs={
                "preprocess_mode": "all_as_refs",
                "preprocess_image_prompts": ["combine everything"],
            },
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image", data={},
        )
        assert resp.status_code == 200, resp.text
        job = db_session.get(RunSegmentImageJob, resp.json()["job_id"])
        assert job.prompt == "combine everything"

    def test_enqueues_the_job(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        _p, run, _sd, rs = self._setup(s)
        rid, rsid = run.id, rs.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/segments/{rsid}/regenerate-image", data={},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["job_id"] in spy["segment_image"]

    def test_missing_run_is_404(self, client, db_session):
        resp = client.post(
            "/api/v2/runs/no-such-run/segments/no-such-seg/regenerate-image", data={},
        )
        assert resp.status_code == 404

    def test_missing_segment_is_404(self, client, db_session):
        _p, run, _sd, _rs = self._setup(db_session)
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/no-such-seg/regenerate-image", data={},
        )
        assert resp.status_code == 404


class TestSegmentEditGatesRefuseWhileImageJobActive:
    """patch_run_segment and rerun_segment both refuse while an image job is
    active on that segment — rerun_segment especially: it reads refs ONCE at
    submit time, so a still finishing afterward would bill a video generated
    from the OLD image, silently."""

    def _setup_with_active_job(self, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.completed,
        )
        db_session.add(
            RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
                status=SegmentImageStatus.generating,
            )
        )
        db_session.commit()
        return run, rs

    def test_patch_run_segment_409(self, client, db_session):
        run, rs = self._setup_with_active_job(db_session)
        resp = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "new prompt"},
        )
        assert resp.status_code == 409
        assert "gpt image" in resp.json()["detail"].lower()

    def test_rerun_segment_409(self, client, db_session):
        run, rs = self._setup_with_active_job(db_session)
        resp = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert resp.status_code == 409
        assert "gpt image" in resp.json()["detail"].lower()

    def test_rerun_segment_409_when_a_DIFFERENT_sibling_has_the_active_job(
        self, client, db_session
    ):
        """Cursor-gate catch (2nd pass): re-enqueuing submit resubmits EVERY
        active swap segment, not just the one targeted by /rerun — so a
        gate that only checks THIS segment's own job misses a failed sibling
        whose image job is still generating. That sibling would be reset and
        resubmitted from its stale reference state too."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd_a = _make_segment_def(db_session, project.id, 0)
        sd_b = _make_segment_def(db_session, project.id, 1)
        run = _make_run(db_session, project.id, status=RunStatus.incomplete)
        rs_a = _make_run_segment(
            db_session, run.id, sd_a.id, index=0, status=SegmentStatus.failed,
        )
        rs_b = _make_run_segment(
            db_session, run.id, sd_b.id, index=1, status=SegmentStatus.failed,
        )
        db_session.add(
            RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs_a.id, prompt="p", frame_sec=1.0,
                status=SegmentImageStatus.generating,
            )
        )
        db_session.commit()

        # Re-run B (not A, which owns the active job) — must still refuse.
        resp = client.post(f"/api/v2/runs/{run.id}/segments/{rs_b.id}/rerun")
        assert resp.status_code == 409
        assert "gpt image" in resp.json()["detail"].lower()

    def test_patch_run_segment_ok_once_job_is_terminal(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.completed,
        )
        db_session.add(
            RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
                status=SegmentImageStatus.done, output_path="/data/fake/o.png",
            )
        )
        db_session.commit()

        resp = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "new prompt"},
        )
        assert resp.status_code == 200


class TestApplySegmentImageJob:
    def _setup(self, db_session, job_status=SegmentImageStatus.done):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.completed,
        )
        job = RunSegmentImageJob(
            run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
            input_paths=["/data/fake/char.jpg", "/data/fake/frame.jpg"],
            status=job_status, output_path="/data/fake/output.png",
        )
        db_session.add(job)
        db_session.commit()
        return run, rs, job

    def test_apply_sets_generated_reference_urls(self, client, db_session):
        run, rs, job = self._setup(db_session)
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/apply"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["generated_reference_urls"] == [
            "/data/fake/char.jpg", "/data/fake/output.png",
        ]
        assert body["generated_image_job_id"] == job.id

    def test_apply_409_when_job_not_done(self, client, db_session):
        run, rs, job = self._setup(db_session, job_status=SegmentImageStatus.generating)
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/apply"
        )
        assert resp.status_code == 409

    def test_apply_404_for_wrong_segment(self, client, db_session):
        run, rs, job = self._setup(db_session)
        other_sd = _make_segment_def(db_session, run.project_id, 1)
        other_rs = _make_run_segment(
            db_session, run.id, other_sd.id, status=SegmentStatus.completed,
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{other_rs.id}/image-jobs/{job.id}/apply"
        )
        assert resp.status_code == 404


class TestClearSegmentGeneratedImage:
    def test_clears_both_fields(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.completed,
            generated_reference_urls=["/data/fake/char.jpg", "/data/fake/gen.png"],
            generated_image_job_id="fake-job-id",
        )
        db_session.commit()

        resp = client.delete(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/generated-image"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["generated_reference_urls"] is None
        assert body["generated_image_job_id"] is None

    def test_409_when_run_not_editable(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)
        rs = _make_run_segment(db_session, run.id, sd.id)
        resp = client.delete(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/generated-image"
        )
        assert resp.status_code == 409


class TestSegmentImageJobFile:
    def _setup(self, db_session, **job_kwargs):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.completed,
        )
        defaults = dict(
            run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
            status=SegmentImageStatus.done,
        )
        defaults.update(job_kwargs)
        job = RunSegmentImageJob(**defaults)
        db_session.add(job)
        db_session.commit()
        return run, rs, job

    def test_serves_output(self, client, db_session, tmp_path):
        out = tmp_path / "out.png"
        out.write_bytes(b"\x89PNG")
        run, rs, job = self._setup(db_session, output_path=str(out))
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/image"
        )
        assert resp.status_code == 200

    def test_output_missing_is_404(self, client, db_session):
        run, rs, job = self._setup(db_session, output_path=None)
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/image"
        )
        assert resp.status_code == 404

    def test_serves_frame(self, client, db_session, tmp_path):
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"\xff\xd8")
        run, rs, job = self._setup(
            db_session, input_paths=["/data/fake/char.jpg", str(frame)],
        )
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/image",
            params={"which": "frame"},
        )
        assert resp.status_code == 200

    def test_character_as_remote_url_is_404(self, client, db_session):
        run, rs, job = self._setup(
            db_session, input_paths=["https://example.com/char.jpg", "/data/fake/f.jpg"],
        )
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/image",
            params={"which": "character"},
        )
        assert resp.status_code == 404

    def test_character_as_local_path_is_served(self, client, db_session, tmp_path):
        char = tmp_path / "char.jpg"
        char.write_bytes(b"\xff\xd8")
        run, rs, job = self._setup(
            db_session, input_paths=[str(char), "/data/fake/f.jpg"],
        )
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/image",
            params={"which": "character"},
        )
        assert resp.status_code == 200

    def test_wrong_segment_is_404(self, client, db_session):
        run, rs, job = self._setup(db_session)
        other_sd = _make_segment_def(db_session, run.project_id, 1)
        other_rs = _make_run_segment(db_session, run.id, other_sd.id)
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{other_rs.id}/image-jobs/{job.id}/image"
        )
        assert resp.status_code == 404


class TestSegmentOverrideReferenceFile:
    def test_serves_local_override_reference(self, client, db_session, tmp_path):
        ref = tmp_path / "ref.jpg"
        ref.write_bytes(b"\xff\xd8")
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id,
            reference_image_urls_override=[str(ref)],
        )
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/override-reference",
            params={"index": 0},
        )
        assert resp.status_code == 200

    def test_index_out_of_range_is_404(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id)
        resp = client.get(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/override-reference",
            params={"index": 0},
        )
        assert resp.status_code == 404


class TestDefaultSegmentImagePrompt:
    """Unit tests for api_v2._default_segment_image_prompt."""

    def test_per_image_uses_first_non_empty_trim_prompt(self, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            preprocess_mode="per_image",
            preprocess_image_prompts=["", "", "second trim prompt"],
        )
        assert api_v2_module._default_segment_image_prompt(run) == "second trim prompt"

    def test_all_as_refs_uses_the_single_prompt(self, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            preprocess_mode="all_as_refs",
            preprocess_image_prompts=["the one prompt"],
        )
        assert api_v2_module._default_segment_image_prompt(run) == "the one prompt"

    def test_blank_when_no_prompts(self, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, preprocess_image_prompts=None)
        assert api_v2_module._default_segment_image_prompt(run) == ""


# ---------------------------------------------------------------------------
# Segmentation cap — VideoProject.max_segment_sec
# ---------------------------------------------------------------------------


class TestProjectMaxSegmentSec:
    """The analyze-time cap on proposed segment length.

    NULL is a meaningful value: "no explicit choice", which analysis resolves to
    ai_models.UNIVERSAL_MAX_SEGMENT_SEC (the shortest per-clip ceiling in the
    registry). The API stores NULL rather than that number so a project is not
    frozen to today's default. Accepted band is
    [api_v2._MIN_SEGMENT_CAP_SEC, ABSOLUTE_MAX_SEGMENT_SEC] = [4.0, 30.0], where
    the floor is the largest ``min_duration_sec`` in the registry — the shortest
    clip every backend will actually generate. It is deliberately NOT
    MIN_SWAP_VIDEO_SEC (2.0), which bounds the reference clip we upload, not the
    output we pay for.
    """

    # Bounds are read from the code, not restated, so that adding a model with a
    # different min_duration_sec moves these tests with it instead of breaking
    # them for the wrong reason.
    FLOOR = api_v2_module._MIN_SEGMENT_CAP_SEC
    CEILING = ai_models.ABSOLUTE_MAX_SEGMENT_SEC

    # -- POST /api/v2/projects -------------------------------------------

    def test_omitted_is_stored_as_null(self, spy_client, SessionFactory):
        """The default path must not write a number — NULL means "no choice"."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.max_segment_sec is None

    def test_get_exposes_the_field(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        body = client.get(f"/api/v2/projects/{project.id}").json()
        assert "max_segment_sec" in body
        assert body["max_segment_sec"] is None

    @pytest.mark.parametrize(
        "value, expected",
        [
            (30, 30.0),
            (15, 15.0),
            # The floor itself — the smallest cap any backend can honour, and the
            # boundary the band is inclusive of.
            (api_v2_module._MIN_SEGMENT_CAP_SEC, api_v2_module._MIN_SEGMENT_CAP_SEC),
        ],
    )
    def test_create_with_explicit_cap(
        self, spy_client, SessionFactory, value, expected
    ):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": value},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.max_segment_sec == expected

    def test_blank_form_value_is_stored_as_null(self, spy_client, SessionFactory):
        """The dashboard's default <option value=""> must not 422 or store 0.

        The new-project form posts the select verbatim, so the "works with every
        model (default)" choice arrives as an empty string — it has to land as
        NULL, the same as omitting the field.
        """
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": ""},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.max_segment_sec is None

    def test_create_gdrive_with_cap(self, spy_client, SessionFactory):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={
                "gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view",
                "max_segment_sec": 30,
            },
        )
        assert response.status_code == 201

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.max_segment_sec == 30.0

    @pytest.mark.parametrize(
        "value",
        [
            0,      # no output at all
            1.5,    # below even the reference-clip floor
            45,     # beyond every model's ceiling
            -5,
        ],
    )
    def test_create_out_of_range_is_400(self, client, value):
        response = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": value},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "max_segment_sec" in response.json()["detail"]

    @pytest.mark.parametrize("value", [2, 2.0, 3, 3.5])
    def test_create_below_the_output_floor_is_400(self, spy_client, value):
        """Caps under the registry's output floor are refused, not honoured.

        These used to be ACCEPTED, because the floor was MIN_SWAP_VIDEO_SEC (2.0)
        — the shortest reference clip a model takes as input — rather than the
        shortest clip it will emit. Nothing can generate a 2s or 3s segment:
        ai_models.duration_for() clamps every request up to min_duration_sec (4),
        so a 9s face interval cut at 2.0 becomes five 1.8s chunks each billed as
        4s = 20s of generated video for 9s of source, and the sub-2s chunks are
        additionally padded back up at submit time, stealing footage from their
        neighbours. Rejecting is the only honest answer: silently ignoring the
        operator's cap would be just as wrong as obeying it.
        """
        client, spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": value},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400, response.text
        assert "max_segment_sec" in response.json()["detail"]
        # Rejected before anything was persisted or queued.
        assert spy["analyze_project"] == []

    def test_floor_is_the_registry_output_minimum(self):
        """Pin WHERE the floor comes from, so it tracks the registry.

        Not hardcoded as 4.0: the bound is "the largest min_duration_sec any
        model declares", i.e. the shortest clip EVERY backend can produce. A new
        model with a higher floor must raise this bound automatically, and one
        with a lower floor must not (a cap only some models honour is still
        un-generatable on the others).
        """
        assert self.FLOOR == float(
            max(spec.min_duration_sec for spec in ai_models.AI_MODELS.values())
        )
        # Sanity: the accepted band is non-empty and sits below the ceiling.
        assert self.FLOOR < self.CEILING
        # And it really is above the reference-clip floor it used to be confused
        # with — otherwise this whole fix would be a no-op.
        assert self.FLOOR > ai_models.MIN_SWAP_VIDEO_SEC

    def test_the_floor_is_accepted_and_one_step_below_it_is_not(
        self, spy_client, SessionFactory
    ):
        """The band is inclusive at the bottom, and nothing under it gets in.

        ``math.nextafter`` gives the largest float strictly below the floor, so
        this pins the exact boundary rather than a value comfortably far from it
        (a ``>=`` accidentally written as ``>`` would pass a 3.9-vs-4.0 test).
        """
        client, _spy = spy_client
        ok = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": self.FLOOR},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert ok.status_code == 201, ok.text
        session = SessionFactory()
        project = session.get(VideoProject, ok.json()["project_id"])
        session.close()
        assert project.max_segment_sec == self.FLOOR

        just_below = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": repr(math.nextafter(self.FLOOR, 0.0))},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert just_below.status_code == 400, just_below.text

    def test_the_ceiling_is_accepted_and_one_step_above_it_is_not(self, spy_client):
        """Same boundary check at the top of the band."""
        client, _spy = spy_client
        ok = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": self.CEILING},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert ok.status_code == 201, ok.text

        just_above = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": repr(math.nextafter(self.CEILING, math.inf))},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert just_above.status_code == 400, just_above.text

    @pytest.mark.parametrize("value", ["nan", "NaN", "-nan"])
    def test_create_nan_is_400(self, spy_client, value):
        """NaN must not reach the column — see test_patch_nan_is_400 for why.

        The form parser hands "nan" to Pydantic's float validator, which accepts
        it, so this is a real request an automation or script can send.
        """
        client, spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": value},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400, response.text
        assert "max_segment_sec" in response.json()["detail"]
        assert spy["analyze_project"] == []

    @pytest.mark.parametrize("value", ["inf", "Infinity", "-inf", "-Infinity"])
    def test_create_infinite_is_400(self, spy_client, value):
        """The interval form rejects both infinities as well as NaN."""
        client, spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"max_segment_sec": value},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400, response.text
        assert "max_segment_sec" in response.json()["detail"]
        assert spy["analyze_project"] == []

    # -- PATCH /api/v2/projects/{pid} ------------------------------------

    def test_patch_sets_the_cap(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)

        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"max_segment_sec": 30}
        )
        assert resp.status_code == 200
        assert resp.json()["max_segment_sec"] == 30.0
        assert (
            client.get(f"/api/v2/projects/{project.id}").json()["max_segment_sec"]
            == 30.0
        )

    def test_patch_explicit_null_clears_the_cap(self, client, db_session):
        """`{"max_segment_sec": null}` resets to the universal default.

        This is the settings dropdown's default option. It must be told apart
        from an omitted key — treating both as "no change" made choosing the
        default a silent no-op on a project already set to 15s/30s.
        """
        project = _make_project(db_session, status=ProjectStatus.ready)
        client.patch(f"/api/v2/projects/{project.id}", json={"max_segment_sec": 30})

        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"max_segment_sec": None}
        )
        assert resp.status_code == 200
        assert resp.json()["max_segment_sec"] is None

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).max_segment_sec is None

    def test_patch_omitting_the_key_leaves_it_untouched(self, client, db_session):
        """An omitted key is NOT the same as an explicit null."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        client.patch(f"/api/v2/projects/{project.id}", json={"max_segment_sec": 15})

        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "Promo"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Promo"
        assert resp.json()["max_segment_sec"] == 15.0

    @pytest.mark.parametrize("value", [0, 1.5, 45, -1])
    def test_patch_out_of_range_is_400(self, client, db_session, value):
        project = _make_project(
            db_session, status=ProjectStatus.ready, max_segment_sec=15.0
        )
        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"max_segment_sec": value}
        )
        assert resp.status_code == 400
        assert "max_segment_sec" in resp.json()["detail"]

        # The rejected value never lands on the row.
        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).max_segment_sec == 15.0

    @pytest.mark.parametrize("value", [2, 3, 3.5])
    def test_patch_below_the_output_floor_is_400(self, client, db_session, value):
        """The values POST used to wrongly accept are refused on PATCH too.

        Both entry points share _validate_max_segment_sec; this pins that they
        stay shared rather than drifting into two different bands.
        """
        project = _make_project(
            db_session, status=ProjectStatus.ready, max_segment_sec=15.0
        )
        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"max_segment_sec": value}
        )
        assert resp.status_code == 400, resp.text
        assert "max_segment_sec" in resp.json()["detail"]

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).max_segment_sec == 15.0

    def test_patch_accepts_exactly_the_floor(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"max_segment_sec": self.FLOOR}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["max_segment_sec"] == self.FLOOR

    # Raw JSON bodies, not dicts. json={"max_segment_sec": float("nan")} would be
    # re-serialised by httpx's encoder and never proves what the server parsed, so
    # these send the bytes on the wire verbatim. `NaN`/`Infinity` are the
    # non-standard literals Python's json module emits and Pydantic's JSON parser
    # accepts; the quoted "nan" form is what a form post or a lax client sends.
    _NAN_BODIES = [
        b'{"max_segment_sec": NaN}',
        b'{"max_segment_sec": "nan"}',
        b'{"max_segment_sec": "NaN"}',
    ]

    @pytest.mark.parametrize("body", _NAN_BODIES)
    def test_the_nan_bodies_really_parse_to_nan(self, body):
        """Guard the guard: prove these payloads produce a float NaN.

        Without this, a payload Pydantic rejected outright (422) would make the
        400 assertions below pass for the wrong reason and the bounds check would
        never be exercised at all.
        """
        parsed = ProjectUpdate.model_validate_json(body)
        assert "max_segment_sec" in parsed.model_fields_set
        assert isinstance(parsed.max_segment_sec, float)
        assert math.isnan(parsed.max_segment_sec)

    @pytest.mark.parametrize("body", _NAN_BODIES)
    def test_patch_nan_is_400(self, client, db_session, body):
        """NaN must be rejected — the old ``v < lo or v > hi`` let it through.

        Every comparison against NaN is False, so both halves of that guard were
        False and NaN was stored (Postgres DOUBLE PRECISION takes it). Analysis
        then does min(nan, 30.0) -> nan -> face.split_max_duration ->
        math.ceil(dur / nan) and raises "cannot convert float NaN to integer", so
        the project fails with an opaque message and fails again on every retry.
        """
        project = _make_project(
            db_session, status=ProjectStatus.ready, max_segment_sec=15.0
        )
        resp = client.patch(
            f"/api/v2/projects/{project.id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400, resp.text
        assert "max_segment_sec" in resp.json()["detail"]

        # The NaN never reached the column — the project is still usable.
        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).max_segment_sec == 15.0

    @pytest.mark.parametrize(
        "body",
        [
            b'{"max_segment_sec": Infinity}',
            b'{"max_segment_sec": -Infinity}',
            b'{"max_segment_sec": "inf"}',
            b'{"max_segment_sec": "-inf"}',
        ],
    )
    def test_patch_infinite_is_400(self, client, db_session, body):
        """+inf/-inf were already rejected by the old guard; keep them rejected.

        The interval form must not regress this while fixing NaN — and unlike
        NaN, SQLite *does* persist an infinity, so a leak here would be visible.
        """
        project = _make_project(
            db_session, status=ProjectStatus.ready, max_segment_sec=15.0
        )
        resp = client.patch(
            f"/api/v2/projects/{project.id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400, resp.text
        assert "max_segment_sec" in resp.json()["detail"]

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).max_segment_sec == 15.0

    def test_patch_does_not_reanalyze_or_recut_segments(
        self, spy_client, SessionFactory, db_session
    ):
        """The cap is consumed at ANALYZE time, so a PATCH is inert on an
        already-analyzed project: no re-enqueue, no re-cut of the existing
        SegmentDef rows. Getting longer segments means analyzing again.
        """
        client, spy = spy_client
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0)
        _make_segment_def(db_session, project.id, 1, action="keep", has_face=False)
        before = [
            (sd.index, sd.start_sec, sd.end_sec, sd.action)
            for sd in sorted(project.segments, key=lambda s: s.index)
        ]

        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"max_segment_sec": 30}
        )
        assert resp.status_code == 200
        assert resp.json()["max_segment_sec"] == 30.0

        # No analysis was re-queued …
        assert spy["analyze_project"] == []

        # … and the segments are byte-for-byte the ones analysis already cut.
        session = SessionFactory()
        after_project = session.get(VideoProject, project.id)
        after = [
            (sd.index, sd.start_sec, sd.end_sec, sd.action)
            for sd in sorted(after_project.segments, key=lambda s: s.index)
        ]
        session.close()
        assert after == before

    def test_patch_missing_project_is_404(self, client):
        resp = client.patch(
            "/api/v2/projects/does-not-exist", json={"max_segment_sec": 30}
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/v2/runs/{rid} — editable run name
# ---------------------------------------------------------------------------


class TestRunName:
    def test_patch_name_persists_and_shows_in_get(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id)

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"name": "  Blonde lead  "})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Blonde lead"  # trimmed

        get_resp = client.get(f"/api/v2/runs/{run.id}")
        assert get_resp.json()["name"] == "Blonde lead"

    def test_patch_empty_name_clears_to_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, name="old name")
        resp = client.patch(f"/api/v2/runs/{run.id}", json={"name": "   "})
        assert resp.status_code == 200
        assert resp.json()["name"] is None

    def test_patch_missing_run_is_404(self, client):
        resp = client.patch("/api/v2/runs/no-such-run", json={"name": "x"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Run language — the Job ID suffix the renaming pipeline reads
# ---------------------------------------------------------------------------


class TestRunLanguage:
    def test_create_run_stores_the_chosen_language(self, spy_client, SessionFactory):
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": "ja"},
        )
        assert resp.status_code == 201, resp.text

        session = SessionFactory()
        run = session.get(Run, resp.json()["run_id"])
        assert run.language == "ja"
        assert run.short_id == f"{run.id[:8]}-ja"
        session.close()

    def test_create_run_without_language_stays_null(self, spy_client, SessionFactory):
        """Unset is the pre-feature state: bare id, renames as English."""
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p"},
        )
        assert resp.status_code == 201, resp.text

        session = SessionFactory()
        run = session.get(Run, resp.json()["run_id"])
        assert run.language is None
        assert run.short_id == run.id[:8]
        session.close()

    def test_create_run_rejects_an_unsupported_language(self, client, db_session):
        """A code the renaming pipeline can't parse is refused, not stored."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": "nl"},
        )
        assert resp.status_code == 400
        assert "language" in resp.json()["detail"]

    def test_patch_language_changes_the_short_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id)

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "ja"})
        assert resp.status_code == 200
        assert resp.json()["language"] == "ja"

        db_session.expire_all()
        assert db_session.get(Run, run.id).short_id == f"{run.id[:8]}-ja"

    def test_patch_empty_language_clears_it(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="ja")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": ""})
        assert resp.status_code == 200
        assert resp.json()["language"] is None

    def test_patch_rejects_an_unsupported_language(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="ja")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "nl"})
        assert resp.status_code == 400
        db_session.expire_all()
        assert db_session.get(Run, run.id).language == "ja"  # unchanged

    def test_patch_name_alone_leaves_language_untouched(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="ja")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"name": "take 2"})
        assert resp.status_code == 200
        assert resp.json()["language"] == "ja"


# ---------------------------------------------------------------------------
# DELETE /api/v2/projects/{pid} and /api/v2/runs/{rid} — DB + disk
# ---------------------------------------------------------------------------


class TestDeleteProject:
    def test_delete_removes_db_rows_and_disk(self, client, db_session, SessionFactory):
        from app.storage import project_dir

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        pid, rid = project.id, run.id

        pdir = project_dir(pid)  # creates the dir
        assert os.path.isdir(pdir)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 204
        assert not os.path.exists(pdir)

        s = SessionFactory()
        assert s.get(VideoProject, pid) is None
        assert s.get(Run, rid) is None  # cascade
        s.close()

    def test_delete_blocked_while_analyzing(self, client, db_session):
        from app.storage import project_dir

        project = _make_project(db_session, status=ProjectStatus.analyzing)
        pdir = project_dir(project.id)
        resp = client.delete(f"/api/v2/projects/{project.id}")
        assert resp.status_code == 409
        assert os.path.isdir(pdir)  # untouched

    def test_delete_blocked_while_run_active(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(db_session, project.id, status=RunStatus.processing)
        resp = client.delete(f"/api/v2/projects/{project.id}")
        assert resp.status_code == 409

    def test_delete_missing_project_is_404(self, client):
        assert client.delete("/api/v2/projects/no-such-project").status_code == 404


class TestReanalyzeProject:
    """POST /projects/{pid}/analyze — the only way an analyze-time setting
    (max_segment_sec, hook_sec) can be made to take effect after creation.

    analyze_project deletes every live SegmentDef before writing the new
    proposal. RunSegment.segment_def_id is not an FK into that table any
    more (models.py; see TestSegmentDefDeletionSurvival above for the
    dedicated coverage), so this no longer cascades into any Run's own
    RunSegments or RunSegmentRangeAttempt history — only the live
    SegmentDef rows themselves, and a legacy pre-snapshot run's launch-window
    display, are affected. The guards below still mirror
    DELETE /projects/{pid}: both delete rows a worker may be holding while a
    run is actively processing.
    """

    def test_enqueues_analysis_and_returns_202(self, spy_client, db_session):
        client, spy = spy_client
        project = _make_project(db_session, status=ProjectStatus.ready)

        resp = client.post(f"/api/v2/projects/{project.id}/analyze")
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["project_id"] == project.id
        # The plain enum VALUE, never a Python repr — this string reaches the UI.
        # AICP-350: the admission commits `analyzing` (counted by the drain).
        assert body["status"] == "analyzing"
        assert spy["analyze_project"] == [project.id]

    def test_a_failed_project_can_be_re_analyzed(self, spy_client, db_session):
        """failed → analyzing is a legal transition; a bad source that has since
        been fixed must not require deleting the project."""
        client, spy = spy_client
        project = _make_project(db_session, status=ProjectStatus.failed)

        assert client.post(f"/api/v2/projects/{project.id}/analyze").status_code == 202
        assert spy["analyze_project"] == [project.id]

    def test_already_analyzing_is_409(self, spy_client, db_session):
        client, spy = spy_client
        project = _make_project(db_session, status=ProjectStatus.analyzing)

        resp = client.post(f"/api/v2/projects/{project.id}/analyze")
        assert resp.status_code == 409, resp.text
        assert "analyzing" in resp.json()["detail"]
        assert spy["analyze_project"] == []

    @pytest.mark.parametrize(
        "run_status",
        [
            RunStatus.queued,
            RunStatus.processing,
            RunStatus.stitching,
            RunStatus.delivering,
        ],
    )
    def test_active_run_is_409(self, spy_client, db_session, run_status):
        """A live run's RunSegments would be deleted out from under the worker."""
        client, spy = spy_client
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(db_session, project.id, status=run_status)

        resp = client.post(f"/api/v2/projects/{project.id}/analyze")
        assert resp.status_code == 409, resp.text
        assert "active" in resp.json()["detail"]
        assert spy["analyze_project"] == []

    @pytest.mark.parametrize(
        "run_status",
        [
            RunStatus.created,
            RunStatus.done,
            RunStatus.failed,
            RunStatus.incomplete,
        ],
    )
    def test_terminal_runs_do_not_block(self, spy_client, db_session, run_status):
        """Same statuses DELETE lets through: nothing is processing them.

        A `done` run keeps its row and its delivered video — only the
        per-segment breakdown goes.
        """
        client, spy = spy_client
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(db_session, project.id, status=run_status)

        assert client.post(f"/api/v2/projects/{project.id}/analyze").status_code == 202
        assert spy["analyze_project"] == [project.id]

    def test_a_done_run_survives_the_request_itself(
        self, spy_client, db_session, SessionFactory
    ):
        """The endpoint only enqueues — it must not pre-emptively delete
        anything itself. analyze_project owns the segment rewrite."""
        client, _spy = spy_client
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(
            db_session, project.id, status=RunStatus.done,
            result_local_path="/tmp/final.mp4",
        )
        rs = _make_run_segment(db_session, run.id, sd.id)
        rid, sd_id, rs_id = run.id, sd.id, rs.id

        assert client.post(f"/api/v2/projects/{project.id}/analyze").status_code == 202

        session = SessionFactory()
        assert session.get(Run, rid).result_local_path == "/tmp/final.mp4"
        assert session.get(SegmentDef, sd_id) is not None
        assert session.get(RunSegment, rs_id) is not None
        session.close()

    def test_localisation_project_is_marked_transcript_pending(
        self, spy_client, db_session
    ):
        """Analysis chains a fresh transcription, so the transcript is spoken
        for from now — same reason project creation writes "pending"."""
        client, spy = spy_client
        project = _localisation_project(db_session, transcript_error="kie.ai 500")

        assert client.post(f"/api/v2/projects/{project.id}/analyze").status_code == 202

        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id)
        assert stored.transcript_status == "pending"
        assert stored.transcript_error is None
        # The PREVIOUS transcript stays readable until the new one lands.
        assert stored.transcript["lines"][0]["text"] == _HOOK_LINES[0]["text"]
        assert spy["analyze_project"] == [project.id]

    def test_non_localisation_transcript_is_untouched(self, spy_client, db_session):
        """No transcription is chained for other types, so nothing is promised."""
        client, _spy = spy_client
        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            transcript_status="ready",
            transcript=_transcript(_HOOK_LINES),
        )

        assert client.post(f"/api/v2/projects/{project.id}/analyze").status_code == 202

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).transcript_status == "ready"

    def test_unknown_project_is_404(self, spy_client):
        client, spy = spy_client
        assert client.post("/api/v2/projects/nope/analyze").status_code == 404
        assert spy["analyze_project"] == []


class TestDeleteRun:
    def test_delete_removes_db_rows_and_disk(self, client, db_session, SessionFactory):
        from app.storage import run_clips_dir, run_dir

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id)
        rid, rs_id = run.id, rs.id

        run_clips_dir(rid, project.id)  # creates runs/<rid>/clips
        rdir = run_dir(rid, project.id)
        assert os.path.isdir(rdir)

        resp = client.delete(f"/api/v2/runs/{rid}")
        assert resp.status_code == 204
        assert not os.path.exists(rdir)

        s = SessionFactory()
        assert s.get(Run, rid) is None
        assert s.get(RunSegment, rs_id) is None  # cascade
        s.close()

    def test_delete_blocked_while_active(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.stitching)
        resp = client.delete(f"/api/v2/runs/{run.id}")
        assert resp.status_code == 409

    def test_delete_missing_run_is_404(self, client):
        assert client.delete("/api/v2/runs/no-such-run").status_code == 404


# ---------------------------------------------------------------------------
# Single-file uploads (regression: Optional[List[UploadFile]] coerced a single
# file to a 422 "Input should be a valid list" on FastAPI 0.104)
# ---------------------------------------------------------------------------


class TestSingleFileUpload:
    def test_patch_segment_with_one_reference_file(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        resp = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "new prompt"},
            files={"reference_files": ("a.jpg", io.BytesIO(b"img-bytes"), "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["reference_image_urls_override"]) == 1

    def test_rerun_segment_with_one_reference_file(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun",
            data={"prompt": "redo"},
            files={"reference_files": ("a.jpg", io.BytesIO(b"img-bytes"), "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text

    def test_create_run_with_one_reference_file(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
            files={"reference_files": ("a.jpg", io.BytesIO(b"img-bytes"), "image/jpeg")},
        )
        assert resp.status_code == 201, resp.text
        run_id = resp.json()["run_id"]
        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert len(run.reference_image_urls) == 1


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/copy — duplicate a run at a new resolution
# ---------------------------------------------------------------------------


class TestCopyRun:
    def test_copy_clones_config_and_enqueues(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, name="Test", prompt="hello there",
            model="seedance", resolution="480p", audio_mode="seedance",
            status=RunStatus.done, language="ja",
        )
        pid, rid = project.id, run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "1080p"})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        assert resp.json()["status"] == "queued"
        assert new_id in spy["process_run"]
        assert new_id != rid

        s = SessionFactory()
        nr = s.get(Run, new_id)
        s.close()
        assert nr.project_id == pid
        assert nr.prompt == "hello there"
        assert nr.model == "seedance"
        assert nr.audio_mode == "seedance"
        assert nr.resolution == "1080p"
        # A copy is the same creative in the same language — this is also how
        # sheet-driven automation batches inherit it (they reuse
        # _build_copied_run), so a Japanese template can't spawn English runs.
        assert nr.language == "ja"
        assert nr.status == RunStatus.queued

    def test_copy_uses_custom_name(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, resolution="480p", status=RunStatus.done)
        rid = run.id
        s.close()
        resp = client.post(
            f"/api/v2/runs/{rid}/copy", data={"resolution": "720p", "name": "Prod cut"}
        )
        assert resp.status_code == 201
        s = SessionFactory()
        nr = s.get(Run, resp.json()["run_id"])
        s.close()
        assert nr.name == "Prod cut"

    def test_copy_clones_segment_overrides(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        sd = _make_segment_def(s, project.id, 0)
        run = _make_run(s, project.id, resolution="480p", status=RunStatus.done)
        _make_run_segment(
            s, run.id, sd.id, status=SegmentStatus.completed,
            prompt_override="tuned segment prompt",
        )
        rid, sd_id = run.id, sd.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "720p"})
        assert resp.status_code == 201
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        overrides = [rs for rs in nr.run_segments if rs.prompt_override]
        s.close()
        assert len(overrides) == 1
        assert overrides[0].segment_def_id == sd_id
        assert overrides[0].prompt_override == "tuned segment prompt"

    def test_copy_clones_local_reference_file(self, spy_client, SessionFactory, tmp_path):
        client, spy = spy_client
        ref = tmp_path / "face.jpg"
        ref.write_bytes(b"img-bytes")
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=[str(ref)],
        )
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "720p"})
        assert resp.status_code == 201
        s = SessionFactory()
        nr = s.get(Run, resp.json()["run_id"])
        new_refs = list(nr.reference_image_urls)
        s.close()
        assert len(new_refs) == 1
        assert new_refs[0] != str(ref)       # copied into the new run's dir
        assert os.path.exists(new_refs[0])   # and the copy is on disk

    def test_copy_with_new_reference_url_replaces_refs(self, spy_client, SessionFactory):
        """A new reference URL on copy replaces the run-level references."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face1.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={"reference_urls": "https://new.example/newface.jpg"},
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        assert new_id in spy["process_run"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        refs = list(nr.reference_image_urls)
        res = nr.resolution
        s.close()
        assert refs == ["https://new.example/newface.jpg"]
        # resolution omitted → defaults to the source run's resolution
        assert res == "480p"

    def test_copy_new_ref_drops_segment_photo_override_keeps_prompt(
        self, spy_client, SessionFactory
    ):
        """With a new photo, per-segment photo overrides are dropped (so every
        segment uses the new photo) but per-segment prompt tweaks are kept."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        sd = _make_segment_def(s, project.id, 0)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face.jpg"],
        )
        _make_run_segment(
            s, run.id, sd.id, status=SegmentStatus.completed,
            prompt_override="tuned segment prompt",
            reference_image_urls_override=["https://old.example/seg-specific.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={"reference_urls": "https://new.example/newface.jpg"},
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        segs = list(nr.run_segments)
        run_refs = list(nr.reference_image_urls)
        s.close()
        assert run_refs == ["https://new.example/newface.jpg"]
        assert len(segs) == 1
        # prompt tweak preserved, photo override dropped (segment now inherits the
        # new run-level photo).
        assert segs[0].prompt_override == "tuned segment prompt"
        assert not segs[0].reference_image_urls_override

    def test_copy_no_resolution_no_ref_clones_same_resolution(
        self, spy_client, SessionFactory
    ):
        """Backward-compat: copy with neither resolution nor photo clones as-is."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="720p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        s = SessionFactory()
        nr = s.get(Run, resp.json()["run_id"])
        res = nr.resolution
        refs = list(nr.reference_image_urls)
        s.close()
        assert res == "720p"
        assert refs == ["https://old.example/face.jpg"]

    def test_copy_too_many_new_refs_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        many = ",".join(f"https://e/{i}.jpg" for i in range(settings.MAX_REFERENCE_IMAGES + 1))
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy", data={"reference_urls": many}
        )
        assert resp.status_code == 400
        assert "reference images" in resp.json()["detail"].lower()

    def test_copy_invalid_resolution_for_model_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, model="gemini-omni", resolution="720p",
            status=RunStatus.done,
        )
        resp = client.post(f"/api/v2/runs/{run.id}/copy", data={"resolution": "480p"})
        assert resp.status_code == 400
        assert "resolution" in resp.json()["detail"].lower()

    def test_copy_blocked_when_project_not_ready(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        resp = client.post(f"/api/v2/runs/{run.id}/copy", data={"resolution": "720p"})
        assert resp.status_code == 409

    def test_copy_missing_run_is_404(self, client):
        resp = client.post("/api/v2/runs/no-such-run/copy", data={"resolution": "720p"})
        assert resp.status_code == 404


class TestCopyRunExcludesGeneratedImage:
    """A generated still is a per-segment rescue cut from the SOURCE run's
    frozen snapshot — cloning it onto a copy (especially one with a new
    photo) would pin the OLD face into the reference priority chain ahead of
    the run-level set, the same class of bug Task 0's copy-preprocess-refs
    fix closed at the run level. prompt_override keeps cloning as it does
    today; only the generated-image state (and its job history) is dropped.
    """

    def _run_with_generated_image(self, session, project_id):
        run = _make_run(session, project_id, resolution="480p", status=RunStatus.done)
        sd = _make_segment_def(session, project_id, 0)
        rs = _make_run_segment(
            session, run.id, sd.id, status=SegmentStatus.completed,
            prompt_override="tuned prompt",
            generated_reference_urls=["/data/fake/char.jpg", "/data/fake/gen.png"],
        )
        job = RunSegmentImageJob(
            run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
            status=SegmentImageStatus.done, output_path="/data/fake/gen.png",
        )
        session.add(job)
        session.commit()
        rs.generated_image_job_id = job.id
        session.commit()
        return run, rs

    def test_copy_drops_generated_image_but_keeps_prompt_override(
        self, spy_client, SessionFactory
    ):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run, _rs = self._run_with_generated_image(s, project.id)
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "720p"})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        overrides = [rs for rs in nr.run_segments if rs.prompt_override]
        s.close()
        assert len(overrides) == 1
        assert overrides[0].prompt_override == "tuned prompt"
        assert overrides[0].generated_reference_urls is None
        assert overrides[0].generated_image_job_id is None

    def test_copy_clones_no_image_job_rows(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run, _rs = self._run_with_generated_image(s, project.id)
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "720p"})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        jobs = (
            s.query(RunSegmentImageJob)
            .filter(RunSegmentImageJob.run_id == new_id)
            .all()
        )
        s.close()
        assert jobs == []


# ---------------------------------------------------------------------------
# Copy run · re-run GPT Image 2 on the new photo
#
# A preprocessing run's refs are [character photo, frame, frame, ...] with
# preprocess_image_prompts positional against them. A preprocess copy keeps that
# geometry: the new photo lands at index 0, the source's frames and prompts come
# along verbatim, and the copy generates its OWN preprocessed images.
# ---------------------------------------------------------------------------


def _preprocess_source_run(session, project_id, **kwargs) -> Run:
    """A done run that went through GPT Image 2: char photo + 2 frames."""
    defaults = dict(
        resolution="480p",
        status=RunStatus.done,
        reference_image_urls=[
            "https://old.example/char.jpg",
            "https://old.example/frame1.jpg",
            "https://old.example/frame2.jpg",
        ],
        preprocess_enabled=True,
        preprocess_mode="per_image",
        preprocess_image_prompts=["", "put her in this shot", "and this one"],
        preprocessed_reference_urls=[
            "https://old.example/char.jpg",
            "https://old.example/gen1.png",
            "https://old.example/gen2.png",
        ],
        preprocess_task_ids=[None, "task-1", "task-2"],
    )
    defaults.update(kwargs)
    return _make_run(session, project_id, **defaults)


class TestCopyRunPreprocess:
    def test_copy_preprocess_reuses_frames_and_prompts_with_new_photo(
        self, spy_client, SessionFactory
    ):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(s, project.id)
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={
                "reference_urls": "https://new.example/face.jpg",
                "preprocess": "true",
            },
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        # Enters the GPT Image 2 chain, NOT submit — sending the unprocessed
        # refs to Seedance would spend video credits on the wrong images.
        assert resp.json()["status"] == "preprocessing"
        assert new_id in spy["preprocess_run"]
        assert new_id not in spy["process_run"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        refs = list(nr.reference_image_urls)
        prompts = list(nr.preprocess_image_prompts)
        pp = nr.preprocessed_reference_urls
        task_ids = nr.preprocess_task_ids
        enabled, auto, mode = (
            nr.preprocess_enabled, nr.preprocess_auto_approve, nr.preprocess_mode,
        )
        s.close()

        # New photo at index 0, the source's frames after it.
        assert refs == [
            "https://new.example/face.jpg",
            "https://old.example/frame1.jpg",
            "https://old.example/frame2.jpg",
        ]
        # Prompts are positional against refs — they transfer unchanged.
        assert prompts == ["", "put her in this shot", "and this one"]
        assert enabled is True
        assert auto is True
        assert mode == "per_image"
        # The source's generated images are the OLD face — cloning them would
        # make the new photo inert, which is the bug this option fixes.
        assert not pp
        assert not task_ids

    def test_copy_preprocess_with_uploaded_photo(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(s, project.id)
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={"preprocess": "true"},
            files={"reference_files": ("face.jpg", io.BytesIO(b"img"), "image/jpeg")},
        )
        assert resp.status_code == 201, resp.text

        s = SessionFactory()
        refs = list(s.get(Run, resp.json()["run_id"]).reference_image_urls)
        s.close()
        assert len(refs) == 3
        assert os.path.exists(refs[0])          # uploaded photo saved to disk
        assert refs[1:] == [
            "https://old.example/frame1.jpg", "https://old.example/frame2.jpg",
        ]

    def test_copy_preprocess_copies_local_frames_into_the_new_run(
        self, spy_client, SessionFactory, tmp_path
    ):
        """Local frames are COPIED — deleting the source run must not strand
        the copy mid-preprocessing."""
        client, spy = spy_client
        frame = tmp_path / "frame1.jpg"
        frame.write_bytes(b"frame-bytes")
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(
            s, project.id,
            reference_image_urls=["https://old.example/char.jpg", str(frame)],
            preprocess_image_prompts=["", "put her in this shot"],
            preprocessed_reference_urls=None,
            preprocess_task_ids=None,
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={"reference_urls": "https://new.example/face.jpg", "preprocess": "1"},
        )
        assert resp.status_code == 201, resp.text

        s = SessionFactory()
        refs = list(s.get(Run, resp.json()["run_id"]).reference_image_urls)
        s.close()
        assert refs[1] != str(frame)
        assert os.path.exists(refs[1])

    def test_copy_with_new_photo_drops_generated_images(
        self, spy_client, SessionFactory
    ):
        """A copy WITH a new photo carries none of the source's generated GPT
        Image 2 images — they are the OLD face, and cloning them would make
        the new photo inert (the preprocess-copy bug this mirrors). The new
        photo goes to Seedance raw; preprocess_enabled stays False since
        nothing was cloned for submit_run to read. Same for the prompts —
        they are half of the recipe and must not travel alone."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(s, project.id)
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={"reference_urls": "https://new.example/face.jpg"},
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        assert resp.json()["status"] == "queued"
        assert new_id in spy["process_run"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        refs, pp = list(nr.reference_image_urls), nr.preprocessed_reference_urls
        prompts = nr.preprocess_image_prompts
        enabled, auto = nr.preprocess_enabled, nr.preprocess_auto_approve
        s.close()
        assert refs == ["https://new.example/face.jpg"]
        assert pp is None
        assert prompts is None
        assert enabled is False
        assert auto is False

    def test_copy_without_new_photo_clones_generated_images_and_enables_preprocess(
        self, spy_client, SessionFactory
    ):
        """A plain copy (no new photo) of a GPT Image 2 run clones the source's
        generated images AND flips preprocess_enabled on, so submit_run's
        existing `preprocess_enabled and preprocessed_reference_urls` gate
        actually reads them instead of silently falling back to the raw
        originals (the bug this test guards against). It also clones the
        image prompts — otherwise the copy is `preprocess_enabled=True` with
        no prompts, which lets a FURTHER `preprocess=true` copy of it past
        the "was created with preprocessing" guard while having nothing for
        GPT Image 2 to actually do (the addendum's regression)."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(s, project.id)
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        assert resp.json()["status"] == "queued"
        assert new_id in spy["process_run"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        refs, pp = list(nr.reference_image_urls), list(nr.preprocessed_reference_urls)
        prompts = list(nr.preprocess_image_prompts)
        enabled, auto = nr.preprocess_enabled, nr.preprocess_auto_approve
        s.close()
        # Run-level references are also cloned unchanged (no new photo).
        assert refs == [
            "https://old.example/char.jpg",
            "https://old.example/frame1.jpg",
            "https://old.example/frame2.jpg",
        ]
        assert pp == [
            "https://old.example/char.jpg",
            "https://old.example/gen1.png",
            "https://old.example/gen2.png",
        ]
        # Prompts cloned verbatim, positional against refs, index 0 still "".
        assert prompts == ["", "put her in this shot", "and this one"]
        assert len(prompts) == len(refs)
        assert enabled is True
        assert auto is False

    def test_copy_of_a_copy_without_new_photo_stays_preprocess_enabled(
        self, spy_client, SessionFactory
    ):
        """A copy of a copy (still no new photo) must keep carrying the
        generated images with preprocess_enabled True — otherwise the bug
        reappears one generation later."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(s, project.id)
        rid = run.id
        s.close()

        first = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert first.status_code == 201, first.text
        first_id = first.json()["run_id"]

        second = client.post(f"/api/v2/runs/{first_id}/copy", data={})
        assert second.status_code == 201, second.text
        second_id = second.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, second_id)
        pp = list(nr.preprocessed_reference_urls)
        enabled = nr.preprocess_enabled
        s.close()
        assert pp == [
            "https://old.example/char.jpg",
            "https://old.example/gen1.png",
            "https://old.example/gen2.png",
        ]
        assert enabled is True

    def test_copy_of_a_pre_fix_contaminated_row_does_not_clone_mismatched_images(
        self, spy_client, SessionFactory
    ):
        """Regression: before this fix, a copy made WITH a new photo (and no
        `preprocess` flag) cloned the source's generated images unconditionally
        — there was no `has_new_refs` guard at all — while leaving
        `preprocess_enabled=False`. That can leave a legacy row on disk today
        with `reference_image_urls` = the new face but `preprocessed_reference_urls`
        = the OLD face's generated images, inert only because the flag is False.

        Copying THAT row again (no new photo) must NOT clone the mismatched
        images or flip the flag — doing so would revive a dormant wrong-face
        bug instead of fixing one. Gating on `src.preprocess_enabled` (only
        ever true when a run's own pp images are known to match its own face)
        is what stops it."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        contaminated = _make_run(
            s, project.id, status=RunStatus.done,
            reference_image_urls=["https://new.example/faceB.jpg"],
            preprocess_enabled=False,
            preprocessed_reference_urls=[
                "https://old.example/char.jpg",
                "https://old.example/gen1.png",
                "https://old.example/gen2.png",
            ],
        )
        rid = contaminated.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        refs, pp = list(nr.reference_image_urls), nr.preprocessed_reference_urls
        enabled = nr.preprocess_enabled
        s.close()
        assert refs == ["https://new.example/faceB.jpg"]
        assert pp is None
        assert enabled is False

    def test_copy_preprocess_from_a_source_with_no_prompts_is_400(
        self, client, db_session
    ):
        """Belt-and-suspenders for the Part 1 clone fix: a source that is
        `preprocess_enabled=True` but carries no `preprocess_image_prompts`
        (the exact shape a plain copy had before this addendum) must be
        refused, not silently accepted into a no-op preprocessing run."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done,
            reference_image_urls=[
                "https://old.example/char.jpg",
                "https://old.example/frame1.jpg",
            ],
            preprocess_enabled=True,
            preprocess_mode="per_image",
            preprocess_image_prompts=None,
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy",
            data={"reference_urls": "https://new.example/face.jpg", "preprocess": "true"},
        )
        assert resp.status_code == 400
        assert "image prompt" in resp.json()["detail"].lower()

    def test_copy_preprocess_without_a_photo_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _preprocess_source_run(db_session, project.id)
        resp = client.post(f"/api/v2/runs/{run.id}/copy", data={"preprocess": "true"})
        assert resp.status_code == 400
        assert "exactly one" in resp.json()["detail"].lower()

    def test_copy_preprocess_with_two_photos_is_400(self, client, db_session):
        """A second photo has nowhere to go — the frames come from the source."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _preprocess_source_run(db_session, project.id)
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy",
            data={
                "reference_urls": "https://new.example/a.jpg,https://new.example/b.jpg",
                "preprocess": "true",
            },
        )
        assert resp.status_code == 400
        assert "exactly one" in resp.json()["detail"].lower()

    def test_copy_preprocess_from_a_non_preprocess_run_is_400(
        self, client, db_session
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face.jpg"],
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy",
            data={"reference_urls": "https://new.example/a.jpg", "preprocess": "true"},
        )
        assert resp.status_code == 400
        assert "not created with preprocessing" in resp.json()["detail"].lower()

    def test_copy_preprocess_without_frames_is_400(self, client, db_session):
        """A preprocessing run whose ref list holds only the character photo has
        no recipe to re-run."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _preprocess_source_run(
            db_session, project.id,
            reference_image_urls=["https://old.example/char.jpg"],
            preprocess_image_prompts=[""],
            preprocessed_reference_urls=None,
            preprocess_task_ids=None,
        )
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy",
            data={"reference_urls": "https://new.example/a.jpg", "preprocess": "true"},
        )
        assert resp.status_code == 400
        assert "no frames" in resp.json()["detail"].lower()

    def test_batch_mixes_preprocessed_and_raw_copies(self, spy_client, SessionFactory):
        """Rows are independent: one copy re-runs GPT Image 2, the other doesn't,
        and each is enqueued at its own entry point."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(s, project.id)
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "480p",
                "runs[0][reference_urls]": "https://new.example/a.jpg",
                "runs[0][preprocess]": "1",
                "runs[1][resolution]": "720p",
                "runs[1][reference_urls]": "https://new.example/b.jpg",
            },
        )
        assert resp.status_code == 201, resp.text
        runs = resp.json()["runs"]
        assert [r["status"] for r in runs] == ["preprocessing", "queued"]
        assert runs[0]["run_id"] in spy["preprocess_run"]
        assert runs[1]["run_id"] in spy["process_run"]

        s = SessionFactory()
        r0, r1 = s.get(Run, runs[0]["run_id"]), s.get(Run, runs[1]["run_id"])
        refs0, refs1 = list(r0.reference_image_urls), list(r1.reference_image_urls)
        auto0, auto1 = r0.preprocess_auto_approve, r1.preprocess_auto_approve
        s.close()
        assert len(refs0) == 3 and refs0[0] == "https://new.example/a.jpg"
        assert refs1 == ["https://new.example/b.jpg"]
        assert auto0 is True and auto1 is False

    def test_preprocessing_run_is_protected_like_any_active_run(
        self, client, db_session
    ):
        """The GPT Image 2 poll chain downloads into the run's own dir, so a run
        in `preprocessing` must not be deletable and must block re-analysis —
        recovery.py has always counted it active; api_v2 now agrees."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _preprocess_source_run(
            db_session, project.id, status=RunStatus.preprocessing,
        )

        assert client.delete(f"/api/v2/runs/{run.id}").status_code == 409
        resp = client.post(f"/api/v2/projects/{project.id}/analyze")
        assert resp.status_code == 409
        assert "active" in resp.json()["detail"].lower()

    def test_batch_bad_preprocess_spec_rolls_back_whole_batch(
        self, spy_client, SessionFactory
    ):
        """A preprocess row with no photo fails every copy — nothing is created
        and nothing is enqueued."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _preprocess_source_run(s, project.id)
        pid, rid = project.id, run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "480p",
                "runs[0][reference_urls]": "https://new.example/a.jpg",
                "runs[0][preprocess]": "1",
                "runs[1][resolution]": "720p",
                "runs[1][preprocess]": "1",       # no photo → invalid
            },
        )
        assert resp.status_code == 400
        assert spy["preprocess_run"] == [] and spy["process_run"] == []

        s = SessionFactory()
        remaining = s.query(Run).filter(Run.project_id == pid).all()
        s.close()
        assert [r.id for r in remaining] == [rid]


# ---------------------------------------------------------------------------
# Copy run vs the localisation modes (docs/localisation.md §7)
# ---------------------------------------------------------------------------


def _localisation_run(session, *, swap: bool, **kwargs) -> Run:
    """A done run on a ready localisation project, with the prompt the New Run
    form's radio would have produced for *swap* (speech + character) or not
    (speech only)."""
    project = _make_project(
        session, status=ProjectStatus.ready, project_type="localisation"
    )
    template = (
        project_types._LOCALISATION_SWAP_PROMPT
        if swap
        else project_types._LOCALISATION_KEEP_PROMPT
    )
    return _make_run(
        session,
        project.id,
        prompt=template.format(
            source_language="english", target_language="japanese",
            dialogue="**Woman:**     はい",
        ),
        model="seedance-2-5",
        resolution="720p",
        audio_mode="seedance",
        language="ja",
        status=RunStatus.done,
        **kwargs,
    )


class TestCopyLocalisationRun:
    """A speech-only run's prompt says to keep the person on screen, so a copy
    can only repeat itself — the photo it attaches is inert. Copying stays open
    on speech + character runs, where a new photo is the point."""

    def test_copy_speech_only_run_is_409(self, client, db_session):
        run = _localisation_run(db_session, swap=False)
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy", data={"reference_urls": "https://e/1.jpg"}
        )
        assert resp.status_code == 409
        assert "speech-only" in resp.json()["detail"]

    def test_copy_batch_speech_only_run_is_409(self, client, db_session):
        run = _localisation_run(db_session, swap=False)
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy-batch",
            data={"runs[0][resolution]": "720p", "runs[0][name]": "Second face"},
        )
        assert resp.status_code == 409

    def test_copy_speech_plus_character_run_is_allowed(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        run = _localisation_run(
            s, swap=True, reference_image_urls=["https://e/face.jpg"]
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy", data={"reference_urls": "https://e/other.jpg"}
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        assert new_id in spy["process_run"]
        s = SessionFactory()
        nr = s.get(Run, new_id)
        s.close()
        assert nr.reference_image_urls == ["https://e/other.jpg"]

    def test_copy_swap_run_without_a_photo_is_allowed(self, spy_client, SessionFactory):
        """The mode is read off the prompt, not off the reference list: a
        speech + character run submitted without a photo is exactly the case
        where copying WITH one is the fix."""
        client, spy = spy_client
        s = SessionFactory()
        run = _localisation_run(s, swap=True)
        rid = run.id
        s.close()
        resp = client.post(
            f"/api/v2/runs/{rid}/copy", data={"reference_urls": "https://e/face.jpg"}
        )
        assert resp.status_code == 201, resp.text

    def test_other_project_types_are_not_gated(self, spy_client, SessionFactory):
        """Only localisation projects have modes — a face-swap run whose prompt
        happens to mention keeping the character still copies."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, prompt="Do not replace the character, drop the subtitles",
            resolution="720p", status=RunStatus.done,
        )
        rid = run.id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "720p"})
        assert resp.status_code == 201, resp.text

    def test_cloning_itself_is_not_gated(self, db_session):
        """The refusal lives at the entry points (the two HTTP routes, and
        automation's launch phase), NOT in _build_copied_run — the cloning
        helper stays a mechanism with no policy of its own."""
        from app.api_v2 import _build_copied_run

        run = _localisation_run(db_session, swap=False)
        new_run = _build_copied_run(
            run, db_session, resolution=None, name="from sheet",
            ref_files=[], ref_urls=["https://e/face.jpg"],
        )
        db_session.commit()
        assert new_run.id != run.id

    def test_enabling_automation_on_a_speech_only_run_is_409(
        self, client, db_session
    ):
        """Automation is Copy run on a daily quota — same premise, same
        refusal, or the rule would be one click away from being bypassed."""
        run = _localisation_run(db_session, swap=False)
        resp = client.put(
            f"/api/v2/runs/{run.id}/automation",
            json={"enabled": True, "plan": "close-up", "daily_count": 3},
        )
        assert resp.status_code == 409
        assert "speech-only" in resp.json()["detail"]

    def test_disabling_automation_on_a_speech_only_run_still_works(
        self, client, db_session
    ):
        """Only enabling is blocked: an automation switched on before this rule
        existed must stay switchable-off through the same endpoint."""
        run = _localisation_run(db_session, swap=False)
        resp = client.put(
            f"/api/v2/runs/{run.id}/automation",
            json={"enabled": False, "plan": "close-up", "daily_count": 3},
        )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_enabling_automation_on_a_swap_run_is_allowed(self, client, db_session):
        run = _localisation_run(db_session, swap=True)
        resp = client.put(
            f"/api/v2/runs/{run.id}/automation",
            json={"enabled": True, "plan": "close-up", "daily_count": 3},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["enabled"] is True


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/copy-batch — launch up to 10 copies in one request
# ---------------------------------------------------------------------------


class TestCopyRunBatch:
    def test_batch_creates_multiple_runs_and_enqueues(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, name="Base", prompt="hi", model="seedance",
            resolution="480p", status=RunStatus.done,
        )
        pid, rid = project.id, run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "480p", "runs[0][name]": "First",
                "runs[1][resolution]": "720p", "runs[1][name]": "Second",
                "runs[2][resolution]": "1080p",
            },
        )
        assert resp.status_code == 201, resp.text
        runs = resp.json()["runs"]
        assert len(runs) == 3
        assert all(r["status"] == "queued" for r in runs)
        ids = [r["run_id"] for r in runs]
        assert all(i in spy["process_run"] for i in ids)
        assert len(set(ids)) == 3 and rid not in ids

        s = SessionFactory()
        rows = [s.get(Run, i) for i in ids]
        names = [r.name for r in rows]
        resolutions = [r.resolution for r in rows]
        s.close()
        # Order is preserved (runs[0], runs[1], runs[2]).
        assert names[0] == "First" and names[1] == "Second"
        assert resolutions == ["480p", "720p", "1080p"]
        # Unnamed copy gets the default name suffix.
        assert names[2].startswith("Base ·")

    def test_batch_per_run_reference_url(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "480p",
                "runs[0][reference_urls]": "https://new.example/a.jpg",
                "runs[1][resolution]": "720p",  # no new photo → clones source refs
            },
        )
        assert resp.status_code == 201, resp.text
        ids = [r["run_id"] for r in resp.json()["runs"]]

        s = SessionFactory()
        refs0 = list(s.get(Run, ids[0]).reference_image_urls)
        refs1 = list(s.get(Run, ids[1]).reference_image_urls)
        s.close()
        assert refs0 == ["https://new.example/a.jpg"]
        assert refs1 == ["https://old.example/face.jpg"]

    def test_batch_with_uploaded_file(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, resolution="480p", status=RunStatus.done)
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={"runs[0][resolution]": "720p"},
            files=[("runs[0][reference_files]", ("face.jpg", io.BytesIO(b"img"), "image/jpeg"))],
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["runs"][0]["run_id"]

        s = SessionFactory()
        refs = list(s.get(Run, new_id).reference_image_urls)
        s.close()
        assert len(refs) == 1
        assert os.path.exists(refs[0])

    def test_batch_empty_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        resp = client.post(f"/api/v2/runs/{run.id}/copy-batch", data={})
        assert resp.status_code == 400
        assert "no runs" in resp.json()["detail"].lower()

    def test_batch_too_many_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        data = {f"runs[{i}][resolution]": "480p" for i in range(11)}
        resp = client.post(f"/api/v2/runs/{run.id}/copy-batch", data=data)
        assert resp.status_code == 400
        assert "too many" in resp.json()["detail"].lower()

    def test_batch_invalid_spec_rolls_back_whole_batch(self, spy_client, SessionFactory):
        """One bad resolution fails the entire batch — no partial runs created."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, model="seedance", resolution="480p", status=RunStatus.done,
        )
        pid, rid = project.id, run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "720p",   # valid
                "runs[1][resolution]": "4k",     # invalid for seedance
            },
        )
        assert resp.status_code == 400
        assert spy["process_run"] == []  # nothing enqueued

        s = SessionFactory()
        # Only the original run exists; neither copy was committed.
        count = len([r for r in s.query(Run).filter(Run.project_id == pid).all()])
        s.close()
        assert count == 1

    def test_batch_blocked_when_project_not_ready(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy-batch", data={"runs[0][resolution]": "720p"}
        )
        assert resp.status_code == 409

    def test_batch_missing_run_is_404(self, client):
        resp = client.post(
            "/api/v2/runs/no-such-run/copy-batch", data={"runs[0][resolution]": "720p"}
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Per-segment prompts supplied at run creation
# ---------------------------------------------------------------------------


class TestSegmentPromptsOnCreate:
    def test_segment_prompt_used_as_primary_override(self, client, db_session, SessionFactory):
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd0 = _make_segment_def(db_session, project.id, 0)
        sd1 = _make_segment_def(db_session, project.id, 1)

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "base prompt",
                "segment_prompts": _json.dumps({sd0.id: "make the jacket red"}),
            },
        )
        assert resp.status_code == 201, resp.text

        s = SessionFactory()
        run = s.get(Run, resp.json()["run_id"])
        rss = {rs.segment_def_id: rs for rs in run.run_segments}
        s.close()
        # Only the segment with a per-segment prompt gets a pre-created override
        # RunSegment. The per-segment prompt is the PRIMARY prompt for that
        # segment (it fully replaces the run prompt, not appended to it).
        assert sd0.id in rss
        assert rss[sd0.id].prompt_override == "make the jacket red"
        assert sd1.id not in rss

    def test_blank_and_unknown_segment_prompts_ignored(self, client, db_session, SessionFactory):
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd0 = _make_segment_def(db_session, project.id, 0)

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "base",
                "segment_prompts": _json.dumps({sd0.id: "   ", "no-such-id": "x"}),
            },
        )
        assert resp.status_code == 201
        s = SessionFactory()
        run = s.get(Run, resp.json()["run_id"])
        n = len(run.run_segments)
        s.close()
        assert n == 0  # blank text + unknown id both ignored

    def test_invalid_segment_prompts_json_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "base", "segment_prompts": "{not valid json"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Public token-signed source link (/public/projects/{pid}/source)
# ---------------------------------------------------------------------------


class TestPublicSourceLink:
    def test_valid_token_streams_source(self, client, db_session, tmp_path):
        from app.public import make_source_token

        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00\x01videodata")
        project = _make_project(
            db_session, status=ProjectStatus.ready, source_local_path=str(src)
        )
        tok = make_source_token(project.id)

        r = client.get(f"/public/projects/{project.id}/source", params={"token": tok})
        assert r.status_code == 200
        assert r.content == b"\x00\x01videodata"
        assert r.headers["content-type"].startswith("video/mp4")

    def test_bad_token_is_403(self, client, db_session, tmp_path):
        src = tmp_path / "s.mp4"
        src.write_bytes(b"x")
        project = _make_project(
            db_session, status=ProjectStatus.ready, source_local_path=str(src)
        )
        r = client.get(
            f"/public/projects/{project.id}/source", params={"token": "deadbeef"}
        )
        assert r.status_code == 403

    def test_missing_token_is_403(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        r = client.get(f"/public/projects/{project.id}/source")
        assert r.status_code == 403

    def test_valid_token_missing_file_is_404(self, client, db_session):
        from app.public import make_source_token

        project = _make_project(
            db_session, status=ProjectStatus.ready, source_local_path="/nope/x.mp4"
        )
        tok = make_source_token(project.id)
        r = client.get(f"/public/projects/{project.id}/source", params={"token": tok})
        assert r.status_code == 404


class TestPublicResultLink:
    def test_valid_token_streams_result(self, client, db_session, tmp_path):
        from app.public import make_result_token

        out = tmp_path / "final.mp4"
        out.write_bytes(b"resultbytes")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_local_path=str(out)
        )
        tok = make_result_token(run.id)

        r = client.get(f"/public/runs/{run.id}/result", params={"token": tok})
        assert r.status_code == 200
        assert r.content == b"resultbytes"
        assert r.headers["content-type"].startswith("video/mp4")
        # Cache-Control changed from no-store to a private cacheable policy.
        assert "private" in r.headers["cache-control"]
        assert "no-store" not in r.headers["cache-control"]

    def test_bad_token_is_403(self, client, db_session, tmp_path):
        out = tmp_path / "final.mp4"
        out.write_bytes(b"x")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_local_path=str(out)
        )
        r = client.get(f"/public/runs/{run.id}/result", params={"token": "nope"})
        assert r.status_code == 403

    def test_missing_result_file_is_404(self, client, db_session):
        from app.public import make_result_token

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.failed)
        tok = make_result_token(run.id)
        r = client.get(f"/public/runs/{run.id}/result", params={"token": tok})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# X-Accel-Redirect path (NGINX_ACCEL_ENABLED=True)
# ---------------------------------------------------------------------------


class TestNginxAccelRedirect:
    """Verify that when NGINX_ACCEL_ENABLED is on, media endpoints return an
    X-Accel-Redirect header and no body instead of streaming through Python."""

    def _enable_accel(self, monkeypatch, data_dir: str):
        """Patch settings so NGINX_ACCEL_ENABLED=True and DATA_DIR is set."""
        import app.accel as accel_module
        import app.config as config_module

        monkeypatch.setattr(config_module.settings, "NGINX_ACCEL_ENABLED", True)
        monkeypatch.setattr(config_module.settings, "DATA_DIR", data_dir)
        # Also patch in accel.py since it imported settings at module load time.
        monkeypatch.setattr(accel_module.settings, "NGINX_ACCEL_ENABLED", True)
        monkeypatch.setattr(accel_module.settings, "DATA_DIR", data_dir)

    def test_project_source_returns_accel_header(
        self, client, db_session, tmp_path, monkeypatch
    ):
        """GET /api/v2/projects/{pid}/source with accel on → X-Accel-Redirect."""
        data_dir = str(tmp_path)
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00FAKE")
        self._enable_accel(monkeypatch, data_dir)

        project = _make_project(
            db_session,
            source_local_path=str(video),
            status=ProjectStatus.ready,
        )
        r = client.get(f"/api/v2/projects/{project.id}/source")
        assert r.status_code == 200
        assert "x-accel-redirect" in r.headers
        assert r.headers["x-accel-redirect"].startswith("/internal/data/")
        # No body should be streamed through Python in accel mode.
        assert r.content == b""

    def test_run_result_returns_accel_header(
        self, client, db_session, tmp_path, monkeypatch
    ):
        """GET /api/v2/runs/{rid}/result with accel on → X-Accel-Redirect."""
        data_dir = str(tmp_path)
        video = tmp_path / "result.mp4"
        video.write_bytes(b"\x00FAKE")
        self._enable_accel(monkeypatch, data_dir)

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(video),
        )
        r = client.get(f"/api/v2/runs/{run.id}/result")
        assert r.status_code == 200
        assert "x-accel-redirect" in r.headers
        accel_uri = r.headers["x-accel-redirect"]
        assert accel_uri.startswith("/internal/data/")
        # Cache policy must be private/cacheable (not no-store).
        assert "private" in r.headers.get("cache-control", "")
        assert r.content == b""

    def test_public_source_returns_accel_header(
        self, client, db_session, tmp_path, monkeypatch
    ):
        """GET /public/projects/{pid}/source with accel on → X-Accel-Redirect."""
        from app.public import make_source_token

        data_dir = str(tmp_path)
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00FAKE")
        self._enable_accel(monkeypatch, data_dir)

        project = _make_project(
            db_session,
            source_local_path=str(video),
            status=ProjectStatus.ready,
        )
        tok = make_source_token(project.id)
        r = client.get(f"/public/projects/{project.id}/source", params={"token": tok})
        assert r.status_code == 200
        assert "x-accel-redirect" in r.headers
        assert r.headers["x-accel-redirect"].startswith("/internal/data/")
        assert r.content == b""

    def test_public_result_returns_accel_header(
        self, client, db_session, tmp_path, monkeypatch
    ):
        """GET /public/runs/{rid}/result with accel on → X-Accel-Redirect."""
        from app.public import make_result_token

        data_dir = str(tmp_path)
        video = tmp_path / "result.mp4"
        video.write_bytes(b"\x00FAKE")
        self._enable_accel(monkeypatch, data_dir)

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(video),
        )
        tok = make_result_token(run.id)
        r = client.get(f"/public/runs/{run.id}/result", params={"token": tok})
        assert r.status_code == 200
        assert "x-accel-redirect" in r.headers
        assert r.headers["x-accel-redirect"].startswith("/internal/data/")
        assert r.content == b""

    def test_accel_off_still_streams_file(self, client, db_session, tmp_path):
        """With NGINX_ACCEL_ENABLED off (default), FileResponse still streams."""
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00STREAMING")
        project = _make_project(
            db_session,
            source_local_path=str(video),
            status=ProjectStatus.ready,
        )
        r = client.get(f"/api/v2/projects/{project.id}/source")
        assert r.status_code == 200
        assert "x-accel-redirect" not in r.headers
        assert r.content == b"\x00STREAMING"


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/export-to-sheet  (Final Creatives export)
# ---------------------------------------------------------------------------


class _FakeSheets:
    """Stand-in for SheetsClient that records appended rows in-memory."""

    HEADER = [
        "Date",
        "Job ID",
        "Input Link",
        "Creative Link",
        "Pipeline Type",
        "Run on Meta",
    ]

    def __init__(self, existing_job_ids=None):
        self.existing_job_ids = list(existing_job_ids or [])
        self.appended = []  # list of rows

    def get_header(self, spreadsheet_id, tab):
        return list(self.HEADER)

    def column_values(self, spreadsheet_id, tab, column_index, *, last_n=None):
        # column_index 1 == "Job ID" header + values
        return ["Job ID", *self.existing_job_ids]

    def append_row(self, spreadsheet_id, tab, row):
        self.appended.append(row)
        return {"updates": {"updatedRows": 1}}


class TestExportToSheet:
    def test_export_enqueues_and_marks_exported(self, client, db_session, monkeypatch):
        """Endpoint enqueues the Sheets job and optimistically sets exported_at."""
        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_export_run_to_sheet", enqueued.append)

        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            source_type="gdrive",
            source_ref="https://drive.google.com/file/d/INPUT123/view",
        )
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
        )

        r = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "added"

        # The run is marked exported immediately so the fast-path / UI work on reload.
        db_session.expire_all()
        assert db_session.get(Run, run.id).final_creatives_exported_at is not None

        # Sheets I/O is enqueued (not done synchronously in the endpoint).
        assert enqueued == [run.id]

    def test_export_rolls_back_the_stamp_when_the_enqueue_fails(
        self, client, db_session, monkeypatch
    ):
        """Redis down: 503 with a usable message, and the run is NOT left marked
        exported. Nothing was queued, so no on_failure callback exists to repair
        it — the endpoint has to undo its own optimistic write."""
        def _boom(_rid):
            raise ConnectionError("Error 111 connecting to redis:6379")

        monkeypatch.setattr(api_v2_module, "enqueue_export_run_to_sheet", _boom)

        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            source_type="gdrive",
            source_ref="https://drive.google.com/file/d/INPUT123/view",
        )
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
        )

        r = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")

        assert r.status_code == 503, r.text
        assert "try again" in r.json()["detail"].lower()

        # The lie is undone: the Send button comes back and the sheet state is honest.
        db_session.expire_all()
        assert db_session.get(Run, run.id).final_creatives_exported_at is None


    def test_export_response_includes_fields(self, client, db_session, monkeypatch):
        """Response still includes the fields dict so callers can inspect what was sent."""
        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_export_run_to_sheet", enqueued.append)

        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            source_type="gdrive",
            source_ref="https://drive.google.com/file/d/INPUT123/view",
        )
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
        )
        r = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        body = r.json()
        fields = body.get("fields", {})
        assert fields["Job ID"] == run.id[:8]  # no language set → bare prefix
        assert fields["Creative Link"] == "https://drive.google.com/file/d/OUT456/view"
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", fields["Date"])

    def test_export_job_id_carries_the_run_language(
        self, client, db_session, monkeypatch
    ):
        """A Japanese run reports Job ID '<prefix>-ja' — what the renamer parses."""
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_run_to_sheet", lambda _rid: None
        )
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
            language="ja",
        )

        r = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        assert r.json()["fields"]["Job ID"] == f"{run.id[:8]}-ja"

    def test_export_is_idempotent_via_exported_at(self, client, db_session, monkeypatch):
        """Second call returns 'exists' via the DB fast-path (exported_at already set)."""
        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_export_run_to_sheet", enqueued.append)

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
        )

        # First call — marks exported and enqueues.
        r1 = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r1.status_code == 200
        assert r1.json()["status"] == "added"
        assert len(enqueued) == 1

        # Second call — fast-path (exported_at already set); no extra enqueue.
        r2 = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r2.status_code == 200
        assert r2.json()["status"] == "exists"
        assert len(enqueued) == 1  # not enqueued again

        db_session.expire_all()
        assert db_session.get(Run, run.id).final_creatives_exported_at is not None

    def test_export_requires_done_status(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.failed,
            result_gdrive_file_id="OUT456",
        )
        r = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r.status_code == 409

    def test_export_requires_gdrive_result(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        r = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r.status_code == 409

    def test_export_missing_run_is_404(self, client):
        r = client.post("/api/v2/runs/no-such-run/export-to-sheet")
        assert r.status_code == 404

    def test_export_fast_path_skips_enqueue_when_already_marked(
        self, client, db_session, monkeypatch
    ):
        """An already-exported run returns 'exists' without enqueuing a new job."""
        from datetime import datetime, timezone

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_export_run_to_sheet", enqueued.append)

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        r = client.post(f"/api/v2/runs/{run.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "exists"
        assert enqueued == []  # fast-path: no enqueue


class TestExportCaptionDemoRemoved:
    """Regression: the removed "Add to Caption Demo" run export stays gone."""

    def test_export_caption_demo_is_404(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_gdrive_file_id="OUT456"
        )
        r = client.post(f"/api/v2/runs/{run.id}/export-caption-demo")
        assert r.status_code == 404

    def test_export_caption_demo_get_is_404(self, client):
        r = client.get("/api/v2/runs/no-such-run/export-caption-demo")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v2/tail-edits/{edit_id}/export-to-sheet  (Final Creatives export)
# ---------------------------------------------------------------------------


class TestExportTailEditToSheet:
    def test_export_enqueues_and_marks_exported(self, client, db_session, monkeypatch):
        """Endpoint enqueues the Sheets job and optimistically sets exported_at."""
        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_tail_edit_to_sheet", enqueued.append
        )

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
        )
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            gdrive_file_id="TAIL789",
        )

        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "added"

        # The tail edit is marked exported immediately so the fast-path / UI
        # work on reload.
        db_session.expire_all()
        assert db_session.get(RunTailEdit, edit.id).final_creatives_exported_at is not None

        # Sheets I/O is enqueued (not done synchronously in the endpoint).
        assert enqueued == [edit.id]

    def test_export_response_includes_fields(self, client, db_session, monkeypatch):
        """Response includes the fields dict: own Job ID/Creative Link, parent run's Input Link."""
        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_tail_edit_to_sheet", enqueued.append
        )

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
        )
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            gdrive_file_id="TAIL789",
        )

        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        body = r.json()
        fields = body.get("fields", {})
        assert fields["Job ID"] == edit.id[:8]  # parent run has no language set
        assert fields["Creative Link"] == "https://drive.google.com/file/d/TAIL789/view"
        assert fields["Input Link"] == "https://drive.google.com/file/d/OUT456/view"
        assert fields["Pipeline Type"] == "re-skin-tail-edit"
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", fields["Date"])

    def test_export_job_id_inherits_the_parent_run_language(
        self, client, db_session, monkeypatch
    ):
        """Own 8-char prefix (no collision with the run's row) + parent's language."""
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_tail_edit_to_sheet", lambda _eid: None
        )
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
            language="ja",
        )
        edit = _make_tail_edit(
            db_session, run.id, status=TailEditStatus.done, gdrive_file_id="TAIL789"
        )

        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        job_id = r.json()["fields"]["Job ID"]
        assert job_id == f"{edit.id[:8]}-ja"
        assert not job_id.startswith(run.id[:8])

    def test_export_input_link_falls_back_to_run_id(self, client, db_session, monkeypatch):
        """When the parent run has no Drive result yet, Input Link falls back to run_id."""
        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_tail_edit_to_sheet", enqueued.append
        )

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            gdrive_file_id="TAIL789",
        )

        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        assert r.json()["fields"]["Input Link"] == run.id

    def test_export_is_idempotent_via_exported_at(self, client, db_session, monkeypatch):
        """Second call returns 'exists' via the DB fast-path (exported_at already set)."""
        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_tail_edit_to_sheet", enqueued.append
        )

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            gdrive_file_id="TAIL789",
        )

        # First call — marks exported and enqueues.
        r1 = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r1.status_code == 200
        assert r1.json()["status"] == "added"
        assert len(enqueued) == 1

        # Second call — fast-path (exported_at already set); no extra enqueue.
        r2 = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r2.status_code == 200
        assert r2.json()["status"] == "exists"
        assert len(enqueued) == 1  # not enqueued again

        db_session.expire_all()
        assert db_session.get(RunTailEdit, edit.id).final_creatives_exported_at is not None

    def test_export_requires_done_status(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.processing,
            gdrive_file_id="TAIL789",
        )
        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r.status_code == 409

    def test_export_requires_gdrive_result(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.done)
        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r.status_code == 409

    def test_export_missing_edit_is_404(self, client):
        r = client.post("/api/v2/tail-edits/no-such-edit/export-to-sheet")
        assert r.status_code == 404

    def test_export_fast_path_skips_enqueue_when_already_marked(
        self, client, db_session, monkeypatch
    ):
        """An already-exported tail edit returns 'exists' without enqueuing a new job."""
        from datetime import datetime, timezone

        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_tail_edit_to_sheet", enqueued.append
        )

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            gdrive_file_id="TAIL789",
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-to-sheet")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "exists"
        assert enqueued == []  # fast-path: no enqueue


class TestExportTailEditToCaptionDemoRemoved:
    """Regression: the removed "Add to Caption Demo" tail-edit export stays gone."""

    def test_export_caption_demo_is_404(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            gdrive_file_id="TAIL789",
        )
        r = client.post(f"/api/v2/tail-edits/{edit.id}/export-caption-demo")
        assert r.status_code == 404

    def test_missing_edit_export_caption_demo_is_404(self, client):
        r = client.post("/api/v2/tail-edits/no-such-edit/export-caption-demo")
        assert r.status_code == 404


class _FakeExec:
    """Stand-in for a googleapiclient HttpRequest: only ``.execute()``."""

    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeSheetsService:
    """Minimal double for the googleapiclient Sheets ``service`` object
    (what ``SheetsClient(service=...)`` accepts directly for testing),
    keyed by exact request range so tests can assert precisely which
    ranges were hit.

    ``spreadsheets().get(...)`` (grid metadata — no cell values) and
    ``spreadsheets().values().get(...)`` (cell data) both route to this
    single ``get`` — the same object is returned for ``spreadsheets()`` and
    ``.values()``, so the two calls are told apart by which kwargs they pass
    (``ranges=``/``fields=`` for metadata vs. ``range=`` for cell data).
    """

    def __init__(self, row_count: int, range_responses: dict):
        self._row_count = row_count
        self._range_responses = range_responses  # {range_str: [values]}
        self.value_gets: list[str] = []
        self.meta_gets = 0

    def spreadsheets(self):
        return self

    def values(self):
        return self

    def get(self, spreadsheetId, ranges=None, fields=None, range=None, majorDimension=None):
        if ranges is not None:
            self.meta_gets += 1
            return _FakeExec(
                {"sheets": [{"properties": {"gridProperties": {"rowCount": self._row_count}}}]}
            )
        self.value_gets.append(range)
        vals = self._range_responses.get(range, [])
        payload = {"values": [vals]} if vals else {}
        return _FakeExec(payload)


class TestSheetsHelpers:
    def test_build_row_maps_by_header_name_reordered(self):
        from app.sheets_client import build_final_creatives_row

        # Header order differs from field order; matching is by name.
        header = ["Run on Meta", "Job ID", "Date", "Creative Link", "Input Link", "Pipeline Type"]
        row = build_final_creatives_row(
            header,
            fields={
                "Date": "2026-06-29",
                "Job ID": "abc",
                "Input Link": "in",
                "Creative Link": "out",
                "Pipeline Type": "re-skin-v2",
                "Run on Meta": "need_to_rename",
            },
        )
        assert row == ["need_to_rename", "abc", "2026-06-29", "out", "in", "re-skin-v2"]

    def test_build_row_blank_for_unmatched_and_extra_columns(self):
        from app.sheets_client import build_final_creatives_row

        header = ["Job ID", "Notes", "Date"]  # Notes has no field; others missing
        row = build_final_creatives_row(
            header, fields={"Date": "2026-06-29", "Job ID": "abc", "Creative Link": "x"}
        )
        assert row == ["abc", "", "2026-06-29"]

    def test_build_row_case_insensitive(self):
        from app.sheets_client import build_final_creatives_row

        row = build_final_creatives_row(["  JOB id ", "date"], fields={"Job ID": "abc", "Date": "d"})
        assert row == ["abc", "d"]

    # -- Idea column -----------------------------------------------------

    def test_build_idea_shape_and_model_tag(self):
        from app.sheets_client import build_idea

        # "Q-00873" is exactly the 7-char run slot — nothing to clamp.
        assert build_idea("Gardener", "Q-00873", "seedance-2-5") == "Gardener_Q-00873_SD25"
        assert build_idea("Gardener", "Q-00873", "gemini-omni") == "Gardener_Q-00873_GO"
        # An 8th char is dropped.
        assert build_idea("Gardener", "Q-008731", "seedance-2-5") == "Gardener_Q-00873_SD25"

    def test_build_idea_drops_spaces_and_punctuation(self):
        from app.sheets_client import build_idea

        # "My-Idea+ES" -> "My-IdeaES": '-' and '_' survive, everything else goes.
        assert build_idea("My-Idea+ES", "take_2", "seedance") == "My-IdeaES_take_2_SD2"
        assert build_idea("Listenify Work & Shopify", "v1", "seedance") == (
            "ListenifyWorkShopify_v1_SD2"
        )

    def test_build_idea_removes_bracketed_parts_whole(self):
        from app.sheets_client import build_idea

        # Brackets AND their contents go — "(Jp)" is a qualifier, not budget.
        assert build_idea("SheIsCheatingOnMe (Jp)", "a", "seedance") == (
            "SheIsCheatingOnMe_a_SD2"
        )
        # Unmatched opener drops the remainder; nesting is handled by depth.
        assert build_idea("Name (Jp", "a", "seedance") == "Name_a_SD2"
        assert build_idea("A[x(y)]B", "a", "seedance") == "AB_a_SD2"

    def test_build_idea_clamps_each_part_to_its_own_slot(self):
        from app.sheets_client import IDEA_MAX_LEN, build_idea

        idea = build_idea("Gardener Morning Routine Hooks Batch", "Q-00873 fast", "seedance-2-5")
        # Project 27 + "_" + run 7 + "_" + model 4 == 40 exactly.
        assert idea == "GardenerMorningRoutineHooks_Q-00873_SD25"
        assert len(idea) == IDEA_MAX_LEN

    def test_build_idea_never_exceeds_the_cap(self):
        from app.sheets_client import IDEA_MAX_LEN, build_idea

        long_name = "x" * 300
        for model in ("seedance", "seedance-fast", "seedance-mini", "seedance-2-5", "gemini-omni"):
            assert len(build_idea(long_name, long_name, model)) <= IDEA_MAX_LEN

    def test_build_idea_falls_back_to_id_prefix_for_blank_names(self):
        from app.sheets_client import build_idea

        # Unnamed project/run, and a name that sanitizes to nothing (Cyrillic).
        assert build_idea(None, "", "seedance", project_id="abcd1234-ef56", run_id="9876fedc-ba01") == (
            "abcd1234_9876fed_SD2"
        )
        assert build_idea("Проект", None, "seedance", project_id="abcd1234-ef56", run_id="9876fedc-ba01") == (
            "abcd1234_9876fed_SD2"
        )

    def test_col_letter(self):
        from app.sheets_client import _col_letter

        assert _col_letter(0) == "A"
        assert _col_letter(25) == "Z"
        assert _col_letter(26) == "AA"
        assert _col_letter(27) == "AB"

    # -- column_values(last_n=...) bounding + fallback -----------------------
    #
    # These drive SheetsClient.column_values directly against a fake Sheets
    # *service* object (the constructor's ``service=`` testing hook — see
    # SheetsClient.__init__), rather than faking SheetsClient itself, since
    # the behavior under test lives inside column_values's own range-building
    # and fallback logic.

    def test_column_values_bounded_read_used_when_it_finds_data(self):
        """When the bounded tail range comes back non-empty, that's used as-is
        (no full-column fallback read is issued)."""
        from app.sheets_client import SheetsClient, _quote_tab

        tab = "Final Creatives"
        full_range = f"{_quote_tab(tab)}!D:D"
        bounded_range = f"{_quote_tab(tab)}!D500:D"
        fake = _FakeSheetsService(
            row_count=1000,  # sheet's allocated grid size
            range_responses={
                bounded_range: ["abc123", "def456"],
                full_range: ["Job ID", "should-not-be-read"],
            },
        )
        client = SheetsClient(service=fake)

        result = client.column_values("SID", tab, 3, last_n=500)

        assert result == ["abc123", "def456"]
        assert fake.value_gets == [bounded_range]  # no fallback read issued
        assert fake.meta_gets == 1

    def test_column_values_falls_back_to_full_column_when_bounded_read_is_empty(self):
        """A young tab (real data doesn't fill the grid) must not look empty
        to the dedupe check.

        ``_row_count`` reflects the sheet's allocated GRID size (Sheets pads
        new tabs to a fixed row count, e.g. 1000, regardless of how much data
        exists), not the data extent. If the bounded start row lands past the
        real data, ``values.get`` trims the response down to nothing and
        column_values would wrongly report "not present" — causing a retried
        export to double-append. So an empty bounded read must fall back to
        a full-column read.
        """
        from app.sheets_client import SheetsClient, _quote_tab

        tab = "Final Creatives"
        full_range = f"{_quote_tab(tab)}!D:D"
        bounded_range = f"{_quote_tab(tab)}!D500:D"  # past the ~300 real data rows
        fake = _FakeSheetsService(
            row_count=1000,  # grid padding, far beyond the real data extent
            range_responses={
                bounded_range: [],  # trimmed away entirely — nothing that far down
                full_range: ["Job ID", "abc123", "def456"],
            },
        )
        client = SheetsClient(service=fake)

        result = client.column_values("SID", tab, 3, last_n=500)

        assert result == ["Job ID", "abc123", "def456"]
        assert fake.value_gets == [bounded_range, full_range]  # bounded, then fallback
        assert fake.meta_gets == 1

    def test_column_values_last_n_none_reads_full_column_directly(self):
        """The default (last_n=None) is unchanged: one full-column read, no
        row-count metadata call at all."""
        from app.sheets_client import SheetsClient, _quote_tab

        tab = "Final Creatives"
        full_range = f"{_quote_tab(tab)}!D:D"
        fake = _FakeSheetsService(
            row_count=1000, range_responses={full_range: ["Job ID", "abc123"]}
        )
        client = SheetsClient(service=fake)

        result = client.column_values("SID", tab, 3)

        assert result == ["Job ID", "abc123"]
        assert fake.value_gets == [full_range]
        assert fake.meta_gets == 0  # no row-count lookup needed


# ---------------------------------------------------------------------------
# Stitch library — /api/v2/library-clips + create_run stitch_plan
# ---------------------------------------------------------------------------


def _fake_media_info(duration=3.5, width=320, height=240, fps=25.0):
    from app.media import MediaInfo

    return MediaInfo(
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        aspect_ratio="4:3",
        has_audio=True,
    )


def _make_library_clip(session, tmp_path=None, **kwargs) -> LibraryClip:
    local = kwargs.pop("local_path", None)
    if local is None:
        local = str(tmp_path / "clip.mp4") if tmp_path else "/nonexistent/clip.mp4"
        if tmp_path:
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 32)
    defaults = dict(name="Outro EN", local_path=local, duration_sec=3.5)
    defaults.update(kwargs)
    clip = LibraryClip(**defaults)
    session.add(clip)
    session.commit()
    return clip


class TestLibraryClips:
    def test_upload_list_delete_roundtrip(self, client, SessionFactory, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        response = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro test"},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "Outro test"
        assert body["duration_sec"] == pytest.approx(3.5)
        clip_id = body["id"]

        # File landed on disk under the library dir.
        session = SessionFactory()
        clip = session.get(LibraryClip, clip_id)
        session.close()
        assert clip is not None
        assert os.path.exists(clip.local_path)

        # Listed
        listing = client.get("/api/v2/library-clips")
        assert listing.status_code == 200
        assert clip_id in [c["id"] for c in listing.json()]

        # File endpoint streams it
        file_resp = client.get(f"/api/v2/library-clips/{clip_id}/file")
        assert file_resp.status_code == 200

        # Delete removes row + file
        del_resp = client.delete(f"/api/v2/library-clips/{clip_id}")
        assert del_resp.status_code == 204
        session = SessionFactory()
        assert session.get(LibraryClip, clip_id) is None
        session.close()
        assert not os.path.exists(clip.local_path)

    def test_upload_name_defaults_to_filename(self, client, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        response = client.post(
            "/api/v2/library-clips",
            files={"file": ("my outro final.mp4", b"\x00" * 16, "video/mp4")},
        )
        assert response.status_code == 201
        assert response.json()["name"] == "my outro final.mp4"

    def test_upload_rejects_non_video(self, client, monkeypatch):
        from app.media import MediaError

        def _bad_probe(path):
            raise MediaError("no video stream")

        monkeypatch.setattr(api_v2_module.media_mod, "probe", _bad_probe)
        response = client.post(
            "/api/v2/library-clips",
            data={"name": "not a video"},
            files={"file": ("evil.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "not a playable video" in response.json()["detail"]

    def test_delete_blocked_while_active_run_references_clip(
        self, client, db_session, tmp_path
    ):
        clip = _make_library_clip(db_session, tmp_path)
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.queued,
            stitch_plan=[{"kind": "library", "clip_id": clip.id}],
        )

        response = client.delete(f"/api/v2/library-clips/{clip.id}")
        assert response.status_code == 409
        assert "active run" in response.json()["detail"]

        # Terminal run no longer blocks deletion.
        run.status = RunStatus.done
        db_session.commit()
        response = client.delete(f"/api/v2/library-clips/{clip.id}")
        assert response.status_code == 204

    def test_file_missing_on_disk_is_404(self, client, db_session):
        clip = _make_library_clip(db_session, local_path="/nonexistent/clip.mp4")
        response = client.get(f"/api/v2/library-clips/{clip.id}/file")
        assert response.status_code == 404

    def test_unknown_clip_is_404(self, client):
        assert client.get("/api/v2/library-clips/nope/file").status_code == 404
        assert client.delete("/api/v2/library-clips/nope").status_code == 404

    def test_gdrive_url_succeeds(self, client, SessionFactory, monkeypatch, tmp_path):
        """Create clip via Google Drive link — row + file exist, name defaults to file ID."""
        from app.gdrive_client import GDriveClient

        # Monkeypatch GDriveClient to fake a download.
        class FakeGDriveClient:
            def download_file(self, link_or_id, dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                with open(dst_path, "wb") as fh:
                    fh.write(b"\x00\x01" * 32)
                return dst_path

        monkeypatch.setattr(api_v2_module, "GDriveClient", FakeGDriveClient)
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )

        response = client.post(
            "/api/v2/library-clips",
            data={"gdrive_url": "https://drive.google.com/file/d/abc123def456/view"},
        )
        assert response.status_code == 201
        body = response.json()
        # Default name should be the file ID extracted from the URL.
        assert body["name"] == "abc123def456"
        assert body["duration_sec"] == pytest.approx(3.5)
        clip_id = body["id"]

        # File and row exist on disk.
        session = SessionFactory()
        clip = session.get(LibraryClip, clip_id)
        session.close()
        assert clip is not None
        assert os.path.exists(clip.local_path)

    def test_gdrive_url_with_explicit_name(self, client, monkeypatch):
        """Create clip via Drive link with explicit name."""
        from app.gdrive_client import GDriveClient

        class FakeGDriveClient:
            def download_file(self, link_or_id, dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                with open(dst_path, "wb") as fh:
                    fh.write(b"\x00" * 16)
                return dst_path

        monkeypatch.setattr(api_v2_module, "GDriveClient", FakeGDriveClient)
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )

        response = client.post(
            "/api/v2/library-clips",
            data={
                "gdrive_url": "https://drive.google.com/open?id=xyz789",
                "name": "Custom Drive Clip",
            },
        )
        assert response.status_code == 201
        assert response.json()["name"] == "Custom Drive Clip"

    def test_neither_file_nor_gdrive_url_is_400(self, client):
        """Reject request with neither file nor gdrive_url."""
        response = client.post("/api/v2/library-clips")
        assert response.status_code == 400
        assert "Provide either a file or a Drive link" in response.json()["detail"]

    def test_both_file_and_gdrive_url_is_400(self, client, monkeypatch):
        """Reject request with both file and gdrive_url."""
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        response = client.post(
            "/api/v2/library-clips",
            data={"gdrive_url": "https://drive.google.com/file/d/abc/view"},
            files={"file": ("clip.mp4", b"\x00" * 16, "video/mp4")},
        )
        assert response.status_code == 400
        assert "not both" in response.json()["detail"]

    def test_gdrive_download_error_is_400_and_cleanup(self, client, monkeypatch, SessionFactory):
        """Drive download error → 400, library clip dir is removed."""
        from app.gdrive_client import GDriveClient, GDriveDownloadError

        clip_dir_holder = {"path": None}

        class FailingGDriveClient:
            def download_file(self, link_or_id, dst_path):
                # Capture the directory path before error is raised.
                # Use os.path.dirname to get the dir WITHOUT calling library_clip_dir again.
                clip_dir_holder["path"] = os.path.dirname(dst_path)
                raise GDriveDownloadError("Permission denied")

        monkeypatch.setattr(api_v2_module, "GDriveClient", FailingGDriveClient)

        response = client.post(
            "/api/v2/library-clips",
            data={"gdrive_url": "https://drive.google.com/file/d/nope/view"},
        )
        assert response.status_code == 400
        assert "Failed to download from Drive" in response.json()["detail"]

        # Verify the clip dir was cleaned up (does not exist).
        # Use the captured path directly to avoid recreating it via library_clip_dir().
        if clip_dir_holder["path"]:
            assert not os.path.exists(
                clip_dir_holder["path"]
            ), f"Clip dir {clip_dir_holder['path']} should be cleaned up"


class TestCreateRunStitchPlan:
    def _ready_project_with_segments(self, session):
        project = _make_project(session, status=ProjectStatus.ready)
        segs = [
            _make_segment_def(session, project.id, 0, action="swap"),
            _make_segment_def(session, project.id, 1, action="keep", has_face=False),
            _make_segment_def(session, project.id, 2, action="swap"),
        ]
        return project, segs

    def _post_run(self, client, project_id, stitch_plan):
        data = {"prompt": "swap character"}
        if stitch_plan is not None:
            data["stitch_plan"] = stitch_plan
        return client.post(f"/api/v2/projects/{project_id}/runs", data=data)

    def test_invalid_json_is_400(self, client, db_session):
        project, _ = self._ready_project_with_segments(db_session)
        response = self._post_run(client, project.id, "{not json")
        assert response.status_code == 400
        assert "valid JSON" in response.json()["detail"]

    def test_non_list_is_400(self, client, db_session):
        project, _ = self._ready_project_with_segments(db_session)
        response = self._post_run(client, project.id, '{"kind": "segment"}')
        assert response.status_code == 400
        assert "JSON list" in response.json()["detail"]

    def test_unknown_kind_is_400(self, client, db_session):
        project, _ = self._ready_project_with_segments(db_session)
        response = self._post_run(client, project.id, '[{"kind": "banana"}]')
        assert response.status_code == 400
        assert "kind" in response.json()["detail"]

    def test_foreign_segment_is_400(self, client, db_session):
        project, _ = self._ready_project_with_segments(db_session)
        other_project, other_segs = self._ready_project_with_segments(db_session)
        plan = json.dumps(
            [{"kind": "segment", "segment_def_id": other_segs[0].id}]
        )
        response = self._post_run(client, project.id, plan)
        assert response.status_code == 400
        assert "does not belong" in response.json()["detail"]

    def test_duplicate_segment_is_400(self, client, db_session):
        project, segs = self._ready_project_with_segments(db_session)
        plan = json.dumps(
            [
                {"kind": "segment", "segment_def_id": segs[0].id},
                {"kind": "segment", "segment_def_id": segs[0].id},
            ]
        )
        response = self._post_run(client, project.id, plan)
        assert response.status_code == 400
        assert "twice" in response.json()["detail"]

    def test_unknown_library_clip_is_400(self, client, db_session):
        project, segs = self._ready_project_with_segments(db_session)
        plan = json.dumps([{"kind": "library", "clip_id": "no-such-clip"}])
        response = self._post_run(client, project.id, plan)
        assert response.status_code == 400
        assert "library clip" in response.json()["detail"]

    def test_empty_plan_stored_as_empty_list(self, client, db_session, SessionFactory):
        project, _ = self._ready_project_with_segments(db_session)
        response = self._post_run(client, project.id, "[]")
        assert response.status_code == 201
        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        session.close()
        assert run.stitch_plan == []

    def test_custom_plan_stored_normalized(
        self, client, db_session, SessionFactory, tmp_path
    ):
        project, segs = self._ready_project_with_segments(db_session)
        clip = _make_library_clip(db_session, tmp_path)
        plan = json.dumps(
            [
                {"kind": "segment", "segment_def_id": segs[0].id, "junk": "x"},
                {"kind": "segment", "segment_def_id": segs[1].id},
                {"kind": "library", "clip_id": clip.id},
            ]
        )
        response = self._post_run(client, project.id, plan)
        assert response.status_code == 201
        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        session.close()
        assert run.stitch_plan == [
            {"kind": "segment", "segment_def_id": segs[0].id},
            {"kind": "segment", "segment_def_id": segs[1].id},
            {"kind": "library", "clip_id": clip.id},
        ]
        # Exposed on the run response too.
        detail = client.get(f"/api/v2/runs/{run.id}")
        assert detail.status_code == 200
        assert detail.json()["stitch_plan"] == run.stitch_plan

    def test_plan_equal_to_default_stored_as_null(
        self, client, db_session, SessionFactory
    ):
        project, segs = self._ready_project_with_segments(db_session)
        plan = json.dumps(
            [{"kind": "segment", "segment_def_id": s.id} for s in segs]
        )
        response = self._post_run(client, project.id, plan)
        assert response.status_code == 201
        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        session.close()
        assert run.stitch_plan is None

    def test_absent_plan_stored_as_null(self, client, db_session, SessionFactory):
        project, _ = self._ready_project_with_segments(db_session)
        response = self._post_run(client, project.id, None)
        assert response.status_code == 201
        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        session.close()
        assert run.stitch_plan is None

    def test_copy_run_clones_stitch_plan(
        self, client, db_session, SessionFactory, tmp_path
    ):
        project, segs = self._ready_project_with_segments(db_session)
        clip = _make_library_clip(db_session, tmp_path)
        src_plan = [
            {"kind": "segment", "segment_def_id": segs[0].id},
            {"kind": "library", "clip_id": clip.id},
        ]
        src = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            stitch_plan=src_plan,
        )
        response = client.post(f"/api/v2/runs/{src.id}/copy", data={})
        assert response.status_code == 201
        session = SessionFactory()
        copied = session.get(Run, response.json()["run_id"])
        session.close()
        assert copied.stitch_plan == src_plan


class TestStitchPlanRegressions:
    """Regressions from review: excluded segments must not leave pending
    RunSegments, and copies must fail fast on stale plan references."""

    def test_segment_prompt_for_excluded_segment_creates_no_run_segment(
        self, client, db_session, SessionFactory
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg0 = _make_segment_def(db_session, project.id, 0, action="swap")
        seg1 = _make_segment_def(db_session, project.id, 1, action="swap")
        # Custom plan keeps only seg0 — seg1 is excluded.
        plan = json.dumps([{"kind": "segment", "segment_def_id": seg0.id}])
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "stitch_plan": plan,
                "segment_prompts": json.dumps(
                    {seg0.id: "extra A", seg1.id: "extra B"}
                ),
            },
        )
        assert response.status_code == 201
        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        rs_seg_ids = {rs.segment_def_id for rs in run.run_segments}
        session.close()
        assert seg0.id in rs_seg_ids
        assert seg1.id not in rs_seg_ids  # excluded → no pending row

    def test_copy_run_with_deleted_clip_in_plan_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0, action="swap")
        src = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            stitch_plan=[
                {"kind": "segment", "segment_def_id": seg.id},
                {"kind": "library", "clip_id": "deleted-clip"},
            ],
        )
        response = client.post(f"/api/v2/runs/{src.id}/copy", data={})
        assert response.status_code == 400
        assert "library clip" in response.json()["detail"]

    def test_copy_run_with_stale_segment_in_plan_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")
        src = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            stitch_plan=[
                {"kind": "segment", "segment_def_id": "gone-segment-id"}
            ],
        )
        response = client.post(f"/api/v2/runs/{src.id}/copy", data={})
        assert response.status_code == 400
        assert "no longer exists" in response.json()["detail"]

    def test_copy_run_skips_run_segments_excluded_by_plan(
        self, client, db_session, SessionFactory
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg0 = _make_segment_def(db_session, project.id, 0, action="swap")
        seg1 = _make_segment_def(db_session, project.id, 1, action="swap")
        src = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            stitch_plan=[{"kind": "segment", "segment_def_id": seg0.id}],
        )
        # Source run has prompt overrides on BOTH segments.
        _make_run_segment(
            db_session, src.id, seg0.id, index=0, prompt_override="tuned A"
        )
        _make_run_segment(
            db_session, src.id, seg1.id, index=1, prompt_override="tuned B"
        )
        response = client.post(f"/api/v2/runs/{src.id}/copy", data={})
        assert response.status_code == 201
        session = SessionFactory()
        copied = session.get(Run, response.json()["run_id"])
        rs_seg_ids = {rs.segment_def_id for rs in copied.run_segments}
        session.close()
        assert seg0.id in rs_seg_ids
        assert seg1.id not in rs_seg_ids  # excluded by plan → not cloned

    def test_delete_clip_blocked_by_failed_run(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path)
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(
            db_session,
            project.id,
            status=RunStatus.failed,  # retryable → must block deletion
            stitch_plan=[{"kind": "library", "clip_id": clip.id}],
        )
        response = client.delete(f"/api/v2/library-clips/{clip.id}")
        assert response.status_code == 409

    def test_delete_clip_succeeds_when_active_run_references_other_clip(
        self, client, db_session, tmp_path
    ):
        """SQL prefilter (LIKE on serialized stitch_plan) must not produce a
        false block: an active run whose plan references a *different*
        library clip (or none at all) shouldn't stop this clip's deletion.
        """
        clip = _make_library_clip(db_session, tmp_path)
        other_clip = _make_library_clip(db_session, tmp_path, name="other")
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0, action="swap")
        _make_run(
            db_session,
            project.id,
            status=RunStatus.queued,
            stitch_plan=[
                {"kind": "segment", "segment_def_id": seg.id},
                {"kind": "library", "clip_id": other_clip.id},
            ],
        )
        response = client.delete(f"/api/v2/library-clips/{clip.id}")
        assert response.status_code == 204


# ---------------------------------------------------------------------------
# Tail edits ("Change ending" — trim a done run's result + optional
# library-clip append; app/pipeline_tail.py)
# ---------------------------------------------------------------------------


def _make_tail_edit(session, run_id: str, **kwargs) -> RunTailEdit:
    defaults = dict(run_id=run_id, cut_sec=5.0, status=TailEditStatus.queued)
    defaults.update(kwargs)
    edit = RunTailEdit(**defaults)
    session.add(edit)
    session.commit()
    return edit


def _make_caption(session, run_id: str, **kwargs) -> RunCaption:
    """Mirrors _make_tail_edit — used here only as a chained-source fixture
    for TestTailEdits' source_caption_id tests (docs/burned-captions.md §3.7).
    """
    defaults = dict(run_id=run_id, status=CaptionStatus.queued)
    defaults.update(kwargs)
    c = RunCaption(**defaults)
    session.add(c)
    session.commit()
    return c


class TestTailEdits:
    def test_create_on_non_done_run_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 3.0}
        )
        assert response.status_code == 409

    def test_create_on_done_run_without_result_file_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 3.0}
        )
        assert response.status_code == 409

    def test_create_with_unknown_clip_is_400(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 3.0, "library_clip_id": "does-not-exist"},
        )
        assert response.status_code == 400

    def test_create_nonpositive_cut_sec_is_400(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 0}
        )
        assert response.status_code == 400

    def test_create_happy_path_creates_row_and_enqueues(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_duration_sec=100.0,
        )
        clip = _make_library_clip(db_session, tmp_path)

        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_process_tail_edit", enqueued.append
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 6.5, "library_clip_id": clip.id},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "queued"
        edit_id = body["tail_edit_id"]
        assert enqueued == [edit_id]

        session = SessionFactory()
        edit = session.get(RunTailEdit, edit_id)
        session.close()
        assert edit is not None
        assert edit.run_id == run.id
        assert edit.cut_sec == pytest.approx(6.5)
        assert edit.library_clip_id == clip.id
        assert edit.status == TailEditStatus.queued

    def test_create_trim_only_omits_clip(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_duration_sec=100.0,
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 4.0}
        )
        assert response.status_code == 201
        assert response.json()["status"] == "queued"

    def test_create_omitting_source_caption_id_defaults_to_run(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """v1 regression guard (docs/burned-captions.md §3.7.3): no
        source_caption_id field at all still yields source_type='run' and a
        NULL source_caption_id, byte-for-byte the pre-chaining behaviour."""
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_duration_sec=100.0,
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 3.0}
        )
        assert response.status_code == 201
        edit_id = response.json()["tail_edit_id"]

        session = SessionFactory()
        edit = session.get(RunTailEdit, edit_id)
        session.close()
        assert edit.source_type == "run"
        assert edit.source_caption_id is None

    def test_create_with_source_caption_id_sets_source_type_and_id(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        # A REAL 1s video: source_type="caption" has no duration cache, so
        # create_tail_edit always ffprobes it — a fake byte string would
        # 409 here instead of the 201 this test expects.
        caption_output = tmp_path / "captioned.mp4"
        _make_ffmpeg_video(str(caption_output))
        caption = _make_caption(
            db_session, run.id,
            status=CaptionStatus.done,
            output_path=str(caption_output),
        )

        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_process_tail_edit", enqueued.append
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 0.5, "source_caption_id": caption.id},
        )
        assert response.status_code == 201
        edit_id = response.json()["tail_edit_id"]
        assert enqueued == [edit_id]

        session = SessionFactory()
        edit = session.get(RunTailEdit, edit_id)
        session.close()
        assert edit.source_type == "caption"
        assert edit.source_caption_id == caption.id

    def test_create_with_unknown_source_caption_id_is_400(
        self, client, db_session, tmp_path
    ):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 3.0, "source_caption_id": "does-not-exist"},
        )
        assert response.status_code == 400

    def test_create_with_source_caption_from_different_run_is_400(
        self, client, db_session, tmp_path
    ):
        """A caption belonging to a DIFFERENT run must not be usable as a
        source even though it is done with a file on disk — 400, not 409."""
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        other_run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        caption_output = tmp_path / "captioned.mp4"
        caption_output.write_bytes(b"\x00CAPTIONED\xff")
        caption = _make_caption(
            db_session, other_run.id,
            status=CaptionStatus.done,
            output_path=str(caption_output),
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 3.0, "source_caption_id": caption.id},
        )
        assert response.status_code == 400

    def test_create_with_source_caption_not_done_is_409(
        self, client, db_session, tmp_path
    ):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        caption = _make_caption(db_session, run.id, status=CaptionStatus.transcribing)

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 3.0, "source_caption_id": caption.id},
        )
        assert response.status_code == 409

    def test_create_with_source_caption_done_but_file_missing_is_409(
        self, client, db_session, tmp_path
    ):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        caption = _make_caption(
            db_session, run.id,
            status=CaptionStatus.done,
            output_path="/no/such/path/captioned.mp4",
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 3.0, "source_caption_id": caption.id},
        )
        assert response.status_code == 409

    # -----------------------------------------------------------------
    # cut_sec bounds (fix/tail-edit-cut-bounds) — a real, enforced upper
    # bound instead of a job that's created, queued, and fails a worker
    # cycle later.
    # -----------------------------------------------------------------

    def test_create_cut_sec_beyond_duration_is_400(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        """cut_sec strictly greater than the resolved source's duration must
        400 synchronously — no RunTailEdit row, nothing enqueued."""
        result_file = tmp_path / "final.mp4"
        _make_ffmpeg_video(str(result_file))  # real video, probes to 1.0s
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_process_tail_edit", enqueued.append
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 5.0}
        )
        assert response.status_code == 400
        assert "1.00" in response.json()["detail"]
        assert enqueued == []

        session = SessionFactory()
        assert session.query(RunTailEdit).filter_by(run_id=run.id).count() == 0
        session.close()

    def test_create_cut_sec_at_or_over_duration_is_400(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """The exact "hit the very end" case: cut_sec == duration, and a hair
        OVER duration — the shape of the reported bug, where a rounded-UP
        hint ("12.3s" for a real 12.28s video) led the operator to type a
        value that was never achievable. Both must 400, not just land in the
        worker a minute later."""
        result_file = tmp_path / "final.mp4"
        _make_ffmpeg_video(str(result_file))  # real video, probes to 1.0s
        project = _make_project(db_session, status=ProjectStatus.ready)
        run_ids = []

        for cut_sec in (1.0, 1.001):
            run = _make_run(
                db_session, project.id,
                status=RunStatus.done,
                result_local_path=str(result_file),
            )
            run_ids.append(run.id)
            response = client.post(
                f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": cut_sec}
            )
            assert response.status_code == 400, cut_sec

        session = SessionFactory()
        assert (
            session.query(RunTailEdit)
            .filter(RunTailEdit.run_id.in_(run_ids))
            .count() == 0
        )
        session.close()

    def test_create_400_names_the_bound_floored_not_rounded(
        self, client, db_session, tmp_path
    ):
        """Cursor-review regression (fix/tail-edit-cut-bounds §1 gate): the
        400 detail must FLOOR the duration, not `:.2f`-format it — `:.2f`
        rounds, so a 12.288s source would be named "12.29" here, a value
        >= the real duration. An operator who retypes exactly the number
        this message told them would get the same 400 again."""
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_duration_sec=12.288,
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 13.0}
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "12.28" in detail
        assert "12.29" not in detail

    def test_create_cut_sec_within_margin_of_zero_is_400(
        self, client, db_session, tmp_path
    ):
        """cut_sec in (0, _MIN_CUT_MARGIN_SEC] used to pass here (the old
        check was a bare `cut_sec <= 0`) and only failed a cycle later in the
        worker (`> _MIN_CUT_MARGIN_SEC`) — the two must agree."""
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_duration_sec=100.0,
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 0.03}
        )
        assert response.status_code == 400

    def test_create_with_source_caption_shorter_than_run_beyond_caption_is_400(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """Key regression: a caption source can be SHORTER than its run (it
        may itself be built on a tail edit — docs/burned-captions.md §3.7).
        A cut_sec between the caption's length and the run's must 400 — today
        this passes validation (checked against nothing) and only fails in
        the worker, which correctly probes the RESOLVED source."""
        result_file = tmp_path / "final.mp4"
        _make_ffmpeg_video(str(result_file))  # the run: real, 1.0s
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

        caption_output = tmp_path / "captioned_short.mp4"
        _make_ffmpeg_video_with_duration(str(caption_output), 0.3)  # shorter
        caption = _make_caption(
            db_session, run.id,
            status=CaptionStatus.done,
            output_path=str(caption_output),
        )

        # 0.5s is well inside the RUN's (margin, 1.0) but beyond the
        # CAPTION's (margin, 0.3) — the bug this test guards against.
        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits",
            data={"cut_sec": 0.5, "source_caption_id": caption.id},
        )
        assert response.status_code == 400
        assert "0.30" in response.json()["detail"]

        session = SessionFactory()
        assert session.query(RunTailEdit).filter_by(run_id=run.id).count() == 0
        session.close()

    def test_create_backfills_null_result_duration_sec(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """A run finalized before result_duration_sec existed (NULL): the
        upper-bound check probes it once and writes the value back to the
        row, the same self-healing backfill web_v2._get_tail_edit_context
        already does for the run-detail page."""
        result_file = tmp_path / "final.mp4"
        _make_ffmpeg_video(str(result_file))  # real video, probes to 1.0s
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_duration_sec=None,
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": 0.5}
        )
        assert response.status_code == 201

        session = SessionFactory()
        refreshed = session.get(Run, run.id)
        assert refreshed.result_duration_sec == pytest.approx(1.0)
        session.close()

    def test_list_tail_edits(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        _make_tail_edit(db_session, run.id, cut_sec=2.0)
        _make_tail_edit(db_session, run.id, cut_sec=4.0, status=TailEditStatus.done)

        response = client.get(f"/api/v2/runs/{run.id}/tail-edits")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 2
        assert {item["cut_sec"] for item in body} == {2.0, 4.0}

    def test_list_tail_edits_resolves_library_clip_name(self, client, db_session, tmp_path):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        clip = _make_library_clip(db_session, tmp_path, name="Outro EN")
        _make_tail_edit(db_session, run.id, cut_sec=2.0, library_clip_id=clip.id)

        response = client.get(f"/api/v2/runs/{run.id}/tail-edits")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["library_clip_id"] == clip.id
        assert body[0]["library_clip_name"] == "Outro EN"

    def test_list_tail_edits_library_clip_name_none_when_clip_deleted(
        self, client, db_session
    ):
        """library_clip_id is a soft reference (no FK) — a deleted clip resolves to None."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        _make_tail_edit(db_session, run.id, cut_sec=2.0, library_clip_id="deleted-clip-id")

        response = client.get(f"/api/v2/runs/{run.id}/tail-edits")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["library_clip_id"] == "deleted-clip-id"
        assert body[0]["library_clip_name"] is None

    def test_list_tail_edits_includes_source_fields(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.done)
        _make_tail_edit(
            db_session, run.id, cut_sec=2.0,
            source_type="caption", source_caption_id=caption.id,
        )
        _make_tail_edit(db_session, run.id, cut_sec=4.0)  # default source_type="run"

        response = client.get(f"/api/v2/runs/{run.id}/tail-edits")
        assert response.status_code == 200
        body = response.json()
        by_cut = {item["cut_sec"]: item for item in body}
        assert by_cut[2.0]["source_type"] == "caption"
        assert by_cut[2.0]["source_caption_id"] == caption.id
        assert by_cut[4.0]["source_type"] == "run"
        assert by_cut[4.0]["source_caption_id"] is None

    def test_list_tail_edits_includes_export_fields(self, client, db_session):
        from datetime import datetime, timezone

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        exported_at = datetime.now(timezone.utc)
        _make_tail_edit(
            db_session, run.id,
            cut_sec=2.0,
            status=TailEditStatus.done,
            final_creatives_exported_at=exported_at,
        )

        response = client.get(f"/api/v2/runs/{run.id}/tail-edits")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["final_creatives_exported_at"] is not None

    def test_retry_failed_tail_edit(
        self, client, db_session, SessionFactory, monkeypatch
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(
            db_session, run.id, status=TailEditStatus.failed, error_message="boom"
        )

        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_process_tail_edit", enqueued.append
        )

        response = client.post(f"/api/v2/tail-edits/{edit.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert enqueued == [edit.id]

    def test_retry_done_tail_edit(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.done)

        response = client.post(f"/api/v2/tail-edits/{edit.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    def test_retry_active_tail_edit_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.queued)

        response = client.post(f"/api/v2/tail-edits/{edit.id}/retry")
        assert response.status_code == 409

    def test_retry_unknown_id_is_404(self, client):
        response = client.post("/api/v2/tail-edits/nope/retry")
        assert response.status_code == 404

    def test_result_download_on_done_edit(self, client, db_session, tmp_path):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        output_file = tmp_path / "tail_edited.mp4"
        output_file.write_bytes(b"\x00TAILEDIT\xff")
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            output_path=str(output_file),
        )

        response = client.get(f"/api/v2/tail-edits/{edit.id}/result")
        assert response.status_code == 200
        assert response.content == b"\x00TAILEDIT\xff"

    def test_result_download_not_done_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.queued)

        response = client.get(f"/api/v2/tail-edits/{edit.id}/result")
        assert response.status_code == 409

    def test_delete_tail_edit_removes_row_and_dir(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.done)

        edit_dir = tmp_path / "tail_edit_out"
        edit_dir.mkdir()
        (edit_dir / "out.mp4").write_bytes(b"\x00TAILEDIT\xff")
        monkeypatch.setattr(
            api_v2_module, "tail_edit_dir", lambda eid, pid: str(edit_dir)
        )

        response = client.delete(f"/api/v2/tail-edits/{edit.id}")
        assert response.status_code == 204
        assert not edit_dir.exists()

        session = SessionFactory()
        assert session.get(RunTailEdit, edit.id) is None
        session.close()

    def test_delete_active_tail_edit_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.processing)

        response = client.delete(f"/api/v2/tail-edits/{edit.id}")
        assert response.status_code == 409
        assert "processing" in response.json()["detail"]

    def test_delete_tail_edit_blocked_by_active_overlay_job(
        self, client, db_session
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.done)
        job = OverlayJob(
            status=OverlayStatus.processing,
            source_type="tail_edit",
            source_tail_edit_id=edit.id,
            overlays=[],
        )
        db_session.add(job)
        db_session.commit()

        response = client.delete(f"/api/v2/tail-edits/{edit.id}")
        assert response.status_code == 409
        assert "active overlay" in response.json()["detail"]

        # A terminal overlay job keeps only a soft reference — no longer blocks.
        job.status = OverlayStatus.done
        db_session.commit()
        response = client.delete(f"/api/v2/tail-edits/{edit.id}")
        assert response.status_code == 204

    def test_delete_tail_edit_unknown_id_is_404(self, client):
        assert client.delete("/api/v2/tail-edits/nope").status_code == 404

    def test_delete_library_clip_blocked_by_active_tail_edit(
        self, client, db_session, tmp_path
    ):
        clip = _make_library_clip(db_session, tmp_path)
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.processing,
            library_clip_id=clip.id,
        )

        response = client.delete(f"/api/v2/library-clips/{clip.id}")
        assert response.status_code == 409
        assert "active tail edit" in response.json()["detail"]

        # Terminal tail edit no longer blocks deletion.
        edit.status = TailEditStatus.done
        db_session.commit()
        response = client.delete(f"/api/v2/library-clips/{clip.id}")
        assert response.status_code == 204


# ---------------------------------------------------------------------------
# Project types  (POST /api/v2/projects?project_type=…)
# ---------------------------------------------------------------------------


class TestProjectType:
    def test_defaults_to_face_swap(self, spy_client, SessionFactory):
        client, _ = spy_client
        response = client.post(
            "/api/v2/projects",
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.project_type == "face_swap"

    @pytest.mark.parametrize(
        "project_type", ["face_swap", "subtitle_removal", "cover_change"]
    )
    def test_accepts_each_registered_type(
        self, spy_client, SessionFactory, project_type
    ):
        client, _ = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": project_type},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.project_type == project_type

    def test_gdrive_source_carries_the_type(self, spy_client, SessionFactory):
        client, _ = spy_client
        response = client.post(
            "/api/v2/projects",
            data={
                "gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view",
                "project_type": "cover_change",
            },
        )
        assert response.status_code == 201

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.project_type == "cover_change"

    def test_unknown_type_is_400(self, client):
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "teleportation"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "project_type" in response.json()["detail"]

    def test_type_is_exposed_on_get(self, client, db_session):
        project = _make_project(db_session, project_type="subtitle_removal")
        response = client.get(f"/api/v2/projects/{project.id}")
        assert response.status_code == 200
        assert response.json()["project_type"] == "subtitle_removal"


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/to-project  ("Move to new project")
# ---------------------------------------------------------------------------


class TestPromoteRunToProject:
    def _done_run(self, session, tmp_path, **run_kwargs):
        project = _make_project(session, status=ProjectStatus.ready, name="Parent")
        result = tmp_path / "final.mp4"
        result.write_bytes(b"final-video-bytes")
        run = _make_run(
            session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(result),
            **run_kwargs,
        )
        return project, run, result

    def test_creates_new_project_with_copied_source(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        client, spy = spy_client
        _, run, result = self._done_run(db_session, tmp_path, name="Redhead woman")

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project",
            data={"project_type": "subtitle_removal"},
        )
        assert response.status_code == 201
        new_id = response.json()["project_id"]
        assert response.json()["status"] == "analyzing"  # AICP-350
        assert spy["analyze_project"] == [new_id]

        session = SessionFactory()
        new_project = session.get(VideoProject, new_id)
        session.close()
        assert new_project.project_type == "subtitle_removal"
        assert new_project.source_run_id == run.id
        assert new_project.source_type == "upload"
        assert new_project.name == "Redhead woman → Removing subtitles"

        # The video is COPIED — the new project owns its own file.
        assert new_project.source_local_path != str(result)
        assert os.path.exists(new_project.source_local_path)
        with open(new_project.source_local_path, "rb") as fh:
            assert fh.read() == b"final-video-bytes"
        assert result.exists()

    def test_inherits_mute_source_from_source_project(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        """The follow-up project continues the lineage's "remove original
        audio" choice instead of silently re-enabling audio input."""
        client, _ = spy_client
        project, run, _ = self._done_run(db_session, tmp_path)
        project.mute_source = True
        db_session.commit()

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project", data={"project_type": "cover_change"}
        )
        assert response.status_code == 201

        session = SessionFactory()
        new_project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert new_project.mute_source is True

    @pytest.mark.parametrize("parent_muted", [False, True])
    def test_localisation_project_from_run_is_muted(
        self, spy_client, SessionFactory, db_session, tmp_path, parent_muted
    ):
        """AICP-339: a localisation project gets its type's default
        mute_source=True even when the source project was not muted — there is
        no Create Project form here to pre-check the box."""
        client, _ = spy_client
        project, run, _ = self._done_run(db_session, tmp_path)
        project.mute_source = parent_muted
        db_session.commit()

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"}
        )
        assert response.status_code == 201

        session = SessionFactory()
        new_project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert new_project.mute_source is True

    @pytest.mark.parametrize(
        "project_type", ["face_swap", "subtitle_removal", "cover_change"]
    )
    def test_other_types_keep_inheriting_unmuted(
        self, spy_client, SessionFactory, db_session, tmp_path, project_type
    ):
        """Types whose default is off keep the lineage's (unmuted) choice."""
        client, _ = spy_client
        _, run, _ = self._done_run(db_session, tmp_path)

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project", data={"project_type": project_type}
        )
        assert response.status_code == 201

        session = SessionFactory()
        new_project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert new_project.mute_source is False

    def test_inherits_max_segment_sec_from_source_project(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        """The follow-up project is cut to the same maximum segment length.

        It is analyzed immediately, and an operator who chose 30s segments for
        the face swap wants the same cut for the cover change — dropping back to
        the universal default would silently re-fragment the video.
        """
        client, _ = spy_client
        project, run, _ = self._done_run(db_session, tmp_path)
        project.max_segment_sec = 30.0
        db_session.commit()

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project", data={"project_type": "cover_change"}
        )
        assert response.status_code == 201

        session = SessionFactory()
        new_project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert new_project.max_segment_sec == 30.0

    def test_null_max_segment_sec_is_inherited_as_null(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        """A parent with no explicit choice does not gain one."""
        client, _ = spy_client
        _, run, _ = self._done_run(db_session, tmp_path)

        response = client.post(f"/api/v2/runs/{run.id}/to-project")
        assert response.status_code == 201

        session = SessionFactory()
        new_project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert new_project.max_segment_sec is None

    @pytest.mark.parametrize("bad_cap", [2.0, 3.0, 45.0, float("inf")])
    def test_unusable_parent_max_segment_sec_is_not_propagated(
        self, spy_client, SessionFactory, db_session, tmp_path, bad_cap
    ):
        """A cap already in the DB that no backend can honour stops here.

        Rows written before the floor was raised to the registry's output minimum
        (and anything set by direct SQL) can hold a value the API would now
        reject. Copying it verbatim would launder it into a fresh project — and
        into every project promoted from that one — so the copy re-validates and
        falls back to NULL.

        2.0/3.0 are the legacy accepted-but-un-generatable values; inf is what a
        pre-fix POST could persist. NaN is the fourth case but cannot be tested
        here: SQLite silently stores NaN as NULL (Postgres keeps it), so it is
        covered at the validator level in TestProjectMaxSegmentSec instead.
        """
        client, spy = spy_client
        project, run, _ = self._done_run(db_session, tmp_path)
        project.max_segment_sec = bad_cap
        db_session.commit()

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project", data={"project_type": "cover_change"}
        )
        # The copy itself is valid — the caller never sent this field — so it
        # succeeds and the child simply gets the universal default.
        assert response.status_code == 201, response.text
        new_id = response.json()["project_id"]
        assert spy["analyze_project"] == [new_id]

        session = SessionFactory()
        new_project = session.get(VideoProject, new_id)
        session.close()
        assert new_project.max_segment_sec is None

        # The parent is left exactly as it was: this endpoint copies, it does not
        # quietly repair someone else's row (that is a PATCH the operator makes).
        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id).max_segment_sec
        assert stored == bad_cap

    def test_explicit_name_wins(self, spy_client, SessionFactory, db_session, tmp_path):
        client, _ = spy_client
        _, run, _ = self._done_run(db_session, tmp_path)

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project",
            data={"project_type": "cover_change", "name": "Cover v2"},
        )
        assert response.status_code == 201

        session = SessionFactory()
        new_project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert new_project.name == "Cover v2"

    def test_defaults_to_face_swap_type(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        client, _ = spy_client
        _, run, _ = self._done_run(db_session, tmp_path)

        response = client.post(f"/api/v2/runs/{run.id}/to-project")
        assert response.status_code == 201

        session = SessionFactory()
        new_project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert new_project.project_type == "face_swap"

    def test_run_not_done_is_409(self, client, db_session, tmp_path):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.post(f"/api/v2/runs/{run.id}/to-project")
        assert response.status_code == 409
        assert "not done" in response.json()["detail"]

    def test_missing_result_file_is_409(self, client, db_session, tmp_path):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(tmp_path / "gone.mp4"),
        )

        response = client.post(f"/api/v2/runs/{run.id}/to-project")
        assert response.status_code == 409
        assert "no final video" in response.json()["detail"]

    def test_unknown_type_is_400(self, client, db_session, tmp_path):
        project = _make_project(db_session, status=ProjectStatus.ready)
        result = tmp_path / "final.mp4"
        result.write_bytes(b"x")
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_local_path=str(result)
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/to-project", data={"project_type": "nope"}
        )
        assert response.status_code == 400

    def test_missing_run_is_404(self, client):
        response = client.post("/api/v2/runs/does-not-exist/to-project")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v2/projects/{pid}/extract-insert — settings the child inherits
# ---------------------------------------------------------------------------


class TestExtractInsertInheritsProjectSettings:
    """The PIP-insert child is a crop of the parent, so it carries the parent's
    project-level settings over rather than restarting from the defaults.

    (The endpoint's own validation and provenance rows are covered in
    tests/test_extract_insert.py; this class only pins the inheritance.)
    """

    @staticmethod
    def _stub_crop_clip(monkeypatch):
        """Write a placeholder file and return evened dims — no real ffmpeg."""
        def _fake_crop_clip(src, start_sec, end_sec, x, y, width, height, dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"fake-cropped-clip")
            return (width - (width % 2), height - (height % 2))

        monkeypatch.setattr(api_v2_module.media_mod, "crop_clip", _fake_crop_clip)

    @staticmethod
    def _analyzed_parent(session, **kwargs):
        """A probed parent project with a source file on disk."""
        from app.storage import project_source_path

        parent = _make_project(
            session,
            status=ProjectStatus.ready,
            width=640,
            height=480,
            duration_sec=30.0,
            fps=30.0,
            aspect_ratio="4:3",
            **kwargs,
        )
        src_path = project_source_path(parent.id, "mp4")
        with open(src_path, "wb") as fh:
            fh.write(b"fake-source-video-bytes")
        parent.source_local_path = src_path
        session.commit()
        return parent

    _FORM = {
        "x": 10, "y": 20, "width": 100, "height": 50,
        "start_sec": 1.0, "end_sec": 5.0,
    }

    def _extract(self, client, parent_id):
        response = client.post(
            f"/api/v2/projects/{parent_id}/extract-insert", data=self._FORM
        )
        assert response.status_code == 201, response.text
        return response.json()["child_project_id"]

    def test_child_inherits_max_segment_sec(
        self, client, db_session, SessionFactory, monkeypatch
    ):
        """The child is analyzed straight away and is meant for the same model,
        so it must be cut to the same maximum segment length."""
        self._stub_crop_clip(monkeypatch)
        parent = self._analyzed_parent(db_session, max_segment_sec=30.0)

        child_id = self._extract(client, parent.id)

        session = SessionFactory()
        child = session.get(VideoProject, child_id)
        session.close()
        assert child.max_segment_sec == 30.0

    def test_child_inherits_null_max_segment_sec(
        self, client, db_session, SessionFactory, monkeypatch
    ):
        """A parent with no explicit cap gives the child no explicit cap."""
        self._stub_crop_clip(monkeypatch)
        parent = self._analyzed_parent(db_session)

        child_id = self._extract(client, parent.id)

        session = SessionFactory()
        child = session.get(VideoProject, child_id)
        session.close()
        assert child.max_segment_sec is None

    @pytest.mark.parametrize("bad_cap", [2.0, 3.0, 45.0, float("inf")])
    def test_child_does_not_inherit_an_unusable_max_segment_sec(
        self, client, db_session, SessionFactory, monkeypatch, bad_cap
    ):
        """Same rule as the run→project copy: a bad stored cap is not forwarded.

        The insert child is analyzed the moment the crop lands, so inheriting an
        un-generatable cap would hand it a failed analysis it had no part in. The
        extraction still succeeds (the crop request is perfectly valid and never
        mentioned max_segment_sec) — the child just falls back to NULL, i.e. the
        universal default.
        """
        self._stub_crop_clip(monkeypatch)
        parent = self._analyzed_parent(db_session, max_segment_sec=bad_cap)

        child_id = self._extract(client, parent.id)

        session = SessionFactory()
        child = session.get(VideoProject, child_id)
        session.close()
        assert child.max_segment_sec is None

        # And the parent row is untouched by the copy.
        db_session.expire_all()
        assert db_session.get(VideoProject, parent.id).max_segment_sec == bad_cap

    def test_child_inherits_mute_source(
        self, client, db_session, SessionFactory, monkeypatch
    ):
        """Same soundtrack as the parent → same "remove original audio" choice."""
        self._stub_crop_clip(monkeypatch)
        parent = self._analyzed_parent(db_session, mute_source=True)

        child_id = self._extract(client, parent.id)

        session = SessionFactory()
        child = session.get(VideoProject, child_id)
        session.close()
        assert child.mute_source is True


# ---------------------------------------------------------------------------
# Localisation — hook_sec, transcripts, translated prompts
# (docs/localisation.md §4.4)
# ---------------------------------------------------------------------------


def _line(line_id, start, end, text, speaker="the woman in the red jacket", **extra):
    """One docs/localisation.md §4.1 transcript line."""
    line = {
        "id": line_id,
        "start": start,
        "end": end,
        "speaker": speaker,
        "on_screen": True,
        "text": text,
    }
    line.update(extra)
    return line


def _transcript(lines, *, source_language="en", **extra):
    """A §4.1 transcript envelope around *lines*."""
    doc = {
        "schema_version": 1,
        "model": "gemini-2.5-pro",
        "prompt_version": "transcribe/v1",
        "created_at": "2026-08-09T12:00:00+00:00",
        "source_language": source_language,
        "lines": lines,
        "on_screen_text": [],
    }
    doc.update(extra)
    return doc


# Two short lines inside a 10s hook. Lengths are chosen so the crude
# chars-per-second estimate in api_v2._localisation_warnings finds them
# plausible in a Latin-script target — a warning here would be noise, and the
# warning tests below make their own over/under-long lines on purpose.
_HOOK_LINES = [
    _line(1, 0.0, 2.5, "Hey, what are you doing?"),
    _line(2, 2.5, 6.0, "I'm just listening to my emails.", speaker="the man"),
]


def _localisation_project(session, **kwargs):
    """A ready localisation project whose transcript is ready to translate."""
    defaults = dict(
        status=ProjectStatus.ready,
        project_type="localisation",
        source_local_path="/tmp/source.mp4",
        transcript=_transcript(_HOOK_LINES),
        transcript_status="ready",
    )
    defaults.update(kwargs)
    return _make_project(session, **defaults)


@pytest.fixture()
def fake_translate(monkeypatch):
    """Replace the kie.ai translation call — no live network in tests.

    Honours the real contract of localisation.translate_lines (same ids, same
    order, ``text`` replaced, ``source_text`` carrying the original) so the
    endpoint is exercised exactly as it would be in production. Returns the
    list of calls it received. Accepts the optional video_summary /
    scene_context kwargs so a real caller that forwards cached context does
    not TypeError the stub.
    """
    calls: list[dict] = []

    def _translate(
        lines,
        *,
        source_language,
        target_language,
        model=None,
        video_summary="",
        scene_context="",
        context_lines=None,
        on_screen_text=None,
        frames=None,
        cuts=None,
        hook_end=None,
        spend_scope=None,
        retry_budget_sec=None,
        shorten_admit_sec=None,
        shorten_timeout_sec=None,
    ):
        calls.append(
            {
                "lines": lines,
                "source_language": source_language,
                "target_language": target_language,
                "video_summary": video_summary,
                "scene_context": scene_context,
                "context_lines": context_lines,
                "on_screen_text": on_screen_text,
                "frames": frames,
                "cuts": cuts,
                "hook_end": hook_end,
                "retry_budget_sec": retry_budget_sec,
                "shorten_admit_sec": shorten_admit_sec,
                "shorten_timeout_sec": shorten_timeout_sec,
            }
        )
        out = []
        for line in lines:
            translated = dict(line)
            translated["source_text"] = line["text"]
            translated["text"] = f"<{target_language}> {line['text']}"
            out.append(translated)
        return out

    monkeypatch.setattr(api_v2_module.localisation, "translate_lines", _translate)
    return calls


class TestProjectHookSec:
    """The analyze-time hook length (docs/localisation.md §3.1, §5).

    Same contract as max_segment_sec: NULL means "no explicit choice" (analysis
    falls back to settings.LOCALISATION_DEFAULT_HOOK_SEC) and the value is
    consumed at ANALYZE time, so changing it re-cuts nothing. Unlike
    max_segment_sec it has no upper bound — a hook longer than the video is
    clamped at analyze time, not rejected — but 0 and NaN are refused, because
    neither can ever produce a segment.
    """

    DEFAULT = settings.LOCALISATION_DEFAULT_HOOK_SEC

    # -- POST /api/v2/projects -------------------------------------------

    def test_omitted_is_stored_as_null(self, spy_client, SessionFactory):
        """NULL, not the resolved default — the default stays in one place."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec is None

    def test_localisation_is_marked_pending_from_creation(
        self, spy_client, SessionFactory
    ):
        """NULL must mean exactly "nobody asked for a transcript".

        A localisation project is transcribed automatically once analysis
        returns, so leaving transcript_status NULL for the whole analyze window
        made NULL mean two things at once — and the panel, reading it, offered
        a Transcribe button next to a job that was already promised. One click
        there paid for a second full model call on the same video.
        """
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.transcript_status == "pending"
        assert project.transcript is None

    def test_other_project_types_are_not_marked_pending(
        self, spy_client, SessionFactory
    ):
        """Nothing transcribes a face-swap project, so a "pending" there would
        poll a panel that is never rendered and never resolves."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "face_swap"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.transcript_status is None

    def test_create_with_explicit_hook(self, spy_client, SessionFactory):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": 12.5},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec == 12.5

    def test_blank_form_value_is_stored_as_null(self, spy_client, SessionFactory):
        """The create form posts its input verbatim: "" must mean unset, not 0."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": ""},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec is None

    def test_create_gdrive_with_hook(self, spy_client, SessionFactory):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={
                "gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view",
                "project_type": "localisation",
                "hook_sec": 8,
            },
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec == 8.0

    def test_zero_is_a_value_and_is_rejected(self, client, SessionFactory):
        """0.0 is NOT "unset": a zero-length hook has no speech and no segment.

        This is the whole reason blank/absent maps to None rather than 0.0 —
        the two have to stay distinguishable, and only one of them is legal.
        """
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": 0},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "hook_sec" in response.json()["detail"]

    @pytest.mark.parametrize("value", [-1, -0.5, "nan", "inf", "-inf"])
    def test_create_out_of_range_is_400(self, client, value):
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": value},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400, response.text
        assert "hook_sec" in response.json()["detail"]

    def test_a_hook_longer_than_any_segment_cap_is_allowed(
        self, spy_client, SessionFactory
    ):
        """No ceiling: analysis clamps a too-long hook, it is not a bad request."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": 600},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec == 600.0

    # -- GET / PATCH /api/v2/projects/{pid} -------------------------------

    def test_get_exposes_the_field(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=7.0)
        body = client.get(f"/api/v2/projects/{project.id}").json()
        assert body["hook_sec"] == 7.0

    def test_patch_sets_and_persists(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"hook_sec": 12})
        assert resp.status_code == 200, resp.text
        assert resp.json()["hook_sec"] == 12.0
        assert client.get(f"/api/v2/projects/{project.id}").json()["hook_sec"] == 12.0

    def test_patch_explicit_null_clears_it(self, client, db_session):
        """`{"hook_sec": null}` resets to the settings default.

        NULL is a value here, not "omitted" — update_project tells the two apart
        via model_fields_set, exactly as it does for max_segment_sec.
        """
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"hook_sec": None})
        assert resp.status_code == 200, resp.text
        assert resp.json()["hook_sec"] is None

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).hook_sec is None

    def test_patch_without_the_key_leaves_it_untouched(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "renamed"})
        assert resp.status_code == 200
        assert resp.json()["hook_sec"] == 12.0

    @pytest.mark.parametrize("value", [0, -3])
    def test_patch_out_of_range_is_400_and_stores_nothing(
        self, client, db_session, value
    ):
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"hook_sec": value}
        )
        assert resp.status_code == 400, resp.text
        assert "hook_sec" in resp.json()["detail"]

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).hook_sec == 12.0

    @pytest.mark.parametrize(
        "body",
        [
            b'{"hook_sec": NaN}',
            b'{"hook_sec": "nan"}',
            b'{"hook_sec": Infinity}',
            b'{"hook_sec": "inf"}',
            b'{"hook_sec": -Infinity}',
        ],
    )
    def test_patch_nan_and_infinity_are_400(self, client, db_session, body):
        """Same trap max_segment_sec fell into: Pydantic parses these to floats.

        A NaN hook survives every `<` comparison in slice_lines (so the hook
        window silently comes back empty) and poisons the analyze-time clamp.
        The chained-comparison guard rejects it; a `value <= 0 or value == inf`
        guard would not have.
        """
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(
            f"/api/v2/projects/{project.id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400, resp.text
        assert "hook_sec" in resp.json()["detail"]

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).hook_sec == 12.0

    def test_patch_does_not_reanalyze_or_recut_segments(self, spy_client, db_session):
        """hook_sec is consumed at ANALYZE time — a PATCH is inert on an
        already-analyzed project: no re-enqueue, no re-cut. Same contract as
        max_segment_sec, and the reason both docstrings say so.
        """
        client, spy = spy_client
        project = _make_project(
            db_session, status=ProjectStatus.ready, project_type="localisation",
            hook_sec=10.0,
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )
        before = [
            (sd.index, sd.start_sec, sd.end_sec, sd.action)
            for sd in sorted(project.segments, key=lambda s: s.index)
        ]

        resp = client.patch(f"/api/v2/projects/{project.id}", json={"hook_sec": 20})
        assert resp.status_code == 200
        assert resp.json()["hook_sec"] == 20.0

        assert spy["analyze_project"] == []

        db_session.expire_all()
        after_project = db_session.get(VideoProject, project.id)
        after = [
            (sd.index, sd.start_sec, sd.end_sec, sd.action)
            for sd in sorted(after_project.segments, key=lambda s: s.index)
        ]
        assert after == before


class TestLibraryClipLanguage:
    """LibraryClip.language — drives the New Run form's tail auto-pick.

    NULL stays legal (a language-agnostic clip, which every clip uploaded
    before this column existed is); an unknown code is refused exactly as
    create_run refuses one for Run.language.
    """

    def test_create_with_language(self, client, SessionFactory, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        response = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro JA", "language": "ja"},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["language"] == "ja"

        session = SessionFactory()
        clip = session.get(LibraryClip, response.json()["id"])
        session.close()
        assert clip.language == "ja"

    def test_create_without_language_is_null(self, client, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        response = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro"},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["language"] is None

    def test_create_normalizes_case_and_whitespace(self, client, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        response = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro JA", "language": "  JA "},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["language"] == "ja"

    def test_create_rejects_unknown_language_and_stores_nothing(
        self, client, SessionFactory, monkeypatch
    ):
        """Rejected before the upload is written — no orphaned file, no row."""
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        session = SessionFactory()
        before = len(session.query(LibraryClip).all())
        session.close()

        response = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro NL", "language": "nl"},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert response.status_code == 400
        assert "language" in response.json()["detail"]

        session = SessionFactory()
        assert len(session.query(LibraryClip).all()) == before
        session.close()

    def test_list_exposes_language(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path, language="pt")
        listing = client.get("/api/v2/library-clips").json()
        assert {c["id"]: c["language"] for c in listing}[clip.id] == "pt"

    def test_patch_sets_language(self, client, db_session, tmp_path):
        """The reason PATCH exists: tag the tails uploaded before the column did."""
        clip = _make_library_clip(db_session, tmp_path)
        assert clip.language is None

        resp = client.patch(f"/api/v2/library-clips/{clip.id}", json={"language": "de"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["language"] == "de"

        db_session.expire_all()
        assert db_session.get(LibraryClip, clip.id).language == "de"

    def test_patch_empty_language_clears_it(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path, language="ja")
        resp = client.patch(f"/api/v2/library-clips/{clip.id}", json={"language": ""})
        assert resp.status_code == 200, resp.text
        assert resp.json()["language"] is None

    def test_patch_rejects_unknown_language(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path, language="ja")
        resp = client.patch(f"/api/v2/library-clips/{clip.id}", json={"language": "nl"})
        assert resp.status_code == 400
        db_session.expire_all()
        assert db_session.get(LibraryClip, clip.id).language == "ja"

    def test_patch_name_alone_leaves_language(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path, language="ja")
        resp = client.patch(
            f"/api/v2/library-clips/{clip.id}", json={"name": "Outro JA v2"}
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Outro JA v2"
        assert resp.json()["language"] == "ja"

    def test_patch_unknown_clip_is_404(self, client):
        assert (
            client.patch("/api/v2/library-clips/nope", json={"language": "ja"}).status_code
            == 404
        )


class TestLibraryClipKind:
    """LibraryClip.kind (AICP-381): end_card vs insert, editable on /v2/library."""

    def _post(self, client, monkeypatch, **data):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        return client.post(
            "/api/v2/library-clips",
            data={"name": "CTA_X", **data},
            files={"file": ("cta.mp4", b"\x00\x01" * 32, "video/mp4")},
        )

    def test_create_defaults_to_insert(self, client, SessionFactory, monkeypatch):
        response = self._post(client, monkeypatch)
        assert response.status_code == 201, response.text
        assert response.json()["kind"] == "insert"
        session = SessionFactory()
        assert session.get(LibraryClip, response.json()["id"]).kind == "insert"
        session.close()

    def test_create_end_card(self, client, monkeypatch):
        response = self._post(client, monkeypatch, kind=" End_Card ")
        assert response.status_code == 201, response.text
        assert response.json()["kind"] == "end_card"

    def test_create_rejects_unknown_kind_and_stores_nothing(
        self, client, SessionFactory, monkeypatch
    ):
        session = SessionFactory()
        before = len(session.query(LibraryClip).all())
        session.close()
        response = self._post(client, monkeypatch, kind="outro")
        assert response.status_code == 400 and "kind" in response.json()["detail"]
        session = SessionFactory()
        assert len(session.query(LibraryClip).all()) == before
        session.close()

    def test_list_and_patch_kind(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path, language="en")
        assert clip.kind == "insert"
        listing = client.get("/api/v2/library-clips").json()
        assert {c["id"]: c["kind"] for c in listing}[clip.id] == "insert"
        resp = client.patch(f"/api/v2/library-clips/{clip.id}", json={"kind": "end_card"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["kind"] == "end_card" and resp.json()["language"] == "en"
        db_session.expire_all()
        assert db_session.get(LibraryClip, clip.id).kind == "end_card"
        # Omitted = untouched.
        resp = client.patch(f"/api/v2/library-clips/{clip.id}", json={"name": "CTA_English"})
        assert resp.status_code == 200 and resp.json()["kind"] == "end_card"

    def test_patch_rejects_unknown_or_blank_kind(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path)
        for bad in ("outro", "", "  "):
            resp = client.patch(f"/api/v2/library-clips/{clip.id}", json={"kind": bad})
            assert resp.status_code == 400, bad
        db_session.expire_all()
        assert db_session.get(LibraryClip, clip.id).kind == "insert"

    def test_library_page_shows_the_kind_select(self, client, db_session, tmp_path):
        clip = _make_library_clip(db_session, tmp_path)
        clip.kind = "end_card"
        db_session.commit()
        html = client.get("/v2/library").text
        assert f"saveClipKind(this, '{clip.id}')" in html
        assert 'id="lib-kind"' in html


class TestTranscribeProject:
    """POST /projects/{pid}/transcribe — 202, status=pending, enqueue."""

    def test_enqueues_and_marks_pending(self, spy_client, db_session):
        client, spy = spy_client
        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            source_local_path="/tmp/source.mp4",
            transcript_status="failed",
            transcript_error="kie.ai timed out",
        )

        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "pending"
        assert body["error"] is None

        assert spy["transcribe_project"] == [project.id]

        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id)
        assert stored.transcript_status == "pending"
        assert stored.transcript_error is None

    def test_rerun_keeps_the_previous_transcript_readable(self, spy_client, db_session):
        """"Распознать заново" must not blank the panel while the job runs."""
        client, _spy = spy_client
        project = _localisation_project(db_session)

        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"
        assert resp.json()["transcript"]["lines"][0]["text"] == _HOOK_LINES[0]["text"]

    def test_no_local_source_is_409(self, spy_client, db_session):
        """A Drive project before analysis has nothing on disk to transcribe."""
        client, spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None,
        )

        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 409, resp.text
        assert spy["transcribe_project"] == []

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).transcript_status is None

    @pytest.mark.parametrize(
        "project_status",
        [
            ProjectStatus.created,
            ProjectStatus.analyzing,
            ProjectStatus.ready,
            ProjectStatus.failed,
        ],
    )
    def test_409_detail_never_leaks_a_python_repr(
        self, spy_client, db_session, project_status
    ):
        """The panel puts this detail straight into a browser alert.

        `f"{project.status!r}"` renders `<ProjectStatus.ready: 'ready'>` — a
        Python repr shown to an operator. The bare value is what belongs there.
        """
        client, _spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None, status=project_status,
        )

        detail = client.post(
            f"/api/v2/projects/{project.id}/transcribe"
        ).json()["detail"]
        assert "ProjectStatus" not in detail
        assert "<" not in detail
        assert f"project status: {project_status.value}" in detail

    def test_409_before_analysis_says_to_wait(self, spy_client, db_session):
        client, _spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None, status=ProjectStatus.analyzing,
        )
        detail = client.post(
            f"/api/v2/projects/{project.id}/transcribe"
        ).json()["detail"]
        assert "wait" in detail

    @pytest.mark.parametrize(
        "project_status", [ProjectStatus.ready, ProjectStatus.failed]
    )
    def test_409_after_analysis_points_at_re_analysis_not_at_waiting(
        self, spy_client, db_session, project_status
    ):
        """Telling an operator to "wait for analysis" on a project that is
        already `ready` is advice they can never act on: analysis has finished
        and no file arrived. Re-analysis is what fetches it again."""
        client, _spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None, status=project_status,
        )
        detail = client.post(
            f"/api/v2/projects/{project.id}/transcribe"
        ).json()["detail"]
        assert "wait" not in detail
        assert f"/api/v2/projects/{project.id}/analyze" in detail

    def test_unknown_project_is_404(self, client):
        assert client.post("/api/v2/projects/nope/transcribe").status_code == 404

    def test_second_click_while_live_is_409(self, spy_client, db_session):
        """AICP-365 (operator decision 2026-10-31): one live transcription per
        project; the durable job record makes a stale "running" resolvable."""
        from app.localisation_jobs_models import ProjectTranscription

        client, spy = spy_client
        project = _make_project(
            db_session, status=ProjectStatus.ready, source_local_path="/tmp/source.mp4",
        )
        assert client.post(f"/api/v2/projects/{project.id}/transcribe").status_code == 202
        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 409, resp.text
        assert "already queued" in resp.json()["detail"]
        assert spy["transcribe_project"] == [project.id]
        db_session.expire_all()
        jobs = db_session.query(ProjectTranscription).filter_by(project_id=project.id).all()
        assert [(j.status, j.origin, j.dispatch_gen) for j in jobs] == [("queued", "manual", 1)]


class TestGetTranscript:
    def test_never_requested_is_all_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get(f"/api/v2/projects/{project.id}/transcript")
        assert resp.status_code == 200
        assert resp.json() == {"status": None, "error": None, "transcript": None}

    def test_returns_the_stored_transcript_verbatim(self, client, db_session):
        project = _localisation_project(db_session)
        body = client.get(f"/api/v2/projects/{project.id}/transcript").json()
        assert body["status"] == "ready"
        assert body["transcript"] == _transcript(_HOOK_LINES)

    def test_failed_surfaces_the_error(self, client, db_session):
        project = _make_project(
            db_session, transcript_status="failed", transcript_error="upload failed"
        )
        body = client.get(f"/api/v2/projects/{project.id}/transcript").json()
        assert body["status"] == "failed"
        assert body["error"] == "upload failed"

    def test_empty_is_a_status_not_an_error(self, client, db_session):
        """No speech is a legal outcome — status "empty", no error text."""
        project = _make_project(
            db_session,
            transcript_status="empty",
            transcript=_transcript([], source_language="en"),
        )
        body = client.get(f"/api/v2/projects/{project.id}/transcript").json()
        assert body["status"] == "empty"
        assert body["error"] is None
        assert body["transcript"]["lines"] == []

    def test_unknown_project_is_404(self, client):
        assert client.get("/api/v2/projects/nope/transcript").status_code == 404


class TestPatchTranscript:
    """Operator fix-ups. The transcript becomes a Seedance prompt, so a
    malformed edit is refused (400) rather than stored and discovered later."""

    def test_accepts_a_corrected_transcript(self, client, db_session):
        project = _make_project(
            db_session, transcript_status="failed", transcript_error="kie.ai 500"
        )
        fixed = _transcript(
            [_line(1, 0.0, 2.0, "Эй, что ты делаешь?", speaker="the woman")],
            source_language="ru",
        )

        resp = client.patch(f"/api/v2/projects/{project.id}/transcript", json=fixed)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ready"
        # A hand-fixed transcript clears the failure that prompted the edit.
        assert body["error"] is None
        assert body["transcript"]["lines"][0]["text"] == "Эй, что ты делаешь?"
        assert body["transcript"]["source_language"] == "ru"

        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id)
        assert stored.transcript_status == "ready"
        assert stored.transcript_error is None
        assert stored.transcript["lines"][0]["speaker"] == "the woman"

    def test_no_lines_is_stored_as_empty(self, client, db_session):
        project = _localisation_project(db_session)
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=_transcript([])
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "empty"

    def test_envelope_keys_survive_the_round_trip(self, client, db_session):
        """model/prompt_version/created_at record which model produced it —
        an operator edit must not strip them."""
        project = _localisation_project(db_session)
        doc = _transcript(_HOOK_LINES, extra_key="kept")
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["model"] == "gemini-2.5-pro"
        assert body["transcript"]["prompt_version"] == "transcribe/v1"
        assert body["transcript"]["created_at"] == "2026-08-09T12:00:00+00:00"
        assert body["transcript"]["extra_key"] == "kept"

    def test_video_context_fields_are_preserved_and_stripped(self, client, db_session):
        """UI Save merges onto the full §4.1 object — context must survive PATCH."""
        project = _localisation_project(db_session)
        doc = _transcript(
            _HOOK_LINES,
            video_summary="  A promo for Listenify.  ",
            scene_context="  Woman holds a phone; 'this' is the app.  ",
        )
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["video_summary"] == "A promo for Listenify."
        assert body["transcript"]["scene_context"] == (
            "Woman holds a phone; 'this' is the app."
        )

    def test_omitted_video_context_normalises_to_empty_strings(self, client, db_session):
        """v1 envelopes without the keys must still PATCH cleanly."""
        project = _localisation_project(db_session)
        doc = _transcript(_HOOK_LINES)
        assert "video_summary" not in doc
        assert "scene_context" not in doc
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["video_summary"] == ""
        assert body["transcript"]["scene_context"] == ""

    @pytest.mark.parametrize(
        "field, value",
        [
            ("video_summary", ["not", "a", "string"]),
            ("video_summary", {"nested": True}),
            ("video_summary", 12),
            ("video_summary", True),
            ("scene_context", ["nope"]),
            ("scene_context", {"x": 1}),
            ("scene_context", 3.14),
            ("scene_context", False),
        ],
    )
    def test_non_string_video_context_is_400(
        self, client, db_session, field, value
    ):
        project = _localisation_project(db_session)
        doc = _transcript(_HOOK_LINES, **{field: value})
        resp = client.patch(f"/api/v2/projects/{project.id}/transcript", json=doc)
        assert resp.status_code == 400, resp.text
        assert field in resp.json()["detail"]

    def test_line_order_is_preserved_not_resorted(self, client, db_session):
        """format_dialogue merges consecutive same-speaker lines, so re-ordering
        would silently change the prompt the operator just proof-read."""
        project = _localisation_project(db_session)
        doc = _transcript(
            [_line(1, 5.0, 6.0, "second half"), _line(2, 0.0, 1.0, "first half")]
        )
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert [ln["id"] for ln in body["transcript"]["lines"]] == [1, 2]

    def test_on_screen_defaults_to_false_when_omitted(self, client, db_session):
        project = _localisation_project(db_session)
        line = _line(1, 0.0, 1.0, "hi")
        line.pop("on_screen")
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=_transcript([line])
        ).json()
        assert body["transcript"]["lines"][0]["on_screen"] is False

    def test_on_screen_text_is_validated_and_kept(self, client, db_session):
        project = _localisation_project(db_session)
        doc = _transcript(
            _HOOK_LINES,
            on_screen_text=[{"start": 0.0, "end": 3.0, "text": "Listenify"}],
        )
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["on_screen_text"] == [
            {"start": 0.0, "end": 3.0, "text": "Listenify"}
        ]

    @pytest.mark.parametrize(
        "payload, expect_in_detail",
        [
            ([{"id": 1}], "lines"),                                  # not an object
            ({"source_language": "en"}, "lines"),                    # no lines key
            ({"lines": "nope"}, "lines"),                            # lines not a list
            ({"lines": ["nope"]}, "lines[0]"),                       # line not an object
            ({"lines": [{"start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": "1", "start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": 0, "start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": True, "start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": 1, "start": 2, "end": 1, "speaker": "a", "text": "t"}]}, "end"),
            ({"lines": [{"id": 1, "start": -1, "end": 1, "speaker": "a", "text": "t"}]}, "start"),
            ({"lines": [{"id": 1, "start": "x", "end": 1, "speaker": "a", "text": "t"}]}, "start"),
            ({"lines": [{"id": 1, "end": 1, "speaker": "a", "text": "t"}]}, "start"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "speaker": "a", "text": "  "}]}, "text"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "speaker": "a"}]}, "text"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "text": "t"}]}, "speaker"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "speaker": "", "text": "t"}]}, "speaker"),
            ({"lines": [], "on_screen_text": "nope"}, "on_screen_text"),
            ({"lines": [], "on_screen_text": [{"start": 3, "end": 1, "text": "x"}]}, "on_screen_text[0]"),
            ({"lines": [], "source_language": 5}, "source_language"),
        ],
    )
    def test_garbage_is_400(self, client, db_session, payload, expect_in_detail):
        project = _localisation_project(db_session)
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=payload
        )
        assert resp.status_code == 400, resp.text
        assert expect_in_detail in resp.json()["detail"]

        # …and the previous transcript is still the one on file.
        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id)
        assert stored.transcript == _transcript(_HOOK_LINES)
        assert stored.transcript_status == "ready"

    def test_duplicate_ids_are_400(self, client, db_session):
        """Translation rejoins on the id — a duplicate loses a line silently."""
        project = _localisation_project(db_session)
        doc = _transcript([_line(1, 0.0, 1.0, "a"), _line(1, 1.0, 2.0, "b")])
        resp = client.patch(f"/api/v2/projects/{project.id}/transcript", json=doc)
        assert resp.status_code == 400
        assert "duplicate" in resp.json()["detail"]

    def test_nan_timestamp_is_400(self, client, db_session):
        """NaN survives every `<` in slice_lines, so the line would vanish from
        the hook window with nothing to show for it."""
        project = _localisation_project(db_session)
        body = (
            b'{"lines": [{"id": 1, "start": NaN, "end": 1.0, '
            b'"speaker": "a", "text": "t"}]}'
        )
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400, resp.text
        assert "finite" in resp.json()["detail"]

    def test_accepts_a_transcript_on_a_never_transcribed_project(
        self, client, db_session
    ):
        """Pasting one in by hand is the documented fallback (§8)."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=_transcript(_HOOK_LINES)
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "ready"

    def test_unknown_project_is_404(self, client):
        resp = client.patch(
            "/api/v2/projects/nope/transcript", json=_transcript(_HOOK_LINES)
        )
        assert resp.status_code == 404


class TestLocalisationPrompt:
    """POST /projects/{pid}/localisation-prompt (docs/localisation.md §4.4).

    Every test stubs localisation.translate_lines — the real one is a kie.ai
    call, and nothing here may touch the network. build_prompt is NOT stubbed:
    it renders the real app.project_types templates, so a broken template is a
    failing test rather than a surprise in production.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def test_single_segment_hook_puts_everything_in_the_prompt(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        resp = self._post(client, project.id, language="es")
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["source_language"] == "en"
        assert body["target_language"] == "es"
        assert body["hook_sec"] == settings.LOCALISATION_DEFAULT_HOOK_SEC
        # The usual case: one swap segment, so no per-segment overrides at all.
        assert body["segment_prompts"] == {}
        # Both lines, in the timed AICP-335 dialogue shape, timecoded against
        # the single swap clip [0, 10).
        prompt = body["prompt"]
        assert "<es> Hey, what are you doing?" in prompt
        assert "<es> I'm just listening to my emails." in prompt
        # The man speaks longest, so he is primary and listed first; the
        # woman, secondary, gets the line count for this clip.
        assert "\nSpeakers:\n- MAN: the man. Male voice. Never a female voice.\n" in prompt
        assert (
            "- WOMAN: the woman in the red jacket. Female voice. Never a male "
            "voice. She says ONLY ONE line in this clip." in prompt
        )
        assert "spoken strictly in this order, at these times" in prompt
        assert "] WOMAN: <es> Hey, what are you doing?" in prompt
        assert "[6.0–10.0] Nobody speaks. Mouths stay closed." in prompt
        assert "**the woman in the red jacket:**" not in prompt
        assert body["warnings"] == []

    def test_returns_translated_lines_with_their_source(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        body = self._post(client, project.id, language="es").json()

        assert [ln["id"] for ln in body["lines"]] == [1, 2]
        assert body["lines"][0]["text"] == "<es> Hey, what are you doing?"
        assert body["lines"][0]["source_text"] == "Hey, what are you doing?"
        assert body["lines"][0]["speaker"] == "the woman in the red jacket"

    def test_passes_cached_video_context_to_translate(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                _HOOK_LINES,
                video_summary="Promo for a reading app.",
                scene_context="Woman points at her phone.",
            ),
        )
        resp = self._post(client, project.id, language="es")
        assert resp.status_code == 200, resp.text
        assert len(fake_translate) == 1
        assert fake_translate[0]["video_summary"] == "Promo for a reading app."
        assert fake_translate[0]["scene_context"] == "Woman points at her phone."

    def test_passes_full_transcript_and_hook_frames_to_translate(
        self, client, db_session, fake_translate, monkeypatch
    ):
        """translate/v3: the whole transcript, the burned-in text and stills of
        the hook (taken from the project source) all reach the translator."""
        grabbed = {}

        def _frames(src, timestamps, **_kw):
            grabbed["src"], grabbed["ts"] = src, list(timestamps)
            return [(t, b"jpeg") for t in timestamps]

        monkeypatch.setattr(api_v2_module.media_mod, "extract_jpeg_frames", _frames)
        transcript = _transcript(_HOOK_LINES)
        transcript["on_screen_text"] = [{"start": 0.0, "end": 2.0, "text": "SALE"}]
        project = _localisation_project(db_session, transcript=transcript)

        resp = self._post(client, project.id, language="es")
        assert resp.status_code == 200, resp.text
        call = fake_translate[0]
        assert call["context_lines"] == transcript["lines"]
        assert call["on_screen_text"] == [{"start": 0.0, "end": 2.0, "text": "SALE"}]
        assert grabbed["src"] == project.source_local_path
        assert grabbed["ts"] and all(t < 10.0 for t in grabbed["ts"])
        assert call["frames"] == [(t, b"jpeg") for t in grabbed["ts"]]

    def test_frame_grab_failure_still_translates(
        self, client, db_session, fake_translate, monkeypatch
    ):
        def _boom(*_a, **_kw):
            raise api_v2_module.media_mod.MediaError("ffmpeg died")

        monkeypatch.setattr(api_v2_module.media_mod, "extract_jpeg_frames", _boom)
        project = _localisation_project(db_session)

        resp = self._post(client, project.id, language="es")
        assert resp.status_code == 200, resp.text
        assert fake_translate[0]["frames"] == []

    def test_old_transcript_without_context_still_translates(
        self, client, db_session, fake_translate
    ):
        """v1 cached envelopes omit the fields — empty-string defaults apply."""
        project = _localisation_project(db_session)
        assert "video_summary" not in project.transcript
        assert "scene_context" not in project.transcript

        resp = self._post(client, project.id, language="ja")
        assert resp.status_code == 200, resp.text
        assert fake_translate[0]["video_summary"] == ""
        assert fake_translate[0]["scene_context"] == ""
        assert [ln["id"] for ln in resp.json()["lines"]] == [1, 2]

    def test_swap_character_picks_a_different_template(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        swap = self._post(client, project.id, swap_character="true").json()["prompt"]
        keep = self._post(client, project.id, swap_character="false").json()["prompt"]

        assert swap != keep
        # Both still carry the dialogue and the language pair.
        for prompt in (swap, keep):
            assert "<es> Hey, what are you doing?" in prompt
            assert "from english to spanish" in prompt
            assert "lip-sync the translated speech" in prompt.lower()
            assert "facial expressions" in prompt.lower()
            assert "original motion and lip movements" not in prompt.lower()
        assert "Replace the main person" in swap
        assert "Do not replace the character" in keep
        assert "reference image" not in keep.lower()

    def test_only_the_hook_window_is_translated(
        self, client, db_session, fake_translate
    ):
        """Lines past the hook belong to the tail, which is never regenerated."""
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                _HOOK_LINES + [_line(3, 20.0, 22.0, "This is in the tail.")]
            ),
        )

        body = self._post(client, project.id).json()
        assert [ln["id"] for ln in body["lines"]] == [1, 2]
        assert "This is in the tail." not in body["prompt"]
        # The translator was never asked to pay for the tail either.
        assert [ln["id"] for ln in fake_translate[0]["lines"]] == [1, 2]

    def test_hook_sec_override_widens_the_window(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                _HOOK_LINES + [_line(3, 20.0, 22.0, "This is in the tail.")]
            ),
        )

        body = self._post(client, project.id, hook_sec=25).json()
        assert body["hook_sec"] == 25.0
        assert [ln["id"] for ln in body["lines"]] == [1, 2, 3]

    def test_stored_hook_sec_is_the_default_window(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=3.0,
            transcript=_transcript(_HOOK_LINES),
        )
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 3.0
        # Line 2 starts at 2.5s, so it overlaps a 3s hook and comes along.
        assert [ln["id"] for ln in body["lines"]] == [1, 2]

    def test_unusable_stored_hook_sec_falls_back_to_the_default(
        self, client, db_session, fake_translate
    ):
        """A 0 written straight into the column (or by an older build) must not
        turn every translate call into a 422 the operator cannot act on."""
        project = _localisation_project(db_session, hook_sec=0.0)
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == settings.LOCALISATION_DEFAULT_HOOK_SEC

    def test_multi_segment_hook_splits_the_dialogue(
        self, client, db_session, fake_translate
    ):
        """A hook longer than the per-clip cap is cut into several swap
        segments, and each is generated as its OWN clip — so each gets only its
        own lines. Handing all of it to every segment would make each clip try
        to say the whole hook.
        """
        project = _localisation_project(db_session, hook_sec=20.0)
        first = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        second = _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=20.0
        )
        _make_segment_def(
            db_session, project.id, 2, start_sec=20.0, end_sec=30.0,
            action="keep", has_face=False,
        )
        project.transcript = _transcript(
            [
                _line(1, 0.0, 4.0, "Front half of the hook.", speaker="the woman"),
                _line(2, 12.0, 16.0, "Back half of the hook.", speaker="the man"),
            ]
        )
        db_session.commit()

        body = self._post(client, project.id).json()

        assert set(body["segment_prompts"]) == {first.id, second.id}
        assert "Front half" in body["segment_prompts"][first.id]
        assert "Back half" not in body["segment_prompts"][first.id]
        assert "Back half" in body["segment_prompts"][second.id]
        assert "Front half" not in body["segment_prompts"][second.id]
        # The run prompt still carries the whole hook: it is the fallback for
        # any segment not in the map.
        assert "Front half" in body["prompt"]
        assert "Back half" in body["prompt"]

    def test_keep_segments_never_get_a_prompt(
        self, client, db_session, fake_translate
    ):
        """Keep segments are never generated, so a prompt for one is dead text."""
        project = _localisation_project(db_session, hook_sec=20.0)
        swap_a = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        swap_b = _make_segment_def(db_session, project.id, 1, start_sec=10.0, end_sec=20.0)
        keep = _make_segment_def(
            db_session, project.id, 2, start_sec=20.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id).json()
        assert keep.id not in body["segment_prompts"]
        assert set(body["segment_prompts"]) == {swap_a.id, swap_b.id}

    def test_a_silent_hook_segment_still_gets_its_own_empty_prompt(
        self, client, db_session, fake_translate
    ):
        """A swap segment with no speech in it gets a prompt with an EMPTY
        dialogue block, not no prompt at all.

        Leaving it out of the map would make it fall back to the run prompt,
        i.e. the WHOLE hook's dialogue — and that clip would try to say all of
        it a second time. Silence is the correct instruction for a silent
        stretch of the hook.
        """
        project = _localisation_project(db_session, hook_sec=20.0)
        speaking = _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=10.0
        )
        silent = _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=20.0
        )

        body = self._post(client, project.id).json()
        assert silent.id in body["segment_prompts"]
        assert "<es> Hey, what are you doing?" in body["segment_prompts"][speaking.id]
        assert "<es> Hey, what are you doing?" not in body["segment_prompts"][silent.id]
        assert "**" not in body["segment_prompts"][silent.id]  # no dialogue at all

    def test_writes_nothing_to_the_database(
        self, client, db_session, SessionFactory, fake_translate
    ):
        """The whole point of the endpoint: it returns text, it does not record
        anything. The run the operator submits afterwards is the only record.
        """
        project = _localisation_project(db_session, hook_sec=8.0)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=8.0)
        before = {
            "transcript": json.dumps(project.transcript, sort_keys=True),
            "transcript_status": project.transcript_status,
            "transcript_error": project.transcript_error,
            "hook_sec": project.hook_sec,
            "name": project.name,
        }

        assert self._post(client, project.id, language="ja").status_code == 200

        session = SessionFactory()
        after_project = session.get(VideoProject, project.id)
        assert json.dumps(after_project.transcript, sort_keys=True) == before["transcript"]
        assert after_project.transcript_status == before["transcript_status"]
        assert after_project.transcript_error == before["transcript_error"]
        assert after_project.hook_sec == before["hook_sec"]
        assert after_project.name == before["name"]
        # No run, and no new segment, was created behind the operator's back.
        assert session.query(Run).filter(Run.project_id == project.id).count() == 0
        assert (
            session.query(SegmentDef).filter(SegmentDef.project_id == project.id).count()
            == 1
        )
        session.close()

    @pytest.mark.parametrize("status", [None, "pending", "running", "failed", "empty"])
    def test_transcript_not_ready_is_409(
        self, client, db_session, fake_translate, status
    ):
        project = _localisation_project(db_session, transcript_status=status)
        resp = self._post(client, project.id)
        assert resp.status_code == 409, resp.text
        assert "not ready" in resp.json()["detail"]
        assert fake_translate == []

    def test_no_speech_in_the_hook_is_422(self, client, db_session, fake_translate):
        """A wordless hook is a legal transcript, but there is nothing to say."""
        project = _localisation_project(
            db_session,
            transcript=_transcript([_line(1, 30.0, 32.0, "Only speech in the tail.")]),
        )
        resp = self._post(client, project.id)
        assert resp.status_code == 422, resp.text
        assert "No speech" in resp.json()["detail"]
        assert fake_translate == []

    def test_hook_sec_override_of_zero_is_400(self, client, db_session, fake_translate):
        project = _localisation_project(db_session)
        resp = self._post(client, project.id, hook_sec=0)
        assert resp.status_code == 400, resp.text
        assert "hook_sec" in resp.json()["detail"]
        assert fake_translate == []

    @pytest.mark.parametrize("language", ["nl", "klingon", "   "])
    def test_unknown_language_is_400(
        self, client, db_session, fake_translate, language
    ):
        """An unrecognised code is refused, never silently treated as English.

        Whitespace is in here deliberately: it is non-empty, so it reaches the
        handler (unlike "" — see below) and has to be caught there.
        """
        project = _localisation_project(db_session)
        resp = self._post(client, project.id, language=language)
        assert resp.status_code == 400, resp.text
        assert "language" in resp.json()["detail"]
        assert fake_translate == []

    @pytest.mark.parametrize("data", [{}, {"language": ""}])
    def test_absent_language_is_a_422_from_the_form(
        self, client, db_session, fake_translate, data
    ):
        """FastAPI's own validation — the field is required, not defaulted.

        An EMPTY form value counts as absent to FastAPI (it maps ``""`` to a
        missing required Form field), so a blank <select> is a 422 rather than
        the handler's 400. Both refuse it; pinning which one keeps the UI's
        error handling honest.
        """
        project = _localisation_project(db_session)
        resp = client.post(
            f"/api/v2/projects/{project.id}/localisation-prompt", data=data
        )
        assert resp.status_code == 422
        assert fake_translate == []

    def test_translation_failure_is_502(self, client, db_session, monkeypatch):
        """A model outage is not the operator's mistake — and §8 says the
        fallback is writing the prompt by hand, so nothing is stored."""
        def _boom(lines, **kwargs):
            raise api_v2_module.localisation.LocalisationError("kie.ai returned 401")

        monkeypatch.setattr(api_v2_module.localisation, "translate_lines", _boom)
        project = _localisation_project(db_session)

        resp = self._post(client, project.id)
        assert resp.status_code == 502, resp.text
        assert "kie.ai returned 401" in resp.json()["detail"]

    def test_unknown_project_is_404(self, client, fake_translate):
        resp = self._post(client, "nope")
        assert resp.status_code == 404

    # -- warnings ---------------------------------------------------------

    def test_warns_when_a_line_straddles_the_hook_boundary(
        self, client, db_session, fake_translate
    ):
        """slice_lines takes overlapping lines, so this one is translated whole
        — and its translation is far too long for the 2s left before the hook
        segment ends, so it is cut mid-sentence."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(
                1, 8.0, 14.0,
                "Half in, half out of it, and it keeps on going well past the cut.",
            )]),
        )
        body = self._post(client, project.id).json()
        straddle = [w for w in body["warnings"] if "cut mid-sentence" in w]
        assert len(straddle) == 1
        assert "10.0s hook boundary" in straddle[0]
        assert "does not fit the 2.0s before the cut" in straddle[0]

    def test_no_straddle_warning_when_the_translation_fits_before_the_cut(
        self, client, db_session, fake_translate
    ):
        """translate/v3 hands the translator the 2s window that is left; a
        translation that fits it is a complete shorter line, not a cut one."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 8.0, 14.0, "Half in.")]),
        )
        body = self._post(client, project.id).json()
        assert [w for w in body["warnings"] if "cut mid-sentence" in w] == []
        assert [w for w in body["warnings"] if "overrun" in w] == []
        assert fake_translate[0]["hook_end"] == 10.0

    def test_no_straddle_warning_when_a_line_ends_on_the_boundary(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 8.0, 10.0, "Ends right on it.")]),
        )
        body = self._post(client, project.id).json()
        assert [w for w in body["warnings"] if "cut mid-sentence" in w] == []

    def test_warns_when_the_translation_will_overrun(
        self, client, db_session, fake_translate
    ):
        """The failure that actually shipped: a literal EN→JA hook that took far
        longer to say than the shot allowed."""
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                [
                    _line(
                        1, 0.0, 1.0,
                        "This is a very long sentence for one single second of footage.",
                    )
                ]
            ),
        )
        body = self._post(client, project.id, language="ja").json()
        assert any("will overrun" in w for w in body["warnings"])

    def test_warns_when_the_translation_looks_too_short(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            transcript=_transcript([_line(1, 0.0, 9.0, "Hi.")]),
        )
        body = self._post(client, project.id).json()
        assert any("check nothing was dropped" in w for w in body["warnings"])

    def test_warns_when_source_and_target_match(
        self, client, db_session, fake_translate
    ):
        """Almost always a mis-picked dropdown — a whole generation spent
        reproducing the original."""
        project = _localisation_project(db_session)  # source_language="en"
        body = self._post(client, project.id, language="en").json()
        assert any("no-op" in w for w in body["warnings"])
        # Advisory only: the prompt still comes back.
        assert body["prompt"]

    def test_warns_when_the_hook_override_overruns_the_analyzed_segments(
        self, client, db_session, fake_translate
    ):
        """hook_sec is a per-call override that re-segments nothing, so it can
        ask for more dialogue than the generated clip is able to say — the
        "whole script in one clip" failure the per-segment split prevents."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 9.0, "Inside the analyzed hook."),
                    _line(2, 11.0, 18.0, "Past the end of it."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id, hook_sec=20.0).json()
        overrun = [w for w in body["warnings"] if "only cover" in w]
        assert len(overrun) == 1
        assert "20.0s" in overrun[0] and "10.0s" in overrun[0]
        # Advisory only — the prompt still comes back with the wider window.
        assert "Past the end of it" in body["prompt"]

    def test_warns_when_no_swap_segment_covers_the_hook(
        self, client, db_session, fake_translate
    ):
        """A project analyzed under another type (or with the hook flipped to
        keep) would receive a prompt no clip is ever handed."""
        project = _localisation_project(db_session)
        _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id).json()
        assert any("nothing will be generated" in w for w in body["warnings"])
        assert body["prompt"]

    def test_warnings_never_block_the_response(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 9.0, 12.0, "Hi.")], source_language="en"),
        )
        resp = self._post(client, project.id, language="en")
        assert resp.status_code == 200
        assert len(resp.json()["warnings"]) >= 2
        assert resp.json()["prompt"]


_REAL_LOCALISE_ADMIT = api_v2_module._localise_admit
_REAL_LOCALISE_STATE = api_v2_module._localise_state


class TestLocalisationPromptJob:
    """AICP-338: POST /localisation-prompt queues a job; the UI polls it.

    The job itself is the real localisation_prompt_job.run (run inline by the
    client fixture's _InlineLocaliseQueue); what is pinned here is the
    lifecycle around it — 202 + job id, dedupe, polling states, the stale-
    result refusal and the 503 admission guards."""

    def _url(self, pid):
        return f"/api/v2/projects/{pid}/localisation-prompt"

    def test_post_answers_202_with_a_job_id(self, client, db_session, fake_translate):
        project = _localisation_project(db_session)
        resp = client.post(self._url(project.id), data={"language": "es"})
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["job_id"].startswith("localise-prompt-")
        assert body["status"] == "queued"
        assert body["deduplicated"] is False

    def test_job_snapshot_carries_request_params_and_the_transcript(
        self, client, db_session, fake_translate, localise_queue
    ):
        """The seam a later optional request field (e.g. `model`) rides on."""
        localise_queue.hold = True
        project = _localisation_project(db_session)
        job_id = client.post(self._url(project.id), data={"language": "es"}).json()["job_id"]
        job = localise_queue.jobs[job_id]
        assert job["snapshot"]["request_params"] == {}
        assert job["meta"]["request_params"] == {}
        assert job["snapshot"]["transcript"] == project.transcript

    def test_poll_returns_the_old_synchronous_answer(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        job_id = client.post(self._url(project.id), data={"language": "es"}).json()["job_id"]
        poll = client.get(f"{self._url(project.id)}/jobs/{job_id}")
        assert poll.status_code == 200, poll.text
        body = poll.json()
        assert body["status"] == "done"
        assert body["language"] == "es"
        assert body["swap_character"] is False
        assert "<es> Hey, what are you doing?" in body["result"]["prompt"]
        assert body["error"] is None

    def test_identical_request_while_queued_joins_the_same_job(
        self, client, db_session, fake_translate, localise_queue
    ):
        """A second click (or a second tab) must not pay for a second call."""
        localise_queue.hold = True
        project = _localisation_project(db_session)
        first = client.post(self._url(project.id), data={"language": "es"}).json()
        second = client.post(self._url(project.id), data={"language": "es"}).json()
        assert second["job_id"] == first["job_id"]
        assert second["deduplicated"] is True
        assert localise_queue.reserves == 1

        poll = client.get(f"{self._url(project.id)}/jobs/{first['job_id']}").json()
        assert poll["status"] == "queued"
        assert fake_translate == []

    def test_different_language_or_mode_is_a_different_job(
        self, client, db_session, fake_translate, localise_queue
    ):
        localise_queue.hold = True
        project = _localisation_project(db_session)
        a = client.post(self._url(project.id), data={"language": "es"}).json()
        b = client.post(self._url(project.id), data={"language": "pt"}).json()
        c = client.post(
            self._url(project.id), data={"language": "es", "swap_character": "true"}
        ).json()
        assert len({a["job_id"], b["job_id"], c["job_id"]}) == 3
        assert localise_queue.reserves == 3

    def test_a_finished_job_does_not_block_a_fresh_click(
        self, client, db_session, fake_translate, localise_queue
    ):
        """Re-translating is meant to be repeatable: dedupe covers only a job
        that is still queued or running."""
        project = _localisation_project(db_session)
        a = client.post(self._url(project.id), data={"language": "es"}).json()
        b = client.post(self._url(project.id), data={"language": "es"}).json()
        assert a["job_id"] != b["job_id"]
        assert b["deduplicated"] is False
        assert len(fake_translate) == 2

    def test_job_translates_the_snapshot_and_refuses_a_stale_answer(
        self, client, db_session, fake_translate, localise_queue, SessionFactory
    ):
        """The project changed while the job waited: the job still translates
        what was asked for, but the poll will not hand that answer to a form
        whose segments/transcript no longer match it."""
        localise_queue.hold = True
        project = _localisation_project(db_session)
        job_id = client.post(self._url(project.id), data={"language": "es"}).json()["job_id"]

        session = SessionFactory()
        row = session.get(VideoProject, project.id)
        edited = copy.deepcopy(row.transcript)
        edited["lines"][0]["text"] = "Something else entirely."
        row.transcript = edited
        session.commit()
        session.close()

        localise_queue.run(job_id)
        assert fake_translate[0]["lines"][0]["text"] == "Hey, what are you doing?"
        poll = client.get(f"{self._url(project.id)}/jobs/{job_id}").json()
        assert poll["status"] == "failed"
        assert poll["error"]["status_code"] == 409
        assert "changed" in poll["error"]["detail"]
        assert poll["result"] is None

    def test_segment_edit_also_makes_the_answer_stale(
        self, client, db_session, fake_translate, localise_queue, SessionFactory
    ):
        localise_queue.hold = True
        project = _localisation_project(db_session)
        seg = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        job_id = client.post(self._url(project.id), data={"language": "es"}).json()["job_id"]

        session = SessionFactory()
        session.get(SegmentDef, seg.id).end_sec = 8.0
        session.commit()
        session.close()

        localise_queue.run(job_id)
        poll = client.get(f"{self._url(project.id)}/jobs/{job_id}").json()
        assert poll["status"] == "failed"
        assert poll["error"]["status_code"] == 409

    def test_model_failure_arrives_through_the_poll_as_502(
        self, client, db_session, monkeypatch
    ):
        def _boom(lines, **kwargs):
            raise api_v2_module.localisation.LocalisationError("kie.ai returned 401")

        monkeypatch.setattr(api_v2_module.localisation, "translate_lines", _boom)
        project = _localisation_project(db_session)
        resp = client.post(self._url(project.id), data={"language": "es"})
        assert resp.status_code == 202
        poll = client.get(f"{self._url(project.id)}/jobs/{resp.json()['job_id']}").json()
        assert poll["status"] == "failed"
        assert poll["error"] == {"status_code": 502, "detail": "kie.ai returned 401"}

    def test_another_projects_job_is_404(self, client, db_session, fake_translate):
        mine = _localisation_project(db_session)
        other = _localisation_project(db_session)
        job_id = client.post(self._url(mine.id), data={"language": "es"}).json()["job_id"]
        resp = client.get(f"{self._url(other.id)}/jobs/{job_id}")
        assert resp.status_code == 404

    def test_an_unknown_job_is_expired_and_a_foreign_id_404(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        gone = client.get(f"{self._url(project.id)}/jobs/localise-prompt-deadbeef")
        assert gone.status_code == 200
        assert gone.json()["status"] == "expired"
        assert client.get(f"{self._url(project.id)}/jobs/rq-something").status_code == 404

    def test_synchronous_refusals_never_enqueue(
        self, client, db_session, fake_translate, localise_queue
    ):
        project = _localisation_project(db_session, transcript_status="pending")
        assert client.post(self._url(project.id), data={"language": "es"}).status_code == 409
        assert client.post(self._url("nope"), data={"language": "es"}).status_code == 404
        assert localise_queue.jobs == {}
        assert localise_queue.reserves == 0

    def test_no_live_localise_worker_is_503_and_reserves_nothing(
        self, client, db_session, fake_translate, monkeypatch
    ):
        from app import localisation_prompt_job
        import app.tasks as tasks_module

        monkeypatch.setattr(api_v2_module, "_localise_admit", _REAL_LOCALISE_ADMIT)
        monkeypatch.setattr(tasks_module, "_get_redis", lambda: object())
        monkeypatch.setattr(localisation_prompt_job, "worker_alive", lambda conn: False)
        reserved = []
        monkeypatch.setattr(
            api_v2_module.spend_mod, "reserve",
            lambda *a, **k: reserved.append(a),
        )
        project = _localisation_project(db_session)
        resp = client.post(self._url(project.id), data={"language": "es"})
        assert resp.status_code == 503, resp.text
        assert "localise-worker" in resp.json()["detail"]
        assert reserved == []

    def test_redis_down_is_503_on_post_and_poll(
        self, client, db_session, fake_translate, monkeypatch
    ):
        from redis.exceptions import ConnectionError as RedisConnectionError

        from app import localisation_prompt_job
        import app.tasks as tasks_module

        def _down(*_a, **_k):
            raise RedisConnectionError("redis is down")

        monkeypatch.setattr(api_v2_module, "_localise_admit", _REAL_LOCALISE_ADMIT)
        monkeypatch.setattr(api_v2_module, "_localise_state", _REAL_LOCALISE_STATE)
        monkeypatch.setattr(tasks_module, "_get_redis", lambda: object())
        monkeypatch.setattr(localisation_prompt_job, "worker_alive", _down)
        monkeypatch.setattr(localisation_prompt_job, "job_state", _down)
        project = _localisation_project(db_session)
        assert client.post(self._url(project.id), data={"language": "es"}).status_code == 503
        poll = client.get(f"{self._url(project.id)}/jobs/localise-prompt-x")
        assert poll.status_code == 503


class TestLocalisationSegmentAssignment:
    """Every hook line belongs to EXACTLY ONE swap segment.

    A per-segment prompt REPLACES the run prompt (pipeline_v2 submits
    ``prompt_override if prompt_override else run_prompt``), so the run prompt
    is NOT a fallback for a segment that got one. That makes the split a
    partition problem, and localisation.slice_lines — whose semantics are
    OVERLAP — the wrong tool for it: overlap duplicates a boundary-straddling
    line into both clips and loses a line that falls in a keep gap. The rule is
    the line's MIDPOINT.

    Both shapes below are reachable by hand-splitting the hook in the Segment
    Editor, which is exactly what the new hook editor invites.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def _split_hook_project(self, session):
        """swap[0,5] + swap[5,10], with a line straddling the 5s seam."""
        project = _localisation_project(
            session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.8, "Before the seam.", speaker="the woman"),
                    _line(2, 4.8, 5.4, "Right on the seam.", speaker="the woman"),
                    _line(3, 5.4, 9.5, "After the seam.", speaker="the man"),
                ]
            ),
        )
        first = _make_segment_def(session, project.id, 0, start_sec=0.0, end_sec=5.0)
        second = _make_segment_def(session, project.id, 1, start_sec=5.0, end_sec=10.0)
        return project, first, second

    def test_a_straddling_line_lands_in_exactly_one_segment(
        self, client, db_session, fake_translate
    ):
        """The duplication bug: with overlap semantics the 4.8-5.4s line was in
        BOTH overrides, both clips generated it, and the stitched hook said the
        sentence twice."""
        project, first, second = self._split_hook_project(db_session)

        prompts = self._post(client, project.id).json()["segment_prompts"]
        assert set(prompts) == {first.id, second.id}

        occurrences = sum(
            "Right on the seam." in prompts[sid] for sid in (first.id, second.id)
        )
        assert occurrences == 1
        # Midpoint 5.1s falls in [5, 10), so the SECOND clip speaks it.
        assert "Right on the seam." in prompts[second.id]
        assert "Right on the seam." not in prompts[first.id]

    def test_every_line_is_assigned_and_none_are_shared(
        self, client, db_session, fake_translate
    ):
        project, first, second = self._split_hook_project(db_session)
        prompts = self._post(client, project.id).json()["segment_prompts"]

        assert "Before the seam." in prompts[first.id]
        assert "Before the seam." not in prompts[second.id]
        assert "After the seam." in prompts[second.id]
        assert "After the seam." not in prompts[first.id]

    def test_a_straddling_line_is_still_warned_about(
        self, client, db_session, fake_translate
    ):
        """Assigned whole to one clip, but the stitch still cuts through it —
        the same advisory the hook edge has always had."""
        project, _first, _second = self._split_hook_project(db_session)
        warnings = self._post(client, project.id).json()["warnings"]

        seam = [w for w in warnings if "5.0s boundary between two swap segments" in w]
        assert len(seam) == 1
        assert "Line 2" in seam[0]
        assert "cut mid-sentence" in seam[0]

    def test_a_line_ending_on_a_seam_is_not_warned_about(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 2.0, 5.0, "Ends right on the seam.")]),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=5.0)
        _make_segment_def(db_session, project.id, 1, start_sec=5.0, end_sec=10.0)

        warnings = self._post(client, project.id).json()["warnings"]
        assert [w for w in warnings if "cut mid-sentence" in w] == []

    # -- the keep gap ------------------------------------------------------

    def _gapped_hook_project(self, session):
        """swap[0,3] + keep[3,5] + swap[5,10] — the hand-split shape."""
        project = _localisation_project(
            session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 2.5, "In the first swap.", speaker="the woman"),
                    _line(2, 3.2, 4.6, "Stranded in the gap.", speaker="the woman"),
                    _line(3, 5.5, 9.0, "In the second swap.", speaker="the man"),
                ]
            ),
        )
        first = _make_segment_def(session, project.id, 0, start_sec=0.0, end_sec=3.0)
        _make_segment_def(
            session, project.id, 1, start_sec=3.0, end_sec=5.0,
            action="keep", has_face=False,
        )
        second = _make_segment_def(session, project.id, 2, start_sec=5.0, end_sec=10.0)
        return project, first, second

    def test_a_line_in_a_keep_gap_reaches_no_segment_prompt(
        self, client, db_session, fake_translate
    ):
        project, first, second = self._gapped_hook_project(db_session)
        body = self._post(client, project.id).json()

        prompts = body["segment_prompts"]
        assert set(prompts) == {first.id, second.id}
        assert "Stranded in the gap." not in prompts[first.id]
        assert "Stranded in the gap." not in prompts[second.id]
        # It stays in the run prompt so the operator can see what is at stake —
        # they can fix the segmentation, but not text they were never shown.
        assert "Stranded in the gap." in body["prompt"]

    def test_a_line_in_a_keep_gap_is_warned_about_by_name(
        self, client, db_session, fake_translate
    ):
        """The silent failure: nothing generates this line, and the first sign
        used to be a delivered hook missing a sentence."""
        project, _first, _second = self._gapped_hook_project(db_session)
        warnings = self._post(client, project.id).json()["warnings"]

        stranded = [w for w in warnings if "will not be spoken at all" in w]
        assert len(stranded) == 1
        assert "Line 2" in stranded[0]
        assert "3.2s to 4.6s" in stranded[0]

    def test_the_coverage_warning_measures_the_union_not_the_last_end(
        self, client, db_session, fake_translate
    ):
        """max(end_sec) reads 10s here — the hole in the middle was invisible."""
        project, _first, _second = self._gapped_hook_project(db_session)
        warnings = self._post(client, project.id).json()["warnings"]

        coverage = [w for w in warnings if "only cover" in w]
        assert len(coverage) == 1
        assert "10.0s" in coverage[0]      # the hook window
        assert "only cover 8.0s" in coverage[0]  # 3 + 5, not 10
        assert "3.0-5.0s" in coverage[0]   # and it names the gap

    def test_a_gap_free_hook_raises_no_coverage_warning(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session, hook_sec=10.0)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=5.0)
        _make_segment_def(db_session, project.id, 1, start_sec=5.0, end_sec=10.0)

        warnings = self._post(client, project.id).json()["warnings"]
        assert [w for w in warnings if "only cover" in w] == []
        assert [w for w in warnings if "will not be spoken at all" in w] == []

    def test_a_single_swap_segment_never_strands_a_line(
        self, client, db_session, fake_translate
    ):
        """With one hook segment there are no per-segment prompts at all, so the
        run prompt IS what that clip receives — nothing is lost, and claiming
        otherwise would be a false alarm. The coverage warning still fires."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 0.5, 1.5, "In the uncovered head.")]),
        )
        _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=2.0,
            action="keep", has_face=False,
        )
        _make_segment_def(db_session, project.id, 1, start_sec=2.0, end_sec=10.0)

        body = self._post(client, project.id).json()
        assert body["segment_prompts"] == {}
        assert [w for w in body["warnings"] if "will not be spoken at all" in w] == []
        assert any("only cover" in w for w in body["warnings"])


class TestLocalisationHookWindow:
    """Three numbers call themselves "the hook length"; only one is the window.

    1. the per-call ``hook_sec`` form argument — an override for this call only,
       never persisted, re-segments nothing;
    2. the ANALYZED reality — where the project's swap segments actually end.
       This is the DEFAULT window, because it is what a run will really
       generate;
    3. the STORED ``VideoProject.hook_sec`` — the operator's intent. It bounds
       (2), it drives the next analysis, and it is the fallback when there is
       nothing analyzed to measure.

    The bug this pins: a project created with Hook length 15 under the default
    10s segmentation cap is cut ``swap[0,10]`` while the column still says 15,
    so the endpoint used to slice 15s of transcript into a prompt for a 10s
    clip.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def test_the_analyzed_window_beats_the_stored_intent(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=15.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Inside the clip that will exist."),
                    _line(2, 11.0, 14.0, "Only inside the 15s the operator asked for."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 10.0
        assert [ln["id"] for ln in body["lines"]] == [1]
        assert "Only inside the 15s" not in body["prompt"]
        # And the clamped window is fully covered, so no coverage alarm.
        assert [w for w in body["warnings"] if "only cover" in w] == []

    def test_the_stored_intent_is_left_alone(
        self, client, db_session, SessionFactory, fake_translate
    ):
        """The column is the operator's INTENT and still drives the next
        analysis — raise the segmentation cap, re-analyze, and 15 finally
        happens. Reading it as reality is what was wrong, not storing it."""
        project = _localisation_project(db_session, hook_sec=15.0)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        assert self._post(client, project.id).json()["hook_sec"] == 10.0

        session = SessionFactory()
        assert session.get(VideoProject, project.id).hook_sec == 15.0
        session.close()

    def test_the_intent_bounds_the_analyzed_window(
        self, client, db_session, fake_translate
    ):
        """A project analyzed under another type can have swap segments running
        far past the hook; those are not hook, and must not widen the window."""
        project = _localisation_project(
            db_session,
            hook_sec=5.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Inside the hook."),
                    _line(2, 20.0, 24.0, "Deep in the tail."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=30.0)

        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 5.0
        assert [ln["id"] for ln in body["lines"]] == [1]

    def test_an_unanalyzed_project_falls_back_to_the_intent(
        self, client, db_session, fake_translate
    ):
        """Nothing to measure — the stored value is the only number there is."""
        project = _localisation_project(db_session, hook_sec=4.0)
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 4.0

    def test_a_hook_flipped_to_keep_falls_back_to_the_intent(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session, hook_sec=6.0)
        _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=30.0,
            action="keep", has_face=False,
        )
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 6.0
        assert any("nothing will be generated" in w for w in body["warnings"])

    def test_a_hand_split_hook_keeps_its_full_analyzed_width(
        self, client, db_session, fake_translate
    ):
        """swap[0,3] + keep[3,5] + swap[5,10]: the analyzed window is 10s, not
        3s. Shrinking to the first gap would orphan the second swap segment —
        it would drop out of the hook, get no per-segment prompt, and fall back
        to re-speaking the whole hook."""
        project = _localisation_project(db_session, hook_sec=10.0)
        first = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=3.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=3.0, end_sec=5.0,
            action="keep", has_face=False,
        )
        second = _make_segment_def(db_session, project.id, 2, start_sec=5.0, end_sec=10.0)

        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 10.0
        assert set(body["segment_prompts"]) == {first.id, second.id}

    def test_an_explicit_override_still_beats_the_analyzed_window(
        self, client, db_session, fake_translate
    ):
        """The override is for retranslating against a different window on
        purpose, so it outranks reality — and keeps its warning for doing so."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Inside the analyzed hook."),
                    _line(2, 11.0, 14.0, "Past the end of it."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        body = self._post(client, project.id, hook_sec=15).json()
        assert body["hook_sec"] == 15.0
        assert [ln["id"] for ln in body["lines"]] == [1, 2]
        coverage = [w for w in body["warnings"] if "only cover" in w]
        assert len(coverage) == 1
        assert "10.0-15.0s" in coverage[0]

    def test_an_override_narrower_than_the_analyzed_window_wins_too(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 2.0, "Inside the narrowed window."),
                    _line(2, 6.0, 9.0, "Outside it."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        body = self._post(client, project.id, hook_sec=4).json()
        assert body["hook_sec"] == 4.0
        assert [ln["id"] for ln in body["lines"]] == [1]


class TestLocalisationReleaseIntegration:
    """Combined release checks for video-context + Fast default + adaptive lips.

    Exercises the three reviewed Localisation features together through real
    application helpers / endpoints. Translation stays stubbed (fake_translate);
    Seedance templates are the live project_types strings.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def test_v2_context_translates_into_adaptive_lips_prompt(
        self, client, db_session, fake_translate
    ):
        """§D.1 — schema-v2 transcript context → adaptive-lips Seedance prompt."""
        project = _localisation_project(
            db_session,
            mute_source=True,
            transcript=_transcript(
                _HOOK_LINES,
                schema_version=2,
                prompt_version="transcribe/v2",
                video_summary="Promo for a reading app.",
                scene_context="Woman points at her phone; 'this' means the app.",
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        for swap_character in ("true", "false"):
            body = self._post(
                client, project.id, language="ja", swap_character=swap_character
            ).json()
            prompt = body["prompt"]
            lowered = prompt.lower()
            assert "{source_language}" not in prompt
            assert "{target_language}" not in prompt
            assert "{dialogue}" not in prompt
            assert "lip-sync the translated speech" in lowered
            assert "facial expressions" in lowered
            assert "emotional reactions" in lowered
            assert "original motion and lip movements" not in lowered
            assert "<ja> Hey, what are you doing?" in prompt

        assert fake_translate[-1]["video_summary"] == "Promo for a reading app."
        assert fake_translate[-1]["scene_context"] == (
            "Woman points at her phone; 'this' means the app."
        )

    def test_new_run_defaults_to_seedance_fast(self, db_session):
        """§D.2 — registry-driven New Run default under normal project settings."""
        from app.project_types import LOCALISATION, spec_for
        from app.web_v2 import _new_run_defaults

        assert spec_for(LOCALISATION).default_model == "seedance-fast"
        assert spec_for(LOCALISATION).default_audio_mode == "seedance"
        assert spec_for(LOCALISATION).default_mute_source is True

        project = _localisation_project(db_session, mute_source=True)
        ctx = _new_run_defaults(project)
        assert ctx["default_model"] == "seedance-fast"
        assert ctx["default_audio_mode"] == "seedance"
        assert "seedance-2-5" not in ctx["models_without_audio"]
        assert "gemini-omni" in ctx["models_without_audio"]

    def test_multi_segment_context_assigns_dialogue_once_with_adaptive_lips(
        self, client, db_session, fake_translate
    ):
        """§D.3 — multi-segment partition + adaptive lips + forwarded context."""
        project = _localisation_project(
            db_session,
            hook_sec=20.0,
            mute_source=True,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Front half of the hook.", speaker="the woman"),
                    _line(2, 12.0, 16.0, "Back half of the hook.", speaker="the man"),
                ],
                schema_version=2,
                prompt_version="transcribe/v2",
                video_summary="Two-shot street promo.",
                scene_context="Woman then man; each owns their half.",
            ),
        )
        first = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        second = _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=20.0
        )

        body = self._post(client, project.id, swap_character="true").json()
        assert fake_translate[0]["video_summary"] == "Two-shot street promo."
        assert fake_translate[0]["scene_context"] == (
            "Woman then man; each owns their half."
        )
        assert set(body["segment_prompts"]) == {first.id, second.id}
        front = body["segment_prompts"][first.id]
        back = body["segment_prompts"][second.id]
        assert "Front half" in front and "Back half" not in front
        assert "Back half" in back and "Front half" not in back
        for prompt in (front, back, body["prompt"]):
            lowered = prompt.lower()
            assert "lip-sync the translated speech" in lowered
            assert "original motion and lip movements" not in lowered
            assert "{dialogue}" not in prompt

    def test_v1_transcript_builds_adaptive_prompt(
        self, client, db_session, fake_translate
    ):
        """§D.4 — cached v1 envelopes still produce the adaptive-lips templates."""
        project = _localisation_project(db_session)
        assert "video_summary" not in project.transcript
        assert "scene_context" not in project.transcript
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        body = self._post(client, project.id, swap_character="false").json()
        assert fake_translate[0]["video_summary"] == ""
        assert fake_translate[0]["scene_context"] == ""
        prompt = body["prompt"]
        lowered = prompt.lower()
        assert "do not replace the character" in lowered
        assert "lip-sync the translated speech" in lowered
        assert "original motion and lip movements" not in lowered
        assert "{source_language}" not in prompt


# ---------------------------------------------------------------------------
# Voice library — /api/v2/voice-clips + attaching a voice to a run
# ---------------------------------------------------------------------------


def _make_voice_clip(session, tmp_path=None, **kwargs) -> VoiceClip:
    local = kwargs.pop("local_path", None)
    if local is None:
        local = str(tmp_path / "voice.mp3") if tmp_path else "/nonexistent/voice.mp3"
        if tmp_path:
            (tmp_path / "voice.mp3").write_bytes(b"\x00" * 32)
    defaults = dict(name="Marissa EN", local_path=local, duration_sec=8.0)
    defaults.update(kwargs)
    voice = VoiceClip(**defaults)
    session.add(voice)
    session.commit()
    return voice


class TestVoiceClips:
    """The voice library itself: upload → list → preview → delete."""

    def test_upload_list_delete_roundtrip(self, client, SessionFactory, monkeypatch):
        # Stub the transcode: this test is about the endpoint, not ffmpeg
        # (make_voice_reference has its own coverage in test_media.py).
        def fake_normalise(src, dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"ID3fake")
            return 7.5

        monkeypatch.setattr(
            api_v2_module.media_mod, "make_voice_reference", fake_normalise
        )
        response = client.post(
            "/api/v2/voice-clips",
            data={"name": "Marissa JA", "language": "ja"},
            files={"file": ("marissa.mp3", b"\x00\x01" * 32, "audio/mpeg")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "Marissa JA"
        assert body["language"] == "ja"
        # The stored duration is the NORMALISED one, not the upload's.
        assert body["duration_sec"] == pytest.approx(7.5)
        voice_id = body["id"]

        session = SessionFactory()
        voice = session.get(VoiceClip, voice_id)
        session.close()
        assert voice is not None
        assert os.path.exists(voice.local_path)
        # Only the normalised mp3 is kept — the original upload is dropped.
        assert not os.path.exists(
            os.path.join(os.path.dirname(voice.local_path), "upload.src")
        )

        listing = client.get("/api/v2/voice-clips")
        assert listing.status_code == 200
        assert voice_id in [v["id"] for v in listing.json()]

        file_resp = client.get(f"/api/v2/voice-clips/{voice_id}/file")
        assert file_resp.status_code == 200

        del_resp = client.delete(f"/api/v2/voice-clips/{voice_id}")
        assert del_resp.status_code == 204
        session = SessionFactory()
        assert session.get(VoiceClip, voice_id) is None
        session.close()
        assert not os.path.exists(voice.local_path)

    def test_unusable_audio_is_400_and_cleans_up(self, client, monkeypatch):
        """A source with no audio (or too short) must not orphan a directory."""
        def boom(src, dst):
            raise api_v2_module.media_mod.MediaError("has no audio track")

        monkeypatch.setattr(api_v2_module.media_mod, "make_voice_reference", boom)
        response = client.post(
            "/api/v2/voice-clips",
            files={"file": ("silent.mp4", b"\x00" * 16, "video/mp4")},
        )
        assert response.status_code == 400
        assert "Unusable voice reference" in response.json()["detail"]
        assert client.get("/api/v2/voice-clips").json() == []

    def test_requires_exactly_one_source(self, client):
        both = client.post(
            "/api/v2/voice-clips",
            data={"gdrive_url": "https://drive.google.com/file/d/x/view"},
            files={"file": ("v.mp3", b"\x00" * 8, "audio/mpeg")},
        )
        assert both.status_code == 400
        assert "not both" in both.json()["detail"]

        neither = client.post("/api/v2/voice-clips", data={"name": "x"})
        assert neither.status_code == 400

    def test_bad_language_is_rejected_before_anything_is_written(
        self, client, monkeypatch
    ):
        called = {"n": 0}

        def counting(src, dst):
            called["n"] += 1
            return 5.0

        monkeypatch.setattr(
            api_v2_module.media_mod, "make_voice_reference", counting
        )
        response = client.post(
            "/api/v2/voice-clips",
            data={"language": "klingon"},
            files={"file": ("v.mp3", b"\x00" * 8, "audio/mpeg")},
        )
        assert response.status_code == 400
        assert called["n"] == 0

    def test_patch_renames_and_retags(self, client, db_session, tmp_path):
        voice = _make_voice_clip(db_session, tmp_path)
        response = client.patch(
            f"/api/v2/voice-clips/{voice.id}",
            json={"name": "Marissa — calm", "language": "es"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Marissa — calm"
        assert response.json()["language"] == "es"

        # "" clears the tag back to language-agnostic.
        cleared = client.patch(
            f"/api/v2/voice-clips/{voice.id}", json={"language": ""}
        )
        assert cleared.json()["language"] is None

    def test_delete_is_never_blocked_by_a_run(
        self, client, db_session, tmp_path
    ):
        """The copy-on-attach rule's payoff: no active-reference guard needed.

        A run owns its audio, so deleting the library row cannot affect it —
        unlike a stitch clip, where the same request would 409.
        """
        voice = _make_voice_clip(db_session, tmp_path)
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.processing,
            model="seedance-2-5",
            reference_audio_urls=["/data/projects/p/runs/r/voice/voice.mp3"],
        )
        assert run.status == RunStatus.processing

        response = client.delete(f"/api/v2/voice-clips/{voice.id}")
        assert response.status_code == 204

        # The run's own copy is untouched by the delete.
        db_session.refresh(run)
        assert run.reference_audio_urls == [
            "/data/projects/p/runs/r/voice/voice.mp3"
        ]


class TestCreateRunVoiceReference:
    """Attaching a voice at run creation."""

    def _ready_project(self, session):
        project = _make_project(session, status=ProjectStatus.ready)
        _make_segment_def(session, project.id, 0, action="swap")
        return project

    def test_library_voice_is_copied_into_the_run(
        self, client, db_session, SessionFactory, tmp_path
    ):
        voice = _make_voice_clip(db_session, tmp_path)
        (tmp_path / "voice.mp3").write_bytes(b"REALAUDIO")
        project = self._ready_project(db_session)

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "model": "seedance-2-5",
                "resolution": "720p",
                "voice_clip_id": voice.id,
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        paths = list(run.reference_audio_urls or [])
        session.close()

        assert len(paths) == 1
        # A COPY, not a pointer at the library file — that is what makes the
        # library row safe to delete.
        assert paths[0] != voice.local_path
        assert run_id in paths[0]
        assert os.path.exists(paths[0])
        with open(paths[0], "rb") as fh:
            assert fh.read() == b"REALAUDIO"

    def test_uploaded_voice_is_normalised(
        self, client, db_session, SessionFactory, monkeypatch
    ):
        seen = {}

        def fake_normalise(src, dst):
            seen["src"] = src
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"NORMALISED")
            return 6.0

        monkeypatch.setattr(
            api_v2_module.media_mod, "make_voice_reference", fake_normalise
        )
        project = self._ready_project(db_session)

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "model": "seedance-2-5",
                "resolution": "720p",
            },
            files={"voice_file": ("take1.wav", b"\x00" * 64, "audio/wav")},
        )
        assert response.status_code == 201

        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        paths = list(run.reference_audio_urls or [])
        session.close()
        assert len(paths) == 1
        with open(paths[0], "rb") as fh:
            assert fh.read() == b"NORMALISED"
        # The raw upload is transcoded, never sent as-is.
        assert seen["src"].endswith("upload.src")

    def test_no_voice_leaves_the_column_empty(self, client, db_session, SessionFactory):
        project = self._ready_project(db_session)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap character", "model": "seedance-2-5",
                  "resolution": "720p"},
        )
        assert response.status_code == 201
        session = SessionFactory()
        run = session.get(Run, response.json()["run_id"])
        assert (run.reference_audio_urls or []) == []
        session.close()

    def test_both_sources_is_400(self, client, db_session, tmp_path):
        voice = _make_voice_clip(db_session, tmp_path)
        project = self._ready_project(db_session)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "model": "seedance-2-5",
                "resolution": "720p",
                "voice_clip_id": voice.id,
            },
            files={"voice_file": ("take1.wav", b"\x00" * 8, "audio/wav")},
        )
        assert response.status_code == 400
        assert "not both" in response.json()["detail"]

    @pytest.mark.parametrize(
        "model,resolution",
        [("seedance", "720p"), ("seedance-fast", "720p"), ("gemini-omni", "720p")],
    )
    def test_rejected_for_a_model_without_the_field(
        self, client, db_session, tmp_path, model, resolution
    ):
        """Rejected, not silently dropped — a control that does nothing is worse.

        The New Run form hides the picker for these models, so this only fires
        on direct API use.
        """
        voice = _make_voice_clip(db_session, tmp_path)
        project = self._ready_project(db_session)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "model": model,
                "resolution": resolution,
                "voice_clip_id": voice.id,
            },
        )
        assert response.status_code == 400
        assert "does not accept a voice reference" in response.json()["detail"]

    def test_unknown_voice_id_is_404(self, client, db_session):
        project = self._ready_project(db_session)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "model": "seedance-2-5",
                "resolution": "720p",
                "voice_clip_id": "no-such-voice",
            },
        )
        assert response.status_code == 404

    def test_a_rejected_voice_leaves_no_orphan_run(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """The run is built and flushed before the voice is resolved.

        A 400 there must roll the whole thing back — an orphaned `created` run
        would sit in the project's run list forever, having never been submitted.
        """
        voice = _make_voice_clip(db_session, tmp_path)
        project = self._ready_project(db_session)

        before = SessionFactory()
        n_before = len(
            before.query(Run).filter(Run.project_id == project.id).all()
        )
        before.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "model": "seedance",  # no reference_audio_urls field
                "resolution": "720p",
                "voice_clip_id": voice.id,
            },
        )
        assert response.status_code == 400

        after = SessionFactory()
        n_after = len(after.query(Run).filter(Run.project_id == project.id).all())
        after.close()
        assert n_after == n_before

class TestCopyRunVoiceReference:
    def test_copy_clones_the_voice(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """A copy in a different voice would defeat the whole feature."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")

        vdir = tmp_path / "src_run_voice"
        vdir.mkdir()
        src_voice = vdir / "voice.mp3"
        src_voice.write_bytes(b"SAMEVOICE")

        src_run = _make_run(
            db_session, project.id, status=RunStatus.done,
            model="seedance-2-5", audio_mode="seedance",
            reference_audio_urls=[str(src_voice)],
        )

        response = client.post(f"/api/v2/runs/{src_run.id}/copy", data={})
        assert response.status_code == 201
        new_id = response.json()["run_id"]

        session = SessionFactory()
        new_run = session.get(Run, new_id)
        paths = list(new_run.reference_audio_urls or [])
        session.close()

        assert len(paths) == 1
        # Cloned into the copy's own dir, so deleting the source run cannot
        # silence it.
        assert paths[0] != str(src_voice)
        assert os.path.exists(paths[0])
        with open(paths[0], "rb") as fh:
            assert fh.read() == b"SAMEVOICE"

    def test_a_new_reference_photo_keeps_the_voice(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """"Same voice, new face" must stay expressible.

        A new photo replaces the FACE; the voice is a different axis and is
        carried over untouched.
        """
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")
        vdir = tmp_path / "voice_keep"
        vdir.mkdir()
        src_voice = vdir / "voice.mp3"
        src_voice.write_bytes(b"KEEPME")
        src_run = _make_run(
            db_session, project.id, status=RunStatus.done,
            model="seedance-2-5", audio_mode="seedance",
            reference_audio_urls=[str(src_voice)],
        )

        response = client.post(
            f"/api/v2/runs/{src_run.id}/copy",
            files={"reference_files": ("newface.jpg", b"\xff\xd8\xff", "image/jpeg")},
        )
        assert response.status_code == 201

        session = SessionFactory()
        new_run = session.get(Run, response.json()["run_id"])
        paths = list(new_run.reference_audio_urls or [])
        session.close()
        assert len(paths) == 1
        with open(paths[0], "rb") as fh:
            assert fh.read() == b"KEEPME"

    def test_copy_without_a_voice_stays_empty(
        self, client, db_session, SessionFactory
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")
        src_run = _make_run(
            db_session, project.id, status=RunStatus.done, model="seedance-2-5"
        )
        response = client.post(f"/api/v2/runs/{src_run.id}/copy", data={})
        assert response.status_code == 201
        session = SessionFactory()
        new_run = session.get(Run, response.json()["run_id"])
        assert (new_run.reference_audio_urls or []) == []
        session.close()


class TestOverlongSwapSegmentLaunchGate:
    """A swap segment longer than the model's per-clip ceiling must be refused
    at LAUNCH, not silently delivered as original footage.

    Before this gate, such a run reached `done` and delivered: submit marked
    the segment source_fallback and the stitch substituted the un-swapped
    source, so a 59s creative shipped with its back 31.5s never swapped and
    nothing in the delivered file said so.
    """

    def _project(self, session, *, swap_len: float, second_action: str = "swap"):
        project = _make_project(session, status=ProjectStatus.ready)
        segs = [
            _make_segment_def(
                session, project.id, 0, action="swap", start_sec=0.0, end_sec=5.0
            ),
            _make_segment_def(
                session,
                project.id,
                1,
                action=second_action,
                has_face=(second_action == "swap"),
                start_sec=5.0,
                end_sec=5.0 + swap_len,
            ),
        ]
        return project, segs

    def _post_run(self, client, project_id, model="seedance-2-5"):
        return client.post(
            f"/api/v2/projects/{project_id}/runs",
            data={"prompt": "swap character", "model": model},
        )

    # ---- create_run ----

    def test_overlong_swap_segment_is_400(self, client, db_session):
        # 31.5s swap on Seedance 2.5 (30s ceiling) — the shape that shipped
        # a half-swapped video.
        project, _ = self._project(db_session, swap_len=31.5)
        response = self._post_run(client, project.id)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "30s per segment" in detail
        assert "#1 (31.5s)" in detail
        assert "keep" in detail

    def test_segment_at_the_ceiling_is_allowed(self, client, db_session):
        project, _ = self._project(db_session, swap_len=30.0)
        assert self._post_run(client, project.id).status_code == 201

    def test_epsilon_over_the_ceiling_is_allowed(self, client, db_session):
        # Matches pipeline_v2's own +0.05 tolerance: a segment this endpoint
        # accepts must never be one submit then skips as un-generatable.
        project, _ = self._project(db_session, swap_len=30.04)
        assert self._post_run(client, project.id).status_code == 201

    def test_overlong_keep_segment_is_allowed(self, client, db_session):
        # `keep` is the documented escape hatch — same delivered footage,
        # stated on the timeline instead of hidden behind a failed swap.
        project, _ = self._project(db_session, swap_len=31.5, second_action="keep")
        assert self._post_run(client, project.id).status_code == 201

    def test_gate_is_per_model(self, client, db_session):
        # 20s: fine on 2.5 (30s), impossible on the 2.0 family (15s).
        project, _ = self._project(db_session, swap_len=20.0)
        assert self._post_run(client, project.id, "seedance-2-5").status_code == 201
        response = self._post_run(client, project.id, "seedance")
        assert response.status_code == 400
        assert "#1 (20.0s)" in response.json()["detail"]

    def test_offenders_are_listed_longest_first(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(
            db_session, project.id, 0, action="swap", start_sec=0.0, end_sec=31.0
        )
        _make_segment_def(
            db_session, project.id, 1, action="swap", start_sec=31.0, end_sec=71.0
        )
        response = self._post_run(client, project.id)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "2 swap segment(s)" in detail
        assert detail.index("#1 (40.0s)") < detail.index("#0 (31.0s)")

    # ---- _build_copied_run (copy / batch copy / automation launches) ----

    def test_copy_of_a_run_with_an_overlong_snapshot_is_400(
        self, client, db_session
    ):
        """A copy inherits the SOURCE's frozen snapshot, so that is what the
        gate has to read — not the project's live partition."""
        project, segs = self._project(db_session, swap_len=10.0)
        src = _make_run(
            db_session,
            project.id,
            model="seedance-2-5",
            status=RunStatus.done,
            segment_snapshot=[
                {
                    "id": segs[0].id, "index": 0, "action": "swap",
                    "start_sec": 0.0, "end_sec": 5.0,
                    "pre_roll_sec": 0.0, "post_roll_sec": 0.0,
                },
                {
                    "id": segs[1].id, "index": 1, "action": "swap",
                    "start_sec": 5.0, "end_sec": 36.5,
                    "pre_roll_sec": 0.0, "post_roll_sec": 0.0,
                },
            ],
        )
        response = client.post(f"/api/v2/runs/{src.id}/copy", data={})
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Cannot copy run" in detail
        assert "#1 (31.5s)" in detail

    def test_copy_reads_the_snapshot_not_the_live_project(self, client, db_session):
        """The mirror case: the PROJECT has grown an over-long segment, but the
        source run's frozen snapshot is still fine — the copy runs the
        snapshot, so it must be allowed."""
        project, segs = self._project(db_session, swap_len=31.5)
        src = _make_run(
            db_session,
            project.id,
            model="seedance-2-5",
            status=RunStatus.done,
            segment_snapshot=[
                {
                    "id": segs[0].id, "index": 0, "action": "swap",
                    "start_sec": 0.0, "end_sec": 5.0,
                    "pre_roll_sec": 0.0, "post_roll_sec": 0.0,
                },
                {
                    "id": segs[1].id, "index": 1, "action": "keep",
                    "start_sec": 5.0, "end_sec": 36.5,
                    "pre_roll_sec": 0.0, "post_roll_sec": 0.0,
                },
            ],
        )
        assert client.post(f"/api/v2/runs/{src.id}/copy", data={}).status_code == 201

    def test_copy_of_a_snapshotless_run_falls_back_to_the_project(
        self, client, db_session
    ):
        """A legacy source with no snapshot re-freezes the project's live
        partition at its own submit, so that is what gets validated."""
        project, _ = self._project(db_session, swap_len=31.5)
        src = _make_run(
            db_session,
            project.id,
            model="seedance-2-5",
            status=RunStatus.done,
            segment_snapshot=None,
        )
        response = client.post(f"/api/v2/runs/{src.id}/copy", data={})
        assert response.status_code == 400
        assert "#1 (31.5s)" in response.json()["detail"]

    # ---- stitch plans ----
    #
    # A CUSTOM plan submits only the swap segments it lists, so an over-long
    # segment the plan drops is never generated, never billed and never
    # source_fallback'd — blocking that launch would be a false positive.

    def test_custom_plan_excluding_the_overlong_segment_is_allowed(
        self, client, db_session
    ):
        project, segs = self._project(db_session, swap_len=31.5)
        plan = json.dumps([{"kind": "segment", "segment_def_id": segs[0].id}])
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "p", "model": "seedance-2-5", "stitch_plan": plan},
        )
        assert response.status_code == 201

    def test_custom_plan_including_the_overlong_segment_is_400(
        self, client, db_session
    ):
        project, segs = self._project(db_session, swap_len=31.5)
        plan = json.dumps(
            [
                {"kind": "segment", "segment_def_id": segs[0].id},
                {"kind": "segment", "segment_def_id": segs[1].id},
            ]
        )
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "p", "model": "seedance-2-5", "stitch_plan": plan},
        )
        assert response.status_code == 400
        assert "#1 (31.5s)" in response.json()["detail"]

    def test_no_stitch_plan_still_processes_every_swap_and_is_400(
        self, client, db_session
    ):
        # `[]` means "generate every swap segment, produce no final video" —
        # the credits are still spent, so the gate still applies.
        project, _ = self._project(db_session, swap_len=31.5)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "p", "model": "seedance-2-5", "stitch_plan": "[]"},
        )
        assert response.status_code == 400
        assert "#1 (31.5s)" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Italian (it) — the sixth registry language, end to end through the REST API
# ---------------------------------------------------------------------------


class TestItalianRunLanguage:
    """Italian must travel the same generic paths every other language does.

    Nothing here is Italian-specific code: these pin that the registry addition
    actually reaches storage, validation, copies and the derived short id, and
    that the closed set still refuses everything else.
    """

    def test_create_run_stores_italian(self, spy_client, SessionFactory):
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": "it"},
        )
        assert resp.status_code == 201, resp.text

        session = SessionFactory()
        run = session.get(Run, resp.json()["run_id"])
        assert run.language == "it"
        assert run.short_id == f"{run.id[:8]}-it"
        session.close()

    def test_create_run_normalizes_case_and_whitespace(self, spy_client, SessionFactory):
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": "  IT "},
        )
        assert resp.status_code == 201, resp.text

        session = SessionFactory()
        run = session.get(Run, resp.json()["run_id"])
        assert run.language == "it"          # persisted canonical, not "  IT "
        session.close()

    @pytest.mark.parametrize("alias", ["ita", "it-IT", "it_IT", "italian"])
    def test_create_run_rejects_italian_aliases(self, client, db_session, alias):
        """Only the canonical code — a locale variant would reach the sheet."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": alias},
        )
        assert resp.status_code == 400
        assert "language" in resp.json()["detail"]

    def test_patch_to_italian_changes_the_short_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id)

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "it"})
        assert resp.status_code == 200
        assert resp.json()["language"] == "it"

        db_session.expire_all()
        assert db_session.get(Run, run.id).short_id == f"{run.id[:8]}-it"

    def test_patch_from_italian_back_to_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="it")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": ""})
        assert resp.status_code == 200
        assert resp.json()["language"] is None
        db_session.expire_all()
        assert db_session.get(Run, run.id).short_id == run.id[:8]

    def test_patch_rejects_unknown_and_leaves_italian_intact(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="it")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "nl"})
        assert resp.status_code == 400
        db_session.expire_all()
        assert db_session.get(Run, run.id).language == "it"


class TestItalianInheritsThroughCopies:
    """Copy, batch copy and (via the same _build_copied_run) automation."""

    def test_copy_inherits_italian(self, spy_client, SessionFactory):
        client, _spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, prompt="hello", model="seedance", resolution="480p",
            status=RunStatus.done, language="it",
        )
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        s.close()
        assert nr.language == "it"
        assert nr.short_id == f"{new_id[:8]}-it"

    def test_copy_of_a_copy_stays_italian(self, spy_client, SessionFactory):
        client, _spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, status=RunStatus.done, language="it")
        rid = run.id
        s.close()

        first = client.post(f"/api/v2/runs/{rid}/copy", data={}).json()["run_id"]
        second = client.post(f"/api/v2/runs/{first}/copy", data={}).json()["run_id"]

        s = SessionFactory()
        assert s.get(Run, first).language == "it"
        assert s.get(Run, second).language == "it"
        s.close()

    def test_batch_copy_inherits_italian_for_every_child(
        self, spy_client, SessionFactory
    ):
        client, _spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, status=RunStatus.done, language="it")
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={"runs[0][resolution]": "480p", "runs[1][resolution]": "720p"},
        )
        assert resp.status_code == 201, resp.text
        ids = [r["run_id"] for r in resp.json()["runs"]]
        assert len(ids) == 2

        s = SessionFactory()
        for run_id in ids:
            r = s.get(Run, run_id)
            assert r.language == "it", run_id
            assert r.short_id == f"{run_id[:8]}-it"
        s.close()


class TestItalianLibraryAndVoiceTags:
    def test_library_clip_accepts_italian(self, client, SessionFactory, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        resp = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro IT", "language": "  IT "},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["language"] == "it"

        session = SessionFactory()
        clip = session.get(LibraryClip, resp.json()["id"])
        session.close()
        assert clip.language == "it"

    def test_library_clip_language_can_be_edited_to_and_from_italian(
        self, client, SessionFactory, monkeypatch
    ):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        clip_id = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro", "language": "es"},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        ).json()["id"]

        resp = client.patch(f"/api/v2/library-clips/{clip_id}", json={"language": "it"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["language"] == "it"

        # Cleared back to language-agnostic (NULL), which is untagged, not English.
        resp = client.patch(f"/api/v2/library-clips/{clip_id}", json={"language": ""})
        assert resp.status_code == 200
        assert resp.json()["language"] is None

        session = SessionFactory()
        assert session.get(LibraryClip, clip_id).language is None
        session.close()

    def test_voice_clip_accepts_italian(self, client, SessionFactory, monkeypatch):
        def fake_normalise(src, dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"ID3fake")
            return 7.5

        monkeypatch.setattr(
            api_v2_module.media_mod, "make_voice_reference", fake_normalise
        )
        resp = client.post(
            "/api/v2/voice-clips",
            data={"name": "Giulia IT", "language": "it"},
            files={"file": ("giulia.mp3", b"\x00\x01" * 32, "audio/mpeg")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["language"] == "it"

        session = SessionFactory()
        assert session.get(VoiceClip, resp.json()["id"]).language == "it"
        session.close()

    def test_voice_clip_rejects_an_unsupported_language(self, client, monkeypatch):
        def fake_normalise(src, dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"ID3fake")
            return 7.5

        monkeypatch.setattr(
            api_v2_module.media_mod, "make_voice_reference", fake_normalise
        )
        resp = client.post(
            "/api/v2/voice-clips",
            data={"name": "Nope", "language": "nl"},
            files={"file": ("x.mp3", b"\x00\x01" * 32, "audio/mpeg")},
        )
        assert resp.status_code == 400
        assert "language" in resp.json()["detail"]


class TestItalianLocalisationTarget:
    """POST /projects/{pid}/localisation-prompt with Italian as the target.

    Translation is stubbed (fake_translate) — no kie.ai call, no cost. What is
    NOT stubbed is build_prompt, so these render the real project_types
    templates and prove the Italian language NAME reaches both of them.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "it"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def test_italian_is_an_accepted_target(self, client, db_session, fake_translate):
        project = _localisation_project(db_session)
        resp = self._post(client, project.id)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["target_language"] == "it"
        assert body["source_language"] == "en"

    def test_translate_is_asked_for_italian(self, client, db_session, fake_translate):
        """The registry code — not a label, not a locale — reaches the provider."""
        project = _localisation_project(db_session)
        assert self._post(client, project.id).status_code == 200
        assert len(fake_translate) == 1
        assert fake_translate[0]["target_language"] == "it"
        assert fake_translate[0]["source_language"] == "en"

    @pytest.mark.parametrize("swap_character", ["true", "false"])
    def test_both_modes_name_italian_in_the_prompt(
        self, client, db_session, fake_translate, swap_character
    ):
        """Speech-only AND speech-plus-character assembly must say 'italian'."""
        project = _localisation_project(db_session)
        resp = self._post(client, project.id, swap_character=swap_character)
        assert resp.status_code == 200, resp.text
        prompt = resp.json()["prompt"]

        assert "from english to italian" in prompt
        # The other languages' names must not leak into an Italian prompt.
        for other in ("spanish", "japanese", "german", "portuguese"):
            assert other not in prompt.lower()

    def test_the_two_modes_still_pick_different_templates(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        swap = self._post(client, project.id, swap_character="true").json()["prompt"]
        keep = self._post(client, project.id, swap_character="false").json()["prompt"]

        assert swap != keep
        assert "Replace the main person" in swap
        assert "Do not replace the character" in keep
        for prompt in (swap, keep):
            assert "from english to italian" in prompt

    def test_line_ids_speakers_and_source_text_survive(
        self, client, db_session, fake_translate
    ):
        """Per-line correspondence is the contract translate_lines guarantees."""
        project = _localisation_project(db_session)
        body = self._post(client, project.id).json()

        assert [ln["id"] for ln in body["lines"]] == [1, 2]
        assert body["lines"][0]["source_text"] == "Hey, what are you doing?"
        assert body["lines"][0]["speaker"] == "the woman in the red jacket"
        assert body["lines"][1]["source_text"] == "I'm just listening to my emails."
        # Timings come straight off the transcript, untouched by the target.
        assert body["lines"][0]["start"] == _HOOK_LINES[0]["start"]
        assert body["lines"][0]["end"] == _HOOK_LINES[0]["end"]

    def test_italian_source_and_italian_target_is_refused_as_a_no_op(
        self, client, db_session, fake_translate
    ):
        """Same-language translation is the existing mis-pick guard, not new."""
        project = _localisation_project(
            db_session, transcript=_transcript(_HOOK_LINES, source_language="it")
        )
        resp = self._post(client, project.id, language="it")
        assert resp.status_code == 200, resp.text
        assert any("no-op" in w.lower() or "italian" in w.lower()
                   for w in resp.json()["warnings"]), resp.json()["warnings"]

    def test_italian_can_be_the_detected_source_of_a_spanish_run(
        self, client, db_session, fake_translate
    ):
        """Detected source stays unrestricted — Italian in, Spanish out."""
        project = _localisation_project(
            db_session, transcript=_transcript(_HOOK_LINES, source_language="it")
        )
        resp = self._post(client, project.id, language="es")
        assert resp.status_code == 200, resp.text
        assert resp.json()["source_language"] == "it"
        assert fake_translate[0]["source_language"] == "it"
        assert "from italian to spanish" in resp.json()["prompt"]


# ---------------------------------------------------------------------------
# Polish (pl) — the seventh registry language, end to end through the REST API
# ---------------------------------------------------------------------------
#
# Mirrors the Italian block above. Like Italian, nothing here is Polish-specific
# code; these pin that the one-line registry addition reaches storage,
# validation, copies, library tags, localisation and the derived short id.


class TestPolishRunLanguage:
    def test_create_run_stores_polish(self, spy_client, SessionFactory):
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": "  PL "},
        )
        assert resp.status_code == 201, resp.text

        session = SessionFactory()
        run = session.get(Run, resp.json()["run_id"])
        assert run.language == "pl"          # persisted canonical, not "  PL "
        assert run.short_id == f"{run.id[:8]}-pl"
        session.close()

    @pytest.mark.parametrize("alias", ["pol", "pl-PL", "pl_PL", "polish"])
    def test_create_run_rejects_polish_aliases(self, client, db_session, alias):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": alias},
        )
        assert resp.status_code == 400
        assert "language" in resp.json()["detail"]

    def test_patch_to_polish_changes_the_short_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="it")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "pl"})
        assert resp.status_code == 200
        assert resp.json()["language"] == "pl"

        db_session.expire_all()
        assert db_session.get(Run, run.id).short_id == f"{run.id[:8]}-pl"

    def test_patch_from_polish_back_to_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="pl")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": ""})
        assert resp.status_code == 200
        assert resp.json()["language"] is None
        db_session.expire_all()
        assert db_session.get(Run, run.id).short_id == run.id[:8]

    def test_patch_rejects_unknown_and_leaves_polish_intact(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="pl")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "cs"})
        assert resp.status_code == 400
        db_session.expire_all()
        assert db_session.get(Run, run.id).language == "pl"


class TestPolishInheritsThroughCopies:
    """Copy, batch copy and (via the same _build_copied_run) automation."""

    def test_copy_inherits_polish(self, spy_client, SessionFactory):
        client, _spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, prompt="hello", model="seedance", resolution="480p",
            status=RunStatus.done, language="pl",
        )
        rid = run.id
        s.close()

        first = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert first.status_code == 201, first.text
        first_id = first.json()["run_id"]
        second_id = client.post(f"/api/v2/runs/{first_id}/copy", data={}).json()["run_id"]

        s = SessionFactory()
        for run_id in (first_id, second_id):
            r = s.get(Run, run_id)
            assert r.language == "pl", run_id
            assert r.short_id == f"{run_id[:8]}-pl"
        s.close()

    def test_batch_copy_inherits_polish_for_every_child(
        self, spy_client, SessionFactory
    ):
        client, _spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, status=RunStatus.done, language="pl")
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={"runs[0][resolution]": "480p", "runs[1][resolution]": "720p"},
        )
        assert resp.status_code == 201, resp.text
        ids = [r["run_id"] for r in resp.json()["runs"]]
        assert len(ids) == 2

        s = SessionFactory()
        for run_id in ids:
            r = s.get(Run, run_id)
            assert r.language == "pl", run_id
            assert r.short_id == f"{run_id[:8]}-pl"
        s.close()


class TestPolishLibraryAndVoiceTags:
    def test_library_clip_accepts_polish(self, client, SessionFactory, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        resp = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro PL", "language": "  PL "},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["language"] == "pl"

        session = SessionFactory()
        clip = session.get(LibraryClip, resp.json()["id"])
        session.close()
        assert clip.language == "pl"

    def test_library_clip_language_can_be_edited_to_polish(
        self, client, SessionFactory, monkeypatch
    ):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        clip_id = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro", "language": "it"},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        ).json()["id"]

        resp = client.patch(f"/api/v2/library-clips/{clip_id}", json={"language": "pl"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["language"] == "pl"

        session = SessionFactory()
        assert session.get(LibraryClip, clip_id).language == "pl"
        session.close()

    def test_voice_clip_accepts_polish(self, client, SessionFactory, monkeypatch):
        def fake_normalise(src, dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"ID3fake")
            return 7.5

        monkeypatch.setattr(
            api_v2_module.media_mod, "make_voice_reference", fake_normalise
        )
        resp = client.post(
            "/api/v2/voice-clips",
            data={"name": "Zofia PL", "language": "pl"},
            files={"file": ("zofia.mp3", b"\x00\x01" * 32, "audio/mpeg")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["language"] == "pl"

        session = SessionFactory()
        assert session.get(VoiceClip, resp.json()["id"]).language == "pl"
        session.close()


class TestPolishLocalisationTarget:
    """POST /projects/{pid}/localisation-prompt with Polish as the target.

    Translation is stubbed (fake_translate) — no kie.ai call, no cost.
    build_prompt is NOT stubbed, so these render the real project_types
    templates and prove the Polish language NAME reaches the Seedance prompt.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "pl"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def test_polish_uses_the_default_latin_speech_rate(self):
        """Latin script — no per-language override like Japanese's 6 chars/s."""
        assert "pl" not in api_v2_module._CHARS_PER_SEC

    def test_translate_is_asked_for_polish(self, client, db_session, fake_translate):
        """The registry code — not a label, not a locale — reaches the provider."""
        project = _localisation_project(db_session)
        resp = self._post(client, project.id)
        assert resp.status_code == 200, resp.text
        assert resp.json()["target_language"] == "pl"
        assert len(fake_translate) == 1
        assert fake_translate[0]["target_language"] == "pl"
        assert fake_translate[0]["source_language"] == "en"

    @pytest.mark.parametrize("swap_character", ["true", "false"])
    def test_both_modes_name_polish_in_the_prompt(
        self, client, db_session, fake_translate, swap_character
    ):
        project = _localisation_project(db_session)
        resp = self._post(client, project.id, swap_character=swap_character)
        assert resp.status_code == 200, resp.text
        prompt = resp.json()["prompt"]

        assert "from english to polish" in prompt
        for other in ("spanish", "japanese", "german", "portuguese", "italian"):
            assert other not in prompt.lower()

    def test_polish_can_be_the_detected_source_of_an_italian_run(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session, transcript=_transcript(_HOOK_LINES, source_language="pl")
        )
        resp = self._post(client, project.id, language="it")
        assert resp.status_code == 200, resp.text
        assert resp.json()["source_language"] == "pl"
        assert "from polish to italian" in resp.json()["prompt"]


# French (fr): the same registry-only addition as Polish. "-fr" was
# already parsed downstream, so French runs are exportable from day one.


class TestFrenchRunLanguage:
    def test_create_run_stores_french(self, spy_client, SessionFactory):
        client, _spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": "  FR "},
        )
        assert resp.status_code == 201, resp.text

        session = SessionFactory()
        run = session.get(Run, resp.json()["run_id"])
        assert run.language == "fr"          # persisted canonical, not "  FR "
        assert run.short_id == f"{run.id[:8]}-fr"
        session.close()

    @pytest.mark.parametrize("alias", ["fra", "fr-FR", "pl_PL", "french"])
    def test_create_run_rejects_french_aliases(self, client, db_session, alias):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "language": alias},
        )
        assert resp.status_code == 400
        assert "language" in resp.json()["detail"]

    def test_patch_to_french_changes_the_short_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="it")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "fr"})
        assert resp.status_code == 200
        assert resp.json()["language"] == "fr"

        db_session.expire_all()
        assert db_session.get(Run, run.id).short_id == f"{run.id[:8]}-fr"

    def test_patch_from_french_back_to_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="fr")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": ""})
        assert resp.status_code == 200
        assert resp.json()["language"] is None
        db_session.expire_all()
        assert db_session.get(Run, run.id).short_id == run.id[:8]

    def test_patch_rejects_unknown_and_leaves_french_intact(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, language="fr")

        resp = client.patch(f"/api/v2/runs/{run.id}", json={"language": "cs"})
        assert resp.status_code == 400
        db_session.expire_all()
        assert db_session.get(Run, run.id).language == "fr"


class TestFrenchInheritsThroughCopies:
    """Copy, batch copy and (via the same _build_copied_run) automation."""

    def test_copy_inherits_french(self, spy_client, SessionFactory):
        client, _spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, prompt="hello", model="seedance", resolution="480p",
            status=RunStatus.done, language="fr",
        )
        rid = run.id
        s.close()

        first = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert first.status_code == 201, first.text
        first_id = first.json()["run_id"]
        second_id = client.post(f"/api/v2/runs/{first_id}/copy", data={}).json()["run_id"]

        s = SessionFactory()
        for run_id in (first_id, second_id):
            r = s.get(Run, run_id)
            assert r.language == "fr", run_id
            assert r.short_id == f"{run_id[:8]}-fr"
        s.close()

    def test_batch_copy_inherits_french_for_every_child(
        self, spy_client, SessionFactory
    ):
        client, _spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, status=RunStatus.done, language="fr")
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={"runs[0][resolution]": "480p", "runs[1][resolution]": "720p"},
        )
        assert resp.status_code == 201, resp.text
        ids = [r["run_id"] for r in resp.json()["runs"]]
        assert len(ids) == 2

        s = SessionFactory()
        for run_id in ids:
            r = s.get(Run, run_id)
            assert r.language == "fr", run_id
            assert r.short_id == f"{run_id[:8]}-fr"
        s.close()


class TestFrenchLibraryAndVoiceTags:
    def test_library_clip_accepts_french(self, client, SessionFactory, monkeypatch):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        resp = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro FR", "language": "  FR "},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["language"] == "fr"

        session = SessionFactory()
        clip = session.get(LibraryClip, resp.json()["id"])
        session.close()
        assert clip.language == "fr"

    def test_library_clip_language_can_be_edited_to_french(
        self, client, SessionFactory, monkeypatch
    ):
        monkeypatch.setattr(
            api_v2_module.media_mod, "probe", lambda path: _fake_media_info()
        )
        clip_id = client.post(
            "/api/v2/library-clips",
            data={"name": "Outro", "language": "it"},
            files={"file": ("outro.mp4", b"\x00\x01" * 32, "video/mp4")},
        ).json()["id"]

        resp = client.patch(f"/api/v2/library-clips/{clip_id}", json={"language": "fr"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["language"] == "fr"

        session = SessionFactory()
        assert session.get(LibraryClip, clip_id).language == "fr"
        session.close()

    def test_voice_clip_accepts_french(self, client, SessionFactory, monkeypatch):
        def fake_normalise(src, dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"ID3fake")
            return 7.5

        monkeypatch.setattr(
            api_v2_module.media_mod, "make_voice_reference", fake_normalise
        )
        resp = client.post(
            "/api/v2/voice-clips",
            data={"name": "Camille FR", "language": "fr"},
            files={"file": ("camille.mp3", b"\x00\x01" * 32, "audio/mpeg")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["language"] == "fr"

        session = SessionFactory()
        assert session.get(VoiceClip, resp.json()["id"]).language == "fr"
        session.close()


class TestFrenchLocalisationTarget:
    """POST /projects/{pid}/localisation-prompt with French as the target.

    Translation is stubbed (fake_translate) — no kie.ai call, no cost.
    build_prompt is NOT stubbed, so these render the real project_types
    templates and prove the French language NAME reaches the Seedance prompt.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "fr"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def test_french_uses_the_default_latin_speech_rate(self):
        """Latin script — no per-language override like Japanese's 6 chars/s."""
        assert "fr" not in api_v2_module._CHARS_PER_SEC

    def test_translate_is_asked_for_french(self, client, db_session, fake_translate):
        """The registry code — not a label, not a locale — reaches the provider."""
        project = _localisation_project(db_session)
        resp = self._post(client, project.id)
        assert resp.status_code == 200, resp.text
        assert resp.json()["target_language"] == "fr"
        assert len(fake_translate) == 1
        assert fake_translate[0]["target_language"] == "fr"
        assert fake_translate[0]["source_language"] == "en"

    @pytest.mark.parametrize("swap_character", ["true", "false"])
    def test_both_modes_name_french_in_the_prompt(
        self, client, db_session, fake_translate, swap_character
    ):
        project = _localisation_project(db_session)
        resp = self._post(client, project.id, swap_character=swap_character)
        assert resp.status_code == 200, resp.text
        prompt = resp.json()["prompt"]

        assert "from english to french" in prompt
        for other in ("spanish", "japanese", "german", "portuguese", "italian", "polish"):
            assert other not in prompt.lower()

    def test_french_can_be_the_detected_source_of_an_italian_run(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session, transcript=_transcript(_HOOK_LINES, source_language="fr")
        )
        resp = self._post(client, project.id, language="it")
        assert resp.status_code == 200, resp.text
        assert resp.json()["source_language"] == "fr"
        assert "from french to italian" in resp.json()["prompt"]


# ---------------------------------------------------------------------------
# AICP-148 — reference paths: request validation + serving containment
# ---------------------------------------------------------------------------


def _outside_paths():
    """Stored values that must never be opened: absolute paths outside
    DATA_DIR, a ``..`` traversal that starts inside it, a symlink inside it
    pointing out, and legacy junk like ``T3``."""
    from app import storage

    root = storage.data_dir_root()
    link_dir = tempfile.mkdtemp(prefix="aicp148-", dir=root)
    link = os.path.join(link_dir, "evil.jpg")
    os.symlink("/etc/passwd", link)
    return [
        "/etc/passwd",
        "/proc/self/environ",
        os.path.join(root, "..", "..", "etc", "passwd"),
        link,
        "T3",
    ]


class TestReferenceServingContainment:
    """Every endpoint that serves a stored reference path 404s outside DATA_DIR."""

    def test_reference_image(self, client, db_session):
        project = _make_project(db_session)
        bad = _outside_paths()
        run = _make_run(db_session, project.id, reference_image_urls=bad)
        for i in range(len(bad)):
            r = client.get(f"/api/v2/runs/{run.id}/reference-image?index={i}")
            assert r.status_code == 404, (bad[i], r.status_code)
            assert b"root:" not in r.content

    def test_preprocessed_image(self, client, db_session):
        project = _make_project(db_session)
        bad = _outside_paths()
        run = _make_run(db_session, project.id, preprocessed_reference_urls=bad)
        for i in range(len(bad)):
            r = client.get(f"/api/v2/runs/{run.id}/preprocessed-image?index={i}")
            assert r.status_code == 404, bad[i]

    def test_voice_reference(self, client, db_session):
        project = _make_project(db_session)
        bad = _outside_paths()
        run = _make_run(db_session, project.id, reference_audio_urls=bad)
        for i in range(len(bad)):
            r = client.get(f"/api/v2/runs/{run.id}/voice-reference?index={i}")
            assert r.status_code == 404, bad[i]

    def test_segment_override_reference(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        bad = _outside_paths()
        rs = _make_run_segment(
            db_session, run.id, sd.id, reference_image_urls_override=bad,
        )
        for i in range(len(bad)):
            r = client.get(
                f"/api/v2/runs/{run.id}/segments/{rs.id}/override-reference",
                params={"index": i},
            )
            assert r.status_code == 404, bad[i]
            assert b"root:" not in r.content

    def test_segment_image_job_file(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id)
        for bad in _outside_paths():
            job = RunSegmentImageJob(
                run_id=run.id, run_segment_id=rs.id, prompt="p", frame_sec=1.0,
                input_paths=[bad, bad], output_path=bad,
                status=SegmentImageStatus.done,
            )
            db_session.add(job)
            db_session.commit()
            for which in ("output", "frame", "character"):
                r = client.get(
                    f"/api/v2/runs/{run.id}/segments/{rs.id}/image-jobs/{job.id}/image",
                    params={"which": which},
                )
                assert r.status_code == 404, (bad, which)

    def test_inside_data_dir_still_served(self, client, db_session, tmp_path):
        img = tmp_path / "ok.jpg"
        img.write_bytes(b"\xff\xd8ok")
        project = _make_project(db_session)
        run = _make_run(
            db_session, project.id,
            reference_image_urls=[str(img)], preprocessed_reference_urls=[str(img)],
        )
        assert client.get(f"/api/v2/runs/{run.id}/reference-image?index=0").content == b"\xff\xd8ok"
        assert client.get(f"/api/v2/runs/{run.id}/preprocessed-image?index=0").status_code == 200


class TestAccelContainment:
    """accel_or_file_response fails closed outside DATA_DIR in both modes."""

    @pytest.mark.parametrize("accel_on", [False, True])
    def test_outside_paths_404(self, client, db_session, monkeypatch, accel_on):
        import app.accel as accel_module
        from app import storage

        monkeypatch.setattr(accel_module.settings, "NGINX_ACCEL_ENABLED", accel_on)
        monkeypatch.setattr(accel_module.settings, "DATA_DIR", storage.data_dir_root())
        for bad in _outside_paths():
            project = _make_project(
                db_session, source_local_path=bad, status=ProjectStatus.ready,
            )
            r = client.get(f"/api/v2/projects/{project.id}/source")
            assert r.status_code == 404, (bad, accel_on)
            assert "x-accel-redirect" not in r.headers
            assert b"root:" not in r.content

    def test_prefix_sibling_dir_is_outside(self, tmp_path):
        """``<root>2/x`` shares a string prefix with ``<root>`` but is outside."""
        from app import storage
        from app.storage import PathOutsideDataDir, ensure_under_data_dir

        root = storage.data_dir_root()
        with pytest.raises(PathOutsideDataDir):
            ensure_under_data_dir(root + "2/x.jpg")
        with pytest.raises(PathOutsideDataDir):
            ensure_under_data_dir(root)
        assert ensure_under_data_dir(str(tmp_path / "a.jpg")).startswith(root + os.sep)


_BAD_USER_REFS = [
    "/etc/passwd",
    "/proc/self/environ",
    "/app/secrets/gdrive-sa.json",
    "../../etc/passwd",
    "T3",
    "file:///etc/passwd",
    "ftp://example.com/x.jpg",
    "https://",
]


class TestReferenceUrlValidation:
    """User-supplied reference entries must be http(s) URLs (or a path the
    server itself handed out) on every entry point — 400 otherwise."""

    @pytest.mark.parametrize("bad", _BAD_USER_REFS)
    def test_create_run_rejects(self, client, db_session, bad):
        project = _make_project(db_session, status=ProjectStatus.ready)
        r = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p",
                  "reference_urls": f"https://example.com/a.jpg,{bad}"},
        )
        assert r.status_code == 400, r.text
        assert "not an http(s) URL" in r.json()["detail"]
        assert db_session.query(Run).filter_by(project_id=project.id).count() == 0

    def test_create_run_rejects_staged_prefix_traversal(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sneaky = _staged_frame(project.id, "../../../../../../../../etc/passwd")
        r = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "reference_urls": sneaky},
        )
        assert r.status_code == 400

    def test_create_run_rejects_other_projects_staged_frame(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        other = _make_project(db_session, status=ProjectStatus.ready)
        r = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p",
                  "reference_urls": _staged_frame(other.id, "frame_0_abcdef12.jpg")},
        )
        assert r.status_code == 400

    def test_create_run_accepts_urls_drive_and_staged(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        staged = _staged_frame(project.id, "frame_0_abcdef12.jpg")
        urls = [
            "https://example.com/a.jpg",
            "http://example.com/b.jpg",
            "https://drive.google.com/file/d/abc/view",
            staged,
        ]
        r = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "reference_urls": ",".join(urls)},
        )
        assert r.status_code == 201, r.text

    @pytest.mark.parametrize("bad", ["/etc/passwd", "T3"])
    def test_create_run_rejects_bad_character_url(self, client, db_session, bad):
        project = _make_project(db_session, status=ProjectStatus.ready)
        r = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p", "preprocess_enabled": "true",
                  "preprocess_reference_url": bad,
                  "reference_urls": _staged_frame(project.id, "frame_0_abcdef12.jpg")},
        )
        assert r.status_code == 400, r.text
        assert "preprocess_reference_url" in r.json()["detail"]

    @pytest.mark.parametrize("bad", ["/proc/self/environ", "T4"])
    def test_regenerate_preprocess_rejects_bad_character_url(self, client, db_session, bad):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            reference_image_urls=["https://example.com/c.jpg"],
            preprocess_image_prompts=[""],
        )
        r = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={"preprocess_reference_url": bad},
        )
        assert r.status_code == 400, r.text
        db_session.refresh(run)
        assert run.reference_image_urls == ["https://example.com/c.jpg"]
        assert run.status == RunStatus.preprocess_review

    def test_regenerate_preprocess_rejects_trim_traversal(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            reference_image_urls=["https://example.com/c.jpg"],
            preprocess_image_prompts=[""],
        )
        r = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            data={"reference_urls": _staged_frame(project.id, "../../../../../../../etc/passwd")},
        )
        assert r.status_code == 400

    def _override_rs(self, db_session, existing=None):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id, status=SegmentStatus.completed,
            reference_image_urls_override=existing,
        )
        return run, rs

    @pytest.mark.parametrize("bad", _BAD_USER_REFS)
    def test_patch_segment_override_rejects(self, client, db_session, bad):
        run, rs = self._override_rs(db_session)
        r = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "p", "reference_urls": bad},
        )
        assert r.status_code == 400, r.text
        db_session.refresh(rs)
        assert rs.reference_image_urls_override is None
        assert rs.prompt_override is None

    def test_patch_segment_override_keeps_its_own_local_entry(
        self, client, db_session, tmp_path
    ):
        """The override form re-posts the segment's existing uploaded entries."""
        held = tmp_path / "uploaded.jpg"
        held.write_bytes(b"\xff\xd8")
        run, rs = self._override_rs(db_session, existing=[str(held)])
        r = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "p", "reference_urls": f"https://example.com/x.jpg,{held}"},
        )
        assert r.status_code == 200, r.text
        db_session.refresh(rs)
        assert rs.reference_image_urls_override == ["https://example.com/x.jpg", str(held)]

    def test_patch_segment_override_rejects_unheld_local_entry(
        self, client, db_session, tmp_path
    ):
        """A path under DATA_DIR the override does NOT already hold is refused
        too — DATA_DIR also holds app.db and the DB backups."""
        other = tmp_path / "someone-elses.jpg"
        other.write_bytes(b"\xff\xd8")
        run, rs = self._override_rs(db_session)
        r = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "p", "reference_urls": str(other)},
        )
        assert r.status_code == 400

    def test_patch_segment_override_rejects_held_entry_outside_data_dir(
        self, client, db_session
    ):
        """A legacy junk entry already held by the override is not re-accepted."""
        run, rs = self._override_rs(db_session, existing=["/etc/passwd"])
        r = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "p", "reference_urls": "/etc/passwd"},
        )
        assert r.status_code == 400

    def test_rerun_segment_rejects(self, client, db_session):
        run, rs = self._override_rs(db_session)
        r = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun",
            data={"prompt": "p", "reference_urls": "/proc/self/environ"},
        )
        assert r.status_code == 400, r.text
        db_session.refresh(run)
        assert run.status == RunStatus.done

    def test_range_rerun_rejects(self, client, db_session):
        project, run, _k0, _s1, _k2, rs = _make_run_with_snapshot(db_session)
        r = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/range-rerun",
            data={"start_sec": "4.6", "end_sec": "10.0",
                  "reference_urls": "/app/secrets/gdrive-sa.json"},
        )
        assert r.status_code == 400, r.text
        assert db_session.query(RunSegmentRangeAttempt).filter_by(run_id=run.id).count() == 0

    @pytest.mark.parametrize("bad", ["/etc/passwd", "T3", "../x.jpg"])
    def test_regenerate_segment_image_rejects_character_path(self, client, db_session, bad):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0, start_sec=10.0, end_sec=20.0)
        run = _make_preprocess_ready_run(
            db_session, project.id,
            segment_snapshot=[{"id": sd.id, "index": 0, "action": "swap",
                               "start_sec": 10.0, "end_sec": 20.0}],
        )
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)
        r = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/regenerate-image",
            data={"character_url": bad},
        )
        assert r.status_code == 400, r.text
        assert db_session.query(RunSegmentImageJob).filter_by(run_id=run.id).count() == 0

    @pytest.mark.parametrize("bad", ["/etc/passwd", "T3"])
    def test_copy_run_rejects(self, client, db_session, bad):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        r = client.post(f"/api/v2/runs/{run.id}/copy", data={"reference_urls": bad})
        assert r.status_code == 400, r.text
        assert db_session.query(Run).filter_by(project_id=project.id).count() == 1

    def test_copy_batch_rejects(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        r = client.post(
            f"/api/v2/runs/{run.id}/copy-batch",
            data={"runs[0][reference_urls]": "https://example.com/ok.jpg",
                  "runs[1][reference_urls]": "/proc/self/environ"},
        )
        assert r.status_code == 400, r.text
        assert db_session.query(Run).filter_by(project_id=project.id).count() == 1

    def test_copy_does_not_launder_legacy_outside_ref(self, client, db_session, tmp_path):
        """A pre-fix row holding /etc/passwd is passed through, never copied
        into the new run's own references dir (where it would look legit)."""
        ok = tmp_path / "ok.jpg"
        ok.write_bytes(b"\xff\xd8")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=[str(ok), "/etc/passwd", "T3"],
        )
        r = client.post(f"/api/v2/runs/{run.id}/copy", data={})
        assert r.status_code == 201, r.text
        new = db_session.get(Run, r.json()["run_id"])
        assert new.reference_image_urls[0] != str(ok)
        assert os.path.exists(new.reference_image_urls[0])
        assert new.reference_image_urls[1:] == ["/etc/passwd", "T3"]


# ---------------------------------------------------------------------------
# Daily kie.ai budget (AICP-155) — every NEW paid action is admitted against
# today's budget; over it the API answers 429 and creates/queues nothing.
# The suite shares one DB, so each test sets the cap relative to whatever
# today's total already is.
# ---------------------------------------------------------------------------


from sqlalchemy import select  # noqa: E402


def _set_headroom(monkeypatch, SessionFactory, credits):
    from app import spend

    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    s = SessionFactory()
    charged = spend.today_summary(s)["charged_credits"]
    s.close()
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", (charged + credits) * 0.005)


def _budget_project(session, seconds=5.0):
    project = _make_project(session, status=ProjectStatus.ready)
    _make_segment_def(session, project.id, 0, start_sec=0.0, end_sec=seconds, action="swap")
    return project


@pytest.mark.spend_cap
class TestDailyBudget:
    # seedance (2.0) at 720p: 52 credits per clip-second -> a 5s swap = 260.
    COST = 5 * 52

    def test_new_run_within_budget_is_admitted_and_reserved(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        from app.models import KieSpendReservation

        client, spy = spy_client
        s = SessionFactory()
        project = _budget_project(s)
        s.close()
        _set_headroom(monkeypatch, SessionFactory, self.COST)

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p"},
        )
        assert resp.status_code == 201, resp.text
        rid = resp.json()["run_id"]
        assert spy["process_run"] == [rid]
        s = SessionFactory()
        res = s.execute(
            select(KieSpendReservation).where(KieSpendReservation.scope_key == f"run:{rid}")
        ).scalar_one()
        assert res.reserved_mcredits == self.COST * 1000
        s.close()

    def test_new_run_over_budget_is_429_and_nothing_is_created(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        client, spy = spy_client
        s = SessionFactory()
        project = _budget_project(s)
        s.close()
        _set_headroom(monkeypatch, SessionFactory, self.COST - 1)

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p"},
        )
        assert resp.status_code == 429, resp.text
        detail = resp.json()["detail"]
        assert "Daily kie.ai budget reached" in detail
        assert "new run" in detail and "00:00 UTC" in detail
        assert spy["process_run"] == []
        s = SessionFactory()
        assert s.execute(select(Run).where(Run.project_id == project.id)).first() is None
        s.close()

    def test_batch_copy_over_budget_is_refused_whole(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        client, spy = spy_client
        s = SessionFactory()
        project = _budget_project(s)
        src = _make_run(s, project.id, status=RunStatus.done, model="seedance")
        s.close()
        # Room for two copies, three asked for.
        _set_headroom(monkeypatch, SessionFactory, 2 * self.COST)

        resp = client.post(
            f"/api/v2/runs/{src.id}/copy-batch",
            data={f"runs[{i}][resolution]": "720p" for i in range(3)},
        )
        assert resp.status_code == 429, resp.text
        assert "batch copy" in resp.json()["detail"]
        s = SessionFactory()
        assert len(s.execute(select(Run).where(Run.project_id == project.id)).all()) == 1
        s.close()
        assert spy["process_run"] == []

        resp = client.post(
            f"/api/v2/runs/{src.id}/copy-batch",
            data={f"runs[{i}][resolution]": "720p" for i in range(2)},
        )
        assert resp.status_code == 201, resp.text
        assert len(spy["process_run"]) == 2

    def test_exhausted_budget_refuses_copy_before_touching_disk(
        self, client, SessionFactory, monkeypatch,
    ):
        s = SessionFactory()
        project = _budget_project(s)
        src = _make_run(s, project.id, status=RunStatus.done, model="seedance")
        s.close()
        _set_headroom(monkeypatch, SessionFactory, 0)

        resp = client.post(
            f"/api/v2/runs/{src.id}/copy",
            data={},
            files=[("reference_files", ("face.jpg", b"\xff\xd8\xff", "image/jpeg"))],
        )
        assert resp.status_code == 429, resp.text
        assert "nothing more can be started" in resp.json()["detail"]

    def test_retry_with_nothing_left_to_bill_is_never_refused(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        """A stitch/delivery-only retry estimates 0 credits — admitted even on
        an exhausted day, so a finished-generation run is never stranded."""
        client, spy = spy_client
        s = SessionFactory()
        project = _budget_project(s)
        run = _make_run(s, project.id, status=RunStatus.failed, model="seedance")
        sd = s.execute(select(SegmentDef).where(SegmentDef.project_id == project.id)).scalar_one()
        result = tempfile.mkstemp(suffix=".mp4")[1]
        _make_run_segment(
            s, run.id, sd.id, status=SegmentStatus.completed, local_result_path=result,
        )
        s.close()
        _set_headroom(monkeypatch, SessionFactory, 0)

        resp = client.post(f"/api/v2/runs/{run.id}/retry")
        assert resp.status_code == 200, resp.text
        assert spy["process_run"] == [run.id]

    def test_retry_that_would_rebill_is_refused_over_budget(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        client, spy = spy_client
        s = SessionFactory()
        project = _budget_project(s)
        run = _make_run(s, project.id, status=RunStatus.failed, model="seedance")
        sd = s.execute(select(SegmentDef).where(SegmentDef.project_id == project.id)).scalar_one()
        _make_run_segment(s, run.id, sd.id, status=SegmentStatus.failed)
        s.close()
        _set_headroom(monkeypatch, SessionFactory, self.COST - 1)

        resp = client.post(f"/api/v2/runs/{run.id}/retry")
        assert resp.status_code == 429, resp.text
        s = SessionFactory()
        assert s.get(Run, run.id).status == RunStatus.failed  # untouched
        s.close()
        assert spy["process_run"] == []

    def test_orphan_retry_of_an_active_run_is_continuation_not_new_spend(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        """Retry on a run stuck in `queued`/`processing` resumes work its launch
        already reserved: no second reservation, never refused."""
        from app.models import KieSpendReservation

        client, spy = spy_client
        s = SessionFactory()
        project = _budget_project(s)
        run = _make_run(s, project.id, status=RunStatus.queued, model="seedance")
        s.close()
        _set_headroom(monkeypatch, SessionFactory, 0)

        resp = client.post(f"/api/v2/runs/{run.id}/retry")
        assert resp.status_code == 200, resp.text
        assert spy["process_run"] == [run.id]
        s = SessionFactory()
        assert s.execute(
            select(KieSpendReservation).where(KieSpendReservation.scope_key == f"run:{run.id}")
        ).first() is None
        s.close()

    def test_over_budget_rerun_is_refused_before_touching_the_override(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        client, spy = spy_client
        s = SessionFactory()
        project = _budget_project(s)
        run = _make_run(s, project.id, status=RunStatus.done, model="seedance")
        sd = s.execute(select(SegmentDef).where(SegmentDef.project_id == project.id)).scalar_one()
        rs = _make_run_segment(
            s, run.id, sd.id, status=SegmentStatus.completed, prompt_override="keep me",
        )
        s.close()
        _set_headroom(monkeypatch, SessionFactory, self.COST - 1)

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun", data={"prompt": "new"},
        )
        assert resp.status_code == 429, resp.text
        s = SessionFactory()
        assert s.get(RunSegment, rs.id).prompt_override == "keep me"
        assert s.get(Run, run.id).status == RunStatus.done
        s.close()
        assert spy["process_run"] == []

    def test_over_budget_multi_rerun_is_refused_before_touching_any_override(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        """The multi rerun prices EVERY listed paid segment, not just one."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        sds = [
            _make_segment_def(s, project.id, i, start_sec=5.0 * i, end_sec=5.0 * i + 5.0)
            for i in range(2)
        ]
        run = _make_run(s, project.id, status=RunStatus.done, model="seedance")
        segs = []
        for sd in sds:
            from app.storage import data_dir_root
            path = os.path.join(data_dir_root(), "fake-results", f"res-{run.id}-{sd.index}.mp4")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(b"x")
            segs.append(_make_run_segment(
                s, run.id, sd.id, index=sd.index, status=SegmentStatus.completed,
                local_result_path=path, prompt_override="keep me",
            ))
        rid, ids = run.id, [rs.id for rs in segs]
        s.close()
        _set_headroom(monkeypatch, SessionFactory, 2 * self.COST - 1)

        resp = client.post(f"/api/v2/runs/{rid}/segments/rerun", json={"segments": [
            {"run_segment_id": ids[0], "prompt": "new"},
            {"run_segment_id": ids[1], "prompt": "new"},
        ]})
        assert resp.status_code == 429, resp.text
        s = SessionFactory()
        assert [s.get(RunSegment, i).prompt_override for i in ids] == ["keep me"] * 2
        assert s.get(Run, rid).status == RunStatus.done
        s.close()
        assert spy["process_run"] == []

    def test_over_budget_preprocess_regen_never_overwrites_a_stored_ref(
        self, spy_client, SessionFactory, monkeypatch,
    ):
        """The exact image price is refused BEFORE uploads are saved: a 429
        after a same-named upload would leave the rolled-back DB pointing at
        a file whose bytes were already replaced."""
        from app import spend
        from app.storage import project_dir

        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id,
            status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=[
                "https://x.test/user.jpg", "https://x.test/t1.jpg", "https://x.test/t2.jpg",
            ],
            preprocess_image_prompts=["", "p1", "p2"],
        )
        s.close()
        refs_dir = os.path.join(project_dir(project.id), "runs", run.id, "references")
        os.makedirs(refs_dir, exist_ok=True)
        stored = os.path.join(refs_dir, "face.jpg")
        with open(stored, "wb") as fh:
            fh.write(b"ORIGINAL")
        # Two prompted trims = two images; room for one only.
        _set_headroom(monkeypatch, SessionFactory, spend.image_credits())

        resp = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            files=[("preprocess_reference_file", ("face.jpg", b"REPLACEMENT", "image/jpeg"))],
        )
        assert resp.status_code == 429, resp.text
        with open(stored, "rb") as fh:
            assert fh.read() == b"ORIGINAL"
        s = SessionFactory()
        assert s.get(Run, run.id).status == RunStatus.preprocess_review
        s.close()
        assert spy["preprocess_run"] == []

        _set_headroom(monkeypatch, SessionFactory, 2 * spend.image_credits())
        resp = client.post(
            f"/api/v2/runs/{run.id}/regenerate-preprocess",
            files=[("preprocess_reference_file", ("face.jpg", b"REPLACEMENT", "image/jpeg"))],
        )
        assert resp.status_code == 200, resp.text
        assert spy["preprocess_run"] == [run.id]

    def test_transcribe_is_admitted_too(
        self, spy_client, SessionFactory, monkeypatch, tmp_path,
    ):
        client, spy = spy_client
        src = tmp_path / "src.mp4"
        src.write_bytes(b"x")
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready, source_local_path=str(src))
        s.close()
        _set_headroom(monkeypatch, SessionFactory, 0)

        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 429, resp.text
        assert "transcription" in resp.json()["detail"]
        assert spy["transcribe_project"] == []
        s = SessionFactory()
        assert s.get(VideoProject, project.id).transcript_status is None
        s.close()

        _set_headroom(monkeypatch, SessionFactory, 20)
        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 202, resp.text

    def test_header_shows_todays_spend_against_the_cap(self, client, SessionFactory, monkeypatch):
        import app.web_v2 as web_v2

        monkeypatch.setattr(web_v2, "_cached_credit_balance", lambda: 1000.0)
        _set_headroom(monkeypatch, SessionFactory, 1_000)
        resp = client.get("/v2/credits-fragment")
        assert resp.status_code == 200
        assert "today: " in resp.text and "/ $" in resp.text

        _set_headroom(monkeypatch, SessionFactory, 0)
        resp = client.get("/v2/credits-fragment")
        assert "budget reached" in resp.text


class TestLocalisationTimedPrompt:
    """AICP-335: the endpoint emits the timed format, per segment clip.

    Through the AICP-338 job path (POST 202 + poll), like every other test of
    this endpoint."""

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return _translate_via_job(client, pid, payload)

    def test_patch_round_trips_v3_fields(self, client, db_session):
        project = _localisation_project(db_session)
        doc = _transcript(
            [_line(1, 0.0, 1.0, "hi", face_visible=False, frame="the phone"),
             _line(2, 1.0, 2.0, "yo", face_visible="maybe")],
            speakers=[{"label": "the woman in the red jacket", "tag": "woman",
                       "kind": "person", "gender": "female", "voice": "low",
                       "description": "d"}, "junk"],
            actions=[{"start": 0.5, "end": 0.6, "speaker": "x", "action": "sits"}],
            brands="not-a-list",
        )
        body = client.patch(f"/api/v2/projects/{project.id}/transcript", json=doc).json()
        tr = body["transcript"]
        assert tr["lines"][0]["face_visible"] is False
        assert tr["lines"][0]["frame"] == "the phone"
        assert "face_visible" not in tr["lines"][1]
        assert tr["speakers"] == [{"label": "the woman in the red jacket", "tag": "WOMAN",
                                   "kind": "person", "gender": "female", "voice": "low",
                                   "description": "d"}]
        assert tr["actions"][0]["action"] == "sits"
        assert tr["brands"] == []

    def test_segment_prompts_are_timed_against_their_own_clip(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Before.", speaker="the woman"),
                    _line(2, 6.0, 8.0, "After.", speaker="off-screen interviewer",
                          on_screen=False),
                    _line(3, 6.2, 9.0, "Under.", speaker="female voice from the phone app",
                          on_screen=False),
                    _line(4, 8.0, 9.5, "End.", speaker="the woman"),
                ],
                speakers=[{"label": "off-screen interviewer", "tag": "INTERVIEWER",
                           "kind": "person", "gender": "female", "voice": "young",
                           "description": "a young woman off screen"}],
            ),
        )
        first = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=5.0)
        second = _make_segment_def(db_session, project.id, 1, start_sec=5.0, end_sec=10.0)

        body = self._post(client, project.id).json()
        prompts = body["segment_prompts"]
        assert "[0.0–4.0] WOMAN: <es> Before." in prompts[first.id]
        assert "She does not speak in this clip." in prompts[first.id]
        assert "[1.0–3.0] INTERVIEWER: <es> After." in prompts[second.id]
        assert "Female, young voice. Never a male voice. She says ONLY ONE line" in prompts[second.id]
        assert "[3.0–4.5] WOMAN: <es> End." in prompts[second.id]
        # Line 3 is a device voice wholly under the others: kept, marked, named.
        assert "[1.2–4.0] APP (BACKGROUND" in prompts[second.id]
        assert "<es> Under." in prompts[second.id]
        assert any(
            "Line 3" in w and "kept in the prompt" in w and "Under." in w
            for w in body["warnings"]
        )

    def test_rolls_warn_that_timecodes_depend_on_the_model(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0,
                          pre_roll_sec=1.0)
        body = self._post(client, project.id).json()
        assert any("depend on the run's model" in w for w in body["warnings"])

    def test_model_field_times_the_clip_and_silences_the_roll_warning(
        self, client, db_session, fake_translate
    ):
        # 0-10s segment with 3s pre-roll at the source start: nothing to trim,
        # so test the trim with a segment later in the video.
        project = _localisation_project(
            db_session, hook_sec=40.0,
            transcript=_transcript([
                _line(1, 20.0, 22.0, "First.", speaker="the woman"),
                _line(2, 32.0, 34.0, "Second.", speaker="the woman"),
            ]),
        )
        a = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=20.0)
        b = _make_segment_def(db_session, project.id, 1, start_sec=20.0, end_sec=35.0,
                              pre_roll_sec=2.0)
        # A 15s-limit model trims the 2s pre-roll off a 15s segment...
        short = next(k for k, m in ai_models.AI_MODELS.items() if m.max_clip_sec == 15)
        body = self._post(client, project.id, model=short).json()
        assert "[0.0–2.0] WOMAN: <es> First." in body["segment_prompts"][b.id]
        assert not any("depend on the run's model" in w for w in body["warnings"])
        # ...a 30s one keeps it, so every timecode moves by 2s.
        body = self._post(client, project.id, model="seedance-2-5").json()
        assert "[2.0–4.0] WOMAN: <es> First." in body["segment_prompts"][b.id]

    def test_unknown_model_is_400_before_any_translation(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        resp = self._post(client, project.id, model="nope")
        assert resp.status_code == 400
        assert "unknown model" in resp.json()["detail"]
        assert fake_translate == []
class TestProjectCaptionDetectorConfig:
    """AICP-343: per-project QA caption detector config on PATCH /projects/{pid}."""

    def test_patch_stores_only_non_defaults_and_normalises(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"caption_detector": {
            "band_y": [0.7, 0.9], "text_color": "#FF0", "outline_color": "Black",
            "threshold": 60}})
        assert resp.status_code == 200, resp.text
        # outline_color "Black" is the default once normalised -> not stored.
        assert resp.json()["caption_detector"] == {
            "band_y": [0.7, 0.9], "text_color": "#ffff00", "threshold": 60}
        db_session.refresh(project)
        assert project.caption_detector == resp.json()["caption_detector"]

        # Omitted key leaves it alone.
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "x"})
        assert resp.json()["caption_detector"]["threshold"] == 60

        # Explicit null, {} and an all-defaults object all reset to NULL.
        for body in (None, {}, {"threshold": 40, "text_color": "white"}):
            client.patch(f"/api/v2/projects/{project.id}",
                         json={"caption_detector": {"threshold": 99}})
            resp = client.patch(f"/api/v2/projects/{project.id}",
                                json={"caption_detector": body})
            assert resp.status_code == 200, resp.text
            assert resp.json()["caption_detector"] is None
            db_session.refresh(project)
            assert project.caption_detector is None
            # SQL NULL, not a JSON 'null' value.
            from sqlalchemy import text as sa_text
            raw = db_session.execute(sa_text(
                "SELECT caption_detector IS NULL FROM video_projects WHERE id = :i"),
                {"i": project.id}).scalar()
            assert raw in (1, True)

    @pytest.mark.parametrize("bad", [
        {"band_y": [0.9, 0.7]},
        {"band_y": [0.5, 1.5]},
        {"band_y": [0.5]},
        {"text_color": "#12345"},
        {"text_color": "rgb(1,2,3)"},
        {"outline_color": 5},
        {"outline_style": "shadow"},
        {"threshold": 0},
        {"threshold": 10 ** 9},
        {"text_tolerance": 999},
        {"glyph_height_px": [60, 12]},
        {"min_aspect": float("inf")},
        {"unknown_key": 1},
        "white",
    ])
    def test_patch_rejects_invalid_config(self, client, db_session, bad):
        project = _make_project(db_session, status=ProjectStatus.ready,
                                caption_detector={"threshold": 77})
        resp = client.patch(f"/api/v2/projects/{project.id}",
                            content=json.dumps({"caption_detector": bad}, allow_nan=True),
                            headers={"Content-Type": "application/json"})
        # 400 from update_project's validation; a non-object is FastAPI's 422.
        assert resp.status_code == (422 if bad == "white" else 400), resp.text
        if bad != "white":
            assert "invalid caption detector config" in resp.json()["detail"]
        db_session.refresh(project)
        assert project.caption_detector == {"threshold": 77}

    def test_get_project_returns_it(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        assert client.get(f"/api/v2/projects/{project.id}").json()["caption_detector"] is None


# ---------------------------------------------------------------------------
# AICP-358: segment caption fill (source + masked VOID inpaint)
# ---------------------------------------------------------------------------


def _burn_word(frame, word, org, scale=1.6):
    """White glyphs with a ~5 px black outline, like a burned caption."""
    import cv2
    import numpy as np

    m = np.zeros(frame.shape[:2], np.uint8)
    cv2.putText(m, word, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 5)
    ring = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) & ~m
    frame[ring > 0] = 0
    frame[m > 0] = 255
    return frame


def _caption_video(path: str, *, n: int = 90, w: int = 540, h: int = 960, words=("HELLO", "WORLD")):
    """A 9:16 30 fps clip with burned word-by-word captions: white glyphs with a
    thick black outline at y ~0.62 over a grey gradient (frames streamed)."""
    import cv2
    import numpy as np
    from app.pipeline_segment_fill import Encoder

    enc = Encoder(path, w, h, "30")
    ramp = np.tile(np.linspace(60, 160, w, dtype=np.uint8), (h, 1))
    for i in range(n):
        f = cv2.merge([ramp, ramp, ramp]).copy()
        word = words[(i // 15) % len(words)]
        org = (int(w * 0.30), int(h * 0.64))
        _burn_word(f, word, org)
        enc.write(f)
    enc.close()


def _plain_video(path: str, *, n: int, w: int = 720, h: int = 1280, fps: str = "12"):
    import numpy as np
    from app.pipeline_segment_fill import Encoder

    enc = Encoder(path, w, h, fps)
    for _ in range(n):
        enc.write(np.full((h, w, 3), 110, np.uint8))
    enc.close()


class TestSegmentFillPure:
    def test_void_sampling_counts_and_cover(self):
        from app.pipeline_segment_fill import FillError, frame_to_void, void_sampling

        for n, m in ((240, 197), (90, 85), (69, 69), (300, 197)):
            s = void_sampling(n)
            assert len(s) == m and (m - 69) % 8 == 0
            idx = frame_to_void(n, s)
            assert idx[0] == 0 and idx[-1] == m - 1 and sorted(set(idx)) == list(range(m))
            assert s[0][0] == 0 and s[-1][1] == n - 1
        with pytest.raises(FillError):
            void_sampling(68)

    def test_validate_box(self):
        from app.pipeline_segment_fill import FillError, validate_box

        assert validate_box([0.1, 0.5, 0.9, 0.7]) == [0.1, 0.5, 0.9, 0.7]
        for bad in ([0.5, 0.5, 0.4, 0.7], [0, 0, float("nan"), 1], [0.1, 0.1, 0.11, 0.9], [1, 2]):
            with pytest.raises(FillError):
                validate_box(bad)

    def test_void_endpoint_contract_and_reframe_unchanged(self):
        from app import fal_models

        void = fal_models.endpoint(fal_models.VOID_ENDPOINT)
        assert void.family == "inpaint" and void.media_check == "frames"
        assert void.worst_case_usd("720p", 8.0) == 0.10 and void.send_resolution is False
        with pytest.raises(fal_models.UntariffedResolution):
            void.worst_case_usd("1080p", 8.0)
        rf = fal_models.endpoint("minimax/h3-max/reframe-video")
        assert rf.flat_usd is None and rf.media_check == "duration" and rf.send_resolution
        assert rf.worst_case_usd("480p", 10.0) == 1.0

    def test_project_allowed_matrix(self, monkeypatch):
        from app import pipeline_provider as pp

        monkeypatch.setattr(pp.settings, "FAL_PROJECT_IDS", "allow-me")
        assert pp.project_allowed("x", "reframe") and pp.project_allowed("x", "reframe", "inpaint")
        assert pp.project_allowed("allow-me", "face_swap", "reframe")
        assert pp.project_allowed("x", "subtitle_removal", "inpaint")
        assert not pp.project_allowed("x", "subtitle_removal", "reframe")
        assert not pp.project_allowed("x", "subtitle_removal")
        assert not pp.project_allowed("x", "face_swap", "inpaint")

    def test_verify_media_contract(self, tmp_path):
        from app import pipeline_provider as pp

        p = str(tmp_path / "v.mp4")
        _plain_video(p, n=24, w=64, h=112, fps="12")  # 2.0 s
        assert pp._verify_media(p, 2.0) is None                      # legacy: duration
        assert pp._verify_media(p, 5.0) is not None
        assert pp._verify_media(p, 99.0, {"kind": "frames", "frames": 24}) is None
        assert "frames" in pp._verify_media(p, 2.0, {"kind": "frames", "frames": 25})
        assert "unknown" in pp._verify_media(p, 2.0, {"kind": "seconds", "frames": 24})

    def test_text_mask_stays_inside_the_box(self):
        import cv2
        import numpy as np
        from app.pipeline_segment_fill import text_mask

        f = np.full((960, 540, 3), 100, np.uint8)
        _burn_word(f, "WORD", (150, 615))
        _burn_word(f, "UI", (150, 200))
        m = text_mask(f, [0.1, 0.55, 0.9, 0.7])
        assert m[560:640, 140:400].sum() > 200
        assert m[:500].sum() == 0  # the "UI" word is outside the box


class TestSegmentFillFlow:
    """Preview → start → composite → promote against SQLite, with the provider
    result faked on disk (no Fal call)."""

    @pytest.fixture()
    def world(self, SessionFactory, monkeypatch):
        from contextlib import contextmanager

        import app.pipeline_segment_fill as psf
        import app.tasks as tasks_mod
        from app import pipeline_provider as pp
        from app import storage

        @contextmanager
        def _sess():
            s = SessionFactory()
            try:
                yield s
            finally:
                s.close()

        monkeypatch.setattr(psf, "get_session", _sess)
        calls = {"prepare": [], "composite": [], "fal": []}
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_prepare",
                            lambda j: calls["prepare"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_composite",
                            lambda j: calls["composite"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_fal_job",
                            lambda fn, t, **kw: calls["fal"].append((fn, t)))
        monkeypatch.setattr(pp.settings, "FAL_ENABLED", True)
        monkeypatch.setattr(pp.settings, "FAL_API_KEY", "test-key")
        monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)

        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready, project_type="subtitle_removal")
        src = os.path.join(storage.project_dir(project.id), "source.mp4")
        _caption_video(src, n=150)  # 5.0 s
        project.source_local_path = src
        sd0 = _make_segment_def(s, project.id, 0, start_sec=0.0, end_sec=3.0)
        sd1 = _make_segment_def(s, project.id, 1, start_sec=3.0, end_sec=5.0)
        run = _make_run(
            s, project.id, status=RunStatus.done, model="gemini-omni", audio_mode="original",
            result_local_path="/data/fake/final.mp4", result_gdrive_file_id="DRIVE1",
            segment_snapshot=[{"id": sd.id, "index": sd.index, "action": "swap",
                               "start_sec": sd.start_sec, "end_sec": sd.end_sec,
                               "pre_roll_sec": 0.0, "post_roll_sec": 0.0} for sd in (sd0, sd1)],
        )
        r_dir = storage.run_results_dir(run.id, project.id)
        segs = []
        for sd in (sd0, sd1):
            path = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
            with open(path, "wb") as fh:
                fh.write(f"omni-{sd.index}".encode())
            segs.append(_make_run_segment(
                s, run.id, sd.id, index=sd.index, status=SegmentStatus.completed,
                local_result_path=path, seedance_task_id=f"omni-task-{sd.index}",
                cost_credits=168.0))
        s.commit()
        ids = {"project": project.id, "run": run.id, "rs": segs[0].id, "rs1": segs[1].id}
        s.close()
        return ids, calls

    def _preview(self, client, ids, **body):
        resp = client.post(f"/api/v2/runs/{ids['run']}/segments/{ids['rs']}/fill/preview", json=body)
        assert resp.status_code == 202, resp.text
        return resp.json()["id"]

    def _prepare(self, fid):
        from app.pipeline_segment_fill import segment_fill_prepare

        segment_fill_prepare(fid)

    def _start(self, client, ids, fid, usd=0.10, **kw):
        return client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/start",
                           json={"approved_usd": usd, **kw})

    def _fake_provider_done(self, SessionFactory, fid):
        from app.provider_models import ProviderTask
        from app.segment_fill_models import RunSegmentFillJob

        s = SessionFactory()
        job = s.get(RunSegmentFillJob, fid)
        task = s.get(ProviderTask, job.provider_task_id)
        _plain_video(task.destination_path, n=job.media["void_frames"])
        task.status, task.result_path = "done", task.destination_path
        task.reservation_held = False
        s.commit()
        s.close()

    def test_preview_refuses_other_project_types(self, spy_client, SessionFactory, world):
        ids, _ = world
        client, _ = spy_client
        s = SessionFactory()
        s.get(VideoProject, ids["project"]).project_type = "face_swap"
        s.commit()
        s.close()
        resp = client.post(f"/api/v2/runs/{ids['run']}/segments/{ids['rs']}/fill/preview", json={})
        assert resp.status_code == 422

    def test_full_flow_applies_composite_and_keeps_identity(self, spy_client, SessionFactory, world):
        from app.models import RunSegmentTake
        from app.provider_models import ProviderTask
        from app.pipeline_segment_fill import probe_video, segment_fill_composite
        from app.segment_fill_models import RunSegmentFillJob

        ids, calls = world
        client, spy = spy_client
        fid = self._preview(client, ids)
        assert calls["prepare"] == [fid]
        self._prepare(fid)
        s = SessionFactory()
        job = s.get(RunSegmentFillJob, fid)
        assert job.status == "ready", (job.error_code, job.error_message)
        assert job.box_source == "measured" and 0.5 < job.mask_box[1] < job.mask_box[3] < 0.75
        assert job.media["frames"] == 90 and job.media["void_frames"] == 85
        assert job.media["frames_masked"] >= 80 and os.path.isfile(job.preview_path)
        assert job.estimated_usd == 0.10
        s.close()

        assert self._start(client, ids, fid, usd=0.05).status_code == 409  # below the reservation
        resp = self._start(client, ids, fid)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "admitted"
        s = SessionFactory()
        job = s.get(RunSegmentFillJob, fid)
        task = s.get(ProviderTask, job.provider_task_id)
        assert calls["fal"] == [("fal_submit", task.id)]
        assert task.request_spec["verify"] == {"kind": "frames", "frames": 85}
        assert task.request_spec["send_resolution"] is False
        assert {i["param"] for i in task.input_refs} == {"video_url", "quad_mask_video_url"}
        assert task.owner_kind == "segment_fill" and task.owner_run_id == ids["run"]
        s.close()

        # The active fill blocks every other result mutation of the run.
        assert client.post(f"/api/v2/runs/{ids['run']}/segments/{ids['rs1']}/rerun",
                           data={"prompt": "x"}).status_code == 409
        assert client.post(f"/api/v2/runs/{ids['run']}/retry").status_code == 409

        self._fake_provider_done(SessionFactory, fid)
        segment_fill_composite(fid)
        s = SessionFactory()
        job = s.get(RunSegmentFillJob, fid)
        assert job.status == "applied", (job.error_code, job.error_message)
        rs = s.get(RunSegment, ids["rs"])
        run = s.get(Run, ids["run"])
        assert rs.local_result_path == job.composite_path
        assert rs.seedance_task_id is None and rs.status == SegmentStatus.completed
        assert probe_video(job.composite_path)["frames"] == 90
        assert run.status == RunStatus.queued and run.result_local_path is None
        assert spy["process_run"] == [ids["run"]]
        (take,) = s.execute(select(RunSegmentTake).where(
            RunSegmentTake.run_segment_id == ids["rs"])).scalars().all()
        assert take.id == job.replaced_take_id and take.seedance_task_id == "omni-task-0"
        assert open(take.result_path, "rb").read() == b"omni-0"
        # Identity: archiving the composite later must NOT reuse the Omni take.
        import app.api_v2 as api_v2_module
        new_take = api_v2_module._archive_current_take(run, rs, s)
        assert new_take is not None and new_take.id != take.id
        assert new_take.seedance_task_id is None
        s.rollback()
        s.close()

    def test_start_refuses_a_stale_preview(self, spy_client, SessionFactory, world):
        from app.segment_fill_models import RunSegmentFillJob

        ids, _ = world
        client, _ = spy_client
        fid = self._preview(client, ids)
        self._prepare(fid)
        s = SessionFactory()
        rs = s.get(RunSegment, ids["rs"])
        with open(rs.local_result_path, "ab") as fh:
            fh.write(b"changed")
        s.close()
        resp = self._start(client, ids, fid)
        assert resp.status_code == 409 and "segment.identity" in resp.text
        s = SessionFactory()
        assert s.get(RunSegmentFillJob, fid).status == "canceled"
        s.close()

    def test_start_refuses_a_paid_restitch(self, spy_client, SessionFactory, world):
        ids, _ = world
        client, _ = spy_client
        fid = self._preview(client, ids)
        self._prepare(fid)
        s = SessionFactory()
        s.get(RunSegment, ids["rs1"]).source_fallback = True
        s.commit()
        s.close()
        # source_fallback on the OTHER segment changes nothing in the base, but
        # the re-stitch would resubmit it.
        resp = self._start(client, ids, fid)
        assert resp.status_code == 409 and "have no finished result" in resp.text

    def test_reconcile_provider_outcomes(self, spy_client, SessionFactory, world):
        from app.provider_models import ProviderTask
        from app.pipeline_segment_fill import reconcile_fill_jobs
        from app.segment_fill_models import RunSegmentFillJob

        ids, calls = world
        client, _ = spy_client
        fid = self._preview(client, ids)
        self._prepare(fid)
        assert self._start(client, ids, fid).status_code == 200
        s = SessionFactory()
        task = s.get(ProviderTask, s.get(RunSegmentFillJob, fid).provider_task_id)
        task.status, task.reservation_held = "refused", True   # still holds money
        s.commit()
        s.close()
        reconcile_fill_jobs(set(), grace_sec=0)
        s = SessionFactory()
        assert s.get(RunSegmentFillJob, fid).status == "admitted"
        task = s.get(ProviderTask, s.get(RunSegmentFillJob, fid).provider_task_id)
        task.reservation_held = False                         # operator closed it
        s.commit()
        s.close()
        reconcile_fill_jobs(set(), grace_sec=0)
        s = SessionFactory()
        job = s.get(RunSegmentFillJob, fid)
        assert job.status == "failed" and job.error_code == "provider_refused"
        s.close()
        # The run is free again.
        assert client.post(f"/api/v2/runs/{ids['run']}/segments/{ids['rs1']}/rerun",
                           data={"prompt": "x"}).status_code == 200

    def test_reconcile_enqueues_composite_for_a_done_task(self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import reconcile_fill_jobs

        ids, calls = world
        client, _ = spy_client
        fid = self._preview(client, ids)
        self._prepare(fid)
        assert self._start(client, ids, fid).status_code == 200
        self._fake_provider_done(SessionFactory, fid)
        reconcile_fill_jobs(set(), grace_sec=0)
        assert calls["composite"] == [fid]

    def test_run_page_shows_the_fill_block(self, spy_client, SessionFactory, world):
        ids, _ = world
        client, _ = spy_client
        page = client.get(f"/v2/runs/{ids['run']}/status-fragment")
        assert page.status_code == 200 and "Caption fill (VOID)" in page.text
        assert "Preview mask (free)" in page.text
        fid = self._preview(client, ids)
        self._prepare(fid)
        page = client.get(f"/v2/runs/{ids['run']}/status-fragment")
        assert "Start (reserve $0.10)" in page.text and "mask preview" in page.text
        assert client.get(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/preview.jpg").status_code == 200
        listing = client.get(f"/api/v2/runs/{ids['run']}/segments/fill").json()
        assert listing[0]["id"] == fid and listing[0]["status"] == "ready"
        assert client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/cancel").status_code == 200

    def test_delivery_state_needs_the_composite_to_be_current(self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import delivery_state, segment_fill_composite
        from app.segment_fill_models import RunSegmentFillJob

        ids, _ = world
        client, _ = spy_client
        fid = self._preview(client, ids)
        self._prepare(fid)
        assert self._start(client, ids, fid).status_code == 200
        self._fake_provider_done(SessionFactory, fid)
        segment_fill_composite(fid)
        s = SessionFactory()
        job, run, rs = (s.get(RunSegmentFillJob, fid), s.get(Run, ids["run"]),
                        s.get(RunSegment, ids["rs"]))
        assert delivery_state(job, run, rs) == "queued"
        from app import storage
        final = os.path.join(storage.run_results_dir(run.id, run.project_id), "final.mp4")
        open(final, "wb").write(b"new final")
        run.status, run.result_local_path = RunStatus.done, final
        assert delivery_state(job, run, rs) == "delivered"
        copy = os.path.join(os.path.dirname(final), "restored-copy.mp4")
        import shutil
        shutil.copyfile(job.composite_path, copy)
        rs.local_result_path = copy                           # restore of a take of it
        assert delivery_state(job, run, rs) == "delivered"
        rs.local_result_path = "/elsewhere/other-take.mp4"   # a restore replaced it
        assert delivery_state(job, run, rs) == "superseded"
        rs.local_result_path = job.composite_path
        run.result_local_path = "/elsewhere/assistant-version.mp4"
        assert delivery_state(job, run, rs) == "republished"
        s.rollback()
        s.close()


SAFETY_MSG = ("failed after 3 attempt(s): Request blocked: The input content was flagged by "
              "safety filters as potentially dangerous and cannot be processed.")


class TestSegmentFillBatch:
    """Batch VOID fills (2026-11-03, docs/segment-fill-batch-design.md): two
    segments previewed in parallel, ONE paid start (one ledger pass), members
    composited independently, ONE promote / re-stitch / re-delivery; a failed
    member does not block the others; the per-run allowance is enforced in the
    admission; one live batch per run; crash / hold recovery."""

    @pytest.fixture()
    def world(self, SessionFactory, monkeypatch):
        from contextlib import contextmanager

        import app.pipeline_segment_fill as psf
        import app.tasks as tasks_mod
        from app import pipeline_provider as pp
        from app import storage

        @contextmanager
        def _sess():
            s = SessionFactory()
            try:
                yield s
            finally:
                s.close()

        monkeypatch.setattr(psf, "get_session", _sess)
        calls = {"prepare": [], "composite": [], "fal": [], "promote": []}
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_prepare",
                            lambda j: calls["prepare"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_composite",
                            lambda j: calls["composite"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_batch_promote",
                            lambda b: calls["promote"].append(b))
        monkeypatch.setattr(tasks_mod, "enqueue_fal_job",
                            lambda fn, t, **kw: calls["fal"].append((fn, t)))
        monkeypatch.setattr(pp.settings, "FAL_ENABLED", True)
        monkeypatch.setattr(pp.settings, "FAL_API_KEY", "test-key")
        monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)

        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready, project_type="subtitle_removal")
        src = os.path.join(storage.project_dir(project.id), "source.mp4")
        _caption_video(src, n=225)  # 7.5 s
        project.source_local_path = src
        sds = [_make_segment_def(s, project.id, i, start_sec=2.5 * i, end_sec=2.5 * (i + 1))
               for i in range(3)]
        run = _make_run(
            s, project.id, status=RunStatus.done, model="gemini-omni", audio_mode="original",
            result_local_path="/data/fake/final.mp4", result_gdrive_file_id="DRIVE1",
            segment_snapshot=[{"id": sd.id, "index": sd.index, "action": "swap",
                               "start_sec": sd.start_sec, "end_sec": sd.end_sec,
                               "pre_roll_sec": 0.0, "post_roll_sec": 0.0} for sd in sds],
        )
        r_dir = storage.run_results_dir(run.id, project.id)
        segs = []
        for sd in sds:
            path = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
            with open(path, "wb") as fh:
                fh.write(f"omni-{sd.index}".encode())
            segs.append(_make_run_segment(
                s, run.id, sd.id, index=sd.index, status=SegmentStatus.completed,
                local_result_path=path, seedance_task_id=f"omni-task-{sd.index}",
                cost_credits=168.0))
        s.commit()
        ids = {"project": project.id, "run": run.id, "rs": [x.id for x in segs]}
        s.close()
        return ids, calls

    def _ready(self, client, ids, idxs):
        from app.pipeline_segment_fill import segment_fill_prepare

        resp = client.post(f"/api/v2/runs/{ids['run']}/segments/fill/batch/preview",
                           json={"run_segment_ids": [ids["rs"][i] for i in idxs]})
        assert resp.status_code == 202, resp.text
        fids = [j["id"] for j in resp.json()]
        assert all(j["segment_bound"] for j in resp.json())
        for fid in fids:
            segment_fill_prepare(fid)
        return fids

    def _start(self, client, ids, fids, usd):
        return client.post(f"/api/v2/runs/{ids['run']}/segments/fill/batch/start",
                           json={"fill_ids": fids, "approved_usd": usd})

    def _done(self, SessionFactory, fid, status="done"):
        from app.provider_models import ProviderTask
        from app.segment_fill_models import RunSegmentFillJob

        s = SessionFactory()
        job = s.get(RunSegmentFillJob, fid)
        task = s.get(ProviderTask, job.provider_task_id)
        if status == "done":
            _plain_video(task.destination_path, n=job.media["void_frames"])
            task.status, task.result_path = "done", task.destination_path
            task.reservation_held = False
        else:
            task.status = status
            task.reservation_held = status in ("refused", "submit_unknown")
        s.commit()
        s.close()

    def test_two_members_one_start_one_promote(self, spy_client, SessionFactory, world):
        from app.models import RunSegmentTake
        from app.pipeline_segment_fill import segment_fill_composite
        from app.provider_models import ProviderTask
        from app.segment_fill_models import RunSegmentFillBatch, RunSegmentFillJob
        from app.spend import KieSpendReservation

        ids, calls = world
        client, spy = spy_client
        fids = self._ready(client, ids, (0, 1))
        assert sorted(calls["prepare"]) == sorted(fids)          # previews in parallel
        assert self._start(client, ids, fids, 0.15).status_code == 409   # below 2 x $0.10
        resp = self._start(client, ids, fids, 0.20)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "active" and body["authorization"] == "operator"
        assert {m["status"] for m in body["members"]} == {"admitted"}
        s = SessionFactory()
        batch = s.get(RunSegmentFillBatch, body["id"])
        assert batch.reserved_mcredits == 40_000 and batch.reserved_usd == 0.2
        tasks = [s.get(ProviderTask, s.get(RunSegmentFillJob, f).provider_task_id) for f in fids]
        assert sorted(t for _fn, t in calls["fal"]) == sorted(t.id for t in tasks)
        for t in tasks:                       # the ledger holds every member's reservation
            res = s.execute(select(KieSpendReservation).where(
                KieSpendReservation.scope_key == f"fal:{t.id}")).scalar_one()
            assert res.reserved_mcredits == 20_000
        s.close()
        # one live batch per run; a single start is refused meanwhile too
        more = self._ready(client, ids, (2,))
        assert self._start(client, ids, more, 0.10).status_code == 409
        assert client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{more[0]}/start",
                           json={"approved_usd": 0.10}).status_code == 409
        # the batch blocks every other result mutation of the run
        assert client.post(f"/api/v2/runs/{ids['run']}/retry").status_code == 409

        # member 0 finishes first: composited, waits; member 1 then promotes BOTH
        self._done(SessionFactory, fids[0])
        segment_fill_composite(fids[0])
        s = SessionFactory()
        assert s.get(RunSegmentFillJob, fids[0]).status == "composited"
        assert s.get(RunSegmentFillBatch, body["id"]).status == "active"
        s.close()
        assert spy["process_run"] == []
        self._done(SessionFactory, fids[1])
        segment_fill_composite(fids[1])
        s = SessionFactory()
        batch = s.get(RunSegmentFillBatch, body["id"])
        assert batch.status == "applied", (batch.error_code, batch.error_message)
        assert batch.summary == {fids[0]: "applied", fids[1]: "applied"}
        run = s.get(Run, ids["run"])
        assert run.status == RunStatus.queued and run.result_local_path is None
        assert spy["process_run"] == [ids["run"]]                # ONE re-stitch
        for fid in fids:
            job = s.get(RunSegmentFillJob, fid)
            rs = s.get(RunSegment, job.run_segment_id)
            assert job.status == "applied" and rs.local_result_path == job.composite_path
            assert rs.seedance_task_id is None
            take = s.get(RunSegmentTake, job.replaced_take_id)
            assert take is not None and take.run_segment_id == rs.id
        s.close()

    def test_a_failed_member_does_not_block_the_others(self, spy_client, SessionFactory,
                                                        world):
        from app.pipeline_segment_fill import reconcile_fill_jobs, segment_fill_composite
        from app.provider_models import ProviderTask
        from app.segment_fill_models import RunSegmentFillBatch, RunSegmentFillJob

        ids, calls = world
        client, spy = spy_client
        fids = self._ready(client, ids, (0, 1))
        bid = self._start(client, ids, fids, 0.20).json()["id"]
        # member 1 fails at Fal (its reservation released)
        s = SessionFactory()
        t1 = s.get(ProviderTask, s.get(RunSegmentFillJob, fids[1]).provider_task_id)
        t1.status, t1.reservation_held = "failed", False
        t1.updated_at = t1.updated_at
        s.commit()
        s.close()
        reconcile_fill_jobs(set(), grace_sec=0)
        s = SessionFactory()
        assert s.get(RunSegmentFillJob, fids[1]).status == "failed"
        s.close()
        self._done(SessionFactory, fids[0])
        segment_fill_composite(fids[0])
        s = SessionFactory()
        b = s.get(RunSegmentFillBatch, bid)
        assert b.status == "applied" and b.summary[fids[0]] == "applied"
        assert b.summary[fids[1]].startswith("failed")
        rs1 = s.get(RunSegment, ids["rs"][1])
        assert rs1.seedance_task_id == "omni-task-1"            # untouched
        s.close()
        assert spy["process_run"] == [ids["run"]]

    def test_previews_are_bound_to_their_segment_not_to_the_final(self, spy_client,
                                                                  SessionFactory, world):
        """Another segment's fill re-delivering the final does not make a preview
        stale; a change of its OWN segment does (and cancels it)."""
        from app.segment_fill_models import RunSegmentFillJob

        ids, calls = world
        client, _ = spy_client
        fids = self._ready(client, ids, (0, 1))
        s = SessionFactory()
        run = s.get(Run, ids["run"])
        run.result_local_path = "/data/fake/final-after-another-fill.mp4"   # a new final
        rs1 = s.get(RunSegment, ids["rs"][1])
        with open(rs1.local_result_path, "wb") as fh:            # segment 1 re-generated
            fh.write(b"omni-1-rerun-with-more-bytes")
        s.commit()
        s.close()
        resp = self._start(client, ids, fids, 0.20)
        assert resp.status_code == 409 and fids[1] in resp.json()["detail"]
        s = SessionFactory()
        assert s.get(RunSegmentFillJob, fids[1]).status == "canceled"
        assert s.get(RunSegmentFillJob, fids[0]).status == "ready"
        s.close()
        assert self._start(client, ids, [fids[0]], 0.10).status_code == 200

    def test_the_allowance_is_enforced_in_the_admission(self, SessionFactory, spy_client,
                                                        world):
        import app.pipeline_segment_fill as psf
        from app.segment_fill_models import RunSegmentFillBatch

        ids, _ = world
        client, _ = spy_client
        fids = self._ready(client, ids, (0, 1, 2))
        s = SessionFactory()
        s.get(VideoProject, ids["project"]).void_auto_usd_per_run = 0.25
        s.commit()
        run = s.get(Run, ids["run"])
        with pytest.raises(psf.AllowanceExceeded):
            psf.admit_batch_start(s, run, fids, approved_usd=0.30, authorization="auto")
        s.rollback()
        s.close()
        s = SessionFactory()
        run = s.get(Run, ids["run"])
        batch, members = psf.admit_batch_start(s, run, fids[:2], approved_usd=0.20,
                                               authorization="auto")
        s.commit()
        assert psf.auto_spent_usd(s, ids["run"]) == pytest.approx(0.20)
        # the allowance is per RUN and survives the batch: only $0.05 left now
        s.get(RunSegmentFillBatch, batch.id).status = "applied"
        for j, _t, _c in members:
            j.status = "applied"
        s.commit()
        run = s.get(Run, ids["run"])
        with pytest.raises(psf.AllowanceExceeded):
            psf.admit_batch_start(s, run, fids[2:], approved_usd=0.10, authorization="auto")
        s.rollback()
        # an operator-approved batch is not bound by it
        run = s.get(Run, ids["run"])
        psf.admit_batch_start(s, run, fids[2:], approved_usd=0.10)
        s.rollback()
        s.close()

    def test_the_daily_cap_refuses_the_whole_batch(self, spy_client, SessionFactory, world,
                                                   monkeypatch):
        from app import spend
        from app.provider_models import ProviderTask
        from app.segment_fill_models import RunSegmentFillBatch, RunSegmentFillJob

        ids, _ = world
        client, _ = spy_client
        fids = self._ready(client, ids, (0, 1))
        monkeypatch.setattr(spend, "cap_credits", lambda: 30.0)     # room for one, not two
        resp = self._start(client, ids, fids, 0.20)
        assert resp.status_code == 409 and "daily budget" in resp.json()["detail"]
        s = SessionFactory()
        assert s.execute(select(RunSegmentFillBatch).where(
            RunSegmentFillBatch.run_id == ids["run"])).scalars().all() == []
        # (no ledger row for either member; on PostgreSQL the provider rows roll
        # back with the transaction -- pysqlite turns a first SAVEPOINT's release
        # into a commit, so they are not checked here)
        tasks = s.execute(select(ProviderTask.id).where(
            ProviderTask.owner_run_id == ids["run"])).scalars().all()
        from app.spend import KieSpendReservation
        assert not s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key.in_([f"fal:{t}" for t in tasks]))).scalars().all()
        assert {s.get(RunSegmentFillJob, f).status for f in fids} == {"ready"}
        s.close()

    def test_a_crashed_promote_is_retried_then_failed(self, spy_client, SessionFactory, world):
        from datetime import timedelta

        import app.pipeline_segment_fill as psf
        from app.segment_fill_models import RunSegmentFillBatch, RunSegmentFillJob

        ids, calls = world
        client, spy = spy_client
        fids = self._ready(client, ids, (0,))
        bid = self._start(client, ids, fids, 0.10).json()["id"]
        self._done(SessionFactory, fids[0])
        # composite without the inline promote: claim + mark only
        claimed = psf._claim(fids[0], "compositing", from_status="admitted")
        token, job = claimed
        work = os.path.join(psf._job_dir(job), f"comp-{token}")
        os.makedirs(work)
        out = psf._composite_into(job, s_task_result(SessionFactory, fids[0]), work)
        assert psf._mark_composited(fids[0], token, out, work)
        # a promote claimed the batch and died
        assert psf._claim_batch(bid) is not None
        s = SessionFactory()
        b = s.get(RunSegmentFillBatch, bid)
        assert b.status == "promoting"
        b.claim_expires_at = psf._now() - timedelta(seconds=1)
        s.commit()
        s.close()
        psf.reconcile_fill_jobs(set(), grace_sec=0)
        assert bid in calls["promote"]        # (the shared test DB may hold other batches)
        assert psf.promote_batch(bid) == "applied"
        assert spy["process_run"] == [ids["run"]]
        # bounded: a batch whose promote keeps dying ends failed, nothing applied
        s = SessionFactory()
        b = s.get(RunSegmentFillBatch, bid)
        b.status, b.attempts = "promoting", psf.MAX_ATTEMPTS
        b.claim_expires_at = psf._now() - timedelta(seconds=1)
        s.commit()
        s.close()
        assert psf._claim_batch(bid) is None
        s = SessionFactory()
        assert s.get(RunSegmentFillBatch, bid).status == "failed"
        s.close()

    def test_a_held_member_is_detached_after_the_hold(self, spy_client, SessionFactory, world,
                                                      monkeypatch):
        import app.pipeline_segment_fill as psf
        from app.config import settings
        from app.segment_fill_models import RunSegmentFillBatch, RunSegmentFillJob

        ids, calls = world
        client, spy = spy_client
        fids = self._ready(client, ids, (0, 1))
        bid = self._start(client, ids, fids, 0.20).json()["id"]
        self._done(SessionFactory, fids[0])
        psf.segment_fill_composite(fids[0])                        # composited, waits
        self._done(SessionFactory, fids[1], status="submit_unknown")
        psf.reconcile_fill_jobs(set(), grace_sec=0)
        s = SessionFactory()
        assert s.get(RunSegmentFillJob, fids[1]).status == "admitted"   # within the hold
        s.close()
        monkeypatch.setattr(settings, "SEGMENT_FILL_BATCH_HOLD_SEC", 0)
        psf.reconcile_fill_jobs(set(), grace_sec=0)
        s = SessionFactory()
        m1 = s.get(RunSegmentFillJob, fids[1])
        assert m1.status == "failed" and m1.error_code == "held_detached"
        s.close()
        assert bid in calls["promote"]        # (the shared test DB may hold other batches)
        assert psf.promote_batch(bid) == "applied"
        s = SessionFactory()
        assert s.get(RunSegmentFillBatch, bid).summary[fids[0]] == "applied"
        s.close()


def _caption_then_plain_video(path: str, *, n_captions: int, n: int):
    """Captions on the first *n_captions* frames, a plain picture after."""
    import cv2
    import numpy as np
    from app.pipeline_segment_fill import Encoder

    w, h = 540, 960
    enc = Encoder(path, w, h, "30")
    ramp = np.tile(np.linspace(60, 160, w, dtype=np.uint8), (h, 1))
    for i in range(n):
        f = cv2.merge([ramp, ramp, ramp]).copy()
        if i < n_captions:
            _burn_word(f, ("HELLO", "WORLD")[(i // 15) % 2], (int(w * 0.30), int(h * 0.64)))
        enc.write(f)
    enc.close()


class TestVoidSourceMode:
    """VOID from the source as the project's caption removal method (2026-11-03,
    docs/segment-fill-batch-design.md §7): the run sends nothing to the model,
    is parked ``incomplete`` with its free source previews, the caption-less
    segment goes in as is, ONE batch fills the others, and the run is requeued
    once -- for its stitch + first delivery -- when every segment has a fill."""

    @pytest.fixture()
    def world(self, SessionFactory, monkeypatch):
        from contextlib import contextmanager

        import app.pipeline_segment_fill as psf
        import app.tasks as tasks_mod
        from app import pipeline_provider as pp
        from app import storage

        @contextmanager
        def _sess():
            s = SessionFactory()
            try:
                yield s
            finally:
                s.close()

        monkeypatch.setattr(psf, "get_session", _sess)
        calls = {"prepare": [], "composite": [], "fal": [], "promote": []}
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_prepare",
                            lambda j: calls["prepare"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_composite",
                            lambda j: calls["composite"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_batch_promote",
                            lambda b: calls["promote"].append(b))
        monkeypatch.setattr(tasks_mod, "enqueue_fal_job",
                            lambda fn, t, **kw: calls["fal"].append((fn, t)))
        monkeypatch.setattr(pp.settings, "FAL_ENABLED", True)
        monkeypatch.setattr(pp.settings, "FAL_API_KEY", "test-key")
        monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready, project_type="subtitle_removal",
                                caption_removal_method="void_source")
        src = os.path.join(storage.project_dir(project.id), "source.mp4")
        _caption_then_plain_video(src, n_captions=150, n=225)    # segment 2 has none
        project.source_local_path = src
        for i in range(3):
            _make_segment_def(s, project.id, i, start_sec=2.5 * i, end_sec=2.5 * (i + 1))
        s.commit()
        pid = project.id
        s.close()
        return pid, calls

    def _create(self, client, pid, **extra):
        return client.post(f"/api/v2/projects/{pid}/runs",
                           data={"prompt": "remove the captions", "model": "gemini-omni",
                                 "resolution": "720p", **extra})

    def test_the_run_is_parked_with_its_previews_and_reserves_nothing(
            self, spy_client, SessionFactory, world):
        from app import spend
        from app.segment_fill_models import RunSegmentFillJob

        pid, calls = world
        client, spy = spy_client
        r = self._create(client, pid, reference_urls="https://x/y.jpg")
        assert r.status_code == 422 and "VOID from the source" in r.json()["detail"]
        resp = self._create(client, pid)
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "incomplete"
        rid = resp.json()["run_id"]
        assert spy["process_run"] == []                       # nothing goes to the model
        s = SessionFactory()
        run = s.get(Run, rid)
        assert run.generation_mode == "void_source" and run.segment_snapshot
        segs = s.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalars().all()
        assert sorted(rs.index for rs in segs) == [0, 1, 2]
        assert {rs.status for rs in segs} == {SegmentStatus.pending}
        jobs = s.execute(select(RunSegmentFillJob).where(
            RunSegmentFillJob.run_id == rid)).scalars().all()
        assert sorted(calls["prepare"]) == sorted(j.id for j in jobs) and len(jobs) == 3
        assert {j.base["basis"] for j in jobs} == {"source"}
        assert spend.estimate_run_credits(s, run) == 0.0
        s.close()
        # a copy would be a v2v run sending every segment to the model
        assert client.post(f"/api/v2/runs/{rid}/copy", data={}).status_code in (409, 422)

    def test_source_fills_in_one_batch_then_one_requeue(self, spy_client, SessionFactory,
                                                        world):
        import app.pipeline_segment_fill as psf
        from app.provider_models import ProviderTask
        from app.segment_fill_models import RunSegmentFillBatch, RunSegmentFillJob

        pid, calls = world
        client, spy = spy_client
        rid = self._create(client, pid).json()["run_id"]
        for fid in list(calls["prepare"]):
            psf.segment_fill_prepare(fid)
        s = SessionFactory()
        jobs = {s.get(RunSegment, j.run_segment_id).index: j for j in s.execute(
            select(RunSegmentFillJob).where(RunSegmentFillJob.run_id == rid)).scalars()}
        assert jobs[0].status == jobs[1].status == jobs[2].status == "ready"
        assert jobs[2].media.get("source_as_is") and not jobs[0].media.get("source_as_is")
        f0, f1, f2 = jobs[0].id, jobs[1].id, jobs[2].id
        s.close()
        # the caption-less segment goes in as is, for free; the run stays parked
        assert client.post(f"/api/v2/runs/{rid}/segments/fill/{f2}/apply-source").status_code \
            == 200
        assert spy["process_run"] == []
        # ONE batch for the other two
        resp = client.post(f"/api/v2/runs/{rid}/segments/fill/batch/start",
                           json={"fill_ids": [f0, f1], "approved_usd": 0.20})
        assert resp.status_code == 200, resp.text
        bid = resp.json()["id"]
        s = SessionFactory()
        assert s.get(RunSegmentFillBatch, bid).basis == "source"
        for fid in (f0, f1):
            job = s.get(RunSegmentFillJob, fid)
            task = s.get(ProviderTask, job.provider_task_id)
            _plain_video(task.destination_path, n=job.media["void_frames"])
            task.status, task.result_path, task.reservation_held = "done", task.destination_path, False
        s.commit()
        s.close()
        psf.segment_fill_composite(f0)
        assert spy["process_run"] == []
        psf.segment_fill_composite(f1)
        s = SessionFactory()
        b = s.get(RunSegmentFillBatch, bid)
        assert b.status == "applied", (b.error_code, b.error_message)
        run = s.get(Run, rid)
        assert run.status == RunStatus.queued                  # complete and free: requeued
        assert spy["process_run"] == [rid]                     # ONCE: stitch + first delivery
        segs = s.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalars().all()
        assert {rs.status for rs in segs} == {SegmentStatus.completed}
        assert all(rs.seedance_task_id is None for rs in segs)
        s.close()

    def test_the_omni_fallback_of_one_segment_is_priced_and_marked(
            self, spy_client, SessionFactory, world):
        from app import spend

        pid, _calls = world
        client, spy = spy_client
        rid = self._create(client, pid).json()["run_id"]
        s = SessionFactory()
        rs1 = s.execute(select(RunSegment).where(RunSegment.run_id == rid,
                                                 RunSegment.index == 1)).scalar_one()
        rsid = rs1.id
        s.close()
        resp = client.post(f"/api/v2/runs/{rid}/segments/{rsid}/rerun", data={})
        assert resp.status_code == 200, resp.text
        s = SessionFactory()
        rs1 = s.get(RunSegment, rsid)
        run = s.get(Run, rid)
        assert rs1.model_fallback is True and run.status == RunStatus.queued
        need = spend.estimate_run_credits(s, run)
        assert need == spend.video_credits("gemini-omni", run.resolution, 2.5)   # ONLY seg 1
        s.close()
        assert spy["process_run"] == [rid]

    def test_project_setting_default_and_patch(self, client, db_session):
        from app import project_types

        p = _make_project(db_session, project_type="subtitle_removal")
        assert project_types.caption_removal_method(p) == "omni"   # NULL: an older project
        assert project_types.default_caption_removal_method("subtitle_removal") == "void_source"
        assert project_types.default_caption_removal_method("face_swap") is None
        r = client.patch(f"/api/v2/projects/{p.id}", json={"caption_removal_method": "x"})
        assert r.status_code == 400
        r = client.patch(f"/api/v2/projects/{p.id}",
                         json={"caption_removal_method": "void_source"})
        assert r.status_code == 200 and r.json()["caption_removal_method"] == "void_source"


def test_screen_share_flags_a_lit_screen_not_skin():
    """The screen heuristic (design §7.4) on synthetic frames: captions over a
    pale blue-grey screen count, over skin-coloured texture they do not."""
    import numpy as np

    from app.autopilot.flows.subtitle_compute import SCREEN_FRAME_SHARE, screen_share

    w, h = 540, 960
    box = [0.25, 0.58, 0.75, 0.70]
    screen = np.zeros((h, w, 3), np.uint8)
    screen[:] = (225, 205, 190)                       # BGR: pale blue-grey screen
    _burn_word(screen, "HELLO", (int(w * 0.30), int(h * 0.64)))
    skin = np.zeros((h, w, 3), np.uint8)
    skin[:] = (120, 150, 205)                         # BGR: warm skin tone
    rng = np.random.default_rng(1)
    skin = np.clip(skin.astype(int) + rng.integers(-12, 12, skin.shape), 0, 255).astype(np.uint8)
    _burn_word(skin, "HELLO", (int(w * 0.30), int(h * 0.64)))
    assert screen_share(screen, box) >= SCREEN_FRAME_SHARE
    assert screen_share(skin, box) < SCREEN_FRAME_SHARE
    blank = np.full((h, w, 3), 100, np.uint8)
    assert screen_share(blank, box) is None           # no caption there: not measured


def s_task_result(SessionFactory, fid):
    from app.provider_models import ProviderTask
    from app.segment_fill_models import RunSegmentFillJob

    s = SessionFactory()
    try:
        return s.get(ProviderTask, s.get(RunSegmentFillJob, fid).provider_task_id).result_path
    finally:
        s.close()


def test_content_block_classifier():
    from app.content_block import content_block

    assert content_block(SAFETY_MSG) == "safety_filter"
    assert content_block("failed after 3 attempt(s): Request blocked: The input content was "
                         "flagged by safety filters for involving a prominent public figure.") \
        == "safety_filter"
    assert content_block("failed after 3 attempt(s): The request failed because the output "
                         "video may contain sensitive information.") == "sensitive_output"
    assert content_block("failed after 3 attempt(s): InputAudioSensitiveContentDetected") \
        == "sensitive_audio"
    assert content_block("The request failed because the output video may be related to "
                         "copyright restrictions.") == "copyright"
    for transient in (None, "", "Seedance failed: Internal Error, Please try again later.",
                      "timed out after 1800s (last state='generating')",
                      "Request blocked: rate limit", "short: model returned 14.7s"):
        assert content_block(transient) is None


class TestSegmentFillFromSource:
    """A swap segment the provider refused with a content filter, on an
    incomplete run: VOID fill from the SOURCE (or the source as is), then the
    run's first stitch + delivery -- against SQLite, Fal faked on disk."""

    @pytest.fixture()
    def world(self, SessionFactory, monkeypatch):
        from contextlib import contextmanager

        import app.pipeline_segment_fill as psf
        import app.tasks as tasks_mod
        from app import pipeline_provider as pp
        from app import storage

        @contextmanager
        def _sess():
            s = SessionFactory()
            try:
                yield s
            finally:
                s.close()

        monkeypatch.setattr(psf, "get_session", _sess)
        calls = {"prepare": [], "composite": [], "fal": []}
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_prepare",
                            lambda j: calls["prepare"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_composite",
                            lambda j: calls["composite"].append(j))
        monkeypatch.setattr(tasks_mod, "enqueue_fal_job",
                            lambda fn, t, **kw: calls["fal"].append((fn, t)))
        monkeypatch.setattr(pp.settings, "FAL_ENABLED", True)
        monkeypatch.setattr(pp.settings, "FAL_API_KEY", "test-key")
        monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)

        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready, project_type="subtitle_removal",
                                caption_detector={"band_y": [0.55, 0.75]})
        src = os.path.join(storage.project_dir(project.id), "source.mp4")
        _caption_video(src, n=240)  # 8.0 s, captions everywhere
        project.source_local_path = src
        sds = [_make_segment_def(s, project.id, i, start_sec=a, end_sec=b)
               for i, (a, b) in enumerate(((0.0, 3.0), (3.0, 5.5), (5.5, 8.0)))]
        run = _make_run(
            s, project.id, status=RunStatus.incomplete, model="gemini-omni",
            audio_mode="original", error_message="1 swap segment(s) did not complete: [0].",
            segment_snapshot=[{"id": sd.id, "index": sd.index, "action": "swap",
                               "start_sec": sd.start_sec, "end_sec": sd.end_sec,
                               "pre_roll_sec": 0.0, "post_roll_sec": 0.0} for sd in sds],
        )
        r_dir = storage.run_results_dir(run.id, project.id)
        segs = [_make_run_segment(s, run.id, sds[0].id, index=0, status=SegmentStatus.failed,
                                  seedance_task_id="omni-blocked-0", error_message=SAFETY_MSG)]
        for sd in sds[1:]:
            path = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
            with open(path, "wb") as fh:
                fh.write(f"omni-{sd.index}".encode())
            segs.append(_make_run_segment(
                s, run.id, sd.id, index=sd.index, status=SegmentStatus.completed,
                local_result_path=path, seedance_task_id=f"omni-task-{sd.index}",
                cost_credits=168.0))
        s.commit()
        ids = {"project": project.id, "run": run.id, "rs": segs[0].id, "rs1": segs[1].id,
               "rs2": segs[2].id, "src": src}
        s.close()
        return ids, calls

    def _preview(self, client, ids, rsid=None, **body):
        body.setdefault("basis", "source")
        return client.post(f"/api/v2/runs/{ids['run']}/segments/{rsid or ids['rs']}/fill/preview",
                           json=body)

    def _job(self, SessionFactory, fid):
        from app.segment_fill_models import RunSegmentFillJob

        s = SessionFactory()
        try:
            return s.get(RunSegmentFillJob, fid)
        finally:
            s.close()

    def _fake_provider_done(self, SessionFactory, fid):
        from app.provider_models import ProviderTask
        from app.segment_fill_models import RunSegmentFillJob

        s = SessionFactory()
        job = s.get(RunSegmentFillJob, fid)
        task = s.get(ProviderTask, job.provider_task_id)
        _plain_video(task.destination_path, n=job.media["void_frames"])
        task.status, task.result_path = "done", task.destination_path
        task.reservation_held = False
        s.commit()
        s.close()

    def test_gates(self, spy_client, SessionFactory, world):
        ids, _ = world
        client, _ = spy_client
        # The model basis needs a done run and a model result.
        assert self._preview(client, ids, basis="model_result").status_code == 409
        # The source basis needs a content-blocked failed segment ...
        r = self._preview(client, ids, rsid=ids["rs1"])
        assert r.status_code == 409 and "content filter" in r.text
        s = SessionFactory()
        rs = s.get(RunSegment, ids["rs"])
        rs.error_message = "failed after 3 attempt(s): Internal Error, Please try again later."
        s.commit()
        s.close()
        assert self._preview(client, ids).status_code == 409        # a transient failure
        s = SessionFactory()
        s.get(RunSegment, ids["rs"]).error_message = SAFETY_MSG
        s.get(Run, ids["run"]).audio_mode = "seedance"
        s.commit()
        s.close()
        assert self._preview(client, ids).status_code == 422        # audio_mode original only
        s = SessionFactory()
        s.get(Run, ids["run"]).audio_mode = "original"
        s.get(Run, ids["run"]).status = RunStatus.done
        s.commit()
        s.close()
        assert self._preview(client, ids).status_code == 409        # ... on an incomplete run
        bad = client.post(f"/api/v2/runs/{ids['run']}/segments/{ids['rs']}/fill/preview",
                          json={"basis": "nonsense"})
        assert bad.status_code == 422

    def test_void_from_source_applies_then_the_run_requeues_for_free(
            self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import probe_video, segment_fill_composite
        from app.pipeline_segment_fill import segment_fill_prepare
        from app.provider_models import ProviderTask

        ids, calls = world
        client, spy = spy_client
        r = self._preview(client, ids)
        assert r.status_code == 202, r.text
        fid = r.json()["id"]
        assert r.json()["basis"] == "source"
        segment_fill_prepare(fid)
        job = self._job(SessionFactory, fid)
        assert job.status == "ready", (job.error_code, job.error_message)
        assert job.base["basis"] == "source" and job.base["segment"]["block"] == "safety_filter"
        assert job.base["source"]["detector_sha"]
        assert job.media["frames"] == 90 and not job.media.get("source_as_is")
        assert job.estimated_usd == 0.10
        listing = client.get(f"/api/v2/runs/{ids['run']}/segments/fill").json()
        assert listing[0]["basis"] == "source" and listing[0]["source_as_is"] is False
        page = client.get(f"/v2/runs/{ids['run']}/status-fragment")
        assert "VOID from source" in page.text and "Start (reserve $0.10)" in page.text

        resp = client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/start",
                           json={"approved_usd": 0.10})
        assert resp.status_code == 200, resp.text
        s = SessionFactory()
        task = s.get(ProviderTask, s.get(type(job), fid).provider_task_id)
        assert task.owner_kind == "segment_fill" and calls["fal"] == [("fal_submit", task.id)]
        s.close()
        self._fake_provider_done(SessionFactory, fid)
        segment_fill_composite(fid)
        job = self._job(SessionFactory, fid)
        assert job.status == "applied", (job.error_code, job.error_message)
        assert job.replaced_take_id is None
        s = SessionFactory()
        rs, run = s.get(RunSegment, ids["rs"]), s.get(Run, ids["run"])
        assert rs.status == SegmentStatus.completed and rs.local_result_path == job.composite_path
        assert rs.seedance_task_id is None and rs.error_message is None
        assert probe_video(job.composite_path)["frames"] == 90
        assert run.status == RunStatus.queued and spy["process_run"] == [ids["run"]]
        s.close()

    def test_two_blocked_segments_requeue_only_after_the_last(
            self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import segment_fill_composite, segment_fill_prepare

        ids, calls = world
        client, spy = spy_client
        s = SessionFactory()
        rs2 = s.get(RunSegment, ids["rs2"])
        rs2.status, rs2.local_result_path = SegmentStatus.failed, None
        rs2.seedance_task_id, rs2.error_message = "omni-blocked-2", SAFETY_MSG
        s.commit()
        s.close()
        for n, rsid in enumerate((ids["rs"], ids["rs2"])):
            fid = self._preview(client, ids, rsid=rsid).json()["id"]
            segment_fill_prepare(fid)
            assert client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/start",
                               json={"approved_usd": 0.10}).status_code == 200
            self._fake_provider_done(SessionFactory, fid)
            segment_fill_composite(fid)
            assert self._job(SessionFactory, fid).status == "applied"
            s = SessionFactory()
            run = s.get(Run, ids["run"])
            if n == 0:
                # The other blocked segment is unfinished: no re-stitch (it
                # would resubmit it), the run stays incomplete.
                assert run.status == RunStatus.incomplete and spy["process_run"] == []
                assert "[2]" in run.error_message
            else:
                assert run.status == RunStatus.queued and spy["process_run"] == [ids["run"]]
            s.close()

    def test_a_transient_sibling_never_requeues_after_the_fill(
            self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import segment_fill_composite, segment_fill_prepare

        ids, _ = world
        client, spy = spy_client
        s = SessionFactory()
        rs2 = s.get(RunSegment, ids["rs2"])
        rs2.status, rs2.local_result_path, rs2.error_message = SegmentStatus.failed, None, "timeout"
        s.commit()
        s.close()
        fid = self._preview(client, ids).json()["id"]
        segment_fill_prepare(fid)
        assert client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/start",
                           json={"approved_usd": 0.10}).status_code == 200
        self._fake_provider_done(SessionFactory, fid)
        segment_fill_composite(fid)
        s = SessionFactory()
        assert s.get(Run, ids["run"]).status == RunStatus.incomplete
        assert spy["process_run"] == []
        s.close()

    def test_source_as_is_when_the_source_clip_has_no_caption(
            self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import probe_video, segment_fill_prepare

        ids, calls = world
        client, spy = spy_client
        _plain_video(ids["src"], n=240, w=540, h=960, fps="30")      # no captions at all
        s = SessionFactory()
        s.get(VideoProject, ids["project"]).source_local_path = ids["src"]
        s.commit()
        s.close()
        fid = self._preview(client, ids).json()["id"]
        segment_fill_prepare(fid)
        job = self._job(SessionFactory, fid)
        assert job.status == "ready", (job.error_code, job.error_message)
        assert job.media["source_as_is"] and job.estimated_usd == 0.0
        assert job.media["band_y"] == [0.55, 0.75] and os.path.isfile(job.preview_path)
        assert not os.path.exists(os.path.join(job.prep_dir, "void_v.mp4"))
        page = client.get(f"/v2/runs/{ids['run']}/status-fragment")
        assert "Use the source clip as is (free)" in page.text and "source as is" in page.text
        # Never a VOID start for it ...
        r = client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/start",
                        json={"approved_usd": 0.0})
        assert r.status_code == 409 and "apply-source" in r.text
        # ... and apply-source refuses a VOID preview (covered below) / applies this one.
        r = client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/apply-source")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "applied" and r.json()["source_as_is"] is True
        job = self._job(SessionFactory, fid)
        assert probe_video(job.composite_path)["frames"] == job.media["frames"] == 90
        assert calls["fal"] == []                                   # nothing paid
        s = SessionFactory()
        rs, run = s.get(RunSegment, ids["rs"]), s.get(Run, ids["run"])
        assert rs.status == SegmentStatus.completed and rs.local_result_path == job.composite_path
        assert run.status == RunStatus.queued and spy["process_run"] == [ids["run"]]
        s.close()
        # Applied: a second apply is refused.
        assert client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/apply-source"
                           ).status_code == 409

    def test_apply_source_refuses_a_void_preview(self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import segment_fill_prepare

        ids, _ = world
        client, _ = spy_client
        fid = self._preview(client, ids).json()["id"]
        segment_fill_prepare(fid)
        r = client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/apply-source")
        assert r.status_code == 409 and "start the VOID fill" in r.text
        assert self._job(SessionFactory, fid).status == "ready"

    def test_a_retry_between_preview_and_start_makes_it_stale(
            self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import segment_fill_prepare

        ids, _ = world
        client, _ = spy_client
        fid = self._preview(client, ids).json()["id"]
        segment_fill_prepare(fid)
        s = SessionFactory()
        rs = s.get(RunSegment, ids["rs"])
        rs.error_message = SAFETY_MSG.replace("3 attempt", "2 attempt")   # a new block
        s.commit()
        s.close()
        r = client.post(f"/api/v2/runs/{ids['run']}/segments/fill/{fid}/start",
                        json={"approved_usd": 0.10})
        assert r.status_code == 409 and "segment.error_message" in r.text
        assert self._job(SessionFactory, fid).status == "canceled"

    def test_a_detector_change_before_prepare_refuses_the_preview(
            self, spy_client, SessionFactory, world):
        from app.pipeline_segment_fill import segment_fill_prepare

        ids, _ = world
        client, _ = spy_client
        fid = self._preview(client, ids).json()["id"]
        s = SessionFactory()
        s.get(VideoProject, ids["project"]).caption_detector = {"band_y": [0.1, 0.3]}
        s.commit()
        s.close()
        segment_fill_prepare(fid)
        job = self._job(SessionFactory, fid)
        assert job.status == "failed" and job.error_code == "base_changed"

    def test_an_old_row_without_basis_is_a_model_result_fill(self, SessionFactory, world):
        from app import pipeline_segment_fill as psf

        assert psf.fill_basis({"segment": {}}) == psf.BASIS_MODEL
        with pytest.raises(psf.FillError):
            psf.fill_basis({"basis": "weird"})
        ids, _ = world
        s = SessionFactory()
        run, project = s.get(Run, ids["run"]), s.get(VideoProject, ids["project"])
        rs = s.get(RunSegment, ids["rs1"])
        legacy = psf.capture_base(s, run, rs, project)
        legacy.pop("basis")
        # A model-result base keeps exactly AICP-358's parts: an in-flight fill
        # previewed before the deploy is not stale after it.
        assert psf.base_mismatch(legacy, psf.capture_base(s, run, rs, project)) is None
        assert "detector_sha" not in legacy["source"] and "block" not in legacy["segment"]
        s.close()

    def test_fill_context_gives_a_blocked_segment_the_source_basis(self, SessionFactory, world):
        from app import subtitle_loop as sl

        ids, _ = world
        s = SessionFactory()
        run, project = s.get(Run, ids["run"]), s.get(VideoProject, ids["project"])
        snap = sl.fill_snapshot(s, run, project, src_fps=30.0)
        seg = snap["context"]["segments"][ids["rs"]]
        assert seg["basis"] == "source" and seg["block"] == "safety_filter"
        assert seg["blockers"] == []
        other = snap["context"]["segments"][ids["rs1"]]
        assert other["basis"] == "model_result" and any("not done" in b for b in other["blockers"])
        s.close()


class TestSubtitleLoopFillContext:
    """AICP-359: the read-only DB facts behind subtitle_loop's VOID fill
    proposal (result provenance, the real API gates, fill history, evidence)."""

    @pytest.fixture()
    def world(self, SessionFactory, monkeypatch):
        from app import pipeline_provider as pp
        from app import storage

        monkeypatch.setattr(pp.settings, "FAL_ENABLED", True)
        monkeypatch.setattr(pp.settings, "FAL_API_KEY", "test-key")
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready, project_type="subtitle_removal")
        src = os.path.join(storage.project_dir(project.id), "source.mp4")
        _caption_video(src, n=150)  # 5.0 s at 30 fps
        project.source_local_path = src
        sd0 = _make_segment_def(s, project.id, 0, start_sec=0.0, end_sec=3.0)
        sd1 = _make_segment_def(s, project.id, 1, start_sec=3.0, end_sec=5.0)
        run = _make_run(
            s, project.id, status=RunStatus.done, model="gemini-omni", audio_mode="original",
            result_gdrive_file_id="DRIVE1",
            segment_snapshot=[{"id": sd.id, "index": sd.index, "action": "swap",
                               "start_sec": sd.start_sec, "end_sec": sd.end_sec,
                               "pre_roll_sec": 0.0, "post_roll_sec": 0.0} for sd in (sd0, sd1)],
        )
        r_dir = storage.run_results_dir(run.id, project.id)
        final = os.path.join(r_dir, "final.mp4")
        open(final, "wb").write(b"final v1")
        run.result_local_path = final
        p0 = os.path.join(r_dir, "result_0000.mp4")
        open(p0, "wb").write(b"omni-0")
        rs0 = _make_run_segment(s, run.id, sd0.id, index=0, status=SegmentStatus.completed,
                                local_result_path=p0, seedance_task_id="omni-task-0",
                                cost_credits=168.0)
        comp = os.path.join(r_dir, "composite_1.mp4")
        open(comp, "wb").write(b"void composite")
        rs1 = _make_run_segment(s, run.id, sd1.id, index=1, status=SegmentStatus.completed,
                                local_result_path=comp, seedance_task_id=None,
                                cost_credits=168.0)
        s.commit()
        ids = {"project": project.id, "run": run.id, "rs0": rs0.id, "rs1": rs1.id}
        s.close()
        return ids

    def _job(self, s, ids, rsid, **kw):
        from app.segment_fill_models import RunSegmentFillJob

        job = RunSegmentFillJob(run_id=ids["run"], run_segment_id=rsid,
                                project_id=ids["project"], model="void", prompt="",
                                base={"run": {"final_revision": "older"}}, **kw)
        s.add(job)
        s.commit()
        return job

    def test_provenance_gates_and_history(self, SessionFactory, world):
        from app import pipeline_segment_fill as psf
        from app import subtitle_loop as sl

        ids = world
        s = SessionFactory()
        rs1 = s.get(RunSegment, ids["rs1"])
        applied = self._job(s, ids, ids["rs1"], status="applied", mask_box=[0.2, 0.55, 0.8, 0.7],
                            box_source="measured", estimated_usd=0.10, approved_usd=0.10,
                            provider_task_id="task-1", composite_path=rs1.local_result_path,
                            replaced_take_id="take-1",
                            media={"frames_masked": 40})
        inputs = sl._run_inputs(s, ids["run"])
        by_rs = {seg["run_segment_id"]: seg for seg in inputs["segments"]}
        assert by_rs[ids["rs0"]]["result_kind"] == "model"
        assert by_rs[ids["rs1"]]["result_kind"] == "void_fill"
        fill = by_rs[ids["rs1"]]["fill"]
        assert fill["job_id"] == applied.id and fill["delivery"] == "delivered"
        assert fill["replaced_take_id"] == "take-1" and fill["frames_masked"] == 40
        ctx = inputs["fill_context"]
        assert ctx["run_blockers"] == [] and ctx["active_fill"] is None
        assert ctx["segments"][ids["rs0"]] == {"blockers": [], "jobs": [],
                                               "basis": "model_result", "block": None}
        hist = ctx["segments"][ids["rs1"]]["jobs"]
        assert [h["job_id"] for h in hist] == [applied.id]
        assert hist[0]["provider_task_id"] == "task-1"

        # A ready preview that matches the run, then goes stale when the final moves.
        run, project = s.get(Run, ids["run"]), s.get(VideoProject, ids["project"])
        rs0 = s.get(RunSegment, ids["rs0"])
        ready = self._job(s, ids, ids["rs0"], status="ready", mask_box=[0.2, 0.55, 0.8, 0.7],
                          box_source="measured", estimated_usd=0.10,
                          media={"frames_masked": 50})
        ready.base = psf.capture_base(s, run, rs0, project)
        s.commit()
        inputs = sl._run_inputs(s, ids["run"])
        job = inputs["fill_context"]["segments"][ids["rs0"]]["jobs"][0]
        assert job["job_id"] == ready.id and job["stale"] is None and job["default_prompt"]
        ev1 = sl.run_evidence(s, run, project)
        assert ev1 == sl.run_evidence(s, run, project)
        project.caption_detector = {"band_y": [0.5, 0.75]}
        assert sl.run_evidence(s, run, project) != ev1   # detector config is bound too
        project.caption_detector = None
        assert sl.run_evidence(s, run, project) == ev1
        sd = project.segments[0]
        sd.end_sec = 2.5                                  # live segments are bound too
        assert sl.run_evidence(s, run, project) != ev1
        sd.end_sec = 3.0
        assert sl.run_evidence(s, run, project) == ev1
        with open(run.result_local_path, "wb") as fh:
            fh.write(b"final v2, re-delivered")
        inputs = sl._run_inputs(s, ids["run"])
        job = inputs["fill_context"]["segments"][ids["rs0"]]["jobs"][0]
        assert job["stale"] == "run.final_revision"
        assert sl.run_evidence(s, run, project) != ev1
        s.rollback()
        s.close()

    def test_blockers_come_from_the_real_gates(self, SessionFactory, world, monkeypatch):
        from app import pipeline_provider as pp
        from app import subtitle_loop as sl

        ids = world
        s = SessionFactory()
        run = s.get(Run, ids["run"])
        run.audio_mode = "seedance"
        self._job(s, ids, ids["rs1"], status="admitted", provider_task_id="task-2",
                  estimated_usd=0.10)
        monkeypatch.setattr(pp.settings, "FAL_ENABLED", False)
        ctx = sl._run_inputs(s, ids["run"])["fill_context"]
        assert ctx["active_fill"]["run_segment_id"] == ids["rs1"]
        assert any("one fill at a time" in b for b in ctx["run_blockers"])
        assert any("Fal is disabled" in b for b in ctx["run_blockers"])
        assert any("audio_mode=original" in b for b in ctx["segments"][ids["rs0"]]["blockers"])
        # A missing sibling result would make the re-stitch paid.
        rs1 = s.get(RunSegment, ids["rs1"])
        rs1.status = SegmentStatus.failed
        ctx = sl._run_inputs(s, ids["run"])["fill_context"]
        assert any("re-stitch" in b for b in ctx["segments"][ids["rs0"]]["blockers"])
        s.rollback()
        s.close()

    def test_hand_placed_result_is_unknown(self, SessionFactory, world):
        from app import subtitle_loop as sl

        ids = world
        s = SessionFactory()
        segs = {seg["run_segment_id"]: seg for seg in sl._run_inputs(s, ids["run"])["segments"]}
        # rs1 has no seedance task and no fill record: a hand-placed clip.
        assert segs[ids["rs1"]]["result_kind"] == "unknown"
        assert "fill" not in segs[ids["rs1"]]
        assert segs[ids["rs0"]]["result_kind"] == "model"
        # A run on another model (flow rule 2: Omni only) is not a known Omni output.
        s.get(Run, ids["run"]).model = "seedance-2"
        segs = {seg["run_segment_id"]: seg for seg in sl._run_inputs(s, ids["run"])["segments"]}
        assert segs[ids["rs0"]]["result_kind"] == "unknown"
        s.rollback()
        s.close()


# ---------------------------------------------------------------------------
# AICP-348 — GET /runs/{rid}/drive-file, POST /runs/{rid}/drive-relink
# ---------------------------------------------------------------------------


class TestDriveRelinkApi:
    @pytest.fixture()
    def relink_env(self, client, SessionFactory, monkeypatch, tmp_path):
        import app.drive_relink as dr
        import app.tasks as tasks_mod

        state = {"drive": "missing", "folder": "present", "enqueued": [], "live": False}

        class FakeDrive:
            def file_state(self, fid):
                if fid == "folder-run":
                    return state["folder"]
                if isinstance(state["drive"], Exception):
                    raise state["drive"]
                return state["drive"]

            def generate_file_id(self):
                return "reserved-1"

        monkeypatch.setattr(dr, "GDriveClient", FakeDrive)
        monkeypatch.setattr(dr, "scan_sheet", lambda fid, client=None: {"rows": [], "error": None})
        monkeypatch.setattr(tasks_mod, "enqueue_drive_relink", lambda i: state["enqueued"].append(i))
        monkeypatch.setattr(tasks_mod, "drive_relink_job_live", lambda i: state["live"])

        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        final = tmp_path / "final.mp4"
        final.write_bytes(b"final")
        run = _make_run(s, project.id, status=RunStatus.failed, gdrive_folder_id="folder-run",
                        result_local_path=str(final), result_gdrive_file_id="old-id")
        rid = run.id
        s.close()
        return client, rid, state

    def _post(self, client, rid, **body):
        payload = {"expected_old_file_id": "old-id", "confirm_link_change": True}
        payload.update(body)
        return client.post(f"/api/v2/runs/{rid}/drive-relink", json=payload)

    def test_get_reports_fresh_state_and_offer(self, relink_env):
        client, rid, state = relink_env
        body = client.get(f"/api/v2/runs/{rid}/drive-file").json()
        assert body["state"] == "missing" and body["can_relink"] is True
        assert body["link"] == "https://drive.google.com/file/d/old-id/view"
        state["drive"] = "present"
        body = client.get(f"/api/v2/runs/{rid}/drive-file").json()
        assert body["can_relink"] is False and "present" in body["reason"]

    def test_requires_explicit_confirmation(self, relink_env):
        client, rid, state = relink_env
        resp = self._post(client, rid, confirm_link_change=False)
        assert resp.status_code == 400 and "Creative Link" in resp.json()["detail"]
        assert state["enqueued"] == []

    @pytest.mark.parametrize("drive_state", ["present", "trashed"])
    def test_refused_unless_confirmed_missing(self, relink_env, drive_state):
        client, rid, state = relink_env
        state["drive"] = drive_state
        resp = self._post(client, rid)
        assert resp.status_code == 409 and drive_state in resp.json()["detail"]
        assert state["enqueued"] == []

    def test_drive_uncertainty_is_503_not_missing(self, relink_env):
        client, rid, state = relink_env
        state["drive"] = RuntimeError("Drive 500")
        assert self._post(client, rid).status_code == 503
        assert state["enqueued"] == []

    def test_unreadable_folder_is_refused_as_an_access_problem(self, relink_env):
        client, rid, state = relink_env
        state["folder"] = "missing"
        resp = self._post(client, rid)
        assert resp.status_code == 409 and "access problem" in resp.json()["detail"]
        assert client.get(f"/api/v2/runs/{rid}/drive-file").json()["can_relink"] is False
        assert state["enqueued"] == []

    def test_stale_expected_id_is_409(self, relink_env):
        client, rid, state = relink_env
        assert self._post(client, rid, expected_old_file_id="x").status_code == 409

    def test_admits_one_row_then_resumes_it_when_its_job_died(self, relink_env, SessionFactory):
        from app.models import RunDriveRelink

        client, rid, state = relink_env
        resp = self._post(client, rid)
        assert resp.status_code == 202, resp.text
        relink_id = resp.json()["relink"]["id"]
        assert state["enqueued"] == [relink_id]
        s = SessionFactory()
        row = s.get(RunDriveRelink, relink_id)
        assert (row.status, row.old_file_id, row.run_status) == ("queued", "old-id", "failed")
        assert row.target_file_id == "reserved-1"
        assert s.get(Run, rid).result_gdrive_file_id == "old-id", "admission never re-points"
        s.close()

        state["live"] = True
        assert self._post(client, rid).status_code == 409
        body = client.get(f"/api/v2/runs/{rid}/drive-file").json()
        assert body["can_relink"] is False and "in progress" in body["reason"]
        state["live"] = False
        body = client.get(f"/api/v2/runs/{rid}/drive-file").json()
        assert body["can_relink"] is True and body["resumes"] == relink_id
        resp = self._post(client, rid)
        assert resp.status_code == 202 and resp.json()["resumed"] is True
        assert resp.json()["relink"]["id"] == relink_id
        assert state["enqueued"] == [relink_id, relink_id]

    def test_a_new_relink_inherits_an_unfinished_rows_reserved_id(self, relink_env, SessionFactory):
        """A refused row's reserved id is reused, so its upload landing late is
        adopted by the next re-link instead of becoming a second file."""
        from app.models import RunDriveRelink

        client, rid, state = relink_env
        s = SessionFactory()
        s.add(RunDriveRelink(run_id=rid, status="refused", old_file_id="old-id",
                             target_file_id="earlier-reserved", result_path="/x",
                             run_status="failed"))
        s.add(RunDriveRelink(run_id=rid, status="done", old_file_id="older-id",
                             target_file_id="unrelated", result_path="/x",
                             run_status="done"))
        s.commit()
        s.close()
        resp = self._post(client, rid)
        assert resp.status_code == 202, resp.text
        s = SessionFactory()
        row = s.get(RunDriveRelink, resp.json()["relink"]["id"])
        assert row.target_file_id == "earlier-reserved"
        s.close()

    def test_a_trashed_inherited_id_is_not_reused(self, relink_env, SessionFactory, monkeypatch):
        import app.drive_relink as dr
        from app.models import RunDriveRelink

        client, rid, state = relink_env
        real = dr.GDriveClient

        class TrashedTarget(real):
            def file_state(self, fid):
                return "trashed" if fid == "binned" else super().file_state(fid)

        monkeypatch.setattr(dr, "GDriveClient", TrashedTarget)
        s = SessionFactory()
        s.add(RunDriveRelink(run_id=rid, status="refused", old_file_id="old-id",
                             target_file_id="binned", result_path="/x", run_status="failed"))
        s.commit()
        s.close()
        resp = self._post(client, rid)
        assert resp.status_code == 202, resp.text
        s = SessionFactory()
        assert s.get(RunDriveRelink, resp.json()["relink"]["id"]).target_file_id == "reserved-1"
        s.close()

    def test_active_run_is_refused(self, relink_env, SessionFactory):
        client, rid, state = relink_env
        s = SessionFactory()
        s.get(Run, rid).status = RunStatus.processing
        s.commit()
        s.close()
        resp = self._post(client, rid)
        assert resp.status_code == 409 and "status" in resp.json()["detail"]


def test_default_resolution_falls_back_to_720p_without_env():
    """AICP-391: with no DEFAULT_RESOLUTION in the environment (and no .env
    picked up) new runs default to 720p, not the 480p test tier."""
    env = {k: v for k, v in os.environ.items() if k != "DEFAULT_RESOLUTION"}
    code = (
        "import dotenv; dotenv.load_dotenv = lambda *a, **k: False\n"
        "from app.config import settings; print(settings.DEFAULT_RESOLUTION)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "720p"


def test_run_orm_default_resolution_is_720p(db_session):
    """AICP-391: a Run built without a resolution (script / internal paths)
    is 720p."""
    project = _make_project(db_session, status=ProjectStatus.ready)
    run = Run(project_id=project.id, prompt="swap", status=RunStatus.created,
              reference_image_urls=[])
    db_session.add(run)
    db_session.commit()
    assert run.resolution == "720p"
