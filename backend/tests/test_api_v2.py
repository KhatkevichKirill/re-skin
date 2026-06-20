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

import io
import os
import subprocess
import sys
import tempfile

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

from app.db import Base, get_db
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus


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


@pytest.fixture()
def client(engine, SessionFactory, monkeypatch):
    """TestClient with DB dependency overridden and task queues mocked."""
    import app.api_v2 as api_v2_module

    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project", lambda pid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid: None)

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

    calls: dict[str, list[str]] = {"analyze_project": [], "process_run": []}

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
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:size=64x64:rate=10:duration=1",
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
        assert body["status"] == "created"
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
                    "https://example.com/ref3.jpg"
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

    def test_retry_non_failed_run_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 409

    def test_retry_missing_run_is_404(self, client):
        response = client.post("/api/v2/runs/no-such-run/retry")
        assert response.status_code == 404


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
            data={"reference_urls": "https://a.com/1.jpg,https://a.com/2.jpg,https://a.com/3.jpg"},
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
        run = _make_run(session, project.id, status=RunStatus.done)
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
        session2.close()
        assert rs_fetched.status == SegmentStatus.pending
        assert rs_fetched.seedance_task_id is None

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
