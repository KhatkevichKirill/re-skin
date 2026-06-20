"""
Tests for the REST API (app/api.py).

Strategy
--------
- Temp SQLite DB per test session (DATABASE_URL set before importing app).
- enqueue_analyze and enqueue_process are monkeypatched to no-ops so no Redis
  or RQ is required.
- A TestClient wraps the FastAPI app for request-level tests.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

# Must set DATABASE_URL before any app import so db.py picks up the temp path
_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
# Prevent DATA_DIR from writing into the repo tree during tests
_data_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _data_tmp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.models import Job, Segment
from app.state_machine import JobStatus, SegmentStatus


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
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Per-test DB override + TestClient
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
    # Patch enqueue functions BEFORE importing main so the module-level symbols
    # inside api.py are replaced.
    import app.api as api_module

    monkeypatch.setattr(api_module, "enqueue_analyze", lambda job_id: None)
    monkeypatch.setattr(api_module, "enqueue_process", lambda job_id: None)

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
    """Returns dicts that record calls to enqueue_analyze and enqueue_process."""
    import app.api as api_module

    calls: dict[str, list[str]] = {"analyze": [], "process": []}

    monkeypatch.setattr(api_module, "enqueue_analyze", lambda job_id: calls["analyze"].append(job_id))
    monkeypatch.setattr(api_module, "enqueue_process", lambda job_id: calls["process"].append(job_id))
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
    """Return a few bytes that pass as a video upload (we only test storage, not ffprobe)."""
    return b"\x00\x01\x02\x03" * 16


def _make_job(session, **kwargs) -> Job:
    defaults = dict(
        source_type="upload",
        source_ref="test.mp4",
        default_prompt="swap character",
        resolution="720p",
        status=JobStatus.created,
    )
    defaults.update(kwargs)
    j = Job(**defaults)
    session.add(j)
    session.commit()
    return j


def _make_segment(session, job_id: str, index: int, **kwargs) -> Segment:
    defaults = dict(
        job_id=job_id,
        index=index,
        start_sec=float(index * 5),
        end_sec=float(index * 5 + 5),
        has_face=True,
        action="swap",
        status=SegmentStatus.pending,
    )
    defaults.update(kwargs)
    s = Segment(**defaults)
    session.add(s)
    session.commit()
    return s


# ---------------------------------------------------------------------------
# POST /api/jobs
# ---------------------------------------------------------------------------


class TestCreateJob:
    def test_upload_video_returns_201(self, spy_client):
        client, spy = spy_client
        response = client.post(
            "/api/jobs",
            data={"prompt": "replace character", "resolution": "720p"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201
        body = response.json()
        assert "job_id" in body
        assert body["status"] in ("created", "analyzing")
        assert spy["analyze"] == [body["job_id"]]

    def test_gdrive_link_variant(self, spy_client):
        client, spy = spy_client
        response = client.post(
            "/api/jobs",
            data={
                "prompt": "replace character",
                "gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert "job_id" in body
        assert spy["analyze"] == [body["job_id"]]

    def test_neither_video_nor_link_is_400(self, client):
        response = client.post("/api/jobs", data={"prompt": "swap"})
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"].lower()

    def test_both_video_and_link_is_400(self, client):
        response = client.post(
            "/api/jobs",
            data={"prompt": "swap", "gdrive_link": "https://drive.google.com/file/d/X/view"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"].lower()

    def test_too_many_references_is_400(self, client):
        # reference_urls sent as comma-separated string (API contract for multipart)
        response = client.post(
            "/api/jobs",
            data={
                "prompt": "swap",
                "reference_urls": (
                    "https://example.com/ref1.jpg,"
                    "https://example.com/ref2.jpg,"
                    "https://example.com/ref3.jpg"
                ),
            },
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "too many reference" in response.json()["detail"].lower()

    def test_bad_resolution_is_400(self, client):
        response = client.post(
            "/api/jobs",
            data={"prompt": "swap", "resolution": "4k"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    def test_job_row_created_in_db(self, spy_client, SessionFactory):
        client, _ = spy_client
        response = client.post(
            "/api/jobs",
            data={"prompt": "replace character"},
            files={"video_file": ("vid.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201
        job_id = response.json()["job_id"]

        session = SessionFactory()
        job = session.get(Job, job_id)
        session.close()
        assert job is not None
        assert job.source_type == "upload"
        assert job.default_prompt == "replace character"


# ---------------------------------------------------------------------------
# GET /api/jobs  and  GET /api/jobs/{id}
# ---------------------------------------------------------------------------


class TestGetJob:
    def test_get_existing_job_200(self, client, db_session):
        job = _make_job(db_session)
        response = client.get(f"/api/jobs/{job.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == job.id
        assert body["status"] == "created"

    def test_get_missing_job_404(self, client):
        response = client.get("/api/jobs/does-not-exist-id")
        assert response.status_code == 404

    def test_list_jobs_returns_array(self, client, db_session):
        _make_job(db_session)
        response = client.get("/api/jobs")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert "id" in body[0]
        assert "status" in body[0]


# ---------------------------------------------------------------------------
# GET /api/jobs/{id}/segments  and  PATCH /api/jobs/{id}/segments
# ---------------------------------------------------------------------------


class TestSegments:
    def test_get_segments_returns_list(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.review)
        _make_segment(db_session, job.id, 0)
        _make_segment(db_session, job.id, 1)

        response = client.get(f"/api/jobs/{job.id}/segments")
        assert response.status_code == 200
        segs = response.json()
        assert len(segs) == 2
        assert segs[0]["index"] < segs[1]["index"]

    def test_get_segments_404_for_missing_job(self, client):
        response = client.get("/api/jobs/no-such-job/segments")
        assert response.status_code == 404

    def test_patch_segments_edits_field(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.review)
        seg = _make_segment(db_session, job.id, 0, action="keep")

        response = client.patch(
            f"/api/jobs/{job.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap", "end_sec": 7.5}]},
        )
        assert response.status_code == 200
        segs = response.json()
        assert len(segs) == 1
        assert segs[0]["action"] == "swap"
        assert segs[0]["end_sec"] == pytest.approx(7.5)

    def test_patch_segments_reorders_by_start_sec(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.review)
        s0 = _make_segment(db_session, job.id, 0, start_sec=0.0, end_sec=5.0)
        s1 = _make_segment(db_session, job.id, 1, start_sec=5.0, end_sec=10.0)

        # Move s1 to start before s0
        response = client.patch(
            f"/api/jobs/{job.id}/segments",
            json={"updates": [{"id": s1.id, "start_sec": -1.0, "end_sec": 4.0}]},
        )
        assert response.status_code == 200
        segs = response.json()
        # s1 should now be first (index 0)
        assert segs[0]["id"] == s1.id
        assert segs[0]["index"] == 0
        assert segs[1]["id"] == s0.id
        assert segs[1]["index"] == 1

    def test_patch_on_non_review_job_is_409(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.queued)
        seg = _make_segment(db_session, job.id, 0)

        response = client.patch(
            f"/api/jobs/{job.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap"}]},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# POST /api/jobs/{id}/submit
# ---------------------------------------------------------------------------


class TestSubmitJob:
    def test_submit_review_job_transitions_to_queued(self, spy_client, db_session):
        client, spy = spy_client
        job = _make_job(db_session, status=JobStatus.review)

        response = client.post(f"/api/jobs/{job.id}/submit")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert spy["process"] == [job.id]

    def test_submit_non_review_job_is_409(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.created)
        response = client.post(f"/api/jobs/{job.id}/submit")
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/jobs/{id}/result
# ---------------------------------------------------------------------------


class TestResult:
    def test_result_on_non_done_job_is_409(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.processing)
        response = client.get(f"/api/jobs/{job.id}/result")
        assert response.status_code == 409

    def test_result_on_done_job_returns_file(self, client, db_session, tmp_path):
        # Write a tiny fake result file
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")

        job = _make_job(
            db_session,
            status=JobStatus.done,
            result_local_path=str(result_file),
        )

        response = client.get(f"/api/jobs/{job.id}/result")
        assert response.status_code == 200
        assert response.content == b"\x00VIDEO\xff"

    def test_result_info_on_done_job(self, client, db_session, tmp_path):
        result_file = tmp_path / "output.mp4"
        result_file.write_bytes(b"\x00\xff")

        job = _make_job(
            db_session,
            status=JobStatus.done,
            result_local_path=str(result_file),
            result_gdrive_file_id="GDRIVE_FILE_123",
        )

        response = client.get(f"/api/jobs/{job.id}/result/info")
        assert response.status_code == 200
        body = response.json()
        assert body["result_gdrive_file_id"] == "GDRIVE_FILE_123"
        assert "drive.google.com" in body["result_gdrive_link"]
