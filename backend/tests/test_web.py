"""
Tests for the Jinja2 + HTMX web UI (app/web.py) and the frame endpoint.

Strategy
--------
- Temp SQLite DB per session (DATABASE_URL set before any app import).
- enqueue_analyze / enqueue_process monkeypatched to no-ops.
- TestClient wraps the FastAPI app.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile

# Must set env vars before importing app modules
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
from app.models import Job, Segment
from app.state_machine import JobStatus, SegmentStatus


# ---------------------------------------------------------------------------
# Session-scoped fixtures
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
# Per-test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session(SessionFactory):
    session = SessionFactory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def client(engine, SessionFactory, monkeypatch):
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
# Helpers
# ---------------------------------------------------------------------------


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


def _make_tiny_video(path: str) -> None:
    """Generate a tiny valid MP4 using ffmpeg (1-second black frame)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=black:size=64x36:rate=1",
            "-t", "1",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            path,
        ],
        capture_output=True,
        check=True,
    )


# ---------------------------------------------------------------------------
# Dashboard tests
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_get_dashboard_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_dashboard_contains_new_job_form_fields(self, client):
        resp = client.get("/")
        html = resp.text
        # Required form fields
        assert 'name="video_file"' in html
        assert 'name="gdrive_link"' in html
        assert 'name="prompt"' in html
        assert 'name="resolution"' in html

    def test_dashboard_resolution_options(self, client):
        resp = client.get("/")
        html = resp.text
        assert "480p" in html
        assert "720p" in html
        assert "1080p" in html

    def test_dashboard_lists_jobs(self, client, db_session):
        job = _make_job(db_session)
        resp = client.get("/")
        assert resp.status_code == 200
        # Short ID prefix should appear
        assert job.id[:8] in resp.text


# ---------------------------------------------------------------------------
# Job detail tests
# ---------------------------------------------------------------------------


class TestJobDetail:
    def test_job_detail_unknown_id_404(self, client):
        resp = client.get("/jobs/no-such-id-ever")
        assert resp.status_code == 404

    def test_job_detail_200(self, client, db_session):
        job = _make_job(db_session)
        resp = client.get(f"/jobs/{job.id}")
        assert resp.status_code == 200

    def test_job_detail_shows_status_badge(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.analyzing)
        resp = client.get(f"/jobs/{job.id}")
        assert "analyzing" in resp.text

    def test_job_detail_review_shows_segment_editor(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.review)
        seg0 = _make_segment(db_session, job.id, 0, start_sec=0.0, end_sec=5.0, action="swap")
        seg1 = _make_segment(db_session, job.id, 1, start_sec=5.0, end_sec=10.0, action="keep")

        resp = client.get(f"/jobs/{job.id}")
        assert resp.status_code == 200
        html = resp.text

        # Segment data should appear
        assert str(seg0.start_sec) in html or "0.0" in html
        assert str(seg1.end_sec) in html or "10.0" in html
        # Editor controls
        assert "seg-tbody" in html
        assert "Submit for processing" in html
        assert "Save edits" in html

    def test_job_detail_review_segment_values_present(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.review)
        seg = _make_segment(
            db_session, job.id, 0,
            start_sec=3.5, end_sec=8.2, action="swap",
            prompt_override="custom prompt here",
        )
        resp = client.get(f"/jobs/{job.id}")
        html = resp.text
        assert "3.5" in html
        assert "8.2" in html
        assert "custom prompt here" in html

    def test_job_detail_done_shows_video_and_download(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        job = _make_job(
            db_session,
            status=JobStatus.done,
            result_local_path=str(result_file),
        )
        resp = client.get(f"/jobs/{job.id}")
        html = resp.text
        assert "<video" in html
        assert "Download MP4" in html

    def test_job_detail_failed_shows_retry(self, client, db_session):
        job = _make_job(
            db_session,
            status=JobStatus.failed,
            error_message="Something went wrong",
        )
        resp = client.get(f"/jobs/{job.id}")
        html = resp.text
        assert "failed" in html.lower()
        assert "Something went wrong" in html
        assert "Retry Job" in html

    def test_job_detail_processing_shows_spinner(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.processing)
        resp = client.get(f"/jobs/{job.id}")
        html = resp.text
        assert "spinner" in html or "processing" in html.lower()


# ---------------------------------------------------------------------------
# Status fragment tests
# ---------------------------------------------------------------------------


class TestStatusFragment:
    def test_status_fragment_200(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.analyzing)
        resp = client.get(f"/jobs/{job.id}/status-fragment")
        assert resp.status_code == 200

    def test_status_fragment_reflects_status(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.queued)
        resp = client.get(f"/jobs/{job.id}/status-fragment")
        assert resp.status_code == 200
        assert "queued" in resp.text

    def test_status_fragment_unknown_id_404(self, client):
        resp = client.get("/jobs/no-such-id/status-fragment")
        assert resp.status_code == 404

    def test_status_fragment_review_has_editor(self, client, db_session):
        job = _make_job(db_session, status=JobStatus.review)
        _make_segment(db_session, job.id, 0)
        resp = client.get(f"/jobs/{job.id}/status-fragment")
        assert resp.status_code == 200
        assert "seg-tbody" in resp.text


# ---------------------------------------------------------------------------
# Frame endpoint tests
# ---------------------------------------------------------------------------


class TestFrameEndpoint:
    def test_frame_404_for_missing_job(self, client):
        resp = client.get("/api/jobs/no-such-job/frame?t=0")
        assert resp.status_code == 404

    def test_frame_404_when_no_source_file(self, client, db_session):
        # Job with no local video file
        job = _make_job(db_session, source_local_path=None)
        resp = client.get(f"/api/jobs/{job.id}/frame?t=0")
        assert resp.status_code == 404

    @pytest.mark.skipif(
        subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
        reason="ffmpeg not available",
    )
    def test_frame_returns_jpeg_from_real_video(self, client, db_session, tmp_path):
        """End-to-end: generate a tiny video, extract frame 0, verify JPEG."""
        video_path = str(tmp_path / "source.mp4")
        _make_tiny_video(video_path)

        job = _make_job(db_session, source_local_path=video_path)

        resp = client.get(f"/api/jobs/{job.id}/frame?t=0")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        # JPEG magic bytes
        assert resp.content[:2] == b"\xff\xd8"

    @pytest.mark.skipif(
        subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
        reason="ffmpeg not available",
    )
    def test_frame_cached_on_second_request(self, client, db_session, tmp_path):
        """Second request should hit the cache (same bytes, still 200)."""
        video_path = str(tmp_path / "source_cache.mp4")
        _make_tiny_video(video_path)

        job = _make_job(db_session, source_local_path=video_path)

        resp1 = client.get(f"/api/jobs/{job.id}/frame?t=0")
        resp2 = client.get(f"/api/jobs/{job.id}/frame?t=0")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.content == resp2.content
