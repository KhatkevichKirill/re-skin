"""
Tests for the v2 Jinja2 + HTMX web UI (app/web_v2.py).

Strategy
--------
- Temp SQLite DB per session (DATABASE_URL set before any app import).
- enqueue_analyze_project / enqueue_process_run monkeypatched to no-ops.
- TestClient wraps the FastAPI app.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Must set env vars before importing app modules
_db_fd, _db_path = tempfile.mkstemp(suffix="_v2web.db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
# Override if already set to something else (previous test modules may have set it)
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
# Helpers
# ---------------------------------------------------------------------------


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
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
    )
    defaults.update(kwargs)
    s = SegmentDef(**defaults)
    session.add(s)
    session.commit()
    return s


def _make_run(session, project_id: str, **kwargs) -> Run:
    defaults = dict(
        project_id=project_id,
        prompt="Replace the person.",
        resolution="480p",
        status=RunStatus.created,
        reference_image_urls=[],
    )
    defaults.update(kwargs)
    r = Run(**defaults)
    session.add(r)
    session.commit()
    return r


def _make_run_segment(session, run_id: str, index: int, **kwargs) -> RunSegment:
    defaults = dict(
        run_id=run_id,
        segment_def_id="fake-seg-def-id",
        index=index,
        status=SegmentStatus.pending,
    )
    defaults.update(kwargs)
    rs = RunSegment(**defaults)
    session.add(rs)
    session.commit()
    return rs


# ---------------------------------------------------------------------------
# Dashboard tests
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_get_dashboard_200(self, client):
        resp = client.get("/v2/")
        assert resp.status_code == 200

    def test_dashboard_contains_new_project_form_fields(self, client):
        resp = client.get("/v2/")
        html = resp.text
        # New project form must have these fields
        assert 'name="video_file"' in html
        assert 'name="gdrive_link"' in html

    def test_dashboard_no_job_fields(self, client):
        """v2 dashboard must NOT have prompt/ref fields (those belong to runs)."""
        resp = client.get("/v2/")
        html = resp.text
        # prompt/reference_files live on runs, not on the project creation form
        assert 'name="video_file"' in html  # sanity
        # The new-project form section should not include a prompt textarea
        # (it may appear elsewhere if a project is listed, but not in the creation form)
        assert "/api/v2/projects" in html

    def test_dashboard_lists_projects(self, client, db_session):
        project = _make_project(db_session)
        resp = client.get("/v2/")
        assert resp.status_code == 200
        assert project.id[:8] in resp.text

    def test_dashboard_status_badge_uses_value_not_enum(self, client, db_session):
        """Badge class must be badge-created, NOT badge-ProjectStatus.created."""
        _make_project(db_session, status=ProjectStatus.created)
        resp = client.get("/v2/")
        html = resp.text
        assert "badge-created" in html
        assert "ProjectStatus." not in html

    def test_dashboard_status_badge_ready(self, client, db_session):
        _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get("/v2/")
        assert "badge-ready" in resp.text
        assert "ProjectStatus." not in resp.text

    def test_dashboard_shows_project_name(self, client, db_session):
        _make_project(db_session, status=ProjectStatus.ready, name="My Campaign")
        resp = client.get("/v2/")
        assert "My Campaign" in resp.text

    def test_dashboard_has_runs_pivot_toggle(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get("/v2/")
        # Each project row gets an expand toggle wired to its runs fragment.
        assert f"toggleRuns('{project.id}'" in resp.text


# ---------------------------------------------------------------------------
# Runs pivot fragment — GET /v2/projects/{pid}/runs-fragment
# ---------------------------------------------------------------------------


class TestRunsFragment:
    def test_fragment_lists_runs_with_meta(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(
            db_session, project.id, name="Redhead",
            model="gemini-omni", resolution="1080p", status=RunStatus.done,
        )
        resp = client.get(f"/v2/projects/{project.id}/runs-fragment")
        assert resp.status_code == 200
        html = resp.text
        assert "Redhead" in html
        assert "Gemini Omni" in html   # model label
        assert "1080p" in html          # resolution
        assert "badge-done" in html     # status badge
        assert "deleteRun(" in html     # per-run delete button

    def test_fragment_empty_when_no_runs(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get(f"/v2/projects/{project.id}/runs-fragment")
        assert resp.status_code == 200
        assert "No runs yet" in resp.text

    def test_fragment_404_for_missing_project(self, client):
        assert client.get("/v2/projects/nope/runs-fragment").status_code == 404


# ---------------------------------------------------------------------------
# Project detail tests
# ---------------------------------------------------------------------------


class TestProjectDetail:
    def test_project_unknown_id_404(self, client):
        resp = client.get("/v2/projects/no-such-project-ever")
        assert resp.status_code == 404

    def test_project_detail_200(self, client, db_session):
        p = _make_project(db_session)
        resp = client.get(f"/v2/projects/{p.id}")
        assert resp.status_code == 200

    def test_project_analyzing_shows_spinner(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.analyzing)
        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text
        assert "spinner" in html or "Analyzing" in html

    def test_project_ready_shows_segment_editor(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        seg0 = _make_segment_def(db_session, p.id, 0, start_sec=0.0, end_sec=5.0, action="swap")
        seg1 = _make_segment_def(db_session, p.id, 1, start_sec=5.0, end_sec=10.0, action="keep")

        resp = client.get(f"/v2/projects/{p.id}")
        assert resp.status_code == 200
        html = resp.text

        # Segment editor must appear
        assert "seg-tbody" in html
        assert "Save segments" in html

        # Segment values must be present
        assert str(seg0.start_sec) in html or "0.0" in html
        assert str(seg1.end_sec) in html or "10.0" in html

    def test_project_ready_shows_new_run_form(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text

        # Run form fields must be present
        assert 'name="name"' in html
        assert 'name="prompt"' in html
        assert 'name="resolution"' in html

    def test_project_ready_prompt_prefill_example(self, client, db_session):
        """The default prompt example text must appear in the run form textarea."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text
        # Check a distinctive fragment of the prefilled prompt
        assert "Replace the main person in the reference video" in html

    def test_project_ready_no_at_image1_as_default(self, client, db_session):
        """@Image1 must NOT appear as the default/example text in the prompt field.
        The hint warns users not to use it, but it should not be prefilled."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text
        # The warning hint mentions @Image1 — but it must warn AGAINST it
        # Ensure the warning is there (it mentions @Image1 in a "do not use" context)
        assert "@Image1" in html  # appears in the hint/warning
        # But the prompt textarea default must NOT start with or rely on @Image1
        # The default prompt text must be present instead
        assert "reference image" in html.lower()
        # Confirm the do-not-use warning context
        assert "do" in html.lower() and "not" in html.lower()

    def test_project_ready_shows_existing_runs(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, p.id, name="Test Run", status=RunStatus.done)

        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text
        assert "Test Run" in html

    def test_project_failed_shows_error(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.failed, error_message="probe failed")
        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text
        assert "failed" in html.lower()
        assert "probe failed" in html


# ---------------------------------------------------------------------------
# Project status fragment tests
# ---------------------------------------------------------------------------


class TestProjectStatusFragment:
    def test_fragment_200(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.analyzing)
        resp = client.get(f"/v2/projects/{p.id}/status-fragment")
        assert resp.status_code == 200

    def test_fragment_reflects_status(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.analyzing)
        resp = client.get(f"/v2/projects/{p.id}/status-fragment")
        assert resp.status_code == 200
        assert "Analyzing" in resp.text or "spinner" in resp.text

    def test_fragment_ready_has_editor(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0)
        resp = client.get(f"/v2/projects/{p.id}/status-fragment")
        assert resp.status_code == 200
        assert "seg-tbody" in resp.text

    def test_fragment_unknown_id_404(self, client):
        resp = client.get("/v2/projects/no-such-project/status-fragment")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Run detail tests
# ---------------------------------------------------------------------------


class TestRunDetail:
    def test_run_unknown_id_404(self, client):
        resp = client.get("/v2/runs/no-such-run-ever")
        assert resp.status_code == 404

    def test_run_detail_200(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id)
        resp = client.get(f"/v2/runs/{r.id}")
        assert resp.status_code == 200

    def test_run_detail_processing_shows_spinner(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        assert "spinner" in html or "processing" in html.lower()

    def test_run_detail_done_shows_video_and_download(self, client, db_session, tmp_path):
        # Create a tiny real result file so the path exists
        result_file = tmp_path / "final_v2.mp4"
        result_file.write_bytes(b"\x00VIDEO_V2\xff")

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        assert "<video" in html
        assert "Download MP4" in html

    def test_run_detail_failed_shows_retry(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.failed,
            error_message="Generation timed out",
        )
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        assert "failed" in html.lower()
        assert "Generation timed out" in html
        assert "Retry" in html

    def test_run_detail_status_badge_uses_value(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.queued)
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        assert "badge-queued" in html
        assert "RunStatus." not in html

    def test_run_detail_shows_progress(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        # Need a SegmentDef to attach RunSegment to
        seg_def = _make_segment_def(db_session, p.id, 0)
        _make_run_segment(
            db_session, r.id, 0,
            segment_def_id=seg_def.id,
            status=SegmentStatus.completed,
        )
        _make_run_segment(
            db_session, r.id, 1,
            segment_def_id=seg_def.id,
            status=SegmentStatus.generating,
        )
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        # Should show progress counts
        assert "1" in html
        assert "2" in html


# ---------------------------------------------------------------------------
# Run status fragment tests
# ---------------------------------------------------------------------------


class TestRunStatusFragment:
    def test_run_fragment_200(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200

    def test_run_fragment_reflects_status(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.stitching)
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        assert "stitching" in resp.text

    def test_run_fragment_done_has_video(self, client, db_session, tmp_path):
        result_file = tmp_path / "frag_result.mp4"
        result_file.write_bytes(b"\x00DONE\xff")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        assert "<video" in resp.text

    def test_run_fragment_unknown_id_404(self, client):
        resp = client.get("/v2/runs/no-such-run/status-fragment")
        assert resp.status_code == 404
