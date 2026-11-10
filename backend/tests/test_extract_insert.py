"""
Tests for the PIP-insert extraction endpoint (POST /api/v2/projects/{pid}/extract-insert)
and the M4 merge endpoints (POST /api/v2/runs/{run_id}/merge,
POST /api/v2/merges/{merge_id}/retry, GET /api/v2/merges/{merge_id},
GET /api/v2/projects/{pid}/merges).

Strategy
--------
- Own temp SQLite DB + DATA_DIR (isolated from other test modules, same pattern
  as test_api_v2.py / test_web_v2.py).
- enqueue_analyze_project / enqueue_process_run / enqueue_process_merge are
  monkeypatched to spies (no real RQ/Redis).
- app.media.crop_clip is monkeypatched to a stub that writes a placeholder file
  and returns evened (width, height) — no real ffmpeg is invoked.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Must set DATABASE_URL and DATA_DIR before any app import
_db_fd, _db_path = tempfile.mkstemp(suffix="_extract_insert.db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
_data_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _data_tmp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.media import MediaError
from app.models import InsertExtraction, InsertMerge, Run, VideoProject
from app.state_machine import MergeStatus, ProjectStatus, RunStatus
from app.storage import project_dir, project_source_path, run_results_dir


# ---------------------------------------------------------------------------
# Session-scoped engine + schema
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


@pytest.fixture()
def db_session(SessionFactory):
    session = SessionFactory()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# crop_clip stub — records calls, writes a placeholder file, returns evened dims
# ---------------------------------------------------------------------------


@pytest.fixture()
def crop_clip_calls(monkeypatch):
    calls: list[dict] = []

    def _fake_crop_clip(src, start_sec, end_sec, x, y, width, height, dst):
        calls.append(
            {
                "src": src,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "dst": dst,
            }
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(b"fake-cropped-clip")
        even_w = width - (width % 2)
        even_h = height - (height % 2)
        return (even_w, even_h)

    import app.api_v2 as api_v2_module

    monkeypatch.setattr(api_v2_module.media_mod, "crop_clip", _fake_crop_clip)
    return calls


@pytest.fixture()
def client(engine, SessionFactory, monkeypatch, crop_clip_calls):
    """TestClient with DB overridden, RQ enqueue mocked, crop_clip stubbed."""
    import app.api_v2 as api_v2_module

    enqueue_calls: list[str] = []
    monkeypatch.setattr(
        api_v2_module, "enqueue_analyze_project", lambda pid: enqueue_calls.append(pid)
    )
    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid: None)
    merge_enqueue_calls: list[str] = []
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_process_merge",
        lambda mid: merge_enqueue_calls.append(mid),
    )

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
        c.enqueue_calls = enqueue_calls  # type: ignore[attr-defined]
        c.crop_clip_calls = crop_clip_calls  # type: ignore[attr-defined]
        c.merge_enqueue_calls = merge_enqueue_calls  # type: ignore[attr-defined]
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_analyzed_project(session, **kwargs) -> VideoProject:
    """A parent project that has finished analyzing (probed) with a real source
    file on disk (content is irrelevant — crop_clip is stubbed)."""
    project_id = kwargs.pop("id", None)
    defaults = dict(
        source_type="upload",
        source_ref="parent.mp4",
        status=ProjectStatus.ready,
        width=640,
        height=480,
        duration_sec=10.0,
        fps=30.0,
        aspect_ratio="4:3",
    )
    defaults.update(kwargs)
    p = VideoProject(id=project_id, **defaults) if project_id else VideoProject(**defaults)
    session.add(p)
    session.flush()

    src_path = project_source_path(p.id, "mp4")
    with open(src_path, "wb") as fh:
        fh.write(b"fake-source-video-bytes")
    p.source_local_path = src_path

    session.commit()
    return p


def _make_done_run(session, project_id: str, **kwargs) -> Run:
    """A Run in status=done with a real result video file on disk."""
    defaults = dict(
        project_id=project_id,
        prompt="swap the character",
        status=RunStatus.done,
    )
    defaults.update(kwargs)
    run = Run(**defaults)
    session.add(run)
    session.flush()

    if run.result_local_path is None:
        result_dir = run_results_dir(run.id, project_id)
        result_path = os.path.join(result_dir, "final.mp4")
        with open(result_path, "wb") as fh:
            fh.write(b"fake-child-result-bytes")
        run.result_local_path = result_path

    session.commit()
    return run


_VALID_FORM = {
    "x": 10,
    "y": 20,
    "width": 101,  # odd — exercises even-down rounding
    "height": 51,
    "start_sec": 1.0,
    "end_sec": 5.0,
    "corner_radius_px": 8,
}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestExtractInsertHappyPath:
    def test_creates_child_project_and_extraction_row(self, client, db_session):
        parent = _make_analyzed_project(db_session)

        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 201
        body = response.json()
        assert "child_project_id" in body
        assert "extraction_id" in body
        assert body["status"] == "analyzing"  # AICP-350: counted at admission

        child_id = body["child_project_id"]

        # Child project row
        child = db_session.get(VideoProject, child_id)
        assert child is not None
        assert child.source_type == "upload"
        assert child.status == ProjectStatus.analyzing
        assert child.source_local_path is not None
        assert os.path.exists(child.source_local_path)
        assert child.name is not None and "insert" in child.name.lower()

        # Extraction row — evened dims (101 -> 100, 51 -> 50)
        extraction = db_session.get(InsertExtraction, body["extraction_id"])
        assert extraction is not None
        assert extraction.parent_project_id == parent.id
        assert extraction.child_project_id == child_id
        assert extraction.x == 10
        assert extraction.y == 20
        assert extraction.width == 100
        assert extraction.height == 50
        assert extraction.start_sec == 1.0
        assert extraction.end_sec == 5.0
        assert extraction.corner_radius_px == 8

    def test_child_inherits_mute_source(self, client, db_session):
        """The child is a crop of the same video — it carries the parent's
        "remove original audio" choice rather than re-enabling audio input."""
        parent = _make_analyzed_project(db_session)
        parent.mute_source = True
        db_session.commit()

        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 201
        child = db_session.get(VideoProject, response.json()["child_project_id"])
        assert child.mute_source is True

    def test_enqueues_analyze_for_child(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 201
        child_id = response.json()["child_project_id"]
        assert client.enqueue_calls == [child_id]

    def test_crop_clip_called_with_parent_source_and_rect(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 201
        child_id = response.json()["child_project_id"]

        assert len(client.crop_clip_calls) == 1
        call = client.crop_clip_calls[0]
        assert call["src"] == parent.source_local_path
        assert call["start_sec"] == 1.0
        assert call["end_sec"] == 5.0
        assert call["x"] == 10
        assert call["y"] == 20
        assert call["width"] == 101
        assert call["height"] == 51
        assert call["dst"] == project_source_path(child_id, "mp4")

    def test_default_corner_radius_is_zero(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        form = dict(_VALID_FORM)
        del form["corner_radius_px"]
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 201
        extraction = db_session.get(InsertExtraction, response.json()["extraction_id"])
        assert extraction.corner_radius_px == 0


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


class TestExtractInsertValidation:
    def _count_projects(self, db_session) -> int:
        return len(db_session.execute(select(VideoProject)).scalars().all())

    def test_missing_project_is_404(self, client, db_session):
        response = client.post(
            "/api/v2/projects/does-not-exist/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 404

    def test_parent_without_source_on_disk_is_409(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        parent.source_local_path = "/nonexistent/path/source.mp4"
        db_session.commit()

        before = self._count_projects(db_session)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 409
        assert self._count_projects(db_session) == before

    def test_parent_not_yet_analyzed_is_409(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        parent.width = None
        parent.height = None
        parent.duration_sec = None
        db_session.commit()

        before = self._count_projects(db_session)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 409
        assert self._count_projects(db_session) == before

    def test_rect_out_of_bounds_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)  # 640x480
        form = dict(_VALID_FORM, x=600, y=0, width=100, height=50)

        before = self._count_projects(db_session)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 400
        assert self._count_projects(db_session) == before

    def test_negative_xy_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        form = dict(_VALID_FORM, x=-5)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 400

    def test_non_positive_width_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        form = dict(_VALID_FORM, width=0)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 400

    def test_negative_corner_radius_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        form = dict(_VALID_FORM, corner_radius_px=-1)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 400

    def test_start_after_end_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        form = dict(_VALID_FORM, start_sec=5.0, end_sec=1.0)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 400

    def test_end_beyond_duration_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)  # duration_sec=10.0
        form = dict(_VALID_FORM, start_sec=0.0, end_sec=20.0)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 400

    def test_no_child_row_left_behind_on_validation_failure(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        before = self._count_projects(db_session)
        before_extractions = len(
            db_session.execute(select(InsertExtraction)).scalars().all()
        )
        form = dict(_VALID_FORM, x=600)  # out of bounds
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=form
        )
        assert response.status_code == 400
        assert self._count_projects(db_session) == before
        after_extractions = len(
            db_session.execute(select(InsertExtraction)).scalars().all()
        )
        assert after_extractions == before_extractions


# ---------------------------------------------------------------------------
# crop_clip failure — no half-created child project
# ---------------------------------------------------------------------------


class TestExtractInsertCropFailure:
    def test_media_error_leaves_no_child_project_or_extraction(
        self, client, db_session, monkeypatch
    ):
        import app.api_v2 as api_v2_module

        def _raise(*args, **kwargs):
            raise MediaError("ffmpeg exploded")

        monkeypatch.setattr(api_v2_module.media_mod, "crop_clip", _raise)

        parent = _make_analyzed_project(db_session)
        before = len(db_session.execute(select(VideoProject)).scalars().all())
        before_extractions = len(
            db_session.execute(select(InsertExtraction)).scalars().all()
        )

        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 500

        after = len(db_session.execute(select(VideoProject)).scalars().all())
        assert after == before  # no half-created child project persisted
        after_extractions = len(
            db_session.execute(select(InsertExtraction)).scalars().all()
        )
        assert after_extractions == before_extractions
        assert client.enqueue_calls == []


# ---------------------------------------------------------------------------
# Web UI: banner / parent listing
# ---------------------------------------------------------------------------


class TestExtractInsertUI:
    def test_child_page_shows_extracted_from_banner(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 201
        child_id = response.json()["child_project_id"]

        page = client.get(f"/v2/projects/{child_id}")
        assert page.status_code == 200
        assert "Extracted from" in page.text
        assert parent.id[:8] in page.text

    def test_parent_page_lists_child_insert(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        response = client.post(
            f"/api/v2/projects/{parent.id}/extract-insert", data=_VALID_FORM
        )
        assert response.status_code == 201
        child_id = response.json()["child_project_id"]

        page = client.get(f"/v2/projects/{parent.id}")
        assert page.status_code == 200
        assert "Extracted inserts" in page.text
        assert child_id[:8] in page.text

    def test_extract_insert_form_present_on_ready_project_page(
        self, client, db_session
    ):
        parent = _make_analyzed_project(db_session)
        page = client.get(f"/v2/projects/{parent.id}")
        assert page.status_code == 200
        assert "extract-insert" in page.text
        assert "Extract insert" in page.text

    def test_preview_time_scrubber_renders_with_duration_max(
        self, client, db_session
    ):
        """The preview-time slider + numeric input render on the ready page with
        max = the project's duration (so the operator can scrub to a mid-video
        frame where the insert is visible)."""
        parent = _make_analyzed_project(db_session)  # duration_sec=10.0
        page = client.get(f"/v2/projects/{parent.id}")
        assert page.status_code == 200
        assert 'id="extract-preview-range"' in page.text
        assert 'id="extract-preview-time"' in page.text
        assert 'max="10.0"' in page.text
        # "set start to this time" affordance next to start_sec
        assert 'id="extract-set-start"' in page.text


# ---------------------------------------------------------------------------
# M4 — merge endpoints (POST /api/v2/runs/{run_id}/merge, retry, get, list)
# ---------------------------------------------------------------------------


def _extract_child(client, db_session, parent, form=None):
    """Extract an insert from *parent*, returning (child_project_id, extraction_id)."""
    response = client.post(
        f"/api/v2/projects/{parent.id}/extract-insert", data=form or _VALID_FORM
    )
    assert response.status_code == 201
    body = response.json()
    return body["child_project_id"], body["extraction_id"]


class TestCreateMergeDefaultsFromExtraction:
    def test_defaults_target_and_placement_from_extraction(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, extraction_id = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["status"] == "queued"

        merge = db_session.get(InsertMerge, body["merge_id"])
        assert merge is not None
        assert merge.target_project_id == parent.id
        extraction = db_session.get(InsertExtraction, extraction_id)
        assert merge.x == extraction.x
        assert merge.y == extraction.y
        assert merge.width == extraction.width
        assert merge.height == extraction.height
        assert merge.start_sec == extraction.start_sec
        assert merge.end_sec == extraction.end_sec
        assert merge.corner_radius_px == extraction.corner_radius_px
        assert merge.status == MergeStatus.queued
        assert client.merge_enqueue_calls == [merge.id]

    def test_explicit_fields_override_extraction_defaults(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, extraction_id = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(
            f"/api/v2/runs/{run.id}/merge",
            data={"x": 1, "y": 2, "corner_radius_px": 20},
        )
        assert response.status_code == 201, response.text
        merge = db_session.get(InsertMerge, response.json()["merge_id"])
        extraction = db_session.get(InsertExtraction, extraction_id)
        assert merge.x == 1
        assert merge.y == 2
        assert merge.corner_radius_px == 20
        # Untouched fields still default from the extraction.
        assert merge.width == extraction.width
        assert merge.height == extraction.height

    def test_explicit_target_project_overrides_extraction_parent(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        other_target = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(
            f"/api/v2/runs/{run.id}/merge",
            data={"target_project_id": other_target.id},
        )
        assert response.status_code == 201, response.text
        merge = db_session.get(InsertMerge, response.json()["merge_id"])
        assert merge.target_project_id == other_target.id


class TestCreateMergeValidation:
    def test_run_not_done_is_409(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id, status=RunStatus.processing)

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        assert response.status_code == 409

    def test_run_without_result_is_409(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)
        run.result_local_path = None
        db_session.commit()

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        assert response.status_code == 409

    def test_missing_run_is_404(self, client, db_session):
        response = client.post("/api/v2/runs/does-not-exist/merge", data={})
        assert response.status_code == 404

    def test_no_extraction_and_no_target_is_400(self, client, db_session):
        # A plain (non-extracted) project's done run has nothing to default from.
        project = _make_analyzed_project(db_session)
        run = _make_done_run(db_session, project.id)

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        assert response.status_code == 400

    def test_no_extraction_with_full_explicit_payload_succeeds(self, client, db_session):
        project = _make_analyzed_project(db_session)
        target = _make_analyzed_project(db_session)
        run = _make_done_run(db_session, project.id)

        response = client.post(
            f"/api/v2/runs/{run.id}/merge",
            data={
                "target_project_id": target.id,
                "x": 5, "y": 5, "width": 50, "height": 50,
                "start_sec": 0.5, "end_sec": 3.0, "corner_radius_px": 0,
            },
        )
        assert response.status_code == 201, response.text

    def test_missing_target_project_is_404(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(
            f"/api/v2/runs/{run.id}/merge",
            data={"target_project_id": "does-not-exist"},
        )
        assert response.status_code == 404

    def test_target_without_source_on_disk_is_409(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        # Break the target's source AFTER extraction (extraction itself needs it).
        parent.source_local_path = "/nonexistent/source.mp4"
        db_session.commit()

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        assert response.status_code == 409

    def test_rect_out_of_target_bounds_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)  # 640x480
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(
            f"/api/v2/runs/{run.id}/merge",
            data={"x": 600, "y": 0, "width": 100, "height": 50},
        )
        assert response.status_code == 400

    def test_negative_corner_radius_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(
            f"/api/v2/runs/{run.id}/merge",
            data={"corner_radius_px": -1},
        )
        assert response.status_code == 400

    def test_time_window_beyond_duration_is_400(self, client, db_session):
        parent = _make_analyzed_project(db_session)  # duration_sec=10.0
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(
            f"/api/v2/runs/{run.id}/merge",
            data={"start_sec": 0.0, "end_sec": 20.0},
        )
        assert response.status_code == 400


class TestMergeRetryAndGet:
    def test_retry_transitions_failed_to_queued_and_reenqueues(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        merge_id = response.json()["merge_id"]

        merge = db_session.get(InsertMerge, merge_id)
        merge.status = MergeStatus.failed
        merge.error_message = "boom"
        db_session.commit()

        retry_response = client.post(f"/api/v2/merges/{merge_id}/retry")
        assert retry_response.status_code == 200, retry_response.text
        assert retry_response.json()["status"] == "queued"
        assert merge_id in client.merge_enqueue_calls

        db_session.refresh(merge)
        assert merge.status == MergeStatus.queued

    def test_retry_non_failed_merge_is_409(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        merge_id = response.json()["merge_id"]  # still queued

        retry_response = client.post(f"/api/v2/merges/{merge_id}/retry")
        assert retry_response.status_code == 409

    def test_retry_missing_merge_is_404(self, client, db_session):
        response = client.post("/api/v2/merges/does-not-exist/retry")
        assert response.status_code == 404

    def test_get_merge_returns_current_state(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        merge_id = response.json()["merge_id"]

        get_response = client.get(f"/api/v2/merges/{merge_id}")
        assert get_response.status_code == 200
        body = get_response.json()
        assert body["id"] == merge_id
        assert body["status"] == "queued"
        assert body["target_project_id"] == parent.id

    def test_get_missing_merge_is_404(self, client, db_session):
        response = client.get("/api/v2/merges/does-not-exist")
        assert response.status_code == 404


class TestListProjectMerges:
    def test_lists_merges_targeting_project_newest_first(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run1 = _make_done_run(db_session, child_id)
        run2 = _make_done_run(db_session, child_id)

        r1 = client.post(f"/api/v2/runs/{run1.id}/merge", data={})
        r2 = client.post(f"/api/v2/runs/{run2.id}/merge", data={})
        assert r1.status_code == 201 and r2.status_code == 201

        response = client.get(f"/api/v2/projects/{parent.id}/merges")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 2
        ids = {m["id"] for m in body}
        assert r1.json()["merge_id"] in ids
        assert r2.json()["merge_id"] in ids

    def test_missing_project_is_404(self, client, db_session):
        response = client.get("/api/v2/projects/does-not-exist/merges")
        assert response.status_code == 404


class TestMergeBackUI:
    def test_run_page_shows_merge_back_form_for_extracted_child_done_run(
        self, client, db_session
    ):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        page = client.get(f"/v2/runs/{run.id}")
        assert page.status_code == 200
        assert "Merge back" in page.text

    def test_run_page_omits_merge_back_form_without_extraction(self, client, db_session):
        project = _make_analyzed_project(db_session)
        run = _make_done_run(db_session, project.id)

        page = client.get(f"/v2/runs/{run.id}")
        assert page.status_code == 200
        assert "Merge back" not in page.text

    def test_target_project_page_lists_merge(self, client, db_session):
        parent = _make_analyzed_project(db_session)
        child_id, _ = _extract_child(client, db_session, parent)
        run = _make_done_run(db_session, child_id)

        response = client.post(f"/api/v2/runs/{run.id}/merge", data={})
        assert response.status_code == 201

        page = client.get(f"/v2/projects/{parent.id}")
        assert page.status_code == 200
        assert "Merges" in page.text
        assert run.id[:8] in page.text
