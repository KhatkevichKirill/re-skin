"""
Tests for the burned-captions v2 REST API additions (app/api_v2.py, per
docs/burned-captions.md §3.5):

    POST   /runs/{rid}/captions
    GET    /runs/{rid}/captions
    POST   /captions/{cid}/retry
    PATCH  /captions/{cid}/lines
    GET    /captions/{cid}/result
    DELETE /captions/{cid}
    POST   /captions/{cid}/export-to-sheet

Follows test_api_v2.py's conventions: own temp SQLite DB + engine (this
module never shares fixtures with other test files), a `client` fixture that
overrides the DB dependency and stubs the caption enqueue functions to
no-ops by default, and a `db_session` fixture for direct ORM setup/asserts.
Per-test monkeypatching of the enqueue functions (via app.api_v2's imported
names, not app.tasks's) mirrors TestTailEdits' spy pattern.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

# Must set DATABASE_URL and DATA_DIR before any app import.
_db_fd, _db_path = tempfile.mkstemp(suffix="_api_captions.db")
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
from app.models import Run, RunCaption, RunTailEdit, VideoProject
from app.state_machine import CaptionStatus, ProjectStatus, RunStatus, TailEditStatus
import app.api_v2 as api_v2_module


@pytest.fixture()
def tmp_path():
    """Per-test scratch dir INSIDE the storage root, shadowing pytest's own.

    Serving and kie.ai uploads refuse any stored path outside DATA_DIR
    (AICP-148), and real stored paths always live there — so fixture files
    must too.
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


@pytest.fixture()
def client(engine, SessionFactory, monkeypatch):
    """TestClient with DB dependency overridden and caption enqueues mocked."""
    monkeypatch.setattr(api_v2_module, "enqueue_process_caption", lambda cid: None)
    monkeypatch.setattr(
        api_v2_module, "enqueue_export_caption_to_sheet", lambda cid: None
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

    # download_caption_result manages its own session via `with get_session()
    # as db:` instead of the overridden get_db dependency (mirrors
    # download_tail_edit_result). app.db's module-level get_session() is bound
    # to whichever test module's DATABASE_URL happened to be in effect when
    # app.db was FIRST imported in this test process — not necessarily this
    # file's own temp DB when multiple test files run together in one pytest
    # session. Patch the name in api_v2 so that endpoint uses this test's
    # engine regardless of import order (mirrors test_api_v2.py's client
    # fixture and test_api_overlay.py's client fixture).
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


def _make_caption(session, run_id: str, **kwargs) -> RunCaption:
    defaults = dict(run_id=run_id, status=CaptionStatus.queued)
    defaults.update(kwargs)
    c = RunCaption(**defaults)
    session.add(c)
    session.commit()
    return c


def _make_tail_edit(session, run_id: str, **kwargs) -> RunTailEdit:
    """Mirrors _make_caption — used here only as a chained-source fixture
    for TestCreateCaption's source_tail_edit_id tests
    (docs/burned-captions.md §3.7)."""
    defaults = dict(run_id=run_id, cut_sec=5.0, status=TailEditStatus.queued)
    defaults.update(kwargs)
    edit = RunTailEdit(**defaults)
    session.add(edit)
    session.commit()
    return edit


def _done_run_with_result(session, tmp_path, **run_kwargs) -> Run:
    """A `done` run whose result_local_path exists on disk — the precondition
    every create_caption happy-path test needs."""
    result_file = tmp_path / "final.mp4"
    result_file.write_bytes(b"\x00VIDEO\xff")
    project = _make_project(session, status=ProjectStatus.ready)
    defaults = dict(status=RunStatus.done, result_local_path=str(result_file))
    defaults.update(run_kwargs)
    return _make_run(session, project.id, **defaults)


# ---------------------------------------------------------------------------
# POST /runs/{rid}/captions
# ---------------------------------------------------------------------------


class TestCreateCaption:
    def test_create_on_non_done_run_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 409

    def test_create_on_done_run_without_result_file_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 409

    def test_create_on_done_run_with_missing_file_on_disk_is_409(
        self, client, db_session
    ):
        """result_local_path is set, but the file was never written / was
        cleaned up — still 409, mirroring create_tail_edit's disk check."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_local_path="/no/such/path/final.mp4",
        )

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 409

    def test_create_with_bad_language_is_400(self, client, db_session, tmp_path):
        run = _done_run_with_result(db_session, tmp_path)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": "nl"}
        )
        assert response.status_code == 400

    def test_create_unknown_run_is_404(self, client):
        response = client.post("/api/v2/runs/no-such-run/captions")
        assert response.status_code == 404

    def test_create_happy_path_creates_row_and_enqueues(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        run = _done_run_with_result(db_session, tmp_path)

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": "ja"}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "queued"
        caption_id = body["caption_id"]
        assert enqueued == [caption_id]

        session = SessionFactory()
        caption = session.get(RunCaption, caption_id)
        session.close()
        assert caption is not None
        assert caption.run_id == run.id
        assert caption.language == "ja"
        assert caption.status == CaptionStatus.queued
        assert caption.lines is None
        assert caption.lines_source is None

    def test_create_language_omitted_is_fine(
        self, client, db_session, tmp_path, monkeypatch
    ):
        """No language field at all (the "Add Captions" button sends none) —
        defers resolution to process time, per docs/burned-captions.md §2.1."""
        run = _done_run_with_result(db_session, tmp_path)

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 201
        assert response.json()["status"] == "queued"
        assert len(enqueued) == 1

    def test_create_with_blank_source_tail_edit_id_is_fine(
        self, client, db_session, tmp_path, monkeypatch
    ):
        """The shape the *browser* posts when the source <select> is left on
        "Run result": a multipart body carrying the field with an empty value.

        The other tests here post either no body at all or urlencoded `data=`,
        neither of which is what the Add Captions button sends. Sending a truly
        empty FormData (no fields appended) produced a body of nothing but the
        closing delimiter, which python-multipart rejects — the operator saw
        400 "There was an error parsing the body" and no caption was ever
        queued. The button now always appends the field, blank or not.
        """
        run = _done_run_with_result(db_session, tmp_path)

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions",
            files={"source_tail_edit_id": (None, "")},
        )
        assert response.status_code == 201
        assert response.json()["status"] == "queued"
        assert len(enqueued) == 1

    def test_create_multiple_jobs_per_run_allowed(
        self, client, db_session, tmp_path, monkeypatch
    ):
        run = _done_run_with_result(db_session, tmp_path)
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", lambda cid: None)

        r1 = client.post(f"/api/v2/runs/{run.id}/captions")
        r2 = client.post(f"/api/v2/runs/{run.id}/captions")
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["caption_id"] != r2.json()["caption_id"]

    def test_create_omitting_source_tail_edit_id_defaults_to_run(
        self, client, db_session, SessionFactory, tmp_path
    ):
        """v1 regression guard (docs/burned-captions.md §3.7.3): no
        source_tail_edit_id field at all still yields source_type='run' and
        a NULL source_tail_edit_id, byte-for-byte the pre-chaining
        behaviour."""
        run = _done_run_with_result(db_session, tmp_path)

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 201
        caption_id = response.json()["caption_id"]

        session = SessionFactory()
        caption = session.get(RunCaption, caption_id)
        session.close()
        assert caption.source_type == "run"
        assert caption.source_tail_edit_id is None

    def test_create_with_source_tail_edit_id_sets_source_type_and_id(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        run = _done_run_with_result(db_session, tmp_path)
        edit_output = tmp_path / "tail_edited.mp4"
        edit_output.write_bytes(b"\x00TAILEDIT\xff")
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            output_path=str(edit_output),
        )

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions",
            data={"source_tail_edit_id": edit.id},
        )
        assert response.status_code == 201
        caption_id = response.json()["caption_id"]
        assert enqueued == [caption_id]

        session = SessionFactory()
        caption = session.get(RunCaption, caption_id)
        session.close()
        assert caption.source_type == "tail_edit"
        assert caption.source_tail_edit_id == edit.id

    def test_create_with_unknown_source_tail_edit_id_is_400(
        self, client, db_session, tmp_path
    ):
        run = _done_run_with_result(db_session, tmp_path)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions",
            data={"source_tail_edit_id": "does-not-exist"},
        )
        assert response.status_code == 400

    def test_create_with_source_tail_edit_from_different_run_is_400(
        self, client, db_session, tmp_path
    ):
        """A tail edit belonging to a DIFFERENT run must not be usable as a
        source even though it is done with a file on disk — 400, not 409."""
        run = _done_run_with_result(db_session, tmp_path)
        other_run = _done_run_with_result(db_session, tmp_path)
        edit_output = tmp_path / "tail_edited.mp4"
        edit_output.write_bytes(b"\x00TAILEDIT\xff")
        edit = _make_tail_edit(
            db_session, other_run.id,
            status=TailEditStatus.done,
            output_path=str(edit_output),
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/captions",
            data={"source_tail_edit_id": edit.id},
        )
        assert response.status_code == 400

    def test_create_with_source_tail_edit_not_done_is_409(
        self, client, db_session, tmp_path
    ):
        run = _done_run_with_result(db_session, tmp_path)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.processing)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions",
            data={"source_tail_edit_id": edit.id},
        )
        assert response.status_code == 409

    def test_create_with_source_tail_edit_done_but_file_missing_is_409(
        self, client, db_session, tmp_path
    ):
        run = _done_run_with_result(db_session, tmp_path)
        edit = _make_tail_edit(
            db_session, run.id,
            status=TailEditStatus.done,
            output_path="/no/such/path/tail_edited.mp4",
        )

        response = client.post(
            f"/api/v2/runs/{run.id}/captions",
            data={"source_tail_edit_id": edit.id},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# GET /runs/{rid}/captions
# ---------------------------------------------------------------------------


class TestListCaptions:
    def test_list_newest_first(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        now = datetime.now(timezone.utc)
        older = _make_caption(db_session, run.id, created_at=now - timedelta(minutes=5))
        newer = _make_caption(db_session, run.id, created_at=now)

        response = client.get(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 200
        body = response.json()
        assert [item["id"] for item in body] == [newer.id, older.id]

    def test_list_includes_expected_fields_and_line_count(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        _make_caption(
            db_session, run.id,
            status=CaptionStatus.done,
            language="ja",
            lines_source="transcribe",
            gdrive_file_id="GDCAP1",
            lines=[
                {"start": 0.0, "end": 1.0, "text": "hi"},
                {"start": 1.0, "end": 2.0, "text": "there"},
            ],
        )

        response = client.get(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        item = body[0]
        assert set(item.keys()) >= {
            "id", "status", "language", "lines_source", "created_at",
            "error_message", "gdrive_file_id", "final_creatives_exported_at",
            "line_count",
        }
        assert item["status"] == "done"
        assert item["language"] == "ja"
        assert item["lines_source"] == "transcribe"
        assert item["gdrive_file_id"] == "GDCAP1"
        assert item["line_count"] == 2

    def test_list_includes_source_fields(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, run.id)
        _make_caption(
            db_session, run.id,
            source_type="tail_edit", source_tail_edit_id=edit.id,
        )
        _make_caption(db_session, run.id)  # default source_type="run"

        response = client.get(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 200
        body = response.json()
        by_source_type = {item["source_type"]: item for item in body}
        assert by_source_type["tail_edit"]["source_tail_edit_id"] == edit.id
        assert by_source_type["run"]["source_tail_edit_id"] is None

    def test_list_line_count_zero_when_no_lines(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        _make_caption(db_session, run.id, status=CaptionStatus.queued)

        response = client.get(f"/api/v2/runs/{run.id}/captions")
        body = response.json()
        assert body[0]["line_count"] == 0

    def test_list_only_returns_this_run_s_captions(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run_a = _make_run(db_session, project.id, status=RunStatus.done)
        run_b = _make_run(db_session, project.id, status=RunStatus.done)
        _make_caption(db_session, run_a.id)
        _make_caption(db_session, run_b.id)

        response = client.get(f"/api/v2/runs/{run_a.id}/captions")
        body = response.json()
        assert len(body) == 1

    def test_list_unknown_run_is_404(self, client):
        response = client.get("/api/v2/runs/no-such-run/captions")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /captions/{cid}/retry
# ---------------------------------------------------------------------------


class TestRetryCaption:
    def test_retry_failed_caption(self, client, db_session, monkeypatch):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(
            db_session, run.id, status=CaptionStatus.failed, error_message="boom"
        )

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(f"/api/v2/captions/{caption.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert body["error_message"] is None
        assert enqueued == [caption.id]

    def test_retry_done_caption(self, client, db_session, monkeypatch):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.done)

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(f"/api/v2/captions/{caption.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert enqueued == [caption.id]

    @pytest.mark.parametrize(
        "mid_flight_status",
        [
            CaptionStatus.queued,
            CaptionStatus.transcribing,
            CaptionStatus.burning,
            CaptionStatus.delivering,
        ],
    )
    def test_retry_mid_flight_is_409(self, client, db_session, mid_flight_status):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=mid_flight_status)

        response = client.post(f"/api/v2/captions/{caption.id}/retry")
        assert response.status_code == 409

    def test_retry_unknown_id_is_404(self, client):
        response = client.post("/api/v2/captions/no-such-caption/retry")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /captions/{cid}/lines
# ---------------------------------------------------------------------------


class TestUpdateCaptionLines:
    def test_patch_happy_path_sets_manual_and_persists_lines(
        self, client, db_session, SessionFactory
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(
            db_session, run.id, status=CaptionStatus.queued, lines_source="transcribe"
        )

        payload = {
            "lines": [
                {"start": 0.0, "end": 1.5, "text": "Hello"},
                {"start": 1.5, "end": 3.0, "text": "world"},
            ]
        }
        response = client.patch(f"/api/v2/captions/{caption.id}/lines", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["lines_source"] == "manual"
        assert body["lines"] == payload["lines"]

        session = SessionFactory()
        reloaded = session.get(RunCaption, caption.id)
        session.close()
        assert reloaded.lines_source == "manual"
        assert reloaded.lines == payload["lines"]

    def test_patch_clears_reuse_provenance(self, client, db_session):
        """An edited copy is the operator's text, no longer the donor's."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(
            db_session, run.id, status=CaptionStatus.done,
            lines_source="reused", lines_reused_from="donor-caption-id",
        )
        response = client.patch(
            f"/api/v2/captions/{caption.id}/lines",
            json={"lines": [{"start": 0.0, "end": 1.0, "text": "fixed"}]},
        )
        assert response.status_code == 200
        assert response.json()["lines_source"] == "manual"
        assert response.json()["lines_reused_from"] is None

    def test_patch_from_done_is_allowed(self, client, db_session):
        """Not "in flight" — an operator may pre-edit before pressing retry."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.done)

        response = client.patch(
            f"/api/v2/captions/{caption.id}/lines",
            json={"lines": [{"start": 0.0, "end": 1.0, "text": "hi"}]},
        )
        assert response.status_code == 200
        assert response.json()["lines_source"] == "manual"

    def test_patch_does_not_reenqueue(self, client, db_session, monkeypatch):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.queued)

        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.patch(
            f"/api/v2/captions/{caption.id}/lines",
            json={"lines": [{"start": 0.0, "end": 1.0, "text": "hi"}]},
        )
        assert response.status_code == 200
        assert enqueued == []

    @pytest.mark.parametrize(
        "in_flight_status",
        [CaptionStatus.transcribing, CaptionStatus.burning, CaptionStatus.delivering],
    )
    def test_patch_mid_flight_is_409(self, client, db_session, in_flight_status):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=in_flight_status)

        response = client.patch(
            f"/api/v2/captions/{caption.id}/lines",
            json={"lines": [{"start": 0.0, "end": 1.0, "text": "hi"}]},
        )
        assert response.status_code == 409

    @pytest.mark.parametrize(
        "bad_body",
        [
            {},  # no "lines" key at all
            {"lines": "not-a-list"},
            {"lines": ["not-a-dict"]},
            {"lines": [{"start": "zero", "end": 1.0, "text": "hi"}]},  # non-numeric start
            {"lines": [{"start": 0.0, "end": 1.0, "text": 123}]},  # non-string text
            {"lines": [{"start": 0.0, "end": 1.0}]},  # missing text
            {"lines": [{"start": -1.0, "end": 1.0, "text": "hi"}]},  # negative
        ],
    )
    def test_patch_malformed_lines_is_400(self, client, db_session, bad_body):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.queued)

        response = client.patch(
            f"/api/v2/captions/{caption.id}/lines", json=bad_body
        )
        assert response.status_code == 400, response.text

    def test_patch_nan_line_timestamp_is_400(self, client, db_session):
        """A NaN start/end must be rejected — it would sail through every
        comparison in captions.prepare_lines silently dropping the line.

        Sent as raw bytes (not the `json=` kwarg) so the wire body really
        contains the bare `NaN` token Python's json module emits/accepts,
        rather than relying on httpx's encoder to round-trip float('nan')
        the same way (mirrors TestProjectMaxSegmentSec._NAN_BODIES in
        test_api_v2.py).
        """
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.queued)

        response = client.patch(
            f"/api/v2/captions/{caption.id}/lines",
            content=b'{"lines": [{"start": NaN, "end": 1.0, "text": "hi"}]}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400, response.text

    def test_patch_unknown_id_is_404(self, client):
        response = client.patch(
            "/api/v2/captions/no-such-caption/lines",
            json={"lines": []},
        )
        assert response.status_code == 404

    def test_patch_empty_lines_array_is_valid(self, client, db_session):
        """An empty list is a legal ('no captions') manual edit, not garbage."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.queued)

        response = client.patch(
            f"/api/v2/captions/{caption.id}/lines", json={"lines": []}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["lines"] == []
        assert body["lines_source"] == "manual"


# ---------------------------------------------------------------------------
# GET /captions/{cid}/result
# ---------------------------------------------------------------------------


class TestDownloadCaptionResult:
    def test_result_download_happy_path(self, client, db_session, tmp_path):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        output_file = tmp_path / "captioned.mp4"
        output_file.write_bytes(b"\x00CAPTIONED\xff")
        caption = _make_caption(
            db_session, run.id,
            status=CaptionStatus.done,
            output_path=str(output_file),
        )

        response = client.get(f"/api/v2/captions/{caption.id}/result")
        assert response.status_code == 200
        assert response.content == b"\x00CAPTIONED\xff"

    def test_result_download_unset_path_is_404(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.done)

        response = client.get(f"/api/v2/captions/{caption.id}/result")
        assert response.status_code == 404

    def test_result_download_missing_on_disk_is_404(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(
            db_session, run.id,
            status=CaptionStatus.done,
            output_path="/no/such/path/captioned.mp4",
        )

        response = client.get(f"/api/v2/captions/{caption.id}/result")
        assert response.status_code == 404

    def test_result_download_not_done_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.queued)

        response = client.get(f"/api/v2/captions/{caption.id}/result")
        assert response.status_code == 409

    def test_result_download_unknown_id_is_404(self, client):
        response = client.get("/api/v2/captions/no-such-caption/result")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /captions/{cid}
# ---------------------------------------------------------------------------


class TestDeleteCaption:
    def test_delete_removes_row_and_dir(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.done)

        caption_dir_path = tmp_path / "caption_out"
        caption_dir_path.mkdir()
        (caption_dir_path / "out.mp4").write_bytes(b"\x00CAPTIONED\xff")
        monkeypatch.setattr(
            api_v2_module, "caption_dir", lambda cid, pid: str(caption_dir_path)
        )

        response = client.delete(f"/api/v2/captions/{caption.id}")
        assert response.status_code == 204
        assert not caption_dir_path.exists()

        session = SessionFactory()
        assert session.get(RunCaption, caption.id) is None
        session.close()

    def test_delete_failed_caption_allowed(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.failed)

        response = client.delete(f"/api/v2/captions/{caption.id}")
        assert response.status_code == 204

    @pytest.mark.parametrize(
        "active_status",
        [
            CaptionStatus.queued,
            CaptionStatus.transcribing,
            CaptionStatus.burning,
            CaptionStatus.delivering,
        ],
    )
    def test_delete_active_caption_is_409(self, client, db_session, active_status):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=active_status)

        response = client.delete(f"/api/v2/captions/{caption.id}")
        assert response.status_code == 409

    def test_delete_unknown_id_is_404(self, client):
        response = client.delete("/api/v2/captions/no-such-caption")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /captions/{cid}/export-to-sheet
# ---------------------------------------------------------------------------


class TestExportCaptionToSheet:
    def test_export_enqueues_and_marks_exported(self, client, db_session, monkeypatch):
        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_caption_to_sheet", enqueued.append
        )

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id,
            status=RunStatus.done,
            result_gdrive_file_id="OUT456",
        )
        caption = _make_caption(
            db_session, run.id,
            status=CaptionStatus.done,
            gdrive_file_id="CAP789",
        )

        response = client.post(f"/api/v2/captions/{caption.id}/export-to-sheet")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "added"
        fields = body["fields"]
        assert fields["Job ID"] == caption.id[:8]
        assert fields["Creative Link"] == "https://drive.google.com/file/d/CAP789/view"
        assert fields["Input Link"] == "https://drive.google.com/file/d/OUT456/view"
        assert fields["Pipeline Type"] == "re-skin-captions"

        db_session.expire_all()
        assert (
            db_session.get(RunCaption, caption.id).final_creatives_exported_at
            is not None
        )
        assert enqueued == [caption.id]

    def test_export_is_idempotent(self, client, db_session, monkeypatch):
        enqueued: list[str] = []
        monkeypatch.setattr(
            api_v2_module, "enqueue_export_caption_to_sheet", enqueued.append
        )

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(
            db_session, run.id, status=CaptionStatus.done, gdrive_file_id="CAP789"
        )

        r1 = client.post(f"/api/v2/captions/{caption.id}/export-to-sheet")
        assert r1.status_code == 200
        assert r1.json()["status"] == "added"
        assert len(enqueued) == 1

        r2 = client.post(f"/api/v2/captions/{caption.id}/export-to-sheet")
        assert r2.status_code == 200
        assert r2.json()["status"] == "exists"
        assert len(enqueued) == 1  # not enqueued again

    def test_export_requires_done_status(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(
            db_session, run.id, status=CaptionStatus.burning, gdrive_file_id="CAP789"
        )

        response = client.post(f"/api/v2/captions/{caption.id}/export-to-sheet")
        assert response.status_code == 409

    def test_export_requires_gdrive_result(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        caption = _make_caption(db_session, run.id, status=CaptionStatus.done)

        response = client.post(f"/api/v2/captions/{caption.id}/export-to-sheet")
        assert response.status_code == 409

    def test_export_missing_caption_is_404(self, client):
        response = client.post("/api/v2/captions/no-such-caption/export-to-sheet")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Italian (it) — caption creation
# ---------------------------------------------------------------------------


class TestItalianCaptionCreation:
    def test_create_with_explicit_italian(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        run = _done_run_with_result(db_session, tmp_path)
        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": "  IT "}
        )
        assert response.status_code == 201, response.text
        caption_id = response.json()["caption_id"]
        assert enqueued == [caption_id]

        session = SessionFactory()
        caption = session.get(RunCaption, caption_id)
        session.close()
        assert caption.language == "it"     # normalised, not "  IT "

    def test_create_on_an_italian_run_without_a_language_defers(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        """Omitting the field still stores NULL — the run's 'it' is resolved at
        PROCESS time, not baked in at creation (docs/burned-captions.md §2.1)."""
        run = _done_run_with_result(db_session, tmp_path, language="it")
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", lambda cid: None)

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 201, response.text

        session = SessionFactory()
        caption = session.get(RunCaption, response.json()["caption_id"])
        session.close()
        assert caption.language is None

    @pytest.mark.parametrize("alias", ["ita", "it-IT", "italian"])
    def test_create_rejects_italian_aliases(self, client, db_session, tmp_path, alias):
        run = _done_run_with_result(db_session, tmp_path)
        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": alias}
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Polish (pl) — caption creation
# ---------------------------------------------------------------------------


class TestPolishCaptionCreation:
    def test_create_with_explicit_polish(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        run = _done_run_with_result(db_session, tmp_path)
        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": "  PL "}
        )
        assert response.status_code == 201, response.text
        caption_id = response.json()["caption_id"]
        assert enqueued == [caption_id]

        session = SessionFactory()
        caption = session.get(RunCaption, caption_id)
        session.close()
        assert caption.language == "pl"     # normalised, not "  PL "

    def test_create_on_a_polish_run_without_a_language_defers(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        """Omitting the field still stores NULL — the run's 'pl' is resolved at
        PROCESS time, not baked in at creation (docs/burned-captions.md §2.1)."""
        run = _done_run_with_result(db_session, tmp_path, language="pl")
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", lambda cid: None)

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 201, response.text

        session = SessionFactory()
        caption = session.get(RunCaption, response.json()["caption_id"])
        session.close()
        assert caption.language is None

    @pytest.mark.parametrize("alias", ["pol", "pl-PL", "polish"])
    def test_create_rejects_polish_aliases(self, client, db_session, tmp_path, alias):
        run = _done_run_with_result(db_session, tmp_path)
        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": alias}
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# French (fr) — caption creation
# ---------------------------------------------------------------------------


class TestFrenchCaptionCreation:
    def test_create_with_explicit_french(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        run = _done_run_with_result(db_session, tmp_path)
        enqueued: list[str] = []
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", enqueued.append)

        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": "  FR "}
        )
        assert response.status_code == 201, response.text
        caption_id = response.json()["caption_id"]
        assert enqueued == [caption_id]

        session = SessionFactory()
        caption = session.get(RunCaption, caption_id)
        session.close()
        assert caption.language == "fr"     # normalised, not "  FR "

    def test_create_on_a_french_run_without_a_language_defers(
        self, client, db_session, SessionFactory, tmp_path, monkeypatch
    ):
        """Omitting the field still stores NULL — the run's 'fr' is resolved at
        PROCESS time, not baked in at creation (docs/burned-captions.md §2.1)."""
        run = _done_run_with_result(db_session, tmp_path, language="fr")
        monkeypatch.setattr(api_v2_module, "enqueue_process_caption", lambda cid: None)

        response = client.post(f"/api/v2/runs/{run.id}/captions")
        assert response.status_code == 201, response.text

        session = SessionFactory()
        caption = session.get(RunCaption, response.json()["caption_id"])
        session.close()
        assert caption.language is None

    @pytest.mark.parametrize("alias", ["fra", "fr-FR", "french"])
    def test_create_rejects_french_aliases(self, client, db_session, tmp_path, alias):
        run = _done_run_with_result(db_session, tmp_path)
        response = client.post(
            f"/api/v2/runs/{run.id}/captions", data={"language": alias}
        )
        assert response.status_code == 400
