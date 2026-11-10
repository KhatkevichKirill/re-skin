"""
Tests for the overlay REST API (app/api_overlay.py).

Strategy mirrors test_api_v2.py: temp SQLite DB, get_db overridden, the RQ
enqueue monkeypatched to a spy. Covers create -> upload asset -> put overlays ->
submit, asserting the job ends queued and the burn was enqueued.
"""

from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import zlib
from contextlib import contextmanager

# Must set DATABASE_URL / DATA_DIR before importing the app.
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
from app.models import OverlayJob, Run, RunTailEdit, VideoProject
from app.state_machine import OverlayStatus, ProjectStatus, RunStatus, TailEditStatus


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


def _png_bytes(w=4, h=4):
    """Return a minimal valid RGBA PNG (signature + IHDR + IDAT + IEND)."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    raw = b"".join(b"\x00" + b"\xff\x00\x00\x80" * w for _ in range(h))
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})

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


@pytest.fixture()
def done_run(SessionFactory):
    """A completed Run with a result file on disk (a valid overlay source)."""
    s = SessionFactory()
    proj = VideoProject(
        source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready,
        name="Test Project", width=1080, height=1920, duration_sec=10.0,
    )
    s.add(proj)
    s.flush()
    result_path = os.path.join(_data_tmp, "result.mp4")
    with open(result_path, "wb") as fh:
        fh.write(b"\x00" * 64)  # placeholder file; existence is what matters
    run = Run(
        project_id=proj.id, prompt="p", reference_image_urls=[],
        resolution="1080p", status=RunStatus.done, result_local_path=result_path,
    )
    s.add(run)
    s.commit()
    rid = run.id
    s.close()
    return rid


@pytest.fixture()
def done_tail_edit(SessionFactory, done_run):
    """A completed RunTailEdit (off ``done_run``) with an output file on disk."""
    s = SessionFactory()
    output_path = os.path.join(_data_tmp, "tail_edit_result.mp4")
    with open(output_path, "wb") as fh:
        fh.write(b"\x00" * 64)  # placeholder file; existence is what matters
    edit = RunTailEdit(
        run_id=done_run, cut_sec=3.0, status=TailEditStatus.done,
        output_path=output_path,
    )
    s.add(edit)
    s.commit()
    eid = edit.id
    s.close()
    return eid


@pytest.fixture()
def spy(monkeypatch):
    calls = {"process_overlay": [], "export_overlay": []}
    import app.api_overlay as mod
    monkeypatch.setattr(mod, "enqueue_process_overlay", lambda jid: calls["process_overlay"].append(jid))
    monkeypatch.setattr(mod, "enqueue_export_overlay_to_sheet", lambda jid: calls["export_overlay"].append(jid))
    return calls


@pytest.fixture()
def client(engine, SessionFactory, spy, monkeypatch):
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

    import app.api_overlay as api_overlay_module
    import app.public as public_module
    import app.web_overlay as web_overlay_module

    monkeypatch.setattr(api_overlay_module, "get_session", _test_get_session)
    monkeypatch.setattr(public_module, "get_session", _test_get_session)
    monkeypatch.setattr(web_overlay_module, "get_session", _test_get_session)

    # Same hazard, one layer deeper: the RQ export tasks bypass get_session and
    # do `from .db import SessionLocal` INSIDE the function body, so they
    # resolve the name off app.db at call time — the singleton bound to
    # whichever test module imported app.db first. Tests that invoke a task
    # directly (rather than through a mocked enqueue) therefore read from a
    # different database than the one they wrote their fixtures into, and pass
    # or fail purely on test-file ordering. Patching the attribute on app.db
    # itself is what the late import actually looks up.
    import app.db as db_module

    monkeypatch.setattr(db_module, "SessionLocal", SessionFactory)

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


class TestOverlayFlow:
    def test_sources_lists_done_run(self, client, done_run):
        resp = client.get("/api/v2/overlays/sources")
        assert resp.status_code == 200
        ids = [s["run_id"] for s in resp.json()]
        assert done_run in ids

    def test_sources_respects_limit(self, client, SessionFactory):
        """Verify that limit parameter restricts results to newest N runs."""
        # Create 5 done runs with the same project
        s = SessionFactory()
        proj = VideoProject(
            source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready,
            name="Test Project", width=1080, height=1920, duration_sec=10.0,
        )
        s.add(proj)
        s.flush()
        result_path = os.path.join(_data_tmp, "result_limit.mp4")
        with open(result_path, "wb") as fh:
            fh.write(b"\x00" * 64)

        run_ids = []
        for i in range(5):
            run = Run(
                project_id=proj.id, prompt=f"plimit{i}", reference_image_urls=[],
                resolution="1080p", status=RunStatus.done, result_local_path=result_path,
            )
            s.add(run)
            s.flush()
            run_ids.append(run.id)
        s.commit()
        s.close()

        # Fetch with limit=2, should get newest 2 (run_ids[-2:])
        resp = client.get("/api/v2/overlays/sources?limit=2")
        assert resp.status_code == 200
        results = resp.json()
        # Filter to just our runs
        my_runs = [r for r in results if r["run_id"] in run_ids]
        assert len(my_runs) == 2
        # Results should be ordered newest first (desc created_at)
        result_ids = [r["run_id"] for r in my_runs]
        assert result_ids == list(reversed(run_ids[-2:]))

    def test_sources_newest_first(self, client, SessionFactory):
        """Verify that sources are returned in newest-first order."""
        # Create 3 done runs with the same project
        s = SessionFactory()
        proj = VideoProject(
            source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready,
            name="Test Project", width=1080, height=1920, duration_sec=10.0,
        )
        s.add(proj)
        s.flush()
        result_path = os.path.join(_data_tmp, "result_newest_first.mp4")
        with open(result_path, "wb") as fh:
            fh.write(b"\x00" * 64)

        run_ids = []
        for i in range(3):
            run = Run(
                project_id=proj.id, prompt=f"pnf{i}", reference_image_urls=[],
                resolution="1080p", status=RunStatus.done, result_local_path=result_path,
            )
            s.add(run)
            s.flush()
            run_ids.append(run.id)
        s.commit()
        s.close()

        resp = client.get("/api/v2/overlays/sources")
        assert resp.status_code == 200
        results = resp.json()
        # Filter to just our runs
        my_runs = [r for r in results if r["run_id"] in run_ids]
        result_ids = [r["run_id"] for r in my_runs]
        # Should be newest (last created) first
        assert result_ids == list(reversed(run_ids))

    def test_sources_project_fields_populated(self, client, done_run):
        """Verify that project fields are eagerly loaded and available."""
        resp = client.get("/api/v2/overlays/sources")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) > 0
        # Find our test run
        item = next((r for r in results if r["run_id"] == done_run), None)
        assert item is not None
        assert item["project_name"] == "Test Project"
        assert item["width"] == 1080
        assert item["height"] == 1920
        assert item["duration_sec"] == 10.0

    def test_full_flow_create_upload_put_submit(self, client, done_run, spy):
        # create
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]
        assert resp.json()["status"] == "created"

        # upload asset
        resp = client.post(
            f"/api/v2/overlays/{job_id}/assets",
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        asset_path = resp.json()["asset_path"]
        assert asset_path.startswith("assets/")

        # put overlays
        resp = client.put(
            f"/api/v2/overlays/{job_id}/overlays",
            json={"overlays": [{
                "asset_path": asset_path, "nx": 0.1, "ny": 0.7,
                "sw": 0.4, "start_sec": 1.0, "end_sec": 5.0, "fade_sec": 0.3,
            }]},
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["overlays"]) == 1

        # submit
        resp = client.post(f"/api/v2/overlays/{job_id}/submit")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "queued"
        assert spy["process_overlay"] == [job_id]

    def test_submit_without_overlays_409(self, client, done_run):
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        resp = client.post(f"/api/v2/overlays/{job_id}/submit")
        assert resp.status_code == 409

    def test_non_png_asset_rejected(self, client, done_run):
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        resp = client.post(
            f"/api/v2/overlays/{job_id}/assets",
            files={"file": ("x.png", io.BytesIO(b"not a png"), "image/png")},
        )
        assert resp.status_code == 400

    def test_bad_overlay_coords_rejected_by_schema(self, client, done_run):
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        # nx > 1 is out of range -> 422 from pydantic
        resp = client.put(
            f"/api/v2/overlays/{job_id}/overlays",
            json={"overlays": [{
                "asset_path": "assets/card_01.png", "nx": 1.5, "ny": 0.5,
                "start_sec": 0, "end_sec": 2,
            }]},
        )
        assert resp.status_code == 422

    def test_create_run_source_must_be_done(self, client, SessionFactory):
        # A run that is not done cannot be an overlay source.
        s = SessionFactory()
        proj = VideoProject(source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready)
        s.add(proj); s.flush()
        run = Run(project_id=proj.id, prompt="p", reference_image_urls=[],
                  resolution="1080p", status=RunStatus.processing)
        s.add(run); s.commit()
        rid = run.id
        s.close()
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": rid})
        assert resp.status_code == 409

    def test_create_run_source_needs_final_video(self, client, SessionFactory):
        # A no-stitch run (stitch_plan == []) is done WITHOUT a final video —
        # it must be rejected as an overlay source instead of failing at render.
        s = SessionFactory()
        proj = VideoProject(source_type="upload", source_ref="x.mp4", status=ProjectStatus.ready)
        s.add(proj); s.flush()
        run = Run(project_id=proj.id, prompt="p", reference_image_urls=[],
                  resolution="1080p", status=RunStatus.done, stitch_plan=[])
        s.add(run); s.commit()
        rid = run.id
        s.close()
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": rid})
        assert resp.status_code == 409
        assert "no final video" in resp.json()["detail"]

    def test_create_defaults_deliver_to_drive_true(self, client, done_run):
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        got = client.get(f"/api/v2/overlays/{job_id}").json()
        assert got["deliver_to_drive"] is True

    def _make_asset(self, client, job_id):
        r = client.post(
            f"/api/v2/overlays/{job_id}/assets",
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert r.status_code == 200, r.text
        return r.json()["asset_path"]

    def test_render_first_time_renders_in_place(self, client, done_run, spy):
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        asset = self._make_asset(client, job_id)
        resp = client.post(
            f"/api/v2/overlays/{job_id}/render",
            json={"overlays": [{
                "asset_path": asset, "nx": 0.1, "ny": 0.7, "sw": 0.4,
                "start_sec": 0, "end_sec": 5, "fade_sec": 0,
            }]},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["id"] == job_id            # first render is in place
        assert resp.json()["status"] == "queued"
        assert spy["process_overlay"] == [job_id]

    def test_render_without_overlays_409(self, client, done_run):
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        resp = client.post(f"/api/v2/overlays/{job_id}/render", json={"overlays": []})
        assert resp.status_code == 409

    def test_rerender_forks_a_new_record(self, client, done_run, spy, SessionFactory):
        # A job that already has a rendered result forks a NEW OverlayJob on the
        # next render — the prior result is preserved and each render is its own
        # DB record (unlimited renders per source run).
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        asset = self._make_asset(client, job_id)

        # Simulate a completed first render.
        s = SessionFactory()
        row = s.get(OverlayJob, job_id)
        row.status = OverlayStatus.done
        row.result_local_path = "/tmp/first-render.mp4"
        row.gdrive_file_id = "drive-first"
        row.overlays = [{"asset_path": asset, "nx": 0.1, "ny": 0.7, "sw": 0.4,
                         "start_sec": 0, "end_sec": 5, "fade_sec": 0}]
        s.commit(); s.close()

        # Re-render with an edited placement (ny changed).
        resp = client.post(
            f"/api/v2/overlays/{job_id}/render",
            json={"overlays": [{
                "asset_path": asset, "nx": 0.2, "ny": 0.3, "sw": 0.5,
                "start_sec": 0, "end_sec": 4, "fade_sec": 0,
            }]},
        )
        assert resp.status_code == 200, resp.text
        new_id = resp.json()["id"]
        assert new_id != job_id                       # forked a new record
        assert resp.json()["status"] == "queued"
        assert spy["process_overlay"] == [new_id]

        s = SessionFactory()
        try:
            orig = s.get(OverlayJob, job_id)
            clone = s.get(OverlayJob, new_id)
            # Original untouched — keeps its result + source run.
            assert orig.status == OverlayStatus.done
            assert orig.result_local_path == "/tmp/first-render.mp4"
            # Clone inherits the source, gets the edited overlays, is queued.
            assert clone.source_run_id == done_run
            assert clone.status == OverlayStatus.queued
            assert clone.overlays[0]["ny"] == 0.3
            assert clone.result_local_path is None
        finally:
            s.close()
        # The PNG asset was copied into the clone's own dir.
        from app.storage import overlay_job_dir
        assert os.path.exists(os.path.join(overlay_job_dir(new_id), asset))

    def test_delete_soft_deletes_hides_but_keeps_row(self, client, done_run, SessionFactory):
        # Deleting an overlay hides it (404 from the API) but keeps the DB row —
        # with its source_run_id intact — so the source video stays findable.
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]

        resp = client.delete(f"/api/v2/overlays/{job_id}")
        assert resp.status_code == 204
        # Hidden from the API.
        assert client.get(f"/api/v2/overlays/{job_id}").status_code == 404
        # But the row survives with deleted_at set and the source reference kept.
        s = SessionFactory()
        try:
            row = s.get(OverlayJob, job_id)
            assert row is not None
            assert row.deleted_at is not None
            assert row.source_run_id == done_run
        finally:
            s.close()

    def test_deleted_job_excluded_from_dashboard(self, client, done_run):
        resp = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        job_id = resp.json()["id"]
        client.delete(f"/api/v2/overlays/{job_id}")
        # The overlays dashboard must not list a soft-deleted job.
        html = client.get("/v2/overlays/").text
        assert job_id[:8] not in html

    def test_delete_removes_render_files_keeps_source(self, client, done_run, SessionFactory):
        # Upload-source job: the source video must survive delete; render files go.
        from app.storage import overlay_job_dir
        resp = client.post("/api/v2/overlays", json={"source_type": "upload"})
        job_id = resp.json()["id"]
        # Stage a fake source + render output + asset on disk.
        root = overlay_job_dir(job_id)
        src = os.path.join(root, "source.mp4")
        with open(src, "wb") as fh:
            fh.write(b"\x00" * 8)
        os.makedirs(os.path.join(root, "output"), exist_ok=True)
        with open(os.path.join(root, "output", "result.mp4"), "wb") as fh:
            fh.write(b"\x00" * 8)
        os.makedirs(os.path.join(root, "assets"), exist_ok=True)
        with open(os.path.join(root, "assets", "card_01.png"), "wb") as fh:
            fh.write(_png_bytes())

        resp = client.delete(f"/api/v2/overlays/{job_id}")
        assert resp.status_code == 204
        assert os.path.exists(src)                                   # source kept
        assert not os.path.exists(os.path.join(root, "output"))      # render gone
        assert not os.path.exists(os.path.join(root, "assets"))      # PNGs gone

    def test_add_asset_from_drive(self, client, done_run, monkeypatch):
        import app.api_overlay as mod
        png = _png_bytes()

        class FakeGDrive:
            def download_file(self, link, dst):
                with open(dst, "wb") as fh:
                    fh.write(png)
                return dst

        monkeypatch.setattr(mod, "GDriveClient", lambda: FakeGDrive())
        job_id = client.post(
            "/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run}
        ).json()["id"]
        resp = client.post(
            f"/api/v2/overlays/{job_id}/assets/from-drive",
            json={"link": "https://drive.google.com/file/d/abc123/view"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["asset_path"].startswith("assets/")

    def test_add_asset_from_drive_rejects_non_png(self, client, done_run, monkeypatch):
        import app.api_overlay as mod

        class FakeGDrive:
            def download_file(self, link, dst):
                with open(dst, "wb") as fh:
                    fh.write(b"not a png at all")
                return dst

        monkeypatch.setattr(mod, "GDriveClient", lambda: FakeGDrive())
        job_id = client.post(
            "/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run}
        ).json()["id"]
        resp = client.post(
            f"/api/v2/overlays/{job_id}/assets/from-drive", json={"link": "abc123"}
        )
        assert resp.status_code == 400

    def test_export_to_sheet_enqueues_and_marks_exported(
        self, client, done_run, monkeypatch, SessionFactory
    ):
        """Endpoint enqueues the Sheets job and optimistically sets exported_at."""
        import app.api_overlay as mod
        from app.models import OverlayJob
        from app.state_machine import OverlayStatus

        enqueued: list[str] = []
        monkeypatch.setattr(mod, "enqueue_export_overlay_to_sheet", enqueued.append)

        s = SessionFactory()
        job = OverlayJob(source_type="run", source_run_id=done_run,
                         status=OverlayStatus.done, gdrive_file_id="drivefile123",
                         overlays=[])
        s.add(job); s.commit(); jid = job.id; s.close()

        resp = client.post(f"/api/v2/overlays/{jid}/export-to-sheet")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "added"

        # Job is marked exported immediately (optimistic).
        s = SessionFactory()
        db_job = s.get(OverlayJob, jid)
        assert db_job.final_creatives_exported_at is not None
        s.close()

        assert enqueued == [jid]

        # Idempotent second call returns "exists" via the DB fast-path.
        resp2 = client.post(f"/api/v2/overlays/{jid}/export-to-sheet")
        assert resp2.json()["status"] == "exists"
        # No second enqueue.
        assert len(enqueued) == 1

    def test_export_to_sheet_refuses_a_registered_manual_creative(
        self, client, SessionFactory, spy
    ):
        """An overlay registered in manual_creatives exports only from there.

        Exporting it here too would write the same Drive file twice, under two
        different Job IDs.
        """
        from app.models import ManualCreative, OverlayJob
        from app.state_machine import OverlayStatus
        s = SessionFactory()
        job = OverlayJob(source_type="upload", status=OverlayStatus.done,
                         gdrive_file_id="manualfile1", overlays=[])
        s.add(job); s.flush()
        s.add(ManualCreative(pipeline_type="re-skin-manual-script", idea="x-Sport-DE",
                             language="de", gdrive_file_id="manualfile1",
                             overlay_job_id=job.id))
        s.commit(); jid = job.id; s.close()

        resp = client.post(f"/api/v2/overlays/{jid}/export-to-sheet")

        assert resp.status_code == 409
        assert "manual creative" in resp.json()["detail"]
        assert spy["export_overlay"] == []
        s = SessionFactory()
        assert s.get(OverlayJob, jid).final_creatives_exported_at is None
        s.close()

    def test_export_to_sheet_requires_done(self, client, done_run, SessionFactory):
        from app.models import OverlayJob
        from app.state_machine import OverlayStatus
        s = SessionFactory()
        job = OverlayJob(source_type="run", source_run_id=done_run,
                         status=OverlayStatus.created, overlays=[])
        s.add(job); s.commit(); jid = job.id; s.close()
        resp = client.post(f"/api/v2/overlays/{jid}/export-to-sheet")
        assert resp.status_code == 409

    def test_export_caption_demo_is_404(self, client, done_run, SessionFactory):
        """Regression: the removed "Add to Caption Demo" overlay export stays gone."""
        from app.models import OverlayJob
        from app.state_machine import OverlayStatus
        s = SessionFactory()
        job = OverlayJob(source_type="run", source_run_id=done_run,
                         status=OverlayStatus.done, gdrive_file_id="drivefile123",
                         overlays=[])
        s.add(job); s.commit(); jid = job.id; s.close()
        resp = client.post(f"/api/v2/overlays/{jid}/export-caption-demo")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Overlay asset library — reusable PNG cards (/api/v2/overlays/library-assets)
# ---------------------------------------------------------------------------


class TestOverlayAssetLibrary:
    def test_upload_list_preview_delete_roundtrip(self, client, SessionFactory):
        resp = client.post(
            "/api/v2/overlays/library-assets",
            data={"name": "Logo bug"},
            files={"file": ("logo.png", io.BytesIO(_png_bytes(8, 6)), "image/png")},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "Logo bug"
        assert body["width"] == 8
        assert body["height"] == 6
        asset_id = body["id"]

        # File landed on disk under the overlay library dir.
        from app.models import OverlayAsset

        session = SessionFactory()
        asset = session.get(OverlayAsset, asset_id)
        session.close()
        assert asset is not None
        assert os.path.exists(asset.local_path)

        # Listed
        listing = client.get("/api/v2/overlays/library-assets")
        assert listing.status_code == 200
        assert asset_id in [a["id"] for a in listing.json()]

        # File endpoint streams it
        file_resp = client.get(f"/api/v2/overlays/library-assets/{asset_id}/file")
        assert file_resp.status_code == 200

        # Delete removes row + file
        del_resp = client.delete(f"/api/v2/overlays/library-assets/{asset_id}")
        assert del_resp.status_code == 204
        session = SessionFactory()
        assert session.get(OverlayAsset, asset_id) is None
        session.close()
        assert not os.path.exists(asset.local_path)

    def test_upload_name_defaults_to_filename(self, client):
        resp = client.post(
            "/api/v2/overlays/library-assets",
            files={"file": ("my card final.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["name"] == "my card final.png"

    def test_upload_rejects_non_png(self, client):
        resp = client.post(
            "/api/v2/overlays/library-assets",
            data={"name": "not a png"},
            files={"file": ("evil.txt", io.BytesIO(b"not a png at all"), "text/plain")},
        )
        assert resp.status_code == 400
        assert "PNG" in resp.json()["detail"]

    def test_create_requires_exactly_one_of_file_or_drive(self, client):
        resp = client.post("/api/v2/overlays/library-assets", data={"name": "nothing"})
        assert resp.status_code == 400

        resp = client.post(
            "/api/v2/overlays/library-assets",
            data={"name": "both", "gdrive_url": "https://drive.google.com/file/d/abc/view"},
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert resp.status_code == 400

    def test_gdrive_url_succeeds(self, client, monkeypatch):
        import app.api_overlay as mod
        png = _png_bytes()

        class FakeGDrive:
            def download_file(self, link, dst):
                with open(dst, "wb") as fh:
                    fh.write(png)
                return dst

        monkeypatch.setattr(mod, "GDriveClient", lambda: FakeGDrive())
        resp = client.post(
            "/api/v2/overlays/library-assets",
            data={"gdrive_url": "https://drive.google.com/file/d/abc123def456/view"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["name"] == "abc123def456"

    def test_gdrive_url_rejects_non_png(self, client, monkeypatch):
        import app.api_overlay as mod

        class FakeGDrive:
            def download_file(self, link, dst):
                with open(dst, "wb") as fh:
                    fh.write(b"not a png")
                return dst

        monkeypatch.setattr(mod, "GDriveClient", lambda: FakeGDrive())
        resp = client.post(
            "/api/v2/overlays/library-assets",
            data={"gdrive_url": "https://drive.google.com/file/d/abc123/view"},
        )
        assert resp.status_code == 400

    def test_unknown_asset_is_404(self, client):
        assert client.get("/api/v2/overlays/library-assets/nope/file").status_code == 404
        assert client.delete("/api/v2/overlays/library-assets/nope").status_code == 404

    def test_attach_to_job_copies_file_and_matches_upload_shape(self, client, done_run):
        # Add a library card.
        resp = client.post(
            "/api/v2/overlays/library-assets",
            data={"name": "Attach me"},
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        asset_id = resp.json()["id"]

        # Create a job and attach the card.
        job_id = client.post(
            "/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run}
        ).json()["id"]
        resp = client.post(f"/api/v2/overlays/{job_id}/library-assets/{asset_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body.keys()) == {"asset_path"}
        asset_path = body["asset_path"]
        assert asset_path.startswith("assets/")

        # The copy is independent of the library row: deleting the library
        # asset must not disturb the job's own copy.
        from app.storage import overlay_job_dir

        job_file = os.path.join(overlay_job_dir(job_id), asset_path)
        assert os.path.exists(job_file)

        del_resp = client.delete(f"/api/v2/overlays/library-assets/{asset_id}")
        assert del_resp.status_code == 204
        assert os.path.exists(job_file)  # job's copy survives

        # The attached asset_path is usable exactly like an uploaded one —
        # PUT /overlays accepts it.
        resp = client.put(
            f"/api/v2/overlays/{job_id}/overlays",
            json={"overlays": [{
                "asset_path": asset_path, "nx": 0.1, "ny": 0.7,
                "sw": 0.4, "start_sec": 1.0, "end_sec": 5.0, "fade_sec": 0.3,
            }]},
        )
        assert resp.status_code == 200, resp.text

    def test_attach_unknown_asset_is_404(self, client, done_run):
        job_id = client.post(
            "/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run}
        ).json()["id"]
        resp = client.post(f"/api/v2/overlays/{job_id}/library-assets/nope")
        assert resp.status_code == 404

    def test_attach_to_unknown_job_is_404(self, client):
        resp = client.post(
            "/api/v2/overlays/library-assets",
            data={"name": "orphan target"},
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        asset_id = resp.json()["id"]
        resp = client.post(f"/api/v2/overlays/nope/library-assets/{asset_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tail edit as an overlay source (the "Change ending" tool's result can be
# burned onto, alongside a run's result or a directly-uploaded video).
# ---------------------------------------------------------------------------


class TestOverlayTailEditSource:
    def test_create_from_tail_edit_succeeds(self, client, done_tail_edit):
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "tail_edit", "source_tail_edit_id": done_tail_edit},
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]
        got = client.get(f"/api/v2/overlays/{job_id}").json()
        assert got["source_type"] == "tail_edit"
        assert got["source_tail_edit_id"] == done_tail_edit
        assert got["source_run_id"] is None

    def test_create_tail_edit_source_must_be_done(self, client, done_run, SessionFactory):
        # A tail edit that hasn't finished cannot be an overlay source.
        s = SessionFactory()
        edit = RunTailEdit(run_id=done_run, cut_sec=2.0, status=TailEditStatus.processing)
        s.add(edit); s.commit()
        eid = edit.id
        s.close()
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "tail_edit", "source_tail_edit_id": eid},
        )
        assert resp.status_code == 409

    def test_create_tail_edit_source_needs_output_file(self, client, done_run, SessionFactory):
        # Done in the DB but the output file isn't on disk — reject at create
        # time instead of failing at render time.
        s = SessionFactory()
        edit = RunTailEdit(
            run_id=done_run, cut_sec=2.0, status=TailEditStatus.done,
            output_path="/nope/missing.mp4",
        )
        s.add(edit); s.commit()
        eid = edit.id
        s.close()
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "tail_edit", "source_tail_edit_id": eid},
        )
        assert resp.status_code == 409
        assert "no output video" in resp.json()["detail"]

    def test_create_tail_edit_source_unknown_id_404(self, client):
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "tail_edit", "source_tail_edit_id": "does-not-exist"},
        )
        assert resp.status_code == 404

    def test_rerender_preserves_tail_edit_source(self, client, done_tail_edit, spy, SessionFactory):
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "tail_edit", "source_tail_edit_id": done_tail_edit},
        )
        job_id = resp.json()["id"]
        asset_resp = client.post(
            f"/api/v2/overlays/{job_id}/assets",
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert asset_resp.status_code == 200, asset_resp.text
        asset = asset_resp.json()["asset_path"]

        # Simulate a completed first render.
        s = SessionFactory()
        row = s.get(OverlayJob, job_id)
        row.status = OverlayStatus.done
        row.result_local_path = "/tmp/first-render-tail.mp4"
        row.gdrive_file_id = "drive-first-tail"
        row.overlays = [{"asset_path": asset, "nx": 0.1, "ny": 0.7, "sw": 0.4,
                         "start_sec": 0, "end_sec": 5, "fade_sec": 0}]
        s.commit(); s.close()

        resp = client.post(
            f"/api/v2/overlays/{job_id}/render",
            json={"overlays": [{
                "asset_path": asset, "nx": 0.2, "ny": 0.3, "sw": 0.5,
                "start_sec": 0, "end_sec": 4, "fade_sec": 0,
            }]},
        )
        assert resp.status_code == 200, resp.text
        new_id = resp.json()["id"]
        assert new_id != job_id

        s = SessionFactory()
        try:
            clone = s.get(OverlayJob, new_id)
            assert clone.source_type == "tail_edit"
            assert clone.source_tail_edit_id == done_tail_edit
        finally:
            s.close()

    def test_export_to_sheet_input_link_uses_tail_edit_drive_link(
        self, client, done_tail_edit, monkeypatch, SessionFactory
    ):
        """run_export_overlay_to_sheet's Input Link resolves the tail edit's own
        Drive file when the overlay job's source is a tail edit (not a run)."""
        import app.sheets_client as sc
        import app.tasks as tasks_mod

        s = SessionFactory()
        edit = s.get(RunTailEdit, done_tail_edit)
        edit.gdrive_file_id = "tail-drive-xyz"
        job = OverlayJob(source_type="tail_edit", source_tail_edit_id=done_tail_edit,
                         status=OverlayStatus.done, gdrive_file_id="overlay-drive-abc",
                         overlays=[])
        s.add(job); s.commit(); jid = job.id; s.close()

        class _FakeSheets:
            HEADER = ["Date", "Job ID", "Input Link", "Creative Link", "Pipeline Type", "Run on Meta"]

            def __init__(self):
                self.appended = []

            def get_header(self, sid, tab):
                return list(self.HEADER)

            def column_values(self, sid, tab, idx, *, last_n=None):
                return []  # no existing rows -> dedupe check finds nothing

            def append_row(self, sid, tab, row):
                self.appended.append(row)

        fake = _FakeSheets()
        monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

        tasks_mod.run_export_overlay_to_sheet(jid)

        assert len(fake.appended) == 1
        row = fake.appended[0]
        input_idx = _FakeSheets.HEADER.index("Input Link")
        creative_idx = _FakeSheets.HEADER.index("Creative Link")
        assert row[input_idx] == "https://drive.google.com/file/d/tail-drive-xyz/view"
        assert row[creative_idx] == "https://drive.google.com/file/d/overlay-drive-abc/view"

    def test_resolve_source_reads_tail_edit_output_path(self, done_tail_edit, SessionFactory):
        """overlay_pipeline._resolve_source resolves a tail_edit-sourced job to
        the tail edit's own output_path on disk."""
        from app.overlay_pipeline import _resolve_source

        s = SessionFactory()
        try:
            job = OverlayJob(
                source_type="tail_edit", source_tail_edit_id=done_tail_edit, overlays=[]
            )
            s.add(job); s.commit()
            edit = s.get(RunTailEdit, done_tail_edit)
            resolved = _resolve_source(job, s)
            assert resolved == edit.output_path
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Public, token-signed streaming of a tail edit's result (mirrors
# TestPublicResultLink for runs in test_api_v2.py — separate token purpose so
# a run's token can't be replayed against a tail edit and vice versa).
# ---------------------------------------------------------------------------


class TestPublicTailEditResult:
    def test_valid_token_streams_result(self, client, done_run, SessionFactory, tmp_path):
        from app.public import make_tail_edit_result_token

        out = tmp_path / "tail_result.mp4"
        out.write_bytes(b"tailresultbytes")
        s = SessionFactory()
        edit = RunTailEdit(
            run_id=done_run, cut_sec=1.0, status=TailEditStatus.done, output_path=str(out)
        )
        s.add(edit); s.commit()
        eid = edit.id
        s.close()

        tok = make_tail_edit_result_token(eid)
        r = client.get(f"/public/tail-edits/{eid}/result", params={"token": tok})
        assert r.status_code == 200
        assert r.content == b"tailresultbytes"
        assert r.headers["content-type"].startswith("video/mp4")

    def test_bad_token_is_403(self, client, done_tail_edit):
        r = client.get(f"/public/tail-edits/{done_tail_edit}/result", params={"token": "nope"})
        assert r.status_code == 403

    def test_missing_result_file_is_404(self, client, done_run, SessionFactory):
        from app.public import make_tail_edit_result_token

        s = SessionFactory()
        edit = RunTailEdit(run_id=done_run, cut_sec=1.0, status=TailEditStatus.failed)
        s.add(edit); s.commit()
        eid = edit.id
        s.close()

        tok = make_tail_edit_result_token(eid)
        r = client.get(f"/public/tail-edits/{eid}/result", params={"token": tok})
        assert r.status_code == 404

    def test_run_token_cannot_fetch_tail_edit(self, client, done_run, done_tail_edit):
        """A run's result token is scoped to a different purpose string, so it
        must not validate against the tail-edit endpoint."""
        from app.public import make_result_token

        tok = make_result_token(done_run)
        r = client.get(f"/public/tail-edits/{done_tail_edit}/result", params={"token": tok})
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# app/overlay_runs.py — an overlay job's link back to its parent Run, shared by
# the run page's "Overlays" card and the Final Creatives export.
# ---------------------------------------------------------------------------


class TestOverlayRuns:
    def test_parent_run_of_a_run_sourced_job(self, done_run, SessionFactory):
        from app import overlay_runs

        s = SessionFactory()
        try:
            job = OverlayJob(source_type="run", source_run_id=done_run, overlays=[])
            s.add(job); s.commit()
            assert overlay_runs.parent_run(job, s).id == done_run
        finally:
            s.close()

    def test_parent_run_resolves_through_a_tail_edit(
        self, done_run, done_tail_edit, SessionFactory
    ):
        from app import overlay_runs

        s = SessionFactory()
        try:
            job = OverlayJob(
                source_type="tail_edit", source_tail_edit_id=done_tail_edit, overlays=[]
            )
            s.add(job); s.commit()
            assert overlay_runs.parent_run(job, s).id == done_run
        finally:
            s.close()

    def test_parent_run_is_none_for_an_upload(self, SessionFactory):
        from app import overlay_runs

        s = SessionFactory()
        try:
            job = OverlayJob(source_type="upload", overlays=[])
            s.add(job); s.commit()
            assert overlay_runs.parent_run(job, s) is None
        finally:
            s.close()

    def test_jobs_for_run_covers_both_source_kinds(
        self, done_run, done_tail_edit, SessionFactory
    ):
        """A run's overlays are the ones on its result AND on its tail edits."""
        from app import overlay_runs

        s = SessionFactory()
        try:
            direct = OverlayJob(source_type="run", source_run_id=done_run, overlays=[])
            chained = OverlayJob(
                source_type="tail_edit", source_tail_edit_id=done_tail_edit, overlays=[]
            )
            unrelated = OverlayJob(source_type="upload", overlays=[])
            s.add_all([direct, chained, unrelated]); s.commit()

            ids = {j.id for j in overlay_runs.overlay_jobs_for_run(done_run, s)}
            assert direct.id in ids
            assert chained.id in ids
            assert unrelated.id not in ids
        finally:
            s.close()

    def test_jobs_for_run_hides_soft_deleted_but_the_count_keeps_them(
        self, done_run, SessionFactory
    ):
        """Deleting an overlay must not let the next one reuse the bare run id.

        The listing hides a soft-deleted job, but the sibling COUNT still sees
        it — otherwise a run whose first overlay was deleted would export its
        second one under the bare run id and collide with the row the first
        one already wrote to the sheet.
        """
        from datetime import datetime, timezone

        from app import overlay_runs

        s = SessionFactory()
        try:
            gone = OverlayJob(
                source_type="run", source_run_id=done_run, overlays=[],
                deleted_at=datetime.now(timezone.utc),
            )
            live = OverlayJob(source_type="run", source_run_id=done_run, overlays=[])
            s.add_all([gone, live]); s.commit()

            listed = {j.id for j in overlay_runs.overlay_jobs_for_run(done_run, s)}
            assert listed == {live.id}
            assert overlay_runs.count_overlay_jobs_for_run(done_run, s) == 2
        finally:
            s.close()

    def test_sheet_job_id_is_the_runs_own_id_for_a_lone_overlay(
        self, done_run, SessionFactory
    ):
        from app import overlay_runs

        s = SessionFactory()
        try:
            run = s.get(Run, done_run)
            run.language = "ja"
            job = OverlayJob(source_type="run", source_run_id=done_run, overlays=[])
            s.add(job); s.commit()

            assert (
                overlay_runs.sheet_job_id(job, run, sibling_count=1)
                == run.short_id
                == f"{done_run[:8]}-ja"
            )
        finally:
            s.close()

    def test_sheet_job_id_combines_prefixes_for_several_overlays(
        self, done_run, SessionFactory
    ):
        from app import overlay_runs

        s = SessionFactory()
        try:
            run = s.get(Run, done_run)
            run.language = None
            job = OverlayJob(source_type="run", source_run_id=done_run, overlays=[])
            s.add(job); s.commit()

            assert (
                overlay_runs.sheet_job_id(job, run, sibling_count=3)
                == f"{done_run[:8]}_{job.id[:8]}"
            )
        finally:
            s.close()

    def test_sheet_job_id_falls_back_to_own_prefix_without_a_parent(
        self, SessionFactory
    ):
        """An uploaded source has no run to inherit from — unchanged behaviour."""
        from app import overlay_runs

        s = SessionFactory()
        try:
            job = OverlayJob(source_type="upload", overlays=[])
            s.add(job); s.commit()

            assert overlay_runs.sheet_job_id(job, None, sibling_count=0) == job.id[:8]
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Caption jobs as an overlay source (migration d7c9e1a3f582)
# ---------------------------------------------------------------------------


@pytest.fixture()
def done_caption(SessionFactory, done_run):
    """A done RunCaption (off ``done_run``) with a burned file on disk."""
    from app.models import RunCaption
    from app.state_machine import CaptionStatus

    s = SessionFactory()
    output_path = os.path.join(_data_tmp, "caption_result.mp4")
    with open(output_path, "wb") as fh:
        fh.write(b"\x00" * 64)  # placeholder file; existence is what matters
    caption = RunCaption(
        run_id=done_run, status=CaptionStatus.done, output_path=output_path,
        language="ja",
    )
    s.add(caption)
    s.commit()
    cid = caption.id
    s.close()
    return cid


class TestCaptionSource:
    def test_create_with_caption_source(self, client, done_caption, SessionFactory):
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "caption", "source_caption_id": done_caption},
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]

        s = SessionFactory()
        try:
            job = s.get(OverlayJob, job_id)
            assert job.source_type == "caption"
            assert job.source_caption_id == done_caption
            assert job.source_run_id is None
        finally:
            s.close()

    def test_create_caption_source_requires_id(self, client):
        resp = client.post("/api/v2/overlays", json={"source_type": "caption"})
        assert resp.status_code == 422
        assert "source_caption_id" in resp.text

    def test_create_caption_source_unknown_id_404(self, client):
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "caption", "source_caption_id": "does-not-exist"},
        )
        assert resp.status_code == 404

    def test_create_rejects_caption_that_is_not_done(
        self, client, done_run, SessionFactory
    ):
        from app.models import RunCaption
        from app.state_machine import CaptionStatus

        s = SessionFactory()
        caption = RunCaption(run_id=done_run, status=CaptionStatus.burning)
        s.add(caption); s.commit(); cid = caption.id; s.close()

        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "caption", "source_caption_id": cid},
        )
        assert resp.status_code == 409
        assert "not done" in resp.json()["detail"]

    def test_create_rejects_caption_without_a_file(
        self, client, done_run, SessionFactory
    ):
        from app.models import RunCaption
        from app.state_machine import CaptionStatus

        s = SessionFactory()
        caption = RunCaption(
            run_id=done_run, status=CaptionStatus.done, output_path="/nope/missing.mp4"
        )
        s.add(caption); s.commit(); cid = caption.id; s.close()

        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "caption", "source_caption_id": cid},
        )
        assert resp.status_code == 409
        assert "no output video" in resp.json()["detail"]

    def test_resolve_source_reads_the_captions_output(
        self, done_caption, SessionFactory
    ):
        """The worker burns onto the CAPTIONED file, not the run's raw result."""
        from app.models import RunCaption
        from app.overlay_pipeline import _resolve_source

        s = SessionFactory()
        try:
            job = OverlayJob(
                source_type="caption", source_caption_id=done_caption, overlays=[]
            )
            s.add(job); s.commit()
            caption = s.get(RunCaption, done_caption)
            assert _resolve_source(job, s) == caption.output_path
        finally:
            s.close()

    def test_rerender_preserves_caption_source(self, client, done_caption, spy, SessionFactory):
        resp = client.post(
            "/api/v2/overlays",
            json={"source_type": "caption", "source_caption_id": done_caption},
        )
        job_id = resp.json()["id"]
        asset = client.post(
            f"/api/v2/overlays/{job_id}/assets",
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        ).json()["asset_path"]

        s = SessionFactory()
        row = s.get(OverlayJob, job_id)
        row.status = OverlayStatus.done
        row.result_local_path = "/tmp/first-render-caption.mp4"
        row.overlays = [{"asset_path": asset, "nx": 0.1, "ny": 0.7, "sw": 0.4,
                         "start_sec": 0, "end_sec": 5, "fade_sec": 0}]
        s.commit(); s.close()

        new_id = client.post(
            f"/api/v2/overlays/{job_id}/render",
            json={"overlays": [{"asset_path": asset, "nx": 0.2, "ny": 0.3,
                                "sw": 0.5, "start_sec": 0, "end_sec": 4,
                                "fade_sec": 0}]},
        ).json()["id"]
        assert new_id != job_id

        s = SessionFactory()
        try:
            clone = s.get(OverlayJob, new_id)
            assert clone.source_type == "caption"
            assert clone.source_caption_id == done_caption
            # The fork must keep the run link too, or the clone falls out of
            # the run's history and its sibling count.
            assert clone.parent_run_id == s.get(OverlayJob, job_id).parent_run_id
        finally:
            s.close()

    def test_export_input_link_uses_the_captions_drive_file(
        self, client, done_run, done_caption, monkeypatch, SessionFactory
    ):
        """Input Link points at the video actually burned onto."""
        import app.sheets_client as sc
        import app.tasks as tasks_mod
        from app.models import RunCaption

        s = SessionFactory()
        s.get(RunCaption, done_caption).gdrive_file_id = "caption-drive-xyz"
        job = OverlayJob(
            source_type="caption", source_caption_id=done_caption,
            status=OverlayStatus.done, gdrive_file_id="overlay-drive-abc",
            overlays=[],
        )
        s.add(job); s.commit(); jid = job.id; s.close()

        class _FakeSheets:
            HEADER = ["Date", "Job ID", "Input Link", "Creative Link",
                      "Pipeline Type", "Run on Meta"]

            def __init__(self):
                self.appended = []

            def get_header(self, sid, tab):
                return list(self.HEADER)

            def column_values(self, sid, tab, idx, *, last_n=None):
                return []

            def append_row(self, sid, tab, row):
                self.appended.append(row)

        fake = _FakeSheets()
        monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

        tasks_mod.run_export_overlay_to_sheet(jid)

        assert len(fake.appended) == 1
        row = fake.appended[0]
        assert row[_FakeSheets.HEADER.index("Input Link")] == (
            "https://drive.google.com/file/d/caption-drive-xyz/view"
        )
        # And the Job ID still resolves through the caption to the parent run.
        assert row[_FakeSheets.HEADER.index("Job ID")].startswith(done_run[:8])

    def test_parent_run_survives_deleting_the_caption(
        self, client, done_run, done_caption, SessionFactory
    ):
        """Captions are hard-deleted; the overlay must stay attached to its run.

        Without the stamped parent_run_id the job would drop out of the run's
        history AND its sibling count — which is what keeps a later overlay
        from reusing the bare run id in the sheet.
        """
        from app import overlay_runs
        from app.models import RunCaption

        job_id = client.post(
            "/api/v2/overlays",
            json={"source_type": "caption", "source_caption_id": done_caption},
        ).json()["id"]

        s = SessionFactory()
        try:
            s.delete(s.get(RunCaption, done_caption))
            s.commit()

            job = s.get(OverlayJob, job_id)
            assert overlay_runs.parent_run(job, s).id == done_run
            assert job_id in {j.id for j in overlay_runs.overlay_jobs_for_run(done_run, s)}
            assert overlay_runs.count_overlay_jobs_for_run(done_run, s) >= 1
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Public, token-signed streaming of a caption's burned video — what the overlay
# editor's <video> loads for a caption-sourced job. Own token purpose, so a run
# or tail-edit token cannot be replayed here.
# ---------------------------------------------------------------------------


class TestPublicCaptionResult:
    def test_valid_token_streams_result(self, client, done_run, SessionFactory, tmp_path):
        from app.models import RunCaption
        from app.public import make_caption_result_token
        from app.state_machine import CaptionStatus

        out = tmp_path / "caption_result.mp4"
        out.write_bytes(b"captionresultbytes")
        s = SessionFactory()
        caption = RunCaption(
            run_id=done_run, status=CaptionStatus.done, output_path=str(out)
        )
        s.add(caption); s.commit(); cid = caption.id; s.close()

        r = client.get(
            f"/public/captions/{cid}/result",
            params={"token": make_caption_result_token(cid)},
        )
        assert r.status_code == 200
        assert r.content == b"captionresultbytes"
        assert r.headers["content-type"].startswith("video/mp4")

    def test_bad_token_is_403(self, client, done_caption):
        r = client.get(
            f"/public/captions/{done_caption}/result", params={"token": "nope"}
        )
        assert r.status_code == 403

    def test_missing_result_file_is_404(self, client, done_run, SessionFactory):
        from app.models import RunCaption
        from app.public import make_caption_result_token
        from app.state_machine import CaptionStatus

        s = SessionFactory()
        caption = RunCaption(run_id=done_run, status=CaptionStatus.failed)
        s.add(caption); s.commit(); cid = caption.id; s.close()

        r = client.get(
            f"/public/captions/{cid}/result",
            params={"token": make_caption_result_token(cid)},
        )
        assert r.status_code == 404

    def test_tail_edit_token_cannot_fetch_a_caption(
        self, client, done_caption
    ):
        """Purpose-scoped: a token minted for a tail edit must not validate."""
        from app.public import make_tail_edit_result_token

        r = client.get(
            f"/public/captions/{done_caption}/result",
            params={"token": make_tail_edit_result_token(done_caption)},
        )
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# The dashboard's source picker fragment (/v2/overlays/projects/{pid}/
# sources-fragment) — a project's done runs with their derived creatives
# nested underneath, each row carrying the source_type the create call needs.
# ---------------------------------------------------------------------------


class TestOverlayDashboardSourceCell:
    """The overlay project list shares the v2 dashboard's compact Source cell.

    Its rows are plain dicts, not ORM objects, so the keys the macro reads have
    to be carried explicitly — a missing key renders as "file" for everything.
    """

    def test_upload_project_shows_file_with_the_ref_in_the_title(
        self, client, done_run
    ):
        html = client.get("/v2/overlays/").text
        assert ">file</span>" in html
        assert 'title="x.mp4"' in html

    def test_gdrive_project_shows_gd(self, client, SessionFactory, tmp_path):
        link = "https://drive.google.com/file/d/1OverlayGdSource/view"
        out = tmp_path / "gd_run.mp4"
        out.write_bytes(b"\x00" * 32)
        s = SessionFactory()
        proj = VideoProject(source_type="gdrive", source_ref=link,
                            status=ProjectStatus.ready, name="GD Overlay Proj")
        s.add(proj); s.flush()
        s.add(Run(project_id=proj.id, prompt="p", reference_image_urls=[],
                  resolution="1080p", status=RunStatus.done,
                  result_local_path=str(out)))
        s.commit(); s.close()

        html = client.get("/v2/overlays/").text
        assert ">GD</span>" in html
        assert f'title="{link}"' in html


class TestSourcesFragment:
    def test_lists_run_with_its_derived_creatives(
        self, client, done_run, done_tail_edit, done_caption, SessionFactory
    ):
        s = SessionFactory()
        pid = s.get(Run, done_run).project_id
        s.close()

        html = client.get(f"/v2/overlays/projects/{pid}/sources-fragment").text
        assert f'data-source-type="run" data-source-id="{done_run}"' in html
        assert f'data-source-type="tail_edit" data-source-id="{done_tail_edit}"' in html
        assert f'data-source-type="caption" data-source-id="{done_caption}"' in html
        # Both derived creatives counted on the run's own row.
        assert "2 derived" in html

    def test_excludes_derived_creatives_that_are_not_done(
        self, client, done_run, SessionFactory
    ):
        from app.models import RunCaption
        from app.state_machine import CaptionStatus

        s = SessionFactory()
        pid = s.get(Run, done_run).project_id
        in_flight = RunCaption(run_id=done_run, status=CaptionStatus.burning)
        s.add(in_flight); s.commit(); cid = in_flight.id; s.close()

        html = client.get(f"/v2/overlays/projects/{pid}/sources-fragment").text
        assert f'data-source-id="{cid}"' not in html

    def test_caption_of_an_unfinished_run_lands_in_the_orphan_group(
        self, client, SessionFactory, tmp_path
    ):
        """A run that isn't done is not offered, but its caption still is —
        grouped separately so it doesn't look like it belongs to a run above."""
        from app.models import RunCaption
        from app.state_machine import CaptionStatus

        out = tmp_path / "orphan_caption.mp4"
        out.write_bytes(b"\x00" * 32)
        s = SessionFactory()
        proj = VideoProject(source_type="upload", source_ref="o.mp4",
                            status=ProjectStatus.ready, name="Orphans")
        s.add(proj); s.flush()
        run = Run(project_id=proj.id, prompt="p", reference_image_urls=[],
                  resolution="1080p", status=RunStatus.processing)
        s.add(run); s.flush()
        cap = RunCaption(run_id=run.id, status=CaptionStatus.done,
                         output_path=str(out))
        s.add(cap); s.commit()
        pid, rid, cid = proj.id, run.id, cap.id
        s.close()

        html = client.get(f"/v2/overlays/projects/{pid}/sources-fragment").text
        assert "From runs not shown above" in html
        assert f'data-source-type="caption" data-source-id="{cid}"' in html
        assert f'data-source-id="{rid}"' not in html


# ---------------------------------------------------------------------------
# AICP-160: overlay asset_path containment
# ---------------------------------------------------------------------------


class TestOverlayAssetContainment:
    def _job(self, client, done_run):
        r = client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": done_run})
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def _put(self, client, job_id, asset_path):
        return client.put(
            f"/api/v2/overlays/{job_id}/overlays",
            json={"overlays": [{
                "asset_path": asset_path, "nx": 0.1, "ny": 0.7,
                "sw": 0.4, "start_sec": 1.0, "end_sec": 5.0, "fade_sec": 0.3,
            }]},
        )

    def test_legit_asset_still_accepted(self, client, done_run):
        job_id = self._job(client, done_run)
        r = client.post(
            f"/api/v2/overlays/{job_id}/assets",
            files={"file": ("card.png", io.BytesIO(_png_bytes()), "image/png")},
        )
        assert self._put(client, job_id, r.json()["asset_path"]).status_code == 200

    @pytest.mark.parametrize("evil", ["/etc/passwd", "../../../../etc/passwd", "assets/../../../etc/passwd"])
    def test_absolute_and_traversal_rejected_on_put(self, client, done_run, evil):
        job_id = self._job(client, done_run)
        r = self._put(client, job_id, evil)
        assert r.status_code == 400, r.text
        assert "asset not found" in r.text

    def test_symlink_out_of_job_dir_rejected(self, client, done_run):
        from app.storage import overlay_assets_dir

        job_id = self._job(client, done_run)
        target = os.path.realpath(os.path.join(os.path.dirname(overlay_assets_dir(job_id)), "..", "..", "leak.png"))
        with open(target, "wb") as fh:
            fh.write(_png_bytes())
        link = os.path.join(overlay_assets_dir(job_id), "evil.png")
        os.symlink(target, link)
        try:
            assert self._put(client, job_id, "assets/evil.png").status_code == 400
            body = {"overlays": [{
                "asset_path": "assets/evil.png", "nx": 0.1, "ny": 0.1,
                "start_sec": 0, "end_sec": 2,
            }]}
            assert client.post(f"/api/v2/overlays/{job_id}/render", json=body).status_code == 400
        finally:
            os.unlink(target)

    def test_absolute_rejected_on_render(self, client, done_run):
        job_id = self._job(client, done_run)
        body = {"overlays": [{
            "asset_path": "/etc/passwd", "nx": 0.1, "ny": 0.1, "start_sec": 0, "end_sec": 2,
        }]}
        assert client.post(f"/api/v2/overlays/{job_id}/render", json=body).status_code == 400

    def test_pipeline_build_specs_refuses_stored_absolute_path(self):
        from types import SimpleNamespace

        from app.media import MediaError
        from app.overlay_pipeline import _build_specs

        job = SimpleNamespace(id="aicp160-pipe", overlays=[{
            "asset_path": "/etc/passwd", "nx": 0.1, "ny": 0.1, "start_sec": 0, "end_sec": 2,
        }])
        with pytest.raises(MediaError, match="asset not found"):
            _build_specs(job, 10.0)
        job.overlays[0]["asset_path"] = "../../../etc/passwd"
        with pytest.raises(MediaError, match="asset not found"):
            _build_specs(job, 10.0)

    def test_pipeline_build_specs_accepts_contained_asset(self):
        from types import SimpleNamespace

        from app.overlay_pipeline import _build_specs
        from app.storage import overlay_assets_dir

        job = SimpleNamespace(id="aicp160-ok", overlays=[{
            "asset_path": "assets/card_01.png", "nx": 0.1, "ny": 0.1, "start_sec": 0, "end_sec": 2,
        }])
        dest = os.path.join(overlay_assets_dir(job.id), "card_01.png")
        with open(dest, "wb") as fh:
            fh.write(_png_bytes())
        specs = _build_specs(job, 10.0)
        assert specs[0].asset_path == os.path.realpath(dest)
