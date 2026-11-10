"""
Tests for the burned-captions v2 web UI additions (app/web_v2.py §3.6) and
the chaining-derived-creatives additions (§3.7.4):

- GET /v2/runs/{rid}/captions-fragment
- The "Captions" card in run_status_content.html / run_captions_list.html
- The "Add Captions" button in the existing .action-row
- Regression: the two pre-existing export buttons stay byte-for-byte present
- The captions control's mirror-image source select (a done tail edit, or
  the run result) and each row's resolved source label, deleted-safe.

Follows the conventions of test_web_v2.py (same fixtures, same TestClient
strategy, same style of asserting on rendered HTML strings).
"""

from __future__ import annotations

import os
import sys
import tempfile

# Must set env vars before importing app modules
_db_fd, _db_path = tempfile.mkstemp(suffix="_captions_web.db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
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


def _make_caption(session, run_id: str, **kwargs) -> RunCaption:
    defaults = dict(run_id=run_id, status=CaptionStatus.queued)
    defaults.update(kwargs)
    c = RunCaption(**defaults)
    session.add(c)
    session.commit()
    return c


def _make_tail_edit(session, run_id: str, **kwargs) -> RunTailEdit:
    defaults = dict(run_id=run_id, cut_sec=5.0, status=TailEditStatus.queued)
    defaults.update(kwargs)
    edit = RunTailEdit(**defaults)
    session.add(edit)
    session.commit()
    return edit


# ---------------------------------------------------------------------------
# Captions fragment — GET /v2/runs/{rid}/captions-fragment
# ---------------------------------------------------------------------------


class TestCaptionsFragment:
    def test_captions_fragment_renders_list(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(
            db_session, r.id, language="ja", status=CaptionStatus.done,
            gdrive_file_id="GDCAP1",
        )

        resp = client.get(f"/v2/runs/{r.id}/captions-fragment")
        assert resp.status_code == 200
        html = resp.text
        assert "ja" in html
        assert "badge-done" in html

    def test_captions_fragment_polls_while_active(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(db_session, r.id, status=CaptionStatus.transcribing)

        resp = client.get(f"/v2/runs/{r.id}/captions-fragment")
        assert resp.status_code == 200
        assert "captions-fragment" in resp.text
        assert "hx-trigger" in resp.text

    def test_captions_fragment_stops_polling_when_terminal(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(
            db_session, r.id, status=CaptionStatus.failed, error_message="boom"
        )

        resp = client.get(f"/v2/runs/{r.id}/captions-fragment")
        assert resp.status_code == 200
        assert "hx-trigger" not in resp.text

    def test_captions_fragment_unknown_run_404(self, client):
        resp = client.get("/v2/runs/no-such-run/captions-fragment")
        assert resp.status_code == 404

    def test_captions_fragment_empty_when_no_jobs(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)

        resp = client.get(f"/v2/runs/{r.id}/captions-fragment")
        assert resp.status_code == 200
        # Card is gated on `{% if captions %}` — no jobs means no card at all.
        assert "captions-card" not in resp.text


# ---------------------------------------------------------------------------
# Row rendering — done / failed states
# ---------------------------------------------------------------------------


class TestCaptionsRowRendering:
    def test_done_caption_shows_download_drive_and_creatives(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        c = _make_caption(
            db_session, r.id, status=CaptionStatus.done, gdrive_file_id="GDCAP2",
        )

        html = client.get(f"/v2/runs/{r.id}/captions-fragment").text
        assert f"/api/v2/captions/{c.id}/result" in html
        assert "Download" in html
        assert f"https://drive.google.com/file/d/{c.gdrive_file_id}/view" in html
        assert "Drive" in html
        assert "captionExportToSheet(this)" in html
        assert "Creatives" in html

    def test_done_caption_already_exported_shows_checkmark(self, client, db_session):
        from datetime import datetime, timezone

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(
            db_session, r.id, status=CaptionStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )

        html = client.get(f"/v2/runs/{r.id}/captions-fragment").text
        assert "Already logged to the Final Creatives sheet" in html
        assert "captionExportToSheet(this)" not in html

    def test_failed_caption_shows_error_and_retry(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        c = _make_caption(
            db_session, r.id, status=CaptionStatus.failed,
            error_message="Transcription failed: timeout",
        )

        html = client.get(f"/v2/runs/{r.id}/captions-fragment").text
        assert "badge-failed" in html
        assert "Transcription failed: timeout" in html
        assert f"retryCaption('{c.id}'" in html
        assert f"deleteCaption('{c.id}'" in html


# ---------------------------------------------------------------------------
# Run-detail page — the "Add Captions" button + card gating
# ---------------------------------------------------------------------------


class TestAddCaptionsButton:
    def test_done_run_shows_add_captions_button(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            result_gdrive_file_id="GD123",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert 'onclick="addCaptions(this)"' in html
        assert "Add Captions" in html
        # The POST target is built via JS string concat, not a literal URL.
        assert "function addCaptions(btn)" in html
        assert "'/captions'" in html

    def test_non_done_run_omits_captions_card(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        html = client.get(f"/v2/runs/{r.id}").text
        assert "captions-card" not in html
        # The button itself lives inside the done-only branch of the template.
        assert 'onclick="addCaptions(this)"' not in html

    def test_done_run_page_includes_captions_card_when_jobs_exist(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(db_session, r.id, language="es", status=CaptionStatus.burning)

        html = client.get(f"/v2/runs/{r.id}").text
        assert "captions-card" in html
        assert "es" in html
        assert "badge-burning" in html

    def test_status_fragment_carries_captions_context(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(db_session, r.id, status=CaptionStatus.delivering)

        html = client.get(f"/v2/runs/{r.id}/status-fragment").text
        assert "captions-card" in html
        assert "badge-delivering" in html


# ---------------------------------------------------------------------------
# Chaining derived creatives — caption sourced from a tail edit
# (docs/burned-captions.md §3.7.4)
# ---------------------------------------------------------------------------


class TestCaptionSourcePicker:
    def test_source_select_defaults_to_run_result_with_no_tail_edits(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done, result_gdrive_file_id="GD1",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert 'id="captions-source-tail-edit"' in html
        assert '<option value="">Run result</option>' in html

    def test_source_select_offers_one_option_per_done_tail_edit(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done, result_gdrive_file_id="GD1",
        )
        e1 = _make_tail_edit(db_session, r.id, cut_sec=3.0, status=TailEditStatus.done)
        e2 = _make_tail_edit(db_session, r.id, cut_sec=9.5, status=TailEditStatus.done)

        html = client.get(f"/v2/runs/{r.id}").text
        assert f'<option value="{e1.id}">' in html
        assert f'<option value="{e2.id}">' in html
        assert "3.0s" in html
        assert "9.5s" in html

    def test_source_select_excludes_non_done_tail_edits(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done, result_gdrive_file_id="GD1",
        )
        in_flight = _make_tail_edit(db_session, r.id, status=TailEditStatus.processing)

        html = client.get(f"/v2/runs/{r.id}").text
        assert f'<option value="{in_flight.id}">' not in html

    def test_source_select_shows_clip_name_when_tail_edit_appended_one(
        self, client, db_session
    ):
        from app.models import LibraryClip

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done, result_gdrive_file_id="GD1",
        )
        clip = LibraryClip(name="Outro JA", local_path="/tmp/nowhere.mp4")
        db_session.add(clip)
        db_session.commit()
        _make_tail_edit(
            db_session, r.id, cut_sec=4.0, status=TailEditStatus.done,
            library_clip_id=clip.id,
        )

        html = client.get(f"/v2/runs/{r.id}").text
        assert "Outro JA" in html

    def test_add_captions_js_posts_source_tail_edit_id(self, client, db_session):
        # No JS engine on this box — assert on the submit function body
        # rather than executing it (docs/burned-captions.md §3.7.4 test note).
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done, result_gdrive_file_id="GD1",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert "source_tail_edit_id" in html
        assert "captions-source-tail-edit" in html

    def test_add_captions_js_appends_source_field_unconditionally(
        self, client, db_session
    ):
        """Guards the empty-FormData 400 (see test_api_captions.py's
        test_create_with_blank_source_tail_edit_id_is_fine).

        v1 appended the field only when a tail edit was picked, so the default
        "Run result" path built an *empty* FormData. The browser then sent a
        multipart body of nothing but the closing delimiter, python-multipart
        refused it, and every unchained Add Captions click died at 400 "There
        was an error parsing the body" before reaching the endpoint. The append
        must stay unconditional — a blank value already means "the run's own
        result" server-side.
        """
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done, result_gdrive_file_id="GD1",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert (
            "fd.append('source_tail_edit_id', sourceSelect ? sourceSelect.value : '')"
            in html
        )
        assert (
            "if (sourceSelect && sourceSelect.value) "
            "fd.append('source_tail_edit_id'" not in html
        )

    def test_caption_row_default_source_shows_run_result(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(db_session, r.id, status=CaptionStatus.done)

        html = client.get(f"/v2/runs/{r.id}/captions-fragment").text
        assert "Run result" in html

    def test_caption_row_shows_chained_tail_edit_source_label(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, r.id, cut_sec=7.5, status=TailEditStatus.done)
        _make_caption(
            db_session, r.id, status=CaptionStatus.done,
            source_type="tail_edit", source_tail_edit_id=edit.id,
        )

        html = client.get(f"/v2/runs/{r.id}/captions-fragment").text
        assert "Tail edit:" in html
        assert "7.5s" in html
        assert "(deleted)" not in html

    def test_caption_row_shows_deleted_when_tail_edit_source_missing(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_caption(
            db_session, r.id, status=CaptionStatus.done,
            source_type="tail_edit", source_tail_edit_id="no-such-tail-edit-id",
        )

        resp = client.get(f"/v2/runs/{r.id}/captions-fragment")
        assert resp.status_code == 200
        assert "(deleted)" in resp.text

    def test_several_chained_captions_each_resolve_their_own_source(
        self, client, db_session
    ):
        """State-matrix coverage: several jobs at once, each pointing at a
        different (or missing) source, must each resolve independently."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, r.id, cut_sec=2.0, status=TailEditStatus.done)
        _make_caption(db_session, r.id, status=CaptionStatus.done)  # run result
        _make_caption(
            db_session, r.id, status=CaptionStatus.done,
            source_type="tail_edit", source_tail_edit_id=edit.id,
        )
        _make_caption(
            db_session, r.id, status=CaptionStatus.done,
            source_type="tail_edit", source_tail_edit_id="gone",
        )
        _make_caption(db_session, r.id, status=CaptionStatus.transcribing)  # in-flight, run result

        resp = client.get(f"/v2/runs/{r.id}/captions-fragment")
        assert resp.status_code == 200
        html = resp.text
        # Two rows default to the run result (the explicit one + the
        # in-flight one, which also never set source_type); each of the
        # other two resolves its own chained/deleted source independently.
        assert html.count("Run result") == 2
        assert "Tail edit:" in html
        assert "2.0s" in html
        assert "(deleted)" in html
        assert "badge-transcribing" in html

    def test_non_done_run_page_is_query_free_for_chaining_context(
        self, client, db_session, monkeypatch
    ):
        import app.web_v2 as web_v2

        def _boom(*args, **kwargs):
            raise AssertionError("chaining query ran on a non-done run")

        monkeypatch.setattr(web_v2, "_get_done_captions", _boom)
        monkeypatch.setattr(web_v2, "_get_done_tail_edits", _boom)
        monkeypatch.setattr(web_v2, "_referenced_caption_labels", _boom)
        monkeypatch.setattr(web_v2, "_referenced_tail_edit_labels", _boom)

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)

        resp = client.get(f"/v2/runs/{r.id}")
        assert resp.status_code == 200
        assert "captions-card" not in resp.text
        assert "Change ending" not in resp.text


# ---------------------------------------------------------------------------
# Regression: the two pre-existing export buttons are unchanged
# ---------------------------------------------------------------------------


class TestExistingButtonsUnchanged:
    def test_add_to_final_creatives_button_present(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            result_gdrive_file_id="GD123",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert 'onclick="exportToSheet(this)"' in html
        assert ">Add to Final Creatives</button>" in html
        assert "function exportToSheet(btn)" in html

    def test_both_existing_buttons_and_new_button_coexist(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            result_gdrive_file_id="GD123",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert 'onclick="exportToSheet(this)"' in html
        assert 'onclick="addCaptions(this)"' in html
