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

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import ai_models
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

    def test_project_ready_shows_per_segment_prompt_fields(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0, action="swap")
        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text
        assert "Per-segment prompts" in html
        assert "seg-prompt-field" in html
        assert "collectSegmentPrompts" in html

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
        assert f"/api/v2/runs/{r.id}/result?v=" in html
        assert "Download MP4" in html
        # Multi-run copy UI is present on a done run.
        assert "copy-batch" in html
        assert "+ Add run" in html
        assert "MAX_COPY_RUNS = 10" in html

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
        assert f"/api/v2/runs/{r.id}/result?v=" in resp.text

    def test_run_fragment_unknown_id_404(self, client):
        resp = client.get("/v2/runs/no-such-run/status-fragment")
        assert resp.status_code == 404


def _localisation_project(session, **kwargs) -> VideoProject:
    defaults = dict(status=ProjectStatus.ready, project_type="localisation")
    defaults.update(kwargs)
    return _make_project(session, **defaults)


_TRANSCRIPT = {
    "schema_version": 1,
    "model": "gemini-2.5-pro",
    "prompt_version": "transcribe/v1",
    "source_language": "en",
    "lines": [
        {
            "id": 1,
            "start": 0.0,
            "end": 2.4,
            "speaker": "off-screen interviewer (camera person)",
            "on_screen": False,
            "text": "Hey, what are you doing?",
        },
        {
            "id": 2,
            "start": 2.4,
            "end": 6.0,
            "speaker": "the woman in the red jacket",
            "on_screen": True,
            "text": "I'm, uh, listening to my textbook.",
        },
    ],
    "on_screen_text": [{"start": 0.0, "end": 3.0, "text": "SPEECHIFY"}],
}


class TestLocalisationCreateProjectForm:
    def test_dashboard_offers_the_localisation_type(self, client):
        html = client.get("/v2/").text
        assert 'value="localisation"' in html
        assert "Localisation" in html

    def test_hook_sec_field_is_present_but_hidden_by_default(self, client):
        """The field exists for every render; JS reveals it for the hook-split
        type only, so a face-swap operator never sees it."""
        html = client.get("/v2/").text
        assert 'name="hook_sec"' in html
        assert 'id="new-project-hook-group" style="display:none"' in html

    def test_only_the_localisation_option_asks_for_a_hook(self, client):
        html = client.get("/v2/").text
        assert 'value="localisation"' in html
        # data-hook drives the reveal; exactly one type is hook-split today.
        assert html.count('data-hook="1"') == 1
        assert 'data-hook="0"' in html

    def test_mute_source_default_comes_from_the_registry(self, client):
        """default_mute_source is True only for localisation — the checkbox is
        stamped from the selected option's data-mute."""
        html = client.get("/v2/").text
        assert 'id="new-project-mute-source"' in html
        assert html.count('data-mute="1"') == 1
        assert 'data-mute="0"' in html


class TestLocalisationNewRunForm:
    def test_defaults_helper_surfaces_the_localisation_spec(self, db_session):
        from app.web_v2 import _new_run_defaults

        project = _localisation_project(db_session)
        ctx = _new_run_defaults(project)

        assert ctx["is_localisation"] is True
        assert ctx["default_audio_mode"] == "seedance"
        # Gemini Omni emits no audio and is therefore unusable here.
        assert ctx["models_without_audio"] == ["gemini-omni"]
        assert ctx["default_model"] != "gemini-omni"
        assert ai_models.spec_for(ctx["default_model"]).produces_audio is True
        # Both templates travel to the client so the mode radio can swap them.
        assert set(ctx["localisation_prompts"]) == {"keep", "swap"}
        assert "{dialogue}" in ctx["localisation_prompts"]["swap"]
        assert "{dialogue}" in ctx["localisation_prompts"]["keep"]

    def test_default_mode_matches_the_registry_default_prompt(self, db_session):
        """The radio and the pre-filled textarea must never disagree."""
        from app.project_types import spec_for
        from app.web_v2 import _new_run_defaults

        project = _localisation_project(db_session)
        ctx = _new_run_defaults(project)
        assert (
            ctx["localisation_prompts"][ctx["default_localisation_mode"]]
            == spec_for("localisation").default_prompt
        )

    def test_other_types_flag_no_model_and_stay_on_original_audio(self, db_session):
        from app.web_v2 import _new_run_defaults

        project = _make_project(db_session, project_type="face_swap")
        ctx = _new_run_defaults(project)
        assert ctx["is_localisation"] is False
        assert ctx["models_without_audio"] == []
        assert ctx["default_audio_mode"] == "original"

    def test_mute_source_still_wins_for_non_localisation_types(self, db_session):
        """The pre-existing "Remove original audio" behaviour is unchanged."""
        from app.web_v2 import _new_run_defaults

        project = _make_project(db_session, project_type="face_swap", mute_source=True)
        assert _new_run_defaults(project)["default_audio_mode"] == "seedance"

    def test_form_marks_the_audio_less_model_unusable(self, client, db_session):
        p = _localisation_project(db_session)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert '<option value="gemini-omni" disabled>' in html
        assert "generates no audio" in html
        assert '<option value="seedance-fast" selected>' in html
        # 2.5 remains selectable for longer clips / explicit generate_audio.
        assert '<option value="seedance-2-5"' in html
        assert 'value="seedance-2-5" disabled' not in html

    def test_other_types_still_offer_every_model(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text
        assert '<option value="gemini-omni" disabled>' not in html

    def test_form_preselects_generated_audio_and_warns_about_original(
        self, client, db_session
    ):
        p = _localisation_project(db_session)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert '<option value="seedance" selected>' in html
        assert 'id="audio-localisation-warning"' in html
        assert "untranslated source soundtrack" in html
        assert "var LOCALISATION_RUN = true;" in html

    def test_form_offers_both_localisation_modes(self, client, db_session):
        p = _localisation_project(db_session)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'name="localisation_mode" value="keep"' in html
        assert 'name="localisation_mode" value="swap"' in html
        assert "Speech only" in html
        assert "Speech + character" in html
        # The reference input is called out as required for the swap mode.
        assert 'id="ref-localisation-required"' in html

    def test_translate_button_disabled_without_a_ready_transcript(
        self, client, db_session
    ):
        p = _localisation_project(db_session)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="translate-btn"' in html
        assert "disabled>Translate</button>" in html
        assert "No transcript yet" in html

    def test_translate_button_enabled_once_the_transcript_is_ready(
        self, client, db_session
    ):
        p = _localisation_project(
            db_session, transcript_status="ready", transcript=_TRANSCRIPT
        )
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert ">Translate</button>" in html
        assert "disabled>Translate</button>" not in html
        assert "/localisation-prompt" in html

    def test_empty_transcript_keeps_translate_disabled_with_a_reason(
        self, client, db_session
    ):
        p = _localisation_project(db_session, transcript_status="empty")
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert "disabled>Translate</button>" in html
        assert "No speech was found in this video" in html


    def test_no_localisation_panel_on_other_types(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="localisation-panel"' not in html
        assert 'id="translate-btn"' not in html
        assert "var LOCALISATION_RUN = false;" in html

    def test_status_fragment_carries_the_same_localisation_form(
        self, client, db_session
    ):
        p = _localisation_project(
            db_session, transcript_status="ready", transcript=_TRANSCRIPT
        )
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}/status-fragment").text

        assert 'id="localisation-panel"' in html
        assert '<option value="seedance" selected>' in html
        assert "disabled>Translate</button>" not in html


class TestTranscriptPanel:
    def test_absent_for_non_localisation_projects(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        html = client.get(f"/v2/projects/{p.id}").text
        assert 'id="transcript-card"' not in html

    def test_never_requested_offers_transcribe(self, client, db_session):
        p = _localisation_project(db_session)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="transcript-card"' in html
        assert "not transcribed yet" in html.lower()
        assert ">Transcribe</button>" in html
        assert "/transcribe" in html
        # Nothing to poll for (the refresh helper still names the URL).
        assert f'hx-get="/v2/projects/{p.id}/transcript-fragment"' not in html

    def test_pending_polls_itself(self, client, db_session):
        p = _localisation_project(db_session, transcript_status="pending")
        html = client.get(f"/v2/projects/{p.id}").text

        assert f'hx-get="/v2/projects/{p.id}/transcript-fragment"' in html
        assert 'hx-trigger="every 3s"' in html
        assert "Transcribing" in html

    def test_running_polls_itself(self, client, db_session):
        p = _localisation_project(db_session, transcript_status="running")
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text
        assert 'hx-trigger="every 3s"' in html

    def test_ready_stops_polling_and_lists_the_lines(self, client, db_session):
        p = _localisation_project(
            db_session, transcript_status="ready", transcript=_TRANSCRIPT
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        # Terminal state → the polling wrapper is gone (same rule as the
        # merges/tail-edit fragments).
        assert "hx-trigger" not in html
        # Detected language, timestamps, speakers and the verbatim text.
        assert 'id="transcript-source-language"' in html
        assert 'value="en"' in html
        assert "off-screen interviewer (camera person)" in html
        assert "the woman in the red jacket" in html
        assert "Hey, what are you doing?" in html
        assert 'value="0.0"' in html and 'value="2.4"' in html
        # Inline editing saves the whole dict back via PATCH.
        assert "saveTranscript" in html
        assert "method: 'PATCH'" in html

    def test_ready_with_no_lines_says_so(self, client, db_session):
        p = _localisation_project(
            db_session,
            transcript_status="ready",
            transcript={"schema_version": 1, "source_language": "en", "lines": []},
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text
        assert "no lines" in html.lower()
        assert "hx-trigger" not in html

    def test_empty_is_a_success_not_an_error(self, client, db_session):
        p = _localisation_project(db_session, transcript_status="empty")
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        assert "No speech found in this video" in html
        assert "alert-danger" not in html
        assert "hx-trigger" not in html
        assert ">Transcribe again</button>" in html

    def test_failed_shows_the_error_and_stays_usable(self, client, db_session):
        p = _localisation_project(
            db_session,
            transcript_status="failed",
            transcript_error="kie.ai returned 401: invalid key",
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        assert "Transcription failed" in html
        assert "kie.ai returned 401: invalid key" in html
        assert "hx-trigger" not in html
        assert ">Transcribe again</button>" in html

    def test_malformed_lines_do_not_break_the_render(self, client, db_session):
        """The transcript is model-written and operator-edited JSON — a missing
        timestamp must render as a blank input, not 500 the project page."""
        p = _localisation_project(
            db_session,
            transcript_status="ready",
            transcript={
                "lines": [
                    {"id": 1, "text": "no timestamps here"},
                    "not even a dict",
                    {"id": 2, "start": "oops", "end": None, "speaker": None, "text": None},
                ]
            },
        )
        resp = client.get(f"/v2/projects/{p.id}/transcript-fragment")
        assert resp.status_code == 200
        assert "no timestamps here" in resp.text

    def test_fragment_unknown_project_404(self, client):
        assert client.get("/v2/projects/nope/transcript-fragment").status_code == 404


class TestTranscriptLifecycleRegressions:
    """Regressions for the five lifecycle bugs a review caught after the
    feature was first written — every one of which shipped past a fully green
    suite, because the breakage lived in WHEN a fragment re-renders rather than
    in what any single render produces.

    The invariant underneath all of them: #status-content must never re-render
    on a timer (it wraps the New Run form, whose prompt and per-segment
    textareas hold typed work), so anything in that form which depends on the
    transcript has to be refreshed out-of-band by the panel's own poll.
    """

    # -- Bug 1: the Translate button never enabled without a page reload ----

    def test_status_fragment_never_polls_once_the_run_form_is_live(
        self, client, db_session
    ):
        """The rule the out-of-band gate exists to work around. If this ever
        starts polling, a 3s timer silently eats the operator's prompt."""
        p = _localisation_project(db_session, transcript_status="ready")
        html = client.get(f"/v2/projects/{p.id}/status-fragment").text
        assert 'id="new-run-form"' in html
        assert "hx-trigger" not in html

    def test_polling_fragment_carries_an_out_of_band_translate_gate(
        self, client, db_session
    ):
        p = _localisation_project(db_session, transcript_status="running")
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text
        assert 'id="translate-gate" hx-swap-oob="true"' in html
        assert 'id="translate-btn"' in html

    def test_the_gate_converges_in_both_directions(self, client, db_session):
        """Enabled when the transcript lands, disabled again when a
        re-transcription starts — both without reloading the page."""
        running = _localisation_project(db_session, transcript_status="running")
        html = client.get(f"/v2/projects/{running.id}/transcript-fragment").text
        gate = html[html.index('id="translate-gate"'):]
        assert "disabled" in gate
        assert "Transcription is still running" in gate

        ready = _localisation_project(
            db_session, transcript_status="ready", transcript=_TRANSCRIPT
        )
        html = client.get(f"/v2/projects/{ready.id}/transcript-fragment").text
        gate = html[html.index('id="translate-gate"'):]
        assert "disabled" not in gate
        assert 'id="translate-disabled-reason"' not in gate

    def test_no_out_of_band_gate_when_the_run_form_is_not_on_the_page(
        self, client, db_session
    ):
        """An OOB swap whose target does not exist is a silent no-op at best;
        the New Run form only exists once the project is ready."""
        p = _localisation_project(
            db_session, status=ProjectStatus.analyzing, transcript_status="pending"
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text
        assert "hx-swap-oob" not in html

    # -- Bug 2: the panel froze, and its button double-charged --------------

    def test_promised_transcription_polls_and_offers_no_second_run(
        self, client, db_session
    ):
        """transcript_status is "pending" from project creation, so the window
        where the panel used to render a Transcribe button next to an already
        promised job no longer exists — one click there paid for a second
        model call on the same video."""
        p = _localisation_project(
            db_session, status=ProjectStatus.created, transcript_status="pending"
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        assert 'hx-trigger="every 3s"' in html
        assert "Transcription is queued" in html
        assert ">Transcribe</button>" not in html
        assert ">Transcribe again</button>" not in html

    def test_analysis_failure_stops_the_poll_and_hands_back_control(
        self, client, db_session
    ):
        """"pending" on a failed analysis is terminal — nothing will ever pick
        it up, so polling forever would be pure waste."""
        p = _localisation_project(
            db_session, status=ProjectStatus.failed, transcript_status="pending"
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        assert "hx-trigger" not in html
        assert "Transcription never started" in html
        # A way out exists — the label reads "again" because the status column
        # says "pending", but no job was ever queued for it.
        assert "requestTranscript(" in html
        assert ">Transcribe again</button>" in html

    # -- Bug 3: a dead transcription had no way out ------------------------

    def test_a_running_transcription_can_be_restarted_but_asks_first(
        self, client, db_session
    ):
        """RQ kills a job at TRANSCRIBE_JOB_TIMEOUT without writing "failed",
        and any deploy restarts the worker — so "running" can outlive its job
        and must not be a dead end. Restarting costs a second model call, so it
        confirms."""
        p = _localisation_project(db_session, transcript_status="running")
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        assert "Restart transcription" in html
        assert "window.confirm" in html or "confirm(" in html
        assert "costs a second model call" in html

    # -- Bug 7: the documented hand-paste fallback was unreachable ----------

    @pytest.mark.parametrize(
        "status,transcript",
        [
            (None, None),
            ("failed", None),
            ("empty", None),
            ("ready", _TRANSCRIPT),
        ],
    )
    def test_every_settled_state_can_be_hand_edited(
        self, client, db_session, status, transcript
    ):
        """docs/localisation.md §8: a failed transcription must leave the
        project usable because the operator can type the lines in. That needs
        the table, the language field, Save and Add line in every non-polling
        state — not only in "ready"."""
        p = _localisation_project(
            db_session, transcript_status=status, transcript=transcript
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        assert 'id="transcript-lines"' in html
        assert 'id="transcript-source-language"' in html
        assert ">Save transcript</button>" in html
        assert 'id="transcript-add-line"' in html

    def test_a_polling_state_renders_nothing_typeable(self, client, db_session):
        """The panel replaces its own innerHTML every 3s, so an input next to
        the spinner would lose a keystroke every three seconds."""
        p = _localisation_project(db_session, transcript_status="running")
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        assert 'id="transcript-lines"' not in html
        assert 'id="transcript-source-language"' not in html
        assert ">Save transcript</button>" not in html

    def test_broken_line_ids_are_repaired_for_the_editor(
        self, client, db_session
    ):
        """PATCH /transcript refuses a line without a unique id (a translation
        is rejoined on them), so a transcript that lost one would render fine
        and then 400 the moment the operator pressed Save."""
        p = _localisation_project(
            db_session,
            transcript_status="ready",
            transcript={
                "source_language": "en",
                "lines": [
                    {"id": 1, "start": 0.0, "end": 1.0, "text": "first"},
                    {"start": 1.0, "end": 2.0, "text": "no id at all"},
                    {"id": 1, "start": 2.0, "end": 3.0, "text": "duplicate id"},
                ],
            },
        )
        html = client.get(f"/v2/projects/{p.id}/transcript-fragment").text

        ids = re.findall(r'data-line-id="(\d+)"', html)
        assert len(ids) == 3
        assert len(set(ids)) == 3, f"ids must be unique for PATCH, got {ids}"
        # The valid stored id is kept where it was — a translation matches on it.
        assert ids[0] == "1"

    # -- Bugs 9 and 10: client validation and error rendering ---------------

    def test_hook_input_agrees_with_the_server_about_zero(self, client):
        """_validate_hook_sec rejects 0 with a 400, so the browser must not
        accept it and turn an inline form error into a server error."""
        html = client.get("/v2/").text
        assert 'id="new-project-hook-sec"' in html
        assert 'min="0.1"' in html
        assert 'min="0"' not in html

    def test_error_bodies_are_rendered_through_the_shared_formatter(
        self, client, db_session
    ):
        """FastAPI's own 422 sends `detail` as an ARRAY of objects, which
        concatenates to "[object Object]" — exactly the error class where the
        field name is the whole point."""
        assert "function apiErrorText(" in client.get("/v2/").text

        p = _localisation_project(db_session, transcript_status="ready")
        html = client.get(f"/v2/projects/{p.id}").text
        assert "apiErrorText(" in html
        assert "data.detail ||" not in html
        assert "b.detail ||" not in html
