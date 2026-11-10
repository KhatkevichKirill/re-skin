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
import re
import sys
import tempfile
import types

from datetime import datetime, timezone

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

from app import ai_models, web_v2
from app.db import Base, get_db
from app.models import (
    OverlayJob,
    Run,
    RunCaption,
    RunSegment,
    RunSegmentImageJob,
    RunTailEdit,
    SegmentDef,
    VideoProject,
)
from app.state_machine import (
    CaptionStatus,
    OverlayStatus,
    ProjectStatus,
    RunStatus,
    SegmentImageStatus,
    SegmentStatus,
    TailEditStatus,
)


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


def _make_tail_edit(session, run_id: str, **kwargs) -> RunTailEdit:
    defaults = dict(run_id=run_id, cut_sec=5.0, status=TailEditStatus.queued)
    defaults.update(kwargs)
    edit = RunTailEdit(**defaults)
    session.add(edit)
    session.commit()
    return edit


def _make_caption(session, run_id: str, **kwargs) -> RunCaption:
    defaults = dict(run_id=run_id, status=CaptionStatus.queued)
    defaults.update(kwargs)
    c = RunCaption(**defaults)
    session.add(c)
    session.commit()
    return c


def _make_overlay_job(session, **kwargs) -> OverlayJob:
    """An OverlayJob as the API would create it.

    `parent_run_id` is resolved through the same helper the create endpoint
    uses rather than being left NULL, so a factory row behaves like a real one
    — notably, it stays attached to its run after the tail edit or caption it
    was burned from is deleted.
    """
    from app import overlay_runs

    defaults = dict(source_type="run", status=OverlayStatus.created, overlays=[])
    defaults.update(kwargs)
    defaults.setdefault(
        "parent_run_id",
        overlay_runs.resolve_parent_run_id(
            source_type=defaults["source_type"],
            source_run_id=defaults.get("source_run_id"),
            source_tail_edit_id=defaults.get("source_tail_edit_id"),
            source_caption_id=defaults.get("source_caption_id"),
            db=session,
        ),
    )
    job = OverlayJob(**defaults)
    session.add(job)
    session.commit()
    return job


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


class TestDashboardSourceCell:
    """The Source column shows the KIND of source, full ref in the hover title."""

    _GD_LINK = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUv/view"

    def test_gdrive_project_shows_gd(self, client, db_session):
        _make_project(db_session, source_type="gdrive", source_ref=self._GD_LINK)
        html = client.get("/v2/").text
        assert ">GD</span>" in html
        # The pasted link is hover-only, never printed into the cell.
        assert f'title="{self._GD_LINK}"' in html

    def test_upload_project_shows_file(self, client, db_session):
        _make_project(db_session, source_type="upload", source_ref="my_source.mp4")
        html = client.get("/v2/").text
        assert ">file</span>" in html
        assert 'title="my_source.mp4"' in html

    def test_project_from_a_run_shows_the_run_short_id(self, client, db_session):
        """A promoted run is stored as an upload — the run id must still win."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, p.id, name="Origin Run", status=RunStatus.done)
        _make_project(
            db_session,
            source_type="upload",
            source_ref=f"Origin Run ({run.id[:8]}) — final video",
            source_run_id=run.id,
        )
        html = client.get("/v2/").text
        assert f">{run.id[:8]}</code>" in html
        assert f'title="Origin Run ({run.id[:8]}) — final video"' in html


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

    def test_fragment_filters_to_matching_run_id(self, client, db_session):
        # With a run_id filter, the pivot lists only the matching run — not
        # every run in the project.
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, name="Wanted")
        other = _make_run(db_session, project.id, name="Other")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?run_id={wanted.id[:8]}"
        )
        assert resp.status_code == 200
        html = resp.text
        assert wanted.id in html
        assert other.id not in html

    def test_fragment_without_run_id_lists_all_runs(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        r1 = _make_run(db_session, project.id, name="One")
        r2 = _make_run(db_session, project.id, name="Two")

        resp = client.get(f"/v2/projects/{project.id}/runs-fragment")
        assert r1.id in resp.text
        assert r2.id in resp.text

    def test_fragment_shows_the_language_suffixed_short_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, name="Redhead", language="ja")

        resp = client.get(f"/v2/projects/{project.id}/runs-fragment")
        assert f"{run.id[:8]}-ja" in resp.text

    def test_fragment_honours_the_model_filter(self, client, db_session):
        """The pivot takes the dashboard's whole filter set, not just run_id."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, model="seedance-2-5")
        other = _make_run(db_session, project.id, model="gemini-omni")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?model=seedance-2-5"
        )
        assert wanted.id in resp.text
        assert other.id not in resp.text

    def test_fragment_honours_the_sent_filter(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        exported = _make_run(
            db_session, project.id,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        outstanding = _make_run(db_session, project.id)

        resp = client.get(f"/v2/projects/{project.id}/runs-fragment?sent=no")
        assert outstanding.id in resp.text
        assert exported.id not in resp.text

    def test_fragment_language_filter_treats_null_as_english(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        unset = _make_run(db_session, project.id, language=None)
        japanese = _make_run(db_session, project.id, language="ja")

        resp = client.get(f"/v2/projects/{project.id}/runs-fragment?lang=en")
        assert unset.id in resp.text
        assert japanese.id not in resp.text

    def test_fragment_filter_accepts_the_suffixed_id(self, client, db_session):
        """The operator can paste the id they see ('1736426f-ja') into the filter."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, name="Wanted", language="ja")
        other = _make_run(db_session, project.id, name="Other")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?run_id={wanted.id[:8]}-ja"
        )
        assert resp.status_code == 200
        assert wanted.id in resp.text
        assert other.id not in resp.text


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

    def test_new_run_form_preselects_default_resolution(self, client, db_session, monkeypatch):
        """AICP-391: the Resolution dropdown preselects 720p by default."""
        from app.config import settings
        monkeypatch.setattr(settings, "DEFAULT_RESOLUTION", "720p")
        p = _make_project(db_session, status=ProjectStatus.ready)
        html = client.get(f"/v2/projects/{p.id}").text
        assert 'data-default="720p"' in html
        assert re.search(r'<option value="720p"\s+selected>', html)
        assert not re.search(r'<option value="480p"\s+selected>', html)

    def test_project_ready_shows_preprocess_toggle(self, client, db_session):
        """The GPT Image 2 preprocessing toggle + per-image prompts field are
        present on the New Run form, including the explicit Character
        reference photo field in the preprocess panel."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get(f"/v2/projects/{p.id}")
        html = resp.text
        assert 'name="preprocess_enabled"' in html
        assert 'id="preprocess-image-prompts"' in html
        assert "GPT Image" in html
        # Explicit Character reference photo field in the preprocess panel.
        assert 'name="preprocess_reference_file"' in html
        assert 'name="preprocess_reference_url"' in html
        assert "Character reference photo" in html

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
# "Sent to Final" column on the project's Existing-runs table
# ---------------------------------------------------------------------------


class TestSentToFinalColumn:
    """The Existing-runs table names WHICH KIND of video reached the Final
    Creatives sheet (original / tail / captioned / overlay) — replacing the old
    "Final / Caption" sheet-status pair. See web_v2._final_creatives_sent."""

    def _page(self, client, db_session, **run_kwargs):
        p = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, p.id, status=RunStatus.done, **run_kwargs)
        return p, run

    def test_nothing_sent_shows_no_kind_badges(self, client, db_session):
        p, _run = self._page(client, db_session)
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-" not in html

    def test_run_result_exported_shows_original(self, client, db_session):
        p, _run = self._page(
            client, db_session, final_creatives_exported_at=datetime.now(timezone.utc)
        )
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-original" in html
        assert "badge-sent-tail" not in html

    def test_exported_tail_edit_shows_tail_without_original(self, client, db_session):
        """A tail edit on the sheet counts for its run even when the run's own
        result never went out — that is the whole point of the column."""
        p, run = self._page(client, db_session)
        _make_tail_edit(
            db_session,
            run.id,
            status=TailEditStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-tail" in html
        assert "badge-sent-original" not in html

    def test_unexported_tail_edit_is_not_counted(self, client, db_session):
        p, run = self._page(client, db_session)
        _make_tail_edit(db_session, run.id, status=TailEditStatus.done)
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-" not in html

    def test_exported_caption_shows_captioned(self, client, db_session):
        p, run = self._page(client, db_session)
        _make_caption(
            db_session,
            run.id,
            status=CaptionStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-captioned" in html

    def test_exported_overlay_shows_overlay(self, client, db_session):
        p, run = self._page(client, db_session)
        _make_overlay_job(
            db_session,
            source_type="run",
            source_run_id=run.id,
            status=OverlayStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-overlay" in html

    def test_overlay_burned_from_a_tail_edit_counts_for_the_parent_run(
        self, client, db_session
    ):
        p, run = self._page(client, db_session)
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.done)
        _make_overlay_job(
            db_session,
            source_type="tail_edit",
            source_tail_edit_id=edit.id,
            status=OverlayStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        html = client.get(f"/v2/projects/{p.id}").text
        # The overlay is the thing on the sheet — the tail edit it read is not.
        assert "badge-sent-overlay" in html
        assert "badge-sent-tail" not in html

    def test_legacy_overlay_without_parent_stamp_still_counts(self, client, db_session):
        """Rows created between migration d7c9e1a3f582 and its deploy have a
        NULL parent_run_id; the soft source reference must still find the run."""
        p, run = self._page(client, db_session)
        _make_overlay_job(
            db_session,
            source_type="run",
            source_run_id=run.id,
            parent_run_id=None,
            status=OverlayStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-overlay" in html

    def test_soft_deleted_overlay_is_not_counted(self, client, db_session):
        p, run = self._page(client, db_session)
        _make_overlay_job(
            db_session,
            source_type="run",
            source_run_id=run.id,
            status=OverlayStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
            deleted_at=datetime.now(timezone.utc),
        )
        html = client.get(f"/v2/projects/{p.id}").text
        assert "badge-sent-" not in html

    def test_two_exported_tail_edits_show_a_count(self, client, db_session):
        p, run = self._page(client, db_session)
        for _ in range(2):
            _make_tail_edit(
                db_session,
                run.id,
                status=TailEditStatus.done,
                final_creatives_exported_at=datetime.now(timezone.utc),
            )
        html = client.get(f"/v2/projects/{p.id}").text
        assert "×2" in html

    def test_kinds_are_per_run_not_per_project(self, client, db_session):
        """Two runs in one project: only the exporting run gets a badge."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(
            db_session,
            p.id,
            name="RUN-WITH-EXPORT",
            status=RunStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        _make_run(db_session, p.id, name="RUN-WITHOUT-EXPORT", status=RunStatus.done)
        html = client.get(f"/v2/projects/{p.id}").text
        assert html.count("badge-sent-original") == 1
        # Rows are newest-first, so the exporting run is the LAST of the two —
        # the single badge must fall inside its row, i.e. after its name.
        assert html.index("RUN-WITHOUT-EXPORT") < html.index("RUN-WITH-EXPORT")
        assert html.index("RUN-WITH-EXPORT") < html.index("badge-sent-original")

    def test_legacy_caption_demo_status_is_not_shown(self, client, db_session):
        """``caption_demo_status`` is a legacy column (the "Add to Caption Demo"
        sheet export was removed) — an old non-null value must not leak into
        this column. Only the kind badges belong here."""
        p, _run = self._page(client, db_session, caption_demo_status="added_to_table")
        html = client.get(f"/v2/projects/{p.id}").text
        assert "added to table" not in html
        assert "Final / Caption" not in html
        assert "Sent to Final" in html

    def test_model_column_shows_the_short_tag(self, client, db_session):
        """The runs table carries the compact model tag, not the full label —
        the same value the Final Creatives Idea column exports."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(db_session, p.id, status=RunStatus.done, model="seedance-fast")
        html = client.get(f"/v2/projects/{p.id}").text
        assert ">SDf2<" in html
        # Full label stays reachable as the hover title.
        assert 'title="Seedance 2.0 Fast"' in html

    def test_model_column_covers_every_registry_model(self, client, db_session):
        """Every model key a run can hold renders its registry short_label —
        pins the table to ai_models rather than a hand-kept copy."""
        for key, spec in ai_models.AI_MODELS.items():
            p = _make_project(db_session, status=ProjectStatus.ready)
            _make_run(db_session, p.id, status=RunStatus.done, model=key)
            html = client.get(f"/v2/projects/{p.id}").text
            assert f">{spec.short_label}<" in html, key

    def test_status_fragment_renders_the_same_column(self, client, db_session):
        """The HTMX status fragment and the full page render the same partial —
        the context key must be supplied on both paths or one 500s."""
        p, run = self._page(client, db_session)
        _make_caption(
            db_session,
            run.id,
            status=CaptionStatus.done,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        resp = client.get(f"/v2/projects/{p.id}/status-fragment")
        assert resp.status_code == 200
        assert "badge-sent-captioned" in resp.text


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

    def test_fragment_analyzing_does_not_render_segments_or_runs(self, client, db_session):
        """Fix #19: analyzing-status fragment should pass empty segments/runs
        to the template (avoiding expensive queries) since the template only
        renders them when status is 'ready'. Verify that the segment editor
        UI is not present in the analyzing response."""
        p = _make_project(db_session, status=ProjectStatus.analyzing)
        # Create some segments/runs in the DB so they would be loaded if we queried
        _make_segment_def(db_session, p.id, 0)
        _make_run(db_session, p.id)

        resp = client.get(f"/v2/projects/{p.id}/status-fragment")
        assert resp.status_code == 200

        # The response should show the spinner, not the segment editor
        assert "Analyzing" in resp.text or "spinner" in resp.text
        # Critically: the segment editor UI must NOT be present when status is
        # "analyzing" (it only appears when status is "ready" and segments are passed)
        assert "seg-tbody" not in resp.text
        assert "Save segments" not in resp.text


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

    def test_run_detail_header_and_selector_reflect_the_language(
        self, client, db_session
    ):
        """The header shows the suffixed id and the selector pre-picks Japanese."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, language="ja")
        html = client.get(f"/v2/runs/{r.id}").text
        assert f"{r.id[:8]}-ja" in html
        assert '<option value="ja" selected>Japanese</option>' in html

    def test_run_detail_defaults_the_selector_to_english(self, client, db_session):
        """A run with no language shows a bare id and English pre-picked."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id)
        html = client.get(f"/v2/runs/{r.id}").text
        assert '<option value="en" selected>English</option>' in html
        assert f"{r.id[:8]}-ja" not in html

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

    def test_run_detail_done_shows_the_audio_patch_card(
        self, client, db_session, tmp_path
    ):
        """The "Fix audio from another run" card, with the project's other done
        runs offered as donors (docs/run-audio-patch.md §6)."""
        result_file = tmp_path / "final_ap.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            result_local_path=str(result_file),
        )
        donor = _make_run(
            db_session, p.id, status=RunStatus.done, name="RplDcTop",
            result_local_path=str(result_file),
        )

        html = client.get(f"/v2/runs/{r.id}").text
        assert "Fix audio from another run" in html
        assert 'onclick="submitAudioPatch(' in html
        assert donor.short_id in html
        # Collapsed by default — two players are the tallest thing on the page
        # and most visits are not about fixing audio. `<details>` with no
        # `open` attribute is what makes that true.
        assert '<details id="audio-patch-details">' in html
        assert '<details id="audio-patch-details" open' not in html
        # ...and the whole tool is INSIDE it. A </details> that closed early
        # would still render and still pass the checks above while leaving the
        # players and the submit button permanently visible below the summary.
        start = html.index('<details id="audio-patch-details">')
        end = html.index("</details>", start)
        block = html[start:end]
        for inside in ("ap-target-video", "ap-donor-video", "submitAudioPatch("):
            assert inside in block, f"{inside} escaped the collapsible block"
        # The donor player reads the sibling's own result endpoint.
        assert "ap-donor-video" in html
        # And the card says what it does to the delivered creative.
        assert "same Job ID, same URL" in html

    def test_audio_patch_card_says_so_when_there_is_no_donor(
        self, client, db_session, tmp_path
    ):
        result_file = tmp_path / "final_solo.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            result_local_path=str(result_file),
        )

        html = client.get(f"/v2/runs/{r.id}").text
        assert "Fix audio from another run" in html
        assert "No other finished run in this project" in html
        assert 'onclick="submitAudioPatch(' not in html

    def test_run_detail_done_shows_add_to_sheet_button(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_gdrive_file_id="GD123",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert 'onclick="exportToSheet(this)"' in html
        assert "Already logged to the Final Creatives sheet" not in html

    def test_run_detail_done_highlights_when_already_exported(self, client, db_session):
        from datetime import datetime, timezone

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_gdrive_file_id="GD123",
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert "Already logged to the Final Creatives sheet" in html
        # The active "Add" button must not be offered again
        assert 'onclick="exportToSheet(this)"' not in html

    def test_failed_run_with_a_drive_file_shows_the_drive_check(self, client, db_session):
        """AICP-348: a check button, never a re-link offered from a stale flag
        — the re-link panel starts hidden until a fresh check says missing."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.failed, result_gdrive_file_id="GD123")
        html = client.get(f"/v2/runs/{r.id}").text
        assert 'id="drive-relink-card"' in html
        assert "Check Drive file" in html
        assert 'class="js-drive-relink" style="display:none' in html
        assert "This run stays <em>failed</em>" in html

    def test_run_without_a_drive_file_has_no_drive_check(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        assert 'id="drive-relink-card"' not in client.get(f"/v2/runs/{r.id}").text

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

    def test_run_detail_retry_is_a_fetch_button_not_a_navigating_form(
        self, client, db_session
    ):
        """Retry must POST via fetch and stay on the run page.

        As a plain <form method="post"> the browser navigated to
        /api/v2/runs/{id}/retry and dumped the raw RunResponse JSON on the
        operator, with no way back and no sight of a 409's detail.
        """
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.failed,
            error_message="Generation timed out",
        )
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        assert 'js-retry-run' in html
        assert f'action="/api/v2/runs/{r.id}/retry"' not in html

    def test_run_detail_status_badge_uses_value(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.queued)
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        assert "badge-queued" in html
        assert "RunStatus." not in html

    def test_done_run_links_each_segment_to_its_own_clip(
        self, client, db_session, tmp_path
    ):
        """The per-segment panel links the raw clip kie.ai returned, so a bad
        generation can be told apart from a bad stitch."""
        clip = tmp_path / "result_0000.mp4"
        clip.write_bytes(b"\x00CLIP")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        sd = _make_segment_def(db_session, p.id, 0)
        rs = _make_run_segment(
            db_session, r.id, 0,
            segment_def_id=sd.id,
            status=SegmentStatus.completed,
            local_result_path=str(clip),
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert f"/api/v2/runs/{r.id}/segments/{rs.id}/result?v=" in html
        assert "download=1" in html

    def test_segment_without_a_clip_gets_no_link(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.incomplete)
        sd = _make_segment_def(db_session, p.id, 0)
        rs = _make_run_segment(
            db_session, r.id, 0,
            segment_def_id=sd.id,
            status=SegmentStatus.failed,
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert f"/segments/{rs.id}/result" not in html

    def test_in_progress_run_links_the_clips_that_already_landed(
        self, client, db_session, tmp_path
    ):
        """Mid-run the operator can open the segments that are done — the
        remaining ones are still being paid for."""
        clip = tmp_path / "result_0000.mp4"
        clip.write_bytes(b"\x00CLIP")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        sd = _make_segment_def(db_session, p.id, 0)
        done_rs = _make_run_segment(
            db_session, r.id, 0,
            segment_def_id=sd.id,
            status=SegmentStatus.completed,
            local_result_path=str(clip),
        )
        pending_rs = _make_run_segment(
            db_session, r.id, 1,
            segment_def_id=sd.id,
            status=SegmentStatus.generating,
        )
        html = client.get(f"/v2/runs/{r.id}/status-fragment").text
        assert "Generated so far" in html
        assert f"/api/v2/runs/{r.id}/segments/{done_rs.id}/result?v=" in html
        assert f"/segments/{pending_rs.id}/result" not in html

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

    def test_run_fragment_preprocessing_shows_message(self, client, db_session):
        """A run in `preprocessing` shows the GPT Image 2 status message."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.preprocessing)
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        assert "preprocessing" in resp.text
        assert "GPT Image" in resp.text

    def test_run_fragment_preprocess_review_shows_approve_and_regenerate(
        self, client, db_session
    ):
        """R2-new: a run in `preprocess_review` shows the Approve and Regenerate
        buttons."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            reference_image_urls=["https://x.test/a.jpg", "https://x.test/b.jpg"],
            preprocessed_reference_urls=["https://pp.test/a.png", "https://pp.test/b.png"],
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        # The review card is rendered (not the in-progress polling section).
        assert "Review GPT Image results" in resp.text
        assert "Approve" in resp.text
        assert "Regenerate" in resp.text
        # The approve endpoint is wired up.
        assert "/approve-preprocess" in resp.text
        assert "/regenerate-preprocess" in resp.text

    def test_run_fragment_review_offers_per_image_regenerate(
        self, client, db_session
    ):
        """Every generated image carries its own regenerate control, so one bad
        image out of three does not mean redoing all three."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=[
                "https://x.test/char.jpg", "https://x.test/t1.jpg",
                "https://x.test/t2.jpg",
            ],
            preprocess_image_prompts=["", "pose it", "pose it again"],
            preprocessed_reference_urls=[
                "https://x.test/char.jpg", "https://pp.test/a.png",
                "https://pp.test/b.png",
            ],
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        # One opener per generated trim — never for index 0, the character photo.
        assert 'onclick="openRegenerateOne(1)"' in resp.text
        assert 'onclick="openRegenerateOne(2)"' in resp.text
        assert 'onclick="openRegenerateOne(0)"' not in resp.text
        assert 'id="regen-one-panel"' in resp.text
        # The editor opens on the prompt that produced the image.
        assert "pose it again" in resp.text

    def test_run_fragment_review_hides_per_image_regenerate_for_all_as_refs(
        self, client, db_session
    ):
        """One combined image covers every ref there — "just this one" is the
        full Regenerate, and the API refuses the per-index call anyway."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="all_as_refs",
            reference_image_urls=["https://x.test/char.jpg", "https://x.test/t1.jpg"],
            preprocess_image_prompts=["combine them"],
            preprocessed_reference_urls=["https://pp.test/one.png"],
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        # No per-image buttons and no editor — only the whole-phase Regenerate.
        # (The helper function itself is always in the script block; it is the
        # call sites that are mode-dependent.)
        assert 'onclick="openRegenerateOne(' not in resp.text
        assert 'id="regen-one-panel"' not in resp.text
        assert "/regenerate-preprocess" in resp.text

    def test_run_fragment_review_marks_an_empty_slot_and_the_error(
        self, client, db_session
    ):
        """A regeneration that failed parks here: the operator must see which
        image is missing and why, not a silently absent thumbnail."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=["https://x.test/char.jpg", "https://x.test/t1.jpg"],
            preprocess_image_prompts=["", "pose it"],
            preprocessed_reference_urls=["https://x.test/char.jpg", None],
            error_message="GPT Image 2 preprocessing failed: content policy",
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        assert "no image" in resp.text
        assert "content policy" in resp.text

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


# ---------------------------------------------------------------------------
# Trim timestamps in the Reference images block
#
# A trim staged off the source video keeps the moment it was cut at in its
# filename and nowhere else (storage.staged_frame_name). The Reference images
# block reads it back so an operator looking at a run can tell WHICH moment
# each frame is — the same way the GPT Image 2 results carry their prompt.
# ---------------------------------------------------------------------------


class TestTrimTimestamps:
    def test_trim_time_filter_reads_the_staged_filename(self):
        """The filter is the whole feature: name in, "12.4s" out."""
        from app.storage import (
            staged_frame_name,
            staged_frame_time_label,
            staged_frame_timestamp,
        )

        staged = "/data/projects/p1/runs/r1/references/" + staged_frame_name(12.4)
        assert staged_frame_timestamp(staged) == 12.4
        assert staged_frame_time_label(staged) == "12.4s"

    def test_trim_time_filter_ignores_everything_that_is_not_a_staged_frame(self):
        """Uploads, outside URLs and a bare "frame_120.jpg" someone uploaded get
        no label — a wrong timestamp would be worse than none."""
        from app.storage import staged_frame_time_label

        for path in (
            "https://x.test/a.jpg",
            "/data/projects/p1/runs/r1/references/char.png",
            "/data/projects/p1/runs/r1/references/frame_120.jpg",
            "",
            None,
        ):
            assert staged_frame_time_label(path) is None

    def test_reference_block_shows_the_trim_timestamp(self, client, db_session):
        """A staged trim renders its timestamp under the thumbnail."""
        from app.storage import staged_frame_name

        p = _make_project(db_session, status=ProjectStatus.ready)
        trim = "/data/projects/p1/runs/r1/references/" + staged_frame_name(12.4)
        r = _make_run(
            db_session, p.id, status=RunStatus.processing,
            reference_image_urls=[trim],
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        assert '<span class="trim-time"' in resp.text
        assert "12.4s" in resp.text

    def test_reference_block_labels_only_the_staged_frames(self, client, db_session):
        """An ordinary reference next to a trim gets no timestamp of its own."""
        from app.storage import staged_frame_name

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.processing,
            reference_image_urls=[
                "https://x.test/character.jpg",
                "/data/projects/p1/runs/r1/references/" + staged_frame_name(3.0),
            ],
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        # Two thumbnails, one timestamp caption.
        assert resp.text.count("reference-image?index=") == 2
        assert resp.text.count('class="trim-time"') == 1
        assert "3.0s" in resp.text

    def test_review_grid_shows_the_trim_each_image_came_from(
        self, client, db_session
    ):
        """At preprocess_review the generated images sit next to their prompt —
        the trim's timestamp joins it, so regenerating one image does not mean
        guessing which moment of the video it was."""
        from app.storage import staged_frame_name

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            reference_image_urls=[
                "https://x.test/char.jpg",
                "/data/projects/p1/runs/r1/references/" + staged_frame_name(7.2),
            ],
            preprocess_image_prompts=["", "pose it"],
            preprocessed_reference_urls=[
                "https://x.test/char.jpg", "https://pp.test/a.png",
            ],
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        # Once in the originals grid, once under the image it produced.
        assert resp.text.count('class="trim-time"') == 2
        assert resp.text.count("&#9201; 7.2s") == 2

    def test_review_grid_omits_the_timestamp_for_all_as_refs(
        self, client, db_session
    ):
        """all_as_refs melts every trim into ONE image, so no single timestamp
        is true of it — only the originals grid is labelled."""
        from app.storage import staged_frame_name

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.preprocess_review,
            preprocess_enabled=True,
            preprocess_mode="all_as_refs",
            reference_image_urls=[
                "https://x.test/char.jpg",
                "/data/projects/p1/runs/r1/references/" + staged_frame_name(7.2),
            ],
            preprocess_image_prompts=["combine them"],
            preprocessed_reference_urls=["https://pp.test/one.png"],
        )
        resp = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp.status_code == 200
        # Only the originals grid labels it.
        assert resp.text.count('class="trim-time"') == 1


# ---------------------------------------------------------------------------
# Tail edits UI ("Change ending" — app/pipeline_tail.py)
# ---------------------------------------------------------------------------


class TestTailEditsUI:
    def test_done_run_page_shows_change_ending(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        resp = client.get(f"/v2/runs/{r.id}")
        assert resp.status_code == 200
        assert "Change ending" in resp.text
        assert "tail-edit-cut-sec" in resp.text
        assert "trim only" in resp.text

    def test_change_ending_cut_bounds_rendered(self, client, db_session):
        """fix/tail-edit-cut-bounds: the "Cut at (s)" field takes 0.01
        steps, a real "Max" button is offered, and the duration shown is
        rounded DOWN (never "%.1f", which rounds up and can show a value the
        real video can't satisfy)."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_duration_sec=12.28,
        )
        resp = client.get(f"/v2/runs/{r.id}")
        assert resp.status_code == 200
        assert 'step="0.01"' in resp.text
        assert 'id="tail-edit-max-btn"' in resp.text
        assert ">Max<" in resp.text
        # Rounded DOWN to 2 decimals: never "12.29" or the old %.1f "12.3"
        # (both of which exceed the real duration).
        assert "12.28" in resp.text
        assert "12.29" not in resp.text
        assert "12.3s" not in resp.text
        # 12.28 - 0.05 == 12.229999999999999 in IEEE754 double precision — a
        # naive floor (Jinja's `round(2, 'floor')`, or a bare
        # Math.floor(x*100)/100 in JS) floors that DOWN to 12.22, one cent
        # below the intended 12.23. Regression guard for that exact bug.
        assert 'max="12.23"' in resp.text
        assert 'max="12.22"' not in resp.text

    def test_non_done_run_page_omits_change_ending(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        resp = client.get(f"/v2/runs/{r.id}")
        assert resp.status_code == 200
        assert "Change ending" not in resp.text

    def test_change_ending_offers_library_clips(self, client, db_session, tmp_path):
        from app.models import LibraryClip

        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        clip = LibraryClip(name="Outro FR", local_path="/tmp/nowhere.mp4", duration_sec=2.5)
        db_session.add(clip)
        db_session.commit()

        resp = client.get(f"/v2/runs/{r.id}")
        assert resp.status_code == 200
        assert "Outro FR" in resp.text

    def test_tail_edits_fragment_renders_list(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )
        _make_tail_edit(db_session, r.id, cut_sec=3.5, status=TailEditStatus.done)

        resp = client.get(f"/v2/runs/{r.id}/tail-edits-fragment")
        assert resp.status_code == 200
        assert "3.5s" in resp.text
        assert "badge-done" in resp.text

    def test_tail_edits_fragment_polls_while_active(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_tail_edit(db_session, r.id, status=TailEditStatus.processing)

        resp = client.get(f"/v2/runs/{r.id}/tail-edits-fragment")
        assert resp.status_code == 200
        assert "tail-edits-fragment" in resp.text
        assert "hx-trigger" in resp.text

    def test_tail_edits_fragment_stops_polling_when_terminal(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_tail_edit(db_session, r.id, status=TailEditStatus.failed, error_message="x")

        resp = client.get(f"/v2/runs/{r.id}/tail-edits-fragment")
        assert resp.status_code == 200
        assert "hx-trigger" not in resp.text

    def test_tail_edits_fragment_unknown_run_404(self, client):
        resp = client.get("/v2/runs/no-such-run/tail-edits-fragment")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Overlays card + history on the run page (the tail-edit block's shape, for
# the overlay tool) — app/web_v2.py::_get_overlay_context
# ---------------------------------------------------------------------------


class TestRunOutcomeCard:
    """AICP-299 Outcome card mount: both knowledge flags, a done run, project in scope."""

    @pytest.fixture()
    def knowledge_flags(self, monkeypatch):
        from app.config import settings

        for name, value in (("KNOWLEDGE_ENABLED", True), ("KNOWLEDGE_FEEDBACK_UI", True),
                            ("KNOWLEDGE_PROJECT_IDS", "*"), ("ASSISTANT_ALL_PROJECTS", True)):
            monkeypatch.setattr(settings, name, value, raising=False)
        return settings

    def test_flags_off_renders_nothing(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        resp = client.get(f"/v2/runs/{r.id}")
        assert resp.status_code == 200
        assert resp.context["knowledge_card"] is False
        assert 'id="knowledge-card"' not in resp.text and "knowledge-fragment" not in resp.text

    def test_ui_flag_alone_or_knowledge_flag_alone_renders_nothing(self, client, db_session, knowledge_flags,
                                                                  monkeypatch):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        monkeypatch.setattr(knowledge_flags, "KNOWLEDGE_FEEDBACK_UI", False)
        assert 'id="knowledge-card"' not in client.get(f"/v2/runs/{r.id}").text
        monkeypatch.setattr(knowledge_flags, "KNOWLEDGE_FEEDBACK_UI", True)
        monkeypatch.setattr(knowledge_flags, "KNOWLEDGE_ENABLED", False)
        assert 'id="knowledge-card"' not in client.get(f"/v2/runs/{r.id}").text

    def test_done_in_scope_run_mounts_the_card_outside_the_status_poll(self, client, db_session, knowledge_flags):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        html = client.get(f"/v2/runs/{r.id}").text
        mount = f'<div id="knowledge-card" hx-get="/v2/runs/{r.id}/knowledge-fragment" hx-trigger="load"'
        assert mount in html
        # A sibling after the polling target, never inside it.
        assert html.index('id="run-status-content"') < html.index(mount)
        status_html = client.get(f"/v2/runs/{r.id}/status-fragment").text
        assert "knowledge-card" not in status_html and "knowledge-fragment" not in status_html

    def test_not_done_run_has_no_card(self, client, db_session, knowledge_flags):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.stitching)
        assert 'id="knowledge-card"' not in client.get(f"/v2/runs/{r.id}").text

    def test_out_of_scope_project_has_no_card(self, client, db_session, knowledge_flags, monkeypatch):
        p = _make_project(db_session, status=ProjectStatus.ready)
        other = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        monkeypatch.setattr(knowledge_flags, "KNOWLEDGE_PROJECT_IDS", other.id)
        assert 'id="knowledge-card"' not in client.get(f"/v2/runs/{r.id}").text
        monkeypatch.setattr(knowledge_flags, "KNOWLEDGE_PROJECT_IDS", p.id)
        assert 'id="knowledge-card"' in client.get(f"/v2/runs/{r.id}").text

    def test_fragment_404s_with_flags_off(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        assert client.get(f"/v2/runs/{r.id}/knowledge-fragment").status_code == 404
        assert client.post(f"/v2/runs/{r.id}/knowledge/start").status_code == 404


class TestRunOverlaysUI:
    def test_done_run_page_shows_overlays_card(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)

        html = client.get(f"/v2/runs/{r.id}").text
        assert "Overlays" in html
        assert 'id="overlay-source"' in html
        assert "startOverlay" in html

    def test_non_done_run_page_omits_overlays_card(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.processing)

        html = client.get(f"/v2/runs/{r.id}").text
        assert 'id="overlay-source"' not in html

    def test_source_picker_offers_done_tail_edits(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        done = _make_tail_edit(db_session, r.id, cut_sec=4.5, status=TailEditStatus.done)
        in_flight = _make_tail_edit(db_session, r.id, status=TailEditStatus.processing)

        html = client.get(f"/v2/runs/{r.id}").text
        assert f'<option value="tail_edit:{done.id}">Tail edit' in html
        assert f'<option value="tail_edit:{in_flight.id}">Tail edit' not in html

    def test_source_picker_offers_done_captions(self, client, db_session):
        """A burned-captions video is a source too, tagged with its own type."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        done = _make_caption(db_session, r.id, status=CaptionStatus.done)
        in_flight = _make_caption(db_session, r.id, status=CaptionStatus.burning)

        html = client.get(f"/v2/runs/{r.id}").text
        assert f'<option value="caption:{done.id}">Captions' in html
        assert f'<option value="caption:{in_flight.id}">Captions' not in html
        # The run's own result stays the default, and carries its type too.
        assert '<option value="run:">Run result</option>' in html

    def test_history_lists_overlay_of_this_run(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        job = _make_overlay_job(
            db_session, source_run_id=r.id, status=OverlayStatus.done,
            gdrive_file_id="drive-1",
            overlays=[{"asset_path": "assets/card_01.png"}],
        )

        html = client.get(f"/v2/runs/{r.id}").text
        assert "Overlay history" in html
        assert f"/v2/overlays/{job.id}" in html
        assert "Run result" in html

    def test_history_shows_tail_edit_sourced_overlay(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        edit = _make_tail_edit(db_session, r.id, cut_sec=7.25, status=TailEditStatus.done)
        _make_overlay_job(
            db_session, source_type="tail_edit", source_tail_edit_id=edit.id,
            status=OverlayStatus.done,
        )

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert "Tail edit: Cut @7.2s" in html

    def test_history_shows_caption_sourced_overlay(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        cap = _make_caption(db_session, r.id, status=CaptionStatus.done)
        _make_overlay_job(
            db_session, source_type="caption", source_caption_id=cap.id,
            status=OverlayStatus.done,
        )

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert "Caption:" in html
        assert cap.id[:8] in html
        assert "(deleted)" not in html

    def test_history_shows_deleted_when_caption_source_missing(self, client, db_session):
        """The source is a soft reference — a vanished caption must not 500."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        cap = _make_caption(db_session, r.id, status=CaptionStatus.done)
        _make_overlay_job(
            db_session, source_type="caption", source_caption_id=cap.id,
            status=OverlayStatus.done,
        )
        db_session.delete(cap)
        db_session.commit()

        resp = client.get(f"/v2/runs/{r.id}/overlays-fragment")
        assert resp.status_code == 200
        assert "Caption: (deleted)" in resp.text

    def test_history_excludes_other_runs_overlays(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        mine = _make_run(db_session, p.id, status=RunStatus.done)
        theirs = _make_run(db_session, p.id, status=RunStatus.done)
        other_job = _make_overlay_job(
            db_session, source_run_id=theirs.id, status=OverlayStatus.done
        )

        html = client.get(f"/v2/runs/{mine.id}/overlays-fragment").text
        assert other_job.id not in html

    def test_history_hides_soft_deleted_overlays(self, client, db_session):
        from datetime import datetime, timezone

        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        gone = _make_overlay_job(
            db_session, source_run_id=r.id, status=OverlayStatus.done,
            deleted_at=datetime.now(timezone.utc),
        )

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert gone.id not in html

    def test_history_polls_while_active(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_overlay_job(
            db_session, source_run_id=r.id, status=OverlayStatus.processing
        )

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert "overlays-fragment" in html
        assert "hx-trigger" in html

    def test_history_stops_polling_when_terminal(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_overlay_job(
            db_session, source_run_id=r.id, status=OverlayStatus.failed, error="boom"
        )

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert "hx-trigger" not in html
        assert "boom" in html
        # A failed render is one click from re-queuing, same as a tail edit.
        assert "retryRunOverlay" in html

    def test_history_row_shows_run_job_id_for_a_single_overlay(
        self, client, db_session
    ):
        """One overlay on the run → the sheet gets the RUN's id, language and all."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done, language="ja")
        _make_overlay_job(db_session, source_run_id=r.id, status=OverlayStatus.done)

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert f"<code style=\"font-size:11px\" title=\"What the Final Creatives sheet receives as Job ID\">{r.id[:8]}-ja</code>" in html

    def test_history_rows_show_combined_job_ids_for_several_overlays(
        self, client, db_session
    ):
        """Two overlays on one run → each row carries run prefix + own prefix."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done, language="ja")
        a = _make_overlay_job(db_session, source_run_id=r.id, status=OverlayStatus.done)
        b = _make_overlay_job(db_session, source_run_id=r.id, status=OverlayStatus.done)

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert f"{r.id[:8]}_{a.id[:8]}-ja" in html
        assert f"{r.id[:8]}_{b.id[:8]}-ja" in html

    def test_overlays_fragment_unknown_run_404(self, client):
        resp = client.get("/v2/runs/no-such-run/overlays-fragment")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chaining derived creatives — tail edit sourced from a caption
# (docs/burned-captions.md §3.7.4)
# ---------------------------------------------------------------------------


class TestTailEditSourcePicker:
    def test_source_select_defaults_to_run_result_with_no_captions(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)

        html = client.get(f"/v2/runs/{r.id}").text
        assert 'id="tail-edit-source-caption"' in html
        assert "Run result (no captions)" in html

    def test_source_select_offers_one_option_per_done_caption(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        c1 = _make_caption(db_session, r.id, status=CaptionStatus.done)
        c2 = _make_caption(db_session, r.id, status=CaptionStatus.done)

        html = client.get(f"/v2/runs/{r.id}").text
        assert f'<option value="{c1.id}" data-duration=' in html
        assert f'<option value="{c2.id}" data-duration=' in html
        assert c1.id[:8] in html
        assert c2.id[:8] in html

    def test_source_select_excludes_non_done_captions(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        in_flight = _make_caption(db_session, r.id, status=CaptionStatus.transcribing)

        html = client.get(f"/v2/runs/{r.id}").text
        assert f'<option value="{in_flight.id}">' not in html

    def test_submit_tail_edit_js_posts_source_caption_id(self, client, db_session):
        # The rendered-HTML-is-the-harness lesson (docs/burned-captions.md
        # §3.7.4 test note): there is no JS engine here, so assert on the
        # actual submit function body instead of executing it.
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        html = client.get(f"/v2/runs/{r.id}").text
        assert "source_caption_id" in html
        assert "tail-edit-source-caption" in html

    def test_tail_edit_row_default_source_shows_run_result(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_tail_edit(db_session, r.id, status=TailEditStatus.done)

        html = client.get(f"/v2/runs/{r.id}/tail-edits-fragment").text
        assert "Run result" in html

    def test_tail_edit_row_shows_chained_caption_source_label(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        cap = _make_caption(db_session, r.id, status=CaptionStatus.done)
        _make_tail_edit(
            db_session, r.id, status=TailEditStatus.done,
            source_type="caption", source_caption_id=cap.id,
        )

        html = client.get(f"/v2/runs/{r.id}/tail-edits-fragment").text
        assert "Caption:" in html
        assert cap.id[:8] in html
        assert "(deleted)" not in html

    def test_tail_edit_row_shows_deleted_when_caption_source_missing(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_tail_edit(
            db_session, r.id, status=TailEditStatus.done,
            source_type="caption", source_caption_id="no-such-caption-id",
        )

        resp = client.get(f"/v2/runs/{r.id}/tail-edits-fragment")
        assert resp.status_code == 200
        assert "(deleted)" in resp.text

    def test_tail_edit_row_source_label_also_shown_on_run_detail_page(
        self, client, db_session
    ):
        # Same context flows into the full page (not just the fragment).
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        cap = _make_caption(db_session, r.id, status=CaptionStatus.done)
        _make_tail_edit(
            db_session, r.id, status=TailEditStatus.done,
            source_type="caption", source_caption_id=cap.id,
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert "Caption:" in html
        assert cap.id[:8] in html

    def test_non_done_run_page_is_query_free_for_chaining_context(
        self, client, db_session, monkeypatch
    ):
        """The chaining lookups must never run on the every-2s poll for a
        non-done run — same rule as the pre-existing tail-edit/caption
        context queries (see `_get_tail_edit_context`'s docstring)."""
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
        assert "Change ending" not in resp.text
        assert "captions-card" not in resp.text

        resp2 = client.get(f"/v2/runs/{r.id}/status-fragment")
        assert resp2.status_code == 200


# ---------------------------------------------------------------------------
# Dashboard run-parameter filter — GET /v2/?run_status=&resolution=&date_from=&date_to=
# ---------------------------------------------------------------------------


class TestDashboardFilter:
    """The dashboard filter bar: Type (project) + run-level filters, ANDed.

    Multi-selects mean "any of the ticked values"; a project is listed when ONE
    of its runs matches every active run filter at once.
    """

    def test_filter_by_resolution(self, client, db_session):
        p_hd = _make_project(db_session, name="HD")
        _make_run(db_session, p_hd.id, status=RunStatus.done, resolution="1080p")
        p_sd = _make_project(db_session, name="SD")
        _make_run(db_session, p_sd.id, status=RunStatus.done, resolution="480p")

        resp = client.get("/v2/?resolution=1080p")
        assert p_hd.id[:8] in resp.text
        assert p_sd.id[:8] not in resp.text

    def test_resolution_multi_select_is_an_or(self, client, db_session):
        p_hd = _make_project(db_session, name="HD")
        _make_run(db_session, p_hd.id, resolution="1080p")
        p_720 = _make_project(db_session, name="720")
        _make_run(db_session, p_720.id, resolution="720p")
        p_sd = _make_project(db_session, name="SD")
        _make_run(db_session, p_sd.id, resolution="480p")

        resp = client.get("/v2/?resolution=1080p&resolution=720p")
        assert p_hd.id[:8] in resp.text
        assert p_720.id[:8] in resp.text
        assert p_sd.id[:8] not in resp.text

    def test_filter_by_model(self, client, db_session):
        p_25 = _make_project(db_session, name="SD25")
        _make_run(db_session, p_25.id, model="seedance-2-5")
        p_omni = _make_project(db_session, name="Omni")
        _make_run(db_session, p_omni.id, model="gemini-omni")

        resp = client.get("/v2/?model=seedance-2-5")
        assert p_25.id[:8] in resp.text
        assert p_omni.id[:8] not in resp.text

    def test_filter_by_project_type(self, client, db_session):
        p_loc = _make_project(db_session, name="Loc", project_type="localisation")
        p_swap = _make_project(db_session, name="Swap", project_type="face_swap")

        resp = client.get("/v2/?project_type=localisation")
        assert p_loc.id[:8] in resp.text
        assert p_swap.id[:8] not in resp.text

    def test_project_type_filter_needs_no_runs(self, client, db_session):
        """Type filters the PROJECT, so a project with no runs still matches."""
        p = _make_project(db_session, name="Empty", project_type="cover_change")
        resp = client.get("/v2/?project_type=cover_change")
        assert p.id[:8] in resp.text

    def test_filter_by_language_any_of_selected(self, client, db_session):
        p_es = _make_project(db_session, name="ES")
        _make_run(db_session, p_es.id, language="es")
        p_ja = _make_project(db_session, name="JA")
        _make_run(db_session, p_ja.id, language="ja")
        p_pt = _make_project(db_session, name="PT")
        _make_run(db_session, p_pt.id, language="pt")

        resp = client.get("/v2/?lang=es&lang=ja")
        assert p_es.id[:8] in resp.text
        assert p_ja.id[:8] in resp.text
        assert p_pt.id[:8] not in resp.text

    def test_english_filter_matches_a_run_with_no_language(self, client, db_session):
        """NULL language IS English (app/languages.py) — most prod rows are NULL."""
        p_null = _make_project(db_session, name="Unset")
        _make_run(db_session, p_null.id, language=None)
        p_en = _make_project(db_session, name="EN")
        _make_run(db_session, p_en.id, language="en")
        p_es = _make_project(db_session, name="ES")
        _make_run(db_session, p_es.id, language="es")

        resp = client.get("/v2/?lang=en")
        assert p_null.id[:8] in resp.text
        assert p_en.id[:8] in resp.text
        assert p_es.id[:8] not in resp.text

    def test_filters_are_anded_on_the_same_run(self, client, db_session):
        # A project whose runs each match ONE filter but no single run matches
        # BOTH must be excluded (any-run-matches-all semantics).
        p_split = _make_project(db_session, name="Split")
        _make_run(db_session, p_split.id, model="seedance", resolution="1080p")
        _make_run(db_session, p_split.id, model="seedance-2-5", resolution="480p")
        p_match = _make_project(db_session, name="Match")
        _make_run(db_session, p_match.id, model="seedance-2-5", resolution="1080p")

        resp = client.get("/v2/?model=seedance-2-5&resolution=1080p")
        assert p_match.id[:8] in resp.text
        assert p_split.id[:8] not in resp.text

    def test_filter_by_date_range(self, client, db_session):
        from datetime import datetime, timezone

        p_old = _make_project(db_session, name="Old")
        _make_run(
            db_session, p_old.id, status=RunStatus.done,
            created_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )
        p_new = _make_project(db_session, name="New")
        _make_run(
            db_session, p_new.id, status=RunStatus.done,
            created_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
        )

        # from/to fully covers the June run only
        resp = client.get("/v2/?date_from=2026-06-01&date_to=2026-06-30")
        assert p_new.id[:8] in resp.text
        assert p_old.id[:8] not in resp.text

    def test_date_to_is_inclusive_of_the_day(self, client, db_session):
        from datetime import datetime, timezone

        p = _make_project(db_session, name="SameDay")
        _make_run(
            db_session, p.id, status=RunStatus.done,
            created_at=datetime(2026, 6, 15, 14, 30, tzinfo=timezone.utc),
        )
        # date_to equal to the run's day must still include it.
        resp = client.get("/v2/?date_from=2026-06-15&date_to=2026-06-15")
        assert p.id[:8] in resp.text

    def test_active_date_range_is_readable_on_the_closed_button(
        self, client, db_session
    ):
        """The dates hide behind a dropdown — but an ACTIVE one names itself.

        The panel is no longer force-opened over the table: the button spells
        the range out instead, so the filter is visible with it shut.
        """
        _make_project(db_session, name="Any")
        idle = client.get("/v2/").text
        assert "filter-menu is-active" not in idle
        active = client.get("/v2/?date_from=2026-06-01&date_to=2026-06-30").text
        assert "filter-menu is-active" in active
        assert "Dates: 2026-06-01 → 2026-06-30" in active

    def test_filter_by_run_id_short_prefix(self, client, db_session):
        # The short id shown in the UI (run.id[:8]) filters to its project.
        p_a = _make_project(db_session, name="A")
        run_a = _make_run(db_session, p_a.id, status=RunStatus.done)
        p_b = _make_project(db_session, name="B")
        _make_run(db_session, p_b.id, status=RunStatus.done)

        resp = client.get(f"/v2/?run_id={run_a.id[:8]}")
        assert resp.status_code == 200
        assert p_a.id[:8] in resp.text
        assert p_b.id[:8] not in resp.text

    def test_filter_by_run_id_is_case_insensitive(self, client, db_session):
        p = _make_project(db_session, name="Case")
        run = _make_run(db_session, p.id, status=RunStatus.done)

        resp = client.get(f"/v2/?run_id={run.id[:8].upper()}")
        assert p.id[:8] in resp.text

    def test_filter_by_run_id_no_match(self, client, db_session):
        p = _make_project(db_session, name="None")
        _make_run(db_session, p.id, status=RunStatus.done)

        resp = client.get("/v2/?run_id=ffffffff")
        assert p.id[:8] not in resp.text

    def test_run_id_sanitizer_strips_like_wildcards(self):
        # LIKE wildcards (% and _) and any non-uuid chars must be dropped so a
        # crafted value can't match everything; hex/dash are lowercased & kept.
        from app.web_v2 import _sanitize_run_id_prefix

        assert _sanitize_run_id_prefix("%") == ""
        assert _sanitize_run_id_prefix("_") == ""
        assert _sanitize_run_id_prefix("3F8A-1c9e%") == "3f8a-1c9e"
        assert _sanitize_run_id_prefix("  abc DEF  ") == "abcdef"
        assert _sanitize_run_id_prefix(None) == ""

    def test_multi_filter_values_drops_unknown_and_orders_by_registry(self):
        from app.web_v2 import _multi_filter_values

        allowed = ["480p", "720p", "1080p"]
        assert _multi_filter_values(["1080p", "480p"], allowed) == ["480p", "1080p"]
        assert _multi_filter_values(["nope", " 720p "], allowed) == ["720p"]
        assert _multi_filter_values(None, allowed) == []
        assert _multi_filter_values([""], allowed) == []

    def test_run_id_filter_input_present_and_preserved(self, client, db_session):
        _make_project(db_session, name="Any")
        resp = client.get("/v2/?run_id=3f8a1c9e")
        html = resp.text
        assert 'name="run_id"' in html
        assert 'value="3f8a1c9e"' in html

    def test_no_filters_lists_all(self, client, db_session):
        p1 = _make_project(db_session, name="P1")
        p2 = _make_project(db_session, name="P2")
        resp = client.get("/v2/")
        assert p1.id[:8] in resp.text
        assert p2.id[:8] in resp.text

    def test_filter_bar_offers_every_filter_and_keeps_the_selection(
        self, client, db_session
    ):
        _make_project(db_session, name="Any")
        html = client.get("/v2/?resolution=720p&model=seedance-2-5&lang=ja").text
        for name in ("run_id", "resolution", "project_type", "lang", "model",
                     "sent", "date_from", "date_to"):
            assert f'name="{name}"' in html
        # Ticked boxes come back ticked, and the closed button names them.
        flat = " ".join(html.split())
        assert 'value="720p" data-short="720p" checked' in flat
        assert 'value="seedance-2-5" data-short="SD25" checked' in flat
        assert 'value="ja" data-short="JA" checked' in flat
        # Untouched controls stay unticked.
        assert 'value="480p" data-short="480p" checked' not in flat

    def test_dropdown_captions_itself_with_its_name_then_its_selection(
        self, client, db_session
    ):
        """Idle a control shows the filter's NAME — that is its only label.

        The bar has no <label>s above the controls (that is what made it two
        rows tall), so a control that lost its caption would be unidentifiable.
        """
        _make_project(db_session, name="Any")
        flat = " ".join(client.get("/v2/").text.split())
        for name in ("Type", "Model", "Language", "Resolution", "Sheet", "Dates"):
            assert f'<span class="filter-menu-label">{name}</span>' in flat

        flat = " ".join(client.get("/v2/?model=seedance-2-5").text.split())
        assert '<span class="filter-menu-label">Model: SD25</span>' in flat
        assert "filter-menu is-active" in flat

    def test_a_long_selection_collapses_to_a_count(self, client, db_session):
        """Past two ticks the button shows "+N" so the control stays narrow."""
        _make_project(db_session, name="Any")
        flat = " ".join(
            client.get("/v2/?lang=en&lang=es&lang=pt&lang=ja").text.split()
        )
        assert '<span class="filter-menu-label">Language: EN, ES +2</span>' in flat

    def test_sent_filter_is_a_radio_dropdown_with_an_any_reset(
        self, client, db_session
    ):
        """Sent-to-sheet is tri-state, so it is radios, not checkboxes.

        The "Any" radio posts an empty value, which _run_filter_conditions
        already sanitises to "no condition" — the same thing the old
        <option value=""> did.
        """
        _make_project(db_session, name="Any")
        flat = " ".join(client.get("/v2/").text.split())
        assert 'type="radio" name="sent" value=""' in flat
        assert 'value="" data-short="" checked' in flat

        flat = " ".join(client.get("/v2/?sent=no").text.split())
        assert 'value="no" data-short="Not sent" checked' in flat
        assert 'value="" data-short="" checked' not in flat
        assert '<span class="filter-menu-label">Sheet: Not sent</span>' in flat

    def test_removed_run_status_filter_is_ignored(self, client, db_session):
        """The status filter is gone; a stale bookmark must not hide projects."""
        p = _make_project(db_session, name="Old bookmark")
        _make_run(db_session, p.id, status=RunStatus.failed)
        resp = client.get("/v2/?run_status=done")
        assert resp.status_code == 200
        assert p.id[:8] in resp.text


class TestDashboardSentToSheetFilter:
    """"Sent to sheet" — has ANYTHING from this run reached a spreadsheet?

    Deliberately blind to WHICH sheet (Final Creatives or the retired Caption
    Demo) and to which kind of video: the operator's question is "what is still
    outstanding?", answered by `sent=no`.
    """

    def _pair(self, db_session):
        """A project whose single run has exported nothing, plus its run."""
        project = _make_project(db_session, name="Fresh")
        return project, _make_run(db_session, project.id, status=RunStatus.done)

    def test_unsent_run_is_listed_by_not_sent_and_hidden_by_sent(
        self, client, db_session
    ):
        project, _ = self._pair(db_session)
        assert project.id[:8] in client.get("/v2/?sent=no").text
        assert project.id[:8] not in client.get("/v2/?sent=yes").text

    def test_run_exported_to_final_creatives_counts_as_sent(self, client, db_session):
        project, run = self._pair(db_session)
        run.final_creatives_exported_at = datetime.now(timezone.utc)
        db_session.commit()

        assert project.id[:8] in client.get("/v2/?sent=yes").text
        assert project.id[:8] not in client.get("/v2/?sent=no").text

    def test_legacy_caption_demo_export_counts_as_sent(self, client, db_session):
        """The Caption Demo tab is retired, but its rows DID reach a sheet."""
        project, run = self._pair(db_session)
        run.caption_demo_status = "queued"
        db_session.commit()

        assert project.id[:8] in client.get("/v2/?sent=yes").text
        assert project.id[:8] not in client.get("/v2/?sent=no").text

    def test_exported_tail_edit_counts_as_sent(self, client, db_session):
        project, run = self._pair(db_session)
        _make_tail_edit(
            db_session, run.id, final_creatives_exported_at=datetime.now(timezone.utc)
        )
        assert project.id[:8] in client.get("/v2/?sent=yes").text
        assert project.id[:8] not in client.get("/v2/?sent=no").text

    def test_unexported_tail_edit_leaves_the_run_unsent(self, client, db_session):
        project, run = self._pair(db_session)
        _make_tail_edit(db_session, run.id)
        assert project.id[:8] in client.get("/v2/?sent=no").text

    def test_exported_caption_counts_as_sent(self, client, db_session):
        project, run = self._pair(db_session)
        _make_caption(
            db_session, run.id, final_creatives_exported_at=datetime.now(timezone.utc)
        )
        assert project.id[:8] in client.get("/v2/?sent=yes").text
        assert project.id[:8] not in client.get("/v2/?sent=no").text

    def test_exported_overlay_counts_as_sent(self, client, db_session):
        project, run = self._pair(db_session)
        _make_overlay_job(
            db_session,
            source_type="run",
            source_run_id=run.id,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        assert project.id[:8] in client.get("/v2/?sent=yes").text
        assert project.id[:8] not in client.get("/v2/?sent=no").text

    def test_exported_overlay_from_a_tail_edit_counts_as_sent(
        self, client, db_session
    ):
        project, run = self._pair(db_session)
        edit = _make_tail_edit(db_session, run.id)
        _make_overlay_job(
            db_session,
            source_type="tail_edit",
            source_tail_edit_id=edit.id,
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        assert project.id[:8] in client.get("/v2/?sent=yes").text
        assert project.id[:8] not in client.get("/v2/?sent=no").text

    def test_soft_deleted_overlay_does_not_count(self, client, db_session):
        """Same rule as the Sent column: a burn whose file is gone is not an export."""
        project, run = self._pair(db_session)
        _make_overlay_job(
            db_session,
            source_type="run",
            source_run_id=run.id,
            final_creatives_exported_at=datetime.now(timezone.utc),
            deleted_at=datetime.now(timezone.utc),
        )
        assert project.id[:8] in client.get("/v2/?sent=no").text
        assert project.id[:8] not in client.get("/v2/?sent=yes").text

    def test_not_sent_lists_the_project_when_only_ONE_run_is_outstanding(
        self, client, db_session
    ):
        project = _make_project(db_session, name="Mixed")
        _make_run(
            db_session, project.id, name="gone out",
            final_creatives_exported_at=datetime.now(timezone.utc),
        )
        left = _make_run(db_session, project.id, name="still here")

        html = client.get("/v2/?sent=no").text
        assert project.id[:8] in html
        # ...and the pivot shows ONLY the outstanding run.
        assert left.id in html

    def test_sent_column_marks_the_exported_run_in_the_pivot(
        self, client, db_session
    ):
        project, run = self._pair(db_session)
        run.final_creatives_exported_at = datetime.now(timezone.utc)
        db_session.commit()

        html = client.get(f"/v2/projects/{project.id}/runs-fragment").text
        assert ">sent<" in html


class TestDashboardPivotFiltering:
    """A filtered dashboard renders the matching runs inline, already expanded."""

    def test_matching_runs_are_prerendered_and_expanded(self, client, db_session):
        project = _make_project(db_session, name="Pivot")
        wanted = _make_run(db_session, project.id, name="Wanted", resolution="1080p")
        other = _make_run(db_session, project.id, name="Other", resolution="480p")

        html = client.get("/v2/?resolution=1080p").text
        assert wanted.id in html
        assert other.id not in html
        # Rendered inline, not behind a "Loading…" placeholder.
        assert f'id="runs-cell-{project.id}" data-loaded="1"' in html

    def test_unfiltered_pivot_stays_lazy(self, client, db_session):
        project = _make_project(db_session, name="Lazy")
        run = _make_run(db_session, project.id, name="Hidden")

        html = client.get("/v2/").text
        assert f'id="runs-cell-{project.id}" data-loaded="0"' in html
        assert run.id not in html

    def test_type_only_filter_leaves_the_pivot_lazy(self, client, db_session):
        """Type selects PROJECTS — there is no matched-run set to expand."""
        project = _make_project(db_session, name="Typed", project_type="localisation")
        _make_run(db_session, project.id)

        html = client.get("/v2/?project_type=localisation").text
        assert project.id[:8] in html
        assert f'id="runs-cell-{project.id}" data-loaded="0"' in html

    def test_lazy_pivot_is_fetched_with_the_active_filters(self, client, db_session):
        """The page hands its own filter query string to the HTMX fragment."""
        _make_project(db_session, name="Any", project_type="localisation")
        html = client.get("/v2/?project_type=localisation&lang=ja").text
        assert 'var ACTIVE_RUN_FILTERS = "lang=ja"' in html


# ---------------------------------------------------------------------------
# Credit balance header — GET /v2/credits-fragment
# ---------------------------------------------------------------------------


class TestCreditBalance:
    def test_credits_fragment_shows_balance(self, client, monkeypatch):
        import app.web_v2 as web_v2

        monkeypatch.setattr(web_v2, "_cached_credit_balance", lambda: 1234.0)
        resp = client.get("/v2/credits-fragment")
        assert resp.status_code == 200
        assert "credits" in resp.text
        assert "1,234" in resp.text  # thousands-formatted

    def test_credits_fragment_degrades_on_error(self, client, monkeypatch):
        import app.web_v2 as web_v2

        def _boom():
            raise RuntimeError("kie down")

        monkeypatch.setattr(web_v2, "_cached_credit_balance", _boom)
        resp = client.get("/v2/credits-fragment")
        assert resp.status_code == 200
        assert "n/a" in resp.text

    def test_dashboard_header_polls_credits(self, client):
        resp = client.get("/v2/")
        assert 'hx-get="/v2/credits-fragment"' in resp.text


# ---------------------------------------------------------------------------
# Per-task cost display on the run detail page
# ---------------------------------------------------------------------------


class TestRunCostDisplay:
    def test_segment_cost_and_run_total_shown(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd0 = _make_segment_def(db_session, p.id, index=0)
        sd1 = _make_segment_def(db_session, p.id, index=1)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd0.id,
            status=SegmentStatus.completed, cost_credits=30.0,
        )
        _make_run_segment(
            db_session, r.id, index=1, segment_def_id=sd1.id,
            status=SegmentStatus.completed, cost_credits=20.0,
        )
        resp = client.get(f"/v2/runs/{r.id}")
        html = resp.text
        assert "credits" in html
        # Per-segment costs and the run total (30 + 20 = 50).
        assert "30 credits" in html
        assert "Total cost: 50 credits" in html

    def test_no_cost_badge_when_uncosted(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd0 = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd0.id,
            status=SegmentStatus.pending,
        )
        resp = client.get(f"/v2/runs/{r.id}")
        assert "Total cost:" not in resp.text


# ---------------------------------------------------------------------------
# Launch windows — a run's segment timings must survive later segment edits
# ---------------------------------------------------------------------------


class TestRunSegmentLaunchWindow:
    """The run page reports the window a segment RAN on, not where its def sits.

    Regression guard for run fdc49979 (2026-08-12), whose page showed
    "34.6s – 56.1s" for a swap actually generated at 8.6s – 17.3s: a mid-run
    segment edit had repurposed that SegmentDef as the project's keep segment,
    and the page read start/end straight off the live def.
    """

    def _run_with_edited_def(self, db_session):
        """A run frozen at 8.625–17.25 whose def was later re-cut to 34.6–56.1."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(
            db_session, p.id, index=1, start_sec=8.625, end_sec=17.25,
        )
        r = _make_run(
            db_session, p.id, status=RunStatus.incomplete,
            segment_snapshot=[
                {
                    "id": sd.id, "index": 1, "action": "swap",
                    "start_sec": 8.625, "end_sec": 17.25,
                },
            ],
        )
        _make_run_segment(
            db_session, r.id, index=1, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )
        # The edit that broke fdc49979: same row, repurposed.
        sd.start_sec, sd.end_sec, sd.action = 34.6, 56.121905, "keep"
        db_session.commit()
        return p, r, sd

    def test_shows_frozen_window_not_live_def(self, client, db_session):
        _p, r, _sd = self._run_with_edited_def(db_session)

        html = client.get(f"/v2/runs/{r.id}").text
        # "%.1f" rounds half to even, so the frozen 17.25 renders as 17.2.
        assert "8.6s &ndash; 17.2s" in html
        assert "34.6s" not in html
        assert "56.1s" not in html

    def test_thumbnail_uses_frozen_start(self, client, db_session):
        _p, r, _sd = self._run_with_edited_def(db_session)

        html = client.get(f"/v2/runs/{r.id}").text
        assert "frame?t=8.6" in html
        assert "frame?t=34.6" not in html

    def test_status_fragment_agrees_with_the_page(self, client, db_session):
        """The HTMX poll swaps this in — it must not re-introduce the live def."""
        _p, r, _sd = self._run_with_edited_def(db_session)

        html = client.get(f"/v2/runs/{r.id}/status-fragment").text
        assert "8.6s &ndash; 17.2s" in html
        assert "34.6s" not in html

    def test_generations_export_uses_frozen_window(self, client, db_session):
        """The table no longer shows a Segment column; the CSV still does, and
        it is built from the same rows — so the guard lives there now."""
        self._run_with_edited_def(db_session)

        body = client.get("/v2/generations.csv").text
        assert "8.6–17.2s" in body
        assert "34.6–56.1s" not in body

    def test_pre_snapshot_run_falls_back_to_live_def(self, client, db_session):
        """Runs predating segment_snapshot keep the old behaviour."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(
            db_session, p.id, index=0, start_sec=2.0, end_sec=7.5,
        )
        r = _make_run(db_session, p.id, status=RunStatus.done, segment_snapshot=None)
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )

        html = client.get(f"/v2/runs/{r.id}").text
        assert "2.0s &ndash; 7.5s" in html

    def test_no_window_when_neither_source_has_the_def(self, db_session):
        """Neither snapshot nor live def → no range, rather than a guess.

        Unit-level: the FK cascade means a RunSegment outliving its SegmentDef
        is unreachable through the API, so this branch cannot be built from a
        fixture — but the template still guards on it, so it is worth pinning.
        """
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.incomplete)
        orphan = RunSegment(
            run_id=r.id, segment_def_id="deleted-def-id",
            index=0, status=SegmentStatus.pending,
        )

        assert web_v2._launch_window(r, orphan) == (None, None)

    def test_window_reads_snapshot_before_live_def(self, db_session):
        """Unit-level statement of the precedence the page depends on."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(
            db_session, p.id, index=0, start_sec=40.0, end_sec=50.0,
        )
        r = _make_run(
            db_session, p.id,
            segment_snapshot=[{"id": sd.id, "index": 0, "action": "swap",
                              "start_sec": 1.0, "end_sec": 2.0}],
        )
        rs = _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )

        assert web_v2._launch_window(r, rs) == (1.0, 2.0)
        # ...and without a snapshot, the live def is the fallback.
        r.segment_snapshot = None
        assert web_v2._launch_window(r, rs) == (40.0, 50.0)


# ---------------------------------------------------------------------------
# Per-segment GPT Image 2 regeneration block
# ---------------------------------------------------------------------------


class TestSegmentImageRegenerationUI:
    def _run_with_segment(self, db_session, *, preprocess_enabled, rs_status):
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0, start_sec=10.0, end_sec=20.0)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            preprocess_enabled=preprocess_enabled,
            reference_image_urls=["/data/refs/character.jpg"],
            segment_snapshot=[
                {"id": sd.id, "index": 0, "action": "swap", "start_sec": 10.0, "end_sec": 20.0},
            ],
        )
        rs = _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id, status=rs_status,
        )
        return p, r, sd, rs

    def test_hidden_when_preprocess_disabled(self, client, db_session):
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=False, rs_status=SegmentStatus.completed,
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert f'id="segimg-{rs.id}"' not in html
        assert "Generated image (GPT Image)" not in html

    def test_shown_for_completed_segment_when_preprocess_enabled(self, client, db_session):
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=True, rs_status=SegmentStatus.completed,
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert f'id="segimg-{rs.id}"' in html
        assert "Generated image (GPT Image)" in html

    def test_shown_for_failed_segment_when_preprocess_enabled(self, client, db_session):
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=True, rs_status=SegmentStatus.failed,
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert f'id="segimg-{rs.id}"' in html

    def test_hidden_for_pending_segment_even_when_preprocess_enabled(self, client, db_session):
        """Only completed/failed segments get the block — pending/generating
        ones have nothing to rescue yet."""
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=True, rs_status=SegmentStatus.pending,
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert f'id="segimg-{rs.id}"' not in html

    def test_frame_sec_defaults_to_the_window_midpoint(self, client, db_session):
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=True, rs_status=SegmentStatus.completed,
        )
        html = client.get(f"/v2/runs/{r.id}").text
        # window is [10.0, 20.0] -> midpoint 15.0, NOT window_start (10.0).
        assert 'name="frame_sec"' in html
        assert 'value="15.0"' in html

    def test_status_fragment_carries_the_block_too(self, client, db_session):
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=True, rs_status=SegmentStatus.completed,
        )
        html = client.get(f"/v2/runs/{r.id}/status-fragment").text
        assert f'id="segimg-{rs.id}"' in html

    def test_override_url_box_carries_no_local_disk_paths(self, client, db_session):
        """Regression guard: a local worker-disk override path must never be
        shown as editable text — only http(s) entries belong in the box. A
        local entry is a thumbnail (served through the new override-reference
        endpoint) plus a hidden input, never raw text."""
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=False, rs_status=SegmentStatus.completed,
        )
        rs.reference_image_urls_override = [
            "/data/projects/x/runs/y/segment_refs/z/local.jpg",
            "https://example.com/ref.jpg",
        ]
        db_session.commit()

        html = client.get(f"/v2/runs/{r.id}").text
        # The local path survives (round-tripped via a hidden input, not the
        # visible text box) so a plain Save doesn't drop it...
        assert 'class="segref-local-hidden" value="/data/projects/x/runs/y/segment_refs/z/local.jpg"' in html
        assert f'/api/v2/runs/{r.id}/segments/{rs.id}/override-reference?index=0' in html
        # ...but the EDITABLE text box carries only the http(s) entry.
        assert 'value="https://example.com/ref.jpg"' in html
        m = re.search(r'name="reference_urls"[^>]*value="([^"]*)"', html)
        assert m is not None
        assert "/data/projects/x/runs/y/segment_refs/z/local.jpg" not in m.group(1)

    def test_cost_badge_sums_video_and_image_credits(self, client, db_session):
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=True, rs_status=SegmentStatus.completed,
        )
        rs.cost_credits = 10.0
        db_session.add(
            RunSegmentImageJob(
                run_id=r.id, run_segment_id=rs.id, prompt="p",
                frame_sec=15.0, status=SegmentImageStatus.done,
                output_path="/data/x/result.png", cost_credits=4.0,
            )
        )
        db_session.commit()

        html = client.get(f"/v2/runs/{r.id}").text
        assert "Total cost: 14 credits" in html
        assert "10 video" in html
        assert "4 image" in html

    def test_segment_image_jobs_fragment_endpoint(self, client, db_session):
        """The dedicated self-poll fragment renders without error and carries
        the oob wrapper for a segment with an active job."""
        _p, r, _sd, rs = self._run_with_segment(
            db_session, preprocess_enabled=True, rs_status=SegmentStatus.completed,
        )
        db_session.add(
            RunSegmentImageJob(
                run_id=r.id, run_segment_id=rs.id, prompt="p",
                frame_sec=15.0, status=SegmentImageStatus.generating,
            )
        )
        db_session.commit()

        resp = client.get(f"/v2/runs/{r.id}/segment-image-jobs-fragment")
        assert resp.status_code == 200
        html = resp.text
        assert 'id="segimg-jobs-trigger"' in html
        assert "hx-trigger" in html  # still active -> keeps polling
        assert f'id="segimg-{rs.id}"' in html
        assert 'hx-swap-oob="true"' in html


# ---------------------------------------------------------------------------
# Generations dashboard — GET /v2/generations
# ---------------------------------------------------------------------------


class TestGenerationsDashboard:
    def test_page_200(self, client):
        resp = client.get("/v2/generations")
        assert resp.status_code == 200
        assert "Latest Generations" in resp.text

    def test_lists_segment_with_all_columns(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready, name="Gen Proj")
        sd = _make_segment_def(db_session, p.id, index=0, start_sec=2.0, end_sec=7.5)
        r = _make_run(
            db_session, p.id, name="Gen Run",
            model="gemini-omni", resolution="720p", status=RunStatus.processing,
        )
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.generating,
        )
        resp = client.get("/v2/generations")
        assert resp.status_code == 200
        html = resp.text
        assert "Gen Proj" in html                 # project (linked)
        assert "Gen Run" in html                  # run (linked)
        assert f"/v2/runs/{r.id}" in html
        assert f"/v2/projects/{p.id}" in html
        assert "badge-generating" in html         # status badge
        assert "Gemini Omni" in html              # model label
        assert "720p" in html                     # resolution
        # The per-segment index/window is CSV-only now — not in the table.
        assert "<th>Segment</th>" not in html
        assert "2.0–7.5s" not in html

    def test_shows_run_id_even_when_named(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready, name="ID Test Proj")
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, name="ID Test Run", status=RunStatus.done)
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )
        resp = client.get("/v2/generations")
        assert resp.status_code == 200
        html = resp.text
        # Even though the run has a name, the short ID should be visible
        assert "ID Test Run" in html               # run name (linked)
        assert r.id[:8] in html                    # short run id always shown
        assert f'title="{r.id}"' in html           # full id in title for copy

    def test_run_id_column_shows_the_language_suffix(self, client, db_session):
        """The id column shows what the sheet gets — '<prefix>-ja', not '<prefix>…'."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, status=RunStatus.done, language="ja")
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )
        resp = client.get("/v2/generations")
        assert resp.status_code == 200
        assert f"{r.id[:8]}-ja" in resp.text

    def test_csv_export_carries_the_job_id_column(self, client, db_session):
        """The CSV pairs the full uuid with the language-suffixed sheet Job ID."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, status=RunStatus.done, language="ja")
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )
        resp = client.get("/v2/generations.csv")
        assert resp.status_code == 200
        body = resp.text
        assert "Job ID" in body.splitlines()[0]
        assert f"{r.id},{r.id[:8]}-ja" in body

    def test_csv_export_neutralises_formula_cells(self, client, db_session):
        """AICP-160: an operator-typed name starting with = + - @ must not be a
        live formula when the CSV is opened in Excel/Sheets."""
        import csv as _csv
        import io as _io

        p = _make_project(db_session, status=ProjectStatus.ready, name="=IMPORTXML(\"http://x\",\"//a\")")
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, status=RunStatus.done, name="@SUM(A1)")
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )
        rows = list(_csv.DictReader(_io.StringIO(client.get("/v2/generations.csv").text)))
        mine = [x for x in rows if x["Run ID"] == r.id]
        assert mine
        assert mine[0]["Project Name"].startswith("'=IMPORTXML")
        assert mine[0]["Run Name"] == "'@SUM(A1)"
        assert mine[0]["Job ID"] == r.id[:8]  # ordinary cells unchanged

    def test_status_filter_exact_match(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        done_run = _make_run(db_session, p.id, name="FilterDone")
        pend_run = _make_run(db_session, p.id, name="FilterPending")
        _make_run_segment(
            db_session, done_run.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )
        _make_run_segment(
            db_session, pend_run.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.pending,
        )
        resp = client.get("/v2/generations?status=completed")
        assert resp.status_code == 200
        assert "FilterDone" in resp.text
        assert "FilterPending" not in resp.text

    def test_unknown_status_filter_ignored(self, client):
        resp = client.get("/v2/generations?status=bogus")
        assert resp.status_code == 200

    def test_size_mb_shown_for_existing_result_file(self, client, db_session, tmp_path):
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, name="Sized Run", status=RunStatus.done)
        result = tmp_path / "seg_result.mp4"
        result.write_bytes(b"\0" * int(2.5 * 1024 * 1024))
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed, local_result_path=str(result),
        )
        resp = client.get("/v2/generations")
        assert "2.5" in resp.text

    def test_size_placeholder_when_no_result_file(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, name="NoFile Run", status=RunStatus.processing)
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.pending,
            local_result_path="/nonexistent/path.mp4",
        )
        resp = client.get("/v2/generations?status=pending")
        assert resp.status_code == 200
        assert "NoFile Run" in resp.text

    def test_clip_column_links_to_the_segment_video(self, client, db_session, tmp_path):
        """A generation with a file on disk gets a link to that exact clip."""
        clip = tmp_path / "result_0000.mp4"
        clip.write_bytes(b"\x00CLIP")
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed, local_result_path=str(clip),
        )
        resp = client.get("/v2/generations")
        assert resp.status_code == 200
        assert f"/api/v2/runs/{r.id}/segments/{rs.id}/result?v=" in resp.text

    def test_clip_column_empty_when_nothing_on_disk(self, client, db_session):
        """A segment still generating has no clip link — not a broken one."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, status=RunStatus.processing)
        rs = _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.generating,
        )
        resp = client.get("/v2/generations")
        assert resp.status_code == 200
        assert f"/segments/{rs.id}/result" not in resp.text

    def test_header_nav_links_to_generations(self, client):
        resp = client.get("/v2/")
        assert 'href="/v2/generations"' in resp.text


# ---------------------------------------------------------------------------
# Stitch library page + stitching controls
# ---------------------------------------------------------------------------


class TestLibraryPage:
    def test_library_page_renders_empty(self, client):
        response = client.get("/v2/library")
        assert response.status_code == 200
        # Page now covers two reusable-asset sections (stitch clips + overlay
        # PNG cards) — the old single-purpose "Stitch library" heading was
        # replaced with a generic "Library" title and per-section subheadings.
        assert "<h2>Library</h2>" in response.text
        assert "Stitch clips" in response.text
        assert "Overlay PNG cards" in response.text

    def test_library_page_lists_clips(self, client, db_session):
        from app.models import LibraryClip

        clip = LibraryClip(
            name="Outro EN v2", local_path="/tmp/nowhere.mp4", duration_sec=4.2
        )
        db_session.add(clip)
        db_session.commit()

        response = client.get("/v2/library")
        assert response.status_code == 200
        assert "Outro EN v2" in response.text

    def test_library_page_lists_overlay_assets(self, client, db_session):
        from app.models import OverlayAsset

        asset = OverlayAsset(
            name="Logo bug — bottom left", local_path="/tmp/nowhere.png",
            width=200, height=80,
        )
        db_session.add(asset)
        db_session.commit()

        response = client.get("/v2/library")
        assert response.status_code == 200
        assert "Logo bug — bottom left" in response.text

    def test_project_page_shows_stitching_controls(self, client, db_session):
        from app.models import LibraryClip

        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")
        clip = LibraryClip(
            name="Outro RU", local_path="/tmp/nowhere.mp4", duration_sec=4.2
        )
        db_session.add(clip)
        db_session.commit()

        response = client.get(f"/v2/projects/{project.id}")
        assert response.status_code == 200
        assert "Stitching (optional)" in response.text
        assert "Outro RU" in response.text
        assert "stitch_mode" in response.text

    def test_run_detail_shows_custom_plan_summary(self, client, db_session):
        from app.models import LibraryClip

        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0, action="swap")
        clip = LibraryClip(
            name="Outro DE", local_path="/tmp/nowhere.mp4", duration_sec=4.2
        )
        db_session.add(clip)
        db_session.commit()
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            stitch_plan=[
                {"kind": "segment", "segment_def_id": seg.id},
                {"kind": "library", "clip_id": clip.id},
            ],
        )

        response = client.get(f"/v2/runs/{run.id}")
        assert response.status_code == 200
        assert "Custom stitch plan" in response.text
        assert "Outro DE" in response.text

    def test_run_detail_shows_no_stitch_summary(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, stitch_plan=[]
        )
        response = client.get(f"/v2/runs/{run.id}")
        assert response.status_code == 200
        assert "Stitching disabled" in response.text


# ---------------------------------------------------------------------------
# Project types in the UI
# ---------------------------------------------------------------------------


class TestProjectTypeUI:
    def test_dashboard_offers_every_type(self, client):
        html = client.get("/v2/").text
        assert 'name="project_type"' in html
        assert 'value="face_swap"' in html
        assert 'value="subtitle_removal"' in html
        assert 'value="cover_change"' in html
        assert "Removing subtitles" in html
        assert "Change cover" in html

    def test_dashboard_lists_project_type(self, client, db_session):
        _make_project(db_session, name="Sub job", project_type="subtitle_removal")
        html = client.get("/v2/").text
        assert "Removing subtitles" in html

    def test_project_detail_shows_type(self, client, db_session):
        p = _make_project(db_session, project_type="cover_change")
        html = client.get(f"/v2/projects/{p.id}").text
        assert "Change cover" in html

    def test_new_run_form_prefills_subtitle_defaults(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready,
                          project_type="subtitle_removal")
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        # Gemini Omni is pre-selected and the subtitle prompt is in the textarea.
        assert '<option value="gemini-omni" selected>' in html
        assert "Remove all subtitles, captions and overlay text" in html
        # Reference images are called out as unnecessary for this flow.
        assert "Not needed for" in html
        # The scene-cut editor hint explains the auto-cut behaviour.
        assert "auto-cut this video on scene changes" in html

    def test_new_run_form_prefills_cover_defaults(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready,
                          project_type="cover_change")
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert '<option value="seedance-fast" selected>' in html
        assert "The handheld book is the target object." in html

    def test_new_run_form_keeps_face_swap_defaults(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert '<option value="seedance" selected>' in html
        assert "Replace the main person in the reference video" in html

    def test_status_fragment_uses_the_same_defaults(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready,
                          project_type="subtitle_removal")
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}/status-fragment").text

        assert '<option value="gemini-omni" selected>' in html
        assert "Remove all subtitles, captions and overlay text" in html


# ---------------------------------------------------------------------------
# "Move to new project" panel
# ---------------------------------------------------------------------------


class TestMoveRunToProjectUI:
    def _done_run(self, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")
        p = _make_project(db_session, status=ProjectStatus.ready)
        return _make_run(
            db_session, p.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

    def test_done_run_shows_panel(self, client, db_session, tmp_path):
        run = self._done_run(db_session, tmp_path)
        html = client.get(f"/v2/runs/{run.id}").text
        assert "Move to new project" in html
        assert f"moveRunToProject('{run.id}')" in html
        assert 'id="move-project-type"' in html
        assert 'value="subtitle_removal"' in html

    def test_status_fragment_shows_panel(self, client, db_session, tmp_path):
        run = self._done_run(db_session, tmp_path)
        html = client.get(f"/v2/runs/{run.id}/status-fragment").text
        assert "Move to new project" in html
        assert 'id="move-project-type"' in html

    def test_unfinished_run_has_no_panel(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, p.id, status=RunStatus.processing)
        html = client.get(f"/v2/runs/{run.id}").text
        assert "Move to new project" not in html

    def test_project_shows_source_run_backlink(self, client, db_session, tmp_path):
        run = self._done_run(db_session, tmp_path)
        moved = _make_project(
            db_session,
            name="Redhead woman → Removing subtitles",
            project_type="subtitle_removal",
            source_run_id=run.id,
        )
        html = client.get(f"/v2/projects/{moved.id}").text
        assert "Created from run" in html
        assert f"/v2/runs/{run.id}" in html

    def test_backlink_omitted_when_source_run_is_gone(self, client, db_session):
        moved = _make_project(db_session, source_run_id="deleted-run-id")
        html = client.get(f"/v2/projects/{moved.id}").text
        assert "Created from run" not in html

    # -- AICP-331: from a caption / tail edit / overlay of the run ----------

    def _steps(self, db_session, tmp_path, run):
        out = tmp_path / "step.mp4"
        out.write_bytes(b"step")
        cap = _make_caption(db_session, run.id, status=CaptionStatus.done, output_path=str(out))
        edit = _make_tail_edit(db_session, run.id, status=TailEditStatus.done,
                               output_path=str(out), cut_sec=7.5)
        job = _make_overlay_job(db_session, source_run_id=run.id, status=OverlayStatus.done,
                                result_local_path=str(out))
        return cap, edit, job

    def test_done_steps_are_offered_as_sources(self, client, db_session, tmp_path):
        run = self._done_run(db_session, tmp_path)
        cap, edit, job = self._steps(db_session, tmp_path, run)
        html = client.get(f"/v2/runs/{run.id}").text
        assert 'id="move-project-source"' in html
        assert 'id="move-to-project"' in html
        assert '<option value="run:" selected>Run final video</option>' in html
        for value in (f"caption:{cap.id}", f"tail_edit:{edit.id}", f"overlay:{job.id}"):
            assert f'value="{value}"' in html
        # Each done row has its own "New project" action and a stable anchor.
        assert f'data-move-kind="caption" data-move-id="{cap.id}"' in html
        assert f'data-move-kind="tail_edit" data-move-id="{edit.id}"' in html
        assert f'data-move-kind="overlay" data-move-id="{job.id}"' in html
        for anchor in (f'id="caption-{cap.id}"', f'id="tail-edit-{edit.id}"',
                       f'id="overlay-{job.id}"'):
            assert anchor in html
        assert "fd.append('source_kind', src[0])" in html

    def test_unfinished_steps_are_not_offered(self, client, db_session, tmp_path):
        run = self._done_run(db_session, tmp_path)
        cap = _make_caption(db_session, run.id, status=CaptionStatus.burning)
        job = _make_overlay_job(db_session, source_run_id=run.id, status=OverlayStatus.processing)
        html = client.get(f"/v2/runs/{run.id}").text
        assert f'value="caption:{cap.id}"' not in html
        assert f'value="overlay:{job.id}"' not in html
        assert f'data-move-id="{cap.id}"' not in html

    def test_overlay_editor_links_to_the_move_card(self, client, db_session, tmp_path):
        run = self._done_run(db_session, tmp_path)
        _, _, job = self._steps(db_session, tmp_path, run)
        html = client.get(f"/v2/overlays/{job.id}").text
        assert f"/v2/runs/{run.id}?move_source=overlay:{job.id}#move-to-project" in html

    @pytest.mark.parametrize("kind", ["caption", "tail_edit", "overlay"])
    def test_banner_names_and_links_the_step(self, client, db_session, tmp_path, kind):
        run = self._done_run(db_session, tmp_path)
        cap, edit, job = self._steps(db_session, tmp_path, run)
        step = {"caption": cap, "tail_edit": edit, "overlay": job}[kind]
        moved = _make_project(db_session, source_run_id=run.id, source_kind=kind,
                              source_artifact_id=step.id)
        html = client.get(f"/v2/projects/{moved.id}").text
        url = {"caption": f"/v2/runs/{run.id}#caption-{step.id}",
               "tail_edit": f"/v2/runs/{run.id}#tail-edit-{step.id}",
               "overlay": f"/v2/overlays/{step.id}"}[kind]
        label = {"caption": "captions", "tail_edit": "tail edit", "overlay": "overlay"}[kind]
        assert 'id="source-step-banner"' in html
        assert f'href="{url}"' in html
        assert label in html
        assert f"/v2/runs/{run.id}" in html

    @pytest.mark.parametrize("kind", ["caption", "tail_edit", "overlay"])
    def test_deleted_step_degrades_to_run_backlink(self, client, db_session, tmp_path, kind):
        from datetime import datetime, timezone

        run = self._done_run(db_session, tmp_path)
        cap, edit, job = self._steps(db_session, tmp_path, run)
        step = {"caption": cap, "tail_edit": edit, "overlay": job}[kind]
        moved = _make_project(db_session, source_run_id=run.id, source_kind=kind,
                              source_artifact_id=step.id)
        if kind == "overlay":
            job.deleted_at = datetime.now(timezone.utc)
        else:
            db_session.delete(step)
        db_session.commit()
        html = client.get(f"/v2/projects/{moved.id}").text
        assert 'id="source-step-banner"' not in html
        assert "Created from run" in html
        assert "(since deleted)" in html
        assert f"/v2/runs/{run.id}" in html
        # And the projects list keeps its Source cell (the run's short id).
        dash = client.get("/v2/").text
        assert run.id[:8] in dash

    def test_step_of_another_run_is_not_linked(self, client, db_session, tmp_path):
        run = self._done_run(db_session, tmp_path)
        other = self._done_run(db_session, tmp_path)
        out = tmp_path / "o.mp4"
        out.write_bytes(b"o")
        foreign = _make_caption(db_session, other.id, status=CaptionStatus.done,
                                output_path=str(out))
        moved = _make_project(db_session, source_run_id=run.id, source_kind="caption",
                              source_artifact_id=foreign.id)
        html = client.get(f"/v2/projects/{moved.id}").text
        assert 'id="source-step-banner"' not in html
        assert f"#caption-{foreign.id}" not in html


# ---------------------------------------------------------------------------
# Tail edit context function tests — _get_tail_edit_context
# ---------------------------------------------------------------------------


class TestFloor2:
    def test_floor2_handles_the_margin_subtraction_float_gap(self):
        """12.28 - 0.05 == 12.229999999999999 in IEEE754 double precision —
        a naive math.floor(x*100)/100 (or Jinja's own `round(2, 'floor')`)
        floors that DOWN to 12.22, one cent below the mathematically correct
        12.23. This is exactly the "Max" button's computation, so getting it
        wrong silently under-fills the button by a cent."""
        from app.web_v2 import _floor2

        assert _floor2(12.28 - 0.05) == 12.23
        assert _floor2(12.28) == 12.28

    def test_floor2_still_floors_down_for_a_genuine_fraction(self):
        """The epsilon must not mask a real fractional-cent difference the
        other direction."""
        from app.web_v2 import _floor2

        assert _floor2(12.229) == 12.22
        assert _floor2(12.221) == 12.22


class TestTailEditDurationBackfill:
    def test_self_heal_backfill_probes_and_commits(self, db_session, tmp_path, monkeypatch):
        """Done run with result_duration_sec=None: probes the file, backfills, commits."""
        from app.web_v2 import _get_tail_edit_context

        # Create a project and run
        project = _make_project(db_session)
        dummy_file = tmp_path / "result.mp4"
        dummy_file.write_bytes(b"fake video data")

        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(dummy_file),
            result_duration_sec=None,
        )

        # Mock media.probe to return a stub with duration_sec
        import app.media
        monkeypatch.setattr(
            app.media,
            "probe",
            lambda path: types.SimpleNamespace(duration_sec=12.5),
        )

        # Call the function
        context = _get_tail_edit_context(run, db_session)

        # Assert the returned context has the probed duration
        assert context["tail_edit_result_duration_sec"] == 12.5

        # Assert the run object was updated in-memory
        assert run.result_duration_sec == 12.5

        # Assert the value persisted to the database
        refreshed = db_session.get(Run, run.id)
        assert refreshed.result_duration_sec == 12.5

    def test_fast_path_no_probe_when_duration_cached(self, db_session, monkeypatch):
        """Done run with result_duration_sec already set: no probe, fast path."""
        from app.web_v2 import _get_tail_edit_context

        project = _make_project(db_session)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_duration_sec=7.0,
        )

        # Mock probe to raise if called—it shouldn't be
        def probe_should_not_be_called(path):
            raise AssertionError("probe should not be called for cached duration")

        import app.media
        monkeypatch.setattr(app.media, "probe", probe_should_not_be_called)

        # Call the function
        context = _get_tail_edit_context(run, db_session)

        # Assert the cached value is returned
        assert context["tail_edit_result_duration_sec"] == 7.0

    def test_non_done_status_no_probe_no_write(self, db_session, monkeypatch):
        """Non-done run: no probe, no DB write, empty context."""
        from app.web_v2 import _get_tail_edit_context

        project = _make_project(db_session)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.processing,
            result_duration_sec=None,
        )

        # Mock probe to raise if called—it shouldn't be
        def probe_should_not_be_called(path):
            raise AssertionError("probe should not be called for non-done run")

        import app.media
        monkeypatch.setattr(app.media, "probe", probe_should_not_be_called)

        # Call the function
        context = _get_tail_edit_context(run, db_session)

        # Assert empty values for non-done run
        assert context["tail_edit_result_duration_sec"] is None
        assert context["tail_edit_library_clips"] == []

        # Assert the run was not modified
        assert run.result_duration_sec is None

    def test_min_cut_margin_sec_always_present(self, db_session):
        """tail_edit_min_cut_margin_sec is a fixed constant, not something
        that depends on the run — present (and equal to the shared
        pipeline_common constant) on both the early-return and populated
        paths, so the template's max/"Max"-button math never has a hole."""
        from app.pipeline_common import TAIL_EDIT_MIN_CUT_MARGIN_SEC
        from app.web_v2 import _get_tail_edit_context

        project = _make_project(db_session)
        done_run = _make_run(
            db_session, project.id, status=RunStatus.done, result_duration_sec=10.0,
        )
        not_done_run = _make_run(db_session, project.id, status=RunStatus.processing)

        assert (
            _get_tail_edit_context(done_run, db_session)["tail_edit_min_cut_margin_sec"]
            == TAIL_EDIT_MIN_CUT_MARGIN_SEC
        )
        assert (
            _get_tail_edit_context(not_done_run, db_session)["tail_edit_min_cut_margin_sec"]
            == TAIL_EDIT_MIN_CUT_MARGIN_SEC
        )

    def test_caption_durations_probed_for_source_picker(
        self, db_session, tmp_path, monkeypatch
    ):
        """Each done caption's duration is probed and exposed keyed by its
        id, so the front end can recompute the "Cut at (s)" max when the
        operator switches Source without a round trip."""
        from app.web_v2 import _get_tail_edit_context

        project = _make_project(db_session)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_duration_sec=10.0,
        )
        cap_output = tmp_path / "captioned.mp4"
        cap_output.write_bytes(b"fake caption video")
        caption = _make_caption(
            db_session, run.id, status=CaptionStatus.done,
            output_path=str(cap_output),
        )

        import app.media
        monkeypatch.setattr(
            app.media, "probe",
            lambda path: types.SimpleNamespace(duration_sec=4.2),
        )

        context = _get_tail_edit_context(run, db_session)
        assert context["tail_edit_source_caption_durations"] == {caption.id: 4.2}

    def test_caption_with_no_file_on_disk_omitted_from_durations(
        self, db_session
    ):
        """A done caption with no output_path (or a missing file) degrades
        gracefully — it's simply absent from the durations dict, same
        tolerance as the run's own probe, rather than 500ing the page."""
        from app.web_v2 import _get_tail_edit_context

        project = _make_project(db_session)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_duration_sec=10.0,
        )
        _make_caption(db_session, run.id, status=CaptionStatus.done)  # no output_path

        context = _get_tail_edit_context(run, db_session)
        assert context["tail_edit_source_caption_durations"] == {}


# ---------------------------------------------------------------------------
# Tests: _new_run_defaults — model narrowing against the segmentation cap
# ---------------------------------------------------------------------------
# A project's max_segment_sec decides which models can actually generate its
# segments, but the *suggested* model comes from the project type. Offering a
# pre-selected model that cannot cover the cap is a trap: the form's warning
# fires, but the default is one click from a run that bills every segment and
# delivers nothing (over-long segments fail, and the completeness gate then
# refuses to stitch). These pin the narrowing so that trap stays closed.


class TestNewRunDefaultModelRespectsCap:
    def test_null_cap_keeps_the_project_type_default(self, db_session):
        """No cap → the historical 10s universal default → nothing to narrow."""
        from app.web_v2 import _new_run_defaults
        from app.project_types import FACE_SWAP, spec_for

        project = _make_project(
            db_session, project_type=FACE_SWAP, max_segment_sec=None
        )
        ctx = _new_run_defaults(project)

        assert ctx["default_model"] == spec_for(FACE_SWAP).default_model
        assert ctx["segment_cap_sec"] == ai_models.UNIVERSAL_MAX_SEGMENT_SEC
        # Every model can do 10s, so none are flagged.
        assert ctx["models_over_cap"] == []

    def test_cap_of_10_keeps_the_project_type_default(self, db_session):
        from app.web_v2 import _new_run_defaults
        from app.project_types import FACE_SWAP, spec_for

        project = _make_project(
            db_session, project_type=FACE_SWAP, max_segment_sec=10.0
        )
        ctx = _new_run_defaults(project)
        assert ctx["default_model"] == spec_for(FACE_SWAP).default_model
        assert ctx["models_over_cap"] == []

    def test_30s_cap_narrows_face_swap_away_from_seedance_2_0(self, db_session):
        """face_swap prefers `seedance` (15s) — a 30s project must not default to it."""
        from app.web_v2 import _new_run_defaults
        from app.project_types import FACE_SWAP

        project = _make_project(
            db_session, project_type=FACE_SWAP, max_segment_sec=30.0
        )
        ctx = _new_run_defaults(project)

        assert ctx["default_model"] == "seedance-2-5"
        assert ctx["segment_cap_sec"] == 30.0
        # Everything except 2.5 is flagged as unable to honour the cap.
        assert set(ctx["models_over_cap"]) == {
            "seedance", "seedance-fast", "seedance-mini", "gemini-omni",
        }

    def test_30s_cap_narrows_subtitle_removal_away_from_gemini(self, db_session):
        """subtitle_removal prefers gemini-omni (10s) — the worst offender."""
        from app.web_v2 import _new_run_defaults
        from app.project_types import SUBTITLE_REMOVAL

        project = _make_project(
            db_session, project_type=SUBTITLE_REMOVAL, max_segment_sec=30.0
        )
        ctx = _new_run_defaults(project)

        assert ctx["default_model"] == "seedance-2-5"
        assert "gemini-omni" in ctx["models_over_cap"]

    def test_15s_cap_excludes_only_gemini(self, db_session):
        from app.web_v2 import _new_run_defaults
        from app.project_types import SUBTITLE_REMOVAL

        project = _make_project(
            db_session, project_type=SUBTITLE_REMOVAL, max_segment_sec=15.0
        )
        ctx = _new_run_defaults(project)

        # Gemini can't do 15s, so the gemini-preferring type gets moved off it,
        # but a 15s-capable Seedance is enough — no need to jump to 2.5.
        assert ctx["models_over_cap"] == ["gemini-omni"]
        assert ctx["default_model"] != "gemini-omni"
        assert ai_models.max_clip_sec(ctx["default_model"]) >= 15.0

    def test_chosen_default_can_always_honour_the_cap(self, db_session):
        """The invariant that matters, across every type/cap combination."""
        from app.web_v2 import _new_run_defaults
        from app.project_types import PROJECT_TYPES

        for ptype in PROJECT_TYPES:
            for cap in (None, 10.0, 15.0, 30.0):
                project = _make_project(
                    db_session, project_type=ptype, max_segment_sec=cap
                )
                ctx = _new_run_defaults(project)
                effective = cap or ai_models.UNIVERSAL_MAX_SEGMENT_SEC
                assert (
                    ai_models.max_clip_sec(ctx["default_model"]) + 0.05 >= effective
                ), f"{ptype} @ cap={cap} defaulted to {ctx['default_model']!r}"
                assert ctx["default_model"] not in ctx["models_over_cap"]


# ---------------------------------------------------------------------------
# Localisation UI (docs/localisation.md §7)
#
# The type re-generates a video's hook in another language. Three things can
# destroy a localisation without anything failing — a model that emits no audio
# (create_run downgrades the run to audio_mode="original"), audio_mode
# "original" itself (it overlays the untranslated source track), and a stitch
# plan that keeps the untranslated tail — so the pre-fills below are the whole
# safety story, not cosmetics.
# ---------------------------------------------------------------------------


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
    "on_screen_text": [{"start": 0.0, "end": 3.0, "text": "LISTENIFY"}],
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
        assert ctx["default_model"] == "seedance-fast"
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
        assert ctx["default_model"] == "seedance"

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
        # Seedance 2.5 stays selectable as the longer-clip / generate_audio option.
        assert 'value="seedance-2-5"' in html
        assert '<option value="seedance-2-5" disabled>' not in html
        assert '<option value="seedance-2-5" selected>' not in html

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

    def test_30s_cap_narrows_localisation_default_to_seedance_2_5(self, db_session):
        """A 30s project cannot default to Fast (15s) — the form must fall
        through to the only model that can honour the cap."""
        from app.web_v2 import _new_run_defaults

        project = _localisation_project(db_session, max_segment_sec=30.0)
        ctx = _new_run_defaults(project)
        assert ctx["default_model"] == "seedance-2-5"
        assert "seedance-fast" in ctx["models_over_cap"]
        assert "gemini-omni" in ctx["models_without_audio"]
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

    def test_form_offers_the_voice_reference_header_toggle(self, client, db_session):
        """AICP-335: unticked by default; the header text comes from the transcript."""
        p = _localisation_project(
            db_session, transcript_status="ready", transcript=_TRANSCRIPT
        )
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text
        assert '<input type="checkbox" id="ref-audio-header" onchange="applyRefAudioHeader()" />' in html
        assert "var LOCALISATION_REF_AUDIO_HEADER = \"Speakers (the reference audio contains" in html
        # The Translate request carries the run's model (clip limit -> roll trim).
        assert "body.append('model', modelSel.value)" in html
        # A model switch after Translate warns that the timecodes are stale.
        assert "if (typeof onLocalisationModelChange === 'function') onLocalisationModelChange();" in html
        # The gate carries the live header, refreshed by the transcript poll.
        assert 'data-ref-audio-header="Speakers (the reference audio contains' in html
        frag = client.get(f"/v2/projects/{p.id}/transcript-fragment").text
        assert 'data-ref-audio-header="Speakers (the reference audio contains' in frag

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

    def test_stitch_controls_carry_the_prefill_inputs(self, client, db_session):
        """The auto-plan reads the segment action and the clip language off the
        DOM — without these attributes it silently degrades to no pre-fill."""
        from app.models import LibraryClip

        clip = LibraryClip(name="Outro JA", local_path="/tmp/ja.mp4", language="ja")
        db_session.add(clip)
        db_session.commit()

        p = _localisation_project(db_session)
        _make_segment_def(db_session, p.id, 0, action="swap")
        _make_segment_def(db_session, p.id, 1, action="keep")
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'data-action="swap"' in html
        assert 'data-action="keep"' in html
        assert 'data-lang="ja"' in html
        assert "pickLocalisationTail" in html

    def test_language_picker_lives_inside_the_localisation_panel(
        self, client, db_session
    ):
        """Language is the Translate target, so on a localisation project it is
        rendered once, inside #localisation-panel, above the Translate button —
        not as a separate group further down the form."""
        p = _localisation_project(
            db_session, transcript_status="ready", transcript=_TRANSCRIPT
        )
        _make_segment_def(db_session, p.id, 0)
        for url in (f"/v2/projects/{p.id}", f"/v2/projects/{p.id}/status-fragment"):
            html = client.get(url).text
            assert html.count('id="run-language"') == 1
            panel = html.index('id="localisation-panel"')
            lang = html.index('id="run-language"')
            assert panel < lang < html.index('id="translate-btn"')
            assert "translated into by <strong>Translate</strong>" in html
            # Starts blank (no pre-selected English) and says why Translate is off.
            assert '<option value="" selected>— choose the target language —</option>' in html
            assert '<option value="en" selected>' not in html
            assert 'id="translate-language-reason"' in html

    def test_language_picker_stays_a_form_group_on_other_types(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert html.count('id="run-language"') == 1
        assert "The spoken language of the delivered video. It changes nothing" in html
        assert "translated into by <strong>Translate</strong>" not in html
        assert '<option value="en" selected>' in html
        assert "— choose the target language —" not in html

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


class TestLocalisationCopyRunCard:
    """The run page's "Copy run" card is withdrawn for a speech-only
    localisation run: its prompt keeps the person on screen, so the reference
    photo the card asks for cannot change anything and the copy is the same
    generation again. api_v2._check_copyable enforces the same rule with a 409.
    """

    def _run(self, session, *, swap: bool, **kwargs) -> Run:
        from app import project_types

        project = _localisation_project(session)
        template = (
            project_types._LOCALISATION_SWAP_PROMPT
            if swap
            else project_types._LOCALISATION_KEEP_PROMPT
        )
        return _make_run(
            session, project.id,
            prompt=template.format(
                source_language="english", target_language="japanese",
                dialogue="**Woman:**     はい",
            ),
            model="seedance-2-5", resolution="720p", status=RunStatus.done,
            **kwargs,
        )

    def test_speech_only_run_hides_the_form_and_says_why(self, client, db_session):
        r = self._run(db_session, swap=False)
        html = client.get(f"/v2/runs/{r.id}").text

        # The heading stays (the operator is looking for it) but nothing that
        # would submit a copy is rendered.
        assert "Copy run" in html
        assert "speech-only" in html
        assert 'id="copy-rows"' not in html
        assert "copy-batch" not in html

    def test_speech_only_run_hides_the_automation_form_too(self, client, db_session):
        """Automation is Copy run on a daily quota — the same refusal, or the
        card below would offer to make the pointless copy every day."""
        r = self._run(db_session, swap=False)
        html = client.get(f"/v2/runs/{r.id}").text

        assert "Automation" in html
        assert 'id="auto-enabled"' not in html
        assert "loadAutomation(" not in html

    def test_speech_plus_character_run_keeps_the_automation_form(
        self, client, db_session
    ):
        r = self._run(db_session, swap=True)
        html = client.get(f"/v2/runs/{r.id}").text

        assert 'id="auto-enabled"' in html
        assert "loadAutomation(" in html

    def test_speech_plus_character_run_keeps_the_form(self, client, db_session):
        r = self._run(db_session, swap=True, reference_image_urls=["https://e/f.jpg"])
        html = client.get(f"/v2/runs/{r.id}").text

        assert 'id="copy-rows"' in html
        assert "copy-batch" in html
        assert "Reference photo URL (optional)" in html

    def test_face_swap_run_is_untouched(self, client, db_session):
        """Only localisation projects have modes — every other type keeps the
        card whatever its prompt says."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            prompt="Do not replace the character, just drop the subtitles.",
        )
        html = client.get(f"/v2/runs/{r.id}").text
        assert 'id="copy-rows"' in html


class TestCopyRunPreprocessControls:
    """The "Re-run GPT Image 2" controls only exist on a run that HAS a recipe
    to re-run — offering them elsewhere would just produce a 400 on submit."""

    def _preprocess_run(self, session, **kwargs):
        p = _make_project(session, status=ProjectStatus.ready)
        defaults = dict(
            status=RunStatus.done,
            reference_image_urls=[
                "https://e/char.jpg", "https://e/f1.jpg", "https://e/f2.jpg",
            ],
            preprocess_enabled=True,
            preprocess_image_prompts=["", "shot one", "shot two"],
        )
        defaults.update(kwargs)
        return _make_run(session, p.id, **defaults)

    def test_preprocess_run_offers_the_checkboxes(self, client, db_session):
        r = self._preprocess_run(db_session)
        html = client.get(f"/v2/runs/{r.id}").text

        assert 'class="copy-preprocess"' in html      # per copy row, pre-ticked
        assert 'id="auto-preprocess"' in html         # and on the automation panel
        # The hint names the frame count so the operator knows what carries over.
        assert "the same 2 frame(s)" in html

    def test_plain_run_has_no_preprocess_controls(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(
            db_session, p.id, status=RunStatus.done,
            reference_image_urls=["https://e/face.jpg"],
        )
        html = client.get(f"/v2/runs/{r.id}").text

        assert 'id="copy-rows"' in html               # the card itself is there
        assert 'class="copy-preprocess"' not in html
        assert 'id="auto-preprocess"' not in html

    def test_auto_approved_run_says_it_will_not_pause(self, client, db_session):
        """A copy mid-preprocessing must not tell the operator to expect an
        approval step that will never come."""
        r = self._preprocess_run(
            db_session,
            status=RunStatus.preprocessing,
            preprocess_auto_approve=True,
        )
        html = client.get(f"/v2/runs/{r.id}").text

        assert "approved automatically" in html
        assert "you'll be asked to approve" not in html


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


class TestLibraryClipLanguage:
    """The tail auto-pick matches on LibraryClip.language, so the library page
    has to be able to set it — otherwise every clip is NULL and the pre-fill
    can never match anything."""

    def test_upload_form_offers_a_language(self, client):
        html = client.get("/v2/library").text
        assert 'id="lib-language"' in html
        assert 'value="ja"' in html
        assert "Japanese" in html

    def test_existing_clip_shows_its_tag_and_can_be_retagged(self, client, db_session):
        from app.models import LibraryClip

        clip = LibraryClip(name="Outro JA", local_path="/tmp/ja.mp4", language="ja")
        db_session.add(clip)
        db_session.commit()

        html = client.get("/v2/library").text
        assert 'data-saved="ja"' in html
        assert '<option value="ja" selected>Japanese</option>' in html
        assert "saveClipLanguage" in html

    def test_untagged_clip_defaults_to_any(self, client, db_session):
        from app.models import LibraryClip

        clip = LibraryClip(name="Outro generic", local_path="/tmp/g.mp4")
        db_session.add(clip)
        db_session.commit()

        html = client.get("/v2/library").text
        assert 'data-saved=""' in html


# ---------------------------------------------------------------------------
# Hook length: the clamp made visible + the in-page editor + Re-analyze
# (docs/localisation.md §5, §7)
# ---------------------------------------------------------------------------


def _squash(html: str) -> str:
    """Collapse runs of whitespace, so an assertion about a SENTENCE is not a
    hostage to where Jinja happened to wrap the line."""
    return re.sub(r"\s+", " ", html)


def _hook_project(session, **kwargs) -> VideoProject:
    """A localisation project that has been analyzed (so the segment editor,
    and with it the hook report, actually renders)."""
    defaults = dict(
        status=ProjectStatus.ready,
        project_type="localisation",
        duration_sec=25.0,
        width=1080,
        height=1920,
        fps=30.0,
    )
    defaults.update(kwargs)
    return _make_project(session, **defaults)


def _hook_segments(session, project_id, hook_end, duration=25.0):
    """The rows a hook_split analysis leaves behind: swap[0, hook] + keep."""
    _make_segment_def(
        session, project_id, 0, start_sec=0.0, end_sec=hook_end,
        action="swap", has_face=False,
    )
    if duration - hook_end > 1e-6:
        _make_segment_def(
            session, project_id, 1, start_sec=hook_end, end_sec=duration,
            action="keep", has_face=False,
        )


class TestHookClampIsVisible:
    """Regression: the hook clamp was silent.

    A project created with Hook length = 15 and the default 10s segmentation
    cap was analyzed into swap[0,10] + keep[10,25] while the row still stored
    hook_sec=15. Nothing on any page said the hook asked for was not the hook
    cut, nor which bound had done it. These assert on the RENDERED page, in the
    Segment Editor — next to the segments that are the evidence.
    """

    def test_a_hook_that_was_honoured_reports_the_number_without_a_warning(
        self, client, db_session
    ):
        p = _hook_project(db_session, hook_sec=8.0)
        _hook_segments(db_session, p.id, 8.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert "Hook: 8.0s" in _squash(html)
        assert "exactly the 8.0s" in _squash(html)
        assert 'id="hook-clamp-notice"' not in html

    def test_the_segmentation_cap_clamp_is_reported_with_both_numbers(
        self, client, db_session
    ):
        """The exact bug: asked 15, got 10, cap 10."""
        p = _hook_project(db_session, hook_sec=15.0, max_segment_sec=None)
        _hook_segments(db_session, p.id, 10.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="hook-clamp-notice"' in html
        assert "Hook asked for: 15.0s" in _squash(html)
        assert "hook actually cut: 10.0s" in _squash(html)
        # ...and WHY, naming the setting that has to change.
        assert "Max segment length" in html

    def test_a_video_shorter_than_the_hook_says_so(self, client, db_session):
        p = _hook_project(
            db_session, hook_sec=20.0, max_segment_sec=30.0, duration_sec=3.1
        )
        _hook_segments(db_session, p.id, 3.1, duration=3.1)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="hook-clamp-notice"' in html
        assert "Hook asked for: 20.0s" in _squash(html)
        assert "source video is only 3.1s long" in _squash(html)
        # Not the cap's fault — do not send the operator to change a setting
        # that would change nothing.
        assert "raise the cap" not in html

    def test_a_hook_below_the_model_floor_reports_the_padding(
        self, client, db_session
    ):
        floor = min(s.min_clip_sec for s in ai_models.AI_MODELS.values())
        p = _hook_project(db_session, hook_sec=0.5)
        _hook_segments(db_session, p.id, floor)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="hook-clamp-notice"' in html
        assert "padded up to" in html

    def test_a_setting_changed_since_the_analysis_is_called_out(
        self, client, db_session
    ):
        """hook_sec is consumed at ANALYZE time, and it is now editable on this
        page — so "saved but not applied" is one keystroke away and must not
        look like a clamp."""
        p = _hook_project(db_session, hook_sec=6.0)
        _hook_segments(db_session, p.id, 8.0)  # cut under the old hook_sec=8
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="hook-clamp-notice"' in html
        assert "analyze-time settings have changed" in _squash(html)
        assert "Re-analyze video" in html

    def test_hand_edited_segments_report_no_hook_rather_than_a_wrong_one(
        self, client, db_session
    ):
        """The hook is read off the segments, so a split the operator has
        rearranged must report honestly instead of echoing hook_sec."""
        p = _hook_project(db_session, hook_sec=8.0)
        _make_segment_def(
            db_session, p.id, 0, start_sec=0.0, end_sec=8.0, action="keep"
        )
        _make_segment_def(
            db_session, p.id, 1, start_sec=8.0, end_sec=25.0, action="swap"
        )
        html = client.get(f"/v2/projects/{p.id}").text

        assert "no hook to report" in _squash(html)
        assert "edited by hand" in html

    def test_the_polled_status_fragment_carries_it_too(self, client, db_session):
        """It lives in #status-content on purpose: that is what a re-analysis
        refreshes. If it only rendered on the full page it would go stale."""
        p = _hook_project(db_session, hook_sec=15.0)
        _hook_segments(db_session, p.id, 10.0)
        html = client.get(f"/v2/projects/{p.id}/status-fragment").text

        assert 'id="hook-clamp-notice"' in html
        assert "Hook asked for: 15.0s" in _squash(html)

    def test_localisation_is_not_described_as_a_blank_slate(
        self, client, db_session
    ):
        """It used to fall into the "every non-face_swap type starts as one
        full-length keep segment" hint, which is false for hook_split."""
        p = _hook_project(db_session, hook_sec=8.0)
        _hook_segments(db_session, p.id, 8.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert "starts as one full-length" not in _squash(html)
        assert "skip face detection" not in html
        assert "cuts the front of the video into a" in _squash(html)

    def test_none_of_it_appears_on_a_non_localisation_project(
        self, client, db_session
    ):
        p = _make_project(
            db_session, status=ProjectStatus.ready, duration_sec=25.0,
            project_type="cover_change",
        )
        _make_segment_def(
            db_session, p.id, 0, start_sec=0.0, end_sec=25.0, action="keep"
        )
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="hook-clamp-notice"' not in html
        assert 'id="project-hook-sec"' not in html
        assert 'id="project-hook-clamp"' not in html
        assert "Hook asked for" not in html
        # The blank-slate hint is still the right one for this type.
        assert "skip face detection" in html


class TestHookEditor:
    """hook_sec appeared in no control on the project page — the only remedy
    for a wrong hook was deleting the project. It is an analyze-time setting,
    so it follows saveProjectMaxSegment's precedent exactly."""

    def test_the_field_is_rendered_with_the_stored_value(self, client, db_session):
        p = _hook_project(db_session, hook_sec=15.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="project-hook-sec"' in html
        assert 'value="15"' in html
        assert 'id="project-hook-sec-status"' in html
        assert "saveProjectHookSec(" in html

    def test_an_unset_hook_leaves_the_box_empty_and_shows_the_default(
        self, client, db_session
    ):
        """NULL means "no explicit choice"; pre-filling the resolved default
        would freeze today's value into the row on the next save."""
        p = _hook_project(db_session, hook_sec=None)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="project-hook-sec"' in html
        assert 'placeholder="10 (default)"' in html

    def test_saving_says_plainly_that_it_changes_nothing_until_re_analysis(
        self, client, db_session
    ):
        p = _hook_project(db_session, hook_sec=15.0)
        html = client.get(f"/v2/projects/{p.id}").text

        # In the hint...
        assert "only takes effect the next time the video is analyzed" in _squash(html)
        # ...and in the status line the save itself writes, the way
        # saveProjectMaxSegment does.
        assert "Applies on the next analysis" in html

    def test_it_patches_the_project_through_the_shared_helper(
        self, client, db_session
    ):
        p = _hook_project(db_session, hook_sec=15.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert "patchProject(pid, { hook_sec: wanted }" in html
        assert "function apiErrorText(" in client.get("/v2/").text

    def test_an_unparseable_entry_is_refused_instead_of_clearing_the_hook(
        self, client, db_session
    ):
        """JSON.stringify turns NaN into null, so a typo would silently reset
        the hook to the default — the opposite of what was typed."""
        p = _hook_project(db_session, hook_sec=15.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert "validity.badInput" in html
        assert "isFinite(wanted)" in html

    def test_the_prospective_clamp_warning_sits_next_to_the_field(
        self, client, db_session
    ):
        """The retrospective report is in the Segment Editor; this one is about
        the number currently in the box."""
        p = _hook_project(db_session, hook_sec=15.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="project-hook-clamp"' in html
        assert "The next analysis will cut a 10.0s hook, not 15.0s" in _squash(html)

    def test_no_warning_when_the_stored_hook_needs_no_clamping(
        self, client, db_session
    ):
        p = _hook_project(db_session, hook_sec=8.0)
        html = client.get(f"/v2/projects/{p.id}").text
        assert 'id="project-hook-clamp"' not in html

    def test_the_editor_renders_before_analysis_too(self, client, db_session):
        """Duration is unknown then; the cap can still clamp and the field must
        still be fixable without deleting the project."""
        p = _hook_project(
            db_session, status=ProjectStatus.analyzing, hook_sec=15.0,
            duration_sec=None,
        )
        html = client.get(f"/v2/projects/{p.id}").text

        assert 'id="project-hook-sec"' in html
        assert 'id="project-hook-clamp"' in html
        assert "Max segment length" in html

    def test_the_helper_is_not_even_defined_off_localisation(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready, duration_sec=25.0)
        html = client.get(f"/v2/projects/{p.id}").text

        assert "function saveProjectHookSec(" not in html
        assert 'id="project-hook-sec"' not in html


class TestReanalyzeControl:
    """POST /api/v2/projects/{pid}/analyze — the only way an analyze-time
    setting can be made to take effect after creation."""

    def test_the_button_is_offered_on_every_project_type(self, client, db_session):
        for ptype in ("face_swap", "localisation", "cover_change"):
            p = _make_project(
                db_session, status=ProjectStatus.ready, duration_sec=25.0,
                project_type=ptype,
            )
            html = client.get(f"/v2/projects/{p.id}").text
            assert 'id="reanalyze-btn"' in html, ptype
            assert "reanalyzeProject(" in html, ptype

    @pytest.mark.parametrize(
        "status", [ProjectStatus.ready, ProjectStatus.failed, ProjectStatus.analyzing]
    )
    def test_the_button_is_offered_in_every_status(self, client, db_session, status):
        """Never greyed out by a page render that may be minutes stale — a
        failed project is the one that most needs re-analyzing, and the two
        409s are the authority on when it is refused."""
        p = _hook_project(db_session, status=status)
        html = client.get(f"/v2/projects/{p.id}").text
        assert 'id="reanalyze-btn"' in html
        assert 'id="reanalyze-status"' in html

    def test_it_calls_the_documented_endpoint(self, client, db_session):
        p = _hook_project(db_session)
        html = client.get(f"/v2/projects/{p.id}").text
        assert "'/api/v2/projects/' + encodeURIComponent(pid) + '/analyze'" in html
        assert "method: 'POST'" in html

    def test_the_endpoint_it_calls_exists(self, client):
        """The typo class this feature has already been bitten by: a fetch URL
        that no route serves is a silent no-op in production."""
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v2/projects/{pid}/analyze" in paths
        assert "post" in paths["/api/v2/projects/{pid}/analyze"]
        # The watcher's two polling URLs, same rule.
        assert "/api/v2/projects/{pid}" in paths
        assert "/api/v2/projects/{pid}/segments" in paths

    def test_it_confirms_first_and_names_what_is_lost_and_what_survives(
        self, client, db_session
    ):
        p = _hook_project(db_session)
        html = client.get(f"/v2/projects/{p.id}").text

        assert "window.confirm(REANALYZE_CONFIRM)" in html
        # Lost.
        assert "DELETES every segment" in _squash(html)
        assert "per-segment generation history" in html
        assert "stitch plan" in html
        # Survives — specifically a done run's delivered video.
        assert "What survives" in html
        assert "delivered video and Drive link" in _squash(html)

    def test_both_409s_are_shown_with_the_server_sentence(self, client, db_session):
        p = _hook_project(db_session)
        html = client.get(f"/v2/projects/{p.id}").text
        assert "'Error: ' + apiErrorText(b, r)" in html

    def test_a_ready_project_polls_nothing_so_the_watcher_is_required(
        self, client, db_session
    ):
        """The reason the button cannot just fire and forget: at "ready" the
        status fragment carries NO hx-trigger, so nothing on the page is on a
        timer, and POST /analyze answers before the row leaves "ready" — a bare
        reload would re-render the stale segments with no way out."""
        p = _hook_project(db_session, hook_sec=8.0)
        _hook_segments(db_session, p.id, 8.0)

        fragment = client.get(f"/v2/projects/{p.id}/status-fragment").text
        assert "hx-trigger" not in fragment

        html = client.get(f"/v2/projects/{p.id}").text
        assert "function watchReanalysis(" in html
        assert "watchReanalysis(pid, before)" in html
        # It waits for the SegmentDef ids to change — the only signal that the
        # new segmentation is committed.
        assert "'/segments'" in html
        assert "location.reload()" in html

    def test_an_analyzing_project_does_carry_the_poll(self, client, db_session):
        """The other side of the same fact: once the worker has flipped the
        row, the existing fragment polling does take over."""
        p = _hook_project(db_session, status=ProjectStatus.analyzing)
        fragment = client.get(f"/v2/projects/{p.id}/status-fragment").text
        assert 'hx-trigger="every 2s"' in fragment
        assert f'hx-get="/v2/projects/{p.id}/status-fragment"' in fragment

    def test_the_transcript_panel_is_refreshed_too(self, client, db_session):
        """A re-analysis sets transcript_status back to "pending", and the
        panel does not poll in a terminal state — so it would keep showing the
        old transcript as "ready"."""
        p = _hook_project(db_session, transcript_status="ready")
        html = client.get(f"/v2/projects/{p.id}").text
        assert "typeof refreshTranscriptPanel === 'function'" in html
        assert "function refreshTranscriptPanel(" in html


# ---------------------------------------------------------------------------
# Voice references — library page, New Run form, run page
# ---------------------------------------------------------------------------


class TestVoiceReferenceUI:
    """Rendered-HTML assertions.

    The New Run form is an HTMX fragment, and this project has been bitten
    before by a control that renders correctly yet is never wired up because
    the breakage lives in *when* the fragment re-renders. Unit tests cannot see
    that, so the checks here are on actual markup across the state matrix.
    """

    @staticmethod
    def _make_voice(db_session, **kwargs):
        from app.models import VoiceClip

        defaults = dict(
            name="Marissa JA", local_path="/tmp/nowhere.mp3", duration_sec=9.5
        )
        defaults.update(kwargs)
        voice = VoiceClip(**defaults)
        db_session.add(voice)
        db_session.commit()
        return voice

    def test_library_page_has_a_voice_section(self, client):
        response = client.get("/v2/library")
        assert response.status_code == 200
        assert "Voice references" in response.text
        # The bounds the server enforces are the bounds the operator is shown.
        assert "30" in response.text

    def test_library_page_lists_voices(self, client, db_session):
        self._make_voice(db_session, language="ja")
        response = client.get("/v2/library")
        assert response.status_code == 200
        assert "Marissa JA" in response.text
        assert "9.5s" in response.text

    def test_new_run_form_offers_the_voice_picker(self, client, db_session):
        voice = self._make_voice(db_session, language="ja")
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")

        html = client.get(f"/v2/projects/{project.id}").text
        assert 'id="voice-ref-group"' in html
        assert 'name="voice_clip_id"' in html
        assert 'name="voice_file"' in html
        assert voice.id in html
        # Tagged language rides along so the localisation pre-fill can match on it.
        assert 'data-lang="ja"' in html

    def test_voice_controls_start_disabled(self, client, db_session):
        """Both inputs ship disabled so nothing is submitted until a mode is picked.

        This is the mechanism that lets the server see exactly one of them — a
        disabled control is not part of the form data.
        """
        self._make_voice(db_session)
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")

        html = client.get(f"/v2/projects/{project.id}").text
        select_tag = html.split('name="voice_clip_id"')[1].split(">")[0]
        assert "disabled" in select_tag
        file_tag = html.split('name="voice_file"')[1].split(">")[0]
        assert "disabled" in file_tag

    def test_empty_library_disables_the_library_option(self, client, db_session):
        """With no voices, "From the library" cannot be chosen — and says why."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")

        html = client.get(f"/v2/projects/{project.id}").text
        assert 'id="voice-ref-group"' in html
        assert "/v2/library" in html

    def test_status_fragment_also_carries_the_picker(self, client, db_session):
        """The HTMX-swapped fragment must carry it too — that is the normal path.

        A project page reached after analysis gets its New Run form through this
        fragment, not a full page load.
        """
        voice = self._make_voice(db_session)
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")

        html = client.get(f"/v2/projects/{project.id}/status-fragment").text
        assert 'id="voice-ref-group"' in html
        assert voice.id in html

    def test_model_limits_drive_the_pickers_visibility(self, client, db_session):
        """The show/hide decision reads the registry, not a hardcoded model name."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0, action="swap")

        html = client.get(f"/v2/projects/{project.id}").text
        assert '"reference_audio"' in html
        assert "limits.reference_audio" in html

    def test_run_page_shows_the_voice_it_used(self, client, db_session, tmp_path):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done,
            model="seedance-2-5", audio_mode="seedance",
            reference_audio_urls=[str(tmp_path / "voice.mp3")],
        )
        html = client.get(f"/v2/runs/{run.id}").text
        assert "Voice reference" in html
        assert f"/api/v2/runs/{run.id}/voice-reference?index=0" in html

    def test_run_page_flags_a_voice_that_was_discarded(
        self, client, db_session, tmp_path
    ):
        """A stitching run on "original" audio threw the generated tracks away.

        Nothing else tells the operator why the delivered video does not use the
        voice they attached.
        """
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done,
            model="seedance-2-5", audio_mode="original",
            reference_audio_urls=[str(tmp_path / "voice.mp3")],
        )
        html = client.get(f"/v2/runs/{run.id}").text
        assert "did not reach the delivered video" in html

    def test_run_page_without_a_voice_shows_no_card(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        html = client.get(f"/v2/runs/{run.id}").text
        assert "Voice reference" not in html

    def test_switching_to_a_voice_model_triggers_the_prefill(
        self, client, db_session
    ):
        """Localisation defaults to seedance-fast, which takes no voice.

        The group therefore starts hidden, so the language pre-pick has to be
        re-run at the moment a model that DOES take one is chosen — otherwise
        the picker sits empty on the one flow that cares most about the voice.
        """
        self._make_voice(db_session, language="ja")
        project = _make_project(
            db_session, status=ProjectStatus.ready, project_type="localisation"
        )
        _make_segment_def(db_session, project.id, 0, action="swap")

        html = client.get(f"/v2/projects/{project.id}").text
        assert "pickLocalisationVoice" in html
        # Wired into the model switch, not only into form init.
        assert "wasHidden" in html


# ---------------------------------------------------------------------------
# Italian (it) — rendered options, filters, suffixed ids, CSV
# ---------------------------------------------------------------------------


class TestItalianRendering:
    def test_run_detail_selector_offers_and_preselects_italian(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, language="it")
        html = client.get(f"/v2/runs/{r.id}").text

        assert '<option value="it" selected>Italian</option>' in html
        # The header/export hint shows the suffixed id the sheet would receive.
        assert f"{r.id[:8]}-it" in html

    def test_italian_is_offered_on_an_english_run_too(self, client, db_session):
        """The option list is the registry, so it does not depend on the run."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id)
        html = client.get(f"/v2/runs/{r.id}").text

        assert '<option value="it" ' in html or '<option value="it">' in html
        assert "Italian" in html
        assert '<option value="en" selected>English</option>' in html

    def test_new_run_form_offers_every_registry_language(self, client, db_session):
        """The New Run form's <select> renders straight off languages.LANGUAGES."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        html = client.get(f"/v2/projects/{p.id}").text

        for label in ("English", "Spanish", "Portuguese", "Japanese",
                      "German", "Italian", "Polish"):
            assert label in html, label
        assert '<option value="it"' in html

    def test_library_page_shows_an_italian_tag(self, client, db_session):
        from app.models import LibraryClip

        clip = LibraryClip(
            name="Outro IT v1", local_path="/tmp/nowhere.mp4",
            duration_sec=4.2, language="it",
        )
        db_session.add(clip)
        db_session.commit()

        html = client.get("/v2/library").text
        assert "Outro IT v1" in html
        assert "Italian" in html


class TestItalianFilters:
    def test_dashboard_language_filter_selects_italian_runs(self, client, db_session):
        p_it = _make_project(db_session, name="IT")
        _make_run(db_session, p_it.id, language="it")
        p_es = _make_project(db_session, name="ES")
        _make_run(db_session, p_es.id, language="es")

        resp = client.get("/v2/?lang=it")
        assert p_it.id[:8] in resp.text
        assert p_es.id[:8] not in resp.text

    def test_italian_is_not_swept_up_by_the_english_filter(self, client, db_session):
        """NULL is English; 'it' is emphatically not."""
        p_null = _make_project(db_session, name="Unset")
        _make_run(db_session, p_null.id, language=None)
        p_it = _make_project(db_session, name="IT")
        _make_run(db_session, p_it.id, language="it")

        resp = client.get("/v2/?lang=en")
        assert p_null.id[:8] in resp.text
        assert p_it.id[:8] not in resp.text

    def test_italian_combines_with_other_languages_in_a_multi_select(
        self, client, db_session
    ):
        p_it = _make_project(db_session, name="IT")
        _make_run(db_session, p_it.id, language="it")
        p_ja = _make_project(db_session, name="JA")
        _make_run(db_session, p_ja.id, language="ja")
        p_pt = _make_project(db_session, name="PT")
        _make_run(db_session, p_pt.id, language="pt")

        resp = client.get("/v2/?lang=it&lang=ja")
        assert p_it.id[:8] in resp.text
        assert p_ja.id[:8] in resp.text
        assert p_pt.id[:8] not in resp.text

    def test_run_id_filter_accepts_the_italian_suffixed_id(self, client, db_session):
        """The operator pastes '1736426f-it' — the suffix is stripped first."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, name="Wanted", language="it")
        other = _make_run(db_session, project.id, name="Other")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?run_id={wanted.id[:8]}-it"
        )
        assert resp.status_code == 200
        assert wanted.id in resp.text
        assert other.id not in resp.text

    def test_run_id_filter_is_case_insensitive_about_the_suffix(
        self, client, db_session
    ):
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, name="Wanted", language="it")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?run_id={wanted.id[:8].upper()}-IT"
        )
        assert resp.status_code == 200
        assert wanted.id in resp.text

    def test_run_id_filter_still_accepts_the_full_uuid_of_an_italian_run(
        self, client, db_session
    ):
        """A full uuid must survive the strip untouched."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, name="Wanted", language="it")
        other = _make_run(db_session, project.id, name="Other")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?run_id={wanted.id}"
        )
        assert resp.status_code == 200
        assert wanted.id in resp.text
        assert other.id not in resp.text


class TestItalianGenerationsAndCsv:
    def test_runs_fragment_shows_the_italian_short_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, name="Redhead", language="it")

        resp = client.get(f"/v2/projects/{project.id}/runs-fragment")
        assert f"{run.id[:8]}-it" in resp.text

    def test_csv_pairs_the_full_uuid_with_the_italian_job_id(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, p.id, index=0)
        r = _make_run(db_session, p.id, status=RunStatus.done, language="it")
        _make_run_segment(
            db_session, r.id, index=0, segment_def_id=sd.id,
            status=SegmentStatus.completed,
        )
        resp = client.get("/v2/generations.csv")
        assert resp.status_code == 200
        body = resp.text
        assert "Job ID" in body.splitlines()[0]
        assert f"{r.id},{r.id[:8]}-it" in body

    def test_single_overlay_history_row_carries_the_italian_run_job_id(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done, language="it")
        _make_overlay_job(db_session, source_run_id=r.id, status=OverlayStatus.done)

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert f"{r.id[:8]}-it" in html

    def test_several_overlays_carry_the_composite_italian_job_id(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, status=RunStatus.done, language="it")
        a = _make_overlay_job(db_session, source_run_id=r.id, status=OverlayStatus.done)
        b = _make_overlay_job(db_session, source_run_id=r.id, status=OverlayStatus.done)

        html = client.get(f"/v2/runs/{r.id}/overlays-fragment").text
        assert f"{r.id[:8]}_{a.id[:8]}-it" in html
        assert f"{r.id[:8]}_{b.id[:8]}-it" in html


# ---------------------------------------------------------------------------
# Polish (pl) — rendered options, library tag, filters, suffixed ids
# ---------------------------------------------------------------------------


class TestPolishWebSurfaces:
    def test_run_detail_selector_offers_and_preselects_polish(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, language="pl")
        html = client.get(f"/v2/runs/{r.id}").text

        assert '<option value="pl" selected>Polish</option>' in html
        assert f"{r.id[:8]}-pl" in html

    def test_new_run_form_offers_polish(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        html = client.get(f"/v2/projects/{p.id}").text
        assert '<option value="pl"' in html
        assert "Polish" in html

    def test_library_page_offers_and_shows_a_polish_tag(self, client, db_session):
        from app.models import LibraryClip

        db_session.add(LibraryClip(
            name="Outro PL v1", local_path="/tmp/nowhere.mp4",
            duration_sec=4.2, language="pl",
        ))
        db_session.commit()

        html = client.get("/v2/library").text
        assert "Outro PL v1" in html
        assert '<option value="pl"' in html
        assert "Polish" in html

    def test_dashboard_language_filter_offers_and_selects_polish(
        self, client, db_session
    ):
        p_pl = _make_project(db_session, name="PL")
        _make_run(db_session, p_pl.id, language="pl")
        p_it = _make_project(db_session, name="IT")
        _make_run(db_session, p_it.id, language="it")
        p_null = _make_project(db_session, name="Unset")
        _make_run(db_session, p_null.id, language=None)

        html = client.get("/v2/?lang=pl").text
        assert 'value="pl"' in html
        assert p_pl.id[:8] in html
        assert p_it.id[:8] not in html
        assert p_null.id[:8] not in html

        # Not swept up by English (NULL is English; 'pl' is not).
        html = client.get("/v2/?lang=en").text
        assert p_null.id[:8] in html
        assert p_pl.id[:8] not in html

    def test_run_id_filter_accepts_the_polish_suffixed_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, name="Wanted", language="pl")
        other = _make_run(db_session, project.id, name="Other")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?run_id={wanted.id[:8].upper()}-PL"
        )
        assert resp.status_code == 200
        assert wanted.id in resp.text
        assert other.id not in resp.text


# ---------------------------------------------------------------------------
# French (fr) — rendered options, library tag, filters, suffixed ids
# ---------------------------------------------------------------------------


class TestFrenchWebSurfaces:
    def test_run_detail_selector_offers_and_preselects_french(
        self, client, db_session
    ):
        p = _make_project(db_session, status=ProjectStatus.ready)
        r = _make_run(db_session, p.id, language="fr")
        html = client.get(f"/v2/runs/{r.id}").text

        assert '<option value="fr" selected>French</option>' in html
        assert f"{r.id[:8]}-fr" in html

    def test_new_run_form_offers_french(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        html = client.get(f"/v2/projects/{p.id}").text
        assert '<option value="fr"' in html
        assert "French" in html

    def test_library_page_offers_and_shows_a_french_tag(self, client, db_session):
        from app.models import LibraryClip

        db_session.add(LibraryClip(
            name="Outro FR v1", local_path="/tmp/nowhere.mp4",
            duration_sec=4.2, language="fr",
        ))
        db_session.commit()

        html = client.get("/v2/library").text
        assert "Outro FR v1" in html
        assert '<option value="fr"' in html
        assert "French" in html

    def test_dashboard_language_filter_offers_and_selects_french(
        self, client, db_session
    ):
        p_fr = _make_project(db_session, name="FR")
        _make_run(db_session, p_fr.id, language="fr")
        p_it = _make_project(db_session, name="IT")
        _make_run(db_session, p_it.id, language="it")
        p_null = _make_project(db_session, name="Unset")
        _make_run(db_session, p_null.id, language=None)

        html = client.get("/v2/?lang=fr").text
        assert 'value="fr"' in html
        assert p_fr.id[:8] in html
        assert p_it.id[:8] not in html
        assert p_null.id[:8] not in html

        # Not swept up by English (NULL is English; 'fr' is not).
        html = client.get("/v2/?lang=en").text
        assert p_null.id[:8] in html
        assert p_fr.id[:8] not in html

    def test_run_id_filter_accepts_the_french_suffixed_id(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        wanted = _make_run(db_session, project.id, name="Wanted", language="fr")
        other = _make_run(db_session, project.id, name="Other")

        resp = client.get(
            f"/v2/projects/{project.id}/runs-fragment?run_id={wanted.id[:8].upper()}-FR"
        )
        assert resp.status_code == 200
        assert wanted.id in resp.text
        assert other.id not in resp.text


class TestProjectCaptionDetectorSettings:
    """AICP-343: the project page edits the QA caption detector config.

    AICP-361: only on subtitle-removal projects."""

    def test_defaults_render_blank_with_placeholders(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready,
                                project_type="subtitle_removal")
        html = client.get(f"/v2/projects/{project.id}").text
        assert 'id="cd-band-y0"' in html and 'placeholder="0.50"' in html
        assert "var _captionDetectorSaved = {};" in html
        tag = html.split('id="cd-text-color"')[1].split(">")[0]
        assert 'value=""' in tag

    def test_stored_config_is_shown_and_kept(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready,
                                project_type="subtitle_removal", caption_detector={
            "band_y": [0.7, 0.9], "text_color": "#ffff00", "outline_style": "none",
            "glyph_height_px": [10, 80], "threshold": 60})
        html = client.get(f"/v2/projects/{project.id}").text
        assert 'value="0.7"' in html.split('id="cd-band-y0"')[1].split(">")[0]
        assert 'value=""' in html.split('id="cd-band-x0"')[1].split(">")[0]
        assert 'value="#ffff00"' in html.split('id="cd-text-color"')[1].split(">")[0]
        assert 'value="60"' in html.split('id="cd-threshold"')[1].split(">")[0]
        assert '<option value="none" selected>' in html
        # API-only fields ride along in the saved object so a UI save keeps them.
        assert '"glyph_height_px": [10, 80]' in html

    @pytest.mark.parametrize("project_type",
                             ["face_swap", "localisation", "cover_change", "reframe"])
    def test_hidden_on_other_project_types(self, client, db_session, project_type):
        project = _make_project(db_session, status=ProjectStatus.ready,
                                project_type=project_type)
        resp = client.get(f"/v2/projects/{project.id}")
        assert resp.status_code == 200
        html = resp.text
        assert "QA caption detector" not in html
        assert 'id="cd-band-y0"' not in html
        assert "_captionDetectorSaved" not in html
        assert "saveCaptionDetector" not in html
        # The shared helper the other settings use still renders.
        assert "function patchProject(" in html

    def test_saved_value_on_other_type_is_kept_not_cleared(self, client, db_session):
        cfg = {"band_y": [0.7, 0.9], "threshold": 60}
        project = _make_project(db_session, status=ProjectStatus.ready,
                                project_type="face_swap", caption_detector=cfg)
        html = client.get(f"/v2/projects/{project.id}").text
        assert 'id="cd-band-y0"' not in html
        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).caption_detector == cfg
        # The API still exposes it for every type.
        body = client.get(f"/api/v2/projects/{project.id}").json()
        assert body["caption_detector"] == cfg


# ---------------------------------------------------------------------------
# AICP-274: "share this with the service account" notice next to Drive inputs
# ---------------------------------------------------------------------------


class TestDriveServiceAccountNotice:
    EMAIL = "sa-notice-test@example-project.iam.gserviceaccount.com"

    @pytest.fixture(autouse=True)
    def _key(self, tmp_path, monkeypatch):
        import json as _json

        from app import drive_sa
        from app.config import settings as _settings

        key = tmp_path / "sa.json"
        key.write_text(_json.dumps({"client_email": self.EMAIL, "private_key": "x"}))
        monkeypatch.setattr(_settings, "GOOGLE_SERVICE_ACCOUNT_FILE", str(key))
        drive_sa.reset_cache()
        yield
        drive_sa.reset_cache()

    def test_dashboard_source_link(self, client):
        html = client.get("/v2/").text
        assert 'name="gdrive_link"' in html
        assert self.EMAIL in html and f'data-copy-text="{self.EMAIL}"' in html

    def test_library_has_a_notice_per_drive_input(self, client):
        html = client.get("/v2/library").text
        assert html.count('name="gdrive_url"') == 3
        assert html.count(f'data-copy-text="{self.EMAIL}"') == 3

    def test_delivery_folder_asks_for_editor(self, client, db_session):
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}/status-fragment").text
        assert 'name="gdrive_folder_id"' in html
        assert f'data-copy-text="{self.EMAIL}"' in html
        assert "Editor access" in html

    def test_overlay_editor_drive_png(self, client, db_session):
        job = _make_overlay_job(db_session)
        html = client.get(f"/v2/overlays/{job.id}").text
        assert 'id="drive-link"' in html
        assert f'data-copy-text="{self.EMAIL}"' in html

    def test_pages_render_without_a_key_file(self, client, db_session, monkeypatch):
        from app import drive_sa
        from app.config import settings as _settings

        monkeypatch.setattr(_settings, "GOOGLE_SERVICE_ACCOUNT_FILE", "/nonexistent/sa.json")
        drive_sa.reset_cache()
        for url in ("/v2/", "/v2/library"):
            resp = client.get(url)
            assert resp.status_code == 200, url
            assert "must be shared with" in resp.text
            assert "data-copy-text" not in resp.text

    def test_reference_url_inputs_that_accept_drive_links(self, client, db_session):
        """Reference/character URL boxes take Drive share links too
        (pipeline_common.resolve_reference_urls), so each carries the notice."""
        p = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, p.id, 0)
        html = client.get(f"/v2/projects/{p.id}/status-fragment").text
        for marker in ('name="reference_urls"', 'name="preprocess_reference_url"'):
            i = html.index(marker)
            assert "drive-sa-notice" in html[i : i + 900], marker
