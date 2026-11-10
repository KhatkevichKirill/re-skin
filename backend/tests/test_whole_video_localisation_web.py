"""Whole-video localisation scope on the project pages (AICP-340)."""

from __future__ import annotations

from app.models import ProjectStatus

from app.models import RunStatus

from tests.test_web_v2 import (  # noqa: F401
    SessionFactory,
    _make_project,
    _make_run,
    _make_segment_def,
    client,
    db_session,
    engine,
)


def _project(session, scope, **kw):
    p = _make_project(
        session,
        status=ProjectStatus.ready,
        project_type="localisation",
        duration_sec=30.0, width=1080, height=1920, fps=30.0,
        localisation_scope=scope,
        **kw,
    )
    return p


def test_hook_project_shows_the_scope_selector_on_hook(client, db_session):
    p = _project(db_session, "hook")
    _make_segment_def(db_session, p.id, 0, start_sec=0.0, end_sec=10.0, action="swap")
    _make_segment_def(db_session, p.id, 1, start_sec=10.0, end_sec=30.0, action="keep")
    html = client.get(f"/v2/projects/{p.id}").text
    assert 'id="project-localisation-scope"' in html
    assert '<option value="hook" selected>' in html
    assert "Currently analyzed as <strong>hook only</strong>" in html
    assert "var LOCALISATION_WHOLE = false;" in html
    assert 'id="whole-video-hint"' not in html


def test_whole_project_renders_the_whole_video_state(client, db_session):
    p = _project(db_session, "whole", hook_sec=88.0)
    for i, (s, e) in enumerate([(0.0, 12.0), (12.0, 24.0), (24.0, 30.0)]):
        _make_segment_def(db_session, p.id, i, start_sec=s, end_sec=e, action="swap")
    html = client.get(f"/v2/projects/{p.id}").text
    assert '<option value="whole" selected>' in html
    assert "Currently analyzed as <strong>whole video</strong>" in html
    assert 'id="whole-video-hint"' in html
    assert "var LOCALISATION_WHOLE = true;" in html
    # No hook clamp warning for a scope that has no hook.
    assert 'id="project-hook-clamp"' not in html
    assert 'id="hook-clamp-notice"' not in html
    assert "Not used while the project is analyzed as the whole" in html
    # The Re-analyze button carries the selected scope.
    assert "reanalyzeQuery()" in html


def test_non_localisation_page_has_no_scope_selector(client, db_session):
    p = _make_project(db_session, status=ProjectStatus.ready, project_type="face_swap",
                      duration_sec=10.0)
    html = client.get(f"/v2/projects/{p.id}").text
    assert 'id="project-localisation-scope"' not in html


def test_create_form_offers_the_scope(client):
    html = client.get("/v2/").text
    assert 'id="new-project-localisation-scope"' in html
    assert 'name="localisation_scope" disabled' in html


def test_a_pending_request_is_shown_and_preselected(client, db_session):
    p = _project(db_session, "hook")
    p.localisation_scope_requested = "whole"
    db_session.commit()
    html = client.get(f"/v2/projects/{p.id}").text
    assert '<option value="whole" selected>' in html
    assert "was requested and has not been applied yet" in html
    assert "Currently analyzed as <strong>hook only</strong>" in html


def test_new_run_form_has_the_whole_video_roll_check(client, db_session):
    p = _project(db_session, "whole")
    _make_segment_def(db_session, p.id, 0, start_sec=0.0, end_sec=12.0, action="swap")
    html = client.get(f"/v2/projects/{p.id}").text
    assert 'id="whole-video-roll-warning"' in html
    assert "var WHOLE_VIDEO_SCOPE = true;" in html
    assert "checkWholeVideoClipLengths(maxSec);" in html
    assert "Run created, with a warning" in html


def test_move_to_project_offers_the_scope(client, db_session, tmp_path):
    result_file = tmp_path / "final.mp4"
    result_file.write_bytes(b"\x00VIDEO\xff")
    p = _make_project(db_session, status=ProjectStatus.ready)
    run = _make_run(db_session, p.id, status=RunStatus.done, result_local_path=str(result_file))
    html = client.get(f"/v2/runs/{run.id}").text
    assert 'id="move-project-scope"' in html
    assert 'data-hook="1">Localisation' in html
    assert "fd.append('localisation_scope', scopeSel.value)" in html


def test_recut_at_pauses_button_only_in_whole_scope(client, db_session):
    """AICP-347: the Re-cut at speech pauses action is a whole-video tool."""
    whole = _project(db_session, "whole")
    _make_segment_def(db_session, whole.id, 0, start_sec=0.0, end_sec=30.0, action="swap")
    hook = _project(db_session, "hook")
    _make_segment_def(db_session, hook.id, 0, start_sec=0.0, end_sec=10.0, action="swap")
    _make_segment_def(db_session, hook.id, 1, start_sec=10.0, end_sec=30.0, action="keep")
    html = client.get(f"/v2/projects/{whole.id}").text
    assert 'id="recut-btn"' in html
    assert "/recut-at-pauses" in html
    assert "14.72s" in html
    assert 'id="recut-btn"' not in client.get(f"/v2/projects/{hook.id}").text
