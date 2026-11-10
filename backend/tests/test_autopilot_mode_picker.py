"""AICP-378: the mode cards and localisation fields on the New project and the
project-from-run forms (partials/autopilot_mode.html)."""

from __future__ import annotations

import json
import re

import pytest

from app.config import settings
from app.models import LibraryClip, ProjectStatus, RunStatus

from tests.test_web_v2 import (  # noqa: F401
    SessionFactory,
    _make_project,
    _make_run,
    client,
    db_session,
    engine,
)


def _box(html, prefix):
    m = re.search(r'<div class="form-group ap-mode" id="%s-mode-box"[^>]*>' % prefix, html)
    assert m, f"no {prefix} mode box"
    return m.group(0)


def _attr(tag, name):
    m = re.search(r"%s='([^']*)'" % name, tag) or re.search(r'%s="([^"]*)"' % name, tag)
    return m.group(1) if m else None


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "", raising=False)


def test_new_project_form_has_the_cards_and_no_json_mode_select(client, enabled):
    html = client.get("/v2/").text
    box = _box(html, "new-project")
    assert set(_attr(box, "data-types").split(",")) >= {"localisation", "subtitle_removal"}
    assert 'name="mode" value="manual" checked' in html and 'name="mode" value="assistant"' in html
    assert 'id="new-project-mode"' not in html and 'id="new-project-ap-params"' not in html
    # the params travel in one hidden field, disabled (not posted) until there is something to send
    assert 'name="autopilot_params" data-ap-params disabled' in html
    assert "It asks you to approve a budget before its first paid step" in html
    assert "apMode.init('new-project', {typeSelect: 'new-project-type', scopeSelect: 'new-project-localisation-scope'})" in html
    langs = json.loads(_attr(box, "data-langs").replace("&#34;", '"'))
    assert [x["code"] for x in langs][:3] == ["en", "es", "pt"]


def test_assistant_unavailable_when_the_autopilot_is_off(client, monkeypatch):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", False)
    html = client.get("/v2/").text
    assert _attr(_box(html, "new-project"), "data-types") == ""
    assert "Not available: the autopilot is off." in html      # the JS reason text


def test_telegram_line_only_when_notifications_are_configured(client, enabled, monkeypatch):
    assert "Questions also go to Telegram." not in client.get("/v2/").text
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "x", raising=False)
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "1", raising=False)
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://example.test", raising=False)
    assert "Questions also go to Telegram." in client.get("/v2/").text


def test_tail_picker_reads_the_library_endpoint_with_languages(client, enabled, db_session):
    db_session.add(LibraryClip(name="CTA ES", language="es", local_path="/x/a.mp4", duration_sec=3.2))
    db_session.commit()
    html = client.get("/v2/").text
    assert "fetch('/api/v2/library-clips'" in html and "return c && c.language;" in html
    rows = client.get("/api/v2/library-clips").json()
    assert {"name": "CTA ES", "language": "es"}.items() <= next(r for r in rows if r["name"] == "CTA ES").items()


def _done_run(db_session, tmp_path, mode):
    f = tmp_path / "final.mp4"
    f.write_bytes(b"\x00VIDEO\xff")
    p = _make_project(db_session, status=ProjectStatus.ready, mode=mode)
    return _make_run(db_session, p.id, status=RunStatus.done, result_local_path=str(f))


@pytest.mark.parametrize("parent", ["manual", "assistant"])
def test_project_from_run_shows_the_cards_with_the_parents_default(client, enabled, db_session,
                                                                   tmp_path, parent):
    run = _done_run(db_session, tmp_path, parent)
    html = client.get(f"/v2/runs/{run.id}").text
    _box(html, "move-project")
    assert 'name="move-project-mode" value="assistant"' in html
    assert 'id="move-project-mode"' not in html and 'id="move-project-ap-params"' not in html
    assert f"return \"{parent}\" === 'assistant'" in html
    assert "apMode.value('move-project')" in html
    assert "fd.append('mode', ap.mode)" in html
