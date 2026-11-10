"""Rebuild from reference — the project page's JavaScript in a real browser.

What rendered HTML cannot prove (the localisation lesson: HTMX lifecycle bugs
are invisible to unit tests):

* the launch form (``project_rebuild_card.html::rbLaunch``) posts the form's
  fields and NAVIGATES to the new run on 201, or shows the refusal's
  ``detail`` and re-enables the button;
* switching the model swaps the resolution options and the precondition list
  (``rbLaunchModel``);
* the card's self-polling (``hx-trigger="every 4s [...]"``) PAUSES while the
  operator has focus in one of its forms, and resumes after;
* the project page (all selected) and the run page (done with QA flags) render
  — screenshots go to ``$REBUILD_SCREENSHOT_DIR`` when set.

The state is built through the real API and job steps of
tests/test_rebuild_e2e.py (fake kie.ai counting POSTs, real ffmpeg). A uvicorn
thread on 127.0.0.1 serves the same app. Skipped unless playwright and a
chromium build are installed (neither is in requirements.txt).
"""

from __future__ import annotations

import os
import socket
import threading
import time

import pytest

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed").sync_playwright

from sqlalchemy import select  # noqa: E402

from tests.test_rebuild_e2e import (  # noqa: E402,F401 - fixtures by import
    SF,
    FakeDrive,
    _clean_libraries,
    _drive_run,
    _library_end_card,
    _fragment,
    client,
    env,
    form_fields,
    stage_project,
)
from app.models import LibraryClip, Run  # noqa: E402
from app.rebuild_models import RebuildKeyframe  # noqa: E402

SHOTS = os.environ.get("REBUILD_SCREENSHOT_DIR")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live(client):
    """uvicorn on the same app (and the same get_db override / stubs the
    ``client`` and ``env`` fixtures installed; monkeypatch restores those)."""
    import uvicorn

    from app.main import app

    server = thread = None
    try:
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                               log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 30
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        if not server.started:
            pytest.fail("uvicorn did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # no browser build downloaded
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    try:
        yield pg
    finally:
        ctx.close()


def _shot(pg, name):
    if SHOTS:
        os.makedirs(SHOTS, exist_ok=True)
        pg.screenshot(path=os.path.join(SHOTS, name), full_page=True)


def _open_project(pg, base, pid):
    pg.goto(f"{base}/v2/projects/{pid}")
    pg.wait_for_selector("#rebuild-card", timeout=15000)  # hx-trigger="load"


def test_launch_form_navigates_to_the_new_run(client, env, SF, tmp_path, live, page):
    _library_end_card(SF, tmp_path, "en")
    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    _open_project(page, live, pid)
    btn = page.locator("#rb-launch-btn")
    assert btn.is_enabled()
    assert page.locator("#rb-unmet").count() == 0
    assert page.locator("#rb-resolution").input_value() == "720p"
    assert "Estimate: about" in page.locator("#rb-estimate").inner_text()
    _shot(page, "project-all-selected.png")
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    with page.expect_navigation(url="**/v2/runs/**", timeout=20000):
        btn.click()
    assert dialogs and dialogs[0].startswith("Launch the rebuild run?")
    rid = page.url.rstrip("/").split("/")[-1]
    with SF() as s:
        run = s.get(Run, rid)
        assert run.generation_mode == "images_only" and run.model == "kling-4-0"
        assert run.resolution == "720p" and run.language == "en"
        assert run.stitch_plan and run.stitch_plan[-1]["kind"] == "library"
    page.wait_for_selector("#rebuild-chunks")
    assert client.runs == [rid]


def test_end_card_line_follows_the_language_and_the_picker(client, env, SF, tmp_path,
                                                           live, page):
    """AICP-381: the picker lists end cards only, and the line under it says
    which clip (name + length) will close the video before launch."""
    card = _library_end_card(SF, tmp_path, "en")
    with SF() as s:  # a NEWER English insert: never listed, never auto-picked
        path = s.get(LibraryClip, card).local_path
        s.add(LibraryClip(name="ip_reader_demo", local_path=path, language="en",
                          kind="insert", duration_sec=27.8))
        s.commit()
    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    _open_project(page, live, pid)
    opts = page.locator("#rb-end-card option").all_inner_texts()
    assert any("Outro en" in o for o in opts)
    assert not any("ip_reader_demo" in o for o in opts)
    info = page.locator("#rb-end-card-info")
    assert "Outro en" in info.inner_text() and "replaces the source end card" in info.inner_text()
    page.select_option("#rb-language", "ja")
    assert "No Japanese end card in the library" in info.inner_text()
    page.select_option("#rb-end-card", "source")
    assert "source end card is kept" in info.inner_text()
    page.select_option("#rb-end-card", f"clip:{card}")
    assert "Outro en" in info.inner_text()
    page.select_option("#rb-end-card", "auto")
    page.select_option("#rb-language", "en")
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    page.locator("#rb-launch-btn").click()
    page.wait_for_timeout(500)
    assert dialogs and "End card: “Outro en”" in dialogs[0]
    with SF() as s:  # dismissed: nothing launched
        assert not s.execute(select(Run).where(Run.project_id == pid)).first()


def test_model_switch_updates_resolutions_and_the_launch_state(client, env, SF, tmp_path,
                                                               live, page):
    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    _open_project(page, live, pid)
    kling = page.locator("#rb-estimate").inner_text()
    page.select_option("#rb-model", "seedance-2-5")
    opts = page.locator("#rb-resolution option").all_inner_texts()
    assert opts == ["480p", "720p", "1080p"]
    assert page.locator("#rb-resolution").input_value() == "720p"  # kept
    assert page.locator("#rb-launch-btn").is_enabled()
    assert page.locator("#rb-estimate").inner_text() != kling
    page.select_option("#rb-model", "kling-4-0")
    assert page.locator("#rb-resolution option").all_inner_texts() == ["720p", "1080p"]


def test_refused_launch_shows_the_detail_and_stays(client, env, SF, tmp_path, live, page):
    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    _open_project(page, live, pid)
    # A precondition breaks after the page rendered "ready".
    with SF() as s:
        kf = s.execute(select(RebuildKeyframe).where(
            RebuildKeyframe.project_id == pid, RebuildKeyframe.shot_id.is_not(None),
            RebuildKeyframe.selected.is_(True))).scalars().first()
        os.remove(kf.image_path)
    page.on("dialog", lambda d: d.accept())
    url = page.url
    page.locator("#rb-launch-btn").click()
    page.wait_for_function(
        "document.getElementById('rb-msg').textContent.indexOf('Error:') === 0",
        timeout=15000)
    msg = page.locator("#rb-msg").inner_text()
    assert msg.startswith("Error: Cannot launch the rebuild: chunk 1 (")
    assert "no selected keyframe for shot(s) 1" in msg
    assert page.url == url and page.locator("#rb-launch-btn").is_enabled()
    _shot(page, "project-launch-refused.png")
    with SF() as s:
        assert not s.execute(select(Run).where(Run.project_id == pid)).first()


def test_polling_pauses_while_a_card_form_has_focus(client, env, SF, tmp_path, live, page):
    pid = stage_project(client, env, SF, tmp_path, "storyboard_running")
    hits = []
    page.on("request", lambda r: hits.append(time.time())
            if r.url.endswith(f"/v2/projects/{pid}/rebuild-fragment") else None)
    _open_project(page, live, pid)
    assert page.locator("#rb-sb-running").count() == 1
    page.wait_for_timeout(9000)
    assert len(hits) >= 3  # the hx load + at least two 4 s polls
    # Focus a field of a card form and type: no swap may wipe it.
    page.locator("#rb-name").click()
    page.keyboard.type("operator typing")
    before = len(hits)
    page.wait_for_timeout(9000)
    assert len(hits) == before
    assert page.locator("#rb-name").input_value() == "operator typing"
    # Focus leaves the form: polling resumes.
    page.locator("body").click(position={"x": 5, "y": 5})
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.wait_for_timeout(9000)
    assert len(hits) > before
    env.drive()  # finish the storyboard so nothing is left live


def test_run_page_done_with_flags(client, env, SF, tmp_path, live, page):
    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    r = client.post(f"/api/v2/projects/{pid}/runs", data=form_fields(_fragment(client, pid)))
    assert r.status_code == 201, r.text
    rid = r.json()["run_id"]
    assert _drive_run(env, client, rid, FakeDrive()) == "finalize"
    page.goto(f"{live}/v2/runs/{rid}")
    page.wait_for_selector("#rebuild-chunks")
    text = page.locator("#rebuild-chunks").text_content()  # badges are upper-cased by CSS
    assert "⚑ shot 1: missing C1" in text and "character check: flagged" in text
    assert page.locator(".js-chunk-flags .badge-warning").count() == 1
    assert page.locator(".js-reroll-btn").count() == 2
    _shot(page, "run-done-with-flags.png")
