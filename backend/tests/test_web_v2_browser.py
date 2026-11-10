"""
Browser tests for the v2 UI's language JavaScript (app/templates/v2/…).

Why this file exists
--------------------
Two behaviours are implemented in JS, so rendered HTML cannot prove them:

1. ``run_detail.html::saveRunLanguage`` builds the displayed short id itself
   (``rid.slice(0,8) + '-' + code``) instead of re-reading ``Run.short_id``.
   It is a SECOND implementation of app/languages.py's suffix rule, and the
   only thing that keeps the header honest between page loads.
2. ``project_status_content.html::pickLocalisationTail`` /
   ``pickLocalisationVoice`` match ``<option data-lang>`` against the selected
   language, and must leave a hand-picked asset alone afterwards.

Isolation
---------
A throwaway SQLite file and DATA_DIR, a uvicorn bound to 127.0.0.1 on an
ephemeral port, and fake credentials. Nothing here reaches Postgres, Redis, a
queue, Drive, Sheets or any paid provider.

Unlike test_web_v2.py / test_api_v2.py, which assign DATABASE_URL and DATA_DIR
unconditionally at import, this module uses ``setdefault`` so it never rebinds
a process another module has already configured — see the note at that code.
Its own DB comes from the SessionFactory override either way.

Skipped unless BOTH playwright and its chromium build are present — neither is
in backend/requirements.txt, so the default suite skips this module rather than
failing on it. To run it:

    pip install playwright && playwright install chromium
    python -m pytest tests/test_web_v2_browser.py
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time

import pytest

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed"
).sync_playwright

# Isolated storage — set before any app import, as the other web tests do.
#
# setdefault, NOT unconditional assignment: app.config.settings binds on first
# import, so if another test module has already configured this process, taking
# DATABASE_URL/DATA_DIR off it would only bind app.db and app.storage to THIS
# module's temp paths for everyone else while changing nothing for us — the
# tests below reach the DB exclusively through the SessionFactory override in
# live_server, which is bound to _db_path explicitly either way. Deferring
# keeps `pytest tests/test_web_v2_browser.py tests/test_x.py` (any order, or a
# randomised one) from leaking this module's storage into test_x.
_db_fd, _db_path = tempfile.mkstemp(suffix="_v2browser.db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
# Never let a stray real credential be picked up by this module.
os.environ.setdefault("KIE_API_KEY", "test-not-a-real-key")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import Base, get_db  # noqa: E402
from app.models import LibraryClip, Run, SegmentDef, VideoProject, VoiceClip  # noqa: E402
from app.state_machine import ProjectStatus, RunStatus  # noqa: E402


_MISSING = object()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        f"sqlite:///{_db_path}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="module")
def SessionFactory(engine):
    return sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )


@pytest.fixture(scope="module")
def live_server(engine, SessionFactory):
    """A real uvicorn on localhost so a browser can drive the page."""
    import uvicorn

    import app.api_v2 as api_v2_module
    import app.tasks as tasks_mod
    import app.web_v2 as web_v2_mod

    # These are module GLOBALS, not fixtures, so every one of them is restored
    # in the teardown below. Leaving a no-op lambda behind would silently
    # disarm enqueueing (or the credit fetch) for every later test in the same
    # pytest process — a passing suite that never exercised the real call.
    _patched = [
        (api_v2_module, "enqueue_analyze_project", lambda pid: None),
        (api_v2_module, "enqueue_process_run", lambda rid: None),
        (tasks_mod, "enqueue_analyze_project", lambda pid: None),
        (tasks_mod, "enqueue_process_run", lambda rid: None),
        # The site header's credits fragment calls kie.ai. Stub it: an offline
        # test must make no external request at all, not even a failing one.
        (web_v2_mod, "_cached_credit_balance", lambda: None),
    ]
    _saved = [
        (mod, name, getattr(mod, name, _MISSING)) for mod, name, _ in _patched
    ]
    for mod, name, stub in _patched:
        setattr(mod, name, stub)

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

    # Restore this ONE key, rather than .clear()ing the shared dict: another
    # fixture's override must survive this module.
    _prev_override = app.dependency_overrides.get(get_db, _MISSING)
    app.dependency_overrides[get_db] = override_get_db

    server = None
    thread = None
    # Everything above this point is already undoable, and everything below can
    # fail (a bind race, a uvicorn that never starts). try/finally so teardown
    # runs on those paths too — a bare `pytest.fail()` before the yield would
    # skip it entirely and leave the stubs above installed for the rest of the
    # pytest process.
    try:
        port = _free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning"
        )
        server = uvicorn.Server(config)
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
        if _prev_override is _MISSING:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = _prev_override
        for mod, name, original in _saved:
            if original is _MISSING:
                delattr(mod, name)
            else:
                setattr(mod, name, original)


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
    ctx = browser.new_context()
    pg = ctx.new_page()
    yield pg
    ctx.close()


@pytest.fixture(autouse=True)
def clean_library(SessionFactory):
    """The clip/voice libraries are GLOBAL, not per project.

    Without this, an Italian clip created by one test is still on the page in
    the next one — which silently turns a "no Italian asset" assertion into a
    passing auto-pick, or a timeout.
    """
    def _wipe():
        s = SessionFactory()
        s.query(LibraryClip).delete()
        s.query(VoiceClip).delete()
        s.commit()
        s.close()

    _wipe()
    yield
    _wipe()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _project(session, **kw) -> VideoProject:
    defaults = dict(
        source_type="upload", source_ref="test.mp4", status=ProjectStatus.ready
    )
    defaults.update(kw)
    p = VideoProject(**defaults)
    session.add(p)
    session.commit()
    return p


def _run(session, project_id, **kw) -> Run:
    defaults = dict(
        project_id=project_id, prompt="Replace the person.", resolution="480p",
        status=RunStatus.created, reference_image_urls=[],
    )
    defaults.update(kw)
    r = Run(**defaults)
    session.add(r)
    session.commit()
    return r


def _clip(session, name, language) -> LibraryClip:
    c = LibraryClip(
        name=name, local_path=f"/tmp/{name}.mp4", duration_sec=4.0, language=language
    )
    session.add(c)
    session.commit()
    return c


# ---------------------------------------------------------------------------
# 1. The run page's own suffix implementation
# ---------------------------------------------------------------------------


class TestRunLanguageJavaScript:
    """saveRunLanguage rebuilds the short id in the browser, not on the server."""

    def test_switching_to_italian_updates_the_header_and_job_id_hint(
        self, live_server, page, SessionFactory
    ):
        s = SessionFactory()
        p = _project(s)
        r = _run(s, p.id)
        rid = r.id
        s.close()

        page.goto(f"{live_server}/v2/runs/{rid}")
        page.wait_for_selector("#run-language")
        assert page.text_content("#run-short-id").strip() == rid[:8]

        page.select_option("#run-language", "it")
        page.wait_for_function(
            "id => document.getElementById('run-short-id').textContent.trim() === id",
            arg=f"{rid[:8]}-it",
        )

        assert page.text_content("#run-short-id").strip() == f"{rid[:8]}-it"
        job_hint = page.query_selector("#export-job-id")
        if job_hint:
            assert job_hint.text_content().strip() == f"{rid[:8]}-it"
        assert page.title() == f"Run {rid[:8]}-it — re-skin v2"

        # And it really persisted, not just repainted.
        s = SessionFactory()
        assert s.get(Run, rid).language == "it"
        s.close()

    def test_switching_from_italian_back_to_english_drops_the_suffix(
        self, live_server, page, SessionFactory
    ):
        s = SessionFactory()
        p = _project(s)
        r = _run(s, p.id, language="it")
        rid = r.id
        s.close()

        page.goto(f"{live_server}/v2/runs/{rid}")
        page.wait_for_selector("#run-language")
        assert page.text_content("#run-short-id").strip() == f"{rid[:8]}-it"

        page.select_option("#run-language", "en")
        page.wait_for_function(
            "id => document.getElementById('run-short-id').textContent.trim() === id",
            arg=rid[:8],
        )
        assert page.text_content("#run-short-id").strip() == rid[:8]

        s = SessionFactory()
        assert s.get(Run, rid).language == "en"
        s.close()

    def test_the_selector_offers_italian(self, live_server, page, SessionFactory):
        s = SessionFactory()
        p = _project(s)
        r = _run(s, p.id)
        rid = r.id
        s.close()

        page.goto(f"{live_server}/v2/runs/{rid}")
        page.wait_for_selector("#run-language")
        values = page.eval_on_selector(
            "#run-language",
            "el => Array.from(el.options).map(o => o.value)",
        )
        assert values == ["en", "es", "pt", "ja", "de", "it", "pl", "fr"]


# ---------------------------------------------------------------------------
# 2. Localisation tail / voice auto-pick
# ---------------------------------------------------------------------------


def _localisation_project(session):
    """A ready localisation project with a hook swap + a tail keep segment."""
    from app import project_types

    p = _project(session, project_type=project_types.LOCALISATION)
    session.add_all([
        SegmentDef(
            project_id=p.id, index=0, start_sec=0.0, end_sec=6.0,
            has_face=True, action="swap", pre_roll_sec=0.0, post_roll_sec=0.0,
        ),
        SegmentDef(
            project_id=p.id, index=1, start_sec=6.0, end_sec=30.0,
            has_face=False, action="keep", pre_roll_sec=0.0, post_roll_sec=0.0,
        ),
    ])
    session.commit()
    return p


class TestLocalisationTailAutoPick:
    """pickLocalisationTail matches option[data-lang] against #run-language."""

    def test_italian_language_auto_picks_the_italian_tail(
        self, live_server, page, SessionFactory
    ):
        s = SessionFactory()
        p = _localisation_project(s)
        # Rendered newest-first, so the Italian clip must win over the others
        # regardless of insertion order.
        _clip(s, "Outro JA", "ja")
        it_clip = _clip(s, "Outro IT", "it")
        _clip(s, "Outro ES", "es")
        pid, it_id = p.id, it_clip.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#stitch-append-select")

        page.select_option("#run-language", "it")
        page.wait_for_function(
            "id => document.getElementById('stitch-append-select').value === id",
            arg=it_id,
        )
        assert page.input_value("#stitch-append-select") == it_id
        status = page.text_content("#stitch-tail-status")
        assert "auto-selected" in status.lower()
        assert "Outro IT" in status

    def test_no_italian_tail_warns_and_selects_nothing(
        self, live_server, page, SessionFactory
    ):
        """The existing missing-resource message — never a silent English tail."""
        s = SessionFactory()
        p = _localisation_project(s)
        _clip(s, "Outro EN only", "en")
        _clip(s, "Outro JA only", "ja")
        pid = p.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#stitch-append-select")

        page.select_option("#run-language", "it")
        page.wait_for_function(
            "() => document.getElementById('stitch-tail-status')"
            ".textContent.indexOf('No library clip') !== -1"
        )
        # Empty selection, not a fallback to the English or Japanese clip.
        assert page.input_value("#stitch-append-select") == ""
        status = page.text_content("#stitch-tail-status")
        assert "No library clip is tagged with this language" in status

    def test_a_hand_picked_tail_survives_a_switch_to_italian(
        self, live_server, page, SessionFactory
    ):
        """Manual choice wins — a later language change must not overwrite it."""
        s = SessionFactory()
        p = _localisation_project(s)
        _clip(s, "Outro IT", "it")
        en_clip = _clip(s, "Outro EN", "en")
        pid, en_id = p.id, en_clip.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#stitch-append-select")

        # Pick the English tail BY HAND (dispatch change, as a real click does).
        page.select_option("#stitch-append-select", en_id)
        page.wait_for_function(
            "() => document.getElementById('stitch-tail-status')"
            ".textContent.indexOf('by hand') !== -1"
        )

        page.select_option("#run-language", "it")
        page.wait_for_timeout(250)  # give any auto-pick a chance to misfire

        assert page.input_value("#stitch-append-select") == en_id
        assert "by hand" in page.text_content("#stitch-tail-status")


class TestLocalisationVoiceAutoPick:
    """pickLocalisationVoice — same contract, on the voice library."""

    def _voice(self, session, name, language) -> VoiceClip:
        v = VoiceClip(
            name=name, local_path=f"/tmp/{name}.mp3",
            duration_sec=8.0, language=language,
        )
        session.add(v)
        session.commit()
        return v

    def _select_a_voice_model(self, page):
        """Pick a model whose spec allows voice references, so the group shows."""
        from app import ai_models

        wanted = [
            key for key, spec in ai_models.AI_MODELS.items()
            if getattr(spec, "max_reference_audios", 0)
        ]
        if not wanted:
            pytest.skip("no model in the registry accepts a voice reference")
        page.select_option("#run-model", wanted[0])
        page.wait_for_function(
            "() => { const g = document.getElementById('voice-ref-group');"
            " return g && g.style.display !== 'none'; }"
        )
        return wanted[0]

    def test_italian_language_auto_picks_the_italian_voice(
        self, live_server, page, SessionFactory
    ):
        s = SessionFactory()
        p = _localisation_project(s)
        self._voice(s, "Kenji JA", "ja")
        it_voice = self._voice(s, "Giulia IT", "it")
        pid, it_id = p.id, it_voice.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#run-language")
        self._select_a_voice_model(page)

        page.select_option("#run-language", "it")
        page.wait_for_function(
            "id => document.getElementById('voice-clip-select').value === id",
            arg=it_id,
        )
        assert page.input_value("#voice-clip-select") == it_id
        # The "from the library" radio was switched on for us.
        assert page.eval_on_selector(
            "#new-run-form input[name='voice_mode'][value='library']",
            "el => el.checked",
        )
        assert "auto-selected" in page.text_content("#voice-ref-status").lower()

    def test_no_italian_voice_warns_and_picks_nothing(
        self, live_server, page, SessionFactory
    ):
        s = SessionFactory()
        p = _localisation_project(s)
        self._voice(s, "Kenji JA", "ja")
        self._voice(s, "Sam EN", "en")
        pid = p.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#run-language")
        self._select_a_voice_model(page)

        page.select_option("#run-language", "it")
        page.wait_for_function(
            "() => document.getElementById('voice-ref-status')"
            ".textContent.indexOf('No library voice') !== -1"
        )
        # The library radio was NOT switched on, so no voice is submitted.
        assert not page.eval_on_selector(
            "#new-run-form input[name='voice_mode'][value='library']",
            "el => el.checked",
        )
        assert "No library voice is tagged with this language" in page.text_content(
            "#voice-ref-status"
        )

    def test_switching_to_italian_drops_a_previously_auto_picked_voice(
        self, live_server, page, SessionFactory
    ):
        """Regression: the auto-picked English voice must not survive the switch.

        English is picked first, so an English voice is auto-picked
        (a localisation form starts with NO language). Switching to Italian (which has no voice) used to leave
        that English sample selected and submitted while displaying "the model
        will invent one" — the run would have been re-spoken in Italian by a
        voice sampled from the English take. Not Italian-specific: the same
        held for any language with no matching voice (German, say).
        """
        s = SessionFactory()
        p = _localisation_project(s)
        en_voice = self._voice(s, "Sam EN", "en")
        pid, en_id = p.id, en_voice.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#run-language")
        self._select_a_voice_model(page)
        page.select_option("#run-language", "en")

        # The English voice really was auto-picked first.
        page.wait_for_function(
            "id => document.getElementById('voice-clip-select').value === id",
            arg=en_id,
        )
        assert page.eval_on_selector(
            "#new-run-form input[name='voice_mode'][value='library']",
            "el => el.checked",
        )

        page.select_option("#run-language", "it")
        page.wait_for_function(
            "() => document.getElementById('voice-ref-status')"
            ".textContent.indexOf('No library voice') !== -1"
        )

        # Back to "none": nothing is submitted, so the warning is now true.
        assert page.eval_on_selector(
            "#new-run-form input[name='voice_mode'][value='none']",
            "el => el.checked",
        )
        assert page.input_value("#voice-clip-select") == ""
        assert page.eval_on_selector("#voice-clip-select", "el => el.disabled")

    def test_a_hand_picked_voice_survives_a_switch_to_italian(
        self, live_server, page, SessionFactory
    ):
        """The clearing above must never touch a DELIBERATE choice."""
        s = SessionFactory()
        p = _localisation_project(s)
        en_voice = self._voice(s, "Sam EN", "en")
        pid, en_id = p.id, en_voice.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#run-language")
        self._select_a_voice_model(page)
        page.select_option("#run-language", "en")
        page.wait_for_function(
            "id => document.getElementById('voice-clip-select').value === id",
            arg=en_id,
        )

        # Pick it BY HAND — the select's own onchange marks it deliberate.
        page.select_option("#voice-clip-select", en_id)
        page.wait_for_function(
            "() => document.getElementById('voice-ref-status')"
            ".textContent.indexOf('by hand') !== -1"
        )

        page.select_option("#run-language", "it")
        page.wait_for_timeout(300)  # let any auto-pick misfire

        assert page.input_value("#voice-clip-select") == en_id
        assert page.eval_on_selector(
            "#new-run-form input[name='voice_mode'][value='library']",
            "el => el.checked",
        )
        assert "by hand" in page.text_content("#voice-ref-status")


_READY_TRANSCRIPT = {
    "schema_version": 1, "model": "gemini-2.5-pro", "prompt_version": "transcribe/v1",
    "source_language": "en",
    "lines": [{"id": 1, "start": 0.0, "end": 2.4, "speaker": "woman",
               "on_screen": True, "text": "Hey, what are you doing?"}],
}


class TestLocalisationLanguageGate:
    """The Language select starts blank on a localisation form, and Translate
    and Start run both stay blocked until a language is chosen — so the hook
    is never translated (or exported) as English by accident."""

    def _ready(self, s):
        p = _localisation_project(s)
        p.transcript_status = "ready"
        p.transcript = _READY_TRANSCRIPT
        s.commit()
        return p.id

    def _btn_disabled(self, page):
        return page.eval_on_selector("#translate-btn", "el => el.disabled")

    def test_translate_is_off_until_a_language_is_chosen(
        self, live_server, page, SessionFactory
    ):
        s = SessionFactory()
        pid = self._ready(s)
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#translate-btn")
        assert page.input_value("#run-language") == ""
        assert self._btn_disabled(page)
        assert page.is_visible("#translate-language-reason")

        page.select_option("#run-language", "it")
        assert not self._btn_disabled(page)
        assert not page.is_visible("#translate-language-reason")

        page.select_option("#run-language", "")
        assert self._btn_disabled(page)
        assert page.is_visible("#translate-language-reason")

    def test_the_out_of_band_gate_refresh_keeps_translate_off(
        self, live_server, page, SessionFactory
    ):
        """The transcript poll re-renders the button enabled (it only knows
        the transcript); the form must switch it back off while blank."""
        s = SessionFactory()
        pid = self._ready(s)
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#translate-btn")
        page.eval_on_selector("#translate-btn", "el => el.dataset.oldGate = '1'")
        page.evaluate("pid => refreshTranscriptPanel(pid)", pid)
        page.wait_for_function(
            "() => { const b = document.getElementById('translate-btn');"
            " return b && !b.dataset.oldGate; }"
        )
        page.wait_for_timeout(200)
        assert self._btn_disabled(page)

        page.select_option("#run-language", "ja")
        assert not self._btn_disabled(page)

    def test_start_run_is_blocked_without_a_language(
        self, live_server, page, SessionFactory
    ):
        s = SessionFactory()
        p = _localisation_project(s)
        pid = p.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#new-run-form")
        page.check("#new-run-form input[name='localisation_mode'][value='keep']")
        page.click("#new-run-form button[type='submit']")
        page.wait_for_function(
            "() => document.getElementById('new-run-result')"
            ".textContent.indexOf('Choose the Language') !== -1"
        )
        s = SessionFactory()
        try:
            assert s.query(Run).filter(Run.project_id == pid).count() == 0
        finally:
            s.close()

    def test_a_blank_language_auto_picks_no_tail(
        self, live_server, page, SessionFactory
    ):
        """A blank must not "match" a language-agnostic (untagged) clip."""
        s = SessionFactory()
        p = _localisation_project(s)
        _clip(s, "Outro untagged", None)
        pid = p.id
        s.close()

        page.goto(f"{live_server}/v2/projects/{pid}")
        page.wait_for_selector("#stitch-append-select")
        assert page.input_value("#stitch-append-select") == ""
        assert "Choose the Language" in page.text_content("#stitch-tail-status")


# ---------------------------------------------------------------------------
# 4. AICP-152 — cross-site requests are refused, the dashboard's own are not
# ---------------------------------------------------------------------------


@pytest.fixture()
def foreign_site(live_server):
    """A second HTTP server on another port: an attacker's page.

    It auto-submits a multipart form at the dashboard's create-project
    endpoint — exactly the cross-site POST a browser re-sends basic-auth
    credentials on. Reached as ``localhost`` it is cross-site to the
    dashboard's ``127.0.0.1``; as ``127.0.0.1:<other port>`` it is same-site
    but still a foreign origin.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    html = (
        "<!doctype html><html><body>"
        f'<form id="f" method="post" enctype="multipart/form-data" '
        f'action="{live_server}/api/v2/projects">'
        '<input name="gdrive_link" value="https://drive.google.com/file/d/csrf-evil/view">'
        "</form><script>document.getElementById('f').submit()</script>"
        "</body></html>"
    ).encode()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _gdrive_projects(SessionFactory, marker):
    s = SessionFactory()
    try:
        return s.query(VideoProject).filter(
            VideoProject.source_ref.contains(marker)).count()
    finally:
        s.close()


class TestCrossSiteRequests:
    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
    def test_foreign_page_form_post_is_rejected(
        self, live_server, foreign_site, page, SessionFactory, host
    ):
        before = _gdrive_projects(SessionFactory, "csrf-evil")
        with page.expect_response(lambda r: r.url.endswith("/api/v2/projects")) as resp:
            page.goto(f"http://{host}:{foreign_site}/")
        assert resp.value.status == 403
        assert _gdrive_projects(SessionFactory, "csrf-evil") == before

    def test_dashboard_htmx_form_and_fetch_still_work(
        self, live_server, page, SessionFactory
    ):
        page.goto(f"{live_server}/v2/")
        page.click("#new-project-details summary")
        page.fill("#new-project-form input[name='gdrive_link']",
                  "https://drive.google.com/file/d/csrf-own/view")
        with page.expect_response(lambda r: r.url.endswith("/api/v2/projects")) as resp:
            page.click("#new-project-form button[type='submit']")
        assert resp.value.status == 201
        assert _gdrive_projects(SessionFactory, "csrf-own") == 1

        s = SessionFactory()
        pid = s.query(VideoProject).filter(
            VideoProject.source_ref.contains("csrf-own")).one().id
        s.close()
        # The success handler navigates to the new project's page; wait for
        # it, then a plain fetch() DELETE — the shape every delete button uses.
        page.wait_for_url(f"**/v2/projects/{pid}")
        status = page.evaluate(
            "pid => fetch('/api/v2/projects/' + pid, {method: 'DELETE'}).then(r => r.status)",
            pid,
        )
        assert status == 204
        assert _gdrive_projects(SessionFactory, "csrf-own") == 0
