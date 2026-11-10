"""AICP-274: the service-account e-mail shown next to every Drive input.

The address is read from the key file (settings.GOOGLE_SERVICE_ACCOUNT_FILE),
lazily and fail-safe, and exposed to templates as the `drive_sa_email` Jinja
global on every templates environment.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app import drive_sa
from app.config import settings

EMAIL = "reskin-sa@example-project.iam.gserviceaccount.com"
SECRET = "-----BEGIN PRIVATE KEY-----do-not-leak"


@pytest.fixture(autouse=True)
def _fresh_cache():
    drive_sa.reset_cache()
    yield
    drive_sa.reset_cache()


def _key_file(tmp_path, monkeypatch, payload=None, raw=None):
    p = tmp_path / "sa.json"
    p.write_text(raw if raw is not None else json.dumps(
        payload if payload is not None else {"client_email": EMAIL, "private_key": SECRET}
    ))
    monkeypatch.setattr(settings, "GOOGLE_SERVICE_ACCOUNT_FILE", str(p))
    return p


def _envs():
    from app import web_overlay, web_v2

    return {"v2": web_v2.templates.env, "overlay": web_overlay.templates.env}


def _render_notice(env, **ctx):
    return env.get_template("_drive_sa_notice.html").render(**ctx)


class TestServiceAccountEmail:
    def test_reads_client_email_from_key_file(self, tmp_path, monkeypatch):
        _key_file(tmp_path, monkeypatch)
        assert drive_sa.service_account_email() == EMAIL

    def test_missing_file_is_none_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "GOOGLE_SERVICE_ACCOUNT_FILE", str(tmp_path / "nope.json"))
        assert drive_sa.service_account_email() is None

    @pytest.mark.parametrize("raw", ["", "not json", "[]", '{"client_email": 5}', '{"client_email": "  "}', "{}"])
    def test_unreadable_or_malformed_is_none(self, tmp_path, monkeypatch, raw):
        _key_file(tmp_path, monkeypatch, raw=raw)
        assert drive_sa.service_account_email() is None

    def test_failure_is_retried_once_the_file_is_fixed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "GOOGLE_SERVICE_ACCOUNT_FILE", str(tmp_path / "sa.json"))
        assert drive_sa.service_account_email() is None
        _key_file(tmp_path, monkeypatch)  # fixed after startup
        assert drive_sa.service_account_email() == EMAIL

    def test_success_is_cached_until_the_file_changes(self, tmp_path, monkeypatch):
        p = _key_file(tmp_path, monkeypatch)
        calls = []
        real = drive_sa._read_client_email
        monkeypatch.setattr(drive_sa, "_read_client_email", lambda path: calls.append(path) or real(path))
        assert drive_sa.service_account_email() == EMAIL
        assert drive_sa.service_account_email() == EMAIL
        assert len(calls) == 1
        # Key rotated in place: new account, new mtime -> picked up, no restart.
        p.write_text(json.dumps({"client_email": "new@x.iam.gserviceaccount.com"}))
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        assert drive_sa.service_account_email() == "new@x.iam.gserviceaccount.com"
        # File removed: no stale address.
        os.remove(p)
        assert drive_sa.service_account_email() is None

    def test_is_registered_on_every_templates_env(self):
        for name, env in _envs().items():
            assert env.globals.get("drive_sa_email") is drive_sa.service_account_email, name


class TestNoticePartial:
    @pytest.mark.parametrize("name", ["v2", "overlay"])
    def test_renders_email_and_copy_button_from_key_file(self, name, tmp_path, monkeypatch):
        _key_file(tmp_path, monkeypatch)
        html = _render_notice(_envs()[name])
        assert EMAIL in html
        assert f'data-copy-text="{EMAIL}"' in html
        assert "<button" in html and "Copy" in html
        assert "onclick" not in html and "<script" not in html  # CSP-friendly
        assert SECRET not in html

    def test_editor_role_asks_for_editor(self, tmp_path, monkeypatch):
        _key_file(tmp_path, monkeypatch)
        env = _envs()["v2"]
        assert "Editor access" in _render_notice(env, drive_sa_role="editor")
        viewer = _render_notice(env)
        assert "Viewer" in viewer and "Editor for delivery folders" in viewer

    def test_missing_key_file_degrades_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "GOOGLE_SERVICE_ACCOUNT_FILE", str(tmp_path / "nope.json"))
        for env in _envs().values():
            html = _render_notice(env)
            assert "must be shared with" in html
            assert "service account" in html
            assert "<button" not in html and "data-copy-text" not in html

    def test_email_is_html_escaped(self, tmp_path, monkeypatch):
        _key_file(tmp_path, monkeypatch, payload={"client_email": 'x"><script>alert(1)</script>@a.b'})
        html = _render_notice(_envs()["v2"])
        assert "<script>alert" not in html

    def test_renders_when_global_is_missing(self):
        # A bare environment (e.g. a future one that forgot to register the
        # global) must not blow up the page.
        import jinja2

        base = os.path.join(os.path.dirname(drive_sa.__file__), "templates")
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(base), autoescape=True)
        html = env.get_template("_drive_sa_notice.html").render()
        assert "must be shared with" in html and "<button" not in html


def test_copy_script_is_static_and_loaded_by_both_bases():
    root = os.path.join(os.path.dirname(drive_sa.__file__))
    js = os.path.join(root, "static", "drive_sa_notice.js")
    assert os.path.exists(js)
    for base in ("v2", "overlay"):
        with open(os.path.join(root, "templates", base, "base.html")) as fh:
            assert "/static/drive_sa_notice.js" in fh.read(), base


# Every input that takes a Drive link/ID (or a URL list that resolves Drive
# share links via pipeline_common.resolve_reference_urls) -> the notice must
# follow it. Source-level so inputs that only render in rare run states are
# covered too; the page-level tests in test_web_v2.py cover the common ones.
_DRIVE_INPUTS = [
    ("v2/dashboard.html", 'name="gdrive_link"'),
    ("v2/library.html", 'id="lib-drive-url"'),
    ("v2/library.html", 'id="ovlib-drive-url"'),
    ("v2/library.html", 'id="voice-drive-url"'),
    ("overlay/editor.html", 'id="drive-link"'),
    ("v2/partials/project_status_content.html", 'name="gdrive_folder_id"'),
    ("v2/partials/project_status_content.html", 'id="reference-urls"'),
    ("v2/partials/project_status_content.html", 'name="preprocess_reference_url"'),
    ("v2/partials/run_status_content.html", 'id="regenerate-preprocess-reference-url"'),
    ("v2/partials/run_status_content.html", 'placeholder="URL1, URL2 (comma-separated)"'),
    ("v2/partials/segment_image_block.html", 'name="character_url"'),
    ("v2/partials/knowledge_feedback.html", 'name="drive_link"'),
    ("v2/run_detail.html", 'class="copy-ref-url"'),
]


@pytest.mark.parametrize("rel,marker", _DRIVE_INPUTS)
def test_every_drive_input_is_followed_by_the_notice(rel, marker):
    root = os.path.join(os.path.dirname(drive_sa.__file__), "templates")
    with open(os.path.join(root, rel)) as fh:
        src = fh.read()
    assert src.count(marker) == 1, (rel, marker)
    window = src[src.index(marker) : src.index(marker) + 700]
    assert '_drive_sa_notice.html' in window, (rel, marker)


def test_delivery_folder_notice_asks_for_editor():
    root = os.path.join(os.path.dirname(drive_sa.__file__), "templates")
    with open(os.path.join(root, "v2/partials/project_status_content.html")) as fh:
        src = fh.read()
    i = src.index('name="gdrive_folder_id"')
    assert 'drive_sa_role="editor"' in src[i : i + 400]
