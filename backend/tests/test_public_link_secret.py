"""AICP-151: dedicated, fail-closed signing secret for /public links."""

import hashlib
import hmac
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import public
from app.config import settings

NEW = "N" * 20 + "ew-secret-" + "0123456789abcdef"
OLD = "O" * 20 + "ld-secret-" + "0123456789abcdef"
BASIC_PASS = "hunter2-basic-auth-password-that-is-long-enough"


@pytest.fixture()
def client(monkeypatch):
    import tempfile

    from app import storage

    os.makedirs(storage.data_dir_root(), exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="aicp151-", dir=storage.data_dir_root())) / "final.mp4"
    out.write_bytes(b"resultbytes")

    class _DB:
        def get(self, model, ident):
            return SimpleNamespace(
                result_local_path=str(out),
                source_local_path=str(out),
                output_path=str(out),
            )

    @contextmanager
    def _session():
        yield _DB()

    monkeypatch.setattr(public, "get_session", _session)
    monkeypatch.setattr(public, "_last_logged", {})
    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", NEW, raising=False)
    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", "", raising=False)
    monkeypatch.setattr(settings, "BASIC_AUTH_PASS", BASIC_PASS, raising=False)
    app = FastAPI()
    app.include_router(public.router, prefix="/public")
    yield TestClient(app)
    out.unlink(missing_ok=True)


def _sign(secret, purpose, ident):
    return hmac.new(secret.encode(), f"{purpose}:{ident}".encode(), hashlib.sha256).hexdigest()


ENDPOINTS = [
    ("/public/projects/p1/source", "project-source", "p1"),
    ("/public/runs/r1/result", "run-result", "r1"),
    ("/public/tail-edits/t1/result", "tail-edit-result", "t1"),
    ("/public/captions/c1/result", "caption-result", "c1"),
]


@pytest.mark.parametrize("url,purpose,ident", ENDPOINTS)
class TestEndpoints:
    def test_new_secret_token_ok(self, client, url, purpose, ident):
        r = client.get(url, params={"token": _sign(NEW, purpose, ident)})
        assert r.status_code == 200 and r.content == b"resultbytes"

    def test_tampered_token_403(self, client, url, purpose, ident):
        tok = _sign(NEW, purpose, ident)
        bad = ("0" if tok[0] != "0" else "1") + tok[1:]
        assert client.get(url, params={"token": bad}).status_code == 403
        assert client.get(url, params={"token": "é"}).status_code == 403
        assert client.get(url, params={"token": ""}).status_code == 403

    def test_basic_auth_pass_signed_token_rejected(self, client, url, purpose, ident):
        r = client.get(url, params={"token": _sign(BASIC_PASS, purpose, ident)})
        assert r.status_code == 403

    def test_empty_key_token_rejected(self, client, url, purpose, ident):
        tok = hmac.new(b"", f"{purpose}:{ident}".encode(), hashlib.sha256).hexdigest()
        assert client.get(url, params={"token": tok}).status_code == 403

    def test_cross_purpose_rejected(self, client, url, purpose, ident):
        r = client.get(url, params={"token": _sign(NEW, "other-purpose", ident)})
        assert r.status_code == 403

    @pytest.mark.parametrize("secret", ["", "   ", "short-secret", "x" * 31])
    def test_missing_or_short_secret_503(self, client, monkeypatch, url, purpose, ident, secret):
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", secret, raising=False)
        # Even a token forged with that empty/short key must not work.
        forged = _sign(secret.strip(), purpose, ident)
        assert client.get(url, params={"token": forged}).status_code == 503

    def test_previous_secret_verify_only(self, client, monkeypatch, url, purpose, ident):
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", OLD, raising=False)
        assert client.get(url, params={"token": _sign(OLD, purpose, ident)}).status_code == 200
        assert client.get(url, params={"token": _sign(NEW, purpose, ident)}).status_code == 200

    def test_previous_unset_means_no_transition(self, client, url, purpose, ident):
        assert client.get(url, params={"token": _sign(OLD, purpose, ident)}).status_code == 403

    def test_previous_alone_is_not_enough(self, client, monkeypatch, url, purpose, ident):
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", "", raising=False)
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", OLD, raising=False)
        assert client.get(url, params={"token": _sign(OLD, purpose, ident)}).status_code == 503


class TestMinting:
    def test_mint_matches_new_secret(self, client):
        assert public.make_result_token("r1") == _sign(NEW, "run-result", "r1")
        assert public.make_source_token("p1") == _sign(NEW, "project-source", "p1")
        assert public.make_tail_edit_result_token("t1") == _sign(NEW, "tail-edit-result", "t1")
        assert public.make_caption_result_token("c1") == _sign(NEW, "caption-result", "c1")

    def test_never_mints_with_previous(self, client, monkeypatch):
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", OLD, raising=False)
        assert public.make_result_token("r1") != _sign(OLD, "run-result", "r1")

    @pytest.mark.parametrize("secret", ["", "short"])
    def test_refuses_to_mint_and_logs(self, client, monkeypatch, caplog, secret):
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", secret, raising=False)
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", OLD, raising=False)
        with caplog.at_level("ERROR", logger="app.public"):
            assert public.make_result_token("r1") == ""
            assert public.make_source_token("p1") == ""
        assert "PUBLIC_LINK_SECRET" in caplog.text
        assert OLD not in caplog.text
        assert public.public_links_enabled() is False

    def test_short_previous_ignored(self, client, monkeypatch):
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", "short", raising=False)
        assert public._previous_keys() == []

    def test_no_basic_auth_fallback_in_config(self):
        # Fresh interpreter: reloading config in-process would swap the
        # `settings` object other test modules already hold.
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "PUBLIC_LINK_SECRET"}
        # Empty, not absent: config's load_dotenv() never overrides a set
        # variable, so a real .env next to the checkout (the documented test
        # setup symlinks prod's) cannot leak its secret into this check.
        env["PUBLIC_LINK_SECRET"] = ""
        env["BASIC_AUTH_PASS"] = BASIC_PASS
        out = subprocess.run(
            [sys.executable, "-c",
             "from app.config import settings; print(repr(settings.PUBLIC_LINK_SECRET))"],
            env=env, capture_output=True, text=True, check=True,
        ).stdout.strip()
        # No `assert out == ...`: pytest would print a configured secret.
        fell_back = out != "''"
        assert not fell_back, "PUBLIC_LINK_SECRET was not empty (value withheld)"


class TestAppBootsWithoutSecret:
    def test_app_boots_and_dashboard_survives(self, monkeypatch):
        monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", "", raising=False)
        from app.main import app

        c = TestClient(app)
        assert c.get("/public/runs/r1/result", params={"token": "x"}).status_code == 503
        assert c.get("/favicon.ico").status_code == 200


def test_webhook_omits_result_url_without_secret(monkeypatch):
    from app import automation

    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://example.test", raising=False)
    run = SimpleNamespace(id="r1", result_gdrive_file_id=None, result_local_path="/x.mp4")
    assert automation._run_links(run)["result_url"].startswith(
        "https://example.test/public/runs/r1/result?token="
    )
    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", "", raising=False)
    links = automation._run_links(run)
    assert links["result_url"] is None
    assert links["run_url"] == "https://example.test/v2/runs/r1"


def test_missing_token_param_is_503_not_422_when_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", "", raising=False)
    assert client.get("/public/runs/r1/result").status_code == 503
    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", NEW, raising=False)
    assert client.get("/public/runs/r1/result").status_code == 403


def test_overlay_preview_url_none_without_secret(monkeypatch):
    from app import web_overlay

    job = SimpleNamespace(
        source_type="run", source_run_id="r1", source_tail_edit_id=None, source_caption_id=None
    )
    assert "token=" in web_overlay._source_preview_url(job, None)
    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", "", raising=False)
    assert web_overlay._source_preview_url(job, None) is None


def test_overlay_source_item_allows_no_preview_url():
    from datetime import datetime

    from app.schemas_overlay import OverlaySourceItem

    item = OverlaySourceItem(
        run_id="r1", run_name=None, project_name=None, created_at=datetime.now(), preview_url=None
    )
    assert item.preview_url is None
