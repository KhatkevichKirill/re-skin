"""AICP-152: cross-site POST/PUT/PATCH/DELETE are rejected (app/csrf.py)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import csrf
from app.config import settings
from app.csrf import CrossSiteRequestMiddleware, check_request

PUBLIC = "http://203.0.113.10:8847"


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", PUBLIC, raising=False)
    monkeypatch.setattr(settings, "CSRF_TRUSTED_ORIGINS", "", raising=False)


def _app():
    app = FastAPI()
    app.add_middleware(CrossSiteRequestMiddleware)

    @app.api_route("/thing", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def thing():
        return {"ok": True}

    return app


@pytest.fixture()
def client():
    # Host as nginx forwards it with $http_host.
    return TestClient(_app(), base_url=PUBLIC)


# ── middleware over HTTP ──────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_cross_site_unsafe_methods_rejected(client, method):
    r = client.request(method, "/thing", headers={
        "Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert r.json() == {"detail": "Cross-site request rejected"}


def test_cross_site_form_post_with_body_rejected(client):
    r = client.post("/thing", data={"prompt": "x"}, headers={
        "Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_same_origin_htmx_post_ok(client):
    r = client.post("/thing", headers={
        "Origin": PUBLIC, "Sec-Fetch-Site": "same-origin",
        "Referer": PUBLIC + "/v2/projects/abc", "HX-Request": "true"})
    assert r.status_code == 200


def test_same_origin_fetch_delete_ok(client):
    r = client.delete("/thing", headers={"Origin": PUBLIC, "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 200


def test_no_headers_post_ok(client):
    """curl / scripts / in-container uploads send no Origin/Referer/Sec-Fetch."""
    assert client.post("/thing", data={"a": "b"}).status_code == 200


def test_cross_site_get_unchanged(client):
    r = client.get("/thing", headers={
        "Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 200


def test_origin_mismatch_without_sec_fetch_rejected(client):
    """Old browsers without Sec-Fetch-*: the Origin check alone rejects."""
    r = client.post("/thing", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_null_origin_rejected(client):
    assert client.post("/thing", headers={"Origin": "null"}).status_code == 403


def test_same_site_other_port_rejected(client):
    """Another service on the same box is same-site, but not same-origin."""
    r = client.post("/thing", headers={
        "Origin": "http://203.0.113.10:9000", "Sec-Fetch-Site": "same-site"})
    assert r.status_code == 403


def test_referer_fallback(client):
    assert client.post("/thing", headers={"Referer": PUBLIC + "/v2/"}).status_code == 200
    assert client.post("/thing", headers={"Referer": "http://evil.example/x"}).status_code == 403


def test_full_app_registers_middleware():
    from app.main import app
    c = TestClient(app, base_url=PUBLIC)
    r = c.post("/api/v2/projects", headers={
        "Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert c.get("/health", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


# ── rule unit tests (port handling) ───────────────────────────────────────


def _h(**kw):
    return {k.replace("_", "-"): v for k, v in kw.items()}


def test_host_without_port_falls_back_to_public_base_url():
    """Legacy nginx `$host` strips :8847; PUBLIC_BASE_URL still matches."""
    hdrs = _h(host="203.0.113.10", origin=PUBLIC)
    assert check_request("POST", hdrs) is None


def test_host_without_port_and_no_public_base_url(monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
    # Host ip (port 80 by default) vs Origin :8847 -> different origin.
    assert check_request("POST", _h(host="203.0.113.10", origin=PUBLIC)) is not None
    # Explicit-port Host matches.
    assert check_request("POST", _h(host="203.0.113.10:8847", origin=PUBLIC)) is None


def test_host_match_any_hostname(monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
    assert check_request("POST", _h(host="localhost:8847", origin="http://localhost:8847")) is None
    assert check_request("POST", _h(host="LocalHost:8847", origin="http://localhost:8847")) is None
    assert check_request("POST", _h(host="localhost:8847", origin="http://localhost:8848")) is not None


def test_default_ports_follow_scheme(monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
    assert check_request("POST", _h(host="reskin.example", origin="http://reskin.example")) is None
    assert check_request("POST", _h(host="reskin.example", origin="https://reskin.example",
                                    x_forwarded_proto="https")) is None
    assert check_request("POST", _h(host="reskin.example:443", origin="https://reskin.example")) is None
    assert check_request("POST", _h(host="reskin.example", origin="https://reskin.example")) is not None


def test_public_base_url_requires_exact_scheme_and_port():
    assert check_request("POST", _h(host="x", origin="https://203.0.113.10:8847")) is not None
    assert check_request("POST", _h(host="x", origin="http://203.0.113.10")) is not None


def test_trusted_origins_setting(monkeypatch):
    monkeypatch.setattr(settings, "CSRF_TRUSTED_ORIGINS", " http://localhost:8847 , https://a.example")
    assert check_request("POST", _h(host="api:8000", origin="http://localhost:8847")) is None
    assert check_request("POST", _h(host="api:8000", origin="https://a.example")) is None
    assert check_request("POST", _h(host="api:8000", origin="https://b.example")) is not None


def test_ipv6_host():
    assert check_request("POST", _h(host="[::1]:8847", origin="http://[::1]:8847")) is None


@pytest.mark.parametrize("bad", ["garbage", "file:///etc", "http://", "http://h:99999", "javascript:x"])
def test_unparseable_origin_rejected(bad):
    assert check_request("POST", _h(host="203.0.113.10:8847", origin=bad)) is not None


def test_sec_fetch_none_and_same_site_allowed_when_origin_ok():
    for sfs in ("none", "same-site", "same-origin"):
        assert check_request("POST", _h(host="x", origin=PUBLIC, sec_fetch_site=sfs)) is None


def test_safe_methods_never_checked():
    for m in ("GET", "HEAD", "OPTIONS", "TRACE", "get"):
        assert check_request(m, _h(origin="https://evil.example", sec_fetch_site="cross-site")) is None


def test_unknown_method_is_unsafe():
    assert check_request("PROPFIND", _h(origin="https://evil.example")) is not None


def test_module_exports():
    assert "POST" not in csrf.SAFE_METHODS


# ── AICP-150: the :443 front door (sslip.io name) ─────────────────────────

TLS_ORIGIN = "https://203-0-113-10.sslip.io"


def _tls_headers(origin):
    # What nginx's :443 listener forwards: $http_host has no port, and
    # X-Forwarded-Proto is https.
    return {"host": "203-0-113-10.sslip.io", "x-forwarded-proto": "https",
            "origin": origin, "sec-fetch-site": "same-origin"}


def test_https_front_door_trusted_by_host_before_public_base_url_flip():
    # PUBLIC_BASE_URL still the legacy http one (autouse fixture).
    assert check_request("POST", _tls_headers(TLS_ORIGIN)) is None


def test_https_front_door_rejects_http_origin_on_same_name():
    # http://<name> is port 80 -- a different origin from the :443 page.
    assert check_request("POST", _tls_headers("http://203-0-113-10.sslip.io"))


def test_legacy_8847_still_trusted_after_public_base_url_flip(monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", TLS_ORIGIN, raising=False)
    assert check_request("POST", {
        "host": "203.0.113.10:8847", "x-forwarded-proto": "http",
        "origin": PUBLIC}) is None
    assert check_request("POST", _tls_headers(TLS_ORIGIN)) is None
