"""AICP-156: webhook target lock-down."""

from __future__ import annotations

import logging
import os
import socket
import sys
import tempfile

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
import pytest

import app.automation as automation_mod
from app import webhook_safety as ws
from app.config import settings

SECRET = "00000000-1111-2222-3333-444444444444"
LIVE = f"https://hooks.example.com/webhook/{SECRET}?token=s3cret"


@pytest.fixture()
def live(monkeypatch):
    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_URL", LIVE)
    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_ALLOWED_HOSTS", "")


_REAL_CLIENT = httpx.Client


def _mock_http(monkeypatch, handler, seen=None):
    """Route httpx.Client through a MockTransport: the REAL request-building
    path (kwargs, extensions) runs, only the socket is faked."""
    real = _REAL_CLIENT

    def factory(**kw):
        return real(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(automation_mod.httpx, "Client", factory)


def _dns(monkeypatch, *addrs):
    def fake(host, port, *a, **k):
        return [(socket.AF_INET6 if ":" in x else socket.AF_INET, 1, 6, "", (x, port)) for x in addrs]

    monkeypatch.setattr(ws.socket, "getaddrinfo", fake)


def test_no_default_url_in_code():
    src = open(os.path.join(os.path.dirname(__file__), "..", "app", "config.py")).read()
    assert "hooks.example" not in src and "/webhook/" not in src


def test_default_allowlist_is_host_of_configured_url(live):
    assert ws.allowed_hosts() == {"hooks.example.com"}
    assert ws.validate_webhook_url(LIVE) == "hooks.example.com"


def test_explicit_allowlist(live, monkeypatch):
    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_ALLOWED_HOSTS", "a.example, B.example")
    assert ws.allowed_hosts() == {"a.example", "b.example"}
    with pytest.raises(ws.WebhookRefused):
        ws.validate_webhook_url(LIVE)


@pytest.mark.parametrize("url", [
    "http://hooks.example.com/x",
    "https://evil.example/x",
    "https://hooks.example.com.evil.example/x",
    "https://user:pw@hooks.example.com/x",
    "https://127.0.0.1/x",
    "https://169.254.169.254/latest",
    "https://[::1]/x",
    "ftp://hooks.example.com/x",
    "https://hooks.example.com:bad/x",
    "https://hooks.example.com:0/x",
])
def test_static_refusals(live, url):
    with pytest.raises(ws.WebhookRefused):
        ws.validate_webhook_url(url)


def test_nothing_allowed_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_URL", "")
    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_ALLOWED_HOSTS", "")
    with pytest.raises(ws.WebhookRefused):
        ws.validate_webhook_url("https://hook.test/x")


@pytest.mark.parametrize("addr", [
    "127.0.0.1", "10.1.2.3", "172.16.0.5", "192.168.1.1", "169.254.169.254",
    "::1", "fc00::1", "fe80::1", "fec0::1", "::ffff:127.0.0.1", "0.0.0.0", "100.64.0.1",
])
def test_hostname_resolving_to_private_is_refused(live, monkeypatch, addr):
    _dns(monkeypatch, addr)
    with pytest.raises(ws.WebhookRefused):
        ws.pinned_targets(LIVE)


def test_any_private_address_among_several_refuses(live, monkeypatch):
    _dns(monkeypatch, "93.184.216.34", "10.0.0.1")
    with pytest.raises(ws.WebhookRefused):
        ws.pinned_targets(LIVE)


def test_live_style_url_accepted_and_pinned(live, monkeypatch):
    _dns(monkeypatch, "93.184.216.34")
    [(ip_url, host, port, ip)] = ws.pinned_targets(LIVE)
    assert host == "hooks.example.com" and port == 443 and ip == "93.184.216.34"
    assert ip_url.startswith("https://93.184.216.34/webhook/")


def test_send_pins_ip_no_redirect_timeout(live, monkeypatch):
    _dns(monkeypatch, "93.184.216.34")
    reqs = []

    def handler(req):
        reqs.append(req)
        if req.url.host == "93.184.216.34":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/"})
        raise AssertionError("redirect followed")

    _mock_http(monkeypatch, handler)
    assert automation_mod._post_webhook(LIVE, {"a": 1}) is False  # 3xx = failure
    assert len(reqs) == 1
    assert reqs[0].headers["Host"] == "hooks.example.com"
    assert reqs[0].extensions["sni_hostname"] == "hooks.example.com"
    assert reqs[0].url.scheme == "https"


def test_send_success(live, monkeypatch):
    _dns(monkeypatch, "93.184.216.34")
    _mock_http(monkeypatch, lambda req: httpx.Response(200))
    assert automation_mod._post_webhook(LIVE, {}) is True


def test_send_refuses_rebound_host_without_connecting(live, monkeypatch):
    _dns(monkeypatch, "127.0.0.1")

    def boom(*a, **k):
        raise AssertionError("must not connect")

    _mock_http(monkeypatch, boom)
    with pytest.raises(ws.WebhookRefused):
        automation_mod._post_webhook(LIVE, {})


def test_logs_have_no_token_path_or_body(live, monkeypatch, caplog):
    _dns(monkeypatch, "93.184.216.34")
    _mock_http(monkeypatch, lambda req: httpx.Response(500, text=f"echo {SECRET} s3cret"))
    caplog.set_level(logging.DEBUG)
    assert automation_mod._post_webhook(LIVE, {"token": "tok123"}) is False

    def err(req):
        raise httpx.ConnectError(f"cannot reach {req.url}")

    _mock_http(monkeypatch, err)
    automation_mod._post_webhook(LIVE, {"token": "tok123"})
    text = caplog.text
    assert "hooks.example.com" in text
    for bad in (SECRET, "s3cret", "/webhook", "tok123", "echo"):
        assert bad not in text


def test_dns_failure_is_retryable_not_refused(live, monkeypatch):
    def fail(*a, **k):
        raise socket.gaierror(-3, "Temporary failure")

    monkeypatch.setattr(ws.socket, "getaddrinfo", fail)
    _mock_http(monkeypatch, lambda req: 1 / 0)
    assert automation_mod._post_webhook(LIVE, {}) is False  # no raise -> retried


def test_falls_through_dead_address_to_next(live, monkeypatch):
    _dns(monkeypatch, "2606:2800::1", "93.184.216.34")
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        if ":" in req.url.host:
            raise httpx.ConnectTimeout("dead")
        return httpx.Response(200)

    _mock_http(monkeypatch, handler)
    assert automation_mod._post_webhook(LIVE, {}) is True
    assert len(hosts) == 2


def test_client_ignores_proxy_env(live, monkeypatch):
    _dns(monkeypatch, "93.184.216.34")
    kws = []
    real = _REAL_CLIENT

    def factory(**kw):
        kws.append(kw)
        return real(transport=httpx.MockTransport(lambda r: httpx.Response(200)), **kw)

    monkeypatch.setattr(automation_mod.httpx, "Client", factory)
    assert automation_mod._post_webhook(LIVE, {}) is True
    assert kws[0]["trust_env"] is False
