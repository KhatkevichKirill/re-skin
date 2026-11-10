"""
Cross-site request rejection for state-changing endpoints (AICP-152).

The dashboard's only auth is nginx basic auth, and browsers re-attach cached
basic-auth credentials to cross-site form POSTs. Without a check, any page an
operator visits could launch runs (spending kie.ai credits), export to Sheets,
overwrite delivered Drive files, delete things, etc.

Rule, applied to every unsafe method (anything but GET/HEAD/OPTIONS/TRACE):

1. ``Sec-Fetch-Site: cross-site``                      -> 403.
2. ``Origin`` present and not a trusted origin          -> 403 (``null`` included).
3. No ``Origin`` but ``Referer`` present and untrusted   -> 403.
4. Neither ``Origin`` nor ``Referer``                    -> allowed.

Trusted origins are: the request's own ``Host`` (host + port, the port
defaulting by ``X-Forwarded-Proto``/scheme when absent), ``PUBLIC_BASE_URL``,
and any ``CSRF_TRUSTED_ORIGINS`` entries. Host/port compare exactly -- a page
served from another port on the same box is a different origin and is refused.

Policy for curl / scripts / server-to-server callers: they send neither
``Origin`` nor ``Referer`` (and no ``Sec-Fetch-*``), so rule 4 lets them
through. That is safe because a browser always sends ``Origin`` on a
cross-origin POST/PUT/PATCH/DELETE (fetch, XHR and HTML forms alike), so a
forged browser request can never reach rule 4. A script that does set
``Origin`` must set the dashboard's own origin.

GET behaviour is unchanged. ``/public/*`` is GET-only, so it needs no
exclusion.

Behind nginx: nginx forwards ``Host`` (``$http_host`` keeps the :8847 port);
with an older config that sends ``$host`` (port stripped) the browser's
``Origin: http://<ip>:8847`` still matches via ``PUBLIC_BASE_URL``.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, Optional, Tuple
from urllib.parse import urlsplit

from .config import settings

logger = logging.getLogger(__name__)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

# (scheme or None, hostname, port)
_Origin = Tuple[Optional[str], str, int]


def _parse_origin(value: str) -> Optional[_Origin]:
    """Parse an Origin / Referer / base-URL value; None if unusable."""
    value = (value or "").strip()
    if not value or value == "null":
        return None
    try:
        parts = urlsplit(value)
        scheme = (parts.scheme or "").lower()
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if port is None:
        port = _DEFAULT_PORTS[scheme]
    return scheme, host, port


def _parse_host(host_header: str, scheme: str) -> Optional[_Origin]:
    """Parse a Host header (``name[:port]``, IPv6 in brackets)."""
    host_header = (host_header or "").strip()
    if not host_header:
        return None
    try:
        parts = urlsplit("//" + host_header)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if not host:
        return None
    if port is None:
        port = _DEFAULT_PORTS.get(scheme, 80)
    # Scheme is not compared for the Host match: behind the proxy we cannot
    # be sure of it, and host+port already pins the listener.
    return None, host, port


def _configured_origins() -> Iterable[_Origin]:
    raw = [settings.PUBLIC_BASE_URL]
    raw.extend((getattr(settings, "CSRF_TRUSTED_ORIGINS", "") or "").split(","))
    for item in raw:
        parsed = _parse_origin(item)
        if parsed is not None:
            yield parsed


def _is_trusted(candidate: _Origin, host_origin: Optional[_Origin]) -> bool:
    _, c_host, c_port = candidate
    if host_origin is not None and (c_host, c_port) == host_origin[1:]:
        return True
    return any(candidate == trusted for trusted in _configured_origins())


def check_request(method: str, headers: dict, scope_scheme: str = "http") -> Optional[str]:
    """Return a rejection reason for this request, or None to allow it.

    ``headers`` maps lower-case header names to (latin-1 decoded) values.
    """
    if method.upper() in SAFE_METHODS:
        return None

    if headers.get("sec-fetch-site", "").strip().lower() == "cross-site":
        return "Sec-Fetch-Site: cross-site"

    origin = headers.get("origin")
    referer = headers.get("referer")
    if origin is None and referer is None:
        return None  # curl / scripts / server-to-server; see module docstring

    forwarded_proto = headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    host_origin = _parse_host(headers.get("host", ""), forwarded_proto or scope_scheme)

    if origin is not None:
        parsed = _parse_origin(origin)
        if parsed is None or not _is_trusted(parsed, host_origin):
            return f"Origin {origin!r} not trusted"
        return None

    parsed = _parse_origin(referer or "")
    if parsed is None or not _is_trusted(parsed, host_origin):
        return f"Referer {referer!r} not trusted"
    return None


class CrossSiteRequestMiddleware:
    """Pure ASGI middleware (no BaseHTTPMiddleware: uploads stream through)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers: dict = {}
        for name, value in scope.get("headers") or []:
            key = name.decode("latin-1").lower()
            # Duplicate headers: the first copy wins (browsers never send two).
            headers.setdefault(key, value.decode("latin-1"))

        reason = check_request(scope.get("method", "GET"), headers,
                               scope.get("scheme", "http"))
        if reason is None:
            await self.app(scope, receive, send)
            return

        logger.warning("CSRF: rejected %s %s (%s)", scope.get("method"),
                       scope.get("path"), reason)
        body = json.dumps({"detail": "Cross-site request rejected"}).encode()
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
