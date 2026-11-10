"""Outbound-webhook guard (AICP-156).

Rules, shared by the API (validate on save) and the sender (re-checked on
every send, because DNS can change between save and send):

* https only, no userinfo, no IP-literal hosts;
* host must be in the allow-list (``AUTOMATION_WEBHOOK_ALLOWED_HOSTS``,
  defaulting to the host of ``AUTOMATION_WEBHOOK_URL``);
* at send time the host is resolved and EVERY address must be public; the
  request then connects to the vetted IP itself (Host header + SNI keep the
  name), so a second, rebinding lookup never happens;
* redirects are never followed (a 3xx is a failed delivery).

Logging helpers never emit path, query or response body.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlsplit

from .config import settings

# Ranges that are not "global" in the ipaddress sense but we want named anyway;
# is_global already excludes 10/8, 172.16/12, 192.168/16, 127/8, 169.254/16
# (incl. 169.254.169.254), ::1, fc00::/7, fe80::/10, 100.64/10, multicast etc.


class WebhookRefused(Exception):
    """The URL breaks the webhook policy. Permanent: retrying cannot help."""


class WebhookUnresolved(Exception):
    """DNS failed (possibly transient). Not a policy breach: retry next tick."""


def configured_url() -> str:
    return (settings.AUTOMATION_WEBHOOK_URL or "").strip()


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def redact(url: str) -> str:
    """``https://host`` only: the path/query of a webhook is usually the secret."""
    host = host_of(url)
    return f"https://{host}" if host else "<invalid url>"


def allowed_hosts() -> set[str]:
    raw = (settings.AUTOMATION_WEBHOOK_ALLOWED_HOSTS or "").strip()
    hosts = {h.strip().lower() for h in raw.split(",") if h.strip()}
    if not hosts:
        default = host_of(configured_url())
        if default:
            hosts.add(default)
    return hosts


def validate_webhook_url(url: str) -> str:
    """Static checks (no DNS). Returns the lower-cased host or raises."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        port = parts.port  # raises ValueError on a bad port
    except ValueError:
        raise WebhookRefused("webhook URL is not valid")
    if port is not None and not 1 <= port <= 65535:
        raise WebhookRefused("webhook URL has an invalid port")
    if parts.scheme != "https":
        raise WebhookRefused("webhook URL must use https")
    if not host:
        raise WebhookRefused("webhook URL has no host")
    if parts.username is not None or parts.password is not None:
        raise WebhookRefused("webhook URL must not contain credentials")
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        raise WebhookRefused("webhook host must be a name, not an IP address")
    if host not in allowed_hosts():
        raise WebhookRefused(f"webhook host {host!r} is not in the allow-list")
    return host


_SITE_LOCAL = ipaddress.ip_network("fec0::/10")


def _public(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if ip in _SITE_LOCAL:  # deprecated fec0::/10; is_global misses it on 3.10
        return False
    return ip.is_global and not ip.is_multicast


def resolve_public(host: str, port: int) -> list[str]:
    """Resolve *host*; refuse unless EVERY address is public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise WebhookUnresolved("webhook host did not resolve")
    addrs: list[str] = []
    for info in infos:
        addr = info[4][0]
        if not _public(addr):
            raise WebhookRefused("webhook host resolves to a non-public address")
        if addr not in addrs:
            addrs.append(addr)
    if not addrs:
        raise WebhookUnresolved("webhook host did not resolve")
    return addrs


def pinned_targets(url: str) -> list[tuple[str, str, int, str]]:
    """Validate *url* and DNS; one (ip_url, host, port, ip) per vetted address."""
    host = validate_webhook_url(url)
    parts = urlsplit(url)
    port = parts.port or 443
    out = []
    for ip in resolve_public(host, port):
        netloc = f"[{ip}]" if ":" in ip else ip
        if parts.port:
            netloc += f":{parts.port}"
        out.append((parts._replace(netloc=netloc).geturl(), host, port, ip))
    return out
