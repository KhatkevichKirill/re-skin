"""Synthetic DNS and loopback/Unix canaries; never provider traffic or auth."""

import os
from pathlib import Path
import socket
import threading

import pytest

from app.assistant_egress import EgressServer, ProxyRelay, connect_destination, public_address


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "172.17.0.1", "192.168.1.1", "169.254.169.254",
                                     "0.0.0.0", "224.0.0.1", "240.0.0.1", "100.64.0.1", "192.0.2.1",
                                     "::1", "fe80::1", "fc00::1", "ff02::1", "::ffff:127.0.0.1"])
def test_nonpublic_and_special_addresses_denied(address):
    assert not public_address(address)


def info(address):
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))


@pytest.mark.parametrize("authority", ["127.0.0.1:443", "169.254.169.254:80", "api.anthropic.com:80",
                                       "api.anthropic.com.evil.test:443", "evil.test:443", "claude.ai.:443",
                                       "user@claude.ai:443", "CLAUDE.AI:443", "[::1]:443", "claude.ai:0443"])
def test_exact_host_and_port_allowlist_precedes_dns(authority):
    def forbidden(*args, **kwargs):
        pytest.fail("denied host must not resolve or connect")
    with pytest.raises(ValueError):
        connect_destination(authority, resolver=forbidden, connector=forbidden)


@pytest.mark.parametrize("addresses", [[], [info("127.0.0.1")], [info("8.8.8.8"), info("10.0.0.1")]])
def test_empty_private_and_mixed_dns_fail_closed(addresses):
    def forbidden(*args, **kwargs):
        pytest.fail("untrusted DNS must not connect")
    with pytest.raises(ValueError):
        connect_destination("api.anthropic.com:443", resolver=lambda *a, **kw: addresses, connector=forbidden)


def test_connect_uses_validated_ip_without_reresolution():
    calls = []
    marker = object()
    def connector(address, **kwargs):
        calls.append((address, kwargs))
        return marker
    assert connect_destination("platform.claude.com:443", resolver=lambda *a, **kw: [info("8.8.8.8")],
                               connector=connector) is marker
    assert calls == [(("8.8.8.8", 443), {"timeout": 5})]


def test_unix_peer_denial_and_networknone_relay_tunnel(tmp_path):
    path = tmp_path / "egress.sock"
    tunnels = []
    def synthetic_connector(authority):
        def connector(address, **kwargs):
            left, right = socket.socketpair()
            def echo():
                with right:
                    right.settimeout(2)
                    right.sendall(right.recv(1024))
            thread = threading.Thread(target=echo)
            thread.start()
            tunnels.append(thread)
            return left
        return connect_destination(authority, resolver=lambda *a, **kw: [info("8.8.8.8")], connector=connector)
    server = EgressServer(path, destination_connector=synthetic_connector)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
    thread.start()
    relay = ProxyRelay(path, port=0)
    relay.start()
    def request(raw):
        connection = socket.create_connection(relay.server.server_address, timeout=2)
        connection.sendall(raw)
        return connection
    try:
        if os.getuid() != 999:
            with request(b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n\r\n") as connection:
                denied = connection.recv(1024)
                assert not denied or b"403 Forbidden" in denied
        server.allowed_uids = frozenset({os.getuid()})  # Test process only.
        for denied in (b"GET https://api.anthropic.com/ HTTP/1.1\r\n\r\n",
                       b"CONNECT 169.254.169.254:80 HTTP/1.1\r\n\r\n",
                       b"CONNECT evil.test:443 HTTP/1.1\r\n\r\n",
                       b"CONNECT claude.ai:443 HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}"):
            with request(denied) as connection:
                assert b"403 Forbidden" in connection.recv(1024)
        with request(b"CONNECT api.anthropic.com:443 HTTP/1.1\r\nHost: api.anthropic.com:443\r\n\r\n") as connection:
            assert b"200 Connection Established" in connection.recv(1024)
            connection.sendall(b"synthetic-canary")
            assert connection.recv(1024) == b"synthetic-canary"
        assert len(tunnels) == 1
    finally:
        relay.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
        for tunnel in tunnels:
            tunnel.join(2)
