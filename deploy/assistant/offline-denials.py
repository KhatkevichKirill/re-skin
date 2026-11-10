"""Run ONLY in a disposable network-none container with empty /config.

No native executable, provider request or real credential mount is needed.
Prints booleans/counts only. The synthetic gateway connects a socketpair canary.
"""
import json
import os
from pathlib import Path
import socket
import threading

from app.assistant_egress import EgressServer, ProxyRelay, connect_destination

assert os.getuid() == 999
assert {name for _, name in socket.if_nameindex()} == {"lo"}
assert Path("/config").is_dir() and not any(Path("/config").iterdir())
for forbidden in ("/root/.codex", "/root/.claude", "/var/run/docker.sock", "/app/secrets", "/app/data"):
    try:
        assert not Path(forbidden).exists()
    except PermissionError:
        pass  # An inaccessible developer home also satisfies the canary boundary.

denied = 0
for host, port in (("172.17.0.1", 80), ("172.17.0.1", 5432), ("172.17.0.1", 6379),
                   ("10.0.0.1", 443), ("169.254.169.254", 80), ("1.1.1.1", 443),
                   ("2001:4860:4860::8888", 443), ("fe80::1", 443)):
    try:
        with socket.create_connection((host, port), timeout=.2):
            raise AssertionError("direct egress unexpectedly allowed")
    except OSError:
        denied += 1

echoes = []
def synthetic_connector(authority):
    def connector(address, **kwargs):
        left, right = socket.socketpair()
        def echo():
            with right:
                right.settimeout(2)
                right.sendall(right.recv(1024))
        thread = threading.Thread(target=echo)
        thread.start()
        echoes.append(thread)
        return left
    resolver = lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
    return connect_destination(authority, resolver=resolver, connector=connector)

path = Path("/tmp/egress-canary.sock")
server = EgressServer(path, destination_connector=synthetic_connector)
thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
thread.start()
relay = ProxyRelay(path)
relay.start()
proxy_denied = 0
try:
    for authority in ("evil.invalid:443", "127.0.0.1:443", "169.254.169.254:80", "api.anthropic.com:80"):
        with socket.create_connection(("127.0.0.1", 18080), timeout=2) as connection:
            connection.sendall(("CONNECT " + authority + " HTTP/1.1\r\n\r\n").encode())
            assert b"403 Forbidden" in connection.recv(1024)
            proxy_denied += 1
    with socket.create_connection(("127.0.0.1", 18080), timeout=2) as connection:
        connection.sendall(b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n\r\n")
        assert b"200 Connection Established" in connection.recv(1024)
        connection.sendall(b"harmless-offline-canary")
        assert connection.recv(1024) == b"harmless-offline-canary"
    assert len(echoes) == 1
finally:
    relay.close()
    server.shutdown()
    server.server_close()
    thread.join(2)
    for echo in echoes:
        echo.join(2)
    path.unlink()

print(json.dumps({"network_none": True, "empty_config": True, "uid": os.getuid(),
                  "direct_denials": denied, "proxy_denials": proxy_denied, "synthetic_tunnel": True}))
