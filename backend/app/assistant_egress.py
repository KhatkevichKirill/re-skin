"""Credential-free CONNECT gateway and loopback-to-Unix relay.

The runner has network_mode:none. Its only egress is this Unix socket tunnel.
TLS is end-to-end; this gateway never parses provider requests or credentials.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import selectors
import signal
import socket
import socketserver
import stat
import struct
import threading
import time

ALLOWED_HOSTS = frozenset(("api.anthropic.com", "claude.ai", "platform.claude.com"))
MAX_HEADER = 8192
MAX_TUNNEL_BYTES = 32 * 1024 * 1024


def public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if getattr(ip, "ipv4_mapped", None):
        return public_address(str(ip.ipv4_mapped))
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_loopback or ip.is_link_local or ip.is_unspecified)


def connect_destination(authority: str, *, resolver=socket.getaddrinfo, connector=socket.create_connection):
    # No wildcards, credentials, IP literals, suffix matching, arbitrary ports,
    # trailing dots, escaped names, redirects, or client-controlled DNS.
    if authority not in {host + ":443" for host in ALLOWED_HOSTS}:
        raise ValueError("destination denied")
    host = authority[:-4]
    addresses = resolver(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
    if not addresses or any(not public_address(info[4][0]) for info in addresses):
        raise ValueError("destination denied")
    # Pin the validated IP rather than resolving the hostname again on connect.
    for info in addresses:
        try:
            return connector((info[4][0], 443), timeout=5)
        except OSError:
            continue
    raise OSError("destination unavailable")


def _pipe(left, right, *, lifetime=180, byte_limit=MAX_TUNNEL_BYTES):
    deadline = time.monotonic() + lifetime
    total = 0
    # Blocking writes have a short deadline; bounded parallel connections keep
    # a stalled client from accumulating unbounded threads or queued buffers.
    left.settimeout(2)
    right.settimeout(2)
    with selectors.DefaultSelector() as selector:
        selector.register(left, selectors.EVENT_READ, right)
        selector.register(right, selectors.EVENT_READ, left)
        while time.monotonic() < deadline:
            for key, _ in selector.select(0.2):
                chunk = key.fileobj.recv(16384)
                if not chunk:
                    return
                total += len(chunk)
                if total > byte_limit:
                    return
                key.data.sendall(chunk)


class _BoundedThreads(socketserver.ThreadingMixIn):
    daemon_threads = True
    request_queue_size = 8

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, address):
        pass  # Never log tunneled payloads, request headers, or exceptions.


class _ConnectHandler(socketserver.BaseRequestHandler):
    def handle(self):
        remote = None
        try:
            self.request.settimeout(5)
            _, uid, _ = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid not in self.server.allowed_uids:
                raise ValueError("peer denied")
            data = bytearray()
            while b"\r\n\r\n" not in data:
                chunk = self.request.recv(min(1024, MAX_HEADER + 1 - len(data)))
                if not chunk:
                    raise ValueError("incomplete request")
                data.extend(chunk)
                if len(data) > MAX_HEADER:
                    raise ValueError("oversized request")
            headers, initial = bytes(data).split(b"\r\n\r\n", 1)
            lines = headers.decode("ascii", errors="strict").split("\r\n")
            method, authority, protocol = lines[0].split(" ")
            if method != "CONNECT" or protocol not in ("HTTP/1.0", "HTTP/1.1"):
                raise ValueError("method denied")
            for line in lines[1:]:
                key, value = line.split(":", 1)
                if not key or key.lower() in ("content-length", "transfer-encoding"):
                    raise ValueError("invalid headers")
            remote = self.server.connect_destination(authority)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if initial:
                remote.sendall(initial)
            _pipe(self.request, remote)
        except (OSError, ValueError, UnicodeError):
            if remote is None:
                try:
                    self.request.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                except OSError:
                    pass
        finally:
            if remote is not None:
                remote.close()


class EgressServer(_BoundedThreads, socketserver.UnixStreamServer):
    def __init__(self, path: Path, *, allowed_uids=(999,), destination_connector=connect_destination):
        if not allowed_uids or any(type(uid) is not int or uid < 1 for uid in allowed_uids):
            raise ValueError("explicit unprivileged peer UIDs required")
        self.allowed_uids = frozenset(allowed_uids)
        self.connect_destination = destination_connector
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(str(path), _ConnectHandler)
        os.chmod(path, 0o666)


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        remote = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            remote.settimeout(5)
            remote.connect(str(self.server.unix_path))
            _pipe(self.request, remote)
        except OSError:
            pass
        finally:
            remote.close()


class _RelayServer(_BoundedThreads, socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, path, port):
        self.unix_path = path
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(("127.0.0.1", port), _RelayHandler)


class ProxyRelay:
    def __init__(self, unix_path: Path, port: int = 18080):
        self.server = _RelayServer(unix_path, port)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, default=Path("/egress/proxy.sock"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.socket.exists():
        if not stat.S_ISSOCK(args.socket.lstat().st_mode):
            raise ValueError("unexpected socket entry")
        args.socket.unlink()
    server = EgressServer(args.socket)
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
