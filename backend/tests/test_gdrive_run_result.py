"""Real googleapiclient request objects; fake transport only, no credentials."""
import hashlib
import json
import time
from types import SimpleNamespace

import httplib2
import pytest
from googleapiclient.discovery import build
from googleapiclient.http import HttpRequest
from google.auth.credentials import AnonymousCredentials

from app.gdrive_client import GDriveClient, GDriveUploadError


URI = "https://www.googleapis.com/upload/drive/v3/files/file-id?upload_id=private-session"


class Transport:
    def __init__(self, replies):
        self.calls = []
        self.replies = list(replies)
        self.timeout = None
        self.connections = {}

    def close(self):
        pass

    def request(self, method, uri, data=None, headers=None, **kwargs):
        self.calls.append((uri, method, data, headers))
        assert kwargs["allow_redirects"] is False
        self.timeout = kwargs["timeout"]
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        status, headers_out, payload = reply
        return SimpleNamespace(status_code=status, headers=headers_out,
            content=json.dumps(payload).encode() if isinstance(payload, dict) else payload)


def client(transport):
    service = build("drive", "v3", http=transport, cache_discovery=False, static_discovery=True)
    return GDriveClient(service=service, promotion_session=transport, promotion_credentials=AnonymousCredentials())


def metadata(content):
    return {"id": "file-id", "size": str(len(content)), "md5Checksum": hashlib.md5(content).hexdigest(),
        "sha256Checksum": hashlib.sha256(content).hexdigest()}


def finish(c, path):
    data = path.read_bytes()
    return c.finish_result_update(URI, "file-id", str(path), size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(), md5=hashlib.md5(data).hexdigest(), deadline=time.monotonic() + 90)


def test_initiation_is_metadata_only_then_exact_session_status_and_media(tmp_path):
    path = tmp_path / "result.mp4"
    path.write_bytes(b"known-full-video")
    transport = Transport([(200, {"location": URI}, b""), (308, {}, b""), (200, {}, metadata(path.read_bytes()))])
    c = client(transport)
    uri = c.begin_result_update("file-id", str(path), deadline=time.monotonic() + 90)
    assert uri == URI and len(transport.calls) == 1
    request_uri, method, body, headers = transport.calls[0]
    assert method == "PATCH" and body is None  # no media before durable URI commit
    assert "uploadType=resumable" in request_uri and "md5Checksum" in request_uri
    assert headers["X-Upload-Content-Length"] == str(path.stat().st_size)
    assert headers["X-Upload-Content-Type"] == "video/mp4"
    finish(c, path)
    assert transport.calls[1][0] == transport.calls[2][0] == URI
    assert transport.calls[1][2] == b"" and transport.calls[1][3]["Content-Range"] == f"bytes */{path.stat().st_size}"
    assert transport.calls[2][2] == path.read_bytes()
    assert 0 < transport.timeout <= 30


def test_query_of_already_completed_session_sends_no_media(tmp_path):
    path = tmp_path / "result.mp4"
    path.write_bytes(b"already-committed")
    transport = Transport([(200, {}, metadata(path.read_bytes()))])
    finish(client(transport), path)
    assert len(transport.calls) == 1 and transport.calls[0][2] == b""


def test_resume_from_provider_offset_and_terminal_identity(tmp_path):
    path = tmp_path / "result.mp4"
    path.write_bytes(b"abcdefghij")
    transport = Transport([(308, {"range": "bytes=0-3"}, b""), (201, {}, metadata(path.read_bytes()))])
    finish(client(transport), path)
    assert transport.calls[1][2] == b"efghij"
    assert transport.calls[1][3]["Content-Range"] == "bytes 4-9/10"


@pytest.mark.parametrize("reply", [(404, {}, b"expired"), (200, {}, {"id": "wrong", "size": "3"}), TimeoutError(URI)])
def test_expired_ambiguous_or_wrong_session_never_reinitializes_and_redacts(tmp_path, reply):
    path = tmp_path / "result.mp4"
    path.write_bytes(b"abc")
    transport = Transport([reply])
    with pytest.raises(GDriveUploadError) as error:
        finish(client(transport), path)
    assert "private-session" not in str(error.value)
    assert len(transport.calls) == 1 and transport.calls[0][0] == URI


def test_real_one_attempt_transport_does_not_resend_media_after_disconnect(tmp_path):
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import httplib2
    from google.auth.credentials import AnonymousCredentials
    received = []
    class DisconnectAfterBody(BaseHTTPRequestHandler):
        def do_PUT(self):
            length = int(self.headers["Content-Length"])
            received.append(self.rfile.read(length))
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), DisconnectAfterBody)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous_timeout, previous_retries = socket.getdefaulttimeout(), httplib2.RETRIES
    c = GDriveClient(promotion_credentials=AnonymousCredentials())
    try:
        with pytest.raises(GDriveUploadError):
            c._promotion_request(f"http://127.0.0.1:{server.server_port}/session", "PUT",
                deadline=time.monotonic() + 3, body=b"sent-once-media", headers={"Content-Length": "15"})
        assert received == [b"sent-once-media"]
        assert socket.getdefaulttimeout() == previous_timeout and httplib2.RETRIES == previous_retries
        for adapter in c._result_http().adapters.values():
            retry = adapter.max_retries
            assert retry.total == retry.connect == retry.read == retry.status == retry.redirect == retry.other == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        c._result_http().close()


def test_one_attempt_transport_refuses_redirect_replay():
    c = GDriveClient(promotion_credentials=AnonymousCredentials())
    transport = Transport([(307, {"location": "https://example.invalid/replay"}, b"")])
    c._promotion_session = transport
    response, _ = c._promotion_request(URI, "PUT", deadline=time.monotonic() + 3, body=b"media")
    assert response.status_code == 307 and len(transport.calls) == 1


def test_promotion_token_grant_disables_hidden_auth_retries(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from google.oauth2 import _client
    seen = []
    credentials = SimpleNamespace(valid=False, _token_uri="https://oauth2.googleapis.com/token",
        _make_authorization_grant_assertion=lambda: b"test-assertion",
        apply=lambda headers: headers.update({"Authorization": "Bearer isolated-test-token"}))
    def grant(request, uri, assertion, can_retry=True):
        seen.append(can_retry)
        return "isolated-test-token", datetime.now(timezone.utc) + timedelta(hours=1), {}
    monkeypatch.setattr(_client, "jwt_grant", grant)
    transport = Transport([(308, {}, b"")])
    c = GDriveClient(promotion_credentials=credentials, promotion_session=transport)
    c._promotion_request(URI, "PUT", deadline=time.monotonic() + 3, body=b"")
    assert seen == [False] and len(transport.calls) == 1
