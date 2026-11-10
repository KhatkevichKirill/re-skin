"""
Tests for app.gdrive_client.

No real network calls, no real credentials.
"""

import io
import os
import ssl
import tempfile
from http.client import IncompleteRead
from unittest.mock import MagicMock, patch, call

import pytest

from app import gdrive_client
from app.gdrive_client import (
    GDriveAuthError,
    GDriveDownloadError,
    GDriveError,
    GDriveUploadError,
    GDriveClient,
    extract_file_id,
)


# ---------------------------------------------------------------------------
# extract_file_id — parametrized over all supported formats
# ---------------------------------------------------------------------------

FILE_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"

VALID_LINKS = [
    # /file/d/<ID>/view?usp=sharing
    (
        f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing",
        FILE_ID,
    ),
    # /open?id=<ID>
    (
        f"https://drive.google.com/open?id={FILE_ID}",
        FILE_ID,
    ),
    # /uc?id=<ID>&export=download
    (
        f"https://drive.google.com/uc?id={FILE_ID}&export=download",
        FILE_ID,
    ),
    # docs.google.com/.../d/<ID>/edit
    (
        f"https://docs.google.com/document/d/{FILE_ID}/edit",
        FILE_ID,
    ),
    # bare ID — returned as-is
    (
        FILE_ID,
        FILE_ID,
    ),
]


@pytest.mark.parametrize("link,expected", VALID_LINKS)
def test_extract_file_id_valid(link, expected):
    assert extract_file_id(link) == expected


@pytest.mark.parametrize(
    "bad_input",
    [
        "",
        "not-a-url",
        "https://example.com/no-id-here",
        "short",  # too short to be a bare ID
    ],
)
def test_extract_file_id_invalid(bad_input):
    with pytest.raises(GDriveError):
        extract_file_id(bad_input)


# ---------------------------------------------------------------------------
# Helpers — build a fully-mocked service
# ---------------------------------------------------------------------------

def _make_mock_service():
    """Return a MagicMock that quacks like a Drive v3 service."""
    svc = MagicMock()
    return svc


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------

class TestUploadFile:
    def test_upload_calls_create_with_correct_body(self, tmp_path):
        """files().create() is called with the right body and supportsAllDrives."""
        local = tmp_path / "output.mp4"
        local.write_bytes(b"fake video data")

        folder_id = "FOLDER123"
        expected_result = {"id": "NEW_FILE_ID", "webViewLink": "https://drive.google.com/file/d/NEW_FILE_ID/view"}

        svc = _make_mock_service()
        svc.files().create().next_chunk.return_value = (None, expected_result)

        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaFileUpload") as MockMedia:
            MockMedia.return_value = MagicMock()
            result = client.upload_file(str(local), folder_id=folder_id, name="output.mp4")

        # Check the body passed to create()
        create_kwargs = svc.files().create.call_args
        body_arg = create_kwargs.kwargs.get("body") or create_kwargs[1].get("body")
        assert body_arg["name"] == "output.mp4"
        assert body_arg["parents"] == [folder_id]

        # supportsAllDrives must be True
        assert create_kwargs.kwargs.get("supportsAllDrives") is True or \
               create_kwargs[1].get("supportsAllDrives") is True

        assert result == {"id": "NEW_FILE_ID", "webViewLink": "https://drive.google.com/file/d/NEW_FILE_ID/view"}

    def test_upload_name_defaults_to_basename(self, tmp_path):
        local = tmp_path / "my_video.mp4"
        local.write_bytes(b"x")

        svc = _make_mock_service()
        svc.files().create().next_chunk.return_value = (None, {"id": "ID1", "webViewLink": "http://link"})

        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaFileUpload"):
            client.upload_file(str(local), folder_id="F1")

        create_kwargs = svc.files().create.call_args
        body_arg = create_kwargs.kwargs.get("body") or create_kwargs[1].get("body")
        assert body_arg["name"] == "my_video.mp4"

    def test_upload_folder_id_defaults_to_settings(self, tmp_path, monkeypatch):
        """When folder_id is omitted, settings.GDRIVE_DEFAULT_FOLDER_ID is used."""
        local = tmp_path / "vid.mp4"
        local.write_bytes(b"x")

        # Monkeypatch the settings object used inside upload_file
        import app.config as cfg_module
        monkeypatch.setattr(cfg_module.settings, "GDRIVE_DEFAULT_FOLDER_ID", "DEFAULT_FOLDER")

        svc = _make_mock_service()
        svc.files().create().next_chunk.return_value = (None, {"id": "ID2", "webViewLink": "http://link2"})

        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaFileUpload"):
            client.upload_file(str(local))

        create_kwargs = svc.files().create.call_args
        body_arg = create_kwargs.kwargs.get("body") or create_kwargs[1].get("body")
        assert body_arg["parents"] == ["DEFAULT_FOLDER"]

    def test_upload_raises_gdrive_upload_error_on_exception(self, tmp_path):
        local = tmp_path / "vid.mp4"
        local.write_bytes(b"x")

        svc = _make_mock_service()
        svc.files().create().next_chunk.side_effect = RuntimeError("API down")

        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaFileUpload"):
            with pytest.raises(GDriveUploadError, match="API down"):
                client.upload_file(str(local), folder_id="F1")


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_download_resolves_link_and_writes_file(self, tmp_path):
        """A share link is resolved to a file ID and bytes are written to dst."""
        dst = str(tmp_path / "downloaded.mp4")
        fake_bytes = b"fake video content"

        # Build a fake MediaIoBaseDownload that writes bytes and signals done.
        class FakeDownloader:
            def __init__(self, fh, request, chunksize=None):
                self._fh = fh

            def next_chunk(self, num_retries=0):
                self._fh.write(fake_bytes)
                return None, True  # (status, done)

        svc = _make_mock_service()
        # get_media just needs to return something (passed to downloader)
        svc.files().get_media.return_value = MagicMock()

        client = GDriveClient(service=svc)

        link = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"

        with patch("app.gdrive_client.MediaIoBaseDownload", FakeDownloader):
            result = client.download_file(link, dst)

        # Correct file ID was passed to get_media
        svc.files().get_media.assert_called_once_with(
            fileId=FILE_ID, supportsAllDrives=True
        )

        assert result == dst
        with open(dst, "rb") as fh:
            assert fh.read() == fake_bytes

    def test_download_raises_download_error_on_bad_link(self, tmp_path):
        dst = str(tmp_path / "out.mp4")
        svc = _make_mock_service()
        client = GDriveClient(service=svc)

        with pytest.raises(GDriveDownloadError):
            client.download_file("https://example.com/no-id", dst)

    def test_download_raises_download_error_on_api_failure(self, tmp_path):
        dst = str(tmp_path / "out.mp4")

        svc = _make_mock_service()
        svc.files().get_media.side_effect = RuntimeError("network error")

        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaIoBaseDownload"):
            with pytest.raises(GDriveDownloadError, match="network error"):
                client.download_file(FILE_ID, dst)

    def test_download_is_chunked_and_retried(self, tmp_path):
        """Downloads are bounded chunks with per-chunk retries, like uploads.

        The default (100 MB / num_retries=0) streams a whole file in one request
        with no retry at all, so one transient socket hiccup fails a run submit.
        """
        dst = str(tmp_path / "out.mp4")
        seen = {}

        class FakeDownloader:
            def __init__(self, fh, request, chunksize=None):
                seen["chunksize"] = chunksize

            def next_chunk(self, num_retries=0):
                seen["num_retries"] = num_retries
                return None, True

        svc = _make_mock_service()
        svc.files().get_media.return_value = MagicMock()
        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaIoBaseDownload", FakeDownloader):
            client.download_file(FILE_ID, dst)

        assert seen["chunksize"] == gdrive_client._DOWNLOAD_CHUNK_SIZE
        assert seen["num_retries"] == gdrive_client._DOWNLOAD_NUM_RETRIES

    @pytest.mark.parametrize(
        "exc",
        [
            IncompleteRead(b"", 42),
            ssl.SSLError("record layer failure"),
            ConnectionResetError("reset by peer"),
            TimeoutError("read timed out"),
        ],
    )
    def test_download_retries_transient_transport_errors(self, tmp_path, exc):
        """A transient transport error mid-download is retried, not fatal.

        IncompleteRead in particular is an HTTPException, so googleapiclient's
        own _retry_request never covers it.
        """
        dst = str(tmp_path / "out.mp4")
        payload = b"chunked bytes"

        class FlakyDownloader:
            calls = 0

            def __init__(self, fh, request, chunksize=None):
                self._fh = fh

            def next_chunk(self, num_retries=0):
                FlakyDownloader.calls += 1
                if FlakyDownloader.calls == 1:
                    raise exc
                self._fh.write(payload)
                return None, True

        svc = _make_mock_service()
        svc.files().get_media.return_value = MagicMock()
        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaIoBaseDownload", FlakyDownloader):
            client.download_file(FILE_ID, dst)

        assert FlakyDownloader.calls == 2
        with open(dst, "rb") as fh:
            assert fh.read() == payload

    def test_download_gives_up_after_retry_budget(self, tmp_path):
        """A permanently broken transfer still fails — retries are bounded."""
        dst = str(tmp_path / "out.mp4")

        class DeadDownloader:
            calls = 0

            def __init__(self, fh, request, chunksize=None):
                pass

            def next_chunk(self, num_retries=0):
                DeadDownloader.calls += 1
                raise ConnectionResetError("reset by peer")

        svc = _make_mock_service()
        svc.files().get_media.return_value = MagicMock()
        client = GDriveClient(service=svc)

        with patch("app.gdrive_client.MediaIoBaseDownload", DeadDownloader):
            with pytest.raises(GDriveDownloadError, match="reset by peer"):
                client.download_file(FILE_ID, dst)

        assert DeadDownloader.calls == gdrive_client._DOWNLOAD_TIMEOUT_RETRIES + 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestTransportIsPerRequest:
    def test_each_request_gets_its_own_http(self, tmp_path):
        """Two requests off one service must not share an httplib2.Http.

        httplib2.Http pools one HTTPSConnection per host and is not thread-safe.
        Sharing it across the reference-resolution thread pool interleaved two
        downloads on one TLS socket: crossed responses, `[SSL] record layer
        failure`, and OpenSSL heap corruption that killed the worker outright.
        """
        sa_file = tmp_path / "sa.json"
        sa_file.write_text("{}")

        captured = {}

        def fake_build(serviceName, version, **kwargs):
            captured["requestBuilder"] = kwargs["requestBuilder"]
            return MagicMock()

        client = GDriveClient(service_account_file=str(sa_file))

        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=MagicMock(),
        ), patch("googleapiclient.discovery.build", fake_build):
            _ = client.service

        build_request = captured["requestBuilder"]
        req_a = build_request(MagicMock(), MagicMock(), "https://example.test/a")
        req_b = build_request(MagicMock(), MagicMock(), "https://example.test/b")

        assert req_a.http is not req_b.http
        # …and each transport must keep 308 out of httplib2's redirect codes,
        # or resumable uploads break (308 = "Resume Incomplete", not a redirect).
        assert 308 not in req_a.http.http.redirect_codes


# ---------------------------------------------------------------------------
# Constructor / auth
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_auth_error_on_missing_sa_file(self):
        """Accessing .service with a non-existent SA file raises GDriveAuthError."""
        client = GDriveClient(service_account_file="/nonexistent/path/sa.json")
        with pytest.raises(GDriveAuthError, match="not found"):
            _ = client.service

    def test_injected_service_used_directly(self):
        """When service= is passed, it is returned without loading any SA file."""
        fake_svc = MagicMock()
        client = GDriveClient(service=fake_svc)
        assert client.service is fake_svc

    def test_import_never_requires_creds(self):
        """Importing the module must not raise even without credentials present."""
        import importlib
        import app.gdrive_client as mod
        importlib.reload(mod)  # re-import to confirm no side effects


# ---------------------------------------------------------------------------
# file_md5 (AICP-349): settles an ambiguous in-place update
# ---------------------------------------------------------------------------


def _http_error(status: int):
    from googleapiclient.errors import HttpError

    resp = MagicMock()
    resp.status = status
    resp.reason = "x"
    return HttpError(resp, b"{}")


def test_file_md5_returns_drive_checksum():
    svc = MagicMock()
    svc.files.return_value.get.return_value.execute.return_value = {"md5Checksum": "abc"}
    assert GDriveClient(service=svc).file_md5("fid") == "abc"
    svc.files.return_value.get.assert_called_with(
        fileId="fid", fields="md5Checksum", supportsAllDrives=True
    )


def test_file_md5_404_is_not_found_other_errors_are_not():
    # The module's own classes: another test in this file reloads it.
    GDriveFileNotFound = GDriveClient.file_md5.__globals__["GDriveFileNotFound"]
    GDriveError = GDriveClient.file_md5.__globals__["GDriveError"]

    svc = MagicMock()
    svc.files.return_value.get.return_value.execute.side_effect = _http_error(404)
    with pytest.raises(GDriveFileNotFound):
        GDriveClient(service=svc).file_md5("fid")
    svc.files.return_value.get.return_value.execute.side_effect = _http_error(503)
    with pytest.raises(GDriveError) as exc:
        GDriveClient(service=svc).file_md5("fid")
    assert not isinstance(exc.value, GDriveFileNotFound)


# ---------------------------------------------------------------------------
# AICP-348: fresh missing-check, tagged upload, lookup by tag
# ---------------------------------------------------------------------------


def test_file_state_present_trashed_missing_and_unknown():
    GDriveError_ = GDriveClient.file_state.__globals__["GDriveError"]
    svc = MagicMock()
    get = svc.files.return_value.get
    get.return_value.execute.return_value = {"id": "f", "trashed": False}
    assert GDriveClient(service=svc).file_state("f") == "present"
    get.return_value.execute.return_value = {"id": "f", "trashed": True}
    assert GDriveClient(service=svc).file_state("f") == "trashed"
    get.return_value.execute.side_effect = _http_error(404)
    assert GDriveClient(service=svc).file_state("f") == "missing"
    get.return_value.execute.side_effect = _http_error(403)
    with pytest.raises(GDriveError_):
        GDriveClient(service=svc).file_state("f")
    get.assert_called_with(fileId="f", fields="id,trashed", supportsAllDrives=True)


def test_generate_file_id():
    svc = MagicMock()
    gen = svc.files.return_value.generateIds
    gen.return_value.execute.return_value = {"ids": ["reserved-1"]}
    assert GDriveClient(service=svc).generate_file_id() == "reserved-1"
    gen.assert_called_with(count=1, space="drive", type="files")
    gen.return_value.execute.return_value = {"ids": []}
    with pytest.raises(GDriveClient.file_state.__globals__["GDriveError"]):
        GDriveClient(service=svc).generate_file_id()


def test_upload_file_sets_app_properties(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"abc")
    svc = MagicMock()
    create = svc.files.return_value.create
    create.return_value.next_chunk.return_value = (None, {"id": "new", "webViewLink": "w"})
    with patch.object(gdrive_client, "MediaFileUpload"):
        out = GDriveClient(service=svc).upload_file(
            str(p), "folder", "n.mp4", app_properties={"reskinRelinkId": "r1"},
            file_id="reserved-1")
    assert out["id"] == "new"
    body = create.call_args.kwargs["body"]
    assert body["appProperties"] == {"reskinRelinkId": "r1"}
    assert body["id"] == "reserved-1"
    assert body["parents"] == ["folder"]
