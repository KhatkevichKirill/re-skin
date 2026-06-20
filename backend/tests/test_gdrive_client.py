"""
Tests for app.gdrive_client.

No real network calls, no real credentials.
"""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

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
        svc.files().create().execute.return_value = expected_result

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
        svc.files().create().execute.return_value = {"id": "ID1", "webViewLink": "http://link"}

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
        svc.files().create().execute.return_value = {"id": "ID2", "webViewLink": "http://link2"}

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
        svc.files().create().execute.side_effect = RuntimeError("API down")

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
            def __init__(self, fh, request):
                self._fh = fh

            def next_chunk(self):
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
