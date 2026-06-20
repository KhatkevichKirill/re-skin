"""
Google Drive client for re-skin.

Wraps Drive v3 API for downloading input videos and uploading finished videos.
Auth is via a service account JSON key.
"""

import io
import logging
import os
import re
from typing import Optional

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GDriveError(Exception):
    """Base exception for Google Drive errors."""


class GDriveAuthError(GDriveError):
    """Raised when service-account authentication fails."""


class GDriveDownloadError(GDriveError):
    """Raised when a file download fails."""


class GDriveUploadError(GDriveError):
    """Raised when a file upload fails."""


# ---------------------------------------------------------------------------
# Pure helper — no I/O
# ---------------------------------------------------------------------------

# Patterns ordered from most specific to least specific.
_ID_PATTERNS = [
    # /file/d/<ID>/
    re.compile(r"/file/d/([a-zA-Z0-9_-]+)"),
    # /d/<ID>/  (Docs, Sheets, Slides …)
    re.compile(r"/d/([a-zA-Z0-9_-]+)"),
    # ?id=<ID> or &id=<ID>
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
]

# A bare file ID is alphanumeric + dash + underscore, at least 10 chars.
_BARE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{10,}$")


def extract_file_id(link_or_id: str) -> str:
    """Return the bare Drive file ID from any supported link format or a bare ID.

    Supported inputs
    ----------------
    - ``https://drive.google.com/file/d/<ID>/view?usp=sharing``
    - ``https://drive.google.com/open?id=<ID>``
    - ``https://drive.google.com/uc?id=<ID>&export=download``
    - ``https://docs.google.com/.../d/<ID>/edit``
    - A bare ``<ID>`` string (returned as-is).

    Raises
    ------
    GDriveError
        If the string cannot be parsed into a recognisable file ID.
    """
    link_or_id = link_or_id.strip()

    # Not a URL — treat as a bare ID if it looks like one.
    if not link_or_id.startswith("http"):
        if _BARE_ID_RE.match(link_or_id):
            return link_or_id
        raise GDriveError(f"Cannot parse file ID from: {link_or_id!r}")

    for pattern in _ID_PATTERNS:
        m = pattern.search(link_or_id)
        if m:
            return m.group(1)

    raise GDriveError(f"Cannot parse file ID from URL: {link_or_id!r}")


# ---------------------------------------------------------------------------
# GDriveClient
# ---------------------------------------------------------------------------

class GDriveClient:
    """Thin wrapper around the Drive v3 API.

    Parameters
    ----------
    service_account_file:
        Path to the service-account JSON key.  Defaults to
        ``settings.GOOGLE_SERVICE_ACCOUNT_FILE``.
    service:
        Pre-built Drive service object (for testing).  When supplied the
        ``service_account_file`` is ignored and no credentials are needed.
    """

    _SCOPES = ["https://www.googleapis.com/auth/drive"]

    def __init__(
        self,
        service_account_file: Optional[str] = None,
        service=None,
    ) -> None:
        self._sa_file = service_account_file
        self._service = service  # may be None; built lazily on first use

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def service(self):
        """Return (or lazily build) the Drive service object."""
        if self._service is not None:
            return self._service

        # Resolve SA file path
        if self._sa_file is None:
            from app.config import settings  # local import for testability

            self._sa_file = settings.GOOGLE_SERVICE_ACCOUNT_FILE

        sa_file = self._sa_file

        if not sa_file or not os.path.isfile(sa_file):
            raise GDriveAuthError(
                f"Service-account file not found: {sa_file!r}. "
                "Set GOOGLE_SERVICE_ACCOUNT_FILE or pass service_account_file."
            )

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                sa_file, scopes=self._SCOPES
            )
            self._service = build("drive", "v3", credentials=creds)
        except Exception as exc:
            raise GDriveAuthError(
                f"Failed to build Drive service from {sa_file!r}: {exc}"
            ) from exc

        return self._service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_file(self, link_or_id: str, dst_path: str) -> str:
        """Download a Drive file to *dst_path* and return *dst_path*.

        Parameters
        ----------
        link_or_id:
            A Drive share link in any supported format, or a bare file ID.
        dst_path:
            Local file path to write the downloaded content to.

        Returns
        -------
        str
            The *dst_path* that was written.

        Raises
        ------
        GDriveDownloadError
            On any failure (resolution, API error, I/O error).
        """
        try:
            file_id = extract_file_id(link_or_id)
        except GDriveError as exc:
            raise GDriveDownloadError(str(exc)) from exc

        log.info("Downloading Drive file %s -> %s", file_id, dst_path)
        try:
            request = self.service.files().get_media(
                fileId=file_id, supportsAllDrives=True
            )
            with open(dst_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        log.debug(
                            "Download progress: %.0f%%",
                            status.progress() * 100,
                        )
        except GDriveAuthError:
            raise
        except Exception as exc:
            raise GDriveDownloadError(
                f"Failed to download file {file_id!r} to {dst_path!r}: {exc}"
            ) from exc

        return dst_path

    def upload_file(
        self,
        local_path: str,
        folder_id: Optional[str] = None,
        name: Optional[str] = None,
        mime_type: str = "video/mp4",
    ) -> dict:
        """Upload *local_path* to Drive and return ``{"id", "webViewLink"}``.

        Parameters
        ----------
        local_path:
            Path to the local file to upload.
        folder_id:
            Drive folder ID to upload into.  Defaults to
            ``settings.GDRIVE_DEFAULT_FOLDER_ID``.
        name:
            Name to give the file in Drive.  Defaults to the local basename.
        mime_type:
            MIME type of the file.

        Returns
        -------
        dict
            ``{"id": str, "webViewLink": str}``

        Raises
        ------
        GDriveUploadError
            On any failure.
        """
        from app.config import settings  # local import for testability

        if folder_id is None:
            folder_id = settings.GDRIVE_DEFAULT_FOLDER_ID

        if name is None:
            name = os.path.basename(local_path)

        log.info(
            "Uploading %s -> Drive folder %s as %r", local_path, folder_id, name
        )
        try:
            body: dict = {"name": name}
            if folder_id:
                body["parents"] = [folder_id]

            media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
            result = (
                self.service.files()
                .create(
                    body=body,
                    media_body=media,
                    fields="id,webViewLink",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except GDriveAuthError:
            raise
        except Exception as exc:
            raise GDriveUploadError(
                f"Failed to upload {local_path!r} to folder {folder_id!r}: {exc}"
            ) from exc

        return {"id": result.get("id"), "webViewLink": result.get("webViewLink")}
