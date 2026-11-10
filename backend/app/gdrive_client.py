"""
Google Drive client for re-skin.

Wraps Drive v3 API for downloading input videos and uploading finished videos.
Auth is via a service account JSON key.
"""

import io
import json
import time
from urllib.parse import urlsplit
import logging
import os
import re
import ssl
from http.client import IncompleteRead
from typing import Optional

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload  # noqa: E402

log = logging.getLogger(__name__)

# Resumable-upload tuning. The default chunksize (-1) streams the whole file in
# ONE request; on a slow/uneven uplink a large final.mp4 then hits the socket
# read timeout. Upload in bounded chunks and let next_chunk retry transient
# failures so delivery is robust regardless of file size / link quality.
_UPLOAD_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB per request
_UPLOAD_NUM_RETRIES = 5  # per-chunk retries with exponential backoff (HttpError 5xx)

# Socket timeout for the Drive HTTP transport. The httplib2 default is short and
# a single 5 MB chunk on a slow uplink reads past it → `TimeoutError: The read
# operation timed out`, which `next_chunk(num_retries=...)` does NOT retry (it
# only retries HttpError 5xx). 1080p deliveries (~45 MB) hit this repeatedly.
# Give each chunk a generous read window; tunable via env.
_HTTP_TIMEOUT_SEC = int(os.getenv("GDRIVE_HTTP_TIMEOUT_SEC", "300"))
# On top of next_chunk's own retries, retry a chunk that still raises a socket
# read timeout — resumable uploads resume from the last confirmed byte.
_UPLOAD_TIMEOUT_RETRIES = int(os.getenv("GDRIVE_UPLOAD_TIMEOUT_RETRIES", "8"))

# Download tuning, mirroring the upload side. The MediaIoBaseDownload default
# (100 MB) streams a reference image or a source video in ONE request with zero
# retries, so a single transient socket hiccup fails the whole run submit.
# Bounded chunks + per-chunk retries: each chunk is a fresh ranged GET, and
# next_chunk only advances `_progress` after the bytes are written, so a retried
# chunk re-requests the same range — no truncation, no duplicated bytes.
_DOWNLOAD_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB per request
_DOWNLOAD_NUM_RETRIES = 5  # per-chunk retries inside next_chunk (5xx, socket, SSL)
# next_chunk's own retry does not cover IncompleteRead (an HTTPException, not an
# OSError), so retry those a few times around it as well.
_DOWNLOAD_TIMEOUT_RETRIES = int(os.getenv("GDRIVE_DOWNLOAD_TIMEOUT_RETRIES", "5"))
# What next_chunk's own retry misses, or has already given up on. socket.timeout
# is an alias of TimeoutError on 3.10+; ssl.SSLError and ConnectionError are
# OSError subclasses, IncompleteRead is an HTTPException and is NOT covered by
# googleapiclient's _retry_request at all.
_RETRYABLE_DOWNLOAD_ERRORS = (
    TimeoutError,
    ConnectionError,
    ssl.SSLError,
    IncompleteRead,
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GDriveError(Exception):
    """Base exception for Google Drive errors."""


class GDriveAuthError(GDriveError):
    """Raised when service-account authentication fails."""


class GDriveDownloadError(GDriveError):
    """Raised when a file download fails."""


class GDriveFileTooLargeError(GDriveDownloadError):
    """The Drive file exceeds the download size cap (AICP-160)."""


class GDriveUploadError(GDriveError):
    """Raised when a file upload fails."""


class GDriveFileNotFound(GDriveError):
    """Drive answered 404 for a file id: the file is gone (deleted, or the
    service account lost access to it)."""


def _is_http_not_found(exc: BaseException) -> bool:
    """True for a googleapiclient ``HttpError`` with status 404."""
    resp = getattr(exc, "resp", None)
    return str(getattr(resp, "status", "")) == "404"


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
        promotion_session=None,
        promotion_credentials=None,
    ) -> None:
        self._promotion_session = promotion_session
        self._promotion_credentials = promotion_credentials
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
            import google_auth_httplib2
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import HttpRequest, build_http

            creds = service_account.Credentials.from_service_account_file(
                sa_file, scopes=self._SCOPES
            )

            # ONE TRANSPORT PER REQUEST — this is a correctness requirement, not
            # a tuning knob. A discovery service owns a single httplib2.Http,
            # which pools one HTTPSConnection per host and is NOT thread-safe.
            # `resolve_reference_urls` resolves references on a thread pool, so
            # two threads downloading two Drive links would grab the same open
            # connection and interleave request/response on one TLS socket. That
            # surfaced as a run whose refs came back crossed (IncompleteRead
            # quoting the *other* file's length), and as `[SSL] record layer
            # failure` / `'NoneType' object has no attribute 'read'` — and, about
            # a third of the time, as OpenSSL heap corruption that killed the RQ
            # work-horse outright (SIGSEGV/SIGABRT), stranding the run.
            #
            # `build_http()` (not a raw httplib2.Http()) is what keeps resumable
            # uploads working: it excludes 308 from httplib2's redirect codes, so
            # "Resume Incomplete" is not mistaken for a redirect. Credentials are
            # shared — only the transport is per-request — so this costs a TLS
            # handshake per request, not an extra token fetch.
            if self._promotion_credentials is None:
                self._promotion_credentials = creds

            def _threadsafe_request_builder(_http, *args, **kwargs):
                return HttpRequest(
                    google_auth_httplib2.AuthorizedHttp(creds, http=build_http()),
                    *args,
                    **kwargs,
                )

            # cache_discovery=False silences a noisy warning.
            self._service = build(
                "drive",
                "v3",
                credentials=creds,
                requestBuilder=_threadsafe_request_builder,
                cache_discovery=False,
            )
        except Exception as exc:
            raise GDriveAuthError(
                f"Failed to build Drive service from {sa_file!r}: {exc}"
            ) from exc

        return self._service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _drive_size_bytes(self, file_id: str) -> Optional[int]:
        """Size from Drive metadata, or None when unknown (Google-native docs
        have no size) or the metadata call itself fails -- the streaming cap in
        :meth:`download_file` still applies then."""
        try:
            meta = (
                self.service.files()
                .get(fileId=file_id, fields="size", supportsAllDrives=True)
                .execute()
            )
            raw = meta.get("size") if isinstance(meta, dict) else None
            return int(raw) if isinstance(raw, (str, int)) and not isinstance(raw, bool) else None
        except Exception as exc:  # noqa: BLE001 - advisory pre-check only
            log.warning("Drive size pre-check failed for %s: %s", file_id, exc)
            return None

    def file_state(self, file_id: str) -> str:
        """``"present"``, ``"trashed"`` or ``"missing"`` (Drive answered 404).

        A FRESH metadata read every call (AICP-348): the re-link action is
        allowed only on ``"missing"``, so anything Drive did not positively
        answer — auth, 403, 5xx, network — raises :class:`GDriveError` rather
        than being read as missing. A trashed file still exists (it can be
        restored, and an in-place update would succeed), so it is not missing.
        """
        try:
            meta = (
                self.service.files()
                .get(fileId=file_id, fields="id,trashed", supportsAllDrives=True)
                .execute()
            )
        except GDriveAuthError:
            raise
        except Exception as exc:
            if _is_http_not_found(exc):
                return "missing"
            raise GDriveError(f"Drive metadata read failed for {file_id}: {exc}") from exc
        if isinstance(meta, dict) and meta.get("trashed"):
            return "trashed"
        return "present"

    def generate_file_id(self) -> str:
        """A Drive file id reserved for a later :meth:`upload_file` (``file_id=``).

        Creating with a pre-generated id makes the create idempotent (AICP-348):
        a second create with the same id cannot make a second file (Drive
        answers 409), and "did it land?" is a strongly consistent
        ``files.get`` by that id instead of an eventually consistent search.
        """
        try:
            resp = (
                self.service.files()
                .generateIds(count=1, space="drive", type="files")
                .execute()
            )
        except GDriveAuthError:
            raise
        except Exception as exc:
            raise GDriveError(f"Drive could not generate a file id: {exc}") from exc
        ids = resp.get("ids") if isinstance(resp, dict) else None
        if not ids or not isinstance(ids[0], str):
            raise GDriveError("Drive returned no generated file id")
        return ids[0]

    def file_md5(self, file_id: str) -> Optional[str]:
        """Drive's ``md5Checksum`` of the file's current content (None when
        Drive reports none, e.g. a Google-native doc).

        Used to settle an ambiguous :meth:`update_file_content` failure (the
        transfer may have landed with its response lost): compare against the
        local file's md5. Raises :class:`GDriveFileNotFound` on a 404 and
        :class:`GDriveError` on any other failure, so a caller never mistakes
        "could not tell" for "not there".
        """
        try:
            meta = (
                self.service.files()
                .get(fileId=file_id, fields="md5Checksum", supportsAllDrives=True)
                .execute()
            )
        except GDriveAuthError:
            raise
        except Exception as exc:
            if _is_http_not_found(exc):
                raise GDriveFileNotFound(f"Drive file {file_id} not found") from exc
            raise GDriveError(f"Drive metadata read failed for {file_id}: {exc}") from exc
        raw = meta.get("md5Checksum") if isinstance(meta, dict) else None
        return raw if isinstance(raw, str) and raw else None

    def download_file(
        self, link_or_id: str, dst_path: str, *, max_bytes: Optional[int] = None
    ) -> str:
        """Download a Drive file to *dst_path* and return *dst_path*.

        Parameters
        ----------
        link_or_id:
            A Drive share link in any supported format, or a bare file ID.
        dst_path:
            Local file path to write the downloaded content to.
        max_bytes:
            Size cap. Defaults to ``settings.MAX_UPLOAD_SIZE_MB``; a file whose
            Drive metadata size is above it is refused before any byte is
            fetched, and the stream is cut (partial file removed) the moment
            the bytes written pass it (metadata can be absent or wrong).

        Returns
        -------
        str
            The *dst_path* that was written.

        Raises
        ------
        GDriveFileTooLargeError
            Over the cap (a :class:`GDriveDownloadError` subclass).
        GDriveDownloadError
            On any failure (resolution, API error, I/O error).
        """
        try:
            file_id = extract_file_id(link_or_id)
        except GDriveError as exc:
            raise GDriveDownloadError(str(exc)) from exc

        if max_bytes is None:
            from .config import settings

            max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
        declared = self._drive_size_bytes(file_id)
        if declared is not None and declared > max_bytes:
            raise GDriveFileTooLargeError(
                f"Drive file {file_id!r} is {declared} bytes, over the "
                f"{max_bytes}-byte download cap"
            )

        log.info("Downloading Drive file %s -> %s", file_id, dst_path)
        try:
            request = self.service.files().get_media(
                fileId=file_id, supportsAllDrives=True
            )
            with open(dst_path, "wb") as fh:
                downloader = MediaIoBaseDownload(
                    fh, request, chunksize=_DOWNLOAD_CHUNK_SIZE
                )
                done = False
                transport_retries = 0
                while not done:
                    try:
                        status, done = downloader.next_chunk(
                            num_retries=_DOWNLOAD_NUM_RETRIES
                        )
                    except _RETRYABLE_DOWNLOAD_ERRORS as exc:
                        transport_retries += 1
                        if transport_retries > _DOWNLOAD_TIMEOUT_RETRIES:
                            raise
                        log.warning(
                            "Drive download chunk failed (%s), retrying (%d/%d)",
                            exc, transport_retries, _DOWNLOAD_TIMEOUT_RETRIES,
                        )
                        continue
                    if fh.tell() > max_bytes:
                        raise GDriveFileTooLargeError(
                            f"Drive file {file_id!r} passed the {max_bytes}-byte "
                            f"download cap mid-stream; aborted"
                        )
                    if status:
                        log.debug(
                            "Download progress: %.0f%%",
                            status.progress() * 100,
                        )
        except GDriveFileTooLargeError:
            try:
                os.unlink(dst_path)
            except OSError:
                pass
            raise
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
        app_properties: Optional[dict] = None,
        file_id: Optional[str] = None,
    ) -> dict:
        """Upload *local_path* to Drive and return ``{"id", "webViewLink"}``.

        *app_properties* (optional) are set on the new file, private to this
        app (the re-link tags its upload with its operation id, for
        traceability). *file_id* (optional) is an id from
        :meth:`generate_file_id`: the file is created with exactly that id.

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
        import socket

        # Widen the socket read timeout for the duration of THIS upload only.
        # A 5 MB chunk on a slow uplink can exceed the short httplib2 default and
        # raise "read operation timed out". Scoped via getdefaulttimeout/restore
        # so it never affects the worker's redis/DB sockets (used between jobs),
        # and applied to the http transport WITHOUT replacing it (a custom
        # httplib2.Http breaks the resumable-upload 308 handling).
        _prev_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_HTTP_TIMEOUT_SEC)
        try:
            body: dict = {"name": name}
            if folder_id:
                body["parents"] = [folder_id]
            if app_properties:
                body["appProperties"] = dict(app_properties)
            if file_id:
                body["id"] = file_id

            media = MediaFileUpload(
                local_path,
                mimetype=mime_type,
                resumable=True,
                chunksize=_UPLOAD_CHUNK_SIZE,
            )
            request = self.service.files().create(
                body=body,
                media_body=media,
                fields="id,webViewLink",
                supportsAllDrives=True,
            )
            # Drive the resumable upload chunk-by-chunk. next_chunk retries
            # HttpError 5xx via num_retries; socket READ timeouts are not covered
            # by that, so we catch them here and re-call next_chunk — a resumable
            # upload resumes from the last confirmed byte, so no work is lost.
            result = None
            timeout_retries = 0
            while result is None:
                try:
                    status, result = request.next_chunk(num_retries=_UPLOAD_NUM_RETRIES)
                except (socket.timeout, TimeoutError, ConnectionError) as exc:
                    timeout_retries += 1
                    if timeout_retries > _UPLOAD_TIMEOUT_RETRIES:
                        raise
                    log.warning(
                        "Drive chunk timed out (%s), resuming (%d/%d)",
                        exc, timeout_retries, _UPLOAD_TIMEOUT_RETRIES,
                    )
                    continue
                if status:
                    log.debug("Upload progress: %.0f%%", status.progress() * 100)
        except GDriveAuthError:
            raise
        except Exception as exc:
            raise GDriveUploadError(
                f"Failed to upload {local_path!r} to folder {folder_id!r}: {exc}"
            ) from exc
        finally:
            socket.setdefaulttimeout(_prev_timeout)

        return {"id": result.get("id"), "webViewLink": result.get("webViewLink")}

    def update_file_content(
        self,
        file_id: str,
        local_path: str,
        mime_type: str = "video/mp4",
    ) -> dict:
        """Replace the CONTENT of an existing Drive file, keeping its id.

        The point is the id: ``https://drive.google.com/file/d/<id>/view`` is
        what a Final Creatives row's ``Creative Link`` holds and what everyone
        downstream has already shared, so repairing a delivered creative by
        rewriting the bytes behind the same id fixes every existing reference
        at once — where a fresh upload would leave the sheet pointing at the
        broken video. Used by :mod:`app.pipeline_audio_patch`; see
        docs/run-audio-patch.md.

        The file's NAME and parents are left alone (no ``body``), so a rename
        the operator or the downstream renaming pipeline already applied
        survives the update. Drive keeps the previous bytes as a revision, so
        this is recoverable, not destructive.

        A resumable upload either completes or publishes nothing, so a worker
        killed mid-transfer leaves the old revision live rather than a
        truncated file.

        Returns ``{"id", "webViewLink"}`` like :meth:`upload_file`.

        Raises
        ------
        GDriveUploadError
            On any failure.
        """
        import socket

        log.info("Replacing Drive file %s content with %s", file_id, local_path)

        # Same widened socket timeout + resumable-chunk loop as upload_file —
        # see the comments there for why the transport is left untouched and
        # why read timeouts are retried by re-calling next_chunk.
        _prev_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_HTTP_TIMEOUT_SEC)
        try:
            media = MediaFileUpload(
                local_path,
                mimetype=mime_type,
                resumable=True,
                chunksize=_UPLOAD_CHUNK_SIZE,
            )
            request = self.service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id,webViewLink",
                supportsAllDrives=True,
            )
            result = None
            timeout_retries = 0
            while result is None:
                try:
                    status, result = request.next_chunk(num_retries=_UPLOAD_NUM_RETRIES)
                except (socket.timeout, TimeoutError, ConnectionError) as exc:
                    timeout_retries += 1
                    if timeout_retries > _UPLOAD_TIMEOUT_RETRIES:
                        raise
                    log.warning(
                        "Drive chunk timed out (%s), resuming (%d/%d)",
                        exc, timeout_retries, _UPLOAD_TIMEOUT_RETRIES,
                    )
                    continue
                if status:
                    log.debug("Update progress: %.0f%%", status.progress() * 100)
        except GDriveAuthError:
            raise
        except Exception as exc:
            raise GDriveUploadError(
                f"Failed to replace content of Drive file {file_id!r} "
                f"with {local_path!r}: {exc}"
            ) from exc
        finally:
            socket.setdefaulttimeout(_prev_timeout)

        return {"id": result.get("id"), "webViewLink": result.get("webViewLink")}


    @staticmethod
    def _promotion_remaining(deadline):
        remaining = min(30.0, deadline - time.monotonic())
        if remaining <= 0:
            raise GDriveUploadError("Main-result transfer deadline reached; exact-session recovery required")
        return remaining

    @staticmethod
    def _result_session_uri(uri):
        # Persisted private data must never become an arbitrary authenticated URL.
        parsed = urlsplit(uri)
        if (parsed.scheme != "https" or parsed.netloc != "www.googleapis.com"
                or not parsed.path.startswith("/upload/drive/v3/files")
                or not parsed.query or parsed.fragment):
            raise GDriveUploadError("Invalid main-result transfer session; manual reconciliation required")
        return uri

    def _result_http(self):
        if self._promotion_session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            session = requests.Session()
            session.trust_env = False
            retry = Retry(total=0, connect=0, read=0, status=0, redirect=0, other=0)
            session.mount("https://", HTTPAdapter(max_retries=retry))
            session.mount("http://", HTTPAdapter(max_retries=retry))
            self._promotion_session = session
        return self._promotion_session

    def _result_auth_headers(self, headers, deadline):
        credentials = self._promotion_credentials
        if credentials is None:
            raise GDriveAuthError("Main-result transfer credentials unavailable")
        if not credentials.valid:
            # Credentials.refresh() calls jwt_grant with hidden token retries.
            # The pinned google-auth service-account implementation exposes this
            # explicit one-attempt grant; use it only for this transfer client.
            from google.auth.transport.requests import Request
            from google.oauth2 import _client
            auth_request = Request(session=self._result_http())
            def once(url, method="GET", body=None, headers=None, **kwargs):
                return auth_request(url, method=method, body=body, headers=headers,
                    timeout=self._promotion_remaining(deadline), allow_redirects=False)
            token, expiry, _ = _client.jwt_grant(once, credentials._token_uri,
                credentials._make_authorization_grant_assertion(), can_retry=False)
            credentials.token, credentials.expiry = token, expiry
        credentials.apply(headers)
        return headers

    def _promotion_request(self, uri, method, *, deadline, body=None, headers=None):
        try:
            self._promotion_remaining(deadline)
            request_headers = self._result_auth_headers(dict(headers or {}), deadline)
            # Unlike httplib2, this transport has zero connection/read/status
            # retries. Redirect and credential-refresh replay are also disabled.
            # A disconnect after body transmission is UNKNOWN, never a resend.
            response = self._result_http().request(method, uri, data=body, headers=request_headers,
                timeout=self._promotion_remaining(deadline), allow_redirects=False)
            content = response.content
        except Exception:
            # Transport exceptions may contain the private session URI.
            raise GDriveUploadError("Main-result transfer outcome unknown; retry the same saved request") from None
        self._promotion_remaining(deadline)
        return response, content

    def _result_request(self, file_id, local_path, deadline):
        self._promotion_remaining(deadline)
        media = MediaFileUpload(local_path, mimetype="video/mp4", resumable=True, chunksize=_UPLOAD_CHUNK_SIZE)
        return self.service.files().update(fileId=file_id, media_body=media,
            fields="id,size,md5Checksum,sha256Checksum", supportsAllDrives=True)

    def begin_result_update(self, file_id, local_path, *, deadline):
        """Initiate only: NO media bytes are sent until caller persists URI."""
        request = self._result_request(file_id, local_path, deadline)
        headers = dict(request.headers)
        headers.update({"X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(os.stat(local_path).st_size),
            "Content-Length": str(request.body_size)})
        response, _ = self._promotion_request(request.uri, request.method,
            deadline=deadline, body=request.body, headers=headers)
        if int(response.status_code) not in (200, 201) or not response.headers.get("location"):
            raise GDriveUploadError("Main-result session initiation failed; retry the same saved request")
        return self._result_session_uri(response.headers["location"])

    def finish_result_update(self, session_uri, file_id, local_path, *, size_bytes, sha256, md5, deadline):
        """Query/resume exactly one durable session; no new-session fallback."""
        uri = self._result_session_uri(session_uri)
        request = self._result_request(file_id, local_path, deadline)
        response, content = self._promotion_request(uri, "PUT", deadline=deadline,
            headers={"Content-Range": f"bytes */{size_bytes}", "Content-Length": "0"}, body=b"")
        with open(local_path, "rb") as video:
            while int(response.status_code) == 308:
                acknowledged = response.headers.get("range", "")
                if acknowledged:
                    match = re.fullmatch(r"bytes=0-([0-9]+)", acknowledged)
                    if match is None:
                        raise GDriveUploadError("Invalid transfer status; exact-session reconciliation required")
                    offset = int(match.group(1)) + 1
                else:
                    offset = 0
                if not 0 <= offset < size_bytes:
                    raise GDriveUploadError("Invalid transfer offset; exact-session reconciliation required")
                self._promotion_remaining(deadline)
                video.seek(offset)
                chunk = video.read(min(_UPLOAD_CHUNK_SIZE, size_bytes - offset))
                if not chunk:
                    raise GDriveUploadError("Publication source unavailable; exact-session reconciliation required")
                response, content = self._promotion_request(uri, "PUT", deadline=deadline, body=chunk,
                    headers={"Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{offset + len(chunk) - 1}/{size_bytes}"})
        if int(response.status_code) not in (200, 201):
            raise GDriveUploadError("Main-result session unresolved; manual reconciliation or exact retry required")
        try:
            result = json.loads(content)
            confirmed = (result.get("id") == file_id and int(result.get("size", -1)) == size_bytes
                and result.get("md5Checksum") == md5
                and (not result.get("sha256Checksum") or result["sha256Checksum"] == sha256))
        except (ValueError, TypeError, AttributeError):
            confirmed = False
        if not confirmed:
            raise GDriveUploadError("Main-result completion identity differs; manual reconciliation required")
        return {"id": file_id, "size": size_bytes}
