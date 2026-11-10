"""
accel.py — X-Accel-Redirect helper for nginx-accelerated media serving.

When NGINX_ACCEL_ENABLED is True, media endpoints return a lightweight
Response with an ``X-Accel-Redirect`` header instead of streaming file bytes
through Python.  nginx picks up the header, opens the file directly (using
sendfile), handles Range requests natively, and the uvicorn process is freed
immediately.

The internal location ``/internal/data/`` in nginx.conf maps to the bind-mounted
``./data`` directory.  We convert an absolute on-disk path like
``/app/data/projects/<pid>/source.mp4`` to the URI
``/internal/data/projects/<pid>/source.mp4``.

Range requests continue to work: nginx honours the client's ``Range`` header
transparently once it takes over the response.

DB connections:  callers should commit + close the session BEFORE calling
``accel_or_file_response`` so the connection is returned to the pool before
the (potentially long) transfer starts.
"""

from __future__ import annotations

import mimetypes
import os
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from .config import settings
from .storage import PathOutsideDataDir, ensure_under_data_dir

# Internal nginx prefix that maps to the data directory root.
_INTERNAL_PREFIX = "/internal/data"


def _abs_path_to_accel_uri(abs_path: str) -> Optional[str]:
    """Convert *abs_path* to an X-Accel-Redirect URI.

    Returns None if the path cannot be mapped (e.g. DATA_DIR not set or path
    is not under the data dir).
    """
    data_dir = settings.DATA_DIR
    if not data_dir:
        return None

    # Resolve both sides so symlinks or trailing slashes don't trip us up.
    data_dir = os.path.realpath(data_dir)
    abs_path = os.path.realpath(abs_path)

    if not abs_path.startswith(data_dir + os.sep):
        return None

    relative = abs_path[len(data_dir):]  # starts with "/"
    return _INTERNAL_PREFIX + relative


def accel_or_file_response(
    abs_path: str,
    *,
    media_type: str = "video/mp4",
    filename: Optional[str] = None,
    extra_headers: Optional[dict] = None,
) -> Response:
    """Return an X-Accel-Redirect response (nginx mode) or FileResponse (dev mode).

    Parameters
    ----------
    abs_path:
        Absolute path to the file on disk (must already be verified to exist).
        Anything that does not resolve to a path under DATA_DIR is refused
        with a 404 — this helper fails closed (AICP-148).
    media_type:
        MIME type sent in Content-Type.
    filename:
        If set, adds ``Content-Disposition: attachment; filename=...``.
    extra_headers:
        Additional response headers merged in (e.g. X-GDrive-File-Id,
        Cache-Control).
    """
    try:
        abs_path = ensure_under_data_dir(abs_path)
    except PathOutsideDataDir:
        raise HTTPException(status_code=404, detail="File not found")

    headers: dict = {}
    if extra_headers:
        headers.update(extra_headers)

    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    if settings.NGINX_ACCEL_ENABLED:
        accel_uri = _abs_path_to_accel_uri(abs_path)
        if accel_uri is not None:
            headers["X-Accel-Redirect"] = accel_uri
            headers["Content-Type"] = media_type
            return Response(status_code=200, headers=headers)
        # Fallthrough: settings.DATA_DIR unset or not the storage root — the
        # path is already proven to be under the storage root, so stream it.

    # Dev / test / fallback: stream via FileResponse.
    # Build FileResponse first, then merge extra headers on top.
    file_resp = FileResponse(abs_path, media_type=media_type)
    file_resp.headers.update(headers)
    return file_resp
