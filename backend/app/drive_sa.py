"""Service-account identity for the "share this with ..." notices (AICP-274).

Every Google Drive file or folder re-skin reads or writes must be shared with
the service account in ``settings.GOOGLE_SERVICE_ACCOUNT_FILE``; otherwise the
app fails later with an opaque 404. The UI therefore prints the account's
e-mail next to every Drive input, and that e-mail is read from the key file
(never hard-coded) so a rotated service account shows up without a code change.
The parsed JSON necessarily includes the private key (there is no partial parse),
but it is dropped immediately: only ``client_email`` is kept or returned.

Only ``client_email`` is ever read from the key file. The lookup is lazy (first
render, not import), cached once it succeeds, and fail-safe: a missing,
unreadable or malformed file yields ``None`` and the notice degrades to a
sentence without the address.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from .config import settings

log = logging.getLogger(__name__)

_lock = threading.Lock()
# (path, mtime_ns, email) of the last successful read. Keyed on the file's
# mtime so a rotated key (same path, new account) is picked up without a
# restart: the Drive clients re-read the file themselves, and a stale address
# here would send operators to share files with the OLD account.
_cached: Optional[tuple] = None
_warned = False


def _read_client_email(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    email = data.get("client_email")
    if isinstance(email, str) and email.strip():
        return email.strip()
    return None


def service_account_email() -> Optional[str]:
    """The service account's e-mail, or None if the key file can't supply it.

    Lazy (first render, not import). A successful read is reused until the key
    file's path or mtime changes; a failure is never cached, and its warning is
    logged only once so a missing file does not spam the log on every render.
    """
    global _cached, _warned
    path = settings.GOOGLE_SERVICE_ACCOUNT_FILE
    try:
        st = os.stat(path)
        # mtime alone misses a timestamp-preserving copy (cp -p / rsync -t);
        # size and inode catch the usual rotations.
        mtime = (st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        mtime = None
    cached = _cached
    if mtime is not None and cached and cached[0] == path and cached[1] == mtime:
        return cached[2]
    with _lock:
        email = _read_client_email(path) if mtime is not None else None
        if email:
            _cached = (path, mtime, email)
            return email
        _cached = None
        if not _warned:
            _warned = True
            log.warning(
                "drive_sa: no client_email readable from GOOGLE_SERVICE_ACCOUNT_FILE; "
                "Drive sharing notices will omit the address"
            )
        return None


def reset_cache() -> None:
    """Forget the cached e-mail (tests)."""
    global _cached, _warned
    with _lock:
        _cached = None
        _warned = False


def register(env) -> None:
    """Expose ``drive_sa_email()`` as a Jinja global on ``env``."""
    env.globals["drive_sa_email"] = service_account_email
