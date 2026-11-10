"""
public.py — Unauthenticated, token-signed access to project media.

Mounted at /public (see app/main.py). nginx lets /public/ through WITHOUT basic
auth, so these endpoints MUST validate a per-resource HMAC token themselves.

Use case: hand an external tool (e.g. Gemini for on-screen text recognition) a
direct, unguessable link to a project's ORIGINAL source video without exposing
the basic-auth master password.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .accel import accel_or_file_response
from .config import settings
from .db import get_session
from .models import Run, RunCaption, RunTailEdit, VideoProject

log = logging.getLogger(__name__)

router = APIRouter(tags=["public"])


# --- Signing keys (AICP-151) -------------------------------------------------
#
# The secret is dedicated (PUBLIC_LINK_SECRET) and has no fallback. When it is
# missing/short the feature fails CLOSED: endpoints answer 503, nothing is
# minted, and the rest of the app keeps working.

_NOT_CONFIGURED_DETAIL = "Public links are not configured"
_LOG_EVERY_SEC = 300.0
_last_logged: dict[str, float] = {}


def _log_throttled(key: str, msg: str, *args) -> None:
    now = time.monotonic()
    if now - _last_logged.get(key, -_LOG_EVERY_SEC) >= _LOG_EVERY_SEC:
        _last_logged[key] = now
        log.error(msg, *args)


def _min_len() -> int:
    return int(getattr(settings, "PUBLIC_LINK_SECRET_MIN_LEN", 32))


def _signing_key() -> Optional[bytes]:
    """The key tokens are minted with, or None when public links are disabled."""
    secret = (getattr(settings, "PUBLIC_LINK_SECRET", "") or "").strip()
    if not secret:
        _log_throttled(
            "missing",
            "PUBLIC_LINK_SECRET is not set: public links are disabled "
            "(/public/* returns 503, no tokens minted). Generate one with "
            "`python3 -c 'import secrets; print(secrets.token_urlsafe(48))'`.",
        )
        return None
    if len(secret) < _min_len():
        _log_throttled(
            "short",
            "PUBLIC_LINK_SECRET is shorter than %d characters: public links "
            "are disabled.",
            _min_len(),
        )
        return None
    return secret.encode()


def _previous_keys() -> list[bytes]:
    """Verify-only keys from PUBLIC_LINK_SECRET_PREVIOUS (short ones ignored)."""
    raw = getattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", "") or ""
    keys: list[bytes] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if len(part) < _min_len():
            _log_throttled(
                "prev-short",
                "Ignoring a PUBLIC_LINK_SECRET_PREVIOUS entry shorter than %d "
                "characters.",
                _min_len(),
            )
            continue
        keys.append(part.encode())
    return keys


def public_links_enabled() -> bool:
    """True when a valid PUBLIC_LINK_SECRET is configured."""
    return _signing_key() is not None


def _sign(key: bytes, purpose: str, ident: str) -> str:
    return hmac.new(key, f"{purpose}:{ident}".encode(), hashlib.sha256).hexdigest()


def _mint(purpose: str, ident: str) -> str:
    """Token for a link, or "" when public links are disabled (fail closed).

    Only ever signs with the CURRENT secret, never a previous one.
    """
    key = _signing_key()
    if key is None:
        return ""
    return _sign(key, purpose, ident)


def _require_valid_token(purpose: str, ident: str, token: str) -> None:
    """503 if links are disabled, 403 if *token* is wrong. Constant-time.

    Accepts a token signed with the current secret or, during a rotation
    grace period, with a PUBLIC_LINK_SECRET_PREVIOUS entry (verify-only).
    A previous secret alone is not enough: the current one must be valid too.
    """
    key = _signing_key()
    if key is None:
        raise HTTPException(status_code=503, detail=_NOT_CONFIGURED_DETAIL)
    supplied = (token or "").encode("utf-8", "replace")
    ok = False
    for k in [key, *_previous_keys()]:
        # No short-circuit: every candidate is compared.
        ok |= hmac.compare_digest(supplied, _sign(k, purpose, ident).encode())
    if not ok:
        raise HTTPException(status_code=403, detail="Invalid or missing token")


def make_source_token(project_id: str) -> str:
    """Return the per-project HMAC token for its public source link.

    Scoped with a purpose prefix so the same secret can sign other link types
    later without token reuse across purposes. Returns "" (and logs) when
    public links are disabled; callers must not build a link from it.
    """
    return _mint("project-source", project_id)


@router.get("/projects/{pid}/source")
def public_project_source(
    pid: str,
    token: str = Query("", description="Per-project signed access token"),
) -> FileResponse:
    """Stream a project's original source video if *token* is valid.

    Token is checked first (constant-time) so this can't be used to enumerate
    project ids. Supports HTTP Range, so external fetchers can stream/seek.
    """
    _require_valid_token("project-source", pid, token)

    # Session is closed before the (potentially long) file transfer.
    with get_session() as db:
        project = db.get(VideoProject, pid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        src = project.source_local_path
        if not src or not os.path.exists(src):
            raise HTTPException(status_code=404, detail="Source video not available")

    log.info("Public source fetch for project %s", pid)
    return accel_or_file_response(src, media_type="video/mp4", filename="source.mp4")


def make_result_token(run_id: str) -> str:
    """Return the per-run HMAC token for its public final-video link."""
    return _mint("run-result", run_id)


@router.get("/runs/{rid}/result")
def public_run_result(
    rid: str,
    token: str = Query("", description="Per-run signed access token"),
) -> FileResponse:
    """Stream a run's final video if *token* is valid (Range supported)."""
    _require_valid_token("run-result", rid, token)

    # Session is closed before the (potentially long) file transfer.
    with get_session() as db:
        run = db.get(Run, rid)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        path = run.result_local_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result video not available")

    log.info("Public result fetch for run %s", rid)
    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename="result.mp4",
        extra_headers={
            "Cache-Control": "private, no-cache",
        },
    )


def make_tail_edit_result_token(edit_id: str) -> str:
    """Return the per-tail-edit HMAC token for its public final-video link.

    Uses a distinct purpose prefix from ``make_result_token`` so a run's token
    can't be replayed against a tail edit's endpoint (and vice versa) even if
    the two ids happened to collide.
    """
    return _mint("tail-edit-result", edit_id)


@router.get("/tail-edits/{edit_id}/result")
def public_tail_edit_result(
    edit_id: str,
    token: str = Query("", description="Per-tail-edit signed access token"),
) -> FileResponse:
    """Stream a tail edit's result video if *token* is valid (Range supported)."""
    _require_valid_token("tail-edit-result", edit_id, token)

    # Session is closed before the (potentially long) file transfer.
    with get_session() as db:
        edit = db.get(RunTailEdit, edit_id)
        if edit is None:
            raise HTTPException(status_code=404, detail="Tail edit not found")

        path = edit.output_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result video not available")

    log.info("Public result fetch for tail edit %s", edit_id)
    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename="result.mp4",
        extra_headers={
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


def make_caption_result_token(caption_id: str) -> str:
    """Return the per-caption HMAC token for its public burned-video link.

    Its own purpose prefix, for the same reason
    ``make_tail_edit_result_token`` has one: a token minted for a run or a
    tail edit must not validate here even if the ids collided.
    """
    return _mint("caption-result", caption_id)


@router.get("/captions/{caption_id}/result")
def public_caption_result(
    caption_id: str,
    token: str = Query("", description="Per-caption signed access token"),
) -> FileResponse:
    """Stream a caption job's burned video if *token* is valid (Range supported).

    Exists so the overlay editor can preview a caption-sourced job the same way
    it previews a run or a tail edit — the <video> element needs a URL nginx
    will serve without basic auth.
    """
    _require_valid_token("caption-result", caption_id, token)

    # Session is closed before the (potentially long) file transfer.
    with get_session() as db:
        caption = db.get(RunCaption, caption_id)
        if caption is None:
            raise HTTPException(status_code=404, detail="Caption not found")

        path = caption.output_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result video not available")

    log.info("Public result fetch for caption %s", caption_id)
    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename="result.mp4",
        extra_headers={
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )
