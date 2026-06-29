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

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import Run, VideoProject

log = logging.getLogger(__name__)

router = APIRouter(tags=["public"])


def make_source_token(project_id: str) -> str:
    """Return the per-project HMAC token for its public source link.

    Scoped with a purpose prefix so the same secret can sign other link types
    later without token reuse across purposes.
    """
    secret = (settings.PUBLIC_LINK_SECRET or "").encode()
    msg = f"project-source:{project_id}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


@router.get("/projects/{pid}/source")
def public_project_source(
    pid: str,
    token: str = Query(..., description="Per-project signed access token"),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream a project's original source video if *token* is valid.

    Token is checked first (constant-time) so this can't be used to enumerate
    project ids. Supports HTTP Range, so external fetchers can stream/seek.
    """
    expected = make_source_token(pid)
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    project = db.get(VideoProject, pid)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    src = project.source_local_path
    if not src or not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Source video not available")

    log.info("Public source fetch for project %s", pid)
    return FileResponse(src, media_type="video/mp4", filename="source.mp4")


def make_result_token(run_id: str) -> str:
    """Return the per-run HMAC token for its public final-video link."""
    secret = (settings.PUBLIC_LINK_SECRET or "").encode()
    msg = f"run-result:{run_id}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


@router.get("/runs/{rid}/result")
def public_run_result(
    rid: str,
    token: str = Query(..., description="Per-run signed access token"),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Stream a run's final video if *token* is valid (Range supported)."""
    expected = make_result_token(rid)
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    run = db.get(Run, rid)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    path = run.result_local_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result video not available")

    log.info("Public result fetch for run %s", rid)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="result.mp4",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
