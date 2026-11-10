"""Autopilot API (``/api/v2/projects/{pid}/autopilot…``) and the project-page
card (``/v2/projects/{pid}/autopilot-card…``), docs/autopilot-design.md §7.

Every write locks the autopilot row only, bumps its version and journals
(core.py operator actions). The card is functional only (UX is AICP-367).
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Run, VideoProject
from . import core, run_view

router = APIRouter(tags=["autopilot"])
web_router = APIRouter(tags=["autopilot-web"])


def _who(request: Optional[Request]) -> str:
    if request is None:
        return "operator"
    return (request.headers.get("x-remote-user") or "operator")[:128]


def _project(db: Session, pid: str) -> VideoProject:
    p = db.get(VideoProject, pid)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    return p


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except core.OperatorError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from None


class AnswerBody(BaseModel):
    decision: str
    option: Optional[str] = None
    comment: Optional[str] = None


class CommentBody(BaseModel):
    text: str


class SwitchBody(BaseModel):
    reason: Optional[str] = None


class PatchBody(BaseModel):
    tolerance: Optional[float] = None
    params: Optional[dict] = None


@router.get("/projects/{pid}/autopilot")
def get_autopilot(pid: str, journal_before: Optional[int] = Query(None),
                  db: Session = Depends(get_db)) -> dict:
    _project(db, pid)
    v = core.view(db, pid, journal_before=journal_before)
    if v is None:
        raise HTTPException(status_code=404, detail="this project has no autopilot")
    return v


@router.patch("/projects/{pid}/autopilot")
def patch_autopilot(pid: str, body: PatchBody, db: Session = Depends(get_db)) -> dict:
    _project(db, pid)
    _call(core.patch, db, pid, tolerance=body.tolerance, params=body.params)
    return core.view(db, pid)


@router.post("/projects/{pid}/autopilot/{action}")
def switch_autopilot(pid: str, action: str, request: Request,
                     body: Optional[SwitchBody] = None, db: Session = Depends(get_db)) -> dict:
    _project(db, pid)
    if action == "start":
        _call(core.start_new, db, pid)
    elif action in ("pause", "resume", "stop"):
        _call(core.set_switch, db, pid, action, reason=(body.reason if body else "") or "",
              by=_who(request))
    else:
        raise HTTPException(status_code=404, detail="unknown autopilot action")
    return core.view(db, pid)


@router.post("/projects/{pid}/autopilot/approvals/{aid}/answer")
def answer_approval(pid: str, aid: str, body: AnswerBody, request: Request,
                    db: Session = Depends(get_db)) -> dict:
    _project(db, pid)
    _call(core.answer, db, pid, aid, body.decision, option=body.option, comment=body.comment,
          by=_who(request))
    return core.view(db, pid)


@router.post("/projects/{pid}/autopilot/approvals/{aid}/reopen")
def reopen_approval(pid: str, aid: str, request: Request,
                    db: Session = Depends(get_db)) -> dict:
    """"Ask again": take back an answered choice that launched nothing."""
    _project(db, pid)
    _call(core.reopen, db, pid, aid, by=_who(request))
    return core.view(db, pid)


@router.post("/projects/{pid}/autopilot/approvals/{aid}/comments")
def comment_approval(pid: str, aid: str, body: CommentBody, request: Request,
                     db: Session = Depends(get_db)) -> dict:
    _project(db, pid)
    _call(core.comment, db, pid, aid, body.text, by=_who(request))
    return core.view(db, pid)


@router.get("/projects/{pid}/autopilot/artifacts/{sha256}")
def get_artifact(pid: str, sha256: str, db: Session = Depends(get_db)):
    _project(db, pid)
    path = core.artifact_path(db, pid, sha256)
    if path is None or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="artifact not found")
    ext = os.path.splitext(path)[1].lower()
    media = {".mp4": "video/mp4", ".mov": "video/quicktime", ".jpg": "image/jpeg",
             ".jpeg": "image/jpeg", ".png": "image/png", ".wav": "audio/wav",
             ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media,
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


# ---------------------------------------------------------------------------
# The project-page card (HTMX)
# ---------------------------------------------------------------------------


def _card(request: Request, db: Session, pid: str, error: Optional[str] = None) -> HTMLResponse:
    from ..web_v2 import templates

    project = _project(db, pid)
    ap = core.view(db, pid, journal_limit=20)
    extras = run_view.project_card_extras(db, pid, ap)
    return templates.TemplateResponse(request, "partials/autopilot_card.html", {
        "request": request, "project": project, "ap": ap, "error": error, **extras,
    })


@web_router.get("/projects/{pid}/autopilot-card", response_class=HTMLResponse)
def autopilot_card(pid: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _card(request, db, pid)


@web_router.post("/projects/{pid}/autopilot-card/answer", response_class=HTMLResponse)
def autopilot_card_answer(pid: str, request: Request, approval_id: str = Form(...),
                          decision: str = Form(...), option: Optional[str] = Form(None),
                          comment: Optional[str] = Form(None),
                          autopilot_id: Optional[str] = Form(None),
                          db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        core.answer(db, pid, approval_id, decision, option=option or None, comment=comment,
                    by=_who(request), expected_id=autopilot_id or None)
        err = None
    except core.OperatorError as exc:
        db.rollback()
        err = exc.detail
    return _card(request, db, pid, err)


@web_router.post("/projects/{pid}/autopilot-card/reopen", response_class=HTMLResponse)
def autopilot_card_reopen(pid: str, request: Request, approval_id: str = Form(...),
                          autopilot_id: Optional[str] = Form(None),
                          db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        core.reopen(db, pid, approval_id, by=_who(request), expected_id=autopilot_id or None)
        err = None
    except core.OperatorError as exc:
        db.rollback()
        err = exc.detail
    return _card(request, db, pid, err)


@web_router.post("/projects/{pid}/autopilot-card/switch", response_class=HTMLResponse)
def autopilot_card_switch(pid: str, request: Request, action: str = Form(...),
                          autopilot_id: Optional[str] = Form(None),
                          db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        if action == "start":
            core.start_new(db, pid)
        else:
            core.set_switch(db, pid, action, by=_who(request), expected_id=autopilot_id or None)
        err = None
    except core.OperatorError as exc:
        db.rollback()
        err = exc.detail
    return _card(request, db, pid, err)


# ---------------------------------------------------------------------------
# The Run-page card (AICP-369): the assistant's work on ONE of its runs
# ---------------------------------------------------------------------------


def _run(db: Session, rid: str) -> Run:
    r = db.get(Run, rid)
    if r is None:
        raise HTTPException(status_code=404, detail="run not found")
    return r


def _run_card(request: Request, db: Session, rid: str, error: Optional[str] = None) -> HTMLResponse:
    from ..web_v2 import templates

    run = _run(db, rid)
    project = _project(db, run.project_id)
    return templates.TemplateResponse(request, "partials/autopilot_run_card.html", {
        "request": request, "project": project, "run": run,
        "card": run_view.run_card(db, run), "error": error,
    })


def _run_owner(db: Session, run: Run) -> str:
    """The autopilot attempt that created this run; actions from its page act on it only."""
    ar = run_view.project_runs(db, run.project_id).by_run.get(run.id)
    if ar is None:
        raise core.OperatorError(409, "the assistant did not create this run")
    return ar.autopilot_id


@web_router.get("/runs/{rid}/autopilot-card", response_class=HTMLResponse)
def autopilot_run_card(rid: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _run_card(request, db, rid)


@web_router.post("/runs/{rid}/autopilot-card/answer", response_class=HTMLResponse)
def autopilot_run_card_answer(rid: str, request: Request, approval_id: str = Form(...),
                              decision: str = Form(...), option: Optional[str] = Form(None),
                              comment: Optional[str] = Form(None),
                              db: Session = Depends(get_db)) -> HTMLResponse:
    run = _run(db, rid)
    try:
        owner = _run_owner(db, run)
        core.answer(db, run.project_id, approval_id, decision, option=option or None,
                    comment=comment, by=_who(request), expected_id=owner)
        err = None
    except core.OperatorError as exc:
        db.rollback()
        err = exc.detail
    return _run_card(request, db, rid, err)


@web_router.post("/runs/{rid}/autopilot-card/reopen", response_class=HTMLResponse)
def autopilot_run_card_reopen(rid: str, request: Request, approval_id: str = Form(...),
                              db: Session = Depends(get_db)) -> HTMLResponse:
    run = _run(db, rid)
    try:
        core.reopen(db, run.project_id, approval_id, by=_who(request),
                    expected_id=_run_owner(db, run))
        err = None
    except core.OperatorError as exc:
        db.rollback()
        err = exc.detail
    return _run_card(request, db, rid, err)


@web_router.post("/runs/{rid}/autopilot-card/switch", response_class=HTMLResponse)
def autopilot_run_card_switch(rid: str, request: Request, action: str = Form(...),
                              db: Session = Depends(get_db)) -> HTMLResponse:
    run = _run(db, rid)
    try:
        if action not in ("pause", "resume", "stop"):
            raise core.OperatorError(400, f"unknown action {action!r}")
        core.set_switch(db, run.project_id, action, by=_who(request),
                        expected_id=_run_owner(db, run))
        err = None
    except core.OperatorError as exc:
        db.rollback()
        err = exc.detail
    return _run_card(request, db, rid, err)


# ---------------------------------------------------------------------------
# Journal timeline (AICP-377, docs/autopilot-operator-ux.md §6): loaded when the
# card's Journal is opened, filtered and paged on the server.
# ---------------------------------------------------------------------------


def _journal(request: Request, pg: dict, url: str, all_lines: bool, before: Optional[int],
             during: bool) -> HTMLResponse:
    from ..web_v2 import templates
    from .status import step_label

    return templates.TemplateResponse(request, "partials/autopilot_journal.html", {
        "request": request, "entries": pg["entries"], "next_before": pg["next_before"],
        "url": url, "all": 1 if all_lines else 0, "before": before, "during": during,
        "step_label": step_label})


@web_router.get("/projects/{pid}/autopilot-journal", response_class=HTMLResponse)
def autopilot_journal(pid: str, request: Request, autopilot_id: Optional[str] = Query(None),
                      all: int = Query(0), before: Optional[int] = Query(None),
                      db: Session = Depends(get_db)) -> HTMLResponse:
    from ..models_autopilot import ProjectAutopilot
    from . import timeline

    _project(db, pid)
    ap = db.get(ProjectAutopilot, autopilot_id) if autopilot_id else core._live_or_latest(db, pid)
    if ap is None or ap.project_id != pid:
        raise HTTPException(status_code=404, detail="autopilot not found")
    pg = timeline.page(db, ap.id, important=not all, before=before)
    return _journal(request, pg, f"/v2/projects/{pid}/autopilot-journal?autopilot_id={ap.id}&",
                    bool(all), before, False)


@web_router.get("/runs/{rid}/autopilot-journal", response_class=HTMLResponse)
def autopilot_run_journal(rid: str, request: Request, all: int = Query(0),
                          before: Optional[int] = Query(None),
                          db: Session = Depends(get_db)) -> HTMLResponse:
    from . import timeline

    run = _run(db, rid)
    scope = run_view.run_journal_scope(db, run)
    if scope is None:
        raise HTTPException(status_code=404, detail="the assistant did not create this run")
    ap_id, cond = scope
    pg = timeline.page(db, ap_id, important=not all, before=before, scope=cond, mark_during=True)
    return _journal(request, pg, f"/v2/runs/{rid}/autopilot-journal?", bool(all), before, True)


# ---------------------------------------------------------------------------
# "Waiting for you" inbox + header badge (AICP-375, docs/autopilot-operator-ux.md §3)
# ---------------------------------------------------------------------------


@web_router.get("/assistant", response_class=HTMLResponse)
def assistant_inbox(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    from ..web_v2 import templates
    from . import inbox

    return templates.TemplateResponse(request, "assistant.html", {
        "request": request, "v": inbox.view(db)})


@web_router.get("/assistant/inbox-fragment", response_class=HTMLResponse)
def assistant_inbox_fragment(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    from ..web_v2 import templates
    from . import inbox

    return templates.TemplateResponse(request, "partials/assistant_inbox.html", {
        "request": request, "v": inbox.view(db)})


@web_router.get("/assistant/badge", response_class=HTMLResponse)
def assistant_badge(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    from ..web_v2 import templates
    from . import inbox

    return templates.TemplateResponse(request, "partials/assistant_badge.html", {
        "request": request, "b": inbox.badge(db)})
