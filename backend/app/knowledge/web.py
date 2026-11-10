"""Run-page "Outcome" card — the HTMX twins of the feedback API (AICP-299).

Design: docs/knowledge-feedback-design.md §7; docs/knowledge-capture-hooks.md
§8 (unpreserved versions) and §10 (card). Mounted under ``/v2``. Everything
404s unless ``KNOWLEDGE_ENABLED`` and ``KNOWLEDGE_FEEDBACK_UI`` are both on and
the Run's stored project is in ``KnowledgeScope``.

The card is fetched once into ``#knowledge-card`` (a sibling AFTER
``#run-status-content``, so status polling never wipes a half-filled form)
and every form re-renders the whole card. Each render mints a fresh
``idempotency_key`` per form, so a reload or a 409 re-render never replays an
earlier request by accident, while a double click of one rendered form is a
same-key replay that returns the stored receipt.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from . import feedback, store
from . import taxonomy as t
from .access import KnowledgeAccessDenied, KnowledgeScope
from .api import (CorrectionIn, FeedbackIn, RetractIn, defects_for_store, external_repair_call,
                  read_external_repair)

router = APIRouter(tags=["knowledge-web"])

ACTOR_BADGES = {"system_check": "Auto", "assistant": "Assistant", "operator": "Operator", "importer": "Imported"}
VERDICT_LABELS = {"accepted_as_is": "Accepted as-is", "needs_repair": "Needs repair", "rejected": "Rejected",
                  "unreviewed": "Unreviewed", "hidden_holdout": "Hidden (evaluation holdout)"}
DERIVED_LABELS = {"run_tail_edits": "Tail edit", "run_captions": "Captioned version", "overlay_jobs": "Overlay",
                  "run_audio_patches": "Audio patch"}
JUDGED_EVENTS = ("operator_verdict", "defect", "correction", "link_external")


def card_enabled() -> bool:
    return (getattr(settings, "KNOWLEDGE_ENABLED", False) is True
            and getattr(settings, "KNOWLEDGE_FEEDBACK_UI", False) is True)


def card_visible(run, project_exists: bool) -> bool:
    """Run page gate: both flags, a done run, and its project in knowledge scope."""
    if not card_enabled() or not project_exists:
        return False
    status = getattr(run.status, "value", run.status)
    return status == "done" and KnowledgeScope.from_settings(settings).allows_project(run.project_id)


def get_card_scope() -> KnowledgeScope:
    if not card_enabled():
        raise HTTPException(404, "Not Found")
    return KnowledgeScope.from_settings(settings)


# ---------------------------------------------------------------- view model


def _key() -> str:
    return uuid.uuid4().hex


def _case_title(case: dict) -> str:
    scope = case["scope"]
    if scope == "run":
        return "Delivered final"
    if scope == "external":
        return "External repair " + case["case_key"].rsplit(":", 1)[-1]
    if scope == "derived":
        table = case["case_key"].split(":")[1] if case["case_key"].count(":") >= 2 else ""
        return DERIVED_LABELS.get(table, "Derived creative")
    if scope == "manual":
        return "Manual creative"
    return scope.title()


def _event_summary(e: dict, events_by_id: dict) -> str:
    kind = e["type"]
    if kind == "operator_verdict":
        text = VERDICT_LABELS.get(e["verdict"], e["verdict"])
    elif kind == "defect":
        text = f"Defect {e['defect_code']} ({e['severity']})"
        start, end = e["time_range"]
        if start is not None and end is not None:
            text += f" {start:.2f}s–{end:.2f}s"
    elif kind == "correction":
        text = f"Correction ({e['method']}, resolved: {e['resolved']})"
    elif kind == "link_external":
        text = "Linked an external repair"
    elif kind == "auto_assessment":
        text = f"{e['check_kind']}: {e['claim']}"
    elif kind == "export_observed":
        text = "Exported to Final Creatives (not an acceptance)"
    elif kind == "retraction":
        target = events_by_id.get(e["supersedes_event_id"])
        text = "Retracted " + (_event_summary(target, {}) if target else "an event")
    else:
        text = kind
    if e["version_no"] is not None and kind != "retraction":
        text += f" · v{e['version_no']}" + (" (older version)" if e["historical"] else "")
    return text


def _case_block(session: Session, case: dict, run_case: bool, kept: Optional[dict]) -> Optional[dict]:
    versions = [v for v in case["versions"] if not v["invalidated"]]
    current = [v for v in versions if not v["historical"]]
    if not current:
        return None
    latest = current[-1]
    for v in versions:
        artifact = session.get(store.KbArtifact, v["artifact_id"])
        v["preserved"] = artifact is not None and store.bytes_preserved(session, artifact)
        v["locator_kind"] = artifact.locator_kind if artifact is not None else None
        v["duration_sec"] = artifact.duration_sec if artifact is not None else None
        v["hash_prefix"] = (v["sha256"] or "")[:12] or "unhashed"
    events_by_id = {e["id"]: e for e in case["events"]}
    events = []
    for e in case["events"]:
        events.append({**e, "badge": ACTOR_BADGES.get(e["actor_kind"], e["actor_kind"]),
                       "summary": _event_summary(e, events_by_id),
                       "retractable": (e["effective"] and not e["invalidated"] and e["actor_kind"] == "operator"
                                       and e["type"] in JUDGED_EVENTS),
                       "retract_key": _key()})
    state = case["state"]
    return {
        "id": case["id"], "title": _case_title(case), "scope": case["scope"], "holdout": case["holdout"],
        "run_case": run_case, "latest": latest, "state": state,
        "verdict_label": VERDICT_LABELS.get(state["verdict"], state["verdict"]),
        "previous_accepted": state.get("previous_accepted_version_no"),
        "older": [v for v in versions if v["version_no"] != latest["version_no"]],
        "events": events,
        "corrections": [e for e in events if e["type"] == "correction" and e["effective"]],
        "artifacts": versions,
        "feedback_key": _key(), "correction_key": _key(),
        "historical_keys": {v["version_no"]: _key() for v in versions},
        "kept": kept or {},
    }


def card_context(session: Session, scope: KnowledgeScope, run_id: str, *, banner: Optional[str] = None,
                 banner_kind: str = "info", kept: Optional[dict] = None) -> dict:
    """Read-only: one of ``not_done`` / ``start`` / ``cases`` (capture hooks §10)."""
    from ..models import Run

    run = feedback.scoped_run(session, scope, run_id)
    ctx = {"run_id": run_id, "banner": banner, "banner_kind": banner_kind, "start_error": None,
           "defect_codes": list(t.DEFECT_CODES), "severities": t.SEVERITIES, "verdicts": t.VERDICTS,
           "verdict_labels": VERDICT_LABELS, "methods": t.CORRECTION_METHODS, "resolved_values": t.RESOLVED,
           "external_key": _key(), "cases": []}
    status = getattr(run.status, "value", run.status)
    if status != "done":
        ctx["state"] = "not_done"
        return ctx
    try:
        timeline = store.run_timeline(session, scope, run_id)
    except KnowledgeAccessDenied:
        raise feedback.FeedbackError(404, "Knowledge resource unavailable", kind="not_found") from None
    kept = kept or {}
    blocks = []
    for case in timeline["cases"]:
        if case["scope"] == "segment":
            continue  # segment cases live in the JSON timeline; the card judges deliverables (§7 item 1)
        block = _case_block(session, case, case["scope"] == "run", kept.get(case["id"]))
        if block is not None:
            blocks.append(block)
    has_run_version = any(b["run_case"] for b in blocks)
    ctx["state"] = "cases" if has_run_version else "start"
    ctx["cases"] = blocks
    ctx["start_key"] = _key()
    session.rollback()
    return ctx


def _render(request: Request, ctx: dict, status_code: int = 200) -> HTMLResponse:
    from ..web_v2 import templates

    return templates.TemplateResponse(request, "partials/knowledge_feedback.html", ctx, status_code=status_code)


def _respond(request: Request, db: Session, scope: KnowledgeScope, run_id: str, *, banner=None,
             banner_kind="info", status_code=200, kept=None):
    try:
        ctx = card_context(db, scope, run_id, banner=banner, banner_kind=banner_kind, kept=kept)
    except feedback.FeedbackError as exc:
        if exc.status == 404:
            return Response(status_code=404)
        raise
    return _render(request, ctx, status_code)


def _error_response(request: Request, db: Session, scope: KnowledgeScope, run_id: str,
                    exc: feedback.FeedbackError, kept=None):
    if exc.status == 404:
        return Response(status_code=404)
    message = exc.message
    if exc.kind == "conflict" and exc.current_version_no is not None:
        # The refused request's sync rolled back with it: record the live final
        # now (done-gated, state-idempotent) so the card shows what was judged against.
        try:
            feedback.start_run(db, scope, run_id)
        except feedback.FeedbackError:
            pass
        message += f" — now showing version {exc.current_version_no}; review it and submit again."
    return _respond(request, db, scope, run_id, banner=message, banner_kind="danger" if exc.status != 409
                    else "warning", status_code=exc.status, kept=kept)


def _run_id_for_case(db: Session, scope: KnowledgeScope, case_id: str) -> str:
    """The Run whose card holds this case (root: itself; child: its parent's run)."""
    invalid = store.InvalidSet.load(db)
    try:
        case = store._scoped_case(db, scope, case_id, invalid)
    except KnowledgeAccessDenied:
        db.rollback()
        raise feedback.FeedbackError(404, "Knowledge resource unavailable", kind="not_found") from None
    run_id = case.run_id
    if run_id is None and case.parent_case_id is not None:
        run_id = db.get(store.KbCase, case.parent_case_id).run_id
    db.rollback()
    if run_id is None:
        raise feedback.FeedbackError(404, "Knowledge resource unavailable", kind="not_found")
    return run_id


# ---------------------------------------------------------------- form parsing


def _blank(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _float(value) -> Optional[float]:
    value = _blank(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        raise feedback.FeedbackError(422, f"{value!r} is not a number of seconds", kind="invalid") from None


def _int(value, what: str) -> Optional[int]:
    value = _blank(value)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise feedback.FeedbackError(422, f"{what} must be a whole number", kind="invalid") from None


def _validated(model, raw: dict):
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_context=False)[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise feedback.FeedbackError(422, f"{where}: {first.get('msg')}", kind="invalid") from None


def _defects_from_form(form) -> list[dict]:
    codes, severities = form.getlist("defect_code"), form.getlist("defect_severity")
    starts, ends, notes = form.getlist("defect_start"), form.getlist("defect_end"), form.getlist("defect_note")
    out = []
    for i, code in enumerate(codes):
        if not _blank(code):
            continue  # an untouched defect row
        raw = {"code": code.strip(), "severity": _blank(severities[i] if i < len(severities) else None),
               "start": _float(starts[i] if i < len(starts) else None),
               "end": _float(ends[i] if i < len(ends) else None),
               "note": _blank(notes[i] if i < len(notes) else None)}
        out.append({k: v for k, v in raw.items() if v is not None})
    return out


def _kept_feedback(form) -> dict:
    return {"verdict": _blank(form.get("verdict")), "note": form.get("note") or "",
            "defects": [{"code": c, "severity": s, "start": a, "end": b, "note": n} for c, s, a, b, n in zip(
                form.getlist("defect_code"), form.getlist("defect_severity"), form.getlist("defect_start"),
                form.getlist("defect_end"), form.getlist("defect_note")) if _blank(c)]}


# ---------------------------------------------------------------- routes


@router.get("/runs/{run_id}/knowledge-fragment", response_class=HTMLResponse)
def knowledge_fragment(run_id: str, request: Request, scope: KnowledgeScope = Depends(get_card_scope),
                       db: Session = Depends(get_db)):
    """Read-only GET: never creates a case or a version."""
    return _respond(request, db, scope, run_id)


@router.post("/runs/{run_id}/knowledge/start", response_class=HTMLResponse)
def knowledge_start(run_id: str, request: Request, scope: KnowledgeScope = Depends(get_card_scope),
                    db: Session = Depends(get_db)):
    try:
        feedback.start_run(db, scope, run_id)
    except feedback.FeedbackError as exc:
        return _error_response(request, db, scope, run_id, exc)
    return _respond(request, db, scope, run_id)


@router.post("/knowledge/cases/{case_id}/feedback", response_class=HTMLResponse)
async def knowledge_feedback(case_id: str, request: Request, scope: KnowledgeScope = Depends(get_card_scope),
                             db: Session = Depends(get_db)):
    form = await request.form()
    return await run_in_threadpool(_feedback_sync, case_id, request, form, scope, db)


def _feedback_sync(case_id, request, form, scope, db):
    try:
        run_id = _run_id_for_case(db, scope, case_id)
    except feedback.FeedbackError:
        return Response(status_code=404)
    kept = {case_id: _kept_feedback(form)}
    try:
        body = _validated(FeedbackIn, {
            "idempotency_key": _blank(form.get("idempotency_key")),
            "expected_version_no": _int(form.get("expected_version_no"), "expected_version_no"),
            "historical": _blank(form.get("historical")) == "true", "verdict": _blank(form.get("verdict")),
            "defects": _defects_from_form(form), "note": _blank(form.get("note")),
            "actor_label": _blank(form.get("actor_label"))})
        feedback.submit_feedback(db, scope, case_id, idempotency_key=body.idempotency_key,
                                 expected_version_no=body.expected_version_no, verdict=body.verdict,
                                 defects=defects_for_store(body.defects), note=body.note,
                                 historical=body.historical, actor_label=body.actor_label)
    except feedback.FeedbackError as exc:
        return _error_response(request, db, scope, run_id, exc, kept=kept)
    return _respond(request, db, scope, run_id, banner="Saved.")


@router.post("/knowledge/cases/{case_id}/corrections", response_class=HTMLResponse)
async def knowledge_correction(case_id: str, request: Request, scope: KnowledgeScope = Depends(get_card_scope),
                               db: Session = Depends(get_db)):
    form = await request.form()
    return await run_in_threadpool(_correction_sync, case_id, request, form, scope, db)


def _correction_sync(case_id, request, form, scope, db):
    try:
        run_id = _run_id_for_case(db, scope, case_id)
    except feedback.FeedbackError:
        return Response(status_code=404)
    try:
        body = _validated(CorrectionIn, {k: v for k, v in {
            "idempotency_key": _blank(form.get("idempotency_key")),
            "before_artifact_id": _blank(form.get("before_artifact_id")),
            "after_artifact_id": _blank(form.get("after_artifact_id")),
            "method": _blank(form.get("method")), "resolved": _blank(form.get("resolved")),
            "minutes": _float(form.get("minutes")), "note": _blank(form.get("note")),
            "actor_label": _blank(form.get("actor_label"))}.items() if v is not None})
        feedback.record_correction(db, scope, case_id, idempotency_key=body.idempotency_key,
                                   before_artifact_id=body.before_artifact_id,
                                   after_artifact_id=body.after_artifact_id, method=body.method,
                                   resolved=body.resolved, operator_minutes=body.minutes, note=body.note,
                                   actor_label=body.actor_label)
    except feedback.FeedbackError as exc:
        return _error_response(request, db, scope, run_id, exc)
    return _respond(request, db, scope, run_id, banner="Correction recorded.")


@router.post("/knowledge/events/{event_id}/retract", response_class=HTMLResponse)
async def knowledge_retract(event_id: str, request: Request, scope: KnowledgeScope = Depends(get_card_scope),
                            db: Session = Depends(get_db)):
    form = await request.form()
    return await run_in_threadpool(_retract_sync, event_id, request, form, scope, db)


def _retract_sync(event_id, request, form, scope, db):
    event = db.get(store.KbFeedbackEvent, event_id)
    if event is None:
        return Response(status_code=404)
    try:
        run_id = _run_id_for_case(db, scope, event.case_id)
    except feedback.FeedbackError:
        return Response(status_code=404)
    try:
        body = _validated(RetractIn, {k: v for k, v in {
            "idempotency_key": _blank(form.get("idempotency_key")), "reason": _blank(form.get("reason")),
            "actor_label": _blank(form.get("actor_label"))}.items() if v is not None})
        feedback.retract(db, scope, event_id, idempotency_key=body.idempotency_key, reason=body.reason,
                         actor_label=body.actor_label)
    except feedback.FeedbackError as exc:
        return _error_response(request, db, scope, run_id, exc)
    return _respond(request, db, scope, run_id, banner="Retracted.")


@router.post("/runs/{run_id}/knowledge/external-repair", response_class=HTMLResponse)
async def knowledge_external_repair(run_id: str, request: Request,
                                    scope: KnowledgeScope = Depends(get_card_scope),
                                    db: Session = Depends(get_db)):
    try:
        parsed = await read_external_repair(request)
    except HTTPException as exc:
        err = feedback.FeedbackError(exc.status_code, str(exc.detail), kind="invalid")
        return await run_in_threadpool(_error_response, request, db, scope, run_id, err)
    return await run_in_threadpool(_external_sync, run_id, request, parsed, scope, db)


def _external_sync(run_id, request, parsed, scope, db):
    try:
        external_repair_call(db, scope, run_id, parsed)
    except feedback.FeedbackError as exc:
        return _error_response(request, db, scope, run_id, exc)
    return _respond(request, db, scope, run_id, banner="External repair linked.")
