"""``/api/v2/knowledge`` — timeline, operator feedback and rule lifecycle (design §6.6).

404 unless ``KNOWLEDGE_ENABLED``; every id is resolved to its stored project
and checked against ``KnowledgeScope`` (out of scope = 404, same as missing).
The feedback/correction/retraction/external-repair writes and the done-only
``start`` bootstrap (AICP-299) go through ``knowledge.feedback``, which the
Run-page card's HTMX twins (``knowledge.web``) share.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import Field, StrictBool, StrictInt, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..assistant_contracts import ContractModel
from ..config import settings
from ..db import get_db
from . import feedback, store
from . import taxonomy as t
from .access import KnowledgeAccessDenied, KnowledgeScope
from .models import KbRule, KbRuleReview, KbRuleVersion

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

IdempotencyKey = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]
Label = Annotated[str, Field(min_length=1, max_length=128)]


def get_scope() -> KnowledgeScope:
    if getattr(settings, "KNOWLEDGE_ENABLED", False) is not True:
        raise HTTPException(404, "Not Found")
    return KnowledgeScope.from_settings(settings)


def _call(session: Session, fn, *args, **kwargs):
    try:
        result = fn(session, *args, **kwargs)
        session.commit()
        return result
    except KnowledgeAccessDenied:
        session.rollback()
        raise HTTPException(404, "Knowledge resource unavailable") from None
    except store.KnowledgeInvalid as exc:
        session.rollback()
        raise HTTPException(422, str(exc)) from None
    except store.KnowledgeConflict as exc:
        session.rollback()
        raise HTTPException(409, {"detail": str(exc), "current_version_no": exc.current_version_no}) from None
    except IntegrityError:
        session.rollback()
        raise HTTPException(409, "Knowledge request conflicts with current state; reload and retry") from None


class EvidenceIn(ContractModel):
    case_id: Annotated[str, Field(min_length=1, max_length=36)]
    event_id: Optional[Annotated[str, Field(min_length=1, max_length=36)]] = None
    stance: Literal["supports", "contradicts", "context"]
    note: Optional[Annotated[str, Field(max_length=2000)]] = None


class Recommendation(ContractModel):
    action: Literal["forbid", "require", "avoid", "prefer", "warn"]
    target: str | dict[str, Any]


class RuleVersionIn(ContractModel):
    idempotency_key: IdempotencyKey
    kind: Literal["operator_constraint", "empirical_guidance", "machine_hypothesis"]
    statement: Annotated[str, Field(min_length=1, max_length=4000)]
    rationale: Optional[Annotated[str, Field(max_length=8000)]] = None
    applicability: dict[str, Any] = {}
    recommendation: Recommendation
    confidence: Literal["operator_asserted", "low", "medium", "high"]
    shareable: bool = False
    scope_project_ids: Optional[Annotated[list[str], Field(min_length=1, max_length=50)]] = None
    model_registry_revision: Optional[Annotated[str, Field(max_length=64)]] = None
    effective_from: datetime
    effective_to: Optional[datetime] = None
    author_kind: Literal["operator", "claude"] = "operator"
    author_label: Optional[Label] = None
    provenance: dict[str, Any] = {}
    evidence_gaps: Annotated[list[str], Field(max_length=50)] = []
    evidence: Annotated[list[EvidenceIn], Field(max_length=100)] = []


class NewRuleIn(RuleVersionIn):
    slug: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,79}$")]


class ReviewIn(ContractModel):
    idempotency_key: IdempotencyKey
    status: Literal["proposed", "approved", "rejected", "retired"]
    reviewer_label: Optional[Label] = None
    note: Optional[Annotated[str, Field(max_length=4000)]] = None


def _version_out(session: Session, scope: KnowledgeScope, v: KbRuleVersion, effective_ids: set,
                 invalid: store.InvalidSet) -> Optional[dict]:
    if invalid.row_invalid("rule_version", v):
        return None  # withdrawn evidence is never browsable (§4.8)
    visibility = store.rule_visibility(session, scope, v)
    if visibility is None:
        return None
    reviews = session.execute(select(KbRuleReview).where(KbRuleReview.rule_version_id == v.id)
                              .order_by(KbRuleReview.seq)).scalars().all()
    evidence, hidden = [], 0
    for e in store.rule_evidence(session, v.id, invalid):
        case_project = session.get(store.KbCase, e.case_id).project_id
        if scope.allows_project(case_project):
            evidence.append({"case_id": e.case_id, "event_id": e.event_id, "stance": e.stance, "note": e.note})
        else:
            hidden += 1
    return {
        "id": v.id, "version_no": v.version_no, "kind": v.kind, "statement": v.statement,
        "rationale": v.rationale, "applicability": v.applicability, "recommendation": v.recommendation,
        "confidence": v.confidence, "shareable": v.shareable, "scope_project_ids": v.scope_project_ids
        if visibility == "full" else None, "model_registry_revision": v.model_registry_revision,
        "effective_from": v.effective_from.isoformat(),
        "effective_to": v.effective_to.isoformat() if v.effective_to else None,
        "author_kind": v.author_kind, "author_label": v.author_label if visibility == "full" else None,
        "author_verified": False,
        # Shareable attests only statement + rationale; free text beyond them stays in scope.
        "provenance": v.provenance if visibility == "full" else {},
        "evidence_gaps": v.evidence_gaps if visibility == "full" else [],
        "supersedes_version_id": v.supersedes_version_id, "effective": v.id in effective_ids,
        "reviews": [{"status": r.status, "reviewer_label": r.reviewer_label if visibility == "full" else None,
                     "note": r.note if visibility == "full" else None,
                     "created_at": r.created_at.isoformat()} for r in reviews],
        "evidence": evidence, "out_of_scope_evidence_count": hidden,
    }


@router.get("/runs/{run_id}/timeline")
def run_timeline(run_id: str, scope: KnowledgeScope = Depends(get_scope), db: Session = Depends(get_db)):
    return _call(db, store.run_timeline, scope, run_id)


@router.get("/rules")
def list_rules(scope: KnowledgeScope = Depends(get_scope), db: Session = Depends(get_db)):
    invalid = store.InvalidSet.load(db)
    effective_ids = {v.id for v in store.effective_rule_versions(db, invalid=invalid)}
    out = []
    for rule in db.execute(select(KbRule).order_by(KbRule.seq)).scalars():
        rows = db.execute(select(KbRuleVersion).where(KbRuleVersion.rule_id == rule.id)
                          .order_by(KbRuleVersion.version_no)).scalars().all()
        visible = [x for x in (_version_out(db, scope, v, effective_ids, invalid) for v in rows) if x]
        if visible:
            out.append({"id": rule.id, "slug": rule.slug, "versions": visible})
    return {"rules": out}


def _version_kwargs(body: RuleVersionIn) -> dict:
    data = body.model_dump(exclude={"slug"})
    data["recommendation"] = body.recommendation.model_dump()
    data["evidence"] = [e.model_dump() for e in body.evidence]
    return data


@router.post("/rules", status_code=201)
def create_rule(body: NewRuleIn, scope: KnowledgeScope = Depends(get_scope), db: Session = Depends(get_db)):
    version = _call(db, store.add_rule_version, scope, slug=body.slug, require_new=True, **_version_kwargs(body))
    return {"rule_id": version.rule_id, "rule_version_id": version.id, "version_no": version.version_no}


@router.post("/rules/{rule_id}/versions", status_code=201)
def add_version(rule_id: str, body: RuleVersionIn, scope: KnowledgeScope = Depends(get_scope),
                db: Session = Depends(get_db)):
    rule = db.get(KbRule, rule_id)
    latest = rule and db.execute(select(KbRuleVersion).where(KbRuleVersion.rule_id == rule.id)
                                 .order_by(KbRuleVersion.version_no.desc()).limit(1)).scalar_one_or_none()
    if latest is None or store.rule_visibility(db, scope, latest) != "full":
        raise HTTPException(404, "Knowledge resource unavailable")
    version = _call(db, store.add_rule_version, scope, slug=rule.slug, **_version_kwargs(body))
    return {"rule_id": version.rule_id, "rule_version_id": version.id, "version_no": version.version_no}


@router.post("/rule-versions/{version_id}/reviews", status_code=201)
def review(version_id: str, body: ReviewIn, scope: KnowledgeScope = Depends(get_scope),
           db: Session = Depends(get_db)):
    result = _call(db, store.review_rule_version, scope, rule_version_id=version_id, **body.model_dump())
    return {"review_id": result.id, "status": result.status}


# ---------------------------------------------------------------- operator feedback (AICP-299)

ShortId = Annotated[str, Field(min_length=1, max_length=36)]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Note = Annotated[str, Field(max_length=4000)]
DefectCode = Literal[tuple(t.DEFECT_CODES)]  # type: ignore[valid-type]


class DefectIn(ContractModel):
    code: DefectCode
    severity: Literal[t.SEVERITIES]  # type: ignore[valid-type]
    start: Optional[Seconds] = None
    end: Optional[Seconds] = None
    note: Optional[Note] = None


class FeedbackIn(ContractModel):
    idempotency_key: IdempotencyKey
    expected_version_no: Annotated[StrictInt, Field(ge=1)]
    historical: StrictBool = False
    verdict: Optional[Literal[t.VERDICTS]] = None  # type: ignore[valid-type]
    defects: Annotated[list[DefectIn], Field(max_length=20)] = []
    note: Optional[Note] = None
    actor_label: Optional[Label] = None


class CorrectionIn(ContractModel):
    idempotency_key: IdempotencyKey
    before_artifact_id: ShortId
    after_artifact_id: Optional[ShortId] = None
    method: Literal[t.CORRECTION_METHODS]  # type: ignore[valid-type]
    resolved: Literal[t.RESOLVED] = "unknown"  # type: ignore[valid-type]
    mcredits: Optional[Annotated[StrictInt, Field(ge=0)]] = None
    cost_source: Optional[Literal[t.COST_SOURCES]] = None  # type: ignore[valid-type]
    minutes: Optional[Seconds] = None
    note: Optional[Note] = None
    actor_label: Optional[Label] = None


class RetractIn(ContractModel):
    idempotency_key: IdempotencyKey
    reason: Annotated[str, Field(min_length=1, max_length=2000)]
    actor_label: Optional[Label] = None


class ExternalRepairIn(ContractModel):
    """The external repair's fields; ``upload`` travels beside it (multipart only)."""
    idempotency_key: IdempotencyKey
    drive_link: Optional[Annotated[str, Field(min_length=1, max_length=2000)]] = None
    replaced_delivered: Optional[StrictBool] = None
    note: Optional[Note] = None
    actor_label: Optional[Label] = None


def _feedback_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except feedback.FeedbackError as exc:
        if exc.status == 409:
            raise HTTPException(409, {"detail": exc.message, "current_version_no": exc.current_version_no,
                                      "kind": exc.kind}) from None
        raise HTTPException(exc.status, exc.message) from None


def defects_for_store(defects) -> list[dict]:
    """Drop absent optional keys so the request hash matches whichever surface sent it."""
    return [{k: v for k, v in d.model_dump().items() if v is not None} for d in defects]


@router.post("/cases/{case_id}/feedback")
def post_feedback(case_id: str, body: FeedbackIn, scope: KnowledgeScope = Depends(get_scope),
                  db: Session = Depends(get_db)):
    out = _feedback_call(feedback.submit_feedback, db, scope, case_id, idempotency_key=body.idempotency_key,
                         expected_version_no=body.expected_version_no, verdict=body.verdict,
                         defects=defects_for_store(body.defects), note=body.note, historical=body.historical,
                         actor_label=body.actor_label)
    out["state"] = store.case_state(db, case_id)
    db.rollback()
    return out


@router.post("/cases/{case_id}/corrections")
def post_correction(case_id: str, body: CorrectionIn, scope: KnowledgeScope = Depends(get_scope),
                    db: Session = Depends(get_db)):
    return _feedback_call(feedback.record_correction, db, scope, case_id, idempotency_key=body.idempotency_key,
                          before_artifact_id=body.before_artifact_id, after_artifact_id=body.after_artifact_id,
                          method=body.method, resolved=body.resolved, incremental_mcredits=body.mcredits,
                          cost_source=body.cost_source, operator_minutes=body.minutes, note=body.note,
                          actor_label=body.actor_label)


@router.post("/events/{event_id}/retract")
def post_retract(event_id: str, body: RetractIn, scope: KnowledgeScope = Depends(get_scope),
                 db: Session = Depends(get_db)):
    return _feedback_call(feedback.retract, db, scope, event_id, idempotency_key=body.idempotency_key,
                          reason=body.reason, actor_label=body.actor_label)


@router.post("/runs/{run_id}/start")
def post_start(run_id: str, scope: KnowledgeScope = Depends(get_scope), db: Session = Depends(get_db)):
    """Done-only, state-idempotent bootstrap of the Run case (capture hooks §10)."""
    return _feedback_call(feedback.start_run, db, scope, run_id)


def _form_bool(value) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.lower() in ("true", "1", "on", "yes"):
        return True
    if isinstance(value, str) and value.lower() in ("false", "0", "off", "no"):
        return False
    raise HTTPException(422, "replaced_delivered must be true or false")


async def read_external_repair(request: Request) -> dict:
    """``{idempotency_key, drive_link | upload, replaced_delivered?, note?, actor_label?}``,
    as JSON (Drive link) or multipart (either). Validated by ``ExternalRepairIn``."""
    content_type = request.headers.get("content-type", "")
    upload = None
    if content_type.startswith("application/json"):
        try:
            raw = await request.json()
        except ValueError:
            raise HTTPException(422, "Invalid JSON body") from None
        if not isinstance(raw, dict):
            raise HTTPException(422, "Invalid JSON body")
    else:
        # Refuse before Starlette spools a single byte of a multipart body to disk:
        # the size cap and the free-space guard apply to the declared length.
        declared = request.headers.get("content-length")
        if declared is None or not declared.isdigit():
            raise HTTPException(411, "Content-Length is required for an upload")
        if int(declared) > feedback.UPLOAD_MAX_BYTES + feedback.MULTIPART_OVERHEAD_BYTES:
            raise HTTPException(413, "The upload is larger than 500 MB — use a Drive link instead")
        import os as _os
        _os.makedirs(feedback.knowledge_root(), exist_ok=True)
        if not feedback._disk_ok(feedback.knowledge_root(), 2 * int(declared)):  # spool + stored copy
            raise HTTPException(507, "Not enough free disk space for this upload; use a Drive link")
        form = await request.form()
        raw = {k: v for k, v in form.items() if k != "upload" and v != ""}
        candidate = form.get("upload")
        if candidate is not None and not isinstance(candidate, str) and getattr(candidate, "filename", ""):
            upload = candidate
        if "replaced_delivered" in raw:
            raw["replaced_delivered"] = _form_bool(raw["replaced_delivered"])
    try:
        body = ExternalRepairIn.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from None
    if (body.drive_link is None) == (upload is None):
        raise HTTPException(422, "Give either a Drive link or an upload")
    return {"idempotency_key": body.idempotency_key, "drive_link": body.drive_link,
            "upload": upload, "replaced_delivered": body.replaced_delivered, "note": body.note,
            "actor_label": body.actor_label}


def external_repair_call(db: Session, scope: KnowledgeScope, run_id: str, parsed: dict) -> dict:
    upload = parsed["upload"]
    return feedback.link_external_repair(
        db, scope, run_id, idempotency_key=parsed["idempotency_key"], drive_link=parsed["drive_link"],
        upload=upload.file if upload is not None else None,
        upload_name=upload.filename if upload is not None else None,
        upload_size=getattr(upload, "size", None) if upload is not None else None,
        replaced_delivered=parsed["replaced_delivered"], note=parsed["note"], actor_label=parsed["actor_label"])


@router.post("/runs/{run_id}/external-repair", status_code=201)
async def post_external_repair(run_id: str, request: Request, scope: KnowledgeScope = Depends(get_scope),
                               db: Session = Depends(get_db)):
    parsed = await read_external_repair(request)
    return await run_in_threadpool(_feedback_call, external_repair_call, db, scope, run_id, parsed)
