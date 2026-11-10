"""Private operator assistant endpoints, behind the existing nginx login.

No public token routes, local paths, provider links or arbitrary metadata are
returned. Every request and media delivery checks current project scope.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Annotated, Literal
import uuid
import time
import hashlib
import re
import math
import threading
from fractions import Fraction

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import Field, StrictInt, field_validator
from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from .assistant_access import AssistantAccessDenied, ProjectAllowlist, canonical_uuid
from .assistant_contracts import ContractModel, EditPlan, EntityId, TimeRange, MAX_OPERATIONS
from .assistant_jobs import AssistantJob, JobConflict, _atomic, request_cancel as cancel_job
from .assistant_models import AssistantArtifact, AssistantMessage, AssistantSelection, AssistantVersion, AssistantVersionArtifact
from .assistant_planner import PlannerContext, ProposeReply
from .assistant_service import AssistantAdmissionService, AssistantCapabilityError, AssistantAdmissionConflict
from .assistant_store import AssistantArtifactStorage, AssistantStoreError, get_version_artifact, select_version, verify_artifact
from .assistant_run_result import (PublicationUnavailable, history_media, promote_run_result, promote_run_result_published, run_result_context)
from .assistant_preparation import (PreparationCapabilityError, needs_preparation, safe_preparation,
    prepared_child, preparation_receipts, prepare_editable_version)
from .assistant_media import AssistantMediaError, AssistantMediaTimeout
from .assistant_legacy import import_legacy_run
from .assistant_turns import AssistantTurn, AssistantTurnEdit, AssistantTurnService, TurnConflict, TurnContextTooLarge, TurnsDisabled, request_cancel
from .config import settings
from .db import get_db
from .ops_gate import GATED
from .storage import data_dir_root

router = APIRouter(prefix="/assistant", tags=["assistant"])
# Current private deployment has one API process. Native preparation admission
# is nonblocking and bounded independently of the FastAPI thread pool.
_PREPARATION_SLOT = threading.BoundedSemaphore(1)


class SourceReference(ContractModel):
    asset_id: EntityId
    version_id: EntityId


class StartTurn(ContractModel):
    base_version_id: EntityId
    instruction: Annotated[str, Field(min_length=1, max_length=4000)]
    idempotency_key: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]
    sources: Annotated[tuple[SourceReference, ...], Field(max_length=32)] = ()
    selected_passage: TimeRange | None = None
    proposal_turn_id: EntityId | None = None


class RetryTurn(ContractModel):
    idempotency_key: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]


class AcceptProposal(ContractModel):
    operation_indexes: Annotated[tuple[Annotated[StrictInt, Field(ge=0)], ...],
        Field(min_length=1, max_length=MAX_OPERATIONS)] | None = None

    @field_validator("operation_indexes")
    @classmethod
    def ordered_unique(cls, value):
        if value is not None and list(value) != sorted(set(value)):
            raise ValueError("change numbers must be unique and in proposal order")
        return value


class PublishRunResult(ContractModel):
    version_id: EntityId
    expected_selection_epoch: Annotated[StrictInt, Field(ge=0)]
    expected_result_revision: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    idempotency_key: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]


class SelectVersion(ContractModel):
    version_id: EntityId
    expected_epoch: Annotated[StrictInt, Field(ge=0)]


def get_assistant_service() -> AssistantTurnService:
    enabled = getattr(settings, "ASSISTANT_ENABLED", False) is True
    if not enabled:
        raise HTTPException(404, "Assistant unavailable")
    return AssistantTurnService(storage=AssistantArtifactStorage(data_dir_root()),
        policy=ProjectAllowlist.from_settings(settings), enabled=True,
        agent_enabled=getattr(settings, "ASSISTANT_AGENT_ENABLED", False) is True)


def _now():
    return datetime.now(timezone.utc)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except AssistantAccessDenied:
        raise HTTPException(404, "Assistant resource unavailable") from None
    except PreparationCapabilityError:
        raise HTTPException(422, "Prepare an editable copy of this version before using it for editing or publication") from None
    except AssistantMediaTimeout:
        raise HTTPException(503, "Media preparation or inspection reached its time limit; the original is unchanged. Retry the same saved request") from None
    except AssistantMediaError:
        raise HTTPException(422, "This media profile cannot be safely edited or prepared; the original is unchanged") from None
    except TurnsDisabled:
        raise HTTPException(404, "Assistant unavailable") from None
    except TurnContextTooLarge:
        raise HTTPException(422, "Assistant initial context exceeds the supported 32 KiB limit; reduce the request or selected context") from None
    except (TurnConflict, AssistantStoreError, JobConflict, AssistantAdmissionConflict):
        raise HTTPException(409, "Assistant request conflicts with current state; reload and retry") from None
    except AssistantCapabilityError as exc:
        # These messages are authored by the trusted capability checks, never
        # subprocess output, file paths, credentials or model-provided errors.
        raise HTTPException(422, "This edit cannot be rendered: " + str(exc)) from None
    except ValueError:
        raise HTTPException(422, "Assistant request is invalid or unsupported") from None


def _safe_fps(value):
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]{0,5}(?:/[1-9][0-9]{0,5})?", value) is None:
        return None
    return value if 0 < Fraction(value) <= 120 else None


def _version(version: AssistantVersion, run_id: str, db: Session | None = None, *, fps: str | None = None):
    root = f"/api/v2/assistant/runs/{run_id}/versions/{version.id}"
    preview_metadata = {}
    duration_sec = None
    preparation, prepared_version_id, source_version_id = None, None, None
    requires_preparation = False
    fps = _safe_fps(fps)
    if db is not None and version.status == "succeeded":
        try:
            full_artifact = get_version_artifact(db, version.id, role="full")
            full_metadata = full_artifact.metadata_json
            requires_preparation = needs_preparation(full_artifact)
            if requires_preparation:
                preparation = safe_preparation(full_artifact)
                child = prepared_child(db, version.id)
                prepared_version_id = child.id if child else None
            if version.recipe.get("kind") == "audio_tail_preparation_v1":
                source_version_id = version.recipe.get("source_version_id")
            if isinstance(full_metadata,dict):
                duration = full_metadata.get('duration_sec')
                if type(duration) in (int,float) and 0 < duration <= 86400 and math.isfinite(duration):
                    duration_sec = float(duration)
            if fps is None and isinstance(full_metadata, dict):
                fps = _safe_fps(full_metadata.get("fps"))
            metadata = get_version_artifact(db, version.id, role="preview").metadata_json
            if isinstance(metadata, dict):
                for key in ("preview_mode", "sequence", "switch_sec", "source_start_sec", "source_end_sec", "affected_start_sec", "affected_end_sec", "preview_truncated", "preview_quality", "full_quality_preserved"):
                    value = metadata.get(key)
                    if key == "preview_mode" and value in ("before_after", "affected_window", "full_as_preview"):
                        preview_metadata[key] = value
                    elif key == "sequence" and value == ["before", "after"]:
                        preview_metadata[key] = value
                    elif key in ("source_start_sec", "source_end_sec", "affected_start_sec", "affected_end_sec") and type(value) in (int, float) and 0 <= value <= 86400:
                        preview_metadata[key] = value
                    elif key == "switch_sec" and type(value) in (int, float) and 0 < value <= 30.04:
                        preview_metadata[key] = value
                    elif key == "preview_truncated" and type(value) is bool:
                        preview_metadata[key] = value
                    elif key == "preview_quality" and value == "proxy_max_dimension_640":
                        preview_metadata[key] = value
                    elif key == "full_quality_preserved" and value is True:
                        preview_metadata[key] = value
        except AssistantStoreError:
            pass
    return {"id": version.id, "needs_editing_preparation": requires_preparation,
        "editing_preparation": preparation, "prepared_version_id": prepared_version_id,
        "preparation_source_version_id": source_version_id, "sequence": version.sequence, "fps": fps, "duration_sec":duration_sec, "preview_metadata": preview_metadata, "parent_version_id": version.parent_version_id,
        "base_version_id": version.base_version_id, "status": version.status, "created_at": version.created_at,
        "completed_at": version.completed_at, "preview_url": root + "/media/preview" if version.status == "succeeded" else None,
        "full_url": root + "/media/full" if version.status == "succeeded" else None}


def _context_visible(turn: AssistantTurn, db: Session, service: AssistantTurnService, *,
        authorization_cache: dict | None = None, agent_states: dict | None = None) -> bool:
    context = service.effective_context(turn)
    try:
        service.reauthorize_context(db, context, verify_bytes=False, authorization_cache=authorization_cache)
        if turn.workflow_revision == 2:
            state = service.agent_public_state(db, turn)
            if agent_states is not None:
                agent_states[turn.id] = state
        return True
    except AssistantAccessDenied:
        if agent_states is not None:
            agent_states.pop(turn.id, None)
        return False


def _turn(turn: AssistantTurn, db: Session, service: AssistantTurnService, *, visible: bool | None = None,
        agent_states: dict | None = None):
    context = service.effective_context(turn)
    agent_states = {} if agent_states is None else agent_states
    visible = _context_visible(turn, db, service, agent_states=agent_states) if visible is None else visible
    accepted_indexes = None
    agent = None
    if visible and turn.workflow_revision == 2:
        agent = agent_states[turn.id] if turn.id in agent_states else service.agent_public_state(db, turn)
    if agent and agent.get("verification"):
        candidate_id = agent["verification"]["candidate_id"]
        agent["candidate_url"] = f"/api/v2/assistant/runs/{context.run_id}/turns/{turn.id}/candidates/{candidate_id}/media"
    if visible and turn.edit_job_id and turn.reply and turn.reply.get("kind") == "propose":
        job = db.get(AssistantJob, turn.edit_job_id)
        if job is not None and job.pinned_base_version_id == turn.base_version_id:
            accepted = job.accepted_plan.get("operations", [])
            proposed = turn.reply["plan"]["operations"]
            accepted_indexes = [index for index, operation in enumerate(proposed) if operation in accepted]
    return {"id": turn.id, "base_version_id": turn.base_version_id, "operator_message_id": turn.operator_message_id,
        "assistant_message_id": turn.assistant_message_id if visible else None, "parent_turn_id": turn.parent_turn_id, "status": turn.status,
        "reply": turn.reply if visible else None, "edit_job_id": turn.edit_job_id, "source_access_changed": not visible,
        "accepted_operation_indexes": accepted_indexes,
        "analysis": agent,
        "speech_evidence": [item.model_dump(mode="json") for item in context.speech_evidence] if visible else [],
        "evidence_limitations": context.evidence_limitations if visible else ["A previously referenced source is outside the current assistant scope; related analysis is hidden."],
        "selected_passage": context.selected_passage.model_dump(mode="json") if context.selected_passage else None, "error_code": turn.error_code,
        "created_at": turn.created_at, "finished_at": turn.finished_at}

@router.post("/runs/{run_id}/import")
def import_run(run_id: str, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    run = _call(service.require_run, db, run_id)
    with _atomic(db):
        version = _call(import_legacy_run, db, run, actor="operator", storage=service.storage, policy=service.policy)
    return _version(version, run.id, db)


@router.get("/runs/{run_id}/context")
def get_context(run_id: str, before_version_sequence: int | None = None,
        before_turn_id: str | None = None, limit: int = 100,
        bound_version_ids: Annotated[list[EntityId] | None, Query(max_length=8)] = None,
        db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    if type(limit) is not int or not 1 <= limit <= 100 or (before_version_sequence is not None
            and (type(before_version_sequence) is not int or before_version_sequence < 1)):
        raise HTTPException(422, "Invalid assistant history cursor or page limit")
    conversation = _call(service.conversation, db, run_id)
    from .assistant_models import AssistantConversation
    from .models import Run, VideoProject
    def scoped_versions(ids):
        if not ids:
            return []
        # Capability hints remain scoped to this exact live Run and Project;
        # allowing another project never transfers a version between chats.
        return db.scalars(select(AssistantVersion).join(AssistantConversation,
            AssistantConversation.id == AssistantVersion.conversation_id).join(Run,
            (Run.id == AssistantConversation.run_id) & (Run.project_id == AssistantConversation.project_id))
            .join(VideoProject, VideoProject.id == Run.project_id).where(
                AssistantConversation.id == conversation.id, AssistantVersion.id.in_(ids),
                service.policy.sql_filter(Run.project_id)).execution_options(populate_existing=True)).all()
    explicit_ids = set(bound_version_ids or ())
    capability_versions = {version.id: version for version in scoped_versions(explicit_ids)}
    if set(capability_versions) != explicit_ids:
        raise HTTPException(404, "Assistant resource unavailable")
    selection = db.get(AssistantSelection, conversation.id, populate_existing=True)
    version_query = select(AssistantVersion).where(AssistantVersion.conversation_id == conversation.id)
    if before_version_sequence is not None:
        version_query = version_query.where(AssistantVersion.sequence < before_version_sequence)
    versions = db.scalars(version_query.order_by(AssistantVersion.sequence.desc()).limit(limit + 1)).all()
    versions_truncated = len(versions) > limit
    versions = list(reversed(versions[:limit]))
    turn_query = select(AssistantTurn).where(AssistantTurn.conversation_id == conversation.id)
    if before_turn_id is not None:
        cursor_id = _call(canonical_uuid, before_turn_id)
        anchor = db.scalar(select(AssistantTurn).where(AssistantTurn.conversation_id == conversation.id, AssistantTurn.id == cursor_id))
        if anchor is None:
            raise HTTPException(404, "Assistant resource unavailable")
        turn_query = turn_query.where(or_(AssistantTurn.created_at < anchor.created_at,
            and_(AssistantTurn.created_at == anchor.created_at, AssistantTurn.id < anchor.id)))
    turns = db.scalars(turn_query.order_by(AssistantTurn.created_at.desc(), AssistantTurn.id.desc()).limit(limit + 1)).all()
    turns_truncated = len(turns) > limit
    turns = list(reversed(turns[:limit]))
    edits = db.scalars(select(AssistantTurnEdit).where(AssistantTurnEdit.turn_id.in_([turn.id for turn in turns])).order_by(AssistantTurnEdit.created_at, AssistantTurnEdit.id)).all()
    jobs = {job.id: job for job in db.scalars(select(AssistantJob).where(AssistantJob.id.in_([edit.job_id for edit in edits]))).all()}
    authorization_cache = {}
    agent_states = {}
    visibility = {turn.id: _context_visible(turn, db, service, authorization_cache=authorization_cache,
        agent_states=agent_states) for turn in turns}
    visible_bases = {turn.base_version_id for turn in turns if visibility[turn.id]}
    capability_versions.update({version.id: version for version in scoped_versions(visible_bases - set(capability_versions))})
    capability_flags = {version_id: metadata.get("needs_editing_preparation") is True
        for version_id, metadata in db.execute(select(AssistantVersionArtifact.version_id, AssistantArtifact.metadata_json)
            .join(AssistantArtifact, AssistantArtifact.id == AssistantVersionArtifact.artifact_id).where(
                AssistantVersionArtifact.version_id.in_(capability_versions), AssistantVersionArtifact.role == "full",
                AssistantVersionArtifact.slot == 0)).all() if isinstance(metadata, dict)}
    version_capabilities = [{"id": version.id, "status": version.status, "sequence": version.sequence,
        "needs_editing_preparation": capability_flags.get(version.id, False)}
        for version in sorted(capability_versions.values(), key=lambda version: version.sequence)]
    message_ids = {turn.operator_message_id for turn in turns}
    message_ids.update(turn.assistant_message_id for turn in turns if turn.assistant_message_id and visibility[turn.id])
    message_ids.update(edit.execution_message_id for edit in edits if edit.execution_message_id and visibility.get(edit.turn_id))
    messages = db.scalars(select(AssistantMessage).where(AssistantMessage.conversation_id == conversation.id,
        AssistantMessage.id.in_(message_ids)).order_by(AssistantMessage.sequence)).all()
    base_context = None
    if selection and selection.selected_version_id:
        selected_artifact = _call(get_version_artifact, db, selection.selected_version_id)
        _call(service.catalog(db).require_target, selection.selected_version_id)
        _call(verify_artifact, selected_artifact, storage=service.storage)
        if not needs_preparation(selected_artifact):
            base_context = _call(service.version_context, db, run_id=run_id, base_version_id=selection.selected_version_id)
    if base_context:
        base_context = {key: ([item.model_dump(mode="json") for item in value] if key == "subtitle_cues" else
            value.model_dump(mode="json") if hasattr(value, "model_dump") else value) for key, value in base_context.items()}
    current_versions = {version.id: version for version in db.scalars(select(AssistantVersion).where(
        AssistantVersion.conversation_id == conversation.id, AssistantVersion.id.in_([
            selection.selected_version_id, selection.latest_version_id] if selection else []))).all()}
    selected = current_versions.get(selection.selected_version_id) if selection else None
    latest = current_versions.get(selection.latest_version_id) if selection else None
    return {"run_id": conversation.run_id, "conversation_id": conversation.id, "base_context": base_context,
        "version_capabilities": version_capabilities,
        "preparation_receipts": preparation_receipts(db, conversation.id),
        "run_result": _call(run_result_context, db, service, run_id),
        "selected_version": _version(selected, conversation.run_id, db, fps=base_context["selected"].get("fps") if base_context else None) if selected else None,
        "latest_version": _version(latest, conversation.run_id, db) if latest else None,
        "history_cursors": {"before_version_sequence": versions[0].sequence if versions_truncated and versions else None,
            "before_turn_id": turns[0].id if turns_truncated and turns else None},
        "selected_version_id": selection.selected_version_id if selection else None,
        "latest_version_id": selection.latest_version_id if selection else None, "selection_epoch": selection.epoch if selection else 0,
        "history_truncated": versions_truncated or turns_truncated,
        "versions": [_version(version, conversation.run_id, db, fps=base_context["selected"].get("fps") if base_context and version.id == selection.selected_version_id else None) for version in versions], "turns": [_turn(turn, db, service, visible=visibility[turn.id], agent_states=agent_states) for turn in turns],
        "jobs": [{**_job(jobs[edit.job_id]), "turn_id": edit.turn_id, "execution_message_id": edit.execution_message_id,
            "explanation": next((message.content for message in messages if message.id == edit.execution_message_id), None)} for edit in edits if edit.job_id in jobs],
        "messages": [{"id": message.id, "sequence": message.sequence, "role": message.role,
            "content": message.content, "base_version_id": message.base_version_id} for message in messages]}


@router.post("/runs/{run_id}/turns", status_code=202, dependencies=[GATED])
def start_turn(run_id: str, body: StartTurn, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    turn = _call(service.accept, db, run_id=run_id, base_version_id=body.base_version_id, instruction=body.instruction,
        idempotency_key=body.idempotency_key, actor="operator", now=_now(),
        source_refs=tuple((source.asset_id, source.version_id) for source in body.sources), selected_passage=body.selected_passage,
        proposal_turn_id=body.proposal_turn_id)
    return _turn(turn, db, service)


@router.get("/runs/{run_id}/turns/{turn_id}")
def get_turn(run_id: str, turn_id: str, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    return _turn(_call(service.require_turn, db, run_id, turn_id), db, service)


@router.post("/runs/{run_id}/turns/{turn_id}/cancel")
def cancel_turn(run_id: str, turn_id: str, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    turn = _call(service.require_turn, db, run_id, turn_id)
    if turn.edit_job_id:
        job = _call(cancel_job, db, turn.edit_job_id, now=_now())
        return {"turn": _turn(turn, db, service), "job": _job(job)}
    return {"turn": _turn(_call(request_cancel, db, turn.id, now=_now()), db, service), "job": None}


@router.post("/runs/{run_id}/turns/{turn_id}/retry", status_code=202, dependencies=[GATED])
def retry_turn(run_id: str, turn_id: str, body: RetryTurn, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    return _turn(_call(service.retry, db, run_id=run_id, turn_id=turn_id, idempotency_key=body.idempotency_key, now=_now()), db, service)


@router.post("/runs/{run_id}/turns/{turn_id}/accept", status_code=202, dependencies=[GATED])
def accept_proposal(run_id: str, turn_id: str, body: AcceptProposal | None = None, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    turn = _call(service.require_turn, db, run_id, turn_id)
    with _atomic(db):
        db.execute(update(AssistantTurn).where(AssistantTurn.id == turn.id).values(updated_at=_now()).execution_options(synchronize_session=False))
        turn = _call(service.require_turn, db, run_id, turn_id)
        if turn.status != "succeeded" or not turn.reply or turn.reply.get("kind") != "propose":
            raise HTTPException(409, "This turn has no edit proposal")
        reply = ProposeReply.model_validate(turn.reply)
        plan = reply.plan
        if body is not None and body.operation_indexes is not None:
            if any(index >= len(plan.operations) for index in body.operation_indexes):
                raise HTTPException(422, "Selected change number is not in this proposal")
            plan = EditPlan.model_validate({**plan.model_dump(mode="json"),
                "operations": [plan.operations[index].model_dump(mode="json") for index in body.operation_indexes]})
        context = _call(service.effective_context, turn)
        _call(service.reauthorize_context, db, context)
        admission = AssistantAdmissionService(storage=service.storage, policy=service.policy, enabled=True)
        receipt = _call(admission.admit, db, run_id=run_id, plan=plan, instruction=context.dialogue[-1].text,
            idempotency_key="turn:" + turn.id, actor=turn.actor, now=_now(), explicit_base_version_id=turn.base_version_id,
            verified_candidate_turn_id=turn.id)
        existing_edit = db.scalar(select(AssistantTurnEdit).where(AssistantTurnEdit.turn_id == turn.id,
            AssistantTurnEdit.idempotency_key == "accept"))
        if existing_edit is None:
            db.add(AssistantTurnEdit(id=str(uuid.uuid4()), turn_id=turn.id, job_id=receipt.job_id,
                idempotency_key="accept", created_at=_now()))
        if turn.edit_job_id is None:
            turn.edit_job_id = receipt.job_id
        db.flush()
        return {"turn": _turn(turn, db, service), "job": _job(db.get(AssistantJob, receipt.job_id))}


class ApplyCaptionLines(ContractModel):
    # Required when the script used a library-entry donor the resolver could
    # not prove (lineage library_donor_attestation_required).
    attest_library_donor: bool = False


@router.post("/runs/{run_id}/turns/{turn_id}/apply-captions", dependencies=[GATED])
def apply_caption_lines(run_id: str, turn_id: str, body: ApplyCaptionLines | None = None,
                        db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    """Apply a caption_lines proposal (AICP-345): exactly the operator's manual
    PATCH /captions/{id}/lines + POST .../retry, in ONE transaction.

    The lines come from the turn's durable script_captions evidence, never
    from the request. Run then caption rows are locked; the caption must be
    unchanged since the proposal (lines, language, source, done/failed) and
    the run result must still be the hashed final. Once per turn: a repeat
    returns the recorded application without queueing again.
    """
    from . import spend as spend_mod
    from .api_v2 import _admit_retained_run_source, _spend_reserve
    from .assistant_agent_contracts import script_evidence
    from .assistant_agent_steps import AssistantAgentStep, restore_step
    from .assistant_planner import CaptionLinesReply
    from .assistant_qa import digest
    from .assistant_script_captions import ScriptCaptionsRefused, hash_file
    from .assistant_store import append_message
    from .models import CaptionStatus, Run, RunCaption
    from .state_machine import transition
    from .tasks import enqueue_process_caption

    body = body or ApplyCaptionLines()
    turn = _call(service.require_turn, db, run_id, turn_id)
    with _atomic(db):
        # Serialize Apply per turn (the same row-lock idiom as accept).
        db.execute(update(AssistantTurn).where(AssistantTurn.id == turn.id).values(updated_at=_now()).execution_options(synchronize_session=False))
        turn = _call(service.require_turn, db, run_id, turn_id)
        if (turn.status != "succeeded" or turn.workflow_revision != 2 or not turn.reply
                or turn.reply.get("kind") != "caption_lines"):
            raise HTTPException(409, "This turn has no caption proposal")
        reply = CaptionLinesReply.model_validate(turn.reply)
        key = "caption-apply:" + turn.id
        done = db.scalar(select(AssistantMessage).where(AssistantMessage.conversation_id == turn.conversation_id,
            AssistantMessage.idempotency_key == key))
        if done is not None:
            caption = db.get(RunCaption, reply.caption_id)
            return {"applied": True, "repeated": True, "caption_id": reply.caption_id,
                    "caption_status": getattr(caption.status, "value", caption.status) if caption else None}
        final = db.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == turn.id)
            .order_by(AssistantAgentStep.step_index.desc())).first()
        evidence = script_evidence(restore_step(db, final), reply.lines_sha256) if final is not None else None
        target = next((c for c in (evidence.captions if evidence else ()) if c.caption_id == reply.caption_id
                       and c.apply_allowed), None)
        if evidence is None or target is None:
            raise HTTPException(409, "The caption proposal's evidence is unavailable")
        if evidence.lineage != "run_segments" and not body.attest_library_donor:
            raise HTTPException(409, "Confirm the library clip's source segment before applying")
        context = _call(service.effective_context, turn)
        _call(service.reauthorize_context, db, context)
        if context.run_id != run_id:
            raise HTTPException(404, "Assistant resource unavailable")
        # The same admission/lock as the manual Retry (run row first), then the caption.
        _admit_retained_run_source(db, [run_id])
        run = db.get(Run, run_id, populate_existing=True)
        caption = db.scalar(select(RunCaption).where(RunCaption.id == reply.caption_id)
            .with_for_update().execution_options(populate_existing=True))
        status = getattr(caption.status, "value", caption.status) if caption is not None else None
        if (caption is None or caption.run_id != run.id or caption.source_type != "run"
                or status not in ("done", "failed") or caption.language != target.language
                or (digest(caption.lines) if caption.lines else None) != target.lines_sha256):
            raise HTTPException(409, "The caption changed since the proposal; ask the assistant again")
        try:
            current, _ = hash_file(run.result_local_path or "", data_root=service.storage.data_root, check=lambda: None)
        except ScriptCaptionsRefused:
            current = None
        if current != evidence.final_sha256:
            raise HTTPException(409, "The run result changed since the proposal; ask the assistant again")
        lines = [line.model_dump(mode="json") for line in evidence.lines]
        caption.lines = lines
        caption.lines_source = "manual"
        caption.lines_reused_from = None
        transition(caption, CaptionStatus.queued)
        caption.error_message = None
        # Manual lines skip transcription; this is the same reservation the
        # manual Retry makes (released unused), not a paid call.
        _spend_reserve(spend_mod.reserve, db, spend_mod.caption_scope(caption.id), spend_mod.llm_credits(),
            kind="assistant captions apply", release_unused=True, run_id=run.id)
        append_message(db, turn.conversation_id, idempotency_key=key, actor="operator", role="system",
            content=f"Applied {len(lines)} script caption lines to caption {caption.id[:8]}; re-burn queued.",
            base_version_id=turn.base_version_id,
            context_snapshot={"turn_id": turn.id, "caption_id": caption.id, "lines_sha256": reply.lines_sha256,
                              "final_sha256": evidence.final_sha256, "lineage": evidence.lineage,
                              "attested_library_donor": bool(body.attest_library_donor)})
    db.commit()
    try:
        enqueue_process_caption(reply.caption_id)
    except Exception:
        raise HTTPException(503, "The captions were saved and queued, but the burn could not be started. "
                                 "It will be marked failed shortly; press Retry on the caption then.") from None
    return {"applied": True, "repeated": False, "caption_id": reply.caption_id, "caption_status": "queued"}


@router.post("/runs/{run_id}/turns/{turn_id}/retry-edit", status_code=202, dependencies=[GATED])
def retry_edit(run_id: str, turn_id: str, body: RetryTurn, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    turn = _call(service.require_turn, db, run_id, turn_id)
    with _atomic(db):
        db.execute(update(AssistantTurn).where(AssistantTurn.id == turn.id).values(updated_at=_now()).execution_options(synchronize_session=False))
        turn = _call(service.require_turn, db, run_id, turn_id)
        context = _call(service.effective_context, turn)
        _call(service.reauthorize_context, db, context)
        key = "retry:" + hashlib.sha256(body.idempotency_key.encode()).hexdigest()
        existing = db.scalar(select(AssistantTurnEdit).where(AssistantTurnEdit.turn_id == turn.id,
            AssistantTurnEdit.idempotency_key == key))
        if existing is not None:
            return {"turn": _turn(turn, db, service), "job": _job(db.get(AssistantJob, existing.job_id))}
        latest = db.get(AssistantJob, turn.edit_job_id, populate_existing=True) if turn.edit_job_id else None
        if turn.status != "succeeded" or not turn.reply or turn.reply.get("kind") != "propose" or latest is None or latest.status not in ("failed", "cancelled"):
            raise HTTPException(409, "Only failed or cancelled edits can be retried")
        from .assistant_jobs import AssistantPaidCallIntent
        uncertain = db.scalar(select(AssistantPaidCallIntent.id).where(AssistantPaidCallIntent.job_id == latest.id,
            AssistantPaidCallIntent.status.in_(("submitting", "unknown", "submitted"))).limit(1))
        if uncertain is not None:
            raise HTTPException(409, "Reconcile the previous provider request before retrying")
        # Retry exactly what was accepted, including a single selected change.
        # The original proposal can contain additional unapproved operations.
        selected_plan = EditPlan.model_validate(latest.accepted_plan)
        request_key = "turn-retry:" + hashlib.sha256((turn.id + "\0" + body.idempotency_key).encode()).hexdigest()
        admission = AssistantAdmissionService(storage=service.storage, policy=service.policy, enabled=True)
        receipt = _call(admission.admit, db, run_id=run_id, plan=selected_plan, instruction=context.dialogue[-1].text,
            idempotency_key=request_key, actor=turn.actor, now=_now(), explicit_base_version_id=turn.base_version_id,
            verified_candidate_turn_id=turn.id)
        db.add(AssistantTurnEdit(id=str(uuid.uuid4()), turn_id=turn.id, job_id=receipt.job_id,
            idempotency_key=key, created_at=_now()))
        turn.edit_job_id = receipt.job_id
        db.flush()
        return {"turn": _turn(turn, db, service), "job": _job(db.get(AssistantJob, receipt.job_id))}


def _job(job):
    return {"id": job.id, "version_id": job.pending_version_id, "base_version_id": job.pinned_base_version_id,
        "status": job.status, "phase": job.phase, "error_code": job.error_code}


@router.get("/runs/{run_id}/jobs/{job_id}")
def get_job(run_id: str, job_id: str, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    conversation = _call(service.conversation, db, run_id)
    job = db.scalar(select(AssistantJob).join(AssistantTurnEdit, AssistantTurnEdit.job_id == AssistantJob.id)
        .join(AssistantTurn, AssistantTurn.id == AssistantTurnEdit.turn_id).where(
        AssistantTurn.conversation_id == conversation.id, AssistantJob.id == job_id))
    if job is None:
        raise HTTPException(404, "Assistant resource unavailable")
    return _job(job)


class PrepareEditable(ContractModel):
    expected_selection_epoch: Annotated[StrictInt, Field(ge=0)]
    idempotency_key: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]


@router.post("/runs/{run_id}/versions/{version_id}/prepare-editable", dependencies=[GATED])
def prepare_editable(run_id: str, version_id: str, body: PrepareEditable, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    from sqlalchemy.exc import DBAPIError
    if not _PREPARATION_SLOT.acquire(blocking=False):
        raise HTTPException(503, "Another editable-copy preparation is running; retry the same saved request")
    try:
        version = _call(prepare_editable_version, db, service, run_id, version_id,
            expected_selection_epoch=body.expected_selection_epoch, idempotency_key=body.idempotency_key)
        return _version(version, canonical_uuid(run_id), db)
    except DBAPIError:
        db.rollback()
        raise HTTPException(503, "Preparation could not be confirmed; retry the same saved preparation request") from None
    finally:
        _PREPARATION_SLOT.release()


@router.post("/runs/{run_id}/select")
def choose_version(run_id: str, body: SelectVersion, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    conversation = _call(service.conversation, db, run_id)
    _call(service.catalog(db).require_target, body.version_id)
    selection = _call(select_version, db, conversation.id, body.version_id, expected_epoch=body.expected_epoch)
    return {"selected_version_id": selection.selected_version_id, "selection_epoch": selection.epoch}


@router.post("/runs/{run_id}/result")
def publish_run_result(run_id: str, body: PublishRunResult, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    from .gdrive_client import GDriveError
    from sqlalchemy.exc import DBAPIError
    try:
        receipt, published = _call(promote_run_result_published, db, service, run_id, body.model_dump(mode="json"))
    except (GDriveError, PublicationUnavailable):
        db.rollback()
        raise HTTPException(503, "Publication has not been confirmed. Retry the same saved request to reconcile it; other result changes remain blocked.") from None
    except DBAPIError:
        db.rollback()
        raise HTTPException(503, "Publication could not be confirmed. Retry the same saved request; do not start a different publication.") from None
    if published is not None:
        # Knowledge publication link (AICP-299 §4), after the promotion's commit;
        # syscalls + at most a bounded failure-row write; never raises.
        from .knowledge import capture

        capture.record_link_failure(capture.link_publication(published))
    return receipt


@router.get("/runs/{run_id}/result-history/{receipt_id}/{which}")
def deliver_result_history(run_id: str, receipt_id: str, which: Literal["before", "after"], db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    path = _call(history_media, db, service, run_id, receipt_id, which)
    return FileResponse(path, media_type="video/mp4", filename=f"run-{run_id[:8]}-main-{which}.mp4",
        headers={"Cache-Control": "private, no-store"})


class ImportSources(ContractModel):
    project_id: EntityId | None = None
    after_run_id: EntityId | None = None
    limit: Annotated[StrictInt, Field(ge=1, le=10)] = 5


@router.post("/runs/{run_id}/sources/import")
def import_sources(run_id: str, body: ImportSources, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    _call(service.require_run, db, run_id)
    from .models import Run, VideoProject
    from .assistant_models import AssistantConversation
    statement = select(Run).join(VideoProject, VideoProject.id == Run.project_id).outerjoin(AssistantConversation,
        AssistantConversation.run_id == Run.id).where(service.policy.sql_filter(Run.project_id), Run.status == "done",
        Run.result_local_path.is_not(None), AssistantConversation.id.is_(None))
    if body.project_id is not None:
        project_id = _call(service.policy.require_project, body.project_id)
        statement = statement.where(Run.project_id == project_id)
    if body.after_run_id is not None:
        statement = statement.where(Run.id > canonical_uuid(body.after_run_id))
    candidates = db.scalars(statement.order_by(Run.id).limit(body.limit + 1)).all()
    imported, unavailable, scanned, total_bytes = [], [], [], 0
    started = time.monotonic()
    for candidate in candidates[:body.limit]:
        if time.monotonic() - started >= 60:
            break
        scanned.append(candidate.id)
        try:
            # Resolve only an already scoped backend-owned Run path. The store
            # repeats confinement/integrity checks before copying any bytes.
            path = service.storage._source(candidate.result_local_path)
            size = path.stat().st_size
            if size > 128 * 1024 * 1024 or total_bytes + size > 512 * 1024 * 1024:
                unavailable.append({"run_id": candidate.id, "status": "batch_size_limit"})
                continue
            total_bytes += size
            with _atomic(db):
                version = import_legacy_run(db, candidate, actor="operator", storage=service.storage, policy=service.policy)
            imported.append({"run_id": candidate.id, "version_id": version.id})
        except (ValueError, RuntimeError, OSError):
            unavailable.append({"run_id": candidate.id, "status": "unavailable"})
    more = len(candidates) > len(scanned)
    return {"imported": imported, "unavailable": unavailable,
        "next_after_run_id": scanned[-1] if more and scanned else None, "budget_exhausted": len(scanned) < min(body.limit, len(candidates))}


@router.get("/runs/{run_id}/audio-candidates")
def audio_candidates(run_id: str, base_version_id: str, phrase: str = Query(min_length=1, max_length=200),
        limit: int = Query(10, ge=1, le=25), db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    _call(service.conversation, db, run_id)
    catalog = service.catalog(db)
    target = _call(catalog.require_target, base_version_id)
    if target.run_id != canonical_uuid(run_id):
        raise HTTPException(404, "Assistant resource unavailable")
    from .assistant_audio import find_audio_donor_candidates
    candidates = _call(find_audio_donor_candidates, db, catalog=catalog, storage=service.storage,
        target_version_id=base_version_id, phrase=phrase, limit=limit, scan_limit=100, timeout_sec=20)
    return {"candidates": [asdict(candidate) for candidate in candidates]}


@router.get("/runs/{run_id}/catalog")
def list_sources(run_id: str, base_version_id: str, project_id: str | None = None, after_asset_id: str | None = None,
                 limit: int = Query(50, ge=1, le=100), db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    _call(service.conversation, db, run_id)
    catalog = service.catalog(db)
    target = _call(catalog.require_target, base_version_id)
    if target.run_id != canonical_uuid(run_id):
        raise HTTPException(404, "Assistant resource unavailable")
    page = _call(catalog.list_editable_artifacts, target_version_id=base_version_id, project_id=project_id, after_asset_id=after_asset_id, limit=limit)
    full_ids = set(db.scalars(select(AssistantVersionArtifact.artifact_id).where(
        AssistantVersionArtifact.role == "full", AssistantVersionArtifact.artifact_id.in_([item.asset_id for item in page.artifacts]))).all())
    versions = {version.id: version.sequence for version in db.scalars(select(AssistantVersion).where(
        AssistantVersion.id.in_([item.version_id for item in page.artifacts]))).all()}
    return {"artifacts": [{**asdict(artifact), "label": f"Run {artifact.run_id[:8]} V{versions.get(artifact.version_id, '?')}",
        "media_url": f"/api/v2/assistant/runs/{canonical_uuid(run_id)}/assets/{artifact.asset_id}/media?base_version_id={canonical_uuid(base_version_id)}&version_id={artifact.version_id}"}
        for artifact in page.artifacts if artifact.asset_id in full_ids and artifact.media_kind in ("video", "audio")],
        "next_after_asset_id": page.next_after_asset_id}


@router.get("/runs/{run_id}/turns/{turn_id}/candidates/{candidate_id}/media")
def deliver_candidate(run_id: str, turn_id: str, candidate_id: str,
                      db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    from .assistant_agent_steps import AssistantAgentStep
    from .assistant_agent_contracts import plan_fingerprint
    from .assistant_store import PinnedFile
    turn = _call(service.require_turn, db, run_id, turn_id)
    candidate_id = _call(canonical_uuid, candidate_id)
    # This checks every source observed during analysis, including donors not
    # used in the final edit, before any retained candidate bytes are served.
    state = _call(service.agent_public_state, db, turn)
    if (turn.status != "succeeded" or not state or not state.get("verification")
            or state["verification"].get("candidate_id") != candidate_id):
        raise HTTPException(404, "Assistant candidate unavailable")
    rows = db.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == turn.id,
        AssistantAgentStep.status == "completed")).all()
    item = next((row.private_artifact for row in rows if row.private_artifact
                 and row.private_artifact.get("candidate_id") == candidate_id), None)
    if item is None:
        raise HTTPException(404, "Assistant candidate unavailable")
    candidate = next((row.observation.get("candidate") for row in rows
        if row.observation and row.observation.get("candidate")
        and row.observation["candidate"].get("candidate_id") == candidate_id), None)
    if (candidate is None or not turn.reply or turn.reply.get("kind") != "propose"
            or any(item.get(key) != candidate.get(key) for key in ("sha256", "size_bytes", "plan_sha256"))
            or candidate.get("plan") != turn.reply.get("plan")
            or item["plan_sha256"] != plan_fingerprint(EditPlan.model_validate(turn.reply["plan"]))):
        raise HTTPException(404, "Assistant candidate unavailable")
    path = _call(service.storage.verify, PinnedFile(item["storage_key"], item["sha256"], item["size_bytes"]))
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


@router.get("/runs/{run_id}/versions/{version_id}/media/{role}")
def deliver_media(run_id: str, version_id: str, role: Literal["preview", "full"], db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    _call(service.conversation, db, run_id)
    catalog = service.catalog(db)
    target = _call(catalog.require_target, version_id)
    if target.run_id != canonical_uuid(run_id):
        raise HTTPException(404, "Assistant resource unavailable")
    artifact = _call(get_version_artifact, db, version_id, role=role)
    _call(catalog.resolve_reference, target_version_id=version_id, asset_id=artifact.id, version_id=artifact.owner_version_id)
    path = _call(verify_artifact, artifact, storage=service.storage)
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@router.get("/runs/{run_id}/assets/{asset_id}/media")
def deliver_source(run_id: str, asset_id: str, version_id: str, base_version_id: str,
        db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    _call(service.conversation, db, run_id)
    catalog = service.catalog(db)
    target = _call(catalog.require_target, base_version_id)
    if target.run_id != canonical_uuid(run_id):
        raise HTTPException(404, "Assistant resource unavailable")
    descriptor = _call(catalog.resolve_reference, target_version_id=base_version_id, asset_id=asset_id, version_id=version_id)
    artifact = db.get(AssistantArtifact, descriptor.asset_id, populate_existing=True)
    path = _call(verify_artifact, artifact, storage=service.storage)
    if descriptor.media_kind == "audio":
        with path.open("rb") as source:
            magic = source.read(4)
        media_type = "audio/wav" if magic == b"RIFF" else "audio/flac" if magic == b"fLaC" else "audio/mp4"
    elif descriptor.media_kind == "video":
        media_type = "video/mp4"
    else:
        raise HTTPException(404, "Assistant resource unavailable")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@router.post("/runs/{run_id}/captions/{caption_id}/import")
def import_caption(run_id: str, caption_id: str, db: Session = Depends(get_db), service: AssistantTurnService = Depends(get_assistant_service)):
    run = _call(service.require_run, db, run_id)
    caption_id = _call(canonical_uuid, caption_id)
    from .assistant_legacy import import_legacy_caption
    try:
        with _atomic(db):
            version = import_legacy_caption(db, run, caption_id, actor="operator", storage=service.storage, policy=service.policy)
    except AssistantAccessDenied:
        raise HTTPException(404, "Assistant resource unavailable") from None
    except (ValueError, RuntimeError, OSError):
        raise HTTPException(422, "Matching clean source and saved rendered subtitles could not be verified; original remains unchanged") from None
    return _version(version, run.id, db)
