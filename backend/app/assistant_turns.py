"""Durable, separately fenced planning turns; no inference or media execution.

A turn pins an authorized PlannerContext before any model work. Lease loss is
terminal and never starts a second inference implicitly. Callers own commits.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import uuid

from pydantic import TypeAdapter
from sqlalchemy import JSON, Boolean, CheckConstraint, DDL, DateTime, ForeignKey, ForeignKeyConstraint, Index, Integer, String, UniqueConstraint, and_, event, false, func, inspect, or_, select, update
from sqlalchemy.orm import Mapped, Session, mapped_column

from .assistant_access import AssistantAccessDenied, AuthorizedArtifactCatalog, ProjectAllowlist, canonical_uuid
from .assistant_jobs import LostLease, _OperationClock, _atomic, _expires, _identifier, _now
from .assistant_models import AssistantConversation, AssistantMessage, AssistantVersion
from .assistant_planner import AgentReply, PlannerContext, PlannerDialogue, PlannerReply, PlannerSettings, PlannerSource, PlannerSubtitleCue, ProposeReply, UnavailableReply
from .assistant_contracts import AssetManifest, VersionManifest, ExtendEndingOperation, validate_plan_sources, validate_local_plan_capability
from .assistant_store import AssistantArtifactStorage, append_message, get_version_artifact, verify_artifact
from . import assistant_media
from .db import Base
from .assistant_agent_steps import AssistantAgentStep, restore_step


_MAX_HISTORY_DEPENDENCY_TURNS = 256
_MAX_HISTORY_DEPENDENCY_SOURCES = 256
_MAX_HISTORY_DEPENDENCY_INTERNAL_READS = 256
MAX_INITIAL_AGENT_CONTEXT_BYTES = 32 * 1024


class TurnConflict(ValueError):
    pass


class TurnContextTooLarge(TurnConflict):
    """Essential fresh agent context cannot leave room for tool observations."""


class TurnsDisabled(TurnConflict):
    pass


class AssistantTurn(Base):
    __tablename__ = "assistant_turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "idempotency_key", name="uq_assistant_turn_request"),
        UniqueConstraint("operator_message_id", name="uq_assistant_turn_operator_message"),
        ForeignKeyConstraint(["conversation_id", "base_version_id"],
            ["assistant_versions.conversation_id", "assistant_versions.id"], ondelete="RESTRICT", name="fk_assistant_turn_base_scope"),
        CheckConstraint("status IN ('queued','running','cancel_requested','succeeded','failed','cancelled')", name="ck_assistant_turn_status"),
        CheckConstraint("lease_generation >= 0", name="ck_assistant_turn_generation"),
        CheckConstraint("((status IN ('running','cancel_requested')) AND lease_owner IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL) OR ((status NOT IN ('running','cancel_requested')) AND lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)", name="ck_assistant_turn_lease"),
        Index("ix_assistant_turns_queue", "status", "created_at"),
        Index("ix_assistant_turns_recovery", "status", "lease_expires_at"),
        Index("ix_assistant_turns_broker_cleanup", "broker_cleanup_pending", "broker_cleanup_next_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False)
    base_version_id: Mapped[str] = mapped_column(String(36), nullable=False)
    operator_message_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_messages.id", ondelete="RESTRICT"), nullable=False)
    assistant_message_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("assistant_messages.id", ondelete="RESTRICT"))
    parent_turn_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("assistant_turns.id", ondelete="RESTRICT"))
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    context_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    reply: Mapped[dict | None] = mapped_column(JSON)
    enrichment_snapshot: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    workflow_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    agent_context_snapshot: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    broker_context_sha256: Mapped[str | None] = mapped_column(String(64))
    broker_cleanup_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=false())
    broker_cleanup_next_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    edit_job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("assistant_jobs.id", ondelete="RESTRICT"))
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token: Mapped[str | None] = mapped_column(String(36))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AssistantTurnEdit(Base):
    """Every explicitly accepted render attempt; retries never need inference."""
    __tablename__ = "assistant_turn_edits"
    __table_args__ = (
        UniqueConstraint("turn_id", "idempotency_key", name="uq_assistant_turn_edit_request"),
        UniqueConstraint("job_id", name="uq_assistant_turn_edit_job"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    turn_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_turns.id", ondelete="RESTRICT"), nullable=False)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_jobs.id", ondelete="RESTRICT"), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    execution_message_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("assistant_messages.id", ondelete="RESTRICT"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


_TURN_PINNED = {"id", "conversation_id", "base_version_id", "operator_message_id", "parent_turn_id",
    "idempotency_key", "request_sha256", "actor", "context_snapshot", "created_at", "workflow_revision"}


@event.listens_for(AssistantTurn, "before_update")
def _preserve_turn_request(mapper, connection, target):
    for name in _TURN_PINNED:
        if inspect(target).attrs[name].history.has_changes():
            raise TurnConflict("accepted planning request is immutable")
    fingerprint = inspect(target).attrs.broker_context_sha256.history
    if fingerprint.has_changes() and fingerprint.deleted and fingerprint.deleted[0] is not None:
        raise TurnConflict("broker context fingerprint is write-once")
    agent_context = inspect(target).attrs.agent_context_snapshot.history
    if agent_context.has_changes() and agent_context.deleted and agent_context.deleted[0] is not None:
        raise TurnConflict("agent final context is write-once")
    if agent_context.has_changes() and target.agent_context_snapshot is not None and (target.workflow_revision != 2 or target.status != "running"):
        raise TurnConflict("agent context requires a running revision-two turn")
    enrichment = inspect(target).attrs.enrichment_snapshot.history
    if enrichment.has_changes() and enrichment.deleted and enrichment.deleted[0] is not None:
        raise TurnConflict("planning enrichment is write-once")
    if enrichment.has_changes() and target.enrichment_snapshot is not None and target.status != "running":
        raise TurnConflict("planning enrichment requires a running turn")
    history = inspect(target).attrs.status.history
    if history.deleted and history.deleted[0] in ("succeeded", "failed", "cancelled") and history.has_changes():
        raise TurnConflict("terminal planning turn cannot change status")


@event.listens_for(AssistantTurnEdit, "before_update")
def _preserve_turn_edit(mapper, connection, target):
    for attribute in inspect(target).attrs:
        if attribute.key != "execution_message_id" and attribute.history.has_changes():
            raise TurnConflict("accepted edit association is immutable")
    history = inspect(target).attrs.execution_message_id.history
    if history.deleted and history.deleted[0] is not None and history.has_changes():
        raise TurnConflict("execution message binding is immutable")


@dataclass(frozen=True)
class TurnLease:
    turn_id: str
    owner: str
    token: str
    generation: int
    expires_at: datetime


def _get(session: Session, turn_id: str) -> AssistantTurn:
    turn = session.get(AssistantTurn, canonical_uuid(turn_id), populate_existing=True)
    if turn is None:
        raise TurnConflict("planning turn does not exist")
    return turn


def _owned(lease: TurnLease, now: datetime):
    return and_(AssistantTurn.id == lease.turn_id, AssistantTurn.status == "running",
        AssistantTurn.cancel_requested_at.is_(None), AssistantTurn.lease_owner == lease.owner,
        AssistantTurn.lease_token == lease.token, AssistantTurn.lease_generation == lease.generation,
        AssistantTurn.lease_expires_at > now)


def _lock_owned(session: Session, lease: TurnLease, clock: _OperationClock) -> AssistantTurn:
    result = session.execute(update(AssistantTurn).where(_owned(lease, clock.now())).values(updated_at=clock.now()).execution_options(synchronize_session=False))
    if result.rowcount != 1:
        raise LostLease("planning turn lease was lost or cancelled")
    turn = _get(session, lease.turn_id)
    turn.updated_at = clock.require_live(turn.lease_expires_at)
    return turn


def _terminal(turn: AssistantTurn, status: str, now: datetime, error_code: str | None = None):
    turn.status, turn.updated_at, turn.finished_at, turn.error_code = status, now, now, error_code
    turn.lease_owner = turn.lease_token = turn.lease_expires_at = None


def claim_turn(session: Session, turn_id: str, *, owner: str, now: datetime, lease_sec: int = 60) -> TurnLease | None:
    clock, owner = _OperationClock.start(now), _identifier(owner, "owner")
    _expires(clock.now(), lease_sec)
    with _atomic(session):
        result = session.execute(update(AssistantTurn).where(AssistantTurn.id == canonical_uuid(turn_id),
            AssistantTurn.status == "queued", AssistantTurn.cancel_requested_at.is_(None)).values(updated_at=clock.now()).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            return None
        turn = _get(session, turn_id)
        turn.status, turn.lease_owner, turn.lease_token = "running", owner, str(uuid.uuid4())
        turn.lease_generation += 1
        turn.lease_expires_at = _expires(clock.now(), lease_sec)
        session.flush()
        clock.require_live(turn.lease_expires_at)
        return TurnLease(turn.id, owner, turn.lease_token, turn.lease_generation, turn.lease_expires_at)


def renew_turn(session: Session, lease: TurnLease, *, now: datetime, lease_sec: int = 60) -> TurnLease:
    clock = _OperationClock.start(now)
    _expires(clock.now(), lease_sec)
    with _atomic(session):
        turn = _lock_owned(session, lease, clock)
        turn.lease_expires_at = _expires(clock.now(), lease_sec)
        session.flush()
        return TurnLease(turn.id, lease.owner, lease.token, lease.generation, turn.lease_expires_at)


def fail_turn(session: Session, lease: TurnLease, *, error_code: str, now: datetime) -> AssistantTurn:
    clock = _OperationClock.start(now)
    with _atomic(session):
        turn = _lock_owned(session, lease, clock)
        _terminal(turn, "failed", clock.now(), _identifier(error_code, "error_code"))
        session.flush()
        return turn


def request_cancel(session: Session, turn_id: str, *, now: datetime) -> AssistantTurn:
    clock = _OperationClock.start(now)
    with _atomic(session):
        session.execute(update(AssistantTurn).where(AssistantTurn.id == canonical_uuid(turn_id)).values(updated_at=clock.now()).execution_options(synchronize_session=False))
        turn = _get(session, turn_id)
        if turn.status == "queued":
            turn.cancel_requested_at = clock.now()
            _terminal(turn, "cancelled", clock.now())
        elif turn.status == "running":
            turn.cancel_requested_at, turn.status = clock.now(), "cancel_requested"
        session.flush()
        return turn


def finalize_cancel(session: Session, turn_id: str, *, now: datetime, lease: TurnLease | None = None) -> AssistantTurn:
    """Caller with a live lease attests its process has exited; otherwise expiry required."""
    clock = _OperationClock.start(now)
    with _atomic(session):
        session.execute(update(AssistantTurn).where(AssistantTurn.id == canonical_uuid(turn_id)).values(updated_at=clock.now()).execution_options(synchronize_session=False))
        turn = _get(session, turn_id)
        if turn.status == "cancelled":
            return turn
        owned = lease is not None and (turn.id, turn.lease_owner, turn.lease_token, turn.lease_generation) == (lease.turn_id, lease.owner, lease.token, lease.generation)
        expires = turn.lease_expires_at
        if expires is not None and expires.tzinfo is None:
            from datetime import timezone
            expires = expires.replace(tzinfo=timezone.utc)
        if turn.status != "cancel_requested" or (not owned and expires is not None and expires > clock.now()):
            raise LostLease("cancellation requires process acknowledgement or expired lease")
        _terminal(turn, "cancelled", clock.now())
        session.flush()
        return turn


def recover_expired(session: Session, *, now: datetime, limit: int = 100) -> tuple[str, ...]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise TurnConflict("recovery limit must be within 1..100")
    clock = _OperationClock.start(now)
    rows = session.scalars(select(AssistantTurn.id).where(AssistantTurn.status.in_(("running", "cancel_requested")),
        AssistantTurn.lease_expires_at <= clock.now()).order_by(AssistantTurn.created_at, AssistantTurn.id).limit(limit)).all()
    recovered = []
    for turn_id in rows:
        with _atomic(session):
            result = session.execute(update(AssistantTurn).where(AssistantTurn.id == turn_id,
                AssistantTurn.status.in_(("running", "cancel_requested")), AssistantTurn.lease_expires_at <= clock.now()).values(updated_at=clock.now()).execution_options(synchronize_session=False))
            if result.rowcount == 1:
                turn = _get(session, turn_id)
                _terminal(turn, "cancelled" if turn.status == "cancel_requested" else "failed", clock.now(), "interrupted")
                session.flush()
                recovered.append(turn_id)
    return tuple(recovered)


def mark_broker_submission(session: Session, lease: TurnLease, *, context_sha256: str, now: datetime) -> None:
    """Durable cleanup intent BEFORE a possibly accepted native submission."""
    import re
    if not isinstance(context_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", context_sha256) is None:
        raise TurnConflict("invalid broker context fingerprint")
    clock = _OperationClock.start(now)
    with _atomic(session):
        turn = _lock_owned(session, lease, clock)
        if turn.broker_context_sha256 not in (None, context_sha256):
            raise TurnConflict("broker context fingerprint changed")
        turn.broker_context_sha256 = context_sha256
        turn.broker_cleanup_pending, turn.broker_cleanup_next_at = True, None
        session.flush()


class AssistantTurnService:
    def __init__(self, *, storage: AssistantArtifactStorage, policy: ProjectAllowlist, enabled: bool = False, max_pending: int = 8, agent_enabled: bool = False):
        if type(max_pending) is not int or not 1 <= max_pending <= 100:
            raise TurnConflict("pending turn limit must be within 1..100")
        self.storage, self.policy, self.enabled, self.max_pending = storage, policy, enabled is True, max_pending
        self.agent_enabled = agent_enabled is True

    def gate(self):
        if not self.enabled:
            raise TurnsDisabled("assistant is disabled")

    def require_run(self, session: Session, run_id: str):
        self.gate()
        from .models import Run, VideoProject
        run = session.scalar(select(Run).join(VideoProject, VideoProject.id == Run.project_id).where(
            Run.id == canonical_uuid(run_id), self.policy.sql_filter(Run.project_id)).execution_options(populate_existing=True, autoflush=False))
        if run is None:
            raise AssistantAccessDenied("assistant access denied")
        self.policy.require_project(run.project_id)
        return run

    def conversation(self, session: Session, run_id: str) -> AssistantConversation:
        run = self.require_run(session, run_id)
        conversation = session.scalar(select(AssistantConversation).where(AssistantConversation.run_id == run.id,
            AssistantConversation.project_id == run.project_id).execution_options(populate_existing=True))
        if conversation is None:
            raise TurnConflict("import the successful Run before starting chat")
        return conversation

    def catalog(self, session: Session, *, verification_cache: dict | None = None):
        def available(artifact):
            options = {"verification_cache": verification_cache} if verification_cache is not None else {}
            verify_artifact(artifact, storage=self.storage, **options)
            return True
        return AuthorizedArtifactCatalog(session, policy=self.policy, is_available=available)

    def require_turn(self, session: Session, run_id: str, turn_id: str) -> AssistantTurn:
        conversation = self.conversation(session, run_id)
        turn = session.scalar(select(AssistantTurn).where(AssistantTurn.id == canonical_uuid(turn_id),
            AssistantTurn.conversation_id == conversation.id).execution_options(populate_existing=True))
        if turn is None:
            raise AssistantAccessDenied("assistant access denied")
        return turn

    def _source(self, session, descriptor, *, label):
        artifact = get_version_artifact(session, descriptor.version_id) if descriptor.media_kind == "video" else None
        if artifact is None or artifact.id != descriptor.asset_id:
            from .assistant_models import AssistantArtifact
            artifact = session.get(AssistantArtifact, descriptor.asset_id, populate_existing=True)
        path = verify_artifact(artifact, storage=self.storage)
        if descriptor.media_kind == "audio":
            from .assistant_audio import probe_trusted_audio
            media = probe_trusted_audio(path)
            duration, has_audio, fps = media.duration_sec, True, None
        else:
            media = assistant_media.probe_trusted_media(path)
            duration, has_audio, fps = media.video_duration_sec, media.has_audio, media.fps
        verify_artifact(artifact, storage=self.storage)
        return PlannerSource(asset_id=descriptor.asset_id, version_id=descriptor.version_id,
            run_id=descriptor.run_id, project_id=descriptor.project_id, media_kind=descriptor.media_kind,
            duration_sec=duration, has_audio=has_audio, label=label, fps=fps)

    def version_context(self, session: Session, *, run_id: str, base_version_id: str) -> dict:
        """Bounded safe fields only; original settings come from the pinned import."""
        conversation = self.conversation(session, run_id)
        catalog = self.catalog(session)
        target = catalog.require_editable_target(base_version_id)
        if target.run_id != canonical_uuid(run_id):
            raise TurnConflict("base version belongs to another Run")
        artifact = get_version_artifact(session, target.version_id)
        selected = self._source(session, catalog.resolve_editable_reference(target_version_id=target.version_id,
            asset_id=artifact.id, version_id=target.version_id), label="Selected result")
        root = session.scalar(select(AssistantVersion).where(AssistantVersion.conversation_id == conversation.id,
            AssistantVersion.sequence == 1))
        original = root.recipe.get("settings", {}) if root and isinstance(root.recipe, dict) else {}
        known = {}
        allowed = {
            "language": {"EN", "ES", "PT", "JA", "DE", "IT", "PL", "FR"},
            "model": {"seedance", "gemini-omni", "seedance-fast", "seedance-mini", "seedance-2-5"},
            "audio_mode": {"original", "seedance"}, "resolution": {"480p", "720p", "1080p", "4k"},
        }
        for field, options in allowed.items():
            value = original.get(field)
            if field == "language" and isinstance(value, str):
                value = value.upper()
            if isinstance(value, str) and value in options:
                known[field] = value
        if type(original.get("mute_source")) is bool:
            known["mute_source"] = original["mute_source"]
        stitch_plan = original.get("stitch_plan")
        if isinstance(stitch_plan, list):
            if not stitch_plan:
                known["original_has_library_outro"] = False
            elif isinstance(stitch_plan[-1], dict):
                last = stitch_plan[-1]
                if last.get("kind") == "library":
                    try:
                        canonical_uuid(last.get("clip_id"))
                        known["original_has_library_outro"] = True
                    except ValueError:
                        pass
                elif last.get("kind") == "segment":
                    known["original_has_library_outro"] = False
        from .assistant_subtitles import get_subtitle_context
        track = get_subtitle_context(session, target.version_id, storage=self.storage)
        return {"selected": selected, "settings": PlannerSettings(**known),
            "editable_subtitle_cue_ids": tuple(cue.cue_id for cue in track.cues) if track else (),
            "subtitle_cues": tuple(PlannerSubtitleCue(cue_id=cue.cue_id, start_sec=cue.start_sec,
                end_sec=cue.end_sec, text=cue.text) for cue in track.cues) if track else (),
            "subtitle_style": track.style if track else None}

    def build_context(self, session: Session, *, run_id: str, base_version_id: str, instruction: str, source_refs: tuple[tuple[str, str], ...] = (), selected_passage=None,
                      reference_turn_id=None, reference_plan=None) -> PlannerContext:
        conversation = self.conversation(session, run_id)
        catalog = self.catalog(session)
        target = catalog.require_editable_target(base_version_id)
        if target.run_id != canonical_uuid(run_id):
            raise TurnConflict("base version belongs to another Run")
        version_context = self.version_context(session, run_id=run_id, base_version_id=base_version_id)
        if len(source_refs) > 32:
            raise TurnConflict("at most 32 source references are supported")
        sources = tuple(self._source(session, catalog.resolve_editable_reference(target_version_id=target.version_id,
            asset_id=asset, version_id=version), label=f"Source {index + 1}") for index, (asset, version) in enumerate(source_refs)
            if (asset, version) != (version_context["selected"].asset_id, version_context["selected"].version_id))
        ids = select(AssistantTurn.operator_message_id).where(AssistantTurn.conversation_id == conversation.id).union(
            select(AssistantTurn.assistant_message_id).where(AssistantTurn.conversation_id == conversation.id),
            select(AssistantTurnEdit.execution_message_id).join(AssistantTurn, AssistantTurn.id == AssistantTurnEdit.turn_id)
                .where(AssistantTurn.conversation_id == conversation.id))
        messages = session.scalars(select(AssistantMessage).where(AssistantMessage.id.in_(ids),
            AssistantMessage.role.in_(("operator", "assistant"))).order_by(AssistantMessage.sequence.desc()).limit(31)).all()
        candidate_ids = {message.id for message in messages}
        related = session.scalars(select(AssistantTurn).where(AssistantTurn.conversation_id == conversation.id,
            AssistantTurn.assistant_message_id.in_(candidate_ids))).all()
        message_owner = {turn.assistant_message_id: turn for turn in related}
        executions = session.execute(select(AssistantTurnEdit.execution_message_id, AssistantTurn).join(
            AssistantTurn, AssistantTurn.id == AssistantTurnEdit.turn_id).where(
                AssistantTurn.conversation_id == conversation.id, AssistantTurnEdit.execution_message_id.in_(candidate_ids))).all()
        message_owner.update({message_id: turn for message_id, turn in executions})
        visibility = {}
        authorization_cache = {}
        for turn in message_owner.values():
            if turn.id not in visibility:
                try:
                    self.reauthorize_context(session, self.effective_context(turn), verify_bytes=False,
                        authorization_cache=authorization_cache)
                    visibility[turn.id] = True
                except AssistantAccessDenied:
                    visibility[turn.id] = False
        messages = [message for message in messages if message.role == "operator" or (
            message.id in message_owner and visibility[message_owner[message.id].id])]
        dialogue = tuple(PlannerDialogue(role=message.role, text=message.content,
            source_message_id=message.id if message.role == "assistant" else None) for message in reversed(messages))
        from .assistant_broker_client import context_fingerprint
        dialogue = (*dialogue, PlannerDialogue(role="operator", text=instruction))
        while True:
            context = PlannerContext(run_id=target.run_id, base_version_id=target.version_id, **version_context,
                sources=sources, selected_passage=selected_passage, dialogue=dialogue,
                reference_turn_id=reference_turn_id, reference_plan=reference_plan)
            # Match the expanded UTF-8 JSON used by the agent metadata block.
            # Reserve observation headroom without changing canonical hashes or
            # rewriting durable history, legacy turns or pinned retry contexts.
            if self.agent_enabled and len(json.dumps(context.model_dump(mode="json"),
                    ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_INITIAL_AGENT_CONTEXT_BYTES:
                if len(dialogue) == 1:
                    raise TurnContextTooLarge("essential initial agent context exceeds size limit")
                dialogue = dialogue[1:]
                continue
            try:
                context_fingerprint(context)
            except ValueError:
                if len(dialogue) == 1:
                    raise TurnConflict("pinned source context and current instruction exceed the planning size limit") from None
                dialogue = dialogue[1:]  # Keep the current request and trusted pinned context.
                continue
            try:
                # Combining individually authorized replies can cross the graph
                # bound. Omit derived history rather than blocking future turns.
                self.reauthorize_context(session, context, verify_bytes=False)
                return context
            except AssistantAccessDenied:
                derived = next((index for index, item in enumerate(dialogue[:-1]) if item.role == "assistant"), None)
                if derived is None:
                    raise
                dialogue = (*dialogue[:derived], *dialogue[derived + 1:])


    def reauthorize_context(self, session: Session, context: PlannerContext, *, verify_bytes: bool = True,
                            authorization_cache: dict | None = None, verification_cache: dict | None = None) -> None:
        """Authorize direct sources and the immutable provenance of copied text.

        Assistant output may depend on every input in its owning turn, including
        older copied output. Traverse that graph rather than treating a later
        turn's direct source list as a complete authorization boundary. Legacy
        assistant dialogue without message provenance fails closed.
        """
        conversation = self.conversation(session, context.run_id)
        catalog = self.catalog(session, verification_cache=verification_cache) if verify_bytes else AuthorizedArtifactCatalog(
            session, policy=self.policy, is_available=lambda artifact: True)
        # Reuse only inside the caller's current request/transaction. No scope or
        # byte-verification decision survives a subsequent API/worker request.
        cache = {} if authorization_cache is None else authorization_cache
        state = cache.setdefault((conversation.id, verify_bytes), {
            "authorized": set(), "sources": set(), "targets": set(), "owners": {},
        })
        active, traversed = set(), set()
        pending = [(context, None, None, False)]
        while pending:
            current, owner_id, before_sequence, exiting = pending.pop()
            if exiting:
                active.remove(owner_id)
                state["authorized"].add(owner_id)
                continue
            if owner_id is not None:
                if owner_id in active:
                    raise AssistantAccessDenied("assistant history access denied")
                if owner_id in state["authorized"]:
                    continue
                traversed.add(owner_id)
                if len(traversed) > _MAX_HISTORY_DEPENDENCY_TURNS:
                    raise AssistantAccessDenied("assistant history access denied")
                active.add(owner_id)
                pending.append((None, owner_id, None, True))
            if current.run_id != context.run_id:
                raise AssistantAccessDenied("assistant history access denied")
            if current.reference_turn_id is not None:
                referenced = session.get(AssistantTurn, current.reference_turn_id, populate_existing=True)
                if (referenced is None or referenced.conversation_id != conversation.id
                        or referenced.base_version_id != current.base_version_id or referenced.status != "succeeded"
                        or not referenced.reply or referenced.reply.get("kind") != "propose"
                        or referenced.reply.get("plan") != current.reference_plan.model_dump(mode="json")):
                    raise AssistantAccessDenied("assistant proposal access denied")
                if referenced.id in active:
                    raise AssistantAccessDenied("assistant proposal access denied")
                if referenced.id not in state["authorized"]:
                    pending.append((self.effective_context(referenced), referenced.id, None, False))
            if current.base_version_id not in state["targets"]:
                target = catalog.require_editable_target(current.base_version_id)
                if (target.run_id, target.project_id) != (conversation.run_id, conversation.project_id):
                    raise AssistantAccessDenied("assistant history access denied")
                state["targets"].add(current.base_version_id)
            from .assistant_internal_api import reauthorize_dependencies
            dependencies = state.setdefault('internal_reads', set())
            pending_dependencies = tuple(dependency for dependency in current.internal_read_dependencies if dependency not in dependencies)
            if len(dependencies) + len(pending_dependencies) > _MAX_HISTORY_DEPENDENCY_INTERNAL_READS:
                raise AssistantAccessDenied('assistant history access denied')
            if pending_dependencies:
                reauthorize_dependencies(session, service=self, dependencies=pending_dependencies,
                    target_version_id=current.base_version_id, verify_bytes=False)
                dependencies.update(pending_dependencies)
            for source in (current.selected, *current.sources):
                binding = (source.asset_id, source.version_id, source.project_id, source.run_id, source.media_kind)
                if binding in state["sources"]:
                    continue
                if len(state["sources"]) >= _MAX_HISTORY_DEPENDENCY_SOURCES:
                    raise AssistantAccessDenied("assistant history access denied")
                descriptor = catalog.resolve_editable_reference(target_version_id=current.base_version_id,
                    asset_id=source.asset_id, version_id=source.version_id)
                if (descriptor.project_id, descriptor.run_id, descriptor.media_kind) != (source.project_id, source.run_id, source.media_kind):
                    raise AssistantAccessDenied("assistant access denied")
                state["sources"].add(binding)
            for entry in current.dialogue:
                if entry.role == "operator":
                    if entry.source_message_id is not None:
                        raise AssistantAccessDenied("assistant history access denied")
                    continue
                message_id = entry.source_message_id
                if message_id is None:
                    raise AssistantAccessDenied("assistant history access denied")
                if message_id not in state["owners"]:
                    message = session.get(AssistantMessage, message_id, populate_existing=True)
                    if (message is None or message.conversation_id != conversation.id or message.role != "assistant"):
                        raise AssistantAccessDenied("assistant history access denied")
                    # Exact persisted pointers identify provenance; matching text
                    # or a user-controlled idempotency-key prefix never does.
                    owner_ids = session.scalars(select(AssistantTurn.id).outerjoin(AssistantTurnEdit,
                        AssistantTurnEdit.turn_id == AssistantTurn.id).where(
                            AssistantTurn.conversation_id == conversation.id,
                            or_(AssistantTurn.assistant_message_id == message.id,
                                AssistantTurnEdit.execution_message_id == message.id)).distinct().limit(2)).all()
                    if len(owner_ids) != 1:
                        raise AssistantAccessDenied("assistant history access denied")
                    owner = session.get(AssistantTurn, owner_ids[0], populate_existing=True)
                    operator = session.get(AssistantMessage, owner.operator_message_id, populate_existing=True)
                    if (operator is None or operator.conversation_id != conversation.id or operator.role != "operator"
                            or message.base_version_id != owner.base_version_id or operator.sequence >= message.sequence):
                        raise AssistantAccessDenied("assistant history access denied")
                    state["owners"][message_id] = (message, owner, operator.sequence)
                message, owner, operator_sequence = state["owners"][message_id]
                if message.content != entry.text or (before_sequence is not None and message.sequence >= before_sequence):
                    raise AssistantAccessDenied("assistant history access denied")
                if owner.id in active:
                    raise AssistantAccessDenied("assistant history access denied")
                if owner.id not in state["authorized"]:
                    pending.append((self.effective_context(owner), owner.id, operator_sequence, False))


    def accept(self, session: Session, *, run_id: str, base_version_id: str, instruction: str, idempotency_key: str,
               actor: str, now: datetime, source_refs: tuple[tuple[str, str], ...] = (), parent_turn_id: str | None = None,
               pinned_context: PlannerContext | None = None, selected_passage=None, proposal_turn_id: str | None = None) -> AssistantTurn:
        now, key, actor = _now(now), _identifier(idempotency_key, "idempotency_key"), _identifier(actor, "actor")
        conversation = self.conversation(session, run_id)
        request = {"base": canonical_uuid(base_version_id), "instruction": instruction,
            "sources": source_refs, "actor": actor, "parent": parent_turn_id}
        if selected_passage is not None:
            request["selected_passage"] = selected_passage.model_dump(mode="json")
        if proposal_turn_id is not None:
            request["proposal_turn_id"] = canonical_uuid(proposal_turn_id)
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with _atomic(session):
            session.execute(update(AssistantConversation).where(AssistantConversation.id == conversation.id).values(
                next_message_sequence=AssistantConversation.next_message_sequence))
            old = session.scalar(select(AssistantTurn).where(AssistantTurn.conversation_id == conversation.id,
                AssistantTurn.idempotency_key == key).execution_options(populate_existing=True))
            if old is not None:
                if old.request_sha256 != digest:
                    raise TurnConflict("request key was already used for another planning turn")
                self.reauthorize_context(session, self.effective_context(old))
                return old
            from .assistant_run_result import ensure_no_unresolved_intent
            ensure_no_unresolved_intent(session, conversation.run_id)
            pending = session.scalar(select(func.count()).select_from(AssistantTurn).where(AssistantTurn.conversation_id == conversation.id,
                AssistantTurn.status.in_(("queued", "running", "cancel_requested"))))
            if pending >= self.max_pending:
                raise TurnConflict("planning queue is full")
            reference_plan = None
            if proposal_turn_id is not None:
                original = self.require_turn(session, run_id, proposal_turn_id)
                if (original.status != "succeeded" or not original.reply or original.reply.get("kind") != "propose"
                        or original.base_version_id != canonical_uuid(base_version_id)):
                    raise TurnConflict("custom option requires a successful proposal on the same base version")
                self.reauthorize_context(session, self.effective_context(original))
                reference_plan = ProposeReply.model_validate(original.reply).plan
                inherited = tuple((op.source.asset_id, op.source.version_id) for op in reference_plan.operations
                    if getattr(op, "source", None) is not None)
                source_refs = tuple(dict.fromkeys((*source_refs, *inherited)))
            context = pinned_context or self.build_context(session, run_id=run_id, base_version_id=base_version_id,
                instruction=instruction, source_refs=source_refs, selected_passage=selected_passage,
                reference_turn_id=proposal_turn_id, reference_plan=reference_plan)
            context = PlannerContext.model_validate(context.model_dump(mode="json"))
            if context.run_id != canonical_uuid(run_id) or context.base_version_id != canonical_uuid(base_version_id) or context.dialogue[-1].text != instruction:
                raise TurnConflict("planning context does not match the request")
            from .assistant_broker_client import context_fingerprint
            context_fingerprint(context)
            self.reauthorize_context(session, context)
            turn_id = str(uuid.uuid4())
            message = append_message(session, conversation.id, idempotency_key="turn:" + turn_id, actor=actor,
                role="operator", content=instruction, base_version_id=context.base_version_id, context_snapshot=context.model_dump(mode="json"))
            turn = AssistantTurn(id=turn_id, conversation_id=conversation.id, base_version_id=context.base_version_id,
                operator_message_id=message.id, parent_turn_id=parent_turn_id, idempotency_key=key, request_sha256=digest,
                actor=actor, context_snapshot=context.model_dump(mode="json"), status="queued", lease_generation=0,
                workflow_revision=2 if self.agent_enabled else 1,
                created_at=now, updated_at=now)
            session.add(turn)
            session.flush()
            return turn

    @staticmethod
    def _enrichment_binding(original: PlannerContext, enriched: PlannerContext) -> None:
        for field in ("run_id", "base_version_id", "selected", "dialogue", "settings", "selected_passage",
                      "subtitle_cues", "subtitle_style", "editable_subtitle_cue_ids", "reference_turn_id", "reference_plan"):
            if getattr(original, field) != getattr(enriched, field):
                raise TurnConflict("enrichment changed the pinned planning request")
        supplied = {source.asset_id: source for source in enriched.sources}
        if any(supplied.get(source.asset_id) != source for source in original.sources):
            raise TurnConflict("enrichment changed an explicitly supplied source")
        if not set(original.internal_read_dependencies).issubset(enriched.internal_read_dependencies):
            raise TurnConflict('enrichment removed metadata provenance')

    @classmethod
    def effective_context(cls, turn: AssistantTurn) -> PlannerContext:
        original = PlannerContext.model_validate(turn.context_snapshot)
        if turn.agent_context_snapshot is not None:
            enriched = PlannerContext.model_validate(turn.agent_context_snapshot)
            cls._enrichment_binding(original, enriched)
            return enriched
        if turn.enrichment_snapshot is None:
            return original
        from .assistant_broker_client import context_fingerprint
        snapshot = turn.enrichment_snapshot
        if not isinstance(snapshot, dict) or set(snapshot) != {"context", "sha256"}:
            raise TurnConflict("invalid persisted planning enrichment")
        enriched = PlannerContext.model_validate(snapshot["context"])
        cls._enrichment_binding(original, enriched)
        if context_fingerprint(enriched) != snapshot["sha256"]:
            raise TurnConflict("persisted planning enrichment fingerprint disagrees")
        return enriched

    def save_enrichment(self, session: Session, lease: TurnLease, context: PlannerContext, *, now: datetime) -> PlannerContext:
        from .assistant_broker_client import context_fingerprint
        clock = _OperationClock.start(now)
        with _atomic(session):
            turn = _lock_owned(session, lease, clock)
            context = PlannerContext.model_validate(context.model_dump(mode="json"))
            self._enrichment_binding(PlannerContext.model_validate(turn.context_snapshot), context)
            self.reauthorize_context(session, context)
            if turn.enrichment_snapshot is not None:
                existing = self.effective_context(turn)
                if existing != context:
                    raise TurnConflict("planning enrichment is already pinned")
                return existing
            clock.require_live(turn.lease_expires_at)
            turn.enrichment_snapshot = {"context": context.model_dump(mode="json"), "sha256": context_fingerprint(context)}
            session.flush()
            return context

    def load_context(self, session: Session, lease: TurnLease, *, now: datetime) -> PlannerContext:
        clock = _OperationClock.start(now)
        with _atomic(session):
            turn = _lock_owned(session, lease, clock)
            context = self.effective_context(turn)
            self.reauthorize_context(session, context)
            clock.require_live(turn.lease_expires_at)
            return context

    def save_agent_context(self, session, lease, context, *, now):
        clock = _OperationClock.start(now)
        with _atomic(session):
            turn = _lock_owned(session, lease, clock)
            self._enrichment_binding(self.effective_context(turn), context)
            self.reauthorize_context(session, context)
            if turn.agent_context_snapshot is not None:
                raise TurnConflict("agent final context already exists")
            turn.agent_context_snapshot = context.model_dump(mode="json")
            session.flush()

    def agent_public_state(self, session, turn):
        """Authorize every observed dependency before exposing any evidence."""
        from .assistant_agent_contracts import AgentStepContext, AgentObservation
        rows = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == turn.id)
            .order_by(AssistantAgentStep.step_index)).all()
        if not rows:
            return None
        # Byte evidence is reused only within this invocation. Every context
        # still gets fresh live ownership/policy checks; storage reopens and
        # compares the exact file identity before reusing a successful hash.
        verification_cache = {}
        public = []
        verification = None
        for row in rows:
            step = restore_step(session, row)
            self.reauthorize_context(session, step.context, verification_cache=verification_cache)
            observation = AgentObservation.model_validate(row.observation) if row.observation is not None else None
            if observation is not None and observation.sources:
                sources = {source.asset_id: source for source in step.context.sources}
                sources.update({source.asset_id: source for source in observation.sources})
                self.reauthorize_context(session, step.context.model_copy(update={"sources": tuple(sources.values())}),
                    verification_cache=verification_cache)
            if observation is not None and observation.internal_read is not None:
                from .assistant_agent_contracts import internal_read_dependencies
                from .assistant_internal_api import extend_dependencies
                self.reauthorize_context(session, extend_dependencies(step.context,
                    internal_read_dependencies(observation.internal_read)), verification_cache=verification_cache)
            item = {"step_index": row.step_index, "status": row.status,
                "action": row.action.get("kind") if row.action else None, "observation": None}
            if observation is not None:
                safe = observation.model_dump(mode="json")
                for frame in safe.get("frames", []):
                    frame.pop("data_base64", None)
                item["observation"] = safe
            if row.private_artifact is not None:
                verification = {key: value for key, value in row.private_artifact.items()
                    if key in ("candidate_id", "plan_sha256", "candidate_rendered", "candidate_frames_inspected", "operator_review_required", "inspected_timestamps_sec", "candidate_speech_inspected")}
            public.append(item)
        return {"phase": public[-1]["action"] or "planning", "step_index": rows[-1].step_index,
            "steps": public, "verification": verification}

    def finish(self, session: Session, lease: TurnLease, reply: PlannerReply, *, now: datetime) -> AssistantTurn:
        clock = _OperationClock.start(now)
        # AgentReply is PlannerReply plus the agent-only caption_lines (AICP-345).
        reply = TypeAdapter(AgentReply).validate_python(TypeAdapter(AgentReply).dump_python(reply, mode="json"))
        with _atomic(session):
            turn = _lock_owned(session, lease, clock)
            context = self.effective_context(turn)
            self.reauthorize_context(session, context)
            if (reply.run_id, reply.base_version_id) != (context.run_id, context.base_version_id):
                raise TurnConflict("planner reply does not match the pinned context")
            if reply.kind == "caption_lines" and turn.workflow_revision != 2:
                raise TurnConflict("caption proposals require the evidence-backed agent workflow")
            if isinstance(reply, ProposeReply):
                if any(isinstance(operation,ExtendEndingOperation) for operation in reply.plan.operations):
                    self.require_verified_ending(session,turn,reply.plan)
                validate_plan_sources(reply.plan, base_version=VersionManifest(version_id=context.base_version_id,
                    run_id=context.run_id, duration_sec=context.selected.duration_sec, status="succeeded", available=True,
                    editable_subtitle_cue_ids=context.editable_subtitle_cue_ids), assets={source.asset_id: AssetManifest(
                        asset_id=source.asset_id, version_id=source.version_id, media_kind=source.media_kind,
                        duration_sec=source.duration_sec, has_audio=source.has_audio, available=True) for source in (context.selected, *context.sources)})
                # The schema describes more than the currently executable beta.
                # Never offer a Render action for a known unsupported plan shape.
                try:
                    validate_local_plan_capability(reply.plan, duration_sec=context.selected.duration_sec,
                        fps=context.selected.fps)
                except ValueError as exc:
                    reply = UnavailableReply(kind="unavailable", run_id=context.run_id,
                        base_version_id=context.base_version_id,
                        explanation=f"This proposal cannot be rendered in the current beta: {exc}. "
                            "No edit was applied. Adjust the marked boundaries or ask for supported changes in a separate version.")
            text = reply.explanation + ("\n\n" + reply.question if reply.kind == "clarify" else "")
            message = append_message(session, turn.conversation_id, idempotency_key="reply:" + turn.id,
                actor="assistant", role="assistant", content=text, base_version_id=turn.base_version_id,
                context_snapshot={"turn_id": turn.id, "reply": TypeAdapter(AgentReply).dump_python(reply, mode="json")})
            clock.require_live(turn.lease_expires_at)
            turn.assistant_message_id, turn.reply = message.id, TypeAdapter(AgentReply).dump_python(reply, mode="json")
            _terminal(turn, "succeeded", clock.now())
            session.flush()
            return turn

    def require_verified_ending(self, session, turn, plan):
        """Ending extension is admitted only from its exact durable agent trial."""
        from .assistant_agent_contracts import AgentActionEnvelope, FinalAction, validate_agent_action, plan_fingerprint
        from .assistant_store import PinnedFile
        if turn.workflow_revision != 2 or turn.status not in ('running','succeeded'):
            raise TurnConflict('ending extension requires a candidate-capable agent turn')
        if turn.status == 'succeeded' and (not turn.reply or turn.reply.get('kind') != 'propose'
                or ProposeReply.model_validate(turn.reply).plan != plan):
            raise TurnConflict('ending extension must match the persisted proposal')
        context = self.effective_context(turn)
        self.reauthorize_context(session,context)
        if (plan.run_id,plan.base_version_id) != (context.run_id,context.base_version_id):
            raise TurnConflict('ending extension changed the pinned proposal')
        rows = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == turn.id,
            AssistantAgentStep.status == 'completed',AssistantAgentStep.private_artifact.is_not(None))).all()
        finals = [row for row in rows if (row.action or {}).get('kind') == 'final']
        if len(finals) != 1:
            raise TurnConflict('ending extension requires its retained final candidate')
        row = finals[0]
        step = restore_step(session,row)
        if step.context != context:
            raise TurnConflict('ending candidate changed its context')
        action = AgentActionEnvelope.model_validate({'action':row.action}).action
        if not isinstance(action,FinalAction) or not isinstance(action.reply,ProposeReply) or action.reply.plan != plan:
            raise TurnConflict('ending extension must match its complete verified plan')
        validate_agent_action(action,step)
        candidate = next((obs.candidate for obs in step.observations if obs.candidate is not None
            and obs.candidate.candidate_id == action.candidate_id),None)
        private = row.private_artifact
        if (candidate is None or private.get('candidate_id') != candidate.candidate_id
                or private.get('plan_sha256') != plan_fingerprint(plan)
                or private.get('sha256') != candidate.sha256 or private.get('size_bytes') != candidate.size_bytes
                or private.get('candidate_rendered') is not True or private.get('candidate_frames_inspected') is not True):
            raise TurnConflict('ending extension requires exact candidate integrity bindings')
        self.storage.verify(PinnedFile(storage_key=private.get('storage_key'),sha256=candidate.sha256,
            size_bytes=candidate.size_bytes))

    def retry(self, session: Session, *, run_id: str, turn_id: str, idempotency_key: str, now: datetime) -> AssistantTurn:
        turn = self.require_turn(session, run_id, turn_id)
        if turn.status not in ("failed", "cancelled") or turn.edit_job_id is not None:
            raise TurnConflict("only a failed or cancelled planning turn can be retried")
        context = self.effective_context(turn)
        return self.accept(session, run_id=run_id, base_version_id=turn.base_version_id,
            instruction=context.dialogue[-1].text, idempotency_key=idempotency_key, actor=turn.actor, now=now,
            source_refs=tuple((source.asset_id, source.version_id) for source in context.sources),
            parent_turn_id=turn.id, pinned_context=context, selected_passage=context.selected_passage)

# Keep the accepted request immutable for direct SQL too, including isolated
# create_all fixtures. The additive migration contains a fixed equivalent.

_turn_columns = sorted(_TURN_PINNED)
_sqlite_changed = " OR ".join(f"NEW.{name} IS NOT OLD.{name}" for name in _turn_columns)
_sqlite_changed += " OR (OLD.agent_context_snapshot IS NOT NULL AND NEW.agent_context_snapshot IS NOT OLD.agent_context_snapshot) OR (OLD.agent_context_snapshot IS NULL AND NEW.agent_context_snapshot IS NOT NULL AND (OLD.status <> 'running' OR OLD.workflow_revision <> 2))"
_sqlite_changed += " OR (OLD.broker_context_sha256 IS NOT NULL AND NEW.broker_context_sha256 IS NOT OLD.broker_context_sha256)"
_sqlite_changed += " OR (OLD.enrichment_snapshot IS NOT NULL AND NEW.enrichment_snapshot IS NOT OLD.enrichment_snapshot) OR (OLD.enrichment_snapshot IS NULL AND NEW.enrichment_snapshot IS NOT NULL AND OLD.status <> 'running')"
_sqlite_changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS NOT OLD.status OR NEW.reply IS NOT OLD.reply OR NEW.assistant_message_id IS NOT OLD.assistant_message_id OR NEW.finished_at IS NOT OLD.finished_at OR NEW.error_code IS NOT OLD.error_code))"
_pg_changed = " OR ".join(f"to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name})" for name in _turn_columns)
_pg_changed += " OR (OLD.agent_context_snapshot IS NOT NULL AND NEW.agent_context_snapshot::jsonb IS DISTINCT FROM OLD.agent_context_snapshot::jsonb) OR (OLD.agent_context_snapshot IS NULL AND NEW.agent_context_snapshot IS NOT NULL AND (OLD.status <> 'running' OR OLD.workflow_revision <> 2))"
_pg_changed += " OR (OLD.broker_context_sha256 IS NOT NULL AND NEW.broker_context_sha256 IS DISTINCT FROM OLD.broker_context_sha256)"
_pg_changed += " OR (OLD.enrichment_snapshot IS NOT NULL AND NEW.enrichment_snapshot::jsonb IS DISTINCT FROM OLD.enrichment_snapshot::jsonb) OR (OLD.enrichment_snapshot IS NULL AND NEW.enrichment_snapshot IS NOT NULL AND OLD.status <> 'running')"
_pg_changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS DISTINCT FROM OLD.status OR NEW.reply::jsonb IS DISTINCT FROM OLD.reply::jsonb OR NEW.assistant_message_id IS DISTINCT FROM OLD.assistant_message_id OR NEW.finished_at IS DISTINCT FROM OLD.finished_at OR NEW.error_code IS DISTINCT FROM OLD.error_code))"
event.listen(AssistantTurn.__table__, "after_create", DDL(
    "CREATE TRIGGER assistant_turns_immutable BEFORE UPDATE ON assistant_turns "
    f"WHEN {_sqlite_changed} BEGIN SELECT RAISE(ABORT, 'immutable planning turn'); END"
).execute_if(dialect="sqlite"))
event.listen(AssistantTurn.__table__, "after_create", DDL(
    "CREATE OR REPLACE FUNCTION assistant_turns_reject_update() RETURNS trigger AS $$ "
    f"BEGIN IF {_pg_changed} THEN RAISE EXCEPTION 'immutable planning turn'; END IF; RETURN NEW; END; "
    "$$ LANGUAGE plpgsql; CREATE TRIGGER assistant_turns_immutable BEFORE UPDATE ON assistant_turns "
    "FOR EACH ROW EXECUTE FUNCTION assistant_turns_reject_update()"
).execute_if(dialect="postgresql"))
event.listen(AssistantTurn.__table__, "after_drop", DDL(
    "DROP FUNCTION IF EXISTS assistant_turns_reject_update()"
).execute_if(dialect="postgresql"))
