"""Assistant history independent of legacy Run deletion and mutable outputs.

Run/project IDs are provenance, deliberately not cascading legacy foreign keys.
Inside this graph all references RESTRICT deletion. Semantic records are immutable
through ORM events; status/selection/counters are the explicit mutable boundary.
No table here represents a render queue or grants project access.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON, BigInteger, Boolean, CheckConstraint, DDL, ForeignKey, ForeignKeyConstraint, Index,
    Integer, String, Text, UniqueConstraint, event, func, inspect, text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AssistantConversation(Base):
    __tablename__ = "assistant_conversations"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_assistant_conversation_run"),
        CheckConstraint("next_version_sequence > 0 AND next_message_sequence > 0",
                        name="ck_assistant_conversation_counters"),
        Index("ix_assistant_conversations_project_id", "project_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    next_version_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    next_message_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())


class AssistantVersion(Base):
    __tablename__ = "assistant_versions"
    __table_args__ = (
        UniqueConstraint("conversation_id", "id", name="uq_assistant_version_scope_id"),
        UniqueConstraint("conversation_id", "sequence", name="uq_assistant_version_sequence"),
        UniqueConstraint("conversation_id", "idempotency_key", name="uq_assistant_version_request"),
        ForeignKeyConstraint(
            ["conversation_id", "parent_version_id"],
            ["assistant_versions.conversation_id", "assistant_versions.id"],
            name="fk_assistant_version_parent_scope", ondelete="RESTRICT",
        ),
        CheckConstraint("sequence > 0 AND selection_epoch >= 0", name="ck_assistant_version_counters"),
        CheckConstraint("status IN ('pending','succeeded','failed','cancelled')", name="ck_assistant_version_status"),
        CheckConstraint(
            "(parent_version_id IS NULL AND base_version_id IS NULL AND sequence = 1) OR "
            "(parent_version_id IS NOT NULL AND base_version_id IS NOT NULL "
            "AND parent_version_id = base_version_id AND sequence > 1 AND parent_version_id <> id)",
            name="ck_assistant_version_base_parent",
        ),
        Index("ix_assistant_versions_parent_version_id", "parent_version_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    base_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    selection_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    recipe: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", server_default="pending")
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class AssistantSelection(Base):
    """Only manual selection changes epoch; automatic successes preserve it."""

    __tablename__ = "assistant_selections"
    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id", "selected_version_id"],
            ["assistant_versions.conversation_id", "assistant_versions.id"],
            name="fk_assistant_selection_selected_scope", ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["conversation_id", "latest_version_id"],
            ["assistant_versions.conversation_id", "assistant_versions.id"],
            name="fk_assistant_selection_latest_scope", ondelete="RESTRICT",
        ),
        CheckConstraint("epoch >= 0", name="ck_assistant_selection_epoch"),
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), primary_key=True,
    )
    selected_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    latest_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")


class AssistantArtifact(Base):
    __tablename__ = "assistant_artifacts"
    __table_args__ = (
        CheckConstraint("size_bytes > 0", name="ck_assistant_artifact_size"),
        CheckConstraint("media_kind IN ('video','audio','image','subtitle')", name="ck_assistant_artifact_media"),
        Index("ix_assistant_artifacts_owner_version_id", "owner_version_id"),
        Index("ix_assistant_artifacts_sha256", "sha256"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False,
    )
    storage_key: Mapped[str] = mapped_column(String(96), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())


class AssistantVersionArtifact(Base):
    __tablename__ = "assistant_version_artifacts"
    __table_args__ = (
        UniqueConstraint("version_id", "role", "slot", name="uq_assistant_version_artifact_slot"),
        CheckConstraint("role IN ('full','preview','input','clean','captions') AND slot >= 0", name="ck_assistant_version_artifact_role"),
        CheckConstraint("role = 'input' OR slot = 0", name="ck_assistant_version_artifact_output_slot"),
        Index("ix_assistant_version_artifacts_artifact_id", "artifact_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False,
    )
    artifact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assistant_artifacts.id", ondelete="RESTRICT"), nullable=False,
    )
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    slot: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")


class AssistantMessage(Base):
    __tablename__ = "assistant_messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence", name="uq_assistant_message_sequence"),
        UniqueConstraint("conversation_id", "idempotency_key", name="uq_assistant_message_request"),
        ForeignKeyConstraint(
            ["conversation_id", "base_version_id"],
            ["assistant_versions.conversation_id", "assistant_versions.id"],
            name="fk_assistant_message_base_scope", ondelete="RESTRICT",
        ),
        CheckConstraint("sequence > 0", name="ck_assistant_message_sequence"),
        CheckConstraint("role IN ('operator','assistant','tool','system')", name="ck_assistant_message_role"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    base_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())



class AssistantRunResultIntent(Base):
    """Durable writer exclusion and private exact-session recovery identity."""
    __tablename__ = "assistant_run_result_intents"
    __table_args__ = (
        UniqueConstraint("conversation_id", "idempotency_key", name="uq_assistant_run_result_intent_request"),
        UniqueConstraint("id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "actor", "request_sha256", name="uq_assistant_run_result_intent_binding"),
        ForeignKeyConstraint(["conversation_id", "version_id"],
            ["assistant_versions.conversation_id", "assistant_versions.id"],
            name="fk_assistant_run_result_intent_version_scope", ondelete="RESTRICT"),
        Index("uq_assistant_run_result_intent_unresolved", "run_id", unique=True,
              postgresql_where=text("completed_at IS NULL"), sqlite_where=text("completed_at IS NULL")),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    version_id: Mapped[str] = mapped_column(String(36), nullable=False)
    artifact_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_artifacts.id", ondelete="RESTRICT"), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    request_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    before_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    after_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Never return or log this bearer-like provider session identity.
    resumable_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class AssistantRunResultPromotion(Base):
    """Append-only receipt. Run IDs remain provenance after legacy deletion."""
    __tablename__ = "assistant_run_result_promotions"
    __table_args__ = (
        Index("ix_assistant_run_result_promotions_run", "run_id", "created_at"),
        ForeignKeyConstraint(["id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "actor", "request_sha256"],
            ["assistant_run_result_intents." + column for column in ("id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "actor", "request_sha256")],
            name="fk_assistant_run_result_receipt_intent_binding", ondelete="RESTRICT"),
        CheckConstraint("delivery IN ('drive_updated','local_only')", name="ck_assistant_run_result_delivery"),
    )
    id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_run_result_intents.id", ondelete="RESTRICT"), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False)
    artifact_id: Mapped[str] = mapped_column(String(36), ForeignKey("assistant_artifacts.id", ondelete="RESTRICT"), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    before_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    after_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    delivery: Mapped[str] = mapped_column(String(16), nullable=False)
    noop: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(default=_now, server_default=func.now())

_VERSION_MUTABLE = {"status", "completed_at"}
_CONVERSATION_MUTABLE = {"next_version_sequence", "next_message_sequence"}


def _prevent_semantic_update(mapper, connection, target):
    mutable = (_VERSION_MUTABLE if isinstance(target, AssistantVersion) else
               _CONVERSATION_MUTABLE if isinstance(target, AssistantConversation) else set())
    if isinstance(target, AssistantVersion):
        history = inspect(target).attrs.status.history
        if history.deleted and history.deleted[0] != "pending" and history.has_changes():
            raise ValueError("terminal assistant version cannot change status")
    for attribute in inspect(target).attrs:
        if attribute.key not in mutable and attribute.history.has_changes():
            raise ValueError(f"immutable assistant record field: {attribute.key}")


for _model in (AssistantConversation, AssistantVersion, AssistantArtifact, AssistantVersionArtifact, AssistantMessage):
    event.listen(_model, "before_update", _prevent_semantic_update)


# Enforce the same boundary for Core/direct SQL, not just ORM assignment. These
# DDL hooks cover isolated create_all schemas; the Alembic migration contains its
# own fixed equivalent so migration replay never depends on application imports.
for _model in (AssistantConversation, AssistantVersion, AssistantArtifact, AssistantVersionArtifact, AssistantMessage):
    _table = _model.__table__
    _mutable = (_VERSION_MUTABLE if _model is AssistantVersion else
                _CONVERSATION_MUTABLE if _model is AssistantConversation else set())
    _protected = [column.name for column in _table.columns
                  if column.name not in _mutable]
    _sqlite_changed = " OR ".join(f"NEW.{column} IS NOT OLD.{column}" for column in _protected)
    _pg_changed = "to_jsonb(NEW) IS DISTINCT FROM to_jsonb(OLD)"
    if _model is AssistantConversation:
        _pg_changed = (
            "(to_jsonb(NEW) - ARRAY['next_version_sequence','next_message_sequence']) IS DISTINCT FROM "
            "(to_jsonb(OLD) - ARRAY['next_version_sequence','next_message_sequence'])"
        )
    if _model is AssistantVersion:
        _sqlite_changed += " OR (OLD.status <> 'pending' AND (NEW.status IS NOT OLD.status OR NEW.completed_at IS NOT OLD.completed_at))"
        _pg_changed = (
            "(to_jsonb(NEW) - ARRAY['status','completed_at']) IS DISTINCT FROM "
            "(to_jsonb(OLD) - ARRAY['status','completed_at']) OR "
            "(OLD.status <> 'pending' AND to_jsonb(NEW) IS DISTINCT FROM to_jsonb(OLD))"
        )
    event.listen(_table, "after_create", DDL(
        f"CREATE TRIGGER {_table.name}_immutable BEFORE UPDATE ON {_table.name} "
        f"WHEN {_sqlite_changed} BEGIN SELECT RAISE(ABORT, 'immutable assistant record'); END"
    ).execute_if(dialect="sqlite"))
    event.listen(_table, "after_create", DDL(
        f"CREATE OR REPLACE FUNCTION {_table.name}_reject_update() RETURNS trigger AS $$ "
        f"BEGIN IF {_pg_changed} THEN RAISE EXCEPTION 'immutable assistant record'; END IF; RETURN NEW; END; "
        f"$$ LANGUAGE plpgsql; CREATE TRIGGER {_table.name}_immutable BEFORE UPDATE ON {_table.name} "
        f"FOR EACH ROW EXECUTE FUNCTION {_table.name}_reject_update()"
    ).execute_if(dialect="postgresql"))
    event.listen(_table, "after_drop", DDL(
        f"DROP FUNCTION IF EXISTS {_table.name}_reject_update()"
    ).execute_if(dialect="postgresql"))


# Recovery fields may advance only once; semantic snapshots and receipts never
# change, including through direct SQL. Deletion cannot erase a writer fence.
def _result_guard(mapper, connection, target):
    allowed = {"resumable_uri", "completed_at"} if isinstance(target, AssistantRunResultIntent) else set()
    if isinstance(target, AssistantRunResultIntent):
        completed_history = inspect(target).attrs.completed_at.history
        if completed_history.unchanged and completed_history.unchanged[0] is not None:
            allowed = set()
    for attribute in inspect(target).attrs:
        if attribute.history.has_changes():
            if attribute.key not in allowed or (attribute.history.deleted and attribute.history.deleted[0] is not None):
                raise ValueError("immutable assistant publication evidence")


def _result_no_delete(mapper, connection, target):
    raise ValueError("immutable assistant publication evidence")


for _result_model in (AssistantRunResultIntent, AssistantRunResultPromotion):
    event.listen(_result_model, "before_update", _result_guard)
    event.listen(_result_model, "before_delete", _result_no_delete)
    _name = _result_model.__tablename__
    _once = ("resumable_uri", "completed_at") if _result_model is AssistantRunResultIntent else ()
    _protected = [c.name for c in _result_model.__table__.columns if c.name not in _once]
    _changed = " OR ".join(f"NEW.{c} IS NOT OLD.{c}" for c in _protected)
    _pg_changed = " OR ".join(f"to_jsonb(NEW.{c}) IS DISTINCT FROM to_jsonb(OLD.{c})" for c in _protected)
    for _c in _once:
        _changed += f" OR (OLD.{_c} IS NOT NULL AND NEW.{_c} IS NOT OLD.{_c})"
        _pg_changed += f" OR (OLD.{_c} IS NOT NULL AND to_jsonb(NEW.{_c}) IS DISTINCT FROM to_jsonb(OLD.{_c}))"
    if _result_model is AssistantRunResultIntent:
        _changed += " OR (OLD.completed_at IS NOT NULL AND NEW.resumable_uri IS NOT OLD.resumable_uri)"
        _pg_changed += " OR (OLD.completed_at IS NOT NULL AND to_jsonb(NEW) IS DISTINCT FROM to_jsonb(OLD))"
    event.listen(_result_model.__table__, "after_create", DDL(
        f"CREATE TRIGGER {_name}_immutable BEFORE UPDATE ON {_name} WHEN {_changed} BEGIN SELECT RAISE(ABORT, 'immutable assistant publication evidence'); END"
    ).execute_if(dialect="sqlite"))
    event.listen(_result_model.__table__, "after_create", DDL(
        f"CREATE TRIGGER {_name}_no_delete BEFORE DELETE ON {_name} BEGIN SELECT RAISE(ABORT, 'immutable assistant publication evidence'); END"
    ).execute_if(dialect="sqlite"))
    event.listen(_result_model.__table__, "after_create", DDL(
        f"CREATE FUNCTION {_name}_reject_update() RETURNS trigger AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'immutable assistant publication evidence'; END IF; IF {_pg_changed} THEN RAISE EXCEPTION 'immutable assistant publication evidence'; END IF; RETURN NEW; END; $$ LANGUAGE plpgsql; CREATE TRIGGER {_name}_immutable BEFORE UPDATE OR DELETE ON {_name} FOR EACH ROW EXECUTE FUNCTION {_name}_reject_update()"
    ).execute_if(dialect="postgresql"))
    event.listen(_result_model.__table__, "after_drop", DDL(
        f"DROP FUNCTION IF EXISTS {_name}_reject_update()"
    ).execute_if(dialect="postgresql"))
