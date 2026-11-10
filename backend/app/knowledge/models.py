"""Knowledge-base schema (design docs/knowledge-feedback-design.md §4).

Evidence is immutable identity rows plus insert-only observation rows; every
table except ``kb_import_cursors`` rejects UPDATE, DELETE and TRUNCATE in the
database (and through ORM events). References to existing production tables
(runs, projects, segments, draws) are soft strings: Runs can be hard-deleted
and the knowledge record must outlive them. Inside the ``kb_`` graph every
foreign key RESTRICTs. ``seq`` is the total order; timestamps never break ties.

The Alembic revision ``k1a7c3e9f205`` carries a fixed copy of this schema so
migration replay never imports application code; a PostgreSQL test compares
the two.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON, BigInteger, Boolean, CheckConstraint, DDL, DateTime, Float, ForeignKey, Identity,
    Index, Integer, String, Text, UniqueConstraint, event, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base
from . import taxonomy as t
from .taxonomy import check_sql

Json = JSON().with_variant(JSONB(), "postgresql")
ID = String(36)


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id():
    return mapped_column(ID, primary_key=True, default=_uuid)


def _seq():
    return mapped_column(BigInteger, Identity(always=True), nullable=False, unique=True)


def _created():
    return mapped_column(DateTime(timezone=True), nullable=False, default=_now, server_default=func.now())


def _pg_check(sql: str, name: str):
    """Regex CHECKs are PostgreSQL syntax; SQLite create_all fixtures skip them."""
    return CheckConstraint(sql, name=name).ddl_if(dialect="postgresql")


def _fk(target: str, nullable=True):
    return mapped_column(ID, ForeignKey(target, ondelete="RESTRICT"), nullable=nullable)


class KbImportBatch(Base):
    __tablename__ = "kb_import_batches"
    id: Mapped[str] = _id()
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    code_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False)
    started_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


class KbImportBatchObservation(Base):
    __tablename__ = "kb_import_batch_observations"
    __table_args__ = (CheckConstraint(check_sql("kind", t.IMPORT_OUTCOMES), name="ck_kb_import_obs_kind"),)
    id: Mapped[str] = _id()
    batch_id: Mapped[str] = _fk("kb_import_batches.id", nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    counts: Mapped[Optional[dict]] = mapped_column(Json)
    report: Mapped[Optional[dict]] = mapped_column(Json)
    observed_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


class KbArtifact(Base):
    __tablename__ = "kb_artifacts"
    __table_args__ = (
        UniqueConstraint("locator_kind", "locator", "sha256", "revision", name="uq_kb_artifact_locator_content"),
        CheckConstraint("revision > 0", name="ck_kb_artifact_revision"),
        CheckConstraint(check_sql("kind", t.ARTIFACT_KINDS), name="ck_kb_artifact_kind"),
        CheckConstraint(check_sql("locator_kind", t.LOCATOR_KINDS), name="ck_kb_artifact_locator_kind"),
        _pg_check("sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'", name="ck_kb_artifact_sha256"),
        CheckConstraint("bytes IS NULL OR bytes >= 0", name="ck_kb_artifact_bytes"),
        CheckConstraint("duration_sec IS NULL OR duration_sec >= 0", name="ck_kb_artifact_duration"),
        CheckConstraint("time_start_sec IS NULL OR time_end_sec IS NULL OR "
                        "(time_start_sec >= 0 AND time_end_sec > time_start_sec)", name="ck_kb_artifact_range"),
        Index("ix_kb_artifacts_project", "project_id"),
        Index("ix_kb_artifacts_sha256", "sha256"),
    )
    id: Mapped[str] = _id()
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    locator_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    locator: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    # A withdrawn (invalidated) artifact is replaced by revision n+1, never reused.
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    parent_artifact_id: Mapped[Optional[str]] = _fk("kb_artifacts.id")
    time_start_sec: Mapped[Optional[float]] = mapped_column(Float)
    time_end_sec: Mapped[Optional[float]] = mapped_column(Float)
    project_id: Mapped[Optional[str]] = mapped_column(ID)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbArtifactCheck(Base):
    __tablename__ = "kb_artifact_checks"
    __table_args__ = (Index("ix_kb_artifact_checks_artifact", "artifact_id", "seq"),)
    id: Mapped[str] = _id()
    artifact_id: Mapped[str] = _fk("kb_artifacts.id", nullable=False)
    present: Mapped[bool] = mapped_column(Boolean, nullable=False)
    sha256_matches: Mapped[Optional[bool]] = mapped_column(Boolean)
    checked_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


class KbCase(Base):
    __tablename__ = "kb_cases"
    __table_args__ = (
        UniqueConstraint("case_key", "revision", name="uq_kb_case_key"),
        CheckConstraint("revision > 0", name="ck_kb_case_revision"),
        CheckConstraint(check_sql("scope", t.CASE_SCOPES), name="ck_kb_case_scope"),
        CheckConstraint(check_sql("operation", t.CASE_OPERATIONS), name="ck_kb_case_operation"),
        CheckConstraint(check_sql("origin", t.ORIGINS), name="ck_kb_case_origin"),
        CheckConstraint("project_id IS NOT NULL OR scope = 'external'", name="ck_kb_case_project"),
        CheckConstraint("(origin = 'import') = (import_batch_id IS NOT NULL)", name="ck_kb_case_import_batch"),
        CheckConstraint("time_start_sec IS NULL OR time_end_sec IS NULL OR "
                        "(time_start_sec >= 0 AND time_end_sec > time_start_sec)", name="ck_kb_case_range"),
        CheckConstraint("(scope = 'run') = (parent_case_id IS NULL) OR scope IN ('manual','external')",
                        name="ck_kb_case_parent"),
        Index("ix_kb_cases_project", "project_id"),
        Index("ix_kb_cases_run", "run_id"),
        Index("ix_kb_cases_parent", "parent_case_id"),
    )
    id: Mapped[str] = _id()
    case_key: Mapped[str] = mapped_column(String(200), nullable=False)
    # A withdrawn (invalidated) case is replaced by revision n+1 under the same key.
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    project_id: Mapped[Optional[str]] = mapped_column(ID)
    run_id: Mapped[Optional[str]] = mapped_column(ID)
    run_segment_id: Mapped[Optional[str]] = mapped_column(ID)
    segment_def_id: Mapped[Optional[str]] = mapped_column(ID)
    parent_case_id: Mapped[Optional[str]] = _fk("kb_cases.id")
    source_artifact_id: Mapped[Optional[str]] = _fk("kb_artifacts.id")
    source_group_key: Mapped[Optional[str]] = mapped_column(String(128))
    language: Mapped[Optional[str]] = mapped_column(String(8))
    time_start_sec: Mapped[Optional[float]] = mapped_column(Float)
    time_end_sec: Mapped[Optional[float]] = mapped_column(Float)
    origin: Mapped[str] = mapped_column(String(8), nullable=False)
    import_batch_id: Mapped[Optional[str]] = _fk("kb_import_batches.id")
    snapshot: Mapped[dict] = mapped_column(Json, nullable=False, default=dict)
    missing_facts: Mapped[list] = mapped_column(Json, nullable=False, default=list)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbCaseVersion(Base):
    __tablename__ = "kb_case_versions"
    __table_args__ = (
        UniqueConstraint("case_id", "version_no", name="uq_kb_case_version"),
        CheckConstraint("version_no > 0", name="ck_kb_case_version_no"),
        CheckConstraint(check_sql("reason", t.VERSION_REASONS), name="ck_kb_case_version_reason"),
        # One publication is one version, however often a writer retries it.
        Index("uq_kb_case_version_source", "case_id", "source_record", unique=True,
              postgresql_where=text("source_record IS NOT NULL"), sqlite_where=text("source_record IS NOT NULL")),
    )
    id: Mapped[str] = _id()
    case_id: Mapped[str] = _fk("kb_cases.id", nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_id: Mapped[str] = _fk("kb_artifacts.id", nullable=False)
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    source_record: Mapped[Optional[str]] = mapped_column(String(200))
    # A historical version records a publication that is no longer (or never
    # was) the live output; it never becomes current and is judged only with
    # historical feedback.
    historical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


class KbAttempt(Base):
    __tablename__ = "kb_attempts"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_kb_attempt_idempotency"),
        CheckConstraint(check_sql("attempt_kind", t.ATTEMPT_KINDS), name="ck_kb_attempt_kind"),
        CheckConstraint(check_sql("origin", t.ATTEMPT_ORIGINS), name="ck_kb_attempt_origin"),
        CheckConstraint(check_sql("provider", t.PROVIDERS), name="ck_kb_attempt_provider"),
        _pg_check("request_sha256 IS NULL OR request_sha256 ~ '^[0-9a-f]{64}$'", name="ck_kb_attempt_request_sha256"),
        Index("ix_kb_attempts_case", "case_id", "seq"),
        Index("ix_kb_attempts_dispatch", "dispatch_key"),
        Index("ix_kb_attempts_target", "target_kind", "target_id"),
        Index("ix_kb_attempts_model", "model_key"),
        Index("ix_kb_attempts_reference_set", "reference_set_sha256"),
    )
    id: Mapped[str] = _id()
    case_id: Mapped[str] = _fk("kb_cases.id", nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    attempt_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    origin: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_attempt_id: Mapped[Optional[str]] = _fk("kb_attempts.id")
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    model_key: Mapped[Optional[str]] = mapped_column(String(64))
    provider_model_id: Mapped[Optional[str]] = mapped_column(String(128))
    model_family: Mapped[Optional[str]] = mapped_column(String(64))
    target_kind: Mapped[Optional[str]] = mapped_column(String(32))
    target_id: Mapped[Optional[str]] = mapped_column(String(64))
    dispatch_key: Mapped[Optional[str]] = mapped_column(String(64))
    predecessor_task_id: Mapped[Optional[str]] = mapped_column(String(128))
    request: Mapped[Optional[dict]] = mapped_column(Json)
    request_sha256: Mapped[Optional[str]] = mapped_column(String(64))
    input_hashes: Mapped[Optional[dict]] = mapped_column(Json)
    reference_set_sha256: Mapped[Optional[str]] = mapped_column(String(64))
    recipe_context: Mapped[Optional[dict]] = mapped_column(Json)
    code_revision: Mapped[Optional[str]] = mapped_column(String(64))
    import_batch_id: Mapped[Optional[str]] = _fk("kb_import_batches.id")
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbAttemptSubmission(Base):
    __tablename__ = "kb_attempt_submissions"
    __table_args__ = (
        UniqueConstraint("attempt_id", name="uq_kb_submission_attempt"),
        UniqueConstraint("provider_task_id", name="uq_kb_submission_task"),
        UniqueConstraint("spend_call_key", name="uq_kb_submission_call_key"),
    )
    id: Mapped[str] = _id()
    attempt_id: Mapped[str] = _fk("kb_attempts.id", nullable=False)
    provider_task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    spend_call_key: Mapped[Optional[str]] = mapped_column(String(128))
    submitted_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


class KbAttemptObservation(Base):
    __tablename__ = "kb_attempt_observations"
    __table_args__ = (
        CheckConstraint(check_sql("kind", t.OBSERVATION_KINDS), name="ck_kb_observation_kind"),
        CheckConstraint("status IS NULL OR " + check_sql("status", t.ATTEMPT_STATUSES),
                        name="ck_kb_observation_status"),
        CheckConstraint("cost_source IS NULL OR " + check_sql("cost_source", t.COST_SOURCES),
                        name="ck_kb_observation_cost_source"),
        CheckConstraint("kind <> 'status' OR status IS NOT NULL", name="ck_kb_observation_status_kind"),
        CheckConstraint("kind <> 'output' OR output_artifact_id IS NOT NULL", name="ck_kb_observation_output"),
        CheckConstraint("kind NOT IN ('cost_estimate','cost_settled') OR "
                        "(mcredits IS NOT NULL AND mcredits >= 0 AND cost_source IS NOT NULL)",
                        name="ck_kb_observation_cost"),
        Index("ix_kb_observations_attempt", "attempt_id", "seq"),
        # The importer's task-id fallback (a submission withheld because its task id
        # is held by a withdrawn row) looks these up under a Run lock: indexed.
        Index("ix_kb_observations_submitted_task", text("(payload ->> 'provider_task_id')"),
              postgresql_where=text("kind = 'status' AND status = 'submitted'")).ddl_if(dialect="postgresql"),
        Index("ix_kb_observations_submitted_call", text("(payload ->> 'spend_call_key')"),
              postgresql_where=text("kind = 'status' AND status = 'submitted'")).ddl_if(dialect="postgresql"),
    )
    id: Mapped[str] = _id()
    attempt_id: Mapped[str] = _fk("kb_attempts.id", nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[Optional[str]] = mapped_column(String(32))
    output_artifact_id: Mapped[Optional[str]] = _fk("kb_artifacts.id")
    mcredits: Mapped[Optional[int]] = mapped_column(BigInteger)
    cost_source: Mapped[Optional[str]] = mapped_column(String(32))
    error_class: Mapped[Optional[str]] = mapped_column(String(64))
    payload: Mapped[Optional[dict]] = mapped_column(Json)
    import_batch_id: Mapped[Optional[str]] = _fk("kb_import_batches.id")
    observed_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


class KbRequest(Base):
    """One HTTP submit's receipt: replay returns it, a different body 409s."""
    __tablename__ = "kb_requests"
    __table_args__ = (_pg_check("request_sha256 ~ '^[0-9a-f]{64}$'", name="ck_kb_request_sha256"),)
    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    case_id: Mapped[Optional[str]] = _fk("kb_cases.id")
    child_event_ids: Mapped[list] = mapped_column(Json, nullable=False, default=list)
    created_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


class KbFeedbackEvent(Base):
    __tablename__ = "kb_feedback_events"
    __table_args__ = (
        UniqueConstraint("request_id", "ordinal", name="uq_kb_event_request_ordinal"),
        CheckConstraint(check_sql("event_type", t.EVENT_TYPES), name="ck_kb_event_type"),
        CheckConstraint(check_sql("actor_kind", t.ACTOR_KINDS), name="ck_kb_event_actor"),
        CheckConstraint("verdict IS NULL OR " + check_sql("verdict", t.VERDICTS), name="ck_kb_event_verdict"),
        CheckConstraint("defect_code IS NULL OR " + check_sql("defect_code", t.DEFECT_CODES),
                        name="ck_kb_event_defect_code"),
        CheckConstraint("severity IS NULL OR " + check_sql("severity", t.SEVERITIES), name="ck_kb_event_severity"),
        CheckConstraint("method IS NULL OR " + check_sql("method", t.CORRECTION_METHODS), name="ck_kb_event_method"),
        CheckConstraint("resolved IS NULL OR " + check_sql("resolved", t.RESOLVED), name="ck_kb_event_resolved"),
        CheckConstraint("cost_source IS NULL OR " + check_sql("cost_source", t.COST_SOURCES),
                        name="ck_kb_event_cost_source"),
        CheckConstraint("event_type <> 'operator_verdict' OR (verdict IS NOT NULL AND actor_kind = 'operator')",
                        name="ck_kb_event_verdict_actor"),
        CheckConstraint("event_type <> 'defect' OR (defect_code IS NOT NULL AND severity IS NOT NULL)",
                        name="ck_kb_event_defect"),
        CheckConstraint("event_type NOT IN ('operator_verdict','defect','auto_assessment') OR "
                        "(case_version_no IS NOT NULL AND artifact_id IS NOT NULL)", name="ck_kb_event_version"),
        CheckConstraint("event_type <> 'auto_assessment' OR (check_kind IS NOT NULL AND claim IS NOT NULL "
                        "AND actor_kind IN ('system_check','assistant','importer'))", name="ck_kb_event_auto"),
        CheckConstraint("event_type <> 'retraction' OR supersedes_event_id IS NOT NULL",
                        name="ck_kb_event_retraction"),
        CheckConstraint("event_type <> 'correction' OR (method IS NOT NULL AND resolved IS NOT NULL "
                        "AND before_artifact_id IS NOT NULL)", name="ck_kb_event_correction"),
        CheckConstraint("defect_code <> 'other' OR note IS NOT NULL", name="ck_kb_event_other_note"),
        CheckConstraint("defect_code NOT IN ('pronunciation','lip_sync') OR actor_kind = 'operator'",
                        name="ck_kb_event_operator_only_defect"),
        CheckConstraint("time_start_sec IS NULL OR time_end_sec IS NULL OR "
                        "(time_start_sec >= 0 AND time_end_sec > time_start_sec)", name="ck_kb_event_range"),
        CheckConstraint("(incremental_mcredits IS NULL OR incremental_mcredits >= 0) AND "
                        "(operator_minutes IS NULL OR operator_minutes >= 0)", name="ck_kb_event_cost"),
        Index("ix_kb_events_case", "case_id", "seq"),
        Index("ix_kb_events_defect", "defect_code"),
        Index("uq_kb_event_one_retraction", "supersedes_event_id", unique=True,
              postgresql_where=text("event_type = 'retraction'"), sqlite_where=text("event_type = 'retraction'")),
    )
    id: Mapped[str] = _id()
    request_id: Mapped[str] = mapped_column(String(128), ForeignKey("kb_requests.idempotency_key",
                                            ondelete="RESTRICT"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    case_id: Mapped[str] = _fk("kb_cases.id", nullable=False)
    case_version_no: Mapped[Optional[int]] = mapped_column(Integer)
    artifact_id: Mapped[Optional[str]] = _fk("kb_artifacts.id")
    historical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_label: Mapped[Optional[str]] = mapped_column(String(128))
    verdict: Mapped[Optional[str]] = mapped_column(String(16))
    defect_code: Mapped[Optional[str]] = mapped_column(String(32))
    severity: Mapped[Optional[str]] = mapped_column(String(8))
    time_start_sec: Mapped[Optional[float]] = mapped_column(Float)
    time_end_sec: Mapped[Optional[float]] = mapped_column(Float)
    check_kind: Mapped[Optional[str]] = mapped_column(String(32))
    claim: Mapped[Optional[str]] = mapped_column(String(64))
    coverage: Mapped[Optional[dict]] = mapped_column(Json)
    limitations: Mapped[Optional[str]] = mapped_column(Text)
    method: Mapped[Optional[str]] = mapped_column(String(16))
    before_artifact_id: Mapped[Optional[str]] = _fk("kb_artifacts.id")
    after_artifact_id: Mapped[Optional[str]] = _fk("kb_artifacts.id")
    resolved: Mapped[Optional[str]] = mapped_column(String(8))
    incremental_mcredits: Mapped[Optional[int]] = mapped_column(BigInteger)
    operator_minutes: Mapped[Optional[float]] = mapped_column(Float)
    cost_source: Mapped[Optional[str]] = mapped_column(String(32))
    note: Mapped[Optional[str]] = mapped_column(Text)
    supersedes_event_id: Mapped[Optional[str]] = _fk("kb_feedback_events.id")
    import_batch_id: Mapped[Optional[str]] = _fk("kb_import_batches.id")
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbRule(Base):
    __tablename__ = "kb_rules"
    __table_args__ = (UniqueConstraint("slug", name="uq_kb_rule_slug"),
                      _pg_check("slug ~ '^[a-z0-9][a-z0-9-]{1,79}$'", name="ck_kb_rule_slug"))
    id: Mapped[str] = _id()
    slug: Mapped[str] = mapped_column(String(80), nullable=False)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbRuleVersion(Base):
    __tablename__ = "kb_rule_versions"
    __table_args__ = (
        UniqueConstraint("rule_id", "version_no", name="uq_kb_rule_version"),
        CheckConstraint("version_no > 0", name="ck_kb_rule_version_no"),
        CheckConstraint(check_sql("kind", t.RULE_KINDS), name="ck_kb_rule_version_kind"),
        CheckConstraint(check_sql("confidence", t.RULE_CONFIDENCE), name="ck_kb_rule_version_confidence"),
        CheckConstraint(check_sql("author_kind", t.AUTHOR_KINDS), name="ck_kb_rule_version_author"),
        CheckConstraint("kind <> 'operator_constraint' OR author_kind IN ('operator','importer')",
                        name="ck_kb_rule_version_constraint_author"),
        CheckConstraint("effective_to IS NULL OR effective_to > effective_from", name="ck_kb_rule_version_window"),
        Index("ix_kb_rule_versions_rule", "rule_id", "version_no"),
        Index("ix_kb_rule_versions_statement_fts", text("to_tsvector('simple', statement)"),
              postgresql_using="gin").ddl_if(dialect="postgresql"),
    )
    id: Mapped[str] = _id()
    rule_id: Mapped[str] = _fk("kb_rules.id", nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    applicability: Mapped[dict] = mapped_column(Json, nullable=False, default=dict)
    recommendation: Mapped[dict] = mapped_column(Json, nullable=False)
    confidence: Mapped[str] = mapped_column(String(32), nullable=False)
    shareable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    scope_project_ids: Mapped[Optional[list]] = mapped_column(Json)
    model_registry_revision: Mapped[Optional[str]] = mapped_column(String(64))
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_to: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    author_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    author_label: Mapped[Optional[str]] = mapped_column(String(128))
    provenance: Mapped[dict] = mapped_column(Json, nullable=False, default=dict)
    evidence_gaps: Mapped[list] = mapped_column(Json, nullable=False, default=list)
    supersedes_version_id: Mapped[Optional[str]] = _fk("kb_rule_versions.id")
    import_batch_id: Mapped[Optional[str]] = _fk("kb_import_batches.id")
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbRuleReview(Base):
    __tablename__ = "kb_rule_reviews"
    __table_args__ = (CheckConstraint(check_sql("status", t.REVIEW_STATUSES), name="ck_kb_rule_review_status"),
                      Index("ix_kb_rule_reviews_version", "rule_version_id", "seq"))
    id: Mapped[str] = _id()
    rule_version_id: Mapped[str] = _fk("kb_rule_versions.id", nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    reviewer_label: Mapped[Optional[str]] = mapped_column(String(128))
    note: Mapped[Optional[str]] = mapped_column(Text)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbRuleEvidence(Base):
    __tablename__ = "kb_rule_evidence"
    __table_args__ = (
        CheckConstraint(check_sql("stance", t.EVIDENCE_STANCES), name="ck_kb_rule_evidence_stance"),
        UniqueConstraint("rule_version_id", "case_id", "event_id", "stance", name="uq_kb_rule_evidence"),
        Index("ix_kb_rule_evidence_version", "rule_version_id"),
        Index("ix_kb_rule_evidence_case", "case_id"),
    )
    id: Mapped[str] = _id()
    rule_version_id: Mapped[str] = _fk("kb_rule_versions.id", nullable=False)
    case_id: Mapped[str] = _fk("kb_cases.id", nullable=False)
    event_id: Mapped[Optional[str]] = _fk("kb_feedback_events.id")
    stance: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text)
    import_batch_id: Mapped[Optional[str]] = _fk("kb_import_batches.id")
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbDecision(Base):
    __tablename__ = "kb_decisions"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_kb_decision_idempotency"),
        CheckConstraint(check_sql("mode", t.DECISION_MODES), name="ck_kb_decision_mode"),
        CheckConstraint(check_sql("requested_by", t.DECISION_REQUESTERS), name="ck_kb_decision_requester"),
        _pg_check("consulted_sha256 ~ '^[0-9a-f]{64}$'", name="ck_kb_decision_sha256"),
    )
    id: Mapped[str] = _id()
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(32), nullable=False)
    project_id: Mapped[Optional[str]] = mapped_column(ID)
    context: Mapped[dict] = mapped_column(Json, nullable=False)
    advisor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieval_params: Mapped[dict] = mapped_column(Json, nullable=False, default=dict)
    consulted_rule_version_ids: Mapped[list] = mapped_column(Json, nullable=False, default=list)
    consulted_case_ids: Mapped[list] = mapped_column(Json, nullable=False, default=list)
    consulted_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    recommendation: Mapped[dict] = mapped_column(Json, nullable=False)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbDecisionOutcome(Base):
    __tablename__ = "kb_decision_outcomes"
    __table_args__ = (Index("ix_kb_decision_outcomes_decision", "decision_id"),)
    id: Mapped[str] = _id()
    decision_id: Mapped[str] = _fk("kb_decisions.id", nullable=False)
    chosen: Mapped[dict] = mapped_column(Json, nullable=False)
    overrode: Mapped[bool] = mapped_column(Boolean, nullable=False)
    override_note: Mapped[Optional[str]] = mapped_column(Text)
    attempt_id: Mapped[Optional[str]] = _fk("kb_attempts.id")
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbImportMember(Base):
    __tablename__ = "kb_import_members"
    __table_args__ = (UniqueConstraint("target_kind", "target_id", "batch_id", name="uq_kb_import_member"),
                      CheckConstraint(check_sql("target_kind", t.IMPORT_MEMBER_KINDS),
                                      name="ck_kb_import_member_kind"),
                      Index("ix_kb_import_members_batch", "batch_id"))
    id: Mapped[str] = _id()
    batch_id: Mapped[str] = _fk("kb_import_batches.id", nullable=False)
    target_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(128), nullable=False)
    seq: Mapped[int] = _seq()


class KbInvalidation(Base):
    __tablename__ = "kb_invalidations"
    __table_args__ = (CheckConstraint(check_sql("target_kind", t.INVALIDATION_TARGETS),
                                      name="ck_kb_invalidation_target"),
                      UniqueConstraint("target_kind", "target_id", name="uq_kb_invalidation_target"))
    id: Mapped[str] = _id()
    target_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    actor_label: Mapped[Optional[str]] = mapped_column(String(128))
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbImportCursor(Base):
    """Importer bookkeeping — the only mutable ``kb_`` table; not evidence."""
    __tablename__ = "kb_import_cursors"
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    cursor: Mapped[Optional[dict]] = mapped_column(Json)
    lease_owner: Mapped[Optional[str]] = mapped_column(String(128))
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now,
                                                 server_default=func.now())


class KbCaptureQueue(Base):
    __tablename__ = "kb_capture_queue"
    __table_args__ = (UniqueConstraint("capture_key", name="uq_kb_capture_key"),
                      CheckConstraint(check_sql("reason", t.VERSION_REASONS), name="ck_kb_capture_reason"),
                      Index("ix_kb_capture_queue_run", "run_id", "c1_seq"))
    id: Mapped[str] = _id()
    capture_key: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(ID, nullable=False)
    artifact_id: Mapped[str] = _fk("kb_artifacts.id", nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    source_record: Mapped[str] = mapped_column(String(200), nullable=False)
    # "dev:ino:mtime_ns" of the published file when captured; C2 re-checks it
    # against the live final under the Run lock before making it current.
    file_identity: Mapped[str] = mapped_column(String(80), nullable=False)
    c1_seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False, unique=True)
    created_at: Mapped[datetime] = _created()


class KbCaptureDone(Base):
    __tablename__ = "kb_capture_done"
    __table_args__ = (UniqueConstraint("queue_id", name="uq_kb_capture_done_queue"),
                      CheckConstraint(check_sql("outcome", t.CAPTURE_OUTCOMES), name="ck_kb_capture_outcome"),
                      CheckConstraint("(outcome IN ('appended', 'historical')) = (case_version_no IS NOT NULL)",
                                      name="ck_kb_capture_done_version"))
    id: Mapped[str] = _id()
    queue_id: Mapped[str] = _fk("kb_capture_queue.id", nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    case_version_no: Mapped[Optional[int]] = mapped_column(Integer)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbEvalManifest(Base):
    __tablename__ = "kb_eval_manifests"
    __table_args__ = (_pg_check("sha256 ~ '^[0-9a-f]{64}$'", name="ck_kb_eval_manifest_sha256"),)
    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    body: Mapped[dict] = mapped_column(Json, nullable=False)
    created_at: Mapped[datetime] = _created()


class KbEvalMembership(Base):
    __tablename__ = "kb_eval_membership"
    __table_args__ = (UniqueConstraint("case_id", "manifest_version", name="uq_kb_eval_membership"),
                      CheckConstraint(check_sql("split", t.EVAL_SPLITS), name="ck_kb_eval_split"))
    id: Mapped[str] = _id()
    case_id: Mapped[str] = _fk("kb_cases.id", nullable=False)
    manifest_version: Mapped[str] = mapped_column(String(32), ForeignKey("kb_eval_manifests.version",
                                                  ondelete="RESTRICT"), nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    group_key: Mapped[str] = mapped_column(String(128), nullable=False)
    import_batch_id: Mapped[Optional[str]] = _fk("kb_import_batches.id")
    seq: Mapped[int] = _seq()


class KbSourceSubject(Base):
    __tablename__ = "kb_source_subjects"
    __table_args__ = (UniqueConstraint("label", name="uq_kb_subject_label"),)
    id: Mapped[str] = _id()
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbSubjectLink(Base):
    __tablename__ = "kb_subject_links"
    __table_args__ = (CheckConstraint("time_start_sec IS NULL OR time_end_sec IS NULL OR "
                                      "(time_start_sec >= 0 AND time_end_sec > time_start_sec)",
                                      name="ck_kb_subject_link_range"),
                      Index("ix_kb_subject_links_project", "project_id"))
    id: Mapped[str] = _id()
    subject_id: Mapped[str] = _fk("kb_source_subjects.id", nullable=False)
    project_id: Mapped[str] = mapped_column(ID, nullable=False)
    segment_def_id: Mapped[Optional[str]] = mapped_column(ID)
    time_start_sec: Mapped[Optional[float]] = mapped_column(Float)
    time_end_sec: Mapped[Optional[float]] = mapped_column(Float)
    seq: Mapped[int] = _seq()
    created_at: Mapped[datetime] = _created()


class KbCaptureScope(Base):
    """The operator-approved capture scope, as provenance (AICP-299 §5, S13a).

    One row per distinct explicit project list; every capture-made row carries
    the list's sha256, so it can be traced to the decision that allowed it.
    """
    __tablename__ = "kb_capture_scopes"
    __table_args__ = (_pg_check("scope_sha256 ~ '^[0-9a-f]{64}$'", name="ck_kb_capture_scope_sha256"),)
    scope_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_ids: Mapped[list] = mapped_column(Json, nullable=False)
    first_seen_at: Mapped[datetime] = _created()
    recorded_by: Mapped[Optional[str]] = mapped_column(String(128))
    seq: Mapped[int] = _seq()


class KbLinkFailure(Base):
    """A publication link (point i) that could not be made (AICP-299 §3.2, S15).

    The identity is the published file's ``dev:ino:mtime_ns``; all three are
    NULL together when even ``os.stat`` failed (paired by run, path and time).
    """
    __tablename__ = "kb_link_failures"
    __table_args__ = (UniqueConstraint("publication_id", name="uq_kb_link_failure_publication"),
                      CheckConstraint(check_sql("kind", t.LINK_KINDS), name="ck_kb_link_failure_kind"),
                      CheckConstraint("(st_dev IS NULL) = (st_ino IS NULL) AND (st_ino IS NULL) = (st_mtime_ns IS NULL)",
                                      name="ck_kb_link_failure_identity"),
                      Index("ix_kb_link_failures_run", "run_id", "failed_at_ns"),
                      Index("ix_kb_link_failures_identity", "st_dev", "st_ino", "st_mtime_ns"))
    id: Mapped[str] = _id()
    publication_id: Mapped[str] = mapped_column(String(200), nullable=False)
    run_id: Mapped[str] = mapped_column(ID, nullable=False)
    project_id: Mapped[Optional[str]] = mapped_column(ID)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    st_dev: Mapped[Optional[int]] = mapped_column(BigInteger)
    st_ino: Mapped[Optional[int]] = mapped_column(BigInteger)
    st_mtime_ns: Mapped[Optional[int]] = mapped_column(BigInteger)
    failed_at_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    error_class: Mapped[Optional[str]] = mapped_column(String(64))
    recorded_at: Mapped[datetime] = _created()
    seq: Mapped[int] = _seq()


INSERT_ONLY_MODELS = (
    KbImportBatch, KbImportBatchObservation, KbArtifact, KbArtifactCheck, KbCase, KbCaseVersion,
    KbAttempt, KbAttemptSubmission, KbAttemptObservation, KbRequest, KbFeedbackEvent, KbRule,
    KbRuleVersion, KbRuleReview, KbRuleEvidence, KbDecision, KbDecisionOutcome, KbImportMember,
    KbInvalidation, KbCaptureQueue, KbCaptureDone, KbEvalManifest, KbEvalMembership,
    KbSourceSubject, KbSubjectLink, KbCaptureScope, KbLinkFailure,
)
INSERT_ONLY_TABLES = tuple(m.__tablename__ for m in INSERT_ONLY_MODELS)
ALL_TABLES = INSERT_ONLY_TABLES + (KbImportCursor.__tablename__,)

IMMUTABLE_MESSAGE = "immutable knowledge evidence"


def pg_guard_sql(table: str) -> list[str]:
    """Row + TRUNCATE guard; the migration carries a literal copy."""
    return [
        f"CREATE FUNCTION {table}_reject_change() RETURNS trigger AS $$ BEGIN "
        f"RAISE EXCEPTION USING MESSAGE = '{IMMUTABLE_MESSAGE}: ' || TG_TABLE_NAME; END; $$ LANGUAGE plpgsql",
        f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION {table}_reject_change()",
        f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {table}_reject_change()",
    ]


def _reject(mapper, connection, target):
    raise ValueError(IMMUTABLE_MESSAGE)


for _model in INSERT_ONLY_MODELS:
    event.listen(_model, "before_update", _reject)
    event.listen(_model, "before_delete", _reject)
    _name = _model.__tablename__
    for _sql in pg_guard_sql(_name):
        event.listen(_model.__table__, "after_create", DDL(_sql).execute_if(dialect="postgresql"))
    event.listen(_model.__table__, "after_drop",
                 DDL(f"DROP FUNCTION IF EXISTS {_name}_reject_change()").execute_if(dialect="postgresql"))
    event.listen(_model.__table__, "after_create", DDL(
        f"CREATE TRIGGER {_name}_no_update BEFORE UPDATE ON {_name} "
        f"BEGIN SELECT RAISE(ABORT, '{IMMUTABLE_MESSAGE}'); END").execute_if(dialect="sqlite"))
    event.listen(_model.__table__, "after_create", DDL(
        f"CREATE TRIGGER {_name}_no_delete BEFORE DELETE ON {_name} "
        f"BEGIN SELECT RAISE(ABORT, '{IMMUTABLE_MESSAGE}'); END").execute_if(dialect="sqlite"))
