"""Recoverable explicit assistant publication and immutable receipts.

Revision ID: j0e5f2a7b914
Revises: i9d4e1f6a803
"""
from alembic import op
import sqlalchemy as sa

revision = "j0e5f2a7b914"
down_revision = "i9d4e1f6a803"
branch_labels = None
depends_on = None


def _guard(name, columns, once=()):
    dialect = op.get_bind().dialect.name
    protected = [c for c in columns if c not in once]
    if dialect == "sqlite":
        changed = " OR ".join(f"NEW.{c} IS NOT OLD.{c}" for c in protected)
        for c in once:
            changed += f" OR (OLD.{c} IS NOT NULL AND NEW.{c} IS NOT OLD.{c})"
        if once:
            changed += " OR (OLD.completed_at IS NOT NULL AND NEW.resumable_uri IS NOT OLD.resumable_uri)"
        op.execute(f"CREATE TRIGGER {name}_immutable BEFORE UPDATE ON {name} WHEN {changed} BEGIN SELECT RAISE(ABORT, 'immutable assistant publication evidence'); END")
        op.execute(f"CREATE TRIGGER {name}_no_delete BEFORE DELETE ON {name} BEGIN SELECT RAISE(ABORT, 'immutable assistant publication evidence'); END")
    elif dialect == "postgresql":
        changed = " OR ".join(f"to_jsonb(NEW.{c}) IS DISTINCT FROM to_jsonb(OLD.{c})" for c in protected)
        for c in once:
            changed += f" OR (OLD.{c} IS NOT NULL AND to_jsonb(NEW.{c}) IS DISTINCT FROM to_jsonb(OLD.{c}))"
        if once:
            changed += " OR (OLD.completed_at IS NOT NULL AND to_jsonb(NEW) IS DISTINCT FROM to_jsonb(OLD))"
        op.execute(f"CREATE FUNCTION {name}_reject_update() RETURNS trigger AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'immutable assistant publication evidence'; END IF; IF {changed} THEN RAISE EXCEPTION 'immutable assistant publication evidence'; END IF; RETURN NEW; END; $$ LANGUAGE plpgsql")
        op.execute(f"CREATE TRIGGER {name}_immutable BEFORE UPDATE OR DELETE ON {name} FOR EACH ROW EXECUTE FUNCTION {name}_reject_update()")


def upgrade():
    op.create_table("assistant_run_result_intents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("artifact_id", sa.String(36), sa.ForeignKey("assistant_artifacts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("before_json", sa.JSON(), nullable=False),
        sa.Column("after_json", sa.JSON(), nullable=False),
        sa.Column("resumable_uri", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime()),
        sa.UniqueConstraint("conversation_id", "idempotency_key", name="uq_assistant_run_result_intent_request"),
        sa.UniqueConstraint("id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "actor", "request_sha256", name="uq_assistant_run_result_intent_binding"),
        sa.ForeignKeyConstraint(["conversation_id", "version_id"], ["assistant_versions.conversation_id", "assistant_versions.id"],
            name="fk_assistant_run_result_intent_version_scope", ondelete="RESTRICT"))
    op.create_index("uq_assistant_run_result_intent_unresolved", "assistant_run_result_intents", ["run_id"], unique=True,
        postgresql_where=sa.text("completed_at IS NULL"), sqlite_where=sa.text("completed_at IS NULL"))
    op.create_table("assistant_run_result_promotions",
        sa.Column("id", sa.String(36), sa.ForeignKey("assistant_run_result_intents.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), sa.ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("artifact_id", sa.String(36), sa.ForeignKey("assistant_artifacts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("before_json", sa.JSON(), nullable=False),
        sa.Column("after_json", sa.JSON(), nullable=False),
        sa.Column("delivery", sa.String(16), nullable=False),
        sa.Column("noop", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("delivery IN ('drive_updated','local_only')", name="ck_assistant_run_result_delivery"),
        sa.ForeignKeyConstraint(["id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "actor", "request_sha256"],
            ["assistant_run_result_intents." + column for column in ("id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "actor", "request_sha256")],
            name="fk_assistant_run_result_receipt_intent_binding", ondelete="RESTRICT"))
    op.create_index("ix_assistant_run_result_promotions_run", "assistant_run_result_promotions", ["run_id", "created_at"])
    _guard("assistant_run_result_intents", ("id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "request_sha256", "actor", "request_json", "before_json", "after_json", "resumable_uri", "created_at", "completed_at"), ("resumable_uri", "completed_at"))
    _guard("assistant_run_result_promotions", ("id", "conversation_id", "run_id", "version_id", "artifact_id", "idempotency_key", "request_sha256", "actor", "before_json", "after_json", "delivery", "noop", "created_at"))


def downgrade():
    for name in ("assistant_run_result_promotions", "assistant_run_result_intents"):
        if op.get_bind().scalar(sa.text(f"SELECT COUNT(*) FROM {name}")):
            raise RuntimeError("refusing to discard durable main-result publication evidence")
    for name in ("assistant_run_result_promotions", "assistant_run_result_intents"):
        op.drop_table(name)
        if op.get_bind().dialect.name == "postgresql":
            op.execute(f"DROP FUNCTION IF EXISTS {name}_reject_update()")
