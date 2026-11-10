"""Durable planning turns, independent from accepted edit jobs.

Revision ID: e6c1d8f9a403
Revises: d5b0f3a8c702
"""
from alembic import op
import sqlalchemy as sa

revision = "e6c1d8f9a403"
down_revision = "d5b0f3a8c702"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "assistant_turns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("assistant_conversations.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("base_version_id", sa.String(36), nullable=False),
        sa.Column("operator_message_id", sa.String(36), sa.ForeignKey("assistant_messages.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("assistant_message_id", sa.String(36), sa.ForeignKey("assistant_messages.id", ondelete="RESTRICT")),
        sa.Column("parent_turn_id", sa.String(36), sa.ForeignKey("assistant_turns.id", ondelete="RESTRICT")),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("context_snapshot", sa.JSON(), nullable=False),
        sa.Column("reply", sa.JSON()),
        sa.Column("edit_job_id", sa.String(36), sa.ForeignKey("assistant_jobs.id", ondelete="RESTRICT")),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("lease_owner", sa.String(128)),
        sa.Column("lease_token", sa.String(36)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("lease_generation", sa.Integer(), nullable=False),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("conversation_id", "idempotency_key", name="uq_assistant_turn_request"),
        sa.UniqueConstraint("operator_message_id", name="uq_assistant_turn_operator_message"),
        sa.ForeignKeyConstraint(["conversation_id", "base_version_id"], ["assistant_versions.conversation_id", "assistant_versions.id"], ondelete="RESTRICT", name="fk_assistant_turn_base_scope"),
        sa.CheckConstraint("status IN ('queued','running','cancel_requested','succeeded','failed','cancelled')", name="ck_assistant_turn_status"),
        sa.CheckConstraint("lease_generation >= 0", name="ck_assistant_turn_generation"),
        sa.CheckConstraint("((status IN ('running','cancel_requested')) AND lease_owner IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL) OR ((status NOT IN ('running','cancel_requested')) AND lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)", name="ck_assistant_turn_lease"),
    )
    op.create_index("ix_assistant_turns_queue", "assistant_turns", ["status", "created_at"])
    op.create_index("ix_assistant_turns_recovery", "assistant_turns", ["status", "lease_expires_at"])
    op.create_table(
        "assistant_turn_edits",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("turn_id", sa.String(36), sa.ForeignKey("assistant_turns.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("assistant_jobs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("execution_message_id", sa.String(36), sa.ForeignKey("assistant_messages.id", ondelete="RESTRICT")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("turn_id", "idempotency_key", name="uq_assistant_turn_edit_request"),
        sa.UniqueConstraint("job_id", name="uq_assistant_turn_edit_job"),
    )


    protected = ("id", "conversation_id", "base_version_id", "operator_message_id", "parent_turn_id",
        "idempotency_key", "request_sha256", "actor", "context_snapshot", "created_at")
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        changed = " OR ".join(f"NEW.{name} IS NOT OLD.{name}" for name in protected)
        changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS NOT OLD.status OR NEW.reply IS NOT OLD.reply OR NEW.assistant_message_id IS NOT OLD.assistant_message_id OR NEW.finished_at IS NOT OLD.finished_at OR NEW.error_code IS NOT OLD.error_code))"
        op.execute("CREATE TRIGGER assistant_turns_immutable BEFORE UPDATE ON assistant_turns "
            f"WHEN {changed} BEGIN SELECT RAISE(ABORT, 'immutable planning turn'); END")
    elif dialect == "postgresql":
        changed = " OR ".join(f"to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name})" for name in protected)
        changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS DISTINCT FROM OLD.status OR NEW.reply::jsonb IS DISTINCT FROM OLD.reply::jsonb OR NEW.assistant_message_id IS DISTINCT FROM OLD.assistant_message_id OR NEW.finished_at IS DISTINCT FROM OLD.finished_at OR NEW.error_code IS DISTINCT FROM OLD.error_code))"
        op.execute("CREATE OR REPLACE FUNCTION assistant_turns_reject_update() RETURNS trigger AS $$ "
            f"BEGIN IF {changed} THEN RAISE EXCEPTION 'immutable planning turn'; END IF; RETURN NEW; END; "
            "$$ LANGUAGE plpgsql")
        op.execute("CREATE TRIGGER assistant_turns_immutable BEFORE UPDATE ON assistant_turns "
            "FOR EACH ROW EXECUTE FUNCTION assistant_turns_reject_update()")


def downgrade():
    op.drop_table("assistant_turn_edits")
    op.drop_index("ix_assistant_turns_recovery", table_name="assistant_turns")
    op.drop_index("ix_assistant_turns_queue", table_name="assistant_turns")
    op.drop_table("assistant_turns")
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS assistant_turns_reject_update()")
