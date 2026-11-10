"""Write-once optional speech evidence, before planning inference.

Revision ID: h8c3d0e5f792
Revises: f7a2b9c4d681
"""
from alembic import op
import sqlalchemy as sa

revision = "h8c3d0e5f792"
down_revision = "f7a2b9c4d681"
branch_labels = None
depends_on = None


def _guards(*, enrichment):
    protected = ("id", "conversation_id", "base_version_id", "operator_message_id", "parent_turn_id",
        "idempotency_key", "request_sha256", "actor", "context_snapshot", "created_at")
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS assistant_turns_immutable")
        changed = " OR ".join(f"NEW.{name} IS NOT OLD.{name}" for name in protected)
        changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS NOT OLD.status OR NEW.reply IS NOT OLD.reply OR NEW.assistant_message_id IS NOT OLD.assistant_message_id OR NEW.finished_at IS NOT OLD.finished_at OR NEW.error_code IS NOT OLD.error_code))"
        if enrichment:
            changed += " OR (OLD.broker_context_sha256 IS NOT NULL AND NEW.broker_context_sha256 IS NOT OLD.broker_context_sha256)"
            changed += " OR (OLD.enrichment_snapshot IS NOT NULL AND NEW.enrichment_snapshot IS NOT OLD.enrichment_snapshot) OR (OLD.enrichment_snapshot IS NULL AND NEW.enrichment_snapshot IS NOT NULL AND OLD.status <> 'running')"
        op.execute("CREATE TRIGGER assistant_turns_immutable BEFORE UPDATE ON assistant_turns "
            f"WHEN {changed} BEGIN SELECT RAISE(ABORT, 'immutable planning turn'); END")
    elif dialect == "postgresql":
        changed = " OR ".join(f"to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name})" for name in protected)
        changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS DISTINCT FROM OLD.status OR NEW.reply::jsonb IS DISTINCT FROM OLD.reply::jsonb OR NEW.assistant_message_id IS DISTINCT FROM OLD.assistant_message_id OR NEW.finished_at IS DISTINCT FROM OLD.finished_at OR NEW.error_code IS DISTINCT FROM OLD.error_code))"
        if enrichment:
            changed += " OR (OLD.broker_context_sha256 IS NOT NULL AND NEW.broker_context_sha256 IS DISTINCT FROM OLD.broker_context_sha256)"
            changed += " OR (OLD.enrichment_snapshot IS NOT NULL AND NEW.enrichment_snapshot::jsonb IS DISTINCT FROM OLD.enrichment_snapshot::jsonb) OR (OLD.enrichment_snapshot IS NULL AND NEW.enrichment_snapshot IS NOT NULL AND OLD.status <> 'running')"
        op.execute("CREATE OR REPLACE FUNCTION assistant_turns_reject_update() RETURNS trigger AS $$ "
            f"BEGIN IF {changed} THEN RAISE EXCEPTION 'immutable planning turn'; END IF; RETURN NEW; END; "
            "$$ LANGUAGE plpgsql")


def upgrade():
    op.add_column("assistant_turns", sa.Column("enrichment_snapshot", sa.JSON(none_as_null=True), nullable=True))
    op.add_column("assistant_turns", sa.Column("broker_context_sha256", sa.String(64), nullable=True))
    op.add_column("assistant_turns", sa.Column("broker_cleanup_pending", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("assistant_turns", sa.Column("broker_cleanup_next_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_assistant_turns_broker_cleanup", "assistant_turns", ["broker_cleanup_pending", "broker_cleanup_next_at"])
    _guards(enrichment=True)


def downgrade():
    connection = op.get_bind()
    if connection.scalar(sa.text("SELECT COUNT(*) FROM assistant_turns WHERE enrichment_snapshot IS NOT NULL OR broker_context_sha256 IS NOT NULL OR broker_cleanup_pending")):
        raise RuntimeError("refusing to discard persisted planning evidence or broker cleanup identity")
    _guards(enrichment=False)
    op.drop_index("ix_assistant_turns_broker_cleanup", table_name="assistant_turns")
    op.drop_column("assistant_turns", "broker_cleanup_next_at")
    op.drop_column("assistant_turns", "broker_cleanup_pending")
    op.drop_column("assistant_turns", "broker_context_sha256")
    op.drop_column("assistant_turns", "enrichment_snapshot")
