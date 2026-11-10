"""Immutable assistant history and retained artifacts (AICP-264).

Additive schema only; no legacy changes or media backfill. Apply explicitly after
the required backup. Old code can ignore these tables. Downgrade destroys assistant
history and must not be used after live versions exist without an approved export.

Revision ID: c4a9e2f7b601
Revises: b8d0f2a4c6e9
Create Date: 2026-10-23
"""

import sqlalchemy as sa
from alembic import op

revision = "c4a9e2f7b601"
down_revision = "b8d0f2a4c6e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'assistant_conversations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('project_id', sa.String(length=36), nullable=False),
        sa.Column('next_version_sequence', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('next_message_sequence', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint('next_version_sequence > 0 AND next_message_sequence > 0', name='ck_assistant_conversation_counters'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id', name='uq_assistant_conversation_run'),
    )
    op.create_index('ix_assistant_conversations_project_id', 'assistant_conversations', ['project_id'])
    op.create_table(
        'assistant_versions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('conversation_id', sa.String(length=36), nullable=False),
        sa.Column('sequence', sa.Integer(), nullable=False),
        sa.Column('parent_version_id', sa.String(length=36), nullable=True),
        sa.Column('base_version_id', sa.String(length=36), nullable=True),
        sa.Column('selection_epoch', sa.Integer(), nullable=False),
        sa.Column('idempotency_key', sa.String(length=128), nullable=False),
        sa.Column('request_sha256', sa.String(length=64), nullable=False),
        sa.Column('actor', sa.String(length=128), nullable=False),
        sa.Column('recipe', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.CheckConstraint('(parent_version_id IS NULL AND base_version_id IS NULL AND sequence = 1) OR (parent_version_id IS NOT NULL AND base_version_id IS NOT NULL AND parent_version_id = base_version_id AND sequence > 1 AND parent_version_id <> id)', name='ck_assistant_version_base_parent'),
        sa.CheckConstraint('sequence > 0 AND selection_epoch >= 0', name='ck_assistant_version_counters'),
        sa.CheckConstraint("status IN ('pending','succeeded','failed','cancelled')", name='ck_assistant_version_status'),
        sa.ForeignKeyConstraint(['conversation_id'], ['assistant_conversations.id'], name=None, ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(['conversation_id', 'parent_version_id'], ['assistant_versions.conversation_id', 'assistant_versions.id'], name='fk_assistant_version_parent_scope', ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('conversation_id', 'idempotency_key', name='uq_assistant_version_request'),
        sa.UniqueConstraint('conversation_id', 'id', name='uq_assistant_version_scope_id'),
        sa.UniqueConstraint('conversation_id', 'sequence', name='uq_assistant_version_sequence'),
    )
    op.create_index('ix_assistant_versions_parent_version_id', 'assistant_versions', ['parent_version_id'])
    op.create_table(
        'assistant_selections',
        sa.Column('conversation_id', sa.String(length=36), nullable=False),
        sa.Column('selected_version_id', sa.String(length=36), nullable=True),
        sa.Column('latest_version_id', sa.String(length=36), nullable=True),
        sa.Column('epoch', sa.Integer(), nullable=False, server_default='0'),
        sa.CheckConstraint('epoch >= 0', name='ck_assistant_selection_epoch'),
        sa.ForeignKeyConstraint(['conversation_id'], ['assistant_conversations.id'], name=None, ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(['conversation_id', 'latest_version_id'], ['assistant_versions.conversation_id', 'assistant_versions.id'], name='fk_assistant_selection_latest_scope', ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(['conversation_id', 'selected_version_id'], ['assistant_versions.conversation_id', 'assistant_versions.id'], name='fk_assistant_selection_selected_scope', ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint('conversation_id'),
    )
    op.create_table(
        'assistant_artifacts',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('owner_version_id', sa.String(length=36), nullable=False),
        sa.Column('storage_key', sa.String(length=96), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('media_kind', sa.String(length=8), nullable=False),
        sa.Column('metadata_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("media_kind IN ('video','audio','image')", name='ck_assistant_artifact_media'),
        sa.CheckConstraint('size_bytes > 0', name='ck_assistant_artifact_size'),
        sa.ForeignKeyConstraint(['owner_version_id'], ['assistant_versions.id'], name=None, ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_assistant_artifacts_owner_version_id', 'assistant_artifacts', ['owner_version_id'])
    op.create_index('ix_assistant_artifacts_sha256', 'assistant_artifacts', ['sha256'])
    op.create_table(
        'assistant_version_artifacts',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('version_id', sa.String(length=36), nullable=False),
        sa.Column('artifact_id', sa.String(length=36), nullable=False),
        sa.Column('role', sa.String(length=8), nullable=False),
        sa.Column('slot', sa.Integer(), nullable=False, server_default='0'),
        sa.CheckConstraint("role = 'input' OR slot = 0", name='ck_assistant_version_artifact_output_slot'),
        sa.CheckConstraint("role IN ('full','preview','input') AND slot >= 0", name='ck_assistant_version_artifact_role'),
        sa.ForeignKeyConstraint(['version_id'], ['assistant_versions.id'], name=None, ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(['artifact_id'], ['assistant_artifacts.id'], name=None, ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('version_id', 'role', 'slot', name='uq_assistant_version_artifact_slot'),
    )
    op.create_index('ix_assistant_version_artifacts_artifact_id', 'assistant_version_artifacts', ['artifact_id'])
    op.create_table(
        'assistant_messages',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('conversation_id', sa.String(length=36), nullable=False),
        sa.Column('sequence', sa.Integer(), nullable=False),
        sa.Column('base_version_id', sa.String(length=36), nullable=True),
        sa.Column('idempotency_key', sa.String(length=128), nullable=False),
        sa.Column('request_sha256', sa.String(length=64), nullable=False),
        sa.Column('actor', sa.String(length=128), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('context_snapshot', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("role IN ('operator','assistant','tool','system')", name='ck_assistant_message_role'),
        sa.CheckConstraint('sequence > 0', name='ck_assistant_message_sequence'),
        sa.ForeignKeyConstraint(['conversation_id'], ['assistant_conversations.id'], name=None, ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(['conversation_id', 'base_version_id'], ['assistant_versions.conversation_id', 'assistant_versions.id'], name='fk_assistant_message_base_scope', ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('conversation_id', 'idempotency_key', name='uq_assistant_message_request'),
        sa.UniqueConstraint('conversation_id', 'sequence', name='uq_assistant_message_sequence'),
    )
    _install_immutable_guards()


def _install_immutable_guards() -> None:
    protected = {
        'assistant_conversations': ['id', 'run_id', 'project_id', 'created_at'],
        'assistant_versions': ['id', 'conversation_id', 'sequence', 'parent_version_id', 'base_version_id', 'selection_epoch', 'idempotency_key', 'request_sha256', 'actor', 'recipe', 'created_at'],
        'assistant_artifacts': ['id', 'owner_version_id', 'storage_key', 'sha256', 'size_bytes', 'media_kind', 'metadata_json', 'created_at'],
        'assistant_version_artifacts': ['id', 'version_id', 'artifact_id', 'role', 'slot'],
        'assistant_messages': ['id', 'conversation_id', 'sequence', 'base_version_id', 'idempotency_key', 'request_sha256', 'actor', 'role', 'content', 'context_snapshot', 'created_at'],
    }
    dialect = op.get_bind().dialect.name
    for table, columns in protected.items():
        if dialect == "sqlite":
            changed = " OR ".join(f"NEW.{column} IS NOT OLD.{column}" for column in columns)
            if table == "assistant_versions":
                changed += " OR (OLD.status <> 'pending' AND (NEW.status IS NOT OLD.status OR NEW.completed_at IS NOT OLD.completed_at))"
            op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE ON {table} "
                       f"WHEN {changed} BEGIN SELECT RAISE(ABORT, 'immutable assistant record'); END")
        elif dialect == "postgresql":
            changed = "to_jsonb(NEW) IS DISTINCT FROM to_jsonb(OLD)"
            if table == "assistant_conversations":
                changed = ("(to_jsonb(NEW) - ARRAY['next_version_sequence','next_message_sequence']) IS DISTINCT FROM "
                           "(to_jsonb(OLD) - ARRAY['next_version_sequence','next_message_sequence'])")
            if table == "assistant_versions":
                changed = ("(to_jsonb(NEW) - ARRAY['status','completed_at']) IS DISTINCT FROM "
                           "(to_jsonb(OLD) - ARRAY['status','completed_at']) OR "
                           "(OLD.status <> 'pending' AND to_jsonb(NEW) IS DISTINCT FROM to_jsonb(OLD))")
            op.execute(f"CREATE OR REPLACE FUNCTION {table}_reject_update() RETURNS trigger AS $$ "
                       f"BEGIN IF {changed} THEN RAISE EXCEPTION 'immutable assistant record'; END IF; "
                       "RETURN NEW; END; $$ LANGUAGE plpgsql")
            op.execute(f"CREATE TRIGGER {table}_immutable BEFORE UPDATE ON {table} "
                       f"FOR EACH ROW EXECUTE FUNCTION {table}_reject_update()")
        else:
            raise RuntimeError("assistant schema supports PostgreSQL and SQLite only")


def downgrade() -> None:
    op.drop_table('assistant_messages')
    op.drop_index('ix_assistant_version_artifacts_artifact_id', table_name='assistant_version_artifacts')
    op.drop_table('assistant_version_artifacts')
    op.drop_index('ix_assistant_artifacts_owner_version_id', table_name='assistant_artifacts')
    op.drop_index('ix_assistant_artifacts_sha256', table_name='assistant_artifacts')
    op.drop_table('assistant_artifacts')
    op.drop_table('assistant_selections')
    op.drop_index('ix_assistant_versions_parent_version_id', table_name='assistant_versions')
    op.drop_table('assistant_versions')
    op.drop_index('ix_assistant_conversations_project_id', table_name='assistant_conversations')
    op.drop_table('assistant_conversations')
    if op.get_bind().dialect.name == "postgresql":
        for table in ("assistant_conversations", "assistant_versions", "assistant_artifacts",
                      "assistant_version_artifacts", "assistant_messages"):
            op.execute(f"DROP FUNCTION IF EXISTS {table}_reject_update()")
