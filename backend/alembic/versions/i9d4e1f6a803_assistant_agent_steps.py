"""Durable bounded agent workflow, additive to legacy planning turns.

Revision ID: i9d4e1f6a803
Revises: h8c3d0e5f792
"""
from alembic import op
import sqlalchemy as sa

revision = 'i9d4e1f6a803'
down_revision = 'h8c3d0e5f792'
branch_labels = None
depends_on = None


def _turn_guard():
    protected = ('id','conversation_id','base_version_id','operator_message_id','parent_turn_id',
        'idempotency_key','request_sha256','actor','context_snapshot','created_at','workflow_revision')
    dialect = op.get_bind().dialect.name
    if dialect == 'sqlite':
        op.execute('DROP TRIGGER IF EXISTS assistant_turns_immutable')
        changed = ' OR '.join(f'NEW.{name} IS NOT OLD.{name}' for name in protected)
        for name in ('broker_context_sha256','enrichment_snapshot','agent_context_snapshot'):
            changed += f' OR (OLD.{name} IS NOT NULL AND NEW.{name} IS NOT OLD.{name})'
        changed += " OR (OLD.agent_context_snapshot IS NULL AND NEW.agent_context_snapshot IS NOT NULL AND (OLD.status <> 'running' OR OLD.workflow_revision <> 2))"
        changed += " OR (OLD.enrichment_snapshot IS NULL AND NEW.enrichment_snapshot IS NOT NULL AND OLD.status <> 'running')"
        changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS NOT OLD.status OR NEW.reply IS NOT OLD.reply OR NEW.assistant_message_id IS NOT OLD.assistant_message_id OR NEW.finished_at IS NOT OLD.finished_at OR NEW.error_code IS NOT OLD.error_code))"
        op.execute(f"CREATE TRIGGER assistant_turns_immutable BEFORE UPDATE ON assistant_turns WHEN {changed} BEGIN SELECT RAISE(ABORT, 'immutable planning turn'); END")
    elif dialect == 'postgresql':
        changed = ' OR '.join(f'to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name})' for name in protected)
        for name in ('broker_context_sha256','enrichment_snapshot','agent_context_snapshot'):
            changed += f' OR (OLD.{name} IS NOT NULL AND to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name}))'
        changed += " OR (OLD.agent_context_snapshot IS NULL AND NEW.agent_context_snapshot IS NOT NULL AND (OLD.status <> 'running' OR OLD.workflow_revision <> 2))"
        changed += " OR (OLD.enrichment_snapshot IS NULL AND NEW.enrichment_snapshot IS NOT NULL AND OLD.status <> 'running')"
        changed += " OR (OLD.status IN ('succeeded','failed','cancelled') AND (NEW.status IS DISTINCT FROM OLD.status OR NEW.reply::jsonb IS DISTINCT FROM OLD.reply::jsonb OR NEW.assistant_message_id IS DISTINCT FROM OLD.assistant_message_id OR NEW.finished_at IS DISTINCT FROM OLD.finished_at OR NEW.error_code IS DISTINCT FROM OLD.error_code))"
        op.execute(f"CREATE OR REPLACE FUNCTION assistant_turns_reject_update() RETURNS trigger AS $$ BEGIN IF {changed} THEN RAISE EXCEPTION 'immutable planning turn'; END IF; RETURN NEW; END; $$ LANGUAGE plpgsql")


def _step_guard():
    pinned = ('id','turn_id','step_index','lease_generation','context_sha256','context_snapshot','created_at')
    once = ('action','observation','private_artifact','completed_at')
    dialect = op.get_bind().dialect.name
    if dialect == 'sqlite':
        changed = ' OR '.join(f'NEW.{name} IS NOT OLD.{name}' for name in pinned)
        changed += ' OR ' + ' OR '.join(f'(OLD.{name} IS NOT NULL AND NEW.{name} IS NOT OLD.{name})' for name in once)
        changed += " OR (OLD.status = 'completed' AND NEW.status IS NOT OLD.status)"
        op.execute(f"CREATE TRIGGER assistant_agent_steps_immutable BEFORE UPDATE ON assistant_agent_steps WHEN {changed} BEGIN SELECT RAISE(ABORT, 'immutable agent evidence'); END")
        op.execute("CREATE TRIGGER assistant_agent_steps_no_delete BEFORE DELETE ON assistant_agent_steps BEGIN SELECT RAISE(ABORT, 'immutable agent evidence'); END")
    elif dialect == 'postgresql':
        changed = ' OR '.join(f'to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name})' for name in pinned)
        changed += ' OR ' + ' OR '.join(f'(OLD.{name} IS NOT NULL AND to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name}))' for name in once)
        changed += " OR (OLD.status = 'completed' AND NEW.status IS DISTINCT FROM OLD.status)"
        op.execute(f"CREATE FUNCTION assistant_agent_steps_reject_update() RETURNS trigger AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'immutable agent evidence'; END IF; IF {changed} THEN RAISE EXCEPTION 'immutable agent evidence'; END IF; RETURN NEW; END; $$ LANGUAGE plpgsql")
        op.execute('CREATE TRIGGER assistant_agent_steps_immutable BEFORE UPDATE OR DELETE ON assistant_agent_steps FOR EACH ROW EXECUTE FUNCTION assistant_agent_steps_reject_update()')


def upgrade():
    op.add_column('assistant_turns', sa.Column('workflow_revision', sa.Integer(), nullable=False, server_default='1'))
    op.add_column('assistant_turns', sa.Column('agent_context_snapshot', sa.JSON(none_as_null=True), nullable=True))
    op.create_table('assistant_agent_steps',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('turn_id', sa.String(36), sa.ForeignKey('assistant_turns.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('step_index', sa.Integer(), nullable=False),
        sa.Column('lease_generation', sa.Integer(), nullable=False),
        sa.Column('context_sha256', sa.String(64), nullable=False),
        sa.Column('context_snapshot', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('action', sa.JSON(none_as_null=True)),
        sa.Column('observation', sa.JSON(none_as_null=True)),
        sa.Column('private_artifact', sa.JSON(none_as_null=True)),
        sa.Column('cleanup_pending', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('cleanup_next_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.UniqueConstraint('turn_id','step_index',name='uq_assistant_agent_step_index'),
        sa.CheckConstraint('step_index >= 0 AND step_index < 12', name='ck_assistant_agent_step_index'),
        sa.CheckConstraint("status IN ('submitted','acting','completed')", name='ck_assistant_agent_step_status'))
    op.create_index('ix_assistant_agent_step_cleanup','assistant_agent_steps',['cleanup_pending','cleanup_next_at'])
    _turn_guard()
    _step_guard()


def downgrade():
    if op.get_bind().scalar(sa.text('SELECT COUNT(*) FROM assistant_agent_steps')) or op.get_bind().scalar(sa.text('SELECT COUNT(*) FROM assistant_turns WHERE workflow_revision <> 1 OR agent_context_snapshot IS NOT NULL')):
        raise RuntimeError('refusing to discard durable agent evidence')
    op.drop_table('assistant_agent_steps')
    if op.get_bind().dialect.name == 'postgresql':
        op.execute('DROP FUNCTION IF EXISTS assistant_agent_steps_reject_update()')
    # Restore the preceding migration guard before dropping its new fields.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('prior_turn_guard', Path(__file__).with_name('h8c3d0e5f792_assistant_turn_enrichment.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._guards(enrichment=True)
    op.drop_column('assistant_turns','agent_context_snapshot')
    op.drop_column('assistant_turns','workflow_revision')
