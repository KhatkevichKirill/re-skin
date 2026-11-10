"""create_run_automations

Creates `run_automations` (per-Run automation settings: sheet filter rule,
plan, daily quota) and `automation_launches` (ledger of consumed sheet rows →
spawned runs, grouped into batches for the completion webhook).

`plan` is a plain VARCHAR (validated in the API) so we don't grow another
Postgres enum type. Works on both SQLite and Postgres.

Revision ID: a1d2e3f4b5c6
Revises: b4f6d8e0c357
Create Date: 2026-07-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1d2e3f4b5c6'
down_revision: Union[str, None] = 'b4f6d8e0c357'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'run_automations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('filter_persona', sa.String(length=255), nullable=True),
        sa.Column('filter_age', sa.String(length=64), nullable=True),
        sa.Column('filter_gender', sa.String(length=32), nullable=True),
        sa.Column('filter_ethnicity', sa.String(length=64), nullable=True),
        sa.Column('plan', sa.String(length=16), nullable=False, server_default='close-up'),
        sa.Column('daily_count', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('webhook_url', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_run_automations_run_id'), 'run_automations', ['run_id'], unique=True
    )

    op.create_table(
        'automation_launches',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('automation_id', sa.String(length=36), nullable=False),
        sa.Column('batch_id', sa.String(length=36), nullable=False),
        sa.Column('queue_id', sa.String(length=64), nullable=False),
        sa.Column('ref_url', sa.Text(), nullable=False),
        sa.Column('new_run_id', sa.String(length=36), nullable=True),
        sa.Column('launched_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('notified_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['automation_id'], ['run_automations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['new_run_id'], ['runs.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('automation_id', 'queue_id', name='uq_automation_launch_queue'),
    )
    op.create_index(
        op.f('ix_automation_launches_automation_id'),
        'automation_launches', ['automation_id'], unique=False,
    )
    op.create_index(
        op.f('ix_automation_launches_batch_id'),
        'automation_launches', ['batch_id'], unique=False,
    )
    op.create_index(
        op.f('ix_automation_launches_new_run_id'),
        'automation_launches', ['new_run_id'], unique=False,
    )
    op.create_index(
        op.f('ix_automation_launches_launched_at'),
        'automation_launches', ['launched_at'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_automation_launches_launched_at'), table_name='automation_launches')
    op.drop_index(op.f('ix_automation_launches_new_run_id'), table_name='automation_launches')
    op.drop_index(op.f('ix_automation_launches_batch_id'), table_name='automation_launches')
    op.drop_index(op.f('ix_automation_launches_automation_id'), table_name='automation_launches')
    op.drop_table('automation_launches')
    op.drop_index(op.f('ix_run_automations_run_id'), table_name='run_automations')
    op.drop_table('run_automations')
