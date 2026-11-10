"""create_pip_insert_tables

Creates the `insert_extractions` and `insert_merges` tables for the PIP
(picture-in-picture) insert extraction and reinsertion feature. The extracted
insert becomes a normal child VideoProject that flows through the existing
analyze → Run → stitch pipeline. The merge composites the processed child Run
result back onto a target video at specified coordinates and timing window,
with geometric rounded-corner masking.

The merge status lifecycle (merge_status_enum, created implicitly by the
sa.Enum column on Postgres; VARCHAR on SQLite):
  queued → compositing → delivering → done
  queued/compositing/delivering → failed
  failed/done → queued (retry/re-run)

Revision ID: 1819cbb7dcd9
Revises: a1d2e3f4b5c6
Create Date: 2026-07-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1819cbb7dcd9'
down_revision: Union[str, None] = 'a1d2e3f4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create insert_extractions table
    op.create_table(
        'insert_extractions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('parent_project_id', sa.String(length=36), nullable=False),
        sa.Column('child_project_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('x', sa.Integer(), nullable=False),
        sa.Column('y', sa.Integer(), nullable=False),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('start_sec', sa.Float(), nullable=False),
        sa.Column('end_sec', sa.Float(), nullable=False),
        sa.Column('corner_radius_px', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['parent_project_id'], ['video_projects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['child_project_id'], ['video_projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('child_project_id', name='uq_insert_extraction_child'),
    )
    op.create_index(
        op.f('ix_insert_extractions_parent_project_id'),
        'insert_extractions', ['parent_project_id'], unique=False,
    )

    # Create insert_merges table
    op.create_table(
        'insert_merges',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('target_project_id', sa.String(length=36), nullable=False),
        sa.Column('child_run_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('x', sa.Integer(), nullable=False),
        sa.Column('y', sa.Integer(), nullable=False),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('start_sec', sa.Float(), nullable=False),
        sa.Column('end_sec', sa.Float(), nullable=False),
        sa.Column('corner_radius_px', sa.Integer(), nullable=False, server_default='0'),
        sa.Column(
            'status',
            sa.Enum('queued', 'compositing', 'delivering', 'done', 'failed', name='merge_status_enum'),
            nullable=False,
            server_default='queued',
        ),
        sa.Column('output_path', sa.Text(), nullable=True),
        sa.Column('gdrive_url', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['target_project_id'], ['video_projects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['child_run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_insert_merges_target_project_id'),
        'insert_merges', ['target_project_id'], unique=False,
    )
    op.create_index(
        op.f('ix_insert_merges_child_run_id'),
        'insert_merges', ['child_run_id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_insert_merges_child_run_id'), table_name='insert_merges')
    op.drop_index(op.f('ix_insert_merges_target_project_id'), table_name='insert_merges')
    op.drop_table('insert_merges')
    op.drop_index(op.f('ix_insert_extractions_parent_project_id'), table_name='insert_extractions')
    op.drop_table('insert_extractions')

    # Postgres leaves the enum types behind after drop_table; remove them so a
    # re-upgrade can recreate them cleanly. No-op on SQLite.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name='merge_status_enum').drop(bind, checkfirst=True)
