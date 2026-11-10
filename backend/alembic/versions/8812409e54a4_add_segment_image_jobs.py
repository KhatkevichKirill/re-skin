"""add_segment_image_jobs

Per-segment GPT Image 2 regeneration: an operator can regenerate the
generated reference image for ONE swap segment before re-running it, without
touching the run-level reference set or the segment's hand override.

Creates `run_segment_image_jobs` — the append-only attempt history (one row
per "Regenerate image" click) — and adds two nullable columns to
`run_segments`:
  - generated_reference_urls  JSON — [character photo, GPT Image 2 output]
    when a job's result is applied; NULL otherwise (normal reference
    priority applies — see pipeline_v2.submit_run).
  - generated_image_job_id    VARCHAR(36) — which job's result is currently
    applied, for the UI's "current attempt" badge.

Purely additive: no backfill, no change to any existing column or enum.

Revision ID: 8812409e54a4
Revises: c9f2a4e6b807
Create Date: 2026-08-29 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8812409e54a4'
down_revision: Union[str, None] = 'c9f2a4e6b807'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'run_segment_image_jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('run_segment_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False, server_default=''),
        sa.Column('input_paths', sa.JSON(), nullable=True),
        sa.Column('frame_sec', sa.Float(), nullable=False),
        sa.Column('kie_task_id', sa.String(length=255), nullable=True),
        sa.Column(
            'status',
            sa.Enum('queued', 'generating', 'done', 'failed', name='segment_image_status_enum'),
            nullable=False,
            server_default='queued',
        ),
        sa.Column('output_path', sa.Text(), nullable=True),
        sa.Column('cost_credits', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['run_segment_id'], ['run_segments.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_run_segment_image_jobs_run_id'),
        'run_segment_image_jobs', ['run_id'], unique=False,
    )
    op.create_index(
        op.f('ix_run_segment_image_jobs_run_segment_id'),
        'run_segment_image_jobs', ['run_segment_id'], unique=False,
    )
    op.create_index(
        op.f('ix_run_segment_image_jobs_created_at'),
        'run_segment_image_jobs', ['created_at'], unique=False,
    )

    op.add_column(
        'run_segments',
        sa.Column('generated_reference_urls', sa.JSON(), nullable=True),
    )
    op.add_column(
        'run_segments',
        sa.Column('generated_image_job_id', sa.String(length=36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('run_segments', 'generated_image_job_id')
    op.drop_column('run_segments', 'generated_reference_urls')

    op.drop_index(op.f('ix_run_segment_image_jobs_created_at'), table_name='run_segment_image_jobs')
    op.drop_index(op.f('ix_run_segment_image_jobs_run_segment_id'), table_name='run_segment_image_jobs')
    op.drop_index(op.f('ix_run_segment_image_jobs_run_id'), table_name='run_segment_image_jobs')
    op.drop_table('run_segment_image_jobs')

    # Postgres leaves the enum type behind after drop_table; remove it so a
    # re-upgrade can recreate it cleanly. No-op on SQLite.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name='segment_image_status_enum').drop(bind, checkfirst=True)
