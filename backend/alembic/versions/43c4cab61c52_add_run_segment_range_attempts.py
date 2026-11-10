"""add_run_segment_range_attempts

Run-local segment range regeneration: an operator can enlarge/shrink a
completed swap segment's time range through an adjacent `keep` boundary and
regenerate just that segment, as a non-destructive candidate revision that
only replaces the Run's active timeline/result once generation, stitch and
delivery all succeed.

Creates `run_segment_range_attempts` — the append-only attempt history (one
row per "Regenerate with new range" click), mirroring
`run_segment_image_jobs`'s shape (added in 8812409e54a4) but for a full
video generation + re-stitch + re-deliver instead of a single image call.

Purely additive: no backfill, no change to any existing table, column or
enum.

Revision ID: 43c4cab61c52
Revises: 8812409e54a4
Create Date: 2026-08-29 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '43c4cab61c52'
down_revision: Union[str, None] = '8812409e54a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'run_segment_range_attempts',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('run_segment_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('requested_start_sec', sa.Float(), nullable=False),
        sa.Column('requested_end_sec', sa.Float(), nullable=False),
        sa.Column('prompt_override', sa.Text(), nullable=True),
        sa.Column('reference_image_urls_override', sa.JSON(), nullable=True),
        sa.Column('base_segment_snapshot', sa.JSON(), nullable=False),
        sa.Column('candidate_segment_snapshot', sa.JSON(), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'queued', 'generating', 'stitching', 'delivering', 'done', 'failed',
                name='range_attempt_status_enum',
            ),
            nullable=False,
            server_default='queued',
        ),
        sa.Column('seedance_task_id', sa.String(length=255), nullable=True),
        sa.Column('seedance_result_url', sa.Text(), nullable=True),
        sa.Column('cost_credits', sa.Float(), nullable=True),
        sa.Column('local_clip_path', sa.Text(), nullable=True),
        sa.Column('local_result_path', sa.Text(), nullable=True),
        sa.Column('keep_recut_paths', sa.JSON(), nullable=True),
        sa.Column('final_local_path', sa.Text(), nullable=True),
        sa.Column('result_duration_sec', sa.Float(), nullable=True),
        sa.Column('source_fallback_segments', sa.Integer(), nullable=True),
        sa.Column('gdrive_file_id', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('promoted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['run_segment_id'], ['run_segments.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_run_segment_range_attempts_run_id'),
        'run_segment_range_attempts', ['run_id'], unique=False,
    )
    op.create_index(
        op.f('ix_run_segment_range_attempts_run_segment_id'),
        'run_segment_range_attempts', ['run_segment_id'], unique=False,
    )
    op.create_index(
        op.f('ix_run_segment_range_attempts_created_at'),
        'run_segment_range_attempts', ['created_at'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_run_segment_range_attempts_created_at'), table_name='run_segment_range_attempts')
    op.drop_index(op.f('ix_run_segment_range_attempts_run_segment_id'), table_name='run_segment_range_attempts')
    op.drop_index(op.f('ix_run_segment_range_attempts_run_id'), table_name='run_segment_range_attempts')
    op.drop_table('run_segment_range_attempts')

    # Postgres leaves the enum type behind after drop_table; remove it so a
    # re-upgrade can recreate it cleanly. No-op on SQLite.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name='range_attempt_status_enum').drop(bind, checkfirst=True)
