"""create_run_tail_edits

Creates the `run_tail_edits` table for the "Change ending" tool: an operator
takes a done Run's stitched final video, picks a cut time, and optionally a
clip from the stitch library (`library_clips`); a background worker produces
final.mp4[0..cut_sec] + library clip appended, delivered to Google Drive. Pure
ffmpeg, no AI generation. See app/pipeline_tail.py.

The tail edit status lifecycle (tail_edit_status_enum, created implicitly by
the sa.Enum column on Postgres; VARCHAR on SQLite):
  queued -> processing -> delivering -> done
  queued/processing/delivering -> failed
  failed/done -> queued (retry/re-run)

Revision ID: 19338f2b4ebc
Revises: d4a1b6c8e390
Create Date: 2026-07-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '19338f2b4ebc'
down_revision: Union[str, None] = 'd4a1b6c8e390'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'run_tail_edits',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('cut_sec', sa.Float(), nullable=False),
        sa.Column('library_clip_id', sa.String(length=36), nullable=True),
        sa.Column(
            'status',
            sa.Enum('queued', 'processing', 'delivering', 'done', 'failed', name='tail_edit_status_enum'),
            nullable=False,
            server_default='queued',
        ),
        sa.Column('output_path', sa.Text(), nullable=True),
        sa.Column('gdrive_file_id', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_run_tail_edits_run_id'),
        'run_tail_edits', ['run_id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_run_tail_edits_run_id'), table_name='run_tail_edits')
    op.drop_table('run_tail_edits')

    # Postgres leaves the enum type behind after drop_table; remove it so a
    # re-upgrade can recreate it cleanly. No-op on SQLite.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name='tail_edit_status_enum').drop(bind, checkfirst=True)
