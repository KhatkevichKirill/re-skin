"""add_run_captions

Creates the `run_captions` table for the "Add Captions" tool: an operator
takes a done Run's delivered result, re-transcribes it, renders the spoken
lines into an ASS subtitle file, burns them in with ffmpeg, and delivers the
captioned copy to Google Drive as a new derived creative. Mirrors
run_tail_edits — "a derived creative built from a done Run's result,
delivered to Drive, exportable to the sheet." See docs/burned-captions.md.

The caption status lifecycle (caption_status_enum, created implicitly by the
sa.Enum column on Postgres; VARCHAR on SQLite):
  queued -> transcribing -> burning -> delivering -> done
  queued -> burning directly when lines_source == "manual" (operator-supplied
    lines skip the transcribe step)
  queued/transcribing/burning/delivering -> failed
  failed/done -> queued (retry/re-run)

Revision ID: b5d7f9a1c3e6
Revises: a4c8e2b6d139
Create Date: 2026-08-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b5d7f9a1c3e6'
down_revision: Union[str, None] = 'a4c8e2b6d139'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'run_captions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('language', sa.String(length=8), nullable=True),
        sa.Column('lines', sa.JSON(), nullable=True),
        sa.Column('lines_source', sa.String(length=16), nullable=True),
        sa.Column('style_version', sa.String(length=32), nullable=True),
        sa.Column(
            'status',
            sa.Enum('queued', 'transcribing', 'burning', 'delivering', 'done', 'failed', name='caption_status_enum'),
            nullable=False,
            server_default='queued',
        ),
        sa.Column('output_path', sa.Text(), nullable=True),
        sa.Column('gdrive_file_id', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('final_creatives_exported_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_run_captions_run_id'),
        'run_captions', ['run_id'], unique=False,
    )
    op.create_index(
        op.f('ix_run_captions_created_at'),
        'run_captions', ['created_at'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_run_captions_created_at'), table_name='run_captions')
    op.drop_index(op.f('ix_run_captions_run_id'), table_name='run_captions')
    op.drop_table('run_captions')

    # Postgres leaves the enum type behind after drop_table; remove it so a
    # re-upgrade can recreate it cleanly. No-op on SQLite.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name='caption_status_enum').drop(bind, checkfirst=True)
