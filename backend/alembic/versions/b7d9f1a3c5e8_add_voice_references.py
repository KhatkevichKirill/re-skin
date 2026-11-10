"""add_voice_references

Two additions for Seedance 2.5 voice references (`reference_audio_urls`), the
control that stops the model inventing a fresh voice on every generation:

1. `voice_clips` — a global reusable voice library, mirroring `overlay_assets`
   rather than `library_clips`: attaching a voice COPIES the file into the run,
   so no run ever holds a soft reference to a row here and no downstream
   column/guard is needed. See models.VoiceClip's docstring.
2. `runs.reference_audio_urls` — the run's own copy of that audio, resolved to
   public URLs at submit exactly like `reference_image_urls`.

Nullable with no backfill: every existing run keeps its behaviour (no key in
the payload at all), and the pipeline treats NULL and [] identically.

Revision ID: b7d9f1a3c5e8
Revises: a3f5b7c9d201
Create Date: 2026-08-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7d9f1a3c5e8'
down_revision: Union[str, None] = 'a3f5b7c9d201'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'voice_clips',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column(
            'created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False
        ),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('local_path', sa.Text(), nullable=False),
        sa.Column('language', sa.String(length=8), nullable=True),
        sa.Column('duration_sec', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    # Matches LibraryClip's index: the library is always listed newest-first.
    op.create_index(
        op.f('ix_voice_clips_created_at'), 'voice_clips', ['created_at'], unique=False
    )
    op.add_column('runs', sa.Column('reference_audio_urls', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('runs', 'reference_audio_urls')
    op.drop_index(op.f('ix_voice_clips_created_at'), table_name='voice_clips')
    op.drop_table('voice_clips')
