"""add_caption_audio_fingerprint

Lets a caption job reuse another caption's transcript when the two videos
carry the same audio, instead of sending the same soundtrack to Gemini again.
Every face_swap run of a project under audio_mode="original" delivers a
bit-identical soundtrack, so captioning N runs used to cost N transcriptions.

  run_captions.audio_sha256      — media.audio_fingerprint of the source the
      job read; indexed, it is the lookup key for a donor caption.
  run_captions.lines_reused_from — soft ref to the donor run_captions.id when
      lines_source == 'reused' (no FK: the donor can be hard-deleted).

No backfill here: computing a fingerprint means decoding every captioned
video, which does not belong in a schema migration. Rows left NULL simply
never act as donors.

Revision ID: f1c3e5a7b902
Revises: e4b7c1a9d260
Create Date: 2026-09-28 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = 'f1c3e5a7b902'
down_revision: Union[str, None] = 'e4b7c1a9d260'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "run_captions",
        sa.Column("audio_sha256", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "run_captions",
        sa.Column("lines_reused_from", sa.String(length=36), nullable=True),
    )
    op.create_index(
        "ix_run_captions_audio_sha256", "run_captions", ["audio_sha256"]
    )


def downgrade() -> None:
    op.drop_index("ix_run_captions_audio_sha256", table_name="run_captions")
    op.drop_column("run_captions", "lines_reused_from")
    op.drop_column("run_captions", "audio_sha256")
