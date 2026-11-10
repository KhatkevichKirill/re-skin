"""add_project_mute_source

Adds the "remove original audio" project setting.

When `video_projects.mute_source` is true, the swap clips uploaded to
Seedance are cut video-only (ffmpeg -an) so the model never hears the source
soundtrack and generates audio from the prompt instead. The source file and
the "keep" segment clips are untouched, so both Run.audio_mode options keep
working.

New column on `video_projects` (default-off, so existing projects behave
exactly as before):
  - mute_source  BOOLEAN NOT NULL DEFAULT FALSE

Revision ID: c5e7a9b1d3f4
Revises: b3c4d5e6f7a8
Create Date: 2026-08-07 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5e7a9b1d3f4'
down_revision: Union[str, None] = 'b3c4d5e6f7a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'video_projects',
        sa.Column(
            'mute_source',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column('video_projects', 'mute_source')
