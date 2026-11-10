"""add_project_caption_detector

Per-project config for the free QA caption detector (AICP-343,
``app.qa_tools.CaptionDetectorConfig``): caption band, text and outline
colour/style, glyph size, hit threshold.

New column on ``video_projects``, additive and nullable with no default:
  - caption_detector  JSON NULL

NULL means the detector's defaults (white text with a black outline in the
centre band -- exactly what AICP-332 used), so every existing project behaves
as before and nothing needs a backfill. Only the QA tools read it; the
pipeline does not.

Downgrade drops the column (and with it any per-project config).

Revision ID: o5e2a8c4d613
Revises: p7a3c9e1f5b2
Create Date: 2026-10-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "o5e2a8c4d613"
down_revision: Union[str, None] = "p7a3c9e1f5b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "video_projects",
        sa.Column("caption_detector", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("video_projects", "caption_detector")
