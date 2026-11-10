"""add_caption_placement

Where generated (burned) captions sit (AICP-368, ``app.caption_placement``):
placed over the ORIGINAL caption area of a subtitle-removal lineage so the
VOID / Omni patches left where the old captions were stay covered.

Two new columns, additive and nullable with no default:
  - video_projects.caption_placement  JSON NULL  -- the project's placement
    (derived at project-from-run, inherited, or set by the operator)
  - run_captions.placement            JSON NULL  -- the placement a caption
    job burns with, frozen when the job is created

NULL means today's house style (bottom-centred, MarginV 16%), so every
existing project and caption burns exactly as before and nothing needs a
backfill. Existing derived projects get a placement only through the explicit
``POST /api/v2/projects/{pid}/caption-placement/derive`` (or the CLI).

Downgrade drops both columns.

Revision ID: s3c8e1a6d204
Revises: s3b5d7f9a1c2
Create Date: 2026-10-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "s3c8e1a6d204"
down_revision: Union[str, None] = "s3b5d7f9a1c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "video_projects",
        sa.Column("caption_placement", sa.JSON(), nullable=True),
    )
    op.add_column(
        "run_captions",
        sa.Column("placement", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run_captions", "placement")
    op.drop_column("video_projects", "caption_placement")
