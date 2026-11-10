"""add_created_at_indexes

Adds indexes to the created_at columns of tables used for listing endpoints
that ORDER BY created_at DESC. This prevents full-table scans as history grows.

Indexes are created on:
  - video_projects.created_at
  - library_clips.created_at
  - overlay_assets.created_at
  - overlay_jobs.created_at
  - insert_merges.created_at
  - run_tail_edits.created_at

Revision ID: 08433b4098d9
Revises: 9230657199c5
Create Date: 2026-07-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '08433b4098d9'
down_revision: Union[str, None] = '9230657199c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(op.f("ix_video_projects_created_at"), "video_projects", ["created_at"])
    op.create_index(op.f("ix_library_clips_created_at"), "library_clips", ["created_at"])
    op.create_index(op.f("ix_overlay_assets_created_at"), "overlay_assets", ["created_at"])
    op.create_index(op.f("ix_overlay_jobs_created_at"), "overlay_jobs", ["created_at"])
    op.create_index(op.f("ix_insert_merges_created_at"), "insert_merges", ["created_at"])
    op.create_index(op.f("ix_run_tail_edits_created_at"), "run_tail_edits", ["created_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_run_tail_edits_created_at"), table_name="run_tail_edits")
    op.drop_index(op.f("ix_insert_merges_created_at"), table_name="insert_merges")
    op.drop_index(op.f("ix_overlay_jobs_created_at"), table_name="overlay_jobs")
    op.drop_index(op.f("ix_overlay_assets_created_at"), table_name="overlay_assets")
    op.drop_index(op.f("ix_library_clips_created_at"), table_name="library_clips")
    op.drop_index(op.f("ix_video_projects_created_at"), table_name="video_projects")
