"""Run segment takes (AICP-342).

Additive only: the ``run_segment_takes`` table — preserved previous outputs of
a swap RunSegment, written when a rerun or a take restore replaces a completed
result (``file_purged_at`` marks a take whose file the disk-cleanup retention
sweep deleted). No enum change, no backfill: runs rerun before this revision have no
preserved takes, which is the truth.

Revision ID: p7a3c9e1f5b2
Revises: o5e1a7c3d649
Create Date: 2026-10-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "p7a3c9e1f5b2"
down_revision: Union[str, None] = "o5e1a7c3d649"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "run_segment_takes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36),
                  sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_segment_id", sa.String(36),
                  sa.ForeignKey("run_segments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("segment_def_id", sa.String(36), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=True),
        sa.Column("end_sec", sa.Float(), nullable=True),
        sa.Column("result_path", sa.Text(), nullable=False),
        sa.Column("seedance_task_id", sa.String(255), nullable=True),
        sa.Column("seedance_result_url", sa.Text(), nullable=True),
        sa.Column("cost_credits", sa.Float(), nullable=True),
        sa.Column("prompt_override", sa.Text(), nullable=True),
        sa.Column("reference_image_urls_override", sa.JSON(), nullable=True),
        sa.Column("generated_reference_urls", sa.JSON(), nullable=True),
        sa.Column("generated_image_job_id", sa.String(36), nullable=True),
        sa.Column("file_purged_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_run_segment_takes_run_id", "run_segment_takes", ["run_id"])
    op.create_index("ix_run_segment_takes_run_segment_id", "run_segment_takes", ["run_segment_id"])


def downgrade() -> None:
    # Dropping the rows does not delete the take files on disk
    # (runs/<rid>/segment_takes/); they are only unreferenced afterwards.
    op.drop_index("ix_run_segment_takes_run_segment_id", table_name="run_segment_takes")
    op.drop_index("ix_run_segment_takes_run_id", table_name="run_segment_takes")
    op.drop_table("run_segment_takes")
