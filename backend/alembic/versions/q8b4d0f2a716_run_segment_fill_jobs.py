"""Segment caption fill jobs (AICP-358).

Additive only: the ``run_segment_fill_jobs`` table (status and model as
VARCHAR + CHECK, no Postgres enum) and its partial unique index — at most one
active (``admitted``/``compositing``) fill per run. No backfill: no fill was
ever made through the platform before this revision.

Revision ID: q8b4d0f2a716
Revises: o5e2a8c4d613
Create Date: 2026-10-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "q8b4d0f2a716"
down_revision: Union[str, None] = "o5e2a8c4d613"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ACTIVE = "status IN ('admitted','compositing')"


def upgrade() -> None:
    op.create_table(
        "run_segment_fill_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36),
                  sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_segment_id", sa.String(36),
                  sa.ForeignKey("run_segments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("model", sa.String(16), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("mask_box", sa.JSON(), nullable=True),
        sa.Column("box_source", sa.String(16), nullable=True),
        sa.Column("base", sa.JSON(), nullable=False),
        sa.Column("media", sa.JSON(), nullable=True),
        sa.Column("prep_dir", sa.Text(), nullable=True),
        sa.Column("preview_path", sa.Text(), nullable=True),
        sa.Column("estimated_usd", sa.Float(), nullable=True),
        sa.Column("approved_usd", sa.Float(), nullable=True),
        sa.Column("discard_audio_patch", sa.Boolean(), server_default=sa.text("false"),
                  nullable=False),
        sa.Column("provider_task_id", sa.String(36), nullable=True),
        sa.Column("claim_token", sa.String(64), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("composite_path", sa.Text(), nullable=True),
        sa.Column("replaced_take_id", sa.String(36), nullable=True),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        sa.Column("files_cleaned_at", sa.DateTime(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('preparing','ready','admitted','compositing','applied',"
            "'failed','canceled')", name="ck_run_segment_fill_jobs_status"),
        sa.CheckConstraint("model IN ('void')", name="ck_run_segment_fill_jobs_model"),
    )
    op.create_index("ix_run_segment_fill_jobs_run_id", "run_segment_fill_jobs", ["run_id"])
    op.create_index("ix_run_segment_fill_jobs_run_segment_id", "run_segment_fill_jobs",
                    ["run_segment_id"])
    op.create_index("ix_run_segment_fill_jobs_status", "run_segment_fill_jobs", ["status"])
    op.create_index("ix_run_segment_fill_jobs_provider_task_id", "run_segment_fill_jobs",
                    ["provider_task_id"])
    op.create_index("uq_run_segment_fill_jobs_active_per_run", "run_segment_fill_jobs",
                    ["run_id"], unique=True, postgresql_where=sa.text(_ACTIVE),
                    sqlite_where=sa.text(_ACTIVE))


def downgrade() -> None:
    # Dropping the rows leaves runs/<rid>/segment_fills/ on disk, unreferenced.
    op.drop_index("uq_run_segment_fill_jobs_active_per_run", table_name="run_segment_fill_jobs")
    op.drop_index("ix_run_segment_fill_jobs_provider_task_id", table_name="run_segment_fill_jobs")
    op.drop_index("ix_run_segment_fill_jobs_status", table_name="run_segment_fill_jobs")
    op.drop_index("ix_run_segment_fill_jobs_run_segment_id", table_name="run_segment_fill_jobs")
    op.drop_index("ix_run_segment_fill_jobs_run_id", table_name="run_segment_fill_jobs")
    op.drop_table("run_segment_fill_jobs")
