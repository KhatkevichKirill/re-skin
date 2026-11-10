"""add_manual_creatives

Creatives produced outside the pipelines by an operator script get a row of
their own — history + a Final Creatives Job ID (AICP-239). One new table,
nothing existing is touched, no backfill.

Safe to apply before the code deploys (old code never reads the table) and
safe to leave applied on a rollback.

Revision ID: b8d0f2a4c6e9
Revises: a7c9e1b3d5f8
Create Date: 2026-10-17 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = 'b8d0f2a4c6e9'
down_revision: Union[str, None] = 'a7c9e1b3d5f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "manual_creatives",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("pipeline_type", sa.String(length=64), nullable=False),
        sa.Column("idea", sa.String(length=40), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=True),
        sa.Column("source_link", sa.Text(), nullable=True),
        sa.Column("gdrive_file_id", sa.String(length=255), nullable=False),
        sa.Column("overlay_job_id", sa.String(length=36), nullable=True),
        sa.Column("overlay_asset_id", sa.String(length=36), nullable=True),
        sa.Column("overlay_asset_name", sa.String(length=255), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("linear_issue", sa.String(length=32), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("final_creatives_exported_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("gdrive_file_id"),
        sa.UniqueConstraint("overlay_job_id"),
    )
    op.create_index("ix_manual_creatives_created_at", "manual_creatives", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_manual_creatives_created_at", table_name="manual_creatives")
    op.drop_table("manual_creatives")
