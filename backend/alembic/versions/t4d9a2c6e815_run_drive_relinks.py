"""Run Drive re-links (AICP-348).

Additive only: the ``run_drive_relinks`` table — one row per operator re-link
of a run whose delivered Drive file was deleted to a NEW Drive file (the
durable operation and its audit trail), with a partial unique index allowing
one ``queued`` row per run. No enum change, no backfill.

Revision ID: t4d9a2c6e815
Revises: s3c8e1a6d204
Create Date: 2026-11-02
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "t4d9a2c6e815"
down_revision: Union[str, None] = "s3c8e1a6d204"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "run_drive_relinks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36),
                  sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("old_file_id", sa.String(255), nullable=False),
        sa.Column("target_file_id", sa.String(255), nullable=False),
        sa.Column("new_file_id", sa.String(255), nullable=True),
        sa.Column("result_path", sa.Text(), nullable=False),
        sa.Column("run_status", sa.String(32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("exported_rows", sa.JSON(), nullable=True),
        sa.Column("sheet_scan", sa.JSON(), nullable=True),
    )
    op.create_index("ix_run_drive_relinks_run_id", "run_drive_relinks", ["run_id"])
    op.create_index(
        "uq_run_drive_relinks_active_per_run", "run_drive_relinks", ["run_id"],
        unique=True, postgresql_where=sa.text("status = 'queued'"),
        sqlite_where=sa.text("status = 'queued'"),
    )


def downgrade() -> None:
    # The new Drive files stay on Drive; only the operation history is dropped.
    op.drop_index("uq_run_drive_relinks_active_per_run", table_name="run_drive_relinks")
    op.drop_index("ix_run_drive_relinks_run_id", table_name="run_drive_relinks")
    op.drop_table("run_drive_relinks")
