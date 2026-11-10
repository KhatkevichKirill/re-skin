"""Localisation scope: hook or whole video (AICP-340).

Additive only:

- ``video_projects.localisation_scope`` VARCHAR(16) NOT NULL DEFAULT 'hook'
  with CHECK ``ck_video_projects_localisation_scope`` (``'hook'``/``'whole'``)
  — the scope that cut the current segments. The server default backfills
  every existing row to ``'hook'``, which is exactly what they do today, so no
  data is rewritten and no project changes behaviour until an operator
  re-analyzes it as ``whole``.
- ``video_projects.localisation_scope_requested`` VARCHAR(16) NULL with CHECK
  ``ck_video_projects_localisation_scope_requested`` — the scope the next
  analysis must apply (NULL = re-apply the current one).

Apply BEFORE deploying the code that reads the column (CLAUDE.md: manual
migration, predeploy backup first). The old code ignores the column, so
applying it early is safe; the new code against an unmigrated table fails on
every VideoProject read.

Downgrade drops both columns (and their CHECKs). A project analyzed as ``whole``
keeps its all-swap segments; only the label is lost, and the old code then
treats it as a hook project whose hook covers its swap segments.

Revision ID: o5e1a7c3d649
Revises: n4d0f6b2c538
Create Date: 2026-10-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "o5e1a7c3d649"
down_revision: Union[str, None] = "n4d0f6b2c538"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CHECK_NAME = "ck_video_projects_localisation_scope"
REQUESTED_CHECK_NAME = "ck_video_projects_localisation_scope_requested"


def upgrade() -> None:
    with op.batch_alter_table("video_projects") as batch:
        batch.add_column(
            sa.Column(
                "localisation_scope",
                sa.String(length=16),
                nullable=False,
                server_default="hook",
            )
        )
        batch.create_check_constraint(
            CHECK_NAME, "localisation_scope IN ('hook', 'whole')"
        )
        batch.add_column(
            sa.Column("localisation_scope_requested", sa.String(length=16), nullable=True)
        )
        batch.create_check_constraint(
            REQUESTED_CHECK_NAME,
            "localisation_scope_requested IS NULL "
            "OR localisation_scope_requested IN ('hook', 'whole')",
        )


def downgrade() -> None:
    with op.batch_alter_table("video_projects") as batch:
        batch.drop_constraint(REQUESTED_CHECK_NAME, type_="check")
        batch.drop_column("localisation_scope_requested")
        batch.drop_constraint(CHECK_NAME, type_="check")
        batch.drop_column("localisation_scope")
