"""Project-from-step: which step of a run a project was started from (AICP-331).

Additive only. ``video_projects`` gets two nullable soft references next to
the existing ``source_run_id``:

* ``source_kind`` — ``caption`` | ``tail_edit`` | ``overlay`` when "Move to
  new project" copied that derived creative's output instead of the run's
  final video. NULL = the run's final (every project created before this
  revision, and the run path, which is unchanged) or no run at all.
* ``source_artifact_id`` — the ``run_captions`` / ``run_tail_edits`` /
  ``overlay_jobs`` id. No FK: captions and tail edits are hard-deleted by
  their operator Delete buttons (and cascade with their run), and the project
  owns its own copy of the video, so a dangling id only degrades the project
  page's banner to the run backlink.

``source_run_id`` keeps naming the OWNING run for every kind, so the
backlink, the projects-list Source cell, insert-merge lineage and the caption
placement lineage walk read exactly what they read before. No enum change, no
backfill.

Revision ID: w4c8e2a6d1f3
Revises: w2c6e8a0b4d7
Create Date: 2026-11-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "w4c8e2a6d1f3"
down_revision: Union[str, None] = "w2c6e8a0b4d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CHECK_NAME = "ck_video_projects_source_kind"
# Both or neither: a kind always names its artifact, and the run final keeps
# the one legacy representation (both NULL).
CHECK_SQL = (
    "(source_kind IS NULL AND source_artifact_id IS NULL) OR "
    "(source_kind IN ('caption', 'tail_edit', 'overlay') AND source_artifact_id IS NOT NULL)"
)


def upgrade() -> None:
    # batch_alter_table, like o5e1a7c3d649 / u5e1b7c9d3f4: on PostgreSQL it is
    # plain ALTER TABLEs; on SQLite (which cannot ADD CONSTRAINT) it rebuilds
    # the table so the CHECK lands too.
    with op.batch_alter_table("video_projects") as batch:
        batch.add_column(sa.Column("source_kind", sa.String(16), nullable=True))
        batch.add_column(sa.Column("source_artifact_id", sa.String(36), nullable=True))
        batch.create_check_constraint(CHECK_NAME, CHECK_SQL)


def downgrade() -> None:
    # Projects keep their source_run_id backlink; only the step detail is lost.
    with op.batch_alter_table("video_projects") as batch:
        batch.drop_constraint(CHECK_NAME, type_="check")
        batch.drop_column("source_artifact_id")
        batch.drop_column("source_kind")
