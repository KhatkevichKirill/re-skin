"""Localisation autopilot flow: durable transcription / translate jobs + Run.mute_source (AICP-365).

Additive:

* ``runs.mute_source`` BOOLEAN NULL — "remove original audio" frozen at
  admission; NULL (every existing run, every manual run) = read the project's
  live setting at submit, i.e. today's behaviour.
* ``project_transcriptions`` — durable transcription jobs, one live per project
  (partial unique index).
* ``localisation_prompt_jobs`` — durable translate jobs of the autopilot.

No backfill. Downgrade refuses while a job is live (it would orphan an admitted,
possibly billed call).

Revision ID: s3b5d7f9a1c2
Revises: r1a7c3e5b9d2
Create Date: 2026-10-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "s3b5d7f9a1c2"
down_revision: Union[str, None] = "r1a7c3e5b9d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STATUSES = ("queued", "running", "succeeded", "failed", "interrupted")
_ORIGINS = ("manual", "analysis", "autopilot")
_LIVE = "status IN ('queued', 'running')"


def _in(col, values):
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def _dispatch_columns():
    return [
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("dispatch_gen", sa.Integer(), nullable=False),
        sa.Column("claim_token", sa.String(64), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("rq_job_id", sa.String(128), nullable=True),
        sa.Column("not_live_since", sa.DateTime(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    ]


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("mute_source", sa.Boolean(), nullable=True))

    op.create_table(
        "project_transcriptions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("origin", sa.String(16), nullable=False),
        *_dispatch_columns(),
        sa.CheckConstraint(_in("status", _STATUSES), name="ck_project_transcriptions_status"),
        sa.CheckConstraint(_in("origin", _ORIGINS), name="ck_project_transcriptions_origin"),
    )
    op.create_index("ix_project_transcriptions_project_id", "project_transcriptions",
                    ["project_id"])
    op.create_index("uq_project_transcriptions_live", "project_transcriptions", ["project_id"],
                    unique=True, postgresql_where=sa.text(_LIVE), sqlite_where=sa.text(_LIVE))

    op.create_table(
        "localisation_prompt_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_detail", sa.JSON(), nullable=True),
        *_dispatch_columns(),
        sa.CheckConstraint(_in("status", _STATUSES), name="ck_localisation_prompt_jobs_status"),
    )
    op.create_index("ix_localisation_prompt_jobs_project", "localisation_prompt_jobs",
                    ["project_id", "language"])


def downgrade() -> None:
    conn = op.get_bind()
    for table in ("project_transcriptions", "localisation_prompt_jobs"):
        live = conn.execute(sa.text(f"SELECT count(*) FROM {table} WHERE {_LIVE}")).scalar()
        if live:
            raise RuntimeError(
                f"refusing to downgrade: {live} live row(s) in {table} — wait for them to finish"
            )
    op.drop_index("ix_localisation_prompt_jobs_project", table_name="localisation_prompt_jobs")
    op.drop_table("localisation_prompt_jobs")
    op.drop_index("uq_project_transcriptions_live", table_name="project_transcriptions")
    op.drop_index("ix_project_transcriptions_project_id", table_name="project_transcriptions")
    op.drop_table("project_transcriptions")
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("mute_source")
