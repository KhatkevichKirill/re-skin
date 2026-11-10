"""Reframe workflow (AICP-251, docs/reframe-design.md §4).

Additive only:

- ``project_type_enum`` gains the label ``'reframe'`` (the a4c8e2b6d139
  precedent: ``ADD VALUE IF NOT EXISTS``, a no-op on SQLite). **This revision
  never uses the new label** — no backfill, default or CHECK names it — because
  PostgreSQL refuses to use an enum value added in the same transaction, and an
  ``upgrade head`` runs every pending revision in one.
- ``reframe_jobs`` (status / qa_status / aspect / resolution CHECKs and the
  ``uq_reframe_jobs_preparing_per_project`` partial unique index);
- ``reframe_renders``.

Downgrade drops the two tables and refuses while any ``reframe_jobs`` row has a
``provider_task_id``: that column is the consumer's link to a possibly billed
request. The enum label stays, as in every enum migration here.

This is a fixed copy of the schema in ``app/reframe_models.py``; it imports no
application code. tests/test_reframe.py compares the two.

Revision ID: n4d0f6b2c538
Revises: m3c9e5a1b427
Create Date: 2026-10-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "n4d0f6b2c538"
down_revision: Union[str, None] = "m3c9e5a1b427"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PREPARING_WHERE = "status = 'preparing'"


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE project_type_enum ADD VALUE IF NOT EXISTS 'reframe'")

    op.create_table(
        "reframe_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("target_aspect", sa.String(8), nullable=False),
        sa.Column("resolution", sa.String(8), nullable=False),
        sa.Column("window_start_sec", sa.Float(), nullable=False),
        sa.Column("window_end_sec", sa.Float(), nullable=False),
        sa.Column("input_start_sec", sa.Float(), nullable=True),
        sa.Column("input_frames", sa.Integer(), nullable=True),
        sa.Column("input_duration_sec", sa.Float(), nullable=True),
        sa.Column("input_path", sa.Text(), nullable=True),
        sa.Column("input_sha256", sa.String(64), nullable=True),
        sa.Column("input_bytes", sa.BigInteger(), nullable=True),
        sa.Column("audio_path", sa.Text(), nullable=True),
        sa.Column("audio_start_sec", sa.Float(), nullable=True),
        sa.Column("audio_duration_sec", sa.Float(), nullable=True),
        sa.Column("source_has_audio", sa.Boolean(), nullable=True),
        sa.Column("estimated_usd_base", sa.Float(), nullable=False),
        sa.Column("estimated_usd_reserved", sa.Float(), nullable=False),
        sa.Column("provider_task_id", sa.String(36), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("prep_claim_owner", sa.String(64), nullable=True),
        sa.Column("prep_claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("prep_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("qa_status", sa.String(16), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("job_qa", sa.JSON(), nullable=True),
        sa.Column("qa_claim_owner", sa.String(64), nullable=True),
        sa.Column("qa_claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('preparing', 'prepare_failed', 'admitted')",
                           name="ck_reframe_jobs_status"),
        sa.CheckConstraint("qa_status IN ('pending', 'running', 'passed', 'failed')",
                           name="ck_reframe_jobs_qa_status"),
        sa.CheckConstraint("target_aspect IN ('9:16', '16:9')", name="ck_reframe_jobs_aspect"),
        sa.CheckConstraint("resolution IN ('480p', '768p')", name="ck_reframe_jobs_resolution"),
    )
    op.create_index("ix_reframe_jobs_project_id", "reframe_jobs", ["project_id"])
    op.create_index("ix_reframe_jobs_provider_task_id", "reframe_jobs", ["provider_task_id"])
    op.create_index(
        "uq_reframe_jobs_preparing_per_project", "reframe_jobs", ["project_id"], unique=True,
        postgresql_where=sa.text(PREPARING_WHERE), sqlite_where=sa.text(PREPARING_WHERE),
    )

    op.create_table(
        "reframe_renders",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36),
                  sa.ForeignKey("reframe_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("claim_owner", sa.String(64), nullable=True),
        sa.Column("picture_end_sec", sa.Float(), nullable=False),
        sa.Column("fade_sec", sa.Float(), nullable=False),
        sa.Column("audio_end_sec", sa.Float(), nullable=True),
        sa.Column("black_tail_sec", sa.Float(), nullable=False),
        sa.Column("library_clip_id", sa.String(36), nullable=True),
        sa.Column("cta_path", sa.Text(), nullable=True),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("result_local_path", sa.Text(), nullable=True),
        sa.Column("result_bytes", sa.BigInteger(), nullable=True),
        sa.Column("result_sha256", sa.String(64), nullable=True),
        sa.Column("result_duration_sec", sa.Float(), nullable=True),
        sa.Column("result_width", sa.Integer(), nullable=True),
        sa.Column("result_height", sa.Integer(), nullable=True),
        sa.Column("qa", sa.JSON(), nullable=True),
        sa.Column("gdrive_file_id", sa.String(255), nullable=True),
        sa.Column("gdrive_link", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('queued', 'processing', 'delivering', 'done', 'failed')",
                           name="ck_reframe_renders_status"),
    )
    op.create_index("ix_reframe_renders_job_id", "reframe_renders", ["job_id"])


def downgrade() -> None:
    bind = op.get_bind()
    linked = bind.execute(
        sa.text("SELECT count(*) FROM reframe_jobs WHERE provider_task_id IS NOT NULL")
    ).scalar()
    if linked:
        raise RuntimeError(
            f"refusing to drop reframe_jobs: {linked} job(s) are linked to a provider "
            "task (a possibly billed request); export or settle them first"
        )
    op.drop_index("ix_reframe_renders_job_id", table_name="reframe_renders")
    op.drop_table("reframe_renders")
    op.drop_index("uq_reframe_jobs_preparing_per_project", table_name="reframe_jobs")
    op.drop_index("ix_reframe_jobs_provider_task_id", table_name="reframe_jobs")
    op.drop_index("ix_reframe_jobs_project_id", table_name="reframe_jobs")
    op.drop_table("reframe_jobs")
