"""Batch VOID fills (2026-11-03, docs/segment-fill-batch-design.md).

Additive, plus two recreated constraints of ``run_segment_fill_jobs``:

- table ``run_segment_fill_batches`` (status / authorization VARCHAR + CHECK,
  partial unique index: at most one ``active``/``promoting`` batch per run);
- ``run_segment_fill_jobs.batch_id`` (FK, ON DELETE SET NULL), ``mask_grow_px``,
  ``retry_of``; the status CHECK gains ``composited``; the partial unique index
  "one active fill per run" is recreated with ``AND batch_id IS NULL`` (it now
  covers only legacy single fills -- every existing row has ``batch_id`` NULL,
  so it accepts every row the old index accepted);
- ``video_projects.void_auto_usd_per_run`` FLOAT NULL (NULL = the setting);
- VOID from the source (§7): ``video_projects.caption_removal_method`` VARCHAR(16)
  NULL + CHECK (NULL reads as 'omni': every existing project keeps its
  behaviour), ``runs.generation_mode`` CHECK gains 'void_source',
  ``run_segments.model_fallback`` BOOLEAN NOT NULL DEFAULT false.

No backfill. Downgrade is REFUSED while any batch exists or any fill row is
``composited`` (it would drop the only record of a paid batch); otherwise it
restores the old constraints.

This is a fixed copy of ``app/segment_fill_models.py``; it imports no
application code.

Revision ID: u5e1b7c9d3f4
Revises: t4d9f2b7c315
Create Date: 2026-11-03
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "u5e1b7c9d3f4"
down_revision: Union[str, None] = "t4d9f2b7c315"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_ACTIVE = "status IN ('admitted','compositing')"
_NEW_ACTIVE = "status IN ('admitted','compositing') AND batch_id IS NULL"
_OLD_STATUS = ("status IN ('preparing','ready','admitted','compositing','applied',"
               "'failed','canceled')")
_NEW_STATUS = ("status IN ('preparing','ready','admitted','compositing','composited','applied',"
               "'failed','canceled')")
_METHOD_CHECK = ("caption_removal_method IS NULL "
                 "OR caption_removal_method IN ('void_source', 'omni')")
_OLD_GEN = "generation_mode IN ('v2v', 'images_only')"
_NEW_GEN = "generation_mode IN ('v2v', 'images_only', 'void_source')"
_BATCH_LIVE = "status IN ('active','promoting')"
_BATCH_STATUS = "status IN ('active','promoting','applied','failed')"


def upgrade() -> None:
    op.create_table(
        "run_segment_fill_batches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("basis", sa.String(16), nullable=False),
        sa.Column("authorization_kind", sa.String(16), nullable=False),
        sa.Column("approved_usd", sa.Float(), nullable=False),
        sa.Column("reserved_mcredits", sa.Integer(), nullable=False),
        sa.Column("reserved_usd", sa.Float(), nullable=False),
        sa.Column("discard_audio_patch", sa.Boolean(), server_default=sa.text("false"),
                  nullable=False),
        sa.Column("base", sa.JSON(), nullable=False),
        sa.Column("claim_token", sa.String(64), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(_BATCH_STATUS, name="ck_run_segment_fill_batches_status"),
        sa.CheckConstraint("authorization_kind IN ('operator','auto')",
                           name="ck_run_segment_fill_batches_authorization"),
    )
    op.create_index("ix_run_segment_fill_batches_run_id", "run_segment_fill_batches",
                    ["run_id"])
    op.create_index("ix_run_segment_fill_batches_status", "run_segment_fill_batches",
                    ["status"])
    op.create_index("uq_run_segment_fill_batches_live_per_run", "run_segment_fill_batches",
                    ["run_id"], unique=True, postgresql_where=sa.text(_BATCH_LIVE),
                    sqlite_where=sa.text(_BATCH_LIVE))

    with op.batch_alter_table("run_segment_fill_jobs") as b:
        b.add_column(sa.Column("batch_id", sa.String(36), nullable=True))
        b.add_column(sa.Column("mask_grow_px", sa.Integer(), nullable=True))
        b.add_column(sa.Column("retry_of", sa.String(36), nullable=True))
        b.create_foreign_key("fk_run_segment_fill_jobs_batch_id", "run_segment_fill_batches",
                             ["batch_id"], ["id"], ondelete="SET NULL")
        b.drop_constraint("ck_run_segment_fill_jobs_status", type_="check")
        b.create_check_constraint("ck_run_segment_fill_jobs_status", _NEW_STATUS)
    op.create_index("ix_run_segment_fill_jobs_batch_id", "run_segment_fill_jobs", ["batch_id"])
    op.drop_index("uq_run_segment_fill_jobs_active_per_run", table_name="run_segment_fill_jobs")
    op.create_index("uq_run_segment_fill_jobs_active_per_run", "run_segment_fill_jobs",
                    ["run_id"], unique=True, postgresql_where=sa.text(_NEW_ACTIVE),
                    sqlite_where=sa.text(_NEW_ACTIVE))

    op.add_column("video_projects", sa.Column("void_auto_usd_per_run", sa.Float(),
                                              nullable=True))
    with op.batch_alter_table("video_projects") as b:
        b.add_column(sa.Column("caption_removal_method", sa.String(16), nullable=True))
        b.create_check_constraint("ck_video_projects_caption_removal_method",
                                  _METHOD_CHECK)
    with op.batch_alter_table("runs") as b:
        b.drop_constraint("ck_runs_generation_mode", type_="check")
        b.create_check_constraint("ck_runs_generation_mode", _NEW_GEN)
    op.add_column("run_segments", sa.Column("model_fallback", sa.Boolean(),
                                            server_default=sa.text("false"), nullable=False))


def downgrade() -> None:
    bind = op.get_bind()
    batches = bind.execute(sa.text("SELECT count(*) FROM run_segment_fill_batches")).scalar()
    composited = bind.execute(sa.text(
        "SELECT count(*) FROM run_segment_fill_jobs WHERE status = 'composited'")).scalar()
    void_runs = bind.execute(sa.text(
        "SELECT count(*) FROM runs WHERE generation_mode = 'void_source'")).scalar()
    if batches or composited or void_runs:
        raise RuntimeError(
            f"refusing to downgrade: {batches} fill batch(es) / {composited} composited fill "
            f"row(s) / {void_runs} VOID-from-source run(s) exist (the only record of paid batch "
            "fills / of how those runs were made)")
    op.drop_column("run_segments", "model_fallback")
    with op.batch_alter_table("runs") as b:
        b.drop_constraint("ck_runs_generation_mode", type_="check")
        b.create_check_constraint("ck_runs_generation_mode", _OLD_GEN)
    with op.batch_alter_table("video_projects") as b:
        b.drop_constraint("ck_video_projects_caption_removal_method", type_="check")
        b.drop_column("caption_removal_method")
    op.drop_column("video_projects", "void_auto_usd_per_run")
    op.drop_index("uq_run_segment_fill_jobs_active_per_run", table_name="run_segment_fill_jobs")
    op.create_index("uq_run_segment_fill_jobs_active_per_run", "run_segment_fill_jobs",
                    ["run_id"], unique=True, postgresql_where=sa.text(_OLD_ACTIVE),
                    sqlite_where=sa.text(_OLD_ACTIVE))
    op.drop_index("ix_run_segment_fill_jobs_batch_id", table_name="run_segment_fill_jobs")
    with op.batch_alter_table("run_segment_fill_jobs") as b:
        b.drop_constraint("ck_run_segment_fill_jobs_status", type_="check")
        b.create_check_constraint("ck_run_segment_fill_jobs_status", _OLD_STATUS)
        b.drop_constraint("fk_run_segment_fill_jobs_batch_id", type_="foreignkey")
        b.drop_column("retry_of")
        b.drop_column("mask_grow_px")
        b.drop_column("batch_id")
    op.drop_index("uq_run_segment_fill_batches_live_per_run",
                  table_name="run_segment_fill_batches")
    op.drop_index("ix_run_segment_fill_batches_status", table_name="run_segment_fill_batches")
    op.drop_index("ix_run_segment_fill_batches_run_id", table_name="run_segment_fill_batches")
    op.drop_table("run_segment_fill_batches")
