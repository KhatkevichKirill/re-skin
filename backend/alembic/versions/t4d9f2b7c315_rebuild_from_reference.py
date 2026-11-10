"""Rebuild from reference — project type, images-only runs, shots/storyboards/
keyframes/chunk attempts/LLM calls (docs/rebuild-from-reference-design.md §4).

Additive only:

- ``project_type_enum`` gains ``'rebuild'`` and ``run_model_enum`` gains
  ``'kling-4-0'`` (``ADD VALUE IF NOT EXISTS``, a no-op on SQLite). **This
  revision never uses either label** — no default, backfill or CHECK names them
  — because PostgreSQL refuses to use an enum value added in the same
  transaction, and ``upgrade head`` runs every pending revision in one. The
  labels are permanent; the names are final.
- ``video_projects.rebuild_analysis_token`` VARCHAR(36) NULL — the analysis
  that owns a rebuild project's ``analyzing`` status (stale-job fence);
- ``runs.generation_mode`` VARCHAR(16) NOT NULL DEFAULT 'v2v' + CHECK in
  ('v2v', 'images_only') — the server default backfills every existing run as
  v2v; ``runs.rebuild_snapshot`` JSON NULL.
- Tables ``rebuild_shots``, ``project_storyboards``, ``rebuild_keyframes``,
  ``rebuild_chunk_attempts``, ``rebuild_llm_calls`` with their partial unique
  indexes (WHERE texts byte-identical to ``app/rebuild_models.py``).

Downgrade is REFUSED while any rebuild project, any images-only run or any row
in the new tables exists (it would destroy the only record of how those runs
were made and of billed provider tasks); on an empty feature it drops the
tables/columns and leaves the enum labels.

This is a fixed copy of the schema in ``app/rebuild_models.py``; it imports no
application code. tests/test_rebuild_migration.py compares the two.

Revision ID: t4d9f2b7c315
Revises: t4d9a2c6e815
Create Date: 2026-11-03
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "t4d9f2b7c315"
down_revision: Union[str, None] = "t4d9a2c6e815"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _in(col, values):
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


PAID_CALL_STATUSES = (
    "queued", "claimed", "sending", "submitted", "fetching",
    "succeeded", "failed", "unknown",
)
PAID_CALL_LIVE_STATUSES = ("queued", "claimed", "sending", "submitted", "fetching")
JOB_STATUSES = ("queued", "running", "succeeded", "failed", "interrupted")
LLM_KINDS = ("global", "shot", "character_check")
STORYBOARD_BUILD_MODES = ("resume", "from_scratch")

PAID_CALL_STATUS_CHECK = _in("status", PAID_CALL_STATUSES)
PAID_CALL_LIVE_WHERE = _in("status", PAID_CALL_LIVE_STATUSES)
STORYBOARD_LIVE_WHERE = "status IN ('queued', 'running')"
LLM_SHOT_LIVE_WHERE = f"kind = 'shot' AND {PAID_CALL_LIVE_WHERE}"
LLM_GLOBAL_LIVE_WHERE = f"kind = 'global' AND {PAID_CALL_LIVE_WHERE}"
LLM_CHECK_LIVE_WHERE = f"kind = 'character_check' AND {PAID_CALL_LIVE_WHERE}"
LLM_CHECK_SLOT_WHERE = "kind = 'character_check'"
KF_SHOT_SELECTED_WHERE = "selected AND owner_run_id IS NULL AND shot_id IS NOT NULL"
KF_SHEET_SELECTED_WHERE = "selected AND owner_run_id IS NULL AND shot_id IS NULL"
ATTEMPT_SELECTED_WHERE = "selected"
GENERATION_MODE_CHECK = "generation_mode IN ('v2v', 'images_only')"

NEW_TABLES = (
    "rebuild_llm_calls", "rebuild_chunk_attempts", "rebuild_keyframes",
    "project_storyboards", "rebuild_shots",
)


def _paid_call_columns():
    return [
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("dispatch_gen", sa.Integer(), nullable=False),
        sa.Column("claim_token", sa.String(64), nullable=True),
        sa.Column("rq_job_id", sa.String(160), nullable=True),
        sa.Column("not_live_since", sa.DateTime(), nullable=True),
        sa.Column("send_deadline", sa.DateTime(), nullable=True),
        sa.Column("sender_host", sa.String(128), nullable=True),
        sa.Column("sender_boot", sa.String(64), nullable=True),
        sa.Column("sender_job", sa.String(160), nullable=True),
        sa.Column("sender_note", sa.Text(), nullable=True),
        sa.Column("task_id", sa.String(128), nullable=True),
        sa.Column("request", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("estimate_credits", sa.Float(), nullable=False),
        sa.Column("cost_credits", sa.Float(), nullable=True),
        sa.Column("fetch_attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    ]


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
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE project_type_enum ADD VALUE IF NOT EXISTS 'rebuild'")
        op.execute("ALTER TYPE run_model_enum ADD VALUE IF NOT EXISTS 'kling-4-0'")

    with op.batch_alter_table("video_projects") as batch:
        batch.add_column(sa.Column("rebuild_analysis_token", sa.String(36), nullable=True))

    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column(
            "generation_mode", sa.String(16), nullable=False,
            server_default=sa.text("'v2v'"),
        ))
        batch.add_column(sa.Column("rebuild_snapshot", sa.JSON(), nullable=True))
        batch.create_check_constraint("ck_runs_generation_mode", GENERATION_MODE_CHECK)

    op.create_table(
        "rebuild_shots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("frame_sec", sa.Float(), nullable=False),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("framing", sa.Text(), nullable=True),
        sa.Column("camera", sa.Text(), nullable=True),
        sa.Column("characters", sa.JSON(), nullable=True),
        sa.Column("action", sa.Text(), nullable=True),
        sa.Column("speaking_on_camera", sa.String(64), nullable=True),
        sa.Column("is_brand_card", sa.Boolean(), nullable=False),
        sa.Column("sheet_key", sa.String(64), nullable=True),
        sa.Column("storyboard_id", sa.String(36), nullable=True),
        sa.Column("folded_into", sa.String(36), nullable=True),
        sa.Column("force_anchor", sa.Boolean(), nullable=False),
        sa.Column("edited_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("project_id", "index", name="uq_rebuild_shots_project_index"),
    )
    op.create_index("ix_rebuild_shots_project_id", "rebuild_shots", ["project_id"])

    op.create_table(
        "project_storyboards",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("shots_generation", sa.Integer(), nullable=False),
        sa.Column("build_mode", sa.String(16), nullable=True),
        sa.Column("global_call_id", sa.String(36), nullable=True),
        sa.Column("model", sa.String(64), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("budget_credits", sa.Float(), nullable=False),
        sa.Column("overwrite_edits", sa.Boolean(), nullable=False),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        *_dispatch_columns(),
        sa.CheckConstraint(_in("status", JOB_STATUSES), name="ck_project_storyboards_status"),
        sa.CheckConstraint("build_mode IS NULL OR " + _in("build_mode", STORYBOARD_BUILD_MODES),
                           name="ck_project_storyboards_build_mode"),
    )
    op.create_index("ix_project_storyboards_project_id", "project_storyboards", ["project_id"])
    op.create_index("uq_project_storyboards_live", "project_storyboards", ["project_id"],
                    unique=True, postgresql_where=sa.text(STORYBOARD_LIVE_WHERE),
                    sqlite_where=sa.text(STORYBOARD_LIVE_WHERE))

    op.create_table(
        "rebuild_keyframes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("batch_id", sa.String(36), nullable=False),
        sa.Column("shot_id", sa.String(36),
                  sa.ForeignKey("rebuild_shots.id", ondelete="CASCADE"), nullable=True),
        sa.Column("owner_run_id", sa.String(36),
                  sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=True),
        sa.Column("sheet_key", sa.String(64), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("input_paths", sa.JSON(), nullable=True),
        sa.Column("resolution", sa.String(8), nullable=False),
        sa.Column("aspect_ratio", sa.String(8), nullable=True),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("image_sha256", sa.String(64), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("selected", sa.Boolean(), nullable=False),
        *_paid_call_columns(),
        sa.CheckConstraint(PAID_CALL_STATUS_CHECK, name="ck_rebuild_keyframes_status"),
        sa.CheckConstraint(
            "(shot_id IS NULL AND sheet_key IS NOT NULL) OR "
            "(shot_id IS NOT NULL AND sheet_key IS NULL)",
            name="ck_rebuild_keyframes_slot",
        ),
    )
    op.create_index("ix_rebuild_keyframes_project_id", "rebuild_keyframes", ["project_id"])
    op.create_index("ix_rebuild_keyframes_batch_id", "rebuild_keyframes", ["batch_id"])
    op.create_index("ix_rebuild_keyframes_shot_id", "rebuild_keyframes", ["shot_id"])
    op.create_index("ix_rebuild_keyframes_status", "rebuild_keyframes", ["status"])
    op.create_index("uq_rebuild_kf_shot_selected", "rebuild_keyframes", ["shot_id"],
                    unique=True, postgresql_where=sa.text(KF_SHOT_SELECTED_WHERE),
                    sqlite_where=sa.text(KF_SHOT_SELECTED_WHERE))
    op.create_index("uq_rebuild_kf_sheet_selected", "rebuild_keyframes",
                    ["project_id", "sheet_key"],
                    unique=True, postgresql_where=sa.text(KF_SHEET_SELECTED_WHERE),
                    sqlite_where=sa.text(KF_SHEET_SELECTED_WHERE))

    op.create_table(
        "rebuild_chunk_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36),
                  sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_segment_id", sa.String(36),
                  sa.ForeignKey("run_segments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(32), nullable=False),
        sa.Column("resolution", sa.String(8), nullable=False),
        sa.Column("request_sec", sa.Integer(), nullable=False),
        sa.Column("chunk_frames", sa.Integer(), nullable=False),
        sa.Column("raw_path", sa.Text(), nullable=True),
        sa.Column("fitted_path", sa.Text(), nullable=True),
        sa.Column("qa", sa.JSON(), nullable=True),
        sa.Column("selected", sa.Boolean(), nullable=False),
        *_paid_call_columns(),
        sa.CheckConstraint(PAID_CALL_STATUS_CHECK, name="ck_rebuild_chunk_attempts_status"),
        sa.UniqueConstraint("run_segment_id", "attempt_no",
                            name="uq_rebuild_chunk_attempts_segment_no"),
    )
    op.create_index("ix_rebuild_chunk_attempts_run_id", "rebuild_chunk_attempts", ["run_id"])
    op.create_index("ix_rebuild_chunk_attempts_status", "rebuild_chunk_attempts", ["status"])
    op.create_index("uq_rebuild_attempt_live", "rebuild_chunk_attempts", ["run_segment_id"],
                    unique=True, postgresql_where=sa.text(PAID_CALL_LIVE_WHERE),
                    sqlite_where=sa.text(PAID_CALL_LIVE_WHERE))
    op.create_index("uq_rebuild_attempt_selected", "rebuild_chunk_attempts", ["run_segment_id"],
                    unique=True, postgresql_where=sa.text(ATTEMPT_SELECTED_WHERE),
                    sqlite_where=sa.text(ATTEMPT_SELECTED_WHERE))

    op.create_table(
        "rebuild_llm_calls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("storyboard_id", sa.String(36),
                  sa.ForeignKey("project_storyboards.id", ondelete="CASCADE"), nullable=True),
        sa.Column("attempt_id", sa.String(36),
                  sa.ForeignKey("rebuild_chunk_attempts.id", ondelete="CASCADE"), nullable=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("shot_id", sa.String(36), nullable=True),
        sa.Column("slot", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("global_call_id", sa.String(36), nullable=True),
        sa.Column("response", sa.JSON(), nullable=True),
        *_paid_call_columns(),
        sa.CheckConstraint(PAID_CALL_STATUS_CHECK, name="ck_rebuild_llm_calls_status"),
        sa.CheckConstraint(_in("kind", LLM_KINDS), name="ck_rebuild_llm_calls_kind"),
        sa.CheckConstraint("(storyboard_id IS NULL) <> (attempt_id IS NULL)",
                           name="ck_rebuild_llm_calls_owner"),
        sa.CheckConstraint(
            "kind <> 'shot' OR (shot_id IS NOT NULL AND global_call_id IS NOT NULL)",
            name="ck_rebuild_llm_calls_shot",
        ),
        sa.CheckConstraint(
            "kind <> 'character_check' OR (attempt_id IS NOT NULL AND slot IS NOT NULL)",
            name="ck_rebuild_llm_calls_check",
        ),
    )
    op.create_index("ix_rebuild_llm_calls_project_id", "rebuild_llm_calls", ["project_id"])
    op.create_index("ix_rebuild_llm_calls_storyboard_id", "rebuild_llm_calls", ["storyboard_id"])
    op.create_index("ix_rebuild_llm_calls_attempt_id", "rebuild_llm_calls", ["attempt_id"])
    op.create_index("ix_rebuild_llm_calls_status", "rebuild_llm_calls", ["status"])
    op.create_index("uq_rebuild_llm_shot_live", "rebuild_llm_calls",
                    ["global_call_id", "shot_id"],
                    unique=True, postgresql_where=sa.text(LLM_SHOT_LIVE_WHERE),
                    sqlite_where=sa.text(LLM_SHOT_LIVE_WHERE))
    op.create_index("uq_rebuild_llm_global_live", "rebuild_llm_calls", ["project_id"],
                    unique=True, postgresql_where=sa.text(LLM_GLOBAL_LIVE_WHERE),
                    sqlite_where=sa.text(LLM_GLOBAL_LIVE_WHERE))
    op.create_index("uq_rebuild_llm_check_live", "rebuild_llm_calls", ["attempt_id"],
                    unique=True, postgresql_where=sa.text(LLM_CHECK_LIVE_WHERE),
                    sqlite_where=sa.text(LLM_CHECK_LIVE_WHERE))
    op.create_index("uq_rebuild_llm_check_slot", "rebuild_llm_calls", ["attempt_id", "slot"],
                    unique=True, postgresql_where=sa.text(LLM_CHECK_SLOT_WHERE),
                    sqlite_where=sa.text(LLM_CHECK_SLOT_WHERE))


def downgrade() -> None:
    bind = op.get_bind()
    blockers = []
    # Compared as text so this works whether or not the label exists yet.
    n = bind.execute(sa.text(
        "SELECT count(*) FROM video_projects WHERE CAST(project_type AS VARCHAR) = 'rebuild'"
    )).scalar()
    if n:
        blockers.append(f"{n} rebuild project(s)")
    n = bind.execute(sa.text(
        "SELECT count(*) FROM runs WHERE generation_mode = 'images_only' "
        "OR rebuild_snapshot IS NOT NULL OR CAST(model AS VARCHAR) = 'kling-4-0'"
    )).scalar()
    if n:
        blockers.append(f"{n} images-only / Kling run(s)")
    for table in NEW_TABLES:
        n = bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar()
        if n:
            blockers.append(f"{n} row(s) in {table}")
    if blockers:
        raise RuntimeError(
            "refusing to downgrade the rebuild feature: " + ", ".join(blockers)
            + " — these are the only record of how rebuild runs were made and of "
              "billed provider tasks; export and remove them first "
              "(scripts/rebuild_quarantine.py)"
        )

    op.drop_index("uq_rebuild_llm_check_slot", table_name="rebuild_llm_calls")
    op.drop_index("uq_rebuild_llm_check_live", table_name="rebuild_llm_calls")
    op.drop_index("uq_rebuild_llm_global_live", table_name="rebuild_llm_calls")
    op.drop_index("uq_rebuild_llm_shot_live", table_name="rebuild_llm_calls")
    op.drop_index("ix_rebuild_llm_calls_status", table_name="rebuild_llm_calls")
    op.drop_index("ix_rebuild_llm_calls_attempt_id", table_name="rebuild_llm_calls")
    op.drop_index("ix_rebuild_llm_calls_storyboard_id", table_name="rebuild_llm_calls")
    op.drop_index("ix_rebuild_llm_calls_project_id", table_name="rebuild_llm_calls")
    op.drop_table("rebuild_llm_calls")
    op.drop_index("uq_rebuild_attempt_selected", table_name="rebuild_chunk_attempts")
    op.drop_index("uq_rebuild_attempt_live", table_name="rebuild_chunk_attempts")
    op.drop_index("ix_rebuild_chunk_attempts_status", table_name="rebuild_chunk_attempts")
    op.drop_index("ix_rebuild_chunk_attempts_run_id", table_name="rebuild_chunk_attempts")
    op.drop_table("rebuild_chunk_attempts")
    op.drop_index("uq_rebuild_kf_sheet_selected", table_name="rebuild_keyframes")
    op.drop_index("uq_rebuild_kf_shot_selected", table_name="rebuild_keyframes")
    op.drop_index("ix_rebuild_keyframes_status", table_name="rebuild_keyframes")
    op.drop_index("ix_rebuild_keyframes_shot_id", table_name="rebuild_keyframes")
    op.drop_index("ix_rebuild_keyframes_batch_id", table_name="rebuild_keyframes")
    op.drop_index("ix_rebuild_keyframes_project_id", table_name="rebuild_keyframes")
    op.drop_table("rebuild_keyframes")
    op.drop_index("uq_project_storyboards_live", table_name="project_storyboards")
    op.drop_index("ix_project_storyboards_project_id", table_name="project_storyboards")
    op.drop_table("project_storyboards")
    op.drop_index("ix_rebuild_shots_project_id", table_name="rebuild_shots")
    op.drop_table("rebuild_shots")
    with op.batch_alter_table("runs") as batch:
        batch.drop_constraint("ck_runs_generation_mode", type_="check")
        batch.drop_column("rebuild_snapshot")
        batch.drop_column("generation_mode")
    with op.batch_alter_table("video_projects") as batch:
        batch.drop_column("rebuild_analysis_token")
