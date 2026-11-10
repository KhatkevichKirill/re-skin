"""Autopilot core: project mode + autopilot tables (AICP-363).

Additive: ``video_projects.mode`` (VARCHAR + CHECK, server default 'manual',
so every existing row and every old-image writer stays manual) and the
autopilot tables of docs/autopilot-design.md §2, plus the singleton
``autopilot_gate`` row. No backfill.

Downgrade refuses while any ``project_autopilots`` row exists: it would
destroy live approvals, effects and the spend attribution.

Revision ID: r1a7c3e5b9d2
Revises: q8b4d0f2a716
Create Date: 2026-10-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "r1a7c3e5b9d2"
down_revision: Union[str, None] = "q8b4d0f2a716"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LIVE = "status IN ('active', 'paused')"


def _in(col, values):
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    # batch_alter_table: SQLite cannot ALTER ADD CONSTRAINT (Postgres runs it in place).
    with op.batch_alter_table("video_projects") as batch:
        batch.add_column(
            sa.Column("mode", sa.String(16), nullable=False, server_default="manual"))
        batch.create_check_constraint("ck_video_projects_mode", "mode IN ('manual', 'assistant')")

    op.create_table(
        "project_autopilots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36),
                  sa.ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("flow_id", sa.String(64), nullable=False),
        sa.Column("flow_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("approved_budget_mcredits", sa.BigInteger(), nullable=True),
        sa.Column("tolerance", sa.Float(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.String(64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("outstanding", sa.Integer(), nullable=False),
        sa.Column("next_check_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            _in("status", ("active", "paused", "completed", "stopped", "failed")),
            name="ck_project_autopilots_status",
        ),
        sa.CheckConstraint("tolerance >= 0 AND tolerance <= 0.5",
                           name="ck_project_autopilots_tolerance"),
    )
    op.create_index("ix_project_autopilots_project_id", "project_autopilots", ["project_id"])
    op.create_index(
        "uq_project_autopilots_live", "project_autopilots", ["project_id"], unique=True,
        postgresql_where=sa.text(_LIVE), sqlite_where=sa.text(_LIVE),
    )

    op.create_table(
        "autopilot_steps",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("autopilot_id", sa.String(36),
                  sa.ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_id", sa.String(96), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("output", sa.JSON(), nullable=True),
        sa.Column("depends_on", sa.JSON(), nullable=True),
        sa.Column("previous_outputs", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("next_check_at", sa.DateTime(), nullable=True),
        sa.Column("refusals", sa.Integer(), nullable=False),
        sa.Column("errors", sa.Integer(), nullable=False),
        sa.Column("last_refusal", sa.JSON(), nullable=True),
        sa.Column("reopen_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("autopilot_id", "step_id", name="uq_autopilot_steps_step"),
        sa.CheckConstraint(
            _in("status", ("pending", "active", "completed", "failed", "skipped")),
            name="ck_autopilot_steps_status",
        ),
    )
    op.create_index("ix_autopilot_steps_autopilot_id", "autopilot_steps", ["autopilot_id"])

    op.create_table(
        "autopilot_effects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("autopilot_id", sa.String(36),
                  sa.ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_id", sa.String(96), nullable=False),
        sa.Column("step_generation", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(96), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("handler", sa.String(96), nullable=False),
        sa.Column("handler_version", sa.Integer(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("params_sha256", sa.String(64), nullable=False),
        sa.Column("paid", sa.Boolean(), nullable=False),
        sa.Column("estimate_mcredits", sa.BigInteger(), nullable=False),
        sa.Column("targets", sa.JSON(), nullable=True),
        sa.Column("target_versions", sa.JSON(), nullable=True),
        sa.Column("created_refs", sa.JSON(), nullable=True),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("approval_id", sa.String(36), nullable=True),
        sa.Column("claim_token", sa.String(64), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("requeues", sa.Integer(), nullable=False),
        sa.Column("rq_job_id", sa.String(128), nullable=True),
        sa.Column("not_live_since", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_autopilot_effects_key"),
        sa.CheckConstraint(_in("kind", ("action", "compute", "judgement")),
                           name="ck_autopilot_effects_kind"),
        sa.CheckConstraint(
            _in("state", ("launching", "launched", "not_launched", "unknown", "queued",
                          "running", "succeeded", "failed", "prepared", "sending",
                          "submitted")),
            name="ck_autopilot_effects_state",
        ),
    )
    op.create_index("ix_autopilot_effects_autopilot_state", "autopilot_effects",
                    ["autopilot_id", "state"])

    op.create_table(
        "autopilot_approvals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("autopilot_id", sa.String(36),
                  sa.ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_id", sa.String(96), nullable=False),
        sa.Column("step_generation", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("dedupe_key", sa.String(400), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("subject", sa.String(160), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("proposal", sa.JSON(), nullable=False),
        sa.Column("options", sa.JSON(), nullable=True),
        sa.Column("chosen_option", sa.String(64), nullable=True),
        sa.Column("artifacts", sa.JSON(), nullable=True),
        sa.Column("action", sa.JSON(), nullable=True),
        sa.Column("action_sha256", sa.String(64), nullable=True),
        sa.Column("bound_versions", sa.JSON(), nullable=True),
        sa.Column("cost_mcredits", sa.BigInteger(), nullable=False),
        sa.Column("max_cost_mcredits", sa.BigInteger(), nullable=True),
        sa.Column("committed_at_ask_mcredits", sa.BigInteger(), nullable=True),
        sa.Column("budget_after_mcredits", sa.BigInteger(), nullable=True),
        sa.Column("comments", sa.JSON(), nullable=False),
        sa.Column("answer_comment", sa.Text(), nullable=True),
        sa.Column("answered_decision", sa.String(16), nullable=True),
        sa.Column("answered_at", sa.DateTime(), nullable=True),
        sa.Column("answered_by", sa.String(128), nullable=True),
        sa.Column("consumed_by_effect_id", sa.String(36), nullable=True),
        sa.Column("consumed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("dedupe_key", name="uq_autopilot_approvals_dedupe"),
        sa.CheckConstraint(
            _in("kind", ("budget", "regeneration", "overrun", "listening_probe",
                         "listening_final", "handoff")),
            name="ck_autopilot_approvals_kind",
        ),
        sa.CheckConstraint(
            _in("status", ("pending", "approved", "rejected", "superseded", "consumed")),
            name="ck_autopilot_approvals_status",
        ),
    )
    op.create_index("ix_autopilot_approvals_autopilot_status", "autopilot_approvals",
                    ["autopilot_id", "status"])

    op.create_table(
        "project_spend_scopes",
        sa.Column("project_id", sa.String(36), primary_key=True),
        sa.Column("scope_key", sa.String(128), primary_key=True),
        sa.Column("registered_at", sa.DateTime(), nullable=False),
        sa.Column("registered_by_autopilot_id", sa.String(36), nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
    )

    op.create_table(
        "autopilot_journal",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
                  primary_key=True, autoincrement=True),
        sa.Column("autopilot_id", sa.String(36),
                  sa.ForeignKey("project_autopilots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("step_id", sa.String(96), nullable=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("cost_mcredits", sa.BigInteger(), nullable=True),
        sa.Column("effect_id", sa.String(36), nullable=True),
        sa.Column("approval_id", sa.String(36), nullable=True),
    )
    op.create_index("ix_autopilot_journal_autopilot_id", "autopilot_journal",
                    ["autopilot_id", "id"])

    op.create_table(
        "autopilot_gate",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("held", sa.Boolean(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("held_since", sa.DateTime(), nullable=True),
        sa.Column("held_by", sa.String(128), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
    )
    op.bulk_insert(sa.table("autopilot_gate", sa.column("id", sa.Integer),
                            sa.column("held", sa.Boolean), sa.column("generation", sa.Integer)),
                   [{"id": 1, "held": False, "generation": 0}])

    op.create_table(
        "project_manual_spend_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("scope_key", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("mcredits", sa.BigInteger(), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("seen_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_project_manual_spend_events_project", "project_manual_spend_events",
                    ["project_id", "seen_at"])


def downgrade() -> None:
    bind = op.get_bind()
    n = bind.execute(sa.text("SELECT count(*) FROM project_autopilots")).scalar()
    if n:
        raise RuntimeError(
            f"refusing to downgrade r1a7c3e5b9d2: {n} project_autopilots row(s) exist; "
            "dropping them would destroy live approvals, effects and spend attribution"
        )
    op.drop_table("project_manual_spend_events")
    op.drop_table("autopilot_gate")
    op.drop_table("autopilot_journal")
    op.drop_table("project_spend_scopes")
    op.drop_table("autopilot_approvals")
    op.drop_table("autopilot_effects")
    op.drop_table("autopilot_steps")
    op.drop_table("project_autopilots")
    with op.batch_alter_table("video_projects") as batch:
        batch.drop_constraint("ck_video_projects_mode", type_="check")
        batch.drop_column("mode")
