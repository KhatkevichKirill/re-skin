"""add_kie_spend_ledger

Daily kie.ai spend cap (AICP-155). Three new tables, nothing existing is
touched, no backfill:

  kie_spend_days          — one row per UTC day: the charged total (integer
      milli-credits). Its row is also the lock every reserve/draw takes.
  kie_spend_reservations  — what one action (scope_key, e.g. run:<id>)
      reserved and consumed on one day; charged = max(reserved, consumed).
  kie_spend_draws         — one row per paid kie.ai call, unique call_key
      (the kie taskId), so a replayed submit/poll cannot double-count; the
      estimate is replaced by kie.ai's creditsConsumed once reported.

Safe to apply before the code deploys (old code never reads the tables) and
safe to leave applied on a rollback.

Revision ID: a7c9e1b3d5f8
Revises: f1c3e5a7b902
Create Date: 2026-10-16 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = 'a7c9e1b3d5f8'
down_revision: Union[str, None] = 'f1c3e5a7b902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "kie_spend_days",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("charged_mcredits", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("day"),
        sa.CheckConstraint("charged_mcredits >= 0", name="ck_kie_spend_days_nonneg"),
    )
    op.create_table(
        "kie_spend_reservations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("scope_key", sa.String(length=128), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("reserved_mcredits", sa.BigInteger(), nullable=False),
        sa.Column("consumed_mcredits", sa.BigInteger(), nullable=False),
        sa.Column("charged_mcredits", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope_key", "day", name="uq_kie_spend_reservation_scope_day"),
        sa.CheckConstraint(
            "reserved_mcredits >= 0 AND consumed_mcredits >= 0 AND charged_mcredits >= 0",
            name="ck_kie_spend_reservations_nonneg",
        ),
    )
    op.create_index("ix_kie_spend_reservations_scope_key", "kie_spend_reservations", ["scope_key"])
    op.create_index("ix_kie_spend_reservations_day", "kie_spend_reservations", ["day"])
    op.create_index("ix_kie_spend_reservations_run_id", "kie_spend_reservations", ["run_id"])
    op.create_table(
        "kie_spend_draws",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("call_key", sa.String(length=255), nullable=False),
        sa.Column("reservation_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("estimated_mcredits", sa.BigInteger(), nullable=False),
        sa.Column("mcredits", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("settled_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_key"),
        sa.ForeignKeyConstraint(
            ["reservation_id"], ["kie_spend_reservations.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_kie_spend_draws_reservation_id", "kie_spend_draws", ["reservation_id"])


def downgrade() -> None:
    op.drop_index("ix_kie_spend_draws_reservation_id", table_name="kie_spend_draws")
    op.drop_table("kie_spend_draws")
    op.drop_index("ix_kie_spend_reservations_run_id", table_name="kie_spend_reservations")
    op.drop_index("ix_kie_spend_reservations_day", table_name="kie_spend_reservations")
    op.drop_index("ix_kie_spend_reservations_scope_key", table_name="kie_spend_reservations")
    op.drop_table("kie_spend_reservations")
    op.drop_table("kie_spend_days")
