"""Autopilot chat notifications (AICP-374).

Additive only: the ``autopilot_notifications`` table — one row per chat
message the notifier keeps in step with an approval or an attempt
(docs/autopilot-operator-ux.md §2.2). No enum change, no backfill: the
notifier writes its own ``notifier:since`` marker on its first pass.

Revision ID: u7c2e4a6b8d1
Revises: u5e1b7c9d3f4
Create Date: 2026-11-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "u7c2e4a6b8d1"
down_revision: Union[str, None] = "u5e1b7c9d3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "autopilot_notifications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("project_id", sa.String(36), nullable=True),
        sa.Column("autopilot_id", sa.String(36), nullable=True),
        sa.Column("approval_id", sa.String(36), nullable=True),
        sa.Column("journal_id", sa.BigInteger(), nullable=True),
        sa.Column("chat_id", sa.String(64), nullable=False, server_default=""),
        sa.Column("message_id", sa.BigInteger(), nullable=True),
        sa.Column("posted_at", sa.DateTime(), nullable=True),
        sa.Column("state", sa.String(32), nullable=True),
        sa.Column("state_hash", sa.String(64), nullable=True),
        sa.Column("gone", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("last_reminder_at", sa.DateTime(), nullable=True),
        sa.Column("claim_owner", sa.String(64), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("next_try_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("key", name="uq_autopilot_notifications_key"),
        sa.CheckConstraint("kind IN ('approval', 'pause', 'end', 'marker')",
                           name="ck_autopilot_notifications_kind"),
    )
    op.create_index("ix_autopilot_notifications_open", "autopilot_notifications",
                    ["closed_at", "created_at"])


def downgrade() -> None:
    # Messages already in the chat stay there; only their bookkeeping is dropped.
    op.drop_index("ix_autopilot_notifications_open", table_name="autopilot_notifications")
    op.drop_table("autopilot_notifications")
