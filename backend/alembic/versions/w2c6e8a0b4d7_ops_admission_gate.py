"""Deploy admission gate + API idempotency keys (AICP-350, AICP-344 stage 1).

Additive only (docs/assistant-drain-check-design.md §7):

* ``ops_admission_gate`` — singleton row (id = 1, CHECK) saying whether new
  work may be admitted. Written only by ``deploy/ops-gate/*.sql``; every
  admission reads it ``FOR SHARE`` (``app/ops_gate.py``). The row is inserted
  here, open (``paused = false``, generation 0).
* ``api_idempotency_keys`` — ``Idempotency-Key`` claims recorded inside the
  admission transaction (design §3).

No enum change, no backfill. The old image ignores both tables, so rolling the
image back is code only (it just stops enforcing the gate — use the manual
drain then).

Revision ID: w2c6e8a0b4d7
Revises: v3a9c5e7b1d4
Create Date: 2026-11-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "w2c6e8a0b4d7"
down_revision: Union[str, None] = "v3a9c5e7b1d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ops_admission_gate",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("paused", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("owner", sa.String(200), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("generation", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_by", sa.String(200), nullable=True),
        sa.CheckConstraint("id = 1", name="ck_ops_admission_gate_singleton"),
        sa.CheckConstraint("NOT paused OR (owner IS NOT NULL AND paused_at IS NOT NULL)",
                           name="ck_ops_admission_gate_paused_fields"),
    )
    op.execute("INSERT INTO ops_admission_gate (id, paused, generation) VALUES (1, false, 0)")

    op.create_table(
        "api_idempotency_keys",
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("route", sa.String(255), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("operation", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_api_idempotency_keys_created_at", "api_idempotency_keys", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_api_idempotency_keys_created_at", table_name="api_idempotency_keys")
    op.drop_table("api_idempotency_keys")
    op.drop_table("ops_admission_gate")
