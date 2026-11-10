"""add_preprocess_review_to_run_status_enum

Adds the 'preprocess_review' value to run_status_enum. After GPT Image 2
preprocessing succeeds, a run pauses in 'preprocess_review' so the operator
can approve the preprocessed reference images (or regenerate them) before
Seedance/Gemini Omni starts — no AI video credits are spent until the
operator explicitly approves.

No-op on SQLite (ENUMs stored as VARCHAR, no type object to alter).

Revision ID: b3c4d5e6f7a8
Revises: a1b2c3d4e5f6
Create Date: 2026-07-30 18:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3c4d5e6f7a8'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # ADD VALUE is allowed inside a transaction on Postgres 12+ as long as
        # the new value is not used in the same transaction (it isn't here).
        op.execute(
            "ALTER TYPE run_status_enum ADD VALUE IF NOT EXISTS 'preprocess_review'"
        )


def downgrade() -> None:
    # Postgres cannot drop an enum value without recreating the type; no-op.
    pass
