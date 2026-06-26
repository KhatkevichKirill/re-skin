"""add_incomplete_to_run_status_enum

Adds the 'incomplete' value to run_status_enum. A run becomes 'incomplete' when
one or more swap segments could not be generated (after automatic retries): the
final video is NOT stitched (no mix of swapped + original clips), the completed
segments' results are preserved, and the operator re-runs the failed segments
from the UI. Once every swap segment is completed the run stitches + delivers
and reaches 'done'.

No-op on SQLite (ENUMs stored as VARCHAR, no type object to alter).

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f7
Create Date: 2026-06-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # ADD VALUE is allowed inside a transaction on Postgres 12+ as long as
        # the new value is not used in the same transaction (it isn't here).
        op.execute("ALTER TYPE run_status_enum ADD VALUE IF NOT EXISTS 'incomplete'")


def downgrade() -> None:
    # Postgres cannot drop an enum value without recreating the type; no-op.
    pass
