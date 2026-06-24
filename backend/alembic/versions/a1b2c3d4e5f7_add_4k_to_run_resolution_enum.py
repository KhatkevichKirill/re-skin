"""add_4k_to_run_resolution_enum

The run_resolution_enum type was created with only ('480p', '720p', '1080p')
in the initial v2 migration (81b441a0932d).  The Run model was later updated
to include '4k' for Gemini Omni support, but no migration was added at that
point (SQLite doesn't enforce enum values at DB level so it went unnoticed).

This migration adds '4k' to the Postgres enum type.  It is a no-op on SQLite
(which stores ENUMs as VARCHAR and has no type object to alter).

Revision ID: a1b2c3d4e5f7
Revises: f1b2c3d4e5a6
Create Date: 2026-06-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, None] = 'f1b2c3d4e5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # ALTER TYPE ... ADD VALUE must be run outside a transaction block on
        # older Postgres, but Postgres 12+ allows it inside a transaction.
        # Alembic wraps migrations in a transaction; use execute() directly.
        op.execute("ALTER TYPE run_resolution_enum ADD VALUE IF NOT EXISTS '4k'")


def downgrade() -> None:
    # Postgres does not support removing values from an enum type without
    # recreating it — a safe downgrade is not possible.  Leave as a no-op.
    pass
