"""add_seedance_variants_to_run_model_enum

The run_model_enum type was created with only ('seedance', 'gemini-omni') in
the v2 schema migration (e5a1c7d2f3b6).  Two new Seedance 2.0 variants —
"seedance-fast" (kie model id bytedance/seedance-2-fast) and "seedance-mini"
(kie model id bytedance/seedance-2-mini) — are now selectable per Run. They
share the same createTask input schema as the base "seedance" model and only
support 480p/720p resolutions.

This migration adds both values to the Postgres enum type.  It is a no-op on
SQLite (which stores ENUMs as VARCHAR and has no type object to alter).

Revision ID: b4f6d8e0c357
Revises: f3a5c7e9b246
Create Date: 2026-07-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b4f6d8e0c357'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # ALTER TYPE ... ADD VALUE must be run outside a transaction block on
        # older Postgres, but Postgres 12+ allows it inside a transaction.
        # Alembic wraps migrations in a transaction; use execute() directly.
        op.execute("ALTER TYPE run_model_enum ADD VALUE IF NOT EXISTS 'seedance-fast'")
        op.execute("ALTER TYPE run_model_enum ADD VALUE IF NOT EXISTS 'seedance-mini'")


def downgrade() -> None:
    # Postgres does not support removing values from an enum type without
    # recreating it — a safe downgrade is not possible.  Leave as a no-op.
    pass
