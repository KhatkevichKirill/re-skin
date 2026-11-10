"""add_overlay_final_creatives_exported_at

Adds a nullable timestamp to overlay_jobs recording when an overlay result was
logged to the "Final Creatives" sheet (mirrors runs.final_creatives_exported_at).
NULL = not yet added; a value drives the "already added" UI state.

Revision ID: a3b5c7d9e024
Revises: f2c4d6e8a012
Create Date: 2026-07-01 00:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3b5c7d9e024'
down_revision: Union[str, None] = 'f2c4d6e8a012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'overlay_jobs',
        sa.Column('final_creatives_exported_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('overlay_jobs', 'final_creatives_exported_at')
