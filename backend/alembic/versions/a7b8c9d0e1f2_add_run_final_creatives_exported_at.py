"""add_run_final_creatives_exported_at

Adds a nullable timestamp to the runs table recording when a run was logged to
the "Final Creatives" tracking spreadsheet. NULL = not yet added; a value drives
the "already added" UI state so operators don't add the same run twice.

Revision ID: a7b8c9d0e1f2
Revises: b2c3d4e5f6a7
Create Date: 2026-06-29 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7b8c9d0e1f2'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'runs',
        sa.Column('final_creatives_exported_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('runs', 'final_creatives_exported_at')
