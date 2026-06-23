"""add_project_name

Adds an operator-editable name column to the video_projects table so projects
can carry a human-readable label instead of only a UUID + source filename.

Nullable: existing projects keep NULL and the UI falls back to source_ref.

Revision ID: f1b2c3d4e5a6
Revises: e5a1c7d2f3b6
Create Date: 2026-06-22 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1b2c3d4e5a6'
down_revision: Union[str, None] = 'e5a1c7d2f3b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'video_projects',
        sa.Column('name', sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('video_projects', 'name')
