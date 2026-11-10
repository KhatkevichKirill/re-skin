"""add_run_language

Adds the nullable ``language`` column to the runs table.

The value is one of app/languages.py's codes ("en"/"es"/"pt"/"ja"/"de") and its
only effect is to suffix the id written to the Final Creatives ``Job ID`` cell
and the Caption Demo ``job_key``, so the downstream renaming pipeline can read
the creative's language back out of it.

Deliberately a plain VARCHAR and NOT an Enum: the supported set is pinned by the
downstream parser, and adding a language should be a one-line change in the
registry rather than an `ALTER TYPE ... ADD VALUE` (which cannot be reversed).

NULL is the correct value for every existing row — no suffix means English
downstream, which is exactly how those rows already behave.

Revision ID: b3d5f7a9c142
Revises: d3b8e5c710f4
Create Date: 2026-08-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3d5f7a9c142'
down_revision: Union[str, None] = 'd3b8e5c710f4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('runs', sa.Column('language', sa.String(length=8), nullable=True))


def downgrade() -> None:
    op.drop_column('runs', 'language')
