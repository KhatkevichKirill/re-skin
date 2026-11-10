"""add_caption_demo_status

Adds a nullable ``caption_demo_status`` string to both the runs and overlay_jobs
tables, recording the Caption Demo action taken for a delivered creative:

  "queued"          — a row was appended to the "Caption Demo" sheet tab.
  "added_to_table"  — the creative was already in the Final Creatives sheet, so
                      it is only marked here (no Caption Demo row written).

NULL = not acted on. Drives the "Caption Demo" column in the project runs table
and the overlay jobs table, and the idempotent Add-to-Caption-Demo button.

Revision ID: f3a5c7e9b246
Revises: e6f7a2b3c4d5
Create Date: 2026-07-03 22:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3a5c7e9b246'
down_revision: Union[str, None] = 'e6f7a2b3c4d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'runs',
        sa.Column('caption_demo_status', sa.String(length=32), nullable=True),
    )
    op.add_column(
        'overlay_jobs',
        sa.Column('caption_demo_status', sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('overlay_jobs', 'caption_demo_status')
    op.drop_column('runs', 'caption_demo_status')
