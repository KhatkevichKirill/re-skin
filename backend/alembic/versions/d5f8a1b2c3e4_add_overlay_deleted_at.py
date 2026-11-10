"""add_overlay_deleted_at

Adds a nullable soft-delete marker to overlay_jobs. When set, the job is hidden
from all listings/lookups (treated as 404) but the row is retained so the source
video reference (source_run_id / source_local_path) stays findable in the DB.
NULL = live job.

Revision ID: d5f8a1b2c3e4
Revises: c9e2f4a6b810
Create Date: 2026-07-01 22:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5f8a1b2c3e4'
down_revision: Union[str, None] = 'c9e2f4a6b810'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'overlay_jobs',
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('overlay_jobs', 'deleted_at')
