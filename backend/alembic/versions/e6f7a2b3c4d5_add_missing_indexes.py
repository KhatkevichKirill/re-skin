"""add_missing_indexes

Adds indexes to two columns that are frequently filtered/joined:
- runs.status: filtered by overlay dashboards with WHERE status='done'
- run_segments.segment_def_id: an ON DELETE CASCADE FK with no index,
  forcing seq scans on run_segments for every SegmentDef delete

Revision ID: e6f7a2b3c4d5
Revises: d5f8a1b2c3e4
Create Date: 2026-07-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e6f7a2b3c4d5'
down_revision: Union[str, None] = 'd5f8a1b2c3e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(op.f('ix_runs_status'), 'runs', ['status'], unique=False)
    op.create_index(op.f('ix_run_segments_segment_def_id'), 'run_segments', ['segment_def_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_run_segments_segment_def_id'), table_name='run_segments')
    op.drop_index(op.f('ix_runs_status'), table_name='runs')
