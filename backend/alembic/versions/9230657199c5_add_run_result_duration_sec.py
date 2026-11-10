"""add_run_result_duration_sec

Adds a nullable float `result_duration_sec` to runs, storing the probed
duration of the result video (populated once at finalize — fresh stitch or
delivery-retry reuse). Avoids re-running ffprobe on every run-detail /
status-fragment render for the "Change ending" card. NULL for runs finalized
before this column existed; callers fall back to an on-demand probe of
`result_local_path` in that case.

Revision ID: 9230657199c5
Revises: a2c4e6f8b135
Create Date: 2026-07-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9230657199c5'
down_revision: Union[str, None] = 'a2c4e6f8b135'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'runs',
        sa.Column('result_duration_sec', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('runs', 'result_duration_sec')
