"""add_run_segment_cost_credits

Adds a nullable float `cost_credits` to run_segments recording the kie.ai
`creditsConsumed` for that segment's AI task (captured on task success). NULL
until the task completes / for skipped segments. Summing across a run's segments
gives the run's total cost.

Revision ID: b7d1e9f3a204
Revises: a3b5c7d9e024
Create Date: 2026-07-01 04:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7d1e9f3a204'
down_revision: Union[str, None] = 'a3b5c7d9e024'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'run_segments',
        sa.Column('cost_credits', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('run_segments', 'cost_credits')
