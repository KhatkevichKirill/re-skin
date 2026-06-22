"""add_run_segment_overrides

Adds prompt_override (Text, nullable) and reference_image_urls_override (JSON, nullable)
to the run_segments table, enabling per-segment prompt and reference image overrides for
individual segment re-runs (TR6).

Revision ID: c3f9a2e1b450
Revises: 81b441a0932d
Create Date: 2026-06-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3f9a2e1b450'
down_revision: Union[str, None] = '81b441a0932d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'run_segments',
        sa.Column('prompt_override', sa.Text(), nullable=True),
    )
    op.add_column(
        'run_segments',
        sa.Column('reference_image_urls_override', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('run_segments', 'reference_image_urls_override')
    op.drop_column('run_segments', 'prompt_override')
