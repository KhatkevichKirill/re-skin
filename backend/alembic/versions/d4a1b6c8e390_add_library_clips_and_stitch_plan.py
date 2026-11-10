"""add_library_clips_and_stitch_plan

Creates the library_clips table (global reusable stitch clips, e.g. alternative
outros) and adds runs.stitch_plan (JSON, nullable) — the per-run plan of what
the finalize phase concatenates (NULL = all segments, [] = no stitch, or a
custom ordered list of segment/library entries).

Revision ID: d4a1b6c8e390
Revises: c7e8f9a0b1d2
Create Date: 2026-07-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4a1b6c8e390'
down_revision: Union[str, None] = 'c7e8f9a0b1d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'library_clips',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('local_path', sa.Text(), nullable=False),
        sa.Column('duration_sec', sa.Float(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.add_column(
        'runs',
        sa.Column('stitch_plan', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('runs', 'stitch_plan')
    op.drop_table('library_clips')
