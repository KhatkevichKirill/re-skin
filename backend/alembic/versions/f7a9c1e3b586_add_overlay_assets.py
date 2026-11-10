"""add_overlay_assets

Creates the overlay_assets table: a global reusable-PNG-card library for the
overlay-burn feature, mirroring library_clips for stitch clips. See
models.OverlayAsset's docstring for why this is a *copy-on-attach* library
(no soft reference from overlay_jobs.overlays, unlike stitch_plan -> clip_id)
— so, unlike library_clips, no downstream table/column is added here.

Revision ID: f7a9c1e3b586
Revises: e1f2a3b4c5d6
Create Date: 2026-07-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7a9c1e3b586'
down_revision: Union[str, None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'overlay_assets',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('local_path', sa.Text(), nullable=False),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('overlay_assets')
