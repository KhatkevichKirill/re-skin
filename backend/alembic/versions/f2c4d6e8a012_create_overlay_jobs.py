"""create_overlay_jobs

Creates the `overlay_jobs` table for the standalone overlay-burn feature
(burn a transparent PNG onto a finished video). Shares no schema with the run
pipeline. `overlays` is a JSON column so it works on both SQLite and Postgres.

Revision ID: f2c4d6e8a012
Revises: a7b8c9d0e1f2
Create Date: 2026-06-30 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2c4d6e8a012'
down_revision: Union[str, None] = 'a7b8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'overlay_jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column(
            'status',
            sa.Enum('created', 'queued', 'processing', 'delivering', 'done', 'failed', name='overlay_status_enum'),
            nullable=False,
        ),
        sa.Column(
            'source_type',
            sa.Enum('run', 'upload', name='overlay_source_type_enum'),
            nullable=False,
        ),
        sa.Column('source_run_id', sa.String(length=36), nullable=True),
        sa.Column('source_local_path', sa.Text(), nullable=True),
        sa.Column('src_width', sa.Integer(), nullable=True),
        sa.Column('src_height', sa.Integer(), nullable=True),
        sa.Column('src_duration_sec', sa.Float(), nullable=True),
        sa.Column('overlays', sa.JSON(), nullable=False),
        sa.Column('deliver_to_drive', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('result_local_path', sa.Text(), nullable=True),
        sa.Column('gdrive_file_id', sa.String(length=255), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('overlay_jobs')
    # Postgres leaves the enum types behind after drop_table; remove them so a
    # re-upgrade can recreate them cleanly. No-op on SQLite.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name='overlay_status_enum').drop(bind, checkfirst=True)
        sa.Enum(name='overlay_source_type_enum').drop(bind, checkfirst=True)
