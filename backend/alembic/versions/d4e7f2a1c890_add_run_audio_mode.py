"""add_run_audio_mode

Adds audio_mode column to the runs table (TR7).
Operator can choose per-run between:
  "original" — mux the full continuous source audio (current behaviour, default)
  "seedance" — use each clip's own audio to avoid drift when Seedance changes duration

Revision ID: d4e7f2a1c890
Revises: c3f9a2e1b450
Create Date: 2026-06-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e7f2a1c890'
down_revision: Union[str, None] = 'c3f9a2e1b450'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite does not support native ENUMs — the column is stored as VARCHAR.
    # The CHECK constraint is omitted for SQLite compatibility; application-layer
    # validation (API + stitch()) enforces the allowed set.
    op.add_column(
        'runs',
        sa.Column(
            'audio_mode',
            sa.Enum('original', 'seedance', name='run_audio_mode_enum'),
            nullable=False,
            server_default='original',
        ),
    )


def downgrade() -> None:
    op.drop_column('runs', 'audio_mode')
    # Drop the enum type (no-op on SQLite; required on PostgreSQL).
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute('DROP TYPE IF EXISTS run_audio_mode_enum')
