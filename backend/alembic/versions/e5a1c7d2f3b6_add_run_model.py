"""add_run_model

Adds the model column to the runs table so a Run can target one of two kie.ai
generation backends:
  "seedance"    — bytedance/seedance-2 (current behaviour, default)
  "gemini-omni" — gemini-omni-video

The resolution column is NOT altered here: it has no CHECK constraint on SQLite
(non-native enum), so the new "4k" value Gemini supports is governed purely by
application-layer validation — same convention as the audio_mode migration.

Revision ID: e5a1c7d2f3b6
Revises: d4e7f2a1c890
Create Date: 2026-06-22 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5a1c7d2f3b6'
down_revision: Union[str, None] = 'd4e7f2a1c890'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite does not support native ENUMs — the column is stored as VARCHAR.
    # The CHECK constraint is omitted for SQLite compatibility; application-layer
    # validation (API) enforces the allowed set.
    op.add_column(
        'runs',
        sa.Column(
            'model',
            sa.Enum('seedance', 'gemini-omni', name='run_model_enum'),
            nullable=False,
            server_default='seedance',
        ),
    )


def downgrade() -> None:
    op.drop_column('runs', 'model')
    # Drop the enum type (no-op on SQLite; required on PostgreSQL).
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute('DROP TYPE IF EXISTS run_model_enum')
