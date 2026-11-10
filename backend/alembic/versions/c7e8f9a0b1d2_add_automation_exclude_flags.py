"""add_automation_exclude_flags

Adds per-field exclude flags to `run_automations`: when set, the corresponding
filter (persona / gender / ethnicity) inverts — rows matching any of the listed
values are SKIPPED instead of selected. Age has no flag (it uses range-aware
matching instead).

Revision ID: c7e8f9a0b1d2
Revises: 1819cbb7dcd9
Create Date: 2026-07-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7e8f9a0b1d2'
down_revision: Union[str, None] = '1819cbb7dcd9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_FLAGS = ('exclude_persona', 'exclude_gender', 'exclude_ethnicity')


def upgrade() -> None:
    for flag in _FLAGS:
        op.add_column(
            'run_automations',
            sa.Column(flag, sa.Boolean(), nullable=False, server_default=sa.text('false')),
        )


def downgrade() -> None:
    for flag in reversed(_FLAGS):
        op.drop_column('run_automations', flag)
