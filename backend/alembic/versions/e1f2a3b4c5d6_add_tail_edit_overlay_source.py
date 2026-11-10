"""add_tail_edit_overlay_source

Adds "tail_edit" as a valid OverlayJob.source_type: a completed tail edit's
own result (from the "Change ending" tool) can now be used as an overlay
source, alongside a Run's result or a directly-uploaded video.

  overlay_source_type_enum        — add value 'tail_edit' (Postgres only;
      SQLite stores the Enum as VARCHAR and does not enforce it at the DB
      level, so no action is needed there).
  overlay_jobs.source_tail_edit_id — nullable soft reference to
      run_tail_edits.id (no FK — mirrors source_run_id): set when
      source_type == 'tail_edit'.

Revision ID: e1f2a3b4c5d6
Revises: c1d2e3f4a5b6
Create Date: 2026-07-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # ALTER TYPE ... ADD VALUE must be run outside a transaction block on
        # older Postgres, but Postgres 12+ allows it inside a transaction.
        # Alembic wraps migrations in a transaction; use execute() directly.
        op.execute("ALTER TYPE overlay_source_type_enum ADD VALUE IF NOT EXISTS 'tail_edit'")
    op.add_column(
        'overlay_jobs',
        sa.Column('source_tail_edit_id', sa.String(length=36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('overlay_jobs', 'source_tail_edit_id')
    # Postgres does not support removing values from an enum type without
    # recreating it — a safe downgrade is not possible. Leave as a no-op.
