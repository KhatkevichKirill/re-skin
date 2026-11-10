"""add_tail_edit_export_fields

Adds export-tracking fields to run_tail_edits, mirroring runs/overlay_jobs:

  final_creatives_exported_at — nullable timestamp set when a tail edit's
      result has been logged to the "Final Creatives" tracking spreadsheet.
      NULL = not yet added; a value drives the "already added" UI state.
  caption_demo_status — nullable string recording the Caption Demo action
      taken for a delivered tail edit:
        "queued"          — a row was appended to the "Caption Demo" sheet tab.
        "added_to_table"  — the tail edit was already in the Final Creatives
                             sheet, so it is only marked here (no Caption Demo
                             row written).
      NULL = not acted on.

Revision ID: c1d2e3f4a5b6
Revises: 19338f2b4ebc
Create Date: 2026-07-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1d2e3f4a5b6'
down_revision: Union[str, None] = '19338f2b4ebc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'run_tail_edits',
        sa.Column('final_creatives_exported_at', sa.DateTime(), nullable=True),
    )
    op.add_column(
        'run_tail_edits',
        sa.Column('caption_demo_status', sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('run_tail_edits', 'caption_demo_status')
    op.drop_column('run_tail_edits', 'final_creatives_exported_at')
