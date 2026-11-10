"""add_copy_run_preprocess

Copy run (and its automation equivalent) can now re-run the source run's GPT
Image 2 preprocessing on a NEW character photo instead of handing that raw
photo to Seedance: the copy clones the template's frames and image prompt and
swaps in the new face. Two flags carry that:

1. `runs.preprocess_auto_approve` — the copy skips the `preprocess_review`
   pause and submits as soon as GPT Image 2 succeeds. The recipe (frames +
   prompt) was already approved by hand on the source run; only the photo
   changed, so there is no new decision to make. A FAILED preprocess task
   still fails the run.
2. `run_automations.preprocess_refs` — the same behaviour for the daily
   sheet-driven batches, so each persona's photo goes through GPT Image 2
   before Seedance sees it.

Both NOT NULL DEFAULT FALSE with no backfill: every existing run and
automation keeps today's behaviour exactly (manual review, raw photo).

Revision ID: c9f2a4e6b807
Revises: b7d9f1a3c5e8
Create Date: 2026-08-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9f2a4e6b807'
down_revision: Union[str, None] = 'b7d9f1a3c5e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'runs',
        sa.Column(
            'preprocess_auto_approve',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        'run_automations',
        sa.Column(
            'preprocess_refs',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column('run_automations', 'preprocess_refs')
    op.drop_column('runs', 'preprocess_auto_approve')
