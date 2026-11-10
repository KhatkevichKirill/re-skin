"""add_caption_overlay_source

Adds "caption" as a valid OverlayJob.source_type: a done caption job's burned
result can now be used as an overlay source, alongside a Run's result, a tail
edit's output, or a directly-uploaded video. Completes the grid — every derived
creative a run can produce is now something an overlay can be burned onto.

  overlay_source_type_enum      — add value 'caption' (Postgres only; SQLite
      stores the Enum as VARCHAR and does not enforce it at the DB level, so
      no action is needed there).
  overlay_jobs.source_caption_id — nullable soft reference to run_captions.id
      (no FK — mirrors source_run_id / source_tail_edit_id): set when
      source_type == 'caption'.
  overlay_jobs.parent_run_id     — denormalized "which Run is this overlay's
      creative", indexed, backfilled below.

Shape copied from e1f2a3b4c5d6, which did exactly this for 'tail_edit'.

Why parent_run_id, and why now
------------------------------
The run page and the Final Creatives export both ask "which run does this
overlay belong to?", and until now that was answered by walking the soft
reference: source_run_id, or source_tail_edit_id -> run_tail_edits.run_id.
Tail edits and caption jobs are HARD-deleted by their operator Delete buttons,
so that walk dead-ends: the overlay silently vanished from its run's history
(unfindable, unexportable) and stopped counting toward
overlay_runs.count_overlay_jobs_for_run — which is what stops a later overlay
from reusing the bare run id and colliding with the sheet row the vanished one
already wrote. Storing the parent at create time makes the link survive the
source being deleted. Backfill covers every existing row; the resolver still
falls back to the old walk for anything created in the window between this
migration and the code deploy.

Revision ID: d7c9e1a3f582
Revises: c2e4f6a8b1d3
Create Date: 2026-08-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd7c9e1a3f582'
down_revision: Union[str, None] = 'c2e4f6a8b1d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # Postgres 12+ allows ALTER TYPE ... ADD VALUE inside Alembic's
        # transaction; IF NOT EXISTS keeps a re-run idempotent.
        op.execute("ALTER TYPE overlay_source_type_enum ADD VALUE IF NOT EXISTS 'caption'")
    op.add_column(
        'overlay_jobs',
        sa.Column('source_caption_id', sa.String(length=36), nullable=True),
    )
    op.add_column(
        'overlay_jobs',
        sa.Column('parent_run_id', sa.String(length=36), nullable=True),
    )
    op.create_index(
        'ix_overlay_jobs_parent_run_id', 'overlay_jobs', ['parent_run_id']
    )

    # Backfill: resolve the parent exactly as the old walk did, so every row
    # that predates this column answers the same as before — just without
    # depending on the source row still existing. Correlated subqueries work
    # identically on SQLite and Postgres. There is no caption clause: no
    # caption-sourced overlay can exist yet (this migration is what allows
    # one).
    op.execute(
        """
        UPDATE overlay_jobs
           SET parent_run_id = source_run_id
         WHERE source_run_id IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE overlay_jobs
           SET parent_run_id = (
                   SELECT run_id FROM run_tail_edits
                    WHERE run_tail_edits.id = overlay_jobs.source_tail_edit_id
               )
         WHERE source_tail_edit_id IS NOT NULL
           AND parent_run_id IS NULL
        """
    )


def downgrade() -> None:
    op.drop_index('ix_overlay_jobs_parent_run_id', table_name='overlay_jobs')
    op.drop_column('overlay_jobs', 'parent_run_id')
    op.drop_column('overlay_jobs', 'source_caption_id')
    # Postgres does not support removing values from an enum type without
    # recreating it — a safe downgrade is not possible. Leave as a no-op.
