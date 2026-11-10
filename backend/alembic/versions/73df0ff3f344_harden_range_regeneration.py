"""harden_range_regeneration

Follow-up hardening for run-local segment range regeneration (43c4cab61c52),
closing two structural gaps found by an independent audit before merge:

1. ``run_segments.segment_def_id`` had ``ON DELETE CASCADE`` against the
   mutable ``segment_defs`` table. A later Project re-analysis or bulk
   segment edit that deletes a live ``SegmentDef`` therefore cascade-deleted
   every ``RunSegment`` that ever referenced it — including a TERMINAL run's
   rows, and with them every ``RunSegmentRangeAttempt`` correction history —
   even though ``Run.segment_snapshot`` is the run's own frozen, authoritative
   timeline and no longer needs the live row. It also meant a Copy/Automation
   launch built from such a run could fail on insert (a fresh RunSegment
   naming a segment_def_id whose live row is gone violates the FK). Dropping
   the FK removes this destructive dependency; the column stays as the same
   stable identity key, just without a DB-level referential constraint to a
   table it must now safely outlive. See app/models.py's RunSegment.
   segment_def_id docstring.

2. The "at most one active RunSegmentRangeAttempt per run" invariant was
   enforced only by a non-locking application SELECT
   (api_v2._active_range_attempt), so two concurrent create requests could
   both observe no active row and both create/enqueue an attempt — a real
   double-billing risk. A partial unique index makes the invariant a DB
   constraint: at most one row per run_id may be in a non-terminal status
   at a time, on every dialect this app runs on. See api_v2.create_range_rerun
   for the IntegrityError handling this pairs with, and
   pipeline_range_attempt._promote for the companion SELECT ... FOR UPDATE
   fix to the promotion race (no schema change needed for that half).

No data is rewritten and no existing row is touched by either change.

Revision ID: 73df0ff3f344
Revises: 43c4cab61c52
Create Date: 2026-08-29 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '73df0ff3f344'
down_revision: Union[str, None] = '43c4cab61c52'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Must match state_machine.RANGE_ATTEMPT_ACTIVE_STATUSES exactly — this is
# the DB-level mirror of that Python set, not derived from it (Alembic
# migrations must not import app code, so the values are duplicated here by
# hand; a future change to the active-status set needs a matching migration).
_ACTIVE_STATUSES = ('queued', 'generating', 'stitching', 'delivering')
_ACTIVE_UNIQUE_INDEX = 'uq_run_segment_range_attempts_active_per_run'
_ACTIVE_WHERE = sa.text(
    "status IN ('" + "','".join(_ACTIVE_STATUSES) + "')"
)


def upgrade() -> None:
    # --- Finding 1: decouple run_segments.segment_def_id from segment_defs ---
    # Constraint name confirmed empirically against a fresh apply of every
    # prior migration on disposable PostgreSQL 15 (Postgres's own default
    # single-column FK naming: "<table>_<column>_fkey" — this table has no
    # explicitly-named constraints anywhere else in its history, so nothing
    # here relies on a naming_convention that isn't actually configured).
    with op.batch_alter_table('run_segments') as batch_op:
        batch_op.drop_constraint(
            'run_segments_segment_def_id_fkey', type_='foreignkey',
        )
    # The column, its NOT NULL and its plain index are all left exactly as
    # they were — only the referential constraint to segment_defs is gone.

    # --- Finding 2: DB-level one-active-attempt-per-run invariant ---
    # Additive alongside the existing plain ix_..._run_id index (unchanged,
    # still used by full-history queries across every status); this one only
    # ever has at most one matching row per run_id, by construction.
    op.create_index(
        _ACTIVE_UNIQUE_INDEX,
        'run_segment_range_attempts',
        ['run_id'],
        unique=True,
        postgresql_where=_ACTIVE_WHERE,
        sqlite_where=_ACTIVE_WHERE,
    )


def downgrade() -> None:
    op.drop_index(
        _ACTIVE_UNIQUE_INDEX,
        table_name='run_segment_range_attempts',
        postgresql_where=_ACTIVE_WHERE,
        sqlite_where=_ACTIVE_WHERE,
    )

    # Re-add the FK. This will fail if any run_segments.segment_def_id row
    # written since the upgrade no longer matches a live segment_defs.id —
    # e.g. a re-analysis that ran while this migration was applied, which is
    # exactly the scenario the upgrade exists to make safe. That failure is
    # the correct behaviour for a downgrade: it means data now genuinely
    # depends on the relaxed constraint, so silently dropping those rows'
    # integrity instead would be the destructive choice. Recovering from it
    # is an operator decision (re-analyze again to reconcile, or accept the
    # branch stays on the new schema), not something this migration should
    # guess at.
    with op.batch_alter_table('run_segments') as batch_op:
        batch_op.create_foreign_key(
            'run_segments_segment_def_id_fkey',
            'segment_defs',
            ['segment_def_id'], ['id'],
            ondelete='CASCADE',
        )
