"""add_run_segment_snapshot

Adds runs.segment_snapshot — a frozen copy of the SegmentDefs a run was
launched against, written by submit_run before the first clip is cut:

    [{"id", "index", "action", "start_sec", "end_sec"}, ...]   (index order)

Why
---
SegmentDefs belong to the Project, not the Run, and operators keep editing them
(splitting a segment, nudging a boundary) while an earlier run is still
processing. The finalize phase read `project.segments` live, so those edits
retroactively changed what it believed the in-flight run contained:

  * a def ADDED mid-run has no RunSegment for that run, so the completeness gate
    scored it as a swap segment that "did not complete" and parked the run at
    `incomplete` — with every segment it actually launched finished and paid
    for. Hit on 2026-08-11 by run 76734005, which generated its full 38.3s and
    was never stitched. Had the gate passed, `_cut_or_lookup` would have
    KeyError'd on the same phantom def at stitch time.
  * a boundary REWRITTEN mid-run silently moved where a "keep" segment gets cut
    from the source, so the stitched timeline no longer matched what the
    operator launched.

Backfill
--------
Existing runs are left NULL rather than backfilled from the current defs: the
current defs are exactly what we cannot trust, and inventing a snapshot from
them would freeze the drift instead of recording the truth. The finalize path
treats NULL as "fall back to project.segments", which is the pre-existing
behaviour, so old runs are unaffected either way.

Revision ID: a3f5b7c9d201
Revises: d7c9e1a3f582
Create Date: 2026-08-11 20:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f5b7c9d201'
down_revision: Union[str, None] = 'd7c9e1a3f582'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('runs', sa.Column('segment_snapshot', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('runs', 'segment_snapshot')
