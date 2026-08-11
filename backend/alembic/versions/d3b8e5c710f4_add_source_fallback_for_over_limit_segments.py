"""add source_fallback for over-limit swap segments

Makes the "segment is too long for this run's model" case degrade gracefully
instead of killing the run.

Background. VideoProject.max_segment_sec (migration c8d2f4a6e971) lets a project
be segmented up to 30s so Seedance 2.5 can be used to its full length. The model
is still chosen per-Run, so a project cut at 30s can be run on a 15s model. Until
now that combination marked each over-long segment `failed`, and the completeness
gate in pipeline_v2.process_run — which refuses to stitch a mix of swapped and
un-swapped clips — then left the whole run `incomplete` with no video at all.

That is the wrong trade for THIS failure specifically, because it is
deterministic: the segment can never generate on that model, so "retry it" is not
a fix and blocking is permanent. Substituting the original footage for those
segments is safe on the timeline (the original clip has exactly the segment's
duration, unlike a partial swap, so nothing desyncs against the soundtrack).

Transient failures — task error, timeout, missing result url — deliberately keep
the old behaviour: they still leave the run `incomplete`, because a retry
produces the real swap and silently shipping the original instead would throw
that away.

Two columns:

  - run_segments.source_fallback  BOOLEAN NOT NULL DEFAULT false
        Marks a segment whose swap was skipped for a deterministic reason and
        whose original footage is therefore in the delivered video. NOT NULL with
        a server_default so every existing row backfills to false — no existing
        run is retroactively reinterpreted as partly un-swapped.

  - runs.source_fallback_segments  INTEGER NULL
        Count of the above per run, written once at stitch time. Denormalised on
        purpose: a `done` run whose video is only partly swapped has to be
        visible in the runs list without a per-row segment query. Nullable
        because NULL ("not recorded" — run predates this column, or never
        stitched) is meaningfully different from 0 ("checked, fully swapped"), so
        there is deliberately no backfill.

Revision ID: d3b8e5c710f4
Revises: c8d2f4a6e971
Create Date: 2026-08-10 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3b8e5c710f4'
down_revision: Union[str, None] = 'c8d2f4a6e971'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOT NULL + server_default: existing rows backfill to false, so no run
    # already in the DB starts claiming it delivered original footage.
    op.add_column(
        'run_segments',
        sa.Column(
            'source_fallback',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Nullable with NO server_default: NULL means "not recorded", which is not
    # the same as 0 ("stitched and fully swapped"). Backfilling to 0 would claim
    # we had verified pre-existing runs when we had not.
    op.add_column(
        'runs',
        sa.Column('source_fallback_segments', sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('runs', 'source_fallback_segments')
    op.drop_column('run_segments', 'source_fallback')
