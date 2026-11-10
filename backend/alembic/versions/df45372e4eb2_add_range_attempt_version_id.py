"""add_range_attempt_version_id

Follow-up to 73df0ff3f344 (found by the mandatory Cursor CLI review gate
run on that migration's own diff): the DB-level partial unique index
there stops a SECOND, DIFFERENT RunSegmentRangeAttempt from ever
coexisting with an active one, and the Run-row lock added to
pipeline_range_attempt._promote() stops two sessions from both promoting
based on the same base snapshot — but neither one stops two sessions
processing the SAME attempt (a duplicate poll enqueue, or a recovery
re-enqueue racing a still-running job) from corrupting each other via
their own separate, unlocked commits to the attempt row's intermediate
fields (status, keep_recut_paths, gdrive_file_id, ...) before either
reaches promotion. A loser session's stale in-memory status could
silently overwrite a winner's already-committed `done`.

Adds `version_id`, wired as SQLAlchemy's `version_id_col` on
RunSegmentRangeAttempt (see models.py): every UPDATE to this row is now
conditioned on this value, so a session writing from state it read
before another session's commit gets a StaleDataError instead of
silently clobbering it. See pipeline_range_attempt.poll_range_attempt_once
for the handling this pairs with.

Purely additive: existing rows default to version 0; no data rewritten.

Revision ID: df45372e4eb2
Revises: 73df0ff3f344
Create Date: 2026-08-29 20:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'df45372e4eb2'
down_revision: Union[str, None] = '73df0ff3f344'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'run_segment_range_attempts',
        sa.Column('version_id', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('run_segment_range_attempts', 'version_id')
