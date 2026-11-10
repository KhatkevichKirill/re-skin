"""add_range_attempt_claim_lease

Follow-up to df45372e4eb2 (found by the independent audit that requested this
hardening pass): version_id stops a duplicate worker from silently
OVERWRITING another session's committed state, but it does not stop a
duplicate worker from RUNNING the same shared-file/external-call side effects
in the first place — cutting the candidate clip, uploading it, calling
kie.ai's createTask, downloading the result, or re-cutting/stitching/
delivering — before either commit lands. Two workers delivering the same
attempt can both reach ffmpeg/kie.ai/Drive concurrently; version_id only
arbitrates which of their *commits* wins afterwards.

Adds a claim/lease pair used to gate those side effects themselves:

- `claim_owner` — an opaque per-call token. A worker must commit this (via
  optimistic-versioned CAS on the existing `version_id` column — no new
  locking primitive) *before* doing any cut/upload/create-task/download/
  stitch/deliver/promote for the phase it is about to run. A second worker
  finding a live (non-expired) claim exits benignly without touching any
  file or external client.
- `claim_expires_at` — the lease deadline. Only once it has passed may a
  later call safely take the phase over (submit-phase takeover is further
  gated by `submit_billed_at` below).
- `submit_billed_at` — set and committed immediately before the ONE paid
  external call this feature makes (kie.ai's createTask, inside
  submit_range_attempt). kie.ai's jobs API has no idempotency key or
  lookup-by-reference (see kie_client.KieClient.create_task/get_task), so a
  worker that crashes between this commit and persisting the resulting
  `seedance_task_id` leaves the row in a state nobody can safely resolve
  automatically: retrying risks a second billed generation, and giving up
  risks discarding one that already succeeded. A later call finding this set
  with no `seedance_task_id`, past the claim's lease, marks the attempt
  `failed` with an explicit message instead of guessing either way — see
  pipeline_range_attempt.py and docs/run-local-segment-range-regeneration.md.

Purely additive: existing rows default to NULL on all three columns, which
is a safe, deterministic, immediately-recoverable state — "no claim is
held", so the very next worker to touch a pre-migration active row claims it
fresh, exactly like a brand new attempt. No data is rewritten. No existing
constraint, index, or enum value is touched.

Downgrade drops all three columns. Any attempt that is sitting in the
submission-ambiguous `failed` state at downgrade time becomes indistinguishable
from an ordinary failed attempt — its forensic "this may have double-billed"
marker is lost, but it was already terminal and non-actionable by automation
either way, so this is a diagnostics-only regression, not a correctness one.

Revision ID: bc2e47aad2a2
Revises: df45372e4eb2
Create Date: 2026-08-30 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bc2e47aad2a2'
down_revision: Union[str, None] = 'df45372e4eb2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'run_segment_range_attempts',
        sa.Column('claim_owner', sa.String(length=64), nullable=True),
    )
    op.add_column(
        'run_segment_range_attempts',
        sa.Column('claim_expires_at', sa.DateTime(), nullable=True),
    )
    op.add_column(
        'run_segment_range_attempts',
        sa.Column('submit_billed_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('run_segment_range_attempts', 'submit_billed_at')
    op.drop_column('run_segment_range_attempts', 'claim_expires_at')
    op.drop_column('run_segment_range_attempts', 'claim_owner')
