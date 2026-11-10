"""Audio patch kinds: the same-run low-pass fix of a trailing brand sibilant (AICP-389).

Additive. ``run_audio_patches`` gets:

* ``kind`` — ``donor`` (the existing "re-voice a window from a sibling run")
  or ``same_run_lowpass`` (the autopilot's free brand fix). Existing rows are
  backfilled to ``donor`` by the server default.
* ``filter_params`` — the versioned render parameters of a filter kind.
* ``base_sha256`` / ``candidate_sha256`` — the exact bytes the operator
  approved: the base the candidate was rendered from and the candidate itself.
  The promotion re-checks both under the run-row lock.
* ``candidate_path`` — the approved candidate file (inside its compute
  attempt directory); promoted as is, never re-rendered.

``donor_run_id`` and ``donor_start_sec`` become nullable, ONLY for the filter
kind: a CHECK constraint keeps a donor row complete and a filter row fully
bound. No enum change; nothing else reads the new columns.

Revision ID: v3a9c5e7b1d4
Revises: u7c2e4a6b8d1
Create Date: 2026-11-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "v3a9c5e7b1d4"
down_revision: Union[str, None] = "u7c2e4a6b8d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CHECK_NAME = "ck_run_audio_patches_kind_fields"
CHECK_SQL = (
    "(kind = 'donor' AND donor_run_id IS NOT NULL AND donor_start_sec IS NOT NULL)"
    " OR (kind = 'same_run_lowpass' AND filter_params IS NOT NULL"
    " AND base_sha256 IS NOT NULL AND candidate_sha256 IS NOT NULL"
    " AND candidate_path IS NOT NULL)"
)


def upgrade() -> None:
    op.add_column("run_audio_patches", sa.Column(
        "kind", sa.String(32), nullable=False, server_default="donor"))
    op.add_column("run_audio_patches", sa.Column(
        "filter_params", sa.JSON(), nullable=True))
    op.add_column("run_audio_patches", sa.Column("base_sha256", sa.String(64), nullable=True))
    op.add_column("run_audio_patches", sa.Column("candidate_sha256", sa.String(64), nullable=True))
    op.add_column("run_audio_patches", sa.Column("candidate_path", sa.Text(), nullable=True))
    op.alter_column("run_audio_patches", "donor_run_id", existing_type=sa.String(36), nullable=True)
    op.alter_column("run_audio_patches", "donor_start_sec", existing_type=sa.Float(), nullable=True)
    op.create_check_constraint(CHECK_NAME, "run_audio_patches", CHECK_SQL)


def downgrade() -> None:
    # Only safe while no filter row exists (their donor fields are NULL).
    op.drop_constraint(CHECK_NAME, "run_audio_patches", type_="check")
    op.alter_column("run_audio_patches", "donor_start_sec", existing_type=sa.Float(), nullable=False)
    op.alter_column("run_audio_patches", "donor_run_id", existing_type=sa.String(36), nullable=False)
    for col in ("candidate_path", "candidate_sha256", "base_sha256", "filter_params", "kind"):
        op.drop_column("run_audio_patches", col)
