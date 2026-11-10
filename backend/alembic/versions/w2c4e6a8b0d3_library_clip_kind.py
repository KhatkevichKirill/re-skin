"""Library clip kind: end card vs insert (AICP-381).

Additive. ``library_clips.kind`` (VARCHAR(16) NOT NULL, server default
``insert``, CHECK ``kind IN ('end_card', 'insert')``) says what a clip is for.
A rebuild's default end card used to be "the newest library clip in the run's
language", which picked the 27.8 s demo ``ip_reader`` on run 813b251f; it is
now the newest language-matched ``end_card``.

Data: every existing row becomes ``insert`` through the server default, then
the short ``CTA_*`` cards become ``end_card``. On prod (read 2026-11-05) that
is CTA_English / CTA_Spanish / CTA_Portuguese / CTA_Poland_2 (7.0 s) and
CTA_Poland (5.2 s). ``CTA_Spanish_Book`` / ``CTA_Portuguese_Book`` (24.6 /
24.8 s: an audiobook scene with the card at its end) stay ``insert`` — the
length guard keeps them out, because swapping one in for a source card would
repeat the 813b251f over-length final. Operators can change any row on
/v2/library.

Nothing else reads the column: the old image ignores it (and its INSERTs get
the server default), so applying this before the image is safe, and so is
the old image running on top of it.

Revision ID: w2c4e6a8b0d3
Revises: w4c8e2a6d1f3
Create Date: 2026-11-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "w2c4e6a8b0d3"
down_revision: Union[str, None] = "w4c8e2a6d1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CHECK_NAME = "ck_library_clips_kind"
#: A CTA clip longer than this is a scene with a card, not an end card.
END_CARD_MAX_SEC = 10.0


def upgrade() -> None:
    op.add_column("library_clips", sa.Column(
        "kind", sa.String(16), nullable=False, server_default="insert"))
    op.create_check_constraint(CHECK_NAME, "library_clips", "kind IN ('end_card', 'insert')")
    op.execute(sa.text(
        "UPDATE library_clips SET kind = 'end_card' "
        "WHERE name LIKE 'CTA\\_%' ESCAPE '\\' "
        "AND duration_sec IS NOT NULL AND duration_sec <= :max_sec"
    ).bindparams(max_sec=END_CARD_MAX_SEC))


def downgrade() -> None:
    op.drop_constraint(CHECK_NAME, "library_clips", type_="check")
    op.drop_column("library_clips", "kind")
