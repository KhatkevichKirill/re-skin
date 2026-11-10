"""add_localisation_project_type

Everything the `localisation` project type needs from the schema, in one
migration (see docs/localisation.md §3).

1. project_type_enum gains the label 'localisation' — a project whose hook is
   re-spoken in another language by the video model while the rest of the
   source is discarded and a language-matched tail is stitched on. No-op on
   SQLite, which stores ENUMs as VARCHAR and has no type object to alter.

New columns on `video_projects` (all nullable, no server_default — they are
meaningless for the three pre-existing project types, so every existing row
must stay NULL rather than be backfilled):

  - hook_sec          DOUBLE PRECISION NULL — length of the localised hook.
                      NULL means settings.LOCALISATION_DEFAULT_HOOK_SEC, so the
                      default stays a single source of truth in Python. Read at
                      ANALYZE time, exactly like max_segment_sec: hook_split
                      segmentation cuts swap[0, hook_sec] + keep[hook_sec, end]
                      once, and changing the value afterwards does not re-cut
                      anything — the project has to be re-analyzed.
  - transcript        JSON NULL — the cached docs/localisation.md §4.1 dict
                      (detected source_language + timestamped, speaker-labelled
                      verbatim lines). On the project rather than the run
                      because it describes the uploaded footage: one expensive
                      video-model call is reused by every run and every target
                      language. JSON, not a table — nothing queries into the
                      lines, they are read back whole and fed to the prompt.
  - transcript_status VARCHAR(16) NULL — 'pending' | 'running' | 'ready' |
                      'failed' | 'empty'. NULL = never requested. 'empty' is a
                      success (no speech in the hook is legal), 'failed' is
                      non-fatal (the operator pastes a translation by hand).
                      A plain VARCHAR and not an enum type on purpose: this is
                      a side-channel task's own state, nothing gates on it, and
                      adding a state should not need an irreversible
                      `ALTER TYPE ... ADD VALUE`.
  - transcript_error  TEXT NULL — why the last attempt failed; carries raw
                      provider error bodies, hence TEXT rather than VARCHAR(n).

New column on `library_clips`:

  - language          VARCHAR(8) NULL — one of app/languages.py's codes. NULL
                      means language-agnostic, which is what every existing
                      clip is. Only drives the New Run form's automatic tail
                      pick; the stitch phase never branches on it, so a NULL or
                      mismatched clip behaves exactly as it does today.

Revision ID: a4c8e2b6d139
Revises: b3d5f7a9c142
Create Date: 2026-08-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4c8e2b6d139'
down_revision: Union[str, None] = 'b3d5f7a9c142'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # Postgres 12+ allows ALTER TYPE ... ADD VALUE inside the transaction
        # Alembic wraps the migration in; IF NOT EXISTS keeps it idempotent
        # across re-runs. SQLite has no enum type object, so it is skipped.
        op.execute(
            "ALTER TYPE project_type_enum ADD VALUE IF NOT EXISTS 'localisation'"
        )

    # Run on both dialects.
    op.add_column(
        'video_projects',
        sa.Column('hook_sec', sa.Float(), nullable=True),
    )
    op.add_column(
        'video_projects',
        sa.Column('transcript', sa.JSON(), nullable=True),
    )
    op.add_column(
        'video_projects',
        sa.Column('transcript_status', sa.String(length=16), nullable=True),
    )
    op.add_column(
        'video_projects',
        sa.Column('transcript_error', sa.Text(), nullable=True),
    )
    op.add_column(
        'library_clips',
        sa.Column('language', sa.String(length=8), nullable=True),
    )


def downgrade() -> None:
    # Only the columns come back off. Postgres cannot remove a value from an
    # enum type without recreating it, so 'localisation' stays in
    # project_type_enum — harmless, since rolling back the code that writes it
    # leaves no row referencing the label.
    op.drop_column('library_clips', 'language')
    op.drop_column('video_projects', 'transcript_error')
    op.drop_column('video_projects', 'transcript_status')
    op.drop_column('video_projects', 'transcript')
    op.drop_column('video_projects', 'hook_sec')
