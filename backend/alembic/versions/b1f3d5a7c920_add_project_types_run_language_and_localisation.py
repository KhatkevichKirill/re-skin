"""add project types, run language and the localisation columns

Everything the `localisation` project type needs, in one migration because on
this branch it all arrives together. (In the private repo the same ground is
covered by four separate migrations — project types, mute_source, run language,
then localisation — because each shipped as its own feature.)

New Postgres type
-----------------
`project_type_enum` with all four values: 'face_swap' | 'subtitle_removal' |
'cover_change' | 'localisation'. Mirrors app/project_types.PROJECT_TYPES, which
is the single source of truth; adding a type later means one ALTER TYPE and one
spec. Created with a plain CREATE TYPE rather than autogenerate so the labels are
visible here.

New columns on `video_projects`
------------------------------
  - project_type       project_type_enum NOT NULL DEFAULT 'face_swap'
        NOT NULL with a server_default so every existing project backfills to the
        face-swap flow — i.e. keeps exactly the behaviour it has today, since that
        was the only flow before this migration.

  - mute_source        BOOLEAN NOT NULL DEFAULT false
        "Remove original audio": swap clips are uploaded video-only so the model
        writes audio from the prompt instead of reacting to the source track.
        Backfills to false = today's behaviour.

  - hook_sec           DOUBLE PRECISION NULL
        Localisation only: how much of the front of the video is the hook. NULL
        means settings.LOCALISATION_DEFAULT_HOOK_SEC, so the default stays in
        Python rather than being frozen into the schema, and nothing needs a
        backfill. Consumed at ANALYZE time, like max_segment_sec.

  - transcript         JSON NULL          (docs/localisation.md §4.1)
  - transcript_status  VARCHAR(16) NULL   pending|running|ready|failed|empty
  - transcript_error   TEXT NULL
        The cached transcript and its side-channel task state. NULL status means
        "never requested". Plain VARCHAR, not an enum: nothing gates on it and a
        new state should not need an irreversible ALTER TYPE.

New column on `runs`
--------------------
  - language           VARCHAR(8) NULL
        One of app/languages.py's codes. On a localisation run this is the
        translation target; elsewhere it is descriptive metadata. VARCHAR so
        adding a language is a registry edit, not a migration. NULL behaves as
        English, which is what every pre-existing row already is.

Downgrade drops all seven columns and the enum type. Safe here (unlike an
ALTER TYPE ... ADD VALUE, which Postgres cannot reverse) because the type is
created by this migration and nothing outside it references the type.

Revision ID: b1f3d5a7c920
Revises: d3b8e5c710f4
Create Date: 2026-08-10 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1f3d5a7c920'
down_revision: Union[str, None] = 'd3b8e5c710f4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_PROJECT_TYPES = ('face_swap', 'subtitle_removal', 'cover_change', 'localisation')


def upgrade() -> None:
    bind = op.get_bind()
    enum_type = sa.Enum(*_PROJECT_TYPES, name='project_type_enum')
    if bind.dialect.name == 'postgresql':
        # Create the type explicitly (checkfirst so a re-run against a
        # half-applied DB is a no-op), then tell add_column not to create it
        # again.
        enum_type.create(bind, checkfirst=True)
        col_type = sa.Enum(*_PROJECT_TYPES, name='project_type_enum', create_type=False)
    else:
        # SQLite stores enums as VARCHAR with a CHECK constraint; no type object.
        col_type = enum_type

    op.add_column(
        'video_projects',
        sa.Column(
            'project_type',
            col_type,
            nullable=False,
            server_default='face_swap',
        ),
    )
    op.add_column(
        'video_projects',
        sa.Column(
            'mute_source', sa.Boolean(), nullable=False,
            server_default=sa.false(),
        ),
    )
    # The four nullable localisation columns: no server_default, because NULL is
    # a meaningful value for each of them ("use the Python default" / "never
    # requested") rather than a missing one.
    op.add_column('video_projects', sa.Column('hook_sec', sa.Float(), nullable=True))
    op.add_column('video_projects', sa.Column('transcript', sa.JSON(), nullable=True))
    op.add_column(
        'video_projects',
        sa.Column('transcript_status', sa.String(length=16), nullable=True),
    )
    op.add_column(
        'video_projects', sa.Column('transcript_error', sa.Text(), nullable=True)
    )

    op.add_column('runs', sa.Column('language', sa.String(length=8), nullable=True))


def downgrade() -> None:
    op.drop_column('runs', 'language')
    op.drop_column('video_projects', 'transcript_error')
    op.drop_column('video_projects', 'transcript_status')
    op.drop_column('video_projects', 'transcript')
    op.drop_column('video_projects', 'hook_sec')
    op.drop_column('video_projects', 'mute_source')
    op.drop_column('video_projects', 'project_type')
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute('DROP TYPE IF EXISTS project_type_enum')
