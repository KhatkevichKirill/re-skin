"""add_project_type_and_source_run

Adds two columns to video_projects.

1. project_type — the flow an operator is running on a video:

  face_swap         the classic InsightFace-detected character swap (default,
                    and what every pre-existing project is backfilled to)
  subtitle_removal  no face detection; analysis lays down one full-length
                    "keep" segment, New Run defaults to Gemini Omni
  cover_change      no face detection; New Run defaults to Seedance Fast

The column is NOT NULL with a server_default of 'face_swap', so existing rows
are backfilled by the DDL itself and inserts from older code keep working.
See app/project_types.py for the registry these values mirror.

2. source_run_id — nullable soft reference to runs.id, set when the project was
   started from a finished run's final video ("Move to new project"). No FK, so
   deleting the run leaves the project intact (it owns its own copy of the file).

Revision ID: a2c4e6f8b135
Revises: f7a9c1e3b586
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2c4e6f8b135'
down_revision: Union[str, None] = 'f7a9c1e3b586'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ENUM_NAME = 'project_type_enum'
_VALUES = ('face_swap', 'subtitle_removal', 'cover_change')


def upgrade() -> None:
    bind = op.get_bind()
    enum_type = sa.Enum(*_VALUES, name=_ENUM_NAME)
    if bind.dialect.name == 'postgresql':
        # create_type=False on add_column below — create it explicitly first so
        # a re-run against a half-applied DB doesn't blow up on the type.
        enum_type.create(bind, checkfirst=True)
        col_type = sa.Enum(*_VALUES, name=_ENUM_NAME, create_type=False)
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
        sa.Column('source_run_id', sa.String(length=36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('video_projects', 'source_run_id')
    op.drop_column('video_projects', 'project_type')
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name=_ENUM_NAME).drop(bind, checkfirst=True)
