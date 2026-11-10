"""add_derived_creative_sources

v1 hung both derived creatives (tail edits, caption jobs) off the Run's
result and nothing else, so the two could never be combined: you could
caption the result OR re-cut its ending, never both. This lets each one take
the other as its input, mirroring the OverlayJob source_type / soft-reference
precedent (app/overlay_pipeline.py::_resolve_source is the shape copied here).
See docs/burned-captions.md §3.7.

Adds to `run_tail_edits`:
  - source_type ("run" | "caption", tail_edit_source_type_enum), NOT NULL,
    server_default 'run'.
  - source_caption_id — String(36) NULL, soft reference to run_captions.id
    (no FK — a caption job may be deleted independently and the tail edit
    keeps whatever it already resolved).

Adds to `run_captions`:
  - source_type ("run" | "tail_edit", caption_source_type_enum), NOT NULL,
    server_default 'run'.
  - source_tail_edit_id — String(36) NULL, soft reference to
    run_tail_edits.id (no FK, same reasoning).

`run_id` on both tables is untouched and stays NOT NULL in every case: it is
the PARENT RUN — the provenance used for the project id, the language
suffix, the sheet `Idea`, and the output filename — never the video actually
being read as input. Only the new source_type / source_*_id fields say which
file is read. The server_default='run' is what makes this backward
compatible: every existing row keeps meaning exactly what it meant before
this migration — "read the parent run's result" — with no backfill needed.

Revision ID: c2e4f6a8b1d3
Revises: b5d7f9a1c3e6
Create Date: 2026-08-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2e4f6a8b1d3'
down_revision: Union[str, None] = 'b5d7f9a1c3e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TAIL_EDIT_ENUM_NAME = 'tail_edit_source_type_enum'
_TAIL_EDIT_VALUES = ('run', 'caption')

_CAPTION_ENUM_NAME = 'caption_source_type_enum'
_CAPTION_VALUES = ('run', 'tail_edit')


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == 'postgresql'

    tail_edit_enum = sa.Enum(*_TAIL_EDIT_VALUES, name=_TAIL_EDIT_ENUM_NAME)
    caption_enum = sa.Enum(*_CAPTION_VALUES, name=_CAPTION_ENUM_NAME)

    if is_postgres:
        # On Postgres a new enum TYPE must exist before a column can use it.
        # create_table (as in b5d7f9a1c3e6) lets SQLAlchemy emit CREATE TYPE
        # for us implicitly; add_column does not, so create both explicitly
        # first — checkfirst=True so a re-run against a half-applied DB
        # doesn't blow up on the type already existing.
        tail_edit_enum.create(bind, checkfirst=True)
        caption_enum.create(bind, checkfirst=True)
        tail_edit_col_type = sa.Enum(
            *_TAIL_EDIT_VALUES, name=_TAIL_EDIT_ENUM_NAME, create_type=False
        )
        caption_col_type = sa.Enum(
            *_CAPTION_VALUES, name=_CAPTION_ENUM_NAME, create_type=False
        )
    else:
        # SQLite stores enums as VARCHAR with a CHECK constraint; no type
        # object needs to be created up front.
        tail_edit_col_type = tail_edit_enum
        caption_col_type = caption_enum

    op.add_column(
        'run_tail_edits',
        sa.Column(
            'source_type',
            tail_edit_col_type,
            nullable=False,
            server_default='run',
        ),
    )
    op.add_column(
        'run_tail_edits',
        sa.Column('source_caption_id', sa.String(length=36), nullable=True),
    )

    op.add_column(
        'run_captions',
        sa.Column(
            'source_type',
            caption_col_type,
            nullable=False,
            server_default='run',
        ),
    )
    op.add_column(
        'run_captions',
        sa.Column('source_tail_edit_id', sa.String(length=36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('run_captions', 'source_tail_edit_id')
    op.drop_column('run_captions', 'source_type')

    op.drop_column('run_tail_edits', 'source_caption_id')
    op.drop_column('run_tail_edits', 'source_type')

    # Postgres leaves the enum types behind after drop_column; remove them so
    # a re-upgrade can recreate them cleanly. No-op on SQLite.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name=_CAPTION_ENUM_NAME).drop(bind, checkfirst=True)
        sa.Enum(name=_TAIL_EDIT_ENUM_NAME).drop(bind, checkfirst=True)
