"""create_run_audio_patches

Creates the `run_audio_patches` table for the "Fix audio from another run"
tool: an operator picks a window of a done Run's delivered final and a donor
run whose audio is correct there, and a worker splices the donor's audio into
that window — video stream copied, no sample outside the window moved
(app/media.py::splice_audio_window) — then replaces the run's own result and
the CONTENT behind its existing Drive id.

Deliberately not a derived creative: no Job ID, no sheet row, no new Drive
file, so nothing about the Final Creatives export, Sent-to-Final, or the
tail-edit/caption/overlay chain needs to learn about it. See
docs/run-audio-patch.md and app/pipeline_audio_patch.py.

The audio patch status lifecycle (audio_patch_status_enum, created implicitly
by the sa.Enum column on Postgres; VARCHAR on SQLite):
  queued -> processing -> delivering -> done
  queued/processing/delivering -> failed
  `done` and `failed` are TERMINAL — a further correction is a new row.

Also creates the DB-level "at most one active patch per run" partial unique
index. The application check (api_v2._active_audio_patch) is a non-locking
SELECT that two concurrent create requests can both pass; this index is what
actually enforces the invariant, and it matters more here than for a tail edit
because promotion mutates the parent Run.

Purely additive: creates one new table and one index, touches no existing row.

Revision ID: e4b7c1a9d260
Revises: bc2e47aad2a2
Create Date: 2026-09-07 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4b7c1a9d260'
down_revision: Union[str, None] = 'bc2e47aad2a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Must match state_machine.AUDIO_PATCH_ACTIVE_STATUSES and the sqlite_where /
# postgresql_where on models.RunAudioPatch.__table_args__ exactly. Duplicated
# by hand in all three places because an Alembic migration must not import app
# code — a future change to the active-status set needs a matching migration.
_ACTIVE_STATUSES = ('queued', 'processing', 'delivering')
_ACTIVE_UNIQUE_INDEX = 'uq_run_audio_patches_active_per_run'
_ACTIVE_WHERE = sa.text(
    "status IN ('" + "','".join(_ACTIVE_STATUSES) + "')"
)


def upgrade() -> None:
    op.create_table(
        'run_audio_patches',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('version_id', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        # Soft reference to runs.id — no FK on purpose: the donor is read once,
        # during processing, and deleting it later must not cascade away the
        # record of what was done to the patched run's audio.
        sa.Column('donor_run_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('target_start_sec', sa.Float(), nullable=False),
        sa.Column('target_end_sec', sa.Float(), nullable=False),
        sa.Column('donor_start_sec', sa.Float(), nullable=False),
        sa.Column('crossfade_sec', sa.Float(), nullable=False),
        sa.Column('base_result_path', sa.Text(), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'queued', 'processing', 'delivering', 'done', 'failed',
                name='audio_patch_status_enum',
            ),
            nullable=False,
            server_default='queued',
        ),
        sa.Column('output_path', sa.Text(), nullable=True),
        sa.Column('result_duration_sec', sa.Float(), nullable=True),
        sa.Column('gdrive_file_id', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('promoted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_run_audio_patches_run_id'),
        'run_audio_patches', ['run_id'], unique=False,
    )
    op.create_index(
        op.f('ix_run_audio_patches_created_at'),
        'run_audio_patches', ['created_at'], unique=False,
    )
    op.create_index(
        _ACTIVE_UNIQUE_INDEX,
        'run_audio_patches',
        ['run_id'],
        unique=True,
        postgresql_where=_ACTIVE_WHERE,
        sqlite_where=_ACTIVE_WHERE,
    )


def downgrade() -> None:
    op.drop_index(
        _ACTIVE_UNIQUE_INDEX,
        table_name='run_audio_patches',
        postgresql_where=_ACTIVE_WHERE,
        sqlite_where=_ACTIVE_WHERE,
    )
    op.drop_index(op.f('ix_run_audio_patches_created_at'), table_name='run_audio_patches')
    op.drop_index(op.f('ix_run_audio_patches_run_id'), table_name='run_audio_patches')
    op.drop_table('run_audio_patches')
    # Postgres keeps an enum type created implicitly by a table column when the
    # table goes; drop it explicitly so a re-upgrade does not fail on "type
    # already exists". Mirrors f2c4d6e8a012's handling of overlay_status_enum.
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        sa.Enum(name='audio_patch_status_enum').drop(bind, checkfirst=True)
