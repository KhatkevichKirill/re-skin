"""drop_v1_jobs_segments

Removes the legacy v1 tables (`segments`, then `jobs`) now that the v1 Job
pipeline has been deleted from the codebase. Also drops the v1-only Postgres
enum types (v2 uses distinct enum names, so these are unused).

Irreversible in practice (v1 code is gone), but downgrade recreates the empty
tables + enums so an operator can roll the *schema* back if needed.

Revision ID: c9e2f4a6b810
Revises: b7d1e9f3a204
Create Date: 2026-07-01 04:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9e2f4a6b810'
down_revision: Union[str, None] = 'b7d1e9f3a204'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# v1-only enum types (v2 uses run_*/project_*/segment_def_* names instead).
_V1_ENUMS = (
    "job_status_enum",
    "segment_status_enum",
    "segment_action_enum",
    "source_type_enum",
    "resolution_enum",
)


def upgrade() -> None:
    bind = op.get_bind()
    # segments FK-references jobs (ondelete CASCADE) → drop the child first.
    op.drop_index(op.f('ix_segments_job_id'), table_name='segments')
    op.drop_table('segments')
    op.drop_table('jobs')
    # Drop the now-orphaned enum types (Postgres only; SQLite has no enum types).
    if bind.dialect.name == "postgresql":
        for enum_name in _V1_ENUMS:
            op.execute(f'DROP TYPE IF EXISTS {enum_name}')


def downgrade() -> None:
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('source_type', sa.Enum('upload', 'gdrive', name='source_type_enum'), nullable=False),
        sa.Column('source_ref', sa.Text(), nullable=False),
        sa.Column('source_local_path', sa.Text(), nullable=True),
        sa.Column('duration_sec', sa.Float(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('aspect_ratio', sa.String(length=20), nullable=True),
        sa.Column('default_prompt', sa.Text(), nullable=False),
        sa.Column('default_reference_image_urls', sa.JSON(), nullable=True),
        sa.Column('resolution', sa.Enum('480p', '720p', '1080p', name='resolution_enum'), nullable=False),
        sa.Column('gdrive_folder_id', sa.String(length=255), nullable=True),
        sa.Column('status', sa.Enum('created', 'analyzing', 'review', 'queued', 'processing', 'stitching', 'delivering', 'done', 'failed', name='job_status_enum'), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('result_local_path', sa.Text(), nullable=True),
        sa.Column('result_gdrive_file_id', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'segments',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('job_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('index', sa.Integer(), nullable=False),
        sa.Column('start_sec', sa.Float(), nullable=False),
        sa.Column('end_sec', sa.Float(), nullable=False),
        sa.Column('has_face', sa.Boolean(), nullable=False),
        sa.Column('action', sa.Enum('swap', 'keep', name='segment_action_enum'), nullable=False),
        sa.Column('prompt_override', sa.Text(), nullable=True),
        sa.Column('reference_image_urls_override', sa.JSON(), nullable=True),
        sa.Column('pre_roll_sec', sa.Float(), nullable=False),
        sa.Column('post_roll_sec', sa.Float(), nullable=False),
        sa.Column('kie_upload_url', sa.Text(), nullable=True),
        sa.Column('seedance_task_id', sa.String(length=255), nullable=True),
        sa.Column('seedance_result_url', sa.Text(), nullable=True),
        sa.Column('local_clip_path', sa.Text(), nullable=True),
        sa.Column('local_result_path', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('pending', 'uploading', 'submitted', 'generating', 'completed', 'failed', 'skipped', name='segment_status_enum'), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_segments_job_id'), 'segments', ['job_id'], unique=False)
