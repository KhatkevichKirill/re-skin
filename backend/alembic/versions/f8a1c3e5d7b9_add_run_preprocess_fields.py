"""add_run_preprocess_fields

Adds optional GPT Image 2 (image-to-image) reference-image preprocessing to a
Run. Before refs reach Seedance/Gemini Omni, each reference image can be run
through GPT Image 2 with a per-image prompt; the generated image is what the AI
video model actually sees. See docs/gpt-image-2-preprocess.md.

New columns on `runs` (all nullable / default-off so existing runs are
unaffected):
  - preprocess_enabled          BOOLEAN NOT NULL DEFAULT FALSE
  - preprocess_image_prompts    JSON — list parallel to reference_image_urls
  - preprocessed_reference_urls JSON — local paths to generated images
  - preprocess_task_ids          JSON — kie.ai taskIds for resume/no-rebill

New enum value `preprocessing` on `run_status_enum` (between `created` and
`queued`).

Revision ID: f8a1c3e5d7b9
Revises: 08433b4098d9
Create Date: 2026-07-29 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8a1c3e5d7b9'
down_revision: Union[str, None] = '08433b4098d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    op.add_column(
        'runs',
        sa.Column(
            'preprocess_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        'runs',
        sa.Column('preprocess_image_prompts', sa.JSON(), nullable=True),
    )
    op.add_column(
        'runs',
        sa.Column('preprocessed_reference_urls', sa.JSON(), nullable=True),
    )
    op.add_column(
        'runs',
        sa.Column('preprocess_task_ids', sa.JSON(), nullable=True),
    )
    if bind.dialect.name == 'postgresql':
        # ADD VALUE is allowed inside a transaction on Postgres 12+ as long as
        # the new value is not used in the same transaction (it isn't here).
        op.execute(
            "ALTER TYPE run_status_enum ADD VALUE IF NOT EXISTS 'preprocessing'"
        )


def downgrade() -> None:
    op.drop_column('runs', 'preprocess_task_ids')
    op.drop_column('runs', 'preprocessed_reference_urls')
    op.drop_column('runs', 'preprocess_image_prompts')
    op.drop_column('runs', 'preprocess_enabled')
    # Postgres cannot drop an enum value without recreating the type; no-op.
