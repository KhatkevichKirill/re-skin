"""add_run_preprocess_mode

Adds the `preprocess_mode` column to `runs` controlling how GPT Image 2
reference preprocessing combines inputs:

  - "per_image" (default, existing behaviour): one image-to-image task per
    reference image, prompts[i] applies to refs[i].
  - "all_as_refs": a SINGLE task with input_urls = ALL refs and a single
    prompt → ONE generated image becomes the sole preprocessed reference.

Nullable with server_default 'per_image' so existing runs keep the original
per-image behaviour.

Revision ID: a1b2c3d4e5f6
Revises: f8a1c3e5d7b9
Create Date: 2026-07-30 16:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f8a1c3e5d7b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'runs',
        sa.Column(
            'preprocess_mode',
            sa.String(),
            nullable=True,
            server_default='per_image',
        ),
    )


def downgrade() -> None:
    op.drop_column('runs', 'preprocess_mode')
