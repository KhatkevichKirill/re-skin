"""add_seedance_2_5_and_project_max_segment_sec

Two changes that arrive together because they only make sense together.

1. A new model, Seedance 2.5 (kie model id bytedance/seedance-2-5), joins the
   Seedance family in app/ai_models.py. It shares the createTask input schema
   of the other Seedance variants, tops out at 720p, exposes an explicit
   generate_audio switch, and — the headline gain — accepts clips up to 30s
   instead of 15s. This migration adds the 'seedance-2-5' label to the
   run_model_enum Postgres type. No-op on SQLite, which stores ENUMs as VARCHAR
   and has no type object to alter.

2. The analyze-time segmentation cap becomes a per-project choice. Until now
   analyze_project always cut swap segments at a hardcoded 10s — the smallest
   per-clip ceiling across all models (Gemini Omni), so the result was runnable
   on anything. A 30s-capable model is wasted on 10s pieces, but raising the cap
   globally would break Gemini/Seedance-2.0 runs, so the cap moves onto the
   project.

New column on `video_projects`:
  - max_segment_sec  DOUBLE PRECISION NULL

Deliberately nullable with NO server_default: NULL is meaningful and means
"use ai_models.UNIVERSAL_MAX_SEGMENT_SEC" (10.0s today). Every existing project
therefore keeps exactly its current segmentation behaviour without a backfill,
and the universal default stays a single source of truth in Python rather than
being frozen into the schema.

The cap is read at analyze time, so setting it only affects the next analysis;
segments already cut are untouched. A cap above a given model's max_clip_sec is
permitted, but it costs swap coverage on runs that use a shorter-limit model:
the over-long segments are marked failed with RunSegment.source_fallback (added
in the next migration, d3b8e5c710f4) and the stitch substitutes the original
footage for them.

Revision ID: c8d2f4a6e971
Revises: b4f6d8e0c357
Create Date: 2026-08-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8d2f4a6e971'
down_revision: Union[str, None] = 'b4f6d8e0c357'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        # ALTER TYPE ... ADD VALUE must be run outside a transaction block on
        # older Postgres, but Postgres 12+ allows it inside a transaction.
        # Alembic wraps migrations in a transaction; use execute() directly.
        # IF NOT EXISTS keeps this idempotent across re-runs.
        op.execute("ALTER TYPE run_model_enum ADD VALUE IF NOT EXISTS 'seedance-2-5'")

    # Runs on both dialects. No server_default: NULL means "use the universal
    # 10s default", so existing rows must stay NULL rather than be backfilled.
    op.add_column(
        'video_projects',
        sa.Column('max_segment_sec', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    # Only the column comes back off. Postgres does not support removing values
    # from an enum type without recreating it, so 'seedance-2-5' stays in
    # run_model_enum — harmless, since no row will reference it once the code
    # that writes it is rolled back too.
    op.drop_column('video_projects', 'max_segment_sec')
