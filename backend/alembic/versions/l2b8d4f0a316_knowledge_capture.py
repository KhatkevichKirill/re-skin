"""Knowledge capture hooks (AICP-299, docs/knowledge-capture-hooks.md §11.1).

Additive only, on top of k1a7c3e9f205 (which already carries S16, the
publication UNIQUE and kb_capture_queue.file_identity):

- S4: version reasons += regeneration, external_repair (kb_case_versions.reason
  and kb_capture_queue.reason share the list, so both CHECKs are recreated);
- S13a: kb_capture_scopes (insert-only provenance of the capture scope);
- S13b: observation kinds += input_content;
- S15: kb_link_failures (insert-only record of a failed publication link).

No existing non-``kb_`` table is touched and nothing is backfilled. Downgrade
refuses while any ``kb_`` table holds a row (the k1a7c3e9f205 guard).

This is a fixed copy of the schema in ``app/knowledge/models.py``; it imports
no application code. tests/test_knowledge_pg.py compares the two.

Revision ID: l2b8d4f0a316
Revises: k1a7c3e9f205
Create Date: 2026-10-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'l2b8d4f0a316'
down_revision: Union[str, None] = 'k1a7c3e9f205'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_REASONS = "reason IN ('initial_output', 'restitch', 'range_attempt', 'audio_patch', 'assistant_promotion', 'import')"
NEW_REASONS = ("reason IN ('initial_output', 'restitch', 'range_attempt', 'audio_patch', 'assistant_promotion', "
               "'import', 'regeneration', 'external_repair')")
OLD_KINDS = "kind IN ('status', 'output', 'cost_estimate', 'cost_settled', 'error', 'ambiguous', 'redispatch')"
NEW_KINDS = ("kind IN ('status', 'output', 'cost_estimate', 'cost_settled', 'error', 'ambiguous', 'redispatch', "
             "'input_content')")

NEW_TABLES = ('kb_capture_scopes', 'kb_link_failures')
ALL_KB_TABLES = (
    'kb_import_batches', 'kb_import_batch_observations', 'kb_artifacts', 'kb_artifact_checks', 'kb_cases',
    'kb_case_versions', 'kb_attempts', 'kb_attempt_submissions', 'kb_attempt_observations', 'kb_requests',
    'kb_feedback_events', 'kb_rules', 'kb_rule_versions', 'kb_rule_reviews', 'kb_rule_evidence', 'kb_decisions',
    'kb_decision_outcomes', 'kb_import_members', 'kb_invalidations', 'kb_capture_queue', 'kb_capture_done',
    'kb_eval_manifests', 'kb_eval_membership', 'kb_source_subjects', 'kb_subject_links', 'kb_import_cursors',
) + NEW_TABLES


def _guard_sql(table):
    return [
        f"CREATE FUNCTION {table}_reject_change() RETURNS trigger AS $$ BEGIN "
        f"RAISE EXCEPTION USING MESSAGE = 'immutable knowledge evidence: ' || TG_TABLE_NAME; END; $$ LANGUAGE plpgsql",
        f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION {table}_reject_change()",
        f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {table}_reject_change()",
    ]


def _replace_check(table, name, sql):
    op.drop_constraint(name, table, type_='check')
    op.create_check_constraint(name, table, sql)


def upgrade() -> None:
    _replace_check('kb_case_versions', 'ck_kb_case_version_reason', NEW_REASONS)
    _replace_check('kb_capture_queue', 'ck_kb_capture_reason', NEW_REASONS)
    _replace_check('kb_attempt_observations', 'ck_kb_observation_kind', NEW_KINDS)

    op.create_table('kb_capture_scopes',
    sa.Column('scope_sha256', sa.String(length=64), nullable=False),
    sa.Column('project_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('first_seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('recorded_by', sa.String(length=128), nullable=True),
    sa.Column('seq', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.CheckConstraint("scope_sha256 ~ '^[0-9a-f]{64}$'", name='ck_kb_capture_scope_sha256'),
    sa.PrimaryKeyConstraint('scope_sha256'),
    sa.UniqueConstraint('seq')
    )
    op.create_table('kb_link_failures',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('publication_id', sa.String(length=200), nullable=False),
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=True),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('path', sa.Text(), nullable=False),
    sa.Column('st_dev', sa.BigInteger(), nullable=True),
    sa.Column('st_ino', sa.BigInteger(), nullable=True),
    sa.Column('st_mtime_ns', sa.BigInteger(), nullable=True),
    sa.Column('failed_at_ns', sa.BigInteger(), nullable=False),
    sa.Column('error_class', sa.String(length=64), nullable=True),
    sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('seq', sa.BigInteger(), sa.Identity(always=True), nullable=False),
    sa.CheckConstraint("kind IN ('stitch', 'range', 'audio', 'assistant')", name='ck_kb_link_failure_kind'),
    sa.CheckConstraint('(st_dev IS NULL) = (st_ino IS NULL) AND (st_ino IS NULL) = (st_mtime_ns IS NULL)',
                       name='ck_kb_link_failure_identity'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('publication_id', name='uq_kb_link_failure_publication'),
    sa.UniqueConstraint('seq')
    )
    op.create_index('ix_kb_link_failures_run', 'kb_link_failures', ['run_id', 'failed_at_ns'], unique=False)
    op.create_index('ix_kb_link_failures_identity', 'kb_link_failures', ['st_dev', 'st_ino', 'st_mtime_ns'],
                    unique=False)
    for table in NEW_TABLES:
        for sql in _guard_sql(table):
            op.execute(sql)


def downgrade() -> None:
    bind = op.get_bind()
    for table in ALL_KB_TABLES:
        if bind.scalar(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")):
            raise RuntimeError(f"refusing to discard knowledge evidence: {table} has rows")
    op.drop_index('ix_kb_link_failures_identity', table_name='kb_link_failures')
    op.drop_index('ix_kb_link_failures_run', table_name='kb_link_failures')
    op.drop_table('kb_link_failures')
    op.drop_table('kb_capture_scopes')
    for table in NEW_TABLES:
        op.execute(f"DROP FUNCTION IF EXISTS {table}_reject_change()")
    _replace_check('kb_attempt_observations', 'ck_kb_observation_kind', OLD_KINDS)
    _replace_check('kb_capture_queue', 'ck_kb_capture_reason', OLD_REASONS)
    _replace_check('kb_case_versions', 'ck_kb_case_version_reason', OLD_REASONS)
