"""Durable assistant jobs, outbox and paid-call intent ledger (AICP-266).

Revision ID: d5b0f3a8c702
Revises: c4a9e2f7b601

Additive schema only. Applying this migration is a separately authorized live
operation; it does not start dispatch, recovery or provider submission.
"""

from alembic import op
import sqlalchemy as sa

revision = "d5b0f3a8c702"
down_revision = "c4a9e2f7b601"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "assistant_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_scope", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("pinned_base_version_id", sa.String(36), sa.ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("pending_version_id", sa.String(36), sa.ForeignKey("assistant_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("accepted_plan", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("phase", sa.String(24), nullable=False),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_token", sa.String(36), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_generation", sa.Integer(), nullable=False),
        sa.Column("claim_count", sa.Integer(), nullable=False),
        sa.Column("max_claims", sa.Integer(), nullable=False),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("request_scope", "idempotency_key", name="uq_assistant_job_request"),
        sa.UniqueConstraint("pending_version_id", name="uq_assistant_job_pending_version"),
        sa.CheckConstraint("status IN ('queued','running','cancel_requested','succeeded','failed','cancelled')", name="ck_assistant_job_status"),
        sa.CheckConstraint("phase IN ('resolve','render','inspect','publish')", name="ck_assistant_job_phase"),
        sa.CheckConstraint("lease_generation >= 0 AND claim_count >= 0 AND max_claims BETWEEN 1 AND 10", name="ck_assistant_job_counters"),
        sa.CheckConstraint("((status IN ('running','cancel_requested')) AND lease_owner IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL) OR ((status NOT IN ('running','cancel_requested')) AND lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)", name="ck_assistant_job_lease"),
    )
    op.create_index("ix_assistant_jobs_recovery", "assistant_jobs", ["status", "lease_expires_at"])
    op.create_table(
        "assistant_job_outbox",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("assistant_jobs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("event_key", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("job_id", "event_key", name="uq_assistant_job_outbox_event"),
    )
    op.create_index("ix_assistant_job_outbox_pending", "assistant_job_outbox", ["dispatched_at", "created_at"])
    op.create_table(
        "assistant_paid_call_intents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("assistant_jobs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("logical_key", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("provider_request_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("submission_token", sa.String(36), nullable=True),
        sa.Column("provider_task_id", sa.String(255), nullable=True),
        sa.Column("ambiguity_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_id", "logical_key", name="uq_assistant_paid_call_logical"),
        sa.UniqueConstraint("job_id", "provider", "request_sha256", name="uq_assistant_paid_call_fingerprint"),
        sa.UniqueConstraint("provider", "provider_task_id", name="uq_assistant_paid_call_receipt"),
        sa.UniqueConstraint("provider_request_key", name="uq_assistant_paid_call_request_key"),
        sa.CheckConstraint("status IN ('prepared','submitting','submitted','unknown','completed','failed')", name="ck_assistant_paid_call_status"),
        sa.CheckConstraint("(status = 'prepared' AND submission_token IS NULL) OR (status <> 'prepared' AND submission_token IS NOT NULL)", name="ck_assistant_paid_call_permit"),
    )
    op.create_index("ix_assistant_paid_call_ambiguity", "assistant_paid_call_intents", ["status", "updated_at"])


def downgrade():
    op.drop_index("ix_assistant_paid_call_ambiguity", table_name="assistant_paid_call_intents")
    op.drop_table("assistant_paid_call_intents")
    op.drop_index("ix_assistant_job_outbox_pending", table_name="assistant_job_outbox")
    op.drop_table("assistant_job_outbox")
    op.drop_index("ix_assistant_jobs_recovery", table_name="assistant_jobs")
    op.drop_table("assistant_jobs")
