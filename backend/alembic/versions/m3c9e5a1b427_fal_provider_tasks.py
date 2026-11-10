"""Fal provider tasks (AICP-312 Stage 1 / AICP-330, docs/fal-provider-design.md §9).

Additive only:

- ``provider_tasks`` (status CHECK over the thirteen statuses, the operation /
  request / destination unique constraints, and the one-active-task-per-owner-
  slot partial unique index whose WHERE text is byte-identical to
  ``state_machine.PROVIDER_TASK_ACTIVE_WHERE``);
- ``provider_task_actions``, insert-only (``_immutable`` / ``_no_delete``
  triggers on both dialects, as j0e5f2a7b914 does);
- ``provider VARCHAR(16) NOT NULL DEFAULT 'kie'`` on ``kie_spend_reservations``
  and ``kie_spend_draws`` (AICP-330 B2). The SERVER default is what keeps the
  still-running old code's inserts (which do not name the column) valid between
  this migration and the deploy; the new code also passes it explicitly.

On PostgreSQL the ALTERs run under ``lock_timeout = '5s'``: ADD COLUMN with a
constant default is metadata-only, but its ACCESS EXCLUSIVE lock must not queue
behind a long transaction and block every spend writer behind it; on timeout
the migration fails cleanly and is simply re-run.

Downgrade refuses while ANY ``provider_tasks`` row exists: a task may have been
billed and its row is the only record tying the request to its owner.

This is a fixed copy of the schema in ``app/provider_models.py``; it imports no
application code. tests/test_pipeline_provider.py compares the two.

Revision ID: m3c9e5a1b427
Revises: l2b8d4f0a316
Create Date: 2026-10-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "m3c9e5a1b427"
down_revision: Union[str, None] = "l2b8d4f0a316"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

STATUSES = (
    "queued", "submit_started", "accepted_unrecorded", "submitted", "fetching",
    "cancelling", "needs_reconciliation", "closing", "done", "refused", "failed",
    "submit_unknown", "abandoned",
)
NON_TERMINAL = (
    "accepted_unrecorded", "cancelling", "closing", "fetching", "needs_reconciliation",
    "queued", "submit_started", "submit_unknown", "submitted",
)
STATUS_CHECK = "status IN (" + ", ".join(f"'{s}'" for s in sorted(STATUSES)) + ")"
ACTIVE_WHERE = (
    "(status IN (" + ", ".join(f"'{s}'" for s in NON_TERMINAL) + ") "
    "OR (status = 'refused' AND reservation_held))"
)
ACTIONS_TABLE = "provider_task_actions"


def _insert_only(name: str) -> None:
    dialect = op.get_bind().dialect.name
    msg = "provider_task_actions is insert-only"
    if dialect == "sqlite":
        op.execute(f"CREATE TRIGGER {name}_immutable BEFORE UPDATE ON {name} "
                   f"BEGIN SELECT RAISE(ABORT, '{msg}'); END")
        op.execute(f"CREATE TRIGGER {name}_no_delete BEFORE DELETE ON {name} "
                   f"BEGIN SELECT RAISE(ABORT, '{msg}'); END")
    elif dialect == "postgresql":
        op.execute(f"CREATE FUNCTION {name}_reject() RETURNS trigger AS $$ BEGIN "
                   f"RAISE EXCEPTION '{msg}'; END; $$ LANGUAGE plpgsql")
        op.execute(f"CREATE TRIGGER {name}_immutable BEFORE UPDATE OR DELETE ON {name} "
                   f"FOR EACH ROW EXECUTE FUNCTION {name}_reject()")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("SET LOCAL lock_timeout = '5s'")
    for table in ("kie_spend_reservations", "kie_spend_draws"):
        op.add_column(table, sa.Column("provider", sa.String(16), nullable=False,
                                       server_default=sa.text("'kie'")))

    op.create_table(
        "provider_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("endpoint", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("owner_kind", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("owner_slot", sa.String(64), nullable=False, server_default=sa.text("'main'")),
        sa.Column("owner_project_id", sa.String(36), nullable=False),
        sa.Column("owner_run_id", sa.String(36), nullable=True),
        sa.Column("destination_path", sa.Text(), nullable=False),
        sa.Column("operation_key", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("claim_owner", sa.String(64), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(), nullable=True),
        sa.Column("submit_started_at", sa.DateTime(), nullable=True),
        sa.Column("request_id", sa.String(128), nullable=True),
        sa.Column("accepted_at", sa.DateTime(), nullable=True),
        sa.Column("status_url", sa.Text(), nullable=True),
        sa.Column("response_url", sa.Text(), nullable=True),
        sa.Column("cancel_url", sa.Text(), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(), nullable=True),
        sa.Column("cancel_delivered_at", sa.DateTime(), nullable=True),
        sa.Column("cancel_attempted_at", sa.DateTime(), nullable=True),
        sa.Column("reservation_held", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("draw_pending", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("reserved_mcredits", sa.BigInteger(), nullable=False),
        sa.Column("fetch_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("fetch_deadline_at", sa.DateTime(), nullable=True),
        sa.Column("completed_seen_at", sa.DateTime(), nullable=True),
        sa.Column("poll_failures", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("request_spec", sa.JSON(), nullable=False),
        sa.Column("input_refs", sa.JSON(), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("output_sec", sa.Float(), nullable=False),
        sa.Column("resolution", sa.String(16), nullable=False),
        sa.Column("estimated_usd", sa.Float(), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result_path", sa.Text(), nullable=True),
        sa.Column("result_meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.CheckConstraint(STATUS_CHECK, name="ck_provider_tasks_status"),
        sa.UniqueConstraint("provider", "operation_key", name="uq_provider_tasks_operation"),
        sa.UniqueConstraint("provider", "request_id", name="uq_provider_tasks_request"),
        sa.UniqueConstraint("destination_path", name="uq_provider_tasks_destination"),
    )
    op.create_index("uq_provider_tasks_active_per_owner_slot", "provider_tasks",
                    ["owner_kind", "owner_id", "owner_slot"], unique=True,
                    postgresql_where=sa.text(ACTIVE_WHERE), sqlite_where=sa.text(ACTIVE_WHERE))
    op.create_index("ix_provider_tasks_status", "provider_tasks", ["status"])
    op.create_index("ix_provider_tasks_owner_project", "provider_tasks", ["owner_project_id"])
    op.create_index("ix_provider_tasks_owner_run_id", "provider_tasks", ["owner_run_id"])

    op.create_table(
        ACTIONS_TABLE,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("task_id", sa.String(36),
                  sa.ForeignKey("provider_tasks.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("from_status", sa.String(32), nullable=False),
        sa.Column("to_status", sa.String(32), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(128), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.CheckConstraint("action IN ('adopt', 'abandon', 'close', 'resume')",
                           name="ck_provider_task_actions_action"),
    )
    op.create_index("ix_provider_task_actions_task_id", ACTIONS_TABLE, ["task_id"])
    _insert_only(ACTIONS_TABLE)


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(sa.text("SELECT count(*) FROM provider_tasks")).scalar()
    if count:
        raise RuntimeError(
            f"refusing to downgrade: provider_tasks holds {count} row(s); a task may have "
            "been billed and its row is the only record of it — roll back by flag "
            "(FAL_ENABLED=false), not by schema"
        )
    if bind.dialect.name == "postgresql":
        op.execute(f"DROP TRIGGER IF EXISTS {ACTIONS_TABLE}_immutable ON {ACTIONS_TABLE}")
        op.execute(f"DROP FUNCTION IF EXISTS {ACTIONS_TABLE}_reject()")
    else:
        op.execute(f"DROP TRIGGER IF EXISTS {ACTIONS_TABLE}_immutable")
        op.execute(f"DROP TRIGGER IF EXISTS {ACTIONS_TABLE}_no_delete")
    op.drop_table(ACTIONS_TABLE)
    op.drop_table("provider_tasks")
    for table in ("kie_spend_draws", "kie_spend_reservations"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("provider")
