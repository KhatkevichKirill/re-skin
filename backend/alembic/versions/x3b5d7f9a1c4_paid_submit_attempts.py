"""Paid submit attempts (AICP-351 = AICP-344 stage 2, docs/assistant-drain-check-design.md §5, §7).

Additive only: two new tables and two nullable/defaulted columns on
``run_captions`` (``asr_claim``, ``asr_epoch``: the caption ASR dispatch fence
and the operator-retry epoch of the §5.2 bound).

- ``paid_submit_attempts`` — one row per attempt at a paid kie.ai video/image
  call (§5.1) or an LLM-only job's attempt (§5.2), committed BEFORE the call.
  ``status`` is a VARCHAR with a CHECK (no Postgres enum, no ``ALTER TYPE``);
  UNIQUE (owner_kind, owner_id, attempt_no) and a partial UNIQUE index allowing
  at most one open (``started`` / ``parked``) attempt per owner.
- ``paid_submit_attempt_actions`` — insert-only audit of the operator's
  resubmit / abandon / release_sender (``_immutable`` / ``_no_delete``
  triggers on both dialects, as m3c9e5a1b427 does for provider_task_actions).

No existing table or enum changes, no backfill: the old image ignores both
tables, so applying this before the image is safe. The NEW image needs it
before it starts (every paid submit writes an attempt; ops_status and
deploy/drain-check.sql read it).

Downgrade refuses while any attempt is open (``started`` / ``parked`` /
``superseded``): such a row is the only record that a paid call may have been
billed. Rolling the image back past this revision follows the rollback barrier
of design §6 (resolve parked attempts, then drain).

This is a fixed copy of the schema in ``app/paid_attempts_models.py``; it
imports no application code. tests/test_paid_attempts.py compares the two.

Revision ID: x3b5d7f9a1c4
Revises: w2c4e6a8b0d3
Create Date: 2026-11-10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "x3b5d7f9a1c4"
down_revision: Union[str, None] = "w2c4e6a8b0d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

STATUSES = (
    "started", "accepted", "failed_unbilled", "parked", "resubmitted", "abandoned",
    "failed", "unknown", "superseded", "stopped",
)
OWNER_KINDS = (
    "run_segment", "run_preprocess", "segment_image_job",
    "llm_translate", "llm_transcribe", "llm_caption_asr",
)
KINDS = ("video", "image", "llm")
ACTIONS = ("resubmit", "abandon", "release_sender")
OPEN_WHERE = "status IN ('started', 'parked')"
TABLE = "paid_submit_attempts"
ACTIONS_TABLE = "paid_submit_attempt_actions"


def _in(col: str, values) -> str:
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def _insert_only(name: str) -> None:
    dialect = op.get_bind().dialect.name
    msg = f"{name} is insert-only"
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


def bind_is_pg() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_kind", sa.String(32), nullable=False),
        sa.Column("owner_id", sa.String(80), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=True),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("provider", sa.String(8), nullable=False, server_default=sa.text("'kie'")),
        sa.Column("request_sha256", sa.String(64), nullable=True),
        sa.Column("lease_generation", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("provider_task_id", sa.String(128), nullable=True),
        sa.Column("predecessor_task_id", sa.String(128), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("calls_sent", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("max_calls", sa.Integer(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("sender_host", sa.String(64), nullable=True),
        sa.Column("sender_boot", sa.String(64), nullable=True),
        sa.Column("sender_job", sa.String(160), nullable=True),
        sa.Column("sender_released_at", sa.DateTime(), nullable=True),
        sa.Column("sender_note", sa.Text(), nullable=True),
        sa.CheckConstraint(_in("status", sorted(STATUSES)), name="ck_paid_submit_attempts_status"),
        sa.CheckConstraint(_in("owner_kind", sorted(OWNER_KINDS)),
                           name="ck_paid_submit_attempts_owner_kind"),
        sa.CheckConstraint(_in("kind", KINDS), name="ck_paid_submit_attempts_kind"),
        sa.UniqueConstraint("owner_kind", "owner_id", "attempt_no",
                            name="uq_paid_submit_attempts_owner_attempt"),
    )
    op.create_index("uq_paid_submit_attempts_open", TABLE, ["owner_kind", "owner_id"],
                    unique=True, postgresql_where=sa.text(OPEN_WHERE),
                    sqlite_where=sa.text(OPEN_WHERE))
    op.create_index("ix_paid_submit_attempts_run", TABLE, ["run_id"])
    op.create_index("ix_paid_submit_attempts_status", TABLE, ["status"])

    op.create_table(
        ACTIONS_TABLE,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("attempt_id", sa.String(36),
                  sa.ForeignKey(f"{TABLE}.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("from_status", sa.String(16), nullable=False),
        sa.Column("to_status", sa.String(16), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.CheckConstraint(_in("action", ACTIONS), name="ck_paid_submit_attempt_actions_action"),
    )
    op.create_index(f"ix_{ACTIONS_TABLE}_attempt_id", ACTIONS_TABLE, ["attempt_id"])
    _insert_only(ACTIONS_TABLE)

    # The caption ASR dispatch fence + operator-retry epoch (nullable / server
    # default: metadata-only on PG; the old image's INSERTs get the default).
    if bind_is_pg():
        op.execute("SET LOCAL lock_timeout = '5s'")
    op.add_column("run_captions", sa.Column("asr_claim", sa.String(64), nullable=True))
    op.add_column("run_captions", sa.Column("asr_epoch", sa.Integer(), nullable=False,
                                            server_default=sa.text("0")))


def downgrade() -> None:
    bind = op.get_bind()
    open_rows = bind.execute(sa.text(
        f"SELECT count(*) FROM {TABLE} WHERE status IN ('started', 'parked', 'superseded')"
    )).scalar_one()
    if open_rows:
        raise RuntimeError(
            f"refusing to drop {TABLE}: {open_rows} open attempt(s) may have billed — "
            "resolve them first (python -m app.paid_attempts list)")
    with op.batch_alter_table("run_captions") as batch:
        batch.drop_column("asr_epoch")
        batch.drop_column("asr_claim")
    dialect = bind.dialect.name
    if dialect == "postgresql":
        op.execute(f"DROP TRIGGER IF EXISTS {ACTIONS_TABLE}_immutable ON {ACTIONS_TABLE}")
        op.execute(f"DROP FUNCTION IF EXISTS {ACTIONS_TABLE}_reject()")
    elif dialect == "sqlite":
        op.execute(f"DROP TRIGGER IF EXISTS {ACTIONS_TABLE}_immutable")
        op.execute(f"DROP TRIGGER IF EXISTS {ACTIONS_TABLE}_no_delete")
    op.drop_index(f"ix_{ACTIONS_TABLE}_attempt_id", table_name=ACTIONS_TABLE)
    op.drop_table(ACTIONS_TABLE)
    op.drop_index("ix_paid_submit_attempts_status", table_name=TABLE)
    op.drop_index("ix_paid_submit_attempts_run", table_name=TABLE)
    op.drop_index("uq_paid_submit_attempts_open", table_name=TABLE)
    op.drop_table(TABLE)
