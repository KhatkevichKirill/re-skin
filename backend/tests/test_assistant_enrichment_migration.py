"""Populated h8 upgrade/guard/rollback rehearsal in disposable schemas only."""
import importlib.util
from pathlib import Path
import uuid
import os

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from app.db import Base
from app.assistant_models import AssistantConversation, AssistantVersion
from app.assistant_store import append_message
from app.assistant_turns import AssistantTurn, claim_turn
from tests.test_assistant_turns import NOW
from tests.test_assistant_postgres import pg_namespace


_COLUMNS = {"enrichment_snapshot", "broker_context_sha256", "broker_cleanup_pending", "broker_cleanup_next_at"}


def migration(name):
    path = Path(__file__).parents[1] / "alembic/versions" / name
    spec = importlib.util.spec_from_file_location("migration_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exercise(engine):
    Base.metadata.create_all(engine, tables=[table for table in Base.metadata.sorted_tables
        if table.name not in ("assistant_turns", "assistant_turn_edits", "assistant_agent_steps")])
    e6 = migration("e6c1d8f9a403_assistant_planning_turns.py")
    h8 = migration("h8c3d0e5f792_assistant_turn_enrichment.py")
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            e6.upgrade()
            h8.upgrade()
        assert _COLUMNS <= {column["name"] for column in inspect(connection).get_columns("assistant_turns")}
        assert "ix_assistant_turns_broker_cleanup" in {item["name"] for item in inspect(connection).get_indexes("assistant_turns")}
        # Empty optional state can be rolled back without discarding evidence.
        with Operations.context(MigrationContext.configure(connection)):
            h8.downgrade()
            assert not _COLUMNS & {column["name"] for column in inspect(connection).get_columns("assistant_turns")}
            h8.upgrade()
        # ORM current model includes the additive agent columns; keep this test
        # focused on the preceding enrichment guard while creating those fields.
        with Operations.context(MigrationContext.configure(connection)):
            migration("i9d4e1f6a803_assistant_agent_steps.py").upgrade()
        with Session(connection, join_transaction_mode="create_savepoint", expire_on_commit=False) as session:
            with session.begin():
                conversation = AssistantConversation(id=str(uuid.uuid4()), run_id=str(uuid.uuid4()), project_id=str(uuid.uuid4()))
                session.add(conversation)
                session.flush()
                version = AssistantVersion(id=str(uuid.uuid4()), conversation_id=conversation.id, sequence=1,
                    selection_epoch=0, idempotency_key="base", request_sha256="a"*64, actor="operator", recipe={}, status="succeeded")
                session.add(version)
                session.flush()
                message = append_message(session, conversation.id, idempotency_key="turn", actor="operator",
                    role="operator", content="Fix speech", base_version_id=version.id)
                turn = AssistantTurn(id=str(uuid.uuid4()), conversation_id=conversation.id, base_version_id=version.id,
                    operator_message_id=message.id, idempotency_key="turn", request_sha256="a"*64, actor="operator",
                    context_snapshot={}, status="queued", lease_generation=0, created_at=NOW, updated_at=NOW)
                session.add(turn)
                session.flush()
                assert session.scalar(text("SELECT enrichment_snapshot IS NULL FROM assistant_turns WHERE id=:id"), {"id": turn.id})
                with pytest.raises(DatabaseError), session.begin_nested():
                    session.execute(text("UPDATE assistant_turns SET enrichment_snapshot='{}' WHERE id=:id"), {"id": turn.id})
                claim_turn(session, turn.id, owner="fixture", now=NOW)
                session.execute(text("UPDATE assistant_turns SET enrichment_snapshot='{}' WHERE id=:id"), {"id": turn.id})
                with pytest.raises(DatabaseError), session.begin_nested():
                    session.execute(text("UPDATE assistant_turns SET enrichment_snapshot=NULL WHERE id=:id"), {"id": turn.id})
        with Operations.context(MigrationContext.configure(connection)):
            with pytest.raises(RuntimeError, match="refusing to discard"):
                h8.downgrade()
        assert _COLUMNS <= {column["name"] for column in inspect(connection).get_columns("assistant_turns")}


def test_sqlite_enrichment_migration_preserves_populated_history(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'h8.db'}")
    try:
        exercise(engine)
    finally:
        engine.dispose()


@pytest.mark.skipif(not os.environ.get("ASSISTANT_TEST_PG_URL"), reason="isolated PostgreSQL opt-in required")
def test_postgres_enrichment_migration_preserves_populated_history(pg_namespace):
    engine, _, _ = pg_namespace
    exercise(engine)
