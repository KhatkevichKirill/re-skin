"""Rehearse this additive migration on a disposable SQLite graph."""
import importlib.util
from pathlib import Path
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from app.db import Base
from app import models


def test_turn_migration_upgrade_downgrade_matches_model_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    tables = [table for table in Base.metadata.sorted_tables if table.name not in ("assistant_turns", "assistant_turn_edits", "assistant_agent_steps")]
    Base.metadata.create_all(engine, tables=tables)
    path = Path(__file__).parents[1] / "alembic/versions/e6c1d8f9a403_assistant_planning_turns.py"
    spec = importlib.util.spec_from_file_location("turn_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        inspector = inspect(connection)
        for name in ("assistant_turns", "assistant_turn_edits"):
            assert {column["name"] for column in inspector.get_columns(name)} == (set(Base.metadata.tables[name].columns.keys()) - ({"enrichment_snapshot", "broker_context_sha256", "broker_cleanup_pending", "broker_cleanup_next_at", "workflow_revision", "agent_context_snapshot"} if name == "assistant_turns" else set()))
        assert {index["name"] for index in inspector.get_indexes("assistant_turns")} == {"ix_assistant_turns_queue", "ix_assistant_turns_recovery"}
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        assert "assistant_turns" not in inspect(connection).get_table_names()
    engine.dispose()
