"""Full additive assistant migration chain on an explicitly isolated DB."""
import importlib.util
import os
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import inspect, text
from app.db import Base
from tests.test_assistant_postgres import pg_namespace

pytestmark=pytest.mark.skipif(not os.environ.get("ASSISTANT_TEST_PG_URL"),reason="isolated PostgreSQL opt-in only")


def test_complete_assistant_migration_chain_matches_models_and_reverses(pg_namespace):
    engine,_,_=pg_namespace
    directory=Path(__file__).parents[1]/"alembic/versions"
    prefixes=["c4a9e2f7b601", "d5b0f3a8c702", "e6c1d8f9a403", "f7a2b9c4d681", "h8c3d0e5f792", "i9d4e1f6a803", "j0e5f2a7b914"]
    migrations=[]
    for prefix in prefixes:
        path=next(directory.glob(prefix+"*.py"))
        spec=importlib.util.spec_from_file_location("migration_"+prefix,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        migrations.append(module)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            for migration in migrations: migration.upgrade()
        inspector=inspect(connection)
        names={name for name in Base.metadata.tables if name.startswith("assistant_")}
        assert set(inspector.get_table_names())==names
        for name in names:
            expected=Base.metadata.tables[name]
            assert {c["name"] for c in inspector.get_columns(name)}==set(expected.columns.keys())
            assert {c["name"] for c in inspector.get_check_constraints(name)}=={c.name for c in expected.constraints if c.__class__.__name__=="CheckConstraint"}
        checks=" ".join(c["sqltext"] for c in inspector.get_check_constraints("assistant_artifacts"))
        assert "subtitle" in checks
        triggers=set(connection.execute(text("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")).scalars())
        assert "assistant_turns_immutable" in triggers
        with Operations.context(MigrationContext.configure(connection)):
            for migration in reversed(migrations): migration.downgrade()
        assert inspect(connection).get_table_names()==[]
