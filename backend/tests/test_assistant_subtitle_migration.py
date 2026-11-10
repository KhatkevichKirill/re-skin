"""SQLite history survives constraint rebuilds with foreign keys enabled."""
import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from tests.test_assistant_executor import setup
from app.assistant_store import import_run_version
from app.models import Run


def test_populated_sqlite_upgrade_and_downgrade_preserve_history_and_guards(setup):
    factory,storage,_,(rid,_)=setup
    with factory.begin() as session:
        import_run_version(session,session.get(Run,rid),actor="operator",storage=storage)
    path=Path(__file__).parents[1]/"alembic/versions/f7a2b9c4d681_assistant_editable_subtitle_artifacts.py"
    spec=importlib.util.spec_from_file_location("subtitle_migration",path)
    migration=importlib.util.module_from_spec(spec);spec.loader.exec_module(migration)
    engine=factory.kw["bind"]
    with engine.begin() as connection:
        before=connection.execute(text("SELECT id,sha256 FROM assistant_artifacts ORDER BY id")).all()
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar()==1
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
            migration.upgrade()
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all()==[]
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar()==1
        assert connection.execute(text("SELECT id,sha256 FROM assistant_artifacts ORDER BY id")).all()==before
    with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(text("UPDATE assistant_artifacts SET sha256=:value"),{"value":"a"*64})
