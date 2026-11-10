"""Opt-in PostgreSQL proof in the explicitly authorized disposable test DB.

ASSISTANT_TEST_PG_URL must point to assistant_test@127.0.0.1:55439/assistant_test.
No configured application database is read or used. Every fixture creates a unique
aicp264_<uuid> schema and drops only that schema in finally. All files use tmp_path.
No provider, Drive, Docker, runtime login or production resource is used here.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import InternalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema, DropSchema

from app.db import Base
from app.models import Run, VideoProject
from app.assistant_contracts import EditPlan
from app.assistant_models import AssistantConversation, AssistantSelection, AssistantVersion
from app.assistant_store import (
    ArtifactInput, AssistantArtifactStorage, get_or_create_conversation,
    get_version_artifact, import_run_version, list_versions, publish_version,
    reserve_version, select_version, verify_artifact,
)

_URL = os.environ.get("ASSISTANT_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not _URL, reason="ASSISTANT_TEST_PG_URL not set; isolated PostgreSQL opt-in only")


@pytest.fixture
def pg_namespace(tmp_path):
    url = make_url(_URL)
    # Refuse both accidental production URLs and broad arbitrary test targets.
    assert (url.get_backend_name(), url.host, url.port, url.database, url.username) == (
        "postgresql", "127.0.0.1", 55439, "assistant_test", "assistant_test",
    ), "only the authorized disposable assistant_test database is allowed"
    assert not url.query, "connection URL overrides are not allowed"
    schema = "aicp264_" + uuid.uuid4().hex
    controller = create_engine(url, connect_args={"connect_timeout": 5})
    engine = None
    try:
        with controller.begin() as connection:
            assert connection.execute(text("SELECT current_database(), current_user")).one() == (
                "assistant_test", "assistant_test",
            )
            connection.execute(CreateSchema(schema))
        engine = create_engine(url, pool_size=8, max_overflow=0, connect_args={
            "connect_timeout": 5,
            "options": f"-csearch_path={schema} -clock_timeout=5000 -cstatement_timeout=15000",
        })
        with engine.connect() as connection:
            assert connection.execute(text("SELECT current_schema()")).scalar_one() == schema
        yield engine, schema, tmp_path
    finally:
        if engine is not None:
            engine.dispose()
        with controller.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
            assert connection.execute(text("SELECT count(*) FROM pg_namespace WHERE nspname=:schema"),
                                      {"schema": schema}).scalar_one() == 0
        controller.dispose()


@pytest.fixture
def pg_store(pg_namespace):
    engine, schema, directory = pg_namespace
    Base.metadata.create_all(engine)
    data = directory / "data"
    data.mkdir()
    sessions = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    return sessions, AssistantArtifactStorage(data), data, engine


def _legacy(store):
    sessions, storage, data, _ = store
    with sessions.begin() as session:
        project = VideoProject(source_type="upload", source_ref="isolated fixture", status="ready")
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status="done", prompt="frozen prompt",
                  result_duration_sec=20, result_gdrive_file_id="test-only-original-drive-id")
        session.add(run)
        session.flush()
        source = data / (run.id + ".mp4")
        source.write_bytes(b"isolated-original-bytes")
        run.result_local_path = str(source)
        return run.id, project.id, source


def _baseline(store):
    sessions, storage, _, _ = store
    run_id, project_id, source = _legacy(store)
    with sessions.begin() as session:
        version = import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage)
        return run_id, project_id, source, version.id, version.conversation_id


def _plan(run_id, base_id):
    return EditPlan.model_validate({"run_id": run_id, "base_version_id": base_id,
                                   "operations": [{"kind": "trim", "start_sec": 0, "end_sec": 10}]})


def _output(data, index):
    path = data / f"output-{index}.mp4"
    path.write_bytes(f"isolated-edited-bytes-{index}".encode())
    return ArtifactInput(path, metadata={"duration_sec": 10})


def _race(sessions, actions):
    barrier = threading.Barrier(len(actions))
    results, failures = [], []

    def execute(index, action):
        try:
            with sessions.begin() as session:
                barrier.wait(timeout=5)
                value = action(session)
            results.append((index, value))
        except BaseException as exc:
            failures.append((index, repr(exc)))

    threads = [threading.Thread(target=execute, args=(i, action), daemon=True)
               for i, action in enumerate(actions)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive(), "isolated PostgreSQL transaction did not terminate"
    assert failures == []
    return [value for _, value in sorted(results)]


def _migration():
    path = Path(__file__).parents[1] / "alembic/versions/c4a9e2f7b601_add_assistant_history.py"
    spec = importlib.util.spec_from_file_location("assistant_pg_history_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def test_postgres_migration_parity_guards_and_downgrade(pg_namespace, monkeypatch):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    engine, schema, _ = pg_namespace
    migration = _migration()
    names = {"assistant_conversations", "assistant_versions", "assistant_selections",
             "assistant_artifacts", "assistant_version_artifacts", "assistant_messages"}
    with engine.begin() as connection:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        inspector = inspect(connection)
        assert set(inspector.get_table_names()) == names
        for name in names:
            actual = {c["name"]: c for c in inspector.get_columns(name)}
            expected = Base.metadata.tables[name]
            assert set(actual) == set(expected.columns.keys())
            assert all(actual[c.name]["nullable"] == c.nullable for c in expected.columns)
            assert all(str(actual[c.name]["type"].compile(dialect=engine.dialect)) ==
                       str(c.type.compile(dialect=engine.dialect)) for c in expected.columns)
            assert {c["name"] for c in inspector.get_unique_constraints(name)} == {
                c.name for c in expected.constraints if c.__class__.__name__ == "UniqueConstraint"
            }
            assert {c["name"] for c in inspector.get_check_constraints(name)} == {
                c.name for c in expected.constraints if c.__class__.__name__ == "CheckConstraint"
            }
            reflected_indexes = {i["name"] for i in inspector.get_indexes(name)
                                 if not i.get("duplicates_constraint")}
            assert reflected_indexes == {i.name for i in expected.indexes}
            actual_fks = {(tuple(f["constrained_columns"]), f["referred_table"],
                           tuple(f["referred_columns"]), f["options"].get("ondelete"))
                          for f in inspector.get_foreign_keys(name)}
            expected_fks = {(tuple(e.parent.name for e in f.elements), f.referred_table.name,
                             tuple(e.column.name for e in f.elements), f.ondelete)
                            for f in expected.foreign_key_constraints}
            assert actual_fks == expected_fks
        conversation_id, version_id = str(uuid.uuid4()), str(uuid.uuid4())
        connection.execute(text("INSERT INTO assistant_conversations (id,run_id,project_id) VALUES (:id,:run,:project)"),
                           {"id": conversation_id, "run": str(uuid.uuid4()), "project": str(uuid.uuid4())})
        connection.execute(text("INSERT INTO assistant_versions "
                                "(id,conversation_id,sequence,selection_epoch,idempotency_key,request_sha256,actor,recipe) "
                                "VALUES (:id,:conversation,1,0,'migration-test',:digest,'operator','{}')"),
                           {"id": version_id, "conversation": conversation_id, "digest": "a" * 64})
        connection.execute(text("UPDATE assistant_conversations SET next_version_sequence=2 WHERE id=:id"), {"id": conversation_id})
        with pytest.raises(InternalError, match="immutable") as rejected:
            with connection.begin_nested():
                connection.execute(text("UPDATE assistant_versions SET recipe=:recipe WHERE id=:id"),
                                   {"id": version_id, "recipe": '{"changed":true}'})
        assert rejected.value.orig.pgcode == "P0001"
        with pytest.raises(InternalError, match="immutable") as rejected:
            with connection.begin_nested():
                connection.execute(text("UPDATE assistant_conversations SET project_id=:project WHERE id=:id"),
                                   {"id": conversation_id, "project": str(uuid.uuid4())})
        assert rejected.value.orig.pgcode == "P0001"
        connection.execute(text("UPDATE assistant_versions SET status='succeeded',completed_at=CURRENT_TIMESTAMP WHERE id=:id"),
                           {"id": version_id})
        with pytest.raises(InternalError, match="immutable") as rejected:
            with connection.begin_nested():
                connection.execute(text("UPDATE assistant_versions SET status='pending' WHERE id=:id"), {"id": version_id})
        assert rejected.value.orig.pgcode == "P0001"
        connection.execute(text("DELETE FROM assistant_versions WHERE id=:id"), {"id": version_id})
        connection.execute(text("DELETE FROM assistant_conversations WHERE id=:id"), {"id": conversation_id})
        migration.downgrade()
        assert inspect(connection).get_table_names() == []
        assert connection.execute(text("SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                                       "WHERE n.nspname=:schema"), {"schema": schema}).scalar_one() == 0


def test_postgres_metadata_guards_and_caller_rollback(pg_store):
    sessions, storage, data, _ = pg_store
    run_id, _, source, base_id, conversation_id = _baseline(pg_store)
    item = _output(data, 0)
    with sessions.begin() as session:
        pending = reserve_version(session, _plan(run_id, base_id), idempotency_key="rollback", actor="operator")
        pending_id = pending.id
    with sessions() as session:
        publish_version(session, pending_id, full=item, preview=item, storage=storage)
        session.rollback()
    with sessions() as session:
        assert session.get(AssistantVersion, pending_id).status == "pending"
        assert session.get(AssistantSelection, conversation_id).selected_version_id == base_id
        assert verify_artifact(get_version_artifact(session, base_id), storage=storage).read_bytes() == source.read_bytes()
    with pytest.raises(InternalError, match="immutable") as rejected, sessions.begin() as session:
        session.execute(update(AssistantVersion).where(AssistantVersion.id == base_id).values(recipe={"tampered": True}))
    assert rejected.value.orig.pgcode == "P0001"


def test_postgres_concurrent_get_or_create_has_one_conversation(pg_store):
    sessions, storage, _, _ = pg_store
    run_id, project_id, _ = _legacy(pg_store)
    ids = _race(sessions, [lambda session: get_or_create_conversation(
        session, run_id=run_id, project_id=project_id).id for _ in range(3)])
    assert len(set(ids)) == 1
    with sessions() as session:
        assert session.query(AssistantConversation).count() == 1


def test_postgres_concurrent_import_is_idempotent_and_pins_one_snapshot(pg_store):
    sessions, storage, _, _ = pg_store
    run_id, _, source = _legacy(pg_store)
    ids = _race(sessions, [lambda session: import_run_version(
        session, session.get(Run, run_id), actor="operator", storage=storage).id for _ in range(2)])
    assert len(set(ids)) == 1
    with sessions() as session:
        assert session.query(AssistantVersion).count() == 1
        assert verify_artifact(get_version_artifact(session, ids[0]), storage=storage).read_bytes() == source.read_bytes()
        assert session.get(Run, run_id).result_gdrive_file_id == "test-only-original-drive-id"


@pytest.mark.parametrize("keys", [("same-request", "same-request"), ("first-request", "second-request")])
def test_postgres_concurrent_reservations_serialize_sequence_and_idempotency(pg_store, keys):
    sessions, _, _, _ = pg_store
    run_id, _, _, base_id, conversation_id = _baseline(pg_store)
    actions = [lambda session, key=key: reserve_version(
        session, _plan(run_id, base_id), idempotency_key=key, actor="operator").id for key in keys]
    ids = _race(sessions, actions)
    assert len(set(ids)) == len(set(keys))
    with sessions() as session:
        assert [v.sequence for v in list_versions(session, conversation_id)] == list(range(1, len(set(keys)) + 2))


@pytest.mark.parametrize("first", [0, 1])
def test_postgres_completion_order_never_overrides_newer_success(pg_store, first):
    sessions, storage, data, _ = pg_store
    run_id, _, _, base_id, conversation_id = _baseline(pg_store)
    with sessions.begin() as session:
        versions = [reserve_version(session, _plan(run_id, base_id), idempotency_key=f"job-{i}", actor="operator").id
                    for i in range(2)]
    items = [_output(data, i) for i in range(2)]
    # Both worker threads start together. The second waits for the explicitly
    # selected first transaction's commit, proving both finish-order branches.
    committed = threading.Event()

    def complete(session, index):
        if index != first:
            assert committed.wait(timeout=10)
        version = publish_version(session, versions[index], full=items[index], preview=items[index], storage=storage)
        return version.id

    barrier = threading.Barrier(2)
    failures = []

    def worker(index):
        try:
            with sessions.begin() as session:
                barrier.wait(timeout=5)
                complete(session, index)
            if index == first:
                committed.set()
        except BaseException as exc:
            failures.append(repr(exc))
            committed.set()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()
    assert failures == []
    with sessions() as session:
        selection = session.get(AssistantSelection, conversation_id)
        assert selection.latest_version_id == selection.selected_version_id == versions[1]
        assert selection.epoch == 0


def test_postgres_manual_selection_racing_publish_always_preserves_operator_choice(pg_store):
    sessions, storage, data, _ = pg_store
    run_id, _, _, base_id, conversation_id = _baseline(pg_store)
    with sessions.begin() as session:
        pending = reserve_version(session, _plan(run_id, base_id), idempotency_key="old-job", actor="operator")
        pending_id = pending.id
    item = _output(data, 0)
    _race(sessions, [
        lambda session: publish_version(session, pending_id, full=item, preview=item, storage=storage).id,
        lambda session: select_version(session, conversation_id, base_id, expected_epoch=0).epoch,
    ])
    with sessions.begin() as session:
        selection = session.get(AssistantSelection, conversation_id)
        assert selection.selected_version_id == base_id and selection.epoch == 1
        assert selection.latest_version_id == pending_id
        branch = reserve_version(session, _plan(run_id, base_id), idempotency_key="new-branch", actor="operator")
        branch_id = branch.id
    with sessions.begin() as session:
        next_item = _output(data, 1)
        publish_version(session, branch_id, full=next_item, preview=next_item, storage=storage)
        selection = session.get(AssistantSelection, conversation_id)
        assert selection.selected_version_id == selection.latest_version_id == branch_id
        assert selection.epoch == 1
