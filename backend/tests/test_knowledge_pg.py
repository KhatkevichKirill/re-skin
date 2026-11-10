"""Knowledge store on PostgreSQL (AICP-295; design docs/knowledge-feedback-design.md).

The store relies on row locks, advisory locks, ``ON CONFLICT`` and database
triggers, so it is tested only on a real PostgreSQL. Opt-in like
test_spend_pg_concurrency.py: set RESKIN_TEST_PG_URL to a DISPOSABLE server
(never production — this creates and drops databases), e.g.

    docker run --rm -d --name reskin-kb-pgtest -e POSTGRES_PASSWORD=t \\
        -p 127.0.0.1:15499:5432 postgres:16-alpine
    RESKIN_TEST_PG_URL=postgresql://postgres:t@127.0.0.1:15499/postgres \\
        pytest tests/test_knowledge_pg.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)")

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_kb_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKEND = os.path.join(os.path.dirname(__file__), "..")
MANIFEST = os.path.join(BACKEND, "..", "docs", "knowledge", "holdout-manifest-v1.json")
MANIFEST_SHA = "ea5f5d52ce1d5b5d8485fb5d29116090d9cf337113ca35405f843bf194d2766f"
T0 = datetime(2026, 10, 27, tzinfo=timezone.utc)


def _database(admin_url: str, name: str) -> str:
    from sqlalchemy.engine import make_url

    return make_url(admin_url).set(database=name).render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    import app.models  # noqa: F401 — registers every table, including kb_
    from app.db import Base

    name = "kb_" + uuid.uuid4().hex[:12]
    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    eng = create_engine(_database(_PG_URL, name), pool_size=30, max_overflow=0)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture(scope="module")
def Session(engine):
    from sqlalchemy.orm import sessionmaker

    from app.knowledge import store

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    s = factory()
    store.load_eval_manifest(s, open(MANIFEST, "rb").read(), expected_sha256=MANIFEST_SHA)
    s.commit()
    s.close()
    return factory


@pytest.fixture()
def s(Session):
    session = Session()
    yield session
    session.rollback()
    session.close()


REAL_IDENTITY_TESTS = {"test_drain_makes_a_no_longer_live_publication_historical",
                       "test_drain_uses_the_locked_row_not_a_cached_run",
                       "test_feedback_sync_never_lets_an_older_queued_final_overtake_the_live_one"}


@pytest.fixture(autouse=True)
def _stub_live_identity(request, monkeypatch):
    """Ordering tests model publications without real files: every queued
    item names the live final unless the test exercises real identities."""
    if request.node.originalname not in REAL_IDENTITY_TESTS:
        from app.knowledge import store

        monkeypatch.setattr(store, "_live_identity", lambda session, run_id: "test-live")
    yield


def scope_all():
    from app.assistant_access import ProjectAllowlist
    from app.knowledge.access import KnowledgeScope

    return KnowledgeScope.build(ProjectAllowlist(all_projects=True), "*")


def scope_for(*pids):
    from app.assistant_access import ProjectAllowlist
    from app.knowledge.access import KnowledgeScope

    return KnowledgeScope.build(ProjectAllowlist(all_projects=True), ",".join(pids))


def project(s, project_type="face_swap", pid=None):
    from app.models import VideoProject

    p = VideoProject(id=pid or str(uuid.uuid4()), source_type="upload", source_ref="fixture",
                     project_type=project_type)
    s.add(p)
    s.flush()
    return p.id


def run(s, pid, path=None):
    from app.models import Run

    r = Run(project_id=pid, result_local_path=path)
    s.add(r)
    s.flush()
    return r.id


def sha(text_value: str) -> str:
    return hashlib.sha256(text_value.encode()).hexdigest()


def run_case_with_version(s, pid=None, duration=20.0):
    """Committed Run case with version 1 (no live file, so sync is a no-op)."""
    from app.knowledge import store

    pid = pid or project(s)
    rid = run(s, pid)
    case = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid)
    art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path",
                                       locator=f"/data/{rid}/final.mp4", sha256=sha(rid + "1"), project_id=pid,
                                       duration_sec=duration)
    store.append_case_version(s, case, artifact_id=art.id, reason="initial_output")
    s.commit()
    return pid, rid, case


def new_version(s, case, tag, duration=20.0):
    from app.knowledge import store

    art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path",
                                       locator=f"/data/{case.run_id}/final.mp4", sha256=sha(case.id + tag),
                                       project_id=case.project_id, duration_sec=duration)
    store.lock_run_and_cases(s, case.run_id, [case.id])
    v = store.append_case_version(s, case, artifact_id=art.id, reason="restitch")
    s.commit()
    return v


# ---------------------------------------------------------------- schema / migration


def _load_migration(filename="k1a7c3e9f205_knowledge_base_storage.py"):
    path = os.path.join(BACKEND, "alembic", "versions", filename)
    spec = importlib.util.spec_from_file_location("kb_migration_" + filename[:12], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_capture_migration():
    return _load_migration("l2b8d4f0a316_knowledge_capture.py")


def _schema(conn, schema):
    from sqlalchemy import inspect, text

    insp = inspect(conn)
    out = {}
    for table in sorted(t for t in insp.get_table_names(schema=schema) if t.startswith("kb_")):
        out[table] = {
            "columns": sorted((c["name"], str(c["type"]), c["nullable"], bool(c.get("identity")))
                              for c in insp.get_columns(table, schema=schema)),
            "checks": sorted((c["name"], " ".join(c["sqltext"].split())) for c in insp.get_check_constraints(table, schema=schema)),
            "uniques": sorted((u["name"], tuple(u["column_names"])) for u in insp.get_unique_constraints(table, schema=schema)),
            "indexes": sorted((i["name"], bool(i["unique"])) for i in insp.get_indexes(table, schema=schema)),
            "fks": sorted((tuple(f["constrained_columns"]), f["referred_table"], f["options"].get("ondelete"))
                          for f in insp.get_foreign_keys(table, schema=schema)),
            "pk": tuple(insp.get_pk_constraint(table, schema=schema)["constrained_columns"]),
            "triggers": sorted(r[0] for r in conn.execute(text(
                "SELECT tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE NOT tgisinternal "
                "AND c.relname = :t AND n.nspname = :s"), {"t": table, "s": schema})),
        }
    return out


def test_migration_matches_models_and_downgrade_refuses_with_data(engine):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import text

    mod = _load_migration()
    cap = _load_capture_migration()  # AICP-299 (S4, S13a/b, S15) on top of k1a7c3e9f205
    assert mod.down_revision == "j0e5f2a7b914"
    assert cap.down_revision == mod.revision
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA mig"))
        conn.execute(text("SET search_path TO mig"))
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
            cap.upgrade()
        migrated = _schema(conn, "mig")
        conn.execute(text("SET search_path TO public"))
        modeled = _schema(conn, "public")
        assert set(migrated) == set(modeled)
        assert len(migrated) == 28
        for table in modeled:
            assert migrated[table] == modeled[table], table
        conn.execute(text("SET search_path TO mig"))
        with Operations.context(MigrationContext.configure(conn)):
            cap.downgrade()  # empty: allowed
            mod.downgrade()
            mod.upgrade()
            cap.upgrade()
        conn.execute(text("INSERT INTO kb_rules (id, slug) VALUES ('r1', 'rule-one')"))
        with Operations.context(MigrationContext.configure(conn)), pytest.raises(RuntimeError, match="kb_rules"):
            cap.downgrade()
        conn.rollback()
        conn.execute(text("DROP SCHEMA IF EXISTS mig CASCADE"))
        conn.commit()
        conn.invalidate()  # never return a connection whose search_path pointed at a dropped schema


def test_capture_migration_downgrade_refuses_with_a_link_failure_and_restores_old_checks(engine):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    mod, cap = _load_migration(), _load_capture_migration()
    with engine.connect() as conn:
        for schema in ("mig2", "mig3"):
            conn.execute(text(f"CREATE SCHEMA {schema}"))
            conn.execute(text(f"SET search_path TO {schema}"))
            with Operations.context(MigrationContext.configure(conn)):
                mod.upgrade()
                cap.upgrade()
            conn.commit()
        conn.execute(text("SET search_path TO mig2"))
        conn.execute(text("INSERT INTO kb_link_failures (id, publication_id, run_id, kind, path, failed_at_ns) "
                          "VALUES ('f1', 'stitch:r:1', 'r', 'stitch', '/x/final.mp4', 1)"))
        conn.commit()
        conn.execute(text("SET search_path TO mig2"))
        with pytest.raises(DBAPIError, match="immutable knowledge evidence"):
            conn.execute(text("DELETE FROM kb_link_failures"))
        conn.rollback()
        conn.execute(text("SET search_path TO mig2"))
        with pytest.raises(DBAPIError, match="ck_kb_link_failure_identity"):
            conn.execute(text("INSERT INTO kb_link_failures (id, publication_id, run_id, kind, path, failed_at_ns, "
                              "st_dev) VALUES ('f2', 'stitch:r:2', 'r', 'stitch', '/x', 1, 5)"))
        conn.rollback()
        conn.execute(text("SET search_path TO mig2"))
        with Operations.context(MigrationContext.configure(conn)), \
                pytest.raises(RuntimeError, match="kb_link_failures"):
            cap.downgrade()
        conn.rollback()
        conn.execute(text("SET search_path TO mig3"))
        with Operations.context(MigrationContext.configure(conn)):
            cap.downgrade()
        checks = dict(conn.execute(text(
            "SELECT conname, pg_get_constraintdef(c.oid) FROM pg_constraint c JOIN pg_namespace n "
            "ON n.oid = c.connamespace WHERE n.nspname = 'mig3' AND conname IN "
            "('ck_kb_case_version_reason', 'ck_kb_observation_kind', 'ck_kb_capture_reason')")).all())
        assert "regeneration" not in checks["ck_kb_case_version_reason"]
        assert "regeneration" not in checks["ck_kb_capture_reason"]
        assert "input_content" not in checks["ck_kb_observation_kind"]
        assert "kb_link_failures" not in {r[0] for r in conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'mig3'"))}
        conn.rollback()
        conn.execute(text("DROP SCHEMA IF EXISTS mig2 CASCADE"))
        conn.execute(text("DROP SCHEMA IF EXISTS mig3 CASCADE"))
        conn.commit()
        conn.invalidate()


@pytest.mark.parametrize("statement", [
    "UPDATE kb_rules SET slug = 'other-slug'",
    "DELETE FROM kb_rules",
    "TRUNCATE kb_rules CASCADE",
])
def test_insert_only_tables_reject_update_delete_truncate(Session, statement):
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    s = Session()
    s.execute(text("INSERT INTO kb_rules (id, slug) VALUES (:i, :s)"),
              {"i": str(uuid.uuid4()), "s": "imm-" + uuid.uuid4().hex[:8]})
    s.commit()
    with pytest.raises(DBAPIError, match="immutable knowledge evidence"):
        s.execute(text(statement))
    s.rollback()
    s.close()


def test_orm_refuses_to_mutate_evidence(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    case.language = "es"
    with pytest.raises(ValueError, match="immutable"):
        s.flush()


# ---------------------------------------------------------------- cases / artifacts


def test_case_creation_is_idempotent_and_identity_bound(s):
    from app.knowledge import store

    pid = project(s)
    rid = run(s, pid)
    a = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                          project_id=pid, run_id=rid)
    b = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                          project_id=pid, run_id=rid)
    assert a.id == b.id
    with pytest.raises(store.KnowledgeConflict):
        store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="localisation",
                          project_id=pid, run_id=rid)
    with pytest.raises(store.KnowledgeInvalid):
        store.ensure_case(s, case_key="segment:x:y", scope="run", operation="face_swap", project_id=pid)
    with pytest.raises(store.KnowledgeInvalid):
        store.ensure_case(s, case_key="run:zz", scope="run", operation="face_swap", project_id=pid,
                          origin="import")


def test_unknown_facts_stay_null(s):
    from app.knowledge import store

    pid = project(s)
    art = store.get_or_create_artifact(s, kind="source", locator_kind="drive_file", locator="drive:abc",
                                       sha256=None, project_id=pid)
    again = store.get_or_create_artifact(s, kind="source", locator_kind="drive_file", locator="drive:abc",
                                         sha256=None, project_id=pid)
    assert art.id != again.id  # two unhashed observations are not asserted to be one content
    assert art.sha256 is None and art.duration_sec is None and art.bytes is None
    hashed = store.get_or_create_artifact(s, kind="source", locator_kind="drive_file", locator="drive:abc",
                                          sha256=sha("x"), project_id=pid)
    assert hashed.id == store.get_or_create_artifact(s, kind="source", locator_kind="drive_file",
                                                     locator="drive:abc", sha256=sha("x"), project_id=pid).id


def test_derived_case_has_exactly_one_version(s):
    from app.knowledge import store

    pid, rid, root = run_case_with_version(s)
    tail = store.ensure_case(s, case_key=store.derived_case_key("tail_edits", "t1" + rid), scope="derived",
                             operation="face_swap", project_id=pid, run_id=rid, parent_case_id=root.id)
    a1 = store.get_or_create_artifact(s, kind="tail_edit", locator_kind="local_path", locator="/t1",
                                      sha256=sha("t1" + rid), project_id=pid)
    store.append_case_version(s, tail, artifact_id=a1.id, reason="import")
    with pytest.raises(store.KnowledgeConflict):
        store.append_case_version(s, tail, artifact_id=a1.id, reason="import")


def test_version_artifact_must_share_the_case_project(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    other = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path", locator="/o",
                                         sha256=sha("o" + case.id), project_id=project(s))
    with pytest.raises(store.KnowledgeInvalid):
        store.append_case_version(s, case, artifact_id=other.id, reason="restitch")


# ---------------------------------------------------------------- feedback


def test_feedback_replay_returns_the_same_receipt_without_duplicates(s, Session):
    from sqlalchemy import func, select

    from app.knowledge import store
    from app.knowledge.models import KbFeedbackEvent

    _, _, case = run_case_with_version(s)
    args = dict(case_id=case.id, idempotency_key="k-" + case.id, expected_version_no=1, verdict="needs_repair",
                defects=[{"code": "lip_sync", "severity": "major", "start": 1.0, "end": 2.5}])
    first = store.submit_feedback(s, scope_all(), **args)
    s.commit()
    again = store.submit_feedback(s, scope_all(), **args)
    s.commit()
    assert again.replayed and again.ids == first.ids and len(first.ids) == 2
    assert s.execute(select(func.count()).select_from(KbFeedbackEvent)
                     .where(KbFeedbackEvent.case_id == case.id)).scalar() == 2
    with pytest.raises(store.KnowledgeConflict):
        store.submit_feedback(s, scope_all(), **{**args, "verdict": "rejected"})


def test_stale_version_conflicts_and_historical_is_stored_apart(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    new_version(s, case, "v2")
    with pytest.raises(store.KnowledgeConflict) as exc:
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="st-" + case.id,
                              expected_version_no=1, verdict="accepted_as_is")
    assert exc.value.current_version_no == 2
    s.rollback()
    # v1 lives at a mutable local path (final.mp4 is rewritten): S10 refuses to judge it after the fact.
    with pytest.raises(store.KnowledgeInvalid, match="not preserved"):
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="hi-" + case.id,
                              expected_version_no=1, verdict="accepted_as_is", historical=True)
    s.rollback()
    assert store.case_state(s, case.id)["verdict"] == "unreviewed"
    with pytest.raises(store.KnowledgeInvalid):
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="hi2-" + case.id,
                              expected_version_no=2, verdict="accepted_as_is", historical=True)


def test_acceptance_never_carries_to_a_new_version(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="a1-" + case.id,
                          expected_version_no=1, verdict="accepted_as_is")
    s.commit()
    state = store.case_state(s, case.id)
    assert state["verdict"] == "accepted_as_is" and state["accepted_artifact_id"] is not None
    new_version(s, case, "v2")
    state = store.case_state(s, case.id)
    assert state == {**state, "verdict": "unreviewed", "accepted_artifact_id": None,
                     "previous_accepted_version_no": 1, "latest_version_no": 2}


def test_serious_defect_cannot_coexist_with_live_acceptance(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="b1-" + case.id,
                          expected_version_no=1, verdict="accepted_as_is")
    s.commit()
    defect = [{"code": "source_face_retained", "severity": "blocker"}]
    with pytest.raises(store.KnowledgeConflict):
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="b2-" + case.id,
                              expected_version_no=1, defects=defect)
    s.rollback()
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="b3-" + case.id,
                          expected_version_no=1, defects=[{"code": "stitching", "severity": "minor"}])
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="b4-" + case.id,
                          expected_version_no=1, verdict="needs_repair", defects=defect)
    s.commit()
    assert store.case_state(s, case.id)["verdict"] == "needs_repair"


@pytest.mark.parametrize("defect", [
    {"code": "lip_sync", "severity": "major", "start": 19.0, "end": 25.0},   # past the 20 s artifact
    {"code": "lip_sync", "severity": "major", "start": 3.0, "end": 2.0},
    {"code": "lip_sync", "severity": "major", "start": 3.0},
    {"code": "other", "severity": "minor"},                                   # needs a note
    {"code": "not_a_code", "severity": "minor"},
])
def test_invalid_time_or_code_bindings_are_rejected(s, defect):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    with pytest.raises(store.KnowledgeInvalid):
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="iv-" + uuid.uuid4().hex,
                              expected_version_no=1, defects=[defect])


def test_retraction_rules(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    receipt = store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="r1-" + case.id,
                                    expected_version_no=1, verdict="accepted_as_is")
    s.commit()
    verdict_id = receipt.ids[0]
    with pytest.raises(store.KnowledgeInvalid):
        store.retract_event(s, scope_all(), event_id=verdict_id, idempotency_key="r2-" + case.id,
                            reason="x", actor_kind="importer")
    r = store.retract_event(s, scope_all(), event_id=verdict_id, idempotency_key="r3-" + case.id,
                            reason="clicked the wrong button")
    s.commit()
    assert store.case_state(s, case.id)["verdict"] == "unreviewed"
    with pytest.raises(store.KnowledgeConflict):
        store.retract_event(s, scope_all(), event_id=verdict_id, idempotency_key="r4-" + case.id, reason="again")
    s.rollback()
    with pytest.raises(store.KnowledgeInvalid):
        store.retract_event(s, scope_all(), event_id=r.ids[0], idempotency_key="r5-" + case.id, reason="undo")


def test_automatic_assessments_obey_claim_rules(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    ok = store.record_auto_assessment(s, case_id=case.id, idempotency_key="aa-" + case.id, case_version_no=1,
                                      check_kind="asr", claim="transcript_match",
                                      coverage={"seconds": [0, 20]}, limitations="ASR is not pronunciation")
    assert not ok.replayed
    for check, claim, code in (("asr", "defect_observed", None), ("frame_sample", "defect_observed", "lip_sync"),
                               ("face_similarity", "full_length", None)):
        with pytest.raises(store.KnowledgeInvalid):
            store.record_auto_assessment(s, case_id=case.id, idempotency_key="ab-" + uuid.uuid4().hex,
                                         case_version_no=1, check_kind=check, claim=claim, defect_code=code)
    assert store.case_state(s, case.id)["verdict"] == "unreviewed"  # never a verdict


def test_correction_must_start_from_a_case_version(s):
    from app.knowledge import store

    pid, _, case = run_case_with_version(s)
    before = store.latest_version(s, case.id).artifact_id
    store.record_correction(s, scope_all(), case_id=case.id, idempotency_key="c1-" + case.id,
                            before_artifact_id=before, method="manual", resolved="unknown")
    stranger = store.get_or_create_artifact(s, kind="model_output", locator_kind="local_path", locator="/s",
                                            sha256=sha("s" + case.id), project_id=pid)
    stranger_id = stranger.id
    s.commit()
    from app.knowledge.access import KnowledgeAccessDenied

    with pytest.raises(KnowledgeAccessDenied):  # not this case's artifact: unknown here (404)
        store.record_correction(s, scope_all(), case_id=case.id, idempotency_key="c2-" + case.id,
                                before_artifact_id=stranger_id, method="manual", resolved="no")
    s.rollback()
    with pytest.raises(KnowledgeAccessDenied):  # an unrelated creative of the project as "after"
        store.record_correction(s, scope_all(), case_id=case.id, idempotency_key="c3-" + case.id,
                                before_artifact_id=before, after_artifact_id=stranger_id, method="manual",
                                resolved="yes")


def test_run_case_sync_appends_when_the_live_final_changed(s, tmp_path):
    from app.knowledge import store
    from app.models import Run

    pid = project(s)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"first")
    rid = run(s, pid, str(final))
    case = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid)
    s.commit()
    first = store.sync_run_case_version(s, case)
    assert first.version_no == 1 and first.reason == "initial_output"
    assert store.sync_run_case_version(s, case) is None
    final.write_bytes(b"second")
    assert store.sync_run_case_version(s, case).version_no == 2
    s.commit()
    with pytest.raises(store.KnowledgeConflict):
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="sy-" + case.id,
                              expected_version_no=1, verdict="accepted_as_is")
    s.rollback()
    final.write_bytes(b"third")  # the feedback path syncs under the Run lock before checking
    with pytest.raises(store.KnowledgeConflict) as exc:
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="sz-" + case.id,
                              expected_version_no=2, verdict="accepted_as_is")
    assert exc.value.current_version_no == 3


# ---------------------------------------------------------------- scope


def test_out_of_scope_cases_look_missing(s):
    from app.knowledge import store
    from app.knowledge.access import KnowledgeAccessDenied

    pid, rid, case = run_case_with_version(s)
    other = scope_for(str(uuid.uuid4()))
    with pytest.raises(KnowledgeAccessDenied):
        store.submit_feedback(s, other, case_id=case.id, idempotency_key="sc-" + case.id,
                              expected_version_no=1, verdict="accepted_as_is")
    with pytest.raises(KnowledgeAccessDenied):
        store.run_timeline(s, other, rid)
    assert store.run_timeline(s, scope_for(pid), rid)["cases"][0]["id"] == case.id
    with pytest.raises(KnowledgeAccessDenied):
        store.run_timeline(s, scope_all(), str(uuid.uuid4()))


def test_scope_is_capped_by_the_assistant_allowlist():
    from app.assistant_access import ProjectAllowlist
    from app.knowledge.access import KnowledgeScope

    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    scope = KnowledgeScope.build(ProjectAllowlist(a), f"{a},{b}")
    assert scope.allows_project(a) and not scope.allows_project(b)
    assert not KnowledgeScope.build(ProjectAllowlist(all_projects=True), "").allows_project(a)
    assert not KnowledgeScope.build(ProjectAllowlist(all_projects=True), f"{a},not-a-uuid").allows_project(a)
    assert not KnowledgeScope.build(ProjectAllowlist(a), "*").allows_project(None)


def test_rule_visibility_suppresses_project_specific_text(s):
    from app.knowledge import store

    pid, _, case = run_case_with_version(s)
    common = dict(kind="empirical_guidance", recommendation={"action": "prefer", "target": "seedance-2-5"},
                  confidence="low", author_kind="operator", effective_from=T0,
                  evidence=[{"case_id": case.id, "stance": "supports"}])
    private = store.add_rule_version(s, scope_all(), slug="vis-" + pid[:8], idempotency_key="v1-" + pid,
                                     statement="Project-specific detail", **common)
    shared = store.add_rule_version(s, scope_all(), slug="vis2-" + pid[:8], idempotency_key="v2-" + pid,
                                    statement="General lesson", shareable=True,
                                    **{**common, "recommendation": {"action": "prefer",
                                                                    "target": {"model_key": "seedance-2-5"}}})
    s.commit()
    outsider = scope_for(str(uuid.uuid4()))
    assert store.rule_visibility(s, outsider, private) is None
    assert store.rule_visibility(s, outsider, shared) == "shareable"
    assert store.rule_visibility(s, scope_for(pid), private) == "full"


# ---------------------------------------------------------------- concurrency


def _parallel(n, fn):
    barrier, errors, results = threading.Barrier(n), [], []

    def worker(i):
        try:
            barrier.wait()
            results.append(fn(i))
        except Exception as exc:  # noqa: BLE001 — collected and asserted on
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results, errors


def test_concurrent_replays_of_one_request_write_once(Session):
    from sqlalchemy import func, select

    from app.knowledge import store
    from app.knowledge.models import KbFeedbackEvent

    s = Session()
    _, _, case = run_case_with_version(s)
    s.close()

    def submit(_):
        session = Session()
        try:
            receipt = store.submit_feedback(session, scope_all(), case_id=case.id, idempotency_key="cc-" + case.id,
                                            expected_version_no=1, verdict="accepted_as_is")
            session.commit()
            return receipt.ids
        finally:
            session.close()

    results, errors = _parallel(12, submit)
    assert not errors and len({r for r in results}) == 1
    s = Session()
    assert s.execute(select(func.count()).select_from(KbFeedbackEvent)
                     .where(KbFeedbackEvent.case_id == case.id)).scalar() == 1
    s.close()


def test_verdict_races_a_new_version_and_never_judges_unseen_output(Session):
    """Acceptances of v1 and the append of v2 serialize on the Run lock: every
    committed acceptance names a version that was latest when it committed."""
    from app.knowledge import store

    s = Session()
    _, _, case = run_case_with_version(s)
    s.close()

    def act(i):
        session = Session()
        try:
            if i == 0:
                return ("version", new_version(session, case, "race").version_no)
            store.submit_feedback(session, scope_all(), case_id=case.id, idempotency_key=f"race-{i}-{case.id}",
                                  expected_version_no=1, verdict="accepted_as_is")
            session.commit()
            return ("accepted", 1)
        except store.KnowledgeConflict:
            return ("conflict", None)
        finally:
            session.close()

    results, errors = _parallel(8, act)
    assert not errors
    s = Session()
    assert store.case_state(s, case.id)["latest_version_no"] == 2
    assert store.case_state(s, case.id)["verdict"] == "unreviewed"
    s.close()


def test_concurrent_rule_versions_are_numbered_without_gaps(Session):
    from app.knowledge import store

    slug = "race-" + uuid.uuid4().hex[:8]

    def add(i):
        session = Session()
        try:
            v = store.add_rule_version(session, scope_all(), slug=slug, idempotency_key=f"{slug}-{i}",
                                       kind="empirical_guidance", statement=f"take {i}",
                                       recommendation={"action": "warn", "target": "x"}, confidence="low",
                                       author_kind="operator", effective_from=T0)
            session.commit()
            return v.version_no
        finally:
            session.close()

    results, errors = _parallel(10, add)
    assert not errors and sorted(results) == list(range(1, 11))


# ---------------------------------------------------------------- attempts


def test_open_intent_is_reused_as_a_redispatch(s):
    from app.knowledge import store

    pid, rid, case = run_case_with_version(s)
    req = {"prompt": "swap", "refs": ["u1", "u2"]}
    common = dict(case_id=case.id, attempt_kind="generation", origin="automatic", provider="kie", request=req,
                  target_kind="run_segment", target_id="seg-" + rid, model_key="seedance-2-5",
                  input_hashes={"references": [sha("a"), sha("b")]})
    first, reused = store.record_attempt_intent(s, **common)
    assert not reused and first.reference_set_sha256 is not None
    again, reused = store.record_attempt_intent(s, **common, redispatch_payload={"job": "j2"})
    assert reused and again.id == first.id
    assert store.attempt_state(s, first.id)["possibly_charged_more_than_once"]
    store.record_submission(s, first.id, provider_task_id="task-" + rid, spend_call_key="call-" + rid)
    assert store.attempt_state(s, first.id)["status"] == "submitted"
    after, reused = store.record_attempt_intent(s, **common)  # same recipe after a submission = new attempt
    assert not reused and after.id != first.id
    resub, reused = store.record_attempt_intent(s, **{**common, "predecessor_task_id": "task-" + rid})
    assert not reused and resub.dispatch_key != after.dispatch_key
    unknown, _ = store.record_attempt_intent(s, **{**common, "target_id": "other",
                                                   "input_hashes": {"references": [sha("a"), None]}})
    assert unknown.reference_set_sha256 is None


def test_cost_estimate_and_settlement_are_separate_observations(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    attempt, _ = store.record_attempt_intent(s, case_id=case.id, attempt_kind="generation", origin="automatic",
                                             provider="kie", request={"p": rid}, target_kind="run_segment",
                                             target_id=rid)
    store.record_observation(s, attempt.id, kind="cost_estimate", mcredits=1_000_000, cost_source="spend_draw")
    store.record_observation(s, attempt.id, kind="cost_settled", mcredits=950_000, cost_source="spend_draw")
    state = store.attempt_state(s, attempt.id)
    assert (state["estimated_mcredits"], state["settled_mcredits"]) == (1_000_000, 950_000)
    with pytest.raises(store.KnowledgeInvalid):
        store.record_observation(s, attempt.id, kind="cost_settled", mcredits=1)
    with pytest.raises(store.KnowledgeConflict):
        store.record_submission(s, attempt.id, provider_task_id="t1")
        store.record_submission(s, attempt.id, provider_task_id="t2")


# ---------------------------------------------------------------- deliverable capture


def _final(s, case, tag):
    from app.knowledge import store

    return store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy",
                                        locator=f"finals/{tag}.mp4", sha256=sha(case.id + tag),
                                        project_id=case.project_id)


def test_capture_queue_orders_duplicates_and_stale_finals(s):
    from app.knowledge import store

    pid, rid, case = run_case_with_version(s)
    a, b = _final(s, case, "A"), _final(s, case, "B")
    q1 = store.enqueue_final_capture(s, run_id=rid, artifact_id=a.id, reason="restitch", publication_id="stitch:1", file_identity="test-live")
    assert store.enqueue_final_capture(s, run_id=rid, artifact_id=a.id, reason="restitch",
                                       publication_id="stitch:1", file_identity="test-live").id == q1.id
    store.enqueue_final_capture(s, run_id=rid, artifact_id=a.id, reason="restitch", publication_id="stitch:1b", file_identity="test-live")
    store.enqueue_final_capture(s, run_id=rid, artifact_id=b.id, reason="range_attempt",
                                publication_id="range_attempt:9", file_identity="test-live")
    store.enqueue_final_capture(s, run_id=rid, artifact_id=a.id, reason="restitch", publication_id="stitch:2", file_identity="test-live")
    s.commit()
    outcomes = [d.outcome for d in store.drain_run_captures(s, rid)]
    s.commit()
    assert outcomes == ["appended", "duplicate", "appended", "appended"]  # A, A again, B, A→B→A
    assert [v.source_record for v in store.versions(s, case.id)][1:] == ["stitch:1", "range_attempt:9", "stitch:2"]
    assert store.drain_run_captures(s, rid) == []


def test_concurrent_drains_never_double_allocate(Session):
    from app.knowledge import store

    s = Session()
    pid, rid, case = run_case_with_version(s)
    for i in range(6):
        store.enqueue_final_capture(s, run_id=rid, artifact_id=_final(s, case, f"F{i}").id, reason="restitch",
                                    publication_id=f"stitch:{i}", file_identity="test-live")
    s.commit()
    s.close()

    def drain(_):
        session = Session()
        try:
            out = [d.outcome for d in store.drain_run_captures(session, rid)]
            session.commit()
            return out
        finally:
            session.close()

    results, errors = _parallel(5, drain)
    assert not errors and sum(len(r) for r in results) == 6
    s = Session()
    assert [v.version_no for v in store.versions(s, case.id)] == list(range(1, 8))
    s.close()


# ---------------------------------------------------------------- rules / decisions


def _rule(s, slug, statement, **kw):
    from app.knowledge import store

    defaults = dict(kind="empirical_guidance", recommendation={"action": "prefer", "target": "seedance-2-5"},
                    confidence="medium", author_kind="operator", effective_from=T0)
    return store.add_rule_version(s, scope_all(), slug=slug, idempotency_key=f"{slug}-{statement}",
                                  statement=statement, **{**defaults, **kw})


def _review(s, version, status):
    from app.knowledge import store

    return store.review_rule_version(s, scope_all(), rule_version_id=version.id, status=status,
                                     idempotency_key=f"rv-{version.id}-{status}-{uuid.uuid4().hex[:6]}")


def test_effective_version_follows_supersession_and_retirement(s):
    from app.knowledge import store

    slug = "eff-" + uuid.uuid4().hex[:8]
    v1 = _rule(s, slug, "one")
    _review(s, v1, "approved")
    assert v1.id in {v.id for v in store.effective_rule_versions(s)}
    v2 = _rule(s, slug, "two")
    assert v2.supersedes_version_id == v1.id
    assert v1.id in {v.id for v in store.effective_rule_versions(s)}  # v2 not yet approved
    _review(s, v2, "approved")
    ids = {v.id for v in store.effective_rule_versions(s)}
    assert v2.id in ids and v1.id not in ids
    _review(s, v2, "retired")
    ids = {v.id for v in store.effective_rule_versions(s)}
    assert v1.id not in ids and v2.id not in ids  # retiring never resurrects v1
    assert v1.id not in {v.id for v in store.effective_rule_versions(s, at=T0 - timedelta(days=1))}


def test_operator_constraint_o1_and_machine_hypotheses(s):
    from app.knowledge import store

    o1 = _rule(s, "o1-" + uuid.uuid4().hex[:6], "Do not use Gemini Omni for text replacement",
               kind="operator_constraint", recommendation={"action": "warn", "target": {"model_key": "gemini-omni"}},
               confidence="operator_asserted", applicability={"operation": ["subtitle_removal"]},
               provenance={"source": "AICP-293 O1", "by": "Operator"})
    assert o1.applicability == {"operation": ["subtitle_removal"]}
    with pytest.raises(store.KnowledgeInvalid):
        _rule(s, "o1b-" + uuid.uuid4().hex[:6], "x", kind="operator_constraint", author_kind="claude")
    hyp = _rule(s, "hyp-" + uuid.uuid4().hex[:6], "maybe", kind="machine_hypothesis", author_kind="claude")
    with pytest.raises(store.KnowledgeInvalid):
        _review(s, hyp, "approved")
    with pytest.raises(store.KnowledgeInvalid):
        _rule(s, "bad-" + uuid.uuid4().hex[:6], "x", applicability={"colour": ["red"]})


def test_rule_evidence_refuses_holdout_and_foreign_events(s):
    from app.knowledge import store

    store.load_eval_manifest(s, open(MANIFEST, "rb").read(), expected_sha256=MANIFEST_SHA)
    body = json.load(open(MANIFEST))
    holdout_pid = next(g for g in body["groups"] if g["split"] == "holdout")["project_ids"][0]
    if s.get(__import__("app.models", fromlist=["VideoProject"]).VideoProject, holdout_pid) is None:
        project(s, pid=holdout_pid)
    _, _, held = run_case_with_version(s, pid=holdout_pid)
    with pytest.raises(store.KnowledgeInvalid, match="holdout"):
        _rule(s, "ho-" + uuid.uuid4().hex[:6], "x", evidence=[{"case_id": held.id, "stance": "supports"}])
    s.rollback()
    _, _, dev = run_case_with_version(s)
    _, _, dev2 = run_case_with_version(s)
    ev = store.submit_feedback(s, scope_all(), case_id=dev2.id, idempotency_key="fe-" + dev2.id,
                               expected_version_no=1, verdict="rejected")
    with pytest.raises(store.KnowledgeInvalid):
        _rule(s, "fe-" + uuid.uuid4().hex[:6], "x",
              evidence=[{"case_id": dev.id, "event_id": ev.ids[0], "stance": "supports"}])


def test_decision_pins_consulted_versions(s):
    from app.knowledge import store
    from app.knowledge.access import KnowledgeAccessDenied

    pid, _, case = run_case_with_version(s)
    v = _rule(s, "dec-" + uuid.uuid4().hex[:6], "pin me", evidence=[{"case_id": case.id, "stance": "supports"}])
    args = dict(idempotency_key="d-" + case.id, mode="shadow", requested_by="new_run_form", project_id=pid,
                context={"operation": "face_swap"}, advisor_version="advisor-0",
                recommendation={"options": [{"model_key": "seedance-2-5", "cites": [v.id]}]},
                consulted_rule_version_ids=[v.id], consulted_case_ids=[case.id])
    d = store.record_decision(s, scope_all(), **args)
    assert d.consulted_rule_version_ids == [v.id] and len(d.consulted_sha256) == 64
    assert store.record_decision(s, scope_all(), **args).id == d.id
    with pytest.raises(store.KnowledgeConflict):
        store.record_decision(s, scope_all(), **{**args, "context": {"operation": "localisation"}})
    with pytest.raises(KnowledgeAccessDenied):
        store.record_decision(s, scope_for(str(uuid.uuid4())), **{**args, "idempotency_key": "d2-" + case.id,
                                                                  "project_id": None})
    store.record_decision_outcome(s, scope_all(), decision_id=d.id, chosen={"model_key": "seedance"},
                                  overrode=True, override_note="operator preferred base 2.0")


# ---------------------------------------------------------------- invalidation / manifest / subjects


def test_invalidated_import_disappears_from_reads_but_is_kept(s):
    from app.knowledge import store

    pid = project(s)
    rid = run(s, pid)
    batch = store.start_import_batch(s, source="history", code_revision="test", dry_run=False)
    case = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid, origin="import", import_batch_id=batch.id)
    store.add_import_member(s, batch.id, "case", case.id)
    art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path", locator="/i",
                                       sha256=sha("imp" + rid), project_id=pid)
    store.append_case_version(s, case, artifact_id=art.id, reason="import")
    rv = _rule(s, "imp-" + uuid.uuid4().hex[:6], "imported", import_batch_id=batch.id,
               evidence=[{"case_id": case.id, "stance": "supports"}])
    _review(s, rv, "approved")
    s.commit()
    assert store.run_timeline(s, scope_all(), rid)["cases"]
    assert rv.id in {v.id for v in store.effective_rule_versions(s)}
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="wrong source mapping")
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="again")  # idempotent
    s.commit()
    assert store.run_timeline(s, scope_all(), rid)["cases"] == []
    assert rv.id not in {v.id for v in store.effective_rule_versions(s)}
    assert store.rule_evidence(s, rv.id) == []
    assert s.get(type(case), case.id) is not None  # nothing physically deleted


def test_manifest_is_verified_and_membership_comes_from_it(s):
    from app.knowledge import store

    raw = open(MANIFEST, "rb").read()
    store.load_eval_manifest(s, raw, expected_sha256=MANIFEST_SHA)
    tampered = json.loads(raw)
    tampered["groups"][0]["split"] = "holdout" if tampered["groups"][0]["split"] == "development" else "development"
    with pytest.raises(store.KnowledgeInvalid):
        store.load_eval_manifest(s, json.dumps(tampered).encode(), expected_sha256=MANIFEST_SHA)
    group = next(g for g in json.loads(raw)["groups"] if g["split"] == "development")
    from app.models import VideoProject
    if s.get(VideoProject, group["project_ids"][0]) is None:
        project(s, pid=group["project_ids"][0])
    _, _, case = run_case_with_version(s, pid=group["project_ids"][0])
    m = store.assign_eval_membership(s, case.id, "1")
    assert (m.split, m.group_key) == ("development", group["group_key"])
    assert store.assign_eval_membership(s, case.id, "1").id == m.id
    _, _, outside = run_case_with_version(s)
    with pytest.raises(store.KnowledgeInvalid):
        store.assign_eval_membership(s, outside.id, "1")


def test_subject_links_reject_overlaps_with_another_subject(s):
    from app.knowledge import store

    pid = project(s)
    label = "subject-" + pid[:8]
    store.link_subject(s, scope_all(), label=label, project_id=pid, time_start_sec=0.0, time_end_sec=10.0)
    store.link_subject(s, scope_all(), label=label, project_id=pid, time_start_sec=5.0, time_end_sec=12.0)
    with pytest.raises(store.KnowledgeConflict):
        store.link_subject(s, scope_all(), label=label + "-b", project_id=pid, time_start_sec=9.0,
                           time_end_sec=11.0)
    store.link_subject(s, scope_all(), label=label + "-b", project_id=pid, time_start_sec=12.0, time_end_sec=20.0)


# ---------------------------------------------------------------- HTTP


def test_rule_api_round_trip(engine, Session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.db import get_db
    from app.main import app

    def db():
        session = Session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    monkeypatch.setattr(settings, "KNOWLEDGE_ENABLED", True)
    monkeypatch.setattr(settings, "ASSISTANT_ALL_PROJECTS", True)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", "*")
    app.dependency_overrides[get_db] = db
    try:
        client = TestClient(app)
        slug = "api-" + uuid.uuid4().hex[:8]
        body = {"slug": slug, "idempotency_key": "api-" + slug, "kind": "operator_constraint",
                "statement": "Use base Seedance 2.0 when comparing with 2.5 (O2)",
                "recommendation": {"action": "prefer", "target": {"model_key": "seedance"}},
                "confidence": "operator_asserted", "effective_from": T0.isoformat()}
        r = client.post("/api/v2/knowledge/rules", json=body)
        assert r.status_code == 201, r.text
        assert client.post("/api/v2/knowledge/rules", json=body).json() == r.json()  # replay
        assert client.post("/api/v2/knowledge/rules", json={**body, "idempotency_key": "x-" + slug}).status_code == 409
        vid = r.json()["rule_version_id"]
        assert client.post(f"/api/v2/knowledge/rule-versions/{vid}/reviews",
                           json={"idempotency_key": "rv-" + slug, "status": "approved"}).status_code == 201
        rules = {x["slug"]: x for x in client.get("/api/v2/knowledge/rules").json()["rules"]}
        assert rules[slug]["versions"][0]["effective"] is True
        assert rules[slug]["versions"][0]["author_verified"] is False
        assert client.post(f"/api/v2/knowledge/rules/{r.json()['rule_id']}/versions",
                           json={k: v for k, v in body.items() if k != "slug"} | {"idempotency_key": "v2-" + slug}
                           ).json()["version_no"] == 2
        assert client.get(f"/api/v2/knowledge/runs/{uuid.uuid4()}/timeline").status_code == 404
        monkeypatch.setattr(settings, "KNOWLEDGE_ENABLED", False)
        assert client.get("/api/v2/knowledge/rules").status_code == 404
    finally:
        app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------- review round 1 (Terra) regressions


def test_rule_replay_checks_evidence_scope_before_revealing_a_receipt(s):
    from app.knowledge import store
    from app.knowledge.access import KnowledgeAccessDenied

    pid, _, case = run_case_with_version(s)
    slug = "rp-" + uuid.uuid4().hex[:6]
    kw = dict(slug=slug, idempotency_key="rp-" + slug, kind="empirical_guidance", statement="s",
              recommendation={"action": "warn", "target": "x"}, confidence="low", author_kind="operator",
              effective_from=T0, evidence=[{"case_id": case.id, "stance": "supports"}])
    store.add_rule_version(s, scope_all(), **kw)
    s.commit()
    with pytest.raises(KnowledgeAccessDenied):
        store.add_rule_version(s, scope_for(str(uuid.uuid4())), **kw)


def test_invalidated_rule_versions_are_not_browsable(engine, Session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.db import get_db
    from app.knowledge import store
    from app.main import app

    s = Session()
    slug = "inv-" + uuid.uuid4().hex[:6]
    v = _rule(s, slug, "withdrawn text")
    store.invalidate(s, target_kind="rule_version", target_id=v.id, reason="bad import")
    s.commit()
    s.close()

    def db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    for name, value in (("KNOWLEDGE_ENABLED", True), ("ASSISTANT_ALL_PROJECTS", True),
                        ("KNOWLEDGE_PROJECT_IDS", "*")):
        monkeypatch.setattr(settings, name, value)
    app.dependency_overrides[get_db] = db
    try:
        rules = TestClient(app).get("/api/v2/knowledge/rules").json()["rules"]
        assert slug not in {r["slug"] for r in rules}
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_manifest_amendment_releases_a_former_holdout_project(s):
    from app.knowledge import store
    from app.models import VideoProject

    raw = open(MANIFEST, "rb").read()
    store.load_eval_manifest(s, raw, expected_sha256=MANIFEST_SHA)
    body = json.loads(raw)
    group = next(g for g in body["groups"] if g["split"] == "holdout")
    pid = group["project_ids"][0]
    if s.get(VideoProject, pid) is None:
        project(s, pid=pid)
    _, _, case = run_case_with_version(s, pid=pid)
    store.assign_eval_membership(s, case.id, "1")
    assert store.is_holdout(s, case)
    v2 = {**body, "version": 2, "groups": [{**g, "split": "development"} if g is group else g
                                           for g in body["groups"]]}
    v2.pop("manifest_sha256")
    v2["manifest_sha256"] = store.manifest_sha256(v2)
    store.load_eval_manifest(s, json.dumps(v2).encode(), expected_sha256=v2["manifest_sha256"])
    assert not store.is_holdout(s, case)


def test_retracted_event_stops_supporting_a_rule(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    r = store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="rt-" + case.id,
                              expected_version_no=1, verdict="rejected")
    v = _rule(s, "rt-" + uuid.uuid4().hex[:6], "cites a verdict",
              evidence=[{"case_id": case.id, "event_id": r.ids[0], "stance": "supports"}])
    assert len(store.rule_evidence(s, v.id)) == 1
    store.retract_event(s, scope_all(), event_id=r.ids[0], idempotency_key="rt2-" + case.id, reason="mistake")
    assert store.rule_evidence(s, v.id) == []


def test_invalidated_imported_observation_no_longer_sets_attempt_state(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    attempt, _ = store.record_attempt_intent(s, case_id=case.id, attempt_kind="generation", origin="automatic",
                                             provider="kie", request={"p": "obs" + rid}, target_kind="run_segment",
                                             target_id="obs" + rid)
    store.record_observation(s, attempt.id, kind="cost_settled", mcredits=5, cost_source="spend_draw")
    batch = store.start_import_batch(s, source="draws", code_revision="t", dry_run=False)
    store.record_observation(s, attempt.id, kind="cost_settled", mcredits=999, cost_source="spend_draw",
                             import_batch_id=batch.id)
    store.record_observation(s, attempt.id, kind="status", status="succeeded", import_batch_id=batch.id)
    assert store.attempt_state(s, attempt.id)["settled_mcredits"] == 999
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="wrong draw mapping")
    state = store.attempt_state(s, attempt.id)
    assert (state["settled_mcredits"], state["status"]) == (5, None)



# ---------------------------------------------------------------- review round 2 (Terra) regressions


def _holdout_case(s):
    from app.models import VideoProject

    body = json.load(open(MANIFEST))
    # The last holdout group: the amendment test releases the first one.
    pid = [g for g in body["groups"] if g["split"] == "holdout"][-1]["project_ids"][-1]
    if s.get(VideoProject, pid) is None:
        project(s, pid=pid)
    return run_case_with_version(s, pid=pid)


def test_timeline_hides_holdout_operator_labels(s):
    from app.knowledge import store

    _, rid, case = _holdout_case(s)
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="hl-" + case.id,
                          expected_version_no=1, verdict="rejected",
                          defects=[{"code": "lip_sync", "severity": "major", "note": "secret"}])
    s.commit()
    view = next(c for c in store.run_timeline(s, scope_all(), rid)["cases"] if c["id"] == case.id)
    assert view["holdout"] and view["state"]["verdict"] == "hidden_holdout" and view["events"] == []


def test_decision_refuses_a_withdrawn_rule(s):
    from app.knowledge import store

    v = _rule(s, "wd-" + uuid.uuid4().hex[:6], "withdrawn")
    store.invalidate(s, target_kind="rule_version", target_id=v.id, reason="bad")
    with pytest.raises(store.KnowledgeConflict):
        store.record_decision(s, scope_all(), idempotency_key="wd-" + v.id, mode="shadow",
                              requested_by="offline_eval", context={}, advisor_version="a",
                              recommendation={}, consulted_rule_version_ids=[v.id])


def test_without_a_manifest_evidence_fails_closed_and_late_holdout_rules_never_apply(engine):
    """A fresh database: evidence before any manifest is refused; a rule that got
    holdout evidence anyway (written pre-manifest via the importer path) stops
    being effective once the manifest is loaded."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from app.db import Base
    from app.knowledge import store
    from app.knowledge.models import KbRuleEvidence

    name = "kbe_" + uuid.uuid4().hex[:10]
    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    eng = create_engine(_database(_PG_URL, name))
    try:
        Base.metadata.create_all(eng)
        s = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)()
        _, _, held = _holdout_case(s)
        with pytest.raises(store.KnowledgeInvalid, match="manifest"):
            _rule(s, "nm-" + uuid.uuid4().hex[:6], "x", evidence=[{"case_id": held.id, "stance": "supports"}])
        s.rollback()
        _, _, held = _holdout_case(s)
        v = _rule(s, "late-" + uuid.uuid4().hex[:6], "late")
        s.add(KbRuleEvidence(rule_version_id=v.id, case_id=held.id, stance="supports"))
        _review(s, v, "approved")
        s.commit()
        assert v.id in {x.id for x in store.effective_rule_versions(s)}
        store.load_eval_manifest(s, open(MANIFEST, "rb").read(), expected_sha256=MANIFEST_SHA)
        s.commit()
        assert v.id not in {x.id for x in store.effective_rule_versions(s)}
        s.close()
    finally:
        eng.dispose()
        with admin.connect() as c:
            c.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
        admin.dispose()


# ---------------------------------------------------------------- review round 3 (Terra) regressions


def test_acceptance_is_refused_while_a_serious_defect_is_live(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    d = store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="ld-" + case.id,
                              expected_version_no=1, defects=[{"code": "lip_sync", "severity": "major"}])
    s.commit()
    with pytest.raises(store.KnowledgeConflict):
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="ld2-" + case.id,
                              expected_version_no=1, verdict="accepted_as_is")
    s.rollback()
    store.retract_event(s, scope_all(), event_id=d.ids[0], idempotency_key="ld3-" + case.id, reason="fine")
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="ld4-" + case.id,
                          expected_version_no=1, verdict="accepted_as_is")
    assert store.case_state(s, case.id)["verdict"] == "accepted_as_is"


def test_capture_refuses_another_projects_artifact(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    foreign = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy",
                                           locator="finals/foreign.mp4", sha256=sha("foreign" + rid),
                                           project_id=project(s))
    with pytest.raises(store.KnowledgeInvalid):
        store.enqueue_final_capture(s, run_id=rid, artifact_id=foreign.id, reason="restitch",
                                    publication_id="stitch:x", file_identity="test-live")


def test_drain_compares_against_the_latest_valid_version(s):
    from app.knowledge import store

    pid = project(s)
    rid = run(s, pid)
    batch = store.start_import_batch(s, source="history", code_revision="t", dry_run=False)
    case = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid)
    art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy",
                                       locator="finals/same.mp4", sha256=sha("same" + rid), project_id=pid)
    v1 = store.append_case_version(s, case, artifact_id=art.id, reason="import")
    store.add_import_member(s, batch.id, "case_version", v1.id)
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="bad import")
    store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch", publication_id="stitch:live", file_identity="test-live")
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["appended"]
    assert store.latest_version(s, case.id).version_no == 2


# ---------------------------------------------------------------- review round 4 (Terra) regressions


def test_global_decision_outcome_scope_checks_its_attempt(s):
    from app.knowledge import store
    from app.knowledge.access import KnowledgeAccessDenied

    _, rid, case = run_case_with_version(s)
    attempt, _ = store.record_attempt_intent(s, case_id=case.id, attempt_kind="generation", origin="automatic",
                                             provider="kie", request={"o": rid})
    d = store.record_decision(s, scope_all(), idempotency_key="go-" + rid, mode="shadow",
                              requested_by="offline_eval", context={}, advisor_version="a", recommendation={})
    with pytest.raises(KnowledgeAccessDenied):
        store.record_decision_outcome(s, scope_for(str(uuid.uuid4())), decision_id=d.id, chosen={},
                                      overrode=False, attempt_id=attempt.id)


def test_retraction_cannot_revive_an_acceptance_over_a_live_defect(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="rv1-" + case.id,
                          expected_version_no=1, verdict="accepted_as_is")
    repair = store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="rv2-" + case.id,
                                   expected_version_no=1, verdict="needs_repair",
                                   defects=[{"code": "lip_sync", "severity": "major"}])
    s.commit()
    with pytest.raises(store.KnowledgeConflict):
        store.retract_event(s, scope_all(), event_id=repair.ids[0], idempotency_key="rv3-" + case.id,
                            reason="oops")
    s.rollback()
    assert store.case_state(s, case.id)["verdict"] == "needs_repair"


def test_invalidated_attempt_exposes_no_state(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    attempt, _ = store.record_attempt_intent(s, case_id=case.id, attempt_kind="generation", origin="automatic",
                                             provider="kie", request={"i": rid})
    store.record_submission(s, attempt.id, provider_task_id="ti-" + rid)
    store.record_observation(s, attempt.id, kind="cost_settled", mcredits=7, cost_source="spend_draw")
    store.invalidate(s, target_kind="attempt", target_id=attempt.id, reason="wrong case")
    state = store.attempt_state(s, attempt.id)
    assert state["invalidated"] and state["provider_task_id"] is None and state["settled_mcredits"] is None


# ---------------------------------------------------------------- review round 5 (Terra) regressions


def test_shareable_rule_hides_free_text_outside_scope(engine, Session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.db import get_db
    from app.knowledge import store
    from app.main import app

    s = Session()
    _, _, case = run_case_with_version(s)
    slug = "sh-" + uuid.uuid4().hex[:6]
    v = _rule(s, slug, "general lesson", shareable=True, evidence_gaps=["project A gap"],
              recommendation={"action": "prefer", "target": {"model_key": "seedance-2-5"}},
              author_label="op-a", evidence=[{"case_id": case.id, "stance": "supports", "note": "A detail"}])
    store.review_rule_version(s, scope_all(), rule_version_id=v.id, status="approved",
                              idempotency_key="sh-" + v.id, reviewer_label="rev", note="seen on project A")
    s.commit()
    s.close()

    def db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    for name, value in (("KNOWLEDGE_ENABLED", True), ("ASSISTANT_ALL_PROJECTS", True),
                        ("KNOWLEDGE_PROJECT_IDS", str(uuid.uuid4()))):
        monkeypatch.setattr(settings, name, value)
    app.dependency_overrides[get_db] = db
    try:
        rules = {r["slug"]: r for r in TestClient(app).get("/api/v2/knowledge/rules").json()["rules"]}
        out = rules[slug]["versions"][0]
        assert out["statement"] == "general lesson" and out["evidence"] == []
        assert out["out_of_scope_evidence_count"] == 1
        assert out["evidence_gaps"] == [] and out["author_label"] is None
        assert out["reviews"][0]["note"] is None and out["reviews"][0]["reviewer_label"] is None
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_decision_refuses_a_holdout_backed_rule(s):
    from app.knowledge import store
    from app.knowledge.models import KbRuleEvidence

    _, _, held = _holdout_case(s)
    v = _rule(s, "hd-" + uuid.uuid4().hex[:6], "legacy")
    s.add(KbRuleEvidence(rule_version_id=v.id, case_id=held.id, stance="supports"))  # pre-manifest legacy row
    s.flush()
    with pytest.raises(store.KnowledgeConflict, match="holdout"):
        store.record_decision(s, scope_all(), idempotency_key="hd-" + v.id, mode="shadow",
                              requested_by="offline_eval", context={}, advisor_version="a", recommendation={},
                              consulted_rule_version_ids=[v.id])


def test_import_member_kinds_are_closed(s):
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    from app.knowledge import store

    batch = store.start_import_batch(s, source="history", code_revision="t", dry_run=True)
    with pytest.raises(store.KnowledgeInvalid):
        store.add_import_member(s, batch.id, "case_versions", "x")
    with pytest.raises(IntegrityError):
        s.execute(text("INSERT INTO kb_import_members (id, batch_id, target_kind, target_id) "
                       "VALUES ('m1', :b, 'case_versions', 'x')"), {"b": batch.id})



# ---------------------------------------------------------------- review round 6 (Terra) regressions


def test_corrected_import_replaces_a_withdrawn_case_and_artifact(s):
    from app.knowledge import store

    pid = project(s)
    rid = run(s, pid)
    bad = store.start_import_batch(s, source="history", code_revision="t", dry_run=False)
    old = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="localisation",
                            project_id=pid, run_id=rid, origin="import", import_batch_id=bad.id)
    art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path", locator="/c",
                                       sha256=sha("c" + rid), project_id=pid)
    store.add_import_member(s, bad.id, "case", old.id)
    store.add_import_member(s, bad.id, "artifact", art.id)
    store.invalidate(s, target_kind="import_batch", target_id=bad.id, reason="wrong operation")
    good = store.start_import_batch(s, source="history", code_revision="t2", dry_run=False)
    new = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                            project_id=pid, run_id=rid, origin="import", import_batch_id=good.id)
    art2 = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path", locator="/c",
                                        sha256=sha("c" + rid), project_id=pid)
    assert (new.id, new.revision, art2.revision) != (old.id, 1, 1) and new.revision == 2 and art2.revision == 2
    store.append_case_version(s, new, artifact_id=art2.id, reason="import")
    s.commit()
    view = store.run_timeline(s, scope_all(), rid)["cases"]
    assert [c["id"] for c in view] == [new.id] and view[0]["state"]["latest_version_no"] == 1


def test_shareable_rules_carry_no_project_structure(s):
    from app.knowledge import store

    for kw in ({"applicability": {"project_id": [str(uuid.uuid4())]},
                "recommendation": {"action": "warn", "target": {"model_key": "gemini-omni"}}},
               {"recommendation": {"action": "warn", "target": "use what worked on project A"}},
               {"recommendation": {"action": "warn", "target": {"note": "A"}}}):
        with pytest.raises(store.KnowledgeInvalid):
            _rule(s, "shs-" + uuid.uuid4().hex[:6], "x", shareable=True, **kw)


def test_withdrawn_artifact_is_never_captured_as_current(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    art = _final(s, case, "W")
    store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch", publication_id="stitch:w", file_identity="test-live")
    store.invalidate(s, target_kind="artifact", target_id=art.id, reason="hash mismatch")
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["withdrawn"]
    assert store.latest_version(s, case.id).version_no == 1
    with pytest.raises(store.KnowledgeInvalid):
        store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch",
                                    publication_id="stitch:w2", file_identity="test-live")


# ---------------------------------------------------------------- review round 7 (Terra) regressions


def test_decision_refuses_a_holdout_case(s):
    from app.knowledge import store

    _, _, held = _holdout_case(s)
    with pytest.raises(store.KnowledgeConflict, match="holdout"):
        store.record_decision(s, scope_all(), idempotency_key="hc-" + held.id, mode="shadow",
                              requested_by="offline_eval", context={}, advisor_version="a", recommendation={},
                              consulted_case_ids=[held.id])


def test_children_survive_a_root_replacement(s):
    from app.knowledge import store

    pid = project(s)
    rid = run(s, pid)
    bad = store.start_import_batch(s, source="history", code_revision="t", dry_run=False)
    old = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                            project_id=pid, run_id=rid, origin="import", import_batch_id=bad.id)
    tail = store.ensure_case(s, case_key=store.derived_case_key("run_tail_edits", rid), scope="derived",
                             operation="face_swap", project_id=pid, run_id=rid, parent_case_id=old.id)
    store.invalidate(s, target_kind="case", target_id=old.id, reason="wrong snapshot")
    new = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                            project_id=pid, run_id=rid)
    s.commit()
    assert [c["id"] for c in store.run_timeline(s, scope_all(), rid)["cases"]] == [new.id, tail.id]


def test_replacement_artifact_gets_its_own_capture(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    first = _final(s, case, "R")
    q1 = store.enqueue_final_capture(s, run_id=rid, artifact_id=first.id, reason="restitch",
                                     publication_id="stitch:r", file_identity="test-live")
    store.invalidate(s, target_kind="artifact", target_id=first.id, reason="bad metadata")
    second = _final(s, case, "R")
    assert second.id != first.id and second.revision == 2
    q2 = store.enqueue_final_capture(s, run_id=rid, artifact_id=second.id, reason="restitch",
                                     publication_id="stitch:r", file_identity="test-live")
    assert q2.id != q1.id
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["withdrawn", "appended"]


def test_withdrawn_knowledge_never_authorizes_a_deleted_run(s):
    from app.knowledge import store
    from app.knowledge.access import KnowledgeAccessDenied

    pid = project(s)
    gone = str(uuid.uuid4())  # no Run row: deleted
    case = store.ensure_case(s, case_key=store.run_case_key(gone), scope="run", operation="face_swap",
                             project_id=pid, run_id=gone)
    assert store.run_timeline(s, scope_for(pid), gone)["cases"][0]["id"] == case.id
    store.invalidate(s, target_kind="case", target_id=case.id, reason="x")
    with pytest.raises(KnowledgeAccessDenied):
        store.run_timeline(s, scope_for(pid), gone)


# ---------------------------------------------------------------- review round 8 (Terra) regressions


def _intent(s, case, **kw):
    from app.knowledge import store

    base = dict(case_id=case.id, attempt_kind="generation", origin="automatic", provider="kie",
                request={"r": case.id}, target_kind="run_segment", target_id="t-" + case.id,
                model_key="seedance-2-5")
    return store.record_attempt_intent(s, **{**base, **kw})


def test_withdrawn_rows_do_not_decide_whether_an_intent_is_open(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    first, _ = _intent(s, case)
    batch = store.start_import_batch(s, source="draws", code_revision="t", dry_run=False)
    store.record_observation(s, first.id, kind="status", status="failed", import_batch_id=batch.id)
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="wrong task mapping")
    again, reused = _intent(s, case)
    assert reused and again.id == first.id  # the withdrawn terminal status no longer closes it
    store.invalidate(s, target_kind="attempt", target_id=first.id, reason="wrong case")
    fresh, reused = _intent(s, case)
    assert not reused and fresh.id != first.id  # a withdrawn intent is never reused


def test_attempt_replay_must_match_model_provenance(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    first, _ = _intent(s, case, idempotency_key="ak-" + case.id)
    assert _intent(s, case, idempotency_key="ak-" + case.id)[0].id == first.id
    with pytest.raises(store.KnowledgeConflict):
        _intent(s, case, idempotency_key="ak-" + case.id, model_key="seedance")
    other, reused = _intent(s, case, model_key="seedance")  # same recipe, different model: not the same dispatch
    assert not reused and other.id != first.id


def test_concurrent_submission_replays_do_not_fail(Session):
    from app.knowledge import store

    s = Session()
    _, rid, case = run_case_with_version(s)
    attempt, _ = _intent(s, case)
    s.commit()
    s.close()

    def submit(_):
        session = Session()
        try:
            sub = store.record_submission(session, attempt.id, provider_task_id="cs-" + rid, spend_call_key="k-" + rid)
            session.commit()
            return sub.id
        finally:
            session.close()

    results, errors = _parallel(8, submit)
    assert not errors and len(set(results)) == 1


# ---------------------------------------------------------------- review round 9 (Terra) regressions


def test_withdrawn_attempt_is_neither_replayed_nor_listed(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    first, _ = _intent(s, case, idempotency_key="wa-" + case.id)
    store.invalidate(s, target_kind="attempt", target_id=first.id, reason="wrong case")
    s.commit()
    with pytest.raises(store.KnowledgeConflict, match="withdrawn"):
        _intent(s, case, idempotency_key="wa-" + case.id)
    s.rollback()
    view = next(c for c in store.run_timeline(s, scope_all(), rid)["cases"] if c["id"] == case.id)
    assert first.id not in {a["id"] for a in view["attempts"]}


# ---------------------------------------------------------------- AICP-299 hook design review, risk R4


def test_feedback_sync_never_lets_an_older_queued_final_overtake_the_live_one(s, tmp_path):
    """Queue [A, B], live file = B. A feedback sync must not leave A to be appended over B."""
    from app.knowledge import store

    pid = project(s)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"B")
    rid = run(s, pid, str(final))
    case = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid)
    a = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy", locator="finals/A",
                                     sha256=hashlib.sha256(b"A").hexdigest(), project_id=pid)
    b = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy", locator="finals/B",
                                     sha256=hashlib.sha256(b"B").hexdigest(), project_id=pid)
    store.enqueue_final_capture(s, run_id=rid, artifact_id=a.id, reason="restitch", publication_id="stitch:a",
                                file_identity="0:0:0")  # A's inode, since replaced by B
    store.enqueue_final_capture(s, run_id=rid, artifact_id=b.id, reason="restitch", publication_id="stitch:b",
                                file_identity=store.file_identity(str(final)))
    s.commit()
    store.lock_run_and_cases(s, rid, [case.id])
    store.sync_run_case_version(s, case)
    s.commit()
    store.drain_run_captures(s, rid)
    s.commit()
    shas = [s.get(type(b), v.artifact_id).sha256 for v in store.versions(s, case.id)]
    assert shas == [hashlib.sha256(b"A").hexdigest(), hashlib.sha256(b"B").hexdigest()]  # A never current after B


# ---------------------------------------------------------------- S16: historical versions + publication uniqueness


def test_historical_version_never_becomes_current_and_is_judged_only_historically(s):
    from app.knowledge import store

    _, _, case = run_case_with_version(s)
    old = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy",
                                       locator="finals/late-proven", sha256=sha("late" + case.id),
                                       project_id=case.project_id, duration_sec=20.0)
    store.lock_run_and_cases(s, case.run_id, [case.id])
    h = store.append_case_version(s, case, artifact_id=old.id, reason="restitch", source_record="stitch:late",
                                  historical=True, published_at=T0)
    s.commit()
    assert h.version_no == 2
    state = store.case_state(s, case.id)
    assert state["latest_version_no"] == 1 and state["verdict"] == "unreviewed"
    with pytest.raises(store.KnowledgeConflict) as exc:
        store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="h1-" + case.id,
                              expected_version_no=2, verdict="accepted_as_is")
    assert exc.value.current_version_no == 1
    s.rollback()
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="h2-" + case.id,
                          expected_version_no=2, verdict="rejected", historical=True)
    store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="h3-" + case.id,
                          expected_version_no=1, verdict="accepted_as_is")
    s.commit()
    assert store.case_state(s, case.id)["verdict"] == "accepted_as_is"
    view = next(c for c in store.run_timeline(s, scope_all(), case.run_id)["cases"] if c["id"] == case.id)
    assert [v["historical"] for v in view["versions"]] == [False, True]


def test_one_publication_is_one_version(s):
    from sqlalchemy.exc import IntegrityError

    from app.knowledge import store
    from app.knowledge.models import KbCaseVersion

    pid, rid, case = run_case_with_version(s)
    a = _final(s, case, "P")
    store.lock_run_and_cases(s, rid, [case.id])
    v = store.append_case_version(s, case, artifact_id=a.id, reason="range_attempt", source_record="range_attempt:7")
    s.commit()
    store.lock_run_and_cases(s, rid, [case.id])
    assert store.append_case_version(s, case, artifact_id=a.id, reason="range_attempt",
                                     source_record="range_attempt:7").id == v.id
    with pytest.raises(store.KnowledgeConflict):
        store.append_case_version(s, case, artifact_id=_final(s, case, "Q").id, reason="range_attempt",
                                  source_record="range_attempt:7")
    s.rollback()
    store.enqueue_final_capture(s, run_id=rid, artifact_id=_final(s, case, "R").id, reason="restitch",
                                publication_id="stitch:r1", file_identity="test-live")
    s.commit()
    store.drain_run_captures(s, rid)
    s.commit()
    store.enqueue_final_capture(s, run_id=rid, artifact_id=_final(s, case, "S").id, reason="restitch",
                                publication_id="stitch:r1", file_identity="test-live")  # same publication, other bytes: a retry artefact
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["duplicate"]
    with pytest.raises(IntegrityError):
        s.add(KbCaseVersion(case_id=case.id, version_no=99, artifact_id=a.id, reason="restitch",
                            source_record="stitch:r1"))
        s.flush()


def test_derived_cases_have_no_history(s):
    from app.knowledge import store

    pid, rid, root = run_case_with_version(s)
    tail = store.ensure_case(s, case_key=store.derived_case_key("run_tail_edits", "h" + rid), scope="derived",
                             operation="face_swap", project_id=pid, run_id=rid, parent_case_id=root.id)
    art = store.get_or_create_artifact(s, kind="tail_edit", locator_kind="local_path", locator="/th",
                                       sha256=sha("th" + rid), project_id=pid)
    with pytest.raises(store.KnowledgeInvalid):
        store.append_case_version(s, tail, artifact_id=art.id, reason="import", historical=True)


# ---------------------------------------------------------------- C2 live revalidation + historical auto assessments


def test_drain_makes_a_no_longer_live_publication_historical(s, tmp_path):
    from app.knowledge import store

    pid = project(s)
    live = tmp_path / "final.mp4"
    live.write_bytes(b"B")
    rid = run(s, pid, str(live))
    case = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid)
    gone = tmp_path / "old.mp4"
    gone.write_bytes(b"A")
    a = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy", locator="finals/A2",
                                     sha256=hashlib.sha256(b"A").hexdigest(), project_id=pid)
    b = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy", locator="finals/B2",
                                     sha256=hashlib.sha256(b"B").hexdigest(), project_id=pid)
    store.enqueue_final_capture(s, run_id=rid, artifact_id=a.id, reason="restitch", publication_id="stitch:A",
                                file_identity=store.file_identity(str(gone)))  # a promotion replaced it since
    store.enqueue_final_capture(s, run_id=rid, artifact_id=b.id, reason="range_attempt",
                                publication_id="range_attempt:B", file_identity=store.file_identity(str(live)))
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["historical", "appended"]
    s.commit()
    vs = store.versions(s, case.id)
    assert [(v.version_no, v.historical) for v in vs] == [(1, True), (2, False)]
    assert store.latest_version(s, case.id).artifact_id == b.id


def test_auto_assessment_of_a_historical_version_is_historical(s):
    from app.knowledge import store
    from app.knowledge.models import KbFeedbackEvent

    _, _, case = run_case_with_version(s)
    old = _final(s, case, "HA")
    store.lock_run_and_cases(s, case.run_id, [case.id])
    h = store.append_case_version(s, case, artifact_id=old.id, reason="restitch", source_record="stitch:ha",
                                  historical=True)
    r = store.record_auto_assessment(s, case_id=case.id, idempotency_key="ha-" + case.id, case_version_no=h.version_no,
                                     check_kind="duration_probe", claim="full_length")
    assert s.get(KbFeedbackEvent, r.ids[0]).historical is True


def test_drain_uses_the_locked_row_not_a_cached_run(Session, tmp_path):
    """Round 13: a session that cached the Run before another transaction
    promoted a new final must compare against the committed path."""
    from app.knowledge import store
    from app.models import Run

    s = Session()
    pid = project(s)
    a_file, b_file = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a_file.write_bytes(b"A")
    b_file.write_bytes(b"B")
    rid = run(s, pid, str(a_file))
    store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap", project_id=pid,
                      run_id=rid)
    art = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="preserved_copy", locator="finals/ca",
                                       sha256=hashlib.sha256(b"A").hexdigest(), project_id=pid)
    store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch", publication_id="stitch:ca",
                                file_identity=store.file_identity(str(a_file)))
    s.commit()
    assert s.get(Run, rid).result_local_path == str(a_file)  # cached in s's identity map
    other = Session()
    other.get(Run, rid).result_local_path = str(b_file)
    other.commit()
    other.close()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["historical"]
    s.rollback()
    s.close()


# ---------------------------------------------------------------- review round 14 / importer r7 / design r10


def test_capture_requires_the_published_file_identity(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    with pytest.raises(store.KnowledgeInvalid):
        store.enqueue_final_capture(s, run_id=rid, artifact_id=_final(s, case, "NI").id, reason="restitch",
                                    publication_id="stitch:ni", file_identity="")


def test_feedback_sync_refuses_to_drain_a_backlog_under_the_run_lock(s, tmp_path, monkeypatch):
    from app.knowledge import store

    monkeypatch.setattr(store, "SYNC_DRAIN_MAX", 2)
    pid = project(s)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"live")
    rid = run(s, pid, str(final))
    case = store.ensure_case(s, case_key=store.run_case_key(rid), scope="run", operation="face_swap",
                             project_id=pid, run_id=rid)
    for i in range(5):
        store.enqueue_final_capture(s, run_id=rid, artifact_id=_final(s, case, f"BL{i}").id, reason="restitch",
                                    publication_id=f"stitch:bl{i}", file_identity="test-live")
    s.commit()
    store.lock_run_and_cases(s, rid, [case.id])
    with pytest.raises(store.KnowledgeConflict, match="still being recorded"):
        store.sync_run_case_version(s, case)


def test_a_withdrawn_publication_can_be_recorded_again(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    first = _final(s, case, "WP")
    store.lock_run_and_cases(s, rid, [case.id])
    v1 = store.append_case_version(s, case, artifact_id=first.id, reason="range_attempt",
                                   source_record="range_attempt:wp")
    store.invalidate(s, target_kind="artifact", target_id=first.id, reason="wrong metadata")
    replacement = _final(s, case, "WP")
    assert replacement.id != first.id
    v2 = store.append_case_version(s, case, artifact_id=replacement.id, reason="range_attempt",
                                   source_record="range_attempt:wp")
    assert v2.id != v1.id and v2.source_record == "range_attempt:wp~1"
    assert store.append_case_version(s, case, artifact_id=replacement.id, reason="range_attempt",
                                     source_record="range_attempt:wp").id == v2.id
    assert store.latest_version(s, case.id).id == v2.id


def test_ensure_run_case_reads_the_committed_run(Session):
    from app.knowledge import store
    from app.models import Run

    s = Session()
    pid = project(s)
    rid = run(s, pid)
    s.commit()
    assert s.get(Run, rid).language is None  # cached
    other = Session()
    other.get(Run, rid).language = "es"
    other.commit()
    other.close()
    store.lock_run_and_cases(s, rid, [])
    assert store.ensure_run_case(s, rid).language == "es"
    s.rollback()
    s.close()


# ---------------------------------------------------------------- review round 15


def test_drain_re_records_a_withdrawn_publication(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    first = _final(s, case, "DR")
    store.enqueue_final_capture(s, run_id=rid, artifact_id=first.id, reason="restitch", publication_id="stitch:dr",
                                file_identity="test-live")
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["appended"]
    store.invalidate(s, target_kind="artifact", target_id=first.id, reason="bad")
    replacement = _final(s, case, "DR")
    store.enqueue_final_capture(s, run_id=rid, artifact_id=replacement.id, reason="restitch",
                                publication_id="stitch:dr", file_identity="test-live")
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["appended"]
    assert store.latest_version(s, case.id).source_record == "stitch:dr~1"


def test_publication_ids_sharing_a_prefix_stay_distinct(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    store.lock_run_and_cases(s, rid, [case.id])
    a = store.append_case_version(s, case, artifact_id=_final(s, case, "PX1").id, reason="restitch",
                                  source_record="stitch:421")
    b = store.append_case_version(s, case, artifact_id=_final(s, case, "PX2").id, reason="restitch",
                                  source_record="stitch:42")
    assert a.id != b.id and b.source_record == "stitch:42"
    for reserved in ("stitch:42~1", "stitch:42~manual"):  # "~" is reserved for re-records
        with pytest.raises(store.KnowledgeInvalid):
            store.append_case_version(s, case, artifact_id=_final(s, case, "PX4").id, reason="restitch",
                                      source_record=reserved)
    with pytest.raises(store.KnowledgeInvalid):
        store.append_case_version(s, case, artifact_id=_final(s, case, "PX3").id, reason="restitch",
                                  source_record="x" * 181)



def test_auto_assessment_refuses_an_unpreserved_historical_version(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    loose = store.get_or_create_artifact(s, kind="stitched_final", locator_kind="local_path", locator="/loose.mp4",
                                         sha256=sha("loose" + case.id), project_id=case.project_id)
    store.lock_run_and_cases(s, rid, [case.id])
    h = store.append_case_version(s, case, artifact_id=loose.id, reason="restitch", source_record="stitch:loose",
                                  historical=True)
    with pytest.raises(store.KnowledgeInvalid, match="not preserved"):
        store.record_auto_assessment(s, case_id=case.id, idempotency_key="ua-" + case.id,
                                     case_version_no=h.version_no, check_kind="duration_probe", claim="full_length")


# ---------------------------------------------------------------- design r12 storage contracts


def test_auto_assessment_of_a_superseded_mutable_version_or_withdrawn_version_is_refused(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)  # v1 at a mutable local path
    new_version(s, case, "v2")
    with pytest.raises(store.KnowledgeInvalid, match="not preserved"):
        store.record_auto_assessment(s, case_id=case.id, idempotency_key="sv-" + case.id, case_version_no=1,
                                     check_kind="duration_probe", claim="full_length")
    s.rollback()
    v2 = store.latest_version(s, case.id)
    store.invalidate(s, target_kind="artifact", target_id=v2.artifact_id, reason="bad")
    with pytest.raises(store.KnowledgeInvalid, match="withdrawn"):
        store.record_auto_assessment(s, case_id=case.id, idempotency_key="sw-" + case.id, case_version_no=2,
                                     check_kind="duration_probe", claim="full_length")


def test_case_invalidation_requeues_the_publication_for_the_replacement_case(s):
    from app.knowledge import store

    pid = project(s)
    rid = run(s, pid)
    store.lock_run_and_cases(s, rid, [])
    old_case = store.ensure_run_case(s, rid)
    art = store.get_or_create_artifact(s, kind="delivered_final", locator_kind="preserved_copy",
                                       locator="finals/ci", sha256=sha("ci" + rid), project_id=pid)
    q1 = store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch", publication_id="stitch:ci",
                                     file_identity="test-live")
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["appended"]
    s.commit()
    store.invalidate(s, target_kind="case", target_id=old_case.id, reason="bad import snapshot")
    q2 = store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch", publication_id="stitch:ci",
                                     file_identity="test-live")
    assert q2.id != q1.id
    assert store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch",
                                       publication_id="stitch:ci", file_identity="test-live").id == q2.id
    s.commit()
    assert [d.outcome for d in store.drain_run_captures(s, rid)] == ["appended"]
    new_case = store.current_case(s, store.run_case_key(rid))
    assert new_case.id != old_case.id and store.latest_version(s, new_case.id).artifact_id == art.id


def test_withdrawing_a_case_version_withdraws_its_judgements(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    r = store.record_auto_assessment(s, case_id=case.id, idempotency_key="wv-" + case.id, case_version_no=1,
                                     check_kind="duration_probe", claim="full_length")
    v1 = store.latest_version(s, case.id)
    assert any(e.id == r.ids[0] for e in store.effective_events(s, case.id))
    batch = store.start_import_batch(s, source="t", code_revision="t", dry_run=False)
    store.add_import_member(s, batch.id, "case_version", v1.id)
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="bad version")
    assert not any(e.id == r.ids[0] for e in store.effective_events(s, case.id))


def test_publication_coverage_follows_the_current_case(s):
    from app.knowledge import store

    pid = project(s)
    rid = run(s, pid)
    store.lock_run_and_cases(s, rid, [])
    old_case = store.ensure_run_case(s, rid)
    art = store.get_or_create_artifact(s, kind="delivered_final", locator_kind="preserved_copy",
                                       locator="finals/cov", sha256=sha("cov" + rid), project_id=pid)
    assert not store.publication_covered(s, rid, "stitch:cov")
    store.enqueue_final_capture(s, run_id=rid, artifact_id=art.id, reason="restitch", publication_id="stitch:cov",
                                file_identity="test-live")
    assert store.publication_covered(s, rid, "stitch:cov")  # pending item covers
    s.commit()
    store.drain_run_captures(s, rid)
    s.commit()
    assert store.publication_covered(s, rid, "stitch:cov")
    store.invalidate(s, target_kind="case", target_id=old_case.id, reason="bad")
    assert not store.publication_covered(s, rid, "stitch:cov")  # pending again for the replacement case


def test_rule_evidence_refuses_an_event_of_a_withdrawn_version(s):
    from app.knowledge import store

    _, rid, case = run_case_with_version(s)
    r = store.submit_feedback(s, scope_all(), case_id=case.id, idempotency_key="rw-" + case.id,
                              expected_version_no=1, verdict="rejected")
    batch = store.start_import_batch(s, source="t", code_revision="t", dry_run=False)
    store.add_import_member(s, batch.id, "case_version", store.latest_version(s, case.id).id)
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="bad")
    with pytest.raises(store.KnowledgeInvalid, match="effective"):
        _rule(s, "rw-" + uuid.uuid4().hex[:6], "cites withdrawn",
              evidence=[{"case_id": case.id, "event_id": r.ids[0], "stance": "supports"}])


def test_an_invalidated_batch_accepts_no_new_members(s):
    from app.knowledge import store

    batch = store.start_import_batch(s, source="t", code_revision="t", dry_run=False)
    store.add_import_member(s, batch.id, "case", str(uuid.uuid4()))
    store.invalidate(s, target_kind="import_batch", target_id=batch.id, reason="bad")
    with pytest.raises(store.KnowledgeConflict):
        store.add_import_member(s, batch.id, "case", str(uuid.uuid4()))
