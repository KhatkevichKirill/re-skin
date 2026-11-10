"""Opt-in PostgreSQL races in the explicitly isolated AICP-261 test DB.

Run with ASSISTANT_TEST_POSTGRES_URL pointing to the dummy test instance at
127.0.0.1:55439/assistant_test. Other targets are refused. Each test owns a
random schema; there is no application-engine, production config, provider or
Redis access. Parent owns the disposable Docker instance's lifecycle.
"""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app import assistant_store
from app.assistant_contracts import EditPlan, TrimOperation
from app.assistant_jobs import (
    AssistantJob, AssistantJobOutbox, AssistantPaidCallIntent, BlindRetryProhibited,
    JobConflict, LostLease, accept_job, advance_phase, begin_paid_submission,
    claim_job, finalize_cancel, pending_outbox, prepare_paid_call, publish_job,
    request_cancel, renew_lease,
)
from app.assistant_models import (
    AssistantArtifact, AssistantConversation, AssistantSelection,
    AssistantVersion, AssistantVersionArtifact,
)
from app.assistant_store import ArtifactInput, AssistantArtifactStorage
from app.db import Base

_OPT_IN = "ASSISTANT_TEST_POSTGRES_URL"
pytestmark = pytest.mark.skipif(not os.environ.get(_OPT_IN), reason=f"opt in with {_OPT_IN}; isolated test DB only")
NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)


def _url():
    url = make_url(os.environ[_OPT_IN])
    if (url.drivername, url.host, url.port, url.database, url.username, url.password, dict(url.query)) != (
        "postgresql+psycopg2", "127.0.0.1", 55439, "assistant_test", "assistant_test", "isolated-test-only", {},
    ):
        pytest.fail("refusing non-isolated PostgreSQL target; expected the authorized local dummy database")
    return url


@pytest.fixture
def pg(tmp_path):
    schema = "aicp266_jobs_" + uuid.uuid4().hex
    options = "-c statement_timeout=8000 -c lock_timeout=4000"
    admin = create_engine(_url(), poolclass=NullPool, connect_args={"connect_timeout": 3, "options": options})
    with admin.begin() as connection:
        assert connection.execute(text("SELECT current_database(), current_user")).one() == ("assistant_test", "assistant_test")
        assert int(connection.scalar(text("SHOW server_version_num"))) // 10000 == 16
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(_url(), poolclass=NullPool, connect_args={
        "connect_timeout": 3, "options": options + f" -c search_path={schema}",
    })
    factory = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        # Only the independent assistant graph, not legacy production tables.
        tables = [table for table in Base.metadata.sorted_tables if table.name.startswith("assistant_")]
        Base.metadata.create_all(engine, tables=tables)
        data = tmp_path / "data"
        data.mkdir()
        original, full, preview = (data / name for name in ("original.mp4", "full.mp4", "preview.mp4"))
        original.write_bytes(b"trusted PostgreSQL fixture original")
        full.write_bytes(b"trusted PostgreSQL fixture output")
        preview.write_bytes(b"trusted PostgreSQL fixture preview")
        storage = AssistantArtifactStorage(data)
        pinned = storage.pin(original)
        run_id, base_id, conversation_id, artifact_id = (str(uuid.uuid4()) for _ in range(4))
        with factory.begin() as session:
            session.add(AssistantConversation(
                id=conversation_id, run_id=run_id, project_id=str(uuid.uuid4()),
                next_version_sequence=2, next_message_sequence=1,
            ))
            session.flush()
            session.add(AssistantVersion(
                id=base_id, conversation_id=conversation_id, sequence=1, selection_epoch=0,
                idempotency_key="test-base", request_sha256="a"*64, actor="operator",
                recipe={"kind": "isolated_pg_fixture"}, status="succeeded",
            ))
            session.flush()
            session.add(AssistantArtifact(
                id=artifact_id, owner_version_id=base_id, storage_key=pinned.storage_key,
                sha256=pinned.sha256, size_bytes=pinned.size_bytes, media_kind="video",
                metadata_json={"duration_sec": 3},
            ))
            session.flush()
            session.add(AssistantSelection(conversation_id=conversation_id, selected_version_id=base_id,
                                           latest_version_id=base_id, epoch=0))
            for role in ("full", "preview"):
                session.add(AssistantVersionArtifact(version_id=base_id, artifact_id=artifact_id, role=role, slot=0))
        yield SimpleNamespace(
            factory=factory, engine=engine, storage=storage, schema=schema,
            run_id=run_id, base_id=base_id, conversation_id=conversation_id,
            full=ArtifactInput(full, metadata={"duration_sec": 2}),
            preview=ArtifactInput(preview, metadata={"duration_sec": 2}),
            original=original,
        )
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _plan(pg, end=2):
    return EditPlan(run_id=pg.run_id, base_version_id=pg.base_id,
                    operations=(TrimOperation(start_sec=0, end_sec=end),))


def _accept(session, pg, *, key="request", end=2):
    return accept_job(session, _plan(pg, end), request_scope="private-operator", idempotency_key=key,
                      actor="operator", now=NOW)


def _seed_job(pg, *, publishing=False, lease_sec=60):
    with pg.factory.begin() as session:
        job = _accept(session, pg)
        job_id, version_id = job.id, job.pending_version_id
        lease = None
        if publishing:
            lease = claim_job(session, job.id, owner="worker-1", now=NOW, lease_sec=lease_sec)
            for phase in ("render", "inspect", "publish"):
                lease = advance_phase(session, lease, phase=phase, now=NOW, lease_sec=lease_sec)
        return job_id, version_id, lease


def _count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def _wait(event: Event):
    assert event.wait(timeout=5), "concurrent test synchronization timed out"


def _assert_blocked_on_database_lock(pg, pid):
    """Prove actual lock contention instead of relying on a scheduling sleep."""
    deadline = time.monotonic() + 3
    with pg.engine.connect() as connection:
        while time.monotonic() < deadline:
            state = connection.scalar(text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"), {"pid": pid})
            if state == "Lock":
                return
            # pg_stat_activity uses a transaction snapshot unless cleared.
            connection.rollback()
            Event().wait(0.01)
    pytest.fail("second independent backend never reached the expected PostgreSQL row-lock wait")


def _final(pg):
    with pg.factory() as session:
        job = session.scalar(select(AssistantJob))
        version = session.get(AssistantVersion, job.pending_version_id)
        return job.status, version.status, [e.kind for e in pending_outbox(session)], _count(session, AssistantArtifact)


def test_concurrent_duplicate_acceptance_has_one_job_version_and_outbox(pg, monkeypatch):
    entering_reservation = Barrier(2, timeout=5)
    original = assistant_store.reserve_version

    def reserve(*args, **kwargs):
        # Both acceptors have already missed their first idempotency read.
        entering_reservation.wait()
        return original(*args, **kwargs)

    monkeypatch.setattr(assistant_store, "reserve_version", reserve)

    def accept():
        with pg.factory.begin() as session:
            pid = session.scalar(text("SELECT pg_backend_pid()"))
            job = _accept(session, pg)
            return pid, job.id, job.pending_version_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(accept) for _ in range(2)]
        results = [future.result(timeout=12) for future in futures]
    assert results[0][0] != results[1][0]
    assert results[0][1:] == results[1][1:]
    with pg.factory() as session:
        assert _count(session, AssistantJob) == 1
        assert _count(session, AssistantVersion) == 2
        assert _count(session, AssistantJobOutbox) == 1
        assert session.get(AssistantConversation, pg.conversation_id).next_version_sequence == 3


def test_concurrent_distinct_requests_remain_pinned_and_allocate_distinct_sequences(pg):
    barrier = Barrier(2, timeout=5)

    def accept(index):
        with pg.factory.begin() as session:
            pid = session.scalar(text("SELECT pg_backend_pid()"))
            barrier.wait()
            job = _accept(session, pg, key=f"request-{index}")
            return pid, job.id, job.pinned_base_version_id, job.pending_version_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(accept, index) for index in range(2)]
        results = [future.result(timeout=12) for future in futures]
    assert results[0][0] != results[1][0]
    assert results[0][1] != results[1][1]
    assert {r[2] for r in results} == {pg.base_id}
    with pg.factory() as session:
        assert list(session.scalars(select(AssistantVersion.sequence).order_by(AssistantVersion.sequence))) == [1, 2, 3]
        assert _count(session, AssistantJobOutbox) == 2


def test_duplicate_concurrent_claims_grant_exactly_one_fence(pg):
    job_id, _, _ = _seed_job(pg)
    barrier = Barrier(2, timeout=5)

    def claim(index):
        with pg.factory.begin() as session:
            pid = session.scalar(text("SELECT pg_backend_pid()"))
            barrier.wait()
            return pid, claim_job(session, job_id, owner=f"worker-{index}", now=NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim, index) for index in range(2)]
        results = [future.result(timeout=12) for future in futures]
    assert results[0][0] != results[1][0]
    leases = [lease for _, lease in results if lease is not None]
    assert len(leases) == 1
    with pg.factory() as session:
        job = session.get(AssistantJob, job_id)
        assert job.lease_token == leases[0].token
        assert job.claim_count == job.lease_generation == 1


def test_cancellation_committing_first_fences_a_waiting_publisher(pg):
    job_id, _, lease = _seed_job(pg, publishing=True)
    cancellation_written, release_cancellation, publication_started = Event(), Event(), Event()
    pids = {}

    def cancel():
        with pg.factory.begin() as session:
            pids["cancel"] = session.scalar(text("SELECT pg_backend_pid()"))
            request_cancel(session, job_id, now=NOW)
            cancellation_written.set()
            _wait(release_cancellation)

    def publish():
        _wait(cancellation_written)
        with pg.factory.begin() as session:
            pids["publish"] = session.scalar(text("SELECT pg_backend_pid()"))
            publication_started.set()
            with pytest.raises(LostLease):
                publish_job(session, lease, full=pg.full, preview=pg.preview, storage=pg.storage, now=NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        cancel_future, publish_future = pool.submit(cancel), pool.submit(publish)
        try:
            _wait(publication_started)
            assert pids["cancel"] != pids["publish"]
            _assert_blocked_on_database_lock(pg, pids["publish"])
        finally:
            release_cancellation.set()
        cancel_future.result(timeout=12)
        publish_future.result(timeout=12)
    assert _final(pg)[:2] == ("cancel_requested", "pending")
    with pg.factory.begin() as session:
        finalize_cancel(session, job_id, lease=lease, now=NOW)
    state, version, events, artifacts = _final(pg)
    assert (state, version, artifacts) == ("cancelled", "cancelled", 1)
    assert set(events) == {"job.accepted", "job.cancel_requested", "job.cancelled"}


def test_publication_committing_first_is_not_overwritten_by_waiting_cancel(pg):
    job_id, _, lease = _seed_job(pg, publishing=True)
    publisher_locked, release_publisher, cancellation_started = Event(), Event(), Event()
    pids = {}

    class HoldingStorage:
        def pin(self, path):
            publisher_locked.set()
            _wait(release_publisher)
            return pg.storage.pin(path)

    def publish():
        with pg.factory.begin() as session:
            pids["publish"] = session.scalar(text("SELECT pg_backend_pid()"))
            publish_job(session, lease, full=pg.full, preview=pg.preview, storage=HoldingStorage(), now=NOW)

    def cancel():
        _wait(publisher_locked)
        with pg.factory.begin() as session:
            pids["cancel"] = session.scalar(text("SELECT pg_backend_pid()"))
            cancellation_started.set()
            return request_cancel(session, job_id, now=NOW).status

    with ThreadPoolExecutor(max_workers=2) as pool:
        publish_future, cancel_future = pool.submit(publish), pool.submit(cancel)
        try:
            _wait(cancellation_started)
            assert pids["publish"] != pids["cancel"]
            _assert_blocked_on_database_lock(pg, pids["cancel"])
        finally:
            release_publisher.set()
        publish_future.result(timeout=12)
        assert cancel_future.result(timeout=12) == "succeeded"
    state, version, events, artifacts = _final(pg)
    assert (state, version, artifacts) == ("succeeded", "succeeded", 3)
    assert set(events) == {"job.accepted", "job.succeeded"}


def test_publication_rollback_releases_waiting_cancellation_without_success_event(pg):
    job_id, _, lease = _seed_job(pg, publishing=True)
    published_uncommitted, release_rollback, cancellation_started = Event(), Event(), Event()
    pids = {}

    def publish_then_rollback():
        with pg.factory() as session:
            pids["publish"] = session.scalar(text("SELECT pg_backend_pid()"))
            publish_job(session, lease, full=pg.full, preview=pg.preview, storage=pg.storage, now=NOW)
            published_uncommitted.set()
            _wait(release_rollback)
            session.rollback()

    def cancel():
        _wait(published_uncommitted)
        with pg.factory.begin() as session:
            pids["cancel"] = session.scalar(text("SELECT pg_backend_pid()"))
            cancellation_started.set()
            return request_cancel(session, job_id, now=NOW).status

    with ThreadPoolExecutor(max_workers=2) as pool:
        publisher, canceller = pool.submit(publish_then_rollback), pool.submit(cancel)
        try:
            _wait(cancellation_started)
            _assert_blocked_on_database_lock(pg, pids["cancel"])
            # A third connection sees only the previously committed graph.
            assert _final(pg) == ("running", "pending", ["job.accepted"], 1)
        finally:
            release_rollback.set()
        publisher.result(timeout=12)
        assert canceller.result(timeout=12) == "cancel_requested"
    with pg.factory.begin() as session:
        finalize_cancel(session, job_id, lease=lease, now=NOW)
    state, version, events, artifacts = _final(pg)
    assert (state, version, artifacts) == ("cancelled", "cancelled", 1)
    assert "job.succeeded" not in events


def test_expired_worker_with_stale_loaded_state_is_fenced_after_reclaim(pg):
    job_id, _, old_lease = _seed_job(pg, publishing=True, lease_sec=1)
    old_loaded, reclaimed, old_rejected = Event(), Event(), Event()
    pids = {}
    later = NOW + timedelta(seconds=2)

    def old_worker():
        with pg.factory.begin() as session:
            pids["old"] = session.scalar(text("SELECT pg_backend_pid()"))
            stale = session.get(AssistantJob, job_id)
            assert stale.lease_token == old_lease.token
            old_loaded.set()
            _wait(reclaimed)
            with pytest.raises(LostLease):
                publish_job(session, old_lease, full=pg.full, preview=pg.preview, storage=pg.storage, now=later)
            old_rejected.set()

    def new_worker():
        _wait(old_loaded)
        with pg.factory.begin() as session:
            pids["new"] = session.scalar(text("SELECT pg_backend_pid()"))
            lease = claim_job(session, job_id, owner="worker-2", now=later)
            assert lease and lease.token != old_lease.token and lease.generation > old_lease.generation
        reclaimed.set()
        _wait(old_rejected)
        with pg.factory.begin() as session:
            publish_job(session, lease, full=pg.full, preview=pg.preview, storage=pg.storage, now=later)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(old_worker), pool.submit(new_worker)]
        for future in futures:
            future.result(timeout=12)
    assert pids["old"] != pids["new"]
    assert _final(pg) == ("succeeded", "succeeded", ["job.accepted", "job.succeeded"], 3)


def test_acceptance_rollback_hides_job_version_outbox_and_allows_clean_retry(pg):
    accepted_uncommitted, release_rollback = Event(), Event()
    pids = {}

    def accept_then_rollback():
        with pg.factory() as session:
            pids["writer"] = session.scalar(text("SELECT pg_backend_pid()"))
            _accept(session, pg)
            accepted_uncommitted.set()
            _wait(release_rollback)
            session.rollback()

    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(accept_then_rollback)
        try:
            _wait(accepted_uncommitted)
            with pg.factory() as session:
                pids["reader"] = session.scalar(text("SELECT pg_backend_pid()"))
                assert _count(session, AssistantJob) == _count(session, AssistantJobOutbox) == 0
                assert _count(session, AssistantVersion) == 1
                assert session.get(AssistantConversation, pg.conversation_id).next_version_sequence == 2
        finally:
            release_rollback.set()
        writer.result(timeout=12)
    assert pids["writer"] != pids["reader"]
    with pg.factory.begin() as session:
        _accept(session, pg)
        assert _count(session, AssistantJob) == _count(session, AssistantJobOutbox) == 1
        assert _count(session, AssistantVersion) == 2
        assert session.get(AssistantConversation, pg.conversation_id).next_version_sequence == 3


def test_concurrent_submission_permits_cannot_duplicate_paid_call_intent(pg):
    job_id, _, lease = _seed_job(pg, publishing=True)
    with pg.factory.begin() as session:
        intent = prepare_paid_call(session, lease, logical_key="provider-render", provider="test-provider",
                                   request_sha256="b"*64, now=NOW)
        intent_id = intent.id
    barrier = Barrier(2, timeout=5)

    def submit():
        with pg.factory.begin() as session:
            pid = session.scalar(text("SELECT pg_backend_pid()"))
            barrier.wait()
            try:
                return pid, begin_paid_submission(session, lease, intent_id, now=NOW)
            except BlindRetryProhibited:
                return pid, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        results = [future.result(timeout=12) for future in futures]
    assert results[0][0] != results[1][0]
    assert sum(permit is not None for _, permit in results) == 1
    with pg.factory() as session:
        assert _count(session, AssistantPaidCallIntent) == 1
        assert session.get(AssistantPaidCallIntent, intent_id).status == "submitting"


@pytest.mark.parametrize("operation,locked_table", [
    ("renew", "job"), ("submit", "job"), ("submit", "intent"), ("cancel_ack", "job"),
])
def test_delayed_database_lock_cannot_revive_expired_worker(pg, operation, locked_table, record_property):
    job_id, version_id, lease = _seed_job(pg, publishing=True, lease_sec=1)
    # A granted deadline includes measured setup/SQL time. Reusing historical
    # NOW here gives the worker MORE than one second on a loaded test host.
    # Anchor this operation to the actual grant, making the tested remaining
    # interval exactly one second regardless of fixture construction latency.
    operation_now = lease.expires_at - timedelta(seconds=1)
    assert lease.expires_at - operation_now == timedelta(seconds=1)
    record_property("setup_extension_sec", (operation_now - NOW).total_seconds())
    intent_id = None
    if operation == "submit":
        with pg.factory.begin() as session:
            intent_id = prepare_paid_call(
                session, lease, logical_key="delayed-submit", provider="test-provider",
                request_sha256="c"*64, now=operation_now,
            ).id
    if operation == "cancel_ack":
        with pg.factory.begin() as session:
            request_cancel(session, job_id, now=operation_now)
    entered = Event()
    pids = {}

    def worker():
        with pg.factory.begin() as session:
            pids["worker"] = session.scalar(text("SELECT pg_backend_pid()"))
            entered.set()
            started = time.monotonic()
            with pytest.raises(LostLease, match="expired while waiting"):
                if operation == "renew":
                    renew_lease(session, lease, now=operation_now)
                elif operation == "submit":
                    begin_paid_submission(session, lease, intent_id, now=operation_now)
                else:
                    finalize_cancel(session, job_id, lease=lease, now=operation_now)
            waited = time.monotonic() - started
            record_property("operation_elapsed_sec", waited)
            assert waited > (lease.expires_at-operation_now).total_seconds()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pg.factory.begin() as locker:
            pids["locker"] = locker.scalar(text("SELECT pg_backend_pid()"))
            model, object_id = (AssistantJob, job_id) if locked_table == "job" else (AssistantPaidCallIntent, intent_id)
            locker.execute(select(model.id).where(model.id == object_id).with_for_update())
            future = pool.submit(worker)
            _wait(entered)
            assert pids["locker"] != pids["worker"]
            _assert_blocked_on_database_lock(pg, pids["worker"])
            # The one-second lease expires during a real PostgreSQL lock wait.
            Event().wait(1.25)
        future.result(timeout=12)
    with pg.factory() as session:
        job = session.get(AssistantJob, job_id)
        assert job.status == ("cancel_requested" if operation == "cancel_ack" else "running")
        assert job.lease_expires_at == lease.expires_at
        assert session.get(AssistantVersion, version_id).status == "pending"
        assert not any(e.event_key == "terminal" for e in pending_outbox(session))
        if intent_id:
            intent = session.get(AssistantPaidCallIntent, intent_id)
            assert intent.status == "prepared" and intent.submission_token is None


def test_queued_claim_gets_full_lease_after_delayed_database_lock(pg):
    job_id, _, _ = _seed_job(pg)
    entered = Event()
    pids = {}

    def worker():
        with pg.factory.begin() as session:
            pids["worker"] = session.scalar(text("SELECT pg_backend_pid()"))
            entered.set()
            return claim_job(session, job_id, owner="delayed-worker", now=NOW, lease_sec=1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pg.factory.begin() as locker:
            locker.execute(select(AssistantJob.id).where(AssistantJob.id == job_id).with_for_update())
            future = pool.submit(worker)
            _wait(entered)
            _assert_blocked_on_database_lock(pg, pids["worker"])
            Event().wait(1.25)
        lease = future.result(timeout=12)
    assert lease is not None
    assert lease.expires_at >= NOW + timedelta(seconds=2.2)
