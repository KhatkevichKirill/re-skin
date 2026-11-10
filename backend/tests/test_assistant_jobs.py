"""SQL-only lifecycle checks against a private temporary SQLite database.

These are logical/rollback checks, not PostgreSQL concurrent-worker proof.
No provider, Redis, production database, live migration or network is used.
"""

from __future__ import annotations

import os
import importlib.util
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.orm import sessionmaker

# Avoid opening/configuring the production engine when this file runs alone.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.assistant_contracts import EditPlan, TrimOperation
from app.assistant_jobs import (
    AssistantJob, AssistantJobOutbox, AssistantPaidCallIntent, BlindRetryProhibited,
    JobConflict, LostLease, accept_job, acknowledge_outbox, advance_phase,
    begin_paid_submission, claim_job, complete_paid_call, fail_job, finalize_cancel,
    pending_outbox, prepare_paid_call, publish_job, record_submission, renew_lease,
    request_cancel,
)
from app.assistant_models import AssistantArtifact, AssistantVersion, AssistantVersionArtifact
from app.assistant_store import ArtifactInput, AssistantArtifactStorage, import_run_version
from app.db import Base
from app.models import Run, VideoProject
from app.state_machine import RunStatus

NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)


@pytest.fixture
def context(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    data = tmp_path / "data"
    data.mkdir()
    original = data / "original.mp4"
    original.write_bytes(b"trusted original fixture bytes")
    full = data / "full.mp4"
    full.write_bytes(b"trusted rendered full fixture bytes")
    preview = data / "preview.mp4"
    preview.write_bytes(b"trusted rendered preview fixture bytes")
    storage = AssistantArtifactStorage(data)
    with factory() as session:
        project = VideoProject(source_type="upload", source_ref="local fixture")
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status=RunStatus.done, result_local_path=str(original), result_duration_sec=3)
        session.add(run)
        session.flush()
        base = import_run_version(session, run, actor="operator", storage=storage)
        run_id, base_id = run.id, base.id
        session.commit()
    yield SimpleNamespace(
        factory=factory, storage=storage, run_id=run_id, base_id=base_id,
        original=original, full=ArtifactInput(full, metadata={"duration_sec": 2}),
        preview=ArtifactInput(preview, metadata={"duration_sec": 2}),
    )
    engine.dispose()


def _plan(context, end=2):
    return EditPlan(run_id=context.run_id, base_version_id=context.base_id,
                    operations=(TrimOperation(start_sec=0, end_sec=end),))


def _accept(session, context, *, key="request-1", **kwargs):
    return accept_job(session, _plan(context), request_scope="private-operator", idempotency_key=key,
                      actor="operator", now=NOW, **kwargs)


def _claim(session, job_id, **kwargs):
    return claim_job(session, job_id, owner="worker-1", now=NOW, **kwargs)


def _publishing(session, job_id):
    lease = _claim(session, job_id)
    for phase in ("render", "inspect", "publish"):
        lease = advance_phase(session, lease, phase=phase, now=NOW)
    return lease


def _count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def test_acceptance_is_idempotent_and_pins_one_base_version(context):
    with context.factory() as session:
        first = _accept(session, context)
        same = _accept(session, context)
        assert same.id == first.id
        assert first.pinned_base_version_id == context.base_id
        assert first.accepted_plan == _plan(context).model_dump(mode="json")
        assert _count(session, AssistantJob) == 1
        assert _count(session, AssistantVersion) == 2
        assert len(pending_outbox(session)) == 1
        assert pending_outbox(session)[0].payload["version_id"] == first.pending_version_id
        session.commit()
    with context.factory() as session:
        assert _accept(session, context).id == first.id


def test_changed_request_cannot_reuse_scoped_key(context):
    with context.factory() as session:
        _accept(session, context)
        with pytest.raises(JobConflict, match="another request"):
            accept_job(session, _plan(context, end=1), request_scope="private-operator",
                       idempotency_key="request-1", actor="operator", now=NOW)
        with pytest.raises(JobConflict, match="another request"):
            accept_job(session, _plan(context), request_scope="private-operator",
                       idempotency_key="request-1", actor="another-operator", now=NOW)


def test_caller_rollback_removes_job_version_and_outbox_even_first_savepoint(context):
    with context.factory() as session:
        _accept(session, context)
        session.rollback()
    with context.factory() as session:
        assert _count(session, AssistantJob) == 0
        assert _count(session, AssistantVersion) == 1
        assert _count(session, AssistantJobOutbox) == 0


def test_claim_deduplicates_work_and_rotates_fence_for_next_phase(context):
    with context.factory() as session:
        job = _accept(session, context)
        lease = _claim(session, job.id)
        assert lease and _claim(session, job.id) is None
        next_lease = advance_phase(session, lease, phase="render", now=NOW)
        assert next_lease.generation == lease.generation + 1
        assert next_lease.token != lease.token
        with pytest.raises(LostLease):
            fail_job(session, lease, error_code="stale", now=NOW)
        with pytest.raises(JobConflict, match="phase must advance"):
            advance_phase(session, next_lease, phase="publish", now=NOW)
        renewed = renew_lease(session, next_lease, now=NOW + timedelta(seconds=20))
        assert timedelta(0) <= renewed.expires_at - (NOW + timedelta(seconds=80)) < timedelta(seconds=1)


def test_expired_lease_cannot_renew_publish_or_complete_after_reclaim(context):
    with context.factory() as session:
        job = _accept(session, context)
        old = _publishing(session, job.id)
        later = old.expires_at + timedelta(milliseconds=1)
        with pytest.raises(LostLease):
            renew_lease(session, old, now=later)
        new = claim_job(session, job.id, owner="worker-2", now=later)
        assert new.generation > old.generation
        assert new.phase == "publish"
        with pytest.raises(LostLease):
            publish_job(session, old, full=context.full, preview=context.preview, storage=context.storage, now=later)
        publish_job(session, new, full=context.full, preview=context.preview, storage=context.storage, now=later)
        assert session.get(AssistantJob, job.id).status == "succeeded"


def test_claim_limit_fails_reserved_version_and_keeps_terminal_job_unclaimable(context):
    with context.factory() as session:
        job = _accept(session, context, max_claims=1)
        assert _claim(session, job.id)
        assert claim_job(session, job.id, owner="worker-2", now=NOW + timedelta(seconds=61)) is None
        session.refresh(job)
        assert job.status == "failed"
        assert job.error_code == "claim_limit_exhausted"
        assert session.get(AssistantVersion, job.pending_version_id).status == "failed"
        assert _claim(session, job.id) is None
        assert len(pending_outbox(session)) == 2


def test_queued_cancel_is_atomic_terminal_and_idempotent(context):
    with context.factory() as session:
        job = _accept(session, context)
        request_cancel(session, job.id, now=NOW)
        request_cancel(session, job.id, now=NOW)
        assert session.get(AssistantVersion, job.pending_version_id).status == "cancelled"
        assert _claim(session, job.id) is None
        assert len(pending_outbox(session)) == 2
        session.commit()


def test_running_cancel_blocks_publication_and_requires_current_ack_or_expiry(context):
    with context.factory() as session:
        job = _accept(session, context)
        lease = _publishing(session, job.id)
        request_cancel(session, job.id, now=NOW)
        with pytest.raises(LostLease):
            publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
        with pytest.raises(LostLease):
            finalize_cancel(session, job.id, now=NOW)
        final = finalize_cancel(session, job.id, lease=lease, now=NOW)
        assert final.status == "cancelled"
        assert session.get(AssistantVersion, job.pending_version_id).status == "cancelled"
        assert _count(session, AssistantArtifact) == 1


def test_reaper_can_finalize_abandoned_cancellation_after_expiry(context):
    with context.factory() as session:
        job = _accept(session, context)
        _claim(session, job.id)
        request_cancel(session, job.id, now=NOW)
        final = finalize_cancel(session, job.id, now=NOW + timedelta(seconds=61))
        assert final.status == "cancelled"


def test_stale_loaded_job_does_not_bypass_cancellation_in_another_session(context):
    with context.factory() as session:
        job = _accept(session, context)
        lease = _publishing(session, job.id)
        session.commit()
        assert job.status == "running"
        with context.factory() as other:
            request_cancel(other, job.id, now=NOW)
            other.commit()
        with pytest.raises(LostLease):
            publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)


def test_publication_requires_publish_phase_and_preserves_original(context):
    original_bytes = context.original.read_bytes()
    with context.factory() as session:
        job = _accept(session, context)
        early = _claim(session, job.id)
        with pytest.raises(LostLease):
            publish_job(session, early, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
        lease = early
        for phase in ("render", "inspect", "publish"):
            lease = advance_phase(session, lease, phase=phase, now=NOW)
        result = publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
        assert result.status == "succeeded" and result.lease_token is None
        assert session.get(AssistantVersion, result.pending_version_id).status == "succeeded"
        assert session.get(AssistantVersion, context.base_id).status == "succeeded"
        assert _count(session, AssistantArtifact) == 3
        assert [e.kind for e in pending_outbox(session)] == ["job.accepted", "job.succeeded"]
        assert _claim(session, job.id) is None
        with pytest.raises(LostLease):
            publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
        session.commit()
    assert context.original.read_bytes() == original_bytes


def test_failed_preview_rolls_back_full_publication_and_job_state(context):
    empty = context.original.parent / "empty.mp4"
    empty.write_bytes(b"")
    with context.factory() as session:
        job = _accept(session, context)
        lease = _publishing(session, job.id)
        with pytest.raises(ValueError):
            publish_job(session, lease, full=context.full, preview=ArtifactInput(empty), storage=context.storage, now=NOW)
        session.refresh(job)
        assert job.status == "running"
        assert session.get(AssistantVersion, job.pending_version_id).status == "pending"
        assert _count(session, AssistantArtifact) == 1
        assert _count(session, AssistantVersionArtifact) == 3  # two V1 roles + pinned job input
        assert len(pending_outbox(session)) == 1
        session.commit()


def test_expiry_during_artifact_copy_rolls_back_publication(context, monkeypatch):
    import app.assistant_jobs as jobs

    with context.factory() as session:
        job = _accept(session, context)
        lease = _publishing(session, job.id)
        elapsed = [0.0]
        monkeypatch.setattr(jobs, "_monotonic", lambda: elapsed[0])
        original_pin = context.storage.pin

        def slow_pin(path):
            result = original_pin(path)
            elapsed[0] = 61
            return result

        monkeypatch.setattr(context.storage, "pin", slow_pin)
        with pytest.raises(LostLease):
            publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
        session.refresh(job)
        assert job.status == "running"
        assert session.get(AssistantVersion, job.pending_version_id).status == "pending"
        assert _count(session, AssistantArtifact) == 1


def test_caller_rollback_reverts_terminal_job_version_and_outbox_together(context):
    with context.factory() as session:
        job = _accept(session, context)
        job_id = job.id
        lease = _publishing(session, job.id)
        session.commit()
        publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
        session.rollback()
    with context.factory() as session:
        restored = session.get(AssistantJob, job_id)
        assert restored.status == "running"
        assert session.get(AssistantVersion, restored.pending_version_id).status == "pending"
        assert _count(session, AssistantArtifact) == 1
        assert len(pending_outbox(session)) == 1


def test_failure_rolls_back_if_version_transition_fails(context, monkeypatch):
    import app.assistant_store as store

    with context.factory() as session:
        job = _accept(session, context)
        lease = _claim(session, job.id)
        original = store.mark_version_terminal

        def fail_after_mutation(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected failure")

        monkeypatch.setattr(store, "mark_version_terminal", fail_after_mutation)
        with pytest.raises(RuntimeError, match="injected"):
            fail_job(session, lease, error_code="render_failed", now=NOW)
        session.refresh(job)
        assert job.status == "running"
        assert session.get(AssistantVersion, job.pending_version_id).status == "pending"
        assert len(pending_outbox(session)) == 1


def test_outbox_survives_reopen_and_ack_is_idempotent(context):
    with context.factory() as session:
        _accept(session, context)
        session.commit()
    with context.factory() as session:
        events = pending_outbox(session)
        assert len(events) == 1
        acknowledge_outbox(session, events[0].id, now=NOW)
        acknowledge_outbox(session, events[0].id, now=NOW)
        assert pending_outbox(session) == []
        session.commit()


def _intent(session, lease):
    return prepare_paid_call(session, lease, logical_key="render-1", provider="provider-test",
                             request_sha256="a"*64, now=NOW)


def test_paid_intent_is_idempotent_but_different_payload_conflicts(context):
    with context.factory() as session:
        lease = _claim(session, _accept(session, context).id)
        intent = _intent(session, lease)
        assert _intent(session, lease).id == intent.id
        with pytest.raises(JobConflict, match="another request"):
            prepare_paid_call(session, lease, logical_key="render-1", provider="provider-test",
                              request_sha256="b"*64, now=NOW)
        assert _count(session, AssistantPaidCallIntent) == 1


def test_submitting_and_unknown_intents_cannot_blind_retry(context):
    with context.factory() as session:
        lease = _claim(session, _accept(session, context).id)
        intent = _intent(session, lease)
        permit = begin_paid_submission(session, lease, intent.id, now=NOW)
        session.commit()  # required durable boundary BEFORE any hypothetical call
        with pytest.raises(BlindRetryProhibited):
            begin_paid_submission(session, lease, intent.id, now=NOW)
        record_submission(session, permit, provider_task_id=None, now=NOW)
        with pytest.raises(JobConflict):
            prepare_paid_call(session, lease, logical_key="changed-retry-key", provider="provider-test",
                              request_sha256="a"*64, now=NOW)
        with pytest.raises(BlindRetryProhibited):
            begin_paid_submission(session, lease, intent.id, now=NOW)
        session.refresh(intent)
        assert intent.status == "unknown"
        session.commit()


def test_reclaim_marks_interrupted_submission_unknown_and_fences_old_worker(context):
    with context.factory() as session:
        job = _accept(session, context)
        old = _claim(session, job.id)
        intent = _intent(session, old)
        permit = begin_paid_submission(session, old, intent.id, now=NOW)
        later = NOW + timedelta(seconds=61)
        new = claim_job(session, job.id, owner="worker-2", now=later)
        session.refresh(intent)
        assert intent.status == "unknown"
        with pytest.raises(LostLease):
            begin_paid_submission(session, old, intent.id, now=later)
        with pytest.raises(BlindRetryProhibited):
            begin_paid_submission(session, new, intent.id, now=later)
        receipt = record_submission(session, permit, provider_task_id="existing-task", now=later)
        assert receipt.status == "submitted"


def test_late_receipt_after_cancel_is_recorded_without_resurrecting_job(context):
    with context.factory() as session:
        job = _accept(session, context)
        lease = _claim(session, job.id)
        intent = _intent(session, lease)
        permit = begin_paid_submission(session, lease, intent.id, now=NOW)
        request_cancel(session, job.id, now=NOW)
        finalize_cancel(session, job.id, lease=lease, now=NOW)
        session.refresh(intent)
        assert intent.status == "unknown"
        record_submission(session, permit, provider_task_id="late-task", now=NOW)
        assert session.get(AssistantJob, job.id).status == "cancelled"
        assert session.get(AssistantVersion, job.pending_version_id).status == "cancelled"
        assert record_submission(session, permit, provider_task_id=None, now=NOW).provider_task_id == "late-task"
        with pytest.raises(JobConflict, match="different provider receipt"):
            record_submission(session, permit, provider_task_id="other-task", now=NOW)


def test_unknown_receipts_require_reconciliation_before_completion_and_publication(context):
    with context.factory() as session:
        job = _accept(session, context)
        lease = _publishing(session, job.id)
        intent = _intent(session, lease)
        permit = begin_paid_submission(session, lease, intent.id, now=NOW)
        record_submission(session, permit, provider_task_id=None, now=NOW)
        with pytest.raises(BlindRetryProhibited):
            complete_paid_call(session, permit, succeeded=True, now=NOW)
        with pytest.raises(JobConflict, match="outstanding paid calls"):
            publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
        record_submission(session, permit, provider_task_id="task-1", now=NOW)
        complete_paid_call(session, permit, succeeded=True, now=NOW)
        assert complete_paid_call(session, permit, succeeded=True, now=NOW).status == "completed"
        with pytest.raises(BlindRetryProhibited):
            complete_paid_call(session, permit, succeeded=False, now=NOW)
        publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)


def test_forged_submission_permit_cannot_attach_receipt(context):
    with context.factory() as session:
        lease = _claim(session, _accept(session, context).id)
        permit = begin_paid_submission(session, lease, _intent(session, lease).id, now=NOW)
        with pytest.raises(JobConflict, match="unknown provider submission permit"):
            record_submission(session, replace(permit, token="wrong"), provider_task_id="task-1", now=NOW)


@pytest.mark.parametrize("kwargs", [{"max_claims": True}, {"max_claims": 0}, {"max_claims": 11}])
def test_invalid_claim_policy_rejected_before_job_creation(context, kwargs):
    with context.factory() as session:
        with pytest.raises(JobConflict):
            _accept(session, context, **kwargs)
        assert _count(session, AssistantJob) == 0


def test_naive_clock_is_rejected(context):
    with context.factory() as session:
        with pytest.raises(JobConflict, match="timezone-aware"):
            accept_job(session, _plan(context), request_scope="private-operator", idempotency_key="key",
                       actor="operator", now=NOW.replace(tzinfo=None))


def test_accepted_job_binding_cannot_be_edited_through_orm(context):
    with context.factory() as session:
        job = _accept(session, context)
        session.commit()
        job.pinned_base_version_id = job.pending_version_id
        with pytest.raises(JobConflict, match="immutable assistant job field"):
            session.flush()
        session.rollback()


def test_job_migration_matches_model_schema_in_temporary_database(tmp_path):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = Path(__file__).parents[1] / "alembic/versions/d5b0f3a8c702_assistant_jobs_outbox_intents.py"
    spec = importlib.util.spec_from_file_location("assistant_jobs_migration_test", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.down_revision == "c4a9e2f7b601"
    own_tables = {AssistantJob.__tablename__, AssistantJobOutbox.__tablename__, AssistantPaidCallIntent.__tablename__}
    engine = create_engine(f"sqlite:///{tmp_path / 'migration-only.db'}")
    # Prerequisites are local metadata fixtures; never use application engine or
    # Alembic env.py, which may load a production connection configuration.
    Base.metadata.create_all(engine, tables=[table for table in Base.metadata.sorted_tables if table.name not in own_tables])
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        actual = inspect(connection)
        for name in own_tables:
            table = Base.metadata.tables[name]
            assert {c["name"]: c["nullable"] for c in actual.get_columns(name)} == {
                c.name: c.nullable for c in table.columns
            }
            assert {c["name"] for c in actual.get_unique_constraints(name)} == {
                c.name for c in table.constraints if c.__class__.__name__ == "UniqueConstraint"
            }
            assert {c["name"] for c in actual.get_check_constraints(name)} == {
                c.name for c in table.constraints if c.__class__.__name__ == "CheckConstraint"
            }
            assert {i["name"] for i in actual.get_indexes(name)} == {i.name for i in table.indexes}
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        assert own_tables.isdisjoint(inspect(connection).get_table_names())
        assert "assistant_versions" in inspect(connection).get_table_names()
    engine.dispose()


@pytest.mark.parametrize("operation", ["renew", "phase", "prepare", "submit", "fail", "publish", "cancel_ack"])
def test_expiry_while_acquiring_job_lock_rejects_all_worker_mutations(context, monkeypatch, operation):
    import app.assistant_jobs as jobs

    with context.factory() as session:
        job = _accept(session, context)
        lease = _publishing(session, job.id) if operation == "publish" else _claim(session, job.id)
        intent_id = _intent(session, lease).id if operation == "submit" else None
        if operation == "cancel_ack":
            request_cancel(session, job.id, now=NOW)
        session.commit()
        before_status = "cancel_requested" if operation == "cancel_ack" else "running"
        original_deadline = lease.expires_at
        elapsed = [0.0]
        monkeypatch.setattr(jobs, "_monotonic", lambda: elapsed[0])
        execute = session.execute

        def delayed_lock(statement, *args, **kwargs):
            result = execute(statement, *args, **kwargs)
            if getattr(statement, "is_update", False) and statement.table.name == "assistant_jobs":
                elapsed[0] = 61  # lock obtained after the original 60s expiry
            return result

        monkeypatch.setattr(session, "execute", delayed_lock)
        with pytest.raises(LostLease, match="expired while waiting"):
            if operation == "renew":
                renew_lease(session, lease, now=NOW)
            elif operation == "phase":
                advance_phase(session, lease, phase="render", now=NOW)
            elif operation == "prepare":
                _intent(session, lease)
            elif operation == "submit":
                begin_paid_submission(session, lease, intent_id, now=NOW)
            elif operation == "fail":
                fail_job(session, lease, error_code="late_failure", now=NOW)
            elif operation == "publish":
                publish_job(session, lease, full=context.full, preview=context.preview, storage=context.storage, now=NOW)
            else:
                finalize_cancel(session, job.id, lease=lease, now=NOW)
        session.refresh(job)
        assert job.status == before_status
        assert job.lease_expires_at.replace(tzinfo=timezone.utc) == original_deadline
        assert session.get(AssistantVersion, job.pending_version_id).status == "pending"
        if intent_id:
            assert session.get(AssistantPaidCallIntent, intent_id).status == "prepared"
        assert _count(session, AssistantArtifact) == 1
        assert not any(e.event_key == "terminal" for e in pending_outbox(session))


def test_expiry_on_paid_intent_lock_rolls_back_submission_permit(context, monkeypatch):
    import app.assistant_jobs as jobs

    with context.factory() as session:
        job = _accept(session, context)
        lease = _claim(session, job.id)
        intent = _intent(session, lease)
        session.commit()
        elapsed = [0.0]
        monkeypatch.setattr(jobs, "_monotonic", lambda: elapsed[0])
        execute = session.execute

        def delayed_intent(statement, *args, **kwargs):
            result = execute(statement, *args, **kwargs)
            if getattr(statement, "is_update", False) and statement.table.name == "assistant_paid_call_intents":
                elapsed[0] = 61
            return result

        monkeypatch.setattr(session, "execute", delayed_intent)
        with pytest.raises(LostLease):
            begin_paid_submission(session, lease, intent.id, now=NOW)
        session.refresh(intent)
        assert intent.status == "prepared" and intent.submission_token is None


def test_queued_claim_deadline_starts_after_waiting_for_lock(context, monkeypatch):
    import app.assistant_jobs as jobs

    with context.factory() as session:
        job = _accept(session, context)
        session.commit()
        elapsed = [0.0]
        monkeypatch.setattr(jobs, "_monotonic", lambda: elapsed[0])
        execute = session.execute

        def delayed_lock(statement, *args, **kwargs):
            result = execute(statement, *args, **kwargs)
            if getattr(statement, "is_update", False) and statement.table.name == "assistant_jobs":
                elapsed[0] = 61
            return result

        monkeypatch.setattr(session, "execute", delayed_lock)
        lease = _claim(session, job.id)
        assert lease.expires_at == NOW + timedelta(seconds=121)


@pytest.mark.parametrize("operation", ["fail", "cancel_ack"])
def test_expiry_during_version_terminal_lock_rolls_back_both_states(context, monkeypatch, operation):
    import app.assistant_jobs as jobs
    import app.assistant_store as store

    with context.factory() as session:
        job = _accept(session, context)
        lease = _claim(session, job.id)
        if operation == "cancel_ack":
            request_cancel(session, job.id, now=NOW)
        session.commit()
        elapsed = [0.0]
        monkeypatch.setattr(jobs, "_monotonic", lambda: elapsed[0])
        original = store.mark_version_terminal

        def delayed_version(*args, **kwargs):
            result = original(*args, **kwargs)
            elapsed[0] = 61
            return result

        monkeypatch.setattr(store, "mark_version_terminal", delayed_version)
        with pytest.raises(LostLease):
            if operation == "fail":
                fail_job(session, lease, error_code="late_failure", now=NOW)
            else:
                finalize_cancel(session, job.id, lease=lease, now=NOW)
        session.refresh(job)
        assert job.status == ("running" if operation == "fail" else "cancel_requested")
        assert session.get(AssistantVersion, job.pending_version_id).status == "pending"
        assert not any(e.event_key == "terminal" for e in pending_outbox(session))


@pytest.mark.parametrize("operation", ["renew", "phase"])
def test_new_lease_must_still_be_live_when_renewal_or_phase_returns(context, monkeypatch, operation):
    import app.assistant_jobs as jobs

    with context.factory() as session:
        job = _accept(session, context)
        lease = _claim(session, job.id)
        session.commit()
        elapsed = [0.0]
        monkeypatch.setattr(jobs, "_monotonic", lambda: elapsed[0])
        flush = session.flush

        def delayed_flush(*args, **kwargs):
            granting_short_lease = any(
                isinstance(item, AssistantJob) and item.lease_expires_at == NOW + timedelta(seconds=1)
                for item in session.dirty
            )
            result = flush(*args, **kwargs)
            if granting_short_lease:
                elapsed[0] = 2
            return result

        monkeypatch.setattr(session, "flush", delayed_flush)
        with pytest.raises(LostLease):
            if operation == "renew":
                renew_lease(session, lease, now=NOW, lease_sec=1)
            else:
                advance_phase(session, lease, phase="render", now=NOW, lease_sec=1)
        session.refresh(job)
        assert job.lease_token == lease.token
        assert job.lease_generation == lease.generation
        assert job.phase == lease.phase
