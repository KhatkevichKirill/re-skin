"""SQL/broker integration with a fake broker; never starts native inference."""
from datetime import timedelta
import pytest
from sqlalchemy import select
from tests.test_assistant_turns import setup, service, accept, reply, NOW
from app.assistant_broker_client import BrokerError, BrokerResult, context_fingerprint
from app.assistant_jobs import AssistantJob, accept_job, claim_job, advance_phase, publish_job
from app.assistant_models import AssistantMessage
from app.assistant_store import ArtifactInput, append_message
from app.assistant_supervisor import AssistantSupervisor
from app.assistant_turns import AssistantTurn, AssistantTurnEdit, claim_turn, request_cancel


class Broker:
    def __init__(self, setup, *, state="succeeded", error=False, on_submit=None):
        self.setup, self.state, self.error, self.on_submit = setup, state, error, on_submit
        self.submitted, self.cancelled, self.acked = [], [], []
    def submit(self, turn_id, context):
        self.submitted.append(turn_id)
        self.digest = context_fingerprint(context)
        if self.on_submit:
            self.on_submit(turn_id)
        if self.error:
            raise BrokerError("unavailable")
        return self.result(turn_id, self.state)
    def result(self, turn_id, state):
        self.last_state = state
        return BrokerResult(turn_id=turn_id, context_sha256=self.digest, state=state,
            reply=reply(self.setup) if state == "succeeded" else None)
    def poll(self, turn_id):
        return self.result(turn_id, "succeeded")
    def cancel(self, turn_id):
        self.cancelled.append(turn_id)
        return self.result(turn_id, "cancelled")
    def acknowledge(self, turn_id):
        # Durable completion MUST precede acknowledgement.
        with self.setup.factory() as session:
            assert session.get(AssistantTurn, turn_id).status in ("succeeded", "failed", "cancelled")
        self.acked.append(turn_id)
        return BrokerResult(turn_id=turn_id, context_sha256=self.digest, state=self.last_state, acknowledged=True)


def supervisor(setup, broker, **kwargs):
    return AssistantSupervisor(setup.factory, service=service(setup), broker=broker,
        owner="supervisor", execute=lambda *a, **kw: None, clock=lambda: NOW, sleep=lambda _: None, **kwargs)


def test_submit_persist_reply_then_ack_and_no_reinference(setup):
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
    broker = Broker(setup)
    worker = supervisor(setup, broker)
    assert worker.run_once() is True
    assert worker.run_once() is False
    assert broker.submitted == broker.acked == [turn_id]
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).status == "succeeded"


def test_unknown_submit_outcome_cancels_same_id_and_never_resubmits(setup):
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
    broker = Broker(setup, error=True)
    worker = supervisor(setup, broker)
    worker.run_once()
    worker.run_once()
    assert broker.submitted == [turn_id] and broker.cancelled == [turn_id]
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).status == "failed"


def test_operator_cancel_waits_for_broker_ack_before_terminal_state(setup):
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
    def cancel(turn_id):
        with setup.factory.begin() as session:
            request_cancel(session, turn_id, now=NOW)
    broker = Broker(setup, state="running", on_submit=cancel)
    supervisor(setup, broker).run_once()
    assert broker.cancelled == broker.acked == [turn_id]
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).status == "cancelled"
        assert session.get(AssistantTurn, turn_id).assistant_message_id is None


def test_recovery_fences_old_claim_before_broker_cancel(setup):
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
        lease = claim_turn(session, turn_id, owner="old", now=NOW - timedelta(seconds=5), lease_sec=1)
        from app.assistant_turns import mark_broker_submission
        mark_broker_submission(session, lease, context_sha256="a" * 64, now=NOW - timedelta(seconds=5))
    broker = Broker(setup)
    broker.digest = "a" * 64
    broker.poll = lambda identity: broker.result(identity, "running")
    supervisor(setup, broker).run_once()
    assert broker.submitted == [] and broker.cancelled == [turn_id]
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).error_code == "interrupted"


def test_execution_message_is_bound_by_authorized_job_and_exact_snapshot(setup):
    from tests.test_assistant_turns import plan
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        job = accept_job(session, plan(setup), request_scope="fixture", idempotency_key="job", actor="operator", now=NOW)
        edit = AssistantTurnEdit(id=__import__('uuid').uuid4().__str__(), turn_id=turn.id, job_id=job.id, idempotency_key="accept", created_at=NOW)
        session.add(edit)
        lease = claim_job(session, job.id, owner="worker", now=NOW)
        for phase in ("render", "inspect", "publish"):
            lease = advance_phase(session, lease, phase=phase, now=NOW)
        artifact = ArtifactInput(setup.base.path, metadata={"duration_sec": 2})
        publish_job(session, lease, full=artifact, preview=artifact, storage=setup.storage, now=NOW)
        message = append_message(session, turn.conversation_id, idempotency_key="execution:" + job.id,
            actor="operator", role="assistant", content="Applied actual edit", base_version_id=job.pinned_base_version_id,
            context_snapshot={"job_id": job.id, "version_id": job.pending_version_id, "status": "succeeded"})
        edit_id, message_id = edit.id, message.id
    worker = supervisor(setup, Broker(setup))
    worker.bind_execution_messages()
    with setup.factory() as session:
        assert session.get(AssistantTurnEdit, edit_id).execution_message_id == message_id


def test_unresponsive_cancel_has_bounded_wait_without_false_process_ack(setup):
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
    def cancel(turn_id):
        with setup.factory.begin() as session:
            request_cancel(session, turn_id, now=NOW)
    broker = Broker(setup, state="running", on_submit=cancel)
    broker.cancel = lambda turn_id: broker.result(turn_id, "cancel_requested")
    elapsed = [0]
    worker = supervisor(setup, broker, planning_deadline_sec=30, monotonic=lambda: elapsed[0])
    worker.sleep = lambda _: elapsed.__setitem__(0, elapsed[0] + 10)
    worker.run_once()
    assert elapsed[0] <= 50 and broker.acked == []
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).status == "cancel_requested"


def test_recovery_never_cancels_a_turn_whose_expiry_transition_lost_to_renewal(setup, monkeypatch):
    from app import assistant_supervisor
    from app.assistant_turns import mark_broker_submission
    from sqlalchemy import update
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
        lease = claim_turn(session, turn_id, owner="worker", now=NOW - timedelta(seconds=5), lease_sec=1)
        mark_broker_submission(session, lease, context_sha256="a"*64, now=NOW - timedelta(seconds=5))
    def renewed_before_expiry_compare_and_set(session, **kwargs):
        session.execute(update(AssistantTurn).where(AssistantTurn.id == turn_id).values(lease_expires_at=NOW + timedelta(seconds=120)))
        return ()
    monkeypatch.setattr(assistant_supervisor, "recover_expired", renewed_before_expiry_compare_and_set)
    broker = Broker(setup)
    broker.digest = "a"*64
    worker = supervisor(setup, broker)
    worker.recover()
    assert broker.cancelled == broker.acked == []
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).status == "running"


def test_terminal_cleanup_retries_async_cancel_and_lost_ack_response(setup):
    from app.assistant_turns import mark_broker_submission, fail_turn
    clock = [NOW]
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        mark_broker_submission(session, lease, context_sha256="a"*64, now=NOW)
        fail_turn(session, lease, error_code="ambiguous_runtime", now=NOW)
    class AsyncBroker:
        state, acknowledged, cancelled, acks = "running", False, 0, 0
        def poll(self, identity):
            return BrokerResult(turn_id=identity, context_sha256="a"*64, state=self.state, acknowledged=self.acknowledged)
        def cancel(self, identity):
            self.cancelled += 1
            self.state = "cancel_requested"
            return self.poll(identity)
        def acknowledge(self, identity):
            self.acks += 1
            self.acknowledged = True
            if self.acks == 1:
                raise BrokerError("unavailable")  # Broker persisted ACK, response lost.
            return self.poll(identity)
    broker = AsyncBroker()
    worker = AssistantSupervisor(setup.factory, service=service(setup), broker=broker, owner="worker",
        clock=lambda: clock[0], execute=lambda *args, **kwargs: None)
    worker.reconcile_broker_cleanup()
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).broker_cleanup_pending is True
    broker.state = "cancelled"
    clock[0] += timedelta(seconds=6)
    worker.reconcile_broker_cleanup()
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).broker_cleanup_pending is True
    clock[0] += timedelta(seconds=6)
    worker.reconcile_broker_cleanup()
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).broker_cleanup_pending is False
    assert broker.cancelled == 1 and broker.acks == 2


@pytest.mark.parametrize("missing_on_ack", [False, True])
def test_terminal_cleanup_requires_durable_fence_for_missing_submission(setup, missing_on_ack):
    from app.assistant_turns import mark_broker_submission, fail_turn
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        mark_broker_submission(session, lease, context_sha256="a"*64, now=NOW)
        fail_turn(session, lease, error_code="not_submitted", now=NOW)
    class GoneBroker:
        fenced = False
        def poll(self, identity):
            raise BrokerError("not_found", 404)
        def acknowledge(self, identity):
            raise BrokerError("not_found", 404)
        def fence(self, identity, fingerprint):
            self.fenced = True
            return BrokerResult(turn_id=identity, context_sha256=fingerprint,
                                state="cancelled", error_code="cancelled", acknowledged=True)
    broker = GoneBroker()
    worker = supervisor(setup, broker)
    if missing_on_ack:
        worker._ack(turn_id)
    else:
        worker.reconcile_broker_cleanup()
    assert broker.fenced
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).broker_cleanup_pending is False


def test_recovery_fences_submit_delayed_after_committed_intent(setup, tmp_path):
    """The recovering owner wins before the delayed HTTP submission arrives.

    Use the actual broker ledger: clearing the SQL flag must leave a durable
    broker tombstone, rather than creating an unacknowledged runtime later.
    """
    from app.assistant_broker import Broker as RuntimeBroker
    from tests.test_assistant_broker import FakePlanner
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
    directory = tmp_path / "ledger"
    directory.mkdir(mode=0o700)
    planner = FakePlanner()
    runtime = RuntimeBroker(directory / "state.sqlite", planner)
    clock = [NOW]
    recovery = AssistantSupervisor(setup.factory, service=service(setup), broker=runtime,
        owner="recovery", clock=lambda: clock[0], execute=lambda *args, **kwargs: None)
    class DelayedSubmit:
        def submit(self, identity, context):
            # _plan already committed mark_broker_submission. Simulate a paused
            # owner, expiry cleanup, and only then arrival at the broker.
            clock[0] += timedelta(seconds=121)
            recovery.recover()
            with setup.factory() as session:
                turn = session.get(AssistantTurn, identity)
                assert turn.status == "failed" and not turn.broker_cleanup_pending
            return runtime.submit(identity, context)
        def cancel(self, identity):
            return runtime.cancel(identity)
    try:
        worker = AssistantSupervisor(setup.factory, service=service(setup), broker=DelayedSubmit(),
            owner="paused", clock=lambda: clock[0], execute=lambda *args, **kwargs: None)
        worker._plan(turn_id)
        result = runtime.poll(turn_id)
        assert result.state == "cancelled" and result.acknowledged and result.reply is None
        assert planner.calls == 0 and runtime.health().queued == 0
        recovery.reconcile_broker_cleanup()
        with setup.factory() as session:
            assert not session.get(AssistantTurn, turn_id).broker_cleanup_pending
    finally:
        runtime.close()


def test_lost_fence_response_retains_cleanup_until_tombstone_is_reconciled(setup, tmp_path):
    from app.assistant_broker import Broker as RuntimeBroker
    from app.assistant_turns import mark_broker_submission, fail_turn
    from tests.test_assistant_broker import FakePlanner
    clock = [NOW]
    with setup.factory.begin() as session:
        turn_id = accept(setup, session).id
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        mark_broker_submission(session, lease, context_sha256="a"*64, now=NOW)
        fail_turn(session, lease, error_code="interrupted", now=NOW)
    directory = tmp_path / "ledger"
    directory.mkdir(mode=0o700)
    runtime = RuntimeBroker(directory / "state.sqlite", FakePlanner())
    class LostFence:
        def poll(self, identity): return runtime.poll(identity)
        def acknowledge(self, identity): return runtime.acknowledge(identity)
        def fence(self, identity, fingerprint):
            runtime.fence(identity, fingerprint)
            raise BrokerError("unavailable")
    try:
        worker = AssistantSupervisor(setup.factory, service=service(setup), broker=LostFence(),
            owner="worker", clock=lambda: clock[0], execute=lambda *args, **kwargs: None)
        worker.reconcile_broker_cleanup()
        with setup.factory() as session:
            assert session.get(AssistantTurn, turn_id).broker_cleanup_pending
        clock[0] += timedelta(seconds=6)
        worker.reconcile_broker_cleanup()
        with setup.factory() as session:
            assert not session.get(AssistantTurn, turn_id).broker_cleanup_pending
        assert runtime.poll(turn_id).acknowledged
    finally:
        runtime.close()
