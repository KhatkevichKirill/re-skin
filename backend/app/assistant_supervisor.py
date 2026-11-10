"""SQL queue supervisor for the isolated planner broker and trusted renderer.

Only the broker owns native credentials. This process gets app DB/media access
and the broker socket; it cannot request native tools or choose runtime flags.
All mutations use durable leases. No automatic inference retry is performed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import time

from sqlalchemy import or_, select, update

from .assistant_broker_client import BrokerError, context_fingerprint
from .assistant_jobs import (AssistantJob, AssistantJobOutbox, LostLease, acknowledge_outbox,
    finalize_cancel as finalize_job_cancel)
from .assistant_models import AssistantMessage
from .assistant_turns import (AssistantTurn, AssistantTurnEdit, AssistantTurnService, claim_turn,
    fail_turn, finalize_cancel, mark_broker_submission, recover_expired, renew_turn)

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


class _PlanningLeaseGuard:
    """Fresh SQL cancellation and heartbeats during optional slow offline ASR."""
    def __init__(self, supervisor, lease):
        self.supervisor, self.lease = supervisor, lease
        self.checked, self.renewed = float("-inf"), supervisor.monotonic()

    def __call__(self):
        supervisor = self.supervisor
        if supervisor.stopping():
            return True
        tick = supervisor.monotonic()
        if tick - self.checked < 0.25:
            return False
        self.checked = tick
        with supervisor.factory() as session:
            turn = session.get(AssistantTurn, self.lease.turn_id, populate_existing=True)
            if turn is not None and turn.status == "cancel_requested":
                return True
            if turn is None or (turn.status, turn.lease_owner, turn.lease_token, turn.lease_generation) != (
                    "running", self.lease.owner, self.lease.token, self.lease.generation):
                raise LostLease("speech enrichment lost its planning lease")
            expires = turn.lease_expires_at
            if expires is not None and expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires is None or expires <= supervisor.clock():
                raise LostLease("speech enrichment planning lease expired")
            if tick - self.renewed >= supervisor.lease_sec / 4:
                self.lease = renew_turn(session, self.lease, now=supervisor.clock(), lease_sec=supervisor.lease_sec)
                session.commit()
                self.renewed = tick
        return False


class AssistantSupervisor:
    def __init__(self, session_factory, *, service: AssistantTurnService, broker, owner: str,
                 execute=None, clock=_now, monotonic=time.monotonic, sleep=time.sleep,
                 stopping=lambda: False, enricher=None, lease_sec: int = 120, planning_deadline_sec: int = 300,
                 edit_timeout_sec: int = 600, media_factory=None, agent_deadline_sec: int = 1200,
                 agent_max_steps: int = 12):
        from .assistant_jobs import _identifier
        if type(lease_sec) is not int or not 30 <= lease_sec <= 300:
            raise ValueError("supervisor lease must be within [30,300] seconds")
        if type(planning_deadline_sec) is not int or not 30 <= planning_deadline_sec <= 600:
            raise ValueError("planning deadline must be within [30,600] seconds")
        if type(edit_timeout_sec) is not int or not 30 <= edit_timeout_sec <= 600:
            raise ValueError("edit timeout must be within [30,600] seconds")
        if execute is None:
            from .assistant_executor import execute_assistant_job
            execute = execute_assistant_job
        self.factory, self.service, self.broker = session_factory, service, broker
        self.owner, self.execute = _identifier(owner, "owner"), execute
        self.clock, self.monotonic, self.sleep, self.stopping = clock, monotonic, sleep, stopping
        self.lease_sec, self.deadline_sec = lease_sec, planning_deadline_sec
        self.edit_timeout_sec = edit_timeout_sec
        self.enricher = enricher
        if type(agent_deadline_sec) is not int or not 30 <= agent_deadline_sec <= 1800:
            raise ValueError("agent deadline must be within [30,1800] seconds")
        if type(agent_max_steps) is not int or not 1 <= agent_max_steps <= 12:
            raise ValueError("agent step budget must be within [1,12]")
        self.media_factory, self.agent_deadline_sec, self.agent_max_steps = media_factory, agent_deadline_sec, agent_max_steps

    def _safe_cancel(self, turn_id):
        try:
            return self.broker.cancel(turn_id)
        except BrokerError:
            return None

    def _cleanup_state(self, turn_id):
        with self.factory() as session:
            turn = session.get(AssistantTurn, turn_id, populate_existing=True)
            if turn is None or turn.status not in ("succeeded", "failed", "cancelled") or not turn.broker_cleanup_pending:
                return None
            return turn.broker_context_sha256

    def _cleanup_result(self, turn_id, fingerprint, *, done):
        with self.factory() as session:
            session.execute(update(AssistantTurn).where(AssistantTurn.id == turn_id,
                AssistantTurn.status.in_(("succeeded", "failed", "cancelled")), AssistantTurn.broker_cleanup_pending.is_(True),
                AssistantTurn.broker_context_sha256 == fingerprint).values(broker_cleanup_pending=not done,
                    broker_cleanup_next_at=None if done else self.clock() + timedelta(seconds=5)).execution_options(synchronize_session=False))
            session.commit()

    def _ack(self, turn_id):
        fingerprint = self._cleanup_state(turn_id)
        if fingerprint is None:
            return
        done = False
        try:
            result = self.broker.acknowledge(turn_id)
            done = (result is not None and result.turn_id == turn_id and result.context_sha256 == fingerprint and result.acknowledged
                and result.state in ("succeeded", "failed", "cancelled", "interrupted"))
        except BrokerError as error:
            if error.code == "not_found":
                done = self._fence_missing(turn_id, fingerprint)
        finally:
            self._cleanup_result(turn_id, fingerprint, done=done)

    def _fence_missing(self, turn_id, fingerprint):
        # SQL recovery can run while the prior owner is paused after committing
        # its submission intent. Absence alone cannot prove it will not submit.
        # The broker's durable hash-bound tombstone fences that delayed request.
        try:
            result = self.broker.fence(turn_id, fingerprint)
            if result.turn_id != turn_id or result.context_sha256 != fingerprint:
                return False
            if result.state not in ("succeeded", "failed", "cancelled", "interrupted"):
                return False
            if not result.acknowledged:
                result = self.broker.acknowledge(turn_id)
            return (result.turn_id == turn_id and result.context_sha256 == fingerprint and result.acknowledged
                and result.state in ("succeeded", "failed", "cancelled", "interrupted"))
        except BrokerError:
            return False

    def reconcile_broker_cleanup(self, *, turn_ids=None):
        """Persistently cancel then ACK terminal IDs, including lost responses.

        A durable indexed flag bounds this sweep; no historical full scan and no
        cancellation of a currently running/renewed database turn is allowed.
        """
        if turn_ids is None:
            with self.factory() as session:
                turn_ids = session.scalars(select(AssistantTurn.id).where(
                    AssistantTurn.broker_cleanup_pending.is_(True),
                    AssistantTurn.status.in_(("succeeded", "failed", "cancelled")),
                    or_(AssistantTurn.broker_cleanup_next_at.is_(None), AssistantTurn.broker_cleanup_next_at <= self.clock()))
                    .order_by(AssistantTurn.broker_cleanup_next_at, AssistantTurn.id).limit(4)).all()
        for turn_id in turn_ids:
            fingerprint = self._cleanup_state(turn_id)
            if fingerprint is None:
                continue
            try:
                result = self.broker.poll(turn_id)
                if result.turn_id != turn_id or result.context_sha256 != fingerprint:
                    self._cleanup_result(turn_id, fingerprint, done=False)
                    continue
                if result.state in ("queued", "running", "cancel_requested"):
                    result = self.broker.cancel(turn_id)
                    if result.turn_id != turn_id or result.context_sha256 != fingerprint:
                        self._cleanup_result(turn_id, fingerprint, done=False)
                        continue
                if result.state in ("succeeded", "failed", "cancelled", "interrupted"):
                    self._ack(turn_id)
                else:
                    self._cleanup_result(turn_id, fingerprint, done=False)
            except BrokerError as error:
                done = error.code == "not_found" and self._fence_missing(turn_id, fingerprint)
                self._cleanup_result(turn_id, fingerprint, done=done)

    def recover(self):
        """Cancel only IDs actually transitioned by the expiry compare-and-set."""
        with self.factory() as session:
            fenced = recover_expired(session, now=self.clock())
            session.commit()
        self.reconcile_broker_cleanup(turn_ids=fenced)
        self.reconcile_broker_cleanup()
        from .assistant_agent_loop import AgentLoop
        AgentLoop(self).cleanup()
        # A requested cancellation cannot be renewed. Reach the broker while
        # its current owner is absent; finalization still needs exit or expiry.
        with self.factory() as session:
            cancelled = session.scalars(select(AssistantTurn.id).where(AssistantTurn.status == "cancel_requested",
                AssistantTurn.broker_cleanup_pending.is_(True)).order_by(AssistantTurn.updated_at, AssistantTurn.id).limit(4)).all()
            jobs = session.scalars(select(AssistantJob.id).join(AssistantTurnEdit, AssistantTurnEdit.job_id == AssistantJob.id).where(
                AssistantJob.status == "cancel_requested", AssistantJob.lease_expires_at <= self.clock()).limit(100)).all()
            for job_id in jobs:
                finalize_job_cancel(session, job_id, now=self.clock())
            session.commit()
        for turn_id in cancelled:
            self._safe_cancel(turn_id)

    def _plan(self, turn_id):
        with self.factory() as session:
            turn = session.get(AssistantTurn, turn_id)
            revision = turn.workflow_revision if turn is not None else 1
        if revision == 2:
            if self.media_factory is None:
                return  # A compatible supervisor must claim revision-two turns.
            from .assistant_agent_loop import AgentLoop
            return AgentLoop(self).run(turn_id)
        with self.factory() as session:
            lease = claim_turn(session, turn_id, owner=self.owner, now=self.clock(), lease_sec=self.lease_sec)
            session.commit()
        if lease is None:
            return
        started, renewed = self.monotonic(), self.monotonic()
        terminal = False
        try:
            with self.factory() as session:
                context = self.service.load_context(session, lease, now=self.clock())
                lease = renew_turn(session, lease, now=self.clock(), lease_sec=self.lease_sec)
                session.commit()
            if self.enricher is not None:
                guard = _PlanningLeaseGuard(self, lease)
                context = self.enricher.enrich(context, cancelled=guard)
                lease = guard.lease
                with self.factory() as session:
                    context = self.service.save_enrichment(session, lease, context, now=self.clock())
                    session.commit()
                renewed = self.monotonic()
            fingerprint = context_fingerprint(context)
            with self.factory() as session:
                mark_broker_submission(session, lease, context_sha256=fingerprint, now=self.clock())
                session.commit()
            result = self.broker.submit(turn_id, context)
            while True:
                if result.context_sha256 != fingerprint or result.turn_id != turn_id:
                    raise ValueError("broker result changed the pinned turn")
                with self.factory() as session:
                    current = session.get(AssistantTurn, turn_id, populate_existing=True)
                    cancelled = current is not None and current.status == "cancel_requested"
                    if current is None or current.status not in ("running", "cancel_requested"):
                        raise LostLease("planning turn is no longer active")
                if cancelled:
                    result = self.broker.cancel(turn_id)
                    if result.context_sha256 != fingerprint or result.turn_id != turn_id:
                        raise ValueError("broker cancellation changed the pinned turn")
                    if result.state in ("succeeded", "failed", "cancelled", "interrupted"):
                        with self.factory() as session:
                            finalize_cancel(session, turn_id, lease=lease, now=self.clock())
                            session.commit()
                        terminal = True
                        break
                    if self.stopping() or self.monotonic() - started > self.deadline_sec:
                        # Keep cancel_requested until broker process exit or
                        # SQL lease expiry; do not wait forever for the broker.
                        break
                elif self.stopping() or self.monotonic() - started > self.deadline_sec:
                    self._safe_cancel(turn_id)
                    with self.factory() as session:
                        fail_turn(session, lease, error_code="supervisor_stopping" if self.stopping() else "planning_timeout", now=self.clock())
                        session.commit()
                    terminal = True
                    break
                elif result.state == "succeeded":
                    if result.reply is None or result.acknowledged:
                        raise ValueError("broker result was acknowledged before durable completion")
                    with self.factory() as session:
                        self.service.finish(session, lease, result.reply, now=self.clock())
                        session.commit()
                    terminal = True
                    break
                elif result.state in ("failed", "cancelled", "interrupted"):
                    with self.factory() as session:
                        fail_turn(session, lease, error_code="runtime_" + (result.error_code or result.state), now=self.clock())
                        session.commit()
                    terminal = True
                    break
                if self.monotonic() - renewed >= self.lease_sec / 4 and not cancelled:
                    with self.factory() as session:
                        lease = renew_turn(session, lease, now=self.clock(), lease_sec=self.lease_sec)
                        session.commit()
                    renewed = self.monotonic()
                self.sleep(0.5)
                result = self.broker.poll(turn_id)
        except LostLease:
            self._safe_cancel(turn_id)
        except Exception:
            # A timeout may follow accepted inference. Cancel this SAME broker
            # ID, fail durably and require an explicit new operator retry.
            self._safe_cancel(turn_id)
            with self.factory() as session:
                try:
                    current = session.get(AssistantTurn, turn_id, populate_existing=True)
                    if current and current.status == "cancel_requested":
                        # No process-exit evidence: recovery waits for expiry.
                        pass
                    else:
                        fail_turn(session, lease, error_code="planning_failed", now=self.clock())
                        session.commit()
                        terminal = True
                except LostLease:
                    session.rollback()
            log.warning("Assistant planning turn ended without a publishable reply", extra={"turn_id": turn_id})
        if terminal:
            self._ack(turn_id)

    def bind_execution_messages(self):
        with self.factory() as session:
            rows = session.execute(select(AssistantTurnEdit, AssistantJob).join(AssistantJob,
                AssistantJob.id == AssistantTurnEdit.job_id).where(AssistantTurnEdit.execution_message_id.is_(None),
                AssistantJob.status == "succeeded").limit(100)).all()
            for edit, job in rows:
                turn = session.get(AssistantTurn, edit.turn_id)
                message = session.scalar(select(AssistantMessage).where(AssistantMessage.conversation_id == turn.conversation_id,
                    AssistantMessage.idempotency_key == "execution:" + job.id))
                if (message is not None and message.role == "assistant" and message.base_version_id == job.pinned_base_version_id
                        and message.context_snapshot == {"job_id": job.id, "version_id": job.pending_version_id, "status": "succeeded"}):
                    edit.execution_message_id = message.id
            session.commit()

    def run_once(self):
        self.service.gate()
        self.recover()
        self.bind_execution_messages()
        if self.stopping():
            return False
        with self.factory() as session:
            turn_id = session.scalar(select(AssistantTurn.id).where(AssistantTurn.status == "queued",
                AssistantTurn.workflow_revision.in_((1, 2) if self.media_factory is not None else (1,)))
                .order_by(AssistantTurn.created_at, AssistantTurn.id).limit(1))
        if turn_id is not None:
            self._plan(turn_id)
        if self.stopping():
            return turn_id is not None
        with self.factory() as session:
            job_id = session.scalar(select(AssistantJob.id).join(AssistantTurnEdit, AssistantTurnEdit.job_id == AssistantJob.id).where(
                (AssistantJob.status == "queued") | ((AssistantJob.status == "running") & (AssistantJob.lease_expires_at <= self.clock())))
                .order_by(AssistantJob.created_at, AssistantJob.id).limit(1))
        if job_id is not None:
            self.execute(self.factory, job_id, owner=self.owner, storage=self.service.storage,
                policy=self.service.policy, enabled=True, lease_sec=self.lease_sec, timeout_sec=self.edit_timeout_sec)
            self.bind_execution_messages()
        # SQL polling is the dispatcher: observation is durable and Redis loss
        # cannot lose an accepted request. Outbox events remain audit records.
        with self.factory() as session:
            event_ids = session.scalars(select(AssistantJobOutbox.id).join(AssistantTurnEdit,
                AssistantTurnEdit.job_id == AssistantJobOutbox.job_id).where(AssistantJobOutbox.dispatched_at.is_(None)).limit(100)).all()
            for event_id in event_ids:
                acknowledge_outbox(session, event_id, now=self.clock())
            session.commit()
        return turn_id is not None or job_id is not None
