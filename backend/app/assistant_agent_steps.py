"""Append-only inference inputs and write-once action/evidence for bounded turns."""
from __future__ import annotations

from datetime import datetime
import uuid
from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, DDL, ForeignKey, Index, Integer, String, UniqueConstraint, event, inspect, select
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base
from .assistant_jobs import _OperationClock, _atomic
import copy


class AssistantAgentStep(Base):
    __tablename__ = 'assistant_agent_steps'
    __table_args__ = (
        UniqueConstraint('turn_id', 'step_index', name='uq_assistant_agent_step_index'),
        CheckConstraint('step_index >= 0 AND step_index < 12', name='ck_assistant_agent_step_index'),
        CheckConstraint("status IN ('submitted','acting','completed')", name='ck_assistant_agent_step_status'),
        Index('ix_assistant_agent_step_cleanup', 'cleanup_pending', 'cleanup_next_at'),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    turn_id: Mapped[str] = mapped_column(String(36), ForeignKey('assistant_turns.id', ondelete='RESTRICT'), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    context_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    context_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    observation: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    private_artifact: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    cleanup_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default='true')
    cleanup_next_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


@event.listens_for(AssistantAgentStep, 'before_update')
def _immutable_step(mapper, connection, target):
    from .assistant_turns import TurnConflict
    for name in ('id','turn_id','step_index','lease_generation','context_sha256','context_snapshot','created_at'):
        if inspect(target).attrs[name].history.has_changes():
            raise TurnConflict('agent inference input is immutable')
    for name in ('action','observation','private_artifact','completed_at'):
        history = inspect(target).attrs[name].history
        if history.has_changes() and history.deleted and history.deleted[0] is not None:
            raise TurnConflict('agent action and evidence are write-once')
    history = inspect(target).attrs.status.history
    if history.has_changes() and history.deleted and history.deleted[0] == 'completed':
        raise TurnConflict('completed agent step is immutable')


def begin_step(session, lease, step, *, now):
    from .assistant_turns import _lock_owned, TurnConflict
    from .assistant_broker_client import context_fingerprint
    clock = _OperationClock.start(now)
    with _atomic(session):
        _lock_owned(session, lease, clock)
        previous = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == lease.turn_id)
            .order_by(AssistantAgentStep.step_index)).all()
        if len(previous) != step.step_index or any(row.status != 'completed' for row in previous):
            raise TurnConflict('agent steps cannot be replayed or skipped')
        import json
        total = sum(len(json.dumps(row.context_snapshot).encode()) + len(json.dumps(row.observation).encode()) for row in previous)
        if total + len(json.dumps(compact_snapshot(step)).encode()) > 24 * 1024 * 1024:
            raise TurnConflict('agent persisted evidence budget exceeded')
        row = AssistantAgentStep(id=str(uuid.uuid4()), turn_id=lease.turn_id, step_index=step.step_index,
            lease_generation=lease.generation, context_sha256=context_fingerprint(step.context, agent_step=step),
            context_snapshot=compact_snapshot(step), status='submitted', cleanup_pending=True, created_at=clock.now())
        session.add(row)
        session.flush()
        return row.id, row.context_sha256


def save_action(session, lease, step_id, action, *, now):
    from .assistant_turns import _lock_owned, TurnConflict
    clock = _OperationClock.start(now)
    with _atomic(session):
        _lock_owned(session, lease, clock)
        row = session.get(AssistantAgentStep, step_id, populate_existing=True)
        if row is None or row.turn_id != lease.turn_id or row.lease_generation != lease.generation or row.status != 'submitted':
            raise TurnConflict('agent action no longer owns its step')
        row.action, row.status = action.model_dump(mode='json'), 'acting'
        session.flush()


def complete_step(session, lease, step_id, *, observation=None, private_artifact=None, now):
    from .assistant_turns import _lock_owned, TurnConflict
    clock = _OperationClock.start(now)
    with _atomic(session):
        _lock_owned(session, lease, clock)
        row = session.get(AssistantAgentStep, step_id, populate_existing=True)
        if row is None or row.turn_id != lease.turn_id or row.lease_generation != lease.generation or row.status != 'acting':
            raise TurnConflict('agent evidence no longer owns its step')
        import json
        payload = observation.model_dump(mode='json') if observation is not None else None
        previous = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == lease.turn_id)).all()
        total = sum(len(json.dumps(item.context_snapshot).encode()) + len(json.dumps(item.observation).encode())
            + len(json.dumps(item.action).encode()) for item in previous)
        if total + len(json.dumps(payload).encode()) > 24 * 1024 * 1024:
            raise TurnConflict('agent persisted evidence budget exceeded')
        row.observation = payload
        row.private_artifact, row.status, row.completed_at = private_artifact, 'completed', clock.now()
        session.flush()


def compact_snapshot(step):
    snapshot = step.model_dump(mode='json')
    for observation in snapshot['observations']:
        for frame in observation['frames']:
            frame.pop('data_base64')
    return snapshot


def restore_step(session, row):
    from .assistant_agent_contracts import AgentStepContext
    from .assistant_broker_client import context_fingerprint
    from .assistant_turns import TurnConflict
    snapshot = copy.deepcopy(row.context_snapshot)
    payloads = {}
    prior = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == row.turn_id,
        AssistantAgentStep.step_index < row.step_index, AssistantAgentStep.status == 'completed')).all()
    for previous in prior:
        for frame in (previous.observation or {}).get('frames', []):
            payloads[frame['frame_id']] = frame
    for observation in snapshot['observations']:
        for frame in observation['frames']:
            original = payloads.get(frame['frame_id'])
            if original is None or {k:v for k,v in original.items() if k != 'data_base64'} != frame:
                raise TurnConflict('agent frame evidence reference changed')
            frame['data_base64'] = original['data_base64']
    step = AgentStepContext.model_validate(snapshot)
    if context_fingerprint(step.context, agent_step=step) != row.context_sha256:
        raise TurnConflict('agent inference evidence fingerprint changed')
    return step


_pinned = ('id','turn_id','step_index','lease_generation','context_sha256','context_snapshot','created_at')
_once = ('action','observation','private_artifact','completed_at')
_sqlite_guard = ' OR '.join(f'NEW.{name} IS NOT OLD.{name}' for name in _pinned)
_sqlite_guard += ' OR ' + ' OR '.join(f'(OLD.{name} IS NOT NULL AND NEW.{name} IS NOT OLD.{name})' for name in _once)
_sqlite_guard += " OR (OLD.status = 'completed' AND NEW.status IS NOT OLD.status)"
_pg_guard = ' OR '.join(f'to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name})' for name in _pinned)
_pg_guard += ' OR ' + ' OR '.join(f'(OLD.{name} IS NOT NULL AND to_jsonb(NEW.{name}) IS DISTINCT FROM to_jsonb(OLD.{name}))' for name in _once)
_pg_guard += " OR (OLD.status = 'completed' AND NEW.status IS DISTINCT FROM OLD.status)"
event.listen(AssistantAgentStep.__table__, 'after_create', DDL(
    f"CREATE TRIGGER assistant_agent_steps_immutable BEFORE UPDATE ON assistant_agent_steps WHEN {_sqlite_guard} BEGIN SELECT RAISE(ABORT, 'immutable agent evidence'); END").execute_if(dialect='sqlite'))
event.listen(AssistantAgentStep.__table__, 'after_create', DDL(
    f"CREATE OR REPLACE FUNCTION assistant_agent_steps_reject_update() RETURNS trigger AS $$ BEGIN IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'immutable agent evidence'; END IF; IF {_pg_guard} THEN RAISE EXCEPTION 'immutable agent evidence'; END IF; RETURN NEW; END; $$ LANGUAGE plpgsql; CREATE TRIGGER assistant_agent_steps_immutable BEFORE UPDATE OR DELETE ON assistant_agent_steps FOR EACH ROW EXECUTE FUNCTION assistant_agent_steps_reject_update()").execute_if(dialect='postgresql'))
event.listen(AssistantAgentStep.__table__, 'after_drop', DDL('DROP FUNCTION IF EXISTS assistant_agent_steps_reject_update()').execute_if(dialect='postgresql'))


@event.listens_for(AssistantAgentStep, 'before_delete')
def _no_delete_step(mapper, connection, target):
    from .assistant_turns import TurnConflict
    raise TurnConflict('durable agent evidence cannot be deleted')

event.listen(AssistantAgentStep.__table__, 'after_create', DDL(
    "CREATE TRIGGER assistant_agent_steps_no_delete BEFORE DELETE ON assistant_agent_steps BEGIN SELECT RAISE(ABORT, 'immutable agent evidence'); END").execute_if(dialect='sqlite'))
