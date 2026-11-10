"""Rehearse additive agent DDL and evidence-preserving downgrade fence."""
from sqlalchemy import inspect, text
from sqlalchemy.exc import DatabaseError
from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
from tests.test_assistant_turns import setup, accept, NOW
from tests.test_assistant_enrichment_migration import migration
from app.assistant_agent_steps import begin_step
from app.assistant_agent_contracts import AgentStepContext
from app.assistant_turns import claim_turn


def test_populated_legacy_turn_survives_agent_round_trip(setup):
    with setup.factory.begin() as session:
        turn_id = accept(setup,session).id
    engine = setup.factory.kw['bind']
    agent = migration('i9d4e1f6a803_assistant_agent_steps.py')
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            agent.downgrade()
            assert 'assistant_agent_steps' not in inspect(connection).get_table_names()
            agent.upgrade()
        assert connection.scalar(text('SELECT workflow_revision FROM assistant_turns WHERE id=:id'),{'id':turn_id}) == 1
        with pytest.raises(DatabaseError), connection.begin_nested():
            connection.execute(text('UPDATE assistant_turns SET workflow_revision=2 WHERE id=:id'),{'id':turn_id})
    from tests.test_assistant_turns import service
    with setup.factory.begin() as session:
        lease=claim_turn(session,turn_id,owner='test',now=NOW)
        context=service(setup).load_context(session,lease,now=NOW)
        step_id,_=begin_step(session,lease,AgentStepContext(context=context,step_index=0),now=NOW)
    with engine.begin() as connection:
        with pytest.raises(DatabaseError), connection.begin_nested():
            connection.execute(text("UPDATE assistant_agent_steps SET context_sha256=:hash WHERE id=:id"),{'hash':'f'*64,'id':step_id})
        with Operations.context(MigrationContext.configure(connection)):
            with pytest.raises(RuntimeError,match='refusing to discard'):
                agent.downgrade()


import os
import uuid
from app.db import Base
from app.assistant_models import AssistantConversation, AssistantVersion
from app.assistant_store import append_message
from app.assistant_turns import AssistantTurn
from sqlalchemy.orm import Session
from tests.test_assistant_postgres import pg_namespace


@pytest.mark.skipif(not os.environ.get('ASSISTANT_TEST_PG_URL'), reason='isolated PostgreSQL opt-in required')
def test_postgres_populated_agent_migration_guards_and_downgrade(pg_namespace):
    engine, _, _ = pg_namespace
    Base.metadata.create_all(engine)
    agent = migration('i9d4e1f6a803_assistant_agent_steps.py')
    with Session(engine) as session, session.begin():
        conversation=AssistantConversation(id=str(uuid.uuid4()),run_id=str(uuid.uuid4()),project_id=str(uuid.uuid4()))
        session.add(conversation)
        session.flush()
        version=AssistantVersion(id=str(uuid.uuid4()),conversation_id=conversation.id,sequence=1,
            selection_epoch=0,idempotency_key='base',request_sha256='a'*64,actor='operator',recipe={},status='succeeded')
        session.add(version)
        session.flush()
        message=append_message(session,conversation.id,idempotency_key='turn',actor='operator',role='operator',content='Inspect',base_version_id=version.id)
        turn=AssistantTurn(id=str(uuid.uuid4()),conversation_id=conversation.id,base_version_id=version.id,
            operator_message_id=message.id,idempotency_key='turn',request_sha256='a'*64,actor='operator',
            context_snapshot={},status='queued',lease_generation=0,created_at=NOW,updated_at=NOW)
        session.add(turn)
        session.flush()
        turn_id=turn.id
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            agent.downgrade()
            agent.upgrade()
        assert connection.scalar(text('SELECT workflow_revision FROM assistant_turns WHERE id=:id'),{'id':turn_id})==1
        with pytest.raises(DatabaseError), connection.begin_nested():
            connection.execute(text('UPDATE assistant_turns SET workflow_revision=2 WHERE id=:id'),{'id':turn_id})
        step_id=str(uuid.uuid4())
        connection.execute(text("INSERT INTO assistant_agent_steps (id,turn_id,step_index,lease_generation,context_sha256,context_snapshot,status,cleanup_pending,created_at) VALUES (:id,:turn,0,1,:hash,'{}','submitted',true,:created)"),
            {'id':step_id,'turn':turn_id,'hash':'a'*64,'created':NOW})
        with pytest.raises(DatabaseError), connection.begin_nested():
            connection.execute(text('UPDATE assistant_agent_steps SET context_sha256=:hash WHERE id=:id'),{'hash':'b'*64,'id':step_id})
        with Operations.context(MigrationContext.configure(connection)):
            with pytest.raises(RuntimeError,match='refusing to discard'):
                agent.downgrade()
