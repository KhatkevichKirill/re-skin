"""Opt-in planning lease races on the existing isolated dummy PostgreSQL only."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import uuid
from sqlalchemy import select
from tests.test_assistant_jobs_postgres import pg, NOW
from app.assistant_models import AssistantConversation
from app.assistant_store import append_message
from app.assistant_turns import AssistantTurn, claim_turn, request_cancel, renew_turn
from app.assistant_jobs import LostLease
import pytest
import os

pytestmark = pytest.mark.skipif(not os.environ.get("ASSISTANT_TEST_POSTGRES_URL"), reason="explicit isolated PostgreSQL opt-in required")


def seed(pg):
    with pg.factory.begin() as session:
        conversation_id = session.scalar(select(AssistantConversation.id))
        message = append_message(session, conversation_id, idempotency_key="fixture-turn", actor="operator",
            role="operator", content="Trim ending", base_version_id=pg.base_id)
        turn = AssistantTurn(id=str(uuid.uuid4()), conversation_id=conversation_id, base_version_id=pg.base_id,
            operator_message_id=message.id, idempotency_key="fixture-turn", request_sha256="a"*64, actor="operator",
            context_snapshot={}, status="queued", lease_generation=0, created_at=NOW, updated_at=NOW)
        session.add(turn)
        session.flush()
        return turn.id


def test_two_supervisors_claim_one_planning_turn(pg):
    turn_id = seed(pg)
    barrier = Barrier(2)
    def claim(owner):
        with pg.factory.begin() as session:
            barrier.wait(timeout=5)
            return claim_turn(session, turn_id, owner=owner, now=NOW)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("first", "second")))
    assert sum(lease is not None for lease in results) == 1
    with pg.factory() as session:
        assert session.get(AssistantTurn, turn_id).lease_generation == 1


def test_cancellation_fences_renewal_after_cross_connection_commit(pg):
    turn_id = seed(pg)
    with pg.factory.begin() as session:
        lease = claim_turn(session, turn_id, owner="first", now=NOW)
    with pg.factory.begin() as session:
        assert request_cancel(session, turn_id, now=NOW).status == "cancel_requested"
    with pg.factory.begin() as session:
        with pytest.raises(LostLease):
            renew_turn(session, lease, now=NOW)
