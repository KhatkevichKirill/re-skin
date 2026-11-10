"""Transitive history ACL regression; no live inference or production media."""
import hashlib
import json
import pytest
from tests.test_assistant_turns import setup, service, accept, NOW
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_broker_client import context_fingerprint
from app.assistant_models import AssistantMessage
from app.assistant_planner import PlannerDialogue, UnavailableReply
from app.assistant_store import append_message
from app.assistant_turns import AssistantTurnService, claim_turn

SECRET = "private-donor-phrase-not-in-operator-text"


def complete(setup, session, turn, explanation):
    lease = claim_turn(session, turn.id, owner="fixture", now=NOW)
    service(setup).finish(session, lease, UnavailableReply(kind="unavailable", run_id=setup.base.run,
        base_version_id=setup.base.version, explanation=explanation), now=NOW)


def chain(setup, session, complete_third=True):
    first = accept(setup, session, idempotency_key="A", source_refs=((setup.donor.artifact, setup.donor.version),))
    complete(setup, session, first, SECRET + " A")
    second = accept(setup, session, idempotency_key="B", instruction="Continue the edit")
    assert second.context_snapshot["sources"] == []
    assert any(m.get("source_message_id") == first.assistant_message_id for m in second.context_snapshot["dialogue"])
    complete(setup, session, second, SECRET + " B")
    third = accept(setup, session, idempotency_key="C", instruction="Continue again")
    if complete_third:
        complete(setup, session, third, SECRET + " C")
    return first, second, third


@pytest.mark.parametrize("complete_third", [False, True])
def test_transitive_a_b_c_revocation_blocks_planner_and_redacts_ui(setup, complete_third):
    from app.api_assistant import get_context, get_turn
    with setup.factory.begin() as session:
        turns = chain(setup, session, complete_third)
        restricted = AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist([setup.base.project]), enabled=True)
        for turn in turns:
            with pytest.raises(AssistantAccessDenied):
                restricted.reauthorize_context(session, restricted.effective_context(turn), verify_bytes=False)
            data = get_turn(setup.base.run, turn.id, db=session, service=restricted)
            assert data["source_access_changed"] and data["reply"] is None and data["speech_evidence"] == []
        if not complete_third:
            lease = claim_turn(session, turns[-1].id, owner="fixture", now=NOW)
            with pytest.raises(AssistantAccessDenied):
                restricted.load_context(session, lease, now=NOW)
        history = get_context(setup.base.run, db=session, service=restricted)
        assert SECRET not in json.dumps(history, default=str)
        assert [m["role"] for m in history["messages"]] == ["operator"] * 3
        clean = restricted.accept(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Trim ending", idempotency_key="clean", actor="operator", now=NOW)
        assert all(m["role"] == "operator" for m in clean.context_snapshot["dialogue"])
        assert SECRET not in json.dumps(clean.context_snapshot)
        lease = claim_turn(session, clean.id, owner="fixture", now=NOW)
        restricted.load_context(session, lease, now=NOW)


def test_original_dependency_remains_when_its_message_is_outside_recent_window(setup):
    with setup.factory.begin() as session:
        first, second, third = chain(setup, session)
        message = session.get(AssistantMessage, second.assistant_message_id)
        narrowed = service(setup).effective_context(third).model_copy(update={"dialogue": (
            PlannerDialogue(role="assistant", text=message.content, source_message_id=message.id),
            PlannerDialogue(role="operator", text="Current request"))})
        service(setup).reauthorize_context(session, narrowed, verify_bytes=False)
        restricted = AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist([setup.base.project]), enabled=True)
        with pytest.raises(AssistantAccessDenied):
            restricted.reauthorize_context(session, narrowed, verify_bytes=False)


def test_provenance_rejects_foreign_operator_unowned_and_changed_text(setup):
    with setup.factory.begin() as session:
        first = accept(setup, session, idempotency_key="first")
        complete(setup, session, first, "Original assistant text")
        source = session.get(AssistantMessage, first.assistant_message_id)
        operator = session.get(AssistantMessage, first.operator_message_id)
        unowned = append_message(session, setup.base.conversation, idempotency_key="unowned", actor="operator",
            role="assistant", content="Unowned text", base_version_id=setup.base.version)
        foreign = service(setup).accept(session, run_id=setup.donor.run, base_version_id=setup.donor.version,
            instruction="Foreign", idempotency_key="foreign", actor="operator", now=NOW)
        lease = claim_turn(session, foreign.id, owner="fixture", now=NOW)
        service(setup).finish(session, lease, UnavailableReply(kind="unavailable", run_id=setup.donor.run,
            base_version_id=setup.donor.version, explanation="Foreign text"), now=NOW)
        foreign_message = session.get(AssistantMessage, foreign.assistant_message_id)
        original = service(setup).effective_context(first)
        for identity, content in [(operator.id, operator.content), (unowned.id, unowned.content),
                (foreign_message.id, foreign_message.content), (source.id, "Changed text")]:
            forged = original.model_copy(update={"dialogue": (
                PlannerDialogue(role="assistant", text=content, source_message_id=identity),
                PlannerDialogue(role="operator", text="Current request"))})
            with pytest.raises(AssistantAccessDenied):
                service(setup).reauthorize_context(session, forged, verify_bytes=False)
        forged = original.model_copy(update={"dialogue": (
            PlannerDialogue(role="operator", text=operator.content, source_message_id=source.id),)})
        with pytest.raises(AssistantAccessDenied):
            service(setup).reauthorize_context(session, forged, verify_bytes=False)


def test_legacy_unprovenanced_history_denied_and_none_keeps_canonical_hash(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        plain = service(setup).effective_context(turn)
        encoded = json.dumps(plain.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        assert "source_message_id" not in encoded.decode()
        assert context_fingerprint(plain) == hashlib.sha256(encoded).hexdigest()
        legacy = plain.model_copy(update={"dialogue": (
            PlannerDialogue(role="assistant", text=SECRET), PlannerDialogue(role="operator", text="New request"))})
        with pytest.raises(AssistantAccessDenied):
            service(setup).reauthorize_context(session, legacy, verify_bytes=False)


def test_history_graph_is_bounded_and_cycle_fails_closed(setup, monkeypatch):
    from app import assistant_turns
    with setup.factory.begin() as session:
        first, _, third = chain(setup, session)
        value = service(setup).effective_context(third)
        monkeypatch.setattr(assistant_turns, "_MAX_HISTORY_DEPENDENCY_TURNS", 1)
        with pytest.raises(AssistantAccessDenied):
            service(setup).reauthorize_context(session, value, verify_bytes=False)
        monkeypatch.setattr(assistant_turns, "_MAX_HISTORY_DEPENDENCY_TURNS", 256)
        cyclic = service(setup).effective_context(first).model_copy(update={"dialogue": (
            PlannerDialogue(role="assistant", text=SECRET + " A", source_message_id=first.assistant_message_id),
            PlannerDialogue(role="operator", text="Original request"))})
        checker = service(setup)
        original = checker.effective_context
        monkeypatch.setattr(checker, "effective_context", lambda turn: cyclic if turn.id == first.id else original(turn))
        with pytest.raises(AssistantAccessDenied):
            checker.reauthorize_context(session, cyclic, verify_bytes=False)


def test_reaching_history_graph_limit_does_not_permanently_block_new_turns(setup, monkeypatch):
    from app import assistant_turns
    with setup.factory.begin() as session:
        first = accept(setup, session, idempotency_key="first")
        complete(setup, session, first, "First reply")
        second = accept(setup, session, idempotency_key="second")
        complete(setup, session, second, "Second reply")
        monkeypatch.setattr(assistant_turns, "_MAX_HISTORY_DEPENDENCY_TURNS", 1)
        third = accept(setup, session, idempotency_key="third", instruction="Current own request")
        assert third.context_snapshot["dialogue"][-1]["text"] == "Current own request"
        assert all(item["role"] == "operator" for item in third.context_snapshot["dialogue"])
        lease = claim_turn(session, third.id, owner="fixture", now=NOW)
        service(setup).load_context(session, lease, now=NOW)
