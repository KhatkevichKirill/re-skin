"""Private fixtures only; no inference, broker or production media."""
from datetime import timedelta
import pytest
from sqlalchemy import func, select
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from app import assistant_media
from app.assistant_store import (AssistantArtifactStorage, ArtifactInput, get_version_artifact,
    import_run_version, reserve_version, publish_version)
from app.assistant_contracts import EditPlan
from app.db import Base
from app.models import Run, VideoProject
NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)

@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'turns.db'}")
    @event.listens_for(engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    data = tmp_path / "data"
    data.mkdir()
    storage = AssistantArtifactStorage(data)
    sources = []
    with factory.begin() as session:
        for index in range(3):
            path = data / f"source-{index}.mp4"
            path.write_bytes(f"fixture-{index}".encode())
            project = VideoProject(source_type="upload", source_ref="fixture")
            session.add(project)
            session.flush()
            run = Run(project_id=project.id, status="done", result_local_path=str(path), result_duration_sec=3)
            session.add(run)
            session.flush()
            version = import_run_version(session, run, actor="operator", storage=storage)
            sources.append(SimpleNamespace(project=project.id, run=run.id, version=version.id,
                conversation=version.conversation_id, artifact=get_version_artifact(session, version.id).id, path=path))
    monkeypatch.setattr(assistant_media, "probe_trusted_media", lambda *args, **kwargs: assistant_media.MediaMetadata(3, 3, 160, 120, "30", "1:1", True))
    yield SimpleNamespace(factory=factory, storage=storage, data=data, base=sources[0], donor=sources[1], denied=sources[2])
    engine.dispose()


def plan(context):
    return EditPlan.model_validate({"run_id": context.base.run, "base_version_id": context.base.version,
        "operations": [{"kind": "trim", "start_sec": 0, "end_sec": 2}]})


def child(context, session):
    version = reserve_version(session, plan(context), idempotency_key="other-edit", actor="operator")
    artifact = ArtifactInput(context.base.path, metadata={"duration_sec": 3})
    publish_version(session, version.id, full=artifact, preview=artifact, storage=context.storage)
    return version.id
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_jobs import AssistantJob, LostLease
from app.assistant_models import AssistantMessage
from app.assistant_planner import ClarifyReply, ProposeReply, UnavailableReply
from app.assistant_turns import (AssistantTurn, AssistantTurnService, TurnConflict, TurnsDisabled,
    claim_turn, fail_turn, finalize_cancel, recover_expired, renew_turn, request_cancel)


def service(setup, **kwargs):
    return AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist([setup.base.project, setup.donor.project]), enabled=True, **kwargs)


def accept(setup, session, **kwargs):
    fields = dict(run_id=setup.base.run, base_version_id=setup.base.version, instruction="Trim ending", idempotency_key="request-1", actor="operator", now=NOW)
    return service(setup).accept(session, **{**fields, **kwargs})


def reply(setup):
    return ProposeReply(kind="propose", run_id=setup.base.run, base_version_id=setup.base.version, explanation="Trim to two seconds.", plan=plan(setup))


def test_durable_idempotent_acceptance_and_original_base(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        assert accept(setup, session).id == turn.id and turn.status == "queued"
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 0
        child(setup, session)
        assert accept(setup, session).base_version_id == setup.base.version
        for changes in (dict(instruction="Different"), dict(actor="other"), dict(base_version_id=setup.donor.version)):
            with pytest.raises(TurnConflict):
                accept(setup, session, **changes)
        assert session.scalar(select(func.count()).select_from(AssistantMessage)) == 1


def test_outer_rollback_removes_acceptance(setup):
    with setup.factory() as session:
        accept(setup, session)
        session.rollback()
    with setup.factory() as session:
        assert session.scalar(select(func.count()).select_from(AssistantTurn)) == 0
        assert session.scalar(select(func.count()).select_from(AssistantMessage)) == 0


def test_reply_fenced_and_context_uses_real_turn_messages(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        lease = claim_turn(session, turn.id, owner="worker", now=NOW)
        assert claim_turn(session, turn.id, owner="other", now=NOW) is None
        assert service(setup).load_context(session, lease, now=NOW).dialogue[-1].text == "Trim ending"
        lease = renew_turn(session, lease, now=NOW, lease_sec=120)
        assert service(setup).finish(session, lease, reply(setup), now=NOW).status == "succeeded"
        with pytest.raises(LostLease):
            service(setup).finish(session, lease, reply(setup), now=NOW)
        from app.assistant_store import append_message
        append_message(session, setup.base.conversation, idempotency_key="turn:spoof", actor="operator", role="operator", content="Not a turn", base_version_id=setup.base.version)
        next_turn = accept(setup, session, idempotency_key="next", instruction="Now fade")
        assert [m["text"] for m in next_turn.context_snapshot["dialogue"]] == ["Trim ending", "Trim to two seconds.", "Now fade"]


def test_running_cancel_requires_process_ack_and_fences_reply(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        lease = claim_turn(session, turn.id, owner="worker", now=NOW)
        assert request_cancel(session, turn.id, now=NOW).status == "cancel_requested"
        with pytest.raises(LostLease):
            service(setup).finish(session, lease, reply(setup), now=NOW)
        with pytest.raises(LostLease):
            finalize_cancel(session, turn.id, now=NOW)
        assert finalize_cancel(session, turn.id, lease=lease, now=NOW).status == "cancelled"


def test_expiry_terminal_and_explicit_retry_pins_original_context(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        lease = claim_turn(session, turn.id, owner="worker", now=NOW, lease_sec=1)
        assert recover_expired(session, now=NOW + timedelta(seconds=2)) == (turn.id,)
        assert session.get(AssistantTurn, turn.id).error_code == "interrupted"
        assert claim_turn(session, turn.id, owner="other", now=NOW + timedelta(seconds=2)) is None
        with pytest.raises(LostLease):
            service(setup).finish(session, lease, reply(setup), now=NOW + timedelta(seconds=2))
        child(setup, session)
        retried = service(setup).retry(session, run_id=setup.base.run, turn_id=turn.id, idempotency_key="retry", now=NOW)
        assert retried.id != turn.id and retried.parent_turn_id == turn.id
        assert retried.context_snapshot == turn.context_snapshot
        assert service(setup).retry(session, run_id=setup.base.run, turn_id=turn.id, idempotency_key="retry", now=NOW).id == retried.id


def test_revoked_donor_denies_context_reply_and_retry(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session, source_refs=((setup.donor.artifact, setup.donor.version),))
        lease = claim_turn(session, turn.id, owner="worker", now=NOW)
        restricted = AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist([setup.base.project]), enabled=True)
        with pytest.raises(AssistantAccessDenied):
            restricted.load_context(session, lease, now=NOW)
        with pytest.raises(AssistantAccessDenied):
            restricted.finish(session, lease, reply(setup), now=NOW)
        fail_turn(session, lease, error_code="permission_denied", now=NOW)
        with pytest.raises(AssistantAccessDenied):
            restricted.retry(session, run_id=setup.base.run, turn_id=turn.id, idempotency_key="retry", now=NOW)


def test_queue_limit_and_disabled_gate(setup):
    with setup.factory.begin() as session:
        bounded = service(setup, max_pending=1)
        kwargs = dict(run_id=setup.base.run, base_version_id=setup.base.version, instruction="trim", actor="operator", now=NOW)
        bounded.accept(session, idempotency_key="first", **kwargs)
        with pytest.raises(TurnConflict, match="queue is full"):
            bounded.accept(session, idempotency_key="second", **kwargs)
        disabled = AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist([setup.base.project]))
        with pytest.raises(TurnsDisabled):
            disabled.accept(session, idempotency_key="disabled", **kwargs)


def test_reply_scope_checked_before_persisting_message(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        lease = claim_turn(session, turn.id, owner="worker", now=NOW)
        with pytest.raises(TurnConflict):
            service(setup).finish(session, lease, reply(setup).model_copy(update={"base_version_id": setup.donor.version}), now=NOW)
        assert session.scalar(select(func.count()).select_from(AssistantMessage)) == 1


def test_clarify_and_unavailable_complete_without_edit(setup):
    with setup.factory.begin() as session:
        for index, model in enumerate((ClarifyReply, UnavailableReply)):
            turn = accept(setup, session, idempotency_key=f"request-{index}")
            lease = claim_turn(session, turn.id, owner="worker", now=NOW)
            fields = dict(kind="clarify" if model is ClarifyReply else "unavailable", run_id=setup.base.run, base_version_id=setup.base.version, explanation="Need a boundary.")
            if model is ClarifyReply:
                fields["question"] = "Which second?"
            assert service(setup).finish(session, lease, model(**fields), now=NOW).status == "succeeded"
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 0


@pytest.mark.parametrize("operations", [
    [{"kind": "cut", "start_sec": 0.0, "end_sec": 1.0}],
    [{"kind": "image_fade", "start_sec": 0.51, "end_sec": 1.0, "direction": "out"}],
])
def test_unsupported_plan_completes_with_explanation_and_no_render_action(setup, operations):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        lease = claim_turn(session, turn.id, owner="worker", now=NOW)
        proposal = reply(setup).model_copy(update={"plan": EditPlan.model_validate({
            "run_id": setup.base.run, "base_version_id": setup.base.version,
            "operations": operations})})
        result = service(setup).finish(session, lease, proposal, now=NOW)
        assert result.status == "succeeded"
        assert result.reply["kind"] == "unavailable"
        assert "No edit was applied" in result.reply["explanation"]
        assert "plan" not in result.reply
        message = session.get(AssistantMessage, result.assistant_message_id)
        assert message.content == result.reply["explanation"]
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 0


def test_paired_fade_remains_renderable_proposal(setup):
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        lease = claim_turn(session, turn.id, owner="worker", now=NOW)
        proposal = reply(setup).model_copy(update={"plan": EditPlan.model_validate({
            "run_id": setup.base.run, "base_version_id": setup.base.version,
            "operations": [
                {"kind": "image_fade", "start_sec": 0.5, "end_sec": 1.0, "direction": "out"},
                {"kind": "image_fade", "start_sec": 1.0, "end_sec": 1.5, "direction": "in"}]})})
        result = service(setup).finish(session, lease, proposal, now=NOW)
        assert result.reply["kind"] == "propose"
        assert len(result.reply["plan"]["operations"]) == 2


def test_custom_plan_is_immutable_during_enrichment_and_legacy_context_keeps_shape(setup):
    from app.assistant_planner import PlannerContext
    with setup.factory.begin() as session:
        original = accept(setup,session)
        lease = claim_turn(session,original.id,owner="worker",now=NOW)
        service(setup).finish(session,lease,reply(setup),now=NOW)
        old = PlannerContext.model_validate(original.context_snapshot)
        assert "reference_plan" not in old.model_dump(mode="json")
        assert "reference_turn_id" not in old.model_dump(mode="json")
        custom = accept(setup,session,idempotency_key="custom",proposal_turn_id=original.id)
        context = PlannerContext.model_validate(custom.context_snapshot)
        stripped = context.model_copy(update={"reference_plan":None,"reference_turn_id":None})
        with pytest.raises(TurnConflict,match="enrichment changed"):
            service(setup)._enrichment_binding(context,stripped)
        # Reference authorization survives history trimming independently.
        without_history = context.model_copy(update={"dialogue":(context.dialogue[-1],)})
        service(setup).reauthorize_context(session,without_history)
        with pytest.raises(AssistantAccessDenied):
            service(setup).reauthorize_context(session,without_history.model_copy(update={"reference_turn_id":custom.id}))


def test_pinned_request_and_terminal_reply_are_immutable_in_sql(setup):
    from sqlalchemy import text
    from sqlalchemy.exc import DatabaseError
    with setup.factory.begin() as session:
        turn = accept(setup, session)
        with pytest.raises(DatabaseError), session.begin_nested():
            session.execute(text("UPDATE assistant_turns SET context_snapshot = '{}' WHERE id = :id"), {"id": turn.id})
        lease = claim_turn(session, turn.id, owner="worker", now=NOW)
        service(setup).finish(session, lease, reply(setup), now=NOW)
        with pytest.raises(DatabaseError), session.begin_nested():
            session.execute(text("UPDATE assistant_turns SET reply = '{}' WHERE id = :id"), {"id": turn.id})


def test_large_utf8_history_is_trimmed_to_transport_budget_without_blocking_new_turn(setup):
    from app.assistant_broker_client import context_fingerprint
    from app.assistant_planner import PlannerContext
    with setup.factory.begin() as session:
        for index in range(18):
            turn = accept(setup, session, instruction="ы" * 3900, idempotency_key=f"long-{index}")
            request_cancel(session, turn.id, now=NOW)
        final = accept(setup, session, instruction="Trim the ending", idempotency_key="current")
        context = PlannerContext.model_validate(final.context_snapshot)
        assert context.dialogue[-1].text == "Trim the ending"
        assert len(context.dialogue) < 19
        assert context_fingerprint(context)
        assert session.scalar(select(func.count()).select_from(AssistantMessage)) == 19


def test_fresh_agent_context_reserves_observations_without_rewriting_history(setup):
    import json
    from app.assistant_turns import MAX_INITIAL_AGENT_CONTEXT_BYTES
    legacy=service(setup)
    agent=service(setup,agent_enabled=True)
    with setup.factory.begin() as session:
        for index in range(10):
            turn=accept(setup,session,instruction=f'{index} '+('ы'*3500),idempotency_key=f'reserve-{index}')
            request_cancel(session,turn.id,now=NOW)
        recent=accept(setup,session,idempotency_key='recent',source_refs=((setup.donor.artifact,setup.donor.version),))
        lease=claim_turn(session,recent.id,owner='worker',now=NOW)
        legacy.finish(session,lease,reply(setup),now=NOW)
        before=[(m.id,m.content,m.context_snapshot) for m in session.scalars(select(AssistantMessage).order_by(AssistantMessage.sequence))]
        fields=dict(run_id=setup.base.run,base_version_id=setup.base.version,instruction='Current request',
            source_refs=((setup.donor.artifact,setup.donor.version),),selected_passage={'start_sec':1.,'end_sec':2.},
            reference_turn_id=recent.id,reference_plan=reply(setup).plan)
        full=legacy.build_context(session,**fields)
        bounded=agent.build_context(session,**fields)
        assert len(json.dumps(full.model_dump(mode='json'),ensure_ascii=False,allow_nan=False).encode())>MAX_INITIAL_AGENT_CONTEXT_BYTES
        assert len(json.dumps(bounded.model_dump(mode='json'),ensure_ascii=False,allow_nan=False).encode())<=MAX_INITIAL_AGENT_CONTEXT_BYTES
        assert bounded.dialogue==full.dialogue[-len(bounded.dialogue):]
        assert bounded.dialogue[-1].text=='Current request'
        assert bounded.dialogue[-2].source_message_id==recent.assistant_message_id
        assert bounded.model_dump(exclude={'dialogue'})==full.model_dump(exclude={'dialogue'})
        assert before==[(m.id,m.content,m.context_snapshot) for m in session.scalars(select(AssistantMessage).order_by(AssistantMessage.sequence))]
        agent.reauthorize_context(session,bounded,verify_bytes=False)


@pytest.mark.parametrize('extra',[0,1])
def test_initial_agent_budget_uses_exact_expanded_json_boundary(setup,monkeypatch,extra):
    import json
    from app.assistant_turns import MAX_INITIAL_AGENT_CONTEXT_BYTES,TurnContextTooLarge
    from app.assistant_planner import PlannerSubtitleCue
    svc=service(setup,agent_enabled=True)
    original=svc.version_context
    def pinned(*args,**kwargs):
        result=original(*args,**kwargs)
        cues=tuple(PlannerSubtitleCue(cue_id=f'cue-{i}',start_sec=0.,end_sec=1.,text='x'*1000) for i in range(27))
        return {**result,'subtitle_cues':cues,'editable_subtitle_cue_ids':tuple(c.cue_id for c in cues)}
    monkeypatch.setattr(svc,'version_context',pinned)
    with setup.factory.begin() as session:
        fields=dict(run_id=setup.base.run,base_version_id=setup.base.version)
        seed=svc.build_context(session,**fields,instruction='x')
        length=MAX_INITIAL_AGENT_CONTEXT_BYTES-len(json.dumps(seed.model_dump(mode='json'),ensure_ascii=False,allow_nan=False).encode())+1
        assert 1<=length<4000
        if extra:
            with pytest.raises(TurnContextTooLarge):svc.build_context(session,**fields,instruction='x'*(length+extra))
        else:
            result=svc.build_context(session,**fields,instruction='x'*length)
            assert len(json.dumps(result.model_dump(mode='json'),ensure_ascii=False,allow_nan=False).encode())==MAX_INITIAL_AGENT_CONTEXT_BYTES
