"""Offline scoped inventory, cursor and transitive metadata provenance checks."""
import json
from uuid import UUID
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_agent_contracts import AgentActionEnvelope, AgentObservation, AgentStepContext, RunsReadPage, InternalRunRow, validate_agent_action
from app.assistant_internal_api import extend_dependencies, read_internal_metadata
from app.assistant_models import AssistantArtifact, AssistantConversation, AssistantMessage, AssistantVersion
from app.assistant_planner import ClarifyReply, InternalReadDependency
from app.assistant_turns import TurnConflict, claim_turn, fail_turn, finalize_cancel, request_cancel
from app.models import Run, VideoProject
from tests.test_assistant_turns import setup, service, NOW


def action(setup, kind, **kwargs):
    return AgentActionEnvelope.model_validate({'action': dict(kind=kind, run_id=setup.base.run,
        base_version_id=setup.base.version, **kwargs)}).action


def context(setup, svc, session):
    return svc.build_context(session, run_id=setup.base.run, base_version_id=setup.base.version, instruction='Read inventory')


def read(setup, svc, session, kind, ctx=None, **kwargs):
    ctx = context(setup, svc, session) if ctx is None else ctx
    return read_internal_metadata(session, service=svc, context=ctx, action=action(setup, kind, **kwargs))


def test_four_filtered_tools_read_only_and_no_media_authority(setup):
    svc = service(setup)
    with setup.factory() as session:
        before = {model: session.scalar(select(func.count()).select_from(model)) for model in
            (VideoProject, Run, AssistantConversation, AssistantVersion, AssistantArtifact, AssistantMessage)}
        ctx = context(setup, svc, session)
        pages = []
        for kind, fields in [('list_projects', {}), ('list_runs', {'project_id': setup.donor.project}),
                ('list_versions', {'source_run_id': setup.donor.run}), ('list_assets', {'source_version_id': setup.donor.version})]:
            page, ctx = read(setup, svc, session, kind, ctx=ctx, **fields)
            pages.append(page)
            assert page.records
        assert {r.project_id for r in pages[0].records} == {setup.base.project, setup.donor.project}
        assert pages[-1].records[0].asset_id == setup.donor.artifact
        assert ctx.sources == ()
        with pytest.raises(ValueError, match='unknown source'):
            validate_agent_action(action(setup, 'inspect_frames', asset_id=setup.donor.artifact,
                version_id=setup.donor.version, timestamps_sec=[1]), AgentStepContext(context=ctx, step_index=0))
        wire = json.dumps([page.model_dump(mode='json') for page in pages])
        for forbidden in ('storage_key', 'source_ref', 'local_path', 'result_url', 'prompt', 'recipe', 'error_message', 'http', 'fixture'):
            assert forbidden not in wire
        assert not session.new and not session.dirty and not session.deleted
        assert before == {model: session.scalar(select(func.count()).select_from(model)) for model in before}


@pytest.mark.parametrize('kind,key,attribute', [('list_runs','project_id','project'),
    ('list_versions','source_run_id','run'), ('list_assets','source_version_id','version')])
def test_denied_and_missing_owners_have_same_error(setup, kind, key, attribute):
    svc = service(setup)
    with setup.factory() as session:
        for identifier in (getattr(setup.denied, attribute), str(UUID(int=99999))):
            with pytest.raises(AssistantAccessDenied, match='^assistant access denied$'):
                read(setup, svc, session, kind, **{key:identifier})


@pytest.mark.parametrize('extra', [{'path':'/etc/passwd'}, {'url':'http://internal'}, {'headers':{}},
    {'query':'SELECT *'}, {'limit':1000}, {'after_project_id':7}, {'command':'id'}])
def test_strict_schema_has_no_general_access_or_integer_cursor(setup, extra):
    with pytest.raises(ValidationError):
        action(setup, 'list_projects', **extra)


def test_cursor_requires_exact_resource_and_parent(setup):
    svc = service(setup)
    cursor = str(UUID(int=999))
    page = RunsReadPage(kind='list_runs', owner=InternalReadDependency(kind='project', project_id=setup.donor.project), records=(), next_after_id=cursor)
    with setup.factory() as session:
        ctx = extend_dependencies(context(setup, svc, session), (page.owner,))
    step = AgentStepContext(context=ctx, step_index=1, observations=(AgentObservation(kind='internal_read', action_index=0, internal_read=page),))
    good = action(setup, 'list_runs', project_id=setup.donor.project, after_run_id=cursor)
    assert validate_agent_action(good, step) == good
    for bad in (action(setup,'list_runs',project_id=setup.base.project,after_run_id=cursor),
            action(setup,'list_versions',source_run_id=setup.donor.run,after_version_id=cursor),
            action(setup,'list_runs',project_id=setup.donor.project,after_run_id=str(UUID(int=998)))):
        with pytest.raises(ValueError, match='scoped cursor'):
            validate_agent_action(bad, step)


def test_empty_page_retains_parent_and_live_ownership(setup):
    svc = service(setup)
    with setup.factory.begin() as session:
        empty = Run(project_id=setup.donor.project)
        session.add(empty)
        session.flush()
        page, ctx = read(setup, svc, session, 'list_versions', source_run_id=empty.id)
        assert not page.records and page.owner.run_id == empty.id
        empty.project_id = setup.base.project
        session.flush()
        with pytest.raises(AssistantAccessDenied):
            svc.reauthorize_context(session, ctx)


def test_page_and_context_require_owner_and_provenance(setup):
    with pytest.raises(ValidationError, match='owner scope'):
        RunsReadPage(kind='list_runs', owner=InternalReadDependency(kind='project', project_id=setup.base.project),
            records=(InternalRunRow(run_id=setup.donor.run, project_id=setup.donor.project,status='done'),))
    svc = service(setup)
    with setup.factory() as session:
        original = context(setup, svc, session)
        page, enriched = read(setup, svc, session, 'list_projects', ctx=original)
    obs = AgentObservation(kind='internal_read', action_index=0, internal_read=page)
    with pytest.raises(ValidationError, match='provenance'):
        AgentStepContext(context=original, step_index=1, observations=(obs,))
    assert AgentStepContext(context=enriched, step_index=1, observations=(obs,))
    with pytest.raises(TurnConflict, match='metadata provenance'):
        svc._enrichment_binding(enriched, original)
    assert 'internal_read_dependencies' not in original.model_dump(mode='json')
    assert 'internal_read' not in AgentObservation(kind='failure',action_index=0,error='unavailable').model_dump(mode='json')


def test_revocation_rechecked_after_query(setup, monkeypatch):
    import app.assistant_internal_api as internal
    svc = service(setup)
    original = internal._page
    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        svc.policy = ProjectAllowlist([setup.base.project])
        return result
    monkeypatch.setattr(internal, '_page', revoke)
    with setup.factory() as session, pytest.raises(AssistantAccessDenied):
        read(setup, svc, session, 'list_runs', project_id=setup.donor.project)


def test_unavailable_assets_never_import(setup,monkeypatch):
    import app.assistant_internal_api as internal
    svc = service(setup)
    original = internal.verify_artifact
    def unavailable(artifact,**kwargs):
        if artifact.id == setup.donor.artifact:
            raise AssistantAccessDenied('assistant access denied')
        return original(artifact,**kwargs)
    monkeypatch.setattr(internal,'verify_artifact',unavailable)
    with setup.factory() as session:
        count = session.scalar(select(func.count()).select_from(AssistantArtifact))
        page, _ = read(setup,svc,session,'list_assets',source_version_id=setup.donor.version)
        assert not page.records and page.owner.kind == 'version'
        assert session.scalar(select(func.count()).select_from(AssistantArtifact)) == count


def test_metadata_revocation_across_three_generation_dialogue(setup):
    svc = service(setup,agent_enabled=True)
    with setup.factory.begin() as session:
        for index in range(3):
            turn = svc.accept(session,run_id=setup.base.run,base_version_id=setup.base.version,
                instruction=f'Generation {index}',idempotency_key=f'chain-{index}',actor='operator',now=NOW)
            lease = claim_turn(session,turn.id,owner='test',now=NOW)
            ctx = svc.load_context(session,lease,now=NOW)
            if index == 0:
                _,ctx = read(setup,svc,session,'list_runs',ctx=ctx,project_id=setup.donor.project)
            svc.save_agent_context(session,lease,ctx,now=NOW)
            if index < 2:
                svc.finish(session,lease,ClarifyReply(kind='clarify',run_id=setup.base.run,
                    base_version_id=setup.base.version,explanation='Read inventory',question='Which result?'),now=NOW)
        assert ctx.internal_read_dependencies == ()
        svc.policy = ProjectAllowlist([setup.base.project])
        with pytest.raises(AssistantAccessDenied):
            svc.reauthorize_context(session,ctx)


@pytest.mark.parametrize('terminal',['failed','cancelled'])
def test_same_step_metadata_hidden_after_revocation_for_terminal_turn(setup,terminal):
    from app.assistant_agent_steps import begin_step,save_action,complete_step
    svc = service(setup,agent_enabled=True)
    with setup.factory.begin() as session:
        turn = svc.accept(session,run_id=setup.base.run,base_version_id=setup.base.version,
            instruction='List inventory',idempotency_key='terminal',actor='operator',now=NOW)
        lease = claim_turn(session,turn.id,owner='test',now=NOW)
        ctx = svc.load_context(session,lease,now=NOW)
        step_id,_ = begin_step(session,lease,AgentStepContext(context=ctx,step_index=0),now=NOW)
        save_action(session,lease,step_id,action(setup,'list_runs',project_id=setup.donor.project),now=NOW)
        page,_ = read(setup,svc,session,'list_runs',ctx=ctx,project_id=setup.donor.project)
        complete_step(session,lease,step_id,observation=AgentObservation(kind='internal_read',action_index=0,internal_read=page),now=NOW)
        if terminal == 'failed':
            fail_turn(session,lease,error_code='fixture',now=NOW)
        else:
            request_cancel(session,turn.id,now=NOW)
            finalize_cancel(session,turn.id,lease=lease,now=NOW)
        assert svc.agent_public_state(session,turn)['steps'][0]['observation']['internal_read']
        svc.policy = ProjectAllowlist([setup.base.project])
        with pytest.raises(AssistantAccessDenied):
            svc.agent_public_state(session,turn)


def test_twenty_candidate_rows_per_call_and_ordered_cursor(setup):
    svc = service(setup)
    with setup.factory.begin() as session:
        for index in range(25):
            session.add(Run(id=str(UUID(int=index+1000)), project_id=setup.donor.project))
        session.flush()
        page, ctx = read(setup,svc,session,'list_runs',project_id=setup.donor.project)
        assert len(page.records) == 20 and page.next_after_id == page.records[-1].run_id
        following, _ = read(setup,svc,session,'list_runs',ctx=ctx,project_id=setup.donor.project,after_run_id=page.next_after_id)
        assert len(following.records) == 6 and following.next_after_id is None
        assert not {row.run_id for row in page.records} & {row.run_id for row in following.records}


def test_four_read_budget_is_durable_before_fifth_tool_executes(setup):
    from tests.test_assistant_agent_loop import fixture_loop
    from app.assistant_agent_steps import AssistantAgentStep
    from app.assistant_turns import AssistantTurn
    svc,turn_id,broker,media,supervisor = fixture_loop(setup)
    broker.actions = [action(setup,'list_projects'), action(setup,'list_runs',project_id=setup.donor.project),
        action(setup,'list_versions',source_run_id=setup.donor.run),action(setup,'list_assets',source_version_id=setup.donor.version),
        action(setup,'list_runs',project_id=setup.base.project)]
    supervisor.run_once()
    with setup.factory() as session:
        turn = session.get(AssistantTurn,turn_id)
        assert turn.status == 'failed' and turn.error_code == 'agent_budget_exhausted'
        rows = session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == turn_id).order_by(AssistantAgentStep.step_index)).all()
        assert len(rows) == 5 and all(row.status == 'completed' for row in rows[:4])
        assert rows[-1].action is None and rows[-1].observation is None
        assert [row.observation['internal_read']['kind'] for row in rows[:4]] == ['list_projects','list_runs','list_versions','list_assets']
        state = svc.agent_public_state(session,turn)
        assert state['steps'][3]['observation']['internal_read']['records']
    supervisor.run_once()
    assert len(broker.submitted) == 5 and media.renders == 0


def test_inventory_success_context_and_public_provenance(setup):
    from tests.test_assistant_agent_loop import fixture_loop
    from app.assistant_turns import AssistantTurn
    svc,turn_id,broker,media,supervisor = fixture_loop(setup)
    broker.actions.insert(0,action(setup,'list_runs',project_id=setup.donor.project))
    supervisor.run_once()
    with setup.factory() as session:
        turn = session.get(AssistantTurn,turn_id)
        assert turn.status == 'succeeded'
        assert svc.effective_context(turn).internal_read_dependencies
        svc.policy = ProjectAllowlist([setup.base.project])
        with pytest.raises(AssistantAccessDenied):
            svc.agent_public_state(session,turn)


def test_metadata_page_and_aggregate_byte_budgets(setup):
    from app.assistant_agent_contracts import AssetsReadPage,InternalAssetRow,internal_read_dependencies
    svc = service(setup)
    owner = InternalReadDependency(kind='version',project_id=setup.donor.project,run_id=setup.donor.run,version_id=setup.donor.version)
    def page(index, oversized=False):
        records = tuple(InternalAssetRow(asset_id=str(UUID(int=1000+index*20+offset)),project_id=setup.donor.project,
            run_id=setup.donor.run,version_id=setup.donor.version,media_kind='video',sha256='a'*64,
            size_bytes=9223372036854775807,duration_sec=33333.33333333333 if oversized else 86400.,has_audio=True)
            for offset in range(20))
        return AssetsReadPage(kind='list_assets',owner=owner,records=records,next_after_id=records[-1].asset_id)
    with pytest.raises(ValidationError,match='byte budget'):
        page(0,oversized=True)
    pages = tuple(page(index) for index in range(4))
    with setup.factory() as session:
        ctx = context(setup,svc,session)
    for item in pages:
        ctx = extend_dependencies(ctx,internal_read_dependencies(item))
    observations = tuple(AgentObservation(kind='internal_read',action_index=index,internal_read=item) for index,item in enumerate(pages))
    with pytest.raises(ValidationError,match='aggregate byte budget'):
        AgentStepContext(context=ctx,step_index=4,observations=observations)


def test_metadata_history_authorization_has_aggregate_bound(setup,monkeypatch):
    import app.assistant_turns as turns
    svc = service(setup)
    with setup.factory() as session:
        _,ctx = read(setup,svc,session,'list_projects')
        assert len(ctx.internal_read_dependencies) == 2
        monkeypatch.setattr(turns,'_MAX_HISTORY_DEPENDENCY_INTERNAL_READS',1)
        with pytest.raises(AssistantAccessDenied,match='history access denied'):
            svc.reauthorize_context(session,ctx)


@pytest.mark.parametrize('revocation',['policy','run_owner'])
def test_metadata_asset_history_reauthorizes_owners_without_hashing_inventory(setup,monkeypatch,revocation):
    import app.assistant_internal_api as internal
    import app.assistant_turns as turns
    svc = service(setup)
    with setup.factory.begin() as session:
        page,ctx = read(setup,svc,session,'list_assets',source_version_id=setup.donor.version)
        assert page.records and any(item.asset_id == setup.donor.artifact for item in ctx.internal_read_dependencies)
        metadata_byte_checks = []
        selected_byte_checks = []
        def forbidden_metadata_hash(artifact,**kwargs):
            metadata_byte_checks.append(artifact.id)
            raise AssertionError('metadata provenance cannot rehash inventory')
        original = turns.verify_artifact
        def track_selected(artifact,**kwargs):
            selected_byte_checks.append(artifact.id)
            return original(artifact,**kwargs)
        monkeypatch.setattr(internal,'verify_artifact',forbidden_metadata_hash)
        monkeypatch.setattr(turns,'verify_artifact',track_selected)
        svc.reauthorize_context(session,ctx)
        assert metadata_byte_checks == []
        assert selected_byte_checks and set(selected_byte_checks) == {setup.base.artifact}
        if revocation == 'policy':
            svc.policy = ProjectAllowlist([setup.base.project])
        else:
            session.get(Run,setup.donor.run).project_id = setup.base.project
            session.flush()
        with pytest.raises(AssistantAccessDenied):
            svc.reauthorize_context(session,ctx)
        assert metadata_byte_checks == []
