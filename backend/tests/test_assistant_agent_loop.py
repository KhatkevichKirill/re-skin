"""Durable bounded tool loop with offline broker/media fakes."""
import base64
import hashlib
import uuid
import pytest
from sqlalchemy import select
from tests.test_assistant_turns import setup, service, reply, NOW
from app.assistant_agent_contracts import CandidateEvidence, FinalAction, RenderCandidateAction, InspectFramesAction, plan_fingerprint
from app.assistant_agent_steps import AssistantAgentStep, restore_step
from app.assistant_broker_client import BrokerResult, BrokerError, context_fingerprint
from app.assistant_supervisor import AssistantSupervisor
from app.assistant_turns import AssistantTurn

from io import BytesIO
from PIL import Image
_buffer = BytesIO()
Image.new('RGB', (4,4), 'red').save(_buffer, format='JPEG')
JPEG = _buffer.getvalue()
DIGEST = hashlib.sha256(JPEG).hexdigest()

class Broker:
    def __init__(self, actions, unknown=False):
        self.actions, self.unknown = list(actions), unknown
        self.submitted, self.acked, self.cancelled, self.rows = [], [], [], {}
    def submit(self, step_id, context, *, agent_step):
        self.submitted.append(step_id)
        self.rows[step_id] = (context_fingerprint(context, agent_step=agent_step), self.actions.pop(0))
        if self.unknown:
            raise BrokerError('unavailable')
        return self.poll(step_id)
    def poll(self, step_id):
        digest, action = self.rows[step_id]
        return BrokerResult(turn_id=step_id, context_sha256=digest, state='succeeded', action=action)
    def acknowledge(self, step_id):
        self.acked.append(step_id)
        return BrokerResult(turn_id=step_id, context_sha256=self.rows[step_id][0], state='succeeded', acknowledged=True)
    def cancel(self, step_id):
        self.cancelled.append(step_id)
        return BrokerResult(turn_id=step_id, context_sha256=self.rows[step_id][0], state='cancelled')

class Media:
    def __init__(self, candidate):
        self.candidate = candidate
        self.renders = 0
    def render_candidate(self, plan):
        self.renders += 1
        return {'candidate': self.candidate.model_dump(mode='json')}
    def inspect_frames(self, **kwargs):
        return {'frames': [{'frame_id': str(uuid.uuid4()), 'candidate_id': kwargs.get('candidate_id'),
            'asset_id':kwargs.get('asset_id'),'version_id':kwargs.get('version_id'),'timestamp_sec':kwargs['timestamps_sec'][0],
            'width':4,'height':4,'source_sha256':self.candidate.sha256,'sha256':DIGEST,'size_bytes':len(JPEG),'mime_type':'image/jpeg'}]}
    def read_frame(self, frame_id):
        return JPEG
    def persist_candidate(self, candidate_id):
        return {'storage_key':'objects/fixture', 'sha256':self.candidate.sha256,'size_bytes':42}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def fixture_loop(setup, *, unknown=False, repeat=False):
    svc = service(setup, agent_enabled=True)
    with setup.factory.begin() as session:
        turn = svc.accept(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction='Trim', idempotency_key='agent', actor='operator', now=NOW)
        turn_id = turn.id
    proposal = reply(setup)
    scope = {'run_id':setup.base.run,'base_version_id':setup.base.version}
    candidate = CandidateEvidence(candidate_id=str(uuid.uuid4()), base_version_id=setup.base.version,
        plan=proposal.plan, plan_sha256=plan_fingerprint(proposal.plan), sha256='a'*64,size_bytes=42,
        duration_sec=2.,has_audio=True,subtitle_status='absent',mixed_audio_changed=False)
    render = RenderCandidateAction(kind='render_candidate',plan=proposal.plan,**scope)
    initial = InspectFramesAction(kind='inspect_frames',asset_id=setup.base.artifact,version_id=setup.base.version,timestamps_sec=(.5,),**scope)
    actions = [initial, render, render] if repeat else [initial, render,
        InspectFramesAction(kind='inspect_frames',candidate_id=candidate.candidate_id,timestamps_sec=(1.95,),**scope),
        FinalAction(kind='final',reply=proposal,candidate_id=candidate.candidate_id,**scope)]
    broker, media = Broker(actions, unknown=unknown), Media(candidate)
    supervisor = AssistantSupervisor(setup.factory,service=svc,broker=broker,owner='agent-test',
        execute=lambda *a,**k:None,clock=lambda:NOW,sleep=lambda _:None,
        media_factory=lambda **kwargs:media)
    return svc, turn_id, broker, media, supervisor


def test_durable_loop_exact_candidate_and_private_retention(setup):
    svc, turn_id, broker, media, supervisor = fixture_loop(setup)
    supervisor.run_once()
    with setup.factory() as session:
        turn = session.get(AssistantTurn,turn_id)
        assert turn.status == 'succeeded'
        assert turn.workflow_revision == 2
        rows = session.scalars(select(AssistantAgentStep).order_by(AssistantAgentStep.step_index)).all()
        assert len(rows)==4 and len(set(broker.submitted))==4
        assert all(row.id != turn_id and row.status=='completed' and not row.cleanup_pending for row in rows)
        assert rows[-1].private_artifact['candidate_id'] == media.candidate.candidate_id
        frame = rows[-1].context_snapshot['observations'][2]['frames'][0]
        assert 'data_base64' not in frame
        restored = restore_step(session,rows[-1])
        assert restored.observations[2].frames[0].data_base64 == base64.b64encode(JPEG).decode()
        state=svc.agent_public_state(session,turn)
        assert state['verification']['operator_review_required'] is True
        assert 'data_base64' not in str(state)
        assert 'storage_key' not in str(state)
    supervisor.run_once()
    assert len(broker.submitted)==4 and media.renders==1


def test_unknown_step_submission_fails_without_replay(setup):
    _, turn_id, broker, media, supervisor = fixture_loop(setup,unknown=True)
    supervisor.run_once()
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).status=='failed'
    assert len(broker.submitted)==1 and media.renders==0
    assert broker.cancelled == broker.submitted


def test_repeated_action_is_not_executed_again(setup):
    _, turn_id, broker, media, supervisor = fixture_loop(setup,repeat=True)
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).status=='failed'
    assert len(broker.submitted)==3 and media.renders==1


def test_revocation_between_steps_prevents_render_and_public_evidence(setup):
    from app.assistant_access import ProjectAllowlist, AssistantAccessDenied
    import pytest
    svc, turn_id, broker, media, supervisor = fixture_loop(setup)
    submit = broker.submit
    def revoke(step_id, context, *, agent_step):
        result = submit(step_id, context, agent_step=agent_step)
        if agent_step.step_index == 1:
            svc.policy = ProjectAllowlist([])
        return result
    broker.submit = revoke
    supervisor.run_once()
    with setup.factory() as session:
        turn = session.get(AssistantTurn,turn_id)
        assert turn.status=='failed' and media.renders==0
        with pytest.raises(AssistantAccessDenied):
            svc.agent_public_state(session,turn)


def test_recovery_fails_expired_step_without_resubmission(setup):
    from datetime import timedelta
    from app.assistant_turns import claim_turn
    from app.assistant_agent_steps import begin_step
    from app.assistant_agent_contracts import AgentStepContext
    svc, turn_id, broker, media, supervisor = fixture_loop(setup)
    with setup.factory.begin() as session:
        lease=claim_turn(session,turn_id,owner='old',now=NOW-timedelta(seconds=40),lease_sec=30)
        context=svc.load_context(session,lease,now=NOW-timedelta(seconds=40))
        step=AgentStepContext(context=context,step_index=0)
        step_id,digest=begin_step(session,lease,step,now=NOW-timedelta(seconds=40))
    broker.rows[step_id]=(digest,broker.actions[0])
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).error_code=='interrupted'
        assert not session.get(AssistantAgentStep,step_id).cleanup_pending
    assert broker.submitted==[] and media.renders==0 and broker.acked==[step_id]


def test_source_frames_required_before_candidate_render(setup):
    svc, turn_id, broker, media, supervisor = fixture_loop(setup)
    initial, render, candidate_frames, final = broker.actions
    submit = broker.submit
    def respond_to_feedback(step_id, context, *, agent_step):
        index = agent_step.step_index
        if index == 0:
            next_action = render
        elif index == 1:
            assert agent_step.observations[-1].error == 'source_frames_required'
            assert media.renders == 0
            next_action = initial
        elif index == 2:
            assert agent_step.observations[-1].frames[0].asset_id == setup.base.artifact
            next_action = render
        elif index == 3:
            assert agent_step.observations[-1].candidate is not None
            next_action = candidate_frames
        else:
            assert index == 4 and agent_step.observations[-1].frames[0].candidate_id == media.candidate.candidate_id
            next_action = final
        broker.actions = [next_action]
        return submit(step_id,context,agent_step=agent_step)
    broker.submit = respond_to_feedback
    supervisor.run_once()
    with setup.factory() as session:
        turn=session.get(AssistantTurn,turn_id)
        assert turn.status=='succeeded'
        first=session.scalar(select(AssistantAgentStep).where(AssistantAgentStep.step_index==0))
        assert first.observation['error']=='source_frames_required'
        assert svc.agent_public_state(session,turn)['steps'][0]['observation']['error']=='source_frames_required'
    assert media.renders==1 and len(broker.submitted)==5


@pytest.mark.parametrize('field',['run_id','base_version_id'])
def test_source_frame_feedback_does_not_relax_pinned_render_scope(setup,field):
    _,turn_id,broker,media,supervisor = fixture_loop(setup)
    other = setup.donor.run if field == 'run_id' else setup.donor.version
    broker.actions = [broker.actions[1].model_copy(update={field:other})]
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).status == 'failed'
        first=session.scalar(select(AssistantAgentStep).where(AssistantAgentStep.step_index==0))
        assert first.action is None and first.observation is None
    assert media.renders == 0 and len(broker.submitted) == 1


def test_refused_render_attempts_retain_existing_three_attempt_budget(setup):
    _,turn_id,broker,media,supervisor = fixture_loop(setup)
    broker.actions = [broker.actions[1]]*4
    supervisor.run_once()
    with setup.factory() as session:
        turn=session.get(AssistantTurn,turn_id)
        assert turn.status == 'failed' and turn.error_code == 'agent_budget_exhausted'
        rows=session.scalars(select(AssistantAgentStep).order_by(AssistantAgentStep.step_index)).all()
        assert len(rows)==4 and rows[-1].action is None
        assert [row.observation['error'] for row in rows[:3]] == ['source_frames_required']*3
    assert media.renders == 0 and len(broker.submitted) == 4


def test_capability_unavailable_observation_allows_alternative(setup):
    from app.assistant_agent_media import AgentToolUnavailable
    from app.assistant_agent_contracts import TranscribeWindowAction
    svc, turn_id, broker, media, supervisor=fixture_loop(setup)
    broker.actions.insert(0,TranscribeWindowAction(kind='transcribe_window',run_id=setup.base.run,
        base_version_id=setup.base.version,asset_id=setup.base.artifact,version_id=setup.base.version,start_sec=0.,end_sec=2.))
    def unavailable(**kwargs):
        raise AgentToolUnavailable('offline speech unavailable')
    media.transcribe_window=unavailable
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).status=='succeeded'
        first=session.scalar(select(AssistantAgentStep).where(AssistantAgentStep.step_index==0))
        assert first.observation['kind']=='failure' and first.observation['error']=='unavailable'
    assert media.renders==1


def test_cancellation_during_step_exits_without_render(setup):
    from app.assistant_turns import request_cancel
    svc, turn_id, broker, media, supervisor=fixture_loop(setup)
    submit=broker.submit
    def cancel(step_id,context,*,agent_step):
        result=submit(step_id,context,agent_step=agent_step)
        with setup.factory.begin() as session:
            request_cancel(session,turn_id,now=NOW)
        return result
    broker.submit=cancel
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).status=='cancelled'
    assert media.renders==0 and len(broker.submitted)==1



def test_alternate_valid_retained_blob_is_never_published(setup):
    _, turn_id, broker, media, supervisor = fixture_loop(setup)
    from dataclasses import asdict
    alternate = setup.storage.pin(setup.donor.path)
    setup.storage.verify(alternate)  # A valid immutable blob, belonging to another artifact.
    media.persist_candidate=lambda candidate_id: asdict(alternate)
    supervisor.run_once()
    with setup.factory() as session:
        turn=session.get(AssistantTurn,turn_id)
        assert turn.status=='failed' and turn.reply is None and turn.assistant_message_id is None
        rows=session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id==turn_id)).all()
        assert all(row.private_artifact is None for row in rows)


def test_source_discovery_cursor_survives_filtered_empty_page(setup):
    from app.assistant_agent_contracts import SearchSourcesAction
    svc, turn_id, broker, media, supervisor=fixture_loop(setup)
    with setup.factory() as session:
        context=svc.build_context(session,run_id=setup.base.run,base_version_id=setup.base.version,
            instruction='Find sources',source_refs=((setup.donor.artifact,setup.donor.version),))
        donor=context.sources[0]
    calls=[]
    def search_sources(*,query,after_asset_id=None):
        calls.append(after_asset_id)
        if after_asset_id is None:
            return {'sources':[],'next_after_asset_id':setup.donor.artifact}
        assert after_asset_id==setup.donor.artifact
        return {'sources':[donor.model_dump(mode='json')],'next_after_asset_id':None}
    media.search_sources=search_sources
    scope={'run_id':setup.base.run,'base_version_id':setup.base.version}
    broker.actions[:0]=[SearchSourcesAction(kind='search_sources',query='video',**scope),
        SearchSourcesAction(kind='search_sources',query='video',after_asset_id=setup.donor.artifact,**scope)]
    supervisor.run_once()
    with setup.factory() as session:
        turn=session.get(AssistantTurn,turn_id)
        assert turn.status=='succeeded'
        rows=session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id==turn_id)
            .order_by(AssistantAgentStep.step_index)).all()
        assert rows[0].observation['next_after_asset_id']==setup.donor.artifact
        assert 'source_search_partial' in rows[0].observation['limitations']
        assert rows[1].action['after_asset_id']==setup.donor.artifact
        assert rows[1].observation['sources'][0]['asset_id']==setup.donor.artifact
    assert calls==[None,setup.donor.artifact]


def test_balanced_temporal_memory_retains_both_sources_then_candidate(monkeypatch):
    # Exercise constrained selection independently of production's 32-frame view.
    import app.assistant_agent_contracts as contracts
    monkeypatch.setattr(contracts, 'MAX_CONTEXT_FRAMES', 8)
    from app.assistant_agent_contracts import AgentObservation, FrameEvidence
    from app.assistant_agent_loop import select_evidence
    target_version, original_version = str(uuid.uuid4()), str(uuid.uuid4())
    def source_frames(asset_id,version_id,index):
        return AgentObservation(kind='frames',action_index=index,frames=tuple(FrameEvidence(
            frame_id=str(uuid.uuid4()),asset_id=asset_id,version_id=version_id,timestamp_sec=time,
            source_sha256='a'*64,sha256=DIGEST,size_bytes=len(JPEG),width=4,height=4,
            data_base64=base64.b64encode(JPEG).decode())
            for time in (6.8,6.9,7.,7.1,24.8,24.9,25.,25.1)))
    target,original=str(uuid.uuid4()),str(uuid.uuid4())
    first=source_frames(target,target_version,0)
    second=source_frames(original,original_version,1)
    all_fresh=select_evidence((first,second))
    assert len(all_fresh)==1 and all_fresh[0].frames==second.frames
    second=second.model_copy(update={'frames':tuple(second.frames[index] for index in (0,3,4,7))})
    selected=select_evidence((first,second))
    assert sum(len(observation.frames) for observation in selected)==8
    for asset in (target,original):
        times=[frame.timestamp_sec for observation in selected for frame in observation.frames if frame.asset_id==asset]
        assert len(times)==4 and min(times)==6.8 and max(times)==25.1
        assert any(6.5<=time<=7.5 for time in times) and any(24.5<=time<=25.5 for time in times)
    candidate_id=str(uuid.uuid4())
    candidate=AgentObservation(kind='frames',action_index=2,frames=tuple(FrameEvidence(
        frame_id=str(uuid.uuid4()),candidate_id=candidate_id,timestamp_sec=time,source_sha256='b'*64,
        sha256=DIGEST,size_bytes=len(JPEG),width=4,height=4,data_base64=base64.b64encode(JPEG).decode())
        for time in (1.,7.,20.,25.)))
    final=select_evidence((*selected,candidate))
    assert len([frame for observation in final for frame in observation.frames if frame.candidate_id==candidate_id])==4
    for asset in (target,original):
        times=[frame.timestamp_sec for observation in final for frame in observation.frames if frame.asset_id==asset]
        assert times==[6.8,25.1]
    assert select_evidence(final)==final



def test_revision_two_does_not_run_legacy_keyword_enrichment(setup):
    _,turn_id,_,media,supervisor=fixture_loop(setup)
    class ForbiddenLegacyEnricher:
        def enrich(self,*args,**kwargs):
            raise AssertionError('revision-two source and ASR calls belong to bounded agent tools')
    supervisor.enricher=ForbiddenLegacyEnricher()
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).status=='succeeded'
        assert session.get(AssistantTurn,turn_id).enrichment_snapshot is None
    assert media.renders==1



def test_fresh_targeted_inspection_survives_historical_temporal_samples():
    from app.assistant_agent_contracts import AgentObservation, FrameEvidence
    from app.assistant_agent_loop import select_evidence
    target,original,version_a,version_b=(str(uuid.uuid4()) for _ in range(4))
    def observation(asset,version,index,times):
        return AgentObservation(kind='frames',action_index=index,frames=tuple(FrameEvidence(
            frame_id=str(uuid.uuid4()),asset_id=asset,version_id=version,timestamp_sec=time,
            source_sha256='a'*64,sha256=DIGEST,size_bytes=len(JPEG),width=4,height=4,
            data_base64=base64.b64encode(JPEG).decode()) for time in times))
    old_target=observation(target,version_a,0,(6.8,7.1,24.8,25.1))
    old_original=observation(original,version_b,1,(6.8,7.1,24.8,25.1))
    fresh=observation(target,version_a,2,(7.,25.))
    chosen=select_evidence((old_target,old_original,fresh))
    frames=[frame for item in chosen for frame in item.frames]
    assert len(frames)==10
    assert {frame.frame_id for frame in fresh.frames}<={frame.frame_id for frame in frames}
    assert {frame.asset_id for frame in frames}=={target,original}
    assert len([frame for frame in frames if frame.asset_id==original])>=2



def test_all_accumulated_comparison_frames_remain_when_bytes_fit():
    from app.assistant_agent_contracts import AgentObservation, FrameEvidence, MAX_CONTEXT_FRAMES
    from app.assistant_agent_loop import select_evidence
    def samples(index):
        asset,version=str(uuid.uuid4()),str(uuid.uuid4())
        return AgentObservation(kind='frames',action_index=index,frames=tuple(FrameEvidence(
            frame_id=str(uuid.uuid4()),asset_id=asset,version_id=version,timestamp_sec=float(time),
            source_sha256='a'*64,sha256=DIGEST,size_bytes=len(JPEG),width=4,height=4,
            data_base64=base64.b64encode(JPEG).decode()) for time in (0,3,7,10,15,20,25,26)))
    first,second,third,fourth=(samples(index) for index in range(4))
    assert MAX_CONTEXT_FRAMES==32
    for history in ((first,second),(first,second,third),(first,second,third,fourth)):
        selected=select_evidence(history)
        assert selected==history
        assert sum(len(item.frames) for item in selected)==8*len(history)


def test_loop_keeps_full_archive_when_model_byte_view_is_compacted(setup,monkeypatch):
    import app.assistant_agent_loop as loop
    import app.assistant_agent_contracts as contracts
    svc,turn_id,broker,media,supervisor=fixture_loop(setup)
    monkeypatch.setattr(contracts,'MAX_IMAGE_BYTES',len(JPEG)*8)
    seen=[]
    selector=loop.select_evidence
    def observe_archive(observations):
        seen.append(sum(len(item.frames) for item in observations))
        return selector(observations)
    monkeypatch.setattr(loop,'select_evidence',observe_archive)
    scope={'run_id':setup.base.run,'base_version_id':setup.base.version}
    inspections=[InspectFramesAction(kind='inspect_frames',asset_id=setup.base.artifact,version_id=setup.base.version,
        timestamps_sec=tuple(round(.3*index+offset,3) for index in range(8)),**scope) for offset in (0.,.05,.1)]
    broker.actions=inspections+broker.actions[1:]
    original_inspect=media.inspect_frames
    def inspect_all(**kwargs):
        records=[]
        for timestamp in kwargs['timestamps_sec']:
            one=original_inspect(**{**kwargs,'timestamps_sec':(timestamp,)})['frames'][0]
            records.append(one)
        return {'frames':records}
    media.inspect_frames=inspect_all
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn,turn_id).status=='succeeded'
        rows=session.scalars(select(AssistantAgentStep).where(AssistantAgentStep.turn_id==turn_id)).all()
        assert sum(len((row.observation or {}).get('frames',[])) for row in rows)==25
    assert max(seen)==25  # Archive survives compaction of each 8-image model view.
    assert 24 in seen


def test_qa_action_is_dispatched_persisted_and_budgeted(setup):
    from app.assistant_agent_contracts import DetectCaptionsAction
    svc, turn_id, broker, media, supervisor=fixture_loop(setup)
    ref = {'asset_id': setup.base.artifact, 'version_id': setup.base.version}
    scope = {'run_id': setup.base.run, 'base_version_id': setup.base.version}
    calls = []
    def detect(**kwargs):
        calls.append(kwargs)
        sha = {**ref, 'sha256': 'b' * 64}
        return {'tool': 'detect_captions', 'target': sha, 'config_sha256': 'c' * 64, 'project_config': False,
            'fps': 10.0, 'threshold': 40, 'frames': 20, 'hits': 0, 'hit_ratio': 0.0, 'verdict': 'clean',
            'spans': [], 'spans_total': 0, 'scores_sha256': 'd' * 64,
            'calibration': {'source': sha, 'frames': 20, 'hits': 15, 'hit_ratio': .75, 'calibrated': True}}
    media.detect_captions = detect
    broker.actions.insert(0, DetectCaptionsAction(kind='detect_captions', target=ref, calibration=ref, **scope))
    supervisor.run_once()
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).status == 'succeeded'
        first = session.scalar(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == turn_id,
            AssistantAgentStep.step_index == 0))
        assert first.observation['kind'] == 'qa' and first.observation['qa']['verdict'] == 'clean'
        later = session.scalar(select(AssistantAgentStep).where(AssistantAgentStep.turn_id == turn_id,
            AssistantAgentStep.step_index == 1))
        # The verdict reaches the next inference as evidence.
        assert restore_step(session, later).observations[0].qa.verdict == 'clean'
    assert calls == [{'target': {**ref, 'candidate_id': None}, 'calibration': {**ref, 'candidate_id': None},
        'start_sec': None, 'duration_sec': None, 'calibration_start_sec': None, 'calibration_duration_sec': None}]


def test_qa_call_budget_ends_the_turn(setup):
    from app.assistant_agent_contracts import CheckAlignmentAction, MAX_QA_CALLS
    svc, turn_id, broker, media, supervisor=fixture_loop(setup)
    ref = {'asset_id': setup.base.artifact, 'version_id': setup.base.version}
    scope = {'run_id': setup.base.run, 'base_version_id': setup.base.version}
    def align(**kwargs):
        sha = {**ref, 'sha256': 'b' * 64}
        return {'tool': 'check_alignment', 'mode': 'timeline', 'source': sha, 'result': sha, 'source_frames': 60,
            'result_frames': 60, 'points': 2, 'nonzero_points': 0, 'outliers_total': 0, 'at_search_limit': 0,
            'max_abs_window_median': 0.0, 'length_ok': True, 'in_sync': True, 'tolerance_frames': 1,
            'detail_sha256': 'e' * 64}
    media.check_alignment = align
    for index in range(MAX_QA_CALLS + 1):
        broker.actions.insert(index, CheckAlignmentAction(kind='check_alignment', source=ref, result=ref,
            duration_sec=float(index + 1) / 10, **scope))
    supervisor.run_once()
    with setup.factory() as session:
        turn = session.get(AssistantTurn, turn_id)
        assert turn.status == 'failed' and turn.error_code == 'agent_budget_exhausted'
