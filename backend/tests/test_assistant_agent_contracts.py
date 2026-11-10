"""Offline invariants for multimodal proposals; never authenticates a runtime."""
import base64
import hashlib
import io
import json

import pytest
from PIL import Image
from pydantic import ValidationError

from app.assistant_agent_contracts import (
    AgentActionEnvelope, AgentObservation, AgentStepContext, CandidateEvidence,
    FrameEvidence, plan_fingerprint, validate_agent_action,
)
from app.assistant_broker_client import BrokerResult, context_fingerprint
from app import assistant_planner as planner
from tests.test_assistant_planner import context, reply, fake, RUN, BASE, ASSET, OTHER


def frame(**overrides):
    buffer = io.BytesIO()
    Image.new('RGB', (12, 8), 'red').save(buffer, format='JPEG')
    raw = buffer.getvalue()
    return FrameEvidence.model_validate(dict(frame_id=OTHER, asset_id=ASSET, version_id=BASE,
        timestamp_sec=1, width=12, height=8, source_sha256='a' * 64, sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw), data_base64=base64.b64encode(raw).decode(), **overrides))


def step_with_frame():
    return AgentStepContext(context=context(), step_index=1,
        observations=(AgentObservation(kind='frames', action_index=0, frames=(frame(),)),))


def action(kind='inspect_frames', **extra):
    return AgentActionEnvelope.model_validate({'action': dict(kind=kind,run_id=RUN,
        base_version_id=BASE,**extra)}).action


def test_legacy_fingerprint_and_omissions_unchanged():
    ctx = context()
    expected = hashlib.sha256(json.dumps(ctx.model_dump(mode='json'), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert context_fingerprint(ctx) == expected
    result = BrokerResult(turn_id=OTHER, context_sha256=expected, state='succeeded',
        reply=planner.ClarifyReply.model_validate(reply()))
    assert 'action' not in result.model_dump(mode='json')


def test_inline_content_is_image_block_and_metadata_contains_no_base64():
    step = step_with_frame()
    original=step.model_dump(mode='json')
    wire = json.loads(planner._agent_input(step))
    blocks = wire['message']['content']
    assert wire['type'] == 'user'
    assert blocks[-1]['type'] == 'image'
    assert blocks[-1]['source']['data'] == step.observations[0].frames[0].data_base64
    assert 'data_base64' not in blocks[0]['text']
    assert 'data_base64' not in blocks[1]['text']
    assert blocks[1]['text']=='Frame evidence '+step.observations[0].frames[0].frame_id
    assert json.loads(blocks[0]['text'])['observations'][0]['frames'][0]=={
        key:value for key,value in original['observations'][0]['frames'][0].items() if key!='data_base64'}
    assert step.model_dump(mode='json')==original
    assert context_fingerprint(step.context, step) != context_fingerprint(step.context)


def test_jpeg_structure_and_bounded_dimensions_required_even_when_digest_matches():
    raw = b'\xff\xd8\xffinvalid-jpeg'
    values = frame().model_dump()
    values.update(data_base64=base64.b64encode(raw).decode(),size_bytes=len(raw),
                  sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValidationError):
        FrameEvidence.model_validate(values)
    buffer = io.BytesIO()
    Image.new('RGB',(769,2)).save(buffer,format='JPEG')
    raw = buffer.getvalue()
    values.update(data_base64=base64.b64encode(raw).decode(),size_bytes=len(raw),
                  sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValidationError):
        FrameEvidence.model_validate(values)


def test_action_cannot_invent_source_or_inspect_outside_timeline():
    step = AgentStepContext(context=context(), step_index=0)
    for payload in [dict(asset_id=OTHER,version_id=BASE,timestamps_sec=[1]),
                    dict(asset_id=ASSET,version_id=BASE,timestamps_sec=[3])]:
        with pytest.raises(ValueError):
            validate_agent_action(action(**payload),step)
    valid = action(asset_id=ASSET,version_id=BASE,timestamps_sec=[1])
    assert validate_agent_action(valid,step) == valid


def test_candidate_proposal_requires_exact_plan_and_candidate_pixels():
    ctx = context()
    plan = planner.ProposeReply.model_validate(reply('propose')).plan
    candidate = CandidateEvidence(candidate_id=OTHER,base_version_id=BASE,plan=plan,
        sha256='b'*64,plan_sha256=plan_fingerprint(plan),size_bytes=100,duration_sec=2,
        has_audio=True,subtitle_status='unknown',mixed_audio_changed=False)
    obs = AgentObservation(kind='candidate',action_index=0,candidate=candidate)
    final = action('final',reply=reply('propose'),candidate_id=OTHER)
    step = AgentStepContext(context=ctx,step_index=1,observations=(obs,))
    with pytest.raises(ValueError,match='frame evidence'):
        validate_agent_action(final,step)
    image = frame().model_dump()
    image.update(frame_id=BASE,asset_id=None,version_id=None,candidate_id=OTHER,
        source_sha256='b'*64,timestamp_sec=1.95)
    visual = AgentObservation(kind='frames',action_index=1,frames=(FrameEvidence.model_validate(image),))
    step = AgentStepContext(context=ctx,step_index=2,observations=(obs,visual))
    assert validate_agent_action(final,step) == final
    changed = reply('propose')
    changed['plan']['operations'][0]['end_sec'] = 1
    with pytest.raises(ValueError,match='exact rendered'):
        validate_agent_action(action('final',reply=changed,candidate_id=OTHER),step)


def test_step_uses_stream_input_and_strict_action_result(fake):
    proposed = dict(kind='inspect_frames',run_id=RUN,base_version_id=BASE,
        asset_id=ASSET,version_id=BASE,timestamps_sec=[1])
    output = {'type':'result','subtype':'success','is_error':False,
              'structured_output':{'action':proposed}}
    body = '''wire=json.loads(sys.stdin.buffer.read())
assert wire['type']=='user'
assert '--input-format' in sys.argv
assert sys.argv[sys.argv.index('--input-format')+1]=='stream-json'
assert sys.argv[sys.argv.index('--output-format')+1]=='stream-json'
assert '--verbose' in sys.argv
assert wire['message']['content'][-1]['type']=='image'
sys.stdout.write(%r)
''' % json.dumps(output)
    config = fake(body)
    result = planner.ClaudePlanner(config).plan_step(step_with_frame())
    assert result.kind == 'inspect_frames'


def test_reject_action_output_errors_and_native_tool_denials():
    step = AgentStepContext(context=context(),step_index=0)
    proposed = action('final',reply=reply()).model_dump(mode='json')
    for extra in [{'is_error':True},{'permission_denials':[{'tool_name':'Read'}]},
                  {'stop_reason':'max_tokens'}]:
        envelope = dict(type='result',subtype='success',is_error=False,
            structured_output={'action':proposed})
        envelope.update(extra)
        with pytest.raises(planner.PlannerError):
            planner._agent_reply(json.dumps(envelope).encode(),step)

from tests.test_assistant_broker import setup_broker, terminal, TURN


def test_broker_action_result_idempotency_and_ack_preserve_tombstone(setup_broker):
    make, _ = setup_broker
    broker, injected = make()
    step = AgentStepContext(context=context(),step_index=0)
    proposed = action('final',reply=reply())
    calls = []
    def plan_step(supplied, *, cancelled):
        calls.append(supplied)
        return proposed
    injected.plan_step = plan_step
    broker.submit(TURN,step.context,agent_step=step)
    result = terminal(broker)
    assert result.state == 'succeeded' and result.reply is None
    assert result.action == proposed
    assert result.context_sha256 == context_fingerprint(step.context,step)
    assert broker.submit(TURN,step.context,agent_step=step).action == proposed
    assert len(calls) == 1
    ack = broker.acknowledge(TURN)
    assert ack.acknowledged and ack.action is None and ack.reply is None
    assert broker.submit(TURN,step.context,agent_step=step).acknowledged


def test_audio_final_requires_union_of_changed_candidate_speech_coverage():
    from app.assistant_contracts import EditPlan
    from app.assistant_agent_contracts import CandidateSpeechEvidence
    ctx = context()
    plan = EditPlan.model_validate({'run_id':RUN,'base_version_id':BASE,'operations':[
        {'kind':'full_track_audio','source':{'asset_id':ASSET,'version_id':BASE}}]})
    candidate = CandidateEvidence(candidate_id=OTHER,base_version_id=BASE,plan=plan,
        sha256='b'*64,plan_sha256=plan_fingerprint(plan),size_bytes=100,duration_sec=3,
        has_audio=True,subtitle_status='unknown',mixed_audio_changed=True)
    image = frame().model_dump()
    image.update(frame_id=BASE,asset_id=None,version_id=None,candidate_id=OTHER,source_sha256='b'*64)
    observations = [AgentObservation(kind='candidate',action_index=0,candidate=candidate),
        AgentObservation(kind='frames',action_index=1,frames=(FrameEvidence.model_validate(image),))]
    final_reply = reply('propose'); final_reply['plan'] = plan.model_dump(mode='json')
    final = action('final',reply=final_reply,candidate_id=OTHER)
    with pytest.raises(ValueError,match='candidate speech evidence'):
        validate_agent_action(final,AgentStepContext(context=ctx,step_index=2,observations=observations))
    def speech(start,end):
        return CandidateSpeechEvidence(candidate_id=OTHER,source_sha256='b'*64,
            source_start_sec=start,source_end_sec=end,source_duration_sec=3,
            language='en',language_probability=1,words=(),coverage_complete=False)
    observations.append(AgentObservation(kind='speech',action_index=2,candidate_speech=(speech(0,1),)))
    with pytest.raises(ValueError,match='candidate speech evidence'):
        validate_agent_action(final,AgentStepContext(context=ctx,step_index=3,observations=observations))
    observations.append(AgentObservation(kind='speech',action_index=3,candidate_speech=(speech(1,3),)))
    assert validate_agent_action(final,AgentStepContext(context=ctx,step_index=4,observations=observations)) == final


def test_visual_coverage_maps_duration_changes_to_output_timeline():
    from app.assistant_contracts import EditPlan
    from app.assistant_agent_contracts import candidate_visual_windows
    ctx = context()
    trim = EditPlan.model_validate(reply('propose')['plan'])
    assert candidate_visual_windows(trim,ctx,2) == ((1.9,2),)
    cut = EditPlan.model_validate({'run_id':RUN,'base_version_id':BASE,
        'operations':[{'kind':'cut','start_sec':1,'end_sec':2}]})
    assert candidate_visual_windows(cut,ctx,2) == ((0.9,1.1),)


def test_stream_parser_rejects_early_terminal_and_native_tools():
    step = AgentStepContext(context=context(),step_index=0)
    result = {'type':'result','subtype':'success','is_error':False,
        'structured_output':{'action':action('final',reply=reply()).model_dump(mode='json')}}
    for events in [[result,result],[{'type':'system','tools':['Read']},result],
                   [{'type':'assistant','message':{'content':[{'type':'tool_use','name':'Bash'}]}},result]]:
        with pytest.raises(planner.PlannerError):
            planner._agent_reply(('\n'.join(json.dumps(event) for event in events)).encode(),step)
    assert planner._agent_reply((json.dumps({'type':'system','tools':['StructuredOutput']})+'\n'+json.dumps({'type':'rate_limit_event','rate_limit_info':{'status':'allowed'}})+'\n'+json.dumps(result)).encode(),step).kind == 'final'


def test_subtitle_move_requires_old_and_new_passages_and_global_style_all_cues():
    from app.assistant_contracts import EditPlan, SubtitleStylePatch
    from app.assistant_agent_contracts import candidate_visual_windows
    ctx = context(editable_subtitle_cue_ids=['cue-1','cue-2'],subtitle_cues=[
        {'cue_id':'cue-1','start_sec':0,'end_sec':1,'text':'First'},
        {'cue_id':'cue-2','start_sec':1,'end_sec':2,'text':'Second'}])
    operation = {'kind':'subtitle_edit','changes':[
        {'action':'update','cue_id':'cue-1','start_sec':2,'end_sec':3,'text':'Moved'}]}
    moved = EditPlan.model_validate({'run_id':RUN,'base_version_id':BASE,'operations':[operation]})
    assert set(candidate_visual_windows(moved,ctx,3)) == {(0,1),(2,3)}
    styled = moved.model_copy(update={'operations': (moved.operations[0].model_copy(update={
        'style': SubtitleStylePatch(bold=True)}),)})
    assert set(candidate_visual_windows(styled,ctx,3)) == {(0,1),(1,2),(2,3)}
    candidate = CandidateEvidence(candidate_id=OTHER,base_version_id=BASE,plan=styled,
        sha256='b'*64,plan_sha256=plan_fingerprint(styled),size_bytes=100,duration_sec=3,
        has_audio=True,subtitle_status='editable_burned',mixed_audio_changed=False)
    final_reply = reply('propose');final_reply['plan'] = styled.model_dump(mode='json')
    final = action('final',reply=final_reply,candidate_id=OTHER)
    def step(timestamps):
        frames=[]
        for frame_id,timestamp in zip((ASSET,BASE,RUN),timestamps):
            values=frame().model_dump()
            values.update(frame_id=frame_id,asset_id=None,version_id=None,candidate_id=OTHER,
                source_sha256='b'*64,timestamp_sec=timestamp)
            frames.append(FrameEvidence.model_validate(values))
        return AgentStepContext(context=ctx,step_index=2,observations=(
            AgentObservation(kind='candidate',action_index=0,candidate=candidate),
            AgentObservation(kind='frames',action_index=1,frames=frames)))
    for timestamps in [(2.5,),(.5,2.5)]:
        with pytest.raises(ValueError,match='each affected visual interval'):
            validate_agent_action(final,step(timestamps))
    assert validate_agent_action(final,step((.5,1.5,2.5))) == final


@pytest.mark.parametrize('audio_extent,windows,accepted',[
    (2.98,[(0,2.98)],True),          # Proven genuine 20 ms AAC tail.
    (2.94,[(0,2.94)],False),         # A 60 ms tail exceeds supported tolerance.
    (2.98,[(0,2.97)],False),         # Missing 10 ms of actual audio is not a tail.
    (2.98,[(0,1),(1.001,2.98)],False), # Internal gaps never inherit tail tolerance.
    (3.0,[(0,2.98)],False),          # Video duration alone proves no short audio tail.
])
def test_audio_tail_tolerance_requires_proven_extent_not_coverage_gaps(audio_extent,windows,accepted):
    from app.assistant_contracts import EditPlan
    from app.assistant_agent_contracts import CandidateSpeechEvidence
    ctx=context()
    plan=EditPlan.model_validate({'run_id':RUN,'base_version_id':BASE,'operations':[
        {'kind':'full_track_audio','source':{'asset_id':ASSET,'version_id':BASE}}]})
    candidate=CandidateEvidence(candidate_id=OTHER,base_version_id=BASE,plan=plan,
        sha256='b'*64,plan_sha256=plan_fingerprint(plan),size_bytes=100,duration_sec=3,
        has_audio=True,subtitle_status='unknown',mixed_audio_changed=True)
    values=frame().model_dump()
    values.update(asset_id=None,version_id=None,candidate_id=OTHER,source_sha256='b'*64)
    observations=[AgentObservation(kind='candidate',action_index=0,candidate=candidate),
        AgentObservation(kind='frames',action_index=1,frames=(FrameEvidence.model_validate(values),))]
    for index,(start,end) in enumerate(windows,2):
        speech=CandidateSpeechEvidence(candidate_id=OTHER,source_sha256='b'*64,
            source_start_sec=start,source_end_sec=end,source_duration_sec=audio_extent,
            language='en',language_probability=1,words=(),coverage_complete=start==0 and end==audio_extent)
        observations.append(AgentObservation(kind='speech',action_index=index,candidate_speech=(speech,)))
    step=AgentStepContext(context=ctx,step_index=len(observations),observations=observations)
    final_reply=reply('propose');final_reply['plan']=plan.model_dump(mode='json')
    final=action('final',reply=final_reply,candidate_id=OTHER)
    if accepted:
        assert validate_agent_action(final,step)==final
    else:
        with pytest.raises(ValueError,match='candidate speech evidence'):
            validate_agent_action(final,step)


def test_source_pagination_preserves_server_cursor_and_legacy_omissions():
    search=action('search_sources',query='original')
    assert 'after_asset_id' not in search.model_dump(mode='json')
    first=AgentObservation(kind='sources',action_index=0)
    assert 'next_after_asset_id' not in first.model_dump(mode='json')
    assert 'limitations' not in first.model_dump(mode='json')
    page=AgentObservation(kind='sources',action_index=0,next_after_asset_id=OTHER,
        limitations=('source_search_partial','original_source_unavailable'))
    step=AgentStepContext(context=context(),step_index=1,observations=(page,))
    next_search=action('search_sources',query='original',after_asset_id=OTHER)
    assert validate_agent_action(next_search,step)==next_search
    with pytest.raises(ValueError,match='previously supplied catalog cursor'):
        validate_agent_action(action('search_sources',query='original',after_asset_id=ASSET),step)
    with pytest.raises(ValidationError):
        AgentObservation(kind='failure',action_index=0,error='unavailable',next_after_asset_id=OTHER)


def _frame_batches(values):
    import uuid
    frames=[]
    for index in range(len(values)):
        item=dict(values[index],frame_id=str(uuid.UUID(int=index+1)))
        frames.append(FrameEvidence.model_validate(item))
    return tuple(AgentObservation(kind='frames',action_index=index//8,frames=frames[index:index+8])
        for index in range(0,len(frames),8))


def test_accumulated_context_admits_32_valid_images_but_rejects_33():
    from app.assistant_agent_contracts import MAX_CONTEXT_FRAMES, MAX_AGENT_INPUT_BYTES
    assert MAX_CONTEXT_FRAMES == 32
    values=frame().model_dump()
    observations=_frame_batches([values]*32)
    step=AgentStepContext(context=context(),step_index=4,observations=observations)
    wire=planner._agent_input(step)
    assert len(wire) < MAX_AGENT_INPUT_BYTES
    assert sum(block['type']=='image' for block in json.loads(wire)['message']['content']) == 32
    blocks=json.loads(wire)['message']['content']
    for index,image in enumerate(f for obs in observations for f in obs.frames):
        assert blocks[2*index+1]['text']=='Frame evidence '+image.frame_id
        assert blocks[2*index+2]['source']['data']==image.data_base64
    with pytest.raises(ValidationError,match='visual evidence exceeds aggregate budget'):
        AgentStepContext(context=context(),step_index=5,observations=_frame_batches([values]*33))
    # The broader context does not expand the independently bounded tool call.
    with pytest.raises(ValidationError):
        action(asset_id=ASSET,version_id=BASE,timestamps_sec=[index/10 for index in range(9)])


def test_32_frame_context_still_rejects_more_than_2mib_decoded_images():
    from app.assistant_agent_contracts import MAX_IMAGE_BYTES, MAX_FRAME_BYTES
    # Dense real pixels ensure the byte budget, rather than frame count or
    # malformed image rejection, is the reason this request fails.
    raw_pixels=hashlib.shake_256(b'bounded-context-fixture').digest(768*768*3)
    buffer=io.BytesIO()
    Image.frombytes('RGB',(768,768),raw_pixels).save(buffer,format='JPEG',quality=80)
    raw=buffer.getvalue()
    assert len(raw) <= MAX_FRAME_BYTES
    assert len(raw)*6 > MAX_IMAGE_BYTES
    values=frame().model_dump()
    values.update(width=768,height=768,sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),data_base64=base64.b64encode(raw).decode())
    observations=_frame_batches([values]*6)
    with pytest.raises(ValidationError,match='visual evidence exceeds aggregate budget'):
        AgentStepContext(context=context(),step_index=1,observations=observations)


def test_actual_multimodal_text_still_has_independent_64kib_bound():
    from app.assistant_agent_contracts import MAX_AGENT_TEXT_BYTES
    observations=_frame_batches([frame().model_dump()]*32)
    dialogue=[{'role':'operator','text':'x'} for _ in range(16)]
    step=AgentStepContext(context=context(dialogue=dialogue),step_index=4,observations=observations)
    metadata=step.model_dump(mode='json')
    for observation in metadata['observations']:
        for item in observation['frames']:
            item.pop('data_base64')
    spare=MAX_AGENT_TEXT_BYTES-128-len(json.dumps(metadata,ensure_ascii=False,allow_nan=False).encode())
    assert 0 < spare < 3999*len(dialogue)
    for entry in dialogue:
        extra=min(spare,3999)
        entry['text']+='x'*extra
        spare-=extra
    assert spare==0
    step=AgentStepContext(context=context(dialogue=dialogue),step_index=4,observations=observations)
    # The metadata fits; short image-binding labels are text too and cannot
    # silently expand the independently bounded model text payload.
    with pytest.raises(ValueError,match='text payload exceeds aggregate budget'):
        planner._agent_input(step)


def test_eight_frames_fit_near_full_metadata_without_duplicate_binding_headers():
    from app.assistant_agent_contracts import MAX_AGENT_TEXT_BYTES
    observations=_frame_batches([frame().model_dump()]*8)
    dialogue=[{'role':'operator','text':'x'} for _ in range(16)]
    def metadata(step):
        value=step.model_dump(mode='json')
        for obs in value['observations']:
            for image in obs['frames']:image.pop('data_base64')
        return value
    seed=AgentStepContext(context=context(dialogue=dialogue),step_index=1,observations=observations)
    spare=MAX_AGENT_TEXT_BYTES-700-len(json.dumps(metadata(seed),ensure_ascii=False,allow_nan=False).encode())
    for item in dialogue:
        extra=min(spare,3999);item['text']+='x'*extra;spare-=extra
    assert spare==0
    step=AgentStepContext(context=context(dialogue=dialogue),step_index=1,observations=observations)
    main=metadata(step)
    main_bytes=len(json.dumps(main,ensure_ascii=False,allow_nan=False).encode())
    duplicated=sum(len(('Frame evidence '+json.dumps(image,separators=(',',':'))).encode())
        for obs in main['observations'] for image in obs['frames'])
    assert main_bytes+duplicated>MAX_AGENT_TEXT_BYTES
    blocks=json.loads(planner._agent_input(step))['message']['content']
    assert sum(len(b['text'].encode()) for b in blocks if b['type']=='text')<=MAX_AGENT_TEXT_BYTES
    assert sum(b['type']=='image' for b in blocks)==8
