"""Ending contracts/evidence and real shared subtitle rendering; offline only."""
import base64
from dataclasses import replace
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError
from app.assistant_contracts import EditPlan,PlanSourceError,ExtendEndingOperation,validate_local_plan_capability,validate_plan_sources
from app.assistant_agent_contracts import (AgentObservation,AgentStepContext,CandidateEvidence,CandidateSpeechEvidence,
    FrameEvidence,FinalAction,plan_fingerprint,validate_agent_action,ending_source_evidence)
from app.assistant_planner import PlannerSpeechEvidence,ProposeReply,PlannerError,_reply
from tests.test_assistant_turns import setup,service
from tests.test_assistant_agent_loop import JPEG,DIGEST


def operation(setup,**changes):
    return dict(kind='extend_ending',start_sec=2.,end_sec=4.,source={'asset_id':setup.donor.artifact,
        'version_id':setup.donor.version},donor_start_sec=.5,visual={'kind':'black'},**changes)


def plan(setup,**changes):
    values=operation(setup)
    values.update(changes)
    return EditPlan(run_id=setup.base.run,base_version_id=setup.base.version,operations=(values,))


def context(setup):
    with setup.factory() as session:
        return service(setup).build_context(session,run_id=setup.base.run,base_version_id=setup.base.version,
            instruction='Extend ending',source_refs=((setup.donor.artifact,setup.donor.version),))


@pytest.mark.parametrize('changes',[{'start_sec':0.},{'start_sec':3.},{'start_sec':4.},{'end_sec':3.},
    {'end_sec':120.1},{'end_sec':12.1},{'visual':{'kind':'black','timestamp_sec':1.}},
    {'visual':{'kind':'freeze_selected'}},{'visual':{'kind':'freeze_selected','timestamp_sec':2.}},
    {'fade_out_sec':.5},{'end_sec':'4'},{'start_sec':True},{'end_sec':float('nan')}])
def test_ending_invalid_shapes_and_source_bounds_fail_closed(setup,changes):
    from app.assistant_contracts import ending_bounds
    with pytest.raises(ValueError):
        candidate=plan(setup,**changes)
        ending_bounds(candidate.operations[0],3.,'30')


def test_ending_does_not_relax_existing_bounds_or_composition(setup):
    from app.assistant_contracts import AssetManifest,VersionManifest
    base=VersionManifest(version_id=setup.base.version,run_id=setup.base.run,duration_sec=3.,status='succeeded',available=True)
    assets={setup.donor.artifact:AssetManifest(asset_id=setup.donor.artifact,version_id=setup.donor.version,
        media_kind='video',duration_sec=3.,has_audio=True,available=True)}
    validate_plan_sources(plan(setup),base_version=base,assets=assets)
    for kind in ('trim','cut','image_fade'):
        old=EditPlan(run_id=setup.base.run,base_version_id=setup.base.version,operations=[{'kind':kind,'start_sec':2.,'end_sec':4.}])
        with pytest.raises(PlanSourceError):
            validate_plan_sources(old,base_version=base,assets=assets)
    with pytest.raises(ValidationError,match='sole operation'):
        EditPlan(run_id=setup.base.run,base_version_id=setup.base.version,operations=[operation(setup),{'kind':'image_fade','start_sec':1.,'end_sec':2.}])
    raw={'type':'result','subtype':'success','is_error':False,'structured_output':{'reply':ProposeReply(
        kind='propose',run_id=setup.base.run,base_version_id=setup.base.version,explanation='Extend',plan=plan(setup)).model_dump(mode='json')}}
    import json
    with pytest.raises(PlannerError):
        _reply(json.dumps(raw).encode(),context(setup))


def evidence(setup,*,freeze=False):
    import uuid
    ctx=context(setup)
    proposed=plan(setup,visual={'kind':'freeze_selected','timestamp_sec':1.5} if freeze else {'kind':'black'},
        fade_out_sec=.5 if freeze else 0.)
    candidate=CandidateEvidence(candidate_id=str(uuid.uuid4()),base_version_id=setup.base.version,plan=proposed,
        sha256='a'*64,plan_sha256=plan_fingerprint(proposed),size_bytes=42,duration_sec=4.,has_audio=True,
        subtitle_status='unknown',mixed_audio_changed=True)
    def frames(index,times,*,is_candidate=False):
        return AgentObservation(kind='frames',action_index=index,frames=tuple(FrameEvidence(frame_id=str(uuid.uuid4()),
            asset_id=None if is_candidate else setup.base.artifact,version_id=None if is_candidate else setup.base.version,
            candidate_id=candidate.candidate_id if is_candidate else None,timestamp_sec=stamp,width=4,height=4,
            source_sha256='a'*64,sha256=DIGEST,size_bytes=len(JPEG),data_base64=base64.b64encode(JPEG).decode()) for stamp in times))
    source_frames=frames(0,(1.5,1.9666666666666666,2.))
    donor=AgentObservation(kind='speech',action_index=1,speech=(PlannerSpeechEvidence(asset_id=setup.donor.artifact,
        version_id=setup.donor.version,source_start_sec=.5,source_end_sec=2.5,source_duration_sec=3.,language='en',
        language_probability=1.,coverage_complete=False,words=()),))
    rendered=AgentObservation(kind='candidate',action_index=2,candidate=candidate)
    pictures=frames(3,(2.,3.1,3.5,3.966666666666667),is_candidate=True)
    speech=AgentObservation(kind='speech',action_index=4,candidate_speech=(CandidateSpeechEvidence(candidate_id=candidate.candidate_id,
        source_sha256='a'*64,source_start_sec=2.,source_end_sec=4.,source_duration_sec=4.,language='en',
        language_probability=1.,coverage_complete=False,words=()),))
    final=FinalAction(kind='final',run_id=setup.base.run,base_version_id=setup.base.version,candidate_id=candidate.candidate_id,
        reply=ProposeReply(kind='propose',run_id=setup.base.run,base_version_id=setup.base.version,explanation='Extend',plan=proposed))
    return ctx,(source_frames,donor,rendered,pictures,speech),final


@pytest.mark.parametrize('freeze',[False,True])
def test_ending_complete_candidate_evidence_accepts_and_missing_regions_reject(setup,freeze):
    ctx,observations,final=evidence(setup,freeze=freeze)
    step=AgentStepContext(context=ctx,step_index=5,observations=observations)
    assert ending_source_evidence(final.reply.plan,step)
    assert validate_agent_action(final,step)==final
    # A pre-extension picture alone cannot establish the added ending.
    deficient=list(observations)
    deficient[3]=deficient[3].model_copy(update={'frames':(deficient[3].frames[0],)})
    with pytest.raises(ValueError,match='candidate frames'):
        validate_agent_action(final,AgentStepContext(context=ctx,step_index=5,observations=tuple(deficient)))
    deficient=list(observations)
    deficient[4]=deficient[4].model_copy(update={'candidate_speech':(deficient[4].candidate_speech[0].model_copy(update={'source_end_sec':3.5}),)})
    with pytest.raises(ValueError,match='complete changed-window'):
        validate_agent_action(final,AgentStepContext(context=ctx,step_index=5,observations=tuple(deficient)))


@pytest.mark.parametrize('missing',['before','after','freeze','donor_end'])
def test_ending_source_pixels_and_complete_donor_speech_are_required(setup,missing):
    ctx,observations,final=evidence(setup,freeze=True)
    observations=list(observations)
    if missing == 'donor_end':
        observations[1]=observations[1].model_copy(update={'speech':(observations[1].speech[0].model_copy(update={'source_end_sec':2.4}),)})
    else:
        removed={'before':1.9666666666666666,'after':2.,'freeze':1.5}[missing]
        observations[0]=observations[0].model_copy(update={'frames':tuple(frame for frame in observations[0].frames if frame.timestamp_sec != removed)})
    assert not ending_source_evidence(final.reply.plan,AgentStepContext(context=ctx,step_index=5,observations=tuple(observations)))


def test_ending_service_uses_actual_audio_extent_not_longer_video(setup,monkeypatch):
    from app.assistant_service import AssistantAdmissionService,AssistantCapabilityError
    from app.assistant_access import ProjectAllowlist
    from app import assistant_audio
    from app import assistant_ending  # Load its native adapter before installing this service-only probe double.
    svc=AssistantAdmissionService(storage=setup.storage,policy=ProjectAllowlist([setup.base.project,setup.donor.project]),enabled=True)
    metadata=assistant_audio.AudioMetadata(2.499,48000,2,'stereo')
    monkeypatch.setattr(assistant_audio,'probe_trusted_audio',lambda *a,**k:metadata)
    with setup.factory() as session,pytest.raises(AssistantCapabilityError,match='audio'):
        svc._validate(session,plan(setup))
    metadata=replace(metadata,duration_sec=3.)
    with setup.factory() as session:
        assert svc._validate(session,plan(setup))['plan']['operations'][0]['kind']=='extend_ending'


@pytest.mark.parametrize('value',[None,True,False,0,-1,float('inf'),float('nan'),'4',{'secret':'not returned'}])
def test_public_version_duration_rejects_untrusted_shape(setup,monkeypatch,value):
    from app import api_assistant
    from app.assistant_models import AssistantVersion
    from types import SimpleNamespace
    def artifact(*args,role='full',**kwargs):
        return SimpleNamespace(metadata_json={'duration_sec':value} if role=='full' else {})
    monkeypatch.setattr(api_assistant,'get_version_artifact',artifact)
    with setup.factory() as session:
        assert api_assistant._version(session.get(AssistantVersion,setup.base.version),setup.base.run,session)['duration_sec'] is None


@pytest.mark.parametrize('fps,base_frames,start_frame,end_frame,visual_kind',[
    ('30',90,60,120,'freeze_selected'),('24000/1001',72,48,96,'freeze_selected'),
    ('30',90,62,120,'black'),('30',90,62,120,'freeze_selected'),('30',90,62,121,'black')])
def test_shared_ending_renderer_preserves_editable_prefix_lineage(tmp_path,fps,base_frames,start_frame,end_frame,visual_kind):
    from sqlalchemy import create_engine,event
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    from app.models import Run,VideoProject
    from app.assistant_store import (AssistantArtifactStorage,ArtifactInput,import_media_version,reserve_version,publish_version,get_version_artifact)
    from app.assistant_store import verify_artifact
    from app.assistant_service import AssistantAdmissionService
    from app.assistant_access import ProjectAllowlist
    from app.assistant_executor import render_local_candidate
    from app.assistant_subtitles import SubtitleTrack,SubtitleCue,burn_track,get_subtitle_context
    from app.assistant_audio import probe_trusted_audio
    from app import assistant_media as media
    rate=float(Fraction(fps)); duration=base_frames/rate; start=start_frame/rate; end=end_frame/rate
    root=tmp_path/'data';root.mkdir();storage=AssistantArtifactStorage(root)
    source=root/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'color=c=white:s=160x120:r={fps}:d={duration}',
        '-f','lavfi','-i',f'sine=frequency=440:sample_rate=48000:duration={duration}',
        '-frames:v',str(base_frames),'-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-c:a','aac',
        '-movie_timescale','48000',str(source)],check=True)
    track=SubtitleTrack(cues=(SubtitleCue(cue_id='prefix',start_sec=.2,end_sec=.5,text='Keep'),
        SubtitleCue(cue_id='cross',start_sec=1.8,end_sec=2.2,text='Cross'),
        SubtitleCue(cue_id='tail',start_sec=2.4,end_sec=2.8,text='Drop')))
    captioned=root/'captioned.mp4'
    burn_track(source,captioned,track=track,clean_source_verified=True,exact_audio_timing=True)
    captions=root/'captions.json';captions.write_text(track.model_dump_json())
    engine=create_engine(f"sqlite:///{tmp_path/'lineage.db'}")
    @event.listens_for(engine,'connect')
    def foreign_keys(connection,_):connection.execute('PRAGMA foreign_keys=ON')
    Base.metadata.create_all(engine);factory=sessionmaker(engine,expire_on_commit=False)
    with factory.begin() as session:
        project=VideoProject(source_type='upload',source_ref='fixture');session.add(project);session.flush()
        run=Run(project_id=project.id,status='done',result_local_path=str(captioned));session.add(run);session.flush()
        version=import_media_version(session,run,actor='operator',storage=storage,full=ArtifactInput(captioned,
            metadata={'duration_sec':duration,'subtitle_status':'editable_burned'}),clean=ArtifactInput(source,
            metadata={'duration_sec':duration}),captions=ArtifactInput(captions,'subtitle'))
        asset=get_version_artifact(session,version.id)
        visual={'kind':visual_kind}
        if visual_kind=='freeze_selected':visual['timestamp_sec']=30/rate
        proposed=EditPlan(run_id=run.id,base_version_id=version.id,operations=[dict(kind='extend_ending',start_sec=start,end_sec=end,
            donor_start_sec=0.,source={'asset_id':asset.id,'version_id':version.id},visual=visual)])
        class Guard:
            def __call__(self):return False
            def check(self):pass
        work=root/'work';work.mkdir()
        result,clean,retained,fits=render_local_candidate(proposed,captioned,{asset.id:source},work,Guard(),120,track=track,clean_path=source)
        assert result.operation=='extend_ending' and abs(result.metadata.video_duration_sec-end)<=1e-6
        assert [cue.cue_id for cue in retained.cues]==['prefix','cross']
        clock=Fraction(start_frame,1)/Fraction(fps)
        end_ticks=(clock*100).numerator//(clock*100).denominator
        assert retained.cues[-1].end_sec==end_ticks/100 and retained.cues[-1].text=='Cross'
        assert result.base_timeline==(media.TimelineSpan(0,start,0),) and not fits
        assert abs(probe_trusted_audio(result.output_path).duration_sec-result.ending_timing.end_sample/48000)<=1/48000+1e-8
        out=root/'retained.json';out.write_text(retained.model_dump_json())
        child=reserve_version(session,proposed,idempotency_key='ending-fixture',actor='operator')
        publish_version(session,child.id,full=ArtifactInput(result.output_path,metadata={'duration_sec':end,'subtitle_status':'editable_burned'}),
            preview=ArtifactInput(result.output_path,metadata={'duration_sec':end}),
            clean=clean,captions=ArtifactInput(out,'subtitle'),storage=storage)
        assert get_version_artifact(session,child.id,role='clean').metadata_json['exact_audio_timing'] is True
        assert get_subtitle_context(session,child.id,storage=storage)==retained
        subsequent=EditPlan(run_id=run.id,base_version_id=child.id,operations=[{'kind':'subtitle_edit','changes':[
            {'action':'update','cue_id':'cross','start_sec':1.8,'end_sec':retained.cues[-1].end_sec,'text':'Still editable'}]}])
        svc=AssistantAdmissionService(storage=storage,policy=ProjectAllowlist([project.id]),enabled=True)
        assert svc._validate(session,subsequent)['plan']['base_version_id']==child.id
        if fps=='30' and start_frame==62:
            def pixels(path,index):
                value=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-vf',
                    f'select=eq(n\\,{index}),format=gray','-frames:v','1','-threads','1','-f','rawvideo','-'],
                    capture_output=True,check=True,timeout=30).stdout
                assert len(value)==160*120
                return value
            # Verify the last retained picture still has its caption, while the
            # first replacement frame has only the intended black/frozen pixels.
            before=pixels(result.output_path,start_frame-1)
            assert max(before)-min(before)>40
            def assert_clean_first_tail(path):
                image=pixels(path,start_frame)
                if visual_kind=='black':assert max(image)<=32
                else:assert min(image)>=220 and max(image)-min(image)<=8
            assert_clean_first_tail(result.output_path)
            ordinary_child=EditPlan(run_id=run.id,base_version_id=child.id,
                operations=[{'kind':'subtitle_edit','style':{'bold':False}}])
            child_work=root/'ordinary-child';child_work.mkdir()
            child_clean_artifact=get_version_artifact(session,child.id,role='clean')
            child_result,child_clean,child_track,_=render_local_candidate(ordinary_child,
                verify_artifact(get_version_artifact(session,child.id),storage=storage),{},child_work,Guard(),120,
                track=get_subtitle_context(session,child.id,storage=storage),
                clean_path=verify_artifact(child_clean_artifact,storage=storage),
                exact_audio_timing=child_clean_artifact.metadata_json.get('exact_audio_timing',False))
            assert child_track.cues[-1].end_sec==2.06
            assert_clean_first_tail(child_result.output_path)
            assert abs(probe_trusted_audio(child_result.output_path).duration_sec-result.ending_timing.end_sample/48000)<=1/48000+1e-8
            child_captions=root/'child-captions.json';child_captions.write_text(child_track.model_dump_json())
            second=reserve_version(session,ordinary_child,idempotency_key='ordinary-caption-child',actor='operator')
            publish_version(session,second.id,full=ArtifactInput(child_result.output_path,
                metadata={'duration_sec':end,'subtitle_status':'editable_burned'}),
                preview=ArtifactInput(child_result.output_path,metadata={'duration_sec':end}),
                clean=child_clean,captions=ArtifactInput(child_captions,'subtitle'),storage=storage)
            assert get_version_artifact(session,second.id,role='clean').metadata_json['exact_audio_timing'] is True
            next_edit=EditPlan(run_id=run.id,base_version_id=second.id,
                operations=[{'kind':'subtitle_edit','style':{'color_hex':'#000000'}}])
            assert svc._validate(session,next_edit)['plan']['base_version_id']==second.id
    engine.dispose()


@pytest.mark.parametrize('profile',[None,0,1,'true',{},[]])
def test_clean_audio_timing_profile_requires_boolean_before_render(profile):
    from app.assistant_executor import render_local_candidate
    from app.assistant_media import AssistantMediaError
    # Invalid trusted metadata must fail before plans, paths or media are used.
    with pytest.raises(AssistantMediaError,match='must be a boolean'):
        render_local_candidate(None,None,{},None,None,30,exact_audio_timing=profile)


def test_ending_caption_quantization_rejects_collapsed_fragment():
    from types import SimpleNamespace
    from app.assistant_executor import _ending_prefix_track
    from app.assistant_subtitles import SubtitleTrack,SubtitleCue
    from app.assistant_media import TimelineSpan,AssistantMediaError
    rendered=SimpleNamespace(ending_timing=SimpleNamespace(start_frame=1),metadata=SimpleNamespace(fps='120'),
        base_timeline=(TimelineSpan(0,1/120,0),))
    track=SubtitleTrack(cues=(SubtitleCue(cue_id='tiny',start_sec=0.,end_sec=.02,text='Retained fragment'),))
    with pytest.raises(AssistantMediaError,match='shorter than ASS resolution'):
        _ending_prefix_track(track,rendered)
