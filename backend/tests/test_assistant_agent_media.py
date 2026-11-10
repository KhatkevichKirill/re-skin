import hashlib
import pytest
from sqlalchemy import select, func
from tests.test_assistant_executor import setup
from app.assistant_access import ProjectAllowlist, AssistantAccessDenied
from app.assistant_agent_media import AgentMediaService
from app.assistant_contracts import EditPlan
from app.assistant_models import AssistantVersion
from app.assistant_store import import_run_version, get_version_artifact, PinnedFile
from app.models import Run

@pytest.fixture
def tools(setup):
    factory,storage,source,(run_id,project_id)=setup
    with factory.begin() as session:
        version=import_run_version(session,session.get(Run,run_id),actor="operator",storage=storage)
        version_id=version.id
        asset_id=get_version_artifact(session,version_id).id
    with AgentMediaService(factory,storage=storage,policy=ProjectAllowlist([project_id]),target_version_id=version_id) as service:
        yield service,version_id,asset_id,run_id

def test_pixels_candidate_retention_without_publication(tools,setup):
    service,version,asset,run=tools
    frames=service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[.51,1.0])["frames"]
    assert frames[0]["timestamp_sec"] == .5
    data=service.read_frame(frames[0]["frame_id"])
    assert data.startswith(b"\xff\xd8") and hashlib.sha256(data).hexdigest()==frames[0]["sha256"]
    plan=EditPlan(run_id=run,base_version_id=version,operations=[{"kind":"image_fade","start_sec":1,"end_sec":2,"direction":"out"}])
    candidate=service.render_candidate(plan)["candidate"]
    assert candidate["plan"]==plan.model_dump(mode="json")
    frame=service.inspect_frames(candidate_id=candidate["candidate_id"],timestamps_sec=[1.5])["frames"][0]
    assert frame["source_sha256"]==candidate["sha256"]
    pinned=service.persist_candidate(candidate["candidate_id"])
    assert service.storage.verify(PinnedFile(**pinned)).is_file()
    with setup[0]() as session:
        assert session.scalar(select(func.count()).select_from(AssistantVersion))==1
        assert session.get(Run,run).result_local_path==str(setup[2])

def test_invalid_times_bytes_and_closed(tools):
    service,version,asset,_=tools
    for stamps in ([True],[float("nan")],[-1],[3],list(range(9))):
        with pytest.raises((ValueError,RuntimeError)):
            service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=stamps)
    service.MAX_CALL_BYTES=1
    with pytest.raises(RuntimeError,match="byte budget"):
        service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])
    service.close()
    with pytest.raises(RuntimeError,match="closed"):
        service.read_frame(asset)

def test_authorization_rechecked_for_frame_read(tools):
    service,version,asset,_=tools
    frame=service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])["frames"][0]
    service.policy=ProjectAllowlist([])
    with pytest.raises(AssistantAccessDenied):
        service.read_frame(frame["frame_id"])

def test_cancel_before_subprocess(tools):
    service,version,asset,_=tools
    service.budget.cancelled=lambda:True
    with pytest.raises(RuntimeError,match="cancelled"):
        service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])

def test_source_and_candidate_speech_are_window_bound(tools):
    from app.assistant_asr import ASRTranscript, ASRWord
    service,version,asset,run=tools
    class ASR:
        def transcribe(self,path,**kwargs):
            assert kwargs["start_sec"]==.2 and kwargs["end_sec"]==.8
            assert path.relative_to(service.storage.root).parts[0] == "objects"
            assert path.name == hashlib.sha256(path.read_bytes()).hexdigest()
            kwargs["cancelled"]()
            return ASRTranscript(language="en",language_probability=.9,network_denied=True,max_rss_kib=100,
                source_start_sec=.2,source_end_sec=.8,source_duration_sec=3,
                words=(ASRWord(text="hello",start_sec=.3,end_sec=.5,probability=.9),))
    service.asr=ASR()
    speech=service.transcribe_window(asset_id=asset,version_id=version,start_sec=.2,end_sec=.8)["speech"][0]
    assert not speech["pronunciation_verified"] and not speech["coverage_complete"]
    plan=EditPlan(run_id=run,base_version_id=version,operations=[{"kind":"trim","start_sec":0,"end_sec":2}])
    candidate=service.render_candidate(plan)["candidate"]
    speech=service.transcribe_window(candidate_id=candidate["candidate_id"],start_sec=.2,end_sec=.8)["candidate_speech"][0]
    assert speech["source_sha256"]==candidate["sha256"] and "asset_id" not in speech
    with pytest.raises(ValueError):
        service.transcribe_window(asset_id=asset,version_id=version,start_sec=0,end_sec=61)

def test_candidate_hash_change_prevents_evidence(tools):
    service,version,asset,run=tools
    plan=EditPlan(run_id=run,base_version_id=version,operations=[{"kind":"trim","start_sec":0,"end_sec":2}])
    candidate=service.render_candidate(plan)["candidate"]
    service.candidates[candidate["candidate_id"]]["path"].write_bytes(b"changed")
    with pytest.raises(RuntimeError,match="integrity"):
        service.inspect_frames(candidate_id=candidate["candidate_id"],timestamps_sec=[0])

def test_discovery_original_is_immutable_catalog_source(tools,setup):
    from app.models import VideoProject
    service,version,asset,run=tools
    with setup[0].begin() as session:
        session.get(VideoProject,setup[3][1]).source_local_path=str(setup[2])
    sources=service.search_sources(query="original")["sources"]
    assert len(sources)==1 and sources[0]["label"]=="Original project source video"
    assert sources[0]["asset_id"] != asset and sources[0]["version_id"]==version
    with setup[0]() as session:
        assert session.scalar(select(func.count()).select_from(AssistantVersion))==1

def test_known_unavailable_is_distinct_from_integrity_and_cancellation(tools):
    from app.assistant_agent_media import AgentToolUnavailable
    service,version,asset,_=tools
    with pytest.raises(AgentToolUnavailable,match="offline speech"):
        service.transcribe_window(asset_id=asset,version_id=version,start_sec=0,end_sec=1)
    service.budget.cancelled=lambda:True
    with pytest.raises(RuntimeError) as error:
        service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])
    assert not isinstance(error.value,AgentToolUnavailable)

def test_media_total_budget_allows_agent_deadline(setup):
    factory,storage,_,(_,project)=setup
    with AgentMediaService(factory,storage=storage,policy=ProjectAllowlist([project]),
            target_version_id="00000000-0000-0000-0000-000000000000",timeout_sec=1200) as service:
        assert service.budget.deadline > 0

def test_unavailable_original_does_not_block_donors_but_authority_failure_does(tools,monkeypatch):
    from app import assistant_agent_sources as sources
    from app.assistant_agent_media import AgentToolUnavailable
    service,version,asset,_=tools
    def unsupported(*args,**kwargs):
        raise sources.SourceCapabilityUnavailable("unsupported private detail")
    monkeypatch.setattr(sources,'import_original_source',unsupported)
    result=service.search_sources(query='video')
    assert any(item['asset_id']==asset for item in result['sources'])
    assert result['limitations']==['original_source_unavailable']
    assert 'private detail' not in str(result)
    with pytest.raises(AgentToolUnavailable):
        service.search_sources(query='original')
    def denied(*args,**kwargs):
        raise AssistantAccessDenied('denied')
    monkeypatch.setattr(sources,'import_original_source',denied)
    with pytest.raises(AssistantAccessDenied):
        service.search_sources(query='video')

def test_source_and_candidate_video_tail_clamps_to_actual_audio(tools,monkeypatch):
    from types import SimpleNamespace
    from app.assistant_asr import ASRTranscript,ASRWord
    from app import assistant_audio
    from app.assistant_agent_media import AgentToolUnavailable
    service,version,asset,run=tools
    monkeypatch.setattr(assistant_audio,'probe_trusted_audio',lambda *a,**k:SimpleNamespace(duration_sec=2.98))
    class ASR:
        def transcribe(self,path,**kwargs):
            assert kwargs['end_sec']==2.98
            return ASRTranscript(language='en',language_probability=.9,network_denied=True,max_rss_kib=1,
                source_start_sec=2,source_end_sec=2.98,source_duration_sec=2.98,
                words=(ASRWord(text='ending',start_sec=2.2,end_sec=2.9,probability=.9),))
    service.asr=ASR()
    speech=service.transcribe_window(asset_id=asset,version_id=version,start_sec=2,end_sec=3)['speech'][0]
    assert speech['source_end_sec']==speech['source_duration_sec']==2.98
    plan=EditPlan(run_id=run,base_version_id=version,operations=[{'kind':'image_fade','direction':'out','start_sec':2,'end_sec':3}])
    candidate=service.render_candidate(plan)['candidate']
    speech=service.transcribe_window(candidate_id=candidate['candidate_id'],start_sec=2,end_sec=3)['candidate_speech'][0]
    assert speech['source_end_sec']==speech['source_duration_sec']==2.98
    with pytest.raises(AgentToolUnavailable):
        service.transcribe_window(asset_id=asset,version_id=version,start_sec=2,end_sec=3.1)

def test_turn_frame_count_and_byte_boundaries(tools):
    from app.assistant_agent_media import AgentToolUnavailable
    service,version,asset,_=tools
    assert service.MAX_TOTAL_FRAMES==32
    assert service.MAX_TOTAL_FRAME_BYTES==8*1024*1024
    assert service.MAX_CALL_BYTES==2*1024*1024
    assert service.MAX_SCRATCH_BYTES==512*1024*1024
    # Count enforcement happens before any resolution/FFmpeg; 32nd frame is
    # permitted but requesting a 33rd is refused before touching media.
    service.frames.update({str(index):{} for index in range(31)})
    evidence=service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])['frames'][0]
    with pytest.raises(AgentToolUnavailable,match='turn frame budget'):
        service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])
    service.frames.clear()
    service.total_frame_bytes=service.MAX_TOTAL_FRAME_BYTES-evidence['size_bytes']
    service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])
    assert service.total_frame_bytes==service.MAX_TOTAL_FRAME_BYTES
    with pytest.raises(AgentToolUnavailable,match='byte budget'):
        service.inspect_frames(asset_id=asset,version_id=version,timestamps_sec=[0])

def test_actual_24fps_decimal_frame_snapping_and_rounded_end(setup):
    import io
    import subprocess
    from PIL import Image
    factory,storage,source,(run_id,project_id)=setup
    # Frame169 is white; frame170 and later are black. This checks actual
    # decoded pixels as well as the reported canonical source timestamp.
    source.unlink()
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=white:s=160x120:r=24:d=26.666667',
        '-vf',"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(n,170)'",
        '-frames:v','640','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(source)],check=True)
    with factory.begin() as session:
        run=session.get(Run,run_id)
        run.result_duration_sec=26.666667
        version=import_run_version(session,run,actor='operator',storage=storage)
        version_id=version.id;asset=get_version_artifact(session,version_id).id
    with AgentMediaService(factory,storage=storage,policy=ProjectAllowlist([project_id]),target_version_id=version_id) as service:
        frames=service.inspect_frames(asset_id=asset,version_id=version_id,
            timestamps_sec=[7.083333,7.0833333,7.083333333333333,7.083332,26.666666])['frames']
        assert [frame['timestamp_sec'] for frame in frames[:3]]==[170/24]*3
        assert frames[3]['timestamp_sec']==169/24
        assert frames[4]['timestamp_sec']==639/24
        for frame in frames[:3]:
            with Image.open(io.BytesIO(service.read_frame(frame['frame_id']))) as image:
                assert image.convert('L').getextrema()[1]<10
        with Image.open(io.BytesIO(service.read_frame(frames[3]['frame_id']))) as image:
            assert image.convert('L').getextrema()[0]>240
        with pytest.raises(ValueError,match='outside source'):
            service.inspect_frames(asset_id=asset,version_id=version_id,timestamps_sec=[26.666667])
