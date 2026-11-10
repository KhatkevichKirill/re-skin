"""Bounded explicit preparation; synthetic native media and isolated SQLite."""
import asyncio
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from sqlalchemy import select

from app import assistant_media as media
from app.assistant_access import ProjectAllowlist
from app.assistant_models import AssistantSelection, AssistantVersion, AssistantVersionArtifact
from app.assistant_preparation import prepare_editable_version, prepare_audio_tail
from app.assistant_store import AssistantConflict, get_version_artifact, select_version, verify_artifact
from app.assistant_legacy import import_legacy_run
from app.assistant_turns import AssistantTurnService
from app.db import get_db
from app.models import Run
from tests.test_assistant_store import setup, legacy


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, args)], check=True, capture_output=True, timeout=30)


@pytest.fixture
def native(setup):
    sessions, storage, data = setup
    video, audio, result = data/"picture.mp4", data/"audio.wav", data/"raw.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=160x120:r=30", "-frames:v", "892", "-an",
        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", video)
    count = 1315106
    tone = np.sin(np.arange(count)*2*np.pi*700/44100)*12000
    tone[-5000:] = np.sin(np.arange(5000)*2*np.pi*1400/44100)*20000
    samples = np.repeat(tone.astype('<i2')[:,None], 2, axis=1).tobytes()
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((2,2,44100,0,'NONE','not compressed'))
        handle.writeframes(samples)
    ffmpeg("-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", result)
    run_id, project_id, original = legacy(sessions, data, content=result.read_bytes(), duration=29.820998)
    service = AssistantTurnService(storage=storage, policy=ProjectAllowlist(project_id), enabled=True)
    with sessions.begin() as session:
        version = import_legacy_run(session, session.get(Run,run_id),actor="operator",storage=storage,policy=service.policy)
        source_id, conversation_id = version.id, version.conversation_id
    return sessions, storage, data, service, run_id, source_id, conversation_id, original


def test_native_import_retains_bytes_and_explicit_prepare_replay_never_reselects(native):
    sessions,storage,data,svc,rid,source,conversation,original=native
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    with pytest.raises(media.EditingPreparationRequired): media.probe_trusted_media(original)
    with sessions() as session:
        artifact=get_version_artifact(session,source)
        assert artifact.metadata_json['needs_editing_preparation'] is True
        assert artifact.sha256==original_hash
        child=prepare_editable_version(session,svc,rid,source,expected_selection_epoch=0,idempotency_key='prepare-one')
        child_id=child.id
        assert child.sequence==2 and child.parent_version_id==child.base_version_id==source
        output=get_version_artifact(session,child_id)
        assert output.owner_version_id==child_id and not output.metadata_json.get('needs_editing_preparation')
        info=media.probe_trusted_media(verify_artifact(output,storage=storage))
        assert info.video_duration_sec==float(Fraction(895,30))
        assert {row.role for row in session.scalars(select(AssistantVersionArtifact).where(AssistantVersionArtifact.version_id==child_id))}=={'input','full','preview'}
        assert session.get(Run,rid).result_local_path==str(original)
        assert original_hash==hashlib.sha256(original.read_bytes()).hexdigest()
    with sessions.begin() as session:
        selected=session.get(AssistantSelection,conversation)
        select_version(session,conversation,source,expected_epoch=selected.epoch)
    with sessions() as session:
        assert prepare_editable_version(session,svc,rid,source,expected_selection_epoch=0,idempotency_key='prepare-one').id==child_id
        epoch=session.get(AssistantSelection,conversation).epoch
        assert session.get(AssistantSelection,conversation).selected_version_id==source
        assert prepare_editable_version(session,svc,rid,source,expected_selection_epoch=epoch,idempotency_key='prepare-two').id==child_id
        assert session.get(AssistantSelection,conversation).selected_version_id==source
        with pytest.raises(AssistantConflict):
            prepare_editable_version(session,svc,rid,source,expected_selection_epoch=epoch,idempotency_key='prepare-one')
        assert len(session.scalars(select(AssistantVersion)).all())==2


def test_readonly_api_context_and_backend_execution_denial(native,monkeypatch):
    from app.api_assistant import router,get_assistant_service
    sessions,storage,data,svc,rid,source,conversation,original=native
    app=FastAPI();app.include_router(router,prefix='/api/v2')
    def db():
        with sessions() as session:
            yield session
            session.commit()
    app.dependency_overrides[get_db]=db
    app.dependency_overrides[get_assistant_service]=lambda:svc
    root=f'/api/v2/assistant/runs/{rid}'
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            context=await client.get(root+'/context')
            assert context.status_code==200,context.text
            value=context.json()
            assert value['base_context'] is None and value['selected_version']['needs_editing_preparation']
            assert (await client.get(value['selected_version']['full_url'])).content==original.read_bytes()
            turn=await client.post(root+'/turns',json={'base_version_id':source,'instruction':'Fade ending','idempotency_key':'blocked'})
            assert turn.status_code==422,turn.text
            publication=await client.post(root+'/result',json={'version_id':source,'expected_selection_epoch':0,
                'expected_result_revision':value['run_result']['revision'],'idempotency_key':'blocked-publication'})
            assert publication.status_code==422,publication.text
            prepared=await client.post(root+f'/versions/{source}/prepare-editable',json={'expected_selection_epoch':0,'idempotency_key':'api-prepare'})
            assert prepared.status_code==200,prepared.text
            assert not prepared.json()['needs_editing_preparation']
            ctx=(await client.get(root+'/context')).json()
            assert ctx['selected_version_id']==prepared.json()['id']
            assert ctx['preparation_receipts'][0]['idempotency_key']=='api-prepare'
    asyncio.run(run())


def test_classification_only_accepts_isolated_positive_aac_tail(monkeypatch):
    metadata={'streams':[{'codec_type':'video','codec_name':'h264','pix_fmt':'yuv420p','avg_frame_rate':'30','r_frame_rate':'30',
        'sample_aspect_ratio':'1:1','duration':'29.733333','start_time':'0','width':160,'height':120},
        {'codec_type':'audio','codec_name':'aac','duration':'29.820998','duration_ts':1315106,'time_base':'1/44100','start_time':'0'}],
        'format':{'duration':'29.820998','format_name':'mov,mp4,m4a'}}
    def call(argv,*args,**kwargs):
        return json.dumps({'packets':[{'pts_time':f'{i/30:.6f}'} for i in range(892)]}) if '-show_packets' in argv else json.dumps(metadata)
    monkeypatch.setattr(media,'_run',call)
    with pytest.raises(media.EditingPreparationRequired) as classified:
        media._probe(Path('/private/source.mp4'),media._budget(30,None))
    assert classified.value.preparation['added_frames']==3
    original=json.loads(json.dumps(metadata))
    for mutate in (lambda m:m['streams'][1].update(duration='30',duration_ts=1323000),
            lambda m:m['streams'][1].update(duration='29.6',duration_ts=1305360),
            lambda m:m['streams'][1].update(duration='nan'),lambda m:m['streams'][1].update(start_time='nan'),
            lambda m:m['streams'][0].update(start_time='nan'),lambda m:m['streams'][0].update(pix_fmt='yuv444p'),
            lambda m:m['streams'][0].update(codec_name='hevc'),lambda m:m['streams'][1].update(start_time='0.01')):
        metadata.clear();metadata.update(json.loads(json.dumps(original)));mutate(metadata)
        with pytest.raises(media.AssistantMediaError) as error:
            media._probe(Path('/private/source.mp4'),media._budget(30,None))
        assert not isinstance(error.value,media.EditingPreparationRequired)


def test_full_track_895frame_44100_clock_and_audible_donor_end(native):
    from app.assistant_audio import replace_full_track
    sessions,storage,data,svc,rid,source,conversation,original=native
    prepared=data/'prepared-native.mp4';prepare_audio_tail(original,prepared)
    donor=data/'donor.wav'
    count=31*44100
    tone=np.zeros(count,dtype='<i2')
    # A sustained terminal donor signal proves the padded base did not shorten
    # full-track replacement to its original AAC extent.
    tone[int(29.7*44100):]= (np.sin(np.arange(count-int(29.7*44100))*2*np.pi*2000/44100)*20000).astype('<i2')
    with wave.open(str(donor),'wb') as handle:
        handle.setparams((2,2,44100,0,'NONE','not compressed'));handle.writeframes(np.repeat(tone[:,None],2,axis=1).tobytes())
    output=data/'full-track.mp4'
    result=replace_full_track(prepared,donor,output)
    assert result.metadata.video_duration_sec==float(Fraction(895,30))
    raw=data/'full-track.wav';ffmpeg('-i',output,'-vn','-c:a','pcm_s16le',raw)
    with wave.open(str(raw),'rb') as handle:
        samples=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').reshape(-1,2)
    assert len(samples)>=1315650-44
    assert np.sqrt(np.mean(samples[1315200:1315600].astype(float)**2))>5000


def test_audio_only_execution_and_asr_cannot_bypass_raw_owner_capability(native):
    from app.assistant_preparation import PreparationCapabilityError
    from app.assistant_asr import transcribe_authorized_artifact
    from app.assistant_models import AssistantArtifact
    from app.assistant_store import ArtifactInput, import_run_version
    from types import SimpleNamespace
    sessions,storage,data,svc,rid,source,conversation,original=native
    # A strict supported target plus an audio-kind artifact owned by the raw V1
    # proves audio-only donor paths check their enclosing owner capability.
    target_run,pid,target_path=legacy(sessions,data,content=(data/'picture.mp4').read_bytes(),duration=892/30)
    with sessions.begin() as session:
        session.get(Run,target_run).project_id=session.get(Run,rid).project_id
        session.flush()
        target=import_legacy_run(session,session.get(Run,target_run),actor='operator',storage=storage,policy=svc.policy)
        original_artifact=get_version_artifact(session,source)
        audio=AssistantArtifact(owner_version_id=source,storage_key=original_artifact.storage_key,sha256=original_artifact.sha256,
            size_bytes=original_artifact.size_bytes,media_kind='audio',metadata_json={'duration_sec':29.820998,'has_audio':True})
        session.add(audio);session.flush()
        catalog=svc.catalog(session)
        with pytest.raises(PreparationCapabilityError):
            catalog.resolve_editable_reference(target_version_id=target.id,asset_id=audio.id,version_id=source)
        calls=[]
        with pytest.raises(PreparationCapabilityError):
            transcribe_authorized_artifact(session,catalog=catalog,storage=storage,target_version_id=target.id,
                asset_id=audio.id,version_id=source,asr=SimpleNamespace(transcribe=lambda *a,**k:calls.append(1)),end_sec=1)
        assert calls==[]
        assert original_artifact.id not in {item.asset_id for item in catalog.list_editable_artifacts(target_version_id=target.id).artifacts}


def test_cancelled_preparation_never_creates_output(native):
    sessions,storage,data,svc,rid,source,conversation,original=native
    destination=data/'cancelled.mp4'
    with pytest.raises(media.AssistantMediaCancelled):
        prepare_audio_tail(original,destination,cancelled=lambda:True)
    assert not destination.exists()
    with sessions() as session:
        assert len(session.scalars(select(AssistantVersion)).all())==1
        assert session.get(Run,rid).result_local_path==str(original)


def test_preparation_commit_success_then_serialization_db_failure_is_retryable_503(native,monkeypatch):
    from app import api_assistant
    from sqlalchemy.exc import DBAPIError
    sessions,storage,data,svc,rid,source,conversation,original=native
    app=FastAPI();app.include_router(api_assistant.router,prefix='/api/v2')
    def db():
        with sessions() as session:
            yield session
            session.commit()
    app.dependency_overrides[get_db]=db
    app.dependency_overrides[api_assistant.get_assistant_service]=lambda:svc
    original_version=api_assistant._version
    failures=[1]
    def fail_once(*args,**kwargs):
        if failures[0]:
            failures[0]-=1
            raise DBAPIError('isolated receipt read',{},Exception('isolated connection loss'))
        return original_version(*args,**kwargs)
    monkeypatch.setattr(api_assistant,'_version',fail_once)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            path=f'/api/v2/assistant/runs/{rid}/versions/{source}/prepare-editable'
            body={'expected_selection_epoch':0,'idempotency_key':'confirmed-then-db-loss'}
            first=await client.post(path,json=body)
            assert first.status_code==503,first.text
            retry=await client.post(path,json=body)
            assert retry.status_code==200,retry.text
            with sessions() as session:
                assert len(session.scalars(select(AssistantVersion)).all())==2
                assert session.get(AssistantSelection,conversation).selected_version_id==retry.json()['id']
                assert session.get(Run,rid).result_local_path==str(original)
    asyncio.run(run())


def test_http_preparation_slot_busy_is_retryable_without_starting_native_work(monkeypatch):
    from app import api_assistant
    from fastapi import HTTPException
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(api_assistant,'_PREPARATION_SLOT',SimpleNamespace(acquire=lambda **kwargs:False,
        release=lambda:pytest.fail('unacquired slot must not release')))
    monkeypatch.setattr(api_assistant,'prepare_editable_version',lambda *args,**kwargs:calls.append(1))
    with pytest.raises(HTTPException) as error:
        api_assistant.prepare_editable('run','version',api_assistant.PrepareEditable(
            expected_selection_epoch=0,idempotency_key='busy-exact-request'),db=None,service=None)
    assert error.value.status_code==503 and calls==[]


def test_automatic_speech_enrichment_uses_actual_prepared_audio_horizon(native):
    from app.assistant_enrichment import AssistantContextEnricher
    from app.assistant_asr import ASRTranscript,ASRWord
    sessions,storage,data,svc,rid,source,conversation,original=native
    with sessions() as session:
        prepared=prepare_editable_version(session,svc,rid,source,expected_selection_epoch=0,idempotency_key='speech-ready')
        context=svc.build_context(session,run_id=rid,base_version_id=prepared.id,instruction='Find the ending speech')
    calls=[]
    class ASR:
        def transcribe(self,path,*,start_sec,end_sec,**kwargs):
            calls.append(end_sec)
            return ASRTranscript(language='en',language_probability=1,network_denied=True,max_rss_kib=1,
                source_start_sec=start_sec,source_end_sec=end_sec,source_duration_sec=end_sec,
                words=(ASRWord(text='CTA',start_sec=.1,end_sec=.3,probability=1),))
    enriched=AssistantContextEnricher(sessions,service=svc,asr=ASR()).enrich(context)
    assert len(calls)==1 and calls[0]==pytest.approx(29.820998,abs=1e-5)
    assert calls[0]<float(Fraction(895,30))
    assert enriched.speech_evidence and any('soundtrack ends' in text for text in enriched.evidence_limitations)


def test_preparation_timeout_is_retryable_503_and_releases_http_slot(monkeypatch):
    from app import api_assistant
    from fastapi import HTTPException
    from types import SimpleNamespace
    released=[]
    monkeypatch.setattr(api_assistant,'_PREPARATION_SLOT',SimpleNamespace(acquire=lambda **kwargs:True,
        release=lambda:released.append(True)))
    def timeout(*args,**kwargs):
        raise media.AssistantMediaTimeout('isolated native deadline')
    monkeypatch.setattr(api_assistant,'prepare_editable_version',timeout)
    with pytest.raises(HTTPException) as error:
        api_assistant.prepare_editable('run','version',api_assistant.PrepareEditable(
            expected_selection_epoch=0,idempotency_key='timeout-exact-request'),db=None,service=None)
    assert error.value.status_code==503 and released==[True]
    assert 'time limit' in error.value.detail and 'unchanged' in error.value.detail
