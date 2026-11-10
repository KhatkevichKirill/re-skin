"""Real local worker lifecycle against disposable SQLite and synthetic media."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.assistant_access import ProjectAllowlist
from app.assistant_contracts import EditPlan
from app.assistant_executor import execute_assistant_job
from app.assistant_jobs import AssistantJob, request_cancel
from app.assistant_models import AssistantMessage, AssistantVersion
from app.assistant_service import AssistantAdmissionService
from app.assistant_store import (AssistantArtifactStorage, ArtifactInput, get_version_artifact,
                                 import_run_version, import_media_version, verify_artifact)
from app.assistant_subtitles import SubtitleTrack, SubtitleCue, get_subtitle_context
from app import assistant_media
from app.db import Base
from app.models import Run, VideoProject


@pytest.fixture
def setup(tmp_path,request):
    duration=getattr(request,"param",3)
    engine = create_engine(f"sqlite:///{tmp_path/'worker.db'}")
    @event.listens_for(engine, "connect")
    def pragma(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    data = tmp_path/"data"
    data.mkdir()
    storage = AssistantArtifactStorage(data)
    source = data/"source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=white:s=160x120:r=30:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)], check=True)
    with factory.begin() as session:
        project = VideoProject(source_type="upload", source_ref="fixture")
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status="done", result_local_path=str(source), result_duration_sec=duration)
        session.add(run)
        session.flush()
        ids = (run.id, project.id)
    yield factory, storage, source, ids
    engine.dispose()


def accepted(setup, operation, *, editable=False, donor_audio=None):
    factory, storage, source, (run_id, project_id) = setup
    with factory.begin() as session:
        if editable:
            track = SubtitleTrack(cues=(SubtitleCue(cue_id="a", start_sec=.2, end_sec=.4, text="First"),
                                       SubtitleCue(cue_id="b", start_sec=.4, end_sec=.6, text="Second"),
                                       SubtitleCue(cue_id="c", start_sec=1.2, end_sec=1.8, text="Later")))
            path = storage.data_root/"track.json"
            path.write_text(track.model_dump_json())
            from app.assistant_subtitles import burn_track
            captioned=storage.data_root/"captioned-source.mp4"
            burn_track(source,captioned,track=track,clean_source_verified=True)
            version = import_media_version(session, session.get(Run,run_id), actor="operator", storage=storage,
                full=ArtifactInput(captioned, metadata={"duration_sec":3, "subtitle_status":"editable_burned"}),
                clean=ArtifactInput(source, metadata={"duration_sec":3}),
                captions=ArtifactInput(path,"subtitle"))
        else:
            version = import_run_version(session, session.get(Run,run_id), actor="operator", storage=storage)
        operations = operation if isinstance(operation,list) else [operation]
        for index, selected in enumerate(operations):
            if selected["kind"] in ("audio_splice","full_track_audio","replace_footage") and "source" not in selected:
                asset = get_version_artifact(session,version.id)
                if donor_audio is not None:
                    from app.assistant_store import _add_artifact
                    asset = _add_artifact(session,version,ArtifactInput(donor_audio,"audio",{"duration_sec":1.25,"has_audio":True}),storage)
                operations[index] = {**selected,"source":{"asset_id":asset.id,"version_id":version.id}}
        plan = EditPlan(run_id=run_id, base_version_id=version.id, operations=operations)
        receipt = AssistantAdmissionService(storage=storage, policy=ProjectAllowlist([project_id]), enabled=True).admit(
            session, run_id=run_id, plan=plan, instruction="Fixture edit", idempotency_key="request",
            actor="operator", now=datetime.now(timezone.utc))
        return receipt


def execute(setup, job_id, **kwargs):
    factory, storage, _, (_, project_id) = setup
    return execute_assistant_job(factory,job_id,owner="test-worker",storage=storage,
                                policy=ProjectAllowlist([project_id]),enabled=True,**kwargs)


def test_worker_publishes_full_preview_explanation_and_never_changes_run(setup):
    receipt = accepted(setup, {"kind":"cut", "start_sec":1, "end_sec":2})
    result = execute(setup, receipt.job_id)
    assert result.status == "succeeded"
    factory, storage, source, (run_id, _) = setup
    with factory() as session:
        job = session.get(AssistantJob,receipt.job_id)
        assert (job.status,job.phase) == ("succeeded","publish")
        version = session.get(AssistantVersion,receipt.pending_version_id)
        assert version.status == "succeeded"
        assert get_version_artifact(session,version.id).metadata_json["duration_sec"] == 2
        assert verify_artifact(get_version_artifact(session,version.id,role="preview"),storage=storage).exists()
        assert session.get(Run,run_id).result_local_path == str(source)
        assert "Removed [1, 2)" in result.explanation
        assert len(session.scalars(select(AssistantMessage)).all()) == 2
    assert execute(setup, receipt.job_id).status == "not_claimed"


def test_worker_preserves_short_editable_cues_after_trim_and_reburn(setup):
    receipt = accepted(setup, {"kind":"trim", "start_sec":0, "end_sec":2}, editable=True)
    result = execute(setup, receipt.job_id)
    assert result.status == "succeeded"
    factory, storage, _, _ = setup
    with factory() as session:
        track = get_subtitle_context(session,result.version_id,storage=storage)
        assert [c.text for c in track.cues] == ["First","Second","Later"]
        assert track.cues[0].end_sec-track.cues[0].start_sec == pytest.approx(.2)


def test_worker_cancellation_inside_renderer_never_publishes(setup, monkeypatch):
    receipt = accepted(setup, {"kind":"trim", "start_sec":0, "end_sec":2})
    factory, _, _, _ = setup
    def cancel(*args, **kwargs):
        with factory.begin() as session:
            request_cancel(session,receipt.job_id,now=datetime.now(timezone.utc))
        raise assistant_media.AssistantMediaCancelled("fixture cancel")
    monkeypatch.setattr(assistant_media,"trim_video",cancel)
    assert execute(setup,receipt.job_id).status == "cancelled"
    with factory() as session:
        assert session.get(AssistantVersion,receipt.pending_version_id).status == "cancelled"


def test_worker_rechecks_scope_before_work(setup):
    receipt = accepted(setup, {"kind":"trim", "start_sec":0, "end_sec":2})
    factory, storage, _, _ = setup
    result = execute_assistant_job(factory,receipt.job_id,owner="worker",storage=storage,
                                  policy=ProjectAllowlist(),enabled=True)
    assert result.status == "failed"


def test_worker_audio_has_listenable_before_after_preview(setup):
    receipt = accepted(setup, {"kind":"audio_splice", "start_sec":1, "end_sec":2,"gain_db":-6})
    result = execute(setup,receipt.job_id)
    assert result.status == "succeeded"
    factory, storage, _, _ = setup
    with factory() as session:
        preview = get_version_artifact(session,result.version_id,role="preview")
        assert preview.metadata_json["preview_mode"] == "before_after"
        assert preview.metadata_json["sequence"] == ["before","after"]
        assert preview.metadata_json["switch_sec"] == 3
        assert assistant_media.probe_trusted_media(verify_artifact(preview,storage=storage)).video_duration_sec == 6
        assert "music/background also change" in result.explanation


@pytest.mark.parametrize("setup",[8],indirect=True)
def test_phrase_preview_includes_entire_changed_interval_and_context(setup):
    receipt=accepted(setup,{"kind":"audio_splice","start_sec":1,"end_sec":7,"gain_db":-6})
    result=execute(setup,receipt.job_id)
    assert result.status == "succeeded"
    factory,storage,_,_=setup
    with factory() as session:
        preview=get_version_artifact(session,result.version_id,role="preview")
        metadata=preview.metadata_json
        assert (metadata["source_start_sec"],metadata["source_end_sec"],metadata["switch_sec"]) == (0,8,8)
        assert not metadata["preview_truncated"]
        assert (metadata["affected_start_sec"],metadata["affected_end_sec"]) == (1,7)
        assert assistant_media.probe_trusted_media(verify_artifact(preview,storage=storage)).video_duration_sec == 16


def test_worker_revoked_policy_before_publish_never_selects_output(setup, monkeypatch):
    receipt = accepted(setup, {"kind":"trim", "start_sec":0, "end_sec":2})
    factory, storage, _, _ = setup
    original = AssistantAdmissionService.revalidate_job
    calls = []
    def revalidate(self,session,job_id,**kwargs):
        calls.append(job_id)
        if len(calls)>1:
            from app.assistant_access import AssistantAccessDenied
            raise AssistantAccessDenied("fixture policy revoked")
        return original(self,session,job_id,**kwargs)
    monkeypatch.setattr(AssistantAdmissionService,"revalidate_job",revalidate)
    assert execute(setup,receipt.job_id).status == "failed"
    with factory() as session:
        assert session.get(AssistantVersion,receipt.pending_version_id).status == "failed"


def test_actual_authorized_donor_search_returns_ranked_cues_and_metrics(setup):
    from app.assistant_access import AuthorizedArtifactCatalog
    from app.assistant_audio import find_audio_donor_candidates
    receipt=accepted(setup,{"kind":"trim","start_sec":0,"end_sec":2},editable=True)
    factory,storage,_,(_,project_id)=setup
    with factory() as session:
        catalog=AuthorizedArtifactCatalog(session,policy=ProjectAllowlist([project_id]),
                    is_available=lambda artifact:bool(verify_artifact(artifact,storage=storage)))
        candidates=find_audio_donor_candidates(session,catalog=catalog,storage=storage,
                    target_version_id=receipt.base_version_id,phrase="First")
        assert len(candidates)==1
        candidate=candidates[0]
        assert candidate.text == "First" and candidate.start_sec == .2
        assert candidate.review_required and not candidate.pronunciation_verified
        assert candidate.level_rms_dbfs is not None


def test_discovery_localizes_with_fake_asr_and_remaining_budget(setup):
    from app.assistant_access import AuthorizedArtifactCatalog
    from app.assistant_audio import discover_audio_donors
    from app.assistant_asr import ASRTranscript
    receipt=accepted(setup,{"kind":"trim","start_sec":0,"end_sec":2})
    factory,storage,_,(_,project_id)=setup
    calls=[]
    class FakeASR:
        def transcribe(self,path,**kwargs):
            calls.append(kwargs)
            return ASRTranscript(language="en",language_probability=.9,network_denied=True,max_rss_kib=100,
                source_start_sec=0,source_end_sec=3,source_duration_sec=3,
                words=[{"text":" Hello","start_sec":.2,"end_sec":.5,"probability":.9}])
    with factory() as session:
        catalog=AuthorizedArtifactCatalog(session,policy=ProjectAllowlist([project_id]),
                    is_available=lambda artifact:bool(verify_artifact(artifact,storage=storage)))
        report=discover_audio_donors(session,catalog=catalog,storage=storage,
            target_version_id=receipt.base_version_id,phrase="Hello",asr=FakeASR(),timeout_sec=30)
        assert len(calls)==1 and 0<calls[0]["timeout_sec"]<=30
        assert len(report.candidates)==1 and report.candidates[0].timing_precision=="estimated_word"
        assert report.candidates[0].review_required and not report.candidates[0].pronunciation_verified


def test_worker_persists_duration_fit_provenance_and_actual_explanation(setup):
    factory,storage,_,_=setup
    donor=storage.data_root/"duration-donor.wav"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i",
        "aevalsrc='if(lt(t,1.15),0.15*sin(2*PI*880*t),0)':s=48000:d=1.25",
        "-c:a","pcm_s16le",str(donor)],check=True)
    receipt=accepted(setup,{"kind":"audio_splice","start_sec":1,"end_sec":2,"donor_duration_sec":1.25},donor_audio=donor)
    result=execute(setup,receipt.job_id)
    assert result.status=="succeeded"
    with factory() as session:
        full=get_version_artifact(session,result.version_id)
        fit=full.metadata_json["audio_fit"]
        assert fit["donor_duration_sec"]==1.25 and fit["target_duration_sec"]==1
        assert fit["speed_ratio"]==1.25 and fit["review_required"]
        assert "Fitted 1.25s of donor audio into 1s" in result.explanation
        assert "Cadence changes require listening review" in result.explanation


def test_audio_edit_copies_burned_picture_and_future_subtitle_edit_keeps_new_audio(setup):
    from tests.test_assistant_audio_subtitles import packets
    import json
    def audio_packets(path):
        return json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","a:0","-show_packets",
            "-show_data_hash","sha256","-show_entries","packet=pts,dts,duration,data_hash","-of","json",str(path)],
            capture_output=True,check=True).stdout)["packets"]
    receipt=accepted(setup,{"kind":"audio_splice","start_sec":1,"end_sec":2,"gain_db":-6},editable=True)
    factory,storage,_,(run_id,project_id)=setup
    with factory() as session:
        original=verify_artifact(get_version_artifact(session,receipt.base_version_id),storage=storage)
        original_video=packets(original)
        original_audio=audio_packets(original)
    result=execute(setup,receipt.job_id)
    assert result.status=="succeeded"
    with factory.begin() as session:
        updated=verify_artifact(get_version_artifact(session,result.version_id),storage=storage)
        assert packets(updated)==original_video
        changed_audio=audio_packets(updated)
        assert changed_audio!=original_audio
        clean=verify_artifact(get_version_artifact(session,result.version_id,role="clean"),storage=storage)
        assert audio_packets(clean)==changed_audio
        plan=EditPlan(run_id=run_id,base_version_id=result.version_id,
                      operations=[{"kind":"subtitle_edit","style":{"color_hex":"#FF0000"}}])
        following=AssistantAdmissionService(storage=storage,policy=ProjectAllowlist([project_id]),enabled=True).admit(
            session,run_id=run_id,plan=plan,instruction="Change subtitle color",idempotency_key="subtitle-after-audio",
            actor="operator",now=datetime.now(timezone.utc))
    rerender=execute(setup,following.job_id)
    assert rerender.status=="succeeded"
    with factory() as session:
        full=verify_artifact(get_version_artifact(session,rerender.version_id),storage=storage)
        assert audio_packets(full)==changed_audio
        assert get_subtitle_context(session,rerender.version_id,storage=storage).style.color_hex=="#FF0000"


def test_unsafe_tempo_fit_fails_with_review_code_without_publishing(setup):
    factory,storage,_,_=setup
    donor=storage.data_root/"audible-ending.wav"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i",
        "aevalsrc='0.15*sin(2*PI*880*t)':s=48000:d=1.25","-c:a","pcm_s16le",str(donor)],check=True)
    receipt=accepted(setup,{"kind":"audio_splice","start_sec":1,"end_sec":2,"donor_duration_sec":1.25},donor_audio=donor)
    assert execute(setup,receipt.job_id).status=="failed"
    with factory() as session:
        assert session.get(AssistantJob,receipt.job_id).error_code=="audio_fit_requires_review"
        assert session.get(AssistantVersion,receipt.pending_version_id).status=="failed"


def test_whole_edit_deadline_fails_with_distinct_timeout_code(setup):
    receipt=accepted(setup,{"kind":"trim","start_sec":0,"end_sec":2})
    result=execute(setup,receipt.job_id,timeout_sec=.000001)
    assert result.status=="failed"
    factory,_,_,_=setup
    with factory() as session:
        assert session.get(AssistantJob,receipt.job_id).error_code=="local_edit_timeout"
        assert session.get(AssistantVersion,receipt.pending_version_id).status=="failed"


def test_invalid_edit_timeout_does_not_claim_queued_job(setup):
    receipt=accepted(setup,{"kind":"trim","start_sec":0,"end_sec":2})
    for timeout in (0,601):
        with pytest.raises(ValueError,match="executor timeout"):
            execute(setup,receipt.job_id,timeout_sec=timeout)
    factory,_,_,_=setup
    with factory() as session:
        job=session.get(AssistantJob,receipt.job_id)
        assert job.status=="queued" and job.claim_count==0


def test_timeout_during_publication_rolls_back_output_and_reports_deadline(setup,monkeypatch):
    from app import assistant_executor as executor
    from app.assistant_models import AssistantSelection
    receipt=accepted(setup,{"kind":"trim","start_sec":0,"end_sec":2})
    guards=[]
    original_guard=executor._Guard
    class Guard(original_guard):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            guards.append(self)
    monkeypatch.setattr(executor,"_Guard",Guard)
    publish=executor.publish_job
    def expire_after_pin(*args,**kwargs):
        result=publish(*args,**kwargs)
        guards[0].budget.deadline=0
        return result
    monkeypatch.setattr(executor,"publish_job",expire_after_pin)
    assert execute(setup,receipt.job_id).status=="failed"
    factory,_,_,_=setup
    with factory() as session:
        job=session.get(AssistantJob,receipt.job_id)
        version=session.get(AssistantVersion,receipt.pending_version_id)
        assert job.error_code=="local_edit_timeout" and version.status=="failed"
        assert session.get(AssistantSelection,version.conversation_id).selected_version_id==receipt.base_version_id


@pytest.mark.parametrize('editable', [False, True])
def test_dip_executes_pair_and_preserves_caption_lineage(setup, editable):
    receipt = accepted(setup, [
        {'kind':'image_fade', 'start_sec':.5, 'end_sec':1.5, 'direction':'out'},
        {'kind':'image_fade', 'start_sec':1.5, 'end_sec':2.5, 'direction':'in'}], editable=editable)
    result = execute(setup, receipt.job_id)
    assert result.status == 'succeeded'
    factory, storage, source, (run_id, _) = setup
    with factory() as session:
        full = get_version_artifact(session, result.version_id)
        preview = get_version_artifact(session, result.version_id, role='preview')
        assert full.metadata_json['duration_sec'] == 3
        assert preview.metadata_json['source_start_sec'] <= .5
        assert preview.metadata_json['source_end_sec'] >= 2.5
        assert preview.metadata_json['affected_end_sec'] == 2.5
        assert not preview.metadata_json['preview_truncated']
        assert session.get(Run, run_id).result_local_path == str(source)
        if editable:
            clean = get_version_artifact(session, result.version_id, role='clean')
            assert clean.metadata_json['duration_sec'] == 3
            track = get_subtitle_context(session, result.version_id, storage=storage)
            assert [(c.cue_id,c.start_sec,c.end_sec) for c in track.cues] == [('a',.2,.4),('b',.4,.6),('c',1.2,1.8)]
        assert 'Faded picture out over' in result.explanation
        assert 'Faded picture in over' in result.explanation


@pytest.mark.parametrize('setup', [12], indirect=True)
def test_dip_preview_covers_both_ramps_beyond_six_seconds(setup):
    receipt = accepted(setup, [
        {'kind':'image_fade', 'start_sec':1, 'end_sec':6, 'direction':'out'},
        {'kind':'image_fade', 'start_sec':6, 'end_sec':11, 'direction':'in'}])
    result = execute(setup, receipt.job_id)
    assert result.status == 'succeeded'
    factory, storage, _, _ = setup
    with factory() as session:
        preview = get_version_artifact(session, result.version_id, role='preview')
        assert preview.metadata_json['source_start_sec'] == 0
        assert preview.metadata_json['source_end_sec'] == 12
        assert not preview.metadata_json['preview_truncated']


def test_all_selected_picture_audio_and_subtitle_edits_publish_once(setup):
    receipt = accepted(setup,[
        {'kind':'image_fade','start_sec':.5,'end_sec':1,'direction':'out'},
        {'kind':'replace_footage','start_sec':2,'end_sec':2.5},
        {'kind':'audio_splice','start_sec':1,'end_sec':2,'gain_db':-6},
        {'kind':'subtitle_edit','changes':[{'action':'delete','cue_id':'a'}]}],editable=True)
    result = execute(setup,receipt.job_id)
    assert result.status == 'succeeded'
    factory,storage,_,_=setup
    with factory() as session:
        full = get_version_artifact(session,result.version_id)
        assert len(full.metadata_json['operations']) == 4
        assert full.metadata_json['operation'] == 'composite'
        assert full.metadata_json['audio_policy'] == 'replace_mixed_audio'
        track = get_subtitle_context(session,result.version_id,storage=storage)
        assert [c.cue_id for c in track.cues] == ['b','c']
        preview = get_version_artifact(session,result.version_id,role='preview')
        assert preview.metadata_json['preview_mode'] == 'before_after'
        assert preview.metadata_json['affected_end_sec'] >= 2.5
        assert 'Faded picture out' in result.explanation and 'Replaced mixed audio' in result.explanation
        assert 'Replaced only the requested picture interval' in result.explanation
        assert 'revised complete cue track' in result.explanation


def test_second_stage_failure_publishes_no_partial_result_or_selection(setup,monkeypatch):
    receipt = accepted(setup,[
        {'kind':'replace_footage','start_sec':0,'end_sec':.5},
        {'kind':'image_fade','start_sec':1,'end_sec':2,'direction':'out'}])
    factory,storage,source,_ = setup
    from app.assistant_models import AssistantSelection
    with factory() as session:
        before = session.scalars(select(AssistantSelection)).one().selected_version_id
        base = get_version_artifact(session,receipt.base_version_id)
        before_bytes = verify_artifact(base,storage=storage).read_bytes()
    def fail(*args,**kwargs):
        raise assistant_media.AssistantMediaError('second stage failed')
    monkeypatch.setattr(assistant_media,'image_fades',fail)
    assert execute(setup,receipt.job_id).status == 'failed'
    with factory() as session:
        assert session.get(AssistantVersion,receipt.pending_version_id).status == 'failed'
        assert session.scalars(select(AssistantSelection)).one().selected_version_id == before
        assert verify_artifact(get_version_artifact(session,receipt.base_version_id),storage=storage).read_bytes() == before_bytes


def test_multiple_audio_edits_preserve_captioned_picture_packets(setup):
    receipt = accepted(setup,[
        {'kind':'audio_splice','start_sec':.5,'end_sec':1,'gain_db':-6},
        {'kind':'audio_splice','start_sec':2,'end_sec':2.5,'gain_db':-12}],editable=True)
    factory,storage,_,_ = setup
    def packets(path):
        return subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_packets',
            '-show_data_hash','sha256','-show_entries','packet=pts,dts,duration,data_hash','-of','json',str(path)],
            check=True,capture_output=True).stdout
    with factory() as session:
        before = packets(verify_artifact(get_version_artifact(session,receipt.base_version_id),storage=storage))
    result = execute(setup,receipt.job_id)
    assert result.status == 'succeeded'
    with factory() as session:
        full = get_version_artifact(session,result.version_id)
        assert packets(verify_artifact(full,storage=storage)) == before
        assert len(full.metadata_json['operations']) == 2
        assert len(get_subtitle_context(session,result.version_id,storage=storage).cues) == 3
        preview = get_version_artifact(session,result.version_id,role='preview')
        assert preview.metadata_json['affected_start_sec'] == .5
        assert preview.metadata_json['affected_end_sec'] == 2.5
