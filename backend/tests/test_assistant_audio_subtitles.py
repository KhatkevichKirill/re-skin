"""Synthetic local media acceptance; no network, providers or private assets."""
from pathlib import Path
import json
import subprocess

import numpy as np
import pytest

from app import assistant_audio as audio, assistant_media as media, assistant_subtitles as subs
from app.assistant_contracts import SubtitleEditOperation


def run(argv):
    result = subprocess.run(argv, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


@pytest.fixture
def sources(tmp_path):
    paths = []
    for name, tone in (("base", 440), ("donor", 880)):
        path = tmp_path/f"{name}.mp4"
        run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=160x120:r=30:d=3",
             "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration=3",
             "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)])
        paths.append(path)
    return paths


def pitch(path, start):
    raw = run(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path), "-t", "0.2",
               "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"])
    samples = np.frombuffer(raw, dtype="<f4")
    return np.fft.rfftfreq(len(samples), 1/48000)[np.argmax(abs(np.fft.rfft(samples)))]


def packets(path):
    return json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_packets",
        "-show_data_hash", "sha256", "-show_entries", "packet=pts,dts,duration,data_hash", "-of", "json", str(path)]))["packets"]


@pytest.mark.parametrize("fade", [0, .05])
def test_splice_replaces_only_window_and_copies_all_video_packets(sources, tmp_path, fade):
    base, donor = sources
    result = audio.splice_audio(base, donor, tmp_path/"out.mp4", start_sec=1, end_sec=2, crossfade_sec=fade)
    assert packets(result.output_path) == packets(base)
    assert pitch(result.output_path, .3) == pytest.approx(440, abs=10)
    assert pitch(result.output_path, 1.3) == pytest.approx(880, abs=10)
    assert pitch(result.output_path, 2.3) == pytest.approx(440, abs=10)
    assert result.metadata.video_duration_sec == 3


def test_full_track_and_audio_only_donor(sources, tmp_path):
    donor = tmp_path/"donor.wav"
    run(["ffmpeg", "-v", "error", "-i", str(sources[1]), "-vn", "-c:a", "pcm_s16le", str(donor)])
    assert audio.probe_trusted_audio(donor).sample_rate == 48000
    result = audio.replace_full_track(sources[0], donor, tmp_path/"full.mp4", gain_db=-3)
    assert result.operation=="full_track_audio" and result.audio_fit is None
    assert pitch(result.output_path, .3) == pytest.approx(880, abs=10)


def test_audio_bounds_cancel_and_no_clobber(sources, tmp_path):
    dst = tmp_path/"out.mp4"
    for extra in ({"crossfade_sec": .6}, {"gain_db": 13}, {"donor_start_sec": 2.1}, {"start_sec": 1.000001}):
        args = {"start_sec": 1, "end_sec": 2, **extra}
        with pytest.raises(media.AssistantMediaError):
            audio.splice_audio(*sources, dst, **args)
        assert not dst.exists()
    with pytest.raises(media.AssistantMediaCancelled):
        audio.splice_audio(*sources, dst, start_sec=1, end_sec=2, cancelled=lambda: True)
    dst.write_bytes(b"keep")
    with pytest.raises(media.AssistantMediaError):
        audio.splice_audio(*sources, dst, start_sec=1, end_sec=2)
    assert dst.read_bytes() == b"keep"


def track():
    return subs.SubtitleTrack(cues=tuple(subs.SubtitleCue(cue_id=f"cue{i}", start_sec=i*.2, end_sec=(i+1)*.2,
        text=text) for i,text in enumerate(("First", "Second", "日本語", "{evil} \\N text"))))


def test_short_consecutive_cues_aicp247_and_text_tail_all_survive():
    value = track()
    ass = subs.build_ass(value, width=160, height=120, duration_sec=3)
    assert ass.count("Dialogue:") == 4
    for text in ("First", "Second", "日本語", "evil", "text"):
        assert text in ass
    assert "{evil}" not in ass
    assert "0:00:00.00,0:00:00.20" in ass


def test_track_edits_deletion_timing_style_and_remap():
    updated = subs.edit_track(track(), SubtitleEditOperation.model_validate({
        "changes": [{"action":"delete", "cue_id":"cue0"},
                    {"action":"update", "cue_id":"cue1", "start_sec":.1, "end_sec":.4, "text":"Fixed"}],
        "style":{"color_hex":"#FF0000", "alignment":"top"}}), duration_sec=3)
    assert [c.cue_id for c in updated.cues] == ["cue1", "cue2", "cue3"]
    assert updated.style.alignment == "top"
    mapped = subs.remap_track(updated, (media.TimelineSpan(.2,.5,0), media.TimelineSpan(.6,3,.3)))
    assert mapped.cues[0].start_sec == 0
    assert mapped.cues[0].text == "Fixed"
    assert len(mapped.cues) == 3


def test_no_silent_overlap_or_short_cue_loss():
    with pytest.raises(ValueError):
        subs.SubtitleTrack(cues=(subs.SubtitleCue(cue_id="a", start_sec=0, end_sec=.4, text="a"),
                                subs.SubtitleCue(cue_id="b", start_sec=.2, end_sec=.5, text="b")))
    with pytest.raises(ValueError):
        subs.SubtitleTrack(cues=(subs.SubtitleCue(cue_id="a", start_sec=0, end_sec=.001, text="a"),))


def test_clean_source_required_and_burn_preserves_audio(sources, tmp_path):
    with pytest.raises(media.AssistantMediaError):
        subs.burn_track(sources[0], tmp_path/"bad.mp4", track=track())
    ascii_track = subs.SubtitleTrack(cues=track().cues[:2])
    result = subs.burn_track(sources[0], tmp_path/"burn.mp4", track=ascii_track, clean_source_verified=True)
    assert result.metadata.video_duration_sec == 3
    assert pitch(result.output_path, 1) == pytest.approx(440, abs=10)


@pytest.mark.parametrize("width,height",[(160,120),(1080,1920),(2160,3840)])
def test_legacy_ass_parser_preserves_box_style_and_rejects_overrides(width,height):
    from app.captions import build_ass
    old = build_ass([{"start":0,"end":1,"text":"Hello"}],width=width,height=height,duration_sec=3)
    parsed = subs.parse_legacy_ass(old,width=width,height=height)
    assert parsed.cues[0].text == "Hello"
    assert parsed.style.bold and parsed.style.border_style == 3
    assert parsed.style.outline_alpha == 89
    recolored=subs.edit_track(parsed,SubtitleEditOperation(style={"color_hex":"#FF0000"}),duration_sec=3)
    assert recolored.style.model_dump(exclude={"color_hex"}) == parsed.style.model_dump(exclude={"color_hex"})
    for bad in (old.replace("Hello","{\\pos(0,0)}Hello"),old+"\n[Fonts]\nfontname:test",
                old.replace(f"PlayResX: {width}","PlayResX: 9999")):
        with pytest.raises((media.AssistantMediaError,ValueError)):
            subs.parse_legacy_ass(bad,width=width,height=height)


def test_acoustic_metrics_report_finite_levels_with_no_pronunciation_claim(sources):
    metrics = audio.audio_window_metrics(sources[0],start_sec=.2,end_sec=.8)
    assert -40<metrics["level_rms_dbfs"]<-10
    assert -40<metrics["boundary_rms_dbfs"]<-10
    assert metrics["clipped_fraction"] == 0


def test_asr_phrase_windows_are_estimates_and_never_pronunciation_verdicts():
    from app.assistant_asr import ASRTranscript,phrase_windows
    transcript=ASRTranscript(language="en",language_probability=.9,network_denied=True,max_rss_kib=1,
        words=[{"text":" Listen", "start_sec":1,"end_sec":1.2,"probability":.8},
               {"text":"ify", "start_sec":1.2,"end_sec":1.5,"probability":.7}])
    match=phrase_windows(transcript,"Listenify")[0]
    assert (match["start_sec"],match["end_sec"]) == (1,1.5)
    assert match["review_required"] and not match["pronunciation_verified"] and not match["speaker_verified"]


def _fit_donor(tmp_path,duration,*,quiet_end):
    path=tmp_path/f"donor-{duration}-{quiet_end}.wav"
    tone="0.15*sin(2*PI*880*t)"
    signal=f"if(lt(t,{duration-.1}),{tone},0)" if quiet_end else tone
    run(["ffmpeg","-v","error","-f","lavfi","-i",f"aevalsrc='{signal}':s=48000:d={duration}",
         "-c:a","pcm_s16le",str(path)])
    return path


def _samples(path):
    return np.frombuffer(run(["ffmpeg","-v","error","-i",str(path),"-vn","-ac","1","-ar","48000","-f","f32le","-"]),dtype="<f4")


@pytest.mark.parametrize("donor_duration",[.8,1.25])
def test_gentle_duration_fit_preserves_pitch_picture_and_outside_timeline(sources,tmp_path,donor_duration):
    donor=_fit_donor(tmp_path,donor_duration,quiet_end=True)
    result=audio.splice_audio(sources[0],donor,tmp_path/"fitted.mp4",start_sec=1,end_sec=2,
                              donor_duration_sec=donor_duration,crossfade_sec=.03)
    assert packets(result.output_path) == packets(sources[0])
    assert result.metadata.video_duration_sec == 3
    assert pitch(result.output_path,1.3) == pytest.approx(880,abs=10)
    assert pitch(result.output_path,.3) == pytest.approx(440,abs=10)
    assert pitch(result.output_path,2.3) == pytest.approx(440,abs=10)
    assert result.audio_fit.speed_ratio == donor_duration
    assert result.audio_fit.pad_samples+result.audio_fit.trim_samples<=2400
    original,output=_samples(sources[0]),_samples(result.output_path)
    for start,end in ((.2,.8),(2.2,2.8)):
        a,b=round(start*48000),round(end*48000)
        assert float(np.sqrt(np.mean((original[a:b]-output[a:b])**2)))<.002


@pytest.mark.parametrize("donor_duration",[.8,1.25])
def test_duration_fit_refuses_audible_trim_or_padding(sources,tmp_path,donor_duration):
    donor=_fit_donor(tmp_path,donor_duration,quiet_end=False)
    with pytest.raises(media.AssistantMediaError,match="audible"):
        audio.splice_audio(sources[0],donor,tmp_path/"unsafe-fit.mp4",start_sec=1,end_sec=2,
                           donor_duration_sec=donor_duration)
    assert not (tmp_path/"unsafe-fit.mp4").exists()


def test_duration_contract_uses_donor_own_coverage_and_preserves_legacy_default_serialization():
    from app.assistant_contracts import EditPlan,VersionManifest,AssetManifest,validate_plan_sources,PlanSourceError
    from uuid import uuid4
    run_id,version_id,asset_id,donor_id=map(str,(uuid4(),uuid4(),uuid4(),uuid4()))
    payload={"run_id":run_id,"base_version_id":version_id,"operations":[{"kind":"audio_splice","start_sec":1,"end_sec":2,
        "source":{"asset_id":asset_id,"version_id":donor_id}}]}
    plan=EditPlan.model_validate(payload)
    assert "donor_duration_sec" not in plan.model_dump()["operations"][0]
    payload["operations"][0]["donor_duration_sec"]=1.25
    plan=EditPlan.model_validate(payload)
    with pytest.raises(PlanSourceError,match="donor does not cover"):
        validate_plan_sources(plan,base_version=VersionManifest(version_id=version_id,run_id=run_id,duration_sec=3,status="succeeded",available=True),
            assets={asset_id:AssetManifest(asset_id=asset_id,version_id=donor_id,media_kind="audio",duration_sec=1.1,has_audio=True,available=True)})
    payload["operations"][0]["donor_duration_sec"]=1.3
    with pytest.raises(ValueError,match="ratio"):
        EditPlan.model_validate(payload)
    payload["operations"][0].update(start_sec=1,end_sec=1.3,donor_duration_sec=.24)
    assert EditPlan.model_validate(payload).operations[0].donor_duration_sec==.24


def test_synthetic_voice_duration_fit_retains_speech_envelope_and_boundary_timing(sources,tmp_path):
    available=run(["ffmpeg","-hide_banner","-filters"])
    if b" flite " not in available:
        pytest.skip("FFmpeg Flite is unavailable for synthetic speech")
    donor=tmp_path/"voice.wav"
    run(["ffmpeg","-v","error","-f","lavfi","-i","flite=text='Hello world':voice=slt",
         "-af","apad=pad_dur=0.2","-ar","48000","-c:a","pcm_s16le",str(donor)])
    duration=audio.probe_trusted_audio(donor).duration_sec
    target_samples=round(duration*48000/1.2)
    target=target_samples/48000
    assert target<2
    result=audio.splice_audio(sources[0],donor,tmp_path/"voice-fit.mp4",start_sec=1,end_sec=1+target,
                              donor_duration_sec=duration)
    assert packets(result.output_path)==packets(sources[0])
    assert pitch(result.output_path, .3)==pytest.approx(440,abs=10)
    assert pitch(result.output_path,1+target+.1)==pytest.approx(440,abs=10)
    before,after=_samples(donor),_samples(result.output_path)[48000:48000+target_samples]
    def envelope(samples):
        count=len(samples)//480
        return np.sqrt(np.mean(samples[:count*480].reshape(count,480)**2,axis=1))
    expected,observed=envelope(before),envelope(after)
    expected=np.interp(np.linspace(0,1,len(observed)),np.linspace(0,1,len(expected)),expected)
    assert np.corrcoef(expected,observed)[0,1]>.5
    assert result.audio_fit.review_required
