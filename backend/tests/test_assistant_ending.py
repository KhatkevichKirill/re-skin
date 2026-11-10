"""Observable bounded ending extension with owned real media, no providers."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from app import assistant_media as media
from app.assistant_ending import extend_ending, resolve_ending_timing


def run(argv):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    return result.stdout


def video(path, *, audio=True, fps='24', frames=72, short_audio=False, audio_gap=.04):
    duration=float(Fraction(frames,1)/Fraction(fps))
    args=['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i',f'color=c=white:s=160x120:r={fps}:d={duration:.12f}']
    if audio:
        args+=['-f','lavfi','-i',f'sine=frequency=440:sample_rate=48000:duration={duration-audio_gap if short_audio else duration:.12f}']
    args+=['-vf',"drawbox=w=iw:h=ih:c=red:t=fill:enable='gte(t,1)',drawbox=w=iw:h=ih:c=blue:t=fill:enable='gte(t,2)',setsar=4/3",
        '-frames:v',str(frames),'-c:v','libx264','-preset','fast','-crf','18','-threads','1','-pix_fmt','yuv420p']
    args+=['-c:a','aac'] if audio else ['-an']
    run([*args,str(path)])
    return path


def donor(path, duration=6):
    signal='0.2*sin(2*PI*if(lt(t,3),880,1320)*t)'
    run(['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i',f"aevalsrc='{signal}':s=48000:d={duration}",
        '-c:a','pcm_s16le',str(path)])
    return path


@pytest.fixture
def inputs(tmp_path):
    return video(tmp_path/'source.mp4'),donor(tmp_path/'donor.wav')


def pixels(path,index):
    raw=run(['ffmpeg','-v','error','-i',str(path),'-vf',f'select=eq(n\\,{index}),format=rgb24','-frames:v','1',
        '-vsync','0','-f','rawvideo','-'])
    return np.frombuffer(raw,dtype=np.uint8).reshape(120,160,3)


def samples(path):
    return np.frombuffer(run(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar','48000','-f','f32le','-']),dtype='<f4')


def pitch(data,start,end):
    chunk=data[round(start*48000):round(end*48000)]
    return np.argmax(np.abs(np.fft.rfft(chunk)))*48000/len(chunk)


@pytest.mark.parametrize('kind', ['black','freeze_selected'])
def test_prefix_tail_tones_duration_frames_and_source_identity(inputs,tmp_path,kind):
    src,audio=inputs;before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    output=tmp_path/'result.mp4'
    result=extend_ending(src,audio,output,start_sec=2,end_sec=4,donor_start_sec=2,
        visual_kind=kind,freeze_timestamp_sec=.5 if kind=='freeze_selected' else None)
    assert result.operation=='extend_ending' and result.metadata.has_audio
    assert result.metadata.video_duration_sec==4
    assert result.ending_timing.end_frame==96 and result.ending_timing.tail_samples==96000
    assert result.base_timeline==(media.TimelineSpan(0,2,0),)
    assert [item.sha256 for item in result.sources]==before
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]==before
    assert np.abs(pixels(src,36).astype(float)-pixels(output,36)).mean()<2
    tail=pixels(output,84)
    if kind=='black':assert tail.max()<8
    else:assert tail.min()>240
    pcm=samples(output)
    assert abs(pitch(pcm,.3,.8)-440)<3
    assert abs(pitch(pcm,2.2,2.8)-880)<3
    assert abs(pitch(pcm,3.2,3.8)-1320)<3
    assert len(pcm)>=192000 and len(pcm)<193024
    raw=run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(output)])
    assert int(raw)==96


def test_frozen_final_fade_reaches_black(inputs,tmp_path):
    src,audio=inputs
    result=extend_ending(src,audio,tmp_path/'fade.mp4',start_sec=2,end_sec=4,donor_start_sec=2,
        visual_kind='freeze_selected',freeze_timestamp_sec=.5,fade_out_sec=.5)
    assert pixels(result.output_path,60).mean()>240
    assert pixels(result.output_path,89).mean()>100
    assert pixels(result.output_path,95).max()<8


def test_silent_base_gets_silent_prefix_and_real_donor_tail(tmp_path):
    src=video(tmp_path/'silent.mp4',audio=False);audio=donor(tmp_path/'donor.wav')
    result=extend_ending(src,audio,tmp_path/'out.mp4',start_sec=2,end_sec=4,donor_start_sec=2,visual_kind='black')
    pcm=samples(result.output_path)
    assert np.max(np.abs(pcm[:48000]))<1e-5
    assert abs(pitch(pcm,3.2,3.8)-1320)<3


def test_fractional_fps_uses_one_canonical_frame_and_sample_resolution(tmp_path):
    src=video(tmp_path/'fractional.mp4',fps='30000/1001',frames=90)
    audio=donor(tmp_path/'donor.wav')
    start=61/(30000/1001);end=121/(30000/1001)
    result=extend_ending(src,audio,tmp_path/'fractional-out.mp4',start_sec=start,end_sec=end,
        donor_start_sec=1,visual_kind='freeze_selected',freeze_timestamp_sec=10/(30000/1001))
    timing=result.ending_timing
    assert timing.start_frame==61 and timing.end_frame==121
    assert timing.start_sample==round(start*48000) and timing.end_sample==round(end*48000)
    assert result.metadata.fps=='30000/1001'
    assert abs(result.metadata.video_duration_sec-end)<1e-6


@pytest.mark.parametrize('changes', [
    {'start_sec':0},{'start_sec':3},{'end_sec':3},{'end_sec':121},
    {'end_sec':13.5},{'start_sec':.5,'end_sec':11}, {'start_sec':2.01},
    {'start_sec':True},{'end_sec':float('nan')},{'donor_start_sec':'1'},
    {'visual_kind':'other'},{'freeze_timestamp_sec':.5}, {'fade_out_sec':.5},
    {'visual_kind':'freeze_selected'}, {'visual_kind':'freeze_selected','freeze_timestamp_sec':2},
    {'visual_kind':'freeze_selected','freeze_timestamp_sec':.5,'fade_out_sec':1/24},
    {'visual_kind':'freeze_selected','freeze_timestamp_sec':.5,'fade_out_sec':1e-8},
])
def test_invalid_bounds_shape_and_types_are_rejected(inputs,tmp_path,changes):
    src,audio=inputs
    options=dict(start_sec=2,end_sec=4,donor_start_sec=2,visual_kind='black')
    options.update(changes)
    with pytest.raises(media.AssistantMediaError):
        extend_ending(src,audio,tmp_path/'invalid.mp4',**options)
    assert not (tmp_path/'invalid.mp4').exists()


def test_post_rounding_caps_are_not_extended_by_frame_tolerance():
    info=media.MediaMetadata(111,111,160,120,'30000/1001','1:1',True)
    # 120s is accepted as a raw cap but is not a frame boundary at this FPS.
    with pytest.raises(media.AssistantMediaError):
        resolve_ending_timing(info,start_sec=110.11,end_sec=120,donor_start_sec=0,visual_kind='black',donor_audio_duration_sec=120)
    info=replace(info,video_duration_sec=100, fps='24')
    with pytest.raises(media.AssistantMediaError):
        resolve_ending_timing(info,start_sec=99,end_sec=110.000001,donor_start_sec=0,visual_kind='black',donor_audio_duration_sec=120)


def test_actual_audio_extent_not_video_container_controls_donor_coverage(inputs,tmp_path):
    src,_=inputs
    short=video(tmp_path/'short-audio.mp4',short_audio=True)
    assert media.probe_trusted_media(short).video_duration_sec==3
    with pytest.raises(media.AssistantMediaError,match='actual audio'):
        extend_ending(src,short,tmp_path/'short-out.mp4',start_sec=2,end_sec=4,donor_start_sec=1,visual_kind='black')


def test_cancel_output_exists_and_decoded_donor_shortage(inputs,tmp_path,monkeypatch):
    from app import assistant_ending as ending
    src,audio=inputs
    with pytest.raises(media.AssistantMediaCancelled):
        extend_ending(src,audio,tmp_path/'cancel.mp4',start_sec=2,end_sec=4,donor_start_sec=2,visual_kind='black',cancelled=lambda:True)
    occupied=tmp_path/'exists.mp4';occupied.write_bytes(b'keep')
    with pytest.raises(media.AssistantMediaError):
        extend_ending(src,audio,occupied,start_sec=2,end_sec=4,donor_start_sec=2,visual_kind='black')
    assert occupied.read_bytes()==b'keep'
    real=ending.probe_trusted_audio
    monkeypatch.setattr(ending,'probe_trusted_audio',lambda *a,**k:replace(real(*a,**k),duration_sec=10))
    with pytest.raises(media.AssistantMediaError,match='exact required samples'):
        extend_ending(src,audio,tmp_path/'no-pad.mp4',start_sec=2,end_sec=4,donor_start_sec=5,visual_kind='black')

def test_caps_are_rechecked_after_near_integer_frame_resolution():
    fps=Fraction(47999999,2000000)
    info=media.MediaMetadata(float(2664/fps),float(2664/fps),160,120,str(fps),'1:1',True)
    with pytest.raises(media.AssistantMediaError,match='canonical ending frame'):
        resolve_ending_timing(info,start_sec=110.5,end_sec=120,donor_start_sec=0,visual_kind='black',donor_audio_duration_sec=120)
    fps=Fraction(299999999,10000000)
    info=replace(info,video_duration_sec=float(3000/fps),fps=str(fps))
    with pytest.raises(media.AssistantMediaError,match='canonical ending frame'):
        resolve_ending_timing(info,start_sec=99,end_sec=109,donor_start_sec=0,visual_kind='black',donor_audio_duration_sec=120)
    fps=Fraction(47999999,2000000)
    info=replace(info,video_duration_sec=float(72/fps),fps=str(fps))
    with pytest.raises(media.AssistantMediaError,match='canonical ending fade'):
        resolve_ending_timing(info,start_sec=2,end_sec=4,donor_start_sec=0,visual_kind='freeze_selected',
            freeze_timestamp_sec=.5,fade_out_sec=2,donor_audio_duration_sec=120)


def test_padding_old_base_audio_gap_keeps_donor_seam_time(tmp_path):
    src=video(tmp_path/'base-gap.mp4',short_audio=True,audio_gap=.049);audio=donor(tmp_path/'donor.wav')
    from app.assistant_audio import probe_trusted_audio
    assert probe_trusted_audio(src).duration_sec<71/24
    result=extend_ending(src,audio,tmp_path/'base-gap-out.mp4',start_sec=71/24,end_sec=4,
        donor_start_sec=3,visual_kind='black')
    pcm=samples(result.output_path)
    assert abs(pitch(pcm,3.1,3.8)-1320)<3
    assert result.ending_timing.start_sample==142000
    assert result.ending_timing.tail_samples==50000


def test_changed_original_donor_during_decode_refuses_final_output(inputs,tmp_path,monkeypatch):
    from app import assistant_ending as ending
    src,audio=inputs
    real=ending._validated_audio_clip
    def change_donor(*args,**kwargs):
        real(*args,**kwargs)
        with audio.open('ab') as handle:handle.write(b'changed')
    monkeypatch.setattr(ending,'_validated_audio_clip',change_donor)
    with pytest.raises(media.AssistantMediaError,match='source changed'):
        extend_ending(src,audio,tmp_path/'changed-out.mp4',start_sec=2,end_sec=4,
            donor_start_sec=2,visual_kind='black')
    assert not (tmp_path/'changed-out.mp4').exists()

def test_mid_operation_cancellation_leaves_no_final_output(inputs,tmp_path,monkeypatch):
    from app import assistant_ending as ending
    src,audio=inputs;state={'cancelled':False};real=ending._validated_audio_clip
    def cancel_after_decode(*args,**kwargs):
        real(*args,**kwargs);state['cancelled']=True
    monkeypatch.setattr(ending,'_validated_audio_clip',cancel_after_decode)
    with pytest.raises(media.AssistantMediaCancelled):
        extend_ending(src,audio,tmp_path/'mid-cancel.mp4',start_sec=2,end_sec=4,donor_start_sec=2,
            visual_kind='black',cancelled=lambda:state['cancelled'])
    assert not (tmp_path/'mid-cancel.mp4').exists()
    assert not list(tmp_path.glob('.assistant-ending-*'))


def test_exact_120s_output_cap_is_valid_real_media(tmp_path):
    src=video(tmp_path/'long-silent.mp4',audio=False,frames=119*24)
    audio=donor(tmp_path/'donor.wav')
    result=extend_ending(src,audio,tmp_path/'exact-cap.mp4',start_sec=118,end_sec=120,
        donor_start_sec=2,visual_kind='black')
    assert result.metadata.video_duration_sec==120
    assert result.metadata.duration_sec<=120
    assert result.ending_timing.end_frame==2880
    assert result.ending_timing.end_sample==5760000


def test_exact_audio_clock_option_rejects_truthy_nonbool_before_io(tmp_path):
    info=media.MediaMetadata(1,1,160,120,'24','1:1',False)
    with pytest.raises(media.AssistantMediaError,match='must be a boolean'):
        media._render('invalid',(),tmp_path/'unused.mp4',info,'',1,(),'','',media._budget(1,None),exact_audio_timing=1)


@pytest.mark.parametrize('fps,frames,start_frame,end_frame', [('30',90,60,120),('24000/1001',72,48,96)])
def test_actual_aac_video_donor_same_path_keeps_complete_tail(tmp_path,fps,frames,start_frame,end_frame):
    from app.assistant_audio import probe_trusted_audio
    src=video(tmp_path/'same-aac-source.mp4',fps=fps,frames=frames)
    original_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    rate=Fraction(fps);start=float(Fraction(start_frame,1)/rate);end=float(Fraction(end_frame,1)/rate)
    result=extend_ending(src,src,tmp_path/'same-aac-out.mp4',start_sec=start,end_sec=end,donor_start_sec=0,
        visual_kind='freeze_selected',freeze_timestamp_sec=float(Fraction(15,1)/rate))
    assert result.ending_timing.end_frame==end_frame
    assert abs(probe_trusted_audio(result.output_path).duration_sec-result.ending_timing.end_sample/48000)<=1/48000+1e-8
    assert result.metadata.fps==str(rate)
    assert abs(result.metadata.video_duration_sec-end)<1e-6
    pcm=samples(result.output_path)
    assert abs(pitch(pcm,start+.2,start+.7)-440)<3
    assert abs(pitch(pcm,end-.4,end-.1)-440)<4
    assert hashlib.sha256(src.read_bytes()).hexdigest()==original_hash
    assert [item.sha256 for item in result.sources]==[original_hash,original_hash]


def test_44100_long_prefix_keeps_canonical_48k_output_clock(tmp_path,monkeypatch):
    """Real incident shape: resampled25s prefix used to add two timestamp samples."""
    from app import assistant_ending as ending
    from app.assistant_audio import probe_trusted_audio
    src=tmp_path/'base-44100.mp4';audio=tmp_path/'donor-48000.mp4'
    def make(path,frames,audio_rate,audio_duration,frequency):
        run(['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i',
            f'color=c=white:s=160x120:r=24:d={frames/24:.12f}',
            '-f','lavfi','-i',f'sine=frequency={frequency}:sample_rate={audio_rate}:duration={audio_duration}',
            '-frames:v',str(frames),'-c:v','libx264','-preset','fast','-crf','18','-threads','1',
            '-pix_fmt','yuv420p','-af',"asetpts=PTS+if(gte(T\\,12)\\,2\\,0)" if audio_rate==44100 else 'anull',
            '-c:a','aac','-ac','2','-ar',str(audio_rate),str(path)])
    # A zero-start AAC stream can have a tiny packet-clock discontinuity from
    # previous edits. Its decoded samples remain continuous; the supported
    # media probe does not require an audio-packet CFR clock. Model two native
    # timestamp ticks after 12s, then preserve that history through a prior cut.
    uncut=tmp_path/'before-cut-44100.mp4'
    make(uncut,619,44100,25.773991,440)
    media.cut_section(uncut,src,start_sec=169/24,end_sec=171/24)
    make(audio,640,48000,26.65,880)
    assert probe_trusted_audio(src).sample_rate==44100
    assert probe_trusted_audio(audio).sample_rate==48000
    originals=[hashlib.sha256(path.read_bytes()).hexdigest() for path in (src,audio)]
    observed={};validate=ending._validate_ending_output
    def measured(output,timing,budget,**kwargs):
        actual=probe_trusted_audio(output)
        observed.update(sample_rate=actual.sample_rate,duration_sec=actual.duration_sec,
            expected_end_samples=timing.end_sample,expected_end_sec=timing.end_sample/48000)
        return validate(output,timing,budget,**kwargs)
    monkeypatch.setattr(ending,'_validate_ending_output',measured)
    try:
        result=ending.extend_ending(src,audio,tmp_path/'canonical-clock.mp4',start_sec=25,end_sec=26.5,
            donor_start_sec=25,visual_kind='black')
    except media.AssistantMediaError as error:
        pytest.fail(f'{error}; actual audio clock: {observed}')
    assert result.ending_timing.end_sample==1272000
    assert observed['sample_rate']==48000
    assert observed['duration_sec']==26.5
    assert result.metadata.video_duration_sec==26.5
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (src,audio)]==originals
    pcm=samples(result.output_path)
    assert abs(pitch(pcm,25.5,26.4)-880)<3
