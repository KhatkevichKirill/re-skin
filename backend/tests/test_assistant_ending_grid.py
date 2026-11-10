"""Exact rational late-final grid classification; no decoder or external media."""
import copy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from app import assistant_media as media


@pytest.fixture
def grid(monkeypatch):
    count,clock,rate=892,Fraction(1,15360),Fraction(30)
    slots=list(range(count-1))+[count]
    data={'streams':[{'codec_type':'video','codec_name':'h264','pix_fmt':'yuv420p','avg_frame_rate':'30','r_frame_rate':'30',
        'sample_aspect_ratio':'1:1','duration':'29.733333','duration_ts':456704,'time_base':'1/15360','nb_frames':'892',
        'start_time':'0','width':160,'height':120},
        {'codec_type':'audio','codec_name':'aac','duration':'29.820998','duration_ts':1315106,'time_base':'1/44100','start_time':'0'}],
        'format':{'duration':'29.820998','format_name':'mov,mp4,m4a'}}
    packets=[{'pts':int(slot/(clock*rate)),'pts_time':f'{float(Fraction(slot)/rate):.6f}','duration':512} for slot in slots]
    def run(argv,*args,**kwargs):
        return json.dumps({'packets':packets}) if '-show_packets' in argv else json.dumps(data)
    monkeypatch.setattr(media,'_run',run)
    return data,packets


def inspect():
    return media._probe(Path('/isolated/fixture.mp4'),media._budget(30,None))


def test_exact_grid_gap_gets_distinct_readonly_profile_with_truthful_counts(grid):
    with pytest.raises(media.EditingPreparationRequired) as error:inspect()
    timing=error.value.preparation
    assert timing['kind']=='audio_tail_grid_gap'
    assert timing['gap_fill_frames']==1 and timing['tail_hold_frames']==2
    assert timing['added_frames']==3 and timing['output_frame_count']==895
    assert timing['gap_start_sec']==float(Fraction(891,30))
    assert timing['source_final_pts_sec']==timing['source_declared_duration_sec']==float(Fraction(892,30))
    assert error.value.info.video_duration_sec==float(Fraction(892,30))


def test_grid_gap_is_never_editable_when_audio_difference_is_below_fifty_ms(grid):
    data,_=grid
    data['streams'][1].update(duration='29.753333',duration_ts=1312122)
    with pytest.raises(media.EditingPreparationRequired) as error:inspect()
    assert error.value.preparation['gap_fill_frames']==1
    assert error.value.preparation['tail_hold_frames']==0
    assert error.value.preparation['output_frame_count']==893


@pytest.mark.parametrize('mutation', ['off_grid','duplicate','interior_gap','bad_end','bad_count','zero_packet_duration',
    'negative_tail','excess_tail','no_audio','nonfinite_pts'])
def test_other_grid_shapes_and_audio_profiles_are_rejected(grid,mutation):
    data,packets=grid
    if mutation=='off_grid':packets[100]['pts']+=1
    elif mutation=='duplicate':packets[100]['pts']=packets[99]['pts']
    elif mutation=='interior_gap':packets[100]['pts']+=512
    elif mutation=='bad_end':data['streams'][0]['duration_ts']+=1
    elif mutation=='bad_count':data['streams'][0]['nb_frames']='893'
    elif mutation=='zero_packet_duration':packets[-1]['duration']=0
    elif mutation=='negative_tail':data['streams'][1].update(duration='29.713333',duration_ts=1310358)
    elif mutation=='excess_tail':data['streams'][1].update(duration='29.833356',duration_ts=1315651)
    elif mutation=='no_audio':data['streams'].pop()
    else:packets[-1]['pts_time']='nan'
    with pytest.raises(media.AssistantMediaError) as error:inspect()
    assert not isinstance(error.value,media.EditingPreparationRequired)


def test_ordinary_cfr_tail_keeps_old_profile_without_new_raw_fields(grid):
    data,packets=grid
    for index,packet in enumerate(packets):
        packet.clear();packet['pts_time']=f'{index/30:.6f}'
    for key in ('duration_ts','time_base','nb_frames'):data['streams'][0].pop(key)
    with pytest.raises(media.EditingPreparationRequired) as error:inspect()
    assert error.value.preparation['kind']=='audio_tail'
    assert error.value.preparation['added_frames']==3
    assert 'gap_fill_frames' not in error.value.preparation


@pytest.mark.parametrize('shift', [-1, 15361, 153600000000])
def test_video_recovery_offset_is_bounded_before_decode_or_padding(monkeypatch,shift):
    from app import assistant_preparation as prep
    stream={'codec_name':'h264','profile':'High','pix_fmt':'yuv420p','width':160,'height':120,
        'sample_aspect_ratio':'1:1','extradata_hash':'SHA256:fixture','time_base':'1/15360'}
    before=[{'pts':0,'dts':0,'duration':512,'size':'1','data_hash':'SHA256:a'},
        {'pts':1024,'dts':512,'duration':512,'size':'1','data_hash':'SHA256:b'}]
    after=[{**packet,'pts':packet['pts']+shift,'dts':packet['dts']+shift} for packet in before]
    monkeypatch.setattr(prep,'_video_packets',lambda *args,ignore_editlist=False,**kwargs:(stream,after if ignore_editlist else before))
    monkeypatch.setattr(prep,'_decoded_picture',lambda *args,**kwargs:pytest.fail('out-of-bound clocks must not decode or pad'))
    with pytest.raises(media.AssistantMediaError,match='recovery proof failed'):
        prep._recover_video_clock(Path('/isolated/input.mp4'),{'source_video_frames':2},Fraction(30),media._budget(30,None))
