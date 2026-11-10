"""Real small FFmpeg fixtures for the trusted AICP-268 adapters.

No providers, live assets, DB writes or network access. These test observable
media behavior rather than the implementation's FFmpeg argument spelling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from app import assistant_media as media

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and ffprobe are required",
)


def _run(argv: list[str]) -> bytes:
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout


def _make_video(path: Path, *, audio: bool = True, color: str = "white", fps: str = "30", sar: str = "4/3") -> Path:
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "lavfi", "-i", f"color=c={color}:size=160x120:rate={fps}:duration=3"]
    if audio:
        # Each second has a distinct pitch, making cut/trim timing observable.
        signal = "0.2*sin(2*PI*if(lt(t,1),440,if(lt(t,2),880,1320))*t)"
        argv += ["-f", "lavfi", "-i", f"aevalsrc='{signal}':s=48000:d=3"]
    argv += ["-vf", f"setsar={sar}", "-c:v", "libx264", "-preset", "ultrafast",
             "-threads", "1", "-pix_fmt", "yuv420p"]
    argv += ["-c:a", "aac", "-b:a", "128k"] if audio else ["-an"]
    _run([*argv, str(path)])
    return path


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    return _make_video(tmp_path_factory.mktemp("assistant-media-source") / "source.mp4")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audio_packets(path: Path) -> list[dict]:
    return json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
        "-show_data_hash", "sha256", "-show_entries", "packet=pts,dts,duration,data_hash",
        "-of", "json", str(path),
    ]))["packets"]


def _frame_mean(path: Path, seconds: float) -> float:
    raw = _run([
        "ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path),
        "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    return float(np.frombuffer(raw, dtype=np.uint8).mean())


def _pitch(path: Path, seconds: float) -> float:
    raw = _run([
        "ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path),
        "-t", "0.2", "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "-",
    ])
    samples = np.frombuffer(raw, dtype="<f4")
    assert len(samples) >= 9000
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(len(samples), 1 / 48000)[np.argmax(spectrum)])


def _assert_geometry(result: media.MediaEditResult, *, fps: str = "30") -> None:
    assert (result.metadata.width, result.metadata.height) == (160, 120)
    assert result.metadata.fps == fps
    assert result.metadata.sample_aspect_ratio == "4:3"


def test_image_fade_preserves_audio_packets_timestamps_duration_and_original(source, tmp_path):
    original_hash = _sha(source)
    before_packets = _audio_packets(source)
    result = media.image_only_fade(source, tmp_path / "fade.mp4", start_sec=1, end_sec=3)
    _assert_geometry(result)
    assert result.metadata.duration_sec == pytest.approx(3, abs=0.001)
    assert result.metadata.video_duration_sec == pytest.approx(3, abs=0.001)
    assert _audio_packets(Path(result.output_path)) == before_packets
    assert _frame_mean(Path(result.output_path), 0.5) > 245
    assert 100 < _frame_mean(Path(result.output_path), 2) < 150
    assert _frame_mean(Path(result.output_path), 2.9) < 20
    assert _sha(source) == original_hash == result.sources[0].sha256
    assert result.output_sha256 == _sha(Path(result.output_path))
    assert result.audio_policy == "stream_copy_base"
    assert "Burned-in text fades" in result.subtitle_effect


def test_trim_keeps_exact_frame_range_and_corresponding_audio(source, tmp_path):
    original_hash = _sha(source)
    result = media.trim_video(source, tmp_path / "trim.mp4", start_sec=0.5, end_sec=2.5)
    output = Path(result.output_path)
    _assert_geometry(result)
    assert result.metadata.video_duration_sec == pytest.approx(2, abs=0.001)
    assert result.metadata.duration_sec == pytest.approx(2, abs=0.03)
    assert _pitch(output, 0.1) == pytest.approx(440, abs=10)
    assert _pitch(output, 0.7) == pytest.approx(880, abs=10)
    assert _pitch(output, 1.7) == pytest.approx(1320, abs=10)
    assert result.base_timeline == (media.TimelineSpan(0.5, 2.5, 0),)
    assert "time remapping" in result.subtitle_effect
    assert _sha(source) == original_hash


def test_end_trim_ends_picture_and_audio_at_selected_frame(source, tmp_path):
    result = media.trim_video(source, tmp_path / "end.mp4", end_sec=1.5)
    assert result.metadata.video_duration_sec == pytest.approx(1.5, abs=0.001)
    assert result.metadata.duration_sec == pytest.approx(1.5, abs=0.03)
    assert _pitch(Path(result.output_path), 1.2) == pytest.approx(880, abs=10)


def test_middle_cut_removes_same_audio_interval_without_later_drift(source, tmp_path):
    original_hash = _sha(source)
    result = media.cut_section(source, tmp_path / "cut.mp4", start_sec=1, end_sec=2)
    output = Path(result.output_path)
    _assert_geometry(result)
    assert result.metadata.video_duration_sec == pytest.approx(2, abs=0.001)
    assert result.metadata.duration_sec == pytest.approx(2, abs=0.03)
    assert _pitch(output, 0.2) == pytest.approx(440, abs=10)
    assert _pitch(output, 1.2) == pytest.approx(1320, abs=10)
    assert result.base_timeline == (media.TimelineSpan(0, 1, 0), media.TimelineSpan(2, 3, 1))
    assert _sha(source) == original_hash


def test_replacement_changes_picture_only_and_keeps_base_audio(source, tmp_path):
    donor = _make_video(tmp_path / "donor.mp4", color="black")
    before = (_sha(source), _sha(donor))
    result = media.replace_footage(source, donor, tmp_path / "replacement.mp4", start_sec=1, end_sec=2)
    output = Path(result.output_path)
    _assert_geometry(result)
    assert result.metadata.duration_sec == pytest.approx(3, abs=0.001)
    assert _audio_packets(output) == _audio_packets(source)
    assert _frame_mean(output, 0.5) > 245
    assert _frame_mean(output, 1.5) < 3
    assert _frame_mean(output, 2.5) > 245
    assert tuple(p.sha256 for p in result.sources) == before
    assert (_sha(source), _sha(donor)) == before
    assert "burned-in" in result.subtitle_effect


def test_silent_media_does_not_gain_an_audio_track(tmp_path):
    source = _make_video(tmp_path / "silent.mp4", audio=False)
    fade = media.image_only_fade(source, tmp_path / "fade.mp4", start_sec=1, end_sec=3)
    trim = media.trim_video(source, tmp_path / "trim.mp4", end_sec=2)
    cut = media.cut_section(source, tmp_path / "cut.mp4", start_sec=1, end_sec=2)
    assert not any(r.metadata.has_audio for r in (fade, trim, cut))


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (2, 1), (0, 4), (float("nan"), 2), (0, float("inf")), (True, 2)])
@pytest.mark.parametrize("operation", [media.image_only_fade, media.trim_video, media.cut_section])
def test_invalid_spans_never_publish(source, tmp_path, start, end, operation):
    destination = tmp_path / "invalid.mp4"
    before = _sha(source)
    with pytest.raises(media.AssistantMediaError):
        operation(source, destination, start_sec=start, end_sec=end)
    assert not destination.exists()
    assert _sha(source) == before


@pytest.mark.parametrize("operation", [media.trim_video, media.cut_section])
def test_timeline_changing_edits_reject_nonframe_boundaries(source, tmp_path, operation):
    with pytest.raises(media.AssistantMediaError, match="align to video frames"):
        operation(source, tmp_path / "invalid.mp4", start_sec=0.51, end_sec=2)


def test_input_and_existing_output_cannot_be_overwritten(source, tmp_path):
    original_hash = _sha(source)
    with pytest.raises(media.AssistantMediaError, match="overwriting"):
        media.image_only_fade(source, source, start_sec=1, end_sec=2)
    destination = tmp_path / "existing.mp4"
    destination.write_bytes(b"existing version")
    with pytest.raises(media.AssistantMediaError, match="overwriting"):
        media.trim_video(source, destination, end_sec=2)
    assert destination.read_bytes() == b"existing version"
    assert _sha(source) == original_hash


def test_cancellation_and_timeout_leave_no_output_or_staging(source, tmp_path):
    destination = tmp_path / "cancelled.mp4"
    with pytest.raises(media.AssistantMediaCancelled):
        media.image_only_fade(source, destination, start_sec=1, end_sec=2, cancelled=lambda: True)
    with pytest.raises(media.AssistantMediaTimeout, match="time limit"):
        media.trim_video(source, destination, end_sec=2, timeout_sec=0.000001)
    assert list(tmp_path.iterdir()) == []


def test_cancel_during_encoder_terminates_and_cleans_staging(source, tmp_path, monkeypatch):
    original_run = media._run
    encoding = False
    checks = 0

    def run(argv, budget, **access):
        nonlocal encoding
        encoding = argv[0] == media.FFMPEG
        return original_run(argv, budget, **access)

    def cancelled():
        nonlocal checks
        if encoding:
            checks += 1
        return checks >= 2

    monkeypatch.setattr(media, "_run", run)
    with pytest.raises(media.AssistantMediaCancelled):
        media.image_only_fade(source, tmp_path / "cancelled.mp4", start_sec=1, end_sec=3, cancelled=cancelled)
    assert list(tmp_path.iterdir()) == []


def test_replacement_rejects_mismatched_geometry_and_short_donor(source, tmp_path):
    donor = _make_video(tmp_path / "different.mp4", sar="1/1")
    with pytest.raises(media.AssistantMediaError, match="geometry"):
        media.replace_footage(source, donor, tmp_path / "invalid.mp4", start_sec=1, end_sec=2)
    with pytest.raises(media.AssistantMediaError, match="range must satisfy"):
        media.replace_footage(source, source, tmp_path / "invalid.mp4", start_sec=1, end_sec=2, donor_start_sec=2.5)


def test_fractional_frame_rate_is_preserved_without_a_30fps_cap(tmp_path):
    source = _make_video(tmp_path / "fractional.mp4", fps="60000/1001", audio=False)
    result = media.trim_video(source, tmp_path / "fractional-trim.mp4", end_sec=60*1001/60000)
    _assert_geometry(result, fps="60000/1001")
    assert result.metadata.video_duration_sec == pytest.approx(1.001, abs=0.00001)


def test_relative_paths_urls_and_nonmp4_output_are_rejected(source, tmp_path):
    with pytest.raises(media.AssistantMediaError, match="absolute"):
        media.trim_video("relative.mp4", tmp_path / "x.mp4", end_sec=2)
    with pytest.raises(media.AssistantMediaError, match="absolute"):
        media.trim_video("https://example.invalid/input.mp4", tmp_path / "x.mp4", end_sec=2)
    with pytest.raises(media.AssistantMediaError, match=".mp4"):
        media.trim_video(source, tmp_path / "x.mkv", end_sec=2)


def test_ending_fade_rejects_mid_video_end(source, tmp_path):
    with pytest.raises(media.AssistantMediaError, match="must end at the video duration"):
        media.image_only_fade(source, tmp_path / "invalid.mp4", start_sec=1, end_sec=2)
    assert not (tmp_path / "invalid.mp4").exists()


@pytest.mark.parametrize("operation", ["trim", "cut", "replacement", "fractional_trim"])
def test_filtered_video_retains_every_frame_duration_including_last(source, tmp_path, operation):
    """FFmpeg 7 drops filtered-frame durations under passthrough encoding.

    Counting frames alone misses the regression: the last frame's zero duration
    shortens the video track while audio still runs for the intended duration.
    """
    destination = tmp_path / "durations.mp4"
    if operation == "fractional_trim":
        source = _make_video(tmp_path / "fractional.mp4", fps="60000/1001", audio=False)
        result = media.trim_video(source, destination, end_sec=1.001)
        fps, frame_count, expected_end = Fraction(60000, 1001), 60, 1.001
    elif operation == "trim":
        result = media.trim_video(source, destination, start_sec=.5, end_sec=2.5)
        fps, frame_count, expected_end = Fraction(30), 60, 2
    elif operation == "cut":
        result = media.cut_section(source, destination, start_sec=1, end_sec=2)
        fps, frame_count, expected_end = Fraction(30), 60, 2
    else:
        donor = _make_video(tmp_path / "donor.mp4", color="black")
        result = media.replace_footage(source, donor, destination, start_sec=1, end_sec=2)
        fps, frame_count, expected_end = Fraction(30), 90, 3
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets", "-show_streams",
        "-show_entries", "packet=pts_time,duration_time:stream=avg_frame_rate,r_frame_rate,duration,nb_frames",
        "-of", "json", str(destination),
    ]))
    packets = sorted(data["packets"], key=lambda packet: float(packet["pts_time"]))
    stream = data["streams"][0]
    assert len(packets) == int(stream["nb_frames"]) == frame_count
    assert Fraction(stream["avg_frame_rate"]) == Fraction(stream["r_frame_rate"]) == fps
    for index, packet in enumerate(packets):
        assert float(packet["pts_time"]) == pytest.approx(float(index / fps), abs=0.000001)
        assert float(packet["duration_time"]) == pytest.approx(float(1 / fps), abs=0.000001)
    assert float(packets[-1]["pts_time"]) + float(packets[-1]["duration_time"]) == pytest.approx(expected_end, abs=0.000002)
    assert float(stream["duration"]) == pytest.approx(expected_end, abs=0.000001)
    assert result.metadata.video_duration_sec == pytest.approx(expected_end, abs=0.000001)


def test_dip_preserves_surrounding_picture_audio_duration_and_source(source, tmp_path):
    original = _sha(source)
    result = media.image_dip_to_black(source, tmp_path / 'dip.mp4', start_sec=.5, midpoint_sec=1.5, end_sec=2.5)
    output = Path(result.output_path)
    _assert_geometry(result)
    assert result.metadata.video_duration_sec == pytest.approx(3, abs=.001)
    assert _audio_packets(output) == _audio_packets(source)
    assert _sha(source) == original == result.sources[0].sha256
    assert result.base_timeline == (media.TimelineSpan(0, 3, 0),)
    for seconds in (.1, .4, 2.5, 2.9):
        assert _frame_mean(output, seconds) > 245
    assert 100 < _frame_mean(output, 1) < 155
    assert _frame_mean(output, 1.5) < 3
    assert 100 < _frame_mean(output, 2) < 155


@pytest.mark.parametrize('start,midpoint,end', [(1,1,2), (1,2,2), (1,2,4), (1,0,2), (0,float('nan'),2)])
def test_dip_rejects_invalid_ramps_without_publication(source, tmp_path, start, midpoint, end):
    dst = tmp_path / 'invalid-dip.mp4'
    with pytest.raises(media.AssistantMediaError):
        media.image_dip_to_black(source, dst, start_sec=start, midpoint_sec=midpoint, end_sec=end)
    assert not dst.exists()


@pytest.mark.parametrize('fps', ['30','60000/1001'])
def test_adjacent_two_frame_fades_reach_endpoints_and_restore_picture(tmp_path, fps):
    source = _make_video(tmp_path / 'source.mp4',fps=fps,audio=False)
    rate = float(Fraction(fps))
    result = media.image_fades(source,tmp_path / 'fades.mp4',fades=((10/rate,12/rate,'out'),(12/rate,14/rate,'in')))
    frames = np.frombuffer(_run(['ffmpeg','-v','error','-i',result.output_path,'-pix_fmt','gray','-f','rawvideo','-']),dtype=np.uint8).reshape(-1,120,160).mean(axis=(1,2))
    assert frames[9] > 245
    assert frames[10] > 245
    assert frames[11] < 3
    assert frames[12] < 3
    assert frames[13] > 245
    assert frames[14] > 245
    assert result.metadata.fps == fps
