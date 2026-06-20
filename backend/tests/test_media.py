"""
Tests for backend/app/media.py.

All tests use small synthetic videos generated with ffmpeg lavfi sources
(testsrc + sine wave). No large external files required.

Run: pytest tests/test_media.py -v
"""

from __future__ import annotations

import os
import subprocess
import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module if ffmpeg is not available
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    try:
        r = subprocess.run(
            ["/usr/bin/ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False


if not _ffmpeg_available():
    pytest.skip("ffmpeg not available", allow_module_level=True)


from app.media import (  # noqa: E402 — must follow guard
    MediaError,
    MediaInfo,
    cut_clip,
    cut_segments,
    get_default_target,
    probe,
    stitch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_video(path: str, duration: float, width: int, height: int, fps: int,
                with_audio: bool = True) -> None:
    """Generate a synthetic test video at *path* using ffmpeg lavfi sources."""
    cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
    ]
    if with_audio:
        cmd += [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-an",  # no audio
        ]
    cmd.append(path)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not create test video: {result.stderr.decode()[-500:]}"
        )


@pytest.fixture(scope="module")
def video_with_audio(tmp_path_factory):
    """2-second 320x240@30fps video with audio."""
    p = tmp_path_factory.mktemp("media") / "src_audio.mp4"
    _make_video(str(p), duration=2.0, width=320, height=240, fps=30, with_audio=True)
    return str(p)


@pytest.fixture(scope="module")
def video_no_audio(tmp_path_factory):
    """2-second 320x240@30fps video WITHOUT audio."""
    p = tmp_path_factory.mktemp("media") / "src_no_audio.mp4"
    _make_video(str(p), duration=2.0, width=320, height=240, fps=30, with_audio=False)
    return str(p)


@pytest.fixture(scope="module")
def small_clip_24fps(tmp_path_factory):
    """1-second 256x144@24fps clip (simulating Seedance downsize)."""
    p = tmp_path_factory.mktemp("media") / "seedance_out.mp4"
    _make_video(str(p), duration=1.0, width=256, height=144, fps=24, with_audio=False)
    return str(p)


# ---------------------------------------------------------------------------
# probe() tests
# ---------------------------------------------------------------------------

class TestProbe:
    def test_duration_approx(self, video_with_audio):
        info = probe(video_with_audio)
        assert abs(info.duration_sec - 2.0) <= 0.2, f"duration={info.duration_sec}"

    def test_resolution(self, video_with_audio):
        info = probe(video_with_audio)
        assert info.width == 320
        assert info.height == 240

    def test_fps(self, video_with_audio):
        info = probe(video_with_audio)
        assert abs(info.fps - 30.0) < 0.01, f"fps={info.fps}"

    def test_has_audio_true(self, video_with_audio):
        info = probe(video_with_audio)
        assert info.has_audio is True

    def test_has_audio_false(self, video_no_audio):
        info = probe(video_no_audio)
        assert info.has_audio is False

    def test_aspect_ratio(self, video_with_audio):
        info = probe(video_with_audio)
        # 320x240 -> 4:3
        assert info.aspect_ratio == "4:3"

    def test_returns_media_info(self, video_with_audio):
        info = probe(video_with_audio)
        assert isinstance(info, MediaInfo)

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(MediaError):
            probe(str(tmp_path / "nonexistent.mp4"))


# ---------------------------------------------------------------------------
# cut_clip() tests
# ---------------------------------------------------------------------------

class TestCutClip:
    def test_output_duration(self, video_with_audio, tmp_path):
        dst = str(tmp_path / "cut.mp4")
        cut_clip(video_with_audio, start_sec=0.0, end_sec=1.0, dst=dst)
        info = probe(dst)
        assert abs(info.duration_sec - 1.0) <= 0.2, f"cut duration={info.duration_sec}"

    def test_file_created(self, video_with_audio, tmp_path):
        dst = str(tmp_path / "cut.mp4")
        cut_clip(video_with_audio, start_sec=0.2, end_sec=1.5, dst=dst)
        assert os.path.exists(dst)
        assert os.path.getsize(dst) > 0

    def test_invalid_range_raises(self, video_with_audio, tmp_path):
        dst = str(tmp_path / "bad.mp4")
        with pytest.raises(MediaError):
            cut_clip(video_with_audio, start_sec=1.5, end_sec=0.5, dst=dst)

    def test_cut_to_near_eof(self, video_with_audio, tmp_path):
        """Cut to near end-of-file must not produce a near-empty file."""
        dst = str(tmp_path / "cut_eof.mp4")
        cut_clip(video_with_audio, start_sec=1.5, end_sec=2.0, dst=dst)
        assert os.path.exists(dst)
        assert os.path.getsize(dst) > 1000, "File suspiciously small near EOF"


# ---------------------------------------------------------------------------
# cut_segments() tests
# ---------------------------------------------------------------------------

class TestCutSegments:
    def test_returns_ordered_paths(self, video_with_audio, tmp_path):
        out_dir = str(tmp_path / "segs")
        paths = cut_segments(video_with_audio, [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)], out_dir)
        assert len(paths) == 3
        assert paths[0].endswith("seg_000.mp4")
        assert paths[1].endswith("seg_001.mp4")
        assert paths[2].endswith("seg_002.mp4")

    def test_all_files_exist(self, video_with_audio, tmp_path):
        out_dir = str(tmp_path / "segs2")
        paths = cut_segments(video_with_audio, [(0.0, 1.0), (1.0, 2.0)], out_dir)
        for p in paths:
            assert os.path.exists(p)


# ---------------------------------------------------------------------------
# stitch() tests  — KEY: normalization of mixed resolution/fps
# ---------------------------------------------------------------------------

class TestStitch:
    def test_stitch_normalizes_resolution_and_fps(
        self, video_with_audio, small_clip_24fps, tmp_path
    ):
        """
        Stitch a 320x240@30fps clip and a 256x144@24fps clip (mimicking a
        Seedance-processed segment) into a 320x240@30fps output.

        Verifies:
        - Output resolution == target (320x240)
        - Output fps == target (30)
        - Output duration ≈ sum of clip durations (± 0.3s)
        """
        # First clip: 1-second segment from source (320x240 @ 30fps)
        clip_a = str(tmp_path / "clip_a.mp4")
        cut_clip(video_with_audio, start_sec=0.0, end_sec=1.0, dst=clip_a)

        # clip_b is the 256x144@24fps synthetic clip (already a fixture)
        clip_b = small_clip_24fps

        dst = str(tmp_path / "stitched.mp4")
        stitch(
            clips=[clip_a, clip_b],
            audio_source=video_with_audio,
            dst=dst,
            width=320,
            height=240,
            fps=30.0,
        )

        out = probe(dst)
        assert out.width == 320, f"width={out.width}"
        assert out.height == 240, f"height={out.height}"
        assert abs(out.fps - 30.0) < 0.5, f"fps={out.fps}"
        # clip_a ≈ 1s, clip_b ≈ 1s → total ≈ 2s; audio_source is 2s so -shortest kicks in
        assert abs(out.duration_sec - 2.0) <= 0.3, f"duration={out.duration_sec}"

    def test_stitch_output_file_created(self, video_with_audio, tmp_path):
        clip = str(tmp_path / "c.mp4")
        cut_clip(video_with_audio, start_sec=0.0, end_sec=1.0, dst=clip)
        dst = str(tmp_path / "out.mp4")
        stitch([clip], audio_source=video_with_audio, dst=dst, width=320, height=240, fps=30.0)
        assert os.path.exists(dst)
        assert os.path.getsize(dst) > 0

    def test_stitch_empty_clips_raises(self, video_with_audio, tmp_path):
        with pytest.raises(MediaError):
            stitch([], audio_source=video_with_audio, dst=str(tmp_path / "x.mp4"),
                   width=320, height=240, fps=30.0)

    def test_stitch_invalid_audio_mode_raises(self, video_with_audio, tmp_path):
        clip = str(tmp_path / "c.mp4")
        cut_clip(video_with_audio, start_sec=0.0, end_sec=1.0, dst=clip)
        with pytest.raises(ValueError, match="audio_mode"):
            stitch([clip], audio_source=video_with_audio, dst=str(tmp_path / "x.mp4"),
                   width=320, height=240, fps=30.0, audio_mode="invalid")


class TestStitchSeedanceMode:
    """Tests for stitch() with audio_mode='seedance'."""

    @pytest.fixture(scope="class")
    def clip_a_with_audio(self, tmp_path_factory, video_with_audio):
        """1-second clip cut from the 2-second source — has audio."""
        p = tmp_path_factory.mktemp("seedance_a") / "clip_a.mp4"
        cut_clip(video_with_audio, 0.0, 1.0, str(p))
        return str(p)

    @pytest.fixture(scope="class")
    def clip_b_diff_res_with_audio(self, tmp_path_factory):
        """1-second 256x144@24fps clip WITH audio (simulating Seedance result)."""
        p = tmp_path_factory.mktemp("seedance_b") / "clip_b.mp4"
        _make_video(str(p), duration=1.0, width=256, height=144, fps=24, with_audio=True)
        return str(p)

    @pytest.fixture(scope="class")
    def clip_c_no_audio(self, tmp_path_factory):
        """1-second 320x240@30fps clip WITHOUT audio (edge-case: clip has no audio stream)."""
        p = tmp_path_factory.mktemp("seedance_c") / "clip_c.mp4"
        _make_video(str(p), duration=1.0, width=320, height=240, fps=30, with_audio=False)
        return str(p)

    def test_seedance_mode_duration_and_has_audio(
        self, clip_a_with_audio, clip_b_diff_res_with_audio, tmp_path
    ):
        """
        stitch(..., audio_mode='seedance') with two clips (each WITH audio, different
        res/fps) produces an output whose duration approximates the sum of clip
        durations and that has an audio stream.
        """
        dst = str(tmp_path / "seedance_out.mp4")
        stitch(
            clips=[clip_a_with_audio, clip_b_diff_res_with_audio],
            audio_source=clip_a_with_audio,  # ignored in seedance mode
            dst=dst,
            width=320,
            height=240,
            fps=30.0,
            audio_mode="seedance",
        )
        out = probe(dst)
        # Both clips ≈ 1s each → total ≈ 2s (±0.4s tolerance for encoding)
        assert abs(out.duration_sec - 2.0) <= 0.4, f"duration={out.duration_sec}"
        assert out.has_audio, "Output should have an audio stream in seedance mode"
        assert out.width == 320
        assert out.height == 240

    def test_seedance_mode_clip_without_audio_produces_valid_output(
        self, clip_a_with_audio, clip_c_no_audio, tmp_path
    ):
        """
        When one clip has no audio stream, stitch should synthesise silence for it
        and produce a valid output file that still has audio (from the other clip
        or from synthesised silence).
        """
        dst = str(tmp_path / "seedance_silence.mp4")
        stitch(
            clips=[clip_a_with_audio, clip_c_no_audio],
            audio_source=clip_a_with_audio,  # ignored in seedance mode
            dst=dst,
            width=320,
            height=240,
            fps=30.0,
            audio_mode="seedance",
        )
        out = probe(dst)
        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        assert out.has_audio, "Output should have audio even when one clip had none"
        assert abs(out.duration_sec - 2.0) <= 0.4, f"duration={out.duration_sec}"

    def test_seedance_mode_file_created(self, clip_a_with_audio, tmp_path):
        """Single clip in seedance mode produces a non-empty output file."""
        dst = str(tmp_path / "single_seedance.mp4")
        stitch(
            clips=[clip_a_with_audio],
            audio_source=clip_a_with_audio,
            dst=dst,
            width=320,
            height=240,
            fps=30.0,
            audio_mode="seedance",
        )
        assert os.path.exists(dst)
        assert os.path.getsize(dst) > 0


# ---------------------------------------------------------------------------
# get_default_target() tests
# ---------------------------------------------------------------------------

class TestGetDefaultTarget:
    def test_returns_source_dimensions(self, video_with_audio):
        info = probe(video_with_audio)
        w, h, fps = get_default_target(info)
        assert w == 320
        assert h == 240
        assert abs(fps - 30.0) < 0.01

    def test_returns_tuple(self, video_with_audio):
        info = probe(video_with_audio)
        result = get_default_target(info)
        assert isinstance(result, tuple)
        assert len(result) == 3
