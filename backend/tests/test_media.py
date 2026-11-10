"""
Tests for backend/app/media.py.

All tests use small synthetic videos generated with ffmpeg lavfi sources
(testsrc + sine wave). No large external files required.

Run: pytest tests/test_media.py -v
"""

from __future__ import annotations

import os
import re
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


import app.media as media_mod  # noqa: E402 — must follow guard
from app import ai_models  # noqa: E402 — voice-reference bounds live in the registry
from app.media import (  # noqa: E402 — must follow guard
    MediaError,
    MediaInfo,
    burn_subtitles,
    cut_clip,
    cut_segments,
    get_default_target,
    make_llm_proxy,
    mux_original_audio,
    normalize_source,
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
    # Force yuv420p: testsrc is RGB and ffmpeg would otherwise auto-pick
    # yuv444p, which normalize_source() treats as non-compliant.
    cmd += ["-pix_fmt", "yuv420p"]
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

    def test_pix_fmt(self, video_with_audio):
        info = probe(video_with_audio)
        assert info.pix_fmt == "yuv420p"


# ---------------------------------------------------------------------------
# normalize_source() tests
# ---------------------------------------------------------------------------

class TestNormalizeSource:
    def _make_444_video(self, path: str, width: int = 320, height: int = 240) -> None:
        """Synthetic video in yuv444p — a pix_fmt outside the 8-bit 4:2:0 baseline."""
        cmd = [
            "/usr/bin/ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration=1:size={width}x{height}:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv444p",
            "-an",
            path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not create yuv444p test video: {result.stderr.decode()[-500:]}"
            )

    def test_compliant_source_untouched(self, tmp_path):
        src = str(tmp_path / "source.mp4")
        _make_video(src, duration=1.0, width=320, height=240, fps=30, with_audio=True)
        before = os.path.getmtime(src)

        assert normalize_source(src) == src
        assert os.path.getmtime(src) == before

    def test_downscales_landscape(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media_mod, "_NORMALIZE_MAX_SHORT_SIDE", 120)
        src = str(tmp_path / "source.mp4")
        _make_video(src, duration=1.0, width=320, height=240, fps=30, with_audio=True)

        out = normalize_source(src)
        assert out == src
        info = probe(out)
        assert info.height == 120
        assert info.width == 160
        assert info.has_audio is True

    def test_downscales_portrait(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media_mod, "_NORMALIZE_MAX_SHORT_SIDE", 120)
        src = str(tmp_path / "source.mp4")
        _make_video(src, duration=1.0, width=240, height=320, fps=30, with_audio=False)

        out = normalize_source(src)
        info = probe(out)
        assert info.width == 120
        assert info.height == 160

    def test_converts_pix_fmt(self, tmp_path):
        src = str(tmp_path / "source.mp4")
        self._make_444_video(src)

        out = normalize_source(src)
        info = probe(out)
        assert info.pix_fmt == "yuv420p"
        assert info.width == 320
        assert info.height == 240

    def test_replaces_non_mp4_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media_mod, "_NORMALIZE_MAX_SHORT_SIDE", 120)
        src = str(tmp_path / "source.mov")
        _make_video(src, duration=1.0, width=320, height=240, fps=30, with_audio=True)

        out = normalize_source(src)
        assert out == str(tmp_path / "source.mp4")
        assert os.path.exists(out)
        assert not os.path.exists(src)

    def test_disabled_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media_mod, "_NORMALIZE_MAX_SHORT_SIDE", 0)
        src = str(tmp_path / "source.mp4")
        self._make_444_video(src)
        before = os.path.getmtime(src)

        assert normalize_source(src) == src
        assert os.path.getmtime(src) == before

    def test_no_tmp_left_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media_mod, "_NORMALIZE_MAX_SHORT_SIDE", 120)
        src = str(tmp_path / "source.mp4")
        _make_video(src, duration=1.0, width=320, height=240, fps=30, with_audio=False)

        real_run = media_mod._run

        def failing_run(cmd, **kwargs):
            if cmd[0] == media_mod.FFMPEG:
                raise MediaError("boom")
            return real_run(cmd, **kwargs)

        monkeypatch.setattr(media_mod, "_run", failing_run)
        with pytest.raises(MediaError):
            normalize_source(src)
        assert not os.path.exists(str(tmp_path / ".normalizing.mp4"))
        assert os.path.exists(src)


# ---------------------------------------------------------------------------
# make_llm_proxy() tests
# ---------------------------------------------------------------------------

class TestMakeLlmProxy:
    def test_downscales_short_side_and_keeps_aspect(self, tmp_path):
        """A portrait source is scaled by its short side (the width)."""
        src = str(tmp_path / "portrait.mp4")
        _make_video(src, duration=2, width=1080, height=1920, fps=30)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        info = probe(dst)
        assert info.width == 480
        assert info.height == 854  # 1920 * 480/1080, rounded to even by -2

    def test_downscales_landscape_by_height(self, tmp_path):
        """The short side of a landscape source is its height."""
        src = str(tmp_path / "landscape.mp4")
        _make_video(src, duration=2, width=1920, height=1080, fps=30)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        info = probe(dst)
        assert info.height == 480
        assert info.width == 854

    def test_smaller_source_is_not_upscaled(self, tmp_path):
        """Both caps are one-way — a small, slow source keeps its own geometry."""
        src = str(tmp_path / "small.mp4")
        _make_video(src, duration=2, width=320, height=240, fps=10)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        info = probe(dst)
        assert (info.width, info.height) == (320, 240)
        assert info.fps == pytest.approx(10, abs=0.5)

    def test_caps_frame_rate(self, tmp_path):
        src = str(tmp_path / "fast.mp4")
        _make_video(src, duration=2, width=640, height=480, fps=30)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        assert probe(dst).fps == pytest.approx(15, abs=0.5)

    def test_keeps_audio(self, tmp_path):
        """Audio is half the transcript's input — it must survive the proxy."""
        src = str(tmp_path / "with_audio.mp4")
        _make_video(src, duration=2, width=640, height=480, fps=30, with_audio=True)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        assert probe(dst).has_audio is True

    def test_silent_source_stays_silent(self, tmp_path):
        src = str(tmp_path / "silent.mp4")
        _make_video(src, duration=2, width=640, height=480, fps=30, with_audio=False)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        assert probe(dst).has_audio is False

    def test_preserves_duration(self, tmp_path):
        """Timestamps in the transcript are only useful against the real clock."""
        src = str(tmp_path / "src.mp4")
        _make_video(src, duration=3, width=1080, height=1920, fps=30)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        assert probe(dst).duration_sec == pytest.approx(3, abs=0.2)

    def test_shrinks_the_file(self, tmp_path):
        """The whole point: a 1080p source must get materially smaller."""
        src = str(tmp_path / "big.mp4")
        _make_video(src, duration=3, width=1080, height=1920, fps=30)
        dst = str(tmp_path / "proxy.mp4")

        make_llm_proxy(src, dst)

        assert os.path.getsize(dst) < os.path.getsize(src) / 2

    def test_raises_on_unreadable_source(self, tmp_path):
        with pytest.raises(MediaError):
            make_llm_proxy(str(tmp_path / "nope.mp4"), str(tmp_path / "proxy.mp4"))

    def test_no_partial_output_on_failure(self, tmp_path):
        """A half-written proxy would be uploaded as if it were whole."""
        bad = str(tmp_path / "bad.mp4")
        with open(bad, "wb") as fh:
            fh.write(b"not a video")
        dst = str(tmp_path / "proxy.mp4")

        with pytest.raises(MediaError):
            make_llm_proxy(bad, dst)

        assert not os.path.exists(dst)


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

    def test_stitch_original_mode_source_without_audio(
        self, video_with_audio, video_no_audio, tmp_path
    ):
        """
        Regression (run 3d8e02d7): an audio_source with NO audio stream must not
        fail the stitch. ffmpeg's -map is strict, so mapping the source's absent
        audio used to abort the whole command at the final step — after every
        swap segment had already been generated and paid for. Silence is
        synthesised instead, so a silent source yields a silent delivery.
        """
        clip = str(tmp_path / "c.mp4")
        cut_clip(video_with_audio, start_sec=0.0, end_sec=1.0, dst=clip)
        dst = str(tmp_path / "silent_source_out.mp4")

        stitch(
            clips=[clip],
            audio_source=video_no_audio,
            dst=dst,
            width=320,
            height=240,
            fps=30.0,
        )

        out = probe(dst)
        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        assert out.has_audio, "Silence should be synthesised for an audio-less source"
        # The synthesised silence is infinite; -shortest must cut the output at
        # the end of the concatenated video (≈1s), not run on forever.
        assert abs(out.duration_sec - 1.0) <= 0.3, f"duration={out.duration_sec}"

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

    def test_seedance_mode_probes_each_clip_once(self, tmp_path, monkeypatch):
        """No-audio clip duration reuses the initial probe result.

        Probing now runs on a bounded thread pool (B8), so completion order
        across clips is not guaranteed — only that each clip is probed
        exactly once, regardless of order.
        """
        import threading
        import app.media as media_mod

        calls: list[str] = []
        lock = threading.Lock()

        def fake_probe(path: str) -> MediaInfo:
            with lock:
                calls.append(path)
            return MediaInfo(
                duration_sec=1.0,
                width=320,
                height=240,
                fps=30.0,
                aspect_ratio="4:3",
                has_audio=path.endswith("with-audio.mp4"),
            )

        monkeypatch.setattr(media_mod, "probe", fake_probe)
        # A stand-in ffmpeg still writes its output: stitch() replaces the
        # destination with it atomically (AICP-299 §3.1).
        monkeypatch.setattr(media_mod, "_run", lambda cmd: open(cmd[-1], "wb").close())

        with_audio = str(tmp_path / "with-audio.mp4")
        no_audio = str(tmp_path / "no-audio.mp4")
        stitch(
            clips=[with_audio, no_audio],
            audio_source=with_audio,
            dst=str(tmp_path / "out.mp4"),
            width=320,
            height=240,
            fps=30.0,
            audio_mode="seedance",
        )

        assert sorted(calls) == sorted([with_audio, no_audio])


# ---------------------------------------------------------------------------
# mux_original_audio() tests
# ---------------------------------------------------------------------------

class TestMuxOriginalAudio:
    """Tests for mux_original_audio(), including the B8 video_duration_sec
    pass-through that lets a caller skip the internal probe(video) call."""

    def test_video_duration_sec_skips_probe(
        self, video_with_audio, small_clip_24fps, tmp_path, monkeypatch
    ):
        """When video_duration_sec is supplied, probe() is never called
        (source_has_audio is also supplied, so no probe of audio_source
        either) — the whole call should run zero probe subprocesses."""
        import app.media as media_mod

        calls: list[str] = []
        real_probe = media_mod.probe

        def spy_probe(path: str) -> MediaInfo:
            calls.append(path)
            return real_probe(path)

        monkeypatch.setattr(media_mod, "probe", spy_probe)

        dst = str(tmp_path / "muxed.mp4")
        mux_original_audio(
            small_clip_24fps, video_with_audio, 0.0, dst,
            source_has_audio=True,
            video_duration_sec=1.0,
        )

        assert calls == [], "probe() should not be called when both " \
            "source_has_audio and video_duration_sec are provided"
        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        out = probe(dst)
        assert abs(out.duration_sec - 1.0) <= 0.4
        assert out.has_audio

    def test_video_duration_sec_omitted_still_probes(
        self, video_with_audio, small_clip_24fps, tmp_path, monkeypatch
    ):
        """Backward compatibility: without video_duration_sec, the function
        still probes *video* itself to learn its duration."""
        import app.media as media_mod

        calls: list[str] = []
        real_probe = media_mod.probe

        def spy_probe(path: str) -> MediaInfo:
            calls.append(path)
            return real_probe(path)

        monkeypatch.setattr(media_mod, "probe", spy_probe)

        dst = str(tmp_path / "muxed_no_dur.mp4")
        mux_original_audio(
            small_clip_24fps, video_with_audio, 0.0, dst,
            source_has_audio=True,
        )

        assert calls == [small_clip_24fps]
        assert os.path.exists(dst) and os.path.getsize(dst) > 0


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

    def test_fps_capped_at_30(self):
        """A 60fps source is stitched at the 30fps cap (default FFMPEG_MAX_FPS)."""
        from app.media import MediaInfo
        info = MediaInfo(
            duration_sec=10.0, width=1080, height=1920, fps=60.0,
            aspect_ratio="9:16", has_audio=True,
        )
        w, h, fps = get_default_target(info)
        assert (w, h) == (1080, 1920)
        assert fps == 30.0

    def test_fps_below_cap_unchanged(self):
        """A 24fps source is left untouched (below the cap)."""
        from app.media import MediaInfo
        info = MediaInfo(
            duration_sec=10.0, width=1920, height=1080, fps=24.0,
            aspect_ratio="16:9", has_audio=True,
        )
        _, _, fps = get_default_target(info)
        assert fps == 24.0


# ---------------------------------------------------------------------------
# Delivery size cap: delivery_video_bitrate() / delivery_video_opts()
# ---------------------------------------------------------------------------

class TestDeliveryVideoBitrate:
    def test_budget_fits_the_cap(self):
        """maxrate*duration + bufsize + audio must stay inside the byte cap."""
        max_bytes = 100_000_000
        duration = 60.0
        audio_bps = 192_000
        bps = media_mod.delivery_video_bitrate(
            duration, audio_bitrate_bps=audio_bps, max_bytes=max_bytes
        )
        assert bps is not None
        worst_case_bits = (
            bps * duration                                   # video, at the cap
            + bps * media_mod._DELIVERY_VBV_SECONDS          # VBV slack
            + audio_bps * duration                           # audio
        )
        assert worst_case_bits / 8 <= max_bytes

    def test_longer_video_gets_a_lower_budget(self):
        short = media_mod.delivery_video_bitrate(30.0, max_bytes=100_000_000)
        long = media_mod.delivery_video_bitrate(120.0, max_bytes=100_000_000)
        assert short > long

    def test_audio_comes_out_of_the_same_budget(self):
        silent = media_mod.delivery_video_bitrate(
            60.0, audio_bitrate_bps=0, max_bytes=100_000_000
        )
        with_audio = media_mod.delivery_video_bitrate(
            60.0, audio_bitrate_bps=192_000, max_bytes=100_000_000
        )
        assert with_audio < silent

    def test_disabled_cap_returns_none(self):
        assert media_mod.delivery_video_bitrate(60.0, max_bytes=0) is None

    def test_unknown_duration_returns_none(self):
        assert media_mod.delivery_video_bitrate(0.0, max_bytes=100_000_000) is None

    def test_below_floor_returns_none(self):
        """A video too long to fit above the bitrate floor is left uncapped."""
        # 100 MB over 10 hours is ~22 kbps — far below _DELIVERY_MIN_VIDEO_BPS.
        assert media_mod.delivery_video_bitrate(36000.0, max_bytes=100_000_000) is None


class TestDeliveryVideoOpts:
    def test_capped_encode_carries_maxrate_and_bufsize(self):
        opts = media_mod.delivery_video_opts(
            60.0, audio_bitrate_bps=192_000, max_bytes=100_000_000
        )
        assert opts[:2] == ["-c:v", "libx264"]
        assert "-crf" in opts
        maxrate = int(opts[opts.index("-maxrate") + 1])
        bufsize = int(opts[opts.index("-bufsize") + 1])
        assert maxrate > 0
        assert bufsize == int(maxrate * media_mod._DELIVERY_VBV_SECONDS)

    def test_uncapped_encode_is_plain_crf(self):
        opts = media_mod.delivery_video_opts(60.0, max_bytes=0)
        assert "-maxrate" not in opts
        assert "-bufsize" not in opts
        assert "-crf" in opts

    def test_crf_and_preset_overridable(self):
        opts = media_mod.delivery_video_opts(60.0, max_bytes=0, crf="26", preset="slow")
        assert opts[opts.index("-crf") + 1] == "26"
        assert opts[opts.index("-preset") + 1] == "slow"


class TestEnforceSizeLimit:
    def test_no_op_when_already_under_the_cap(self, video_with_audio, tmp_path):
        dst = str(tmp_path / "under.mp4")
        cut_clip(video_with_audio, 0, 1.0, dst)
        before = os.path.getsize(dst)
        assert media_mod.enforce_size_limit(dst, max_bytes=100_000_000) is False
        assert os.path.getsize(dst) == before

    def test_no_op_when_cap_disabled(self, video_with_audio):
        assert media_mod.enforce_size_limit(video_with_audio, max_bytes=0) is False

    def test_missing_file_is_a_no_op(self, tmp_path):
        assert media_mod.enforce_size_limit(str(tmp_path / "nope.mp4")) is False

    def test_shrinks_an_oversized_file_in_place(self, video_no_audio, tmp_path,
                                                monkeypatch):
        """A cap the source already exceeds forces a re-encode under it."""
        # The synthetic clips are tiny, so drop the bitrate floor — otherwise
        # every cap small enough to bite is also below the floor. A silent clip
        # keeps the budget maths legible at this scale (128k of AAC would eat
        # the whole budget of a 2-second file).
        monkeypatch.setattr(media_mod, "_DELIVERY_MIN_VIDEO_BPS", 5_000)
        dst = str(tmp_path / "big.mp4")
        cut_clip(video_no_audio, 0, 2.0, dst, include_audio=False)
        original = os.path.getsize(dst)
        cap = int(original * 0.6)
        assert media_mod.enforce_size_limit(dst, max_bytes=cap) is True
        assert os.path.getsize(dst) <= cap
        # Still a playable video of the same length.
        info = probe(dst)
        assert abs(info.duration_sec - 2.0) < 0.3

    def test_failed_reencode_leaves_the_file_untouched(self, video_no_audio, tmp_path,
                                                      monkeypatch):
        monkeypatch.setattr(media_mod, "_DELIVERY_MIN_VIDEO_BPS", 5_000)
        dst = str(tmp_path / "keepme.mp4")
        cut_clip(video_no_audio, 0, 2.0, dst, include_audio=False)
        original = os.path.getsize(dst)

        real_run = media_mod._run

        def _boom_on_ffmpeg(cmd, **kwargs):
            """Let the ffprobe pass through; blow up on the encode."""
            if cmd[0] == media_mod.FFMPEG:
                raise MediaError("ffmpeg exploded")
            return real_run(cmd, **kwargs)

        monkeypatch.setattr(media_mod, "_run", _boom_on_ffmpeg)
        with pytest.raises(MediaError):
            media_mod.enforce_size_limit(dst, max_bytes=int(original * 0.6))
        assert os.path.getsize(dst) == original
        assert not os.path.exists(f"{dst}.shrink.mp4")


_MINIMAL_ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 320
PlayResY: 240

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello world
"""


# ---------------------------------------------------------------------------
# _escape_filter_path() tests
# ---------------------------------------------------------------------------

class TestEscapeFilterPath:
    """
    Unit tests for the string transform, plus (below) real-ffmpeg round-trip
    tests. The string-level assertions alone are not sufficient proof of
    correctness here: a naive single-backslash escape (backslash, colon,
    quote each preceded by one backslash) looks right and is what every
    "how to escape ffmpeg subtitle paths"
    guide suggests, but it silently fails against a real ffmpeg binary
    because the `subtitles=` argument is unescaped *twice* on its way to
    disk (see the docstring on `_escape_filter_path`). Only the
    TestEscapeFilterPathRealFfmpeg class below actually catches that.
    """

    def test_backslash_quadrupled(self):
        assert media_mod._escape_filter_path("a\\b") == "a" + "\\" * 4 + "b"

    def test_colon_gets_two_backslashes(self):
        assert media_mod._escape_filter_path("a:b") == "a" + "\\" * 2 + ":b"

    def test_single_quote_gets_three_backslashes(self):
        assert media_mod._escape_filter_path("it's") == "it" + "\\" * 3 + "'s"

    def test_space_left_alone(self):
        assert media_mod._escape_filter_path("a path/with space.ass") == \
            "a path/with space.ass"

    def test_nasty_path_all_four(self):
        """A path carrying a colon, a single quote, a backslash and a space
        at once — the exact combination an operator-renamed project
        directory could produce."""
        nasty = "/data/proj's:dir/weird\\name with space.ass"
        escaped = media_mod._escape_filter_path(nasty)
        assert escaped == (
            "/data/proj" + "\\" * 3 + "'s" + "\\" * 2 + ":dir/weird"
            + "\\" * 4 + "name with space.ass"
        )

    def test_plain_path_unchanged(self):
        """A uuid-clean path (the common case) round-trips untouched."""
        plain = "/data/projects/abc-123/runs/def-456/result.mp4"
        assert media_mod._escape_filter_path(plain) == plain

    def test_order_does_not_double_escape_inserted_backslashes(self):
        """Backslash-escaping runs first, so the backslashes this function
        inserts for ':' and "'" are never themselves re-escaped by a later
        replacement in the same call."""
        escaped = media_mod._escape_filter_path("a:b")
        # If colon's inserted backslashes were re-processed by a (buggy)
        # later backslash pass, we'd see 4 backslashes here, not 2.
        assert escaped.count("\\") == 2


class TestEscapeFilterPathRealFfmpeg:
    """
    Round-trips _escape_filter_path()'s output through a real ffmpeg
    `subtitles` filter call, burning an ASS file whose path is deliberately
    littered with the characters the helper defends against. This is the
    test that actually matters: burning subtitles from a path with a
    literal apostrophe looked fine with a naive single-backslash escape
    right up until it hit real ffmpeg, which unescapes the `subtitles=`
    argument twice (see the docstring) — a plain string-equality test on
    the escaped output can't see that, only a real invocation can.
    """

    def _burn_with_src(self, video_with_audio, tmp_path, ass_dir_name, ass_filename):
        ass_dir = tmp_path / ass_dir_name
        ass_dir.mkdir()
        ass_path = ass_dir / ass_filename
        ass_path.write_text(_MINIMAL_ASS)
        dst = str(tmp_path / "out.mp4")
        burn_subtitles(video_with_audio, str(ass_path), dst, duration_sec=2.0, has_audio=True)
        return dst

    @pytest.mark.parametrize("ass_dir_name,ass_filename", [
        ("plaindir", "plain.ass"),
        ("a:colon_dir", "colon.ass"),
        ("a'quote_dir", "quote.ass"),
        ("plaindir2", "back\\slash.ass"),
        ("with space dir", "with space.ass"),
        ("proj's:dir", "cap\\tion file.ass"),  # every special char at once
        ("adjdir", "a\\:b.ass"),               # backslash immediately before colon
        ("adjdir2", "a\\'b.ass"),              # backslash immediately before quote
        ("adjdir3", "a:'b.ass"),               # colon immediately before quote
        ("adjdir4", "a''b.ass"),               # two quotes back to back
    ])
    def test_burns_successfully(self, video_with_audio, tmp_path, ass_dir_name, ass_filename):
        dst = self._burn_with_src(video_with_audio, tmp_path, ass_dir_name, ass_filename)
        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        info = probe(dst)
        assert abs(info.duration_sec - 2.0) <= 0.3


# ---------------------------------------------------------------------------
# burn_subtitles() tests
# ---------------------------------------------------------------------------

class TestBurnSubtitles:
    def _capture_cmd(self, monkeypatch):
        """Monkeypatch _run to capture the argv without invoking ffmpeg."""
        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(media_mod, "_run", fake_run)
        return captured

    def test_audio_copy_when_has_audio(self, tmp_path, monkeypatch):
        captured = self._capture_cmd(monkeypatch)
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
            duration_sec=5.0, has_audio=True,
        )
        cmd = captured[0]
        assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
        assert "-an" not in cmd

    def test_an_when_no_audio(self, tmp_path, monkeypatch):
        captured = self._capture_cmd(monkeypatch)
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
            duration_sec=5.0, has_audio=False,
        )
        cmd = captured[0]
        assert "-an" in cmd
        assert "-c:a" not in cmd

    def test_vf_carries_escaped_ass_path(self, tmp_path, monkeypatch):
        captured = self._capture_cmd(monkeypatch)
        nasty_ass = "/data/proj's:dir/cap\\tion file.ass"
        burn_subtitles(
            "src.mp4", nasty_ass, str(tmp_path / "out.mp4"),
            duration_sec=5.0, has_audio=True,
        )
        cmd = captured[0]
        assert "-vf" in cmd
        vf_value = cmd[cmd.index("-vf") + 1]
        expected = f"subtitles={media_mod._escape_filter_path(nasty_ass)}"
        assert vf_value == expected
        # Sanity: the raw (unescaped) path must not appear verbatim.
        assert nasty_ass not in vf_value

    def test_delivery_video_opts_present(self, tmp_path, monkeypatch):
        captured = self._capture_cmd(monkeypatch)
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
            duration_sec=60.0, has_audio=True,
        )
        cmd = captured[0]
        expected_opts = media_mod.delivery_video_opts(
            60.0, audio_bitrate_bps=media_mod._STITCH_AUDIO_BPS
        )
        for i in range(len(cmd) - len(expected_opts) + 1):
            if cmd[i:i + len(expected_opts)] == expected_opts:
                break
        else:
            pytest.fail(f"delivery_video_opts {expected_opts} not found contiguously in {cmd}")

    def test_no_audio_charges_zero_against_budget(self, tmp_path, monkeypatch):
        """has_audio=False must charge 0 bps against the size budget, which
        yields a different (higher) -maxrate than the has_audio=True case —
        proving the flag actually changes the cap maths, not just the -an vs
        -c:a copy choice."""
        captured = self._capture_cmd(monkeypatch)
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
            duration_sec=60.0, has_audio=False,
        )
        silent_cmd = captured[0]

        captured.clear()
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out2.mp4"),
            duration_sec=60.0, has_audio=True,
        )
        with_audio_cmd = captured[0]

        silent_maxrate = int(silent_cmd[silent_cmd.index("-maxrate") + 1])
        with_audio_maxrate = int(with_audio_cmd[with_audio_cmd.index("-maxrate") + 1])
        assert silent_maxrate > with_audio_maxrate

        expected_silent_opts = media_mod.delivery_video_opts(60.0, audio_bitrate_bps=0)
        assert str(silent_maxrate) == expected_silent_opts[
            expected_silent_opts.index("-maxrate") + 1
        ]

    def test_movflags_threads_and_overwrite_flags(self, tmp_path, monkeypatch):
        captured = self._capture_cmd(monkeypatch)
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
            duration_sec=5.0, has_audio=True,
        )
        cmd = captured[0]
        assert cmd[0] == media_mod.FFMPEG
        assert "-y" in cmd
        assert "-movflags" in cmd and cmd[cmd.index("-movflags") + 1] == "+faststart"
        assert "-threads" in cmd and cmd[cmd.index("-threads") + 1] == media_mod._FFMPEG_THREADS

    def test_probes_once_when_neither_supplied(self, tmp_path, monkeypatch):
        """Both duration_sec and has_audio unset -> exactly one probe() call,
        not two independent ones."""
        self._capture_cmd(monkeypatch)
        calls: list[str] = []

        def fake_probe(path):
            calls.append(path)
            return MediaInfo(
                duration_sec=3.0, width=320, height=240, fps=30.0,
                aspect_ratio="4:3", has_audio=True,
            )

        monkeypatch.setattr(media_mod, "probe", fake_probe)
        burn_subtitles("src.mp4", "cap.ass", str(tmp_path / "out.mp4"))
        assert calls == ["src.mp4"]

    def test_no_probe_when_both_supplied(self, tmp_path, monkeypatch):
        self._capture_cmd(monkeypatch)

        def fail_probe(path):
            raise AssertionError("probe() should not be called")

        monkeypatch.setattr(media_mod, "probe", fail_probe)
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
            duration_sec=5.0, has_audio=True,
        )  # must not raise

    def test_probes_only_missing_field(self, tmp_path, monkeypatch):
        """has_audio supplied but duration_sec omitted -> still probes once
        (for duration), and the caller-supplied has_audio wins over the
        probed value."""
        self._capture_cmd(monkeypatch)
        calls: list[str] = []

        def fake_probe(path):
            calls.append(path)
            return MediaInfo(
                duration_sec=7.5, width=320, height=240, fps=30.0,
                aspect_ratio="4:3", has_audio=False,
            )

        monkeypatch.setattr(media_mod, "probe", fake_probe)
        captured = []
        real_run = media_mod._run

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(media_mod, "_run", fake_run)
        burn_subtitles(
            "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
            has_audio=True,
        )
        assert calls == ["src.mp4"]
        cmd = captured[0]
        assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"

    def test_failing_ffmpeg_raises_mediaerror(self, tmp_path, monkeypatch):
        def boom(cmd, **kwargs):
            raise MediaError("ffmpeg exploded")

        monkeypatch.setattr(media_mod, "_run", boom)
        with pytest.raises(MediaError):
            burn_subtitles(
                "src.mp4", "cap.ass", str(tmp_path / "out.mp4"),
                duration_sec=5.0, has_audio=True,
            )

    def test_real_burn_with_audio(self, video_with_audio, tmp_path):
        """End-to-end: real ffmpeg burns a real ASS file onto a real video."""
        ass_path = tmp_path / "captions.ass"
        ass_path.write_text(_MINIMAL_ASS)
        dst = str(tmp_path / "burned.mp4")

        burn_subtitles(video_with_audio, str(ass_path), dst)

        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        info = probe(dst)
        assert info.has_audio is True
        assert abs(info.duration_sec - 2.0) <= 0.3

    def test_real_burn_without_audio(self, video_no_audio, tmp_path):
        ass_path = tmp_path / "captions.ass"
        ass_path.write_text(_MINIMAL_ASS)
        dst = str(tmp_path / "burned_silent.mp4")

        burn_subtitles(video_no_audio, str(ass_path), dst)

        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        info = probe(dst)
        assert info.has_audio is False
        assert abs(info.duration_sec - 2.0) <= 0.3

    def test_real_burn_with_nasty_ass_path(self, video_with_audio, tmp_path):
        """The escaping actually has to work against real ffmpeg, not just
        the string transform in isolation: a project/run directory carrying
        a colon, a single quote, a backslash and a space in its path."""
        nasty_dir = tmp_path / "proj's:dir"
        nasty_dir.mkdir()
        ass_path = nasty_dir / "cap\\tion file.ass"
        ass_path.write_text(_MINIMAL_ASS)
        dst = str(tmp_path / "out dir" / "burned.mp4")
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        burn_subtitles(video_with_audio, str(ass_path), dst, duration_sec=2.0, has_audio=True)

        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        info = probe(dst)
        assert abs(info.duration_sec - 2.0) <= 0.3
        assert info.has_audio is True

class TestStitchDeliveryCap:
    def test_stitched_output_carries_the_cap(self, video_with_audio, tmp_path, monkeypatch):
        """stitch() sizes -maxrate from the concatenated clip duration."""
        captured: list[list[str]] = []
        real_run = media_mod._run

        def _capture(cmd, **kwargs):
            """Record the encode; let stitch's clip ffprobes run for real."""
            if cmd[0] != media_mod.FFMPEG:
                return real_run(cmd, **kwargs)
            captured.append(cmd)
            open(cmd[-1], "wb").close()  # stitch() then replaces dst with it atomically
            return None

        monkeypatch.setattr(media_mod, "_run", _capture)
        stitch(
            [video_with_audio, video_with_audio],
            audio_source=video_with_audio,
            dst=str(tmp_path / "out.mp4"),
            width=320, height=240, fps=30,
        )
        cmd = captured[0]
        assert "-maxrate" in cmd
        # Two 2s clips -> a ~4s budget, not a 2s one.
        four_sec = media_mod.delivery_video_bitrate(4.0, audio_bitrate_bps=192_000)
        assert abs(int(cmd[cmd.index("-maxrate") + 1]) - four_sec) < four_sec * 0.05


class TestStitchOriginalTwoPass:
    """Regression (runs 3b79ba7d / d46bb1b8): the concat encode must not carry
    the source's audio as an extra input. With it, ffmpeg 7.x stops pacing the
    clip decoders once the output passes the longest clip and buffers the rest
    of the video as raw frames — a 95 s 1080p stitch was OOM-killed at 6.2 GB."""

    def _capture(self, monkeypatch):
        captured: list[list[str]] = []
        real_run = media_mod._run

        def fake(cmd, **kwargs):
            if cmd[0] != media_mod.FFMPEG:
                return real_run(cmd, **kwargs)
            captured.append(cmd)
            open(cmd[-1], "wb").close()
            return None

        monkeypatch.setattr(media_mod, "_run", fake)
        return captured

    @staticmethod
    def _inputs(cmd):
        return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"]

    def test_concat_is_video_only_and_audio_is_muxed_with_copy(
        self, video_with_audio, tmp_path, monkeypatch
    ):
        captured = self._capture(monkeypatch)
        clips = [video_with_audio, video_with_audio, video_with_audio]
        dst = tmp_path / "out.mp4"
        stitch(clips, audio_source=video_with_audio, dst=str(dst),
               width=320, height=240, fps=30)

        assert len(captured) == 2
        concat, mux = captured
        assert self._inputs(concat) == clips
        assert "-an" in concat and "-shortest" not in concat
        assert "concat=n=3:v=1:a=0[vout]" in concat[concat.index("-filter_complex") + 1]

        assert self._inputs(mux) == [concat[-1], video_with_audio]
        assert mux[mux.index("-c:v") + 1] == "copy"
        assert "-filter_complex" not in mux and "-shortest" in mux
        assert dst.exists()
        # The intermediate video-only file is cleaned up.
        assert not os.path.exists(concat[-1])

    def test_intermediate_removed_when_mux_fails(
        self, video_with_audio, tmp_path, monkeypatch
    ):
        captured = self._capture(monkeypatch)
        inner = media_mod._run

        def fail_mux(cmd, **kwargs):
            if cmd[0] == media_mod.FFMPEG and "-filter_complex" not in cmd:
                raise MediaError("mux boom")
            return inner(cmd, **kwargs)

        monkeypatch.setattr(media_mod, "_run", fail_mux)
        dst = tmp_path / "out.mp4"
        with pytest.raises(MediaError, match="mux boom"):
            stitch([video_with_audio], audio_source=video_with_audio, dst=str(dst),
                   width=320, height=240, fps=30)
        assert len(captured) == 1
        assert not os.path.exists(captured[0][-1])
        assert not dst.exists()
        assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# make_voice_reference — normalising a voice sample for Seedance 2.5
# ---------------------------------------------------------------------------


class TestMakeVoiceReference:
    """Whatever the operator supplies becomes a mono mp3 inside the API's bounds.

    Normalising unconditionally is the point: every file that ever reaches
    kie.ai came out of this function, so a container or length the API rejects
    cannot slip through by being named ``.mp3``.
    """

    def test_extracts_audio_from_a_video(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        _make_video(src, duration=5.0, width=320, height=240, fps=30, with_audio=True)
        dst = str(tmp_path / "voice.mp3")

        duration = media_mod.make_voice_reference(src, dst)

        assert os.path.exists(dst)
        has_audio, out_sec = media_mod.probe_audio(dst)
        assert has_audio
        assert out_sec == pytest.approx(5.0, abs=0.3)
        assert duration == pytest.approx(5.0, abs=0.3)
        # Video is dropped: this is handed to an audio-only API field.
        with pytest.raises(media_mod.MediaError, match="No video stream"):
            media_mod.probe(dst)

    def test_output_is_mono(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        _make_video(src, duration=3.0, width=320, height=240, fps=30, with_audio=True)
        dst = str(tmp_path / "voice.mp3")
        media_mod.make_voice_reference(src, dst)

        out = subprocess.run(
            ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels,codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", dst],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        codec, channels = out.stdout.decode().split()
        assert codec == "mp3"
        assert channels == "1"

    def test_truncates_to_the_api_ceiling(self, tmp_path):
        """A long sample is trimmed, not rejected — it is still a good voice."""
        src = str(tmp_path / "long.mp4")
        over = ai_models.VOICE_REF_MAX_SEC + 6
        _make_video(src, duration=over, width=160, height=120, fps=15, with_audio=True)
        dst = str(tmp_path / "voice.mp3")

        duration = media_mod.make_voice_reference(src, dst)

        assert duration == ai_models.VOICE_REF_MAX_SEC
        _, out_sec = media_mod.probe_audio(dst)
        assert out_sec <= ai_models.VOICE_REF_MAX_SEC + 0.5

    def test_rejects_a_source_with_no_audio_track(self, tmp_path):
        src = str(tmp_path / "silent.mp4")
        _make_video(src, duration=4.0, width=320, height=240, fps=30, with_audio=False)
        dst = str(tmp_path / "voice.mp3")

        with pytest.raises(media_mod.MediaError, match="no audio track"):
            media_mod.make_voice_reference(src, dst)

    def test_rejects_a_sample_below_the_api_floor(self, tmp_path):
        """Padding with silence would satisfy the validator and ruin the voice.

        So this fails loudly rather than producing a technically-valid file that
        makes the generation worse.
        """
        src = str(tmp_path / "tiny.mp4")
        under = ai_models.VOICE_REF_MIN_SEC - 1.0
        _make_video(src, duration=under, width=160, height=120, fps=15, with_audio=True)
        dst = str(tmp_path / "voice.mp3")

        with pytest.raises(media_mod.MediaError, match="at least"):
            media_mod.make_voice_reference(src, dst)
        assert not os.path.exists(dst)

    def test_stays_well_under_the_size_ceiling(self, tmp_path):
        """A max-length output must not approach kie.ai's 15 MB per-file cap."""
        src = str(tmp_path / "long.mp4")
        _make_video(
            src, duration=ai_models.VOICE_REF_MAX_SEC,
            width=160, height=120, fps=15, with_audio=True,
        )
        dst = str(tmp_path / "voice.mp3")
        media_mod.make_voice_reference(src, dst)

        assert os.path.getsize(dst) < ai_models.VOICE_REF_MAX_BYTES / 10


    def test_accepts_an_audio_only_source(self, tmp_path):
        """The primary case: an operator uploads an mp3, not a video.

        Regression guard — this went through media.probe() first, which rejects
        anything without a video stream, so every plain audio upload 400'd with
        "not a playable video" while videos worked fine.
        """
        src = str(tmp_path / "sample.mp3")
        subprocess.run(
            ["/usr/bin/ffmpeg", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=6",
             "-c:a", "libmp3lame", src],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        dst = str(tmp_path / "voice.mp3")

        duration = media_mod.make_voice_reference(src, dst)

        assert os.path.exists(dst)
        assert duration == pytest.approx(6.0, abs=0.3)


class TestProbeAudio:
    """probe_audio tolerates the audio-only files probe() is right to reject."""

    def test_reports_no_audio_for_a_silent_video(self, tmp_path):
        src = str(tmp_path / "silent.mp4")
        _make_video(src, duration=3.0, width=160, height=120, fps=15, with_audio=False)
        assert media_mod.probe_audio(src) == (False, 0.0)

    def test_reads_duration_from_an_audio_only_file(self, tmp_path):
        src = str(tmp_path / "tone.mp3")
        subprocess.run(
            ["/usr/bin/ffmpeg", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=4",
             "-c:a", "libmp3lame", src],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        has_audio, duration = media_mod.probe_audio(src)
        assert has_audio
        assert duration == pytest.approx(4.0, abs=0.3)


class TestSpliceAudioWindow:
    """splice_audio_window: the donor's audio lands in the window and NOTHING
    else moves — the property the whole audio-patch feature rests on."""

    _MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?[0-9.]+) dB")

    def _toned_video(self, path: str, duration: float, freq: int) -> None:
        subprocess.run(
            ["/usr/bin/ffmpeg", "-y",
             "-f", "lavfi",
             "-i", f"testsrc=duration={duration}:size=160x120:rate=10",
             "-f", "lavfi",
             "-i", f"sine=frequency={freq}:duration={duration}",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
             "-shortest", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )

    def _tone_db(self, path: str, start: float, end: float, freq: int) -> float:
        r = subprocess.run(
            ["/usr/bin/ffmpeg", "-hide_banner", "-ss", str(start), "-to", str(end),
             "-i", path,
             "-af", f"bandpass=f={freq}:width_type=h:w=60,volumedetect",
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        m = self._MEAN_VOLUME_RE.search(r.stderr.decode())
        assert m, r.stderr.decode()[-400:]
        return float(m.group(1))

    @pytest.fixture()
    def pair(self, tmp_path):
        base = str(tmp_path / "base.mp4")
        donor = str(tmp_path / "donor.mp4")
        self._toned_video(base, 6.0, 440)
        self._toned_video(donor, 6.0, 880)
        return base, donor

    def test_window_carries_the_donor_and_the_rest_carries_the_base(self, pair, tmp_path):
        base, donor = pair
        dst = str(tmp_path / "out.mp4")

        media_mod.splice_audio_window(
            base, donor, dst, base_start=3.0, base_end=4.0, donor_start=1.0
        )

        assert self._tone_db(dst, 3.2, 3.8, 880) > self._tone_db(dst, 3.2, 3.8, 440) + 10
        assert self._tone_db(dst, 0.5, 2.5, 440) > self._tone_db(dst, 0.5, 2.5, 880) + 10
        assert self._tone_db(dst, 4.5, 5.5, 440) > self._tone_db(dst, 4.5, 5.5, 880) + 10

    def test_audio_outside_the_window_is_left_at_its_original_level(self, pair, tmp_path):
        """Not just "still the base tone" — the SAME level, because nothing
        outside the window is faded, shifted or re-mixed."""
        base, donor = pair
        dst = str(tmp_path / "out.mp4")

        media_mod.splice_audio_window(
            base, donor, dst, base_start=3.0, base_end=4.0, donor_start=1.0
        )

        for a, b in [(0.5, 2.5), (4.5, 5.5)]:
            assert self._tone_db(dst, a, b, 440) == pytest.approx(
                self._tone_db(base, a, b, 440), abs=0.5
            )

    def test_video_stream_is_copied_unchanged(self, pair, tmp_path):
        base, donor = pair
        dst = str(tmp_path / "out.mp4")
        before = media_mod.probe(base)

        media_mod.splice_audio_window(
            base, donor, dst, base_start=3.0, base_end=4.0, donor_start=1.0
        )

        after = media_mod.probe(dst)
        assert after.duration_sec == pytest.approx(before.duration_sec, abs=0.05)
        assert (after.width, after.height) == (before.width, before.height)
        assert after.fps == pytest.approx(before.fps, abs=0.01)

    def test_rejects_a_window_shorter_than_the_minimum(self, pair, tmp_path):
        base, donor = pair
        with pytest.raises(media_mod.MediaError, match="at least"):
            media_mod.splice_audio_window(
                base, donor, str(tmp_path / "x.mp4"),
                base_start=3.0, base_end=3.01, donor_start=1.0,
            )

    def test_rejects_a_window_past_the_end_of_the_base(self, pair, tmp_path):
        base, donor = pair
        with pytest.raises(media_mod.MediaError, match="past the end"):
            media_mod.splice_audio_window(
                base, donor, str(tmp_path / "x.mp4"),
                base_start=3.0, base_end=99.0, donor_start=1.0,
            )

    def test_rejects_a_donor_too_short_for_the_window(self, pair, tmp_path):
        base, donor = pair
        with pytest.raises(media_mod.MediaError, match="only"):
            media_mod.splice_audio_window(
                base, donor, str(tmp_path / "x.mp4"),
                base_start=1.0, base_end=3.0, donor_start=5.5,
            )

    def test_rejects_a_base_with_no_audio(self, pair, tmp_path):
        _base, donor = pair
        silent = str(tmp_path / "silent.mp4")
        _make_video(silent, duration=6.0, width=160, height=120, fps=10,
                    with_audio=False)
        with pytest.raises(media_mod.MediaError, match="no audio"):
            media_mod.splice_audio_window(
                silent, donor, str(tmp_path / "x.mp4"),
                base_start=1.0, base_end=2.0, donor_start=1.0,
            )

    def test_a_mono_base_keeps_its_level(self, tmp_path):
        """Regression: conforming both branches to a hardcoded stereo layout
        rematrixed a MONO base with ffmpeg's power-preserving 1/sqrt(2), which
        dropped the WHOLE delivered soundtrack by 3 dB — outside the patched
        window as much as inside it. The layout is taken from the base."""
        base = str(tmp_path / "mono_base.mp4")
        donor = str(tmp_path / "mono_donor.mp4")
        dst = str(tmp_path / "out.mp4")
        # `sine` is mono and nothing here upmixes it.
        self._toned_video(base, 6.0, 440)
        self._toned_video(donor, 6.0, 880)
        assert media_mod._audio_layout(base)[0] == "mono"

        media_mod.splice_audio_window(
            base, donor, dst, base_start=3.0, base_end=4.0, donor_start=1.0
        )

        assert self._tone_db(dst, 0.5, 2.5, 440) == pytest.approx(
            self._tone_db(base, 0.5, 2.5, 440), abs=0.5
        )
        # ...and the window still carries the donor.
        assert self._tone_db(dst, 3.2, 3.8, 880) > self._tone_db(dst, 3.2, 3.8, 440) + 10

    def test_an_oversized_crossfade_is_clamped_not_rejected(self, pair, tmp_path):
        """A 0.5s seam against a 0.2s window is a tight window, not an error."""
        base, donor = pair
        dst = str(tmp_path / "out.mp4")
        media_mod.splice_audio_window(
            base, donor, dst, base_start=3.0, base_end=3.2, donor_start=1.0,
            crossfade_sec=0.5,
        )
        assert os.path.exists(dst)


class TestAudioFingerprint:
    """audio_fingerprint keys caption transcript reuse on the decoded audio."""

    def test_same_audio_under_different_video_is_equal(self, video_with_audio, tmp_path):
        other = str(tmp_path / "other_picture.mp4")
        subprocess.run(
            ["/usr/bin/ffmpeg", "-y",
             "-f", "lavfi", "-i", "testsrc2=duration=2:size=160x120:rate=15",
             "-i", video_with_audio,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "copy", "-shortest", other],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        fp = media_mod.audio_fingerprint(video_with_audio)
        assert fp is not None and len(fp) == 64
        assert media_mod.audio_fingerprint(other) == fp

    def test_different_audio_differs(self, video_with_audio, tmp_path):
        other = str(tmp_path / "other_tone.mp4")
        subprocess.run(
            ["/usr/bin/ffmpeg", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=30",
             "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", other],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        assert media_mod.audio_fingerprint(other) != media_mod.audio_fingerprint(
            video_with_audio
        )

    def test_same_samples_shifted_against_picture_differ(self, video_with_audio, tmp_path):
        shifted = str(tmp_path / "shifted.mp4")
        subprocess.run(
            ["/usr/bin/ffmpeg", "-y", "-i", video_with_audio,
             "-itsoffset", "0.5", "-i", video_with_audio,
             "-map", "0:v", "-map", "1:a", "-c", "copy", shifted],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        assert media_mod.audio_fingerprint(shifted) != media_mod.audio_fingerprint(
            video_with_audio
        )

    def test_no_audio_or_unreadable_is_none(self, video_no_audio, tmp_path):
        assert media_mod.audio_fingerprint(video_no_audio) is None
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"not a video")
        assert media_mod.audio_fingerprint(str(junk)) is None
        assert media_mod.audio_fingerprint(str(tmp_path / "missing.mp4")) is None


# ---------------------------------------------------------------------------
# extract_jpeg_frames — stills for the localisation translator
# ---------------------------------------------------------------------------


def test_extract_jpeg_frames_returns_jpegs_and_skips_past_the_end(tmp_path):
    src = str(tmp_path / "src.mp4")
    _make_video(src, 2.0, 320, 240, 25)

    out = media_mod.extract_jpeg_frames(src, [0.3, 1.5, 9.0], width=160)

    assert [t for t, _ in out] == [0.3, 1.5]
    for _, jpeg in out:
        assert jpeg[:2] == b"\xff\xd8"


def test_extract_jpeg_frames_raises_when_nothing_comes_out(tmp_path):
    with pytest.raises(MediaError, match="no frame"):
        media_mod.extract_jpeg_frames(str(tmp_path / "missing.mp4"), [0.3])
