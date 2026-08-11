"""
Tests for app/scene.py — PySceneDetect content cuts for subtitle removal.

The happy-path test builds a real multi-scene video with ffmpeg and runs the
actual PySceneDetect detector, so it doubles as an integration check that the
dependency is wired up. It is skipped when ffmpeg or scenedetect are absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.scene import detect_scenes


def _have(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


_HAVE_FFMPEG = _have(["ffmpeg", "-version"])
try:
    import scenedetect  # noqa: F401

    _HAVE_SCENEDETECT = True
except Exception:
    _HAVE_SCENEDETECT = False


@pytest.fixture(scope="module")
def four_scene_video():
    """A 20s clip of four hard-cut 5s color shots (red/green/blue/yellow)."""
    if not _HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    path = os.path.join(tempfile.mkdtemp(), "scenes.mp4")
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:r=10:d=5",
            "-f", "lavfi", "-i", "color=c=green:s=320x240:r=10:d=5",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=10:d=5",
            "-f", "lavfi", "-i", "color=c=yellow:s=320x240:r=10:d=5",
            "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1[out]",
            "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            path,
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return path


@pytest.mark.skipif(not _HAVE_SCENEDETECT, reason="scenedetect not installed")
def test_detects_hard_cuts(four_scene_video):
    ranges = detect_scenes(four_scene_video, fps=10.0, min_scene_len_sec=4.0)
    # Four distinct color shots → four scenes covering the whole 20s.
    assert len(ranges) == 4
    assert ranges[0][0] == pytest.approx(0.0, abs=0.2)
    assert ranges[-1][1] == pytest.approx(20.0, abs=0.5)
    # Contiguous, non-overlapping.
    for (a_s, a_e), (b_s, b_e) in zip(ranges, ranges[1:]):
        assert b_s == pytest.approx(a_e, abs=0.2)


@pytest.mark.skipif(not _HAVE_SCENEDETECT, reason="scenedetect not installed")
def test_min_scene_len_merges_short_cuts(four_scene_video):
    # A 25s minimum is longer than the whole 20s clip, so every cut is merged
    # away — the detector finds no distinct scenes (empty list). The analyze
    # caller treats that as "single shot" and uses one full-length range.
    ranges = detect_scenes(four_scene_video, fps=10.0, min_scene_len_sec=25.0)
    assert len(ranges) <= 1


def test_none_fps_does_not_crash_conversion(monkeypatch):
    """A missing fps falls back to 25 rather than raising on the frame math."""
    calls = {}

    class _FakeDetector:
        def __init__(self, min_scene_len):
            calls["min_scene_len"] = min_scene_len

    def _fake_detect(path, detector):
        return []

    import app.scene as scene_mod

    fake_module = type(sys)("scenedetect")
    fake_module.ContentDetector = _FakeDetector
    fake_module.detect = _fake_detect
    monkeypatch.setitem(sys.modules, "scenedetect", fake_module)

    ranges = scene_mod.detect_scenes("x.mp4", fps=None, min_scene_len_sec=4.0)
    assert ranges == []
    # 4s * 25fps fallback = 100 frames
    assert calls["min_scene_len"] == 100
