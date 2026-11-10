"""
Tests for the overlay ffmpeg builder (app/overlay.py) and OverlayStatus state
machine. Mirrors the string-assertion style of test_media.py for ffmpeg cmds.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import app.overlay as overlay_mod
from app.media import MediaError, MediaInfo
from app.overlay import OverlaySpec, burn_overlays
from app.state_machine import (
    OverlayStatus,
    can_transition,
    InvalidTransition,
    transition,
)


# ---------------------------------------------------------------------------
# Shared: capture the ffmpeg command without running it
# ---------------------------------------------------------------------------


@pytest.fixture()
def captured_cmd(monkeypatch):
    """Patch probe() to fixed dims and _run() to capture the built command."""
    box = {}

    def fake_probe(path):
        return MediaInfo(
            duration_sec=10.0, width=1080, height=1920, fps=30.0,
            aspect_ratio="9:16", has_audio=True,
        )

    def fake_run(cmd, **kw):
        box["cmd"] = cmd
        return None

    monkeypatch.setattr(overlay_mod, "probe", fake_probe)
    monkeypatch.setattr(overlay_mod, "_run", fake_run)
    return box


def _filter_complex(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


# ---------------------------------------------------------------------------
# test_overlay_filter_graph
# ---------------------------------------------------------------------------


class TestFilterGraph:
    def test_empty_specs_raises(self, captured_cmd):
        with pytest.raises(MediaError):
            burn_overlays("src.mp4", "out.mp4", [])

    def test_single_overlay_graph(self, captured_cmd, tmp_path):
        spec = OverlaySpec(
            asset_path="card.png", nx=0.06, ny=0.72, sw=0.40,
            start_sec=2.0, end_sec=7.0, fade_sec=0.4,
        )
        burn_overlays("src.mp4", str(tmp_path / "out.mp4"), [spec])
        cmd = captured_cmd["cmd"]
        fc = _filter_complex(cmd)

        # One looped image input (-loop 1 -i) besides the base video.
        assert cmd.count("-loop") == 1
        assert cmd.count("-i") == 2  # base + 1 overlay

        # scale to fraction of frame width: round(1080 * 0.40) = 432
        assert "scale=432:-1" in fc
        # fade in/out on alpha
        assert "fade=in:st=2.0:d=0.4:alpha=1" in fc
        assert "fade=out:st=6.6:d=0.4:alpha=1" in fc  # end(7) - fade(0.4) = 6.6
        # position px: round(1080*0.06)=65, round(1920*0.72)=1382
        assert "overlay=x='min(65,W-w)':y='min(1382,H-h)'" in fc
        assert "enable='between(t,2.0,7.0)'" in fc
        assert "[vout]" in fc

        # audio stream-copied, source audio optional, shortest bounds output.
        assert cmd[cmd.index("-c:a") + 1] == "copy"
        assert "0:a?" in cmd
        assert "-shortest" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "libx264"

    def test_two_overlays_chain_and_inputs(self, captured_cmd, tmp_path):
        specs = [
            OverlaySpec(asset_path="a.png", nx=0.1, ny=0.1, start_sec=0, end_sec=3),
            OverlaySpec(asset_path="b.png", nx=0.5, ny=0.5, start_sec=1, end_sec=4),
        ]
        burn_overlays("src.mp4", str(tmp_path / "out.mp4"), specs)
        cmd = captured_cmd["cmd"]
        fc = _filter_complex(cmd)
        # Two looped inputs.
        assert cmd.count("-loop") == 2
        # Chained overlays: first writes [b0], last writes [vout].
        assert "[b0]" in fc
        assert "[ov0]" in fc and "[ov1]" in fc
        assert fc.strip().endswith("[vout]")

    def test_no_scale_when_sw_none(self, captured_cmd, tmp_path):
        spec = OverlaySpec(asset_path="c.png", nx=0, ny=0, start_sec=0, end_sec=2)
        burn_overlays("src.mp4", str(tmp_path / "out.mp4"), [spec])
        fc = _filter_complex(captured_cmd["cmd"])
        assert "scale=" not in fc
        assert "fade=" not in fc  # fade_sec default 0


# ---------------------------------------------------------------------------
# test_overlay_clamp — end_sec beyond duration is clamped in the graph
# ---------------------------------------------------------------------------


class TestClamp:
    def test_end_beyond_duration_clamped(self, captured_cmd, tmp_path):
        # probe() returns duration 10.0; end_sec=99 should clamp to 10.0.
        spec = OverlaySpec(asset_path="c.png", nx=0, ny=0, start_sec=1, end_sec=99)
        burn_overlays("src.mp4", str(tmp_path / "out.mp4"), [spec])
        fc = _filter_complex(captured_cmd["cmd"])
        assert "enable='between(t,1,10.0)'" in fc


# ---------------------------------------------------------------------------
# test_state_machine_overlay
# ---------------------------------------------------------------------------


class _FakeJob:
    def __init__(self, status):
        self.status = status
        self.updated_at = None


class TestOverlayStateMachine:
    def test_valid_forward_path(self):
        assert can_transition(OverlayStatus.created, OverlayStatus.queued)
        assert can_transition(OverlayStatus.queued, OverlayStatus.processing)
        assert can_transition(OverlayStatus.processing, OverlayStatus.done)
        assert can_transition(OverlayStatus.processing, OverlayStatus.delivering)
        assert can_transition(OverlayStatus.delivering, OverlayStatus.done)

    def test_retry_edges(self):
        assert can_transition(OverlayStatus.failed, OverlayStatus.queued)
        assert can_transition(OverlayStatus.done, OverlayStatus.queued)

    def test_invalid_edges(self):
        assert not can_transition(OverlayStatus.created, OverlayStatus.done)
        assert not can_transition(OverlayStatus.queued, OverlayStatus.delivering)
        assert not can_transition(OverlayStatus.done, OverlayStatus.processing)

    def test_transition_applies_and_rejects(self):
        job = _FakeJob(OverlayStatus.created)
        transition(job, OverlayStatus.queued)
        assert job.status == OverlayStatus.queued
        assert job.updated_at is not None
        with pytest.raises(InvalidTransition):
            transition(job, OverlayStatus.done)


# ---------------------------------------------------------------------------
# Integration: burn a tiny PNG onto a 2s clip and probe the output
# ---------------------------------------------------------------------------


def _make_video(path, duration, w, h, fps, with_audio):
    cmd = [
        "/usr/bin/ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size={w}x{h}:rate={fps}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-an"]
    cmd.append(path)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-500:])


def _make_png(path, w, h):
    r = subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-f", "lavfi",
         "-i", f"color=c=red@0.5:size={w}x{h},format=rgba",
         "-frames:v", "1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-500:])


class TestBurnIntegration:
    def test_burn_preserves_dims_and_audio(self, tmp_path):
        from app.media import probe as real_probe

        src = str(tmp_path / "src.mp4")
        png = str(tmp_path / "card.png")
        out = str(tmp_path / "out.mp4")
        _make_video(src, 2.0, 320, 240, 30, with_audio=True)
        _make_png(png, 80, 40)

        spec = OverlaySpec(asset_path=png, nx=0.1, ny=0.1, sw=0.25,
                           start_sec=0.2, end_sec=1.5, fade_sec=0.2)
        burn_overlays(src, out, [spec])

        assert os.path.exists(out)
        info = real_probe(out)
        assert info.width == 320 and info.height == 240
        assert info.has_audio is True
