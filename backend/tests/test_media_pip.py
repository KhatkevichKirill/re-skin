"""
Tests for the PIP-insert building blocks in app/media.py:
crop_clip, make_rounded_mask, overlay_insert.

All tests use small synthetic videos generated with ffmpeg lavfi sources
(color/testsrc). No large external files required. Mirrors the synthetic-clip
approach used in test_media.py.

Run: pytest tests/test_media_pip.py -v
"""

from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np
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
    crop_clip,
    make_rounded_mask,
    overlay_insert,
    probe,
)

FFMPEG = "/usr/bin/ffmpeg"


# ---------------------------------------------------------------------------
# Synthetic clip helpers
# ---------------------------------------------------------------------------

def _make_color_video(
    path: str, color: str, duration: float, width: int, height: int,
    fps: int = 30, with_audio: bool = False,
) -> None:
    """Generate a solid-color synthetic video using ffmpeg lavfi."""
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if with_audio:
        cmd += [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
        ]
    else:
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-an"]
    cmd.append(path)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Could not create test video: {result.stderr.decode()[-500:]}")


def _make_two_color_video(
    path: str, color_a: str, color_b: str, seg_dur: float, width: int, height: int,
    fps: int = 30,
) -> None:
    """Concat two solid-color segments into one clip (for PTS-shift tests)."""
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", f"color=c={color_a}:size={width}x{height}:rate={fps}:duration={seg_dur}",
        "-f", "lavfi", "-i", f"color=c={color_b}:size={width}x{height}:rate={fps}:duration={seg_dur}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-an",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Could not create test video: {result.stderr.decode()[-500:]}")


def _extract_frame(video: str, t: float, out_png: str) -> None:
    cmd = [FFMPEG, "-y", "-ss", str(t), "-i", video, "-frames:v", "1", out_png]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Could not extract frame: {result.stderr.decode()[-500:]}")


def _pixel_bgr(png_path: str, x: int, y: int):
    img = cv2.imread(png_path)
    assert img is not None, f"Could not read {png_path}"
    return img[y, x]  # numpy indexing is [row, col] = [y, x]


def _assert_close(actual, expected, tol=40):
    for a, e in zip(actual, expected):
        assert abs(int(a) - int(e)) <= tol, f"pixel {actual} not close to expected {expected}"


# BGR references for ffmpeg's named colors (as produced by libx264 + crf28).
RED_BGR = (0, 0, 255)
GREEN_BGR = (0, 128, 0)
YELLOW_BGR = (0, 255, 255)
BLUE_BGR = (255, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_video(tmp_path_factory):
    """4s 320x240@30fps red video with audio."""
    p = tmp_path_factory.mktemp("pip_base") / "base.mp4"
    _make_color_video(str(p), "red", duration=4.0, width=320, height=240, fps=30, with_audio=True)
    return str(p)


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    """4s 400x300@30fps blue video, source for crop_clip tests."""
    p = tmp_path_factory.mktemp("pip_src") / "source.mp4"
    _make_color_video(str(p), "blue", duration=4.0, width=400, height=300, fps=30, with_audio=True)
    return str(p)


# ---------------------------------------------------------------------------
# crop_clip() tests
# ---------------------------------------------------------------------------

class TestCropClip:
    def test_dimensions_and_duration(self, source_video, tmp_path):
        dst = str(tmp_path / "cropped.mp4")
        w, h = crop_clip(source_video, start_sec=0.5, end_sec=2.5, x=10, y=10,
                          width=100, height=80, dst=dst)
        assert (w, h) == (100, 80)
        info = probe(dst)
        assert info.width == 100
        assert info.height == 80
        assert abs(info.duration_sec - 2.0) <= 0.2

    def test_odd_dims_rounded_down(self, source_video, tmp_path):
        dst = str(tmp_path / "cropped_odd.mp4")
        w, h = crop_clip(source_video, start_sec=0.0, end_sec=1.0, x=5, y=5,
                          width=101, height=81, dst=dst)
        assert (w, h) == (100, 80)
        info = probe(dst)
        assert info.width == 100
        assert info.height == 80

    def test_odd_xy_preserved(self, source_video, tmp_path):
        """x/y may stay odd — only width/height are evened."""
        dst = str(tmp_path / "cropped_xy.mp4")
        w, h = crop_clip(source_video, start_sec=0.0, end_sec=1.0, x=7, y=9,
                          width=50, height=50, dst=dst)
        assert (w, h) == (50, 50)
        assert os.path.exists(dst) and os.path.getsize(dst) > 0

    def test_rect_outside_source_raises(self, source_video, tmp_path):
        dst = str(tmp_path / "bad.mp4")
        with pytest.raises(MediaError):
            crop_clip(source_video, start_sec=0.0, end_sec=1.0, x=350, y=250,
                      width=100, height=100, dst=dst)

    def test_negative_xy_raises(self, source_video, tmp_path):
        dst = str(tmp_path / "bad2.mp4")
        with pytest.raises(MediaError):
            crop_clip(source_video, start_sec=0.0, end_sec=1.0, x=-5, y=0,
                      width=50, height=50, dst=dst)

    def test_invalid_range_raises(self, source_video, tmp_path):
        dst = str(tmp_path / "bad3.mp4")
        with pytest.raises(MediaError):
            crop_clip(source_video, start_sec=2.0, end_sec=1.0, x=0, y=0,
                      width=50, height=50, dst=dst)


# ---------------------------------------------------------------------------
# make_rounded_mask() tests
# ---------------------------------------------------------------------------

class TestMakeRoundedMask:
    def test_dimensions(self, tmp_path):
        dst = str(tmp_path / "mask.png")
        make_rounded_mask(100, 80, 20, dst)
        img = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        assert img.shape == (80, 100)

    def test_corner_black_center_white(self, tmp_path):
        dst = str(tmp_path / "mask2.png")
        make_rounded_mask(100, 100, 20, dst)
        img = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        assert img[0, 0] == 0, "corner pixel should be black"
        assert img[50, 50] == 255, "center pixel should be white"

    def test_edge_midpoints_white(self, tmp_path):
        dst = str(tmp_path / "mask3.png")
        make_rounded_mask(100, 100, 20, dst)
        img = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        assert img[0, 50] == 255, "top edge-midpoint should be white"
        assert img[99, 50] == 255, "bottom edge-midpoint should be white"
        assert img[50, 0] == 255, "left edge-midpoint should be white"
        assert img[50, 99] == 255, "right edge-midpoint should be white"

    def test_radius_zero_all_white(self, tmp_path):
        dst = str(tmp_path / "mask4.png")
        make_rounded_mask(60, 40, 0, dst)
        img = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        assert (img == 255).all()

    def test_radius_clamped(self, tmp_path):
        """A radius larger than min(w,h)//2 is clamped, not an error."""
        dst = str(tmp_path / "mask5.png")
        make_rounded_mask(40, 40, 1000, dst)
        img = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        assert img.shape == (40, 40)
        assert img[20, 20] == 255  # center still white


# ---------------------------------------------------------------------------
# overlay_insert() tests
# ---------------------------------------------------------------------------

class TestOverlayInsert:
    def test_output_duration_and_dims_match_base(self, base_video, tmp_path):
        insert = str(tmp_path / "insert_green.mp4")
        _make_color_video(insert, "green", duration=2.0, width=100, height=100)
        dst = str(tmp_path / "out.mp4")
        overlay_insert(
            base_video, insert, dst,
            x=50, y=50, start_sec=1.0, end_sec=3.0,
            width=100, height=100, fps=30,
        )
        out = probe(dst)
        base_info = probe(base_video)
        assert abs(out.duration_sec - base_info.duration_sec) <= 0.2
        assert out.width == base_info.width
        assert out.height == base_info.height

    def test_base_audio_present(self, base_video, tmp_path):
        insert = str(tmp_path / "insert_green2.mp4")
        _make_color_video(insert, "green", duration=2.0, width=100, height=100)
        dst = str(tmp_path / "out_audio.mp4")
        overlay_insert(
            base_video, insert, dst,
            x=50, y=50, start_sec=1.0, end_sec=3.0,
            width=100, height=100, fps=30,
        )
        out = probe(dst)
        assert out.has_audio is True

    def test_pixel_inside_and_outside_window(self, base_video, tmp_path):
        """Center of insert rect shows insert color inside [start,end], base
        color outside it."""
        insert = str(tmp_path / "insert_green3.mp4")
        _make_color_video(insert, "green", duration=2.0, width=100, height=100)
        dst = str(tmp_path / "out_px.mp4")
        overlay_insert(
            base_video, insert, dst,
            x=50, y=50, start_sec=1.0, end_sec=3.0,
            width=100, height=100, fps=30,
        )

        inside_png = str(tmp_path / "inside.png")
        outside_png = str(tmp_path / "outside.png")
        _extract_frame(dst, 1.5, inside_png)   # inside window -> green
        _extract_frame(dst, 0.3, outside_png)  # before window -> base red

        _assert_close(_pixel_bgr(inside_png, 100, 100), GREEN_BGR)
        _assert_close(_pixel_bgr(outside_png, 100, 100), RED_BGR)

    def test_pts_shift_shows_first_insert_content(self, base_video, tmp_path):
        """A 2-color insert (green then yellow) placed at start_sec=1: the
        frame just after start_sec must show the insert's FIRST content
        (green), not content sampled from later in the insert."""
        insert = str(tmp_path / "insert_2color.mp4")
        _make_two_color_video(insert, "green", "yellow", seg_dur=1.0, width=100, height=100)
        dst = str(tmp_path / "out_pts.mp4")
        overlay_insert(
            base_video, insert, dst,
            x=50, y=50, start_sec=1.0, end_sec=3.0,
            width=100, height=100, fps=30,
        )

        early_png = str(tmp_path / "early.png")
        late_png = str(tmp_path / "late.png")
        _extract_frame(dst, 1.05, early_png)  # just after start -> insert's FIRST half (green)
        _extract_frame(dst, 2.05, late_png)   # into second half of window -> yellow

        _assert_close(_pixel_bgr(early_png, 100, 100), GREEN_BGR)
        _assert_close(_pixel_bgr(late_png, 100, 100), YELLOW_BGR)

    def test_short_insert_padded_by_freezing_last_frame(self, base_video, tmp_path):
        """Insert shorter than the [start,end] window is padded by freezing
        its last frame (tpad), not left blank/frozen-on-base."""
        insert = str(tmp_path / "insert_short.mp4")
        _make_color_video(insert, "yellow", duration=1.0, width=100, height=100)
        dst = str(tmp_path / "out_pad.mp4")
        overlay_insert(
            base_video, insert, dst,
            x=50, y=50, start_sec=1.0, end_sec=3.0,  # 2s window, 1s insert
            width=100, height=100, fps=30,
        )
        late_png = str(tmp_path / "late_pad.png")
        _extract_frame(dst, 2.5, late_png)  # past insert's native 1s length
        _assert_close(_pixel_bgr(late_png, 100, 100), YELLOW_BGR)

    def test_mask_corner_shows_base_color(self, base_video, tmp_path):
        """With a rounded mask, the extreme corner pixel of the insert rect
        shows the base color (masked out), while the center shows insert
        color."""
        insert = str(tmp_path / "insert_green_mask.mp4")
        _make_color_video(insert, "green", duration=2.0, width=100, height=100)
        mask_png = str(tmp_path / "mask.png")
        make_rounded_mask(100, 100, 20, mask_png)

        dst = str(tmp_path / "out_mask.mp4")
        overlay_insert(
            base_video, insert, dst,
            x=50, y=50, start_sec=1.0, end_sec=3.0,
            width=100, height=100, fps=30,
            mask_png=mask_png,
        )

        frame_png = str(tmp_path / "mask_frame.png")
        _extract_frame(dst, 1.5, frame_png)

        # corner of the insert rect (x=50,y=50 top-left in base coords)
        _assert_close(_pixel_bgr(frame_png, 50, 50), RED_BGR)
        # center of the insert rect (x=100,y=100)
        _assert_close(_pixel_bgr(frame_png, 100, 100), GREEN_BGR)

    def test_mask_does_not_freeze_insert(self, base_video, tmp_path):
        """The mask PNG is a single-frame input; alphamerge must not stall
        the insert at its first frame when the mask input hits EOF (framesync
        repeatlast). A two-color insert (green then yellow) under a rounded
        mask must still play through: green just after start_sec, yellow
        later in the window — while the masked corner shows base color in
        both frames."""
        insert = str(tmp_path / "insert_2color_mask.mp4")
        _make_two_color_video(insert, "green", "yellow", seg_dur=1.0, width=100, height=100)
        mask_png = str(tmp_path / "mask_2color.png")
        make_rounded_mask(100, 100, 20, mask_png)

        dst = str(tmp_path / "out_mask_2color.mp4")
        overlay_insert(
            base_video, insert, dst,
            x=50, y=50, start_sec=1.0, end_sec=3.0,
            width=100, height=100, fps=30,
            mask_png=mask_png,
        )

        early_png = str(tmp_path / "mask_early.png")
        late_png = str(tmp_path / "mask_late.png")
        _extract_frame(dst, 1.05, early_png)  # first half of insert -> green
        _extract_frame(dst, 2.05, late_png)   # second half -> yellow (not frozen green)

        _assert_close(_pixel_bgr(early_png, 100, 100), GREEN_BGR)
        _assert_close(_pixel_bgr(late_png, 100, 100), YELLOW_BGR)
        # masked-out corner of the insert rect stays base color in both frames
        _assert_close(_pixel_bgr(early_png, 50, 50), RED_BGR)
        _assert_close(_pixel_bgr(late_png, 50, 50), RED_BGR)

    def test_invalid_range_raises(self, base_video, tmp_path):
        insert = str(tmp_path / "insert_bad.mp4")
        _make_color_video(insert, "green", duration=1.0, width=100, height=100)
        dst = str(tmp_path / "bad.mp4")
        with pytest.raises(MediaError):
            overlay_insert(
                base_video, insert, dst,
                x=0, y=0, start_sec=2.0, end_sec=1.0,
                width=100, height=100, fps=30,
            )

    def test_negative_start_raises(self, base_video, tmp_path):
        insert = str(tmp_path / "insert_bad2.mp4")
        _make_color_video(insert, "green", duration=1.0, width=100, height=100)
        dst = str(tmp_path / "bad2.mp4")
        with pytest.raises(MediaError):
            overlay_insert(
                base_video, insert, dst,
                x=0, y=0, start_sec=-1.0, end_sec=1.0,
                width=100, height=100, fps=30,
            )
