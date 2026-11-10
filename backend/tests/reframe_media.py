"""Synthetic media for the reframe tests (real ffmpeg, lavfi sources)."""

from __future__ import annotations

import subprocess


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_source(path: str, *, w: int = 1280, h: int = 720, fps: int = 30, sec: float = 6.0,
                audio: bool = True) -> str:
    """A testsrc2 clip with a 440 Hz tone (or none)."""
    args = ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={sec}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={sec}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    ffmpeg(*args, path)
    return path


def make_raw(path: str, *, w: int = 768, h: int = 1344, frames: int = 120,
             audio: bool = True) -> str:
    """What Fal returns: 24 fps at the native size, the source audio kept."""
    sec = frames / 24
    args = ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24:duration={sec}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={sec}",
                 "-c:a", "aac"]
    ffmpeg(*args, "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", path)
    return path


def make_flac(path: str, *, sec: float, freq: int = 440, rate: int = 48000) -> str:
    ffmpeg("-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={rate}:duration={sec}",
           "-c:a", "flac", path)
    return path


def frame_luma(path: str, n: int) -> float:
    """Mean luma (0-255) of frame *n*."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf",
         f"select='eq(n\\,{n})',scale=32:32,format=gray", "-frames:v", "1",
         "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return sum(out) / max(1, len(out))
