"""
media.py — FFmpeg building blocks for re-skin.

Provides: probe, cut_clip, cut_segments, stitch, get_default_target.
All heavy work shells out to /usr/bin/ffmpeg and /usr/bin/ffprobe.
Raises MediaError on any ffmpeg/ffprobe failure (includes stderr tail).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple

log = logging.getLogger(__name__)

FFMPEG = os.environ.get("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "/usr/bin/ffprobe")

# Maximum lines of stderr to include in MediaError messages.
_STDERR_TAIL = 20


class MediaError(RuntimeError):
    """Raised when an ffmpeg/ffprobe subprocess fails."""


@dataclass
class MediaInfo:
    duration_sec: float
    width: int
    height: int
    fps: float
    aspect_ratio: str
    has_audio: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing stdout + stderr. Raises MediaError on failure."""
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-_STDERR_TAIL:])
        raise MediaError(
            f"Command failed (rc={result.returncode}): {' '.join(cmd)}\n--- stderr ---\n{tail}"
        )
    return result


def _parse_fps(r_frame_rate: str) -> float:
    """Convert a fractional frame-rate string like '30000/1001' to a float."""
    try:
        return float(Fraction(r_frame_rate))
    except (ValueError, ZeroDivisionError) as exc:
        raise MediaError(f"Cannot parse fps from {r_frame_rate!r}: {exc}") from exc


def _compute_aspect_ratio(width: int, height: int) -> str:
    """Return a simplified 'W:H' aspect-ratio string, e.g. '16:9'."""
    from math import gcd
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe(path: str) -> MediaInfo:
    """
    Return metadata for *path* using ffprobe.

    Parses the first video stream for resolution / fps and checks whether any
    audio stream is present.
    """
    cmd = [
        FFPROBE,
        "-v", "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,duration"
        ":format=duration",
        "-of", "json",
        path,
    ]
    result = _run(cmd)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned non-JSON output for {path!r}: {exc}") from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise MediaError(f"No video stream found in {path!r}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    # Duration: prefer format-level (most reliable), fall back to stream.
    raw_duration = fmt.get("duration") or video_stream.get("duration")
    if raw_duration is None:
        raise MediaError(f"Cannot determine duration for {path!r}")
    duration_sec = float(raw_duration)

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps = _parse_fps(video_stream["r_frame_rate"])
    aspect_ratio = _compute_aspect_ratio(width, height)

    info = MediaInfo(
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=fps,
        aspect_ratio=aspect_ratio,
        has_audio=has_audio,
    )
    log.info("probe(%s) -> %s", path, info)
    return info


def cut_clip(src: str, start_sec: float, end_sec: float, dst: str) -> None:
    """
    Cut *src* from *start_sec* to *end_sec* into *dst*.

    Uses re-encoding (libx264/aac) to guarantee frame-accurate cuts at
    non-keyframe boundaries and avoids near-empty files at EOF.

    Note: ``-ss`` is placed *before* ``-i`` for fast seeking; ``-to`` is the
    duration relative to the seek point (= end_sec - start_sec).
    """
    duration = end_sec - start_sec
    if duration <= 0:
        raise MediaError(
            f"cut_clip: end_sec ({end_sec}) must be greater than start_sec ({start_sec})"
        )

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",
        "-ss", str(start_sec),
        "-i", src,
        "-to", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        dst,
    ]
    log.info("cut_clip: %s [%.3f, %.3f] -> %s", src, start_sec, end_sec, dst)
    _run(cmd)


def cut_segments(
    src: str,
    ranges: List[Tuple[float, float]],
    out_dir: str,
) -> List[str]:
    """
    Cut multiple time ranges from *src* into *out_dir*.

    Returns ordered list of output paths (``seg_000.mp4``, ``seg_001.mp4``, …).
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for idx, (start, end) in enumerate(ranges):
        dst = os.path.join(out_dir, f"seg_{idx:03d}.mp4")
        cut_clip(src, start, end, dst)
        paths.append(dst)
    log.info("cut_segments: produced %d segments in %s", len(paths), out_dir)
    return paths


def stitch(
    clips: List[str],
    audio_source: str,
    dst: str,
    width: int,
    height: int,
    fps: float,
) -> None:
    """
    Concatenate *clips* into *dst*, normalizing each to *width*×*height*@*fps*.

    Steps
    -----
    1. For each clip build a filter chain:
       ``scale=W:H:force_original_aspect_ratio=decrease,
         pad=W:H:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=FPS``
    2. Concat all video-only streams.
    3. Mux the FULL audio track from *audio_source* (not from the clips).
    4. Encode with libx264 / aac and ``-shortest``.
    """
    if not clips:
        raise MediaError("stitch: clips list is empty")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    # Build the ffmpeg command.
    # -i 0 … -i N-1 are the clips; -i N is the audio source.
    cmd = [FFMPEG, "-y"]
    for clip in clips:
        cmd += ["-i", clip]
    cmd += ["-i", audio_source]

    audio_input_idx = len(clips)

    # Build filter_complex.
    n = len(clips)
    filter_parts: List[str] = []
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"fps={fps}"
            f"[v{i}]"
        )

    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vout]")

    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{audio_input_idx}:a",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        dst,
    ]

    log.info(
        "stitch: %d clips -> %s  (%dx%d @ %.2ffps)  audio_source=%s",
        n, dst, width, height, fps, audio_source,
    )
    _run(cmd)


def get_default_target(info: MediaInfo) -> Tuple[int, int, float]:
    """
    Return ``(width, height, fps)`` that the final stitched video should use.

    Always the source's native resolution + fps so that AI-processed clips
    (which may be downscaled/retimed by Seedance) are normalized *up* to match
    the untouched original.
    """
    return (info.width, info.height, info.fps)
