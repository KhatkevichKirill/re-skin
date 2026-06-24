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

# Cap ffmpeg/x264 worker threads during the stitch re-encode. More threads hold
# more in-flight frames (1080p frames are ~3 MB each), which on a memory-capped
# container can trip the OOM killer (ffmpeg dies with rc=-9). 2 keeps memory
# bounded and matches the worker's CPU cap. Override via FFMPEG_THREADS.
_FFMPEG_THREADS = os.environ.get("FFMPEG_THREADS", "2")

# Cap the stitched output frame rate. 60fps sources double the stitch frame
# count (CPU + memory) for content that gains almost nothing from it, and the
# AI clips are typically <=30fps anyway. Sources at/below the cap are untouched.
# Set FFMPEG_MAX_FPS=0 to disable the cap (keep the source fps).
_MAX_FPS = float(os.environ.get("FFMPEG_MAX_FPS", "30"))

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
    audio_mode: str = "original",
) -> None:
    """
    Concatenate *clips* into *dst*, normalizing each to *width*×*height*@*fps*.

    Parameters
    ----------
    clips:
        Ordered list of clip paths to concatenate.
    audio_source:
        Path to the source video whose full audio track is used when
        ``audio_mode="original"``.  Ignored when ``audio_mode="seedance"``.
    dst:
        Output path for the stitched video.
    width, height, fps:
        Target output dimensions / frame rate.
    audio_mode:
        ``"original"`` (default) — mux the full continuous audio track from
        *audio_source* over the concatenated video (``-shortest``).  This is the
        existing behaviour; Seedance clip audio is discarded.

        ``"seedance"`` — take audio from each clip itself: swap-segment clips use
        the Seedance result audio; keep-segment clips use the original cut audio.
        Each clip's audio is resampled to 44 100 Hz stereo so the concat filter
        has consistent stream parameters.  Clips that have no audio stream get a
        silent audio track synthesised via ``aevalsrc=0`` so the concat never
        fails on a missing stream.  *audio_source* is not used in this mode.

    Raises
    ------
    MediaError
        On any ffmpeg failure or if *clips* is empty.
    ValueError
        If *audio_mode* is not ``"original"`` or ``"seedance"``.
    """
    if audio_mode not in ("original", "seedance"):
        raise ValueError(
            f"stitch: audio_mode must be 'original' or 'seedance', got {audio_mode!r}"
        )
    if not clips:
        raise MediaError("stitch: clips list is empty")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    n = len(clips)

    # ------------------------------------------------------------------
    # audio_mode == "original" — EXACTLY the existing behaviour.
    # ------------------------------------------------------------------
    if audio_mode == "original":
        cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS]
        for clip in clips:
            cmd += ["-i", clip]
        cmd += ["-i", audio_source]

        audio_input_idx = n

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
            "stitch[original]: %d clips -> %s  (%dx%d @ %.2ffps)  audio_source=%s",
            n, dst, width, height, fps, audio_source,
        )
        _run(cmd)
        return

    # ------------------------------------------------------------------
    # audio_mode == "seedance" — use each clip's own audio.
    # Clips without an audio stream get synthesised silence.
    # ------------------------------------------------------------------
    # Probe each clip once to know which ones have audio.
    clip_has_audio: List[bool] = []
    for clip in clips:
        try:
            info = probe(clip)
            clip_has_audio.append(info.has_audio)
        except MediaError:
            clip_has_audio.append(False)

    # Count clips that need a silence source input.
    # We will add one extra input per audio-less clip (anullsrc).
    silence_input_indices: List[int] = []
    cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS]
    for clip in clips:
        cmd += ["-i", clip]

    # Add anullsrc inputs for each clip that has no audio.
    next_input_idx = n
    extra_audio_map: List[int] = []  # index of the audio source for each clip
    for has_a in clip_has_audio:
        if has_a:
            extra_audio_map.append(-1)  # will use [i:a] from the clip itself
        else:
            # Add a virtual anullsrc input.
            cmd += [
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            ]
            silence_input_indices.append(next_input_idx)
            extra_audio_map.append(next_input_idx)
            next_input_idx += 1

    # Build filter_complex.
    # For each clip:
    #   video: [i:v]scale/pad/setsar/fps[vi]
    #   audio: [i:a]aresample...[ai]  OR  [silence_idx:a]atrim=duration=...[ai]
    filter_parts_s: List[str] = []
    silence_ptr = 0
    for i, (clip, has_a) in enumerate(zip(clips, clip_has_audio)):
        # Video normalisation (identical to original mode).
        filter_parts_s.append(
            f"[{i}:v]"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"fps={fps}"
            f"[v{i}]"
        )
        if has_a:
            # Normalise real audio to consistent format.
            filter_parts_s.append(
                f"[{i}:a]aresample=async=1,"
                f"aformat=sample_rates=44100:channel_layouts=stereo"
                f"[a{i}]"
            )
        else:
            # Silence: use anullsrc input, probe the clip duration to trim it
            # so the audio matches the video length (avoids duration mismatch).
            try:
                info = probe(clip)
                clip_dur = info.duration_sec
            except MediaError:
                clip_dur = 5.0  # safe fallback

            silence_idx = silence_input_indices[silence_ptr]
            silence_ptr += 1
            filter_parts_s.append(
                f"[{silence_idx}:a]"
                f"atrim=duration={clip_dur},"
                f"aformat=sample_rates=44100:channel_layouts=stereo"
                f"[a{i}]"
            )

    # Concat: v=1 a=1 — interleaved [v0][a0][v1][a1]...
    concat_inputs_s = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_parts_s.append(
        f"{concat_inputs_s}concat=n={n}:v=1:a=1[vout][aout]"
    )

    filter_complex_s = ";".join(filter_parts_s)

    cmd += [
        "-filter_complex", filter_complex_s,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        dst,
    ]

    log.info(
        "stitch[seedance]: %d clips -> %s  (%dx%d @ %.2ffps)",
        n, dst, width, height, fps,
    )
    _run(cmd)


def get_default_target(info: MediaInfo) -> Tuple[int, int, float]:
    """
    Return ``(width, height, fps)`` that the final stitched video should use.

    The source's native resolution + fps, so AI-processed clips (which may be
    downscaled/retimed by Seedance) are normalized to match the original — except
    fps is capped at ``FFMPEG_MAX_FPS`` (default 30): a 60fps source is stitched
    at 30fps to roughly halve the stitch cost, while sources at/below the cap are
    left untouched.
    """
    fps = info.fps
    if _MAX_FPS and fps and fps > _MAX_FPS:
        fps = _MAX_FPS
    return (info.width, info.height, fps)
