"""
media.py — FFmpeg building blocks for re-skin.

Provides: probe, normalize_source, cut_clip, cut_segments, stitch, get_default_target,
crop_clip, make_rounded_mask, overlay_insert (PIP-insert extract/merge
building blocks — see docs/pip-insert.md), burn_subtitles (subtitle burn-in
— see docs/burned-captions.md).
All heavy work shells out to /usr/bin/ffmpeg and /usr/bin/ffprobe.
Raises MediaError on any ffmpeg/ffprobe failure (includes stderr tail).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple

import cv2
import numpy as np

# The one app-level import this module takes, and deliberately so: the voice
# reference bounds make_voice_reference encodes to are *model API* facts, which
# live in the registry with every other one. ai_models imports nothing beyond
# the stdlib, so this stays acyclic and keeps media.py standalone-importable.
from app import ai_models

log = logging.getLogger(__name__)

# Voice-reference bounds, re-exported so callers reading this module see them
# next to the function that enforces them. Single source: app.ai_models.
VOICE_REF_MIN_SEC = ai_models.VOICE_REF_MIN_SEC
VOICE_REF_MAX_SEC = ai_models.VOICE_REF_MAX_SEC

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

# Cap on the source video's short side, enforced once at ingest by
# normalize_source(). The worker memory sizing (docker-compose mem_limit,
# FFMPEG_THREADS) assumes <=1080p 8-bit frames; a 4K 10-bit source holds
# ~25 MB/frame and two concurrent cut_clip calls then trip the cgroup OOM
# killer (ffmpeg rc=-9). Set NORMALIZE_MAX_SHORT_SIDE=0 to disable ingest
# normalization entirely.
_NORMALIZE_MAX_SHORT_SIDE = int(os.environ.get("NORMALIZE_MAX_SHORT_SIDE", "1080"))

# Pixel formats normalize_source() accepts as-is; anything else (10-bit,
# 4:2:2/4:4:4, ...) is converted to yuv420p at ingest.
_NORMALIZED_PIX_FMTS = ("yuv420p", "yuvj420p")

# --- LLM upload proxy (make_llm_proxy) ---
# A transcription model watches a video to hear what is said and to see which
# person on screen says it. Neither job needs delivery resolution, delivery
# frame rate or delivery bitrate, but both the kie.ai upload host (which
# refuses a large body outright) and the model's per-input-token bill are sized
# by exactly the file we hand over. So transcription uploads a deliberately
# cheap re-encode instead of the source: normalize_source() caps resolution but
# not bitrate, and a 59s 1080x1920 phone capture at ~19 Mbps is 136 MB — over
# the upload ceiling and pure waste besides.
# Set LLM_PROXY_SHORT_SIDE=0 or LLM_PROXY_FPS=0 to disable that dimension.
_LLM_PROXY_SHORT_SIDE = int(os.environ.get("LLM_PROXY_SHORT_SIDE", "480"))
_LLM_PROXY_FPS = float(os.environ.get("LLM_PROXY_FPS", "15"))
_LLM_PROXY_CRF = os.environ.get("LLM_PROXY_CRF", "30")
# Mono, and low — but not as low as it could go. Audio is the signal the
# transcript is actually built from, while video is ~85% of the proxy's bytes,
# so squeezing audio buys almost nothing in size and costs accuracy. Spend here,
# save on the picture.
_LLM_PROXY_AUDIO_BPS = 96_000

# Maximum lines of stderr to include in MediaError messages.
_STDERR_TAIL = 20

# ---------------------------------------------------------------------------
# Audio window splice (splice_audio_window)
# ---------------------------------------------------------------------------
# Re-voicing ONE word of an already-delivered video from a sibling run's audio.
# Everything here is a constant rather than a caller argument because these are
# properties of the operation, not of the edit an operator is making.
#
# The gain envelope is evaluated once per audio frame, so the frame has to be
# short or the crossfade ramps in steps you can hear; 128 samples is ~2.9 ms.
_AUDIO_SPLICE_GAIN_FRAME_SAMPLES = 128
# Both branches are conformed to the BASE's own layout before amix — the base
# and the donor are separate deliveries and need not already agree. This is the
# fallback for a base whose sample rate cannot be read; the layout is never
# guessed, it is probed (see _audio_layout).
_AUDIO_SPLICE_FALLBACK_SAMPLE_RATE = 44_100
# The whole track is re-encoded (only the picture is stream-copied), so this is
# the bitrate of the DELIVERED audio, not of a temporary — matched to
# mux_original_audio's, which is the other place a finished video's soundtrack
# is rebuilt.
_AUDIO_SPLICE_AUDIO_BITRATE_BPS = 192_000
# Default seam length. Long enough that a hard cut never clicks, short enough
# that it cannot swallow the first or last phoneme of the replaced word.
# Public — it is also the New-patch form's default.
AUDIO_SPLICE_DEFAULT_CROSSFADE_SEC = 0.06
# A ramp is never shortened past this, even for a very tight window — below a
# couple of milliseconds it stops being a ramp and starts being the click it
# exists to prevent.
_AUDIO_SPLICE_MIN_CROSSFADE_SEC = 0.005
# Shorter than this and there is no room for two ramps plus replaced
# material. Public (like the default above): pipeline_audio_patch validates
# an operator's window against it before a row exists, so the 400 and the
# ffmpeg-level MediaError can never disagree about the minimum.
AUDIO_SPLICE_MIN_WINDOW_SEC = 0.05
# Slack when comparing a requested bound against a probed duration: a container
# duration and its audio stream's duration differ by an AAC frame or two, and
# an operator who scrubs to the very end should not get a validation error for
# a rounding difference they cannot see.
_AUDIO_SPLICE_BOUND_EPS_SEC = 0.05


# Bound on concurrent ffprobe subprocesses when stitch() probes every clip up
# front (to learn total duration, and in "seedance" audio mode has_audio for
# silence synthesis). probe() is subprocess-bound (shells out to ffprobe), so
# threads parallelize it fine — a 30-segment stitch otherwise runs 30 serial
# ffprobe round-trips before ffmpeg even starts. Mirrors the small bounded
# pools already used elsewhere in the pipeline (e.g. pipeline_v2.py's
# STITCH_CUT_CONCURRENCY). Override via STITCH_PROBE_CONCURRENCY.
_STITCH_PROBE_CONCURRENCY = int(os.environ.get("STITCH_PROBE_CONCURRENCY", "4"))

# ---------------------------------------------------------------------------
# Delivery size cap
# ---------------------------------------------------------------------------
# Hard ceiling on a delivered video's file size. Every encode that produces a
# file we hand to Google Drive (the stitched run result, a tail edit, an
# overlay burn) is VBV-capped to fit under it — see delivery_video_opts().
# Decimal megabytes (1e6), not MiB, so "100 MB" is read the strict way.
# Set DELIVERY_MAX_SIZE_MB=0 to disable the cap and encode on CRF alone.
_DELIVERY_MAX_BYTES = int(float(os.environ.get("DELIVERY_MAX_SIZE_MB", "100")) * 1_000_000)

# Quality target for a delivery encode. x264 CRF — lower is better quality and
# a bigger file. 20 is visually transparent on this (AI-generated, detailed)
# 1080x1920 content while producing ~30% smaller files than the CRF 18 the
# stitch used to hard-code. The cap below only engages when even CRF 20 would
# overshoot, so easy content is never padded up to the budget.
#
# The preset stays at "fast" — the same one the stitch already used — so the
# size win costs no extra encode time on the critical path. "medium" buys
# roughly another 10% off the file for ~2x the stitch phase; worth setting
# only if the cap is being hit regularly.
_DELIVERY_CRF = os.environ.get("DELIVERY_CRF", "20")
_DELIVERY_PRESET = os.environ.get("DELIVERY_PRESET", "fast")

# The VBV buffer, expressed in seconds of -maxrate. x264 guarantees the stream
# never exceeds maxrate over any bufsize-long window, so total video bits are
# bounded by maxrate*duration + bufsize — the budget maths below subtracts
# this slack explicitly rather than hiding it in a fudge factor.
_DELIVERY_VBV_SECONDS = 2.0

# Fraction of the byte budget available to the elementary streams; the rest is
# mp4 container overhead (headers, per-frame index). ~1% is typical at these
# durations; 2% leaves room for short clips where the moov atom weighs more.
_DELIVERY_CONTAINER_EFFICIENCY = 0.98

# Floor on the capped video bitrate. Below this a 1080x1920 encode stops being
# watchable, so a video long enough to need it keeps the floor and is allowed
# to exceed the size cap (logged) rather than being destroyed to fit. At the
# default 100 MB / 600 kbps that threshold is ~22 minutes of video.
_DELIVERY_MIN_VIDEO_BPS = int(os.environ.get("DELIVERY_MIN_VIDEO_BPS", "600000"))

# Audio bitrates, in bits per second, that have to be charged against the size
# budget: the stitch muxes 192k, the enforce_size_limit() rescue drops to 128k.
_STITCH_AUDIO_BPS = 192_000
_DELIVERY_SHRINK_AUDIO_BPS = 128_000


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
    # Defaults to the normalized format so hand-built instances (tests, stubs)
    # are treated as already pipeline-safe; probe() always sets the real value.
    pix_fmt: str = "yuv420p"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# AICP-160: inputs reach ffmpeg/ffprobe from uploads and Drive downloads, so a
# file named ``x.mp4`` that is really an HLS playlist / concat script / ``.sdp``
# must not be demuxed as one (it would pull in other local files or, on a
# build with network protocols, remote URLs).  Two allow-lists, applied to every
# input by :func:`harden_media_argv`: URL protocols (``file,pipe``) and demuxers.
# The protocol list alone does NOT stop a local HLS playlist -- ``file`` is
# enough for it to read sibling files -- hence the demuxer list as well.
PROTOCOL_WHITELIST = "file,pipe"
FORMAT_WHITELIST = ",".join([
    # containers
    "mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "avi", "flv",
    "mpegts", "mpeg", "mpegvideo", "asf", "ogg", "nut", "mxf", "gxf", "rm",
    # audio
    "wav", "mp3", "aac", "flac", "ac3", "eac3", "opus", "caf", "aiff", "amr",
    "w64", "wsd", "s16le",
    # raw video elementary streams
    "m4v", "h264", "hevc", "ivf",
    # stills (overlay PNGs, reference frames)
    "image2", "png_pipe", "jpeg_pipe", "webp_pipe", "gif", "bmp_pipe",
    # synthetic sources built by this module (anullsrc / color)
    "lavfi",
])
_HARDEN_OPTS = [
    "-protocol_whitelist", PROTOCOL_WHITELIST,
    "-format_whitelist", FORMAT_WHITELIST,
]


def harden_media_argv(cmd: List[str]) -> List[str]:
    """Return *cmd* with the input allow-lists added; *cmd* is not modified.

    ffprobe: the options go right after the binary.  ffmpeg: they go in front
    of every ``-i`` (they are per-input options).  Anything that is not an
    ffmpeg/ffprobe argv is returned unchanged, and an argv that already carries
    ``-protocol_whitelist`` is left alone.
    """
    if not cmd or "-protocol_whitelist" in cmd:
        return list(cmd)
    name = os.path.basename(str(cmd[0]))
    if name.startswith("ffprobe"):
        return [cmd[0], *_HARDEN_OPTS, *cmd[1:]]
    if not name.startswith("ffmpeg"):
        return list(cmd)
    out: List[str] = []
    for tok in cmd:
        if tok == "-i":
            out.extend(_HARDEN_OPTS)
        out.append(tok)
    return out


def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing stdout + stderr. Raises MediaError on failure."""
    cmd = harden_media_argv(cmd)
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


def _escape_filter_path(path: str) -> str:
    """
    Escape *path* for safe interpolation into an ffmpeg filtergraph argument
    value, e.g. ``subtitles=<this>``.

    The filtergraph parser (not the shell — this string lands in a single
    argv element, so shell quoting never enters into it) gives special
    meaning to three characters: ``\\`` is its escape character, ``:``
    separates a filter's ``key=value`` options, and ``'`` opens/closes a
    quoted section. A path containing any of them — a Windows-style path,
    or, more realistically for us, a project/run directory an operator
    renamed to include an apostrophe or was handed with a colon in it —
    would otherwise truncate the filter option at the first unescaped ``:``
    or unbalance the quoting, either silently burning the wrong (empty)
    subtitles or failing ffmpeg outright. Do not rely on our paths being
    uuid-clean; that assumption breaks the day someone renames a project
    directory.

    The multipliers below (4 backslashes for a literal ``\\``, 2 for a
    literal ``:``, 3 for a literal ``'``) look arbitrary but are not: a
    ``subtitles=...`` value is unescaped *twice* before it reaches disk —
    once while ffmpeg's graph parser extracts this filter's argument out of
    the whole ``-vf`` string, and again when the filter's own option parser
    reads the ``filename`` value out of that (colon-delimited) argument.
    Both passes apply the same rule (an unescaped backslash consumes and
    escapes the very next character); one level of backslash-doubling
    survives only the second pass, so anything meant to still be an escape
    by the time the *second* pass runs has to survive the *first* pass too.
    Verified empirically against this repo's ffmpeg build with paths mixing
    all three characters plus spaces, in every adjacency (e.g. ``\\:``,
    ``\\'``, ``:'``, back-to-back quotes) — see backend/tests/test_media.py.
    Backslash is escaped first so the backslashes this function itself
    inserts for ``:`` and ``'`` are not themselves re-escaped on a later
    replacement in this same function.
    """
    s = path.replace("\\", "\\\\\\\\")
    s = s.replace(":", "\\\\:")
    s = s.replace("'", "\\\\\\'")
    return s


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
        "stream=codec_type,width,height,r_frame_rate,duration,pix_fmt"
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
        pix_fmt=video_stream.get("pix_fmt", ""),
    )
    log.info("probe(%s) -> %s", path, info)
    return info


def probe_audio(path: str) -> tuple[bool, float]:
    """Return ``(has_audio, duration_sec)`` for a file that may have no video.

    :func:`probe` raises when there is no video stream — correct for everything
    else in this module, since a "video" with no picture is a broken input. But
    a voice reference is very often an audio-only file (an mp3, a voice memo),
    which that check would reject before the format was even considered.

    Duration is read from the AUDIO stream, falling back to the container's.
    Returns ``(False, 0.0)`` for a file with no audio at all, so the caller
    decides what a silent input means rather than catching an exception for it.
    """
    cmd = [
        FFPROBE,
        "-v", "error",
        "-show_entries", "stream=codec_type,duration:format=duration",
        "-of", "json",
        path,
    ]
    result = _run(cmd)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(
            f"ffprobe returned non-JSON output for {path!r}: {exc}"
        ) from exc

    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if audio_stream is None:
        return False, 0.0

    raw = audio_stream.get("duration") or data.get("format", {}).get("duration")
    if raw is None:
        raise MediaError(f"Cannot determine audio duration for {path!r}")
    return True, float(raw)


# Bumped whenever the decode parameters below change, so an old fingerprint
# can never compare equal to a new one computed from a different decode.
AUDIO_FINGERPRINT_VERSION = "pcm-s16le-16k-mono+av-offset/v1"


def _audio_video_offset(path: str) -> Optional[float]:
    """Audio start minus video start (seconds), or None if either is unknown."""
    result = _run([
        FFPROBE, "-v", "error",
        "-show_entries", "stream=codec_type,start_time",
        "-of", "json", path,
    ], check=False)
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    starts = {}
    for st in streams:
        kind = st.get("codec_type")
        if kind in ("audio", "video") and kind not in starts:
            try:
                starts[kind] = float(st.get("start_time"))
            except (TypeError, ValueError):
                return None
    # Both are required: an unknown picture start means an unknown alignment,
    # and a fingerprint must never claim one it does not know.
    if "audio" not in starts or "video" not in starts:
        return None
    return starts["audio"] - starts["video"]


def audio_fingerprint(path: str) -> Optional[str]:
    """Return a fingerprint of *path*'s first audio track, or ``None``.

    sha256 over the version tag, the audio-vs-video start offset (ms) and the
    DECODED samples (mono, 16 kHz, s16le) — not the container or compressed
    packets, so two finals whose picture differs but whose soundtrack is the
    same (every face_swap run of one project under ``audio_mode="original"``)
    fingerprint equal. The offset is in the key because caption timings are
    read against the picture: equal samples starting at a different point of
    the timeline would put every caption off by that much. Used by
    pipeline_captions to reuse a sibling's transcript instead of re-running
    Gemini on the same audio.

    Exact equality only: a single differing sample is a different fingerprint,
    which costs one transcription and never a wrong caption. ``None`` for a
    file with no audio or any probe/decode failure — the caller transcribes.
    """
    offset = _audio_video_offset(path)
    if offset is None:
        log.info("audio_fingerprint: no audio stream / start time in %s", path)
        return None
    cmd = [
        FFMPEG, "-nostdin", "-v", "error",
        "-i", path,
        "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "16000", "-f", "s16le",
        "-",
    ]
    digest = hashlib.sha256(
        f"{AUDIO_FINGERPRINT_VERSION}|{round(offset * 1000)}|".encode()
    )
    n_bytes = 0
    # stderr goes to a file, not a pipe: only stdout is drained while ffmpeg
    # runs, and a chatty failure filling an unread stderr pipe would block it.
    with tempfile.TemporaryFile() as errf:
        try:
            proc = subprocess.Popen(harden_media_argv(cmd), stdout=subprocess.PIPE, stderr=errf)
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(1 << 16), b""):
                digest.update(chunk)
                n_bytes += len(chunk)
            proc.wait()
        except OSError as exc:
            log.warning("audio_fingerprint: cannot run ffmpeg on %s: %s", path, exc)
            return None
        if proc.returncode != 0 or n_bytes == 0:
            errf.seek(0)
            log.info(
                "audio_fingerprint: no fingerprint for %s (rc=%s, %d bytes): %s",
                path, proc.returncode, n_bytes,
                errf.read().decode(errors="replace").strip()[-300:],
            )
            return None
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Delivery size cap
# ---------------------------------------------------------------------------

def delivery_video_bitrate(
    duration_sec: float,
    *,
    audio_bitrate_bps: int = 0,
    max_bytes: Optional[int] = None,
) -> Optional[int]:
    """
    Highest video bitrate (bits/s) a *duration_sec*-long file can carry and
    still fit in *max_bytes*, or ``None`` when no cap applies.

    Returns ``None`` if the cap is disabled (``max_bytes <= 0``), the duration
    is unknown/zero, or the budget lands below ``_DELIVERY_MIN_VIDEO_BPS`` — in
    that last case the video is too long to fit at a watchable bitrate, so the
    caller encodes uncapped and the file is allowed to exceed the limit.

    The budget accounts for the audio track, the mp4 container overhead and the
    VBV buffer slack, so an encode held to the returned bitrate cannot overshoot
    ``max_bytes``:  ``video_bps * (duration + bufsize_secs) + audio_bits
    <= max_bytes * 8 * efficiency``.
    """
    if max_bytes is None:
        max_bytes = _DELIVERY_MAX_BYTES
    if max_bytes <= 0 or not duration_sec or duration_sec <= 0:
        return None

    budget_bits = max_bytes * 8 * _DELIVERY_CONTAINER_EFFICIENCY
    video_bits = budget_bits - audio_bitrate_bps * duration_sec
    video_bps = int(video_bits / (duration_sec + _DELIVERY_VBV_SECONDS))

    if video_bps < _DELIVERY_MIN_VIDEO_BPS:
        log.warning(
            "delivery cap: %.1fs of video cannot fit in %d bytes above the "
            "%d bps floor (budget would be %d bps) — encoding uncapped, the "
            "output will exceed the size limit",
            duration_sec, max_bytes, _DELIVERY_MIN_VIDEO_BPS, video_bps,
        )
        return None
    return video_bps


def delivery_video_opts(
    duration_sec: Optional[float],
    *,
    audio_bitrate_bps: int = 0,
    max_bytes: Optional[int] = None,
    crf: Optional[str] = None,
    preset: Optional[str] = None,
) -> List[str]:
    """
    ffmpeg video-encoder flags for a file that will be delivered to Drive.

    Quality is CRF-driven, so content that compresses easily is encoded at
    constant quality and comes out well under the size cap instead of being
    padded up to a target bitrate. On top of the CRF sits an x264 VBV cap
    (``-maxrate``/``-bufsize``) sized from *duration_sec* so the muxed file
    cannot exceed the cap even when the content is hard to encode. That is the
    "compress without losing quality" trade: the cap only binds where CRF alone
    would have overshot.

    Callers keep their own audio flags; pass the bitrate they encode audio at
    as *audio_bitrate_bps* so it is charged against the same budget.

    When no cap applies (see :func:`delivery_video_bitrate`) the returned flags
    are a plain CRF encode.
    """
    opts = [
        "-c:v", "libx264",
        "-preset", preset or _DELIVERY_PRESET,
        "-crf", str(crf or _DELIVERY_CRF),
        "-pix_fmt", "yuv420p",
    ]

    video_bps = delivery_video_bitrate(
        duration_sec or 0.0,
        audio_bitrate_bps=audio_bitrate_bps,
        max_bytes=max_bytes,
    )
    if video_bps is None:
        return opts

    opts += [
        "-maxrate", str(video_bps),
        "-bufsize", str(int(video_bps * _DELIVERY_VBV_SECONDS)),
    ]
    return opts


def enforce_size_limit(path: str, *, max_bytes: Optional[int] = None) -> bool:
    """
    Shrink *path* in place if it is still over the delivery size cap.

    A no-op (one ``stat``) for anything produced with
    :func:`delivery_video_opts` — this is the guard for files the cap never
    reached: results stitched before the cap existed and re-delivered by a
    retry, videos uploaded from outside the pipeline, or an encode whose VBV
    was disabled by the bitrate floor.

    Unlike the capped-CRF encode, this pass is pure ABR (``-b:v`` at the
    budget) — the file is already known to be too big, so there is nothing to
    gain from letting CRF pick a higher bitrate. Audio is re-encoded to 128k
    AAC (from the pipeline's 192k) to buy back a little of the budget.

    Returns ``True`` if *path* was re-encoded, ``False`` if it was left alone.
    Raises :class:`MediaError` if the re-encode fails; *path* is untouched in
    that case, so a failure here never destroys a deliverable.
    """
    if max_bytes is None:
        max_bytes = _DELIVERY_MAX_BYTES
    if max_bytes <= 0 or not os.path.exists(path):
        return False

    size = os.path.getsize(path)
    if size <= max_bytes:
        return False

    info = probe(path)
    audio_bps = _DELIVERY_SHRINK_AUDIO_BPS if info.has_audio else 0
    video_bps = delivery_video_bitrate(
        info.duration_sec, audio_bitrate_bps=audio_bps, max_bytes=max_bytes
    )
    if video_bps is None:
        log.warning(
            "enforce_size_limit: %s is %d bytes (> %d) but cannot be capped "
            "— leaving it as is", path, size, max_bytes,
        )
        return False

    tmp = f"{path}.shrink.mp4"
    cmd = [
        FFMPEG, "-y", "-threads", _FFMPEG_THREADS,
        "-i", path,
        "-c:v", "libx264",
        "-preset", _DELIVERY_PRESET,
        "-b:v", str(video_bps),
        "-maxrate", str(video_bps),
        "-bufsize", str(int(video_bps * _DELIVERY_VBV_SECONDS)),
        "-pix_fmt", "yuv420p",
    ]
    if info.has_audio:
        cmd += ["-c:a", "aac", "-b:a", f"{_DELIVERY_SHRINK_AUDIO_BPS // 1000}k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", tmp]

    log.info(
        "enforce_size_limit: %s is %d bytes (> %d) — re-encoding %.1fs at %d bps",
        path, size, max_bytes, info.duration_sec, video_bps,
    )
    try:
        _run(cmd)
    except MediaError:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    new_size = os.path.getsize(tmp)
    os.replace(tmp, path)
    log.info(
        "enforce_size_limit: %s %d -> %d bytes (%.0f%% of original)",
        path, size, new_size, 100.0 * new_size / size,
    )
    return True


def normalize_source(src: str) -> str:
    """
    Re-encode *src* to the pipeline baseline (<=1080p short side, 8-bit
    yuv420p) if it exceeds it; return the path to use afterwards.

    Every later phase (cut_clip, stitch, InsightFace analyze) is memory-sized
    for 1080p 8-bit frames, and the AI models don't benefit from larger input.
    A 4K 10-bit source (e.g. an iPhone HDR .mov) holds ~25 MB raw frames, so a
    single cut needs ~3 GB and two concurrent cuts blow the worker's cgroup
    limit — the OOM killer then SIGKILLs ffmpeg (rc=-9). Normalizing once at
    ingest keeps that sizing valid for any upload.

    A re-encode is triggered when either holds (both fixes applied together):

    - short side > ``NORMALIZE_MAX_SHORT_SIDE`` (default 1080) → downscaled so
      the short side equals the cap, aspect preserved, even dimensions;
    - pixel format outside 8-bit 4:2:0 (``yuv420p``/``yuvj420p``) → converted
      to ``yuv420p``.

    Compliant sources are returned unchanged. Re-encoded output replaces the
    original: it is written as a sibling ``source.mp4`` (mp4 container with
    faststart, for the in-page review player) and the original file is
    removed if its path differs. Set ``NORMALIZE_MAX_SHORT_SIDE=0`` to
    disable normalization entirely.
    """
    if _NORMALIZE_MAX_SHORT_SIDE <= 0:
        return src

    info = probe(src)
    needs_scale = min(info.width, info.height) > _NORMALIZE_MAX_SHORT_SIDE
    needs_pix_fmt = info.pix_fmt not in _NORMALIZED_PIX_FMTS
    if not needs_scale and not needs_pix_fmt:
        return src

    src_dir = os.path.dirname(src) or "."
    dst = os.path.join(src_dir, "source.mp4")
    tmp = os.path.join(src_dir, ".normalizing.mp4")

    cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS, "-i", src]
    if needs_scale:
        cap = _NORMALIZE_MAX_SHORT_SIDE
        scale = f"scale=-2:{cap}" if info.width >= info.height else f"scale={cap}:-2"
        cmd += ["-vf", scale]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if info.has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", tmp]

    log.info(
        "normalize_source: %s (%dx%d %s) -> %s (scale=%s, pix_fmt=%s)",
        src, info.width, info.height, info.pix_fmt, dst, needs_scale, needs_pix_fmt,
    )
    try:
        _run(cmd)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    os.replace(tmp, dst)
    if os.path.abspath(src) != os.path.abspath(dst):
        os.remove(src)
    return dst


def make_llm_proxy(src: str, dst: str) -> str:
    """
    Write a small, upload-sized copy of *src* to *dst*; return *dst*.

    For handing a video to an LLM that will **read** it (transcription), never
    for anything a viewer will watch. The output is downscaled to
    ``LLM_PROXY_SHORT_SIDE`` (default 480) on its short side, capped at
    ``LLM_PROXY_FPS`` (default 15), encoded at CRF 30 and given a 96k mono
    audio track — a 59s 1080x1920 source drops from ~136 MB to single-digit
    MB, which is the difference between an upload the kie.ai host accepts and
    one it rejects with a bare HTTP 500.

    Both caps are one-way: a source already below them is left at its own
    resolution / frame rate rather than upscaled to meet them.

    Audio is **kept** (unless the source has none) — it is the half of the
    input the transcript is actually built from. Video is kept for speaker
    grounding: the transcript's ``speaker`` labels describe who is visible
    ("the woman in the red jacket"), which is the whole reason this pipeline
    uses a video model rather than ASR, so an audio-only proxy is not a valid
    substitute however much smaller it would be (docs/localisation.md §4.1).

    Raises MediaError if the re-encode fails; callers may fall back to
    uploading the original, which the upload size guard then judges on its
    own merits.
    """
    info = probe(src)

    filters: List[str] = []
    if _LLM_PROXY_SHORT_SIDE > 0 and min(info.width, info.height) > _LLM_PROXY_SHORT_SIDE:
        cap = _LLM_PROXY_SHORT_SIDE
        # Cap the SHORT side, so the filter flips with the orientation — same
        # convention as normalize_source().
        filters.append(f"scale=-2:{cap}" if info.width >= info.height else f"scale={cap}:-2")
    if _LLM_PROXY_FPS > 0 and info.fps > _LLM_PROXY_FPS:
        filters.append(f"fps={_LLM_PROXY_FPS:g}")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS, "-i", src]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", _LLM_PROXY_CRF,
        "-pix_fmt", "yuv420p",
    ]
    if info.has_audio:
        cmd += ["-c:a", "aac", "-ac", "1", "-b:a", f"{_LLM_PROXY_AUDIO_BPS // 1000}k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", dst]

    try:
        _run(cmd)
    except BaseException:
        if os.path.exists(dst):
            os.remove(dst)
        raise

    log.info(
        "make_llm_proxy: %s (%dx%d @%.3gfps, %d bytes) -> %s (%d bytes, filters=%s)",
        src, info.width, info.height, info.fps, os.path.getsize(src),
        dst, os.path.getsize(dst), ",".join(filters) or "none",
    )
    return dst


def extract_jpeg_frames(
    src: str, timestamps: List[float], *, width: int = 512
) -> List[Tuple[float, bytes]]:
    """
    Grab one JPEG still of *src* at each of *timestamps*; return them in order.

    For an LLM to **look at** (the localisation translator), never for
    delivery: scaled to *width* pixels wide at a modest quality, ~30-60 KB
    each. A timestamp that yields no frame is skipped rather than failing the
    batch — ffmpeg 7 exits non-zero for a seek past the end of the file
    instead of writing nothing.

    Raises MediaError only when timestamps were asked for and not one frame
    came out (a missing or unreadable *src*).
    """
    out: List[Tuple[float, bytes]] = []
    last_error = ""
    with tempfile.TemporaryDirectory(prefix="llm-frames-") as tmpdir:
        for i, ts in enumerate(timestamps):
            dst = os.path.join(tmpdir, f"f{i}.jpg")
            result = _run([
                FFMPEG, "-y", "-ss", f"{max(0.0, ts):.3f}", "-i", src,
                "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "5", dst,
            ], check=False)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                with open(dst, "rb") as fh:
                    out.append((ts, fh.read()))
            elif result.returncode != 0:
                last_error = "\n".join(result.stderr.splitlines()[-_STDERR_TAIL:])
    if timestamps and not out:
        raise MediaError(f"extract_jpeg_frames: no frame from {src}\n{last_error}")
    return out


def make_voice_reference(src: str, dst: str) -> float:
    """
    Write *src*'s audio to *dst* as a Seedance-acceptable voice reference.

    Returns the output duration in seconds.

    Seedance 2.5's ``reference_audio_urls`` accepts **only wav/mp3**, 2-30s per
    file and <=15 MB, so this normalises whatever the operator supplied — an
    m4a, an ogg, or the audio of a whole mp4 — into a mono 44.1 kHz 128k mp3
    truncated to :data:`VOICE_REF_MAX_SEC`. Normalising unconditionally (rather
    than passing an already-valid mp3 through) is what makes the format rule
    unenforceable-by-accident: every file that reaches kie.ai came out of this
    function, so there is no path where a container the API rejects slips
    through because it happened to be named ``.mp3``.

    A source **shorter** than :data:`VOICE_REF_MIN_SEC` raises MediaError: the
    API floor is 2s and nothing here can invent speech to pad it with. Padding
    with silence would satisfy the validator while degrading the very thing the
    reference exists to convey, so this fails loudly instead — the operator
    needs to supply a longer sample.

    Raises MediaError if the source has no audio track at all, or the encode
    fails.
    """
    # probe_audio, not probe: the common input here is an audio-ONLY file, which
    # probe() rejects for having no video stream.
    has_audio, src_sec = probe_audio(src)
    if not has_audio:
        raise MediaError(f"{src} has no audio track to use as a voice reference")
    if src_sec < VOICE_REF_MIN_SEC:
        raise MediaError(
            f"Voice reference must be at least {VOICE_REF_MIN_SEC:g}s long, "
            f"got {src_sec:.1f}s"
        )

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    cmd = [
        FFMPEG, "-y", "-threads", _FFMPEG_THREADS,
        "-i", src,
        # Truncate rather than reject: a 60s sample is a perfectly good voice,
        # the API just will not take more than 30s of it.
        "-t", f"{VOICE_REF_MAX_SEC:g}",
        "-vn",
        "-c:a", "libmp3lame",
        "-ar", "44100",
        "-ac", "1",
        "-b:a", "128k",
        dst,
    ]

    try:
        _run(cmd)
    except BaseException:
        if os.path.exists(dst):
            os.remove(dst)
        raise

    out_sec = min(src_sec, VOICE_REF_MAX_SEC)
    log.info(
        "make_voice_reference: %s (%.1fs, %d bytes) -> %s (%.1fs, %d bytes)",
        src, src_sec, os.path.getsize(src),
        dst, out_sec, os.path.getsize(dst),
    )
    return out_sec


def cut_clip(
    src: str,
    start_sec: float,
    end_sec: float,
    dst: str,
    *,
    include_audio: bool = True,
    fast_intermediate: bool = False,
) -> None:
    """
    Cut *src* from *start_sec* to *end_sec* into *dst*.

    Uses re-encoding (libx264/aac) to guarantee frame-accurate cuts at
    non-keyframe boundaries and avoids near-empty files at EOF.

    Note: ``-ss`` is placed *before* ``-i`` for fast seeking; ``-to`` is the
    duration relative to the seek point (= end_sec - start_sec).

    Parameters
    ----------
    include_audio:
        ``True`` (default) keeps the source audio track (re-encoded to AAC).
        ``False`` drops audio entirely (``-an``) — used for the reference clip
        sent to Gemini Omni, which fails when its input carries an audio track;
        the original audio is re-applied later during ``stitch``.
    fast_intermediate:
        ``True`` encodes with ``-preset ultrafast -crf 14`` instead of the
        default ``-preset fast -crf 18``.  Use this for "keep" clips that are
        only fed into ``stitch`` (which re-encodes anyway); the quality
        difference is invisible to viewers but encoding is ~4× faster.
        Swap clips uploaded to external AI services should always use the
        default quality (``fast_intermediate=False``).
    """
    duration = end_sec - start_sec
    if duration <= 0:
        raise MediaError(
            f"cut_clip: end_sec ({end_sec}) must be greater than start_sec ({start_sec})"
        )

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    preset = "ultrafast" if fast_intermediate else "fast"
    crf = "14" if fast_intermediate else "18"

    cmd = [
        FFMPEG,
        "-y",
        "-threads", _FFMPEG_THREADS,
        "-ss", str(start_sec),
        "-i", src,
        "-to", str(duration),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", crf,
    ]
    if include_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += [dst]
    log.info(
        "cut_clip: %s [%.3f, %.3f] -> %s (audio=%s, fast_intermediate=%s)",
        src, start_sec, end_sec, dst, include_audio, fast_intermediate,
    )
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
    frame_counts: Optional[List[Optional[int]]] = None,
) -> None:
    """
    Concatenate *clips* into *dst* — atomically (AICP-299 §3.1).

    ffmpeg writes a sibling ``<dst root>.<uuid8>.part.mp4`` (the ``.mp4`` suffix
    keeps ffmpeg's muxer inference), which then replaces *dst* with
    ``os.replace``. So *dst* is always either the previous complete file or the
    new complete file, never a truncated one, and the previous file's inode is
    left intact for any reader or hardlink still holding it. The temp file is
    removed on failure. Everything else is :func:`_stitch_into`.
    """
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    root, _ext = os.path.splitext(dst)
    tmp = f"{root}.{uuid.uuid4().hex[:8]}.part.mp4"
    try:
        extra = {"frame_counts": frame_counts} if frame_counts is not None else {}
        _stitch_into(clips, audio_source, tmp, width, height, fps, audio_mode, **extra)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                log.warning("stitch: could not remove temp file %s", tmp)


def _stitch_into(
    clips: List[str],
    audio_source: str,
    dst: str,
    width: int,
    height: int,
    fps: float,
    audio_mode: str = "original",
    frame_counts: Optional[List[Optional[int]]] = None,
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
        existing behaviour; Seedance clip audio is discarded.  If *audio_source*
        has no audio stream (or cannot be probed), a silent track is synthesised
        via ``anullsrc`` instead, so a silent source yields a silent delivery
        rather than failing the stitch.

        ``"seedance"`` — take audio from each clip itself: swap-segment clips use
        the Seedance result audio; keep-segment clips use the original cut audio.
        Each clip's audio is resampled to 44 100 Hz stereo so the concat filter
        has consistent stream parameters.  Clips that have no audio stream get a
        silent audio track synthesised via ``aevalsrc=0`` so the concat never
        fails on a missing stream.  *audio_source* is not used in this mode.
    frame_counts:
        ``"original"`` mode only (``None`` everywhere else): per clip, the
        number of OUTPUT frames (at *fps*) the clip may contribute at most, or
        ``None`` for "as it is". A clip longer than that is cut at its end
        after the frame-rate conversion (``fps``, then ``trim=end_frame``), so
        a generated clip that came back a few frames long (Gemini Omni rounds
        its length up to whole 4-frame groups at 24 fps) no longer pushes
        every later clip late against the continuous source soundtrack. A
        shorter clip is left as it is (never padded with invented picture).

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
    if frame_counts is not None:
        if audio_mode != "original":
            raise ValueError("stitch: frame_counts apply to audio_mode 'original' only")
        if len(frame_counts) != len(clips):
            raise ValueError("stitch: one frame count (or None) per clip")
        for c in frame_counts:
            if c is not None and (not isinstance(c, int) or c < 1):
                raise ValueError(f"stitch: bad frame count {c!r}")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    n = len(clips)

    # Probe each clip once, up front. Both audio modes need the total duration
    # to size the delivery bitrate cap; "seedance" additionally needs to know
    # which clips have audio and how long the synthesized silence should be for
    # the ones that don't. Probing is pure subprocess I/O (ffprobe), so a small
    # bounded thread pool runs the probes concurrently instead of serially —
    # for a 30-clip stitch that's the difference between 30 sequential ffprobe
    # round-trips and ~30/N of them. Results are written into pre-sized lists
    # by index so the outcome is identical to a serial loop regardless of
    # completion order, and each clip's MediaError -> (None, no-audio)
    # degradation is preserved per-clip.
    clip_infos: List[Optional[MediaInfo]] = [None] * n
    clip_has_audio: List[bool] = [False] * n

    def _probe_clip(idx: int) -> Tuple[int, Optional[MediaInfo]]:
        try:
            return idx, probe(clips[idx])
        except MediaError:
            return idx, None

    with ThreadPoolExecutor(max_workers=min(_STITCH_PROBE_CONCURRENCY, n)) as pool:
        for idx, info in pool.map(_probe_clip, range(n)):
            clip_infos[idx] = info
            clip_has_audio[idx] = info.has_audio if info is not None else False

    # Concatenated output length. A clip whose probe failed contributes nothing;
    # a total of 0 (every probe failed) leaves the cap off rather than deriving
    # a bogus bitrate from it.
    total_duration = sum(i.duration_sec for i in clip_infos if i is not None)
    video_opts = delivery_video_opts(
        total_duration, audio_bitrate_bps=_STITCH_AUDIO_BPS
    )

    # ------------------------------------------------------------------
    # audio_mode == "original" — full continuous soundtrack over the concat.
    # ------------------------------------------------------------------
    if audio_mode == "original":
        # ffmpeg's -map is strict: mapping an audio stream the input does not
        # have aborts the whole command. Some sources are exported video-only,
        # and that used to sink the run at the very last step — after every swap
        # segment had already been generated and paid for. Fall back to
        # synthesised silence, mirroring what the "seedance" branch does per
        # clip. An unprobeable source degrades the same way (as the per-clip
        # probes above already do) rather than raising here.
        try:
            source_has_audio = probe(audio_source).has_audio
        except MediaError:
            source_has_audio = False

        # Two passes, never one (runs 3b79ba7d / d46bb1b8, 2026-11-02).
        #
        # Pass 1 concatenates the clips VIDEO-ONLY. Pass 2 puts the soundtrack
        # on with the picture stream-copied. The single command this replaced
        # (clips + the source's audio as one more input, "-shortest") got
        # OOM-killed (rc=-9) at 6.2 GB on a 95 s 1080x1920 stitch of 11 clips:
        # ffmpeg 7.x paces each input by its timestamps, every AI clip's
        # timestamps restart at 0, and once the output passed the longest
        # clip (~9 s) the remaining clips' decoders were no longer held back —
        # they decoded the whole rest of the video into RAM ahead of the
        # concat (~3 MB per 1080p frame). Reproduced on the real clips: with
        # the extra audio input memory jumps by a whole clip at ~9 s and keeps
        # climbing; the same concat without it stays flat. So peak memory grew
        # with video length and any long enough 1080p stitch died, after every
        # segment had already been generated and paid for.
        video_only = os.path.splitext(dst)[0] + ".video.mp4"
        cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS]
        for clip in clips:
            cmd += ["-i", clip]

        filter_parts: List[str] = []
        for i in range(n):
            # Conform (frame_counts): the clip is first put on the output frame
            # grid (PTS from 0, fps), THEN cut at its frame budget, so the cut
            # is counted in delivered frames, whatever the clip's own rate.
            count = frame_counts[i] if frame_counts is not None else None
            conform = (f",trim=end_frame={count},setpts=PTS-STARTPTS"
                       if count is not None else "")
            filter_parts.append(
                f"[{i}:v]"
                + ("setpts=PTS-STARTPTS," if count is not None else "")
                + f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,"
                f"fps={fps}"
                f"{conform}"
                f"[v{i}]"
            )

        concat_inputs = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vout]")

        filter_complex = ";".join(filter_parts)

        # video_opts still reserve the audio bitrate (audio_bitrate_bps above):
        # pass 2 adds that audio, and the delivered file is what the cap binds.
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-an",
        ] + video_opts + [
            video_only,
        ]

        mux = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS, "-i", video_only]
        if source_has_audio:
            mux += ["-i", audio_source]
        else:
            # Infinite silent input; "-shortest" (below) cuts the output at the
            # end of the concatenated video, so no atrim is needed here.
            mux += [
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            ]
        mux += [
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            dst,
        ]

        log.info(
            "stitch[original]: %d clips -> %s  (%dx%d @ %.2ffps, %.1fs)  "
            "audio_source=%s%s  video_opts=%s",
            n, dst, width, height, fps, total_duration, audio_source,
            "" if source_has_audio else " (no audio — synthesising silence)",
            " ".join(video_opts),
        )
        try:
            _run(cmd)
            _run(mux)
        finally:
            if os.path.exists(video_only):
                try:
                    os.remove(video_only)
                except OSError:
                    log.warning("stitch: could not remove temp file %s", video_only)
        return

    # ------------------------------------------------------------------
    # audio_mode == "seedance" — use each clip's own audio.
    # Clips without an audio stream get synthesised silence.
    # ------------------------------------------------------------------
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
            # Silence: use anullsrc input, trimming to the already-probed clip
            # duration so the audio matches the video length.
            info = clip_infos[i]
            clip_dur = info.duration_sec if info is not None else 5.0

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
    ] + video_opts + [
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        dst,
    ]

    log.info(
        "stitch[seedance]: %d clips -> %s  (%dx%d @ %.2ffps, %.1fs)  video_opts=%s",
        n, dst, width, height, fps, total_duration, " ".join(video_opts),
    )
    _run(cmd)


def mux_original_audio(
    video: str,
    audio_source: str,
    start_sec: float,
    dst: str,
    *,
    source_has_audio: Optional[bool] = None,
    video_duration_sec: Optional[float] = None,
) -> None:
    """
    Replace *video*'s audio with the *audio_source* track starting at
    *start_sec*, producing *dst*. The video stream is copied untouched; the
    audio is padded with silence (``apad``) and cut to the video's probed
    duration so the clip stays exactly as long as its video.

    Used by custom stitch plans: their per-clip audio concat needs each swap
    result to carry the ORIGINAL soundtrack for its own time range (the
    classic full-track overlay only works when the stitched timeline equals
    the source timeline). If *audio_source* has no audio stream, *video* is
    copied to *dst* unchanged (stitch synthesises silence for audio-less
    clips).

    Pass *source_has_audio* when the caller already probed *audio_source*
    (the stitch phase calls this once per swap segment against the same
    source file — re-probing it every time is wasted subprocess work).

    Pass *video_duration_sec* when the caller already knows *video*'s exact
    duration (e.g. from an earlier probe of the same file) to skip the
    ``probe(video)`` call this function would otherwise run on every
    invocation. Left unset, the duration is probed here as before.
    """
    if source_has_audio is None:
        source_has_audio = probe(audio_source).has_audio

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    if not source_has_audio:
        shutil.copyfile(video, dst)
        log.info(
            "mux_original_audio: %s has no audio — copied %s unchanged",
            audio_source, video,
        )
        return

    video_dur = (
        video_duration_sec if video_duration_sec is not None
        else probe(video).duration_sec
    )

    cmd = [
        FFMPEG,
        "-y",
        "-threads", _FFMPEG_THREADS,
        "-i", video,
        "-ss", str(start_sec),
        "-i", audio_source,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "apad",
        "-t", f"{video_dur:.6f}",
        dst,
    ]
    log.info(
        "mux_original_audio: %s + %s[%.3fs..] -> %s (dur=%.3fs)",
        video, audio_source, start_sec, dst, video_dur,
    )
    _run(cmd)


def _audio_layout(path: str) -> tuple[str, int]:
    """Return ``(channel_layout_name, sample_rate)`` for *path*'s audio.

    Used to conform BOTH branches of :func:`splice_audio_window` to the base's
    own layout instead of a hardcoded one. That is not a stylistic choice: an
    ``aformat`` that upmixes mono to stereo applies ffmpeg's power-preserving
    rematrix (1/sqrt(2) per channel), which would quietly drop the WHOLE
    delivered soundtrack by 3 dB — outside the patched window as much as
    inside it. Conforming to the base makes its branch a no-op by construction.

    Anything above two channels is reported as stereo (no delivery this app
    produces has more, and amix needs a layout both branches can agree on).
    Falls back to stereo / :data:`_AUDIO_SPLICE_FALLBACK_SAMPLE_RATE` when
    ffprobe cannot answer, which is the shape every current delivery has.
    """
    try:
        out = _run([
            FFPROBE, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels,sample_rate",
            "-of", "json", path,
        ]).stdout
        stream = (json.loads(out).get("streams") or [{}])[0]
        channels = int(stream.get("channels") or 0)
        sample_rate = int(stream.get("sample_rate") or 0)
    except Exception as exc:  # ffprobe/JSON failure — degrade, never fail here
        log.warning("_audio_layout: could not probe %s (%s) — assuming stereo", path, exc)
        channels, sample_rate = 0, 0
    layout = "mono" if channels == 1 else "stereo"
    return layout, sample_rate or _AUDIO_SPLICE_FALLBACK_SAMPLE_RATE


def splice_audio_window(
    base: str,
    donor: str,
    dst: str,
    *,
    base_start: float,
    base_end: float,
    donor_start: float,
    crossfade_sec: float = AUDIO_SPLICE_DEFAULT_CROSSFADE_SEC,
) -> float:
    """
    Replace *base*'s audio over ``[base_start, base_end]`` with *donor*'s audio
    starting at *donor_start*, producing *dst*. Returns *dst*'s probed duration.

    The video stream is **copied** (``-c:v copy``) — not one frame is re-encoded
    — and no audio sample outside the window is moved, so the result is the base
    video with one word (or phrase) re-voiced and everything else, lip-sync
    included, byte-identical in timing. That "nothing moves" property is the
    whole point and it is why this is NOT built out of ``acrossfade``/``concat``:
    ``acrossfade`` consumes its fade length from both sides, which would pull
    every sample after the window earlier by twice the fade and desync the rest
    of the video against its own picture.

    Instead both tracks run the full base timeline and are summed under
    complementary gain envelopes::

        base gain  1 ────╲__________╱──── 1
        donor gain 0 ____╱‾‾‾‾‾‾‾‾‾‾╲____ 0
                    t0  t0+xf   t1-xf  t1

    The two envelopes sum to exactly 1 at every instant (the donor's is
    literally ``1 - base's``), so the crossfade cannot dip or clip, and outside
    ``[t0, t1]`` the donor contributes exactly nothing.

    Implementation notes, each of which is a bug someone would otherwise hit:

    * ``asetnsamples=n=128`` before the ``volume`` expression. ``eval=frame``
      re-evaluates the gain once per audio frame, and a decoder's native frame
      is ~1024 samples (~23 ms at 44.1 kHz) — a 60 ms ramp would be three
      audible steps. 128 samples is ~2.9 ms, which makes the ramp smooth.
    * ``amix=duration=first`` with the base first, plus ``apad`` on the donor
      branch: the output is exactly as long as the base's own audio, whatever
      the donor's length. ``normalize=0`` because the envelopes already sum to
      unity — amix's default would halve the level of the whole track.
    * Both branches are ``aformat``-ed to the BASE's own layout and sample rate
      (``_audio_layout``), not to a fixed one: ``amix`` refuses to mix inputs
      that disagree, but conforming to a hardcoded stereo would rematrix a MONO
      base and drop the whole soundtrack by 3 dB. Conforming to the base makes
      its branch a no-op and only ever converts the donor.
    * Neither ``-t`` nor ``-shortest`` is passed. The copied video stream must
      reach its own last frame; the audio branch is already bounded by
      ``duration=first``.

    Raises
    ------
    MediaError
        If either file has no audio, if the window is not a sane sub-range of
        the base, or if the donor is too short to supply ``base_end -
        base_start`` seconds from *donor_start*.
    """
    window = base_end - base_start
    if base_start < 0:
        raise MediaError(f"splice_audio_window: base_start ({base_start}) must be >= 0")
    if window < AUDIO_SPLICE_MIN_WINDOW_SEC:
        raise MediaError(
            f"splice_audio_window: window [{base_start}, {base_end}] is "
            f"{window:.3f}s — must be at least {AUDIO_SPLICE_MIN_WINDOW_SEC}s"
        )
    if donor_start < 0:
        raise MediaError(
            f"splice_audio_window: donor_start ({donor_start}) must be >= 0"
        )

    base_info = probe(base)
    if not base_info.has_audio:
        raise MediaError(f"splice_audio_window: {base} has no audio track")
    if base_end > base_info.duration_sec + _AUDIO_SPLICE_BOUND_EPS_SEC:
        raise MediaError(
            f"splice_audio_window: base_end ({base_end}) is past the end of "
            f"{base} ({base_info.duration_sec:.3f}s)"
        )

    donor_has_audio, donor_audio_sec = probe_audio(donor)
    if not donor_has_audio:
        raise MediaError(f"splice_audio_window: {donor} has no audio track")
    donor_end = donor_start + window
    if donor_end > donor_audio_sec + _AUDIO_SPLICE_BOUND_EPS_SEC:
        raise MediaError(
            f"splice_audio_window: the donor would have to supply audio up to "
            f"{donor_end:.3f}s but {donor} is only {donor_audio_sec:.3f}s long"
        )

    # A ramp may never eat more than half the window from each side, or the
    # two envelopes would cross and the replaced audio would never reach full
    # level. Clamped rather than rejected: a 0.06s default against a 0.08s
    # window is an operator picking a very tight window, not an error.
    xf = max(
        _AUDIO_SPLICE_MIN_CROSSFADE_SEC,
        min(float(crossfade_sec), window / 2 - _AUDIO_SPLICE_BOUND_EPS_SEC),
    )
    fade_out_end = base_start + xf
    fade_in_start = base_end - xf

    # Gain envelope for the BASE track; the donor's is `1 - this`, spelled out
    # below rather than nested so the ffmpeg expression stays readable in logs.
    base_gain = (
        f"if(lt(t,{base_start:.6f}),1,"
        f"if(lt(t,{fade_out_end:.6f}),1-(t-{base_start:.6f})/{xf:.6f},"
        f"if(lt(t,{fade_in_start:.6f}),0,"
        f"if(lt(t,{base_end:.6f}),(t-{fade_in_start:.6f})/{xf:.6f},1))))"
    )
    donor_gain = f"1-({base_gain})"

    layout, sample_rate = _audio_layout(base)
    fmt = (
        f"aformat=sample_fmts=fltp:sample_rates={sample_rate}"
        f":channel_layouts={layout}"
    )
    delay_ms = int(round(base_start * 1000))
    filter_complex = (
        f"[0:a]{fmt},asetnsamples=n={_AUDIO_SPLICE_GAIN_FRAME_SAMPLES}:p=0,"
        f"volume=volume='{base_gain}':eval=frame[a0];"
        f"[1:a]atrim=start={donor_start:.6f}:end={donor_end:.6f},"
        f"asetpts=PTS-STARTPTS,{fmt},"
        f"adelay=delays={delay_ms}:all=1,"
        f"asetnsamples=n={_AUDIO_SPLICE_GAIN_FRAME_SAMPLES}:p=0,"
        f"volume=volume='{donor_gain}':eval=frame,apad[a1];"
        f"[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]"
    )

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",
        "-threads", _FFMPEG_THREADS,
        "-i", base,
        "-i", donor,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", f"{_AUDIO_SPLICE_AUDIO_BITRATE_BPS // 1000}k",
        "-movflags", "+faststart",
        dst,
    ]
    log.info(
        "splice_audio_window: %s[%.3f..%.3f] <- %s[%.3f..%.3f] (xf=%.3fs) -> %s",
        base, base_start, base_end, donor, donor_start, donor_end, xf, dst,
    )
    _run(cmd)
    return probe(dst).duration_sec


def crop_clip(
    src: str,
    start_sec: float,
    end_sec: float,
    x: int,
    y: int,
    width: int,
    height: int,
    dst: str,
) -> Tuple[int, int]:
    """
    Trim *src* to ``[start_sec, end_sec]`` and spatially crop to the
    ``(x, y, width, height)`` rect, in a single ffmpeg pass.

    ``width``/``height`` are rounded *down* to the nearest even number before
    building the crop filter (libx264/yuv420p requires even dimensions); ``x``
    and ``y`` may remain odd. The rect (using the *requested* width/height) is
    validated against the source's probed dimensions before running ffmpeg.

    As with ``cut_clip``, ``-ss`` is placed before ``-i`` for fast seeking and
    ``-to`` is the duration relative to the seek point.

    Returns
    -------
    Tuple[int, int]
        The actual (possibly evened-down) ``(width, height)`` used for the
        crop, so callers can record the real output dimensions.

    Raises
    ------
    MediaError
        If ``end_sec <= start_sec``, or the requested rect does not fit
        inside the source's probed dimensions.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        raise MediaError(
            f"crop_clip: end_sec ({end_sec}) must be greater than start_sec ({start_sec})"
        )

    info = probe(src)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > info.width
        or y + height > info.height
    ):
        raise MediaError(
            f"crop_clip: rect (x={x}, y={y}, w={width}, h={height}) does not fit "
            f"inside source dimensions {info.width}x{info.height}"
        )

    # Round down to even (libx264/yuv420p requirement). x/y may stay odd.
    even_width = width - (width % 2)
    even_height = height - (height % 2)
    if even_width <= 0 or even_height <= 0:
        raise MediaError(
            f"crop_clip: width/height too small after evening down "
            f"({width}x{height} -> {even_width}x{even_height})"
        )

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",
        "-threads", _FFMPEG_THREADS,
        "-ss", str(start_sec),
        "-i", src,
        "-to", str(duration),
        "-vf", f"crop={even_width}:{even_height}:{x}:{y}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        dst,
    ]
    log.info(
        "crop_clip: %s [%.3f, %.3f] rect=(%d,%d,%d,%d) -> %s -> %s",
        src, start_sec, end_sec, x, y, even_width, even_height, dst,
    )
    _run(cmd)
    return (even_width, even_height)


def make_rounded_mask(width: int, height: int, radius_px: int, dst_png: str) -> None:
    """
    Render a grayscale rounded-rectangle alpha mask to *dst_png*.

    White (255) inside an axis-aligned ``width`` x ``height`` rounded
    rectangle of corner radius ``radius_px``, black (0) in the corner
    cutouts. ``radius_px`` is clamped to ``min(width, height) // 2``;
    ``radius_px <= 0`` produces an all-white mask (no alphamerge needed by
    the caller in that case, but the mask is still valid).

    Pure numpy/cv2 — no ffmpeg involved.
    """
    width = int(width)
    height = int(height)
    radius = max(0, min(int(radius_px), width // 2, height // 2))

    mask = np.zeros((height, width), dtype=np.uint8)
    if radius <= 0:
        mask[:] = 255
    else:
        white = 255
        # Two overlapping full-bleed rectangles cover everything except the
        # four corners, then a circle fills each corner back in.
        cv2.rectangle(mask, (radius, 0), (width - radius, height), white, -1)
        cv2.rectangle(mask, (0, radius), (width, height - radius), white, -1)
        cv2.circle(mask, (radius, radius), radius, white, -1)
        cv2.circle(mask, (width - radius, radius), radius, white, -1)
        cv2.circle(mask, (radius, height - radius), radius, white, -1)
        cv2.circle(mask, (width - radius, height - radius), radius, white, -1)

    os.makedirs(os.path.dirname(dst_png) or ".", exist_ok=True)
    ok = cv2.imwrite(dst_png, mask)
    if not ok:
        raise MediaError(f"make_rounded_mask: cv2.imwrite failed for {dst_png!r}")
    log.info(
        "make_rounded_mask: %dx%d radius=%d (clamped=%d) -> %s",
        width, height, radius_px, radius, dst_png,
    )


def overlay_insert(
    base: str,
    insert: str,
    dst: str,
    *,
    x: int,
    y: int,
    start_sec: float,
    end_sec: float,
    width: int,
    height: int,
    fps: float,
    mask_png: Optional[str] = None,
) -> None:
    """
    Composite *insert* onto *base* at ``(x, y)`` for ``[start_sec, end_sec]``,
    producing *dst*. Single ffmpeg pass; the base's original audio is kept
    untouched (stream-copied, never decoded).

    The insert stream is conformed before compositing: scaled to
    ``width`` x ``height``, ``setsar=1``, resampled to ``fps``, and trimmed so
    it cannot run longer than ``end_sec - start_sec``; if the insert is
    shorter than that window it is padded by freezing (cloning) its last
    frame via ``tpad``.

    Critically, ``overlay`` with ``enable='between(t,START,END)'`` matches
    insert frames to the base timeline by the insert's *own* timestamps — so
    after trim/pad the insert's PTS is reset and shifted
    (``setpts=PTS-STARTPTS+START/TB``) so its first frame lands exactly at
    ``start_sec`` on the base timeline. Without this shift, the overlay
    window would show whatever insert frame happens to share that raw
    timestamp (typically the wrong, or a frozen, frame).

    If ``mask_png`` is given, it is scaled to ``width`` x ``height`` and
    alpha-merged onto the (alpha-capable) conformed insert before the
    overlay, producing a rounded/masked composite; the final output is
    forced back to ``yuv420p`` for libx264.

    Raises
    ------
    MediaError
        If ``start_sec < 0``, ``end_sec <= start_sec``, or ``width``/``height``
        are not positive, or the ffmpeg pass fails.
    """
    if start_sec < 0:
        raise MediaError(f"overlay_insert: start_sec ({start_sec}) must be >= 0")
    if end_sec <= start_sec:
        raise MediaError(
            f"overlay_insert: end_sec ({end_sec}) must be greater than start_sec ({start_sec})"
        )
    if width <= 0 or height <= 0:
        raise MediaError(f"overlay_insert: width/height must be positive, got {width}x{height}")
    if fps <= 0:
        raise MediaError(f"overlay_insert: fps must be positive, got {fps}")

    window_duration = end_sec - start_sec
    insert_info = probe(insert)
    pad_dur = window_duration - insert_info.duration_sec

    # The composite runs the full length of the base, so its duration sizes the
    # delivery bitrate cap. Audio is stream-copied, charged at the pipeline's
    # 192k rather than re-measured.
    base_info = probe(base)

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    insert_chain = (
        f"[1:v]scale={width}:{height},setsar=1,fps={fps},"
        f"trim=duration={window_duration:.6f}"
    )
    if pad_dur > 0.001:
        insert_chain += f",tpad=stop_mode=clone:stop_duration={pad_dur:.6f}"
    insert_chain += f",setpts=PTS-STARTPTS+{start_sec}/TB"

    cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS, "-i", base, "-i", insert]

    if mask_png:
        cmd += ["-i", mask_png]
        insert_chain += ",format=rgba[insrgba]"
        filter_complex = (
            f"{insert_chain};"
            f"[2:v]scale={width}:{height}[maskv];"
            f"[insrgba][maskv]alphamerge[insa];"
            f"[0:v][insa]overlay=x={x}:y={y}:"
            f"enable='between(t,{start_sec},{end_sec})',format=yuv420p[vout]"
        )
    else:
        insert_chain += "[ins]"
        filter_complex = (
            f"{insert_chain};"
            f"[0:v][ins]overlay=x={x}:y={y}:"
            f"enable='between(t,{start_sec},{end_sec})',format=yuv420p[vout]"
        )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:a", "copy",
    ] + delivery_video_opts(
        base_info.duration_sec,
        audio_bitrate_bps=_STITCH_AUDIO_BPS if base_info.has_audio else 0,
    ) + [
        "-movflags", "+faststart",
        dst,
    ]
    log.info(
        "overlay_insert: %s + %s -> %s rect=(%d,%d,%d,%d) window=[%.3f,%.3f] fps=%s mask=%s",
        base, insert, dst, x, y, width, height, start_sec, end_sec, fps, bool(mask_png),
    )
    _run(cmd)


def burn_subtitles(
    src: str,
    ass_path: str,
    dst: str,
    *,
    duration_sec: Optional[float] = None,
    has_audio: Optional[bool] = None,
) -> None:
    """
    Burn the ASS subtitle file at *ass_path* onto *src*, writing *dst*, in a
    single ffmpeg pass.

    Audio is STREAM-COPIED (``-c:a copy``) when the source has audio, and
    dropped (``-an``) when it does not. ``subtitles`` is a video-only filter —
    it rasterizes text into frames and has no opinion on the audio track — so
    decoding and re-encoding audio here would spend CPU and shed quality on a
    stream this pass never touches. Pass *has_audio* when the caller already
    probed *src* (the caption pipeline probes once up front for
    width/height/duration too, to build the ASS file) to skip a redundant
    ffprobe; leave it unset to have this function probe it.

    Like every other function whose output is handed to Google Drive, the
    video flags come from :func:`delivery_video_opts` so the burn respects
    the 100 MB delivery cap. *duration_sec* sizes that cap's bitrate the same
    way and is probed here when not supplied. The copied audio track is
    charged at ``_STITCH_AUDIO_BPS`` against the same budget when present (0
    when the source is silent) — the same accounting ``stitch`` and
    ``overlay_insert`` use for a track whose bitrate this function does not
    control.

    The ``subtitles=`` filter argument is parsed by ffmpeg's filtergraph
    parser, where ``\\``, ``:`` and ``'`` are syntax rather than literal path
    characters (see :func:`_escape_filter_path`) — do not rely on our paths
    being uuid-clean; that assumption breaks the day someone renames a
    project directory to include one of them.

    Raises
    ------
    MediaError
        If *src* cannot be probed, or the ffmpeg pass fails.
    """
    if has_audio is None or duration_sec is None:
        info = probe(src)
        if has_audio is None:
            has_audio = info.has_audio
        if duration_sec is None:
            duration_sec = info.duration_sec

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    video_opts = delivery_video_opts(
        duration_sec,
        audio_bitrate_bps=_STITCH_AUDIO_BPS if has_audio else 0,
    )

    cmd = [
        FFMPEG,
        "-y",
        "-threads", _FFMPEG_THREADS,
        "-i", src,
        "-vf", f"subtitles={_escape_filter_path(ass_path)}",
    ] + video_opts
    if has_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", dst]

    log.info(
        "burn_subtitles: %s + %s -> %s (has_audio=%s, duration=%.3fs)",
        src, ass_path, dst, has_audio, duration_sec,
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
    return (info.width, info.height, stitch_fps(info.fps))


def stitch_fps(fps: float) -> float:
    """The frame rate a source at *fps* is stitched at (capped at
    ``FFMPEG_MAX_FPS``, see :func:`get_default_target`)."""
    if _MAX_FPS and fps and fps > _MAX_FPS:
        return _MAX_FPS
    return fps
