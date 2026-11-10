"""
Pure helpers of the reframe workflow (AICP-251, docs/reframe-design.md §5-§6):
window and ending validation, target dimensions, the cost estimate, the ffmpeg
argv for the prepare demux and the render, and the QA measurements.

Nothing here touches the database. ``pipeline_reframe`` runs these from RQ
jobs; ``api_reframe`` uses the validation and the defaults.

All render arithmetic is in **output frames at 24 fps** and **audio samples at
48 kHz**, measured from the job's ``input_start_sec`` (the instant output frame
0 shows), so the picture and the sound are cut on one grid (design §6):

    s  = input_start
    NP = round((P - s) x 24)          generated frames used, 1 <= NP <= raw_frames
    NF = round(F x 24)                fade frames (0 = cut to black)
    NA = round((A - s) x 48000)       snapshot audio samples (0 without audio)
    NL = max(NP + NF, ceil(NA / 2000)) + round(T x 24)   body frames
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from . import fal_models
from . import media as media_mod
from .config import settings

log = logging.getLogger(__name__)

ENDPOINT_ID = "minimax/h3-max/reframe-video"
OUTPUT_FPS = 24
AUDIO_RATE = 48000
SAMPLES_PER_FRAME = AUDIO_RATE // OUTPUT_FPS  # 2000
TARGET_ASPECTS = ("9:16", "16:9")
RESOLUTIONS = ("480p", "768p")

# Native output sizes of the model (probe T1-T3, design §2 fact 1).
OUTPUT_SIZES: dict[tuple[str, str], tuple[int, int]] = {
    ("9:16", "768p"): (768, 1344),
    ("16:9", "768p"): (1344, 768),
    ("9:16", "480p"): (480, 832),
    ("16:9", "480p"): (832, 480),
}

MIN_WINDOW_SEC = 1.0
MAX_FADE_SEC = 2.0
MAX_BLACK_TAIL_SEC = 5.0
DEFAULT_FADE_SEC = 0.65
DEFAULT_BLACK_TAIL_SEC = 0.25
CLICK_RAMP_SAMPLES = 480  # 10 ms: only to avoid a click, not an audible fade

# Prepare input encode (design §5).
_INPUT_MAX_BPS = 12_000_000
_INPUT_AUDIO_BPS = 128_000
_INPUT_FALLBACK_FACTOR = 0.6
_INPUT_FALLBACK_SHORT_SIDE = 768
_INPUT_MAX_FPS = 30.0

# QA thresholds (design §6).
J3_SSIM_MIN = 0.85
R2_LOUD_DBFS = -50.0
R2_TOLERANCE_DB = 1.5
R2_QUIET_MAX_DBFS = -45.0
R2_SILENT_PEAK_DBFS = -60.0
R2_WINDOW_SAMPLES = 2400  # 50 ms


class ReframeError(ValueError):
    """A request the workflow refuses, with an operator-readable message."""


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------


def target_aspect_for(width: Optional[int], height: Optional[int]) -> Optional[str]:
    """The aspect a source implies: landscape -> 9:16, portrait -> 16:9; None
    for a square (or unknown) source, where the operator picks."""
    if not width or not height or width == height:
        return None
    return "9:16" if width > height else "16:9"


def source_aspect(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if not width or not height or width == height:
        return None
    return "16:9" if width > height else "9:16"


def output_size(aspect: str, resolution: str) -> tuple[int, int]:
    return OUTPUT_SIZES[(aspect, resolution)]


def estimate_usd(resolution: str, output_sec: float) -> tuple[float, float]:
    """(base, reserved): ``output_sec x tariff`` and the layer's worst case."""
    spec = fal_models.endpoint(ENDPOINT_ID)
    base = round(float(output_sec) * spec.usd_per_sec(resolution), 4)
    return base, spec.worst_case_usd(resolution, output_sec)


def validate_generation(
    *, window_start: float, window_end: float, aspect: str, resolution: str,
    source_duration: Optional[float], source_width: Optional[int],
    source_height: Optional[int],
) -> None:
    """Raise :class:`ReframeError` for an invalid window, aspect or resolution."""
    if aspect not in TARGET_ASPECTS:
        raise ReframeError(f"target aspect must be one of {list(TARGET_ASPECTS)}")
    if resolution not in RESOLUTIONS:
        raise ReframeError(
            f"resolution must be one of {list(RESOLUTIONS)} (1080p has no Fal price)"
        )
    if source_aspect(source_width, source_height) == aspect:
        raise ReframeError(f"the source is already {aspect}")
    if not source_duration or source_duration <= 0:
        raise ReframeError("the source duration is unknown; re-create the project")
    for v in (window_start, window_end):
        if v is None or not math.isfinite(v):
            raise ReframeError("the window must be two finite numbers of seconds")
    if not 0 <= window_start < window_end <= source_duration + 1e-6:
        raise ReframeError(
            f"the window must satisfy 0 <= start < end <= {source_duration:.2f} s"
        )
    length = window_end - window_start
    if length < MIN_WINDOW_SEC:
        raise ReframeError(f"the window must be at least {MIN_WINDOW_SEC:g} s long")
    if length > settings.REFRAME_MAX_WINDOW_SEC + 1e-6:
        raise ReframeError(
            f"the window must be at most {settings.REFRAME_MAX_WINDOW_SEC:g} s long"
        )


# ---------------------------------------------------------------------------
# Ending: defaults, validation, frame math
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobTimes:
    """The job facts the ending math needs (from the job row and its QA)."""
    input_start: float
    window_end: float
    raw_frames: int
    has_audio: bool
    audio_end_max: float  # audio_start + audio_duration


@dataclass(frozen=True)
class Ending:
    picture_end: float
    fade: float
    audio_end: Optional[float]
    black_tail: float


@dataclass(frozen=True)
class Frames:
    np: int
    nf: int
    na: int
    nt: int
    nl: int


def default_ending(t: JobTimes) -> Ending:
    """The form's defaults, chosen to satisfy :func:`validate_ending` by
    construction (design §6)."""
    np_ = min(t.raw_frames, round((t.window_end - t.input_start) * OUTPUT_FPS))
    np_ = max(1, np_)
    p = t.input_start + np_ / OUTPUT_FPS
    a = min(max(p, t.window_end), t.audio_end_max) if t.has_audio else None
    return Ending(round(p, 4), DEFAULT_FADE_SEC, round(a, 4) if a is not None else None,
                  DEFAULT_BLACK_TAIL_SEC)


def frames_for(t: JobTimes, e: Ending) -> Frames:
    np_ = round((e.picture_end - t.input_start) * OUTPUT_FPS)
    nf = round(e.fade * OUTPUT_FPS)
    na = 0
    if t.has_audio and e.audio_end is not None:
        na = max(0, round((e.audio_end - t.input_start) * AUDIO_RATE))
    nt = round(e.black_tail * OUTPUT_FPS)
    nl = max(np_ + nf, math.ceil(na / SAMPLES_PER_FRAME)) + nt
    return Frames(np_, nf, na, nt, nl)


def validate_ending(t: JobTimes, e: Ending) -> Frames:
    """Raise :class:`ReframeError` unless the ending is valid; return its frames."""
    for name, v in (("picture_end_sec", e.picture_end), ("fade_sec", e.fade),
                    ("black_tail_sec", e.black_tail)):
        if v is None or not math.isfinite(v):
            raise ReframeError(f"{name} must be a finite number")
    f = frames_for(t, e)
    if not 1 <= f.np <= t.raw_frames:
        last = t.input_start + t.raw_frames / OUTPUT_FPS
        raise ReframeError(
            f"picture_end_sec must lie within the generated picture "
            f"({t.input_start + 1 / OUTPUT_FPS:.3f}-{last:.3f} s)"
        )
    if not 0 <= e.fade <= MAX_FADE_SEC:
        raise ReframeError(f"fade_sec must be within 0-{MAX_FADE_SEC:g} s")
    if not 0 <= e.black_tail <= MAX_BLACK_TAIL_SEC:
        raise ReframeError(f"black_tail_sec must be within 0-{MAX_BLACK_TAIL_SEC:g} s")
    if t.has_audio:
        if e.audio_end is None or not math.isfinite(e.audio_end):
            raise ReframeError("audio_end_sec is required for a source with audio")
        lo = e.picture_end - 1 / OUTPUT_FPS
        if not lo - 1e-6 <= e.audio_end <= t.audio_end_max + 1e-6:
            raise ReframeError(
                f"audio_end_sec must be within {lo:.3f}-{t.audio_end_max:.3f} s "
                "(not before the picture end, not past the pinned audio)"
            )
    if f.nl < 1:
        raise ReframeError("the ending would be empty")
    return f


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------


def _ffprobe_json(args: list[str]) -> dict:
    out = media_mod._run([media_mod.FFPROBE, "-v", "error", *args, "-of", "json"])
    try:
        return json.loads(out.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise media_mod.MediaError(f"ffprobe returned non-JSON output: {exc}") from exc


def count_video_frames(path: str) -> int:
    data = _ffprobe_json(["-count_packets", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_packets", path])
    streams = data.get("streams") or []
    if not streams:
        raise media_mod.MediaError(f"no video stream in {path!r}")
    return int(streams[0]["nb_read_packets"])


def first_frame_at_or_after(path: str, t: float) -> float:
    """Presentation time of the first video frame at or after *t*, read from
    the packets (VFR-safe), on the CONTAINER timeline: relative to the
    format's start_time, which is what an input ``-ss`` seeks on (a video
    stream that starts after the audio is offset from it)."""
    data = _ffprobe_json(["-show_entries", "format=start_time", path])
    start = float((data.get("format") or {}).get("start_time") or 0.0)
    lo = max(0.0, t - 2.0)
    pk = _ffprobe_json([
        "-select_streams", "v:0", "-read_intervals", f"{start + lo:.6f}%{start + t + 2.0:.6f}",
        "-show_entries", "packet=pts_time", path,
    ])
    best: Optional[float] = None
    for p in pk.get("packets") or []:
        try:
            rel = float(p["pts_time"]) - start
        except (KeyError, TypeError, ValueError):
            continue
        if rel >= t - 1e-6 and (best is None or rel < best):
            best = rel
    if best is None:
        raise media_mod.MediaError(f"no video frame at or after {t:.3f} s in {path!r}")
    return max(0.0, best)


# ---------------------------------------------------------------------------
# Prepare: one demux, two outputs
# ---------------------------------------------------------------------------


def input_bitrate(duration: float, cap_bytes: int) -> int:
    """VBV ceiling R for the input encode (design §5)."""
    budget = 0.85 * cap_bytes * 8 / max(duration, 0.25) - _INPUT_AUDIO_BPS
    return int(max(200_000, min(_INPUT_MAX_BPS, budget)))


def prepare_argv(
    *, src: str, input_start: float, video_sec: float, audio_sec: Optional[float],
    fps: float, video_dst: str, audio_dst: Optional[str], bitrate: int,
    short_side: Optional[int] = None, has_audio: bool,
) -> list[str]:
    """One ffmpeg call: accurate seek to *input_start*, then the video input
    (CFR, VBV-capped) and, when the source has audio, the lossless snapshot."""
    vf = [f"fps={min(fps or _INPUT_MAX_FPS, _INPUT_MAX_FPS):.6f}"]
    if short_side:
        vf.append(
            f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
        )
    vf.append("format=yuv420p")
    cmd = [media_mod.FFMPEG, "-y", "-threads", media_mod._FFMPEG_THREADS,
           "-ss", f"{input_start:.6f}", "-i", src,
           "-map", "0:v:0", "-t", f"{video_sec:.6f}", "-vf", ",".join(vf),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-maxrate", str(bitrate), "-bufsize", str(bitrate)]
    if has_audio:
        cmd += ["-map", "0:a:0", "-af", "aresample=async=1:first_pts=0",
                "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-movflags", "+faststart", video_dst]
    if has_audio and audio_dst and audio_sec:
        cmd += ["-map", "0:a:0", "-t", f"{audio_sec:.6f}",
                "-af", "aresample=async=1:first_pts=0", "-c:a", "flac", audio_dst]
    return cmd


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_body_argv(*, raw: str, audio: Optional[str], f: Frames, dst: str,
                     width: int, height: int) -> list[str]:
    """The body: generated picture to NP, held and faded over NF frames,
    black to NL; snapshot audio to NA samples, then silence (design §6)."""
    v = [f"[0:v]trim=end_frame={f.np},setpts=PTS-STARTPTS,fps={OUTPUT_FPS},"
         f"scale={width}:{height},setsar=1,format=yuv420p,"
         f"tpad=stop_mode=clone:stop={max(0, f.nl - f.np)}"]
    if f.nf >= 1:
        v.append(f"fade=t=out:start_frame={f.np}:nb_frames={f.nf}:color=black")
    else:
        v.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(n,{f.np})'")
    graph = [",".join(v) + "[v]"]
    total = f.nl * SAMPLES_PER_FRAME
    inputs = ["-i", raw]
    if audio and f.na > 0:
        inputs += ["-i", audio]
        ramp = min(CLICK_RAMP_SAMPLES, f.na)
        graph.append(
            f"[1:a]aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=end_sample={f.na},asetpts=PTS-STARTPTS,"
            f"afade=t=out:start_sample={f.na - ramp}:nb_samples={ramp},"
            f"apad=whole_len={total}[a]"
        )
    else:
        graph.append(
            f"anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=end_sample={total},"
            "asetpts=PTS-STARTPTS[a]"
        )
    dur = f.nl / OUTPUT_FPS
    return [media_mod.FFMPEG, "-y", "-threads", media_mod._FFMPEG_THREADS, *inputs,
            "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]",
            "-frames:v", str(f.nl), "-r", str(OUTPUT_FPS),
            *media_mod.delivery_video_opts(dur, audio_bitrate_bps=192_000),
            "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE),
            "-movflags", "+faststart", dst]


# ---------------------------------------------------------------------------
# QA measurements
# ---------------------------------------------------------------------------


def media_facts(path: str) -> dict:
    """Frames, size, durations and audio samples of a rendered file."""
    info = media_mod.probe(path)
    frames = count_video_frames(path)
    data = _ffprobe_json(["-select_streams", "a:0",
                          "-show_entries", "stream=duration,sample_rate", path])
    a = (data.get("streams") or [{}])
    a_dur = float(a[0].get("duration") or 0.0) if a and a[0] else 0.0
    return {"frames": frames, "width": info.width, "height": info.height,
            "duration": info.duration_sec, "audio_duration": a_dur,
            "has_audio": info.has_audio}


def parse_placement(meta: Optional[dict]) -> Optional[tuple[int, int, int, int]]:
    """(x, y, w, h) of the preserved source inside the output, from Fal's
    ``placement`` (accepts the shapes a JSON rectangle usually takes)."""
    p = (meta or {}).get("placement")
    try:
        if isinstance(p, dict):
            x = p.get("x", p.get("left"))
            y = p.get("y", p.get("top"))
            w = p.get("width", p.get("w"))
            h = p.get("height", p.get("h"))
            if w is None and "right" in p and x is not None:
                w = p["right"] - x
            if h is None and "bottom" in p and y is not None:
                h = p["bottom"] - y
            vals = (x, y, w, h)
        elif isinstance(p, (list, tuple)) and len(p) == 4:
            vals = tuple(p)
        else:
            return None
        x, y, w, h = (int(round(float(v))) for v in vals)
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return None
        return x, y, w, h
    except (TypeError, ValueError, KeyError):
        return None


_SSIM_RE = re.compile(r"All:([0-9.]+)")


def ssim_at(raw: str, inp: str, t: float, rect: tuple[int, int, int, int]) -> Optional[float]:
    x, y, w, h = rect
    w, h = w - w % 2, h - h % 2
    # One frame from each side, timestamps and timebases normalised: with
    # `-frames:v 1` and mismatched timebases the ssim filter's frame sync can
    # end the output before it has compared a pair (it then prints nothing).
    one = "trim=end_frame=1,setpts=PTS-STARTPTS,settb=AVTB"
    cmd = [media_mod.FFMPEG, "-v", "info", "-ss", f"{t:.3f}", "-i", raw, "-ss", f"{t:.3f}",
           "-i", inp, "-filter_complex",
           f"[0:v]{one},crop={w}:{h}:{x}:{y},format=yuv420p[a];"
           f"[1:v]{one},scale={w}:{h},format=yuv420p[b];[a][b]ssim", "-f", "null", "-"]
    res = media_mod._run(cmd, check=False)
    m = _SSIM_RE.search(res.stderr or "")
    return float(m.group(1)) if m else None


def contact_sheet(raw: str, frames: int, dst: str) -> None:
    step = max(1, frames // 9)
    media_mod._run([media_mod.FFMPEG, "-y", "-i", raw, "-vf",
                    f"select='not(mod(n\\,{step}))',scale=320:-2,tile=3x3",
                    "-frames:v", "1", "-q:v", "4", dst])


def _pcm(path: str, *, start_sample: int = 0, samples: Optional[int] = None):
    import numpy as np

    af = [f"aresample={AUDIO_RATE}", "aformat=channel_layouts=mono"]
    if samples is not None:
        af.append(f"atrim=start_sample={start_sample}:end_sample={start_sample + samples}")
    cmd = media_mod.harden_media_argv([
        media_mod.FFMPEG, "-v", "error", "-i", path, "-vn", "-af", ",".join(af),
        "-f", "f32le", "-acodec", "pcm_f32le", "-"])
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if out.returncode != 0:
        raise media_mod.MediaError(out.stderr.decode(errors="replace")[-2000:])
    return np.frombuffer(out.stdout, dtype="<f4")


def _envelope_db(x):
    import numpy as np

    n = len(x) // R2_WINDOW_SAMPLES
    if n == 0:
        return np.zeros(0)
    w = x[: n * R2_WINDOW_SAMPLES].astype("f8").reshape(n, R2_WINDOW_SAMPLES)
    rms = np.sqrt(np.mean(w * w, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-9))


def audio_check(body: str, snapshot: Optional[str], na: int, nl: int, *,
                final: bool = False) -> dict:
    """R2: duration, silence-aware envelope match over the first NA samples,
    and silence after them up to NL frames (design §6). With *final* the file
    continues past NL (the CTA's own audio): the duration is R3's business
    and only the body part ``[0, NL)`` is judged."""
    import numpy as np

    out: dict = {}
    total = nl * SAMPLES_PER_FRAME
    body_pcm = _pcm(body)
    out["samples"] = int(len(body_pcm))
    if final:
        ok = len(body_pcm) >= total - SAMPLES_PER_FRAME
        body_pcm = body_pcm[:total]
    else:
        ok = abs(len(body_pcm) - total) <= SAMPLES_PER_FRAME
    out["duration_ok"] = bool(ok)
    if snapshot and na > 0:
        ref = _pcm(snapshot, samples=na)
        b = body_pcm[:na]
        n = min(len(ref), len(b))
        e_ref, e_b = _envelope_db(ref[:n]), _envelope_db(b[:n])
        loud = e_ref > R2_LOUD_DBFS
        dev = np.abs(e_b - e_ref)
        out["max_dev_db"] = float(dev[loud].max()) if loud.any() else 0.0
        out["quiet_max_dbfs"] = float(e_b[~loud].max()) if (~loud).any() else None
        env_ok = (not loud.any() or out["max_dev_db"] <= R2_TOLERANCE_DB) and (
            out["quiet_max_dbfs"] is None or out["quiet_max_dbfs"] < R2_QUIET_MAX_DBFS)
        # Best alignment of the envelopes must be lag 0 +- 1 window.
        best_lag, best = 0, None
        for lag in range(-3, 4):
            a = e_ref[max(0, lag): len(e_ref) + min(0, lag)]
            c = e_b[max(0, -lag): len(e_b) + min(0, -lag)]
            if len(a) < 2:
                continue
            err = float(np.mean(np.abs(a - c)))
            if best is None or err < best - 1e-9:
                best, best_lag = err, lag
        out["lag_windows"] = best_lag
        ok = ok and env_ok and abs(best_lag) <= 1
        tail = body_pcm[na:]
    else:
        tail = body_pcm
    # Ignore the AAC tail of the click ramp (one frame after NA) and, on a
    # final, the stitch's boundary with the CTA (the last frame of the body).
    tail = tail[SAMPLES_PER_FRAME:] if snapshot and na > 0 else tail
    if final:
        tail = tail[:-SAMPLES_PER_FRAME] if len(tail) > SAMPLES_PER_FRAME else tail[:0]
    peak = float(np.max(np.abs(tail))) if len(tail) else 0.0
    out["tail_peak_dbfs"] = float(20 * math.log10(max(peak, 1e-9)))
    silent = out["tail_peak_dbfs"] < R2_SILENT_PEAK_DBFS
    out["tail_silent"] = bool(silent)
    out["ok"] = bool(ok and silent)
    return out
