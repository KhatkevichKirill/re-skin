"""Trusted local media adapters for immutable assistant edits (AICP-268).

These functions are NOT planner tools: a backend asset resolver must supply the
local paths. They neither find assets nor update Runs, Drive or subtitle jobs.
Only normalized MP4 (one CFR H.264/yuv420p video and optional AAC audio) is
supported initially. Timeline-changing edits require video-frame boundaries;
unsupported inputs fail rather than silently normalizing timing or geometry.
Burned-in text is picture content; these adapters cannot preserve its meaning.
"""

from __future__ import annotations

import hashlib
from contextlib import suppress
import json
import math
import os
import signal
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable

from .media import FFMPEG, FFPROBE, harden_media_argv

CancelCheck = Callable[[], bool]


class AssistantMediaError(RuntimeError):
    """Invalid/unsupported media, cancelled processing, or failed rendering."""


class EditingPreparationRequired(AssistantMediaError):
    """Readable legacy media with a verified, bounded AAC picture overhang."""
    def __init__(self, info, preparation):
        super().__init__("Prepare an editable copy before editing or publishing this version")
        self.info, self.preparation = info, preparation


class AssistantMediaCancelled(AssistantMediaError):
    """The caller cancelled this attempt; no output was published."""


class AssistantMediaTimeout(AssistantMediaError):
    """A bounded media operation exceeded its deadline; no result may publish."""


@dataclass(frozen=True)
class MediaMetadata:
    duration_sec: float
    video_duration_sec: float
    width: int
    height: int
    fps: str
    sample_aspect_ratio: str
    has_audio: bool


@dataclass(frozen=True)
class SourceProvenance:
    path: str
    sha256: str


@dataclass(frozen=True)
class TimelineSpan:
    """Retained base timeline interval and its start on the output timeline."""

    source_start_sec: float
    source_end_sec: float
    output_start_sec: float


@dataclass(frozen=True)
class MediaEditResult:
    operation: str
    sources: tuple[SourceProvenance, ...]
    output_path: str
    output_sha256: str
    metadata: MediaMetadata
    expected_video_duration_sec: float
    base_timeline: tuple[TimelineSpan, ...]
    audio_policy: str
    subtitle_effect: str


@dataclass
class _Budget:
    deadline: float
    cancelled: CancelCheck | None

    def check(self) -> None:
        if self.cancelled and self.cancelled():
            raise AssistantMediaCancelled("media edit cancelled")
        if time.monotonic() >= self.deadline:
            raise AssistantMediaTimeout("media edit exceeded its time limit")


def _number(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssistantMediaError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise AssistantMediaError(f"{label} must be a finite number")
    return value


def _budget(timeout_sec: float, cancelled: CancelCheck | None) -> _Budget:
    timeout_sec = _number(timeout_sec, "timeout_sec")
    if not 0 < timeout_sec <= 3600:
        raise AssistantMediaError("timeout_sec must be within (0, 3600]")
    return _Budget(time.monotonic() + timeout_sec, cancelled)


def _run(argv: list[str], budget: _Budget, *, inputs=(), outputs=()) -> str:
    """Execute only through the mandatory per-command media sandbox.

    File capabilities are supplied explicitly by trusted callers, never inferred
    from command/filter text. A missing sandbox has no direct-execution fallback.
    """
    from .assistant_media_sandbox import MediaSandboxError, prepare_media_command
    budget.check()
    try:
        with prepare_media_command(harden_media_argv(argv), inputs=inputs, outputs=outputs) as command:
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                process = subprocess.Popen(
                    command.argv, env=command.env, cwd=command.cwd,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True, start_new_session=True,
                )
                try:
                    while process.poll() is None:
                        budget.check()
                        if stdout.tell() > 4_000_000 or stderr.tell() > 1_000_000:
                            raise AssistantMediaError("media subprocess output exceeded limit")
                        try:
                            process.wait(timeout=0.1)
                        except subprocess.TimeoutExpired:
                            pass
                    budget.check()
                    stdout.seek(0)
                    result = stdout.read(4_000_001)
                    if len(result) > 4_000_000 or stderr.tell() > 1_000_000:
                        raise AssistantMediaError("media subprocess output exceeded limit")
                    if process.returncode:
                        # Media metadata and diagnostics are untrusted; never
                        # relay decoder stderr or private paths to the planner.
                        raise AssistantMediaError("confined media operation failed")
                    result = result.decode("utf-8")
                    budget.check()
                finally:
                    if process.poll() is None:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
        return result
    except (MediaSandboxError, OSError) as exc:
        raise AssistantMediaError("media sandbox unavailable or operation refused") from exc


MAX_STREAM_CHUNK_BYTES = 64 * 1024 * 1024


def _stream(argv: list[str], budget: _Budget, *, inputs=(), chunk_bytes: int, max_chunks: int):
    """Yield fixed-size binary stdout records of one confined media command.

    The same launcher, input grants and argv hardening as :func:`_run`, but no
    outputs: decoded raw frames are read from a pipe one record at a time, so
    a single continuous decode (one filter state) needs no scratch file. A
    short trailing record is discarded. More than *max_chunks* records, a
    stalled pipe past the budget, or a non-zero exit fail closed; the process
    group is killed and reaped whenever the consumer stops early.
    """
    from .assistant_media_sandbox import MediaSandboxError, prepare_media_command
    import select
    if (type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_STREAM_CHUNK_BYTES
            or type(max_chunks) is not int or max_chunks < 1):
        raise AssistantMediaError("invalid media stream bounds")
    budget.check()
    try:
        with prepare_media_command(harden_media_argv(argv), inputs=inputs) as command, \
                tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command.argv, env=command.env, cwd=command.cwd,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                close_fds=True, start_new_session=True,
            )
            fd = process.stdout.fileno()
            try:
                emitted = 0
                eof = False
                while not eof:
                    record = bytearray()
                    while len(record) < chunk_bytes:
                        budget.check()
                        if stderr.tell() > 1_000_000:
                            raise AssistantMediaError("media subprocess output exceeded limit")
                        ready, _, _ = select.select([fd], [], [], 0.1)
                        if not ready:
                            continue
                        data = os.read(fd, min(1 << 20, chunk_bytes - len(record)))
                        if not data:
                            eof = True
                            break
                        record += data
                    if len(record) == chunk_bytes:
                        emitted += 1
                        if emitted > max_chunks:
                            raise AssistantMediaError("media stream exceeded its record limit")
                        yield bytes(record)
                while process.poll() is None:
                    budget.check()
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                budget.check()
                if process.returncode or stderr.tell() > 1_000_000:
                    # Decoder diagnostics are untrusted and never relayed.
                    raise AssistantMediaError("confined media operation failed")
            finally:
                process.stdout.close()
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    except (MediaSandboxError, OSError) as exc:
        raise AssistantMediaError("media sandbox unavailable or operation refused") from exc


def _input(path: str | Path) -> Path:
    # Requiring absolute regular files also excludes URLs, pipes and devices.
    path = Path(path)
    if not path.is_absolute():
        raise AssistantMediaError("trusted local input must be an absolute path")
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AssistantMediaError("input does not exist") from exc
    if not path.is_file():
        raise AssistantMediaError("input must be a regular local file")
    return path


def _destination(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.suffix.lower() != ".mp4":
        raise AssistantMediaError("output must be an absolute .mp4 path")
    if os.path.lexists(path):
        raise AssistantMediaError("output already exists; overwriting is forbidden")
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AssistantMediaError("output parent must exist") from exc
    if not parent.is_dir():
        raise AssistantMediaError("output parent must be a directory")
    return parent / path.name


def _hash(path: Path, budget: _Budget) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            budget.check()
            digest.update(chunk)
    budget.check()
    return digest.hexdigest()


def _probe(path: Path, budget: _Budget) -> MediaMetadata:
    raw = _run([
        FFPROBE, "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], budget, inputs=(path,))
    try:
        data = json.loads(raw)
        streams = data["streams"]
        videos = [s for s in streams if s["codec_type"] == "video"]
        audios = [s for s in streams if s["codec_type"] == "audio"]
        if len(videos) != 1 or len(audios) > 1 or len(streams) != 1 + len(audios):
            raise ValueError("requires one video, optional single audio, no other tracks")
        video = videos[0]
        fps = Fraction(video["avg_frame_rate"])
        sar = Fraction(video.get("sample_aspect_ratio", "1:1").replace(":", "/"))
        duration = float(video["duration"])
        container_duration = float(data["format"]["duration"])
        if ("mp4" not in data["format"]["format_name"].split(",")
                or video["codec_name"] != "h264" or video["pix_fmt"] != "yuv420p"
                or fps <= 0 or fps > 120 or fps != Fraction(video["r_frame_rate"])
                or sar <= 0 or duration <= 0 or not math.isfinite(duration)
                or not math.isfinite(container_duration)
                or not math.isfinite(float(video.get("start_time", "0")))
                or abs(float(video.get("start_time", "0"))) > 0.00001
                or int(video["width"]) % 2 or int(video["height"]) % 2
                or video.get("field_order", "progressive") not in ("progressive", "unknown")
                or float(video.get("tags", {}).get("rotate", 0)) % 360
                or any(float(s.get("rotation", 0)) % 360 for s in video.get("side_data_list", []))):
            raise ValueError("requires normalized zero-start CFR H.264/yuv420p MP4")
        audio_mismatch = False
        if audios:
            audio = audios[0]
            audio_duration = float(audio["duration"])
            if (audio["codec_name"] != "aac" or not math.isfinite(audio_duration)
                    or not math.isfinite(float(audio.get("start_time", "0")))
                    or abs(float(audio.get("start_time", "0"))) > 0.00001):
                raise ValueError("requires zero-start AAC audio spanning the picture")
            audio_mismatch = abs(audio_duration - duration) > 0.05
        # Verify actual frame timestamps: matching nominal frame rates alone
        # does not prove CFR, particularly for mixed/VFR phone recordings.
        packets = json.loads(_run([
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_packets", "-show_entries", "packet=pts,pts_time,duration",
            "-of", "json", str(path),
        ], budget, inputs=(path,)))["packets"]
        pts = sorted(float(p["pts_time"]) for p in packets)
        step = 1 / float(fps)
        if not pts or any(not math.isfinite(point) for point in pts):
            raise ValueError("video packet timestamps are not finite")
        grid_gap = False
        strict_grid = (pts and all(math.isfinite(point) for point in pts) and abs(pts[0]) <= 0.00001
            and abs(pts[-1] + step - duration) <= 0.0001
            and all(abs(b - a - step) <= 0.00001 for a, b in zip(pts, pts[1:])))
        if not strict_grid:
            # One preparation-only MOV shape: exactly one missing slot before
            # a final source picture whose PTS equals the declared stream end.
            # Ordinary CFR inputs/mocks do not need these extra native fields.
            clock = Fraction(video["time_base"])
            source_count = len(packets)
            declared = Fraction(video["duration_ts"]) * clock
            slots = sorted(Fraction(packet["pts"]) * clock * fps for packet in packets)
            if (source_count < 2 or int(video["nb_frames"]) != source_count or clock <= 0
                    or slots != list(range(source_count - 1)) + [source_count]
                    or declared != Fraction(source_count, 1) / fps
                    or abs(float(declared) - duration) > .00001
                    or any(Fraction(packet["duration"]) * clock != 1/fps for packet in packets)):
                raise ValueError("video packet timestamps are not a zero-start CFR timeline")
            grid_gap = True
        # The proved CFR packet grid is the authoritative picture clock.
        # ffprobe decimal duration (29.833333) is not exactly895/30, and can
        # otherwise reject a valid44.1kHz sample boundary by0.0147 samples.
        picture = Fraction(len(pts), 1) / fps
        info = MediaMetadata(container_duration, float(picture), int(video["width"]), int(video["height"]),
            str(fps), f"{sar.numerator}:{sar.denominator}", bool(audios))
        if audio_mismatch or grid_gap:
            if not audios:
                raise ValueError("grid-gap preparation requires a bounded positive AAC tail")
            audio_end = Fraction(str(audio["duration_ts"])) * Fraction(audio["time_base"])
            overhang = audio_end - picture
            if abs(float(audio_end) - audio_duration) > .00001:
                raise ValueError("inconsistent audio stream timing")
            if not 0 < overhang <= Fraction(1, 10):
                raise ValueError("requires zero-start AAC audio spanning the picture")
            output_frames = max(len(pts) + int(grid_gap), math.ceil(audio_end * fps))
            frames = output_frames - len(pts)
            prepared = Fraction(output_frames, 1) / fps
            # An explicit preparation must still fit the unchanged strict
            # output profile. Low-FPS holds can exceed its50ms audio margin.
            if prepared - audio_end > Fraction(1, 20) or float(prepared) > 120:
                raise ValueError("audio-tail preparation cannot fit the supported media profile")
            raise EditingPreparationRequired(info, {
                "kind": "audio_tail_grid_gap" if grid_gap else "audio_tail", "added_frames": frames, "added_duration_sec": float(Fraction(frames, 1) / fps),
                "target_duration_sec": float(prepared), "audio_copied": True,
                "source_video_frames": len(pts), "audio_duration_sec": float(audio_end),
                "audio_duration_ts": int(audio["duration_ts"]), "audio_time_base": audio["time_base"],
                "audio_overhang_sec": float(overhang),
                **({"gap_fill_frames": 1, "tail_hold_frames": output_frames - (len(pts) + 1),
                    "gap_start_sec": float(Fraction(len(pts)-1,1)/fps),
                    "source_declared_duration_sec": float(picture), "source_final_pts_sec": float(picture),
                    "source_final_grid_index": len(pts), "output_frame_count": output_frames,
                    "video_time_base": video["time_base"], "video_duration_ts": int(video["duration_ts"])} if grid_gap else {}),
            })
        return info
    except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise AssistantMediaError(f"unsupported media: {exc}") from exc


def probe_trusted_media(
    path: str | Path, *, timeout_sec: float = 30,
    cancelled: CancelCheck | None = None,
) -> MediaMetadata:
    """Inspect a backend-resolved local input under the renderer's constraints.

    This establishes media capabilities, not authorization. Callers must resolve
    and verify the immutable artifact before passing its private local path.
    No output is written and no provider or model is contacted.
    """
    budget = _budget(timeout_sec, cancelled)
    budget.check()
    return _probe(_input(path), budget)


def _span(start_sec: float, end_sec: float, info: MediaMetadata, *, frames: bool) -> tuple[float, float]:
    start = _number(start_sec, "start_sec")
    end = _number(end_sec, "end_sec")
    if not 0 <= start < end <= info.video_duration_sec + 0.000001:
        raise AssistantMediaError("range must satisfy 0 <= start < end <= video duration")
    end = min(end, info.video_duration_sec)
    if frames:
        fps = float(Fraction(info.fps))
        if any(abs(t * fps - round(t * fps)) > 0.0001 for t in (start, end)):
            raise AssistantMediaError("timeline edit boundaries must align to video frames")
        start, end = (round(t * fps) / fps for t in (start, end))
    return start, end


def _video_options(fps: str) -> list[str]:
    # No stitch/get_default_target: those intentionally cap FPS/resolution.
    # trim/setpts/concat clear frame durations on FFmpeg 7. Passthrough then
    # muxes a zero-duration last frame, shortening video by one frame while
    # audio remains intact. Explicit CFR at the already-validated SOURCE rate
    # restores frame durations, including fractional FPS, without a rate cap.
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2", "-vsync", "1", "-r:v", fps]


def _render(
    operation: str, sources: tuple[Path, ...], dst: Path, info: MediaMetadata,
    filters: str, expected_duration: float, timeline: tuple[TimelineSpan, ...],
    audio_policy: str, subtitle_effect: str, budget: _Budget, *, auxiliary_inputs=(), exact_audio_timing=False,
) -> MediaEditResult:
    if type(exact_audio_timing) is not bool:
        raise AssistantMediaError("exact_audio_timing must be a boolean")
    provenance = tuple(SourceProvenance(str(p), _hash(p, budget)) for p in sources)
    # A staging directory on the destination filesystem allows an atomic,
    # exclusive hard-link publication. Unlike rename/replace it cannot clobber
    # an output created by a concurrent worker after the initial check.
    with tempfile.TemporaryDirectory(prefix=".assistant-media-", dir=dst.parent) as temporary:
        stage = Path(temporary) / "candidate.mp4"
        argv = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-filter_complex_threads", "1"]
        for source in sources:
            argv += ["-noautorotate", "-i", str(source)]
        argv += ["-filter_complex", filters, "-map", "[v]"]
        if info.has_audio:
            argv += ["-map", "0:a:0" if audio_policy == "stream_copy_base" else "[a]"]
            argv += ["-c:a", "copy"] if audio_policy == "stream_copy_base" else ["-c:a", "aac", "-b:a", "192k"]
        else:
            argv += ["-an"]
        argv += [*_video_options(info.fps)]
        if exact_audio_timing:
            # Ending extension resolves one 48kHz sample clock. MP4's default
            # millisecond edit-list clock would round fractional-FPS audio ends.
            argv += ["-movie_timescale", "48000"]
        argv += ["-map_metadata", "-1", "-movflags", "+faststart", str(stage)]
        _run(argv, budget, inputs=(*sources, *auxiliary_inputs), outputs=(stage,))
        output_info = _probe(stage, budget)
        if (output_info.width, output_info.height, output_info.fps, output_info.sample_aspect_ratio,
                output_info.has_audio) != (info.width, info.height, info.fps, info.sample_aspect_ratio, info.has_audio):
            raise AssistantMediaError("render changed media geometry, frame rate, SAR or audio presence")
        if abs(output_info.video_duration_sec - expected_duration) > 0.0001:
            raise AssistantMediaError("render duration does not match the edit plan")
        for source in provenance:
            if _hash(Path(source.path), budget) != source.sha256:
                raise AssistantMediaError("source changed while rendering; refusing publication")
        output_hash = _hash(stage, budget)
        budget.check()
        try:
            os.link(stage, dst)
        except FileExistsError as exc:
            raise AssistantMediaError("output already exists; overwriting is forbidden") from exc
    return MediaEditResult(
        operation, provenance, str(dst), output_hash, output_info, expected_duration,
        timeline, audio_policy, subtitle_effect,
    )


def image_only_fade(
    src: str | Path, dst: str | Path, *, start_sec: float, end_sec: float,
    timeout_sec: float = 120, cancelled: CancelCheck | None = None,
) -> MediaEditResult:
    """Fade the ending picture to black through video end; copy base audio.

    Burned-in subtitles fade with the picture. Preserving separately rendered
    captions requires a clean-picture source and a later subtitle render.
    """
    budget = _budget(timeout_sec, cancelled)
    source, destination = _input(src), _destination(dst)
    info = _probe(source, budget)
    start, end = _span(start_sec, end_sec, info, frames=False)
    if abs(end - info.video_duration_sec) > 0.000001:
        raise AssistantMediaError("ending fade must end at the video duration")
    return _render(
        "image_only_fade", (source,), destination, info,
        f"[0:v:0]fade=t=out:st={start:.9f}:d={end-start:.9f}[v]",
        info.video_duration_sec, (TimelineSpan(0, info.video_duration_sec, 0),),
        "stream_copy_base", "Burned-in text fades with picture; no separate subtitle edits.", budget,
    )


def image_fades(
    src: str | Path, dst: str | Path, *, fades: tuple[tuple[float, float, str], ...],
    timeout_sec: float = 120, cancelled: CancelCheck | None = None,
) -> MediaEditResult:
    """Apply bounded picture fades in one encode; copy the base soundtrack.

    Each ramp reaches its endpoint on the last INCLUDED frame. Outside each
    half-open interval the original picture passes through unchanged.
    """
    budget = _budget(timeout_sec, cancelled)
    source, destination = _input(src), _destination(dst)
    info = _probe(source, budget)
    fps = float(Fraction(info.fps))
    if not fades:
        raise AssistantMediaError("at least one image fade is required")
    parts, spans = [], []
    for start_sec, end_sec, direction in fades:
        start, end = _span(start_sec, end_sec, info, frames=True)
        count = round((end-start)*fps)
        if direction not in ("out", "in") or count < 2:
            raise AssistantMediaError("image fade requires a direction and at least two video frames")
        if any(max(start, a) < min(end, b) for a, b in spans):
            raise AssistantMediaError("overlapping image fades are ambiguous")
        spans.append((start,end))
        # Time based fade durations round differently for fractional rates.
        # Explicit frame indexes give exactly N-1 steps on CFR input.
        first = round(start*fps)
        parts.append(f"fade=t={direction}:s={first}:n={count-1}:"
                     f"enable='gte(n,{first})*lt(n,{first+count})'")
    return _render(
        "image_fades", (source,), destination, info, "[0:v:0]" + ",".join(parts) + "[v]",
        info.video_duration_sec, (TimelineSpan(0, info.video_duration_sec, 0),),
        "stream_copy_base", "Burned-in text fades with picture; separate subtitle cues retain timing.", budget,
    )


def image_dip_to_black(src, dst, *, start_sec, midpoint_sec, end_sec, timeout_sec=120, cancelled=None):
    return image_fades(src, dst, fades=((start_sec, midpoint_sec, "out"), (midpoint_sec, end_sec, "in")),
        timeout_sec=timeout_sec, cancelled=cancelled)


def trim_video(
    src: str | Path, dst: str | Path, *, start_sec: float = 0, end_sec: float,
    timeout_sec: float = 120, cancelled: CancelCheck | None = None,
) -> MediaEditResult:
    """Keep frame-aligned [start_sec, end_sec), trimming picture and audio."""
    budget = _budget(timeout_sec, cancelled)
    source, destination = _input(src), _destination(dst)
    info = _probe(source, budget)
    start, end = _span(start_sec, end_sec, info, frames=True)
    filters = f"[0:v:0]trim=start={start:.9f}:end={end:.9f},setpts=PTS-STARTPTS[v]"
    if info.has_audio:
        filters += f";[0:a:0]atrim=start={start:.9f}:end={end:.9f},asetpts=PTS-STARTPTS[a]"
    return _render(
        "trim", (source,), destination, info, filters, end-start,
        (TimelineSpan(start, end, 0),), "trim_and_reencode",
        "Burned-in pixels retained only in kept range; separate cues require clipping and time remapping.", budget,
    )


def cut_section(
    src: str | Path, dst: str | Path, *, start_sec: float, end_sec: float,
    timeout_sec: float = 120, cancelled: CancelCheck | None = None,
) -> MediaEditResult:
    """Remove an internal frame-aligned interval from picture and audio.

    Later media moves earlier by end_sec-start_sec. Speech crossing either
    boundary is cut; choosing a semantically safe interval is the caller's job.
    """
    budget = _budget(timeout_sec, cancelled)
    source, destination = _input(src), _destination(dst)
    info = _probe(source, budget)
    start, end = _span(start_sec, end_sec, info, frames=True)
    if start == 0 or end >= info.video_duration_sec:
        raise AssistantMediaError("cut must be internal; use trim for an ending or beginning")
    parts = []
    for index, (a, b) in enumerate(((0, start), (end, info.video_duration_sec))):
        parts.append(f"[0:v:0]trim=start={a:.9f}:end={b:.9f},setpts=PTS-STARTPTS[v{index}]")
        if info.has_audio:
            parts.append(f"[0:a:0]atrim=start={a:.9f}:end={b:.9f},asetpts=PTS-STARTPTS[a{index}]")
    if info.has_audio:
        parts.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
    else:
        parts.append("[v0][v1]concat=n=2:v=1:a=0[v]")
    return _render(
        "cut", (source,), destination, info, ";".join(parts),
        info.video_duration_sec-(end-start),
        (TimelineSpan(0, start, 0), TimelineSpan(end, info.video_duration_sec, start)),
        "cut_and_reencode", "Burned-in pixels are cut with picture; separate cues require splitting/clipping and time remapping.", budget,
    )


def replace_footage(
    src: str | Path, donor: str | Path, dst: str | Path, *,
    start_sec: float, end_sec: float, donor_start_sec: float = 0,
    timeout_sec: float = 120, cancelled: CancelCheck | None = None,
) -> MediaEditResult:
    """Replace only picture with equal-length available footage; copy base audio.

    Donor geometry, CFR and SAR must match. No scaling, stretching, generation,
    lip-sync claim or subtitle reconstruction is performed.
    """
    budget = _budget(timeout_sec, cancelled)
    source, replacement, destination = _input(src), _input(donor), _destination(dst)
    info, donor_info = _probe(source, budget), _probe(replacement, budget)
    start, end = _span(start_sec, end_sec, info, frames=True)
    donor_start = _number(donor_start_sec, "donor_start_sec")
    donor_start, donor_end = _span(donor_start, donor_start+end-start, donor_info, frames=True)
    if (info.width, info.height, info.fps, info.sample_aspect_ratio) != (
            donor_info.width, donor_info.height, donor_info.fps, donor_info.sample_aspect_ratio):
        raise AssistantMediaError("replacement geometry, FPS and SAR must match the base")
    spans = []
    if start:
        spans.append((0, 0, start))
    spans.append((1, donor_start, donor_end))
    if end < info.video_duration_sec:
        spans.append((0, end, info.video_duration_sec))
    parts = [f"[{source_index}:v:0]trim=start={a:.9f}:end={b:.9f},setpts=PTS-STARTPTS[v{i}]"
             for i, (source_index, a, b) in enumerate(spans)]
    parts.append("".join(f"[v{i}]" for i in range(len(spans))) + f"concat=n={len(spans)}:v=1:a=0[v]")
    return _render(
        "replace_footage", (source, replacement), destination, info, ";".join(parts),
        info.video_duration_sec, (TimelineSpan(0, info.video_duration_sec, 0),),
        "stream_copy_base", "Replacement substitutes burned-in picture/text; separately editable cue timing is unchanged, content still requires review.", budget,
    )
