"""Atomic bounded ending replacement/extension over trusted immutable inputs.

No synthesis or publication. Whole picture and AAC audio are re-encoded; the
prefix keeps its content and timing, not encoded packet/decoded sample identity.
Donor audio is the complete mix, used at 1x without padding or duration fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
import tempfile
import os
import time
import wave

from . import assistant_media as media
from .assistant_audio import probe_trusted_audio

RATE = 48000
MAX_INPUT_BYTES = 512 * 1024 * 1024
FRAME_TOLERANCE = Fraction(1, 10000)


@dataclass(frozen=True)
class EndingTiming:
    sample_rate: int
    start_frame: int
    end_frame: int
    base_frames: int
    freeze_frame: int | None
    fade_frames: int
    start_sample: int
    end_sample: int
    donor_start_sample: int
    tail_samples: int
    start_sec: float
    end_sec: float
    donor_start_sec: float
    donor_end_sec: float


@dataclass(frozen=True)
class EndingRenderResult(media.MediaEditResult):
    ending_timing: EndingTiming


def _fraction(value, name):
    return Fraction(str(media._number(value, name)))


def _frame(value, fps, name):
    position = value * fps
    index = round(position)
    if abs(position - index) > FRAME_TOLERANCE:
        raise media.AssistantMediaError(f"{name} must align to a video frame")
    return index


def _shape(visual_kind, freeze_timestamp_sec, fade_out_sec):
    if visual_kind not in ("black", "freeze_selected"):
        raise media.AssistantMediaError("unsupported ending visual kind")
    if visual_kind == "black" and (freeze_timestamp_sec is not None or fade_out_sec != 0):
        raise media.AssistantMediaError("black ending cannot freeze a frame or add a redundant fade")
    if visual_kind == "freeze_selected" and freeze_timestamp_sec is None:
        raise media.AssistantMediaError("freeze ending requires an earlier selected frame")


def resolve_ending_timing(base_info, *, start_sec, end_sec, donor_start_sec,
                          visual_kind, donor_audio_duration_sec,
                          freeze_timestamp_sec=None, fade_out_sec=0):
    """Pure shared admission/render timing policy, including post-rounding caps.

    Frame boundaries accept only the existing <=1e-4 frame representation error.
    Canonical frame times resolve to 48kHz samples once (nearest, <=0.5 sample).
    Donor coverage uses actual audio extent; decoding must prove exact samples.
    """
    start = _fraction(start_sec, "start_sec")
    end = _fraction(end_sec, "end_sec")
    donor = _fraction(donor_start_sec, "donor_start_sec")
    fade = _fraction(fade_out_sec, "fade_out_sec")
    audio_duration = _fraction(donor_audio_duration_sec, "donor_audio_duration_sec")
    freeze = None if freeze_timestamp_sec is None else _fraction(freeze_timestamp_sec, "freeze_timestamp_sec")
    _shape(visual_kind, freeze_timestamp_sec, fade)
    raw_duration = _fraction(base_info.video_duration_sec, "base duration")
    if not 0 < start < raw_duration < end <= min(raw_duration + 10, 120) or end-start > 10:
        raise media.AssistantMediaError("ending requires 0 < start < base end < output end, <=10s tail/extension, <=120s output")
    if not 0 < audio_duration <= 120 or donor < 0 or not 0 <= fade <= 2:
        raise media.AssistantMediaError("invalid ending donor extent, offset or fade")
    if (max(base_info.width, base_info.height) > 1920
            or base_info.width * base_info.height > 2073600):
        raise media.AssistantMediaError("ending source exceeds bounded picture geometry")
    fps = Fraction(base_info.fps)
    if not 0 < fps <= 120:
        raise media.AssistantMediaError("unsupported ending frame rate")
    base_frames = round(raw_duration * fps)
    first, last = _frame(start, fps, "start_sec"), _frame(end, fps, "end_sec")
    start, end, duration = Fraction(first, 1) / fps, Fraction(last, 1) / fps, Fraction(base_frames, 1) / fps
    if not 0 < first < base_frames < last or end > 120 or end-duration > 10 or end-start > 10:
        raise media.AssistantMediaError("canonical ending frame boundaries exceed duration caps")
    freeze_frame = None
    if freeze is not None:
        freeze_frame = _frame(freeze, fps, "freeze_timestamp_sec")
        if not 0 <= freeze < start or not 0 <= freeze_frame < first:
            raise media.AssistantMediaError("freeze frame must precede the replacement start")
    fade_frames = _frame(fade, fps, "fade_out_sec")
    if fade > 0 and (fade_frames < 2 or fade_frames > last-first):
        raise media.AssistantMediaError("ending fade requires >=2 frames and must fit the new tail")
    # Also enforce the cap after rounding; a tolerance must not authorize a
    # single extra frame on fractional rates.
    if Fraction(fade_frames, 1) / fps > 2:
        raise media.AssistantMediaError("canonical ending fade exceeds two seconds")
    a, b, d = round(start * RATE), round(end * RATE), round(donor * RATE)
    samples = b-a
    if samples <= 0 or d+samples > round(audio_duration * RATE):
        raise media.AssistantMediaError("donor actual audio does not cover the complete ending")
    return EndingTiming(RATE, first, last, base_frames, freeze_frame, fade_frames,
        a, b, d, samples, float(start), float(end), float(Fraction(d, RATE)), float(Fraction(d+samples, RATE)))


def _validated_audio_clip(donor, destination, timing, *, channels, layout, budget):
    """Decode exactly the requested donor samples. Never pad a short donor."""
    media._run([media.FFMPEG, "-v", "error", "-xerror", "-nostdin", "-n", "-filter_threads", "1",
        "-i", str(donor), "-vn", "-af",
        f"aresample={RATE},aformat=sample_fmts=s16:sample_rates={RATE}:channel_layouts={layout},"
        f"atrim=start_sample={timing.donor_start_sample}:end_sample={timing.donor_start_sample+timing.tail_samples},asetpts=PTS-STARTPTS",
        "-c:a", "pcm_s16le", "-ar", str(RATE), "-ac", str(channels), str(destination)],
        budget, inputs=(donor,), outputs=(destination,))
    if destination.stat().st_size > 2 * RATE * channels * 10 + 65536:
        raise media.AssistantMediaError("decoded ending donor exceeded its byte budget")
    with wave.open(str(destination), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()) != (
                channels, 2, RATE, timing.tail_samples):
            raise media.AssistantMediaError("decoded donor does not supply the exact required samples; padding is forbidden")


def extend_ending(src, donor, dst, *, start_sec, end_sec, donor_start_sec,
                  visual_kind, freeze_timestamp_sec=None, fade_out_sec=0,
                  timeout_sec=120, cancelled=None):
    """Keep [0,start), replace ending picture/audio, and extend to end_sec.

    Freeze repeats a selected earlier base frame; black adds no picture content.
    For editable captions callers supply verified clean picture and remap cues
    only through the returned prefix timeline. Unknown burned text is pixels.
    """
    # Reject nonfinite/ambiguous shape before any probe/decode/subprocess work.
    for name, value in (("start_sec", start_sec), ("end_sec", end_sec),
                        ("donor_start_sec", donor_start_sec), ("fade_out_sec", fade_out_sec)):
        media._number(value, name)
    if freeze_timestamp_sec is not None:
        media._number(freeze_timestamp_sec, "freeze_timestamp_sec")
    _shape(visual_kind, freeze_timestamp_sec, fade_out_sec)
    budget = media._budget(timeout_sec, cancelled)
    budget.check()
    source, replacement, destination = media._input(src), media._input(donor), media._destination(dst)
    if any(not 0 < path.stat().st_size <= MAX_INPUT_BYTES for path in (source, replacement)):
        raise media.AssistantMediaError("ending input exceeds its byte limit")
    originals = tuple(media.SourceProvenance(str(path), media._hash(path, budget)) for path in (source, replacement))
    info = media._probe(source, budget)
    donor_audio = probe_trusted_audio(replacement, timeout_sec=max(.000001, budget.deadline-time.monotonic()), cancelled=cancelled)
    base_audio = probe_trusted_audio(source, timeout_sec=max(.000001, budget.deadline-time.monotonic()), cancelled=cancelled) if info.has_audio else None
    timing = resolve_ending_timing(info, start_sec=start_sec, end_sec=end_sec,
        donor_start_sec=donor_start_sec, visual_kind=visual_kind, freeze_timestamp_sec=freeze_timestamp_sec,
        fade_out_sec=fade_out_sec, donor_audio_duration_sec=donor_audio.duration_sec)
    # Probes cannot change the pinned identities from which timing was resolved.
    for item in originals:
        if media._hash(Path(item.path), budget) != item.sha256:
            raise media.AssistantMediaError("ending source changed during inspection")
    channels = base_audio.channels if base_audio else donor_audio.channels
    layout = "mono" if channels == 1 else "stereo"
    fps = Fraction(info.fps)
    tail_frames = timing.end_frame-timing.start_frame
    with tempfile.TemporaryDirectory(prefix=".assistant-ending-", dir=destination.parent) as temporary:
        donor_clip = Path(temporary)/"donor.wav"
        _validated_audio_clip(replacement, donor_clip, timing, channels=channels, layout=layout, budget=budget)
        for item in originals:
            if media._hash(Path(item.path), budget) != item.sha256:
                raise media.AssistantMediaError("ending source changed while decoding donor")
        prefix = f"[0:v:0]trim=end_frame={timing.start_frame},setpts=PTS-STARTPTS[v0]"
        if visual_kind == "freeze_selected":
            tail = (f"[0:v:0]trim=start_frame={timing.freeze_frame}:end_frame={timing.freeze_frame+1},setpts=PTS-STARTPTS,"
                f"loop=loop={tail_frames-1}:size=1:start=0,trim=end_frame={tail_frames},setpts=N/({fps}*TB)")
        else:
            tail = (f"color=c=black:s={info.width}x{info.height}:r={fps}:d={float(Fraction(tail_frames,1)/fps):.12f},"
                f"trim=end_frame={tail_frames},setpts=N/({fps}*TB)")
        tail += f",setsar={info.sample_aspect_ratio.replace(':','/')}"
        if timing.fade_frames:
            tail += f",fade=t=out:s={tail_frames-timing.fade_frames}:n={timing.fade_frames-1}"
        tail += "[v1]"
        parts = [prefix, tail, "[v0][v1]concat=n=2:v=1:a=0[v]"]
        fmt = f"aresample={RATE},aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts={layout}"
        if base_audio:
            # Only the old normalized base's <=50ms AAC gap may be silent.
            # Never fill absent samples in the selected donor ending.
            parts.append(f"[0:a:0]{fmt},apad=whole_len={timing.start_sample},atrim=end_sample={timing.start_sample},asetpts=PTS-STARTPTS[a0]")
        else:
            parts.append(f"anullsrc=r={RATE}:cl={layout},atrim=end_sample={timing.start_sample},asetpts=PTS-STARTPTS[a0]")
        parts.append(f"[1:a:0]{fmt},atrim=end_sample={timing.tail_samples},asetpts=PTS-STARTPTS[a1]")
        # Resampling or prior AAC edits can leave tiny packet-clock gaps.
        # Anchor the final PCM clock to its actual sample count, not inherited
        # timestamps; otherwise a 25s prefix can add two samples to the AAC end.
        parts.append(f"[a0][a1]concat=n=2:v=0:a=1,atrim=end_sample={timing.end_sample},asetpts=N/SR/TB[a]")
        # The second FFmpeg input is our exact, bounded decoded donor clip.
        # Original donor provenance is independently rechecked before publication.
        result = media._render("extend_ending", (source, donor_clip), Path(temporary)/"validated.mp4",
            replace(info, has_audio=True), ";".join(parts), timing.end_sec,
            (media.TimelineSpan(0, timing.start_sec, 0),), "extend_ending_reencode",
            "Only prefix cues survive; frozen burned text remains picture content; new donor speech has no invented captions.",
            budget, exact_audio_timing=True)
        return _finish(result, originals, timing, destination, budget, channels=channels)


def validate_ending_output(path, timing, *, timeout_sec=30, cancelled=None):
    """Revalidate exact ending totals after an editable-caption re-burn."""
    budget = media._budget(timeout_sec, cancelled)
    output = media._input(path)
    return _validate_ending_output(output, timing, budget)


def _validate_ending_output(output, timing, budget, *, channels=None):
    info = media._probe(output,budget)
    raw = media._run([media.FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(output)], budget, inputs=(output,))
    if int(raw.strip()) != timing.end_frame:
        raise media.AssistantMediaError("ending output has an unexpected decoded frame count")
    audio = probe_trusted_audio(output, timeout_sec=max(.000001, budget.deadline-time.monotonic()), cancelled=budget.cancelled)
    if (audio.sample_rate != RATE or (channels is not None and audio.channels != channels)
            or abs(audio.duration_sec-timing.end_sample/RATE)>1/RATE+.000001
            or abs(info.duration_sec-timing.end_sec)>.002
            or info.duration_sec > 120 or info.video_duration_sec > 120):
        raise media.AssistantMediaError("ending audio/container duration does not match bounded output")
    if abs(info.video_duration_sec-timing.end_sec)>.0001:
        raise media.AssistantMediaError("ending picture duration does not match canonical frames")
    return info


def _finish(result, originals, timing, destination, budget, *, channels):
    """Validate exact rendered totals before exclusive destination publication."""
    output = Path(result.output_path)
    _validate_ending_output(output, timing, budget, channels=channels)
    for item in originals:
        if media._hash(Path(item.path), budget) != item.sha256:
            raise media.AssistantMediaError("ending source changed while rendering")
    if media._hash(output,budget) != result.output_sha256:
        raise media.AssistantMediaError("ending output changed during validation")
    budget.check()
    try:
        os.link(output, destination)
    except FileExistsError as exc:
        raise media.AssistantMediaError("ending output already exists") from exc
    return EndingRenderResult("extend_ending", originals, str(destination), result.output_sha256,
        result.metadata, timing.end_sec, result.base_timeline, result.audio_policy, result.subtitle_effect, timing)
