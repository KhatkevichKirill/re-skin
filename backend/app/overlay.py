"""
overlay.py — Burn static PNG overlays onto a finished video.

Standalone feature: takes an already-generated video and burns one or more
transparent PNGs onto it at operator-chosen positions and time windows. One
ffmpeg re-encode of the video (size-capped CRF, see
``media.delivery_video_opts``); audio is stream-copied. Looped image inputs are
infinite, so ``-shortest`` bounds the output to the base video.

Reuses the shared ffmpeg helpers from :mod:`media` (``probe``, ``MediaError``,
``_run``, ``FFMPEG``, ``_FFMPEG_THREADS``). Does NOT touch ``media.stitch`` or
the run pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from .media import (
    FFMPEG,
    MediaError,
    _FFMPEG_THREADS,
    _run,
    _STITCH_AUDIO_BPS,
    delivery_video_opts,
    probe,
)


@dataclass
class OverlaySpec:
    asset_path: str            # absolute path resolved by the caller
    nx: float                  # top-left X, normalized 0..1 of frame width
    ny: float                  # top-left Y, normalized 0..1 of frame height
    start_sec: float
    end_sec: float
    sw: float | None = None    # target width as fraction of frame width; None = native px
    fade_sec: float = 0.0      # 0 / None = hard cut


def burn_overlays(src: str, dst: str, specs: List[OverlaySpec]) -> None:
    """Burn static PNG overlays onto *src* -> *dst*.

    Video is re-encoded at the delivery quality/size cap; audio is
    stream-copied (so its bitrate is charged to the budget at the pipeline's
    192k rather than re-measured). Positions are
    computed in integer pixels from the probed source dimensions, so placement is
    exact and resolution-independent. Each image is clamped to stay within the
    frame via ffmpeg ``min(x, W-w)`` / ``min(y, H-h)`` expressions.
    """
    if not specs:
        raise MediaError("burn_overlays: specs list is empty")

    info = probe(src)
    W, H, dur = info.width, info.height, info.duration_sec
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS, "-i", src]
    for s in specs:
        cmd += ["-loop", "1", "-i", s.asset_path]

    parts: List[str] = []
    for i, s in enumerate(specs):
        in_idx = i + 1                      # 0 = base video
        chain = f"[{in_idx}:v]format=rgba"
        if s.sw:                            # scale to a fraction of frame width
            target_w = max(1, round(W * s.sw))
            chain += f",scale={target_w}:-1"
        if s.fade_sec and s.fade_sec > 0:
            end = min(s.end_sec, dur)
            fade_out_st = max(s.start_sec, end - s.fade_sec)
            chain += (
                f",fade=in:st={s.start_sec}:d={s.fade_sec}:alpha=1"
                f",fade=out:st={fade_out_st}:d={s.fade_sec}:alpha=1"
            )
        parts.append(f"{chain}[ov{i}]")

    # Position in integer px, clamped so the image never spills off-frame.
    prev = "[0:v]"
    for i, s in enumerate(specs):
        x = max(0, round(W * s.nx))
        y = max(0, round(H * s.ny))
        end = min(s.end_sec, dur)
        out = "[vout]" if i == len(specs) - 1 else f"[b{i}]"
        # overlay clamps x/y against (W-w)/(H-h) at runtime via min(); using
        # ffmpeg expressions keeps it correct even after scaling.
        parts.append(
            f"{prev}[ov{i}]overlay="
            f"x='min({x},W-w)':y='min({y},H-h)':"
            f"enable='between(t,{s.start_sec},{end})'{out}"
        )
        prev = out

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]",
        "-map", "0:a?",                 # source audio if present
    ] + delivery_video_opts(
        dur, audio_bitrate_bps=_STITCH_AUDIO_BPS if info.has_audio else 0
    ) + [
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-shortest",
        dst,
    ]
    _run(cmd)
