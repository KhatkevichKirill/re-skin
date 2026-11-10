"""Free, local QA tools for delivered/candidate videos (AICP-343).

Platform versions of the scratch scripts the AICP-332 run (projects 5b1b9b67 /
46cc6443, runs 657e1f7f and the 46cc6443 localisations) relied on. Everything
here is ffmpeg + numpy + OpenCV on local files: no model call, no network, no
database, no writes outside the output path a caller names.

The tools:

* :func:`detect_captions` -- burned-caption detector (white text touching a
  black outline, a short wide cluster in the centre band), sampled at 10 fps.
  Calibrate it on the SOURCE first (:func:`calibrate_captions`): a detector that
  does not see the source's captions proves nothing by reporting 0 hits on the
  result.
* :func:`locate_captions` -- where the captions sit: the same pixel rule over
  the whole frame (2 fps, streamed), centred changing text lines grouped by
  height into positions, a candidate band and a verdict (located / ambiguous
  / none). Evidence for choosing ``band_y``, not a calibration.
* :func:`comparison_sheet` / :func:`contact_sheet` -- a JPEG grid of source vs
  result frames at the same moments (actions, props, seams), plus a per-pair
  numeric difference so a caller without vision still gets a signal.
* :func:`alignment_check` / :func:`match_times` -- picture alignment of a
  result against its source: per-point best frame offset within +-N frames
  (stitched final vs source on one timeline), or a free search for each
  sampled result moment inside the source window (one generated clip whose
  length differs from its window).
* :func:`captions_from_script` -- caption lines built from the timed dialogue
  of the segment prompts (``[t0-t1] WHO: text``), scaled by each generated
  clip's real length and offset into the stitched final -- for when the ASR of
  the generated audio is garbled. The output is the ``{"lines": [...]}`` body
  ``PATCH /api/v2/captions/{id}/lines`` takes.

CLI (JSON on stdout), e.g. inside the api container::

    python -m app.qa_tools captions RESULT.mp4 --calibrate SOURCE.mp4 [--project PID | --config '{...}']
    python -m app.qa_tools locate SOURCE.mp4 [--project PID] [--sheet OUT.jpg]
    python -m app.qa_tools align SOURCE.mp4 FINAL.mp4
    python -m app.qa_tools match SOURCE.mp4 CLIP.mp4 --source-start 49.45 --duration 8.25
    python -m app.qa_tools sheet SOURCE.mp4 RESULT.mp4 OUT.jpg [--windows W.json]
    python -m app.qa_tools contact VIDEO.mp4 OUT.jpg --every 3
    python -m app.qa_tools script-captions SPEC.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Annotated, Iterable, Iterator, Literal, Optional, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError, field_validator
from pydantic import Field as PydField

from app import media


class QAToolError(RuntimeError):
    """Bad input or a failed ffmpeg decode."""


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def _finite(value, label: str, *, minimum: Optional[float] = None,
            positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise QAToolError(f"{label} must be a number") from None
    if isinstance(value, bool) or not math.isfinite(number):
        raise QAToolError(f"{label} must be a finite number")
    if positive and number <= 0:
        raise QAToolError(f"{label} must be > 0")
    if minimum is not None and number < minimum:
        raise QAToolError(f"{label} must be >= {minimum}")
    return number


def _window_args(start: Optional[float], duration: Optional[float]) -> list[str]:
    args: list[str] = []
    if start:
        args += ["-ss", f"{start:.6f}"]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]
    return args


def _decode_argv(path: str, *, fps: float, width: int, height: int, pix_fmt: str,
                 start: Optional[float], duration: Optional[float]) -> list[str]:
    return [
        media.FFMPEG, "-nostdin", "-v", "error",
        *_window_args(start, duration), "-i", path,
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]


_CHANNELS = {"rgb24": 3, "bgr24": 3, "gray": 1}


class LocalMedia:
    """Today's decoder: ffmpeg/ffprobe spawned directly (CLI, operator use).

    The assistant swaps in a confined backend with :func:`use_media`; every
    tool below reaches ffmpeg only through the active backend's three methods.
    """

    def probe(self, path: str) -> media.MediaInfo:
        try:
            return media.probe(path)
        except media.MediaError as exc:
            raise QAToolError(str(exc)) from exc

    def frames(self, path: str, *, fps: float, width: int, height: int,
               pix_fmt: str = "rgb24", start: Optional[float] = None,
               duration: Optional[float] = None) -> Iterator[np.ndarray]:
        channels = _CHANNELS[pix_fmt]
        cmd = media.harden_media_argv(_decode_argv(
            path, fps=fps, width=width, height=height, pix_fmt=pix_fmt,
            start=start, duration=duration))
        size = width * height * channels
        shape = (height, width) if channels == 1 else (height, width, channels)
        # stderr goes to a file, not a pipe: a damaged stream can log an error per
        # frame, and a full stderr pipe nobody reads would stall the decode.
        with tempfile.TemporaryFile() as errf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
            try:
                while True:
                    buf = proc.stdout.read(size)
                    if len(buf) < size:
                        break
                    yield np.frombuffer(buf, np.uint8).reshape(shape)
                proc.stdout.close()
                proc.wait()
                if proc.returncode != 0:
                    errf.seek(0)
                    tail = errf.read().decode("utf-8", "replace").strip().splitlines()[-5:]
                    raise QAToolError(f"ffmpeg decode of {path!r} failed: {' | '.join(tail)}")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if proc.stdout and not proc.stdout.closed:
                    proc.stdout.close()

    def grab(self, path: str, t: float) -> Optional[np.ndarray]:
        """One BGR frame at *t* (None past the end)."""
        cmd = media.harden_media_argv([
            media.FFMPEG, "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ])
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            raise QAToolError(f"ffmpeg frame grab of {path!r} at {t:.3f}s failed: {' | '.join(tail)}")
        if not proc.stdout:
            return None
        return cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)


_MEDIA: ContextVar = ContextVar("qa_tools_media", default=LocalMedia())


@contextmanager
def use_media(backend):
    """Run the tools in this context against *backend* (probe/frames/grab)."""
    token = _MEDIA.set(backend)
    try:
        yield backend
    finally:
        _MEDIA.reset(token)


def _iter_frames(path: str, *, fps: float, width: int, height: int,
                 pix_fmt: str = "rgb24", start: Optional[float] = None,
                 duration: Optional[float] = None) -> Iterator[np.ndarray]:
    """Yield decoded frames of *path* resampled to *fps* and *width*x*height*.

    Frames are read one at a time (one continuous decode), so memory stays
    flat whatever the video length. Raises :class:`QAToolError` (local) or the
    active backend's error if the decode fails.
    """
    return _MEDIA.get().frames(path, fps=fps, width=width, height=height,
                               pix_fmt=pix_fmt, start=start, duration=duration)


def _probe(path: str) -> media.MediaInfo:
    return _MEDIA.get().probe(path)


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


# ---------------------------------------------------------------------------
# 1. Burned-caption detector
# ---------------------------------------------------------------------------

# Calibrated on the 5b1b9b67 source (one-word white captions with a black
# outline at y~0.59-0.635): the scratch capdet.py decoded every video at
# 540x960 and the pixel bounds below are for that height.
CAPTION_ANALYSIS_HEIGHT = 960
CAPTION_BAND_Y = (0.50, 0.72)
CAPTION_BAND_X = (0.12, 0.88)
# A frame "hits" when its best cluster has MORE than this many outline-touching
# text pixels. 40 reproduces every per-segment hit count of the AICP-332 run
# (round 0 of run 657e1f7f: 0/0/0/21/8/6/15/0/9/0/32) and the source's 84 %.
CAPTION_HIT_THRESHOLD = 40
# A source hit ratio below this means the detector does not see THIS video's
# caption style, so 0 hits on a result is not evidence of a clean result.
CALIBRATION_MIN_RATIO = 0.2

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# CSS names accepted besides #RGB / #RRGGBB.
_NAMED_COLOURS = {
    "white": "#ffffff", "black": "#000000", "yellow": "#ffff00", "red": "#ff0000",
    "lime": "#00ff00", "green": "#008000", "blue": "#0000ff", "cyan": "#00ffff",
    "magenta": "#ff00ff",
}
_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _normalise_colour(value) -> str:
    if not isinstance(value, str):
        raise ValueError("colour must be a string like #FFFFFF")
    v = value.strip().lower()
    if v in _NAMED_COLOURS:
        return _NAMED_COLOURS[v]
    m = _HEX_RE.match(v)
    if not m:
        raise ValueError(f"unknown colour {value!r}: use #RGB, #RRGGBB or one of "
                         + ", ".join(sorted(_NAMED_COLOURS)))
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], np.int16)


Fraction01 = Annotated[float, PydField(ge=0.0, le=1.0, allow_inf_nan=False)]
Colour = Annotated[str, BeforeValidator(_normalise_colour)]


class CaptionDetectorConfig(BaseModel):
    """What a project's burned captions look like (AICP-343).

    Stored per project in ``video_projects.caption_detector``; NULL or ``{}``
    is the defaults below, which are the AICP-332 detector exactly (white
    text, black outline, centre band). Every field is optional.

    A pixel is TEXT when each channel is within ``text_tolerance`` of
    ``text_color`` and its channel deviations from that colour differ by less
    than ``text_spread`` (an unsaturated match). It is OUTLINE when each
    channel is within ``outline_tolerance`` of ``outline_color``. With
    ``outline_style="outline"`` only text pixels within ``outline_width_px``
    of an outline pixel count (white-on-black-stroke captions); ``"none"``
    counts every text-coloured pixel (plain captions: raise ``threshold`` and
    expect more false hits on same-coloured clothing). Clusters
    ``glyph_height_px`` tall (at 960 px frame height) and at least
    ``min_aspect`` x as wide as tall are text-shaped; a frame hits when its
    best cluster holds MORE than ``threshold`` such pixels.
    """

    model_config = ConfigDict(extra="forbid")

    band_y: tuple[Fraction01, Fraction01] = CAPTION_BAND_Y
    band_x: tuple[Fraction01, Fraction01] = CAPTION_BAND_X
    text_color: Colour = "#ffffff"
    text_tolerance: int = PydField(30, ge=1, le=255)
    text_spread: int = PydField(25, ge=1, le=511)
    outline_style: Literal["outline", "none"] = "outline"
    outline_color: Colour = "#000000"
    outline_tolerance: int = PydField(60, ge=1, le=255)
    outline_width_px: int = PydField(2, ge=1, le=8)
    glyph_height_px: tuple[int, int] = (12, 60)
    min_aspect: float = PydField(0.8, ge=0.1, le=10.0, allow_inf_nan=False)
    threshold: int = PydField(CAPTION_HIT_THRESHOLD, ge=1, le=100_000)

    @field_validator("band_y", "band_x")
    @classmethod
    def _band(cls, v):
        lo, hi = v
        if hi - lo < 0.02:
            raise ValueError("band must be [low, high] with high - low >= 0.02")
        return v

    @field_validator("glyph_height_px")
    @classmethod
    def _glyph(cls, v):
        lo, hi = v
        if not (2 <= lo < hi <= 600):
            raise ValueError("glyph_height_px must be [min, max] with 2 <= min < max <= 600")
        return v

    def stored(self) -> Optional[dict]:
        """What goes in the DB column: only the non-default fields, None when
        everything is default (so NULL keeps meaning "today's defaults").

        Exception: an explicitly given ``band_y`` is kept even when it equals
        the default -- it is an operator's choice, and "no band_y" means
        "locate the captions" to the autopilot review (:func:`has_band`)."""
        out = self.model_dump(mode="json", exclude_defaults=True)
        if "band_y" in self.model_fields_set and "band_y" not in out:
            out["band_y"] = [float(v) for v in self.band_y]
        return out or None


def has_band(stored: Optional[dict]) -> bool:
    """True when a stored ``caption_detector`` value names its own ``band_y``
    (an operator's or an applied located band); False = the default band,
    which the autopilot review checks against the located captions."""
    return isinstance(stored, dict) and stored.get("band_y") is not None


DEFAULT_CAPTION_CONFIG = CaptionDetectorConfig()


def caption_config(value=None) -> CaptionDetectorConfig:
    """Coerce None / a dict / a config into a validated config.

    Raises :class:`QAToolError` on an invalid dict (e.g. a hand-edited row).
    """
    if value is None:
        return DEFAULT_CAPTION_CONFIG
    if isinstance(value, CaptionDetectorConfig):
        return value
    if not isinstance(value, dict):
        raise QAToolError("caption detector config must be a JSON object")
    try:
        return CaptionDetectorConfig.model_validate(value)
    except ValidationError as exc:
        # Field paths and messages only: echoing the input could carry a
        # non-finite float, which is not JSON-serialisable in an API error.
        parts = ["%s: %s" % (".".join(str(x) for x in e["loc"]) or "config", e["msg"])
                 for e in exc.errors(include_url=False, include_input=False,
                                     include_context=False)]
        raise QAToolError("invalid caption detector config: " + "; ".join(parts)[:500]) from None


@dataclass
class CaptionFrame:
    t: float
    score: int
    # Best cluster's box in full-frame fractions (x0, y0, x1, y1), hits only.
    box: Optional[tuple[float, float, float, float]] = None
    # The whole caption LINE around that cluster (hits only): the best
    # cluster grown by the text-shaped clusters on its row next to it
    # (:func:`_line_around`). ``box`` is one word of a multi-word caption
    # ("KNOWING" of "KNOWING NOBODY."); a fill box needs the line.
    line: Optional[tuple[float, float, float, float]] = None


@dataclass
class CaptionSpan:
    start: float
    end: float
    frames: int
    max_score: int


@dataclass
class CaptionScan:
    path: str
    fps: float
    threshold: int
    frames: int
    hits: int
    hit_ratio: float
    spans: list[CaptionSpan]
    scores: list[CaptionFrame] = field(repr=False)
    calibration: Optional[dict] = None
    verdict: str = ""
    config: Optional[dict] = None

    def to_dict(self, *, include_scores: bool = False) -> dict:
        out = asdict(self)
        if not include_scores:
            out.pop("scores")
        return out


def _edge_mask(region: np.ndarray, cfg: CaptionDetectorConfig) -> np.ndarray:
    """The text-pixel mask of one RGB region (the detector's pixel rule).

    For white (255,255,255): |dev| < 30 <=> min channel > 225, and the dev
    spread is max - min channel, i.e. exactly the scratch rule. For a black
    outline: every channel < 60 <=> max channel < 60.
    """
    band = region.astype(np.int16)
    dev = band - _rgb(cfg.text_color)
    text = ((np.abs(dev).max(axis=2) < cfg.text_tolerance)
            & (dev.max(axis=2) - dev.min(axis=2) < cfg.text_spread))
    if cfg.outline_style != "outline":
        return text
    outline = (np.abs(band - _rgb(cfg.outline_color)).max(axis=2)
               < cfg.outline_tolerance).astype(np.uint8)
    near = cv2.dilate(outline, _CROSS, iterations=cfg.outline_width_px).astype(bool)
    return text & near


def _glyph_clusters(edge: np.ndarray, cfg: CaptionDetectorConfig
                    ) -> tuple[list[tuple[int, int, int, int, int]],
                               list[tuple[int, int, int, int, int]]]:
    """``(text_shaped, narrow)`` clusters of an edge mask as ``(x, y, w, h,
    count)`` (pixels of *edge*): the mask dilated three times, 4-connected,
    ``glyph_height_px`` tall; text-shaped ones are at least ``min_aspect`` x
    as wide as tall, narrow ones are not (a lone "I", "1" or "!": never a
    score, only part of a caption line). ``count`` = edge pixels inside the
    cluster's bounding box."""
    if not edge.any():
        return [], []
    grown = cv2.dilate(edge.astype(np.uint8), _CROSS, iterations=3)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=4)
    hmin, hmax = cfg.glyph_height_px
    shaped, narrow = [], []
    for i in range(1, n):
        bx, by, bw, bh = (int(v) for v in stats[i, :4])
        if not hmin <= bh <= hmax:
            continue
        (shaped if bw >= bh * cfg.min_aspect else narrow).append(
            (bx, by, bw, bh, int(edge[by:by + bh, bx:bx + bw].sum())))
    return shaped, narrow


def _text_clusters(edge: np.ndarray, cfg: CaptionDetectorConfig) -> list[tuple[int, int, int, int, int]]:
    """Text-shaped clusters of an edge mask (:func:`_glyph_clusters`)."""
    return _glyph_clusters(edge, cfg)[0]


#: A glyph-high cluster (text-shaped or narrow, :func:`_glyph_clusters`)
#: joins the caption line when it shares the line's row (vertical
#: overlap >= half the smaller height, :func:`_lines`' rule) and its gap to
#: the line is at most this many line heights. Word gaps of the white
#: outlined captions are ~0.2 line heights (81d0afc0: 6 px at 28 px).
LINE_GAP_HEIGHTS = 1.0


def _line_around(clusters: list[tuple[int, int, int, int, int]], best: int
                 ) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` of the text line holding
    ``clusters[best]``: the clusters on its row (vertical overlap with it >=
    half the smaller height) chained left to right while the horizontal gap
    is at most ``LINE_GAP_HEIGHTS`` x its height -- word by word, never
    across a wide gap. O(n log n) in the clusters."""
    bx, by, bw, bh, _ = clusters[best]
    gap = LINE_GAP_HEIGHTS * bh
    row = []
    for i, (cx, cy, cw, ch, _c) in enumerate(clusters):
        if i == best or min(by + bh, cy + ch) - max(by, cy) >= 0.5 * min(bh, ch):
            row.append((cx, cy, cx + cw, cy + ch, i == best))
    row.sort()
    found = None
    cur = None
    for cx0, cy0, cx1, cy1, is_best in row:
        if cur is not None and cx0 - cur[2] <= gap:
            cur = [cur[0], min(cur[1], cy0), max(cur[2], cx1), max(cur[3], cy1),
                   cur[4] or is_best]
        else:
            if cur is not None and cur[4]:
                found = cur
                break
            cur = [cx0, cy0, cx1, cy1, is_best]
    if found is None:
        found = cur
    return found[0], found[1], found[2], found[3]


def caption_score_line(frame: np.ndarray, config=None):
    """:func:`caption_score` plus the caption line around the best cluster:
    ``(score, box, line)``; ``line`` holds ``box`` and is None exactly
    when ``box`` is."""
    cfg = caption_config(config)
    H, W = frame.shape[:2]
    y0, y1 = int(H * cfg.band_y[0]), int(H * cfg.band_y[1])
    x0, x1 = int(W * cfg.band_x[0]), int(W * cfg.band_x[1])
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        return 0, None, None
    clusters, narrow = _glyph_clusters(_edge_mask(region, cfg), cfg)
    best, best_i = 0, None
    for i, (_bx, _by, _bw, _bh, count) in enumerate(clusters):
        if count > best:
            best, best_i = count, i
    if best_i is None:
        return 0, None, None
    bx, by, bw, bh, _ = clusters[best_i]
    box = ((bx + x0) / W, (by + y0) / H, (bx + bw + x0) / W, (by + bh + y0) / H)
    # Narrow glyphs (a one-letter word) never score, but they are words of
    # the line: without them a line ending in "I" stops short of it.
    lx0, ly0, lx1, ly1 = _line_around(clusters + narrow, best_i)
    line = ((lx0 + x0) / W, (ly0 + y0) / H, (lx1 + x0) / W, (ly1 + y0) / H)
    return best, box, line


def caption_score(frame: np.ndarray, config=None):
    """Score one RGB frame; returns ``(score, box)``.

    With the default config: white, unsaturated pixels (min channel > 225,
    spread < 25) that touch a dark outline (max channel < 60, dilated twice)
    are grouped into clusters (the edge mask dilated three times,
    4-connected). A cluster 12-60 px tall and at least 0.8x as wide as tall is
    text-shaped; the score is the largest edge-pixel count inside such a
    cluster's bounding box. Bit-for-bit the scratch ``capdet.py``
    (scipy.ndimage), redone with OpenCV. See :class:`CaptionDetectorConfig`
    for the general rule. Only the band is scored; :func:`locate_captions`
    applies the same rule to the whole frame.
    """
    score, box, _line = caption_score_line(frame, config)
    return score, box


def _caption_dims(path: str) -> tuple[int, int]:
    info = _probe(path)
    if info.width <= 0 or info.height <= 0:
        raise QAToolError(f"{path!r} has no usable video size")
    h = CAPTION_ANALYSIS_HEIGHT
    return _even(info.width * h / info.height), h


def _spans(frames: Sequence[CaptionFrame], threshold: int, fps: float,
           merge_gap_sec: float) -> list[CaptionSpan]:
    spans: list[CaptionSpan] = []
    for fr in frames:
        if fr.score <= threshold:
            continue
        if spans and fr.t - spans[-1].end <= merge_gap_sec + 1e-6:
            sp = spans[-1]
            sp.end = fr.t
            sp.frames += 1
            sp.max_score = max(sp.max_score, fr.score)
        else:
            spans.append(CaptionSpan(start=fr.t, end=fr.t, frames=1, max_score=fr.score))
    return spans


def scan_captions(path: str, *, fps: float = 10.0, offset: float = 0.0,
                  start: Optional[float] = None, duration: Optional[float] = None,
                  threshold: Optional[int] = None, config=None,
                  merge_gap_sec: float = 0.5) -> CaptionScan:
    """Score every sampled frame of *path* (or of ``[start, start+duration)``).

    Timestamps are ``offset + start + i/fps`` so a per-segment clip can be
    reported on the source/final timeline by passing its offset. *config* is
    a :class:`CaptionDetectorConfig` or its dict (None = defaults); an
    explicit *threshold* overrides the config's.
    """
    cfg = caption_config(config)
    if threshold is not None:
        threshold = int(_finite(threshold, "threshold", minimum=1))
        if threshold != cfg.threshold:
            cfg = cfg.model_copy(update={"threshold": threshold})
    threshold = cfg.threshold
    fps = _finite(fps, "fps", positive=True)
    offset = _finite(offset, "offset")
    start = None if start is None else _finite(start, "start", minimum=0)
    duration = None if duration is None else _finite(duration, "duration", positive=True)
    width, height = _caption_dims(path)
    base = offset + (start or 0.0)
    frames: list[CaptionFrame] = []
    for i, frame in enumerate(_iter_frames(path, fps=fps, width=width, height=height,
                                           start=start, duration=duration)):
        score, box, line = caption_score_line(frame, cfg)
        t = round(base + i / fps, 3)
        hit = bool(box) and score > threshold
        frames.append(CaptionFrame(
            t=t, score=score,
            box=tuple(round(v, 4) for v in box) if hit else None,
            line=tuple(round(v, 4) for v in line) if hit else None))
    if not frames:
        raise QAToolError(f"no frames decoded from {path!r} (window past the end?)")
    hits = sum(1 for f in frames if f.score > threshold)
    return CaptionScan(
        path=path, fps=fps, threshold=threshold, frames=len(frames), hits=hits,
        hit_ratio=round(hits / len(frames), 4) if frames else 0.0,
        spans=_spans(frames, threshold, fps, merge_gap_sec), scores=frames,
        config=cfg.model_dump(mode="json"))


def caption_geometry(scan: CaptionScan) -> Optional[dict]:
    """Percentiles of the hit boxes (full-frame fractions) -- where the
    captions sit, e.g. for a coordinate-based removal prompt."""
    boxes = [f.box for f in scan.scores if f.box]
    if not boxes:
        return None
    a = np.array(boxes)
    p = lambda col, q: round(float(np.percentile(a[:, col], q)), 3)  # noqa: E731
    return {"n": len(boxes), "x0_p5": p(0, 5), "x1_p95": p(2, 95),
            "y0_p5": p(1, 5), "y0_p50": p(1, 50), "y1_p50": p(3, 50), "y1_p95": p(3, 95)}


def calibrate_captions(source: str, **kwargs) -> dict:
    """Run the detector on the SOURCE (which is known to carry captions)."""
    scan = scan_captions(source, **kwargs)
    return {
        "source": source, "frames": scan.frames, "hits": scan.hits,
        "hit_ratio": scan.hit_ratio,
        "calibrated": scan.hit_ratio >= CALIBRATION_MIN_RATIO,
        "geometry": caption_geometry(scan),
    }


def detect_captions(path: str, *, calibrate: Optional[str] = None,
                    calibrate_start: Optional[float] = None,
                    calibrate_duration: Optional[float] = None,
                    **kwargs) -> CaptionScan:
    """Scan *path*; with ``calibrate=SOURCE`` also scan the source first,
    with the SAME config and threshold.

    ``verdict`` is one of ``clean`` (0 hits, calibrated), ``captions_found``,
    ``clean_uncalibrated`` (0 hits but no calibration or a failed one -- not
    evidence on its own; also what a config that does not match the source's
    caption style yields).
    """
    calibration = None
    if calibrate:
        cal_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("fps", "threshold", "merge_gap_sec", "config")}
        calibration = calibrate_captions(calibrate, start=calibrate_start,
                                         duration=calibrate_duration, **cal_kwargs)
    scan = scan_captions(path, **kwargs)
    scan.calibration = calibration
    if scan.hits:
        scan.verdict = "captions_found"
    elif calibration and calibration["calibrated"]:
        scan.verdict = "clean"
    else:
        scan.verdict = "clean_uncalibrated"
    return scan


# ---------------------------------------------------------------------------
# 1b. Caption locator: where in the frame are this video's captions?
# ---------------------------------------------------------------------------
#
# The band above is a guess (the 5b1b9b67 captions sat at y ~0.6); captions
# sit anywhere -- top, middle, bottom -- and a band that misses them makes the
# detector "uncalibrated" (514067f2: white words at y 0.76-0.80, 0 hits in the
# default band). :func:`locate_captions` applies the SAME pixel rule to the
# whole frame and finds where caption-like text lives:
#
# * every text-shaped cluster of a frame (not just the best one) is merged
#   into lines (clusters overlapping vertically by half the smaller height);
# * a caption line is horizontally centred-ish (centre x within
#   ``LOCATE_CENTRE_TOL`` of the middle) -- a book cover, a phone screen or a
#   sign rarely is, and its glyphs are usually outside the caption size;
# * text held still (a cluster in a run of IoU >= ``STATIC_IOU`` for
#   ``LOCATE_STATIC_SEC``) is counted as static: UI, an end card, scene text.
#   Static evidence is reported, never silently dropped -- a held caption line
#   looks the same;
# * the centre y of every centred line is binned (0.01 of the frame height);
#   contiguous bins (gaps <= ``LOCATE_GAP``) form a *position*. Captions stay
#   at one baseline across the video, so they pile into one narrow position
#   while scene text that moves with the camera spreads thin.
#
# Location is EVIDENCE, not calibration: the review re-measures the candidate
# band with the normal detector (one best cluster per frame, 10 fps) before
# anything uses it (app.subtitle_loop.review_segmentation).

LOCATE_FPS = 2.0
LOCATE_MAX_FRAMES = 1200          # a 10-minute source at 2 fps; longer -> lower fps
LOCATE_MAX_CLUSTERS = 64          # per frame (largest first): bounds memory and time
LOCATE_CENTRE_TOL = 0.15
LOCATE_STATIC_IOU = 0.9
LOCATE_STATIC_SEC = 2.0
LOCATE_BINS = 100
LOCATE_GAP = 0.03
LOCATE_MARGIN = 0.03
#: A located band is never thinner than this (centred on the captions): room
#: for a second caption line and for a leftover a few pixels off.
LOCATE_MIN_HEIGHT = 0.16
#: A candidate taller than this (two distant positions) is ambiguous: the
#: band would take in everything between them.
LOCATE_MAX_HEIGHT = 0.30
#: Support (share of sampled frames with a non-static centred line in the
#: position) of the main position for a "located" verdict ...
LOCATE_SUPPORT = 0.30
#: ... and the floor below which nothing counts as a caption position at all.
LOCATE_SUPPORT_MIN = 0.10
#: A position is caption-like when at least this share of its hit frames show
#: a non-static line (words change; UI / end cards / scene text hold still).
LOCATE_CAPTION_LIKE = 0.4
#: Caption-like text elsewhere with at least this support makes it ambiguous.
LOCATE_COMPETING = 0.05
#: Centred text held still (static) in at least this share of the frames
#: outside the candidate competes too (a hook bubble, a held caption track).
LOCATE_HELD = 0.20
#: Positions kept in the result (members and competitors first).
LOCATE_MAX_POSITIONS = 12
#: More than this share of the candidate's lines crossing its edges: ambiguous.
LOCATE_LINES_OUTSIDE = 0.05
LOCATE_MIN_FRAMES = 20
LOCATE_WINDOWS = 5
LOCATE_MIN_WINDOWS = 3
_LOCATE_SAMPLE_TIMES = 12

LOCATED = "located"
AMBIGUOUS = "ambiguous"
NOT_FOUND = "none"


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _frame_clusters(frame: np.ndarray, cfg: CaptionDetectorConfig) -> list[tuple]:
    """Every text-shaped cluster of the WHOLE frame scoring above the
    threshold, as ``(x0, y0, x1, y1, count)`` fractions, largest first."""
    H, W = frame.shape[:2]
    found = [(bx / W, by / H, (bx + bw) / W, (by + bh) / H, c)
             for bx, by, bw, bh, c in _text_clusters(_edge_mask(frame, cfg), cfg)
             if c > cfg.threshold]
    found.sort(key=lambda b: -b[4])
    return found[:LOCATE_MAX_CLUSTERS]


def _static_marks(per_frame: list[list[tuple]], fps: float) -> list[list[bool]]:
    """Per cluster: part of a run of the same box (IoU >= LOCATE_STATIC_IOU
    frame to frame) lasting at least LOCATE_STATIC_SEC?"""
    need = max(2, int(round(LOCATE_STATIC_SEC * fps)))
    run = [[1] * len(b) for b in per_frame]
    prev = [[None] * len(b) for b in per_frame]
    for i in range(1, len(per_frame)):
        for j, box in enumerate(per_frame[i]):
            for k, p in enumerate(per_frame[i - 1]):
                if _box_iou(box, p) >= LOCATE_STATIC_IOU:
                    run[i][j] = run[i - 1][k] + 1
                    prev[i][j] = k
                    break
    marks = [[False] * len(b) for b in per_frame]
    for i in range(len(per_frame) - 1, -1, -1):
        for j in range(len(per_frame[i])):
            if run[i][j] >= need and not marks[i][j]:
                ii, jj = i, j
                while jj is not None and ii >= 0 and not marks[ii][jj]:
                    marks[ii][jj] = True
                    jj = prev[ii][jj]
                    ii -= 1
    return marks


def _lines(clusters: list[tuple], marks: list[bool]) -> list[list]:
    """Clusters merged into text lines: ``[x0, y0, x1, y1, count, static_count]``."""
    order = sorted(range(len(clusters)), key=lambda i: (clusters[i][1] + clusters[i][3]) / 2)
    out: list[list] = []
    for i in order:
        x0, y0, x1, y1, c = clusters[i]
        st = c if marks[i] else 0
        for ln in out:
            overlap = min(ln[3], y1) - max(ln[1], y0)
            if overlap >= 0.5 * min(ln[3] - ln[1], y1 - y0):
                ln[0], ln[1] = min(ln[0], x0), min(ln[1], y0)
                ln[2], ln[3] = max(ln[2], x1), max(ln[3], y1)
                ln[4] += c
                ln[5] += st
                break
        else:
            out.append([x0, y0, x1, y1, c, st])
    return out


def _pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(np.array(values), q)), 4)


def _spread(times: list[float], n: int) -> list[float]:
    if n <= 0:
        return []
    if len(times) <= n:
        return list(times)
    if n == 1:
        return [times[len(times) // 2]]
    return [times[int(round(i * (len(times) - 1) / (n - 1)))] for i in range(n)]


def _band_around(y0: float, y1: float) -> list[float]:
    """``[y0 - margin, y1 + margin]``, at least LOCATE_MIN_HEIGHT tall
    (grown around its centre), clamped to the frame."""
    lo, hi = y0 - LOCATE_MARGIN, y1 + LOCATE_MARGIN
    if hi - lo < LOCATE_MIN_HEIGHT:
        mid = (lo + hi) / 2
        lo, hi = mid - LOCATE_MIN_HEIGHT / 2, mid + LOCATE_MIN_HEIGHT / 2
    if lo < 0:
        lo, hi = 0.0, min(1.0, hi - lo)
    if hi > 1:
        lo, hi = max(0.0, lo - (hi - 1.0)), 1.0
    return [round(lo, 2), round(hi, 2)]


def locate_captions(path: str, *, config=None, fps: float = LOCATE_FPS,
                    start: Optional[float] = None,
                    duration: Optional[float] = None) -> dict:
    """Find where *path*'s burned captions sit (full frame, streamed).

    *config* supplies the pixel rule (text / outline colour, glyph size,
    threshold); its band is ignored. Returns::

        {"state": "located" | "ambiguous" | "none",
         "candidate_band_y": [y0, y1] | None, "frames", "fps",
         "positions": [...strongest first...], "members": [indices],
         "reasons": [...], "detector": {...the pixel rule used...}}

    ``located`` = one caption position (or several close ones) with enough
    support spread over the video; ``ambiguous`` = a candidate exists but is
    weak, brief, tall (distant positions) or has a competitor; ``none`` = no
    caption-like centred text the rule can see -- which does NOT mean the
    video has no captions (another colour/outline/size is invisible to it).
    """
    cfg = caption_config(config)
    fps = _finite(fps, "fps", positive=True)
    start = None if start is None else _finite(start, "start", minimum=0)
    duration = None if duration is None else _finite(duration, "duration", positive=True)
    info = _probe(path)
    span = duration if duration is not None else max(0.0, info.duration_sec - (start or 0.0))
    if span > 0 and span * fps > LOCATE_MAX_FRAMES:
        fps = LOCATE_MAX_FRAMES / span
    width, height = _caption_dims(path)
    per_frame: list[list[tuple]] = []
    for frame in _iter_frames(path, fps=fps, width=width, height=height,
                              start=start, duration=duration):
        per_frame.append(_frame_clusters(frame, cfg))
    if not per_frame:
        raise QAToolError(f"no frames decoded from {path!r} (window past the end?)")
    n = len(per_frame)
    base = start or 0.0
    marks = _static_marks([[c[:4] for c in f] for f in per_frame], fps)

    # Centred lines per frame, with their static share.
    lines: list[list[tuple]] = []
    for clusters, mk in zip(per_frame, marks):
        kept = []
        for ln in _lines(clusters, mk):
            if abs((ln[0] + ln[2]) / 2 - 0.5) > LOCATE_CENTRE_TOL:
                continue
            kept.append((ln[0], ln[1], ln[2], ln[3], ln[4], ln[5] >= 0.5 * ln[4]))
        lines.append(kept)

    occupied = [False] * LOCATE_BINS
    for kept in lines:
        for ln in kept:
            occupied[min(LOCATE_BINS - 1, int((ln[1] + ln[3]) / 2 * LOCATE_BINS))] = True
    regions: list[list[int]] = []
    gap = int(round(LOCATE_GAP * LOCATE_BINS))
    for b in range(LOCATE_BINS):
        if not occupied[b]:
            continue
        if regions and b - regions[-1][1] <= gap + 1:
            regions[-1][1] = b
        else:
            regions.append([b, b])

    positions = []
    for b0, b1 in regions:
        lo, hi = b0 / LOCATE_BINS, (b1 + 1) / LOCATE_BINS
        nonstatic = static = 0
        y0s, y1s, scores, times, windows = [], [], [], [], set()
        held_y0, held_y1, held_times = [], [], []
        for i, kept in enumerate(lines):
            inside = [ln for ln in kept if lo <= (ln[1] + ln[3]) / 2 < hi]
            if not inside:
                continue
            moving = [ln for ln in inside if not ln[5]]
            if not moving:
                static += 1
                held_times.append(round(base + i / fps, 3))
                held_y0 += [ln[1] for ln in inside]
                held_y1 += [ln[3] for ln in inside]
                continue
            nonstatic += 1
            t = round(base + i / fps, 3)
            times.append(t)
            windows.add(min(LOCATE_WINDOWS - 1, int(i * LOCATE_WINDOWS / n)))
            for ln in moving:
                y0s.append(ln[1])
                y1s.append(ln[3])
                scores.append(ln[4])
        if not nonstatic:
            # Held-still text only: its extent and moments, for the evidence.
            y0_p5, y1_p95 = _pct(held_y0, 5), _pct(held_y1, 95)
            positions.append({"y_range": [round(lo, 2), round(hi, 2)], "support": 0.0,
                              "nonstatic_frames": 0, "static_frames": static,
                              "caption_like": False, "windows_hit": 0,
                              "y0_p5": y0_p5, "y1_p95": y1_p95,
                              "band_y": _band_around(y0_p5, y1_p95),
                              "sample_times": _spread(held_times, _LOCATE_SAMPLE_TIMES),
                              "_y": ([], [])})
            continue
        y0_p5, y1_p95 = _pct(y0s, 5), _pct(y1s, 95)
        positions.append({
            "y_range": [round(lo, 2), round(hi, 2)],
            "support": round(nonstatic / n, 4),
            "nonstatic_frames": nonstatic, "static_frames": static,
            "caption_like": nonstatic / (nonstatic + static) >= LOCATE_CAPTION_LIKE,
            "windows_hit": len(windows),
            "y0_min": round(min(y0s), 4), "y0_p5": y0_p5, "y1_p95": y1_p95,
            "y1_max": round(max(y1s), 4),
            "band_y": _band_around(y0_p5, y1_p95),
            "median_score": int(np.median(np.array(scores))),
            "sample_times": _spread(times, _LOCATE_SAMPLE_TIMES),
            "_y": (y0s, y1s),
        })
    positions.sort(key=lambda p: (-p["support"], p["y_range"][0]))
    lines_y = {id(p): p.pop("_y", ([], [])) for p in positions}

    out = {"path": path, "fps": round(fps, 4), "frames": n, "state": NOT_FOUND,
           "candidate_band_y": None, "positions": positions, "members": [],
           "competing": [], "lines_outside_band": None,
           "reasons": [],
           "detector": {k: cfg.model_dump(mode="json")[k] for k in (
               "text_color", "outline_style", "outline_color", "glyph_height_px",
               "threshold")}}
    likes = [i for i, p in enumerate(out["positions"]) if p["caption_like"]
             and p["support"] >= LOCATE_SUPPORT_MIN]
    if not likes:
        # No candidate; weak caption-like text and text held still in a large
        # share of the frames are still reported as competitors (they may be
        # captions the configured band misses -- the review's eye check shows
        # them).
        out["positions"] = sorted(out["positions"], key=lambda p: (
            -max(p["support"], p["static_frames"] / n)))[:LOCATE_MAX_POSITIONS]
        out["competing"] = [i for i, p in enumerate(out["positions"])
                            if p["static_frames"] >= LOCATE_HELD * n
                            or (p["caption_like"] and p["support"] >= LOCATE_COMPETING)]
        best = positions[0] if positions else None
        held = max(positions, key=lambda p: p["static_frames"]) if positions else None
        if best is None:
            out["reasons"].append("no text-shaped clusters at all")
        else:
            out["reasons"].append("no caption-like centred text: the strongest covers %.0f %% "
                                  "of the frames" % (100 * best["support"]))
        if held is not None and held["static_frames"] >= LOCATE_SUPPORT_MIN * n:
            # Held still for >= 2 s: UI / an end card -- or captions that each
            # stay on screen that long, which this detector treats as static.
            out["reasons"].append("centred text held still at y %.2f-%.2f in %.0f %% of the "
                                  "frames (UI, an end card, or long-held captions)" % (
                                      held["y_range"][0], held["y_range"][1],
                                      100 * held["static_frames"] / n))
        return out
    main = out["positions"][likes[0]]
    members = [i for i in likes
               if out["positions"][i]["support"] >= max(LOCATE_SUPPORT_MIN, main["support"] / 3)]
    member_set = set(members)
    out["members"] = members
    band = _band_around(min(out["positions"][i]["y0_p5"] for i in members),
                        max(out["positions"][i]["y1_p95"] for i in members))
    out["candidate_band_y"] = band
    reasons = out["reasons"]
    if n < LOCATE_MIN_FRAMES:
        reasons.append(f"only {n} sampled frames (needs {LOCATE_MIN_FRAMES})")
    if main["support"] < LOCATE_SUPPORT:
        reasons.append("the captions show in %.0f %% of the frames (needs %.0f %%)"
                       % (100 * main["support"], 100 * LOCATE_SUPPORT))
    if main["windows_hit"] < LOCATE_MIN_WINDOWS:
        reasons.append("the captions show in %d of %d parts of the video (needs %d)"
                       % (main["windows_hit"], LOCATE_WINDOWS, LOCATE_MIN_WINDOWS))
    if band[1] - band[0] > LOCATE_MAX_HEIGHT + 1e-9:
        reasons.append("caption text at distant heights (band y %.2f-%.2f is taller than "
                       "%.2f)" % (band[0], band[1], LOCATE_MAX_HEIGHT))
    # Every line of the candidate's positions, not just the p5/p95 the band
    # is built from: a cut-off second line or drift shows up here.
    y0s = [y for i in members for y in lines_y[id(out["positions"][i])][0]]
    y1s = [y for i in members for y in lines_y[id(out["positions"][i])][1]]
    cut = sum(1 for a, b in zip(y0s, y1s) if a < band[0] - 1e-6 or b > band[1] + 1e-6)
    out["lines_outside_band"] = round(cut / len(y0s), 4) if y0s else 0.0
    if y0s and out["lines_outside_band"] > LOCATE_LINES_OUTSIDE:
        reasons.append("%.0f %% of the caption lines reach outside the band"
                       % (100 * out["lines_outside_band"]))
    # Any other caption-like centred text with real support competes: it may
    # be captions the band would miss (a moved caption, a second track).
    competing = [i for i, p in enumerate(out["positions"])
                 if i not in member_set and p["caption_like"]
                 and p["support"] >= LOCATE_COMPETING]
    # Text held still elsewhere for a large share of the video: UI -- or a
    # second caption track whose lines stay up for seconds (a hook bubble).
    held = [i for i, p in enumerate(out["positions"])
            if i not in member_set and i not in competing
            and p["static_frames"] >= LOCATE_HELD * n]
    if competing:
        reasons.append("other centred text at y %s" % ", ".join(
            "%.2f-%.2f (%.0f %%)" % (out["positions"][i]["y_range"][0],
                                     out["positions"][i]["y_range"][1],
                                     100 * out["positions"][i]["support"]) for i in competing))
    if held:
        reasons.append("centred text held still at y %s" % ", ".join(
            "%.2f-%.2f (%.0f %% of the frames)" % (
                out["positions"][i]["y_range"][0], out["positions"][i]["y_range"][1],
                100 * out["positions"][i]["static_frames"] / n) for i in held))
    competing = competing + held
    # Members and competitors first (summaries keep the head of the list).
    order = members + competing + [i for i in range(len(out["positions"]))
                                   if i not in member_set and i not in competing]
    remap = {old: new for new, old in enumerate(order)}
    out["positions"] = [out["positions"][i] for i in order][:LOCATE_MAX_POSITIONS]
    out["members"] = [remap[i] for i in members if remap[i] < LOCATE_MAX_POSITIONS]
    if len(members) > len(out["members"]):
        reasons.append("text at %d heights" % len(members))
    out["competing"] = [remap[i] for i in competing if remap[i] < LOCATE_MAX_POSITIONS]
    out["state"] = AMBIGUOUS if reasons else LOCATED
    return out


def project_caption_config(project_id: str) -> Optional[dict]:
    """Read ``video_projects.caption_detector`` (read-only, for the CLI)."""
    from app.db import SessionLocal  # lazy: the pure tools need no database
    from app.models import VideoProject

    from sqlalchemy.exc import SQLAlchemyError

    try:
        with SessionLocal() as session:
            project = session.get(VideoProject, project_id)
            if project is None:
                raise QAToolError(f"project {project_id!r} not found")
            return project.caption_detector
    except SQLAlchemyError as exc:
        raise QAToolError(f"could not read project {project_id!r}: {exc}") from None


# ---------------------------------------------------------------------------
# 2. Picture alignment against the source
# ---------------------------------------------------------------------------

ALIGN_FPS = 30.0
ALIGN_SIZE = 72  # short side of the grayscale thumbnails the matcher compares


def _align_dims(path: str) -> tuple[int, int]:
    info = _probe(path)
    long_side = round(ALIGN_SIZE * 16 / 9)  # 128
    return (ALIGN_SIZE, long_side) if info.height >= info.width else (long_side, ALIGN_SIZE)


def _gray_stack(path: str, *, dims: tuple[int, int], fps: float,
                start: Optional[float] = None,
                duration: Optional[float] = None) -> np.ndarray:
    w, h = dims
    frames = list(_iter_frames(path, fps=fps, width=w, height=h, pix_fmt="gray",
                               start=start, duration=duration))
    if not frames:
        raise QAToolError(f"no frames decoded from {path!r}")
    return np.stack(frames)


def _mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _best_offset_index(diffs: Sequence[float], max_offset: int) -> int:
    """Index of the smallest diff; among equal minima the one closest to
    offset 0 (static footage matches every neighbour equally well)."""
    best = min(diffs)
    ties = [i for i, d in enumerate(diffs) if d <= best + 1e-9]
    return min(ties, key=lambda i: (abs(i - max_offset), i))


def alignment_check(source: str, result: str, *, fps: float = ALIGN_FPS,
                    step_frames: int = 15, max_offset: int = 9,
                    tolerance: int = 1, source_start: float = 0.0,
                    result_start: float = 0.0, duration: Optional[float] = None,
                    window_sec: float = 10.0) -> dict:
    """Best frame offset of *result* against *source* every *step_frames*.

    Both videos are on ONE timeline (a stitched final vs its source, or a
    window of each): for each checked result frame ``k`` the source frames
    ``k-max_offset .. k+max_offset`` are compared and the closest one wins.
    Offset > 0 means the result shows a LATER source frame (picture is
    early). Points at scene cuts and in motion blur are matcher noise, so the
    verdict is on the per-window MEDIAN offset (drift), with the individual
    outliers listed for a visual check. Same numbers as the scratch
    ``align_full.py`` at the defaults.
    """
    fps = _finite(fps, "fps", positive=True)
    step_frames, max_offset, tolerance = int(step_frames), int(max_offset), int(tolerance)
    if step_frames < 1 or max_offset < 0 or tolerance < 0:
        raise QAToolError("step_frames >= 1, max_offset >= 0, tolerance >= 0")
    dims = _align_dims(source)
    S = _gray_stack(source, dims=dims, fps=fps, start=source_start or None, duration=duration)
    F = _gray_stack(result, dims=dims, fps=fps, start=result_start or None, duration=duration)
    n = min(len(S), len(F))
    margin = max(step_frames, max_offset)
    points = []
    for k in range(margin, n - margin, step_frames):
        diffs = [_mad(S[k + o], F[k]) for o in range(-max_offset, max_offset + 1)]
        j = _best_offset_index(diffs, max_offset)
        points.append({"t": round(result_start + k / fps, 2),
                       "offset": j - max_offset, "err": round(diffs[j], 1)})
    windows = []
    if points:
        t0 = points[0]["t"]
        buckets: dict[int, list[int]] = {}
        for p in points:
            buckets.setdefault(int((p["t"] - t0) // window_sec), []).append(p["offset"])
        for b, offs in sorted(buckets.items()):
            windows.append({"start": round(t0 + b * window_sec, 2), "points": len(offs),
                            "median_offset": float(np.median(offs))})
    outliers = [p for p in points if abs(p["offset"]) > tolerance]
    drift = [w for w in windows if abs(w["median_offset"]) > tolerance]
    # Frames past the shorter video are never compared, so a result that is
    # missing (or adds) picture at the end must fail on its length alone.
    length_ok = abs(len(F) - len(S)) <= tolerance
    return {
        "source": source, "result": result, "fps": fps,
        "source_frames": len(S), "result_frames": len(F),
        "frame_count_delta": len(F) - len(S),
        "points": len(points),
        "nonzero": [[p["t"], p["offset"]] for p in points if p["offset"]],
        "outliers": outliers,
        "at_search_limit": [p["t"] for p in points if abs(p["offset"]) == max_offset and max_offset],
        "windows": windows,
        "max_abs_window_median": max((abs(w["median_offset"]) for w in windows), default=0.0),
        "length_ok": length_ok,
        "in_sync": bool(points) and not drift and length_ok,
        "tolerance_frames": tolerance,
        "_points": points,
    }


def match_times(source: str, result: str, *, source_start: float, duration: float,
                every_sec: float = 0.5, first_sec: float = 0.2,
                fps: float = ALIGN_FPS, pad_sec: float = 0.5) -> dict:
    """For sampled moments of a generated CLIP, find the best-matching frame
    of the source window ``[source_start, source_start+duration]`` (searched
    with *pad_sec* either side). ``drift`` = matched source time minus where
    the moment should be; a growing drift is a speed change, a constant one a
    shift. Same numbers as the scratch ``align.py``.
    """
    source_start = _finite(source_start, "source_start", minimum=0)
    duration = _finite(duration, "duration", positive=True)
    every_sec = _finite(every_sec, "every_sec", positive=True)
    dims = _align_dims(source)
    s0 = max(0.0, source_start - pad_sec)
    S = _gray_stack(source, dims=dims, fps=fps, start=s0 or None,
                    duration=duration + (source_start - s0) + pad_sec)
    # The whole clip: a generated clip can run longer than its window.
    R = _gray_stack(result, dims=dims, fps=fps)
    samples = []
    for rt in np.arange(first_sec, len(R) / fps - 0.1, every_sec):
        r = R[int(rt * fps)]
        diffs = [_mad(S[j], r) for j in range(len(S))]
        j = int(np.argmin(diffs))  # earliest of equal matches, as align.py
        src_t = s0 + j / fps
        samples.append({"result_t": round(float(rt), 2), "source_t": round(src_t, 2),
                        "offset_in_window": round(src_t - source_start, 2),
                        "drift": round(src_t - source_start - float(rt), 2),
                        "err": round(diffs[j], 1)})
    drifts = [s["drift"] for s in samples]
    return {"source": source, "result": result, "source_start": source_start,
            "duration": duration, "result_duration": round(len(R) / fps, 3),
            "samples": samples,
            "median_drift": float(np.median(drifts)) if drifts else None,
            "max_abs_drift": max((abs(d) for d in drifts), default=None)}


# ---------------------------------------------------------------------------
# 3. Frame sheets
# ---------------------------------------------------------------------------

_SHEET_DIFF_DIMS = (72, 128)


def _grab_frame(path: str, t: float) -> Optional[np.ndarray]:
    """One BGR frame at *t* (None past the end)."""
    return _MEDIA.get().grab(path, t)


def _cell(img: Optional[np.ndarray], label: str, width: int, height: int) -> np.ndarray:
    if img is None:
        cell = np.full((height, width, 3), 40, np.uint8)
        label += " (none)"
    else:
        cell = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    scale = max(0.35, width / 450)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(cell, (0, 0), (min(width, tw + 8), th + base + 8), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 255, 255), 1, cv2.LINE_AA)
    return cell


def _write_grid(cells: list[np.ndarray], columns: int, out_path: str) -> tuple[int, int]:
    if not cells:
        raise QAToolError("nothing to put on the sheet")
    h, w = cells[0].shape[:2]
    rows = math.ceil(len(cells) / columns)
    grid = np.zeros((rows * h, columns * w, 3), np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, columns)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = c
    if not cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise QAToolError(f"could not write {out_path!r}")
    return grid.shape[1], grid.shape[0]


def _cell_dims(path: str, cell_width: int) -> tuple[int, int]:
    info = _probe(path)
    return int(cell_width), _even(cell_width * info.height / info.width)


def _sample_times(start: float, end: float, *, fractions: Optional[Sequence[float]],
                  every_sec: Optional[float]) -> list[float]:
    length = end - start
    if every_sec:
        n = int(math.floor(length / every_sec + 1e-9))
        return [start + every_sec * (i + 0.5) for i in range(n)] or [start + length / 2]
    return [start + length * f for f in (fractions or (0.1, 0.5, 0.9))]


def comparison_sheet(source: str, result: str, out_path: Optional[str], *,
                     windows: Optional[Iterable[dict]] = None,
                     fractions: Optional[Sequence[float]] = (0.1, 0.5, 0.9),
                     every_sec: Optional[float] = None, time_map: str = "offset",
                     cell_width: int = 200, columns: int = 3) -> dict:
    """Source/result frame pairs side by side, one JPEG.

    *windows* is a list of ``{"source_start", "source_end", "result"?,
    "result_start"?, "result_end"?, "label"?}``. Without it the whole source is one window
    and *result* is on the same timeline (a stitched final). A window with its
    own ``result`` (a per-segment clip) starts at ``result_start`` (default 0).
    ``time_map="scale"`` stretches the window onto the clip's real length
    (Seedance returns clips up to ~0.8 s off); ``"offset"`` keeps 1:1 time.
    Samples: *fractions* of each window, or one frame every *every_sec*.
    Each pair also gets ``diff`` = mean absolute grayscale difference at
    72x128 (0 = identical; a jump flags a changed prop/action/seam).
    ``out_path=None`` computes the samples only (no grid, nothing written).
    """
    if time_map not in ("offset", "scale"):
        raise QAToolError("time_map must be 'offset' or 'scale'")
    columns = max(1, int(columns))
    src_info = _probe(source)
    if windows is None:
        windows = [{"source_start": 0.0, "source_end": src_info.duration_sec,
                    "result": result, "result_start": 0.0}]
    cw, ch = _cell_dims(source, cell_width) if out_path is not None else (0, 0)
    cells, samples = [], []
    for wi, w in enumerate(windows):
        if not isinstance(w, dict):
            raise QAToolError(f"window {wi} must be an object")
        s0 = _finite(w.get("source_start", 0.0), "source_start", minimum=0)
        s1 = _finite(w.get("source_end", src_info.duration_sec), "source_end")
        if s1 <= s0:
            raise QAToolError(f"window {wi}: source_end must be > source_start")
        rpath = w.get("result") or result
        own = bool(w.get("result"))
        r0 = _finite(w.get("result_start", 0.0 if own else s0), "result_start", minimum=0)
        scale = 1.0
        if time_map == "scale":
            # Stretch the window onto [result_start, result_end]; a window's
            # own clip ends at the clip's end. A window on the shared
            # timeline without result_end keeps 1:1 time.
            if w.get("result_end") is not None:
                rdur = _finite(w["result_end"], "result_end") - r0
            elif own:
                rdur = _probe(rpath).duration_sec - r0
            else:
                rdur = s1 - s0
            if rdur <= 0:
                raise QAToolError(f"window {wi}: result range must be > 0")
            scale = rdur / (s1 - s0)
        tag = w.get("label", f"w{wi}")
        for st in _sample_times(s0, s1, fractions=fractions, every_sec=every_sec):
            rt = r0 + (st - s0) * scale
            a, b = _grab_frame(source, st), _grab_frame(rpath, rt)
            diff = None
            if a is not None and b is not None:
                ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), _SHEET_DIFF_DIMS)
                gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), _SHEET_DIFF_DIMS)
                diff = round(_mad(ga, gb), 1)
            if out_path is not None:
                cells += [_cell(a, f"S{tag} {st:.2f}s", cw, ch),
                          _cell(b, f"R{tag} {rt:.2f}s", cw, ch)]
            samples.append({"window": tag, "source_t": round(st, 3),
                            "result_t": round(rt, 3), "diff": diff})
    if out_path is None:
        return {"samples": samples}
    gw, gh = _write_grid(cells, columns * 2, out_path)
    return {"out": out_path, "width": gw, "height": gh, "samples": samples}


_BAND_COLOURS = ((0, 255, 0), (0, 200, 255), (255, 0, 255))  # BGR: green, amber, magenta


def _draw_bands(cell: np.ndarray, bands: Sequence[Sequence[float]]) -> np.ndarray:
    """Horizontal lines at each band's top and bottom (fractions of the
    height); the first band green, the next ones amber / magenta."""
    h, w = cell.shape[:2]
    for i, band in enumerate(bands):
        colour = _BAND_COLOURS[min(i, len(_BAND_COLOURS) - 1)]
        for y in band[:2]:
            yy = min(h - 1, max(0, int(round(float(y) * h))))
            cv2.line(cell, (0, yy), (w - 1, yy), colour, 2 if i == 0 else 1)
    return cell


def contact_sheet(path: str, out_path: str, *, every_sec: Optional[float] = 3.0,
                  times: Optional[Sequence[float]] = None, cell_width: int = 200,
                  columns: int = 6, bands: Optional[Sequence[Sequence[float]]] = None) -> dict:
    """Frames of one video (intake look at captions/risk zones, seams).
    *bands* (``[[y0, y1], ...]`` fractions) are drawn on every frame: the
    first in green (e.g. a located caption band), the others amber/magenta."""
    info = _probe(path)
    if times is None:
        every = _finite(every_sec, "every_sec", positive=True)
        times = _sample_times(0.0, info.duration_sec, fractions=None, every_sec=every)
    times = [_finite(t, "time", minimum=0) for t in times]
    cw, ch = _cell_dims(path, cell_width)
    cells = [_cell(_grab_frame(path, t), f"{t:.2f}s", cw, ch) for t in times]
    if bands:
        cells = [_draw_bands(c, bands) for c in cells]
    gw, gh = _write_grid(cells, max(1, int(columns)), out_path)
    return {"out": out_path, "width": gw, "height": gh,
            "times": [round(t, 3) for t in times]}


def location_sheet(path: str, out_path: str, location: dict, *,
                   other_bands: Sequence[Sequence[float]] = (), max_frames: int = 6,
                   cell_width: int = 200) -> Optional[dict]:
    """Contact sheet for a :func:`locate_captions` result: frames at the
    candidate's own hit times (spread over the video) with the candidate band
    in green and *other_bands* (e.g. the configured band, competing text) in
    amber/magenta. None when there is no candidate."""
    band = location.get("candidate_band_y")
    if not band:
        return None
    positions = location.get("positions") or []
    members = [i for i in location.get("members") or [] if i < len(positions)]
    competing = [i for i in location.get("competing") or [] if i < len(positions)]
    times: list[float] = []
    for i in members:
        times += positions[i].get("sample_times") or []
    times = _spread(sorted(set(times)), max(1, int(max_frames)))
    # The competitors' own moments too: a held caption track elsewhere may
    # never be on screen at the candidate's times.
    extra: list[float] = []
    for i in competing:
        extra += positions[i].get("sample_times") or []
    extra = _spread(sorted(set(extra) - set(times)), max(0, int(max_frames) // 2))
    times = sorted(times + extra)
    if not times:
        return None
    bands = [band, *other_bands] + [positions[i]["band_y"] for i in competing
                                    if positions[i].get("band_y")]
    return contact_sheet(path, out_path, times=times, cell_width=cell_width,
                         columns=min(6, len(times)), bands=bands)


# ---------------------------------------------------------------------------
# 4. Captions from the script
# ---------------------------------------------------------------------------

# "[7.2–13.9] WOMAN (off-face cue …): text" -- the timed dialogue lines of an
# AICP-335 localisation prompt (times in seconds from the start of the clip).
_DIALOGUE_RE = re.compile(
    r"^\s*\[\s*(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)\s*\]\s*"
    r"([^:()\[\]]+?)\s*(\([^()]*\))?\s*:\s*(\S.*?)\s*$")


_SCRIPT_TIME_SLACK = 0.05


@dataclass
class DialogueLine:
    start: float
    end: float
    speaker: str
    text: str


def parse_dialogue(prompt: str) -> list[DialogueLine]:
    """Timed ``[t0–t1] SPEAKER (cue): text`` lines of a segment prompt.

    Lines without a speaker colon (``[6.9–8.2] Nobody speaks. …``) and any
    other prompt text are ignored; the parenthesised cue is dropped.
    """
    out = []
    for raw in (prompt or "").splitlines():
        m = _DIALOGUE_RE.match(raw)
        if not m:
            continue
        out.append(DialogueLine(start=float(m.group(1)), end=float(m.group(2)),
                                speaker=m.group(3).strip(), text=m.group(5)))
    return out


def captions_from_script(segments: Sequence[dict], *, gap_sec: float = 0.05,
                         decimals: int = 2) -> list[dict]:
    """Caption lines on the stitched final's timeline from the segment script.

    *segments* are in final order, one entry per piece of the final (keep
    pieces included, with no lines), each::

        {"source_start": 26.0333, "source_end": 35.9,   # the window it was cut from
         "clip_duration": 9.728 | "clip": "/path/result_0002.mp4",
         "prompt": "...[0.2–5.0] WOMAN: ..."           # or
         "lines": [{"start": 0.2, "end": 5.0, "text": "..."}],   # clip-relative
         "offset": 22.064}                              # optional, default = sum of previous clip durations

    Script times refer to the source window; each is scaled by
    ``clip_duration / (source_end - source_start)`` (the generated clip's
    real length), shifted by the clip's offset in the final, and each end is
    pulled in by *gap_sec* so consecutive lines never touch. Reproduces the
    Polish captions of run 6c4a9818 (``caption_lines_script.json``).
    """
    gap_sec = _finite(gap_sec, "gap_sec", minimum=0)
    out: list[dict] = []
    cursor = 0.0
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise QAToolError(f"segments[{i}] must be an object")
        s0 = _finite(seg.get("source_start"), f"segments[{i}].source_start", minimum=0)
        s1 = _finite(seg.get("source_end"), f"segments[{i}].source_end")
        if s1 <= s0:
            raise QAToolError(f"segments[{i}]: source_end must be > source_start")
        if seg.get("clip_duration") is not None:
            clip_dur = _finite(seg["clip_duration"], f"segments[{i}].clip_duration", positive=True)
        elif seg.get("clip"):
            clip_dur = _probe(str(seg["clip"])).duration_sec
        else:
            clip_dur = s1 - s0
        offset = cursor if seg.get("offset") is None else _finite(
            seg["offset"], f"segments[{i}].offset", minimum=0)
        scale = clip_dur / (s1 - s0)
        if seg.get("lines") is not None:
            if not isinstance(seg["lines"], list):
                raise QAToolError(f"segments[{i}].lines must be an array")
            script = []
            for j, ln in enumerate(seg["lines"]):
                if not isinstance(ln, dict) or not isinstance(ln.get("text"), str):
                    raise QAToolError(f"segments[{i}].lines[{j}] needs start/end/text")
                script.append(DialogueLine(
                    start=_finite(ln.get("start"), f"segments[{i}].lines[{j}].start", minimum=0),
                    end=_finite(ln.get("end"), f"segments[{i}].lines[{j}].end", minimum=0),
                    speaker=str(ln.get("speaker", "")), text=ln["text"]))
        else:
            prompt = seg.get("prompt")
            if prompt is not None and not isinstance(prompt, str):
                raise QAToolError(f"segments[{i}].prompt must be a string")
            script = parse_dialogue(prompt or "")
        for j, ln in enumerate(script):
            if ln.end < ln.start:
                raise QAToolError(f"segments[{i}] line {j}: end is before start")
            text = " ".join(ln.text.split())
            if not text:
                continue
            # Prompt times are printed to 0.1 s, so a time up to half a step
            # past the window end is rounding, not out of range.
            limit = (s1 - s0) + _SCRIPT_TIME_SLACK
            t0 = min(max(ln.start, 0.0), limit) * scale
            t1 = min(max(ln.end, 0.0), limit) * scale
            start = offset + t0
            end = min(offset + t1 - gap_sec, offset + clip_dur)
            if end <= start:
                end = min(offset + t1, offset + clip_dur)
            if end <= start:
                continue
            out.append({"start": round(start, decimals), "end": round(end, decimals),
                        "text": text})
        cursor = offset + clip_dur
    out.sort(key=lambda ln: ln["start"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _dump(obj) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


def _load_json_arg(value: str):
    """Inline JSON (starts with ``{``) or a path to a JSON file."""
    text = value
    if not value.lstrip().startswith("{"):
        with open(value, encoding="utf-8") as fh:
            text = fh.read()
    return json.loads(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.qa_tools", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("captions", help="burned-caption detector (10 fps)")
    c.add_argument("video")
    c.add_argument("--calibrate", metavar="SOURCE")
    c.add_argument("--calibrate-start", type=float)
    c.add_argument("--calibrate-duration", type=float)
    c.add_argument("--fps", type=float, default=10.0)
    c.add_argument("--offset", type=float, default=0.0)
    c.add_argument("--start", type=float)
    c.add_argument("--duration", type=float)
    c.add_argument("--threshold", type=int,
                   help="override the config's hit threshold (default %d)" % CAPTION_HIT_THRESHOLD)
    c.add_argument("--project", metavar="PID",
                   help="use this project's caption_detector config (read from the DB)")
    c.add_argument("--config", metavar="JSON",
                   help="caption detector config: a JSON object or a path to one; "
                        "replaces the --project config")
    c.add_argument("--scores", action="store_true", help="include every frame score")

    a = sub.add_parser("align", help="per-point frame offset vs the source (one timeline)")
    a.add_argument("source")
    a.add_argument("result")
    a.add_argument("--source-start", type=float, default=0.0)
    a.add_argument("--result-start", type=float, default=0.0)
    a.add_argument("--duration", type=float)
    a.add_argument("--step-frames", type=int, default=15)
    a.add_argument("--max-offset", type=int, default=9)
    a.add_argument("--tolerance", type=int, default=1)
    a.add_argument("--points", action="store_true", help="include every checked point")

    m = sub.add_parser("match", help="best source time for sampled moments of a clip")
    m.add_argument("source")
    m.add_argument("result")
    m.add_argument("--source-start", type=float, required=True)
    m.add_argument("--duration", type=float, required=True)
    m.add_argument("--every", type=float, default=0.5)

    s = sub.add_parser("sheet", help="source vs result frame sheet (JPEG)")
    s.add_argument("source")
    s.add_argument("result")
    s.add_argument("out")
    s.add_argument("--windows", metavar="JSON", help="file with a list of windows")
    s.add_argument("--every", type=float, help="one pair every N seconds")
    s.add_argument("--fractions", default="0.1,0.5,0.9")
    s.add_argument("--time-map", choices=("offset", "scale"), default="offset")
    s.add_argument("--cell-width", type=int, default=200)
    s.add_argument("--columns", type=int, default=3, help="pairs per row")

    k = sub.add_parser("contact", help="contact sheet of one video (JPEG)")
    k.add_argument("video")
    k.add_argument("out")
    k.add_argument("--every", type=float, default=3.0)
    k.add_argument("--times", help="comma-separated seconds instead of --every")
    k.add_argument("--cell-width", type=int, default=200)
    k.add_argument("--columns", type=int, default=6)

    lo = sub.add_parser("locate", help="where the captions sit (full frame, 2 fps)")
    lo.add_argument("video")
    lo.add_argument("--project", metavar="PID",
                    help="use this project's caption_detector pixel rule (its band is ignored)")
    lo.add_argument("--config", metavar="JSON", help="caption detector config (JSON or path)")
    lo.add_argument("--fps", type=float, default=LOCATE_FPS)
    lo.add_argument("--sheet", metavar="OUT.jpg", help="also write a contact sheet with the band")

    t = sub.add_parser("script-captions", help="caption lines from the segment script")
    t.add_argument("spec", help='JSON file: {"segments": [...], "gap_sec"?: 0.05}')

    args = ap.parse_args(argv)
    try:
        if args.cmd == "captions":
            config = None
            if args.project:
                config = project_caption_config(args.project)
            if args.config:
                config = _load_json_arg(args.config)
            scan = detect_captions(
                args.video, calibrate=args.calibrate, calibrate_start=args.calibrate_start,
                calibrate_duration=args.calibrate_duration, fps=args.fps,
                offset=args.offset, start=args.start, duration=args.duration,
                threshold=args.threshold, config=caption_config(config))
            out = scan.to_dict(include_scores=args.scores)
            out["geometry"] = caption_geometry(scan)
            _dump(out)
        elif args.cmd == "align":
            res = alignment_check(
                args.source, args.result, source_start=args.source_start,
                result_start=args.result_start, duration=args.duration,
                step_frames=args.step_frames, max_offset=args.max_offset,
                tolerance=args.tolerance)
            pts = res.pop("_points")
            if args.points:
                res["all_points"] = pts
            _dump(res)
        elif args.cmd == "match":
            _dump(match_times(args.source, args.result, source_start=args.source_start,
                              duration=args.duration, every_sec=args.every))
        elif args.cmd == "sheet":
            windows = None
            if args.windows:
                with open(args.windows, encoding="utf-8") as fh:
                    windows = json.load(fh)
            fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
            _dump(comparison_sheet(args.source, args.result, args.out, windows=windows,
                                   fractions=fractions, every_sec=args.every,
                                   time_map=args.time_map, cell_width=args.cell_width,
                                   columns=args.columns))
        elif args.cmd == "contact":
            times = [float(x) for x in args.times.split(",")] if args.times else None
            _dump(contact_sheet(args.video, args.out, every_sec=args.every, times=times,
                                cell_width=args.cell_width, columns=args.columns))
        elif args.cmd == "locate":
            config = project_caption_config(args.project) if args.project else None
            if args.config:
                config = _load_json_arg(args.config)
            loc = locate_captions(args.video, config=config, fps=args.fps)
            if args.sheet:
                loc["sheet"] = (location_sheet(args.video, args.sheet, loc) or {}).get("out")
            if loc["candidate_band_y"]:
                loc["hint"] = ("project page -> QA caption detector -> Band y %.2f to %.2f "
                               "(then Save)" % tuple(loc["candidate_band_y"]))
            _dump(loc)
        elif args.cmd == "script-captions":
            with open(args.spec, encoding="utf-8") as fh:
                spec = json.load(fh)
            if not isinstance(spec, dict) or not isinstance(spec.get("segments"), list):
                raise QAToolError('spec must be {"segments": [...]}')
            lines = captions_from_script(spec["segments"], gap_sec=spec.get("gap_sec", 0.05))
            if not lines:
                # An empty body would wipe the caption's lines through PATCH.
                raise QAToolError("the spec produced no caption lines")
            _dump({"lines": lines})
    except (QAToolError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
