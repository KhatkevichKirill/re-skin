"""Pure helpers of the "Rebuild from reference" project type.

docs/rebuild-from-reference-design.md. Everything here is deterministic and
free of DB / network / provider calls, so every rule is unit-testable:

* shot detection wrapper (:func:`detect_shots`, the only function that reads
  frames — through ``scene.detect_scenes``) and its normalisation;
* short-shot folding into effective-shot groups (:func:`effective_shots`, §5.1.3);
* chunking whole groups (:func:`build_chunks`, §5.1.4) and the project-level
  segment plan with optional brand-card keeps (:func:`plan_segments`, §5.1.5);
* the shots inside a chunk (:func:`chunk_shot_pieces`, §6.2), the per-chunk
  sheet pick and image list (§6.1–6.2);
* prompt rendering per model (:func:`render_chunk_prompt`, §6.3);
* the duration request (:func:`request_sec_for_chunk`, §6.4);
* cumulative frame quantisation and the fit decision (§6.4), plus the ffmpeg
  filter strings the fit / keep cut use;
* cut-timing QA maths (:func:`cut_timing_report`, §6.5);
* the trailing end-card rule (§6.1).

Estimates live in ``app/spend.py`` (``images_only_video_credits``,
``storyboard_credits``, ``keyframes_credits``, ``estimate_images_only_run_credits``).

Paid provider calls are NEVER made from this module; see
``app/rebuild_paid_call.py`` for the paid-call protocol.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from . import ai_models

# --- Constants (design §5.1, §6.4, §6.5; decided 2026-11-03, Q5) -------------

# PySceneDetect minimum shot length for rebuild analysis. The pilot's 54-cut
# list for 81d0afc0 came from the detector default; 0.4s keeps flash frames
# from becoming shots of their own.
REBUILD_SHOT_MIN_SEC = 0.4
# Shots shorter than this are folded into a neighbour's group: the model merges
# them anyway (pilot defect 4: the only cut errors > 0.3s were 0.67s shots).
REBUILD_SHOT_FOLD_SEC = 0.7
# Longest chunk: Seedance 2.5 intermittently truncates to exactly 14.72s, so a
# chunk <= 14.72s requested as ceil(...) = 15s survives a truncation. Same cap
# for Kling (Q5).
REBUILD_CHUNK_MAX_SEC = 14.72
# Most effective shots (groups) per chunk: Seedance takes 9 image references
# (8 keyframes + the character sheet); Kling takes 10.
REBUILD_CHUNK_MAX_SHOTS = 8
# Cut-timing QA flags a chunk when any planned cut misses by more than this.
CUT_QA_FLAG_SEC = 0.5
# A detected cut further than this from a planned cut does not match it.
CUT_QA_MATCH_WINDOW_SEC = 1.5
# The fit holds the last frame for at most this much missing output.
FIT_SHORT_TOLERANCE_SEC = 0.1

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Shots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    """One detected shot (source timeline). *index* is its position."""

    index: int
    start: float
    end: float
    force_anchor: bool = False
    is_brand_card: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def normalize_shot_ranges(
    ranges: Sequence[tuple[float, float]], duration_sec: float,
) -> list[tuple[float, float]]:
    """Contiguous ranges covering exactly ``[0, duration]``.

    PySceneDetect's last scene can end a frame or two before the probed
    duration, and "no cuts" comes back as ``[]``; both are repaired here: the
    first range starts at 0, each range starts where the previous ended, the
    last one ends at *duration*, and empty ranges are dropped.
    """
    duration = float(duration_sec)
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration_sec!r}")
    cuts = sorted(
        float(s) for s, _e in ranges if _EPS < float(s) < duration - _EPS
    )
    bounds = [0.0]
    for c in cuts:
        if c - bounds[-1] > _EPS:
            bounds.append(c)
    bounds.append(duration)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def detect_shots(
    video_path: str, *, fps: Optional[float], duration_sec: float,
    min_shot_sec: float = REBUILD_SHOT_MIN_SEC,
) -> list[tuple[float, float]]:
    """Fine shot detection for rebuild analysis (design §5.1.2)."""
    from . import scene as scene_mod

    detected = scene_mod.detect_scenes(video_path, fps=fps, min_scene_len_sec=min_shot_sec)
    return normalize_shot_ranges(detected, duration_sec)


# ---------------------------------------------------------------------------
# Effective shots (folding, §5.1.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotGroup:
    """One effective shot: contiguous detected shots sharing ONE keyframe.

    *anchor* is the index of the shot whose keyframe the group uses; *members*
    are every shot index in the group, in order (anchor included). *piece* is
    ``(k, n)`` (1-based) when an anchor longer than the cap was split into
    *n* equal pieces — each piece is its own group covering only its interval.
    """

    anchor: int
    members: tuple[int, ...]
    start: float
    end: float
    piece: Optional[tuple[int, int]] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def _close_buffer(buf: list[Shot]) -> ShotGroup:
    # Anchor = the longest shot; ties -> the earliest (max keeps the first).
    anchor = max(buf, key=lambda s: (s.duration, -s.index))
    return ShotGroup(
        anchor=anchor.index, members=tuple(s.index for s in buf),
        start=buf[0].start, end=buf[-1].end,
    )


def _split_anchor(shot: Shot, cap: float) -> list[ShotGroup]:
    n = max(1, math.ceil(shot.duration / cap - 1e-9))
    step = shot.duration / n
    out = []
    for k in range(n):
        start = shot.start + k * step
        end = shot.end if k == n - 1 else shot.start + (k + 1) * step
        out.append(ShotGroup(anchor=shot.index, members=(shot.index,), start=start,
                             end=end, piece=(k + 1, n) if n > 1 else None))
    return out


def effective_shots(
    shots: Sequence[Shot],
    *,
    fold_sec: float = REBUILD_SHOT_FOLD_SEC,
    cap_sec: float = REBUILD_CHUNK_MAX_SEC,
) -> list[ShotGroup]:
    """Partition *shots* (contiguous, in order) into effective-shot groups.

    One deterministic left-to-right pass (design §5.1.3):

    * a shot >= *fold_sec* (or ``force_anchor``) is an anchor and opens a
      group; a short shot waits in a buffer and joins the group of the NEXT
      anchor;
    * no group may exceed *cap_sec*: a buffer that would grow past the cap is
      closed as a group of its own (anchor = its longest shot); an anchor whose
      buffered shorts would push it past the cap stands alone and the shorts
      form their own group; an anchor alone longer than the cap is split into
      equal pieces <= the cap, each its own group sharing the anchor;
    * shorts left after the last anchor join the last group if it stays within
      the cap, else they form their own group(s);
    * a video with no anchor at all is grouped by the buffer rule alone (one
      group if it fits the cap).

    The groups cover the shots' span exactly, are each <= *cap_sec*, are never
    empty, and folding only ever points at the group's own anchor (no cycles).
    """
    if not shots:
        return []
    for a, b in zip(shots, shots[1:]):
        if abs(a.end - b.start) > 1e-4 or b.index != a.index + 1:
            raise ValueError("shots must be contiguous and consecutively indexed")
    cap = float(cap_sec)
    groups: list[ShotGroup] = []
    buf: list[Shot] = []

    def buf_len() -> float:
        return buf[-1].end - buf[0].start if buf else 0.0

    for shot in shots:
        is_anchor = shot.force_anchor or shot.duration >= fold_sec - _EPS
        if not is_anchor:
            if buf and buf_len() + shot.duration > cap + _EPS:
                groups.append(_close_buffer(buf))
                buf = []
            buf.append(shot)
            continue
        if shot.duration > cap + _EPS:
            if buf:
                groups.append(_close_buffer(buf))
                buf = []
            groups.extend(_split_anchor(shot, cap))
            continue
        if buf and buf_len() + shot.duration > cap + _EPS:
            groups.append(_close_buffer(buf))
            buf = []
        members = tuple(s.index for s in buf) + (shot.index,)
        start = buf[0].start if buf else shot.start
        groups.append(ShotGroup(anchor=shot.index, members=members, start=start,
                                end=shot.end))
        buf = []
    if buf:
        last = groups[-1] if groups else None
        if last is not None and buf[-1].end - last.start <= cap + _EPS:
            groups[-1] = ShotGroup(
                anchor=last.anchor, members=last.members + tuple(s.index for s in buf),
                start=last.start, end=buf[-1].end, piece=last.piece,
            )
        else:
            groups.append(_close_buffer(buf))
    return groups


def folded_into_map(groups: Iterable[ShotGroup]) -> dict[int, Optional[int]]:
    """Shot index -> the anchor index it is folded into (None for an anchor)."""
    out: dict[int, Optional[int]] = {}
    for g in groups:
        for m in g.members:
            out[m] = None if m == g.anchor else g.anchor
    return out


# ---------------------------------------------------------------------------
# Chunks (§5.1.4) and the project segment plan (§5.1.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """Consecutive whole groups generated by ONE video task (a swap segment)."""

    start: float
    end: float
    groups: tuple[ShotGroup, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_chunks(
    groups: Sequence[ShotGroup],
    *,
    target_sec: float = REBUILD_CHUNK_MAX_SEC,
    max_groups: int = REBUILD_CHUNK_MAX_SHOTS,
) -> list[Chunk]:
    """Greedy left-to-right merge of whole groups while the chunk stays
    <= *target_sec* AND <= *max_groups* groups (design §5.1.4). Chunking never
    splits a group; every group is already <= the cap. There is no short-tail
    rebalance: a chunk shorter than the model's minimum output is requested at
    that minimum and trimmed back by the fit."""
    if max_groups < 1:
        raise ValueError("max_groups must be >= 1")
    chunks: list[list[ShotGroup]] = []
    for g in groups:
        if g.duration > target_sec + _EPS:
            raise ValueError(
                f"group [{g.start:.3f},{g.end:.3f}] is longer than the chunk cap "
                f"{target_sec}s — fold with the same cap first"
            )
        cur = chunks[-1] if chunks else None
        if (
            cur is not None
            and len(cur) < max_groups
            and g.end - cur[0].start <= target_sec + _EPS
        ):
            cur.append(g)
        else:
            chunks.append([g])
    return [Chunk(start=c[0].start, end=c[-1].end, groups=tuple(c)) for c in chunks]


@dataclass(frozen=True)
class SegmentPlan:
    """One SegmentDef to write: ``swap`` = a chunk, ``keep`` = original footage."""

    start: float
    end: float
    action: str
    groups: tuple[ShotGroup, ...] = field(default_factory=tuple)


def plan_segments(
    shots: Sequence[Shot],
    *,
    keep_brand_cards: bool = False,
    fold_sec: float = REBUILD_SHOT_FOLD_SEC,
    cap_sec: float = REBUILD_CHUNK_MAX_SEC,
    max_groups: int = REBUILD_CHUNK_MAX_SHOTS,
) -> tuple[list[ShotGroup], list[SegmentPlan]]:
    """Fold *shots*, then cut the project's segments.

    Without *keep_brand_cards* every chunk is a ``swap`` segment (analysis,
    §5.1.5). With it (the storyboard card's "Keep brand-card shots as original
    footage"), every group whose ANCHOR shot ``is_brand_card`` becomes original
    footage: consecutive such groups merge into one ``keep`` segment, and each
    run of other groups is chunked as usual. Free and deterministic; the caller
    refuses it while a run is active (as every segment edit).
    """
    groups = effective_shots(shots, fold_sec=fold_sec, cap_sec=cap_sec)
    by_index = {s.index: s for s in shots}
    out: list[SegmentPlan] = []
    run: list[ShotGroup] = []

    def flush_swaps():
        for c in build_chunks(run, target_sec=cap_sec, max_groups=max_groups):
            out.append(SegmentPlan(c.start, c.end, "swap", c.groups))
        run.clear()

    for g in groups:
        if keep_brand_cards and by_index[g.anchor].is_brand_card:
            flush_swaps()
            if out and out[-1].action == "keep":
                prev = out[-1]
                out[-1] = SegmentPlan(prev.start, g.end, "keep", prev.groups + (g,))
            else:
                out.append(SegmentPlan(g.start, g.end, "keep", (g,)))
        else:
            run.append(g)
    flush_swaps()
    return groups, out


# ---------------------------------------------------------------------------
# Shots inside a chunk (§6.2), sheet pick, image list (§6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotSpan:
    """A shot as the run launch sees it: id, source times, and the anchor it is
    folded into (``folded_into`` = None for an anchor)."""

    shot_id: str
    start: float
    end: float
    folded_into: Optional[str] = None


@dataclass(frozen=True)
class ChunkPiece:
    """One shot line of a chunk: the keyframe shot (*shot_id*, always an
    anchor) and its interval RELATIVE to the chunk start."""

    shot_id: str
    rel_start: float
    rel_end: float

    @property
    def duration(self) -> float:
        return self.rel_end - self.rel_start


def chunk_shot_pieces(
    chunk_start: float,
    chunk_end: float,
    shots: Sequence[ShotSpan],
    *,
    fold_sec: float = REBUILD_SHOT_FOLD_SEC,
) -> list[ChunkPiece]:
    """The shot lines of one chunk (design §6.2).

    Each shot intersecting the chunk contributes ``[max(start, chunk.start),
    min(end, chunk.end)]``; a folded shot contributes its time to its anchor
    (consecutive pieces of the same anchor merge); a piece shorter than
    *fold_sec* created by an operator boundary is merged into its neighbour
    (the following one, else the preceding one). A long shot split across two
    chunks appears in both, with the same keyframe. The result is contiguous
    and covers ``[0, chunk_end - chunk_start]``.
    """
    cs, ce = float(chunk_start), float(chunk_end)
    if ce - cs <= _EPS:
        raise ValueError("empty chunk")
    raw: list[list] = []
    for sh in sorted(shots, key=lambda s: s.start):
        s, e = max(sh.start, cs), min(sh.end, ce)
        if e - s <= _EPS:
            continue
        key = sh.folded_into or sh.shot_id
        if raw and raw[-1][0] == key and abs(raw[-1][2] - s) <= 1e-4:
            raw[-1][2] = e
        else:
            raw.append([key, s, e])
    if not raw:
        raise ValueError(f"no shot intersects the chunk [{cs:.3f},{ce:.3f}]")
    # Close any gap / overhang so the pieces tile the chunk exactly.
    raw[0][1] = cs
    raw[-1][2] = ce
    for a, b in zip(raw, raw[1:]):
        b[1] = a[2]
    # Merge short pieces into a neighbour, left to right, until none is short
    # (a single piece is kept whatever its length).
    changed = True
    while changed and len(raw) > 1:
        changed = False
        for i, (key, s, e) in enumerate(raw):
            if e - s < fold_sec - _EPS:
                if i + 1 < len(raw):
                    raw[i + 1][1] = s
                else:
                    raw[i - 1][2] = e
                del raw[i]
                # Re-merge equal neighbours created by the deletion.
                j = 0
                while j + 1 < len(raw):
                    if raw[j][0] == raw[j + 1][0]:
                        raw[j][2] = raw[j + 1][2]
                        del raw[j + 1]
                    else:
                        j += 1
                changed = True
                break
    return [ChunkPiece(k, round(s - cs, 6), round(e - cs, 6)) for k, s, e in raw]


def chunk_keyframe_ids(pieces: Sequence[ChunkPiece]) -> list[str]:
    """The chunk's keyframe shots in order, deduplicated (image refs 1..k)."""
    out: list[str] = []
    for p in pieces:
        if p.shot_id not in out:
            out.append(p.shot_id)
    return out


def check_chunk_image_count(model: str, pieces: Sequence[ChunkPiece]) -> None:
    """Raise ``ValueError`` ("split this chunk") when the chunk needs more
    keyframes than *model* can take next to the sheet (``max_reference_images
    - 1``)."""
    spec = ai_models.AI_MODELS[model]
    n = len(chunk_keyframe_ids(pieces))
    if n > spec.max_reference_images - 1:
        raise ValueError(
            f"this chunk has {n} shots; {spec.label} takes at most "
            f"{spec.max_reference_images - 1} keyframes plus the character sheet "
            "— split this chunk"
        )


def sheet_for_chunk(sheet_keys_in_order: Sequence[Optional[str]]) -> Optional[str]:
    """The character sheet a chunk uses (design §6.1): the ``sheet_key`` held
    by most of its unfolded shots; ties -> the earliest shot's key. ``None``
    keys are ignored; ``None`` when no shot has one."""
    keys = [k for k in sheet_keys_in_order if k]
    if not keys:
        return None
    counts = Counter(keys)
    best = max(counts.values())
    return next(k for k in keys if counts[k] == best)


# ---------------------------------------------------------------------------
# Prompt (§6.3) and duration request (§6.4)
# ---------------------------------------------------------------------------


def request_sec_for_chunk(model: str, chunk_sec: float) -> int:
    """``clamp(ceil(chunk_len - 0.02), images_only_min_sec, images_only_max_sec)``."""
    return ai_models.images_only_duration(model, chunk_sec)


def _orientation(aspect_ratio: str) -> str:
    return {"9:16": "vertical", "16:9": "horizontal", "1:1": "square"}.get(aspect_ratio, "")


def _clean(text: Optional[str]) -> str:
    return " ".join(str(text or "").split()).rstrip(".")


def render_chunk_prompt(
    chunk: dict,
    storyboard: dict,
    model: str,
    *,
    aspect_ratio: str = "9:16",
) -> str:
    """Deterministic prompt of one chunk (design §6.3).

    *chunk*: ``{"request_sec": int, "shots": [{"shot_id", "rel_start",
    "rel_end", "framing", "camera", "action", "characters": [ids]}, ...]}`` —
    the shots in order, one line per piece, keyframe refs numbered by first
    appearance (the same order as :func:`chunk_keyframe_ids`), the sheet last.
    *storyboard*: ``{"style": str, "characters": [{"id", "name_hint", "look",
    "outfit"}]}``. The last shot's end is stretched to ``request_sec`` (the
    model fills the extra tail, which the fit trims).
    """
    shots = list(chunk.get("shots") or [])
    if not shots:
        raise ValueError("a chunk prompt needs at least one shot")
    request_sec = int(chunk["request_sec"])
    order: list[str] = []
    for sh in shots:
        if sh["shot_id"] not in order:
            order.append(sh["shot_id"])
    sheet_ref = ai_models.image_ref(model, len(order) + 1)
    style = _clean(storyboard.get("style")) or "animated film"
    orient = _orientation(aspect_ratio)
    shape = f"{orient} {aspect_ratio}".strip()

    present: list[str] = []
    for sh in shots:
        for cid in sh.get("characters") or []:
            if cid not in present:
                present.append(cid)
    clauses = []
    for c in storyboard.get("characters") or []:
        if c.get("id") not in present:
            continue
        name = _clean(c.get("name_hint")) or str(c.get("id"))
        details = "; ".join(x for x in (_clean(c.get("look")), _clean(c.get("outfit"))) if x)
        clauses.append(f"{name} ({details})" if details else name)
    k = len(shots)
    head = (
        f"A {request_sec}-second {shape} {style}. "
        + (
            f"It consists of {k} shots joined by hard cuts, in exactly this order "
            "and timing. Each shot starts from the composition of its keyframe image. "
            if k > 1 else
            "It is one continuous shot that starts from the composition of its "
            "keyframe image. "
        )
        + "The characters must look exactly the same in every shot, as in the "
        f"character sheet {sheet_ref} and the keyframes"
        + (": " + ", ".join(clauses) + "." if clauses else ".")
    )
    lines = [head]
    for i, sh in enumerate(shots):
        start = float(sh["rel_start"])
        end = float(request_sec) if i == k - 1 else float(sh["rel_end"])
        ref = ai_models.image_ref(model, order.index(sh["shot_id"]) + 1)
        desc = [x for x in (_clean(sh.get("framing")), _clean(sh.get("camera"))) if x]
        action = _clean(sh.get("action"))
        body = ", ".join(desc)
        if action:
            body = f"{body}: {action}" if body else action
        lines.append(
            f"Shot {i + 1} ({start:.1f}-{end:.1f} s), starts from {ref}"
            + (f": {body}." if body else ".")
        )
    lines.append(
        "No on-screen text, no captions, no subtitles, no logos. No dialogue is "
        "needed; ambient sound only."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame grid, fit and keep cut (§6.4)
# ---------------------------------------------------------------------------


def frame_index(t: float, fps: float) -> int:
    """``F(t) = round(t x fps)`` on the stitch's output grid (half up; a
    1e-9 slack absorbs float noise such as 1.7 x 30 = 50.99999999)."""
    if not fps or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return int(math.floor(float(t) * float(fps) + 0.5 + 1e-9))


def chunk_frames(start: float, end: float, fps: float) -> int:
    """Frames allocated to ``[start, end]``: ``F(end) - F(start)``. Quantised
    CUMULATIVELY, so adjacent segments' frames sum to ``F(duration)`` and
    rounding never accumulates across chunks."""
    return frame_index(end, fps) - frame_index(start, fps)


@dataclass(frozen=True)
class FitPlan:
    """What the fit does with a raw result of *raw_frames* frames (on the
    output grid) for a chunk of *chunk_frames* frames."""

    ok: bool
    take_frames: int
    padded_frames: int
    reason: str = ""


def fit_plan(raw_frames: int, chunk_frames_: int, fps: float) -> FitPlan:
    """Fit decision (design §6.4): take the first ``chunk_frames`` frames;
    a raw result short by at most ``round(0.1 x fps)`` frames is padded by
    holding its last frame; shorter than that the attempt fails ("result too
    short") — never an automatic resubmit."""
    if chunk_frames_ <= 0:
        raise ValueError("chunk_frames must be positive")
    raw = max(0, int(raw_frames))
    if raw >= chunk_frames_:
        return FitPlan(True, chunk_frames_, 0)
    tolerance = int(round(FIT_SHORT_TOLERANCE_SEC * float(fps)))
    if raw >= chunk_frames_ - tolerance and raw > 0:
        return FitPlan(True, raw, chunk_frames_ - raw)
    return FitPlan(
        False, raw, 0,
        f"result too short ({raw / fps:.2f} s < chunk {chunk_frames_ / fps:.2f} s)",
    )


def _fps_expr(fps: float) -> str:
    """An exact ffmpeg rate for *fps* (29.97 -> 30000/1001)."""
    f = float(fps)
    for num, den in ((24000, 1001), (30000, 1001), (60000, 1001)):
        if abs(f - num / den) < 1e-3:
            return f"{num}/{den}"
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return repr(f)


def fit_filter(plan: FitPlan, fps: float) -> str:
    """``-vf`` of the fit: resample to the output grid, hold the last frame
    for ``padded_frames``, keep exactly ``take_frames + padded_frames`` frames.

    The pad goes BEFORE the trim: ffmpeg 7.1 drops a ``tpad`` placed after a
    ``trim`` that ended the stream (verified: ``trim,tpad=stop=3`` gave 60
    frames, ``tpad=stop=3,trim`` gave 63)."""
    if not plan.ok:
        raise ValueError("cannot build a fit filter for a failed fit plan")
    total = plan.take_frames + plan.padded_frames
    vf = f"fps={_fps_expr(fps)}"
    if plan.padded_frames:
        vf += f",tpad=stop_mode=clone:stop={plan.padded_frames}"
    return vf + f",trim=end_frame={total},setpts=PTS-STARTPTS"


def _encode_args(fps: float) -> list[str]:
    # -fps_mode cfr -r: with the mp4 muxer's default (vfr) ffmpeg 7.1 drops a
    # cloned tail frame (verified: the filter emitted 61 frames, the file held
    # 60); constant frame rate output keeps every frame the filter emits.
    return ["-an", "-fps_mode", "cfr", "-r", _fps_expr(fps), "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p"]


def fit_ffmpeg_args(src: str, dst: str, plan: FitPlan, fps: float) -> list[str]:
    """Full ffmpeg argv of the fit (raw -> fitted, video only, CRF 18)."""
    return ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", fit_filter(plan, fps),
            *_encode_args(fps), dst]


def keep_cut_ffmpeg_args(src: str, dst: str, start_frame: int, end_frame: int,
                         fps: float) -> list[str]:
    """Full ffmpeg argv of a frame-exact keep cut (video only; the stitch
    muxes the soundtrack)."""
    return ["ffmpeg", "-v", "error", "-y", "-i", src,
            "-vf", keep_cut_filter(start_frame, end_frame, fps), *_encode_args(fps), dst]


def keep_cut_filter(start_frame: int, end_frame: int, fps: float) -> str:
    """``-vf`` that cuts frames ``[start_frame, end_frame)`` of the source
    AFTER conversion to the output grid (frame-exact keep segments of an
    images-only run)."""
    if end_frame <= start_frame or start_frame < 0:
        raise ValueError(f"bad frame range [{start_frame}, {end_frame})")
    return (
        f"fps={_fps_expr(fps)},trim=start_frame={start_frame}:end_frame={end_frame},"
        "setpts=PTS-STARTPTS"
    )


# ---------------------------------------------------------------------------
# Cut-timing QA (§6.5)
# ---------------------------------------------------------------------------


def planned_cuts(pieces: Sequence[ChunkPiece]) -> list[float]:
    """Internal cut times of a chunk (relative): every piece start but the first."""
    return [p.rel_start for p in pieces[1:]]


def cut_timing_report(
    planned: Sequence[float],
    detected: Sequence[float],
    *,
    flag_sec: float = CUT_QA_FLAG_SEC,
    match_window_sec: float = CUT_QA_MATCH_WINDOW_SEC,
) -> dict:
    """Compare planned internal cuts with cuts detected on the fitted clip.

    Order-preserving greedy match: each planned cut takes the nearest
    still-unused detected cut after the previously matched one, within
    *match_window_sec*. Returns ``{"cuts": [{"planned", "detected", "delta"}],
    "missing": [planned...], "extra": [detected...], "max_abs_delta",
    "flagged"}``; flagged when any |delta| > *flag_sec* or the counts differ
    (a missing or an extra cut).
    """
    det = sorted(float(d) for d in detected)
    used = [False] * len(det)
    cuts, missing = [], []
    lo = 0
    for p in sorted(float(x) for x in planned):
        best = None
        for j in range(lo, len(det)):
            if used[j]:
                continue
            d = abs(det[j] - p)
            if d <= match_window_sec + _EPS and (best is None or d < abs(det[best] - p)):
                best = j
            if det[j] > p + match_window_sec:
                break
        if best is None:
            missing.append(round(p, 3))
            continue
        used[best] = True
        lo = best + 1
        cuts.append({
            "planned": round(p, 3), "detected": round(det[best], 3),
            "delta": round(det[best] - p, 3),
        })
    extra = [round(d, 3) for j, d in enumerate(det) if not used[j]]
    max_abs = max((abs(c["delta"]) for c in cuts), default=0.0)
    flagged = bool(missing or extra or max_abs > flag_sec + _EPS)
    return {"cuts": cuts, "missing": missing, "extra": extra,
            "max_abs_delta": round(max_abs, 3), "flagged": flagged}


# ---------------------------------------------------------------------------
# Trailing end card (§6.1, Q2)
# ---------------------------------------------------------------------------


def trailing_end_card(segments: Sequence[tuple[str, str, bool]]) -> list[str]:
    """Ids of the trailing end card: the maximal run of ``keep`` segments at
    the END of the timeline whose shots are all brand cards.

    *segments*: ``(segment_def_id, action, all_shots_brand_card)`` in timeline
    order. Middle keeps are never part of it. ``[]`` when the last segment is
    not such a keep.
    """
    out: list[str] = []
    for seg_id, action, all_brand in reversed(list(segments)):
        if action != "keep" or not all_brand:
            break
        out.append(seg_id)
    return list(reversed(out))


def default_stitch_plan_with_end_card(
    segment_ids: Sequence[str], end_card_ids: Sequence[str], library_clip_id: Optional[str],
) -> Optional[list[dict]]:
    """The default stitch plan of a rebuild run (§6.1): every segment in order
    up to the trailing end card, then the language-matched library clip in its
    place. ``None`` (= the default plan, every segment) when there is no
    trailing end card or no matching library clip."""
    if not end_card_ids or not library_clip_id:
        return None
    drop = set(end_card_ids)
    plan = [{"kind": "segment", "segment_def_id": sid} for sid in segment_ids if sid not in drop]
    plan.append({"kind": "library", "clip_id": library_clip_id})
    return plan
