"""Re-cut whole-video localisation segments at speech pauses (AICP-347).

In whole-video scope (AICP-340) Analyze cuts all-swap segments at scene cuts,
before the transcript exists, so a boundary can land inside a spoken line. A
line split across two generations is spoken by two separate Seedance takes and
sounds broken at the seam. This module moves the boundaries of an existing
partition off the transcript's lines, keeping every swap segment within the
whole-video target.

:func:`recut_at_speech_pauses` is PURE (no DB, no files, no frames): the
operator's "Re-cut at speech pauses" button (``POST
/api/v2/projects/{pid}/recut-at-pauses``) and the localisation autopilot
(AICP-365) both call it with what they already hold.

The rule
--------
* **Speech** is the union of the transcript's line intervals. Lines that
  overlap (an app voice over the speaker) merge into one block; lines that only
  touch (``end == next start``) leave a zero-width pause at the touch point.
* A boundary is **inside a line** when it is more than ``edge_tol_sec`` from
  both of that line's edges. The tolerance is the transcript's own timing
  precision (a video model's 0.1 s grid), so a boundary 0.1 s past a line's
  start is not treated as splitting it, and a scene cut that close to a pause
  counts as inside that pause.
* Each **pause** offers its *ideal* point — the scene cut inside it nearest
  its middle, else its midpoint — plus its edges and, when it is wide, points
  every :data:`PAUSE_GRID_SEC` across it, so a pause whose middle would push a
  neighbour over the target can still be used near one edge.
* Each boundary either moves to one of those points or stays where it is.
  The segment COUNT is preserved (actions, pre/post-roll and SegmentDef ids
  stay with their index; the run cost stays what the operator expected), and
  every swap segment — and the clip it sends the model, pre/post-roll
  applied — must be within ``[floor_sec, target_sec]``. A small dynamic programme over the boundaries minimises, in
  order: the number of boundaries left inside a line, then the total
  displacement plus ``OFF_IDEAL_WEIGHT`` x the distance from each pause's
  ideal point. A greedy per-boundary rule is not enough: on 46cc6443 the
  overlapping lines 9-11 form one 11.7 s block, and only a joint choice of
  the neighbouring boundaries keeps every segment under 14.72 s.
* A boundary that no pause can take while keeping its neighbours within the
  target stays where it is and is reported in ``warnings``. If even the
  current partition breaks the cap (a hand edit), nothing changes and the
  plan says why (``feasible`` is False).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# A boundary within this distance of a line's edge is not "inside" the line,
# and a scene cut within it of a pause counts as inside the pause. The
# transcription model times lines on a ~0.1 s grid (46cc6443's scene cuts sit
# 0.07-0.1 s from the nearest line edge), so anything finer is noise.
EDGE_TOL_SEC = 0.15

# Spacing of the extra candidate points across a wide pause.
PAUSE_GRID_SEC = 0.25

# Weight of "distance from the pause's ideal point" against displacement from
# the original boundary. > 1 so that, with every constraint satisfied, the
# ideal point (the scene cut, else the middle of the pause) beats an edge of
# the same pause even though the edge is nearer the original boundary.
OFF_IDEAL_WEIGHT = 2.0

# Candidate positions are rounded to the millisecond (well below a frame).
_ROUND = 3
_EPS = 1e-6


@dataclass(frozen=True)
class RecutBoundary:
    """One interior boundary (between segment ``index`` and ``index + 1``)."""

    index: int
    old_sec: float
    new_sec: float
    moved: bool
    # Where the new position comes from: "kept" (unchanged), "scene_cut",
    # "pause_middle" or "pause" (another point of the pause, chosen to keep a
    # neighbour within the target).
    source: str
    # Transcript line ids the boundary was / is still strictly inside.
    old_inside_line_ids: tuple = ()
    new_inside_line_ids: tuple = ()


@dataclass
class RecutPlan:
    ranges: list  # [(start, end)], same count as the input, contiguous
    boundaries: list  # [RecutBoundary]
    warnings: list = field(default_factory=list)
    feasible: bool = True
    target_sec: float = 0.0
    floor_sec: float = 0.0
    scene_cuts_used: bool = False

    @property
    def changed(self) -> bool:
        return any(b.moved for b in self.boundaries)

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "changed": self.changed,
            "target_sec": self.target_sec,
            "floor_sec": self.floor_sec,
            "scene_cuts_used": self.scene_cuts_used,
            "ranges": [[s, e] for s, e in self.ranges],
            "boundaries": [
                {
                    "index": b.index,
                    "old_sec": b.old_sec,
                    "new_sec": b.new_sec,
                    "moved": b.moved,
                    "source": b.source,
                    "old_inside_line_ids": list(b.old_inside_line_ids),
                    "new_inside_line_ids": list(b.new_inside_line_ids),
                }
                for b in self.boundaries
            ],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _finite(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def clean_lines(lines: Iterable) -> list[tuple[float, float, object]]:
    """``(start, end, id)`` for every line with finite, positive-length timing,
    sorted by start. Malformed lines are skipped (they cannot be split)."""
    out = []
    for i, ln in enumerate(lines or []):
        s, e = _finite(_get(ln, "start")), _finite(_get(ln, "end"))
        if s is None or e is None or e - s <= _EPS:
            continue
        lid = _get(ln, "id")
        out.append((s, e, lid if lid is not None else i + 1))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def speech_pauses(
    lines: Sequence[tuple[float, float, object]], duration_sec: float
) -> list[tuple[float, float]]:
    """The gaps in the union of *lines* within ``[0, duration]``, as
    ``(start, end)``; zero-width where two lines touch. Leading and trailing
    silence included."""
    merged: list[list[float]] = []
    for s, e, _ in lines:
        s, e = max(0.0, s), min(duration_sec, e)
        if e <= s:
            continue
        if merged and s < merged[-1][1] - _EPS:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    pauses = []
    cursor = 0.0
    for s, e in merged:
        if s >= cursor - _EPS:
            pauses.append((cursor, max(cursor, s)))
        cursor = max(cursor, e)
    if duration_sec >= cursor - _EPS:
        pauses.append((cursor, max(cursor, duration_sec)))
    return pauses


def lines_containing(
    t: float, lines: Sequence[tuple[float, float, object]], tol: float = EDGE_TOL_SEC
) -> tuple:
    """Ids of the lines *t* is strictly inside (more than *tol* from both edges;
    the tolerance shrinks to a quarter of a very short line)."""
    hits = []
    for s, e, lid in lines:
        margin = min(tol, (e - s) / 4.0)
        if s + margin < t < e - margin:
            hits.append(lid)
    return tuple(hits)


def _pause_candidates(
    pauses, scene_cuts, duration_sec, tol
) -> list[tuple[float, str, float]]:
    """``(position, source, off_ideal)`` for every usable point of every pause."""
    cands: dict[float, tuple[float, str, float]] = {}

    def add(p, source, off):
        # Rounded HERE, before the plan is optimised, so the written position
        # is exactly the one every length constraint was checked against.
        p = round(p, _ROUND)
        if not (_EPS < p < duration_sec - _EPS):
            return
        key = p
        cur = cands.get(key)
        if cur is None or off < cur[2]:
            cands[key] = (p, source, off)

    for gs, ge in pauses:
        mid = (gs + ge) / 2.0
        inside = [c for c in scene_cuts if gs - tol <= c <= ge + tol]
        if inside:
            ideal, ideal_src = min(inside, key=lambda c: (abs(c - mid), c)), "scene_cut"
        else:
            ideal, ideal_src = mid, "pause_middle"
        add(ideal, ideal_src, 0.0)
        if ideal_src == "scene_cut":
            add(mid, "pause_middle", abs(mid - ideal))
        for p in (gs, ge):
            add(p, "pause", abs(p - ideal))
        width = ge - gs
        if width > 2 * PAUSE_GRID_SEC:
            n = int(width // PAUSE_GRID_SEC)
            for k in range(1, n + 1):
                p = gs + k * PAUSE_GRID_SEC
                if p < ge - _EPS:
                    add(p, "pause", abs(p - ideal))
    return sorted(cands.values(), key=lambda c: c[0])


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def recut_at_speech_pauses(
    segments: Sequence,
    lines: Iterable,
    *,
    duration_sec: float,
    target_sec: float,
    floor_sec: float = 0.0,
    scene_cuts: Optional[Iterable[float]] = None,
    edge_tol_sec: float = EDGE_TOL_SEC,
) -> RecutPlan:
    """Move the boundaries of *segments* off the spoken *lines*.

    *segments* are dicts or objects with ``start_sec``/``end_sec``/``action``
    and optional ``pre_roll_sec``/``post_roll_sec``, ordered, contiguous and
    covering ``[0, duration_sec]``. *lines* are transcript lines
    (``start``/``end``/``id``). *scene_cuts* are cut times in seconds (``None``
    or empty when unknown). Only ``swap`` segments are held to
    ``[floor_sec, target_sec]`` — both the segment and the clip the model is
    sent (pre/post-roll applied and clamped to the video, as the pipeline
    cuts it); a ``keep`` segment only has to stay longer than zero.

    Returns a :class:`RecutPlan` with exactly ``len(segments)`` ranges.
    Raises ``ValueError`` on malformed input (not contiguous, unordered,
    non-finite, or a non-positive target).
    """
    duration = _finite(duration_sec)
    target = _finite(target_sec)
    floor = _finite(floor_sec) or 0.0
    if duration is None or duration <= 0:
        raise ValueError(f"duration_sec must be a positive number, got {duration_sec!r}")
    if target is None or target <= 0:
        raise ValueError(f"target_sec must be a positive number, got {target_sec!r}")
    floor = max(0.0, min(floor, target))
    segs = list(segments)
    if not segs:
        raise ValueError("no segments")

    starts, ends, actions, rolls = [], [], [], []
    for seg in segs:
        s, e = _finite(_get(seg, "start_sec")), _finite(_get(seg, "end_sec"))
        if s is None or e is None or e - s <= _EPS:
            raise ValueError(f"segment with invalid bounds: {s!r}..{e!r}")
        starts.append(s)
        ends.append(e)
        actions.append(_get(seg, "action") or "swap")
        rolls.append((
            _finite(_get(seg, "pre_roll_sec")) or 0.0,
            _finite(_get(seg, "post_roll_sec")) or 0.0,
        ))
    if abs(starts[0]) > 1e-3 or abs(ends[-1] - duration) > 1e-3:
        raise ValueError(
            f"segments must cover [0, {duration:.3f}]: they cover "
            f"[{starts[0]:.3f}, {ends[-1]:.3f}]"
        )
    for i in range(1, len(segs)):
        if abs(starts[i] - ends[i - 1]) > 1e-3:
            raise ValueError(
                f"segments are not contiguous at index {i}: "
                f"{ends[i - 1]:.3f} != {starts[i]:.3f}"
            )

    clines = clean_lines(lines)
    cuts = sorted(
        c for c in (_finite(x) for x in (scene_cuts or [])) if c is not None
        and _EPS < c < duration - _EPS
    )
    # Boundary positions: old[0] is the start, old[k + 1] the end; both fixed.
    old = [starts[0]] + ends[:-1] + [ends[-1]]
    n_seg = len(segs)
    k = n_seg - 1  # interior boundaries

    plan_kwargs = dict(
        target_sec=target, floor_sec=floor, scene_cuts_used=bool(cuts)
    )

    def clip_len(i: int, a: float, b: float) -> float:
        # The window the model is actually sent, exactly as the pipeline cuts
        # it (pipeline_v2: clip_start = max(0, start - pre_roll), clip_end =
        # min(duration, end + post_roll)); a roll may be negative.
        pre, post = rolls[i]
        return min(duration, b + post) - max(0.0, a - pre)

    def seg_ok(i: int, a: float, b: float) -> bool:
        length = b - a
        if length <= 1e-3:
            return False
        if actions[i] != "swap":
            return True
        clip = clip_len(i, a, b)
        return (
            floor - _EPS <= min(length, clip)
            and max(length, clip) <= target + _EPS
        )

    def unchanged_plan(feasible: bool, warnings: list) -> RecutPlan:
        bounds = [
            RecutBoundary(
                index=i - 1, old_sec=old[i], new_sec=old[i], moved=False,
                source="kept",
                old_inside_line_ids=lines_containing(old[i], clines, edge_tol_sec),
                new_inside_line_ids=lines_containing(old[i], clines, edge_tol_sec),
            )
            for i in range(1, k + 1)
        ]
        return RecutPlan(
            ranges=[(starts[i], ends[i]) for i in range(n_seg)],
            boundaries=bounds, warnings=warnings, feasible=feasible, **plan_kwargs,
        )

    bad = [i for i in range(n_seg) if not seg_ok(i, old[i], old[i + 1])]
    if bad:
        details = ", ".join(
            f"segment {i + 1} ({old[i + 1] - old[i]:.2f}s"
            + (
                f", {clip_len(i, old[i], old[i + 1]):.2f}s with pre/post-roll"
                if any(rolls[i]) else ""
            )
            + ")"
            for i in bad
        )
        return unchanged_plan(False, [
            f"The current segments already break the {floor:g}-{target:g}s "
            f"limit for swap segments: {details}. Nothing was re-cut — fix "
            "those segments (or re-analyze) first."
        ])
    if k == 0:
        return unchanged_plan(True, [])
    if not clines:
        return unchanged_plan(True, ["The transcript has no timed lines; nothing to re-cut."])

    pause_cands = _pause_candidates(
        speech_pauses(clines, duration), cuts, duration, edge_tol_sec
    )

    # layers[j] = candidate positions for position j (0 and k+1 fixed).
    # Each entry: (pos, source, off_ideal, inside_ids)
    layers: list[list[tuple]] = [[(old[0], "kept", 0.0, ())]]
    for j in range(1, k + 1):
        layer = [(old[j], "kept", 0.0, lines_containing(old[j], clines, edge_tol_sec))]
        for p, src, off in pause_cands:
            if abs(p - old[j]) <= _EPS:
                continue  # the original position is already the "kept" entry
            layer.append((p, src, off, lines_containing(p, clines, edge_tol_sec)))
        layers.append(layer)
    layers.append([(old[k + 1], "kept", 0.0, ())])

    # best[j][c] = (cost tuple, back-pointer)
    best: list[dict] = [{0: ((0, 0.0), None)}]
    for j in range(1, k + 2):
        cur: dict = {}
        # Reachable predecessors sorted by position, so a swap segment only
        # scans the ones within the target (a bisect, not every pair).
        prev = sorted(
            ((layers[j - 1][pi][0], pi) for pi in best[j - 1]),
            key=lambda t: (t[0], t[1]),
        )
        prev_pos = [t[0] for t in prev]
        for ci, (pos, _src, off, inside) in enumerate(layers[j]):
            step_inside = 1 if inside else 0
            step_cost = (abs(pos - old[j]) + OFF_IDEAL_WEIGHT * off) if j <= k else 0.0
            if actions[j - 1] == "swap":
                lo = bisect.bisect_left(prev_pos, pos - target - _EPS)
                hi = bisect.bisect_right(prev_pos, pos - floor + _EPS)
            else:
                lo, hi = 0, bisect.bisect_left(prev_pos, pos)
            # Deterministic order: by predecessor index (the "kept" entry first).
            window = sorted(prev[lo:hi], key=lambda t: t[1])
            choice = None
            for ppos, pi in window:
                pcost = best[j - 1][pi][0]
                if not seg_ok(j - 1, ppos, pos):
                    continue
                cost = (pcost[0] + step_inside, pcost[1] + step_cost)
                # Deterministic: the lower cost wins; an equal cost keeps the
                # first predecessor in candidate order.
                if choice is None or cost[0] < choice[0][0] or (
                    cost[0] == choice[0][0] and cost[1] < choice[0][1] - 1e-9
                ):
                    choice = (cost, pi)
            if choice is not None:
                cur[ci] = choice
        best.append(cur)
        if not cur:
            # Unreachable: the all-"kept" path is always valid once the current
            # partition passed seg_ok above.
            return unchanged_plan(False, ["No valid re-cut exists."])

    # Back-track from the fixed end.
    picks = [None] * (k + 2)
    ci = 0
    for j in range(k + 1, -1, -1):
        picks[j] = layers[j][ci]
        if j > 0:
            ci = best[j][ci][1]

    boundaries = []
    warnings = []
    new_pos = [old[0]]
    for j in range(1, k + 1):
        pos, src, _off, inside = picks[j]
        moved = src != "kept"
        new_pos.append(pos)
        old_inside = lines_containing(old[j], clines, edge_tol_sec)
        boundaries.append(RecutBoundary(
            index=j - 1, old_sec=old[j], new_sec=pos, moved=moved, source=src,
            old_inside_line_ids=old_inside, new_inside_line_ids=inside,
        ))
        if inside:
            warnings.append(
                f"The boundary at {old[j]:.2f}s (between segments {j} and {j + 1}) "
                f"stays inside line {', '.join(str(x) for x in inside)}: no pause "
                f"keeps both neighbouring segments within {target:g}s with "
                f"{n_seg} segments."
            )
    new_pos.append(duration)
    new_pos[-1] = old[k + 1]
    ranges = [(new_pos[i], new_pos[i + 1]) for i in range(n_seg)]
    return RecutPlan(ranges=ranges, boundaries=boundaries, warnings=warnings,
                     feasible=True, **plan_kwargs)


def plan_fingerprint(
    segments: Sequence, lines: Iterable, *, duration_sec, target_sec, floor_sec,
    scene_cuts,
) -> str:
    """Hash of every input of a plan, so a preview can be applied only if
    nothing it was computed from has changed since (the endpoint's
    ``expected_fingerprint``)."""
    payload = {
        "segments": [
            [
                str(_get(s, "id")), _get(s, "action"),
                round(float(_get(s, "start_sec")), 4),
                round(float(_get(s, "end_sec")), 4),
                round(float(_get(s, "pre_roll_sec") or 0.0), 4),
                round(float(_get(s, "post_roll_sec") or 0.0), 4),
            ]
            for s in segments
        ],
        "lines": [[round(s, 4), round(e, 4), str(lid)] for s, e, lid in clean_lines(lines)],
        "duration": round(float(duration_sec), 4),
        "target": round(float(target_sec), 4),
        "floor": round(float(floor_sec), 4),
        "scene_cuts": [round(float(c), 4) for c in (scene_cuts or [])],
        "rule": [EDGE_TOL_SEC, PAUSE_GRID_SEC, OFF_IDEAL_WEIGHT],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Project glue (not pure): what the endpoint and the autopilot share
# ---------------------------------------------------------------------------


@dataclass
class ProjectRecut:
    plan: RecutPlan
    fingerprint: str
    scene_cuts: Optional[list]  # None = unknown for this source


def plan_for_project(project, segments: Sequence) -> ProjectRecut:
    """Plan a re-cut of *project*'s *segments* (SegmentDefs ordered by index)
    from its stored transcript, its whole-video target and the scene cuts its
    analysis recorded. Reads the scene-cut cache file; writes nothing.

    Raises ``ValueError`` when the segments are not a valid partition of the
    probed duration (see :func:`recut_at_speech_pauses`)."""
    from .config import settings
    from .pipeline_v2 import whole_video_floor_sec, whole_video_target_sec
    from .scene import read_scene_cuts
    from .storage import scene_cuts_cache_path

    target = whole_video_target_sec(project.max_segment_sec)
    floor = whole_video_floor_sec()
    cuts = read_scene_cuts(
        scene_cuts_cache_path(project.id),
        source_path=project.source_local_path,
        min_scene_len_sec=settings.SCENE_MIN_LEN_SECONDS,
    )
    lines = (project.transcript or {}).get("lines") or []
    duration = float(project.duration_sec)
    plan = recut_at_speech_pauses(
        segments, lines, duration_sec=duration, target_sec=target,
        floor_sec=floor, scene_cuts=cuts,
    )
    if cuts is None and plan.changed:
        plan.warnings.append(
            "The scene cuts of this source are not on record (it was analyzed "
            "before scene cuts were kept), so moved boundaries sit in the middle "
            "of their pause rather than on a scene cut. A new whole-video "
            "analysis records them."
        )
    fingerprint = plan_fingerprint(
        segments, lines, duration_sec=duration, target_sec=target,
        floor_sec=floor, scene_cuts=cuts,
    )
    return ProjectRecut(plan=plan, fingerprint=fingerprint, scene_cuts=cuts)


def apply_plan(segments: Sequence, plan: RecutPlan) -> int:
    """Write *plan*'s ranges onto *segments* (SegmentDef rows, ordered by
    index) IN PLACE: ids, actions, pre/post-roll and index stay. Returns how
    many rows changed. The caller commits."""
    if len(segments) != len(plan.ranges):
        raise ValueError(
            f"plan has {len(plan.ranges)} ranges for {len(segments)} segments"
        )
    changed = 0
    for seg, (s, e) in zip(segments, plan.ranges):
        if abs(float(seg.start_sec) - s) > _EPS or abs(float(seg.end_sec) - e) > _EPS:
            seg.start_sec, seg.end_sec = s, e
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Transaction-neutral service: the endpoint and the autopilot (AICP-365)
# ---------------------------------------------------------------------------


class RecutRefused(Exception):
    """The project cannot be re-cut now. ``str(exc)`` is the operator-facing
    reason; the endpoint answers 409 with it. Raised BEFORE anything is
    written, so the caller's transaction holds no change from this call."""


@dataclass
class RecutOutcome:
    plan: RecutPlan
    fingerprint: str
    applied: bool  # True when segment rows were changed (in the session, NOT committed)
    warnings: list  # plan warnings plus project-level ones
    segments: list  # the project's SegmentDefs, ordered by index


def recut_project_segments(
    db,
    project,
    *,
    dry_run: bool = False,
    expected_fingerprint: Optional[str] = None,
) -> RecutOutcome:
    """Plan and (unless *dry_run*) apply a speech-pause re-cut of *project*.

    TRANSACTION-NEUTRAL: never calls ``commit()`` or ``rollback()`` — the
    caller owns the transaction (the endpoint commits/rolls back; the
    autopilot commits once in its own action handler), and nothing is
    flushed: the new bounds are set on the SegmentDef objects IN PLACE and
    reach the database with the caller's commit. An apply first locks the
    project row (``SELECT ... FOR UPDATE`` on its id, the lock PATCH
    /segments, project deletion and a Fal admission take first, re-reading
    the project under it) and then reads the segments FOR UPDATE with fresh
    values, so the plan is computed from exactly the rows it writes; the
    locks are held until the caller ends the transaction. It refuses while
    the session holds unflushed changes to any VideoProject, SegmentDef or
    Run (its guards read the database and could not see them): call it
    before touching those rows in the transaction, or flush first.

    Raises :class:`RecutRefused` (nothing written) unless the project is a
    localisation project analyzed in whole scope, ``ready``, with a ready
    transcript and a probed duration, no run of it is active, its segments
    are a valid partition, and — when given — *expected_fingerprint* still
    matches. An infeasible or unchanged plan is returned with
    ``applied=False`` and writes nothing.
    """
    with db.no_autoflush:
        # No autoflush anywhere in here: nothing the caller has pending reaches
        # the database before the project lock is held (PATCH /segments locks
        # the project row first, so flushing a segment edit before it would
        # invert the lock order), and nothing at all is flushed by a refusal.
        return _recut_project_segments(
            db, project, dry_run=dry_run, expected_fingerprint=expected_fingerprint,
        )


def _recut_project_segments(db, project, *, dry_run, expected_fingerprint):
    from sqlalchemy import select

    from .api_v2 import _ACTIVE_RUN_STATUSES  # lazy: api_v2 imports this module's callers
    from .models import ProjectStatus, Run, RunStatus, SegmentDef, VideoProject
    from .project_types import LOCALISATION, LOCALISATION_SCOPE_WHOLE, spec_for

    # Every guard below reads the database, and under no_autoflush it cannot
    # see changes the caller has not flushed: a run it just re-queued, a
    # segment it added through a relationship (project_id unset until flush),
    # a transcript it edited. So the session must hold none for these tables.
    # Call this before changing projects/segments/runs in the transaction, or
    # flush them first.
    pending = [
        obj for obj in (*db.new, *db.dirty, *db.deleted)
        if isinstance(obj, (VideoProject, SegmentDef, Run))
    ]
    if pending:
        raise RecutRefused(
            "This transaction has unflushed changes to projects, segments or "
            "runs; flush or commit them before re-cutting."
        )
    # Nothing of the caller's is pending (checked above), so re-reading loses
    # nothing; an apply re-reads under the row lock.
    db.refresh(project, with_for_update=not dry_run)
    if spec_for(project.project_type).key != LOCALISATION or (
        project.localisation_scope != LOCALISATION_SCOPE_WHOLE
    ):
        raise RecutRefused(
            "Re-cut at speech pauses is only for a localisation project "
            "analyzed as the whole video (re-analyze it with the whole-video "
            "scope first)."
        )
    if project.status != ProjectStatus.ready:
        status = getattr(project.status, "value", project.status)
        raise RecutRefused(
            f"Cannot re-cut: project status is {status!r}, expected 'ready'"
        )
    transcript = project.transcript if isinstance(project.transcript, dict) else {}
    if project.transcript_status != "ready" or not transcript.get("lines"):
        raise RecutRefused(
            "Cannot re-cut at speech pauses without a ready transcript "
            f"(transcript status: {project.transcript_status or 'none'}). "
            "Transcribe the video first."
        )
    if not isinstance(project.duration_sec, (int, float)) or project.duration_sec <= 0:
        raise RecutRefused("The video has not been probed yet")
    active = db.execute(
        select(Run.id, Run.name).where(
            Run.project_id == project.id, Run.status.in_(_ACTIVE_RUN_STATUSES)
        )
    ).all()
    if active:
        names = ", ".join((name or rid[:8]) for rid, name in active)
        raise RecutRefused(
            f"Cannot re-cut segments while a run is in flight ({names}). "
            "A run that has not yet frozen its own timeline reads the "
            "project's live segments. Wait for it to finish, then re-cut."
        )

    seg_query = (
        select(SegmentDef)
        .where(SegmentDef.project_id == project.id)
        .order_by(SegmentDef.index)
    )
    if not dry_run:
        seg_query = seg_query.with_for_update()
    seg_query = seg_query.execution_options(populate_existing=True)
    segments = list(db.execute(seg_query).scalars().all())
    if not segments:
        raise RecutRefused("The project has no segments")
    try:
        recut = plan_for_project(project, segments)
    except ValueError as exc:
        raise RecutRefused(f"Cannot re-cut these segments: {exc}") from exc
    plan = recut.plan

    warnings = list(plan.warnings)
    if plan.changed:
        # Counted in Python: a JSON column stores an explicit None as JSON
        # 'null' (a copy is created with segment_snapshot=None), which IS NULL
        # would miss.
        unfrozen = sum(
            1 for snap in db.execute(
                select(Run.segment_snapshot).where(
                    Run.project_id == project.id, Run.status != RunStatus.done,
                )
            ).scalars()
            if not snap
        )
        if unfrozen:
            warnings.append(
                f"{unfrozen} unfinished run(s) of this project never froze their "
                "segments; retrying one uses the new boundaries."
            )

    applied = False
    if not dry_run:
        if expected_fingerprint and expected_fingerprint != recut.fingerprint:
            raise RecutRefused(
                "The segments, transcript or scene cuts changed since this "
                "re-cut was previewed. Preview it again."
            )
        if plan.feasible and plan.changed:
            apply_plan(segments, plan)  # pending in the session; the caller's commit writes it
            applied = True
    return RecutOutcome(
        plan=plan, fingerprint=recut.fingerprint, applied=applied,
        warnings=warnings, segments=segments,
    )
