"""
face.py — Face-detection segmentation module for re-skin.

Produces a proposed segmentation of a video: which time ranges contain a
person's face (action="swap") and which don't (action="keep").

Design
------
- I/O layer  : detect_timeline() — wraps InsightFace + OpenCV.
- Pure logic  : all other public functions — no I/O, fully unit-testable.

Usage
-----
    segments = propose_segments(
        "clip.mp4",
        duration_sec=probe("clip.mp4").duration_sec,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded module-level detector cache
# ---------------------------------------------------------------------------

_detector_cache: object | None = None


def _build_detector() -> object:
    """Build (and cache) the InsightFace FaceAnalysis detector."""
    global _detector_cache
    if _detector_cache is None:
        import insightface  # type: ignore[import]

        app = insightface.app.FaceAnalysis(allowed_modules=["detection"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _detector_cache = app
    return _detector_cache


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class FaceBox:
    """A single detected face bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def size(self) -> float:
        """Max of width and height in pixels."""
        return max(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class FrameDetection:
    """Detection result for one sampled frame."""

    t_sec: float
    faces: list[FaceBox] = field(default_factory=list)


@dataclass
class ProposedSegment:
    """A contiguous time range with a proposed action."""

    start_sec: float
    end_sec: float
    has_face: bool
    action: str  # "swap" | "keep"


# ---------------------------------------------------------------------------
# I/O layer
# ---------------------------------------------------------------------------


def detect_timeline(
    video_path: str,
    sample_fps: float = 2.0,
    detector: object | None = None,
) -> list[FrameDetection]:
    """
    Open *video_path*, sample frames at *sample_fps*, run face detection on
    each, and return a list of FrameDetection ordered by time.

    Parameters
    ----------
    video_path  : Path to the source video file.
    sample_fps  : Frames per second to sample (e.g. 2.0 = one frame every 0.5s).
    detector    : Optional injectable InsightFace FaceAnalysis instance.
                  If None, the module-level cached detector is used.
    """
    import cv2  # type: ignore[import]

    if detector is None:
        detector = _build_detector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path!r}")

    try:
        native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / native_fps if native_fps else 0.0

        frame_interval = max(1, round(native_fps / sample_fps))
        results: list[FrameDetection] = []
        frame_idx = 0

        while True:
            if not cap.grab():
                break
            if frame_idx % frame_interval == 0:
                ret, bgr = cap.retrieve()
                if not ret:
                    break
                t_sec = frame_idx / native_fps
                faces_raw = detector.get(bgr)
                faces = [
                    FaceBox(
                        x1=float(f.bbox[0]),
                        y1=float(f.bbox[1]),
                        x2=float(f.bbox[2]),
                        y2=float(f.bbox[3]),
                        score=float(f.det_score),
                    )
                    for f in (faces_raw or [])
                ]
                results.append(FrameDetection(t_sec=t_sec, faces=faces))
            frame_idx += 1

        log.info(
            "detect_timeline(%s): native_fps=%.2f sample_fps=%.2f frames_sampled=%d",
            video_path,
            native_fps,
            sample_fps,
            len(results),
        )
        return results
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Pure logic layer
# ---------------------------------------------------------------------------


def filter_small_faces(
    frames: list[FrameDetection],
    min_face_px: int,
) -> list[FrameDetection]:
    """
    Remove FaceBox entries whose *size* (max of width, height) is below
    *min_face_px*.  Returns new FrameDetection objects; originals are not
    mutated.
    """
    filtered: list[FrameDetection] = []
    for fd in frames:
        kept = [fb for fb in fd.faces if fb.size >= min_face_px]
        filtered.append(FrameDetection(t_sec=fd.t_sec, faces=kept))
    return filtered


def presence_timeline(
    frames: list[FrameDetection],
) -> list[tuple[float, bool]]:
    """
    Convert a list of FrameDetection to (t_sec, has_any_face) tuples.
    """
    return [(fd.t_sec, bool(fd.faces)) for fd in frames]


def group_intervals(
    timeline: list[tuple[float, bool]],
    bridge_gaps: int = 1,
) -> list[tuple[float, float]]:
    """
    Merge contiguous True frames into (start, end) intervals.

    A gap of at most *bridge_gaps* consecutive False frames between two True
    regions is bridged (treated as True) to avoid fragmenting on a single
    missed detection.

    Parameters
    ----------
    timeline    : Ordered list of (t_sec, has_face) pairs.
    bridge_gaps : Maximum number of consecutive False frames to bridge over.

    Returns
    -------
    List of (start_sec, end_sec) intervals where faces were detected.
    """
    if not timeline:
        return []

    # Build a boolean sequence first so we can do gap bridging.
    times = [t for t, _ in timeline]
    flags = [has_face for _, has_face in timeline]

    # Bridge short gaps: if a sequence of ≤bridge_gaps Falses is surrounded
    # by Trues on both sides, flip them to True.
    if bridge_gaps > 0:
        n = len(flags)
        i = 0
        while i < n:
            if not flags[i]:
                # Count consecutive Falses starting at i
                j = i
                while j < n and not flags[j]:
                    j += 1
                gap_len = j - i
                # Bridge only if gap is short enough AND has True on both sides
                if gap_len <= bridge_gaps and i > 0 and j < n and flags[i - 1] and flags[j]:
                    for k in range(i, j):
                        flags[k] = True
                i = j
            else:
                i += 1

    # Merge contiguous True runs into intervals.
    intervals: list[tuple[float, float]] = []
    in_interval = False
    start_t = 0.0

    for idx, (t, has_face) in enumerate(zip(times, flags)):
        if has_face and not in_interval:
            start_t = t
            in_interval = True
        elif not has_face and in_interval:
            # End interval at the *previous* timestamp
            end_t = times[idx - 1]
            intervals.append((start_t, end_t))
            in_interval = False

    # Close any open interval at the last timestamp
    if in_interval:
        intervals.append((start_t, times[-1]))

    return intervals


def apply_lead_in(
    intervals: list[tuple[float, float]],
    lead_in_sec: float,
    lower_bound: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Extend each interval's start backward by *lead_in_sec*.

    Constraints
    -----------
    - Start is clamped to *lower_bound* (default 0).
    - Start cannot overlap the *end* of the previous interval.

    Returns a new list; inputs are not mutated.
    """
    result: list[tuple[float, float]] = []
    for i, (start, end) in enumerate(intervals):
        new_start = max(lower_bound, start - lead_in_sec)
        # Don't overlap the previous interval's end
        if i > 0:
            prev_end = result[i - 1][1]
            new_start = max(new_start, prev_end)
        result.append((new_start, end))
    return result


def split_max_duration(
    intervals: list[tuple[float, float]],
    max_sec: float = 15.0,
) -> list[tuple[float, float]]:
    """
    Split any interval longer than *max_sec* into consecutive chunks, covering
    the same range exactly. Splitting is *even*: an interval is divided into the
    fewest equal chunks that all fit within *max_sec*, which avoids leaving a
    tiny trailing chunk (e.g. 16s -> 8s+8s, not 15s+1s).
    """
    import math

    result: list[tuple[float, float]] = []
    for start, end in intervals:
        dur = end - start
        if dur <= max_sec:
            result.append((start, end))
            continue
        n = math.ceil(dur / max_sec)
        step = dur / n
        for i in range(n):
            chunk_start = start + i * step
            chunk_end = end if i == n - 1 else start + (i + 1) * step
            result.append((chunk_start, chunk_end))
    return result


def drop_short_intervals(
    intervals: list[tuple[float, float]],
    min_sec: float,
) -> list[tuple[float, float]]:
    """
    Remove intervals shorter than *min_sec*.

    Used to discard spurious face blips (e.g. a face detected on a single
    sampled frame) and to satisfy the downstream AI model's minimum reference
    video duration. Dropped ranges simply become part of the surrounding
    untouched ("keep") gaps in the final partition.
    """
    return [(s, e) for (s, e) in intervals if (e - s) >= min_sec]


def apply_rolls(
    start: float,
    end: float,
    pre_roll_sec: float,
    post_roll_sec: float,
    lower: float = 0.0,
    upper: float | None = None,
) -> tuple[float, float]:
    """
    Adjust a segment's boundaries by pre/post roll amounts.

    *pre_roll_sec*  extends (or contracts) the start backward.
    *post_roll_sec* extends (or contracts) the end forward.

    Both are clamped to [lower, upper].
    """
    new_start = max(lower, start - pre_roll_sec)
    new_end = end + post_roll_sec
    if upper is not None:
        new_end = min(upper, new_end)
    return (new_start, new_end)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_EPSILON = 1e-4  # gaps smaller than this are dropped


def propose_segments(
    video_path: str,
    *,
    duration_sec: float,
    sample_fps: float = 2.0,
    min_face_px: int = 120,
    lead_in_sec: float = 0.5,
    bridge_gaps: int = 1,
    max_segment_sec: float = 15.0,
    min_segment_sec: float = 2.0,
    detector: object | None = None,
) -> list[ProposedSegment]:
    """
    Produce the full ordered partition of [0, duration_sec].

    Steps
    -----
    1. detect_timeline          — sample frames, run InsightFace.
    2. filter_small_faces       — drop spurious small faces.
    3. presence_timeline        — per-frame boolean.
    4. group_intervals          — merge True runs (with gap bridging).
    5. apply_lead_in            — extend starts backward.
    6. drop_short_intervals     — discard blips < min_segment_sec.
    7. split_max_duration       — enforce ≤max_segment_sec per chunk (even split).
    8. Interleave face intervals with keep gaps → full partition.

    Returns
    -------
    Sorted, contiguous, non-overlapping ProposedSegment list covering
    exactly [0, duration_sec].
    """
    raw_frames = detect_timeline(video_path, sample_fps=sample_fps, detector=detector)
    filtered = filter_small_faces(raw_frames, min_face_px=min_face_px)
    timeline = presence_timeline(filtered)
    face_intervals = group_intervals(timeline, bridge_gaps=bridge_gaps)
    face_intervals = apply_lead_in(face_intervals, lead_in_sec=lead_in_sec, lower_bound=0.0)
    # Drop spurious short blips (and satisfy the AI model's min reference duration)
    # before splitting, so even-splitting never produces sub-minimum chunks.
    face_intervals = drop_short_intervals(face_intervals, min_sec=min_segment_sec)
    face_intervals = split_max_duration(face_intervals, max_sec=max_segment_sec)

    # Build the full partition of [0, duration_sec].
    segments: list[ProposedSegment] = []
    cursor = 0.0

    for start, end in face_intervals:
        # Clamp to valid range
        start = max(0.0, min(start, duration_sec))
        end = max(0.0, min(end, duration_sec))
        if end <= start:
            continue

        # Gap before this face interval
        if start - cursor > _EPSILON:
            segments.append(
                ProposedSegment(
                    start_sec=cursor,
                    end_sec=start,
                    has_face=False,
                    action="keep",
                )
            )

        segments.append(
            ProposedSegment(
                start_sec=start,
                end_sec=end,
                has_face=True,
                action="swap",
            )
        )
        cursor = end

    # Trailing gap
    if duration_sec - cursor > _EPSILON:
        segments.append(
            ProposedSegment(
                start_sec=cursor,
                end_sec=duration_sec,
                has_face=False,
                action="keep",
            )
        )

    return segments
