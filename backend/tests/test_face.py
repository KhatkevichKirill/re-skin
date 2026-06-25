"""
Tests for backend/app/face.py.

Pure-logic tests use fabricated FrameDetection / FaceBox data — no real
images or I/O required.  A single integration smoke-test generates a tiny
synthetic video via ffmpeg and runs detect_timeline() against it; it is
skipped if InsightFace cannot initialise.
"""

from __future__ import annotations

import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.face import (
    FaceBox,
    FrameDetection,
    ProposedSegment,
    apply_lead_in,
    apply_rolls,
    detect_timeline,
    drop_short_intervals,
    filter_small_faces,
    group_intervals,
    presence_timeline,
    propose_segments,
    split_max_duration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fb(x1=0.0, y1=0.0, x2=100.0, y2=100.0, score=0.95) -> FaceBox:
    return FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def _small_fb() -> FaceBox:
    """A 30×40 px box (size=40 < typical min_face_px=120)."""
    return FaceBox(x1=0.0, y1=0.0, x2=30.0, y2=40.0, score=0.9)


def _large_fb() -> FaceBox:
    """A 200×200 px box (size=200 ≥ any reasonable min_face_px)."""
    return FaceBox(x1=0.0, y1=0.0, x2=200.0, y2=200.0, score=0.95)


def _frames_from_bools(flags: list[bool], step: float = 0.5) -> list[FrameDetection]:
    """Build FrameDetection list from a boolean sequence with *step*-second spacing."""
    return [
        FrameDetection(t_sec=i * step, faces=[_large_fb()] if flag else [])
        for i, flag in enumerate(flags)
    ]


# ---------------------------------------------------------------------------
# FaceBox.size
# ---------------------------------------------------------------------------


class TestFaceBoxSize:
    def test_size_uses_max_dimension(self):
        fb = FaceBox(x1=0, y1=0, x2=50, y2=80, score=0.9)
        assert fb.size == 80.0

    def test_size_equal_dimensions(self):
        fb = FaceBox(x1=10, y1=10, x2=110, y2=110, score=0.9)
        assert fb.size == 100.0


# ---------------------------------------------------------------------------
# filter_small_faces
# ---------------------------------------------------------------------------


class TestFilterSmallFaces:
    def test_removes_sub_threshold_boxes(self):
        frames = [
            FrameDetection(t_sec=0.0, faces=[_small_fb()]),
            FrameDetection(t_sec=0.5, faces=[_large_fb()]),
        ]
        result = filter_small_faces(frames, min_face_px=120)
        assert result[0].faces == []
        assert len(result[1].faces) == 1

    def test_keeps_boxes_at_threshold(self):
        fb = FaceBox(x1=0, y1=0, x2=120, y2=50, score=0.9)  # size=120
        frames = [FrameDetection(t_sec=0.0, faces=[fb])]
        result = filter_small_faces(frames, min_face_px=120)
        assert len(result[0].faces) == 1

    def test_removes_all_when_all_small(self):
        frames = [FrameDetection(t_sec=0.0, faces=[_small_fb(), _small_fb()])]
        result = filter_small_faces(frames, min_face_px=120)
        assert result[0].faces == []

    def test_does_not_mutate_original(self):
        frames = [FrameDetection(t_sec=0.0, faces=[_small_fb(), _large_fb()])]
        _ = filter_small_faces(frames, min_face_px=120)
        assert len(frames[0].faces) == 2  # original unchanged

    def test_mixed_sizes_in_single_frame(self):
        frames = [
            FrameDetection(t_sec=0.0, faces=[_small_fb(), _large_fb(), _small_fb()])
        ]
        result = filter_small_faces(frames, min_face_px=120)
        assert len(result[0].faces) == 1

    def test_empty_frames_list(self):
        assert filter_small_faces([], min_face_px=120) == []


# ---------------------------------------------------------------------------
# presence_timeline
# ---------------------------------------------------------------------------


class TestPresenceTimeline:
    def test_basic(self):
        frames = [
            FrameDetection(t_sec=0.0, faces=[_large_fb()]),
            FrameDetection(t_sec=0.5, faces=[]),
        ]
        result = presence_timeline(frames)
        assert result == [(0.0, True), (0.5, False)]

    def test_empty(self):
        assert presence_timeline([]) == []


# ---------------------------------------------------------------------------
# group_intervals
# ---------------------------------------------------------------------------


class TestGroupIntervals:
    def test_single_true_run(self):
        # t=0,1,2 True; t=3 False
        tl = [(0.0, True), (0.5, True), (1.0, True), (1.5, False)]
        result = group_intervals(tl)
        assert len(result) == 1
        assert result[0] == (0.0, 1.0)

    def test_two_distinct_runs(self):
        tl = [(0.0, True), (0.5, False), (1.0, False), (1.5, True), (2.0, True)]
        result = group_intervals(tl, bridge_gaps=0)
        assert len(result) == 2
        assert result[0] == (0.0, 0.0)  # single-frame interval
        assert result[1] == (1.5, 2.0)

    def test_bridge_single_gap(self):
        # True, False, True — gap of 1 should be bridged with bridge_gaps=1
        tl = [(0.0, True), (0.5, False), (1.0, True)]
        result = group_intervals(tl, bridge_gaps=1)
        assert len(result) == 1
        assert result[0] == (0.0, 1.0)

    def test_no_bridge_when_gap_too_large(self):
        # True, False, False, True — gap of 2 must NOT be bridged with bridge_gaps=1
        tl = [(0.0, True), (0.5, False), (1.0, False), (1.5, True)]
        result = group_intervals(tl, bridge_gaps=1)
        assert len(result) == 2

    def test_all_false(self):
        tl = [(0.0, False), (0.5, False)]
        assert group_intervals(tl) == []

    def test_all_true(self):
        tl = [(0.0, True), (0.5, True), (1.0, True)]
        result = group_intervals(tl)
        assert result == [(0.0, 1.0)]

    def test_empty_timeline(self):
        assert group_intervals([]) == []

    def test_starts_with_false(self):
        tl = [(0.0, False), (0.5, True), (1.0, True)]
        result = group_intervals(tl, bridge_gaps=0)
        assert result == [(0.5, 1.0)]

    def test_bridge_gap_at_start_not_triggered(self):
        # flags = [F, T, F, T]
        # The False at index 0 has no True before it → NOT bridged.
        # The False at index 2 is surrounded by Trues → bridged.
        # Result after bridging: [F, T, T, T] → single interval (0.5, 1.5)
        tl = [(0.0, False), (0.5, True), (1.0, False), (1.5, True)]
        result = group_intervals(tl, bridge_gaps=1)
        assert result == [(0.5, 1.5)]


# ---------------------------------------------------------------------------
# apply_lead_in
# ---------------------------------------------------------------------------


class TestApplyLeadIn:
    def test_extends_start_backward(self):
        result = apply_lead_in([(5.0, 10.0)], lead_in_sec=0.5)
        assert result == [(4.5, 10.0)]

    def test_clamps_to_zero(self):
        result = apply_lead_in([(0.3, 5.0)], lead_in_sec=1.0, lower_bound=0.0)
        assert result[0][0] == pytest.approx(0.0)

    def test_clamps_to_custom_lower_bound(self):
        result = apply_lead_in([(2.0, 5.0)], lead_in_sec=1.5, lower_bound=1.0)
        assert result[0][0] == pytest.approx(1.0)

    def test_no_overlap_with_previous_interval(self):
        # Two intervals: first ends at 5.0, second starts at 6.0.
        # lead_in=2.0 would push second start to 4.0, but that overlaps first end=5.0.
        intervals = [(0.0, 5.0), (6.0, 10.0)]
        result = apply_lead_in(intervals, lead_in_sec=2.0)
        assert result[0] == (0.0, 5.0)  # first unchanged
        assert result[1][0] == pytest.approx(5.0)  # clamped to previous end

    def test_zero_lead_in_unchanged(self):
        intervals = [(3.0, 7.0)]
        result = apply_lead_in(intervals, lead_in_sec=0.0)
        assert result == [(3.0, 7.0)]

    def test_empty_intervals(self):
        assert apply_lead_in([], lead_in_sec=1.0) == []

    def test_multiple_intervals_independent_clamping(self):
        intervals = [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]
        result = apply_lead_in(intervals, lead_in_sec=0.5)
        assert result[0] == (1.5, 4.0)
        assert result[1] == (5.5, 8.0)
        assert result[2] == (9.5, 12.0)


# ---------------------------------------------------------------------------
# split_max_duration
# ---------------------------------------------------------------------------


class TestSplitMaxDuration:
    def test_40s_splits_into_three_even_chunks_of_15_or_less(self):
        # Even split: ceil(40/15)=3 chunks of ~13.33s (no tiny tail).
        result = split_max_duration([(0.0, 40.0)], max_sec=15.0)
        assert len(result) == 3
        assert result[0][0] == pytest.approx(0.0)
        assert result[-1][1] == pytest.approx(40.0)
        # Contiguous and all ≤ 15s, roughly equal.
        for (s, e), (ns, _) in zip(result, result[1:]):
            assert e == pytest.approx(ns)
        for start, end in result:
            assert end - start <= 15.0 + 1e-9
            assert end - start == pytest.approx(40.0 / 3, abs=0.1)

    def test_exact_multiple(self):
        result = split_max_duration([(0.0, 30.0)], max_sec=15.0)
        assert len(result) == 2
        assert result[-1] == (15.0, 30.0)

    def test_short_interval_unchanged(self):
        result = split_max_duration([(5.0, 10.0)], max_sec=15.0)
        assert result == [(5.0, 10.0)]

    def test_covers_original_range(self):
        intervals = [(0.0, 40.0), (50.0, 70.0)]
        result = split_max_duration(intervals, max_sec=15.0)
        # Check that starts/ends are contiguous within each original interval
        for orig_start, orig_end in intervals:
            chunks = [(s, e) for s, e in result if s >= orig_start and e <= orig_end]
            assert chunks[0][0] == pytest.approx(orig_start)
            assert chunks[-1][1] == pytest.approx(orig_end)

    def test_empty(self):
        assert split_max_duration([], max_sec=15.0) == []


# ---------------------------------------------------------------------------
# apply_rolls
# ---------------------------------------------------------------------------


class TestApplyRolls:
    def test_extends_both_ends(self):
        start, end = apply_rolls(5.0, 10.0, pre_roll_sec=1.0, post_roll_sec=0.5)
        assert start == pytest.approx(4.0)
        assert end == pytest.approx(10.5)

    def test_clamped_by_lower(self):
        start, end = apply_rolls(0.3, 5.0, pre_roll_sec=1.0, post_roll_sec=0.0, lower=0.0)
        assert start == pytest.approx(0.0)

    def test_clamped_by_upper(self):
        _, end = apply_rolls(5.0, 9.8, pre_roll_sec=0.0, post_roll_sec=1.0, upper=10.0)
        assert end == pytest.approx(10.0)

    def test_no_roll(self):
        assert apply_rolls(3.0, 7.0, 0.0, 0.0) == (3.0, 7.0)


# ---------------------------------------------------------------------------
# propose_segments — end-to-end with fake detector
# ---------------------------------------------------------------------------


class _FakeDetector:
    """
    Synthetic detector that injects pre-defined per-frame results.

    *face_frames* is a set of frame indices (by position in the detection
    call sequence) that should report a large face.
    """

    def __init__(self, face_t_secs: set[float], threshold: float = 0.05):
        self._face_t_secs = face_t_secs
        self._threshold = threshold
        self._call_count = 0

    def get(self, bgr_image):
        return []  # unused — we patch detect_timeline instead


def _make_fake_frames(
    face_ranges: list[tuple[float, float]],
    duration: float,
    step: float = 0.5,
) -> list[FrameDetection]:
    """
    Build fabricated FrameDetection list for *duration* with *step* spacing.
    Frames whose time falls within any of *face_ranges* get a large face box.
    """
    frames: list[FrameDetection] = []
    t = 0.0
    while t <= duration + 1e-9:
        t_clamped = min(t, duration)
        in_face = any(start <= t_clamped < end for start, end in face_ranges)
        frames.append(
            FrameDetection(
                t_sec=t_clamped,
                faces=[_large_fb()] if in_face else [],
            )
        )
        t += step
    return frames


def test_propose_segments_end_to_end(monkeypatch):
    """
    Faces present 0–9s and 15–25s in a 30s video.
    Expected partition (with default lead_in=0.5, bridge_gaps=1):
    - [0, 9] swap   (lead_in clamps to 0)
    - [9, 14.5] keep
    - [14.5, 25] swap  (15-0.5=14.5)
    - [25, 30] keep
    All chunks ≤15s.
    """
    duration = 30.0
    face_ranges = [(0.0, 9.0), (15.0, 25.0)]
    fake_frames = _make_fake_frames(face_ranges, duration=duration, step=0.5)

    # Monkeypatch detect_timeline to return our fabricated frames
    import app.face as face_mod

    monkeypatch.setattr(face_mod, "detect_timeline", lambda *a, **kw: fake_frames)

    segments = propose_segments(
        "fake.mp4",
        duration_sec=duration,
        sample_fps=2.0,
        min_face_px=120,
        lead_in_sec=0.5,
        bridge_gaps=1,
        max_segment_sec=15.0,
    )

    # --- Structural assertions ---
    assert len(segments) >= 2, "must have at least one swap and one keep"

    # Sorted and contiguous
    for i in range(len(segments) - 1):
        assert segments[i].end_sec == pytest.approx(segments[i + 1].start_sec, abs=1e-6), (
            f"gap between segment {i} and {i+1}"
        )

    # Covers [0, 30]
    assert segments[0].start_sec == pytest.approx(0.0, abs=1e-6)
    assert segments[-1].end_sec == pytest.approx(30.0, abs=1e-6)

    # All actions are either swap or keep
    assert all(s.action in ("swap", "keep") for s in segments)
    assert all(s.has_face == (s.action == "swap") for s in segments)

    # swap segments must exist
    swap_segs = [s for s in segments if s.action == "swap"]
    assert len(swap_segs) >= 1

    # No swap chunk exceeds 15s
    for s in swap_segs:
        assert s.end_sec - s.start_sec <= 15.0 + 1e-9

    # The first swap segment should start at or before where faces first appear (0s)
    # with lead_in clamped to 0
    assert swap_segs[0].start_sec == pytest.approx(0.0, abs=1.0)

    # The second face region (15–25s) should produce a swap segment starting ≤15s
    if len(swap_segs) >= 2:
        assert swap_segs[-1].start_sec < 15.0 + 0.01  # lead_in pulled it back


def test_propose_segments_no_faces(monkeypatch):
    """When no faces are detected the entire video is a single keep segment."""
    import app.face as face_mod

    frames = _make_fake_frames([], duration=10.0, step=0.5)
    monkeypatch.setattr(face_mod, "detect_timeline", lambda *a, **kw: frames)

    segments = propose_segments("fake.mp4", duration_sec=10.0)
    assert len(segments) == 1
    assert segments[0].action == "keep"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(10.0)


def test_propose_segments_contiguous_and_complete(monkeypatch):
    """Partition must be gap-free and cover [0, duration] exactly."""
    import app.face as face_mod

    duration = 20.0
    frames = _make_fake_frames([(3.0, 8.0), (12.0, 17.0)], duration=duration, step=0.5)
    monkeypatch.setattr(face_mod, "detect_timeline", lambda *a, **kw: frames)

    segments = propose_segments("fake.mp4", duration_sec=duration, lead_in_sec=0.5)

    assert segments[0].start_sec == pytest.approx(0.0, abs=1e-6)
    assert segments[-1].end_sec == pytest.approx(duration, abs=1e-6)

    for i in range(len(segments) - 1):
        assert segments[i].end_sec == pytest.approx(segments[i + 1].start_sec, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration smoke test (requires InsightFace model + ffmpeg)
# ---------------------------------------------------------------------------


def test_detect_timeline_grabs_skipped_frames_without_retrieving(monkeypatch):
    """Sampled frames are retrieved/decoded; skipped frames are only grabbed."""

    class _FakeCapture:
        def __init__(self):
            self.frame_count = 31
            self.current = -1
            self.grab_calls = 0
            self.retrieve_calls = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return self.frame_count
            return 0

        def grab(self):
            if self.current + 1 >= self.frame_count:
                return False
            self.current += 1
            self.grab_calls += 1
            return True

        def retrieve(self):
            self.retrieve_calls += 1
            return True, f"frame-{self.current}"

        def release(self):
            pass

    fake_cap = _FakeCapture()
    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        VideoCapture=lambda _path: fake_cap,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    class _Detector:
        def __init__(self):
            self.frames = []

        def get(self, frame):
            self.frames.append(frame)
            return []

    detector = _Detector()

    frames = detect_timeline("fake.mp4", sample_fps=2.0, detector=detector)

    assert fake_cap.grab_calls == 31
    assert fake_cap.retrieve_calls == 3
    assert detector.frames == ["frame-0", "frame-15", "frame-30"]
    assert [fd.t_sec for fd in frames] == [0.0, 0.5, 1.0]


def test_detect_timeline_integration(tmp_path):
    """
    Generate a 1-second synthetic testsrc clip via ffmpeg and run
    detect_timeline on it.  Asserts the correct number of sampled frames
    (faces may be 0 — that's fine).  Skipped if InsightFace or ffmpeg is
    unavailable.
    """
    import subprocess

    ffmpeg = "/usr/bin/ffmpeg"
    try:
        r = subprocess.run(
            [ffmpeg, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if r.returncode != 0:
            pytest.skip("ffmpeg not available")
    except FileNotFoundError:
        pytest.skip("ffmpeg not found")

    clip = str(tmp_path / "smoke.mp4")
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=1:size=320x240:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-an",
            clip,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        pytest.skip("Could not generate synthetic clip")

    try:
        frames = detect_timeline(clip, sample_fps=2.0)
    except Exception as exc:
        pytest.skip(f"InsightFace initialisation failed: {exc}")

    # At sample_fps=2.0 over a 1-second clip @ 30fps, we expect ~2 sampled frames.
    assert 1 <= len(frames) <= 4, f"Unexpected frame count: {len(frames)}"
    # Each FrameDetection has a valid timestamp
    for fd in frames:
        assert fd.t_sec >= 0.0
        assert isinstance(fd.faces, list)


# ---------------------------------------------------------------------------
# drop_short_intervals + even split + min_segment_sec in propose_segments
# ---------------------------------------------------------------------------

class TestDropShortIntervals:
    def test_drops_below_min(self):
        ivs = [(0.0, 0.5), (2.0, 6.0), (10.0, 10.4)]
        assert drop_short_intervals(ivs, min_sec=2.0) == [(2.0, 6.0)]

    def test_keeps_at_min(self):
        assert drop_short_intervals([(0.0, 2.0)], min_sec=2.0) == [(0.0, 2.0)]

    def test_empty(self):
        assert drop_short_intervals([], min_sec=2.0) == []


class TestEvenSplit:
    def test_no_tiny_tail(self):
        # 16s with max 15 -> two ~8s chunks, NOT 15 + 1
        chunks = split_max_duration([(0.0, 16.0)], max_sec=15.0)
        assert len(chunks) == 2
        for s, e in chunks:
            assert e - s <= 15.0
            assert e - s >= 7.0  # even split, no tiny tail
        assert chunks[0][0] == 0.0 and abs(chunks[-1][1] - 16.0) < 1e-6

    def test_short_interval_untouched(self):
        assert split_max_duration([(0.0, 5.0)], max_sec=15.0) == [(0.0, 5.0)]


class TestProposeMinSegment:
    def test_short_swap_blips_dropped(self):
        # 30s video; faces present briefly 10.0-10.4 (blip) and solidly 15-25.
        frames = []
        t = 0.0
        while t < 30.0:
            faces = []
            if 10.0 <= t < 10.4 or 15.0 <= t < 25.0:
                faces = [FaceBox(0, 0, 200, 200, 0.9)]
            frames.append(FrameDetection(t_sec=t, faces=faces))
            t += 0.5

        class _FakeDetector:
            pass

        import app.face as face_mod
        orig = face_mod.detect_timeline
        face_mod.detect_timeline = lambda *a, **k: frames
        try:
            segs = propose_segments(
                "x.mp4", duration_sec=30.0, min_segment_sec=2.0, lead_in_sec=0.0
            )
        finally:
            face_mod.detect_timeline = orig

        swaps = [s for s in segs if s.action == "swap"]
        # The 0.4s blip is dropped; only the 15-25 appearance remains.
        assert len(swaps) == 1
        assert swaps[0].start_sec == pytest.approx(15.0, abs=0.6)
        assert swaps[0].end_sec == pytest.approx(25.0, abs=0.6)
        # Partition still covers [0, 30] contiguously.
        assert segs[0].start_sec == 0.0
        assert segs[-1].end_sec == pytest.approx(30.0)
        for a, b in zip(segs, segs[1:]):
            assert a.end_sec == pytest.approx(b.start_sec)
