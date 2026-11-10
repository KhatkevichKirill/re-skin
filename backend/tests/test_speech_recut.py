"""Re-cut whole-video localisation segments at speech pauses (AICP-347).

The rule (app.speech_recut.recut_at_speech_pauses) is pure, so the acceptance
case runs on the read-only copy of prod project 46cc6443 already used by
AICP-335/340 (tests/fixtures/localisation_46cc6443.json: its transcript and its
hand cut 0-12.13-26.03-35.9-49.45-57.7-65.95-78.85) and the scene cuts
PySceneDetect found on that source (tests/fixtures/scene_cuts_5b1b9b67.json).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from app import scene as scene_mod
from app.models import VideoProject
from app.pipeline_v2 import whole_video_floor_sec, whole_video_ranges, whole_video_target_sec
from app.speech_recut import (
    EDGE_TOL_SEC,
    apply_plan,
    clean_lines,
    lines_containing,
    plan_fingerprint,
    recut_at_speech_pauses,
    speech_pauses,
)

# Fixtures and helpers of the pipeline suite (the session-scoped engine is
# shared). Only fixtures/helpers are imported, never its Test* classes.
from tests.test_pipeline_v2 import (  # noqa: F401
    _create_project,
    db_engine,
    db_session,
    _cleanup_tmp,
    patch_get_session,
    tmp_path,
)

FIX = Path(__file__).parent / "fixtures"
DATA = json.loads((FIX / "localisation_46cc6443.json").read_text("utf-8"))
CUTS = json.loads((FIX / "scene_cuts_5b1b9b67.json").read_text("utf-8"))
LINES = DATA["transcript_v2"]["lines"]
DURATION = DATA["duration_sec"]
SEGMENTS = [dict(s, action="swap") for s in DATA["segments"]]
CLEANED_CUTS = [s for s, _e in CUTS["cleaned"]["scenes"]][1:]
TARGET = 14.72
FLOOR = 4.0


def _seg(start, end, action="swap", **kw):
    return dict(start_sec=start, end_sec=end, action=action, **kw)


def _line(lid, start, end):
    return {"id": lid, "start": start, "end": end, "text": f"line {lid}"}


def _strictly_inside(t, line):
    return line["start"] < t < line["end"]


def _assert_partition(ranges, *, duration, target, floor=FLOOR, count=None):
    assert ranges[0][0] == pytest.approx(0.0)
    assert ranges[-1][1] == pytest.approx(duration)
    for (_s, e), (s2, _e2) in zip(ranges, ranges[1:]):
        assert s2 == e
    for s, e in ranges:
        assert floor - 1e-6 <= e - s <= target + 1e-6, (s, e)
    if count is not None:
        assert len(ranges) == count


# ---------------------------------------------------------------------------
# Acceptance: 46cc6443
# ---------------------------------------------------------------------------


class TestAcceptance46cc6443:
    def test_the_hand_cut_crosses_lines_5_10_16_20(self):
        """The issue's premise, checked on the fixture."""
        by_id = {ln["id"]: ln for ln in LINES}
        bounds = [s["end_sec"] for s in SEGMENTS[:-1]]
        for lid in (5, 10, 16, 20):
            assert any(_strictly_inside(b, by_id[lid]) for b in bounds), lid

    @pytest.mark.parametrize("cuts", [None, CLEANED_CUTS], ids=["no-scene-cuts", "scene-cuts"])
    def test_no_boundary_inside_a_line_and_every_segment_within_target(self, cuts):
        plan = recut_at_speech_pauses(
            SEGMENTS, LINES, duration_sec=DURATION, target_sec=TARGET,
            floor_sec=FLOOR, scene_cuts=cuts,
        )
        assert plan.feasible and plan.changed
        assert plan.warnings == []
        _assert_partition(plan.ranges, duration=DURATION, target=TARGET, count=7)
        by_id = {ln["id"]: ln for ln in LINES}
        new_bounds = [b.new_sec for b in plan.boundaries]
        for lid in (5, 10, 16, 20):
            assert not any(_strictly_inside(b, by_id[lid]) for b in new_bounds), lid
        # Stronger than the acceptance: inside no line at all, tolerance 0.
        for b in new_bounds:
            assert not any(_strictly_inside(b, ln) for ln in LINES), b
        assert new_bounds == pytest.approx([13.0, 19.3, 33.3, 43.1, 50.1, 64.6])
        assert plan.scene_cuts_used is bool(cuts)

    def test_the_analyze_cut_of_the_same_source_recuts_too(self):
        """Analyze (AICP-340) on 5b1b9b67, then re-cut with that transcript."""
        cut = whole_video_ranges(
            [tuple(r) for r in CUTS["scenes"]],
            duration_sec=CUTS["duration_sec"], target_sec=TARGET,
        )
        segs = [_seg(s, e) for s, e in cut]
        plan = recut_at_speech_pauses(
            segs, LINES, duration_sec=CUTS["duration_sec"], target_sec=TARGET,
            floor_sec=FLOOR, scene_cuts=[s for s, _e in CUTS["scenes"]][1:],
        )
        assert plan.feasible and plan.warnings == []
        _assert_partition(plan.ranges, duration=CUTS["duration_sec"], target=TARGET, count=len(cut))
        for b in plan.boundaries:
            assert not any(_strictly_inside(b.new_sec, ln) for ln in LINES)

    def test_the_project_target_and_floor_are_the_whole_video_ones(self):
        assert whole_video_target_sec(None) == pytest.approx(TARGET)
        assert whole_video_floor_sec() == pytest.approx(FLOOR)

    def test_recut_is_idempotent(self):
        plan = recut_at_speech_pauses(
            SEGMENTS, LINES, duration_sec=DURATION, target_sec=TARGET, floor_sec=FLOOR,
        )
        again = recut_at_speech_pauses(
            [_seg(s, e) for s, e in plan.ranges], LINES, duration_sec=DURATION,
            target_sec=TARGET, floor_sec=FLOOR,
        )
        assert not again.changed
        assert again.ranges == plan.ranges

    def test_every_move_is_reported(self):
        plan = recut_at_speech_pauses(
            SEGMENTS, LINES, duration_sec=DURATION, target_sec=TARGET, floor_sec=FLOOR,
        )
        moved = {round(b.old_sec, 2): b for b in plan.boundaries if b.moved}
        # 35.9 was already safe (0.1 s past line 14's start is within the
        # transcript's tolerance) but had to move so that 26.03 could leave the
        # overlapping block of lines 9-11: the plan says so.
        assert set(moved) == {12.13, 26.03, 35.9, 49.45, 57.7, 65.95}
        assert moved[35.9].old_inside_line_ids == ()
        assert moved[26.03].old_inside_line_ids == (9, 10)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


class TestPauses:
    def test_overlapping_lines_merge_and_touching_lines_leave_a_zero_width_pause(self):
        lines = clean_lines([_line(1, 1, 3), _line(2, 2.5, 4), _line(3, 4, 5), _line(4, 6, 7)])
        assert speech_pauses(lines, 8.0) == [(0.0, 1.0), (4.0, 4.0), (5.0, 6.0), (7.0, 8.0)]

    def test_inside_uses_the_transcript_tolerance(self):
        lines = clean_lines([_line(1, 10.0, 20.0)])
        assert lines_containing(10.1, lines) == ()
        assert lines_containing(10.0 + EDGE_TOL_SEC + 0.01, lines) == (1,)
        assert lines_containing(15.0, lines, tol=0.0) == (1,)

    def test_malformed_lines_are_ignored(self):
        lines = clean_lines([
            {"id": 1, "start": "x", "end": 2}, {"id": 2, "start": 3, "end": 3},
            {"id": 3, "start": float("nan"), "end": 4}, {"id": 4, "start": 5, "end": 6},
        ])
        assert [lid for _s, _e, lid in lines] == [4]


class TestRule:
    def test_prefers_a_scene_cut_inside_the_pause(self):
        lines = [_line(1, 0.5, 5.0), _line(2, 6.0, 11.5)]
        segs_inside = [_seg(0, 8.0), _seg(8.0, 12)]  # 8.0 is inside line 2
        with_cut = recut_at_speech_pauses(
            segs_inside, lines, duration_sec=12, target_sec=TARGET, floor_sec=FLOOR,
            scene_cuts=[5.3],
        )
        assert with_cut.boundaries[0].new_sec == pytest.approx(5.3)
        assert with_cut.boundaries[0].source == "scene_cut"
        no_cut = recut_at_speech_pauses(
            segs_inside, lines, duration_sec=12, target_sec=TARGET, floor_sec=FLOOR,
        )
        assert no_cut.boundaries[0].new_sec == pytest.approx(5.5)
        assert no_cut.boundaries[0].source == "pause_middle"
        assert no_cut.boundaries[0].old_inside_line_ids == (2,)

    def test_a_scene_cut_just_off_a_touch_point_counts_as_inside_the_pause(self):
        segs = [_seg(0, 6.5), _seg(6.5, 12)]
        lines = [_line(1, 0.5, 5.0), _line(2, 5.0, 11.5)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=12, target_sec=TARGET, floor_sec=FLOOR, scene_cuts=[5.1],
        )
        assert plan.boundaries[0].new_sec == pytest.approx(5.1)
        assert plan.boundaries[0].source == "scene_cut"
        assert plan.boundaries[0].new_inside_line_ids == ()

    def test_a_scene_cut_outside_every_pause_is_not_used(self):
        segs = [_seg(0, 8.0), _seg(8.0, 12)]
        lines = [_line(1, 0.5, 5.0), _line(2, 6.0, 11.5)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=12, target_sec=TARGET, floor_sec=FLOOR, scene_cuts=[8.0],
        )
        assert plan.boundaries[0].new_sec == pytest.approx(5.5)

    def test_uses_the_edge_of_a_pause_whose_middle_breaks_the_target(self):
        """Terra §2: pause [10.2, 15] with target 11 — its middle (12.6) makes
        the first segment too long, a point near its start does not."""
        segs = [_seg(0, 10), _seg(10, 20)]
        lines = [_line(1, 0, 10.2), _line(2, 15, 20)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=20, target_sec=11, floor_sec=FLOOR,
        )
        b = plan.boundaries[0]
        assert b.moved and b.new_inside_line_ids == ()
        assert 10.2 <= b.new_sec <= 11.0
        _assert_partition(plan.ranges, duration=20, target=11)

    def test_keeps_a_boundary_no_pause_can_take_and_warns(self):
        segs = [_seg(0, 10), _seg(10, 20)]
        lines = [_line(7, 2, 18)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=20, target_sec=11, floor_sec=FLOOR,
        )
        assert plan.feasible and not plan.changed
        assert plan.ranges == [(0, 10), (10, 20)]
        assert plan.boundaries[0].new_inside_line_ids == (7,)
        assert len(plan.warnings) == 1 and "line 7" in plan.warnings[0]

    def test_keeps_the_others_moving_when_one_cannot(self):
        segs = [_seg(0, 10), _seg(10, 20), _seg(20, 30)]
        lines = [_line(1, 2, 18), _line(2, 19, 21.5), _line(3, 22, 29)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=30, target_sec=12, floor_sec=FLOOR,
        )
        first, second = plan.boundaries
        assert not first.moved and first.new_inside_line_ids == (1,)
        assert second.moved and second.new_inside_line_ids == ()
        assert len(plan.warnings) == 1
        _assert_partition(plan.ranges, duration=30, target=12)

    def test_safe_boundaries_stay_put(self):
        segs = [_seg(0, 5.5), _seg(5.5, 12)]
        lines = [_line(1, 0.5, 5.0), _line(2, 6.0, 11.5)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=12, target_sec=TARGET, floor_sec=FLOOR, scene_cuts=[5.3],
        )
        assert not plan.changed

    def test_a_current_cut_over_the_target_changes_nothing(self):
        segs = [_seg(0, 16), _seg(16, 20)]
        plan = recut_at_speech_pauses(
            segs, [_line(1, 15, 17)], duration_sec=20, target_sec=TARGET, floor_sec=FLOOR,
        )
        assert not plan.feasible and not plan.changed
        assert plan.ranges == [(0, 16), (16, 20)]
        assert "segment 1" in plan.warnings[0]

    def test_keep_segments_are_not_held_to_the_target(self):
        segs = [_seg(0, 10), _seg(10, 40, "keep")]
        lines = [_line(1, 8, 12), _line(2, 13, 39)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=40, target_sec=TARGET, floor_sec=FLOOR,
        )
        assert plan.boundaries[0].new_sec == pytest.approx(12.5)

    def test_pre_and_post_roll_count_against_the_target(self):
        segs = [_seg(0, 10, pre_roll_sec=0.0, post_roll_sec=4.0), _seg(10, 20)]
        lines = [_line(1, 1, 11.5), _line(2, 12.5, 19)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=20, target_sec=TARGET, floor_sec=FLOOR,
        )
        # 12.0 + 4s post-roll would be 16s > 14.72: no pause fits.
        assert not plan.changed and plan.warnings
        plan_no_roll = recut_at_speech_pauses(
            [_seg(0, 10), _seg(10, 20)], lines, duration_sec=20, target_sec=TARGET,
            floor_sec=FLOOR,
        )
        assert plan_no_roll.boundaries[0].new_sec == pytest.approx(12.0)

    def test_a_negative_roll_counts_against_the_floor(self):
        """Gate R2: [0,6] with a -2s pre-roll sends the model [2,6]; moving
        the boundary to 5 would send it a 3s clip."""
        lines = [_line(1, 0.2, 4.5), _line(2, 5.5, 9.8)]
        segs = [_seg(0, 6, pre_roll_sec=-2.0), _seg(6, 10)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=10, target_sec=TARGET, floor_sec=FLOOR,
        )
        assert not plan.changed and plan.warnings
        no_roll = recut_at_speech_pauses(
            [_seg(0, 6), _seg(6, 10)], lines, duration_sec=10, target_sec=TARGET,
            floor_sec=FLOOR,
        )
        # Without the roll the only pause leaves a 5s and a 5s segment.
        assert no_roll.boundaries[0].new_sec == pytest.approx(5.0)

    def test_roll_is_clamped_to_the_video_like_the_pipeline(self):
        # A 3s pre-roll on the first segment cannot reach before 0.
        segs = [_seg(0, 10, pre_roll_sec=3.0), _seg(10, 20)]
        lines = [_line(1, 1, 12), _line(2, 14.5, 19.5)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=20, target_sec=TARGET, floor_sec=FLOOR,
        )
        assert plan.boundaries[0].new_sec == pytest.approx(13.25)

    def test_floor_is_respected(self):
        segs = [_seg(0, 6), _seg(6, 12)]
        lines = [_line(1, 0.5, 2.0), _line(2, 2.5, 11.5)]
        plan = recut_at_speech_pauses(
            segs, lines, duration_sec=12, target_sec=TARGET, floor_sec=FLOOR,
        )
        # The only pause (2.0-2.5) would leave a 2.25 s first segment.
        assert not plan.changed and plan.warnings

    def test_single_segment_and_no_lines(self):
        one = recut_at_speech_pauses(
            [_seg(0, 10)], [_line(1, 1, 9)], duration_sec=10, target_sec=TARGET,
        )
        assert one.feasible and not one.changed and one.boundaries == []
        silent = recut_at_speech_pauses(
            [_seg(0, 10), _seg(10, 20)], [], duration_sec=20, target_sec=TARGET,
        )
        assert not silent.changed

    @pytest.mark.parametrize("segs", [
        [_seg(0, 10), _seg(11, 20)],
        [_seg(1, 10), _seg(10, 20)],
        [_seg(0, 10), _seg(10, 19)],
        [_seg(0, 10), _seg(10, 10)],
    ])
    def test_rejects_a_broken_partition(self, segs):
        with pytest.raises(ValueError):
            recut_at_speech_pauses(segs, [], duration_sec=20, target_sec=TARGET)

    def test_rejects_a_non_positive_target(self):
        with pytest.raises(ValueError):
            recut_at_speech_pauses([_seg(0, 10)], [], duration_sec=10, target_sec=0)

    def test_fingerprint_tracks_every_input(self):
        base = dict(duration_sec=DURATION, target_sec=TARGET, floor_sec=FLOOR, scene_cuts=None)
        fp = plan_fingerprint(SEGMENTS, LINES, **base)
        assert fp == plan_fingerprint(SEGMENTS, LINES, **base)
        assert fp != plan_fingerprint(SEGMENTS, LINES, **dict(base, scene_cuts=CLEANED_CUTS))
        assert fp != plan_fingerprint(SEGMENTS, LINES[:-1], **base)
        moved = [dict(s) for s in SEGMENTS]
        moved[0]["end_sec"] = 12.0
        assert fp != plan_fingerprint(moved, LINES, **base)
        assert fp != plan_fingerprint(SEGMENTS, LINES, **dict(base, target_sec=10.0))

    def test_apply_plan_writes_in_place(self):
        class Row:
            def __init__(self, s, e):
                self.start_sec, self.end_sec = s, e

        rows = [Row(s["start_sec"], s["end_sec"]) for s in SEGMENTS]
        plan = recut_at_speech_pauses(
            SEGMENTS, LINES, duration_sec=DURATION, target_sec=TARGET, floor_sec=FLOOR,
        )
        assert apply_plan(rows, plan) == 7
        assert [(r.start_sec, r.end_sec) for r in rows] == plan.ranges
        with pytest.raises(ValueError):
            apply_plan(rows[:-1], plan)


# ---------------------------------------------------------------------------
# Scene-cut cache
# ---------------------------------------------------------------------------


class TestSceneCache:
    def test_round_trip(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"video")
        cache = tmp_path / "p" / "scene_cuts.json"
        assert scene_mod.write_scene_cache(
            str(cache), source_path=str(src), ranges=[(0, 5), (5, 9.5), (9.5, 12)],
            min_scene_len_sec=4.0,
        )
        assert scene_mod.read_scene_cuts(
            str(cache), source_path=str(src), min_scene_len_sec=4.0
        ) == [5.0, 9.5]
        assert not list((tmp_path / "p").glob(".scene_cuts.*"))

    def test_single_shot_is_an_empty_answer(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"video")
        cache = tmp_path / "scene_cuts.json"
        scene_mod.write_scene_cache(str(cache), source_path=str(src), ranges=[], min_scene_len_sec=4.0)
        assert scene_mod.read_scene_cuts(str(cache), source_path=str(src), min_scene_len_sec=4.0) == []

    def test_a_changed_source_or_setting_is_not_trusted(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"video")
        cache = tmp_path / "scene_cuts.json"
        scene_mod.write_scene_cache(
            str(cache), source_path=str(src), ranges=[(0, 5), (5, 9)], min_scene_len_sec=4.0,
        )
        assert scene_mod.read_scene_cuts(str(cache), source_path=str(src), min_scene_len_sec=3.0) is None
        src.write_bytes(b"another video")
        assert scene_mod.read_scene_cuts(str(cache), source_path=str(src), min_scene_len_sec=4.0) is None
        assert scene_mod.read_scene_cuts(str(cache), source_path=None, min_scene_len_sec=4.0) is None

    def test_missing_or_corrupt_cache_is_unknown(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"video")
        cache = tmp_path / "scene_cuts.json"
        assert scene_mod.read_scene_cuts(str(cache), source_path=str(src), min_scene_len_sec=4.0) is None
        cache.write_text("{not json")
        assert scene_mod.read_scene_cuts(str(cache), source_path=str(src), min_scene_len_sec=4.0) is None

    def test_write_never_raises(self, tmp_path):
        assert not scene_mod.write_scene_cache(
            str(tmp_path / "x.json"), source_path=str(tmp_path / "missing.mp4"),
            ranges=[], min_scene_len_sec=4.0,
        )
        blocker = tmp_path / "file"
        blocker.write_text("")
        src = tmp_path / "source.mp4"
        src.write_bytes(b"v")
        assert not scene_mod.write_scene_cache(
            str(blocker / "sub" / "x.json"), source_path=str(src), ranges=[], min_scene_len_sec=4.0,
        )


class TestAnalyzeRecordsSceneCuts:
    def _probe(self, monkeypatch, duration, fps=30.0):
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(duration_sec=duration, width=1080, height=1920,
                                 fps=fps, aspect_ratio="9:16", has_audio=True),
        )

    def _project(self, db_session, source, scope):
        pid = _create_project(db_session, str(source), project_type="localisation")
        p = db_session.get(VideoProject, pid)
        p.localisation_scope = scope
        db_session.commit()
        return pid

    def test_whole_scope_analysis_writes_the_cache_and_cuts_as_before(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project
        from app.storage import scene_cuts_cache_path

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._project(db_session, source, "whole")
        self._probe(monkeypatch, CUTS["duration_sec"])
        monkeypatch.setattr(
            pv2.scene_mod, "detect_scenes",
            lambda *a, **k: [tuple(r) for r in CUTS["scenes"]],
        )
        analyze_project(pid)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, pid)
            segs = sorted(project.segments, key=lambda d: d.index)
            assert [round(d.end_sec, 2) for d in segs] == [
                12.13, 26.03, 35.9, 49.45, 57.7, 65.95, 78.88,
            ]
            cuts = scene_mod.read_scene_cuts(
                scene_cuts_cache_path(pid), source_path=project.source_local_path,
                min_scene_len_sec=4.0,
            )
        assert cuts == pytest.approx([s for s, _e in CUTS["scenes"]][1:])

    def test_a_cache_failure_does_not_fail_analysis(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.models import ProjectStatus
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._project(db_session, source, "whole")
        self._probe(monkeypatch, 30.0)
        monkeypatch.setattr(pv2.scene_mod, "detect_scenes", lambda *a, **k: [(0.0, 12.0), (12.0, 30.0)])
        monkeypatch.setattr(pv2.scene_mod, "write_scene_cache", lambda *a, **k: False)
        analyze_project(pid)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            assert s.get(VideoProject, pid).status == ProjectStatus.ready

    def test_hook_scope_analysis_writes_no_cache(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        from app.pipeline_v2 import analyze_project
        from app.storage import scene_cuts_cache_path

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._project(db_session, source, "hook")
        self._probe(monkeypatch, 30.0)
        analyze_project(pid)
        assert not os.path.exists(scene_cuts_cache_path(pid))
