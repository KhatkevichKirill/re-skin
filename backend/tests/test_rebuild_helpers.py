"""Pure helpers of the rebuild project type (app/rebuild.py, design §5.1, §6).

No DB, no network: folding, chunking, brand-card keeps, the shots inside a
chunk, prompt rendering, the duration request, frame quantisation and the fit,
cut-timing QA and the trailing end card.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import ai_models  # noqa: E402
from app import rebuild as rb  # noqa: E402

CAP = rb.REBUILD_CHUNK_MAX_SEC


def shots_from(durations, *, force=(), brand=()):
    out, t = [], 0.0
    for i, d in enumerate(durations):
        out.append(rb.Shot(index=i, start=round(t, 6), end=round(t + d, 6),
                           force_anchor=i in force, is_brand_card=i in brand))
        t += d
    return out


def assert_partition(groups, shots, cap=CAP):
    """Groups cover the shots exactly, in order, each <= cap, never empty, and
    every member points at its own group's anchor."""
    assert groups, "no groups"
    assert groups[0].start == pytest.approx(shots[0].start)
    assert groups[-1].end == pytest.approx(shots[-1].end)
    for a, b in zip(groups, groups[1:]):
        assert a.end == pytest.approx(b.start)
    seen = []
    for g in groups:
        assert g.members, "empty group"
        assert g.anchor in g.members
        assert g.duration <= cap + 1e-6
        for m in g.members:
            if not seen or seen[-1] != m:
                seen.append(m)
    assert seen == [s.index for s in shots]


# ---------------------------------------------------------------------------
# Shot normalisation
# ---------------------------------------------------------------------------


def test_normalize_shot_ranges_repairs_detector_output():
    assert rb.normalize_shot_ranges([], 5.0) == [(0.0, 5.0)]
    # Last scene ends a frame short of the probed duration; a cut at 0 is no cut.
    got = rb.normalize_shot_ranges([(0.0, 1.5), (1.5, 3.0), (3.0, 4.96)], 5.0)
    assert got == [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0)]
    with pytest.raises(ValueError):
        rb.normalize_shot_ranges([], 0)


def test_detect_shots_uses_the_fine_minimum(monkeypatch):
    seen = {}

    def fake(path, *, fps, min_scene_len_sec):
        seen["min"] = min_scene_len_sec
        return [(0.0, 1.0), (1.0, 2.9)]

    from app import scene

    monkeypatch.setattr(scene, "detect_scenes", fake)
    assert rb.detect_shots("x.mp4", fps=30, duration_sec=3.0) == [(0.0, 1.0), (1.0, 3.0)]
    assert seen["min"] == rb.REBUILD_SHOT_MIN_SEC == 0.4


# ---------------------------------------------------------------------------
# Folding (§5.1.3) — the design's test list
# ---------------------------------------------------------------------------


def test_long_shots_are_their_own_anchors():
    shots = shots_from([1.0, 2.0, 0.7])
    groups = rb.effective_shots(shots)
    assert [g.members for g in groups] == [(0,), (1,), (2,)]
    assert rb.folded_into_map(groups) == {0: None, 1: None, 2: None}


def test_consecutive_shorts_join_the_next_anchor():
    shots = shots_from([1.0, 0.3, 0.5, 0.6, 2.0])
    groups = rb.effective_shots(shots)
    assert [(g.anchor, g.members) for g in groups] == [(0, (0,)), (4, (1, 2, 3, 4))]
    assert rb.folded_into_map(groups)[2] == 4
    assert_partition(groups, shots)


def test_shorts_at_both_ends():
    shots = shots_from([0.3, 0.4, 2.0, 1.0, 0.5, 0.2])
    groups = rb.effective_shots(shots)
    # Leading shorts join the first anchor; trailing ones join the last group.
    assert [(g.anchor, g.members) for g in groups] == [(2, (0, 1, 2)), (3, (3, 4, 5))]
    assert_partition(groups, shots)


def test_all_short_twenty_second_video_is_capped():
    shots = shots_from([0.5] * 40)  # 20 s, no anchor at all
    groups = rb.effective_shots(shots)
    assert_partition(groups, shots)
    assert len(groups) == 2
    assert all(g.duration <= CAP for g in groups)
    # Anchor of a buffer group = its longest shot, ties -> the earliest.
    assert groups[0].anchor == 0


def test_all_short_video_under_the_cap_is_one_group_anchored_on_the_longest():
    shots = shots_from([0.3, 0.6, 0.5])
    groups = rb.effective_shots(shots)
    assert [(g.anchor, g.members) for g in groups] == [(1, (0, 1, 2))]


def test_sixteen_seconds_of_shorts_before_an_anchor():
    shots = shots_from([0.5] * 32 + [3.0])
    groups = rb.effective_shots(shots)
    assert_partition(groups, shots)
    assert groups[-1].anchor == 32
    assert all(g.duration <= CAP for g in groups)


def test_overlong_anchor_is_split_into_equal_pieces():
    shots = shots_from([2.0, 31.0, 1.0])
    groups = rb.effective_shots(shots)
    pieces = [g for g in groups if g.anchor == 1]
    assert [g.piece for g in pieces] == [(1, 3), (2, 3), (3, 3)]
    assert all(g.duration == pytest.approx(31.0 / 3) for g in pieces)
    assert pieces[0].start == pytest.approx(2.0) and pieces[-1].end == pytest.approx(33.0)
    assert_partition(groups, shots)


def test_overlong_anchor_after_shorts_leaves_the_shorts_their_own_group():
    shots = shots_from([0.3, 0.4, 20.0])
    groups = rb.effective_shots(shots)
    assert groups[0].members == (0, 1) and groups[0].anchor == 1
    assert [g.piece for g in groups[1:]] == [(1, 2), (2, 2)]


def test_anchor_plus_buffer_over_the_cap_stands_alone():
    shots = shots_from([0.5] * 4 + [14.5])
    groups = rb.effective_shots(shots)
    assert [(g.anchor, g.members) for g in groups] == [(0, (0, 1, 2, 3)), (4, (4,))]


def test_fourteen_and_a_half_second_anchor_then_one_trailing_short():
    shots = shots_from([14.5, 0.3])
    groups = rb.effective_shots(shots)
    # 14.8 > cap: the trailing short becomes its own group.
    assert [(g.anchor, g.members) for g in groups] == [(0, (0,)), (1, (1,))]
    assert_partition(groups, shots)


def test_fourteen_and_a_half_second_anchor_then_several_trailing_shorts():
    shots = shots_from([14.5, 0.3, 0.2, 0.6])
    groups = rb.effective_shots(shots)
    assert [(g.anchor, g.members) for g in groups] == [(0, (0,)), (3, (1, 2, 3))]
    assert_partition(groups, shots)


def test_trailing_shorts_join_the_last_group_within_the_cap():
    shots = shots_from([10.0, 0.3, 0.2])
    groups = rb.effective_shots(shots)
    assert [(g.anchor, g.members) for g in groups] == [(0, (0, 1, 2))]


def test_unfold_then_rechunk():
    base = shots_from([1.0, 0.5, 2.0])
    assert [g.members for g in rb.effective_shots(base)] == [(0,), (1, 2)]
    unfolded = shots_from([1.0, 0.5, 2.0], force=(1,))
    groups = rb.effective_shots(unfolded)
    assert [g.members for g in groups] == [(0,), (1,), (2,)]
    _, plan = rb.plan_segments(unfolded)
    assert [len(p.groups) for p in plan] == [3]


def test_folding_is_deterministic_and_rejects_gaps():
    shots = shots_from([0.3, 1.0, 0.2, 0.9])
    assert rb.effective_shots(shots) == rb.effective_shots(list(shots))
    gap = [rb.Shot(0, 0.0, 1.0), rb.Shot(1, 1.5, 2.0)]
    with pytest.raises(ValueError):
        rb.effective_shots(gap)


@pytest.mark.parametrize("seed", range(25))
def test_random_partitions_hold_the_invariants(seed):
    import random

    rnd = random.Random(seed)
    durations = [round(rnd.choice([0.2, 0.4, 0.6, 0.69, 0.7, 1.2, 3.0, 9.0, 16.0, 40.0]), 2)
                 for _ in range(rnd.randint(1, 60))]
    shots = shots_from(durations)
    groups = rb.effective_shots(shots)
    assert_partition(groups, shots)
    chunks = rb.build_chunks(groups)
    assert chunks[0].start == pytest.approx(0.0)
    assert chunks[-1].end == pytest.approx(sum(durations), abs=1e-4)
    for c in chunks:
        assert c.duration <= CAP + 1e-6
        assert len(c.groups) <= rb.REBUILD_CHUNK_MAX_SHOTS


# ---------------------------------------------------------------------------
# Chunks (§5.1.4) and brand-card keeps (§5.1.5)
# ---------------------------------------------------------------------------


def test_build_chunks_respects_length_and_group_caps():
    groups = rb.effective_shots(shots_from([1.0] * 20))
    chunks = rb.build_chunks(groups)
    assert [len(c.groups) for c in chunks] == [8, 8, 4]
    groups = rb.effective_shots(shots_from([5.0, 5.0, 4.72, 0.9, 3.0]))
    chunks = rb.build_chunks(groups)
    assert [(c.start, c.end) for c in chunks] == [(0.0, pytest.approx(14.72)),
                                                  (pytest.approx(14.72), pytest.approx(18.62))]


def test_build_chunks_never_splits_a_group_and_has_no_tail_rebalance():
    groups = rb.effective_shots(shots_from([14.0, 1.0]))
    chunks = rb.build_chunks(groups)
    # A 1 s tail stays a 1 s chunk (requested at the model minimum, trimmed by the fit).
    assert [c.duration for c in chunks] == [pytest.approx(14.0), pytest.approx(1.0)]
    with pytest.raises(ValueError):
        rb.build_chunks([rb.ShotGroup(0, (0,), 0.0, 20.0)])


def test_brand_card_keep_recut_with_a_middle_keep_and_an_end_card():
    shots = shots_from([3.0, 2.0, 3.0, 4.0, 2.0, 1.5], brand=(1, 4, 5))
    _, plan = rb.plan_segments(shots, keep_brand_cards=True)
    assert [(p.action, p.start, p.end) for p in plan] == [
        ("swap", 0.0, 3.0), ("keep", 3.0, 5.0), ("swap", 5.0, 12.0), ("keep", 12.0, 15.5),
    ]
    _, plain = rb.plan_segments(shots)
    assert {p.action for p in plain} == {"swap"}


def test_trailing_end_card_only_replaces_the_final_brand_keeps():
    segs = [("a", "swap", False), ("b", "keep", True), ("c", "swap", False),
            ("d", "keep", True), ("e", "keep", True)]
    assert rb.trailing_end_card(segs) == ["d", "e"]
    assert rb.trailing_end_card(segs[:3]) == []
    assert rb.trailing_end_card([("x", "keep", False)]) == []
    plan = rb.default_stitch_plan_with_end_card(["a", "b", "c", "d", "e"], ["d", "e"], "lib1")
    assert plan == [
        {"kind": "segment", "segment_def_id": "a"},
        {"kind": "segment", "segment_def_id": "b"},  # the middle keep stays in place
        {"kind": "segment", "segment_def_id": "c"},
        {"kind": "library", "clip_id": "lib1"},
    ]
    assert rb.default_stitch_plan_with_end_card(["a"], ["a"], None) is None
    assert rb.default_stitch_plan_with_end_card(["a"], [], "lib1") is None


# ---------------------------------------------------------------------------
# Shots inside a chunk (§6.2), sheet, image count
# ---------------------------------------------------------------------------


def spans(*items):
    return [rb.ShotSpan(shot_id=i, start=s, end=e, folded_into=f) for i, s, e, f in items]


def test_chunk_pieces_clip_relative_and_fold():
    shots = spans(("s0", 0.0, 2.0, None), ("s1", 2.0, 2.4, "s2"), ("s2", 2.4, 5.0, None),
                  ("s3", 5.0, 9.0, None))
    pieces = rb.chunk_shot_pieces(0.0, 9.0, shots)
    assert [(p.shot_id, p.rel_start, p.rel_end) for p in pieces] == [
        ("s0", 0.0, 2.0), ("s2", 2.0, 5.0), ("s3", 5.0, 9.0)]
    assert rb.planned_cuts(pieces) == [2.0, 5.0]


def test_long_shot_split_across_two_chunks_appears_in_both():
    shots = spans(("a", 0.0, 4.0, None), ("b", 4.0, 20.0, None))
    first = rb.chunk_shot_pieces(0.0, 12.0, shots)
    second = rb.chunk_shot_pieces(12.0, 20.0, shots)
    assert [(p.shot_id, p.rel_start, p.rel_end) for p in first] == [("a", 0, 4), ("b", 4, 12)]
    assert [(p.shot_id, p.rel_start, p.rel_end) for p in second] == [("b", 0, 8)]


def test_operator_boundary_sliver_merges_into_its_neighbour():
    shots = spans(("a", 0.0, 3.0, None), ("b", 3.0, 6.0, None))
    # Boundary at 2.8: the chunk starting there gets a 0.2 s sliver of "a".
    pieces = rb.chunk_shot_pieces(2.8, 6.0, shots)
    assert [(p.shot_id, p.rel_start, p.rel_end) for p in pieces] == [("b", 0.0, pytest.approx(3.2))]
    # A sliver at the END merges backwards.
    pieces = rb.chunk_shot_pieces(0.0, 3.3, shots)
    assert [(p.shot_id, p.rel_end) for p in pieces] == [("a", pytest.approx(3.3))]
    # A chunk that is only a sliver keeps it.
    assert len(rb.chunk_shot_pieces(2.9, 3.1, shots)) == 1


def test_sheet_majority_ties_to_the_earliest_and_image_count():
    assert rb.sheet_for_chunk(["street", "wedding", "wedding"]) == "wedding"
    assert rb.sheet_for_chunk(["wedding", "street", "street", "wedding"]) == "wedding"
    assert rb.sheet_for_chunk([None, None]) is None
    nine = [rb.ChunkPiece(f"s{i}", i, i + 1) for i in range(9)]
    with pytest.raises(ValueError, match="split this chunk"):
        rb.check_chunk_image_count("seedance-2-5", nine)
    rb.check_chunk_image_count("kling-4-0", nine)  # 9 keyframes + sheet = 10
    with pytest.raises(ValueError):
        rb.check_chunk_image_count("kling-4-0", nine + [rb.ChunkPiece("s9", 9, 10)])
    dup = [rb.ChunkPiece("a", 0, 1), rb.ChunkPiece("b", 1, 2), rb.ChunkPiece("a", 2, 3)]
    assert rb.chunk_keyframe_ids(dup) == ["a", "b"]


# ---------------------------------------------------------------------------
# Prompt (§6.3) and duration (§6.4)
# ---------------------------------------------------------------------------

STORYBOARD = {
    "style": "3D cartoon",
    "characters": [
        {"id": "c1", "name_hint": "the hero", "look": "young man, dark hair", "outfit": "blue suit"},
        {"id": "c2", "name_hint": "the friend", "look": "red hair", "outfit": ""},
        {"id": "c3", "name_hint": "the boss", "look": "grey beard", "outfit": "black"},
    ],
}
CHUNK = {
    "request_sec": 9,
    "shots": [
        {"shot_id": "s1", "rel_start": 0.0, "rel_end": 1.7, "framing": "medium shot",
         "camera": "slow push-in", "action": "the hero opens the door", "characters": ["c1"]},
        {"shot_id": "s2", "rel_start": 1.7, "rel_end": 5.0, "framing": "close-up",
         "camera": "static", "action": "the friend laughs.", "characters": ["c2", "c1"]},
        {"shot_id": "s1", "rel_start": 5.0, "rel_end": 8.4, "framing": "wide",
         "camera": "pan left", "action": "they walk", "characters": ["c1"]},
    ],
}


def test_prompt_for_kling():
    p = rb.render_chunk_prompt(CHUNK, STORYBOARD, "kling-4-0")
    lines = p.split("\n")
    assert lines[0].startswith("A 9-second vertical 9:16 3D cartoon. It consists of 3 shots")
    assert "character sheet @image_3" in lines[0]
    assert "the hero (young man, dark hair; blue suit), the friend (red hair)." in lines[0]
    assert "the boss" not in p
    assert lines[1] == "Shot 1 (0.0-1.7 s), starts from @image_1: medium shot, slow push-in: the hero opens the door."
    assert lines[2] == "Shot 2 (1.7-5.0 s), starts from @image_2: close-up, static: the friend laughs."
    # The last shot is stretched to request_sec; a repeated keyframe keeps its number.
    assert lines[3] == "Shot 3 (5.0-9.0 s), starts from @image_1: wide, pan left: they walk."
    assert lines[-1] == ("No on-screen text, no captions, no subtitles, no logos. No dialogue "
                         "is needed; ambient sound only.")
    assert rb.render_chunk_prompt(CHUNK, STORYBOARD, "kling-4-0") == p  # deterministic


def test_prompt_for_seedance_and_one_shot():
    p = rb.render_chunk_prompt(CHUNK, STORYBOARD, "seedance-2-5")
    assert "[Image 3]" in p and "[Image 1]" in p and "@image" not in p
    one = {"request_sec": 4, "shots": [CHUNK["shots"][0]]}
    p1 = rb.render_chunk_prompt(one, STORYBOARD, "kling-4-0", aspect_ratio="16:9")
    assert p1.startswith("A 4-second horizontal 16:9 3D cartoon. It is one continuous shot")
    assert "Shot 1 (0.0-4.0 s)" in p1
    with pytest.raises(ValueError):
        rb.render_chunk_prompt({"request_sec": 4, "shots": []}, STORYBOARD, "kling-4-0")


def test_duration_request():
    assert rb.request_sec_for_chunk("kling-4-0", 9.3) == 10
    assert rb.request_sec_for_chunk("kling-4-0", 9.01) == 9   # 0.02 s slack
    assert rb.request_sec_for_chunk("kling-4-0", 1.0) == 3    # model minimum
    assert rb.request_sec_for_chunk("seedance-2-5", 1.0) == 4
    assert rb.request_sec_for_chunk("seedance-2-5", 14.72) == 15
    with pytest.raises(ValueError):
        rb.request_sec_for_chunk("seedance-2-5", 15.5)
    with pytest.raises(ValueError):
        rb.request_sec_for_chunk("seedance", 5.0)  # no images-only mode


# ---------------------------------------------------------------------------
# Frame grid and fit (§6.4)
# ---------------------------------------------------------------------------


def test_cumulative_quantisation_never_drifts():
    """Many fractional chunks over a 94.9 s timeline: frames sum to F(end)."""
    for fps in (30.0, 24.0, 29.97, 25.0, 30000 / 1001):
        bounds = [0.0]
        t = 0.0
        while t < 94.9:
            t = min(94.9, round(t + 13.37, 3))
            bounds.append(t)
        total = sum(rb.chunk_frames(a, b, fps) for a, b in zip(bounds, bounds[1:]))
        assert total == rb.frame_index(94.9, fps)
        # Per-chunk rounding would drift: the cumulative rule does not.
    assert rb.frame_index(1.7, 30) == 51
    assert rb.chunk_frames(0.0, 94.9, 30) == 2847


def test_fit_plan_decisions():
    assert rb.fit_plan(400, 300, 30) == rb.FitPlan(True, 300, 0)
    assert rb.fit_plan(298, 300, 30) == rb.FitPlan(True, 298, 2)  # within 3 frames
    assert rb.fit_plan(297, 300, 30) == rb.FitPlan(True, 297, 3)
    bad = rb.fit_plan(296, 300, 30)
    assert not bad.ok and "result too short" in bad.reason
    assert not rb.fit_plan(0, 2, 30).ok
    assert rb.fit_filter(rb.fit_plan(298, 300, 30), 30) == (
        "fps=30,tpad=stop_mode=clone:stop=2,trim=end_frame=300,setpts=PTS-STARTPTS")
    assert rb.fit_filter(rb.fit_plan(400, 300, 30), 30) == (
        "fps=30,trim=end_frame=300,setpts=PTS-STARTPTS")
    assert rb.fit_filter(rb.fit_plan(400, 300, 29.97), 29.97).startswith("fps=30000/1001,")
    assert rb.keep_cut_filter(10, 20, 24) == (
        "fps=24,trim=start_frame=10:end_frame=20,setpts=PTS-STARTPTS")
    with pytest.raises(ValueError):
        rb.keep_cut_filter(20, 20, 24)
    with pytest.raises(ValueError):
        rb.fit_filter(bad, 30)


@pytest.mark.skipif(os.system("ffmpeg -version > /dev/null 2>&1") != 0, reason="no ffmpeg")
def test_fit_filter_produces_exact_frames(tmp_path):
    """Real ffmpeg: a 24 fps raw resampled to 30 fps and trimmed/padded."""
    import subprocess

    src = tmp_path / "raw.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=64x64:rate=24:duration=2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(src)], check=True)

    def frames(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams",
                              "v:0", "-show_entries", "stream=nb_read_frames", "-of",
                              "csv=p=0", str(path)], check=True, capture_output=True, text=True)
        return int(out.stdout.strip())

    for chunk in (45, 60, 61, 63):  # trim, exact, pad by 1 and 3 (2 s @ 30 = 60)
        plan = rb.fit_plan(60, chunk, 30)
        assert plan.ok
        dst = tmp_path / f"fit{chunk}.mp4"
        subprocess.run(rb.fit_ffmpeg_args(str(src), str(dst), plan, 30), check=True)
        assert frames(dst) == chunk
    # Frame-exact keep cut on the 30 fps grid of a 24 fps source.
    dst = tmp_path / "keep.mp4"
    subprocess.run(rb.keep_cut_ffmpeg_args(str(src), str(dst), 7, 40, 30), check=True)
    assert frames(dst) == 33


# ---------------------------------------------------------------------------
# Cut-timing QA (§6.5)
# ---------------------------------------------------------------------------


def test_cut_timing_report_matches_flags_missing_and_extra():
    # Pilot numbers: Kling within 0.18 s -> not flagged.
    planned = [1.7, 3.1, 4.1, 5.4, 7.3, 9.4, 13.4]
    kling = [1.79, 3.17, 4.21, 5.54, 7.42, 9.58, 13.5]
    r = rb.cut_timing_report(planned, kling)
    assert not r["flagged"] and r["max_abs_delta"] == pytest.approx(0.18)
    # Seedance drifted up to 0.9 s -> flagged.
    seed = [2.62, 3.62, 5.29, 7.04, 8.75, 11.42, 14.21]
    assert rb.cut_timing_report(planned, seed)["flagged"]
    # A 0.67 s shot merged by the model -> a missing cut.
    r = rb.cut_timing_report([1.0, 1.67, 4.0], [1.05, 4.02])
    assert r["missing"] == [1.67] and r["flagged"]
    r = rb.cut_timing_report([2.0], [2.0, 6.0])
    assert r["extra"] == [6.0] and r["flagged"]
    assert rb.cut_timing_report([], []) == {"cuts": [], "missing": [], "extra": [],
                                            "max_abs_delta": 0.0, "flagged": False}


def test_images_only_registry_bounds():
    k = ai_models.AI_MODELS["kling-4-0"]
    assert (k.supports_v2v, k.supports_images_only) == (False, True)
    assert (k.images_only_min_sec, k.images_only_max_sec, k.max_reference_images) == (3, 30, 10)
    assert ai_models.image_ref("kling-4-0", 2) == "@image_2"
    assert ai_models.image_ref("seedance-2-5", 9) == "[Image 9]"
    assert list(ai_models.IMAGES_ONLY_MODELS) == ["seedance-2-5", "kling-4-0"]
    assert "kling-4-0" not in ai_models.V2V_MODELS
    with pytest.raises(ValueError):
        ai_models.image_ref("seedance", 1)
