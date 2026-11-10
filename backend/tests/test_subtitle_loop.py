"""AICP-353: subtitle-removal loop helpers (pre-launch review, post-run QA,
rerun / restore / hand-off proposals).

The detector itself is covered by test_qa_tools.py; here the scans are
synthetic CaptionScan objects so every rule is pinned on exact frames.
"""
from __future__ import annotations

import pytest

from app import qa_tools
from app import subtitle_loop as sl
from app.qa_tools import CaptionFrame, CaptionScan, QAToolError

FPS = 10.0
CAPTION = (0.40, 0.60, 0.60, 0.64)   # inside the default band (0.50-0.72)
END_CARD = (0.20, 0.49, 0.47, 0.53)  # straddles the band's top edge


def _word_box(i: int):
    """A caption box that changes every 0.5 s (a new word)."""
    w = 0.10 + 0.04 * (i % 3)
    return (0.5 - w, 0.60, 0.5 + w, 0.64)


def _scan(boxes_by_t: dict, duration: float) -> CaptionScan:
    frames = []
    for i in range(int(round(duration * FPS))):
        t = round(i / FPS, 3)
        box = boxes_by_t.get(t)
        frames.append(CaptionFrame(t=t, score=200 if box else 0, box=box))
    hits = sum(1 for f in frames if f.box)
    return CaptionScan(path="x", fps=FPS, threshold=40, frames=len(frames), hits=hits,
                       hit_ratio=round(hits / len(frames), 4), spans=[], scores=frames)


def _captions(start: float, end: float) -> dict:
    out = {}
    for i in range(int(round(start * FPS)), int(round(end * FPS))):
        out[round(i / FPS, 3)] = _word_box(i // 5)
    return out


def _static(start: float, end: float, box=END_CARD) -> dict:
    return {round(i / FPS, 3): box
            for i in range(int(round(start * FPS)), int(round(end * FPS)))}


SEGS = sl.segments_from([
    {"index": 0, "start_sec": 0.0, "end_sec": 4.0, "run_segment_id": "rs0"},
    {"index": 1, "start_sec": 4.0, "end_sec": 8.0, "run_segment_id": "rs1"},
    {"index": 2, "start_sec": 8.0, "end_sec": 12.0, "run_segment_id": "rs2"},
])


@pytest.fixture
def fake_scans(monkeypatch):
    """Route qa_tools.scan_captions by path: {"src": scan, "fin": scan}."""
    scans = {}

    def scan(path, *, fps=10.0, config=None, **_):
        return scans[path]

    monkeypatch.setattr(qa_tools, "scan_captions", scan)
    monkeypatch.setattr(qa_tools, "contact_sheet", lambda *a, **k: {"out": a[1]})
    # The CLI locates the captions when the config names no band: nothing found.
    monkeypatch.setattr(qa_tools, "locate_captions", lambda path, **k: dict(NO_LOCATION))
    return scans


NO_LOCATION = {"state": "none", "candidate_band_y": None, "positions": [], "members": [],
               "competing": [], "reasons": ["no text-shaped clusters at all"], "frames": 24,
               "fps": 2.0, "detector": {}}


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def test_review_proposes_keep_for_end_card_and_empty_segment(fake_scans):
    boxes = _captions(0.0, 4.0)
    boxes.update(_static(8.2, 12.0))  # seg 2: end card, no captions
    fake_scans["src"] = _scan(boxes, 12.0)
    out = sl.review_segmentation("src", SEGS, price_per_segment=168.0, out_dir="/tmp/x",
                                 admission_estimate=lambda sec: 42.0 * sec)
    rows = {r["index"]: r for r in out["segments"]}
    assert rows[0]["proposed_action"] == "swap"
    assert rows[1]["proposed_action"] == "keep"   # no hits, calibrated
    assert rows[2]["proposed_action"] == "keep"   # static text, faces not sampled
    assert rows[2]["static_text"] and rows[2]["longest_static_sec"] >= 2.0
    assert all(rows[i]["needs_visual_check"] for i in (1, 2))
    assert rows[2]["contact_sheet"].endswith("review_seg2.jpg")
    assert out["cost"]["credits_current"] == 504.0
    assert out["cost"]["credits_proposed"] == 168.0
    assert out["cost"]["credits_saved"] == 336.0
    assert out["cost"]["budget_check_credits_proposed"] == 168.0
    assert out["approval_required"] is True
    # The end card straddles the band edge, but only in a keep segment.
    band = out["band"]
    assert band["verdict"] == "edge_touched_keep_only"
    assert band["edge_hits"] == [{"index": 2, "frames": 38, "proposed_action": "keep"}]
    assert band["proposed_band_y"] is None


def test_review_static_text_with_a_face_stays_swap(fake_scans):
    boxes = _captions(0.0, 8.0)
    boxes.update(_static(8.2, 12.0))
    fake_scans["src"] = _scan(boxes, 12.0)
    faces = [(t + 0.5, 1) for t in range(12)]  # a person in every sample
    out = sl.review_segmentation("src", SEGS, price_per_segment=168.0, faces=faces)
    row = out["segments"][2]
    assert row["proposed_action"] == "swap"
    assert row["face_ratio"] == 1.0
    assert "phone UI" in row["reason"]


def test_review_uncalibrated_never_keeps_on_zero_hits(fake_scans):
    fake_scans["src"] = _scan(_captions(0.0, 1.0), 12.0)  # 10 / 120 frames < 0.2
    out = sl.review_segmentation("src", SEGS, price_per_segment=168.0)
    assert out["calibration"]["calibrated"] is False
    assert [r["proposed_action"] for r in out["segments"]] == ["swap", "swap", "swap"]


def test_review_proposes_wider_band_when_captions_cross_it(fake_scans):
    high = (0.40, 0.44, 0.60, 0.48)  # above the configured 0.50
    boxes = {t: (high if int(t * FPS) % 10 < 5 else CAPTION) for t in _captions(0, 12)}
    fake_scans["src"] = _scan(boxes, 12.0)
    out = sl.review_segmentation("src", SEGS, price_per_segment=168.0)
    band = out["band"]
    assert band["verdict"] == "widen"
    assert band["proposed_band_y"] == [0.41, 0.67]
    assert band["needs_visual_check"] is True


def test_review_keeps_existing_keep_segments():
    segs = sl.segments_from([{"index": 0, "start_sec": 0, "end_sec": 2, "action": "keep"}])
    assert segs[0].action == "keep"
    with pytest.raises(QAToolError):
        sl.segments_from([{"index": 0, "start_sec": 2, "end_sec": 1}])
    with pytest.raises(QAToolError):
        sl.segments_from([{"index": 0, "start_sec": 0, "end_sec": 1, "action": "cut"}])
    with pytest.raises(QAToolError):
        sl.segments_from([])


# ---------------------------------------------------------------------------
# post-run QA
# ---------------------------------------------------------------------------

@pytest.fixture
def qa_env(fake_scans, monkeypatch):
    class Info:
        def __init__(self, d):
            self.duration_sec = d

    durations = {"src": 12.0, "fin": 12.0}
    monkeypatch.setattr(qa_tools, "_probe", lambda p: Info(durations[p]))
    sync = {"in_sync": True, "median": 0.0}
    monkeypatch.setattr(qa_tools, "alignment_check", lambda s, r, **k: {
        "in_sync": sync["in_sync"], "points": 20, "max_abs_window_median": sync["median"],
        "frame_count_delta": 0, "length_ok": True, "windows": []})
    monkeypatch.setattr(sl, "_crop_sheet", lambda s, f, spans, out: {
        "out": out, "times": [t for t, _ in sl._word_moments(spans)]})
    monkeypatch.setattr(qa_tools, "comparison_sheet", lambda *a, **k: {"out": a[2]})
    return {"scans": fake_scans, "durations": durations, "sync": sync}


def _src_with_end_card():
    boxes = _captions(0.0, 8.0)
    boxes.update(_static(8.2, 12.0))
    return _scan(boxes, 12.0)


def test_qa_leftover_false_positive_and_clean(qa_env):
    qa_env["scans"]["src"] = _src_with_end_card()
    fin = _captions(5.0, 5.9)        # seg 1: two leftover words
    # seg 2: the end card again, re-rendered a few pixels lower.
    fin.update(_static(8.6, 11.0, box=(0.24, 0.51, 0.47, 0.55)))
    qa_env["scans"]["fin"] = _scan(fin, 12.0)
    qa = sl.post_run_qa("src", "fin", SEGS, out_dir="/tmp/x")
    rows = {r["index"]: r for r in qa["segments"]}
    assert rows[0]["verdict"] == "clean"
    assert rows[1]["verdict"] == "leftover"
    assert rows[1]["leftovers"][0]["start_sec"] == 5.0
    assert rows[1]["crop_times"] == [5.2, 5.6]  # one crop per word
    assert rows[1]["words"] is None
    assert rows[2]["verdict"] == "false_positive" and rows[2]["needs_visual_check"]
    assert qa["dirty_segments"] == [1]
    assert qa["unconfirmed_false_positives"] == [2]
    assert qa["clean"] is False
    assert qa["comparison_sheet"].endswith("qa_compare.jpg")


def test_qa_caption_in_end_card_segment_is_not_hidden(qa_env):
    qa_env["scans"]["src"] = _src_with_end_card()
    # A caption-shaped box the source never had there: not a false positive.
    qa_env["scans"]["fin"] = _scan(_static(9.0, 9.5, box=CAPTION), 12.0)
    qa = sl.post_run_qa("src", "fin", SEGS)
    assert qa["segments"][2]["verdict"] == "leftover"


def test_qa_clean_needs_calibration_and_timeline(qa_env):
    qa_env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    qa_env["scans"]["fin"] = _scan({}, 12.0)
    qa = sl.post_run_qa("src", "fin", SEGS)
    assert qa["clean"] is True and qa["warnings"] == []
    # A few frames off (each Omni clip): still mappable, a warning only.
    qa_env["sync"].update(in_sync=False, median=3.0)
    qa = sl.post_run_qa("src", "fin", SEGS)
    assert qa["timeline_verified"] is True and qa["clean"] is True
    assert "3 frame" in qa["warnings"][0]
    # At the search limit: not on the source timeline -> never clean.
    qa_env["sync"].update(median=15.0)
    qa = sl.post_run_qa("src", "fin", SEGS)
    assert qa["timeline_verified"] is False and qa["clean"] is False
    assert qa["segments"][0]["verdict"] == "clean"
    qa_env["sync"].update(in_sync=True, median=0.0)
    qa_env["scans"]["src"] = _scan({}, 12.0)
    qa = sl.post_run_qa("src", "fin", SEGS)
    assert qa["segments"][0]["verdict"] == "clean_uncalibrated"
    assert qa["clean"] is False


def test_qa_refuses_a_final_off_the_source_timeline(qa_env):
    qa_env["durations"]["fin"] = 15.0
    with pytest.raises(QAToolError, match="not on the source timeline"):
        sl.post_run_qa("src", "fin", SEGS)


def test_qa_keep_segment_hits_are_expected(qa_env):
    segs = sl.segments_from([
        {"index": 0, "start_sec": 0.0, "end_sec": 6.0},
        {"index": 1, "start_sec": 6.0, "end_sec": 12.0, "action": "keep"},
    ])
    qa_env["scans"]["src"] = _scan(_captions(0.0, 6.0), 12.0)  # keep has no captions
    qa_env["scans"]["fin"] = _scan({}, 12.0)
    qa = sl.post_run_qa("src", "fin", segs)
    assert [r["verdict"] for r in qa["segments"]] == ["clean", "keep"]
    assert qa["clean"] is True


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------

def _qa_with_leftovers(*indices):
    return {"clean": False, "dirty_segments": list(indices), "segments": [
        {"index": i, "verdict": "leftover" if i in indices else "clean",
         "start_sec": SEGS[i].start, "end_sec": SEGS[i].end,
         "leftovers": [{"start_sec": 1, "end_sec": 2, "frames": 5, "max_score": 99,
                        "box": [0.40, 0.60, 0.60, 0.64]}] if i in indices else []}
        for i in range(3)]}


def test_rerun_proposal_coordinate_then_default_and_cost():
    segs = sl.segments_from([
        {"index": 0, "start_sec": 0, "end_sec": 4, "run_segment_id": "rs0"},
        {"index": 1, "start_sec": 4, "end_sec": 8, "run_segment_id": "rs1",
         "prompt_override": sl.coordinate_prompt([0.3, 0.5, 0.7, 0.7])},
        {"index": 2, "start_sec": 8, "end_sec": 12, "run_segment_id": "rs2"},
    ])
    out = sl.rerun_proposal("RID", _qa_with_leftovers(0, 1), segs, takes={},
                            price_per_segment=168.0,
                            admission_estimate=lambda ids: 168.0 * len(ids) * 2)
    body = out["request"]["body"]
    assert out["request"]["path"] == "/api/v2/runs/RID/segments/rerun"
    assert body["discard_audio_patch"] is False
    first, second = body["segments"]
    assert first["run_segment_id"] == "rs0"
    assert "x 0.35–0.65, y 0.56–0.68" in first["prompt"]
    assert sl.is_coordinate_prompt(first["prompt"])
    assert second == {"run_segment_id": "rs1", "prompt": ""}  # back to the run prompt
    assert out["segments"][1]["replaces_override"] is True
    assert out["cost"]["credits"] == 336.0 and out["cost"]["segments_billed"] == 2
    assert out["cost"]["budget_check_credits"] == 672.0
    assert out["approval_required"] is True


def test_rerun_proposal_bills_other_unfinished_segments():
    # rs0 is listed AND unfinished: billed once, not twice.
    out = sl.rerun_proposal("RID", _qa_with_leftovers(0), SEGS, takes={},
                            price_per_segment=168.0, unfinished=["rs0", "rs1", "rs2"])
    assert out["cost"]["other_unfinished"] == 2
    assert out["cost"]["segments_billed"] == 3
    assert out["cost"]["credits"] == 504.0


def test_rerun_proposal_stop_rule():
    out = sl.rerun_proposal("RID", _qa_with_leftovers(0, 2), SEGS,
                            takes={"rs0": 2, "rs2": 1}, price_per_segment=168.0)
    assert [s["index"] for s in out["stop"]] == [0]
    assert out["stop"][0]["outputs"] == 3
    assert len(out["stop"][0]["options"]) == 3
    assert [s["run_segment_id"] for s in out["request"]["body"]["segments"]] == ["rs2"]
    only_stop = sl.rerun_proposal("RID", _qa_with_leftovers(0), SEGS,
                                  takes={"rs0": 5}, price_per_segment=168.0)
    assert only_stop["request"] is None and only_stop["approval_required"] is True


def test_choose_best_takes():
    out = sl.choose_best_takes("RID", {
        "rs0": [{"take_id": None, "hits": 3, "created_at": "2026-10-30T14:00"},
                {"take_id": "t1", "hits": 0, "created_at": "2026-10-30T13:00"}],
        "rs1": [{"take_id": None, "hits": 0, "created_at": "2026-10-30T14:00"},
                {"take_id": "t2", "hits": 0, "created_at": "2026-10-30T13:00"}],
    })
    assert out["best"]["rs0"]["take_id"] == "t1"
    assert out["best"]["rs1"]["take_id"] is None  # tie -> newest (current)
    assert out["request"]["body"] == {"takes": ["t1"], "discard_audio_patch": False}
    assert sl.choose_best_takes("RID", {"rs0": [{"take_id": None, "hits": 0}]})["request"] is None


def test_handoff_only_when_clean():
    assert sl.handoff_proposal("RID", _qa_with_leftovers(1), SEGS)["request"] is None
    qa = {"clean": True, "dirty_segments": [], "segments": [
        {"index": 0, "verdict": "clean", "start_sec": 0.0, "end_sec": 4.0},
        {"index": 2, "verdict": "false_positive", "start_sec": 8.0, "end_sec": 12.0}]}
    out = sl.handoff_proposal("RID", qa, SEGS)
    assert out["request"]["path"] == "/api/v2/runs/RID/to-project"
    assert out["request"]["form"] == {"project_type": "localisation",
                                      "localisation_scope": "whole"}
    assert out["keep_ranges"] == [{"start_sec": 8.0, "end_sec": 12.0}]
    assert "PATCH" in out["then"]


def test_word_moments_one_per_box():
    sp = {"_times": [1.0, 1.1, 1.2, 1.3, 1.4],
          "_boxes": [CAPTION, CAPTION, END_CARD, END_CARD, END_CARD]}
    assert sl._word_moments([sp]) == [(1.0, CAPTION), (1.3, END_CARD)]


def test_false_positive_blocks_clean_until_confirmed(qa_env):
    qa_env["scans"]["src"] = _src_with_end_card()
    qa_env["scans"]["fin"] = _scan(_static(8.6, 11.0), 12.0)
    qa = sl.post_run_qa("src", "fin", SEGS)
    assert qa["segments"][2]["verdict"] == "false_positive"
    assert qa["clean"] is False and qa["unconfirmed_false_positives"] == [2]
    qa = sl.post_run_qa("src", "fin", SEGS, confirmed_false_positives=[2])
    assert qa["segments"][2]["confirmed"] is True
    assert qa["clean"] is True


def test_calibration_ignores_static_text(qa_env, fake_scans):
    # Only an end card on the source: the detector sees no captions at all.
    src = _scan(_static(0.0, 12.0), 12.0)
    assert sl._caption_calibration(src, FPS)["calibrated"] is False
    qa_env["scans"]["src"] = src
    qa_env["scans"]["fin"] = _scan({}, 12.0)
    qa = sl.post_run_qa("src", "fin", SEGS)
    assert qa["segments"][0]["verdict"] == "clean_uncalibrated" and qa["clean"] is False
    fake_scans["src"] = src
    review = sl.review_segmentation("src", SEGS, price_per_segment=168.0)
    assert review["calibration"]["calibrated"] is False


def test_keep_segment_with_captions_blocks_clean(qa_env):
    segs = sl.segments_from([
        {"index": 0, "start_sec": 0.0, "end_sec": 6.0},
        {"index": 1, "start_sec": 6.0, "end_sec": 12.0, "action": "keep"},
    ])
    qa_env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    qa_env["scans"]["fin"] = _scan(_captions(6.0, 12.0), 12.0)
    qa = sl.post_run_qa("src", "fin", segs)
    assert [r["verdict"] for r in qa["segments"]] == ["clean", "keep_captions"]
    assert qa["keep_with_captions"] == [1] and qa["clean"] is False


def test_leftover_just_across_a_keep_boundary_counts_for_the_swap(qa_env):
    segs = sl.segments_from([
        {"index": 0, "start_sec": 0.0, "end_sec": 6.0},
        {"index": 1, "start_sec": 6.0, "end_sec": 12.0, "action": "keep"},
    ])
    src = _captions(0.0, 6.0)
    src.update(_static(6.2, 12.0))  # keep segment: an end card
    qa_env["scans"]["src"] = _scan(src, 12.0)
    qa_env["sync"].update(in_sync=False, median=3.0)  # 0.1 s drift -> 0.2 s slack
    qa_env["scans"]["fin"] = _scan({6.0: CAPTION, 6.1: CAPTION}, 12.0)
    qa = sl.post_run_qa("src", "fin", segs)
    assert qa["segments"][0]["verdict"] == "leftover"


def test_launch_budget_includes_rolls(fake_scans):
    segs = sl.segments_from([{"index": 0, "start_sec": 0.0, "end_sec": 4.0,
                              "pre_roll_sec": 0.5, "post_roll_sec": 0.5}])
    fake_scans["src"] = _scan(_captions(0.0, 4.0), 4.0)
    out = sl.review_segmentation("src", segs, price_per_segment=168.0,
                                 admission_estimate=lambda sec: sec)
    assert out["cost"]["budget_check_credits_proposed"] == 5.0


def test_keep_segment_end_card_plus_captions_is_keep_captions(qa_env):
    segs = sl.segments_from([
        {"index": 0, "start_sec": 0.0, "end_sec": 6.0},
        {"index": 1, "start_sec": 6.0, "end_sec": 12.0, "action": "keep"},
    ])
    src = _captions(0.0, 6.0)
    src.update(_static(6.0, 10.0))     # end card ...
    src.update(_captions(10.0, 11.0))  # ... then changing captions
    qa_env["scans"]["src"] = _scan(src, 12.0)
    qa_env["scans"]["fin"] = _scan({}, 12.0)
    qa = sl.post_run_qa("src", "fin", segs)
    assert qa["segments"][1]["verdict"] == "keep_captions"
    assert qa["clean"] is False


def test_boundary_slack_compares_boxes(qa_env):
    segs = sl.segments_from([
        {"index": 0, "start_sec": 0.0, "end_sec": 6.0},
        {"index": 1, "start_sec": 6.0, "end_sec": 12.0, "action": "keep"},
    ])
    src = _captions(0.0, 6.0)
    src.update(_static(6.0, 12.0, box=(0.20, 0.66, 0.80, 0.70)))  # UI elsewhere
    qa_env["scans"]["src"] = _scan(src, 12.0)
    qa_env["sync"].update(in_sync=False, median=3.0)
    qa_env["scans"]["fin"] = _scan({6.0: CAPTION, 6.1: CAPTION}, 12.0)
    assert sl.post_run_qa("src", "fin", segs)["segments"][0]["verdict"] == "leftover"
    # The keep side's own text box, at the same place: not the swap's leftover.
    qa_env["scans"]["fin"] = _scan({6.0: (0.20, 0.66, 0.80, 0.70)}, 12.0)
    assert sl.post_run_qa("src", "fin", segs)["segments"][0]["verdict"] == "clean"


def test_launch_budget_uses_unrounded_clip_length(fake_scans):
    segs = sl.segments_from([{"index": 0, "start_sec": 0.0, "end_sec": 4.0001}])
    fake_scans["src"] = _scan(_captions(0.0, 4.0), 4.0)
    import math
    out = sl.review_segmentation("src", segs, price_per_segment=168.0,
                                 admission_estimate=lambda sec: math.ceil(sec))
    assert out["cost"]["budget_check_credits_proposed"] == 5.0


def test_slack_and_keep_rules_hold_at_other_fps(qa_env):
    segs = sl.segments_from([
        {"index": 0, "start_sec": 0.0, "end_sec": 6.0},
        {"index": 1, "start_sec": 6.0, "end_sec": 12.0, "action": "keep"},
    ])
    src = _captions(0.0, 6.0)
    # The keep side's UI box appears only at 6.2 s, after the drifted caption.
    src.update(_static(6.2, 12.0, box=CAPTION))
    qa_env["scans"]["src"] = _scan(src, 12.0)
    qa_env["sync"].update(in_sync=False, median=3.0)
    qa_env["scans"]["fin"] = _scan({6.0: CAPTION, 6.1: CAPTION}, 12.0)
    assert sl.post_run_qa("src", "fin", segs)["segments"][0]["verdict"] == "leftover"
    # 1 fps: one caption-like sample in a keep segment is enough.
    one = CaptionScan(path="x", fps=1.0, threshold=40, frames=12, hits=7, hit_ratio=0.58,
                      spans=[], scores=[CaptionFrame(t=float(t), score=200 if t < 7 else 0,
                                                     box=_word_box(t) if t < 7 else None)
                                        for t in range(12)])
    qa_env["scans"]["src"] = one
    qa_env["scans"]["fin"] = CaptionScan(path="x", fps=1.0, threshold=40, frames=12, hits=0,
                                         hit_ratio=0.0, spans=[],
                                         scores=[CaptionFrame(t=float(t), score=0)
                                                 for t in range(12)])
    qa_env["sync"].update(in_sync=True, median=0.0)
    qa = sl.post_run_qa("src", "fin", segs, fps=1.0)
    assert qa["segments"][1]["verdict"] == "keep_captions"


# ---------------------------------------------------------------------------
# AICP-359: VOID fill proposals instead of Omni reruns
# ---------------------------------------------------------------------------

FILL_SEGS = sl.segments_from([
    {"index": 0, "start_sec": 0.0, "end_sec": 4.0, "run_segment_id": "rs0",
     "result_kind": "model"},
    {"index": 1, "start_sec": 4.0, "end_sec": 8.0, "run_segment_id": "rs1",
     "result_kind": "model"},
    {"index": 2, "start_sec": 8.0, "end_sec": 12.0, "run_segment_id": "rs2",
     "result_kind": "model"},
])


def _fill_segs(**overrides):
    items = [{"index": s.index, "start_sec": s.start, "end_sec": s.end,
              "run_segment_id": s.run_segment_id, "result_kind": s.result_kind}
             for s in FILL_SEGS]
    for i, extra in overrides.items():
        items[int(i[1:])].update(extra)
    return sl.segments_from(items)


def _ctx(**segs):
    return {"run_blockers": [], "active_fill": None,
            "segments": {"rs0": {"blockers": [], "jobs": []},
                         "rs1": {"blockers": [], "jobs": []},
                         "rs2": {"blockers": [], "jobs": []}, **segs}}


def _job(job_id="f1", status="ready", **kw):
    out = {"job_id": job_id, "status": status, "box_source": "measured",
           "mask_box": [0.3, 0.58, 0.7, 0.66], "default_prompt": True,
           "estimated_usd": 0.10, "provider_task_id": None, "provider_status": None,
           "error_code": None, "error_message": None, "stale": None,
           "frames_masked": 120, "preview_url": "/p.jpg"}
    out.update(kw)
    return out


def test_qa_rows_carry_the_source_mask_box(qa_env):
    qa_env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    qa_env["scans"]["fin"] = _scan(_captions(5.0, 5.9), 12.0)
    qa = sl.post_run_qa("src", "fin", FILL_SEGS)
    row = qa["segments"][1]
    # Union of the seg-1 source word boxes (widest 0.5 +- 0.18) + the fill's pads.
    assert row["mask_box"] == [0.29, 0.58, 0.71, 0.66]
    assert row["captions_centred"] is True                 # words centred on 0.5
    assert row["result_kind"] == "model"
    assert sl.mask_box_from([]) is None


def test_fill_proposal_previews_dirty_segments_not_omni_reruns(qa_env):
    qa_env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    fin = _captions(5.0, 5.9)
    fin.update(_captions(9.0, 9.6))
    qa_env["scans"]["fin"] = _scan(fin, 12.0)
    qa = sl.post_run_qa("src", "fin", FILL_SEGS)
    assert qa["dirty_segments"] == [1, 2]
    out = sl.fill_proposal("RID", qa, FILL_SEGS, context=_ctx(), reserve_usd=0.10)
    a, b = out["segments"]
    assert [a["state"], b["state"]] == ["propose_preview", "propose_preview"]
    assert a["request"] == {"method": "POST", "path": "/api/v2/runs/RID/segments/rs1/fill/preview",
                            "body": {}, "paid": False}
    assert a["expected_mask_box"] == qa["segments"][1]["mask_box"]
    then = a["then"]
    assert then["executable"] is False and then["paid"] is True
    assert then["path"] == "/api/v2/runs/RID/segments/fill/{fill_id}/start"
    assert then["body"] == {"approved_usd": 0.1, "discard_audio_patch": False}
    # One at a time, the first dirty segment.
    assert out["next"]["run_segment_id"] == "rs1" and out["next"]["state"] == "propose_preview"
    assert out["order"] == [1, 2] and out["sequential"] is True
    cost = out["cost"]
    assert cost["reserve_usd_per_segment"] == 0.1 and cost["expected_usd_per_segment"] == 0.05
    assert cost["reserve_usd_total"] == 0.2 and cost["expected_usd_total"] == 0.1
    assert cost["kie_credits"] == 0
    assert out["approval_required"] is True and out["actionable"] is True
    assert "rerun" not in json_dump(out)


def json_dump(obj) -> str:
    import json
    return json.dumps(obj)


def test_fill_reserve_comes_from_the_fal_registry():
    from app import fal_models

    assert sl.void_reserve_usd() == fal_models.endpoint(fal_models.VOID_ENDPOINT).flat_usd


def test_fill_proposal_not_actionable_has_no_next():
    qa = _qa_with_leftovers(0)
    out = sl.fill_proposal("RID", qa, FILL_SEGS, actionable=False,
                           not_actionable_reason="--final is not the run's current final",
                           reserve_usd=0.10)
    assert out["segments"][0]["state"] == "propose_preview"
    assert out["segments"][0]["prechecked"] is False
    assert out["next"] is None and out["actionable"] is False
    assert out["not_actionable_reason"].startswith("--final")


def test_fill_proposal_waits_while_a_fill_is_in_flight():
    qa = _qa_with_leftovers(0, 2)
    ctx = _ctx(rs2={"blockers": [], "jobs": [_job("f2", "compositing",
                                                   provider_task_id="t2")]})
    ctx["active_fill"] = {"job_id": "f2", "run_segment_id": "rs2", "status": "compositing"}
    ctx["run_blockers"] = ["fill f2 is compositing on run segment rs2: one fill at a time"]
    out = sl.fill_proposal("RID", qa, FILL_SEGS, context=ctx, reserve_usd=0.10)
    states = {i["index"]: i["state"] for i in out["segments"]}
    assert states == {0: "blocked", 2: "in_progress"}
    assert out["next"] is None


def test_fill_proposal_starts_a_fresh_ready_preview_first():
    qa = _qa_with_leftovers(0, 2)
    ctx = _ctx(rs2={"blockers": [], "jobs": [_job("f2", estimated_usd=0.10)]})
    out = sl.fill_proposal("RID", qa, FILL_SEGS, context=ctx, reserve_usd=0.10)
    by = {i["index"]: i for i in out["segments"]}
    assert by[0]["state"] == "propose_preview"
    start = by[2]
    assert start["state"] == "propose_start" and start["needs_visual_check"] is True
    assert start["request"] == {"method": "POST",
                                "path": "/api/v2/runs/RID/segments/fill/f2/start",
                                "body": {"approved_usd": 0.1, "discard_audio_patch": False},
                                "paid": True}
    # The ready preview goes first: a new preview would go stale once it lands.
    assert out["next"]["run_segment_id"] == "rs2"


def test_fill_proposal_ready_preview_variants():
    qa = _qa_with_leftovers(0)
    stale = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [_job(stale="run.final_revision")]}))
    item = stale["segments"][0]
    assert item["state"] == "stale_preview"
    assert item["request"]["path"] == "/api/v2/runs/RID/segments/fill/f1/cancel"
    assert stale["next"]["state"] == "stale_preview"
    custom = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [_job(box_source="request")]}))
    assert custom["segments"][0]["state"] == "operator" and custom["next"] is None
    pricey = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [_job(estimated_usd=0.25)]}))
    assert pricey["segments"][0]["state"] == "operator"
    two = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [_job("f1"), _job("f0", "preparing")]}))
    assert two["segments"][0]["state"] == "operator" and "ambiguous" in two["segments"][0]["reason"]


def test_fill_proposal_never_repeats_a_paid_fill_on_its_own():
    qa = _qa_with_leftovers(0)
    out = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [_job("f1", "failed", provider_task_id="t1",
                                           provider_status="failed",
                                           error_code="provider_failed")]}))
    assert out["segments"][0]["state"] == "operator" and out["next"] is None
    assert "second paid attempt" in out["segments"][0]["reason"]


def test_fill_proposal_blocked_by_gates_and_deterministic_failures():
    qa = _qa_with_leftovers(0)
    gated = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": ["caption fill needs a run with audio_mode=original"], "jobs": []}))
    assert gated["segments"][0]["state"] == "blocked" and gated["next"] is None
    assert gated["segments"][0]["blockers"] == ["caption fill needs a run with audio_mode=original"]
    failed = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [_job("f1", "failed", error_code="segment_too_short",
                                           error_message="68 frames")]}))
    assert failed["segments"][0]["state"] == "blocked"
    assert "segment_too_short" in failed["segments"][0]["blockers"][0]
    # A crashed prepare (not deterministic, nothing paid) is proposed again.
    again = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [_job("f1", "failed", error_code="prepare_crashed")]}))
    assert again["segments"][0]["state"] == "propose_preview"


def test_fill_proposal_unknown_result_goes_to_the_operator():
    segs = _fill_segs(s0={"result_kind": "unknown"})
    out = sl.fill_proposal("RID", _qa_with_leftovers(0), segs, context=_ctx(), reserve_usd=0.10)
    assert out["segments"][0]["state"] == "operator" and out["next"] is None


def test_dirty_void_composite_hits_the_stop_rule():
    fill = {"job_id": "f9", "mask_box": [0.38, 0.59, 0.62, 0.65], "replaced_take_id": "t9",
            "delivery": "delivered"}
    segs = _fill_segs(s0={"result_kind": "void_fill", "fill": fill})
    qa = _qa_with_leftovers(0)
    # A leftover edge letter outside the old box, and one in the boundary slack.
    qa["segments"][0]["leftovers"] = [
        {"start_sec": 1.0, "end_sec": 1.3, "frames": 4, "max_score": 99,
         "box": [0.30, 0.60, 0.40, 0.64]},
        {"start_sec": 4.05, "end_sec": 4.1, "frames": 1, "max_score": 99,
         "box": [0.80, 0.60, 0.90, 0.64]}]
    out = sl.fill_proposal("RID", qa, segs, context=_ctx(), reserve_usd=0.10)
    item = out["segments"][0]
    assert item["state"] == "stop" and item["request"] is None and out["next"] is None
    assert out["stop"] == [item]
    opts = {o["option"]: o for o in item["options"]}
    widen = opts["widen_mask"]
    assert widen["executable"] is False and widen["second_paid_attempt"] is True
    assert widen["mask_box"] == [0.27, 0.57, 0.65, 0.67]   # slack leftover ignored
    assert widen["request"]["body"] == {"mask_box": widen["mask_box"]}
    assert opts["ltx_inpaint"]["available"] is False
    assert opts["delogo"]["manual"] is True and len(opts["delogo"]["windows"]) == 2
    assert opts["restore_previous_take"]["request"]["body"]["takes"] == ["t9"]
    assert "accept" in opts and opts["omni_rerun"]["see"] == "operator_options.omni_rerun"


def test_void_composite_needs_a_visual_check_bound_to_its_fill_id(qa_env, monkeypatch):
    monkeypatch.setattr(sl, "_fill_sheet", lambda s, f, times, box, out: {
        "out": out, "times": list(times)})
    qa_env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    qa_env["scans"]["fin"] = _scan({}, 12.0)
    fill = {"job_id": "f9", "mask_box": [0.3, 0.58, 0.7, 0.66], "delivery": "delivered"}
    segs = _fill_segs(s1={"result_kind": "void_fill", "fill": fill})
    qa = sl.post_run_qa("src", "fin", segs, out_dir="/tmp/x")
    row = qa["segments"][1]
    assert row["verdict"] == "clean"                       # the detector's result
    assert row["fill_needs_visual_check"] and row["needs_visual_check"]
    assert row["fill_sheet"].endswith("qa_seg1_fill.jpg") and row["fill_sheet_times"]
    assert qa["unconfirmed_fills"] == [{"index": 1, "fill_job_id": "f9"}]
    assert qa["clean"] is False
    assert "eye check" in sl.handoff_proposal("RID", qa, segs)["reason"]
    # Confirming by segment index (the false-positive switch) does not count.
    assert sl.post_run_qa("src", "fin", segs, confirmed_false_positives=[1])["clean"] is False
    assert sl.post_run_qa("src", "fin", segs, confirmed_fills=["other"])["clean"] is False
    ok = sl.post_run_qa("src", "fin", segs, confirmed_fills=["f9"])
    assert ok["clean"] is True and ok["segments"][1]["fill"]["confirmed"] is True
    assert not ok["segments"][1]["needs_visual_check"]
    # Its eye-check reject options are on the fill proposal.
    out = sl.fill_proposal("RID", qa, segs, context=_ctx(), reserve_usd=0.10)
    assert out["segments"] == [] and out["next"] is None
    assert out["on_visual_reject"][0]["fill_job_id"] == "f9"


def test_unconfirmed_delivery_blocks_the_fill_confirmation(qa_env):
    qa_env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    qa_env["scans"]["fin"] = _scan({}, 12.0)
    fill = {"job_id": "f9", "mask_box": [0.3, 0.58, 0.7, 0.66], "delivery": "republished"}
    segs = _fill_segs(s1={"result_kind": "void_fill", "fill": fill})
    qa = sl.post_run_qa("src", "fin", segs, confirmed_fills=["f9"])
    assert qa["segments"][1]["fill_delivery_unconfirmed"] == "republished"
    assert qa["clean"] is False


def test_omni_rerun_is_an_operator_option_only():
    out = sl.rerun_proposal("RID", _qa_with_leftovers(0), SEGS, takes={},
                            price_per_segment=168.0)
    assert out["operator_option"] is True and out["executable"] is False
    assert "VOID" in out["why_not_proposed"]
    stop = sl.rerun_proposal("RID", _qa_with_leftovers(0), SEGS, takes={"rs0": 5},
                             price_per_segment=168.0)
    assert stop["executable"] is False and stop["stop"][0]["options"][0].startswith("VOID")


def test_segments_from_validates_fill_fields():
    with pytest.raises(QAToolError, match="result_kind"):
        sl.segments_from([{"index": 0, "start_sec": 0, "end_sec": 1, "result_kind": "x"}])
    with pytest.raises(QAToolError, match="job_id"):
        sl.segments_from([{"index": 0, "start_sec": 0, "end_sec": 1, "fill": {}}])
    seg = sl.segments_from([{"index": 0, "start_sec": 0, "end_sec": 1,
                             "fill": {"job_id": "f1"}}])[0]
    assert seg.result_kind == "void_fill"


def test_cli_qa_outputs_fill_proposal_and_omni_as_operator_option(qa_env, capsys):
    import json

    qa_env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    qa_env["scans"]["fin"] = _scan(_captions(5.0, 5.9), 12.0)
    segs = json.dumps([{"index": s.index, "start_sec": s.start, "end_sec": s.end,
                        "run_segment_id": s.run_segment_id} for s in SEGS])
    assert sl.main(["qa", "--source", "src", "--final", "fin", "--segments", segs,
                    "--price", "168"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"qa", "fill", "operator_options", "handoff"}
    fill = out["fill"]
    assert fill["actionable"] is False and fill["next"] is None   # files only
    assert [i["state"] for i in fill["segments"]] == ["propose_preview"]
    assert fill["segments"][0]["request"]["path"].endswith("/segments/rs1/fill/preview")
    omni = out["operator_options"]["omni_rerun"]
    assert omni["executable"] is False and omni["request"]["path"].endswith("/segments/rerun")


def test_ready_preview_after_a_paid_fill_goes_to_the_operator():
    qa = _qa_with_leftovers(0)
    out = sl.fill_proposal("RID", qa, FILL_SEGS, reserve_usd=0.10, context=_ctx(
        rs0={"blockers": [], "jobs": [
            _job("f2", "ready"),
            _job("f1", "failed", provider_task_id="t1", provider_status="failed",
                 error_code="provider_failed")]}))
    item = out["segments"][0]
    assert item["state"] == "operator" and item["request"] is None
    assert "second paid attempt" in item["reason"] and out["next"] is None


def test_fill_is_not_actionable_off_the_source_timeline():
    qa = _qa_with_leftovers(0)
    qa["timeline_verified"] = False
    out = sl.fill_proposal("RID", qa, FILL_SEGS, context=_ctx(), reserve_usd=0.10)
    assert out["actionable"] is False and out["next"] is None
    assert "timeline" in out["not_actionable_reason"]
    assert out["segments"][0]["state"] == "propose_preview"   # still shown as evidence


# ---------------------------------------------------------------------------
# Caption location (no band of the project's own)
# ---------------------------------------------------------------------------

def _loc_at(y0, y1, band, state="located"):
    return {"state": state, "candidate_band_y": list(band), "frames": 24, "fps": 2.0,
            "members": [0], "competing": [], "reasons": [], "detector": {},
            "lines_outside_band": 0.0,
            "positions": [{"y_range": [y0, y1], "support": 0.7, "caption_like": True,
                           "windows_hit": 5, "y0_p5": y0, "y1_p95": y1, "band_y": list(band),
                           "sample_times": [1.0, 5.0, 9.0], "nonstatic_frames": 17,
                           "static_frames": 0}]}


def test_out_of_band_check_warns_only_for_captions_outside_the_qa_band():
    inside = sl.out_of_band_check(_loc_at(0.60, 0.64, (0.54, 0.70)))
    assert inside["warning"] is None and inside["outside"] == []
    out = sl.out_of_band_check(_loc_at(0.76, 0.80, (0.70, 0.86)))
    assert "outside the QA band y 0.50-0.72" in out["warning"]
    assert "y 0.70-0.86" in out["warning"]
    assert sl.out_of_band_check(_loc_at(0.76, 0.80, (0.70, 0.86)),
                                {"band_y": [0.7, 0.86]})["warning"] is None


def test_cli_review_locates_only_without_a_band(fake_scans, monkeypatch, capsys):
    import json

    fake_scans["src"] = _scan(_captions(0.0, 12.0), 12.0)
    calls = []
    monkeypatch.setattr(qa_tools, "locate_captions",
                        lambda path, **k: calls.append(path) or dict(NO_LOCATION))

    class Info:
        duration_sec = 12.0
    monkeypatch.setattr(qa_tools, "_probe", lambda p: Info())
    segs = json.dumps([{"index": s.index, "start_sec": s.start, "end_sec": s.end} for s in SEGS])
    assert sl.main(["review", "--source", "src", "--segments", segs, "--price", "168"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert calls == ["src"] and out["location_checked"] is True
    assert out["band"]["location_verdict"] == "ok"                # default calibrates
    assert sl.main(["review", "--source", "src", "--segments", segs, "--price", "168",
                    "--config", '{"band_y": [0.5, 0.72]}']) == 0
    out = json.loads(capsys.readouterr().out)
    assert calls == ["src"] and out["location_checked"] is False


def test_out_of_band_check_also_warns_for_a_competing_position():
    loc = _loc_at(0.60, 0.64, (0.54, 0.70))
    loc["positions"].append({"y_range": [0.80, 0.84], "support": 0.08, "caption_like": True,
                             "y0_p5": 0.80, "y1_p95": 0.84, "band_y": [0.74, 0.90]})
    loc["competing"] = [1]
    out = sl.out_of_band_check(loc)
    assert out["outside"] == [[0.74, 0.90]] and out["warning"]


def test_unconfirmed_text_outside_a_calibrating_default_band_needs_an_eye_check(fake_scans):
    # Default band calibrates (captions at 0.60-0.64); the locator also saw a
    # position at 0.80 the detector does not confirm (empty scan there).
    fake_scans["src"] = _scan(_captions(0.0, 12.0), 12.0)
    calls = {"n": 0}
    import app.qa_tools as qt
    real = qt.scan_captions

    def scan(path, *, config=None, **k):
        lo, hi = qt.caption_config(config).band_y
        if lo > 0.72:
            return _scan({}, 12.0)
        return real(path, config=config, **k)
    qt.scan_captions = scan
    try:
        loc = _loc_at(0.80, 0.84, (0.74, 0.90), state="ambiguous")
        loc["positions"].append({"y_range": [0.60, 0.64], "support": 0.6, "caption_like": True,
                                 "y0_p5": 0.60, "y1_p95": 0.64, "band_y": [0.54, 0.70],
                                 "sample_times": [1.0], "windows_hit": 5})
        out = sl.review_segmentation("src", SEGS, price_per_segment=168.0, location=loc)
    finally:
        qt.scan_captions = real
    band = out["band"]
    assert band["location_verdict"] == "not_found" and out["calibration"]["calibrated"]
    assert band["needs_visual_check"] and "not confirmed" in band["note"]
