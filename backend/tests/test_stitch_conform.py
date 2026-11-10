"""2026-11-02 (514067f2 / d8105ab5): every clip is cut to its segment's slot in
the default-timeline stitch, so a model clip that came back a few frames long
no longer pushes every later segment late against the source soundtrack; the
QA measures each segment's picture offset and attributes caption leftovers
only where the shift cannot make them a neighbour's.

The media test stitches real (tiny, synthetic) clips with ffmpeg and checks
the frame count and the cut position frame by frame."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import cv2
import pytest

from app import subtitle_loop as sl
from app.autopilot.flows import subtitle_removal_plan as P
from app.pipeline_v2 import stitch_frame_counts

FFMPEG = "/usr/bin/ffmpeg"


def _ffmpeg_available() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _color(path, color, duration, fps, audio=False):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
           f"color=c={color}:size=64x64:rate={fps}:duration={duration}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True, capture_output=True)


def _frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.append(img[32, 32].tolist())
    cap.release()
    return out


def _is_blue(px):
    b, g, r = px
    return b > 150 and r < 100


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
@pytest.mark.parametrize("conform", [False, True])
def test_stitch_cuts_a_long_24fps_clip_to_its_slot(tmp_path, conform):
    from app import media

    src = str(tmp_path / "src.mp4")
    a = str(tmp_path / "a.mp4")          # segment 0 = 0.0-1.4 s: the model returned 1.5 s
    b = str(tmp_path / "b.mp4")          # segment 1 = 1.4-3.0 s: exact
    _color(src, "gray", 3.0, 30, audio=True)
    _color(a, "red", 1.5, 24)
    _color(b, "blue", 1.6, 30)
    counts = stitch_frame_counts(
        [SimpleNamespace(index=0, start_sec=0.0, end_sec=1.4),
         SimpleNamespace(index=1, start_sec=1.4, end_sec=3.0)], 30.0,
        plan=None, audio_mode="original")
    assert counts == [42, 48]
    dst = str(tmp_path / "final.mp4")
    media.stitch([a, b], src, dst, 64, 64, 30.0, "original",
                 frame_counts=counts if conform else None)
    px = _frames(dst)
    first_blue = next(i for i, p in enumerate(px) if _is_blue(p))
    if conform:
        assert first_blue == 42                     # segment 1 starts in its own slot
        assert len(px) == 90                        # exactly the source's frames
    else:
        assert first_blue == 45                     # 3 frames late (the old behaviour)


def test_stitch_frame_counts_only_for_the_default_timeline_with_the_source_sound():
    defs = [SimpleNamespace(index=0, start_sec=0.0, end_sec=4.4),
            SimpleNamespace(index=1, start_sec=4.4, end_sec=13.233333333333333),
            SimpleNamespace(index=2, start_sec=13.233333333333333, end_sec=19.133333333333333)]
    assert stitch_frame_counts(defs, 30.0, plan=None, audio_mode="original") == [132, 265, 177]
    assert stitch_frame_counts(defs, 30.0, plan=[], audio_mode="original") is None
    assert stitch_frame_counts(defs, 30.0, plan=None, audio_mode="seedance") is None
    rolled = defs[:1] + [SimpleNamespace(index=1, start_sec=4.4, end_sec=8.0, pre_roll_sec=0.5)]
    assert stitch_frame_counts(rolled, 30.0, plan=None, audio_mode="original") is None


def test_stitch_frame_counts_use_the_rebuild_frame_grid():
    """Half-frame boundaries round half UP, like an images-only run's keep cuts
    (rebuild.frame_index): 0.15 s at 30 fps is frame 5, not Python's 4."""
    from app.rebuild import frame_index
    defs = [SimpleNamespace(index=0, start_sec=0.0, end_sec=0.15),
            SimpleNamespace(index=1, start_sec=0.15, end_sec=1.0)]
    counts = stitch_frame_counts(defs, 30.0, plan=None, audio_mode="original")
    assert counts == [5, 25]
    assert counts == [frame_index(d.end_sec, 30.0) - frame_index(d.start_sec, 30.0)
                      for d in defs]


def test_stitch_rejects_frame_counts_outside_the_original_mode(tmp_path):
    from app import media

    with pytest.raises(ValueError):
        media._stitch_into(["x"], "s", str(tmp_path / "o.mp4"), 64, 64, 30.0, "seedance",
                           frame_counts=[3])
    with pytest.raises(ValueError):
        media._stitch_into(["x", "y"], "s", str(tmp_path / "o.mp4"), 64, 64, 30.0, "original",
                           frame_counts=[3])


# ---------------------------------------------------------------------------
# per-segment timing (the real d8105ab5 offsets)
# ---------------------------------------------------------------------------

BOUNDS = [0.0, 4.4, 13.233333333333333, 19.133333333333333, 26.9, 34.666666666666664,
          38.833333333333336]
SEGS = [sl.Seg(index=i, start=BOUNDS[i], end=BOUNDS[i + 1], run_segment_id=f"rs{i}")
        for i in range(6)]


def _pts(per_segment):
    out = []
    for i in range(1, 78):
        t = i * 0.5
        seg = next(k for k in range(6) if BOUNDS[k] <= t < BOUNDS[k + 1])
        out.append({"t": t, "offset": per_segment[seg]})
    return out


def test_segment_timing_measures_each_segment():
    rows = sl.segment_timing(_pts([0, -3, -3, -6, -8, -10]), SEGS)
    assert [r["offset_frames"] for r in rows] == [0, -3, -3, -6, -8, -10]
    assert [r["status"] for r in rows] == ["in_sync"] + ["off"] * 5
    short = sl.segment_timing(_pts([0] * 6), [sl.Seg(index=0, start=1.0, end=1.6)])
    assert short[0]["status"] == "unknown"           # too short to measure
    far = sl.segment_timing(_pts([0, 0, 0, 0, -15, -15]), SEGS)
    assert far[4]["beyond_search"] and far[4]["status"] == "off"


def _qa_rows(offsets, leftovers=None):
    timing = {"fps": 30.0, "length_ok": True,
              "segments": sl.segment_timing(_pts(offsets), SEGS)}
    rows = [{"index": i, "run_segment_id": f"rs{i}", "start_sec": round(BOUNDS[i], 3),
             "end_sec": round(BOUNDS[i + 1], 3), "action": "swap", "verdict": "clean",
             "leftovers": [], "timing": timing["segments"][i]} for i in range(6)]
    for i, spans in (leftovers or {}).items():
        rows[i]["verdict"] = "leftover"
        rows[i]["leftovers"] = [{"start_sec": a, "end_sec": b, "frames": 5} for a, b in spans]
    return {"timing": timing, "segments": rows, "timeline_verified": False}


def test_attribution_uses_the_shift_at_each_cut():
    qa = _qa_rows([0, -3, -3, -6, -8, -10], {3: [(21.4, 22.3), (22.9, 24.4)],
                                            4: [(26.95, 27.1), (32.4, 34.1)]})
    att3 = sl.attribution(qa, qa["segments"][3])
    assert len(att3["own"]) == 2 and not att3["ambiguous"]
    att4 = sl.attribution(qa, qa["segments"][4])
    # 26.95 s: within segment 4's own 8-frame shift (+0.1 s) of the cut at 26.9
    assert [a["span"]["start_sec"] for a in att4["ambiguous"]] == [26.95]
    assert att4["ambiguous"][0]["neighbour"] == 3
    assert [s["start_sec"] for s in att4["own"]] == [32.4]


def test_timing_verified_needs_every_measured_swap_segment_in_sync():
    qa = _qa_rows([0, 0, 0, 0, 0, 0])
    assert sl.timing_verified(qa["timing"], qa["segments"], False)
    qa = _qa_rows([0, 0, 0, -2, 0, 0])
    assert not sl.timing_verified(qa["timing"], qa["segments"], True)
    qa = _qa_rows([0, 0, 0, 0, 0, 0])
    qa["segments"][2]["timing"] = dict(qa["segments"][2]["timing"], status="unknown")
    assert not sl.timing_verified(qa["timing"], qa["segments"], False)
    assert sl.timing_verified(qa["timing"], qa["segments"], True)


def test_fill_proposal_rebuilds_an_out_of_sync_segment_and_defers_an_ambiguous_one():
    qa = _qa_rows([0, 0, 0, 0, -8, 0], {3: [(26.6, 26.85)]})
    for r in qa["segments"]:
        r["source_hits"] = 40
    fill = sl.fill_proposal("r", qa, SEGS, context=None, rebuild=[4], reserve_usd=0.10)
    items = {i["index"]: i for i in fill["segments"]}
    assert items[4]["state"] == "propose_preview" and items[4]["why"] == "sync"
    assert items[3]["state"] == "ambiguous" and "segment 4" in items[3]["reason"]
    assert fill["next"]["index"] == 4
    qa["segments"][4]["source_hits"] = 0
    fill = sl.fill_proposal("r", qa, SEGS, context=None, rebuild=[4], reserve_usd=0.10)
    assert {i["index"]: i["state"] for i in fill["segments"]}[4] == "operator"


def test_stitch_drift_is_explained_by_the_clips_extra_frames():
    qa = _qa_rows([0, -3, -3, -6, -8, -10])
    # the real clip lengths at 30 fps: Omni 135/180/235/235/125 vs slots 132/177/233/233/125
    qa["timing"]["clips"] = [
        {"index": 0, "excess_frames": 3}, {"index": 1, "excess_frames": 0},
        {"index": 2, "excess_frames": 3}, {"index": 3, "excess_frames": 2},
        {"index": 4, "excess_frames": 2}, {"index": 5, "excess_frames": 0}]
    d = P.stitch_drift(qa, 30.0)
    assert d["off"] == [1, 2, 3, 4, 5] and d["explained"]
    assert d["expected"] == {0: 0.0, 1: -3.0, 2: -3.0, 3: -6.0, 4: -8.0, 5: -10.0}
    qa2 = _qa_rows([0, 0, 0, -6, 0, 0])
    qa2["timing"]["clips"] = qa["timing"]["clips"]
    assert not P.stitch_drift(qa2, 30.0)["explained"]   # the model's own picture is shifted


def test_an_unbounded_shift_makes_its_half_of_the_segment_doubtful():
    """Gate R1-1 / R2-2: at the search limit (or unmeasured on an unverified
    timeline) the shift has no bound: every span of the segment waits."""
    qa = _qa_rows([0, 0, 0, 0, -15, 0], {3: [(20.0, 20.5), (25.9, 26.2)]})
    qa["segments"][4]["timing"]["beyond_search"] = True
    att = sl.attribution(qa, qa["segments"][3])
    assert att["own"] == []                                  # the cut may sit anywhere (R2-2)
    assert [a["span"]["start_sec"] for a in att["ambiguous"]] == [20.0, 25.9]
    qa = _qa_rows([0, 0, 0, 0, 0, 0], {3: [(20.0, 20.5)]})
    qa["segments"][3]["timing"] = dict(qa["segments"][3]["timing"], status="unknown",
                                       offset_frames=None)
    qa["alignment"] = {"max_abs_window_median": 0.0}
    assert sl.attribution(qa, qa["segments"][3])["ambiguous"]          # unverified: unbounded
    qa["timeline_verified"] = True
    assert not sl.attribution(qa, qa["segments"][3])["ambiguous"]      # the windows bound it


def test_stitch_fps_is_the_capped_source_rate():
    """QA and the restitch admission count frames at the rate the stitch uses."""
    from app import media
    assert media.stitch_fps(60.0) == media._MAX_FPS == 30.0
    assert media.stitch_fps(24.0) == 24.0


def test_stitch_drift_converts_clip_frames_at_the_stitch_rate():
    """A 60 fps source is stitched at 30: 8 excess stitch frames are 8 QA
    frames at 30 fps, not 4."""
    qa = {"timing": {"fps": 30.0, "clips": [
              {"index": 0, "excess_frames": 8}, {"index": 1, "excess_frames": 0}]},
          "segments": [
              {"index": 0, "start_sec": 0.0, "action": "swap", "timing": {"status": "in_sync"}},
              {"index": 1, "start_sec": 5.0, "action": "swap", "timing": {"status": "off"}}]}
    out = P.stitch_drift(qa, 60.0)
    assert out["expected"][1] == -8.0
