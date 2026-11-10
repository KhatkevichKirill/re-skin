"""Tests for app/qa_tools.py (AICP-343) on small synthetic clips.

The caption clips are drawn with OpenCV (white text, black outline) and piped
into ffmpeg, so no font or drawtext filter is needed. The alignment clips are
ffmpeg ``testsrc2`` (every frame differs), shifted by whole frames.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from app import qa_tools as q

sys.path.insert(0, os.path.dirname(__file__))
from qa_caption_reference import legacy_caption_score  # noqa: E402
from app.media import FFMPEG


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


def _encode(frames, path, fps=10):
    h, w = frames[0].shape[:2]
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
         str(path)],
        input=b"".join(f.tobytes() for f in frames), capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    return str(path)


def _caption_frame(text=None, *, y_frac=0.6, w=540, h=960):
    img = np.full((h, w, 3), 128, np.uint8)  # mid grey: neither "white" nor "dark"
    if text:
        # Glyph mask drawn in 255 on one channel, black outline grown from it
        # with a dilation (OpenCV 5 putText does not draw a (0, 0, 0) colour).
        mask = np.zeros((h, w), np.uint8)
        cv2.putText(mask, text, (int(w * 0.33), int(h * y_frac)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3, cv2.LINE_8)
        outline = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        img[outline > 0] = 0
        img[mask > 0] = 255
    return img


def _caption_clip(path, *, seconds=3.0, fps=10, text_from=1.0, text_to=2.0,
                  y_frac=0.6, text="WORD"):
    frames = []
    for i in range(int(seconds * fps)):
        t = i / fps
        frames.append(_caption_frame(text if text_from <= t < text_to else None, y_frac=y_frac))
    return _encode(frames, path, fps)


def _testsrc(path, *, seconds=4, size="160x120", fps=30, trim_frames=0):
    vf = f"trim=start_frame={trim_frames},setpts=PTS-STARTPTS" if trim_frames else "null"
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=duration={seconds}:size={size}:rate={fps}",
         "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
         "-pix_fmt", "yuv444p", str(path)], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    return str(path)


# ---------------------------------------------------------------------------
# caption detector
# ---------------------------------------------------------------------------

def test_caption_score_on_arrays():
    assert q.caption_score(_caption_frame())[0] == 0
    score, box = q.caption_score(_caption_frame("WORD"))
    assert score > q.CAPTION_HIT_THRESHOLD
    x0, y0, x1, y1 = box
    assert 0.5 < y0 < y1 < 0.72 and 0.12 <= x0 < x1 <= 0.88
    # Text above the band is not a caption here.
    assert q.caption_score(_caption_frame("WORD", y_frac=0.25))[0] == 0
    # White text without an outline is not either.
    plain = _caption_frame("WORD")
    plain[plain == 0] = 128
    assert q.caption_score(plain)[0] == 0


@needs_ffmpeg
def test_scan_finds_the_captioned_second(tmp_path):
    clip = _caption_clip(tmp_path / "cap.mp4")
    scan = q.scan_captions(clip)
    assert scan.frames == 30
    assert scan.hits == 10
    assert len(scan.spans) == 1
    span = scan.spans[0]
    assert span.start == pytest.approx(1.0) and span.end == pytest.approx(1.9)
    assert span.frames == 10
    geo = q.caption_geometry(scan)
    assert geo["n"] == 10 and 0.5 < geo["y0_p5"] < 0.72
    # offset/start put the times on the caller's timeline
    shifted = q.scan_captions(clip, offset=40.0, start=1.0, duration=1.0)
    assert shifted.hits == 10
    assert shifted.spans[0].start == pytest.approx(41.0)


@needs_ffmpeg
def test_spans_merge_short_gaps_only(tmp_path):
    frames = [_caption_frame("AB" if (5 <= i < 10 or 13 <= i < 16 or 30 <= i < 33) else None)
              for i in range(40)]
    scan = q.scan_captions(_encode(frames, tmp_path / "gaps.mp4"))
    assert [(s.start, s.end) for s in scan.spans] == [(0.5, 1.5), (3.0, 3.2)]


@needs_ffmpeg
def test_detect_verdicts(tmp_path):
    captioned = _caption_clip(tmp_path / "src.mp4", text_from=0.0, text_to=3.0)
    clean = _caption_clip(tmp_path / "clean.mp4", text_from=9, text_to=9)
    leftover = _caption_clip(tmp_path / "left.mp4", text_from=2.0, text_to=2.5)

    ok = q.detect_captions(clean, calibrate=captioned)
    assert ok.verdict == "clean" and ok.calibration["calibrated"]
    assert ok.calibration["hit_ratio"] == 1.0

    bad = q.detect_captions(leftover, calibrate=captioned)
    assert bad.verdict == "captions_found" and bad.hits == 5

    # A source the detector cannot see captions on proves nothing.
    blind = q.detect_captions(clean, calibrate=clean)
    assert blind.verdict == "clean_uncalibrated" and not blind.calibration["calibrated"]
    assert q.detect_captions(clean).verdict == "clean_uncalibrated"
    assert "scores" not in ok.to_dict() and len(ok.to_dict(include_scores=True)["scores"]) == 30


@needs_ffmpeg
def test_decode_failure_raises(tmp_path):
    bogus = tmp_path / "x.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(q.QAToolError):
        q.scan_captions(str(bogus))
    clip = _caption_clip(tmp_path / "cap.mp4")
    with pytest.raises(q.QAToolError, match="no frames"):
        q.detect_captions(clip, calibrate=clip, start=50.0)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_alignment_identical_and_shifted(tmp_path):
    src = _testsrc(tmp_path / "src.mp4")
    same = q.alignment_check(src, src)
    assert same["points"] == len(range(15, 120 - 15, 15))
    assert same["nonzero"] == [] and same["in_sync"] and same["frame_count_delta"] == 0

    early = _testsrc(tmp_path / "early.mp4", trim_frames=2)  # shows source frame k+2 at k
    res = q.alignment_check(src, early)
    assert {o for _, o in res["nonzero"]} == {2}
    assert not res["in_sync"] and res["max_abs_window_median"] == 2
    assert res["frame_count_delta"] < 0
    assert not res["length_ok"]
    loose = q.alignment_check(src, early, tolerance=3)
    assert loose["in_sync"] and loose["outliers"] == [] and loose["length_ok"]


@needs_ffmpeg
def test_alignment_static_ties_and_missing_tail(tmp_path):
    still = tmp_path / "still.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=gray:size=72x128:rate=30:duration=3", "-c:v", "libx264",
                    "-qp", "0", str(still)], check=True)
    res = q.alignment_check(str(still), str(still))
    assert res["nonzero"] == [] and res["in_sync"]  # ties resolve to offset 0

    src = _testsrc(tmp_path / "src.mp4", seconds=6)
    short = tmp_path / "short.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, "-t", "3", "-c:v", "libx264",
                    "-qp", "0", "-pix_fmt", "yuv444p", str(short)], check=True)
    res = q.alignment_check(src, str(short))
    assert res["nonzero"] == [] and not res["length_ok"] and not res["in_sync"]


@needs_ffmpeg
def test_match_times_finds_the_window(tmp_path):
    src = _testsrc(tmp_path / "src.mp4", seconds=5)
    clip = tmp_path / "clip.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", "2", "-i", src, "-t", "2",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(clip)], check=True)
    res = q.match_times(src, str(clip), source_start=2.0, duration=2.0)
    assert res["samples"]
    assert all(abs(s["drift"]) <= 0.04 for s in res["samples"])
    assert res["samples"][0]["source_t"] == pytest.approx(2.2, abs=0.04)
    # A clip longer than its window is analysed to its real end.
    longer = q.match_times(src, str(clip), source_start=2.0, duration=1.0)
    assert longer["result_duration"] == pytest.approx(2.0, abs=0.05)
    assert longer["samples"][-1]["result_t"] > 1.5


# ---------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_comparison_sheet(tmp_path):
    src = _testsrc(tmp_path / "src.mp4", size="120x160")
    out = tmp_path / "sheet.jpg"
    res = q.comparison_sheet(src, src, str(out), cell_width=60)
    assert out.exists() and len(res["samples"]) == 3
    assert all(s["diff"] == 0.0 for s in res["samples"])
    img = cv2.imread(str(out))
    assert img.shape[1] == 60 * 6 and img.shape[0] == res["height"]

    # Per-segment clip of a different length: "scale" maps the window onto it.
    clip = tmp_path / "clip.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", "1", "-i", src, "-t", "1",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(clip)], check=True)
    res = q.comparison_sheet(src, src, str(out), cell_width=60, time_map="scale",
                             windows=[{"source_start": 1.0, "source_end": 3.0,
                                       "result": str(clip), "label": "seg2"}],
                             fractions=(0.5,))
    (sample,) = res["samples"]
    assert sample["window"] == "seg2" and sample["source_t"] == pytest.approx(2.0)
    assert sample["result_t"] == pytest.approx(0.5, abs=0.05)
    with pytest.raises(q.QAToolError):
        q.comparison_sheet(src, src, str(out), windows=[{"source_start": 2, "source_end": 1}])


@needs_ffmpeg
def test_contact_sheet_and_missing_frame(tmp_path):
    src = _testsrc(tmp_path / "src.mp4")
    out = tmp_path / "c.jpg"
    res = q.contact_sheet(src, str(out), every_sec=1.0, cell_width=64, columns=2)
    assert res["times"] == [0.5, 1.5, 2.5, 3.5] and out.exists()
    res = q.contact_sheet(src, str(out), times=[1.0, 99.0], cell_width=64)
    assert res["times"] == [1.0, 99.0]  # past the end -> a "(none)" cell, no crash


# ---------------------------------------------------------------------------
# captions from the script
# ---------------------------------------------------------------------------

def test_parse_dialogue():
    prompt = (
        "Keep the person.\n\nDialogue — spoken strictly in this order:\n"
        "[0.0–2.4] WOMAN: Dzięki temu czytam.\n"
        "[7.2-13.9] WOMAN (her face is no longer visible — it is still HER voice): Ma też: funkcje.\n"
        "[6.9–8.2] Nobody speaks. Mouths stay closed.\n"
        "**older woman:** not timed\n")
    lines = q.parse_dialogue(prompt)
    assert [(l.start, l.end, l.speaker, l.text) for l in lines] == [
        (0.0, 2.4, "WOMAN", "Dzięki temu czytam."),
        (7.2, 13.9, "WOMAN", "Ma też: funkcje."),
    ]
    assert q.parse_dialogue("") == [] and q.parse_dialogue(None) == []


def test_captions_scale_offset_gap_and_keep_pieces():
    segs = [
        # clip came back 1 % short
        {"source_start": 0.0, "source_end": 10.0, "clip_duration": 9.9,
         "prompt": "[0.0–5.0] A: one\n[5.0–10.0] B: two"},
        # a keep piece: no lines, just time on the final
        {"source_start": 10.0, "source_end": 12.0},
        {"source_start": 12.0, "source_end": 14.0, "clip_duration": 2.0,
         "lines": [{"start": 0.5, "end": 1.5, "text": "  three \n words "}]},
    ]
    assert q.captions_from_script(segs) == [
        {"start": 0.0, "end": 4.9, "text": "one"},
        {"start": 4.95, "end": 9.85, "text": "two"},
        {"start": 12.4, "end": 13.35, "text": "three words"},
    ]
    # explicit offset wins over the running cursor
    out = q.captions_from_script([dict(segs[2], offset=100.0)], gap_sec=0)
    assert out == [{"start": 100.5, "end": 101.5, "text": "three words"}]


def test_captions_clamp_to_window_and_clip():
    out = q.captions_from_script([{
        "source_start": 0.0, "source_end": 5.0, "clip_duration": 5.0,
        # 5.05 is .1f rounding of the window end -> kept; 9.0 is past it -> clamped
        "prompt": "[4.0–5.05] A: edge\n[4.5–9.0] B: beyond\n[7.0–9.0] C: outside"}],
        gap_sec=0.05)
    assert out[0] == {"start": 4.0, "end": 5.0, "text": "edge"}
    assert out[1] == {"start": 4.5, "end": 5.0, "text": "beyond"}
    assert len(out) == 2  # "outside" starts after the clip end


@pytest.mark.parametrize("bad", [
    [{"source_start": 5, "source_end": 5}],
    [{"source_start": "x", "source_end": 5}],
    [{"source_start": 0, "source_end": 5, "clip_duration": 0}],
    [{"source_start": 0, "source_end": 5, "lines": [{"start": 0, "end": 1}]}],
    [{"source_start": 0, "source_end": float("nan")}],
    [{"source_start": 0, "source_end": 5, "lines": [{"start": 1, "end": 0.5, "text": "x"}]}],
    [{"source_start": 0, "source_end": 5, "prompt": "[2.0–1.0] A: reversed"}],
    [{"source_start": 0, "source_end": 5, "prompt": 42}],
    ["not a dict"],
])
def test_captions_reject_bad_specs(bad):
    with pytest.raises(q.QAToolError):
        q.captions_from_script(bad)


# The AICP-332 Polish run 6c4a9818: the timed dialogue of its 7 segment prompts,
# the run's segment snapshot and the real clip lengths (ffprobe of the probe
# 2497466c result_0000 + 6c4a9818 result_0001..0006). The expected lines are
# refs/caption_lines_script.json, which the operator burned on 2026-10-29.
_PL_SEGMENTS = [
    (0.0, 12.1333, 12.064, [
        "[0.0–2.4] WOMAN: Dzięki temu czytam o wiele więcej książek.",
        "[2.4–4.2] INTERVIEWER: Jak pani czyta tyle książek?",
        "[4.3–6.2] WOMAN: Właściwie nie czytam, słucham.",
        "[6.3–7.3] INTERVIEWER: Słucha pani?",
        "[7.5–12.1] WOMAN: Mam aplikację. Robię zdjęcie strony, a ona od razu czyta ją na głos."]),
    (12.1333, 26.0333, 13.728, [
        "[0.9–3.7] APP: Innymi słowy, jest bardzo przewidywalna.",
        "[3.8–5.1] INTERVIEWER: To Gwyneth Paltrow?",
        "[5.1–7.2] WOMAN: Tak, to mój ulubiony głos.",
        "[7.2–13.9] WOMAN (her face is no longer visible — the camera looks over her shoulder at "
        "the phone in HER hands; it is still HER voice, the older woman, NOT the interviewer): "
        "Ma też inne funkcje, więc mogę zmienić tempo — przyspieszyć albo zwolnić czytanie."]),
    (26.0333, 35.9, 9.728, [
        "[0.2–5.0] WOMAN: Mogę też zmienić głos. Czasami słucham, jak książki czyta Snoop Dogg.",
        "[5.0–7.3] APP: Mutacje genetyczne sprzyjały odstępom.",
        "[7.3–9.8] INTERVIEWER: Lubi pani, gdy Snoop Dogg czyta książki?"]),
    (35.9, 49.45, 13.376, [
        "[0.0–5.4] WOMAN: Tak, jest uroczy. Wieczorem, gdy jestem zmęczona, jego głos naprawdę "
        "świetnie mnie odpręża.",
        "[5.4–7.2] INTERVIEWER: To musi naprawdę wciągać.",
        "[7.2–13.6] WOMAN: Zdecydowanie. Dzięki temu czytam o wiele więcej, a książek mogę "
        "słuchać nawet na spacerze."]),
    (49.45, 57.7, 8.064, [
        "[0.6–8.2] WOMAN: Kiedy jestem zmęczona i nie chce mi się czytać, nie potrzebuję okularów. "
        "Nie muszę też siedzieć przy biurku."]),
    (57.7, 65.95, 8.064, [
        "[0.5–5.6] WOMAN: Jak się uczę, łatwiej mi się skupić, bo mogę jednocześnie czytać i słuchać.",
        "[5.6–6.9] INTERVIEWER: Czego się pani uczy?",
        "[6.9–8.2] Nobody speaks. Mouths stay closed."]),
    (65.95, 78.854966, 12.736, [
        "[0.0–5.5] WOMAN: Przygotowuję się do egzaminu na pośrednika nieruchomości, więc uczę się "
        "na spacerach.",
        "[5.5–8.3] INTERVIEWER: Niesamowite! Jak nazywa się ta aplikacja?",
        "[8.3–11.2] WOMAN: To Listenify, możesz ją pobrać z App Store.",
        "[11.2–12.5] INTERVIEWER: Dziękuję."]),
]

_PL_EXPECTED = [
    (0.0, 2.34), (2.39, 4.13), (4.28, 6.11), (6.26, 7.21), (7.46, 11.98), (12.95, 15.67),
    (15.82, 17.05), (17.1, 19.12), (19.17, 25.74), (25.99, 30.67), (30.72, 32.94),
    (32.99, 35.4), (35.52, 40.8), (40.85, 42.58), (42.63, 48.9), (49.48, 56.86),
    (57.45, 62.38), (62.43, 63.65), (65.02, 70.4), (70.45, 73.17), (73.22, 76.03),
    (76.08, 77.31),
]


def test_captions_reproduce_the_polish_run():
    segs = [{"source_start": a, "source_end": b, "clip_duration": d,
             "prompt": "Header text.\n\n" + "\n".join(lines)}
            for a, b, d, lines in _PL_SEGMENTS]
    out = q.captions_from_script(segs)
    assert [(l["start"], l["end"]) for l in out] == _PL_EXPECTED
    assert out[8]["text"].startswith("Ma też inne funkcje")
    assert out[-1]["text"] == "Dziękuję."


@needs_ffmpeg
def test_captions_probe_clip_duration(tmp_path):
    clip = _testsrc(tmp_path / "clip.mp4", seconds=2)
    out = q.captions_from_script([{"source_start": 0, "source_end": 4, "clip": clip,
                                   "prompt": "[0.0–4.0] A: half speed"}], gap_sec=0)
    assert out == [{"start": 0.0, "end": 2.0, "text": "half speed"}]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_script_captions(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"segments": [
        {"source_start": 0, "source_end": 2, "prompt": "[0.0–1.0] A: hi"}], "gap_sec": 0}))
    assert q.main(["script-captions", str(spec)]) == 0
    assert json.loads(capsys.readouterr().out) == {"lines": [{"start": 0.0, "end": 1.0, "text": "hi"}]}

    spec.write_text(json.dumps({"segments": [{"source_start": 0, "source_end": 2, "prompt": "no dialogue"}]}))
    assert q.main(["script-captions", str(spec)]) == 2
    assert "no caption lines" in capsys.readouterr().err

    spec.write_text(json.dumps({"segments": "nope"}))
    assert q.main(["script-captions", str(spec)]) == 2
    assert "error:" in capsys.readouterr().err


@needs_ffmpeg
def test_cli_captions_and_align(tmp_path, capsys):
    clip = _caption_clip(tmp_path / "cap.mp4")
    assert q.main(["captions", clip, "--calibrate", clip]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hits"] == 10 and out["verdict"] == "captions_found" and "scores" not in out
    src = _testsrc(tmp_path / "src.mp4")
    assert q.main(["align", src, src]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["in_sync"] and "_points" not in out and "all_points" not in out


@needs_ffmpeg
def test_comparison_sheet_shared_timeline_scale(tmp_path):
    src = _testsrc(tmp_path / "src.mp4", size="120x160")
    out = str(tmp_path / "s.jpg")
    # No own clip and no result_end: "scale" keeps 1:1 time on the shared timeline.
    res = q.comparison_sheet(src, src, out, cell_width=60, time_map="scale", fractions=(0.5,),
                             windows=[{"source_start": 1.0, "source_end": 2.0}])
    assert res["samples"][0]["result_t"] == pytest.approx(1.5)
    res = q.comparison_sheet(src, src, out, cell_width=60, time_map="scale", fractions=(0.5,),
                             windows=[{"source_start": 1.0, "source_end": 2.0,
                                       "result_start": 2.0, "result_end": 4.0}])
    assert res["samples"][0]["result_t"] == pytest.approx(3.0)
    with pytest.raises(q.QAToolError):
        q.comparison_sheet(src, src, out, windows=["nope"])


# ---------------------------------------------------------------------------
# per-project caption detector config (AICP-343 follow-up)
# ---------------------------------------------------------------------------



def _coloured_caption(text_bgr, outline_bgr=None, *, y_frac=0.6, bg=128):
    img = np.full((960, 540, 3), bg, np.uint8)
    mask = np.zeros((960, 540), np.uint8)
    cv2.putText(mask, "WORD", (180, int(960 * y_frac)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3)
    if outline_bgr is not None:
        img[cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3) > 0] = outline_bgr
    img[mask > 0] = text_bgr
    return img


def test_default_config_is_the_legacy_rule_exactly():
    rng = np.random.default_rng(343)
    frames = [_caption_frame("WORD"), _caption_frame(), _caption_frame("AB", y_frac=0.66)]
    for _ in range(12):
        f = rng.integers(0, 256, (960, 540, 3), dtype=np.uint8)
        # sprinkle near-white and near-black blocks so every branch is exercised
        for _ in range(30):
            y, x = int(rng.integers(470, 700)), int(rng.integers(60, 470))
            f[y:y + int(rng.integers(3, 50)), x:x + int(rng.integers(3, 90))] = \
                rng.integers(220, 256) if rng.random() < 0.5 else rng.integers(0, 70)
        frames.append(f)
    for f in frames:
        assert q.caption_score(f) == legacy_caption_score(f)
        assert q.caption_score(f, q.caption_config({})) == legacy_caption_score(f)


def test_config_validation_and_storage():
    cfg = q.caption_config({"text_color": "#FF0", "outline_color": "BLACK"})
    assert cfg.text_color == "#ffff00" and cfg.outline_color == "#000000"
    assert cfg.stored() == {"text_color": "#ffff00"}
    assert q.caption_config(None).stored() is None
    assert q.caption_config({"text_color": "green"}).text_color == "#008000"
    assert q.caption_config({"text_color": "lime"}).text_color == "#00ff00"
    assert q.caption_config({"threshold": 40}).stored() is None
    # An explicitly given band is kept even when it is the default (an operator
    # choice; "no band_y" means "locate the captions").
    assert q.caption_config({"band_y": [0.5, 0.72]}).stored() == {"band_y": [0.5, 0.72]}
    assert q.has_band({"band_y": [0.5, 0.72]}) and not q.has_band(None)
    assert not q.has_band({"text_color": "#ffff00"})
    for bad in ({"band_x": [0.5, 0.51]}, {"text_color": "#ggg"}, {"bogus": 1},
                {"outline_width_px": 0}, {"glyph_height_px": [1, 5]}, ["x"]):
        with pytest.raises(q.QAToolError):
            q.caption_config(bad)


def test_config_finds_other_caption_styles():
    yellow = _coloured_caption((0, 255, 255), (0, 0, 0))[:, :, ::-1].copy()  # RGB
    assert q.caption_score(yellow)[0] == 0
    assert q.caption_score(yellow, {"text_color": "yellow"})[0] > q.CAPTION_HIT_THRESHOLD

    plain = _coloured_caption((255, 255, 255))[:, :, ::-1].copy()
    assert q.caption_score(plain)[0] == 0
    assert q.caption_score(plain, {"outline_style": "none"})[0] > q.CAPTION_HIT_THRESHOLD

    top = _caption_frame("WORD", y_frac=0.2)[:, :, ::-1].copy()
    assert q.caption_score(top)[0] == 0
    assert q.caption_score(top, {"band_y": [0.1, 0.3]})[0] > q.CAPTION_HIT_THRESHOLD

    score = q.caption_score(_caption_frame("WORD"))[0]
    assert q.caption_score(_caption_frame("WORD"), {"glyph_height_px": [2, 5]})[0] == 0
    assert score > 0


@needs_ffmpeg
def test_detect_with_config_and_threshold_override(tmp_path):
    frames = [_coloured_caption((0, 255, 255), (0, 0, 0)) for _ in range(10)]
    src = _encode(frames, tmp_path / "yellow.mp4")
    clean = _caption_clip(tmp_path / "clean.mp4", text_from=9, text_to=9)

    # Default config cannot see yellow captions: never "clean", always uncalibrated.
    assert q.detect_captions(clean, calibrate=src).verdict == "clean_uncalibrated"
    cfg = {"text_color": "#ffff00"}
    res = q.detect_captions(clean, calibrate=src, config=cfg)
    assert res.verdict == "clean" and res.config["text_color"] == "#ffff00"
    assert q.detect_captions(src, config=cfg).hits == 10
    # An explicit threshold beats the config's, for the calibration too.
    res = q.detect_captions(src, calibrate=src, config=cfg, threshold=100_000)
    assert res.hits == 0 and res.threshold == 100_000
    assert res.calibration["hits"] == 0 and res.verdict == "clean_uncalibrated"


@needs_ffmpeg
def test_cli_config_and_project(tmp_path, capsys, monkeypatch):
    frames = [_coloured_caption((0, 255, 255), (0, 0, 0)) for _ in range(5)]
    clip = _encode(frames, tmp_path / "yellow.mp4")
    assert q.main(["captions", clip]) == 0
    assert json.loads(capsys.readouterr().out)["hits"] == 0
    assert q.main(["captions", clip, "--config", '{"text_color": "yellow"}']) == 0
    assert json.loads(capsys.readouterr().out)["hits"] == 5
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text('{"text_color": "#ff0"}')
    assert q.main(["captions", clip, "--config", str(cfg_file)]) == 0
    assert json.loads(capsys.readouterr().out)["hits"] == 5

    monkeypatch.setattr(q, "project_caption_config", lambda pid: {"text_color": "yellow"})
    assert q.main(["captions", clip, "--project", "p1"]) == 0
    assert json.loads(capsys.readouterr().out)["hits"] == 5
    # --config replaces the project's config
    assert q.main(["captions", clip, "--project", "p1", "--config", "{}"]) == 0
    assert json.loads(capsys.readouterr().out)["hits"] == 0
    assert q.main(["captions", clip, "--config", '{"text_color": "nope"}']) == 2
    assert "invalid caption detector config" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Caption locator
# ---------------------------------------------------------------------------

def _scene_frame(word=None, *, y_frac=0.2, sign=None, w=540, h=960):
    """Grey frame; an optional centred caption word at *y_frac* and an
    off-centre static 'sign' (white text, black outline) at (x, y)."""
    img = np.full((h, w, 3), 128, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    if word:
        (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(mask, word, ((w - tw) // 2, int(h * y_frac)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, 255, 3, cv2.LINE_8)
    if sign:
        cv2.putText(mask, "SIGN", sign, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3, cv2.LINE_8)
    outline = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    img[outline > 0] = 0
    img[mask > 0] = 255
    return img


WORDS = ["HELLO", "BOOKS", "TO", "ALL", "MY", "FRIENDS", "YOU", "GOSH"]


def _located_clip(path, *, y_frac=0.2, seconds=12, fps=10, sign=(20, 600), words=True):
    frames = []
    for i in range(seconds * fps):
        word = WORDS[(i // 4) % len(WORDS)] if words and (i // 4) % 5 != 4 else None
        frames.append(_scene_frame(word, y_frac=y_frac, sign=sign))
    return _encode(frames, path, fps)


@needs_ffmpeg
def test_locate_finds_top_captions_and_ignores_a_static_off_centre_sign(tmp_path):
    clip = _located_clip(tmp_path / "top.mp4", y_frac=0.2)
    loc = q.locate_captions(clip, fps=2.0)
    assert loc["state"] == "located", loc["reasons"]
    lo, hi = loc["candidate_band_y"]
    assert lo <= 0.17 and hi >= 0.21 and hi - lo >= q.LOCATE_MIN_HEIGHT - 1e-9
    assert hi < 0.5                                       # nowhere near the sign at y 0.62
    main = loc["positions"][loc["members"][0]]
    assert main["support"] >= 0.6 and main["windows_hit"] == q.LOCATE_WINDOWS
    # The default band only sees the sign; the located band sees the captions.
    default = q.scan_captions(clip, fps=2.0)
    assert all(f.box is None or f.box[1] > 0.5 for f in default.scores)
    assert q.calibrate_captions(clip, fps=2.0, config={"band_y": loc["candidate_band_y"]})[
        "calibrated"]


@needs_ffmpeg
def test_locate_reports_none_when_only_static_text_is_seen(tmp_path):
    clip = _located_clip(tmp_path / "sign.mp4", words=False, sign=(170, 600))   # centred sign
    loc = q.locate_captions(clip, fps=2.0)
    assert loc["state"] == "none" and loc["candidate_band_y"] is None
    assert loc["positions"] and loc["positions"][0]["static_frames"] > 0
    assert loc["reasons"] and any("held still" in r for r in loc["reasons"])
    assert loc["competing"] == [0]                 # held text is still reported


@needs_ffmpeg
def test_locate_short_clip_is_ambiguous_not_located(tmp_path):
    clip = _located_clip(tmp_path / "short.mp4", y_frac=0.8, seconds=6)
    loc = q.locate_captions(clip, fps=2.0)
    assert loc["state"] == "ambiguous" and loc["candidate_band_y"]
    assert any("sampled frames" in r for r in loc["reasons"])


@needs_ffmpeg
def test_locate_cli_and_band_sheet(tmp_path, capsys):
    clip = _located_clip(tmp_path / "top.mp4", y_frac=0.2)
    sheet = str(tmp_path / "loc.jpg")
    assert q.main(["locate", clip, "--sheet", sheet]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "located" and out["sheet"] == sheet and os.path.getsize(sheet) > 0
    assert "Band y" in out["hint"]
    img = cv2.imread(sheet)
    assert img is not None and img.shape[0] > 0


def test_band_around_keeps_a_minimum_height_inside_the_frame():
    assert q._band_around(0.76, 0.80) == [0.70, 0.86]
    lo, hi = q._band_around(0.01, 0.03)
    assert lo == 0.0 and hi == pytest.approx(q.LOCATE_MIN_HEIGHT, abs=0.011)
    lo, hi = q._band_around(0.95, 0.99)
    assert hi == 1.0 and 1.0 - lo >= q.LOCATE_MIN_HEIGHT - 1e-9


@needs_ffmpeg
def test_locate_held_text_elsewhere_makes_it_ambiguous(tmp_path):
    # Changing captions at y 0.8 and a centred text held still at y 0.2 for
    # the whole clip (a hook bubble, or a held caption track).
    clip = _located_clip(tmp_path / "held.mp4", y_frac=0.8, sign=(200, 190))
    loc = q.locate_captions(clip, fps=2.0)
    assert loc["state"] == "ambiguous" and loc["candidate_band_y"][0] > 0.6
    assert any("held still" in r for r in loc["reasons"])
    held = [loc["positions"][i] for i in loc["competing"]]
    assert held and held[0]["static_frames"] > 0 and held[0]["band_y"][1] < 0.4


def test_location_sheet_shows_the_competitors_moments_and_bands(monkeypatch):
    got = {}
    monkeypatch.setattr(q, "contact_sheet", lambda path, out, **k: got.update(k) or {"out": out})
    loc = {"candidate_band_y": [0.70, 0.86], "members": [0], "competing": [1], "positions": [
        {"band_y": [0.70, 0.86], "sample_times": [1.0, 2.0, 3.0]},
        {"band_y": [0.12, 0.28], "sample_times": [20.0, 21.0]}]}
    assert q.location_sheet("v.mp4", "o.jpg", loc, other_bands=[[0.5, 0.72]])["out"] == "o.jpg"
    assert got["times"] == [1.0, 2.0, 3.0, 20.0, 21.0]
    assert got["bands"] == [[0.70, 0.86], [0.5, 0.72], [0.12, 0.28]]
    assert q._spread([1.0, 2.0, 3.0], 1) == [2.0] and q._spread([1.0], 0) == []


def test_locate_reports_weak_caption_like_text_as_a_competitor(monkeypatch):
    # 8 % of 50 frames with a changing centred line at y ~0.8, nothing else.
    frames = [[] for _ in range(50)]
    for i in (3, 9, 15, 22):
        frames[i] = [(0.40 + 0.01 * i % 3, 0.78, 0.60, 0.82, 200)]

    class Info:
        duration_sec, width, height = 25.0, 540, 960
    monkeypatch.setattr(q, "_probe", lambda p: Info())
    it = iter(frames)
    monkeypatch.setattr(q, "_iter_frames", lambda *a, **k: (np.zeros((2, 2, 3), np.uint8)
                                                           for _ in frames))
    monkeypatch.setattr(q, "_frame_clusters", lambda frame, cfg: next(it))
    loc = q.locate_captions("v.mp4", fps=2.0)
    assert loc["state"] == "none" and loc["competing"] == [0]
    assert loc["positions"][0]["support"] == pytest.approx(0.08)
