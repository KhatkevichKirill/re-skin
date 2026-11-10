"""AICP-366: the deterministic QA pieces of the localisation finishing flow.

Pure functions (script parsing, ASR check, caption word fixes, cues, comment
parsing, best take, tail cut) and the media side on small ffmpeg-generated
files: the voice pitch of a sine tone and the whole segment-QA compute (a
seeded gender error in segment 3, judged against the accepted probe's voice).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.autopilot.flows import localisation_qa as lqa  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def make_clip(path, seconds, hz, *, size="96x160", color="0x405060"):
    """A tiny H.264 + AAC clip: flat colour picture, a sine tone of *hz*."""
    subprocess.run([
        FFMPEG, "-nostdin", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=12:d={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency={hz}:sample_rate=16000:duration={seconds}",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", str(path),
    ], check=True)
    return str(path)


# ---------------------------------------------------------------------------
# Script lines
# ---------------------------------------------------------------------------

PROMPT = """Keep the main person exactly as they are.

Speakers:
- WOMAN: the older woman with glasses.
- APP: the phone's reading voice.

Dialogue — spoken strictly in this order, at these times (seconds from the start of this clip), nothing before or after:
[0.0–2.4] WOMAN: Acho que leio bem mais livros.
[2.4–4.2] INTERVIEWER (off-face: only her hand with the microphone is visible): Com licença, como lê tanto?
[3.0–6.0] APP (BACKGROUND — it plays quietly at the same time as the WOMAN's lines, underneath them; nobody stops talking for it and it is not lip-synced): Era uma vez.
[6.3–7.3] WOMAN: Uso o Listenify."""


def test_parse_script_keeps_the_background_marker():
    lines = lqa.parse_script(PROMPT)
    assert [ln.tag for ln in lines] == ["WOMAN", "INTERVIEWER", "APP", "WOMAN"]
    assert [ln.background for ln in lines] == [False, False, True, False]
    assert lines[1].text == "Com licença, como lê tanto?"
    assert lines[3].start == 6.3 and lines[3].end == 7.3


def test_speaker_genders_from_the_v3_transcript():
    tr = {"speakers": [{"label": "old woman", "tag": "WOMAN", "kind": "person", "gender": "female"},
                       {"label": "phone voice", "tag": "APP", "kind": "device", "gender": "unknown"}],
          "lines": [{"id": 1, "start": 0, "end": 2, "speaker": "old woman", "text": "a"},
                    {"id": 2, "start": 2, "end": 3, "speaker": "phone voice", "text": "b"}]}
    g = lqa.speaker_genders(tr)
    assert g["WOMAN"] == {"gender": "female", "kind": "person"}
    assert g["APP"]["kind"] == "device"


# ---------------------------------------------------------------------------
# Pieces and the final's timeline
# ---------------------------------------------------------------------------


def _snapshot():
    return [{"id": "s1", "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 4.0},
            {"id": "s2", "index": 1, "action": "keep", "start_sec": 4.0, "end_sec": 6.0},
            {"id": "s3", "index": 2, "action": "swap", "start_sec": 6.0, "end_sec": 10.0}]


def test_build_pieces_and_final_script():
    plan = [{"kind": "library", "clip_id": "c1"}, {"kind": "segment", "segment_def_id": "s2"},
            {"kind": "segment", "segment_def_id": "s3"}]
    pieces = lqa.build_pieces(
        stitch_plan=plan, snapshot=_snapshot(),
        run_segments={"s3": {"id": "rs3", "prompt_override": "[0.0–2.0] WOMAN: Olá mundo.",
                             "local_result_path": "/x/3.mp4", "seedance_task_id": "t3"}},
        run_prompt="run prompt", duration_sec=10.0, max_clip_sec=15.0,
        library_clips={"c1": {"path": "/x/c1.mp4", "prompt": "[0.0–4.0] WOMAN: Primeiro.",
                              "window": [0.0, 4.0], "seg_number": 1, "segment_def_id": "s1"}},
        genders={"WOMAN": {"gender": "female", "kind": "person"}})
    assert [p["kind"] for p in pieces] == ["library", "keep", "swap"]
    assert [p["seg_number"] for p in pieces] == [1, 2, 3]
    assert pieces[2]["window"] == [6.0, 10.0] and pieces[2]["run_segment_id"] == "rs3"
    assert pieces[2]["lines"][0]["gender"] == "female"
    # The swap clip came back 4.4 s long: its script is stretched by 1.1.
    secs = {0: 4.0, 1: 2.0, 2: 4.4}
    fs = lqa.final_script(pieces, secs)
    assert [(ln["pos"], ln["start"], ln["end"]) for ln in fs] == [(0, 0.0, 4.0), (2, 6.0, 8.2)]
    caps = lqa.script_captions(pieces, secs)
    assert [c["text"] for c in caps] == ["Primeiro.", "Olá mundo."]
    assert caps[1]["start"] == 6.0 and caps[1]["end"] == pytest.approx(8.15)


# ---------------------------------------------------------------------------
# ASR check
# ---------------------------------------------------------------------------

EXPECTED = [
    {"start": 0.0, "end": 2.4, "text": "Acho que leio bem mais livros.", "tag": "WOMAN", "pos": 0},
    {"start": 2.4, "end": 4.2, "text": "Com licença, como lê tanto?", "tag": "INTERVIEWER", "pos": 1},
    {"start": 4.3, "end": 6.2, "text": "Eu uso o Listenify todos os dias.", "tag": "WOMAN", "pos": 2},
]


def test_asr_check_all_present():
    asr = [{"start": 0.1, "end": 2.3, "text": "Acho que leio bem mais livros"},
           {"start": 2.5, "end": 4.1, "text": "com licenca como le tanto"},
           {"start": 4.4, "end": 6.0, "text": "Eu uso o Listenify todos os dias."}]
    out = lqa.asr_check(EXPECTED, asr, brands=[{"name": "Listenify"}], language="pt")
    assert out["coverage"] == 1.0 and not out["garbled"] and out["issues"] == []


def test_asr_check_missing_line_and_brand():
    asr = [{"start": 0.1, "end": 2.3, "text": "Acho que leio bem mais livros."},
           {"start": 4.4, "end": 6.0, "text": "Eu uso o Listenifyed todos os dias."}]
    out = lqa.asr_check(EXPECTED, asr, brands=[{"name": "Listenify", "pronunciation": "LISTEN-ih-fy"}],
                        language="pt")
    kinds = sorted((i["kind"], i["pos"]) for i in out["issues"])
    assert kinds == [("brand", 2), ("missing_line", 1)]
    brand = next(i for i in out["issues"] if i["kind"] == "brand")
    assert brand["pronunciation"] == "LISTEN-ih-fy"


def test_asr_check_garbled_reports_no_issue():
    asr = [{"start": 0.0, "end": 6.0, "text": "zrbl kwak tsss mmm"}]
    out = lqa.asr_check(EXPECTED, asr, language="pl")
    assert out["garbled"] is True and out["issues"] == [] and out["garbled_issues"]


def test_asr_missing_line_under_a_background_line_is_only_a_warning():
    exp = EXPECTED[:2] + [{"start": 2.0, "end": 4.5, "text": "Era uma vez um livro.", "tag": "APP",
                           "pos": 1, "background": True}]
    asr = [{"start": 0.1, "end": 2.3, "text": "Acho que leio bem mais livros."},
           {"start": 2.4, "end": 4.4, "text": "Era uma vez um livro."}]
    out = lqa.asr_check(exp, asr, language="pt")
    assert out["issues"] == []
    assert [w["kind"] for w in out["warnings"]] == ["missing_under_background"]


def test_fix_caption_words_keeps_timings_and_splits():
    asr = [{"start": 0.1, "end": 2.3, "text": "Acho que leio bem mais livros."},
           {"start": 4.4, "end": 6.0, "text": "Eu uso o Listenifyed todos os dias."}]
    fixed, changes = lqa.fix_caption_words(asr, EXPECTED, language="pt")
    assert changes == 1
    assert fixed[1] == {"start": 4.4, "end": 6.0, "text": "Eu uso o Listenify todos os dias."}
    assert fixed[0] == asr[0]
    # Two fixes in one line keep the right positions.
    asr2 = [{"start": 0.0, "end": 6.0, "text": "Acho que leyo bem mais libros."}]
    fixed2, n2 = lqa.fix_caption_words(asr2, EXPECTED[:1], language="pt")
    assert fixed2[0]["text"] == "Acho que leio bem mais livros." and n2 == 2


# ---------------------------------------------------------------------------
# Cues, comments, best take, tail cut
# ---------------------------------------------------------------------------


def test_qa_cues_and_apply_is_idempotent():
    cues = lqa.qa_cues([{"kind": "gender", "tag": "INTERVIEWER", "expected": "female"},
                        {"kind": "brand", "name": "Listenify", "pronunciation": "LISTEN-ih-fy"},
                        {"kind": "length", "delta_sec": 1.2},
                        {"kind": "gender", "tag": "INTERVIEWER", "expected": "female"}])
    assert cues == ["INTERVIEWER is a female voice in every line of this clip — never a male voice.",
                    'Pronounce "Listenify" exactly as LISTEN-ih-fy.']
    once = lqa.apply_cues("PROMPT", cues)
    assert once.startswith("PROMPT\n\n" + lqa.QA_CUE_HEADER + "\n- INTERVIEWER")
    assert lqa.apply_cues(once, cues) == once
    assert lqa.apply_cues(once, [lqa.CUE_CAPTIONS]).endswith("\n- " + lqa.CUE_CAPTIONS)
    assert lqa.qa_cues([{"kind": "length"}]) == []


def test_comment_cues_and_segments_named():
    piece = {"lines": [{"tag": "INTERVIEWER", "kind": "person", "gender": "female"},
                       {"tag": "APP", "kind": "device", "gender": "unknown"}]}
    cues = lqa.comment_cues("Segment 5: the interviewer sounds like a man, and the glasses stay on",
                            piece, [])
    assert cues == ["INTERVIEWER is a female voice in every line of this clip — never a male voice.",
                    lqa.CUE_ACTION]
    assert lqa.segments_named("Segment 5: wrong voice") == [5]
    assert lqa.segments_named("segs 2 and 4, also #6") == [2, 4, 6]
    assert lqa.segments_named("s3 and segments 5-7") == [3, 5, 6, 7]
    assert lqa.segments_named("the voice is wrong") == []


def test_best_takes():
    cur = {"rs3": {"clip": "/n.mp4", "seedance_task_id": "t2", "score": 2}}
    hist = [{"rs3": {"clip": "/o.mp4", "seedance_task_id": "t1", "score": 1}}]
    assert lqa.best_takes(hist, cur, ["rs3"]) == {"rs3": hist[0]["rs3"]}
    # equal score keeps the newest; unknown scores never win
    assert lqa.best_takes(hist, {"rs3": dict(cur["rs3"], score=1)}, ["rs3"]) == {}
    assert lqa.best_takes([{"rs3": dict(hist[0]["rs3"], score=None)}], cur, ["rs3"]) == {}
    assert lqa.best_takes(hist, {"rs3": dict(cur["rs3"], score=None)}, ["rs3"]) == {}


@pytest.mark.parametrize("duration,cut", [(12.288, 12.27), (10.0, 9.99), (10.005, 9.99),
                                          (78.9, 78.89)])
def test_tail_cut_sec(duration, cut):
    assert lqa.tail_cut_sec(duration) == pytest.approx(cut)
    assert lqa.tail_cut_sec(duration) < duration


def test_checkable_lines_ignore_background_overlaps():
    w = {"tag": "WOMAN", "kind": "person", "gender": "female", "background": False}
    a = dict(w, start=0.0, end=3.0)
    bg = {"tag": "APP", "kind": "device", "gender": "unknown", "background": True, "start": 1.0, "end": 5.0}
    other = dict(w, tag="INTERVIEWER", start=2.5, end=4.0)
    assert lqa._checkable(a, [a, bg]) is True
    assert lqa._checkable(a, [a, bg, other]) is False
    assert lqa._checkable(bg, [a, bg]) is False


def test_gender_verdicts_need_the_probe_baseline():
    medians = {"INTERVIEWER": {"hz": 112.0, "lines": 2, "expected": "female", "voiced_sec": 3.0}}
    issues, warnings = lqa.gender_verdicts(medians, {"INTERVIEWER": {"hz": 221.0}})
    assert [i["kind"] for i in issues] == ["gender"] and issues[0]["shift_octaves"] < -0.5
    issues, warnings = lqa.gender_verdicts(medians, {})
    assert issues == [] and [w["kind"] for w in warnings] == ["gender_suspect"]
    # a low female voice that the probe already had is no issue
    issues, warnings = lqa.gender_verdicts(
        {"WOMAN": {"hz": 150.0, "lines": 2, "expected": "female", "voiced_sec": 3.0}},
        {"WOMAN": {"hz": 160.0}})
    assert issues == []


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hz,gender", [(110.0, "male"), (130.0, "male"), (215.0, "female"),
                                       (240.0, "female")])
def test_median_pitch_of_a_tone(hz, gender):
    t = np.arange(int(1.5 * lqa.SAMPLE_RATE)) / lqa.SAMPLE_RATE
    x = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    f0, voiced = lqa.median_pitch(x)
    assert f0 == pytest.approx(hz, rel=0.03) and voiced > 1.0
    assert lqa.classify_pitch(f0) == gender
    assert lqa.median_pitch(np.zeros(16000, np.float32)) == (None, 0.0)


@needs_ffmpeg
def test_window_pitch_on_a_file(tmp_path):
    clip = make_clip(tmp_path / "a.mp4", 3, 110)
    m = lqa.window_pitch(clip, 0.5, 2.5)
    assert m["measured"] == "male" and m["hz"] == pytest.approx(110, rel=0.04)


@needs_ffmpeg
def test_segment_qa_finds_the_seeded_gender_error(tmp_path):
    """Probe (library) at 220 Hz is the accepted voice; segment 3 at 110 Hz is the error."""
    src = make_clip(tmp_path / "src.mp4", 10, 220)
    probe = make_clip(tmp_path / "probe.mp4", 4, 220)
    s2 = make_clip(tmp_path / "s2.mp4", 3, 220)
    s3 = make_clip(tmp_path / "s3.mp4", 3, 110)
    final = make_clip(tmp_path / "final.mp4", 10, 220)
    woman = {"tag": "WOMAN", "gender": "female", "kind": "person", "background": False}
    pieces = [
        {"pos": 0, "kind": "library", "clip": probe, "window": [0.0, 4.0], "seg_number": 1,
         "lines": [dict(woman, start=0.0, end=3.5, text="Primeiro.")]},
        {"pos": 1, "kind": "swap", "clip": s2, "window": [4.0, 7.0], "seg_number": 2,
         "run_segment_id": "rs2", "seedance_task_id": "t2",
         "lines": [dict(woman, start=0.2, end=2.8, text="Segundo.")]},
        {"pos": 2, "kind": "swap", "clip": s3, "window": [7.0, 10.0], "seg_number": 3,
         "run_segment_id": "rs3", "seedance_task_id": "t3",
         "lines": [dict(woman, start=0.2, end=2.8, text="Terceiro.")]},
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = lqa.segment_qa({"source": src, "final": final, "pieces": pieces}, str(out_dir))
    by = {p["pos"]: p for p in res["pieces"]}
    assert res["baseline"]["WOMAN"]["hz"] == pytest.approx(220, rel=0.04)
    assert by[1]["issues"] == []
    assert [i["kind"] for i in by[2]["issues"]] == ["gender"]
    assert by[2]["issues"][0]["measured"] == "male"
    assert by[1]["length_delta_sec"] == pytest.approx(0.0, abs=0.1)
    assert by[1]["caption_hits"] == 0 and os.path.isfile(by[1]["sheet"])
    assert res["stable"] is True and res["final_audio_sha256"]
    assert set(res["files"]) == {probe, s2, s3, final}
    assert by[0]["clip_sec"] == pytest.approx(4.0, abs=0.1)
    assert res["scoring_id"] == lqa.scoring_id(res["baseline"], None)


@needs_ffmpeg
def test_deliverable_check(tmp_path):
    clip = make_clip(tmp_path / "d.mp4", 2, 220)
    out = lqa.deliverable_check({"path": clip, "also": clip}, str(tmp_path))
    assert out["ok"] is True and out["size_bytes"] == os.path.getsize(clip)
    assert out["sha256"] == out["also_sha256"] and len(out["sha256"]) == 64
    assert lqa.deliverable_check({"path": clip, "max_bytes": 10}, str(tmp_path))["ok"] is False
