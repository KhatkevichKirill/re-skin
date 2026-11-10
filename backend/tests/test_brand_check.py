"""AICP-389: the trailing-sibilant brand detector (app.brand_check).

Fixtures (tests/fixtures/brand_check/*.wav + .json, NOT published: they are
excerpts of real ad audio; the tests that need them skip) are 16 kHz mono excerpts of
REAL finals with the word timestamps the asr-small model gave on the whole
final (30 s chunks, 5 s overlap) and the full final's reference level:

* fr_speechifies_original — run dea38297-fr, the take said "Speechifies"
  (the extra "-s" measured by hand at 35.34-35.54 s on 2026-11-04);
* fr_patched — the same final after the operator-approved low-pass fix;
* es_correct / pt_correct — the correct es and pt finals of project 2476176f
  (real scripts from their segment-QA records);
* pl_sibilant_next — a pl brand followed by a "cz-" word;
* es_repeated — an es final with several occurrences;
* es_exact, fr_long — more correct finals (the ASR spelled the brand right; a
  78 s final with the brand near its end).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

from app import brand_check as bc

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "brand_check")
FFMPEG = shutil.which("ffmpeg")
# The real-audio fixtures are not part of the public repository; the tests
# that need them skip when the directory is absent.
needs_fixtures = pytest.mark.skipif(not os.path.isdir(FIX), reason="brand_check audio fixtures not present")


def load(name):
    with wave.open(os.path.join(FIX, name + ".wav")) as f:
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    meta = json.load(open(os.path.join(FIX, name + ".json"), encoding="utf-8"))
    return x, meta


def run(name, **kw):
    x, m = load(name)
    return bc.check(x, m["words"], m["expected"], m["brands"], m["language"],
                    reference_level_db=m["reference_level_db"], **kw), m


def by_brand(res, brand="Listenify"):
    return [r for r in res["occurrences"] if r["brand"] == brand]


@needs_fixtures
def test_the_dea38297_take_is_found_within_40ms():
    res, m = run("fr_speechifies_original")
    (occ,) = by_brand(res)
    assert occ["verdict"] == "found", occ
    w0, w1 = (v + m["offset_sec"] for v in occ["window"])
    assert abs(w0 - 35.34) <= 0.04 and abs(w1 - 35.54) <= 0.04, (w0, w1)
    assert res["found"] == [occ["occ_id"]]
    assert occ["asr"]["method"] == "anchors"  # whisper wrote "Listen Fice"


@needs_fixtures
def test_the_patched_final_offers_nothing():
    res, _ = run("fr_patched")
    assert res["found"] == []
    assert by_brand(res)[0]["verdict"] == "clean"


@needs_fixtures
@pytest.mark.parametrize("name", ["es_correct", "pt_correct", "pl_sibilant_next", "es_repeated",
                                  "es_exact", "fr_long"])
def test_correct_finals_offer_nothing(name):
    res, _ = run(name)
    assert res["found"] == [], res["occurrences"]
    assert all(r["verdict"] in ("clean", "unclear", "not_checked") for r in res["occurrences"])


@needs_fixtures
def test_multi_word_brand_is_not_checked_never_passed():
    res, _ = run("fr_speechifies_original")
    (store,) = by_brand(res, "App Store")
    assert store["verdict"] == "not_checked" and "several words" in store["reason"]


@needs_fixtures
def test_repeated_occurrences_each_get_their_own_id():
    res, _ = run("es_repeated")
    ids = [r["occ_id"] for r in res["occurrences"]]
    assert len(ids) == len(set(ids)) >= 3


@needs_fixtures
def test_thresholds_matter_the_positive_needs_its_pause():
    res, _ = run("fr_speechifies_original", th=bc.Thresholds(pause_min_sec=0.5))
    assert res["found"] == []


# ---------------------------------------------------------------------------
# Text: occurrences, inflections, alignment, sibilant onsets
# ---------------------------------------------------------------------------


def test_occurrences_inflection_and_languages_without_spaces():
    lines = [{"start": 0, "end": 2, "text": "Listenify's voices, Listenify.", "tag": "A", "pos": 0}]
    occ = bc.brand_occurrences(lines, [{"name": "Listenify"}], "en")
    assert [(o.status, o.word) for o in occ] == [("not_checked", 0), ("checkable", 2)]
    assert "inflects" in occ[0].reason
    ja = bc.brand_occurrences([{"start": 0, "end": 2, "text": "Listenifyです", "tag": "A"}],
                              [{"name": "Listenify"}], "ja")
    assert [o.status for o in ja] == ["not_checked"]


def test_next_word_crosses_the_line_boundary():
    lines = [{"start": 0, "end": 1, "text": "C'est Listenify.", "tag": "A", "pos": 0},
             {"start": 1, "end": 2, "text": "Sur l'App Store.", "tag": "A", "pos": 0}]
    (o,) = bc.brand_occurrences(lines, [{"name": "Listenify"}], "fr")
    assert o.next_word == "Sur"


def test_alignment_by_neighbours_not_by_spelling():
    lines = [{"start": 0.0, "end": 1.0, "text": "C'est Listenify.", "tag": "A", "pos": 0},
             {"start": 1.0, "end": 2.0, "text": "Sur l'App Store.", "tag": "A", "pos": 0}]
    words = [{"text": "C", "start": 0.1, "end": 0.2}, {"text": "'est", "start": 0.2, "end": 0.3},
             {"text": " pitch", "start": 0.3, "end": 0.6}, {"text": " file", "start": 0.6, "end": 0.9},
             {"text": " sur", "start": 1.1, "end": 1.3}, {"text": " l'App", "start": 1.3, "end": 1.6},
             {"text": " Store.", "start": 1.6, "end": 1.9}]
    occ = bc.brand_occurrences(lines, [{"name": "Listenify"}], "fr")
    span = bc.align_occurrences(occ, words, lines)["o1"]
    assert (span["start"], span["end"], span["method"]) == (0.3, 0.9, "anchors")
    assert span["next_text"] == "sur"


def test_unlocatable_brand_is_unclear_not_clean():
    lines = [{"start": 0.0, "end": 1.0, "text": "Listenify", "tag": "A", "pos": 0}]
    res = bc.check(np.zeros(16000, dtype=np.float32), [{"text": "hello", "start": 0.1, "end": 0.4}],
                   lines, [{"name": "Listenify"}], "en")
    assert res["occurrences"][0]["verdict"] == "unclear"
    assert res["found"] == []


@pytest.mark.parametrize("word,want", [("sur", True), ("Zamura", True), ("czohes", True),
                                       ("cela", True), ("chez", True), ("casa", False),
                                       ("ahora", False), ("Tá", False), (None, False)])
def test_sibilant_onsets(word, want):
    assert bc.starts_with_sibilant(word) is want


def test_overlapping_speech_is_unclear():
    lines = [{"start": 0.0, "end": 1.5, "text": "C'est Listenify", "tag": "A", "pos": 0},
             {"start": 0.5, "end": 1.6, "text": "Quoi donc", "tag": "B", "pos": 0}]
    words = [{"text": "C'est", "start": 0.1, "end": 0.3}, {"text": "Listenify", "start": 0.3, "end": 0.9}]
    res = bc.check(np.zeros(32000, dtype=np.float32), words, lines, [{"name": "Listenify"}], "fr")
    assert res["occurrences"][0]["verdict"] == "unclear"
    assert "overlaps" in res["occurrences"][0]["reason"]


# ---------------------------------------------------------------------------
# Signal: synthetic endings
# ---------------------------------------------------------------------------

SR = bc.SAMPLE_RATE


def _synth(parts):
    """parts: [(kind, seconds)] — kind in vowel / fric / silence."""
    rng = np.random.default_rng(7)
    out = []
    for kind, sec in parts:
        n = int(sec * SR)
        t = np.arange(n) / SR
        if kind == "vowel":
            out.append(0.3 * np.sin(2 * np.pi * 180 * t) + 0.15 * np.sin(2 * np.pi * 540 * t))
        elif kind == "fric":
            noise = rng.standard_normal(n)
            spec = np.fft.rfft(noise)
            f = np.fft.rfftfreq(n, 1 / SR)
            spec[f < 4000] = 0
            out.append(0.01 * np.fft.irfft(spec, n) / (np.std(np.fft.irfft(spec, n)) + 1e-9))
        else:
            out.append(0.0003 * rng.standard_normal(n))
    return np.concatenate(out).astype(np.float32)


def _verdict(x, word=(0.2, 0.7), next_start=None):
    db, hf = bc.frame_features(x)
    return bc.trailing_sibilant(db, hf, level=bc.reference_level(db), word_start=word[0],
                                word_end=word[1], next_start=next_start)


def test_synthetic_trailing_s_then_pause_is_found():
    x = _synth([("silence", 0.2), ("vowel", 0.5), ("fric", 0.15), ("silence", 0.4), ("vowel", 0.3)])
    v = _verdict(x)
    assert v["verdict"] == "found", v
    assert abs(v["window"][0] - 0.70) <= 0.03 and abs(v["window"][1] - 0.86) <= 0.04


def test_synthetic_sibilant_word_directly_after_is_not_found():
    # brand vowel -> "s" of the next word -> its vowel, no pause: never offered
    x = _synth([("silence", 0.2), ("vowel", 0.5), ("fric", 0.12), ("vowel", 0.4)])
    assert _verdict(x)["verdict"] != "found"


def test_synthetic_voiced_ending_into_pause_is_clean():
    x = _synth([("silence", 0.2), ("vowel", 0.5), ("silence", 0.5)])
    assert _verdict(x)["verdict"] == "clean"


def test_synthetic_short_click_is_clean():
    x = _synth([("silence", 0.2), ("vowel", 0.5), ("fric", 0.04), ("silence", 0.4)])
    assert _verdict(x)["verdict"] == "clean"


# ---------------------------------------------------------------------------
# Chunks, merge, render
# ---------------------------------------------------------------------------


def test_chunks_are_bounded_and_overlap():
    cs = bc.chunks(78.0)
    assert cs[0] == (0.0, 30.0) and cs[-1][1] == 78.0
    assert all(b - a <= 30.0 for a, b in cs)
    assert all(cs[i][1] - cs[i + 1][0] == pytest.approx(5.0) for i in range(len(cs) - 1))
    assert bc.chunks(12.0) == [(0.0, 12.0)]


def test_a_word_cut_by_a_chunk_edge_is_taken_from_the_other_chunk():
    a = (0.0, 30.0, [{"text": "C'est", "start": 28.9, "end": 29.3},
                     {"text": "Speech", "start": 29.4, "end": 30.0}])     # cut at the edge
    b = (25.0, 55.0, [{"text": "C'est", "start": 28.9, "end": 29.3},
                      {"text": "Listenify", "start": 29.4, "end": 30.2},
                      {"text": "sur", "start": 30.5, "end": 30.8}])
    merged = bc.merge_words([a, b])
    assert [w["text"] for w in merged] == ["C'est", "Listenify", "sur"]


@needs_fixtures
@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_render_changes_only_the_window_and_copies_the_picture(tmp_path):
    src = tmp_path / "base.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=0x405060:s=96x160:r=25:d=3",
                    "-f", "lavfi", "-i", "anoisesrc=d=3:c=white:a=0.2:r=48000",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", str(src)], check=True)
    out = tmp_path / "cand.mp4"
    p = bc.render_params([1.2, 1.5])
    bc.render_candidate(str(src), str(out), p)
    a, b = bc.probe_streams(str(src)), bc.probe_streams(str(out))
    assert a["video_packets"] == b["video_packets"] > 0
    assert abs(a["duration_sec"] - b["duration_sec"]) < 0.05
    xa, xb = bc.decode_mono(str(src)), bc.decode_mono(str(out))
    assert bc.outside_level_delta_db(xa, xb, [1.2, 1.5]) <= 0.5
    # inside the window the high band is gone
    _, hfa = bc.frame_features(xa[int(1.25 * SR):int(1.45 * SR)])
    _, hfb = bc.frame_features(xb[int(1.25 * SR):int(1.45 * SR)])
    assert float(np.mean(hfb)) < 0.2 * float(np.mean(hfa))
