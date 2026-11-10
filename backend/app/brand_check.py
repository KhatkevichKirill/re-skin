"""Trailing-sibilant check of scripted brand words (AICP-389).

Seedance 2.5 sometimes ends a brand name with an extra sibilant in localised
speech ("Speechifies" for "Listenify"). The Gemini caption ASR normalises the
word back, so the existing ASR QA never sees it. This module finds that one
shape, for free, on the stitched final:

    brand's voiced ending -> a separate fricative (>= ~80 ms, >3.5 kHz energy)
    -> a measured pause

and, when it is there, renders the free fix the operator chose by ear on run
dea38297-fr (2026-11-04): inside the fricative's window the audio is replaced
by a 2x low-pass 2.2 kHz copy, with 25 ms complementary sample-accurate ramps,
``amix normalize=0`` and the video stream-copied.

Pure parts (no I/O): :func:`frame_features`, :func:`brand_occurrences`,
:func:`align_occurrences`, :func:`trailing_sibilant`, :func:`check`. The I/O
parts (decode, the local word ASR, the renders) are small functions below
them. The detector only PROPOSES: operator listening stays the judge.

A word ASR is needed only to find WHERE each brand occurrence is; its spelling
of the brand is never trusted (whisper wrote "Listen-Fice" for the bad take and
"Speechfied" for a good English one). Nothing here contacts a provider.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Optional, Sequence

DETECTOR_VERSION = 1
RENDER_VERSION = 1

SAMPLE_RATE = 16000
FRAME = 320          # 20 ms analysis frame
HOP = 160            # 10 ms hop
HOP_SEC = HOP / SAMPLE_RATE

NO_SPACE_LANGUAGES = {"ja", "zh"}


@dataclass(frozen=True)
class Thresholds:
    """Calibrated on labelled finals (docs/brand-trailing-sibilant.md §4).

    Levels are dB relative to the 95th percentile frame level of the whole
    final (the voice's loud vowels), so they hold across loudness."""

    hf_hz: float = 3500.0          # "high band" lower edge
    voiced_db: float = -18.0       # a voiced frame is at least this loud ...
    voiced_hf_max: float = 0.30    # ... and has at most this share of high-band energy
    fric_hf_min: float = 0.35      # a fricative frame: at least this share above hf_hz (3-frame smoothed)
    fric_db_min: float = -42.0     # ... and at least this loud (not room noise)
    fric_min_sec: float = 0.08     # the separate ending is at least this long
    fric_max_sec: float = 0.40     # longer than this is not one trailing "-s"
    fric_gap_sec: float = 0.15     # its onset is within this of the voiced ending (no voicing between)
    fric_fill: float = 0.6         # share of fricative frames inside its run
    fric_holes: int = 2            # non-fricative frames bridged inside the run
    end_pad_sec: float = 0.01      # the patched window ends this far past the fricative
    fric_hf_mean_min: float = 0.50  # ... and its mean high-band share is at least this
    pause_min_sec: float = 0.10    # a measured pause after it
    sibilant_next_pause_sec: float = 0.15  # next word starts with a sibilant: a longer pause needed
    pause_db: float = -18.0        # a pause frame is below this, and not a fricative
    search_after_sec: float = 0.30  # look for the voiced ending this far past the ASR word end
    pad_sec: float = 0.0           # widen the patched window by this on each side


DEFAULT = Thresholds()


# ---------------------------------------------------------------------------
# Text: brands, occurrences, alignment
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)


def norm(word: str) -> str:
    return _PUNCT_RE.sub("", unicodedata.normalize("NFKC", str(word)).casefold())


def _script_words(text: str) -> list[str]:
    return [w for w in str(text).split() if norm(w)]


@dataclass(frozen=True)
class Occurrence:
    occ_id: str                 # "o<k>": k-th scripted brand occurrence, in script order
    brand: str
    line: int                   # index into the expected lines
    word: int                   # word index inside the line
    pos: Optional[int]          # the piece the line is in
    line_start: float
    line_end: float
    status: str                 # "checkable" | "not_checked"
    reason: str = ""
    next_word: Optional[str] = None


def brand_occurrences(expected: Sequence[Mapping], brands: Iterable[Mapping],
                      language: Optional[str]) -> list[Occurrence]:
    """Every scripted brand occurrence, in order, each with its own id.

    Multi-word brands and languages written without spaces are reported as
    ``not_checked`` (never as passed). A scripted inflection ("Listenify's")
    is the script's own ending: not checked either, with that reason."""
    names = []
    for b in brands or []:
        if isinstance(b, Mapping) and str(b.get("name") or "").strip():
            names.append(str(b["name"]).strip())
    out: list[Occurrence] = []
    k = 0
    lines = [dict(e) for e in expected if not e.get("background")]
    no_space = (language or "") in NO_SPACE_LANGUAGES
    for li, ln in enumerate(lines):
        text = str(ln.get("text") or "")
        words = _script_words(text)
        for name in names:
            toks = name.split()
            single = norm(name)
            if no_space or len(toks) != 1:
                hits = text.casefold().count(name.casefold())
                for _ in range(hits):
                    k += 1
                    out.append(Occurrence(
                        f"o{k}", name, li, -1, ln.get("pos"), float(ln.get("start") or 0),
                        float(ln.get("end") or 0), "not_checked",
                        "a brand of several words or in a language without spaces is not checked"))
                continue
            for wi, w in enumerate(words):
                nw = norm(w)
                if nw == single:
                    k += 1
                    nxt = words[wi + 1] if wi + 1 < len(words) else None
                    if nxt is None and li + 1 < len(lines):
                        follow = _script_words(str(lines[li + 1].get("text") or ""))
                        nxt = follow[0] if follow else None
                    out.append(Occurrence(f"o{k}", name, li, wi, ln.get("pos"),
                                          float(ln.get("start") or 0), float(ln.get("end") or 0),
                                          "checkable", "", nxt))
                elif nw.startswith(single) and len(nw) - len(single) <= 3:
                    k += 1
                    out.append(Occurrence(
                        f"o{k}", name, li, wi, ln.get("pos"), float(ln.get("start") or 0),
                        float(ln.get("end") or 0), "not_checked",
                        f"the script itself inflects the brand ({w!r})"))
    out.sort(key=lambda o: (o.line_start, o.line, o.word))
    return [Occurrence(**{**asdict(o), "occ_id": f"o{i + 1}"}) for i, o in enumerate(out)]


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


ALIGN_MIN_SIMILARITY = 0.6
ALIGN_SLACK_SEC = 1.5
ALIGN_MAX_SPAN_TOKENS = 5
ALIGN_MAX_SPAN_SEC = 2.0
OVERLAP_MIN_SEC = 0.2

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    """Normalised word tokens; apostrophes split ("C'est" -> c, est)."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", str(text)).casefold())


def align_occurrences(occs: Sequence[Occurrence], words: Sequence[Mapping],
                      lines: Sequence[Mapping] = ()) -> dict[str, dict]:
    """``occ_id -> {start, end, text, method, next_start, next_text}`` or absent.

    The ASR spelling of a brand is unreliable ("pitchfile", "Speed Fight",
    "Listen Fice" for Listenify), so a brand is located by its NEIGHBOURS: the
    whole foreground script (every line, in order) is aligned to the whole ASR
    word sequence (difflib over normalised tokens), and the brand is the ASR
    words strictly between the nearest aligned script token before it and the
    one after it — across line boundaries, so a brand that ends its line is
    still bracketed by the next line's first word. Only when one side has no
    anchor does the spelling decide (best 1–3 word span, similarity >= 0.6).
    Every span must lie inside its line's window (±1.5 s)."""
    ws = sorted((dict(w) for w in words if tokens(w.get("text", ""))),
                key=lambda w: (float(w["start"]), float(w["end"])))
    fg = [dict(e) for e in lines if not e.get("background")]
    heard: list[tuple[str, int]] = [(t, i) for i, w in enumerate(ws) for t in tokens(w["text"])]
    script: list[str] = []
    where: dict[tuple[int, int], int] = {}  # (line, word) -> script token index of the brand
    for li, ln in enumerate(fg):
        for wi, w in enumerate(_script_words(str(ln.get("text") or ""))):
            toks = tokens(w)
            if not toks:
                continue
            where[(li, wi)] = len(script)
            script.extend(toks)
    sm = difflib.SequenceMatcher(None, script, [t for t, _ in heard], autojunk=False)
    pairs = [(blk.a + t, blk.b + t) for blk in sm.get_matching_blocks() for t in range(blk.size)]
    out: dict[str, dict] = {}
    used_upto = -1
    for o in occs:
        if o.status != "checkable":
            continue
        lo_t, hi_t = o.line_start - ALIGN_SLACK_SEC, o.line_end + ALIGN_SLACK_SEC
        target = norm(o.brand)
        k = where.get((o.line, o.word))
        span, method = None, ""
        cand = [i for i, w in enumerate(ws) if i > used_upto
                and float(w["end"]) >= lo_t and float(w["start"]) <= hi_t]
        if k is not None:
            exact = [h for si, h in pairs if si == k]
            before = [h for si, h in pairs if si < k]
            after = [h for si, h in pairs if si > k]
            if exact:
                i = heard[exact[0]][1]
                span, method = (i, i), "exact"
            else:
                lo = before[-1] + 1 if before else None
                hi = after[0] - 1 if after else None
                if lo is not None and hi is not None and 0 <= hi - lo < ALIGN_MAX_SPAN_TOKENS \
                        and lo < len(heard) and hi >= 0:
                    i, j = heard[lo][1], heard[hi][1]
                    if float(ws[j]["end"]) - float(ws[i]["start"]) <= ALIGN_MAX_SPAN_SEC:
                        span, method = (i, j), "anchors"
                elif lo is not None and hi is None:
                    cand = [i for i in cand if i >= heard[lo][1]] if lo < len(heard) else []
                elif hi is not None and lo is None:
                    cand = [i for i in cand if i <= heard[hi][1]] if hi >= 0 else []
        if span is not None:
            i, j = span
            if i <= used_upto or float(ws[j]["end"]) < lo_t or float(ws[i]["start"]) > hi_t:
                span = None
        if span is None:
            best = None
            for c, i in enumerate(cand):
                joined = ""
                for j in cand[c:c + 3]:
                    joined += norm(ws[j]["text"])
                    score = _similarity(joined, target)
                    if best is None or score > best[0] + 1e-9:
                        best = (score, i, j)
            if best is None or best[0] < ALIGN_MIN_SIMILARITY:
                continue
            span, method = (best[1], best[2]), f"spelling {best[0]:.2f}"
        i, j = span
        nxt = ws[j + 1] if j + 1 < len(ws) else None
        out[o.occ_id] = {
            "start": float(ws[i]["start"]), "end": float(ws[j]["end"]),
            "text": "".join(str(ws[t]["text"]) for t in range(i, j + 1)).strip(),
            "method": method,
            "next_start": float(nxt["start"]) if nxt else None,
            "next_text": str(nxt["text"]).strip() if nxt else None,
        }
        used_upto = j
    return out


_SIBILANT_START = re.compile(r"^(s|z|ç|ś|ź|ż|š|ž|x|c[eiyéèêíîëï]|ch|sh|sz|cz|tch|ts|tz)", re.IGNORECASE)


def starts_with_sibilant(word: Optional[str]) -> bool:
    w = unicodedata.normalize("NFKC", str(word or "")).casefold().lstrip("\"'«“(¿¡-")
    return bool(_SIBILANT_START.match(w))


# ---------------------------------------------------------------------------
# Signal: frame features and the trailing-sibilant rule
# ---------------------------------------------------------------------------


def frame_features(samples, *, hf_hz: float = DEFAULT.hf_hz):
    """Per 10 ms hop: ``(level_db, hf_share)`` of a 20 ms Hann frame (numpy arrays)."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    if len(x) < FRAME:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    nf = 1 + (len(x) - FRAME) // HOP
    win = np.hanning(FRAME).astype(np.float32)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    lo, hi = freqs >= 100.0, freqs >= hf_hz
    db = np.empty(nf, dtype=np.float32)
    hf = np.empty(nf, dtype=np.float32)
    step = 4096  # bounded memory on long finals
    for a in range(0, nf, step):
        b = min(nf, a + step)
        idx = np.arange(FRAME)[None, :] + HOP * np.arange(a, b)[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        tot = spec[:, lo].sum(axis=1) + 1e-12
        db[a:b] = 10.0 * np.log10(tot / FRAME)
        hf[a:b] = spec[:, hi].sum(axis=1) / tot
    return db, hf


def reference_level(db) -> float:
    import numpy as np

    return float(np.percentile(db, 95)) if len(db) else 0.0


def _runs(mask, a: int, b: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i in range(a, b):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, b))
    return out


def trailing_sibilant(db, hf, *, level: float, word_start: float, word_end: float,
                      next_start: Optional[float], th: Thresholds = DEFAULT) -> dict:
    """The verdict for ONE aligned occurrence.

    ``{"verdict": "found"|"clean"|"unclear", "reason", "window"?, "measurements"}``.
    *found* needs the voiced ending, then a fricative run of ``fric_min_sec``
    to ``fric_max_sec`` starting within ``fric_gap_sec`` of it, then a pause of
    ``pause_min_sec``. Anything that does not fit is *clean* when the ending is
    plainly voiced into a pause or the next word, else *unclear*."""
    import numpy as np

    n = len(db)
    t2i = lambda t: max(0, min(n, int(round(t / HOP_SEC))))  # noqa: E731
    rel = np.asarray(db, dtype=np.float32) - level
    # 3-frame energy-weighted smoothing of the high-band share (one 20 ms
    # frame of a quiet fricative is noisy).
    tot = np.power(10.0, rel / 10.0)
    hfe = tot * np.asarray(hf, dtype=np.float32)
    k = np.ones(3, dtype=np.float32)
    hfs = np.convolve(hfe, k, "same") / (np.convolve(tot, k, "same") + 1e-12)
    voiced = (rel >= th.voiced_db) & (hfs <= th.voiced_hf_max)
    fric = (hfs >= th.fric_hf_min) & (rel >= th.fric_db_min)
    pause = (rel < th.pause_db) & ~fric

    a, b = t2i(word_start), t2i(word_end + th.search_after_sec)
    if b - a < 3:
        return {"verdict": "unclear", "reason": "the word is too short to measure", "measurements": {}}
    vruns = [r for r in _runs(voiced, a, b) if r[1] - r[0] >= 3]
    if not vruns:
        return {"verdict": "unclear", "reason": "no voiced ending found in the brand word",
                "measurements": {}}
    # The ending: the last voiced run that starts inside the ASR word.
    inside = [r for r in vruns if r[0] <= t2i(word_end)]
    vstart, vend = (inside or vruns[:1])[-1]
    meas = {"voiced_end_sec": round(vend * HOP_SEC, 3)}

    # A fricative run starting within fric_gap of the voiced ending; small
    # holes (one non-fricative frame) are bridged.
    gap = int(round(th.fric_gap_sec / HOP_SEC))
    limit = min(n, vend + int(round((th.fric_max_sec + 0.2) / HOP_SEC)))
    fs = None
    for i in range(vend, min(n, vend + gap + 1)):
        if voiced[i]:
            break  # voicing again before any fricative: the next word
        if fric[i]:
            fs = i
            break
    if fs is None:
        reason = "the ending is voiced with no separate fricative"
        return {"verdict": "clean", "reason": reason, "measurements": meas}
    fe, holes, i = fs + 1, 0, fs + 1
    while i < limit:
        if fric[i]:
            fe, holes = i + 1, 0
        elif voiced[i]:
            break
        else:
            holes += 1
            if holes > th.fric_holes:
                break
        i += 1
    flen = (fe - fs) * HOP_SEC
    fill = float(fric[fs:fe].mean()) if fe > fs else 0.0
    meas.update(fric_start_sec=round(fs * HOP_SEC, 3), fric_end_sec=round(fe * HOP_SEC, 3),
                fric_sec=round(flen, 3), fric_fill=round(fill, 2),
                fric_hf_share=round(float(hfs[fs:fe].mean()) if fe > fs else 0.0, 2),
                fric_level_db=round(float(rel[fs:fe].max()) if fe > fs else 0.0, 1))
    if flen < th.fric_min_sec:
        return {"verdict": "clean", "reason": f"the fricative after the ending is only {flen * 1000:.0f} ms",
                "measurements": meas}
    if flen > th.fric_max_sec or fill < th.fric_fill or meas["fric_hf_share"] < th.fric_hf_mean_min:
        return {"verdict": "unclear", "reason": "the noise after the ending does not look like one trailing sibilant",
                "measurements": meas}
    # A pause right after it (measured, not the ASR's next-word time).
    pmin = int(round(th.pause_min_sec / HOP_SEC))
    pe = fe
    while pe < n and pause[pe]:
        pe += 1
    plen = (pe - fe) * HOP_SEC
    meas["pause_sec"] = round(plen, 3)
    if pe - fe < pmin:
        return {"verdict": "unclear", "reason": "no measured pause after the fricative: it may be the next word",
                "measurements": meas}
    if next_start is not None and next_start < fs * HOP_SEC - 0.05:
        return {"verdict": "unclear", "reason": "the next word starts inside the fricative (overlap)",
                "measurements": meas}
    # The window starts at the voiced ending (the weak fricative onset
    # included) and ends just past the fricative; the render's ramps lie
    # outside it.
    w0 = max(0.0, vend * HOP_SEC - th.pad_sec)
    w1 = min(n * HOP_SEC, fe * HOP_SEC + th.end_pad_sec + th.pad_sec)
    return {"verdict": "found", "reason": "a separate trailing sibilant followed by a pause",
            "window": [round(w0, 3), round(w1, 3)], "measurements": meas}


def check(samples, words: Sequence[Mapping], expected: Sequence[Mapping],
          brands: Iterable[Mapping], language: Optional[str], *,
          th: Thresholds = DEFAULT, reference_level_db: Optional[float] = None) -> dict:
    """Every scripted brand occurrence of the final with its verdict.

    ``{"detector_version", "occurrences": [...], "found": [occ_id...]}``; an
    occurrence's ``verdict`` is ``found`` (a fix may be offered), ``clean``,
    ``unclear`` / ``not_checked`` (no fix, with the reason). Nothing here is
    ever "passed" without a measurement."""
    occs = brand_occurrences(expected, brands, language)
    spans = align_occurrences(occs, words, expected)
    db, hf = frame_features(samples, hf_hz=th.hf_hz)
    # The level reference is the WHOLE final's; a caller analysing an excerpt
    # (the test fixtures) passes the full final's value.
    level = reference_level(db) if reference_level_db is None else float(reference_level_db)
    fg = [dict(e) for e in expected if not e.get("background")]
    rows = []
    for o in occs:
        row = {"occ_id": o.occ_id, "brand": o.brand, "line": o.line, "word": o.word, "pos": o.pos,
               "script_window": [round(o.line_start, 3), round(o.line_end, 3)],
               "next_word": o.next_word}
        if o.status != "checkable":
            rows.append(dict(row, verdict="not_checked", reason=o.reason))
            continue
        span = spans.get(o.occ_id)
        if span is None:
            rows.append(dict(row, verdict="unclear", reason="the brand word could not be located in the audio"))
            continue
        row["asr"] = span
        # Overlapping speech = another speaker's SCRIPT line overlapping the
        # brand's line (script times; adjacent lines are not an overlap).
        me = fg[o.line]
        others = [e for i, e in enumerate(fg) if i != o.line and e.get("tag") != me.get("tag")
                  and min(float(e.get("end", 0)), float(me.get("end", 0)))
                  - max(float(e.get("start", 0)), float(me.get("start", 0))) > OVERLAP_MIN_SEC]
        if others:
            rows.append(dict(row, verdict="unclear", reason="another speaker's line overlaps the brand"))
            continue
        res = trailing_sibilant(db, hf, level=level, word_start=span["start"], word_end=span["end"],
                                next_start=span.get("next_start"), th=th)
        if res["verdict"] == "found" and starts_with_sibilant(o.next_word) \
                and res["measurements"].get("pause_sec", 0) < th.sibilant_next_pause_sec:
            res = dict(res, verdict="unclear",
                       reason="the next word starts with a sibilant and the pause is short")
        rows.append(dict(row, **res))
    return {"detector_version": DETECTOR_VERSION, "thresholds": asdict(th),
            "reference_level_db": round(level, 2), "occurrences": rows,
            "found": [r["occ_id"] for r in rows if r.get("verdict") == "found"]}


# ---------------------------------------------------------------------------
# I/O: decode, word ASR chunks, renders
# ---------------------------------------------------------------------------

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

ASR_CHUNK_SEC = 30.0
ASR_OVERLAP_SEC = 5.0
MAX_FINAL_SEC = 600.0


def decode_mono(path: str, *, timeout_sec: int = 300):
    import numpy as np

    raw = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1",
                          "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
                         capture_output=True, check=True, timeout=timeout_sec).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def chunks(duration: float, *, size: float = ASR_CHUNK_SEC, overlap: float = ASR_OVERLAP_SEC):
    """Bounded, overlapping ``(start, end)`` chunks covering ``[0, duration]``."""
    import math

    out, t = [], 0.0
    while True:
        end = min(duration, t + size)
        # floored, never rounded up past the stream (the ASR refuses that)
        out.append((round(t, 3), math.floor(end * 1000.0) / 1000.0))
        if end >= duration - 1e-6:
            return out
        t = end - overlap


def merge_words(per_chunk: Sequence[tuple[float, float, Sequence[Mapping]]]) -> list[dict]:
    """Words of overlapping chunks on one timeline: inside an overlap the word
    whose centre is nearer the middle of its own chunk wins (a word cut by a
    chunk edge is the one dropped)."""
    cands = []
    for c0, c1, words in per_chunk:
        mid = (c0 + c1) / 2.0
        for w in words:
            s, e = float(w["start"]), float(w["end"])
            centre = (s + e) / 2.0
            edge = min(centre - c0, c1 - centre)
            cands.append((s, e, str(w["text"]), float(w.get("probability", 1.0)), edge, abs(centre - mid)))
    cands.sort(key=lambda c: (-c[4], c[5]))
    kept: list[tuple] = []
    for c in cands:
        if any(min(c[1], k[1]) - max(c[0], k[0]) > 0.5 * min(c[1] - c[0], k[1] - k[0]) + 1e-6
               for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda c: c[0])
    return [{"text": c[2], "start": round(c[0], 3), "end": round(c[1], 3),
             "probability": round(c[3], 3)} for c in kept]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


LOWPASS_HZ = 2200
RAMP_SEC = 0.025
AAC_BITRATE = "192k"


def render_params(window: Sequence[float]) -> dict:
    return {"kind": "same_run_lowpass", "render_version": RENDER_VERSION,
            "start_sec": float(window[0]), "end_sec": float(window[1]),
            "lowpass_hz": LOWPASS_HZ, "lowpass_passes": 2, "ramp_sec": RAMP_SEC,
            "audio_codec": "aac", "audio_bitrate": AAC_BITRATE}


def _gate(p: Mapping) -> str:
    a, b, r = float(p["start_sec"]), float(p["end_sec"]), float(p["ramp_sec"])
    # 1 inside [a, b], ramping over r on each side; sample-accurate (aeval t).
    return f"(clip((t-{a - r:.6f})/{r:.6f},0,1)*clip(({b + r:.6f}-t)/{r:.6f},0,1))"


def render_candidate(base: str, out: str, params: Mapping, *, timeout_sec: int = 600) -> None:
    """The full candidate: low-passed copy inside the window, picture copied."""
    g = _gate(params)
    lp = ",".join([f"lowpass=f={int(params['lowpass_hz'])}"] * int(params["lowpass_passes"]))
    graph = (f"[0:a]asplit=2[a][b];[b]{lp}[bl];"
             f"[a]aeval='val(ch)*(1-{g})':c=same[a1];[bl]aeval='val(ch)*{g}':c=same[b1];"
             f"[a1][b1]amix=inputs=2:normalize=0:duration=first[out]")
    subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-y", "-i", base, "-filter_complex", graph,
                    "-map", "0:v:0", "-map", "[out]", "-c:v", "copy", "-c:a", params["audio_codec"],
                    "-b:a", params["audio_bitrate"], "-movflags", "+faststart", out],
                   check=True, capture_output=True, timeout=timeout_sec)


def render_preview(src: str, out: str, start: float, end: float, *, timeout_sec: int = 120) -> None:
    """A short context preview (window ±2 s), small enough for a phone."""
    subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", src,
                    "-t", f"{end - start:.3f}", "-vf", "scale=-2:720", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", out],
                   check=True, capture_output=True, timeout=timeout_sec)


def probe_streams(path: str) -> dict:
    """Video packet count + duration, to prove the picture was copied."""
    import json as _json

    raw = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets,codec_name:format=duration",
                          "-of", "json", path], capture_output=True, check=True, timeout=120).stdout
    data = _json.loads(raw or b"{}")
    st = (data.get("streams") or [{}])[0]
    return {"video_packets": int(st.get("nb_read_packets") or 0), "codec": st.get("codec_name"),
            "duration_sec": float((data.get("format") or {}).get("duration") or 0.0)}


def outside_residual_db(base_samples, cand_samples, window: Sequence[float], *, margin: float = 0.1) -> float:
    """Peak-normalised RMS of (candidate - base) outside the window ± margin, dB."""
    import numpy as np

    n = min(len(base_samples), len(cand_samples))
    if n == 0:
        return 0.0
    a, b = np.asarray(base_samples[:n]), np.asarray(cand_samples[:n])
    mask = np.ones(n, dtype=bool)
    i0 = max(0, int((window[0] - RAMP_SEC - margin) * SAMPLE_RATE))
    i1 = min(n, int((window[1] + RAMP_SEC + margin) * SAMPLE_RATE))
    mask[i0:i1] = False
    diff = a[mask] - b[mask]
    ref = float(np.sqrt(np.mean(a[mask] ** 2))) + 1e-9
    return round(20.0 * float(np.log10(float(np.sqrt(np.mean(diff ** 2))) / ref + 1e-12)), 1)


def outside_level_delta_db(base_samples, cand_samples, window: Sequence[float], *,
                           margin: float = 0.1) -> float:
    """|RMS level(candidate) - RMS level(base)| outside the window ± margin, dB.

    The existing audio-patch tolerance (tests/test_media.py: the level outside
    a patched window stays within 0.5 dB). The sample residual is recorded
    too, but it measures the AAC re-encode, not the fix."""
    import numpy as np

    n = min(len(base_samples), len(cand_samples))
    if n == 0:
        return 0.0
    a, b = np.asarray(base_samples[:n]), np.asarray(cand_samples[:n])
    mask = np.ones(n, dtype=bool)
    i0 = max(0, int((window[0] - RAMP_SEC - margin) * SAMPLE_RATE))
    i1 = min(n, int((window[1] + RAMP_SEC + margin) * SAMPLE_RATE))
    mask[i0:i1] = False
    la = 10.0 * np.log10(float(np.mean(a[mask] ** 2)) + 1e-12)
    lb = 10.0 * np.log10(float(np.mean(b[mask] ** 2)) + 1e-12)
    return round(float(abs(la - lb)), 2)
