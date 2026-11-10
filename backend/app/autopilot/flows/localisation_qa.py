"""Deterministic QA of a stitched localisation final (AICP-366, "C2").

docs/autopilot-localisation-finishing.md §2. Everything here is free and
local: ffmpeg + numpy on files, plus pure functions over the caption job's
ASR lines (the existing caption pipeline's transcription, AICP-366 scope 2).
No model call, no judgement (the Judge port is disabled in production, Q2).

Pure (planning may call them): :func:`parse_script`, :func:`build_pieces`,
:func:`speaker_genders`, :func:`final_script`, :func:`script_captions`,
:func:`asr_check`, :func:`fix_caption_words`, :func:`qa_cues`,
:func:`comment_cues`, :func:`segments_named`, :func:`apply_cues`,
:func:`best_takes`, :func:`tail_cut_sec`.

Media (compute functions only — run workers, one window at a time, never a
whole video in memory): :func:`window_pitch`, :func:`segment_qa`,
:func:`deliverable_check`.
"""

from __future__ import annotations

import difflib
import math
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Thresholds (docs/assistant-flow-localisation.md step 10/11)
# ---------------------------------------------------------------------------

LENGTH_MIN_DELTA_SEC = -0.2   # a generated clip may be this much shorter than its window
LENGTH_MAX_DELTA_SEC = 0.8    # … or this much longer
MALE_MAX_HZ = 155.0           # median F0 at or below: a male voice
FEMALE_MIN_HZ = 175.0         # at or above: a female voice (between: unknown)
MIN_VOICED_SEC = 0.4          # less voiced speech in a line window: no verdict
PITCH_MIN_HZ, PITCH_MAX_HZ = 70.0, 400.0
LINE_PRESENT_COVERAGE = 0.5   # an expected line counts as said when half its words are heard
GARBLED_BELOW_COVERAGE = 0.5  # below this overall, the ASR is garbled (pl in 46cc6443)
TIMING_WARN_SEC = 2.5
DELIVERABLE_MAX_BYTES = 99_000_000
SAMPLE_RATE = 16000

QA_CUE_HEADER = "Corrections from the automatic QA — follow them exactly:"


# ---------------------------------------------------------------------------
# Script lines of a segment prompt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptLine:
    start: float
    end: float
    tag: str
    text: str
    background: bool


def parse_script(prompt: Optional[str]) -> list[ScriptLine]:
    """The timed ``[t0–t1] TAG (cue): text`` lines of an AICP-335 prompt.

    Same grammar as :func:`app.qa_tools.parse_dialogue`, but the cue is kept
    as far as it matters here: a BACKGROUND line (a device playing under other
    speakers, kept and marked since 2026-10-29) is never voice-checked.
    """
    from ... import qa_tools

    out = []
    for raw in (prompt or "").splitlines():
        m = qa_tools._DIALOGUE_RE.match(raw)
        if not m:
            continue
        cue = m.group(4) or ""
        out.append(ScriptLine(start=float(m.group(1)), end=float(m.group(2)),
                              tag=m.group(3).strip(), text=" ".join(m.group(5).split()),
                              background="BACKGROUND" in cue))
    return out


def speaker_genders(transcript: Optional[Mapping]) -> dict[str, dict]:
    """Prompt tag -> ``{"gender", "kind"}`` from the v3 transcript.

    The tags are the prompt builder's own (``localisation._speaker_profiles``
    over the transcript's lines), so ``WOMAN`` / ``INTERVIEWER`` here are the
    tags the prompt's dialogue lines carry."""
    from ... import localisation

    tr = dict(transcript or {})
    lines = [ln for ln in tr.get("lines") or [] if isinstance(ln, dict)]
    out: dict[str, dict] = {}
    for prof in localisation._speaker_profiles(lines, tr).values():
        out.setdefault(prof["tag"], {"gender": prof.get("gender") or "unknown",
                                     "kind": prof.get("kind") or "person"})
    return out


# ---------------------------------------------------------------------------
# The pieces of a stitched final
# ---------------------------------------------------------------------------


def _st(value) -> str:
    return value.value if hasattr(value, "value") else str(value or "")


def build_pieces(*, stitch_plan: Optional[Sequence], snapshot: Sequence[Mapping],
                 run_segments: Mapping[str, Mapping], run_prompt: Optional[str],
                 library_clips: Mapping[str, Mapping], duration_sec: Optional[float],
                 max_clip_sec: Optional[float], genders: Mapping[str, Mapping]) -> list[dict]:
    """The final's pieces in stitch order (pipeline_v2's stitch, audio_mode=seedance).

    *snapshot*: the run's frozen ``segment_snapshot`` entries; *run_segments*:
    ``segment_def_id -> {id, prompt_override, local_result_path,
    seedance_task_id, status}``; *library_clips*: ``clip_id -> {path,
    prompt}`` (the TEMP probe clip carries the probe's prompt).

    A swap clip is concatenated WHOLE, its script times are seconds from the
    start of its clip window (``localisation.clip_window``); a keep piece is
    the source cut of its own segment; a library clip is its whole file.
    """
    from ... import localisation

    by_id = {str(s["id"]): s for s in snapshot}
    ordered = sorted(snapshot, key=lambda s: int(s.get("index") or 0))
    number = {str(s["id"]): i + 1 for i, s in enumerate(ordered)}
    if stitch_plan is None:
        entries = [{"kind": "segment", "segment_def_id": str(s["id"])} for s in ordered]
    else:
        entries = [dict(e) for e in stitch_plan if isinstance(e, Mapping)]

    def lines_of(prompt):
        out = []
        for ln in parse_script(prompt):
            g = genders.get(ln.tag) or {}
            out.append({"start": ln.start, "end": ln.end, "tag": ln.tag, "text": ln.text,
                        "background": ln.background,
                        "gender": g.get("gender") or "unknown", "kind": g.get("kind") or "person"})
        return out

    pieces = []
    for pos, e in enumerate(entries):
        if e.get("kind") == "library":
            clip = library_clips.get(str(e.get("clip_id"))) or {}
            pieces.append({"pos": pos, "kind": "library", "clip_id": e.get("clip_id"),
                           "clip": clip.get("path"), "lines": lines_of(clip.get("prompt")),
                           "seg_number": clip.get("seg_number"), "window": clip.get("window"),
                           "run_segment_id": None, "segment_def_id": clip.get("segment_def_id")})
            continue
        sid = str(e.get("segment_def_id"))
        sd = by_id.get(sid)
        if sd is None:
            continue
        start, end = float(sd["start_sec"]), float(sd["end_sec"])
        piece = {"pos": pos, "segment_def_id": sid, "seg_number": number[sid],
                 "seg": [round(start, 3), round(end, 3)]}
        if sd.get("action") == "keep":
            piece.update(kind="keep", clip=None, window=[start, end], lines=[],
                         run_segment_id=None)
        else:
            rs = run_segments.get(sid) or {}
            w0, w1 = localisation.clip_window(
                start, end, sd.get("pre_roll_sec") or 0.0, sd.get("post_roll_sec") or 0.0,
                duration_sec, max_clip_sec)
            prompt = rs.get("prompt_override") or run_prompt or ""
            piece.update(kind="swap", clip=rs.get("local_result_path"),
                         window=[round(w0, 3), round(w1, 3)], lines=lines_of(prompt),
                         run_segment_id=rs.get("id"), seedance_task_id=rs.get("seedance_task_id"),
                         prompt=prompt)
        pieces.append(piece)
    return pieces


def piece_secs(qa_result: Optional[Mapping], pieces: Sequence[Mapping]) -> dict[int, float]:
    """``pos -> measured clip length`` from a segment-QA result (else the window)."""
    measured = {int(p["pos"]): p.get("clip_sec") for p in (qa_result or {}).get("pieces") or []}
    out = {}
    for p in pieces:
        sec = measured.get(int(p["pos"]))
        if sec is None and p.get("window"):
            sec = float(p["window"][1]) - float(p["window"][0])
        out[int(p["pos"])] = float(sec or 0.0)
    return out


def final_script(pieces: Sequence[Mapping], secs: Mapping[int, float]) -> list[dict]:
    """Every scripted line on the final's timeline: ``{start, end, text, tag, pos}``.

    Script times are scaled by ``clip_sec / window_sec`` and offset by the
    previous pieces' lengths — the arithmetic of
    :func:`app.qa_tools.captions_from_script`, without its caption gap."""
    out, cursor = [], 0.0
    for p in pieces:
        clip = float(secs.get(int(p["pos"])) or 0.0)
        lines = p.get("lines") or []
        if lines:
            if p.get("window"):
                wlen = float(p["window"][1]) - float(p["window"][0])
            else:
                wlen = clip  # a library clip: script times are its own
            scale = clip / wlen if wlen > 0 else 1.0
            for ln in lines:
                t0 = max(0.0, float(ln["start"])) * scale
                t1 = max(t0, float(ln["end"]) * scale)
                out.append({"start": round(cursor + t0, 3), "end": round(cursor + min(t1, clip), 3),
                            "text": ln["text"], "tag": ln["tag"], "pos": int(p["pos"]),
                            "background": bool(ln.get("background"))})
        cursor += clip
    return out


def script_captions(pieces: Sequence[Mapping], secs: Mapping[int, float]) -> list[dict]:
    """Caption lines from the script (garbled ASR), via qa_tools.captions_from_script."""
    from ... import qa_tools

    segs, cursor = [], 0.0
    for p in pieces:
        clip = float(secs.get(int(p["pos"])) or 0.0)
        if clip <= 0:
            continue
        if p.get("window"):
            s0, s1 = float(p["window"][0]), float(p["window"][1])
        else:
            s0, s1 = 0.0, clip
        segs.append({"source_start": s0, "source_end": max(s1, s0 + 1e-3), "clip_duration": clip,
                     "offset": cursor,
                     "lines": [{"start": ln["start"], "end": ln["end"], "text": ln["text"]}
                               for ln in p.get("lines") or [] if not ln.get("background")]})
        cursor += clip
    return qa_tools.captions_from_script(segs)


# ---------------------------------------------------------------------------
# ASR check (the caption job's lines vs the script)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)
_NO_SPACE_LANGUAGES = {"ja", "zh"}


def norm_word(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.casefold())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return _PUNCT_RE.sub("", w)


def norm_tokens(text: str, language: Optional[str]) -> list[str]:
    if (language or "").lower() in _NO_SPACE_LANGUAGES:
        return [c for c in (norm_word(c) for c in str(text or "")) if c]
    return [t for t in (norm_word(w) for w in str(text or "").split()) if t]


def _brand_tokens(brands: Iterable[Mapping], language: Optional[str]) -> dict[str, dict]:
    out = {}
    for b in brands or []:
        if not isinstance(b, Mapping):
            continue
        toks = norm_tokens(str(b.get("name") or ""), language)
        if len(toks) == 1 and len(toks[0]) >= 3:
            out[toks[0]] = {"name": str(b.get("name")), "pronunciation": str(b.get("pronunciation") or "")}
    return out


def asr_check(expected: Sequence[Mapping], asr_lines: Sequence[Mapping], *,
              brands: Iterable[Mapping] = (), language: Optional[str] = None) -> dict:
    """Is every scripted line in the ASR of the generated audio, in order, brand intact?

    ``difflib.SequenceMatcher`` over the two word sequences: the matching
    blocks are monotonic, so a dropped or swapped line shows as unmatched.
    Returns ``{coverage, garbled, lines: [...], issues: [...], warnings: [...]}``;
    each issue carries the expected line's ``pos`` (the piece it is in)."""
    fg = [dict(e) for e in expected if not e.get("background")]
    bg = [dict(e) for e in expected if e.get("background")]
    exp_words: list[tuple[str, int, str]] = []  # (token, line index, raw word)
    for i, e in enumerate(fg):
        for raw in (str(e["text"]).split() if (language or "") not in _NO_SPACE_LANGUAGES
                    else list(str(e["text"]))):
            t = norm_word(raw)
            if t:
                exp_words.append((t, i, raw))
    asr_words: list[tuple[str, int]] = []
    for j, a in enumerate(asr_lines or []):
        if not isinstance(a, Mapping):
            continue
        for t in norm_tokens(str(a.get("text") or ""), language):
            asr_words.append((t, j))
    sm = difflib.SequenceMatcher(None, [w for w, _, _ in exp_words], [w for w, _ in asr_words],
                                 autojunk=False)
    matched = [False] * len(exp_words)
    match_asr = [None] * len(exp_words)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            matched[a + k] = True
            match_asr[a + k] = asr_words[b + k][1]
    total = len(exp_words)
    coverage = (sum(matched) / total) if total else 1.0
    garbled = bool(total) and coverage < GARBLED_BELOW_COVERAGE
    btoks = _brand_tokens(brands, language)
    lines, issues, warnings = [], [], []
    for i, e in enumerate(fg):
        idx = [k for k, (_, li, _) in enumerate(exp_words) if li == i]
        n = len(idx)
        hit = sum(1 for k in idx if matched[k])
        cov = hit / n if n else 1.0
        present = cov >= LINE_PRESENT_COVERAGE
        row = {"pos": e["pos"], "start": e["start"], "text": e["text"], "coverage": round(cov, 3),
               "present": present}
        lines.append(row)
        if not present:
            row_i = {"kind": "missing_line", "pos": e["pos"], "tag": e.get("tag"),
                     "text": e["text"], "start": e["start"], "coverage": round(cov, 3)}
            if any(b["pos"] == e["pos"] and float(b["start"]) < float(e["end"])
                   and float(b["end"]) > float(e["start"]) for b in bg):
                # Talked over by a BACKGROUND device line: the ASR may have
                # heard the device instead. Evidence for the listening, no more.
                warnings.append(dict(row_i, kind="missing_under_background"))
            else:
                issues.append(row_i)
            continue
        for k in idx:
            tok = exp_words[k][0]
            if tok in btoks and not matched[k]:
                issues.append({"kind": "brand", "pos": e["pos"], "name": btoks[tok]["name"],
                               "pronunciation": btoks[tok]["pronunciation"], "text": e["text"]})
                break
        js = [match_asr[k] for k in idx if match_asr[k] is not None]
        if js:
            heard = asr_lines[min(js)]
            try:
                delta = float(heard.get("start")) - float(e["start"])
            except (TypeError, ValueError):
                delta = 0.0
            if abs(delta) > TIMING_WARN_SEC:
                warnings.append({"kind": "timing", "pos": e["pos"], "text": e["text"],
                                 "delta_sec": round(delta, 2)})
    return {"coverage": round(coverage, 3), "garbled": garbled, "words": total,
            "lines": lines, "issues": [] if garbled else issues, "warnings": warnings,
            "garbled_issues": issues if garbled else []}


def fix_caption_words(asr_lines: Sequence[Mapping], expected: Sequence[Mapping], *,
                      language: Optional[str] = None, max_span: int = 3) -> tuple[list[dict], int]:
    """ASR caption lines with misheard words replaced by the script's.

    Word-level alignment of the whole ASR text against the whole script; a
    ``replace`` opcode of at most *max_span* words on both sides that lies
    inside one ASR line takes the script's words (misheard words, a brand's
    spelling). Inserted / deleted words are kept as heard: the operator
    accepted the audio, and captions must follow what is said. Timings and
    line splits never change. Returns ``(lines, number of words changed)``."""
    if (language or "").lower() in _NO_SPACE_LANGUAGES:
        return [dict(a) for a in asr_lines], 0  # no word boundaries to fix on
    script_words = [w for e in expected if not e.get("background") for w in str(e["text"]).split()]
    asr_words: list[tuple[int, int, str]] = []  # (line, word index, raw)
    lines = [dict(a) for a in asr_lines]
    split = [str(a.get("text") or "").split() for a in lines]
    for li, words in enumerate(split):
        for wi, w in enumerate(words):
            asr_words.append((li, wi, w))
    a_norm = [norm_word(w) for _, _, w in asr_words]
    s_norm = [norm_word(w) for w in script_words]
    sm = difflib.SequenceMatcher(None, a_norm, s_norm, autojunk=False)
    edits: list[tuple[int, int, int, list[str]]] = []  # (line, first, last, new words)
    changes = 0
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op != "replace" or (a1 - a0) > max_span or (b1 - b0) > max_span:
            continue
        rows = {asr_words[k][0] for k in range(a0, a1)}
        if len(rows) != 1:
            continue
        li = rows.pop()
        first, last = asr_words[a0][1], asr_words[a1 - 1][1]
        new = list(script_words[b0:b1])
        new[-1] = _carry_punct(split[li][last], new[-1])
        edits.append((li, first, last, new))
        changes += max(a1 - a0, b1 - b0)
    # Right to left inside a line, so earlier word indices stay valid.
    for li, first, last, new in sorted(edits, key=lambda e: (e[0], -e[1])):
        split[li][first:last + 1] = new
    for li, words in enumerate(split):
        lines[li]["text"] = " ".join(words)
    return lines, changes


def _carry_punct(heard: str, script: str) -> str:
    """The script's word with the heard word's trailing punctuation when the
    script word has none (a line end stays a line end)."""
    if re.search(r"\w$", script, re.UNICODE):
        m = re.search(r"[^\w]+$", heard, re.UNICODE)
        if m:
            return script + m.group(0)
    return script


# ---------------------------------------------------------------------------
# Cues (docs/assistant-flow-localisation.md, known failure modes)
# ---------------------------------------------------------------------------

_OTHER = {"female": "male", "male": "female"}
CUE_ACTION = "Reproduce every body action of the source at its time, and do not undo it afterwards."
CUE_OFF_FACE = "Off-face lines are spoken by the off-screen speaker, never by the visible person."
CUE_CAPTIONS = "Do not add any text, captions or subtitles to the picture."


def _gender_cue(tag: str, gender: str) -> str:
    return f"{tag} is a {gender} voice in every line of this clip — never a {_OTHER[gender]} voice."


def _brand_cue(name: str, pronunciation: str) -> str:
    if pronunciation:
        return f'Pronounce "{name}" exactly as {pronunciation}.'
    return f'Pronounce "{name}" exactly as it is written.'


def qa_cues(issues: Iterable[Mapping]) -> list[str]:
    """Cue bullets for one segment's issues, in a stable order, each once."""
    out: list[str] = []

    def add(text):
        if text and text not in out:
            out.append(text)

    for i in issues:
        k = i.get("kind")
        if k == "gender" and i.get("expected") in _OTHER:
            add(_gender_cue(i.get("tag") or "The speaker", i["expected"]))
        elif k == "missing_line":
            add("Say every dialogue line above, strictly in this order — none may be skipped "
                f'(missing: "{str(i.get("text") or "")[:80]}").')
        elif k == "brand":
            add(_brand_cue(str(i.get("name") or ""), str(i.get("pronunciation") or "")))
        elif k == "captions":
            add(CUE_CAPTIONS)
    return out


_GENDER_WORDS = {"gender", "male", "female", "voice", "voices", "man", "men", "woman", "women",
                 "masculine", "feminine", "mujer", "hombre"}
_KEYWORDS = {
    "action": ("glasses", "action", "gesture", "hand", "takes off", "puts", "put on", "take off"),
    "off_face": ("off-face", "off face", "off-screen", "off screen", "offscreen"),
    "brand": ("pronounc", "brand", "listenify"),
}


def comment_cues(comment: str, piece: Mapping, brands: Iterable[Mapping]) -> list[str]:
    """Known-failure-mode cue bullets a rejection comment's keywords select."""
    text = f" {str(comment or '').casefold()} "
    words = set(re.findall(r"[^\W\d_]+", text))
    out: list[str] = []
    if words & _GENDER_WORDS:
        seen = set()
        for ln in piece.get("lines") or []:
            if ln.get("kind") == "person" and ln.get("gender") in _OTHER and ln["tag"] not in seen:
                seen.add(ln["tag"])
                out.append(_gender_cue(ln["tag"], ln["gender"]))
    if any(k in text for k in _KEYWORDS["action"]):
        out.append(CUE_ACTION)
    if any(k in text for k in _KEYWORDS["off_face"]):
        out.append(CUE_OFF_FACE)
    if any(k in text for k in _KEYWORDS["brand"]):
        for b in brands or []:
            if isinstance(b, Mapping) and b.get("name"):
                out.append(_brand_cue(str(b["name"]), str(b.get("pronunciation") or "")))
    return out


_SEG_RE = re.compile(
    r"(?:\b(?:segments?|segs?|clips?|s)|#)\s*((?:\d+)(?:\s*(?:,|and|&|\+|-|–)\s*\d+)*)", re.I)


def segments_named(comment: str) -> list[int]:
    """1-based segment numbers named in a comment ("segment 3", "segs 2 and 5", "#4", "s3")."""
    out: set[int] = set()
    for m in _SEG_RE.finditer(str(comment or "")):
        body = m.group(1)
        for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", body):
            lo, hi = sorted((int(a), int(b)))
            out.update(range(lo, min(hi, lo + 50) + 1))
        out.update(int(n) for n in re.findall(r"\d+", body))
    return sorted(n for n in out if 0 < n < 1000)


def apply_cues(prompt: str, cues: Sequence[str]) -> str:
    """*prompt* with *cues* as bullets under the QA header; idempotent per bullet."""
    body = (prompt or "").rstrip()
    new = [c for c in cues if c and f"- {c}" not in body]
    if not new:
        return body
    bullets = "\n".join(f"- {c}" for c in new)
    if QA_CUE_HEADER in body:
        return body + "\n" + bullets
    return body + "\n\n" + QA_CUE_HEADER + "\n" + bullets


# ---------------------------------------------------------------------------
# Best take, tail cut
# ---------------------------------------------------------------------------


def best_takes(history: Sequence[Mapping], current: Mapping[str, Mapping],
               segments: Iterable[str]) -> dict[str, Mapping]:
    """For each run segment in *segments*: the earlier take to restore, if any.

    *history*: per round, ``{run_segment_id: {clip, seedance_task_id, score}}``
    (oldest first, the current take excluded); *current*: the same for the
    current round; ``score`` None = not fully measured (never restored, and an
    unmeasured current take is kept). A take wins only with STRICTLY fewer
    issues than the current one (ties keep the newest); among earlier takes
    the fewest issues, then the newest."""
    out = {}
    for rsid in segments:
        cur = current.get(rsid)
        if cur is None or cur.get("score") is None:
            continue  # the current take is not fully measured: keep it
        best = None
        for rnd in history:
            t = rnd.get(rsid)
            if t is None or t.get("score") is None or (
                    t.get("clip") == cur.get("clip")
                    and t.get("seedance_task_id") == cur.get("seedance_task_id")):
                continue
            if best is None or t["score"] <= best["score"]:
                best = t
        if best is not None and best["score"] < cur["score"]:
            out[rsid] = best
    return out


def tail_cut_sec(duration_sec: float) -> float:
    """``duration − 0.01`` floored to 2 decimals (docs step 15)."""
    cents = math.floor(round((float(duration_sec) - 0.01) * 100.0, 6))
    return cents / 100.0


# ---------------------------------------------------------------------------
# Media: voice pitch of one audio window
# ---------------------------------------------------------------------------


def _window_samples(path: str, start: float, duration: float):
    """Mono 16 kHz float32 samples of ``[start, start+duration)`` of *path* (one window)."""
    import numpy as np

    from ... import media

    cmd = media.harden_media_argv([
        media.FFMPEG, "-nostdin", "-v", "error", "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{max(0.01, duration):.3f}", "-i", path, "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError(f"audio decode of {path!r} failed: {' | '.join(tail)}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def median_pitch(samples, sr: int = SAMPLE_RATE) -> tuple[Optional[float], float]:
    """(median F0 in Hz over voiced frames, voiced seconds) — autocorrelation per 40 ms frame."""
    import numpy as np

    frame, hop = int(0.04 * sr), int(0.01 * sr)
    if samples is None or len(samples) < frame:
        return None, 0.0
    n = 1 + (len(samples) - frame) // hop
    min_lag, max_lag = int(sr / PITCH_MAX_HZ), int(sr / PITCH_MIN_HZ)
    nfft = 1 << (2 * frame - 1).bit_length()
    window = np.hanning(frame).astype(np.float32)
    rms_all = []
    f0s = []
    # Two passes over small chunks of frames: energies first, then pitch.
    for c0 in range(0, n, 256):
        idx = np.arange(c0, min(n, c0 + 256))[:, None] * hop + np.arange(frame)[None, :]
        fr = samples[idx]
        fr = fr - fr.mean(axis=1, keepdims=True)
        rms_all.append(np.sqrt((fr * fr).mean(axis=1)))
    rms = np.concatenate(rms_all)
    gate = max(1e-3, 0.1 * float(rms.max()))
    for c0 in range(0, n, 256):
        sel = np.arange(c0, min(n, c0 + 256))
        sel = sel[rms[sel] >= gate]
        if not len(sel):
            continue
        idx = sel[:, None] * hop + np.arange(frame)[None, :]
        fr = samples[idx]
        fr = (fr - fr.mean(axis=1, keepdims=True)) * window
        spec = np.fft.rfft(fr, nfft, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, :frame]
        ac = ac / np.maximum(ac[:, :1], 1e-12)
        seg = ac[:, min_lag:max_lag + 1]
        for row in seg:
            peak = float(row.max())
            if peak < 0.5:
                continue
            # The FIRST local maximum within 90 % of the best: no sub-octave error.
            cand = [i for i in range(1, len(row) - 1)
                    if row[i] >= row[i - 1] and row[i] >= row[i + 1] and row[i] >= 0.9 * peak]
            lag = (cand[0] if cand else int(row.argmax())) + min_lag
            f0s.append(sr / lag)
    if not f0s:
        return None, 0.0
    return float(np.median(f0s)), len(f0s) * hop / sr


def classify_pitch(hz: Optional[float]) -> Optional[str]:
    if hz is None:
        return None
    if hz <= MALE_MAX_HZ:
        return "male"
    if hz >= FEMALE_MIN_HZ:
        return "female"
    return None


def window_pitch(path: str, start: float, end: float) -> dict:
    """``{hz, voiced_sec, measured}`` of the voice in ``[start, end)`` of *path*."""
    hz, voiced = median_pitch(_window_samples(path, start, max(0.0, end - start)))
    measured = classify_pitch(hz) if voiced >= MIN_VOICED_SEC else None
    return {"hz": None if hz is None else round(hz, 1), "voiced_sec": round(voiced, 2),
            "measured": measured}


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------


def _overlaps(ln: Mapping, others: Sequence[Mapping]) -> bool:
    return any(o is not ln and float(o["start"]) < float(ln["end"]) and float(o["end"]) > float(ln["start"])
               for o in others)


def _checkable(ln: Mapping, lines: Sequence[Mapping]) -> bool:
    """A foreground person line with a known gender that no other FOREGROUND line
    overlaps. A BACKGROUND device line under it does not exclude it: it is kept
    in the prompt to play quietly, so the foreground voice dominates the pitch."""
    fg = [o for o in lines if not o.get("background")]
    return (not ln.get("background") and ln.get("kind") == "person"
            and ln.get("gender") in _OTHER and not _overlaps(ln, fg))


def _line_pitches(clip: str, lines: Sequence[Mapping], scale: float, clip_sec: float) -> list[dict]:
    out = []
    for ln in lines:
        if not _checkable(ln, lines):
            continue
        t0, t1 = float(ln["start"]) * scale, min(float(ln["end"]) * scale, clip_sec)
        if t1 - t0 < MIN_VOICED_SEC:
            continue
        m = window_pitch(clip, t0, t1)
        out.append({"tag": ln["tag"], "text": ln["text"][:120], "start": round(t0, 2),
                    "end": round(t1, 2), "expected": ln["gender"], **m})
    return out


def _tag_medians(checks: Sequence[Mapping]) -> dict[str, dict]:
    """Per speaker: the median of its lines' F0 (lines with enough voiced speech)."""
    import statistics

    by: dict[str, list] = {}
    for c in checks:
        if c.get("hz") is not None and float(c.get("voiced_sec") or 0.0) >= MIN_VOICED_SEC:
            by.setdefault(c["tag"], []).append(c)
    out = {}
    for tag, cs in by.items():
        hz = statistics.median(float(c["hz"]) for c in cs)
        out[tag] = {"hz": round(hz, 1), "lines": len(cs), "expected": cs[0]["expected"],
                    "voiced_sec": round(sum(float(c["voiced_sec"]) for c in cs), 2)}
    return out


# Half an octave: a voice this far from the accepted probe's is another voice.
GENDER_SHIFT_OCTAVES = 0.5


def gender_verdicts(medians: Mapping[str, Mapping], baseline: Mapping[str, Mapping]) -> tuple[list, list]:
    """(issues, warnings) for one segment's per-speaker medians.

    An ISSUE needs validated evidence: the speaker's accepted probe voice
    (*baseline*) and a segment voice that classifies as the opposite gender
    AND lies at least half an octave from that baseline toward it. A mismatch
    without a baseline is only a warning (the operator listens)."""
    issues, warnings = [], []
    for tag, m in medians.items():
        expected = m["expected"]
        measured = classify_pitch(m["hz"])
        if measured is None or measured == expected:
            continue
        base = baseline.get(tag)
        row = {"kind": "gender", "tag": tag, "expected": expected, "measured": measured,
               "hz": m["hz"], "baseline_hz": base["hz"] if base else None, "lines": m["lines"]}
        if base:
            shift = math.log2(float(m["hz"]) / float(base["hz"]))
            toward = shift < 0 if measured == "male" else shift > 0
            row["shift_octaves"] = round(shift, 2)
            if toward and abs(shift) >= GENDER_SHIFT_OCTAVES:
                issues.append(row)
                continue
        warnings.append(dict(row, kind="gender_suspect"))
    return issues, warnings


def segment_qa(args: dict, out_dir: str) -> dict:
    """Compute ``localisation_segment_qa`` v1: every piece of the final checked.

    Per swap piece: clip length vs its window (issue outside -0.2…+0.8 s), a
    1 fps sheet vs the source (diffs are warnings), the caption detector at
    2 fps on the clip and on the same source window (an issue only when the
    clip has text the source has not), picture drift (warning), and the voice
    gender per speaker against the accepted probe's voice (§5.9 of the
    design). The library piece (the accepted probe) is measured for the
    timeline and as the voice baseline. Frames are streamed, audio is read one
    line window at a time.
    """
    from ... import media, qa_tools

    source = args["source"]
    cfg = args.get("caption_config")
    pieces = list(args.get("pieces") or [])
    read = sorted({p["clip"] for p in pieces if p.get("clip")} | (
        {args["final"]} if args.get("final") else set()))
    sigs_before = {path: file_sig(path) for path in read}
    # Pass 1: the accepted probe's voices (TEMP library clip) — the baseline.
    baseline: dict[str, dict] = {}
    base_checks: dict[int, list] = {}
    lib_secs: dict[int, float] = {}
    for p in pieces:
        if p["kind"] != "library" or not p.get("clip") or not os.path.isfile(p["clip"]):
            continue
        sec = float(media.probe(p["clip"]).duration_sec or 0.0)
        lib_secs[int(p["pos"])] = sec
        w = p.get("window")
        wlen = (float(w[1]) - float(w[0])) if w else sec
        try:
            checks = _line_pitches(p["clip"], p.get("lines") or [], sec / wlen if wlen > 0 else 1.0, sec)
        except Exception:  # no baseline: gender mismatches become warnings
            checks = []
        base_checks[int(p["pos"])] = checks
        for tag, m in _tag_medians(checks).items():
            baseline.setdefault(tag, m)
    pieces_out, cursor = [], 0.0
    for p in pieces:
        kind = p["kind"]
        row = {"pos": p["pos"], "kind": kind, "seg_number": p.get("seg_number"),
               "run_segment_id": p.get("run_segment_id"), "clip": p.get("clip"),
               "seedance_task_id": p.get("seedance_task_id"), "offset_sec": round(cursor, 3),
               "issues": [], "warnings": [], "complete": True}
        if kind == "keep":
            row["clip_sec"] = round(float(p["window"][1]) - float(p["window"][0]), 3)
            cursor += row["clip_sec"]
            pieces_out.append(row)
            continue
        clip = p.get("clip")
        if not clip or not os.path.isfile(clip):
            row["warnings"].append({"kind": "missing_clip", "detail": f"no clip file for piece {p['pos']}"})
            row["clip_sec"] = (float(p["window"][1]) - float(p["window"][0])) if p.get("window") else 0.0
            row["complete"] = False
            cursor += row["clip_sec"]
            pieces_out.append(row)
            continue
        if kind == "library":
            row["clip_sec"] = round(lib_secs.get(int(p["pos"]), 0.0), 3)
            row["voices"] = _tag_medians(base_checks.get(int(p["pos"])) or [])
            cursor += row["clip_sec"]
            pieces_out.append(row)
            continue
        clip_sec = float(media.probe(clip).duration_sec or 0.0)
        row["clip_sec"] = round(clip_sec, 3)
        cursor += clip_sec
        w0, w1 = float(p["window"][0]), float(p["window"][1])
        wlen = w1 - w0
        delta = clip_sec - wlen
        row["window_sec"] = round(wlen, 3)
        row["length_delta_sec"] = round(delta, 3)
        if delta < LENGTH_MIN_DELTA_SEC or delta > LENGTH_MAX_DELTA_SEC:
            row["issues"].append({"kind": "length", "delta_sec": round(delta, 3),
                                  "clip_sec": round(clip_sec, 3), "window_sec": round(wlen, 3)})
        # Frame sheet vs the source (actions, props, hands, seams).
        try:
            sheet = os.path.join(out_dir, f"segment_{int(p.get('seg_number') or p['pos']):02d}.jpg")
            out = qa_tools.comparison_sheet(
                source, clip, sheet, windows=[{"source_start": w0, "source_end": w1, "result": clip,
                                               "result_start": 0.0, "label": str(p.get("seg_number"))}],
                fractions=None, every_sec=1.0, time_map="scale")
            diffs = [s["diff"] for s in out["samples"] if s.get("diff") is not None]
            row["sheet"] = sheet
            row["max_diff"] = max(diffs) if diffs else None
            row["mean_diff"] = round(sum(diffs) / len(diffs), 1) if diffs else None
        except Exception as exc:  # a sheet is evidence, never a verdict
            row["warnings"].append({"kind": "sheet_failed", "detail": str(exc)[:300]})
        # Burned captions the generated picture added (the source window has none).
        try:
            scan = qa_tools.scan_captions(clip, fps=2.0, offset=row["offset_sec"], config=cfg)
            row["caption_hits"] = scan.hits
            if scan.hits:
                src = qa_tools.scan_captions(source, fps=2.0, start=w0, duration=max(wlen, 0.5),
                                             config=cfg)
                row["source_caption_hits"] = src.hits
                hit = {"kind": "captions", "hits": scan.hits, "source_hits": src.hits,
                       "spans": [{"start": s.start, "end": s.end} for s in scan.spans[:5]]}
                if src.hits:
                    row["warnings"].append(dict(hit, kind="captions_in_source_too"))
                else:
                    row["issues"].append(hit)
        except Exception as exc:
            row["complete"] = False
            row["warnings"].append({"kind": "caption_scan_failed", "detail": str(exc)[:300]})
        # Picture drift inside the window (a speed change / shift): warning.
        try:
            mt = qa_tools.match_times(source, clip, source_start=w0, duration=wlen, every_sec=1.0)
            row["drift_sec"] = mt.get("max_abs_drift")
            if (mt.get("max_abs_drift") or 0.0) > 0.5:
                row["warnings"].append({"kind": "drift", "max_abs_drift": mt["max_abs_drift"],
                                        "median_drift": mt.get("median_drift")})
        except Exception as exc:
            row["warnings"].append({"kind": "drift_failed", "detail": str(exc)[:300]})
        # Voice gender per speaker against the accepted probe's voice.
        scale = clip_sec / wlen if wlen > 0 else 1.0
        try:
            checks = _line_pitches(clip, p.get("lines") or [], scale, clip_sec)
            medians = _tag_medians(checks)
            issues, warnings = gender_verdicts(medians, baseline)
            row["issues"] += issues
            row["warnings"] += warnings
            row["voices"] = medians
            row["gender_checks"] = checks[:40]
        except Exception as exc:
            row["complete"] = False
            row["warnings"].append({"kind": "pitch_failed", "detail": str(exc)[:300]})
        pieces_out.append(row)
    final_sec, final_audio = None, None
    if args.get("final") and os.path.isfile(args["final"]):
        try:
            final_sec = round(float(media.probe(args["final"]).duration_sec or 0.0), 3)
        except Exception:
            final_sec = None
        final_audio = media.audio_fingerprint(args["final"])  # streamed decode
    sigs_after = {path: file_sig(path) for path in read}
    return {"pieces": pieces_out, "pieces_sec": round(cursor, 3), "final_sec": final_sec,
            "baseline": baseline, "final_audio_sha256": final_audio,
            # What was read: a result counts only while these still match (design §5b.3).
            "files": sigs_before, "stable": sigs_before == sigs_after,
            "scoring_id": scoring_id(baseline, cfg)}


def file_sig(path: Optional[str]) -> Optional[list]:
    """``[size, mtime_ns]`` of *path* (the core's run_version uses the same pair)."""
    try:
        st = os.stat(path)
    except (OSError, TypeError):
        return None
    return [st.st_size, st.st_mtime_ns]


def scoring_id(baseline: Mapping, caption_config) -> str:
    """Identity of what a take's issue count was measured against (design §5b.9)."""
    import hashlib
    import json

    body = json.dumps([{k: v.get("hz") for k, v in sorted((baseline or {}).items())},
                       caption_config], sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deliverable_check(args: dict, out_dir: str) -> dict:
    """Compute ``localisation_deliverable_check`` v1: size, length and content hash
    of the deliverable (and of the caption output it was cut from)."""
    from ... import media

    path = args["path"]
    size = os.path.getsize(path)
    info = media.probe(path)
    limit = int(args.get("max_bytes") or DELIVERABLE_MAX_BYTES)
    out = {"path": path, "size_bytes": int(size), "duration_sec": float(info.duration_sec or 0.0),
           "max_bytes": limit, "ok": size <= limit, "sha256": _sha256_file(path)}
    if args.get("also") and os.path.isfile(args["also"]):
        out["also_sha256"] = _sha256_file(args["also"])
    return out
