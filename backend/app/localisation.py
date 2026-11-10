"""
Localisation — transcribe a source creative, translate its hook, build the
Seedance prompt that re-speaks it.

This is the LLM layer of the ``localisation`` project type (design contract:
``docs/localisation.md``). It owns three steps and nothing else:

1. :func:`transcribe_video` — a video-capable model watches the source and
   returns the §4.1 transcript: detected source language, a high-level
   ``video_summary`` and translation-oriented ``scene_context``, one entry per
   spoken line with timestamps and a **visually grounded** speaker label, plus
   any burned-in on-screen text kept separately.
2. :func:`translate_lines` — a text model adapts those lines for the target
   market, matched to each line's *spoken duration* rather than its word
   count, using the transcript's video context, the full transcript and still
   frames of the hook to resolve ambiguous dialogue.
3. :func:`build_prompt` — the translated lines are formatted into the (timed,
   AICP-335) dialogue
   block and dropped into one of the two Seedance templates that live in
   :mod:`app.project_types`.

Nothing here touches the database or the RQ queue: callers (``tasks.py``,
``api_v2.py``) own persistence and scheduling. Every failure surfaces as
:class:`LocalisationError`, which the transcription task records in
``video_projects.transcript_error`` — a model outage must never break analysis.

Two providers, two API shapes
-----------------------------
The two calls go to kie.ai on **different, incompatible routes**, so there is no
shared "LLM client" here — just two thin wrappers over
:class:`app.kie_client.KieClient`:

===============  ===============================================  ==============
step             endpoint                                         model lives in
===============  ===============================================  ==============
transcribe       ``POST /{model}/v1/chat/completions``             the URL path
translate        ``POST /codex/v1/responses``                      the body
===============  ===============================================  ==============

Observed live (2026-08-09), and the reason both parses are tolerant:

* Gemini 2.5 Pro returns its JSON wrapped in a ```` ```json ```` fence **even
  when the request pins ``response_format`` to a strict JSON schema**. The
  schema is still sent (it measurably steadies the field set) but it is a hint,
  not a contract, so :func:`_parse_json_object` has to cope either way.
* The Responses route has no ``response_format`` at all, so the translator is
  asked for JSON in the prompt and parsed the same way.
* kie.ai signals errors with **HTTP 200** plus a ``{"code": 401, "msg": ...}``
  envelope on both routes; :class:`~app.kie_client.KieChatError` covers that and
  is re-raised here as :class:`LocalisationError`.
"""

from __future__ import annotations

import json
import logging
import math
import re
import os
import shutil
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app import ai_models, languages, media, project_types
from app.config import settings
from app import spend
from app.kie_client import KieError, KieModelUnavailableError, get_shared_client
from app.media import MediaError

logger = logging.getLogger(__name__)

# Bump when a prompt changes in a way that makes old output non-comparable.
# Stored in the transcript envelope so a cached transcript always records which
# instructions produced it.
TRANSCRIBE_PROMPT_VERSION = "transcribe/v3"
TRANSLATE_PROMPT_VERSION = "translate/v3"

# video_projects.transcript["schema_version"] — see docs/localisation.md §4.1.
# v2 adds top-level video_summary / scene_context strings. Old v1 envelopes
# remain readable; missing context fields degrade to "" at translate time.
# v3 (AICP-335) adds top-level speakers / actions / brands and per-line
# face_visible / frame — everything the timed Seedance prompt
# (format_timed_dialogue) needs to state gender, off-face lines and body
# actions. All of it is OPTIONAL on read: a v1/v2 transcript, or one a PATCH
# left without it, still builds a timed prompt, just without those clauses.
TRANSCRIPT_SCHEMA_VERSION = 3

# Where transcription uploads land on the kie.ai temp-file host.
_UPLOAD_PATH = "charswap/localisation"

# Separator between the speaker label and the line in the dialogue block.
# Five spaces, byte-identical to the prompt of run 1736426f, the manual run this
# whole feature generalises.
_SPEAKER_SEP = "     "

# How many times translate_lines will ask for a 1:1 answer before giving up.
# Two = one re-ask. A broken correspondence (a split, duplicated or dropped
# line) is a *sampling* accident on a non-deterministic route, not a broken
# request, so the same prompt very often comes back clean the second time, and
# one extra call costs cents against an operator hand-writing the hook (§8).
# It stays at one re-ask: a model that misreads the same prompt twice is not
# going to be argued round on a third try, and the operator is waiting.
_TRANSLATE_ATTEMPTS = 2

# Only re-ask when the failed attempt came back inside this budget. The whole
# endpoint sits behind nginx's 300s proxy_read_timeout, so a retry stacked on a
# slow first call would turn a useful 502 ("the model split line 3") into an
# opaque gateway timeout. A malformed answer normally returns in well under a
# minute, so in practice this never blocks the retry — it only stops the one
# case where retrying makes things worse.
_TRANSLATE_RETRY_BUDGET_SEC = 120.0


class LocalisationError(Exception):
    """Any failure of the transcription/translation layer.

    Deliberately one exception, not a family: every caller treats these the
    same way — record the message against the project and let the operator fall
    back to pasting a translation by hand.
    """


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The transcript is the input to a *video* model that later has to act on it, so
# the two properties worth more than anything else are (a) the speaker label
# being something Seedance can point at on screen, and (b) verbatim text, whose
# length is what makes the re-recorded line fit the shot.
TRANSCRIBE_PROMPT = """\
You are transcribing a short advertising video so that its speech can be \
re-recorded in another language. Watch AND listen to the whole clip before you \
answer.

Return ONLY a JSON object of exactly this shape:

{
  "source_language": "<ISO 639-1 code>",
  "video_summary": "<concise summary of what happens and the video's goal>",
  "scene_context": "<situational context needed to interpret the dialogue>",
  "speakers": [
    {"label": "<exactly the speaker text used in lines>",
     "tag": "<one short UPPERCASE role word, e.g. WOMAN>",
     "kind": "person",
     "gender": "female",
     "voice": "<a few words on the voice: age, pitch, timbre>",
     "description": "<one sentence: who they are and where they are>"}
  ],
  "lines": [
    {"id": 1, "start": 0.0, "end": 2.4,
     "speaker": "<what the speaker looks like>",
     "on_screen": true,
     "face_visible": true,
     "frame": "",
     "text": "<verbatim speech>"}
  ],
  "actions": [
    {"start": 4.0, "end": 5.0,
     "speaker": "<speaker label, or a description of the person>",
     "action": "<the visible body action>",
     "do_not": "<what must not happen afterwards, or empty>"}
  ],
  "brands": [
    {"name": "<brand or product name as spoken>",
     "pronunciation": "<how it is pronounced in the video, e.g. LISTEN-ih-fy>"}
  ],
  "on_screen_text": [
    {"start": 0.0, "end": 3.0, "text": "<burned-in caption or overlay>"}
  ]
}

Rules:

1. LANGUAGE. Detect the spoken language from the audio — never assume it is \
English. Report it as a lowercase ISO 639-1 code ("en", "es", "pt", "ja", \
"de", ...). If several languages are spoken, report the one used for most of \
the speech.

2. VIDEO SUMMARY. "video_summary" is a short, high-level description of the \
video's narrative, purpose and progression. Answer "what is this video about \
and what happens?" in a few sentences. Do NOT restate the transcript line by \
line, and do not invent events you cannot see or hear.

3. SCENE CONTEXT. "scene_context" is the concrete situational evidence a \
translator needs to resolve ambiguous dialogue: who is doing what, \
relationships between speakers, referenced objects, visible actions, tone, \
jokes, idioms, and deictic phrases such as "this thing" or "over there". Keep \
it grounded in the video. Do NOT invent facts. Keep "video_summary" and \
"scene_context" distinct — summary describes the piece; scene context explains \
how to read the lines.

4. SPEAKER LABELS. "speaker" is a short visual description that someone \
looking at a still frame of this video could use to point at the person: "the \
woman in the red jacket", "the man holding the phone", "off-screen \
interviewer (camera person)". NEVER use "Speaker 1", "Speaker A", "Person B", \
"narrator" or any other anonymous label. This text is handed to a video model \
that has to map the line onto a person it can see, so describe clothing, hair, \
position in frame, or what they are holding or doing — whatever separates them \
from the other people present. Use the SAME description every time the same \
person speaks.

5. HOW MANY SPEAKERS. Do not assume a number. There may be one, two or many. \
Some may never appear on camera (a voice-over, or the person holding the \
camera). Set "on_screen": true only when that speaker is visible in frame \
while the line is spoken, and false otherwise; for an off-camera voice, say so \
in the description as well.

6. VERBATIM. Transcribe exactly what is said, including filler words ("um", \
"like"), false starts, repetitions and interruptions. Do not fix grammar, do \
not summarise, do not merge sentences, do not translate. How long a line is is \
what makes the re-recording fit the footage, so "improving" the wording breaks \
it. Write the text in the spoken language's own native script.

7. TIMESTAMPS. "start" and "end" are seconds from the beginning of THIS clip, \
as decimal numbers (e.g. 4.2). One entry per continuous utterance by one \
speaker; a line must never cross a speaker change. List the lines in the order \
they are heard.

8. ON-SCREEN TEXT. Burned-in captions, subtitles, titles, price tags and \
overlay graphics belong in "on_screen_text" and ONLY there — never copy them \
into "lines". When the same words are both spoken and captioned, the spoken \
version goes in "lines" and the caption in "on_screen_text". If the video has \
no on-screen text at all, return an empty array.

9. NO SPEECH. If nobody speaks — music only, ambience only, silence — return \
"lines": [] and a best-effort "source_language" (or "" if there is nothing to \
go on). Still fill "video_summary" and "scene_context" from what is visible. \
NEVER invent, guess or reconstruct dialogue you cannot actually hear. An empty \
list is a correct answer, a hallucinated line is not.
10. SPEAKERS. "speakers" has exactly one entry per distinct "speaker" text \
used in "lines", with "label" copied character for character. "tag" is ONE \
short uppercase role word a video model can follow, unique per person: \
"WOMAN", "MAN", "GIRL", "INTERVIEWER", "NARRATOR"; voices that come out of \
the same device may share a tag such as "APP" or "TV". "kind" is "person" for \
a human speaking (on or off camera) and "device" for a voice played by a \
phone, app, TV, radio or speaker. "gender" is "female", "male" or "unknown", \
judged from the VOICE you hear; it is used to stop the re-recording from \
giving a person the wrong voice, so say "unknown" rather than guess. "voice" \
is a few words on age, pitch and timbre ("mature, lower", "young, \
higher-pitched"). "description" says who they are and where they are: on \
screen and what identifies them, or off screen and what of them is visible \
(for example only a hand holding a microphone).

11. FACE VISIBILITY. For every line, "face_visible" is true only when the \
SPEAKER's face is visible in frame while the line is spoken — stricter than \
"on_screen": a speaker seen from behind, or a shot that cuts to a phone \
screen while she keeps talking, is face_visible false. When it is false, \
"frame" says in a few words what the frame shows instead ("the camera looks \
over her shoulder at the phone in her hands"); otherwise "frame" is "".

12. ACTIONS. "actions" lists visible body actions that a re-generation of \
the shot must reproduce because later footage depends on them: taking \
glasses or a hat off or putting them on, picking up or putting down an \
object, standing up, handing something over. Not gestures, not talking. \
"start"/"end" bracket the action in seconds; "do_not" names what must not \
happen afterwards ("put them back on"), or "" if nothing. Return [] when \
there is none.

13. BRANDS. "brands" lists every brand, product or app name that is spoken, \
with "pronunciation" written the way it is said in the video, syllables \
separated by hyphens and the stressed one in capitals ("LISTEN-ih-fy"). \
Return [] when no brand is spoken.
"""

# response_format is sent as a hint (see the module docstring): Gemini honours
# the field set far more reliably with it than without, but still fences the
# output, so _parse_json_object stays the real contract.
_TRANSCRIPT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcript",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "source_language",
                "video_summary",
                "scene_context",
                "speakers",
                "lines",
                "actions",
                "brands",
                "on_screen_text",
            ],
            "properties": {
                "source_language": {"type": "string"},
                "video_summary": {"type": "string"},
                "scene_context": {"type": "string"},
                "speakers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "label", "tag", "kind", "gender", "voice", "description",
                        ],
                        "properties": {
                            "label": {"type": "string"},
                            "tag": {"type": "string"},
                            "kind": {"type": "string", "enum": ["person", "device"]},
                            "gender": {
                                "type": "string",
                                "enum": ["female", "male", "unknown"],
                            },
                            "voice": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["start", "end", "speaker", "action", "do_not"],
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "speaker": {"type": "string"},
                            "action": {"type": "string"},
                            "do_not": {"type": "string"},
                        },
                    },
                },
                "brands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "pronunciation"],
                        "properties": {
                            "name": {"type": "string"},
                            "pronunciation": {"type": "string"},
                        },
                    },
                },
                "lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "id", "start", "end", "speaker", "on_screen",
                            "face_visible", "frame", "text",
                        ],
                        "properties": {
                            "id": {"type": "integer"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "speaker": {"type": "string"},
                            "on_screen": {"type": "boolean"},
                            "face_visible": {"type": "boolean"},
                            "frame": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
                "on_screen_text": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["start", "end", "text"],
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "text": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

# Timing is rule 1 because it is the failure mode that actually shipped: a
# literal EN->JA translation of the 1736426f hook overran the shot badly. The
# model is given seconds per line and told to count seconds, not words.
#
# v3 (2026-10-20) hands the translator what v2 only described: still frames of
# the hook, the whole transcript, the burned-in text and the target market. An
# A/B on six real projects (blind gemini-3.1-pro judge that watched the video)
# scored v2 on gpt-5-6-luna 5.6/10 and v3 on gpt-6-1-sol 8.3/10. The context is
# what moved the score, but only a strong model uses it: v3 on luna scored 5.4.
# The frames catch transcription mistakes — Gemini heard "I'm dazzling" over a
# woman bedazzling her phone, and only the variants that saw the frames wrote
# "le pongo brillitos" instead of a literal "estoy deslumbrante".
TRANSLATE_PROMPT = """\
You are a senior native {target_language} copywriter who localises short-form \
paid social video ads (TikTok, Instagram Reels, YouTube Shorts) from \
{source_language}. TARGET MARKET: {market}. You are adapting the spoken hook \
of the ad below. Your lines will be spoken aloud by an AI video model and \
lip-synced onto the original footage, so they must sound like a real creator \
from that market talking to camera — not like a translation.

You get, in order: a summary and the scene context written by someone who \
watched the video; the FULL transcript of the source (the hook lines you are \
adapting are marked with >>, the rest is context only and will NOT be \
re-spoken); the burned-in on-screen text; the hook lines themselves; and, \
after these instructions, still frames from the hook, each labelled with its \
timestamp, so you can see who speaks, what they are doing and what they are \
pointing at. The transcript was made by a machine and can mishear a word — \
when the frames clearly contradict it, trust the frames.

VIDEO SUMMARY:
{video_summary}

SCENE CONTEXT:
{scene_context}

FULL SOURCE TRANSCRIPT (seconds from the start; >> marks the hook lines):
{full_transcript}

ON-SCREEN TEXT (stays in {source_language} on the video — never translate or \
reproduce it, but do not contradict it):
{on_screen_text}

THE HOOK LINES TO ADAPT. Each carries its id, who says it, whether the speaker \
is visible while saying it (lip-sync matters most then), the DURATION IN \
SECONDS of the original delivery and the pause that follows it:

{lines_block}

Rules:

1. TIMING IS THE HARD CONSTRAINT. Each adapted line must take about as long \
to SAY OUT LOUD in {target_language}, at the energy of the original delivery, \
as the original took — within roughly 15%. Count syllables and seconds. Do \
NOT match the word count or the character count: languages differ enormously \
in how much time the same meaning takes, and a literal translation usually \
overruns. If a draft is too long, cut it down; if it is too short, use a \
fuller phrasing or a natural filler. Overrunning is the worse failure — it \
makes the delivery race or run off the end of the shot. A line must not \
swallow the pause after it. Each line also carries a character budget for its \
window: aim for the "about" figure and NEVER exceed the max — the only \
exception is a line that is nothing but a name or brand that cannot be \
shortened. When a line is marked as cut by the clip, its window is shorter \
than the original: write a complete thought that fits it.

2. ADAPT, DO NOT TRANSLATE. This is ad copy, not a document. Write what a \
native creator from the target market would actually say in this exact \
situation to stop the scroll. Keep each line's intent, emotional beat and its \
job in the ad (hook, problem, reveal, call to action); change wording, \
idioms, examples and jokes freely so they land for that audience. Use the \
frames and the scene context to resolve "this", "here", "look", gestures and \
anything shown on screen — the words must match what the viewer sees.

3. KEEP THE REGISTER AND THE PERSON. Match how casual or formal the original \
is, and the speaker's apparent gender and age — grammatical agreement must fit \
the person on screen. Offhand spoken {source_language} becomes offhand spoken \
{target_language}: contractions, slang, interjections and all. Use the \
vocabulary, spelling and forms of address of the target market. Never turn \
speech into writing.

4. LEAVE THESE UNTRANSLATED, exactly as written: brand, product and app names \
(e.g. Listenify), people's names, URLs, and the numbers themselves. Unit and \
currency WORDS are translated like any other word ("165 miles" becomes "165 \
milhas" in Portuguese), never left in {source_language}.

5. NATIVE SCRIPT ONLY. Write {target_language} in its own writing system. \
Never romanise it (no romaji, no pinyin, no transliteration) and never leave \
{source_language} words in place unless a native speaker of the target market \
would genuinely say that loanword.

6. ONE LINE IN, ONE LINE OUT. Return exactly one entry per hook line, with the \
SAME id, in the SAME order. Never merge two lines, never split one into two, \
never drop one, never add one. Do not translate the speaker labels — they are \
context for you, not output.

7. SPOKEN DIALOGUE ONLY. Adapt only the >> hook lines. Nothing from the rest \
of the transcript and nothing from the on-screen text goes into your output.

Before answering, draft each line, say it aloud in your head, time it against \
its duration and revise. Reply with ONLY this JSON object — no explanation, no \
commentary, no markdown code fence:

{{"lines": [{{"id": 1, "text": "<adapted line>"}}]}}
"""

# Who the localised hook is for, per target language. A bare "Spanish" let the
# translator drift into Spain ("móvil") for a Mexican audience, and a bare
# "Portuguese" into Portugal ("conduzo", "até ao") for a Brazilian one — both
# seen in the 2026-10-20 A/B. Operator-confirmed markets; a language missing
# here gets a neutral "native speakers of <language>".
_TARGET_MARKETS: dict[str, str] = {
    "en": "the United States (American English)",
    "es": (
        "Mexico and Latin America (Mexican Spanish — never Spain Spanish: "
        "no vosotros, \"celular\" not \"móvil\")"
    ),
    "pt": (
        "Brazil (Brazilian Portuguese — never European Portuguese: "
        "\"você\", \"dirigir\", \"até o\")"
    ),
    "fr": "France (French as spoken in France)",
    "de": "Germany (German as spoken in Germany)",
    "it": "Italy",
    "pl": "Poland",
    "ja": "Japan",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json|jsonc)?\s*\n(.*?)\n?\s*```\s*$", re.S | re.I)


def _parse_json_object(raw: str, *, what: str) -> dict:
    """Parse an LLM answer that is *supposed* to be one JSON object.

    Tolerant in exactly two ways, both of which real responses need:

    * a ```` ```json ```` fence around the object is stripped — Gemini 2.5 Pro
      emits one even under a strict ``response_format`` schema;
    * failing that, the outermost ``{...}`` span is extracted, which recovers
      answers with a sentence of preamble.

    Raises:
        LocalisationError: When nothing parseable is found.
    """
    text = (raw or "").strip()
    if not text:
        raise LocalisationError(f"{what}: model returned an empty response")

    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    for candidate in (text, _outermost_object(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LocalisationError(
        f"{what}: could not parse a JSON object out of the model response: "
        f"{raw[:300]!r}"
    )


def _outermost_object(text: str) -> str | None:
    """Return the ``{...}`` span from the first ``{`` to the last ``}``."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _as_float(value, default: float = 0.0) -> float:
    """Coerce a model-supplied number (which may arrive as a string) to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_language(code) -> str:
    """Reduce a model-reported language tag to a bare lowercase code.

    ``"EN"`` and ``"en-US"`` both become ``"en"``; anything unusable becomes
    ``""``, which callers surface rather than silently defaulting to English.
    """
    text = str(code or "").strip().lower()
    if not text:
        return ""
    return re.split(r"[-_]", text)[0]


def _normalise_context_text(value) -> str:
    """Coerce a model-supplied ``video_summary`` / ``scene_context`` to a string.

    Missing, null, non-string or otherwise unusable values become ``""`` rather
    than raising — a transcription that got the lines right must still store a
    usable envelope even when the model skipped the context fields.
    """
    if not isinstance(value, str):
        return ""
    return value.strip()


def _language_name(code: str | None) -> str:
    """Human name for a language code, lowercased for prose interpolation.

    Uses :mod:`app.languages` when the code is one the pipeline delivers, and
    falls back to the raw code otherwise — the *detected* source
    language is not restricted to that set, and printing "english" for an
    unrecognised code (which ``languages.label_for`` would do) would be a lie
    baked into the Seedance prompt.
    """
    key = _normalise_language(code)
    spec = languages.LANGUAGES.get(key)
    return spec.label.lower() if spec else (key or "the source language")


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


def _clean_lines(raw_lines) -> list[dict]:
    """Coerce the model's ``lines`` array into the §4.1 shape.

    Drops entries with no text (a model asked for verbatim speech occasionally
    emits an empty placeholder), sorts by start time, and renumbers ``id``
    1..N so the ids are dense and ordered — :func:`translate_lines` rejoins on
    them, so they have to be trustworthy regardless of what the model numbered.
    """
    if not isinstance(raw_lines, list):
        return []

    cleaned: list[dict] = []
    for item in raw_lines:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"), start)
        line = {
            "id": 0,  # replaced below
            "start": start,
            "end": max(end, start),
            "speaker": str(item.get("speaker") or "").strip() or "speaker",
            "on_screen": bool(item.get("on_screen")),
            "text": text,
        }
        line.update(clean_line_visibility(item))
        cleaned.append(line)

    cleaned.sort(key=lambda ln: ln["start"])
    for index, line in enumerate(cleaned, start=1):
        line["id"] = index
    return cleaned


def clean_line_visibility(item) -> dict:
    """The optional v3 per-line visibility fields of *item*, normalised.

    Returns a dict holding ``face_visible`` (only when it is a real bool — a
    missing or malformed value stays ABSENT, which the prompt builder reads as
    "unknown, fall back to on_screen") and ``frame`` (only when non-empty).
    Shared by the transcribe cleaner and ``PATCH /transcript`` so an operator
    edit round-trips the same fields the model produced.
    """
    extras: dict = {}
    if not isinstance(item, dict):
        return extras
    face_visible = item.get("face_visible")
    if isinstance(face_visible, bool):
        extras["face_visible"] = face_visible
    frame = item.get("frame")
    if isinstance(frame, str) and frame.strip():
        extras["frame"] = " ".join(frame.split())
    return extras


_SPEAKER_KINDS = {"person", "device"}
_GENDERS = {"female", "male"}
_TAG_RE = re.compile(r"[^A-Z0-9]+")


def _clean_tag(value) -> str:
    """An uppercase one-or-two-word role tag, or "" when unusable."""
    if not isinstance(value, str):
        return ""
    tag = _TAG_RE.sub(" ", value.upper()).strip()
    return " ".join(tag.split()[:2])[:24]


def clean_speakers(raw_items) -> list[dict]:
    """Coerce a v3 ``speakers`` array; first entry per label wins.

    Public because ``PATCH /transcript`` normalises an operator-edited array
    with it: a malformed entry is dropped rather than refused, since every
    field is optional prompt decoration, never a reason to lose an edit.
    """
    if not isinstance(raw_items, list):
        return []
    cleaned: list[dict] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        label = _normalise_context_text(item.get("label"))
        if not label or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        kind = _normalise_context_text(item.get("kind")).lower()
        gender = _normalise_context_text(item.get("gender")).lower()
        cleaned.append(
            {
                "label": label,
                "tag": _clean_tag(item.get("tag")),
                "kind": kind if kind in _SPEAKER_KINDS else "person",
                "gender": gender if gender in _GENDERS else "unknown",
                "voice": _normalise_context_text(item.get("voice")),
                "description": _normalise_context_text(item.get("description")),
            }
        )
    return cleaned


def clean_actions(raw_items) -> list[dict]:
    """Coerce a v3 ``actions`` array into ``{start,end,speaker,action,do_not}``."""
    if not isinstance(raw_items, list):
        return []
    cleaned: list[dict] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        action = _normalise_context_text(item.get("action")).rstrip(".").strip()
        if not action:
            continue
        start = _finite_or_none(item.get("start"))
        end = _finite_or_none(item.get("end"))
        if start is None:
            continue
        if end is None or end < start:
            end = start
        do_not = _normalise_context_text(item.get("do_not")).rstrip(".").strip()
        do_not = re.sub(r"^(?:do\s+not|don['’]t|never)\s+", "", do_not, flags=re.I)
        cleaned.append(
            {
                "start": start,
                "end": end,
                "speaker": _normalise_context_text(item.get("speaker")),
                "action": action,
                "do_not": do_not,
            }
        )
    cleaned.sort(key=lambda item: item["start"])
    return cleaned


def clean_brands(raw_items) -> list[dict]:
    """Coerce a v3 ``brands`` array; one entry per name."""
    if not isinstance(raw_items, list):
        return []
    cleaned: list[dict] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = _normalise_context_text(item.get("name"))
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        cleaned.append(
            {
                "name": name,
                "pronunciation": _normalise_context_text(item.get("pronunciation")),
            }
        )
    return cleaned


def _finite_or_none(value) -> float | None:
    """*value* as a finite, non-negative float, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _clean_on_screen_text(raw_items) -> list[dict]:
    """Coerce the model's ``on_screen_text`` array into the §4.1 shape."""
    if not isinstance(raw_items, list):
        return []

    cleaned: list[dict] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"), start)
        cleaned.append({"start": start, "end": max(end, start), "text": text})

    cleaned.sort(key=lambda item: item["start"])
    return cleaned


@contextmanager
def _upload_proxy(video_path: str):
    """Yield the path to upload for *video_path*, cleaning up after the caller.

    Normally a small re-encode built by :func:`app.media.make_llm_proxy`. If
    that fails the ORIGINAL path is yielded instead: a proxy is an optimisation
    on a file that may well have been uploadable as it stood, so a broken
    re-encode must not be the thing that fails a transcription. The upload's own
    size guard then judges the original on its merits and, if it is too big,
    says so in those words.
    """
    tmpdir: str | None = None
    try:
        try:
            tmpdir = tempfile.mkdtemp(prefix="transcribe-proxy-")
            upload_src = media.make_llm_proxy(
                video_path, os.path.join(tmpdir, "proxy.mp4")
            )
        except Exception as exc:  # noqa: BLE001 — deliberately broad, see below
            # Nothing that goes wrong building an optimisation may be what
            # fails a transcription, so this catches everything rather than
            # just MediaError: probe() also raises KeyError on a stream with
            # no width and ValueError on an unparseable duration, and mkdtemp
            # raises OSError on a full or read-only /tmp. transcribe_video's
            # contract with its callers is LocalisationError, and none of those
            # are it.
            logger.warning(
                "transcribe: could not build an upload proxy for %s (%s: %s) — "
                "uploading the original",
                video_path, type(exc).__name__, exc,
            )
            upload_src = video_path
        yield upload_src
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def transcribe_video(
    video_path: str,
    *,
    kie=None,
    model: str | None = None,
    spend_scope: str | None = None,
) -> dict:
    """
    Transcribe a source video into the §4.1 transcript dict.

    Uploads *video_path* to the kie.ai temp-file host, hands the public URL to a
    video-capable model in the ``image_url`` envelope (kie.ai uses that envelope
    for every media type, video included), and normalises the answer.

    A clip with no speech is a **success**, not a failure: it comes back with
    ``lines: []``, which the caller stores as ``transcript_status="empty"``.

    What is uploaded is **not** *video_path* itself but a cheap re-encode of it
    (:func:`app.media.make_llm_proxy`): the model reads the video, so delivery
    resolution and bitrate buy nothing, while the kie.ai upload host refuses a
    large body outright and the model bills per input token. A source too big to
    upload raw is the normal case for a whole-video localisation, not an edge
    one — 59s of 1080x1920 phone footage is 136 MB.

    Args:
        video_path: Local path to the clip to transcribe. A full source is fine
            — it is proxied down before upload — and is what a project spanning
            more than its hook needs.
        kie: Optional :class:`~app.kie_client.KieClient`; defaults to the shared
            module-level client.
        model: Override ``settings.LOCALISATION_TRANSCRIBE_MODEL``. This value
            is a **URL path segment** on the kie.ai chat route, so it must be a
            real kie.ai model slug.
            If it is unavailable, each of
            ``settings.LOCALISATION_TRANSCRIBE_FALLBACK_MODELS`` is tried in
            turn with the same upload; ``model`` in the result names whichever
            one answered.

    Returns:
        The §4.1 dict: ``schema_version``, ``model``, ``prompt_version``,
        ``created_at``, ``source_language``, ``video_summary``,
        ``scene_context``, ``speakers``, ``lines`` (with v3 ``face_visible``/``frame``),
        ``actions``, ``brands``, ``on_screen_text``.

    Raises:
        LocalisationError: On upload failure, API failure, or an unparseable
            response.
    """
    client = kie or get_shared_client()
    resolved_model = model or settings.LOCALISATION_TRANSCRIBE_MODEL

    try:
        with _upload_proxy(video_path) as upload_src:
            video_url = client.upload_file(
                upload_src,
                upload_path=_UPLOAD_PATH,
                # Name the upload after the source, not the throwaway proxy —
                # this string is what shows up in kie.ai's file listing — but
                # make it unique: the host overwrites same-named files in a
                # folder, and every caption source is called "final.mp4". Two
                # captions started together on 2026-09-25 (runs 8959e2fb and
                # 38c51a37) both got the transcript of whichever uploaded last.
                file_name=f"{uuid.uuid4().hex}_{Path(video_path).name}",
            )
    except KieError as exc:
        raise LocalisationError(f"transcribe: upload failed: {exc}") from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSCRIBE_PROMPT},
                # kie.ai passes video, audio and PDFs through the image_url
                # envelope too — this is not a mistake.
                {"type": "image_url", "image_url": {"url": video_url}},
            ],
        }
    ]

    # The primary model, then the fallbacks — each tried only when the one
    # before is *unavailable*, so a bad key or an unparseable answer still
    # fails fast instead of being billed again. The upload above is reused.
    models = [resolved_model]
    for fallback in settings.LOCALISATION_TRANSCRIBE_FALLBACK_MODELS.split(","):
        fallback = fallback.strip()
        if fallback and fallback not in models:
            models.append(fallback)

    unavailable: list[str] = []
    for attempt, candidate in enumerate(models):
        try:
            raw = client.chat_completion(
                model=candidate,
                messages=messages,
                response_format=_TRANSCRIPT_JSON_SCHEMA,
                timeout_sec=settings.LOCALISATION_TIMEOUT_SEC,
            )
        except KieModelUnavailableError as exc:
            unavailable.append(str(exc))
            if attempt + 1 < len(models):
                logger.warning(
                    "transcribe: %s unavailable (%s); falling back to %s",
                    candidate, exc, models[attempt + 1],
                )
                continue
            raise LocalisationError(
                "transcribe: " + "; then ".join(unavailable)
            ) from exc
        except KieError as exc:
            raise LocalisationError(f"transcribe: {exc}") from exc
        # Billed: measure it against the caller's reservation (AICP-155).
        spend.record_draw(spend_scope, spend.llm_credits(), kind="transcribe")
        resolved_model = candidate
        break

    data = _parse_json_object(raw, what="transcribe")
    lines = _clean_lines(data.get("lines"))
    transcript = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "model": resolved_model,
        "prompt_version": TRANSCRIBE_PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_language": _normalise_language(data.get("source_language")),
        "video_summary": _normalise_context_text(data.get("video_summary")),
        "scene_context": _normalise_context_text(data.get("scene_context")),
        "speakers": clean_speakers(data.get("speakers")),
        "lines": lines,
        "actions": clean_actions(data.get("actions")),
        "brands": clean_brands(data.get("brands")),
        "on_screen_text": _clean_on_screen_text(data.get("on_screen_text")),
    }
    logger.info(
        "Transcribed %s: language=%r, lines=%d, on_screen_text=%d (model=%s)",
        video_path,
        transcript["source_language"],
        len(lines),
        len(transcript["on_screen_text"]),
        resolved_model,
    )
    return transcript


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


# Rough conversational throughput, characters per second — the yardstick for
# "will this line fit its shot". Latin-script casual speech runs ~150 wpm, i.e.
# ~14 chars/s counting spaces; Japanese has no spaces and mixes dense kanji with
# kana, so the same second buys far fewer characters. Crude by design. It is
# shared by the translator's per-line character budget, the shorten-pass that
# re-asks for lines that blow it, and the operator warnings in api_v2, so all
# three agree on what "too long" means.
CHARS_PER_SEC: dict[str, float] = {"ja": 6.0}
DEFAULT_CHARS_PER_SEC = 14.0
# A translation estimated past this multiple of its window overruns (and is
# re-asked once, then warned about); below the floor it probably lost content.
SPEECH_OVERRUN_RATIO = 1.5
SPEECH_UNDERRUN_RATIO = 0.5
# The translator is told to stay under this multiple — well inside the
# overrun ratio, because its own character counting is approximate.
_BUDGET_MAX_RATIO = 1.25


def chars_per_sec(language: str | None) -> float:
    return CHARS_PER_SEC.get(_normalise_language(language), DEFAULT_CHARS_PER_SEC)


def estimated_speech_sec(text: str, language: str | None) -> float:
    """Crude estimate of how long *text* takes to say in *language*."""
    return len((text or "").strip()) / chars_per_sec(language)


def speaking_window(
    line: dict, *, cuts=(), hook_end: float | None = None
) -> tuple[float, float]:
    """The ``(start, end)`` the translated *line* actually has to be spoken in.

    Normally the original line's own span. Two things shorten it:

    * the hook end — the picture after it is the stitched tail, so a line that
      runs past the hook must be finished by then;
    * a cut between two swap clips inside the hook — the line is generated by
      the clip holding its MIDPOINT (``api_v2._assign_lines_to_segments``), so
      it keeps only the part of its span on that side of the cut.

    A line crossing one of these used to be translated against its full
    original duration and then flagged "cut mid-sentence"; handing the
    translator the real window lets it write a complete, shorter line instead.
    """
    start = _as_float(line.get("start"))
    end = _as_float(line.get("end"), start)
    midpoint = (start + end) / 2
    if hook_end is not None and end > hook_end > start:
        end = hook_end
    for cut in sorted(cuts or ()):
        if start < cut < end:
            if midpoint < cut:
                end = cut
            else:
                start = cut
    return start, max(start, end)


def _char_budget(seconds: float, language: str | None) -> tuple[int, int]:
    """``(target, max)`` characters for a line spoken in *seconds*."""
    rate = chars_per_sec(language)
    return max(1, round(seconds * rate)), max(1, int(seconds * rate * _BUDGET_MAX_RATIO))


def _lines_block(
    lines: list[dict],
    context_lines: list[dict],
    *,
    target_language: str | None = None,
    cuts=(),
    hook_end: float | None = None,
) -> str:
    """Render the hook lines for the translation prompt, one block per line.

    Each block carries the id (the rejoin key), the speaker and whether they
    are on screen (context for register, gender and lip-sync), the line's
    speaking WINDOW (:func:`speaking_window` — shorter than the original line
    when the clip is cut inside it), the **duration in seconds** rule 1 of
    :data:`TRANSLATE_PROMPT` tells the model to budget against, a character
    budget in the same yardstick the overrun check uses, and the pause before
    the next line of *context_lines*.
    """
    starts = sorted(_as_float(line.get("start")) for line in context_lines or [])
    blocks = []
    for line in lines:
        orig_start = _as_float(line.get("start"))
        orig_end = _as_float(line.get("end"), orig_start)
        start, end = speaking_window(line, cuts=cuts, hook_end=hook_end)
        duration = max(0.0, end - start)
        target, most = _char_budget(duration, target_language)
        later = [s for s in starts if s > orig_start]
        if hook_end is not None and end >= hook_end - 1e-6:
            pause = "none (the clip ends here)"
        else:
            pause = f"{max(0.0, later[0] - end):.1f}s" if later else "none (last line)"
        speaker = str(line.get("speaker") or "").strip() or "speaker"
        visible = "visible" if line.get("on_screen") else "off-screen"
        cut_note = ""
        if (start, end) != (orig_start, orig_end):
            cut_note = (
                f"\n    NOTE: the original runs {orig_start:.1f}-{orig_end:.1f}s but "
                f"the clip is cut at {end if end < orig_end else start:.1f}s — say a "
                f"complete, natural line that fits {start:.1f}-{end:.1f}s, never a "
                f"sentence cut off mid-way."
            )
        blocks.append(
            f"[{line.get('id')}] speaker: {speaker} ({visible}) | "
            f"{start:.1f}-{end:.1f}s | duration: {duration:.1f}s | "
            f"about {target} characters, max {most} | pause after: {pause}\n"
            f"    {line.get('text', '')}{cut_note}"
        )
    return "\n\n".join(blocks)


def _full_transcript_block(context_lines: list[dict], hook_ids: set) -> str:
    """Every transcript line, with ``>>`` on the hook lines being adapted."""
    rows = []
    for line in context_lines:
        marker = ">>" if line.get("id") in hook_ids else "  "
        start = _as_float(line.get("start"))
        end = _as_float(line.get("end"), start)
        speaker = str(line.get("speaker") or "").strip() or "speaker"
        rows.append(
            f"{marker} [{line.get('id')}] {start:.1f}-{end:.1f}s {speaker}: "
            f"{line.get('text', '')}"
        )
    return "\n".join(rows) or "(none provided)"


def _on_screen_block(on_screen_text) -> str:
    rows = []
    for item in on_screen_text or []:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"), start)
        rows.append(f"{start:.1f}-{end:.1f}s: {str(item['text']).strip()}")
    return "\n".join(rows) or "(none)"


def _target_market(code: str) -> str:
    key = _normalise_language(code)
    return _TARGET_MARKETS.get(key) or f"native speakers of {_language_name(key)}"


def hook_frame_times(
    lines: list[dict], hook_end: float, *, max_frames: int | None = None
) -> list[float]:
    """Timestamps of the still frames handed to the translator.

    One near the start of the hook plus one at the midpoint of each hook line —
    the moment the line's speaker is most likely on screen saying it — all
    inside ``[0, hook_end)``. More lines than *max_frames* are thinned evenly
    rather than truncated, so a long hook is still seen end to end.
    """
    cap = settings.LOCALISATION_TRANSLATE_FRAMES if max_frames is None else max_frames
    if cap <= 0 or hook_end <= 0:
        return []
    times = {round(min(0.3, hook_end / 2), 2)}
    for line in lines or []:
        start = _as_float(line.get("start"))
        mid = round((start + _as_float(line.get("end"), start)) / 2, 2)
        if 0.0 <= mid < hook_end:
            times.add(mid)
    ordered = sorted(times)
    if len(ordered) <= cap:
        return ordered
    if cap == 1:
        return ordered[:1]
    step = (len(ordered) - 1) / (cap - 1)
    return [ordered[round(i * step)] for i in range(cap)]


def _entry_line_id(item) -> int | None:
    """The line id an answer entry claims, or ``None`` when it claims none.

    ``None`` means the entry cannot be attributed to any line of the request —
    it is not an object, has no ``id``, or carries something that is not a whole
    number. Those entries are never silently dropped by the caller: an entry
    with a mangled id is most often *half of a split line*, i.e. exactly the
    text we must not lose.

    ``3.5`` and ``"3.5"`` are refused rather than truncated to ``3``, which
    would quietly file a fragment against a real line.
    """
    if not isinstance(item, dict):
        return None
    value = item.get("id")
    if isinstance(value, bool):  # bool is an int subclass; True is not id 1
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rejoin_translations(raw: str, lines: list[dict]) -> list[dict]:
    """Parse the translator's answer and rejoin it onto *lines*, strictly 1:1.

    Split out of :func:`translate_lines` so that "did the model obey the output
    contract?" is one self-contained, retryable question.

    The rejoin is by id, so **every** deviation from one-entry-per-requested-id
    has to be an error rather than a merge: the ids are the only thing tying a
    translated string back to the shot it will be spoken over. In particular a
    repeated id must not be a last-write-wins ``dict`` assignment — when the
    model splits one line into two entries that share an id, keeping the last
    one silently drops the first half of the sentence, and nothing downstream
    can tell that the prompt is now missing half a line.

    Checked, in the order the operator can act on:

    1. duplicated ids — a split or copy-pasted line;
    2. entries with no usable id — unattributable text, usually a fragment;
    3. requested ids with no text — a dropped or merged-away line;
    4. ids that were never requested — an invented line.

    Raises:
        LocalisationError: On an unparseable answer or any of the four above.
    """
    data = _parse_json_object(raw, what="translate")
    raw_out = data.get("lines")
    if not isinstance(raw_out, list):
        raise LocalisationError(
            f"translate: response has no 'lines' array: {raw[:300]!r}"
        )

    by_id: dict[int, str] = {}
    counts: Counter[int] = Counter()
    unusable: list[str] = []
    for position, item in enumerate(raw_out, start=1):
        line_id = _entry_line_id(item)
        if line_id is None:
            unusable.append(f"#{position}={str(item)[:60]}")
            continue
        counts[line_id] += 1
        text = str(item.get("text") or "").strip()
        if text:
            by_id[line_id] = text

    duplicated = sorted(line_id for line_id, seen in counts.items() if seen > 1)
    if duplicated:
        detail = ", ".join(f"id={line_id} x{counts[line_id]}" for line_id in duplicated)
        raise LocalisationError(
            f"translate: model returned duplicate line ids ({detail}) — a split "
            f"or repeated line cannot be rejoined 1:1 without silently dropping "
            f"half of the script"
        )

    if unusable:
        noun = "entry" if len(unusable) == 1 else "entries"
        raise LocalisationError(
            f"translate: {len(unusable)} response {noun} with no usable line id "
            f"({unusable[:3]}) — that text cannot be matched to a line, and is "
            f"usually half of one the model split"
        )

    wanted = [line.get("id") for line in lines]
    matched = sum(1 for line_id in wanted if line_id in by_id)

    translated: list[dict] = []
    for line in lines:
        line_id = line.get("id")
        text = by_id.get(line_id)
        if not text:
            raise LocalisationError(
                f"translate: model returned no text for line id={line_id!r} "
                f"({matched} of {len(lines)} lines came back)"
            )
        new_line = dict(line)
        new_line["source_text"] = line.get("text", "")
        new_line["text"] = text
        translated.append(new_line)

    extra = set(by_id) - set(wanted)
    if extra:
        raise LocalisationError(
            f"translate: model invented line ids not in the request: "
            f"{sorted(extra)}"
        )

    return translated


def translate_lines(
    lines: list[dict],
    *,
    source_language: str,
    target_language: str,
    model: str | None = None,
    video_summary: str = "",
    scene_context: str = "",
    context_lines: list[dict] | None = None,
    on_screen_text: list[dict] | None = None,
    frames: list[tuple[float, bytes]] | None = None,
    cuts: list[float] | None = None,
    hook_end: float | None = None,
    spend_scope: str | None = None,
    retry_budget_sec: float | None = None,
    shorten_admit_sec: float | None = None,
    shorten_timeout_sec: float | None = None,
) -> list[dict]:
    """
    Translate transcript *lines* into *target_language*, 1:1 and in order.

    *retry_budget_sec*, *shorten_admit_sec* and *shorten_timeout_sec* override
    :data:`_TRANSLATE_RETRY_BUDGET_SEC`, :data:`_SHORTEN_ADMIT_SEC` and
    :data:`_SHORTEN_TIMEOUT_SEC` for this call only (None = those defaults).
    The background translate job (AICP-338, no nginx budget) passes larger
    ones so a whole-video call of a few minutes still gets the 1:1 re-ask and
    the shorten pass.

    Each returned line is a copy of the input with ``text`` replaced by the
    translation and ``source_text`` carrying the original, so the operator can
    diff them in the UI and nothing about the source is lost.

    The line's **duration** — not its length — is what the model is asked to
    match, because the result is spoken aloud over fixed footage. Everything
    else is context for adapting rather than translating: *video_summary* /
    *scene_context* and *on_screen_text* from the §4.1 transcript, the whole
    transcript as *context_lines*, and *frames* — JPEG stills of the hook — so
    the model can see who speaks and what "this" is. All of it is optional, so
    old cached transcripts without those fields keep working.

    Args:
        lines: §4.1 line dicts, normally the output of :func:`slice_lines`.
            An empty list short-circuits and costs nothing.
        source_language: Detected source code, e.g. ``"en"``.
        target_language: Target code from :mod:`app.languages`, e.g. ``"ja"``;
            also picks the target market (:data:`_TARGET_MARKETS`).
        model: Override ``settings.LOCALISATION_TRANSLATE_MODEL``. Sent in the
            request **body** (this is the Responses route, not the chat route).
            If it is unavailable, each of
            ``settings.LOCALISATION_TRANSLATE_FALLBACK_MODELS`` is tried in turn.
        video_summary: High-level narrative/purpose of the source video.
        scene_context: Situational evidence for resolving ambiguous dialogue.
        context_lines: The full transcript; defaults to *lines*.
        on_screen_text: §4.1 burned-in text entries.
        frames: ``(timestamp_sec, jpeg_bytes)`` stills from the hook.
        cuts: Seams between swap clips inside the hook; with *hook_end* they
            shorten each line's :func:`speaking_window`. A line whose answer
            still overruns its window is re-asked once (the shorten pass,
            :data:`_SHORTEN_PROMPT`) and the shorter wording kept.

    Returns:
        A new list, same length and order as *lines*, ids preserved.

    Raises:
        LocalisationError: On API failure, an unparseable response, ids that
            are unusable as a rejoin key *before* any call is made, or any
            break in the 1:1 correspondence (a merged, dropped, duplicated or
            invented line) — a silently shortened script would desync the whole
            hook, so it fails loudly instead. A broken correspondence is
            re-asked once (see :data:`_TRANSLATE_ATTEMPTS`) before it is
            raised; API failures are not, because the client underneath already
            retries the transient ones.
    """
    if not lines:
        return []
    retry_budget = (
        _TRANSLATE_RETRY_BUDGET_SEC if retry_budget_sec is None else float(retry_budget_sec)
    )
    shorten_admit = (
        _SHORTEN_ADMIT_SEC if shorten_admit_sec is None else float(shorten_admit_sec)
    )

    # The ids are the rejoin key, so they have to be trustworthy before we pay
    # for a call: two lines sharing an id can only come back as one entry, and
    # both would then be handed the same translation with every check passing.
    # This is the caller's transcript, not the model's answer, so it is checked
    # up front and never retried.
    ids = [line.get("id") for line in lines]
    unusable = [i for i in ids if not isinstance(i, int) or isinstance(i, bool)]
    if unusable:
        raise LocalisationError(
            f"translate: {len(unusable)} line(s) have no usable integer id "
            f"({unusable[:3]}) — ids are the rejoin key for the translation"
        )
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise LocalisationError(
            f"translate: the lines to translate contain duplicate ids "
            f"{repeated} — the translation could not be matched back 1:1"
        )

    context = list(context_lines) if context_lines else list(lines)
    prompt = TRANSLATE_PROMPT.format(
        source_language=_language_name(source_language),
        target_language=_language_name(target_language),
        market=_target_market(target_language),
        lines_block=_lines_block(
            lines, context, target_language=target_language,
            cuts=cuts or (), hook_end=hook_end,
        ),
        full_transcript=_full_transcript_block(context, set(ids)),
        on_screen_text=_on_screen_block(on_screen_text),
        video_summary=_normalise_context_text(video_summary) or "(none provided)",
        scene_context=_normalise_context_text(scene_context) or "(none provided)",
    )
    images = [(f"Frame at {t:.1f}s:", jpeg) for t, jpeg in frames or []]

    # The primary model, then the fallbacks — each tried only when the one
    # before is *unavailable* (same contract as transcribe_video), so a bad key
    # or a broken answer still fails fast instead of being billed again.
    models = [model or settings.LOCALISATION_TRANSLATE_MODEL]
    for fallback in settings.LOCALISATION_TRANSLATE_FALLBACK_MODELS.split(","):
        fallback = fallback.strip()
        if fallback and fallback not in models:
            models.append(fallback)

    resolved_model = models[0]
    unavailable: list[str] = []
    attempt = 1
    begun = time.monotonic()
    while True:
        started = time.monotonic()
        try:
            raw = get_shared_client().create_response(
                model=resolved_model,
                input_text=prompt,
                images=images,
                reasoning_effort=settings.LOCALISATION_TRANSLATE_EFFORT,
                timeout_sec=settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC,
            )
        except KieModelUnavailableError as exc:
            unavailable.append(str(exc))
            position = models.index(resolved_model)
            # Same nginx budget as the 1:1 re-ask: a fallback stacked on a
            # model that took minutes to give up would end in an opaque 504.
            if (
                position + 1 < len(models)
                and time.monotonic() - started <= retry_budget
            ):
                logger.warning(
                    "translate: %s unavailable (%s); falling back to %s",
                    resolved_model, exc, models[position + 1],
                )
                resolved_model = models[position + 1]
                continue
            raise LocalisationError(
                "translate: " + "; then ".join(unavailable)
            ) from exc
        except KieError as exc:
            raise LocalisationError(f"translate: {exc}") from exc
        # Billed even when the answer then fails the 1:1 check (AICP-155).
        spend.record_draw(spend_scope, spend.llm_credits(), kind="translate")

        try:
            translated = _rejoin_translations(raw, lines)
        except LocalisationError as exc:
            elapsed = time.monotonic() - started
            if attempt >= _TRANSLATE_ATTEMPTS or elapsed > retry_budget:
                if attempt > 1:
                    raise LocalisationError(
                        f"{exc} [still wrong after {attempt} attempts]"
                    ) from exc
                raise
            attempt += 1
            logger.warning(
                "Translation answer broke the 1:1 contract (%s); re-asking "
                "once (attempt %d/%d, model=%s)",
                exc, attempt, _TRANSLATE_ATTEMPTS, resolved_model,
            )
            continue

        logger.info(
            "Translated %d lines %s -> %s (model=%s, attempt=%d, frames=%d)",
            len(translated), source_language, target_language,
            resolved_model, attempt, len(images),
        )
        if time.monotonic() - begun <= shorten_admit:
            translated = _shorten_overruns(
                translated,
                prompt=prompt,
                raw=raw,
                images=images,
                model=resolved_model,
                target_language=target_language,
                cuts=cuts or (),
                hook_end=hook_end,
                spend_scope=spend_scope,
                timeout_sec=shorten_timeout_sec,
            )
        return translated


def overrunning_lines(
    lines: list[dict],
    target_language: str | None,
    *,
    cuts=(),
    hook_end: float | None = None,
) -> list[tuple[dict, float, float]]:
    """``(line, estimate_sec, window_sec)`` for each translated line that is
    estimated to run past :data:`SPEECH_OVERRUN_RATIO` times its window."""
    over = []
    for line in lines:
        start, end = speaking_window(line, cuts=cuts, hook_end=hook_end)
        window = end - start
        if window <= 0:
            continue
        estimate = estimated_speech_sec(line.get("text", ""), target_language)
        if estimate > window * SPEECH_OVERRUN_RATIO:
            over.append((line, estimate, window))
    return over


# The shorten pass runs only when the translation itself came back within
# _SHORTEN_ADMIT_SEC, and with its own short per-attempt timeout: KieClient
# retries a transient failure up to 4 times, so 60 + 4 x 45s (+ backoff) still
# lands inside nginx's 300s proxy_read_timeout on this synchronous endpoint.
_SHORTEN_ADMIT_SEC = 60.0
_SHORTEN_TIMEOUT_SEC = 45.0

# Words a shorter rewrite may not lose: tokens that appear verbatim in BOTH
# the source line and the first translation — which is what rule 4 keeps
# untranslated — when they look like a number, a name/brand (any capital, so
# "iPhone" and "eBay" count) or a URL/domain (a dot or slash: "x.co"). Ordinary
# words never match: they differ between the two languages. Over-protecting
# only costs a refused rewrite; under-protecting loses a brand.
_TOKEN_RE = re.compile(r"\w(?:[\w.\-/:]*\w)?", re.UNICODE)


def _protected_tokens(source_text: str, translated_text: str) -> set[str]:
    source = set(_TOKEN_RE.findall(source_text or ""))
    kept = set(_TOKEN_RE.findall(translated_text or ""))
    return {
        token for token in source & kept
        if token.isdigit()
        or any(c in token for c in "./")
        or (len(token) >= 3 and any(c.isupper() for c in token))
    }


_SHORTEN_PROMPT = """\
{prompt}

YOUR PREVIOUS ANSWER:
{previous}

These lines are too long to be spoken in their window — they will overrun \
the shot:

{too_long}

Rewrite ONLY these lines so each fits its character max: shorter, still \
natural spoken {target_language} with the same intent. Never drop or replace \
a person's name, a brand or a number to make a line fit — if a line cannot \
get shorter without that, return it exactly as it was. Keep every other line \
exactly as you had it. Reply with the FULL JSON object again, every id, \
same order — no commentary, no markdown code fence.
"""


def _shorten_overruns(
    translated: list[dict],
    *,
    prompt: str,
    raw: str,
    images,
    model: str,
    target_language: str,
    cuts,
    hook_end: float | None,
    spend_scope: str | None,
    timeout_sec: float | None = None,
) -> list[dict]:
    """One extra call that asks the translator to shorten its overrunning lines.

    Best effort, never a failure: any API or parse problem keeps the first
    answer, and a rewrite is only taken for a line it actually made shorter —
    other lines always keep their first wording, so this pass cannot quietly
    rewrite a line that already fit.
    """
    over = overrunning_lines(
        translated, target_language, cuts=cuts, hook_end=hook_end
    )
    if not over:
        return translated

    too_long = "\n".join(
        f"[{line.get('id')}] {window:.1f}s window, max "
        f"{_char_budget(window, target_language)[1]} characters — yours has "
        f"{len(str(line.get('text') or '').strip())}: {line.get('text', '')}"
        for line, _estimate, window in over
    )
    try:
        retry_raw = get_shared_client().create_response(
            model=model,
            input_text=_SHORTEN_PROMPT.format(
                prompt=prompt,
                previous=raw.strip(),
                too_long=too_long,
                target_language=_language_name(target_language),
            ),
            images=images,
            reasoning_effort=settings.LOCALISATION_TRANSLATE_EFFORT,
            timeout_sec=_SHORTEN_TIMEOUT_SEC if timeout_sec is None else float(timeout_sec),
        )
    except KieError as exc:
        logger.warning("translate: shorten pass failed (%s) — keeping the first answer", exc)
        return translated
    spend.record_draw(spend_scope, spend.llm_credits(), kind="translate")
    try:
        rewritten = {line["id"]: line for line in _rejoin_translations(retry_raw, translated)}
    except LocalisationError as exc:
        logger.warning("translate: shorten pass unusable (%s) — keeping the first answer", exc)
        return translated

    result = []
    shortened = 0
    for line in translated:
        candidate = rewritten.get(line.get("id"))
        first_text = str(line.get("text") or "")
        if (
            any(line is item for item, _e, _w in over)
            and candidate
            and len(candidate["text"].strip()) < len(first_text.strip())
            and _protected_tokens(line.get("source_text", ""), first_text)
            <= set(_TOKEN_RE.findall(candidate["text"]))
        ):
            kept = dict(line)
            kept["text"] = candidate["text"]
            result.append(kept)
            shortened += 1
        else:
            result.append(line)
    logger.info(
        "translate: shorten pass rewrote %d of %d overrunning line(s)",
        shortened, len(over),
    )
    return result


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def slice_lines(lines: list[dict], start: float, end: float) -> list[dict]:
    """
    Return the lines overlapping the half-open window ``[start, end)``.

    Used to cut the hook out of a full-source transcript. Overlap, not
    containment: a line that begins inside the hook and finishes after it is
    part of the hook — the hook is what Seedance re-speaks, and dropping a
    half-covered line would silently lose speech.

    Boundaries: a line ending exactly at *start* is out, a line beginning
    exactly at *end* is out, and a zero-length line counts as in when its
    timestamp falls in ``[start, end)``.
    """
    window: list[dict] = []
    for line in lines or []:
        line_start = _as_float(line.get("start"))
        line_end = _as_float(line.get("end"), line_start)
        if line_end <= line_start:
            hit = start <= line_start < end
        else:
            hit = line_end > start and line_start < end
        if hit:
            window.append(line)
    return window


def format_dialogue(lines: list[dict]) -> str:
    """
    Render lines as the ``**Speaker:**     text`` blocks Seedance was proven on.

    Byte-identical in shape to the hand-written prompt of run ``1736426f``
    (five spaces after the bold label, one block per line, newline-separated) —
    that run is the only evidence this feature works, so its formatting is
    treated as part of the contract rather than as taste.

    Consecutive lines from the same speaker are merged into one block, exactly
    as the proven prompt has them: the model reads a block as one continuous
    delivery, and re-labelling every clause makes it pause between them.
    """
    blocks: list[tuple[str, list[str]]] = []
    for line in lines or []:
        speaker = str(line.get("speaker") or "").strip() or "Speaker"
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        if blocks and blocks[-1][0] == speaker:
            blocks[-1][1].append(text)
        else:
            blocks.append((speaker, [text]))
    return "\n".join(
        f"**{speaker}:**{_SPEAKER_SEP}{' '.join(texts)}" for speaker, texts in blocks
    )


# ---------------------------------------------------------------------------
# Timed dialogue (AICP-335)
# ---------------------------------------------------------------------------
#
# The untimed "**Speaker:** text" block above was what Seedance got until
# 2026-10-29. On project 46cc6443 (es, 7 segments, Seedance 2.5) it produced a
# male voice for the off-screen interviewer in 3 of 6 clips, lines in the wrong
# order in 2, both lines given to the on-screen woman in one, and a line spoken
# over a phone close-up given to the interviewer — 5,358 credits of re-takes.
# A hand-written format fixed every one of those on the first take, and was
# then generated by a scratch script for pt/pl/fr with no re-takes at all:
#
#   Speakers (...):            who exists, their gender and voice, how many
#                              lines a secondary speaker has in THIS clip;
#   Dialogue — strictly ...:   each line timecoded relative to the clip start;
#   off-face cue:              who speaks when their face is out of frame;
#   silent tail:               an explicit "nobody speaks" after the last line;
#   pronunciation / action:    brand names and body actions the clip must keep.
#
# format_timed_dialogue is that script, generalised. Every wording below is
# the one that worked; change it only with a re-take in hand.

# Lead-in sentences, verbatim from the prompts that worked.
_SPEAKERS_HEADER = "Speakers:"
_DIALOGUE_HEADER = (
    "Dialogue — spoken strictly in this order, at these times (seconds from the "
    "start of this clip), nothing before or after:"
)
_SILENCE = "Nobody speaks. Mouths stay closed."
# A gap after the last line shorter than this is the speaker's breath, not a
# silence worth stating (and stating it would ask the model to stop early).
_SILENT_TAIL_MIN_SEC = 0.4
# Two lines whose union leaves less than this uncovered still "cover" a third.
_COVER_EPS = 0.05

# Brand names whose pronunciation has been proven on Seedance (the full
# sentence is emitted verbatim). Keyed casefolded. Any other brand the
# transcript lists gets the generic sentence built in _brand_hint.
_BRAND_HINTS: dict[str, str] = {
    "listenify": (
        'Pronunciation: "Listenify" is an English brand name — say it exactly as '
        'in English, "LISTEN-ih-fy", three syllables ending on "fy". Never add an '
        'ending such as "-ed".'
    ),
}

# Fallbacks for a transcript with no v3 "speakers" entry for a label (v1/v2,
# or a label an operator renamed): a role word from the label, then a gender
# from it. Word-level matching, so "woman" never matches "man".
_ROLE_WORDS = (
    "interviewer", "narrator", "host", "presenter", "reporter", "customer",
    "woman", "man", "girl", "boy", "lady", "guy", "child", "kid", "teenager",
)
# In priority order: "voice from the smartphone app" is the APP.
_DEVICE_WORDS = {
    "app": "APP", "tv": "TV", "television": "TV", "radio": "RADIO",
    "phone": "PHONE", "smartphone": "PHONE", "device": "DEVICE",
    "laptop": "DEVICE", "computer": "DEVICE", "tablet": "DEVICE",
}
_FEMALE_WORDS = {"woman", "girl", "lady", "female", "mother", "mom", "she", "her"}
_MALE_WORDS = {"man", "boy", "guy", "male", "father", "dad", "he", "his"}
_WORD_RE = re.compile(r"[a-z]+")


def _label_key(label) -> str:
    return " ".join(str(label or "").split()).casefold()


def _speaker_key(line: dict) -> str:
    """The profile key of a line's speaker; a blank label is "speaker"."""
    return _label_key(line.get("speaker")) or "speaker"


def _seconds(value) -> float:
    number = _as_float(value)
    return number if math.isfinite(number) else 0.0


def _fallback_profile(label: str) -> dict:
    words = _WORD_RE.findall(label.casefold())
    device = next((tag for word, tag in _DEVICE_WORDS.items() if word in words), "")
    if device and "voice" in words:
        kind, tag = "device", device
    else:
        kind = "person"
        tag = next((w.upper() for w in _ROLE_WORDS if w in words), "SPEAKER")
    if any(w in _FEMALE_WORDS for w in words):
        gender = "female"
    elif any(w in _MALE_WORDS for w in words):
        gender = "male"
    else:
        gender = "unknown"
    return {
        "label": label, "tag": tag, "kind": kind, "gender": gender,
        "voice": "", "description": label,
    }


def _speaker_profiles(all_lines: list[dict], transcript: dict | None) -> dict[str, dict]:
    """label key -> speaker profile, for every speaker in *all_lines*.

    A profile is the v3 ``speakers`` entry for the label when the transcript
    has one, else a fallback guessed from the label (:func:`_fallback_profile`).
    Derived on top: ``order`` (first appearance), ``on_camera`` (seen on screen
    at least once), ``primary`` (the person with the most speech time) and a
    tag made unique among PERSONS (two devices may share "APP"; two people
    may not share "WOMAN", or the model cannot tell them apart).
    """
    declared = {}
    for entry in clean_speakers((transcript or {}).get("speakers")):
        declared.setdefault(_label_key(entry["label"]), entry)

    profiles: dict[str, dict] = {}
    speech: Counter[str] = Counter()
    for line in all_lines:
        label = " ".join(str(line.get("speaker") or "").split()) or "Speaker"
        key = _speaker_key(line)
        if key not in profiles:
            base = dict(declared.get(key) or _fallback_profile(label))
            if not base.get("tag"):
                base["tag"] = _fallback_profile(label)["tag"]
            base.update(key=key, order=len(profiles), on_camera=False, primary=False)
            profiles[key] = base
        speech[key] += max(0.0, _seconds(line.get("end")) - _seconds(line.get("start")))
    # "On camera somewhere in the VIDEO", not just in the lines being
    # rendered: a hook can open on a phone close-up while the speaker's face
    # only appears later, and that line still needs its off-face cue.
    transcript_lines = [
        ln for ln in (transcript or {}).get("lines") or [] if isinstance(ln, dict)
    ]
    for line in list(all_lines) + transcript_lines:
        key = _speaker_key(line)
        if key in profiles and (
            line.get("face_visible") is True
            or (not isinstance(line.get("face_visible"), bool) and line.get("on_screen"))
        ):
            profiles[key]["on_camera"] = True

    persons = [p for p in profiles.values() if p["kind"] == "person"]
    if persons:
        top = max(persons, key=lambda p: (speech[p["key"]], -p["order"]))
        top["primary"] = True
    # Unique among persons — including against a declared "WOMAN 2".
    taken: set[str] = set()
    for profile in sorted(persons, key=lambda p: p["order"]):
        tag, n = profile["tag"], 1
        while tag in taken:
            n += 1
            tag = f"{profile['tag']} {n}"
        profile["tag"] = tag
        taken.add(tag)
    return profiles


def _with_line_meta(lines: list[dict], transcript: dict | None) -> list[dict]:
    """Copies of *lines* carrying the transcript's per-line v3 fields.

    Translated lines are copies of transcript lines, so they normally carry
    ``face_visible``/``frame`` already; the id lookup covers a caller that
    rebuilt them. A field already on the line always wins.
    """
    by_id = {}
    for line in (transcript or {}).get("lines") or []:
        if isinstance(line, dict) and line.get("id") is not None:
            by_id.setdefault(line.get("id"), line)
    out = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        merged = dict(line)
        meta = by_id.get(line.get("id"))
        if meta is not None and _label_key(meta.get("speaker")) == _label_key(line.get("speaker")):
            for key, value in clean_line_visibility(meta).items():
                merged.setdefault(key, value)
        for key in ("face_visible", "frame"):
            if key in merged and key not in clean_line_visibility(merged):
                merged.pop(key)
        out.append(merged)
    return out


def background_lines(lines: list[dict], transcript: dict | None = None) -> list[dict]:
    """Device lines played UNDER someone else's speech — kept, marked background.

    A line from a ``device`` speaker (the app reading a book aloud, a TV)
    whose whole span is covered by OTHER speakers' lines. In 46cc6443 the app
    read 8.7 s of a book (20.5–29.2) under the woman's own lines; Seedance
    cannot voice two speakers at once, and the operator dropped it from every
    language. Such a line is KEPT in the prompt and marked BACKGROUND (it
    keeps its own span and is not clamped into the strict order), and the
    localisation-prompt endpoint names it in a warning, so the operator can
    delete it by hand. A person's interjection is never treated this way —
    only a device, and only when it is fully covered.
    """
    lines = _with_line_meta(
        [ln for ln in lines or [] if isinstance(ln, dict)], transcript
    )
    profiles = _speaker_profiles(lines, transcript)
    background: list[dict] = []
    for line in lines:
        key = _speaker_key(line)
        if profiles[key]["kind"] != "device":
            continue
        start, end = _seconds(line.get("start")), _seconds(line.get("end"))
        if end - start <= _COVER_EPS:
            continue
        others = sorted(
            (_seconds(o.get("start")), _seconds(o.get("end")))
            for o in lines
            if _speaker_key(o) != key
        )
        cursor = start
        for o_start, o_end in others:
            if o_start > cursor + _COVER_EPS:
                break
            cursor = max(cursor, o_end)
            if cursor >= end - _COVER_EPS:
                break
        if cursor >= end - _COVER_EPS:
            background.append(line)
    return background


REFERENCE_AUDIO_HEADER_RE = re.compile(
    r"^Speakers \(the reference audio contains [^)\n]*\):$", re.M
)


def reference_audio_header(transcript: dict | None = None, lines: list[dict] | None = None) -> str:
    """The Speakers header that says the run's voice reference holds the voices.

    "both human voices" for two people (the wording that worked on 46cc6443),
    "this human voice" for one, "all N human voices" for more — counted over
    the transcript's lines plus *lines*, so the New Run form's toggle
    (:func:`set_reference_audio_header`) and :func:`format_timed_dialogue`
    produce the same text.
    """
    pool = [ln for ln in (transcript or {}).get("lines") or [] if isinstance(ln, dict)]
    pool += [ln for ln in lines or [] if isinstance(ln, dict)]
    humans = sum(
        1 for p in _speaker_profiles(pool, transcript).values() if p["kind"] == "person"
    ) if pool else 0
    if humans <= 1:
        noun = "this human voice"
    elif humans == 2:
        noun = "both human voices"
    else:
        noun = f"all {humans} human voices"
    return f"Speakers (the reference audio contains {noun}):"


def set_reference_audio_header(prompt: str, header: str | None) -> str:
    """Switch a timed prompt's Speakers header on (*header*) or off (None).

    The server-side twin of the New Run form's "voice reference" toggle,
    which does the same rewrite in JS. A prompt without a timed Speakers
    block (hand-written, or the untimed format) is returned unchanged.
    """
    target = header or _SPEAKERS_HEADER
    if REFERENCE_AUDIO_HEADER_RE.search(prompt):
        return REFERENCE_AUDIO_HEADER_RE.sub(lambda _: target, prompt, count=1)
    return re.sub(r"^Speakers:$", lambda _: target, prompt, count=1, flags=re.M)


def resolve_prompt_model(project_type: str | None, model: str | None = None):
    """The :class:`~app.ai_models.AIModelSpec` a localisation prompt is timed for.

    *model* is the optional ``model`` field of ``localisation-prompt`` — the
    model picked on the New Run form. Without it, the model that form
    pre-fills for *project_type* (``ProjectTypeSpec.default_model``), which is
    the pre-AICP-335-follow-up behaviour. Its ``max_clip_sec`` drives the roll
    trim of :func:`clip_window`. Shared with the AICP-338 background job.

    Raises:
        ValueError: *model* is given but is not a known model key.
    """
    if model:
        if model not in ai_models.V2V_MODELS:
            raise ValueError(
                f"unknown model {model!r} — must be one of {sorted(ai_models.V2V_MODELS)}"
            )
        return ai_models.V2V_MODELS[model]
    return ai_models.spec_for(project_types.spec_for(project_type).default_model)


def clip_window(
    start_sec: float,
    end_sec: float,
    pre_roll_sec: float = 0.0,
    post_roll_sec: float = 0.0,
    duration_sec: float | None = None,
    max_clip_sec: float | None = None,
) -> tuple[float, float]:
    """The source span a swap segment's generated clip covers.

    Mirrors the first step of ``pipeline_v2._submit_swap_segment_isolated``
    (``clip_start = start - pre_roll``, ``clip_end = end + post_roll``, clamped
    to the source, then the roll trim to *max_clip_sec*), so a timecode
    "seconds from the start of this clip" means the clip Seedance actually
    receives. The model-specific padding a too-short clip gets at submit time
    is NOT mirrored — the model is chosen later, on
    the New Run form — and padding forward does not move any timecode anyway;
    only a sub-minimum segment at the very end of the source is padded
    backwards.
    """
    start, end = float(start_sec), float(end_sec)
    clip_start = max(0.0, start - max(0.0, float(pre_roll_sec or 0.0)))
    clip_end = end + max(0.0, float(post_roll_sec or 0.0))
    if duration_sec is not None and duration_sec > 0:
        clip_end = min(float(duration_sec), clip_end)
    # The roll trim of the submit path, same order: when the rolls push the
    # clip past the model's ceiling, post-roll goes first, then pre-roll,
    # never the segment itself. Without it a 15 s segment with 2 s pre-roll
    # on a 15 s model would be timecoded from 2 s before what the model gets.
    if max_clip_sec and clip_end - clip_start > max_clip_sec:
        clip_end = max(end, clip_start + max_clip_sec)
        if duration_sec is not None and duration_sec > 0:
            clip_end = min(float(duration_sec), clip_end)
        if clip_end - clip_start > max_clip_sec:
            clip_start = min(start, clip_end - max_clip_sec)
    return clip_start, max(clip_start, clip_end)


def _pronouns(gender: str) -> tuple[str, str]:
    """(subject, possessive) — "she"/"her", "he"/"his", "they"/"their"."""
    return {"female": ("she", "her"), "male": ("he", "his")}.get(gender, ("they", "their"))


def _speaker_entry(profile: dict, spoken: int) -> str:
    description = (profile.get("description") or profile["label"]).strip().rstrip(".")
    text = f"- {profile['tag']}: {description}."
    gender = profile.get("gender")
    voice = re.sub(r"\s*\bvoice\.?$", "", (profile.get("voice") or "").strip().rstrip("."), flags=re.I)
    if gender in _GENDERS:
        noun = "Female" if gender == "female" else "Male"
        other = "male" if gender == "female" else "female"
        if profile["kind"] == "person":
            text += f" {noun}, {voice} voice." if voice else f" {noun} voice."
            text += f" Never a {other} voice."
        elif not re.search(rf"\b{gender}\b", description.casefold()):
            text += f" {noun} voice."
    elif voice:
        text += f" {voice[:1].upper()}{voice[1:]} voice."
    if profile["kind"] == "person" and not profile["primary"]:
        subject, _ = _pronouns(gender)
        plural = subject == "they"
        if spoken == 1:
            text += f" {subject.capitalize()} {'say' if plural else 'says'} ONLY ONE line in this clip."
        elif spoken == 0:
            text += f" {subject.capitalize()} {'do' if plural else 'does'} not speak in this clip."
    return text


def _face_visible(line: dict) -> bool | None:
    value = line.get("face_visible")
    return value if isinstance(value, bool) else None


def _off_face_cue(line: dict, profile: dict, earlier: list[dict], others: list[str]) -> str:
    """The parenthetical for a line spoken while its speaker's face is out of frame.

    Only for a PERSON who is on camera elsewhere: an off-screen speaker's face
    is never visible, and the Speakers block already says so. v3 transcripts
    say it per line (``face_visible``/``frame``); for older ones a line marked
    ``on_screen: false`` from an otherwise on-screen speaker is the signal.
    """
    if profile["kind"] != "person" or not profile["on_camera"]:
        return ""
    visible = _face_visible(line)
    if visible is None:
        visible = bool(line.get("on_screen"))
    if visible:
        return ""
    _, possessive = _pronouns(profile.get("gender", "unknown"))
    seen_before = any(
        (_face_visible(e) is True) or (_face_visible(e) is None and e.get("on_screen"))
        for e in earlier
    )
    frame = (line.get("frame") or "").strip().rstrip(".")
    cue = f"{possessive} face is {'no longer' if seen_before else 'not'} visible"
    if frame:
        cue += f" — {frame}"
    cue += f"; it is still {possessive.upper()} voice, the {profile['tag'].lower()}"
    if others:
        cue += ", NOT the " + " or the ".join(o.lower() for o in others)
    return f" ({cue})"


def _brand_hint(name: str, pronunciation: str) -> str:
    known = _BRAND_HINTS.get(name.casefold())
    if known:
        return known
    text = f'Pronunciation: "{name}" is a brand name — never translate it; say it exactly as in the original video'
    if pronunciation:
        text += f', "{pronunciation}"'
    return text + ". Never add an ending to it."


def _mentions(name: str, lines: list[dict]) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.I)
    return any(
        pattern.search(str(line.get(key) or ""))
        for line in lines
        for key in ("text", "source_text")
    )


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}"


def format_timed_dialogue(
    lines: list[dict],
    *,
    window: tuple[float, float],
    transcript: dict | None = None,
    all_lines: list[dict] | None = None,
    reference_audio: bool = False,
) -> str:
    """
    Render one clip's dialogue in the timed AICP-335 format (see the comment
    block above). Port of the scratch builder that produced the pt/pl/fr
    prompts of project 46cc6443 with no re-takes.

    Args:
        lines: The lines THIS clip says (the midpoint partition of
            ``api_v2._assign_lines_to_segments``), translated.
        window: ``(clip_start, clip_end)`` in source seconds — normally
            :func:`clip_window` of the swap segment. Timecodes are relative to
            ``clip_start`` and clamped to the clip and to the next line.
        transcript: The §4.1 transcript, for the v3 ``speakers``, ``actions``,
            ``brands`` and per-line ``face_visible``/``frame``. Everything is
            optional: without it gender clauses and off-face frame text are
            simply absent.
        all_lines: Every line of the video (defaults to the transcript's). It
            decides who the speakers are — every PERSON is listed in every
            clip, with a count of their lines in this one — which one is
            primary, and which device lines are background.
        reference_audio: The run will carry a voice reference containing the
            human voices, so the Speakers header says so.
    """
    clip_start, clip_end = float(window[0]), float(window[1])
    duration = max(0.0, clip_end - clip_start)
    if all_lines is None:
        all_lines = (transcript or {}).get("lines") or lines
    everything = _with_line_meta(
        [ln for ln in all_lines or [] if isinstance(ln, dict)], transcript
    )
    mine = _with_line_meta(lines, transcript)
    # The video's lines plus any of this clip's that the caller's all_lines
    # lacks (matched by id), so a clip can never name a speaker the Speakers
    # block does not list.
    known_ids = {line.get("id") for line in everything}
    pool = everything + [line for line in mine if line.get("id") not in known_ids]
    known_speakers = {_speaker_key(line) for line in pool}
    pool += [line for line in mine if _speaker_key(line) not in known_speakers]
    profiles = _speaker_profiles(pool, transcript)

    def identity(line: dict) -> tuple:
        return (
            line.get("id"),
            _speaker_key(line),
            _seconds(line.get("start")),
        )

    background = {identity(line) for line in background_lines(pool, transcript)}
    spoken = [line for line in mine if str(line.get("text") or "").strip()]
    spoken.sort(key=lambda line: _seconds(line.get("start")))

    def profile_of(line: dict) -> dict:
        return profiles[_speaker_key(line)]

    def is_background(line: dict) -> bool:
        return identity(line) in background

    # Speakers: every person in the video (so "does not speak in this clip" can
    # be said), then the devices that actually speak here.
    counts: Counter[str] = Counter(profile_of(ln)["key"] for ln in spoken)
    persons = sorted(
        (p for p in profiles.values() if p["kind"] == "person"),
        key=lambda p: (not p["primary"], p["order"]),
    )
    devices = sorted(
        (p for p in profiles.values() if p["kind"] == "device" and counts[p["key"]]),
        key=lambda p: p["label"].casefold(),
    )
    header = (
        reference_audio_header(transcript, pool) if reference_audio else _SPEAKERS_HEADER
    )
    speakers = [header] + [_speaker_entry(p, counts[p["key"]]) for p in persons + devices]

    # Foreground lines are clamped to the NEXT foreground line's start, so the
    # strict order holds. A background line (a device playing UNDER other
    # speakers, see background_lines) is kept and marked instead: it keeps its
    # own span, overlaps on purpose, and never shortens the line it sits under.
    foreground = [line for line in spoken if not is_background(line)]
    dialogue = [_DIALOGUE_HEADER]
    cued: list[dict] = []
    last_end = 0.0
    for line in spoken:
        profile = profile_of(line)
        start_rel = _seconds(line.get("start")) - clip_start
        end_rel = _seconds(line.get("end")) - clip_start
        t0 = min(duration, max(0.0, start_rel))
        t1 = min(duration, max(t0, end_rel))
        if is_background(line):
            under = []
            for other in foreground:
                o_start = _seconds(other.get("start")) - clip_start
                o_end = _seconds(other.get("end")) - clip_start
                tag = profile_of(other)["tag"]
                if o_end > start_rel and o_start < end_rel and tag not in under:
                    under.append(tag)
            whose = " and the ".join(under) if under else "other speakers"
            cue = (
                f" (BACKGROUND — it plays quietly at the same time as the {whose}'s "
                "lines, underneath them; nobody stops talking for it and it is not "
                "lip-synced)"
            )
        else:
            index = foreground.index(line)
            if index + 1 < len(foreground):
                next_start = _seconds(foreground[index + 1].get("start")) - clip_start
                t1 = min(t1, max(t0, next_start))
            earlier = [e for e in foreground[:index] if profile_of(e) is profile]
            others = [p["tag"] for p in persons if p is not profile]
            cue = _off_face_cue(line, profile, earlier, others)
            if cue and profile not in cued:
                cued.append(profile)
        dialogue.append(
            f"[{_fmt(t0)}–{_fmt(t1)}] {profile['tag']}{cue}: {str(line.get('text')).strip()}"
        )
        last_end = max(last_end, min(duration, max(0.0, end_rel)))
    if not spoken:
        dialogue.append(f"[{_fmt(0.0)}–{_fmt(duration)}] {_SILENCE}")
    elif duration - last_end > _SILENT_TAIL_MIN_SEC:
        dialogue.append(f"[{_fmt(last_end)}–{_fmt(duration)}] {_SILENCE}")

    extra: list[str] = []
    for action in clean_actions((transcript or {}).get("actions")):
        a0 = action["start"] - clip_start
        a1 = action["end"] - clip_start
        if a1 < 0 or a0 >= duration:
            continue
        a0, a1 = max(0.0, a0), min(duration, max(a1, a0))
        actor = profiles.get(_label_key(action["speaker"]))
        who = f"the {actor['tag']}" if actor else (action["speaker"] or "the person")
        when = f"~{a0:.0f} s" if round(a0) == round(a1) else f"~{a0:.0f}–{a1:.0f} s"
        sentence = (
            f"Action — reproduce it exactly as in the reference video: at {when} "
            f"{who} {action['action']}."
        )
        if action["do_not"]:
            sentence += f" Do not {action['do_not']}."
        extra.append(sentence)
    brands = [(b["name"], b["pronunciation"]) for b in clean_brands((transcript or {}).get("brands"))]
    for name in _BRAND_HINTS:
        if name not in {b[0].casefold() for b in brands}:
            brands.append((name, ""))
    for name, pronunciation in brands:
        if _mentions(name, spoken):
            hint = _brand_hint(name, pronunciation)
            if hint not in extra:
                extra.append(hint)
    for profile in cued:
        _, possessive = _pronouns(profile.get("gender", "unknown"))
        extra.append(
            f"Every word the {profile['tag']} says is in {possessive} own voice, "
            f"also while {possessive} face is out of frame."
        )

    blocks = ["\n".join(speakers), "\n".join(dialogue)]
    if extra:
        blocks.append("\n".join(extra))
    return "\n\n".join(blocks)


def build_prompt(
    *,
    lines: list[dict],
    source_language: str,
    target_language: str,
    swap_character: bool,
    window: tuple[float, float] | None = None,
    transcript: dict | None = None,
    all_lines: list[dict] | None = None,
    reference_audio: bool = False,
) -> str:
    """
    Assemble the Seedance run prompt from a template plus the dialogue block.

    The two templates live in :mod:`app.project_types` beside the other default
    prompts, because they are prompts an operator may want to edit, not logic:

    * ``_LOCALISATION_SWAP_PROMPT`` — replace the person *and* re-speak the
      translated dialogue with adaptive lip sync (reference image expected).
    * ``_LOCALISATION_KEEP_PROMPT`` — re-speak only, preserving the person on
      screen while still adapting mouth/facial motion to the new language.

    Both carry ``{source_language}``, ``{target_language}`` and ``{dialogue}``
    slots. Languages are interpolated as lowercase English names ("english",
    "japanese"). The templates intentionally permit speech-related facial
    motion; output quality remains model-dependent.

    Args:
        lines: Translated lines (from :func:`translate_lines`) for the window
            this prompt covers.
        source_language: Detected source code, e.g. ``"en"``.
        target_language: Target code, e.g. ``"ja"``.
        swap_character: True for "speech + character", False for "speech only".
        window: ``(clip_start, clip_end)`` of the clip this prompt is for, in
            source seconds. When given, the dialogue is rendered in the timed
            AICP-335 format (:func:`format_timed_dialogue`: Speakers block with
            gender, timecoded lines, off-face cues, silent tail, brand and
            action hints). When omitted the untimed :func:`format_dialogue`
            block is used, byte-identical to every prompt before AICP-335.
        transcript, all_lines, reference_audio: passed to
            :func:`format_timed_dialogue`; ignored without *window*.

    Returns:
        The prompt text, ready to drop into the New Run form.

    Raises:
        LocalisationError: If the template is missing or has an unexpected slot.
    """
    name = "_LOCALISATION_SWAP_PROMPT" if swap_character else "_LOCALISATION_KEEP_PROMPT"
    template = getattr(project_types, name, None)
    if not template:
        raise LocalisationError(
            f"app.project_types.{name} is not defined — the localisation "
            f"Seedance templates are missing"
        )

    if window is None:
        dialogue = format_dialogue(lines)
    else:
        # The templates end "...speakers.\n{dialogue}"; the timed format is
        # three paragraphs, so it opens with the blank line that separates
        # them from the instructions — the layout of the prompts that worked.
        dialogue = "\n" + format_timed_dialogue(
            lines,
            window=window,
            transcript=transcript,
            all_lines=all_lines,
            reference_audio=reference_audio,
        )
    try:
        return template.format(
            source_language=_language_name(source_language),
            target_language=_language_name(target_language),
            dialogue=dialogue,
        )
    except (KeyError, IndexError) as exc:
        raise LocalisationError(
            f"app.project_types.{name} has an unexpected placeholder: {exc}"
        ) from exc
