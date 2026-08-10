"""
Localisation — transcribe a source creative, translate its hook, build the
Seedance prompt that re-speaks it.

This is the LLM layer of the ``localisation`` project type (design contract:
``docs/localisation.md``). It owns three steps and nothing else:

1. :func:`transcribe_video` — a video-capable model watches the source and
   returns the §4.1 transcript: detected source language, one entry per spoken
   line with timestamps and a **visually grounded** speaker label, plus any
   burned-in on-screen text kept separately.
2. :func:`translate_lines` — a text model rewrites those lines into the target
   language, matched to each line's *spoken duration* rather than its word
   count.
3. :func:`build_prompt` — the translated lines are formatted into the dialogue
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
import re
import time
from collections import Counter
from datetime import datetime, timezone

from app import languages, project_types
from app.config import settings
from app.kie_client import KieError, get_shared_client

logger = logging.getLogger(__name__)

# Bump when a prompt changes in a way that makes old output non-comparable.
# Stored in the transcript envelope so a cached transcript always records which
# instructions produced it.
TRANSCRIBE_PROMPT_VERSION = "transcribe/v1"
TRANSLATE_PROMPT_VERSION = "translate/v1"

# video_projects.transcript["schema_version"] — see docs/localisation.md §4.1.
TRANSCRIPT_SCHEMA_VERSION = 1

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
  "lines": [
    {"id": 1, "start": 0.0, "end": 2.4,
     "speaker": "<what the speaker looks like>",
     "on_screen": true,
     "text": "<verbatim speech>"}
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

2. SPEAKER LABELS. "speaker" is a short visual description that someone \
looking at a still frame of this video could use to point at the person: "the \
woman in the red jacket", "the man holding the phone", "off-screen \
interviewer (camera person)". NEVER use "Speaker 1", "Speaker A", "Person B", \
"narrator" or any other anonymous label. This text is handed to a video model \
that has to map the line onto a person it can see, so describe clothing, hair, \
position in frame, or what they are holding or doing — whatever separates them \
from the other people present. Use the SAME description every time the same \
person speaks.

3. HOW MANY SPEAKERS. Do not assume a number. There may be one, two or many. \
Some may never appear on camera (a voice-over, or the person holding the \
camera). Set "on_screen": true only when that speaker is visible in frame \
while the line is spoken, and false otherwise; for an off-camera voice, say so \
in the description as well.

4. VERBATIM. Transcribe exactly what is said, including filler words ("um", \
"like"), false starts, repetitions and interruptions. Do not fix grammar, do \
not summarise, do not merge sentences, do not translate. How long a line is is \
what makes the re-recording fit the footage, so "improving" the wording breaks \
it. Write the text in the spoken language's own native script.

5. TIMESTAMPS. "start" and "end" are seconds from the beginning of THIS clip, \
as decimal numbers (e.g. 4.2). One entry per continuous utterance by one \
speaker; a line must never cross a speaker change. List the lines in the order \
they are heard.

6. ON-SCREEN TEXT. Burned-in captions, subtitles, titles, price tags and \
overlay graphics belong in "on_screen_text" and ONLY there — never copy them \
into "lines". When the same words are both spoken and captioned, the spoken \
version goes in "lines" and the caption in "on_screen_text". If the video has \
no on-screen text at all, return an empty array.

7. NO SPEECH. If nobody speaks — music only, ambience only, silence — return \
"lines": [] and a best-effort "source_language" (or "" if there is nothing to \
go on). NEVER invent, guess or reconstruct dialogue you cannot actually hear. \
An empty list is a correct answer, a hallucinated line is not.
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
            "required": ["source_language", "lines", "on_screen_text"],
            "properties": {
                "source_language": {"type": "string"},
                "lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "id", "start", "end", "speaker", "on_screen", "text",
                        ],
                        "properties": {
                            "id": {"type": "integer"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "speaker": {"type": "string"},
                            "on_screen": {"type": "boolean"},
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
TRANSLATE_PROMPT = """\
You are localising the spoken script of a short advertising video from \
{source_language} into {target_language}. Your lines will be spoken aloud by an \
AI video model and lip-synced onto the original footage.

Here are the lines. Each one carries its id, who says it, and the DURATION IN \
SECONDS of the original delivery:

{lines_block}

Rules:

1. TIMING IS THE HARD CONSTRAINT. Each translated line must take about as long \
to SAY OUT LOUD in {target_language} as the original took — the duration in \
seconds given above, within roughly 15%. Say your line to yourself at a normal \
conversational pace and count the seconds. Do NOT match the word count or the \
character count: languages differ enormously in how much time the same meaning \
takes, and a literal translation usually overruns. If a draft is too long, cut \
it down; if it is too short, use a fuller phrasing or a natural filler. \
Overrunning is the worse failure — it makes the delivery race or run off the \
end of the shot.

2. REWRITE, DO NOT TRANSLATE. This is ad copy, not a document. Punch, rhythm \
and naturalness beat literal fidelity. Keep each line's intent and emotional \
beat; change the wording as freely as you need to hit the timing and to sound \
like something a native speaker would really say.

3. KEEP THE REGISTER. Match how casual or formal the original is. Offhand \
spoken street-interview {source_language} becomes offhand spoken \
street-interview {target_language} — contractions, slang, interjections and \
all. Do not turn speech into writing.

4. LEAVE THESE UNTRANSLATED, exactly as written: brand, product and app names \
(e.g. Speechify), people's names, numbers, prices, currencies, percentages, \
units and URLs. Translate everything around them.

5. NATIVE SCRIPT ONLY. Write {target_language} in its own writing system. \
Never romanise it (no romaji, no pinyin, no transliteration) and never leave \
the {source_language} text in place.

6. ONE LINE IN, ONE LINE OUT. Return exactly one entry per input line, with the \
SAME id, in the SAME order. Never merge two lines, never split one into two, \
never drop one, never add one. Do not translate the speaker labels — they are \
context for you, not output.

7. SPOKEN DIALOGUE ONLY. The video's burned-in captions, subtitles and overlay \
text are deliberately not in the list above: they stay in the original \
language. Do not translate them, do not reconstruct them, do not add them to \
your output.

Reply with ONLY this JSON object — no explanation, no commentary, no markdown \
code fence:

{{"lines": [{{"id": 1, "text": "<translated line>"}}]}}
"""


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


def _language_name(code: str | None) -> str:
    """Human name for a language code, lowercased for prose interpolation.

    Uses :mod:`app.languages` when the code is one of the five the pipeline
    delivers, and falls back to the raw code otherwise — the *detected* source
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
        cleaned.append(
            {
                "id": 0,  # replaced below
                "start": start,
                "end": max(end, start),
                "speaker": str(item.get("speaker") or "").strip() or "speaker",
                "on_screen": bool(item.get("on_screen")),
                "text": text,
            }
        )

    cleaned.sort(key=lambda ln: ln["start"])
    for index, line in enumerate(cleaned, start=1):
        line["id"] = index
    return cleaned


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


def transcribe_video(
    video_path: str,
    *,
    kie=None,
    model: str | None = None,
) -> dict:
    """
    Transcribe a source video into the §4.1 transcript dict.

    Uploads *video_path* to the kie.ai temp-file host, hands the public URL to a
    video-capable model in the ``image_url`` envelope (kie.ai uses that envelope
    for every media type, video included), and normalises the answer.

    A clip with no speech is a **success**, not a failure: it comes back with
    ``lines: []``, which the caller stores as ``transcript_status="empty"``.

    Args:
        video_path: Local path to the clip to transcribe. Transcribe the hook
            window rather than a long source when you can — the model is billed
            per input token and only the hook is ever re-spoken.
        kie: Optional :class:`~app.kie_client.KieClient`; defaults to the shared
            module-level client.
        model: Override ``settings.LOCALISATION_TRANSCRIBE_MODEL``. This value
            is a **URL path segment** on the kie.ai chat route, so it must be a
            real kie.ai model slug.

    Returns:
        The §4.1 dict: ``schema_version``, ``model``, ``prompt_version``,
        ``created_at``, ``source_language``, ``lines``, ``on_screen_text``.

    Raises:
        LocalisationError: On upload failure, API failure, or an unparseable
            response.
    """
    client = kie or get_shared_client()
    resolved_model = model or settings.LOCALISATION_TRANSCRIBE_MODEL

    try:
        video_url = client.upload_file(video_path, upload_path=_UPLOAD_PATH)
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

    try:
        raw = client.chat_completion(
            model=resolved_model,
            messages=messages,
            response_format=_TRANSCRIPT_JSON_SCHEMA,
            timeout_sec=settings.LOCALISATION_TIMEOUT_SEC,
        )
    except KieError as exc:
        raise LocalisationError(f"transcribe: {exc}") from exc

    data = _parse_json_object(raw, what="transcribe")
    lines = _clean_lines(data.get("lines"))
    transcript = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "model": resolved_model,
        "prompt_version": TRANSCRIBE_PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_language": _normalise_language(data.get("source_language")),
        "lines": lines,
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


def _lines_block(lines: list[dict]) -> str:
    """Render the lines for the translation prompt, one block per line.

    Each block carries the id (the rejoin key), the speaker (context for
    register and gender) and the **duration in seconds** — the number rule 1 of
    :data:`TRANSLATE_PROMPT` tells the model to budget against.
    """
    blocks = []
    for line in lines:
        duration = max(0.0, _as_float(line.get("end")) - _as_float(line.get("start")))
        speaker = str(line.get("speaker") or "").strip() or "speaker"
        blocks.append(
            f"[{line.get('id')}] speaker: {speaker} | duration: {duration:.1f}s\n"
            f"    {line.get('text', '')}"
        )
    return "\n\n".join(blocks)


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
) -> list[dict]:
    """
    Translate transcript *lines* into *target_language*, 1:1 and in order.

    Each returned line is a copy of the input with ``text`` replaced by the
    translation and ``source_text`` carrying the original, so the operator can
    diff them in the UI and nothing about the source is lost.

    The line's **duration** — not its length — is what the model is asked to
    match, because the result is spoken aloud over fixed footage.

    Args:
        lines: §4.1 line dicts, normally the output of :func:`slice_lines`.
            An empty list short-circuits and costs nothing.
        source_language: Detected source code, e.g. ``"en"``.
        target_language: Target code from :mod:`app.languages`, e.g. ``"ja"``.
        model: Override ``settings.LOCALISATION_TRANSLATE_MODEL``. Sent in the
            request **body** (this is the Responses route, not the chat route).

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

    resolved_model = model or settings.LOCALISATION_TRANSLATE_MODEL
    prompt = TRANSLATE_PROMPT.format(
        source_language=_language_name(source_language),
        target_language=_language_name(target_language),
        lines_block=_lines_block(lines),
    )

    for attempt in range(1, _TRANSLATE_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            raw = get_shared_client().create_response(
                model=resolved_model,
                input_text=prompt,
                reasoning_effort="medium",
                timeout_sec=settings.LOCALISATION_TIMEOUT_SEC,
            )
        except KieError as exc:
            raise LocalisationError(f"translate: {exc}") from exc

        try:
            translated = _rejoin_translations(raw, lines)
        except LocalisationError as exc:
            elapsed = time.monotonic() - started
            if attempt >= _TRANSLATE_ATTEMPTS or elapsed > _TRANSLATE_RETRY_BUDGET_SEC:
                if attempt > 1:
                    raise LocalisationError(
                        f"{exc} [still wrong after {attempt} attempts]"
                    ) from exc
                raise
            logger.warning(
                "Translation answer broke the 1:1 contract (%s); re-asking "
                "once (attempt %d/%d, model=%s)",
                exc, attempt + 1, _TRANSLATE_ATTEMPTS, resolved_model,
            )
            continue

        logger.info(
            "Translated %d lines %s -> %s (model=%s, attempt=%d)",
            len(translated), source_language, target_language,
            resolved_model, attempt,
        )
        return translated


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


def build_prompt(
    *,
    lines: list[dict],
    source_language: str,
    target_language: str,
    swap_character: bool,
) -> str:
    """
    Assemble the Seedance run prompt from a template plus the dialogue block.

    The two templates live in :mod:`app.project_types` beside the other default
    prompts, because they are prompts an operator may want to edit, not logic:

    * ``_LOCALISATION_SWAP_PROMPT`` — replace the person *and* the language
      (what run ``1736426f`` did; a reference image is expected).
    * ``_LOCALISATION_KEEP_PROMPT`` — change only the language, preserving the
      person on screen.

    Both carry ``{source_language}``, ``{target_language}`` and ``{dialogue}``
    slots. Languages are interpolated as lowercase English names ("english",
    "japanese"), matching the proven prompt's "(from english to japanese)".

    Args:
        lines: Translated lines (from :func:`translate_lines`) for the window
            this prompt covers.
        source_language: Detected source code, e.g. ``"en"``.
        target_language: Target code, e.g. ``"ja"``.
        swap_character: True for "speech + character", False for "speech only".

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

    try:
        return template.format(
            source_language=_language_name(source_language),
            target_language=_language_name(target_language),
            dialogue=format_dialogue(lines),
        )
    except (KeyError, IndexError) as exc:
        raise LocalisationError(
            f"app.project_types.{name} has an unexpected placeholder: {exc}"
        ) from exc
