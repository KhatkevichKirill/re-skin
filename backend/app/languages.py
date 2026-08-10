"""
Run languages — the spoken language of a Run's delivered creative.

A Run's ``language`` does exactly ONE thing: it tags the id we hand to the
tracking spreadsheet, so the downstream renaming pipeline can tell what
language the video is in. Nothing in this codebase branches on it — no prompt,
no model, no pipeline phase changes.

Downstream contract (repo ``ai-fb-audit``, ``wiki/rename-creatives-pipeline.md``
and ``wiki/multilingual-test-flow.md``)
--------------------------------------------------------------------------
``scripts/rename_creatives_from_final_creatives.py::parse_job_id`` reads the
language back OUT of the sheet's ``Job ID`` cell, strips the language affix to
get the "core", and builds the delivered Drive filename from both::

    TTS_New_AI_{Idea}_Mix_Broad_{LANG}__{pipeline_token}-{core}.{ext}

So ``1736426f-ja`` renames as ``…_Mix_Broad_JA__re-skin-v2-1736426f.mp4``.
Three properties of that parser drive every decision in this module:

1. **Five languages, and only five.** ``EN | ES | PT | JA | DE``. A code
   outside that set is not recognised — the row silently renames as EN and
   lands in the EN campaign. That is why this registry is a closed set rather
   than free text, and why adding a sixth language here is NOT enough: the
   downstream ``LANG_MAP`` has to learn it first.
2. **English carries NO suffix.** The canonical writer downstream
   (``apply_language_to_job_id``) emits ``-es``/``-pt``/``-de``/``-ja`` only,
   and the parser's trailing rule is ``-(es|pt|de|ja)$``. Writing ``-en`` would
   still *resolve* to EN (it is the default), but ``-en`` would not be
   stripped, so the core would become ``1736426f-en`` and leak into the
   filename. Hence :data:`EN` maps to an empty suffix.
3. **Bare ids stay valid.** No suffix means EN, which is exactly what every
   row exported before this feature shipped already is. Nothing needs
   backfilling, and a run whose ``language`` is NULL behaves as it always did.

JA/DE are Test Flow-only downstream (BAU Cross rejects them with
``422 unsupported_language``); that is a downstream routing concern, not
something we gate on here.

Keep this module free of app imports — api_v2, web_v2 and tasks all import it,
and it is the single source of truth for the set of valid values (the column is
a plain VARCHAR, deliberately: adding a language must never need a migration).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Code stored in runs.language. Also the ISO 639-1 code, lowercased.
EN = "en"
ES = "es"
PT = "pt"
JA = "ja"
DE = "de"

# A run with no language set behaves as English — see property 3 above.
DEFAULT_LANGUAGE = EN


@dataclass(frozen=True)
class LanguageSpec:
    """One selectable language + the Job ID suffix it writes."""

    key: str
    label: str
    # What gets appended to the id, INCLUDING the separator. Empty for English:
    # the downstream parser strips only -es/-pt/-de/-ja, so an "-en" suffix
    # would survive into the filename core.
    suffix: str


# Ordered — this is the order the New Run dropdown renders in.
LANGUAGES: dict[str, LanguageSpec] = {
    EN: LanguageSpec(EN, "English", ""),
    ES: LanguageSpec(ES, "Spanish", "-es"),
    PT: LanguageSpec(PT, "Portuguese", "-pt"),
    JA: LanguageSpec(JA, "Japanese", "-ja"),
    DE: LanguageSpec(DE, "German", "-de"),
}

# Every non-empty suffix, for stripping one back off a displayed id.
_SUFFIX_RE = re.compile(
    r"-(" + "|".join(s.key for s in LANGUAGES.values() if s.suffix) + r")$",
    re.IGNORECASE,
)


def normalize(code: str | None) -> str | None:
    """Return the canonical code for *code*, or None when it isn't a language.

    Accepts any case/whitespace ("  JA " -> "ja"). An empty or unknown value
    returns None rather than falling back to English, so callers can tell
    "operator picked English" from "operator picked nothing" — both behave as
    EN downstream, but only the former is a deliberate choice.
    """
    key = (code or "").strip().lower()
    return key if key in LANGUAGES else None


def is_valid(code: str | None) -> bool:
    """True when *code* names one of the five supported languages."""
    return normalize(code) is not None


def label_for(code: str | None) -> str:
    """Human label for *code* ("ja" -> "Japanese"), defaulting to English."""
    spec = LANGUAGES.get(normalize(code) or DEFAULT_LANGUAGE)
    return spec.label if spec else LANGUAGES[DEFAULT_LANGUAGE].label


def suffix_for(code: str | None) -> str:
    """Job ID suffix for *code* — ``"-ja"``, or ``""`` for English/unset."""
    spec = LANGUAGES.get(normalize(code) or "")
    return spec.suffix if spec else ""


def apply_suffix(core: str, code: str | None) -> str:
    """Append *code*'s language suffix to an already-shortened id.

    ``apply_suffix("1736426f", "ja") -> "1736426f-ja"``;
    ``apply_suffix("1736426f", "en") -> "1736426f"``.
    """
    return f"{core}{suffix_for(code)}"


def strip_suffix(text: str | None) -> str:
    """Remove a trailing language suffix from *text* (``"1736426f-ja"`` ->
    ``"1736426f"``).

    Used where an id is matched against stored uuids — the operator sees (and
    pastes) the suffixed form, but only the core is in the database.
    """
    return _SUFFIX_RE.sub("", (text or "").strip())


def job_id_for(entity_id: str, code: str | None, *, length: int = 8) -> str:
    """Build the sheet Job ID for an entity: id prefix + language suffix.

    *length* is the id prefix width — 8 for the Final Creatives ``Job ID``
    (the short id shown throughout the UI), 16 for the Caption Demo
    ``job_key``, which strips dashes first (see
    ``tasks._caption_demo_job_key``).
    """
    return apply_suffix((entity_id or "")[:length], code)
