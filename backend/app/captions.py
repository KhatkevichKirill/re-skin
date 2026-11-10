"""
captions.py — turn timed transcript lines into a burnable ASS subtitle file.

This is the pure, offline half of the burned-captions feature (design
contract: ``docs/burned-captions.md`` §3.1). It knows nothing about the
database, ffmpeg, or the network — :mod:`app.pipeline_captions` calls
:func:`build_ass`, writes the string to ``captions.ass``, and hands that path
to :func:`app.media.burn_subtitles`, which is the only part of the feature
that ever shells out. Keeping this module pure is deliberate: the house style
(font sizes, colours, wrap widths) is exactly the kind of thing that gets
tuned by eyeballing a rendered frame, and a tuner should be able to import
this module in a plain Python shell — no Postgres, no Redis, no ffmpeg
binary — call :func:`build_ass` a dozen times with different ratios, and diff
the output.

Why ASS and not SRT
--------------------
SRT carries only text and timing — no font, no colour, no position, no
control over how a long line wraps. Every one of those is exactly what §1.2
of the design doc asks for ("one universal style for every caption, every
language"): a fixed font-size ratio, bold white text on a black outline,
bottom-centred, at most three rows. None of that is expressible in SRT; all
of it is a `` [V4+ Styles] `` line in ASS (Advanced SubStation Alpha), which
is what libass — the subtitle renderer ffmpeg's ``subtitles`` filter is
built on — was designed around. Using ASS to express a *single* fixed style
looks like overkill next to SRT's three lines per caption, but it is the only
format that actually has a knob for "bottom margin" or "outline colour" at
all; reproducing that with SRT would mean abandoning ffmpeg's ``subtitles``
filter for a hand-rolled ``drawtext`` overlay per line, which is far more
code for a worse result (no automatic line wrapping, no bidi/shaping via
libraqm/fribidi for languages that need it).

Why one universal style
------------------------
Operator decision, 2026-08-09 (docs/burned-captions.md §1.2): no per-run
styling, no attempt to match any external captioning tool's look. The only
axis that legitimately varies by language is the *font*, because a Latin font
carries no CJK glyphs and vice versa — see :func:`font_for_language`. Every
other knob (size, colour, margins, alignment) is a module-level ratio
constant, not a parameter, on purpose: a caller who wants a different look
is a sign the "universal style" decision needs revisiting, not that this
module needs a styling API.

Fitting the frame (``captions/v2``)
-----------------------------------
``captions/v1`` wrapped by **counting characters** — 28 per row for Latin,
14 for CJK — and set ``WrapStyle: 2`` ("never wrap") so libass would not
re-flow the rows this module had already chosen. Both halves of that were
wrong together, and the combination is what put text off the edge of every
delivered video:

* A character is not a width. At the v1 style on a 1080x1920 delivery (the
  overwhelmingly common frame here) a row was allowed 28 characters of
  86px bold DejaVu — about 1300px of text inside a 908px safe area, in a
  1080px frame. Rows near the budget did not merely leave the margins, they
  left the *picture*, clipped at both edges.
* ``WrapStyle: 2`` then removed the only backstop. ``MarginL``/``MarginR``
  are not a clip rectangle in ASS; they are the box libass wraps *into*, so
  with wrapping switched off they described a safe area nothing enforced.

v2 fixes the cause rather than the symptom: rows are wrapped to a measured
**pixel** width (:func:`measure_text`, real font metrics via Pillow when it
is importable, a per-script em estimate when it is not), and ``WrapStyle``
is ``0`` so libass re-wraps anything our measurement got wrong instead of
letting it overhang. Two independent mechanisms now have to fail before a
caption can leave the frame.

Fitting the width alone is not enough, though — it just moves the loss from
the edges of the frame to the end of the sentence. At a legible size a row
holds ~25 Latin characters, so an utterance long enough to need four rows
cannot be shown in the :data:`MAX_ROWS` this style allows. v1 dropped the
surplus rows (with a warning nobody reads); v2 **splits the caption in
time** instead, showing the rows in :data:`MAX_ROWS`-sized groups across the
line's own start/end window (:func:`layout_text`, :func:`split_line`). The
words all get shown; they just get shown in sequence, which is what a human
captioner would have done with the same sentence.

Line shape
----------
Every public function that takes or returns "lines" uses exactly this shape::

    {"start": <seconds, float>, "end": <seconds, float>, "text": <str>}

This is deliberately the same subset of keys the §4.1 localisation transcript
uses (``id``, ``speaker``, ``on_screen``, ``source_text`` are the extra keys
that transcript carries) — :func:`prepare_lines` returns only
``start``/``end``/``text`` (it reads ``speaker`` solely to lay out merged
quick replies, AICP-247), so a transcript's ``lines`` array can be handed
to :func:`build_ass` untouched, with no adapter step in between.

Untrusted input, by design
---------------------------
``lines`` comes from a video-transcribing LLM by way of
``localisation.transcribe_video``, or from an operator hand-editing a JSON
blob through the API. Both are allowed to send garbage: blank text, a
timestamp that's the string ``"NaN"``, lines that overlap or run past the end
of the clip. :func:`prepare_lines` is the one place that turns "whatever the
model said" into "safe to hand to ffmpeg", and it is written to the standard
this codebase holds transcript-cleaning code to (see
``app.localisation._clean_lines``): never raise, drop what can't be repaired,
and log every drop at WARNING with enough context (index, timestamps, a text
snippet) that an operator asking "why does this caption cut off early" can
find the answer in the worker log.

ASS gotchas
-----------
Four things about the ASS format are easy to get backwards, and getting any
of them wrong produces a file that libass either mis-renders or silently
ignores rather than errors on — so they are each covered by a direct
assertion in ``tests/test_captions.py``, not just exercised incidentally:

* Colours are ``&HAABBGGRR`` — alpha **first**, then **blue before red**.
  This is the reverse of how most people read RGB, and the reverse of the
  AARRGGBB convention used elsewhere (e.g. Android). ``00`` alpha means fully
  opaque, not fully transparent. See :func:`_ass_colour`.
* ``Bold`` is the sentinel ``-1``, not ``1``. ASS booleans in the style line
  are "``-1`` = true, ``0`` = false"; a literal ``1`` is truthy in Python but
  is not the value libass checks for.
* Timestamps are ``H:MM:SS.cc`` — centiseconds (two digits, not three), and
  exactly one hour digit for anything under ten hours (``1:23:45.67``, never
  ``01:23:45.67``). See :func:`format_timestamp`.
* ``{`` opens an override block (font/colour/position control codes) and
  ``\\`` starts an escape sequence (``\\N``, ``\\n``, ``\\h`` are all
  meaningful even outside a ``{}`` block). There is no libass escape that
  renders a literal ``{`` — so dialogue text containing ``{``, ``}`` or ``\\``
  is neutralised by substituting a visually near-identical fullwidth
  character instead, and a *real* newline in the source text becomes the
  ASS row-break control code ``\\N``. See :func:`escape_text`.
* ``BorderStyle: 3`` (the v2 caption band) **repurposes two fields**: the
  box is filled with ``OutlineColour`` — *not* ``BackColour``, which stays
  the drop shadow — and ``Outline`` stops being a border width and becomes
  the box's padding around the text. Set those two thinking in BorderStyle
  1 terms and you get an invisible box of the wrong size, with no error.
  See :func:`_styles_section`.
"""

from __future__ import annotations

import functools
import glob
import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Bumped whenever the rendered look changes in a way that makes an old
# ``captions.ass`` non-comparable to a new one. Stored on ``run_captions.
# style_version`` at burn time (docs/burned-captions.md §2.1) purely as
# forensic metadata — nothing reads it back to change behaviour. v2:
# measured (not character-counted) wrapping, a semi-transparent caption
# band, two balanced rows, and a higher bottom margin.
CAPTION_STYLE_VERSION = "captions/v2"

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

# DejaVu Sans ships in every Linux distro's base font set (and is added
# explicitly to the Docker image per docs/burned-captions.md §5) and covers
# Latin, Cyrillic and Greek — every language app.languages knows about except
# Japanese (Italian, Polish and French, like the other Latin-script codes,
# render here; Polish's ą ć ę ł ń ó ś ź ż and French's é è ê à ç œ « » are all
# in DejaVu Sans, pinned by real libass renders in tests/test_captions.py). It carries no CJK glyphs at all.
DEFAULT_FONT = "DejaVu Sans"

# Noto Sans CJK JP covers Japanese, and (via the same Noto CJK family design)
# renders simplified/traditional Chinese and Korean acceptably too — not
# native-quality for zh/ko, but nobody asked for that yet and "tofu boxes"
# (the empty-rectangle glyph a renderer shows for a codepoint its font has no
# glyph for) is the failure this module exists to avoid. A font name that
# resolves to nothing on the box burning the video renders as tofu *without
# failing ffmpeg* — the burn "succeeds" and produces an unwatchable file —
# which is why fonts-noto-cjk is a Dockerfile change and not an optimisation
# (docs/burned-captions.md §5).
CJK_FONT = "Noto Sans CJK JP"

# Primary-subtag language codes (i.e. before any "-CN" / "-Hans" region or
# script suffix, see font_for_language) that should render in CJK_FONT. Not
# an exhaustive ISO 639 CJK list — just the codes app.languages actually
# hands us ("ja") plus the obvious alternates an operator or an LLM's
# source_language field might use instead (ISO 639-2/3 "jpn", macrolanguage
# "cmn"/"yue" for Chinese, "kor" for Korean).
_CJK_LANGUAGE_CODES = frozenset({"ja", "jpn", "zh", "zho", "chi", "cmn", "yue", "ko", "kor"})

# Unicode ranges that only appear in CJK writing systems: Hiragana+Katakana,
# CJK Unified Ideographs (+ Extension A), Hangul Syllables, Hangul Jamo.
# Deliberately excludes the Halfwidth-and-Fullwidth-Forms block (U+FF00-
# FFEF): that block is where escape_text's substitute characters for a
# literal "{", "}" and "\\" live (see _ESCAPE_SUBSTITUTES below), and this
# sniff must not decide a caption is CJK merely because a *previous* pass of
# escape_text already ran over it.
_CJK_CODEPOINT_RANGES = (
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x1100, 0x11FF),  # Hangul Jamo
    (0xAC00, 0xD7A3),  # Hangul Syllables
)


def _looks_cjk(text: str) -> bool:
    """True if *text* contains at least one CJK-only codepoint.

    Used both by :func:`font_for_language` (when no language code is known)
    and internally by :func:`wrap_text` (when ``cjk`` is not given) to decide
    which wrap width applies. A single matching character is enough — one
    Japanese word dropped into an otherwise-English sentence by a
    code-switching speaker is still a sentence a Latin font cannot fully
    render.
    """
    for ch in text or "":
        cp = ord(ch)
        for lo, hi in _CJK_CODEPOINT_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def font_for_language(language: Optional[str], *, sample_text: str = "") -> str:
    """
    Resolve which of the two shipped fonts a caption file should use.

    Args:
        language: An ISO-639-ish code (``"ja"``, ``"JA"``, ``"ja-JP"``,
            ``"jpn"``, ...). Only the primary subtag (everything before the
            first ``-`` or ``_``) is examined, so region/script suffixes are
            harmless. ``None`` or empty means "unknown" and falls through to
            *sample_text*.
        sample_text: Used only when *language* doesn't resolve to a known
            code — typically the caption's own joined text, so a transcript
            that came back with no (or a wrong) ``source_language`` still
            gets the right font by looking at what it actually says.

    Returns:
        :data:`CJK_FONT` when *language*'s primary subtag is one of
        :data:`_CJK_LANGUAGE_CODES`, **or** when *sample_text* itself
        contains a CJK codepoint — the two checks are OR'd, not "sniff only
        if language is unset". That is a deliberate widening of the design
        doc's literal "when no language is known, sniff sample_text": a
        caption whose actual burned text is Japanese should get a font that
        can render it even if the *language* it was tagged with is wrong,
        missing, or simply not one of the codes this function recognises
        (``"nl"``, a typo, ...) — the alternative is the tofu-box failure
        mode this whole font-selection function exists to avoid, and
        DEFAULT_FONT is Latin-only so there is no matching downside: it
        cannot render CJK text regardless of which branch chose it, while
        CJK_FONT renders Latin acceptably too. :data:`DEFAULT_FONT` is
        returned only when neither check fires.
    """
    code = str(language or "").strip().lower()
    primary = code.replace("_", "-").split("-", 1)[0] if code else ""
    if primary in _CJK_LANGUAGE_CODES or _looks_cjk(sample_text):
        return CJK_FONT
    return DEFAULT_FONT


# ---------------------------------------------------------------------------
# Text measurement
#
# Wrapping to a *width* means knowing how wide a candidate row renders, which
# means reading the same font file libass will. Two paths, in preference
# order, because this module must stay importable and correct in a bare
# Python shell (see the module docstring) — including one with no Pillow and
# no font files at all, which is where the unit tests run:
#
#   1. Real metrics. Pillow is present in the worker image (a transitive
#      dependency of insightface, and pinned in requirements.txt because this
#      module now relies on it) and both shipped fonts are installed there by
#      the Dockerfile. This measures the actual glyph advances, so a row that
#      measures as fitting does fit.
#   2. An em estimate. When Pillow is missing, or the font file is not where
#      we expect, every character is charged a fixed fraction of the em (see
#      _AVG_ADVANCE_EM_*), deliberately rounded UP from the real average so
#      the estimate wraps early rather than late.
#
# Either way ``WrapStyle: 0`` leaves libass free to re-wrap a row we sized
# wrongly, so the worst case of a bad estimate is an ugly break, not text
# hanging off the frame.
# ---------------------------------------------------------------------------

#: Where each shipped font family's BOLD face lives — bold because the house
#: style sets Bold, and a bold face is measurably wider than its regular
#: sibling (measuring the wrong weight is exactly the kind of "off by 8%"
#: error that only shows up on the longest line of a real transcript).
#: Several candidates per family: the Dockerfile's Debian paths first, then a
#: glob for the same file under any other font root, so a dev box that
#: installed the fonts elsewhere still measures rather than estimating.
_FONT_FILE_CANDIDATES = {
    DEFAULT_FONT: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
    ),
    CJK_FONT: (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/**/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/**/NotoSansCJK*-Bold*.*",
    ),
}

#: Fallback average glyph advance, as a fraction of the em (= the font size
#: in px), for a Latin caption. DejaVu Sans Bold measures ~0.55 em over real
#: English sentences; 0.60 is that rounded up, since an estimate that wraps a
#: word early is invisible and an estimate that wraps late is the bug this
#: whole section exists to fix.
_AVG_ADVANCE_EM_LATIN = 0.60

#: The same for CJK, where the ideographs are full-width by construction —
#: one em each, near enough that no safety margin is warranted.
_AVG_ADVANCE_EM_CJK = 1.0


@functools.lru_cache(maxsize=8)
def _font_file(family: str) -> Optional[str]:
    """Resolve a font *family* name to a file on disk, or None.

    Deliberately a small hardcoded table plus a glob rather than a call out
    to ``fc-match``: this module shells out to nothing (see the module
    docstring), the image ships exactly two fonts, and a subprocess per
    caption job to re-derive two constant paths would be a poor trade.
    """
    for candidate in _FONT_FILE_CANDIDATES.get(family, ()):
        if "*" in candidate:
            matches = sorted(glob.glob(candidate, recursive=True))
            if matches:
                return matches[0]
        elif os.path.exists(candidate):
            return candidate
    return None


@functools.lru_cache(maxsize=16)
def _pil_font(family: str, size_px: int):
    """Load *family* at *size_px* through Pillow, or None if unavailable.

    Cached because :func:`wrap_rows` measures once per candidate row and a
    caption job has hundreds of them, while ``(family, size_px)`` takes at
    most two distinct values in a whole file.

    Never raises: a missing Pillow, a missing font file and an unreadable
    one all mean the same thing to the caller — fall back to the estimate.
    """
    path = _font_file(family)
    if not path:
        return None
    try:
        from PIL import ImageFont  # imported lazily: see path 2 above
    except Exception:  # pragma: no cover - exercised only on a Pillow-less box
        logger.debug("captions: Pillow unavailable, estimating text widths")
        return None
    try:
        return ImageFont.truetype(path, int(size_px))
    except Exception:
        logger.warning(
            "captions: could not load %r at %dpx from %s — estimating text "
            "widths instead", family, size_px, path, exc_info=True,
        )
        return None


def measure_text(text: str, *, font: str, size_px: int) -> float:
    """
    Width of *text* rendered in *font* at *size_px*, in pixels.

    The measurement ignores the outline/box padding, which the caller
    accounts for by shrinking the usable width (:func:`usable_width_px`) —
    padding is a per-row constant while this is a per-row variable, and
    keeping them apart is what lets the style's padding change without
    touching the wrap logic.
    """
    text = str(text or "")
    if not text:
        return 0.0
    loaded = _pil_font(font, int(size_px))
    if loaded is not None:
        try:
            return float(loaded.getlength(text))
        except Exception:  # pragma: no cover - defensive, Pillow API drift
            logger.warning("captions: measuring %r failed", _snippet(text), exc_info=True)
    em = _AVG_ADVANCE_EM_CJK if font == CJK_FONT else _AVG_ADVANCE_EM_LATIN
    return len(text) * em * float(size_px)


# ---------------------------------------------------------------------------
# House style — every ratio is of PlayResY (frame height) except
# MARGIN_H_RATIO, which is of PlayResX (frame width) because a left/right
# safe margin has to scale with how wide the frame is, not how tall it is.
# Exported by name (rather than left as literals inside build_ass) so a
# future "caption style tuner" script, and the tests below, can read and
# override them without grepping for magic numbers.
# ---------------------------------------------------------------------------

#: Font size as a fraction of frame height. v1 used 4.5%; v2 drops to 3.6%
#: (69px on a 1080x1920 delivery) because the size interacts with the row
#: cap: a row holds ~950px of text, so 4.5% left MAX_ROWS rows able to carry
#: only a short sentence and forced most real utterances into a time split.
#: 3.6% is still comfortably readable at phone size — it is roughly what the
#: platforms' own auto-captions use — and lets two rows hold ~50 characters,
#: which covers the majority of transcribed lines in one caption.
FONT_SIZE_RATIO = 0.036

#: Padding between the text and the edge of its background box, as a
#: fraction of frame height. This is the ``Outline`` style field, which under
#: :data:`BORDER_STYLE_BOX` means padding rather than border width (see the
#: module docstring's ASS gotchas).
BOX_PADDING_RATIO = 0.0045

#: Drop-shadow distance as a fraction of frame height. **Zero** under
#: :data:`BORDER_STYLE_BOX`, and that is a rendered-frame finding rather
#: than a taste call: libass draws the shadow as a second, *opaque*
#: (``BackColour``) box offset behind the translucent one, so any non-zero
#: shadow fills in the transparency :data:`BOX_ALPHA` asks for and the band
#: comes out solid black. A shadow under a box adds nothing anyway — the box
#: is already the separation from the footage that a shadow would provide.
SHADOW_RATIO = 0.0

#: Bottom margin (distance from the bottom edge to the caption's baseline
#: row) as a fraction of frame height. 16% (v1: 12%) clears the platform
#: chrome most delivery destinations overlay near the bottom of a vertical
#: video — the caption/CTA bar on Reels and TikTok in particular reaches
#: higher than 12% of the frame, which put v1 captions behind it.
MARGIN_V_RATIO = 0.16

#: Left/right margin as a fraction of frame WIDTH (not height — see the
#: section note above). v1 used 8%; with wrapping now measured against this
#: margin instead of ignoring it, 6% buys back ~40px of usable row width
#: without letting text approach the edge.
MARGIN_H_RATIO = 0.06

#: Bottom-centred: see the ASS spec's numeric keypad-style Alignment values,
#: where 1/2/3 are the bottom row and 2 is bottom-CENTER.
ALIGNMENT_BOTTOM_CENTER = 2

#: BorderStyle 1 = "outline + drop shadow". Not the v2 house style (see
#: :data:`BORDER_STYLE_BOX`); kept as a named constant because it is the
#: value every other ASS style in the wild uses, so its absence would
#: otherwise read as an oversight.
BORDER_STYLE_OUTLINE = 1

#: BorderStyle 3 = a filled box behind the text — the v2 house style. An
#: outline alone (v1) is legible over most footage but not over the bright,
#: high-frequency backgrounds this pipeline delivers a lot of (shop shelves,
#: screens, daylight). A semi-transparent band guarantees contrast without
#: hiding the picture behind it.
BORDER_STYLE_BOX = 3

#: ASS boolean sentinel for "true" in the style line. Not ``1`` — see the
#: module docstring's ASS-gotchas list.
_BOLD_ON = -1
_BOLD_OFF = 0

#: At most this many rows are shown at once. Two, not v1's three: three rows
#: of a bottom-anchored caption cover a third of a vertical frame's lower
#: half, and with the rows now balanced (:func:`wrap_rows`) a two-row caption
#: no longer strands a single orphan word on a row of its own. Rows beyond
#: the cap are not dropped — they become the next caption in the same time
#: window (:func:`split_line`).
MAX_ROWS = 2

#: Frame size assumed by the geometry helpers when a caller does not pass
#: one. Not a fallback used by the pipeline (:func:`build_ass` always has the
#: probed frame size); it is what makes the wrap helpers callable on their
#: own in a shell for tuning, at the size deliveries actually use.
NOMINAL_WIDTH = 1080
NOMINAL_HEIGHT = 1920

#: Minimum time a caption stays on screen, in seconds, regardless of how
#: short the transcribed utterance was. Not specified numerically by the
#: design doc (docs/burned-captions.md §3.1 says only "enforce a minimum on
#: -screen duration") — 0.5s is this module's chosen floor: short enough that
#: it never visibly pads a normal line (real spoken utterances this pipeline
#: sees are essentially never under half a second), long enough that a
#: mistranscribed near-zero-length entry doesn't flash for a couple of
#: frames and vanish before anyone can read it.
MIN_LINE_DURATION_SEC = 0.5

#: AICP-247 — how :func:`prepare_lines` joins two transcript lines it has to
#: MERGE into one caption (quick consecutive replies that cannot each get
#: :data:`MIN_LINE_DURATION_SEC` on their own; they used to be dropped).
#: ``True``: when the two lines carry different, known ``speaker`` values, the
#: second one starts a NEW ROW of the same caption ("Quoi ?" / "Mets ça."),
#: provided every speaker's text fits on its own rows within the caption's
#: row cap — otherwise the caption falls back to the ordinary measured,
#: balanced wrap of the space-joined text (:func:`layout_text`). ``False``:
#: always join with a space and wrap normally, whoever is speaking.
#:
#: PENDING APPROVAL (Operator, AICP-247): the merge itself is not optional —
#: dropping dialogue is the bug — but this presentation choice is, and
#: flipping this one flag is the whole switch.
SPEAKER_CHANGE_STARTS_NEW_ROW = True

#: Middle-centred (numeric keypad 5): with a ``\\pos`` override on every
#: event, libass anchors the CENTRE of the multi-row block (box included) on
#: the given point, whatever the row count. Verified on rendered frames
#: (AICP-368): one to three rows at 1080x1920 all centre within a pixel of
#: the requested y. Used only by a :class:`Placement`.
ALIGNMENT_MIDDLE_CENTER = 5

#: Hard cap on rows per caption for a :class:`Placement`. The house
#: :data:`MAX_ROWS` (two) is about a bottom-anchored caption eating the lower
#: half of the frame; a placed caption sits over the original caption band
#: instead, and a longer localised utterance reads better as one three-row
#: caption than as a 2+1 split in time (AICP-368: "keep <= 3 rows").
PLACEMENT_MAX_ROWS = 3


@dataclass(frozen=True)
class Placement:
    """Where to put the captions instead of the house bottom position.

    Built by :func:`app.caption_placement.burn_placement` from a validated
    project/caption placement; this module only renders it.

    * ``center_y`` — fraction of frame height the CENTRE of every caption
      block sits on (rows and box grow symmetrically around it). When the
      text is too short to cover the original band, it still covers the
      band's centre — the AICP-368 fitting rule.
    * ``max_width`` — fraction of frame width a row (box included) may use,
      or None for the house usable width (frame less :data:`MARGIN_H_RATIO`
      on both sides). Never wider than the house width.
    * ``max_rows`` — rows per caption, 1..:data:`PLACEMENT_MAX_ROWS`; longer
      text is split in time exactly as the house style does.
    """

    center_y: float
    max_width: Optional[float] = None
    max_rows: int = PLACEMENT_MAX_ROWS


def _ass_colour(red: int, green: int, blue: int, alpha: int = 0) -> str:
    """
    Render an RGBA colour as the ASS/SSA ``&HAABBGGRR`` hex string.

    ASS stores colours **alpha first, then blue, then green, then red** — the
    reverse of the RGB most people think in, and the opposite byte order
    from the AARRGGBB convention used elsewhere. Getting this backwards is
    invisible on this module's own house colours (pure white, pure black are
    symmetric under any channel permutation) which is exactly why it is a
    named, independently-tested helper rather than two hand-typed literals:
    a hand-typed ``"&H00FFFFFF"`` would pass every test here whether or not
    whoever typed it understood the byte order, and the day someone adds a
    non-symmetric colour to the house style, a swapped channel would ship
    silently.

    Args:
        red, green, blue: 0-255.
        alpha: 0-255. ``0`` (the default) is fully opaque — ASS alpha is
            "amount of transparency", not "amount of colour", which is the
            other easy-to-invert-by-accident bit of this format.
    """
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


#: Bold white fill — the caption text itself.
_COLOUR_WHITE_OPAQUE = _ass_colour(0xFF, 0xFF, 0xFF)

#: Opaque black — the drop shadow behind the box (``BackColour``).
_COLOUR_BLACK_OPAQUE = _ass_colour(0x00, 0x00, 0x00)

#: How transparent the caption band is, 0x00 = opaque .. 0xFF = invisible
#: (ASS alpha is *transparency*, see :func:`_ass_colour`). 0x59 is ~65%
#: opaque: dark enough that white bold text keeps its contrast over daylight
#: footage, light enough that the picture still reads through the band
#: instead of the caption punching a black hole in the frame.
BOX_ALPHA = 0x59

#: The band itself. Under :data:`BORDER_STYLE_BOX` this is the ``Outline``
#: **Colour** field — the box fill, not a border (module docstring, ASS
#: gotchas).
_COLOUR_BOX = _ass_colour(0x00, 0x00, 0x00, alpha=BOX_ALPHA)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """
    Render *seconds* as an ASS ``Dialogue:`` timestamp: ``H:MM:SS.cc``.

    Centiseconds (two digits — ASS timing resolution is 1/100s, not SRT's
    1/1000s), minutes and seconds zero-padded to two digits, and **exactly
    one** hour digit for anything under ten hours (``1:23:45.67``, never
    ``01:23:45.67``, never ``1:2:3.4``) — libass parses the field positionally
    and a mis-padded value shifts every field after it.

    Rounding is done on the total centisecond count, not on independently
    rounded hours/minutes/seconds/centiseconds, specifically so a value like
    ``3599.996`` rolls cleanly over the minute *and* hour boundary
    (``1:00:00.00``) instead of producing a nonsensical ``0:59:100`` (or
    silently truncating to ``0:59:99.99`` and drifting the caption 4ms per
    such rollover).

    Never raises: a non-finite or otherwise unusable input is treated as 0,
    since this is called from :func:`build_ass` on values that have already
    passed through :func:`prepare_lines`, but a subtitle file with one wrong
    timestamp is a worse failure mode than a defensive fallback here.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        value = 0.0
    value = max(0.0, value)

    total_centis = int(round(value * 100))
    hours, remainder = divmod(total_centis, 360000)  # 100 * 60 * 60
    minutes, remainder = divmod(remainder, 6000)  # 100 * 60
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


# ---------------------------------------------------------------------------
# Text escaping
# ---------------------------------------------------------------------------

# Fullwidth lookalikes for the three ASCII characters that mean something
# special in an ASS Dialogue Text field (see the module docstring's ASS-
# gotchas list). Chosen from Unicode's Halfwidth and Fullwidth Forms block
# specifically because they read, to a human, as "the same punctuation mark,
# just wide" rather than as mojibake, and because that block is nowhere near
# any of the CJK ranges _looks_cjk checks (so escaping text does not, as a
# side effect, make it look CJK to font_for_language on a later pass).
_ESCAPE_SUBSTITUTES = {
    "\\": "＼",  # FULLWIDTH REVERSE SOLIDUS  ＼
    "{": "｛",  # FULLWIDTH LEFT CURLY BRACKET  ｛
    "}": "｝",  # FULLWIDTH RIGHT CURLY BRACKET  ｝
}


def escape_text(text: str) -> str:
    """
    Make *text* safe to drop verbatim into an ASS ``Dialogue:`` Text field.

    Three transformations, applied in this order (order matters — see
    below):

    1. Normalise ``\\r\\n``/``\\r`` to ``\\n`` so the next steps only have one
       newline convention to deal with.
    2. Replace every literal ``\\``, ``{`` and ``}`` with the fullwidth
       lookalike from :data:`_ESCAPE_SUBSTITUTES`. There is no libass escape
       that renders a literal ``{`` (it always opens an override block —
       font, colour, position control codes — with no closing brace required
       for libass to start interpreting), and a bare ``\\`` begins an escape
       sequence (``\\N``, ``\\n``, ``\\h``) even outside a ``{}`` block. A
       substitution is the only option that survives libass unchanged; there
       is no combination of backslashes that "escapes the escape".
    3. Replace the now-normalised ``\\n`` with the literal two-character ASS
       row-break code ``\\N``.

    Step 2 must run before step 3: it neutralises every backslash that was
    already in *text* first, so the only real backslash characters left in
    the output are the ones this function deliberately writes in step 3.
    Doing it in the other order would let a caption containing a literal
    backslash immediately followed by an ``n`` (rare, but real — a Windows
    path in an on-screen brand name, say) collide with a row break we insert
    a moment later.

    This is the last step applied to a line's text before it is written into
    the ``[Events]`` section — call it on the *wrapped* text (whose row
    breaks are real ``\\n`` characters at that point), not the other way
    round.
    """
    result = str(text or "")
    result = result.replace("\r\n", "\n").replace("\r", "\n")
    for original, substitute in _ESCAPE_SUBSTITUTES.items():
        result = result.replace(original, substitute)
    return result.replace("\n", "\\N")


# ---------------------------------------------------------------------------
# Wrapping
# ---------------------------------------------------------------------------


def font_size_px(height: int) -> int:
    """The style's font size, in whole pixels, for a frame *height* px tall."""
    return max(1, int(round(int(height) * FONT_SIZE_RATIO)))


def margin_h_px(width: int, max_width: Optional[float] = None) -> int:
    """Left (= right) margin in px: the house :data:`MARGIN_H_RATIO`, or the
    narrower centred column a :class:`Placement` asks for (*max_width* as a
    fraction of the frame width). Never smaller than the house margin."""
    house = int(round(int(width) * MARGIN_H_RATIO))
    if max_width is None:
        return house
    return max(house, int(round(int(width) * (1.0 - float(max_width)) / 2.0)))


def usable_width_px(width: int, height: int, max_width: Optional[float] = None) -> float:
    """
    How many pixels of *text* a row may occupy in a *width* x *height* frame.

    The frame width less both side margins (the box libass wraps into) less
    the box padding on both sides, because the band is drawn around the text
    and it is the band, not the glyphs, that must stay inside the margins.
    Floored at one em so a pathologically narrow frame still wraps somewhere
    instead of dividing by ~zero and emitting one row per character.
    """
    size = font_size_px(height)
    margin = margin_h_px(width, max_width)
    padding = int(height) * BOX_PADDING_RATIO
    return max(float(size), float(int(width) - 2 * margin - 2 * padding))


def _tokens(text: str, *, cjk: bool) -> List[str]:
    """Split *text* into the atoms a row is built from.

    Latin: whitespace-separated words, rejoined with a space. CJK: single
    characters, rejoined with nothing — Japanese and Chinese are written
    without spaces, so a word-wrapper has nothing to break on and would
    return the whole utterance as one unbreakable row.
    """
    return list(text) if cjk else text.split()


def _greedy_rows(tokens: List[str], limit: float, *, joiner: str,
                 font: str, size_px: int) -> List[str]:
    """Fill rows with *tokens* in order, breaking when the next one overflows.

    A single token wider than *limit* on its own (a URL, a long brand name,
    and in CJK a single glyph in a very narrow frame) is split by character
    rather than emitted over-wide — the row cap in :func:`wrap_rows` is about
    how many rows a caption may have, and must not be reachable by one
    unbreakable token.
    """
    rows: List[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current}{joiner}{token}" if current else token
        if measure_text(candidate, font=font, size_px=size_px) <= limit:
            current = candidate
            continue
        if current:
            rows.append(current)
            current = ""
        if measure_text(token, font=font, size_px=size_px) <= limit:
            current = token
            continue
        # Overlong single token: break it at the last character that fits.
        piece = ""
        for char in token:
            if measure_text(piece + char, font=font, size_px=size_px) <= limit or not piece:
                piece += char
            else:
                rows.append(piece)
                piece = char
        current = piece
    if current:
        rows.append(current)
    return rows or [""]


def wrap_rows(
    text: str,
    *,
    cjk: Optional[bool] = None,
    width: int = NOMINAL_WIDTH,
    height: int = NOMINAL_HEIGHT,
    font: Optional[str] = None,
    max_width: Optional[float] = None,
) -> List[str]:
    """
    Wrap *text* into rows that each measure inside the frame's usable width.

    Unlike v1 this imposes **no row cap** — every word comes back. Capping is
    :func:`split_line`'s job, and it caps by splitting the caption in time
    rather than by discarding text (see the module docstring).

    Rows are *balanced*: once a greedy pass has established how many rows the
    text needs, the narrowest limit that still needs that same number of rows
    is found by bisection and the text re-wrapped at it. Greedy alone packs
    early rows full and leaves the remainder stranded on the last one ("So I
    just take pictures of" / "text"), which reads as a mistake; balancing
    costs ~10 extra wrap passes over a handful of words and removes the
    orphan without changing the row count.

    Args:
        text: Raw caption text (not yet escaped — call this *before*
            :func:`escape_text`, which turns the row breaks into ``\\N``).
        cjk: Which tokenisation to use. ``None`` auto-detects from *text*
            via :func:`_looks_cjk`; :func:`build_ass` passes an explicit
            value resolved once from the whole job's language, so every line
            in one file wraps the same way even if one line happens to be a
            pure-Latin brand name inside a Japanese transcript.
        width, height: Frame size in px; everything about the row limit
            (font size, margins, padding) derives from these.
        font: Font family to measure in. Defaults to whichever font *cjk*
            (or the sniffed text) implies, so a standalone caller gets a
            consistent answer without having to resolve it first.
        max_width: A :class:`Placement`'s row width (fraction of the frame
            width); None = the house usable width.
    """
    text = str(text or "")
    if not text:
        return []

    is_cjk = _looks_cjk(text) if cjk is None else bool(cjk)
    if font is None:
        font = CJK_FONT if is_cjk else DEFAULT_FONT
    size_px = font_size_px(height)
    limit = usable_width_px(width, height, max_width)
    joiner = "" if is_cjk else " "
    tokens = _tokens(text, cjk=is_cjk)
    if not tokens:
        return []

    rows = _greedy_rows(tokens, limit, joiner=joiner, font=font, size_px=size_px)
    if len(rows) < 2:
        return rows

    # Bisect for the narrowest limit that still fits in len(rows) rows. The
    # invariant: `high` always achieves the target row count (the greedy pass
    # above proves it for the initial value), `low` never does, so the loop
    # closes on the tightest achievable limit without ever returning a wrap
    # that needs an extra row.
    target = len(rows)
    low, high = 0, int(limit)
    while high - low > 1:
        mid = (low + high) // 2
        if len(_greedy_rows(tokens, mid, joiner=joiner, font=font, size_px=size_px)) <= target:
            high = mid
        else:
            low = mid
    return _greedy_rows(tokens, high, joiner=joiner, font=font, size_px=size_px)


def wrap_text(
    text: str,
    *,
    cjk: Optional[bool] = None,
    width: int = NOMINAL_WIDTH,
    height: int = NOMINAL_HEIGHT,
    font: Optional[str] = None,
) -> str:
    """:func:`wrap_rows`, newline-joined — the shape :func:`escape_text` wants."""
    return "\n".join(wrap_rows(text, cjk=cjk, width=width, height=height, font=font))


def split_line(rows: List[str], *, max_rows: int = MAX_ROWS) -> List[List[str]]:
    """
    Group *rows* into consecutive captions of at most *max_rows* rows each.

    This is where v1 dropped text and v2 does not: a five-row utterance
    becomes three captions shown one after another (see
    :func:`_events_section` for how the line's time window is divided between
    them), not the first two rows and a warning.
    """
    max_rows = max(1, int(max_rows))
    return [rows[i : i + max_rows] for i in range(0, len(rows), max_rows)] or []


def layout_text(
    text: str,
    *,
    cjk: Optional[bool] = None,
    width: int = NOMINAL_WIDTH,
    height: int = NOMINAL_HEIGHT,
    font: Optional[str] = None,
    max_rows: int = MAX_ROWS,
    max_width: Optional[float] = None,
) -> List[List[str]]:
    """
    Wrap *text* to the frame and group the rows into displayable captions.

    Returns a list of captions, each a list of at most *max_rows* rows —
    ``[]`` for empty text. Each caption's rows are re-balanced among
    themselves after the split, so the tail caption of a long line is as
    evenly set as the first (balancing before the split would balance rows
    that end up in different captions against each other, which is exactly
    the wrong comparison).

    A newline in *text* marks a speaker change inside a caption that
    :func:`prepare_lines` merged (AICP-247). Each speaker's part is wrapped on
    its own, and if all the parts together fit in *max_rows* rows they are
    returned as ONE caption with every speaker starting a new row. If they do
    not, the newlines become ordinary spaces and the text takes the normal
    path below — the speaker rows are a presentation nicety and must never
    cost an extra caption (each one needs its own slice of the time window).
    """
    text = str(text or "")
    if "\n" in text:
        parts = [part.strip() for part in text.split("\n") if part.strip()]
        speaker_rows: List[str] = []
        for part in parts:
            speaker_rows.extend(wrap_rows(part, cjk=cjk, width=width, height=height,
                                          font=font, max_width=max_width))
        if 0 < len(speaker_rows) <= max(1, int(max_rows)):
            return [speaker_rows]
        text = _join_parts(parts)

    rows = wrap_rows(text, cjk=cjk, width=width, height=height, font=font,
                     max_width=max_width)
    if not rows:
        return []

    groups = split_line(rows, max_rows=max_rows)
    if len(groups) < 2:
        return groups

    is_cjk = _looks_cjk(text) if cjk is None else bool(cjk)
    joiner = "" if is_cjk else " "
    balanced: List[List[str]] = []
    for group in groups:
        regrouped = wrap_rows(
            joiner.join(group), cjk=cjk, width=width, height=height, font=font,
            max_width=max_width,
        )
        # Re-wrapping a group's own text can only need fewer rows than it
        # already occupies, never more, but guard anyway: a group that came
        # back bigger than the cap would silently re-introduce v1's overflow.
        balanced.append(regrouped if 0 < len(regrouped) <= max_rows else group)
    return balanced


# ---------------------------------------------------------------------------
# Line normalisation
# ---------------------------------------------------------------------------


def _as_seconds(value) -> Optional[float]:
    """Coerce a model- or operator-supplied value to a finite float, or None.

    ``None`` covers everything :func:`prepare_lines` treats as unparseable:
    the wrong type, a string that doesn't parse, NaN, +/-inf. There is no
    default here — callers decide per-field whether a missing value is fatal
    (a missing ``start`` is: there is no reasonable substitute for "when was
    this said") or repairable (a missing ``end`` defaults to ``start``, then
    :data:`MIN_LINE_DURATION_SEC` below gives it a real duration).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _valid_bound(duration_sec: Optional[float]) -> Optional[float]:
    """Normalise the caller-supplied *duration_sec* clamp bound.

    A bound is only usable if it is a finite positive number; ``None``, NaN,
    zero and negative all collapse to "no bound" rather than being treated as
    an error — a caller passing a nonsensical bound almost certainly meant
    "I don't actually know the duration", and clamping is a best-effort
    safety net, not a contract the caller has to get exactly right.
    """
    bound = _as_seconds(duration_sec)
    return bound if (bound is not None and bound > 0) else None


def _snippet(text: str, limit: int = 60) -> str:
    """Truncate *text* for a log line, without ever raising on odd input."""
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "..."


def prepare_lines(lines: Iterable[dict], *, duration_sec: Optional[float] = None,
                  min_duration_for: Optional[Callable[[str], float]] = None) -> list:
    """
    Normalise untrustworthy ``{start, end, text}`` lines into safe, ordered,
    non-overlapping captions.

    This function is the trust boundary for the whole feature: everything
    downstream (:func:`build_ass`, and past that, ffmpeg's ``subtitles``
    filter) assumes its output is clean. Nothing it does can raise — see
    "Never raises" below — because *lines* is, in the worst case, straight
    off a video-transcribing LLM's JSON answer, and a rendering bug in one
    caption must never take down the whole burn.

    Pipeline, in order:

    1. **Parse.** Drop anything that isn't a dict, drop blank/whitespace-only
       ``text``, drop a line whose ``start`` doesn't parse as a finite
       number. A missing or unparseable ``end`` is *not* fatal on its own —
       it defaults to ``start`` (mirroring the transcript-cleaning
       convention in ``app.localisation._clean_lines``) and gets a real
       duration from step 3.
    2. **Clamp.** ``start`` is floored at 0; both ``start`` and ``end`` are
       capped at *duration_sec* when one is given (and usable — see
       :func:`_valid_bound`). ``end`` is then floored at ``start`` so no line
       has negative duration.
    3. **Enforce the minimum duration, independently per line.** A line
       shorter than :data:`MIN_LINE_DURATION_SEC` has its ``end`` pushed out
       to meet the floor (capped at *duration_sec*, never past it). If even
       that isn't enough room — the line starts too close to the clip's end
       for the floor to fit before *duration_sec* — it runs to the clip end
       and step 5 merges it into the caption before it when that caption is
       still up, or ended less than one floor earlier (AICP-247). With no
       such neighbour it is dropped: on its own it would only flash for a
       few frames at the cut. A line starting at or past the clip end is
       always dropped.
    4. **Sort by start.** Stable, so lines that arrived with equal starts
       keep their original relative order.
    5. **De-overlap by truncating the EARLIER line — or merging.** Never
       the later one: a caption's start is when the words are actually
       said, so pushing a line's start back to make room would show it
       before anyone has spoken it. Walking the sorted lines, each one is
       checked against the last *accepted* caption: if it starts before
       that caption ends, the caption's end is pulled back to this line's
       start. If that truncation would leave the caption shorter than
       :data:`MIN_LINE_DURATION_SEC` — or, when *min_duration_for* is given,
       less than the time that caption's text needs — the two are
       **merged** into one
       caption instead (AICP-247 — this used to drop the earlier line, and
       with it real dialogue whenever two quick replies followed each
       other): it keeps the caption's start, ends at the later of the two
       ends (or later, up to what the merged text needs, never past
       *duration_sec*), and shows both texts — joined by a space, or with the second
       speaker on a new row (:data:`SPEAKER_CHANGE_STARTS_NEW_ROW`,
       :func:`layout_text`). A chain of quick lines folds into one caption
       until a line arrives late enough for the truncation to leave the
       floor, which then starts the next caption on time. The price: a
       merged-in line's caption starts before the line is spoken — by less
       than :data:`MIN_LINE_DURATION_SEC` for short texts (the merge
       condition itself), by less than the earlier text's own need for
       longer ones, whose words :func:`_split_windows` then places late in
       the window anyway. Its text shares the caption's row cap: a merged
       text too long for :data:`MAX_ROWS` is split in time inside its window
       like any long line.

       Checking only the immediate predecessor is sufficient: lines are
       processed in ascending start order, a merge never moves a caption's
       start, and every accepted caption already ends at or before the next
       one starts, so by induction nothing earlier than the immediate
       predecessor can overlap the line being considered.

    Args:
        lines: An iterable of dicts (or anything dict-shaped enough to call
            ``.get`` on). Extra keys beyond ``start``/``end``/``text`` — a
            §4.1 transcript's ``id``/``speaker``/``on_screen`` — are not
            preserved: the returned dicts carry exactly the three keys.
            ``speaker`` is the one extra key READ, and only to decide how a
            merge joins two lines (step 5).
        min_duration_for: Optional ``text -> seconds``: how long a caption
            showing *text* must stay up to show all of it.
            :func:`build_ass` passes floor x the number of captions
            :func:`layout_text` makes of it, so de-overlap never cuts a
            caption below what :func:`_split_windows` needs (which would
            drop its tail). ``None``, a raise, or anything under the floor
            means the floor.
        duration_sec: The clip's duration, when known. Lines (after
            clamping) that end up entirely at or beyond this are pushed to
            zero duration by clamping and then dropped by step 3, since
            there is no room left inside the clip to show them at all.

    Returns:
        A new list of ``{"start": float, "end": float, "text": str}`` dicts,
        sorted by start, pairwise non-overlapping, each at least
        :data:`MIN_LINE_DURATION_SEC` long (a merge only ever lengthens the
        caption it folds into). ``text`` contains a newline only where a
        merge put a speaker change. Possibly empty — an empty (or
        entirely-garbage) input is not an error, it is a caption-free file
        (see :func:`build_ass`).

    Never raises: any single malformed entry is logged at WARNING and
    dropped; if *lines* itself isn't iterable at all (a caller passing
    ``None`` is fine — that's just "no lines" — but a caller passing e.g. an
    int is not), the whole call degrades to an empty result rather than
    propagating a ``TypeError``.
    """
    try:
        return _prepare_lines_impl(lines, duration_sec=duration_sec,
                                   min_duration_for=min_duration_for)
    except Exception:
        logger.warning(
            "prepare_lines: input could not be processed at all (type=%s); "
            "treating as zero lines",
            type(lines).__name__,
            exc_info=True,
        )
        return []


def _prepare_lines_impl(lines: Iterable[dict], *, duration_sec: Optional[float],
                        min_duration_for: Optional[Callable[[str], float]] = None) -> list:
    bound = _valid_bound(duration_sec)

    def need(text: str) -> float:
        """Time a caption showing *text* needs: the floor, or — when the
        caller can lay text out (build_ass) — the floor per rendered
        caption, so a truncation or merge never leaves a long caption less
        time than _split_windows needs to show all of it (AICP-247)."""
        if min_duration_for is None:
            return MIN_LINE_DURATION_SEC
        try:
            value = float(min_duration_for(text))
        except Exception:
            logger.warning("prepare_lines: min_duration_for raised; using the floor",
                           exc_info=True)
            return MIN_LINE_DURATION_SEC
        return value if (math.isfinite(value) and value > MIN_LINE_DURATION_SEC) \
            else MIN_LINE_DURATION_SEC

    # --- 1 & 2: parse + clamp -------------------------------------------
    cleaned: list = []
    for index, item in enumerate(lines or []):
        try:
            if not isinstance(item, dict):
                logger.warning(
                    "prepare_lines: dropping line %d — not an object (%r)",
                    index, item,
                )
                continue

            # Any newline inside one transcript line is collapsed to a space:
            # in a PREPARED line a newline means exactly one thing — "a
            # different speaker starts here" (AICP-247 merges, see
            # _join_merged_text) — and layout_text honours it as a row
            # break. Latin wrapping already treated a newline as a space, so
            # this changes nothing there.
            text = " ".join(
                seg.strip() for seg in str(item.get("text") or "").splitlines() if seg.strip()
            )
            if not text:
                logger.warning("prepare_lines: dropping line %d — blank text", index)
                continue

            start = _as_seconds(item.get("start"))
            if start is None:
                logger.warning(
                    "prepare_lines: dropping line %d (%r) — unparseable start=%r",
                    index, _snippet(text), item.get("start"),
                )
                continue

            end = _as_seconds(item.get("end"))
            if end is None:
                end = start  # repairable: step 3 gives it a real duration.

            start = max(0.0, start)
            end = max(start, end)
            if bound is not None:
                start = min(start, bound)
                end = max(start, min(end, bound))

            cleaned.append({"start": start, "end": end, "text": text,
                            _SPEAKER_KEY: _speaker_of(item)})
        except Exception:
            # Belt-and-suspenders: a truly hostile item (e.g. text whose
            # __str__ raises) must still not take down the whole batch.
            logger.warning(
                "prepare_lines: dropping line %d — raised during cleaning",
                index, exc_info=True,
            )
            continue

    # --- 3: per-line minimum duration -----------------------------------
    with_min_duration: list = []
    for line in cleaned:
        duration = line["end"] - line["start"]
        if duration >= MIN_LINE_DURATION_SEC:
            with_min_duration.append(line)
            continue

        candidate_end = line["start"] + MIN_LINE_DURATION_SEC
        if bound is not None:
            candidate_end = min(candidate_end, bound)

        # Compared as `end < start + floor`, NOT `end - start < floor`: with
        # no clip-end clamp candidate_end IS start + floor, and the
        # subtraction form rounds below the floor for ~2% of starts (0.44 ->
        # 0.4999999...), which silently dropped those lines (AICP-247).
        if candidate_end < line["start"] + MIN_LINE_DURATION_SEC:
            if candidate_end > line["start"]:
                # Too close to the clip end to stand on its own, but there is
                # SOME room: step 5 may still show it by merging it into the
                # caption just before it (AICP-247). Only if it has no such
                # neighbour is it dropped, there.
                line["end"] = candidate_end
                line[_TAIL_SHORT_KEY] = True
                with_min_duration.append(line)
                continue
            logger.warning(
                "prepare_lines: dropping line %r (start=%.3f end=%.3f) — "
                "%.3fs from a %ss floor and there is no room left before "
                "duration_sec=%s to extend it",
                _snippet(line["text"]), line["start"], line["end"],
                duration, MIN_LINE_DURATION_SEC, bound,
            )
            continue

        line["end"] = candidate_end
        with_min_duration.append(line)

    # --- 4: sort by start (stable) ---------------------------------------
    with_min_duration.sort(key=lambda ln: ln["start"])

    # --- 5: de-overlap: truncate the earlier line, or merge (AICP-247) ----
    accepted: list = []
    for line in with_min_duration:
        prev = accepted[-1] if accepted else None

        if line.get(_TAIL_SHORT_KEY):
            # Cannot reach the floor before the clip ends. Shown only by
            # joining the caption before it, and only when it follows that
            # caption closely (starts while it is up, or within one floor of
            # its end) — an isolated flash at the cut is still dropped.
            if prev is not None and line["start"] < prev["end"] + MIN_LINE_DURATION_SEC:
                _merge_into(prev, line, need=need, bound=bound,
                            reason="no room for the floor before the clip end")
            else:
                logger.warning(
                    "prepare_lines: dropping line %r (start=%.3f end=%.3f) — "
                    "under the %ss floor before duration_sec and no caption "
                    "close enough before it to merge into",
                    _snippet(line["text"]), line["start"], line["end"],
                    MIN_LINE_DURATION_SEC,
                )
            continue

        if prev is not None and line["start"] < prev["end"]:
            truncated_end = line["start"]
            if truncated_end - prev["start"] < need(prev["text"]):
                # Truncating would leave `prev` under the floor (or, with
                # layout info, less than its rendered captions need). v1/v2
                # dropped it here — real dialogue lost whenever two quick
                # replies follow each other. Merge instead: one caption from
                # prev's start to the later of the two ends (held to what the
                # merged text needs, never past the clip). No caption starts
                # earlier than its first line, and prev's start is unchanged,
                # so it still cannot overlap the caption before it — no
                # cascade back through `accepted`.
                _merge_into(prev, line, need=need, bound=bound,
                            reason="truncating the earlier one would leave it "
                            "too short to read")
                continue
            logger.warning(
                "prepare_lines: truncating line %r end %.3f -> %.3f to avoid "
                "overlapping the next line (starts at %.3f) — captions never "
                "shift a later line's start",
                _snippet(prev["text"]), prev["end"], truncated_end, line["start"],
            )
            prev["end"] = truncated_end
        accepted.append(line)

    return [{"start": ln["start"], "end": ln["end"], "text": ln["text"]} for ln in accepted]


#: Private bookkeeping keys on the working dicts inside prepare_lines; never
#: returned (the output shape stays exactly start/end/text).
_SPEAKER_KEY = "_speaker"
_TAIL_SHORT_KEY = "_tail_short"


def _speaker_of(item: dict) -> str:
    """The line's ``speaker`` label, normalised for comparison; ``""`` when
    absent or unreadable (unknown speakers never force a row break)."""
    try:
        return " ".join(str(item.get("speaker") or "").split()).casefold()
    except Exception:
        return ""


def _join_merged_text(left: str, right: str, *, left_speaker: str,
                      right_speaker: str) -> str:
    """Text of two lines merged into one caption (AICP-247).

    A newline between them when they are different known speakers and
    :data:`SPEAKER_CHANGE_STARTS_NEW_ROW` is on (:func:`layout_text` turns it
    into a row break when it fits); otherwise a space, or nothing between two
    CJK texts, which are written without spaces.
    """
    if (SPEAKER_CHANGE_STARTS_NEW_ROW and left_speaker and right_speaker
            and left_speaker != right_speaker):
        return f"{left}\n{right}"
    return _join_parts([left, right])


def _join_parts(parts: List[str]) -> str:
    """Join caption pieces with a space, except between two CJK pieces (CJK
    is written without spaces). Decided per pair, not per job, so Latin
    replies inside a Japanese job keep their spaces."""
    out = ""
    for part in parts:
        if not part:
            continue
        if out:
            # The two characters that would touch decide, so mixed-script
            # words ("日本hello" + "world日本") keep their space.
            out += "" if _joins_without_space(out[-1], part[0]) else " "
        out += part
    return out


def _join_class(char: str) -> str:
    """``"cjk"`` for CJK letters plus CJK/fullwidth punctuation ("。", "？",
    which :func:`_looks_cjk` deliberately ignores), ``"fw"`` for fullwidth
    Latin letters and digits, ``""`` for everything else."""
    cp = ord(char)
    if 0xFF10 <= cp <= 0xFF19 or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return "fw"
    if _looks_cjk(char) or 0x3000 <= cp <= 0x303F or 0xFF00 <= cp <= 0xFFEF:
        return "cjk"
    return ""


def _joins_without_space(left: str, right: str) -> bool:
    """True when two touching characters run on without a space: both
    CJK-class, or a fullwidth letter/digit against a CJK one ("第１" + "回").
    Two fullwidth Latin words ("ＨＥＬＬＯ" + "ＷＯＲＬＤ") and anything
    involving ASCII keep the space."""
    classes = {_join_class(left), _join_class(right)}
    return "" not in classes and classes != {"fw"}


def _merge_into(prev: dict, line: dict, *, need: Callable[[str], float],
                bound: Optional[float], reason: str) -> None:
    """Fold *line* into the already-accepted caption *prev*, in place.

    The merged caption ends at the later of the two ends, or later still if
    its combined text needs more time than that to be shown in full
    (``need``) — but never past *bound*. Both ends are already clamped, so
    the max of them is too.
    """
    text = _join_merged_text(
        prev["text"], line["text"],
        left_speaker=prev.get(_SPEAKER_KEY, ""), right_speaker=line.get(_SPEAKER_KEY, ""),
    )
    end = max(prev["end"], line["end"])
    wanted = prev["start"] + need(text)
    if bound is not None:
        wanted = min(wanted, bound)
    end = max(end, wanted)
    logger.warning(
        "prepare_lines: merging line %r (start=%.3f end=%.3f) into the caption "
        "before it %r (start=%.3f end=%.3f) — %s; shown together until %.3f",
        _snippet(line["text"]), line["start"], line["end"],
        _snippet(prev["text"]), prev["start"], prev["end"], reason, end,
    )
    prev["text"] = text
    prev["end"] = end
    prev[_SPEAKER_KEY] = line.get(_SPEAKER_KEY, "")


# ---------------------------------------------------------------------------
# ASS assembly
# ---------------------------------------------------------------------------


def _script_info_section(width: int, height: int) -> str:
    """The ``[Script Info]`` header block.

    ``WrapStyle: 0`` ("smart wrapping") is the one non-obvious value here,
    and it is a v2 reversal: v1 set ``2`` ("no word wrapping") to stop libass
    re-flowing rows this module had already chosen. That reasoning holds only
    while our rows are right — and v1's, wrapped by character count, were
    routinely wider than the frame (see the module docstring). Turning
    wrapping off did not make them fit; it removed the last thing that could
    have. With rows now measured against this frame's own geometry, libass
    agreeing with us is the normal case and re-wrapping means our measurement
    was wrong — in which case an extra row is a far better outcome than text
    running off both edges.
    """
    return (
        "[Script Info]\n"
        "; Generated by app.captions (see docs/burned-captions.md §3.1).\n"
        f"; {CAPTION_STYLE_VERSION} — one universal style for every caption,\n"
        "; every language; only the font varies (font_for_language()).\n"
        "Title: Burned captions\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {int(width)}\n"
        f"PlayResY: {int(height)}\n"
    )


def _num(value: float) -> str:
    """Format a sub-pixel-precision style value (Outline, Shadow) tersely.

    ``%g`` drops a trailing ``.0`` (so an outline that lands on an exact
    pixel prints ``2`` rather than ``2.0``) while still keeping up to two
    meaningful decimal places for the common case where a ratio times a real
    video's height doesn't land on a whole pixel.
    """
    return f"{round(float(value), 2):g}"


def _styles_section(font: str, *, width: int, height: int,
                    placement: Optional[Placement] = None) -> str:
    """
    The ``[V4+ Styles]`` block: exactly one style, ``Default``.

    Every size ratio (see the constants section above) is multiplied by
    *height* here except the horizontal margin, which is multiplied by
    *width* — this is the one place in the module where "everything is a
    ratio of the frame" turns into actual pixel numbers, so it is also where
    a future tuner would come to change the ratios' effect without touching
    the ASS-syntax plumbing in this function.

    Fontsize/MarginL/MarginR/MarginV are rounded to whole pixels (the
    conventional unit for those fields); Outline/Shadow keep sub-pixel
    precision via :func:`_num` since a fractional box padding is a normal,
    supported style tweak and rounding it away measurably changes how the
    band sits around the text at small caption sizes.

    Two fields mean something different from what their names suggest under
    :data:`BORDER_STYLE_BOX`, and both are load-bearing for the v2 look:
    ``OutlineColour`` fills the box (so it carries :data:`BOX_ALPHA`, not the
    opaque black an "outline colour" would imply) and ``Outline`` is the
    box's padding rather than a border width.

    With a *placement* (AICP-368) only three fields change: Alignment becomes
    :data:`ALIGNMENT_MIDDLE_CENTER` (every event then carries a ``\\pos``,
    :func:`_events_section`), MarginL/R narrow to the placement's width, and
    MarginV is 0 (unused under ``\\pos``). Without one the line is
    byte-for-byte the house style.
    """
    font_size = font_size_px(height)
    padding = height * BOX_PADDING_RATIO
    shadow = height * SHADOW_RATIO
    if placement is None:
        alignment = ALIGNMENT_BOTTOM_CENTER
        margin_v = int(round(height * MARGIN_V_RATIO))
        margin_h = int(round(width * MARGIN_H_RATIO))
    else:
        alignment = ALIGNMENT_MIDDLE_CENTER
        margin_v = 0
        margin_h = margin_h_px(width, placement.max_width)

    fields = [
        "Default",  # Name
        font,  # Fontname
        str(font_size),  # Fontsize
        _COLOUR_WHITE_OPAQUE,  # PrimaryColour — the text fill
        _COLOUR_WHITE_OPAQUE,  # SecondaryColour — unused (no karaoke here)
        _COLOUR_BOX,  # OutlineColour — under BorderStyle 3, the box FILL
        _COLOUR_BLACK_OPAQUE,  # BackColour — the drop shadow
        str(_BOLD_ON),  # Bold — -1, not 1 (see module docstring)
        str(_BOLD_OFF),  # Italic
        str(_BOLD_OFF),  # Underline
        str(_BOLD_OFF),  # StrikeOut
        "100",  # ScaleX %
        "100",  # ScaleY %
        "0",  # Spacing
        "0",  # Angle
        str(BORDER_STYLE_BOX),  # BorderStyle — 3 = filled box behind the text
        _num(padding),  # Outline — box padding under BorderStyle 3
        _num(shadow),  # Shadow distance
        str(alignment),  # Alignment — bottom-centred (house) / middle (placed)
        str(margin_h),  # MarginL
        str(margin_h),  # MarginR
        str(margin_v),  # MarginV
        "1",  # Encoding
    ]
    return (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {','.join(fields)}\n"
    )


def _split_windows(start: float, end: float, captions: list) -> list:
    """
    Divide ``[start, end]`` between *captions* in proportion to their length.

    A line too long for one caption is shown as several in sequence, and the
    only timing information we have is the one window the transcript gave for
    the whole utterance. Splitting it by character count (rather than evenly)
    approximates the speaker: more words take longer to say. It is an
    approximation either way — the alternative, word-level timings, is a
    different transcription contract (docs/burned-captions.md §4).

    Returns one ``(start, end)`` per caption. Captions that cannot be given
    :data:`MIN_LINE_DURATION_SEC` are dropped from the tail rather than
    flashed unreadably (since AICP-247 :func:`build_ass` sizes windows so
    this happens only when the text needs more floor-time than is left
    before the clip ends): a window too short for the text in it means the
    transcript is wrong about one or the other, and half a second is the
    floor this module reads as "shown at all".
    """
    span = max(0.0, end - start)
    # The epsilon keeps a window prepare_lines sized as exactly k floors
    # (start + k * 0.5, which can subtract back to 0.4999...) at k captions.
    room = (int((span + 1e-6) // MIN_LINE_DURATION_SEC) if MIN_LINE_DURATION_SEC > 0
            else len(captions))
    keep = max(1, min(len(captions), room))
    if keep < len(captions):
        logger.warning(
            "captions: %.2fs window for %d caption(s) of a split line only "
            "fits %d at the %ss floor — dropping the last %d",
            span, len(captions), keep, MIN_LINE_DURATION_SEC, len(captions) - keep,
        )
        captions = captions[:keep]

    weights = [max(1, len(" ".join(caption))) for caption in captions]
    total = float(sum(weights))
    # Every window gets the floor first and only the time beyond that is
    # shared by length — a purely proportional split could hand a one-word
    # tail caption a fraction of the floor (AICP-247). `keep` was chosen so
    # the floors fit (spare >= 0), single-caption windows are unaffected.
    floor = MIN_LINE_DURATION_SEC if len(weights) > 1 else 0.0
    spare = max(0.0, span - floor * len(weights))
    windows = []
    cursor = start
    for index, weight in enumerate(weights):
        # The last window ends exactly on `end` rather than on an accumulated
        # sum of floats, so a split line never overruns the utterance (and,
        # chained with prepare_lines' de-overlap, never overruns the video).
        stop = end if index == len(weights) - 1 else cursor + floor + spare * weight / total
        windows.append((cursor, stop))
        cursor = stop
    return windows


def placement_anchor(width: int, height: int,
                     placement: Placement) -> tuple[int, int]:
    """The ``\\pos`` point for a placed caption: horizontally the frame
    centre (the column is symmetric), vertically ``center_y`` of the frame
    height, both whole pixels in PlayRes coordinates."""
    return int(round(int(width) / 2.0)), int(round(int(height) * float(placement.center_y)))


def _caption_box(placement: Optional[Placement]) -> tuple:
    """``(max_rows, max_width)`` a caption is laid out in: the house style,
    or a :class:`Placement`'s (rows capped at :data:`PLACEMENT_MAX_ROWS`)."""
    if placement is None:
        return MAX_ROWS, None
    return max(1, min(PLACEMENT_MAX_ROWS, int(placement.max_rows))), placement.max_width


def _events_section(lines: list, *, cjk: bool, width: int, height: int,
                    font: str, placement: Optional[Placement] = None) -> str:
    """The ``[Events]`` block: the ``Format:`` line plus the ``Dialogue:``
    rows. *lines* must already be the output of :func:`prepare_lines` — this
    function does no further validation, it only lays out and renders.

    One prepared line becomes **one or more** Dialogue rows: :func:`layout_text`
    returns as many :data:`MAX_ROWS`-tall captions as the text needs, and
    :func:`_split_windows` hands each of them a slice of the line's own time
    window."""
    rows = [
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    ]
    max_rows, max_width = _caption_box(placement)
    prefix = ""
    if placement is not None:
        px, py = placement_anchor(width, height, placement)
        prefix = f"{{\\pos({px},{py})}}"
    for line in lines:
        captions = layout_text(
            line["text"], cjk=cjk, width=width, height=height, font=font,
            max_rows=max_rows, max_width=max_width,
        )
        if not captions:
            continue
        # zip is the mechanism by which a caption _split_windows refused to
        # give a readable window to is dropped: it returns fewer windows than
        # captions in exactly that case.
        for caption, (start, end) in zip(
            captions, _split_windows(line["start"], line["end"], captions)
        ):
            text = escape_text("\n".join(caption))
            start_ts = format_timestamp(start)
            end_ts = format_timestamp(end)
            rows.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{prefix}{text}\n")
    return "".join(rows)


def build_ass(
    lines: Iterable[dict],
    *,
    width: int,
    height: int,
    language: Optional[str] = None,
    duration_sec: Optional[float] = None,
    placement: Optional[Placement] = None,
) -> str:
    """
    Build a complete ``.ass`` subtitle file from timed transcript lines.

    Runs *lines* through :func:`prepare_lines` first (so callers never have
    to remember to sanitise input before calling this — untrusted lines
    straight from a transcript are the normal case, not the exception), then
    emits ``[Script Info]`` (with ``PlayResX``/``PlayResY`` set to *width*/
    *height* — this is the frame size every ratio in the house style is
    measured against, so getting it right is what makes a font-size-as-a-
    fraction-of-height actually come out at the intended fraction of the
    real delivered video), ``[V4+ Styles]`` (one style, ``Default``, per the
    house style constants above), and ``[Events]`` (one ``Dialogue:`` row per
    surviving line).

    The font — the only per-language knob (see the module docstring) — is
    resolved *once* for the whole file via :func:`font_for_language`, using
    every surviving line's text joined together as the *sample_text*
    fallback. That means a run whose stored ``language`` is missing or wrong
    still gets the CJK font if what it actually transcribed is Japanese,
    which matters because a caption in the wrong font renders as tofu boxes
    without ffmpeg raising anything (docs/burned-captions.md §3.1) — nobody
    would find out until an operator watched the output.

    That same font decision also decides ``cjk`` for :func:`wrap_rows`,
    applied uniformly to every line in the file rather than sniffed per
    line. This is a deliberate simplification: one caption job has one
    language and one font, so a transcript's odd Latin-only line (a brand
    name inside an otherwise-Japanese transcript) still tokenises the CJK
    way for consistency with the rest of the file, rather than switching
    wrap strategy mid-file for one line.

    A prepared line does not map one-to-one onto a ``Dialogue:`` row: text
    too long for :data:`MAX_ROWS` rows at this frame size is shown as
    successive captions dividing the line's own time window
    (:func:`_events_section`), so the emitted row count is ``>=`` the
    surviving line count.

    Args:
        lines: See :func:`prepare_lines` — any iterable of dict-like lines,
            trusted or not.
        width, height: The delivered video's frame size, in pixels. Every
            house-style ratio (font size, outline, margins, ...) is measured
            against these.
        language: Passed straight to :func:`font_for_language`.
        duration_sec: Passed straight to :func:`prepare_lines`.
        placement: AICP-368 — centre every caption on ``placement.center_y``
            (middle alignment + a ``\\pos`` per event), wrap to its width and
            allow up to its row count. None (the default) is byte-for-byte
            the house style.

    Returns:
        A complete, syntactically valid ``.ass`` file as a string — even
        when *lines* is empty or entirely unusable, in which case the result
        is a header-only file with an ``[Events]`` section that has the
        ``Format:`` line but zero ``Dialogue:`` rows. That is a deliberate,
        documented outcome (a wordless creative burns to a caption-free
        re-encode, docs/burned-captions.md §3.3), not an error — this
        function never raises.
    """
    # Materialised once: *lines* may be a one-shot iterator and is prepared
    # twice. Anything not iterable is "no lines", as prepare_lines treats it.
    try:
        lines = list(lines or [])
    except Exception:
        lines = []

    # Pass 1 only resolves the font (and so the wrap) from the surviving
    # text. Pass 2 de-overlaps knowing how many captions each text lays out
    # into, so a truncation or merge never leaves a caption less time than
    # its rows need (AICP-247). The windows must be sized in the font that
    # renders them, so pass 2 is repeated if its own text resolves to a
    # different font — only ever Latin -> CJK (a CJK tail line only a longer
    # merged caption reaches, with no language given), never back, so this
    # ends after at most one repeat. The CJK font carries Latin glyphs too.
    def prepare_for(font_name: str) -> list:
        is_cjk = font_name == CJK_FONT
        max_rows, max_width = _caption_box(placement)

        @functools.lru_cache(maxsize=1024)
        def caption_time(text: str) -> float:
            captions = layout_text(text, cjk=is_cjk, width=width, height=height,
                                   font=font_name, max_rows=max_rows,
                                   max_width=max_width)
            return MIN_LINE_DURATION_SEC * max(1, len(captions))

        return prepare_lines(lines, duration_sec=duration_sec,
                             min_duration_for=caption_time)

    def font_of(prepared_lines: list) -> str:
        return font_for_language(
            language, sample_text=" ".join(line["text"] for line in prepared_lines))

    font = font_of(prepare_lines(lines, duration_sec=duration_sec))
    prepared = prepare_for(font)
    if font != CJK_FONT and font_of(prepared) == CJK_FONT:
        font = CJK_FONT
        prepared = prepare_for(font)
    cjk = font == CJK_FONT

    return (
        _script_info_section(width, height)
        + "\n"
        + _styles_section(font, width=width, height=height, placement=placement)
        + "\n"
        + _events_section(prepared, cjk=cjk, width=width, height=height, font=font,
                          placement=placement)
    )
