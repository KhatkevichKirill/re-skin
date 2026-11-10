"""Tests for app/languages.py — the run-language registry.

These pin the CONTRACT with the downstream renaming pipeline (repo
``ai-fb-audit``, ``scripts/rename_creatives_from_final_creatives.py``), which
parses the language back out of the sheet's Job ID cell. Every assertion here
mirrors a rule in that parser; if one of them has to change, the change belongs
downstream first.
"""

import pytest

from app import languages
from app.models import Run
from app.web_v2 import _sanitize_run_id_prefix


# The exact selectable set. Every code here is also in the downstream LANG_MAP
# (see TestDownstreamContract below and wiki/components/run-language.md). A
# code added here that the downstream does not know would silently rename as
# English.
def test_registry_is_exactly_the_eight_supported_languages():
    assert set(languages.LANGUAGES) == {"en", "es", "pt", "ja", "de", "it", "pl", "fr"}


def test_registry_order_is_stable_and_appends_french_last():
    """The dropdown renders in insertion order — French joins after Polish."""
    assert list(languages.LANGUAGES) == ["en", "es", "pt", "ja", "de", "it", "pl", "fr"]


def test_english_carries_no_suffix():
    """The parser strips only -es/-pt/-de/-ja; '-en' would leak into the core."""
    assert languages.suffix_for("en") == ""
    assert languages.apply_suffix("1736426f", "en") == "1736426f"


def test_unset_language_matches_pre_feature_behaviour():
    """NULL must produce the bare id every row exported before this shipped has."""
    assert languages.apply_suffix("1736426f", None) == "1736426f"
    assert languages.apply_suffix("1736426f", "") == "1736426f"


@pytest.mark.parametrize(
    "code,expected",
    [("es", "1736426f-es"), ("pt", "1736426f-pt"), ("ja", "1736426f-ja"),
     ("de", "1736426f-de"), ("it", "1736426f-it"), ("pl", "1736426f-pl"),
     ("fr", "1736426f-fr")],
)
def test_non_english_suffixes_are_the_canonical_trailing_form(code, expected):
    assert languages.apply_suffix("1736426f", code) == expected


def test_normalize_accepts_sloppy_input_and_rejects_unknown():
    assert languages.normalize("  JA ") == "ja"
    assert languages.normalize("  IT ") == "it"
    assert languages.normalize("  PL ") == "pl"
    assert languages.normalize("  FR ") == "fr"
    assert languages.normalize("nl") is None      # real language, not supported
    assert languages.normalize("") is None
    assert languages.normalize(None) is None


@pytest.mark.parametrize("alias", ["ita", "it-IT", "it_IT", "italian", "IT-it"])
def test_italian_aliases_are_not_accepted(alias):
    """Only the canonical ISO 639-1 code is a language; no locale variants.

    The column is VARCHAR(8), so "it-IT" would *store* fine — it must be
    rejected at the registry instead, or the downstream would see "-it-IT".
    """
    assert languages.normalize(alias) is None
    assert not languages.is_valid(alias)


def test_label_for_defaults_to_english():
    assert languages.label_for("ja") == "Japanese"
    assert languages.label_for("it") == "Italian"
    assert languages.label_for("  IT ") == "Italian"
    assert languages.label_for("pl") == "Polish"
    assert languages.label_for("  PL ") == "Polish"
    assert languages.label_for("fr") == "French"
    assert languages.label_for(None) == "English"
    assert languages.label_for("nl") == "English"


def test_italian_suffix_is_the_canonical_trailing_form():
    assert languages.suffix_for("it") == "-it"
    assert languages.suffix_for("  IT ") == "-it"


def test_job_id_for_truncates_then_suffixes():
    full = "1736426f-aaaa-bbbb-cccc-ddddeeeeffff"
    assert languages.job_id_for(full, "ja") == "1736426f-ja"
    assert languages.job_id_for(full, "it") == "1736426f-it"
    assert (
        languages.job_id_for(full.replace("-", ""), "it", length=16)
        == "1736426faaaabbbb-it"
    )
    assert languages.job_id_for(full, None) == "1736426f"
    # A caller may pass a wider prefix window than the Final Creatives default.
    assert (
        languages.job_id_for(full.replace("-", ""), "ja", length=16)
        == "1736426faaaabbbb-ja"
    )


@pytest.mark.parametrize("code", ["es", "pt", "ja", "de", "it", "pl", "fr"])
def test_apply_then_strip_round_trips_for_every_suffixed_language(code):
    assert languages.strip_suffix(languages.apply_suffix("1736426f", code)) == "1736426f"


def test_strip_suffix_round_trips_and_leaves_plain_ids_alone():
    assert languages.strip_suffix("1736426f-ja") == "1736426f"
    assert languages.strip_suffix("1736426f-it") == "1736426f"
    assert languages.strip_suffix("1736426f-IT") == "1736426f"
    assert languages.strip_suffix("  1736426f-it  ") == "1736426f"
    assert languages.strip_suffix("1736426f") == "1736426f"
    # The composite id a run's Nth overlay exports under.
    assert languages.strip_suffix("1736426f_a1b2c3d4-it") == "1736426f_a1b2c3d4"
    # A full uuid must survive untouched — its last group is 12 hex chars, so
    # the anchored two-char suffix pattern can never match it.
    full = "1736426f-aaaa-bbbb-cccc-ddddeeeeffff"
    assert languages.strip_suffix(full) == full


def test_run_short_id_is_prefix_plus_language():
    run = Run(id="1736426f-aaaa-bbbb-cccc-ddddeeeeffff", language="ja")
    assert run.short_id == "1736426f-ja"

    run.language = "en"
    assert run.short_id == "1736426f"

    run.language = None
    assert run.short_id == "1736426f"


def test_dashboard_filter_accepts_the_suffixed_id_operators_see():
    """Pasting '1736426f-ja' into the Run-ID filter must match run 1736426f…

    'ja' is not hex, so without stripping the suffix first the sanitizer would
    degrade it to '1736426f-a' and the LIKE prefix would match nothing.
    """
    assert _sanitize_run_id_prefix("1736426f-ja") == "1736426f"
    assert _sanitize_run_id_prefix("1736426F-JA") == "1736426f"
    assert _sanitize_run_id_prefix("1736426f") == "1736426f"
    # Wildcards still can't slip through into the LIKE pattern.
    assert "%" not in _sanitize_run_id_prefix("1736426f%-ja")
    assert "_" not in _sanitize_run_id_prefix("1736426f_-ja")


def test_run_short_id_carries_the_italian_suffix():
    run = Run(id="1736426f-aaaa-bbbb-cccc-ddddeeeeffff", language="it")
    assert run.short_id == "1736426f-it"


def test_dashboard_filter_accepts_the_italian_suffixed_id():
    """'it' is not hex either — without stripping, '1736426f-it' would degrade
    to '1736426f-' and the LIKE prefix would match nothing."""
    assert _sanitize_run_id_prefix("1736426f-it") == "1736426f"
    assert _sanitize_run_id_prefix("1736426F-IT") == "1736426f"
    assert _sanitize_run_id_prefix("  1736426f-It  ") == "1736426f"
    # Wildcards still can't slip through into the LIKE pattern.
    assert "%" not in _sanitize_run_id_prefix("1736426f%-it")
    assert "_" not in _sanitize_run_id_prefix("1736426f_-it")


def test_a_full_uuid_is_never_eaten_by_the_italian_suffix():
    """The suffix rule is anchored and two chars wide, so no real id can lose a
    tail to it: uuid groups are hex (no 'i'/'t') and 12 chars long, and an
    8-char prefix has no hyphen at all."""
    full = "1736426f-aaaa-bbbb-cccc-ddddeeeeffff"
    assert languages.strip_suffix(full) == full
    assert _sanitize_run_id_prefix(full) == full
    # A hex-only id that merely CONTAINS 'it'-like text cannot exist, but check
    # the neighbouring shapes an operator might paste.
    assert languages.strip_suffix("1736426fit") == "1736426fit"
    assert languages.strip_suffix("1736426f-i") == "1736426f-i"
    assert languages.strip_suffix("1736426f-ite") == "1736426f-ite"


@pytest.mark.parametrize("alias", ["pol", "pl-PL", "pl_PL", "polish", "PL-pl"])
def test_polish_aliases_are_not_accepted(alias):
    """Same rule as Italian: only the canonical ISO 639-1 code is a language."""
    assert languages.normalize(alias) is None
    assert not languages.is_valid(alias)


def test_polish_suffix_is_the_canonical_trailing_form():
    assert languages.suffix_for("pl") == "-pl"
    assert languages.suffix_for("  PL ") == "-pl"
    assert languages.job_id_for("1736426f-aaaa-bbbb-cccc-ddddeeeeffff", "pl") == "1736426f-pl"


def test_strip_suffix_removes_polish():
    assert languages.strip_suffix("1736426f-pl") == "1736426f"
    assert languages.strip_suffix("1736426f-PL") == "1736426f"
    assert languages.strip_suffix("  1736426f-pl  ") == "1736426f"
    assert languages.strip_suffix("1736426f_a1b2c3d4-pl") == "1736426f_a1b2c3d4"
    # Anchored and two chars wide, like every other suffix.
    assert languages.strip_suffix("1736426f-p") == "1736426f-p"
    assert languages.strip_suffix("1736426f-plx") == "1736426f-plx"
    assert languages.strip_suffix("1736426fpl") == "1736426fpl"


def test_run_short_id_carries_the_polish_suffix():
    run = Run(id="1736426f-aaaa-bbbb-cccc-ddddeeeeffff", language="pl")
    assert run.short_id == "1736426f-pl"


def test_dashboard_filter_accepts_the_polish_suffixed_id():
    """'pl' is not hex — without stripping, '1736426f-pl' would degrade to
    '1736426f-' and the LIKE prefix would match nothing."""
    assert _sanitize_run_id_prefix("1736426f-pl") == "1736426f"
    assert _sanitize_run_id_prefix("1736426F-PL") == "1736426f"
    assert _sanitize_run_id_prefix("  1736426f-Pl  ") == "1736426f"
    assert "%" not in _sanitize_run_id_prefix("1736426f%-pl")
    assert "_" not in _sanitize_run_id_prefix("1736426f_-pl")


@pytest.mark.parametrize("alias", ["fra", "fre", "fr-FR", "fr_CA", "french", "FR-fr"])
def test_french_aliases_are_not_accepted(alias):
    """Same rule as Italian: only the canonical ISO 639-1 code is a language."""
    assert languages.normalize(alias) is None
    assert not languages.is_valid(alias)


def test_french_suffix_is_the_canonical_trailing_form():
    assert languages.suffix_for("fr") == "-fr"
    assert languages.suffix_for("  FR ") == "-fr"
    assert languages.job_id_for("1736426f-aaaa-bbbb-cccc-ddddeeeeffff", "fr") == "1736426f-fr"


def test_strip_suffix_removes_french():
    assert languages.strip_suffix("1736426f-fr") == "1736426f"
    assert languages.strip_suffix("1736426f-FR") == "1736426f"
    assert languages.strip_suffix("  1736426f-fr  ") == "1736426f"
    assert languages.strip_suffix("1736426f_a1b2c3d4-fr") == "1736426f_a1b2c3d4"
    # Anchored and two chars wide, like every other suffix.
    assert languages.strip_suffix("1736426f-f") == "1736426f-f"
    assert languages.strip_suffix("1736426f-fra") == "1736426f-fra"
    assert languages.strip_suffix("1736426ffr") == "1736426ffr"


def test_run_short_id_carries_the_french_suffix():
    run = Run(id="1736426f-aaaa-bbbb-cccc-ddddeeeeffff", language="fr")
    assert run.short_id == "1736426f-fr"


def test_dashboard_filter_accepts_the_french_suffixed_id():
    """'r' is not hex — without stripping, '1736426f-fr' would degrade to
    '1736426f-f' and the LIKE prefix would match nothing."""
    assert _sanitize_run_id_prefix("1736426f-fr") == "1736426f"
    assert _sanitize_run_id_prefix("1736426F-FR") == "1736426f"
    assert _sanitize_run_id_prefix("  1736426f-Fr  ") == "1736426f"
    assert "%" not in _sanitize_run_id_prefix("1736426f%-fr")
    assert "_" not in _sanitize_run_id_prefix("1736426f_-fr")


# Codes the downstream parser (ai-fb-audit) recognises, as verified read-only
# at ai-fb-audit main c680e61 on 2026-10-20 (LANG_MAP / _LANG_MAP, trailing rule
# -(es|pt|de|ja|sv|ar|fr|it|pl|zh|ko|kr)$; its real parse_job_id maps
# "1736426f-fr" to ("FR", "1736426f") and "1736426f-pl" to ("PL", "1736426f")).
# Italian joined via PR #178 (merge 503695bc, 2026-09-08), Polish via AICP-181
# (2026-10-08); French was there before it was selectable here. This is a
# snapshot of another repo, not something this repo can enforce — update it
# when the downstream changes, never to make a new language "pass".
DOWNSTREAM_RECOGNISED_AT_C680E61 = {
    "en", "es", "pt", "ja", "de", "it", "sv", "ar", "fr", "pl", "zh", "ko",
}


def test_italian_is_parseable_downstream_since_pr_178():
    """IT used to be selectable-but-unexportable; downstream caught up."""
    assert "it" in DOWNSTREAM_RECOGNISED_AT_C680E61


def test_every_selectable_language_is_parseable_downstream():
    """No registry code is ahead of the downstream parser any more.

    A new code that fails this must not be exported until ai-fb-audit learns
    it (property 1 in app/languages.py) — fix the downstream, then the
    snapshot; never the other way round.
    """
    assert set(languages.LANGUAGES) - DOWNSTREAM_RECOGNISED_AT_C680E61 == set()


class TestDownstreamContract:
    """The exact strings ai-fb-audit's parse_job_id must read for IT, PL and FR.

    These pin OUR side of the cross-repo contract so a downstream change can be
    written and tested against them. All three are parsed downstream at
    ai-fb-audit ``c680e61`` (verified 2026-10-20 by running its real
    ``parse_job_id``) — see ``wiki/components/run-language.md``.
    """

    RUN = "1736426f-aaaa-bbbb-cccc-ddddeeeeffff"
    TAIL = "9c2a41b7-aaaa-bbbb-cccc-ddddeeeeffff"
    CAPTION = "b2c3d4e5-aaaa-bbbb-cccc-ddddeeeeffff"
    OVERLAY = "a1b2c3d4-aaaa-bbbb-cccc-ddddeeeeffff"

    @pytest.mark.parametrize("code", ["it", "pl", "fr"])
    def test_run_job_id(self, code):
        assert Run(id=self.RUN, language=code).short_id == f"1736426f-{code}"

    @pytest.mark.parametrize("code", ["it", "pl", "fr"])
    def test_derived_creatives_keep_their_own_prefix(self, code):
        assert languages.job_id_for(self.TAIL, code) == f"9c2a41b7-{code}"
        assert languages.job_id_for(self.CAPTION, code) == f"b2c3d4e5-{code}"

    @pytest.mark.parametrize("code", ["it", "pl", "fr"])
    def test_composite_overlay_core_survives_a_strip(self, code):
        """The downstream must strip the suffix and keep the '{run8}_{overlay8}' core."""
        composite = languages.apply_suffix(
            f"{self.RUN[:8]}_{self.OVERLAY[:8]}", code
        )
        assert composite == f"1736426f_a1b2c3d4-{code}"
        assert languages.strip_suffix(composite) == "1736426f_a1b2c3d4"
