"""Tests for app/languages.py — the run-language registry.

These pin the CONTRACT with the downstream renaming pipeline (repo
``ai-fb-audit``, ``scripts/rename_creatives_from_final_creatives.py``), which
parses the language back out of the sheet's Job ID cell. Every assertion here
mirrors a rule in that parser; if one of them has to change, the change belongs
downstream first.
"""

import pytest

from app import languages


# The exact set the downstream LANG_MAP recognises. A sixth code added here
# without teaching that map would silently rename as English.
def test_registry_is_exactly_the_five_supported_languages():
    assert set(languages.LANGUAGES) == {"en", "es", "pt", "ja", "de"}


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
     ("de", "1736426f-de")],
)
def test_non_english_suffixes_are_the_canonical_trailing_form(code, expected):
    assert languages.apply_suffix("1736426f", code) == expected


def test_normalize_accepts_sloppy_input_and_rejects_unknown():
    assert languages.normalize("  JA ") == "ja"
    assert languages.normalize("fr") is None      # real language, not supported
    assert languages.normalize("") is None
    assert languages.normalize(None) is None


def test_label_for_defaults_to_english():
    assert languages.label_for("ja") == "Japanese"
    assert languages.label_for(None) == "English"
    assert languages.label_for("fr") == "English"


def test_job_id_for_truncates_then_suffixes():
    full = "1736426f-aaaa-bbbb-cccc-ddddeeeeffff"
    assert languages.job_id_for(full, "ja") == "1736426f-ja"
    assert languages.job_id_for(full, None) == "1736426f"
    # Caption Demo variant: dashes stripped by the caller, 16-char window.
    assert (
        languages.job_id_for(full.replace("-", ""), "ja", length=16)
        == "1736426faaaabbbb-ja"
    )


def test_strip_suffix_round_trips_and_leaves_plain_ids_alone():
    assert languages.strip_suffix("1736426f-ja") == "1736426f"
    assert languages.strip_suffix("1736426f") == "1736426f"
    # A full uuid must survive untouched — its last group is 12 hex chars, so
    # the anchored two-char suffix pattern can never match it.
    full = "1736426f-aaaa-bbbb-cccc-ddddeeeeffff"
    assert languages.strip_suffix(full) == full

