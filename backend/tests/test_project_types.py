"""
Tests for app/project_types.py — the project-type registry.

The registry is a table of constants, so the tests here are about the
*invariants* the rest of the app relies on rather than about behaviour:

* every spec carries every field, and ``segmentation`` is one of the SEG_*
  values pipeline_v2.analyze_project dispatches on (a typo there would silently
  demote a type to the blank-slate branch);
* the localisation spec matches docs/localisation.md §2/§6 — those values are
  properties of the type, not operator preferences;
* the two Seedance templates require adaptive target-language lip sync and
  speech-related facial reactions (docs/localisation.md §4.3).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import ai_models, project_types
from app.project_types import (
    COVER_CHANGE,
    FACE_SWAP,
    LOCALISATION,
    PROJECT_TYPES,
    SEG_BLANK,
    SEG_DETECT_FACES,
    SEG_HOOK_SPLIT,
    SEG_SCENE_DETECT,
    SUBTITLE_REMOVAL,
    spec_for,
)

_SEGMENTATIONS = {SEG_DETECT_FACES, SEG_SCENE_DETECT, SEG_HOOK_SPLIT, SEG_BLANK}


# ---------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------


def test_every_type_is_keyed_by_its_own_key():
    for key, spec in PROJECT_TYPES.items():
        assert spec.key == key


@pytest.mark.parametrize("key", sorted(PROJECT_TYPES))
def test_segmentation_is_a_known_strategy(key):
    """A typo here would fall through to the blank-slate branch, silently."""
    assert PROJECT_TYPES[key].segmentation in _SEGMENTATIONS


def test_segmentation_replaced_the_two_booleans():
    """The old mutually-exclusive flags are gone (docs/localisation.md §5)."""
    spec = spec_for(FACE_SWAP)
    assert not hasattr(spec, "detect_faces")
    assert not hasattr(spec, "scene_detect")


def test_each_strategy_is_used_by_exactly_one_type():
    used = [spec.segmentation for spec in PROJECT_TYPES.values()]
    assert sorted(used) == sorted(_SEGMENTATIONS)


@pytest.mark.parametrize("key", sorted(PROJECT_TYPES))
def test_default_model_is_a_real_model(key):
    assert PROJECT_TYPES[key].default_model in ai_models.VALID_MODELS


@pytest.mark.parametrize("key", sorted(PROJECT_TYPES))
def test_default_audio_mode_is_one_of_the_two(key):
    assert PROJECT_TYPES[key].default_audio_mode in ("original", "seedance")


@pytest.mark.parametrize("key", sorted(PROJECT_TYPES))
def test_requires_audio_model_is_satisfied_by_the_default_model(key):
    """A type that filters on produces_audio must not default to a model the
    filter would hide."""
    spec = PROJECT_TYPES[key]
    if spec.requires_audio_model:
        assert ai_models.spec_for(spec.default_model).produces_audio


@pytest.mark.parametrize("key", sorted(PROJECT_TYPES))
def test_hint_and_label_are_present(key):
    spec = PROJECT_TYPES[key]
    assert spec.label.strip()
    assert spec.hint.strip()


def test_only_localisation_changes_the_audio_defaults():
    """§6: audio_mode "original" and mute_source False everywhere else."""
    for key, spec in PROJECT_TYPES.items():
        if key == LOCALISATION:
            continue
        assert spec.default_audio_mode == "original", key
        assert spec.default_mute_source is False, key
        assert spec.requires_audio_model is False, key


def test_pre_existing_specs_kept_their_strategy():
    assert spec_for(FACE_SWAP).segmentation == SEG_DETECT_FACES
    assert spec_for(SUBTITLE_REMOVAL).segmentation == SEG_SCENE_DETECT
    assert spec_for(COVER_CHANGE).segmentation == SEG_BLANK


def test_unknown_type_falls_back_to_face_swap():
    assert spec_for(None).key == FACE_SWAP
    assert spec_for("").key == FACE_SWAP
    assert spec_for("no-such-type").key == FACE_SWAP


# ---------------------------------------------------------------------------
# The localisation spec (docs/localisation.md §2, §6)
# ---------------------------------------------------------------------------


class TestLocalisationSpec:
    def test_registered_and_valid(self):
        assert LOCALISATION in project_types.VALID_PROJECT_TYPES
        assert project_types.label_for(LOCALISATION) == "Localisation"

    def test_pipeline_settings_are_properties_of_the_type(self):
        spec = spec_for(LOCALISATION)
        assert spec.segmentation == SEG_HOOK_SPLIT
        assert spec.default_model == "seedance-fast"
        # "original" would overlay the untranslated source track over the
        # localised video — the one setting that silently destroys the feature.
        assert spec.default_audio_mode == "seedance"
        assert spec.default_mute_source is True
        assert spec.requires_audio_model is True
        assert spec.uses_references is True

    def test_default_model_can_generate_audio_and_a_full_hook(self):
        model = ai_models.spec_for(spec_for(LOCALISATION).default_model)
        assert model.produces_audio
        # A 10s default hook has to fit in one clip on the default model.
        # Fast's hard ceiling is 15s — longer hooks need a split or Seedance 2.5.
        assert model.max_clip_sec >= 10.0
        assert model.max_clip_sec == 15.0

    def test_seedance_2_5_remains_an_optional_audio_model(self):
        """2.5 is no longer the default, but it stays registered and usable."""
        assert "seedance-2-5" in ai_models.VALID_MODELS
        assert ai_models.spec_for("seedance-2-5").produces_audio is True
        assert ai_models.max_clip_sec("seedance-2-5") == 30.0
        assert spec_for(LOCALISATION).default_model != "seedance-2-5"

    def test_gemini_omni_would_be_filtered_out(self):
        """The filter requires_audio_model drives has real teeth."""
        assert ai_models.spec_for("gemini-omni").produces_audio is False

    def test_other_project_type_defaults_are_unchanged(self):
        assert spec_for(FACE_SWAP).default_model == "seedance"
        assert spec_for(SUBTITLE_REMOVAL).default_model == "gemini-omni"
        assert spec_for(COVER_CHANGE).default_model == "seedance-fast"


# ---------------------------------------------------------------------------
# Seedance prompt templates (docs/localisation.md §4.3)
# ---------------------------------------------------------------------------

_SLOTS = ("{source_language}", "{target_language}", "{dialogue}")

# Phrases that must appear in BOTH localisation templates (adaptive speech).
_ADAPTIVE_LIPS_NEEDLES = (
    "lip-sync the translated speech",
    "mouth shapes",
    "facial expressions",
    "emotional reactions",
)
_CONTINUITY_NEEDLES = (
    "on-screen text",
    "framing",
    "camera movement",
    "background",
    "lighting",
    "phone or tablet screens",
)
_FORBIDDEN_LIP_FREEZE = (
    "original motion and lip movements",
    "original lip movements",
    "keep the original lip",
    "preserve the original lip",
)


class TestLocalisationPrompts:
    @pytest.mark.parametrize(
        "template",
        [
            project_types._LOCALISATION_SWAP_PROMPT,
            project_types._LOCALISATION_KEEP_PROMPT,
        ],
    )
    def test_carries_all_three_slots(self, template):
        for slot in _SLOTS:
            assert slot in template

    @pytest.mark.parametrize(
        "template",
        [
            project_types._LOCALISATION_SWAP_PROMPT,
            project_types._LOCALISATION_KEEP_PROMPT,
        ],
    )
    def test_has_no_other_braces(self, template):
        """build_prompt uses str.format — a stray brace raises at run time."""
        rendered = template.format(
            source_language="english", target_language="japanese", dialogue="D"
        )
        assert "{" not in rendered and "}" not in rendered

    def test_swap_is_the_registry_default_prompt(self):
        assert (
            spec_for(LOCALISATION).default_prompt
            == project_types._LOCALISATION_SWAP_PROMPT
        )

    def test_keep_template_preserves_the_person(self):
        rendered = project_types._LOCALISATION_KEEP_PROMPT.format(
            source_language="english", target_language="japanese", dialogue="D"
        )
        lowered = rendered.lower()
        assert "do not replace the character" in lowered
        assert "face" in lowered and "identity" in lowered and "appearance" in lowered
        assert "hair" in lowered and "clothing" in lowered
        # It must NOT ask for a character swap or mention the reference image —
        # this is the mode where no reference is supplied.
        assert "replace the main person" not in lowered
        assert "reference image" not in lowered

    def test_swap_template_requests_reference_character(self):
        rendered = project_types._LOCALISATION_SWAP_PROMPT.format(
            source_language="english", target_language="japanese", dialogue="D"
        )
        lowered = rendered.lower()
        assert "replace the main person" in lowered
        assert "reference image" in lowered
        assert "identity consistent with the reference image" in lowered
        assert "do not replace the character" not in lowered

    @pytest.mark.parametrize(
        "template",
        [
            project_types._LOCALISATION_SWAP_PROMPT,
            project_types._LOCALISATION_KEEP_PROMPT,
        ],
    )
    def test_both_templates_require_adaptive_lips_and_reactions(self, template):
        lowered = template.lower()
        for needle in _ADAPTIVE_LIPS_NEEDLES:
            assert needle in lowered
        assert "do not freeze the original face or mouth motion" in lowered
        for bad in _FORBIDDEN_LIP_FREEZE:
            assert bad not in lowered

    @pytest.mark.parametrize(
        "template",
        [
            project_types._LOCALISATION_SWAP_PROMPT,
            project_types._LOCALISATION_KEEP_PROMPT,
        ],
    )
    def test_both_templates_protect_non_speech_continuity(self, template):
        lowered = template.lower()
        for needle in _CONTINUITY_NEEDLES:
            assert needle in lowered
        assert "assigned to that speaker" in lowered
        assert "do not swap lines" in lowered

    def test_grammar_has_space_before_language_parenthesis(self):
        """Inherited face-swap quirk was speak(from — localisation must not keep it."""
        for template in (
            project_types._LOCALISATION_SWAP_PROMPT,
            project_types._LOCALISATION_KEEP_PROMPT,
        ):
            assert "speak(from" not in template
            rendered = template.format(
                source_language="english",
                target_language="japanese",
                dialogue="D",
            )
            assert "(from english to japanese)" in rendered

    def test_build_prompt_can_render_both(self):
        """The real consumer (localisation.build_prompt) is satisfied by them —
        it looks the templates up by name off this module."""
        from app.localisation import build_prompt

        lines = [
            {"id": 1, "start": 0.0, "end": 2.0, "speaker": "Woman", "text": "はい"},
        ]
        for swap in (True, False):
            out = build_prompt(
                lines=lines,
                source_language="en",
                target_language="ja",
                swap_character=swap,
            )
            assert "from english to japanese" in out
            assert "{" not in out and "}" not in out
            assert "lip-sync the translated speech" in out.lower()
            assert "facial expressions" in out.lower()
            assert "original motion and lip movements" not in out.lower()
            assert out.endswith("**Woman:**     はい")
            if swap:
                assert "Replace the main person" in out
            else:
                assert "Do not replace the character" in out
                assert "reference image" not in out.lower()
