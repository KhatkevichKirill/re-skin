"""
Tests for app/project_types.py — the project-type registry.

The registry is a table of constants, so the tests here are about the
*invariants* the rest of the app relies on rather than about behaviour:

* every spec carries every field, and ``segmentation`` is one of the SEG_*
  values pipeline_v2.analyze_project dispatches on (a typo there would silently
  demote a type to the blank-slate branch);
* the localisation spec matches docs/localisation.md §2/§6 — those values are
  properties of the type, not operator preferences;
* the two Seedance templates render exactly what run 1736426f was signed off
  with (docs/localisation.md §4.3).
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
        assert spec.default_model == "seedance-2-5"
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
        assert model.max_clip_sec >= 10.0

    def test_gemini_omni_would_be_filtered_out(self):
        """The filter requires_audio_model drives has real teeth."""
        assert ai_models.spec_for("gemini-omni").produces_audio is False


# ---------------------------------------------------------------------------
# Seedance prompt templates (docs/localisation.md §4.3)
# ---------------------------------------------------------------------------

# The run-1736426f paragraph, verbatim from the DB, with the dialogue block
# stripped. Byte-for-byte what the operator signed off on for EN→JA.
_PROVEN_PARAGRAPH = (
    "Replace the woman in the reference video with the woman shown in the "
    "reference image. Keep their face and identity consistent with the reference "
    "image throughout. Change only the character and language which they speak"
    "(from english to japanese) — keep everything else exactly the same: the phone "
    "or tablet screen and its contents, all on-screen text and captions, the "
    "background, lighting, framing, and the original motion and lip movements."
)

_SLOTS = ("{source_language}", "{target_language}", "{dialogue}")


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

    def test_swap_template_reproduces_the_proven_prompt(self):
        """Generalised ONLY where the slots go — plus the gendered noun, which a
        template cannot know, restored to the face-swap default's wording."""
        rendered = project_types._LOCALISATION_SWAP_PROMPT.format(
            source_language="english",
            target_language="japanese",
            dialogue="**Woman:**     あ、読んでないよ。",
        )
        paragraph, _, dialogue = rendered.partition("\n")
        assert dialogue == "**Woman:**     あ、読んでないよ。"
        assert paragraph == _PROVEN_PARAGRAPH.replace(
            "Replace the woman in the reference video with the woman shown",
            "Replace the main person in the reference video with the person shown",
        )
        # The quirk that was signed off on: no space before the parenthesis.
        assert "speak(from english to japanese)" in paragraph

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
        assert "do not replace" in lowered
        assert "face" in lowered and "identity" in lowered and "appearance" in lowered
        # It must NOT ask for a character swap or mention the reference image —
        # this is the mode where no reference is supplied.
        assert "replace the main person" not in lowered
        assert "reference image" not in lowered
        # Only the language changes.
        assert "change only the language which they speak" in lowered

    def test_both_templates_protect_the_same_things(self):
        """The KEEP variant keeps the proven prompt's second half verbatim."""
        tail = (
            " — keep everything else exactly the same: the phone or tablet screen "
            "and its contents, all on-screen text and captions, the background, "
            "lighting, framing, and the original motion and lip movements.\n"
            "{dialogue}"
        )
        assert project_types._LOCALISATION_SWAP_PROMPT.endswith(tail)
        assert project_types._LOCALISATION_KEEP_PROMPT.endswith(tail)

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
            assert out.endswith("**Woman:**     はい")
