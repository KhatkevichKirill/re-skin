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
    SEG_SHOT_CHUNKS,
    SUBTITLE_REMOVAL,
    spec_for,
)

_SEGMENTATIONS = {
    SEG_DETECT_FACES, SEG_SCENE_DETECT, SEG_HOOK_SPLIT, SEG_BLANK, SEG_SHOT_CHUNKS,
}


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


def test_every_strategy_is_used_and_only_blank_is_shared():
    """Each strategy has a user; only the blank slate is shared, by
    cover_change and reframe (whose placeholder segment is never shown)."""
    used = [spec.segmentation for spec in PROJECT_TYPES.values()]
    assert set(used) == set(_SEGMENTATIONS)
    shared = {s for s in used if used.count(s) > 1}
    assert shared <= {SEG_BLANK}
    assert sorted(k for k, sp in PROJECT_TYPES.items() if sp.segmentation == SEG_BLANK) == [
        "cover_change", "reframe"]


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


class TestLocalisationPromptBuildPromptSemantics:
    """End-to-end render checks through build_prompt (not monkeypatched templates)."""

    def test_speaker_labelled_dialogue_formatting_intact(self):
        from app.localisation import build_prompt

        lines = [
            {"id": 1, "start": 0.0, "end": 1.0, "speaker": "Woman", "text": "Hello"},
            {"id": 2, "start": 1.0, "end": 2.0, "speaker": "Woman", "text": "again"},
            {"id": 3, "start": 2.0, "end": 3.0, "speaker": "Man", "text": "Hi"},
        ]
        out = build_prompt(
            lines=lines,
            source_language="en",
            target_language="es",
            swap_character=False,
        )
        # Consecutive same-speaker lines merge; five spaces after the label.
        assert "**Woman:**     Hello again" in out
        assert "**Man:**     Hi" in out
        assert "{dialogue}" not in out
        assert "{source_language}" not in out
        assert "{target_language}" not in out


# ---------------------------------------------------------------------------
# localisation_mode_of — reading the mode back off a run's prompt
# ---------------------------------------------------------------------------


class TestLocalisationModeOf:
    """The mode is not stored (docs/localisation.md §3.3), so Copy run reads it
    back off the prompt. Being wrong in the "keep" direction withdraws a
    feature, so the classifier only says "keep" when it recognises the
    speech-only template."""

    def _rendered(self, swap: bool) -> str:
        from app.localisation import build_prompt

        return build_prompt(
            lines=[{"id": 1, "start": 0.0, "end": 2.0, "speaker": "Woman", "text": "はい"}],
            source_language="en",
            target_language="ja",
            swap_character=swap,
        )

    def test_keep_template_reads_as_keep(self):
        assert (
            project_types.localisation_mode_of(project_types._LOCALISATION_KEEP_PROMPT)
            == project_types.LOCALISATION_MODE_KEEP
        )

    def test_swap_template_reads_as_swap(self):
        assert (
            project_types.localisation_mode_of(project_types._LOCALISATION_SWAP_PROMPT)
            == project_types.LOCALISATION_MODE_SWAP
        )

    def test_translated_prompts_keep_their_mode(self):
        """What a run actually stores is build_prompt's output — the template
        with the languages filled in and a translated dialogue block appended.
        The marker must survive that."""
        assert (
            project_types.localisation_mode_of(self._rendered(swap=False))
            == project_types.LOCALISATION_MODE_KEEP
        )
        assert (
            project_types.localisation_mode_of(self._rendered(swap=True))
            == project_types.LOCALISATION_MODE_SWAP
        )

    def test_edited_keep_prompt_still_reads_as_keep(self):
        """Operators edit the prompt before submitting; the marker sentence is
        the instruction that defines the mode, so it survives ordinary edits."""
        edited = (
            "IMPORTANT: " + project_types._LOCALISATION_KEEP_PROMPT
            + "\nKeep the background music."
        )
        assert (
            project_types.localisation_mode_of(edited)
            == project_types.LOCALISATION_MODE_KEEP
        )

    @pytest.mark.parametrize("prompt", [None, "", "make her speak japanese"])
    def test_unrecognised_prompt_reads_as_swap(self, prompt):
        """Fail open: a prompt this cannot classify keeps Copy run rather than
        losing it on a guess."""
        assert (
            project_types.localisation_mode_of(prompt)
            == project_types.LOCALISATION_MODE_SWAP
        )

    def test_marker_belongs_to_exactly_one_template(self):
        """The whole classifier rests on this — if a future edit moves the
        sentence into the swap template, every localisation run reads as
        speech-only and Copy run disappears."""
        marker = project_types._LOCALISATION_KEEP_MARKER
        assert marker in project_types._LOCALISATION_KEEP_PROMPT.lower()
        assert marker not in project_types._LOCALISATION_SWAP_PROMPT.lower()


class TestRunCopiesAreUseful:
    """The shared premise of Copy run and Automation: both re-run a run with a
    fresh reference photo, so both are pointless on the one prompt that says to
    keep the person on screen."""

    def _prompt(self, swap: bool) -> str:
        template = (
            project_types._LOCALISATION_SWAP_PROMPT if swap
            else project_types._LOCALISATION_KEEP_PROMPT
        )
        return template.format(
            source_language="english", target_language="japanese",
            dialogue="**Woman:**     はい",
        )

    def test_speech_only_localisation_is_the_only_false(self):
        assert project_types.run_copies_are_useful(
            project_types.LOCALISATION, self._prompt(swap=False)
        ) is False
        assert project_types.run_copies_are_useful(
            project_types.LOCALISATION, self._prompt(swap=True)
        ) is True

    @pytest.mark.parametrize(
        "ptype",
        [project_types.FACE_SWAP, project_types.SUBTITLE_REMOVAL,
         project_types.COVER_CHANGE, None, "not-a-type"],
    )
    def test_other_types_are_never_gated(self, ptype):
        """Only localisation has modes. An unknown/NULL type resolves to the
        face-swap fallback like everywhere else, so a legacy row cannot lose
        its buttons."""
        assert project_types.run_copies_are_useful(
            ptype, self._prompt(swap=False)
        ) is True


# ---------------------------------------------------------------------------
# derived_mute_source (AICP-339)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent", [False, True, None])
def test_derived_mute_source_localisation_is_always_on(parent):
    assert project_types.derived_mute_source(project_types.LOCALISATION, parent) is True


@pytest.mark.parametrize(
    "ptype",
    [
        k for k, s in project_types.PROJECT_TYPES.items()
        if not s.default_mute_source and s.inherits_mute_source
    ],
)
@pytest.mark.parametrize("parent,expected", [(False, False), (True, True), (None, False)])
def test_derived_mute_source_default_off_types_inherit(ptype, parent, expected):
    assert project_types.derived_mute_source(ptype, parent) is expected


@pytest.mark.parametrize("parent", [False, True, None])
def test_derived_mute_source_face_swap_is_always_off(parent):
    """A face-swap project made from a muted (localisation) run starts with
    "Remove original audio" OFF — it keeps the original soundtrack."""
    assert project_types.derived_mute_source(project_types.FACE_SWAP, parent) is False


def test_derived_mute_source_inheriting_types_still_exist():
    """Guard: the inherit test above must not silently parametrize to nothing."""
    assert [
        k for k, s in project_types.PROJECT_TYPES.items()
        if not s.default_mute_source and s.inherits_mute_source
    ]
