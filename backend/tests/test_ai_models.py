"""
Tests for app/ai_models.py — the per-model capability registry.

These are the rules every call site now depends on instead of its own copy of
the model facts, so they are tested directly rather than only through the
pipeline.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import ai_models  # noqa: E402


# ---------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------


class TestRegistryInvariants:
    def test_every_key_matches_its_spec_key(self):
        """A mismatch would make spec_for(x).key lie about which model it is."""
        for key, spec in ai_models.AI_MODELS.items():
            assert spec.key == key

    def test_default_model_exists(self):
        assert ai_models.DEFAULT_MODEL in ai_models.AI_MODELS

    def test_valid_models_mirrors_the_table(self):
        assert ai_models.VALID_MODELS == frozenset(ai_models.AI_MODELS)

    def test_seedance_2_5_is_registered(self):
        assert "seedance-2-5" in ai_models.AI_MODELS

    def test_every_family_is_one_the_pipeline_can_build(self):
        """pipeline_v2._create_swap_task branches on exactly these two."""
        assert {s.family for s in ai_models.AI_MODELS.values()} <= {
            "seedance", "omni"
        }

    def test_every_resolution_is_rankable(self):
        """An unranked label would sort below everything and silently become the
        conservative fallback in resolution_or_default."""
        for spec in ai_models.AI_MODELS.values():
            for r in spec.resolutions:
                assert r in ai_models._RESOLUTION_ORDER, (spec.key, r)

    def test_clip_bounds_are_coherent(self):
        for spec in ai_models.AI_MODELS.values():
            assert 0 < spec.min_clip_sec <= spec.max_clip_sec, spec.key

    def test_universal_cap_is_generatable_by_every_model(self):
        """The NULL-max_segment_sec default must be runnable on anything."""
        cap = ai_models.UNIVERSAL_MAX_SEGMENT_SEC
        assert ai_models.models_excluded_by_segment_len(cap) == []

    def test_absolute_cap_is_generatable_by_at_least_one_model(self):
        cap = ai_models.ABSOLUTE_MAX_SEGMENT_SEC
        assert ai_models.models_supporting_segment_len(cap)


# ---------------------------------------------------------------------------
# spec_for / labels / ids
# ---------------------------------------------------------------------------


class TestSpecLookup:
    def test_unknown_and_none_fall_back_to_default(self):
        """Protects rows written before a model existed / with a NULL column."""
        assert ai_models.spec_for(None).key == ai_models.DEFAULT_MODEL
        assert ai_models.spec_for("").key == ai_models.DEFAULT_MODEL
        assert ai_models.spec_for("not-a-model").key == ai_models.DEFAULT_MODEL

    def test_label_for_echoes_an_unknown_key(self):
        """Unlike spec_for, label_for must not claim an unknown model is the
        default — the UI should show what is actually stored."""
        assert ai_models.label_for("seedance-2-5") == "Seedance 2.5"
        assert ai_models.label_for("mystery-model") == "mystery-model"
        assert ai_models.label_for(None) == ""

    def test_kie_model_id(self):
        assert ai_models.kie_model_id("seedance-2-5") == "bytedance/seedance-2-5"
        assert ai_models.kie_model_id("seedance") == "bytedance/seedance-2"

    def test_is_omni(self):
        assert ai_models.is_omni("gemini-omni") is True
        assert ai_models.is_omni("seedance-2-5") is False


# ---------------------------------------------------------------------------
# resolution_or_default
# ---------------------------------------------------------------------------


class TestResolutionOrDefault:
    def test_supported_resolution_passes_through(self):
        assert ai_models.resolution_or_default("seedance", "1080p") == "1080p"

    def test_unsupported_falls_to_the_highest_non_exceeding_tier(self):
        """A 1080p run switched to Seedance 2.5 (720p max) must land on 720p.

        This is the regression the ranking exists for: a positional
        `resolutions[0]` would return 2.5's *480p* and silently downgrade a
        production run to test quality.
        """
        assert ai_models.resolution_or_default("seedance-2-5", "1080p") == "720p"
        assert ai_models.resolution_or_default("seedance-fast", "4k") == "720p"

    def test_below_everything_falls_to_the_models_cheapest(self):
        """480p on Gemini Omni (720p floor) must not become a 4k bill."""
        assert ai_models.resolution_or_default("gemini-omni", "480p") == "720p"

    def test_unknown_and_none_fall_to_the_models_cheapest(self):
        assert ai_models.resolution_or_default("seedance", None) == "480p"
        assert ai_models.resolution_or_default("seedance", "banana") == "480p"


# ---------------------------------------------------------------------------
# duration_for / aspect_for — the Seedance 2.5 geometry contract
# ---------------------------------------------------------------------------


class TestDurationFor:
    def test_seedance_2_0_rounds_and_clamps(self):
        assert ai_models.duration_for("seedance", 0.0, 7.4) == 7
        assert ai_models.duration_for("seedance", 0.0, 1.0) == 4    # min
        assert ai_models.duration_for("seedance", 0.0, 99.0) == 15  # max

    def test_omni_snaps_to_its_allowed_set(self):
        assert ai_models.duration_for("gemini-omni", 0.0, 5.2) == 6
        assert ai_models.duration_for("gemini-omni", 0.0, 9.9) == 10

    def test_seedance_2_5_always_asks_the_input(self):
        """2.5 in video-editing mode rejects any concrete duration; -1 means
        "match the input video", which is also more accurate than rounding."""
        assert ai_models.duration_for("seedance-2-5", 0.0, 7.4) == -1
        assert ai_models.duration_for("seedance-2-5", 0.0, 29.0) == -1
        assert ai_models.FOLLOW_INPUT_DURATION == -1


class TestAspectFor:
    def test_seedance_2_0_passes_a_supported_ratio_through(self):
        assert ai_models.aspect_for("seedance", "9:16") == "9:16"

    def test_seedance_2_0_falls_back_to_adaptive(self):
        assert ai_models.aspect_for("seedance", "7:3") == "adaptive"
        assert ai_models.aspect_for("seedance", None) == "adaptive"

    def test_seedance_2_5_is_always_adaptive(self):
        """Even a ratio the source genuinely has is rejected by 2.5 — it derives
        geometry from the input clip and 500s on anything but "adaptive"."""
        assert ai_models.aspect_for("seedance-2-5", "9:16") == "adaptive"
        assert ai_models.aspect_for("seedance-2-5", None) == "adaptive"


class TestValidateDuration:
    def test_in_range_is_accepted(self):
        ai_models.validate_duration("seedance", 10)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="between 4 and 15"):
            ai_models.validate_duration("seedance", 20)

    def test_omni_rejects_a_duration_outside_its_set(self):
        with pytest.raises(ValueError, match="must be one of"):
            ai_models.validate_duration("gemini-omni", 5)

    def test_seedance_2_5_requires_the_sentinel(self):
        ai_models.validate_duration("seedance-2-5", -1)
        with pytest.raises(ValueError, match="must be -1"):
            ai_models.validate_duration("seedance-2-5", 10)

    def test_the_pair_is_never_half_right(self):
        """aspect_for and duration_for must agree per model: 2.5 needs BOTH
        "adaptive" and -1, and the API validates them jointly."""
        for key in ai_models.AI_MODELS:
            follows = ai_models.spec_for(key).follows_input_geometry
            adaptive = ai_models.aspect_for(key, "9:16") == "adaptive"
            sentinel = ai_models.duration_for(key, 0.0, 8.0) == -1
            assert adaptive == follows, key
            assert sentinel == follows, key


# ---------------------------------------------------------------------------
# Segment-length helpers
# ---------------------------------------------------------------------------


class TestSegmentLengthHelpers:
    def test_partition_is_complete(self):
        """Every model is in exactly one of the two lists, for any length."""
        for seconds in (5.0, 10.0, 12.0, 15.0, 30.0, 45.0):
            ok = set(ai_models.models_supporting_segment_len(seconds))
            no = set(ai_models.models_excluded_by_segment_len(seconds))
            assert ok | no == set(ai_models.AI_MODELS), seconds
            assert ok & no == set(), seconds

    def test_a_30s_segment_needs_seedance_2_5(self):
        assert ai_models.models_supporting_segment_len(30.0) == ["seedance-2-5"]

    def test_a_12s_segment_excludes_omni_only(self):
        assert ai_models.models_excluded_by_segment_len(12.0) == ["gemini-omni"]

    def test_the_epsilon_absorbs_float_noise(self):
        """A segment stored as 15.000000001 must not be read as over the 15s
        ceiling — the tolerance is what keeps a hand-edited boundary usable."""
        assert "seedance" in ai_models.models_supporting_segment_len(15.02)


# ---------------------------------------------------------------------------
# Template JSON
# ---------------------------------------------------------------------------


class TestTemplateJson:
    def test_resolution_choices_cover_every_model(self):
        data = json.loads(ai_models.resolution_choices_json())
        assert set(data) == set(ai_models.AI_MODELS)
        for key, pairs in data.items():
            assert [p[0] for p in pairs] == list(
                ai_models.AI_MODELS[key].resolutions
            )
            # Every option carries a human label, never a bare value.
            assert all(len(p) == 2 and p[1] for p in pairs)

    def test_model_limits_cover_every_model(self):
        data = json.loads(ai_models.model_limits_json())
        assert set(data) == set(ai_models.AI_MODELS)
        for key, limits in data.items():
            spec = ai_models.AI_MODELS[key]
            assert limits["max_clip_sec"] == spec.max_clip_sec
            assert limits["produces_audio"] == spec.produces_audio
            assert limits["label"] == spec.label
