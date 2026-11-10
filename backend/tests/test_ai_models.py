"""
Tests for the AI model registry (app/ai_models.py).

The registry is the single source of truth for per-model capabilities, so these
tests pin the facts that other layers now read instead of hardcoding: the kie
model ids, the resolution sets, the per-clip ceilings, and how an output
duration is derived. A regression here silently changes what gets billed.
"""

from __future__ import annotations

import json

import pytest

from app import ai_models


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------


def test_every_spec_key_matches_its_dict_key():
    for key, spec in ai_models.AI_MODELS.items():
        assert spec.key == key


def test_valid_models_matches_table():
    assert ai_models.VALID_MODELS == frozenset(ai_models.AI_MODELS)


def test_default_model_is_registered():
    assert ai_models.DEFAULT_MODEL in ai_models.AI_MODELS


def test_expected_models_present():
    # The four pre-existing models plus Seedance 2.5. Adding a model is fine;
    # silently losing one is not (the run_model_enum can never drop a value, so
    # a stored run would render/submit wrong).
    assert {
        "seedance",
        "seedance-fast",
        "seedance-mini",
        "seedance-2-5",
        "gemini-omni",
    } <= set(ai_models.AI_MODELS)


def test_short_labels_are_unique_and_fit_the_idea_budget():
    """Idea reserves 4 chars for the model tag — a longer one overruns the cap."""
    shorts = [s.short_label for s in ai_models.AI_MODELS.values()]
    assert all(shorts)
    assert len(shorts) == len(set(shorts))
    assert all(len(s) <= 4 for s in shorts)


def test_short_label_for_maps_every_model():
    assert ai_models.short_label_for("seedance") == "SD2"
    assert ai_models.short_label_for("seedance-fast") == "SDf2"
    assert ai_models.short_label_for("seedance-mini") == "SDm2"
    assert ai_models.short_label_for("seedance-2-5") == "SD25"
    assert ai_models.short_label_for("gemini-omni") == "GO"


@pytest.mark.parametrize("bad", [None, "", "nope"])
def test_short_label_for_falls_back_to_default(bad):
    """Unlike label_for, an unknown key must NOT echo through — Idea is length-capped."""
    assert ai_models.short_label_for(bad) == ai_models.AI_MODELS[ai_models.DEFAULT_MODEL].short_label


def test_kie_model_ids_are_unique_and_nonempty():
    ids = [s.kie_model_id for s in ai_models.AI_MODELS.values()]
    assert all(ids)
    assert len(ids) == len(set(ids))


def test_every_family_is_known():
    assert {s.family for s in ai_models.AI_MODELS.values()} <= {"seedance", "omni", "kling"}
    # Only the images-only Kling family lacks a v2v mode.
    assert {s.family for s in ai_models.V2V_MODELS.values()} <= {"seedance", "omni"}


def test_resolutions_are_nonempty_tuples():
    for spec in ai_models.AI_MODELS.values():
        assert isinstance(spec.resolutions, tuple)
        assert spec.resolutions


def test_duration_bounds_are_coherent():
    for spec in ai_models.AI_MODELS.values():
        assert spec.min_duration_sec <= spec.max_duration_sec
        # A clip we would pad up to must still fit under the ceiling.
        assert spec.min_clip_sec <= spec.max_clip_sec
        # A model must never be asked for more output than its clip ceiling.
        assert spec.max_duration_sec <= spec.max_clip_sec + 0.05


# ---------------------------------------------------------------------------
# Seedance 2.5 — the newly added model
# ---------------------------------------------------------------------------


def test_seedance_2_5_wiring():
    spec = ai_models.AI_MODELS["seedance-2-5"]
    assert spec.kie_model_id == "bytedance/seedance-2-5"
    assert spec.family == "seedance"
    # kie.ai added 1080p on 2026-08-18; 2.5 now matches base Seedance's ladder
    # and IS a drop-in for it at production resolution.
    assert spec.resolutions == ("480p", "720p", "1080p")
    # Still no 4k tier upstream — Gemini Omni remains the only 4k backend.
    assert "4k" not in spec.resolutions
    # The headline capability: 30s clips instead of 15s.
    assert spec.max_clip_sec == 30.0
    assert spec.max_duration_sec == 30
    # Only 2.5 exposes the explicit audio switch.
    assert spec.supports_generate_audio is True


def test_only_seedance_2_5_supports_generate_audio():
    supporting = {
        k for k, s in ai_models.AI_MODELS.items() if s.supports_generate_audio
    }
    assert supporting == {"seedance-2-5"}


# ---------------------------------------------------------------------------
# Pre-existing models keep their exact behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,kie_id,resolutions,max_clip",
    [
        ("seedance", "bytedance/seedance-2", ("480p", "720p", "1080p"), 15.0),
        ("seedance-fast", "bytedance/seedance-2-fast", ("480p", "720p"), 15.0),
        ("seedance-mini", "bytedance/seedance-2-mini", ("480p", "720p"), 15.0),
        ("gemini-omni", "gemini-omni-video", ("720p", "1080p", "4k"), 10.0),
    ],
)
def test_existing_models_unchanged(key, kie_id, resolutions, max_clip):
    spec = ai_models.AI_MODELS[key]
    assert spec.kie_model_id == kie_id
    assert spec.resolutions == resolutions
    assert spec.max_clip_sec == max_clip


def test_gemini_omni_is_the_only_omni_family():
    assert ai_models.is_omni("gemini-omni") is True
    for key in ("seedance", "seedance-fast", "seedance-mini", "seedance-2-5"):
        assert ai_models.is_omni(key) is False


def test_gemini_omni_produces_no_audio():
    assert ai_models.AI_MODELS["gemini-omni"].produces_audio is False
    for key in ("seedance", "seedance-fast", "seedance-mini", "seedance-2-5"):
        assert ai_models.AI_MODELS[key].produces_audio is True


# ---------------------------------------------------------------------------
# Lookup helpers + unknown-key fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, "", "no-such-model"])
def test_spec_for_falls_back_to_default(bad):
    assert ai_models.spec_for(bad).key == ai_models.DEFAULT_MODEL


def test_kie_model_id_helper():
    assert ai_models.kie_model_id("seedance-2-5") == "bytedance/seedance-2-5"
    # Unknown key falls back to the default model's id rather than raising —
    # a stored run must always be submittable.
    assert ai_models.kie_model_id("bogus") == "bytedance/seedance-2"


def test_label_for_passes_through_unknown_values():
    # The Jinja `model_label` filter relies on this: never blow up on a value
    # that is no longer in the table, just show it raw.
    assert ai_models.label_for("seedance-2-5") == "Seedance 2.5"
    assert ai_models.label_for("mystery") == "mystery"
    assert ai_models.label_for(None) == ""


def test_resolution_or_default():
    # Supported → passed through.
    assert ai_models.resolution_or_default("seedance-2-5", "720p") == "720p"
    # 1080p is supported on 2.5 since 2026-08-18, so it is no longer coerced —
    # a 1080p run switched to 2.5 now stays at 1080p.
    assert ai_models.resolution_or_default("seedance-2-5", "1080p") == "1080p"
    # Unsupported → the CLOSEST supported tier that does not exceed the request.
    # 2.5 has no 4k tier, so a 4k request lands on its best, 1080p. It used to
    # return spec.resolutions[0] == "480p", i.e. the guard against an impossible
    # resolution silently delivered test-quality video.
    assert ai_models.resolution_or_default("seedance-2-5", "4k") == "1080p"
    # The 2.0 Fast/Mini variants still stop at 720p and still coerce.
    assert ai_models.resolution_or_default("seedance-fast", "1080p") == "720p"
    assert ai_models.resolution_or_default("seedance-mini", "4k") == "720p"
    # Requesting below everything the model offers → its lowest tier, the
    # minimum upgrade. Never its highest: that would turn a stray 480p into a 4k
    # bill on Gemini Omni.
    assert ai_models.resolution_or_default("gemini-omni", "480p") == "720p"
    # NULL / unrecognised is unranked, so it takes the same conservative floor.
    assert ai_models.resolution_or_default("seedance", None) == "480p"
    assert ai_models.resolution_or_default("gemini-omni", None) == "720p"
    assert ai_models.resolution_or_default("seedance", "8k") == "480p"


def test_resolution_or_default_never_exceeds_a_ranked_request():
    """For every (model, request) pair the answer is supported and not upgraded.

    The fallback exists to keep an impossible resolution off the wire; it must
    never spend more than the run asked for, and never return something the
    model would reject.
    """
    for key, spec in ai_models.AI_MODELS.items():
        for requested in ai_models._RESOLUTION_ORDER:
            got = ai_models.resolution_or_default(key, requested)
            assert got in spec.resolutions
            want_rank = ai_models._RESOLUTION_RANK[requested]
            got_rank = ai_models._RESOLUTION_RANK[got]
            if any(
                ai_models._RESOLUTION_RANK[r] <= want_rank for r in spec.resolutions
            ):
                assert got_rank <= want_rank
            else:
                # Nothing cheap enough exists — the smallest upgrade is taken.
                assert got_rank == min(
                    ai_models._RESOLUTION_RANK[r] for r in spec.resolutions
                )


def test_every_registry_resolution_is_ranked():
    """A model added with an unranked resolution label would break the ordering.

    _RESOLUTION_ORDER is what makes resolution_or_default independent of the
    display order of AIModelSpec.resolutions, so it has to cover the table.
    """
    for spec in ai_models.AI_MODELS.values():
        for res in spec.resolutions:
            assert res in ai_models._RESOLUTION_RANK, (spec.key, res)


def test_max_clip_sec_helper():
    assert ai_models.max_clip_sec("seedance-2-5") == 30.0
    assert ai_models.max_clip_sec("seedance") == 15.0
    assert ai_models.max_clip_sec("gemini-omni") == 10.0


# ---------------------------------------------------------------------------
# duration_for — what actually gets billed
# ---------------------------------------------------------------------------


def test_duration_for_rounds_and_clamps_on_seedance():
    # Rounds to whole seconds.
    assert ai_models.duration_for("seedance", 0.0, 7.4) == 7
    assert ai_models.duration_for("seedance", 0.0, 7.6) == 8
    # Clamps to the model floor (a 2s padded clip still asks for 4s).
    assert ai_models.duration_for("seedance", 0.0, 2.0) == 4
    # Clamps to the 2.0 ceiling.
    assert ai_models.duration_for("seedance", 0.0, 40.0) == 15


def test_duration_for_2_5_always_follows_the_input():
    """2.5 takes the output length from the input video, so we send -1.

    Learned from the live API, which 500s on any concrete value:
      `ratio` must be `adaptive`. `duration` must be -1.
    Clip length is irrelevant to the value we send — that is the point, and it
    also removes the sub-second drift a rounded duration used to introduce.
    """
    for end in (2.0, 4.4, 10.0, 22.0, 30.0, 45.0):
        assert ai_models.duration_for("seedance-2-5", 0.0, end) == -1
    assert ai_models.FOLLOW_INPUT_DURATION == -1

    # Only 2.5 behaves this way; the 2.0 family still sends a real length.
    assert ai_models.duration_for("seedance", 0.0, 10.0) == 10


def test_duration_for_snaps_to_the_omni_fixed_set():
    # Gemini accepts only 4/6/8/10 — snap to nearest.
    assert ai_models.duration_for("gemini-omni", 0.0, 5.0) in (4, 6)
    assert ai_models.duration_for("gemini-omni", 0.0, 6.4) == 6
    assert ai_models.duration_for("gemini-omni", 0.0, 9.9) == 10
    assert ai_models.duration_for("gemini-omni", 0.0, 30.0) == 10
    for end in (1.0, 3.3, 5.5, 7.7, 12.0):
        assert ai_models.duration_for("gemini-omni", 0.0, end) in (4, 6, 8, 10)


def test_duration_for_respects_offset_start():
    assert ai_models.duration_for("seedance", 10.0, 18.0) == 8


# ---------------------------------------------------------------------------
# validate_duration
# ---------------------------------------------------------------------------


def test_validate_duration_accepts_in_range():
    ai_models.validate_duration("seedance", 15)
    ai_models.validate_duration("seedance-2-5", -1)
    ai_models.validate_duration("gemini-omni", 8)


@pytest.mark.parametrize("bad", [3, 16])
def test_validate_duration_rejects_out_of_range_for_seedance_2_0(bad):
    with pytest.raises(ValueError):
        ai_models.validate_duration("seedance", bad)


def test_validate_duration_rejects_over_30_for_2_5():
    with pytest.raises(ValueError):
        ai_models.validate_duration("seedance-2-5", 31)


@pytest.mark.parametrize("bad", [4, 22, 30, 31, 0])
def test_validate_duration_rejects_concrete_lengths_for_2_5(bad):
    # -1 is the ONLY legal value. A concrete length — even one inside 2.5's
    # nominal 4-30s range — is what the live API rejected with
    # "`duration` must be -1", so it must not get past us either.
    with pytest.raises(ValueError):
        ai_models.validate_duration("seedance-2-5", bad)


def test_validate_duration_rejects_follow_input_for_other_models():
    # The converse: -1 is meaningless to a model that wants a real length.
    for key in ("seedance", "seedance-fast", "seedance-mini", "gemini-omni"):
        with pytest.raises(ValueError):
            ai_models.validate_duration(key, -1)


def test_validate_duration_rejects_off_grid_for_omni():
    with pytest.raises(ValueError):
        ai_models.validate_duration("gemini-omni", 5)


def test_duration_for_output_always_passes_validate_duration():
    # These two must never disagree — duration_for feeds createTask, and
    # validate_duration guards it.
    # Holds for the -1 sentinel too, which is the pairing that broke in prod:
    # duration_for said 22 while the API demanded -1.
    for key in ai_models.AI_MODELS:
        for end in (0.5, 2.0, 4.4, 9.9, 15.0, 22.0, 30.0, 60.0):
            ai_models.validate_duration(key, ai_models.duration_for(key, 0.0, end))


# ---------------------------------------------------------------------------
# Segmentation-cap helpers (the per-project max_segment_sec setting)
# ---------------------------------------------------------------------------


def test_universal_cap_is_the_smallest_ceiling():
    assert ai_models.UNIVERSAL_MAX_SEGMENT_SEC == min(
        s.max_clip_sec for s in ai_models.AI_MODELS.values()
    )
    # Historically hardcoded as OMNI_MAX_CLIP_SECONDS = 10.0. A project left at
    # the default must keep segmenting exactly as it did before this change.
    assert ai_models.UNIVERSAL_MAX_SEGMENT_SEC == 10.0


def test_absolute_cap_is_the_largest_ceiling():
    assert ai_models.ABSOLUTE_MAX_SEGMENT_SEC == 30.0


def test_universal_cap_is_runnable_on_every_model():
    excluded = ai_models.models_excluded_by_segment_len(
        ai_models.UNIVERSAL_MAX_SEGMENT_SEC
    )
    assert excluded == []


def test_models_supporting_segment_len():
    # Segment caps are a v2v notion: the images-only Kling 4.0 never appears.
    assert set(ai_models.models_supporting_segment_len(10.0)) == set(
        ai_models.V2V_MODELS
    )
    assert "kling-4-0" not in ai_models.models_supporting_segment_len(10.0)
    # At 15s Gemini drops out; at 22s only 2.5 survives.
    assert "gemini-omni" not in ai_models.models_supporting_segment_len(15.0)
    assert ai_models.models_supporting_segment_len(22.0) == ["seedance-2-5"]
    assert ai_models.models_supporting_segment_len(30.0) == ["seedance-2-5"]


def test_supported_and_excluded_partition_the_table():
    for length in (5.0, 10.0, 15.0, 22.0, 30.0, 45.0):
        supported = set(ai_models.models_supporting_segment_len(length))
        excluded = set(ai_models.models_excluded_by_segment_len(length))
        assert supported | excluded == set(ai_models.V2V_MODELS)
        assert supported & excluded == set()


def test_nothing_supports_a_segment_beyond_the_absolute_cap():
    assert ai_models.models_supporting_segment_len(31.0) == []


# ---------------------------------------------------------------------------
# JSON payloads consumed by the New Run form's JavaScript
# ---------------------------------------------------------------------------


def test_resolution_choices_json_covers_every_model():
    data = json.loads(ai_models.resolution_choices_json())
    assert set(data) == set(ai_models.AI_MODELS)
    for key, pairs in data.items():
        values = [v for v, _label in pairs]
        assert values == list(ai_models.AI_MODELS[key].resolutions)
        # Every option needs a human label for the dropdown.
        assert all(label for _v, label in pairs)


def test_model_limits_json_shape():
    data = json.loads(ai_models.model_limits_json())
    assert set(data) == set(ai_models.AI_MODELS)
    for key, entry in data.items():
        spec = ai_models.AI_MODELS[key]
        assert entry["max_clip_sec"] == spec.max_clip_sec
        assert entry["produces_audio"] == spec.produces_audio
        assert entry["label"] == spec.label


def test_json_helpers_are_serialisable_and_stable():
    # Called on every form render; must not raise and must round-trip.
    for payload in (
        ai_models.resolution_choices_json(),
        ai_models.model_limits_json(),
    ):
        assert json.loads(payload) == json.loads(payload)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------


def test_min_swap_video_sec_floor():
    # Seedance rejects reference videos shorter than ~1.8s.
    assert ai_models.MIN_SWAP_VIDEO_SEC == 2.0


def test_seedance_aspects_set():
    assert "adaptive" in ai_models.SEEDANCE_ASPECTS
    assert "9:16" in ai_models.SEEDANCE_ASPECTS


# ---------------------------------------------------------------------------
# follows_input_geometry — the "adaptive" + duration -1 pairing
# ---------------------------------------------------------------------------
# Seedance 2.5 classifies our submissions as video editing (we always attach a
# reference video), and then derives output ratio AND duration from that video.
# It rejects the request outright otherwise. Learned the hard way in prod:
#   500 ... Issues: [0] `ratio` must be `adaptive`. [1] `duration` must be -1.


def test_only_seedance_2_5_follows_input_geometry():
    following = {
        k for k, s in ai_models.AI_MODELS.items() if s.follows_input_geometry
    }
    assert following == {"seedance-2-5"}


def test_aspect_for_forces_adaptive_on_2_5():
    # Even a ratio the source genuinely has, and one Seedance otherwise accepts,
    # must become "adaptive" — that exact case (source 9:16) was the prod 500.
    for raw in ("9:16", "16:9", "1:1", "21:9", "adaptive", None, "3:2"):
        assert ai_models.aspect_for("seedance-2-5", raw) == "adaptive"


def test_aspect_for_passes_through_on_other_models():
    assert ai_models.aspect_for("seedance", "9:16") == "9:16"
    assert ai_models.aspect_for("seedance-fast", "16:9") == "16:9"
    # Unsupported / missing still degrades to adaptive, as before.
    assert ai_models.aspect_for("seedance", "3:2") == "adaptive"
    assert ai_models.aspect_for("seedance", None) == "adaptive"


def test_aspect_and_duration_are_consistent_per_model():
    """The two must never be sent half-right — the API validates them together."""
    for key, spec in ai_models.AI_MODELS.items():
        aspect = ai_models.aspect_for(key, "9:16")
        duration = ai_models.duration_for(key, 0.0, 12.0)
        if spec.follows_input_geometry:
            assert aspect == "adaptive" and duration == -1, key
        else:
            assert duration != -1, key


def test_min_clip_sec_per_model():
    # 2.5 in video-editing mode requires a 4-30s input; the 2.0 family accepts
    # ~2s. Padding to 2s and submitting to 2.5 would be rejected.
    assert ai_models.min_clip_sec("seedance-2-5") == 4.0
    for key in ("seedance", "seedance-fast", "seedance-mini", "gemini-omni"):
        assert ai_models.min_clip_sec(key) == 2.0
    # Unknown key falls back to the default model's floor rather than raising.
    assert ai_models.min_clip_sec("bogus") == 2.0


def test_min_clip_sec_never_below_the_seedance_floor():
    # Seedance rejects reference videos under ~1.8s regardless of model.
    for spec in ai_models.AI_MODELS.values():
        assert spec.min_clip_sec >= ai_models.MIN_SWAP_VIDEO_SEC


# ---------------------------------------------------------------------------
# Voice references (reference_audio_urls)
# ---------------------------------------------------------------------------


def test_only_seedance_2_5_accepts_a_voice_reference():
    supporting = {
        k for k, s in ai_models.AI_MODELS.items() if s.max_reference_audios > 0
    }
    assert supporting == {"seedance-2-5"}
    assert ai_models.models_supporting_reference_audio() == ["seedance-2-5"]


@pytest.mark.parametrize(
    "model,expected",
    [
        ("seedance-2-5", True),
        ("seedance", False),
        ("seedance-fast", False),
        ("seedance-mini", False),
        ("gemini-omni", False),
    ],
)
def test_supports_reference_audio(model, expected):
    assert ai_models.supports_reference_audio(model) is expected


def test_supports_reference_audio_falls_back_for_unknown_models():
    """Unknown/NULL resolves to the default spec (2.0), which takes no voice.

    Fail-closed matters here: the alternative would send the key to a backend
    that rejects the whole request.
    """
    assert ai_models.supports_reference_audio(None) is False
    assert ai_models.supports_reference_audio("no-such-model") is False


def test_a_model_accepting_voice_must_also_produce_audio():
    """A voice reference steers a track the model writes.

    Offering one on a silent backend would be a control that does nothing, so
    the registry rejects the combination at import — asserted here so the
    invariant is visible, not just enforced.
    """
    for key, spec in ai_models.AI_MODELS.items():
        if spec.max_reference_audios:
            assert spec.produces_audio, f"{key} takes a voice but writes no audio"


def test_voice_reference_bounds_match_the_api_contract():
    """kie.ai's documented limits for reference_audio_urls (2-30s per file)."""
    assert ai_models.VOICE_REF_MIN_SEC == 2.0
    assert ai_models.VOICE_REF_MAX_SEC == 30.0
    assert "mp3" in ai_models.VOICE_REF_FORMATS


def test_model_limits_json_exposes_the_voice_capability():
    """The New Run form hides the picker off this key."""
    limits = json.loads(ai_models.model_limits_json())
    assert limits["seedance-2-5"]["reference_audio"] == 1
    assert limits["seedance"]["reference_audio"] == 0
    assert limits["gemini-omni"]["reference_audio"] == 0
