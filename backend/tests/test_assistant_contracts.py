"""Reject unsafe/ambiguous proposals before runtime or paid work exists."""

import json

import pytest
from pydantic import ValidationError

from app.assistant_contracts import (
    AssetManifest, EditPlan, MAX_OPERATIONS, PlanSourceError, VersionManifest,
    validate_plan_sources,
)

RUN = "11111111-1111-4111-8111-111111111111"
BASE = "22222222-2222-4222-8222-222222222222"
DONOR = "33333333-3333-4333-8333-333333333333"
ASSET = "44444444-4444-4444-8444-444444444444"
SOURCE = {"asset_id": ASSET, "version_id": DONOR}


def parse(*operations, **extra):
    return EditPlan.model_validate({
        "run_id": RUN, "base_version_id": BASE, "operations": operations, **extra,
    })


def base(**overrides):
    return VersionManifest.model_validate({
        "version_id": BASE, "run_id": RUN, "duration_sec": 20,
        "status": "succeeded", "available": True,
        "editable_subtitle_cue_ids": ["cue-1", "cue-2"], **overrides,
    })


def assets(**overrides):
    return {ASSET: AssetManifest.model_validate({
        "asset_id": ASSET, "version_id": DONOR, "media_kind": "video",
        "duration_sec": 30, "has_audio": True, "available": True, **overrides,
    })}


def check(plan, *, version=None, donor_assets=None):
    validate_plan_sources(
        plan, base_version=version or base(),
        assets=assets() if donor_assets is None else donor_assets,
    )


@pytest.mark.parametrize("operation", [
    {"kind": "trim", "start_sec": 2, "end_sec": 15},
    {"kind": "cut", "start_sec": 2, "end_sec": 5},
    {"kind": "image_fade", "start_sec": 18, "end_sec": 20},
    {"kind": "replace_footage", "start_sec": 2, "end_sec": 5, "source": SOURCE},
    {"kind": "audio_splice", "start_sec": 2, "end_sec": 5, "source": SOURCE, "crossfade_sec": 0.1},
    {"kind": "full_track_audio", "source": SOURCE, "donor_start_sec": 10},
    {"kind": "subtitle_edit", "changes": [
        {"action": "update", "cue_id": "cue-1", "start_sec": 1, "end_sec": 3, "text": "Correct wording"},
        {"action": "delete", "cue_id": "cue-2"},
    ]},
])
def test_supported_operations_roundtrip_and_fit_pinned_sources(operation):
    plan = parse(operation)
    check(plan)
    assert EditPlan.model_validate_json(plan.model_dump_json()) == plan


@pytest.mark.parametrize("start,end", [
    (2, 2), (3, 2), (-1, 2), (0, 86_401),
    (float("nan"), 2), (0, float("inf")), (float("-inf"), 2),
    (True, 3), ("0", 3),
])
def test_invalid_ranges_and_nonfinite_times_are_rejected(start, end):
    with pytest.raises(ValidationError):
        parse({"kind": "cut", "start_sec": start, "end_sec": end})


@pytest.mark.parametrize("field,value", [
    ("donor_start_sec", float("nan")), ("donor_start_sec", -1),
    ("crossfade_sec", float("inf")), ("crossfade_sec", -0.1),
    ("crossfade_sec", 1.1),
])
def test_audio_parameters_are_finite_and_window_bounded(field, value):
    with pytest.raises(ValidationError):
        parse({"kind": "audio_splice", "start_sec": 0, "end_sec": 2,
               "source": SOURCE, field: value})


@pytest.mark.parametrize("extra", [
    {"shell": "rm -rf /"}, {"filter": "movie=/etc/passwd"},
    {"output_path": "/app/data/final.mp4"}, {"url": "https://example.org/a.mp4"},
])
def test_executable_or_storage_fields_are_forbidden_at_all_levels(extra):
    op = {"kind": "audio_splice", "start_sec": 0, "end_sec": 2, "source": SOURCE}
    with pytest.raises(ValidationError):
        parse(op, **extra)
    with pytest.raises(ValidationError):
        parse({**op, **extra})
    with pytest.raises(ValidationError):
        parse({**op, "source": {**SOURCE, **extra}})


@pytest.mark.parametrize("bad_id", [
    "../../secrets/gdrive-sa.json", "/etc/passwd", "https://example.org/asset",
    "$(touch /tmp/injected)", "44444444-4444-4444-8444-444444444444;id",
])
def test_ids_cannot_be_paths_urls_or_shell(bad_id):
    with pytest.raises(ValidationError):
        parse({"kind": "full_track_audio", "source": {**SOURCE, "asset_id": bad_id}})
    with pytest.raises(ValidationError):
        parse({"kind": "trim", "start_sec": 0, "end_sec": 2}, base_version_id=bad_id)


def test_unsupported_kind_schema_and_operation_limits():
    with pytest.raises(ValidationError):
        parse({"kind": "generate_speech", "prompt": "Say this"})
    with pytest.raises(ValidationError):
        parse({"kind": "cut", "start_sec": 1, "end_sec": 2}, schema_version=2)
    with pytest.raises(ValidationError):
        parse()
    with pytest.raises(ValidationError):
        parse(*[{"kind": "image_fade", "start_sec": i, "end_sec": i + 1}
                for i in range(MAX_OPERATIONS + 1)])


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_schema_version_does_not_coerce_boolean_float_or_text(value):
    with pytest.raises(ValidationError):
        parse({"kind": "trim", "start_sec": 0, "end_sec": 2}, schema_version=value)


def test_plan_and_nested_operations_are_immutable():
    plan = parse({"kind": "trim", "start_sec": 0, "end_sec": 2})
    assert isinstance(plan.operations, tuple)
    with pytest.raises(ValidationError):
        plan.base_version_id = DONOR
    with pytest.raises(ValidationError):
        plan.operations[0].end_sec = 3


@pytest.mark.parametrize("kind", ["trim", "cut"])
def test_timeline_changes_cannot_mix_with_other_operations(kind):
    with pytest.raises(ValidationError, match="sole operation"):
        parse({"kind": kind, "start_sec": 1, "end_sec": 2},
              {"kind": "image_fade", "start_sec": 3, "end_sec": 4})


def test_compatible_audio_and_picture_mix_uses_the_same_base_coordinates():
    check(parse(
        {"kind": "replace_footage", "start_sec": 2, "end_sec": 4, "source": SOURCE},
        {"kind": "audio_splice", "start_sec": 2, "end_sec": 4, "source": SOURCE},
        {"kind": "image_fade", "start_sec": 18, "end_sec": 20},
    ))


@pytest.mark.parametrize("other", [
    {"kind": "audio_splice", "start_sec": 1, "end_sec": 3, "source": SOURCE},
    {"kind": "full_track_audio", "source": SOURCE},
])
def test_overlapping_audio_operations_are_rejected(other):
    with pytest.raises(ValidationError, match="overlapping audio"):
        parse({"kind": "audio_splice", "start_sec": 0, "end_sec": 2, "source": SOURCE}, other)


def test_adjacent_intervals_are_allowed_and_overlapping_picture_is_rejected():
    check(parse({"kind": "image_fade", "start_sec": 0, "end_sec": 2},
                {"kind": "image_fade", "start_sec": 2, "end_sec": 4}))
    with pytest.raises(ValidationError, match="overlapping picture"):
        parse({"kind": "image_fade", "start_sec": 0, "end_sec": 2},
              {"kind": "replace_footage", "start_sec": 1, "end_sec": 3, "source": SOURCE})


@pytest.mark.parametrize("overrides", [
    {"version_id": DONOR}, {"run_id": DONOR}, {"status": "pending"},
    {"status": "failed"}, {"status": "cancelled"}, {"available": False},
])
def test_base_must_match_the_run_and_be_a_successful_available_version(overrides):
    with pytest.raises(PlanSourceError):
        check(parse({"kind": "trim", "start_sec": 0, "end_sec": 2}), version=base(**overrides))


@pytest.mark.parametrize("operation", [
    {"kind": "trim", "start_sec": 0, "end_sec": 21},
    {"kind": "cut", "start_sec": 0, "end_sec": 20},
    {"kind": "image_fade", "start_sec": 19, "end_sec": 21},
])
def test_base_duration_checks_and_nonempty_cut_result(operation):
    with pytest.raises(PlanSourceError):
        check(parse(operation))


@pytest.mark.parametrize("donor_assets", [
    {}, assets(available=False), assets(version_id=BASE), assets(duration_sec=1),
    assets(has_audio=False),
])
def test_audio_source_must_be_bound_available_audible_and_long_enough(donor_assets):
    with pytest.raises(PlanSourceError):
        check(parse({"kind": "audio_splice", "start_sec": 2, "end_sec": 5,
                     "source": SOURCE}), donor_assets=donor_assets)


def test_full_track_requires_duration_coverage_and_footage_requires_video():
    with pytest.raises(PlanSourceError, match="duration"):
        check(parse({"kind": "full_track_audio", "source": SOURCE, "donor_start_sec": 11}))
    with pytest.raises(PlanSourceError, match="video"):
        check(parse({"kind": "replace_footage", "start_sec": 0, "end_sec": 2,
                     "source": SOURCE}), donor_assets=assets(media_kind="audio"))


@pytest.mark.parametrize("text", ["", "   ", "bad\x00text", "bad\x7ftext", "bad\x85text", "{\\pos(0,0)}override"])
def test_subtitle_text_rejects_controls_and_ass_overrides(text):
    with pytest.raises(ValidationError):
        parse({"kind": "subtitle_edit", "changes": [
            {"action": "update", "cue_id": "cue-1", "start_sec": 0, "end_sec": 1, "text": text},
        ]})


def test_subtitle_text_is_inert_data_and_cues_are_bound_to_editable_track():
    # A renderer must escape text rather than executing or interpolating it.
    plan = parse({"kind": "subtitle_edit", "changes": [
        {"action": "update", "cue_id": "cue-1", "start_sec": 0, "end_sec": 1,
         "text": "Literal $(example) and <words>\nNext line"},
    ]})
    check(plan)
    with pytest.raises(PlanSourceError, match="editable"):
        check(plan, version=base(editable_subtitle_cue_ids=[]))
    with pytest.raises(PlanSourceError, match="duration"):
        check(parse({"kind": "subtitle_edit", "changes": [
            {"action": "update", "cue_id": "cue-1", "start_sec": 19, "end_sec": 21, "text": "Late"},
        ]}))


def test_conflicting_subtitle_changes_cannot_silently_override_each_other():
    change = {"action": "delete", "cue_id": "cue-1"}
    with pytest.raises(ValidationError, match="only once"):
        parse({"kind": "subtitle_edit", "changes": [change, change]})
    with pytest.raises(ValidationError, match="combine subtitle"):
        parse({"kind": "subtitle_edit", "changes": [change]},
              {"kind": "subtitle_edit", "changes": [{"action": "delete", "cue_id": "cue-2"}]})


def test_discriminated_json_schema_has_no_unbounded_executable_fields():
    schema = EditPlan.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["operations"]["items"]["discriminator"]["propertyName"] == "kind"
    for definition in schema["$defs"].values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
    assert "NaN" not in json.dumps(schema, allow_nan=False)
