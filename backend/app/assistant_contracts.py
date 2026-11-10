"""Pure, versioned assistant proposals; validation grants no execution authority.

All intervals are half-open seconds on the pinned base timeline. Trusted code
must resolve IDs, enforce access/policy, and construct media commands itself.
No model in this module accepts executable code, filesystem paths, or URLs.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Literal, Union
import unicodedata

from pydantic import (
    BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator,
)

MAX_TIMELINE_SEC = 86_400.0
MAX_OPERATIONS = 32
MAX_SUBTITLE_CHANGES = 500

EntityId = Annotated[str, StringConstraints(
    strict=True,
    pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
)]
CueId = Annotated[str, StringConstraints(
    strict=True, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
)]
Seconds = Annotated[float, Field(
    strict=True, ge=0, le=MAX_TIMELINE_SEC, allow_inf_nan=False,
)]
Duration = Annotated[float, Field(
    strict=True, gt=0, le=MAX_TIMELINE_SEC, allow_inf_nan=False,
)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetReference(ContractModel):
    asset_id: EntityId
    version_id: EntityId


class TimeRange(ContractModel):
    start_sec: Seconds
    end_sec: Seconds

    @model_validator(mode="after")
    def ordered_range(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self


class TrimOperation(TimeRange):
    """Keep this interval, shortening picture and audio together."""

    kind: Literal["trim"] = "trim"


class ImageFadeOperation(TimeRange):
    """Fade picture to/from black; keep audio and timeline duration."""

    kind: Literal["image_fade"] = "image_fade"
    direction: Literal["in", "out"] = "out"


class CutOperation(TimeRange):
    """Remove this interval from picture and audio together."""

    kind: Literal["cut"] = "cut"


class ReplaceFootageOperation(TimeRange):
    """Replace picture for exactly this duration, preserving base audio."""

    kind: Literal["replace_footage"] = "replace_footage"
    source: AssetReference
    donor_start_sec: Seconds = 0.0


class AudioSpliceOperation(TimeRange):
    """Replace mixed audio, optionally fitting a donor window; no synthesis.

    donor_duration_sec=None keeps equal duration. Explicit fitting supports a
    <=60s target and donor/target duration ratio0.8..1.25; source endpoints must
    permit only near-silent residual trim/pad, with listening review required.
    """

    kind: Literal["audio_splice"] = "audio_splice"
    source: AssetReference
    donor_start_sec: Seconds = 0.0
    donor_duration_sec: Duration | None = Field(default=None,exclude_if=lambda value:value is None)
    gain_db: Annotated[float, Field(strict=True, ge=-24, le=12, allow_inf_nan=False)] = 0.0
    crossfade_sec: Annotated[float, Field(
        strict=True, ge=0, le=2.0, allow_inf_nan=False,
    )] = 0.0

    @model_validator(mode="after")
    def bounded_crossfade(self):
        if self.crossfade_sec * 2 > self.end_sec - self.start_sec:
            raise ValueError("crossfade_sec must not exceed half the splice duration")
        if self.donor_duration_sec is not None:
            ratio=self.donor_duration_sec/(self.end_sec-self.start_sec)
            if not 0.8-1e-9<=ratio<=1.25+1e-9 or self.end_sec-self.start_sec>60+1e-9:
                raise ValueError("audio duration fit requires a <=60s window and donor/target ratio within [0.8,1.25]")
        return self


class FullTrackAudioOperation(ContractModel):
    """Replace the full base soundtrack with an equal-duration donor window."""

    kind: Literal["full_track_audio"] = "full_track_audio"
    source: AssetReference
    donor_start_sec: Seconds = 0.0
    gain_db: Annotated[float, Field(strict=True, ge=-24, le=12, allow_inf_nan=False)] = 0.0


class BlackEndingVisual(ContractModel):
    kind: Literal['black']


class FreezeEndingVisual(ContractModel):
    kind: Literal['freeze_selected']
    timestamp_sec: Seconds


class ExtendEndingOperation(ContractModel):
    """Atomic suffix replacement and extension with existing 1x donor audio."""

    kind: Literal['extend_ending']
    start_sec: Seconds
    end_sec: Duration
    source: AssetReference
    donor_start_sec: Seconds
    visual: Annotated[Union[BlackEndingVisual, FreezeEndingVisual], Field(discriminator='kind')]
    fade_out_sec: Annotated[float, Field(strict=True, ge=0, le=2, allow_inf_nan=False)] = 0.0

    @model_validator(mode='after')
    def bounded_tail(self):
        if not 0 < self.start_sec < self.end_sec <= 120 or self.end_sec-self.start_sec > 10:
            raise ValueError('ending requires a positive prefix and at most ten seconds of tail within 120 seconds')
        if self.visual.kind == 'black' and self.fade_out_sec != 0:
            raise ValueError('black ending cannot have a fade')
        if self.visual.kind == 'freeze_selected' and self.visual.timestamp_sec >= self.start_sec:
            raise ValueError('frozen frame must precede the replacement boundary')
        if self.fade_out_sec > self.end_sec-self.start_sec:
            raise ValueError('ending fade must fit the tail')
        return self


def ending_bounds(operation: ExtendEndingOperation, duration_sec: float, fps: str | None = None):
    """Pure source-bound checks; native renderer repeats checks after probing."""
    start, end, fade = (Fraction(str(value)) for value in (operation.start_sec,operation.end_sec,operation.fade_out_sec))
    duration_sec = Fraction(str(duration_sec))
    freeze = Fraction(str(operation.visual.timestamp_sec)) if operation.visual.kind == 'freeze_selected' else None
    def limits():
        if not (0 < start < duration_sec < end <= 120 and end-duration_sec <= 10 and end-start <= 10):
            raise ValueError('ending must replace a suffix and extend by at most ten seconds within 120 seconds')
        if freeze is not None and not 0 <= freeze < start:
            raise ValueError('frozen frame must precede the replacement boundary')
        if fade > end-start:
            raise ValueError('ending fade must fit the tail')
        if not 0 <= fade <= 2:
            raise ValueError('ending fade must be within two seconds')
    limits()
    if fps is not None:
        rate = Fraction(fps)
        values = (start, end, fade) + (() if freeze is None else (freeze,))
        if any(abs(value*rate-round(value*rate)) > Fraction(1,10000) for value in values):
            raise ValueError('ending boundaries, frozen frame and fade must align to video frames')
        start, end, fade = (round(value*rate)/rate for value in (start,end,fade))
        if abs(duration_sec*rate-round(duration_sec*rate)) > Fraction(1,10000):
            raise ValueError('base duration must align to video frames')
        duration_sec = round(duration_sec*rate)/rate
        if freeze is not None:
            freeze = round(freeze*rate)/rate
        limits()
        if operation.fade_out_sec > 0 and round(fade*rate) < 2:
            raise ValueError('ending fade requires at least two frames')
    return start, end


class SubtitleStyle(ContractModel):
    font_family: Literal["DejaVu Sans", "Noto Sans CJK JP"] = "DejaVu Sans"
    font_size_ratio: Annotated[float, Field(strict=True, ge=0.02, le=0.12, allow_inf_nan=False)] = 0.055
    color_hex: Annotated[str, StringConstraints(strict=True, pattern=r"^#[0-9A-Fa-f]{6}$")] = "#FFFFFF"
    outline_px: Annotated[float, Field(strict=True, ge=0, le=32, allow_inf_nan=False)] = 2.0
    alignment: Literal["bottom", "top"] = "bottom"
    margin_ratio: Annotated[float, Field(strict=True, ge=0, le=0.3, allow_inf_nan=False)] = 0.06
    margin_horizontal_ratio: Annotated[float, Field(strict=True, ge=0, le=0.3, allow_inf_nan=False)] = 0.05
    bold: bool = False
    border_style: Literal[1, 3] = 1
    outline_color_hex: Annotated[str, StringConstraints(strict=True, pattern=r"^#[0-9A-Fa-f]{6}$")] = "#000000"
    outline_alpha: Annotated[int, Field(strict=True, ge=0, le=255)] = 0
    shadow_px: Annotated[float, Field(strict=True, ge=0, le=32, allow_inf_nan=False)] = 0.0


class SubtitleStylePatch(ContractModel):
    font_family: Literal["DejaVu Sans", "Noto Sans CJK JP"] | None = None
    font_size_ratio: Annotated[float, Field(strict=True, ge=0.02, le=0.12, allow_inf_nan=False)] | None = None
    color_hex: Annotated[str, StringConstraints(strict=True, pattern=r"^#[0-9A-Fa-f]{6}$")] | None = None
    outline_px: Annotated[float, Field(strict=True, ge=0, le=32, allow_inf_nan=False)] | None = None
    alignment: Literal["bottom", "top"] | None = None
    margin_ratio: Annotated[float, Field(strict=True, ge=0, le=0.3, allow_inf_nan=False)] | None = None
    margin_horizontal_ratio: Annotated[float, Field(strict=True, ge=0, le=0.3, allow_inf_nan=False)] | None = None
    bold: bool | None = None
    border_style: Literal[1, 3] | None = None
    outline_color_hex: Annotated[str, StringConstraints(strict=True, pattern=r"^#[0-9A-Fa-f]{6}$")] | None = None
    outline_alpha: Annotated[int, Field(strict=True, ge=0, le=255)] | None = None
    shadow_px: Annotated[float, Field(strict=True, ge=0, le=32, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def nonempty_patch(self):
        if not self.model_dump(exclude_none=True):
            raise ValueError("subtitle style patch must change at least one property")
        return self


class UpdateSubtitleCue(TimeRange):
    action: Literal["update"] = "update"
    cue_id: CueId
    text: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2000)]

    @field_validator("text")
    @classmethod
    def plain_text(cls, value: str) -> str:
        # Captions are data, including shell-looking words. Renderers must
        # escape ASS/HTML syntax; raw ASS overrides are not a supported input.
        if not value.strip() or "{\\" in value:
            raise ValueError("subtitle text must be nonempty plain text, without ASS overrides")
        if any(unicodedata.category(char) == "Cc" and char != "\n" for char in value):
            raise ValueError("subtitle text contains a control character")
        return value


class DeleteSubtitleCue(ContractModel):
    action: Literal["delete"] = "delete"
    cue_id: CueId


SubtitleChange = Annotated[
    Union[UpdateSubtitleCue, DeleteSubtitleCue], Field(discriminator="action"),
]


class SubtitleEditOperation(ContractModel):
    """Modify existing editable cues; no transcription or burned-text removal."""

    kind: Literal["subtitle_edit"] = "subtitle_edit"
    changes: Annotated[tuple[SubtitleChange, ...], Field(
        max_length=MAX_SUBTITLE_CHANGES,
    )] = ()
    style: SubtitleStylePatch | None = None

    @model_validator(mode="after")
    def unique_cues(self):
        if not self.changes and self.style is None:
            raise ValueError("subtitle edit requires cue changes or style")
        ids = [change.cue_id for change in self.changes]
        if len(ids) != len(set(ids)):
            raise ValueError("each subtitle cue may be changed only once")
        return self


EditOperation = Annotated[Union[
    TrimOperation, ImageFadeOperation, CutOperation, ReplaceFootageOperation,
    AudioSpliceOperation, FullTrackAudioOperation, SubtitleEditOperation, ExtendEndingOperation,
], Field(discriminator="kind")]


class EditPlan(ContractModel):
    schema_version: Literal[1] = 1
    run_id: EntityId
    base_version_id: EntityId
    operations: Annotated[tuple[EditOperation, ...], Field(
        min_length=1, max_length=MAX_OPERATIONS,
    )]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def unambiguous_operation_mix(self):
        if len(self.operations) > 1 and any(isinstance(op,ExtendEndingOperation) for op in self.operations):
            raise ValueError('ending extension must be the sole operation')
        if len(self.operations) > 1 and any(
            isinstance(op, (TrimOperation, CutOperation)) for op in self.operations
        ):
            raise ValueError("trim/cut must be the sole operation; edit their result in a child version")
        if sum(isinstance(op, SubtitleEditOperation) for op in self.operations) > 1:
            raise ValueError("combine subtitle changes into one subtitle_edit operation")
        channels: dict[str, list[tuple[float, float]]] = {"picture": [], "audio": []}
        for op in self.operations:
            if isinstance(op, (ImageFadeOperation, ReplaceFootageOperation)):
                channel, span = "picture", (op.start_sec, op.end_sec)
            elif isinstance(op, AudioSpliceOperation):
                channel, span = "audio", (op.start_sec, op.end_sec)
            elif isinstance(op, FullTrackAudioOperation):
                channel, span = "audio", (0.0, MAX_TIMELINE_SEC)
            else:
                continue
            if any(max(span[0], other[0]) < min(span[1], other[1]) for other in channels[channel]):
                raise ValueError(f"overlapping {channel} operations are ambiguous")
            channels[channel].append(span)
        return self


def validate_local_plan_capability(
    plan: EditPlan, *, duration_sec: float | None = None, fps: str | None = None,
) -> None:
    """Validate implemented media constraints without granting execution authority."""
    rate = float(Fraction(fps)) if fps is not None else None
    for operation in plan.operations:
        if isinstance(operation, ExtendEndingOperation):
            if duration_sec is not None:
                ending_bounds(operation,duration_sec,fps)
            continue
        if duration_sec is not None and isinstance(operation, TimeRange) and operation.end_sec > duration_sec + 1e-6:
            raise ValueError("operation exceeds video duration")
        if isinstance(operation, CutOperation) and duration_sec is not None:
            if operation.start_sec == 0 or operation.end_sec >= duration_sec:
                raise ValueError("local cut must be internal; use trim for an edge")
        if rate is not None and isinstance(operation, (TrimOperation, CutOperation, ReplaceFootageOperation, ImageFadeOperation)):
            if any(abs(value * rate - round(value * rate)) > .0001 for value in (operation.start_sec, operation.end_sec)):
                raise ValueError("local edit boundaries must align to video frames")
            if isinstance(operation, ImageFadeOperation) and round((operation.end_sec-operation.start_sec)*rate) < 2:
                raise ValueError("image fade requires at least two video frames")


class VersionManifest(ContractModel):
    """Trusted resolver snapshot, never supplied as planner authority."""

    version_id: EntityId
    run_id: EntityId
    duration_sec: Duration
    status: Literal["succeeded", "pending", "failed", "cancelled"]
    available: bool
    editable_subtitle_cue_ids: tuple[CueId, ...] = ()


class AssetManifest(ContractModel):
    asset_id: EntityId
    version_id: EntityId
    media_kind: Literal["video", "audio"]
    duration_sec: Duration
    has_audio: bool
    available: bool


class PlanSourceError(ValueError):
    """A shape-valid plan does not fit the trusted pinned source manifests."""


def validate_plan_sources(
    plan: EditPlan,
    *,
    base_version: VersionManifest,
    assets: Mapping[str, AssetManifest],
) -> None:
    """Check source bindings and durations, without I/O or authorization.

    The caller must supply authorized manifests, recheck them before execution,
    and pin immutable bytes. A matching UUID is not evidence of access rights.
    Bound comparisons are exact; runtime chooses frame/sample rounding explicitly.
    """
    if (plan.base_version_id, plan.run_id) != (base_version.version_id, base_version.run_id):
        raise PlanSourceError("base version/run does not match the pinned manifest")
    if base_version.status != "succeeded" or not base_version.available:
        raise PlanSourceError("base version must be successful and available")
    for op in plan.operations:
        if isinstance(op, ExtendEndingOperation):
            ending_bounds(op,base_version.duration_sec)
        if isinstance(op, TimeRange) and op.end_sec > base_version.duration_sec:
            raise PlanSourceError("operation exceeds the base duration")
        if isinstance(op, CutOperation) and op.end_sec - op.start_sec >= base_version.duration_sec:
            raise PlanSourceError("cut must leave a nonempty result")
        if isinstance(op, SubtitleEditOperation):
            known_cues = set(base_version.editable_subtitle_cue_ids)
            for change in op.changes:
                if change.cue_id not in known_cues:
                    raise PlanSourceError("subtitle cue is absent from the editable base track")
                if isinstance(change, UpdateSubtitleCue) and change.end_sec > base_version.duration_sec:
                    raise PlanSourceError("subtitle cue exceeds the base duration")
        if isinstance(op, (ReplaceFootageOperation, AudioSpliceOperation, FullTrackAudioOperation, ExtendEndingOperation)):
            asset = assets.get(op.source.asset_id)
            if asset is None or asset.asset_id != op.source.asset_id or not asset.available:
                raise PlanSourceError("source asset is unavailable in the trusted manifest")
            if asset.version_id != op.source.version_id:
                raise PlanSourceError("source asset/version binding does not match")
            if isinstance(op, ReplaceFootageOperation) and asset.media_kind != "video":
                raise PlanSourceError("footage replacement requires a video source")
            if isinstance(op, (AudioSpliceOperation, FullTrackAudioOperation, ExtendEndingOperation)) and not asset.has_audio:
                raise PlanSourceError("audio replacement requires a source with audio")
            required = (
                base_version.duration_sec if isinstance(op, FullTrackAudioOperation)
                else op.donor_duration_sec if isinstance(op,AudioSpliceOperation) and op.donor_duration_sec is not None
                else op.end_sec - op.start_sec
            )
            if op.donor_start_sec + required > asset.duration_sec:
                raise PlanSourceError("donor does not cover the required duration")
