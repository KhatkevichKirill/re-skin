"""
State machine for the v2 Project/Run/Segment status transitions.
Defines valid status enums and enforces allowed transition paths.

(The v1 Job pipeline was removed; SegmentStatus is now used by RunSegment.)
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Union


class SegmentStatus(str, enum.Enum):
    pending = "pending"
    uploading = "uploading"
    submitted = "submitted"
    generating = "generating"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


# --- v2 enums ---

class ProjectStatus(str, enum.Enum):
    created = "created"
    analyzing = "analyzing"
    ready = "ready"
    failed = "failed"


class RunStatus(str, enum.Enum):
    created = "created"
    # Optional GPT Image 2 reference-image preprocessing phase, entered from
    # `created` when Run.preprocess_enabled is true (see
    # docs/gpt-image-2-preprocess.md). Transitions to `queued` once all
    # preprocessed images are downloaded, then the normal submit/poll/finalize
    # chain runs. Runs without preprocessing skip straight created → queued.
    preprocessing = "preprocessing"
    # R2-new: GPT Image 2 preprocessing succeeded — the run PAUSES here so the
    # operator can review the preprocessed reference images and approve them
    # (or regenerate with new prompts/refs) before any Seedance/Gemini Omni
    # credit is spent. See docs/gpt-image-2-preprocess.md.
    preprocess_review = "preprocess_review"
    queued = "queued"
    processing = "processing"
    stitching = "stitching"
    delivering = "delivering"
    done = "done"
    failed = "failed"
    # One or more swap segments could not be generated (after retries). The
    # final video is intentionally NOT stitched; the operator re-runs the failed
    # segments and the run resumes from there.
    incomplete = "incomplete"


# Allowed transitions: {current_status -> set of valid next statuses}
SEGMENT_TRANSITIONS: dict[SegmentStatus, set[SegmentStatus]] = {
    SegmentStatus.pending:    {SegmentStatus.uploading, SegmentStatus.skipped},
    SegmentStatus.uploading:  {SegmentStatus.submitted, SegmentStatus.failed},
    SegmentStatus.submitted:  {SegmentStatus.generating, SegmentStatus.failed},
    SegmentStatus.generating: {SegmentStatus.completed, SegmentStatus.failed},
    SegmentStatus.completed:  set(),
    SegmentStatus.failed:     {SegmentStatus.pending},
    SegmentStatus.skipped:    set(),
}

# v2 transition tables
PROJECT_TRANSITIONS: dict[ProjectStatus, set[ProjectStatus]] = {
    ProjectStatus.created:   {ProjectStatus.analyzing},
    ProjectStatus.analyzing: {ProjectStatus.ready, ProjectStatus.failed},
    ProjectStatus.ready:     {ProjectStatus.analyzing},
    ProjectStatus.failed:    {ProjectStatus.analyzing},
}

RUN_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    # created → incomplete (2026-11-03): a VOID-from-the-source run is PARKED
    #   at creation (runs.generation_mode = void_source): nothing is submitted;
    #   its segments are filled from the source, then the run is requeued once
    #   for its stitch + delivery (docs/segment-fill-batch-design.md §7).
    RunStatus.created:    {RunStatus.queued, RunStatus.preprocessing, RunStatus.incomplete},
    # preprocessing → queued: legacy direct path (kept for any code path that
    #   still bypasses review — currently unused by the production chain).
    # preprocessing → failed: preprocessing itself failed (e.g. all tasks errored
    #   and the operator hasn't retried). Mirrors the processing → failed edge.
    # preprocessing → preprocess_review (R2-new): all GPT Image 2 tasks
    #   succeeded — the run PAUSES for human approval before Seedance starts.
    RunStatus.preprocessing: {
        RunStatus.queued, RunStatus.failed, RunStatus.preprocess_review,
    },
    # preprocess_review → queued (R2-new): operator approved the preprocessed
    #   images — submit is enqueued.
    # preprocess_review → preprocessing (R2-new): operator chose to regenerate
    #   with new prompts/refs — preprocessing re-runs.
    RunStatus.preprocess_review: {
        RunStatus.queued, RunStatus.preprocessing,
    },
    RunStatus.queued:     {RunStatus.processing},
    # processing → queued: allowed so an orphaned run (crash-resumed) can be
    # reset to queued before re-enqueuing.  Only reconcile_orphaned_runs and the
    # /retry endpoint use this edge — normal forward processing never does.
    # processing → incomplete: some swap segments failed after retries; we stop
    # before stitching so no mixed (swapped + original) video is produced.
    # processing → done: stitching disabled for the run (stitch_plan == []) —
    # all segments completed and there is no final video to stitch or deliver.
    RunStatus.processing: {
        RunStatus.queued, RunStatus.stitching, RunStatus.failed,
        RunStatus.incomplete, RunStatus.done,
    },
    RunStatus.stitching:  {RunStatus.delivering, RunStatus.failed},
    RunStatus.delivering: {RunStatus.done, RunStatus.failed},
    RunStatus.done:       {RunStatus.queued},  # allowed for per-segment re-run
    # failed → preprocessing: the operator retried a run whose GPT Image 2
    #   phase left an empty reference slot. failed → queued (the only other
    #   edge) goes straight to submit, which cannot upload a None and fails the
    #   run again — a permanent dead end for the operator (run ce9b5bfd,
    #   2026-08-27). Only /retry takes this edge, and only when the run really
    #   has a hole and no Seedance task was ever billed; every other failure
    #   still retries through queued.
    RunStatus.failed:     {RunStatus.queued, RunStatus.preprocessing},
    # incomplete → queued: operator re-runs the failed segment(s).
    RunStatus.incomplete: {RunStatus.queued},
}


# --- Overlay enum (standalone "burn a PNG onto a finished video" feature) ---
# Shares NO state with the run/generation pipeline (see docs/overlay spec §5).

class OverlayStatus(str, enum.Enum):
    created    = "created"
    queued     = "queued"
    processing = "processing"
    delivering = "delivering"   # only entered when deliver_to_drive=True
    done       = "done"
    failed     = "failed"


OVERLAY_TRANSITIONS: dict[OverlayStatus, set[OverlayStatus]] = {
    OverlayStatus.created:    {OverlayStatus.queued},
    OverlayStatus.queued:     {OverlayStatus.processing},
    OverlayStatus.processing: {OverlayStatus.delivering, OverlayStatus.done, OverlayStatus.failed},
    OverlayStatus.delivering: {OverlayStatus.done, OverlayStatus.failed},
    OverlayStatus.done:       {OverlayStatus.queued},   # allow re-run
    OverlayStatus.failed:     {OverlayStatus.queued},   # allow retry
}


# --- PIP Insert Merge enum ---

class MergeStatus(str, enum.Enum):
    queued     = "queued"
    compositing = "compositing"
    delivering = "delivering"
    done       = "done"
    failed     = "failed"


MERGE_TRANSITIONS: dict[MergeStatus, set[MergeStatus]] = {
    MergeStatus.queued:      {MergeStatus.compositing, MergeStatus.failed},
    MergeStatus.compositing: {MergeStatus.delivering, MergeStatus.failed},
    MergeStatus.delivering:  {MergeStatus.done, MergeStatus.failed},
    MergeStatus.done:        {MergeStatus.queued},      # allow re-run
    MergeStatus.failed:      {MergeStatus.queued},      # allow retry
}


# --- Tail Edit enum ("Change ending" — trim a done run's final video and
# optionally append a stitch-library clip; see app/pipeline_tail.py) ---

class TailEditStatus(str, enum.Enum):
    queued     = "queued"
    processing = "processing"
    delivering = "delivering"
    done       = "done"
    failed     = "failed"


TAIL_EDIT_TRANSITIONS: dict[TailEditStatus, set[TailEditStatus]] = {
    TailEditStatus.queued:     {TailEditStatus.processing, TailEditStatus.failed},
    TailEditStatus.processing: {TailEditStatus.delivering, TailEditStatus.failed},
    TailEditStatus.delivering: {TailEditStatus.done, TailEditStatus.failed},
    TailEditStatus.done:       {TailEditStatus.queued},      # allow re-run
    TailEditStatus.failed:     {TailEditStatus.queued},      # allow retry
}


# --- Segment image job enum (per-segment GPT Image 2 regeneration — one
# attempt at replacing a swap segment's generated reference image before a
# Re-run; see docs/gpt-image-2-preprocess.md and app/pipeline_segment_image.py)
#
# Starts at `queued`, not a `created` draft like OverlayStatus: every row here
# is born from a submitted request that already has a worker job behind it, so
# unlike OverlayJob.created (an editor state with no RQ job, deliberately
# excluded from the recovery reaper) there is no pre-queue state to represent.
# `done`/`failed` are terminal — a fresh attempt is a NEW row (the operator's
# "Regenerate image" click), not a transition back to `queued`, so the
# attempt history stays a plain append-only list. ---

class SegmentImageStatus(str, enum.Enum):
    queued     = "queued"
    generating = "generating"
    done       = "done"
    failed     = "failed"


SEGMENT_IMAGE_TRANSITIONS: dict[SegmentImageStatus, set[SegmentImageStatus]] = {
    SegmentImageStatus.queued:     {SegmentImageStatus.generating, SegmentImageStatus.failed},
    SegmentImageStatus.generating: {SegmentImageStatus.done, SegmentImageStatus.failed},
    SegmentImageStatus.done:       set(),
    SegmentImageStatus.failed:     set(),
}


# --- Caption enum ("Add Captions" — re-transcribe a done Run's delivered
# result, burn subtitles in with ffmpeg, deliver as a new derived creative;
# see docs/burned-captions.md and app/pipeline_captions.py) ---

class CaptionStatus(str, enum.Enum):
    queued       = "queued"
    transcribing = "transcribing"
    burning      = "burning"
    delivering   = "delivering"
    done         = "done"
    failed       = "failed"


CAPTION_TRANSITIONS: dict[CaptionStatus, set[CaptionStatus]] = {
    # burning: manual lines (lines_source == "manual") skip the transcribe step.
    CaptionStatus.queued:       {CaptionStatus.transcribing, CaptionStatus.burning, CaptionStatus.failed},
    CaptionStatus.transcribing: {CaptionStatus.burning, CaptionStatus.failed},
    CaptionStatus.burning:      {CaptionStatus.delivering, CaptionStatus.failed},
    CaptionStatus.delivering:   {CaptionStatus.done, CaptionStatus.failed},
    CaptionStatus.done:         {CaptionStatus.queued},      # allow re-run
    CaptionStatus.failed:       {CaptionStatus.queued},      # allow retry
}


# --- Range attempt enum (run-local swap-segment range regeneration — a
# candidate revision of ONE swap segment's time range + generation, held apart
# from the Run/RunSegment it targets until it succeeds; see
# app/pipeline_range_attempt.py and docs/run-local-segment-range-regeneration.md)
#
# Starts at `queued`, like SegmentImageStatus: every row is born from a
# submitted, already-validated request with a worker job behind it. `done` and
# `failed` are terminal — a fresh attempt (retry, or a corrected range) is a
# NEW row, never a transition back to `queued`, so an operator can see every
# past attempt against a segment (what range was requested, whether it
# promoted) instead of only the latest one. `stitching`/`delivering` mirror the
# Run's own finalize phase — a candidate re-stitches and re-delivers before it
# is trusted enough to promote. ---

class RangeAttemptStatus(str, enum.Enum):
    queued     = "queued"
    generating = "generating"
    stitching  = "stitching"
    delivering = "delivering"
    done       = "done"
    failed     = "failed"


RANGE_ATTEMPT_TRANSITIONS: dict[RangeAttemptStatus, set[RangeAttemptStatus]] = {
    RangeAttemptStatus.queued:     {RangeAttemptStatus.generating, RangeAttemptStatus.failed},
    RangeAttemptStatus.generating: {RangeAttemptStatus.stitching, RangeAttemptStatus.failed},
    # `stitching -> done` (skipping `delivering`) is reachable for a NO-STITCH
    # run (Run.stitch_plan == []): _stitch_candidate returns without setting
    # attempt.final_local_path, so poll_range_attempt_once never has anything
    # to enter `delivering` for — the candidate segment result IS the
    # deliverable, exactly like the main pipeline's no-stitch run never
    # reaches a "deliver a final video" step either.
    RangeAttemptStatus.stitching:  {RangeAttemptStatus.delivering, RangeAttemptStatus.done, RangeAttemptStatus.failed},
    RangeAttemptStatus.delivering: {RangeAttemptStatus.done, RangeAttemptStatus.failed},
    RangeAttemptStatus.done:       set(),
    RangeAttemptStatus.failed:     set(),
}


RANGE_ATTEMPT_ACTIVE_STATUSES: set[RangeAttemptStatus] = {
    RangeAttemptStatus.queued, RangeAttemptStatus.generating,
    RangeAttemptStatus.stitching, RangeAttemptStatus.delivering,
}


# --- Audio patch enum (re-voice one time window of a done run's delivered
# final from a sibling run's audio — see app/pipeline_audio_patch.py and
# docs/run-audio-patch.md)
#
# `done` and `failed` are TERMINAL, like RangeAttemptStatus and unlike
# TailEditStatus: an audio patch is not a re-runnable job but a recorded
# change to the run's own result, so a second correction is a NEW row patching
# the (already patched) current result, never this row going round again. That
# keeps the card an append-only history of what was done to this run's audio.

class AudioPatchStatus(str, enum.Enum):
    queued     = "queued"
    processing = "processing"
    delivering = "delivering"
    done       = "done"
    failed     = "failed"


AUDIO_PATCH_TRANSITIONS: dict[AudioPatchStatus, set[AudioPatchStatus]] = {
    AudioPatchStatus.queued:     {AudioPatchStatus.processing, AudioPatchStatus.failed},
    AudioPatchStatus.processing: {AudioPatchStatus.delivering, AudioPatchStatus.failed},
    AudioPatchStatus.delivering: {AudioPatchStatus.done, AudioPatchStatus.failed},
    AudioPatchStatus.done:       set(),
    AudioPatchStatus.failed:     set(),
}


AUDIO_PATCH_ACTIVE_STATUSES: set[AudioPatchStatus] = {
    AudioPatchStatus.queued, AudioPatchStatus.processing,
    AudioPatchStatus.delivering,
}


# --- Provider task enum (one paid request to a second provider — Fal in Stage
# 1; see app/pipeline_provider.py and docs/fal-provider-design.md §4)
#
# The table below IS the design's transition map, edge for edge. `done`,
# `refused`, `failed` and `abandoned` are terminal; a `refused` row that still
# holds its reservation is terminal for this table but still ACTIVE (it may have
# been billed and owes the operator's close/abandon), and its two operator
# edges (`refused -> refused` close, `refused -> abandoned`) are listed here and
# gated on `reservation_held` by pipeline_provider. `submit_unknown` is parked:
# no job works on it, only the operator. Nothing ever moves a row back to
# `queued` or `submit_started`. ---

class ProviderTaskStatus(str, enum.Enum):
    queued               = "queued"
    submit_started       = "submit_started"
    accepted_unrecorded  = "accepted_unrecorded"
    submitted            = "submitted"
    fetching             = "fetching"
    cancelling           = "cancelling"
    needs_reconciliation = "needs_reconciliation"
    closing              = "closing"
    done                 = "done"
    refused              = "refused"
    failed               = "failed"
    submit_unknown       = "submit_unknown"
    abandoned            = "abandoned"


_PT = ProviderTaskStatus
PROVIDER_TASK_TRANSITIONS: dict[ProviderTaskStatus, set[ProviderTaskStatus]] = {
    _PT.queued:               {_PT.abandoned, _PT.submit_started, _PT.failed},
    _PT.submit_started:       {_PT.submitted, _PT.accepted_unrecorded, _PT.refused,
                               _PT.submit_unknown, _PT.closing},
    _PT.submit_unknown:       {_PT.submitted, _PT.abandoned, _PT.closing},
    _PT.accepted_unrecorded:  {_PT.submitted, _PT.closing},
    _PT.submitted:            {_PT.fetching, _PT.failed, _PT.cancelling,
                               _PT.needs_reconciliation, _PT.closing},
    _PT.fetching:             {_PT.done, _PT.refused, _PT.failed, _PT.closing},
    _PT.cancelling:           {_PT.failed, _PT.fetching, _PT.needs_reconciliation,
                               _PT.closing},
    _PT.needs_reconciliation: {_PT.fetching, _PT.failed, _PT.cancelling, _PT.closing},
    _PT.closing:              {_PT.failed},
    _PT.refused:              {_PT.refused, _PT.abandoned},
    _PT.done:                 set(),
    _PT.failed:               set(),
    _PT.abandoned:            set(),
}

PROVIDER_TASK_TERMINAL_STATUSES: frozenset[ProviderTaskStatus] = frozenset(
    {_PT.done, _PT.refused, _PT.failed, _PT.abandoned}
)
PROVIDER_TASK_NON_TERMINAL_STATUSES: frozenset[ProviderTaskStatus] = frozenset(
    s for s in ProviderTaskStatus if s not in PROVIDER_TASK_TERMINAL_STATUSES
)
# What deploy/drain-check.sql counts: every status a job works on, i.e. the
# non-terminal ones except the parked `submit_unknown`.
PROVIDER_TASK_DRAIN_STATUSES: frozenset[ProviderTaskStatus] = (
    PROVIDER_TASK_NON_TERMINAL_STATUSES - {_PT.submit_unknown}
)


def _sql_in(statuses) -> str:
    return ", ".join(f"'{s.value}'" for s in sorted(statuses, key=lambda s: s.value))


# The ACTIVE predicate (design §4): non-terminal, or a refused row still holding
# its reservation. Spelled once; the partial unique index in models.py and the
# migration use this exact text.
PROVIDER_TASK_ACTIVE_WHERE: str = (
    f"(status IN ({_sql_in(PROVIDER_TASK_NON_TERMINAL_STATUSES)}) "
    "OR (status = 'refused' AND reservation_held))"
)
PROVIDER_TASK_STATUS_CHECK: str = f"status IN ({_sql_in(ProviderTaskStatus)})"
del _PT


class InvalidTransition(Exception):
    """Raised when a state transition is not permitted."""

    def __init__(self, current: enum.Enum, target: enum.Enum) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid transition: {current.value!r} -> {target.value!r}"
        )


def can_transition(
    current: Union[SegmentStatus, ProjectStatus, RunStatus, OverlayStatus, MergeStatus, TailEditStatus, CaptionStatus, SegmentImageStatus, RangeAttemptStatus, AudioPatchStatus, ProviderTaskStatus],
    target: Union[SegmentStatus, ProjectStatus, RunStatus, OverlayStatus, MergeStatus, TailEditStatus, CaptionStatus, SegmentImageStatus, RangeAttemptStatus, AudioPatchStatus, ProviderTaskStatus],
) -> bool:
    """Return True if transitioning from current to target is permitted."""
    if isinstance(current, SegmentStatus):
        return target in SEGMENT_TRANSITIONS.get(current, set())
    if isinstance(current, ProjectStatus):
        return target in PROJECT_TRANSITIONS.get(current, set())
    if isinstance(current, RunStatus):
        return target in RUN_TRANSITIONS.get(current, set())
    if isinstance(current, OverlayStatus):
        return target in OVERLAY_TRANSITIONS.get(current, set())
    if isinstance(current, MergeStatus):
        return target in MERGE_TRANSITIONS.get(current, set())
    if isinstance(current, TailEditStatus):
        return target in TAIL_EDIT_TRANSITIONS.get(current, set())
    if isinstance(current, CaptionStatus):
        return target in CAPTION_TRANSITIONS.get(current, set())
    if isinstance(current, SegmentImageStatus):
        return target in SEGMENT_IMAGE_TRANSITIONS.get(current, set())
    if isinstance(current, RangeAttemptStatus):
        return target in RANGE_ATTEMPT_TRANSITIONS.get(current, set())
    if isinstance(current, AudioPatchStatus):
        return target in AUDIO_PATCH_TRANSITIONS.get(current, set())
    if isinstance(current, ProviderTaskStatus):
        return target in PROVIDER_TASK_TRANSITIONS.get(current, set())
    raise TypeError(f"Unsupported status type: {type(current)}")


def transition(
    entity: object,
    target: Union[SegmentStatus, ProjectStatus, RunStatus, OverlayStatus, MergeStatus, TailEditStatus, CaptionStatus, SegmentImageStatus, RangeAttemptStatus, AudioPatchStatus],
) -> None:
    """
    Apply a status transition to a Segment, VideoProject, Run, OverlayJob,
    InsertMerge, RunTailEdit, RunCaption, RunSegmentImageJob, or
    RunSegmentRangeAttempt or RunAudioPatch ORM instance.

    Updates entity.status and entity.updated_at.
    Raises InvalidTransition if the move is not allowed.
    """
    current = entity.status  # type: ignore[attr-defined]
    if not can_transition(current, target):
        raise InvalidTransition(current, target)
    entity.status = target  # type: ignore[attr-defined]
    entity.updated_at = datetime.now(timezone.utc)  # type: ignore[attr-defined]
