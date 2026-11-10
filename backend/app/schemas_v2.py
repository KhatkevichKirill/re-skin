"""
Pydantic V2 schemas for the v2 REST API (VideoProject + Runs).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Project schemas
# ---------------------------------------------------------------------------


class ProjectCreateResponse(BaseModel):
    project_id: str
    status: str


class ProjectResponse(BaseModel):
    id: str
    name: Optional[str] = None
    status: str
    project_type: str = "face_swap"
    # AICP-363: "manual" | "assistant" (set at creation only).
    mode: str = "manual"
    source_type: str
    source_ref: str
    source_local_path: Optional[str] = None
    # "Move to new project" lineage: the owning run, and (AICP-331) the step of
    # it whose output was copied — "caption" | "tail_edit" | "overlay" + its
    # id; both NULL for the run's final video. Soft references.
    source_run_id: Optional[str] = None
    source_kind: Optional[str] = None
    source_artifact_id: Optional[str] = None
    mute_source: bool = False
    # Analyze-time cap on proposed segment length, in seconds. NULL = no
    # explicit choice, i.e. the universal default every model can generate
    # (ai_models.UNIVERSAL_MAX_SEGMENT_SEC).
    max_segment_sec: Optional[float] = None
    # Length of the localised hook, in seconds — how much of the front of the
    # video a `localisation` project re-generates. Like max_segment_sec this is
    # consumed at ANALYZE time. NULL = use settings.LOCALISATION_DEFAULT_HOOK_SEC.
    hook_sec: Optional[float] = None
    # "hook" | "whole" (AICP-340) — the localisation strategy that cut the
    # current segments. Changes only via POST /projects/{pid}/analyze.
    localisation_scope: str = "hook"
    # The scope the next analysis will apply, when one was requested and has
    # not been applied yet (NULL otherwise).
    localisation_scope_requested: Optional[str] = None
    # QA caption detector config (non-default fields only); NULL = defaults.
    # See app.qa_tools.CaptionDetectorConfig.
    caption_detector: Optional[dict] = None
    # Where burned captions sit (AICP-368, app.caption_placement); NULL = the
    # house bottom position. source.kind = derived | inherited | manual.
    caption_placement: Optional[dict] = None
    # Subtitle removal: VOID reservations the autopilot starts without asking
    # on one run (USD); NULL = settings.AUTOPILOT_VOID_AUTO_USD.
    void_auto_usd_per_run: Optional[float] = None
    # Subtitle removal: "void_source" | "omni"; NULL reads as "omni".
    caption_removal_method: Optional[str] = None

    # Probe fields
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    aspect_ratio: Optional[str] = None

    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectListItem(BaseModel):
    id: str
    name: Optional[str] = None
    source_ref: str
    status: str
    project_type: str = "face_swap"
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectUpdate(BaseModel):
    """Editable project settings. Omitted fields are left unchanged."""

    name: Optional[str] = None
    mute_source: Optional[bool] = None
    # Analyze-time segmentation cap in seconds. Unlike the fields above, NULL is
    # a real value rather than a synonym for "omitted": sending an explicit
    # `null` CLEARS the cap back to ai_models.UNIVERSAL_MAX_SEGMENT_SEC, while
    # leaving the key out of the request body leaves the column untouched.
    # update_project tells the two apart via model_fields_set. Because the cap
    # is consumed at analyze time, changing it does NOT re-cut existing
    # segments.
    max_segment_sec: Optional[float] = None
    # Localised-hook length in seconds. Same NULL-is-a-value rule as
    # max_segment_sec above (explicit `null` clears it back to
    # settings.LOCALISATION_DEFAULT_HOOK_SEC; an omitted key leaves the column
    # alone), and the same analyze-time contract: changing it does NOT re-cut
    # the segments an analyzed project already has.
    hook_sec: Optional[float] = None
    # Declared only so a PATCH carrying it is REFUSED (400) rather than
    # silently ignored: the scope changes only with a re-analysis
    # (POST /projects/{pid}/analyze?localisation_scope=), see update_project.
    localisation_scope: Optional[str] = None
    # QA caption detector config (app.qa_tools.CaptionDetectorConfig). Same
    # NULL-is-a-value rule: an explicit `null` (or `{}`, or an all-defaults
    # object) resets to the defaults; an omitted key leaves the column alone.
    # The object REPLACES the stored one -- it is not merged. Validated by
    # update_project (bounded ranges, #RGB/#RRGGBB or a known colour name) ->
    # 400, the same way the float fields above are: a 422 would echo the
    # input, and a non-finite number in it cannot be JSON-encoded.
    caption_detector: Optional[dict] = None
    # Caption placement (AICP-368, app.caption_placement.CaptionPlacement):
    # {"mode": "band", "center_y": 0.62, "max_width": null, "max_rows": 3}.
    # An explicit `null` stores the sticky house position ({"mode":
    # "bottom"}) that a later derivation never overwrites; an omitted key
    # leaves the column alone. `source` is server-written (kind "manual").
    # Validated by update_project -> 400.
    caption_placement: Optional[dict] = None
    # Subtitle removal (2026-11-03): the per-run VOID allowance the autopilot
    # starts without asking, USD 0..20. NULL-is-a-value (explicit null = the
    # default setting); an omitted key leaves the column alone.
    void_auto_usd_per_run: Optional[float] = None
    # Subtitle removal (2026-11-03): "void_source" (every swap segment a VOID
    # fill from the source, Omni only as a per-segment fallback) or "omni".
    # Applies to the NEXT run (a run freezes it). An omitted key leaves it.
    caption_removal_method: Optional[str] = None


# ---------------------------------------------------------------------------
# Localisation schemas (docs/localisation.md §4.4)
# ---------------------------------------------------------------------------


class TranscriptResponse(BaseModel):
    """The cached source transcript plus the transcription task's lifecycle.

    Returned by all three transcript endpoints (POST /transcribe, GET and PATCH
    /transcript) so a caller only ever parses one shape.

    *status* is ``pending | running | ready | failed | empty``, or None when
    transcription was never requested; *transcript* is the docs/localisation.md
    §4.1 dict verbatim (None until the first successful run). Note that the two
    are independent: re-transcribing sets ``status="pending"`` while the
    PREVIOUS transcript stays readable here until the task overwrites it.
    """

    status: Optional[str] = None
    error: Optional[str] = None
    transcript: Optional[dict] = None


class LocalisationPromptResponse(BaseModel):
    """Everything the New Run form needs to pre-fill a localisation run.

    Pure output — the endpoint that returns this writes nothing to the
    database. The operator edits the text and submits it through the ordinary
    `create_run` path, which is what actually records the run.

    *segment_prompts* is keyed by ``segment_def_id`` and is populated ONLY when
    the hook spans more than one swap SegmentDef; with the usual single-segment
    hook it is ``{}`` and the whole dialogue lives in *prompt*. When it IS
    populated it covers EVERY swap segment of the hook, including a silent one
    (whose prompt then carries an empty dialogue block) — a per-segment prompt
    REPLACES the run prompt rather than extending it, so a segment left out of
    the map would fall back to re-speaking the entire hook. Each line appears in
    exactly one segment's prompt, chosen by which segment contains the line's
    midpoint.

    *lines* are the translated §4.1 lines for the hook window — same ids, each
    carrying ``source_text`` alongside the translated ``text`` — so the UI can
    show the operator both versions side by side. *prompt* always carries all of
    them, including any that fall in a stretch of the hook no swap segment
    covers; those are flagged by name in *warnings* rather than dropped, since
    an operator can fix the segmentation but cannot recover text they were never
    shown.
    """

    # Detected in the transcript, not chosen by the caller; "" when the
    # transcription model had nothing to go on.
    source_language: str = ""
    target_language: str
    # The window the transcript was actually sliced to, [0, hook_sec). This is
    # NOT necessarily VideoProject.hook_sec: that column is the operator's
    # intent and drives the next analysis, while the default window here is the
    # ANALYZED reality (where the project's swap segments end), and the request
    # may override both. A project asking for a 15s hook under a 10s
    # segmentation cap keeps hook_sec=15 in the column and gets 10.0 here.
    # Echoed precisely so the UI can show what it really cut at.
    hook_sec: float
    prompt: str
    segment_prompts: dict[str, str] = {}
    lines: list[dict] = []
    # Operator-facing advisories (a line cut mid-sentence at the hook edge or a
    # segment seam, a translation whose spoken length looks implausible,
    # source == target, a hook window swap segments do not fully cover, and a
    # line stranded in one of those uncovered gaps). Never fatal: the prompt is
    # returned regardless.
    warnings: list[str] = []


class LocalisationPromptJobAccepted(BaseModel):
    """202 answer of ``POST /projects/{pid}/localisation-prompt`` (AICP-338).

    The translation runs as a background job; poll
    ``GET /projects/{pid}/localisation-prompt/jobs/{job_id}``. *deduplicated*
    is True when an identical request was already queued or running and this
    one joined it (nothing new was reserved or enqueued)."""

    job_id: str
    status: str = "queued"
    deduplicated: bool = False


class LocalisationPromptJobError(BaseModel):
    # The HTTP status the synchronous endpoint would have answered with
    # (502 model failure, 409 project changed, 503 interrupted, 504 timed out,
    # 500 crashed).
    status_code: int
    detail: str


class LocalisationPromptJobStatus(BaseModel):
    """One poll of a localisation translate job.

    *status*: ``queued`` | ``running`` | ``done`` (``result`` set) |
    ``failed`` (``error`` set) | ``expired`` (Redis no longer has the job —
    press Translate again). Writes nothing, like the POST."""

    job_id: str
    status: str
    result: Optional[LocalisationPromptResponse] = None
    error: Optional[LocalisationPromptJobError] = None
    language: Optional[str] = None
    swap_character: Optional[bool] = None
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Caption schemas ("Add Captions" — re-transcribe a done Run's delivered
# result, burn subtitles in with ffmpeg, deliver as a new derived creative;
# see docs/burned-captions.md and app/pipeline_captions.py)
# ---------------------------------------------------------------------------


class CaptionCreateResponse(BaseModel):
    caption_id: str
    status: str


class CaptionResponse(BaseModel):
    id: str
    run_id: str
    # One of app/languages.py's codes, or None — resolved at process time as
    # caption.language or run.language or the transcript's source_language.
    language: Optional[str] = None
    # The timed lines actually burned: [{"start": float, "end": float, "text":
    # str}, ...]. None until the transcribe step (or an operator PATCH) writes it.
    lines: Optional[list] = None
    # "transcribe", "manual" or "reused". "manual" means an operator
    # supplied/edited `lines` and the transcribe step is skipped (queued ->
    # burning directly). "reused" means the lines were copied from the caption
    # `lines_reused_from`, whose source had the same audio.
    lines_source: Optional[str] = None
    lines_reused_from: Optional[str] = None
    # captions.CAPTION_STYLE_VERSION at burn time.
    style_version: Optional[str] = None
    # "run" (default, v1 behaviour) or "tail_edit" — which video this job
    # actually transcribes/burns. See docs/burned-captions.md §3.7.
    source_type: str = "run"
    # Soft reference to run_tail_edits.id when source_type == "tail_edit";
    # None when source_type == "run".
    source_tail_edit_id: Optional[str] = None
    # The placement this job burns with (AICP-368), frozen at creation; None =
    # the house position.
    placement: Optional[dict] = None
    status: str
    output_path: Optional[str] = None
    gdrive_file_id: Optional[str] = None
    error_message: Optional[str] = None
    final_creatives_exported_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CaptionListItem(BaseModel):
    id: str
    status: str
    language: Optional[str] = None
    lines_source: Optional[str] = None
    # "run" (default, v1 behaviour) or "tail_edit" — which video this job
    # actually transcribes/burns. See docs/burned-captions.md §3.7.
    source_type: str = "run"
    # Soft reference to run_tail_edits.id when source_type == "tail_edit";
    # None when source_type == "run". Resolving a deleted tail edit to a
    # display label is the UI's job (mirrors library_clip_name), not this
    # schema's.
    source_tail_edit_id: Optional[str] = None
    # AICP-368: the frozen placement (None = house position).
    placement: Optional[dict] = None
    created_at: datetime
    error_message: Optional[str] = None
    gdrive_file_id: Optional[str] = None
    final_creatives_exported_at: Optional[datetime] = None
    # len(lines) — not a model column; the endpoint sets this after
    # model_validate() so the list view can show a count without shipping the
    # full `lines` array over the wire.
    line_count: int = 0

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Audio patch schemas ("Fix audio from another run")
# ---------------------------------------------------------------------------


class AudioPatchResponse(BaseModel):
    id: str
    run_id: str
    # AICP-389: NULL for a same-run low-pass fix (kind "same_run_lowpass").
    kind: str = "donor"
    donor_run_id: Optional[str] = None
    target_start_sec: float
    target_end_sec: float
    donor_start_sec: Optional[float] = None
    crossfade_sec: float
    status: str
    output_path: Optional[str] = None
    result_duration_sec: Optional[float] = None
    # The Drive file whose content now holds the patched video. Normally the
    # run's own result_gdrive_file_id — the link does not change.
    gdrive_file_id: Optional[str] = None
    error_message: Optional[str] = None
    promoted_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AudioPatchListItem(BaseModel):
    id: str
    kind: str = "donor"
    donor_run_id: Optional[str] = None
    # Human label for the donor ("RplDcTop · 2714a7c3"), resolved by the
    # endpoint — a donor deleted since is shown by id alone, the same way
    # TailEditListItem.library_clip_name degrades.
    donor_label: Optional[str] = None
    target_start_sec: float
    target_end_sec: float
    donor_start_sec: Optional[float] = None
    crossfade_sec: float
    status: str
    result_duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    promoted_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# PIP insert extraction schemas
# ---------------------------------------------------------------------------


class ExtractInsertResponse(BaseModel):
    child_project_id: str
    extraction_id: str
    status: str


# ---------------------------------------------------------------------------
# PIP insert merge schemas
# ---------------------------------------------------------------------------


class MergeCreateResponse(BaseModel):
    merge_id: str
    status: str


class MergeResponse(BaseModel):
    id: str
    target_project_id: str
    child_run_id: str
    x: int
    y: int
    width: int
    height: int
    start_sec: float
    end_sec: float
    corner_radius_px: int
    status: str
    output_path: Optional[str] = None
    gdrive_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MergeListItem(BaseModel):
    id: str
    child_run_id: str
    status: str
    gdrive_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Tail edit schemas ("Change ending" — trim + optional library-clip append)
# ---------------------------------------------------------------------------


class TailEditCreateResponse(BaseModel):
    tail_edit_id: str
    status: str


class TailEditResponse(BaseModel):
    id: str
    run_id: str
    cut_sec: float
    library_clip_id: Optional[str] = None
    status: str
    output_path: Optional[str] = None
    gdrive_file_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TailEditListItem(BaseModel):
    id: str
    cut_sec: float
    library_clip_id: Optional[str] = None
    library_clip_name: Optional[str] = None
    # "run" (default, v1 behaviour) or "caption" — which video this job
    # actually trims. See docs/burned-captions.md §3.7.
    source_type: str = "run"
    # Soft reference to run_captions.id when source_type == "caption"; None
    # when source_type == "run". Resolving a deleted caption to a display
    # label is the UI's job (mirrors library_clip_name), not this schema's.
    source_caption_id: Optional[str] = None
    status: str
    gdrive_file_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    final_creatives_exported_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# SegmentDef schemas
# ---------------------------------------------------------------------------


class SegmentDefResponse(BaseModel):
    id: str
    project_id: str
    index: int
    start_sec: float
    end_sec: float
    has_face: bool
    action: str
    pre_roll_sec: float
    post_roll_sec: float
    created_at: datetime

    model_config = {"from_attributes": True}


class RecutBoundaryResponse(BaseModel):
    """One interior boundary of a speech-pause re-cut (AICP-347)."""

    index: int
    old_sec: float
    new_sec: float
    moved: bool
    source: str  # kept | scene_cut | pause_middle | pause
    old_inside_line_ids: list
    new_inside_line_ids: list


class RecutResponse(BaseModel):
    """``POST /projects/{pid}/recut-at-pauses`` (AICP-347)."""

    project_id: str
    dry_run: bool
    applied: bool
    # Hash of every input of the plan; pass it back as expected_fingerprint so
    # an apply refuses (409) if anything changed since the preview.
    fingerprint: str
    feasible: bool
    changed: bool
    target_sec: float
    floor_sec: float
    scene_cuts_used: bool
    boundaries: list[RecutBoundaryResponse]
    warnings: list[str]
    segments: list[SegmentDefResponse]


class SegmentDefUpdate(BaseModel):
    """Fields that may be updated on an existing SegmentDef."""

    id: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    action: Optional[str] = None
    pre_roll_sec: Optional[float] = None
    post_roll_sec: Optional[float] = None


class NewSegmentDef(BaseModel):
    """Payload for creating a new SegmentDef during review."""

    start_sec: float
    end_sec: float
    action: str = "keep"
    has_face: bool = False
    pre_roll_sec: float = 0.0
    post_roll_sec: float = 0.0


class SegmentsUpdateRequest(BaseModel):
    updates: list[SegmentDefUpdate] = []
    deletes: list[str] = []
    creates: list[NewSegmentDef] = []


class SegmentRerunItem(BaseModel):
    """One segment of a multi-segment rerun (AICP-341).

    ``prompt``: omitted or null leaves the segment's saved prompt override
    untouched, ``""`` clears it (the run-level prompt applies), any other
    string becomes the override. Reference overrides are never touched here —
    set them with ``PATCH /runs/{rid}/segments/{rsid}`` first.
    """

    run_segment_id: str
    prompt: Optional[str] = None


class SegmentsRerunRequest(BaseModel):
    """Payload for ``POST /runs/{rid}/segments/rerun`` (AICP-341)."""

    segments: list[SegmentRerunItem]
    # AICP-342: required (409 otherwise) when the run's current result is a
    # promoted audio patch — the re-stitch drops the audio fix and replaces the
    # content behind the run's Drive link.
    discard_audio_patch: bool = False


class SegmentTakesRestoreRequest(BaseModel):
    """Payload for ``POST /runs/{rid}/segments/restore`` (AICP-342)."""

    takes: list[str]
    discard_audio_patch: bool = False


class RunSegmentTakeResponse(BaseModel):
    """One preserved previous output of a segment (AICP-342)."""

    id: str
    run_id: str
    run_segment_id: str
    index: int
    created_at: datetime
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    seedance_task_id: Optional[str] = None
    cost_credits: Optional[float] = None
    prompt_override: Optional[str] = None
    is_current: bool = False
    file_exists: bool = True
    # The retention sweep deleted the file (row kept as history); not restorable.
    file_purged: bool = False


# ---------------------------------------------------------------------------
# Library clip schemas (reusable stitch clips, e.g. alternative outros)
# ---------------------------------------------------------------------------


class LibraryClipResponse(BaseModel):
    id: str
    name: str
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    # Spoken language of the clip, one of app/languages.py's codes. NULL =
    # language-agnostic (no speech, or a tail that suits any language), which
    # is what every clip uploaded before this column existed is.
    language: Optional[str] = None
    # "end_card" | "insert" (models.LIBRARY_CLIP_KINDS, AICP-381).
    kind: str = "insert"
    created_at: datetime

    model_config = {"from_attributes": True}


class LibraryClipUpdate(BaseModel):
    """Fields an operator may edit on an existing library clip.

    Exists because ``language`` arrived long after the library did: without it
    the only way to tag the tails already uploaded would be to re-upload them.
    """

    name: Optional[str] = None
    # Language code, or "" to clear it back to language-agnostic (mirrors
    # RunUpdate.language). Omitting the key leaves the column untouched.
    language: Optional[str] = None
    # "end_card" or "insert" (AICP-381); omitted = untouched, anything else 400.
    kind: Optional[str] = None


# ---------------------------------------------------------------------------
# Voice clip schemas (reusable voice references for generated speech)
# ---------------------------------------------------------------------------


class VoiceClipResponse(BaseModel):
    id: str
    name: str
    # Duration of the normalised mp3, i.e. post-truncation to VOICE_REF_MAX_SEC.
    duration_sec: Optional[float] = None
    # Spoken language of the sample, one of app/languages.py's codes. NULL =
    # untagged; used only to pre-select a voice on a localisation run.
    language: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class VoiceClipUpdate(BaseModel):
    """Fields an operator may edit on an existing voice clip.

    Mirrors :class:`LibraryClipUpdate` — the audio itself is immutable (it was
    normalised at upload), so only the labels are editable.
    """

    name: Optional[str] = None
    # Language code, or "" to clear the tag. Omitting the key leaves it alone.
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# Run schemas
# ---------------------------------------------------------------------------


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    # Non-blocking launch advisories (AICP-340): e.g. a whole-video
    # localisation swap clip that pre/post-roll pushes past the whole-video
    # target. The run was created regardless.
    warnings: list[str] = []


class RunBatchCopyResponse(BaseModel):
    runs: list[RunCreateResponse] = []


class RunResponse(BaseModel):
    id: str
    project_id: str
    name: Optional[str] = None
    prompt: str
    reference_image_urls: list = []
    model: str = "seedance"
    resolution: str
    audio_mode: str = "original"
    # One of app/languages.py's codes; None = never chosen (behaves as English).
    language: Optional[str] = None
    gdrive_folder_id: Optional[str] = None
    # None = stitch all segments; [] = no final video; else custom entry list.
    stitch_plan: Optional[list] = None
    # Optional GPT Image 2 reference preprocessing (see
    # docs/gpt-image-2-preprocess.md).
    preprocess_enabled: bool = False
    preprocess_mode: Optional[str] = "per_image"
    preprocess_image_prompts: Optional[list] = None
    preprocessed_reference_urls: Optional[list] = None
    preprocess_task_ids: Optional[list] = None
    # True on copies that re-ran GPT Image 2 on a new photo: they skip the
    # preprocess_review pause because the source run's recipe was approved once
    # already, by hand.
    preprocess_auto_approve: bool = False
    status: str
    result_local_path: Optional[str] = None
    result_gdrive_file_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class RunUpdate(BaseModel):
    """Fields an operator may edit on an existing Run (name + language)."""

    name: Optional[str] = None
    # Language code, or "" to clear it back to unset (both behave as English
    # downstream; clearing simply drops the Job ID suffix again).
    language: Optional[str] = None


class RunListItem(BaseModel):
    id: str
    name: Optional[str] = None
    status: str
    created_at: datetime
    result_available: bool = False

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# RunSegment schemas
# ---------------------------------------------------------------------------


class RunSegmentResponse(BaseModel):
    id: str
    run_id: str
    segment_def_id: str
    index: int
    status: str
    prompt_override: Optional[str] = None
    reference_image_urls_override: Optional[list] = None
    # Per-segment GPT Image 2 rescue — see RunSegment.generated_reference_urls.
    # Read ahead of reference_image_urls_override by submit_run; a separate
    # field so the UI can tell which priority tier produced a segment's refs.
    generated_reference_urls: Optional[list] = None
    generated_image_job_id: Optional[str] = None
    kie_upload_url: Optional[str] = None
    seedance_task_id: Optional[str] = None
    seedance_result_url: Optional[str] = None
    local_clip_path: Optional[str] = None
    local_result_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Per-segment GPT Image 2 regeneration schemas
# ---------------------------------------------------------------------------


class SegmentImageJobCreateResponse(BaseModel):
    job_id: str
    status: str


class SegmentImageJobResponse(BaseModel):
    id: str
    run_id: str
    run_segment_id: str
    prompt: str
    input_paths: Optional[list] = None
    frame_sec: float
    kie_task_id: Optional[str] = None
    status: str
    output_path: Optional[str] = None
    cost_credits: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Run-local segment range regeneration schemas
# ---------------------------------------------------------------------------


class RangeAttemptCreateResponse(BaseModel):
    attempt_id: str
    status: str
    requested_start_sec: float
    requested_end_sec: float


class RangeAttemptResponse(BaseModel):
    id: str
    run_id: str
    run_segment_id: str
    requested_start_sec: float
    requested_end_sec: float
    prompt_override: Optional[str] = None
    reference_image_urls_override: Optional[list] = None
    status: str
    error_message: Optional[str] = None
    result_duration_sec: Optional[float] = None
    promoted_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Run automation schemas
# ---------------------------------------------------------------------------


class AutomationSettingsUpdate(BaseModel):
    """PUT /runs/{rid}/automation body — upserts the automation config."""

    enabled: bool = False
    filter_persona: Optional[str] = None
    filter_age: Optional[str] = None
    filter_gender: Optional[str] = None
    filter_ethnicity: Optional[str] = None
    exclude_persona: bool = False
    exclude_gender: bool = False
    exclude_ethnicity: bool = False
    plan: str = "close-up"
    daily_count: int = 5
    webhook_url: Optional[str] = None
    # Re-run the template's GPT Image 2 recipe on each sheet photo instead of
    # sending that photo to Seedance raw. Only accepted on a template run that
    # used preprocessing (409 otherwise).
    preprocess_refs: bool = False


class AutomationResponse(BaseModel):
    """Automation settings + live counters for the run-detail panel."""

    run_id: str
    configured: bool = False  # False → no automation row exists yet (defaults shown)
    enabled: bool = False
    filter_persona: Optional[str] = None
    filter_age: Optional[str] = None
    filter_gender: Optional[str] = None
    filter_ethnicity: Optional[str] = None
    exclude_persona: bool = False
    exclude_gender: bool = False
    exclude_ethnicity: bool = False
    plan: str = "close-up"
    daily_count: int = 5
    webhook_url: Optional[str] = None
    preprocess_refs: bool = False
    # True when the template run used GPT Image 2 — the UI only offers
    # preprocess_refs then, and the endpoint only accepts it then.
    preprocess_available: bool = False
    # Counters (UTC day)
    launched_today: int = 0
    in_flight: int = 0
    used_total: int = 0


class DriveRelinkRequest(BaseModel):
    """POST /api/v2/runs/{rid}/drive-relink (AICP-348).

    ``expected_old_file_id`` is the Drive id the operator saw reported
    missing; the re-link refuses if the run points anywhere else by then.
    ``confirm_link_change`` must be true: the run's Creative Link changes.
    """

    expected_old_file_id: str
    confirm_link_change: bool = False
