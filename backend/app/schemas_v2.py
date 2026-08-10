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
    source_type: str
    source_ref: str
    source_local_path: Optional[str] = None

    # Probe fields
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    aspect_ratio: Optional[str] = None

    # Analyze-time segmentation cap. NULL = the universal default (see
    # ai_models.UNIVERSAL_MAX_SEGMENT_SEC).
    max_segment_sec: Optional[float] = None

    # What the operator is doing to this video (app/project_types.py).
    project_type: str = "face_swap"
    # "Remove original audio" — swap clips are uploaded video-only.
    mute_source: bool = False
    # Localisation only: the intended hook length. NULL = the configured default.
    hook_sec: Optional[float] = None
    # Transcription lifecycle; the transcript itself is fetched separately (it is
    # large and only the localisation UI needs it).
    transcript_status: Optional[str] = None

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
    """Editable project settings.

    ``max_segment_sec`` distinguishes "not sent" from "sent as null" via
    ``model_fields_set`` — null is a real value here, meaning "clear the cap back
    to the universal default", so it cannot be treated as absence.
    """

    name: Optional[str] = None
    max_segment_sec: Optional[float] = None
    # Read at submit time, so a change affects every subsequent run.
    mute_source: Optional[bool] = None
    # Localisation only, and consumed at ANALYZE time like max_segment_sec: a
    # change does not re-cut segments that already exist.
    hook_sec: Optional[float] = None


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


# ---------------------------------------------------------------------------
# Run schemas
# ---------------------------------------------------------------------------


class RunCreateResponse(BaseModel):
    run_id: str
    status: str


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
    # Spoken language of the delivered creative; also the translation target on a
    # localisation project. NULL behaves as English.
    language: Optional[str] = None
    gdrive_folder_id: Optional[str] = None
    status: str
    result_local_path: Optional[str] = None
    result_gdrive_file_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


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
    kie_upload_url: Optional[str] = None
    seedance_task_id: Optional[str] = None
    seedance_result_url: Optional[str] = None
    local_clip_path: Optional[str] = None
    local_result_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Localisation schemas (docs/localisation.md)
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
