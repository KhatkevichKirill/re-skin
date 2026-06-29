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

    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectListItem(BaseModel):
    id: str
    name: Optional[str] = None
    source_ref: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectUpdate(BaseModel):
    """Editable project settings."""

    name: Optional[str] = None


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
