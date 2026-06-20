"""
Pydantic V2 schemas for the re-skin REST API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Job schemas
# ---------------------------------------------------------------------------


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    id: str
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

    # Configuration
    default_prompt: str
    default_reference_image_urls: Optional[list] = None
    resolution: str
    gdrive_folder_id: Optional[str] = None

    # Results
    result_local_path: Optional[str] = None
    result_gdrive_file_id: Optional[str] = None
    error_message: Optional[str] = None

    created_at: datetime

    model_config = {"from_attributes": True}


class JobListItem(BaseModel):
    id: str
    status: str
    source_ref: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Segment schemas
# ---------------------------------------------------------------------------


class SegmentResponse(BaseModel):
    id: str
    job_id: str
    index: int
    start_sec: float
    end_sec: float
    has_face: bool
    action: str
    prompt_override: Optional[str] = None
    reference_image_urls_override: Optional[list] = None
    pre_roll_sec: float
    post_roll_sec: float
    kie_upload_url: Optional[str] = None
    seedance_task_id: Optional[str] = None
    seedance_result_url: Optional[str] = None
    local_clip_path: Optional[str] = None
    local_result_path: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SegmentUpdate(BaseModel):
    """Fields that may be updated on an existing segment."""

    id: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    action: Optional[str] = None
    prompt_override: Optional[str] = None
    reference_image_urls_override: Optional[list] = None
    pre_roll_sec: Optional[float] = None
    post_roll_sec: Optional[float] = None


class NewSegment(BaseModel):
    """Payload for creating a brand-new segment during review."""

    start_sec: float
    end_sec: float
    action: str = "keep"
    has_face: bool = False
    prompt_override: Optional[str] = None
    reference_image_urls_override: Optional[list] = None
    pre_roll_sec: float = 0.0
    post_roll_sec: float = 0.0


class SegmentsUpdateRequest(BaseModel):
    updates: list[SegmentUpdate] = []
    deletes: list[str] = []
    creates: list[NewSegment] = []
