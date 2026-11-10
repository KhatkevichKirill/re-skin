"""
Pydantic V2 schemas for the overlay REST API (standalone PNG-burn feature).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Overlay placement spec (one per placed PNG)
# ---------------------------------------------------------------------------


class OverlayItem(BaseModel):
    """One PNG placement. Coords are normalized 0..1 of the frame (top-left)."""

    asset_path: str
    nx: float = Field(..., ge=0.0, le=1.0)
    ny: float = Field(..., ge=0.0, le=1.0)
    start_sec: float = Field(..., ge=0.0)
    end_sec: float
    sw: Optional[float] = Field(None, gt=0.0, le=1.0)
    fade_sec: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_window(self) -> "OverlayItem":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        if self.fade_sec and 2 * self.fade_sec > (self.end_sec - self.start_sec):
            raise ValueError(
                "2 * fade_sec must not exceed the visible window (end_sec - start_sec)"
            )
        return self


class OverlaysUpdateRequest(BaseModel):
    """Replace the full ordered list of overlay placements (z-order = list order)."""

    overlays: List[OverlayItem] = []


# ---------------------------------------------------------------------------
# Job create / response
# ---------------------------------------------------------------------------


# Every accepted source_type, and the id field each one requires. Keeping the
# pair in one table is what stops the validator, the create endpoint and the
# error message from drifting as sources are added.
OVERLAY_SOURCE_ID_FIELDS: dict[str, Optional[str]] = {
    "run": "source_run_id",
    "upload": None,                       # the video is uploaded afterwards
    "tail_edit": "source_tail_edit_id",
    "caption": "source_caption_id",
}


class OverlayCreateRequest(BaseModel):
    source_type: str = "run"           # see OVERLAY_SOURCE_ID_FIELDS
    source_run_id: Optional[str] = None
    source_tail_edit_id: Optional[str] = None
    source_caption_id: Optional[str] = None
    deliver_to_drive: bool = True      # overlay results auto-upload to Drive

    @model_validator(mode="after")
    def _check_source(self) -> "OverlayCreateRequest":
        if self.source_type not in OVERLAY_SOURCE_ID_FIELDS:
            allowed = ", ".join(repr(k) for k in OVERLAY_SOURCE_ID_FIELDS)
            raise ValueError(f"source_type must be one of: {allowed}")
        id_field = OVERLAY_SOURCE_ID_FIELDS[self.source_type]
        if id_field and not getattr(self, id_field):
            raise ValueError(
                f"{id_field} is required when source_type={self.source_type!r}"
            )
        return self


class OverlayCreateResponse(BaseModel):
    id: str
    status: str


class OverlayAssetResponse(BaseModel):
    asset_path: str


class OverlayResponse(BaseModel):
    id: str
    status: str
    source_type: str
    source_run_id: Optional[str] = None
    source_tail_edit_id: Optional[str] = None
    source_caption_id: Optional[str] = None
    source_local_path: Optional[str] = None
    src_width: Optional[int] = None
    src_height: Optional[int] = None
    src_duration_sec: Optional[float] = None
    overlays: list = []
    deliver_to_drive: bool = False
    result_local_path: Optional[str] = None
    gdrive_file_id: Optional[str] = None
    error: Optional[str] = None
    final_creatives_exported_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class OverlayDriveAssetRequest(BaseModel):
    """Add a PNG asset by Google Drive link/id instead of a file upload."""

    link: str


class OverlaySourceItem(BaseModel):
    """A selectable source video = one completed Run."""

    run_id: str
    run_name: Optional[str] = None
    project_name: Optional[str] = None
    created_at: datetime
    width: Optional[int] = None
    height: Optional[int] = None
    duration_sec: Optional[float] = None
    preview_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Overlay asset library — reusable PNG cards (see models.OverlayAsset)
# ---------------------------------------------------------------------------


class OverlayLibraryAssetResponse(BaseModel):
    id: str
    name: str
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}
