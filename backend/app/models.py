"""
SQLAlchemy 2.0 ORM models for re-skin.

Tables:
  - jobs      — one per video processing request
  - segments  — time-range slices of a job's video
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base
from .state_machine import JobStatus, ProjectStatus, RunStatus, SegmentStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Job(Base):
    __tablename__ = "jobs"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Source
    source_type: Mapped[str] = mapped_column(
        Enum("upload", "gdrive", name="source_type_enum"), nullable=False
    )
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Probe metadata (populated after ffprobe)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    aspect_ratio: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Job configuration
    default_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    default_reference_image_urls: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True, default=list
    )
    resolution: Mapped[str] = mapped_column(
        Enum("480p", "720p", "1080p", name="resolution_enum"),
        nullable=False,
        default="480p",
    )

    # Delivery
    gdrive_folder_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # State
    status: Mapped[str] = mapped_column(
        Enum(JobStatus, name="job_status_enum"),
        nullable=False,
        default=JobStatus.created,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Results
    result_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_gdrive_file_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    # Relationships
    segments: Mapped[list[Segment]] = relationship(
        "Segment",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="Segment.index",
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id!r} status={self.status!r}>"


class Segment(Base):
    __tablename__ = "segments"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign key
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Ordering
    index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Time range
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Classification
    has_face: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    action: Mapped[str] = mapped_column(
        Enum("swap", "keep", name="segment_action_enum"), nullable=False, default="keep"
    )

    # Per-segment overrides
    prompt_override: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_image_urls_override: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )

    # Manual UI timing adjustments
    pre_roll_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    post_roll_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Seedance / kie.ai fields
    kie_upload_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seedance_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    seedance_result_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Local paths
    local_clip_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # State
    status: Mapped[str] = mapped_column(
        Enum(SegmentStatus, name="segment_status_enum"),
        nullable=False,
        default=SegmentStatus.pending,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship back to Job
    job: Mapped[Job] = relationship("Job", back_populates="segments")

    def __repr__(self) -> str:
        return (
            f"<Segment id={self.id!r} job_id={self.job_id!r} "
            f"index={self.index} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# v2 models — additive; v1 Job/Segment left untouched
# ---------------------------------------------------------------------------


class VideoProject(Base):
    """A reusable video + its segmentation. Many Runs can reference one project."""

    __tablename__ = "video_projects"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Source
    source_type: Mapped[str] = mapped_column(
        Enum("upload", "gdrive", name="project_source_type_enum"), nullable=False
    )
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Probe metadata (populated after ffprobe)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    aspect_ratio: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # State
    status: Mapped[str] = mapped_column(
        Enum(ProjectStatus, name="project_status_enum"),
        nullable=False,
        default=ProjectStatus.created,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    segments: Mapped[list["SegmentDef"]] = relationship(
        "SegmentDef",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="SegmentDef.index",
    )
    runs: Mapped[list["Run"]] = relationship(
        "Run",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<VideoProject id={self.id!r} status={self.status!r}>"


class SegmentDef(Base):
    """Reusable segment definition — timing + swap/keep only (no per-segment char overrides)."""

    __tablename__ = "segment_defs"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign key
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("video_projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Ordering
    index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Time range
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Classification
    has_face: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    action: Mapped[str] = mapped_column(
        Enum("swap", "keep", name="segment_def_action_enum"),
        nullable=False,
        default="keep",
    )

    # Timing adjustments
    pre_roll_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    post_roll_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Relationship back to project
    project: Mapped["VideoProject"] = relationship(
        "VideoProject", back_populates="segments"
    )

    def __repr__(self) -> str:
        return (
            f"<SegmentDef id={self.id!r} project_id={self.project_id!r} "
            f"index={self.index} action={self.action!r}>"
        )


class Run(Base):
    """One character attempt on a VideoProject — owns its own RunSegments and result."""

    __tablename__ = "runs"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign key
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("video_projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Character / run metadata
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reference_image_urls: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list
    )

    # Processing options
    resolution: Mapped[str] = mapped_column(
        Enum("480p", "720p", "1080p", name="run_resolution_enum"),
        nullable=False,
        default="480p",
    )
    gdrive_folder_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # State
    status: Mapped[str] = mapped_column(
        Enum(RunStatus, name="run_status_enum"),
        nullable=False,
        default=RunStatus.created,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Results
    result_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_gdrive_file_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    # Relationships
    project: Mapped["VideoProject"] = relationship(
        "VideoProject", back_populates="runs"
    )
    run_segments: Mapped[list["RunSegment"]] = relationship(
        "RunSegment",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunSegment.index",
    )

    def __repr__(self) -> str:
        return f"<Run id={self.id!r} project_id={self.project_id!r} status={self.status!r}>"


class RunSegment(Base):
    """Per-run processing state for one SegmentDef (only swap segments get one)."""

    __tablename__ = "run_segments"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign keys
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_def_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("segment_defs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Ordering — mirrors the SegmentDef index for stable ORDER BY
    index: Mapped[int] = mapped_column(Integer, nullable=False)

    # State — reuses SegmentStatus values; distinct enum name avoids DB collision
    status: Mapped[str] = mapped_column(
        Enum(SegmentStatus, name="run_segment_status_enum"),
        nullable=False,
        default=SegmentStatus.pending,
    )

    # Seedance / kie.ai fields
    kie_upload_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seedance_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    seedance_result_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Local paths
    local_clip_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship back to Run
    run: Mapped["Run"] = relationship("Run", back_populates="run_segments")

    def __repr__(self) -> str:
        return (
            f"<RunSegment id={self.id!r} run_id={self.run_id!r} "
            f"index={self.index} status={self.status!r}>"
        )
