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
    false,
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

    # Human-readable label (operator-editable; nullable — falls back to source_ref)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

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

    # Longest swap segment analyze_project may propose, in seconds. NULL means
    # ai_models.UNIVERSAL_MAX_SEGMENT_SEC (10.0s), the smallest per-clip ceiling
    # across every model — a project cut that way can be run on ANY model
    # without a segment being too long, so every pre-existing project keeps
    # exactly the behaviour it had before this column existed.
    #
    # Consumed at ANALYZE time: changing it only affects the next analysis.
    # Segments already cut are not re-partitioned, so a project must be
    # re-analyzed for a new cap to take effect.
    #
    # Raising it above a model's ai_models.max_clip_sec is allowed, but it costs
    # swap coverage on runs that use a shorter-limit model: every over-long
    # segment is marked failed with RunSegment.source_fallback
    # (pipeline_v2._submit_swap_segment_isolated), and the stitch delivers the
    # ORIGINAL, un-swapped footage in its place. The run still completes and
    # delivers; Run.source_fallback_segments records how much of it is
    # un-swapped, and the run UI says so. Timing and audio are unaffected — the
    # substituted clip has exactly the segment's duration.
    # Per-clip ceilings today: Seedance 2.5 30s, base Seedance 2.0 (and its
    # fast/mini variants) 15s, Gemini Omni 10s. Set this to match the model you
    # intend to run: longer segments mean fewer stitch seams, at the cost of
    # model portability.
    max_segment_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

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
    # Which generation backend this run uses. The valid set mirrors
    # app/ai_models.AI_MODELS — add a spec there and an ALTER TYPE migration
    # here, nothing else.
    model: Mapped[str] = mapped_column(
        Enum(
            "seedance", "gemini-omni", "seedance-fast", "seedance-mini",
            "seedance-2-5",
            name="run_model_enum",
        ),
        nullable=False,
        default="seedance",
    )
    resolution: Mapped[str] = mapped_column(
        Enum("480p", "720p", "1080p", "4k", name="run_resolution_enum"),
        nullable=False,
        default="480p",
    )
    audio_mode: Mapped[str] = mapped_column(
        Enum("original", "seedance", name="run_audio_mode_enum"),
        nullable=False,
        default="original",
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
    # How many swap segments in the delivered video are the ORIGINAL footage
    # rather than a generated swap, because the run's model could not generate
    # them (see RunSegment.source_fallback). Set once at stitch time.
    #
    # Denormalised deliberately: a `done` run whose video is only partly swapped
    # must be visible without querying its segments, so the runs list can flag
    # it. NULL means "not recorded" — either the run predates this column or it
    # never reached the stitch; 0 means "checked, fully swapped". Do not infer
    # failure from a non-zero value: the run really did deliver, it just
    # delivered less swapping than asked for.
    source_fallback_segments: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
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

    # Per-segment overrides (set by operator for individual re-runs)
    prompt_override: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_image_urls_override: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )

    # Seedance / kie.ai fields
    kie_upload_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seedance_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    seedance_result_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Local paths
    local_clip_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Set when this swap segment could not be generated by the run's model for a
    # DETERMINISTIC reason, so the stitch substitutes the original source
    # footage for it instead of blocking the whole run. Today the only such
    # reasons are "the segment is longer than the model's max_clip_sec" (the
    # project's max_segment_sec was raised above what this run's model can
    # generate) and "the source is shorter than the model's min_clip_sec" — see
    # pipeline_v2._submit_swap_segment_isolated.
    #
    # The distinction that matters: a TRANSIENT failure (task error, timeout,
    # missing result url) does NOT set this. Those still leave the run
    # `incomplete`, because retrying them produces the real swap and delivering
    # the original footage instead would quietly throw that away. An over-limit
    # segment can never succeed on this model however many times it is retried,
    # so blocking forever is the worse answer.
    #
    # The segment's status stays `failed` (the swap genuinely did not happen);
    # this flag is what lets the completeness gate in process_run pass anyway,
    # and what the UI uses to label the segment as un-swapped.
    source_fallback: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )

    # Relationship back to Run
    run: Mapped["Run"] = relationship("Run", back_populates="run_segments")
    segment_def: Mapped["SegmentDef"] = relationship("SegmentDef")

    def __repr__(self) -> str:
        return (
            f"<RunSegment id={self.id!r} run_id={self.run_id!r} "
            f"index={self.index} status={self.status!r}>"
        )
