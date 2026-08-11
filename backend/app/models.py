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
from .project_types import DEFAULT_PROJECT_TYPE
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

    # What the operator is doing to this video. Decides how analysis segments it
    # and which model/prompt the New Run form is pre-filled with; see
    # app/project_types.py for the registry.
    project_type: Mapped[str] = mapped_column(
        Enum(
            "face_swap", "subtitle_removal", "cover_change", "localisation",
            name="project_type_enum",
        ),
        nullable=False,
        default=DEFAULT_PROJECT_TYPE,
        server_default=DEFAULT_PROJECT_TYPE,
    )

    # Source
    source_type: Mapped[str] = mapped_column(
        Enum("upload", "gdrive", name="project_source_type_enum"), nullable=False
    )
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    source_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # "Remove original audio": when true, the swap clips uploaded to the AI
    # model are cut video-only (ffmpeg -an) so the model can't hear — and copy
    # or be confused by — the source soundtrack, and generates audio purely
    # from the prompt instead. Prompt-only instructions ("do not use the
    # original audio") are unreliable; muting the input is not.
    #
    # The source file itself is NOT modified and "keep" segments still carry
    # their original audio, so both Run.audio_mode options stay meaningful:
    # "original" overlays the untouched source track, "seedance" keeps the
    # model-generated audio. Read at submit time, so toggling it affects every
    # subsequent run of this project.
    mute_source: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )

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

    # How much of the front of the video is the "hook" (seconds) — the only
    # part a `localisation` project re-generates. Meaningless for every other
    # project_type, which is why it is nullable with no server_default: NULL
    # means "use settings.LOCALISATION_DEFAULT_HOOK_SEC", so the default lives
    # in one place in Python instead of being frozen into the schema, and no
    # pre-existing project needs a backfill.
    #
    # Like max_segment_sec (and unlike mute_source, which is read at submit
    # time) this is consumed at ANALYZE time: hook_split segmentation lays down
    # one swap segment over [0, hook_sec] and one keep segment over the rest,
    # and those SegmentDefs are then fixed. Changing hook_sec therefore does
    # NOT re-cut an already-analyzed project — the project must be re-analyzed
    # for a new hook length to take effect. (The New Run form's translate call
    # accepts a hook_sec override for slicing the transcript only; that path
    # writes nothing here and re-segments nothing.)
    #
    # The value is clamped at analyze time to what the models can actually
    # generate — no shorter than the smallest min_clip_sec in the registry, no
    # longer than the effective segmentation cap or the video itself — so an
    # out-of-range number degrades to the nearest usable hook rather than
    # producing a segment no model will accept.
    hook_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Cached transcript of the source video (the schema in docs/localisation.md
    # §4.1: detected source_language plus timestamped, speaker-labelled,
    # verbatim lines). It lives on the PROJECT, not the Run, because it
    # describes the uploaded footage — one expensive video-model call is shared
    # by every run and every target language of that footage.
    #
    # Operator-editable (PATCH /transcript): the transcript is an input to the
    # translation prompt, not a pipeline artefact, so a wrong speaker label or
    # a misheard word is fixed by hand instead of by re-running the model.
    # JSON rather than a table: nothing ever queries into the lines, they are
    # only ever read back whole and handed to the prompt builder.
    transcript: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Lifecycle of the transcription task: "pending" | "running" | "ready" |
    # "failed" | "empty". NULL means it was never requested.
    #
    # "empty" is a SUCCESS, not an error — the model found no speech, which is
    # a legal outcome for a wordless hook; the UI says so and keeps the
    # Translate button disabled. "failed" is also non-fatal: the project stays
    # fully usable and the operator pastes a translation by hand, exactly as
    # before this feature existed. Transcription runs as its own RQ task AFTER
    # analyze_project so a model outage can never fail analysis or block
    # segmentation.
    #
    # Deliberately a plain VARCHAR and not an Enum: this is a side-channel
    # task's own state, it is not part of the ProjectStatus state machine and
    # nothing gates on it, so adding a state should be a code change rather
    # than an irreversible `ALTER TYPE ... ADD VALUE`.
    transcript_status: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )

    # Human-readable reason the last transcription attempt failed, shown in the
    # project page's transcript panel. Text (not String(n)) because it carries
    # raw provider error bodies, which are unbounded. Only meaningful while
    # transcript_status == "failed"; cleared on the next successful attempt.
    transcript_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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

    # Spoken language of the delivered creative — one of app/languages.py's
    # codes ("en"/"es"/"pt"/"ja"/"de"), NULL when never chosen (behaves as
    # English). On a `localisation` project it is the TRANSLATION TARGET: the
    # language the hook is re-spoken in, and what the translate call sends to the
    # LLM. On every other project type it is descriptive metadata only — nothing
    # in the pipeline branches on it.
    #
    # Deliberately a plain VARCHAR and not an Enum: adding a language should be a
    # one-line registry change, not an irreversible Postgres type migration.
    language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

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
