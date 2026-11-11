"""
SQLAlchemy 2.0 ORM models for re-skin (v2 Project/Run pipeline).

Tables:
  - video_projects — one per uploaded source video
  - segment_defs   — reusable time-range slices of a project's video
  - runs           — one character attempt on a project
  - run_segments   — per-run swap task for a segment_def
  - overlay_jobs   — standalone PNG-burn jobs
  - library_clips  — global library of reusable stitch clips (e.g. outros)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    CheckConstraint,
    false as sa_false,
    func,
    text,
)
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from .db import Base
from .project_types import DEFAULT_PROJECT_TYPE
from .state_machine import (
    AudioPatchStatus,
    CaptionStatus,
    MergeStatus,
    OverlayStatus,
    ProjectStatus,
    RangeAttemptStatus,
    RunStatus,
    SegmentImageStatus,
    SegmentStatus,
    TailEditStatus,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class VideoProject(Base):
    """A reusable video + its segmentation. Many Runs can reference one project."""

    __tablename__ = "video_projects"
    __table_args__ = (
        # Mirrors migration o5e1a7c3d649 (AICP-340).
        CheckConstraint(
            "localisation_scope IN ('hook', 'whole')",
            name="ck_video_projects_localisation_scope",
        ),
        CheckConstraint(
            "localisation_scope_requested IS NULL "
            "OR localisation_scope_requested IN ('hook', 'whole')",
            name="ck_video_projects_localisation_scope_requested",
        ),
        # Mirrors migration r1a7c3e5b9d2 (AICP-363).
        CheckConstraint(
            "mode IN ('manual', 'assistant')", name="ck_video_projects_mode",
        ),
        # Mirrors migration u5e1b7c9d3f4 (batch VOID fills / VOID from the source).
        CheckConstraint(
            "caption_removal_method IS NULL "
            "OR caption_removal_method IN ('void_source', 'omni')",
            name="ck_video_projects_caption_removal_method",
        ),
        # Mirrors migration w4c8e2a6d1f3 (AICP-331).
        CheckConstraint(
            "(source_kind IS NULL AND source_artifact_id IS NULL) OR "
            "(source_kind IN ('caption', 'tail_edit', 'overlay') "
            "AND source_artifact_id IS NOT NULL)",
            name="ck_video_projects_source_kind",
        ),
    )

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Human-readable label (operator-editable; nullable — falls back to source_ref)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # What the operator is doing to this video. Decides whether analysis runs
    # face detection and which model/prompt the New Run form is pre-filled
    # with; see app/project_types.py for the registry.
    project_type: Mapped[str] = mapped_column(
        Enum(
            "face_swap", "subtitle_removal", "cover_change", "localisation",
            "reframe", "rebuild",
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
        Boolean, nullable=False, default=False, server_default=sa_false()
    )

    # Longest a single swap segment may be (seconds) when analyze_project
    # partitions the video. NULL means "the historical default":
    # ai_models.UNIVERSAL_MAX_SEGMENT_SEC (10.0s), the smallest per-clip ceiling
    # across every model — a project cut that way can be run on ANY model
    # without a segment being too long, so every pre-existing project keeps
    # exactly the behaviour it had before this column existed.
    #
    # Unlike mute_source (read at submit time), this is consumed at ANALYZE
    # time: changing it only affects the next analysis. Segments already cut are
    # not re-partitioned, so a project must be re-analyzed for a new cap to
    # take effect.
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

    # How much of a `localisation` project is re-generated: "hook" (the
    # historical swap[0, hook_sec] + keep tail) or "whole" (all-swap segments
    # at scene cuts, AICP-340). Values: project_types.LOCALISATION_SCOPES;
    # NOT NULL with server_default "hook" so every pre-existing row keeps its
    # behaviour (migration o5e1a7c3d649). Analyze-time like hook_sec, but
    # stricter: it is written only by project creation and by analyze_project
    # (in the same commit as the new SegmentDefs, from
    # localisation_scope_requested below), never by PATCH or the analyze
    # endpoint, so it always names the strategy that produced the current
    # SegmentDefs — the prompt window and the run form read it as a fact about
    # the segments, not a pending wish.
    localisation_scope: Mapped[str] = mapped_column(
        String(16), nullable=False, default="hook", server_default="hook"
    )
    # The scope the NEXT analysis must apply (AICP-340), written by
    # POST /projects/{pid}/analyze?localisation_scope= and cleared by the
    # analysis that applies it (same commit as localisation_scope and the new
    # SegmentDefs). NULL = re-apply localisation_scope. Kept on the row rather
    # than in the RQ job so the analyze job's arguments never change (an old
    # worker can run a new job and vice versa) and so a crashed or failed
    # analysis retried later still applies what the operator asked for.
    localisation_scope_requested: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )

    # "manual" (today's behaviour) or "assistant" (a server-side autopilot
    # runs the project's flow, AICP-363, docs/autopilot-design.md §2.1). Set
    # only at creation (POST /projects, POST /runs/{rid}/to-project), never by
    # PATCH. Nothing in the run pipeline reads it.
    mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="manual", server_default="manual"
    )

    # What this project's burned captions look like, for the free QA caption
    # detector (app.qa_tools.CaptionDetectorConfig, AICP-343): band, text and
    # outline colour/style, glyph size, hit threshold. Only the non-default
    # fields are stored; NULL means the defaults (white text with a black
    # outline in the centre band -- the detector exactly as AICP-332 used it).
    # Read by the QA tools only; nothing in the pipeline consumes it.
    # none_as_null: a reset writes SQL NULL, not a JSON 'null' value, so
    # "IS NULL" keeps meaning "defaults".
    caption_detector: Mapped[Optional[dict]] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )

    # Where this project's generated (burned) captions sit (AICP-368,
    # app.caption_placement.CaptionPlacement): over the ORIGINAL caption area
    # of a subtitle-removal lineage, so the patches left where the old
    # captions were stay covered. Derived at project-from-run (or inherited
    # from the parent project), or set by the operator. NULL = today's house
    # style (bottom-centred); an explicit {"mode": "bottom"} is the operator's
    # sticky "no placement" that a later derivation never overwrites. Read
    # when a caption job is CREATED (copied into run_captions.placement), not
    # at burn time.
    caption_placement: Mapped[Optional[dict]] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )

    # Subtitle removal (2026-11-03, docs/segment-fill-batch-design.md §3): VOID
    # caption fills the autopilot may START WITHOUT ASKING on one run, in USD
    # of reservations. NULL = settings.AUTOPILOT_VOID_AUTO_USD; 0 = always ask.
    void_auto_usd_per_run: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Subtitle removal (2026-11-03, docs/segment-fill-batch-design.md §7): how
    # the captions are removed. "void_source" = every swap segment is a VOID
    # fill from the SOURCE (no model submit; Omni only as a per-segment
    # fallback); "omni" = one model pass, VOID for leftovers. NULL reads as
    # "omni" (every project created before the mode existed). A run freezes it
    # at creation (runs.generation_mode).
    caption_removal_method: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # Rebuild projects only (docs/rebuild-from-reference-design.md §5.1.8):
    # the token of the analysis that currently owns the `analyzing` status.
    # Set by the re-analysis admission (and by the first analysis job when it
    # leaves `created`); an analysis job remembers the token it started under
    # and publishes / marks the project failed only while it is still the
    # owner, so a stale job (reaped by recovery, then superseded by a new
    # admission) can never touch a later analysis. NULL for every other type.
    rebuild_analysis_token: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )

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
    # a legal outcome for a wordless hook; the UI says "в хуке нет речи" and
    # keeps the Translate button disabled. "failed" is also non-fatal: the
    # project stays fully usable and the operator pastes a translation by hand,
    # exactly as before this feature existed. Transcription runs as its own RQ
    # task AFTER analyze_project so a model outage can never fail analysis or
    # block segmentation.
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

    # Soft reference to runs.id when this project was started from a finished
    # run's final video ("Move to new project"). No FK — the source run may be
    # deleted independently and this project keeps its own copy of the video;
    # the UI degrades to showing no backlink. Mirrors OverlayJob.source_run_id.
    source_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Which STEP of that run the copied video came from (AICP-331): "caption",
    # "tail_edit" or "overlay" with the derived creative's id in
    # source_artifact_id. NULL = the run's final video (the run path, and every
    # project created before this column). source_run_id above always names
    # the OWNING run (caption/tail run_id, overlay parent_run_id), so the
    # backlink, Source cell and lineage walks never need these two. Soft
    # references, no FK: captions/tail edits are hard-deleted, overlays
    # soft-deleted, and the project page then falls back to the run backlink.
    source_kind: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    source_artifact_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

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
    __table_args__ = (
        CheckConstraint(
            "generation_mode IN ('v2v', 'images_only', 'void_source')",
            name="ck_runs_generation_mode",
        ),
    )

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

    # Voice reference(s) for the generated speech — local paths, resolved to
    # public URLs at submit time by resolve_reference_urls exactly like
    # reference_image_urls, then sent as Seedance 2.5's `reference_audio_urls`.
    #
    # Without one the model invents a voice per generation, so two runs of the
    # same character (or the same run re-run after a fix) sound like different
    # people — the problem this column exists to solve. With one, the generated
    # speech takes that sample's timbre.
    #
    # Always a run-OWNED copy of the file, never a pointer into the voice
    # library: picking a library voice copies it into the run's own dir (same
    # rule as OverlayAsset, and for the same reason) so deleting a library voice
    # can never change or break an existing run's output.
    #
    # NULL/[] for the runs that predate this column and for every run on a model
    # with no `reference_audio_urls` field (the 2.0 family, Gemini Omni) — the
    # key is then omitted from the payload entirely.
    reference_audio_urls: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )

    # Processing options
    model: Mapped[str] = mapped_column(
        Enum(
            "seedance", "gemini-omni", "seedance-fast", "seedance-mini",
            "seedance-2-5", "kling-4-0",
            name="run_model_enum",
        ),
        nullable=False,
        default="seedance",
    )
    resolution: Mapped[str] = mapped_column(
        Enum("480p", "720p", "1080p", "4k", name="run_resolution_enum"),
        nullable=False,
        default="720p",  # AICP-391; matches settings.DEFAULT_RESOLUTION
    )
    audio_mode: Mapped[str] = mapped_column(
        Enum("original", "seedance", name="run_audio_mode_enum"),
        nullable=False,
        default="original",
    )
    gdrive_folder_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Spoken language of the delivered creative — one of app/languages.py's
    # codes ("en"/"es"/"pt"/"ja"/"de"/"it"/"pl"/"fr"), NULL when never chosen (behaves as
    # English). Affects NOTHING in the pipeline: its only job is to suffix the
    # id we hand to the tracking spreadsheet, so the downstream renaming
    # pipeline can read the language back out of it. Deliberately a plain
    # VARCHAR and not an Enum — adding a language should be a one-line registry
    # change, not a Postgres type migration.
    language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    # Stitch plan — controls what the finalize phase concatenates:
    #   NULL (default) — stitch every SegmentDef in order (classic behaviour).
    #   []             — process all swap segments but produce NO final video
    #                    (operator wants the per-segment results only).
    #   [entry, ...]   — custom ordered plan; each entry is either
    #                    {"kind": "segment", "segment_def_id": "..."} or
    #                    {"kind": "library", "clip_id": "..."} (a LibraryClip).
    #                    Swap SegmentDefs absent from a custom plan are never
    #                    submitted to the AI model (no credits spent on them).
    stitch_plan: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # "Remove original audio" frozen at admission (AICP-365). NULL = read the
    # project's live ``mute_source`` at submit time (every manual run, and every
    # run created before the column existed); True/False = the value this run
    # was admitted with, so an operator toggling the project setting cannot
    # change what an autopilot run, already approved, uploads to the model.
    mute_source: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Frozen copy of the SegmentDefs this run was launched against, written by
    # submit_run before the first clip is cut:
    #   [{"id","index","action","start_sec","end_sec"}, ...]  (ordered by index)
    #
    # SegmentDefs live on the Project and operators keep editing them — a split
    # or a boundary nudge while a run is in flight used to change what finalize
    # thought that run contained. A def added mid-run has no RunSegment, so the
    # completeness gate counted it as a segment that "did not complete" and
    # parked a fully generated run at `incomplete`; the stitch would then have
    # KeyError'd on it anyway. The snapshot makes a run self-describing, so
    # later project edits cannot retroactively change it.
    #
    # NULL for runs that predate this column — the finalize path falls back to
    # the live project segments for those (unchanged legacy behaviour).
    segment_snapshot: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # How this run generates video (docs/rebuild-from-reference-design.md §4):
    # "v2v" (every run before the rebuild type, and the server default that
    # backfilled them) or "images_only" (a rebuild run: each swap chunk is
    # generated from still keyframes by rebuild_chunk_attempts rows, never by
    # the v2v submit/poll code). Set at creation, never changed. Values:
    # project_types.GENERATION_MODES (CHECK ck_runs_generation_mode).
    generation_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="v2v", server_default="v2v"
    )
    # Frozen at launch for an images-only run (design §6.1): per chunk the shot
    # list, the rendered prompt, the requested duration and the RUN-OWNED
    # keyframe copies. NULL for every v2v run.
    rebuild_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Optional GPT Image 2 (image-to-image) preprocessing of the reference
    # images before they reach Seedance/Gemini Omni. See
    # docs/gpt-image-2-preprocess.md.
    #   preprocess_enabled        — toggle (off by default; existing runs unaffected)
    #   preprocess_mode           — "per_image" (default): one GPT Image 2 task per
    #                               reference (prompts[i] applies to refs[i]); or
    #                               "all_as_refs": ONE task with input_urls = ALL
    #                               refs and a single prompt → ONE generated image
    #                               becomes the sole preprocessed ref.
    #   preprocess_image_prompts  — list parallel to reference_image_urls
    #                               (per_image) or a single-element list
    #                               (all_as_refs); prompts[i] applies to
    #                               reference_image_urls[i]; an empty string means
    #                               "pass this ref through unchanged"
    #   preprocessed_reference_urls — local paths to the generated images,
    #                               parallel to reference_image_urls; populated
    #                               by the preprocessing phase
    #   preprocess_task_ids       — kie.ai taskIds, list parallel to
    #                               reference_image_urls (None for pass-through
    #                               refs); persisted for resume/no-rebill on
    #                               worker restart
    preprocess_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    preprocess_mode: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, default="per_image"
    )
    preprocess_image_prompts: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    preprocessed_reference_urls: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    preprocess_task_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Skip the `preprocess_review` pause and go straight to submit once GPT
    # Image 2 succeeds. Set on runs born from "Copy run · re-run GPT Image 2"
    # and from an automation configured the same way: those clone the frames
    # and the image prompt from a run whose recipe an operator already
    # approved, so the only thing that changed is the character photo and
    # there is no new decision to make. A failed GPT Image 2 task still fails
    # the run — auto-approve skips the review, not the error handling.
    preprocess_auto_approve: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # State
    status: Mapped[str] = mapped_column(
        Enum(RunStatus, name="run_status_enum"),
        nullable=False,
        default=RunStatus.created,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Results
    result_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Probed duration of the result video, set once at finalize (stitch/deliver
    # or delivery-retry). Immutable once populated — avoids re-running ffprobe
    # on every run-detail render. NULL for runs finalized before this column
    # existed; callers fall back to an on-demand probe in that case.
    result_duration_sec: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    # How many swap segments in the delivered video are the ORIGINAL footage
    # rather than a generated swap, because the run's model could not generate
    # them (see RunSegment.source_fallback). Set once at finalize, same pattern
    # as result_duration_sec above.
    #
    # Denormalised deliberately: a `done` run whose video is only partly swapped
    # must be visible without querying its segments, so the runs list and the
    # automation webhook can flag it. NULL means "not recorded" — either the run
    # predates this column or it never reached finalize; 0 means "checked, fully
    # swapped". Do not infer failure from a non-zero value: the run really did
    # deliver, it just delivered less swapping than asked for.
    source_fallback_segments: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    result_gdrive_file_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    # Set when the run has been logged to the "Final Creatives" sheet (manual,
    # operator-triggered). Drives the "already added" UI state; NULL = not added.
    final_creatives_exported_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    # LEGACY: the "Add to Caption Demo" sheet export was removed on 2026-08-28.
    # Nothing writes this column any more; the values that are here record what
    # the feature did while it existed. Kept so the ORM matches the live
    # schema — do NOT drop it in an autogenerated migration.
    caption_demo_status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
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

    @property
    def short_id(self) -> str:
        """The id the operator sees and the sheet receives: ``1736426f-ja``.

        The 8-char uuid prefix plus this run's language suffix (nothing extra
        for English/unset — see app/languages.py). Use this everywhere a run id
        is DISPLAYED or exported, so the UI and the Final Creatives ``Job ID``
        never disagree. Derived creatives (tail edits, overlays) build the same
        shape from their OWN prefix and this run's language via
        ``languages.job_id_for``.
        """
        from .languages import job_id_for  # local: keeps models.py import-light

        return job_id_for(self.id, self.language)

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
    # Deliberately NOT a ForeignKey to segment_defs.id. This value is the
    # STABLE IDENTITY KEY a RunSegment shares with its entry in the run's own
    # frozen Run.segment_snapshot (and, before that column existed, with a
    # live SegmentDef row) — every reader that resolves a RunSegment's
    # timing/action (web_v2._launch_window, api_v2._segment_launch_window,
    # web_v2._range_edit_bounds, pipeline_range_attempt, pipeline_v2.
    # run_segment_defs/_build_copied_run) matches on this id against the
    # snapshot FIRST and only falls back to a live SegmentDef for legacy
    # runs that predate the snapshot column.
    #
    # SegmentDefs are mutable and, via Project re-analysis or a bulk segment
    # edit, DELETABLE — an FK with ondelete="CASCADE" here made a live def's
    # deletion cascade-delete this RunSegment (and, transitively, its
    # RunSegmentRangeAttempt history) even for a Run whose snapshot no longer
    # needs the live row at all, silently destroying a terminal Run's
    # per-segment breakdown and correction history, and later breaking a
    # Copy/Automation launch (a fresh RunSegment insert reusing this id would
    # violate the same FK against a since-deleted row). A plain indexed
    # column with no DB-level referential constraint removes that dependency
    # entirely; the id is always written from an already-loaded SegmentDef.id
    # at creation time, so it is correct without the DB re-checking it.
    segment_def_id: Mapped[str] = mapped_column(
        String(36),
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

    # Credits consumed by the AI task for this segment (kie.ai `creditsConsumed`,
    # captured on task success). Null until the task completes / for skipped
    # segments. Summing across a run's segments gives the run's total cost.
    cost_credits: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Local paths
    local_clip_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Set when this swap segment could not be generated by the run's model for a
    # DETERMINISTIC reason, so the stitch substitutes the original source
    # footage for it instead of blocking the whole run. Today the only such
    # reason is "the segment is longer than the model's max_clip_sec" — the
    # project's max_segment_sec was raised above what this run's model can
    # generate (see pipeline_v2._submit_swap_segment_isolated).
    #
    # The distinction that matters: a TRANSIENT failure (task error, timeout,
    # missing result url) does NOT set this. Those still leave the run
    # `incomplete`, because retrying them produces the real swap and delivering
    # the original footage instead would quietly throw that away. An over-limit
    # segment can never succeed on this model however many times it is retried,
    # so blocking forever is the worse answer.
    #
    # The segment's status stays `failed` (the swap genuinely did not happen);
    # this flag is what lets the completeness gate in _run_stitch_and_deliver
    # pass anyway, and what the UI uses to label the segment as un-swapped.
    source_fallback: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false()
    )

    # VOID-from-the-source runs (runs.generation_mode = "void_source",
    # docs/segment-fill-batch-design.md §7): the segments the operator sent to
    # the run's video model as a FALLBACK. The submit phase of such a run
    # submits only these; every other segment is a VOID fill from the source.
    model_fallback: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false()
    )

    # Per-segment GPT Image 2 regeneration result — a rescue of THIS segment's
    # generated reference image, distinct from reference_image_urls_override
    # above (a full replacement of the run-level reference SET; see
    # RunSegmentImageJob). Exactly two entries when set: [character photo, the
    # downloaded GPT Image 2 output] — the honest analogue of one `per_image`
    # task's result, never spliced into the run-level list it was not
    # generated from. Read by submit_run ahead of both
    # reference_image_urls_override and the run-level set (pipeline_v2.py);
    # composed when a RunSegmentImageJob succeeds, or when the operator applies
    # an older attempt from the history list. NULL = not set (the normal
    # priority chain applies). Never cloned by a Run copy — see
    # api_v2._build_copied_run.
    generated_reference_urls: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    # Which RunSegmentImageJob's result is currently applied above — drives the
    # "current attempt" badge in the UI history list. NULL alongside
    # generated_reference_urls=NULL (nothing applied); cleared independently by
    # the "Clear generated image" control, which does not delete the job row.
    generated_image_job_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )

    # Relationship back to Run
    run: Mapped["Run"] = relationship("Run", back_populates="run_segments")

    # No FK backs this join (see segment_def_id above), so the condition is
    # spelled out explicitly; viewonly because this is a read-only lookup of
    # the LIVE row for legacy (pre-snapshot) runs, never something this model
    # writes through. Resolves to None once the live SegmentDef is gone —
    # every caller already treats that as "fall back to no bounds", the same
    # as a legacy run whose def was never created with a snapshot.
    segment_def: Mapped[Optional["SegmentDef"]] = relationship(
        "SegmentDef",
        primaryjoin="foreign(RunSegment.segment_def_id) == SegmentDef.id",
        viewonly=True,
    )

    def __repr__(self) -> str:
        return (
            f"<RunSegment id={self.id!r} run_id={self.run_id!r} "
            f"index={self.index} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Per-segment GPT Image 2 regeneration — a rescue of ONE swap segment's
# generated reference image, run before a Re-run of that segment. See
# RunSegment.generated_reference_urls/generated_image_job_id above and
# app/pipeline_segment_image.py. A derived INPUT job, not a derived creative:
# no Drive upload, no sheet row, no public token — its output only ever feeds
# back into a later Seedance/Gemini Omni submission.
# ---------------------------------------------------------------------------


class RunSegmentImageJob(Base):
    """One GPT Image 2 attempt at regenerating a swap segment's reference image.

    History, not a slot: each "Regenerate image" click creates a new row, so
    ``run_segment_id``'s past attempts stay a plain append-only list (prompt,
    frame, cost, status) the operator can inspect and re-apply — see
    RunSegment.generated_image_job_id.
    """

    __tablename__ = "run_segment_image_jobs"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign keys. run_id is denormalized from run_segment_id's parent Run —
    # provenance (project id, cost rollup) without a join for every listing.
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_segment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("run_segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # The GPT Image 2 prompt used for this attempt (prefilled from the run's
    # preprocess_image_prompts by the API, editable before submit).
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # The exact files fed to kie.ai: [character photo, extracted frame].
    # Written in two steps — the API records the character photo (index 0)
    # at create time; the worker appends the frame it extracted (index 1)
    # once it copies it into this job's own directory (never the shared
    # project frame cache, so a retry does not depend on it surviving).
    input_paths: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # The source timestamp the frame was cut at — for the UI label. Set by the
    # API (the requested or defaulted value) and overwritten by the worker with
    # the clamped/quantised value actually used (extract_project_frame's
    # t_norm), mirroring how a staged frame's filename records the real frame,
    # not the request.
    frame_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # kie.ai taskId, persisted BEFORE polling starts so a worker restart
    # resumes this task instead of creating (and billing) a new one — the same
    # rule preprocess_reference_images follows for run-level GPT Image 2 tasks.
    kie_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # State
    status: Mapped[str] = mapped_column(
        Enum(SegmentImageStatus, name="segment_image_status_enum"),
        nullable=False,
        default=SegmentImageStatus.queued,
    )

    # The downloaded generated image (local path).
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Credits consumed by this attempt's kie.ai task. Kept here rather than on
    # RunSegment.cost_credits, which the next video generation OVERWRITES —
    # this row is the only place an image attempt's cost survives a Re-run.
    cost_credits: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Bounded length, same convention as RunTailEdit.error_message.
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    run: Mapped["Run"] = relationship("Run", viewonly=True)
    run_segment: Mapped["RunSegment"] = relationship("RunSegment", viewonly=True)

    def __repr__(self) -> str:
        return (
            f"<RunSegmentImageJob id={self.id!r} run_segment_id={self.run_segment_id!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Run-local segment range regeneration — a candidate revision of ONE swap
# segment's time range (enlarged/shrunk through an adjacent `keep` boundary),
# held apart from the Run/RunSegment it targets until generation + stitch +
# delivery all succeed. See app/pipeline_range_attempt.py and
# docs/run-local-segment-range-regeneration.md.
#
# History, not a slot — same rule as RunSegmentImageJob: each "Regenerate with
# new range" click is a NEW row, so a rejected/failed attempt is diagnosable
# evidence, not a value that got silently clobbered. Only one row per Run may
# be non-terminal at a time — a real DB-level partial unique index
# (uq_run_segment_range_attempts_active_per_run below; migration
# 73df0ff3f344), not just the application-level check in
# api_v2._active_range_attempt (which is a non-locking SELECT and cannot, by
# itself, stop two concurrent requests from both passing it — the index is
# what actually makes a second one impossible, converting the loser's insert
# into an IntegrityError api_v2 turns into a 409). That index stops a
# SECOND, DIFFERENT attempt from ever coexisting with an active one; it does
# NOT stop the SAME attempt being processed by two worker sessions at once
# (a duplicate poll/submit enqueue, or a recovery re-enqueue racing a still-
# running job) — see version_id and the claim_owner/claim_expires_at/
# submit_billed_at columns below for what closes that race, and
# pipeline_range_attempt.py for how they gate cutting/uploading/creating a
# task/downloading/stitching/delivering/promoting.
# ---------------------------------------------------------------------------


class RunDriveRelink(Base):
    """One operator re-link of a run to a NEW Drive file (AICP-348) — the
    durable operation AND its audit trail.

    Since AICP-342 every re-delivery replaces the content behind the run's
    existing Drive id and never falls back to a fresh upload, so a run whose
    Drive file was deleted cannot be delivered again. The re-link is the
    explicit way out (app/drive_relink.py): admitted under the Run-row lock,
    this row (with a pre-generated ``target_file_id``) is committed BEFORE any
    upload; the worker then re-checks everything under the Run-row lock,
    confirms the old file is gone with a fresh 404, creates the new file with
    exactly ``target_file_id`` (so a lost response, a retry or a crash can
    never make a second file — the outcome is a ``files.get`` by that id),
    re-points ``Run.result_gdrive_file_id`` and marks this row ``done`` in one
    commit.
    The Creative Link CHANGES; ``sheet_scan`` is the point-in-time list of
    Final Creatives cells still holding the old link (never edited here).

    ``status``: ``queued`` (admitted; also what a crashed job leaves — the next
    POST resumes the same row) -> ``done`` | ``refused`` (a re-check said no;
    the message names an unlinked file an earlier attempt of a resumed row
    may have created). A failed upload leaves the row ``queued`` (never a
    new reserved id while one may still land). One ``queued`` row per run.
    """

    __tablename__ = "run_drive_relinks"
    __table_args__ = (
        Index(
            "uq_run_drive_relinks_active_per_run",
            "run_id",
            unique=True,
            postgresql_where=text("status = 'queued'"),
            sqlite_where=text("status = 'queued'"),
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    old_file_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # Pre-generated at admission (Drive generateIds): the id the new file is
    # CREATED with, so the create is idempotent and its outcome is a files.get
    # by id. new_file_id is set (to it) only once the run points at it.
    target_file_id: Mapped[str] = mapped_column(String(255), nullable=False)
    new_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # The final the operator confirmed, and the run's status at admission (a
    # `failed` run stays failed; see app/drive_relink.py).
    result_path: Mapped[str] = mapped_column(Text, nullable=False)
    run_status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Exports the DB knows of that referenced the run's link:
    # [{"kind", "id", "job_id", "column", "pipeline_type"}].
    exported_rows: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Point-in-time scan of the Final Creatives tab for the old link, taken
    # after the re-point: {"scanned_at", "rows": [{"row", "column", "job_id",
    # "value"}], "error"}.
    sheet_scan: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class RunSegmentTake(Base):
    """A preserved previous output of one swap RunSegment (AICP-342).

    Written when a segment rerun or a take restore is about to replace a
    COMPLETED segment's result: the current result file is copied to
    ``storage.segment_take_dir`` (a path nothing writes to again) and this row
    records it, so the operator can later restore the best take per segment.
    Passive history — no status, no worker, nothing for the recovery reaper or
    the reconcile tick to look at. One row per preserved OUTPUT, never per
    restore: re-archiving a result that already is a take (it was restored)
    reuses that row.

    ``prompt_override`` / ``reference_image_urls_override`` /
    ``generated_reference_urls`` are the segment's override fields AT ARCHIVE
    TIME — what produced this output unless someone saved a different override
    after it was generated without rerunning. Local reference files of the
    override are COPIED next to the take (a later PATCH may overwrite the
    segment's own refs by name), and a restore writes all of these back onto
    the RunSegment, so Copy/Automation reproduce the restored take's recipe.
    ``start_sec``/``end_sec`` are the run-local bounds the output was generated
    against; a restore is refused once a promoted range attempt has moved them.
    Files older than ``SEGMENT_TAKE_RETENTION_DAYS`` are purged by the disk
    cleanup sweep (``file_purged_at``).
    """

    __tablename__ = "run_segment_takes"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    # Denormalized from run_segment_id's parent Run (written from rs.run_id),
    # like RunSegmentImageJob.run_id.
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_segment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("run_segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Not an FK, same reasoning as RunSegment.segment_def_id.
    segment_def_id: Mapped[str] = mapped_column(String(36), nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )

    start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    result_path: Mapped[str] = mapped_column(Text, nullable=False)
    seedance_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    seedance_result_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cost_credits: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    prompt_override: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_image_urls_override: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    generated_reference_urls: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    generated_image_job_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )

    # Set by the disk-cleanup retention sweep (app/disk_cleanup.py) when the
    # take's file was deleted. The row is kept as history (task, cost, prompt);
    # a purged take can no longer be restored.
    file_purged_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class RunSegmentRangeAttempt(Base):
    """One attempt at regenerating a swap segment over a NEW run-local range.

    The Run's active ``segment_snapshot`` and the targeted ``RunSegment``'s
    fields are left untouched for the whole lifetime of this row — cutting,
    uploading, generating, stitching and delivering all happen against THIS
    row's own candidate fields and its own working directory
    (``storage.range_attempt_dir``). Only on a successful ``done`` does the
    worker copy this row's candidate fields onto the real Run/RunSegment
    (``pipeline_range_attempt.poll_range_attempt_once`` — see ``promoted_at``
    below); a `failed` attempt never touches them, so the run's last
    successful timeline/result stays exactly what it was.
    """

    __tablename__ = "run_segment_range_attempts"

    # At most one non-terminal attempt per run — a DB-level partial unique
    # index, not just the application-level check in
    # api_v2._active_range_attempt (a non-locking SELECT that two concurrent
    # create requests can both pass). Mirrored in migration 73df0ff3f344; the
    # WHERE clause there must be kept byte-for-byte in sync with this one and
    # with RANGE_ATTEMPT_ACTIVE_STATUSES (state_machine.py) — models.py
    # cannot import that module (it would cycle), so the status list is
    # spelled out by hand in all three places. sqlite_where mirrors
    # postgresql_where so tests built on Base.metadata.create_all(), not
    # Alembic, exercise the same real constraint the migration installs.
    __table_args__ = (
        Index(
            "uq_run_segment_range_attempts_active_per_run",
            "run_id",
            unique=True,
            postgresql_where=text(
                "status IN ('queued','generating','stitching','delivering')"
            ),
            sqlite_where=text(
                "status IN ('queued','generating','stitching','delivering')"
            ),
        ),
    )

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Optimistic-concurrency guard (SQLAlchemy's version_id_col — see
    # __mapper_args__ below). Every UPDATE to this row is conditioned on
    # this value matching what the current session last read; a second
    # session racing the SAME attempt (a duplicate poll enqueue, or a
    # recovery re-enqueue racing a still-running job) has ITS commit
    # rejected (StaleDataError) the moment it tries to write from a copy it
    # read before the first session's commit, instead of silently
    # overwriting a just-committed status (e.g. `done`) with a stale
    # in-progress one. The DB-level one-active-attempt index above stops a
    # SECOND, DIFFERENT attempt from ever coexisting; this stops two
    # sessions processing the SAME attempt from corrupting each other —
    # see pipeline_range_attempt.poll_range_attempt_once's docstring for
    # the failure this closes (found by the independent audit's mandatory
    # review gate on the first hardening pass, which the DB-level index and
    # the Run-row lock in _promote did not, by themselves, fully close).
    version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __mapper_args__ = {"version_id_col": version_id}

    # Per-phase ownership claim — closes the race version_id only detects
    # AFTER the fact: two workers delivering the SAME attempt can both cut,
    # upload, call kie.ai's createTask, download a result, or re-cut/stitch/
    # deliver/promote concurrently, before either commit lands. A worker must
    # commit claim_owner (an opaque per-call token) via an optimistic-
    # versioned CAS on THIS row — no new locking primitive, version_id above
    # already gives an atomic compare-and-set — before doing any of that; a
    # second worker finding a live (non-expired) claim exits benignly without
    # touching a file or external client. NULL means "unclaimed" — the safe,
    # deterministic state every pre-migration row starts in, and the state a
    # terminal (`done`/`failed`) row is simply never checked in again from.
    # See pipeline_range_attempt._try_claim_phase / storage.range_attempt_dir
    # (claim-scoped work directories keyed by this token) and
    # docs/run-local-segment-range-regeneration.md.
    claim_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Lease deadline for claim_owner. A live claim (owner set, not yet
    # expired) blocks another worker outright; only past this deadline may a
    # later call safely consider the previous owner dead and take the phase
    # over — and for the submit phase, ONLY when submit_billed_at (below)
    # is also unset, since taking over past that point risks a second billed
    # generation.
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    # Set and committed immediately before the ONE paid external call this
    # feature makes — kie.ai's createTask, inside submit_range_attempt — and
    # never cleared afterwards (even once seedance_task_id is persisted; a
    # done/failed row is never re-examined for this). kie.ai's jobs API has
    # no idempotency key or lookup-by-reference (kie_client.KieClient has no
    # such parameter or endpoint), so a worker that crashes between this
    # commit and persisting seedance_task_id leaves the row in a state that
    # cannot be safely auto-resolved either way: retrying risks a second
    # billed generation for the same segment, giving up risks silently
    # discarding one that already succeeded. A later call finding this set
    # with seedance_task_id still NULL, past the expired claim above, marks
    # the attempt `failed` with an explicit "submission ambiguous, needs
    # manual reconciliation" message rather than guessing — see
    # submit_range_attempt and the residual-risk note in
    # docs/run-local-segment-range-regeneration.md.
    submit_billed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Foreign keys. run_id is denormalized from run_segment_id's parent Run —
    # same reasoning as RunSegmentImageJob.run_id (provenance + listing
    # without a join).
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_segment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("run_segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # The candidate swap segment's new run-local bounds — fully resolved
    # (never deltas), validated against the active timeline's neighbours at
    # create time (see pipeline_range_attempt.validate_range_request).
    requested_start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    requested_end_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Prompt/reference overrides submitted atomically with the range request —
    # same shape and priority as RunSegment.prompt_override /
    # reference_image_urls_override. NULL means "leave the RunSegment's
    # existing override untouched at promotion", mirroring rerun_segment's
    # "only a form field that was actually sent is applied" rule.
    prompt_override: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_image_urls_override: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )

    # The Run.segment_snapshot this attempt was computed against, captured at
    # create time. Purely a diagnostic/consistency check — promotion is
    # refused (attempt fails) if the run's active snapshot no longer matches
    # this, which should be impossible given the one-active-attempt-per-run
    # API gate, but is cheap insurance against a future caller that bypasses it.
    base_segment_snapshot: Mapped[list] = mapped_column(JSON, nullable=False)

    # The full resulting run-local timeline if this attempt is promoted — a
    # copy of base_segment_snapshot with the targeted swap entry's bounds
    # updated and its neighbour(s)' shared boundary updated to match. Same
    # shape as Run.segment_snapshot (see build_segment_snapshot).
    candidate_segment_snapshot: Mapped[list] = mapped_column(JSON, nullable=False)

    # State
    status: Mapped[str] = mapped_column(
        Enum(RangeAttemptStatus, name="range_attempt_status_enum"),
        nullable=False,
        default=RangeAttemptStatus.queued,
    )

    # Seedance / kie.ai fields for the candidate swap generation — same
    # purpose as the matching RunSegment columns (resume/idempotency: the
    # task id is persisted BEFORE polling starts).
    seedance_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    seedance_result_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cost_credits: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Local paths — all under storage.range_attempt_dir(id, ...), never the
    # run's own index-based clip_%04d.mp4 / result_%04d.mp4 paths, so a
    # candidate byte can never collide with (or overwrite) the active clip cut
    # for the OLD bounds while this attempt is still in flight.
    local_clip_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Adjacent `keep` segment(s) re-cut at the NEW shared boundary —
    # {segment_def_id: local_path}. 0, 1 or 2 entries depending on which side(s)
    # of the swap segment moved. Re-cut from VideoProject.source_local_path
    # exactly like the stitch phase's own keep clips, never from an
    # already-transcoded keep clip cut for the OLD bounds.
    keep_recut_paths: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Candidate stitched final video. NULL when the run's stitch_plan == []
    # (no final video to produce — the candidate result clip is the whole
    # deliverable) or while the attempt has not reached stitching yet.
    final_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Rollup mirroring Run.source_fallback_segments for the CANDIDATE final —
    # always equal to the run's current value (the candidate swap segment
    # itself never source-falls-back: its duration is validated up front, see
    # pipeline_range_attempt.validate_range_request).
    source_fallback_segments: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    # The Drive file the candidate was delivered to. Since AICP-349 that is
    # the run's EXISTING Run.result_gdrive_file_id whenever it has one: the
    # candidate's content replaces the file's content inside _promote (same
    # id, Creative Link and exported row), and this is set only when that
    # update succeeded. Only a run that had no Drive file yet gets a fresh
    # upload here, which promotion then points the Run at.
    gdrive_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Set when this attempt's candidate fields were copied onto the real
    # Run/RunSegment (the moment it stopped being a candidate and became the
    # active result). NULL for every attempt that never reached `done`, and
    # for a `done` attempt is set in the same commit as the promotion writes —
    # never partially.
    promoted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    run: Mapped["Run"] = relationship("Run", viewonly=True)
    run_segment: Mapped["RunSegment"] = relationship("RunSegment", viewonly=True)

    def __repr__(self) -> str:
        return (
            f"<RunSegmentRangeAttempt id={self.id!r} "
            f"run_segment_id={self.run_segment_id!r} status={self.status!r}>"
        )


#: What a library clip is FOR (AICP-381). ``end_card`` = a short closing card
#: (``CTA_*``, 5-7 s) that may replace a source's own end card; ``insert`` =
#: anything else (a demo / scene to append, 19-28 s). Only a rebuild's default
#: end card reads it; nothing in the stitch phase branches on it.
LIBRARY_CLIP_KIND_END_CARD = "end_card"
LIBRARY_CLIP_KIND_INSERT = "insert"
LIBRARY_CLIP_KINDS = (LIBRARY_CLIP_KIND_END_CARD, LIBRARY_CLIP_KIND_INSERT)


class LibraryClip(Base):
    """A reusable video clip for stitching (e.g. an alternative outro).

    Global — not tied to any project. Referenced from Run.stitch_plan by id;
    the reference is soft (JSON), so deletion is blocked in the API while an
    active run still points at the clip, and the stitch phase re-checks that
    the file exists.
    """

    __tablename__ = "library_clips"
    __table_args__ = (
        CheckConstraint("kind IN ('end_card', 'insert')", name="ck_library_clips_kind"),
    )

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )

    # Operator-facing label ("Outro EN v2")
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Local file under <DATA_DIR>/library/<id>/
    local_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Spoken language of this clip — one of app/languages.py's codes
    # ("en"/"es"/"pt"/"ja"/"de"/"it"/"pl"/"fr"). NULL means language-agnostic (no speech, or a
    # clip that suits any language), which is what every clip uploaded before
    # this column existed is.
    #
    # Purely an operator convenience: a localisation run stitches a
    # pre-uploaded, language-matched tail onto the regenerated hook, and the
    # New Run form uses this to pre-select the tail whose language matches
    # Run.language. Nothing in the stitch phase branches on it — a mismatched
    # or NULL clip stitches exactly the same, so this can never break a run.
    #
    # Deliberately a plain VARCHAR and not an Enum, mirroring Run.language:
    # adding a language must stay a one-line registry change, not a Postgres
    # type migration.
    language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    # LIBRARY_CLIP_KINDS (AICP-381): a rebuild's default end card is the newest
    # language-matched ``end_card``, never an ``insert``. Plain VARCHAR + CHECK
    # like ``language`` (no Postgres enum). Existing rows: ``insert``, except
    # the short ``CTA_*`` cards the migration marked ``end_card``.
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default=LIBRARY_CLIP_KIND_INSERT,
        server_default=LIBRARY_CLIP_KIND_INSERT,
    )

    # Probe metadata (populated at upload time)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<LibraryClip id={self.id!r} name={self.name!r} kind={self.kind!r}>"


# ---------------------------------------------------------------------------
# Overlay feature — standalone "burn a transparent PNG onto a finished video".
# Shares NO state with the run/generation pipeline (separate table, separate
# enum, separate storage). See the Static Image Overlay spec.
# ---------------------------------------------------------------------------


class OverlayJob(Base):
    """One overlay-burn request: a source video + a list of PNG placements."""

    __tablename__ = "overlay_jobs"

    # Primary key (same UUID scheme as Run / VideoProject)
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # State
    status: Mapped[str] = mapped_column(
        Enum(OverlayStatus, name="overlay_status_enum"),
        nullable=False,
        default=OverlayStatus.created,
    )

    # Source: a completed Run's result, a directly-uploaded video, or the
    # output of one of that run's derived creatives (tail edit / caption job).
    source_type: Mapped[str] = mapped_column(
        Enum("run", "upload", "tail_edit", "caption", name="overlay_source_type_enum"),
        nullable=False,
    )
    # Soft reference to runs.id when source_type == "run" (no FK: a run may be
    # deleted independently and the overlay job keeps its resolved local path).
    source_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # Soft reference to run_tail_edits.id when source_type == "tail_edit" (no
    # FK, same reasoning as source_run_id).
    source_tail_edit_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # Soft reference to run_captions.id when source_type == "caption" (no FK,
    # same reasoning again) — the burned-in-captions copy of a run's video.
    source_caption_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # Which Run's creative this overlay IS — denormalized at create time from
    # whatever the source resolves to, so the link survives that source being
    # deleted. Tail edits and caption jobs are hard-deleted by their operator
    # Delete buttons; walking the soft reference to find the run therefore
    # dead-ends, which used to drop the overlay out of its run's history AND
    # out of the sibling count that keeps sheet Job IDs from colliding (see
    # overlay_runs). NULL only for an upload source, or for a row created in
    # the window between migration d7c9e1a3f582 and that deploy.
    parent_run_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, index=True
    )
    source_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Probed source dimensions (filled in at job time, used to convert the
    # normalized overlay coords to integer pixels).
    src_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    src_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    src_duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # List of overlay specs (ordered = z-order, later element on top). Each:
    # {asset_path, nx, ny, sw?, start_sec, end_sec, fade_sec?}. JSON so it works
    # on both SQLite and Postgres.
    overlays: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Delivery
    deliver_to_drive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    result_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gdrive_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Set when the overlay result has been logged to the "Final Creatives" sheet
    # (operator-triggered). Drives the "already added" UI state; NULL = not added.
    final_creatives_exported_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    # LEGACY: the "Add to Caption Demo" sheet export was removed on 2026-08-28.
    # Nothing writes this column any more; the values that are here record what
    # the feature did while it existed. Kept so the ORM matches the live
    # schema — do NOT drop it in an autogenerated migration.
    caption_demo_status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )

    # Soft-delete marker. When set, the job is hidden from all listings and
    # lookups (treated as 404) but the row is retained so the source video
    # reference (source_run_id / source_local_path) stays findable in the DB.
    # Only the rendered result + uploaded PNG files are removed from disk.
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return (
            f"<OverlayJob id={self.id!r} status={self.status!r} "
            f"source_type={self.source_type!r}>"
        )


class VoiceClip(Base):
    """A reusable voice sample used to steer generated speech (Seedance 2.5).

    Global — not tied to any project or run. Seedance invents a fresh voice for
    every generation it is asked to write audio for, so the same character comes
    back sounding like a different person on each run; attaching one of these as
    ``reference_audio_urls`` fixes the timbre. Language-tagged for the same
    reason :class:`LibraryClip` is: a localisation run wants the voice matching
    the language it is generating, and the New Run form pre-selects on it.

    Like :class:`OverlayAsset` and unlike :class:`LibraryClip`, a run never
    references one of these rows by id: picking a voice COPIES the file into the
    run's own directory and stores that path in ``Run.reference_audio_urls``.
    Deleting a library voice therefore cannot alter, break, or silently
    re-voice an existing run — including one re-submitted months later — so no
    active-reference guard is needed on delete.

    The file is always the output of ``media.make_voice_reference``: a mono mp3
    of at most ``ai_models.VOICE_REF_MAX_SEC``, which is what makes the format
    and duration rules kie.ai enforces unbreakable by construction.
    """

    __tablename__ = "voice_clips"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )

    # Operator-facing label ("Marissa — calm read")
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Local file under <DATA_DIR>/voice_library/<id>/voice.mp3
    local_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Spoken language of this sample — one of app/languages.py's codes. NULL
    # means untagged/language-agnostic. Purely an operator convenience, exactly
    # as on LibraryClip: nothing in the pipeline branches on it, so a mismatched
    # tag makes a run sound wrong, never fail.
    language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    # Duration of the NORMALISED file (post-truncation), populated at upload.
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<VoiceClip id={self.id!r} name={self.name!r}>"


class OverlayAsset(Base):
    """A reusable transparent PNG "card" for the overlay-burn feature.

    Global — not tied to any project or overlay job. Unlike LibraryClip (which
    is referenced *by id* from Run.stitch_plan, a soft JSON reference resolved
    at stitch time), an overlay job never references an OverlayAsset row
    directly: attaching one to a job COPIES the PNG file into that job's own
    assets dir (see api_overlay.py's attach endpoint), and the job's
    ``overlays`` JSON only ever stores the job-relative ``asset_path`` the
    render pipeline already expects. That keeps the asset_path contract
    (resolved against the job root) completely unchanged and means deleting a
    library asset can never affect an already-rendered or in-flight job — so,
    unlike ``delete_library_clip``, no reference/active-job guard is needed here.
    """

    __tablename__ = "overlay_assets"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )

    # Operator-facing label ("Logo bug — bottom left")
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Local file under <DATA_DIR>/overlay_library/<id>/
    local_path: Mapped[str] = mapped_column(Text, nullable=False)

    # PNG metadata (populated at upload time, read from the IHDR chunk)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<OverlayAsset id={self.id!r} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Run automation — scheduled batch generation from the persona spreadsheet.
# An operator enables automation on a proven Run; the periodic automation tick
# (app/tasks.py) then copies that Run with fresh reference photos pulled from
# the "Daily Queue" sheet, up to daily_count launches per UTC day, and fires a
# webhook when each launched batch finishes. See app/automation.py.
# ---------------------------------------------------------------------------


class RunAutomation(Base):
    """Per-Run automation settings + filter rule for picking sheet rows."""

    __tablename__ = "run_automations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # The template Run whose prompt/model/segment tweaks every launch clones.
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Filter rule — case-insensitive, trimmed match against the sheet columns;
    # NULL/empty means "any". Persona/gender/ethnicity accept several values
    # separated by commas (row matches ANY of them) and each has an exclude
    # flag that inverts the filter (rows matching any listed value are
    # skipped). Age is single-valued and range-aware (see _age_matches).
    filter_persona: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    filter_age: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    filter_gender: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    filter_ethnicity: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    exclude_persona: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    exclude_gender: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    exclude_ethnicity: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Which photo column to use: "close-up" → image_on_gdrive (R),
    # "midplan" → link (Y). Plain string (validated in the API) so we don't
    # grow another Postgres enum type.
    plan: Mapped[str] = mapped_column(
        String(16), nullable=False, default="close-up"
    )

    # How many videos to launch per UTC day for this Run.
    daily_count: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

    # Re-run the template's GPT Image 2 preprocessing on each sheet photo
    # instead of handing the raw photo to Seedance: the launch clones the
    # template's frames and image prompt and swaps in the row's photo as the
    # character reference. Results are auto-approved (the template's recipe was
    # approved once, by hand). Only settable when the template run itself used
    # preprocessing — see api_v2.put_run_automation.
    preprocess_refs: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Override for the batch-completion webhook; falls back to
    # settings.AUTOMATION_WEBHOOK_URL when NULL.
    webhook_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    run: Mapped["Run"] = relationship("Run")
    launches: Mapped[list["AutomationLaunch"]] = relationship(
        "AutomationLaunch",
        back_populates="automation",
        cascade="all, delete-orphan",
        order_by="AutomationLaunch.launched_at",
    )

    def __repr__(self) -> str:
        return (
            f"<RunAutomation id={self.id!r} run_id={self.run_id!r} "
            f"enabled={self.enabled!r} daily_count={self.daily_count!r}>"
        )


class AutomationLaunch(Base):
    """One launched copy: which sheet row was consumed and the Run it spawned.

    Rows double as the automation's ledger: the daily quota counts launches
    with ``launched_at`` in the current UTC day, and a sheet row (queue_id) is
    never reused by the same automation thanks to the unique constraint.
    ``notified_at`` is set on every launch of a batch when that batch's
    completion webhook has been delivered.
    """

    __tablename__ = "automation_launches"
    __table_args__ = (
        # A given sheet row is consumed at most once per automation.
        # (Different automations MAY use the same photo.)
        # Named for Alembic/Postgres friendliness.
        UniqueConstraint(
            "automation_id", "queue_id", name="uq_automation_launch_queue"
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    automation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("run_automations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Launches fired by one tick share a batch_id; the webhook is per batch.
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    # Sheet row identity + the reference photo URL that was used.
    queue_id: Mapped[str] = mapped_column(String(64), nullable=False)
    ref_url: Mapped[str] = mapped_column(Text, nullable=False)

    # The spawned Run. SET NULL so deleting a run keeps the ledger row (the
    # sheet row stays consumed and the daily quota stays honest).
    new_run_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    launched_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    notified_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    automation: Mapped["RunAutomation"] = relationship(
        "RunAutomation", back_populates="launches"
    )
    new_run: Mapped[Optional["Run"]] = relationship("Run", foreign_keys=[new_run_id])

    def __repr__(self) -> str:
        return (
            f"<AutomationLaunch id={self.id!r} queue_id={self.queue_id!r} "
            f"batch_id={self.batch_id!r} new_run_id={self.new_run_id!r}>"
        )


# ---------------------------------------------------------------------------
# PIP Insert — extraction and merge of a picture-in-picture insert region.
# The extracted insert becomes a normal child VideoProject that flows through
# the existing analyze → Run → stitch pipeline. The merge composites the
# processed child Run result back onto a target video at specified coordinates.
# ---------------------------------------------------------------------------


class InsertExtraction(Base):
    """Provenance and placement of an extracted picture-in-picture insert.

    The extraction links a parent project (source of the insert) to a child
    project (the cropped insert clip, processed as a normal project). One
    extraction per child project (unique constraint on child_project_id).
    """

    __tablename__ = "insert_extractions"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign keys
    parent_project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("video_projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    child_project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("video_projects.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )

    # Placement in the parent video (pixel coordinates, not normalized)
    x: Mapped[int] = mapped_column(Integer, nullable=False)
    y: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timing window in the parent (in seconds)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Rounded-corner mask parameter (in pixels, default 0 = square corners)
    corner_radius_px: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relationships
    parent_project: Mapped["VideoProject"] = relationship(
        "VideoProject", foreign_keys=[parent_project_id], viewonly=True
    )
    child_project: Mapped["VideoProject"] = relationship(
        "VideoProject", foreign_keys=[child_project_id], viewonly=True
    )

    def __repr__(self) -> str:
        return (
            f"<InsertExtraction id={self.id!r} parent_project_id={self.parent_project_id!r} "
            f"child_project_id={self.child_project_id!r}>"
        )


class InsertMerge(Base):
    """One merge operation: composite a processed child Run back onto a target video.

    Status tracks the async merge pipeline: queued → compositing → delivering → done.
    The extraction's placement is prefilled but may be overridden here (to merge the
    same processed insert into a different target video or at different coordinates).
    """

    __tablename__ = "insert_merges"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign keys
    target_project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("video_projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    child_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Placement in the target video (pixel coordinates, not normalized)
    x: Mapped[int] = mapped_column(Integer, nullable=False)
    y: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timing window in the target (in seconds)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Rounded-corner mask parameter (in pixels, default 0 = square corners)
    corner_radius_px: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # State
    status: Mapped[str] = mapped_column(
        Enum(MergeStatus, name="merge_status_enum"),
        nullable=False,
        default=MergeStatus.queued,
    )

    # Results and delivery
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gdrive_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    target_project: Mapped["VideoProject"] = relationship(
        "VideoProject", foreign_keys=[target_project_id], viewonly=True
    )
    child_run: Mapped["Run"] = relationship("Run", foreign_keys=[child_run_id])

    def __repr__(self) -> str:
        return (
            f"<InsertMerge id={self.id!r} target_project_id={self.target_project_id!r} "
            f"child_run_id={self.child_run_id!r} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Tail Edit — "Change ending" tool: swap the outro of an already-delivered
# Run's final video without re-running the AI pipeline. Pure ffmpeg: trim the
# run's result to [0, cut_sec) and optionally append a stitch-library clip.
# See app/pipeline_tail.py.
# ---------------------------------------------------------------------------


class RunTailEdit(Base):
    """One "change ending" request against a done Run's stitched result.

    ``cut_sec`` marks where the source final video is cut (the kept prefix is
    ``[0, cut_sec)``). ``library_clip_id`` is a soft reference (like
    ``Run.stitch_plan`` entries) to a LibraryClip appended after the cut point;
    NULL means "trim only" — no clip is appended, the output is just the
    re-encoded prefix. Status tracks the async pipeline: queued -> processing
    -> delivering -> done.
    """

    __tablename__ = "run_tail_edits"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign key
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Where the source final video is cut; the kept prefix is [0, cut_sec).
    cut_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Soft reference to a LibraryClip appended after the cut point (no FK —
    # mirrors Run.stitch_plan's library entries; deletion is blocked in the
    # API while an active tail edit still points at the clip). NULL = trim
    # only, no appended clip.
    library_clip_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Which video this tail edit actually reads. `run_id` above is ALWAYS the
    # parent Run (provenance: project id, language suffix, sheet `Idea`,
    # output filename) — never the input video. `source_type` +
    # `source_caption_id` say what file is read instead:
    #   "run"     (default) -> run.result_local_path, byte-for-byte the v1
    #             behaviour. server_default='run' means every pre-existing
    #             row keeps meaning exactly what it meant before this column
    #             existed.
    #   "caption" -> the referenced RunCaption's output_path (captioned body,
    #             re-cut with an uncaptioned new tail).
    # Mirrors OverlayJob.source_type / source_run_id.
    source_type: Mapped[str] = mapped_column(
        Enum("run", "caption", name="tail_edit_source_type_enum"),
        nullable=False,
        server_default="run",
    )
    # Soft reference to run_captions.id when source_type == "caption" (no FK
    # — same reasoning as OverlayJob.source_run_id: a caption job may be
    # deleted independently and this row's job keeps whatever it already
    # resolved). NULL when source_type == "run".
    source_caption_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # State
    status: Mapped[str] = mapped_column(
        Enum(TailEditStatus, name="tail_edit_status_enum"),
        nullable=False,
        default=TailEditStatus.queued,
    )

    # Results and delivery
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gdrive_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Set when this tail edit's result has been logged to the "Final
    # Creatives" sheet (operator-triggered). Drives the "already added" UI
    # state; NULL = not added. Mirrors Run.final_creatives_exported_at.
    final_creatives_exported_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    # LEGACY: the "Add to Caption Demo" sheet export was removed on 2026-08-28.
    # Nothing writes this column any more; the values that are here record what
    # the feature did while it existed. Kept so the ORM matches the live
    # schema — do NOT drop it in an autogenerated migration.
    caption_demo_status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )

    # Relationship
    run: Mapped["Run"] = relationship("Run", viewonly=True)

    def __repr__(self) -> str:
        return (
            f"<RunTailEdit id={self.id!r} run_id={self.run_id!r} "
            f"cut_sec={self.cut_sec!r} status={self.status!r}>"
        )


class RunCaption(Base):
    """One "Add Captions" request against a done Run's delivered result.

    We re-transcribe the *delivered* video (not the source), render the
    spoken lines into an ASS subtitle file, burn them in with ffmpeg, and
    deliver the captioned copy to Google Drive as a new derived creative.
    Mirrors RunTailEdit — "a derived creative built from a done Run's result,
    delivered to Drive, exportable to the sheet." See docs/burned-captions.md.

    Status tracks the async pipeline: queued -> transcribing -> burning ->
    delivering -> done (transcribing is skipped when lines_source == "manual").
    """

    __tablename__ = "run_captions"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Foreign key
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Caption language. NULL is resolved at process time as
    # `caption.language or run.language or transcript["source_language"]`.
    language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    # The timed lines actually burned: [{"start": float, "end": float, "text":
    # str}, ...]. Written by the transcribe step; also the operator-edit
    # target (PATCH /captions/{id}/lines).
    lines: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # "transcribe", "manual" or "reused". "manual" means an operator
    # supplied/edited `lines` and the transcribe step MUST be skipped
    # (queued -> burning). "reused" means the lines were copied from another
    # caption whose source had the same audio (see `audio_sha256`).
    lines_source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # media.audio_fingerprint of the video this job read (the RESOLVED
    # source, not necessarily the run's result). Equal fingerprints mean
    # bit-identical decoded audio, so a new caption job copies the lines of a
    # caption that shares it instead of transcribing again. NULL = not
    # computed (no audio, decode failure, or a row older than the column).
    audio_sha256: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    # Soft reference to the run_captions.id `lines` were copied from when
    # lines_source == "reused" (no FK: the donor can be hard-deleted and
    # this row keeps its copy).
    lines_reused_from: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Which video this caption job actually transcribes/burns. `run_id` above
    # is ALWAYS the parent Run (provenance: project id, language suffix,
    # sheet `Idea`, output filename) — never the input video. `source_type` +
    # `source_tail_edit_id` say what file is read instead:
    #   "run"       (default) -> run.result_local_path, byte-for-byte the v1
    #               behaviour. server_default='run' means every pre-existing
    #               row keeps meaning exactly what it meant before this
    #               column existed.
    #   "tail_edit" -> the referenced RunTailEdit's output_path (captions
    #               burned across the whole re-cut video, tail included).
    # Mirrors OverlayJob.source_type / source_run_id.
    source_type: Mapped[str] = mapped_column(
        Enum("run", "tail_edit", name="caption_source_type_enum"),
        nullable=False,
        server_default="run",
    )
    # Soft reference to run_tail_edits.id when source_type == "tail_edit" (no
    # FK — same reasoning as OverlayJob.source_run_id: a tail edit may be
    # deleted independently and this row's job keeps whatever it already
    # resolved). NULL when source_type == "run".
    source_tail_edit_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # captions.CAPTION_STYLE_VERSION at burn time.
    style_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # The caption placement this job burns with (AICP-368), frozen from the
    # project's caption_placement when the job was created (or an explicit
    # {"mode": "bottom"} when the operator chose the house position), so a
    # retry after a project edit renders the same. NULL = house style (every
    # caption made before AICP-368).
    placement: Mapped[Optional[dict]] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )

    # AICP-351 (docs/assistant-drain-check-design.md §5.2): the caption's ASR
    # dispatch fence. `asr_claim` is set (a fresh token) by the job that moves
    # the caption into `transcribing` and is checked by every LLM attempt,
    # call claim and transcript write of that job; the reaper clears it when
    # it re-dispatches the ASR, so a stale job is fenced out. `asr_epoch` is
    # bumped by an operator Retry: the 3-attempt automatic bound is per epoch
    # (paid_submit_attempts.owner_id = "<caption id>:<epoch>").
    asr_claim: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    asr_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0,
                                           server_default="0")

    # State
    status: Mapped[str] = mapped_column(
        Enum(CaptionStatus, name="caption_status_enum"),
        nullable=False,
        default=CaptionStatus.queued,
    )

    # Results and delivery
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gdrive_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Set when this caption's result has been logged to the "Final Creatives"
    # sheet (operator-triggered). Mirrors RunTailEdit.final_creatives_exported_at.
    final_creatives_exported_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    # Relationship
    run: Mapped["Run"] = relationship("Run", viewonly=True)

    def __repr__(self) -> str:
        return (
            f"<RunCaption id={self.id!r} run_id={self.run_id!r} "
            f"status={self.status!r}>"
        )


class RunAudioPatch(Base):
    """One "re-voice this window from another run" correction of a done Run.

    Seedance writes the speech for a localisation run itself, and it
    occasionally mispronounces a brand name in an otherwise perfect take while
    a sibling run of the SAME project — same script, same driving footage —
    says it correctly. This row records taking that one window of audio from
    the sibling and splicing it into this run's delivered final: video stream
    copied untouched, no sample outside the window moved
    (``media.splice_audio_window``).

    Unlike RunTailEdit / RunCaption / OverlayJob this is deliberately **not** a
    derived creative. It produces no new Job ID, no sheet row and no new Drive
    file — it repairs the run's OWN result, in place, so the existing "Add to
    Final Creatives" export, the Drive link (the same file's content is
    replaced), Sent-to-Final, and every tail edit / caption / overlay that
    chains off this run keep working with no changes at all. A version of a
    creative with the brand mispronounced is a defect, not a deliverable worth
    tracking alongside the good one.

    Its shape therefore follows RunSegmentRangeAttempt, the codebase's other
    "change a done run in place, with history": append-only rows, candidate
    bytes written under ``storage.audio_patch_dir`` and never over an active
    path, and a promotion that re-points the Run only once the render AND the
    Drive update have both succeeded. ``done``/``failed`` are terminal — a
    further correction is a NEW row, which patches whatever the current result
    is by then, so successive patches compose.
    """

    __tablename__ = "run_audio_patches"

    # At most one non-terminal patch per run — a DB-level partial unique index,
    # not just api_v2._active_audio_patch's non-locking SELECT, which two
    # concurrent create requests can both pass. Same reasoning, spelling and
    # sync obligation as RunSegmentRangeAttempt's index above: the WHERE clause
    # here, the one in the migration, and AUDIO_PATCH_ACTIVE_STATUSES
    # (state_machine.py) must be kept in step by hand, because models.py cannot
    # import that module without a cycle.
    __table_args__ = (
        Index(
            "uq_run_audio_patches_active_per_run",
            "run_id",
            unique=True,
            postgresql_where=text("status IN ('queued','processing','delivering')"),
            sqlite_where=text("status IN ('queued','processing','delivering')"),
        ),
        # AICP-389: a donor row names its donor; a filter row is fully bound to
        # the approved bytes. Same text as migration v3a9c5e7b1d4.
        CheckConstraint(
            "(kind = 'donor' AND donor_run_id IS NOT NULL AND donor_start_sec IS NOT NULL)"
            " OR (kind = 'same_run_lowpass' AND filter_params IS NOT NULL"
            " AND base_sha256 IS NOT NULL AND candidate_sha256 IS NOT NULL"
            " AND candidate_path IS NOT NULL)",
            name="ck_run_audio_patches_kind_fields",
        ),
    )

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )

    # Optimistic-concurrency guard — see RunSegmentRangeAttempt.version_id for
    # the failure this closes. It matters here for the same reason it matters
    # there and not for a tail edit: this row's terminal commit MUTATES the
    # parent Run, so two sessions processing the SAME patch must not both get
    # to promote.
    version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __mapper_args__ = {"version_id_col": version_id}

    # The Run being repaired: its video, its Drive file, its sheet identity.
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The Run whose audio is borrowed. A SOFT reference (no FK) on purpose: the
    # donor is only read, once, during processing — after that this column is
    # pure provenance, and deleting the donor run must not cascade away the
    # record of what was done to THIS run's audio. A donor deleted while the
    # patch is still in flight simply fails the job with a clear message.
    # NULL only for kind ``same_run_lowpass`` (the check constraint above).
    donor_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # AICP-389. ``donor`` = re-voice the window from a sibling run (the
    # original feature); ``same_run_lowpass`` = the autopilot's free brand fix:
    # the window of THIS run's own audio low-passed (app.brand_check). A filter
    # row promotes the operator-approved ``candidate_path`` as is — it is never
    # re-rendered — and the promotion re-checks both hashes under the run lock.
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="donor",
                                      server_default="donor")
    filter_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    base_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    candidate_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    candidate_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # The window of THIS run's final whose audio is replaced.
    target_start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    target_end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    # Where the replacement starts in the DONOR's final. There is no donor end:
    # the replacement is always exactly ``target_end_sec - target_start_sec``
    # long, because anything else would have to time-stretch the borrowed audio
    # or move every sample after the window — and moving them is the desync
    # this whole feature exists to avoid.
    donor_start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Seam ramp length, seconds. Stored per patch so an operator can widen it
    # for a seam that clicks or tighten it when the window is very short.
    crossfade_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # The Run.result_local_path this patch was computed from, captured at
    # create time. Doubles as provenance ("what this was patched from" — the
    # pre-patch file is never deleted, so this is also how to get back) and as
    # the staleness check in _promote: if the run's result has moved since,
    # something else re-pointed it (another patch, a range attempt) and this
    # candidate is refused rather than silently overwriting that.
    base_result_path: Mapped[str] = mapped_column(Text, nullable=False)

    # State
    status: Mapped[str] = mapped_column(
        Enum(AudioPatchStatus, name="audio_patch_status_enum"),
        nullable=False,
        default=AudioPatchStatus.queued,
    )

    # The candidate patched video, under storage.audio_patch_dir — never over
    # the run's own results/final.mp4, so a failed patch leaves the delivered
    # result byte-for-byte intact and a promoted one leaves its predecessor on
    # disk.
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # The Drive file whose CONTENT was replaced with the patched video —
    # normally the run's existing result_gdrive_file_id, which is what keeps
    # the sheet's Creative Link and any already-exported row pointing at the
    # fixed video. Set to a freshly uploaded file's id only when the run had no
    # Drive result to begin with.
    gdrive_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Set in the same commit that re-points the Run at output_path. NULL for
    # every patch that never reached `done`.
    promoted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationship
    run: Mapped["Run"] = relationship("Run", viewonly=True)

    @property
    def window_sec(self) -> float:
        """Length of the replaced window — what the donor must supply."""
        return self.target_end_sec - self.target_start_sec

    def __repr__(self) -> str:
        return (
            f"<RunAudioPatch id={self.id!r} run_id={self.run_id!r} "
            f"donor_run_id={self.donor_run_id!r} "
            f"window=[{self.target_start_sec!r}, {self.target_end_sec!r}] "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# kie.ai daily spend cap (AICP-155) — see app/spend.py for the accounting model.
# Amounts are integer MILLI-credits (1 credit = 1000), so the cap comparison is
# exact; the USD figure is derived at display time from KIE_USD_PER_CREDIT.
# ---------------------------------------------------------------------------


class KieSpendDay(Base):
    """Per-UTC-day total of charged spend; its row is also the day's lock."""

    __tablename__ = "kie_spend_days"
    __table_args__ = (
        CheckConstraint("charged_mcredits >= 0", name="ck_kie_spend_days_nonneg"),
    )

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    charged_mcredits: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now()
    )


class KieSpendReservation(Base):
    """What one action (scope) reserved and consumed on one UTC day.

    ``charged = max(reserved, consumed)`` is what counts against the day.
    """

    __tablename__ = "kie_spend_reservations"
    __table_args__ = (
        UniqueConstraint("scope_key", "day", name="uq_kie_spend_reservation_scope_day"),
        CheckConstraint(
            "reserved_mcredits >= 0 AND consumed_mcredits >= 0 AND charged_mcredits >= 0",
            name="ck_kie_spend_reservations_nonneg",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    scope_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    day: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    # Soft ref (no FK): a hard-deleted run must not take its spend with it.
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    # Which provider the spend is for (AICP-330 B2). BOTH defaults matter: a
    # mapped NOT NULL column without a Python default makes SQLAlchemy send an
    # explicit NULL from every insert that does not name it, and spend's draws
    # swallow that error. spend.py also passes it explicitly on every insert.
    provider: Mapped[str] = mapped_column(
        String(16), nullable=False, default="kie", server_default=text("'kie'")
    )
    reserved_mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    consumed_mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    charged_mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())


class KieSpendDraw(Base):
    """One paid kie.ai call, recorded once (``call_key`` = kie taskId, or a
    generated id for a synchronous LLM call). ``mcredits`` starts as the
    estimate and is replaced by kie.ai's ``creditsConsumed`` when the task's
    terminal state reports it (``settled_at``)."""

    __tablename__ = "kie_spend_draws"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    call_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    reservation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("kie_spend_reservations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(
        String(16), nullable=False, default="kie", server_default=text("'kie'")
    )
    estimated_mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mcredits: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())
    settled_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class ManualCreative(Base):
    """A creative produced OUTSIDE the re-skin pipelines — by an operator
    script or by hand — registered so it has a history and a Job ID.

    The pipelines leave their own trail (Run → tail edit → caption → overlay),
    but a one-off script that, say, burns a localised card over an externally
    dubbed Drive video leaves only a bare ``source_type="upload"`` overlay job:
    no record of where the video came from, and a Final Creatives row with an
    unsuffixed Job ID and no Idea. This row is that missing trail (AICP-239).

    Everything needed to trace the creative is COPIED in at registration
    (source link, overlay asset id/name, placement in ``details``) rather than
    resolved later through ``overlay_job_id``: overlay jobs are soft-deleted
    with their files and library assets are hard-deleted, so a soft reference
    alone could dead-end — the lesson of OverlayJob.parent_run_id.

    The sheet ``Job ID`` is ``languages.job_id_for(id, language)``
    (``b73fe636-de``), under Pipeline Type ``re-skin-manual-script``. See
    app/manual_creatives.py.
    """

    __tablename__ = "manual_creatives"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), index=True
    )

    # Final Creatives "Pipeline Type" (re-skin-manual-script)
    pipeline_type: Mapped[str] = mapped_column(String(64), nullable=False)
    # Final Creatives "Idea" — operator-given, validated against the
    # downstream rename rule (see manual_creatives.validate_idea)
    idea: Mapped[str] = mapped_column(String(40), nullable=False)
    # app/languages.py code; NULL = English (unsuffixed Job ID)
    language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    # Where the input came from (a Drive link, as given) — the sheet's Input Link
    source_link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The delivered creative — the sheet's Creative Link
    gdrive_file_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )

    # Soft ref to the overlay job that rendered it, if any (no FK: see above)
    overlay_job_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, unique=True
    )
    overlay_asset_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    overlay_asset_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Snapshot of anything else worth keeping (overlay placements, …)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    linear_issue: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Stamped only AFTER the sheet append (or dedupe) succeeded
    final_creatives_exported_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    def __repr__(self) -> str:
        return f"<ManualCreative id={self.id!r} idea={self.idea!r}>"


# Register assistant tables for Alembic/Base.metadata; legacy relationships and
# deletion cascades deliberately do not own assistant history or artifacts.
from . import assistant_models, assistant_jobs, assistant_turns  # noqa: E402, F401
# Knowledge-base evidence (AICP-295): soft references only, no relationship to
# legacy tables, so Run/Project deletion never cascades into it.
from .knowledge import models as knowledge_models  # noqa: E402, F401
# Second-provider tasks (Fal, AICP-312/330): soft owner references only, so a
# consumer's deletion never removes the record of a possibly billed request.
from . import provider_models  # noqa: E402, F401
from . import reframe_models  # noqa: E402, F401
from . import segment_fill_models  # noqa: E402, F401
from . import models_autopilot  # noqa: E402, F401
from . import localisation_jobs_models  # noqa: E402, F401
# Rebuild from reference (docs/rebuild-from-reference-design.md §4).
from . import rebuild_models  # noqa: E402, F401
# Deploy admission gate + idempotency keys (AICP-350, app/ops_gate.py).
from . import ops_gate  # noqa: E402, F401
# AICP-351 (docs/assistant-drain-check-design.md §5): paid submit attempts.
from . import paid_attempts_models  # noqa: E402, F401
