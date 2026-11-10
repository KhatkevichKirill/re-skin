"""ORM models of the "Rebuild from reference" project type.

docs/rebuild-from-reference-design.md §4. Five tables:

* ``rebuild_shots`` — detected shots on the source timeline, plus the
  storyboard fields (seeded by the storyboard job, operator-editable) and the
  short-shot folding (``folded_into`` / ``force_anchor``, design §5.1.3).
* ``project_storyboards`` — durable Gemini storyboard jobs (the AICP-365
  ``_DispatchColumns`` protocol); one live row per project.
* ``rebuild_keyframes`` — one row per GPT Image attempt (a shot keyframe or a
  character sheet). Paid-call protocol (design §7.2).
* ``rebuild_chunk_attempts`` — one row per paid video task for one chunk of an
  images-only run. Paid-call protocol.
* ``rebuild_llm_calls`` — one row per Gemini call (storyboard global / shot
  calls, the per-attempt character check). Paid-call protocol (a synchronous
  call has no task id: ``accepted`` stores the response and goes straight to
  ``succeeded``).

Every paid provider call in this feature is ONE row and AT MOST ONE provider
request; the state machine and its primitives live in
``app/rebuild_paid_call.py``. The partial-index WHERE texts below are
byte-identical in the migration (tests compare them).

Imported at the end of ``models.py`` so ``Base.metadata`` and Alembic see them.
Times are naive UTC, as everywhere else.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .localisation_jobs_models import JOB_STATUSES, _DispatchColumns


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _in(col: str, values) -> str:
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


# ---------------------------------------------------------------------------
# Paid-call states (design §7.2)
# ---------------------------------------------------------------------------

PC_QUEUED = "queued"
PC_CLAIMED = "claimed"
PC_SENDING = "sending"
PC_SUBMITTED = "submitted"
PC_FETCHING = "fetching"
PC_SUCCEEDED = "succeeded"
PC_FAILED = "failed"
PC_UNKNOWN = "unknown"

PAID_CALL_STATUSES = (
    PC_QUEUED, PC_CLAIMED, PC_SENDING, PC_SUBMITTED, PC_FETCHING,
    PC_SUCCEEDED, PC_FAILED, PC_UNKNOWN,
)
# Non-terminal: a row in any of these may still lead to (or be) a provider
# request, so it counts as "live" for deletion, re-analysis, a replacement
# attempt and the deploy drain.
PAID_CALL_LIVE_STATUSES = (PC_QUEUED, PC_CLAIMED, PC_SENDING, PC_SUBMITTED, PC_FETCHING)
PAID_CALL_TERMINAL_STATUSES = (PC_SUCCEEDED, PC_FAILED, PC_UNKNOWN)
# Nothing has been sent yet from these (the send step runs only from `claimed`
# after a fenced CAS to `sending`).
PAID_CALL_UNSENT_STATUSES = (PC_QUEUED, PC_CLAIMED)

PAID_CALL_STATUS_CHECK = _in("status", PAID_CALL_STATUSES)
PAID_CALL_LIVE_WHERE = _in("status", PAID_CALL_LIVE_STATUSES)

STORYBOARD_LIVE_WHERE = "status IN ('queued', 'running')"

LLM_KINDS = ("global", "shot", "character_check")
LLM_SHOT_LIVE_WHERE = f"kind = 'shot' AND {PAID_CALL_LIVE_WHERE}"
LLM_GLOBAL_LIVE_WHERE = f"kind = 'global' AND {PAID_CALL_LIVE_WHERE}"
LLM_CHECK_LIVE_WHERE = f"kind = 'character_check' AND {PAID_CALL_LIVE_WHERE}"
LLM_CHECK_SLOT_WHERE = "kind = 'character_check'"

KF_SHOT_SELECTED_WHERE = "selected AND owner_run_id IS NULL AND shot_id IS NOT NULL"
KF_SHEET_SELECTED_WHERE = "selected AND owner_run_id IS NULL AND shot_id IS NULL"

ATTEMPT_SELECTED_WHERE = "selected"

STORYBOARD_BUILD_MODES = ("resume", "from_scratch")


class _PaidCallColumns:
    """Columns of the paid-call protocol (design §7.2), shared by keyframes,
    chunk attempts and LLM calls.

    ``request`` is the exact provider request body (URLs included) — written at
    the ``sending`` CAS, so a row that reached ``sending`` always records what
    was (or may have been) sent. ``sender_*`` identify the process that made
    the request, for the reaper's sender proof; ``sender_note`` logs how a
    ``sending`` row was released (boot change, horse reap, host attestation).
    """

    status: Mapped[str] = mapped_column(String(16), nullable=False, default=PC_QUEUED)
    dispatch_gen: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    claim_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    rq_job_id: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    not_live_since: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    send_deadline: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    sender_host: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    sender_boot: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sender_job: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    sender_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    request: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    estimate_credits: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_credits: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fetch_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class RebuildShot(Base):
    """One detected camera shot of a rebuild project's source (design §4, §5.1)."""

    __tablename__ = "rebuild_shots"
    __table_args__ = (
        UniqueConstraint("project_id", "index", name="uq_rebuild_shots_project_index"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False,
        index=True,
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    # The analysis generation that created this row (design §7.4). A
    # re-analysis writes generation = previous max + 1; a storyboard records
    # the generation it described (project_storyboards.shots_generation).
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    # Where the keyframe's source frame is taken (default the midpoint, Q4).
    frame_sec: Mapped[float] = mapped_column(Float, nullable=False)
    # Storyboard fields (seeded by the storyboard job, operator-editable).
    location: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    framing: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    camera: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    characters: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    speaking_on_camera: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_brand_card: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sheet_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # The storyboard whose result last filled the fields (soft reference).
    storyboard_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # Folding (§5.1.3): NULL for an anchor, else the anchor's id (soft
    # reference, same project + generation). force_anchor = operator "unfold".
    folded_into: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    force_anchor: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Set when the operator edits any storyboard field; a later storyboard
    # does not overwrite an edited shot unless asked.
    edited_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow)

    @property
    def duration_sec(self) -> float:
        return float(self.end_sec) - float(self.start_sec)


class ProjectStoryboard(_DispatchColumns, Base):
    """One durable storyboard build (design §5.2). AICP-365 dispatch protocol."""

    __tablename__ = "project_storyboards"
    __table_args__ = (
        CheckConstraint(_in("status", JOB_STATUSES), name="ck_project_storyboards_status"),
        CheckConstraint(
            "build_mode IS NULL OR " + _in("build_mode", STORYBOARD_BUILD_MODES),
            name="ck_project_storyboards_build_mode",
        ),
        Index("uq_project_storyboards_live", "project_id", unique=True,
              postgresql_where=text(STORYBOARD_LIVE_WHERE),
              sqlite_where=text(STORYBOARD_LIVE_WHERE)),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False,
        index=True,
    )
    # The shot generation this storyboard describes; a result is applied only
    # while it still equals the project's current generation.
    shots_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    build_mode: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # The global call this storyboard's shot results must have been made under
    # (Resume copies it from the previous storyboard, §5.2). Soft reference.
    global_call_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # The model that answered the global call.
    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # {style, characters, sheet_groups, audio, on_screen_text, shots}.
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # What admission reserved ((1 + N) x 2 x llm_credits); the job admits call
    # rows only while the drawn total stays within it.
    budget_credits: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    overwrite_edits: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class RebuildKeyframe(_PaidCallColumns, Base):
    """One GPT Image attempt: a shot keyframe, or a character sheet (shot_id NULL)."""

    __tablename__ = "rebuild_keyframes"
    __table_args__ = (
        CheckConstraint(PAID_CALL_STATUS_CHECK, name="ck_rebuild_keyframes_status"),
        # A sheet row names its outfit/setting group; a shot row does not.
        CheckConstraint(
            "(shot_id IS NULL AND sheet_key IS NOT NULL) OR "
            "(shot_id IS NOT NULL AND sheet_key IS NULL)",
            name="ck_rebuild_keyframes_slot",
        ),
        Index("uq_rebuild_kf_shot_selected", "shot_id", unique=True,
              postgresql_where=text(KF_SHOT_SELECTED_WHERE),
              sqlite_where=text(KF_SHOT_SELECTED_WHERE)),
        Index("uq_rebuild_kf_sheet_selected", "project_id", "sheet_key", unique=True,
              postgresql_where=text(KF_SHEET_SELECTED_WHERE),
              sqlite_where=text(KF_SHEET_SELECTED_WHERE)),
        Index("ix_rebuild_keyframes_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False,
        index=True,
    )
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    shot_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("rebuild_shots.id", ondelete="CASCADE"), nullable=True,
        index=True,
    )
    # NULL in phase 1; phase 2's persona keyframes are run-owned.
    owner_run_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=True,
    )
    sheet_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # The exact local files fed to GPT Image (frames), in order.
    input_paths: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    resolution: Mapped[str] = mapped_column(String(8), nullable=False, default="2K")
    aspect_ratio: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RebuildChunkAttempt(_PaidCallColumns, Base):
    """One paid video task for one chunk (RunSegment) of an images-only run."""

    __tablename__ = "rebuild_chunk_attempts"
    __table_args__ = (
        CheckConstraint(PAID_CALL_STATUS_CHECK, name="ck_rebuild_chunk_attempts_status"),
        UniqueConstraint("run_segment_id", "attempt_no",
                         name="uq_rebuild_chunk_attempts_segment_no"),
        # At most one live paid attempt per chunk: a duplicate admission (two
        # entries racing) fails in the database.
        Index("uq_rebuild_attempt_live", "run_segment_id", unique=True,
              postgresql_where=text(PAID_CALL_LIVE_WHERE),
              sqlite_where=text(PAID_CALL_LIVE_WHERE)),
        Index("uq_rebuild_attempt_selected", "run_segment_id", unique=True,
              postgresql_where=text(ATTEMPT_SELECTED_WHERE),
              sqlite_where=text(ATTEMPT_SELECTED_WHERE)),
        Index("ix_rebuild_chunk_attempts_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    run_segment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("run_segments.id", ondelete="CASCADE"), nullable=False,
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    # Frozen from the run's rebuild_snapshot at admission.
    model: Mapped[str] = mapped_column(String(32), nullable=False)
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)
    request_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    # runs/<rid>/rebuild/attempts/<id>/raw.mp4, fitted.mp4 — immutable once written.
    raw_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fitted_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    qa: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RebuildLlmCall(_PaidCallColumns, Base):
    """One Gemini call of a storyboard or of an attempt's character check."""

    __tablename__ = "rebuild_llm_calls"
    __table_args__ = (
        CheckConstraint(PAID_CALL_STATUS_CHECK, name="ck_rebuild_llm_calls_status"),
        CheckConstraint(_in("kind", LLM_KINDS), name="ck_rebuild_llm_calls_kind"),
        # Exactly one owner.
        CheckConstraint(
            "(storyboard_id IS NULL) <> (attempt_id IS NULL)",
            name="ck_rebuild_llm_calls_owner",
        ),
        CheckConstraint(
            "kind <> 'shot' OR (shot_id IS NOT NULL AND global_call_id IS NOT NULL)",
            name="ck_rebuild_llm_calls_shot",
        ),
        CheckConstraint(
            "kind <> 'character_check' OR (attempt_id IS NOT NULL AND slot IS NOT NULL)",
            name="ck_rebuild_llm_calls_check",
        ),
        # At most one non-terminal call per (global context, shot) and one
        # non-terminal global call per project (design §5.2): admission relies
        # on these, so a duplicate is refused by the database.
        Index("uq_rebuild_llm_shot_live", "global_call_id", "shot_id", unique=True,
              postgresql_where=text(LLM_SHOT_LIVE_WHERE),
              sqlite_where=text(LLM_SHOT_LIVE_WHERE)),
        Index("uq_rebuild_llm_global_live", "project_id", unique=True,
              postgresql_where=text(LLM_GLOBAL_LIVE_WHERE),
              sqlite_where=text(LLM_GLOBAL_LIVE_WHERE)),
        # Character check: one live call per attempt, and each slot (0 = the
        # automatic primary, 1 = its one fallback, >= 2 = an operator's "Run
        # character check") used once.
        Index("uq_rebuild_llm_check_live", "attempt_id", unique=True,
              postgresql_where=text(LLM_CHECK_LIVE_WHERE),
              sqlite_where=text(LLM_CHECK_LIVE_WHERE)),
        Index("uq_rebuild_llm_check_slot", "attempt_id", "slot", unique=True,
              postgresql_where=text(LLM_CHECK_SLOT_WHERE),
              sqlite_where=text(LLM_CHECK_SLOT_WHERE)),
        Index("ix_rebuild_llm_calls_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False,
        index=True,
    )
    storyboard_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("project_storyboards.id", ondelete="CASCADE"), nullable=True,
        index=True,
    )
    attempt_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("rebuild_chunk_attempts.id", ondelete="CASCADE"), nullable=True,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # Soft reference: shot rows are replaced by a re-analysis, the call record
    # (a possibly billed request) is not.
    shot_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    slot: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    # Shot calls: the global call they were made under (soft reference).
    global_call_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    response: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# Every paid-call table, for the reaper, the drain check and the live-row guards.
PAID_CALL_MODELS = (RebuildKeyframe, RebuildChunkAttempt, RebuildLlmCall)
