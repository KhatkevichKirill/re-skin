"""ORM models of the reframe workflow (AICP-251, docs/reframe-design.md §4).

* ``reframe_jobs`` — one paid generation attempt on a ``reframe`` project:
  the window, target aspect and resolution, the locally prepared input and the
  pinned audio snapshot. Its own status covers only what the provider layer
  does not: ``preparing`` → ``admitted`` (or ``prepare_failed``). Once admitted
  its generation state IS its provider task's state (``provider_task_id``, a
  soft reference, no FK), so nothing here mirrors the task's statuses.
* ``reframe_renders`` — one free, repeatable finish of a job's raw output: the
  ending, an optional CTA clip, and Drive delivery.

Times are naive UTC, like ``provider_models``. Imported at the end of
``models.py`` so ``Base.metadata`` and Alembic see the tables.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class ReframeJobStatus(str, enum.Enum):
    preparing = "preparing"
    prepare_failed = "prepare_failed"
    admitted = "admitted"


class ReframeQAStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    passed = "passed"
    failed = "failed"


class ReframeRenderStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    delivering = "delivering"
    done = "done"
    failed = "failed"


REFRAME_RENDER_ACTIVE_STATUSES = frozenset(
    {ReframeRenderStatus.queued, ReframeRenderStatus.processing, ReframeRenderStatus.delivering}
)

JOB_STATUS_CHECK = "status IN ('preparing', 'prepare_failed', 'admitted')"
QA_STATUS_CHECK = "qa_status IN ('pending', 'running', 'passed', 'failed')"
RENDER_STATUS_CHECK = "status IN ('queued', 'processing', 'delivering', 'done', 'failed')"
PREPARING_WHERE = "status = 'preparing'"


class ReframeJob(Base):
    __tablename__ = "reframe_jobs"
    __table_args__ = (
        CheckConstraint(JOB_STATUS_CHECK, name="ck_reframe_jobs_status"),
        CheckConstraint(QA_STATUS_CHECK, name="ck_reframe_jobs_qa_status"),
        CheckConstraint("target_aspect IN ('9:16', '16:9')", name="ck_reframe_jobs_aspect"),
        CheckConstraint("resolution IN ('480p', '768p')", name="ck_reframe_jobs_resolution"),
        # One job being prepared per project: two quick clicks cannot both
        # reach admission (design §4, R5).
        Index(
            "uq_reframe_jobs_preparing_per_project", "project_id", unique=True,
            postgresql_where=text(PREPARING_WHERE), sqlite_where=text(PREPARING_WHERE),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="preparing")
    target_aspect: Mapped[str] = mapped_column(String(8), nullable=False)
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)

    # The window as requested, on the source timeline.
    window_start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    window_end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    # What was actually cut: the PTS of the first source frame at or after
    # the window start, and the frame count of input.mp4 (design §5).
    input_start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    input_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    input_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    input_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    input_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # The pinned source-audio snapshot the renders use (NULL path when the
    # source has no audio).
    audio_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audio_start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_has_audio: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Display only: output_sec x tariff, and the layer's worst case (x2 overlap).
    estimated_usd_base: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_usd_reserved: Mapped[float] = mapped_column(Float, nullable=False)

    # Soft reference to provider_tasks.id, set in the admission transaction.
    provider_task_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)

    # prepare_failed only; generation errors live on the task.
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    prep_claim_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    prep_claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    prep_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )

    qa_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default=text("'pending'")
    )
    job_qa: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    qa_claim_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    qa_claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())

    def __repr__(self) -> str:
        return f"<ReframeJob id={self.id!r} {self.status} qa={self.qa_status}>"


class ReframeRender(Base):
    __tablename__ = "reframe_renders"
    __table_args__ = (
        CheckConstraint(RENDER_STATUS_CHECK, name="ck_reframe_renders_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("reframe_jobs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    # Token of the execution that owns `processing`/`delivering`; every later
    # status change is a CAS on it, so a superseded worker (reaped, then the
    # render retried) can never deliver or overwrite the newer one's files.
    claim_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # The ending as requested, on the source timeline (design §6).
    picture_end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    fade_sec: Mapped[float] = mapped_column(Float, nullable=False)
    audio_end_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    black_tail_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Optional CTA clip: soft reference + the render-owned snapshot of its file.
    library_clip_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    cta_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")

    result_local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    result_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    result_duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    result_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    qa: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    gdrive_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    gdrive_link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())

    def __repr__(self) -> str:
        return f"<ReframeRender id={self.id!r} job={self.job_id!r} {self.status}>"
