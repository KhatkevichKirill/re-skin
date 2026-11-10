"""Durable job records of the localisation side channels (AICP-365).

* ``project_transcriptions`` — one row per admitted transcription of a project's
  source (every origin: the operator's button, the after-analysis hand-off, the
  autopilot). At most one is live (``queued`` / ``running``) per project.
* ``localisation_prompt_jobs`` — one row per translate job admitted by the
  autopilot (the operator's Translate button keeps its Redis-only job).

Both share one dispatch protocol (app/localisation_jobs.py, docs/autopilot-
localisation-flow.md §3.1): the database row is authoritative, RQ is only the
doorbell. ``dispatch_gen`` is bumped on every (re-)enqueue and is part of the RQ
job id, a worker claims only ``status='queued' AND dispatch_gen=<its gen>``,
and every later write of the worker requires ``status='running' AND
claim_token=<its token>``. The reaper re-enqueues a ``queued`` row whose job is
not live and interrupts a ``running`` row whose job is not live (revoking the
claim). An interrupted call may have been billed: nothing resubmits it.

Imported at the end of ``models.py`` so ``Base.metadata`` and Alembic see them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, CheckConstraint, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .models_autopilot import _new_uuid, utcnow

JOB_STATUSES = ("queued", "running", "succeeded", "failed", "interrupted")
LIVE_JOB_STATUSES = ("queued", "running")
TRANSCRIPTION_ORIGINS = ("manual", "analysis", "autopilot")


def _in(col: str, values) -> str:
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


_LIVE = "status IN ('queued', 'running')"


class _DispatchColumns:
    """Columns of the shared dispatch protocol (mixin)."""

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    dispatch_gen: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    claim_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rq_job_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    not_live_since: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    updated_at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)


class ProjectTranscription(_DispatchColumns, Base):
    __tablename__ = "project_transcriptions"
    __table_args__ = (
        CheckConstraint(_in("status", JOB_STATUSES), name="ck_project_transcriptions_status"),
        CheckConstraint(_in("origin", TRANSCRIPTION_ORIGINS),
                        name="ck_project_transcriptions_origin"),
        Index("uq_project_transcriptions_live", "project_id", unique=True,
              postgresql_where=text(_LIVE), sqlite_where=text(_LIVE)),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False,
        index=True)
    origin: Mapped[str] = mapped_column(String(16), nullable=False, default="manual")


class LocalisationPromptJob(_DispatchColumns, Base):
    __tablename__ = "localisation_prompt_jobs"
    __table_args__ = (
        CheckConstraint(_in("status", JOB_STATUSES), name="ck_localisation_prompt_jobs_status"),
        Index("ix_localisation_prompt_jobs_project", "project_id", "language"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_projects.id", ondelete="CASCADE"), nullable=False)
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    request: Mapped[dict] = mapped_column(JSON, nullable=False)
    # The frozen inputs (api_v2._prepare_localisation_snapshot) and their
    # project fingerprint (localisation_prompt_job.project_fingerprint).
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    # LocalisationPromptResponse as JSON once succeeded; {"status_code", "detail"} on failure.
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
