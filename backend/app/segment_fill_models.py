"""ORM model of the segment caption fill (AICP-358, docs/segment-caption-fill-design.md).

``run_segment_fill_jobs`` — one attempt at rebuilding a swap segment of a done
``subtitle_removal`` run as its source frames + a Fal VOID inpaint pasted back
only inside the caption mask. Append-only history, not a slot.

``preparing`` → ``ready`` is free (mask + preview); ``ready`` → ``admitted`` is
the operator's approved paid start (a provider task, soft reference, no FK:
provider rows outlive runs by design); ``admitted`` → ``compositing`` →
``applied`` puts the composite in as the segment's result. ``failed`` and
``canceled`` are terminal. At most one ``admitted``/``compositing`` row per run
(the *active* fill), which every other result-mutating path refuses against.

Times are naive UTC, like ``provider_models``. Imported at the end of
``models.py`` so ``Base.metadata`` and Alembic see the table.
"""

from __future__ import annotations

import enum
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
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class FillStatus(str, enum.Enum):
    preparing = "preparing"
    ready = "ready"
    admitted = "admitted"
    compositing = "compositing"
    # A batch member's composite is built and verified; the batch promote
    # applies it (docs/segment-fill-batch-design.md §1.4). Never a legacy row.
    composited = "composited"
    applied = "applied"
    failed = "failed"
    canceled = "canceled"


FILL_ACTIVE = (FillStatus.admitted.value, FillStatus.compositing.value,
               FillStatus.composited.value)
FILL_TERMINAL = (FillStatus.applied.value, FillStatus.failed.value, FillStatus.canceled.value)
# One active LEGACY (unbatched) fill per run: rows admitted before the batch
# deploy, and source-basis single fills (design R1-1). Batched members are
# governed by the batch's own index.
_ACTIVE_SQL = "status IN ('admitted','compositing') AND batch_id IS NULL"
_STATUS_SQL = ("status IN ('preparing','ready','admitted','compositing','composited','applied',"
               "'failed','canceled')")


class BatchStatus(str, enum.Enum):
    active = "active"
    promoting = "promoting"
    applied = "applied"
    failed = "failed"


BATCH_LIVE = (BatchStatus.active.value, BatchStatus.promoting.value)
_BATCH_LIVE_SQL = "status IN ('active','promoting')"
_BATCH_STATUS_SQL = "status IN ('active','promoting','applied','failed')"


class RunSegmentFillJob(Base):
    __tablename__ = "run_segment_fill_jobs"
    __table_args__ = (
        CheckConstraint(_STATUS_SQL, name="ck_run_segment_fill_jobs_status"),
        CheckConstraint("model IN ('void')", name="ck_run_segment_fill_jobs_model"),
        # Spelled out by hand here and in the migration; sqlite_where mirrors
        # postgresql_where for tests built on Base.metadata.create_all().
        Index(
            "uq_run_segment_fill_jobs_active_per_run",
            "run_id",
            unique=True,
            postgresql_where=text(_ACTIVE_SQL),
            sqlite_where=text(_ACTIVE_SQL),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_segment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("run_segments.id", ondelete="CASCADE"), nullable=False,
        index=True,
    )
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now(), nullable=False
    )

    status: Mapped[str] = mapped_column(String(16), nullable=False,
                                        default=FillStatus.preparing.value, index=True)
    model: Mapped[str] = mapped_column(String(16), nullable=False, default="void")
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Normalised [x0, y0, x1, y1]: the request's box, or (after prepare) the
    # measured one that was actually used.
    mask_box: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    box_source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # What the fill replaces, frozen at preview and compared under the locks at
    # start and at promote (design §4).
    base: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Clip geometry + the VOID sampling map, written by prepare.
    media: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    prep_dir: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    preview_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    estimated_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    approved_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    discard_audio_patch: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    provider_task_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True,
                                                            index=True)

    # One claim for whichever phase owns the row (prepare in ``preparing``,
    # composite in ``compositing``); attempts count per phase.
    claim_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0,
                                          server_default=text("0"))

    # Batch fills (2026-11-03): the batch this member was admitted in (NULL =
    # a legacy single fill); an extra dilation of the glyph mask in pixels; the
    # composite fill this job retries with a grown mask (the stop rule's one
    # automatic retry, design §4.3).
    batch_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("run_segment_fill_batches.id", ondelete="SET NULL"),
        nullable=True, index=True)
    mask_grow_px: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    retry_of: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    composite_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    replaced_take_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    applied_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    files_cleaned_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (f"<RunSegmentFillJob id={self.id!r} run_id={self.run_id!r} "
                f"status={self.status!r}>")


class RunSegmentFillBatch(Base):
    """One paid start over several ready previews of a run (2026-11-03,
    docs/segment-fill-batch-design.md): the members run concurrently at Fal,
    each composite is built on its own, ONE promote applies every successful
    composite with one re-stitch and one in-place re-delivery. At most one
    ``active`` / ``promoting`` batch per run."""

    __tablename__ = "run_segment_fill_batches"
    __table_args__ = (
        CheckConstraint(_BATCH_STATUS_SQL, name="ck_run_segment_fill_batches_status"),
        CheckConstraint("authorization_kind IN ('operator','auto')",
                        name="ck_run_segment_fill_batches_authorization"),
        Index(
            "uq_run_segment_fill_batches_live_per_run",
            "run_id",
            unique=True,
            postgresql_where=text(_BATCH_LIVE_SQL),
            sqlite_where=text(_BATCH_LIVE_SQL),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow, server_default=func.now(), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False,
                                        default=BatchStatus.active.value, index=True)
    basis: Mapped[str] = mapped_column(String(16), nullable=False, default="model_result")
    # "operator" (a person approved this amount) or "auto" (inside the run's
    # VOID allowance, video_projects.void_auto_usd_per_run; design §3).
    authorization_kind: Mapped[str] = mapped_column(String(16), nullable=False,
                                                    default="operator")
    approved_usd: Mapped[float] = mapped_column(Float, nullable=False)
    # What the members' provider tasks reserved, in millicredits (the ledger's unit).
    reserved_mcredits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reserved_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    discard_audio_patch: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    # The run as it was at the ADMISSION: the delivery receipt's baseline.
    base: Mapped[dict] = mapped_column(JSON, nullable=False)
    claim_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0,
                                          server_default=text("0"))
    applied_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (f"<RunSegmentFillBatch id={self.id!r} run_id={self.run_id!r} "
                f"status={self.status!r}>")
