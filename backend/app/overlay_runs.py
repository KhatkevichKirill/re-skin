"""
overlay_runs.py — an overlay job's relationship to the Run it was burned from.

An :class:`OverlayJob`'s source is a *soft* reference: a Run's result
(``source_run_id``), or the output of one of that run's derived creatives —
a tail edit (``source_tail_edit_id``) or a caption job
(``source_caption_id``) — each of which belongs to exactly one Run. Two
consumers need that link resolved the same way:

* the run-detail page's "Overlays" card (``web_v2``), which lists every overlay
  burned onto a run — directly, or through one of that run's tail edits; and
* the Final Creatives export (``tasks.run_export_overlay_to_sheet``), whose
  ``Job ID`` cell is built from the parent run.

Keeping both here means the id the operator reads off the run page is
byte-for-byte the one the spreadsheet receives.
"""

from __future__ import annotations

from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from . import languages
from .models import OverlayJob, Run, RunCaption, RunTailEdit


def resolve_parent_run_id(
    *,
    source_type: str,
    source_run_id: str | None,
    source_tail_edit_id: str | None,
    source_caption_id: str | None,
    db: Session,
) -> str | None:
    """The run id to stamp on a new OverlayJob's ``parent_run_id``.

    Called once, at create/fork time, from the ids the request supplied — by
    which point the source row is known to exist and be done. A derived
    creative always names its parent run in its own ``run_id`` (never the
    video it read as input — see RunCaption's docstring), so one hop is
    always enough. Returns None for an upload source, which has no parent.
    """
    if source_type == "run":
        return source_run_id
    if source_type == "tail_edit" and source_tail_edit_id:
        edit = db.get(RunTailEdit, source_tail_edit_id)
        return edit.run_id if edit else None
    if source_type == "caption" and source_caption_id:
        caption = db.get(RunCaption, source_caption_id)
        return caption.run_id if caption else None
    return None


def _source_filter(run_id: str):
    """SQL predicate: overlay jobs that are *run_id*'s creative.

    ``parent_run_id`` answers this on its own for anything created since
    migration ``d7c9e1a3f582`` (which also backfilled every older row). The
    legacy source-walk is kept as an OR so a job created in the window between
    that migration and the code deploy — the only way to get a NULL
    ``parent_run_id`` with a real source — is still found.
    """
    tail_edit_ids = (
        select(RunTailEdit.id).where(RunTailEdit.run_id == run_id).scalar_subquery()
    )
    caption_ids = (
        select(RunCaption.id).where(RunCaption.run_id == run_id).scalar_subquery()
    )
    return or_(
        OverlayJob.parent_run_id == run_id,
        OverlayJob.source_run_id == run_id,
        OverlayJob.source_tail_edit_id.in_(tail_edit_ids),
        OverlayJob.source_caption_id.in_(caption_ids),
    )


def parent_run(job: OverlayJob, db: Session) -> Run | None:
    """The Run *job* is a creative of, or None for an upload source.

    Reads the stamped ``parent_run_id`` first: unlike the soft references it
    does not dead-end when the tail edit or caption job the overlay was burned
    from is deleted. The walk below is the fallback for a row that predates
    the stamp (see :func:`_source_filter`).
    """
    if job.parent_run_id:
        return db.get(Run, job.parent_run_id)
    if job.source_run_id:
        return db.get(Run, job.source_run_id)
    if job.source_tail_edit_id:
        edit = db.get(RunTailEdit, job.source_tail_edit_id)
        return edit.run if edit else None
    if job.source_caption_id:
        caption = db.get(RunCaption, job.source_caption_id)
        return caption.run if caption else None
    return None


def overlay_jobs_for_run(run_id: str, db: Session) -> list[OverlayJob]:
    """Live (not soft-deleted) overlay jobs of *run_id*, newest first."""
    return list(
        db.execute(
            select(OverlayJob)
            .where(_source_filter(run_id), OverlayJob.deleted_at.is_(None))
            .order_by(desc(OverlayJob.created_at))
        )
        .scalars()
        .all()
    )


def count_overlay_jobs_for_run(run_id: str, db: Session) -> int:
    """How many overlay jobs a run has ever had — soft-deleted ones INCLUDED.

    This is the input to :func:`sheet_job_id`'s "one overlay or several?"
    decision, and counting deleted jobs is what keeps that decision monotonic:
    a run that once had two overlays never falls back to the single-overlay
    (bare run id) shape, so a newly exported overlay can never collide with the
    bare-id row a since-deleted sibling already wrote to the sheet.
    """
    return int(
        db.execute(
            select(func.count(OverlayJob.id)).where(_source_filter(run_id))
        ).scalar_one()
    )


def sheet_job_id(job: OverlayJob, run: Run | None, *, sibling_count: int) -> str:
    """The Final Creatives ``Job ID`` cell for *job*.

    An overlay is the parent run's creative re-dressed, so it goes to the sheet
    under the PARENT RUN's id (``1736426f-ja``) rather than an id of its own —
    the row stays traceable to the run that produced the footage. What keeps it
    from being confused with the run's own row is the ``Pipeline Type`` cell
    (``re-skin-overlay`` vs ``re-skin-v2``), which the downstream renamer puts
    in the filename: ``…__re-skin-overlay-1736426f.mp4`` vs
    ``…__re-skin-v2-1736426f.mp4``. See app/languages.py for that contract.

    That only works while a run has ONE overlay. Once it has several, the ids
    would collide with each other, so every overlay of such a run exports as
    ``{run prefix}_{overlay prefix}`` plus the run's language suffix —
    ``1736426f_a1b2c3d4-ja``. *sibling_count* is
    :func:`count_overlay_jobs_for_run` for the parent run (pass it in: callers
    listing a whole page of rows want one query, not one per row).

    An upload-sourced overlay has no parent to inherit from and keeps its own
    8-char prefix, unsuffixed — exactly what it exported as before.
    """
    if run is None:
        return languages.job_id_for(job.id, None)
    if sibling_count <= 1:
        return run.short_id
    return languages.apply_suffix(f"{run.id[:8]}_{job.id[:8]}", run.language)
