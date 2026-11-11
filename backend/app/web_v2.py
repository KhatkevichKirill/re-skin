"""
v2 Web UI router — server-rendered Jinja2 + HTMX pages.

Mounted at /v2 from app/main.py (the primary operator UI; `/` redirects here).
Provides: Projects dashboard (with a run-parameter filter + kie.ai credit
balance), Generations dashboard (latest per-segment generation videos across
all runs), Project detail (segment editor + runs panel), Run detail (progress
polling, per-task cost, result preview).
"""

from __future__ import annotations

import csv
import io
import logging
import hashlib
import os
import threading

from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import and_, desc, exists, or_, select
from sqlalchemy.orm import Session

from . import ai_models
from . import drive_sa
from . import languages
from . import localisation
from . import overlay_runs
from .api_v2 import _MAX_BATCH_COPY_RUNS, _backfill_result_duration, _default_segment_image_prompt
from .automation import _TERMINAL_RUN_STATUSES, launched_today
from .config import settings
from .db import get_db
from .models import (
    AutomationLaunch,
    InsertExtraction,
    InsertMerge,
    LibraryClip,
    VoiceClip,
    OverlayAsset,
    OverlayJob,
    Run,
    RunAudioPatch,
    RunAutomation,
    RunCaption,
    RunSegment,
    RunSegmentImageJob,
    RunSegmentRangeAttempt,
    RunSegmentTake,
    RunTailEdit,
    SegmentDef,
    VideoProject,
)
from .pipeline_common import TAIL_EDIT_MIN_CUT_MARGIN_SEC, floor2 as _floor2
from sqlalchemy.orm import joinedload, selectinload
from . import project_types as project_types_registry
from .project_types import (
    GENERATION_IMAGES_ONLY,
    selectable_project_types,
    DEFAULT_PROJECT_TYPE,
    LOCALISATION,
    LOCALISATION_SCOPE_HOOK,
    LOCALISATION_SCOPE_WHOLE,
    PROJECT_TYPES,
    label_for,
    run_copies_are_useful,
    spec_for,
)
from .public import make_result_token, make_source_token
from .sheets_client import neutralise_formula
from .storage import staged_frame_time_label
from .state_machine import (
    AudioPatchStatus,
    CaptionStatus,
    MergeStatus,
    OverlayStatus,
    ProjectStatus,
    RANGE_ATTEMPT_ACTIVE_STATUSES,
    RunStatus,
    SegmentImageStatus,
    SegmentStatus,
    TailEditStatus,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["web_v2"])

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates", "v2")
# Search path also covers templates/ (parent) so `{% include "site_header.html" %}`
# in base.html resolves to the one shared header partial used by every area
# (v2 + overlay) — see web_overlay.py, which adds the same fallback dir.
_SHARED_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=[_TEMPLATES_DIR, _SHARED_TEMPLATES_DIR])
# `drive_sa_email()` for the shared "share this with the service account" notice
# (templates/_drive_sa_notice.html, AICP-274).
drive_sa.register(templates.env)

# Canonical model display names come from the AI model registry
# (app/ai_models.py), which is the single source of truth for everything that
# varies per backend. Exposed to every v2 template via the `model_label`
# filter, and used by the Python-side row builders (e.g. the Generations CSV
# export) that need the same mapping outside Jinja. Unknown/NULL values are
# returned unchanged so a row written before a model was added still renders.
templates.env.filters["model_label"] = ai_models.label_for

# The compact form of the same tag ("seedance-fast" -> "SDf2"), for tables
# where a full label would crowd out the columns that carry the information.
# Deliberately the SAME value the Final Creatives `Idea` column exports, so a
# tag read off the runs table matches what the sheet received — see
# ai_models.short_label_for and sheets_client.build_idea.
templates.env.filters["model_short_label"] = ai_models.short_label_for

# Same idea for project types ("face_swap" -> "Face changing"); the registry in
# app/project_types.py is the single source of truth for the labels.
templates.env.filters["project_type_label"] = label_for

# And for run languages ("ja" -> "Japanese"). NULL renders as "English", which
# is what an unset language actually means to the downstream renaming pipeline
# — see app/languages.py.
templates.env.filters["language_label"] = languages.label_for

# The Job ID affix that same language writes ("ja" -> "-ja", English/NULL ->
# ""), for templates that spell out the shape of an id rather than a real one.
templates.env.filters["language_suffix"] = languages.suffix_for

# A reference image staged off the source video ("Add frame from source video")
# carries the timestamp it was cut at in its filename, and nowhere else — this
# filter reads it back ("…/frame_124_ab12cd34.jpg" -> "12.4s") so the Reference
# images block can say WHERE in the video each trim came from. Uploads and
# outside URLs have no timestamp to show and render as nothing.
templates.env.filters["trim_time"] = staged_frame_time_label


# "Change ending"'s duration hint and its "Max" button both need a floor —
# never Jinja's own `round(2, 'floor')`, which has the float-precision gap
# pipeline_common.floor2's docstring describes. Same implementation
# `create_tail_edit`'s 400 detail uses, so the UI and the error message can
# never name a different bound.
templates.env.filters["floor2"] = _floor2


def _autopilot_project_types() -> list[str]:
    """Project types an assistant project can be created for (AICP-363):
    none unless AUTOPILOT_ENABLED and a flow is registered for the type."""
    if not settings.AUTOPILOT_ENABLED:
        return []
    from .autopilot import registry as ap_registry

    return sorted(ap_registry.project_types_with_flow())


templates.env.globals["autopilot_project_types"] = _autopilot_project_types


def _autopilot_languages() -> list[dict]:
    """The languages a localisation autopilot can target, in registry order
    (AICP-378 mode picker)."""
    from . import languages as _langs

    return [{"code": k, "label": v.label} for k, v in _langs.LANGUAGES.items()]


def _autopilot_notify_on() -> bool:
    """Telegram notifications are configured (AICP-374), for the mode picker's
    "what happens next" note."""
    from .autopilot import notify

    return notify.config_problem() is None


templates.env.globals["autopilot_languages"] = _autopilot_languages
templates.env.globals["autopilot_notify_on"] = _autopilot_notify_on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_project_or_404(project_id: str, db: Session) -> VideoProject:
    project = db.get(VideoProject, project_id)
    if project is None:
        raise HTTPException(
            status_code=404, detail=f"Project {project_id!r} not found"
        )
    return project


def _get_run_or_404(run_id: str, db: Session) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return run


def _gdrive_link_for_run(run: Run) -> str | None:
    if run.result_gdrive_file_id:
        return f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
    return None


def _result_version_for_run(run: Run) -> str:
    """Return a stable cache-buster for the current on-disk result file."""
    path = run.result_local_path
    if path and os.path.exists(path):
        stat = os.stat(path)
        identity = f"{path}\0{stat.st_mtime_ns}\0{stat.st_size}"
        return hashlib.sha256(identity.encode()).hexdigest()[:24]
    updated = run.updated_at
    if updated is not None:
        return str(int(updated.timestamp()))
    return "0"


def _segment_clip_url(rs: RunSegment) -> str | None:
    """Link to this segment's generated clip, or None when it isn't on disk.

    The URL carries a mtime-size cache buster for the same reason the final's
    does — a segment re-run rewrites ``result_<index>.mp4`` in place, so a
    stable URL would show the operator the clip they just replaced. Served by
    api_v2.download_run_segment_result.
    """
    path = rs.local_result_path
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (
        f"/api/v2/runs/{rs.run_id}/segments/{rs.id}/result"
        f"?v={int(stat.st_mtime)}-{stat.st_size}"
    )


def _get_segments(project_id: str, db: Session) -> list[SegmentDef]:
    return (
        db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == project_id)
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    )


def _get_runs(project_id: str, db: Session, *, extra_conds=()) -> list[Run]:
    """This project's runs, newest first.

    ``extra_conds`` are the dashboard's run-level filter conditions
    (:func:`_run_filter_conditions`), so an expanded pivot lists exactly the
    runs that made the project match — pass nothing for "every run".
    """
    stmt = select(Run).where(Run.project_id == project_id, *extra_conds)
    return db.execute(stmt.order_by(desc(Run.created_at))).scalars().all()


# The kinds of video a run can put into the "Final Creatives" sheet, in the
# order the project's Existing-runs table lists them: the run's own stitched
# result, a tail edit of it, a burned-captions copy, an overlay burn. Each
# lives in its own table with its own `final_creatives_exported_at` stamp;
# `_final_creatives_sent` folds all four into one per-run answer.
_FINAL_CREATIVE_KINDS: tuple[tuple[str, str], ...] = (
    ("original", "The run's own stitched result was logged to Final Creatives"),
    ("tail", "A tail edit (changed ending) of this run was logged to Final Creatives"),
    ("captioned", "A burned-captions copy of this run was logged to Final Creatives"),
    ("overlay", "An overlay burn of this run was logged to Final Creatives"),
)


def _final_creatives_sent(runs: list[Run], db: Session) -> dict[str, list[dict]]:
    """Which kinds of video each run has already sent to the Final Creatives sheet.

    Returns ``{run_id: [{"kind": ..., "title": ..., "count": n}, ...]}`` in
    ``_FINAL_CREATIVE_KINDS`` order, with the entry omitted entirely for a kind
    that was never exported (so an empty list means "nothing sent from this run
    yet"). The operator reads the runs table for exactly one thing: has anything
    from this run reached the sheet, and what was it?

    Four queries total, regardless of how many runs the project has — this runs
    on every project-page render, so the obvious per-run loop is not an option.
    """
    run_ids = [r.id for r in runs]
    if not run_ids:
        return {}

    counts: dict[str, dict[str, int]] = {rid: {} for rid in run_ids}

    # The run's own result.
    for run in runs:
        if run.final_creatives_exported_at is not None:
            counts[run.id]["original"] = 1

    # Derived creatives that name their parent run directly. The id -> run_id
    # maps are also what resolves a legacy overlay's soft source reference
    # below, so select the ids even though the count only needs the run_id.
    tail_run: dict[str, str] = {}
    for edit_id, run_id, exported in db.execute(
        select(
            RunTailEdit.id, RunTailEdit.run_id, RunTailEdit.final_creatives_exported_at
        ).where(RunTailEdit.run_id.in_(run_ids))
    ).all():
        tail_run[edit_id] = run_id
        if exported is not None:
            counts[run_id]["tail"] = counts[run_id].get("tail", 0) + 1

    caption_run: dict[str, str] = {}
    for caption_id, run_id, exported in db.execute(
        select(
            RunCaption.id, RunCaption.run_id, RunCaption.final_creatives_exported_at
        ).where(RunCaption.run_id.in_(run_ids))
    ).all():
        caption_run[caption_id] = run_id
        if exported is not None:
            counts[run_id]["captioned"] = counts[run_id].get("captioned", 0) + 1

    # Overlays reach their run through `parent_run_id` (stamped at create time)
    # or, for a row predating that stamp, through one of the soft source
    # references — the same OR that overlay_runs._source_filter walks, batched
    # across every run on the page. Soft-deleted jobs are excluded: their result
    # is gone from disk, so claiming the run still has one on the sheet would
    # point at nothing.
    overlay_rows = db.execute(
        select(
            OverlayJob.parent_run_id,
            OverlayJob.source_run_id,
            OverlayJob.source_tail_edit_id,
            OverlayJob.source_caption_id,
        ).where(
            OverlayJob.deleted_at.is_(None),
            OverlayJob.final_creatives_exported_at.isnot(None),
            or_(
                OverlayJob.parent_run_id.in_(run_ids),
                OverlayJob.source_run_id.in_(run_ids),
                OverlayJob.source_tail_edit_id.in_(list(tail_run)),
                OverlayJob.source_caption_id.in_(list(caption_run)),
            ),
        )
    ).all()
    for parent_id, src_run_id, src_tail_id, src_caption_id in overlay_rows:
        # First candidate that names a run on this page wins — an overlay
        # matched by its source reference may carry a parent_run_id pointing
        # somewhere we are not rendering.
        for candidate in (
            parent_id,
            src_run_id,
            tail_run.get(src_tail_id),
            caption_run.get(src_caption_id),
        ):
            if candidate in counts:
                counts[candidate]["overlay"] = counts[candidate].get("overlay", 0) + 1
                break

    return {
        run_id: [
            {"kind": kind, "title": title, "count": kind_counts[kind]}
            for kind, title in _FINAL_CREATIVE_KINDS
            if kind in kind_counts
        ]
        for run_id, kind_counts in counts.items()
    }


def _get_source_run(project: VideoProject, db: Session) -> Run | None:
    """The run this project was created from ("Move to new project"), if any.

    ``source_run_id`` is a soft reference, so a deleted run simply yields None
    and the backlink banner is omitted.
    """
    if not project.source_run_id:
        return None
    return db.get(Run, project.source_run_id)


_SOURCE_STEP_LABELS = {"caption": "captions", "tail_edit": "tail edit", "overlay": "overlay"}


def _get_source_step(project: VideoProject, db: Session) -> dict | None:
    """The run step this project's video was copied from (AICP-331), if any.

    ``{"kind", "label", "id", "short_id", "url"}`` with ``url`` None when the
    step no longer exists — captions and tail edits are hard-deleted, overlays
    soft-deleted — or no longer belongs to the project's ``source_run_id``
    (both are soft references, so the pair is checked rather than trusted);
    the banner then names the step without a link. None for a project made
    from a run's final video or from anything but a run.
    """
    kind, sid = project.source_kind, project.source_artifact_id
    if kind not in _SOURCE_STEP_LABELS or not sid:
        return None
    url = None
    if kind == "caption":
        row = db.get(RunCaption, sid)
        if row is not None and row.run_id == project.source_run_id:
            url = f"/v2/runs/{row.run_id}#caption-{sid}"
    elif kind == "tail_edit":
        row = db.get(RunTailEdit, sid)
        if row is not None and row.run_id == project.source_run_id:
            url = f"/v2/runs/{row.run_id}#tail-edit-{sid}"
    else:
        row = db.get(OverlayJob, sid)
        if row is not None and row.deleted_at is None:
            owner_id = row.parent_run_id
            if owner_id is None:
                owner = overlay_runs.parent_run(row, db)
                owner_id = owner.id if owner is not None else None
            if owner_id is not None and owner_id == project.source_run_id:
                url = f"/v2/overlays/{sid}"
    return {"kind": kind, "label": _SOURCE_STEP_LABELS[kind], "id": sid,
            "short_id": sid[:8], "url": url}


def _get_insert_extraction_context(project_id: str, db: Session) -> dict:
    """PIP-insert provenance for a project page: the "extracted from" banner
    (when *project_id* is a child) and the list of extracted children (when
    *project_id* is a parent). See docs/pip-insert.md.
    """
    parent_extraction = db.execute(
        select(InsertExtraction).where(
            InsertExtraction.child_project_id == project_id
        )
    ).scalar_one_or_none()
    extracted_from = None
    if parent_extraction is not None:
        extracted_from = db.get(VideoProject, parent_extraction.parent_project_id)

    child_extractions = (
        db.execute(
            select(InsertExtraction)
            .where(InsertExtraction.parent_project_id == project_id)
            .order_by(InsertExtraction.created_at)
        )
        .scalars()
        .all()
    )
    child_inserts = []
    for ext in child_extractions:
        child_proj = db.get(VideoProject, ext.child_project_id)
        if child_proj is not None:
            child_inserts.append({"extraction": ext, "project": child_proj})

    return {
        "extracted_from": extracted_from,
        "child_inserts": child_inserts,
    }


# Statuses an InsertMerge can still be actively worked on in — used both to
# decide whether the merges-list fragment keeps polling and (implicitly) to
# gate the retry button in the template (shown only for "failed").
_ACTIVE_MERGE_STATUSES = {
    MergeStatus.queued,
    MergeStatus.compositing,
    MergeStatus.delivering,
}


def _get_insert_merge_context(run: Run, db: Session) -> dict:
    """PIP-insert "merge back" context for a run-detail page.

    Returns the run project's ``InsertExtraction`` (None unless this run's
    project is an extracted PIP-insert child) plus a list of candidate target
    projects (any other probed project) for the merge-back form's target
    dropdown. See docs/pip-insert.md.

    The merge-back card is only rendered by the template for a ``done`` run
    (nested inside ``{% if sv == "done" %}`` in run_status_content.html), so
    skip the queries entirely otherwise — this is called on every 2s status
    poll while a run is in progress.
    """
    if run.status != RunStatus.done:
        return {"insert_extraction": None, "merge_targets": []}

    extraction = db.execute(
        select(InsertExtraction).where(
            InsertExtraction.child_project_id == run.project_id
        )
    ).scalar_one_or_none()
    if extraction is None:
        return {"insert_extraction": None, "merge_targets": []}

    targets = (
        db.execute(
            select(VideoProject)
            .where(VideoProject.width.isnot(None))
            .where(VideoProject.id != run.project_id)
            .order_by(desc(VideoProject.created_at))
            .limit(100)
        )
        .scalars()
        .all()
    )
    return {"insert_extraction": extraction, "merge_targets": targets}


def _get_merges(project_id: str, db: Session) -> list[InsertMerge]:
    """InsertMerges whose target is *project_id*, newest first."""
    return (
        db.execute(
            select(InsertMerge)
            .where(InsertMerge.target_project_id == project_id)
            .order_by(desc(InsertMerge.created_at))
        )
        .scalars()
        .all()
    )


def _merges_still_active(merges: list[InsertMerge]) -> bool:
    return any(m.status in _ACTIVE_MERGE_STATUSES for m in merges)


def _get_library_clips(db: Session) -> list[LibraryClip]:
    """Newest 200 stitch-library clips, newest first (for the run form + library page)."""
    return list(
        db.execute(select(LibraryClip).order_by(desc(LibraryClip.created_at)).limit(200))
        .scalars()
        .all()
    )


def _get_overlay_library_assets(db: Session) -> list[OverlayAsset]:
    """Newest 200 overlay-card library assets, newest first (for the library page)."""
    return list(
        db.execute(select(OverlayAsset).order_by(desc(OverlayAsset.created_at)).limit(200))
        .scalars()
        .all()
    )


def _get_voice_clips(db: Session) -> list[VoiceClip]:
    """Newest 200 library voices, newest first (for the run form + library page)."""
    return list(
        db.execute(select(VoiceClip).order_by(desc(VoiceClip.created_at)).limit(200))
        .scalars()
        .all()
    )


# Statuses a RunTailEdit can still be actively worked on in — used both to
# decide whether the tail-edits-list fragment keeps polling and to gate the
# retry button in the template (shown only for "failed").
_ACTIVE_TAIL_EDIT_STATUSES = {
    TailEditStatus.queued,
    TailEditStatus.processing,
    TailEditStatus.delivering,
}


def _get_tail_edits(run_id: str, db: Session) -> list[RunTailEdit]:
    """RunTailEdits for *run_id*, newest first."""
    return list(
        db.execute(
            select(RunTailEdit)
            .where(RunTailEdit.run_id == run_id)
            .order_by(desc(RunTailEdit.created_at))
        )
        .scalars()
        .all()
    )


def _tail_edits_still_active(edits: list[RunTailEdit]) -> bool:
    return any(e.status in _ACTIVE_TAIL_EDIT_STATUSES for e in edits)


def _tail_clip_names(edits: list[RunTailEdit], db: Session) -> dict[str, str]:
    """``{library_clip_id: clip.name}`` for the clips referenced by *edits*.

    ``library_clip_id`` is a soft reference (no FK/relationship — mirrors
    Run.stitch_plan), so a clip may have been deleted since the edit was
    created; such ids are simply absent from the returned dict.
    """
    clip_ids = {e.library_clip_id for e in edits if e.library_clip_id}
    if not clip_ids:
        return {}
    return {
        c.id: c.name
        for c in db.execute(
            select(LibraryClip).where(LibraryClip.id.in_(clip_ids))
        ).scalars()
    }


def _last_segment_start_sec(project_id: str, db: Session) -> float | None:
    """Start time of the highest-index SegmentDef for *project_id*.

    Used to prefill the tail-edit "Change ending" form's cut_sec: "cut where
    the old outro starts". Returns None if the project has no segments.
    """
    seg = db.execute(
        select(SegmentDef)
        .where(SegmentDef.project_id == project_id)
        .order_by(desc(SegmentDef.index))
        .limit(1)
    ).scalar_one_or_none()
    return seg.start_sec if seg is not None else None


def _tail_edit_context(
    *,
    library_clips: list,
    cut_prefill: float | None,
    result_duration_sec: float | None,
    edits: list,
    edits_active: bool,
    clip_names: dict,
    source_caption_options: dict,
    source_caption_durations: dict,
    source_labels: dict,
) -> dict:
    """Build the "Change ending" card context dict.

    Single source of truth for the shape so the early-return (non-done) path
    and the populated (done) path can never drift on the keys the templates
    consume (run_status_content.html / run_tail_edits_list.html).

    ``source_caption_options`` / ``source_labels`` are docs/burned-captions.md
    §3.7.4's chaining additions: the former is ``{caption_id: label}`` for
    every *done* caption of this run (the "Change ending" form's source
    picker); the latter is ``{caption_id: label}`` for every caption actually
    referenced by one of *edits* as its source, so a row can resolve its own
    ``source_caption_id`` to a label (or "(deleted)" when absent) without a
    per-row query. ``source_caption_durations`` is ``{caption_id: duration_sec}``
    for the same picker (see ``_caption_durations``) — a caption can be
    shorter than its run (it may itself be built on a tail edit), so the
    "Cut at (s)" field's max must follow whichever source is selected, not
    always the run's.

    ``tail_edit_min_cut_margin_sec`` is NOT a parameter: it's a fixed constant
    (``pipeline_common.TAIL_EDIT_MIN_CUT_MARGIN_SEC``), not something that
    varies per run, so every caller gets the same value automatically instead
    of having to thread it through.
    """
    return {
        "tail_edit_library_clips": library_clips,
        "tail_edit_cut_prefill": cut_prefill,
        "tail_edit_result_duration_sec": result_duration_sec,
        "tail_edit_min_cut_margin_sec": TAIL_EDIT_MIN_CUT_MARGIN_SEC,
        "tail_edits": edits,
        "tail_edits_active": edits_active,
        "tail_clip_names": clip_names,
        "tail_edit_source_caption_options": source_caption_options,
        "tail_edit_source_caption_durations": source_caption_durations,
        "tail_edit_source_labels": source_labels,
    }


def _caption_durations(captions: list[RunCaption]) -> dict[str, float]:
    """``{caption_id: duration_sec}`` for *captions* that have a probeable
    file on disk — a caption with no file, or one ffprobe can't read, is
    simply absent, and the front end then falls back to no upper bound for
    that source rather than a wrong one (same tolerance the run's own probe
    below already has).

    Probed on every call, not cached: unlike ``Run.result_duration_sec``,
    ``RunCaption`` has no duration column (this task ships without a
    migration), and there are typically 0-1 done captions per run — measured
    ffprobe overhead on this box is ~0.1-0.2s, dominated by process-spawn, not
    file size (ffprobe reads container metadata only, it never decodes
    frames). That is a one-off render-time cost, not a per-poll one: this
    helper is only reached from ``_get_tail_edit_context``, which itself only
    runs for a ``done`` run — the status where the 2s poll has already
    stopped (see that function's docstring). Chosen over a dedicated
    "duration on demand" endpoint because the picker then needs no round trip
    to recompute the "Cut at (s)" max when the operator switches source.
    """
    from . import media as media_mod

    durations: dict[str, float] = {}
    for c in captions:
        if not c.output_path or not os.path.exists(c.output_path):
            continue
        try:
            durations[c.id] = media_mod.probe(c.output_path).duration_sec
        except Exception as exc:  # ffprobe failure — degrade gracefully, same
            # tolerance as the run's own probe in _get_tail_edit_context.
            log.warning(
                "tail-edit context: probe failed for caption %s: %s", c.id, exc
            )
    return durations


def _get_tail_edit_context(run: Run, db: Session) -> dict:
    """Context for the run-detail page's "Change ending" card (only shown for
    a done run — the template gates on status). Includes the library clip
    picker options, the result video's duration (for display + the number
    input's max), the "cut where the old outro starts" prefill, and the
    current list of tail edits for the initial (non-fragment) render.

    The card is only rendered by the template for a ``done`` run (nested
    inside ``{% if sv == "done" %}`` in run_status_content.html), so skip all
    of this — including the ffprobe subprocess below — otherwise. This is
    called on every 2s status poll while a run is in progress.
    """
    if run.status != RunStatus.done:
        return _tail_edit_context(
            library_clips=[], cut_prefill=None, result_duration_sec=None,
            edits=[], edits_active=False, clip_names={},
            source_caption_options={}, source_caption_durations={},
            source_labels={},
        )

    edits = _get_tail_edits(run.id, db)
    if run.result_duration_sec is not None:
        # Cached on the row at finalize — no ffprobe on the render path.
        result_duration_sec = run.result_duration_sec
    else:
        # Back-compat: runs finalized before result_duration_sec existed.
        # Probe once, then write the value back to the row so this run never
        # re-probes on future views (self-healing backfill).
        result_duration_sec = None
        probed_path = run.result_local_path
        if probed_path and os.path.exists(probed_path):
            try:
                from . import media as media_mod

                result_duration_sec = media_mod.probe(probed_path).duration_sec
            except Exception as exc:  # ffprobe failure — degrade gracefully
                # Expected: media_mod.MediaError / OSError. Kept broad so a
                # render never 500s over a duration lookup.
                log.warning("tail-edit context: probe failed for run %s: %s", run.id, exc)
        if result_duration_sec is not None:
            result_duration_sec = _backfill_result_duration(run, db, probed_path, result_duration_sec)

    done_captions = _get_done_captions(run.id, db)
    return _tail_edit_context(
        library_clips=_get_library_clips(db),
        cut_prefill=_last_segment_start_sec(run.project_id, db),
        result_duration_sec=result_duration_sec,
        edits=edits,
        edits_active=_tail_edits_still_active(edits),
        clip_names=_tail_clip_names(edits, db),
        source_caption_options=_caption_source_labels(done_captions),
        source_caption_durations=_caption_durations(done_captions),
        source_labels=_referenced_caption_labels(edits, db),
    )


# ---------------------------------------------------------------------------
# "Fix audio from another run" card (app/pipeline_audio_patch.py)
# ---------------------------------------------------------------------------

_ACTIVE_AUDIO_PATCH_STATUSES = {
    AudioPatchStatus.queued,
    AudioPatchStatus.processing,
    AudioPatchStatus.delivering,
}


def _get_audio_patches(run_id: str, db: Session) -> list[RunAudioPatch]:
    """RunAudioPatches for *run_id*, newest first."""
    return list(
        db.execute(
            select(RunAudioPatch)
            .where(RunAudioPatch.run_id == run_id)
            .order_by(desc(RunAudioPatch.created_at))
        )
        .scalars()
        .all()
    )


def _audio_patches_still_active(patches: list[RunAudioPatch]) -> bool:
    return any(p.status in _ACTIVE_AUDIO_PATCH_STATUSES for p in patches)


def _audio_patch_donor_runs(run: Run, db: Session) -> list[Run]:
    """The runs offered as audio donors for *run*: its done SIBLINGS.

    Same project only, and deliberately so — a donor is useful exactly when it
    speaks the same script over the same driving footage, which is what makes
    its phrasing land in the same place. A run from another project would
    almost always be the wrong words, and offering every done run in the
    database would bury the handful that are right.
    """
    if run.project_id is None:
        return []
    return list(
        db.execute(
            select(Run)
            .where(
                Run.project_id == run.project_id,
                Run.id != run.id,
                Run.status == RunStatus.done,
            )
            .order_by(desc(Run.created_at))
        )
        .scalars()
        .all()
    )


def _run_label(run: Run) -> str:
    """``"RplDcTop · 2714a7c3"`` — the name an operator recognises plus the id
    they see everywhere else. Falls back to the id alone for an unnamed run."""
    return f"{run.name} · {run.short_id}" if run.name else run.short_id


def _audio_patch_donor_labels(
    patches: list[RunAudioPatch], db: Session
) -> dict[str, str]:
    """``{donor_run_id: label}`` for the donors *patches* actually used.

    ``donor_run_id`` is a soft reference, so a donor deleted since simply is
    not in the dict and the row falls back to the bare id — the same way
    ``_tail_clip_names`` degrades for a deleted library clip.
    """
    donor_ids = {p.donor_run_id for p in patches if p.donor_run_id}
    if not donor_ids:
        return {}
    return {
        r.id: _run_label(r)
        for r in db.execute(select(Run).where(Run.id.in_(donor_ids))).scalars().all()
    }


def _audio_patch_context(
    *,
    patches: list[RunAudioPatch],
    patches_active: bool,
    donor_runs: list[Run],
    donor_labels: dict[str, str],
    result_duration_sec: float | None,
) -> dict:
    """Single source of truth for the audio-patch card's context keys, so the
    full page, the polling fragment and the list-only fragment cannot drift
    (same reasoning as ``_tail_edit_context``)."""
    from . import media as media_mod

    return {
        "audio_patches": patches,
        "audio_patches_active": patches_active,
        "audio_patch_donor_runs": donor_runs,
        "audio_patch_donor_labels": donor_labels,
        "audio_patch_result_duration_sec": result_duration_sec,
        "audio_patch_default_crossfade_sec": media_mod.AUDIO_SPLICE_DEFAULT_CROSSFADE_SEC,
        "audio_patch_min_window_sec": media_mod.AUDIO_SPLICE_MIN_WINDOW_SEC,
    }


def _get_audio_patch_context(run: Run, db: Session) -> dict:
    """Context for the run-detail page's "Fix audio from another run" card.

    Only a done run has a delivered final to patch, and the template gates the
    card on that, so everything here is skipped otherwise — this runs on every
    2s status poll while a run is in progress.

    ``result_duration_sec`` is read straight off the row rather than probed:
    ``_get_tail_edit_context`` already backfills it for a pre-column run on the
    same render, and a missing value here only means the window fields have no
    upper bound in the browser — the API still rejects an impossible window.
    """
    if run.status != RunStatus.done:
        return _audio_patch_context(
            patches=[], patches_active=False, donor_runs=[], donor_labels={},
            result_duration_sec=None,
        )
    patches = _get_audio_patches(run.id, db)
    return _audio_patch_context(
        patches=patches,
        patches_active=_audio_patches_still_active(patches),
        donor_runs=_audio_patch_donor_runs(run, db),
        donor_labels=_audio_patch_donor_labels(patches, db),
        result_duration_sec=run.result_duration_sec,
    )


# ---------------------------------------------------------------------------
# Chaining derived creatives (docs/burned-captions.md §3.7.4) — a tail edit
# may source from a done caption, and a caption may source from a done tail
# edit. Both directions need the same two things: a picker's worth of "done"
# options for the create form, and a way for an existing row to resolve
# whichever one it actually used back to a label — "(deleted)" when the
# soft-referenced source no longer exists, exactly as `_tail_clip_names`
# already tolerates a deleted library clip.
# ---------------------------------------------------------------------------


def _format_caption_source_label(c: RunCaption) -> str:
    """"{short id} ({created})" — label for a caption offered/used as a tail
    edit's source. Short id because the full uuid is unreadable in a
    dropdown; created-at because a run can have several done captions and
    that is the only thing that tells them apart at a glance."""
    created = f"{c.created_at:%Y-%m-%d %H:%M}" if c.created_at else "?"
    return f"{c.id[:8]} ({created})"


def _caption_source_labels(captions: list[RunCaption]) -> dict[str, str]:
    """``{caption_id: label}`` for *captions*, order preserved (callers pass
    an already-ordered list, e.g. newest-first from `_get_done_captions`)."""
    return {c.id: _format_caption_source_label(c) for c in captions}


def _get_done_captions(run_id: str, db: Session) -> list[RunCaption]:
    """Done RunCaptions for *run_id* — the only ones with an output file a
    tail edit could actually read, so the only ones offered as the "Change
    ending" form's source options."""
    return list(
        db.execute(
            select(RunCaption)
            .where(RunCaption.run_id == run_id, RunCaption.status == CaptionStatus.done)
            .order_by(desc(RunCaption.created_at))
        )
        .scalars()
        .all()
    )


def _referenced_caption_labels(edits: list[RunTailEdit], db: Session) -> dict[str, str]:
    """``{caption_id: label}`` for the captions referenced by *edits* as their
    source (``source_type == "caption"``). Soft reference (no FK) — an id
    whose caption has since been deleted is simply absent from the dict, so
    the template renders "(deleted)" instead of erroring."""
    caption_ids = {e.source_caption_id for e in edits if e.source_caption_id}
    if not caption_ids:
        return {}
    captions = list(
        db.execute(select(RunCaption).where(RunCaption.id.in_(caption_ids))).scalars()
    )
    return _caption_source_labels(captions)


def _format_tail_edit_source_label(e: RunTailEdit, clip_names: dict[str, str]) -> str:
    """"Cut @12.3s" (+ the appended clip's name when there is one) — label for
    a tail edit offered/used as a caption's source."""
    label = f"Cut @{e.cut_sec:.1f}s"
    if e.library_clip_id:
        label += f" + {clip_names.get(e.library_clip_id, '(deleted)')}"
    return label


def _tail_edit_source_labels(edits: list[RunTailEdit], db: Session) -> dict[str, str]:
    """``{tail_edit_id: label}`` for *edits*, order preserved."""
    clip_names = _tail_clip_names(edits, db)
    return {e.id: _format_tail_edit_source_label(e, clip_names) for e in edits}


def _get_done_tail_edits(run_id: str, db: Session) -> list[RunTailEdit]:
    """Done RunTailEdits for *run_id* — the only ones with an output file a
    caption job could actually read, so the only ones offered as the
    captions control's source options."""
    return list(
        db.execute(
            select(RunTailEdit)
            .where(RunTailEdit.run_id == run_id, RunTailEdit.status == TailEditStatus.done)
            .order_by(desc(RunTailEdit.created_at))
        )
        .scalars()
        .all()
    )


def _referenced_tail_edit_labels(captions: list[RunCaption], db: Session) -> dict[str, str]:
    """``{tail_edit_id: label}`` for the tail edits referenced by *captions*
    as their source (``source_type == "tail_edit"``). Soft reference (no FK)
    — an id whose tail edit has since been deleted is simply absent from the
    dict, so the template renders "(deleted)" instead of erroring."""
    tail_edit_ids = {c.source_tail_edit_id for c in captions if c.source_tail_edit_id}
    if not tail_edit_ids:
        return {}
    edits = list(
        db.execute(select(RunTailEdit).where(RunTailEdit.id.in_(tail_edit_ids))).scalars()
    )
    return _tail_edit_source_labels(edits, db)


# Statuses a RunCaption can still be actively worked on in — used both to
# decide whether the captions-list fragment keeps polling and to gate the
# retry button in the template (shown only for "failed").
_ACTIVE_CAPTION_STATUSES = {
    CaptionStatus.queued,
    CaptionStatus.transcribing,
    CaptionStatus.burning,
    CaptionStatus.delivering,
}


def _get_captions(run_id: str, db: Session) -> list[RunCaption]:
    """RunCaptions for *run_id*, newest first."""
    return list(
        db.execute(
            select(RunCaption)
            .where(RunCaption.run_id == run_id)
            .order_by(desc(RunCaption.created_at))
        )
        .scalars()
        .all()
    )


def _captions_still_active(captions: list[RunCaption]) -> bool:
    return any(c.status in _ACTIVE_CAPTION_STATUSES for c in captions)


def _caption_context(
    *,
    captions: list,
    captions_active: bool,
    run_id: str,
    source_tail_edit_options: dict,
    source_labels: dict,
    project_placement: dict | None = None,
) -> dict:
    """Build the "Captions" card context dict.

    Single source of truth for the shape so the early-return (non-done) path
    and the populated (done) path can never drift on the keys the templates
    consume (run_status_content.html / run_captions_list.html).

    ``source_tail_edit_options`` / ``source_labels`` mirror
    `_tail_edit_context`'s caption-side keys: the former is
    ``{tail_edit_id: label}`` for every *done* tail edit of this run (the
    captions control's source picker); the latter is ``{tail_edit_id: label}``
    for every tail edit actually referenced by one of *captions* as its
    source, so a row can resolve its own ``source_tail_edit_id`` (or
    "(deleted)") without a per-row query.
    """
    return {
        "captions": captions,
        "captions_active": captions_active,
        "run_id": run_id,
        "caption_source_tail_edit_options": source_tail_edit_options,
        "caption_source_labels": source_labels,
        # AICP-368: the project's band placement for the captions control's
        # position picker (None = house position only).
        "caption_project_placement": project_placement,
    }


def _caption_placement_summary(project) -> dict | None:
    """What the run page's caption position picker shows for *project*'s
    placement: None unless it is a band placement."""
    from .caption_placement import PlacementError, validate

    try:
        placement = validate(getattr(project, "caption_placement", None)
                             if project is not None else None)
    except PlacementError:
        return None  # a hand-edited row: the burn ignores it too
    if placement is None or placement.mode != "band" or placement.center_y is None:
        return None
    source = placement.source or {}
    return {
        "center_y": placement.center_y,
        "kind": source.get("kind"),
        "provisional": (source.get("scan") or source.get("origin_scan")) == "pending",
        "scan_failed": (source.get("scan") or source.get("origin_scan")) == "failed",
        "warnings": list(source.get("warnings") or []),
    }


def _get_caption_context(run: Run, db: Session) -> dict:
    """Context for the run-detail page's "Captions" card (only shown for a
    done run — the template gates on status).

    The card is only rendered by the template for a ``done`` run (nested
    inside ``{% if sv == "done" %}`` in run_status_content.html), so skip the
    query below otherwise. This is called on every 2s status poll while a run
    is in progress, same as `_get_tail_edit_context`.
    """
    if run.status != RunStatus.done:
        return _caption_context(
            captions=[], captions_active=False, run_id=run.id,
            source_tail_edit_options={}, source_labels={},
        )

    captions = _get_captions(run.id, db)
    return _caption_context(
        captions=captions,
        captions_active=_captions_still_active(captions),
        run_id=run.id,
        source_tail_edit_options=_tail_edit_source_labels(
            _get_done_tail_edits(run.id, db), db
        ),
        source_labels=_referenced_tail_edit_labels(captions, db),
        project_placement=_caption_placement_summary(db.get(VideoProject, run.project_id)),
    )


# ---------------------------------------------------------------------------
# Overlays on a run — the run page's own view of the overlay tool
# ---------------------------------------------------------------------------
# The overlay editor itself stays where it is (/v2/overlays/{job_id}, served by
# web_overlay.py): positioning PNG cards needs the drag surface. What the run
# page adds is the same shape the tail edits have — a control that starts one
# against this run (or one of its tail edits) and a history list with the
# Drive / Final Creatives actions — so an operator never has to go hunting
# through the global overlay dashboard for a run they are looking at.


# Statuses an overlay job can still be actively worked on in — drives the
# history fragment's self-polling (and gates Delete, which the API rejects
# while a job is active), exactly as _ACTIVE_TAIL_EDIT_STATUSES does for tail
# edits.
_ACTIVE_OVERLAY_STATUSES = {
    OverlayStatus.queued,
    OverlayStatus.processing,
    OverlayStatus.delivering,
}


def _overlays_still_active(jobs: list[OverlayJob]) -> bool:
    return any(j.status in _ACTIVE_OVERLAY_STATUSES for j in jobs)


def _overlay_row(
    job: OverlayJob, *, source_label: str, job_id_cell: str
) -> dict:
    """One row of the run page's overlay history.

    A dict rather than the ORM object (unlike the tail-edit list) because two
    of the columns — the source label and the sheet Job ID — are resolved once
    per page instead of per row.
    """
    return {
        "id": job.id,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "source_label": source_label,
        "sheet_job_id": job_id_cell,
        "n_overlays": len(job.overlays or []),
        "created_at": job.created_at,
        "gdrive_file_id": job.gdrive_file_id,
        "has_result": bool(job.result_local_path),
        "exported": job.final_creatives_exported_at is not None,
        "error": job.error,
    }


def _overlay_rows(run: Run, jobs: list[OverlayJob], db: Session) -> list[dict]:
    """Build the overlay history rows for *run* from its overlay *jobs*."""
    # Derived-creative sources label with the same strings the tail-edit and
    # caption pickers use ("Cut @12.3s + Outro", "a1b2c3d4 (2026-08-11 09:12)").
    tail_edit_ids = {j.source_tail_edit_id for j in jobs if j.source_tail_edit_id}
    tail_labels: dict[str, str] = {}
    if tail_edit_ids:
        edits = list(
            db.execute(
                select(RunTailEdit).where(RunTailEdit.id.in_(tail_edit_ids))
            ).scalars()
        )
        tail_labels = _tail_edit_source_labels(edits, db)
    caption_ids = {j.source_caption_id for j in jobs if j.source_caption_id}
    caption_labels: dict[str, str] = {}
    if caption_ids:
        caps = list(
            db.execute(
                select(RunCaption).where(RunCaption.id.in_(caption_ids))
            ).scalars()
        )
        caption_labels = _caption_source_labels(caps)
    # One count for the whole page — every row's Job ID depends on the same
    # "does this run have several overlays?" answer (overlay_runs.sheet_job_id).
    sibling_count = overlay_runs.count_overlay_jobs_for_run(run.id, db)
    rows = []
    for job in jobs:
        if job.source_tail_edit_id:
            source_label = (
                f"Tail edit: {tail_labels.get(job.source_tail_edit_id, '(deleted)')}"
            )
        elif job.source_caption_id:
            source_label = (
                f"Caption: {caption_labels.get(job.source_caption_id, '(deleted)')}"
            )
        else:
            source_label = "Run result"
        rows.append(
            _overlay_row(
                job,
                source_label=source_label,
                job_id_cell=overlay_runs.sheet_job_id(
                    job, run, sibling_count=sibling_count
                ),
            )
        )
    return rows


def _overlay_context(
    *,
    overlay_jobs: list,
    overlays_active: bool,
    run_id: str,
    source_tail_edit_options: dict,
    source_caption_options: dict,
) -> dict:
    """Build the "Overlays" card context dict.

    Single source of truth for the shape, so the early-return (non-done) path,
    the full page and the history fragment can never drift on the keys
    run_status_content.html / run_overlays_list.html consume — same pattern as
    `_tail_edit_context` / `_caption_context`.

    The two option dicts are the card's source picker: every *done* tail edit
    and every *done* caption job of this run, each of which OverlayJob can
    take as a source alongside the run's own result.
    """
    return {
        "overlay_jobs": overlay_jobs,
        "overlays_active": overlays_active,
        "run_id": run_id,
        "overlay_source_tail_edit_options": source_tail_edit_options,
        "overlay_source_caption_options": source_caption_options,
    }


def _get_overlay_context(run: Run, db: Session) -> dict:
    """Context for the run-detail page's "Overlays" card + history list.

    Only rendered for a ``done`` run (the template gates on status), so skip
    the queries otherwise — this is called on every 2s status poll while a run
    is in progress, same as `_get_tail_edit_context`.
    """
    if run.status != RunStatus.done:
        return _overlay_context(
            overlay_jobs=[], overlays_active=False, run_id=run.id,
            source_tail_edit_options={}, source_caption_options={},
        )

    jobs = overlay_runs.overlay_jobs_for_run(run.id, db)
    return _overlay_context(
        overlay_jobs=_overlay_rows(run, jobs, db),
        overlays_active=_overlays_still_active(jobs),
        run_id=run.id,
        # An overlay can be burned onto a tail edit's or a caption job's output
        # as well as the run's own result (OverlayJob.source_type), so the
        # picker offers all three — the *done* ones are exactly those with a
        # file on disk.
        source_tail_edit_options=_tail_edit_source_labels(
            _get_done_tail_edits(run.id, db), db
        ),
        source_caption_options=_caption_source_labels(
            _get_done_captions(run.id, db)
        ),
    )


def _stitch_plan_summary(run: Run, db: Session) -> str | None:
    """Human-readable one-liner for a run's custom stitch plan (None = default)."""
    plan = run.stitch_plan
    if plan is None:
        return None
    if not plan:
        return "Stitching disabled — segments are processed but no final video is produced."
    parts: list[str] = []
    for entry in plan:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") == "segment":
            sd = db.get(SegmentDef, entry.get("segment_def_id"))
            parts.append(f"segment {sd.index}" if sd else "segment (deleted)")
        elif entry.get("kind") == "library":
            clip = db.get(LibraryClip, entry.get("clip_id"))
            parts.append(
                f"library clip “{clip.name}”" if clip else "library clip (deleted)"
            )
    return "Custom stitch plan: " + " → ".join(parts)


# ---------------------------------------------------------------------------
# Launch windows
#
# A RunSegment points at a live SegmentDef, but SegmentDefs are mutable and
# shared across runs: a later edit re-cuts them, so reading start/end off the
# def makes an old run's breakdown silently re-label itself with boundaries
# that run never touched. Run fdc49979 displayed "34.6s – 56.1s" for a swap it
# had actually generated at 8.625s – 17.25s, because a mid-run segment edit had
# since repurposed that def as the project's keep segment.
#
# Every operator-facing view of a RunSegment therefore reads the run's own
# frozen Run.segment_snapshot. Runs created before that column existed fall
# back to the live def — the old behaviour, and still the best answer for them.
# ---------------------------------------------------------------------------


def _launch_window(run: Run | None, rs: RunSegment) -> tuple[float | None, float | None]:
    """The (start_sec, end_sec) *rs* was generated against, per the above.

    Returns (None, None) when neither the snapshot nor a live def can supply
    them, which every caller renders as no range at all rather than guessing.
    """
    entry = None
    if run is not None:
        for e in run.segment_snapshot or []:
            if isinstance(e, dict) and e.get("id") == rs.segment_def_id:
                entry = e
                break
    if entry is None:
        entry = rs.segment_def  # pre-snapshot run, or def already deleted
    if entry is None:
        return None, None

    def bound(field: str) -> float | None:
        value = (
            entry.get(field) if isinstance(entry, dict)
            else getattr(entry, field, None)
        )
        return None if value is None else float(value)

    return bound("start_sec"), bound("end_sec")


_ACTIVE_SEGMENT_IMAGE_STATUSES = {
    SegmentImageStatus.queued, SegmentImageStatus.generating,
}

_ACTIVE_RANGE_ATTEMPT_STATUS_VALUES = {s.value for s in RANGE_ATTEMPT_ACTIVE_STATUSES}


def _range_edit_bounds(run: Run, rs: RunSegment) -> dict:
    """UI hints for the range-editor panel: which of this swap segment's two
    boundaries can move, and the span each may move across.

    Pure presentation — the authoritative check is
    pipeline_range_attempt.compute_candidate_snapshot, run again server-side
    on submit. ``editable`` is False when neither boundary has a `keep`
    neighbour (both sides are `swap`, or this run predates run-local
    snapshots) — the range editor is hidden entirely in that case.
    """
    snapshot = sorted(run.segment_snapshot or [], key=lambda e: e["index"])
    pos = next(
        (i for i, e in enumerate(snapshot) if e.get("id") == rs.segment_def_id), None,
    )
    if pos is None:
        return {"editable": False}
    target = snapshot[pos]
    prev_e = snapshot[pos - 1] if pos > 0 else None
    next_e = snapshot[pos + 1] if pos < len(snapshot) - 1 else None
    prev_keep = prev_e is not None and prev_e.get("action") == "keep"
    next_keep = next_e is not None and next_e.get("action") == "keep"
    return {
        "editable": prev_keep or next_keep,
        "start_locked": not prev_keep,
        "start_min": float(prev_e["start_sec"]) if prev_keep else float(target["start_sec"]),
        "start_max": float(target["end_sec"]),
        "end_locked": not next_keep,
        "end_min": float(target["start_sec"]),
        "end_max": float(next_e["end_sec"]) if next_keep else float(target["end_sec"]),
    }


def _range_attempts_by_rs(
    run_id: str, db: Session
) -> dict[str, list[RunSegmentRangeAttempt]]:
    """``{run_segment_id: [attempt, ...]}`` for *run_id*, each list newest
    first — the per-segment range-editor's attempt history."""
    rows = (
        db.execute(
            select(RunSegmentRangeAttempt)
            .where(RunSegmentRangeAttempt.run_id == run_id)
            .order_by(desc(RunSegmentRangeAttempt.created_at))
        )
        .scalars()
        .all()
    )
    by_rs: dict[str, list[RunSegmentRangeAttempt]] = {}
    for row in rows:
        by_rs.setdefault(row.run_segment_id, []).append(row)
    return by_rs


def _segment_image_jobs_by_rs(
    run_id: str, db: Session
) -> dict[str, list[RunSegmentImageJob]]:
    """``{run_segment_id: [job, ...]}`` for *run_id*, each list newest first —
    the per-segment "Generated image" block's attempt history."""
    jobs = (
        db.execute(
            select(RunSegmentImageJob)
            .where(RunSegmentImageJob.run_id == run_id)
            .order_by(desc(RunSegmentImageJob.created_at))
        )
        .scalars()
        .all()
    )
    by_rs: dict[str, list[RunSegmentImageJob]] = {}
    for job in jobs:
        by_rs.setdefault(job.run_segment_id, []).append(job)
    return by_rs


def _rebuild_run_context(run: Run, db: Session) -> dict:
    """Images-only (rebuild) run: the per-chunk card (app/rebuild_run_view.py).
    ``{"rebuild_chunks": None}`` for every v2v run."""
    from .rebuild_run_view import run_context

    return run_context(run, db)


def _get_run_segments(run: Run, db: Session) -> list[RunSegment]:
    """This run's segments, each stamped with the window it actually ran on.

    ``window_start`` / ``window_end`` are plain instance attributes for the
    template — not mapped columns, so nothing is persisted. See _launch_window.
    ``clip_url`` is stamped the same way: the link to this segment's own
    generated clip, None while nothing is on disk (see _segment_clip_url).
    ``image_jobs`` / ``image_jobs_active`` are the per-segment GPT Image 2
    regeneration attempt history and whether one of them is still running
    (queued/generating) — see _segment_image_jobs_by_rs.
    """

    segments = (
        db.execute(
            select(RunSegment)
            .where(RunSegment.run_id == run.id)
            .order_by(RunSegment.index)
            .options(selectinload(RunSegment.segment_def))
        )
        .scalars()
        .all()
    )
    jobs_by_rs = _segment_image_jobs_by_rs(run.id, db)
    range_attempts_by_rs = _range_attempts_by_rs(run.id, db)
    takes_by_rs: dict[str, list] = {}
    for take in db.execute(
        select(RunSegmentTake)
        .where(RunSegmentTake.run_id == run.id)
        .order_by(RunSegmentTake.created_at.desc())
    ).scalars():
        takes_by_rs.setdefault(take.run_segment_id, []).append(take)
    # AICP-358: segment caption fills, newest first (subtitle_removal runs).
    from .segment_fill_models import RunSegmentFillJob
    from .project_types import SUBTITLE_REMOVAL

    _proj = db.get(VideoProject, run.project_id)
    fill_allowed = _proj is not None and _proj.project_type == SUBTITLE_REMOVAL
    fills_by_rs: dict[str, list] = {}
    if fill_allowed:
        for fj in db.execute(
            select(RunSegmentFillJob).where(RunSegmentFillJob.run_id == run.id)
            .order_by(RunSegmentFillJob.created_at.desc())
        ).scalars():
            fills_by_rs.setdefault(fj.run_segment_id, []).append(fj)
    from .content_block import LABELS as _BLOCK_LABELS, segment_block

    _leftover_rsids = _latest_qa_leftovers(run, db) if fill_allowed else set()
    for rs in segments:
        rs.fill_allowed = fill_allowed
        rs.fill_busy = any(fj.status in ("preparing", "admitted", "compositing", "composited")
                           for fj in fills_by_rs.get(rs.id, []))
        rs.fill_jobs = fills_by_rs.get(rs.id, [])[:3]
        # Batch VOID fills (2026-11-03): the newest ready model-result preview
        # of this segment (what "Start N ready" takes) and its reservation.
        _void_run = getattr(run, "generation_mode", "v2v") == "void_source"
        _rs_st = rs.status.value if hasattr(rs.status, "value") else str(rs.status)
        # A VOID-from-the-source run (design §7): a segment still without a
        # result gets its fill from the source (the run page offers it).
        rs.void_source_pending = bool(fill_allowed and _void_run and not (
            _rs_st == "completed" and not rs.source_fallback))
        _want = "source" if rs.void_source_pending else "model_result"
        _ready = next((fj for fj in fills_by_rs.get(rs.id, [])
                       if fj.status == "ready" and (fj.base or {}).get("basis", "model_result")
                       == _want and not (fj.media or {}).get("source_as_is")), None)
        rs.fill_ready_id = _ready.id if _ready is not None else None
        rs.fill_ready_usd = float(_ready.estimated_usd or 0.0) if _ready is not None else None
        rs.fill_leftover = rs.id in _leftover_rsids
        # A segment the provider refused with a content filter: the run page
        # offers a VOID fill from the SOURCE (AICP-358 follow-up).
        rs.content_block = segment_block(rs) if fill_allowed else None
        rs.content_block_label = _BLOCK_LABELS.get(rs.content_block or "")
        # AICP-342: preserved previous outputs, newest first. is_current is a
        # plain attribute for the template (restored takes are current).
        rs.takes = takes_by_rs.get(rs.id, [])
        for take in rs.takes:
            take.is_current = rs.local_result_path == take.result_path
        rs.window_start, rs.window_end = _launch_window(run, rs)
        rs.clip_url = _segment_clip_url(rs)
        rs.image_jobs = jobs_by_rs.get(rs.id, [])
        rs.image_jobs_active = any(
            j.status in _ACTIVE_SEGMENT_IMAGE_STATUSES for j in rs.image_jobs
        )
        rs.range_edit = _range_edit_bounds(run, rs)
        rs.range_attempts = range_attempts_by_rs.get(rs.id, [])
        rs.range_attempts_active = any(
            a.status.value in _ACTIVE_RANGE_ATTEMPT_STATUS_VALUES
            for a in rs.range_attempts
        )
    return segments


def _latest_qa_leftovers(run: Run, db: Session) -> set:
    """Run segments the latest caption QA of this run (the autopilot's
    ``subtitle_qa`` compute, when one exists) found leftovers in: what the run
    page's "Select segments with leftovers" picks. Empty when there is none."""
    try:
        from .models_autopilot import AutopilotEffect, ProjectAutopilot

        rows = db.execute(
            select(AutopilotEffect.result)
            .join(ProjectAutopilot, ProjectAutopilot.id == AutopilotEffect.autopilot_id)
            .where(ProjectAutopilot.project_id == run.project_id,
                   AutopilotEffect.kind == "compute", AutopilotEffect.handler == "subtitle_qa",
                   AutopilotEffect.state == "succeeded")
            .order_by(AutopilotEffect.created_at.desc()).limit(5)).scalars().all()
    except Exception:  # noqa: BLE001 -- a convenience only
        return set()
    mine = {rs.id for rs in run.run_segments}
    for res in rows:
        segs = ((res or {}).get("qa") or {}).get("segments") or []
        if any(r.get("run_segment_id") in mine for r in segs):     # a QA of THIS run
            return {r["run_segment_id"] for r in segs
                    if r.get("verdict") == "leftover" and r.get("run_segment_id") in mine}
    return set()


def _fill_batches_view(run: Run, db: Session) -> list:
    """The run's newest VOID fill batches for the run page (2026-11-03)."""
    from .segment_fill_models import RunSegmentFillBatch, RunSegmentFillJob

    out = []
    for b in db.execute(select(RunSegmentFillBatch).where(RunSegmentFillBatch.run_id == run.id)
                        .order_by(RunSegmentFillBatch.created_at.desc()).limit(3)).scalars():
        members = db.execute(select(RunSegmentFillJob).where(
            RunSegmentFillJob.batch_id == b.id)).scalars().all()
        out.append({"id": b.id, "status": b.status, "authorization": b.authorization_kind,
                    "reserved_usd": b.reserved_usd, "error": b.error_message,
                    "members": [(m.run_segment_id, m.status, m.error_code) for m in members]})
    return out


def _segment_image_jobs_active(run_segments: list[RunSegment]) -> bool:
    """True while any segment of this run has an in-flight GPT Image 2 job.

    Drives the block's self-poll (mirrors ``_tail_edits_still_active``): once
    every job is ``done``/``failed`` the polling wrapper stops rendering its
    ``hx-trigger`` and polling ends naturally.
    """
    return any(rs.image_jobs_active for rs in run_segments)


def _range_attempts_active(run_segments: list[RunSegment]) -> bool:
    """True while any segment of this run has an in-flight range attempt.

    Same self-poll role as ``_segment_image_jobs_active`` — the run-status
    poller already re-fetches this whole fragment periodically while the run
    itself is non-terminal; this flag keeps it polling for a `done` run too,
    since promotion can flip result_local_path/segment_snapshot without any
    Run.status transition at all.
    """
    return any(rs.range_attempts_active for rs in run_segments)


def _is_localisation(project: VideoProject) -> bool:
    """True when this project runs the localisation flow.

    Goes through ``spec_for`` rather than comparing the column directly so a
    row whose ``project_type`` is NULL or unknown resolves the same way every
    other reader resolves it (to the face-swap fallback) instead of quietly
    turning on the localisation UI.
    """
    return spec_for(project.project_type).key == LOCALISATION


def _copies_enabled(run: Run, db: Session) -> bool:
    """Whether the run page offers the two cards that re-run this run with a
    fresh reference photo — "Copy run" and "Automation".

    Off for a **speech-only localisation** run, and only for that: its prompt
    keeps the person on screen, so the photo either card attaches does nothing
    and the copy is the same generation again. Both are enforced server-side
    too (``api_v2._check_copyable`` and ``put_run_automation``, 409 each) —
    this only stops the operator filling in a form that cannot work.
    """
    if getattr(run, "generation_mode", "v2v") == GENERATION_IMAGES_ONLY:
        return False  # design §6.6: no Copy run / automation for a rebuild run
    project = db.get(VideoProject, run.project_id)
    if project is None:
        return True
    return run_copies_are_useful(project.project_type, run.prompt)


def _ref_audio_header(project: VideoProject) -> str:
    """The voice-reference Speakers header for this project's transcript, or "".

    Supplied to BOTH the New Run form and the transcript panel's poll, because
    the out-of-band Translate gate carries it as a data attribute: the person
    count must follow a transcript that lands (or is replaced) after the page
    was opened.
    """
    if not _is_localisation(project):
        return ""
    transcript = project.transcript if isinstance(project.transcript, dict) else None
    return localisation.reference_audio_header(transcript)


def _localisation_prompt_templates() -> dict[str, str]:
    """The two Seedance prompt templates the New Run mode radio swaps between.

    Read off ``app/project_types.py`` by name, the same way
    ``localisation.build_prompt`` reads them, so the form and the server-side
    prompt builder can never end up offering different text. ``getattr`` (not
    an import) for the same reason build_prompt uses it: these are editable
    prompt constants, and a registry that has lost one should degrade to an
    empty template the operator can see is empty, not break the whole form.
    """
    return {
        "keep": getattr(project_types_registry, "_LOCALISATION_KEEP_PROMPT", ""),
        "swap": getattr(project_types_registry, "_LOCALISATION_SWAP_PROMPT", ""),
    }


def _new_run_defaults(project: VideoProject) -> dict:
    """New Run form pre-fill for a project — model + prompt from its type,
    plus the AI model registry the form's `<select>` and JS are built from.

    The project-type defaults are plain defaults: the form fields stay fully
    editable, and the run stores whatever the operator submits. See
    app/project_types.py.

    The three ``*model*`` keys come from app/ai_models.py so the model dropdown,
    the model→resolution options map and the per-model clip-length/audio limits
    all derive from one table instead of hand-kept copies in the template.
    Shared by the full page render and the HTMX status fragment, which both
    render partials/project_status_content.html — keep them in one helper so
    the two can never drift.

    The ``localisation_*`` / ``is_localisation`` keys drive the extra controls
    that flow needs (docs/localisation.md §7): the speech-only vs
    speech+character radio, the Translate button and its disabled reason, and
    the audio warning. They are computed for EVERY project — the template gates
    on ``is_localisation`` — so the two renders stay identical.
    """
    spec = spec_for(project.project_type)

    # The project type suggests a model, but the project's segmentation cap
    # decides which models can actually generate its segments. Offering a
    # pre-selected model that cannot is a trap: the form's red warning fires,
    # but the default is one click from a run that spends nothing and delivers
    # nothing (over-long segments fail, and the completeness gate then refuses
    # to stitch). So when the type's preferred model can't cover the cap, fall
    # back to the cheapest-listed model that can. Registry order is
    # cheapest-first within a family, so "first compatible" is also the
    # least-surprising spend.
    #
    # This only ever *widens* what the operator sees pre-filled — every model
    # stays selectable, and `models_over_cap` lets the template mark the ones
    # that would fail rather than hiding them.
    cap = float(project.max_segment_sec or ai_models.UNIVERSAL_MAX_SEGMENT_SEC)
    compatible = ai_models.models_supporting_segment_len(cap)

    # A type whose whole point is a re-spoken soundtrack has a second hard
    # constraint on the model: it must actually emit audio. Gemini Omni never
    # does, and create_run force-downgrades such a run to audio_mode="original"
    # — which overlays the UNTRANSLATED source track over the localised video.
    # So narrow the candidate list here too, and let the template mark the rest
    # unusable (`models_without_audio`) rather than silently offering them.
    models_without_audio = [
        key for key, mspec in ai_models.V2V_MODELS.items() if not mspec.produces_audio
    ]
    if spec.requires_audio_model:
        compatible = [k for k in compatible if k not in models_without_audio]
    else:
        models_without_audio = []

    # An images-only (rebuild) project offers only images-only models and
    # its own default; every other project offers only v2v models, so Kling
    # 4.0 never appears there (design §9).
    images_only = spec.generation_mode == GENERATION_IMAGES_ONLY
    form_models = ai_models.IMAGES_ONLY_MODELS if images_only else ai_models.V2V_MODELS
    if images_only:
        compatible = list(form_models)
    default_model = spec.default_model
    if compatible and default_model not in compatible:
        default_model = compatible[0]
        log.info(
            "project %s cap=%.1fs excludes the %s default model %r — "
            "New Run form defaults to %r instead",
            project.id, cap, spec.key, spec.default_model, default_model,
        )

    # The type's audio default, overridden to the generated track whenever the
    # project uploads muted clips — there is nothing else worth delivering then
    # (the source track is intact on disk, but the operator asked the model to
    # write the audio). Localisation reaches "seedance" both ways.
    default_audio_mode = "seedance" if project.mute_source else spec.default_audio_mode

    localisation_prompts = _localisation_prompt_templates()

    return {
        "project_type": spec.key,
        "project_type_label": spec.label,
        "project_type_hint": spec.hint,
        "project_uses_references": spec.uses_references,
        "default_model": default_model,
        "default_prompt": spec.default_prompt,
        # Effective segmentation cap and the models that cannot honour it, so
        # the <option> list can label them instead of silently offering a
        # guaranteed-incomplete run.
        "segment_cap_sec": cap,
        "models_over_cap": ai_models.models_excluded_by_segment_len(cap),
        # Models that emit no audio at all, listed ONLY for a type that
        # requires one (empty everywhere else, so the template's marking is
        # driven by the same flag that narrowed `default_model` above).
        "models_without_audio": models_without_audio,
        "scene_min_len_sec": settings.SCENE_MIN_LEN_SECONDS,
        # {model: [[value, label], ...]} — rebuilds the resolution dropdown.
        # Only the models this form offers (a v2v project's page is unchanged
        # by the images-only models).
        "model_resolutions_json": ai_models.resolution_choices_json(form_models),
        # {model: {max_clip_sec, produces_audio, label}} — drives the
        # "segments too long for this model" warning and the audio lock.
        "model_limits_json": ai_models.model_limits_json(form_models),
        # The specs themselves, to loop over when building the <option> list.
        "ai_models": form_models,
        # Language <option> list + the pre-selected code. English is the
        # default because it is what an unset language already means
        # downstream — picking it changes nothing about the Job ID.
        "languages": languages.LANGUAGES,
        "default_language": languages.DEFAULT_LANGUAGE,
        # Run.audio_mode the form pre-selects (see above).
        "default_audio_mode": default_audio_mode,
        # __ Localisation-only form state (docs/localisation.md §7) __
        "is_localisation": _is_localisation(project),
        # Both Seedance templates, so the "speech only" / "speech + character"
        # radio can swap the pre-filled prompt without a round-trip.
        "localisation_prompts": localisation_prompts,
        # Which of the two the registry's own default_prompt is, so the radio
        # and the textarea can never load disagreeing with each other.
        "default_localisation_mode": (
            "swap"
            if spec.default_prompt == localisation_prompts["swap"]
            else "keep"
        ),
        # Gates the "Translate" button: it can only build a prompt from a
        # transcript that is ready (docs/localisation.md §4.4 → 409 otherwise).
        # NULL = never requested. Also provided by _transcript_context for the
        # project page's panel; identical value, computed from the same column.
        "transcript_status": _transcript_status(project),
        # The Speakers header the "voice reference holds these voices" toggle
        # swaps in (AICP-335): same text localisation.format_timed_dialogue
        # writes with reference_audio=True, counted from this transcript.
        "localisation_ref_audio_header": _ref_audio_header(project),
    }


# ---------------------------------------------------------------------------
# Hook length (localisation projects) — docs/localisation.md §5
# ---------------------------------------------------------------------------


def _analyzed_hook(segments: list[SegmentDef]) -> tuple[float | None, int]:
    """The hook the stored segments actually encode: ``(end_sec, n_segments)``.

    Read off the SegmentDefs rather than recomputed from ``hook_sec``, because
    the SegmentDefs are the ground truth about what will be generated: the
    stored setting can have been changed since the analysis that produced them,
    and the segment editor lets the operator move the boundary by hand. Only
    the rows can say what the hook *is*; ``hook_sec`` only says what was asked
    for.

    A hook_split analysis lays down one or more contiguous ``swap`` segments
    starting at 0, then (usually) one ``keep`` segment over the discarded tail.
    So the hook is the end of the leading run of swap segments. Anything else —
    no segments, a project that starts on ``keep``, a gap between the swap
    segments — returns ``(None, 0)``: the shape is no longer a hook split, and
    inventing a number for it would be exactly the kind of confident-but-wrong
    claim this whole change exists to remove.
    """
    end: float | None = None
    count = 0
    for seg in segments:  # _get_segments orders by index
        if seg.action != "swap":
            break
        start = float(seg.start_sec)
        if count == 0:
            if start > 1e-6:
                return None, 0  # does not start at the front of the video
        elif abs(start - (end or 0.0)) > 1e-6:
            break  # not contiguous — the run of hook segments ended here
        end = float(seg.end_sec)
        count += 1
    if count == 0:
        return None, 0
    return end, count


def _whole_video_target_default() -> float:
    """Whole-video localisation target for a project with no explicit cap."""
    from .pipeline_v2 import whole_video_target_sec  # lazy: see _hook_context

    return whole_video_target_sec(None)


def _hook_context(project: VideoProject, segments: list[SegmentDef]) -> dict:
    """What the project page needs to make the hook clamp visible.

    Three numbers, and they are genuinely three different things — conflating
    any two of them is the bug this replaces:

    ``hook_requested_sec``  what the operator asked for (``hook_sec``, or the
                            configured default when it is NULL).
    ``hook_planned_sec``    what the NEXT analysis would cut, i.e. the request
                            after the §5 clamp. ``hook_clamp_reason`` says which
                            bound moved it.
    ``hook_cut_sec``        what the analysis that produced the segments on the
                            page ACTUALLY cut. NULL when the segments are not a
                            hook split (see :func:`_analyzed_hook`).

    ``planned`` and ``cut`` differ whenever a setting has been changed since the
    last analysis — which, now that the page can edit ``hook_sec``, is a state
    an operator can reach in one keystroke and must be told about, because
    ``hook_sec`` is consumed at analyze time and changes nothing on its own.

    Computed for every project (the templates gate on ``hook_visible``) so the
    full-page render and the HTMX status fragment can never drift, the same rule
    :func:`_new_run_defaults` follows.
    """
    # Imported here, not at module scope: app/tasks.py takes the same care, and
    # for the same reason — pipeline_v2 pulls in the whole processing stack
    # (face/scene/kie/gdrive), which the web process has no business loading at
    # import time. The clamp still comes from the pipeline rather than being
    # reimplemented here; a second copy of that arithmetic in the UI layer is
    # how the explanation on screen would start lying about the segments below
    # it.
    from .pipeline_v2 import (
        effective_segment_cap_sec,
        hook_split_plan,
        whole_video_target_sec,
    )

    scope = getattr(project, "localisation_scope", None) or LOCALISATION_SCOPE_HOOK
    whole = scope == LOCALISATION_SCOPE_WHOLE
    cap = effective_segment_cap_sec(project.max_segment_sec)
    plan = hook_split_plan(
        duration_sec=project.duration_sec,
        hook_sec=project.hook_sec,
        max_segment_sec=cap,
        scope=scope,
    )
    # A whole-video partition is not a hook split, so there is no hook "cut"
    # to compare with the request (AICP-340).
    cut_sec, cut_segments = (None, 0) if whole else _analyzed_hook(segments)

    return {
        # Localisation is the only flow with a hook; every other type resolves
        # through spec_for() exactly as the transcript panel does.
        "hook_visible": _is_localisation(project),
        # AICP-340: "hook" | "whole". In whole scope the page marks the hook
        # field unused (analysis does not read it), shows no clamp, and the
        # New Run form appends no library tail (every second is re-spoken).
        "localisation_scope": scope,
        "localisation_whole": whole and _is_localisation(project),
        # A re-analysis asked for this scope and has not applied it yet
        # (queued, crashed or failed) — None otherwise.
        "localisation_scope_requested": (
            getattr(project, "localisation_scope_requested", None)
            if getattr(project, "localisation_scope_requested", None) != scope
            else None
        ),
        "whole_video_target_sec": whole_video_target_sec(project.max_segment_sec),
        "hook_requested_sec": plan["requested"],
        "hook_is_default": plan["defaulted"],
        "hook_planned_sec": plan["effective"],
        "hook_clamp_reason": plan["reason"],
        "hook_cap_sec": plan["cap"],
        "hook_floor_sec": plan["floor"],
        "hook_default_sec": float(settings.LOCALISATION_DEFAULT_HOOK_SEC),
        "hook_cut_sec": cut_sec,
        "hook_cut_segment_count": cut_segments,
        # Did the analysis on record honour the request? (None when there is no
        # hook to compare against.)
        "hook_cut_matches_request": (
            None if cut_sec is None else abs(cut_sec - plan["requested"]) <= 1e-6
        ),
        # Is the stored hook_sec already applied to those segments, or is it a
        # pending edit? (None when there is no hook to compare against.)
        "hook_cut_is_current": (
            None if cut_sec is None else abs(cut_sec - plan["effective"]) <= 1e-6
        ),
    }


# ---------------------------------------------------------------------------
# Transcript panel (localisation projects) — docs/localisation.md §7
# ---------------------------------------------------------------------------

# Transcript states the transcription task is still working through. The panel
# self-polls only while the status is one of these; every other state is
# terminal, so the rendered fragment drops the polling wrapper and polling
# stops naturally (same pattern as project_merges_list.html).
_ACTIVE_TRANSCRIPT_STATUSES = {"pending", "running"}

# Project statuses in which analysis has not finished yet. A localisation
# project is created with transcript_status="pending" (api_v2.create_project)
# so that NULL means exactly "never requested" and the panel never renders a
# Transcribe button for a project whose transcription is already promised —
# but until analysis has RETURNED, that "pending" is a promise, not a queued
# job (tasks._enqueue_transcribe_after_analyze only pushes it afterwards).
# The panel says so, and polls, because analysis is itself a bounded wait that
# the status card next to it is already polling on.
_PRE_ANALYSIS_PROJECT_STATUSES = {"created", "analyzing"}


def _transcript_status(project: VideoProject) -> str | None:
    """Normalised ``transcript_status`` — None when never requested."""
    return (project.transcript_status or "").strip() or None


def _as_float(value) -> float | None:
    """Best-effort float, None for anything that isn't a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _transcript_lines(transcript: dict | None) -> list[dict]:
    """Normalised view of a transcript's lines for the panel.

    The transcript is operator-editable JSON (``PATCH /transcript``) written by
    a language model, so nothing guarantees the shape a template would
    otherwise format blindly — a missing or non-numeric ``start`` has to come
    out as an empty (fixable) input, not 500 the project page.

    ``index`` is the position in the STORED list, which is what the panel's
    save merges its edits back onto: line ids are stable per the §4.1 contract
    but a hand-edited transcript can repeat or drop one, and position is the
    only thing the panel itself controls.

    ``id`` is REPAIRED rather than echoed: ``PATCH /transcript`` refuses a line
    without a unique integer id >= 1 (it is what a translation is rejoined on),
    so a transcript that lost one — an older row, a hand-edited JSON, a model
    that skipped the field — would render fine and then 400 the moment the
    operator pressed Save. Ids that are already valid and unique are kept
    exactly as stored; only the broken ones are filled in, with the lowest free
    number, so nothing the translation matches on moves under it.
    """
    raws = [
        raw
        for raw in (transcript or {}).get("lines") or []
        if isinstance(raw, dict)
    ]

    kept: list[int | None] = []
    used: set[int] = set()
    for raw in raws:
        raw_id = raw.get("id")
        usable = (
            isinstance(raw_id, int)
            and not isinstance(raw_id, bool)
            and raw_id >= 1
            and raw_id not in used
        )
        kept.append(raw_id if usable else None)
        if usable:
            used.add(raw_id)

    candidate = 1
    for position, line_id in enumerate(kept):
        if line_id is not None:
            continue
        while candidate in used:
            candidate += 1
        kept[position] = candidate
        used.add(candidate)

    rows: list[dict] = []
    for index, raw in enumerate(raws):
        rows.append(
            {
                "index": index,
                "id": kept[index],
                "start": _as_float(raw.get("start")),
                "end": _as_float(raw.get("end")),
                "speaker": str(raw.get("speaker") or ""),
                "text": str(raw.get("text") or ""),
                "on_screen": bool(raw.get("on_screen")),
            }
        )
    return rows


def _transcript_context(project: VideoProject, *, oob: bool = False) -> dict:
    """Context for the project page's transcript panel.

    Shared by the full page render and the polling fragment, which both render
    partials/project_transcript_panel.html — keep them in one helper so the two
    can never drift (same rationale as _new_run_defaults).

    The three booleans below split ``transcript_status`` by what the operator
    can DO about it, which is not the same partition as the status column:

    ``transcript_active``    a job is queued or running → poll, offer a restart
                             for the case where it died silently, and render no
                             editable field (a 3s self-swap would eat the
                             keystrokes).
    ``transcript_awaiting``  "pending" written at project creation, before
                             analysis has finished — the job is not queued yet.
                             Still polls: the wait is bounded by analysis.
    ``transcript_stalled``   "pending" on a project whose ANALYSIS failed, so
                             tasks._enqueue_transcribe_after_analyze never ran
                             and never will. Terminal: stop polling, offer the
                             manual Transcribe.

    Everything that is not active is editable, because §8's fallback — "a
    failed transcription leaves the project usable, the operator pastes a
    transcript by hand" — needs the table, the language field and Save in the
    failed / empty / never-requested states, not only in ``ready``.

    *oob* is set by the polling fragment only. It adds an out-of-band copy of
    the New Run form's Translate control to the response, which is how that
    button converges to the true transcript status without re-rendering (and
    destroying) the form the operator is typing into. It is suppressed unless
    the form is actually on the page — the target only exists for a
    localisation project whose status is ``ready``.
    """
    status = _transcript_status(project)
    project_status = (
        project.status.value if hasattr(project.status, "value") else str(project.status)
    )
    transcript = project.transcript if isinstance(project.transcript, dict) else None
    lines = _transcript_lines(transcript)

    awaiting = status == "pending" and project_status in _PRE_ANALYSIS_PROJECT_STATUSES
    stalled = status == "pending" and project_status == "failed"
    active = status in _ACTIVE_TRANSCRIPT_STATUSES and not stalled
    is_localisation = _is_localisation(project)

    return {
        "is_localisation": is_localisation,
        "transcript_status": status,
        "localisation_ref_audio_header": _ref_audio_header(project),
        "transcript_active": active,
        "transcript_awaiting_analysis": awaiting,
        "transcript_stalled": stalled,
        "transcript_editable": not active,
        "transcript_error": project.transcript_error,
        "transcript_source_language": str((transcript or {}).get("source_language") or ""),
        "transcript_lines": lines,
        # First id the "add line" control may hand out: PATCH requires ids to
        # be unique, so a new row must not collide with a stored one.
        "transcript_next_line_id": (max(row["id"] for row in lines) + 1) if lines else 1,
        # The stored dict verbatim — the panel PATCHes the WHOLE §4.1 object
        # back, so it has to keep the keys it doesn't show (schema_version,
        # model, on_screen_text, …) instead of rebuilding a partial one.
        "transcript_json": transcript or {},
        "translate_gate_oob": bool(oob) and is_localisation and project_status == "ready",
    }


# ---------------------------------------------------------------------------
# Credit balance fragment — GET /v2/credits-fragment
# ---------------------------------------------------------------------------

# Small in-process TTL cache so many open tabs (each polling every 120s) don't
# each hit kie.ai. The balance changes slowly relative to a 30s window.
_CREDIT_CACHE: dict[str, float] = {}
_CREDIT_CACHE_TTL_SEC = 30.0
_CREDIT_CACHE_LOCK = threading.Lock()


def _cached_credit_balance() -> float:
    """Return the kie.ai credit balance, cached for a few tens of seconds.

    Uses a lock so on cache TTL expiry, only one caller refreshes while others
    serve the stale value (simple serve-stale-while-refresh).
    """
    import time

    from .kie_client import get_shared_client

    now = time.monotonic()
    ts = _CREDIT_CACHE.get("ts")
    if ts is not None and (now - ts) < _CREDIT_CACHE_TTL_SEC and "value" in _CREDIT_CACHE:
        return _CREDIT_CACHE["value"]

    # Try to acquire lock without blocking readers; if we can't get it immediately,
    # return stale value if available.
    if _CREDIT_CACHE_LOCK.acquire(blocking=False):
        try:
            # Double-check TTL in case another thread just refreshed
            now = time.monotonic()
            ts = _CREDIT_CACHE.get("ts")
            if ts is not None and (now - ts) < _CREDIT_CACHE_TTL_SEC and "value" in _CREDIT_CACHE:
                return _CREDIT_CACHE["value"]

            value = get_shared_client().get_credit_balance()
            _CREDIT_CACHE["value"] = value
            _CREDIT_CACHE["ts"] = now
            return value
        finally:
            _CREDIT_CACHE_LOCK.release()
    else:
        # Lock is held by another thread; return stale value if available
        if "value" in _CREDIT_CACHE:
            return _CREDIT_CACHE["value"]
        # No stale value; wait for lock so we can refresh
        with _CREDIT_CACHE_LOCK:
            # Check again in case the other thread already refreshed
            now = time.monotonic()
            ts = _CREDIT_CACHE.get("ts")
            if ts is not None and (now - ts) < _CREDIT_CACHE_TTL_SEC and "value" in _CREDIT_CACHE:
                return _CREDIT_CACHE["value"]
            value = get_shared_client().get_credit_balance()
            _CREDIT_CACHE["value"] = value
            _CREDIT_CACHE["ts"] = now
            return value


@router.get("/credits-fragment", response_class=HTMLResponse)
def credits_fragment(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    """HTMX fragment: the kie.ai credit balance shown in the site header, plus
    today's committed spend against the daily budget (AICP-155)."""
    from . import spend as spend_mod

    balance = None
    try:
        balance = _cached_credit_balance()
    except Exception as exc:  # network / auth / parse — degrade gracefully
        log.warning("credit balance fetch failed: %s", exc)
    spend = None
    try:
        spend = spend_mod.today_summary(db)
    except Exception as exc:  # e.g. migration not applied yet — never break the header
        log.warning("daily spend summary failed: %s", exc)
    return templates.TemplateResponse(
        request,
        "partials/credit_balance.html",
        {"request": request, "balance": balance, "spend": spend},
    )


# ---------------------------------------------------------------------------
# Projects dashboard — GET /v2/
# ---------------------------------------------------------------------------


# Resolutions offered by the run filter (union across models — Seedance + Omni).
_FILTER_RESOLUTIONS = ["480p", "720p", "1080p", "4k"]


def _sanitize_run_id_prefix(raw: str | None) -> str:
    """Normalise a run-id filter to a safe lowercase uuid prefix.

    Accepts the short id shown in the UI (``run.short_id``, e.g.
    ``1736426f-ja``) or a full uuid. The language suffix is stripped first —
    it is a display/export affix, never part of the stored id, and its letters
    are not all hex (``ja`` would otherwise degrade to ``1736426f-a`` and match
    nothing). What remains keeps only uuid chars (hex + dash) so no LIKE
    wildcards (``%``/``_``) can slip through. Returns "" when nothing usable
    remains (filter inactive).
    """
    stripped = languages.strip_suffix(raw).lower()
    return "".join(c for c in stripped if c in "0123456789abcdef-")


def _parse_filter_date(raw: str | None, *, end: bool = False) -> datetime | None:
    """Parse a ``YYYY-MM-DD`` filter bound to a tz-aware UTC datetime.

    For an end bound we return the START of the *next* day so the comparison
    ``created_at < date_to`` includes the whole selected day. Returns None on a
    blank/malformed value (the filter is then simply ignored).
    """
    if not raw:
        return None
    try:
        d = datetime.strptime(raw.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return d + timedelta(days=1) if end else d



def _multi_filter_values(raw: list[str] | None, allowed: list[str]) -> list[str]:
    """Sanitise a repeated multi-select filter param against a known key set.

    Every multi-select on the dashboard means "matches ANY of the ticked
    boxes", and posts its name once per ticked box. Unknown values are dropped
    rather than rejected — a stale bookmark then filters on what it still
    recognises instead of erroring — and the result is ordered by ``allowed``
    so the echoed query string is stable whatever order the browser sent.
    """
    if not raw:
        return []
    picked = {(v or "").strip() for v in raw}
    return [key for key in allowed if key in picked]


def _sheet_exported_clause():
    """Correlated SQL predicate on ``Run``: it put something into a spreadsheet.

    "Something" is any of the four kinds of video a run can export — its own
    stitched result, a tail edit, a burned-captions copy, an overlay burn — to
    either sheet: the live "Final Creatives" export, or the retired "Caption
    Demo" one, whose ``caption_demo_status`` columns still record what went out
    before it was removed (wiki/components/sheet-exports.md § Removed). The
    operator reads this filter for the plain FACT of an export, not for which
    sheet received it, so all six stamps fold into one predicate.

    Correlated to ``Run``, so it drops straight into the dashboard's per-project
    ``EXISTS`` or into a plain ``WHERE`` over runs, and negates with ``~`` for
    the "never exported" case — every operand is an IS (NOT) NULL or an EXISTS,
    so ``NOT`` has no three-valued-logic surprise to hand back.
    """
    return or_(
        Run.final_creatives_exported_at.isnot(None),
        Run.caption_demo_status.isnot(None),
        select(1)
        .where(
            RunTailEdit.run_id == Run.id,
            or_(
                RunTailEdit.final_creatives_exported_at.isnot(None),
                RunTailEdit.caption_demo_status.isnot(None),
            ),
        )
        .correlate(Run)
        .exists(),
        select(1)
        .where(
            RunCaption.run_id == Run.id,
            RunCaption.final_creatives_exported_at.isnot(None),
        )
        .correlate(Run)
        .exists(),
        # Overlays reach their run through `parent_run_id`, or — for a row that
        # predates that stamp — through one of the soft source references, the
        # same OR `_final_creatives_sent` and overlay_runs._source_filter walk.
        # Soft-deleted overlays are excluded, matching the Sent column: a burn
        # whose file is gone does not earn the run a "sent" mark.
        #
        # The source references are resolved by OUTER JOIN, not by a nested
        # ``source_tail_edit_id.in_(select(...).where(... == Run.id))``: that
        # inner select sits two levels below the run and does NOT correlate —
        # it renders its own ``FROM runs``, so one exported overlay anywhere
        # made every run in the database look exported.
        select(1)
        .select_from(OverlayJob)
        .outerjoin(RunTailEdit, RunTailEdit.id == OverlayJob.source_tail_edit_id)
        .outerjoin(RunCaption, RunCaption.id == OverlayJob.source_caption_id)
        .where(
            OverlayJob.deleted_at.is_(None),
            or_(
                OverlayJob.final_creatives_exported_at.isnot(None),
                OverlayJob.caption_demo_status.isnot(None),
            ),
            or_(
                OverlayJob.parent_run_id == Run.id,
                OverlayJob.source_run_id == Run.id,
                RunTailEdit.run_id == Run.id,
                RunCaption.run_id == Run.id,
            ),
        )
        .correlate(Run)
        .exists(),
    )


def _runs_with_sheet_export(run_ids: list[str], db: Session) -> set[str]:
    """Which of these runs have exported something — ONE query for the page."""
    if not run_ids:
        return set()
    return set(
        db.execute(
            select(Run.id).where(Run.id.in_(run_ids), _sheet_exported_clause())
        )
        .scalars()
        .all()
    )


# Tri-state values of the "Sent to sheet" filter: a run either has an export or
# it has not, so there is nothing to multi-select.
_SENT_FILTER_VALUES = ("yes", "no")


def _filter_options() -> dict[str, list[dict]]:
    """Checkbox options for the dashboard's multi-select filters.

    Every list is read off a registry (``project_types`` / ``languages`` /
    ``ai_models``), so a new project type, language or model shows up in the
    filter bar with no edit here or in the template. ``label`` is what the open
    panel shows, ``short`` what the closed button shows once it is ticked, and
    ``title`` the hover text.
    """
    return {
        "resolution": [
            {"value": r, "label": r, "short": r, "title": ""}
            for r in _FILTER_RESOLUTIONS
        ],
        # The rebuild type and its images-only models only while the feature
        # is on (design §7.6: invisible with the flag off).
        "project_type": [
            {"value": s.key, "label": s.label, "short": s.label, "title": s.hint}
            for s in selectable_project_types(rebuild_enabled=settings.REBUILD_ENABLED)
        ],
        "lang": [
            {"value": s.key, "label": s.label, "short": s.key.upper(), "title": ""}
            for s in languages.LANGUAGES.values()
        ],
        "model": [
            {
                "value": s.key,
                "label": s.label,
                "short": s.short_label,
                "title": s.hint,
            }
            for s in ai_models.AI_MODELS.values()
            if s.supports_v2v or settings.REBUILD_ENABLED
        ],
        # "Sent to sheet" is the one single-select in the bar — a run either
        # exported something or it did not — so it is rendered with radios and
        # carries an explicit "Any" reset whose empty value _run_filter_conditions
        # already reads as "no condition". It lives here anyway so the template
        # gets every filter's options from one place.
        "sent": [
            {"value": "", "label": "Any", "short": "", "title": ""},
            {
                "value": "yes",
                "label": "Sent \u2713",
                "short": "Sent",
                "title": "Something from the run reached a spreadsheet",
            },
            {
                "value": "no",
                "label": "Not sent",
                "short": "Not sent",
                "title": "Nothing from the run has been exported yet",
            },
        ],
    }


def _run_filter_conditions(
    *,
    run_id: str | None = None,
    resolution: list[str] | None = None,
    model: list[str] | None = None,
    lang: list[str] | None = None,
    sent: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list, dict]:
    """Parse the dashboard's RUN-level filters into SQL conditions + an echo.

    The conditions are ANDed against a ``Run``: the dashboard wraps them in a
    per-project ``EXISTS``, the runs pivot applies them directly — one parser
    for both, so a filtered project list and its expanded runs can never
    disagree about what matched. ``echo`` is the sanitised form the template
    re-renders the form from and re-queries the pivot fragment with.
    """
    conds = []

    # Run id — accepts the short prefix shown in the UI (``run.short_id``) or a
    # full uuid. Match as a case-insensitive prefix.
    rid_val = _sanitize_run_id_prefix(run_id)
    if rid_val:
        conds.append(Run.id.like(f"{rid_val}%"))

    res_vals = _multi_filter_values(resolution, _FILTER_RESOLUTIONS)
    if res_vals:
        conds.append(Run.resolution.in_(res_vals))

    model_vals = _multi_filter_values(model, list(ai_models.AI_MODELS))
    if model_vals:
        conds.append(Run.model.in_(model_vals))

    # Language — NULL means English (languages.py property 3: a run that never
    # chose one behaves as EN and exports a bare Job ID), and most rows are
    # NULL, so ticking English has to reach them or the filter looks broken.
    lang_vals = _multi_filter_values(lang, list(languages.LANGUAGES))
    if lang_vals:
        lang_cond = Run.language.in_(lang_vals)
        if languages.DEFAULT_LANGUAGE in lang_vals:
            lang_cond = or_(lang_cond, Run.language.is_(None))
        conds.append(lang_cond)

    sent_val = (sent or "").strip()
    if sent_val not in _SENT_FILTER_VALUES:
        sent_val = ""
    elif sent_val == "yes":
        conds.append(_sheet_exported_clause())
    else:
        conds.append(~_sheet_exported_clause())

    dt_from = _parse_filter_date(date_from)
    if dt_from is not None:
        conds.append(Run.created_at >= dt_from)
    dt_to = _parse_filter_date(date_to, end=True)
    if dt_to is not None:
        conds.append(Run.created_at < dt_to)

    echo = {
        "run_id": rid_val,
        "resolution": res_vals,
        "model": model_vals,
        "lang": lang_vals,
        "sent": sent_val,
        # Echo back the raw date strings for the <input type="date"> values.
        "date_from": (date_from or "").strip(),
        "date_to": (date_to or "").strip(),
    }
    return conds, echo


def _filter_query_string(echo: dict) -> str:
    """The active RUN-level filters as a query string (no leading ``?``).

    The lazy pivot re-fetches its runs with exactly the filters the page was
    rendered under, so an expanded project lists only the runs that made it
    match.
    """
    parts: list[str] = []
    for key in ("run_id", "sent", "date_from", "date_to"):
        if echo.get(key):
            parts.append(f"{key}={quote_plus(echo[key])}")
    for key in ("resolution", "model", "lang"):
        for value in echo.get(key) or []:
            parts.append(f"{key}={quote_plus(value)}")
    return "&".join(parts)


def _autopilot_run_labels(db: Session, project, runs) -> dict:
    """``{run_id: {kind, label}}`` for an assistant project's runs table (AICP-369)."""
    if getattr(project, "mode", None) != "assistant" or not runs:
        return {}
    from .autopilot import run_view

    return run_view.run_labels(db, project, runs)


def _run_pivot_rows(runs: list[Run], db: Session) -> list[dict]:
    """View model for the dashboard's runs pivot (project_runs_list.html).

    A plain dict per run — the template must not lazy-load ORM attributes
    mid-render — including the sheet-export flag the "Sheet" column and the
    "Sent to sheet" filter both read, resolved for the whole page at once.
    """
    sent_ids = _runs_with_sheet_export([r.id for r in runs], db)
    rows = []
    for run in runs:
        status_val = (
            run.status.value if hasattr(run.status, "value") else str(run.status)
        )
        rows.append(
            {
                "id": run.id,
                "project_id": run.project_id,
                "short_id": run.short_id,
                "name": run.name,
                "status": status_val,
                "model": run.model,
                "resolution": run.resolution,
                "language": run.language,
                "sent_to_sheet": run.id in sent_ids,
                # Denormalised un-swapped count (written once at finalize) —
                # the template flags a partly-swapped `done` run with it.
                "source_fallback_segments": run.source_fallback_segments,
                "created_at": run.created_at,
                "result_available": (
                    status_val == "done"
                    and bool(run.result_local_path)
                    and os.path.exists(run.result_local_path)
                ),
            }
        )
    return rows


def _pivot_runs_by_project(
    project_ids: list[str], run_conds: list, db: Session
) -> dict[str, list[dict]]:
    """Matching runs for every listed project, grouped by project id.

    Used when a run filter is active: the point of a filtered dashboard is to
    READ the runs it selected, so they are rendered inline and expanded rather
    than lazy-loaded — and fetching them here is one query for the page instead
    of one HTTP round-trip per listed project. Bounded by the caller's 100-
    project cap.
    """
    if not project_ids:
        return {}
    runs = (
        db.execute(
            select(Run)
            .where(Run.project_id.in_(project_ids), *run_conds)
            .order_by(desc(Run.created_at))
        )
        .scalars()
        .all()
    )
    grouped: dict[str, list[dict]] = {}
    for row in _run_pivot_rows(runs, db):
        grouped.setdefault(row["project_id"], []).append(row)
    return grouped


@router.get("/", response_class=HTMLResponse)
def projects_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    run_id: str | None = Query(default=None),
    project_type: list[str] | None = Query(default=None),
    model: list[str] | None = Query(default=None),
    resolution: list[str] | None = Query(default=None),
    lang: list[str] | None = Query(default=None),
    sent: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
) -> HTMLResponse:
    """v2 dashboard: list of VideoProjects + new-project form.

    Filters: the project's own **Type**, plus the RUN-level ones — run id,
    **Resolution**, **Model**, **Language**, **Sent to sheet**, and a created-
    date range. A project is listed when it has at least one Run matching
    *all* active run filters; within one multi-select the ticked values are
    ORed ("any of these"), and the filters are ANDed with each other.

    With a run filter active the matching runs are rendered inline and already
    expanded (``pivot_runs``) — "show me every project whose runs never reached
    a spreadsheet" is a question about the runs, so the page answers it without
    88 follow-up clicks.
    """
    run_conds, echo = _run_filter_conditions(
        run_id=run_id,
        resolution=resolution,
        model=model,
        lang=lang,
        sent=sent,
        date_from=date_from,
        date_to=date_to,
    )
    type_vals = _multi_filter_values(project_type, list(PROJECT_TYPES))

    stmt = select(VideoProject).order_by(desc(VideoProject.created_at))
    if type_vals:
        stmt = stmt.where(VideoProject.project_type.in_(type_vals))
    if run_conds:
        stmt = stmt.where(
            exists().where(and_(Run.project_id == VideoProject.id, *run_conds))
        )
    # Bound the result set to the newest N projects (#18: no full-table scans
    # on every dashboard load).  Filters get the same cap so filtered views
    # are also bounded.
    stmt = stmt.limit(100)

    projects = db.execute(stmt).scalars().all()

    # Only a RUN filter pre-renders the pivot: a Type-only filter selects
    # projects, not runs, so there is nothing to show expanded.
    pivot_runs = (
        _pivot_runs_by_project([p.id for p in projects], run_conds, db)
        if run_conds
        else {}
    )

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "request": request,
            "projects": projects,
            "project_types": selectable_project_types(rebuild_enabled=settings.REBUILD_ENABLED),
            "default_project_type": DEFAULT_PROJECT_TYPE,
            # Placeholder for the create form's hook_sec input: NULL means
            # "use this", so showing it beats showing an empty box.
            "localisation_default_hook_sec": settings.LOCALISATION_DEFAULT_HOOK_SEC,
            # Whole-video localisation target shown on the create form
            # (AICP-340); a project without an explicit cap gets exactly this.
            "localisation_whole_video_max_sec": _whole_video_target_default(),
            "filter_options": _filter_options(),
            "filters_active": bool(run_conds or type_vals),
            "pivot_runs": pivot_runs,
            "filter_qs": _filter_query_string(echo),
            "filter": dict(echo, project_type=type_vals),
        },
    )


# ---------------------------------------------------------------------------
# Generations dashboard — GET /v2/generations
# ---------------------------------------------------------------------------


# Newest-N cap, same rationale as the projects dashboard (no full-table scans).
_GENERATIONS_LIMIT = 200

# CSV export isn't bound to the dashboard's display cap, but it still needs a
# sane ceiling so a pathological filter (or "no filter at all") can't pull an
# unbounded result set into memory.
_GENERATIONS_EXPORT_LIMIT = 10_000

def _file_size_mb(path: str | None) -> float | None:
    """Size of a local file in MB, or None when the path is unset/missing."""
    if not path:
        return None
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None


def _parse_generation_filters(
    status: str | None, project: str | None
) -> tuple[str, SegmentStatus | None, str]:
    """Shared status/project filter parsing for the dashboard + CSV export.

    Returns (status_val, status_filter, project_val) — the echoed-back raw
    values plus the parsed SegmentStatus enum (or None when absent/invalid).
    """
    status_val = (status or "").strip()
    status_filter = None
    if status_val:
        try:
            status_filter = SegmentStatus(status_val)
        except ValueError:
            status_val = ""  # unknown status → ignore the filter

    project_val = (project or "").strip()
    return status_val, status_filter, project_val


def _generations_stmt(
    status_filter: SegmentStatus | None, project_val: str, limit: int
):
    """Build the RunSegment select for the generations table/export, with the
    optional status + project filters applied and the eager-load options the
    row builder below relies on."""
    stmt = (
        select(RunSegment)
        .order_by(desc(RunSegment.created_at))
        .options(
            joinedload(RunSegment.run).joinedload(Run.project),
            joinedload(RunSegment.segment_def),
        )
        .limit(limit)
    )
    if status_filter is not None:
        stmt = stmt.where(RunSegment.status == status_filter)
    if project_val:
        stmt = stmt.where(
            exists().where(
                and_(Run.id == RunSegment.run_id, Run.project_id == project_val)
            )
        )
    return stmt


def _build_generation_rows(run_segments) -> list[dict]:
    """RunSegment ORM rows → plain dicts for the table template and CSV export.

    ``run_id``/``project_id`` are always the full ids (the template truncates
    for display; the CSV export uses them as-is)."""
    rows = []
    for rs in run_segments:
        rs_status = rs.status.value if hasattr(rs.status, "value") else str(rs.status)
        run = rs.run
        project = run.project if run is not None else None
        # The window this generation ran on, not wherever its def sits today.
        win_start, win_end = _launch_window(run, rs)
        rows.append(
            {
                "segment_index": rs.index,
                "segment_range": (
                    f"{win_start:.1f}–{win_end:.1f}s"
                    if win_start is not None and win_end is not None
                    else None
                ),
                "run_id": run.id if run is not None else rs.run_id,
                # The id as the sheet sees it (prefix + language suffix); the
                # template shows this instead of truncating run_id itself.
                "run_short_id": (
                    run.short_id if run is not None else rs.run_id[:8]
                ),
                "run_name": run.name if run is not None else None,
                "project_id": project.id if project is not None else None,
                "project_name": project.name if project is not None else None,
                "status": rs_status,
                "model": ai_models.label_for(run.model) if run is not None else "—",
                "resolution": run.resolution if run is not None else "—",
                "size_mb": _file_size_mb(rs.local_result_path),
                # Display-only: the table links each row to the clip kie.ai
                # returned for it, None when nothing is on disk. Deliberately
                # absent from the CSV columns — a host-relative URL is no use
                # in a spreadsheet.
                "clip_url": _segment_clip_url(rs),
                "created_at": rs.created_at,
            }
        )
    return rows


def _generations_project_options(db: Session):
    """Distinct (id, name) of projects that have at least one generation
    (RunSegment), for the Project filter dropdown — newest-created first."""
    stmt = (
        select(VideoProject.id, VideoProject.name)
        .join(Run, Run.project_id == VideoProject.id)
        .join(RunSegment, RunSegment.run_id == Run.id)
        .distinct()
        .order_by(VideoProject.name)
    )
    return db.execute(stmt).all()


@router.get("/generations", response_class=HTMLResponse)
def generations_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    status: str | None = Query(default=None),
    project: str | None = Query(default=None),
) -> HTMLResponse:
    """Flat table of the latest per-segment generation videos (RunSegments)
    across all projects, in every status — run, project, status, model,
    resolution, result size, a link to the clip itself, date. The rows still carry the segment index and
    window; only the CSV export prints them (the table drops that column).
    Optional exact status + project filters (combined with AND when both are
    set)."""
    status_val, status_filter, project_val = _parse_generation_filters(status, project)

    stmt = _generations_stmt(status_filter, project_val, _GENERATIONS_LIMIT)
    run_segments = db.execute(stmt).scalars().all()
    rows = _build_generation_rows(run_segments)

    return templates.TemplateResponse(
        request,
        "generations.html",
        {
            "request": request,
            "rows": rows,
            "segment_statuses": [s.value for s in SegmentStatus],
            "filter_status": status_val,
            "filter_project": project_val,
            "project_options": _generations_project_options(db),
            "limit": _GENERATIONS_LIMIT,
        },
    )


@router.get("/generations.csv")
def generations_export_csv(
    db: Session = Depends(get_db),
    status: str | None = Query(default=None),
    project: str | None = Query(default=None),
) -> Response:
    """CSV export of the (status/project-)filtered generations table.

    Reuses the same query as the dashboard but without the 200-row display
    cap — bounded instead by ``_GENERATIONS_EXPORT_LIMIT`` so a wide-open
    filter can't pull an unbounded result set into memory."""
    status_val, status_filter, project_val = _parse_generation_filters(status, project)

    stmt = _generations_stmt(status_filter, project_val, _GENERATIONS_EXPORT_LIMIT)
    run_segments = db.execute(stmt).scalars().all()
    rows = _build_generation_rows(run_segments)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Segment",
            "Segment Range",
            "Run ID",
            # The short, language-suffixed id — the value that reaches the
            # Final Creatives sheet, so a CSV row can be matched to a sheet row
            # without re-deriving it. The full uuid stays in "Run ID".
            "Job ID",
            "Run Name",
            "Project ID",
            "Project Name",
            "Status",
            "Model",
            "Resolution",
            "Size (MB)",
            "Date",
        ]
    )
    for row in rows:
        writer.writerow(
            [neutralise_formula(c) for c in [
                row["segment_index"],
                row["segment_range"] or "",
                row["run_id"] or "",
                row["run_short_id"] or "",
                row["run_name"] or "",
                row["project_id"] or "",
                row["project_name"] or "",
                row["status"],
                row["model"],
                row["resolution"],
                f'{row["size_mb"]:.1f}' if row["size_mb"] is not None else "",
                row["created_at"].strftime("%Y-%m-%d %H:%M:%S") if row["created_at"] else "",
            ]]
        )

    filename = f"generations_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Automations overview — GET /v2/automations
# ---------------------------------------------------------------------------


def _filters_summary(a: RunAutomation) -> str:
    """Human-readable one-line summary of an automation's sheet filter rule."""
    parts = []
    if a.filter_persona:
        parts.append(f"persona={a.filter_persona}")
    if a.filter_age:
        parts.append(f"age={a.filter_age}")
    if a.filter_gender:
        parts.append(f"gender={a.filter_gender}")
    if a.filter_ethnicity:
        parts.append(f"ethnicity={a.filter_ethnicity}")
    return ", ".join(parts) if parts else "any row"


@router.get("/automations", response_class=HTMLResponse)
def automations_overview(
    request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Global view of every run automation, split into:

    * **Active** — ``enabled`` automations (launching batches on each tick).
    * **Previously** — disabled automations that fired at least one launch, so
      the operator can see what ran in the past without opening each run.

    Disabled automations that never launched anything are omitted (they were
    configured but never ran).
    """
    automations = (
        db.execute(
            select(RunAutomation)
            .options(
                selectinload(RunAutomation.run),
                selectinload(RunAutomation.launches),
            )
            .order_by(desc(RunAutomation.updated_at))
        )
        .scalars()
        .all()
    )

    # Batch-resolve in-flight status: collect every pending (un-notified) launched
    # run across all automations, then a single query for their statuses.
    pending_ids = {
        l.new_run_id
        for a in automations
        for l in a.launches
        if l.notified_at is None and l.new_run_id
    }
    live_ids: set[str] = set()
    if pending_ids:
        rows = db.execute(
            select(Run.id, Run.status).where(Run.id.in_(pending_ids))
        ).all()
        live_ids = {rid for rid, status in rows if status not in _TERMINAL_RUN_STATUSES}

    now = datetime.now(timezone.utc)

    def _row(a: RunAutomation) -> dict:
        launches = a.launches  # ordered by launched_at (relationship order_by)
        launch_times = [l.launched_at for l in launches]
        in_flight = sum(
            1
            for l in launches
            if l.notified_at is None and l.new_run_id in live_ids
        )
        run = a.run
        run_status = (
            run.status.value if run and hasattr(run.status, "value")
            else (str(run.status) if run else "—")
        )
        return {
            "id": a.id,
            "run_id": a.run_id,
            "run_short": (a.run_id or "")[:8],
            "run_name": (run.name if run else None),
            "run_status": run_status,
            "enabled": a.enabled,
            "filters": _filters_summary(a),
            "plan": a.plan,
            "daily_count": a.daily_count,
            "has_webhook": bool(a.webhook_url or settings.AUTOMATION_WEBHOOK_URL),
            "used_total": len(launches),
            "launched_today": launched_today(db, a.id, now),
            "in_flight": in_flight,
            "first_launch_at": launch_times[0] if launch_times else None,
            "last_launch_at": launch_times[-1] if launch_times else None,
            "updated_at": a.updated_at,
        }

    active = [_row(a) for a in automations if a.enabled]
    previously = [
        _row(a) for a in automations if not a.enabled and a.launches
    ]

    return templates.TemplateResponse(
        request,
        "automations.html",
        {
            "request": request,
            "active": active,
            "previously": previously,
        },
    )


# ---------------------------------------------------------------------------
# Dashboard runs pivot — GET /v2/projects/{pid}/runs-fragment
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/runs-fragment", response_class=HTMLResponse)
def project_runs_fragment(
    pid: str,
    request: Request,
    db: Session = Depends(get_db),
    run_id: str | None = Query(default=None),
    model: list[str] | None = Query(default=None),
    resolution: list[str] | None = Query(default=None),
    lang: list[str] | None = Query(default=None),
    sent: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
) -> HTMLResponse:
    """HTMX fragment: the Existing-runs list for a project, shown inline in the
    dashboard pivot so finished videos are reachable without opening the project.

    Takes the dashboard's whole run-level filter set (the page hands its own
    query string back through ``ACTIVE_RUN_FILTERS``), so an expanded project
    lists only the runs that made it match — not every run it owns. With no
    filters it lists them all.
    """
    _get_project_or_404(pid, db)
    run_conds, _ = _run_filter_conditions(
        run_id=run_id,
        resolution=resolution,
        model=model,
        lang=lang,
        sent=sent,
        date_from=date_from,
        date_to=date_to,
    )
    runs = _get_runs(pid, db, extra_conds=run_conds)
    return templates.TemplateResponse(
        request,
        "partials/project_runs_list.html",
        {"request": request, "project_id": pid, "runs": _run_pivot_rows(runs, db)},
    )


# ---------------------------------------------------------------------------
# Project page — GET /v2/projects/{pid}
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}", response_class=HTMLResponse)
def project_detail(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Full project detail page."""
    project = _get_project_or_404(pid, db)
    segments = _get_segments(pid, db)
    runs = _get_runs(pid, db)
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)
    context = {
        "request": request,
        "project": project,
        "status_val": status_val,
        "segments": segments,
        "runs": runs,
        "default_resolution": settings.DEFAULT_RESOLUTION,
        "source_public_token": make_source_token(pid),
        "max_refs": settings.MAX_REFERENCE_IMAGES,
        "gdrive_folder_id": settings.GDRIVE_DEFAULT_FOLDER_ID or "",
        "library_clips": _get_library_clips(db),
        "voices": _get_voice_clips(db),
        "source_run": _get_source_run(project, db),
        "source_step": _get_source_step(project, db),  # AICP-331
        "final_creatives_sent": _final_creatives_sent(runs, db),
        # AICP-369: by Assistant / manual label per run (assistant projects only).
        "run_labels": _autopilot_run_labels(db, project, runs),
        # Batch VOID fills: the default per-run VOID allowance (design §3).
        "void_auto_default": float(getattr(settings, "AUTOPILOT_VOID_AUTO_USD", 1.0)),
    }
    context.update(_new_run_defaults(project))
    context.update(_hook_context(project, segments))
    context.update(_transcript_context(project))
    context.update(_get_insert_extraction_context(pid, db))
    merges = _get_merges(pid, db)
    context["merges"] = merges
    context["merges_active"] = _merges_still_active(merges)
    return templates.TemplateResponse(request,"project_detail.html", context)


# ---------------------------------------------------------------------------
# Project merges fragment — GET /v2/projects/{pid}/merges-fragment
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/merges-fragment", response_class=HTMLResponse)
def project_merges_fragment(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment: PIP-insert merges targeting this project.

    Self-polls (every 3s) only while at least one merge is still active
    (queued/compositing/delivering); once every merge is done/failed the
    rendered fragment omits the polling wrapper, so polling stops naturally
    (same pattern as run_status_content.html).
    """
    _get_project_or_404(pid, db)
    merges = _get_merges(pid, db)
    return templates.TemplateResponse(
        request,
        "partials/project_merges_list.html",
        {
            "request": request,
            "project_id": pid,
            "merges": merges,
            "merges_active": _merges_still_active(merges),
        },
    )


# ---------------------------------------------------------------------------
# Project transcript fragment — GET /v2/projects/{pid}/transcript-fragment
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/transcript-fragment", response_class=HTMLResponse)
def project_transcript_fragment(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment: the localisation transcript panel.

    Self-polls (every 3s) only while transcription is still pending/running;
    once it is ready/empty/failed the rendered fragment omits the polling
    wrapper, so polling stops naturally (same pattern as
    project_merges_fragment / run_status_content.html). Also the swap target
    after "Transcribe" and after an inline edit is saved.

    Rendered with ``oob=True``: the response also carries an out-of-band copy
    of the New Run form's Translate control (partials/translate_gate.html), so
    this one request updates BOTH the panel and the button that depends on it.
    That is the whole reason the Translate state converges without a reload —
    #status-content deliberately stops polling at ``ready`` so a timer can
    never wipe the prompt and per-segment textareas the operator is typing in.
    """
    project = _get_project_or_404(pid, db)
    context = {"request": request, "project": project}
    context.update(_transcript_context(project, oob=True))
    return templates.TemplateResponse(
        request,
        "partials/project_transcript_panel.html", context
    )


# ---------------------------------------------------------------------------
# Project status fragment — GET /v2/projects/{pid}/status-fragment
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/status-fragment", response_class=HTMLResponse)
def project_status_fragment(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment for project status-dependent content."""
    project = _get_project_or_404(pid, db)
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)

    # Only query segments/runs when status is "ready"; template only renders them then
    if status_val == "ready":
        segments = _get_segments(pid, db)
        runs = _get_runs(pid, db)
    else:
        segments = []
        runs = []

    context = {
        "request": request,
        "project": project,
        "status_val": status_val,
        "segments": segments,
        "runs": runs,
        "default_resolution": settings.DEFAULT_RESOLUTION,
        "source_public_token": make_source_token(pid),
        "max_refs": settings.MAX_REFERENCE_IMAGES,
        "gdrive_folder_id": settings.GDRIVE_DEFAULT_FOLDER_ID or "",
        "library_clips": _get_library_clips(db) if status_val == "ready" else [],
        # Same gate as library_clips: the New Run form (the only consumer) is
        # rendered only at "ready".
        "voices": _get_voice_clips(db) if status_val == "ready" else [],
        # Empty dict for a non-ready project: `runs` is [] there, and the
        # Existing-runs table the map feeds is not rendered at all.
        "final_creatives_sent": _final_creatives_sent(runs, db),
        # AICP-369: by Assistant / manual label per run (assistant projects only).
        "run_labels": _autopilot_run_labels(db, project, runs),
        # Batch VOID fills: the default per-run VOID allowance (design §3).
        "void_auto_default": float(getattr(settings, "AUTOPILOT_VOID_AUTO_USD", 1.0)),
    }
    context.update(_new_run_defaults(project))
    # Segments are only loaded (and the segment editor only rendered) at
    # "ready"; every other status gets an empty list, which _analyzed_hook
    # reports as "no hook on record" — correct, there are no segments yet.
    context.update(_hook_context(project, segments))
    return templates.TemplateResponse(request,"partials/project_status_content.html", context)


# ---------------------------------------------------------------------------
# Run page — GET /v2/runs/{rid}
# ---------------------------------------------------------------------------


def _paid_attempts_view(run, run_segments, db) -> list:
    """Open paid submit attempts of the run for its "outcome unknown" card
    (AICP-351, design §5.1). Never raises: the run page must render."""
    try:
        from . import paid_attempts

        idx = {rs.id: rs.index for rs in run_segments}
        out = []
        for row in paid_attempts.run_open_attempts(db, run.id):
            d = paid_attempts.describe(row)
            if row.owner_kind == "run_segment" and row.owner_id in idx:
                d["owner_label"] = f"#{idx[row.owner_id]}"
            elif row.owner_kind == "run_preprocess":
                d["owner_label"] = f"#{row.owner_id.rsplit(':', 1)[-1]}"
            else:
                d["owner_label"] = ""
            out.append(d)
        return out
    except Exception:
        log.exception("run %s: paid attempts view failed", run.id)
        return []


@router.get("/runs/{rid}", response_class=HTMLResponse)
def run_detail(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Full run detail page."""
    run = _get_run_or_404(rid, db)
    run_segments = _get_run_segments(run, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    gdrive_link = _gdrive_link_for_run(run)
    from .assistant_access import ProjectAllowlist
    assistant_available = (
        settings.ASSISTANT_ENABLED and status_val == "done"
        and db.get(VideoProject, run.project_id) is not None
        and ProjectAllowlist.from_settings(settings).allows_project(run.project_id)
    )

    # Compute progress for display
    total_swap = len(run_segments)
    completed = sum(
        1 for rs in run_segments
        if (rs.status.value if hasattr(rs.status, "value") else str(rs.status)) == "completed"
    )

    from .knowledge.web import card_visible as _knowledge_card_visible
    knowledge_card = _knowledge_card_visible(run, db.get(VideoProject, run.project_id) is not None)

    _run_project = db.get(VideoProject, run.project_id)
    context = {
        "request": request,
        "run": run,
        # AICP-369: an assistant project's run pages load the assistant card.
        "project_mode": getattr(_run_project, "mode", None),
        "assistant_available": assistant_available,
        # Outcome card (AICP-299): KNOWLEDGE_ENABLED + KNOWLEDGE_FEEDBACK_UI,
        # a done run, and its project in knowledge scope.
        "knowledge_card": knowledge_card,
        "project_id": run.project_id,
        "status_val": status_val,
        "run_segments": run_segments,
        "fill_batches": _fill_batches_view(run, db),
        "gdrive_link": gdrive_link,
        "total_swap": total_swap,
        "completed": completed,
        "result_public_token": make_result_token(rid),
        "sheet_exported": run.final_creatives_exported_at is not None,
        "result_version": _result_version_for_run(run),
        "max_copy_runs": _MAX_BATCH_COPY_RUNS,
        # Copy-run + Automation cards: hidden (with the reason) for a
        # speech-only localisation run, where a copy can only repeat itself.
        "copies_enabled": _copies_enabled(run, db),
        "stitch_plan_summary": _stitch_plan_summary(run, db),
        # "Move to new project" panel — every flow is offered as the target.
        "project_types": selectable_project_types(rebuild_enabled=settings.REBUILD_ENABLED),
        "default_project_type": DEFAULT_PROJECT_TYPE,
        # AICP-378: its mode cards default to the parent's mode (as the API does)
        "parent_mode": getattr(db.get(VideoProject, run.project_id), "mode", None) or "manual",
        # Copy-run panel: the resolutions offered for a copy are the ones this
        # run's model supports, read straight off the registry.
        "ai_models": ai_models.AI_MODELS,
        # Language selector next to the run name — see app/languages.py.
        "languages": languages.LANGUAGES,
        "default_language": languages.DEFAULT_LANGUAGE,
        # Per-segment GPT Image 2 regeneration block — visible only when
        # run.preprocess_enabled (checked in the template).
        "segment_image_jobs_active": _segment_image_jobs_active(run_segments),
        "range_attempts_active": _range_attempts_active(run_segments),
        "range_regen_eligible": run.status == RunStatus.done,
        "default_segment_image_prompt": _default_segment_image_prompt(run),
    }
    context.update(_get_insert_merge_context(run, db))
    context.update(_get_tail_edit_context(run, db))
    context.update(_get_audio_patch_context(run, db))
    context.update(_get_caption_context(run, db))
    context.update(_get_overlay_context(run, db))
    context.update(_rebuild_run_context(run, db))
    context["paid_attempts"] = _paid_attempts_view(run, run_segments, db)
    return templates.TemplateResponse(request,"run_detail.html", context)


# ---------------------------------------------------------------------------
# Run status fragment — GET /v2/runs/{rid}/status-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/status-fragment", response_class=HTMLResponse)
def run_status_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment for run status-dependent content."""
    run = _get_run_or_404(rid, db)
    run_segments = _get_run_segments(run, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    gdrive_link = _gdrive_link_for_run(run)

    total_swap = len(run_segments)
    completed = sum(
        1 for rs in run_segments
        if (rs.status.value if hasattr(rs.status, "value") else str(rs.status)) == "completed"
    )

    # Find current generating segment
    generating_seg = None
    for rs in run_segments:
        rs_status = rs.status.value if hasattr(rs.status, "value") else str(rs.status)
        if rs_status == "generating":
            generating_seg = rs
            break

    context = {
        "request": request,
        "run": run,
        "project_id": run.project_id,
        "status_val": status_val,
        "run_segments": run_segments,
        "fill_batches": _fill_batches_view(run, db),
        "gdrive_link": gdrive_link,
        "total_swap": total_swap,
        "completed": completed,
        "generating_seg": generating_seg,
        "result_public_token": make_result_token(rid),
        "sheet_exported": run.final_creatives_exported_at is not None,
        "result_version": _result_version_for_run(run),
        # Same "Move to new project" panel as the full page — the polling swap
        # replaces it, so the fragment must carry its context too.
        "project_types": selectable_project_types(rebuild_enabled=settings.REBUILD_ENABLED),
        "default_project_type": DEFAULT_PROJECT_TYPE,
        "parent_mode": getattr(db.get(VideoProject, run.project_id), "mode", None) or "manual",
        "segment_image_jobs_active": _segment_image_jobs_active(run_segments),
        "range_attempts_active": _range_attempts_active(run_segments),
        "range_regen_eligible": run.status == RunStatus.done,
        "default_segment_image_prompt": _default_segment_image_prompt(run),
    }
    context.update(_get_insert_merge_context(run, db))
    context.update(_get_tail_edit_context(run, db))
    context.update(_get_audio_patch_context(run, db))
    context.update(_get_caption_context(run, db))
    context.update(_get_overlay_context(run, db))
    context.update(_rebuild_run_context(run, db))
    context["paid_attempts"] = _paid_attempts_view(run, run_segments, db)
    return templates.TemplateResponse(request,"partials/run_status_content.html", context)


# ---------------------------------------------------------------------------
# Run segment-image-jobs fragment — GET /v2/runs/{rid}/segment-image-jobs-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/segment-image-jobs-fragment", response_class=HTMLResponse)
def run_segment_image_jobs_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment: every segment's "Generated image" block.

    Unlike the tail-edit/caption/overlay fragments (one list, one hx-target),
    this one re-renders content scattered across several per-segment panels
    in a single response — the trigger itself (this route's own `hx-target`)
    self-terminates (mirrors ``_tail_edits_still_active``: the polling wrapper
    stops rendering its `hx-trigger` once nothing is active any more) while
    each segment's block rides along as an `hx-swap-oob` payload keyed by
    ``segimg-<rs.id>`` (see ``partials/segment_image_block.html`` and
    ``docs/gpt-image-2-preprocess.md``'s per-segment regeneration section —
    the same oob pattern ``partials/translate_gate.html`` uses).
    """
    run = _get_run_or_404(rid, db)
    run_segments = _get_run_segments(run, db)
    context = {
        "request": request,
        "run": run,
        "project_id": run.project_id,
        "run_segments": run_segments,
        "fill_batches": _fill_batches_view(run, db),
        "segment_image_jobs_active": _segment_image_jobs_active(run_segments),
        "range_attempts_active": _range_attempts_active(run_segments),
        "range_regen_eligible": run.status == RunStatus.done,
        "default_segment_image_prompt": _default_segment_image_prompt(run),
    }
    return templates.TemplateResponse(
        request,
        "partials/run_segment_image_jobs_fragment.html", context
    )


# ---------------------------------------------------------------------------
# Run tail-edits fragment — GET /v2/runs/{rid}/tail-edits-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/tail-edits-fragment", response_class=HTMLResponse)
def run_tail_edits_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment: this run's "Change ending" tail-edit list.

    Self-polls (every 3s) only while at least one tail edit is still active
    (queued/processing/delivering); once every tail edit is done/failed the
    rendered fragment omits the polling wrapper, so polling stops naturally
    (same pattern as project_merges_fragment / run_status_content.html).
    """
    _get_run_or_404(rid, db)
    edits = _get_tail_edits(rid, db)
    # Source-picker options are form-only (they live in run_status_content.html,
    # not this list-only fragment) — omitted here, same as the other form-only
    # keys (library_clips/cut_prefill/result_duration_sec) this route has
    # never populated. Row source labels ARE needed, so they go through
    # `_tail_edit_context` too — the single source of truth for this
    # template's keys, same helper `_get_tail_edit_context` uses.
    context = _tail_edit_context(
        library_clips=[], cut_prefill=None, result_duration_sec=None,
        edits=edits, edits_active=_tail_edits_still_active(edits),
        clip_names=_tail_clip_names(edits, db),
        source_caption_options={}, source_caption_durations={},
        source_labels=_referenced_caption_labels(edits, db),
    )
    return templates.TemplateResponse(
        request,
        "partials/run_tail_edits_list.html",
        context | {"request": request, "run_id": rid},
    )


# ---------------------------------------------------------------------------
# Run audio-patches fragment — GET /v2/runs/{rid}/audio-patches-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/audio-patches-fragment", response_class=HTMLResponse)
def run_audio_patches_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment: this run's audio-patch history.

    Self-polls (every 3s) only while at least one patch is still active; once
    they are all done/failed the rendered fragment omits the polling wrapper
    and polling stops naturally (same pattern as run_tail_edits_fragment).

    A patch that reaches `done` has re-pointed the run's result, so the poll
    that renders it is also the operator's cue that the video above is now the
    fixed one — they reload to see it.
    """
    _get_run_or_404(rid, db)
    patches = _get_audio_patches(rid, db)
    # Form-only keys (donor options, the result duration, the crossfade
    # default) live in run_status_content.html, not this list-only fragment —
    # omitted here exactly like run_tail_edits_fragment omits its own.
    context = _audio_patch_context(
        patches=patches,
        patches_active=_audio_patches_still_active(patches),
        donor_runs=[],
        donor_labels=_audio_patch_donor_labels(patches, db),
        result_duration_sec=None,
    )
    return templates.TemplateResponse(
        request,
        "partials/run_audio_patches_list.html",
        context | {"request": request, "run_id": rid},
    )


# ---------------------------------------------------------------------------
# Run captions fragment — GET /v2/runs/{rid}/captions-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/captions-fragment", response_class=HTMLResponse)
def run_captions_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment: this run's "Captions" job list.

    Self-polls (every 3s) only while at least one caption job is still active
    (queued/transcribing/burning/delivering); once every job is done/failed
    the rendered fragment omits the polling wrapper, so polling stops
    naturally (same pattern as run_tail_edits_fragment).
    """
    _get_run_or_404(rid, db)
    captions = _get_captions(rid, db)
    return templates.TemplateResponse(
        request,
        "partials/run_captions_list.html",
        _caption_context(
            captions=captions,
            captions_active=_captions_still_active(captions),
            run_id=rid,
            # Source-picker options are form-only (they live in
            # run_status_content.html's "Add Captions" control, not this
            # list-only fragment) — omitted here, same as tail-edits-fragment
            # omits library_clips/cut_prefill/result_duration_sec. Row source
            # labels ARE needed by this template.
            source_tail_edit_options={},
            source_labels=_referenced_tail_edit_labels(captions, db),
        )
        | {"request": request},
    )


# ---------------------------------------------------------------------------
# Run overlays fragment — GET /v2/runs/{rid}/overlays-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/overlays-fragment", response_class=HTMLResponse)
def run_overlays_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment: this run's overlay-job history.

    Self-polls (every 3s) only while at least one overlay job is still active
    (queued/processing/delivering); once every job is done/failed the rendered
    fragment omits the polling wrapper, so polling stops naturally (same
    pattern as run_tail_edits_fragment / run_captions_fragment).
    """
    run = _get_run_or_404(rid, db)
    jobs = overlay_runs.overlay_jobs_for_run(rid, db)
    return templates.TemplateResponse(
        request,
        "partials/run_overlays_list.html",
        _overlay_context(
            overlay_jobs=_overlay_rows(run, jobs, db),
            overlays_active=_overlays_still_active(jobs),
            run_id=rid,
            # Source-picker options are form-only (they live in the "Overlays"
            # card in run_status_content.html, not this list-only fragment) —
            # omitted here, same as the tail-edit/caption fragments.
            source_tail_edit_options={},
            source_caption_options={},
        )
        | {"request": request},
    )


# ---------------------------------------------------------------------------
# Reusable asset library page — GET /v2/library
# (stitch clips + overlay PNG cards + voice references, all global and reusable)
# ---------------------------------------------------------------------------


@router.get("/library", response_class=HTMLResponse)
def library_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    """Reusable-asset library: stitch clips, overlay PNG cards and voices."""
    return templates.TemplateResponse(
        request,
        "library.html",
        {
            "request": request,
            "clips": _get_library_clips(db),
            "overlay_assets": _get_overlay_library_assets(db),
            "voices": _get_voice_clips(db),
            # Language <option> list for a stitch clip's / voice's `language`
            # tag — the column the localisation New Run form matches against.
            "languages": languages.LANGUAGES,
            # Stated in the upload hint so the bounds the server enforces and
            # the bounds the operator is told are the same numbers.
            "voice_min_sec": ai_models.VOICE_REF_MIN_SEC,
            "voice_max_sec": ai_models.VOICE_REF_MAX_SEC,
        },
    )
