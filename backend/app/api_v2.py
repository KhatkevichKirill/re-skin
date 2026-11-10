"""
v2 REST API router for re-skin.

Mounted at /api/v2 from app/main.py.

Adds VideoProject (video + segmentation) and Run (one character per project)
endpoints. v1 /api/jobs endpoints are left completely untouched.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import subprocess
import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, List, Optional
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from sqlalchemy import Text, desc, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import ai_models
from . import project_types
from . import caption_placement as caption_placement_mod
from . import languages
from . import localisation
from . import localisation_jobs, localisation_prompt_job
from . import media as media_mod
from . import overlay_runs
from . import rebuild_paid_call
from . import spend as spend_mod
from .accel import accel_or_file_response
from .config import settings
from .db import get_db, get_session
from . import ops_gate
from .ops_gate import ADMISSION, GATED, Admission
from .gdrive_client import GDriveClient, GDriveDownloadError, extract_file_id
from .models import (
    AutomationLaunch,
    InsertExtraction,
    InsertMerge,
    LIBRARY_CLIP_KIND_INSERT,
    LIBRARY_CLIP_KINDS,
    LibraryClip,
    VoiceClip,
    OverlayJob,
    Run,
    RunAutomation,
    RunCaption,
    RunSegment,
    RunSegmentImageJob,
    RunAudioPatch,
    RunSegmentRangeAttempt,
    RunSegmentTake,
    RunTailEdit,
    SegmentDef,
    VideoProject,
)
from .pipeline_common import TAIL_EDIT_MIN_CUT_MARGIN_SEC as _MIN_CUT_MARGIN_SEC
from .pipeline_common import extract_project_frame, floor2 as _floor2
from .project_types import (
    DEFAULT_PROJECT_TYPE,
    LOCALISATION,
    LOCALISATION_SCOPE_HOOK,
    LOCALISATION_SCOPE_WHOLE,
    LOCALISATION_SCOPES,
    GENERATION_IMAGES_ONLY,
    REBUILD,
    REFRAME,
    SUBTITLE_REMOVAL,
    VALID_PROJECT_TYPES,
    derived_mute_source,
    is_rebuild,
    run_copies_are_useful,
    spec_for,
)
from .schemas_v2 import (
    DriveRelinkRequest,
    AutomationResponse,
    AutomationSettingsUpdate,
    CaptionCreateResponse,
    CaptionListItem,
    CaptionResponse,
    ExtractInsertResponse,
    LibraryClipResponse,
    LibraryClipUpdate,
    VoiceClipResponse,
    VoiceClipUpdate,
    LocalisationPromptJobAccepted,
    LocalisationPromptJobStatus,
    LocalisationPromptResponse,
    MergeCreateResponse,
    MergeListItem,
    MergeResponse,
    NewSegmentDef,
    ProjectCreateResponse,
    ProjectListItem,
    ProjectResponse,
    ProjectUpdate,
    AudioPatchListItem,
    AudioPatchResponse,
    RangeAttemptCreateResponse,
    RangeAttemptResponse,
    RunBatchCopyResponse,
    RunCreateResponse,
    RunListItem,
    RunResponse,
    RunUpdate,
    RunSegmentResponse,
    RecutResponse,
    SegmentDefResponse,
    SegmentImageJobCreateResponse,
    RunSegmentTakeResponse,
    SegmentTakesRestoreRequest,
    SegmentsRerunRequest,
    SegmentsUpdateRequest,
    TailEditCreateResponse,
    TailEditListItem,
    TailEditResponse,
    TranscriptResponse,
)
from .state_machine import (
    CaptionStatus,
    InvalidTransition,
    MergeStatus,
    OverlayStatus,
    ProjectStatus,
    AUDIO_PATCH_ACTIVE_STATUSES,
    AudioPatchStatus,
    RANGE_ATTEMPT_ACTIVE_STATUSES,
    RangeAttemptStatus,
    RunStatus,
    SegmentImageStatus,
    SegmentStatus,
    TailEditStatus,
    transition,
)
from .pipeline_provider import active_task_for_project, active_task_for_run
from .pipeline_reframe import reframe_project_busy
from .storage import (
    PathOutsideDataDir,
    caption_dir,
    data_dir_root,
    ensure_under_data_dir,
    library_clip_dir,
    path_within,
    voice_clip_dir,
    run_voice_dir,
    project_dir,
    project_source_path,
    range_attempt_dir,
    run_dir,
    segment_image_job_dir,
    segment_take_dir,
    staged_frame_name,
    tail_edit_dir,
)
from .pipeline_audio_patch import (
    AUDIO_PATCH_ELIGIBLE_RUN_STATUSES,
    AudioPatchValidationError,
    create_audio_patch,
)
from .pipeline_range_attempt import RangeValidationError, create_range_attempt
from .tasks import (
    enqueue_analyze_project,
    enqueue_export_caption_to_sheet,
    enqueue_export_or_rollback,
    enqueue_export_run_to_sheet,
    enqueue_export_tail_edit_to_sheet,
    enqueue_localisation_prompt,
    enqueue_preprocess_run,
    enqueue_process_caption,
    enqueue_process_merge,
    enqueue_process_audio_patch,
    enqueue_process_range_attempt,
    enqueue_process_run,
    enqueue_process_segment_image,
    enqueue_process_tail_edit,
    enqueue_transcribe_project,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["v2"])


# ---------------------------------------------------------------------------
# Daily kie.ai budget (AICP-155) — see app/spend.py. Every endpoint that starts
# NEW paid work reserves its estimate as the last step before its commit; an
# over-budget day answers 429 and nothing is created or queued.
# ---------------------------------------------------------------------------


def _run_segment_defs_for(run: Run) -> list:
    from .pipeline_v2 import run_segment_defs

    return run_segment_defs(run, run.project)


def _budget_exceeded(exc: "spend_mod.BudgetExceeded") -> HTTPException:
    return HTTPException(status_code=429, detail=exc.message)


def _spend_reserve(fn, *args, **kwargs):
    """Call a spend_mod reserve function, mapping a refusal to HTTP 429."""
    try:
        return fn(*args, **kwargs)
    except spend_mod.BudgetExceeded as exc:
        raise _budget_exceeded(exc) from exc


def _admit_analysis(project: VideoProject) -> Optional[str]:
    """Put a project whose analysis THIS request enqueues into ``analyzing``
    inside the admission transaction (AICP-350, design §2 "every gated route
    must leave counted work"): ``deploy/drain-check.sql`` counts ``analyzing``,
    so the gate's drain can never read zero between this commit and the job
    taking the project. ``pipeline_v2.analyze_project`` accepts a project that
    is already ``analyzing``; a job lost before it starts is reaped to
    ``failed`` by recovery (operator re-analyzes), never left in limbo.

    A rebuild project also gets the analysis token its job must carry (the
    same contract as rebuild re-analysis); returns it, else None.
    """
    project.status = ProjectStatus.analyzing
    project.error_message = None
    if is_rebuild(project.project_type):
        project.rebuild_analysis_token = str(uuid.uuid4())
        return project.rebuild_analysis_token
    return None


def _enqueue_analysis(project_id: str, analysis_token: Optional[str]) -> None:
    """Enqueue the analysis :func:`_admit_analysis` admitted (the token only
    for a rebuild project, as rebuild re-analysis does)."""
    if analysis_token:
        enqueue_analyze_project(project_id, analysis_token=analysis_token)
    else:
        enqueue_analyze_project(project_id)


def _budget_precheck(db: Session, kind: str, needed: float = 0.0) -> None:
    """Cheap, unlocked early refusal when today's budget is already spent (or
    cannot cover *needed* credits), so an over-budget request is turned away
    BEFORE it writes uploads to disk — some handlers save over an existing
    file of the same name. Advisory only — the reserve before commit is the
    real check; a concurrent admission can still win between the two."""
    summary = spend_mod.today_summary(db)
    remaining = summary["remaining_credits"]
    if remaining is None:
        return
    if summary["over"] or (needed > 0 and needed > remaining + 1e-9):
        raise _budget_exceeded(spend_mod.BudgetExceeded(
            day=summary["day"], charged=summary["charged_credits"],
            needed=needed, cap=summary["cap_credits"], kind=kind,
        ))

# Every resolution ANY model offers. Derived from the registry rather than
# written out, so adding a model with a new resolution can never leave this
# list stale. Per-model validation always uses ai_models.resolutions_for() —
# this union is only for "is this string a resolution at all" checks.
_VALID_RESOLUTIONS = frozenset(
    r for spec in ai_models.AI_MODELS.values() for r in spec.resolutions
)
_VALID_AUDIO_MODES = {"original", "seedance"}
# Run states that mean a worker may still be touching the run's files — block
# deletion while in any of these (delete would race the worker / rmtree live files).
#
# `preprocessing` belongs here for exactly that reason: the GPT Image 2 poll
# chain downloads into the run's own dir. recovery.py's matching set has always
# listed it; this one had not, so a run could be deleted (or its project
# re-analyzed) out from under a live preprocess worker. Rare while every
# preprocessing run paused for human approval; routine now that copies and
# automation launches pass through the phase unattended.
#
# `preprocess_review` is deliberately NOT here: no worker is touching that run,
# and it can sit there for days waiting on a person — blocking re-analysis on it
# would be a new way to get stuck.
_ACTIVE_RUN_STATUSES = {
    RunStatus.preprocessing,
    RunStatus.queued,
    RunStatus.processing,
    RunStatus.stitching,
    RunStatus.delivering,
}
# Max number of runs a single batch-copy request may launch at once.
_MAX_BATCH_COPY_RUNS = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_gdrive() -> GDriveClient:
    """Return a GDriveClient with default credentials (lazy-loaded service)."""
    return GDriveClient()


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


def _get_merge_or_404(merge_id: str, db: Session) -> InsertMerge:
    merge = db.get(InsertMerge, merge_id)
    if merge is None:
        raise HTTPException(status_code=404, detail=f"InsertMerge {merge_id!r} not found")
    return merge


def _get_tail_edit_or_404(edit_id: str, db: Session) -> RunTailEdit:
    edit = db.get(RunTailEdit, edit_id)
    if edit is None:
        raise HTTPException(status_code=404, detail=f"RunTailEdit {edit_id!r} not found")
    return edit


def _get_caption_or_404(caption_id: str, db: Session) -> RunCaption:
    caption = db.get(RunCaption, caption_id)
    if caption is None:
        raise HTTPException(status_code=404, detail=f"RunCaption {caption_id!r} not found")
    return caption


def _lock_caption_or_404(caption_id: str, db: Session) -> RunCaption:
    """Fresh, row-locked read for caption writes (AICP-345).

    Serialises the lines/retry endpoints with the assistant's Apply, which
    locks the run and then this row: a write that read the row before a
    concurrent one committed would otherwise overwrite it unseen.
    """
    caption = db.scalar(select(RunCaption).where(RunCaption.id == caption_id)
                        .with_for_update().execution_options(populate_existing=True))
    if caption is None:
        raise HTTPException(status_code=404, detail=f"RunCaption {caption_id!r} not found")
    return caption


def _validate_chained_source(
    db: Session,
    model: type,
    source_id: Optional[str],
    run_id: str,
    label: str,
):
    """Resolve an optional chained-source id for the §3.7.3 binding contract
    (docs/burned-captions.md) shared by ``create_tail_edit``'s
    ``source_caption_id`` and ``create_caption``'s ``source_tail_edit_id``.

    Blank/omitted *source_id* returns ``None`` — the caller then keeps
    ``source_type="run"``, byte-for-byte the v1 behaviour. Otherwise the
    referenced row must exist AND belong to *run_id* (400, "not found / not
    this run's" collapsed into one message — the caller doesn't need to
    distinguish them), and must be ``done`` with an ``output_path`` that
    still exists on disk (409, "not done / no file"). *model* is
    ``RunCaption`` or ``RunTailEdit`` — both expose ``run_id``, ``status``,
    and ``output_path`` in the same shape, so one helper covers both
    directions instead of duplicating the checks per endpoint.
    """
    if not source_id:
        return None
    row = db.get(model, source_id)
    if row is None or row.run_id != run_id:
        raise HTTPException(
            status_code=400,
            detail=f"Source {label} not found for this run: {source_id!r}",
        )
    status_val = _status_text(row.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Source {label} is not ready: status is {status_val!r}, "
                "expected 'done'"
            ),
        )
    if not row.output_path or not os.path.exists(row.output_path):
        raise HTTPException(
            status_code=409,
            detail=f"Source {label} has no output file on disk",
        )
    return row


def _status_text(value) -> str:
    """A status enum rendered as the plain string an operator should see.

    ``ProjectStatus``/``RunStatus`` are ``str`` Enums, but neither ``str()``
    nor ``!r`` gives back the bare value: they render the Python identity
    (``ProjectStatus.ready``, ``<ProjectStatus.ready: 'ready'>``). Interpolating
    one straight into an ``HTTPException`` detail therefore ships a Python repr
    into a browser alert — which is exactly what ``POST /transcribe`` used to
    do. Use this anywhere a status reaches a human or a JSON field.

    ``hasattr`` rather than ``isinstance`` because a legacy row can hand back a
    plain string for the same column.
    """
    return value.value if hasattr(value, "value") else str(value)


def _validate_project_type(project_type: str) -> None:
    """400 unless *project_type* is one of the registered flows.

    ``rebuild`` additionally needs ``REBUILD_ENABLED`` (409 while it is off,
    docs/rebuild-from-reference-design.md §7.6): this is the one place a
    project row gets its type (Create Project and Move-run-to-project), so no
    'rebuild' label can reach the table while the flag is off."""
    if project_type not in VALID_PROJECT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"project_type must be one of {sorted(VALID_PROJECT_TYPES)}",
        )
    if project_type == REBUILD and not settings.REBUILD_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="Rebuild from reference is not enabled on this server (REBUILD_ENABLED)",
        )


def _refuse_images_only_run(run: Run, what: str) -> None:
    """409 for an action that does not exist for an images-only (rebuild) run
    (design §6.6) — the server is the guard, the run page only hides it."""
    if spend_mod.is_images_only_run(run):
        raise HTTPException(
            status_code=409,
            detail=f"{what} is not supported for images-only (rebuild) runs",
        )


def _rebuild_repair_refused(exc) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def _require_rebuild_enabled_for_repair(run: Run) -> None:
    """A rerun / Retry of an images-only run is new paid work: refused while
    REBUILD_ENABLED is off (existing runs stay readable and recoverable)."""
    if spend_mod.is_images_only_run(run) and not settings.REBUILD_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="Rebuild from reference is not enabled on this server (REBUILD_ENABLED)",
        )


def _rebuild_live_rows_guard(db: Session, project: VideoProject, action: str) -> None:
    """409 while the project has live rebuild rows (design §5.1.7, §7.4).
    A no-op for every other project type."""
    if not is_rebuild(project.project_type):
        return
    msg = rebuild_paid_call.live_rows_message(
        rebuild_paid_call.project_live_rows(db, project.id)
    )
    if msg:
        raise HTTPException(status_code=409, detail=f"Cannot {action}: {msg}")


# Floor for the per-project segmentation cap, derived from the model registry.
#
# The tempting floor is ai_models.MIN_SWAP_VIDEO_SEC (2.0), but that bounds the
# *reference clip* — the shortest video a model accepts as INPUT. What bounds a
# segmentation cap is the shortest video a model will EMIT: every spec carries a
# min_duration_sec and ai_models.duration_for() clamps UPWARD to it, so a cap
# below the largest min_duration_sec in the table cannot be honoured by any
# backend. Accepting one silently multiplies spend — at cap=2.0 a 9s face
# interval is cut into five 1.8s chunks and each is still submitted asking for 4s
# of output, i.e. we generate and pay for 20s of video to cover 9s of source (at
# cap=3.0, 12s for 9s). Those chunks also fall under MIN_SWAP_VIDEO_SEC, so
# submit pads every clip back up and steals footage from its neighbour.
#
# Computed rather than written as 4.0 so adding a model with a different floor
# moves the bound with it. Kept local to the API layer because it exists to
# validate an operator-supplied setting, which no other module does; promote it
# into ai_models if a second consumer ever appears.
_MIN_SEGMENT_CAP_SEC = float(
    max(spec.min_duration_sec for spec in ai_models.V2V_MODELS.values())
)


def _max_segment_sec_in_range(value: float) -> bool:
    """True when *value* is a segmentation cap some backend can honour.

    Written as a chained comparison, not ``value < lo or value > hi``. Every
    comparison against NaN is False, so BOTH halves of the two-negation form were
    False and it ACCEPTED NaN — and NaN really does reach here: Pydantic v2 parses
    ``{"max_segment_sec": NaN}`` and the string ``"nan"`` into a float NaN by
    default, and Postgres DOUBLE PRECISION stores it happily, so it persists.
    Analysis then computes ``min(nan, ABSOLUTE_MAX_SEGMENT_SEC)`` (NaN wins),
    hands it to face.split_max_duration, and dies in ``math.ceil(dur / nan)``
    with "cannot convert float NaN to integer" — the project lands in ``failed``
    with an opaque message, and every re-analysis repeats it, because the cap is
    read fresh each time. The settings dropdown does at least surface the value
    (it compares the raw float, so NaN misses every option and falls into the
    "custom (set via the API)" branch), but nothing about it says "this is why
    analysis keeps failing". Framing the test as "is it inside the interval"
    instead of "is it outside either end" rejects NaN and both infinities for
    free.
    """
    return _MIN_SEGMENT_CAP_SEC <= value <= ai_models.ABSOLUTE_MAX_SEGMENT_SEC


def _validate_max_segment_sec(value: Optional[float]) -> Optional[float]:
    """Validate the per-project analyze-time segmentation cap, or 400.

    ``None`` is a real, meaningful value here — it means "no explicit choice",
    and analysis falls back to ai_models.UNIVERSAL_MAX_SEGMENT_SEC (the
    shortest max_clip_sec across every model, i.e. the cap that keeps a project
    runnable on ANY backend). So we return it untouched rather than coercing it
    to a number: storing the default explicitly would freeze today's value into
    every row and silently diverge if the registry ever changes.

    The accepted band is [_MIN_SEGMENT_CAP_SEC, ABSOLUTE_MAX_SEGMENT_SEC]: below
    the floor no backend can generate a clip that short, so the request is padded
    back up and the difference is billed (see _MIN_SEGMENT_CAP_SEC), and above
    the longest-clip model's ceiling every segment would be un-generatable.
    The UI only offers blank/15/30, but the REST API, automations and scripts/
    pass whatever they are given, which is how the bad values get in.
    """
    if value is None:
        return None
    lo = _MIN_SEGMENT_CAP_SEC
    hi = ai_models.ABSOLUTE_MAX_SEGMENT_SEC
    if not _max_segment_sec_in_range(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_segment_sec must be a finite number between {lo} and {hi} "
                f"seconds (got {value}); {lo}s is the shortest clip every model "
                "will generate (the largest min_duration_sec in the model "
                f"registry), so a cap below it is still billed as {lo}s per "
                f"segment however short the cut, and a cap above {hi}s can be "
                "generated by no model at all — omit it to use the "
                f"{ai_models.UNIVERSAL_MAX_SEGMENT_SEC}s default that every "
                "model can generate"
            ),
        )
    return float(value)


def _inherited_max_segment_sec(
    parent_value: Optional[float], *, context: str, parent_id: Optional[str]
) -> Optional[float]:
    """Carry a parent project's segmentation cap into a copy, or drop to NULL.

    The project-copy paths (extract-insert, run→project) inherit the parent's cap
    so the child is cut the same way, but the stored number is only as
    trustworthy as whatever wrote it: rows created before the floor above was
    tightened, a direct SQL edit, or the NaN that used to slip past validation.
    Re-checking here stops one bad row from being copied forward into every
    descendant project.

    It falls back to NULL rather than raising 400, unlike _validate_max_segment_sec.
    A 400 would be the wrong shape of failure on a copy: the caller never sent
    max_segment_sec — it is inherited state — so the error would blame them for a
    field absent from their own request, and it would block an operation whose
    actual subject (the crop, the promoted run) is perfectly valid. Worse, the
    only way out would be to first repair the parent row, i.e. one bad row would
    veto every copy of that lineage. NULL is the documented "no explicit choice"
    value and is generatable by every backend, so degrading to it keeps the copy
    working AND keeps the child correct — the child simply gets the universal
    default it would have had before inheritance existed. The parent is left
    untouched (this endpoint has no mandate to rewrite it) and the warning is the
    audit trail pointing at the row that still needs fixing via PATCH.
    """
    if parent_value is None:
        return None
    if not _max_segment_sec_in_range(parent_value):
        log.warning(
            "%s: parent project %s has an unusable stored max_segment_sec=%r "
            "(outside [%s, %s]); not propagating it — the child falls back to "
            "the %ss default",
            context,
            parent_id,
            parent_value,
            _MIN_SEGMENT_CAP_SEC,
            ai_models.ABSOLUTE_MAX_SEGMENT_SEC,
            ai_models.UNIVERSAL_MAX_SEGMENT_SEC,
        )
        return None
    return float(parent_value)


def _hook_sec_in_range(value: float) -> bool:
    """True when *value* is a usable localised-hook length.

    Written as a chained comparison for exactly the reason
    :func:`_max_segment_sec_in_range` is — every comparison against NaN is
    False, so the two-negation form (``value <= 0 or value == inf``) would
    ACCEPT NaN, and Pydantic really does parse ``NaN``/``"nan"`` into a float
    NaN that Postgres stores happily. A NaN hook then poisons the analyze-time
    clamp (``min``/``max`` propagate it) and the hook window of every
    translate call, both silently.

    Deliberately unbounded above: unlike the segmentation cap, a hook longer
    than the video or the cap is not an error — hook_split clamps it down to
    ``min(effective cap, duration)`` (docs/localisation.md §5), and a hook that
    covers the whole video is a legitimate "re-speak all of it" project. Only
    values that can never mean anything (zero, negative, NaN, infinite) are
    refused here.
    """
    return 0.0 < value < math.inf


def _validate_localisation_scope(
    value: Optional[str], project_type: Optional[str]
) -> Optional[str]:
    """Validate a requested ``localisation_scope`` (AICP-340), or 400.

    ``None`` / blank means "not specified" and is returned as None so the
    caller keeps whatever the row already has (``"hook"`` on a new row). A
    value is accepted only from :data:`project_types.LOCALISATION_SCOPES`, and
    ``"whole"`` only on a ``localisation`` project — every other type has no
    hook to lift, and silently storing the flag on one would be a setting that
    does nothing. ``"hook"`` is tolerated on any type (it is the column
    default), so a generic client that always sends it never 400s.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="localisation_scope must be a string")
    scope = value.strip().lower()
    if not scope:
        return None
    if scope not in LOCALISATION_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"localisation_scope must be one of {', '.join(LOCALISATION_SCOPES)} "
                f"(got {value!r})"
            ),
        )
    if scope == LOCALISATION_SCOPE_WHOLE and spec_for(project_type).key != LOCALISATION:
        raise HTTPException(
            status_code=400,
            detail=(
                "localisation_scope='whole' is only valid on a localisation project "
                f"(this one is {project_type!r})"
            ),
        )
    return scope


def _validate_hook_sec(value: Optional[float]) -> Optional[float]:
    """Validate the per-project localised-hook length, or 400.

    ``None`` is a real, meaningful value (mirroring
    :func:`_validate_max_segment_sec`): it means "no explicit choice", and both
    analysis and the translate endpoint fall back to
    ``settings.LOCALISATION_DEFAULT_HOOK_SEC``. It is returned untouched rather
    than resolved to a number, so the default stays in one place instead of
    being frozen into every row at creation time.

    ``0.0``, by contrast, is a VALUE and it is rejected: a zero-length hook
    would produce a swap segment no model can generate and a transcript window
    containing no speech. Blank form fields arrive as None (not 0.0), so the
    "operator left it empty" case never reaches this branch.
    """
    if value is None:
        return None
    if not _hook_sec_in_range(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"hook_sec must be a finite number greater than 0 seconds (got "
                f"{value}) — it is the length of the front-of-video hook a "
                "localisation project re-speaks; omit it (or send null) to use "
                f"the {settings.LOCALISATION_DEFAULT_HOOK_SEC}s default. A hook "
                "longer than the video is allowed: analysis clamps it to the "
                "video's duration and the segmentation cap"
            ),
        )
    return float(value)


def _validate_language(code: Optional[str]) -> Optional[str]:
    """Canonicalise a language code, or 400 when it isn't one of ours.

    Blank/absent/whitespace stays None (the "language-agnostic"/"never chosen"
    value, and the way an edit endpoint CLEARS the field); anything else must be
    one of app/languages.py's codes. An unknown code is refused rather than
    stored, because downstream it would not be recognised either — a run tagged
    ``nl`` renames as English with "nl" stuck in the filename core.
    """
    normalized = languages.normalize(code)
    if (code or "").strip() and normalized is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"language must be one of {sorted(languages.LANGUAGES)} "
                "(app/languages.py's closed set)"
            ),
        )
    return normalized


def _validate_clip_kind(kind: Optional[str]) -> str:
    """A library clip kind (AICP-381): blank/absent = ``insert``, else one of
    ``LIBRARY_CLIP_KINDS`` (400 otherwise)."""
    value = (kind or "").strip().lower() or LIBRARY_CLIP_KIND_INSERT
    if value not in LIBRARY_CLIP_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of {list(LIBRARY_CLIP_KINDS)}",
        )
    return value


def _safe_filename(name: str) -> str:
    """Sanitize an uploaded filename: keep only the basename, strip leading dots."""
    base = os.path.basename(name.replace("\\", "/"))
    base = base.lstrip(".")
    return base or "upload"


def _save_upload(upload: UploadFile, dest: str, max_bytes: int | None = None) -> None:
    """Write an UploadFile to *dest* using streaming 1-MiB chunks.

    Raises HTTPException(413) when *max_bytes* is set and the upload exceeds it.
    The partial file is removed before raising.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _CHUNK = 1 << 20  # 1 MiB
    written = 0
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = upload.file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    fh.close()
                    os.unlink(dest)
                    limit_mb = max_bytes >> 20
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload too large: limit is {limit_mb} MiB",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception:
        if os.path.exists(dest):
            os.unlink(dest)
        raise


def _normalize_partition(segments: list, duration: float, db: Session) -> None:
    """Normalize *segments* into a contiguous partition of [0, duration].

    Forgiving cursor-walk: ends are the authoritative boundaries; starts are
    derived from the running cursor. A segment whose duration collapses to <= 0
    (e.g. its neighbour was extended over it) is DROPPED (deleted) rather than
    rejected — so "shrink a segment to zero" behaves like deleting it, and the
    partition stays contiguous. The first kept segment starts at 0 and the last
    is extended to *duration* for full coverage. Indices are reassigned.

    Raises HTTPException(400) only if every segment would be dropped.
    """
    if not segments:
        return

    ordered = sorted(segments, key=lambda s: s.start_sec)
    EPS = 1e-6
    cursor = 0.0
    kept: list = []
    for seg in ordered:
        seg.start_sec = cursor
        end = min(seg.end_sec, duration)
        if end - cursor > EPS:
            seg.end_sec = end
            kept.append(seg)
            cursor = end
        else:
            # Collapsed (zero/negative duration) → drop it.
            db.delete(seg)

    if not kept:
        raise HTTPException(
            status_code=400,
            detail="No segments with positive duration remain after edits.",
        )

    # Ensure full coverage of [0, duration].
    kept[-1].end_sec = duration
    for i, seg in enumerate(kept):
        seg.index = i


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------


def _validate_project_mode(mode, project_type, raw_params, probe_project):
    """AICP-363 §2.1: (flow, normalised params) for an assistant project, or
    (None, None) for a manual one. 422 when assistant mode is not available."""
    mode = (mode or "manual").strip()
    if mode not in ("manual", "assistant"):
        raise HTTPException(status_code=400, detail="mode must be 'manual' or 'assistant'")
    if mode == "manual":
        if raw_params and raw_params.strip():
            raise HTTPException(status_code=400,
                                detail="autopilot_params are only for mode=assistant")
        return None, None
    from .autopilot import registry as ap_registry
    from .autopilot.contract import FlowParamsError

    if not settings.AUTOPILOT_ENABLED:
        raise HTTPException(status_code=422, detail="the autopilot is not enabled")
    flow = ap_registry.flow_for_project_type(project_type)
    if flow is None:
        raise HTTPException(
            status_code=422,
            detail=f"no autopilot flow exists for {project_type!r} projects yet",
        )
    params: dict = {}
    if raw_params and raw_params.strip():
        try:
            params = json.loads(raw_params)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="autopilot_params must be JSON") from None
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="autopilot_params must be a JSON object")
    try:
        params = flow.validate_params(probe_project, params)
    except FlowParamsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return flow, params


@router.post(
    "/projects",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectCreateResponse,
    dependencies=[GATED],
)
def create_project(
    video_file: Optional[UploadFile] = File(None),
    gdrive_link: Optional[str] = Form(None),
    project_type: str = Form(DEFAULT_PROJECT_TYPE),
    mute_source: bool = Form(False),
    max_segment_sec: Optional[float] = Form(None),
    hook_sec: Optional[float] = Form(None),
    localisation_scope: Optional[str] = Form(None),
    mode: str = Form("manual"),
    autopilot_params: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> ProjectCreateResponse:
    """Create a new VideoProject.

    Exactly one of *video_file* or *gdrive_link* must be provided.
    Analysis is enqueued immediately (ffprobe + segment proposal).

    *project_type* selects the flow (see app/project_types.py): ``face_swap``
    (default) detects faces and proposes swap/keep segments; the other types
    skip detection and seed one full-length keep segment for manual editing.

    *mute_source* ("remove original audio") makes every run of this project
    upload video-only swap clips to the AI model, so it generates audio from
    the prompt instead of reacting to the source soundtrack. Toggleable later
    via ``PATCH /projects/{pid}``.

    *max_segment_sec* caps how long a proposed segment may be. Omit it (NULL)
    to keep the historical ai_models.UNIVERSAL_MAX_SEGMENT_SEC default, which
    every model can generate; raise it (up to
    ai_models.ABSOLUTE_MAX_SEGMENT_SEC) to get fewer, longer segments for a
    long-clip model like Seedance 2.5 — at the cost of the project no longer
    being runnable on the shorter-clip backends. It is consumed by *analysis*,
    which is enqueued right here, so this is the moment that matters: a later
    PATCH will not re-cut the segments this call is about to produce.

    *hook_sec* is the ``localisation`` flow's equivalent knob: how long the
    front-of-video hook that gets re-spoken in another language is. Omit it
    (NULL) to use ``settings.LOCALISATION_DEFAULT_HOOK_SEC``. It carries the
    same analyze-time contract as *max_segment_sec* — hook_split segmentation
    lays down one swap segment over ``[0, hook_sec]`` and one keep segment over
    the rest right here, and a later PATCH does NOT re-cut them; the project
    has to be analyzed again for a new hook length to take effect. It is
    meaningless for the other project types, which never read it.

    *localisation_scope* (``hook`` default | ``whole``, localisation only,
    AICP-340) picks what that first analysis cuts: the hook split above, or
    all-swap segments at scene cuts over the whole video (``hook_sec`` is then
    not read). It can later change only together with a re-analysis
    (``POST /projects/{pid}/analyze?localisation_scope=``).
    """
    _validate_project_type(project_type)
    max_segment_sec = _validate_max_segment_sec(max_segment_sec)
    hook_sec = _validate_hook_sec(hook_sec)
    localisation_scope = (
        _validate_localisation_scope(localisation_scope, project_type)
        or LOCALISATION_SCOPE_HOOK
    )

    # AICP-363: "manual" (default, today's behaviour) or "assistant" (an
    # autopilot runs the type's flow). Validated before anything is written.
    flow, ap_params = _validate_project_mode(
        mode, project_type, autopilot_params,
        VideoProject(project_type=project_type, localisation_scope=localisation_scope,
                     mute_source=mute_source),
    )

    has_file = video_file is not None and video_file.filename
    has_link = gdrive_link is not None and gdrive_link.strip()

    if not has_file and not has_link:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of video_file or gdrive_link",
        )
    if has_file and has_link:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of video_file or gdrive_link, not both",
        )

    project_id = str(uuid.uuid4())

    max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
    if has_file:
        filename = _safe_filename(video_file.filename or "source.mp4")
        ext = os.path.splitext(filename)[-1].lstrip(".") or "mp4"
        src_path = project_source_path(project_id, ext)
        _save_upload(video_file, src_path, max_bytes=max_bytes)

        project = VideoProject(
            id=project_id,
            source_type="upload",
            source_ref=filename,
            source_local_path=src_path,
            project_type=project_type,
            mute_source=mute_source,
            max_segment_sec=max_segment_sec,
            hook_sec=hook_sec,
            localisation_scope=localisation_scope,
            status=ProjectStatus.created,
            mode=mode,
            # A NEW subtitle-removal project removes captions with VOID from
            # the source (2026-11-03, docs/segment-fill-batch-design.md §7).
            caption_removal_method=project_types.default_caption_removal_method(project_type),
        )
    else:
        link = gdrive_link.strip()
        project = VideoProject(
            id=project_id,
            source_type="gdrive",
            source_ref=link,
            project_type=project_type,
            mute_source=mute_source,
            max_segment_sec=max_segment_sec,
            hook_sec=hook_sec,
            localisation_scope=localisation_scope,
            status=ProjectStatus.created,
            mode=mode,
            # A NEW subtitle-removal project removes captions with VOID from
            # the source (2026-11-03, docs/segment-fill-batch-design.md §7).
            caption_removal_method=project_types.default_caption_removal_method(project_type),
        )

    # A localisation project is transcribed automatically once analysis
    # returns (tasks._enqueue_transcribe_after_analyze), so its transcript is
    # spoken for from this moment on — say so NOW rather than at the end of
    # analysis. Otherwise transcript_status is NULL for the whole analyze
    # window and NULL means two different things at once ("nobody asked for a
    # transcript" and "one is coming"), which is what made the panel render a
    # Transcribe button next to a job that was already promised — one click,
    # two paid model calls on the same video. With this, NULL means exactly
    # "never requested" and the UI can act on it without guessing.
    #
    # Only the status column is touched, and only at creation: the task
    # re-writes it anyway, and every other reader (POST /localisation-prompt,
    # the panel) already treats "pending" as "not translatable yet".
    # (Not for an assistant project: its autopilot launches the transcription
    # inside the approved budget — tasks._enqueue_transcribe_after_analyze.)
    if spec_for(project_type).key == LOCALISATION and mode != "assistant":
        project.transcript_status = _TRANSCRIPT_PENDING

    analysis_token = _admit_analysis(project)  # AICP-350: counted at commit
    db.add(project)
    if flow is not None:
        db.flush()
        from .autopilot.core import create_autopilot

        create_autopilot(db, project, flow, ap_params)
    db.commit()

    # Enqueue analysis — import at module level so monkeypatch targets app.api_v2.*
    _enqueue_analysis(project_id, analysis_token)

    log.info(
        "Created project %s source_type=%s project_type=%s mute_source=%s "
        "max_segment_sec=%s hook_sec=%s localisation_scope=%s",
        project_id, project.source_type, project_type, mute_source,
        # None is logged as-is (not as the resolved default) so the log shows
        # what was stored, matching what a later PATCH would see.
        max_segment_sec,
        hook_sec,
        localisation_scope,
    )
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)
    return ProjectCreateResponse(project_id=project_id, status=status_val)


@router.get("/projects", response_model=list[ProjectListItem])
def list_projects(
    limit: int = Query(default=100, ge=1, le=500, description="Max projects to return"),
    offset: int = Query(default=0, ge=0, description="Number of projects to skip"),
    db: Session = Depends(get_db),
) -> list:
    """Return projects, newest first.

    Supports LIMIT/OFFSET pagination: ``limit`` (default 100, max 500) and
    ``offset`` (default 0).
    """
    projects = (
        db.execute(
            select(VideoProject)
            .order_by(desc(VideoProject.created_at))
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return [ProjectListItem.model_validate(p) for p in projects]


@router.get("/projects/{pid}", response_model=ProjectResponse)
def get_project(pid: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """Return full project details."""
    project = _get_project_or_404(pid, db)
    return ProjectResponse.model_validate(project)


def _validate_caption_detector(value: Optional[dict]) -> Optional[dict]:
    """Validate a ``caption_detector`` PATCH value -> what to store, or 400."""
    if value is None:
        return None
    from .qa_tools import QAToolError, caption_config

    try:
        return caption_config(value).stored()
    except QAToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.patch("/projects/{pid}", response_model=ProjectResponse)
def update_project(
    pid: str, body: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """Update editable project settings: name, *mute_source*, *max_segment_sec*,
    *hook_sec*, *caption_detector*.

    *caption_detector* is read by the free QA caption detector
    (``app.qa_tools``), the subtitle-removal autopilot's review/QA, the VOID
    fill mask and caption placement; it changes nothing in the runs. A value
    without ``band_y`` lets the autopilot locate the captions; an explicitly
    given ``band_y`` (even the default) is kept as the operator's.

    *mute_source* is read when a run submits its swap clips, so toggling it
    changes every subsequent run of this project — including runs already
    queued but not yet past the submit phase. Runs whose clips are already
    uploaded are unaffected.

    *max_segment_sec*, by contrast, is read at ANALYZE time — the moment the
    segments were cut. Changing it here does **not** re-cut the segments this
    project already has and does not trigger a re-analysis; it only takes
    effect if the project is analyzed again. To actually get longer segments on
    an analyzed project, re-run analysis (or create the project with the cap
    you want). Editing the existing segment boundaries by hand remains the
    other way to get there.

    *hook_sec* (localisation projects) is analyze-time too, and for the same
    reason: hook_split segmentation already turned it into a swap segment over
    ``[0, hook_sec]`` plus a keep segment over the rest, and those SegmentDefs
    are now fixed. Changing it here re-cuts nothing — re-analyze the project,
    or drag the segment boundary by hand. The one thing it DOES change without
    a re-analysis is the default window ``POST /localisation-prompt`` slices
    the transcript to, and that endpoint takes its own per-request override
    anyway (which likewise re-segments nothing).
    """
    project = _get_project_or_404(pid, db)
    # AICP-363: project row lock first (fences `project_detector:` and the other
    # project version keys of the autopilot). AICP-368 relies on it too: taken
    # BEFORE any edit (a refresh discards unflushed changes), so the
    # caption-placement rev bumped below is the one a racing scan compares
    # against.
    db.refresh(project, with_for_update=True)
    if body.localisation_scope is not None:
        # Not a PATCH field on purpose (AICP-340): the scope must always name
        # the strategy that cut the CURRENT segments, because the prompt
        # window and the run form read it as a fact about them. Changing it
        # without re-cutting would make a hook partition translate the whole
        # transcript (or a whole-video partition translate only the hook).
        raise HTTPException(
            status_code=400,
            detail=(
                "localisation_scope changes only together with a re-analysis: "
                "POST /api/v2/projects/{pid}/analyze?localisation_scope=hook|whole"
            ),
        )
    if body.name is not None:
        name = body.name.strip()
        project.name = name[:255] or None
    if body.mute_source is not None:
        project.mute_source = body.mute_source
    # Unlike the other fields, NULL is a meaningful VALUE here (= "no opinion,
    # use ai_models.UNIVERSAL_MAX_SEGMENT_SEC"), not just "omitted". So this one
    # field distinguishes the two via model_fields_set: an explicit
    # `"max_segment_sec": null` clears the cap back to the default, while
    # leaving the key out entirely still leaves the column untouched. Without
    # this, picking the default option in the settings dropdown on a project
    # already set to 15s/30s would silently no-op and the old value would
    # reappear on reload.
    if "max_segment_sec" in body.model_fields_set:
        project.max_segment_sec = _validate_max_segment_sec(body.max_segment_sec)
    # Same NULL-is-a-value rule, same analyze-time contract (see the docstring).
    if "hook_sec" in body.model_fields_set:
        project.hook_sec = _validate_hook_sec(body.hook_sec)
    # QA caption detector config (AICP-343): validate, then store only the
    # non-default fields so an all-defaults object (or null) is NULL.
    if "caption_detector" in body.model_fields_set:
        project.caption_detector = _validate_caption_detector(body.caption_detector)
    # AICP-368: the operator's placement (kind "manual"); null = the sticky
    # house position, which a later derivation never overwrites. The row was
    # locked at the top of this handler.
    if "caption_placement" in body.model_fields_set:
        try:
            project.caption_placement = caption_placement_mod.manual(
                body.caption_placement, current=project.caption_placement
            )
        except caption_placement_mod.PlacementError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    # Batch VOID fills (2026-11-03): read by the fill admission under the
    # project row lock, which this handler holds.
    if "void_auto_usd_per_run" in body.model_fields_set:
        v = body.void_auto_usd_per_run
        if v is not None and (not math.isfinite(float(v)) or not 0 <= float(v) <= 20):
            raise HTTPException(status_code=400,
                                detail="void_auto_usd_per_run must be 0 .. 20 (USD) or null")
        project.void_auto_usd_per_run = None if v is None else round(float(v), 4)
    if "caption_removal_method" in body.model_fields_set:
        m = body.caption_removal_method
        if m not in project_types.CAPTION_REMOVAL_METHODS:
            raise HTTPException(status_code=400, detail=(
                "caption_removal_method must be one of %s"
                % ", ".join(project_types.CAPTION_REMOVAL_METHODS)))
        if project.project_type != SUBTITLE_REMOVAL:
            raise HTTPException(status_code=400, detail=(
                "caption_removal_method is a subtitle_removal setting"))
        project.caption_removal_method = m
    db.commit()
    db.refresh(project)
    log.info(
        "Updated project %s name=%r mute_source=%s max_segment_sec=%s "
        "hook_sec=%s caption_detector=%s (both the segmentation cap and the "
        "hook length apply to the NEXT analysis only — existing segments are "
        "unchanged)",
        pid, project.name, project.mute_source, project.max_segment_sec,
        project.hook_sec, project.caption_detector,
    )
    return ProjectResponse.model_validate(project)


@router.post("/projects/{pid}/caption-placement/derive", response_model=ProjectResponse, dependencies=[GATED])
def derive_project_caption_placement(pid: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """(Re)derive the project's caption placement from its subtitle-removal
    lineage (AICP-368, ``app.caption_placement``): writes the provisional
    value from the source run's delivered VOID fills now and enqueues the
    streamed detector scan that refines it. Replaces any current value,
    including an operator's — this is the explicit request. 409 without a
    subtitle-removal ancestor; 503 when the scan could not be enqueued (the
    provisional value stays, marked ``scan: pending``)."""
    project = _get_project_or_404(pid, db)
    try:
        rev = caption_placement_mod.request_derivation(db, project)
    except caption_placement_mod.PlacementError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if not caption_placement_mod.enqueue_scan(pid, rev):
        raise HTTPException(status_code=503, detail="caption placement scan could not be queued")
    db.refresh(project)
    return ProjectResponse.model_validate(project)


@router.delete("/projects/{pid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(pid: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a project: DB rows (cascades to segments/runs) + disk.

    Blocked (409) while the project is analyzing or any of its runs is active, so
    we never remove files a worker is still using.
    """
    project = _get_project_or_404(pid, db)
    # Lock order: the project row, then its runs (docs/fal-provider-design.md
    # §6) — the same project row a Fal admission locks, so the two serialize.
    db.refresh(project, with_for_update=True)
    _lock_runs_for_result_admission(db, db.scalars(
        select(Run.id).where(Run.project_id == pid).order_by(Run.id)
    ).all())
    if active_task_for_project(db, pid) is not None:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a project while one of its Fal provider tasks is "
                   "active (it may have been billed; close or abandon it first)",
        )
    reframe_busy = reframe_project_busy(db, pid)
    if reframe_busy:
        raise HTTPException(status_code=409, detail=f"Cannot delete the project: {reframe_busy}")
    _rebuild_live_rows_guard(db, project, "delete the project")
    if project.status == ProjectStatus.analyzing:
        raise HTTPException(
            status_code=409, detail="Cannot delete a project while it is analyzing"
        )
    active = db.execute(
        select(Run.id).where(
            Run.project_id == pid, Run.status.in_(_ACTIVE_RUN_STATUSES)
        )
    ).first()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a project while one of its runs is active",
        )

    db.delete(project)  # cascades to SegmentDef / Run / RunSegment
    db.commit()
    shutil.rmtree(project_dir(pid), ignore_errors=True)
    log.info("Deleted project %s (db + disk)", pid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/projects/{pid}/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProjectCreateResponse,
    dependencies=[GATED],
)
def reanalyze_project(
    pid: str,
    localisation_scope: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> ProjectCreateResponse:
    """Re-run analysis on an existing project (202 Accepted).

    This is the only way an analyze-time setting can be made to take effect
    after creation. ``max_segment_sec`` and ``hook_sec`` are both read when the
    segments are cut, so ``PATCH /projects/{pid}`` alone re-cuts nothing — the
    documented sequence is PATCH, then this.

    **Destructive to live SegmentDefs; NOT to any Run's own history.**
    ``pipeline_v2.analyze_project`` replaces the segmentation wholesale: it
    deletes every existing ``SegmentDef`` before writing the new proposal.
    ``RunSegment.segment_def_id`` is deliberately not a foreign key into that
    table (see its docstring in models.py — hardened after this endpoint's
    original destructive cascade turned out to also erase a Run's own
    correction history), so this delete no longer reaches any Run's rows.

    What SURVIVES:
      * every ``Run`` row — its prompt, refs, model, status and, for a finished
        run, ``result_local_path`` / the delivered Drive video. A ``done`` run
        stays done and its video stays playable and downloadable.
      * every ``RunSegment`` of every run, including finished ones — its
        status, clips, Kie task ids, per-segment overrides and the "re-run
        this segment" button all keep working exactly as before, because
        every reader resolves a RunSegment's timing/action from the Run's own
        frozen ``segment_snapshot`` first (falling back to a live
        ``SegmentDef`` only for a legacy run that predates that column).
      * every ``RunSegmentRangeAttempt`` — a completed run's range-correction
        history and its ability to accept another range regeneration.
      * the project's source file, transcript, name and settings.

    What does NOT survive:
      * every live ``SegmentDef`` — the ids change, so a NEW run launched
        after this uses entirely different segment ids than an old one did.
        A stored ``stitch_plan`` entry or per-segment ``prompt_override`` on
        an OLD run still names its OWN (now-orphaned) segment ids, which is
        fine — those are read from the run's own snapshot/RunSegment rows,
        never re-resolved against the live table.
      * a legacy run that predates ``Run.segment_snapshot`` and therefore
        still depended on reading live ``SegmentDef`` rows for its
        launch-window display — that display falls back to "no range shown",
        the same as it always has for a def that was simply deleted.

    Guards (409), mirroring ``DELETE /projects/{pid}`` — same reason, same
    statuses: analysis and the run pipeline both write the rows this deletes.
      * the project is already ``analyzing`` — a second analyze would race the
        first one's segment rewrite.
      * any run of the project is in a non-terminal state
        (``_ACTIVE_RUN_STATUSES``) — deleting its RunSegments mid-flight pulls
        the rows out from under the worker. ``created``, ``done``, ``failed``
        and ``incomplete`` runs do NOT block: nothing is processing them, and
        (since ``RunSegment.segment_def_id`` stopped being a foreign key —
        see its docstring in models.py) a ``failed``/``incomplete`` run's
        RunSegments survive this endpoint and its retry path keeps working
        exactly as before.

    A ``localisation`` project is additionally marked ``transcript_status =
    "pending"`` here, for the same reason project creation does: analysis
    chains a fresh transcription (``tasks._enqueue_transcribe_after_analyze``),
    so the transcript is spoken for from this moment and the panel must not
    offer a Transcribe button next to a job already queued. The previous
    transcript stays readable until the new one lands, but
    ``POST /localisation-prompt`` gates on the status and will 409 until then.

    *localisation_scope* (``hook`` | ``whole``, localisation only, AICP-340)
    is the ONLY way to change ``VideoProject.localisation_scope`` after
    creation. It is stored as ``localisation_scope_requested`` (after every
    guard above has passed) and ``analyze_project`` applies it to
    ``localisation_scope`` in the same commit as the new SegmentDefs, so the
    scope on the row always describes the segments on the row; a refused
    request writes nothing, and a failed analysis leaves the request for the
    retry. Omitted, a pending request (or else the current scope) is applied.
    """
    project = _get_project_or_404(pid, db)
    # AICP-363 §4.3: project row lock first (fences `project_segments:` keys).
    db.refresh(project, with_for_update=True)
    new_scope = _validate_localisation_scope(localisation_scope, project.project_type)
    if spec_for(project.project_type).key == REFRAME:
        # Its segmentation is a placeholder, and refusing re-analysis removes
        # the only path that rewrites the source after `ready`
        # (docs/reframe-design.md §7). A failed first analysis is fixed by
        # deleting and re-creating the project.
        raise HTTPException(
            status_code=409,
            detail="A reframe project is never re-analyzed; delete it and create "
                   "it again if its first analysis failed",
        )
    if project.status == ProjectStatus.analyzing:
        raise HTTPException(
            status_code=409,
            detail="Cannot re-analyze a project while it is already analyzing",
        )
    active = db.execute(
        select(Run.id).where(
            Run.project_id == pid, Run.status.in_(_ACTIVE_RUN_STATUSES)
        )
    ).first()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot re-analyze a project while one of its runs is active — "
                "an active run that has not yet reached its first submit has "
                "no frozen segment_snapshot yet and falls back to reading the "
                "project's LIVE segments; re-analysis deletes every segment "
                "definition out from under exactly that read. Wait for the "
                "run to finish and try again."
            ),
        )
    if is_rebuild(project.project_type):
        # Design §5.1.7–8: refused while a storyboard / keyframe / LLM call /
        # chunk attempt row is live, and the status flips to `analyzing` HERE,
        # under the project row lock, in the admission transaction — so no
        # rebuild admission (all of which require `ready` under the same lock)
        # can be made against the shot generation this analysis replaces.
        _rebuild_live_rows_guard(db, project, "re-analyze the project")
        transition(project, ProjectStatus.analyzing)
        # The job that runs from this admission owns the status; any older
        # job (e.g. one recovery reaped) is fenced out by the token.
        project.rebuild_analysis_token = str(uuid.uuid4())
        project.error_message = None
        db.commit()
        db.refresh(project)
        enqueue_analyze_project(pid, analysis_token=project.rebuild_analysis_token)
        log.info("Queued RE-analysis for rebuild project %s", pid)
        return ProjectCreateResponse(project_id=pid, status=_status_text(project.status))

    if spec_for(project.project_type).key == LOCALISATION:
        project.transcript_status = _TRANSCRIPT_PENDING
        project.transcript_error = None
        # The REQUEST, never localisation_scope itself: that one is written by
        # analyze_project in the same commit as the new segments. Writing it
        # here would describe segments that do not exist yet for as long as
        # the job is queued (a run launched in that window would freeze the
        # OLD segments under the NEW scope) or forever if analysis failed. The
        # request stays on the row until an analysis applies it, so a crashed
        # or failed analysis retried later still cuts what was asked for.
        if new_scope is not None:
            project.localisation_scope_requested = new_scope
    # AICP-350: `analyzing` in the admission commit, so the drain counts this
    # re-analysis from the moment it is admitted (and a second click 409s).
    transition(project, ProjectStatus.analyzing)
    project.error_message = None
    db.commit()
    db.refresh(project)

    enqueue_analyze_project(pid)
    log.info(
        "Queued RE-analysis for project %s (status=%s, project_type=%s, "
        "requested localisation_scope=%s) — "
        "existing SegmentDefs will be replaced (every Run's own RunSegments "
        "and RunSegmentRangeAttempt history survive — segment_def_id is not "
        "a foreign key into that table)",
        pid, _status_text(project.status), project.project_type,
        new_scope,
    )
    return ProjectCreateResponse(
        project_id=pid, status=_status_text(project.status)
    )


@router.get("/projects/{pid}/segments", response_model=list[SegmentDefResponse])
def get_project_segments(pid: str, db: Session = Depends(get_db)) -> list:
    """Return all SegmentDefs for a project, ordered by index."""
    _get_project_or_404(pid, db)
    segments = (
        db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == pid)
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    )
    return [SegmentDefResponse.model_validate(s) for s in segments]


@router.patch("/projects/{pid}/segments", response_model=list[SegmentDefResponse])
def update_project_segments(
    pid: str,
    body: SegmentsUpdateRequest,
    db: Session = Depends(get_db),
) -> list:
    """Edit SegmentDefs while project is in ready status.

    Blocked (409) while any run of the project is active, for the same reason
    ``reanalyze_project`` is blocked — see the guard below.
    """
    project = _get_project_or_404(pid, db)
    # AICP-363 §4.3: the project row lock fences `project_segments:` version
    # keys against an autopilot launch in flight (lock order: project first).
    # It also serialises with POST /recut-at-pauses (AICP-347), which locks
    # the same row before reading the segments it re-cuts: without it an edit
    # committed between the re-cut's read and write would be overwritten, or
    # would land on bounds planned with the old pre/post-roll.
    db.refresh(project, with_for_update=True)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit segments: project status is {project.status!r}, "
                "expected 'ready'"
            ),
        )

    # Historically an active run's RunSegment rows were FK'd to segment_defs
    # with ON DELETE CASCADE, so editing here deleted those rows out from
    # under the worker mid-flight — post-mortem: run fdc49979, 2026-08-12.
    # That FK is gone (see RunSegment.segment_def_id's docstring in
    # models.py) and an ALREADY-snapshotted run's own RunSegments can no
    # longer be destroyed this way, cascade or not.
    #
    # This guard stays regardless: a run that has not yet reached its first
    # submit has no frozen segment_snapshot yet and reads project.segments
    # LIVE at that moment (pipeline_v2.run_segment_defs's fallback) — editing
    # concurrently could hand it a half-old/half-new partition, or a def id
    # this edit is about to renumber/delete out from under that read. It is
    # also simply not a useful thing for an operator to do mid-run: the
    # in-flight run is generating against boundaries a concurrent edit is
    # about to change, so blocking is the predictable behaviour whether or
    # not any particular status could theoretically be proven safe today.
    active = db.execute(
        select(Run.id, Run.name).where(
            Run.project_id == pid, Run.status.in_(_ACTIVE_RUN_STATUSES)
        )
    ).all()
    if active:
        names = ", ".join((name or rid[:8]) for rid, name in active)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit segments while a run is in flight ({names}). "
                "The run is generating against these exact segment boundaries "
                "— editing them now is unpredictable for a run that has not "
                "yet frozen its own timeline. Wait for it to finish, then edit."
            ),
        )

    # Load all current segments into a dict by id
    segments: dict[str, SegmentDef] = {
        s.id: s
        for s in db.execute(
            select(SegmentDef).where(SegmentDef.project_id == pid)
        )
        .scalars()
        .all()
    }

    # Apply updates
    for upd in body.updates:
        seg = segments.get(upd.id)
        if seg is None:
            raise HTTPException(
                status_code=404, detail=f"SegmentDef {upd.id!r} not found"
            )
        for field, val in upd.model_dump(exclude={"id"}, exclude_none=True).items():
            setattr(seg, field, val)

    # Apply deletes
    for seg_id in body.deletes:
        seg = segments.pop(seg_id, None)
        if seg is None:
            raise HTTPException(
                status_code=404, detail=f"SegmentDef {seg_id!r} not found"
            )
        db.delete(seg)

    # Apply creates
    for new_seg in body.creates:
        seg = SegmentDef(
            id=str(uuid.uuid4()),
            project_id=pid,
            index=0,  # will be renumbered
            **new_seg.model_dump(),
        )
        db.add(seg)
        db.flush()
        segments[seg.id] = seg

    # Normalize to a contiguous partition [0, duration]
    remaining = list(segments.values())
    duration = project.duration_sec
    if duration is None:
        # Fallback: just renumber without partition enforcement
        ordered = sorted(remaining, key=lambda s: s.start_sec)
        for i, seg in enumerate(ordered):
            seg.index = i
    else:
        _normalize_partition(remaining, duration, db)

    db.commit()

    updated = (
        db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == pid)
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    )
    return [SegmentDefResponse.model_validate(s) for s in updated]


@router.post("/projects/{pid}/recut-at-pauses", response_model=RecutResponse)
def recut_project_at_pauses(
    pid: str,
    dry_run: bool = Query(False),
    expected_fingerprint: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> RecutResponse:
    """Move whole-video segment boundaries off spoken lines (AICP-347).

    An explicit operator action — never automatic, so the Analyze cut stays
    reproducible. Each boundary moves to a pause between transcript lines
    (preferring a scene cut inside it) that keeps every swap segment within
    the whole-video target, or stays and is reported in ``warnings``. The rule
    is :func:`app.speech_recut.recut_at_speech_pauses`.

    ``dry_run=true`` only computes the plan. An apply writes the new bounds
    onto the existing SegmentDef rows IN PLACE in one commit (ids, swap/keep,
    pre/post-roll and order stay; the segment count never changes). With
    ``expected_fingerprint`` (the preview's ``fingerprint``) it refuses with
    409 if the segments, transcript, target or recorded scene cuts changed
    since the preview.

    The work is :func:`app.speech_recut.recut_project_segments`, which is
    transaction-neutral (the autopilot calls it inside its own transaction);
    this endpoint owns the commit, the rollback and the 409.

    409 unless the project is a ``localisation`` project analyzed in whole
    scope, ``ready``, with a ``ready`` transcript; and — the guards of
    ``PATCH /segments`` and Re-analyze — while any of its runs is active. A
    queued re-analysis sets the transcript to ``pending`` and so blocks this
    too. Runs keep their own ``segment_snapshot``; the translate job's
    fingerprint hashes the swap bounds, so a translation computed for the old
    cut goes stale.
    """
    from .speech_recut import RecutRefused, recut_project_segments

    project = _get_project_or_404(pid, db)
    try:
        outcome = recut_project_segments(
            db, project, dry_run=dry_run, expected_fingerprint=expected_fingerprint,
        )
    except RecutRefused as exc:
        db.rollback()  # release the row locks of a refused apply
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    plan = outcome.plan
    if outcome.applied:
        db.commit()
        log.info(
            "Re-cut project %s at speech pauses: %s", pid,
            ", ".join(
                f"{b.old_sec:.2f}->{b.new_sec:.2f}"
                for b in plan.boundaries if b.moved
            ),
        )
    else:
        db.rollback()  # nothing was written; release the row locks

    return RecutResponse(
        project_id=pid,
        dry_run=dry_run,
        applied=outcome.applied,
        fingerprint=outcome.fingerprint,
        feasible=plan.feasible,
        changed=plan.changed,
        target_sec=plan.target_sec,
        floor_sec=plan.floor_sec,
        scene_cuts_used=plan.scene_cuts_used,
        boundaries=plan.to_dict()["boundaries"],
        warnings=outcome.warnings,
        segments=[SegmentDefResponse.model_validate(s) for s in outcome.segments],
    )


@router.get("/projects/{pid}/frame")
def get_project_frame(
    pid: str,
    t: float = Query(0.0, description="Timestamp in seconds"),
    db: Session = Depends(get_db),
) -> Response:
    """Extract a single JPEG frame from the project's source video at time *t* seconds.

    Cached as ``frames/frame_<t_tenths>.jpg`` (quantised to 0.1 s) inside the
    project directory.  Returns 404 if the project or its source video file is
    not found on disk.

    Fix #11 — two concurrent requests for the same frame no longer each spawn
    their own ffmpeg process.  A per-path threading.Lock ensures that only the
    first request runs ffmpeg; subsequent requests for the same cache_path wait
    on the lock, then serve the already-written file.

    Cache keys are quantised to 0.1-second resolution so that requests with
    slightly different float timestamps share the same cache entry (the
    templates formerly used different rounding: ``start_sec|int`` in
    project_status_content.html vs ``start_sec`` (float) in
    run_status_content.html).
    """
    cache_path, _t_norm = _extract_project_frame_or_404(pid, t, db)
    with open(cache_path, "rb") as fh:
        data = fh.read()
    return Response(content=data, media_type="image/jpeg")


def _extract_project_frame_or_404(pid: str, t: float, db: Session) -> tuple[str, float]:
    """``pipeline_common.extract_project_frame``, translated to HTTPException.

    Shared by the frame-preview endpoint (returns the bytes) and the
    stage-frame endpoint (copies the cached file into a run's references).
    The worker-side caller (``pipeline_segment_image``) calls
    ``extract_project_frame`` directly instead — a plain exception there
    becomes a job's ``failed`` status, not an HTTP response.
    """
    project = _get_project_or_404(pid, db)
    try:
        return extract_project_frame(project, t)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=500, detail="ffmpeg timed out") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/projects/{pid}/stage-frame")
def stage_project_frame(
    pid: str,
    t: float = Query(0.0, description="Timestamp in seconds"),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Extract a single JPEG frame from the project's source video and stage it
    as a reference image for a future Run.

    Used by the "Add frame" button in the New Run form (see
    docs/gpt-image-2-preprocess.md): the operator picks a timestamp, this
    endpoint extracts the frame (reusing the same cache as the frame-preview
    endpoint) and copies it into ``projects/<pid>/staged_frames/`` so it
    persists. The returned local path is appended to the form's reference URLs
    and treated as an ordinary reference image (no special model field).

    Returns ``{"path": "<local path>", "t": <quantised timestamp>}``.
    """
    # t_norm comes back from the extractor, not from the request: a timestamp
    # past the end of the video is clamped there, and the staged frame must be
    # named after the moment it is actually from.
    cache_path, t_norm = _extract_project_frame_or_404(pid, t, db)

    staged_dir = os.path.join(project_dir(pid), "staged_frames")
    os.makedirs(staged_dir, exist_ok=True)
    # Unique name so repeated "Add frame" at the same timestamp don't collide.
    # The timestamp lives in the name and nowhere else — storage.py owns both
    # halves of that convention so the UI can read it back off a run's
    # reference list (see storage.staged_frame_timestamp).
    name = staged_frame_name(t_norm)
    dest = os.path.join(staged_dir, name)
    shutil.copy2(cache_path, dest)

    log.info("Staged frame for project %s at t=%.1f → %s", pid, t_norm, dest)
    return JSONResponse(content={"path": dest, "t": t_norm})


@router.get("/projects/{pid}/source")
def get_project_source(pid: str):
    """Stream the project's original source video (for in-page review/seeking).

    Range requests work in both FileResponse (dev) and nginx accel (prod) mode.
    404 if the project or its source file is not on disk.

    Fix #5: when NGINX_ACCEL_ENABLED, returns X-Accel-Redirect so nginx serves
    the bytes directly instead of streaming through Python.
    Fix #6: DB session is scoped to a plain `with` block so it's committed and
    closed before the (potentially long) file transfer, freeing the pool
    connection immediately.
    """
    with get_session() as db:
        project = _get_project_or_404(pid, db)
        src = project.source_local_path
        if not src or not os.path.exists(src):
            raise HTTPException(status_code=404, detail="Source video not available on disk")

    return accel_or_file_response(src, media_type="video/mp4", filename="source.mp4")


@router.get("/projects/{pid}/runs", response_model=list[RunListItem])
def list_project_runs(pid: str, db: Session = Depends(get_db)) -> list:
    """Return all runs for a project."""
    _get_project_or_404(pid, db)
    runs = (
        db.execute(
            select(Run)
            .where(Run.project_id == pid)
            .order_by(desc(Run.created_at))
        )
        .scalars()
        .all()
    )
    items = []
    for run in runs:
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        items.append(
            RunListItem(
                id=run.id,
                name=run.name,
                status=status_val,
                created_at=run.created_at,
                result_available=(
                    status_val == "done"
                    and bool(run.result_local_path)
                    and os.path.exists(run.result_local_path)
                ),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Localisation — transcript + translated-prompt assembly
# (docs/localisation.md §4.4)
# ---------------------------------------------------------------------------

# Transcript lifecycle values (VideoProject.transcript_status). Only "ready"
# unlocks translation; "empty" is a SUCCESS (no speech in the clip) and
# "failed" is non-fatal — the project stays usable and the operator pastes a
# translation by hand, exactly as before this feature existed.
_TRANSCRIPT_READY = "ready"
_TRANSCRIPT_EMPTY = "empty"
_TRANSCRIPT_PENDING = "pending"

# The "will this line fit" yardstick lives in app.localisation, because the
# translator is now given the same character budget and re-asked when it blows
# it; the operator warnings below must agree with what the translator was told.
_CHARS_PER_SEC = localisation.CHARS_PER_SEC
_DEFAULT_CHARS_PER_SEC = localisation.DEFAULT_CHARS_PER_SEC
_SPEECH_OVERRUN_RATIO = localisation.SPEECH_OVERRUN_RATIO
_SPEECH_UNDERRUN_RATIO = localisation.SPEECH_UNDERRUN_RATIO

# Slack, in seconds, when comparing segment boundaries against the hook window.
# Segment times are floats that have been through a clamp and an operator's
# drag; a 30-millisecond sliver of "uncovered" hook is a rounding artefact, not
# a gap worth an alert, and no dialogue fits in one either.
_COVERAGE_EPS = 0.05


def _transcript_response(project: VideoProject) -> TranscriptResponse:
    """Project → the {status, error, transcript} payload all three endpoints return."""
    return TranscriptResponse(
        status=project.transcript_status,
        error=project.transcript_error,
        transcript=project.transcript,
    )


def _transcript_number(value, where: str) -> float:
    """Coerce a timestamp from an operator-supplied transcript, or 400.

    Rejects bools (``isinstance(True, int)`` is True in Python, and a
    ``"start": true`` would otherwise be stored as 1.0), non-numbers, NaN and
    the infinities — NaN in particular would sail through every ``<`` in
    :func:`app.localisation.slice_lines` and silently drop the line from every
    hook window, with nothing to show for it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(
            status_code=400, detail=f"{where} must be a number (seconds)"
        )
    number = float(value)
    if not math.isfinite(number):
        raise HTTPException(
            status_code=400, detail=f"{where} must be a finite number (got {value})"
        )
    if number < 0:
        raise HTTPException(
            status_code=400,
            detail=f"{where} must be >= 0 — timestamps are seconds from the clip start",
        )
    return number


def _transcript_text(value, where: str) -> str:
    """Require a non-empty string field, or 400."""
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=400, detail=f"{where} must be a non-empty string"
        )
    return value.strip()


def _validate_transcript_payload(body) -> dict:
    """Validate an operator-edited transcript against §4.1, or 400.

    The transcript is an INPUT to the translation prompt, not a pipeline
    artefact, which is why it is hand-editable at all — but it is also the only
    thing standing between a typo and a run prompt, so garbage is refused here
    rather than stored and discovered later by ``slice_lines`` returning
    nothing (or by Seedance being handed an empty dialogue block).

    Enforced: every line carries an integer ``id`` >= 1, unique across the
    transcript (translation rejoins on it — a duplicate would make one line
    overwrite another); ``start``/``end`` are finite, non-negative and ordered;
    ``text`` and ``speaker`` are non-empty. ``speaker`` matters more than it
    looks: it is handed to Seedance, which has to map the line onto a person it
    can see, so a blank label is a broken prompt, not a cosmetic gap.

    Normalised, not enforced: ``on_screen`` is coerced to a bool (default
    False) — it is a display hint the UI may legitimately omit; strings are
    stripped. Line ORDER is preserved exactly as sent rather than re-sorted by
    timestamp: the ids are the operator's, and ``format_dialogue`` merges
    consecutive same-speaker lines, so re-ordering would silently change the
    prompt they just proof-read.

    Unknown top-level keys (``model``, ``prompt_version``, ``created_at``, …)
    are passed through untouched so a round-trip edit does not strip the
    envelope that records which model produced the transcript.
    ``video_summary`` / ``scene_context`` are normalised to stripped strings
    (omitted or null → ``""``); a non-string value is a 400 rather than a
    silent ``str()``.
    """
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail=(
                "transcript must be a JSON object with a 'lines' array "
                "(docs/localisation.md §4.1)"
            ),
        )

    raw_lines = body.get("lines")
    if not isinstance(raw_lines, list):
        raise HTTPException(
            status_code=400,
            detail=(
                "transcript.lines must be an array — send [] for a clip with no "
                "speech (a legal outcome, stored as transcript_status='empty')"
            ),
        )

    seen_ids: set[int] = set()
    lines: list[dict] = []
    for position, item in enumerate(raw_lines):
        where = f"lines[{position}]"
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"{where} must be an object")

        line_id = item.get("id")
        if isinstance(line_id, bool) or not isinstance(line_id, int) or line_id < 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{where}.id must be an integer >= 1 — ids are how a "
                    "translation is rejoined to its source line"
                ),
            )
        if line_id in seen_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{where}.id={line_id} is a duplicate; line ids must be "
                    "unique within a transcript"
                ),
            )
        seen_ids.add(line_id)

        start = _transcript_number(item.get("start"), f"{where}.start")
        end = _transcript_number(item.get("end"), f"{where}.end")
        if end < start:
            raise HTTPException(
                status_code=400,
                detail=f"{where}: start ({start}) must be <= end ({end})",
            )

        cleaned_line = {
            "id": line_id,
            "start": start,
            "end": end,
            "speaker": _transcript_text(item.get("speaker"), f"{where}.speaker"),
            "on_screen": bool(item.get("on_screen")),
            "text": _transcript_text(item.get("text"), f"{where}.text"),
        }
        # v3 (AICP-335): per-line face visibility survives a hand edit; a
        # malformed value is dropped, never refused — it only decorates the
        # prompt.
        cleaned_line.update(localisation.clean_line_visibility(item))
        lines.append(cleaned_line)

    raw_on_screen = body.get("on_screen_text")
    if raw_on_screen is None:
        raw_on_screen = []
    if not isinstance(raw_on_screen, list):
        raise HTTPException(
            status_code=400, detail="transcript.on_screen_text must be an array"
        )
    on_screen_text: list[dict] = []
    for position, item in enumerate(raw_on_screen):
        where = f"on_screen_text[{position}]"
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"{where} must be an object")
        start = _transcript_number(item.get("start"), f"{where}.start")
        end = _transcript_number(item.get("end"), f"{where}.end")
        if end < start:
            raise HTTPException(
                status_code=400,
                detail=f"{where}: start ({start}) must be <= end ({end})",
            )
        on_screen_text.append(
            {
                "start": start,
                "end": end,
                "text": _transcript_text(item.get("text"), f"{where}.text"),
            }
        )

    source_language = body.get("source_language")
    if source_language is not None and not isinstance(source_language, str):
        raise HTTPException(
            status_code=400,
            detail="transcript.source_language must be a string (ISO 639-1 code)",
        )

    video_summary = _optional_transcript_context(
        body.get("video_summary"), "transcript.video_summary"
    )
    scene_context = _optional_transcript_context(
        body.get("scene_context"), "transcript.scene_context"
    )

    cleaned = dict(body)
    cleaned["schema_version"] = body.get(
        "schema_version", localisation.TRANSCRIPT_SCHEMA_VERSION
    )
    cleaned["source_language"] = (source_language or "").strip().lower()
    cleaned["video_summary"] = video_summary
    cleaned["scene_context"] = scene_context
    cleaned["lines"] = lines
    cleaned["on_screen_text"] = on_screen_text
    # v3 prompt metadata (AICP-335) — normalised when present, absent stays
    # absent (the prompt builder falls back without it).
    for key, clean in (
        ("speakers", localisation.clean_speakers),
        ("actions", localisation.clean_actions),
        ("brands", localisation.clean_brands),
    ):
        if key in body:
            cleaned[key] = clean(body.get(key))
    return cleaned


def _optional_transcript_context(value, where: str) -> str:
    """Strip a transcript context string, or 400 on a non-string value.

    Omitted / null → ``""`` so a v1 envelope (or a UI that only edits lines)
    still validates. An explicit array, object, number or bool is refused —
    silently stringifying those would invent context the operator never typed.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HTTPException(
            status_code=400,
            detail=f"{where} must be a string",
        )
    return value.strip()


def _intended_hook_sec(project: VideoProject) -> float:
    """The operator's stored hook INTENT, in seconds (never the analyzed one).

    The stored column, or ``settings.LOCALISATION_DEFAULT_HOOK_SEC`` when it is
    NULL. A stored value that is out of range (a NaN or a 0 from a direct SQL
    edit or a row written before validation existed) degrades to the default
    with a log line instead of raising, mirroring
    :func:`_inherited_max_segment_sec`: the caller never sent it, so blaming
    their request would be the wrong shape of failure, and a translation
    against the default window is far more useful than a 400 they cannot act
    on.
    """
    if getattr(project, "localisation_scope", None) == LOCALISATION_SCOPE_WHOLE:
        # Whole-video scope (AICP-340): the "hook" is the whole video, so the
        # translation window is every swap segment analysis cut. The probed
        # duration is the honest bound; hook_sec is not read in this scope.
        duration = project.duration_sec
        if duration is not None and _hook_sec_in_range(duration):
            return float(duration)
    stored = project.hook_sec
    if stored is not None:
        if _hook_sec_in_range(stored):
            return float(stored)
        log.warning(
            "Project %s has an unusable stored hook_sec=%r; treating the hook "
            "intent as the %ss default instead",
            project.id, stored, settings.LOCALISATION_DEFAULT_HOOK_SEC,
        )
    return float(settings.LOCALISATION_DEFAULT_HOOK_SEC)


def _analyzed_hook_sec(swap_segments: list, intent: float) -> Optional[float]:
    """Where the swap coverage inside ``[0, intent)`` actually ENDS, or None.

    "Actually" is the whole point: ``pipeline_v2._hook_split_segments`` clamps
    the hook down to ``min(segmentation cap, duration)`` at analyze time, and
    the operator may since have dragged a boundary in the Segment Editor —
    neither writes back to ``VideoProject.hook_sec``. So the row can say 15
    while the clip that will be generated is ``swap[0,10]``.

    Measured as the end of the last swap segment that starts inside the intent
    window, capped at the intent. Max-end rather than "the first gap" on
    purpose: a hand-split hook like ``swap[0,3] + keep[3,5] + swap[5,10]`` is
    still a 10-second hook with a hole in it, and shrinking the window to 3s
    would silently orphan the second swap segment (it would drop out of
    :func:`_hook_swap_segments`, get no per-segment prompt, and fall back to
    re-speaking the whole hook). The hole is reported by the coverage warning
    instead, which is where an operator can act on it.

    None when no swap segment overlaps the window at all — an unanalyzed
    project, or one whose hook was flipped to keep. The caller falls back to
    the intent, and the "nothing will be generated" warning fires.
    """
    ends = [
        float(s.end_sec)
        for s in swap_segments
        if float(s.start_sec) < intent and float(s.end_sec) > 0.0
    ]
    if not ends:
        return None
    return min(max(ends), intent)


def _resolve_hook_sec(
    project: VideoProject, override: Optional[float], swap_segments: list
) -> float:
    """Which hook window to slice the transcript to, in seconds.

    Three different numbers can call themselves "the hook length", and mixing
    them up is what made a 15s script get written for a 10s clip. They are, in
    the order this function consults them:

    1. **The per-call override** (``hook_sec`` on the ``localisation-prompt``
       form) — highest precedence, applies to this one call, is never
       persisted, and re-segments nothing. It exists so an operator can
       retranslate against a wider or narrower window without touching the
       project. Because it deliberately ignores reality, it keeps its warning
       when it asks for more than the analyzed swap segments can say.
    2. **The analyzed reality** (:func:`_analyzed_hook_sec`) — the DEFAULT
       window. What the swap SegmentDefs actually cover is what a run will
       actually generate, so it is the honest window to cut a script to. It is
       derived, never stored.
    3. **The stored intent** (``VideoProject.hook_sec``, via
       :func:`_intended_hook_sec`) — what the operator asked for at creation
       or in a PATCH. It bounds (2) and it is what the NEXT analysis will cut
       to, but on its own it says nothing about the clips that exist today. It
       is the fallback when the project has no swap segments to measure.

    So a project created with Hook length 15 under the default 10s segmentation
    cap keeps ``hook_sec=15`` in the column (raise the cap and re-analyze and
    it will finally get 15), while this endpoint cuts 10s of transcript,
    because ``swap[0,10]`` is the clip that will speak it.
    """
    if override is not None:
        return float(override)
    intent = _intended_hook_sec(project)
    analyzed = _analyzed_hook_sec(swap_segments, intent)
    if analyzed is None:
        return intent
    if abs(analyzed - intent) > _COVERAGE_EPS:
        log.info(
            "Project %s: hook intent is %.2fs but the analyzed swap segments "
            "end at %.2fs — slicing the transcript to the analyzed window",
            project.id, intent, analyzed,
        )
    return analyzed


def _estimated_speech_sec(text: str, language: Optional[str]) -> float:
    """Crude estimate of how long *text* takes to say in *language*."""
    return localisation.estimated_speech_sec(text, language)


def _line_seconds(line: dict, key: str) -> float:
    """A transcript line's timestamp, tolerating whatever is in the column.

    ``PATCH /transcript`` guarantees floats and the transcription task
    normalises them, but the column is free JSON and predates neither guard, so
    the advisory pass degrades to 0.0 rather than 500-ing on a legacy row —
    :func:`app.localisation.slice_lines` is equally forgiving for the same
    reason.
    """
    try:
        return float(line.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _line_midpoint(line: dict) -> float:
    """The instant a transcript line is considered to happen at.

    The midpoint, because that is the one point of a line that belongs to a
    single segment no matter how the boundaries fall — see
    :func:`_assign_lines_to_segments`. A zero-length (or reversed) line
    degrades to its start.
    """
    start = _line_seconds(line, "start")
    end = _line_seconds(line, "end")
    return start if end <= start else (start + end) / 2.0


def _assign_lines_to_segments(
    lines: list[dict], segments: list
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Give every line to EXACTLY ONE swap segment. Returns (by_segment, dropped).

    Each line goes to the segment whose ``[start_sec, end_sec)`` contains the
    line's MIDPOINT; lines whose midpoint lands in no segment come back in
    *dropped*. Segments are consulted in the order given (index order), so
    should two ever overlap the earlier one wins and the result stays
    deterministic.

    Why not ``localisation.slice_lines`` — the function that cuts the hook out
    of the full transcript? Because its semantics are OVERLAP, which is right
    for one window and wrong for a partition:

    * ``swap[0,5] + swap[5,10]`` with a line running 4.8–5.4s: overlap puts it
      in BOTH windows, both clips are generated saying it, and the stitched
      hook says the sentence twice.
    * ``swap[0,3] + keep[3,5] + swap[5,10]`` with a line inside ``[3,5)``:
      overlap puts it in NEITHER window. Since a per-segment prompt REPLACES
      the run prompt (``pipeline_v2._submit_swap_segment_isolated`` uses
      ``prompt_override if prompt_override else run_prompt``), the run prompt
      is not a fallback for a segment that got one — so the line reaches no
      model at all and is silently lost.

    A midpoint falls in at most one half-open interval, which kills the first
    case, and the second is made loud instead of silent: the caller warns by
    name about everything in *dropped*.

    A line that straddles a boundary is still assigned whole (a model cannot be
    handed half a sentence), so it is spoken entirely by one clip and the
    delivery is cut mid-sentence at the seam. That is a real editorial problem,
    and the caller warns about it separately.
    """
    by_segment: dict[str, list[dict]] = {s.id: [] for s in segments}
    dropped: list[dict] = []
    for line in lines or []:
        midpoint = _line_midpoint(line)
        for segment in segments:
            if float(segment.start_sec) <= midpoint < float(segment.end_sec):
                by_segment[segment.id].append(line)
                break
        else:
            dropped.append(line)
    return by_segment, dropped


def _coverage_gaps(segments: list, hook: float) -> list[tuple[float, float]]:
    """The sub-windows of ``[0, hook)`` no swap segment covers.

    A UNION sweep, not ``max(end_sec)``. The old measure could only see a hook
    that ran off the end of the last segment; a hole in the middle
    (``swap[0,3] + keep[3,5] + swap[5,10]``, the shape an operator produces by
    hand-splitting in the Segment Editor) still read as fully covered, because
    the last segment ends exactly at the hook. Every line in that hole is
    dropped from every prompt, so it has to be visible.

    Gaps narrower than :data:`_COVERAGE_EPS` are ignored as float noise.
    """
    intervals = sorted(
        (max(0.0, float(s.start_sec)), min(hook, float(s.end_sec)))
        for s in segments
    )
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in intervals:
        if end <= start:
            continue
        if start - cursor > _COVERAGE_EPS:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if hook - cursor > _COVERAGE_EPS:
        gaps.append((cursor, hook))
    return gaps


def _segment_boundaries(segments: list, hook: float) -> list[float]:
    """Every instant inside ``(0, hook)`` where the delivery cuts, plus the hook.

    A line spanning one of these is spoken by one clip but the stitch cuts
    there anyway, so it needs the same "cut mid-sentence" advisory the hook
    edge has always had. Both ends of a gap qualify, which is why segment
    starts are included and not only ends.
    """
    marks = {float(hook)}
    for segment in segments:
        for edge in (float(segment.start_sec), float(segment.end_sec)):
            if _COVERAGE_EPS < edge < hook - _COVERAGE_EPS:
                marks.add(edge)
    return sorted(marks)


def _localisation_warnings(
    *,
    window_lines: list[dict],
    translated_lines: list[dict],
    hook: float,
    hook_segments: list,
    dropped_lines: list[dict],
    source_language: str,
    target_language: str,
) -> list[str]:
    """Advisories an operator must see before submitting the run.

    Five things, all of them invisible in the returned prompt itself and all
    of them expensive to discover after a generation has been paid for:

    1. **A line straddling a cut.** ``slice_lines`` takes lines that OVERLAP the
       window, so a line starting at 9.5s in a 10s hook is included whole — but
       the hook segment ends at 10s, so its delivery is cut mid-sentence. The
       same applies at every boundary BETWEEN the hook's swap segments: the
       line is assigned whole to the clip holding its midpoint (a model cannot
       be handed half a sentence), and the stitch cuts through it anyway.
       Shortening the line, or moving the boundary, is a decision only the
       operator can make.
    2. **An implausible spoken length.** The translation is asked to match each
       line's DURATION, not its word count, and the model does not always
       comply. A line that will obviously overrun makes the delivery race or
       run off the end of the shot — the exact failure the manual run had to be
       re-cut for.
    3. **Source language == target language.** Almost always a mis-picked
       dropdown: the run would spend a generation reproducing the original.
    4. **The hook window is not fully covered by swap segments.** Measured as
       the UNION of swap coverage inside ``[0, hook)``, so a hole in the middle
       counts — ``swap[0,3] + keep[3,5] + swap[5,10]`` used to read as fully
       covered because the last segment still ended at the hook. Two shapes
       reach this: a per-call ``hook_sec`` override that deliberately
       re-segments nothing and so asks for 15s of dialogue from a 10s clip, and
       a hand-split hook with a keep gap in it. A project with no swap segment
       over the hook at all (analyzed under a different type, or every segment
       flipped to keep) gets the blunter version of the same message: this
       prompt is text no clip will ever receive.
    5. **A line that lands in one of those gaps.** It is generated by nothing —
       not the run prompt either, because a per-segment prompt REPLACES the run
       prompt rather than extending it, so the segments around the gap never
       see it. Naming the line is the whole point: this failure is otherwise
       completely silent, and the first sign of it is a delivered hook missing
       a sentence. The caller passes an empty *dropped_lines* when it produced
       no per-segment prompts at all, because then the run prompt IS what the
       single swap segment receives and nothing is actually lost.

    Everything here is advisory — the prompt is returned regardless, because
    an operator who knows better must not be blocked by a heuristic.
    """
    warnings: list[str] = []

    gaps = _coverage_gaps(hook_segments, hook)
    covered = hook - sum(end - start for start, end in gaps)
    if not hook_segments:
        warnings.append(
            f"This project has no swap segment over the first {hook:.1f}s, so "
            "nothing will be generated from this prompt. Re-analyze the project "
            "as a localisation project, or mark the hook segment as swap."
        )
    elif gaps:
        gap_text = ", ".join(f"{start:.1f}-{end:.1f}s" for start, end in gaps)
        warnings.append(
            f"The hook window is {hook:.1f}s but the analyzed swap segment(s) "
            f"only cover {covered:.1f}s of it — nothing is generated for "
            f"{gap_text}. Dialogue there reaches no clip at all. Set hook_sec on "
            "the project and re-analyze, or mark the uncovered stretch as swap."
        )

    if source_language and source_language == target_language:
        warnings.append(
            f"Source language and target language are both "
            f"{languages.label_for(target_language)} ({target_language}) — the "
            "translation is a no-op; pick a different target language unless "
            "this is deliberate."
        )

    for line in dropped_lines:
        start = _line_seconds(line, "start")
        end = _line_seconds(line, "end")
        warnings.append(
            f"Line {line.get('id')} ({line.get('speaker')}) runs from "
            f"{start:.1f}s to {end:.1f}s, which no swap segment covers — it is "
            "assigned to no clip and will not be spoken at all. Extend a swap "
            "segment over it, or drop the line."
        )

    # Every line is judged against its speaking WINDOW — the part of its span
    # the clip that speaks it actually has (localisation.speaking_window): the
    # translator was given that window and a matching character budget, and
    # re-asked once when it overran. So a line crossing the hook edge or a
    # seam is only a problem when its translation still does not fit before
    # the cut; one that does is a complete, shorter line and needs no alert.
    boundaries = _segment_boundaries(hook_segments, hook)
    cuts = [b for b in boundaries if b < hook]
    for line in translated_lines:
        start = _line_seconds(line, "start")
        end = _line_seconds(line, "end")
        win_start, win_end = localisation.speaking_window(
            line, cuts=cuts, hook_end=hook
        )
        window = win_end - win_start
        estimate = _estimated_speech_sec(line.get("text", ""), target_language)
        fits = window > 0 and estimate <= window * _SPEECH_OVERRUN_RATIO
        crossed = [b for b in boundaries if start < b < end]
        if crossed and not fits:
            boundary = crossed[0]
            if boundary == hook:
                warnings.append(
                    f"Line {line.get('id')} ({line.get('speaker')}) runs from "
                    f"{start:.1f}s to {end:.1f}s and crosses the {hook:.1f}s hook "
                    f"boundary — the translation (~{estimate:.1f}s) does not fit "
                    f"the {window:.1f}s before the cut, so it will be cut "
                    "mid-sentence. Shorten the line or move the hook."
                )
            else:
                warnings.append(
                    f"Line {line.get('id')} ({line.get('speaker')}) runs from "
                    f"{start:.1f}s to {end:.1f}s and crosses the {boundary:.1f}s "
                    "boundary between two swap segments — the translation "
                    f"(~{estimate:.1f}s) does not fit the {window:.1f}s its clip "
                    "has, so the delivery is cut mid-sentence at the seam. "
                    "Move the boundary or split the line."
                )
            continue
        if window <= 0:
            continue
        if estimate > window * _SPEECH_OVERRUN_RATIO:
            warnings.append(
                f"Line {line.get('id')}: the translation is roughly "
                f"{estimate:.1f}s of speech but the shot gives it "
                f"{window:.1f}s — it will overrun. Shorten it."
            )
        elif estimate < window * _SPEECH_UNDERRUN_RATIO:
            warnings.append(
                f"Line {line.get('id')}: the translation is roughly "
                f"{estimate:.1f}s of speech against {window:.1f}s of footage "
                "— check nothing was dropped."
            )

    return warnings


@router.post(
    "/projects/{pid}/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TranscriptResponse,
    dependencies=[GATED],
)
def transcribe_project(pid: str, db: Session = Depends(get_db)) -> TranscriptResponse:
    """Queue transcription of the project's source video (202 Accepted).

    Sets ``transcript_status="pending"``, clears the previous error and pushes
    the job; the worker takes it from there (docs/localisation.md §8 —
    transcription is its own RQ task, never part of analysis, so a model outage
    can neither fail analysis nor block segmentation).

    Re-runnable once the previous transcription has finished (the project
    page's "Распознать заново"): the job overwrites the transcript wholesale.
    While one is queued or running this answers 409 (AICP-365, operator
    decision 2026-10-31) — the durable job record (app/localisation_jobs.py)
    and its reaper resolve a worker killed mid-call to ``failed`` /
    interrupted, which is what the old "always re-runnable" rule was for. The
    PREVIOUS transcript stays readable until the new one lands — but note that
    ``POST /localisation-prompt`` gates on the STATUS, so it will 409 until the
    re-run finishes.

    409 when the source video has not been fetched yet — transcription reads
    the downloaded file, and a Drive project only gets one when analysis
    downloads it. The remedy depends on where the project is, so the detail
    says which: still-to-be-analyzed means "wait", whereas an already-``ready``
    project with no file means the download never produced one (or the file was
    since removed), and waiting for an analysis that already finished would be
    waiting forever — re-analysis is what fetches it again.

    Not gated on project_type: a transcript is meaningful for any footage, and
    the flow that consumes it is the caller's business.
    """
    project = _get_project_or_404(pid, db)
    if not project.source_local_path:
        # `_status_text`, not `!r`: the panel puts this string straight into a
        # browser alert, and `<ProjectStatus.ready: 'ready'>` is not something
        # an operator should ever be shown.
        if project.status in (ProjectStatus.created, ProjectStatus.analyzing):
            remedy = (
                "analysis has not fetched it yet — wait for the project to "
                "become ready, then try again"
            )
        else:
            remedy = (
                "analysis is no longer running, so the file was never "
                f"downloaded or has since been removed — POST /api/v2/projects/"
                f"{pid}/analyze to fetch it again"
            )
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot transcribe: this project has no source video on disk "
                f"(project status: {_status_text(project.status)}) — {remedy}"
            ),
        )

    # AICP-365: one durable admission for every origin (button, after-analysis
    # hand-off, autopilot): the job row, "pending" and the reservation commit
    # together, a lost enqueue is re-sent by the reaper, and a worker killed
    # mid-call is marked interrupted instead of leaving "running" forever. A
    # transcription already live answers 409 (operator decision 2026-10-31);
    # 429 when today's budget cannot cover it.
    job = localisation_jobs.admit_transcription(db, project, origin="manual")
    db.commit()
    db.refresh(project)

    localisation_jobs.enqueue(job)
    log.info("Queued transcription for project %s (job %s)", pid, job.id)
    return _transcript_response(project)


@router.get("/projects/{pid}/transcript", response_model=TranscriptResponse)
def get_project_transcript(
    pid: str, db: Session = Depends(get_db)
) -> TranscriptResponse:
    """Return the cached transcript and the transcription task's state.

    ``status`` is None when transcription was never requested, and ``empty``
    when the model heard no speech — a success, not an error. ``transcript`` is
    the §4.1 dict verbatim.
    """
    project = _get_project_or_404(pid, db)
    return _transcript_response(project)


@router.patch("/projects/{pid}/transcript", response_model=TranscriptResponse)
def update_project_transcript(
    pid: str, body: Any = Body(...), db: Session = Depends(get_db)
) -> TranscriptResponse:
    """Replace the transcript with an operator-corrected version (§4.1 shape).

    A whole-document PUT in PATCH's clothing, matching the panel that produces
    it: the UI holds the transcript in memory, the operator fixes a speaker
    label or a misheard word, and the whole thing comes back. There is no
    per-line patch route because there is no per-line resource — the transcript
    is one JSON column.

    The body is validated against §4.1 (see :func:`_validate_transcript_payload`)
    and a malformed one is a 400: this text becomes a Seedance prompt, so
    storing garbage would only defer the failure to a paid generation.

    Also resets the lifecycle: a hand-fixed transcript is ``ready`` (or
    ``empty`` when the operator submits no lines) and the previous
    ``transcript_error`` is cleared — the whole point of editing a *failed*
    transcription by hand is to make the project usable again.

    Accepted on any project, including one that was never transcribed: pasting
    a transcript in by hand is the documented fallback when the model is down.
    """
    project = _get_project_or_404(pid, db)
    # AICP-363: project row lock first (fences `project_transcript:` keys).
    db.refresh(project, with_for_update=True)
    cleaned = _validate_transcript_payload(body)

    project.transcript = cleaned
    project.transcript_status = (
        _TRANSCRIPT_READY if cleaned["lines"] else _TRANSCRIPT_EMPTY
    )
    project.transcript_error = None
    db.commit()
    db.refresh(project)
    log.info(
        "Transcript for project %s edited by hand: %d line(s), status=%s",
        pid, len(cleaned["lines"]), project.transcript_status,
    )
    return _transcript_response(project)


@router.post(
    "/projects/{pid}/localisation-prompt",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=LocalisationPromptJobAccepted,
    dependencies=[GATED],
)
def build_localisation_prompt(
    pid: str,
    language: str = Form(...),
    swap_character: bool = Form(False),
    hook_sec: Optional[float] = Form(None),
    model: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> LocalisationPromptJobAccepted:
    """Queue the hook translation and Seedance prompt build for a localisation run.

    *model* (optional, AICP-335) is the model picked on the New Run form; the
    prompt's clip windows use its per-clip limit for the pre/post-roll trim
    (:func:`app.localisation.resolve_prompt_model`). Omitted = the project
    type's default model, as before. An unknown key is a 400.

    The "Перевести" button behind the New Run form. It slices the cached
    transcript to the hook window, and a background job (AICP-338,
    :mod:`app.localisation_prompt_job`) translates those lines into *language*
    and assembles the prompt text (plus per-segment prompts when the hook is cut
    into more than one swap segment) for the operator to edit and submit
    through the ordinary ``create_run`` path. Poll
    ``GET /projects/{pid}/localisation-prompt/jobs/{job_id}`` for the
    :class:`LocalisationPromptResponse`.

    Why a job: a whole-video translation (a 79 s source, 23 lines) takes
    180-280 s for one model call, past nginx's 300 s proxy budget once a retry
    is added — the synchronous endpoint answered 502 to exactly the projects
    that need it most.

    **Writes nothing to the database** beyond the spend reservation it always
    made. No transcript is updated, no run is created, no segment is touched —
    the answer lives only in Redis as the job's result, it is a suggestion, and
    the run the operator eventually submits is the only record of what was
    used. That also makes the call freely repeatable: try a language, read the
    warnings, try another. An identical request while the first is still
    queued or running joins it (``deduplicated``) instead of paying twice.

    *swap_character* picks the template: True = "речь + смена персонажа" (the
    classic face swap AND the language change; a reference image is expected),
    False = "только речь" (the person on screen is preserved).

    The hook window comes from three places, in this order (see
    :func:`_resolve_hook_sec` for the full note):

    * *hook_sec*, this request's override — highest precedence, this call only,
      never persisted, re-segments nothing. It only changes which lines are
      translated, so it can legitimately ask for more dialogue than the clips
      can say, and warns when it does.
    * the ANALYZED window — the default. Where the project's swap SegmentDefs
      actually end is what a run will actually generate, so that is what the
      script is cut to. A project whose swap segments cover the whole video is
      therefore translated whole without any override.
    * the STORED ``VideoProject.hook_sec`` — the operator's intent. It bounds
      the analyzed window and drives the NEXT analysis, and it is the fallback
      when there are no swap segments to measure. Use ``PATCH /projects/{pid}``
      + ``POST /projects/{pid}/analyze`` to actually move the hook.

    Status codes (all answered synchronously, before anything is queued):
        202: queued (or joined an identical queued/running job).
        400: unknown language code, or a hook_sec override that is not a
            positive finite number.
        404: no such project.
        409: ``transcript_status`` is not ``ready`` — nothing to translate yet;
            or an identical request is being admitted this very moment.
        422: the hook window contains no speech, so there is nothing to
            re-speak (a wordless hook is a legal transcript, not an error, but
            it is not translatable).
        429: today's kie.ai budget cannot cover the translation.
        503: no localise-worker is running, or Redis is unreachable.

    The model's own failure (the old synchronous 502) is reported by the poll
    as ``status="failed"`` with ``error.status_code=502``.
    """
    # Extra optional request fields travel to the job unchanged in
    # snapshot["request_params"] and are part of the dedupe key. A new Form
    # field (e.g. a per-call `model`) is added here and read in
    # _compute_localisation_prompt via snapshot["request_params"].
    request_params: dict = {}
    model_key = (model or "").strip()
    if model_key:
        # Only when sent, so "no model" stays the same dedupe key as before.
        request_params["model"] = model_key
    snapshot, project, swap_segments = _prepare_localisation_snapshot(
        pid, language, swap_character, hook_sec, db,
        request_params=request_params,
    )
    fingerprint = localisation_prompt_job.project_fingerprint(project, swap_segments)
    key = localisation_prompt_job.dedupe_key(
        pid, snapshot["target_language"], swap_character,
        snapshot["hook_override"], fingerprint,
        extra=snapshot["request_params"],
    )
    meta = {
        "kind": "localisation_prompt",
        "project_id": pid,
        "fingerprint": fingerprint,
        "language": snapshot["target_language"],
        "swap_character": bool(swap_character),
        "request_params": snapshot["request_params"],
    }

    def _reserve() -> None:
        # Reserved and COMMITTED before the job is queued, exactly as the
        # synchronous endpoint did before its call — the cap must see the
        # translation before it is paid for, and a refusal is a 429 answered
        # here, not a failed job. Three calls' worth, the most translate_lines
        # can bill: a broken 1:1 answer is re-asked once, and an overrunning
        # answer then gets one shorten pass (localisation._shorten_overruns).
        # A request that JOINS a running job never gets here.
        _spend_reserve(
            spend_mod.reserve, db, spend_mod.localise_scope(pid),
            3 * spend_mod.llm_credits(), kind="localisation prompt",
        )
        db.commit()

    def _enqueue(job_id: str) -> None:
        enqueue_localisation_prompt(job_id, snapshot, meta=meta)

    try:
        job_id, created = _localise_admit(key=key, reserve=_reserve, enqueue=_enqueue)
    except localisation_prompt_job.AdmissionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    log.info(
        "Localisation prompt for project %s -> %s %s job %s (hook=%.1fs, "
        "swap_character=%s)",
        pid, snapshot["target_language"],
        "queued" if created else "joined existing", job_id,
        snapshot["hook"], swap_character,
    )
    return LocalisationPromptJobAccepted(
        job_id=job_id, status="queued", deduplicated=not created
    )


@router.get(
    "/projects/{pid}/localisation-prompt/jobs/{job_id}",
    response_model=LocalisationPromptJobStatus,
)
def get_localisation_prompt_job(
    pid: str, job_id: str, response: Response, db: Session = Depends(get_db)
) -> LocalisationPromptJobStatus:
    """Poll one translate job queued by ``POST /localisation-prompt``.

    Always 200 for a job of this project (the job's outcome is in *status* /
    *error*, carrying the status code the synchronous endpoint would have
    used), 404 for an unknown project or a job id that belongs to another
    project, 503 when Redis is unreachable (the client should keep polling).

    A finished answer is refused (``failed``, 409) when the project has changed
    since the request — re-transcribed, transcript edited, re-analyzed or
    segments edited — because its per-segment prompts are keyed to segments
    that may no longer exist. Writes nothing.
    """
    project = _get_project_or_404(pid, db)
    response.headers["Cache-Control"] = "no-store"
    try:
        state = _localise_state(job_id, pid)
    except localisation_prompt_job.AdmissionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"No translation job {job_id} for project {pid}",
        )
    if state.get("status") == "done":
        current = localisation_prompt_job.project_fingerprint(
            project, _project_swap_segments(pid, db)
        )
        if state.get("fingerprint") != current:
            state = dict(state)
            state.pop("result", None)
            state["status"] = "failed"
            state["error"] = {
                "status_code": 409,
                "detail": (
                    "The project changed while this translation ran (the "
                    "transcript or the segments were edited, or it was "
                    "re-analyzed), so the answer may not fit it any more. "
                    "Press Translate again."
                ),
            }
    fields = {
        k: state.get(k)
        for k in (
            "status", "result", "error", "language", "swap_character",
            "enqueued_at", "started_at", "ended_at",
        )
    }
    return LocalisationPromptJobStatus(job_id=job_id, **fields)


def _localise_admit(*, key: str, reserve, enqueue) -> tuple[str, bool]:
    """Admit a translate job (see localisation_prompt_job.admit). Test seam:
    the API tests replace this with an inline runner, since there is no Redis
    in the suite."""
    from redis.exceptions import RedisError

    from .tasks import _get_redis

    try:
        conn = _get_redis()
        if not localisation_prompt_job.worker_alive(conn):
            raise localisation_prompt_job.AdmissionError(
                503,
                "No localise-worker is running, so the translation could not "
                "be queued (deploy re-skin-localise-worker). Nothing was "
                "reserved.",
            )
        return localisation_prompt_job.admit(
            conn, key=key, reserve=reserve, enqueue=enqueue
        )
    except RedisError as exc:
        log.warning("localise admission: Redis error: %s", exc)
        raise localisation_prompt_job.AdmissionError(
            503, "The job queue is unreachable right now — try again in a moment."
        ) from exc


def _localise_state(job_id: str, pid: str) -> Optional[dict]:
    """Read a translate job's state (see localisation_prompt_job.job_state).
    Test seam, like :func:`_localise_admit`."""
    from redis.exceptions import RedisError

    from .tasks import _get_redis

    try:
        return localisation_prompt_job.job_state(_get_redis(), job_id, pid)
    except RedisError as exc:
        log.warning("localise poll: Redis error: %s", exc)
        raise localisation_prompt_job.AdmissionError(
            503, "The job queue is unreachable right now — keep polling."
        ) from exc


def _prepare_localisation_snapshot(
    pid: str,
    language: str,
    swap_character: bool,
    hook_sec: Optional[float],
    db: Session,
    *,
    request_params: Optional[dict] = None,
) -> tuple[dict, VideoProject, list]:
    """Validate a translate request and freeze its inputs. Writes nothing.

    Every refusal the synchronous endpoint used to make before its model call
    is made here, with the same status codes, so they still reach the caller
    in the POST's own answer. The returned snapshot carries everything
    :func:`_compute_localisation_prompt` needs, so the job translates what was
    asked for even if the project changes while it waits in the queue (the
    poll then refuses the stale answer — see get_localisation_prompt_job)."""
    project = _get_project_or_404(pid, db)

    target_language = _validate_language(language)
    if target_language is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"language is required and must be one of "
                f"{sorted(languages.LANGUAGES)} — it is the language the hook "
                "will be re-spoken in"
            ),
        )
    # AICP-335: the form's model decides the clip ceiling that trims
    # pre/post-roll in the prompt's clip windows. Validated here — a 400
    # before anything is reserved or queued — and frozen into the snapshot.
    requested_model = (request_params or {}).get("model") or None
    try:
        prompt_model = localisation.resolve_prompt_model(
            project.project_type, requested_model
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    swap_segments = _project_swap_segments(pid, db)
    hook_override = _validate_hook_sec(hook_sec)
    hook = _resolve_hook_sec(project, hook_override, swap_segments)
    hook_segments = _hook_swap_segments(swap_segments, hook)

    if project.transcript_status != _TRANSCRIPT_READY:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Transcript is not ready (status: {project.transcript_status!r}) "
                "— run POST /transcribe first, or paste a transcript with PATCH "
                "/transcript"
            ),
        )

    transcript = project.transcript or {}
    source_language = str(transcript.get("source_language") or "")
    # v1 envelopes (and any hand-edited payload that omitted the fields) have
    # no video_summary / scene_context — empty strings keep translate_lines
    # backward-compatible without inventing context.
    video_summary = transcript.get("video_summary") or ""
    scene_context = transcript.get("scene_context") or ""
    if not isinstance(video_summary, str):
        video_summary = ""
    if not isinstance(scene_context, str):
        scene_context = ""
    window_lines = localisation.slice_lines(transcript.get("lines") or [], 0.0, hook)
    if not window_lines:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No speech in the first {hook:.1f}s of this video, so there is "
                "nothing to translate. Widen the hook (hook_sec) or write the "
                "prompt by hand."
            ),
        )
    all_lines = transcript.get("lines") or []
    on_screen_text = transcript.get("on_screen_text") or []

    snapshot = copy.deepcopy({
        "project_id": pid,
        "target_language": target_language,
        "swap_character": bool(swap_character),
        "hook": float(hook),
        "hook_override": hook_override,
        "stored_hook_sec": project.hook_sec,
        "source_language": source_language,
        "video_summary": video_summary,
        "scene_context": scene_context,
        "window_lines": window_lines,
        # The whole cached transcript, frozen with the rest: the prompt build
        # may need more of it than the lines (kept here so the job never has
        # to re-read the project).
        "transcript": transcript,
        "all_lines": all_lines if isinstance(all_lines, list) else None,
        "on_screen_text": on_screen_text if isinstance(on_screen_text, list) else None,
        "hook_segments": [
            {
                "id": str(s.id),
                "start_sec": float(s.start_sec),
                "end_sec": float(s.end_sec),
                # Not read yet; carried so a clip-window prompt build
                # (AICP-335) needs no DB access in the job.
                "pre_roll_sec": float(getattr(s, "pre_roll_sec", None) or 0.0),
                "post_roll_sec": float(getattr(s, "post_roll_sec", None) or 0.0),
            }
            for s in hook_segments
        ],
        "project_type": project.project_type,
        "duration_sec": getattr(project, "duration_sec", None),
        "source_local_path": project.source_local_path,
        # Optional request fields beyond language/mode/hook, passed through
        # verbatim (JSON-safe values only); see build_localisation_prompt.
        "request_params": dict(request_params or {}),
        # The resolved prompt model (AICP-335), frozen so the job never
        # re-resolves it against a registry that may have changed since.
        "prompt_model": {
            "max_clip_sec": float(prompt_model.max_clip_sec),
            "label": str(prompt_model.label),
        },
    })
    return snapshot, project, swap_segments


def _compute_localisation_prompt(
    snapshot: dict, *, translate_budgets: Optional[dict] = None
) -> LocalisationPromptResponse:
    """Translate a frozen request and build the prompt. No DB access.

    The body the synchronous endpoint used to run after its spend reservation,
    unchanged; it now runs in the localise-worker
    (localisation_prompt_job.run). Raises localisation.LocalisationError on a
    model failure, which the job reports as the old 502. *translate_budgets*
    are extra keyword arguments for localisation.translate_lines (the job's
    larger re-ask / shorten budgets); None keeps translate_lines' defaults."""
    pid = snapshot["project_id"]
    target_language = snapshot["target_language"]
    swap_character = bool(snapshot["swap_character"])
    hook = float(snapshot["hook"])
    source_language = snapshot["source_language"]
    window_lines = snapshot["window_lines"]
    hook_segments = [SimpleNamespace(**s) for s in snapshot["hook_segments"]]

    # Stills of the hook let the translator see who speaks and what "this"
    # is — and catch a word the transcriber misheard. An optimisation, never a
    # reason to fail: any trouble grabbing them just translates without.
    frames: list = []
    if snapshot.get("source_local_path"):
        try:
            frames = media_mod.extract_jpeg_frames(
                snapshot["source_local_path"],
                localisation.hook_frame_times(window_lines, hook),
            )
        except Exception as exc:  # noqa: BLE001 — see above
            log.warning(
                "localise %s: no hook frames for the translator (%s: %s)",
                pid, type(exc).__name__, exc,
            )

    transcript = snapshot["transcript"]
    prompt_model = SimpleNamespace(**snapshot["prompt_model"])
    model_given = bool(snapshot["request_params"].get("model"))

    translated_lines = localisation.translate_lines(
        window_lines,
        source_language=source_language,
        target_language=target_language,
        video_summary=snapshot["video_summary"],
        scene_context=snapshot["scene_context"],
        context_lines=snapshot["all_lines"],
        on_screen_text=snapshot["on_screen_text"],
        frames=frames,
        cuts=[b for b in _segment_boundaries(hook_segments, hook) if b < hook],
        hook_end=hook,
        spend_scope=spend_mod.localise_scope(pid),
        **(translate_budgets or {}),
    )
    # Timed dialogue (AICP-335): each prompt's timecodes are relative to
    # the clip it is for. One hook segment = the run prompt IS that clip's
    # prompt; otherwise the run prompt counts from the video start.
    # The clip ceiling that trims pre/post-roll comes from prompt_model:
    # the form's model when sent, else the type's default (warned below
    # when rolls are set and no model was sent).
    source_duration = snapshot.get("duration_sec")

    def _window(segment) -> tuple[float, float]:
        return localisation.clip_window(
            segment.start_sec, segment.end_sec,
            segment.pre_roll_sec or 0.0, segment.post_roll_sec or 0.0,
            source_duration, prompt_model.max_clip_sec,
        )

    run_window = (
        _window(hook_segments[0]) if len(hook_segments) == 1 else (0.0, hook)
    )
    prompt = localisation.build_prompt(
        lines=translated_lines,
        source_language=source_language,
        target_language=target_language,
        swap_character=swap_character,
        window=run_window,
        transcript=transcript,
        all_lines=translated_lines,
    )
    # Per-segment prompts ONLY when the hook was cut into several swap
    # segments (a hook longer than the model's per-clip ceiling, or one
    # hand-split in the Segment Editor). Each segment gets its OWN share of
    # the dialogue, because each is generated as its own clip: handing all
    # of it to every segment would make each clip try to say the whole hook.
    #
    # The split is a partition by line MIDPOINT, not an overlap slice —
    # every line belongs to exactly one segment. See
    # _assign_lines_to_segments for why overlap semantics duplicate a
    # boundary-straddling line into two clips and lose a line that falls in
    # a keep gap.
    by_segment, dropped_lines = _assign_lines_to_segments(
        translated_lines, hook_segments
    )
    segment_prompts: dict[str, str] = {}
    if len(hook_segments) > 1:
        for segment in hook_segments:
            segment_prompts[segment.id] = localisation.build_prompt(
                lines=by_segment[segment.id],
                source_language=source_language,
                target_language=target_language,
                swap_character=swap_character,
                window=_window(segment),
                transcript=transcript,
                all_lines=translated_lines,
            )
    else:
        # No overrides means the run prompt is what the (single) swap
        # segment receives, so an unassigned line is not lost — the clip
        # still says it, mistimed at worst. Claiming otherwise would be a
        # lie; the coverage warning already names the uncovered stretch.
        dropped_lines = []

    warnings = _localisation_warnings(
        window_lines=window_lines,
        translated_lines=translated_lines,
        hook=hook,
        hook_segments=hook_segments,
        dropped_lines=dropped_lines,
        source_language=source_language,
        target_language=target_language,
    )
    if not model_given and any(
        (s.pre_roll_sec or 0) > 0 or (s.post_roll_sec or 0) > 0 for s in hook_segments
    ):
        warnings.append(
            "A hook segment has pre/post-roll, so the prompt's timecodes depend "
            "on the run's model: they count from the clip start as cut for "
            f"{prompt_model.label} ({prompt_model.max_clip_sec:.0f}s clip limit). "
            "Any other model may keep or trim a different amount of pre-roll and "
            "shift every timecode. Translate again with the run's model "
            "selected, or set the rolls to 0 (they add nothing to a "
            "localisation clip)."
        )
    for line in localisation.background_lines(translated_lines, transcript):
        warnings.append(
            f"Line {line.get('id')} ({line.get('speaker')}, "
            f"{float(line.get('start') or 0):.1f}–{float(line.get('end') or 0):.1f}s) "
            "plays under other speakers' lines. It was kept in the prompt and "
            "marked BACKGROUND; the model may not voice two speakers at once, "
            f"so delete it by hand if it does not need to be heard: {line.get('text')!r}"
        )

    log.info(
        "Built localisation prompt for project %s: %s -> %s, hook=%.1fs "
        "(stored intent=%s), %d line(s), %d segment prompt(s), %d line(s) in "
        "an uncovered gap, %d warning(s), swap_character=%s",
        pid, source_language or "?", target_language, hook,
        snapshot.get("stored_hook_sec"),
        len(translated_lines), len(segment_prompts), len(dropped_lines),
        len(warnings), swap_character,
    )
    return LocalisationPromptResponse(
        source_language=source_language,
        target_language=target_language,
        hook_sec=hook,
        prompt=prompt,
        segment_prompts=segment_prompts,
        lines=translated_lines,
        warnings=warnings,
    )


def _project_swap_segments(pid: str, db: Session) -> list:
    """Every swap SegmentDef of a project, in index order. Read-only.

    Keep segments are excluded everywhere downstream for one reason: they are
    never generated, so neither a prompt nor a share of the hook window means
    anything for one. Fetched once per request because both the hook window
    itself (:func:`_analyzed_hook_sec`) and the per-segment split are derived
    from the same list.
    """
    return list(
        db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == pid, SegmentDef.action == "swap")
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    )


def _hook_swap_segments(swap_segments: list, hook: float) -> list:
    """The swap segments that start inside ``[0, hook)``, in index order.

    Normally exactly one (hook_split lays down a single swap segment over the
    hook), so the caller returns ``segment_prompts={}`` and puts the whole
    dialogue in the run prompt. More than one means the hook was chunked — by
    the segmentation cap, or by an operator splitting it in the Segment Editor
    — and each chunk is generated as its own clip, so each needs its own share
    of the dialogue.
    """
    return [s for s in swap_segments if float(s.start_sec) < hook]


# ---------------------------------------------------------------------------
# Stitch library — global reusable clips (e.g. alternative outros)
# ---------------------------------------------------------------------------

# Run states in which the worker may still read a referenced library clip (or
# the run may still be resumed/retried into stitching) — block clip deletion
# for these. `failed` is included because failed runs are routinely retried.
_STITCH_ACTIVE_RUN_STATUSES = _ACTIVE_RUN_STATUSES | {
    RunStatus.created,
    RunStatus.incomplete,
    RunStatus.failed,
}

# Tail-edit states in which the worker may still read a referenced library
# clip — block clip deletion for these (mirrors _STITCH_ACTIVE_RUN_STATUSES).
_ACTIVE_TAIL_EDIT_STATUSES = {
    TailEditStatus.queued,
    TailEditStatus.processing,
    TailEditStatus.delivering,
}


def _plan_active_segment_ids(plan: Optional[list]) -> Optional[set]:
    """Segment ids a stored stitch plan processes, or None when EVERY segment
    is active (default plan and the no-stitch plan [] both process all swaps).
    Mirrors pipeline_v2._active_swap_segment_ids without needing seg_defs.
    """
    if not plan:
        return None
    return {
        e.get("segment_def_id")
        for e in plan
        if isinstance(e, dict) and e.get("kind") == "segment"
    }


def _get_library_clip_or_404(clip_id: str, db: Session) -> LibraryClip:
    clip = db.get(LibraryClip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Library clip not found: {clip_id}")
    return clip


@router.post(
    "/library-clips",
    status_code=status.HTTP_201_CREATED,
    response_model=LibraryClipResponse,
)
def create_library_clip(
    file: Optional[UploadFile] = File(None),
    gdrive_url: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    kind: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> LibraryClipResponse:
    """Add a reusable stitch clip (e.g. an alternative outro) to the library.

    Accepts either a file upload OR a Google Drive link (not both). The clip is
    probed with ffprobe — a non-video file is rejected (400) and removed from disk.
    Drive errors (bad link, permission) also return 400 with cleanup.

    *language* tags the clip's spoken language with one of app/languages.py's
    codes, which is how the New Run form auto-picks the language-matched tail
    for a localisation run. Leave it blank for a language-agnostic clip (no
    speech, or one that suits any language) — that is what every clip uploaded
    before this field existed is. Nothing in the stitch phase branches on it:
    a mismatched tail still stitches, it is just wrong for the operator.

    *kind* (AICP-381) is ``end_card`` (a short closing card a rebuild may put
    in place of the source's own) or ``insert`` (the default: anything else).
    """
    # Require exactly one of file or gdrive_url.
    if file and gdrive_url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a Drive link, not both"
        )
    if not file and not gdrive_url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a Drive link"
        )

    # Validated before anything is written: a bad code must not leave an
    # uploaded file (or a Drive download) orphaned under <DATA>/library.
    language_val = _validate_language(language)
    kind_val = _validate_clip_kind(kind)

    clip_id = str(uuid.uuid4())
    dest = os.path.join(library_clip_dir(clip_id), "clip.mp4")
    clean_name: str

    try:
        if file:
            # Upload from file.
            ext = os.path.splitext(file.filename or "")[1].lstrip(".").lower()
            if not ext.isalnum():
                ext = "mp4"
            dest = os.path.join(library_clip_dir(clip_id), f"clip.{ext}")
            _save_upload(file, dest, max_bytes=settings.MAX_UPLOAD_SIZE_MB << 20)
            clean_name = (name or "").strip() or (file.filename or "clip")
        else:
            # Download from Google Drive.
            gdrive = _default_gdrive()
            gdrive.download_file(gdrive_url, dest)
            # Default name: use file ID from the Drive link.
            file_id = extract_file_id(gdrive_url)
            clean_name = (name or "").strip() or file_id or "drive clip"

        info = media_mod.probe(dest)
    except media_mod.MediaError as exc:
        shutil.rmtree(library_clip_dir(clip_id), ignore_errors=True)
        raise HTTPException(
            status_code=400, detail=f"Uploaded file is not a playable video: {exc}"
        ) from exc
    except GDriveDownloadError as exc:
        shutil.rmtree(library_clip_dir(clip_id), ignore_errors=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to download from Drive: {exc}"
        ) from exc
    except Exception:
        # Oversize upload (413), unexpected probe failure, or Drive error — don't
        # leave an orphaned file/dir under <DATA>/library that nothing can ever list.
        shutil.rmtree(library_clip_dir(clip_id), ignore_errors=True)
        raise

    clip = LibraryClip(
        id=clip_id,
        name=clean_name[:255],
        local_path=dest,
        duration_sec=info.duration_sec,
        width=info.width,
        height=info.height,
        fps=info.fps,
        language=language_val,
        kind=kind_val,
    )
    db.add(clip)
    db.commit()
    log.info(
        "Created library clip %s (%r, %.1fs, language=%s, kind=%s)",
        clip_id, clip.name, info.duration_sec, language_val, kind_val,
    )
    return LibraryClipResponse.model_validate(clip)


@router.patch("/library-clips/{clip_id}", response_model=LibraryClipResponse)
def update_library_clip(
    clip_id: str, body: LibraryClipUpdate, db: Session = Depends(get_db)
) -> LibraryClipResponse:
    """Rename a library clip and/or change its *language* tag or *kind*.

    Both fields are optional; an omitted key leaves the column untouched.
    Sending ``"language": ""`` clears the tag back to language-agnostic
    (mirrors ``PATCH /runs/{rid}``), an unknown code is a 400.

    This exists mainly for *language*, which arrived long after the library
    did: every clip uploaded before it is NULL, and re-uploading a tail just to
    tag it would orphan the stitch plans that already reference the old id.
    """
    clip = _get_library_clip_or_404(clip_id, db)
    if body.name is not None:
        name = body.name.strip()
        if name:
            clip.name = name[:255]
    if body.language is not None:
        clip.language = _validate_language(body.language)
    if body.kind is not None:
        if not body.kind.strip():
            raise HTTPException(status_code=400,
                                detail=f"kind must be one of {list(LIBRARY_CLIP_KINDS)}")
        clip.kind = _validate_clip_kind(body.kind)
    db.commit()
    db.refresh(clip)
    log.info(
        "Updated library clip %s name=%r language=%s kind=%s",
        clip_id, clip.name, clip.language, clip.kind,
    )
    return LibraryClipResponse.model_validate(clip)


@router.get("/library-clips", response_model=list[LibraryClipResponse])
def list_library_clips(db: Session = Depends(get_db)) -> list:
    """Return newest 200 library clips, newest first."""
    clips = (
        db.execute(select(LibraryClip).order_by(desc(LibraryClip.created_at)).limit(200))
        .scalars()
        .all()
    )
    return [LibraryClipResponse.model_validate(c) for c in clips]


@router.get("/library-clips/{clip_id}/file")
def get_library_clip_file(clip_id: str):
    """Stream a library clip's video file (for in-page preview)."""
    # Session is closed before the file transfer (same as /source).
    with get_session() as db:
        clip = _get_library_clip_or_404(clip_id, db)
        path = clip.local_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Clip file not available on disk")

    return accel_or_file_response(
        path, media_type="video/mp4", filename=os.path.basename(path)
    )


@router.delete("/library-clips/{clip_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_library_clip(clip_id: str, db: Session = Depends(get_db)) -> Response:
    """Delete a library clip (DB row + file).

    409 if a run that may still stitch (created/queued/processing/stitching/
    delivering/incomplete) references the clip in its stitch plan, OR a
    non-terminal (queued/processing/delivering) tail edit references the
    clip. Terminal runs/tail edits keep their soft reference; a later
    re-stitch/re-run fails with a clear "clip no longer exists" error.
    """
    clip = _get_library_clip_or_404(clip_id, db)

    # SQL prefilter: only fetch rows whose serialized stitch_plan could
    # possibly reference clip_id, instead of every active run. clip_id is a
    # UUID, so a literal substring match is effectively unambiguous; the
    # Python walk below still makes the precise (kind == "library" and
    # clip_id ==) decision, so any rare false positive here is harmless.
    # clip_id is attacker-controlled (URL path) and lands inside a LIKE
    # pattern, so a value containing "%"/"_" can only *widen* the SQL match
    # (never narrow or error) — the Python walk is what actually decides,
    # so no escaping is needed here.
    rows = db.execute(
        select(Run.id, Run.stitch_plan).where(
            Run.status.in_(_STITCH_ACTIVE_RUN_STATUSES),
            Run.stitch_plan.is_not(None),
            Run.stitch_plan.cast(Text).like(f"%{clip_id}%"),
        )
    ).all()
    referencing = [
        run_id
        for run_id, plan in rows
        if any(
            isinstance(e, dict)
            and e.get("kind") == "library"
            and e.get("clip_id") == clip_id
            for e in (plan or [])
        )
    ]
    if referencing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Clip is referenced by {len(referencing)} active run(s) "
                f"({', '.join(referencing[:3])}…) — wait for them to finish "
                "or delete them first"
            ),
        )

    tail_edit_referencing = (
        db.execute(
            select(RunTailEdit.id).where(
                RunTailEdit.library_clip_id == clip_id,
                RunTailEdit.status.in_(_ACTIVE_TAIL_EDIT_STATUSES),
            )
        )
        .scalars()
        .all()
    )
    if tail_edit_referencing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Clip is referenced by {len(tail_edit_referencing)} active tail "
                f"edit(s) ({', '.join(tail_edit_referencing[:3])}…) — wait for "
                "them to finish or delete them first"
            ),
        )

    clip_name = clip.name
    db.delete(clip)
    db.commit()
    shutil.rmtree(library_clip_dir(clip_id), ignore_errors=True)
    log.info("Deleted library clip %s (%r)", clip_id, clip_name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Voice library — global reusable voice references for generated speech
#
# Unlike library clips, a voice is never referenced by id from a run: picking
# one COPIES the file into the run (see _attach_voice_reference), so none of
# these endpoints need the active-reference guards delete_library_clip carries.
# ---------------------------------------------------------------------------


def _get_voice_clip_or_404(voice_id: str, db: Session) -> VoiceClip:
    voice = db.get(VoiceClip, voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"Voice clip not found: {voice_id}")
    return voice


@router.post(
    "/voice-clips",
    status_code=status.HTTP_201_CREATED,
    response_model=VoiceClipResponse,
)
def create_voice_clip(
    file: Optional[UploadFile] = File(None),
    gdrive_url: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> VoiceClipResponse:
    """Add a reusable voice reference to the library.

    Accepts either a file upload OR a Google Drive link (not both). **Any**
    media carrying an audio track is accepted — an mp3, an m4a, or a whole
    video — because the file is re-encoded to a mono mp3 capped at
    ``ai_models.VOICE_REF_MAX_SEC`` before it is stored. That normalisation is
    what makes kie.ai's format/duration rules unbreakable downstream: every
    voice that ever reaches the API came out of ``make_voice_reference``.

    400 (with the partial upload cleaned up) if the file has no audio track, is
    shorter than ``ai_models.VOICE_REF_MIN_SEC`` — nothing can invent speech to
    pad it — or the Drive link cannot be read.
    """
    if file and gdrive_url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a Drive link, not both"
        )
    if not file and not gdrive_url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a Drive link"
        )

    # Validated before anything is written, so a bad code cannot orphan an
    # upload under <DATA>/voice_library (same order as create_library_clip).
    language_val = _validate_language(language)

    voice_id = str(uuid.uuid4())
    vdir = voice_clip_dir(voice_id)
    # The upload lands next to the output under a name make_voice_reference
    # will never collide with; only the normalised mp3 is kept.
    raw_dest = os.path.join(vdir, "upload.src")
    dest = os.path.join(vdir, "voice.mp3")
    clean_name: str

    try:
        if file:
            _save_upload(file, raw_dest, max_bytes=settings.MAX_UPLOAD_SIZE_MB << 20)
            clean_name = (name or "").strip() or (file.filename or "voice")
        else:
            gdrive = _default_gdrive()
            gdrive.download_file(gdrive_url, raw_dest)
            file_id = extract_file_id(gdrive_url)
            clean_name = (name or "").strip() or file_id or "drive voice"

        duration = media_mod.make_voice_reference(raw_dest, dest)
    except media_mod.MediaError as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(
            status_code=400, detail=f"Unusable voice reference: {exc}"
        ) from exc
    except GDriveDownloadError as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to download from Drive: {exc}"
        ) from exc
    except Exception:
        shutil.rmtree(vdir, ignore_errors=True)
        raise
    finally:
        # The original is dead weight once transcoded — and may be a 200 MB
        # video whose only useful part was 10 seconds of speech.
        if os.path.exists(raw_dest):
            os.remove(raw_dest)

    voice = VoiceClip(
        id=voice_id,
        name=clean_name[:255],
        local_path=dest,
        duration_sec=duration,
        language=language_val,
    )
    db.add(voice)
    db.commit()
    log.info(
        "Created voice clip %s (%r, %.1fs, language=%s)",
        voice_id, voice.name, duration, language_val,
    )
    return VoiceClipResponse.model_validate(voice)


@router.patch("/voice-clips/{voice_id}", response_model=VoiceClipResponse)
def update_voice_clip(
    voice_id: str, body: VoiceClipUpdate, db: Session = Depends(get_db)
) -> VoiceClipResponse:
    """Rename a voice clip and/or change its *language* tag.

    Both optional; an omitted key leaves the column untouched and
    ``"language": ""`` clears the tag (mirrors ``PATCH /library-clips/{id}``).
    """
    voice = _get_voice_clip_or_404(voice_id, db)
    if body.name is not None:
        new_name = body.name.strip()
        if new_name:
            voice.name = new_name[:255]
    if body.language is not None:
        voice.language = _validate_language(body.language)
    db.commit()
    db.refresh(voice)
    log.info(
        "Updated voice clip %s name=%r language=%s",
        voice_id, voice.name, voice.language,
    )
    return VoiceClipResponse.model_validate(voice)


@router.get("/voice-clips", response_model=list[VoiceClipResponse])
def list_voice_clips(db: Session = Depends(get_db)) -> list:
    """Return newest 200 voice clips, newest first."""
    voices = (
        db.execute(select(VoiceClip).order_by(desc(VoiceClip.created_at)).limit(200))
        .scalars()
        .all()
    )
    return [VoiceClipResponse.model_validate(v) for v in voices]


@router.get("/voice-clips/{voice_id}/file")
def get_voice_clip_file(voice_id: str):
    """Stream a voice clip's mp3 (for in-page preview)."""
    # Session closed before the transfer, same as /library-clips/{id}/file.
    with get_session() as db:
        voice = _get_voice_clip_or_404(voice_id, db)
        path = voice.local_path
        if not path or not os.path.exists(path):
            raise HTTPException(
                status_code=404, detail="Voice file not available on disk"
            )

    return accel_or_file_response(
        path, media_type="audio/mpeg", filename=os.path.basename(path)
    )


@router.delete("/voice-clips/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_voice_clip(voice_id: str, db: Session = Depends(get_db)) -> Response:
    """Delete a voice clip (DB row + file).

    No 409 guard, deliberately: runs own a COPY of the audio, so no run — in
    flight, finished, or re-submitted later — reads this row or this file.
    """
    voice = _get_voice_clip_or_404(voice_id, db)
    voice_name = voice.name
    db.delete(voice)
    db.commit()
    shutil.rmtree(voice_clip_dir(voice_id), ignore_errors=True)
    log.info("Deleted voice clip %s (%r)", voice_id, voice_name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _resolve_voice_reference(
    *,
    run_id: str,
    project_id: str,
    model: str,
    voice_clip_id: Optional[str],
    voice_file: Optional[UploadFile],
    db: Session,
) -> list[str]:
    """Give *run_id* its own copy of a voice reference; return the path list.

    Accepts either a library pick (*voice_clip_id*) or an ad-hoc upload
    (*voice_file*), and returns ``[]`` when neither is supplied. The result is
    what ``Run.reference_audio_urls`` stores.

    The file is always COPIED into the run's own directory, never referenced in
    place. That is what lets ``delete_voice_clip`` skip the active-run guard
    ``delete_library_clip`` needs: a run that owns its audio cannot be re-voiced
    or broken by later library edits, including on a re-submit months later.

    An upload is normalised through ``make_voice_reference`` here; a library
    pick was normalised at upload time and is copied byte-for-byte, so a voice
    is never transcoded twice (mp3 → mp3 would lose quality for nothing).

    Raises HTTPException(400) if both sources are given, the chosen *model*
    has no ``reference_audio_urls`` field, or the upload is unusable.
    """
    has_upload = voice_file is not None and bool(voice_file.filename)
    if voice_clip_id and has_upload:
        raise HTTPException(
            status_code=400,
            detail="Provide either a library voice or an upload, not both",
        )
    if not voice_clip_id and not has_upload:
        return []

    # Rejected rather than dropped: the operator explicitly attached a voice,
    # and a control that silently does nothing is worse than an error. The New
    # Run form hides the picker for models without the field, so this fires
    # only on direct API use.
    if not ai_models.supports_reference_audio(model):
        raise HTTPException(
            status_code=400,
            detail=(
                f"model {model!r} does not accept a voice reference; "
                f"use one of {sorted(ai_models.models_supporting_reference_audio())}"
            ),
        )

    # Resolve the library row BEFORE creating any directory, so a bad id leaves
    # nothing behind on disk (the run row itself is rolled back by get_session).
    voice = _get_voice_clip_or_404(voice_clip_id, db) if voice_clip_id else None
    if voice is not None and (
        not voice.local_path or not os.path.exists(voice.local_path)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Voice {voice.name!r} is missing its file on disk",
        )

    vdir = run_voice_dir(run_id, project_id)
    dest = os.path.join(vdir, "voice.mp3")

    if voice is not None:
        shutil.copy2(voice.local_path, dest)
        log.info(
            "Run %s takes voice %s (%r) from the library", run_id, voice.id, voice.name
        )
        return [dest]

    raw_dest = os.path.join(vdir, "upload.src")
    try:
        _save_upload(voice_file, raw_dest, max_bytes=settings.MAX_UPLOAD_SIZE_MB << 20)
        media_mod.make_voice_reference(raw_dest, dest)
    except media_mod.MediaError as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(
            status_code=400, detail=f"Unusable voice reference: {exc}"
        ) from exc
    except Exception:
        shutil.rmtree(vdir, ignore_errors=True)
        raise
    finally:
        if os.path.exists(raw_dest):
            os.remove(raw_dest)

    log.info("Run %s takes an uploaded voice reference", run_id)
    return [dest]


def _parse_stitch_plan(raw: Optional[str], pid: str, db: Session) -> Optional[list]:
    """Validate + normalize the stitch_plan form field of a run.

    Returns None for "stitch all segments" (absent field, or a custom plan
    that turns out to equal the default order — keeps the classic full-track
    audio path), [] for "don't stitch", else the validated entry list.
    Raises HTTPException(400) on malformed input.
    """
    if raw is None or not raw.strip():
        return None
    try:
        plan = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail="stitch_plan must be valid JSON"
        ) from exc
    if plan is None:
        return None
    if not isinstance(plan, list):
        raise HTTPException(status_code=400, detail="stitch_plan must be a JSON list")
    if not plan:
        return []

    seg_ids = [
        s.id
        for s in db.execute(
            select(SegmentDef)
            .where(SegmentDef.project_id == pid)
            .order_by(SegmentDef.index)
        )
        .scalars()
        .all()
    ]
    seg_id_set = set(seg_ids)

    normalized: list = []
    seen_segments: set = set()
    for pos, entry in enumerate(plan):
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=400, detail=f"stitch_plan[{pos}] must be an object"
            )
        kind = entry.get("kind")
        if kind == "segment":
            sd_id = entry.get("segment_def_id")
            if sd_id not in seg_id_set:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"stitch_plan[{pos}]: segment_def_id {sd_id!r} does not "
                        "belong to this project"
                    ),
                )
            if sd_id in seen_segments:
                raise HTTPException(
                    status_code=400,
                    detail=f"stitch_plan[{pos}]: segment {sd_id!r} appears twice",
                )
            seen_segments.add(sd_id)
            normalized.append({"kind": "segment", "segment_def_id": sd_id})
        elif kind == "library":
            clip_id = entry.get("clip_id")
            if not clip_id or db.get(LibraryClip, clip_id) is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"stitch_plan[{pos}]: unknown library clip {clip_id!r}",
                )
            normalized.append({"kind": "library", "clip_id": clip_id})
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"stitch_plan[{pos}]: kind must be 'segment' or 'library', "
                    f"got {kind!r}"
                ),
            )

    # A plan that is exactly "every segment, original order, nothing else" IS
    # the default — store None so the classic stitch path (and its full-track
    # original audio) is used.
    if all(e["kind"] == "segment" for e in normalized) and [
        e["segment_def_id"] for e in normalized
    ] == seg_ids:
        return None

    return normalized


# ---------------------------------------------------------------------------
# PIP insert — extract endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{pid}/extract-insert",
    status_code=status.HTTP_201_CREATED,
    response_model=ExtractInsertResponse,
    dependencies=[GATED],
)
def extract_insert(
    pid: str,
    x: int = Form(...),
    y: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    start_sec: float = Form(...),
    end_sec: float = Form(...),
    corner_radius_px: int = Form(0),
    db: Session = Depends(get_db),
) -> ExtractInsertResponse:
    """Extract a picture-in-picture insert rect out of a parent project's source
    video into a brand-new child VideoProject.

    The child is a normal project (``source_type="upload"``) that flows through
    the existing analyze -> segment review -> Run pipeline unchanged; the
    ``InsertExtraction`` row is only the provenance/placement back-link (see
    docs/pip-insert.md). Analysis is enqueued for the child exactly like a
    normal upload.

    Validates: the parent project exists and has a probed source video on disk;
    the rect fits inside the parent's probed dimensions; the time window is
    within [0, duration_sec]; corner_radius_px is non-negative. On a crop
    failure (MediaError) the child project is not left half-created — nothing
    was committed, and any partial output file is removed from disk.
    """
    parent = _get_project_or_404(pid, db)

    if not parent.source_local_path or not os.path.exists(parent.source_local_path):
        raise HTTPException(
            status_code=409,
            detail="Cannot extract insert: parent project has no source video on disk",
        )
    if parent.width is None or parent.height is None or parent.duration_sec is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot extract insert: parent project has not finished analyzing "
                "yet (no probed dimensions/duration)"
            ),
        )

    if corner_radius_px < 0:
        raise HTTPException(
            status_code=400, detail="corner_radius_px must be >= 0"
        )
    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=400, detail="width and height must be positive"
        )
    if (
        x < 0
        or y < 0
        or x + width > parent.width
        or y + height > parent.height
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"rect (x={x}, y={y}, w={width}, h={height}) does not fit inside "
                f"parent dimensions {parent.width}x{parent.height}"
            ),
        )
    if not (0 <= start_sec < end_sec <= parent.duration_sec):
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid time range [{start_sec}, {end_sec}]; must satisfy "
                f"0 <= start_sec < end_sec <= duration_sec ({parent.duration_sec})"
            ),
        )

    child_id = str(uuid.uuid4())
    parent_label = parent.name or parent.source_ref
    child_name = f"{parent_label} — insert"[:255]
    child_src_path = project_source_path(child_id, "mp4")

    # Re-validate the inherited cap instead of copying the column verbatim: a
    # value that predates the current bounds (or a NaN from before the interval
    # check) would otherwise be duplicated into the child and break ITS analysis
    # too. Falls back to NULL, never a 400 — see _inherited_max_segment_sec.
    child_max_segment_sec = _inherited_max_segment_sec(
        parent.max_segment_sec, context="extract_insert", parent_id=pid
    )

    child = VideoProject(
        id=child_id,
        name=child_name,
        source_type="upload",
        source_ref=f"{parent.source_ref} (insert)"[:1024],
        source_local_path=child_src_path,
        # The child is a crop of the parent's video, same soundtrack — carry
        # the parent's "remove original audio" choice over with it.
        mute_source=parent.mute_source,
        # Same reasoning for the segmentation cap: the child is analyzed
        # (enqueued below) as soon as the crop lands, and it is meant to be run
        # on the same model as the parent, so it should be cut to the same
        # maximum segment length rather than silently dropping to the
        # universal default.
        max_segment_sec=child_max_segment_sec,
        status=ProjectStatus.created,
    )
    child_analysis_token = _admit_analysis(child)  # AICP-350: counted at commit
    db.add(child)
    db.flush()  # assign id/defaults; NOT committed yet — a crop failure below
    # leaves nothing persisted (the dependency rolls back on the raised
    # HTTPException), so no half-created child project survives.

    try:
        even_width, even_height = media_mod.crop_clip(
            parent.source_local_path,
            start_sec,
            end_sec,
            x,
            y,
            width,
            height,
            child_src_path,
        )
    except media_mod.MediaError as exc:
        shutil.rmtree(project_dir(child_id), ignore_errors=True)
        log.warning("extract_insert: crop_clip failed for parent %s: %s", pid, exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to extract insert: {exc}"
        ) from exc

    extraction = InsertExtraction(
        parent_project_id=pid,
        child_project_id=child_id,
        x=x,
        y=y,
        width=even_width,
        height=even_height,
        start_sec=start_sec,
        end_sec=end_sec,
        corner_radius_px=corner_radius_px,
    )
    db.add(extraction)
    db.commit()

    _enqueue_analysis(child_id, child_analysis_token)

    log.info(
        "Extracted insert from project %s -> child project %s (rect=%d,%d,%d,%d)",
        pid, child_id, x, y, even_width, even_height,
    )
    status_val = child.status.value if hasattr(child.status, "value") else str(child.status)
    return ExtractInsertResponse(
        child_project_id=child_id, extraction_id=extraction.id, status=status_val
    )


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reference-path containment (AICP-148)
#
# Reference lists mix public URLs with LOCAL paths the server wrote (uploads,
# staged frames, GPT Image 2 output), and everything downstream — the image
# endpoints below and resolve_reference_urls' kie.ai upload — opens a local
# entry as a file. So a path is only ever accepted from a REQUEST when it is
# one the server itself handed out: a staged frame of this project (the New
# Run / Regenerate frame picker posts those back) or an entry the segment's
# override already holds (the override form re-posts its thumbnails). Any
# other non-URL entry is a 400. Serving re-checks containment regardless, so
# rows written before this guard 404 instead of leaking.
# ---------------------------------------------------------------------------


def _is_remote_ref(value) -> bool:
    """True for an http(s) URL with a host — Drive share links included.

    Mirrors ``pipeline_common._resolve_one_ref``'s own test (a case-sensitive
    prefix), so nothing accepted here is later mistaken for a local path.
    """
    if not isinstance(value, str):
        return False
    if not (value.startswith("http://") or value.startswith("https://")):
        return False
    try:
        return bool(urlparse(value).netloc)
    except ValueError:
        return False


def _staged_frame_realpath(pid: str, value) -> Optional[str]:
    """Realpath of *value* if it resolves inside project *pid*'s staged_frames/.

    Containment only, not existence: a frame that has since vanished is stored
    as given (as before) and fails at submit with a message naming the slot.
    """
    if not isinstance(value, str) or not os.path.isabs(value):
        return None
    return path_within(value, os.path.join(project_dir(pid), "staged_frames"))


def _reject_bad_reference_urls(
    values: list, *, field: str, local_ok=None
) -> None:
    """400 on any entry that is not an http(s) URL (or accepted by *local_ok*)."""
    for v in values:
        if _is_remote_ref(v):
            continue
        if local_ok is not None and local_ok(v):
            continue
        shown = v if len(v) <= 200 else v[:200] + "…"
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field}: {shown!r} is not an http(s) URL. Reference images "
                "must be uploaded files, http(s)/Google Drive links, or frames "
                "added with \"Add frame from video\"."
            ),
        )


def _existing_override_ok(existing):
    """*local_ok* for a segment override: a local path the override ALREADY
    holds (re-posted by the form) and that lies under DATA_DIR."""
    held = {e for e in (existing or []) if isinstance(e, str)}

    def ok(v: str) -> bool:
        return v in held and path_within(v, data_dir_root()) is not None

    return ok


def _copy_staged_frame(pid: str, url: str, refs_dir: str) -> str:
    """Copy a staged frame of *pid* into *refs_dir*; other values pass through.

    The copy is made from the frame's realpath, so a traversal that merely
    STARTS with the staged_frames prefix is never followed.
    """
    real = _staged_frame_realpath(pid, url)
    if real is None or not os.path.isfile(real):
        return url
    dest = os.path.join(refs_dir, _safe_filename(os.path.basename(real)))
    shutil.copy2(real, dest)
    return dest


def _serve_local_reference(path, *, what: str, media_type: Optional[str] = None):
    """FileResponse for a stored reference path — 404 unless under DATA_DIR."""
    try:
        real = ensure_under_data_dir(path)
    except PathOutsideDataDir:
        raise HTTPException(status_code=404, detail=f"{what} file not found")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail=f"{what} file not found")
    if media_type:
        return FileResponse(real, media_type=media_type)
    return FileResponse(real)


def _assert_trims_are_source_frames(
    pid: str, ref_files: list, ref_urls: list[str]
) -> None:
    """Reject trims that did not come from this project's source video.

    With GPT Image 2 preprocessing on, the general reference field stops being
    "reference images" and becomes the TRIMS: the video frames whose pose and
    framing the character photo is transferred onto. Anything else in there is
    a mistake, and an expensive one — a stray image generates a junk ref that is
    then handed to Seedance for EVERY segment. The classic slip is dropping the
    character photo in a second time, which yields "the character photo made to
    look like the character photo".

    So the only accepted trims are frames the operator staged through
    ``POST /projects/{pid}/stage-frame`` ("Add frame from video"), which live
    under ``projects/<pid>/staged_frames/``. Uploads and outside URLs are
    refused. The CHARACTER photo is unaffected — it has its own field and is
    still any file or URL.

    Enforced here and mirrored in the New Run form, which locks those inputs
    while the toggle is on (partials/project_status_content.html).
    """
    named_files = [rf for rf in (ref_files or []) if getattr(rf, "filename", None)]
    if named_files:
        raise HTTPException(
            status_code=400,
            detail=(
                "With GPT Image preprocessing on, trims must be frames from "
                "the source video — use \"Add frame from video\" instead of "
                "uploading files. The character photo has its own field."
            ),
        )
    stray = [u for u in ref_urls if _staged_frame_realpath(pid, u) is None]
    if stray:
        raise HTTPException(
            status_code=400,
            detail=(
                "With GPT Image preprocessing on, trims must be frames from "
                f"the source video, but {len(stray)} of {len(ref_urls)} are not "
                f"(first: {stray[0]!r}). Use \"Add frame from video\"; the "
                "character photo has its own field."
            ),
        )


# Same tolerance pipeline_v2._submit_swap_segment_isolated compares against its
# own ceiling with. Kept identical on purpose: a segment this module accepts
# must never be one that submit then skips, and vice versa.
_SEGMENT_LEN_EPS_SEC = 0.05


def _overlong_swap_segments(
    entries, model: Optional[str], plan: Optional[list] = None
) -> list[tuple[int, float]]:
    """``[(index, duration_sec), …]`` for every SWAP entry *model* cannot
    generate in one clip, longest first.

    *entries* is either ``SegmentDef``-shaped ORM rows (a project's live
    partition) or ``Run.segment_snapshot`` dicts — the two timelines a run can
    be launched against — so both launch paths ask the same question of
    whichever one their run will actually freeze. Non-swap entries and entries
    with unusable bounds are skipped: this function's job is to find segments
    that are certainly un-generatable, not to re-validate the partition.

    *plan* is the run's ``stitch_plan``. A CUSTOM plan generates only the swap
    segments it lists — the ones it leaves out are never submitted and never
    cost anything (``pipeline_v2._active_swap_segment_ids``), so an over-long
    segment that the plan drops is harmless and must not block the launch.
    ``None`` (default plan) and ``[]`` (no-stitch) both process every swap
    segment, so both are treated as "no filter", exactly as that function does.
    """
    max_clip = ai_models.max_clip_sec(model)
    planned_ids: Optional[set] = None
    if plan:
        planned_ids = {
            e.get("segment_def_id")
            for e in plan
            if isinstance(e, dict) and e.get("kind") == "segment"
        }
    offenders: list[tuple[int, float]] = []
    for entry in entries or []:
        if isinstance(entry, dict):
            action, index, sd_id = (
                entry.get("action"), entry.get("index"), entry.get("id"),
            )
            start, end = entry.get("start_sec"), entry.get("end_sec")
        else:
            action, index, sd_id = entry.action, entry.index, entry.id
            start, end = entry.start_sec, entry.end_sec
        if action != "swap":
            continue
        if planned_ids is not None and sd_id not in planned_ids:
            continue
        try:
            duration = float(end) - float(start)
            index = int(index)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > max_clip + _SEGMENT_LEN_EPS_SEC:
            offenders.append((index, duration))
    return sorted(offenders, key=lambda pair: -pair[1])


def _assert_no_overlong_swap_segments(
    entries, model: Optional[str], *, what: str, plan: Optional[list] = None
) -> None:
    """Refuse to LAUNCH a run whose timeline holds a swap segment longer than
    the model's per-clip ceiling. 400 if it does.

    Such a segment cannot be generated at all: submit marks it failed with
    ``RunSegment.source_fallback`` and the stitch substitutes the ORIGINAL,
    un-swapped footage, so the run still reaches `done` and delivers a video
    that merely LOOKS finished. That is not a theoretical failure — it shipped
    a 59s creative with its back 31.5s never swapped, and nothing about the
    delivered file said so.

    Deliberately a hard 400 rather than another warning. The New Run form has
    warned about exactly these segments client-side since the feature shipped
    (partials/project_status_content.html, ``checkSegmentLengths``); a warning
    is one click to ignore, which is how the silent half-swap reaches Drive.
    The escape hatches are unchanged and both explicit: split the segment, or
    mark it `keep` — which delivers the very same original footage, but says
    so on the timeline instead of hiding it behind a failed swap.

    *what* completes "Cannot {what}: …" so each caller names its own action.
    """
    offenders = _overlong_swap_segments(entries, model, plan)
    if not offenders:
        return
    spec = ai_models.spec_for(model)
    listed = ", ".join(f"#{index} ({duration:.1f}s)" for index, duration in offenders)
    raise HTTPException(
        status_code=400,
        detail=(
            f"Cannot {what}: {spec.label} generates at most "
            f"{spec.max_clip_sec:g}s per segment, but {len(offenders)} swap "
            f"segment(s) are longer: {listed}. They cannot be swapped at all — "
            "the run would deliver the ORIGINAL, un-swapped footage there. "
            f"Split them to {spec.max_clip_sec:g}s or less, mark them \"keep\", "
            "or choose a model with a longer per-clip limit."
        ),
    )


def _whole_video_roll_warnings(
    project: VideoProject, entries, model: Optional[str], plan: Optional[list] = None
) -> list[str]:
    """Advisories for a WHOLE-VIDEO localisation launch (AICP-340). Never raises.

    Analysis keeps every whole-video segment within
    ``pipeline_v2.whole_video_target_sec`` (14.72 s, the length Seedance 2.5
    intermittently truncates to), but the clip actually uploaded is the
    segment PLUS its retained pre/post-roll — ``localisation.clip_window``,
    the submit path's own window and roll trim for *model*'s
    ``max_clip_sec`` (via ``localisation.resolve_prompt_model``). A clip that
    rolls push past the target is still generatable, so this only warns.
    Same plan filter as :func:`_overlong_swap_segments`.
    """
    try:
        if spec_for(project.project_type).key != LOCALISATION:
            return []
        if getattr(project, "localisation_scope", None) != LOCALISATION_SCOPE_WHOLE:
            return []
        from .pipeline_v2 import whole_video_target_sec

        target = whole_video_target_sec(project.max_segment_sec)
        max_clip = localisation.resolve_prompt_model(project.project_type, model).max_clip_sec
        planned_ids: Optional[set] = None
        if plan:
            planned_ids = {
                e.get("segment_def_id")
                for e in plan
                if isinstance(e, dict) and e.get("kind") == "segment"
            }
        over: list[str] = []
        for entry in entries or []:
            get = entry.get if isinstance(entry, dict) else (
                lambda k, d=None, _e=entry: getattr(_e, k, d)
            )
            if get("action") != "swap":
                continue
            if planned_ids is not None and get("id") not in planned_ids:
                continue
            try:
                start, end = float(get("start_sec")), float(get("end_sec"))
            except (TypeError, ValueError):
                continue
            clip_start, clip_end = localisation.clip_window(
                start, end,
                float(get("pre_roll_sec", 0.0) or 0.0),
                float(get("post_roll_sec", 0.0) or 0.0),
                duration_sec=project.duration_sec,
                max_clip_sec=max_clip,
            )
            if clip_end - clip_start > target + 1e-6:
                over.append(
                    f"#{get('index')} ({clip_end - clip_start:.2f}s, segment "
                    f"{end - start:.2f}s)"
                )
        if not over:
            return []
        return [
            f"Whole-video localisation: {len(over)} swap clip(s) exceed the "
            f"{target:g}s whole-video target once pre/post-roll is added: "
            f"{', '.join(over)}. Seedance 2.5 intermittently truncates output "
            f"to 14.72s, so the end of these clips may be lost. Reduce the "
            f"rolls (or shorten the segment) to stay within {target:g}s. "
            f"The run was launched anyway."
        ]
    except Exception:  # advisory only — must never block or break a launch
        log.exception("whole-video roll warning check failed for project %s", project.id)
        return []


def _create_images_only_run(
    project: VideoProject,
    db: Session,
    *,
    model: str,
    resolution: Optional[str],
    audio_mode: str,
    language: Optional[str],
    name: Optional[str],
    gdrive_folder_id: Optional[str],
    stitch_plan: Optional[str],
    keep_source_end_card: bool,
    sheet_overrides: Optional[str],
    v2v_fields: dict,
) -> RunCreateResponse:
    """The images-only launch of ``create_run`` (design §6.1, Q6: launching
    is the approval of the keyframes). Accepted fields: ``model``
    (``kling-4-0`` default of the type, or ``seedance-2-5``), ``resolution``
    (``720p`` default), ``language``, ``name``, ``gdrive_folder_id``,
    ``stitch_plan`` (absent = every segment, the trailing end card replaced by
    a language-matched library clip), ``keep_source_end_card`` and
    ``sheet_overrides`` (JSON ``{segment_def_id: sheet_key}``). ``prompt`` is
    required by the form signature and ignored (prompts are rendered per
    chunk). Forced: ``audio_mode='original'``, no preprocessing, no
    references, no voice, no per-segment prompts — any of those is a 400."""
    from . import rebuild_launch

    used = [k for k, v in v2v_fields.items() if v]
    if used:
        raise HTTPException(
            status_code=400,
            detail=f"not supported for a rebuild run (images-only): {', '.join(sorted(used))}",
        )
    if audio_mode not in (None, "", "original"):
        raise HTTPException(
            status_code=400,
            detail="a rebuild run keeps the original soundtrack (audio_mode='original')",
        )
    language_val = _validate_language(language)
    overrides: dict = {}
    if sheet_overrides and sheet_overrides.strip():
        try:
            overrides = json.loads(sheet_overrides)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400,
                                detail="chunk_sheets must be valid JSON") from exc
        if not isinstance(overrides, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in overrides.items()
        ):
            raise HTTPException(status_code=400,
                                detail="chunk_sheets must be {segment_def_id: sheet_key}")
    given = stitch_plan is not None and bool(stitch_plan.strip())
    parsed_plan = _parse_stitch_plan(stitch_plan, project.id, db) if given else None
    if given and keep_source_end_card:
        raise HTTPException(status_code=400,
                            detail="keep_source_end_card only applies to the default plan")
    _budget_precheck(db, "new run")
    req = rebuild_launch.LaunchRequest(
        model=model, resolution=resolution or None, language=language_val, name=name,
        gdrive_folder_id=gdrive_folder_id, stitch_plan=parsed_plan, stitch_plan_given=given,
        keep_source_end_card=bool(keep_source_end_card), sheet_overrides=overrides,
    )
    try:
        run, warnings = _spend_reserve(rebuild_launch.launch_run, db, project, req)
    except rebuild_launch.LaunchRefused as exc:
        db.rollback()
        # A plain string: the 2a launch form shows `detail` as is (the
        # missing preconditions are joined into it).
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except HTTPException:
        db.rollback()
        raise
    try:
        db.commit()
    except Exception:
        db.rollback()
        rebuild_launch.discard_run_files(run)
        raise
    enqueue_process_run(run.id)
    log.info("Created images-only run %s for project %s", run.id, project.id)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    return RunCreateResponse(run_id=run.id, status=status_val, warnings=warnings)


@router.post(
    "/projects/{pid}/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=RunCreateResponse,
    dependencies=[GATED],
)
def create_run(
    pid: str,
    prompt: str = Form(...),
    name: Optional[str] = Form(None),
    model: str = Form("seedance"),
    # None = not sent: settings.DEFAULT_RESOLUTION for a v2v run, 720p for a
    # rebuild run (design §15 Q1).
    resolution: Optional[str] = Form(None),
    audio_mode: str = Form("original"),
    language: Optional[str] = Form(None),
    gdrive_folder_id: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    segment_prompts: Optional[str] = Form(None),
    stitch_plan: Optional[str] = Form(None),
    preprocess_enabled: bool = Form(False),
    preprocess_mode: str = Form("per_image"),
    preprocess_image_prompts: Optional[str] = Form(None),
    preprocess_reference_file: Optional[UploadFile] = File(None),
    preprocess_reference_url: Optional[str] = Form(None),
    voice_clip_id: Optional[str] = Form(None),
    voice_file: Optional[UploadFile] = File(None),
    keep_source_end_card: bool = Form(False),
    chunk_sheets: Optional[str] = Form(None),
    sheet_overrides: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunCreateResponse:
    """Create a new Run under a project (one character attempt).

    The project must be in *ready* status. Exactly one character prompt is
    required. Reference images (files + URLs) are capped at MAX_REFERENCE_IMAGES.

    *segment_prompts* is an optional JSON object ``{segment_def_id: prompt_text}``
    of per-segment prompts: the text becomes the PRIMARY prompt for that swap
    segment on the very first run, fully replacing the run prompt
    (blank/absent → uses the run prompt).

    *stitch_plan* is an optional JSON list controlling the final stitch:
    absent/null → stitch every segment in order (default); ``[]`` → process
    swap segments but produce no final video; otherwise an ordered list of
    ``{"kind": "segment", "segment_def_id": …}`` / ``{"kind": "library",
    "clip_id": …}`` entries (e.g. to replace the outro with a library clip).
    Swap segments left out of a custom plan are not generated at all.

    *preprocess_enabled* opts into the GPT Image 2 (image-to-image) reference
    preprocessing phase (see docs/gpt-image-2-preprocess.md). When true, a
    Character reference photo is REQUIRED — provide it via
    *preprocess_reference_file* (upload) or *preprocess_reference_url* (URL
    or staged-frame path). It is stored at ``reference_image_urls[0]`` and
    sent as context to every GPT Image 2 task (it is NOT transformed). The
    general *reference_files* / *reference_urls* become the TRIMS
    (``reference_image_urls[1:]``), each transformed by GPT Image 2.
    *preprocess_image_prompts* must be a JSON list of TRIM prompts (one per
    trim, in order); an empty string means "pass this trim through unchanged".
    The run enters ``preprocessing`` instead of ``queued`` and the preprocessed
    images (or originals for pass-through/failed refs) are what Seedance/Gemini
    Omni ultimately receive.

    *voice_clip_id* (a library voice) or *voice_file* (an ad-hoc upload, not
    both) attaches a voice reference: the sample whose timbre the generated
    speech takes, instead of the arbitrary voice the model invents per
    generation. Only models with ``reference_audio_urls`` accept one (Seedance
    2.5); asking for it on any other is a 400.
    """
    # The autopilot calls this function directly (app/autopilot/actions.py)
    # without the rebuild-only fields, which then hold their Form() defaults:
    # read anything that is not a real value as "not sent".
    keep_source_end_card = keep_source_end_card is True
    sheet_overrides = sheet_overrides if isinstance(sheet_overrides, str) else None
    chunk_sheets = chunk_sheets if isinstance(chunk_sheets, str) else None
    # The 2a launch form posts `chunk_sheets`; `sheet_overrides` is its alias.
    if chunk_sheets and chunk_sheets.strip():
        if sheet_overrides and sheet_overrides.strip():
            raise HTTPException(status_code=400,
                                detail="send chunk_sheets or sheet_overrides, not both")
        sheet_overrides = chunk_sheets
    resolution = resolution if isinstance(resolution, str) else None
    project = _get_project_or_404(pid, db)
    if not spec_for(project.project_type).supports_runs:
        raise HTTPException(
            status_code=409,
            detail=f"A {spec_for(project.project_type).label} project has no runs",
        )
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot create run: project status is {project.status!r}, "
                "expected 'ready'"
            ),
        )
    # Images-only (rebuild) runs — docs/rebuild-from-reference-design.md §6.1.
    # Behind REBUILD_ENABLED. The launch (snapshot, run-owned keyframe copies,
    # pending RunSegments, reservation last) is app/rebuild_launch.py; the
    # v2v-only fields below (prompt, references, preprocessing, voice,
    # segment prompts, audio mode) are refused or ignored for it.
    if spec_for(project.project_type).generation_mode == GENERATION_IMAGES_ONLY:
        if not settings.REBUILD_ENABLED:
            raise HTTPException(
                status_code=409,
                detail="Rebuild from reference is not enabled on this server "
                       "(REBUILD_ENABLED)",
            )
        return _create_images_only_run(
            project, db, model=model, resolution=resolution, audio_mode=audio_mode,
            language=language, name=name, gdrive_folder_id=gdrive_folder_id,
            stitch_plan=stitch_plan, keep_source_end_card=keep_source_end_card,
            sheet_overrides=sheet_overrides,
            v2v_fields={
                "reference_files": [f for f in (reference_files or [])
                                    if getattr(f, "filename", None)],
                "reference_urls": (reference_urls or "").strip(),
                "segment_prompts": (segment_prompts or "").strip(),
                "preprocess_enabled": preprocess_enabled,
                "preprocess_reference_file": getattr(preprocess_reference_file,
                                                     "filename", None),
                "preprocess_reference_url": (preprocess_reference_url or "").strip(),
                "voice_clip_id": (voice_clip_id or "").strip(),
                "voice_file": getattr(voice_file, "filename", None),
            },
        )
    if keep_source_end_card or (sheet_overrides or "").strip():
        raise HTTPException(
            status_code=400,
            detail="keep_source_end_card / chunk_sheets apply to rebuild projects only",
        )
    if resolution is None:
        resolution = settings.DEFAULT_RESOLUTION
    _budget_precheck(db, "new run")

    # Validate model against the registry (app/ai_models.py) — adding a spec
    # there is all it takes for a new backend to become selectable here.
    # Only v2v models on a v2v project: an images-only model (Kling 4.0) is
    # refused on every non-rebuild project, flag or no flag.
    if model not in ai_models.V2V_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {sorted(ai_models.V2V_MODELS)}",
        )

    # Validate resolution (must be allowed for the chosen model)
    allowed_res = ai_models.resolutions_for(model)
    if resolution not in allowed_res:
        raise HTTPException(
            status_code=400,
            detail=(
                f"resolution {resolution!r} not allowed for model {model!r}; "
                f"choose one of {sorted(allowed_res)}"
            ),
        )

    # Validate audio_mode
    if audio_mode not in _VALID_AUDIO_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"audio_mode must be one of {sorted(_VALID_AUDIO_MODES)}",
        )
    # A model that generates no audio (Gemini Omni) can only ever be stitched
    # against the original source track — force "original" regardless of what
    # was requested. Driven off the registry so a future silent backend is
    # handled without another literal model name here.
    if not ai_models.spec_for(model).produces_audio:
        audio_mode = "original"

    # Validate the language. A blank field is allowed (NULL = English, the
    # pre-feature behaviour); an unrecognised code is rejected rather than
    # silently stored, because downstream it would not be recognised either —
    # the creative would just rename as English with the bad code stuck in the
    # filename core.
    language_val = _validate_language(language)

    # Validate reference image count.
    #
    # When preprocessing is ENABLED, the general reference_files/reference_urls
    # are the TRIMS and a separate Character reference photo is REQUIRED
    # (preprocess_reference_file / preprocess_reference_url) — it is stored at
    # reference_image_urls[0] and sent as context to every GPT Image 2 task
    # (never transformed). Total stored refs = 1 (char ref) + len(trims).
    #
    # When preprocessing is DISABLED, reference_files/reference_urls are the
    # ordinary reference images (existing behavior).
    # An untouched <input type="file"> still posts a part, with an empty
    # filename — count those as refs and the prompts list silently misaligns.
    ref_files = [f for f in (reference_files or []) if getattr(f, "filename", None)]
    ref_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    if preprocess_enabled:
        n_trims = len(ref_files) + len(ref_urls)
        total_refs = 1 + n_trims
    else:
        n_trims = None
        total_refs = len(ref_files) + len(ref_urls)
    if total_refs > settings.MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many reference images: got {total_refs}, "
                f"max {settings.MAX_REFERENCE_IMAGES}"
            ),
        )

    # Optional GPT Image 2 preprocessing: parse the per-trim prompts list.
    #
    # R1-new prompt model:
    #   - reference_image_urls[0] is the CHARACTER REF (context, never
    #     transformed). prompts[0] is ALWAYS "" (forced) regardless of what the
    #     UI sends.
    #   - reference_image_urls[1:] are TRIMS; the textarea lines correspond to
    #     trims only. The API builds the full prompts list by prepending "" for
    #     index 0.
    #
    # Mode-aware:
    #   per_image   — prompts is a length-(1+N) list parallel to refs (index 0="").
    #                 When 1 trim-prompt + N>1 trims, REUSE that prompt for every
    #                 trim; fewer trim-prompts than trims → pad with "" (pass-
    #                 through); more → truncate.
    #   all_as_refs — a SINGLE task with input_urls = ALL refs and a single
    #                 prompt; preprocess_image_prompts is stored as a length-1
    #                 list whose single entry is the trim prompt (the pipeline
    #                 reads prompts[0] as the single prompt).
    if preprocess_enabled and preprocess_mode not in ("per_image", "all_as_refs"):
        raise HTTPException(
            status_code=400,
            detail="preprocess_mode must be one of ['per_image', 'all_as_refs']",
        )

    # When preprocessing is enabled, a Character reference photo is REQUIRED.
    if preprocess_enabled and (
        not preprocess_reference_file
        or not getattr(preprocess_reference_file, "filename", None)
    ) and not (preprocess_reference_url and preprocess_reference_url.strip()):
        raise HTTPException(
            status_code=400,
            detail=(
                "GPT Image preprocessing requires a Character reference "
                "photo — add one in the preprocess panel."
            ),
        )

    parsed_pp_prompts: Optional[list] = None
    if preprocess_enabled:
        if preprocess_image_prompts:
            try:
                trim_prompts = json.loads(preprocess_image_prompts)
            except (ValueError, TypeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="preprocess_image_prompts must be valid JSON",
                ) from exc
            if not isinstance(trim_prompts, list):
                raise HTTPException(
                    status_code=400,
                    detail="preprocess_image_prompts must be a JSON list",
                )
        else:
            trim_prompts = []

        # n_trims is the number of trims (refs[1:]).
        if preprocess_mode == "all_as_refs":
            # Single prompt for the one task. Stored as a length-1 list; the
            # pipeline reads prompts[0] as the single prompt.
            single = trim_prompts[0] if trim_prompts else ""
            parsed_pp_prompts = [single]
        else:
            # per_image — prompts is length (1 + n_trims), index 0 = "".
            if n_trims == 0:
                # No trims → just the char ref slot, forced "".
                parsed_pp_prompts = [""]
            else:
                if len(trim_prompts) == 1 and n_trims > 1:
                    trim_prompts = [trim_prompts[0]] * n_trims
                elif len(trim_prompts) < n_trims:
                    trim_prompts = trim_prompts + [""] * (n_trims - len(trim_prompts))
                elif len(trim_prompts) > n_trims:
                    trim_prompts = trim_prompts[:n_trims]
                parsed_pp_prompts = [""] + list(trim_prompts)

        # Semantic rule, checked after the request-shape validation above so a
        # malformed prompts list still reports itself first: with preprocessing
        # on, the trims are frames of THIS video and nothing else.
        _assert_trims_are_source_frames(pid, ref_files, ref_urls)

    # AICP-148: a typed reference is a URL; the only local path a request may
    # carry is a staged frame of this project (the frame picker posts those).
    _staged_ok = lambda v: _staged_frame_realpath(pid, v) is not None  # noqa: E731
    _reject_bad_reference_urls(ref_urls, field="reference_urls", local_ok=_staged_ok)
    if (
        preprocess_enabled
        and not (preprocess_reference_file is not None and preprocess_reference_file.filename)
        and preprocess_reference_url and preprocess_reference_url.strip()
    ):
        _reject_bad_reference_urls(
            [preprocess_reference_url.strip()],
            field="preprocess_reference_url", local_ok=_staged_ok,
        )

    parsed_stitch_plan = _parse_stitch_plan(stitch_plan, pid, db)

    # A new run freezes the PROJECT's live partition at its first submit (it is
    # created with segment_snapshot=None), so that partition is what this launch
    # has to be generatable against — filtered by the plan just parsed, since a
    # custom plan may deliberately drop the over-long segment. Placed after
    # every request-shape check above so a malformed plan or a bad model still
    # reports itself first, and still before the Run row and any uploaded
    # reference file is written.
    _assert_no_overlong_swap_segments(
        project.segments, model, what="create run", plan=parsed_stitch_plan,
    )
    launch_warnings = _whole_video_roll_warnings(
        project, project.segments, model, parsed_stitch_plan,
    )
    # 2026-11-03 (docs/segment-fill-batch-design.md §7): a subtitle-removal
    # project in "VOID from the source" mode makes a run that sends nothing to
    # the model; only the inputs that run can use are accepted (R2-8).
    void_source = (project.project_type == SUBTITLE_REMOVAL
                   and project_types.caption_removal_method(project)
                   == project_types.CAPTION_REMOVAL_VOID_SOURCE)
    if void_source:
        bad = [what for what, on in (
            ("GPT Image preprocessing", preprocess_enabled),
            ("reference images", bool(ref_files or ref_urls)),
            ("a custom stitch plan", parsed_stitch_plan is not None),
            ("audio_mode other than original", audio_mode != "original"),
            ("a voice reference", bool((voice_clip_id or "").strip()
                                       or getattr(voice_file, "filename", None))),
        ) if on]
        if bad:
            raise HTTPException(status_code=422, detail=(
                "this project removes captions with VOID from the source (project setting "
                "'Caption removal'): a run takes no %s. Switch the project to Omni for "
                "those." % ", ".join(bad)))

    run_id = str(uuid.uuid4())

    resolved_folder_id = gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID or None

    run = Run(
        id=run_id,
        project_id=pid,
        name=name,
        prompt=prompt,
        model=model,
        resolution=resolution,
        audio_mode=audio_mode,
        language=language_val,
        gdrive_folder_id=resolved_folder_id,
        stitch_plan=parsed_stitch_plan,
        status=RunStatus.created,
        reference_image_urls=[],
        preprocess_enabled=preprocess_enabled,
        preprocess_mode=preprocess_mode,
        preprocess_image_prompts=parsed_pp_prompts,
    )
    if void_source:
        run.generation_mode = project_types.GENERATION_VOID_SOURCE
    db.add(run)
    db.flush()  # persist id before saving reference files

    # Save reference files into project/run dir
    max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
    saved_ref_paths: list[str] = []
    refs_dir = os.path.join(project_dir(pid), "runs", run_id, "references")
    os.makedirs(refs_dir, exist_ok=True)

    def _copy_staged(url: str) -> str:
        """Copy a staged-frame local path into the run's references dir for
        durability; return the destination path. URLs are kept as-is."""
        return _copy_staged_frame(pid, url, refs_dir)

    for rf in ref_files:
        safe = _safe_filename(rf.filename or f"ref_{len(saved_ref_paths)}.jpg")
        dest = os.path.join(refs_dir, safe)
        _save_upload(rf, dest, max_bytes=max_bytes)
        saved_ref_paths.append(dest)

    durable_ref_urls: list[str] = [_copy_staged(u) for u in ref_urls]

    if preprocess_enabled:
        # Resolve the Character reference photo (required, already validated
        # above). Stored at reference_image_urls[0]; the trims saved above are
        # [1:].
        char_ref_path: str
        if preprocess_reference_file is not None and preprocess_reference_file.filename:
            safe = _safe_filename(preprocess_reference_file.filename or "char_ref.jpg")
            dest = os.path.join(refs_dir, safe)
            _save_upload(preprocess_reference_file, dest, max_bytes=max_bytes)
            char_ref_path = dest
        else:
            char_ref_path = _copy_staged(preprocess_reference_url.strip())
        run.reference_image_urls = [char_ref_path] + saved_ref_paths + durable_ref_urls
    else:
        run.reference_image_urls = saved_ref_paths + durable_ref_urls

    # Voice reference — the run gets its own copy (see _resolve_voice_reference).
    run.reference_audio_urls = _resolve_voice_reference(
        run_id=run_id,
        project_id=pid,
        model=model,
        voice_clip_id=(voice_clip_id or "").strip() or None,
        voice_file=voice_file,
        db=db,
    )

    # Optional per-segment prompts: pre-create RunSegments carrying a
    # prompt_override (= the per-segment text) so the FIRST run already
    # submits tailored prompts. The per-segment prompt is the PRIMARY prompt
    # for that segment (it fully replaces the run prompt, not appended).
    # process_run is idempotent and reuses these.
    if segment_prompts:
        try:
            seg_map = json.loads(segment_prompts)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="segment_prompts must be valid JSON"
            ) from exc
        if isinstance(seg_map, dict) and seg_map:
            swap_defs = {
                s.id: s
                for s in db.execute(
                    select(SegmentDef).where(
                        SegmentDef.project_id == pid, SegmentDef.action == "swap"
                    )
                )
                .scalars()
                .all()
            }
            # Swap segments a custom stitch plan excludes are never processed —
            # don't pre-create RunSegments for them (a pending row on a run
            # that never submits it would sit "pending" forever in the UI).
            active_seg_ids = _plan_active_segment_ids(parsed_stitch_plan)
            for sd_id, extra in seg_map.items():
                text = extra.strip() if isinstance(extra, str) else ""
                sd = swap_defs.get(sd_id)
                if active_seg_ids is not None and sd_id not in active_seg_ids:
                    continue
                if text and sd is not None:
                    db.add(
                        RunSegment(
                            run_id=run_id,
                            segment_def_id=sd_id,
                            index=sd.index,
                            status=SegmentStatus.pending,
                            prompt_override=text,
                        )
                    )

    if void_source:
        prepare = _park_void_source_run(run, project, db)
        db.commit()
        _enqueue_void_source_previews(prepare)
        log.info("Created VOID-from-source run %s for project %s (%d previews)",
                 run_id, pid, len(prepare))
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        return RunCreateResponse(run_id=run_id, status=status_val, warnings=launch_warnings)

    # Transition + enqueue: branch on preprocess_enabled.
    #   preprocess_enabled → created → preprocessing, enqueue run_preprocess_refs
    #   else              → created → queued,      enqueue run_submit_run
    try:
        if preprocess_enabled:
            transition(run, RunStatus.preprocessing)
        else:
            transition(run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _spend_reserve(spend_mod.reserve_new_runs, db, [run], kind="new run")
    db.commit()

    if preprocess_enabled:
        enqueue_preprocess_run(run_id)
    else:
        enqueue_process_run(run_id)

    log.info("Created run %s for project %s", run_id, pid)
    for w in launch_warnings:
        log.warning("Run %s launch advisory: %s", run_id, w)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    return RunCreateResponse(run_id=run_id, status=status_val, warnings=launch_warnings)


def _park_void_source_run(run: Run, project: VideoProject, db: Session) -> list[str]:
    """A VOID-from-the-source run at creation (design §7.2), in the caller's
    transaction: the segment snapshot frozen, a ``pending`` RunSegment per
    active swap segment, ``created -> incomplete`` (PARKED: nothing to submit,
    no reservation, never reaped), and a free source preview of every swap
    segment -- all committed together by the caller (R2-4: a crash between the
    run and its previews would otherwise leave nothing to recover); the
    caller enqueues the prepares after the commit (a never-claimed
    ``preparing`` row is re-enqueued by the recovery tick). Returns the
    preview ids."""
    from .api_segment_fill import FillPreviewRequest, _new_preview
    from .pipeline_v2 import _active_swap_segment_ids, build_segment_snapshot, run_segment_defs

    defs = run_segment_defs(run, project)
    if not run.segment_snapshot:
        run.segment_snapshot = build_segment_snapshot(defs)
    active = _active_swap_segment_ids(run.stitch_plan, defs)
    swap = [sd for sd in defs if sd.action == "swap" and sd.id in active]
    if not swap:
        raise HTTPException(status_code=422, detail=(
            "the project has no swap segment: there is nothing to remove captions from"))
    existing = {rs.segment_def_id for rs in db.execute(
        select(RunSegment).where(RunSegment.run_id == run.id)).scalars()}
    for sd in swap:
        if sd.id not in existing:
            db.add(RunSegment(run_id=run.id, segment_def_id=sd.id, index=sd.index,
                              status=SegmentStatus.pending))
    db.flush()
    try:
        transition(run, RunStatus.incomplete)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    run.error_message = ("VOID from the source: every swap segment's captions are removed by "
                         "a VOID fill from the original clip (free previews first, then one "
                         "start); the run stitches and delivers once every segment has one.")
    db.flush()
    ids = []
    for rs in db.execute(select(RunSegment).where(RunSegment.run_id == run.id)
                         .order_by(RunSegment.index)).scalars():
        job = _new_preview(db, run, project, rs, FillPreviewRequest(basis="source"))
        db.flush()
        ids.append(job.id)
    return ids


def _enqueue_void_source_previews(ids) -> None:
    from .api_segment_fill import _enqueue_prepares

    _enqueue_prepares(ids)


@router.get("/runs/{rid}", response_model=RunResponse)
def get_run(rid: str, db: Session = Depends(get_db)) -> RunResponse:
    """Return full run details."""
    run = _get_run_or_404(rid, db)
    return RunResponse.model_validate(run)


@router.patch("/runs/{rid}", response_model=RunResponse)
def update_run(
    rid: str, body: RunUpdate, db: Session = Depends(get_db)
) -> RunResponse:
    """Update editable run settings (display name, language).

    Changing the language changes the run's short id — and therefore the Job ID
    a later export writes. It is deliberately editable after creation: the
    language is only read at export time, so fixing a wrong pick before
    clicking "Add to Final Creatives" costs nothing. Editing it AFTER an export
    does not rewrite the sheet row (the sheet is the downstream's source of
    truth); fix that row by hand.
    """
    run = _get_run_or_404(rid, db)
    if body.name is not None:
        name = body.name.strip()
        run.name = name[:255] or None
    if body.language is not None:
        # "" clears the language back to unset; a non-empty unknown code is an
        # error (see _validate_language for why we never store one).
        run.language = _validate_language(body.language)
    db.commit()
    db.refresh(run)
    log.info(
        "Updated run %s name=%r language=%r", rid, run.name, run.language
    )
    return RunResponse.model_validate(run)


@router.delete("/runs/{rid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(rid: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a run: DB rows (cascades to RunSegments) + disk.

    Blocked (409) while the run is active (queued/processing/stitching/delivering).
    """
    run = _get_run_or_404(rid, db)
    _lock_run_for_result_mutation(run, db)
    if run.status in _ACTIVE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a run while it is {run.status.value!r}",
        )
    if active_task_for_run(db, rid, run.project_id) is not None:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a run while a Fal provider task for it is active "
                   "(it may have been billed; close or abandon it first)",
        )
    if rebuild_paid_call.run_live_attempts(db, rid):
        # Design §7.4: a live chunk attempt may be (or become) a billed task.
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a run while one of its chunk attempts or character "
                   "checks is in flight",
        )
    project_id = run.project_id
    db.delete(run)  # cascades to RunSegment
    db.commit()
    shutil.rmtree(run_dir(rid, project_id), ignore_errors=True)
    log.info("Deleted run %s (db + disk)", rid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/runs/{rid}/segments", response_model=list[RunSegmentResponse])
def get_run_segments(rid: str, db: Session = Depends(get_db)) -> list:
    """Return RunSegments for a run (progress display)."""
    _get_run_or_404(rid, db)
    segments = (
        db.execute(
            select(RunSegment)
            .where(RunSegment.run_id == rid)
            .order_by(RunSegment.index)
        )
        .scalars()
        .all()
    )
    return [RunSegmentResponse.model_validate(s) for s in segments]


@router.get("/runs/{rid}/reference-image")
def get_run_reference_image(
    rid: str, index: int = Query(0, ge=0), db: Session = Depends(get_db)
) -> FileResponse:
    """Serve one of a run's original reference images by index (thumbnail source)."""
    run = _get_run_or_404(rid, db)
    refs = list(run.reference_image_urls or [])
    if index >= len(refs):
        raise HTTPException(
            status_code=404,
            detail=f"Reference image index {index} out of range (have {len(refs)})",
        )
    return _serve_local_reference(refs[index], what="Reference image")


@router.get("/runs/{rid}/voice-reference")
def get_run_voice_reference(
    rid: str, index: int = Query(0, ge=0), db: Session = Depends(get_db)
) -> FileResponse:
    """Serve a run's own copy of its voice reference (run-page playback).

    Distinct from ``/voice-clips/{id}/file``, which serves the LIBRARY copy: the
    two can differ once the library row is renamed, re-uploaded or deleted, and
    what a run was actually generated against is this one.
    """
    run = _get_run_or_404(rid, db)
    voices = list(run.reference_audio_urls or [])
    if index >= len(voices):
        raise HTTPException(
            status_code=404,
            detail=f"Voice reference index {index} out of range (have {len(voices)})",
        )
    return _serve_local_reference(
        voices[index], what="Voice reference", media_type="audio/mpeg"
    )


@router.get("/runs/{rid}/preprocessed-image")
def get_run_preprocessed_image(
    rid: str, index: int = Query(0, ge=0), db: Session = Depends(get_db)
) -> FileResponse:
    """Serve one of a run's GPT Image 2 preprocessed reference images by index."""
    run = _get_run_or_404(rid, db)
    pp = list(run.preprocessed_reference_urls or [])
    if index >= len(pp):
        raise HTTPException(
            status_code=404,
            detail=f"Preprocessed image index {index} out of range (have {len(pp)})",
        )
    return _serve_local_reference(pp[index], what="Preprocessed image")


@router.get("/runs/{rid}/segments/{rsid}/override-reference")
def get_run_segment_override_reference(
    rid: str,
    rsid: str,
    index: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serve one of a segment's hand-override reference images by index.

    Mirrors ``get_run_preprocessed_image``. ``reference_image_urls_override``
    can hold local WORKER-DISK paths (an uploaded file), which must never be
    shown as text in the override form's URL box — this is what lets the
    template render such an entry as a thumbnail instead.
    """
    _get_run_or_404(rid, db)
    rs = _get_run_segment_or_404(rsid, rid, db)
    refs = list(rs.reference_image_urls_override or [])
    if index >= len(refs):
        raise HTTPException(
            status_code=404,
            detail=f"Override reference index {index} out of range (have {len(refs)})",
        )
    return _serve_local_reference(refs[index], what="Override reference")


# ---------------------------------------------------------------------------
# R2-new: GPT Image 2 preprocessing approval / regeneration
# ---------------------------------------------------------------------------


@router.post("/runs/{rid}/approve-preprocess", response_model=RunResponse, dependencies=[GATED])
def approve_preprocess(rid: str, db: Session = Depends(get_db)) -> RunResponse:
    """Approve the GPT Image 2 preprocessed reference images and start the run.

    R2-new: transitions ``preprocess_review → queued`` and enqueues
    ``run_submit_run`` (the preprocess chain released the per-run lock, so
    submit acquires a fresh one). Seedance/Gemini Omni receive the
    preprocessed images.
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "preprocessing")
    if run.status != RunStatus.preprocess_review:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run status is {run.status.value!r}, expected 'preprocess_review'"
            ),
        )

    # An empty slot in preprocessed_reference_urls means one image is missing —
    # a single-image regen whose task failed, or one that never came back. The
    # submit phase feeds this list straight to the model, so approving a hole
    # would crash the submit at best and spend video credits on a half-built
    # reference set at worst. An EMPTY list is fine: that is "preprocessing
    # produced nothing", where submit falls back to the originals.
    pp_slots = list(run.preprocessed_reference_urls or [])
    missing = [i for i, path_ in enumerate(pp_slots) if not path_]
    if missing:
        which = ", ".join(f"#{i}" for i in missing)
        one = len(missing) == 1
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference{'' if one else 's'} {which} "
                f"{'has' if one else 'have'} no image — regenerate "
                f"{'it' if one else 'them'} (or delete "
                f"{'it' if one else 'them'}) before approving."
            ),
        )

    try:
        transition(run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.commit()

    enqueue_process_run(rid)
    log.info("Approved preprocess for run %s — enqueued submit", rid)
    return RunResponse.model_validate(run)


@router.delete("/runs/{rid}/preprocess-refs/{index}", response_model=RunResponse)
def delete_preprocess_ref(
    rid: str, index: int, db: Session = Depends(get_db)
) -> RunResponse:
    """Drop ONE reference (and its GPT Image 2 result) from a run under review.

    The reference lists are positional and parallel (see the ``Run`` model):
    ``reference_image_urls[0]`` is the character photo, ``[1:]`` are the trims,
    and ``preprocess_image_prompts`` / ``preprocessed_reference_urls`` /
    ``preprocess_task_ids`` line up with them index for index. Removing a trim
    therefore means popping the SAME index out of all four lists at once —
    anything else silently misaligns a prompt with the wrong image.

    Why this exists: a stray image in the trims field (e.g. the character photo
    pasted there a second time) produces a junk generated ref that would then be
    handed to Seedance for every segment. The trims field no longer accepts
    arbitrary images (see ``create_run``), but a run that already sits at
    ``preprocess_review`` can only be rescued by deleting the bad entry —
    regenerating would rebill the good ones.

    Rejected when:
      * the run is not at ``preprocess_review`` (409) — no worker may be holding it;
      * ``index`` is 0 (400) — that is the character photo, the one ref every
        GPT Image 2 task needs as context. Use Regenerate to replace it;
      * ``preprocess_mode`` is ``all_as_refs`` (409) — there is a single
        generated image covering every ref, so per-index deletion is meaningless.
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "preprocessing")
    if run.status != RunStatus.preprocess_review:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run status is {run.status.value!r}, expected 'preprocess_review'"
            ),
        )
    if (run.preprocess_mode or "per_image") == "all_as_refs":
        raise HTTPException(
            status_code=409,
            detail=(
                "This run combined every trim into ONE image (all_as_refs mode) "
                "— there is no per-image result to delete. Use Regenerate."
            ),
        )

    refs = list(run.reference_image_urls or [])
    if index == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Reference 0 is the character photo — it is context for every "
                "GPT Image task and cannot be deleted. Use Regenerate to "
                "replace it."
            ),
        )
    if index < 0 or index >= len(refs):
        raise HTTPException(
            status_code=400,
            detail=f"index {index} out of range (run has {len(refs)} references)",
        )

    prompts = list(run.preprocess_image_prompts or [])
    pp = list(run.preprocessed_reference_urls or [])
    task_ids = list(run.preprocess_task_ids or [])

    doomed_pp = pp[index] if index < len(pp) else None

    refs.pop(index)
    if index < len(prompts):
        prompts.pop(index)
    if index < len(pp):
        pp.pop(index)
    if index < len(task_ids):
        task_ids.pop(index)

    run.reference_image_urls = refs
    run.preprocess_image_prompts = prompts
    run.preprocessed_reference_urls = pp
    run.preprocess_task_ids = task_ids
    db.commit()

    # Unlink the generated file, but ONLY a file this phase created. A
    # pass-through ref stores the ORIGINAL path in preprocessed_reference_urls,
    # and a duplicated image (the very case this endpoint exists for) makes two
    # entries share one path — deleting either would take out the character
    # photo. So: only inside the run's own preprocessed/ dir, and only when no
    # surviving entry still points at it.
    if doomed_pp:
        pp_dir = os.path.join(
            project_dir(run.project_id), "runs", rid, "preprocessed"
        )
        still_used = doomed_pp in pp or doomed_pp in refs
        if (
            not still_used
            and path_within(doomed_pp, pp_dir) is not None
            and os.path.isfile(doomed_pp)
        ):
            try:
                os.remove(doomed_pp)
            except OSError:
                log.warning(
                    "Could not unlink preprocessed image %s for run %s",
                    doomed_pp, rid,
                )

    log.info(
        "Deleted preprocess ref %d from run %s — %d refs left", index, rid, len(refs)
    )
    return RunResponse.model_validate(run)


@router.post(
    "/runs/{rid}/regenerate-preprocess-ref/{index}", response_model=RunResponse,
    dependencies=[GATED],
)
def regenerate_preprocess_ref(
    rid: str,
    index: int,
    db: Session = Depends(get_db),
    prompt: Optional[str] = Form(None),
) -> RunResponse:
    """Regenerate ONE GPT Image 2 result and leave the other images untouched.

    The full ``regenerate-preprocess`` wipes ``preprocessed/`` and re-runs every
    trim, so one bad image out of three costs three fresh generations. This
    endpoint clears exactly one index — its result file and its task id — and
    re-enters ``preprocessing``. ``preprocess_reference_images`` skips every ref
    that still holds a result (the ``done[i]`` guard there), so only this index
    is submitted to kie.ai again; the others keep the images the operator
    already liked.

    *prompt* (optional) replaces ``preprocess_image_prompts[index]``; omit it to
    re-roll the current prompt (GPT Image 2 is stochastic — the same prompt does
    yield a different image). A trim that was passed through has no prompt of
    its own, so regenerating it requires supplying one.

    Rejected when:
      * the run is not at ``preprocess_review`` (409) — no worker may be
        holding it;
      * ``preprocess_mode`` is ``all_as_refs`` (409) — a single image covers
        every ref, so "just this one" *is* the full Regenerate;
      * ``index`` is 0 (400) — the character photo is context for every task
        and is never generated;
      * ``index`` is out of range (400);
      * the effective prompt is blank (400) — that would be a pass-through,
        which generates nothing and would leave the old image in place.
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "preprocessing")
    if run.status != RunStatus.preprocess_review:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run status is {run.status.value!r}, expected 'preprocess_review'"
            ),
        )
    if not run.preprocess_enabled:
        # Belt: the phase would no-op and hand the run back with the slot we
        # just cleared still empty, which then blocks Approve. Unreachable in
        # practice (only a preprocessing run reaches this status at all).
        raise HTTPException(
            status_code=409,
            detail="This run does not use GPT Image preprocessing",
        )
    if (run.preprocess_mode or "per_image") == "all_as_refs":
        raise HTTPException(
            status_code=409,
            detail=(
                "This run combined every trim into ONE image (all_as_refs mode) "
                "— there is no single image to regenerate on its own. Use "
                "Regenerate."
            ),
        )

    refs = list(run.reference_image_urls or [])
    if index == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Reference 0 is the character photo — it is context for every "
                "GPT Image task and is never generated. Use Regenerate to "
                "replace it."
            ),
        )
    if index < 0 or index >= len(refs):
        raise HTTPException(
            status_code=400,
            detail=f"index {index} out of range (run has {len(refs)} references)",
        )

    # Parallel lists, padded to len(refs) so an index is always addressable.
    prompts = list(run.preprocess_image_prompts or [])
    prompts += [""] * max(0, len(refs) - len(prompts))
    pp = list(run.preprocessed_reference_urls or [])
    pp += [None] * max(0, len(refs) - len(pp))
    task_ids = list(run.preprocess_task_ids or [])
    task_ids += [None] * max(0, len(refs) - len(task_ids))

    new_prompt = (prompt or "").strip() or (prompts[index] or "").strip()
    if not new_prompt:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference {index} has no GPT Image prompt — it was passed "
                "through as-is. Give it a prompt to generate an image from it."
            ),
        )

    doomed_pp = pp[index]

    prompts[0] = ""  # R1-new invariant: the character photo is never prompted.
    prompts[index] = new_prompt
    pp[index] = None
    task_ids[index] = None

    run.preprocess_image_prompts = prompts
    run.preprocessed_reference_urls = pp
    run.preprocess_task_ids = task_ids
    run.error_message = None
    # An auto-approve run (a copy that reuses an already-approved recipe) does
    # not normally stop here at all. If one somehow does and the operator hand-
    # picks an image to redo, they are standing AT the review gate — submitting
    # to the video model the moment the new image lands is the last thing they
    # asked for. Taking manual control voids the "skip the review" flag.
    if run.preprocess_auto_approve:
        run.preprocess_auto_approve = False
        log.info(
            "Run %s: single-image regen at review — cleared preprocess_auto_"
            "approve so the run pauses for approval again", rid,
        )

    try:
        transition(run, RunStatus.preprocessing)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # One image; the run's video segments were reserved at launch.
    _spend_reserve(
        spend_mod.reserve, db, spend_mod.run_scope(rid), spend_mod.image_credits(),
        kind="regenerate one GPT Image", run_id=rid,
    )
    db.commit()

    # Unlink the old generated image — same rule as delete_preprocess_ref: only
    # a file this phase created (under the run's own preprocessed/ dir) and only
    # when no surviving entry still points at it. A pass-through entry holds the
    # ORIGINAL path, which must survive: it is still the input for the new task.
    if doomed_pp:
        pp_dir = os.path.join(
            project_dir(run.project_id), "runs", rid, "preprocessed"
        )
        still_used = doomed_pp in pp or doomed_pp in refs
        if (
            not still_used
            and path_within(doomed_pp, pp_dir) is not None
            and os.path.isfile(doomed_pp)
        ):
            try:
                os.remove(doomed_pp)
            except OSError:
                log.warning(
                    "Could not unlink preprocessed image %s for run %s",
                    doomed_pp, rid,
                )

    enqueue_preprocess_run(rid)
    log.info(
        "Regenerate preprocess ref %d for run %s — enqueued preprocess", index, rid
    )
    return RunResponse.model_validate(run)


@router.post("/runs/{rid}/regenerate-preprocess", response_model=RunResponse, dependencies=[GATED])
def regenerate_preprocess(
    rid: str,
    db: Session = Depends(get_db),
    preprocess_image_prompts: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    preprocess_reference_file: Optional[UploadFile] = File(None),
    preprocess_reference_url: Optional[str] = Form(None),
) -> RunResponse:
    """Regenerate the GPT Image 2 preprocessed reference images.

    R2-new: accepts optional new ``preprocess_image_prompts`` (JSON list of
    trim prompts), optional new ``reference_files`` / ``reference_urls`` (the
    new TRIMS — index 0 / character ref is preserved unless a new one is
    provided), optional ``preprocess_reference_file`` / ``preprocess_reference_url``
    to REPLACE the character ref at index 0, and re-runs the preprocessing
    phase. Old preprocessed images are deleted; ``preprocessed_reference_urls``
    and ``preprocess_task_ids`` are reset. Transitions
    ``preprocess_review → preprocessing`` and enqueues ``run_preprocess_refs``.

    When the caller provides new trims/refs, the trim portion of
    ``reference_image_urls`` (indices 1:) is replaced with the new ones. If
    only prompts change, the refs are kept as-is. The R1-new prompt model is
    re-applied (index 0 = "").
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "preprocessing")
    if run.status != RunStatus.preprocess_review:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run status is {run.status.value!r}, expected 'preprocess_review'"
            ),
        )

    existing_refs = list(run.reference_image_urls or [])
    if not existing_refs:
        raise HTTPException(
            status_code=400,
            detail="Run has no reference images to regenerate from",
        )
    user_ref = existing_refs[0]
    # Refuse an already-exhausted day before any upload is written. (The image
    # count is only known after the uploads/prompts are parsed; the exact
    # reservation below then decides.)
    _budget_precheck(db, "regenerate GPT Image")

    # --- New trims (optional) ---
    # Same rule as create_run: trims are frames from the source video, nothing
    # else. Regenerate would otherwise be a back door into exactly the mistake
    # the New Run form now prevents. Validated up front, before anything is
    # written to disk, so a rejected request leaves no orphan uploads behind.
    ref_files = [rf for rf in (reference_files or []) if getattr(rf, "filename", None)]
    new_trim_urls = [
        u.strip() for u in (reference_urls or "").split(",") if u.strip()
    ]
    has_new_trims = bool(ref_files) or bool(new_trim_urls)
    if has_new_trims:
        _assert_trims_are_source_frames(run.project_id, ref_files, new_trim_urls)
    # AICP-148: the character photo URL is a URL or a staged frame, nothing else.
    if preprocess_reference_url and preprocess_reference_url.strip() and not (
        preprocess_reference_file is not None and preprocess_reference_file.filename
    ):
        _reject_bad_reference_urls(
            [preprocess_reference_url.strip()],
            field="preprocess_reference_url",
            local_ok=lambda v: _staged_frame_realpath(run.project_id, v) is not None,
        )

    # The ref COUNT (all the image price depends on) is known before anything
    # is saved: a new character ref only replaces index 0.
    total_refs = (
        1 + len(ref_files) + len(new_trim_urls) if has_new_trims else len(existing_refs)
    )

    # --- New prompts (optional) ---
    new_prompts: Optional[list] = None
    if preprocess_image_prompts:
        try:
            trim_prompts = json.loads(preprocess_image_prompts)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail="preprocess_image_prompts must be valid JSON",
            ) from exc
        if not isinstance(trim_prompts, list):
            raise HTTPException(
                status_code=400,
                detail="preprocess_image_prompts must be a JSON list",
            )
    else:
        # Keep existing prompts but re-apply the R1-new model (index 0 = "").
        existing_prompts = list(run.preprocess_image_prompts or [])
        if run.preprocess_mode == "all_as_refs":
            trim_prompts = [existing_prompts[0]] if existing_prompts else []
        else:
            trim_prompts = existing_prompts[1:] if len(existing_prompts) > 1 else []

    mode = run.preprocess_mode or "per_image"
    n_trims = max(0, total_refs - 1)

    if mode == "all_as_refs":
        single = trim_prompts[0] if trim_prompts else ""
        new_prompts = [single]
    else:
        if n_trims == 0:
            new_prompts = [""]
        else:
            if len(trim_prompts) == 1 and n_trims > 1:
                trim_prompts = [trim_prompts[0]] * n_trims
            elif len(trim_prompts) < n_trims:
                trim_prompts = trim_prompts + [""] * (n_trims - len(trim_prompts))
            elif len(trim_prompts) > n_trims:
                trim_prompts = trim_prompts[:n_trims]
            new_prompts = [""] + list(trim_prompts)

    # Exact price of this regeneration, refused BEFORE the uploads below are
    # written — a same-named upload would overwrite the stored reference file
    # that a rolled-back request leaves the DB still pointing at.
    _budget_precheck(
        db, "regenerate GPT Image",
        needed=spend_mod.preprocess_image_count(
            SimpleNamespace(
                preprocess_enabled=getattr(run, "preprocess_enabled", False),
                reference_image_urls=[None] * total_refs,
                preprocess_image_prompts=new_prompts,
                preprocess_mode=mode,
            ),
            fresh=True,
        ) * spend_mod.image_credits(),
    )

    # --- New character ref (optional — replaces index 0) ---
    max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
    refs_dir = os.path.join(project_dir(run.project_id), "runs", rid, "references")
    os.makedirs(refs_dir, exist_ok=True)

    def _copy_staged(url: str) -> str:
        return _copy_staged_frame(run.project_id, url, refs_dir)

    new_user_ref: Optional[str] = None
    if preprocess_reference_file is not None and preprocess_reference_file.filename:
        safe = _safe_filename(preprocess_reference_file.filename or "char_ref.jpg")
        dest = os.path.join(refs_dir, safe)
        _save_upload(preprocess_reference_file, dest, max_bytes=max_bytes)
        new_user_ref = dest
    elif preprocess_reference_url and preprocess_reference_url.strip():
        new_user_ref = _copy_staged(preprocess_reference_url.strip())

    saved_trim_paths: list[str] = []
    for rf in ref_files:
        safe = _safe_filename(rf.filename or f"trim_{len(saved_trim_paths)}.jpg")
        dest = os.path.join(refs_dir, safe)
        _save_upload(rf, dest, max_bytes=max_bytes)
        saved_trim_paths.append(dest)

    durable_trim_urls: list[str] = [_copy_staged(u) for u in new_trim_urls]

    char_ref = new_user_ref if new_user_ref is not None else user_ref
    if has_new_trims:
        new_trims = saved_trim_paths + durable_trim_urls
        new_refs = [char_ref] + new_trims
    elif new_user_ref is not None:
        # New char ref, keep existing trims (1:).
        new_refs = [char_ref] + existing_refs[1:]
    else:
        new_refs = existing_refs

    # --- Reset state (old preprocessed images are deleted after the commit,
    # so a refused or failed commit never leaves the DB pointing at files that
    # are already gone) ---
    pp_dir = os.path.join(project_dir(run.project_id), "runs", rid, "preprocessed")

    run.reference_image_urls = new_refs
    run.preprocess_image_prompts = new_prompts
    run.preprocessed_reference_urls = None
    run.preprocess_task_ids = None
    run.error_message = None
    # Same reasoning as the single-image regen: an operator regenerating from
    # the review card has taken manual control, and this card promises the run
    # will not start until they approve. An auto-approve run only reaches review
    # when a failure parked it here, and leaving the flag set would submit the
    # new images without the look they just asked for.
    if run.preprocess_auto_approve:
        run.preprocess_auto_approve = False
        log.info(
            "Run %s: regenerate at review — cleared preprocess_auto_approve so "
            "the run pauses for approval again", rid,
        )

    try:
        transition(run, RunStatus.preprocessing)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Images only: the video segments were reserved when the run launched.
    _spend_reserve(
        spend_mod.reserve, db, spend_mod.run_scope(rid),
        spend_mod.preprocess_image_count(run) * spend_mod.image_credits(),
        kind="regenerate GPT Image", run_id=rid,
    )
    db.commit()
    if os.path.isdir(pp_dir):
        shutil.rmtree(pp_dir, ignore_errors=True)

    enqueue_preprocess_run(rid)
    log.info("Regenerate preprocess for run %s — enqueued preprocess", rid)
    return RunResponse.model_validate(run)


@router.get("/runs/{rid}/result/info")
def get_run_result_info(rid: str, db: Session = Depends(get_db)) -> dict:
    """Return result metadata without downloading the file."""
    run = _get_run_or_404(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Run not done yet: status is {run.status!r}",
        )
    gdrive_link = (
        f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
        if run.result_gdrive_file_id
        else None
    )
    return {
        "run_id": rid,
        "result_local_path": run.result_local_path,
        "result_gdrive_file_id": run.result_gdrive_file_id,
        "result_gdrive_link": gdrive_link,
    }


@router.get("/runs/{rid}/result")
def download_run_result(rid: str):
    """Download the final video file for a run.

    Fix #5: when NGINX_ACCEL_ENABLED, returns X-Accel-Redirect so nginx serves
    the bytes directly instead of streaming through Python.
    Fix #6: DB session is scoped to a plain `with` block so it's committed and
    closed before the (potentially long) file transfer, freeing the pool
    connection immediately.
    The Run pointer is mutable, so cached media must revalidate. New page
    links also carry a result identity to bypass older immutable URL entries.
    """
    with get_session() as db:
        run = _get_run_or_404(rid, db)
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        if status_val != "done":
            raise HTTPException(
                status_code=409,
                detail=f"Run not done yet: status is {run.status!r}",
            )
        path = run.result_local_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result file not found on disk")

        filename = os.path.basename(path)
        gdrive_id = run.result_gdrive_file_id or ""

    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename=filename,
        extra_headers={
            "X-GDrive-File-Id": gdrive_id,
            "Cache-Control": "private, no-cache",
        },
    )


# AICP-331: which step of a run "Move to new project" may copy. "run" is the
# run's final video (the original behaviour, stored as source_kind=NULL); the
# others are the run's derived creatives, by id.
_PROMOTE_SOURCE_KINDS = ("run", "caption", "tail_edit", "overlay")
#: How a step is named in the default project name / source_ref / banner.
PROMOTE_STEP_LABELS = {"caption": "captions", "tail_edit": "tail edit", "overlay": "overlay"}
_PROMOTE_STEP_TITLES = {"caption": "Caption", "tail_edit": "Tail edit", "overlay": "Overlay"}


def _lock_promote_step(db: Session, kind: str, source_id: str, rid: str):
    """Row-lock and return the derived creative *source_id* of run *rid*.

    Called with the Run row already locked (run -> artifact, the same order as
    caption/tail-edit/overlay retry and the assistant's Apply, so no
    deadlock). The artifact's own row lock is what makes the done-check and
    the copy atomic against a retry (done -> queued, then a worker rewrites
    the output), a worker's orphan-resume, the reaper and a Delete: each of
    them UPDATEs/DELETEs this row first and therefore waits for our commit,
    or committed before we locked and we see it. A no-op on SQLite.

    404 when the step does not exist, belongs to another run, or (overlay) is
    soft-deleted or has no parent run (an upload / manual-script overlay).
    """
    model = {"caption": RunCaption, "tail_edit": RunTailEdit, "overlay": OverlayJob}[kind]
    row = db.scalar(select(model).where(model.id == source_id)
                    .with_for_update().execution_options(populate_existing=True))
    title = _PROMOTE_STEP_TITLES[kind]
    if row is None:
        raise HTTPException(status_code=404, detail=f"{title} {source_id!r} not found")
    if kind == "overlay":
        owner = overlay_runs.parent_run(row, db) if row.deleted_at is None else None
        owner_id = owner.id if owner is not None else None
    else:
        owner_id = row.run_id
    if (kind == "overlay" and row.deleted_at is not None) or owner_id != rid:
        raise HTTPException(
            status_code=404, detail=f"{title} {source_id!r} not found on this run"
        )
    return row


def _promote_source_file(kind: str, row) -> str:
    """The done step's output path, or 409 (not done / nothing on disk)."""
    title = _PROMOTE_STEP_TITLES[kind]
    status_val = row.status.value if hasattr(row.status, "value") else str(row.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"{title} is not done yet: status is {status_val!r}",
        )
    src = row.result_local_path if kind == "overlay" else row.output_path
    if not src or not os.path.exists(src):
        raise HTTPException(
            status_code=409, detail=f"{title} has no output video on disk to move"
        )
    return src


def _discard_promote_copy(db: Session, project_id: str, dest: str) -> None:
    """Remove the copied source of a promote that did not commit.

    Only when the project row is really absent: a failure raised by the commit
    call itself can come after the server committed, and deleting the file
    then would leave a committed project without its source."""
    try:
        db.rollback()
        if db.get(VideoProject, project_id) is not None:
            return
    except Exception:
        log.exception("promote %s: cannot confirm the project is absent; copy kept", project_id)
        return
    try:
        if os.path.exists(dest):
            os.unlink(dest)
        os.rmdir(os.path.dirname(dest))  # only if nothing else was written there
    except OSError:
        pass


@router.post(
    "/runs/{rid}/to-project",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectCreateResponse,
    dependencies=[GATED],
)
def promote_run_to_project(
    rid: str,
    project_type: str = Form(DEFAULT_PROJECT_TYPE),
    name: Optional[str] = Form(None),
    localisation_scope: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    autopilot_params: Optional[str] = Form(None),
    source_kind: Optional[str] = Form(None),
    source_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> ProjectCreateResponse:
    """Start a NEW project from a done run's final video — or (AICP-331) from
    the output of one of its later steps.

    The automation behind "change the book cover after the face swap" (or
    "strip the subtitles first, then swap faces"): instead of downloading the
    result and re-uploading it by hand, this COPIES the run's stitched video
    into a fresh VideoProject of the requested type and enqueues its analysis.

    The copy is deliberate — the new project owns its source file, so deleting
    either side later never breaks the other. ``source_run_id`` keeps a soft
    backlink for the UI; nothing resolves it at pipeline time.

    *localisation_scope* (``hook`` default | ``whole``, AICP-340) is the new
    localisation project's scope, exactly as on ``POST /projects``: stored on
    the row before its first analysis (enqueued below), which applies it.
    ``whole`` on any other type is a 400.

    *source_kind* (AICP-331) = ``run`` (default: the run's final video) |
    ``caption`` | ``tail_edit`` | ``overlay``, with *source_id* naming that
    derived creative of THIS run. It must be done with its output on disk
    (else 409, as for the run); missing, another run's, a soft-deleted or a
    parent-less overlay is a 404. ``source_run_id`` stays the owning run for
    every kind (backlink, Source cell, insert-merge and caption-placement
    lineage), and ``source_kind`` / ``source_artifact_id`` record the step —
    both NULL for the run's final, exactly as before.
    """
    kind = (source_kind or "").strip() or "run"
    step_id = (source_id or "").strip() or None
    if kind not in _PROMOTE_SOURCE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"source_kind must be one of {', '.join(_PROMOTE_SOURCE_KINDS)}",
        )
    if kind == "run" and step_id is not None:
        raise HTTPException(status_code=400, detail="source_id is only valid with a step source_kind")
    if kind != "run" and step_id is None:
        raise HTTPException(status_code=400, detail=f"source_id is required for source_kind={kind!r}")
    run = _get_run_or_404(rid, db)
    _validate_project_type(project_type)
    localisation_scope = (
        _validate_localisation_scope(localisation_scope, project_type)
        or LOCALISATION_SCOPE_HOOK
    )
    # AICP-363 §2.1: the mode is offered again; default = the parent's mode
    # when the target type has an autopilot flow, else manual.
    if mode is None or not mode.strip():
        from .autopilot import registry as ap_registry

        parent_mode = getattr(db.get(VideoProject, run.project_id), "mode", "manual")
        mode = (
            "assistant"
            if parent_mode == "assistant" and settings.AUTOPILOT_ENABLED
            and ap_registry.flow_for_project_type(project_type) is not None
            else "manual"
        )
    flow, ap_params = _validate_project_mode(
        mode, project_type, autopilot_params,
        VideoProject(project_type=project_type, localisation_scope=localisation_scope),
    )
    # AICP-368: hold the Run row through the done-check, the copy and the
    # caption-placement fill lookup, so a VOID fill applied concurrently
    # (which re-points a segment and requeues the run under this same lock)
    # is either fully before this promote or fully after it — the frozen fill
    # evidence then describes exactly the file copied. A no-op on SQLite.
    db.refresh(run, with_for_update=True)

    step = None
    if kind == "run":
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        if status_val != "done":
            raise HTTPException(
                status_code=409,
                detail=f"Run is not done yet: status is {status_val!r}",
            )
        src = run.result_local_path
        if not src or not os.path.exists(src):
            raise HTTPException(
                status_code=409, detail="Run has no final video on disk to move"
            )
    else:
        # The step's file does not depend on the run's CURRENT result, so the
        # run itself need not be done; the step must be (checked under its
        # own row lock, taken after the run's).
        step = _lock_promote_step(db, kind, step_id, rid)
        src = _promote_source_file(kind, step)

    project_id = str(uuid.uuid4())
    ext = os.path.splitext(src)[-1].lstrip(".") or "mp4"
    dest = project_source_path(project_id, ext)
    # Nothing may outlive a promote that did not commit: the copied file is
    # removed on any failure from the copy up to and including the commit
    # (only if the row is really absent), never after it — from then on the
    # project owns it (an enqueue failure below leaves a recoverable project).
    try:
        shutil.copy2(src, dest)

        parent = db.get(VideoProject, run.project_id)
        label = spec_for(project_type).label
        origin = run.name or (parent.name if parent else None) or f"run {rid[:8]}"
        if step is None:
            default_name = f"{origin} → {label}"
            source_ref = f"{origin} ({rid[:8]}) — final video"
        else:
            step_label = PROMOTE_STEP_LABELS[kind]
            default_name = f"{origin} · {step_label} → {label}"
            source_ref = f"{origin} ({rid[:8]}) — {step_label} {step.id[:8]}"
        resolved_name = (name or "").strip()[:255] or default_name[:255]

        # As in extract_insert: the parent's cap is re-validated on the way through
        # so a value already unusable in the DB is not laundered into a fresh project
        # (and a fresh failed analysis). NULL fallback, not a 400 — the caller did
        # not choose this value. See _inherited_max_segment_sec.
        inherited_max_segment_sec = _inherited_max_segment_sec(
            parent.max_segment_sec if parent else None,
            context="promote_run_to_project",
            parent_id=run.project_id,
        )

        project = VideoProject(
            id=project_id,
            name=resolved_name,
            source_type="upload",
            source_ref=source_ref,
            source_local_path=dest,
            source_run_id=rid,
            source_kind=None if step is None else kind,
            source_artifact_id=None if step is None else step.id,
            project_type=project_type,
            caption_removal_method=project_types.default_caption_removal_method(project_type),
            # Continue the lineage's "remove original audio" choice rather than
            # silently re-enabling audio input on the follow-up project — unless
            # the new type defaults it on (localisation), which then wins
            # (AICP-339, project_types.derived_mute_source).
            mute_source=derived_mute_source(
                project_type, parent.mute_source if parent else None
            ),
            # Likewise the segmentation cap: the follow-up project is analyzed
            # immediately below, and an operator who chose long segments for the
            # face swap wants the same cut for the cover change. NULL (no parent, no
            # explicit choice, or a stored value we refuse to propagate) keeps the
            # universal default.
            max_segment_sec=inherited_max_segment_sec,
            localisation_scope=localisation_scope,  # AICP-340
            status=ProjectStatus.created,
            mode=mode.strip(),
        )
        # AICP-368: carry the original caption area down the lineage so this
        # project's burned captions cover the subtitle-removal patches — the
        # parent's placement if it has one, else (subtitle-removal parent) a
        # provisional one from the run's fills, refined by a detector scan below.
        try:
            scan_placement = caption_placement_mod.inherit_on_promote(db, project, run, parent)
        except Exception:  # never fail a promote that works without AICP-368
            log.exception("promote_run_to_project %s: caption placement not set", rid)
            project.caption_placement, scan_placement = None, False
        analysis_token = _admit_analysis(project)  # AICP-350: counted at commit
        db.add(project)
        if flow is not None:
            db.flush()
            from .autopilot.core import create_autopilot

            create_autopilot(db, project, flow, ap_params)
        db.commit()
    except BaseException:
        _discard_promote_copy(db, project_id, dest)
        raise

    _enqueue_analysis(project_id, analysis_token)
    if scan_placement:
        caption_placement_mod.enqueue_scan(
            project_id, caption_placement_mod.rev_of(project.caption_placement)
        )

    log.info(
        "Moved run %s%s into new project %s (project_type=%s localisation_scope=%s "
        "caption_placement=%s)",
        rid, "" if step is None else f" {kind} {step_id}", project_id, project_type,
        localisation_scope, caption_placement_mod.kind_of(project.caption_placement),
    )
    status_out = (
        project.status.value if hasattr(project.status, "value") else str(project.status)
    )
    return ProjectCreateResponse(project_id=project_id, status=status_out)


# Run states in which per-segment prompt/reference overrides may be edited and
# re-run. All three are terminal-for-the-operator: no worker is touching the
# run, and the next step is always a re-run of one segment.  `incomplete` (some
# swap segments failed, nothing stitched) is precisely the state an operator
# lands in when a segment needs a reworded prompt, so editing must be allowed
# there — the run page renders the override panel for it.
_SEGMENT_EDITABLE_RUN_STATUSES = ("done", "failed", "incomplete")


def _get_run_segment_or_404(rsid: str, run_id: str, db: Session) -> RunSegment:
    rs = db.get(RunSegment, rsid)
    if rs is None or rs.run_id != run_id:
        raise HTTPException(
            status_code=404, detail=f"RunSegment {rsid!r} not found in run {run_id!r}"
        )
    return rs


_ACTIVE_SEGMENT_IMAGE_STATUSES = (
    SegmentImageStatus.queued.value, SegmentImageStatus.generating.value,
)


def _active_segment_image_job(
    rsid: str, db: Session
) -> Optional["RunSegmentImageJob"]:
    """The active (queued/generating) RunSegmentImageJob for *rsid*, if any.

    Serialises per-segment GPT Image 2 regeneration IN THE API rather than by
    taking the run's Redis processing lock — see
    pipeline_segment_image.py's module docstring for why that lock would be
    worse than the race it prevents. Two segments of the same run may
    regenerate in parallel; only a second attempt on the SAME segment is
    refused.
    """
    return db.execute(
        select(RunSegmentImageJob)
        .where(
            RunSegmentImageJob.run_segment_id == rsid,
            RunSegmentImageJob.status.in_(_ACTIVE_SEGMENT_IMAGE_STATUSES),
        )
        .limit(1)
    ).scalar_one_or_none()


def _run_has_active_segment_image_job(
    rid: str, db: Session
) -> Optional["RunSegmentImageJob"]:
    """The first active (queued/generating) image job ANYWHERE on this run.

    Used wherever an action re-enqueues ``run_submit_run`` — ``retry_run``
    and ``rerun_segment`` — because ``_build_submit_state`` does not resubmit
    only the segment the operator targeted: it resets and resubmits EVERY
    active swap segment that is not already ``completed`` with a result on
    disk (``pipeline_v2.py``). A per-segment check on just the requested
    ``rsid`` misses a DIFFERENT failed sibling whose own image job is still
    running — that sibling gets reset and resubmitted from its stale
    reference state too, and its job's later auto-apply becomes a UI
    illusion over a video that never used it. ``RunSegmentImageJob.run_id``
    is denormalized (set once, from the segment's parent Run, at create
    time) so this needs no join.
    """
    return db.execute(
        select(RunSegmentImageJob)
        .where(
            RunSegmentImageJob.run_id == rid,
            RunSegmentImageJob.status.in_(_ACTIVE_SEGMENT_IMAGE_STATUSES),
        )
        .limit(1)
    ).scalar_one_or_none()


def _default_segment_image_prompt(run: Run) -> str:
    """The starting-point GPT Image 2 prompt for a per-segment regeneration.

    There is no per-segment preprocess prompt to read — Run.preprocess_image_
    prompts is parallel to the run-level REFERENCE list, not to RunSegment
    rows, and no segment↔trim mapping exists to derive one from. This falls
    back to whichever run-level prompt the run's preprocess_mode produced:
      * ``per_image``   — the first non-empty trim prompt (index >= 1); a
        starting point when several trims differ, never a concatenation.
      * ``all_as_refs`` — the run's single prompt (the one-element list's
        only entry).
    Returns "" when the run has none; the caller 400s on a blank effective
    prompt, same as ``regenerate_preprocess_ref``.
    """
    prompts = list(run.preprocess_image_prompts or [])
    if (run.preprocess_mode or "per_image") == "all_as_refs":
        return (prompts[0] if prompts else "") or ""
    for p in prompts[1:]:
        if p and p.strip():
            return p.strip()
    return ""


def _segment_launch_window(
    run: Run, rs: RunSegment
) -> tuple[Optional[float], Optional[float]]:
    """(start_sec, end_sec) *rs* was generated against.

    A separate small implementation of ``web_v2._launch_window`` — api_v2
    cannot import web_v2 (web_v2 already imports FROM api_v2, and the reverse
    would cycle) — so keep the two in sync by hand if this logic ever changes.
    Reads the run's FROZEN ``segment_snapshot`` first, so a later project
    segment edit can never change what an old run's regeneration window
    means; falls back to the live SegmentDef for runs that predate the
    snapshot column. Returns ``(None, None)`` when neither is available (a
    pre-snapshot run whose SegmentDef has since been deleted).
    """
    entry = None
    for e in run.segment_snapshot or []:
        if isinstance(e, dict) and e.get("id") == rs.segment_def_id:
            entry = e
            break
    if entry is None:
        entry = rs.segment_def
    if entry is None:
        return None, None

    def bound(field: str) -> Optional[float]:
        value = (
            entry.get(field) if isinstance(entry, dict)
            else getattr(entry, field, None)
        )
        return None if value is None else float(value)

    return bound("start_sec"), bound("end_sec")


def _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls):
    """Set a RunSegment's prompt/reference overrides (shared by PATCH and rerun).

    Empty prompt clears the prompt override; empty reference set clears the ref
    override (both fall back to run-level values).
    """
    ref_files = reference_files or []
    parsed_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    total_refs = len(ref_files) + len(parsed_urls)
    if total_refs > settings.MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many reference images: got {total_refs}, "
                f"max {settings.MAX_REFERENCE_IMAGES}"
            ),
        )
    # AICP-148: URLs, plus the local entries this override already holds (the
    # form re-posts its uploaded-file thumbnails as hidden values).
    _reject_bad_reference_urls(
        parsed_urls, field="reference_urls",
        local_ok=_existing_override_ok(rs.reference_image_urls_override),
    )

    rs.prompt_override = (prompt.strip() or None) if prompt is not None else None

    if total_refs == 0:
        rs.reference_image_urls_override = None
    else:
        refs_dir = os.path.join(
            project_dir(run.project_id), "runs", rid, "segment_refs", rsid
        )
        os.makedirs(refs_dir, exist_ok=True)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
        saved_paths: list[str] = []
        for rf in ref_files:
            safe = _safe_filename(rf.filename or f"ref_{len(saved_paths)}.jpg")
            dest = os.path.join(refs_dir, safe)
            _save_upload(rf, dest, max_bytes=max_bytes)
            saved_paths.append(dest)
        rs.reference_image_urls_override = saved_paths + parsed_urls


@router.patch(
    "/runs/{rid}/segments/{rsid}",
    response_model=RunSegmentResponse,
)
def patch_run_segment(
    rid: str,
    rsid: str,
    prompt: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunSegmentResponse:
    """Override prompt and/or reference images for an individual RunSegment.

    Only allowed when the run is in done, failed or incomplete status (see
    ``_SEGMENT_EDITABLE_RUN_STATUSES``). Empty prompt clears the override (falls
    back to run-level prompt). Empty reference list clears the override (falls
    back to run-level references).
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "a segment override (re-roll the chunk with a prompt instead)")
    # AICP-363 §4.3: a generation input of a (terminal) run changes only under
    # its row lock, which fences the autopilot's `run:` version keys.
    db.refresh(run, with_for_update=True)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in _SEGMENT_EDITABLE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit segment: run status is {run.status!r}; "
                "expected 'done', 'failed', or 'incomplete'"
            ),
        )

    rs = _get_run_segment_or_404(rsid, rid, db)

    active_job = _active_segment_image_job(rsid, db)
    if active_job is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot edit segment: a GPT Image regeneration is in "
                f"progress for it (job {active_job.id}, status="
                f"{active_job.status.value!r}) — wait for it to finish or "
                "clear it first."
            ),
        )

    # A promoted range attempt overwrites this RunSegment's override fields
    # too (see pipeline_range_attempt._promote) — editing them here while one
    # is in flight would just be clobbered, or clobber the attempt's own
    # pending override, depending on timing.
    active_attempt = _active_range_attempt(rid, db)
    if active_attempt is not None:
        raise _range_attempt_conflict(active_attempt)

    _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls)

    db.commit()
    db.refresh(rs)
    log.info("Patched RunSegment %s (run %s): prompt_override=%r refs_override=%r",
             rsid, rid, rs.prompt_override, rs.reference_image_urls_override)
    return RunSegmentResponse.model_validate(rs)


def _require_stitch_library_clips(run: Run, db: Session) -> None:
    """409 when *run*'s stitch plan names a library clip that no longer exists.

    Every re-stitch admission calls it AFTER taking the run row lock and
    before any reservation or write: a rerun used to be billed and then fail
    at the stitch ("library clip … no longer exists"). The autopilot deletes
    its TEMP probe clips at the end of a language (AICP-366) and locks every
    run naming the clip first, so a deletion and a re-stitch admission are
    serialised by the run row: whichever locks second sees the other.
    """
    ids = sorted({
        e.get("clip_id") for e in (run.stitch_plan or [])
        if isinstance(e, dict) and e.get("kind") == "library" and e.get("clip_id")
    })
    missing = [cid for cid in ids if db.get(LibraryClip, cid) is None]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot re-stitch run {run.id[:8]}: its stitch plan uses library clip(s) "
                f"{', '.join(m[:8] for m in missing)} that no longer exist — make a new run "
                "instead"
            ),
        )


def _admit_segment_rerun(run: Run, db: Session) -> None:
    """Run-level admission shared by every segment rerun route.

    Takes the Run-row lock FIRST and only then reads status — validating
    first and locking second leaves exactly the window the lock exists to
    close (see _lock_run_for_result_mutation).
    """
    _lock_run_for_result_mutation(run, db)
    _require_segment_reconstruction_allowed(run.id, db)
    _require_stitch_library_clips(run, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in _SEGMENT_EDITABLE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot re-run segment: run status is {run.status!r}; "
                "expected 'done', 'failed', or 'incomplete'"
            ),
        )


def _check_segment_rerun_conflicts(rid: str, db: Session) -> None:
    """The cross-feature 409s every segment rerun route shares."""
    # A range regeneration in flight on this run will PROMOTE onto
    # Run.segment_snapshot / result_local_path / a RunSegment at an
    # unpredictable moment — a rerun's eager reset racing against that
    # promotion could resurrect the OLD range/result right after a successful
    # correction, or promote onto a Run this rerun just moved to `queued`.
    # Refuse until it settles. Same for an audio patch, which also ends by
    # re-pointing result_local_path.
    active_attempt = _active_range_attempt(rid, db)
    if active_attempt is not None:
        raise _range_attempt_conflict(active_attempt)
    active_patch = _active_audio_patch(rid, db)
    if active_patch is not None:
        raise _audio_patch_conflict(active_patch)

    # Run-wide, not just the requested segments: re-enqueuing submit resets
    # and resubmits EVERY active swap segment that isn't already `completed`
    # (see pipeline_v2._build_submit_state) — so a DIFFERENT failed sibling
    # with its own image job in flight would get reset and resubmitted from
    # its stale reference state too, exactly the hazard this gate exists to
    # prevent. Mirrors retry_run's gate (_run_has_active_segment_image_job).
    active_job = _run_has_active_segment_image_job(rid, db)
    if active_job is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot re-run segment: a GPT Image regeneration is in "
                f"progress on segment {active_job.run_segment_id} (job "
                f"{active_job.id}, status={active_job.status.value!r}) — "
                "re-running would resubmit it too, from its OLD image, "
                "before the new one is ready. Wait for it to finish."
            ),
        )


def _segment_rerun_need(db: Session, run: Run, segments: list) -> float:
    """Credits a rerun of *segments* would spend: every unfinished active
    segment (submit_run resubmits those too, listed or not), plus each listed
    segment that is currently paid — once it is reset it is unfinished, and the
    run-level estimate does not count it yet."""
    need = spend_mod.estimate_run_credits(db, run, include_preprocess=False)
    defs_by_id = None
    void_source = project_types.is_void_source_run(run)
    for rs in segments:
        # The estimate already counts an unfinished segment -- except, on a
        # VOID-from-the-source run, one that is not a model fallback yet: the
        # rerun makes it one (design §7.2, R2-2).
        if not spend_mod.segment_is_paid(rs) and not (void_source and not rs.model_fallback):
            continue
        if defs_by_id is None:
            defs_by_id = {sd.id: sd for sd in _run_segment_defs_for(run)}
        sd = defs_by_id.get(rs.segment_def_id)
        if sd is not None:
            need += spend_mod.video_credits(
                run.model, run.resolution, spend_mod._clip_sec(sd)
            )
    return need


def _current_result_is_audio_patch(run: Run, db: Session) -> Optional[RunAudioPatch]:
    """The promoted audio patch whose output IS the run's current result, if any.

    A segment rerun or take restore re-stitches the final from the segment
    clips, which silently drops an audio fix — and since AICP-342 the re-stitch
    also replaces the bytes behind the run's Drive id, i.e. behind the link an
    exported sheet row already holds. Exact path match: successive patches each
    re-point result_local_path at their own output, so the latest one matches.
    """
    if not run.result_local_path:
        return None
    return db.execute(
        select(RunAudioPatch)
        .where(
            RunAudioPatch.run_id == run.id,
            RunAudioPatch.output_path == run.result_local_path,
            RunAudioPatch.status == AudioPatchStatus.done,
        )
        .limit(1)
    ).scalar_one_or_none()


def _require_audio_patch_discard_ack(run: Run, db: Session, discard: bool) -> None:
    patch = _current_result_is_audio_patch(run, db)
    if patch is not None and not discard:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The run's current result is audio patch {patch.id}. Re-"
                "stitching from the segment clips drops that audio fix, and "
                "the new final replaces the content behind the run's Drive "
                "link. Resend with discard_audio_patch=true to go ahead (the "
                "patched video stays downloadable from the patch history), "
                "or redo the audio patch afterwards."
            ),
        )


def _segment_take_bounds(run: Run, rs: RunSegment) -> tuple[Optional[float], Optional[float]]:
    return _segment_launch_window(run, rs)


def _bounds_match(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= 0.02


def _existing_take_for_current_result(rs: RunSegment, db: Session) -> Optional[RunSegmentTake]:
    """A take row that already preserves *rs*'s current result file, if any.

    Either the current result IS a take file (the segment was restored), or a
    take of the same kie.ai task exists — both only when that take's file is
    still on disk. A null task id never matches.
    """
    takes = db.execute(
        select(RunSegmentTake).where(RunSegmentTake.run_segment_id == rs.id)
    ).scalars().all()
    for t in takes:
        if t.result_path == rs.local_result_path and os.path.exists(t.result_path):
            return t
    if rs.seedance_task_id:
        for t in takes:
            if t.seedance_task_id == rs.seedance_task_id and os.path.exists(t.result_path):
                return t
    return None


def _archive_current_take(run: Run, rs: RunSegment, db: Session) -> Optional[RunSegmentTake]:
    """Preserve *rs*'s current completed result as a take before it is replaced.

    COPIES the file (never moves it): the copy is published with an atomic
    ``os.replace`` from a unique staging name, and the original stays where the
    DB says it is until the caller's commit lands — a crash in between leaves
    at worst an unreferenced copy, never a row pointing at a moved file.
    Returns None when there is nothing to preserve (not completed, or no
    result on disk).
    """
    if rs.status != SegmentStatus.completed:
        return None
    path = rs.local_result_path
    if not path or not os.path.exists(path):
        return None
    existing = _existing_take_for_current_result(rs, db)
    if existing is not None:
        return existing
    take_id = str(uuid.uuid4())
    out_dir = segment_take_dir(rs.id, run.id, run.project_id)
    dst = os.path.join(out_dir, f"{take_id}.mp4")
    staging = os.path.join(out_dir, f".staging-{uuid.uuid4().hex}.mp4")
    try:
        shutil.copyfile(path, staging)
        os.replace(staging, dst)
    finally:
        if os.path.exists(staging):
            try:
                os.remove(staging)
            except OSError:
                pass
    start, end = _segment_take_bounds(run, rs)
    # Local reference files of the override are copied next to the take: the
    # segment's own segment_refs/<rsid>/ files can be overwritten by name by a
    # later PATCH, and a restore writes this override back (AICP-342).
    ref_override = rs.reference_image_urls_override
    if ref_override:
        ref_override = _copy_reference_files(
            list(ref_override), os.path.join(out_dir, f"{take_id}_refs"),
            unique_names=True,
        )
    take = RunSegmentTake(
        id=take_id,
        run_id=run.id,
        run_segment_id=rs.id,
        segment_def_id=rs.segment_def_id,
        index=rs.index,
        start_sec=start,
        end_sec=end,
        result_path=dst,
        seedance_task_id=rs.seedance_task_id,
        seedance_result_url=rs.seedance_result_url,
        cost_credits=rs.cost_credits,
        prompt_override=rs.prompt_override,
        reference_image_urls_override=copy.deepcopy(ref_override),
        generated_reference_urls=copy.deepcopy(rs.generated_reference_urls),
        generated_image_job_id=rs.generated_image_job_id,
    )
    db.add(take)
    log.info(
        "Preserved take %s of segment %s (run %s, task %s) -> %s",
        take_id, rs.id, run.id, rs.seedance_task_id, dst,
    )
    return take


def _invalidate_final_and_requeue(run: Run, db: Session, *, kind: str) -> None:
    """Invalidate the stitched final, move the run to ``queued``, reserve its
    spend, commit and enqueue ONE process_run."""
    # Invalidate the stitched final: it contains the OLD segment results.
    # Without this, finalize_run sees "all segments completed + final exists"
    # (the rerun segments are already re-completed by the time the finalize
    # job runs) and re-delivers the stale final.mp4 instead of re-stitching.
    run.result_local_path = None
    # The new takes may have a different length, and the stitch only probes
    # the duration when it is unset — a stale value would outlive the rerun.
    run.result_duration_sec = None

    # Transition run → queued (done→queued or failed→queued both allowed)
    try:
        transition(run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # submit_run resubmits EVERY unfinished active swap segment, not just the
    # listed ones — the estimate reads the whole run's segment state. The run
    # was done/failed/incomplete (idle), so its unconsumed reservation is dead.
    _spend_reserve(
        spend_mod.reserve_run, db, run, kind=kind, release_unused=True,
    )
    db.commit()
    enqueue_process_run(run.id)


def _rerun_images_only(run: Run, segments: list, db: Session, prompts: Optional[dict],
                       *, kind: str) -> None:
    """Re-roll chunks of an images-only run (design §6.6, §7.3): reset them
    to ``pending`` (a NEW attempt from the frozen snapshot), then the same
    invalidate + queue + reserve as every rerun, in one transaction. The
    reservation is the images-only estimator under the scope lock."""
    from . import pipeline_rebuild

    _require_rebuild_enabled_for_repair(run)
    try:
        pipeline_rebuild.reset_for_rerun(db, run, segments, prompts)
    except pipeline_rebuild.RepairRefused as exc:
        raise _rebuild_repair_refused(exc) from exc
    _invalidate_final_and_requeue(run, db, kind=kind)


def _reset_segments_and_requeue(
    run: Run, segments: list, db: Session, *, kind: str
) -> None:
    """Reset *segments* to pending, invalidate the stitched final, move the run
    to ``queued``, reserve its spend, commit and enqueue ONE process_run.

    Everything the caller validated is still under the Run-row lock taken by
    _admit_segment_rerun, which this commit releases. The caller has already
    preserved each segment's current result as a take (_archive_current_take)
    BEFORE applying new overrides, so the take records the overrides that
    produced it.
    """
    for rs in segments:
        rs.status = SegmentStatus.pending
        rs.error_message = None
        # Drop any source-fallback verdict from the previous attempt. It
        # described the model/segment-length pairing of that attempt; this
        # re-run may follow a segment split or a different model, and a stale
        # flag would make the stitch keep substituting original footage even
        # after a successful swap.
        rs.source_fallback = False
        rs.seedance_task_id = None
        rs.seedance_result_url = None
        rs.local_result_path = None

    _invalidate_final_and_requeue(run, db, kind=kind)


@router.post("/runs/{rid}/segments/{rsid}/rerun", response_model=RunResponse, dependencies=[GATED])
def rerun_segment(
    rid: str,
    rsid: str,
    prompt: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    discard_audio_patch: bool = Form(False),
    db: Session = Depends(get_db),
) -> RunResponse:
    """Apply the (optional) prompt/reference override, reset one RunSegment to
    pending, and re-queue the run — atomically, so the re-run always uses the
    prompt sent with THIS request (no separate save needed).

    The run must be in done, failed or incomplete status. Completed
    RunSegments are skipped by process_run (resumability) — but every OTHER
    active swap segment (any still failed/pending, not just this one) is
    reset and resubmitted too, since submit_run reprocesses the whole run's
    active set, not a single segment in isolation. If no prompt field is sent
    at all, the existing override is left untouched.

    To rerun several segments with one re-stitch, use
    ``POST /runs/{rid}/segments/rerun`` (AICP-341).
    """
    run = _get_run_or_404(rid, db)
    _admit_segment_rerun(run, db)
    rs = _get_run_segment_or_404(rsid, rid, db)
    _check_segment_rerun_conflicts(rid, db)
    _require_audio_patch_discard_ack(run, db, discard_audio_patch)
    if spend_mod.is_images_only_run(run):
        # Re-roll one chunk (design §6.5-6.6): a NEW attempt from the frozen
        # snapshot, optionally with an edited prompt; no take archive.
        if any(getattr(f, "filename", None) for f in (reference_files or [])) \
                or (reference_urls or "").strip():
            raise HTTPException(status_code=400, detail=(
                "reference images are not supported for images-only (rebuild) runs"))
        _rerun_images_only(run, [rs], db, None if prompt is None else {rs.id: prompt},
                           kind="rerun segment")
        log.info("Re-rolling chunk %s of images-only run %s", rsid, rid)
        return RunResponse.model_validate(run)

    # Refuse an over-budget rerun before its override files are written (they
    # may replace the segment's current ones). Priced like the reservation
    # below: every unfinished active segment, plus this one once it is reset.
    _budget_precheck(db, "rerun segment", needed=_segment_rerun_need(db, run, [rs]))

    # Apply the prompt/reference sent with this request so the re-run uses
    # exactly what's on screen. Only when a prompt field is present (a form was
    # sent) — otherwise leave any previously-saved override untouched.
    # AICP-342: preserve the current output (with the overrides that made it)
    # before this request's overrides replace them.
    _archive_current_take(run, rs, db)
    if prompt is not None or reference_files or reference_urls:
        _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls)
    if project_types.is_void_source_run(run):
        rs.model_fallback = True        # the operator sends THIS segment to the model

    _reset_segments_and_requeue(run, [rs], db, kind="rerun segment")

    log.info("Rerunning segment %s in run %s", rsid, rid)
    return RunResponse.model_validate(run)


@router.post("/runs/{rid}/segments/rerun", response_model=RunResponse, dependencies=[GATED])
def rerun_segments(
    rid: str,
    body: SegmentsRerunRequest,
    db: Session = Depends(get_db),
) -> RunResponse:
    """Rerun SEVERAL segments of a run in one call, with one re-stitch
    (AICP-341).

    ``{"segments": [{"run_segment_id": "...", "prompt": "..."}, ...]}`` — each
    entry's ``prompt`` is optional: omitted/null keeps the segment's saved
    override, ``""`` clears it, a string sets it. Reference overrides are not
    touched (PATCH the segment first). Same admission, gates and budget rule as
    the single-segment ``.../{rsid}/rerun``; the whole list is validated before
    anything is written, so a bad entry changes nothing.

    As with the single rerun, ``submit_run`` also resubmits every OTHER active
    swap segment that is not completed with a result on disk — on a failed or
    incomplete run those are billed too, listed or not.
    """
    items = body.segments or []
    if not items:
        raise HTTPException(status_code=400, detail="segments must not be empty")
    ids = [it.run_segment_id for it in items]
    if len(set(ids)) != len(ids):
        raise HTTPException(
            status_code=400, detail="each segment may be listed only once",
        )

    run = _get_run_or_404(rid, db)
    _admit_segment_rerun(run, db)
    segments = [_get_run_segment_or_404(sid, rid, db) for sid in ids]

    # Only swap segments the run actually processes: one excluded from a
    # custom stitch plan is never submitted, so "rerunning" it would reset it
    # and then silently leave it pending.
    from .pipeline_v2 import _active_swap_segment_ids

    defs = _run_segment_defs_for(run)
    active_ids = _active_swap_segment_ids(run.stitch_plan, defs)
    inactive = [rs.id for rs in segments if rs.segment_def_id not in active_ids]
    if inactive:
        raise HTTPException(
            status_code=400,
            detail=(
                "not an active swap segment of this run (excluded by the "
                f"stitch plan, or not a swap segment): {inactive}"
            ),
        )

    _check_segment_rerun_conflicts(rid, db)
    _require_audio_patch_discard_ack(run, db, body.discard_audio_patch)
    if spend_mod.is_images_only_run(run):
        _rerun_images_only(run, segments, db,
                           {rs.id: it.prompt for rs, it in zip(segments, items)},
                           kind="rerun segments")
        log.info("Re-rolling %d chunk(s) of images-only run %s", len(segments), rid)
        return RunResponse.model_validate(run)
    _budget_precheck(
        db, "rerun segments", needed=_segment_rerun_need(db, run, segments),
    )

    for rs in segments:
        _archive_current_take(run, rs, db)
    for rs, it in zip(segments, items):
        if it.prompt is not None:
            rs.prompt_override = it.prompt.strip() or None
        if project_types.is_void_source_run(run):
            rs.model_fallback = True    # the operator sends these to the model (§7.2)

    _reset_segments_and_requeue(run, segments, db, kind="rerun segments")

    log.info(
        "Rerunning %d segment(s) %s in run %s", len(segments),
        [rs.index for rs in segments], rid,
    )
    return RunResponse.model_validate(run)


# ---------------------------------------------------------------------------
# Segment takes (AICP-342) — preserved previous outputs, restorable
# ---------------------------------------------------------------------------


def _take_response(take: RunSegmentTake, rs: RunSegment) -> RunSegmentTakeResponse:
    return RunSegmentTakeResponse(
        id=take.id,
        run_id=take.run_id,
        run_segment_id=take.run_segment_id,
        index=take.index,
        created_at=take.created_at,
        start_sec=take.start_sec,
        end_sec=take.end_sec,
        seedance_task_id=take.seedance_task_id,
        cost_credits=take.cost_credits,
        prompt_override=take.prompt_override,
        is_current=(rs.local_result_path == take.result_path),
        file_exists=os.path.exists(take.result_path),
        file_purged=take.file_purged_at is not None,
    )


@router.get(
    "/runs/{rid}/segments/{rsid}/takes",
    response_model=list[RunSegmentTakeResponse],
)
def list_segment_takes(rid: str, rsid: str, db: Session = Depends(get_db)):
    """Preserved previous outputs of one segment, newest first (AICP-342).

    The segment's CURRENT output is a take only once it was restored from one
    (``is_current``); a fresh generation becomes a take when the next rerun
    or restore replaces it.
    """
    _get_run_or_404(rid, db)
    rs = _get_run_segment_or_404(rsid, rid, db)
    takes = db.execute(
        select(RunSegmentTake)
        .where(RunSegmentTake.run_segment_id == rs.id)
        .order_by(desc(RunSegmentTake.created_at))
    ).scalars().all()
    return [_take_response(t, rs) for t in takes]


@router.get("/runs/{rid}/segments/{rsid}/takes/{tid}/video")
def download_segment_take(
    rid: str,
    rsid: str,
    tid: str,
    download: bool = Query(False, description="Force a download instead of inline playback"),
):
    """Serve one preserved take's clip (immutable: a take file is never rewritten)."""
    with get_session() as db:
        run = _get_run_or_404(rid, db)
        _get_run_segment_or_404(rsid, rid, db)
        take = db.get(RunSegmentTake, tid)
        if take is None or take.run_segment_id != rsid or take.run_id != rid:
            raise HTTPException(status_code=404, detail=f"Take {tid!r} not found")
        path = take.result_path
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="The take's file is gone from disk")
        filename = (
            f"{run.short_id}_seg{take.index:02d}_take{tid[:8]}.mp4" if download else None
        )
    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename=filename,
        extra_headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.post("/runs/{rid}/segments/restore", response_model=RunResponse, dependencies=[GATED])
def restore_segment_takes(
    rid: str,
    body: SegmentTakesRestoreRequest,
    db: Session = Depends(get_db),
) -> RunResponse:
    """Put previous takes back as segments' results and re-stitch (AICP-342).

    ``{"takes": [take_id, ...], "discard_audio_patch": false}`` — at most one
    take per segment. Never bills: refused (409) unless, once the takes are
    applied, EVERY active swap segment of the run has a completed result on
    disk, so ``submit_run`` has nothing to submit and the run goes straight to
    the re-stitch and the in-place re-delivery. Each segment's current result
    is preserved as a take first, so a restore is itself undoable. The take's
    recorded prompt / reference / generated-image overrides are written back
    onto the segment too (a recorded "none" clears the override). A take whose
    file the retention cleanup purged is a 409.
    """
    ids = list(body.takes or [])
    if not ids:
        raise HTTPException(status_code=400, detail="takes must not be empty")
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="each take may be listed only once")

    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "take restore (use an earlier attempt instead)")
    _admit_segment_rerun(run, db)

    takes: list[RunSegmentTake] = []
    for tid in ids:
        take = db.get(RunSegmentTake, tid)
        if take is None or take.run_id != rid:
            raise HTTPException(status_code=404, detail=f"Take {tid!r} not found in run {rid!r}")
        takes.append(take)
    if len({t.run_segment_id for t in takes}) != len(takes):
        raise HTTPException(status_code=400, detail="at most one take per segment")

    pairs: list[tuple[RunSegment, RunSegmentTake]] = []
    for take in takes:
        rs = _get_run_segment_or_404(take.run_segment_id, rid, db)
        if take.file_purged_at is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Take {take.id} of segment {rs.index} was purged by the "
                    "take retention cleanup — its file no longer exists"
                ),
            )
        if not os.path.exists(take.result_path):
            raise HTTPException(
                status_code=409,
                detail=f"Take {take.id} of segment {rs.index}: its file is gone from disk",
            )
        start, end = _segment_take_bounds(run, rs)
        if not (_bounds_match(start, take.start_sec) and _bounds_match(end, take.end_sec)):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Take {take.id} of segment {rs.index} was generated for "
                    f"{take.start_sec}-{take.end_sec}s, but the segment is now "
                    f"{start}-{end}s (a range change was promoted since) — it "
                    "no longer fits the run's timeline"
                ),
            )
        pairs.append((rs, take))

    _check_segment_rerun_conflicts(rid, db)
    _require_audio_patch_discard_ack(run, db, body.discard_audio_patch)

    # Never bill: every OTHER active swap segment must already be finished
    # with a result on disk, otherwise submit_run would (re)generate it.
    from .pipeline_v2 import _active_swap_segment_ids

    restoring = {rs.id for rs, _ in pairs}
    defs = _run_segment_defs_for(run)
    active_ids = _active_swap_segment_ids(run.stitch_plan, defs)
    rs_by_def = {rs.segment_def_id: rs for rs in run.run_segments}
    for rs, _take in pairs:
        if rs.segment_def_id not in active_ids:
            raise HTTPException(
                status_code=400,
                detail=f"segment {rs.index} is not an active swap segment of this run",
            )
    unfinished = []
    for sd in defs:
        if sd.id not in active_ids:
            continue
        rs = rs_by_def.get(sd.id)
        if rs is not None and rs.id in restoring:
            continue
        if (
            rs is None
            or rs.status != SegmentStatus.completed
            or not rs.local_result_path
            or not os.path.exists(rs.local_result_path)
        ):
            unfinished.append(sd.index)
    if unfinished:
        raise HTTPException(
            status_code=409,
            detail=(
                f"segment(s) {unfinished} have no finished result, so the "
                "re-stitch would generate them (paid). Restore a take for them "
                "in the same call, or rerun them."
            ),
        )
    if all(rs.local_result_path == take.result_path for rs, take in pairs):
        raise HTTPException(status_code=409, detail="those takes are already current")

    for rs, _take in pairs:
        _archive_current_take(run, rs, db)
    for rs, take in pairs:
        # The take's recipe comes back with it (AICP-342, Operator 2026-10-30):
        # Copy/Automation then reproduce what produced the restored output —
        # prompt and reference overrides; generated_reference_urls stays
        # run-local, as _build_copied_run has always deliberately dropped it.
        # None means "no override" at archive time and clears the field.
        rs.prompt_override = take.prompt_override
        # Copied back into the segment's OWN refs dir rather than pointing into
        # the take's: the retention sweep deletes a take's files, and the
        # segment must never depend on them once it is rerun again.
        rs.reference_image_urls_override = (
            _copy_reference_files(
                list(take.reference_image_urls_override),
                os.path.join(
                    project_dir(run.project_id), "runs", rid, "segment_refs",
                    rs.id, f"take-{take.id}",
                ),
                unique_names=True,
            )
            if take.reference_image_urls_override else None
        )
        rs.generated_reference_urls = copy.deepcopy(take.generated_reference_urls)
        rs.generated_image_job_id = (
            take.generated_image_job_id if take.generated_reference_urls else None
        )
        rs.local_result_path = take.result_path
        rs.seedance_task_id = take.seedance_task_id
        rs.seedance_result_url = take.seedance_result_url
        rs.cost_credits = take.cost_credits
        rs.error_message = None
        rs.source_fallback = False
        rs.status = SegmentStatus.completed

    _invalidate_final_and_requeue(run, db, kind="restore takes")
    log.info(
        "Restored take(s) %s on run %s (segments %s)",
        ids, rid, [rs.index for rs, _ in pairs],
    )
    return RunResponse.model_validate(run)


# ---------------------------------------------------------------------------
# Run-local segment range regeneration
# ---------------------------------------------------------------------------

# Deliberately narrower than _SEGMENT_EDITABLE_RUN_STATUSES: promotion only
# ever has to reconcile with a Run that is already `done` (see
# pipeline_range_attempt._promote's docstring for why a `failed`/`incomplete`
# source run is out of scope for this feature rather than silently guessed
# at). Legacy prompt/reference-only rerun is unaffected — it still accepts
# 'done', 'failed' and 'incomplete' via _SEGMENT_EDITABLE_RUN_STATUSES.
_RANGE_REGEN_ELIGIBLE_RUN_STATUSES = ("done",)


def _active_range_attempt(
    rid: str, db: Session
) -> Optional["RunSegmentRangeAttempt"]:
    """The active (non-terminal) RunSegmentRangeAttempt for this RUN, if any.

    Gated at the RUN level, not per-segment: promotion writes
    Run.segment_snapshot and (usually) Run.result_local_path, so two attempts
    on the same run — even targeting different segments — must never be
    allowed to race to promote. This same check also blocks manual/batch
    Copy and the legacy per-segment rerun/patch endpoints while an attempt is
    in flight (see docs/run-local-segment-range-regeneration.md).
    """
    return db.execute(
        select(RunSegmentRangeAttempt)
        .where(
            RunSegmentRangeAttempt.run_id == rid,
            RunSegmentRangeAttempt.status.in_(
                [s.value for s in RANGE_ATTEMPT_ACTIVE_STATUSES]
            ),
        )
        .limit(1)
    ).scalar_one_or_none()


def _range_attempt_conflict(active: "RunSegmentRangeAttempt") -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            "A run-local segment range regeneration is in progress for this "
            f"run (attempt {active.id}, status={active.status.value!r}) — "
            "wait for it to finish before starting another action that "
            "would change this run's timeline or result."
        ),
    )


@router.post(
    "/runs/{rid}/segments/{rsid}/range-rerun",
    status_code=status.HTTP_201_CREATED,
    response_model=RangeAttemptCreateResponse,
    dependencies=[GATED],
)
def create_range_rerun(
    rid: str,
    rsid: str,
    start_sec: float = Form(...),
    end_sec: float = Form(...),
    prompt: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    discard_audio_patch: bool = Form(False),
    db: Session = Depends(get_db),
) -> RangeAttemptCreateResponse:
    """Regenerate ONE swap segment over a NEW run-local time range.

    Moves the boundary this segment shares with an immediately adjacent
    `keep` segment (never with another `swap`), cuts the candidate — and the
    moved keep neighbour(s) — fresh from the project's ORIGINAL source, and
    creates a ``RunSegmentRangeAttempt`` (an append-only attempt row, not a
    slot — see the model docstring). The Run's active timeline and result
    stay exactly as they are until this attempt succeeds; a failure leaves
    them untouched and this endpoint may be called again.

    Validation is atomic with the (optional) prompt/reference override —
    *start_sec*/*end_sec* are always required by this endpoint (they are
    what distinguishes it from the legacy ``.../rerun``, whose omitted timing
    fields preserve the old prompt/reference-only behaviour exactly).
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "a range regeneration")
    # Lock first, then read: same lock and same reason as create_run_audio_patch's
    # (this is the other side of the cross-table race the per-table indexes
    # cannot see), and taken BEFORE the status and snapshot checks so those are
    # read under it too. Validating first and locking after would leave the
    # window this lock exists to close — a concurrent rerun finishing between
    # the reads and the lock, with the accepted attempt later promoting over
    # its result.
    _lock_run_for_result_mutation(run, db)
    _require_segment_reconstruction_allowed(run.id, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in _RANGE_REGEN_ELIGIBLE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot regenerate a range: run status is {run.status!r}; "
                "expected 'done'. A failed or incomplete run's OTHER broken "
                "segments would leave the run's status ambiguous after only "
                "one segment is fixed — resolve those first (or use the "
                "prompt/reference-only rerun, which is still available in "
                "any of 'done', 'failed' or 'incomplete')."
            ),
        )

    rs = _get_run_segment_or_404(rsid, rid, db)

    if not run.segment_snapshot:
        raise HTTPException(
            status_code=409,
            detail=(
                "This run predates run-local segment snapshots, so its "
                "boundaries cannot be safely resolved against the current "
                "Project segmentation — range editing is not available for "
                "it. The prompt/reference-only rerun is still available."
            ),
        )

    active = _active_range_attempt(rid, db)
    if active is not None:
        raise _range_attempt_conflict(active)

    # The reverse of the gate in create_run_audio_patch: both features finish
    # by re-pointing Run.result_local_path, so whichever starts second would
    # promote onto a base the other has already moved (and be refused there,
    # after paying for a generation). Refuse it here instead, for free.
    active_patch = _active_audio_patch(rid, db)
    if active_patch is not None:
        raise _audio_patch_conflict(active_patch)
    # AICP-349: promotion re-stitches from the segment clips AND replaces the
    # content behind the run's Drive id, so a promoted audio fix would vanish
    # from the exported link — same acknowledgement as a rerun/restore.
    _require_audio_patch_discard_ack(run, db, discard_audio_patch)

    active_job = _run_has_active_segment_image_job(rid, db)
    if active_job is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot regenerate a range: a GPT Image regeneration is in "
                f"progress on this run (job {active_job.id}, status="
                f"{active_job.status.value!r}) — wait for it to finish."
            ),
        )

    project = _get_project_or_404(run.project_id, db)

    if not math.isfinite(start_sec) or not math.isfinite(end_sec):
        raise HTTPException(status_code=400, detail="start_sec/end_sec must be finite numbers")

    # Reference override, if any — same file-saving convention as
    # _apply_segment_override, but into this attempt's OWN directory (never
    # the live RunSegment's segment_refs dir) so a rejected/failed attempt
    # never mutates state a future PATCH/rerun would read.
    ref_files = reference_files or []
    parsed_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    total_refs = len(ref_files) + len(parsed_urls)
    if total_refs > settings.MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many reference images: got {total_refs}, "
                f"max {settings.MAX_REFERENCE_IMAGES}"
            ),
        )
    # AICP-148: same rule as _apply_segment_override — the form re-posts the
    # segment's current local override entries alongside any typed URLs.
    _reject_bad_reference_urls(
        parsed_urls, field="reference_urls",
        local_ok=_existing_override_ok(rs.reference_image_urls_override),
    )

    attempt_id = str(uuid.uuid4())
    reference_override: Optional[list] = None
    if total_refs:
        refs_dir = os.path.join(
            range_attempt_dir(attempt_id, rsid, rid, run.project_id), "refs"
        )
        os.makedirs(refs_dir, exist_ok=True)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
        saved_paths: list[str] = []
        for rf in ref_files:
            safe = _safe_filename(rf.filename or f"ref_{len(saved_paths)}.jpg")
            dest = os.path.join(refs_dir, safe)
            _save_upload(rf, dest, max_bytes=max_bytes)
            saved_paths.append(dest)
        reference_override = saved_paths + parsed_urls

    try:
        attempt = create_range_attempt(
            run=run, rs=rs, project=project,
            start_sec=start_sec, end_sec=end_sec,
            prompt_override=(prompt.strip() or None) if prompt is not None else None,
            reference_image_urls_override=reference_override,
            attempt_id=attempt_id,
            db=db,
        )
    except RangeValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IntegrityError as exc:
        # The application-level check above (`active is not None`) is a
        # non-locking SELECT — it cannot by itself stop two concurrent
        # range-rerun requests for the same run from both passing it and
        # both reaching create_range_attempt's flush. The DB-level partial
        # unique index (uq_run_segment_range_attempts_active_per_run,
        # migration 73df0ff3f344) is what actually enforces "at most one
        # active attempt per run" under that race; this is the loser's path.
        # Roll back before anything the reference files above wrote gets
        # associated with a row that will never exist, and remove exactly
        # THIS attempt's own directory — attempt_id is a fresh uuid4 minted
        # above, so this can never touch another attempt's files.
        db.rollback()
        if total_refs:
            shutil.rmtree(
                range_attempt_dir(attempt_id, rsid, rid, run.project_id),
                ignore_errors=True,
            )
        active = _active_range_attempt(rid, db)
        if active is not None:
            raise _range_attempt_conflict(active) from exc
        # Lost the race against something that is already terminal by the
        # time we look again (astonishingly fast, but not impossible) —
        # report the conflict generically rather than a misleading 500.
        raise HTTPException(
            status_code=409,
            detail=(
                "A concurrent request already started a range regeneration "
                "for this run — try again."
            ),
        ) from exc

    try:
        spend_mod.reserve(
            db,
            spend_mod.range_scope(attempt.id),
            spend_mod.video_credits(
                run.model, run.resolution,
                attempt.requested_end_sec - attempt.requested_start_sec,
            ),
            kind="range rerun",
            run_id=rid,
        )
    except spend_mod.BudgetExceeded as exc:
        db.rollback()
        if total_refs:
            shutil.rmtree(
                range_attempt_dir(attempt_id, rsid, rid, run.project_id),
                ignore_errors=True,
            )
        raise _budget_exceeded(exc) from exc
    db.commit()
    enqueue_process_range_attempt(attempt.id)

    log.info(
        "Created range attempt %s for segment %s (run %s): [%.2f, %.2f)",
        attempt.id, rsid, rid, start_sec, end_sec,
    )
    return RangeAttemptCreateResponse(
        attempt_id=attempt.id,
        status=attempt.status.value,
        requested_start_sec=attempt.requested_start_sec,
        requested_end_sec=attempt.requested_end_sec,
    )


@router.get(
    "/runs/{rid}/segments/{rsid}/range-attempts",
    response_model=list[RangeAttemptResponse],
)
def list_range_attempts(
    rid: str, rsid: str, db: Session = Depends(get_db),
) -> list[RangeAttemptResponse]:
    """This segment's range-regeneration attempt history, newest first."""
    _get_run_or_404(rid, db)
    _get_run_segment_or_404(rsid, rid, db)
    rows = db.execute(
        select(RunSegmentRangeAttempt)
        .where(RunSegmentRangeAttempt.run_segment_id == rsid)
        .order_by(desc(RunSegmentRangeAttempt.created_at))
    ).scalars().all()
    return [RangeAttemptResponse.model_validate(r) for r in rows]


# ---------------------------------------------------------------------------
# Audio patch endpoints ("Fix audio from another run" — re-voice one window of
# a done run's delivered final from a sibling run's audio, IN PLACE; see
# app/pipeline_audio_patch.py and docs/run-audio-patch.md)
# ---------------------------------------------------------------------------


def _duration_before_lock(run: Run) -> Optional[float]:
    """*run*'s result duration, resolved WITHOUT writing anything.

    Exists so ``create_run_audio_patch`` can hand both durations to validation
    and take its Run row lock with nothing left to shell out for:
    ``media._run`` passes no timeout to ``subprocess.run``, so a hung ffprobe
    inside the locked window would hold an open transaction and that run's row
    lock indefinitely (prod runs with ``idle_in_transaction_session_timeout=0``).

    It deliberately does **not** persist what it probes, unlike the
    self-healing backfill ``web_v2._get_tail_edit_context`` does on render.
    That backfill commits, and committing here — before the lock, from a row
    this request has not locked — can overwrite a duration a range attempt
    promoted while the probe was running, leaving the cache describing a file
    the run no longer points at.

    Returns None when there is nothing to probe or the probe fails; the caller
    turns that into a 400 rather than carrying it into the lock.
    """
    cached = run.result_duration_sec
    if cached and cached > 0:
        return float(cached)
    path = run.result_local_path
    if not path or not os.path.exists(path):
        return None
    try:
        return media_mod.probe(path).duration_sec
    except Exception as exc:  # ffprobe failure — degrade, never 500 over it
        log.warning("could not probe result duration for run %s: %s", run.id, exc)
        return None


def _lock_run_for_result_mutation(run: Run, db: Session, *,
                                  own_fill_id: Optional[str] = None,
                                  own_batch_id: Optional[str] = None) -> None:
    """Take a row lock on *run* before the cross-table "is anything else
    already changing this run's result?" checks.

    Audio patches and range attempts each have a DB-level partial unique index,
    but an index is per TABLE: it cannot stop one of each becoming active on the
    same run at the same time, and the two application checks that are supposed
    to are plain non-locking SELECTs that two concurrent requests can both pass.
    That gap is not cosmetic — an audio patch promotes by re-pointing
    ``result_local_path``, while ``pipeline_range_attempt._promote`` validates
    only ``segment_snapshot``, so a range attempt created in the same window
    would later overwrite the patched result AND re-point the run at a NEW
    Drive file, leaving any already-exported sheet row on the old id. (Found by
    the CLAUDE.md §1 gate on 2026-09-07.)

    Locking the Run row first serialises the two create paths: the loser blocks
    until the winner commits, then its own SELECT sees the winner's row and it
    409s. Held only for the rest of an operator click's transaction. A no-op on
    SQLite, which has no row-level locking; the guarantee is against
    PostgreSQL.
    """
    db.refresh(run, with_for_update=True)
    # AICP-358: an active segment caption fill (admitted / compositing) will
    # re-point a segment result and requeue the run; nothing else may change
    # the run's result meanwhile. Checked after the Run lock, like the rest.
    # 2026-11-03: a batch member (admitted / compositing / composited) counts
    # the same; the batch's own promote passes own_batch_id.
    from .pipeline_segment_fill import active_fill
    fill = active_fill(db, run.id, exclude_batch_id=own_batch_id)
    if fill is not None and fill.id != own_fill_id:
        what = f"fill batch {fill.batch_id} (member {fill.id}" if fill.batch_id else \
            f"Segment caption fill {fill.id} ("
        raise HTTPException(
            status_code=409,
            detail=(f"{what}) is {fill.status} on this run; "
                    "wait for it to finish (or close its provider task) first"),
        )
    from .assistant_run_result import ensure_no_unresolved_intent
    from .assistant_store import AssistantConflict
    try:
        ensure_no_unresolved_intent(db, run.id)
    except AssistantConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


def _lock_runs_for_result_admission(db: Session, run_ids) -> dict[str, Run]:
    """Serialize legacy admissions with publication; never take a chat lock."""
    runs = {}
    for run_id in sorted(set(run_ids) - {None}):
        run = _get_run_or_404(run_id, db)
        _lock_run_for_result_mutation(run, db)
        runs[run_id] = run
    return runs


def _require_segment_reconstruction_allowed(run_id: str, db: Session) -> None:
    from .assistant_run_result import ensure_segment_reconstruction_allowed
    from .assistant_store import AssistantConflict
    try:
        ensure_segment_reconstruction_allowed(db, run_id)
    except AssistantConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


def _admit_retained_run_source(db: Session, run_ids) -> None:
    from .assistant_run_result import ensure_retained_run_source_retry_allowed
    from .assistant_store import AssistantConflict
    for run_id in _lock_runs_for_result_admission(db, run_ids):
        try:
            ensure_retained_run_source_retry_allowed(db, run_id)
        except AssistantConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None


def _merge_source_run_ids(db: Session, child_run_id: str,
                          target_project_id: str) -> list[str]:
    target = _get_project_or_404(target_project_id, db)
    # The parent backlink is soft: the target owns a copy and remains usable
    # after its original Run was deleted.
    parent_id = target.source_run_id
    return [child_run_id] + ([parent_id] if parent_id and db.get(Run, parent_id) else [])


def _backfill_result_duration(run: Run, db: Session, path: str, duration: float) -> float | None:
    """A probe describes one path; never overwrite a newer result's cache."""
    db.execute(update(Run).where(
        Run.id == run.id, Run.result_local_path == path,
        Run.result_duration_sec.is_(None),
    ).values(result_duration_sec=duration).execution_options(synchronize_session=False))
    db.commit()
    db.refresh(run)
    return run.result_duration_sec


def _active_audio_patch(rid: str, db: Session) -> Optional["RunAudioPatch"]:
    """The active (non-terminal) RunAudioPatch for this run, if any.

    Backs the 409 below and the mutual exclusion with range regeneration: both
    features end by re-pointing ``Run.result_local_path`` (and an audio patch
    also replaces the bytes behind the run's Drive id), so two of them in
    flight on one run must never race to promote. A non-locking SELECT — the
    DB-level partial unique index is what actually enforces it under a race;
    this is what turns the loser into a readable 409 instead of a 500.
    """
    return db.execute(
        select(RunAudioPatch)
        .where(
            RunAudioPatch.run_id == rid,
            RunAudioPatch.status.in_(
                [s.value for s in AUDIO_PATCH_ACTIVE_STATUSES]
            ),
        )
        .limit(1)
    ).scalar_one_or_none()


def _audio_patch_conflict(active: "RunAudioPatch") -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            "An audio patch is in progress for this run (patch "
            f"{active.id}, status={active.status.value!r}) — wait for it to "
            "finish before starting another action that would change this "
            "run's result."
        ),
    )


def _get_audio_patch_or_404(patch_id: str, db: Session) -> "RunAudioPatch":
    patch = db.get(RunAudioPatch, patch_id)
    if patch is None:
        raise HTTPException(status_code=404, detail="Audio patch not found")
    return patch


@router.post(
    "/runs/{rid}/audio-patches",
    status_code=status.HTTP_201_CREATED,
    response_model=AudioPatchResponse,
    dependencies=[GATED],
)
def create_run_audio_patch(
    rid: str,
    donor_run_id: str = Form(...),
    target_start_sec: float = Form(...),
    target_end_sec: float = Form(...),
    donor_start_sec: float = Form(...),
    crossfade_sec: Optional[float] = Form(None),
    db: Session = Depends(get_db),
) -> AudioPatchResponse:
    """Queue an audio patch: replace this run's audio over
    ``[target_start_sec, target_end_sec]`` with *donor_run_id*'s audio starting
    at *donor_start_sec*.

    The replacement is always exactly as long as the window — there is no donor
    end to send, because any other length would have to move samples after the
    window and desync the rest of the video against its own picture.

    This does NOT create a derived creative: on success the run's own result
    and the content behind its existing Drive id are replaced, so its Job ID,
    its Creative Link and any Final Creatives row already holding it keep
    working and now serve the fixed video. See docs/run-audio-patch.md.

    409 when the run is not ``done``, when it has no result on disk, or when
    another audio patch or a range regeneration is already in flight for it.
    400 with the exact bound when the window does not fit either video.
    """
    run = _get_run_or_404(rid, db)
    donor = db.get(Run, donor_run_id)
    if donor is None:
        raise HTTPException(
            status_code=404, detail=f"Donor run {donor_run_id!r} not found"
        )

    # Resolve both durations BEFORE the lock and hand them to validation, so
    # the locked window shells out to nothing — see _duration_before_lock. A
    # video on disk we cannot read is a 400 here rather than a probe carried
    # into the lock; a run with no file at all falls through to the validator,
    # which has the right message for it.
    probed_target_path = run.result_local_path
    target_duration_sec = _duration_before_lock(run)
    donor_duration_sec = _duration_before_lock(donor)
    for probed, other in ((target_duration_sec, run), (donor_duration_sec, donor)):
        if probed is None and other.result_local_path and os.path.exists(
            other.result_local_path
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not read the video for run {other.id[:8]!r} — its "
                    "duration is not cached and ffprobe could not determine it."
                ),
            )

    # Before the status check, not after: everything read below has to be read
    # under the lock, or a concurrent range-rerun request can slip between them.
    _lock_run_for_result_mutation(run, db)
    run_status = run.status.value if hasattr(run.status, "value") else str(run.status)
    if run_status not in {s.value for s in AUDIO_PATCH_ELIGIBLE_RUN_STATUSES}:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot patch audio: run status is {run_status!r}, expected "
                "'done' — a patch rewrites the delivered final, which is the "
                "one thing a non-done run does not have."
            ),
        )

    # The duration above was resolved before the lock; base_result_path below is
    # captured after it. If a range attempt promoted in between, the two would
    # describe different videos — a near-tail window would be accepted against
    # the old length and die in the worker, or a valid one rejected with a 400
    # naming a length that is no longer true. It also means the operator picked
    # the window while watching a video the run no longer serves, so the window
    # itself is meaningless. Refuse and say so, the same way _promote refuses a
    # moved base.
    #
    # Only the target is checked here: it is the row this request holds a lock
    # on, so this check is airtight. The donor is not locked and cannot be, and
    # a donor re-pointed mid-request only produces a row the worker rejects
    # with a readable error when it re-validates against the real file.
    if run.result_local_path != probed_target_path:
        raise HTTPException(
            status_code=409,
            detail=(
                "This run's result changed while the request was being "
                "validated — reload the run page and pick the window again "
                "against the video it serves now."
            ),
        )

    active_patch = _active_audio_patch(rid, db)
    if active_patch is not None:
        raise _audio_patch_conflict(active_patch)

    active_attempt = _active_range_attempt(rid, db)
    if active_attempt is not None:
        raise _range_attempt_conflict(active_attempt)

    crossfade = (
        media_mod.AUDIO_SPLICE_DEFAULT_CROSSFADE_SEC
        if crossfade_sec is None
        else float(crossfade_sec)
    )

    try:
        patch = create_audio_patch(
            run=run,
            donor=donor,
            target_start_sec=target_start_sec,
            target_end_sec=target_end_sec,
            donor_start_sec=donor_start_sec,
            crossfade_sec=crossfade,
            db=db,
            target_duration_sec=target_duration_sec,
            donor_duration_sec=donor_duration_sec,
        )
    except AudioPatchValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IntegrityError as exc:
        # Same race, and the same loser's path, as the range-rerun endpoint's:
        # the `active_patch is not None` check above is a non-locking SELECT
        # that two concurrent requests can both pass, and
        # uq_run_audio_patches_active_per_run is what actually enforces one at
        # a time.
        db.rollback()
        active_patch = _active_audio_patch(rid, db)
        if active_patch is not None:
            raise _audio_patch_conflict(active_patch) from exc
        raise HTTPException(
            status_code=409,
            detail=(
                "A concurrent request already started an audio patch for this "
                "run — try again."
            ),
        ) from exc

    db.commit()
    db.refresh(patch)
    enqueue_process_audio_patch(patch.id)

    log.info(
        "Created audio patch %s for run %s: window=[%.2f, %.2f) from donor "
        "%s@%.2f", patch.id, rid, target_start_sec, target_end_sec,
        donor.id, donor_start_sec,
    )
    return AudioPatchResponse.model_validate(patch)


@router.get("/runs/{rid}/audio-patches", response_model=list[AudioPatchListItem])
def list_run_audio_patches(
    rid: str, db: Session = Depends(get_db),
) -> list[AudioPatchListItem]:
    """This run's audio-patch history, newest first."""
    _get_run_or_404(rid, db)
    rows = db.execute(
        select(RunAudioPatch)
        .where(RunAudioPatch.run_id == rid)
        .order_by(desc(RunAudioPatch.created_at))
    ).scalars().all()

    # One query for every donor mentioned, not one per row. A donor deleted
    # since is simply absent, and the item falls back to its id — the same way
    # TailEditListItem.library_clip_name degrades for a deleted clip.
    donor_ids = {r.donor_run_id for r in rows if r.donor_run_id}
    donors = {}
    if donor_ids:
        donors = {
            r.id: r
            for r in db.execute(
                select(Run).where(Run.id.in_(donor_ids))
            ).scalars().all()
        }

    items: list[AudioPatchListItem] = []
    for row in rows:
        item = AudioPatchListItem.model_validate(row)
        if not row.donor_run_id:  # AICP-389: a same-run low-pass fix
            item.donor_label = "this run (brand fix, low-pass)"
            items.append(item)
            continue
        donor = donors.get(row.donor_run_id)
        item.donor_label = (
            f"{donor.name} · {donor.short_id}" if donor and donor.name
            else (donor.short_id if donor else row.donor_run_id[:8])
        )
        items.append(item)
    return items


@router.get("/audio-patches/{patch_id}/result")
def download_audio_patch_result(patch_id: str):
    """Download one audio patch's candidate video.

    Useful after the fact: a promoted patch's file IS the run's current result
    (the run's own /result endpoint serves the same bytes), while an EARLIER
    patch's output is the only way back to what the run looked like before the
    next one — nothing here is ever overwritten.
    """
    with get_session() as db:
        patch = _get_audio_patch_or_404(patch_id, db)
        path = patch.output_path
        if not path or not os.path.exists(path):
            raise HTTPException(
                status_code=404, detail="Patched video not found on disk"
            )
        filename = os.path.basename(path)

    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename=filename,
        extra_headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


# ---------------------------------------------------------------------------
# Per-segment GPT Image 2 regeneration
# ---------------------------------------------------------------------------


@router.post(
    "/runs/{rid}/segments/{rsid}/regenerate-image",
    response_model=SegmentImageJobCreateResponse,
    dependencies=[GATED],
)
def regenerate_segment_image(
    rid: str,
    rsid: str,
    prompt: Optional[str] = Form(None),
    character_file: Optional[UploadFile] = File(None),
    character_url: Optional[str] = Form(None),
    frame_sec: Optional[float] = Form(None),
    db: Session = Depends(get_db),
) -> SegmentImageJobCreateResponse:
    """Regenerate the GPT Image 2 reference image for ONE swap segment.

    Creates a new ``RunSegmentImageJob`` row (an append-only attempt, not a
    slot — see the model docstring) and enqueues it; the worker composes
    ``RunSegment.generated_reference_urls`` on success (exactly two entries:
    the character photo + the downloaded output — never spliced into the
    run-level reference set, which this segment's image was not generated
    from). This is step one of two deliberately separate operator actions:
    regenerate the image, look at it, THEN re-run the segment.

    *prompt* — optional; falls back to ``_default_segment_image_prompt`` (the
    run's own preprocess prompt) when blank. 400 if the effective prompt is
    still blank.

    *character_file* / *character_url* — optional; the GPT Image 2 input this
    job pairs with the extracted frame. At most one is used (a file wins over
    a URL when both are given); omitted falls back to
    ``run.reference_image_urls[0]``. Deliberately NOT read from
    ``reference_image_urls_override`` — keeping this decoupled from the hand
    override is what stops one from destroying the other.

    *frame_sec* — optional; defaults to the midpoint of this segment's frozen
    launch window (a boundary usually lands on a cut, a poor still to feed an
    image model). 400 if given and outside that window.
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "a per-segment GPT Image regeneration")
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in _SEGMENT_EDITABLE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot regenerate image: run status is {run.status!r}; "
                "expected 'done', 'failed', or 'incomplete'"
            ),
        )
    if not run.preprocess_enabled:
        raise HTTPException(
            status_code=409,
            detail="This run does not use GPT Image preprocessing",
        )

    rs = _get_run_segment_or_404(rsid, rid, db)

    active_job = _active_segment_image_job(rsid, db)
    if active_job is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "A GPT Image regeneration is already in progress for this "
                f"segment (job {active_job.id}, status="
                f"{active_job.status.value!r})"
            ),
        )

    effective_prompt = (prompt or "").strip() or _default_segment_image_prompt(run)
    if not effective_prompt:
        raise HTTPException(
            status_code=400,
            detail=(
                "No GPT Image prompt to use — this run has no preprocess "
                "prompt to start from. Provide one."
            ),
        )

    window_start, window_end = _segment_launch_window(run, rs)
    if frame_sec is None:
        if window_start is None or window_end is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot default the frame timestamp — this segment's "
                    "window is unknown (its SegmentDef was deleted and the "
                    "run predates segment snapshots). Provide frame_sec."
                ),
            )
        frame_sec = (window_start + window_end) / 2.0
    elif window_start is not None and window_end is not None:
        if not (window_start <= frame_sec <= window_end):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"frame_sec ({frame_sec}) must be within this segment's "
                    f"window [{window_start}, {window_end}]"
                ),
            )

    job_id = str(uuid.uuid4())
    if character_file is not None and (character_file.filename or ""):
        job_dir = segment_image_job_dir(job_id, rsid, rid, run.project_id)
        safe = _safe_filename(character_file.filename or "character.jpg")
        dest = os.path.join(job_dir, safe)
        _save_upload(
            character_file, dest, max_bytes=settings.MAX_UPLOAD_SIZE_MB << 20
        )
        character_photo = dest
    elif character_url and character_url.strip():
        # AICP-148: a typed character photo is a URL, never a server path.
        _reject_bad_reference_urls([character_url.strip()], field="character_url")
        character_photo = character_url.strip()
    else:
        refs = list(run.reference_image_urls or [])
        if not refs or not refs[0]:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This run has no character photo "
                    "(reference_image_urls[0]) to default to"
                ),
            )
        character_photo = refs[0]

    job = RunSegmentImageJob(
        id=job_id,
        run_id=rid,
        run_segment_id=rsid,
        prompt=effective_prompt,
        input_paths=[character_photo],
        frame_sec=frame_sec,
        status=SegmentImageStatus.queued,
    )
    db.add(job)
    _spend_reserve(
        spend_mod.reserve, db, spend_mod.segment_image_scope(job.id),
        spend_mod.image_credits(), kind="segment image", run_id=rid,
    )
    db.commit()

    enqueue_process_segment_image(job.id)
    log.info(
        "Created segment image job %s for run %s segment %s (frame_sec=%.2f)",
        job.id, rid, rsid, frame_sec,
    )
    return SegmentImageJobCreateResponse(job_id=job.id, status=_status_text(job.status))


@router.get("/runs/{rid}/segments/{rsid}/image-jobs/{jid}/image")
def get_segment_image_job_file(
    rid: str,
    rsid: str,
    jid: str,
    which: str = Query("output", pattern="^(output|frame|character)$"),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serve one attempt's output image, its input frame, or its character photo.

    Modelled on ``get_run_preprocessed_image``. ``which=character`` 404s when
    the character photo is a remote URL rather than a local path — the UI
    falls back to a plain link in that case.
    """
    _get_run_or_404(rid, db)
    job = db.get(RunSegmentImageJob, jid)
    if job is None or job.run_id != rid or job.run_segment_id != rsid:
        raise HTTPException(status_code=404, detail=f"Image job {jid!r} not found")

    paths = list(job.input_paths or [])
    if which == "output":
        path = job.output_path
    elif which == "frame":
        path = paths[1] if len(paths) > 1 else None
    else:  # character
        path = paths[0] if paths else None
        if path and (path.startswith("http://") or path.startswith("https://")):
            raise HTTPException(
                status_code=404,
                detail="Character photo is a remote URL, not a local file",
            )

    return _serve_local_reference(path, what=f"{which} image")


@router.post(
    "/runs/{rid}/segments/{rsid}/image-jobs/{jid}/apply",
    response_model=RunSegmentResponse,
)
def apply_segment_image_job(
    rid: str,
    rsid: str,
    jid: str,
    db: Session = Depends(get_db),
) -> RunSegmentResponse:
    """Make attempt *jid*'s result this segment's applied generated image.

    Lets the operator go back to an older ``done`` attempt after a newer one
    superseded it (every success auto-applies itself — see
    ``pipeline_segment_image.poll_segment_image_job_once``) or after
    "Clear generated image".
    """
    run = _get_run_or_404(rid, db)
    _refuse_images_only_run(run, "a per-segment GPT Image regeneration")
    db.refresh(run, with_for_update=True)  # AICP-363 §4.3 (run version fence)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in _SEGMENT_EDITABLE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot apply image: run status is {run.status!r}; "
                "expected 'done', 'failed', or 'incomplete'"
            ),
        )
    rs = _get_run_segment_or_404(rsid, rid, db)
    job = db.get(RunSegmentImageJob, jid)
    if job is None or job.run_segment_id != rsid:
        raise HTTPException(
            status_code=404,
            detail=f"Image job {jid!r} not found for segment {rsid!r}",
        )
    if job.status != SegmentImageStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Image job {jid!r} is not done (status={job.status.value!r})",
        )

    character_photo = (job.input_paths or [None])[0]
    rs.generated_reference_urls = [character_photo, job.output_path]
    rs.generated_image_job_id = job.id
    db.commit()
    db.refresh(rs)
    log.info("Applied image job %s to RunSegment %s (run %s)", jid, rsid, rid)
    return RunSegmentResponse.model_validate(rs)


@router.delete(
    "/runs/{rid}/segments/{rsid}/generated-image",
    response_model=RunSegmentResponse,
)
def clear_segment_generated_image(
    rid: str,
    rsid: str,
    db: Session = Depends(get_db),
) -> RunSegmentResponse:
    """Drop the applied generated image for this segment.

    ``submit_run`` falls back to the hand override (if any) or the run-level
    references. Does not delete the job row or its files — the attempt
    history is unaffected and any ``done`` attempt can still be re-Applied.
    """
    run = _get_run_or_404(rid, db)
    db.refresh(run, with_for_update=True)  # AICP-363 §4.3 (run version fence)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in _SEGMENT_EDITABLE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot clear generated image: run status is {run.status!r}; "
                "expected 'done', 'failed', or 'incomplete'"
            ),
        )
    rs = _get_run_segment_or_404(rsid, rid, db)
    rs.generated_reference_urls = None
    rs.generated_image_job_id = None
    db.commit()
    db.refresh(rs)
    log.info("Cleared generated image for RunSegment %s (run %s)", rsid, rid)
    return RunSegmentResponse.model_validate(rs)


@router.get("/runs/{rid}/segments/{rsid}/result")
def download_run_segment_result(
    rid: str,
    rsid: str,
    download: bool = Query(
        False, description="Force a download instead of inline playback"
    ),
):
    """Serve ONE segment's generated clip — the file we downloaded from kie.ai.

    The stitched final answers "how did the run come out"; this answers "what
    did the model actually return for segment N", which is the only way to tell
    a bad generation from a bad stitch without pulling the whole run apart. The
    bytes are the untouched kie.ai result (``RunSegment.local_result_path``),
    kept on disk under ``runs/<id>/results/result_<index>.mp4`` for the life of
    the run — the same file the stitch consumed.

    Inline by default so a plain link opens the clip in a browser tab;
    ``?download=1`` names it ``<short_id>_seg<NN>.mp4`` and attaches it.

    404 while the segment is still generating (or after a re-run cleared its
    result) — there is genuinely nothing to show yet. No status gate on the
    RUN: a segment that has landed is watchable while its siblings are still
    generating, and on a failed/incomplete run the segments that DID succeed
    are exactly what the operator needs to look at.

    ``?v=`` is a cache buster only (see web_v2._segment_clip_url): a
    re-run rewrites the same path, so the URL has to change with the bytes.
    """
    with get_session() as db:
        run = _get_run_or_404(rid, db)
        rs = _get_run_segment_or_404(rsid, rid, db)
        path = rs.local_result_path
        if not path or not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"No generated clip on disk for segment {rs.index}",
            )
        # Read while the session is open; the transfer happens after it closes.
        filename = f"{run.short_id}_seg{rs.index:02d}.mp4" if download else None

    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename=filename,
        extra_headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


# ---------------------------------------------------------------------------
# AICP-348 — re-link a run whose delivered Drive file was deleted
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/drive-file")
def get_run_drive_file(rid: str, db: Session = Depends(get_db)) -> dict:
    """FRESH state of the run's Drive file (``present`` / ``trashed`` /
    ``missing`` / ``unknown``), whether a re-link is possible, the exports
    that carried the run's link, a point-in-time scan of the Final Creatives
    sheet for it, and the re-link history. Read-only (one Drive metadata read,
    a few Sheets reads); advisory — the POST and the worker re-check."""
    from . import drive_relink

    run = _get_run_or_404(rid, db)
    return drive_relink.inspect(db, run)


@router.post("/runs/{rid}/drive-relink", status_code=status.HTTP_202_ACCEPTED, dependencies=[GATED])
def relink_run_drive_file(
    rid: str, body: DriveRelinkRequest, db: Session = Depends(get_db),
) -> dict:
    """Upload the run's current final as a NEW Drive file and re-point the run
    at it — only for a Drive file confirmed MISSING (AICP-348). The Creative
    Link changes; the Job ID does not. Admission only: under the Run-row lock
    it commits a ``queued`` re-link row and enqueues the worker, which repeats
    every check (and the fresh 404) under the same lock before uploading. A
    ``queued`` row whose job died is resumed, never duplicated. See
    app/drive_relink.py."""
    from . import drive_relink
    from .models import RunDriveRelink
    from .tasks import drive_relink_job_live, enqueue_drive_relink

    if not body.confirm_link_change:
        raise HTTPException(
            status_code=400,
            detail=(
                "Re-linking changes the run's Creative Link (the Job ID stays). "
                "Resend with confirm_link_change=true to go ahead."
            ),
        )
    run = _get_run_or_404(rid, db)
    _lock_run_for_result_mutation(run, db)
    if run.result_gdrive_file_id != body.expected_old_file_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The run's Drive file is now {run.result_gdrive_file_id!r}, not "
                f"{body.expected_old_file_id!r} — reload the run page."
            ),
        )
    existing = drive_relink.active_relink(db, rid)
    if existing is not None:
        try:
            live = drive_relink_job_live(existing.id)
        except Exception as exc:  # noqa: BLE001 - never guess "dead"
            raise HTTPException(
                status_code=503,
                detail=f"Could not check the queued re-link's job: {exc}",
            ) from exc
        if live:
            raise HTTPException(
                status_code=409,
                detail=f"A re-link ({existing.id}) is already in progress for this run.",
            )
        if existing.old_file_id != body.expected_old_file_id:
            raise HTTPException(
                status_code=409,
                detail="A stale re-link for another Drive file is queued — reload the run page.",
            )
        # Its worker died: resume THE SAME row, whose Drive tag lets the worker
        # adopt whatever that worker already uploaded.
        db.commit()
        enqueue_drive_relink(existing.id)
        return {"status": "queued", "resumed": True,
                "relink": drive_relink.relink_view(existing)}

    reason = drive_relink.blocking_reason(db, run)
    if reason is not None:
        raise HTTPException(status_code=409, detail=f"Cannot re-link: {reason}.")
    gdrive = drive_relink.GDriveClient()
    not_missing = drive_relink.confirm_missing(
        gdrive, run.result_gdrive_file_id,
        run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID,
    )
    if not_missing is not None:
        uncertain = not_missing.startswith("could not")
        raise HTTPException(
            status_code=503 if uncertain else 409,
            detail=f"Cannot re-link: {not_missing}.",
        )
    # Reserved id, created by the worker with exactly this id: the create is
    # idempotent and its outcome a files.get by id (app/drive_relink.py). An
    # earlier unfinished re-link of this same old file hands its id on, so an
    # upload of it that lands late is adopted, never a second file.
    target_file_id = db.execute(
        select(RunDriveRelink.target_file_id)
        .where(
            RunDriveRelink.run_id == run.id,
            RunDriveRelink.old_file_id == run.result_gdrive_file_id,
            RunDriveRelink.status != "done",
        )
        .order_by(desc(RunDriveRelink.created_at))
        .limit(1)
    ).scalar_one_or_none()
    if target_file_id is not None:
        try:
            inherited = gdrive.file_state(target_file_id)
        except Exception as exc:  # noqa: BLE001 - keep it: it may still land
            log.warning("drive relink: could not check inherited id %s: %s", target_file_id, exc)
            inherited = None
        if inherited == "trashed":
            # It landed and someone binned it (as a refusal invites): nothing of
            # it is in flight any more, and reusing it would block every resume.
            target_file_id = None
    if target_file_id is None:
        try:
            target_file_id = gdrive.generate_file_id()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503, detail=f"Drive could not reserve a file id: {exc}",
            ) from exc
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    row = RunDriveRelink(
        run_id=run.id, status=drive_relink.ACTIVE,
        old_file_id=run.result_gdrive_file_id, target_file_id=target_file_id,
        result_path=run.result_local_path, run_status=status_val,
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError as exc:  # uq_run_drive_relinks_active_per_run
        db.rollback()
        raise HTTPException(
            status_code=409, detail="A re-link is already in progress for this run.",
        ) from exc
    db.commit()
    try:
        enqueue_drive_relink(row.id)
    except Exception as exc:  # noqa: BLE001
        # The row stays queued with no job: the next POST resumes it.
        log.error("drive relink %s: enqueue failed: %s", row.id, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Re-link recorded but not started ({exc}); press it again to resume.",
        ) from exc
    log.warning("Run %s: drive re-link %s admitted (old file %s confirmed missing)",
                rid, row.id, row.old_file_id)
    return {"status": "queued", "resumed": False, "relink": drive_relink.relink_view(row)}


@router.post("/runs/{rid}/retry", response_model=RunResponse, dependencies=[GATED])
def retry_run(rid: str, db: Session = Depends(get_db)) -> RunResponse:
    """Re-enqueue a failed or stuck run.

    Accepted statuses
    -----------------
    - ``failed``     — normal retry after a processing error.
    - ``queued``     — orphaned run stuck in queue with no live RQ job.
    - ``processing`` — orphaned run whose worker crashed mid-flight.
    - ``stitching``  — orphaned run whose worker crashed during stitch.
    - ``delivering`` — orphaned run whose worker crashed during GDrive upload.

    Safety note
    -----------
    This endpoint does NOT check whether the run is genuinely idle vs. actively
    being worked on by another worker right now.  Do NOT call it on a run that
    is legitimately in-progress: re-enqueuing a live run would double-process it.
    The startup reconciliation routine (``recovery.py``) uses a queue-idle guard
    to avoid this; a human operator must exercise the same judgement when calling
    this endpoint manually.

    For automatic safe recovery, rely on the startup reconciliation.  Use this
    endpoint for manual intervention only.
    """
    run = _get_run_or_404(rid, db)
    _lock_run_for_result_mutation(run, db)
    _require_segment_reconstruction_allowed(run.id, db)
    _require_stitch_library_clips(run, db)
    active_attempt = _active_range_attempt(rid, db)
    if active_attempt is not None:
        raise _range_attempt_conflict(active_attempt)
    active_patch = _active_audio_patch(rid, db)
    if active_patch is not None:
        raise _audio_patch_conflict(active_patch)

    _RETRYABLE = {
        RunStatus.failed,
        RunStatus.incomplete,
        RunStatus.queued,
        RunStatus.processing,
        RunStatus.stitching,
        RunStatus.delivering,
    }

    if run.status not in _RETRYABLE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot retry: run status is {run.status!r}. "
                f"Retryable statuses: {sorted(s.value for s in _RETRYABLE)}"
            ),
        )

    # A retry from `failed`/`incomplete` re-enqueues run_submit_run, which
    # reads references ONCE (pipeline_v2._build_submit_state) — same hazard
    # rerun_segment/patch_run_segment/regenerate_segment_image already guard
    # against, just reached from a different button. Without this, "Retry
    # Run" on an incomplete run bills Seedance/Gemini from the OLD still
    # while a GPT Image 2 regeneration is in flight, and that job's later
    # auto-apply becomes a UI illusion over a video that never used it.
    active_image_job = _run_has_active_segment_image_job(rid, db)
    if active_image_job is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot retry: a GPT Image regeneration is in progress on "
                f"segment {active_image_job.run_segment_id} (job "
                f"{active_image_job.id}, status="
                f"{active_image_job.status.value!r}) — wait for it to finish."
            ),
        )

    original_status = run.status

    if spend_mod.is_images_only_run(run):
        return _retry_images_only_run(run, db)

    # A run whose GPT Image 2 phase left an empty reference slot must go back to
    # PREPROCESSING, not to submit. failed → queued enqueues run_submit_run,
    # which cannot upload a None: it re-uploads the surviving refs, dies on
    # None.strip() and lands the run right back in `failed` — so the operator's
    # Retry button is a permanent dead end (run ce9b5bfd, 2026-08-27). The
    # preprocess phase is resume-aware: a ref that already has its image is
    # skipped, and an outstanding task is re-polled rather than recreated, so
    # this re-downloads the missing image without re-billing GPT Image 2 for
    # the ones that worked.
    #
    # Guarded by "no Seedance task was ever created for this run": once a
    # segment has been billed, the failure being retried is a VIDEO failure and
    # belongs on the normal submit path, whatever the reference list looks like.
    # `preprocess_auto_approve` is deliberately left alone — a copy-run that was
    # launched to skip the review gate should still skip it here; the operator
    # pressed Retry, not Approve.
    #
    # `failed` only, deliberately. An `incomplete` run is incomplete BECAUSE
    # segments were submitted and some came back failed, so it always has a
    # billed segment and could never pass the guard below anyway — and
    # `incomplete → preprocessing` is not an edge, so reaching the transition
    # would 409 the operator out of a retry that works today.
    if run.status == RunStatus.failed:
        from .pipeline_v2 import preprocess_hole_index

        hole = preprocess_hole_index(run)
        if hole is not None and not any(
            rs.seedance_task_id for rs in run.run_segments
        ):
            try:
                transition(run, RunStatus.preprocessing)
            except InvalidTransition as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            run.error_message = None
            _spend_reserve(
                spend_mod.reserve_run, db, run, kind="retry run",
                release_unused=True,
            )
            db.commit()
            enqueue_preprocess_run(rid)
            log.info(
                "Retrying run %s via preprocessing (was %s; reference %d has no "
                "generated image)", rid, original_status, hole,
            )
            return RunResponse.model_validate(run)

    # For already-queued runs, skip the transition (already queued).
    if run.status != RunStatus.queued:
        try:
            transition(run, RunStatus.queued)
        except InvalidTransition:
            # stitching/delivering → queued lacks a direct edge; go via failed.
            try:
                transition(run, RunStatus.failed)
                transition(run, RunStatus.queued)
            except InvalidTransition as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Only a retry of a FINISHED run (failed/incomplete) is new paid work:
    # it reserves what it would newly bill — segments with no result and no
    # live kie.ai task (a stitch/delivery-only retry estimates 0 and is never
    # refused) — and releases the launch's dead, unconsumed remainder. An
    # orphan in an active status (queued/processing/stitching/delivering) is
    # the SAME admitted work resumed: its launch reservation still covers it,
    # so it is neither re-reserved (that would double-count) nor refused.
    if original_status in (RunStatus.failed, RunStatus.incomplete):
        _spend_reserve(
            spend_mod.reserve_run, db, run, kind="retry run", release_unused=True,
        )
    db.commit()
    enqueue_process_run(rid)

    log.info("Retrying run %s (was %s)", rid, original_status)
    return RunResponse.model_validate(run)


def _retry_images_only_run(run: Run, db: Session) -> RunResponse:
    """Retry Run of an images-only run (design §7.3): from failed/incomplete
    every active chunk that is not completed goes back to ``pending`` (a NEW
    attempt), under the Run-row lock and in the transaction of the
    reservation; an orphan in an active status is the same admitted work
    resumed (no reset, no new reservation)."""
    from . import pipeline_rebuild

    original_status = run.status
    finished = original_status in (RunStatus.failed, RunStatus.incomplete)
    if finished:
        _require_rebuild_enabled_for_repair(run)
        try:
            pipeline_rebuild.reset_for_retry(db, run)
        except pipeline_rebuild.RepairRefused as exc:
            raise _rebuild_repair_refused(exc) from exc
    if run.status != RunStatus.queued:
        try:
            transition(run, RunStatus.queued)
        except InvalidTransition:
            try:
                transition(run, RunStatus.failed)
                transition(run, RunStatus.queued)
            except InvalidTransition as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
    run.error_message = None
    if finished:
        _spend_reserve(spend_mod.reserve_run, db, run, kind="retry run", release_unused=True)
    db.commit()
    enqueue_process_run(run.id)
    log.info("Retrying images-only run %s (was %s)", run.id, original_status)
    return RunResponse.model_validate(run)


# ---------------------------------------------------------------------------
# PIP insert — merge endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/runs/{run_id}/merge",
    status_code=status.HTTP_201_CREATED,
    response_model=MergeCreateResponse,
    dependencies=[GATED],
)
def create_merge(
    run_id: str,
    target_project_id: Optional[str] = Form(None),
    x: Optional[int] = Form(None),
    y: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    start_sec: Optional[float] = Form(None),
    end_sec: Optional[float] = Form(None),
    corner_radius_px: Optional[int] = Form(None),
    db: Session = Depends(get_db),
) -> MergeCreateResponse:
    """Composite a completed child Run's result back onto a target video.

    *run_id* must be a Run in status=done with a result video on disk.  When
    the run's project is a PIP-insert child (has an ``InsertExtraction``
    row), *target_project_id* defaults to the extraction's parent project and
    the placement fields (x/y/width/height/start_sec/end_sec/corner_radius_px)
    default to the extraction's recorded rect/window/radius — each field may
    still be overridden individually (e.g. to merge into a different target
    video, or at different coordinates/time window). If the run's project has
    no ``InsertExtraction``, *target_project_id* and the full placement must
    be supplied explicitly.

    Validates: the target project exists and has a probed source video on
    disk; the rect fits inside the target's probed dimensions; the time
    window is within [0, duration_sec]; corner_radius_px is non-negative.
    Creates an ``InsertMerge`` row (status=queued) and enqueues
    :func:`pipeline_merge.process_merge`.
    """
    run = _get_run_or_404(run_id, db)
    if run.status != RunStatus.done:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot merge: run status is {run.status!r}, expected 'done'"
            ),
        )
    if not run.result_local_path or not os.path.exists(run.result_local_path):
        raise HTTPException(
            status_code=409, detail="Cannot merge: run has no result video on disk"
        )

    extraction = db.execute(
        select(InsertExtraction).where(
            InsertExtraction.child_project_id == run.project_id
        )
    ).scalar_one_or_none()

    resolved_target_id = target_project_id or (
        extraction.parent_project_id if extraction is not None else None
    )
    if not resolved_target_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "target_project_id is required (this run's project has no "
                "InsertExtraction to default a target from)"
            ),
        )

    target = _get_project_or_404(resolved_target_id, db)
    _lock_runs_for_result_admission(db, _merge_source_run_ids(db, run_id, resolved_target_id))
    if run.status != RunStatus.done or not run.result_local_path or not os.path.exists(run.result_local_path):
        raise HTTPException(status_code=409, detail="Cannot merge: source Run result changed; reload and retry")
    if not target.source_local_path or not os.path.exists(target.source_local_path):
        raise HTTPException(
            status_code=409,
            detail="Cannot merge: target project has no source video on disk",
        )
    if target.width is None or target.height is None or target.duration_sec is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot merge: target project has not finished analyzing yet "
                "(no probed dimensions/duration)"
            ),
        )

    def _prefill(value, attr: str):
        if value is not None:
            return value
        return getattr(extraction, attr) if extraction is not None else None

    rx = _prefill(x, "x")
    ry = _prefill(y, "y")
    rw = _prefill(width, "width")
    rh = _prefill(height, "height")
    rstart = _prefill(start_sec, "start_sec")
    rend = _prefill(end_sec, "end_sec")
    rradius = (
        corner_radius_px
        if corner_radius_px is not None
        else (extraction.corner_radius_px if extraction is not None else 0)
    )

    if None in (rx, ry, rw, rh, rstart, rend):
        raise HTTPException(
            status_code=400,
            detail=(
                "x/y/width/height/start_sec/end_sec are required (this run's "
                "project has no InsertExtraction to default placement from)"
            ),
        )

    if rradius < 0:
        raise HTTPException(status_code=400, detail="corner_radius_px must be >= 0")
    if rw <= 0 or rh <= 0:
        raise HTTPException(
            status_code=400, detail="width and height must be positive"
        )
    if rx < 0 or ry < 0 or rx + rw > target.width or ry + rh > target.height:
        raise HTTPException(
            status_code=400,
            detail=(
                f"rect (x={rx}, y={ry}, w={rw}, h={rh}) does not fit inside "
                f"target dimensions {target.width}x{target.height}"
            ),
        )
    if not (0 <= rstart < rend <= target.duration_sec):
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid time range [{rstart}, {rend}]; must satisfy "
                f"0 <= start_sec < end_sec <= duration_sec ({target.duration_sec})"
            ),
        )

    merge = InsertMerge(
        target_project_id=resolved_target_id,
        child_run_id=run_id,
        x=rx,
        y=ry,
        width=rw,
        height=rh,
        start_sec=rstart,
        end_sec=rend,
        corner_radius_px=rradius,
        status=MergeStatus.queued,
    )
    db.add(merge)
    db.commit()

    enqueue_process_merge(merge.id)

    log.info(
        "Created merge %s: run %s -> target project %s (rect=%d,%d,%d,%d)",
        merge.id, run_id, resolved_target_id, rx, ry, rw, rh,
    )
    status_val = merge.status.value if hasattr(merge.status, "value") else str(merge.status)
    return MergeCreateResponse(merge_id=merge.id, status=status_val)


@router.post("/merges/{merge_id}/retry", response_model=MergeResponse, dependencies=[GATED])
def retry_merge(merge_id: str, db: Session = Depends(get_db)) -> MergeResponse:
    """Re-enqueue a failed merge (failed -> queued)."""
    merge = _get_merge_or_404(merge_id, db)
    _admit_retained_run_source(db, _merge_source_run_ids(db, merge.child_run_id, merge.target_project_id))
    if merge.status != MergeStatus.failed:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot retry: merge status is {merge.status!r}, expected 'failed'"
            ),
        )
    transition(merge, MergeStatus.queued)
    db.commit()
    enqueue_process_merge(merge_id)

    log.info("Retrying merge %s", merge_id)
    return MergeResponse.model_validate(merge)


@router.get("/merges/{merge_id}", response_model=MergeResponse)
def get_merge(merge_id: str, db: Session = Depends(get_db)) -> MergeResponse:
    """Fetch a single InsertMerge's status/result."""
    merge = _get_merge_or_404(merge_id, db)
    return MergeResponse.model_validate(merge)


@router.get("/projects/{pid}/merges", response_model=list[MergeListItem])
def list_project_merges(pid: str, db: Session = Depends(get_db)) -> list:
    """List InsertMerges whose target is this project, newest first."""
    _get_project_or_404(pid, db)
    merges = (
        db.execute(
            select(InsertMerge)
            .where(InsertMerge.target_project_id == pid)
            .order_by(desc(InsertMerge.created_at))
        )
        .scalars()
        .all()
    )
    return [MergeListItem.model_validate(m) for m in merges]


# ---------------------------------------------------------------------------
# Tail edit endpoints ("Change ending" — trim a done run's final video and
# optionally append a stitch-library clip; see app/pipeline_tail.py)
# ---------------------------------------------------------------------------


@router.post(
    "/runs/{run_id}/tail-edits",
    status_code=status.HTTP_201_CREATED,
    response_model=TailEditCreateResponse,
)
def create_tail_edit(
    run_id: str,
    cut_sec: float = Form(...),
    library_clip_id: Optional[str] = Form(None),
    source_caption_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    adm: Optional[Admission] = ADMISSION,
) -> TailEditCreateResponse:
    """Queue a "change ending" job: trim *run_id*'s final video at *cut_sec*
    and optionally append a library clip.

    *run_id* must be a Run in status=done with a result video on disk (409
    otherwise). *cut_sec* must land strictly inside
    ``(_MIN_CUT_MARGIN_SEC, duration)`` of the RESOLVED source — the run's own
    result, or the referenced caption's output when *source_caption_id* is
    given — checked HERE, synchronously, and returned as 400 naming the exact
    bound (e.g. "cut_sec must be within (0.05, 12.28)"). This used to be
    deferred to the worker on the theory that probing here would add ffprobe
    latency to every request, but this endpoint is an operator click, not a
    poll: one ffprobe (or zero, when the run's duration is already cached in
    ``Run.result_duration_sec``) is cheap next to a job that is created,
    queued, and only fails a worker cycle later. The worker keeps its own
    check as the last line of defence against a file that changed shape after
    this request (a re-encode, a different row winning a race) — this
    endpoint's check does not make that one redundant. *library_clip_id*, if
    given, must reference an existing LibraryClip (400 otherwise); NULL means
    trim only, no clip appended.

    *source_caption_id*, if given, chains this tail edit off a caption job
    instead of the run's own result (docs/burned-captions.md §3.7): it must
    reference a RunCaption that belongs to *run_id* (400 otherwise) and is
    ``done`` with an output file on disk (409 otherwise). Omitted/blank keeps
    ``source_type="run"``, byte-for-byte the v1 behaviour.
    """
    adm = ops_gate.keyed(adm)  # AICP-350 Idempotency-Key (gate already held)
    if adm is not None and adm.replay is not None:
        return ops_gate.replay_one(adm, RunTailEdit, lambda e: TailEditCreateResponse(
            tail_edit_id=e.id, status=_status_text(e.status)))
    run = _get_run_or_404(run_id, db)
    if run.status != RunStatus.done:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot change ending: run status is {run.status!r}, expected 'done'"
            ),
        )
    if not run.result_local_path or not os.path.exists(run.result_local_path):
        raise HTTPException(
            status_code=409, detail="Cannot change ending: run has no result video on disk"
        )
    # Lower bound first, before any probing: a non-positive/near-zero cut_sec
    # is wrong regardless of which source it would be checked against, so
    # there's no reason to pay for a probe just to reject it. Aligned with the
    # worker's own bound (pipeline_tail._MIN_CUT_MARGIN_SEC) — this used to be
    # a bare `cut_sec <= 0` here while the worker required `> _MIN_CUT_MARGIN_SEC`,
    # so a cut_sec in (0, _MIN_CUT_MARGIN_SEC] passed here and failed there.
    if cut_sec <= _MIN_CUT_MARGIN_SEC:
        raise HTTPException(
            status_code=400,
            detail=f"cut_sec must be greater than {_MIN_CUT_MARGIN_SEC}s",
        )

    if library_clip_id and db.get(LibraryClip, library_clip_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Library clip not found: {library_clip_id}",
        )

    source_caption = _validate_chained_source(
        db, RunCaption, source_caption_id, run_id, "caption"
    )
    source_type = "caption" if source_caption is not None else "run"
    source_result_path = run.result_local_path

    # Upper bound: resolve the duration of whichever file this edit will
    # actually cut — never the run's, when a caption source is chained in,
    # since that source can be a different (typically shorter) duration than
    # the run (see _resolve_source in pipeline_tail.py). A caption has no
    # duration cache (no migration in this task), so it's probed every time;
    # there are typically 0-1 done captions per run and this is a one-off
    # operator click, not a poll.
    if source_type == "caption":
        try:
            duration_sec = media_mod.probe(source_caption.output_path).duration_sec
        except media_mod.MediaError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot probe source caption's output video: {exc}",
            ) from exc
    else:
        duration_sec = run.result_duration_sec
        if duration_sec is None:
            # result_duration_sec is cached at finalize; NULL means this run
            # predates the column. Probe once and persist the value so this
            # run never re-probes on a future request — the same self-healing
            # backfill as web_v2._get_tail_edit_context.
            try:
                duration_sec = media_mod.probe(run.result_local_path).duration_sec
            except media_mod.MediaError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot probe run result video: {exc}",
                ) from exc
            duration_sec = _backfill_result_duration(run, db, source_result_path, duration_sec)

    if source_type == "run":
        _lock_run_for_result_mutation(run, db)
        if (run.status != RunStatus.done or run.result_local_path != source_result_path
                or duration_sec is None):
            raise HTTPException(status_code=409, detail="The Run result changed; reload before changing its ending")

    if not (_MIN_CUT_MARGIN_SEC < cut_sec < duration_sec):
        # floor2, never bare :.2f — :.2f ROUNDS, so a 12.288s duration would
        # be named "12.29" here: a value >= the real duration, making the
        # exact number this message tells the operator to retry with just as
        # unreachable as the bug this endpoint exists to close. floor2
        # matches what the UI's "Max" button already offers.
        raise HTTPException(
            status_code=400,
            detail=(
                f"cut_sec must be within ({_MIN_CUT_MARGIN_SEC}, "
                f"{_floor2(duration_sec):.2f}) — the resolved source video "
                "is that long"
            ),
        )

    edit = RunTailEdit(
        run_id=run_id,
        cut_sec=cut_sec,
        library_clip_id=library_clip_id or None,
        source_type=source_type,
        source_caption_id=source_caption.id if source_caption is not None else None,
        status=TailEditStatus.queued,
    )
    db.add(edit)
    if adm is not None:
        db.flush()
        adm.record("run_tail_edit", [edit.id])
    db.commit()

    enqueue_process_tail_edit(edit.id)

    log.info(
        "Created tail edit %s: run %s cut_sec=%.2f library_clip_id=%s "
        "source_type=%s source_caption_id=%s",
        edit.id, run_id, cut_sec, library_clip_id, source_type, source_caption_id,
    )
    status_val = edit.status.value if hasattr(edit.status, "value") else str(edit.status)
    return TailEditCreateResponse(tail_edit_id=edit.id, status=status_val)


@router.get("/runs/{run_id}/tail-edits", response_model=list[TailEditListItem])
def list_run_tail_edits(run_id: str, db: Session = Depends(get_db)) -> list:
    """List tail edits for *run_id*, newest first."""
    _get_run_or_404(run_id, db)
    edits = (
        db.execute(
            select(RunTailEdit)
            .where(RunTailEdit.run_id == run_id)
            .order_by(desc(RunTailEdit.created_at))
        )
        .scalars()
        .all()
    )
    # Batch-resolve library clip names (soft reference — no FK/relationship,
    # so a clip may have been deleted since the edit was created).
    clip_ids = {e.library_clip_id for e in edits if e.library_clip_id}
    clip_names: dict[str, str] = {}
    if clip_ids:
        clip_names = {
            c.id: c.name
            for c in db.execute(
                select(LibraryClip).where(LibraryClip.id.in_(clip_ids))
            ).scalars()
        }
    items = []
    for e in edits:
        item = TailEditListItem.model_validate(e)
        item.library_clip_name = clip_names.get(e.library_clip_id) if e.library_clip_id else None
        items.append(item)
    return items


@router.post("/tail-edits/{edit_id}/retry", response_model=TailEditResponse)
def retry_tail_edit(edit_id: str, db: Session = Depends(get_db),
                    adm: Optional[Admission] = ADMISSION) -> TailEditResponse:
    """Re-enqueue a failed or done tail edit (failed/done -> queued)."""
    adm = ops_gate.keyed(adm)  # AICP-350 Idempotency-Key (gate already held)
    if adm is not None and adm.replay is not None:
        return ops_gate.replay_one(adm, RunTailEdit, TailEditResponse.model_validate)
    edit = _get_tail_edit_or_404(edit_id, db)
    if edit.source_type == "run":
        _admit_retained_run_source(db, [edit.run_id])
    if edit.status not in (TailEditStatus.failed, TailEditStatus.done):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot retry: tail edit status is {edit.status!r}, "
                "expected 'failed' or 'done'"
            ),
        )
    transition(edit, TailEditStatus.queued)
    if adm is not None:
        adm.record("run_tail_edit", [edit_id])
    db.commit()
    enqueue_process_tail_edit(edit_id)

    log.info("Retrying tail edit %s", edit_id)
    return TailEditResponse.model_validate(edit)


@router.get("/tail-edits/{edit_id}/result")
def download_tail_edit_result(edit_id: str):
    """Download the output video for a done tail edit."""
    # Session is closed before the file transfer (same as /source).
    with get_session() as db:
        edit = _get_tail_edit_or_404(edit_id, db)
        status_val = edit.status.value if hasattr(edit.status, "value") else str(edit.status)
        if status_val != "done":
            raise HTTPException(
                status_code=409,
                detail=f"Tail edit not done yet: status is {edit.status!r}",
            )
        path = edit.output_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result file not found on disk")

        filename = os.path.basename(path)

    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename=filename,
        extra_headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


# Overlay states in which the worker may still read a referenced tail edit's
# output — block tail-edit deletion for these (mirrors
# api_overlay._ACTIVE_OVERLAY_STATUSES).
_ACTIVE_OVERLAY_STATUSES = {
    OverlayStatus.queued,
    OverlayStatus.processing,
    OverlayStatus.delivering,
}


@router.delete("/tail-edits/{edit_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tail_edit(edit_id: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a tail edit: DB row + its output directory on disk.

    Blocked (409) while the edit itself is active (queued/processing/
    delivering), or while an active overlay job still points at it. Terminal
    overlay jobs keep their soft reference (no FK); a later re-run of one
    fails with a clear "source tail edit not found" error — same convention
    as delete_library_clip. The Google Drive copy and any Final Creatives
    rows already exported are NOT removed.
    """
    edit = _get_tail_edit_or_404(edit_id, db)
    if edit.status in _ACTIVE_TAIL_EDIT_STATUSES:
        status_val = edit.status.value if hasattr(edit.status, "value") else str(edit.status)
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a tail edit while it is {status_val!r}",
        )

    overlay_referencing = (
        db.execute(
            select(OverlayJob.id).where(
                OverlayJob.source_tail_edit_id == edit_id,
                OverlayJob.status.in_(_ACTIVE_OVERLAY_STATUSES),
            )
        )
        .scalars()
        .all()
    )
    if overlay_referencing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Tail edit is used by {len(overlay_referencing)} active overlay "
                f"job(s) ({', '.join(overlay_referencing[:3])}…) — wait for them "
                "to finish or delete them first"
            ),
        )

    run = db.get(Run, edit.run_id)
    project_id = run.project_id if run else None
    db.delete(edit)
    db.commit()
    if project_id:
        shutil.rmtree(tail_edit_dir(edit_id, project_id), ignore_errors=True)
    log.info("Deleted tail edit %s (db + disk)", edit_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Export a delivered run to the "Final Creatives" tracking spreadsheet
# ---------------------------------------------------------------------------

# Fixed values for the export (per the operator workflow).
_PIPELINE_TYPE = "re-skin-v2"
_RUN_ON_META = "need_to_rename"


@router.post("/runs/{rid}/export-to-sheet", dependencies=[GATED])
def export_run_to_sheet(rid: str, db: Session = Depends(get_db)) -> dict:
    """Append a row for this run to the "Final Creatives" tab of the tracking sheet.

    Columns written (matched by header name): Date (today, YYYY-MM-DD),
    Job ID (short run id — the 8-char prefix shown in the UI plus the run's
    language suffix, since the full UUID makes creative names unreadable),
    Input Link (project source_ref), Creative Link (the run's Google Drive
    result), Pipeline Type (``re-skin-v2``), Run on Meta (``need_to_rename``).

    Idempotent on two levels: if the run is already marked exported
    (``final_creatives_exported_at``) we return immediately without touching the
    sheet; otherwise we still dedupe by Job ID against the sheet before
    appending. Either way the run is marked exported so the UI shows it as added.
    """
    run = _get_run_or_404(rid, db)

    # Fast path: already logged — no Sheets call needed.
    if run.final_creatives_exported_at is not None:
        return {
            "status": "exists",
            "message": "This run is already in Final Creatives.",
        }

    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot export: run status is {status_val!r}, expected 'done'.",
        )
    if not run.result_gdrive_file_id:
        raise HTTPException(
            status_code=409,
            detail="Cannot export: run has no Google Drive result link yet.",
        )

    project = run.project
    # Short Job ID: the 8-char prefix shown in the UI, plus the run's language
    # suffix (``1736426f-ja``; nothing extra for English). The full UUID is used
    # downstream in creative file names and makes them unreadable, and the
    # suffix is what the renaming pipeline parses the language back out of.
    job_id = run.short_id
    creative_link = (
        f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
    )
    fields = {
        "Date": date.today().isoformat(),
        "Job ID": job_id,
        "Input Link": project.source_ref if project else "",
        "Creative Link": creative_link,
        "Pipeline Type": _PIPELINE_TYPE,
        "Run on Meta": _RUN_ON_META,
    }

    # Mark the run exported optimistically so the UI fast-path ("already
    # exported" check) works on page reload and so a duplicate click is
    # rejected before the job even runs.  The actual Sheets I/O (3 round-trips)
    # happens in an RQ job on the overlay queue so it does not block the
    # request.
    run.final_creatives_exported_at = datetime.now(timezone.utc)
    db.commit()

    enqueue_export_or_rollback(
        enqueue_export_run_to_sheet, rid, run, "final_creatives_exported_at", db
    )
    log.info("Enqueued Final Creatives export for run %s", rid)
    return {
        "status": "added",
        "message": "Added to Final Creatives.",
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Export a tail edit's result to the "Final Creatives" tracking spreadsheet
# ---------------------------------------------------------------------------

# Fixed Pipeline Type for the tail-edit export. Distinct from the run's
# ``re-skin-v2`` (mirrors ``re-skin-overlay``): a tail edit is a DERIVED
# creative, so it gets its own id prefix as Job ID and its own type — reusing
# the run's Job ID would collide with the run's own row (and with sibling
# tail edits) in the sheet dedupe.
_TAIL_EDIT_PIPELINE_TYPE = "re-skin-tail-edit"


@router.post("/tail-edits/{edit_id}/export-to-sheet", dependencies=[GATED])
def export_tail_edit_to_sheet(edit_id: str, db: Session = Depends(get_db)) -> dict:
    """Append a row for this tail edit's result to the "Final Creatives" tab.

    Mirrors the overlay export (the existing precedent for a derived
    creative): idempotent via ``final_creatives_exported_at`` and a Job-ID
    dedupe against the sheet. Job ID is the tail edit's own 8-char prefix
    carrying the parent run's language suffix; Input Link is the parent run's
    creative Drive link; Creative Link is the tail edit's own Drive result.
    """
    edit = _get_tail_edit_or_404(edit_id, db)

    # Fast path: already logged — no Sheets call needed.
    if edit.final_creatives_exported_at is not None:
        return {
            "status": "exists",
            "message": "This tail edit is already in Final Creatives.",
        }

    status_val = edit.status.value if hasattr(edit.status, "value") else str(edit.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot export: tail edit status is {status_val!r}, expected 'done'.",
        )
    if not edit.gdrive_file_id:
        raise HTTPException(
            status_code=409,
            detail="Cannot export: tail edit has no Google Drive result link yet.",
        )

    # Short Job ID: the tail edit's OWN 8-char prefix (mirrors the overlay
    # export). Using the parent run's prefix would collide with the run's own
    # Final Creatives row (and with sibling tail edits) in the sheet dedupe.
    # The LANGUAGE, though, is inherited from the parent run — a re-cut of a
    # Japanese creative is still Japanese, and without the suffix the renaming
    # pipeline would default it to English.
    run = edit.run
    job_id = languages.job_id_for(edit.id, run.language if run else None)
    creative_link = f"https://drive.google.com/file/d/{edit.gdrive_file_id}/view"
    # Input Link: the parent run's creative Drive link (mirrors the overlay
    # export's source-run fallback logic).
    if run and run.result_gdrive_file_id:
        input_link = (
            f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
        )
    else:
        input_link = edit.run_id
    fields = {
        "Date": date.today().isoformat(),
        "Job ID": job_id,
        "Input Link": input_link,
        "Creative Link": creative_link,
        "Pipeline Type": _TAIL_EDIT_PIPELINE_TYPE,
        "Run on Meta": _RUN_ON_META,
    }

    # Mark the tail edit exported optimistically so the UI fast-path ("already
    # exported" check) works on page reload and so a duplicate click is
    # rejected before the job even runs.  The actual Sheets I/O (3 round-trips)
    # happens in an RQ job on the overlay queue so it does not block the
    # request.
    edit.final_creatives_exported_at = datetime.now(timezone.utc)
    db.commit()

    enqueue_export_or_rollback(
        enqueue_export_tail_edit_to_sheet, edit_id, edit,
        "final_creatives_exported_at", db,
    )
    log.info("Enqueued Final Creatives export for tail edit %s", edit_id)
    return {
        "status": "added",
        "message": "Added to Final Creatives.",
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Caption endpoints ("Add Captions" — re-transcribe a done run's delivered
# result, burn subtitles in with ffmpeg, deliver as a new derived creative;
# see docs/burned-captions.md and app/pipeline_captions.py). Mirrors the tail
# edit endpoints above ("a derived creative built from a done Run's result,
# delivered to Drive, exportable to the sheet") — same shapes, same guards,
# minus the library-clip concerns a tail edit has and a caption does not
# (docs/burned-captions.md §2.1, §4).
# ---------------------------------------------------------------------------

# "manual" lines_source value (docs/burned-captions.md §2.1) — an operator
# supplied/edited `lines` via PATCH and the transcribe step must be skipped.
# Duplicated from pipeline_captions.LINES_SOURCE_MANUAL rather than imported:
# that module pulls in localisation/media/gdrive_client, which this API layer
# has no other reason to load.
_LINES_SOURCE_MANUAL = "manual"

# Caption states in which a worker may still be touching the row or its output
# file — block deletion while in any of these (mirrors
# _ACTIVE_TAIL_EDIT_STATUSES).
_ACTIVE_CAPTION_STATUSES = {
    CaptionStatus.queued,
    CaptionStatus.transcribing,
    CaptionStatus.burning,
    CaptionStatus.delivering,
}

# Caption states in which the worker has already started reading `lines` —
# PATCH /captions/{id}/lines 409s here rather than racing the burn. `queued`
# is deliberately NOT included: nothing has read the row yet, so an operator
# may still edit before the first attempt (or before pressing retry) starts.
_CAPTION_INFLIGHT_STATUSES = {
    CaptionStatus.transcribing,
    CaptionStatus.burning,
    CaptionStatus.delivering,
}

# Fixed Pipeline Type for the caption export — mirrors _TAIL_EDIT_PIPELINE_TYPE.
_CAPTION_PIPELINE_TYPE = "re-skin-captions"


def _validate_caption_lines_payload(body: Any) -> list[dict]:
    """Validate a ``PATCH /captions/{id}/lines`` body against §3.5, or 400.

    Shape only: every line must be an object with a finite, non-negative
    ``start``/``end`` and a string ``text``. Semantic normalisation (dropping
    blanks, clamping into the video's duration, sorting, de-overlapping) is
    :func:`app.captions.prepare_lines`'s job at burn time, not this
    endpoint's — an operator's raw edit should round-trip back out through
    ``GET`` unchanged, and ``prepare_lines`` is documented to never raise on
    input that IS the right shape but semantically odd (e.g. ``end < start``).
    Reuses :func:`_transcript_number`, the same "reject bools/NaN/negative"
    guard the localisation transcript endpoints use, for exactly the same
    reason: a NaN timestamp would sail through every comparison in
    ``prepare_lines`` silently dropping the line.
    """
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail="body must be a JSON object with a 'lines' array",
        )
    raw_lines = body.get("lines")
    if not isinstance(raw_lines, list):
        raise HTTPException(status_code=400, detail="lines must be an array")

    lines: list[dict] = []
    for position, item in enumerate(raw_lines):
        where = f"lines[{position}]"
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"{where} must be an object")
        start = _transcript_number(item.get("start"), f"{where}.start")
        end = _transcript_number(item.get("end"), f"{where}.end")
        text = item.get("text")
        if not isinstance(text, str):
            raise HTTPException(
                status_code=400, detail=f"{where}.text must be a string"
            )
        lines.append({"start": start, "end": end, "text": text})
    return lines


@router.post(
    "/runs/{run_id}/captions",
    status_code=status.HTTP_201_CREATED,
    response_model=CaptionCreateResponse,
)
def create_caption(
    run_id: str,
    language: Optional[str] = Form(None),
    source_tail_edit_id: Optional[str] = Form(None),
    placement: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    adm: Optional[Admission] = ADMISSION,
) -> CaptionCreateResponse:
    """Queue an "Add Captions" job against *run_id*'s delivered result.

    *run_id* must be a Run in status=done with a result video on disk (409
    otherwise). *language*, if given, must be one of app/languages.py's codes
    (400 otherwise); omitted/blank is fine — it defers language resolution to
    process time (``caption.language or run.language or the transcript's
    source_language``, docs/burned-captions.md §2.1/§3.3). Several caption
    jobs per run are allowed, same as tail edits.

    *source_tail_edit_id*, if given, chains this caption job off a tail edit
    instead of the run's own result (docs/burned-captions.md §3.7): it must
    reference a RunTailEdit that belongs to *run_id* (400 otherwise) and is
    ``done`` with an output file on disk (409 otherwise). Omitted/blank keeps
    ``source_type="run"``, byte-for-byte the v1 behaviour.

    *placement* (AICP-368): ``project`` (default) freezes the project's
    ``caption_placement`` onto this job (none = the house position);
    ``bottom`` burns at the house position whatever the project says.
    """
    adm = ops_gate.keyed(adm)  # AICP-350 Idempotency-Key (gate already held)
    if adm is not None and adm.replay is not None:
        return ops_gate.replay_one(adm, RunCaption, lambda c: CaptionCreateResponse(
            caption_id=c.id, status=_status_text(c.status)))
    placement_choice = (placement or "project").strip() or "project"
    if placement_choice not in caption_placement_mod.CAPTION_CHOICES:
        raise HTTPException(status_code=400, detail="placement must be 'project' or 'bottom'")
    run = _get_run_or_404(run_id, db)
    if not source_tail_edit_id:
        _lock_run_for_result_mutation(run, db)
    if run.status != RunStatus.done:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot add captions: run status is {run.status!r}, expected 'done'"
            ),
        )
    if not run.result_local_path or not os.path.exists(run.result_local_path):
        raise HTTPException(
            status_code=409,
            detail="Cannot add captions: run has no result video on disk",
        )

    language_val = _validate_language(language)

    source_tail_edit = _validate_chained_source(
        db, RunTailEdit, source_tail_edit_id, run_id, "tail edit"
    )
    source_type = "tail_edit" if source_tail_edit is not None else "run"

    placement_val = caption_placement_mod.snapshot_for_caption(
        db.get(VideoProject, run.project_id), placement_choice
    )

    caption = RunCaption(
        # The id is assigned HERE, not at flush: the reservation below is keyed
        # by it, and an unflushed row's id is still None — every caption used
        # to reserve under "caption:None" while its draw went to its own scope
        # (AICP-366: the autopilot attributes spend by these scope keys).
        id=str(uuid.uuid4()),
        run_id=run_id,
        language=language_val,
        source_type=source_type,
        source_tail_edit_id=source_tail_edit.id if source_tail_edit is not None else None,
        placement=placement_val,
        status=CaptionStatus.queued,
    )
    db.add(caption)
    # One transcription call (none if a same-audio caption's lines get
    # reused — the reservation is then simply never consumed).
    _spend_reserve(
        spend_mod.reserve, db, spend_mod.caption_scope(caption.id),
        spend_mod.llm_credits(), kind="captions", run_id=run_id,
    )
    if adm is not None:
        adm.record("run_caption", [caption.id])
    db.commit()

    enqueue_process_caption(caption.id)

    log.info(
        "Created caption %s: run %s language=%s source_type=%s source_tail_edit_id=%s "
        "placement=%s",
        caption.id, run_id, language_val, source_type, source_tail_edit_id,
        (placement_val or {}).get("mode"),
    )
    status_val = _status_text(caption.status)
    return CaptionCreateResponse(caption_id=caption.id, status=status_val)


@router.get("/runs/{run_id}/captions", response_model=list[CaptionListItem])
def list_run_captions(run_id: str, db: Session = Depends(get_db)) -> list:
    """List caption jobs for *run_id*, newest first."""
    _get_run_or_404(run_id, db)
    captions = (
        db.execute(
            select(RunCaption)
            .where(RunCaption.run_id == run_id)
            .order_by(desc(RunCaption.created_at))
        )
        .scalars()
        .all()
    )
    items = []
    for c in captions:
        item = CaptionListItem.model_validate(c)
        # Not a model column — derived here so the UI/API never has to ship
        # the full `lines` array just to show a count.
        item.line_count = len(c.lines) if c.lines else 0
        items.append(item)
    return items


@router.post("/captions/{caption_id}/retry", response_model=CaptionResponse)
def retry_caption(caption_id: str, db: Session = Depends(get_db),
                  adm: Optional[Admission] = ADMISSION) -> CaptionResponse:
    """Re-enqueue a failed or done caption job (failed/done -> queued)."""
    adm = ops_gate.keyed(adm)  # AICP-350 Idempotency-Key (gate already held)
    if adm is not None and adm.replay is not None:
        return ops_gate.replay_one(adm, RunCaption, CaptionResponse.model_validate)
    caption = _get_caption_or_404(caption_id, db)
    if caption.source_type == "run":
        _admit_retained_run_source(db, [caption.run_id])
    # Run row (above) before the caption row, the same order as the
    # assistant's Apply; re-read the caption under its lock.
    caption = _lock_caption_or_404(caption_id, db)
    if caption.status not in (CaptionStatus.failed, CaptionStatus.done):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot retry: caption status is {caption.status!r}, "
                "expected 'failed' or 'done'"
            ),
        )
    transition(caption, CaptionStatus.queued)
    caption.error_message = None
    _spend_reserve(
        spend_mod.reserve, db, spend_mod.caption_scope(caption_id),
        spend_mod.llm_credits(), kind="captions retry", release_unused=True,
        run_id=caption.run_id,
    )
    if adm is not None:
        adm.record("run_caption", [caption_id])
    db.commit()
    enqueue_process_caption(caption_id)

    log.info("Retrying caption %s", caption_id)
    return CaptionResponse.model_validate(caption)


@router.patch("/captions/{caption_id}/lines", response_model=CaptionResponse)
def update_caption_lines(
    caption_id: str,
    body: dict = Body(...),
    db: Session = Depends(get_db),
) -> CaptionResponse:
    """Operator-supplied/edited caption lines: ``{"lines": [{start, end, text}, …]}``.

    Blocked (409) while the job is in flight (transcribing/burning/
    delivering) — editing under a worker that is actively reading `lines`
    would race it. Sets ``lines_source="manual"`` so the *next* run of the
    worker (triggered by a separate ``POST .../retry`` — this endpoint never
    re-enqueues) skips straight to burning instead of re-transcribing over
    the edit.
    """
    caption = _lock_caption_or_404(caption_id, db)
    if caption.status in _CAPTION_INFLIGHT_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit lines: caption status is {caption.status!r} "
                "(in flight) — wait for it to finish or fail"
            ),
        )

    lines = _validate_caption_lines_payload(body)
    caption.lines = lines
    caption.lines_source = _LINES_SOURCE_MANUAL
    # The operator's text is no longer the donor's copy.
    caption.lines_reused_from = None
    db.commit()

    log.info(
        "Updated lines for caption %s (%d line(s), lines_source=manual)",
        caption_id, len(lines),
    )
    return CaptionResponse.model_validate(caption)


@router.get("/captions/{caption_id}/result")
def download_caption_result(caption_id: str):
    """Download the output video for a done caption job."""
    # Session is closed before the file transfer (same as .../tail-edits/result).
    with get_session() as db:
        caption = _get_caption_or_404(caption_id, db)
        status_val = _status_text(caption.status)
        if status_val != "done":
            raise HTTPException(
                status_code=409,
                detail=f"Caption not done yet: status is {caption.status!r}",
            )
        path = caption.output_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result file not found on disk")

        filename = os.path.basename(path)

    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename=filename,
        extra_headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.delete("/captions/{caption_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_caption(caption_id: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a caption job: DB row + its output directory on disk.

    Blocked (409) while the job is active (queued/transcribing/burning/
    delivering) — mirrors delete_tail_edit's guard. Unlike a tail edit,
    nothing else references a caption job (no overlay source, no library
    clip), so there is no second, dependent-entity guard to check. The Google
    Drive copy and any Final Creatives row already exported are NOT removed.
    """
    caption = _lock_caption_or_404(caption_id, db)
    if caption.status in _ACTIVE_CAPTION_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a caption while it is {_status_text(caption.status)!r}",
        )

    run = db.get(Run, caption.run_id)
    project_id = run.project_id if run else None
    db.delete(caption)
    db.commit()
    if project_id:
        shutil.rmtree(caption_dir(caption_id, project_id), ignore_errors=True)
    log.info("Deleted caption %s (db + disk)", caption_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/captions/{caption_id}/export-to-sheet", dependencies=[GATED])
def export_caption_to_sheet(caption_id: str, db: Session = Depends(get_db)) -> dict:
    """Append a row for this caption job's result to the "Final Creatives" tab.

    Mirrors export_tail_edit_to_sheet exactly: idempotent via
    ``final_creatives_exported_at`` (a repeat call returns the "exists" shape
    rather than erroring), Job ID is the caption's OWN 8-char prefix carrying
    the PARENT RUN's language suffix (the run's own prefix would collide with
    the run's own Final Creatives row in the sheet dedupe), Input Link is the
    parent run's creative Drive link (falling back to the bare run id), and
    Creative Link is the caption's own Drive result.
    """
    caption = _get_caption_or_404(caption_id, db)

    # Fast path: already logged — no Sheets call needed.
    if caption.final_creatives_exported_at is not None:
        return {
            "status": "exists",
            "message": "This caption is already in Final Creatives.",
        }

    status_val = _status_text(caption.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot export: caption status is {status_val!r}, expected 'done'.",
        )
    if not caption.gdrive_file_id:
        raise HTTPException(
            status_code=409,
            detail="Cannot export: caption has no Google Drive result link yet.",
        )

    run = caption.run
    job_id = languages.job_id_for(caption.id, run.language if run else None)
    creative_link = f"https://drive.google.com/file/d/{caption.gdrive_file_id}/view"
    if run and run.result_gdrive_file_id:
        input_link = (
            f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
        )
    else:
        input_link = caption.run_id
    fields = {
        "Date": date.today().isoformat(),
        "Job ID": job_id,
        "Input Link": input_link,
        "Creative Link": creative_link,
        "Pipeline Type": _CAPTION_PIPELINE_TYPE,
        "Run on Meta": _RUN_ON_META,
    }

    # Mark exported optimistically, same reasoning as export_tail_edit_to_sheet:
    # the UI fast-path and a duplicate click both need this set before the RQ
    # job (3 Sheets round-trips) even runs.
    caption.final_creatives_exported_at = datetime.now(timezone.utc)
    db.commit()

    enqueue_export_or_rollback(
        enqueue_export_caption_to_sheet, caption_id, caption,
        "final_creatives_exported_at", db,
    )
    log.info("Enqueued Final Creatives export for caption %s", caption_id)
    return {
        "status": "added",
        "message": "Added to Final Creatives.",
        "fields": fields,
    }


def _copy_reference_files(items: list, dest_dir: str, *, unique_names: bool = False) -> list:
    """Clone a run's reference list for a copied run.

    Local file paths are copied into *dest_dir* (so the copy is self-contained and
    survives deleting the source run); http(s) URLs and missing/odd entries are
    passed through unchanged. *unique_names* names each copy by its list
    position plus a short random stem (keeping the extension), so two distinct
    local files sharing a basename never overwrite each other and repeated
    archive/restore cycles never lengthen the name (AICP-342).
    """
    out: list = []
    made = False
    for pos, ref in enumerate(items or []):
        if isinstance(ref, str) and (ref.startswith("http://") or ref.startswith("https://")):
            out.append(ref)
            continue
        # Only a file under DATA_DIR is copied (AICP-148): an old row holding
        # e.g. "/etc/passwd" must not be laundered into this run's own dir,
        # where it would look like a legitimate upload.
        real = path_within(ref, data_dir_root()) if isinstance(ref, str) else None
        if real is not None and os.path.isfile(real):
            if not made:
                os.makedirs(dest_dir, exist_ok=True)
                made = True
            name = os.path.basename(ref)
            if unique_names:
                # Position + short random stem + the original extension: never
                # grows across archive/restore cycles (a prefixed basename
                # would, up to ENAMETOOLONG), never collides.
                ext = os.path.splitext(name)[1][:16]
                name = f"{pos:02d}_{uuid.uuid4().hex[:12]}{ext}"
            dst = os.path.join(dest_dir, name)
            shutil.copy2(real, dst)
            out.append(dst)
        else:
            # best-effort: keep whatever it was — serving 404s it and submit
            # fails the copy cleanly naming the slot.
            out.append(ref)
    return out


def _build_copied_run(
    src: Run,
    db: Session,
    *,
    resolution: Optional[str],
    name: Optional[str],
    ref_files: list,
    ref_urls: list,
    preprocess: bool = False,
) -> Run:
    """Build (add + flush, but do NOT commit or enqueue) a new Run cloned from *src*.

    Clones everything that defines the result — prompt, model, audio mode, Drive
    folder, and any per-segment prompt overrides — and applies the things a copy
    may change: *resolution*, the *reference photo*, and whether that photo is
    re-run through GPT Image 2. Returns the new Run; the caller is responsible
    for committing and enqueueing it — and for reading ``new_run.status`` to
    pick the entry point, because a preprocessing copy starts on the GPT Image 2
    chain, not the submit chain (see ``_enqueue_copied_run``).

    Shared by the single-run copy, the batch copy, and automation launches so the
    cloning rules stay in one place.

    * **resolution** — defaults to the source run's resolution if omitted; pass a
      different one to promote a 480p test to production.
    * **reference photo** — pass *ref_files* (UploadFile list) and/or *ref_urls*
      (list of url strings) to swap the character to a new face of the same type.
      When new references are supplied they REPLACE the photo everywhere — both
      the run-level references AND any per-segment reference overrides are dropped
      so every swap segment uses the new photo (per-segment *prompt* tweaks are
      still carried over). When omitted, the source run's references (run-level
      and per-segment) are cloned unchanged.
    * **preprocess** — re-run *src*'s GPT Image 2 phase on the new photo instead
      of handing that photo to Seedance raw. The copy inherits the source's
      FRAMES (``reference_image_urls[1:]``), its ``preprocess_mode`` and its
      image prompts verbatim; only the character reference at index 0 becomes
      the new photo. The source's own generated images are deliberately NOT
      cloned — they are the old face. Requires a preprocessing source run and
      exactly one new photo (400 otherwise), and the copy auto-approves its
      results: the recipe was approved by hand on the source, only the face
      changed. Without this flag, without a new photo, and when *src* itself
      has ``preprocess_enabled`` set (i.e. its own generated images are known
      to match its own face — true for a genuine preprocessing run and for a
      copy that took this same path; NOT true for a pre-fix copy where a new
      photo was supplied without ``preprocess``, since those historically
      inherited the source's generated images regardless of the new photo),
      the copy carries the WHOLE recipe forward: the source's own generated
      GPT Image 2 images AND its ``preprocess_image_prompts`` are cloned
      (``preprocess_mode`` is already cloned unconditionally above), and
      ``preprocess_enabled`` is set on the copy so submit uses the generated
      images — the copy reproduces the approved result rather than
      re-submitting the raw original references, and is itself a valid
      source for a further ``preprocess=true`` copy or automation launch. A
      new photo with no ``preprocess`` flag carries none of the recipe at
      all: the raw new photo goes to Seedance directly.
    """
    # 2026-11-03 (design R2-1): a VOID-from-the-source run is never copied --
    # a copy would be a v2v run that sends every segment to the paid model.
    if project_types.is_void_source_run(src):
        raise HTTPException(status_code=409, detail=(
            "this run removed captions with VOID from the source; it cannot be copied "
            "(a copy would send every segment to the video model). Create a new run."))
    # Resolution: default to the source run's when not changing it.
    resolution = (resolution or src.resolution).strip()
    allowed = ai_models.resolutions_for(src.model)
    if resolution not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"resolution {resolution!r} not allowed for model {src.model!r}; "
                f"choose one of {sorted(allowed)}"
            ),
        )

    # New reference photo (optional). When provided it replaces the photo
    # everywhere; when absent we clone the source run's references.
    new_ref_files = ref_files or []
    new_ref_urls = ref_urls or []
    has_new_refs = bool(new_ref_files or new_ref_urls)
    if has_new_refs:
        total_refs = len(new_ref_files) + len(new_ref_urls)
        if total_refs > settings.MAX_REFERENCE_IMAGES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Too many reference images: got {total_refs}, "
                    f"max {settings.MAX_REFERENCE_IMAGES}"
                ),
            )
        # AICP-148: a copy's new photo is an upload or a URL — covers the
        # single copy, the batch copy and automation launches alike.
        _reject_bad_reference_urls(new_ref_urls, field="reference_urls")

    # Re-run GPT Image 2 on the new photo. Validated up front, before anything
    # is written, so a bad spec fails the whole batch rather than leaving half
    # of it queued (same contract as the resolution check above).
    if preprocess:
        if not src.preprocess_enabled:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot re-run GPT Image: the source run was not created "
                    "with preprocessing, so there are no frames or image prompt "
                    "to reuse. Copy it without this option, or start a new run "
                    "with preprocessing from the project page."
                ),
            )
        if len(list(src.reference_image_urls or [])) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot re-run GPT Image: the source run has no frames to "
                    "reuse (its reference list holds only the character photo)."
                ),
            )
        if not list(src.preprocess_image_prompts or []):
            # Belt, not the fix (see the Part 1 clone above, which keeps this
            # from ever firing on a run built by THIS function). A genuinely
            # created preprocessing run always has a non-empty prompts list —
            # even an all-pass-through one is stored as [""] — so an empty/
            # None list here means preprocess_enabled is True with nothing for
            # GPT Image 2 to actually reuse. Catching it here makes the
            # `preprocess_enabled` check above's own stated premise ("there is
            # no image prompt to reuse") true again for any future path that
            # produces the same asymmetry.
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot re-run GPT Image: the source run has no image "
                    "prompt to reuse. Copy it without this option, or start a "
                    "new run with preprocessing from the project page."
                ),
            )
        # Exactly one new photo: it becomes reference_image_urls[0], the
        # character reference. A second photo has nowhere to go — the frames at
        # [1:] come from the source run — so guessing would silently drop it.
        if len(new_ref_files) + len(new_ref_urls) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Re-running GPT Image needs exactly one new reference "
                    "photo (it replaces the character photo; the frames come "
                    f"from the source run). Got "
                    f"{len(new_ref_files) + len(new_ref_urls)}."
                ),
            )

    # Re-validate the cloned stitch plan against CURRENT state: segments may
    # have been re-cut and library clips deleted since the source run was
    # created (terminal runs don't block clip deletion). Failing the copy now
    # beats submitting swap segments (credits spent) and dying at stitch time —
    # this also covers automation launches, which clone via this helper.
    cloned_plan = list(src.stitch_plan) if src.stitch_plan is not None else None
    if cloned_plan:
        current_seg_ids = {
            s.id
            for s in db.execute(
                select(SegmentDef).where(SegmentDef.project_id == src.project_id)
            ).scalars()
        }
        for pos, entry in enumerate(cloned_plan):
            kind = entry.get("kind") if isinstance(entry, dict) else None
            if kind == "segment" and entry.get("segment_def_id") not in current_seg_ids:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Cannot copy: stitch plan entry {pos} references a "
                        "segment that no longer exists on the project (segments "
                        "were re-cut). Create a new run with a fresh plan."
                    ),
                )
            if kind == "library" and db.get(LibraryClip, entry.get("clip_id")) is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Cannot copy: stitch plan entry {pos} references a "
                        "library clip that was deleted. Create a new run with "
                        "a fresh plan."
                    ),
                )

    # Same launch gate as create_run, against whichever timeline THIS copy will
    # actually run: a copy inherits the source's frozen snapshot when it has one
    # (including any promoted range correction, which can itself have changed a
    # segment's length), and only falls back to the project's live partition for
    # a legacy source that predates snapshots. Applied here rather than in each
    # caller so the single-run copy, the batch copy and automation launches are
    # all covered — an automation is the case that most needs it, since it
    # launches unattended and a silently half-swapped delivery has nobody
    # watching to catch it.
    _assert_no_overlong_swap_segments(
        src.segment_snapshot if src.segment_snapshot else src.project.segments,
        src.model,
        what="copy run",
        # The plan the COPY will run under — already cloned and re-validated
        # against current DB state above.
        plan=cloned_plan,
    )

    new_id = str(uuid.uuid4())
    if name and name.strip():
        new_name = name.strip()
    else:
        suffix = "new ref" if has_new_refs else resolution
        new_name = f"{src.name or 'run'} · {suffix}"
    new_name = new_name[:255]
    new_run = Run(
        id=new_id,
        project_id=src.project_id,
        name=new_name,
        prompt=src.prompt,
        model=src.model,
        resolution=resolution,
        audio_mode=src.audio_mode,
        # A copy is the same creative in the same language — carried over so
        # Copy-run batches and sheet-driven automation launches keep the
        # template's Job ID suffix without the operator re-picking it.
        language=src.language,
        gdrive_folder_id=src.gdrive_folder_id,
        # Clone the stitch plan (outro replacement etc.) — it defines the
        # result just like the prompt does. Validated against current DB
        # state above.
        stitch_plan=cloned_plan,
        # Deep-copy the source's FROZEN run-local timeline — including any
        # promoted range-regeneration correction — rather than leaving this
        # None. A copy with segment_snapshot=None re-freezes whatever the
        # PROJECT's live segments are at its own first submit
        # (pipeline_v2._build_submit_state), which is exactly the bug this
        # closes: before this line, a Copy of a corrected run silently lost
        # the correction and went back to the shared Project boundaries.
        # None only for a source run that itself predates run-local
        # snapshots (legacy fallback; unchanged pre-existing behaviour).
        segment_snapshot=(
            copy.deepcopy(src.segment_snapshot) if src.segment_snapshot else None
        ),
        status=RunStatus.created,
        reference_image_urls=[],
        preprocess_mode=src.preprocess_mode,
        # Only a *preprocess* copy re-runs the GPT Image 2 phase; a plain copy
        # keeps the historical behaviour (refs go to Seedance as they are) —
        # unless the source's generated images get cloned below, in which case
        # this is flipped to True so submit_run actually reads them.
        preprocess_enabled=preprocess,
        # The frames and the image prompt are the source's, and an operator
        # already approved that pairing when they approved the source run —
        # so the copy has nothing new to show and submits straight through.
        preprocess_auto_approve=preprocess,
        # Prompts are positional against reference_image_urls, and the copy
        # keeps that geometry exactly (index 0 = character photo, [1:] = the
        # source's frames), so the source's list transfers verbatim.
        preprocess_image_prompts=(
            list(src.preprocess_image_prompts or []) if preprocess else None
        ),
    )
    db.add(new_run)
    db.flush()

    refs_dir = os.path.join(project_dir(src.project_id), "runs", new_id, "references")
    if has_new_refs:
        # Use the supplied photo as the new run-level reference.
        _max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
        saved_ref_paths: list[str] = []
        if new_ref_files:
            os.makedirs(refs_dir, exist_ok=True)
            for rf in new_ref_files:
                safe = _safe_filename(rf.filename or f"ref_{len(saved_ref_paths)}.jpg")
                dest = os.path.join(refs_dir, safe)
                _save_upload(rf, dest, max_bytes=_max_bytes)
                saved_ref_paths.append(dest)
        if preprocess:
            # R1-new layout: [character photo] + the source's frames. Validated
            # above to be exactly one new photo, so this list is unambiguous.
            # The frames are COPIED into the new run's own dir like every other
            # reference — deleting the source run must not strand the copy.
            new_run.reference_image_urls = (saved_ref_paths + new_ref_urls) + (
                _copy_reference_files(
                    list(src.reference_image_urls or [])[1:], refs_dir
                )
            )
        else:
            new_run.reference_image_urls = saved_ref_paths + new_ref_urls
    else:
        # Clone run-level reference images into the new run's own dir.
        new_run.reference_image_urls = _copy_reference_files(
            list(src.reference_image_urls or []), refs_dir
        )

    # Clone the voice reference. It defines the result exactly as the prompt and
    # stitch plan do — and more sharply than either, since the whole point of
    # attaching one is that two runs of the same character SOUND the same. A
    # copy that dropped it would come back in a different voice, which is the
    # bug this feature exists to fix. Copied into the new run's own dir like
    # every other reference, so deleting the source run cannot silence it.
    #
    # Unaffected by *has_new_refs*: a new reference PHOTO changes the face, not
    # the voice, so "same voice, new face" stays expressible.
    if src.reference_audio_urls:
        new_run.reference_audio_urls = _copy_reference_files(
            list(src.reference_audio_urls),
            run_voice_dir(new_id, src.project_id),
        )

    # If the source run already went through GPT Image 2 preprocessing
    # (preprocessed_reference_urls populated), clone those preprocessed images
    # too so the copy uses the same preprocessed refs without re-running the
    # preprocessing phase (saves credits + time). preprocess_enabled is
    # flipped to True here — submit_run only reads preprocessed_reference_urls
    # when the flag is set (pipeline_v2.py:1673), so leaving it False would
    # silently submit the raw originals instead of the approved images.
    #
    # NOT when a new photo is supplied (has_new_refs): those cloned images are
    # the OLD face. Cloning them would hand Seedance the old character and
    # make the new photo inert — the exact bug the *preprocess* option exists
    # to fix. The copy carries no preprocessed images at all in that case and
    # either submits the new photo raw or re-runs GPT Image 2 on it.
    #
    # ALSO requires src.preprocess_enabled: before this fix, a copy made WITH
    # a new photo (has_new_refs) cloned the source's preprocessed images
    # unconditionally (no has_new_refs guard existed) while leaving
    # preprocess_enabled False — so a pre-fix row can have
    # preprocessed_reference_urls holding a face that does NOT match its own
    # reference_image_urls, sitting inert only because the flag was False.
    # src.preprocess_enabled is the one invariant that says "this run's own
    # pp images actually match its own face" (true for a genuine preprocessing
    # run, and true for a copy that took THIS branch) — without it, copying
    # such a contaminated legacy row (with no new photo) would clone the
    # mismatched images forward and flip the flag that makes submit_run read
    # them, reviving a dormant wrong-face bug instead of fixing one.
    if (
        src.preprocess_enabled
        and src.preprocessed_reference_urls
        and not preprocess
        and not has_new_refs
    ):
        pp_dir = os.path.join(
            project_dir(src.project_id), "runs", new_id, "preprocessed"
        )
        new_run.preprocessed_reference_urls = _copy_reference_files(
            list(src.preprocessed_reference_urls), pp_dir
        )
        new_run.preprocess_enabled = True
        # preprocess_mode is already cloned unconditionally above; the image
        # prompts complete the recipe. reference_image_urls is a verbatim
        # clone in this branch (not has_new_refs), so the prompts' positional
        # geometry (index 0 = "", [1:] = trim prompts) still lines up. Without
        # this the copy would be preprocess_enabled=True with no prompts —
        # exactly the shape that makes a *further* preprocess=true copy of it
        # pass the "was created with preprocessing" guard below while having
        # nothing for GPT Image 2 to actually do.
        new_run.preprocess_image_prompts = list(src.preprocess_image_prompts or [])

    # Clone per-segment overrides so the copy reproduces the same tuned result.
    # process_run is idempotent: it reuses these RunSegments (by segment_def_id)
    # instead of creating fresh ones, so the overrides take effect.
    #
    # When a new photo is supplied we DROP per-segment reference overrides so the
    # new run-level photo is used in every segment (the "replace everywhere"
    # behaviour); per-segment prompt tweaks are still carried over.
    # Swap segments the cloned plan excludes are never processed — don't clone
    # their RunSegments (a pending row that never submits would make the copy
    # look stuck and defeat the delivery-retry fast path).
    active_seg_ids = _plan_active_segment_ids(cloned_plan)
    for src_rs in src.run_segments:
        if active_seg_ids is not None and src_rs.segment_def_id not in active_seg_ids:
            continue
        if has_new_refs:
            if not src_rs.prompt_override:
                # Only a photo override here → dropped; process_run will create a
                # fresh pending segment that inherits the new run-level photo.
                continue
            override_refs = None
        else:
            if not (src_rs.prompt_override or src_rs.reference_image_urls_override):
                continue
            override_refs = None
            if src_rs.reference_image_urls_override:
                new_rs_id_refs = str(uuid.uuid4())
                seg_dir = os.path.join(
                    project_dir(src.project_id), "runs", new_id, "segment_refs",
                    new_rs_id_refs,
                )
                override_refs = _copy_reference_files(
                    list(src_rs.reference_image_urls_override), seg_dir
                )
        # generated_reference_urls / generated_image_job_id are deliberately
        # NOT carried over (and no RunSegmentImageJob row is cloned either): a
        # generated still is a per-segment rescue of one specific take, cut
        # from THIS run's frozen snapshot. Cloning it onto a copy — especially
        # one with a new photo — would pin the OLD face into the reference
        # priority chain ahead of the run-level set, the same class of bug the
        # copy-preprocess-refs fix closed for the run level. A copy whose only
        # per-segment state was a generated image (no prompt_override, no
        # reference_image_urls_override) hits the `continue` above and gets no
        # cloned row at all — submit_run creates a plain pending one for it.
        db.add(
            RunSegment(
                id=str(uuid.uuid4()),
                run_id=new_id,
                segment_def_id=src_rs.segment_def_id,
                index=src_rs.index,
                status=SegmentStatus.pending,
                prompt_override=src_rs.prompt_override,
                reference_image_urls_override=override_refs,
            )
        )

    try:
        # A preprocess copy enters the GPT Image 2 chain first (created →
        # preprocessing), exactly as create_run does for a preprocessing run;
        # everything else goes straight to the submit chain. The caller reads
        # the resulting status to pick the matching enqueue — see
        # _enqueue_copied_run.
        transition(
            new_run,
            RunStatus.preprocessing if preprocess else RunStatus.queued,
        )
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return new_run


def _enqueue_copied_run(run: Run) -> None:
    """Enqueue a run built by ``_build_copied_run`` at the right entry point.

    A preprocess copy is left in ``preprocessing`` and must start on the GPT
    Image 2 chain; anything else is ``queued`` and starts at submit. Getting
    this wrong is silent and expensive — ``run_submit_run`` on a preprocessing
    run would send the UNPROCESSED refs to Seedance and spend video credits on
    the wrong images.
    """
    if run.status == RunStatus.preprocessing:
        enqueue_preprocess_run(run.id)
    else:
        enqueue_process_run(run.id)


def _check_copyable(src: Run, project: VideoProject) -> None:
    """Raise 409 when *src* is a run a copy cannot do anything useful with.

    Today that is exactly one case: a **speech-only localisation** run (the New
    Run form's "речь только" mode). Copy run exists to re-run tuned settings on
    a NEW face — but that prompt tells the model to keep the person on screen,
    so the photo a copy attaches is inert and the copy is a byte-identical
    re-submission that spends credits for the same clip. Copying stays
    available on "speech + character" runs, where a new photo is the whole
    point. See docs/localisation.md §7.

    The mode is not a stored column (§3.3) — ``localisation_mode_of`` reads it
    back off the prompt, and answers "swap" for anything it cannot recognise,
    so an operator's hand-written prompt never loses the feature.

    Automation gates on the same predicate — see ``put_run_automation`` and
    ``automation._launch_batch``. A daily quota of identical re-generations is
    the same mistake as one copy, only repeated.
    """
    if run_copies_are_useful(project.project_type, src.prompt):
        return
    raise HTTPException(
        status_code=409,
        detail=(
            "Cannot copy run: this is a speech-only localisation run, so a copy "
            "would re-submit the same prompt and the new reference photo would "
            "have no effect. Copy run is available on 'speech + character' runs; "
            "otherwise start a new run from the project page."
        ),
    )


@router.post(
    "/runs/{rid}/copy",
    status_code=status.HTTP_201_CREATED,
    response_model=RunCreateResponse,
)
def copy_run(
    rid: str,
    resolution: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    preprocess: bool = Form(False),
    db: Session = Depends(get_db),
    adm: Optional[Admission] = ADMISSION,
) -> RunCreateResponse:
    """Duplicate a run — optionally at a new resolution and/or with a new
    reference photo — and enqueue it. See ``_build_copied_run`` for the cloning
    rules. To launch several copies (each with its own photo/quality/name) in one
    request, use ``/runs/{rid}/copy-batch``.

    ``preprocess=true`` re-runs the source's GPT Image 2 phase on the new photo
    (same frames, same image prompt) and auto-approves the result.

    AICP-350: accepts ``Idempotency-Key`` (URL references only — a keyed
    request with file parts is refused); a replay returns the recorded copy.
    """
    adm = ops_gate.keyed(adm)  # gate already held by the ADMISSION dependency
    if adm is not None and adm.replay is not None:
        return ops_gate.replay_one(adm, Run, lambda r: RunCreateResponse(
            run_id=r.id, status=_status_text(r.status)))
    src = _get_run_or_404(rid, db)
    _refuse_images_only_run(src, "Copy run")
    project = _get_project_or_404(src.project_id, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot copy run: project status is {project.status!r}, expected 'ready'",
        )
    _check_copyable(src, project)

    # A promotion in flight may replace src.segment_snapshot (and its result)
    # at any moment — cloning right now risks the copy freezing whichever
    # side of that race it lands on. 409 rather than silently cloning the
    # PRE-correction range.
    active_attempt = _active_range_attempt(rid, db)
    if active_attempt is not None:
        raise _range_attempt_conflict(active_attempt)

    _budget_precheck(db, "copy run")
    new_ref_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    new_run = _build_copied_run(
        src, db,
        resolution=resolution,
        name=name,
        ref_files=reference_files or [],
        ref_urls=new_ref_urls,
        preprocess=preprocess,
    )
    new_id = new_run.id

    if adm is not None:
        adm.record("run", [new_id])
    _spend_reserve(spend_mod.reserve_new_runs, db, [new_run], kind="copy run")
    db.commit()
    _enqueue_copied_run(new_run)

    log.info(
        "Copied run %s → %s (resolution=%s, preprocess=%s)",
        rid, new_id, new_run.resolution, preprocess,
    )
    status_val = new_run.status.value if hasattr(new_run.status, "value") else str(new_run.status)
    return RunCreateResponse(run_id=new_id, status=status_val)


# runs[<idx>][<field>] — multipart keys for one batch-copy spec. reference_files
# may repeat (several photos per run); the rest appear at most once per index.
_BATCH_KEY_RE = re.compile(
    r"^runs\[(\d+)\]\[(name|resolution|reference_urls|reference_files|preprocess)\]$"
)

# Form values the UI sends for a ticked checkbox. Anything else (including the
# absent key) reads as false, so an untouched row never spends image credits.
_TRUTHY_FORM_VALUES = {"1", "true", "on", "yes"}


def _parse_batch_specs(form) -> list[dict]:
    """Group a multipart form's ``runs[i][...]`` fields into a list of per-run
    specs, ordered by index. Unknown keys are ignored; empty file slots (a file
    input submitted with no chosen file) are dropped.
    """
    by_index: dict[int, dict] = {}
    for key, value in form.multi_items():
        m = _BATCH_KEY_RE.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        field = m.group(2)
        spec = by_index.setdefault(
            idx,
            {
                "name": None,
                "resolution": None,
                "ref_urls": [],
                "ref_files": [],
                "preprocess": False,
            },
        )
        if field == "reference_files":
            # A multipart UploadFile (has .filename); browsers also send an empty
            # part when no file is chosen — skip those.
            if getattr(value, "filename", None):
                spec["ref_files"].append(value)
        elif field == "reference_urls":
            spec["ref_urls"].extend(
                u.strip() for u in str(value).split(",") if u.strip()
            )
        elif field == "name":
            spec["name"] = str(value).strip() or None
        elif field == "resolution":
            spec["resolution"] = str(value).strip() or None
        elif field == "preprocess":
            spec["preprocess"] = str(value).strip().lower() in _TRUTHY_FORM_VALUES
    return [by_index[i] for i in sorted(by_index)]


@router.post(
    "/runs/{rid}/copy-batch",
    status_code=status.HTTP_201_CREATED,
    response_model=RunBatchCopyResponse,
    dependencies=[GATED],
)
async def copy_run_batch(
    rid: str,
    request: Request,
    db: Session = Depends(get_db),
) -> RunBatchCopyResponse:
    """Launch up to 10 copies of a run in one request — each with its own
    reference photo(s), resolution (quality), and name.

    The body is multipart with one group of fields per run, indexed from 0:

    * ``runs[i][resolution]`` — quality for copy *i* (defaults to the source run)
    * ``runs[i][name]`` — optional name for copy *i*
    * ``runs[i][reference_urls]`` — optional comma-separated reference photo URLs
    * ``runs[i][reference_files]`` — optional uploaded reference photo(s); repeat
      the field for several photos
    * ``runs[i][preprocess]`` — optional flag: re-run the source's GPT Image 2
      phase on this copy's photo (same frames, same image prompt) and
      auto-approve the result. Needs exactly one photo for that copy. Rows are
      independent, so a batch may mix preprocessed and raw copies.

    All copies are validated and built in a single transaction: if any spec is
    invalid (bad resolution, too many photos), nothing is created. On success
    every new run is enqueued and returned in submission order.

    Only the multipart parse happens on the event loop; the blocking work
    (_save_upload file writes, shutil.copy2, DB commit) runs in the threadpool
    via run_in_threadpool so a 10-run batch copy can't stall other requests.
    """
    form = await request.form()
    return await run_in_threadpool(_copy_run_batch_sync, rid, form, db)


def _copy_run_batch_sync(rid: str, form, db: Session) -> RunBatchCopyResponse:
    src = _get_run_or_404(rid, db)
    _refuse_images_only_run(src, "Copy run")
    project = _get_project_or_404(src.project_id, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot copy run: project status is {project.status!r}, expected 'ready'",
        )
    _check_copyable(src, project)

    active_attempt = _active_range_attempt(rid, db)
    if active_attempt is not None:
        raise _range_attempt_conflict(active_attempt)

    specs = _parse_batch_specs(form)
    if not specs:
        raise HTTPException(status_code=400, detail="No runs specified")
    if len(specs) > _MAX_BATCH_COPY_RUNS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many runs: got {len(specs)}, max {_MAX_BATCH_COPY_RUNS}",
        )

    _budget_precheck(db, "batch copy")

    # Build every run first (all validated, added + flushed) before committing,
    # so a bad spec fails the whole batch instead of leaving half of it queued.
    # The budget is reserved for ALL of them only after the last build, so the
    # day lock is not held across the builds' file copies; an over-budget batch
    # is refused whole (429), never half-launched.
    new_runs: list[Run] = []
    for spec in specs:
        new_runs.append(
            _build_copied_run(
                src, db,
                resolution=spec["resolution"],
                name=spec["name"],
                ref_files=spec["ref_files"],
                ref_urls=spec["ref_urls"],
                preprocess=spec["preprocess"],
            )
        )

    _spend_reserve(spend_mod.reserve_new_runs, db, new_runs, kind="batch copy")
    db.commit()
    for nr in new_runs:
        _enqueue_copied_run(nr)

    log.info("Batch-copied run %s → %d new runs %s", rid, len(new_runs), [r.id for r in new_runs])
    return RunBatchCopyResponse(
        runs=[
            RunCreateResponse(
                run_id=nr.id,
                status=nr.status.value if hasattr(nr.status, "value") else str(nr.status),
            )
            for nr in new_runs
        ]
    )


# ---------------------------------------------------------------------------
# Run automation — GET/PUT /runs/{rid}/automation
# ---------------------------------------------------------------------------

# UI guard for daily_count; keeps a typo from launching hundreds of paid runs.
_AUTOMATION_MAX_DAILY = 100


def _automation_response(
    run_id: str,
    automation: Optional[RunAutomation],
    db: Session,
    *,
    preprocess_available: bool = False,
) -> AutomationResponse:
    if automation is None:
        return AutomationResponse(
            run_id=run_id, preprocess_available=preprocess_available
        )

    from .automation import _TERMINAL_RUN_STATUSES, launched_today

    launches = (
        db.execute(
            select(AutomationLaunch).where(
                AutomationLaunch.automation_id == automation.id
            )
        )
        .scalars()
        .all()
    )
    pending_run_ids = [
        l.new_run_id for l in launches if l.notified_at is None and l.new_run_id
    ]
    in_flight = 0
    if pending_run_ids:
        pending_runs = (
            db.execute(select(Run).where(Run.id.in_(pending_run_ids)))
            .scalars()
            .all()
        )
        in_flight = sum(
            1 for r in pending_runs if r.status not in _TERMINAL_RUN_STATUSES
        )
    return AutomationResponse(
        run_id=run_id,
        configured=True,
        enabled=automation.enabled,
        filter_persona=automation.filter_persona,
        filter_age=automation.filter_age,
        filter_gender=automation.filter_gender,
        filter_ethnicity=automation.filter_ethnicity,
        exclude_persona=automation.exclude_persona,
        exclude_gender=automation.exclude_gender,
        exclude_ethnicity=automation.exclude_ethnicity,
        plan=automation.plan,
        daily_count=automation.daily_count,
        webhook_url=automation.webhook_url,
        preprocess_refs=automation.preprocess_refs,
        preprocess_available=preprocess_available,
        launched_today=launched_today(
            db, automation.id, datetime.now(timezone.utc)
        ),
        in_flight=in_flight,
        used_total=len(launches),
    )


@router.get("/runs/{rid}/automation", response_model=AutomationResponse)
def get_run_automation(rid: str, db: Session = Depends(get_db)) -> AutomationResponse:
    """Automation settings + counters for a run (defaults when not configured)."""
    run = _get_run_or_404(rid, db)
    automation = db.execute(
        select(RunAutomation).where(RunAutomation.run_id == rid)
    ).scalar_one_or_none()
    return _automation_response(
        rid, automation, db, preprocess_available=run.preprocess_enabled
    )


@router.put("/runs/{rid}/automation", response_model=AutomationResponse)
def put_run_automation(
    rid: str,
    body: AutomationSettingsUpdate,
    db: Session = Depends(get_db),
) -> AutomationResponse:
    """Create or update the run's automation config.

    The scheduled tick (app/automation.py) picks the change up on its next
    pass — enabling here launches nothing synchronously.
    """
    from .automation import ALLOWED_PLANS

    run = _get_run_or_404(rid, db)

    # Automation is Copy run on a daily quota — same premise, so the same
    # refusal (docs/localisation.md §7). Only ENABLING is blocked: an
    # automation switched on before this rule existed must stay switchable-off
    # through the same endpoint.
    if body.enabled:
        _refuse_images_only_run(run, "an automation")
        project = _get_project_or_404(run.project_id, db)
        if not run_copies_are_useful(project.project_type, run.prompt):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot enable automation: this is a speech-only "
                    "localisation run, so every launched copy would re-submit "
                    "the same prompt and the sheet's reference photo would have "
                    "no effect. Automate a 'speech + character' run instead."
                ),
            )

    plan = (body.plan or "").strip()
    if plan not in ALLOWED_PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"plan must be one of {list(ALLOWED_PLANS)}, got {plan!r}",
        )
    if not (1 <= body.daily_count <= _AUTOMATION_MAX_DAILY):
        raise HTTPException(
            status_code=400,
            detail=f"daily_count must be between 1 and {_AUTOMATION_MAX_DAILY}",
        )
    webhook_url = (body.webhook_url or "").strip() or None
    if webhook_url:
        from . import webhook_safety

        try:
            webhook_safety.validate_webhook_url(webhook_url)
        except webhook_safety.WebhookRefused as exc:
            raise HTTPException(status_code=400, detail=f"webhook_url: {exc}")

    # Re-running GPT Image 2 per launch needs a template that HAS a recipe to
    # re-run (frames + image prompt). Refuse rather than silently ignore: an
    # operator who ticks this and gets raw photos in Seedance would only find
    # out from the delivered videos.
    if body.preprocess_refs and not run.preprocess_enabled:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot re-run GPT Image per launch: this run was not created "
                "with preprocessing, so there are no frames or image prompt to "
                "reuse. Automate a run that was."
            ),
        )

    def _clean(v: Optional[str]) -> Optional[str]:
        v = (v or "").strip()
        return v or None

    automation = db.execute(
        select(RunAutomation).where(RunAutomation.run_id == rid)
    ).scalar_one_or_none()
    if automation is None:
        automation = RunAutomation(run_id=rid)
        db.add(automation)

    automation.enabled = body.enabled
    automation.filter_persona = _clean(body.filter_persona)
    automation.filter_age = _clean(body.filter_age)
    automation.filter_gender = _clean(body.filter_gender)
    automation.filter_ethnicity = _clean(body.filter_ethnicity)
    automation.exclude_persona = body.exclude_persona
    automation.exclude_gender = body.exclude_gender
    automation.exclude_ethnicity = body.exclude_ethnicity
    automation.plan = plan
    automation.daily_count = body.daily_count
    automation.webhook_url = webhook_url
    automation.preprocess_refs = body.preprocess_refs
    db.commit()

    log.info(
        "Automation for run %s: enabled=%s plan=%s daily=%d preprocess_refs=%s "
        "filters=%s",
        rid, automation.enabled, automation.plan, automation.daily_count,
        automation.preprocess_refs,
        {
            "persona": automation.filter_persona,
            "age": automation.filter_age,
            "gender": automation.filter_gender,
            "ethnicity": automation.filter_ethnicity,
        },
    )
    return _automation_response(
        rid, automation, db, preprocess_available=run.preprocess_enabled
    )
