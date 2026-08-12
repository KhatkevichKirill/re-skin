"""
v2 REST API router for re-skin.

Mounted at /api/v2 from app/main.py.

Adds VideoProject (video + segmentation) and Run (one character per project)
endpoints. v1 /api/jobs endpoints are left completely untouched.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import uuid
from typing import Any, List, Optional

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
from fastapi.responses import FileResponse, Response
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from . import ai_models
from . import languages
from . import localisation
from .config import settings
from .db import get_db
from .models import Run, RunSegment, SegmentDef, VideoProject
from .project_types import (
    DEFAULT_PROJECT_TYPE,
    LOCALISATION,
    VALID_PROJECT_TYPES,
    spec_for,
)
from .schemas_v2 import (
    NewSegmentDef,
    ProjectCreateResponse,
    ProjectListItem,
    ProjectResponse,
    ProjectUpdate,
    RunBatchCopyResponse,
    RunCreateResponse,
    RunListItem,
    RunResponse,
    RunSegmentResponse,
    SegmentDefResponse,
    SegmentsUpdateRequest,
    LocalisationPromptResponse,
    TranscriptResponse,
)
from .state_machine import InvalidTransition, ProjectStatus, RunStatus, SegmentStatus, transition
from .storage import project_dir, project_source_path, run_dir
from .tasks import (
    enqueue_analyze_project,
    enqueue_process_run,
    enqueue_transcribe_project,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["v2"])

_VALID_RESOLUTIONS = {"480p", "720p", "1080p", "4k"}
_VALID_AUDIO_MODES = {"original", "seedance"}
# Run states that mean a worker may still be touching the run's files — block
# deletion while in any of these (delete would race the worker / rmtree live files).
_ACTIVE_RUN_STATUSES = {
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


def _validate_project_type(project_type: str) -> None:
    """400 unless *project_type* is one of the registered flows."""
    if project_type not in VALID_PROJECT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"project_type must be one of {sorted(VALID_PROJECT_TYPES)}",
        )


# Floor for the per-project segmentation cap. NOT ai_models.min_clip_sec, which
# bounds the shortest *reference clip* a model accepts as INPUT. What bounds a
# segmentation cap is the shortest video a model will EMIT: every spec carries a
# min_duration_sec and ai_models.duration_for() clamps UPWARD to it, so a cap
# below the largest min_duration_sec in the table cannot be honoured by any
# backend. Accepting one silently multiplies spend — at cap=2.0 a 9s face
# interval is cut into five 1.8s chunks and each is still submitted asking for 4s
# of output, i.e. we generate and pay for 20s of video to cover 9s of source.
#
# Computed rather than written as 4.0 so adding a model with a different floor
# moves the bound with it.
_MIN_SEGMENT_CAP_SEC = float(
    max(spec.min_duration_sec for spec in ai_models.AI_MODELS.values())
)


def _max_segment_sec_in_range(value: float) -> bool:
    """True when *value* is a segmentation cap some backend can honour.

    Written as a chained comparison, not ``value < lo or value > hi``. Every
    comparison against NaN is False, so both halves of the two-negation form
    would be False and it would ACCEPT NaN — and NaN really does reach here:
    Pydantic v2 parses ``{"max_segment_sec": NaN}`` and the string ``"nan"`` into
    a float NaN by default, and Postgres DOUBLE PRECISION stores it happily, so
    it persists. Analysis then computes ``min(nan, ABSOLUTE_MAX_SEGMENT_SEC)``
    (NaN wins), hands it to the segmenter, and dies in ``math.ceil(dur / nan)``.
    Framing the test as "is it inside the interval" rejects NaN and both
    infinities for free.
    """
    return _MIN_SEGMENT_CAP_SEC <= value <= ai_models.ABSOLUTE_MAX_SEGMENT_SEC


def _validate_max_segment_sec(value: Optional[float]) -> Optional[float]:
    """Validate the per-project analyze-time segmentation cap, or 400.

    ``None`` is a real, meaningful value here — it means "no explicit choice",
    and analysis falls back to ai_models.UNIVERSAL_MAX_SEGMENT_SEC (the shortest
    max_clip_sec across every model, i.e. the cap that keeps a project runnable
    on ANY backend). So we return it untouched rather than coercing it to a
    number: storing the default explicitly would freeze today's value into every
    row and silently diverge if the registry ever changes.

    The accepted band is [_MIN_SEGMENT_CAP_SEC, ABSOLUTE_MAX_SEGMENT_SEC]: below
    the floor no backend can generate a clip that short, so the request is padded
    back up and the difference is billed, and above the longest-clip model's
    ceiling every segment would be un-generatable.
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
                "will generate, so a cap below it is still billed as "
                f"{lo}s per segment however short the cut, and a cap above "
                f"{hi}s can be generated by no model at all — omit it to use "
                f"the {ai_models.UNIVERSAL_MAX_SEGMENT_SEC}s default that every "
                "model can generate"
            ),
        )
    return float(value)


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
    one of app/languages.py's five codes. An unknown code is refused rather than
    stored: on a localisation run this value IS the translation target handed to
    the LLM, so an unrecognised code would silently produce a hook in some other
    language than the one recorded.
    """
    normalized = languages.normalize(code)
    if (code or "").strip() and normalized is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"language must be one of {sorted(languages.LANGUAGES)}"
            ),
        )
    return normalized


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


@router.post(
    "/projects",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectCreateResponse,
)
def create_project(
    video_file: Optional[UploadFile] = File(None),
    gdrive_link: Optional[str] = Form(None),
    project_type: str = Form(DEFAULT_PROJECT_TYPE),
    mute_source: bool = Form(False),
    max_segment_sec: Optional[float] = Form(None),
    hook_sec: Optional[float] = Form(None),
    db: Session = Depends(get_db),
) -> ProjectCreateResponse:
    """Create a new VideoProject.

    Exactly one of *video_file* or *gdrive_link* must be provided.
    Analysis is enqueued immediately (ffprobe + segment proposal).

    *project_type* selects the flow (see app/project_types.py): ``face_swap``
    (default) detects faces and proposes swap/keep segments, ``subtitle_removal``
    cuts on scene changes, ``localisation`` splits the hook off the front, and
    ``cover_change`` seeds one full-length keep segment for manual editing.

    *mute_source* ("remove original audio") makes every run of this project
    upload video-only swap clips to the AI model, so it generates audio from
    the prompt instead of reacting to the source soundtrack. Toggleable later
    via ``PATCH /projects/{pid}``.

    *max_segment_sec* is the longest swap segment analysis may propose. Omit it
    to get the universal default (the shortest per-clip ceiling across every
    model, so the segmentation is runnable on any of them); raise it to use a
    longer-clip model such as Seedance 2.5 to its full length, at the cost of
    portability to shorter-clip models. It is consumed by *analysis*, which is
    enqueued right here, so this is the moment that matters: a later PATCH will
    not re-cut the segments this call is about to produce.

    *hook_sec* is the ``localisation`` flow's equivalent knob: how long the
    front-of-video hook that gets re-spoken in another language is. Omit it
    (NULL) to use ``settings.LOCALISATION_DEFAULT_HOOK_SEC``. Same analyze-time
    contract as *max_segment_sec*, and meaningless for the other project types,
    which never read it.
    """
    _validate_project_type(project_type)
    max_segment_sec = _validate_max_segment_sec(max_segment_sec)
    hook_sec = _validate_hook_sec(hook_sec)
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
            status=ProjectStatus.created,
            project_type=project_type,
            mute_source=mute_source,
            max_segment_sec=max_segment_sec,
            hook_sec=hook_sec,
        )
    else:
        link = gdrive_link.strip()
        project = VideoProject(
            id=project_id,
            source_type="gdrive",
            source_ref=link,
            status=ProjectStatus.created,
            project_type=project_type,
            mute_source=mute_source,
            max_segment_sec=max_segment_sec,
            hook_sec=hook_sec,
        )

    # A localisation project is transcribed automatically once analysis returns
    # (tasks._enqueue_transcribe_after_analyze), so its transcript is spoken for
    # from this moment on — say so NOW rather than at the end of analysis.
    # Otherwise transcript_status is NULL for the whole analyze window and NULL
    # means two different things at once ("nobody asked for a transcript" and
    # "one is coming"), which is what makes the panel render a Transcribe button
    # next to a job that is already promised — one click, two paid model calls on
    # the same video. With this, NULL means exactly "never requested" and the UI
    # can act on it without guessing.
    #
    # Only the status column is touched, and only at creation: the task re-writes
    # it anyway, and every other reader (POST /localisation-prompt, the panel)
    # already treats "pending" as "not translatable yet".
    if spec_for(project_type).key == LOCALISATION:
        project.transcript_status = _TRANSCRIPT_PENDING

    db.add(project)
    db.commit()

    # Enqueue analysis — import at module level so monkeypatch targets app.api_v2.*
    enqueue_analyze_project(project_id)

    log.info(
        "Created project %s source_type=%s project_type=%s mute_source=%s "
        "max_segment_sec=%s hook_sec=%s",
        project_id, project.source_type, project_type, mute_source,
        max_segment_sec, hook_sec,
    )
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)
    return ProjectCreateResponse(project_id=project_id, status=status_val)


@router.get("/projects", response_model=list[ProjectListItem])
def list_projects(db: Session = Depends(get_db)) -> list:
    """Return all projects, newest first."""
    projects = (
        db.execute(select(VideoProject).order_by(desc(VideoProject.created_at)))
        .scalars()
        .all()
    )
    return [ProjectListItem.model_validate(p) for p in projects]


@router.get("/projects/{pid}", response_model=ProjectResponse)
def get_project(pid: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """Return full project details."""
    project = _get_project_or_404(pid, db)
    return ProjectResponse.model_validate(project)


@router.patch("/projects/{pid}", response_model=ProjectResponse)
def update_project(
    pid: str, body: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """Update editable project settings.

    Two different contracts here, and the difference matters:

    * ``mute_source`` is read at SUBMIT time, so it takes effect on the next run
      of this project with no re-analysis.
    * ``max_segment_sec`` and ``hook_sec`` are consumed at ANALYZE time, so
      changing them does not re-partition segments that already exist — the
      project has to be re-analyzed (``POST /projects/{pid}/analyze``) for a new
      value to take effect.

    Sending either analyze-time field as ``null`` clears it back to its default
    (the universal segmentation cap / ``LOCALISATION_DEFAULT_HOOK_SEC``).
    ``project_type`` is deliberately NOT editable: it decides how the existing
    segments were cut, so changing it without re-analyzing would leave a project
    whose segmentation contradicts its type.
    """
    project = _get_project_or_404(pid, db)
    if body.name is not None:
        name = body.name.strip()
        project.name = name[:255] or None
    if "max_segment_sec" in body.model_fields_set:
        project.max_segment_sec = _validate_max_segment_sec(body.max_segment_sec)
    if body.mute_source is not None:
        project.mute_source = bool(body.mute_source)
    if "hook_sec" in body.model_fields_set:
        project.hook_sec = _validate_hook_sec(body.hook_sec)
    db.commit()
    db.refresh(project)
    log.info(
        "Updated project %s name=%r max_segment_sec=%s mute_source=%s hook_sec=%s",
        pid, project.name, project.max_segment_sec, project.mute_source,
        project.hook_sec,
    )
    return ProjectResponse.model_validate(project)


@router.post(
    "/projects/{pid}/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProjectCreateResponse,
)
def reanalyze_project(pid: str, db: Session = Depends(get_db)) -> ProjectCreateResponse:
    """Re-run analysis on an existing project (202 Accepted).

    This is the only way an analyze-time setting can be made to take effect
    after creation. ``max_segment_sec`` and ``hook_sec`` are both read when the
    segments are cut, so ``PATCH /projects/{pid}`` alone re-cuts nothing — the
    documented sequence is PATCH, then this.

    **Destructive, by design.** ``pipeline_v2.analyze_project`` replaces the
    segmentation wholesale: it deletes every existing ``SegmentDef`` before
    writing the new proposal, and that cascades to every ``RunSegment`` that
    referenced one.

    What SURVIVES:
      * every ``Run`` row — its prompt, refs, model, status and, for a finished
        run, ``result_local_path`` / the delivered Drive video. A ``done`` run
        stays done and its video stays playable and downloadable.
      * the project's source file, transcript, name and settings.

    What does NOT survive:
      * every ``SegmentDef`` — the ids change, so any per-segment
        ``prompt_override`` naming one is gone.
      * every ``RunSegment`` of every run, including finished ones. The
        per-segment breakdown of an old run (its clips, its Kie task ids, the
        "re-run this segment" button) disappears with them; only the stitched
        result remains.

    Guards (409), mirroring ``DELETE /projects/{pid}`` — same reason, same
    statuses: analysis and the run pipeline both write the rows this deletes.
      * the project is already ``analyzing`` — a second analyze would race the
        first one's segment rewrite.
      * any run of the project is in a non-terminal state
        (``_ACTIVE_RUN_STATUSES``) — deleting its RunSegments mid-flight pulls
        the rows out from under the worker. ``created``, ``done``, ``failed``
        and ``incomplete`` runs do NOT block: nothing is processing them. Note
        that a ``failed``/``incomplete`` run becomes unretryable afterwards
        (its RunSegments are gone) — retry it first if you want it back.

    A ``localisation`` project is additionally marked ``transcript_status =
    "pending"`` here, for the same reason project creation does: analysis
    chains a fresh transcription (``tasks._enqueue_transcribe_after_analyze``),
    so the transcript is spoken for from this moment and the panel must not
    offer a Transcribe button next to a job already queued. The previous
    transcript stays readable until the new one lands, but
    ``POST /localisation-prompt`` gates on the status and will 409 until then.
    """
    project = _get_project_or_404(pid, db)
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
                "re-analysis deletes every segment definition (and with it every "
                "run segment), which would pull rows out from under the worker. "
                "Wait for the run to finish and try again."
            ),
        )

    if spec_for(project.project_type).key == LOCALISATION:
        project.transcript_status = _TRANSCRIPT_PENDING
        project.transcript_error = None
        db.commit()
        db.refresh(project)

    enqueue_analyze_project(pid)
    log.info(
        "Queued RE-analysis for project %s (status=%s, project_type=%s) — "
        "existing SegmentDefs and every RunSegment will be replaced",
        pid, _status_text(project.status), project.project_type,
    )
    return ProjectCreateResponse(
        project_id=pid, status=_status_text(project.status)
    )


@router.delete("/projects/{pid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(pid: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a project: DB rows (cascades to segments/runs) + disk.

    Blocked (409) while the project is analyzing or any of its runs is active, so
    we never remove files a worker is still using.
    """
    project = _get_project_or_404(pid, db)
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
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit segments: project status is {project.status!r}, "
                "expected 'ready'"
            ),
        )

    # An active run has already cut its clips to these boundaries and holds a
    # RunSegment per swap def, FK'd with ON DELETE CASCADE. Editing here deletes
    # those rows out from under the running worker: process_run dies on the
    # vanished row (`rs.seedance_result_url = url` with rs None, once a result
    # comes back), and the segments it had already generated — and paid the AI
    # model for — are gone, with no row left to re-attach the results to.
    #
    # Update-only edits are blocked too, not just deletes: _normalize_partition
    # drops any segment an edit collapses to zero length, so an "updates"
    # payload can cascade just as destructively.
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
                "The run is generating against these exact segment boundaries, "
                "and saving would delete the rows it is working on — losing the "
                "segments it has already generated. Wait for it to finish, then "
                "edit."
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


@router.get("/projects/{pid}/frame")
def get_project_frame(
    pid: str,
    t: float = Query(0.0, description="Timestamp in seconds"),
    db: Session = Depends(get_db),
) -> Response:
    """Extract a single JPEG frame from the project's source video at time *t* seconds.

    Cached as ``frames/frame_<t_ms>.jpg`` inside the project directory.
    Returns 404 if the project or its source video file is not found on disk.
    """
    import subprocess
    import tempfile

    project = _get_project_or_404(pid, db)

    src = project.source_local_path
    if not src or not os.path.exists(src):
        raise HTTPException(
            status_code=404, detail="Source video not available on disk"
        )

    pdir = project_dir(pid)
    frames_dir = os.path.join(pdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    t_ms = int(t * 1000)
    cache_path = os.path.join(frames_dir, f"frame_{t_ms}.jpg")

    if not os.path.exists(cache_path):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=frames_dir)
        os.close(tmp_fd)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(t),
                    "-i", src,
                    "-frames:v", "1",
                    "-q:v", "5",
                    "-f", "image2",
                    tmp_path,
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=500,
                    detail="ffmpeg failed to extract frame",
                )
            os.rename(tmp_path, cache_path)
        except subprocess.TimeoutExpired:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail="ffmpeg timed out")

    with open(cache_path, "rb") as fh:
        data = fh.read()

    return Response(content=data, media_type="image/jpeg")


@router.get("/projects/{pid}/source")
def get_project_source(pid: str, db: Session = Depends(get_db)):
    """Stream the project's original source video (for in-page review/seeking).

    FileResponse handles HTTP Range requests, so the <video> element can scrub.
    404 if the project or its source file is not on disk.
    """
    project = _get_project_or_404(pid, db)
    src = project.source_local_path
    if not src or not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Source video not available on disk")
    return FileResponse(src, media_type="video/mp4", filename="source.mp4")


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
# Run endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{pid}/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=RunCreateResponse,
)
def create_run(
    pid: str,
    prompt: str = Form(...),
    name: Optional[str] = Form(None),
    model: str = Form("seedance"),
    resolution: str = Form(settings.DEFAULT_RESOLUTION),
    audio_mode: str = Form("original"),
    language: Optional[str] = Form(None),
    gdrive_folder_id: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    segment_prompts: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunCreateResponse:
    """Create a new Run under a project (one character attempt).

    The project must be in *ready* status. Exactly one character prompt is
    required. Reference images (files + URLs) are capped at MAX_REFERENCE_IMAGES.

    *segment_prompts* is an optional JSON object ``{segment_def_id: extra_text}``
    of per-segment additions: the extra text is appended to the run prompt for
    that swap segment on the very first run (blank/absent → uses the run prompt).

    *language* is the spoken language of the delivered creative, one of
    app/languages.py's codes. On a ``localisation`` project it is the translation
    target the hook is re-spoken in (``POST /localisation-prompt`` reads it);
    elsewhere it is descriptive metadata. Omit it for English.
    """
    project = _get_project_or_404(pid, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot create run: project status is {project.status!r}, "
                "expected 'ready'"
            ),
        )

    # Validate model
    if model not in ai_models.VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {sorted(ai_models.VALID_MODELS)}",
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
    # A model that emits no audio at all (Gemini Omni) can only have the original
    # source track overlaid. Force "original" regardless of what was requested.
    if not ai_models.spec_for(model).produces_audio:
        audio_mode = "original"

    # Validate language (blank/absent stays NULL, which behaves as English)
    language = _validate_language(language)

    # Validate reference image count
    ref_files = reference_files or []
    ref_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    total_refs = len(ref_files) + len(ref_urls)
    if total_refs > settings.MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many reference images: got {total_refs}, "
                f"max {settings.MAX_REFERENCE_IMAGES}"
            ),
        )

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
        language=language,
        gdrive_folder_id=resolved_folder_id,
        status=RunStatus.created,
        reference_image_urls=[],
    )
    db.add(run)
    db.flush()  # persist id before saving reference files

    # Save reference files into project/run dir
    max_bytes = settings.MAX_UPLOAD_SIZE_MB << 20
    saved_ref_paths: list[str] = []
    refs_dir = os.path.join(project_dir(pid), "runs", run_id, "references")
    os.makedirs(refs_dir, exist_ok=True)
    for rf in ref_files:
        safe = _safe_filename(rf.filename or f"ref_{len(saved_ref_paths)}.jpg")
        dest = os.path.join(refs_dir, safe)
        _save_upload(rf, dest, max_bytes=max_bytes)
        saved_ref_paths.append(dest)

    run.reference_image_urls = saved_ref_paths + list(ref_urls)

    # Optional per-segment prompt additions: pre-create RunSegments carrying a
    # prompt_override (= run prompt + the extra text) so the FIRST run already
    # submits tailored prompts. process_run is idempotent and reuses these.
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
            for sd_id, extra in seg_map.items():
                text = extra.strip() if isinstance(extra, str) else ""
                sd = swap_defs.get(sd_id)
                if text and sd is not None:
                    db.add(
                        RunSegment(
                            run_id=run_id,
                            segment_def_id=sd_id,
                            index=sd.index,
                            status=SegmentStatus.pending,
                            prompt_override=f"{prompt.rstrip()}\n{text}",
                        )
                    )

    # Transition created → queued
    try:
        transition(run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()

    enqueue_process_run(run_id)

    log.info("Created run %s for project %s", run_id, pid)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    return RunCreateResponse(run_id=run_id, status=status_val)


@router.get("/runs/{rid}", response_model=RunResponse)
def get_run(rid: str, db: Session = Depends(get_db)) -> RunResponse:
    """Return full run details."""
    run = _get_run_or_404(rid, db)
    return RunResponse.model_validate(run)


@router.delete("/runs/{rid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(rid: str, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a run: DB rows (cascades to RunSegments) + disk.

    Blocked (409) while the run is active (queued/processing/stitching/delivering).
    """
    run = _get_run_or_404(rid, db)
    if run.status in _ACTIVE_RUN_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a run while it is {run.status.value!r}",
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
def download_run_result(rid: str, db: Session = Depends(get_db)):
    """Download the final video file for a run."""
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
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "X-GDrive-File-Id": run.result_gdrive_file_id or "",
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


def _get_run_segment_or_404(rsid: str, run_id: str, db: Session) -> RunSegment:
    rs = db.get(RunSegment, rsid)
    if rs is None or rs.run_id != run_id:
        raise HTTPException(
            status_code=404, detail=f"RunSegment {rsid!r} not found in run {run_id!r}"
        )
    return rs


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

    Only allowed when the run is in done or failed status. Empty prompt clears the
    override (falls back to run-level prompt). Empty reference list clears the
    override (falls back to run-level references).
    """
    run = _get_run_or_404(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in ("done", "failed"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot edit segment: run status is {run.status!r}; "
                "expected 'done' or 'failed'"
            ),
        )

    rs = _get_run_segment_or_404(rsid, rid, db)

    _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls)

    db.commit()
    db.refresh(rs)
    log.info("Patched RunSegment %s (run %s): prompt_override=%r refs_override=%r",
             rsid, rid, rs.prompt_override, rs.reference_image_urls_override)
    return RunSegmentResponse.model_validate(rs)


@router.post("/runs/{rid}/segments/{rsid}/rerun", response_model=RunResponse)
def rerun_segment(
    rid: str,
    rsid: str,
    prompt: Optional[str] = Form(None),
    reference_files: List[UploadFile] = File(default=[]),
    reference_urls: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> RunResponse:
    """Apply the (optional) prompt/reference override, reset one RunSegment to
    pending, and re-queue the run — atomically, so the re-run always uses the
    prompt sent with THIS request (no separate save needed).

    The run must be in done or failed status. Other completed RunSegments are
    skipped by process_run (resumability); only this segment is reprocessed and
    the final video is re-stitched. If no prompt field is sent at all, the
    existing override is left untouched.
    """
    run = _get_run_or_404(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    if status_val not in ("done", "failed", "incomplete"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot re-run segment: run status is {run.status!r}; "
                "expected 'done', 'failed', or 'incomplete'"
            ),
        )

    rs = _get_run_segment_or_404(rsid, rid, db)

    # Apply the prompt/reference sent with this request so the re-run uses
    # exactly what's on screen. Only when a prompt field is present (a form was
    # sent) — otherwise leave any previously-saved override untouched.
    if prompt is not None or reference_files or reference_urls:
        _apply_segment_override(rs, run, rid, rsid, prompt, reference_files, reference_urls)

    # Reset this RunSegment to pending
    rs.status = SegmentStatus.pending
    rs.error_message = None
    rs.seedance_task_id = None
    rs.seedance_result_url = None
    rs.local_result_path = None
    # Clear the deterministic-failure flag: this segment is about to be submitted
    # again — possibly after being split, or on a model with a longer per-clip
    # ceiling — and a stale flag would make the stitch keep preferring the
    # original footage over the swap this re-run is paying for.
    rs.source_fallback = False

    # Invalidate the stitched final: it contains this segment's OLD result,
    # so it must never be reused by the delivery-only-retry fast path.
    run.result_local_path = None

    # Transition run → queued (done→queued or failed→queued both allowed now)
    try:
        transition(run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    enqueue_process_run(rid)

    log.info("Rerunning segment %s in run %s", rsid, rid)
    return RunResponse.model_validate(run)


@router.post("/runs/{rid}/retry", response_model=RunResponse)
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

    original_status = run.status

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

    db.commit()
    enqueue_process_run(rid)

    log.info("Retrying run %s (was %s)", rid, original_status)
    return RunResponse.model_validate(run)


def _copy_reference_files(items: list, dest_dir: str) -> list:
    """Clone a run's reference list for a copied run.

    Local file paths are copied into *dest_dir* (so the copy is self-contained and
    survives deleting the source run); http(s) URLs and missing/odd entries are
    passed through unchanged.
    """
    out: list = []
    made = False
    for ref in items or []:
        if isinstance(ref, str) and (ref.startswith("http://") or ref.startswith("https://")):
            out.append(ref)
            continue
        if isinstance(ref, str) and os.path.exists(ref):
            if not made:
                os.makedirs(dest_dir, exist_ok=True)
                made = True
            dst = os.path.join(dest_dir, os.path.basename(ref))
            shutil.copy2(ref, dst)
            out.append(dst)
        else:
            out.append(ref)  # best-effort: keep whatever it was
    return out


def _build_copied_run(
    src: Run,
    db: Session,
    *,
    resolution: Optional[str],
    name: Optional[str],
    ref_files: list,
    ref_urls: list,
) -> Run:
    """Build (add + flush, but do NOT commit or enqueue) a new Run cloned from *src*.

    Clones everything that defines the result — prompt, model, audio mode, Drive
    folder, and any per-segment prompt overrides — and applies the two things a
    copy may change: *resolution* and the *reference photo*. Returns the new Run;
    the caller is responsible for committing and enqueueing it.

    Shared by both the single-run copy and the batch copy so the cloning rules
    stay in one place.

    * **resolution** — defaults to the source run's resolution if omitted; pass a
      different one to promote a 480p test to production.
    * **reference photo** — pass *ref_files* (UploadFile list) and/or *ref_urls*
      (list of url strings) to swap the character to a new face of the same type.
      When new references are supplied they REPLACE the photo everywhere — both
      the run-level references AND any per-segment reference overrides are dropped
      so every swap segment uses the new photo (per-segment *prompt* tweaks are
      still carried over). When omitted, the source run's references (run-level
      and per-segment) are cloned unchanged.
    """
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
        gdrive_folder_id=src.gdrive_folder_id,
        status=RunStatus.created,
        reference_image_urls=[],
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
        new_run.reference_image_urls = saved_ref_paths + new_ref_urls
    else:
        # Clone run-level reference images into the new run's own dir.
        new_run.reference_image_urls = _copy_reference_files(
            list(src.reference_image_urls or []), refs_dir
        )

    # Clone per-segment overrides so the copy reproduces the same tuned result.
    # process_run is idempotent: it reuses these RunSegments (by segment_def_id)
    # instead of creating fresh ones, so the overrides take effect.
    #
    # When a new photo is supplied we DROP per-segment reference overrides so the
    # new run-level photo is used in every segment (the "replace everywhere"
    # behaviour); per-segment prompt tweaks are still carried over.
    for src_rs in src.run_segments:
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
        transition(new_run, RunStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return new_run


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
    db: Session = Depends(get_db),
) -> RunCreateResponse:
    """Duplicate a run — optionally at a new resolution and/or with a new
    reference photo — and enqueue it. See ``_build_copied_run`` for the cloning
    rules. To launch several copies (each with its own photo/quality/name) in one
    request, use ``/runs/{rid}/copy-batch``.
    """
    src = _get_run_or_404(rid, db)
    project = _get_project_or_404(src.project_id, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot copy run: project status is {project.status!r}, expected 'ready'",
        )

    new_ref_urls = [u.strip() for u in (reference_urls or "").split(",") if u.strip()]
    new_run = _build_copied_run(
        src, db,
        resolution=resolution,
        name=name,
        ref_files=reference_files or [],
        ref_urls=new_ref_urls,
    )
    new_id = new_run.id

    db.commit()
    enqueue_process_run(new_id)

    log.info("Copied run %s → %s (resolution=%s)", rid, new_id, new_run.resolution)
    status_val = new_run.status.value if hasattr(new_run.status, "value") else str(new_run.status)
    return RunCreateResponse(run_id=new_id, status=status_val)


# runs[<idx>][<field>] — multipart keys for one batch-copy spec. reference_files
# may repeat (several photos per run); the rest appear at most once per index.
_BATCH_KEY_RE = re.compile(
    r"^runs\[(\d+)\]\[(name|resolution|reference_urls|reference_files)\]$"
)


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
            idx, {"name": None, "resolution": None, "ref_urls": [], "ref_files": []}
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
    return [by_index[i] for i in sorted(by_index)]


@router.post(
    "/runs/{rid}/copy-batch",
    status_code=status.HTTP_201_CREATED,
    response_model=RunBatchCopyResponse,
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

    All copies are validated and built in a single transaction: if any spec is
    invalid (bad resolution, too many photos), nothing is created. On success
    every new run is enqueued and returned in submission order.
    """
    src = _get_run_or_404(rid, db)
    project = _get_project_or_404(src.project_id, db)
    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot copy run: project status is {project.status!r}, expected 'ready'",
        )

    form = await request.form()
    specs = _parse_batch_specs(form)
    if not specs:
        raise HTTPException(status_code=400, detail="No runs specified")
    if len(specs) > _MAX_BATCH_COPY_RUNS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many runs: got {len(specs)}, max {_MAX_BATCH_COPY_RUNS}",
        )

    # Build every run first (all validated, added + flushed) before committing,
    # so a bad spec fails the whole batch instead of leaving half of it queued.
    new_runs: list[Run] = []
    for spec in specs:
        new_runs.append(
            _build_copied_run(
                src, db,
                resolution=spec["resolution"],
                name=spec["name"],
                ref_files=spec["ref_files"],
                ref_urls=spec["ref_urls"],
            )
        )

    db.commit()
    for nr in new_runs:
        enqueue_process_run(nr.id)

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
# Localisation — transcript, translation and the New Run prompt builder
# (docs/localisation.md)
# ---------------------------------------------------------------------------

# Transcript lifecycle values (VideoProject.transcript_status). Only "ready"
# unlocks translation; "empty" is a SUCCESS (no speech in the clip) and
# "failed" is non-fatal — the project stays usable and the operator pastes a
# translation by hand, exactly as before this feature existed.
_TRANSCRIPT_READY = "ready"
_TRANSCRIPT_EMPTY = "empty"
_TRANSCRIPT_PENDING = "pending"

# Rough conversational throughput, characters per second, used ONLY to decide
# whether a translated line deserves an operator's second look — never to gate
# anything. Latin-script casual speech runs ~150 wpm, i.e. ~14 chars/s counting
# spaces; Japanese has no spaces and mixes dense kanji with kana, so the same
# second buys far fewer characters. Both numbers are crude by design: the band
# below is wide enough that only a translation that is obviously the wrong
# LENGTH trips it, which is the failure that actually shipped (a literal EN→JA
# hook that overran the shot).
_CHARS_PER_SEC = {"ja": 6.0}
_DEFAULT_CHARS_PER_SEC = 14.0
# Warn past these multiples of the source line's on-screen duration. Overrun is
# the worse failure (the delivery races or runs off the end of the shot), so
# the ceiling is tighter than the floor is loose.
_SPEECH_OVERRUN_RATIO = 1.5
_SPEECH_UNDERRUN_RATIO = 0.5

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

        lines.append(
            {
                "id": line_id,
                "start": start,
                "end": end,
                "speaker": _transcript_text(item.get("speaker"), f"{where}.speaker"),
                "on_screen": bool(item.get("on_screen")),
                "text": _transcript_text(item.get("text"), f"{where}.text"),
            }
        )

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
    rate = _CHARS_PER_SEC.get(language or "", _DEFAULT_CHARS_PER_SEC)
    return len((text or "").strip()) / rate


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

    boundaries = _segment_boundaries(hook_segments, hook)
    for line in window_lines:
        start = _line_seconds(line, "start")
        end = _line_seconds(line, "end")
        for boundary in boundaries:
            if not start < boundary < end:
                continue
            if boundary == hook:
                warnings.append(
                    f"Line {line.get('id')} ({line.get('speaker')}) runs from "
                    f"{start:.1f}s to {end:.1f}s and crosses the {hook:.1f}s hook "
                    "boundary — it will be cut mid-sentence. Shorten the line or "
                    "move the hook."
                )
            else:
                warnings.append(
                    f"Line {line.get('id')} ({line.get('speaker')}) runs from "
                    f"{start:.1f}s to {end:.1f}s and crosses the {boundary:.1f}s "
                    "boundary between two swap segments — one clip speaks it "
                    "whole, so the delivery is cut mid-sentence at the seam. "
                    "Move the boundary or split the line."
                )

    for line in translated_lines:
        duration = _line_seconds(line, "end") - _line_seconds(line, "start")
        if duration <= 0:
            continue
        estimate = _estimated_speech_sec(line.get("text", ""), target_language)
        if estimate > duration * _SPEECH_OVERRUN_RATIO:
            warnings.append(
                f"Line {line.get('id')}: the translation is roughly "
                f"{estimate:.1f}s of speech but the shot gives it "
                f"{duration:.1f}s — it will overrun. Shorten it."
            )
        elif estimate < duration * _SPEECH_UNDERRUN_RATIO:
            warnings.append(
                f"Line {line.get('id')}: the translation is roughly "
                f"{estimate:.1f}s of speech against {duration:.1f}s of footage "
                "— check nothing was dropped."
            )

    return warnings

@router.post(
    "/projects/{pid}/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TranscriptResponse,
)
def transcribe_project(pid: str, db: Session = Depends(get_db)) -> TranscriptResponse:
    """Queue transcription of the project's source video (202 Accepted).

    Sets ``transcript_status="pending"``, clears the previous error and pushes
    the job; the worker takes it from there (docs/localisation.md §8 —
    transcription is its own RQ task, never part of analysis, so a model outage
    can neither fail analysis nor block segmentation).

    Deliberately re-runnable at any time (the project page's "Распознать
    заново"): the task overwrites the transcript wholesale, so a duplicate
    enqueue costs a model call and nothing else, and an operator whose job died
    mid-flight must not be locked out by a stale ``running`` status. The
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

    project.transcript_status = _TRANSCRIPT_PENDING
    project.transcript_error = None
    db.commit()
    db.refresh(project)

    enqueue_transcribe_project(pid)
    log.info("Queued transcription for project %s", pid)
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
    response_model=LocalisationPromptResponse,
)
def build_localisation_prompt(
    pid: str,
    language: str = Form(...),
    swap_character: bool = Form(False),
    hook_sec: Optional[float] = Form(None),
    db: Session = Depends(get_db),
) -> LocalisationPromptResponse:
    """Translate the hook and assemble the Seedance prompt for a localisation run.

    The "Перевести" button behind the New Run form. It slices the cached
    transcript to the hook window, translates those lines into *language*, and
    returns the prompt text (plus per-segment prompts when the hook is cut into
    more than one swap segment) for the operator to edit and submit through the
    ordinary ``create_run`` path.

    **Writes nothing to the database.** No transcript is updated, no run is
    created, no segment is touched — the returned text is a suggestion, and the
    run the operator eventually submits is the only record of what was used.
    That also makes the call freely repeatable: try a language, read the
    warnings, try another.

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
      script is cut to.
    * the STORED ``VideoProject.hook_sec`` — the operator's intent. It bounds
      the analyzed window and drives the NEXT analysis, and it is the fallback
      when there are no swap segments to measure. Use ``PATCH /projects/{pid}``
      + ``POST /projects/{pid}/analyze`` to actually move the hook.

    Status codes:
        200: prompt assembled (possibly with *warnings* — they never block).
        400: unknown language code, or a hook_sec override that is not a
            positive finite number.
        404: no such project.
        409: ``transcript_status`` is not ``ready`` — nothing to translate yet.
        422: the hook window contains no speech, so there is nothing to
            re-speak (a wordless hook is a legal transcript, not an error, but
            it is not translatable).
        502: the translation model failed or returned an unusable answer.
    """
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
    swap_segments = _project_swap_segments(pid, db)
    hook = _resolve_hook_sec(project, _validate_hook_sec(hook_sec), swap_segments)
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

    try:
        translated_lines = localisation.translate_lines(
            window_lines,
            source_language=source_language,
            target_language=target_language,
            video_summary=video_summary,
            scene_context=scene_context,
        )
        prompt = localisation.build_prompt(
            lines=translated_lines,
            source_language=source_language,
            target_language=target_language,
            swap_character=swap_character,
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
                )
        else:
            # No overrides means the run prompt is what the (single) swap
            # segment receives, so an unassigned line is not lost — the clip
            # still says it, mistimed at worst. Claiming otherwise would be a
            # lie; the coverage warning already names the uncovered stretch.
            dropped_lines = []
    except localisation.LocalisationError as exc:
        # An upstream/model failure, not a bad request: 502 tells the operator
        # to retry or fall back to writing the prompt by hand (§8), rather than
        # sending them hunting for a mistake in their own input.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    warnings = _localisation_warnings(
        window_lines=window_lines,
        translated_lines=translated_lines,
        hook=hook,
        hook_segments=hook_segments,
        dropped_lines=dropped_lines,
        source_language=source_language,
        target_language=target_language,
    )

    log.info(
        "Built localisation prompt for project %s: %s -> %s, hook=%.1fs "
        "(stored intent=%s), %d line(s), %d segment prompt(s), %d line(s) in "
        "an uncovered gap, %d warning(s), swap_character=%s",
        pid, source_language or "?", target_language, hook, project.hook_sec,
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
