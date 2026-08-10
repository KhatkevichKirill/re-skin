"""
v2 Web UI router — server-rendered Jinja2 + HTMX pages.

Mounted at /v2 from app/main.py.
Provides: Projects dashboard, Project detail (segment editor + runs panel),
Run detail (progress polling + result preview).

v1 web.py is left completely untouched.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from . import ai_models
from . import languages
from . import project_types as project_types_registry
from .api_v2 import _MAX_BATCH_COPY_RUNS
from .config import settings
from .db import get_db
from .models import Run, RunSegment, SegmentDef, VideoProject
from sqlalchemy.orm import selectinload
from .pipeline_v2 import effective_segment_cap_sec
from .project_types import LOCALISATION, spec_for
from .public import make_result_token, make_source_token
from .state_machine import ProjectStatus, RunStatus, SegmentStatus

log = logging.getLogger(__name__)

router = APIRouter(tags=["web_v2"])

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates", "v2")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

# Every template that names a model renders it through this filter rather than a
# ternary chain, so a new entry in app/ai_models.py is labelled everywhere at
# once.
templates.env.filters["model_label"] = ai_models.label_for
# Same reasoning for the project-type column: one registry lookup, not a
# ternary chain per template.
templates.env.filters["project_type_label"] = project_types_registry.label_for


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


def _new_run_defaults(project: VideoProject) -> dict:
    """New Run form pre-fill — model + prompt from the project's type, plus the
    AI model registry the form's `<select>` and JS are built from.

    The project-type defaults are plain defaults: the form fields stay fully
    editable, and the run stores whatever the operator submits. See
    app/project_types.py.

    The ``model*`` keys come from app/ai_models.py so the model dropdown, the
    model→resolution options map and the per-model clip-length/audio limits derive
    from one table instead of hand-kept copies in the template.

    The ``localisation_*`` / ``is_localisation`` keys drive the extra controls
    that flow needs (docs/localisation.md §7): the speech-only vs
    speech+character radio, the Translate button and its disabled reason, and the
    audio warning. They are computed for EVERY project — the template gates on
    ``is_localisation`` — so the two renders stay identical.

    Shared by the full page render and the HTMX status fragment, which both render
    partials/project_status_content.html — keep them in one helper so the two can
    never drift.
    """
    spec = spec_for(project.project_type)

    # The project type suggests a model, but the project's segmentation cap
    # decides which models can actually generate its segments. Pre-selecting one
    # that cannot is a trap: over-long segments are skipped and the run delivers
    # original footage where a swap was paid for. So when the type's preferred
    # model can't cover the cap, fall back to the first model that can — registry
    # order is cheapest-first within a family, so "first compatible" is also the
    # least-surprising spend.
    #
    # This only ever *widens* what the operator sees pre-filled — every model
    # stays selectable, and `models_over_cap` lets the template mark the ones
    # that would fall back rather than hiding them.
    cap = effective_segment_cap_sec(project.max_segment_sec)
    compatible = ai_models.models_supporting_segment_len(cap)

    # A type whose whole point is a re-spoken soundtrack has a second hard
    # constraint on the model: it must actually emit audio. Gemini Omni never
    # does, and create_run force-downgrades such a run to audio_mode="original"
    # — which overlays the UNTRANSLATED source track over the localised video.
    # So narrow the candidate list here too, and let the template mark the rest
    # unusable (`models_without_audio`) rather than silently offering them.
    models_without_audio = [
        key for key, mspec in ai_models.AI_MODELS.items() if not mspec.produces_audio
    ]
    if spec.requires_audio_model:
        compatible = [k for k in compatible if k not in models_without_audio]
    else:
        models_without_audio = []

    default_model = spec.default_model
    if compatible and default_model not in compatible:
        default_model = compatible[0]
        log.info(
            "project %s cap=%.1fs excludes the %s default model %r — New Run form "
            "defaults to %r instead",
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
        # Effective segmentation cap and the models that cannot honour it, so the
        # <option> list can label them instead of silently offering a run that
        # delivers un-swapped footage.
        "segment_cap_sec": cap,
        "models_over_cap": ai_models.models_excluded_by_segment_len(cap),
        # Models that emit no audio at all, listed ONLY for a type that requires
        # one (empty everywhere else, so the template's marking is driven by the
        # same flag that narrowed `default_model` above).
        "models_without_audio": models_without_audio,
        "scene_min_len_sec": settings.SCENE_MIN_LEN_SECONDS,
        # {model: [[value, label], ...]} — rebuilds the resolution dropdown.
        "model_resolutions_json": ai_models.resolution_choices_json(),
        # {model: {max_clip_sec, produces_audio, label}} — drives the "segments
        # too long for this model" warning and the audio lock.
        "model_limits_json": ai_models.model_limits_json(),
        # The specs themselves, to loop over when building the <option> list.
        "ai_models": ai_models.AI_MODELS,
        # Language <option> list + the pre-selected code. English is the default
        # because it is what an unset language already means.
        "languages": languages.LANGUAGES,
        "default_language": languages.DEFAULT_LANGUAGE,
        # Run.audio_mode the form pre-selects (see above).
        "default_audio_mode": default_audio_mode,
        # __ Localisation-only form state (docs/localisation.md §7) __
        "is_localisation": _is_localisation(project),
        # Both Seedance templates, so the "speech only" / "speech + character"
        # radio can swap the pre-filled prompt without a round-trip.
        "localisation_prompts": localisation_prompts,
        # Which of the two the registry's own default_prompt is, so the radio and
        # the textarea can never load disagreeing with each other.
        "default_localisation_mode": (
            "swap"
            if spec.default_prompt == localisation_prompts["swap"]
            else "keep"
        ),
        # Gates the "Translate" button: it can only build a prompt from a
        # transcript that is ready (docs/localisation.md §4.4 → 409 otherwise).
        # NULL = never requested.
        "transcript_status": _transcript_status(project),
    }


def _is_localisation(project: VideoProject) -> bool:
    """True when this project runs the localisation flow.

    Goes through ``spec_for`` rather than comparing the column directly so a
    row whose ``project_type`` is NULL or unknown resolves the same way every
    other reader resolves it (to the face-swap fallback) instead of quietly
    turning on the localisation UI.
    """
    return spec_for(project.project_type).key == LOCALISATION


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
    from .pipeline_v2 import effective_segment_cap_sec, hook_split_plan

    cap = effective_segment_cap_sec(project.max_segment_sec)
    plan = hook_split_plan(
        duration_sec=project.duration_sec,
        hook_sec=project.hook_sec,
        max_segment_sec=cap,
    )
    cut_sec, cut_segments = _analyzed_hook(segments)

    return {
        # Localisation is the only flow with a hook; every other type resolves
        # through spec_for() exactly as the transcript panel does.
        "hook_visible": _is_localisation(project),
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
        return f"{int(stat.st_mtime)}-{stat.st_size}"
    updated = run.updated_at
    if updated is not None:
        return str(int(updated.timestamp()))
    return "0"


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


def _get_runs(project_id: str, db: Session) -> list[Run]:
    return (
        db.execute(
            select(Run)
            .where(Run.project_id == project_id)
            .order_by(desc(Run.created_at))
        )
        .scalars()
        .all()
    )


def _get_run_segments(run_id: str, db: Session) -> list[RunSegment]:
    return (
        db.execute(
            select(RunSegment)
            .where(RunSegment.run_id == run_id)
            .order_by(RunSegment.index)
            .options(selectinload(RunSegment.segment_def))
        )
        .scalars()
        .all()
    )


# ---------------------------------------------------------------------------
# Projects dashboard — GET /v2/
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def projects_dashboard(
    request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """v2 dashboard: list of VideoProjects + new-project form."""
    projects = (
        db.execute(select(VideoProject).order_by(desc(VideoProject.created_at)))
        .scalars()
        .all()
    )
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "projects": projects,
            # The create form's type <select> is built from the registry, so a
            # type added there appears here (with its own form defaults) without
            # touching the template.
            "project_types": list(project_types_registry.PROJECT_TYPES.values()),
            "default_project_type": project_types_registry.DEFAULT_PROJECT_TYPE,
            "localisation_default_hook_sec": float(
                settings.LOCALISATION_DEFAULT_HOOK_SEC
            ),
        },
    )


# ---------------------------------------------------------------------------
# Dashboard runs pivot — GET /v2/projects/{pid}/runs-fragment
# ---------------------------------------------------------------------------


@router.get("/projects/{pid}/runs-fragment", response_class=HTMLResponse)
def project_runs_fragment(
    pid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX fragment: the Existing-runs list for a project, shown inline in the
    dashboard pivot so finished videos are reachable without opening the project."""
    _get_project_or_404(pid, db)
    runs = _get_runs(pid, db)
    rows = []
    for run in runs:
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        rows.append(
            {
                "id": run.id,
                "name": run.name,
                "status": status_val,
                "model": run.model,
                "resolution": run.resolution,
                "created_at": run.created_at,
                "result_available": (
                    status_val == "done"
                    and bool(run.result_local_path)
                    and os.path.exists(run.result_local_path)
                ),
            }
        )
    return templates.TemplateResponse(
        "partials/project_runs_list.html",
        {"request": request, "project_id": pid, "runs": rows},
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
    return templates.TemplateResponse(
        "project_detail.html",
        {
            "request": request,
            "project": project,
            "status_val": status_val,
            "segments": segments,
            "runs": runs,
            "default_resolution": settings.DEFAULT_RESOLUTION,
            "source_public_token": make_source_token(pid),
            "max_refs": settings.MAX_REFERENCE_IMAGES,
            "gdrive_folder_id": settings.GDRIVE_DEFAULT_FOLDER_ID or "",
            **_new_run_defaults(project),
            **_hook_context(project, segments),
            **_transcript_context(project),
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
    segments = _get_segments(pid, db)
    runs = _get_runs(pid, db)
    status_val = project.status.value if hasattr(project.status, "value") else str(project.status)
    return templates.TemplateResponse(
        "partials/project_status_content.html",
        {
            "request": request,
            "project": project,
            "status_val": status_val,
            "segments": segments,
            "runs": runs,
            "default_resolution": settings.DEFAULT_RESOLUTION,
            "source_public_token": make_source_token(pid),
            "max_refs": settings.MAX_REFERENCE_IMAGES,
            "gdrive_folder_id": settings.GDRIVE_DEFAULT_FOLDER_ID or "",
            **_new_run_defaults(project),
            **_hook_context(project, segments),
            **_transcript_context(project),
        },
    )


# ---------------------------------------------------------------------------
# Run page — GET /v2/runs/{rid}
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}", response_class=HTMLResponse)
def run_detail(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Full run detail page."""
    run = _get_run_or_404(rid, db)
    run_segments = _get_run_segments(rid, db)
    status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
    gdrive_link = _gdrive_link_for_run(run)

    # Compute progress for display
    total_swap = len(run_segments)
    completed = sum(
        1 for rs in run_segments
        if (rs.status.value if hasattr(rs.status, "value") else str(rs.status)) == "completed"
    )

    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run": run,
            "project_id": run.project_id,
            "status_val": status_val,
            "run_segments": run_segments,
            "gdrive_link": gdrive_link,
            "total_swap": total_swap,
            "completed": completed,
            "result_public_token": make_result_token(rid),
            "result_version": _result_version_for_run(run),
            "max_copy_runs": _MAX_BATCH_COPY_RUNS,
            # The copy control's resolution list is built from the run's model's
            # own supported set, so it can never offer a tier create_run rejects.
            "ai_models": ai_models.AI_MODELS,
        },
    )


# ---------------------------------------------------------------------------
# Run status fragment — GET /v2/runs/{rid}/status-fragment
# ---------------------------------------------------------------------------


@router.get("/runs/{rid}/status-fragment", response_class=HTMLResponse)
def run_status_fragment(
    rid: str, request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """HTMX polling fragment for run status-dependent content."""
    run = _get_run_or_404(rid, db)
    run_segments = _get_run_segments(rid, db)
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

    return templates.TemplateResponse(
        "partials/run_status_content.html",
        {
            "request": request,
            "run": run,
            "project_id": run.project_id,
            "status_val": status_val,
            "run_segments": run_segments,
            "gdrive_link": gdrive_link,
            "total_swap": total_swap,
            "completed": completed,
            "generating_seg": generating_seg,
            "result_public_token": make_result_token(rid),
            "result_version": _result_version_for_run(run),
        },
    )
