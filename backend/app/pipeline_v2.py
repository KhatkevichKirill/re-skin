"""
pipeline_v2.py — End-to-end processing for the v2 Project → Runs model.

Public functions
----------------
analyze_project(project_id, *, detector=None)
    Download/locate the source, probe it, propose segments, persist to DB.
    Transitions: created → analyzing → ready.

process_run(run_id, *, kie=None, gdrive=None)
    Submit ALL swap segments to Seedance in parallel, then poll every task
    concurrently (round-robin), downloading each result as it lands. A task
    with no result within RUN_SKIP_TIMEOUT_SEC (default 2h) — or one that
    fails — is skipped: that segment falls back to the original (un-swapped)
    clip so one stuck segment never blocks or fails the whole run.
    Then stitch everything together and deliver to Google Drive.
    Transitions: queued → processing → stitching → delivering → done.

Per-model facts — kie model id, resolutions, clip-length limits, how the output
duration and aspect ratio are derived, whether an audio switch exists — all come
from :mod:`app.ai_models`. Do not reintroduce a per-model dict here.

``resolve_reference_urls`` and ``_map_omni_aspect`` are still imported from
pipeline.py (the frozen v1 module) — the first is genuinely shared, the second is
pure source-geometry logic with no model dimension to it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from . import ai_models
from . import localisation as localisation_mod
from . import media as media_mod
from . import face as face_mod
from . import project_types
from . import scene as scene_mod
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .kie_client import KieClient
from .models import Run, RunSegment, SegmentDef, VideoProject
from .project_types import spec_for
from .pipeline import (
    _map_omni_aspect,
    resolve_reference_urls,
)
from .state_machine import ProjectStatus, RunStatus, SegmentStatus, transition
from .storage import (
    project_source_path,
    run_clips_dir,
    run_results_dir,
)

log = logging.getLogger(__name__)

# Parallel-submit tuning (env-overridable).
# Seedance tasks are submitted all at once and polled concurrently; a task that
# yields no result within RUN_SKIP_TIMEOUT_SEC is skipped (original clip used).
RUN_SKIP_TIMEOUT_SEC = float(os.getenv("RUN_SKIP_TIMEOUT_SEC", "7200"))  # 2 hours
RUN_POLL_INTERVAL_SEC = float(os.getenv("RUN_POLL_INTERVAL_SEC", "15"))

# Total attempts per swap segment before it is marked failed. The AI backends
# (Seedance/Gemini Omni) intermittently return a task-level "Internal Error,
# Please try again later." — we resubmit the same (already-uploaded) clip up to
# this many times. Default 3 = the initial submit + 2 retries.
RUN_TASK_MAX_ATTEMPTS = int(os.getenv("RUN_TASK_MAX_ATTEMPTS", "3"))

# Delivery (Google Drive upload) retry tuning. The upload itself is chunked with
# per-chunk retries; this is a whole-upload retry so a fully-failed delivery
# self-heals without re-generating or re-stitching the video.
RUN_DELIVER_ATTEMPTS = int(os.getenv("RUN_DELIVER_ATTEMPTS", "3"))
RUN_DELIVER_BACKOFF_SEC = float(os.getenv("RUN_DELIVER_BACKOFF_SEC", "10"))

# Stitch-phase parallelism: number of concurrent cut_clip calls for keep/fallback
# segments during assembly.  Bounded so we don't saturate the CPU — each ffmpeg
# process already uses FFMPEG_THREADS threads internally.  Default 2 is safe
# even on a single worker; set higher only if worker has spare CPU headroom.
# Set to 1 to disable parallelism (equivalent to the old serial loop).
STITCH_CUT_CONCURRENCY = int(os.getenv("STITCH_CUT_CONCURRENCY", "2"))

# Submit-phase parallelism: max concurrent cut/upload/create-task calls during
# the submit phase.  Each thread opens its own DB session.  Default 2 keeps I/O
# pressure low on a single-core VPS; raise to 4-6 if upload bandwidth allows.
SUBMIT_CONCURRENCY = int(os.getenv("SUBMIT_CONCURRENCY", "2"))


def _default_kie() -> KieClient:
    return KieClient()


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


def _scene_segments(
    video_path: str,
    *,
    duration_sec: float,
    fps: float | None,
    max_segment_sec: float,
) -> list:
    """Build swap ProposedSegments from PySceneDetect scene cuts.

    Cuts on scene changes (min length = ``settings.SCENE_MIN_LEN_SECONDS``),
    then splits any scene longer than *max_segment_sec* so every clip fits the
    model's per-clip cap. A single-shot video (no cuts found) becomes one
    full-length swap range. Every range is marked ``swap`` — the subtitle-removal
    flow processes the whole video by default.
    """
    ranges = scene_mod.detect_scenes(
        video_path, fps=fps, min_scene_len_sec=settings.SCENE_MIN_LEN_SECONDS
    )
    if not ranges:
        ranges = [(0.0, duration_sec)]

    # Clamp to the probed duration (the last scene's end can round slightly past
    # it) so the partition stays within [0, duration] before the cap-split.
    clamped = [
        (max(0.0, s), min(e, duration_sec)) for s, e in ranges if e - s > 1e-6
    ]
    capped = face_mod.split_max_duration(clamped, max_sec=max_segment_sec)

    return [
        face_mod.ProposedSegment(
            start_sec=s,
            end_sec=e,
            has_face=False,
            action="swap",
        )
        for s, e in capped
    ]


# Why a hook_split hook ended up different from the length the operator asked
# for. Returned by :func:`hook_split_plan` so callers — the segmentation below
# AND the project page, which has to explain the clamp to the operator who is
# looking at its consequences — branch on a value instead of re-deriving the
# comparison and drifting from it.
HOOK_CLAMP_NONE = ""          # got exactly what was asked for
HOOK_CLAMP_FLOOR = "floor"    # padded UP to the shortest generatable clip
HOOK_CLAMP_CAP = "cap"        # cut DOWN by the project's segmentation cap
HOOK_CLAMP_DURATION = "duration"  # cut DOWN by the length of the video


def effective_segment_cap_sec(max_segment_sec: Optional[float]) -> float:
    """The segmentation cap analysis will actually apply to a project.

    ``VideoProject.max_segment_sec`` is a *preference*: NULL means "no explicit
    choice" and resolves to the smallest per-clip ceiling in the registry (so the
    segmentation is runnable on every model), and any value is bounded by the
    largest ceiling any model offers (segmenting beyond that produces segments
    nothing can generate).

    Exposed as a function, not inlined, so callers outside the pipeline — the API
    validating a PATCH, the project page telling the operator what analysis will
    do — resolve the cap the way analysis will rather than approximating it.

    NULL is tested with ``is None``, not ``or`` — 0.0 is falsy but it is a
    *value*, and a project whose cap really is 0 should be visibly broken rather
    than quietly redrawn as the default.
    """
    cap = max_segment_sec
    if cap is None:
        cap = ai_models.UNIVERSAL_MAX_SEGMENT_SEC
    return min(float(cap), ai_models.ABSOLUTE_MAX_SEGMENT_SEC)


def hook_split_plan(
    *,
    duration_sec: float | None,
    hook_sec: float | None,
    max_segment_sec: float,
) -> dict:
    """Resolve a requested hook length into the one analysis will really cut.

    This is the clamp described in the docstring of
    :func:`_hook_split_segments`, lifted out of it so that the *reason* a hook
    changed length is a value two callers can read rather than a log line only
    the worker ever sees:

    * :func:`_hook_split_segments` uses ``effective`` to lay down the segments;
    * ``web_v2`` uses the whole dict to tell the operator, on the page where
      the hook is set and next to the segments it produced, that the 15s they
      asked for became a 10s cut and which of the two bounds did it.

    That second caller is the entire point. The clamp is correct but was
    invisible: nothing outside the worker log distinguished "your hook is 10s
    because you asked for 10s" from "your hook is 10s because your segmentation
    cap is 10s", and the two want opposite fixes.

    *duration_sec* may be ``None`` — the project page renders before analysis
    has probed the video, and a hook can still be clamped by the cap then. With
    an unknown duration the ceiling is the cap alone and
    ``HOOK_CLAMP_DURATION`` can never be the answer.
    """
    # NULL hook_sec → the configured default. Tested with `or`, not `is None`,
    # per docs/localisation.md §5: unlike max_segment_sec (where 0.0 is a
    # meaningful, if wrong, operator choice worth surfacing) a 0-second hook is
    # never anything but a mistake, and degrading it to the default beats
    # clamping it up into a 4-second hook nobody asked for.
    requested = float(hook_sec or settings.LOCALISATION_DEFAULT_HOOK_SEC)

    floor = min(spec.min_clip_sec for spec in ai_models.AI_MODELS.values())
    cap = float(max_segment_sec)
    duration = None if duration_sec is None else float(duration_sec)
    ceiling = cap if duration is None else min(cap, duration)

    # max() first, then min(): on a source shorter than the floor (a 3s upload)
    # the duration bound wins, so the hook is the whole video rather than a
    # segment that runs off the end of it.
    effective = min(max(requested, floor), ceiling)

    if effective > requested + 1e-6:
        reason = HOOK_CLAMP_FLOOR
    elif effective < requested - 1e-6:
        # Both bounds can bind at once (cap == duration); name the video, which
        # is the one the operator cannot argue with. `<=` rather than `<` for
        # that tie.
        reason = (
            HOOK_CLAMP_DURATION
            if duration is not None and duration <= cap
            else HOOK_CLAMP_CAP
        )
    else:
        reason = HOOK_CLAMP_NONE

    return {
        "requested": requested,
        "effective": effective,
        "floor": floor,
        "cap": cap,
        "duration": duration,
        "reason": reason,
        # True when the request is the configured default rather than a number
        # the operator typed, so the UI can say "default" instead of implying
        # they chose it.
        "defaulted": not hook_sec,
    }


def _hook_split_segments(
    *,
    duration_sec: float,
    hook_sec: float | None,
    max_segment_sec: float,
) -> list:
    """Build the localisation partition: swap over the hook, keep over the rest.

    A ``localisation`` project re-generates only the **hook** — the opening
    seconds whose speech is re-spoken in the target language. The rest of the
    source is proposed as a single ``keep`` segment, so the delivered video is
    the localised hook followed by the original remainder, untouched.

    (In the private deployment this branch ports from, that keep segment is
    excluded from the run's stitch plan and a language-matched tail from a clip
    library is stitched on instead. The clip library is not part of this repo, so
    here the remainder is simply kept — trim or replace it downstream if the
    original tail is not what you want to deliver.)

    *hook_sec* is clamped into what a model can actually generate:

    ``lo`` — the smallest ``min_clip_sec`` in the model registry. A swap clip
    shorter than that is rejected outright by the backend (Seedance 2.5 refuses
    anything under 4s), so a 0.5s hook would produce a segment no run could
    ever submit. Clamping up gives the operator a hook that works instead of a
    project that fails at submit time. The floor is read off the registry rather
    than hard-coded so adding a pickier model moves it automatically.

    ``hi`` — ``min(max_segment_sec, duration_sec)``. The segmentation cap is
    what keeps every proposed segment generatable by the models the project has
    committed to (see the long note in analyze_project); a hook may not escape
    it. The duration bound is the obvious one: a hook cannot be longer than the
    video it is cut from.

    Because the clamp already bounds the hook by the cap, the cap-split below is
    a no-op today. It is applied anyway — the §5 pseudocode has it, and it is
    the difference between "one segment that is silently too long" and "two
    valid segments" if the clamp order is ever loosened.

    A hook that covers the whole video yields the swap segment(s) alone: an
    empty trailing keep segment would be a zero-length clip the stitch would
    have to special-case.

    The clamp itself lives in :func:`hook_split_plan` — the project page shows
    the operator what it did, and one implementation is the only way the
    explanation on screen can be trusted to match the segments underneath it.
    """
    plan = hook_split_plan(
        duration_sec=duration_sec,
        hook_sec=hook_sec,
        max_segment_sec=max_segment_sec,
    )
    clamped = plan["effective"]
    if plan["reason"]:
        log.info(
            "hook_split: hook %.2fs clamped to %.2fs by %s (floor=%.2fs, "
            "ceiling=min(cap %.2fs, duration %.2fs))",
            plan["requested"], clamped, plan["reason"], plan["floor"],
            max_segment_sec, duration_sec,
        )

    proposed = [
        face_mod.ProposedSegment(
            start_sec=s, end_sec=e, has_face=False, action="swap",
        )
        for s, e in face_mod.split_max_duration(
            [(0.0, clamped)], max_sec=max_segment_sec
        )
    ]

    # Trailing keep segment, omitted when the hook covers the whole video. The
    # comparison carries a tolerance because a hook clamped to exactly the
    # duration (the usual outcome for a source shorter than the default hook)
    # would otherwise leave a keep segment of a few float-error microseconds.
    if duration_sec - clamped > 1e-6:
        proposed.append(
            face_mod.ProposedSegment(
                start_sec=clamped,
                end_sec=float(duration_sec),
                has_face=False,
                action="keep",
            )
        )
    return proposed


# ---------------------------------------------------------------------------
# analyze_project
# ---------------------------------------------------------------------------


def analyze_project(project_id: str, *, detector=None) -> None:
    """
    Probe the source video and propose segments for a VideoProject.

    Transitions
    -----------
    created → analyzing → ready
    (on error: → failed, error_message set, exception re-raised)
    """
    log.info("analyze_project start: project_id=%s", project_id)

    with get_session() as session:
        project: VideoProject = session.get(VideoProject, project_id)
        if project is None:
            raise ValueError(f"VideoProject not found: {project_id}")

        try:
            # created → analyzing (commit so readers see live state)
            transition(project, ProjectStatus.analyzing)
            session.commit()

            # Resolve local source path.
            if project.source_type == "gdrive":
                gdrive = _default_gdrive()
                local = project_source_path(project_id, "mp4")
                log.info("Downloading gdrive source %s → %s", project.source_ref, local)
                gdrive.download_file(project.source_ref, local)
                project.source_local_path = local
                session.commit()
            else:
                local = project.source_local_path
                if not local or not os.path.exists(local):
                    raise FileNotFoundError(
                        f"Source file not found at {local!r} for project {project_id}"
                    )

            # Probe.
            info = media_mod.probe(local)
            project.duration_sec = info.duration_sec
            project.width = info.width
            project.height = info.height
            project.fps = info.fps
            project.aspect_ratio = info.aspect_ratio
            session.commit()

            # Propose segments — create SegmentDef rows.
            #
            # The model is chosen per-Run (a project's segmentation is reused
            # across runs), so the cap is a property of the PROJECT, not of the
            # run's model. NULL means the universal default: the smallest
            # per-clip ceiling across every model in the registry (10s, Gemini
            # Omni), so a project cut that way is runnable on anything.
            #
            # Setting it higher trades model portability for fewer stitch seams
            # — the point of Seedance 2.5, which takes 30s clips. A run on a
            # shorter-limit model then marks the over-long segments failed with
            # RunSegment.source_fallback and the stitch substitutes the original
            # footage for them (see _submit_swap_segment_isolated).
            max_segment_sec = effective_segment_cap_sec(project.max_segment_sec)
            log.info(
                "project_id=%s segmentation cap=%.1fs (project setting=%s) — "
                "models able to run it: %s",
                project_id, max_segment_sec, project.max_segment_sec,
                ",".join(ai_models.models_supporting_segment_len(max_segment_sec)),
            )
            # One dispatch on spec.segmentation, not a chain of booleans: the
            # strategies are mutually exclusive, and with a field the registry
            # decides which one runs instead of the order of this if/elif
            # (docs/localisation.md §5). An unknown value falls through to the
            # blank slate, which is always safe — the operator can segment by
            # hand — rather than crashing analysis on a row written by a newer
            # deploy.
            spec = spec_for(project.project_type)
            if spec.segmentation == project_types.SEG_DETECT_FACES:
                # Face-swap: InsightFace-driven swap/keep partition.
                proposed = face_mod.propose_segments(
                    local,
                    duration_sec=info.duration_sec,
                    max_segment_sec=max_segment_sec,
                    detector=detector,
                )
                log.info("propose_segments returned %d segments", len(proposed))
            elif spec.segmentation == project_types.SEG_SCENE_DETECT:
                # Subtitle-removal: cut on scene changes (PySceneDetect), cap each
                # scene at the per-clip limit, and mark every scene "swap" so the
                # whole video is processed by default. The operator can flip any
                # scene to "keep" to skip it (e.g. a scene with no on-screen text).
                proposed = _scene_segments(
                    local,
                    duration_sec=float(info.duration_sec),
                    fps=info.fps,
                    max_segment_sec=max_segment_sec,
                )
                log.info(
                    "scene detection produced %d swap segment(s) for project_type=%s",
                    len(proposed), project.project_type,
                )
            elif spec.segmentation == project_types.SEG_HOOK_SPLIT:
                # Localisation: swap over the hook, keep over the discarded
                # tail. Unlike the other two strategies this one looks at
                # nothing but the clock — where the hook ends is the operator's
                # editorial call (VideoProject.hook_sec), not something a
                # detector can find, so no frames are read here at all.
                proposed = _hook_split_segments(
                    duration_sec=float(info.duration_sec),
                    hook_sec=project.hook_sec,
                    max_segment_sec=max_segment_sec,
                )
                log.info(
                    "hook_split produced %d segment(s) for project_type=%s "
                    "(hook_sec=%s, default=%.1fs): %s",
                    len(proposed), project.project_type, project.hook_sec,
                    settings.LOCALISATION_DEFAULT_HOOK_SEC,
                    ", ".join(
                        f"{ps.action}[{ps.start_sec:.2f},{ps.end_sec:.2f}]"
                        for ps in proposed
                    ),
                )
            else:
                # Blank slate: one full-length "keep" segment the operator splits.
                proposed = [
                    face_mod.ProposedSegment(
                        start_sec=0.0,
                        end_sec=float(info.duration_sec),
                        has_face=False,
                        action="keep",
                    )
                ]
                log.info(
                    "project_type=%s — seeded 1 full-length keep segment",
                    project.project_type,
                )


            # Idempotent re-analyze: clear any existing SegmentDefs first so a
            # re-run replaces the segmentation instead of duplicating it.
            # (Cascades to RunSegments via FK — re-analysis invalidates prior runs.)
            for old in list(project.segments):
                session.delete(old)
            session.flush()

            for idx, ps in enumerate(proposed):
                session.add(
                    SegmentDef(
                        project_id=project_id,
                        index=idx,
                        start_sec=ps.start_sec,
                        end_sec=ps.end_sec,
                        has_face=ps.has_face,
                        action=ps.action,
                    )
                )

            transition(project, ProjectStatus.ready)
            session.commit()
            log.info(
                "analyze_project done: project_id=%s, segments=%d",
                project_id, len(proposed),
            )

        except Exception as exc:
            log.exception("analyze_project failed for project_id=%s", project_id)
            if project.status not in (ProjectStatus.ready, ProjectStatus.failed):
                try:
                    transition(project, ProjectStatus.failed)
                except Exception:
                    project.status = ProjectStatus.failed
            project.error_message = str(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# process_run — parallel submit + concurrent poll
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# transcribe_project — the localisation side channel
# ---------------------------------------------------------------------------

# video_projects.transcript_status values (docs/localisation.md §3.1). Plain
# strings, not an enum: this is one task's own state, nothing in the
# ProjectStatus state machine gates on it (see the column comment in models.py).
TRANSCRIPT_PENDING = "pending"
TRANSCRIPT_RUNNING = "running"
TRANSCRIPT_READY = "ready"
TRANSCRIPT_FAILED = "failed"
TRANSCRIPT_EMPTY = "empty"


def transcribe_project(project_id: str, *, kie=None) -> None:
    """
    Transcribe a project's source video into ``VideoProject.transcript``.

    This is a **side channel**, not a pipeline phase. It runs as its own RQ job
    after analysis (docs/localisation.md §8) and touches nothing analysis or a
    run depends on: no ProjectStatus transition, no SegmentDefs, no
    ``error_message``. A Gemini outage costs the operator the automatic
    translation and nothing else — they paste a script by hand, exactly as they
    did before this feature existed.

    Three outcomes, all recorded on the project and none of them raised:

    ``ready``   a transcript with at least one line.
    ``empty``   the model heard no speech. A SUCCESS: a wordless hook is legal,
                and it is stored as its own status rather than as an error so
                the UI can say "no speech in the hook" and disable Translate
                instead of offering a pointless retry.
    ``failed``  :class:`~app.localisation.LocalisationError` (API down, bad key,
                unparseable answer) or a missing source file, with the message
                in ``transcript_error`` for the operator to read.

    Idempotent and safe to re-run: every attempt overwrites the three transcript
    columns and nothing else, so re-running is how the operator retries. A
    failed attempt deliberately LEAVES a previously-good transcript in place —
    losing a good one to a transient outage would be strictly worse than showing
    a stale one next to a "failed" badge.

    The model call happens between two short sessions rather than inside one:
    transcription is a single HTTP request that can legitimately take minutes
    (``settings.LOCALISATION_TIMEOUT_SEC`` defaults to 600s), and holding a DB
    connection open across it would pin a pool slot for the whole wait.

    Args:
        project_id: VideoProject to transcribe.
        kie: Optional KieClient, forwarded to
            :func:`app.localisation.transcribe_video` (tests inject a fake).

    Raises:
        ValueError: If the project does not exist — a genuine programming/
            scheduling bug, unlike every provider failure above, so it is NOT
            swallowed into transcript_error.
    """
    log.info("transcribe_project start: project_id=%s", project_id)

    # --- 1. resolve the source and claim the attempt -----------------------
    with get_session() as session:
        project: VideoProject = session.get(VideoProject, project_id)
        if project is None:
            raise ValueError(f"VideoProject not found: {project_id}")

        # analyze_project has already downloaded (gdrive) the source by the
        # time this job runs, so the local path is simply read here —
        # re-downloading would race with an in-flight re-analysis for a file we
        # already have.
        local = project.source_local_path
        if not local or not os.path.exists(local):
            msg = f"source file not found at {local!r}"
            log.error("transcribe_project: project_id=%s %s", project_id, msg)
            project.transcript_status = TRANSCRIPT_FAILED
            project.transcript_error = msg
            session.commit()
            return

        project.transcript_status = TRANSCRIPT_RUNNING
        project.transcript_error = None
        session.commit()

    # --- 2. the model call (no session held) -------------------------------
    try:
        transcript = localisation_mod.transcribe_video(local, kie=kie)
    except localisation_mod.LocalisationError as exc:
        log.warning(
            "transcribe_project: project_id=%s transcription failed: %s",
            project_id, exc,
        )
        _write_transcript_failure(project_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001 — see below
        # Anything the localisation layer did not classify (a bug there, an
        # unexpected library error) is still recorded rather than left to the RQ
        # failed registry: the operator's view of this feature is the project
        # page, and a transcript that is silently NULL forever is the one
        # outcome the UI cannot explain.
        log.exception("transcribe_project: project_id=%s unexpected error", project_id)
        _write_transcript_failure(project_id, str(exc))
        return

    # --- 3. record the result ----------------------------------------------
    lines = transcript.get("lines") or []
    status = TRANSCRIPT_READY if lines else TRANSCRIPT_EMPTY

    with get_session() as session:
        project = session.get(VideoProject, project_id)
        if project is None:
            # Deleted while the model was thinking — nothing to write to.
            log.warning(
                "transcribe_project: project %s disappeared mid-transcription",
                project_id,
            )
            return
        project.transcript = transcript
        project.transcript_status = status
        project.transcript_error = None
        session.commit()

    log.info(
        "transcribe_project done: project_id=%s status=%s language=%r lines=%d",
        project_id, status, transcript.get("source_language"), len(lines),
    )


def _write_transcript_failure(project_id: str, message: str) -> None:
    """Record a failed transcription attempt, keeping any earlier transcript."""
    with get_session() as session:
        project = session.get(VideoProject, project_id)
        if project is None:
            return
        project.transcript_status = TRANSCRIPT_FAILED
        project.transcript_error = message
        session.commit()


# ---------------------------------------------------------------------------
# process_run — parallel submit + concurrent poll
# ---------------------------------------------------------------------------


def _parse_result_url(data: dict) -> Optional[str]:
    """Extract resultUrls[0] from a recordInfo 'data' dict (resultJson is a string)."""
    raw = data.get("resultJson") or "{}"
    try:
        urls = json.loads(raw).get("resultUrls") or []
    except (ValueError, TypeError):
        urls = []
    return urls[0] if urls else None


def _submit_swap_segment_isolated(
    *,
    rs_id: str,
    sd_start_sec: float,
    sd_end_sec: float,
    sd_pre_roll_sec: float,
    sd_post_roll_sec: float,
    sd_index: int,
    run_model: str,
    run_prompt: Optional[str],
    run_resolution: Optional[str],
    project_aspect_ratio: Optional[str],
    project_width: Optional[int],
    project_height: Optional[int],
    source: str,
    duration_sec: float,
    clip_dst: str,
    ref_urls: list,
    prompt_override: Optional[str],
    mute_audio: bool = False,
    generate_audio: Optional[bool] = None,
    kie: KieClient,
    kie_factory: Optional[Callable[[], KieClient]] = None,
) -> Optional[dict]:
    """Cut, upload, then create_task for one swap segment.

    Opens its own DB session so this function is safe to call from a thread pool
    (no shared SQLAlchemy Session across threads).

    Returns a "resubmit recipe" dict ``{task_id, recipe}`` where ``recipe`` holds
    everything :func:`_create_swap_task` needs to re-create the task against the
    same already-uploaded clip (used by the in-poll retry path). Returns ``None``
    when the segment was skipped (marked failed — e.g. a segment longer than the
    selected model's per-clip ceiling).

    *mute_audio* mirrors the project's "remove original audio" setting: the clip
    is cut video-only so the model generates audio from the prompt instead of
    reacting to the source soundtrack.

    *generate_audio* is the already-decided answer to "ask this model to write an
    audio track?" — computed once per run in :func:`process_run` and carried
    verbatim into the recipe so an in-poll resubmit rebuilds the identical
    request. ``None`` means "don't send the field", i.e. leave the model on its
    own default.

    Logs per-segment timing for observability.
    """
    t0 = time.monotonic()
    thread_kie = kie_factory() if kie_factory is not None else kie
    model_spec = ai_models.spec_for(run_model)
    is_omni = ai_models.is_omni(run_model)

    # Recompute clip bounds from primitive data (mirrors _clip_bounds logic).
    clip_start = max(0.0, sd_start_sec - sd_pre_roll_sec)
    clip_end = min(duration_sec, sd_end_sec + sd_post_roll_sec)
    # Pad up to the MODEL's minimum reference-clip length, not a global constant:
    # the Seedance 2.0 family accepts ~2s, but 2.5 in video-editing mode demands
    # a 4-30s input and rejects the whole request below that. Grow forward first,
    # then backward if the source runs out.
    model_min_clip = model_spec.min_clip_sec
    if clip_end - clip_start < model_min_clip:
        clip_end = min(duration_sec, clip_start + model_min_clip)
        if clip_end - clip_start < model_min_clip:
            clip_start = max(0.0, clip_end - model_min_clip)
        if clip_end - clip_start < model_min_clip - 0.05:
            # The whole source is shorter than this model's floor — nothing we
            # can pad to. Deterministic (a retry cannot lengthen the video), so
            # fall back to the original footage rather than block the run.
            with get_session() as session:
                rs = session.get(RunSegment, rs_id)
                _skip_segment(
                    rs,
                    f"source is {duration_sec:.1f}s < {model_min_clip:.0f}s "
                    f"{model_spec.label} minimum; original footage used",
                    session,
                    source_fallback=True,
                )
            log.warning(
                "segment idx=%d: source %.1fs is shorter than the %s %.1fs "
                "minimum clip length — swap skipped, ORIGINAL footage used",
                sd_index, duration_sec, model_spec.label, model_min_clip,
            )
            return None

    # Every model has a hard per-clip ceiling (AIModelSpec.max_clip_sec: 10s for
    # Gemini Omni, 15s for the Seedance 2.0 family, 30s for 2.5). The project's
    # segmentation cap (VideoProject.max_segment_sec) is chosen independently of
    # the run's model, so a project segmented at 30s for Seedance 2.5 can be run
    # on a 15s model — this guard is what keeps that safe. Two distinct cases:
    #   - the *segment itself* is over the ceiling: skip the swap entirely.
    #     Swapping only its first N seconds would desync the timeline against
    #     the untouched soundtrack. Deterministic, so the ORIGINAL footage is
    #     substituted at stitch time (source_fallback=True) rather than blocking
    #     the run forever.
    #   - only pre/post-roll pushed the *clip* over: trim the clip back to the
    #     ceiling. The segment still fits; we just carry less surrounding
    #     context into the generation.
    model_max_clip = model_spec.max_clip_sec
    seg_len = sd_end_sec - sd_start_sec
    if seg_len > model_max_clip + 0.05:
        # Kept short: the run UI truncates rs.error_message. The segment index is
        # deliberately absent — the message is rendered inside that segment's own
        # panel, right under its "Segment N" heading.
        with get_session() as session:
            rs = session.get(RunSegment, rs_id)
            _skip_segment(
                rs,
                f"{seg_len:.1f}s > {model_max_clip:.0f}s {model_spec.label} "
                "limit; original footage used instead",
                session,
                source_fallback=True,
            )
        log.warning(
            "segment idx=%d (%.1fs) exceeds the %s %.0fs per-clip limit — "
            "swap skipped, ORIGINAL footage will be stitched for this segment. "
            "Split it or choose a longer-clip model to get it swapped.",
            sd_index, seg_len, model_spec.label, model_max_clip,
        )
        return None
    if clip_end - clip_start > model_max_clip:
        # Trim order matters, and "shorten from the end" is the wrong answer.
        # clip_start = sd_start_sec - pre_roll_sec, so a bare
        # `clip_end = clip_start + model_max_clip` spends the budget on pre-roll
        # and drops real segment footage off the tail: with max_clip=10, a 9.5s
        # segment, pre_roll=2.0 and post_roll=0.5 the segment passes the skip
        # check above and then loses its final 1.5s, while 2.0s of pre-roll
        # survives. The generated clip is concatenated whole, so that swaps
        # footage the operator never marked and leaves footage they did marked
        # un-swapped.
        #
        # So give back the ROLL CONTEXT first — post-roll before pre-roll, since
        # trailing context matters less to the generation than the lead-in — and
        # never cut into [sd_start_sec, sd_end_sec] itself.
        # (min(duration_sec, ...) only matters for a segment whose end sits past
        # the source's own end — never cut past EOF, or the uploaded clip is
        # shorter than the duration we then request.)
        clip_end = min(duration_sec, max(sd_end_sec, clip_start + model_max_clip))
        if clip_end - clip_start > model_max_clip:
            clip_start = min(sd_start_sec, clip_end - model_max_clip)
        if clip_end - clip_start > model_max_clip:
            # The segment itself is now the whole clip, so the only thing that
            # can still be over is the +0.05 epsilon the skip check allows —
            # ≤50ms, under one frame at any fps we handle. Clamp and move on.
            clip_end = clip_start + model_max_clip

    # Two reasons to send a video-only clip:
    #   - Gemini Omni fails when its reference clip carries an audio track.
    #   - The project has "remove original audio" set, so the model can't hear
    #     the source soundtrack and generates audio from the prompt instead.
    # Either way the original audio is still available at stitch time via
    # audio_mode="original" (the source file itself is never modified).
    include_audio = not is_omni and not mute_audio
    media_mod.cut_clip(
        source, clip_start, clip_end, clip_dst, include_audio=include_audio
    )
    t_cut = time.monotonic()

    with get_session() as session:
        rs = session.get(RunSegment, rs_id)
        rs.local_clip_path = clip_dst
        transition(rs, SegmentStatus.uploading)
        session.commit()

    clip_url = thread_kie.upload_file(clip_dst, "charswap/segments")
    t_upload = time.monotonic()

    with get_session() as session:
        rs = session.get(RunSegment, rs_id)
        rs.kie_upload_url = clip_url
        transition(rs, SegmentStatus.submitted)
        session.commit()

    effective_prompt = prompt_override if prompt_override else (run_prompt or "")

    # Bundle everything needed to (re)create the task; the poll loop reuses this
    # to resubmit the same clip on a transient task failure.
    recipe = {
        "run_model": run_model,
        "effective_prompt": effective_prompt,
        "ref_urls": list(ref_urls),
        "clip_url": clip_url,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "run_resolution": run_resolution,
        "project_aspect_ratio": project_aspect_ratio,
        "project_width": project_width,
        "project_height": project_height,
        # Carried so an in-poll resubmit bills the same request as the initial
        # submit — a retry that flipped the audio switch would hand back a clip
        # that does not match its siblings.
        "generate_audio": generate_audio,
    }
    task_id = _create_swap_task(kie=thread_kie, **recipe)

    with get_session() as session:
        rs = session.get(RunSegment, rs_id)
        rs.seedance_task_id = task_id
        transition(rs, SegmentStatus.generating)
        session.commit()

    t_done = time.monotonic()
    log.info(
        "segment idx=%d submitted task=%s cut=%.1fs upload=%.1fs create=%.1fs total=%.1fs",
        sd_index, task_id,
        t_cut - t0,
        t_upload - t_cut,
        t_done - t_upload,
        t_done - t0,
    )
    return {"task_id": task_id, "recipe": recipe}


def _create_swap_task(
    *,
    kie: KieClient,
    run_model: str,
    effective_prompt: str,
    ref_urls: list,
    clip_url: str,
    clip_start: float,
    clip_end: float,
    run_resolution: Optional[str],
    project_aspect_ratio: Optional[str],
    project_width: Optional[int],
    project_height: Optional[int],
    generate_audio: Optional[bool] = None,
) -> str:
    """Create a Seedance/Omni task for an already-uploaded swap clip.

    Shared by the initial submit and the in-poll retry path so both build
    identical request parameters. Does no DB or file work — pure API call.

    *generate_audio* is decided once per run by the caller; ``None`` (the
    default) means "omit the field", which is mandatory for models that do not
    accept it.
    """
    spec = ai_models.spec_for(run_model)

    if spec.family == "omni":
        # clip was already trimmed to <= the model's max_clip_sec at cut time;
        # send its full (video-only) length as the trim range.
        trim_end = round(clip_end - clip_start, 2)
        return kie.create_omni_task(
            prompt=effective_prompt,
            image_urls=ref_urls,
            video_url=clip_url,
            video_start=0,
            video_end=trim_end,
            resolution=ai_models.resolution_or_default(run_model, run_resolution),
            aspect_ratio=_map_omni_aspect(
                project_aspect_ratio, project_width, project_height
            ),
            duration=ai_models.duration_for(run_model, clip_start, clip_end),
        )

    # `generate_audio` is 2.5-only — the 2.0 family rejects the field, so the key
    # must be absent from their payload entirely. create_task enforces the
    # capability itself and would raise if we passed the field to a 2.0 model.
    extra: dict = {}
    if spec.supports_generate_audio and generate_audio is not None:
        extra["generate_audio"] = bool(generate_audio)

    return kie.create_task(
        prompt=effective_prompt,
        reference_image_urls=ref_urls,
        reference_video_urls=[clip_url],
        # Never send a resolution this model cannot do (e.g. a 1080p run
        # switched to Seedance 2.5, which stops at 720p) — the registry
        # substitutes the closest supported tier that does not exceed the
        # request (720p here), not the model's cheapest.
        resolution=ai_models.resolution_or_default(
            run_model, run_resolution or settings.DEFAULT_RESOLUTION
        ),
        # A model that derives its geometry from the input video (Seedance 2.5 in
        # video-editing mode, which is every submission we make) REQUIRES
        # "adaptive" + duration -1 and 500s on anything else — even a ratio that
        # matches the source. Both come from the registry so the pair can never
        # be sent half-right.
        aspect_ratio=ai_models.aspect_for(run_model, project_aspect_ratio),
        duration=ai_models.duration_for(run_model, clip_start, clip_end),
        model=ai_models.kie_model_id(run_model),
        **extra,
    )


def _deliver_with_retry(gdrive: GDriveClient, path: str, folder_id: str, run_id: str) -> dict:
    """Upload the final video to Drive, retrying the whole upload on failure.

    Raises the last exception if every attempt fails (the caller then marks the
    run failed — but the final video stays on disk, so a manual Retry re-delivers
    without re-stitching).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, RUN_DELIVER_ATTEMPTS + 1):
        try:
            return gdrive.upload_file(path, folder_id, f"reskin_run_{run_id}.mp4")
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Drive upload attempt %d/%d failed for run %s: %s",
                attempt, RUN_DELIVER_ATTEMPTS, run_id, exc,
            )
            if attempt < RUN_DELIVER_ATTEMPTS:
                time.sleep(RUN_DELIVER_BACKOFF_SEC * attempt)
    raise last_exc  # type: ignore[misc]


def _poll_pending_tasks(
    *,
    pending: dict[str, dict],
    kie: KieClient,
    results_dir: str,
) -> None:
    """Poll external AI tasks without holding a DB session while sleeping.

    The pending map stores only primitive ids/metadata. Network calls and sleeps
    run outside SQLAlchemy sessions; short sessions are opened only to record a
    terminal segment state.
    """
    while pending:
        for task_id in list(pending):
            meta = pending[task_id]
            try:
                data = kie.get_task(task_id)
            except Exception as exc:  # transient - retry next round
                log.warning("get_task(%s) transient error: %s", task_id, exc)
                continue

            state = (data.get("state") or "").lower()

            if state == "success":
                url = _parse_result_url(data)
                if not url:
                    with get_session() as poll_session:
                        rs = poll_session.get(RunSegment, meta["rs_id"])
                        _skip_segment(rs, "success but no result url", poll_session)
                    del pending[task_id]
                    continue

                result_dst = os.path.join(results_dir, f"result_{meta['index']:04d}.mp4")
                kie.download_result(url, result_dst)
                with get_session() as poll_session:
                    rs = poll_session.get(RunSegment, meta["rs_id"])
                    rs.seedance_result_url = url
                    rs.local_result_path = result_dst
                    # Clear any interim "attempt N failed; retrying" message now
                    # that the segment has succeeded.
                    rs.error_message = None
                    # A real swap landed — this segment is no longer standing in
                    # with its original footage.
                    rs.source_fallback = False
                    transition(rs, SegmentStatus.completed)
                    poll_session.commit()
                log.info("RunSegment idx %d completed", meta["index"])
                del pending[task_id]
            elif state == "fail":
                msg = data.get("failMsg") or data.get("failCode") or "unknown"
                attempt = meta.get("attempt", 1)
                recipe = meta.get("recipe")

                # Transient backend failure ("Internal Error, Please try again
                # later.") → resubmit the same already-uploaded clip up to
                # RUN_TASK_MAX_ATTEMPTS total attempts before giving up.
                if recipe and attempt < RUN_TASK_MAX_ATTEMPTS:
                    try:
                        new_task_id = _create_swap_task(kie=kie, **recipe)
                    except Exception as exc:
                        log.warning(
                            "resubmit failed for seg %d (attempt %d): %s",
                            meta["index"], attempt, exc,
                        )
                        new_task_id = None

                    if new_task_id:
                        with get_session() as poll_session:
                            rs = poll_session.get(RunSegment, meta["rs_id"])
                            rs.seedance_task_id = new_task_id
                            rs.error_message = (
                                f"attempt {attempt}/{RUN_TASK_MAX_ATTEMPTS} "
                                f"failed ({msg}); retrying"
                            )
                            poll_session.commit()
                        log.warning(
                            "task %s (seg %d) failed: %s — retry %d/%d as task %s",
                            task_id, meta["index"], msg,
                            attempt + 1, RUN_TASK_MAX_ATTEMPTS, new_task_id,
                        )
                        del pending[task_id]
                        new_meta = dict(meta)
                        new_meta["attempt"] = attempt + 1
                        new_meta["deadline"] = (
                            time.monotonic() + RUN_SKIP_TIMEOUT_SEC
                        )
                        pending[new_task_id] = new_meta
                        continue

                # No recipe (resumed orphan) or attempts exhausted → mark failed.
                # The run will be marked `incomplete` (not stitched) so this
                # segment can be re-run manually.
                with get_session() as poll_session:
                    rs = poll_session.get(RunSegment, meta["rs_id"])
                    _skip_segment(
                        rs,
                        f"failed after {attempt} attempt(s): {msg}",
                        poll_session,
                    )
                log.warning(
                    "task %s (seg %d) failed permanently after %d attempt(s): %s",
                    task_id, meta["index"], attempt, msg,
                )
                del pending[task_id]
            elif time.monotonic() > meta["deadline"]:
                with get_session() as poll_session:
                    rs = poll_session.get(RunSegment, meta["rs_id"])
                    _skip_segment(
                        rs,
                        f"timed out after {RUN_SKIP_TIMEOUT_SEC:.0f}s (last state={state!r})",
                        poll_session,
                    )
                log.warning(
                    "task %s (seg %d) timed out — segment will need a manual re-run",
                    task_id, meta["index"],
                )
                del pending[task_id]
            # else: still waiting/queuing/generating - leave pending

        if pending:
            time.sleep(RUN_POLL_INTERVAL_SEC)


def _final_is_stale(final_path: str, seg_defs, rs_map) -> bool:
    """True if any completed swap result on disk is newer than the final video.

    Belt-and-suspenders for the delivery-only-retry fast path: a result file
    with a newer mtime than final.mp4 means the final does not contain it —
    reusing it would silently deliver the old generation (e.g. after a result
    file was refreshed without a new submit in the same lifecycle).
    """
    try:
        final_mtime = os.path.getmtime(final_path)
    except OSError:
        return True
    for sd in seg_defs:
        if sd.action != "swap":
            continue
        rs = rs_map.get(sd.id)
        if (
            rs is not None
            and rs.local_result_path
            and os.path.exists(rs.local_result_path)
            and os.path.getmtime(rs.local_result_path) > final_mtime
        ):
            return True
    return False


def process_run(
    run_id: str,
    *,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
) -> None:
    """
    Process all swap segments for a Run (concurrent submit + concurrent poll) and
    stitch the final video.

    Transitions: queued → processing → stitching → delivering → done.
    A swap segment that fails or times out is skipped (original clip used); only
    fatal errors (missing source, stitch failure) fail the whole run.
    Phase durations are logged for observability.
    """
    t_run_start = time.monotonic()
    log.info("process_run start: run_id=%s", run_id)

    external_kie = kie is not None
    if kie is None:
        kie = _default_kie()

    with get_session() as session:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        project: VideoProject = session.get(VideoProject, run.project_id)
        if project is None:
            raise ValueError(f"VideoProject not found: {run.project_id}")

        try:
            # queued → processing: normal first-time path.
            # Any other active state (processing/stitching/delivering) means this
            # is an orphan resume — reset to queued first, then advance.
            # The processing → queued edge was added in state_machine.py (TR5b).
            if run.status != RunStatus.queued:
                log.info(
                    "process_run: run %s is in status=%r (not queued) — "
                    "resetting to queued for orphan resume",
                    run_id, run.status,
                )
                try:
                    transition(run, RunStatus.queued)
                except Exception:
                    # Fallback for stitching/delivering which lack a direct →queued
                    # edge: go via failed first.
                    transition(run, RunStatus.failed)
                    transition(run, RunStatus.queued)
                session.commit()
            transition(run, RunStatus.processing)
            session.commit()

            source = project.source_local_path
            if not source or not os.path.exists(source):
                raise FileNotFoundError(
                    f"Source file not found at {source!r} for project {run.project_id}"
                )

            info = media_mod.probe(source)
            width, height, fps = media_mod.get_default_target(info)
            duration_sec = info.duration_sec

            seg_defs: list[SegmentDef] = list(project.segments)
            if not seg_defs:
                raise ValueError(
                    f"Project {run.project_id} has no segments — run analyze_project first"
                )

            # Ensure a RunSegment row exists for every swap SegmentDef (idempotent).
            existing_rs = {rs.segment_def_id: rs for rs in run.run_segments}
            for sd in seg_defs:
                if sd.action == "swap" and sd.id not in existing_rs:
                    session.add(
                        RunSegment(
                            run_id=run_id, segment_def_id=sd.id,
                            index=sd.index, status=SegmentStatus.pending,
                        )
                    )
            session.flush()
            session.refresh(run)
            rs_map = {rs.segment_def_id: rs for rs in run.run_segments}

            c_dir = run_clips_dir(run_id, run.project_id)
            r_dir = run_results_dir(run_id, run.project_id)

            # ---------------------------------------------------------------
            # Reference resolution
            # ---------------------------------------------------------------
            t_ref_start = time.monotonic()
            run_ref_urls = resolve_reference_urls(
                list(run.reference_image_urls or []), kie, gdrive=gdrive
            )
            # Cache per-segment override resolutions to avoid duplicate uploads.
            _override_ref_cache: dict[str, list[str]] = {}
            log.info(
                "run_id=%s ref_resolution=%.1fs refs=%d",
                run_id, time.monotonic() - t_ref_start, len(run_ref_urls),
            )

            # ---------------------------------------------------------------
            # Submit phase — prepare work items (serial) then submit concurrently.
            # ---------------------------------------------------------------
            pending: dict[str, dict] = {}  # task_id -> {rs_id, index, deadline}
            submit_work: list[dict] = []   # segments queued for concurrent submit

            # Should the model write its own audio track? Only Seedance 2.5
            # exposes the switch; for everyone else this value is computed and
            # then dropped by _create_swap_task, which must not send the field.
            #
            # Decided once per run so every segment of a run bills the same
            # request. `audio_mode="original"` means the stitch overlays the
            # source soundtrack on top, so generated audio would be paid for and
            # then discarded — ask for silence. `audio_mode="seedance"` keeps
            # each clip's own audio, so it has to exist.
            run_audio_mode = run.audio_mode or "original"
            generate_audio = run_audio_mode != "original"
            if ai_models.spec_for(run.model).supports_generate_audio:
                log.info(
                    "run_id=%s generate_audio=%s (model=%s audio_mode=%s)",
                    run_id, generate_audio, run.model, run_audio_mode,
                )

            for sd in seg_defs:
                if sd.action != "swap":
                    continue
                rs = rs_map[sd.id]

                # Resume: already completed with a real result → don't resubmit.
                if (
                    rs.status == SegmentStatus.completed
                    and rs.local_result_path
                    and os.path.exists(rs.local_result_path)
                ):
                    log.info("RunSegment %s (idx %d) already completed, skipping submit",
                             rs.id, rs.index)
                    continue

                # Resume / no-rebill: segment not yet completed but has a
                # seedance_task_id (worker crashed during the poll loop).  Check
                # kie.ai first — if the task already succeeded we can download the
                # result without resubmitting, saving a Seedance credit.
                if (
                    rs.status != SegmentStatus.pending
                    and rs.seedance_task_id
                ):
                    task_id = rs.seedance_task_id
                    log.info(
                        "RunSegment %s (idx %d) has existing task_id=%s (status=%s) "
                        "— checking kie.ai before resubmitting",
                        rs.id, rs.index, task_id, rs.status,
                    )
                    try:
                        data = kie.get_task(task_id)
                        state = (data.get("state") or "").lower()
                    except Exception as exc:
                        log.warning(
                            "get_task(%s) failed during resume check: %s — will resubmit",
                            task_id, exc,
                        )
                        state = "unknown"

                    if state == "success":
                        url = _parse_result_url(data)
                        if url:
                            result_dst = os.path.join(
                                r_dir, f"result_{sd.index:04d}.mp4"
                            )
                            try:
                                kie.download_result(url, result_dst)
                                rs.seedance_result_url = url
                                rs.local_result_path = result_dst
                                # Clear any stale failure message from a prior run.
                                rs.error_message = None
                                # A real swap landed — this segment is no longer
                                # standing in with its original footage.
                                rs.source_fallback = False
                                try:
                                    transition(rs, SegmentStatus.completed)
                                except Exception:
                                    rs.status = SegmentStatus.completed
                                session.commit()
                                log.info(
                                    "RunSegment %s (idx %d) task %s was already "
                                    "success — recovered without rebilling",
                                    rs.id, rs.index, task_id,
                                )
                                continue
                            except Exception as exc:
                                log.warning(
                                    "download_result failed for task %s: %s "
                                    "— will resubmit",
                                    task_id, exc,
                                )
                        else:
                            log.warning(
                                "task %s success but no result url — will resubmit",
                                task_id,
                            )
                    elif state not in ("fail",):
                        # Still in-progress or unknown — if we got here via a
                        # restart the task may still be running on kie.ai.
                        # We re-add it to the pending poll set to avoid
                        # re-submitting a task that Seedance is already processing.
                        log.info(
                            "RunSegment %s (idx %d) task %s state=%r — "
                            "resuming poll without resubmitting",
                            rs.id, rs.index, task_id, state,
                        )
                        # Ensure segment is in generating state for the poll loop.
                        if rs.status != SegmentStatus.generating:
                            try:
                                # generating requires submitted→generating path;
                                # force-set status directly since we're resuming.
                                rs.status = SegmentStatus.generating
                            except Exception:
                                pass
                            session.flush()
                        pending[task_id] = {
                            "rs_id": rs.id,
                            "index": sd.index,
                            "deadline": time.monotonic() + RUN_SKIP_TIMEOUT_SEC,
                        }
                        continue
                    # state == "fail" or download failed → fall through to reset+resubmit

                # Retry: reset an interrupted RunSegment before resubmitting.
                if rs.status != SegmentStatus.pending:
                    rs.status = SegmentStatus.pending
                    rs.error_message = None
                    rs.seedance_task_id = None
                    rs.seedance_result_url = None
                    # Clear the deterministic-failure flag too: this segment is
                    # about to be submitted again, possibly after being split or
                    # with a different model, and a stale flag would make the
                    # stitch prefer the original footage over a real swap.
                    rs.source_fallback = False
                    session.flush()

                # Resolve effective refs for this segment (override takes priority).
                if rs.reference_image_urls_override:
                    cache_key = rs.id
                    if cache_key not in _override_ref_cache:
                        _override_ref_cache[cache_key] = resolve_reference_urls(
                            list(rs.reference_image_urls_override), kie, gdrive=gdrive
                        )
                    effective_ref_urls = _override_ref_cache[cache_key]
                else:
                    effective_ref_urls = run_ref_urls

                # Queue work item for concurrent submit (primitive data only —
                # no ORM objects, safe to pass across thread boundaries).
                submit_work.append({
                    "rs_id": rs.id,
                    "sd_start_sec": sd.start_sec,
                    "sd_end_sec": sd.end_sec,
                    "sd_pre_roll_sec": sd.pre_roll_sec,
                    "sd_post_roll_sec": sd.post_roll_sec,
                    "sd_index": sd.index,
                    "run_model": run.model or "seedance",
                    "run_prompt": run.prompt,
                    "run_resolution": run.resolution,
                    "project_aspect_ratio": project.aspect_ratio,
                    "project_width": project.width,
                    "project_height": project.height,
                    "source": source,
                    "duration_sec": duration_sec,
                    "clip_dst": os.path.join(c_dir, f"clip_{sd.index:04d}.mp4"),
                    "ref_urls": list(effective_ref_urls),
                    "prompt_override": rs.prompt_override,
                    "mute_audio": bool(project.mute_source),
                    "generate_audio": generate_audio,
                })

            # Commit so that newly created RunSegment rows are visible to the
            # independent DB sessions opened by each submit thread.
            # (flush() only writes within the current transaction; other sessions
            # cannot see uncommitted rows.)
            session.commit()

            # Concurrent submit via thread pool.
            t_submit_start = time.monotonic()
            if submit_work:
                log.info(
                    "run_id=%s submit_phase: submitting %d segment(s) "
                    "concurrency=%d",
                    run_id, len(submit_work), SUBMIT_CONCURRENCY,
                )
                with ThreadPoolExecutor(max_workers=SUBMIT_CONCURRENCY) as pool:
                    submit_futures = [
                        (
                            work,
                            pool.submit(
                                _submit_swap_segment_isolated,
                                **work,
                                kie=kie,
                                kie_factory=(
                                    None if external_kie else _default_kie
                                ),
                            ),
                        )
                        for work in submit_work
                    ]
                # Collect results; re-raise on first failure (preserves existing
                # serial semantics: any submit failure fails the whole run).
                for work, fut in submit_futures:
                    result = fut.result()
                    if result is None:
                        # Segment skipped at submit time — deterministically
                        # un-generatable on this model (too long for its
                        # max_clip_sec, or a source shorter than its
                        # min_clip_sec). It was already marked failed with
                        # source_fallback set, so the completeness gate lets the
                        # run through and the stitch uses the original footage.
                        continue
                    pending[result["task_id"]] = {
                        "rs_id": work["rs_id"],
                        "index": work["sd_index"],
                        "deadline": time.monotonic() + RUN_SKIP_TIMEOUT_SEC,
                        "attempt": 1,
                        "recipe": result["recipe"],
                    }

            log.info(
                "run_id=%s submit_phase_total=%.1fs segments_submitted=%d",
                run_id, time.monotonic() - t_submit_start, len(submit_work),
            )

            # Did we (re)submit anything this run? If not, and a final video
            # already exists, this is a delivery-only retry → skip the re-stitch.
            did_submit = bool(pending)
            log.info("Submitted %d swap task(s) to Seedance for run %s",
                     len(pending), run_id)

            # ---------------------------------------------------------------
            # Poll phase — round-robin over all pending tasks, act per task.
            # ---------------------------------------------------------------
            t_poll_start = time.monotonic()
            session.commit()
            session.close()
            try:
                _poll_pending_tasks(pending=pending, kie=kie, results_dir=r_dir)
            except Exception as exc:
                with get_session() as fail_session:
                    failed_run = fail_session.get(Run, run_id)
                    if failed_run and failed_run.status not in (
                        RunStatus.done,
                        RunStatus.failed,
                    ):
                        try:
                            transition(failed_run, RunStatus.failed)
                        except Exception:
                            failed_run.status = RunStatus.failed
                    if failed_run and not failed_run.error_message:
                        failed_run.error_message = str(exc)
                    fail_session.commit()
                raise
            run = session.get(Run, run_id)
            project = session.get(VideoProject, run.project_id)
            seg_defs = list(project.segments)
            rs_map = {rs.segment_def_id: rs for rs in run.run_segments}

            log.info(
                "run_id=%s poll_phase_total=%.1fs",
                run_id, time.monotonic() - t_poll_start,
            )

            # ---------------------------------------------------------------
            # Completeness gate — never stitch a mix of swapped + original clips,
            # EXCEPT where the mix is deliberate and unavoidable.
            # If any swap segment did not complete (failed after retries, timed
            # out, or was skipped), stop here and mark the run `incomplete`. The
            # completed segments' results stay on disk; the operator re-runs the
            # failed segment(s) and the full video is stitched only once every
            # swap segment is completed.
            #
            # A segment carrying `source_fallback` could not be generated by this
            # run's model for a deterministic reason (it is longer than the
            # model's max_clip_sec, or the source is shorter than its
            # min_clip_sec), so retrying is futile and blocking would block
            # forever. Those pass the gate and the stitch substitutes the
            # original footage for them; everything else — transient task
            # failures, timeouts — still blocks, because a retry there produces
            # the real swap. See _skip_segment.
            # ---------------------------------------------------------------
            incomplete_idx: list[int] = []
            fallback_idx: list[int] = []
            for sd in seg_defs:
                if sd.action != "swap":
                    continue
                rs = rs_map.get(sd.id)
                if rs is not None and rs.source_fallback:
                    fallback_idx.append(sd.index)
                    continue
                if (
                    rs is None
                    or rs.status != SegmentStatus.completed
                    or not rs.local_result_path
                    or not os.path.exists(rs.local_result_path)
                ):
                    incomplete_idx.append(sd.index)

            if incomplete_idx:
                msg = (
                    f"{len(incomplete_idx)} swap segment(s) did not complete: "
                    f"{incomplete_idx}. Re-run them — the full video is stitched "
                    "only once every segment succeeds."
                )
                transition(run, RunStatus.incomplete)
                run.error_message = msg
                session.commit()
                log.warning("run_id=%s incomplete — %s", run_id, msg)
                return

            # ---------------------------------------------------------------
            # Stitch — unless this is a delivery-only retry: when nothing was
            # (re)processed this run AND a final video already exists on disk,
            # reuse it and skip the expensive re-encode (e.g. a Retry after the
            # Drive upload timed out). We still pass through the `stitching`
            # state so the run's state-machine path stays valid.
            # ---------------------------------------------------------------
            t_stitch_start = time.monotonic()
            transition(run, RunStatus.stitching)
            session.commit()
            final_dst = os.path.join(r_dir, "final.mp4")
            reuse_final = (
                not did_submit
                and run.result_local_path
                and os.path.exists(run.result_local_path)
                and not _final_is_stale(run.result_local_path, seg_defs, rs_map)
            )
            if reuse_final:
                final_dst = run.result_local_path
                log.info(
                    "No segments reprocessed and final video exists — skipping "
                    "re-stitch (delivery-only retry): %s", final_dst,
                )
            else:
                # Assemble clips in order. Keep segments are cut from the source;
                # swap segments use their completed AI result. The completeness
                # gate above guarantees every swap segment is completed by now, so
                # a non-completed swap here is a logic error — raise rather than
                # silently substitute the original clip (which would produce the
                # swapped+original mix we explicitly want to avoid).
                #
                # PARALLELISM: keep-segment cuts are independent ffmpeg calls with
                # no shared state.  We run them concurrently (up to
                # STITCH_CUT_CONCURRENCY) using a thread pool, then collect results
                # in the original seg_defs order so the stitch list is always
                # correctly ordered.

                def _cut_source(sd: SegmentDef) -> str:
                    """Cut this segment's ORIGINAL footage out of the source video.

                    Shared by "keep" segments and by swap segments falling back to
                    the source, so both produce a byte-identical clip at the same
                    path — only one of the two branches can apply to a given index.
                    """
                    dst = os.path.join(c_dir, f"clip_{sd.index:04d}.mp4")
                    media_mod.cut_clip(source, sd.start_sec, sd.end_sec, dst)
                    return dst

                def _cut_or_lookup(sd: SegmentDef) -> str:
                    """Return the clip path for this segment (cut if needed)."""
                    if sd.action == "keep":
                        return _cut_source(sd)
                    rs = session.get(RunSegment, rs_map[sd.id].id)
                    # Deterministically un-generatable on this model: the gate let
                    # the run through on the understanding that we substitute the
                    # original footage here. Timing is exact — the cut spans the
                    # segment's own start/end — so the soundtrack stays in sync
                    # either way.
                    if rs.source_fallback:
                        log.info(
                            "segment %d: using original footage (swap skipped: %s)",
                            sd.index, rs.error_message,
                        )
                        return _cut_source(sd)
                    # Otherwise the segment must be completed (guaranteed by the gate).
                    if (
                        rs.status == SegmentStatus.completed
                        and rs.local_result_path
                        and os.path.exists(rs.local_result_path)
                    ):
                        return rs.local_result_path
                    raise RuntimeError(
                        f"swap segment {sd.index} is not completed at stitch time "
                        f"(status={rs.status}) — refusing to mix in the original clip"
                    )

                # Determine which segments need ffmpeg work (cuts) vs which are
                # already-available result files.  Only segments that require a
                # cut_clip call benefit from concurrency; already-available results
                # are returned immediately.
                #
                # We submit ALL seg_defs (keeps + fallbacks) to the pool and track
                # them by insertion order using an ordered list of (index, future).
                ordered_futures: list[tuple[int, object]] = []  # (sd.index, Future)
                with ThreadPoolExecutor(max_workers=STITCH_CUT_CONCURRENCY) as pool:
                    for sd in seg_defs:
                        fut = pool.submit(_cut_or_lookup, sd)
                        ordered_futures.append((sd.index, fut))

                # Collect results in seg_defs order.  If any future raised an
                # exception, it will re-raise here, marking the run failed (outer
                # except clause handles that).
                clip_paths: list[str] = []
                for _idx, fut in ordered_futures:
                    clip_paths.append(fut.result())  # type: ignore[union-attr]

                log.info("Stitching %d clips → %s (%dx%d @ %.2ffps)",
                         len(clip_paths), final_dst, width, height, fps)
                audio_mode = run.audio_mode if run.audio_mode else "original"
                # A model that produces no audio at all (Gemini Omni: its clips
                # are sent video-only and it returns silent video) leaves the
                # original source track as the only sensible soundtrack. Force
                # "original" even if the run requested "seedance".
                if not ai_models.spec_for(run.model).produces_audio:
                    audio_mode = "original"
                media_mod.stitch(
                    clip_paths, audio_source=source, dst=final_dst,
                    width=width, height=height, fps=fps,
                    audio_mode=audio_mode,
                )
                run.result_local_path = final_dst

            # Record how much of the delivered video is original footage rather
            # than a generated swap. Written on every stitch (not only when
            # non-zero) so 0 positively means "checked, fully swapped" and NULL
            # keeps meaning "not recorded".
            run.source_fallback_segments = len(fallback_idx)
            session.flush()

            log.info(
                "run_id=%s stitch_phase_total=%.1fs reuse=%s",
                run_id, time.monotonic() - t_stitch_start, reuse_final,
            )

            # ---------------------------------------------------------------
            # Deliver
            # ---------------------------------------------------------------
            t_deliver_start = time.monotonic()
            transition(run, RunStatus.delivering)
            session.commit()
            folder_id = run.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID
            if folder_id:
                if gdrive is None:
                    gdrive = _default_gdrive()
                result = _deliver_with_retry(gdrive, final_dst, folder_id, run_id)
                run.result_gdrive_file_id = result.get("id")
                log.info(
                    "run_id=%s deliver_phase=%.1fs gdrive_file=%s",
                    run_id, time.monotonic() - t_deliver_start, result.get("id"),
                )
                session.flush()

            transition(run, RunStatus.done)
            session.commit()
            t_total = time.monotonic() - t_run_start
            log.info(
                "process_run done: run_id=%s total=%.1fs",
                run_id, t_total,
            )

        except Exception as exc:
            if run.status not in (RunStatus.done, RunStatus.failed):
                try:
                    transition(run, RunStatus.failed)
                except Exception:
                    run.status = RunStatus.failed
            if not run.error_message:
                run.error_message = str(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise


def _skip_segment(
    rs: RunSegment, reason: str, session, *, source_fallback: bool = False
) -> None:
    """Mark a RunSegment failed, with *reason* recorded for the UI.

    Two different outcomes, chosen by *source_fallback* — the distinction is the
    whole point, so pass it deliberately rather than by default:

    ``source_fallback=False`` (default) — a TRANSIENT failure: task error,
    timeout, missing result url. The completeness gate in :func:`process_run`
    requires every swap segment to be ``completed``, so the run is left
    ``incomplete`` with no delivered video and the operator retries. That is
    correct here: a retry produces the real swap, and shipping the original
    footage instead would silently throw it away.

    ``source_fallback=True`` — a DETERMINISTIC failure the run's model can never
    satisfy: the segment is longer than the model's ``max_clip_sec``, or the
    source is shorter than its ``min_clip_sec``. Retrying cannot help, so
    blocking would block forever. The gate lets the run through and the stitch
    substitutes the original source footage for this segment (safe on the
    timeline: the original clip has exactly the segment's duration, so nothing
    desyncs against the soundtrack — unlike a partial swap, which is why we still
    never do that). The delivered video is then only partly swapped, which
    ``Run.source_fallback_segments`` surfaces; the operator's fix is to split the
    segment or pick a model that can generate it.

    The status is ``failed`` either way — the swap genuinely did not happen.
    """
    rs.error_message = reason
    rs.source_fallback = source_fallback
    if rs.status != SegmentStatus.failed:
        try:
            transition(rs, SegmentStatus.failed)
        except Exception:
            rs.status = SegmentStatus.failed
    session.commit()
