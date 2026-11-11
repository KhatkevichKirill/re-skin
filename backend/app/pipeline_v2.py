"""
pipeline_v2.py — End-to-end processing for the v2 Project → Runs model.

Public functions
----------------
analyze_project(project_id, *, detector=None)
    Download/locate the source, probe it, propose segments, persist to DB.
    Transitions: created → analyzing → ready.

transcribe_project(project_id, *, kie=None)
    Localisation side channel: transcribe the source into VideoProject
    .transcript / .transcript_status / .transcript_error. Runs as its own RQ
    job AFTER analysis and never raises on a provider failure — a Gemini outage
    must not fail analysis or block segmentation (docs/localisation.md §8).

submit_run(run_id, *, kie=None, gdrive=None) -> dict | None
    Submit phase: cut + upload + create AI tasks for all swap segments.
    Returns the pending-task map {task_id → meta} for the poll phase, or {}
    if all segments are already completed (delivery-only retry), or None if the
    run is in a terminal state.
    Transitions: queued → processing.

preprocess_reference_images(run_id, *, kie=None, gdrive=None, session=None) -> dict
    Optional GPT Image 2 first phase. Mode-aware (``Run.preprocess_mode``):
    ``per_image`` (default) creates one task per ref with a prompt;
    ``all_as_refs`` creates ONE task with all refs as input. Returns
    ``{task_id: ref_index}`` for the poll phase. See
    docs/gpt-image-2-preprocess.md.

poll_preprocess_once(run_id, *, pending, kie=None, gdrive=None) -> dict
    One round of GPT Image 2 task polling; downloads results into
    Run.preprocessed_reference_urls. Returns ``{"pending": ..., "failed": ...}``
    — on failure the run is marked failed (no fallback to originals).

poll_run_once(run_id, *, pending, kie=None) -> dict
    Poll phase: perform ONE round of status checks on the pending-task map.
    Downloads results for completed tasks, retries or skips failed tasks.
    Returns the *remaining* pending map (empty when all tasks are terminal).
    Callers (tasks.py) re-enqueue this function if the result is non-empty.

finalize_run(run_id, *, gdrive=None, publication_id=None) -> Optional[PublishedFinal]
    Finalize phase: completeness gate + stitch (ffmpeg) + deliver to Drive.
    Transitions: processing → stitching → delivering → done (or incomplete/failed).

process_run(run_id, *, kie=None, gdrive=None) -> None
    Monolithic entry point kept for backward compatibility (tests + recovery).
    Delegates to submit_run → poll-loop → finalize_run in the same call.

Shared helpers (resolve_reference_urls, clamp_duration_for) are imported from
pipeline_common.py — single source of
truth. Per-model capabilities (kie id, resolutions, clip ceiling, duration
rules, audio switch) come from app/ai_models.py; nothing in this module should
carry its own copy of them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid as uuid_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Tuple

from . import ai_models
from .knowledge import capture as kcap
from .knowledge.publication import PublishedFinal
from . import localisation as localisation_mod
from . import media as media_mod
from . import face as face_mod
from . import project_types
from . import rebuild as rebuild_mod
from . import scene as scene_mod
from . import spend as spend_mod
from . import paid_attempts
from .paid_attempts_models import (
    OWNER_RUN_PREPROCESS,
    OWNER_RUN_SEGMENT,
    PARKED as ATTEMPT_PARKED,
    STARTED as ATTEMPT_STARTED,
)
from .config import settings
from .db import get_session
from .gdrive_client import GDriveClient
from .kie_client import KieClient, KieResultGone, get_shared_client
from sqlalchemy import update as sa_update

from .models import LibraryClip, Run, RunSegment, SegmentDef, VideoProject
from .project_types import spec_for
from .pipeline_common import (
    _map_omni_aspect,
    _omni_resolution,
    _snap_omni_duration,
    clamp_duration_for,
    resolve_reference_urls,
)
from .state_machine import ProjectStatus, RunStatus, SegmentStatus, transition
from .storage import (
    keep_clip_path,
    project_source_path,
    run_clips_dir,
    run_preprocessed_dir,
    run_results_dir,
    scene_cuts_cache_path,
)

log = logging.getLogger(__name__)

# Parallel-submit tuning (env-overridable).
# Seedance tasks are submitted all at once and polled concurrently; a task that
# yields no result within RUN_SKIP_TIMEOUT_SEC is marked failed, which leaves
# the run incomplete — a timeout is transient, so it blocks rather than falling
# back to the original footage (see _skip_segment).
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

# Poll-phase download parallelism (Fix #16): max concurrent download_result calls
# when multiple AI tasks complete in the same poll round.  Each download thread
# opens its own DB session so session usage is thread-safe.  A small pool (2)
# avoids saturating the uplink; set higher if bandwidth allows.
POLL_DOWNLOAD_CONCURRENCY = int(os.getenv("POLL_DOWNLOAD_CONCURRENCY", "2"))


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


# ---------------------------------------------------------------------------
# analyze_project
# ---------------------------------------------------------------------------


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
    return [
        face_mod.ProposedSegment(
            start_sec=s,
            end_sec=e,
            has_face=False,
            action="swap",
        )
        for s, e in _scene_ranges(
            video_path,
            duration_sec=duration_sec,
            fps=fps,
            max_segment_sec=max_segment_sec,
        )
    ]


def _scene_ranges(
    video_path: str,
    *,
    duration_sec: float,
    fps: float | None,
    max_segment_sec: float,
) -> list[tuple[float, float]]:
    """PySceneDetect cuts as ``(start, end)`` ranges, each <= *max_segment_sec*.

    The frame-reading half of :func:`_scene_segments`.
    """
    ranges = scene_mod.detect_scenes(
        video_path, fps=fps, min_scene_len_sec=settings.SCENE_MIN_LEN_SECONDS
    )
    return _clamp_and_cap_scene_ranges(
        ranges, duration_sec=duration_sec, max_segment_sec=max_segment_sec
    )


def _clamp_and_cap_scene_ranges(
    ranges: list[tuple[float, float]],
    *,
    duration_sec: float,
    max_segment_sec: float,
) -> list[tuple[float, float]]:
    """Clamp detector output into ``[0, duration]`` and split long scenes evenly.

    A single-shot video (no cuts found) becomes one full-length range. Shared
    by subtitle-removal analysis and :func:`whole_video_ranges`.
    """
    if not ranges:
        ranges = [(0.0, duration_sec)]

    # Clamp to the probed duration (the last scene's end can round slightly past
    # it) so the partition stays within [0, duration] before the cap-split.
    clamped = [
        (max(0.0, s), min(e, duration_sec)) for s, e in ranges if e - s > 1e-6
    ]
    return face_mod.split_max_duration(clamped, max_sec=max_segment_sec)


# ---------------------------------------------------------------------------
# Whole-video localisation (VideoProject.localisation_scope == "whole", AICP-340)
# ---------------------------------------------------------------------------

# Length of the even pieces a long scene is first cut into before the greedy
# merge in whole_video_ranges rebuilds segments up to the target. Smaller than
# the target on purpose: the pieces are CANDIDATE boundaries, and finer
# candidates let a short scene next to a long one absorb part of it instead of
# being left as a lone 5s clip. 10s is what the operator's hand cut of the
# driving project (5b1b9b67 -> localisation 46cc6443) was built from — the
# subtitle-removal pass of the same video, segmented at the 10s default cap —
# so the rule reproduces that cut exactly (tests/test_whole_video_localisation.py).
WHOLE_VIDEO_PIECE_SEC = 10.0


def whole_video_target_sec(max_segment_sec: float | None) -> float:
    """Longest swap segment a whole-video localisation analysis proposes.

    ``settings.LOCALISATION_WHOLE_VIDEO_MAX_SEC`` (14.72s, the length Seedance
    2.5 intermittently truncates its output to), never above the shortest
    ``max_clip_sec`` among the models that can localise at all (the ones that
    produce audio — the New Run form offers nothing else on a localisation
    project), and never above an EXPLICIT project segmentation cap.

    A NULL ``max_segment_sec`` is deliberately NOT read as the universal 10s
    default here: that default exists so a segment fits Gemini Omni, which
    cannot localise (it emits no audio). An explicit smaller cap is still
    honoured — it is the operator's way of committing to shorter clips.
    """
    audio_caps = [
        spec.max_clip_sec
        for spec in ai_models.V2V_MODELS.values()
        if spec.produces_audio
    ]
    target = float(settings.LOCALISATION_WHOLE_VIDEO_MAX_SEC)
    if audio_caps:
        target = min(target, min(audio_caps))
    if max_segment_sec is not None:
        target = min(target, float(max_segment_sec))
    target = min(target, ai_models.ABSOLUTE_MAX_SEGMENT_SEC)
    if not target > 0.0:
        # Only reachable from a hand-edited row or env (the API validates
        # max_segment_sec); a non-positive cap would divide by zero in the
        # even split. Degrade to the shipped default rather than fail analysis.
        log.warning(
            "whole_video_target_sec: non-positive target %r (max_segment_sec=%r, "
            "LOCALISATION_WHOLE_VIDEO_MAX_SEC=%r); using 14.72s",
            target, max_segment_sec, settings.LOCALISATION_WHOLE_VIDEO_MAX_SEC,
        )
        target = 14.72
    return target


def whole_video_floor_sec() -> float:
    """Shortest swap segment a whole-video localisation may have: the largest
    ``min_clip_sec`` among the models that produce audio (Seedance 2.5 refuses
    clips under 4s). The same floor :func:`whole_video_ranges` rebalances its
    short tail against; read by the speech-pause re-cut (AICP-347)."""
    return max(
        (spec.min_clip_sec for spec in ai_models.V2V_MODELS.values()
         if spec.produces_audio),
        default=0.0,
    )


def whole_video_ranges(
    scene_ranges: list[tuple[float, float]],
    *,
    duration_sec: float,
    target_sec: float,
) -> list[tuple[float, float]]:
    """Partition ``[0, duration]`` into swap ranges for a whole-video localisation.

    Pure (no frames read) so the rule is testable against a fixture of
    detected cuts. Three steps:

    1. **Candidates.** The detected scenes, clamped to the video, with any
       scene longer than ``min(WHOLE_VIDEO_PIECE_SEC, target)`` split into
       equal pieces — exactly what subtitle-removal analysis proposes at the
       default cap.
    2. **Greedy merge.** Left to right, a candidate joins the current segment
       while the merged length stays <= *target_sec*; otherwise it starts a
       new one. Fewest seams for the target, and every boundary is a scene
       cut or an even split of a long scene. A merge may cross a scene cut —
       the cuts are candidates, not obligations.
    3. **Short tail.** The last scene can be shorter than PySceneDetect's
       minimum (the minimum only binds between cuts), and Seedance 2.5 refuses
       clips under its ``min_clip_sec``. A final segment under the largest
       ``min_clip_sec`` among audio models that the merge could not absorb is
       rebalanced with its predecessor into two equal halves.

    The result is contiguous, starts at 0 and ends at *duration_sec* (a
    detector that stops short of the probed duration has the remainder added
    to its last scene before anything is split).
    """
    duration = float(duration_sec)
    target = float(target_sec)
    piece = min(WHOLE_VIDEO_PIECE_SEC, target)
    scenes = [(float(s), float(e)) for s, e in scene_ranges]
    # PySceneDetect's last scene ends on the last decoded frame, which can sit a
    # frame or two before the probed duration (5b1b9b67: 78.833 vs 78.878).
    # Every second of a whole-video localisation is re-spoken, so whatever
    # follows the last detected scene belongs to it — extended BEFORE the even
    # split, so the remainder can never become a segment of its own.
    if scenes and scenes[-1][1] < duration:
        scenes[-1] = (scenes[-1][0], duration)
    pieces = _clamp_and_cap_scene_ranges(
        scenes, duration_sec=duration, max_segment_sec=piece
    )
    merged: list[tuple[float, float]] = []
    for s, e in pieces:
        if merged and e - merged[-1][0] <= target + 1e-6:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    floor = max(
        (spec.min_clip_sec for spec in ai_models.V2V_MODELS.values()
         if spec.produces_audio),
        default=0.0,
    )
    if len(merged) >= 2 and merged[-1][1] - merged[-1][0] < floor - 1e-6:
        (ps, _pe), (_ls, le) = merged[-2], merged[-1]
        # The merge already refused (ps, le) as one segment, so it is longer
        # than the target but shorter than target + floor; each half is then
        # <= target whenever target >= floor (true for every audio model).
        mid = ps + (le - ps) / 2.0
        merged[-2:] = [(ps, mid), (mid, le)]
    return merged


def _whole_video_segments(
    video_path: str,
    *,
    duration_sec: float,
    fps: float | None,
    target_sec: float,
    scene_cache_path: str | None = None,
) -> list:
    """All-swap ProposedSegments at scene cuts for a whole-video localisation.

    With *scene_cache_path* the detected cuts are also recorded for the source
    (``scene.write_scene_cache``, best effort) so "Re-cut at speech pauses"
    (AICP-347) can prefer them without decoding the video again. The cut
    itself does not depend on the cache.
    """
    detected = scene_mod.detect_scenes(
        video_path, fps=fps, min_scene_len_sec=settings.SCENE_MIN_LEN_SECONDS
    )
    if scene_cache_path:
        scene_mod.write_scene_cache(
            scene_cache_path,
            source_path=video_path,
            ranges=detected,
            min_scene_len_sec=settings.SCENE_MIN_LEN_SECONDS,
        )
    return [
        face_mod.ProposedSegment(
            start_sec=s, end_sec=e, has_face=False, action="swap",
        )
        for s, e in whole_video_ranges(
            detected, duration_sec=duration_sec, target_sec=target_sec
        )
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


def effective_segment_cap_sec(max_segment_sec: float | None) -> float:
    """The segmentation cap analysis will actually apply to a project.

    The stored column is a *preference*: NULL means "no explicit choice" and
    resolves to the smallest per-clip ceiling in the registry, and any value is
    bounded by the largest ceiling any model offers. Both rules live in
    :func:`analyze_project`; this is the same arithmetic exposed as a function
    so a reader outside the pipeline (the project page, which has to show the
    operator which of the two knobs clipped their hook) resolves the cap the
    way analysis will rather than approximating it.

    NULL is tested with ``is None``, not ``or`` — 0.0 is falsy but it is a
    *value*, and a project whose cap really is 0 should be shown as broken, not
    quietly redrawn as the default (the same reasoning analyze_project spells
    out at its own `project_cap is None` check).
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
    scope: str | None = None,
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

    *scope* is ``VideoProject.localisation_scope``. In whole-video scope
    (AICP-340) there is no hook to clamp: analysis re-speaks the whole video
    in scene-cut segments, so ``effective`` is the duration (the requested
    value when it is not probed yet) and ``reason`` is always
    ``HOOK_CLAMP_NONE``. Any other value — including None, the historical
    callers — is the hook scope, unchanged.
    """
    # NULL hook_sec → the configured default. Tested with `or`, not `is None`,
    # per docs/localisation.md §5: unlike max_segment_sec (where 0.0 is a
    # meaningful, if wrong, operator choice worth surfacing) a 0-second hook is
    # never anything but a mistake, and degrading it to the default beats
    # clamping it up into a 4-second hook nobody asked for.
    requested = float(hook_sec or settings.LOCALISATION_DEFAULT_HOOK_SEC)

    floor = min(spec.min_clip_sec for spec in ai_models.V2V_MODELS.values())
    cap = float(max_segment_sec)
    duration = None if duration_sec is None else float(duration_sec)
    ceiling = cap if duration is None else min(cap, duration)

    # max() first, then min(): on a source shorter than the floor (a 3s upload)
    # the duration bound wins, so the hook is the whole video rather than a
    # segment that runs off the end of it.
    if scope == project_types.LOCALISATION_SCOPE_WHOLE:
        return {
            "requested": requested,
            "effective": requested if duration is None else duration,
            "floor": floor,
            "cap": cap,
            "duration": duration,
            "reason": HOOK_CLAMP_NONE,
            "defaulted": not hook_sec,
            "scope": project_types.LOCALISATION_SCOPE_WHOLE,
        }

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
        "scope": project_types.LOCALISATION_SCOPE_HOOK,
    }


def _hook_split_segments(
    *,
    duration_sec: float,
    hook_sec: float | None,
    max_segment_sec: float,
) -> list:
    """Build the localisation partition: swap over the hook, keep over the rest.

    A ``localisation`` project re-generates only the **hook** — the opening
    seconds whose speech is re-spoken in the target language. Everything after
    it is thrown away at run time and replaced by a pre-uploaded,
    language-matched tail from the clip library, so the tail half of the source
    is proposed as a single ``keep`` segment: it exists to be excluded from the
    stitch plan, not to be generated (docs/localisation.md §1, §5).

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


def analyze_project(
    project_id: str, *, detector=None, analysis_token: Optional[str] = None,
) -> None:
    """
    Probe the source video and propose segments for a VideoProject.

    Localisation scope (AICP-340): the scope applied is
    ``VideoProject.localisation_scope_requested`` when set (written by
    ``POST /projects/{pid}/analyze?localisation_scope=``), else the current
    ``localisation_scope``. It is written to ``localisation_scope`` — and the
    request cleared — in the SAME commit as the new SegmentDefs, so the stored
    scope always describes the segments on the row: while the job is queued
    the project keeps its old scope AND its old segments, and a failed or
    crashed analysis changes neither and leaves the request for the retry.

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

        is_rebuild_project = (
            spec_for(project.project_type).segmentation == project_types.SEG_SHOT_CHUNKS
        )
        rebuild_gen0 = None
        rebuild_token = None
        if is_rebuild_project:
            # Fresh state under the project row lock (the lock every rebuild
            # admission and re-analysis takes), and the shot generation this
            # job starts from: publishing and failing below are fenced on it,
            # so a stale or duplicate job can never undo another job's publish
            # (Terra code gate round 2).
            from . import rebuild_analysis

            session.refresh(project, with_for_update=True)
            rebuild_gen0 = rebuild_analysis.current_generation(session, project_id)
        if is_rebuild_project and not (
            # A first analysis (the job takes the status itself), or the job
            # of the admission that wrote this token (Terra round 4: the token
            # travels in the job's arguments, so an old queued job can never
            # adopt a newer admission's token).
            (project.status == ProjectStatus.created and analysis_token is None)
            or (
                project.status == ProjectStatus.analyzing
                and analysis_token is not None
                and project.rebuild_analysis_token == analysis_token
            )
        ):
            # A rebuild analysis only runs for a project its admission put in
            # `analyzing` (re-analysis, under the project lock, after the
            # live-row checks) or a brand-new `created` one. A stale or
            # duplicate job finding `ready`/`failed` must not flip the project
            # itself: that would skip the live-row guard and replace shots
            # under a storyboard/keyframe job admitted meanwhile (design
            # §5.1.7-8). Nothing to do — the operator re-analyzes.
            log.warning(
                "analyze_project %s: rebuild project is %s — stale analysis job, "
                "nothing done", project_id, project.status,
            )
            return
        try:
            # created → analyzing (commit so readers see live state). The
            # admission normally flipped it already (rebuild re-analysis under
            # the project row lock, design §5.1.8; every other admission since
            # AICP-350).
            started_created = project.status == ProjectStatus.created
            # AICP-350: every admission (create, re-analysis, extract-insert,
            # project-from-run) now commits `analyzing` itself so the deploy
            # drain counts it; a job from an older admission path still finds
            # created/ready/failed and takes the status here.
            if project.status != ProjectStatus.analyzing:
                transition(project, ProjectStatus.analyzing)
            if is_rebuild_project:
                # The analysis this job runs under: the admission's token, or a
                # new one when this job leaves `created` itself.
                if started_created:
                    project.rebuild_analysis_token = str(uuid_mod.uuid4())
                rebuild_token = project.rebuild_analysis_token
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

            # Normalize oversized / 10-bit sources to the 1080p 8-bit baseline
            # the worker memory sizing assumes (no-op for compliant files).
            normalized = media_mod.normalize_source(local)
            if normalized != local:
                local = normalized
                project.source_local_path = local
                session.commit()

            # Probe.
            info = media_mod.probe(local)
            project.duration_sec = info.duration_sec
            project.width = info.width
            project.height = info.height
            project.fps = info.fps
            project.aspect_ratio = info.aspect_ratio
            session.commit()

            # Propose segments — create SegmentDef rows. The strategy depends on
            # the project type (see app/project_types.py):
            #
            # How long a segment may be is the PROJECT's choice, not a global
            # constant. The model is picked per-Run and a project's segmentation
            # is reused across every run, so the cap decides which models this
            # project can still be run on:
            #   - NULL (the default) → ai_models.UNIVERSAL_MAX_SEGMENT_SEC, the
            #     smallest ceiling in the registry (10s, Gemini Omni). Every
            #     segment then fits every model — the historical behaviour, and
            #     what all pre-existing projects keep.
            #   - a larger value → opts into longer segments to exploit models
            #     that allow them (Seedance 2.5 generates up to 30s per clip:
            #     fewer stitch seams, less identity drift across cuts). The
            #     trade-off is real: a run on a shorter-limit model marks every
            #     segment it cannot generate as failed (the guard in
            #     _submit_swap_segment_isolated), and the completeness gate then
            #     refuses to stitch — that run yields NO video, it does not
            #     quietly deliver the un-swapped original for those segments.
            #     Raising the cap is therefore a commitment to a model tier.
            # ABSOLUTE_MAX_SEGMENT_SEC (the largest ceiling any model offers) is
            # the safety bound: a cap above it could not be generated by
            # anything, so clamp rather than propose unusable segments.
            #
            # NULL is tested with `is None`, not `or`: 0.0 is falsy but it is a
            # *value*, and silently reinterpreting it as "no choice → 10s" would
            # hide a bad row written by a scripts/ one-off or a manual UPDATE
            # (the REST API cannot produce it — see _validate_max_segment_sec,
            # which is equally scrupulous that None means "unset").
            project_cap = project.max_segment_sec
            if project_cap is None:
                project_cap = ai_models.UNIVERSAL_MAX_SEGMENT_SEC
            max_segment_sec = min(
                float(project_cap),
                ai_models.ABSOLUTE_MAX_SEGMENT_SEC,
            )
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
            scope = project.localisation_scope or project_types.LOCALISATION_SCOPE_HOOK
            requested = project.localisation_scope_requested
            if requested is not None:
                if (
                    requested in project_types.LOCALISATION_SCOPES
                    and spec.segmentation == project_types.SEG_HOOK_SPLIT
                ):
                    scope = requested
                else:
                    # The API and the CHECK prevent this; a hand-written row
                    # must not crash analysis.
                    log.warning(
                        "analyze_project %s: ignoring localisation_scope_requested=%r "
                        "for project_type=%s (keeping %r)",
                        project_id, requested, project.project_type, scope,
                    )
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
            elif (
                spec.segmentation == project_types.SEG_HOOK_SPLIT
                and scope == project_types.LOCALISATION_SCOPE_WHOLE
            ):
                # Whole-video localisation (AICP-340): every scene is
                # re-spoken, so the video is cut ALL-swap at scene changes,
                # each segment within the whole-video target (14.72s), and the
                # hook clamp below never applies. hook_sec is not read.
                target = whole_video_target_sec(project.max_segment_sec)
                proposed = _whole_video_segments(
                    local,
                    duration_sec=float(info.duration_sec),
                    fps=info.fps,
                    target_sec=target,
                    scene_cache_path=scene_cuts_cache_path(project_id),
                )
                log.info(
                    "whole-video localisation produced %d swap segment(s) "
                    "(target=%.2fs, project max_segment_sec=%s): %s",
                    len(proposed), target, project.max_segment_sec,
                    ", ".join(
                        f"[{ps.start_sec:.2f},{ps.end_sec:.2f}]" for ps in proposed
                    ),
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
            elif spec.segmentation == project_types.SEG_SHOT_CHUNKS:
                # Rebuild (design §5.1): fine shot detection; the shots, their
                # folding and the chunk SegmentDefs are published below in ONE
                # transaction under the project row lock.
                rebuild_ranges = rebuild_mod.detect_shots(
                    local, fps=info.fps, duration_sec=float(info.duration_sec),
                )
                log.info(
                    "rebuild shot detection: %d shot(s) for project %s",
                    len(rebuild_ranges), project_id,
                )
                session.refresh(project, with_for_update=True)
                gen_now = rebuild_analysis.current_generation(session, project_id)
                if (
                    project.status != ProjectStatus.analyzing
                    or gen_now != rebuild_gen0
                    or project.rebuild_analysis_token != rebuild_token
                ):
                    # Recovery reset an orphaned analysis meanwhile, or another
                    # analysis job already published: publishing now would
                    # replace shots under work admitted since.
                    log.warning(
                        "analyze_project %s: project is %s, generation %s (job "
                        "started at %s) — rebuild shots NOT published",
                        project_id, project.status, gen_now, rebuild_gen0,
                    )
                    session.rollback()
                    return

                summary = rebuild_analysis.publish_analysis(
                    session, project, rebuild_ranges, source_path=local,
                )
                transition(project, ProjectStatus.ready)
                session.commit()
                log.info(
                    "analyze_project done (rebuild): project_id=%s %s",
                    project_id, summary,
                )
                return
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
            # RunSegment.segment_def_id is NOT an FK to this table (see its
            # docstring in models.py) — a Run's own RunSegments, its frozen
            # segment_snapshot and its RunSegmentRangeAttempt history all
            # survive this delete untouched; only the live SegmentDef rows
            # (and anything that still depends on reading them LIVE, e.g. a
            # legacy pre-snapshot run's launch-window display) are gone.
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

            # Same commit as the SegmentDefs above (see the docstring).
            if spec.segmentation == project_types.SEG_HOOK_SPLIT:
                project.localisation_scope = scope
            if requested is not None:
                # Clear the request only if it is still the one this analysis
                # read: a newer POST /analyze can commit a different request
                # between our first read and this commit (its guard saw the
                # project still `ready`), and an unconditional clear would
                # erase it, so its own queued job would re-apply OUR scope.
                session.execute(
                    sa_update(VideoProject)
                    .where(
                        VideoProject.id == project_id,
                        VideoProject.localisation_scope_requested == requested,
                    )
                    .values(localisation_scope_requested=None)
                    .execution_options(synchronize_session=False)
                )
            transition(project, ProjectStatus.ready)
            session.commit()
            log.info(
                "analyze_project done: project_id=%s, segments=%d",
                project_id, len(proposed),
            )

        except Exception as exc:
            log.exception("analyze_project failed for project_id=%s", project_id)
            if is_rebuild_project:
                # Mark failed only what THIS job still owns: the project is
                # `analyzing` and nobody published since the job started.
                try:
                    session.rollback()
                    session.refresh(project, with_for_update=True)
                    from . import rebuild_analysis

                    owned = (
                        project.status == ProjectStatus.analyzing
                        and project.rebuild_analysis_token == rebuild_token
                        and rebuild_analysis.current_generation(session, project_id)
                        == rebuild_gen0
                    )
                except Exception:
                    owned = False
                if not owned:
                    log.warning(
                        "analyze_project %s: rebuild analysis failed but the project "
                        "moved on (%s) — leaving it as is", project_id, project.status,
                    )
                    try:
                        session.rollback()
                    except Exception:
                        pass
                    raise
            if project.status not in (ProjectStatus.ready, ProjectStatus.failed):
                try:
                    transition(project, ProjectStatus.failed)
                except Exception:
                    project.status = ProjectStatus.failed
            project.error_message = str(exc)
            # AICP-350: the transcript this analysis promised will never start.
            settle_unstarted_transcript(session, project)
            try:
                session.commit()
            except Exception:
                pass
            raise


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

_UNSTARTED_TRANSCRIPT_ERROR = (
    "The analysis did not finish, so the transcription it promised was never "
    "started. Re-analyze the project (or press Transcribe)."
)


def settle_unstarted_transcript(session, project) -> bool:
    """A localisation analysis admission writes ``transcript_status="pending"``
    as a PROMISE (the transcription is admitted only when the analysis
    returns). When that analysis fails or is reaped, nothing will ever keep the
    promise, and a pending status is counted as active work by the deploy drain
    (AICP-350) forever. Settle it as failed — unless a transcription job is
    actually live. Returns True when it changed the row; never raises."""
    try:
        if getattr(project, "transcript_status", None) != TRANSCRIPT_PENDING:
            return False
        from .localisation_jobs import live_transcription

        if live_transcription(session, project.id) is not None:
            return False
        project.transcript_status = TRANSCRIPT_FAILED
        project.transcript_error = _UNSTARTED_TRANSCRIPT_ERROR
        return True
    except Exception:  # noqa: BLE001 - a side channel; never mask the caller's error
        log.warning("settle_unstarted_transcript failed for %s", getattr(project, "id", None),
                    exc_info=True)
        return False


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
        session.refresh(project, with_for_update=True)  # AICP-363 (transcript key fence)

        # analyze_project has already downloaded (gdrive) and normalised the
        # source by the time this job runs, so the local path is simply read
        # here — re-downloading would race with an in-flight re-analysis for a
        # file we already have.
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
        transcript = localisation_mod.transcribe_video(
            local, kie=kie, spend_scope=spend_mod.transcribe_scope(project_id),
        )
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
        if project is not None:
            # AICP-363: the project row lock fences `project_transcript:` keys.
            session.refresh(project, with_for_update=True)
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
        session.refresh(project, with_for_update=True)  # AICP-363 (transcript key fence)
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


def _parse_cost_credits(data: dict) -> Optional[float]:
    """Extract kie.ai `creditsConsumed` (task cost) from a recordInfo 'data' dict.

    Returns a float or None if the field is absent/non-numeric — never raises,
    so cost capture can't fail a segment.
    """
    raw = data.get("creditsConsumed")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def effective_mute_source(run, project) -> bool:
    """The "remove original audio" setting a run's swap clips are cut with.

    ``Run.mute_source`` when the run froze it at admission (AICP-365: the
    localisation autopilot's runs), else the project's live setting (every
    manual run, as before)."""
    frozen = getattr(run, "mute_source", None)
    if frozen is not None:
        return bool(frozen)
    return bool(getattr(project, "mute_source", False))


def _clip_audio_is_stale(clip_path: Optional[str], *, want_muted: bool) -> bool:
    """True when a checkpointed clip's audio no longer matches what we'd cut now.

    The crash-resume path (Fix #9) reuses an already-cut / already-uploaded clip
    to save an encode and an upload. That reuse is only valid while the clip
    still matches the project's "remove original audio" setting — flipping the
    setting and retrying must actually re-cut, not silently resubmit the old
    clip.

    Returns ``False`` (reuse the checkpoint, preserving the existing behaviour)
    whenever the answer can't be established: no path, file already cleaned up,
    or an unprobeable file.
    """
    if not clip_path or not os.path.exists(clip_path):
        return False
    try:
        has_audio = media_mod.probe(clip_path).has_audio
        # Stale either way round: audio present but we now want it gone, or
        # already muted but we now want the source track back.
        return has_audio == want_muted
    except Exception as exc:  # ffprobe failure — don't block the resubmit
        log.warning("could not probe checkpoint clip %s: %s", clip_path, exc)
        return False


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
    audio_ref_urls: Optional[list] = None,
    kie: KieClient,
    kie_factory: Optional[Callable[[], KieClient]] = None,
    # Fix #9 (crash-resume): pre-existing clip checkpoints — skip cut/upload when set.
    resume_clip_url: Optional[str] = None,
    resume_local_clip_path: Optional[str] = None,
    # AICP-351: the run phase's lease (run-lock refresh + sender identity).
    lease: Optional["paid_attempts.SubmitLease"] = None,
) -> Optional[dict]:
    """Cut, upload, then create_task for one swap segment.

    Opens its own DB session so this function is safe to call from a thread pool
    (no shared SQLAlchemy Session across threads).

    Returns a "resubmit recipe" dict ``{task_id, recipe}`` where ``recipe`` holds
    everything :func:`_create_swap_task` needs to re-create the task against the
    same already-uploaded clip (used by the in-poll retry path). Returns ``None``
    when the segment was skipped (marked failed — e.g. a segment longer than the
    selected model's per-clip ceiling).

    *mute_audio* mirrors the project's "remove original audio" setting: the
    clip is cut video-only so the model generates audio from the prompt
    instead of reacting to the source soundtrack.

    *generate_audio* is the already-decided answer to "ask this model to write
    an audio track?" — computed once per run in :func:`_build_submit_state`
    (which can see ``run.stitch_plan``) and carried verbatim into the recipe so
    an in-poll resubmit rebuilds the identical request. ``None`` means "don't
    send the field", i.e. leave the model on its own default.

    Fix #9 — crash-resume clip checkpoint reuse:
    - ``resume_clip_url`` set   → clip already uploaded; skip cut + upload entirely.
    - ``resume_local_clip_path`` set (file exists) → clip already cut; skip cut,
      re-upload the existing file (e.g. upload failed or URL expired).

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
    #     the untouched soundtrack. The segment is marked failed with
    #     source_fallback, and _cut_or_lookup then stitches the ORIGINAL
    #     footage in its place — so the run still reaches `done` and DELIVERS,
    #     just partly un-swapped (Run.source_fallback_segments records how
    #     much). That is why this is only the backstop: a delivered video that
    #     merely looks finished is the worst of the outcomes, so the launch
    #     endpoints refuse such a run up front
    #     (api_v2._assert_no_overlong_swap_segments, covering New Run, Copy run
    #     and automation launches). This branch now catches only what those
    #     cannot see: a project re-analyzed into longer segments after its runs
    #     were created, and any run launched before that gate existed.
    #     (An earlier version of this comment claimed the run ends `incomplete`
    #     with no output. It does not, and never did — the source_fallback
    #     branch in _cut_or_lookup is what actually happens.)
    #   - only pre/post-roll pushed the *clip* over: trim the clip back to the
    #     ceiling. The segment still fits; we just carry less surrounding
    #     context into the generation.
    model_max_clip = model_spec.max_clip_sec
    seg_len = sd_end_sec - sd_start_sec
    if seg_len > model_max_clip + 0.05:
        # source_fallback=True: this is deterministic, not transient. No number
        # of retries makes a 22s segment generatable on a 15s model, so the run
        # is allowed to finish with the ORIGINAL footage in this segment's place
        # rather than being blocked forever. See _skip_segment.
        #
        # Kept under 80 chars: the run UI truncates rs.error_message there, and
        # the consequence is the half an operator must not lose. The segment
        # index is deliberately absent — the message is rendered inside that
        # segment's own panel, right under its "Segment N" heading.
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

    # ------------------------------------------------------------------
    # Fix #9: crash-resume checkpoint — skip cut and/or upload when a
    # prior attempt already produced the clip file / upload URL.
    # ------------------------------------------------------------------
    if resume_clip_url:
        # Clip already uploaded — skip cut + upload entirely.
        clip_url = resume_clip_url
        log.info(
            "segment idx=%d resume: reusing stored kie_upload_url (skip cut+upload)",
            sd_index,
        )
        t_cut = t0
        t_upload = t0
        # Advance through the intermediate states (the state machine requires
        # pending → uploading → submitted before generating).
        with get_session() as session:
            rs = session.get(RunSegment, rs_id)
            rs.kie_upload_url = clip_url
            transition(rs, SegmentStatus.uploading)
            transition(rs, SegmentStatus.submitted)
            session.commit()
    elif resume_local_clip_path and os.path.exists(resume_local_clip_path):
        # Clip already cut to disk — skip cut, just re-upload.
        clip_dst = resume_local_clip_path
        log.info(
            "segment idx=%d resume: clip already on disk at %s (skip cut, re-upload)",
            sd_index, clip_dst,
        )
        t_cut = t0

        with get_session() as session:
            rs = session.get(RunSegment, rs_id)
            transition(rs, SegmentStatus.uploading)
            session.commit()

        # Namespace the remote object by run_segment id. Clip basenames are
        # index-based (clip_0001.mp4) and identical across every Run, so an
        # un-namespaced upload path collides when two Runs process the same
        # segment index concurrently — one Run's clip overwrites the other's at
        # the shared remote path and Seedance fetches the wrong input. rs_id is
        # a globally-unique per-(run, segment) UUID, so this cannot collide.
        clip_url = thread_kie.upload_file(
            clip_dst, "charswap/segments",
            file_name=f"{rs_id}_{os.path.basename(clip_dst)}",
        )
        t_upload = time.monotonic()

        with get_session() as session:
            rs = session.get(RunSegment, rs_id)
            rs.kie_upload_url = clip_url
            transition(rs, SegmentStatus.submitted)
            session.commit()
    else:
        # Normal path: cut + upload.
        # Two reasons to send a video-only clip:
        #   - Gemini Omni fails when its reference clip carries an audio track.
        #   - The project has "remove original audio" set, so Seedance can't
        #     hear the source soundtrack and generates audio from the prompt.
        # Either way the original audio is still available at stitch time via
        # audio_mode="original" (the source file itself is never modified).
        include_audio = not is_omni and not mute_audio
        media_mod.cut_clip(source, clip_start, clip_end, clip_dst, include_audio=include_audio)
        t_cut = time.monotonic()

        with get_session() as session:
            rs = session.get(RunSegment, rs_id)
            rs.local_clip_path = clip_dst
            transition(rs, SegmentStatus.uploading)
            session.commit()

        # Namespace the remote object by run_segment id. Clip basenames are
        # index-based (clip_0001.mp4) and identical across every Run, so an
        # un-namespaced upload path collides when two Runs process the same
        # segment index concurrently — one Run's clip overwrites the other's at
        # the shared remote path and Seedance fetches the wrong input. rs_id is
        # a globally-unique per-(run, segment) UUID, so this cannot collide.
        clip_url = thread_kie.upload_file(
            clip_dst, "charswap/segments",
            file_name=f"{rs_id}_{os.path.basename(clip_dst)}",
        )
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
        # Same contract for the voice: a resubmitted segment must come back in
        # the SAME voice as the segments it will be stitched next to, which is
        # exactly what a run-level reference buys.
        "audio_ref_urls": list(audio_ref_urls or []),
    }
    with get_session() as session:
        run_id = getattr(session.get(RunSegment, rs_id), "run_id", None)
        predecessor = paid_attempts.last_task_id(session, OWNER_RUN_SEGMENT, rs_id)

    # Knowledge capture A1 (AICP-299): the intent, before the paid call. A no-op
    # unless capture is enabled for this project; never raises.
    kb_attempt = kcap.swap_intent(
        rs_id=rs_id, recipe=recipe, include_audio=not is_omni and not mute_audio, site="A1",
    )
    # AICP-351 §5.1: a committed attempt (+ its estimated draw) BEFORE the one
    # POST; the receipt and the task id commit together below.
    try:
        attempt_id, task_id = _paid_swap_send(
            rs_id=rs_id, run_id=run_id, recipe=recipe, kie=thread_kie, lease=lease,
            predecessor_task_id=predecessor,
        )
    except Exception as exc:
        kcap.dispatch_error(kb_attempt, exc, site="A1")
        raise

    with get_session() as session:
        rs = session.get(RunSegment, rs_id)
        owner_ok = _record_swap_receipt(session, attempt_id, task_id, run_id, recipe)
        if owner_ok:
            rs.seedance_task_id = task_id
            transition(rs, SegmentStatus.generating)
        session.commit()
    if not owner_ok:
        # A recovery parked this attempt while the POST was in flight: the run
        # is no longer ours. The task id is kept on the parked attempt.
        raise paid_attempts.AttemptBlocked(
            f"segment {rs_id}: attempt {attempt_id} was parked by recovery while its "
            f"submit was in flight (late task {task_id})")
    kcap.submitted(kb_attempt, task_id, site="A1")  # after the commit (P1)

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


# Recipe keys this worker version does not understand, already logged about.
# Bounded by the number of distinct stray keys (one deploy's worth), and only
# written under the GIL from the poll loop / submit threads, so a plain set is
# enough — the point is not to repeat the same WARNING once per segment per poll
# round for the lifetime of a run.
_WARNED_UNKNOWN_RECIPE_KEYS: set[str] = set()


def _warn_unknown_recipe_keys(extras: dict) -> None:
    """Log once per unrecognised resubmit-recipe key (see _create_swap_task)."""
    fresh = sorted(k for k in extras if k not in _WARNED_UNKNOWN_RECIPE_KEYS)
    if not fresh:
        return
    _WARNED_UNKNOWN_RECIPE_KEYS.update(fresh)
    log.warning(
        "resubmit recipe carried unknown key(s) %s — ignoring them. This means a "
        "recipe was written by a different worker version (rolling deploy or "
        "rollback); the request is built from the keys this version does know.",
        ", ".join(fresh),
    )


def _record_swap_draw(run_id: Optional[str], task_id: str, recipe: dict) -> None:
    """Measure one created video task against its run's reservation
    (AICP-155). Never raises — see spend.record_draw."""
    spend_mod.record_draw(
        spend_mod.run_scope(run_id) if run_id else None,
        spend_mod.video_credits(
            recipe.get("run_model"), recipe.get("run_resolution"),
            float(recipe.get("clip_end") or 0.0) - float(recipe.get("clip_start") or 0.0),
        ),
        kind="video",
        call_key=task_id,
        sessions=get_session,
    )


def _settle_swap_draw(task_id: str, data: dict, rs_id: Optional[str]) -> None:
    """Settle a terminal video task's reported cost against its run's
    reservation (AICP-155). Never raises."""
    cost = _parse_cost_credits(data)
    if cost is None:
        return
    run_id = None
    if rs_id:
        try:
            with get_session() as s:
                run_id = getattr(s.get(RunSegment, rs_id), "run_id", None)
        except Exception:
            run_id = None
    spend_mod.settle_draw(
        task_id, cost,
        scope_key=spend_mod.run_scope(run_id) if run_id else None,
        kind="video", sessions=get_session,
    )


def _swap_task_call(
    *,
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
    audio_ref_urls: Optional[list] = None,
    **unknown_recipe_keys,
) -> tuple[str, dict]:
    """The ``KieClient`` method and kwargs for an already-uploaded swap clip.

    Pure (no DB, no I/O): :func:`_create_swap_task` calls the method, and
    knowledge capture (AICP-299) records the same request through
    ``kie_client.build_*_payload``. Original notes for the task creation:

    Create a Seedance/Omni task for an already-uploaded swap clip.

    Shared by the initial submit and the in-poll retry path so both build
    identical request parameters. Does no DB or file work — pure API call.

    *generate_audio* is decided in :func:`_build_submit_state`; ``None`` (the
    default) means "omit the field", so a recipe written by a worker version that
    did not carry it leaves the model on its own default rather than being
    silenced.

    ``**unknown_recipe_keys`` is deliberate rollback armour, not laziness. The
    caller in the poll loop does ``_create_swap_task(kie=kie, **recipe)`` with a
    recipe that round-tripped through Redis as JSON — possibly written by a
    *different* worker version during a rolling deploy. A key this version does
    not know would raise TypeError, which the poll loop's ``except Exception``
    swallows into ``new_task_id = None``, permanently failing a segment that only
    hit a transient kie.ai error. Losing a whole run to a recipe-schema change is
    far worse than ignoring a field, so extras are dropped with one WARNING.
    """
    if unknown_recipe_keys:
        _warn_unknown_recipe_keys(unknown_recipe_keys)

    spec = ai_models.spec_for(run_model)
    if not spec.supports_v2v:
        # An images-only model (Kling 4.0) has no v2v request: refuse before
        # anything is sent rather than build a Seedance-shaped one.
        raise ValueError(f"model {run_model!r} cannot edit a source clip (images-only)")

    if spec.family == "omni":
        # clip was already trimmed to <= the model's max_clip_sec at cut time;
        # send its full (video-only) length as the trim range.
        trim_end = round(clip_end - clip_start, 2)
        return "create_omni_task", dict(
            prompt=effective_prompt,
            image_urls=ref_urls,
            video_url=clip_url,
            video_start=0,
            video_end=trim_end,
            resolution=_omni_resolution(run_resolution),
            aspect_ratio=_map_omni_aspect(
                project_aspect_ratio, project_width, project_height
            ),
            duration=_snap_omni_duration(clip_start, clip_end),
        )

    # `generate_audio` is 2.5-only — the 2.0 family rejects the field, so the key
    # must be absent from their payload entirely. Forward the decision made in
    # _build_submit_state and nothing more: this function sees a recipe, not a
    # Run, so it cannot tell whether the run stitches (see the truth table
    # there). create_task also enforces the capability itself and would raise if
    # we passed the field to a 2.0 model.
    extra: dict = {}
    if spec.supports_generate_audio and generate_audio is not None:
        extra["generate_audio"] = bool(generate_audio)

    # The voice reference is gated on the same registry flag, for the same
    # reason: a model without the field rejects the whole request. Gated HERE
    # rather than only where the run is read, so a recipe that outlived a model
    # change (rolling deploy, re-run of an old run) cannot send the key to a
    # backend that has no place for it.
    if audio_ref_urls and spec.max_reference_audios:
        extra["reference_audio_urls"] = list(audio_ref_urls)

    return "create_task", dict(
        prompt=effective_prompt,
        reference_image_urls=ref_urls,
        reference_video_urls=[clip_url],
        # Never send a resolution this model cannot do (e.g. a 1080p run
        # switched to Seedance 2.0 Fast, which stops at 720p) — the registry
        # substitutes the closest supported tier that does not exceed the
        # request (720p here), not the model's cheapest.
        resolution=ai_models.resolution_or_default(run_model, run_resolution),
        # A model that derives its geometry from the input video (Seedance 2.5 in
        # video-editing mode, which is every submission we make) REQUIRES
        # "adaptive" + duration -1 and 500s on anything else — even a ratio that
        # matches the source. Both come from the registry so the pair can never
        # be sent half-right.
        aspect_ratio=ai_models.aspect_for(run_model, project_aspect_ratio),
        duration=clamp_duration_for(run_model, clip_start, clip_end),
        model=ai_models.kie_model_id(run_model),
        **extra,
    )


def _create_swap_task(
    *,
    kie: KieClient,
    once: bool = False,
    **recipe,
) -> str:
    """Create a Seedance/Omni task for an already-uploaded swap clip.

    Shared by the initial submit and the in-poll retry path so both build
    identical request parameters; the parameters come from
    :func:`_swap_task_call` (unknown recipe keys are dropped there, with one
    WARNING — the rolling-deploy armour).

    *once* (AICP-351): ONE POST, classified (``KieTaskRefused`` /
    ``KieTaskUnknown``) instead of the client's transient retries — the
    attempt protocol below (:func:`_paid_swap_send`) needs it. The range
    attempt keeps its own ``submit_billed_at`` protocol and the default.
    """
    method, kwargs = _swap_task_call(**recipe)
    if once:
        kwargs["once"] = True
    return getattr(kie, method)(**kwargs)


def _swap_credits(recipe: dict) -> float:
    return spend_mod.video_credits(
        recipe.get("run_model"), recipe.get("run_resolution"),
        float(recipe.get("clip_end") or 0.0) - float(recipe.get("clip_start") or 0.0),
    )


class PaidSubmitParked(RuntimeError):
    """A paid submit's outcome is unknown: its attempt is parked for the
    operator (never resubmitted blind, AICP-351 §5.1)."""


def _paid_swap_send(
    *,
    rs_id: str,
    run_id: Optional[str],
    recipe: dict,
    kie: KieClient,
    lease: Optional["paid_attempts.SubmitLease"],
    predecessor_task_id: Optional[str],
    in_poll_predecessor: Optional[str] = None,
) -> tuple[str, str]:
    """One paid swap createTask under the attempt protocol (AICP-351 §5.1):
    commit a ``started`` attempt + estimated draw, re-check the lease, ONE
    POST. Returns ``(attempt_id, task_id)``; the caller records the outcome
    together with the owner row (:func:`paid_attempts.record_outcome`).

    Raises ``paid_attempts.AttemptBlocked`` (nothing sent), the refusal
    (attempt ``failed_unbilled``) or :class:`PaidSubmitParked` (attempt
    ``parked``: the outcome is unknown)."""
    scope = spend_mod.run_scope(run_id) if run_id else "unscoped"
    attempt_id = paid_attempts.begin(
        OWNER_RUN_SEGMENT, rs_id, kind="video", run_id=run_id, request=recipe,
        credits=_swap_credits(recipe), scope_key=scope, draw_kind="video",
        predecessor_task_id=predecessor_task_id, lease=lease,
        fence=_segment_fence(rs_id, in_poll_predecessor), sessions=get_session,
    )
    paid_attempts.pre_send_check(attempt_id, lease, scope_key=scope, kind="video",
                                 sessions=get_session)
    try:
        task_id = _create_swap_task(kie=kie, once=True, **recipe)
    except Exception as exc:
        outcome = paid_attempts.close_after_error(attempt_id, exc, scope_key=scope,
                                                  kind="video", sessions=get_session)
        if outcome == "unknown":
            raise PaidSubmitParked(
                f"submit outcome unknown — parked for the operator (attempt {attempt_id}): {exc}"
            ) from exc
        raise
    return attempt_id, task_id


def _segment_fence(rs_id: str, in_poll_predecessor: Optional[str]):
    """The owner fence of a swap submit (AICP-351): the segment row, locked, is
    still waiting for THIS submit — ``submitted`` with no task id (initial
    submit) or ``generating`` with the failed predecessor as its task
    (in-poll resubmit)."""

    def fence(db) -> bool:
        rs = db.get(RunSegment, rs_id, with_for_update=True, populate_existing=True)
        if rs is None:
            return False
        if in_poll_predecessor is not None:
            return (rs.status == SegmentStatus.generating
                    and rs.seedance_task_id == in_poll_predecessor)
        return rs.status == SegmentStatus.submitted and not rs.seedance_task_id

    return fence


def _record_swap_receipt(session, attempt_id: str, task_id: str, run_id: Optional[str],
                         recipe: dict) -> bool:
    """The accepted outcome in *session* (committed by the caller with the
    owner row); True when the owner row may take the task id."""
    return paid_attempts.record_outcome(
        session, attempt_id, outcome="accepted", task_id=task_id,
        credits=_swap_credits(recipe),
        scope_key=spend_mod.run_scope(run_id) if run_id else "unscoped", draw_kind="video",
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

    Deadlines are wall-clock epoch timestamps (time.time()) so the map can be
    serialised to Redis across process boundaries.

    NOTE: This is the synchronous poll loop used by the monolithic process_run()
    entry point (backward-compat, tests).  The staged production path uses
    poll_run_once() which returns remaining pending tasks instead of looping.
    """
    while pending:
        pending = poll_run_once_internal(pending=pending, kie=kie, results_dir=results_dir)
        if pending:
            time.sleep(RUN_POLL_INTERVAL_SEC)


# How much shorter than the submitted clip a result may legitimately be before
# we treat it as truncated. Generated clips never land on exactly the input
# length — the worst honest drift measured across every completed task is 0.27s
# (a 10.0s clip returning 9.728s), a mix of frame quantisation at 24fps and the
# model's own rounding. The failure this guards against is an order of magnitude
# larger (12.8s short on run 4e631e56), so 1.0s separates the two cleanly
# without ever tripping on a healthy generation.
RESULT_SHORTFALL_TOLERANCE_SEC = 1.0


class ResultTooShort(NamedTuple):
    """A generation that came back materially shorter than the clip we sent."""

    clip_sec: float
    result_sec: float

    @property
    def shortfall_sec(self) -> float:
        return self.clip_sec - self.result_sec


def _result_too_short(
    clip_path: Optional[str], result_path: str
) -> Optional[ResultTooShort]:
    """Both durations when *result_path* falls short of *clip_path*, else None.

    Seedance 2.5 intermittently returns a result truncated to exactly 14.72s
    (353 frames @ 24fps) instead of the length of the clip it was given. It is
    not a ceiling — the same 27.5s segment came back at 27.392s on run
    `fd3948b2` and 14.720s on run `4e631e56`, and a 29.30s clip produced 29.04s
    — so the length cannot be validated up front at submit time. It has to be
    measured on the way back, which is what this is for.

    Returns ``None`` when the result is the expected length — or when either
    duration cannot be established. That last case is deliberate: this is a
    guard bolted onto a path that previously accepted everything, and a probe
    failure is not evidence of truncation. Failing the segment on it would turn
    an unreadable-but-fine result into a lost swap, so an unknown duration keeps
    the pre-existing behaviour (accept) and says so in the log.
    """
    if not clip_path:
        return None
    try:
        clip_sec = media_mod.probe(clip_path).duration_sec
        result_sec = media_mod.probe(result_path).duration_sec
    except Exception as exc:  # MediaError, missing file, unparseable output
        log.warning(
            "could not compare result duration against clip (%s vs %s): %s — "
            "accepting the result unchecked",
            result_path, clip_path, exc,
        )
        return None
    if clip_sec - result_sec <= RESULT_SHORTFALL_TOLERANCE_SEC:
        return None
    return ResultTooShort(clip_sec=clip_sec, result_sec=result_sec)


def _download_and_complete_segment(
    *,
    task_id: str,
    meta: dict,
    url: str,
    cost: Optional[float],
    results_dir: str,
    kie: KieClient,
) -> Optional[ResultTooShort]:
    """Download the result for one completed AI task and mark the segment done.

    Opens its own DB session so this is safe to call from a ThreadPoolExecutor
    (Fix #16 — concurrent downloads within a poll round).  DB writes are fully
    contained inside the session context manager; no shared session is used.

    EVERY result is completed and stitched, short or long — the operator's
    call. A generation materially shorter than the clip we sent is still flagged
    on the segment (a warning on a completed segment, not a failure) so the
    shortened timeline is visible rather than silent. It is NOT resubmitted:
    three retries of run 4e631e56 with the clip length spelled out in the prompt
    all came back at the same 14.72s, so retrying mostly re-bills for an
    identical result.

    Raises on download or DB error so the calling thread can propagate the
    exception via Future.result().
    """
    result_dst = os.path.join(results_dir, f"result_{meta['index']:04d}.mp4")
    kie.download_result(url, result_dst)

    # Measure before recording anything. Writing local_result_path first would
    # leave a truncated file referenced on a failed segment, and the stitch
    # reads that path.
    with get_session() as poll_session:
        clip_path = poll_session.get(RunSegment, meta["rs_id"]).local_clip_path

    too_short = _result_too_short(clip_path, result_dst)

    with get_session() as poll_session:
        rs = poll_session.get(RunSegment, meta["rs_id"])
        rs.seedance_result_url = url
        rs.local_result_path = result_dst
        rs.cost_credits = cost
        # Clear any interim "attempt N failed; retrying" message now
        # that the segment has succeeded — and any source-fallback verdict from
        # an earlier attempt, so the stitch uses this result rather than the
        # original footage.
        rs.error_message = None
        rs.source_fallback = False
        if too_short is not None:
            # WHATEVER comes back gets stitched — operator's call, and the
            # reason is that refusing it delivers nothing at all while Seedance
            # 2.5 is in one of its truncating spells. The cost is real and is
            # why this is recorded rather than swallowed: a short segment
            # shortens the timeline (run 4e631e56 delivered 25.1s of a 38s
            # video), and on audio_mode="original" everything after it desyncs
            # against the untouched source track.
            #
            # Carried on a COMPLETED segment, so it is a warning, not a failure
            # — the run still delivers. Kept under 80 chars: the run UI
            # truncates error_message there.
            rs.error_message = (
                f"short: model returned {too_short.result_sec:.1f}s for a "
                f"{too_short.clip_sec:.1f}s clip"
            )
        transition(rs, SegmentStatus.completed)
        poll_session.commit()
        run_id = getattr(rs, "run_id", None)

    # kie.ai's own creditsConsumed replaces the submit-time estimate.
    spend_mod.settle_draw(
        task_id, cost, scope_key=spend_mod.run_scope(run_id) if run_id else None, kind="video",
        sessions=get_session,
    )
    # Knowledge capture B1 (AICP-299), after the pipeline's commit; never raises.
    kcap.output(
        task_id=task_id, path=result_dst, run_id=run_id, rs_id=meta["rs_id"],
        status="short_result" if too_short is not None else "succeeded",
        payload={"url": url, **({"clip_sec": too_short.clip_sec, "result_sec": too_short.result_sec}
                                if too_short is not None else {})},
        segment_version=True, site="B1",
    )

    if too_short is not None:
        log.warning(
            "RunSegment idx %d (task %s): model returned %.2fs for a %.2fs clip "
            "(%.1fs SHORT) — stitching it anyway; the delivered video will be "
            "that much shorter than the source timeline.",
            meta["index"], task_id, too_short.result_sec, too_short.clip_sec,
            too_short.shortfall_sec,
        )
    else:
        log.info("RunSegment idx %d completed (task %s)", meta["index"], task_id)
    return too_short


def _resubmit_segment_task(
    *,
    task_id: str,
    meta: dict,
    pending: dict[str, dict],
    kie: KieClient,
    reason: str,
    lease: Optional["paid_attempts.SubmitLease"] = None,
) -> bool:
    """Re-run a segment's recipe as a fresh task; True if a retry was queued.

    Shared by the two ways one generation can come back unusable: kie reporting
    ``fail``, and kie reporting success with a truncated result. Both are
    transient and both have the identical fix — submit the same recipe again —
    so they must not drift into two implementations.

    Returns False when there is nothing to retry with (a resumed orphan carries
    no recipe) or the attempt budget is spent; the caller decides what a
    permanent failure means for its case.
    """
    attempt = meta.get("attempt", 1)
    recipe = meta.get("recipe")
    if not recipe or attempt >= RUN_TASK_MAX_ATTEMPTS:
        return False

    # kie.ai validates an uploaded clip_url asynchronously — createTask always
    # returns a taskId, and a dead URL only surfaces later as this `fail`
    # state. A dead URL is not transient: it fails identically every time, so
    # resubmitting `recipe` unchanged burns the whole retry budget on a request
    # that was never going to work. This is exactly what happened to run
    # 68eb3923 (2026-08-31): "Retry Run" reused a >24h-old kie.ai upload
    # (crash-resume's `resume_clip_url` checkpoint), kie.ai's own storage for
    # that upload had since expired, and all 3 attempts failed identically on
    # "content[6].video_url ... resource not found." Re-upload the clip fresh
    # whenever the failure names a missing resource and the cut clip is still
    # on local disk, so the retry actually has a chance. `reason` is
    # `data.get("failMsg") or data.get("failCode") or "unknown"` upstream, so
    # it is not always a str — a fail with a bare numeric failCode and no
    # failMsg (see test_kie_client.py's `failCode: 4001` fixture) would
    # otherwise crash `.lower()` here and abort the whole poll round.
    if "resource not found" in str(reason).lower():
        with get_session() as poll_session:
            rs = poll_session.get(RunSegment, meta["rs_id"])
            local_clip_path = rs.local_clip_path if rs else None
        if local_clip_path and os.path.exists(local_clip_path):
            try:
                fresh_url = kie.upload_file(
                    local_clip_path, "charswap/segments",
                    file_name=(
                        f"{meta['rs_id']}_retry{attempt + 1}_"
                        f"{os.path.basename(local_clip_path)}"
                    ),
                )
            except Exception as exc:
                log.warning(
                    "seg %d: re-upload after dead-resource failure raised "
                    "(attempt %d): %s — resubmitting the stale URL anyway",
                    meta["index"], attempt, exc,
                )
            else:
                recipe = dict(recipe, clip_url=fresh_url)
                with get_session() as poll_session:
                    rs = poll_session.get(RunSegment, meta["rs_id"])
                    if rs:
                        rs.kie_upload_url = fresh_url
                        poll_session.commit()
                log.info(
                    "seg %d: re-uploaded clip after a dead-resource failure "
                    "(new kie_upload_url=%s)", meta["index"], fresh_url,
                )
        else:
            log.warning(
                "seg %d: dead-resource failure but no local clip on disk to "
                "re-upload from (%s) — resubmitting the stale URL anyway",
                meta["index"], local_clip_path,
            )
    with get_session() as poll_session:
        run_id = getattr(poll_session.get(RunSegment, meta["rs_id"]), "run_id", None)
    # Knowledge capture A2 (AICP-299): the resubmit's intent; never raises.
    kb_attempt = kcap.swap_intent(rs_id=meta["rs_id"], recipe=recipe, resubmit_of=task_id, site="A2")
    # AICP-351 §5.1: the resubmit gets its OWN attempt; `rs.seedance_task_id`
    # still holds the predecessor until the receipt below commits, so recovery
    # judges this window by the attempt row, never by the owner row.
    try:
        attempt_id, new_task_id = _paid_swap_send(
            rs_id=meta["rs_id"], run_id=run_id, recipe=recipe, kie=kie, lease=lease,
            predecessor_task_id=task_id, in_poll_predecessor=task_id,
        )
    except paid_attempts.AttemptBlocked:
        kcap.dispatch_error(kb_attempt, RuntimeError("attempt blocked"), site="A2")
        raise
    except Exception as exc:
        kcap.dispatch_error(kb_attempt, exc, site="A2")
        log.warning(
            "resubmit failed for seg %d (attempt %d): %s",
            meta["index"], attempt, exc,
        )
        if isinstance(exc, PaidSubmitParked):
            meta["parked"] = str(exc)
        return False

    with get_session() as poll_session:
        rs = poll_session.get(RunSegment, meta["rs_id"])
        owner_ok = _record_swap_receipt(poll_session, attempt_id, new_task_id, run_id, recipe)
        if owner_ok:
            rs.seedance_task_id = new_task_id
            rs.error_message = (
                f"attempt {attempt}/{RUN_TASK_MAX_ATTEMPTS} "
                f"failed ({reason}); retrying"
            )
        poll_session.commit()
    if not owner_ok:
        raise paid_attempts.AttemptBlocked(
            f"segment {meta['rs_id']}: resubmit attempt {attempt_id} was parked by recovery "
            f"while in flight (late task {new_task_id})")
    # After the commit (P1): the new submission, and T2 — the predecessor failed.
    kcap.submitted(kb_attempt, new_task_id, site="A2", predecessor_failed={
        "task_id": task_id, "error_class": str(reason), "payload": {"failMsg": str(reason)[:500]}})
    log.warning(
        "task %s (seg %d) unusable: %s — retry %d/%d as task %s",
        task_id, meta["index"], reason,
        attempt + 1, RUN_TASK_MAX_ATTEMPTS, new_task_id,
    )
    # The fail path still holds the old task in `pending`; the truncation path
    # already removed it when the download was collected. pop() covers both.
    pending.pop(task_id, None)
    new_meta = dict(meta)
    new_meta["attempt"] = attempt + 1
    # Carry the (possibly re-uploaded) recipe forward so a further retry on
    # this same chain reuses the fresh URL rather than the dead one.
    new_meta["recipe"] = recipe
    # deadline as wall-clock epoch so it survives Redis serialisation
    new_meta["deadline"] = time.time() + RUN_SKIP_TIMEOUT_SEC
    pending[new_task_id] = new_meta
    return True


def poll_run_once_internal(
    *,
    pending: dict[str, dict],
    kie: KieClient,
    results_dir: str,
    lease: Optional["paid_attempts.SubmitLease"] = None,
) -> dict[str, dict]:
    """Perform ONE round of status checks on all pending AI tasks.

    For each pending task:
    - success: download result, mark segment completed, remove from pending.
    - fail + retries left: resubmit the clip, update pending with new task id.
    - fail + exhausted or no recipe: mark segment failed, remove from pending.
    - timeout exceeded: mark segment failed, remove from pending.
    - still in-progress: leave in pending.

    Fix #16 — concurrent downloads: completed tasks are collected during the
    status sweep, then their results are downloaded concurrently via a bounded
    ThreadPoolExecutor (POLL_DOWNLOAD_CONCURRENCY).  Each download thread opens
    its own DB session, so session usage is thread-safe.  If any download fails,
    the exception is re-raised after all downloads finish.

    Returns the remaining pending map (empty when all tasks are terminal).
    Deadlines are wall-clock epoch timestamps (time.time()) so the map survives
    serialisation to Redis across process restarts.
    """
    now = time.time()

    # Collect tasks that completed successfully so we can download them
    # concurrently after the status sweep (Fix #16).
    completed_downloads: list[dict] = []  # [{task_id, meta, url, cost}, ...]

    for task_id in list(pending):
        meta = pending[task_id]
        try:
            data = kie.get_task(task_id)
        except Exception as exc:  # transient - retry next round
            log.warning("get_task(%s) transient error: %s", task_id, exc)
            continue

        state = (data.get("state") or "").lower()
        if state in ("success", "fail"):
            # Whatever happens next (download, retry, skip), kie.ai's reported
            # cost replaces the estimate now (AICP-155).
            _settle_swap_draw(task_id, data, meta.get("rs_id"))

        if state == "success":
            url = _parse_result_url(data)
            if not url:
                with get_session() as poll_session:
                    rs = poll_session.get(RunSegment, meta["rs_id"])
                    _skip_segment(rs, "success but no result url", poll_session)
                kcap.terminal(task_id=task_id, status="failed", error_class="no_result_url", site="T1")
                del pending[task_id]
                continue

            # Collect for concurrent download below.
            completed_downloads.append({
                "task_id": task_id,
                "meta": meta,
                "url": url,
                "cost": _parse_cost_credits(data),
            })
            del pending[task_id]

        elif state == "fail":
            msg = data.get("failMsg") or data.get("failCode") or "unknown"
            attempt = meta.get("attempt", 1)

            # Transient backend failure ("Internal Error, Please try again
            # later.") → resubmit the same already-uploaded clip up to
            # RUN_TASK_MAX_ATTEMPTS total attempts before giving up.
            if _resubmit_segment_task(
                task_id=task_id, meta=meta, pending=pending, kie=kie, reason=msg,
                lease=lease,
            ):
                continue

            # No recipe (resumed orphan) or attempts exhausted → mark failed.
            # The run will be marked `incomplete` (not stitched) so this
            # segment can be re-run manually. A resubmit whose outcome is
            # unknown is PARKED (AICP-351): the segment fails here and the
            # parked attempt waits for the operator on the run page.
            with get_session() as poll_session:
                rs = poll_session.get(RunSegment, meta["rs_id"])
                _skip_segment(
                    rs,
                    (f"resubmit outcome unknown — parked for the operator (see the run page)"
                     if meta.get("parked") else
                     f"failed after {attempt} attempt(s): {msg}"),
                    poll_session,
                )
            log.warning(
                "task %s (seg %d) failed permanently after %d attempt(s): %s",
                task_id, meta["index"], attempt, msg,
            )
            kcap.terminal(task_id=task_id, status="failed", error_class=str(msg),
                          payload={"failMsg": str(msg)[:500]}, site="T2")
            del pending[task_id]
        elif now > meta["deadline"]:
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
            kcap.terminal(task_id=task_id, status="unknown", error_class="pipeline_timeout", site="T3")
            del pending[task_id]
        # else: still waiting/queuing/generating - leave pending

    # ------------------------------------------------------------------
    # Fix #16: download results for all completed tasks concurrently.
    # Each thread opens its own DB session (thread-safe).  Exceptions from
    # individual downloads are collected and re-raised after all futures
    # settle, so a single bad download doesn't block the others.
    # ------------------------------------------------------------------
    if completed_downloads:
        if len(completed_downloads) == 1:
            # Fast path: avoid thread overhead for a single download.
            item = completed_downloads[0]
            _download_and_complete_segment(
                task_id=item["task_id"],
                meta=item["meta"],
                url=item["url"],
                cost=item["cost"],
                results_dir=results_dir,
                kie=kie,
            )
        else:
            log.info(
                "poll_run_once: downloading %d completed results concurrently "
                "(concurrency=%d)",
                len(completed_downloads), POLL_DOWNLOAD_CONCURRENCY,
            )
            download_errors: list[Exception] = []
            with ThreadPoolExecutor(max_workers=POLL_DOWNLOAD_CONCURRENCY) as pool:
                futures = [
                    pool.submit(
                        _download_and_complete_segment,
                        task_id=item["task_id"],
                        meta=item["meta"],
                        url=item["url"],
                        cost=item["cost"],
                        results_dir=results_dir,
                        kie=kie,
                    )
                    for item in completed_downloads
                ]
            for fut in futures:
                try:
                    fut.result()
                except Exception as exc:
                    download_errors.append(exc)
            if download_errors:
                raise download_errors[0]

    return pending


# ---------------------------------------------------------------------------
# _build_submit_state — shared by submit_run and process_run
# ---------------------------------------------------------------------------


class ImagesOnlyRunRefused(RuntimeError):
    """A v2v submit/poll path was handed an images-only (rebuild) run."""


def _refuse_images_only(run, where: str) -> bool:
    """True (and an ERROR log) when *run* is an images-only run: the v2v
    submit / poll / resubmit / recovery paths must never touch one — they
    create provider tasks through the retrying client and read
    ``seedance_task_id`` (docs/rebuild-from-reference-design.md §6.4). The
    images-only pipeline drives such runs through rebuild_chunk_attempts."""
    if getattr(run, "generation_mode", "v2v") == project_types.GENERATION_IMAGES_ONLY:
        log.error(
            "%s refused images-only run %s — the v2v pipeline never handles it",
            where, getattr(run, "id", "?"),
        )
        return True
    return False


def _build_submit_state(
    run_id: str,
    *,
    kie: KieClient,
    gdrive: Optional[GDriveClient],
    session,
    lease: Optional["paid_attempts.SubmitLease"] = None,
) -> Tuple[dict, bool, str]:
    """
    Internal: transition the run to processing, build and execute the submit
    phase, and return ``(pending, did_submit, r_dir)``.

    ``pending`` is the task_id → meta map populated by the submit phase.
    ``did_submit`` is True if any new tasks were submitted (used to decide
    whether to skip re-stitch on a delivery-only retry).
    ``r_dir`` is the run results directory.

    Raises on any error (caller must handle state-machine transitions).
    """
    run: Run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")
    if _refuse_images_only(run, "_build_submit_state"):
        raise ImagesOnlyRunRefused(f"run {run_id} is images-only")
    project: VideoProject = session.get(VideoProject, run.project_id)
    if project is None:
        raise ValueError(f"VideoProject not found: {run.project_id}")

    external_kie = kie is not None

    # queued → processing: normal first-time path.
    # Any other active state (processing/stitching/delivering) means this
    # is an orphan resume — reset to queued first, then advance.
    # The processing → queued edge was added in state_machine.py (TR5b).
    if run.status != RunStatus.queued:
        log.info(
            "submit_run: run %s is in status=%r (not queued) — "
            "resetting to queued for orphan resume",
            run_id, run.status,
        )
        try:
            transition(run, RunStatus.queued)
        except Exception:
            # Fallback for stitching/delivering which lack a direct →queued edge.
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
    duration_sec = info.duration_sec

    # The FROZEN launch timeline, not the project's live segments — same
    # source of truth finalize already reads (run_segment_defs). Before this
    # fix, a resumed/retried submit re-read project.segments directly: a
    # project edit (or a promoted range-regeneration candidate rewriting
    # Run.segment_snapshot) landing between the original submit and a later
    # crash-resume/retry would silently resubmit against boundaries the
    # already-uploaded clips were never cut to. run_segment_defs falls back to
    # project.segments when no snapshot exists yet (a brand-new run's first
    # submit), so this is a strict narrowing, not a behaviour change for that
    # case.
    seg_defs: list = run_segment_defs(run, project)
    if not seg_defs:
        raise ValueError(
            f"Project {run.project_id} has no segments — run analyze_project first"
        )

    # Freeze the defs onto the run before any clip is cut, so later edits to the
    # project's segments cannot change what finalize thinks this run contains.
    # Written once: a resumed/retried submit must keep the ORIGINAL snapshot,
    # since the clips already uploaded were cut to those boundaries. A no-op
    # once the snapshot exists — seg_defs above already came FROM it in that
    # case, so this would just re-freeze the same values.
    if not run.segment_snapshot:
        run.segment_snapshot = build_segment_snapshot(seg_defs)
        session.flush()

    # Swap segments excluded from a custom stitch plan are never processed —
    # they don't get a RunSegment and are not submitted to the AI model.
    active_swap_ids = _active_swap_segment_ids(run.stitch_plan, seg_defs)

    # Ensure a RunSegment row exists for every active swap SegmentDef (idempotent).
    existing_rs = {rs.segment_def_id: rs for rs in run.run_segments}
    for sd in seg_defs:
        if sd.action == "swap" and sd.id in active_swap_ids and sd.id not in existing_rs:
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
    # When GPT Image 2 preprocessing is enabled and has produced results,
    # Seedance/Gemini Omni receive the preprocessed images instead of the
    # originals (see docs/gpt-image-2-preprocess.md). Originals stay on disk
    # in reference_image_urls for audit.
    if run.preprocess_enabled and run.preprocessed_reference_urls:
        ref_source = list(run.preprocessed_reference_urls)
        # Refuse BEFORE uploading anything: a hole in this list cannot be
        # submitted, and carrying it into resolve_reference_urls wastes a full
        # set of ref uploads per attempt and then dies on None.strip(). The
        # retry endpoint routes a run in this state back into preprocessing
        # instead of here, so reaching this line means something else left it.
        #
        # Deliberately inside this branch, not above it: an EMPTY preprocessed
        # list is not a hole but the documented fallback to the originals (a
        # phase that never wrote a result, or an all-pass-through run), and
        # those submit fine — refusing them would strand exactly the runs this
        # change exists to unstick.
        hole = preprocess_hole_index(run)
        if hole is not None:
            raise ValueError(
                f"Reference {hole} has no preprocessed image — the GPT Image "
                f"phase did not finish. Re-run preprocessing for this run; "
                f"submitting would send the video model an incomplete "
                f"reference set."
            )
    else:
        ref_source = list(run.reference_image_urls or [])

    t_ref_start = time.monotonic()
    run_ref_urls = resolve_reference_urls(ref_source, kie, gdrive=gdrive)
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

    # Project-level "remove original audio": swap clips are uploaded video-only
    # so the model generates audio from the prompt instead of reacting to the
    # source soundtrack. Gemini Omni clips are muted regardless (it rejects
    # inputs that carry audio). Read here rather than snapshotted on the Run, so
    # toggling the project setting takes effect on the next submit — except for
    # a run that froze the setting at admission (Run.mute_source, AICP-365:
    # autopilot runs), see effective_mute_source.
    mute_audio = effective_mute_source(run, project) or ai_models.is_omni(run.model)
    if mute_audio:
        log.info(
            "run_id=%s submitting video-only clips (mute_source=%s, run frozen=%s, "
            "model=%s)",
            run_id, project.mute_source, getattr(run, "mute_source", None), run.model,
        )

    # ---------------------------------------------------------------
    # `generate_audio` — the explicit audio switch some models expose
    # (AIModelSpec.supports_generate_audio; Seedance 2.5 today).
    #
    # Decided here, once, because this is the only place that can see the whole
    # run: _create_swap_task gets a recipe, not a Run, so it cannot know whether
    # this run will ever stitch. The rule is "True unless we are CERTAIN the
    # generated track is discarded":
    #
    #   stitches + audio_mode="original"  → False  (the stitch overlays the
    #       source soundtrack and throws the generated one away, so paying for
    #       it only burns credits and risks artefacts bleeding into the render —
    #       this is the whole credit saving)
    #   stitches + audio_mode="seedance"  → True   (the stitch concatenates each
    #       clip's own audio; it must exist)
    #   no-stitch run (stitch_plan == []) → True   REGARDLESS of audio_mode.
    #
    # That last line is not a nicety. A no-stitch run submits and bills every
    # swap segment and produces no stitched video: the per-segment clips ARE the
    # deliverable (see _run_stitch_and_deliver's empty-plan branch, and the
    # per-segment download buttons in the run UI). audio_mode defaults to
    # "original", so deriving the switch from audio_mode alone would hand back a
    # set of SILENT clips for every no-stitch 2.5 run — a regression triggered by
    # nothing but picking the newer model, since 2.0 has no switch and always
    # writes audio.
    generate_audio = generate_audio_for_run(run)
    if ai_models.spec_for(run.model).supports_generate_audio:
        log.info(
            "run_id=%s generate_audio=%s (model=%s stitches=%s audio_mode=%s)",
            run_id, generate_audio, run.model,
            run.stitch_plan is None or bool(run.stitch_plan),
            run.audio_mode or "original",
        )

    # ---------------------------------------------------------------
    # Voice reference — the sample whose timbre the generated speech takes,
    # in place of the arbitrary voice the model otherwise invents per
    # generation (Run.reference_audio_urls; Seedance 2.5 today).
    #
    # Uploaded once per run and reused by every segment, exactly like the
    # reference images: without that, a multi-segment run would upload the same
    # voice N times AND — because kie.ai hands back a different URL each time —
    # give each segment a nominally different reference for the same file.
    #
    # Deliberately NOT sent when generate_audio is False. That combination means
    # the run stitches with audio_mode="original", so every generated track is
    # thrown away at stitch time and the reference could not reach the delivered
    # video even in principle. The API is not asked to weigh an input that
    # cannot matter; the New Run form warns before the operator gets here.
    voice_ref_urls: list[str] = []
    voice_ref_sources = list(run.reference_audio_urls or [])
    if voice_ref_sources and ai_models.supports_reference_audio(run.model):
        if generate_audio:
            voice_ref_urls = resolve_reference_urls(
                voice_ref_sources, kie, gdrive=gdrive
            )
            log.info(
                "run_id=%s voice_refs=%d resolved for model=%s",
                run_id, len(voice_ref_urls), run.model,
            )
        else:
            log.warning(
                "run_id=%s has %d voice reference(s) but generate_audio=False "
                "(stitches with audio_mode=%r, so the generated track is "
                "discarded) — NOT sending them",
                run_id, len(voice_ref_sources), run.audio_mode or "original",
            )
    elif voice_ref_sources:
        log.warning(
            "run_id=%s has %d voice reference(s) but model=%s does not accept "
            "them — ignoring",
            run_id, len(voice_ref_sources), run.model,
        )

    for sd in seg_defs:
        if sd.action != "swap" or sd.id not in active_swap_ids:
            continue
        rs = rs_map[sd.id]

        # AICP-351 §5.1 — FIRST, before the owner row is read: an open paid
        # attempt for this segment. A `started` one without its own task id
        # belongs to a sender that is gone (this job owns the run lock that
        # recovery took over by CAS) or that will find its attempt parked: it
        # is PARKED, whatever `rs.seedance_task_id` holds (an in-poll resubmit
        # keeps the predecessor there). A `parked` one waits for the operator.
        # Either way the segment is neither submitted nor polled; the run ends
        # `incomplete` through the completeness gate.
        open_att = paid_attempts.open_attempt(session, OWNER_RUN_SEGMENT, rs.id, for_update=True)
        if open_att is not None:
            if open_att.status == ATTEMPT_STARTED:
                paid_attempts.park_started(
                    session, open_att, reason="recovered: submit outcome unknown",
                    evidence=paid_attempts.default_evidence(),
                )
            if rs.status != SegmentStatus.completed:
                rs.error_message = (
                    "paid submit outcome unknown — parked: check kie.ai, then Resubmit or "
                    "Abandon on the run page")[:250]
            session.flush()
            log.error("RunSegment %s (idx %d) has an open paid attempt %s (%s) — not "
                      "submitted, not polled", rs.id, rs.index, open_att.id, open_att.status)
            continue

        # Resume: already completed with a real result → don't resubmit.
        if (
            rs.status == SegmentStatus.completed
            and rs.local_result_path
            and os.path.exists(rs.local_result_path)
        ):
            log.info("RunSegment %s (idx %d) already completed, skipping submit",
                     rs.id, rs.index)
            continue

        # A VOID-from-the-source run (docs/segment-fill-batch-design.md §7)
        # never sends a segment to the video model unless the operator chose
        # the model as that segment's fallback: every other unfinished one is
        # left as it is (a VOID fill from the source; the completeness gate
        # ends the run incomplete until it has one).
        if project_types.is_void_source_run(run) and not rs.model_fallback:
            log.info("RunSegment %s (idx %d) of VOID-source run %s: not a model fallback, "
                     "not submitted", rs.id, rs.index, run_id)
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

            if state in ("success", "fail"):
                # Terminal: settle its reported cost before any resubmit
                # below (AICP-155). Idempotent with the success path's own.
                spend_mod.settle_draw(
                    task_id, _parse_cost_credits(data),
                    scope_key=spend_mod.run_scope(run_id), kind="video",
                    sessions=get_session,
                )

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
                        rs.cost_credits = _parse_cost_credits(data)
                        # Clear any stale failure state from a prior attempt —
                        # including the source-fallback verdict, so a segment
                        # that has now generated successfully is stitched from
                        # its result rather than from the original footage.
                        rs.error_message = None
                        rs.source_fallback = False
                        try:
                            transition(rs, SegmentStatus.completed)
                        except Exception:
                            rs.status = SegmentStatus.completed
                        session.commit()
                        spend_mod.settle_draw(
                            task_id, rs.cost_credits,
                            scope_key=spend_mod.run_scope(run_id),
                            kind="video", sessions=get_session,
                        )
                        log.info(
                            "RunSegment %s (idx %d) task %s was already "
                            "success — recovered without rebilling",
                            rs.id, rs.index, task_id,
                        )
                        # Knowledge capture B2 (AICP-299), after the commit
                        # above; never raises.
                        kcap.output(
                            task_id=task_id, path=result_dst, run_id=run_id, rs_id=rs.id,
                            payload={"url": url, "resumed": True}, segment_version=True, site="B2",
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
                    # wall-clock epoch deadline (survives Redis serialisation)
                    "deadline": time.time() + RUN_SKIP_TIMEOUT_SEC,
                }
                continue
            # state == "fail" or download failed → fall through to reset+resubmit

        # Fix #9 (crash-resume): capture clip checkpoint values before resetting
        # the segment so we can skip re-cut/re-upload when they are available.
        resume_clip_url: Optional[str] = None
        resume_local_clip_path: Optional[str] = None
        if rs.status != SegmentStatus.pending:
            if _clip_audio_is_stale(rs.local_clip_path, want_muted=mute_audio):
                # The checkpointed clip was cut under a different "remove
                # original audio" setting. Reusing it would silently ignore the
                # toggle on a retry — exactly the retry an operator makes after
                # the model reacted to the source audio — so drop both
                # checkpoints and let the submit thread re-cut and re-upload.
                log.info(
                    "RunSegment %s (idx %d) checkpoint clip was cut with "
                    "mute_source=%s (project now %s) — re-cutting",
                    rs.id, rs.index, not mute_audio, mute_audio,
                )
            # Preserve any already-uploaded clip URL — this lets us skip the
            # cut+upload on resubmission (the clip is already on kie.ai storage).
            elif rs.kie_upload_url:
                resume_clip_url = rs.kie_upload_url
                log.info(
                    "RunSegment %s (idx %d) has existing kie_upload_url — "
                    "will reuse clip on resume (skip cut+upload)",
                    rs.id, rs.index,
                )
            elif rs.local_clip_path and os.path.exists(rs.local_clip_path):
                # Clip was cut but upload didn't complete; skip just the cut.
                resume_local_clip_path = rs.local_clip_path
                log.info(
                    "RunSegment %s (idx %d) clip exists at %s — "
                    "will re-upload only (skip cut)",
                    rs.id, rs.index, rs.local_clip_path,
                )

        # Retry: reset an interrupted RunSegment before resubmitting.
        if rs.status != SegmentStatus.pending:
            rs.status = SegmentStatus.pending
            rs.error_message = None
            # Clear the source-fallback verdict too. It is a statement about a
            # PAST attempt ("too long for the model that run used"), and we are
            # about to make a new one — possibly after the operator split the
            # segment or switched models. Leaving it set would make the stitch
            # substitute original footage even for a segment that just swapped
            # successfully.
            rs.source_fallback = False
            rs.seedance_task_id = None
            rs.seedance_result_url = None
            session.flush()

        # Resolve effective refs for this segment — three-way priority:
        #   1. generated_reference_urls — a per-segment GPT Image 2 rescue
        #      (RunSegmentImageJob applied via app/pipeline_segment_image.py).
        #      Exactly two entries: [character photo, the downloaded GPT
        #      Image 2 output] — a different recipe from the run-level set,
        #      not a slot filled into it.
        #   2. reference_image_urls_override — the hand override (unchanged).
        #   3. the run-level set (unchanged).
        # Both (1) and (2) are cached the same way, by rs.id: a given segment
        # can only take ONE of the two branches, so the keys never collide.
        if rs.generated_reference_urls:
            cache_key = rs.id
            if cache_key not in _override_ref_cache:
                _override_ref_cache[cache_key] = resolve_reference_urls(
                    list(rs.generated_reference_urls), kie, gdrive=gdrive
                )
            effective_ref_urls = _override_ref_cache[cache_key]
        elif rs.reference_image_urls_override:
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
            "mute_audio": mute_audio,
            # Decided above for the run as a whole; forwarded unchanged so every
            # segment of a run makes the same request (and so an in-poll
            # resubmit does too).
            "generate_audio": generate_audio,
            # Ditto — already-resolved kie URLs, so a resubmit reuses the same
            # uploaded voice rather than pushing the file again.
            "audio_ref_urls": list(voice_ref_urls),
            # Fix #9: pass checkpoint values so the submit thread can skip
            # the cut and/or upload when a prior attempt already did them.
            "resume_clip_url": resume_clip_url,
            "resume_local_clip_path": resume_local_clip_path,
            "lease": lease,
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
                            None if external_kie else get_shared_client
                        ),
                    ),
                )
                for work in submit_work
            ]
        # Collect EVERY result first. A segment whose createTask outcome is
        # unknown is PARKED (AICP-351 §5.1): like a skipped segment it gets no
        # pending entry, its siblings keep going and the completeness gate
        # ends the run `incomplete` (the run page shows the parked card). Any
        # other submit failure still fails the whole run (re-raised after the
        # loop: the existing semantics).
        first_error: Optional[BaseException] = None
        for work, fut in submit_futures:
            try:
                result = fut.result()
            except PaidSubmitParked as exc:
                log.error("run_id=%s segment idx=%d parked: %s", run_id, work["sd_index"], exc)
                with get_session() as psession:
                    prs = psession.get(RunSegment, work["rs_id"])
                    if prs is not None:
                        prs.error_message = (
                            "paid submit outcome unknown — parked: check kie.ai, then "
                            "Resubmit or Abandon on the run page")
                        psession.commit()
                continue
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                if first_error is None:
                    first_error = exc
                continue
            if result is None:
                # Segment skipped at submit time (longer than the selected
                # model's per-clip ceiling). It was already marked failed; the
                # run will be marked incomplete (not stitched).
                continue
            pending[result["task_id"]] = {
                "rs_id": work["rs_id"],
                "index": work["sd_index"],
                # wall-clock epoch deadline (survives Redis serialisation)
                "deadline": time.time() + RUN_SKIP_TIMEOUT_SEC,
                "attempt": 1,
                "recipe": result["recipe"],
            }

        if first_error is not None:
            raise first_error

    log.info(
        "run_id=%s submit_phase_total=%.1fs segments_submitted=%d",
        run_id, time.monotonic() - t_submit_start, len(submit_work),
    )

    did_submit = bool(pending)
    log.info("Submitted %d swap task(s) to Seedance for run %s",
             len(pending), run_id)

    return pending, did_submit, r_dir


# ---------------------------------------------------------------------------
# Segment snapshot helpers
#
# A Run is launched against the Project's SegmentDefs as they stand at submit
# time, but those rows keep changing underneath it — operators split segments
# and nudge boundaries while an earlier run is still processing. submit_run
# freezes them into Run.segment_snapshot so the finalize phase asks the run
# what it contains instead of asking the project what it contains today.
# See models.Run.segment_snapshot for the failure this prevents.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrozenSegmentDef:
    """A SegmentDef as it was when the run launched.

    Duck-types the SegmentDef ORM object over every field either the submit
    phase or the finalize phase touches (id / index / action / start_sec /
    end_sec / pre_roll_sec / post_roll_sec) so both
    _run_stitch_and_deliver AND _build_submit_state work with either.
    pre_roll_sec/post_roll_sec default to 0.0 for a snapshot written before
    those keys existed (every row from before this fix) — the historical
    behaviour for such a row was "submit reads live project.segments", so a
    missing key here can only be reached via the run_segment_defs fallback
    path anyway; the default only matters for a snapshot missing JUST these
    two keys, which never happens for a snapshot this dataclass itself wrote.
    """

    id: str
    index: int
    action: str
    start_sec: float
    end_sec: float
    pre_roll_sec: float = 0.0
    post_roll_sec: float = 0.0


def build_segment_snapshot(seg_defs) -> list[dict]:
    """Freeze SegmentDefs into the JSON form stored on Run.segment_snapshot.

    Includes pre_roll_sec/post_roll_sec so a later submit-phase RESUME
    (crash-resume, retry, or a run-local range regeneration candidate) can
    read them from the FROZEN snapshot instead of the project's live
    SegmentDefs — see run_segment_defs and _build_submit_state's seg_defs
    line. A project's roll values can change after a run launches exactly
    like its boundaries can; freezing them here closes the same split-
    source-of-truth gap for the same reason.
    """
    return [
        {
            "id": sd.id,
            "index": sd.index,
            "action": sd.action,
            "start_sec": sd.start_sec,
            "end_sec": sd.end_sec,
            "pre_roll_sec": getattr(sd, "pre_roll_sec", 0.0) or 0.0,
            "post_roll_sec": getattr(sd, "post_roll_sec", 0.0) or 0.0,
        }
        for sd in sorted(seg_defs, key=lambda s: s.index)
    ]


def run_segment_defs(run, project) -> list:
    """The segment defs this run was launched against, newest edits excluded.

    Prefers the run's own frozen snapshot; falls back to the project's live
    segments for runs created before the snapshot column existed (and for any
    run whose snapshot failed to write), which is the pre-existing behaviour.
    """
    snapshot = run.segment_snapshot
    if not snapshot:
        return list(project.segments)
    return [
        _FrozenSegmentDef(
            id=e["id"],
            index=e["index"],
            action=e["action"],
            start_sec=e["start_sec"],
            end_sec=e["end_sec"],
            pre_roll_sec=e.get("pre_roll_sec", 0.0) or 0.0,
            post_roll_sec=e.get("post_roll_sec", 0.0) or 0.0,
        )
        for e in sorted(snapshot, key=lambda e: e["index"])
    ]


def stitch_frame_counts(seg_defs, fps: float, *, plan, audio_mode: str) -> Optional[list]:
    """Per-segment output frame budgets for a conformed stitch, or ``None``.

    Only the default timeline (no stitch plan) with the continuous source
    soundtrack (``audio_mode == "original"``) and no pre/post-roll on any
    segment is conformed: there every clip must cover exactly its segment's
    source interval, so segment *i* gets ``F(end_i) - F(start_i)`` frames
    (``rebuild.frame_index``, half up) -- counted at the boundaries, so the budgets
    add up to the source's own frame count and rounding never accumulates.
    A generated clip that came back longer (Gemini Omni: whole 4-frame groups
    at 24 fps, up to 5 frames at 30 fps per clip) is cut at its end; before
    this every such clip pushed all later segments late against the sound
    (514067f2 / d8105ab5: 0, 3, 3, 6, 8, 10 frames late per segment). Rolled
    segments and custom plans keep today's stitch.
    """
    if plan is not None or audio_mode != "original" or not fps or fps <= 0:
        return None
    defs = list(seg_defs)            # the stitch's own entry order
    if any((getattr(sd, "pre_roll_sec", 0.0) or 0.0) > 0
           or (getattr(sd, "post_roll_sec", 0.0) or 0.0) > 0 for sd in defs):
        return None
    out = []
    for sd in defs:
        # The rebuild's own frame grid (half up), so an images-only run's
        # frame-exact keep cuts are never trimmed by a frame.
        n = (rebuild_mod.frame_index(float(sd.end_sec), fps)
             - rebuild_mod.frame_index(float(sd.start_sec), fps))
        if n < 1:
            return None
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Stitch plan helpers
#
# Run.stitch_plan semantics (see models.Run):
#   None — default: process + stitch every SegmentDef in order.
#   []   — "no stitch": process every swap segment, produce NO final video.
#   list — custom ordered plan of {"kind": "segment"|"library", ...} entries.
#          Swap SegmentDefs absent from the plan are not processed at all.
# ---------------------------------------------------------------------------


def generate_audio_for_run(run) -> bool:
    """The `generate_audio` switch (AIModelSpec.supports_generate_audio models
    only) this run's segments should request — see the long rationale where
    this used to be inlined in :func:`_build_submit_state`.

    Factored out so :mod:`pipeline_range_attempt`'s single-segment candidate
    submission makes the identical decision a full run submit would — a
    range-regenerated segment must never come back silent (or carrying an
    audio track that gets thrown away) just because it was submitted through a
    different code path.
    """
    run_stitches = run.stitch_plan is None or bool(run.stitch_plan)
    run_audio_mode = run.audio_mode or "original"
    return not (run_stitches and run_audio_mode == "original")


def _active_swap_segment_ids(plan: Optional[list], seg_defs) -> set:
    """SegmentDef ids of swap segments this run must actually process.

    The default plan (None) and the no-stitch plan ([]) both process every
    swap segment; a custom plan processes only the swap segments it includes
    (excluded ones are never submitted, so no AI credits are spent on them).
    """
    swap_ids = {sd.id for sd in seg_defs if sd.action == "swap"}
    if not plan:  # None or []
        return swap_ids
    plan_ids = {
        e.get("segment_def_id")
        for e in plan
        if isinstance(e, dict) and e.get("kind") == "segment"
    }
    return swap_ids & plan_ids


def _final_is_stale(final_path: str, seg_defs, rs_map) -> bool:
    """True if any completed swap result on disk is newer than the final video.

    Belt-and-suspenders for the delivery-only-retry fast path: a rerun segment
    is already marked completed by the time finalize_run runs (the poll phase
    completes it before enqueuing finalize), so segment state alone cannot
    distinguish "delivery-only retry" from "segment rerun just finished".
    A result file with a newer mtime than final.mp4 means the final does not
    contain it — reusing it would silently deliver the old generation.
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


# A delivery pin is removed as soon as its in-place update ends; one left
# behind belongs to a killed worker. Past the finalize job timeout (3h) plus
# slack, no live finalize can still be using it.
_DELIVERY_PIN_STALE_SEC = 4 * 3600


def _enforce_size_cap(run_id: str, path: str) -> None:
    """Shrink *path* to the delivery size cap; never fail the run over it — an
    oversized deliverable still beats no deliverable."""
    try:
        media_mod.enforce_size_limit(path)
    except Exception as exc:
        log.warning(
            "run_id=%s: could not shrink %s to the delivery size cap: %s",
            run_id, path, exc,
        )


def _sweep_stale_delivery_pins(directory: str) -> None:
    try:
        names = os.listdir(directory)
    except OSError:
        return
    now = time.time()
    for name in names:
        # .deliver-* pins and the .publish-* staging links that publish them
        # (a worker killed between link and rename leaves one behind).
        if not (name.startswith((".deliver-", ".publish-")) and name.endswith(".mp4")):
            continue
        path = os.path.join(directory, name)
        try:
            # ctime, not mtime: a hard link to an OLD final (delivery-only
            # retry) keeps that file's mtime, but creating the link updates
            # the inode's ctime, so a just-made pin is never mistaken for
            # debris.
            if now - os.stat(path).st_ctime > _DELIVERY_PIN_STALE_SEC:
                os.remove(path)
        except OSError:
            pass


class FinalizeSuperseded(RuntimeError):
    """This finalize no longer owns the run's processing lock.

    Raised under the Run-row lock right before an in-place Drive update; the
    caller returns without touching run state, because the lifecycle that DOES
    own the run is the one that must decide what is delivered.
    """


def _deliver_in_place_with_retry(gdrive: GDriveClient, file_id: str, path: str, run_id: str) -> None:
    """Replace the content of *file_id*, retrying the whole transfer (a
    resumable upload either completes or publishes nothing)."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, RUN_DELIVER_ATTEMPTS + 1):
        try:
            gdrive.update_file_content(file_id, path)
            return
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Drive in-place update attempt %d/%d failed for run %s (file %s): %s",
                attempt, RUN_DELIVER_ATTEMPTS, run_id, file_id, exc,
            )
            if attempt < RUN_DELIVER_ATTEMPTS:
                time.sleep(RUN_DELIVER_BACKOFF_SEC * attempt)
    raise last_exc  # type: ignore[misc]


def _deliver_in_place(
    session,
    run: Run,
    *,
    gdrive: GDriveClient,
    final_dst: str,
    pinned_final: str,
    owner_check: Optional[Callable[[], bool]],
) -> None:
    """Re-deliver an already-delivered run by replacing its Drive file's
    content, then mark it ``done`` — Drive update and commit in ONE locked
    transaction, the pipeline_audio_patch._promote pattern (AICP-342).

    1. ``session.refresh(run, with_for_update=True)`` locks the Run row; the
       caller has just committed ``delivering``.
    2. Refuse unless the run is still ``delivering`` with this final as its
       result and the same Drive id, and (staged path) this job still owns the
       run's processing lock — a finalize that lost it raises
       :class:`FinalizeSuperseded` and changes nothing, so a stale job can
       never put its bytes behind the exported link.
    3. ``files().update`` keeps the id; Drive keeps the previous bytes as a
       revision.
    4. ``done`` + commit, releasing the lock.

    Holding the row lock across the upload is the trade audio patch made:
    the other result mutations (rerun, restore, range attempt, audio patch)
    all need a ``done`` run, so nothing contends for it but API admissions,
    which wait and then see ``done``. Safe only while prod's
    ``idle_in_transaction_session_timeout`` / ``statement_timeout`` /
    ``lock_timeout`` are 0, as docs/run-audio-patch.md §4 records.

    A failed update (including a file deleted from Drive) fails the run with
    the final still on disk: a fresh upload would silently move the link the
    sheet holds. Accepted residual risk, the same as audio patch's: a worker
    killed between Drive acknowledging and the commit leaves Drive ahead of
    the row; the retry re-sends the same final, which is idempotent.
    """
    file_id = run.result_gdrive_file_id
    session.refresh(run, with_for_update=True)
    if run.status != RunStatus.delivering:
        raise RuntimeError(
            f"refusing in-place delivery: run status changed to {run.status!r}"
        )
    if run.result_local_path != final_dst or run.result_gdrive_file_id != file_id:
        raise RuntimeError(
            "refusing in-place delivery: the run's result or Drive file changed "
            "while this finalize was running"
        )
    if owner_check is not None and not owner_check():
        raise FinalizeSuperseded(
            f"run {run.id}: finalize lost the processing lock before the "
            "in-place Drive update — leaving delivery to the current owner"
        )
    try:
        same = os.path.samefile(pinned_final, final_dst)
    except OSError:
        same = False
    if not same:
        raise RuntimeError(
            "refusing in-place delivery: final.mp4 was replaced by another "
            "stitch while this finalize was running — Drive left untouched; "
            "retry the run"
        )
    _deliver_in_place_with_retry(gdrive, file_id, pinned_final, run.id)
    run.error_message = None
    transition(run, RunStatus.done)
    session.commit()
    log.info("run_id=%s finalize done (Drive file %s replaced in place)", run.id, file_id)


def _run_stitch_and_deliver(
    run_id: str,
    *,
    gdrive: Optional[GDriveClient],
    session,
    did_submit: bool,
    capture_links: bool = False,
    publication_id: Optional[str] = None,
    owner_check: Optional[Callable[[], bool]] = None,
) -> Optional[PublishedFinal]:
    """
    Internal: completeness gate + stitch + deliver.

    Returns the committed :class:`PublishedFinal` (AICP-299 §3.2) when the run
    reached ``done`` with a final, else ``None`` (incomplete gate, no-stitch
    plan). Exceptions propagate unchanged. ``capture_links`` enables knowledge
    link point (ii) before the stitch (``finalize_run`` sets it; the monolithic
    ``process_run`` does not); ``publication_id`` names this publication
    (minted when None).

    Must be called within an active session.  Raises on any fatal error.
    """
    publication_id = publication_id or f"stitch:{run_id}:{uuid_mod.uuid4().hex}"
    run: Run = session.get(Run, run_id)
    project: VideoProject = session.get(VideoProject, run.project_id)
    # The defs this run LAUNCHED against, not the project's current ones — see
    # run_segment_defs. Reading them live let a mid-run segment edit strand a
    # fully generated run at `incomplete`.
    seg_defs = run_segment_defs(run, project)
    rs_map = {rs.segment_def_id: rs for rs in run.run_segments}

    source = project.source_local_path
    info = media_mod.probe(source)
    width, height, fps = media_mod.get_default_target(info)
    c_dir = run_clips_dir(run_id, run.project_id)
    r_dir = run_results_dir(run_id, run.project_id)

    plan: Optional[list] = (
        list(run.stitch_plan) if run.stitch_plan is not None else None
    )
    active_swap_ids = _active_swap_segment_ids(plan, seg_defs)

    # ---------------------------------------------------------------
    # Completeness gate — never stitch a mix of swapped + original clips,
    # EXCEPT where the mix is deliberate and unavoidable.
    # Only swap segments the plan actually uses are gated on.
    #
    # A segment carrying `source_fallback` could not be generated by this run's
    # model for a deterministic reason (it is longer than the model's
    # max_clip_sec), so retrying is futile and blocking would block forever.
    # Those pass the gate and the stitch substitutes the original footage for
    # them; everything else — transient task failures, timeouts — still blocks,
    # because a retry there produces the real swap. See _skip_segment.
    # ---------------------------------------------------------------
    incomplete_idx: list[int] = []
    fallback_idx: list[int] = []
    for sd in seg_defs:
        if sd.action != "swap" or sd.id not in active_swap_ids:
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

    if fallback_idx:
        log.warning(
            "run_id=%s stitching with ORIGINAL footage for %d swap segment(s) "
            "%s — too long for %s (max %.0fs per clip). The delivered video is "
            "only partly swapped.",
            run_id, len(fallback_idx), fallback_idx,
            ai_models.spec_for(run.model).label,
            ai_models.max_clip_sec(run.model),
        )

    # ---------------------------------------------------------------
    # No-stitch runs (stitch_plan == []) — every segment succeeded; there is
    # no final video to stitch or deliver. The per-segment results stay on
    # disk / in the run-detail UI.
    # ---------------------------------------------------------------
    if plan is not None and not plan:
        run.error_message = None
        transition(run, RunStatus.done)
        session.commit()
        log.info(
            "run_id=%s stitching disabled (empty stitch plan) — "
            "run done without a final video", run_id,
        )
        return

    # ---------------------------------------------------------------
    # Stitch — unless this is a delivery-only retry.
    # ---------------------------------------------------------------
    t_stitch_start = time.monotonic()
    pinned_final: Optional[str] = None
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
        def _cut_source(sd: SimpleNamespace, why: str) -> str:
            """Cut this segment's ORIGINAL footage out of the source video.

            Shared by "keep" segments and by swap segments falling back to the
            source, so both produce a byte-identical clip at the same path —
            only one of the two branches can apply to a given index.

            Path is keyed by INDEX and BOUNDS (storage.keep_clip_path), not
            index alone — see that function's docstring for why a plain
            index-based name is unsafe once a promoted range regeneration can
            change a segment's bounds without changing its index.
            """
            if images_only:
                # Images-only (rebuild) run: a frame-exact cut on the stitch's
                # output grid, so every segment starts on its own boundary
                # frame (docs/rebuild-from-reference-design.md §6.4).
                from .pipeline_rebuild import cut_keep_frames, keep_clip_name

                return cut_keep_frames(
                    source, os.path.join(c_dir, keep_clip_name(
                        sd.index, sd.start_sec, sd.end_sec, fps)),
                    sd.start_sec, sd.end_sec, fps,
                )
            dst = keep_clip_path(run_id, stitch_project_id, sd.index, sd.start_sec, sd.end_sec)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                log.info("%s clip already exists, reusing: %s", why, dst)
                return dst
            media_mod.cut_clip(
                source, sd.start_sec, sd.end_sec, dst, fast_intermediate=True,
            )
            return dst

        def _cut_or_lookup(sd: SimpleNamespace) -> str:
            """Return the clip path for this segment (cut if needed)."""
            if sd.action == "keep":
                return _cut_source(sd, "keep")
            rs = sd.rs
            if rs is None:
                raise RuntimeError(
                    f"swap segment {sd.index} has no run segment at stitch time"
                    " — refusing to mix in the original clip"
                )
            # Deterministically un-generatable on this model (too long for its
            # max_clip_sec): the gate let the run through on the understanding
            # that we substitute the original footage here. Timing is exact —
            # the cut spans the segment's own start/end — so the soundtrack
            # stays in sync either way.
            if rs.source_fallback:
                log.info(
                    "segment %d: using original footage (swap skipped: %s)",
                    sd.index, rs.error_message,
                )
                return _cut_source(sd, "source-fallback")
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

        images_only = spend_mod.is_images_only_run(run)
        audio_mode = run.audio_mode if run.audio_mode else "original"
        # A model that produces no audio at all (Gemini Omni: its clips are sent
        # video-only and it returns silent video) leaves the original source
        # audio overlaid on top as the only sensible track. Force "original"
        # even if the run requested "seedance".
        if not ai_models.spec_for(run.model).produces_audio:
            audio_mode = "original"

        if plan is None:
            # Default plan: every segment in order, classic audio handling.
            entries: list[dict] = [
                {"kind": "segment", "segment_def_id": sd.id} for sd in seg_defs
            ]
            stitch_audio_mode = audio_mode
        else:
            # A custom plan changes the timeline (excluded segments, inserted
            # library clips), so overlaying the full original soundtrack no
            # longer lines up — concat per-clip audio instead. Each clip must
            # therefore carry its own correct audio (see _clip_for_entry).
            entries = plan
            stitch_audio_mode = "seedance"

        # Pre-resolve plan entries serially (DB lookups are not thread-safe on
        # the shared session): validate kinds, segment ids and library files.
        orm_sd_by_id = {sd.id: sd for sd in seg_defs}
        lib_paths: dict[str, str] = {}
        for pos, entry in enumerate(entries):
            kind = entry.get("kind") if isinstance(entry, dict) else None
            if kind == "segment":
                if entry.get("segment_def_id") not in orm_sd_by_id:
                    raise RuntimeError(
                        f"stitch plan entry {pos}: segment_def_id "
                        f"{entry.get('segment_def_id')!r} does not exist on "
                        "this project (segment deleted after the run was created?)"
                    )
            elif kind == "library":
                clip_id = entry.get("clip_id")
                lib = session.get(LibraryClip, clip_id)
                if lib is None:
                    raise RuntimeError(
                        f"stitch plan entry {pos}: library clip {clip_id!r} "
                        "no longer exists"
                    )
                if not lib.local_path or not os.path.exists(lib.local_path):
                    raise RuntimeError(
                        f"stitch plan entry {pos}: file for library clip "
                        f"{lib.name!r} is missing on disk"
                    )
                lib_paths[clip_id] = lib.local_path
            else:
                raise RuntimeError(
                    f"stitch plan entry {pos}: unknown kind {kind!r}"
                )

        # Snapshot everything the cut threads read into plain objects, here,
        # on this thread. The commit above expired every ORM instance, so an
        # attribute read inside a pool thread would lazily refresh it through
        # the ONE shared session — concurrent use of a Session, which raced
        # (ObjectDeletedError / wrong rows) under Python 3.12 + SQLAlchemy
        # 2.0.54 (AICP-153). The threads must not touch the session at all.
        stitch_project_id = run.project_id
        sd_by_id: dict[str, SimpleNamespace] = {}
        for entry in entries:
            if entry.get("kind") != "segment":
                continue
            sd = orm_sd_by_id[entry["segment_def_id"]]
            if sd.id in sd_by_id:
                continue
            rs_snap = None
            if sd.action != "keep":
                rs_row = rs_map.get(sd.id)
                rs = session.get(RunSegment, rs_row.id) if rs_row is not None else None
                if rs is not None:
                    rs_snap = SimpleNamespace(
                        status=rs.status,
                        local_result_path=rs.local_result_path,
                        source_fallback=rs.source_fallback,
                        error_message=rs.error_message,
                    )
            sd_by_id[sd.id] = SimpleNamespace(
                id=sd.id, index=sd.index, start_sec=sd.start_sec,
                end_sec=sd.end_sec, action=sd.action, rs=rs_snap,
            )

        def _clip_for_entry(entry: dict) -> str:
            """Resolve one plan entry to a stitchable clip path."""
            if entry["kind"] == "library":
                return lib_paths[entry["clip_id"]]
            sd = sd_by_id[entry["segment_def_id"]]
            clip = _cut_or_lookup(sd)
            # Custom plans concat per-clip audio; in "original" audio mode a
            # swap result must first get the original soundtrack for its own
            # time range muxed on (keep clips already carry it, library clips
            # bring their own).
            # An images-only run's keep clips are cut video-only (frame-exact),
            # so under a custom plan they get their source audio here too.
            if plan is not None and (sd.action == "swap" or images_only) \
                    and audio_mode == "original":
                # Keyed by the clip's PATH and bounds as well as the index
                # (AICP-342): a restored take is an older file at another path,
                # so an index-only name plus the mtime check below would reuse
                # the muxed copy of whichever take was stitched last.
                mux_key = hashlib.sha1(
                    f"{clip}|{sd.start_sec}|{sd.end_sec}".encode()
                ).hexdigest()[:12]
                muxed = os.path.join(r_dir, f"resultaud_{sd.index:04d}_{mux_key}.mp4")
                # Reuse an up-to-date muxed clip (same pattern as keep clips)
                # so a re-stitch doesn't redo the audio encode for unchanged
                # segments.
                if (
                    os.path.exists(muxed)
                    and os.path.getmtime(muxed) >= os.path.getmtime(clip)
                ):
                    log.info("muxed clip already exists, reusing: %s", muxed)
                    return muxed
                media_mod.mux_original_audio(
                    clip, source, sd.start_sec, muxed,
                    source_has_audio=info.has_audio,
                )
                return muxed
            return clip

        # Submission order is plan order, so the futures list is already
        # correctly ordered for the concat.
        with ThreadPoolExecutor(max_workers=STITCH_CUT_CONCURRENCY) as pool:
            futures = [pool.submit(_clip_for_entry, e) for e in entries]
        clip_paths: list[str] = [fut.result() for fut in futures]

        # Conform each clip to its segment's frame budget (default timeline,
        # source soundtrack, no rolls): entries are the seg_defs in order.
        frame_counts = stitch_frame_counts(
            seg_defs, fps, plan=plan, audio_mode=stitch_audio_mode,
        )
        log.info(
            "Stitching %d clips → %s (%dx%d @ %.2ffps) plan=%s audio=%s conform=%s",
            len(clip_paths), final_dst, width, height, fps,
            "custom" if plan is not None else "default", stitch_audio_mode,
            "on" if frame_counts else "off",
        )
        # Knowledge capture link point (ii) (AICP-299 §3.2): keep the previous
        # final's inode before the stitch replaces the name. Syscalls only; RQ's
        # timeout propagates; a no-op unless capture covers this project.
        if capture_links and os.path.exists(final_dst):
            kcap.link_preserve(run_id, stitch_project_id, final_dst)
        if run.result_gdrive_file_id:
            # AICP-342: an in-place re-delivery. Stitch into a PRIVATE pin
            # (unique name, nothing else writes it), cap its size there, then
            # publish it as final.mp4 by hard link + atomic rename. The upload
            # reads the pin, and _deliver_in_place refuses unless final.mp4 is
            # still the pin's inode — so another finalize's stitch can never
            # be uploaded behind the exported link, nor ours be paired with
            # a final.mp4 that is not ours.
            pin_dir = os.path.dirname(final_dst)
            _sweep_stale_delivery_pins(pin_dir)
            pinned_final = os.path.join(pin_dir, f".deliver-{uuid_mod.uuid4().hex}.mp4")
            media_mod.stitch(
                clip_paths, audio_source=source, dst=pinned_final,
                width=width, height=height, fps=fps,
                audio_mode=stitch_audio_mode, frame_counts=frame_counts,
            )
            _enforce_size_cap(run_id, pinned_final)
            staging = os.path.join(pin_dir, f".publish-{uuid_mod.uuid4().hex}.mp4")
            os.link(pinned_final, staging)
            os.replace(staging, final_dst)
        else:
            media_mod.stitch(
                clip_paths, audio_source=source, dst=final_dst,
                width=width, height=height, fps=fps,
                audio_mode=stitch_audio_mode, frame_counts=frame_counts,
            )
        run.result_local_path = final_dst

    if pinned_final is None:
        # Guard the delivery size cap. Normally a no-op — a fresh stitch is
        # already capped — but a reused final from before the cap existed
        # (delivery-only retry) is shrunk here before it goes to Drive.
        _enforce_size_cap(run_id, final_dst)
        if run.result_gdrive_file_id:
            # Delivery-only retry of an already-delivered run: nothing was
            # stitched by this finalize, so pin the final as it stands (after
            # the cap) — the same inode check then guards the upload.
            pin_dir = os.path.dirname(final_dst)
            _sweep_stale_delivery_pins(pin_dir)
            pinned_final = os.path.join(pin_dir, f".deliver-{uuid_mod.uuid4().hex}.mp4")
            os.link(final_dst, pinned_final)

    session.flush()

    # Record how much of the delivered video is original footage rather than a
    # generated swap. Written every finalize (not only when non-zero) so 0
    # positively means "checked, fully swapped" and NULL keeps meaning "not
    # recorded" — the runs list and the automation webhook read this instead of
    # re-querying every run's segments.
    run.source_fallback_segments = len(fallback_idx)

    # Backfill the probed duration once (fresh stitch or a reused final from
    # a delivery-only retry may both be missing it — e.g. a final reused from
    # before this column existed). Never let a probe failure break finalize.
    if run.result_duration_sec is None:
        try:
            run.result_duration_sec = media_mod.probe(final_dst).duration_sec
        except Exception as exc:
            log.warning(
                "run_id=%s: failed to probe result duration for %s: %s",
                run_id, final_dst, exc,
            )

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
    if run.result_gdrive_file_id:
        # Already delivered once (a segment rerun, a take restore, a retry):
        # replace the CONTENT behind the existing Drive id so the link, the
        # Job ID and an already-exported Final Creatives row keep serving the
        # run (AICP-342). Never a silent fresh upload — see
        # _deliver_in_place.
        if gdrive is None:
            gdrive = _default_gdrive()
        try:
            _deliver_in_place(
                session, run, gdrive=gdrive, final_dst=final_dst,
                pinned_final=pinned_final, owner_check=owner_check,
            )
        finally:
            try:
                os.remove(pinned_final)
            except OSError:
                pass
        log.info(
            "run_id=%s deliver_phase=%.1fs gdrive_file=%s (content replaced in place)",
            run_id, time.monotonic() - t_deliver_start, run.result_gdrive_file_id,
        )
        return PublishedFinal(run_id=run_id, project_id=run.project_id, path=final_dst,
                              kind="stitch", publication_id=publication_id)
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

    # Clear any stale failure message from a prior attempt.
    run.error_message = None
    transition(run, RunStatus.done)
    session.commit()
    log.info("run_id=%s finalize done", run_id)
    return PublishedFinal(run_id=run_id, project_id=run.project_id, path=final_dst, kind="stitch",
                          publication_id=publication_id)


# ---------------------------------------------------------------------------
# Public staged pipeline functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GPT Image reference preprocessing (optional first phase).
#
# Before refs reach Seedance/Gemini Omni, each reference image can be run
# through kie.ai's GPT Image (image-to-image) with a per-image prompt. This
# phase creates one kie.ai task per ref-with-a-prompt, persists the taskIds for
# resume/no-rebill, and returns the pending map. The poll phase (tasks.py)
# downloads results into run_preprocessed_dir() and appends them to
# Run.preprocessed_reference_urls. See docs/gpt-image-2-preprocess.md.
#
# WHICH model runs is settings.KIE_IMAGE_MODEL (default gpt-image-2.5-flare
# since 2026-09-12), read per call inside KieClient.create_image_task. The
# 2.5 variants and the original gpt-image-2 share one request and response
# schema, so nothing in this module or the poll phase varies with it, and a
# rollback is an .env edit plus a `docker compose up -d` recreate (a bare
# `restart` does not re-read env_file). Comments and docstrings elsewhere in the
# codebase still say "GPT Image 2" — read that as this configurable family,
# not as a pin to the old model id.
# ---------------------------------------------------------------------------


def _record_image_draw(run_id: str, task_id: str) -> None:
    """Measure one GPT Image task against its run's reservation (AICP-155)."""
    spend_mod.record_draw(
        spend_mod.run_scope(run_id), spend_mod.image_credits(), kind="image",
        call_key=task_id, sessions=get_session,
    )


def _preprocess_owner(run_id: str, index: int) -> str:
    return f"{run_id}:{int(index)}"


def _preprocess_slot_open(session, run_id: str, index: int) -> Optional[str]:
    """AICP-351: the status of slot *index*'s open paid attempt after recovery's
    rule ran (a ``started`` one without its own task id is PARKED), or None.
    A slot with an open attempt is never submitted again here."""
    att = paid_attempts.open_attempt(session, OWNER_RUN_PREPROCESS,
                                     _preprocess_owner(run_id, index), for_update=True)
    if att is None:
        return None
    if att.status == ATTEMPT_STARTED:
        paid_attempts.park_started(session, att, reason="recovered: image submit outcome unknown",
                                   evidence=paid_attempts.default_evidence())
    session.commit()
    return ATTEMPT_PARKED


def _paid_image_send(
    session, *, run_id: str, index: int, prompt: str, input_urls: list,
    kie: KieClient, lease: Optional["paid_attempts.SubmitLease"],
) -> Optional[str]:
    """One GPT Image createTask for preprocess slot *index* under the attempt
    protocol (AICP-351 §5.1). Commits the outer *session* first (the attempt
    is written in its own session). Returns the task id with the receipt
    committed on the attempt AND in ``run.preprocess_task_ids[index]``
    (the caller's session, committed here), or None when the outcome is
    unknown (the slot's attempt is parked; the loop goes on)."""
    session.commit()
    owner = _preprocess_owner(run_id, index)
    scope = spend_mod.run_scope(run_id)
    request = {"prompt": prompt, "input_urls": list(input_urls), "index": index}
    def fence(db) -> bool:
        # The run, locked, is still in this phase and the slot still empty.
        run = db.get(Run, run_id, with_for_update=True, populate_existing=True)
        ids = list(getattr(run, "preprocess_task_ids", None) or [])
        return (run is not None and run.status == RunStatus.preprocessing
                and (index >= len(ids) or not ids[index]))

    attempt_id = paid_attempts.begin(
        OWNER_RUN_PREPROCESS, owner, kind="image", run_id=run_id, request=request,
        credits=spend_mod.image_credits(), scope_key=scope, draw_kind="image",
        predecessor_task_id=paid_attempts.last_task_id(session, OWNER_RUN_PREPROCESS, owner),
        lease=lease, fence=fence, sessions=get_session,
    )
    paid_attempts.pre_send_check(attempt_id, lease, scope_key=scope, kind="image",
                                 sessions=get_session)
    try:
        task_id = kie.create_image_task(prompt=prompt, input_urls=list(input_urls), once=True)
    except Exception as exc:
        outcome = paid_attempts.close_after_error(attempt_id, exc, scope_key=scope,
                                                  kind="image", sessions=get_session)
        if outcome == "unknown":
            log.error("preprocess_reference_images: run %s slot %d — image submit outcome "
                      "unknown, PARKED (attempt %s): %s", run_id, index, attempt_id, exc)
            return None
        raise
    owner_ok = paid_attempts.record_outcome(
        session, attempt_id, outcome="accepted", task_id=task_id,
        credits=spend_mod.image_credits(), scope_key=scope, draw_kind="image",
    )
    if not owner_ok:
        session.commit()
        raise paid_attempts.AttemptBlocked(
            f"run {run_id} slot {index}: attempt {attempt_id} was parked by recovery while "
            f"its submit was in flight (late task {task_id})")
    run = session.get(Run, run_id)
    ids = list(run.preprocess_task_ids or [])
    ids.extend([None] * max(0, index + 1 - len(ids)))
    ids[index] = task_id
    run.preprocess_task_ids = ids
    session.commit()
    return task_id


def preprocess_reference_images(
    run_id: str,
    *,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
    session=None,
    lease: Optional["paid_attempts.SubmitLease"] = None,
) -> dict:
    """
    Create GPT Image 2 (image-to-image) task(s) for the run's reference images.

    R1-new — input_urls composition. ``reference_image_urls[0]`` is the USER
    REFERENCE image (character context, NEVER transformed); indices 1..N-1
    are TRIMS. Prompts apply to trims only; index 0 is forced to "" and is
    always pass-through.

    Mode-aware (``Run.preprocess_mode``):

    - ``per_image`` (default, also when NULL): one task per TRIM (index 1..N-1)
      with a non-empty prompt. Each task's ``input_urls = [refs[0], refs[i]]``
      (user ref + that trim) and ``prompt = prompts[i]``. ``refs[0]`` is
      pass-through (``preprocessed_reference_urls[0] = refs[0]``); empty-prompt
      trims are also pass-through. Returns ``{task_id: trim_index}``.
    - ``all_as_refs``: a SINGLE task with ``input_urls = ALL refs``
      (``[refs[0], refs[1], ..., refs[N-1]]``) and a single prompt
      (``prompts[0]`` — note: in this mode ``preprocess_image_prompts`` is a
      one-element list whose single entry is the trim prompt). Returns
      ``{task_id: 0}`` (the single generated image becomes the sole
      preprocessed ref). Empty prompt → pass-through, returns ``{}``.

    Edge case: no trims (``len(refs) <= 1``) → preprocessing is a no-op;
    returns ``{}`` (caller pauses for review regardless — see tasks.py).

    Idempotent / resume-aware: ``Run.preprocess_task_ids`` is a list parallel
    to ``reference_image_urls`` (``None`` for pass-through refs). Refs that
    already have a task_id are NOT resubmitted (no rebill) — they're returned
    as pending so the poll phase re-checks them via ``kie.get_task``. A ref
    whose image is already downloaded (``preprocessed_reference_urls[i]`` set)
    is skipped outright: neither resubmitted nor re-polled. That is what makes
    "regenerate only this image" cheap — the API clears one index and re-enters
    this phase, and every other ref falls through untouched.

    The caller owns the ``preprocessing → preprocess_review`` / ``→ failed``
    transition once the poll phase completes; this function only creates
    tasks and persists ``preprocess_task_ids``.

    Returns
    -------
    pending : dict
        ``{task_id: ref_index}`` for tasks to poll. Empty when every ref is
        pass-through (caller still pauses for review) or preprocessing is
        disabled.
    """
    if kie is None:
        kie = get_shared_client()

    def _do(session) -> dict:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        if not run.preprocess_enabled:
            log.info(
                "preprocess_reference_images: run %s has preprocess_enabled=False "
                "— skipping", run_id,
            )
            return {}

        refs = list(run.reference_image_urls or [])
        prompts = list(run.preprocess_image_prompts or [])
        if not refs:
            log.info(
                "preprocess_reference_images: run %s has no refs — skipping",
                run_id,
            )
            return {}

        mode = run.preprocess_mode or "per_image"

        # Resolve originals → public kie.ai URLs (handles local paths, Drive
        # links, and plain https URLs). Reused from pipeline_common.
        input_urls = resolve_reference_urls(refs, kie, gdrive=gdrive)

        # preprocess_task_ids is positional/parallel to reference_image_urls
        # (None for pass-through refs). On resume it already holds the task_ids
        # created in a prior run of this phase — we don't recreate them.
        task_ids: list = list(run.preprocess_task_ids or [])
        # Pad/normalise to len(refs) with None.
        if len(task_ids) < len(refs):
            task_ids.extend([None] * (len(refs) - len(task_ids)))

        pending: dict = {}

        if mode == "all_as_refs":
            # Single prompt lives at prompts[0] (length-1 list in this mode).
            single_prompt = (prompts[0] if prompts else "").strip()
            if not single_prompt:
                # Pass-through: no task; the original refs are used as-is.
                # task_ids stays all-None; preprocessed_reference_urls will be
                # filled with originals by the poll phase (or submit).
                run.preprocess_task_ids = [None] * len(refs)
                session.commit()
                log.info(
                    "preprocess_reference_images: run %s all_as_refs with empty "
                    "prompt — pass-through", run_id,
                )
                return {}
            # Already generated AND downloaded — nothing left to do. Without
            # this the resume below re-polls a task that may have expired,
            # failing the run over an image already sitting on disk. per_image
            # has always had this guard (`done[i]`); all_as_refs did not.
            done_all = list(run.preprocessed_reference_urls or [])
            if done_all and done_all[0]:
                log.info(
                    "preprocess_reference_images: run %s all_as_refs already has "
                    "a result (%s) — skipping", run_id, done_all[0],
                )
                return {}

            # Resume / no-rebill: a single task already exists.
            existing = task_ids[0] if task_ids else None
            if not existing and _preprocess_slot_open(session, run_id, 0):
                # AICP-351: an unknown-outcome submit of this image is parked.
                log.error("preprocess_reference_images: run %s all_as_refs has a parked "
                          "paid attempt — not resubmitted", run_id)
                return {}
            if existing:
                pending[existing] = 0
                log.info(
                    "preprocess_reference_images: run %s all_as_refs → existing "
                    "task %s (resume)", run_id, existing,
                )
                run.preprocess_task_ids = [existing] + [None] * (len(refs) - 1)
                session.commit()
                return pending
            # R1-new: input_urls = ALL refs (user ref at index 0 + every trim).
            # Knowledge capture A4 (AICP-299): a single call, so the intent may
            # precede it (no other paid call is pending); never raises.
            kb_attempt = kcap.image_intent(
                run_id=run_id, rs_id=None, target_kind="run_preprocess", target_id=f"{run_id}:0",
                attempt_kind="preprocess_image", origin="automatic", prompt=single_prompt,
                input_urls=input_urls, locators=refs,
                recipe_context={"ref_index": 0, "mode": "all_as_refs"}, site="A4",
            )
            run.preprocess_task_ids = [None] * len(refs)
            try:
                task_id = _paid_image_send(
                    session, run_id=run_id, index=0, prompt=single_prompt,
                    input_urls=input_urls, kie=kie, lease=lease,
                )
            except Exception as exc:
                kcap.dispatch_error(kb_attempt, exc, site="A4")
                raise
            if task_id is None:
                kcap.dispatch_error(kb_attempt, RuntimeError("outcome unknown (parked)"), site="A4")
                return {}
            run = session.get(Run, run_id)
            pending[task_id] = 0
            log.info(
                "preprocess_reference_images: run %s all_as_refs → task %s "
                "(input_urls=%d, user_ref+all trims)", run_id, task_id,
                len(input_urls),
            )
            session.commit()
            kcap.submitted(kb_attempt, task_id, site="A4")  # after the commit (P1)
            return pending

        # per_image (default) — R1-new: index 0 is the USER REF (pass-through,
        # never transformed). Trims are indices 1..N-1.
        #
        # Edge case: no trims → no-op (caller still pauses for review).
        if len(refs) <= 1:
            run.preprocess_task_ids = [None] * len(refs)
            session.commit()
            log.info(
                "preprocess_reference_images: run %s has no trims (refs=%d) "
                "— no-op", run_id, len(refs),
            )
            return {}

        # Force index 0 (user ref) to pass-through regardless of any prompt.
        task_ids[0] = None
        kb_batch: list = []  # AICP-299 A5, recorded after the commit below

        # Results already downloaded in an earlier pass of this phase. Two
        # callers re-enter preprocessing with some refs already done:
        # a worker restart mid-phase, and "regenerate just this image"
        # (api_v2.regenerate_preprocess_ref), which clears exactly one index and
        # leaves every other result standing. Either way a ref that still holds
        # its image is finished — it must not be re-billed AND must not be
        # re-polled (its kie.ai task is old and may already have expired, which
        # would fail the whole run over an image we already have on disk).
        done = list(run.preprocessed_reference_urls or [])
        done.extend([None] * max(0, len(refs) - len(done)))

        # Iterate over TRIMS only (indices 1..N-1). prompts is parallel to refs
        # with prompts[0] == "" (enforced by the API); fall back to "" if the
        # stored list is shorter.
        for i in range(1, len(refs)):
            prompt = (prompts[i] if i < len(prompts) else "").strip()
            if not prompt:
                # Pass-through: no GPT Image 2 task; original trim used as-is.
                task_ids[i] = None
                continue
            if done[i]:
                # Already generated (and downloaded) — nothing to do for this
                # ref. Leaving its task_id in place keeps the audit trail.
                log.info(
                    "preprocess_reference_images: run %s trim %d already has a "
                    "result (%s) — skipping", run_id, i, done[i],
                )
                continue
            if task_ids[i]:
                # Resume / no-rebill: task already created — re-queue for poll.
                pending[task_ids[i]] = i
                log.info(
                    "preprocess_reference_images: run %s trim %d → existing task "
                    "%s (resume)", run_id, i, task_ids[i],
                )
                continue
            if _preprocess_slot_open(session, run_id, i):
                # AICP-351: this slot's last submit has an unknown outcome — it
                # is parked for the operator, never resubmitted here.
                log.error("preprocess_reference_images: run %s trim %d has a parked paid "
                          "attempt — not resubmitted", run_id, i)
                continue
            # R1-new: input_urls = [user_ref, this_trim] — the user ref (index
            # 0) is sent as character context, the trim is what gets transformed.
            task_input_urls = [input_urls[0], input_urls[i]]
            # AICP-351: one attempt and ONE commit per image (it used to be one
            # commit after the whole loop: N billed calls per commit).
            run.preprocess_task_ids = task_ids
            task_id = _paid_image_send(
                session, run_id=run_id, index=i, prompt=prompt,
                input_urls=task_input_urls, kie=kie, lease=lease,
            )
            run = session.get(Run, run_id)
            task_ids = list(run.preprocess_task_ids or [])
            task_ids.extend([None] * max(0, len(refs) - len(task_ids)))
            if task_id is None:
                continue
            task_ids[i] = task_id
            pending[task_id] = i
            kb_batch.append({"index": i, "task_id": task_id, "prompt": prompt,
                             "input_urls": task_input_urls, "locators": [refs[0], refs[i]]})
            log.info(
                "preprocess_reference_images: run %s trim %d → task %s "
                "(input_urls=[user_ref, trim])", run_id, i, task_id,
            )

        run.preprocess_task_ids = task_ids
        session.commit()
        # Knowledge capture A5 (AICP-299): nothing is captured inside the loop
        # (earlier ids are uncommitted until here); intents and submissions are
        # recorded together after the commit. Never raises.
        kcap.intent_and_submission_batch(run_id=run_id, items=kb_batch, site="A5")
        return pending

    if session is not None:
        return _do(session)
    with get_session() as session:
        return _do(session)


def preprocess_hole_index(run: Run) -> Optional[int]:
    """Index of a reference whose GPT Image 2 image is missing, or None.

    A "hole" is a slot that preprocessing was SUPPOSED to fill and did not —
    the state that makes a run unsubmittable, because ``resolve_reference_urls``
    cannot upload a ``None``. It is deliberately mode-aware rather than a length
    check on ``preprocessed_reference_urls``, because two perfectly healthy
    shapes are shorter than ``reference_image_urls``:

    * ``all_as_refs`` collapses every trim into ONE image, so a successful run
      holds a one-element list by design;
    * a run with no trims (or with no prompt on any trim) never generates
      anything and leaves the list empty — submit then falls back to the
      originals, which is correct.

    Treating either as a hole would route a run that failed LATER (in Seedance,
    stitch or delivery) back into GPT Image 2.

    Returns the first offending index, or ``None`` when nothing is missing.
    """
    if not run.preprocess_enabled:
        return None

    refs = list(run.reference_image_urls or [])
    if not refs:
        return None
    pp = list(run.preprocessed_reference_urls or [])

    prompts = list(run.preprocess_image_prompts or [])

    if (run.preprocess_mode or "per_image") == "all_as_refs":
        # A single prompt (stored at index 0 in this mode) covers every ref and
        # owes exactly ONE generated image at pp[0]; an empty prompt is a
        # pass-through that owes nothing.
        #
        # Keyed off the PROMPT, never off preprocess_task_ids[0]: the poll phase
        # deliberately drops a task id whose result URL is gone, so "has a task
        # id" would read a run that needs regenerating as one that owes nothing
        # — and Retry would send it to submit with an all-None reference list,
        # which is the dead end this whole change exists to close.
        single_prompt = (prompts[0] if prompts else "") or ""
        if not single_prompt.strip():
            return None
        if not (pp and pp[0]):
            return 0
        return None

    # per_image: index 0 is the character photo (never generated), and a trim
    # with no prompt of its own is a pass-through. Only a PROMPTED trim owes an
    # image.
    for i in range(1, len(refs)):
        prompt = (prompts[i] if i < len(prompts) else "") or ""
        if not prompt.strip():
            continue
        if i >= len(pp) or not pp[i]:
            return i
    return None


def poll_preprocess_once(
    run_id: str,
    *,
    pending: dict,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
) -> dict:
    """
    One round of GPT Image 2 task polling. For each pending task, checks
    ``kie.get_task``; on success downloads the generated image into
    ``run_preprocessed_dir()`` and records its path in
    ``Run.preprocessed_reference_urls``; on failure (R2) the task is appended
    to the ``failed`` list — the caller transitions the run to ``failed``
    (no fallback to the original ref). Still-waiting tasks remain pending.

    Mode-aware (``Run.preprocess_mode``):

    - ``per_image`` (default): each pending task maps to one ref index; the
      downloaded image lands at ``preprocessed_reference_urls[idx]``.
      Pass-through refs (no task) are filled with the original ref.
    - ``all_as_refs``: a SINGLE task; its result becomes the sole entry of
      ``preprocessed_reference_urls`` (a one-element list). Pass-through
      (empty prompt) keeps the original refs.

    Returns a dict with two keys (R2):
    - ``"pending"``: ``{task_id: ref_index}`` for tasks still in progress.
    - ``"failed"``: list of ``{"idx", "task_id", "reason"}`` for tasks that
      failed (state=="fail", success without resultUrl, or download error).
      The caller joins the reasons into ``run.error_message`` and marks the
      run failed instead of proceeding to submit.
    """
    if kie is None:
        kie = get_shared_client()

    with get_session() as session:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        mode = run.preprocess_mode or "per_image"
        refs = list(run.reference_image_urls or [])
        pp: list = list(run.preprocessed_reference_urls or [])
        # Normalise length to match refs (None = not yet filled).
        if len(pp) < len(refs):
            pp.extend([None] * (len(refs) - len(pp)))
        task_ids = list(run.preprocess_task_ids or [])

        # Pass-through refs (no task) use the original ref directly. For
        # all_as_refs the single task covers all refs, so we only fill
        # pass-throughs when there is no task at all.
        if mode == "all_as_refs":
            if not any(task_ids):
                for i in range(len(refs)):
                    if pp[i] is None and i < len(refs):
                        pp[i] = refs[i]
        else:
            # "Pass-through" means THIS ref has no prompt of its own (index 0,
            # the character photo, never has one) — not merely "no task id yet".
            # The difference matters for a single-image regen
            # (api_v2.regenerate_preprocess_ref): it clears one slot's task id
            # and result on purpose, and filling that slot with the untouched
            # original here would quietly hand the operator an un-preprocessed
            # frame to approve. A slot that carries a prompt stays empty until
            # its own task lands.
            prompts = list(run.preprocess_image_prompts or [])
            for i in range(len(refs)):
                tid = task_ids[i] if i < len(task_ids) else None
                if tid or pp[i] is not None:
                    continue
                prompt_i = (prompts[i] if i < len(prompts) else "") or ""
                if i == 0 or not prompt_i.strip():
                    pp[i] = refs[i]

        pp_dir = run_preprocessed_dir(run_id, run.project_id)

        still_pending: dict = {}
        failed: list = []
        kb_outputs: list = []  # AICP-299 B6, recorded after the commit

        for task_id, idx in pending.items():
            try:
                data = kie.get_task(task_id)
            except Exception as exc:
                log.warning(
                    "poll_preprocess_once: run %s task %s get_task failed: %s "
                    "— will retry next round", run_id, task_id, exc,
                )
                still_pending[task_id] = idx
                continue

            state = (data.get("state") or "").lower()
            if state in ("success", "fail"):
                spend_mod.settle_draw(
                    task_id, _parse_cost_credits(data),
                    scope_key=spend_mod.run_scope(run_id), kind="image",
                    sessions=get_session,
                )
            if state == "success":
                url = _parse_result_url(data)
                if not url:
                    # R2: no resultUrl → fail (no fallback).
                    failed.append({
                        "idx": idx, "task_id": task_id,
                        "reason": "GPT Image task succeeded but returned no "
                                  "result URL",
                    })
                    log.warning(
                        "poll_preprocess_once: run %s task %s success but no "
                        "resultUrl — marking failed", run_id, task_id,
                    )
                    continue
                ext = _image_ext_from_url(url)
                dst = os.path.join(pp_dir, f"ref_{idx:04d}.{ext}")
                try:
                    kie.download_result(url, dst)
                except Exception as exc:
                    # R2: download failure → fail (no fallback).
                    #
                    # Whether the task id survives decides what a later Retry
                    # can do. A TRANSIENT failure (the CDN dropped the body
                    # mid-stream) leaves the generated image sitting on kie.ai,
                    # so keeping the id lets the retry re-download it for free —
                    # that is how run ce9b5bfd was recovered. A GONE url (4xx)
                    # is never coming back, and re-polling the same id would
                    # dead-end the run on every retry forever, so we drop the id
                    # and let the next preprocess entry generate a fresh image
                    # (one GPT Image 2 rebill for that ref — the only exit).
                    if isinstance(exc, KieResultGone):
                        if idx < len(task_ids):
                            task_ids[idx] = None
                            run.preprocess_task_ids = task_ids
                        log.warning(
                            "poll_preprocess_once: run %s ref %d result url is "
                            "gone (%s) — dropped task %s so a retry regenerates",
                            run_id, idx, exc, task_id,
                        )
                    failed.append({
                        "idx": idx, "task_id": task_id,
                        "reason": f"Failed to download GPT Image result: {exc}",
                    })
                    log.warning(
                        "poll_preprocess_once: run %s ref %d download failed "
                        "(%s) — marking failed", run_id, idx, exc,
                    )
                    continue

                if mode == "all_as_refs":
                    # Single result becomes the sole preprocessed ref.
                    pp = [dst]
                else:
                    pp[idx] = dst
                kb_outputs.append((task_id, idx, dst))
                log.info(
                    "poll_preprocess_once: run %s ref %d downloaded → %s",
                    run_id, idx, dst,
                )
            elif state == "fail":
                # R2: task failed → record reason, no fallback.
                reason = (
                    f"GPT Image task failed: "
                    f"{data.get('failMsg') or 'unknown error'}"
                )
                failed.append({
                    "idx": idx, "task_id": task_id, "reason": reason,
                })
                log.warning(
                    "poll_preprocess_once: run %s ref %d task %s FAILED "
                    "(failMsg=%s) — marking failed",
                    run_id, idx, task_id, data.get("failMsg"),
                )
            else:
                # waiting / running / unknown → keep polling
                still_pending[task_id] = idx

        run.preprocessed_reference_urls = pp
        session.commit()
    # Knowledge capture B6/T7 (AICP-299), after the round's commit; never raises.
    for task_id, idx, dst in kb_outputs:
        kcap.output(task_id=task_id, path=dst, run_id=run_id, rs_id=None, artifact_kind="reference_image",
                    attempt_kind="preprocess_image", target_kind="run_preprocess", target_id=f"{run_id}:{idx}",
                    site="B6")
    for f in failed:
        reason = str(f.get("reason") or "")
        kcap.terminal(task_id=f.get("task_id"), status="failed", site="T7",
                      error_class=("no_result_url" if "no result URL" in reason
                                   else "download_failed" if reason.startswith("Failed to download")
                                   else "provider_failed"),
                      payload={"reason": reason[:500]})
    return {"pending": still_pending, "failed": failed}


def _image_ext_from_url(url: str) -> str:
    """Best-effort image file extension from a result URL; defaults to png."""
    from urllib.parse import urlparse
    path = urlparse(url).path
    for ext in ("png", "jpg", "jpeg", "webp"):
        if path.lower().endswith(f".{ext}"):
            return "jpg" if ext == "jpeg" else ext
    return "png"


def submit_run(
    run_id: str,
    *,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
    lease: Optional["paid_attempts.SubmitLease"] = None,
) -> Optional[dict]:
    """
    Submit phase: transition the run to processing, cut + upload + create AI
    tasks for all pending swap segments.

    Returns
    -------
    pending : dict | None
        The pending-task map ``{task_id → meta}`` for the poll phase.
        Returns ``{}`` when no tasks were submitted (all segments already
        completed — delivery-only retry or nothing to do).
        Returns ``None`` if the run is already in a terminal state.

    Transitions: queued → processing.
    """
    log.info("submit_run start: run_id=%s", run_id)
    if kie is None:
        kie = get_shared_client()

    with get_session() as session:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        # If the run is already terminal, nothing to do.
        if run.status in (RunStatus.done, RunStatus.failed, RunStatus.incomplete):
            log.info(
                "submit_run: run %s already in terminal status=%r — skipping",
                run_id, run.status,
            )
            return None
        if _refuse_images_only(run, "submit_run"):
            # None = "nothing to do": run_submit_run releases the lock and
            # touches nothing else.
            return None

        try:
            pending, _did_submit, _r_dir = _build_submit_state(
                run_id, kie=kie, gdrive=gdrive, session=session, lease=lease,
            )
            return pending
        except paid_attempts.AttemptBlocked as exc:
            # The lease is lost (or a recovery parked an in-flight attempt):
            # the run belongs to another job now — write nothing (AICP-351).
            session.rollback()
            log.warning("submit_run: run %s — %s; leaving the run to its owner", run_id, exc)
            raise
        except Exception as exc:
            run = session.get(Run, run_id)
            if run and run.status not in (RunStatus.done, RunStatus.failed):
                try:
                    transition(run, RunStatus.failed)
                except Exception:
                    run.status = RunStatus.failed
            if run and not run.error_message:
                run.error_message = str(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise


def poll_run_once(
    run_id: str,
    *,
    pending: dict,
    kie: Optional[KieClient] = None,
    lease: Optional["paid_attempts.SubmitLease"] = None,
) -> dict:
    """
    Poll phase: perform ONE round of status checks on the pending-task map.

    Loads the per-run results directory from the DB, then delegates to
    poll_run_once_internal.  Returns the remaining pending map.

    Parameters
    ----------
    run_id :
        The Run UUID (used only to derive the results directory path).
    pending :
        The pending-task map ``{task_id → meta}`` from the previous round.
    kie :
        Optional pre-built KieClient; a new one is created if None.

    Returns
    -------
    still_pending : dict
        The remaining pending map.  Empty when all tasks are terminal.
    """
    if kie is None:
        kie = get_shared_client()

    with get_session() as session:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        if _refuse_images_only(run, "poll_run_once"):
            raise ImagesOnlyRunRefused(f"run {run_id} is images-only")
        r_dir = run_results_dir(run_id, run.project_id)

    log.info("poll_run_once: run_id=%s pending=%d", run_id, len(pending))
    return poll_run_once_internal(pending=pending, kie=kie, results_dir=r_dir, lease=lease)


def finalize_run(
    run_id: str,
    *,
    gdrive: Optional[GDriveClient] = None,
    publication_id: Optional[str] = None,
    owner_check: Optional[Callable[[], bool]] = None,
) -> Optional[PublishedFinal]:
    """
    Finalize phase: completeness gate + stitch (ffmpeg) + deliver to Drive.

    Returns the committed :class:`PublishedFinal` when the run reached ``done``
    with a final (AICP-299 §3.2), else ``None``; the caller
    (``tasks.run_finalize_run``) makes the knowledge publication link from it.

    Reads the run's did_submit flag from the DB (derived from segment state —
    if any segment is not in completed state, we treat it as if tasks were
    submitted in this lifecycle).  Then delegates to _run_stitch_and_deliver.

    Transitions: processing → stitching → delivering → done (or incomplete/failed).
    """
    log.info("finalize_run start: run_id=%s", run_id)

    with get_session() as session:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        # did_submit: True unless every swap segment has a completed result on disk
        # already (i.e. this is a pure delivery retry).  We recompute from DB state
        # rather than relying on the caller passing it through.
        project: VideoProject = session.get(VideoProject, run.project_id)
        seg_defs = run_segment_defs(run, project)
        rs_map = {rs.segment_def_id: rs for rs in run.run_segments}
        # Only swap segments the stitch plan actually processes count — a swap
        # segment excluded from a custom plan has no RunSegment on purpose and
        # must not force did_submit=True (which would disable the
        # delivery-only-retry fast path forever for such runs).
        active_swap_ids = _active_swap_segment_ids(run.stitch_plan, seg_defs)
        all_completed = all(
            rs_map.get(sd.id) is not None
            and rs_map[sd.id].status == SegmentStatus.completed
            and rs_map[sd.id].local_result_path
            and os.path.exists(rs_map[sd.id].local_result_path)
            for sd in seg_defs
            if sd.action == "swap" and sd.id in active_swap_ids
        )
        # If all swap segments are already completed, this might be a delivery
        # retry → did_submit=False signals the stitch phase to reuse an
        # existing final.mp4 rather than re-encoding. NOTE: on the staged
        # poll→finalize path a just-rerun segment is ALSO completed by now, so
        # this alone over-detects "delivery retry"; _run_stitch_and_deliver
        # additionally checks result-file mtimes (_final_is_stale) and
        # re-stitches when any result is newer than the existing final.
        did_submit = not all_completed

        try:
            return _run_stitch_and_deliver(
                run_id, gdrive=gdrive, session=session, did_submit=did_submit,
                capture_links=True, publication_id=publication_id,
                owner_check=owner_check,
            )
        except FinalizeSuperseded as exc:
            # Not this job's run any more: change nothing (the row lock is
            # released by the rollback) and let the owner deliver.
            log.warning("finalize_run: %s", exc)
            session.rollback()
            return None
        except Exception as exc:
            run = session.get(Run, run_id)
            if run and run.status not in (RunStatus.done, RunStatus.failed,
                                          RunStatus.incomplete):
                try:
                    transition(run, RunStatus.failed)
                except Exception:
                    run.status = RunStatus.failed
            if run and not run.error_message:
                run.error_message = str(exc)
            try:
                session.commit()
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# process_run — backward-compatible monolithic entry point
# ---------------------------------------------------------------------------


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
    A swap segment that fails transiently (task error, timeout) leaves the run
    ``incomplete`` — the completeness gate will not stitch a partially-swapped
    timeline, so no video is produced until it is retried. A segment that is
    simply too long for the run's model is instead delivered as the original
    footage (``RunSegment.source_fallback``), because retrying cannot fix it.
    Only fatal errors (missing source, stitch failure) fail the whole run.
    Phase durations are logged for observability.

    This is the monolithic backward-compatible entry point.  The production path
    uses submit_run → poll_run_once (re-enqueued) → finalize_run so workers are
    never blocked waiting.  This function drives all three phases in-process,
    which is useful for tests and manual recovery runs.
    """
    t_run_start = time.monotonic()
    log.info("process_run start: run_id=%s", run_id)

    if kie is None:
        kie = get_shared_client()

    # Submit phase — uses its own session via _build_submit_state.
    with get_session() as session:
        _run = session.get(Run, run_id)
        if _run is not None and _refuse_images_only(_run, "process_run"):
            return
    try:
        with get_session() as session:
            run: Run = session.get(Run, run_id)
            if run is None:
                raise ValueError(f"Run not found: {run_id}")
            pending, did_submit, r_dir = _build_submit_state(
                run_id, kie=kie, gdrive=gdrive, session=session
            )
        # Session is closed when the with-block exits (committed by context manager).
    except Exception as exc:
        with get_session() as err_session:
            run_err = err_session.get(Run, run_id)
            if run_err and run_err.status not in (RunStatus.done, RunStatus.failed):
                try:
                    transition(run_err, RunStatus.failed)
                except Exception:
                    run_err.status = RunStatus.failed
            if run_err and not run_err.error_message:
                run_err.error_message = str(exc)
            try:
                err_session.commit()
            except Exception:
                pass
        raise

    log.info(
        "run_id=%s submit_phase done in %.1fs",
        run_id, time.monotonic() - t_run_start,
    )

    # ---------------------------------------------------------------
    # Poll phase — loop until all tasks are terminal.
    # No DB session held while sleeping.
    # ---------------------------------------------------------------
    t_poll_start = time.monotonic()
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

    log.info(
        "run_id=%s poll_phase_total=%.1fs",
        run_id, time.monotonic() - t_poll_start,
    )

    # ---------------------------------------------------------------
    # Finalize phase.
    # ---------------------------------------------------------------
    try:
        with get_session() as fin_session:
            _run_stitch_and_deliver(
                run_id, gdrive=gdrive, session=fin_session,
                did_submit=did_submit,
            )
    except Exception as exc:
        with get_session() as err_session:
            run_err = err_session.get(Run, run_id)
            if run_err and run_err.status not in (RunStatus.done, RunStatus.failed,
                                                   RunStatus.incomplete):
                try:
                    transition(run_err, RunStatus.failed)
                except Exception:
                    run_err.status = RunStatus.failed
            if run_err and not run_err.error_message:
                run_err.error_message = str(exc)
            try:
                err_session.commit()
            except Exception:
                pass
        raise

    log.info(
        "process_run done: run_id=%s total=%.1fs",
        run_id, time.monotonic() - t_run_start,
    )


def _skip_segment(
    rs: RunSegment, reason: str, session, *, source_fallback: bool = False
) -> None:
    """Mark a RunSegment failed, with *reason* recorded for the UI.

    Two different outcomes, chosen by *source_fallback* — the distinction is the
    whole point, so pass it deliberately rather than by default:

    ``source_fallback=False`` (default) — a TRANSIENT failure: task error,
    timeout, missing result url. The completeness gate in
    :func:`_run_stitch_and_deliver` requires every active swap segment to be
    ``completed``, so the run is left ``incomplete`` with no delivered video and
    the operator retries. That is correct here: a retry produces the real swap,
    and shipping the original footage instead would silently throw it away.

    ``source_fallback=True`` — a DETERMINISTIC failure the run's model can never
    satisfy, today only "the segment is longer than the model's max_clip_sec".
    Retrying cannot help, so blocking would block forever. The gate lets the run
    through and the stitch substitutes the original source footage for this
    segment (safe on the timeline: the original clip has exactly the segment's
    duration, so nothing desyncs against the soundtrack — unlike a partial swap,
    which is why we still never do that). The delivered video is then only
    partly swapped, which the run UI and ``Run.source_fallback_segments`` both
    surface; the operator's fix is to split the segment or pick a model that can
    generate it.

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
