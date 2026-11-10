"""
storage.py — Per-job working directory helpers.

All job files live under  <DATA_DIR>/jobs/<job_id>/
  clips/    — raw cuts from the source video (keep segments + swap source clips)
  results/  — downloaded AI outputs + final stitched video
  source.*  — the local copy of the source video

DATA_DIR is resolved against BASE_DIR (see config.py) so the same absolute path
is used whether the process CWD is / or /app or anywhere else.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

# Resolve the data directory against BASE_DIR (same logic as config.py) so both
# api and worker containers (WORKDIR /app, volume ./data:/app/data) agree on the
# same absolute path regardless of CWD.
_BASE_DIR = Path(
    os.environ.get("APP_BASE_DIR", str(Path(__file__).parent.parent))
).resolve()

_raw = os.environ.get("DATA_DIR", "./data")
if os.path.isabs(_raw):
    _BASE = _raw
else:
    _BASE = str(_BASE_DIR / _raw)


def _ensure(path: str) -> str:
    """Create *path* as a directory (including parents) and return it."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Path containment (AICP-148)
#
# Reference lists (Run.reference_image_urls and friends) mix public URLs with
# LOCAL paths the server wrote itself — uploads, staged frames, GPT Image 2
# output. Every place that opens such a local path (serving it, uploading it
# to kie.ai's public storage) must first prove it lives under DATA_DIR, or a
# stored "/proc/self/environ" is a secrets leak.
# ---------------------------------------------------------------------------


class PathOutsideDataDir(ValueError):
    """A local path that does not resolve to somewhere under the given root."""


def _realpath_or_none(path) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    try:
        return os.path.realpath(path.strip())
    except (ValueError, OSError):  # e.g. an embedded NUL byte
        return None


def path_within(path, root: str) -> str | None:
    """Return the realpath of *path* if it resolves strictly INSIDE *root*.

    Both sides go through ``os.path.realpath``, so ``..`` segments and symlinks
    pointing out of *root* are seen for what they are; the comparison is by
    path component (``commonpath``), so ``/app/data2/x`` is not inside
    ``/app/data``. A relative *path* resolves against the CWD (``T3`` →
    ``/app/T3``) and is therefore outside. *root* itself does not count.
    Returns None when outside or unusable (non-string, empty, NUL byte).
    """
    real = _realpath_or_none(path)
    real_root = _realpath_or_none(root)
    if real is None or real_root is None or real == real_root:
        return None
    try:
        if os.path.commonpath([real, real_root]) != real_root:
            return None
    except ValueError:
        return None
    return real


def data_dir_root() -> str:
    """The resolved (realpath) DATA_DIR every server-written file lives under."""
    return os.path.realpath(_BASE)


def ensure_under_data_dir(path) -> str:
    """Return the realpath of *path*, or raise :class:`PathOutsideDataDir`.

    Fail-closed guard for any code about to open a path that came out of the
    database. Does not check that the file exists — callers still do that.
    """
    real = path_within(path, _BASE)
    if real is None:
        raise PathOutsideDataDir(f"{path!r} is not a file under the data directory")
    return real


# ---------------------------------------------------------------------------
# v2 — VideoProject / Run storage helpers
# ---------------------------------------------------------------------------


def project_dir(project_id: str) -> str:
    """Root working directory for a single VideoProject."""
    return _ensure(os.path.join(_BASE, "projects", project_id))


def project_source_path(project_id: str, ext: str = "mp4") -> str:
    """
    Local path for the downloaded/copied source video of a VideoProject.

    Parameters
    ----------
    project_id:
        The project's UUID.
    ext:
        File extension (without the leading dot), e.g. ``"mp4"`` or ``"mov"``.
    """
    project_dir(project_id)
    ext = ext.lstrip(".")
    return os.path.join(_BASE, "projects", project_id, f"source.{ext}")


def scene_cuts_cache_path(project_id: str) -> str:
    """Where whole-video analysis records the source's detected scene cuts
    (AICP-347, :func:`app.scene.write_scene_cache`). Pure path — creates
    nothing, so the API can look for it without side effects."""
    return os.path.join(_BASE, "projects", project_id, "scene_cuts.json")


def run_dir(run_id: str, project_id: str) -> str:
    """Root working directory for a single Run (parent of clips/ and results/).

    Unlike the other helpers this does NOT create the directory — it's used for
    deletion, where creating it would be pointless.
    """
    return os.path.join(_BASE, "projects", project_id, "runs", run_id)


def run_clips_dir(run_id: str, project_id: str) -> str:
    """Directory that holds raw clips cut from the project source for a Run."""
    return _ensure(os.path.join(_BASE, "projects", project_id, "runs", run_id, "clips"))


def keep_clip_path(
    run_id: str, project_id: str, index: int, start_sec: float, end_sec: float,
) -> str:
    """Path for a run's cached "original footage" clip at *index* — used for
    both `keep` segments and swap segments substituting source footage
    (source_fallback) — keyed by INDEX **and** its BOUNDS, not index alone.

    A promoted run-local range regeneration can change a `keep` neighbour's
    bounds without changing its index (moving the shared boundary is the
    whole feature — see pipeline_range_attempt.py). Every reader of this
    cache (pipeline_v2._cut_source, pipeline_range_attempt._resolve_keep_clip
    and pipeline_range_attempt._promote's cache warm-up) trusts a file's
    presence alone, with no other check — an index-only name would let a
    file cut for the OLD bounds be silently reused for the NEW ones just
    because something already exists at that index. Baking the bounds into
    the filename makes that structurally impossible: a promotion's new
    bounds always resolve to a path nothing has written yet (or, if this
    promotion's own warm-up already wrote it, one holding exactly the right
    bytes), while the OLD path — never written to by a promotion, since a
    promotion only ever WRITES the NEW-bounds path — stays exactly correct
    for as long as the OLD snapshot remains the run's active one. A stale
    file simply becomes a harmless orphan, never a silent wrong answer; a
    missing one is always safe to re-cut, since this only ever holds a plain
    cut of ``VideoProject.source_local_path``, never a transcoded/rendered
    artifact, so cutting the same bounds twice is always byte-identical.

    Bounds are rounded to whole milliseconds so float formatting can't split
    one logical bound across two filenames.
    """
    start_ms = round(start_sec * 1000)
    end_ms = round(end_sec * 1000)
    return os.path.join(
        run_clips_dir(run_id, project_id),
        f"clip_{index:04d}_{start_ms:d}_{end_ms:d}.mp4",
    )


def run_results_dir(run_id: str, project_id: str) -> str:
    """Directory that holds AI result clips and the final stitched video for a Run."""
    return _ensure(os.path.join(_BASE, "projects", project_id, "runs", run_id, "results"))


def run_preprocessed_dir(run_id: str, project_id: str) -> str:
    """Directory that holds GPT Image 2 preprocessed reference images for a Run.

    Populated by the optional preprocessing phase (see
    docs/gpt-image-2-preprocess.md). Files are named ``ref_<i:04d>.<ext>`` where
    *i* is the index into ``Run.reference_image_urls``.
    """
    return _ensure(
        os.path.join(_BASE, "projects", project_id, "runs", run_id, "preprocessed")
    )


# ---------------------------------------------------------------------------
# Staged source-video frames ("Add frame from source video")
#
#   <DATA_DIR>/projects/<pid>/staged_frames/frame_<t_tenths>_<uuid8>.jpg
#
# The timestamp the operator cut the frame at is baked into the FILENAME, and
# that is the only place it survives: a staged frame becomes an ordinary entry
# in Run.reference_image_urls (copied into the run's references/ dir, basename
# kept) with no column of its own.  So the name is the contract — the builder
# and the parser below are the two halves of it.  The uuid is REQUIRED to
# parse: a plain "frame_120.jpg" an operator happened to upload is not a staged
# frame, and mislabelling it would be worse than saying nothing.
# ---------------------------------------------------------------------------

_STAGED_FRAME_RE = re.compile(r"^frame_(\d+)_[0-9a-f]{8}\.jpe?g$", re.IGNORECASE)


def staged_frame_name(t_sec: float) -> str:
    """Filename for a frame staged at *t_sec* seconds (quantised to 0.1 s).

    The uuid suffix keeps repeated "Add frame" at the same timestamp from
    colliding; ``staged_frame_timestamp`` reads the timestamp back out.
    """
    return f"frame_{int(round(round(t_sec, 1) * 10))}_{uuid.uuid4().hex[:8]}.jpg"


def staged_frame_timestamp(path: str | None) -> float | None:
    """Seconds into the source video a reference was cut from, or None.

    *path* is any reference image path/URL: staged frames carry their timestamp
    in the basename, everything else — uploads, outside URLs, GPT Image 2
    output — returns None and simply gets no label.
    """
    if not path:
        return None
    base = os.path.basename(str(path).replace("\\", "/").split("?", 1)[0])
    m = _STAGED_FRAME_RE.match(base)
    if not m:
        return None
    return int(m.group(1)) / 10.0


def staged_frame_time_label(path: str | None) -> str | None:
    """``staged_frame_timestamp`` rendered for the UI ("12.4s"), or None."""
    t = staged_frame_timestamp(path)
    return None if t is None else f"{t:.1f}s"


# ---------------------------------------------------------------------------
# Stitch library — global reusable clips (e.g. alternative outros)
#
#   <DATA_DIR>/library/<clip_id>/clip.<ext>
# ---------------------------------------------------------------------------


def library_clip_dir(clip_id: str) -> str:
    """Root directory for a single LibraryClip's file."""
    return _ensure(os.path.join(_BASE, "library", clip_id))


# ---------------------------------------------------------------------------
# PIP Insert — merge working directory
#
#   <DATA_DIR>/projects/<target_project_id>/merges/<merge_id>/
#       mask.png     — rounded-rect alpha mask (only when corner_radius_px > 0)
#       merged.mp4   — the composited output
# ---------------------------------------------------------------------------


def merge_dir(merge_id: str, target_project_id: str) -> str:
    """Root working directory for a single InsertMerge (mask.png + merged.mp4)."""
    return _ensure(
        os.path.join(_BASE, "projects", target_project_id, "merges", merge_id)
    )


# ---------------------------------------------------------------------------
# Tail Edit — "Change ending" working directory
#
#   <DATA_DIR>/projects/<project_id>/tail_edits/<edit_id>/
#       body.mp4                     — the cut prefix of the run's final video
#                                       (deleted once the output is stitched)
#       {run_id}_{clip_name}.mp4     — the final output (body [+ appended
#                                       library clip]); {run_id}_trim.mp4 when
#                                       no library clip is appended. See
#                                       pipeline_tail._output_filename.
# ---------------------------------------------------------------------------


def tail_edit_dir(edit_id: str, project_id: str) -> str:
    """Root working directory for a single RunTailEdit (body.mp4 + final output)."""
    return _ensure(
        os.path.join(_BASE, "projects", project_id, "tail_edits", edit_id)
    )


def tail_edit_body_path(edit_id: str, project_id: str) -> str:
    """The edit's body.mp4 intermediate. Does NOT create the directory — the
    disk cleanup sweep (app/disk_cleanup.py) only checks and deletes it."""
    return os.path.join(_BASE, "projects", project_id, "tail_edits", edit_id, "body.mp4")


# ---------------------------------------------------------------------------
# Captions — "Add Captions" working directory
#
#   <DATA_DIR>/projects/<project_id>/captions/<caption_id>/
#       captions.ass              — the rendered ASS subtitle file (kept: it
#                                     is the evidence for a "why does this
#                                     caption say that" bug report)
#       {run_id}_captioned.mp4     — the delivered output
# ---------------------------------------------------------------------------


def caption_dir(caption_id: str, project_id: str) -> str:
    """Root working directory for a single RunCaption (captions.ass + delivered output)."""
    return _ensure(
        os.path.join(_BASE, "projects", project_id, "captions", caption_id)
    )


# ---------------------------------------------------------------------------
# Overlay feature — standalone overlay-burn jobs
#
#   <DATA_DIR>/overlays/<overlay_job_id>/
#       assets/   — uploaded PNGs (card_01.png, …)
#       output/   — result.mp4
#
# `asset_path` in the overlays JSON is stored relative to overlay_job_dir() so a
# job survives a container move; the pipeline resolves it at run time.
# ---------------------------------------------------------------------------


def overlay_job_dir(job_id: str) -> str:
    """Root working directory for a single overlay job."""
    return _ensure(os.path.join(_BASE, "overlays", job_id))


def resolve_overlay_asset(job_id: str, rel) -> str | None:
    """Realpath of overlay asset *rel* if it is an existing file inside the
    job's own directory, else None (AICP-160).

    *rel* is operator-supplied JSON (``asset_path``): an absolute path, ``..``
    segments, or a symlink planted inside the job dir would otherwise let any
    readable file on disk be burned into a render. Uses :func:`path_within`
    (realpath + commonpath), the same guard as reference images.
    """
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    real = path_within(os.path.join(overlay_job_dir(job_id), rel), overlay_job_dir(job_id))
    if real is None or not os.path.isfile(real):
        return None
    return real


def overlay_assets_dir(job_id: str) -> str:
    """Directory that holds the uploaded PNG assets for an overlay job."""
    return _ensure(os.path.join(overlay_job_dir(job_id), "assets"))


def overlay_output_dir(job_id: str) -> str:
    """Directory that holds the burned-in result video for an overlay job."""
    return _ensure(os.path.join(overlay_job_dir(job_id), "output"))


# ---------------------------------------------------------------------------
# Overlay asset library — global reusable PNG cards (e.g. branded logo bugs)
#
#   <DATA_DIR>/overlay_library/<asset_id>/card.png
#
# Mirrors library_clip_dir(); see models.OverlayAsset for why attaching one to
# a job copies the file instead of storing a soft reference.
# ---------------------------------------------------------------------------


def overlay_asset_library_dir(asset_id: str) -> str:
    """Root directory for a single OverlayAsset's file."""
    return _ensure(os.path.join(_BASE, "overlay_library", asset_id))


# ---------------------------------------------------------------------------
# Voice reference library — global reusable voice samples for Seedance 2.5
#
#   <DATA_DIR>/voice_library/<voice_id>/voice.mp3
#
# Mirrors overlay_asset_library_dir(): picking a voice for a run COPIES the
# file into the run's own dir (see run_voice_dir), so deleting a library voice
# can never strand an in-flight or re-run generation. See models.VoiceClip.
# ---------------------------------------------------------------------------


def voice_clip_dir(voice_id: str) -> str:
    """Root directory for a single VoiceClip's file."""
    return _ensure(os.path.join(_BASE, "voice_library", voice_id))


def run_voice_dir(run_id: str, project_id: str) -> str:
    """Directory holding a run's own copy of its voice reference."""
    return _ensure(os.path.join(run_dir(run_id, project_id), "voice"))


# ---------------------------------------------------------------------------
# Per-segment GPT Image 2 regeneration — one directory per attempt
#
#   <DATA_DIR>/projects/<pid>/runs/<run_id>/segment_images/<rs_id>/<job_id>/
#       frame.<ext>   — the source frame the worker extracted and copied in
#                        (never referenced from the shared project frame cache,
#                        so a retry never depends on it surviving)
#       result.<ext>  — the downloaded GPT Image 2 output
# ---------------------------------------------------------------------------


def segment_image_job_dir(job_id: str, rs_id: str, run_id: str, project_id: str) -> str:
    """Root working directory for one RunSegmentImageJob attempt."""
    return _ensure(
        os.path.join(
            _BASE, "projects", project_id, "runs", run_id,
            "segment_images", rs_id, job_id,
        )
    )


# ---------------------------------------------------------------------------
# Run-local segment range regeneration — one directory per attempt
#
#   <DATA_DIR>/projects/<pid>/runs/<run_id>/range_attempts/<rs_id>/<attempt_id>/
#       clip.mp4          — the candidate swap input, cut from the ORIGINAL
#                            source at the candidate's exact [start, end)
#       result.mp4        — the downloaded generation for clip.mp4
#       keep_<sd_id>.mp4  — an adjacent `keep` segment re-cut at the new
#                            shared boundary (0, 1 or 2 files)
#       final.mp4         — the candidate's re-stitched final video, if the
#                            run's stitch_plan produces one
#
# Deliberately namespaced by rs_id (the swap segment being resized) AND
# attempt_id, mirroring segment_image_job_dir: never the run's own
# index-based clip_%04d.mp4 / result_%04d.mp4 paths, so a candidate can never
# collide with (or be mistaken for) the clip already cut for the run's
# ACTIVE bounds. See models.RunSegmentRangeAttempt.
# ---------------------------------------------------------------------------


def range_attempt_dir(attempt_id: str, rs_id: str, run_id: str, project_id: str) -> str:
    """Root working directory for one RunSegmentRangeAttempt."""
    return _ensure(
        os.path.join(
            _BASE, "projects", project_id, "runs", run_id,
            "range_attempts", rs_id, attempt_id,
        )
    )


def segment_take_dir(rs_id: str, run_id: str, project_id: str) -> str:
    """Directory holding the preserved previous takes of one RunSegment
    (AICP-342, :class:`app.models.RunSegmentTake`).

    Outside ``results/`` on purpose: the pipeline rewrites
    ``results/result_%04d.mp4`` on every regeneration, and a take must never
    sit at a path anything writes to again. Removed with the run directory
    when the Run is deleted.
    """
    return _ensure(
        os.path.join(
            _BASE, "projects", project_id, "runs", run_id, "segment_takes", rs_id,
        )
    )


def audio_patch_dir(patch_id: str, run_id: str, project_id: str) -> str:
    """Working directory for one RunAudioPatch's candidate video.

    Deliberately NOT the run's ``results/`` directory: on promotion the Run is
    re-pointed at the file in here rather than having ``final.mp4`` overwritten,
    so the pre-patch delivery survives on disk (it is what
    ``RunAudioPatch.base_result_path`` names) and a failed patch cannot leave a
    half-written file where the active result used to be. Same reasoning as
    :func:`range_attempt_dir`.
    """
    return _ensure(
        os.path.join(
            _BASE, "projects", project_id, "runs", run_id,
            "audio_patches", patch_id,
        )
    )


def reframe_job_dir(job_id: str, project_id: str) -> str:
    """Directory of one reframe job (docs/reframe-design.md §4): the prepared
    input, the pinned audio, the provider task's raw output and the renders.

    Absolute (the provider layer refuses a relative destination) and NOT
    created here: only the create endpoint makes it, so a worker of a deleted
    job can never resurrect it on disk."""
    return os.path.join(data_dir_root(), "projects", project_id, "reframe", job_id)


def reframe_render_dir(render_id: str, job_id: str, project_id: str) -> str:
    """Directory of one reframe render (not created here)."""
    return os.path.join(reframe_job_dir(job_id, project_id), "renders", render_id)


# ---------------------------------------------------------------------------
# Rebuild from reference (docs/rebuild-from-reference-design.md §4 "Files").
# Pure paths — none of them creates anything. None is under a path
# disk_cleanup.sweep deletes (raw.mp4 is the only copy of a paid output);
# tests/test_rebuild_core.py pins that.
# ---------------------------------------------------------------------------


def rebuild_project_dir(project_id: str) -> str:
    return os.path.join(data_dir_root(), "projects", project_id, "rebuild")


def rebuild_shots_cache_path(project_id: str) -> str:
    """Detector output of the last analysis (a cache, rebuildable)."""
    return os.path.join(rebuild_project_dir(project_id), "shots.json")


def rebuild_frame_path(project_id: str, shot_id: str) -> str:
    """The source frame extracted for a shot's keyframe."""
    return os.path.join(rebuild_project_dir(project_id), "frames", f"{shot_id}.jpg")


def rebuild_keyframe_path(project_id: str, keyframe_id: str) -> str:
    """A GPT Image result (every attempt keeps its own file)."""
    return os.path.join(rebuild_project_dir(project_id), "keyframes", f"{keyframe_id}.png")


def rebuild_run_chunk_dir(run_id: str, project_id: str, chunk_index: int) -> str:
    """Run-owned keyframe copies of one chunk (frozen at launch)."""
    return os.path.join(run_dir(run_id, project_id), "rebuild", str(int(chunk_index)))


def rebuild_attempt_dir(attempt_id: str, run_id: str, project_id: str) -> str:
    """raw.mp4, fitted.mp4 and sheet.jpg of one chunk attempt (immutable)."""
    return os.path.join(run_dir(run_id, project_id), "rebuild", "attempts", attempt_id)
