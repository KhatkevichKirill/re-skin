"""
Scene detection — PySceneDetect content cuts for the subtitle-removal flow.

Face-swap projects segment on detected faces (app/face.py). Subtitle-removal
projects have no faces to key on, so ``analyze_project`` instead cuts the video
on *scene changes* — each resulting clip is one visually coherent shot, which is
exactly what the video model wants to regenerate cleanly (a cut in the middle of
a generated clip confuses it). This is the code path behind the CLI equivalent
``scenedetect --min-scene-len 4s detect-content``.

The one public function returns a contiguous, gap-free list of ``(start, end)``
second ranges covering ``[0, duration]`` — the same shape ``face.split_max_duration``
consumes — so the caller can cap long scenes at the model's per-clip limit and
build SegmentDefs uniformly. An empty return means "no cuts found" and the caller
falls back to a single full-length segment.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    *,
    fps: float | None,
    min_scene_len_sec: float = 4.0,
) -> list[tuple[float, float]]:
    """Return scene-change ranges for *video_path* as ``(start_sec, end_sec)``.

    Parameters
    ----------
    video_path:
        Local path to the (already normalized) source video.
    fps:
        The video's frame rate, used to convert *min_scene_len_sec* to the whole
        number of frames PySceneDetect's ``ContentDetector`` expects. Falls back
        to 25 fps when unknown — only affects the minimum-length rounding.
    min_scene_len_sec:
        Minimum scene length; cuts closer than this are merged (the
        ``--min-scene-len`` parameter).

    Returns
    -------
    Contiguous ``(start, end)`` ranges covering the whole video, or ``[]`` when
    PySceneDetect finds no cuts (single-shot video) — the caller then treats the
    whole video as one range.
    """
    # Imported lazily so importing this module (and the app) never requires
    # PySceneDetect unless a subtitle-removal project is actually analyzed.
    from scenedetect import ContentDetector, detect

    min_frames = max(1, round((fps or 25.0) * min_scene_len_sec))
    scenes = detect(video_path, ContentDetector(min_scene_len=min_frames))

    ranges = [(start.get_seconds(), end.get_seconds()) for start, end in scenes]
    log.info(
        "detect_scenes: %d scene(s) in %s (min_scene_len=%.1fs / %d frames)",
        len(ranges), video_path, min_scene_len_sec, min_frames,
    )
    return ranges


# ---------------------------------------------------------------------------
# Scene-cut cache (AICP-347)
# ---------------------------------------------------------------------------
#
# Whole-video localisation analysis detects scene cuts once, in the worker; the
# "Re-cut at speech pauses" action (app.speech_recut) runs later, in an API
# request, and prefers a scene cut inside a pause. Re-detecting there would
# decode the whole video inside the API, so analysis records what it found next
# to the source. The cache describes the SOURCE FILE, not an analysis: it is
# keyed by the file's path, size and mtime plus the detector setting, so a
# replaced or re-normalized source (or a changed SCENE_MIN_LEN_SECONDS) simply
# stops matching, and a failed analysis that wrote it describes the same file
# a retry would. A missing, stale or unreadable cache means "scene cuts
# unknown" — the re-cut still works, without the scene-cut preference.

SCENE_CACHE_VERSION = 1


def _source_identity(source_path: str) -> dict | None:
    import os

    try:
        st = os.stat(source_path)
    except (OSError, TypeError, ValueError):
        return None
    return {
        "path": os.path.realpath(source_path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def write_scene_cache(
    cache_path: str,
    *,
    source_path: str,
    ranges: list[tuple[float, float]],
    min_scene_len_sec: float,
) -> bool:
    """Record *ranges* (``detect_scenes`` output) for *source_path*. Atomic
    (temp file + rename). Never raises: a cache is an optimisation, and
    analysis must not fail over it. Returns True when written."""
    import json
    import os
    import tempfile

    ident = _source_identity(source_path)
    if ident is None:
        return False
    payload = {
        "version": SCENE_CACHE_VERSION,
        "source": ident,
        "min_scene_len_sec": float(min_scene_len_sec),
        "ranges": [[float(s), float(e)] for s, e in ranges],
    }
    tmp = None
    try:
        directory = os.path.dirname(cache_path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".scene_cuts.", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, cache_path)
        tmp = None
        return True
    except Exception:  # noqa: BLE001 — best effort, see docstring
        log.warning("write_scene_cache: could not write %s", cache_path, exc_info=True)
        return False
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def read_scene_cuts(
    cache_path: str, *, source_path: str | None, min_scene_len_sec: float
) -> list[float] | None:
    """Cut times (seconds, the start of every scene after the first) recorded
    for *source_path*, or ``None`` when there is no cache, it is unreadable, or
    it describes another file or detector setting. ``[]`` is a real answer: a
    single-shot video has no cuts."""
    import json

    if not source_path:
        return None
    try:
        with open(cache_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != SCENE_CACHE_VERSION:
        return None
    if data.get("source") != _source_identity(source_path):
        return None
    try:
        if abs(float(data.get("min_scene_len_sec")) - float(min_scene_len_sec)) > 1e-9:
            return None
        ranges = [(float(s), float(e)) for s, e in data.get("ranges") or []]
    except (TypeError, ValueError):
        return None
    return [s for s, _e in sorted(ranges)[1:]]
