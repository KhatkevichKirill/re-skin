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
