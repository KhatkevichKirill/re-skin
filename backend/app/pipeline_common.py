"""
pipeline_common.py — shared pipeline helpers.

Model-parameter mapping (aspect ratio / duration / resolution) and reference-URL
resolution that are independent of the Project/Run data model. Extracted here so
:mod:`app.pipeline_v2` can use them without a dependency on the (removed) v1
pipeline.

The per-model *facts* (aspect sets, clip ceilings, duration rules, resolutions)
no longer live here — they belong to :mod:`app.ai_models`, the single capability
table. What remains in this module is the mapping/coercion logic plus a handful
of aliases kept so existing importers (pipeline_v2 and the test suite) do not
have to change their import lines.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from . import ai_models
from .gdrive_client import GDriveClient
from .kie_client import KieClient
from .media import harden_media_argv
from .models import VideoProject
from .storage import ensure_under_data_dir, project_dir

# Fix #17 — concurrent reference resolution: max threads for resolving
# (downloading + uploading) reference images in parallel.
# Capped at 4 to avoid overwhelming a slow uplink; tune via env var.
_RESOLVE_REF_CONCURRENCY = int(os.getenv("RESOLVE_REF_CONCURRENCY", "4"))

# ---------------------------------------------------------------------------
# Tail edit ("Change ending") bounds
# ---------------------------------------------------------------------------

# A tail edit's cut_sec must land strictly inside (0, duration) with a small
# margin on each side — a cut at 0 or at/beyond the source duration produces
# a degenerate (empty or full-length) body clip. Lives here (rather than
# app/pipeline_tail.py, where it originated) so app/api_v2.py can return a
# synchronous 400 at submit time and app/web_v2.py can compute the UI's
# max/"Max"-button bound — both need it without importing the worker module.
# pipeline_tail.py re-imports it under its original name (`_MIN_CUT_MARGIN_SEC`)
# so its own error message and the existing tests are untouched.
TAIL_EDIT_MIN_CUT_MARGIN_SEC = 0.05


def floor2(value: float) -> float:
    """Floor *value* to 2 decimal places, tolerant of the float noise a
    subtraction like ``duration - TAIL_EDIT_MIN_CUT_MARGIN_SEC`` introduces —
    e.g. ``12.28 - 0.05 == 12.229999999999999`` in IEEE754 double precision,
    which a bare ``math.floor(x * 100) / 100`` would floor down to 12.22, one
    cent below the intended 12.23. The epsilon is far smaller than the 0.01
    resolution being rounded to, so it never masks a real fractional-cent
    difference.

    Used everywhere a number derived from a duration reaches an operator: the
    UI's duration hint and "Max" button (``app/web_v2.py``'s `floor2` Jinja
    filter is this same function), and ``create_tail_edit``'s 400 detail —
    that message names the bound with plain ``:.2f`` formatting, which
    *rounds*, not floors; naming a 12.288s source's bound as "12.29" would
    reject a retry of the exact number just shown, the same failure this
    whole feature exists to close.
    """
    return math.floor(value * 100 + 1e-9) / 100

# ---------------------------------------------------------------------------
# Seedance helpers
# ---------------------------------------------------------------------------

# Aspect ratios accepted by Seedance. Alias of the registry's canonical set —
# defined once in ai_models so the API layer and the pipeline cannot drift.
_SEEDANCE_ASPECTS = ai_models.SEEDANCE_ASPECTS


def _map_aspect(raw: Optional[str]) -> str:
    """
    Return a Seedance-compatible aspect-ratio string.

    If *raw* (e.g. ``"16:9"``) is in the allowed set, return it directly.
    Otherwise return ``"adaptive"``.
    """
    if raw and raw in _SEEDANCE_ASPECTS:
        return raw
    return "adaptive"


# Seedance rejects reference videos shorter than 1.8s; pad swap clips to this
# floor. Canonical value lives in ai_models (every family shares it).
MIN_SWAP_VIDEO_SEC = ai_models.MIN_SWAP_VIDEO_SEC


def clamp_duration_for(model: Optional[str], start: float, end: float) -> int:
    """Output duration (whole seconds) to request from *model* for this clip.

    Model-aware because the ceiling is not universal: the Seedance 2.0 family
    stops at 15s while 2.5 goes to 30s, and Gemini Omni accepts only a fixed
    set of lengths. :func:`ai_models.duration_for` owns those rules.
    """
    return ai_models.duration_for(model, start, end)


def _clamp_duration(start: float, end: float) -> int:
    """Round the clip duration to an integer in the base Seedance range [4, 15].

    Backwards-compatible shim for callers that predate per-model durations;
    prefer :func:`clamp_duration_for` so a 2.5 run is not silently truncated to
    the 2.0 ceiling.
    """
    return clamp_duration_for(ai_models.DEFAULT_MODEL, start, end)


# ---------------------------------------------------------------------------
# Gemini Omni Video helpers
# ---------------------------------------------------------------------------

# Gemini Omni accepts only a fixed set of output durations (seconds) and
# resolutions, and only landscape/portrait aspect ratios. Read off the registry
# spec so there is one definition of each.
_OMNI_SPEC = ai_models.spec_for("gemini-omni")
_OMNI_DURATIONS = _OMNI_SPEC.allowed_durations
_OMNI_RESOLUTIONS = frozenset(_OMNI_SPEC.resolutions)

# Hard Gemini Omni limit: the reference-clip trim AND the output duration must
# both be <= 10s. Kept as a named alias because it reads better at the call
# sites that are genuinely Gemini-specific; the submit-time guard is now driven
# by ai_models.max_clip_sec(run_model) instead, since every model has a ceiling.
OMNI_MAX_CLIP_SECONDS = _OMNI_SPEC.max_clip_sec


def _snap_omni_duration(start: float, end: float) -> int:
    """Snap a clip's length to the nearest allowed Gemini Omni duration."""
    return ai_models.duration_for("gemini-omni", start, end)


def _map_omni_aspect(
    raw: Optional[str], width: Optional[int], height: Optional[int]
) -> str:
    """
    Return a Gemini-compatible aspect ratio (``16:9`` or ``9:16``).

    Passes through ``raw`` if it's already one of the two; otherwise picks by
    orientation (portrait → ``9:16``, else ``16:9``).
    """
    if raw in ("16:9", "9:16"):
        return raw
    return "9:16" if (height or 0) >= (width or 0) else "16:9"


def _omni_resolution(res: Optional[str]) -> str:
    """Return a Gemini-compatible resolution, defaulting a stray value to 720p."""
    return ai_models.resolution_or_default("gemini-omni", res)


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


# ---------------------------------------------------------------------------
# Source-video frame extraction
# ---------------------------------------------------------------------------
# Shared by app/api_v2.py's frame-preview/stage-frame endpoints (request
# context — a caller-facing 404/500) and app/pipeline_segment_image.py (worker
# context — a plain exception the job's own try/except turns into a `failed`
# status). Neither raises an HTTP-flavoured error itself: the API wraps the
# exceptions below into HTTPException at the call site instead, so this module
# stays free of a FastAPI dependency.
# ---------------------------------------------------------------------------

# Fix #11: a fixed pool of striped locks (keyed by hash of cache_path) so two
# concurrent requests for the SAME frame don't each spawn their own ffmpeg
# process, without the lock map growing without bound (one Lock per distinct
# frame path for the life of the process, as the original per-path dict did).
_FRAME_LOCK_STRIPES = 64
_frame_locks = [threading.Lock() for _ in range(_FRAME_LOCK_STRIPES)]


def get_frame_lock(cache_path: str) -> threading.Lock:
    """Return the striped lock for *cache_path*."""
    return _frame_locks[hash(cache_path) % _FRAME_LOCK_STRIPES]


def extract_project_frame(project: VideoProject, t: float) -> tuple[str, float]:
    """Extract (and cache) a single JPEG frame from *project*'s source at *t* seconds.

    Returns ``(cache_path, t_norm)`` where *t_norm* is *t* after clamping into
    the video's range and quantising to 0.1 s — i.e. the timestamp the returned
    frame is actually FROM, which is not the requested *t* when the caller
    asked for a moment past the end.

    Raises ``FileNotFoundError`` when the project has no source video on disk,
    and ``RuntimeError`` (or ``subprocess.TimeoutExpired``) when ffmpeg itself
    fails — plain exceptions, not HTTPException, so a worker caller can catch
    and record them exactly like any other pipeline failure.
    """
    src = project.source_local_path
    if not src or not os.path.exists(src):
        raise FileNotFoundError(
            f"Source video not available on disk for project {project.id!r}"
        )

    pdir = project_dir(project.id)
    frames_dir = os.path.join(pdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    # Clamp t into the video's valid range BEFORE quantising. ffmpeg -ss at or
    # past EOF exits rc=0 having encoded nothing, which would rename the empty
    # mkstemp file into the cache and permanently serve a 0-byte "jpeg". Keep a
    # 0.1s margin from the end so the seek always lands on a decodable frame;
    # negative t seeks to 0 (ffmpeg accepts it, but the cache key would be odd).
    if t < 0:
        t = 0.0
    if project.duration_sec:
        t = min(t, max(0.0, project.duration_sec - 0.1))
    # Quantise to 0.1 s so equivalent timestamps share a cache entry.
    t_norm = round(t, 1)
    t_tenths = int(round(t_norm * 10))
    cache_path = os.path.join(frames_dir, f"frame_{t_tenths}.jpg")

    frame_lock = get_frame_lock(cache_path)
    with frame_lock:
        # Double-check after acquiring the lock: a racing request may have
        # written the file while we were waiting.
        if not os.path.exists(cache_path):
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=frames_dir)
            os.close(tmp_fd)
            try:
                result = subprocess.run(
                    harden_media_argv([
                        "ffmpeg", "-y",
                        "-ss", str(t_norm),
                        "-i", src,
                        "-frames:v", "1",
                        "-q:v", "5",
                        "-f", "image2",
                        tmp_path,
                    ]),
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    os.unlink(tmp_path)
                    raise RuntimeError("ffmpeg failed to extract frame")
                os.rename(tmp_path, cache_path)
            except subprocess.TimeoutExpired:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    return cache_path, t_norm


# ---------------------------------------------------------------------------
# resolve_reference_urls
# ---------------------------------------------------------------------------


def _resolve_one_ref(
    item: str,
    kie: KieClient,
    gdrive: Optional[GDriveClient],
    idx: Optional[int] = None,
) -> str:
    """Resolve a single reference item to a public URL.

    This is the per-item worker for :func:`resolve_reference_urls`.  It is
    extracted as a module-level function so it can be submitted to a
    ThreadPoolExecutor safely (lambdas / closures can cause pickling issues on
    some platforms, though in-process threads are fine; a named function is
    cleaner regardless).

    Rules mirror the original serial implementation:
    1. Plain https:// URL → return as-is.
    2. Drive share link → download to a temp file, upload to kie.
    3. Anything else → a local path, uploaded to kie — but ONLY when it
       resolves to an existing file under DATA_DIR. Anything else (a
       traversal, ``/proc/self/environ``, junk like ``T3``) raises
       ``ValueError`` before a byte is read, so it can never be published to
       kie.ai's public storage (AICP-148).
    """
    import tempfile

    if not item or not str(item).strip():
        # A None here means an upstream phase left a hole in the list — in
        # practice a GPT Image 2 result that never downloaded. Fail with a
        # message that names the slot instead of the AttributeError from
        # None.strip(), which told the operator nothing (run ce9b5bfd).
        where = f"Reference {idx}" if idx is not None else "A reference"
        raise ValueError(
            f"{where} is empty — a preprocessed image is missing; re-run the "
            "preprocessing phase rather than submitting an incomplete "
            "reference set"
        )

    item = item.strip()
    is_http = item.startswith("http://") or item.startswith("https://")
    is_drive = is_http and (
        "drive.google.com" in item or "docs.google.com" in item
    )

    if is_http and not is_drive:
        return item

    if is_drive:
        if gdrive is None:
            gdrive = _default_gdrive()
        suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            gdrive.download_file(item, tmp_path)
            return kie.upload_file(tmp_path, "charswap/refs")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Local path — must be a file the server wrote under DATA_DIR. The message
    # deliberately names the slot, not just the value, so an operator knows
    # which reference to replace.
    where = f"Reference {idx}" if idx is not None else "A reference"
    try:
        local = ensure_under_data_dir(item)
    except ValueError:
        raise ValueError(
            f"{where} ({item[:200]!r}) is neither an http(s) URL nor a file "
            "under the data directory — replace it with an uploaded image "
            "or a URL"
        ) from None
    if not os.path.isfile(local):
        raise ValueError(
            f"{where} file is missing on disk ({item[:200]!r}) — replace it "
            "with an uploaded image or a URL"
        )

    # Prefix a random token so two Runs uploading refs with the same basename
    # (e.g. "face.jpg") don't collide at the shared remote path. (The Drive
    # branch above already uploads a uniquely-named tempfile.)
    return kie.upload_file(
        local, "charswap/refs",
        file_name=f"{uuid.uuid4().hex}_{os.path.basename(local)}",
    )


def resolve_reference_urls(
    urls_or_paths: list[str],
    kie: KieClient,
    *,
    gdrive: Optional[GDriveClient] = None,
) -> list[str]:
    """
    Convert a mixed list of URLs / Drive links / local paths to public URLs.

    Rules (applied per item):
    1. Starts with ``http://`` or ``https://`` and is NOT a recognisable Drive
       share link → use as-is.
    2. Google Drive share link (contains ``drive.google.com`` or ``docs.google.com``)
       → download via *gdrive* to a temp file, then upload to kie → public URL.
    3. Anything else → treat as a local file path, upload to kie → public URL.

    Parameters
    ----------
    urls_or_paths:
        Input items to resolve.
    kie:
        KieClient for uploading local/Drive-downloaded files.
    gdrive:
        GDriveClient for downloading Drive links.  Created lazily if None and
        a Drive link is encountered.

    Fix #17 — concurrent resolution:
        Items are resolved concurrently via a ThreadPoolExecutor (up to
        ``_RESOLVE_REF_CONCURRENCY`` threads).  The returned list preserves the
        original order of ``urls_or_paths``.  Error semantics are unchanged: if
        any item fails, the exception propagates to the caller (same as the
        previous serial loop).
    """
    if not urls_or_paths:
        return []

    n = len(urls_or_paths)

    if n == 1:
        # Fast path: no thread overhead for a single item.
        return [_resolve_one_ref(urls_or_paths[0], kie, gdrive, 0)]

    # Concurrent path: submit all items, collect results in original order.
    # We use index-keyed futures so we can reconstruct the ordered list after
    # all futures settle (as_completed would lose order).
    resolved: list[Optional[str]] = [None] * n
    with ThreadPoolExecutor(max_workers=min(_RESOLVE_REF_CONCURRENCY, n)) as pool:
        index_futures = {
            pool.submit(_resolve_one_ref, item, kie, gdrive, idx): idx
            for idx, item in enumerate(urls_or_paths)
        }
        for fut in as_completed(index_futures):
            idx = index_futures[fut]
            # Re-raise on error (same semantics as the old serial loop).
            resolved[idx] = fut.result()

    return resolved  # type: ignore[return-value]
