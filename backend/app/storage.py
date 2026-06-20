"""
storage.py — Per-job working directory helpers.

All job files live under  <DATA_DIR>/jobs/<job_id>/
  clips/    — raw cuts from the source video (keep segments + swap source clips)
  results/  — downloaded AI outputs + final stitched video
  source.*  — the local copy of the source video
"""

from __future__ import annotations

import os

# Base data directory.  Overridable via DATA_DIR env var so tests can
# redirect to a tempdir without touching anything else.
_BASE = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def _ensure(path: str) -> str:
    """Create *path* as a directory (including parents) and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def job_dir(job_id: str) -> str:
    """Root working directory for a single job."""
    return _ensure(os.path.join(_BASE, "jobs", job_id))


def clips_dir(job_id: str) -> str:
    """Directory that holds raw clips cut from the source video."""
    return _ensure(os.path.join(job_dir(job_id), "clips"))


def results_dir(job_id: str) -> str:
    """Directory that holds downloaded AI results and the final stitched video."""
    return _ensure(os.path.join(job_dir(job_id), "results"))


def source_path(job_id: str, ext: str = "mp4") -> str:
    """
    Local path for the downloaded/copied source video.

    Parameters
    ----------
    job_id:
        The job's UUID.
    ext:
        File extension (without the leading dot), e.g. ``"mp4"`` or ``"mov"``.
    """
    # Ensure the job root dir exists (source sits directly in it).
    job_dir(job_id)
    ext = ext.lstrip(".")
    return os.path.join(_BASE, "jobs", job_id, f"source.{ext}")
