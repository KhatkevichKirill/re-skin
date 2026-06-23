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


def run_dir(run_id: str, project_id: str) -> str:
    """Root working directory for a single Run (parent of clips/ and results/).

    Unlike the other helpers this does NOT create the directory — it's used for
    deletion, where creating it would be pointless.
    """
    return os.path.join(_BASE, "projects", project_id, "runs", run_id)


def run_clips_dir(run_id: str, project_id: str) -> str:
    """Directory that holds raw clips cut from the project source for a Run."""
    return _ensure(os.path.join(_BASE, "projects", project_id, "runs", run_id, "clips"))


def run_results_dir(run_id: str, project_id: str) -> str:
    """Directory that holds AI result clips and the final stitched video for a Run."""
    return _ensure(os.path.join(_BASE, "projects", project_id, "runs", run_id, "results"))
