"""Rebuild analysis publishing — shots and chunk SegmentDefs in one transaction.

docs/rebuild-from-reference-design.md §5.1. The pure rules (folding, chunking,
brand-card keeps) are in ``app/rebuild.py``; this module writes their result:

* :func:`publish_analysis` — called by ``pipeline_v2.analyze_project`` for a
  ``SEG_SHOT_CHUNKS`` project, INSIDE its transaction, after it re-checked
  under the project row lock that the project is still ``analyzing``. Replaces
  the project's shots (new ``generation``), drops the old generation's project
  keyframes and character sheets, and replaces the SegmentDefs with the chunks.
* :func:`rechunk` — the same partition re-run on the CURRENT shots without
  detection (operator "unfold" / ``force_anchor``, "Keep brand-card shots as
  original footage"). Free; the caller (wave 2a's API) holds the project row
  lock, requires ``status == ready`` and refuses while a run is active — like
  every segment edit.

Storyboard rows are kept for history: they record the ``shots_generation``
they described and simply stop matching.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional, Sequence

from sqlalchemy import delete, func, select

from . import rebuild as rb
from .models import SegmentDef, VideoProject
from .rebuild_models import ProjectStoryboard, RebuildKeyframe, RebuildShot

log = logging.getLogger(__name__)


def current_generation(db, project_id: str) -> int:
    """The project's newest shot generation (0 before the first analysis):
    the max over its shots and every storyboard's ``shots_generation``, so a
    generation number is never reused."""
    shots = db.execute(
        select(func.max(RebuildShot.generation)).where(RebuildShot.project_id == project_id)
    ).scalar()
    boards = db.execute(
        select(func.max(ProjectStoryboard.shots_generation))
        .where(ProjectStoryboard.project_id == project_id)
    ).scalar()
    return int(max(shots or 0, boards or 0))


def project_shots(db, project_id: str) -> list[RebuildShot]:
    return list(db.execute(
        select(RebuildShot).where(RebuildShot.project_id == project_id)
        .order_by(RebuildShot.index)
    ).scalars())


def _shot_models(rows: Sequence[RebuildShot]) -> list[rb.Shot]:
    return [
        rb.Shot(index=r.index, start=float(r.start_sec), end=float(r.end_sec),
                force_anchor=bool(r.force_anchor), is_brand_card=bool(r.is_brand_card))
        for r in rows
    ]


def _write_segments(db, project: VideoProject, rows: Sequence[RebuildShot], *,
                    keep_brand_cards: bool) -> list[rb.SegmentPlan]:
    groups, plan = rb.plan_segments(_shot_models(rows), keep_brand_cards=keep_brand_cards)
    by_index = {r.index: r for r in rows}
    folded = rb.folded_into_map(groups)
    for idx, anchor in folded.items():
        by_index[idx].folded_into = None if anchor is None else by_index[anchor].id
    db.execute(delete(SegmentDef).where(SegmentDef.project_id == project.id)
               .execution_options(synchronize_session=False))
    # Keep the ORM relationship collection consistent with the bulk delete.
    try:
        db.expire(project, ["segments"])
    except Exception:  # pragma: no cover - a detached project
        pass
    for i, seg in enumerate(plan):
        db.add(SegmentDef(project_id=project.id, index=i, start_sec=seg.start,
                          end_sec=seg.end, has_face=False, action=seg.action))
    db.flush()
    return plan


def publish_analysis(
    db,
    project: VideoProject,
    ranges: Sequence[tuple[float, float]],
    *,
    source_path: Optional[str] = None,
) -> dict:
    """Replace the project's shots + SegmentDefs with a new generation.

    *ranges* are ``rebuild.detect_shots`` output (contiguous, covering the
    source). Runs in the caller's transaction (no commit). Returns a summary.
    """
    gen = current_generation(db, project.id) + 1
    # Old generation: project keyframes and sheets go (they describe frames of
    # shots that no longer exist); the shot delete would cascade the shot
    # keyframes anyway, the sheets (shot_id NULL) need their own delete.
    db.execute(delete(RebuildKeyframe).where(
        RebuildKeyframe.project_id == project.id, RebuildKeyframe.owner_run_id.is_(None),
    ).execution_options(synchronize_session=False))
    db.execute(delete(RebuildShot).where(RebuildShot.project_id == project.id)
               .execution_options(synchronize_session=False))
    db.flush()
    rows = []
    for i, (s, e) in enumerate(ranges):
        row = RebuildShot(project_id=project.id, index=i, generation=gen,
                          start_sec=float(s), end_sec=float(e),
                          frame_sec=round((float(s) + float(e)) / 2.0, 6),
                          is_brand_card=False, force_anchor=False)
        db.add(row)
        rows.append(row)
    db.flush()
    plan = _write_segments(db, project, rows, keep_brand_cards=False)
    _write_shots_cache(project.id, ranges, gen, source_path)
    summary = {
        "generation": gen, "shots": len(rows),
        "anchors": sum(1 for r in rows if r.folded_into is None),
        "chunks": sum(1 for p in plan if p.action == "swap"),
    }
    log.info("rebuild analysis for project %s: %s", project.id, summary)
    return summary


def rechunk(db, project: VideoProject, *, keep_brand_cards: bool = False) -> list[rb.SegmentPlan]:
    """Re-run folding + chunking on the CURRENT shots (no detection, free).
    The caller holds the project row lock and has checked ``ready`` + no
    active run. Returns the new plan."""
    rows = project_shots(db, project.id)
    if not rows:
        raise ValueError("this project has no shots — analyze it first")
    return _write_segments(db, project, rows, keep_brand_cards=keep_brand_cards)


def _write_shots_cache(project_id: str, ranges, generation: int, source_path) -> None:
    """``rebuild/shots.json`` — detector output (a cache, rebuildable; never
    read back by the pipeline). Best effort."""
    from .storage import rebuild_project_dir, rebuild_shots_cache_path

    path = rebuild_shots_cache_path(project_id)
    tmp = None
    try:
        os.makedirs(rebuild_project_dir(project_id), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".shots.", suffix=".tmp",
                                   dir=rebuild_project_dir(project_id))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({
                "generation": generation,
                "source": os.path.realpath(source_path) if source_path else None,
                "min_shot_sec": rb.REBUILD_SHOT_MIN_SEC,
                "ranges": [[float(s), float(e)] for s, e in ranges],
            }, fh)
        os.replace(tmp, path)
        tmp = None
    except Exception:  # noqa: BLE001 - a cache must never fail analysis
        log.warning("rebuild: could not write %s", path, exc_info=True)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
