"""Periodic disk cleanup of regenerable media intermediates.

The VPS disk filled to 100% on 2026-10-27; 75 GB of it was project media, and
none of it was orphaned — every directory belonged to a live project/run. What
can go is media the pipeline knows how to rebuild for free:

- ``tail_edits/<edit>/body.mp4`` — the cut prefix of the run's final, made by
  pipeline_tail on every (re)run of the edit. pipeline_tail now deletes it as
  soon as the output is stitched; the sweep catches ones left behind by a crash
  and the ones made before that change.
- ``runs/<run>/clips/`` of a long-done run — plain cuts of the project source.
  Every reader re-cuts a missing clip: keep/source-fallback clips via
  storage.keep_clip_path's callers, swap clips via the resume check in
  pipeline_v2 (``os.path.exists`` before reusing ``local_clip_path``).

Each deletion re-checks its row under ``SELECT ... FOR UPDATE SKIP LOCKED``
and commits right after, so it is serialised against whatever would start
using the file again: a retry flips the tail edit's status (an UPDATE on the
locked row), and a rerun or range regeneration locks or updates the Run row
(api_v2._lock_run_for_result_mutation). Whichever side commits first wins; a
locked row is simply skipped until the next tick.

- ``runs/<run>/segment_takes/<rs>/<take>.mp4`` (+ its ``<take>_refs/``) older
  than ``SEGMENT_TAKE_RETENTION_DAYS`` by ``RunSegmentTake.created_at``
  (AICP-342): previous takes kept for restore. Not regenerable, so this is a
  retention policy rather than a cache sweep; the row stays as history with
  ``file_purged_at`` set and a restore of it 409s. Skipped while the run is
  not idle (a restore admission holds, and a queued/processing run may be
  stitching from, its take files) and whenever the take is some segment's
  current result. A restore copies the take's reference files into the
  segment's own refs dir, so nothing live points into a take's ``_refs/``.

Never touched: ``results/`` (paid generations, and the run's final), the
project source, captions, overlays, ``projects/<pid>/reframe/`` (the paid
reframe input, its raw output and renders; AICP-251) and every other artifact.

Driven by ``app.tasks.run_disk_cleanup_tick``.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Optional

from sqlalchemy import select

from .models import Run, RunSegment, RunSegmentRangeAttempt, RunSegmentTake, RunTailEdit
from .state_machine import RANGE_ATTEMPT_ACTIVE_STATUSES, RunStatus, TailEditStatus
from .storage import run_dir, tail_edit_body_path

log = logging.getLogger(__name__)

# A done run's clips/ are kept this long after the run (and every clip in it)
# was last touched. 0 disables the clips sweep.
RUN_CLIPS_RETENTION_DAYS = int(os.getenv("RUN_CLIPS_RETENTION_DAYS", "30"))

# A preserved segment take's file is kept this long after the take was made
# (AICP-342, Operator 2026-10-30). 0 disables the take sweep.
SEGMENT_TAKE_RETENTION_DAYS = int(os.getenv("SEGMENT_TAKE_RETENTION_DAYS", "30"))

# A run in one of these may be reading take files (restore re-stitch) — skip.
_RUN_IDLE = (RunStatus.done, RunStatus.failed, RunStatus.incomplete)

# A finished tail edit's body.mp4 is left alone for this long, so a sweep never
# races an edit that has only just finished.
TAIL_BODY_GRACE = timedelta(hours=1)

_TAIL_EDIT_TERMINAL = (TailEditStatus.done, TailEditStatus.failed)


class SweepResult(NamedTuple):
    tail_bodies: int = 0
    run_clip_dirs: int = 0
    freed_bytes: int = 0
    segment_takes: int = 0


def _dir_stats(path: str) -> tuple[int, float]:
    """Total size and newest mtime of the files under *path* (0, 0.0 if empty)."""
    total, newest = 0, 0.0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                st = os.stat(os.path.join(root, name))
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return total, newest


def _sweep_tail_bodies(session, now: datetime) -> tuple[int, int]:
    rows = session.execute(
        select(RunTailEdit.id, Run.project_id)
        .join(Run, Run.id == RunTailEdit.run_id)
        .where(
            RunTailEdit.status.in_(_TAIL_EDIT_TERMINAL),
            RunTailEdit.updated_at < now - TAIL_BODY_GRACE,
        )
    ).all()
    count = freed = 0
    for edit_id, project_id in rows:
        path = tail_edit_body_path(edit_id, project_id)
        if not os.path.exists(path):
            continue
        try:
            locked = session.scalar(
                select(RunTailEdit.id)
                .where(
                    RunTailEdit.id == edit_id,
                    RunTailEdit.status.in_(_TAIL_EDIT_TERMINAL),
                )
                .with_for_update(skip_locked=True)
            )
            if locked is None:
                continue
            try:
                size = os.path.getsize(path)
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                log.warning("disk cleanup: could not remove %s: %s", path, exc)
                continue
            count += 1
            freed += size
        finally:
            session.commit()  # release the row lock
    return count, freed


def _sweep_run_clips(session, now: datetime) -> tuple[int, int]:
    if RUN_CLIPS_RETENTION_DAYS <= 0:
        return 0, 0
    cutoff = now - timedelta(days=RUN_CLIPS_RETENTION_DAYS)
    rows = session.execute(
        select(Run.id, Run.project_id).where(
            Run.status == RunStatus.done, Run.updated_at < cutoff,
        )
    ).all()
    count = freed = 0
    for run_id, project_id in rows:
        clips = os.path.join(run_dir(run_id, project_id), "clips")
        if not os.path.isdir(clips):
            continue
        try:
            locked = session.scalar(
                select(Run.id)
                .where(Run.id == run_id, Run.status == RunStatus.done)
                .with_for_update(skip_locked=True)
            )
            if locked is None:
                continue
            # A range regeneration reads (and re-cuts into) the run's keep
            # clips while the run itself stays `done`.
            busy = session.scalar(
                select(RunSegmentRangeAttempt.id).where(
                    RunSegmentRangeAttempt.run_id == run_id,
                    RunSegmentRangeAttempt.status.in_(RANGE_ATTEMPT_ACTIVE_STATUSES),
                ).limit(1)
            )
            if busy is not None:
                continue
            size, newest = _dir_stats(clips)
            # A clip cut recently means something is using this run again.
            if newest and newest >= cutoff.timestamp():
                continue
            try:
                shutil.rmtree(clips)
            except OSError as exc:
                log.warning("disk cleanup: could not remove %s: %s", clips, exc)
                continue
            count += 1
            freed += size
        finally:
            session.commit()  # release the row lock
    return count, freed


def _sweep_segment_takes(session, now: datetime) -> tuple[int, int]:
    if SEGMENT_TAKE_RETENTION_DAYS <= 0:
        return 0, 0
    cutoff = now - timedelta(days=SEGMENT_TAKE_RETENTION_DAYS)
    rows = session.execute(
        select(RunSegmentTake.id, RunSegmentTake.run_id).where(
            RunSegmentTake.file_purged_at.is_(None),
            RunSegmentTake.created_at < cutoff,
        )
    ).all()
    count = freed = 0
    for take_id, run_id in rows:
        try:
            # Same serialisation as the clips sweep: a restore takes this Run
            # row lock (api_v2._lock_run_for_result_mutation) before it reads
            # the take, so a locked row is skipped and an idle status re-read
            # under the lock means no restore is admitted or in flight.
            locked = session.scalar(
                select(Run.id)
                .where(Run.id == run_id, Run.status.in_(_RUN_IDLE))
                .with_for_update(skip_locked=True)
            )
            if locked is None:
                continue
            take = session.get(RunSegmentTake, take_id)
            if take is None or take.file_purged_at is not None:
                continue
            in_use = session.scalar(
                select(RunSegment.id).where(
                    RunSegment.run_id == run_id,
                    RunSegment.local_result_path == take.result_path,
                ).limit(1)
            )
            if in_use is not None:
                continue  # the current result of a segment (it was restored)
            size = 0
            try:
                size = os.path.getsize(take.result_path)
                os.remove(take.result_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("disk cleanup: could not remove %s: %s", take.result_path, exc)
                continue
            refs = os.path.join(
                os.path.dirname(take.result_path), f"{take.id}_refs"
            )
            if os.path.isdir(refs):
                size += _dir_stats(refs)[0]
                shutil.rmtree(refs, ignore_errors=True)
            take.file_purged_at = now
            count += 1
            freed += size
        finally:
            session.commit()  # persist the marker, release the row lock
    return count, freed


def sweep(session, now: Optional[datetime] = None) -> SweepResult:
    """Delete regenerable intermediates and expired segment takes (see module
    docstring). The only DB write is a purged take's ``file_purged_at``;
    commits *session* after each candidate to release its row lock."""
    now = now or datetime.now(timezone.utc)
    bodies, bodies_freed = _sweep_tail_bodies(session, now)
    clip_dirs, clips_freed = _sweep_run_clips(session, now)
    takes, takes_freed = _sweep_segment_takes(session, now)
    result = SweepResult(
        bodies, clip_dirs, bodies_freed + clips_freed + takes_freed, takes,
    )
    log.info(
        "disk cleanup: removed %d tail-edit body.mp4 + %d run clips/ dirs + "
        "%d segment takes, freed %.1f MB",
        result.tail_bodies, result.run_clip_dirs, result.segment_takes,
        result.freed_bytes / 1e6,
    )
    return result
