"""Tests for app/disk_cleanup.py — the periodic sweep of regenerable media.

Own SQLite DB; storage._BASE is pointed at a per-test tmp dir so paths built by
run_dir / tail_edit_body_path land there.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_disk_cleanup_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import app.disk_cleanup as disk_cleanup  # noqa: E402
import app.storage as storage  # noqa: E402
from app.db import Base  # noqa: E402
from app.models import (  # noqa: E402
    Run, RunSegment, RunSegmentRangeAttempt, RunTailEdit, VideoProject,
)
from app.state_machine import (  # noqa: E402
    ProjectStatus, RangeAttemptStatus, RunStatus, SegmentStatus, TailEditStatus,
)

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=45)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture()
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_BASE", str(tmp_path / "data"))
    monkeypatch.setattr(disk_cleanup, "RUN_CLIPS_RETENTION_DAYS", 30)
    monkeypatch.setattr(disk_cleanup, "SEGMENT_TAKE_RETENTION_DAYS", 30)
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    yield sess
    sess.close()
    engine.dispose()


def _write(path: str, size: int = 1000, mtime: datetime | None = None) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(path, (ts, ts))
    return path


def _run(session, *, status=RunStatus.done, updated_at=OLD) -> Run:
    project = VideoProject(
        source_type="upload", source_ref="src.mp4", status=ProjectStatus.ready
    )
    session.add(project)
    session.flush()
    run = Run(project_id=project.id, prompt="p", status=status)
    session.add(run)
    session.flush()
    run.updated_at = updated_at
    session.commit()
    return run


def _clips(run: Run, mtime: datetime = OLD) -> str:
    d = os.path.join(storage.run_dir(run.id, run.project_id), "clips")
    _write(os.path.join(d, "clip_0000.mp4"), mtime=mtime)
    _write(os.path.join(d, "clip_0001_0_5000.mp4"), mtime=mtime)
    return d


def _result(run: Run) -> str:
    return _write(
        os.path.join(storage.run_dir(run.id, run.project_id), "results", "final.mp4")
    )


def _tail_edit(session, run: Run, *, status=TailEditStatus.done,
               updated_at=NOW - timedelta(hours=2)) -> tuple[RunTailEdit, str, str]:
    edit = RunTailEdit(run_id=run.id, cut_sec=5.0, status=status)
    session.add(edit)
    session.flush()
    edit.updated_at = updated_at
    session.commit()
    body = _write(storage.tail_edit_body_path(edit.id, run.project_id))
    out = _write(os.path.join(os.path.dirname(body), f"{run.id}_trim.mp4"))
    return edit, body, out


def test_old_done_run_loses_clips_but_keeps_results(session):
    run = _run(session)
    clips = _clips(run)
    final = _result(run)

    result = disk_cleanup.sweep(session, now=NOW)

    assert not os.path.exists(clips)
    assert os.path.exists(final)
    assert result.run_clip_dirs == 1
    assert result.freed_bytes == 2000


@pytest.mark.parametrize("status", [RunStatus.failed, RunStatus.processing,
                                    RunStatus.incomplete])
def test_non_done_run_clips_are_kept(session, status):
    run = _run(session, status=status)
    clips = _clips(run)

    disk_cleanup.sweep(session, now=NOW)

    assert os.path.isdir(clips)


def test_recent_done_run_clips_are_kept(session):
    run = _run(session, updated_at=NOW - timedelta(days=5))
    clips = _clips(run)

    disk_cleanup.sweep(session, now=NOW)

    assert os.path.isdir(clips)


def test_recently_cut_clip_keeps_the_dir(session):
    """An old run whose clips/ was written to lately is in use again."""
    run = _run(session)
    clips = _clips(run)
    _write(os.path.join(clips, "clip_0002_5000_9000.mp4"), mtime=NOW - timedelta(days=1))

    disk_cleanup.sweep(session, now=NOW)

    assert os.path.exists(os.path.join(clips, "clip_0000.mp4"))


def test_active_range_attempt_keeps_the_clips(session):
    run = _run(session)
    clips = _clips(run)
    rs = RunSegment(run_id=run.id, segment_def_id="sd-1", index=0,
                    status=SegmentStatus.completed)
    session.add(rs)
    session.flush()
    session.add(RunSegmentRangeAttempt(
        run_id=run.id, run_segment_id=rs.id,
        requested_start_sec=0.0, requested_end_sec=2.0,
        base_segment_snapshot=[], candidate_segment_snapshot=[],
        status=RangeAttemptStatus.generating,
    ))
    session.commit()

    disk_cleanup.sweep(session, now=NOW)

    assert os.path.isdir(clips)


def test_retention_zero_disables_the_clips_sweep(session, monkeypatch):
    monkeypatch.setattr(disk_cleanup, "RUN_CLIPS_RETENTION_DAYS", 0)
    run = _run(session)
    clips = _clips(run)

    disk_cleanup.sweep(session, now=NOW)

    assert os.path.isdir(clips)


@pytest.mark.parametrize("status", [TailEditStatus.done, TailEditStatus.failed])
def test_finished_tail_edit_body_is_removed_output_kept(session, status):
    run = _run(session, updated_at=NOW)
    _edit, body, out = _tail_edit(session, run, status=status)

    result = disk_cleanup.sweep(session, now=NOW)

    assert not os.path.exists(body)
    assert os.path.exists(out)
    assert result.tail_bodies == 1


@pytest.mark.parametrize("status", [TailEditStatus.queued, TailEditStatus.processing,
                                    TailEditStatus.delivering])
def test_active_tail_edit_body_is_kept(session, status):
    run = _run(session, updated_at=NOW)
    _edit, body, _out = _tail_edit(session, run, status=status)

    disk_cleanup.sweep(session, now=NOW)

    assert os.path.exists(body)


def test_just_finished_tail_edit_body_is_kept(session):
    run = _run(session, updated_at=NOW)
    _edit, body, _out = _tail_edit(session, run, updated_at=NOW - timedelta(minutes=10))

    disk_cleanup.sweep(session, now=NOW)

    assert os.path.exists(body)


def test_missing_files_are_not_an_error(session):
    run = _run(session)
    edit, body, _out = _tail_edit(session, run)
    os.remove(body)

    result = disk_cleanup.sweep(session, now=NOW)

    assert result == disk_cleanup.SweepResult(0, 0, 0)


# ---------------------------------------------------------------- segment takes (AICP-342)


def _take(session, run: Run, *, created_at=OLD, current=False):
    from app.models import RunSegmentTake

    rs = RunSegment(run_id=run.id, segment_def_id="sd-0", index=0,
                    status=SegmentStatus.completed)
    session.add(rs)
    session.flush()
    d = storage.segment_take_dir(rs.id, run.id, run.project_id)
    take = RunSegmentTake(run_id=run.id, run_segment_id=rs.id, segment_def_id="sd-0",
                          index=0, result_path=os.path.join(d, "t.mp4"))
    session.add(take)
    session.flush()
    _write(take.result_path)
    refs = os.path.join(d, f"{take.id}_refs")
    _write(os.path.join(refs, "ref.jpg"), size=10)
    take.created_at = created_at
    if current:
        rs.local_result_path = take.result_path
    session.commit()
    return take, refs


def test_an_expired_take_is_purged_and_its_row_kept(session):
    run = _run(session)
    take, refs = _take(session, run)

    result = disk_cleanup.sweep(session, now=NOW)

    assert result.segment_takes == 1
    assert not os.path.exists(take.result_path) and not os.path.exists(refs)
    session.expire_all()
    assert take.file_purged_at is not None
    # A second pass does nothing.
    assert disk_cleanup.sweep(session, now=NOW).segment_takes == 0


def test_a_recent_take_is_kept(session):
    run = _run(session)
    take, _ = _take(session, run, created_at=NOW - timedelta(days=5))
    assert disk_cleanup.sweep(session, now=NOW).segment_takes == 0
    assert os.path.exists(take.result_path)


def test_a_take_that_is_the_current_result_is_kept(session):
    run = _run(session)
    take, _ = _take(session, run, current=True)
    assert disk_cleanup.sweep(session, now=NOW).segment_takes == 0
    assert os.path.exists(take.result_path)
    session.expire_all()
    assert take.file_purged_at is None


@pytest.mark.parametrize("status", [RunStatus.queued, RunStatus.processing,
                                    RunStatus.stitching, RunStatus.delivering])
def test_takes_of_a_busy_run_are_kept(session, status):
    run = _run(session, status=status)
    take, _ = _take(session, run)
    assert disk_cleanup.sweep(session, now=NOW).segment_takes == 0
    assert os.path.exists(take.result_path)


def test_take_retention_zero_disables_the_sweep(session, monkeypatch):
    monkeypatch.setattr(disk_cleanup, "SEGMENT_TAKE_RETENTION_DAYS", 0)
    run = _run(session)
    take, _ = _take(session, run)
    assert disk_cleanup.sweep(session, now=NOW).segment_takes == 0
    assert os.path.exists(take.result_path)
