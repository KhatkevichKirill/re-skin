"""
Tests for app/pipeline_range_attempt.py — run-local segment range
regeneration: boundary validation, candidate creation, and the full
submit -> poll -> stitch -> deliver -> promote pipeline.

Strategy
--------
* Pure boundary-math tests (compute_candidate_snapshot / validate_model_
  duration) need no DB, ffmpeg or fakes at all.
* DB-level tests (create_range_attempt) use a real isolated SQLite engine,
  mirroring test_pipeline_v2.py's pattern.
* Pipeline tests reuse test_pipeline_v2.FakeKieClient / FakeGDriveClient
  against a real synthetic video (own copy — this file does not share a DB
  engine or import fixtures from test_pipeline_v2.py, only its two plain
  Fake* classes, to keep this file's DB isolated).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_range_attempt_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Run, RunSegment, RunSegmentRangeAttempt, SegmentDef, VideoProject
from app.state_machine import (
    RangeAttemptStatus,
    RunStatus,
    SegmentStatus,
    transition,
)
from app.pipeline_range_attempt import (
    RangeValidationError,
    compute_candidate_snapshot,
    create_range_attempt,
    poll_range_attempt_once,
    submit_range_attempt,
    validate_model_duration,
    validate_stitch_plan_coherence,
)
import app.api_v2 as api_v2_mod
import app.pipeline_range_attempt as pra_mod
import app.pipeline_v2 as pipeline_v2_mod
from app import media as media_mod
from app.storage import keep_clip_path, run_clips_dir
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from tests.test_pipeline_v2 import FakeGDriveClient, FakeKieClient

_VIDEO_DURATION = 25.0


# ---------------------------------------------------------------------------
# Pure boundary-math tests — no DB / ffmpeg.
# ---------------------------------------------------------------------------

_BASE = [
    {"id": "keep0", "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 5.0},
    {"id": "swap1", "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 10.0},
    {"id": "keep2", "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
]

_SWAP_SWAP = [
    {"id": "swapA", "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 5.0},
    {"id": "swapB", "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 10.0},
]


def test_enlarge_swap_left_into_preceding_keep():
    c, prev_id, next_id = compute_candidate_snapshot(_BASE, "swap1", 4.6, 10.0, 20.0)
    assert c[0]["end_sec"] == pytest.approx(4.6)
    assert c[1]["start_sec"] == pytest.approx(4.6)
    assert prev_id == "keep0" and next_id is None


def test_shrink_swap_from_left_expands_preceding_keep():
    c, prev_id, next_id = compute_candidate_snapshot(_BASE, "swap1", 6.0, 10.0, 20.0)
    assert c[0]["end_sec"] == pytest.approx(6.0)
    assert c[1]["start_sec"] == pytest.approx(6.0)
    assert prev_id == "keep0"


def test_enlarge_swap_right_into_following_keep():
    c, prev_id, next_id = compute_candidate_snapshot(_BASE, "swap1", 5.0, 10.4, 20.0)
    assert c[1]["end_sec"] == pytest.approx(10.4)
    assert c[2]["start_sec"] == pytest.approx(10.4)
    assert next_id == "keep2" and prev_id is None


def test_shrink_swap_from_right_expands_following_keep():
    c, prev_id, next_id = compute_candidate_snapshot(_BASE, "swap1", 5.0, 9.0, 20.0)
    assert c[1]["end_sec"] == pytest.approx(9.0)
    assert c[2]["start_sec"] == pytest.approx(9.0)
    assert next_id == "keep2"


def test_move_both_sides_when_both_neighbors_are_keep():
    c, prev_id, next_id = compute_candidate_snapshot(_BASE, "swap1", 4.0, 11.0, 20.0)
    assert c[0]["end_sec"] == pytest.approx(4.0)
    assert c[2]["start_sec"] == pytest.approx(11.0)
    assert prev_id == "keep0" and next_id == "keep2"


def test_locked_left_side_beside_a_swap():
    with pytest.raises(RangeValidationError, match="start boundary"):
        compute_candidate_snapshot(_SWAP_SWAP, "swapB", 4.0, 10.0, 20.0)


def test_locked_right_side_beside_a_swap():
    with pytest.raises(RangeValidationError, match="end boundary"):
        compute_candidate_snapshot(_SWAP_SWAP, "swapA", 0.0, 6.0, 20.0)


def test_explicit_rejection_of_swap_to_swap_merge_both_sides():
    # swapB has no keep on either side once swapA is also a swap.
    with pytest.raises(RangeValidationError):
        compute_candidate_snapshot(_SWAP_SWAP, "swapA", -1.0, 6.0, 20.0)


def test_start_outside_source_duration():
    with pytest.raises(RangeValidationError, match="cannot be before 0"):
        compute_candidate_snapshot(_BASE, "swap1", -1.0, 10.0, 20.0)


def test_end_outside_source_duration():
    with pytest.raises(RangeValidationError, match="source's duration"):
        compute_candidate_snapshot(_BASE, "swap1", 5.0, 25.0, 20.0)


@pytest.mark.parametrize("start,end", [(8.0, 5.0), (5.0, 5.0)])
def test_zero_or_reversed_range_rejected(start, end):
    with pytest.raises(RangeValidationError):
        compute_candidate_snapshot(_BASE, "swap1", start, end, 20.0)


def test_non_finite_range_rejected():
    with pytest.raises(RangeValidationError, match="finite"):
        compute_candidate_snapshot(_BASE, "swap1", float("nan"), 10.0, 20.0)
    with pytest.raises(RangeValidationError, match="finite"):
        compute_candidate_snapshot(_BASE, "swap1", 5.0, float("inf"), 20.0)


def test_model_min_duration_violation():
    with pytest.raises(RangeValidationError, match="minimum"):
        validate_model_duration("seedance", 5.0, 5.5)


def test_model_max_duration_violation():
    with pytest.raises(RangeValidationError, match="maximum"):
        validate_model_duration("seedance", 0.0, 20.0)


def test_model_duration_within_bounds_does_not_raise():
    validate_model_duration("seedance", 0.0, 8.0)


def test_keep_consumed_completely_is_rejected():
    with pytest.raises(RangeValidationError, match="zero or negative"):
        compute_candidate_snapshot(_BASE, "swap1", 0.0, 10.0, 20.0)


def test_partition_stays_exactly_contiguous_with_unchanged_total_coverage():
    c, _, _ = compute_candidate_snapshot(_BASE, "swap1", 4.0, 11.0, 20.0)
    total = sum(e["end_sec"] - e["start_sec"] for e in c)
    assert total == pytest.approx(20.0)
    assert c[0]["start_sec"] == pytest.approx(0.0)
    assert c[-1]["end_sec"] == pytest.approx(20.0)
    for i in range(1, len(c)):
        assert c[i]["start_sec"] == pytest.approx(c[i - 1]["end_sec"])


def test_no_op_request_is_rejected():
    with pytest.raises(RangeValidationError, match="same as"):
        compute_candidate_snapshot(_BASE, "swap1", 5.0, 10.0, 20.0)


def test_unknown_segment_id_rejected():
    with pytest.raises(RangeValidationError, match="not part of"):
        compute_candidate_snapshot(_BASE, "does-not-exist", 4.0, 10.0, 20.0)


def test_target_must_be_a_swap_entry():
    with pytest.raises(RangeValidationError, match="only a swap segment"):
        compute_candidate_snapshot(_BASE, "keep0", 0.0, 4.0, 20.0)


def test_stitch_plan_coherence_default_and_empty_pass():
    validate_stitch_plan_coherence(None)
    validate_stitch_plan_coherence([])


def test_stitch_plan_coherence_rejects_unknown_kind():
    with pytest.raises(RangeValidationError):
        validate_stitch_plan_coherence([{"kind": "bogus"}])


def test_stitch_plan_coherence_accepts_segment_and_library_entries():
    validate_stitch_plan_coherence([
        {"kind": "segment", "segment_def_id": "x"},
        {"kind": "library", "clip_id": "y"},
    ])


# ---------------------------------------------------------------------------
# DB-level tests — create_range_attempt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_engine():
    db_path = os.path.join(_TMP_DIR, "range_attempt_test.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.rollback()
    sess.close()


@pytest.fixture(autouse=True)
def patch_get_session(db_engine, monkeypatch):
    """Point pipeline_range_attempt.get_session at this file's own engine."""
    _TestSession = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

    @contextmanager
    def _test_get_session():
        sess = _TestSession()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    import app.pipeline_range_attempt as pra_mod
    monkeypatch.setattr(pra_mod, "get_session", _test_get_session)


def _make_project_run_segment(db_session, *, snapshot=True, duration=20.0, model="seedance", stitch_plan=None):
    """A minimal Project + 3-segment layout (keep|swap|keep) + Run + RunSegment."""
    project = VideoProject(
        source_type="upload", source_ref="src.mp4",
        source_local_path="/nonexistent/src.mp4",
        duration_sec=duration, width=320, height=240, fps=25.0,
    )
    db_session.add(project)
    db_session.flush()

    sd_keep0 = SegmentDef(project_id=project.id, index=0, action="keep", start_sec=0.0, end_sec=5.0)
    sd_swap1 = SegmentDef(project_id=project.id, index=1, action="swap", start_sec=5.0, end_sec=10.0)
    sd_keep2 = SegmentDef(project_id=project.id, index=2, action="keep", start_sec=10.0, end_sec=20.0)
    db_session.add_all([sd_keep0, sd_swap1, sd_keep2])
    db_session.flush()

    run = Run(
        project_id=project.id, prompt="p", model=model, resolution="480p",
        audio_mode="original", status=RunStatus.done, stitch_plan=stitch_plan,
    )
    if snapshot:
        run.segment_snapshot = [
            {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 5.0},
            {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 10.0},
            {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
        ]
    db_session.add(run)
    db_session.flush()

    rs = RunSegment(
        run_id=run.id, segment_def_id=sd_swap1.id, index=1,
        status=SegmentStatus.completed, local_result_path="/nonexistent/result_0001.mp4",
    )
    db_session.add(rs)
    db_session.flush()
    return project, run, rs


def test_create_range_attempt_builds_row(db_session):
    project, run, rs = _make_project_run_segment(db_session)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=4.0, end_sec=10.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    assert attempt.status == RangeAttemptStatus.queued
    assert attempt.requested_start_sec == 4.0
    assert attempt.requested_end_sec == 10.0
    assert attempt.candidate_segment_snapshot[0]["end_sec"] == pytest.approx(4.0)
    assert attempt.keep_recut_paths == {attempt.candidate_segment_snapshot[0]["id"]: None}


def test_create_range_attempt_rejects_legacy_snapshotless_run(db_session):
    project, run, rs = _make_project_run_segment(db_session, snapshot=False)
    with pytest.raises(RangeValidationError, match="predates run-local"):
        create_range_attempt(
            run=run, rs=rs, project=project,
            start_sec=4.0, end_sec=10.0,
            prompt_override=None, reference_image_urls_override=None,
            attempt_id=str(uuid.uuid4()), db=db_session,
        )


def test_create_range_attempt_enforces_model_duration(db_session):
    project, run, rs = _make_project_run_segment(db_session)
    # [1.0, 17.0) is a valid contiguous partition (keep0 -> 1s, keep2 -> 3s)
    # but a 16s swap exceeds Seedance 2.0's 15s max — this must be rejected
    # by validate_model_duration, not by the boundary/partition check.
    with pytest.raises(RangeValidationError, match="maximum"):
        create_range_attempt(
            run=run, rs=rs, project=project,
            start_sec=1.0, end_sec=17.0,
            prompt_override=None, reference_image_urls_override=None,
            attempt_id=str(uuid.uuid4()), db=db_session,
        )


# ---------------------------------------------------------------------------
# Full pipeline tests — synthetic video, FakeKieClient/FakeGDriveClient.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_video():
    path = os.path.join(_TMP_DIR, "source_range_attempt.mp4")
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={int(_VIDEO_DURATION)}:r=25",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={int(_VIDEO_DURATION)}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            "-c:a", "aac", path,
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return path


@pytest.fixture(autouse=True)
def _cleanup_module(request):
    yield
    if request.node.get_closest_marker is None:
        return


def _make_pipeline_run(db_session, synthetic_video, *, stitch_plan=None):
    project = VideoProject(
        source_type="upload", source_ref="src.mp4", source_local_path=synthetic_video,
        duration_sec=_VIDEO_DURATION, width=320, height=240, fps=25.0,
        aspect_ratio="4:3",
    )
    db_session.add(project)
    db_session.flush()

    sd_keep0 = SegmentDef(project_id=project.id, index=0, action="keep", start_sec=0.0, end_sec=9.0)
    sd_swap1 = SegmentDef(project_id=project.id, index=1, action="swap", start_sec=9.0, end_sec=15.0)
    sd_keep2 = SegmentDef(project_id=project.id, index=2, action="keep", start_sec=15.0, end_sec=25.0)
    db_session.add_all([sd_keep0, sd_swap1, sd_keep2])
    db_session.flush()

    run = Run(
        project_id=project.id, prompt="a person", model="seedance", resolution="480p",
        audio_mode="original", status=RunStatus.done, stitch_plan=stitch_plan,
        segment_snapshot=[
            {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 9.0},
            {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 9.0, "end_sec": 15.0},
            {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 15.0, "end_sec": 25.0},
        ],
    )
    db_session.add(run)
    db_session.flush()

    # An UNAFFECTED swap RunSegment already completed on the ORIGINAL bounds
    # — used to prove it is reused, never resubmitted, by a candidate
    # targeting a DIFFERENT segment. (Not used directly in single-segment
    # tests below, but kept minimal here; see the dedicated reuse test.)
    rs = RunSegment(
        run_id=run.id, segment_def_id=sd_swap1.id, index=1,
        status=SegmentStatus.completed,
        local_clip_path=os.path.join(_TMP_DIR, "orig_clip_0001.mp4"),
        local_result_path=os.path.join(_TMP_DIR, "orig_result_0001.mp4"),
        seedance_task_id="orig-task-1", cost_credits=10.0,
    )
    db_session.add(rs)
    db_session.flush()
    # Give it a real (tiny) file on disk so stitch-time checks pass.
    shutil.copy2(synthetic_video, rs.local_result_path)

    run.result_local_path = os.path.join(_TMP_DIR, f"orig_final_{run.id}.mp4")
    shutil.copy2(synthetic_video, run.result_local_path)
    db_session.commit()
    return project, run, sd_keep0, sd_swap1, sd_keep2, rs


def test_submit_cuts_candidate_from_original_source_at_exact_bounds(db_session, synthetic_video):
    from app import media as media_mod

    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    task_id = submit_range_attempt(attempt.id, kie=kie)
    assert task_id is not None
    assert kie.upload_calls, "expected the candidate clip to be uploaded"

    db_session.expire_all()
    reloaded = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded.local_clip_path is not None
    dur = media_mod.probe(reloaded.local_clip_path).duration_sec
    assert dur == pytest.approx(7.0, abs=0.3)  # [8, 15) == 7s
    # The candidate clip must NOT reuse (or overwrite) the run's own
    # index-based clip cache path.
    assert "range_attempts" in reloaded.local_clip_path
    assert reloaded.status == RangeAttemptStatus.generating


def test_submit_is_idempotent_across_resume(db_session, synthetic_video):
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    task_id_1 = submit_range_attempt(attempt.id, kie=kie)
    calls_after_first = len(kie.create_task_calls)
    # Simulate a worker restart: call submit again with a FRESH kie client —
    # it must reuse the persisted task id, not create a new one.
    kie2 = FakeKieClient(synthetic_video)
    task_id_2 = submit_range_attempt(attempt.id, kie=kie2)
    assert task_id_1 == task_id_2
    assert not kie2.create_task_calls, "resume must not create a second task"
    assert len(kie.create_task_calls) == calls_after_first


def test_failed_generation_leaves_run_and_segment_untouched(db_session, synthetic_video):
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    orig_snapshot = run.segment_snapshot
    orig_result_path = run.result_local_path
    orig_rs_result_path = rs.local_result_path

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video, fail_task_id=None)
    task_id = submit_range_attempt(attempt.id, kie=kie)

    class FailingKie(FakeKieClient):
        def get_task(self, task_id, **kw):
            return {"state": "fail", "failMsg": "injected failure"}

    failing_kie = FailingKie(synthetic_video)
    result = poll_range_attempt_once(attempt.id, kie=failing_kie)
    assert result == "failed"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs = db_session.get(RunSegment, rs.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.status == RangeAttemptStatus.failed
    assert reloaded_attempt.promoted_at is None
    assert reloaded_run.segment_snapshot == orig_snapshot
    assert reloaded_run.result_local_path == orig_result_path
    assert reloaded_run.status == RunStatus.done
    assert reloaded_rs.local_result_path == orig_rs_result_path


def test_successful_candidate_promotes_coherent_timeline_and_final(db_session, synthetic_video, monkeypatch):
    import app.pipeline_range_attempt as pra_mod

    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    old_snapshot = list(run.segment_snapshot)
    old_result_path = run.result_local_path

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    fake_gdrive = FakeGDriveClient()
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=fake_gdrive)
    assert result == "done"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs = db_session.get(RunSegment, rs.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)

    assert reloaded_attempt.status == RangeAttemptStatus.done
    assert reloaded_attempt.promoted_at is not None
    assert reloaded_run.status == RunStatus.done

    # Timeline promoted: swap now starts at 8.0, preceding keep now ends at 8.0.
    by_id = {e["id"]: e for e in reloaded_run.segment_snapshot}
    assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(8.0)
    assert by_id[sd_keep0.id]["end_sec"] == pytest.approx(8.0)
    # Still contiguous over the whole source.
    ordered = sorted(reloaded_run.segment_snapshot, key=lambda e: e["index"])
    assert ordered[0]["start_sec"] == pytest.approx(0.0)
    assert ordered[-1]["end_sec"] == pytest.approx(_VIDEO_DURATION)
    for i in range(1, len(ordered)):
        assert ordered[i]["start_sec"] == pytest.approx(ordered[i - 1]["end_sec"])

    assert reloaded_rs.local_result_path != None
    assert reloaded_rs.status == SegmentStatus.completed
    assert reloaded_run.result_local_path != old_result_path
    assert os.path.exists(reloaded_run.result_local_path)
    assert fake_gdrive.upload_calls, "candidate final must be delivered to Drive"


def test_adjacent_keep_is_recut_from_original_source_at_new_bounds(db_session, synthetic_video):
    """Moving swap1's start from 9.0 -> 8.0 shrinks keep0 from [0,9] to [0,8].
    The re-cut keep clip must (a) live under this attempt's own directory, not
    the run's shared clip cache, and (b) actually measure 8.0s — proving it was
    cut fresh from VideoProject.source_local_path at the NEW bound, not the old
    9.0s cache reused."""
    from app import media as media_mod

    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    fake_gdrive = FakeGDriveClient()
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=fake_gdrive)
    assert result == "done"

    db_session.expire_all()
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    recut_path = reloaded_attempt.keep_recut_paths[sd_keep0.id]
    assert recut_path is not None
    assert "range_attempts" in recut_path, "must not overwrite the run's shared clip cache"
    assert os.path.exists(recut_path)
    assert media_mod.probe(recut_path).duration_sec == pytest.approx(8.0, abs=0.3)


def _make_pipeline_run_two_swaps(db_session, synthetic_video):
    """keep0[0,5] | swap1[5,10]=target | keep2[10,15] | swap3[15,20]=UNAFFECTED
    (already completed) | keep4[20,25]. Used to prove a range regeneration on
    swap1 never re-touches swap3's already-completed result."""
    project = VideoProject(
        source_type="upload", source_ref="src.mp4", source_local_path=synthetic_video,
        duration_sec=_VIDEO_DURATION, width=320, height=240, fps=25.0,
        aspect_ratio="4:3",
    )
    db_session.add(project)
    db_session.flush()

    sd_keep0 = SegmentDef(project_id=project.id, index=0, action="keep", start_sec=0.0, end_sec=5.0)
    sd_swap1 = SegmentDef(project_id=project.id, index=1, action="swap", start_sec=5.0, end_sec=10.0)
    sd_keep2 = SegmentDef(project_id=project.id, index=2, action="keep", start_sec=10.0, end_sec=15.0)
    sd_swap3 = SegmentDef(project_id=project.id, index=3, action="swap", start_sec=15.0, end_sec=20.0)
    sd_keep4 = SegmentDef(project_id=project.id, index=4, action="keep", start_sec=20.0, end_sec=25.0)
    db_session.add_all([sd_keep0, sd_swap1, sd_keep2, sd_swap3, sd_keep4])
    db_session.flush()

    snapshot = [
        {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 5.0},
        {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 10.0},
        {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 15.0},
        {"id": sd_swap3.id, "index": 3, "action": "swap", "start_sec": 15.0, "end_sec": 20.0},
        {"id": sd_keep4.id, "index": 4, "action": "keep", "start_sec": 20.0, "end_sec": 25.0},
    ]
    run = Run(
        project_id=project.id, prompt="a person", model="seedance", resolution="480p",
        audio_mode="original", status=RunStatus.done, segment_snapshot=snapshot,
    )
    db_session.add(run)
    db_session.flush()

    rs1 = RunSegment(
        run_id=run.id, segment_def_id=sd_swap1.id, index=1,
        status=SegmentStatus.completed,
        local_clip_path=os.path.join(_TMP_DIR, f"{run.id}_clip1.mp4"),
        local_result_path=os.path.join(_TMP_DIR, f"{run.id}_result1.mp4"),
        seedance_task_id="orig-task-1", cost_credits=10.0,
    )
    rs3 = RunSegment(
        run_id=run.id, segment_def_id=sd_swap3.id, index=3,
        status=SegmentStatus.completed,
        local_clip_path=os.path.join(_TMP_DIR, f"{run.id}_clip3.mp4"),
        local_result_path=os.path.join(_TMP_DIR, f"{run.id}_result3.mp4"),
        seedance_task_id="orig-task-3", cost_credits=10.0,
    )
    db_session.add_all([rs1, rs3])
    db_session.flush()
    shutil.copy2(synthetic_video, rs1.local_result_path)
    shutil.copy2(synthetic_video, rs3.local_result_path)

    run.result_local_path = os.path.join(_TMP_DIR, f"orig_final_{run.id}.mp4")
    shutil.copy2(synthetic_video, run.result_local_path)
    db_session.commit()
    return project, run, dict(
        keep0=sd_keep0, swap1=sd_swap1, keep2=sd_keep2, swap3=sd_swap3, keep4=sd_keep4,
    ), rs1, rs3


def test_unaffected_completed_swap_segment_is_reused_not_resubmitted(db_session, synthetic_video):
    """A range regeneration targeting swap1 must never re-touch swap3 — its
    RunSegment fields stay byte-identical and the fake kie client only ever
    sees ONE create_task call (swap1's candidate), proving swap3 is neither
    resubmitted nor re-billed."""
    project, run, sds, rs1, rs3 = _make_pipeline_run_two_swaps(db_session, synthetic_video)
    rs3_result_path_before = rs3.local_result_path
    rs3_cost_before = rs3.cost_credits
    rs3_task_before = rs3.seedance_task_id

    attempt = create_range_attempt(
        run=run, rs=rs1, project=project,
        start_sec=4.0, end_sec=10.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    assert len(kie.create_task_calls) == 1, "only the targeted segment may be submitted"

    fake_gdrive = FakeGDriveClient()
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=fake_gdrive)
    assert result == "done"
    # Still exactly one create_task call after promotion — swap3 was never
    # resubmitted during stitch.
    assert len(kie.create_task_calls) == 1

    db_session.expire_all()
    reloaded_rs3 = db_session.get(RunSegment, rs3.id)
    assert reloaded_rs3.local_result_path == rs3_result_path_before
    assert reloaded_rs3.cost_credits == rs3_cost_before
    assert reloaded_rs3.seedance_task_id == rs3_task_before


def test_active_artifacts_unchanged_while_candidate_generating(db_session, synthetic_video):
    """Before poll (candidate still 'generating'), the active Run/RunSegment
    result paths must be untouched — the invariant that a candidate cannot
    overwrite an active artifact before promotion holds structurally (distinct
    attempt-scoped paths), verified here by checking nothing changed mid-flight."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    orig_rs_result_path = rs.local_result_path
    orig_run_result_path = run.result_local_path
    orig_snapshot = run.segment_snapshot

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs = db_session.get(RunSegment, rs.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.status == RangeAttemptStatus.generating
    assert reloaded_rs.local_result_path == orig_rs_result_path
    assert reloaded_run.result_local_path == orig_run_result_path
    assert reloaded_run.segment_snapshot == orig_snapshot


def test_failed_stitch_leaves_previous_result_intact(db_session, synthetic_video):
    """Generation succeeds but the candidate stitch fails (an unaffected swap
    segment's result file is missing) — the attempt must fail and the Run/
    RunSegments must be left exactly as they were, per the same non-destructive
    contract as a failed generation."""
    project, run, sds, rs1, rs3 = _make_pipeline_run_two_swaps(db_session, synthetic_video)
    orig_snapshot = run.segment_snapshot
    orig_run_result_path = run.result_local_path
    orig_rs1_result_path = rs1.local_result_path

    # Break the UNAFFECTED segment's result so _stitch_candidate must raise.
    os.remove(rs3.local_result_path)

    attempt = create_range_attempt(
        run=run, rs=rs1, project=project,
        start_sec=4.0, end_sec=10.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "failed"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs1 = db_session.get(RunSegment, rs1.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.status == RangeAttemptStatus.failed
    assert reloaded_attempt.promoted_at is None
    assert reloaded_attempt.error_message and "not completed" in reloaded_attempt.error_message
    assert reloaded_run.segment_snapshot == orig_snapshot
    assert reloaded_run.result_local_path == orig_run_result_path
    assert reloaded_run.status == RunStatus.done
    assert reloaded_rs1.local_result_path == orig_rs1_result_path
    assert reloaded_rs1.status == SegmentStatus.completed


def test_failed_delivery_leaves_previous_result_intact(db_session, synthetic_video, monkeypatch):
    """Generation and stitch succeed but every Drive upload attempt fails —
    the attempt must fail (not silently drop the delivery) and the Run/
    RunSegment must keep the LAST successful result, exactly like a failed
    generation or stitch."""
    import app.pipeline_v2 as pv2_mod

    monkeypatch.setattr(pv2_mod.time, "sleep", lambda _s: None)  # skip retry backoff

    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    run.gdrive_folder_id = "fake-folder-id"
    db_session.commit()
    orig_snapshot = run.segment_snapshot
    orig_run_result_path = run.result_local_path
    orig_rs_result_path = rs.local_result_path

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    class FailingGDrive(FakeGDriveClient):
        def upload_file(self, *a, **kw):
            raise RuntimeError("injected Drive outage")

    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FailingGDrive())
    assert result == "failed"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs = db_session.get(RunSegment, rs.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.status == RangeAttemptStatus.failed
    assert reloaded_attempt.promoted_at is None
    assert "Drive outage" in (reloaded_attempt.error_message or "") or "delivery failed" in (
        reloaded_attempt.error_message or ""
    )
    # The candidate final WAS built (stitch succeeded) — but must not have
    # been promoted onto the active Run despite existing on disk.
    assert reloaded_attempt.final_local_path is not None
    assert reloaded_run.segment_snapshot == orig_snapshot
    assert reloaded_run.result_local_path == orig_run_result_path
    assert reloaded_rs.local_result_path == orig_rs_result_path


# ---------------------------------------------------------------------------
# AICP-349 — promotion delivers IN PLACE behind the run's existing Drive id.
# ---------------------------------------------------------------------------


def _start_delivered_attempt(db_session, synthetic_video, *, file_id="drive-orig-id"):
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    run.gdrive_folder_id = "fake-folder-id"
    run.result_gdrive_file_id = file_id
    db_session.commit()
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    return run, rs, attempt, kie


def _run_state(db_session, run_id, rs_id):
    db_session.expire_all()
    run = db_session.get(Run, run_id)
    rs = db_session.get(RunSegment, rs_id)
    return (
        run.status, run.result_gdrive_file_id, run.result_local_path,
        list(run.segment_snapshot), run.result_duration_sec,
        rs.local_result_path, rs.seedance_task_id, rs.status,
    )


def test_promotion_replaces_the_existing_drive_file_in_place(db_session, synthetic_video):
    """Acceptance: same Drive id (so the same Creative Link, Job ID and
    exported row), new content, no new Drive file."""
    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video)
    job_id_before = run.id

    gdrive = FakeGDriveClient()
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "done"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert gdrive.upload_calls == [], "a delivered run must never get a fresh upload"
    assert [c["file_id"] for c in gdrive.update_calls] == ["drive-orig-id"]
    assert reloaded_run.result_gdrive_file_id == "drive-orig-id"
    assert reloaded_run.id == job_id_before
    assert reloaded_attempt.gdrive_file_id == "drive-orig-id"
    assert reloaded_attempt.status == RangeAttemptStatus.done
    # The new content: the run now points at the candidate final, and the
    # bytes uploaded were that very file (a private hard link of it, gone now).
    assert reloaded_run.result_local_path == reloaded_attempt.final_local_path
    uploaded = gdrive.update_calls[0]["local_path"]
    assert os.path.basename(uploaded).startswith(".deliver-")
    assert os.path.dirname(uploaded) == os.path.dirname(reloaded_attempt.final_local_path)
    assert not os.path.exists(uploaded)
    assert os.path.exists(reloaded_attempt.final_local_path)


def test_failed_in_place_update_leaves_the_run_byte_for_byte_untouched(
    db_session, synthetic_video, monkeypatch,
):
    """A Drive update failure (incl. a deleted file's 404) fails the attempt
    with the run exactly as it was, and never falls back to a fresh upload."""
    monkeypatch.setattr(pipeline_v2_mod.time, "sleep", lambda _s: None)
    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video)
    before = _run_state(db_session, run.id, rs.id)

    # The class the module catches: test_gdrive_client reloads app.gdrive_client.
    GDriveFileNotFound = pra_mod.GDriveFileNotFound

    class DeletedFileGDrive(FakeGDriveClient):
        def file_md5(self, file_id):
            raise GDriveFileNotFound(file_id)

    gdrive = DeletedFileGDrive()
    gdrive.update_error = RuntimeError("HttpError 404 File not found")
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "failed"

    assert _run_state(db_session, run.id, rs.id) == before
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.status == RangeAttemptStatus.failed
    assert reloaded_attempt.promoted_at is None
    assert reloaded_attempt.gdrive_file_id is None
    assert "in-place Drive delivery" in reloaded_attempt.error_message
    assert "404" in reloaded_attempt.error_message
    assert "re-linked" in reloaded_attempt.error_message
    assert gdrive.upload_calls == []
    assert len(gdrive.update_calls) == pipeline_v2_mod.RUN_DELIVER_ATTEMPTS
    # No pin left behind next to the candidate final.
    final_dir = os.path.dirname(reloaded_attempt.final_local_path)
    assert not [n for n in os.listdir(final_dir) if n.startswith(".deliver-")]


def _md5(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def test_update_reported_failed_but_landed_is_promoted(db_session, synthetic_video, monkeypatch):
    """A lost response on a transfer that DID land: Drive's md5 equals the
    candidate's, so the attempt promotes instead of failing with the exported
    link already serving its bytes."""
    monkeypatch.setattr(pipeline_v2_mod.time, "sleep", lambda _s: None)
    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video)

    class LostResponseGDrive(FakeGDriveClient):
        def update_file_content(self, file_id, local_path, mime_type="video/mp4"):
            self.update_calls.append({"file_id": file_id, "local_path": local_path})
            self.landed = _md5(local_path)
            raise RuntimeError("connection reset after the last chunk")

        def file_md5(self, file_id):
            return getattr(self, "landed", None)

    gdrive = LostResponseGDrive()
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "done"
    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_run.result_gdrive_file_id == "drive-orig-id"
    assert reloaded_run.result_local_path == reloaded_attempt.final_local_path
    assert reloaded_attempt.error_message is None


def test_update_outcome_unknown_fails_loudly(db_session, synthetic_video, monkeypatch):
    """Drive unreachable for both the update and the md5 read: the run is
    not promoted and the message says the link MAY hold the candidate."""
    monkeypatch.setattr(pipeline_v2_mod.time, "sleep", lambda _s: None)
    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video)
    before = _run_state(db_session, run.id, rs.id)

    class DownGDrive(FakeGDriveClient):
        def file_md5(self, file_id):
            raise RuntimeError("Drive 503")

    gdrive = DownGDrive()
    gdrive.update_error = RuntimeError("Drive 503")
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "failed"
    assert _run_state(db_session, run.id, rs.id) == before
    msg = db_session.get(RunSegmentRangeAttempt, attempt.id).error_message
    assert "could not be verified" in msg and "MAY already hold" in msg


def test_commit_failure_after_landed_update_is_finalized_again(
    db_session, synthetic_video, monkeypatch,
):
    """Drive took the candidate but the promotion commit failed: the attempt
    is handed back (generating, claim released), not failed, and the next
    poll re-sends the same content to the same id and promotes."""
    from sqlalchemy import event as sa_event
    from sqlalchemy.orm import Session as SASession

    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video)
    before = _run_state(db_session, run.id, rs.id)

    state = {"armed": False, "fired": False}
    real_deliver = pra_mod._deliver_in_place_with_retry

    def _deliver_then_arm(*a, **kw):
        real_deliver(*a, **kw)
        state["armed"] = True

    def _fail_commit_once(session):
        if state["armed"] and not state["fired"]:
            state["fired"] = True
            raise RuntimeError("server closed the connection unexpectedly")

    monkeypatch.setattr(pra_mod, "_deliver_in_place_with_retry", _deliver_then_arm)
    sa_event.listen(SASession, "before_commit", _fail_commit_once)
    try:
        gdrive = FakeGDriveClient()
        assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "pending"
    finally:
        sa_event.remove(SASession, "before_commit", _fail_commit_once)

    assert state["fired"]
    assert _run_state(db_session, run.id, rs.id) == before
    db_session.expire_all()
    a = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert a.status == RangeAttemptStatus.generating
    assert a.claim_owner is None and a.promoted_at is None
    assert "did not commit" in a.error_message

    gdrive2 = FakeGDriveClient()
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive2) == "done"
    assert [c["file_id"] for c in gdrive.update_calls + gdrive2.update_calls] == [
        "drive-orig-id", "drive-orig-id",
    ]
    db_session.expire_all()
    a = db_session.get(RunSegmentRangeAttempt, attempt.id)
    reloaded_run = db_session.get(Run, run.id)
    assert a.status == RangeAttemptStatus.done and a.error_message is None
    assert reloaded_run.result_gdrive_file_id == "drive-orig-id"
    assert reloaded_run.result_local_path == a.final_local_path


def test_stale_timeline_refuses_before_touching_drive(db_session, synthetic_video):
    """A candidate that fails a staleness check under the Run lock must not
    have replaced the exported file's content first."""
    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video)
    # Another promotion moved the timeline meanwhile.
    moved = [dict(e) for e in run.segment_snapshot]
    moved[0]["end_sec"] = 8.5
    moved[1]["start_sec"] = 8.5
    run.segment_snapshot = moved
    db_session.commit()
    before = _run_state(db_session, run.id, rs.id)

    gdrive = FakeGDriveClient()
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "failed"
    assert gdrive.update_calls == [] and gdrive.upload_calls == []
    assert _run_state(db_session, run.id, rs.id) == before


def test_claim_taken_over_at_promotion_never_touches_drive(
    db_session, db_engine, synthetic_video, monkeypatch,
):
    """The claim is re-checked under the attempt row lock inside _promote
    (the range attempt's version of finalize's lock ownership): a takeover
    that lands just before promotion stops the stale owner before the Drive
    update. The plain-SELECT re-checks are disabled here so the in-lock
    check is the one proved."""
    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video)
    before = _run_state(db_session, run.id, rs.id)
    monkeypatch.setattr(pra_mod, "_assert_still_owner", lambda *a, **kw: None)

    real_promote = pra_mod._promote
    Other = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

    def _takeover_then_promote(session, run_, rs_, attempt_, **kw):
        other = Other()
        row = other.get(RunSegmentRangeAttempt, attempt_.id)
        row.claim_owner = "someone-else"
        other.commit()
        other.close()
        return real_promote(session, run_, rs_, attempt_, **kw)

    monkeypatch.setattr(pra_mod, "_promote", _takeover_then_promote)
    gdrive = FakeGDriveClient()
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "pending"
    assert gdrive.update_calls == [] and gdrive.upload_calls == []
    assert _run_state(db_session, run.id, rs.id) == before
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.promoted_at is None


def test_run_without_a_drive_file_still_gets_a_fresh_upload(db_session, synthetic_video):
    """Unchanged path: nothing on Drive yet, so there is no link to keep."""
    run, rs, attempt, kie = _start_delivered_attempt(db_session, synthetic_video, file_id=None)
    gdrive = FakeGDriveClient()
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=gdrive) == "done"
    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    assert len(gdrive.upload_calls) == 1 and gdrive.update_calls == []
    assert reloaded_run.result_gdrive_file_id
    assert reloaded_run.result_gdrive_file_id == db_session.get(
        RunSegmentRangeAttempt, attempt.id).gdrive_file_id


def test_integration_promoted_range_flows_through_copy_to_submit_cut_bounds(
    db_session, db_engine, synthetic_video, monkeypatch,
):
    """The one required end-to-end integration test: drive a REAL promotion,
    then a REAL _build_copied_run clone, then a REAL pipeline_v2.submit_run —
    and assert the downstream READ SITE that actually chooses media cut
    bounds (media_mod.cut_clip, called from _submit_swap_segment_isolated)
    cuts the COPY's swap segment at the PROMOTED bounds, not the ORIGINAL
    Project SegmentDef bounds and not the pre-promotion run bounds. A test
    that only compared JSON snapshots would miss a submit-phase bug like the
    split-source-of-truth one this feature's audit found and fixed in
    pipeline_v2._build_submit_state (see docs/run-local-segment-range-
    regeneration.md)."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    # 1. REAL promotion: 9.0 -> 8.0 start (enlarge swap1 into keep0).
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie_for_attempt = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie_for_attempt)
    result = poll_range_attempt_once(attempt.id, kie=kie_for_attempt, gdrive=FakeGDriveClient())
    assert result == "done"

    db_session.expire_all()
    promoted_run = db_session.get(Run, run.id)
    assert promoted_run.segment_snapshot is not None
    by_id = {e["id"]: e for e in promoted_run.segment_snapshot}
    assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(8.0)

    # Live Project SegmentDef is UNTOUCHED by promotion (still 9.0) — the
    # thing this test must prove submit does NOT fall back to.
    assert sd_swap1.start_sec == pytest.approx(9.0)

    # 2. REAL _build_copied_run — the exact helper Copy/Automation both call.
    new_run = api_v2_mod._build_copied_run(
        promoted_run, db_session,
        resolution=None, name=None, ref_files=[], ref_urls=[],
    )
    db_session.commit()
    assert new_run.segment_snapshot is not None
    new_by_id = {e["id"]: e for e in new_run.segment_snapshot}
    assert new_by_id[sd_swap1.id]["start_sec"] == pytest.approx(8.0)

    # 3. REAL submit — pipeline_v2.submit_run, using the SAME test DB engine.
    _Pv2Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

    @contextmanager
    def _pv2_test_get_session():
        sess = _Pv2Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    monkeypatch.setattr(pipeline_v2_mod, "get_session", _pv2_test_get_session)

    cut_calls = []
    real_cut_clip = media_mod.cut_clip

    def _spy_cut_clip(source, start, end, dst, **kw):
        cut_calls.append((source, start, end))
        return real_cut_clip(source, start, end, dst, **kw)

    monkeypatch.setattr(pipeline_v2_mod.media_mod, "cut_clip", _spy_cut_clip)

    kie_for_submit = FakeKieClient(synthetic_video)
    pending = pipeline_v2_mod.submit_run(new_run.id, kie=kie_for_submit, gdrive=FakeGDriveClient())
    assert pending, "submit_run must have submitted the copy's swap segment"

    swap_cuts = [c for c in cut_calls if c[1] == pytest.approx(8.0, abs=0.01)]
    assert swap_cuts, (
        f"expected a cut_clip call at the PROMOTED start 8.0s; got {cut_calls}"
    )
    source_cut, start_cut, end_cut = swap_cuts[0]
    assert source_cut == synthetic_video, "must cut from the ORIGINAL project source, not a derived file"
    assert start_cut == pytest.approx(8.0)
    assert end_cut == pytest.approx(15.0)
    # And explicitly NOT the live (unpromoted) Project SegmentDef bounds.
    assert not any(c[1] == pytest.approx(9.0, abs=0.01) for c in cut_calls), (
        "submit must not have fallen back to live project.segments bounds"
    )


def test_duplicate_poll_of_the_same_attempt_never_clobbers_the_others_committed_done(
    db_session, db_engine, synthetic_video, monkeypatch,
):
    """Regression for the finding the mandatory Cursor review gate raised on
    the first hardening pass: session.refresh(run, with_for_update=True) in
    _promote only protects the FINAL commit. A second session that already
    holds an in-memory copy of the attempt from BEFORE a first session ran
    it to completion (a duplicate poll enqueue, or a recovery re-enqueue
    racing a still-running job) could still silently overwrite an
    already-committed `done` via one of the EARLIER, previously-unlocked
    intermediate commits (the stitching transition, keep_recut_paths, the
    delivering transition, gdrive_file_id) — none of which were protected
    before RunSegmentRangeAttempt.version_id existed. Reproduces the exact
    shape: worker B loads the attempt (into its OWN session's identity map)
    BEFORE worker A runs poll_range_attempt_once to completion; B's own
    later call must detect the version conflict and report A's real
    outcome, never re-process or fail an attempt that already succeeded."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    attempt_id = attempt.id

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt_id, kie=kie)

    # Worker B "arrives" first and loads the attempt at its post-submit,
    # pre-poll state — its session's identity map now holds this stale copy.
    SessionB = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session_b = SessionB()
    stale_attempt = session_b.get(RunSegmentRangeAttempt, attempt_id)
    assert stale_attempt.status == RangeAttemptStatus.generating

    # Worker A runs the REAL function to completion through a normal fresh
    # session — reaches `done` and promotes.
    assert poll_range_attempt_once(attempt_id, kie=kie, gdrive=FakeGDriveClient()) == "done"

    db_session.expire_all()
    winner_snapshot = db_session.get(Run, run.id).segment_snapshot
    assert winner_snapshot is not None

    # Worker B's own (now provably stale) call: poll_range_attempt_once's
    # session.get() checks the identity map before the DB, so handing it
    # session_b (which already cached the pre-run object) makes it see
    # `generating`, not `done` — it tries to process the SAME attempt again
    # from scratch, using the SAME already-successful kie task id.
    @contextmanager
    def _b_session_factory():
        yield session_b

    monkeypatch.setattr(pra_mod, "get_session", _b_session_factory)
    result_b = poll_range_attempt_once(attempt_id, kie=kie, gdrive=FakeGDriveClient())
    monkeypatch.undo()
    session_b.close()

    # B must report the WINNER's real outcome, never re-process, and never
    # fail an attempt that already succeeded.
    assert result_b == "done"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt_id)
    assert reloaded_attempt.status == RangeAttemptStatus.done
    assert reloaded_attempt.promoted_at is not None
    assert reloaded_attempt.error_message is None
    assert reloaded_run.segment_snapshot == winner_snapshot


def test_integration_copy_survives_live_segment_def_deletion_and_cuts_promoted_bounds(
    db_session, db_engine, synthetic_video, monkeypatch,
):
    """Regression for the independent audit's blocking finding #1, driven
    through the REAL pipeline exactly like the test above: a REAL promotion,
    then the LIVE SegmentDef the promoted segment was created against gets
    deleted (the real pipeline_v2.analyze_project / bulk-segment-edit delete
    idiom: session.delete(sd)) — simulating an operator re-analyzing or
    editing the project sometime after the correction — then a REAL
    _build_copied_run clone, then a REAL pipeline_v2.submit_run. Before the
    fix, inserting the copy's fresh RunSegment (naming the same, now-dead
    segment_def_id) would have violated the FK and this call would have
    raised IntegrityError; the promoted bounds are asserted at the real
    cut_clip call site, not just as a JSON comparison."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie_for_attempt = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie_for_attempt)
    assert poll_range_attempt_once(attempt.id, kie=kie_for_attempt, gdrive=FakeGDriveClient()) == "done"

    db_session.expire_all()
    promoted_run = db_session.get(Run, run.id)
    sd_swap1_id = sd_swap1.id

    # The live SegmentDef the promoted swap segment/RunSegment/attempt were
    # created against is deleted — the exact idiom analyze_project and the
    # bulk segment-edit endpoint both use.
    live_def = db_session.get(SegmentDef, sd_swap1_id)
    db_session.delete(live_def)
    db_session.commit()
    assert db_session.get(SegmentDef, sd_swap1_id) is None

    new_run = api_v2_mod._build_copied_run(
        promoted_run, db_session,
        resolution=None, name=None, ref_files=[], ref_urls=[],
    )
    db_session.commit()  # before the fix: IntegrityError here
    new_by_id = {e["id"]: e for e in new_run.segment_snapshot}
    assert new_by_id[sd_swap1_id]["start_sec"] == pytest.approx(8.0)

    _Pv2Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

    @contextmanager
    def _pv2_test_get_session():
        sess = _Pv2Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    monkeypatch.setattr(pipeline_v2_mod, "get_session", _pv2_test_get_session)

    cut_calls = []
    real_cut_clip = media_mod.cut_clip

    def _spy_cut_clip(source, start, end, dst, **kw):
        cut_calls.append((source, start, end))
        return real_cut_clip(source, start, end, dst, **kw)

    monkeypatch.setattr(pipeline_v2_mod.media_mod, "cut_clip", _spy_cut_clip)

    kie_for_submit = FakeKieClient(synthetic_video)
    pending = pipeline_v2_mod.submit_run(new_run.id, kie=kie_for_submit, gdrive=FakeGDriveClient())
    assert pending, "submit_run must have submitted the copy's swap segment"

    swap_cuts = [c for c in cut_calls if c[1] == pytest.approx(8.0, abs=0.01)]
    assert swap_cuts, f"expected a cut_clip call at the PROMOTED start 8.0s; got {cut_calls}"
    assert swap_cuts[0][0] == synthetic_video


def test_source_fallback_segments_carried_through_promotion(db_session, synthetic_video):
    """run.source_fallback_segments (an existing, unrelated version-tracking
    counter) must survive a successful promotion unchanged — this attempt
    fixes ONE segment's timing, it does not touch the run-wide fallback
    count."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    run.source_fallback_segments = 2
    db_session.commit()

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "done"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_run.source_fallback_segments == 2
    assert reloaded_attempt.source_fallback_segments == 2


# ---------------------------------------------------------------------------
# Regression tests for the 4 findings from the Cursor CLI review gate
# (cursor-grok-4.6-high-fast, see tasks/reports/
# run_local_segment_range_regeneration.md §7).
# ---------------------------------------------------------------------------


def test_no_stitch_run_promotes_successfully_without_invalid_transition(
    db_session, synthetic_video,
):
    """Regression for review finding #1: a no-stitch run (stitch_plan == [])
    has no candidate final to deliver, so poll never enters `delivering` —
    _promote must still be able to reach `done` directly from `stitching`
    (RANGE_ATTEMPT_TRANSITIONS now allows this), not raise InvalidTransition
    partway through, which — before the fix — left _fail_attempt's commit
    flushing the very Run/RunSegment mutations a "failed" attempt must never
    apply."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(
        db_session, synthetic_video, stitch_plan=[],
    )
    orig_result_path = run.result_local_path

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "done"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs = db_session.get(RunSegment, rs.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.status == RangeAttemptStatus.done
    assert reloaded_attempt.promoted_at is not None
    # The candidate segment result IS the deliverable for a no-stitch run —
    # promoted onto the RunSegment...
    assert reloaded_rs.local_result_path == attempt.local_result_path
    by_id = {e["id"]: e for e in reloaded_run.segment_snapshot}
    assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(8.0)
    # ...but there is no candidate FINAL video for a no-stitch run, so
    # run.result_local_path is correctly left exactly as it was.
    assert reloaded_run.result_local_path == orig_result_path


def test_promote_failure_never_leaks_partial_mutations_into_failed_attempt(
    db_session, synthetic_video, monkeypatch,
):
    """Regression for review finding #1's general root cause: ANY exception
    raised inside _promote after it has started mutating run/rs (not just the
    no-stitch InvalidTransition case) must not let those mutations reach the
    DB via _fail_attempt's commit on the same session."""
    import app.pipeline_range_attempt as pra_mod

    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    orig_snapshot = list(run.segment_snapshot)
    orig_run_result_path = run.result_local_path
    orig_rs_result_path = rs.local_result_path

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    # Inject a failure AFTER _promote's mutations but from `transition` itself
    # (the same shape as the real no-stitch bug: an exception raised after
    # `run.segment_snapshot = ...` has already been assigned in-session).
    real_transition = pra_mod.transition

    def _boom_on_done(entity, target):
        if target == RangeAttemptStatus.done:
            raise RuntimeError("injected post-mutation failure")
        return real_transition(entity, target)

    monkeypatch.setattr(pra_mod, "transition", _boom_on_done)

    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "failed"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs = db_session.get(RunSegment, rs.id)
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.status == RangeAttemptStatus.failed
    assert reloaded_attempt.promoted_at is None
    # The critical assertion: run.segment_snapshot was reassigned to the
    # candidate INSIDE _promote before the injected failure — it must NOT
    # have survived the rollback.
    assert reloaded_run.segment_snapshot == orig_snapshot
    assert reloaded_run.result_local_path == orig_run_result_path
    assert reloaded_rs.local_result_path == orig_rs_result_path
    assert reloaded_rs.status == SegmentStatus.completed


def test_promotion_warms_the_new_bounds_keyed_keep_cache(db_session, synthetic_video):
    """Regression for review finding #2 (superseded by the finding-3 hardening
    pass): a promoted range that moves a `keep` neighbour's boundary must make
    a LATER reader (another range attempt, or a plain rerun_segment) resolve
    the NEW bounds, not the pre-promotion cut.

    Originally this was proved by asserting the run's shared clip_%04d.mp4
    cache got overwritten in place. That path-reuse was itself the bug the
    finding-3 hardening pass closed (a crash between the copy and the DB
    commit could leave that shared path holding NEW bytes under the OLD
    snapshot) — see storage.keep_clip_path's docstring. The cache is now
    keyed by (index, bounds), so the correct assertion is that promotion
    warms the file at the NEW bounds' key, and — just as importantly — never
    writes to the OLD bounds' key at all.
    """
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    # Pre-populate the run's clip cache for keep0 (index 0) as it would exist
    # from the run's ORIGINAL stitch — bounds [0, 9), matching the
    # pre-promotion snapshot.
    old_bounds_path = keep_clip_path(run.id, project.id, 0, 0.0, 9.0)
    media_mod.cut_clip(synthetic_video, 0.0, 9.0, old_bounds_path, fast_intermediate=True)
    old_bytes = open(old_bounds_path, "rb").read()

    # Promote a range that shrinks keep0 to [0, 8) (swap1 start 9.0 -> 8.0).
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "done"

    new_bounds_path = keep_clip_path(run.id, project.id, 0, 0.0, 8.0)
    assert os.path.exists(new_bounds_path)
    new_duration = media_mod.probe(new_bounds_path).duration_sec
    assert new_duration == pytest.approx(8.0, abs=0.3)

    # The OLD bounds' file is a harmless orphan now that the run's active
    # snapshot no longer names [0, 9) for this segment — but promotion must
    # never have touched it: same path, same bytes as before.
    assert os.path.exists(old_bounds_path)
    assert open(old_bounds_path, "rb").read() == old_bytes


def test_failed_promotion_never_corrupts_the_old_bounds_keep_cache(
    db_session, synthetic_video, monkeypatch,
):
    """Regression for review finding #3: a promotion that starts writing the
    (now new-bounds-keyed) keep cache and then fails before its commit must
    never leave anything wrong at the path a still-active OLD snapshot would
    read. Pre-fix, the cache write targeted the OLD index-only path directly,
    so an injected failure here reproduced exactly the corruption finding #3
    described: the shared cache silently held NEW bytes under the OLD
    snapshot's own path. Bounds-keying makes this structurally impossible —
    this test proves it stays that way even when promotion fails partway
    through, and that a later real stitch still resolves the OLD bounds
    correctly."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    old_bounds_path = keep_clip_path(run.id, project.id, 0, 0.0, 9.0)
    media_mod.cut_clip(synthetic_video, 0.0, 9.0, old_bounds_path, fast_intermediate=True)
    old_bytes = open(old_bounds_path, "rb").read()
    orig_snapshot = list(run.segment_snapshot)

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    # Fail AFTER _promote's keep-cache warm-up (which runs before the
    # candidate_by_id-driven copy loop's own final `transition(..., done)`
    # call) but before the commit that would have made it real.
    real_transition = pra_mod.transition

    def _boom_on_done(entity, target):
        if target == RangeAttemptStatus.done:
            raise RuntimeError("injected post-cache-warm failure")
        return real_transition(entity, target)

    monkeypatch.setattr(pra_mod, "transition", _boom_on_done)

    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "failed"

    # The OLD bounds' file — what a still-active OLD snapshot would read —
    # is byte-for-byte exactly what it was before this attempt existed.
    assert open(old_bounds_path, "rb").read() == old_bytes

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    assert reloaded_run.segment_snapshot == orig_snapshot

    # A later REAL stitch/rerun path resolves the OLD bounds correctly —
    # exercise _resolve_keep_clip directly with the (unchanged) active
    # snapshot entry, mirroring what _stitch_candidate / pipeline_v2's own
    # stitch would look up for this segment.
    entry = orig_snapshot[0]
    resolved = pra_mod._resolve_keep_clip(entry, synthetic_video, run.id, project.id)
    assert resolved == old_bounds_path
    assert media_mod.probe(resolved).duration_sec == pytest.approx(9.0, abs=0.3)


def test_range_attempt_prompt_falls_back_to_segment_override(db_session, synthetic_video):
    """Regression for review finding #3: a range-only regeneration (no
    attempt-level prompt_override supplied) must still use the segment's
    EXISTING hand prompt override — not skip straight to the run-level
    prompt — matching the reference-URL fallback chain just above it in
    submit_range_attempt, and the product rule that per-segment prompt
    overrides remain applicable for a range-only regeneration."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    rs.prompt_override = "segment-tuned prompt text"
    db_session.commit()

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None,  # no attempt-level override
        reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    assert kie.create_task_records, "expected a create_task call"
    assert kie.create_task_records[-1]["prompt"] == "segment-tuned prompt text"


def test_second_active_attempt_creation_is_rejected_at_db_level(
    db_session, synthetic_video,
):
    """Regression for review finding #4, hardened: the API's active-attempt
    409 gate (api_v2._active_range_attempt) is a non-locking SELECT that
    cannot, by itself, stop two concurrent range-rerun requests from both
    creating an attempt. This is now backstopped by a real DB constraint
    (uq_run_segment_range_attempts_active_per_run, migration 73df0ff3f344,
    mirrored in RunSegmentRangeAttempt.__table_args__) — a second attempt for
    a run that already has a non-terminal one must fail AT INSERT, before
    either could ever race to promote. This is what makes the OLD version of
    this test (which built two simultaneously-active attempts directly via
    create_range_attempt, bypassing the API gate, to prove _promote's
    snapshot-staleness check alone was sufficient) no longer constructible —
    the race it modelled is now impossible one layer earlier. See
    test_duplicate_worker_promotion_of_the_same_attempt_serializes_and_loser_fails_cleanly
    below for what still needs (and gets) the staleness check: the SAME
    attempt processed by two worker sessions at once."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    with pytest.raises(IntegrityError):
        create_range_attempt(
            run=run, rs=rs, project=project,
            start_sec=9.0, end_sec=16.0,
            prompt_override=None, reference_image_urls_override=None,
            attempt_id=str(uuid.uuid4()), db=db_session,
        )
    db_session.rollback()


def test_sequential_range_attempts_after_a_promotion_still_work_normally(
    db_session, synthetic_video,
):
    """The DB-level uniqueness fix must only block CONCURRENT/overlapping
    attempts, not the ordinary sequential correction-after-correction flow: a
    second range regeneration, requested only after the first one has already
    promoted (so it is no longer active), must create and promote normally,
    building on the FIRST attempt's already-updated snapshot."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    attempt_a = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie_a = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt_a.id, kie=kie_a)
    assert poll_range_attempt_once(attempt_a.id, kie=kie_a, gdrive=FakeGDriveClient()) == "done"

    db_session.expire_all()
    run = db_session.get(Run, run.id)
    rs = db_session.get(RunSegment, rs.id)

    # Attempt B is only created NOW, reading the run's CURRENT (A-promoted)
    # snapshot as its base — not a race, the normal case.
    attempt_b = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=7.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie_b = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt_b.id, kie=kie_b)
    assert poll_range_attempt_once(attempt_b.id, kie=kie_b, gdrive=FakeGDriveClient()) == "done"

    db_session.expire_all()
    reloaded_run = db_session.get(Run, run.id)
    by_id = {e["id"]: e for e in reloaded_run.segment_snapshot}
    assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(7.0)


def test_duplicate_worker_promotion_of_the_same_attempt_serializes_and_loser_fails_cleanly(
    db_session, db_engine, synthetic_video, monkeypatch,
):
    """Regression for review finding #4's remaining half: the DB uniqueness
    constraint stops a SECOND attempt from ever coexisting, but it cannot
    stop the SAME attempt being processed by two worker sessions at once — a
    duplicate poll enqueue, or a recovery re-enqueue racing a still-running
    job, both of which have happened for other derived creatives in this
    codebase. Two independent sessions, each holding their OWN loaded copies
    of run/rs/attempt, both call _promote for the SAME attempt; the second
    must fail via the staleness check seeing the first's already-committed
    change, not silently re-apply a stale candidate on top.

    Sequenced deterministically here (SQLite has no real row-level locking,
    so `with_for_update` is a no-op on it) to prove the LOGIC — that
    session.refresh() picks up the winner's committed change and the
    staleness check correctly refuses. The actual concurrency GUARANTEE
    (two sessions genuinely blocking on the same row lock) is proved
    separately against disposable PostgreSQL — see
    tests/test_pipeline_range_attempt_pg_concurrency.py.
    """
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    orig_snapshot = list(run.segment_snapshot)

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    # Drive the attempt through generate+stitch+deliver WITHOUT letting the
    # real _promote run — this is the exact state a duplicate poll would find
    # the row in: everything a promotion needs is ready, but nothing has
    # promoted yet. Restored before the manual _promote calls below.
    monkeypatch.setattr(pra_mod, "_promote", lambda *a, **k: None)
    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "done"  # poll_range_attempt_once's own return, not a real promotion
    monkeypatch.undo()

    db_session.expire_all()
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded_attempt.promoted_at is None  # confirms _promote really was suppressed
    attempt_id = attempt.id

    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session_1 = Session()
    session_2 = Session()
    try:
        run_1 = session_1.get(Run, run.id)
        rs_1 = session_1.get(RunSegment, rs.id)
        attempt_1 = session_1.get(RunSegmentRangeAttempt, attempt_id)
        run_2 = session_2.get(Run, run.id)
        rs_2 = session_2.get(RunSegment, rs.id)
        attempt_2 = session_2.get(RunSegmentRangeAttempt, attempt_id)

        # Winner: promotes and commits first.
        pra_mod._promote(session_1, run_1, rs_1, attempt_1)
        session_1.commit()
        winner_snapshot = list(run_1.segment_snapshot)
        winner_result_path = rs_1.local_result_path
        assert winner_snapshot != orig_snapshot

        # Loser: was holding its OWN copies loaded before session_1
        # committed — session.refresh(with_for_update=True) inside _promote
        # must re-read the CURRENT row (the winner's change), see the
        # staleness mismatch, and raise — never silently reapply attempt_2's
        # (now stale) candidate on top of the winner's.
        with pytest.raises(RuntimeError, match="active timeline changed"):
            pra_mod._promote(session_2, run_2, rs_2, attempt_2)
        session_2.rollback()
    finally:
        session_1.close()
        session_2.close()

    with Session() as verify:
        final_run = verify.get(Run, run.id)
        final_rs = verify.get(RunSegment, rs.id)
        assert final_run.segment_snapshot == winner_snapshot
        assert final_rs.local_result_path == winner_result_path


# ---------------------------------------------------------------------------
# Claim/lease hardening — per-phase ownership before shared-file/external
# side effects (submit_range_attempt / poll_range_attempt_once's finalize
# path). Sequenced deterministically here to prove the LOGIC of each
# individual failure/recovery case; the genuine simultaneous-worker proof
# (two real threads, a barrier, disposable PostgreSQL) lives in
# tests/test_pipeline_range_attempt_pg_concurrency.py.
# ---------------------------------------------------------------------------


def test_submit_live_claim_blocks_a_duplicate_worker_without_touching_anything(
    db_session, synthetic_video,
):
    """A non-expired claim must block a second submit outright — before it
    cuts, uploads, or calls createTask."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    # Simulate another worker actively holding the submit-phase claim.
    attempt.claim_owner = "other-worker-token"
    attempt.claim_expires_at = datetime.now(timezone.utc) + timedelta(seconds=300)
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    result = submit_range_attempt(attempt.id, kie=kie)
    assert result is None
    assert not kie.create_task_records
    assert not kie.upload_calls

    db_session.expire_all()
    reloaded = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert reloaded.status == RangeAttemptStatus.queued
    assert reloaded.local_clip_path is None
    assert reloaded.claim_owner == "other-worker-token"  # untouched, not stolen


def test_poll_takes_over_an_expired_finalize_claim_and_completes(db_session, synthetic_video):
    """Unlike submission, an expired finalize claim is always safe to take
    over — nothing past that point makes a paid external call."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)

    db_session.expire_all()
    attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    # Simulate a crashed previous finalize worker: it claimed, then died —
    # its lease has since expired.
    attempt.claim_owner = "dead-worker-token"
    attempt.claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db_session.commit()

    result = poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "done"

    db_session.expire_all()
    final = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert final.status == RangeAttemptStatus.done
    assert final.promoted_at is not None


def test_submit_ambiguous_billing_window_is_never_retried_and_marked_failed(
    db_session, synthetic_video,
):
    """submit_billed_at set with no seedance_task_id, past an expired claim,
    means a previous worker may have already made kie.ai's paid createTask
    call before crashing. kie.ai has no idempotency key, so this must NEVER
    be auto-retried — it is marked failed for manual reconciliation."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    attempt.claim_owner = "dead-worker-token"
    attempt.claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    billed_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    attempt.submit_billed_at = billed_at
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    result = submit_range_attempt(attempt.id, kie=kie)
    assert result is None
    assert not kie.create_task_records, "must NOT retry the paid call"
    assert not kie.upload_calls

    db_session.expire_all()
    final = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert final.status == RangeAttemptStatus.failed
    assert "AMBIGUOUS" in final.error_message
    assert "no idempotency" in final.error_message.lower() or "idempotency" in final.error_message.lower()
    assert final.seedance_task_id is None


def test_submit_expired_claim_without_billing_is_safely_retaken(db_session, synthetic_video):
    """An expired claim with submit_billed_at still NULL means the previous
    worker died before the paid call — safely retryable from scratch."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    attempt.claim_owner = "dead-worker-token"
    attempt.claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    result = submit_range_attempt(attempt.id, kie=kie)
    assert result is not None
    assert len(kie.create_task_records) == 1

    db_session.expire_all()
    final = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert final.status == RangeAttemptStatus.generating
    assert final.claim_owner is None  # released once the phase concluded


def test_takeover_of_a_dead_claim_never_touches_that_claims_own_files(
    db_session, synthetic_video,
):
    """A takeover of an expired, non-ambiguous claim must not delete or
    overwrite whatever the dead worker's OWN (never-cleaned-up) claim
    directory holds — cleanup is scoped to the caller's OWN superseded
    claim, never a pre-existing dead one it is taking over from."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()

    from app.storage import range_attempt_dir

    dead_token = "dead-worker-token"
    attempt_dir = range_attempt_dir(attempt.id, rs.id, run.id, run.project_id)
    dead_claim_dir = pra_mod._claim_dir(attempt_dir, dead_token)
    dead_marker = os.path.join(dead_claim_dir, "clip.mp4")
    shutil.copy2(synthetic_video, dead_marker)

    attempt.claim_owner = dead_token
    attempt.claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db_session.commit()

    kie = FakeKieClient(synthetic_video)
    result = submit_range_attempt(attempt.id, kie=kie)
    assert result is not None

    assert os.path.exists(dead_marker)
    assert os.path.getsize(dead_marker) == os.path.getsize(synthetic_video)

    db_session.expire_all()
    final = db_session.get(RunSegmentRangeAttempt, attempt.id)
    assert final.local_clip_path != dead_marker
    assert os.path.exists(final.local_clip_path)


def test_claim_superseded_mid_cut_cleans_up_only_its_own_directory(
    db_session, db_engine, synthetic_video, monkeypatch,
):
    """If this call's claim is superseded (by a takeover) while it is still
    mid-flight, it must discard only its OWN claim directory on the way
    out — never fail the attempt, never touch the new owner's claim."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    attempt_id = attempt.id

    real_cut_clip = media_mod.cut_clip

    def _cut_then_get_superseded(*a, **kw):
        # Runs from INSIDE submit_range_attempt, right after it committed
        # its own claim — simulate a second worker's takeover landing before
        # this call's next commit, exactly the window a too-short lease (or
        # a slow ffmpeg) could expose in production.
        Session2 = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
        s2 = Session2()
        other = s2.get(RunSegmentRangeAttempt, attempt_id)
        other.claim_owner = "other-worker-token"
        other.claim_expires_at = datetime.now(timezone.utc) + timedelta(seconds=300)
        s2.commit()
        s2.close()
        return real_cut_clip(*a, **kw)

    monkeypatch.setattr(media_mod, "cut_clip", _cut_then_get_superseded)

    kie = FakeKieClient(synthetic_video)
    result = submit_range_attempt(attempt_id, kie=kie)
    assert result is None, "this call lost the claim mid-flight — must exit benignly"
    assert not kie.create_task_records
    assert not kie.upload_calls

    db_session.expire_all()
    final = db_session.get(RunSegmentRangeAttempt, attempt_id)
    assert final.status == RangeAttemptStatus.queued  # untouched by the loser
    assert final.error_message is None
    assert final.claim_owner == "other-worker-token"  # the new owner's claim stands

    from app.storage import range_attempt_dir

    attempt_dir = range_attempt_dir(attempt_id, rs.id, run.id, run.project_id)
    claims_root = os.path.join(attempt_dir, "claims")
    remaining = os.listdir(claims_root) if os.path.exists(claims_root) else []
    assert remaining == [], "the loser's own claim directory must be cleaned up"


# ---------------------------------------------------------------------------
# Follow-up hardening (2026-08-30, independent audit): lease sized to the RQ
# job timeout, explicit ownership re-checks, claim-scoped keep-cache
# publish, and production recovery of a crashed finalize-phase worker.
# ---------------------------------------------------------------------------


def test_claim_lease_floor_matches_the_job_timeout_relationship():
    """_CLAIM_LEASE_SEC must never be configurable below
    tasks.RANGE_ATTEMPT_POLL_JOB_TIMEOUT — the exact relationship that
    makes an expired claim PROOF (not merely a guess) that RQ has already
    force-killed its owning worker. app.tasks cannot be imported at
    pipeline_range_attempt module scope (it imports this module's own
    functions, deferred, to avoid a cycle), so the two constants are kept
    in sync by hand — this test is what actually enforces that."""
    import app.tasks as tasks_mod

    assert pra_mod._RANGE_ATTEMPT_JOB_TIMEOUT_SEC == tasks_mod.RANGE_ATTEMPT_POLL_JOB_TIMEOUT
    assert pra_mod._CLAIM_LEASE_SEC > tasks_mod.RANGE_ATTEMPT_POLL_JOB_TIMEOUT
    # Mirrors tasks._RUN_LOCK_TTL_SEC's own margin (PROCESS_JOB_TIMEOUT + 300)
    # for the main pipeline's Redis lock — the same proven pattern.
    assert pra_mod._CLAIM_LEASE_SEC >= tasks_mod.RANGE_ATTEMPT_POLL_JOB_TIMEOUT + 300


def test_stale_owner_cannot_deliver_promote_or_fail_the_winner_after_a_takeover(
    db_session, synthetic_video,
):
    """A stale owner (its claim_token no longer matches the row's CURRENT
    claim_owner — e.g. after a legitimate takeover) must be stopped by
    _assert_still_owner before promoting, and must never mark the row
    failed, re-deliver, re-promote, or mutate the Run/RunSegment."""
    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)

    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt.id, kie=kie)
    assert poll_range_attempt_once(attempt.id, kie=kie, gdrive=FakeGDriveClient()) == "done"

    db_session.expire_all()
    final = db_session.get(RunSegmentRangeAttempt, attempt.id)
    final_run = db_session.get(Run, run.id)
    final_rs = db_session.get(RunSegment, rs.id)
    assert final.status == RangeAttemptStatus.done
    winning_snapshot = list(final_run.segment_snapshot)
    winning_result_path = final_rs.local_result_path

    # A stale token (never the row's real, winning claim_owner, whatever
    # that happened to be) must be rejected outright by the ownership
    # check — before touching anything.
    stale_token = "some-other-dead-token"
    assert final.claim_owner != stale_token
    with pytest.raises(pra_mod._ClaimLost):
        pra_mod._assert_still_owner(db_session, attempt.id, stale_token)

    # _promote itself refuses a stale token the same way, and never
    # re-applies (or fails) an already-`done` attempt.
    with pytest.raises(pra_mod._ClaimLost):
        pra_mod._promote(db_session, final_run, final_rs, final, claim_token=stale_token)

    db_session.expire_all()
    reloaded_attempt = db_session.get(RunSegmentRangeAttempt, attempt.id)
    reloaded_run = db_session.get(Run, run.id)
    reloaded_rs = db_session.get(RunSegment, rs.id)
    assert reloaded_attempt.status == RangeAttemptStatus.done
    assert reloaded_attempt.error_message is None
    assert reloaded_run.segment_snapshot == winning_snapshot
    assert reloaded_rs.local_result_path == winning_result_path


def test_recovered_attempt_finalizes_via_the_real_poll_path_with_no_new_createtask(
    db_session, db_engine, synthetic_video, monkeypatch,
):
    """End-to-end for Finding 3: after recovery.py's reaper recovers a
    crashed finalizer (claim cleared, no new paid call), the SAME real
    poll_range_attempt_once() the RQ wrapper would call completes the
    attempt to `done` using the ALREADY-persisted task id — proving no
    NEW kie.ai createTask call is ever made."""
    from unittest.mock import patch as _patch

    import app.recovery as rec

    # recovery.py's reaper opens its OWN sessions via app.recovery.get_session
    # — point it at this test's isolated engine too (this file's own
    # autouse fixture only patches pipeline_range_attempt.get_session).
    _RecoverySession = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

    @contextmanager
    def _recovery_get_session():
        sess = _RecoverySession()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    monkeypatch.setattr(rec, "get_session", _recovery_get_session)

    project, run, sd_keep0, sd_swap1, sd_keep2, rs = _make_pipeline_run(db_session, synthetic_video)
    attempt = create_range_attempt(
        run=run, rs=rs, project=project,
        start_sec=8.0, end_sec=15.0,
        prompt_override=None, reference_image_urls_override=None,
        attempt_id=str(uuid.uuid4()), db=db_session,
    )
    db_session.commit()
    attempt_id = attempt.id

    kie = FakeKieClient(synthetic_video)
    submit_range_attempt(attempt_id, kie=kie)
    assert len(kie.create_task_records) == 1

    # Simulate the crash: worker died partway through finalize, its claim
    # still (falsely) "live" per claim_expires_at, aged past the reaper's
    # 5-minute grace window.
    db_session.expire_all()
    attempt = db_session.get(RunSegmentRangeAttempt, attempt_id)
    attempt.status = RangeAttemptStatus.stitching
    attempt.claim_owner = "dead-worker-token"
    attempt.claim_expires_at = datetime.now(timezone.utc) + timedelta(seconds=3000)
    attempt.updated_at = datetime.utcnow() - timedelta(seconds=600)
    db_session.commit()

    # db_engine is module-scoped (shared across every test in this file) —
    # by the time this test runs, an EARLIER test's own range attempt may
    # have naturally aged past the reaper's 5-minute grace window in real
    # wall-clock time, so assert on THIS attempt specifically
    # (assert_any_call), never that this was the only row reaped/recovered.
    with _patch("app.recovery._enqueue_range_attempt_poll") as mock_enq:
        rec._reap_orphaned_derived_creatives(set())
        mock_enq.assert_any_call(attempt_id, delay_sec=0)

    db_session.expire_all()
    attempt = db_session.get(RunSegmentRangeAttempt, attempt_id)
    assert attempt.claim_owner is None
    # Reset to `generating` — the whole finalize sequence is redone from
    # scratch; there is no "resume from stitching" edge.
    assert attempt.status == RangeAttemptStatus.generating

    # What the enqueued recovery job would actually run:
    result = poll_range_attempt_once(attempt_id, kie=kie, gdrive=FakeGDriveClient())
    assert result == "done"
    assert len(kie.create_task_records) == 1, "no NEW createTask call was made"

    db_session.expire_all()
    final = db_session.get(RunSegmentRangeAttempt, attempt_id)
    final_run = db_session.get(Run, run.id)
    assert final.status == RangeAttemptStatus.done
    assert final.promoted_at is not None
    by_id = {e["id"]: e for e in final_run.segment_snapshot}
    assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(8.0)
