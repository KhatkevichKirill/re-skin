"""
Tests for the audio-patch REST API (app/api_v2.py, per docs/run-audio-patch.md):

    POST /runs/{rid}/audio-patches
    GET  /runs/{rid}/audio-patches
    GET  /audio-patches/{pid}/result

Follows test_api_captions.py's conventions: own temp SQLite DB + engine, a
`client` fixture that overrides the DB dependency and stubs the enqueue to a
no-op, and a `db_session` fixture for direct ORM setup/asserts.

The run/donor videos here are tiny real files (the endpoint probes durations to
produce a bound-naming 400), but no ffmpeg splice runs — the worker half is
covered by tests/test_pipeline_audio_patch.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager

# Must set DATABASE_URL and DATA_DIR before any app import.
_db_fd, _db_path = tempfile.mkstemp(suffix="_api_audio_patch.db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
_data_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _data_tmp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.models import Run, RunAudioPatch, RunSegment, RunSegmentRangeAttempt, VideoProject
from app.state_machine import (
    AudioPatchStatus,
    ProjectStatus,
    RangeAttemptStatus,
    RunStatus,
    SegmentStatus,
)
import app.api_v2 as api_v2_module


_FFMPEG = "/usr/bin/ffmpeg"


def _ffmpeg_available() -> bool:
    try:
        r = subprocess.run(
            [_FFMPEG, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False


if not _ffmpeg_available():
    pytest.skip("ffmpeg not available", allow_module_level=True)


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(
        f"sqlite:///{_db_path}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(eng, "connect")
    def set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="session")
def SessionFactory(engine):
    return sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )


@pytest.fixture()
def db_session(SessionFactory):
    session = SessionFactory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def enqueued():
    return []


@pytest.fixture()
def client(engine, SessionFactory, monkeypatch, enqueued):
    monkeypatch.setattr(
        api_v2_module, "enqueue_process_audio_patch", lambda pid: enqueued.append(pid)
    )

    from app.main import app

    def override_get_db():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    # download_audio_patch_result manages its own session via `with
    # get_session()`, not the overridden dependency — patch the name in api_v2
    # so it uses this test's engine regardless of import order (mirrors
    # test_api_captions.py's client fixture).
    @contextmanager
    def _test_get_session():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(api_v2_module, "get_session", _test_get_session)

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video(path: str, duration: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cmd = [
        _FFMPEG, "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=160x120:rate=10",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-shortest", path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-400:])


def _make_project(session, **kwargs) -> VideoProject:
    defaults = dict(
        source_type="upload", source_ref="test.mp4", status=ProjectStatus.ready
    )
    defaults.update(kwargs)
    p = VideoProject(**defaults)
    session.add(p)
    session.commit()
    return p


def _make_run(session, project, *, duration: float = 8.0, with_result: bool = True,
              **kwargs) -> Run:
    defaults = dict(
        project_id=project.id, prompt="p", status=RunStatus.done,
        result_gdrive_file_id="drive-x",
    )
    defaults.update(kwargs)
    run = Run(**defaults)
    session.add(run)
    session.commit()
    if with_result:
        path = os.path.join(_data_tmp, f"{run.id}_final.mp4")
        _make_video(path, duration)
        run.result_local_path = path
        run.result_duration_sec = duration
        session.commit()
    return run


def _pair(session, **kw):
    project = _make_project(session)
    target = _make_run(session, project, **kw)
    donor = _make_run(session, project)
    return project, target, donor


def _post(client, rid, **overrides):
    data = {
        "donor_run_id": overrides.pop("donor_run_id", None),
        "target_start_sec": 3.0,
        "target_end_sec": 4.0,
        "donor_start_sec": 1.0,
    }
    data.update(overrides)
    return client.post(f"/api/v2/runs/{rid}/audio-patches", data=data)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_returns_201_and_enqueues(client, db_session, enqueued):
    _project, target, donor = _pair(db_session)
    r = _post(client, target.id, donor_run_id=donor.id)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["run_id"] == target.id
    assert body["donor_run_id"] == donor.id
    assert body["status"] == "queued"
    assert body["target_start_sec"] == 3.0
    assert enqueued == [body["id"]]

    row = db_session.get(RunAudioPatch, body["id"])
    # The window was picked against THIS file; _promote refuses any other.
    assert row.base_result_path == target.result_local_path


def test_crossfade_defaults_when_omitted(client, db_session):
    _project, target, donor = _pair(db_session)
    r = _post(client, target.id, donor_run_id=donor.id)
    from app.media import AUDIO_SPLICE_DEFAULT_CROSSFADE_SEC

    assert r.json()["crossfade_sec"] == AUDIO_SPLICE_DEFAULT_CROSSFADE_SEC


def test_create_404_for_unknown_donor(client, db_session):
    _project, target, _donor = _pair(db_session)
    r = _post(client, target.id, donor_run_id="no-such-run")
    assert r.status_code == 404
    assert "Donor run" in r.json()["detail"]


def test_create_409_when_run_not_done(client, db_session):
    project = _make_project(db_session)
    target = _make_run(db_session, project, status=RunStatus.processing)
    donor = _make_run(db_session, project)
    r = _post(client, target.id, donor_run_id=donor.id)
    assert r.status_code == 409
    assert "expected 'done'" in r.json()["detail"]


def test_create_409_when_a_patch_is_already_active(client, db_session):
    _project, target, donor = _pair(db_session)
    assert _post(client, target.id, donor_run_id=donor.id).status_code == 201
    r = _post(client, target.id, donor_run_id=donor.id)
    assert r.status_code == 409
    assert "audio patch is in progress" in r.json()["detail"]


def test_create_409_while_a_range_attempt_is_active(client, db_session):
    """Both features end by re-pointing result_local_path, so they exclude
    each other in both directions."""
    _project, target, donor = _pair(db_session)
    rs = RunSegment(
        run_id=target.id, segment_def_id="sd-1", index=0,
        status=SegmentStatus.completed,
    )
    db_session.add(rs)
    db_session.commit()
    db_session.add(
        RunSegmentRangeAttempt(
            run_id=target.id, run_segment_id=rs.id,
            requested_start_sec=0.0, requested_end_sec=2.0,
            base_segment_snapshot=[], candidate_segment_snapshot=[],
            status=RangeAttemptStatus.generating,
        )
    )
    db_session.commit()

    r = _post(client, target.id, donor_run_id=donor.id)
    assert r.status_code == 409
    assert "range regeneration" in r.json()["detail"]


def test_create_409_when_the_result_moves_during_validation(
    client, db_session, monkeypatch
):
    """§1-gate finding (4th pass): the duration is resolved before the Run-row
    lock and base_result_path is captured after it, so a promotion in between
    would leave the two describing different videos — and the operator picked
    the window against a video the run no longer serves."""
    _project, target, donor = _pair(db_session)
    moved = os.path.join(_data_tmp, f"{target.id}_final_v2.mp4")
    _make_video(moved, 8.0)

    original = api_v2_module._duration_before_lock

    def _promote_midway(run):
        # Stand in for a range attempt promoting between the probe and the lock.
        result = original(run)
        if run.id == target.id:
            db_session.query(Run).filter_by(id=target.id).update(
                {"result_local_path": moved}
            )
            db_session.commit()
        return result

    monkeypatch.setattr(api_v2_module, "_duration_before_lock", _promote_midway)

    r = _post(client, target.id, donor_run_id=donor.id)
    assert r.status_code == 409
    assert "changed while the request was being validated" in r.json()["detail"]


def test_create_400_when_window_runs_past_the_end(client, db_session):
    _project, target, donor = _pair(db_session)
    r = _post(client, target.id, donor_run_id=donor.id,
              target_start_sec=6.0, target_end_sec=30.0)
    assert r.status_code == 400
    # The bound is named, so the operator can fix it without guessing.
    assert "8.00s long" in r.json()["detail"]


def test_create_400_when_donor_is_too_short_for_the_window(client, db_session):
    _project, target, donor = _pair(db_session)
    r = _post(client, target.id, donor_run_id=donor.id,
              target_start_sec=1.0, target_end_sec=5.0, donor_start_sec=7.0)
    assert r.status_code == 400
    assert "only" in r.json()["detail"]


def test_create_400_for_a_donor_from_another_project(client, db_session):
    """§1-gate finding (2026-09-07): the UI offers siblings only; the API has
    to enforce it too, or a pasted id from another project permanently
    replaces a delivered creative's audio with unrelated speech."""
    _project, target, _donor = _pair(db_session)
    other_project = _make_project(db_session)
    stranger = _make_run(db_session, other_project)

    r = _post(client, target.id, donor_run_id=stranger.id)
    assert r.status_code == 400
    assert "different project" in r.json()["detail"]


def test_create_400_for_self_donor(client, db_session):
    _project, target, _donor = _pair(db_session)
    r = _post(client, target.id, donor_run_id=target.id)
    assert r.status_code == 400
    assert "different run" in r.json()["detail"]


def test_create_400_for_a_zero_length_window(client, db_session):
    _project, target, donor = _pair(db_session)
    r = _post(client, target.id, donor_run_id=donor.id,
              target_start_sec=3.0, target_end_sec=3.0)
    assert r.status_code == 400
    assert "at least" in r.json()["detail"]


# ---------------------------------------------------------------------------
# List + download
# ---------------------------------------------------------------------------


def test_list_is_newest_first_and_labels_the_donor(client, db_session):
    _project, target, donor = _pair(db_session)
    donor.name = "RplDcTop"
    db_session.commit()
    _post(client, target.id, donor_run_id=donor.id)
    db_session.query(RunAudioPatch).filter_by(run_id=target.id).update(
        {"status": AudioPatchStatus.done}
    )
    db_session.commit()
    _post(client, target.id, donor_run_id=donor.id,
          target_start_sec=5.0, target_end_sec=6.0)

    r = client.get(f"/api/v2/runs/{target.id}/audio-patches")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 2
    assert items[0]["target_start_sec"] == 5.0  # newest first
    assert items[0]["donor_label"] == f"RplDcTop · {donor.short_id}"


def test_list_falls_back_to_the_id_for_a_deleted_donor(client, db_session):
    _project, target, donor = _pair(db_session)
    _post(client, target.id, donor_run_id=donor.id)
    donor_id = donor.id
    db_session.delete(donor)
    db_session.commit()

    items = client.get(f"/api/v2/runs/{target.id}/audio-patches").json()
    assert items[0]["donor_label"] == donor_id[:8]


def test_download_404_when_nothing_rendered_yet(client, db_session):
    _project, target, donor = _pair(db_session)
    pid = _post(client, target.id, donor_run_id=donor.id).json()["id"]
    r = client.get(f"/api/v2/audio-patches/{pid}/result")
    assert r.status_code == 404


def test_download_serves_the_candidate(client, db_session):
    from app.storage import data_dir_root

    _project, target, donor = _pair(db_session)
    pid = _post(client, target.id, donor_run_id=donor.id).json()["id"]
    # Other API modules may import storage first when suites run together.
    # Result delivery intentionally refuses paths outside that actual root.
    out = os.path.join(data_dir_root(), f"{pid}_patched.mp4")
    _make_video(out, 2.0)
    db_session.query(RunAudioPatch).filter_by(id=pid).update({"output_path": out})
    db_session.commit()

    r = client.get(f"/api/v2/audio-patches/{pid}/result")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
