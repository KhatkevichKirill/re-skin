"""
Tests for app/pipeline_merge.py — process_merge (PIP-insert merge feature).

Strategy
--------
* Own SQLite DB engine (file-based, isolated from other test modules).
  pipeline_merge's get_session() is monkeypatched to use this engine (same
  pattern as test_pipeline_v2.py).
* app.media.probe / get_default_target / make_rounded_mask / overlay_insert
  are monkeypatched to lightweight stubs that record call args and write a
  placeholder output file — no real ffmpeg is invoked.
* A FakeGDriveClient records upload_file calls and returns a predictable id.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_merge_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db import Base
from app.models import InsertMerge, Run, VideoProject
from app.state_machine import MergeStatus, ProjectStatus, RunStatus
from app.storage import project_source_path, run_results_dir


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    db_path = os.path.join(_TMP_DIR, "pipeline_merge_test.db")
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture(autouse=True)
def patch_get_session(db_engine, monkeypatch):
    """Replace app.pipeline_merge's get_session with one bound to the test engine."""
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

    import app.pipeline_merge as pipeline_merge_mod
    monkeypatch.setattr(pipeline_merge_mod, "get_session", _test_get_session)


@pytest.fixture(autouse=True)
def _no_default_folder(monkeypatch):
    """Default to no Drive delivery unless a test explicitly opts in."""
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "")


# ---------------------------------------------------------------------------
# Fake GDriveClient
# ---------------------------------------------------------------------------


class FakeGDriveClient:
    """Records upload_file calls; can be made to fail N times before succeeding."""

    def __init__(self, fail_times: int = 0):
        self.upload_calls: list[dict] = []
        self._fail_times = fail_times

    def upload_file(self, local_path, folder_id=None, name=None, **kw) -> dict:
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("injected upload failure")
        self.upload_calls.append(
            {"local_path": local_path, "folder_id": folder_id, "name": name}
        )
        file_id = f"gdrive-{uuid.uuid4().hex[:8]}"
        return {
            "id": file_id,
            "webViewLink": f"https://drive.google.com/file/d/{file_id}/view",
        }


class AlwaysFailGDriveClient:
    def upload_file(self, *a, **kw):
        raise RuntimeError("Drive is down")


# ---------------------------------------------------------------------------
# DB fixture builders
# ---------------------------------------------------------------------------


def _make_target_project(session, **kwargs) -> VideoProject:
    defaults = dict(
        source_type="upload",
        source_ref="target.mp4",
        status=ProjectStatus.ready,
        width=640,
        height=480,
        duration_sec=20.0,
        fps=30.0,
        aspect_ratio="4:3",
    )
    defaults.update(kwargs)
    p = VideoProject(**defaults)
    session.add(p)
    session.flush()

    src_path = project_source_path(p.id, "mp4")
    with open(src_path, "wb") as fh:
        fh.write(b"fake-target-video-bytes")
    p.source_local_path = src_path
    session.commit()
    return p


def _make_done_child_run(session, *, no_result: bool = False, **kwargs) -> Run:
    child_project = VideoProject(
        source_type="upload", source_ref="child.mp4", status=ProjectStatus.ready
    )
    session.add(child_project)
    session.flush()

    defaults = dict(
        project_id=child_project.id,
        prompt="swap the character",
        status=RunStatus.done,
    )
    defaults.update(kwargs)
    run = Run(**defaults)
    session.add(run)
    session.flush()

    if not no_result:
        result_dir = run_results_dir(run.id, child_project.id)
        result_path = os.path.join(result_dir, "final.mp4")
        with open(result_path, "wb") as fh:
            fh.write(b"fake-child-result-bytes")
        run.result_local_path = result_path

    session.commit()
    return run


def _make_merge(session, target: VideoProject, run: Run, **kwargs) -> InsertMerge:
    defaults = dict(
        target_project_id=target.id,
        child_run_id=run.id,
        x=10, y=20, width=100, height=80,
        start_sec=1.0, end_sec=5.0,
        corner_radius_px=0,
        status=MergeStatus.queued,
    )
    defaults.update(kwargs)
    merge = InsertMerge(**defaults)
    session.add(merge)
    session.commit()
    return merge


# ---------------------------------------------------------------------------
# media.py stubs
# ---------------------------------------------------------------------------


class _FakeInfo:
    width = 640
    height = 480
    fps = 30.0
    duration_sec = 20.0


@pytest.fixture()
def media_stubs(monkeypatch):
    """Stub media_mod.probe/get_default_target/make_rounded_mask/overlay_insert
    so process_merge never shells out to real ffmpeg. Records call args."""
    import app.pipeline_merge as pipeline_merge_mod

    calls = {"probe": [], "make_rounded_mask": [], "overlay_insert": []}

    def _fake_probe(path):
        calls["probe"].append(path)
        return _FakeInfo()

    def _fake_get_default_target(info):
        return (info.width, info.height, info.fps)

    def _fake_make_rounded_mask(width, height, radius_px, dst_png):
        calls["make_rounded_mask"].append(
            {"width": width, "height": height, "radius_px": radius_px, "dst_png": dst_png}
        )
        os.makedirs(os.path.dirname(dst_png), exist_ok=True)
        with open(dst_png, "wb") as fh:
            fh.write(b"fake-mask-png")

    def _fake_overlay_insert(
        base, insert, dst, *, x, y, start_sec, end_sec, width, height, fps, mask_png=None
    ):
        calls["overlay_insert"].append(
            {
                "base": base, "insert": insert, "dst": dst,
                "x": x, "y": y, "start_sec": start_sec, "end_sec": end_sec,
                "width": width, "height": height, "fps": fps, "mask_png": mask_png,
            }
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(b"fake-merged-video-bytes")

    monkeypatch.setattr(pipeline_merge_mod.media_mod, "probe", _fake_probe)
    monkeypatch.setattr(
        pipeline_merge_mod.media_mod, "get_default_target", _fake_get_default_target
    )
    monkeypatch.setattr(
        pipeline_merge_mod.media_mod, "make_rounded_mask", _fake_make_rounded_mask
    )
    monkeypatch.setattr(
        pipeline_merge_mod.media_mod, "overlay_insert", _fake_overlay_insert
    )
    return calls


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestProcessMergeHappyPath:
    def test_happy_path_no_radius(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run, corner_radius_px=0)

        gdrive = FakeGDriveClient()
        process_merge(merge.id, gdrive=gdrive)

        db_session.refresh(merge)
        assert merge.status == MergeStatus.done
        assert merge.output_path is not None and os.path.exists(merge.output_path)
        assert merge.error_message is None

        assert len(media_stubs["make_rounded_mask"]) == 0  # radius=0 -> no mask
        assert len(media_stubs["overlay_insert"]) == 1
        call = media_stubs["overlay_insert"][0]
        assert call["base"] == target.source_local_path
        assert call["insert"] == run.result_local_path
        assert call["x"] == 10 and call["y"] == 20
        assert call["width"] == 100 and call["height"] == 80
        assert call["start_sec"] == 1.0 and call["end_sec"] == 5.0
        assert call["fps"] == 30.0
        assert call["mask_png"] is None

        # No GDRIVE_DEFAULT_FOLDER_ID configured -> no delivery attempted.
        assert gdrive.upload_calls == []
        assert merge.gdrive_url is None

    def test_happy_path_with_radius_makes_mask(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run, corner_radius_px=12)

        process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.done
        assert len(media_stubs["make_rounded_mask"]) == 1
        mask_call = media_stubs["make_rounded_mask"][0]
        assert mask_call["width"] == 100
        assert mask_call["height"] == 80
        assert mask_call["radius_px"] == 12

        overlay_call = media_stubs["overlay_insert"][0]
        assert overlay_call["mask_png"] == mask_call["dst_png"]
        assert overlay_call["mask_png"] is not None

    def test_delivery_uploads_and_sets_gdrive_url(self, db_session, media_stubs, monkeypatch):
        from app.pipeline_merge import process_merge

        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-xyz")

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run)

        gdrive = FakeGDriveClient()
        process_merge(merge.id, gdrive=gdrive)

        db_session.refresh(merge)
        assert merge.status == MergeStatus.done
        assert len(gdrive.upload_calls) == 1
        assert gdrive.upload_calls[0]["folder_id"] == "folder-xyz"
        assert merge.gdrive_url is not None
        assert merge.gdrive_url.startswith("https://drive.google.com/")


# ---------------------------------------------------------------------------
# Orphan resume
# ---------------------------------------------------------------------------


class TestProcessMergeOrphanResume:
    def test_resumes_from_non_queued_status(self, db_session, media_stubs):
        """A merge left in 'compositing' (e.g. worker crash) is reset to queued
        and reprocessed to completion rather than raising InvalidTransition."""
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run, status=MergeStatus.compositing)

        process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.done

    def test_resumes_from_delivering_status(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run, status=MergeStatus.delivering)

        process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.done


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestProcessMergeFailures:
    def test_missing_run_result_marks_failed(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session, no_result=True)
        merge = _make_merge(db_session, target, run)

        with pytest.raises(ValueError):
            process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed
        assert merge.error_message is not None
        assert "no result video" in merge.error_message

    def test_run_not_done_marks_failed(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session, status=RunStatus.processing)
        merge = _make_merge(db_session, target, run)

        with pytest.raises(ValueError):
            process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed
        assert "not done" in merge.error_message

    def test_missing_target_source_marks_failed(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session)
        target.source_local_path = "/nonexistent/path/source.mp4"
        db_session.commit()
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run)

        with pytest.raises(ValueError):
            process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed
        assert "no source" in merge.error_message

    def test_target_not_probed_marks_failed(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        target = _make_target_project(db_session, width=None, height=None, fps=None)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run)

        with pytest.raises(ValueError):
            process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed
        assert "not been" in merge.error_message

    def test_media_error_during_overlay_marks_failed(self, db_session, monkeypatch):
        from app.media import MediaError
        import app.pipeline_merge as pipeline_merge_mod
        from app.pipeline_merge import process_merge

        def _fake_probe(path):
            return _FakeInfo()

        def _fake_get_default_target(info):
            return (info.width, info.height, info.fps)

        def _raise_overlay(*a, **kw):
            raise MediaError("ffmpeg exploded")

        monkeypatch.setattr(pipeline_merge_mod.media_mod, "probe", _fake_probe)
        monkeypatch.setattr(
            pipeline_merge_mod.media_mod, "get_default_target", _fake_get_default_target
        )
        monkeypatch.setattr(pipeline_merge_mod.media_mod, "overlay_insert", _raise_overlay)

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run)

        with pytest.raises(MediaError):
            process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed
        assert "ffmpeg exploded" in merge.error_message

    def test_delivery_failure_marks_failed_but_output_path_kept(
        self, db_session, media_stubs, monkeypatch
    ):
        """A Drive upload failure (after all retries) fails the merge but the
        composited file stays on disk and output_path stays set — a manual
        retry can re-deliver without re-compositing."""
        import app.pipeline_merge as pipeline_merge_mod
        from app.pipeline_merge import process_merge

        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-xyz")
        monkeypatch.setattr(pipeline_merge_mod, "MERGE_DELIVER_BACKOFF_SEC", 0.0)

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run)

        with pytest.raises(RuntimeError):
            process_merge(merge.id, gdrive=AlwaysFailGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed
        assert merge.error_message is not None
        assert merge.output_path is not None and os.path.exists(merge.output_path)
        assert merge.gdrive_url is None

    def test_missing_merge_raises(self, db_session, media_stubs):
        from app.pipeline_merge import process_merge

        with pytest.raises(ValueError):
            process_merge("does-not-exist", gdrive=FakeGDriveClient())

    def test_error_message_is_truncated(self, db_session, monkeypatch):
        import app.pipeline_merge as pipeline_merge_mod
        from app.pipeline_merge import process_merge

        def _fake_probe(path):
            return _FakeInfo()

        def _fake_get_default_target(info):
            return (info.width, info.height, info.fps)

        def _raise_overlay(*a, **kw):
            raise RuntimeError("x" * 5000)

        monkeypatch.setattr(pipeline_merge_mod.media_mod, "probe", _fake_probe)
        monkeypatch.setattr(
            pipeline_merge_mod.media_mod, "get_default_target", _fake_get_default_target
        )
        monkeypatch.setattr(pipeline_merge_mod.media_mod, "overlay_insert", _raise_overlay)

        target = _make_target_project(db_session)
        run = _make_done_child_run(db_session)
        merge = _make_merge(db_session, target, run)

        with pytest.raises(RuntimeError):
            process_merge(merge.id, gdrive=FakeGDriveClient())

        db_session.refresh(merge)
        assert merge.status == MergeStatus.failed
        assert len(merge.error_message) < 5000
        assert merge.error_message.endswith("(truncated)")
