"""
Tests for app/pipeline_tail.py — process_tail_edit (the "Change ending" tool).

Strategy
--------
* Own SQLite DB engine (file-based, isolated from other test modules).
  pipeline_tail's get_session() is monkeypatched to use this engine (same
  pattern as test_pipeline_merge.py).
* REAL ffmpeg — a done Run's result and a LibraryClip are both small synthetic
  videos generated with ffmpeg lavfi sources (testsrc + sine wave), same as
  tests/test_media.py. process_tail_edit's cut_clip/stitch calls are NOT
  stubbed, so this exercises the real ffmpeg pipeline end to end and asserts
  on the real output duration.
* A FakeGDriveClient records upload_file calls and returns a predictable id.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from typing import Optional

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_tail_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ffmpeg_available() -> bool:
    try:
        r = subprocess.run(
            ["/usr/bin/ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False


if not _ffmpeg_available():
    pytest.skip("ffmpeg not available", allow_module_level=True)

from sqlalchemy import create_engine, event  # noqa: E402 — must follow guard
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.media import probe  # noqa: E402
from app.models import LibraryClip, Run, RunCaption, RunTailEdit, VideoProject  # noqa: E402
from app.state_machine import (  # noqa: E402
    CaptionStatus, ProjectStatus, RunStatus, TailEditStatus,
)
from app.storage import caption_dir, run_results_dir  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    db_path = os.path.join(_TMP_DIR, "pipeline_tail_test.db")
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
    """Replace app.pipeline_tail's get_session with one bound to the test engine."""
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

    import app.pipeline_tail as pipeline_tail_mod
    monkeypatch.setattr(pipeline_tail_mod, "get_session", _test_get_session)


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
# Synthetic video helper (mirrors tests/test_media.py's _make_video)
# ---------------------------------------------------------------------------


def _make_video(path: str, duration: float, *, width: int = 320, height: int = 240,
                 fps: int = 10) -> None:
    """Generate a synthetic test video at *path* using ffmpeg lavfi sources."""
    cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not create test video: {result.stderr.decode()[-500:]}"
        )


# ---------------------------------------------------------------------------
# DB fixture builders
# ---------------------------------------------------------------------------


def _make_done_run(session, *, duration: float = 10.0, no_result: bool = False,
                    **kwargs) -> Run:
    project = VideoProject(
        source_type="upload", source_ref="src.mp4", status=ProjectStatus.ready
    )
    session.add(project)
    session.flush()

    defaults = dict(
        project_id=project.id,
        prompt="swap the character",
        status=RunStatus.done,
    )
    defaults.update(kwargs)
    run = Run(**defaults)
    session.add(run)
    session.flush()

    if not no_result:
        result_dir = run_results_dir(run.id, project.id)
        result_path = os.path.join(result_dir, "final.mp4")
        _make_video(result_path, duration)
        run.result_local_path = result_path

    session.commit()
    return run


def _make_library_clip(session, tmp_path, *, duration: float = 3.0,
                        no_file: bool = False, **kwargs) -> LibraryClip:
    if no_file:
        local_path = str(tmp_path / "missing_clip.mp4")
    else:
        local_path = str(tmp_path / f"clip_{uuid.uuid4().hex[:8]}.mp4")
        _make_video(local_path, duration)
    defaults = dict(name="Outro EN", local_path=local_path, duration_sec=duration)
    defaults.update(kwargs)
    clip = LibraryClip(**defaults)
    session.add(clip)
    session.commit()
    return clip


def _make_tail_edit(session, run: Run, **kwargs) -> RunTailEdit:
    defaults = dict(
        run_id=run.id,
        cut_sec=5.0,
        status=TailEditStatus.queued,
    )
    defaults.update(kwargs)
    edit = RunTailEdit(**defaults)
    session.add(edit)
    session.commit()
    return edit


def _make_caption(session, run: Run, *, output_duration: Optional[float] = None,
                   **kwargs) -> RunCaption:
    """Build a RunCaption row (defaults to status=done) usable as a chained
    tail-edit source (docs/burned-captions.md §3.7).

    When *output_duration* is given (and *kwargs* doesn't already set
    ``output_path``), a real synthetic video of that duration is written and
    wired up as ``output_path`` — this lets a test give the caption a
    duration DIFFERENT from the parent run's result, to prove that a chained
    tail edit probes and cuts the resolved caption source, never the run's
    result.
    """
    defaults = dict(run_id=run.id, status=CaptionStatus.done)
    defaults.update(kwargs)
    caption = RunCaption(**defaults)
    session.add(caption)
    session.flush()

    if output_duration is not None and "output_path" not in kwargs:
        out_dir = caption_dir(caption.id, run.project_id)
        output_path = os.path.join(out_dir, f"{run.id}_captioned.mp4")
        _make_video(output_path, output_duration)
        caption.output_path = output_path

    session.commit()
    return caption


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestProcessTailEditHappyPath:
    def test_happy_path_with_appended_clip(self, db_session, tmp_path):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=10.0)
        clip = _make_library_clip(db_session, tmp_path, duration=3.0)
        edit = _make_tail_edit(db_session, run, cut_sec=6.0, library_clip_id=clip.id)

        gdrive = FakeGDriveClient()
        process_tail_edit(edit.id, gdrive=gdrive)

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        assert edit.error_message is None
        assert edit.output_path is not None and os.path.exists(edit.output_path)

        out_duration = probe(edit.output_path).duration_sec
        assert out_duration == pytest.approx(6.0 + 3.0, abs=1.0)
        # The body.mp4 intermediate is dropped once the output is stitched.
        assert not os.path.exists(
            os.path.join(os.path.dirname(edit.output_path), "body.mp4")
        )

        # No GDRIVE_DEFAULT_FOLDER_ID configured and run has no gdrive_folder_id
        # -> no delivery attempted.
        assert gdrive.upload_calls == []
        assert edit.gdrive_file_id is None

    def test_trim_only_path(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=10.0)
        edit = _make_tail_edit(db_session, run, cut_sec=4.0, library_clip_id=None)

        process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        assert edit.output_path is not None and os.path.exists(edit.output_path)

        out_duration = probe(edit.output_path).duration_sec
        assert out_duration == pytest.approx(4.0, abs=1.0)

    def test_output_is_size_capped_before_delivery(self, db_session, monkeypatch):
        """The stitched output is both encoded under the cap and guarded by
        enforce_size_limit() before it reaches Drive."""
        import app.media as media_mod
        from app.pipeline_tail import process_tail_edit

        encodes: list[list[str]] = []
        real_run = media_mod._run

        def _spy_run(cmd, **kwargs):
            if cmd[0] == media_mod.FFMPEG:
                encodes.append(cmd)
            return real_run(cmd, **kwargs)

        guarded: list[str] = []
        real_enforce = media_mod.enforce_size_limit

        def _spy_enforce(path, **kwargs):
            guarded.append(path)
            return real_enforce(path, **kwargs)

        monkeypatch.setattr(media_mod, "_run", _spy_run)
        monkeypatch.setattr(media_mod, "enforce_size_limit", _spy_enforce)

        run = _make_done_run(db_session, duration=10.0)
        edit = _make_tail_edit(db_session, run, cut_sec=4.0, library_clip_id=None)

        process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        assert guarded == [edit.output_path]

        # The stitch (last ffmpeg call) carries a VBV cap sized from the ~4s
        # output, not an unbounded CRF encode.
        stitch_cmd = encodes[-1]
        # The stitch writes a sibling temp file that atomically replaces the
        # output (AICP-299 §3.1).
        root = os.path.splitext(edit.output_path)[0]
        assert any(a.startswith(root + ".") and a.endswith(".part.mp4") for a in stitch_cmd)
        expected = media_mod.delivery_video_bitrate(
            4.0, audio_bitrate_bps=media_mod._STITCH_AUDIO_BPS
        )
        assert int(stitch_cmd[stitch_cmd.index("-maxrate") + 1]) == pytest.approx(
            expected, rel=0.1
        )

    def test_delivery_uses_run_folder_over_default(self, db_session, monkeypatch):
        from app.pipeline_tail import process_tail_edit

        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "default-folder")

        run = _make_done_run(db_session, duration=6.0, gdrive_folder_id="run-folder")
        edit = _make_tail_edit(db_session, run, cut_sec=3.0)

        gdrive = FakeGDriveClient()
        process_tail_edit(edit.id, gdrive=gdrive)

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        assert len(gdrive.upload_calls) == 1
        assert gdrive.upload_calls[0]["folder_id"] == "run-folder"
        assert edit.gdrive_file_id is not None

    def test_delivery_falls_back_to_default_folder(self, db_session, monkeypatch):
        from app.pipeline_tail import process_tail_edit

        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "default-folder")

        run = _make_done_run(db_session, duration=6.0)
        edit = _make_tail_edit(db_session, run, cut_sec=3.0)

        gdrive = FakeGDriveClient()
        process_tail_edit(edit.id, gdrive=gdrive)

        db_session.refresh(edit)
        assert len(gdrive.upload_calls) == 1
        assert gdrive.upload_calls[0]["folder_id"] == "default-folder"
        assert gdrive.upload_calls[0]["name"] == f"{run.id}_trim.mp4"


# ---------------------------------------------------------------------------
# Orphan resume
# ---------------------------------------------------------------------------


class TestProcessTailEditOrphanResume:
    def test_resumes_from_processing_status(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        edit = _make_tail_edit(db_session, run, cut_sec=2.0, status=TailEditStatus.processing)

        process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done

    def test_resumes_from_delivering_status(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        edit = _make_tail_edit(db_session, run, cut_sec=2.0, status=TailEditStatus.delivering)

        process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestProcessTailEditFailures:
    def test_invalid_cut_sec_too_large_marks_failed(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        edit = _make_tail_edit(db_session, run, cut_sec=999.0)

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert edit.error_message is not None
        assert "cut_sec" in edit.error_message

    def test_invalid_cut_sec_too_small_marks_failed(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        edit = _make_tail_edit(db_session, run, cut_sec=0.001)

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert "cut_sec" in edit.error_message

    def test_missing_library_clip_marks_failed(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        edit = _make_tail_edit(
            db_session, run, cut_sec=2.0, library_clip_id="does-not-exist"
        )

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert "not found" in edit.error_message

    def test_clip_file_missing_on_disk_marks_failed(self, db_session, tmp_path):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        clip = _make_library_clip(db_session, tmp_path, no_file=True)
        edit = _make_tail_edit(db_session, run, cut_sec=2.0, library_clip_id=clip.id)

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert "no file on disk" in edit.error_message

    def test_missing_run_result_marks_failed(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, no_result=True)
        edit = _make_tail_edit(db_session, run, cut_sec=2.0)

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert "no result video" in edit.error_message

    def test_delivery_failure_marks_failed_but_output_path_kept(
        self, db_session, monkeypatch
    ):
        """A Drive upload failure (after all retries) fails the tail edit but
        the encoded file stays on disk and output_path stays set — a manual
        retry can re-deliver without re-encoding."""
        import app.pipeline_tail as pipeline_tail_mod
        from app.pipeline_tail import process_tail_edit

        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-xyz")
        monkeypatch.setattr(pipeline_tail_mod, "TAIL_EDIT_DELIVER_BACKOFF_SEC", 0.0)

        run = _make_done_run(db_session, duration=5.0)
        edit = _make_tail_edit(db_session, run, cut_sec=2.0)

        with pytest.raises(RuntimeError):
            process_tail_edit(edit.id, gdrive=AlwaysFailGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert edit.error_message is not None
        assert edit.output_path is not None and os.path.exists(edit.output_path)
        assert edit.gdrive_file_id is None

    def test_missing_edit_raises(self, db_session):
        from app.pipeline_tail import process_tail_edit

        with pytest.raises(ValueError):
            process_tail_edit("does-not-exist", gdrive=FakeGDriveClient())


# ---------------------------------------------------------------------------
# Chaining — tail edit sourced from a caption (docs/burned-captions.md §3.7)
# ---------------------------------------------------------------------------


class TestProcessTailEditChainedFromCaption:
    def test_happy_path_probes_resolved_source_not_run_result(self, db_session):
        """The run's result is short (4s); the caption source is long (10s).
        cut_sec=6.0 is only valid against the CAPTION's duration — it would
        fail the cut_sec bounds check if the run's 4s result were probed
        instead. A successful, correctly-sized cut proves _resolve_source's
        caption branch (not run.result_local_path) is what gets probed and
        fed to cut_clip/stitch."""
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=4.0)
        caption = _make_caption(db_session, run, output_duration=10.0)
        edit = _make_tail_edit(
            db_session, run, cut_sec=6.0,
            source_type="caption", source_caption_id=caption.id,
        )

        process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        assert edit.error_message is None
        assert edit.output_path is not None and os.path.exists(edit.output_path)

        out_duration = probe(edit.output_path).duration_sec
        assert out_duration == pytest.approx(6.0, abs=1.0)

    def test_chained_filename_trim_only(self, db_session, monkeypatch):
        """{run_id}_captioned_trim.mp4 — the trim-only variant keeps the
        existing `_trim` suffix convention, with a `captioned_` segment
        inserted so it cannot collide with an unchained trim's filename."""
        from app.pipeline_tail import process_tail_edit

        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "default-folder")

        run = _make_done_run(db_session, duration=4.0)
        caption = _make_caption(db_session, run, output_duration=10.0)
        edit = _make_tail_edit(
            db_session, run, cut_sec=6.0,
            source_type="caption", source_caption_id=caption.id,
        )

        gdrive = FakeGDriveClient()
        process_tail_edit(edit.id, gdrive=gdrive)

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        expected_name = f"{run.id}_captioned_trim.mp4"
        assert os.path.basename(edit.output_path) == expected_name
        assert gdrive.upload_calls[0]["name"] == expected_name

    def test_chained_filename_with_appended_clip(self, db_session, tmp_path):
        """{run_id}_captioned_{clip_name}.mp4 when a library clip is appended."""
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=4.0)
        caption = _make_caption(db_session, run, output_duration=10.0)
        clip = _make_library_clip(
            db_session, tmp_path, duration=2.0, name="Outro EN"
        )
        edit = _make_tail_edit(
            db_session, run, cut_sec=6.0, library_clip_id=clip.id,
            source_type="caption", source_caption_id=caption.id,
        )

        process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        assert os.path.basename(edit.output_path) == f"{run.id}_captioned_Outro_EN.mp4"

        out_duration = probe(edit.output_path).duration_sec
        assert out_duration == pytest.approx(6.0 + 2.0, abs=1.0)

    def test_source_caption_missing_marks_failed(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        edit = _make_tail_edit(
            db_session, run, cut_sec=2.0,
            source_type="caption", source_caption_id="does-not-exist",
        )

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert edit.error_message is not None
        assert "not found" in edit.error_message

    def test_source_caption_not_done_marks_failed(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        caption = _make_caption(db_session, run, status=CaptionStatus.burning)
        edit = _make_tail_edit(
            db_session, run, cut_sec=2.0,
            source_type="caption", source_caption_id=caption.id,
        )

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert "is not done" in edit.error_message

    def test_source_caption_output_missing_on_disk_marks_failed(self, db_session):
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=5.0)
        caption = _make_caption(
            db_session, run, output_path="/nonexistent/captioned.mp4"
        )
        edit = _make_tail_edit(
            db_session, run, cut_sec=2.0,
            source_type="caption", source_caption_id=caption.id,
        )

        with pytest.raises(ValueError):
            process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.failed
        assert "no output video on disk" in edit.error_message

    def test_explicit_source_type_run_is_byte_for_byte_unchanged(self, db_session):
        """source_type='run' set EXPLICITLY behaves exactly like the
        (implicit-default) unchained path already covered by
        TestProcessTailEditHappyPath — reads run.result_local_path."""
        from app.pipeline_tail import process_tail_edit

        run = _make_done_run(db_session, duration=10.0)
        edit = _make_tail_edit(
            db_session, run, cut_sec=4.0, library_clip_id=None, source_type="run",
        )

        process_tail_edit(edit.id, gdrive=FakeGDriveClient())

        db_session.refresh(edit)
        assert edit.status == TailEditStatus.done
        assert os.path.basename(edit.output_path) == f"{run.id}_trim.mp4"

        out_duration = probe(edit.output_path).duration_sec
        assert out_duration == pytest.approx(4.0, abs=1.0)
