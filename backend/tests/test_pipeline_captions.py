"""
Tests for app/pipeline_captions.py — process_caption (the "Add Captions" tool).

Strategy
--------
* Own SQLite DB engine (file-based, isolated from other test modules).
  pipeline_captions's get_session() is monkeypatched to use this engine (same
  pattern as test_pipeline_tail.py / test_pipeline_merge.py).
* Media (app.media.probe / burn_subtitles / enforce_size_limit) and the LLM
  transcription call (app.localisation.transcribe_video) are ALL mocked —
  unlike test_pipeline_tail.py, which runs real ffmpeg on real synthetic
  clips. This pipeline's two "heavy" calls are a video-model network round
  trip (transcribe) and an ffmpeg burn-in whose correctness is already
  covered by tests/test_captions.py (build_ass, pure) and tests/test_media.py
  (burn_subtitles, real ffmpeg) — this module only needs to prove it wires
  those calls together and drives the state machine correctly, so nothing
  here depends on ffmpeg being installed.
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

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_captions_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import captions as captions_mod  # noqa: E402
from app import media as media_mod  # noqa: E402
from app import localisation as localisation_mod  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.models import Run, RunCaption, RunTailEdit, VideoProject  # noqa: E402
from app.state_machine import (  # noqa: E402
    CaptionStatus, ProjectStatus, RunStatus, TailEditStatus,
)
from app.storage import caption_dir, run_results_dir, tail_edit_dir  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    db_path = os.path.join(_TMP_DIR, "pipeline_captions_test.db")
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
    """Replace app.pipeline_captions's get_session with one bound to the test engine."""
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

    import app.pipeline_captions as pipeline_captions_mod
    monkeypatch.setattr(pipeline_captions_mod, "get_session", _test_get_session)


@pytest.fixture(autouse=True)
def _no_default_folder(monkeypatch):
    """Default to no Drive delivery unless a test explicitly opts in."""
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "")


@pytest.fixture(autouse=True)
def _no_audio_fingerprint(monkeypatch):
    """The placeholder "videos" here have no audio to decode; default to no
    fingerprint (= no transcript reuse) unless a test sets one."""
    monkeypatch.setattr(media_mod, "audio_fingerprint", lambda path: None)
    # Rows other tests leave behind must never make a job really sleep; the
    # wait tests opt back in.
    import app.pipeline_captions as pc
    monkeypatch.setattr(pc, "CAPTION_DONOR_WAIT_SEC", 0.0)
    monkeypatch.setattr(pc, "CAPTION_UNFINGERPRINTED_WAIT_SEC", 0.0)


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
    """Always raises; records how many times upload_file was attempted."""

    def __init__(self):
        self.attempts = 0

    def upload_file(self, *a, **kw):
        self.attempts += 1
        raise RuntimeError("Drive is down")


# ---------------------------------------------------------------------------
# Media / transcribe mocking helpers
# ---------------------------------------------------------------------------


def _patch_probe(monkeypatch, *, duration_sec=12.5, width=1080, height=1920,
                  has_audio=True, fps=30.0):
    """Monkeypatch media.probe to return a fixed MediaInfo, regardless of path.

    The "delivered file" in these tests is a placeholder text file (see
    _make_done_run) — not a real video — so a real ffprobe call would fail.
    process_caption calls media_mod.probe exactly once; asserting that stays
    true is covered indirectly (a second, different-path call would still get
    the same canned info, which is all any of these tests need).
    """
    info = media_mod.MediaInfo(
        duration_sec=duration_sec, width=width, height=height, fps=fps,
        aspect_ratio="9:16", has_audio=has_audio,
    )
    monkeypatch.setattr(media_mod, "probe", lambda path: info)
    return info


def _patch_burn_subtitles(monkeypatch):
    """Monkeypatch media.burn_subtitles to write a placeholder file instead of
    shelling out to ffmpeg. Returns the list of call kwargs recorded."""
    calls: list[dict] = []

    def _fake_burn(src, ass_path, dst, *, duration_sec=None, has_audio=None):
        calls.append(
            {
                "src": src, "ass_path": ass_path, "dst": dst,
                "duration_sec": duration_sec, "has_audio": has_audio,
            }
        )
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "wb") as f:
            f.write(b"fake burned-in video")

    monkeypatch.setattr(media_mod, "burn_subtitles", _fake_burn)
    return calls


def _fake_transcribe(*, lines=None, source_language="en"):
    """Build a fake localisation.transcribe_video replacement."""
    calls: list[str] = []

    def _transcribe(video_path, *, kie=None, model=None, spend_scope=None):
        calls.append(video_path)
        return {
            "schema_version": 1,
            "model": "fake-model",
            "prompt_version": "transcribe/v1",
            "created_at": "2026-08-09T00:00:00+00:00",
            "source_language": source_language,
            "lines": lines if lines is not None else [],
            "on_screen_text": [],
        }

    _transcribe.calls = calls
    return _transcribe


def _boom_transcribe(*a, **kw):
    raise AssertionError("transcribe_video must not be called for manual lines")


# ---------------------------------------------------------------------------
# DB fixture builders
# ---------------------------------------------------------------------------


def _make_done_run(session, *, no_result: bool = False, **kwargs) -> Run:
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
        with open(result_path, "wb") as f:
            f.write(b"not a real video, just needs to exist on disk")
        run.result_local_path = result_path

    session.commit()
    return run


def _make_caption(session, run: Run, **kwargs) -> RunCaption:
    defaults = dict(run_id=run.id, status=CaptionStatus.queued)
    defaults.update(kwargs)
    caption = RunCaption(**defaults)
    session.add(caption)
    session.commit()
    return caption


def _make_tail_edit(session, run: Run, **kwargs) -> RunTailEdit:
    """Build a RunTailEdit row (defaults to status=done) usable as a chained
    caption source (docs/burned-captions.md §3.7). Mirrors _make_done_run's
    placeholder-file approach — this test module mocks media.probe, so the
    "video" just needs to exist on disk, not decode as one.

    Unless *kwargs* already sets ``output_path``, a placeholder file named
    ``{run_id}_trim.mp4`` (the real pipeline_tail._output_filename trim-only
    convention) is created and wired up.
    """
    defaults = dict(run_id=run.id, cut_sec=1.0, status=TailEditStatus.done)
    defaults.update(kwargs)
    edit = RunTailEdit(**defaults)
    session.add(edit)
    session.flush()

    if "output_path" not in kwargs:
        out_dir = tail_edit_dir(edit.id, run.project_id)
        output_path = os.path.join(out_dir, f"{run.id}_trim.mp4")
        with open(output_path, "wb") as f:
            f.write(b"not a real video, just needs to exist on disk")
        edit.output_path = output_path

    session.commit()
    return edit


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestProcessCaptionHappyPath:
    def test_happy_path_with_transcription(self, db_session, monkeypatch):
        import app.pipeline_captions as pipeline_captions_mod
        from app.pipeline_captions import process_caption

        info = _patch_probe(monkeypatch, duration_sec=12.5, width=1080, height=1920)
        burn_calls = _patch_burn_subtitles(monkeypatch)
        lines = [{"start": 0.5, "end": 2.0, "text": "Hello there"}]
        transcribe = _fake_transcribe(lines=lines, source_language="en")
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "default-folder")

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run)

        gdrive = FakeGDriveClient()
        process_caption(caption.id, gdrive=gdrive)

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.error_message is None

        # Transcription happened, over the DELIVERED result — not any source.
        assert transcribe.calls == [run.result_local_path]
        assert caption.lines == lines
        assert caption.lines_source == pipeline_captions_mod.LINES_SOURCE_TRANSCRIBE
        assert caption.language == "en"  # resolved from transcript.source_language
        assert caption.style_version == captions_mod.CAPTION_STYLE_VERSION

        # burn_subtitles got the ONE probe's values, not a re-probe.
        assert len(burn_calls) == 1
        assert burn_calls[0]["duration_sec"] == info.duration_sec
        assert burn_calls[0]["has_audio"] == info.has_audio
        assert burn_calls[0]["src"] == run.result_local_path

        # Output + ass file on disk, in the documented layout.
        assert caption.output_path is not None and os.path.exists(caption.output_path)
        out_dir = caption_dir(caption.id, run.project_id)
        ass_path = os.path.join(out_dir, "captions.ass")
        assert os.path.exists(ass_path)
        with open(ass_path, "r", encoding="utf-8") as f:
            ass_content = f.read()
        assert "Hello there" in ass_content
        assert "Dialogue:" in ass_content

        # Delivered to the default folder under the documented filename.
        assert len(gdrive.upload_calls) == 1
        assert gdrive.upload_calls[0]["folder_id"] == "default-folder"
        assert gdrive.upload_calls[0]["name"] == f"{run.id}_captioned.mp4"
        assert caption.gdrive_file_id is not None

    def test_manual_lines_skip_transcription(self, db_session, monkeypatch):
        import app.pipeline_captions as pipeline_captions_mod
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        monkeypatch.setattr(localisation_mod, "transcribe_video", _boom_transcribe)

        run = _make_done_run(db_session, language="ja")
        manual_lines = [{"start": 0.0, "end": 1.0, "text": "Manually edited"}]
        caption = _make_caption(
            db_session, run,
            lines=manual_lines,
            lines_source=pipeline_captions_mod.LINES_SOURCE_MANUAL,
        )

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        # Untouched: still exactly the operator's manual transcript.
        assert caption.lines == manual_lines
        assert caption.lines_source == pipeline_captions_mod.LINES_SOURCE_MANUAL
        # No caption.language was set, so it falls back to the run's language.
        assert caption.language == "ja"
        assert caption.output_path is not None and os.path.exists(caption.output_path)

    def test_empty_transcript_is_success_not_failure(self, db_session, monkeypatch):
        import app.pipeline_captions as pipeline_captions_mod
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[], source_language="en")
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run)

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.error_message is None
        assert caption.lines == []
        assert caption.lines_source == pipeline_captions_mod.LINES_SOURCE_TRANSCRIBE
        # A caption-free re-encode is still delivered.
        assert caption.output_path is not None and os.path.exists(caption.output_path)

        out_dir = caption_dir(caption.id, run.project_id)
        ass_path = os.path.join(out_dir, "captions.ass")
        with open(ass_path, "r", encoding="utf-8") as f:
            ass_content = f.read()
        assert "Dialogue:" not in ass_content  # header-only, no events

    def test_language_resolution_prefers_existing_caption_language(
        self, db_session, monkeypatch
    ):
        """caption.language, when already set, wins over both run.language and
        the transcript's source_language."""
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[{"start": 0, "end": 1, "text": "x"}],
                                       source_language="ja")
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session, language="de")
        caption = _make_caption(db_session, run, language="es")

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.language == "es"


# ---------------------------------------------------------------------------
# Orphan resume
# ---------------------------------------------------------------------------


class TestProcessCaptionOrphanResume:
    def test_resumes_from_burning_status(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[{"start": 0, "end": 1, "text": "hi"}])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run, status=CaptionStatus.burning)

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        # Orphan resume reset all the way to queued, so transcription re-ran.
        assert transcribe.calls == [run.result_local_path]

    def test_resumes_from_delivering_status(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run, status=CaptionStatus.delivering)

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done

    def test_resumes_from_failed_status(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session)
        caption = _make_caption(
            db_session, run, status=CaptionStatus.failed, error_message="old failure"
        )

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.error_message is None


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestProcessCaptionFailures:
    def test_missing_run_result_marks_failed(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)

        run = _make_done_run(db_session, no_result=True)
        caption = _make_caption(db_session, run)

        with pytest.raises(ValueError):
            process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.failed
        assert caption.error_message is not None
        assert "no result video on disk" in caption.error_message

    def test_result_path_set_but_missing_on_disk_marks_failed(
        self, db_session, monkeypatch
    ):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)

        run = _make_done_run(db_session, no_result=True)
        run.result_local_path = "/nonexistent/path/final.mp4"
        db_session.commit()
        caption = _make_caption(db_session, run)

        with pytest.raises(ValueError):
            process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.failed
        assert "no result video on disk" in caption.error_message

    def test_localisation_error_marks_failed(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)

        def _raise(*a, **kw):
            raise localisation_mod.LocalisationError("transcribe: model exploded")

        monkeypatch.setattr(localisation_mod, "transcribe_video", _raise)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run)

        with pytest.raises(localisation_mod.LocalisationError):
            process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.failed
        assert "model exploded" in caption.error_message

    def test_missing_caption_raises(self, db_session):
        from app.pipeline_captions import process_caption

        with pytest.raises(ValueError):
            process_caption("does-not-exist", gdrive=FakeGDriveClient())

    def test_enforce_size_limit_failure_is_swallowed(self, db_session, monkeypatch):
        """A raising enforce_size_limit must not fail the caption — the burned
        file, even oversized, still beats no deliverable."""
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        def _boom_enforce(path, **kwargs):
            raise media_mod.MediaError("could not shrink")

        monkeypatch.setattr(media_mod, "enforce_size_limit", _boom_enforce)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run)

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.error_message is None
        assert caption.output_path is not None and os.path.exists(caption.output_path)

    def test_delivery_failure_marks_failed_but_output_path_kept(
        self, db_session, monkeypatch
    ):
        """A Drive upload failure (after all retries) fails the caption but the
        encoded file stays on disk and output_path stays set — a manual retry
        can re-deliver without re-transcribing or re-burning."""
        import app.pipeline_captions as pipeline_captions_mod
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-xyz")
        monkeypatch.setattr(pipeline_captions_mod, "CAPTION_DELIVER_BACKOFF_SEC", 0.0)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run)

        gdrive = AlwaysFailGDriveClient()
        with pytest.raises(RuntimeError):
            process_caption(caption.id, gdrive=gdrive)

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.failed
        assert caption.error_message is not None
        assert caption.output_path is not None and os.path.exists(caption.output_path)
        assert caption.gdrive_file_id is None
        # Retried exactly CAPTION_DELIVER_ATTEMPTS times, no more, no fewer.
        assert gdrive.attempts == pipeline_captions_mod.CAPTION_DELIVER_ATTEMPTS

    def test_delivery_retries_then_succeeds(self, db_session, monkeypatch):
        import app.pipeline_captions as pipeline_captions_mod
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-xyz")
        monkeypatch.setattr(pipeline_captions_mod, "CAPTION_DELIVER_BACKOFF_SEC", 0.0)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run)

        gdrive = FakeGDriveClient(fail_times=pipeline_captions_mod.CAPTION_DELIVER_ATTEMPTS - 1)
        process_caption(caption.id, gdrive=gdrive)

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.gdrive_file_id is not None
        assert len(gdrive.upload_calls) == 1


# ---------------------------------------------------------------------------
# Delivery folder resolution
# ---------------------------------------------------------------------------


class TestProcessCaptionDeliveryFolder:
    def test_no_folder_configured_skips_delivery_cleanly(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        # _no_default_folder autouse fixture already blanked
        # settings.GDRIVE_DEFAULT_FOLDER_ID, and the run has no gdrive_folder_id.

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run)

        gdrive = FakeGDriveClient()
        process_caption(caption.id, gdrive=gdrive)

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert gdrive.upload_calls == []
        assert caption.gdrive_file_id is None

    def test_run_folder_takes_precedence_over_default(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "default-folder")

        run = _make_done_run(db_session, gdrive_folder_id="run-folder")
        caption = _make_caption(db_session, run)

        gdrive = FakeGDriveClient()
        process_caption(caption.id, gdrive=gdrive)

        db_session.refresh(caption)
        assert len(gdrive.upload_calls) == 1
        assert gdrive.upload_calls[0]["folder_id"] == "run-folder"


# ---------------------------------------------------------------------------
# Chaining — caption sourced from a tail edit (docs/burned-captions.md §3.7)
# ---------------------------------------------------------------------------


class TestProcessCaptionChainedFromTailEdit:
    def test_happy_path_probes_resolved_source_not_run_result(
        self, db_session, monkeypatch
    ):
        """The run's result and the chained tail-edit source are given
        DIFFERENT durations via a path-aware probe stub. Asserting which
        path/duration reaches probe, transcribe_video and burn_subtitles
        proves the RESOLVED tail-edit source is what gets probed and
        burned — never the run's result."""
        from app.pipeline_captions import process_caption

        run = _make_done_run(db_session)
        edit = _make_tail_edit(db_session, run)

        run_info = media_mod.MediaInfo(
            duration_sec=5.0, width=1080, height=1920, fps=30.0,
            aspect_ratio="9:16", has_audio=True,
        )
        source_info = media_mod.MediaInfo(
            duration_sec=20.0, width=1080, height=1920, fps=30.0,
            aspect_ratio="9:16", has_audio=True,
        )
        probe_calls: list[str] = []

        def _probe(path):
            probe_calls.append(path)
            if path == edit.output_path:
                return source_info
            if path == run.result_local_path:
                return run_info
            raise AssertionError(f"probe called with unexpected path: {path}")

        monkeypatch.setattr(media_mod, "probe", _probe)
        burn_calls = _patch_burn_subtitles(monkeypatch)
        lines = [{"start": 0.5, "end": 2.0, "text": "Hi there"}]
        transcribe = _fake_transcribe(lines=lines, source_language="en")
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        caption = _make_caption(
            db_session, run, source_type="tail_edit", source_tail_edit_id=edit.id,
        )

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.error_message is None

        # Transcribed + probed the RESOLVED tail-edit source, not the run's
        # result — proven by path identity, not just a lucky duration match.
        assert transcribe.calls == [edit.output_path]
        assert probe_calls == [edit.output_path]
        assert burn_calls[0]["src"] == edit.output_path
        # ...and its 20s duration (not the run's 5s) is what reached
        # build_ass/burn_subtitles' sizing/clamping.
        assert burn_calls[0]["duration_sec"] == 20.0

    def test_chained_filename_uses_tail_edit_clip_name_trim(
        self, db_session, monkeypatch
    ):
        """{run_id}_{clip_name}_captioned.mp4 — clip_name recovered from the
        source tail edit's own (trim-only) output filename."""
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "default-folder")

        run = _make_done_run(db_session)
        edit = _make_tail_edit(db_session, run)  # -> {run_id}_trim.mp4
        caption = _make_caption(
            db_session, run, source_type="tail_edit", source_tail_edit_id=edit.id,
        )

        gdrive = FakeGDriveClient()
        process_caption(caption.id, gdrive=gdrive)

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        expected_name = f"{run.id}_trim_captioned.mp4"
        assert os.path.basename(caption.output_path) == expected_name
        assert gdrive.upload_calls[0]["name"] == expected_name

    def test_chained_filename_uses_tail_edit_clip_name_with_clip(
        self, db_session, monkeypatch
    ):
        """Same, but the source tail edit appended a named library clip —
        the caption's filename carries that clip name through."""
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session)
        edit = RunTailEdit(run_id=run.id, cut_sec=1.0, status=TailEditStatus.done)
        db_session.add(edit)
        db_session.flush()
        out_dir = tail_edit_dir(edit.id, run.project_id)
        output_path = os.path.join(out_dir, f"{run.id}_Outro_EN.mp4")
        with open(output_path, "wb") as f:
            f.write(b"not a real video, just needs to exist on disk")
        edit.output_path = output_path
        db_session.commit()

        caption = _make_caption(
            db_session, run, source_type="tail_edit", source_tail_edit_id=edit.id,
        )

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert (
            os.path.basename(caption.output_path)
            == f"{run.id}_Outro_EN_captioned.mp4"
        )

    def test_source_tail_edit_missing_marks_failed(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)

        run = _make_done_run(db_session)
        caption = _make_caption(
            db_session, run,
            source_type="tail_edit", source_tail_edit_id="does-not-exist",
        )

        with pytest.raises(ValueError):
            process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.failed
        assert caption.error_message is not None
        assert "not found" in caption.error_message

    def test_source_tail_edit_not_done_marks_failed(self, db_session, monkeypatch):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)

        run = _make_done_run(db_session)
        edit = _make_tail_edit(db_session, run, status=TailEditStatus.processing)
        caption = _make_caption(
            db_session, run, source_type="tail_edit", source_tail_edit_id=edit.id,
        )

        with pytest.raises(ValueError):
            process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.failed
        assert "is not done" in caption.error_message

    def test_source_tail_edit_output_missing_on_disk_marks_failed(
        self, db_session, monkeypatch
    ):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)

        run = _make_done_run(db_session)
        edit = _make_tail_edit(db_session, run, output_path="/nonexistent/trim.mp4")
        caption = _make_caption(
            db_session, run, source_type="tail_edit", source_tail_edit_id=edit.id,
        )

        with pytest.raises(ValueError):
            process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.failed
        assert "no output video on disk" in caption.error_message

    def test_explicit_source_type_run_is_byte_for_byte_unchanged(
        self, db_session, monkeypatch
    ):
        """source_type='run' set EXPLICITLY behaves exactly like the
        (implicit-default) unchained path already covered by
        TestProcessCaptionHappyPath — reads run.result_local_path and keeps
        the plain {run_id}_captioned.mp4 filename."""
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch, duration_sec=12.5)
        burn_calls = _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(lines=[{"start": 0, "end": 1, "text": "hi"}])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session)
        caption = _make_caption(db_session, run, source_type="run")

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert transcribe.calls == [run.result_local_path]
        assert burn_calls[0]["src"] == run.result_local_path
        assert os.path.basename(caption.output_path) == f"{run.id}_captioned.mp4"


# ---------------------------------------------------------------------------
# Italian (it) — caption language override and inheritance
# ---------------------------------------------------------------------------


class TestItalianCaptionLanguage:
    """The existing three-step resolution (caption.language or run.language or
    transcript.source_language) with Italian at each step.

    Transcription is stubbed throughout — no paid provider call.
    """

    def test_explicit_italian_wins_over_the_runs_language(
        self, db_session, monkeypatch
    ):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(
            lines=[{"start": 0, "end": 1, "text": "Perché è già così?"}],
            source_language="en",
        )
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session, language="de")
        caption = _make_caption(db_session, run, language="it")

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.language == "it"

    def test_italian_is_inherited_from_the_run_when_unset(
        self, db_session, monkeypatch
    ):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(
            lines=[{"start": 0, "end": 1, "text": "Un po' di più."}],
            source_language="en",
        )
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session, language="it")
        caption = _make_caption(db_session, run)

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.language == "it"

    def test_italian_falls_through_to_the_detected_source_language(
        self, db_session, monkeypatch
    ):
        """Neither the caption nor the run says anything — the transcript does."""
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(
            lines=[{"start": 0, "end": 1, "text": "Perché è già così?"}],
            source_language="it",
        )
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session)
        assert run.language is None
        caption = _make_caption(db_session, run)

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        assert caption.language == "it"

    def test_italian_captions_render_with_accents_in_the_ass_file(
        self, db_session, monkeypatch
    ):
        """The burned .ass must carry the accented text, not mojibake."""
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        transcribe = _fake_transcribe(
            lines=[{"start": 0.0, "end": 2.0, "text": "Perché è già così?"},
                   {"start": 2.0, "end": 4.0, "text": "Un po' di più."}],
            source_language="it",
        )
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        run = _make_done_run(db_session, language="it")
        caption = _make_caption(db_session, run)

        process_caption(caption.id, gdrive=FakeGDriveClient())

        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done

        out_dir = caption_dir(caption.id, run.project_id)
        with open(os.path.join(out_dir, "captions.ass"), "r", encoding="utf-8") as f:
            ass = f.read()
        assert "Perché" in ass
        assert "è già così?" in ass
        assert "po'" in ass
        # Latin script -> the default font, not the CJK one.
        from app.captions import CJK_FONT, DEFAULT_FONT
        assert f"Style: Default,{DEFAULT_FONT}," in ass
        assert CJK_FONT not in ass


# ---------------------------------------------------------------------------
# Transcript reuse across captions with the same audio
# ---------------------------------------------------------------------------


class TestTranscriptReuse:
    # The module shares one DB across tests, so every test gets its own
    # fingerprint — otherwise an earlier test's caption becomes a donor.
    @pytest.fixture(autouse=True)
    def _fresh_fingerprint(self):
        self.FP = uuid.uuid4().hex * 2

    def _setup(self, monkeypatch, fp="own"):
        fp = self.FP if fp == "own" else fp
        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        monkeypatch.setattr(media_mod, "audio_fingerprint", lambda path: fp)

    def test_second_caption_reuses_first_transcript(self, db_session, monkeypatch):
        import app.pipeline_captions as pc
        self._setup(monkeypatch)
        lines = [{"start": 0.0, "end": 2.0, "text": "Did you know"}]
        transcribe = _fake_transcribe(lines=lines)
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)

        first = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(first.id, gdrive=FakeGDriveClient())
        second = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(second.id, gdrive=FakeGDriveClient())

        db_session.refresh(first)
        db_session.refresh(second)
        assert len(transcribe.calls) == 1
        assert first.audio_sha256 == self.FP
        assert first.lines_source == pc.LINES_SOURCE_TRANSCRIBE
        assert second.status == CaptionStatus.done
        assert second.lines == lines
        assert second.lines_source == pc.LINES_SOURCE_REUSED
        assert second.lines_reused_from == first.id
        assert second.audio_sha256 == self.FP
        assert second.language == "en"

    def test_manual_edit_wins_over_newer_transcription(self, db_session, monkeypatch):
        import app.pipeline_captions as pc
        self._setup(monkeypatch)
        monkeypatch.setattr(localisation_mod, "transcribe_video", _boom_transcribe)
        manual = [{"start": 0.0, "end": 1.0, "text": "Operator fixed"}]
        edited = _make_caption(
            db_session, _make_done_run(db_session, language="en"),
            status=CaptionStatus.done, language="en", lines=manual,
            lines_source=pc.LINES_SOURCE_MANUAL, audio_sha256=self.FP,
        )
        _make_caption(
            db_session, _make_done_run(db_session, language="en"),
            status=CaptionStatus.done, language="en",
            lines=[{"start": 0.0, "end": 1.0, "text": "Raw"}],
            lines_source=pc.LINES_SOURCE_TRANSCRIBE, audio_sha256=self.FP,
        )
        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())

        db_session.refresh(new)
        assert new.lines == manual
        assert new.lines_reused_from == edited.id

    def test_manual_caption_records_its_fingerprint(self, db_session, monkeypatch):
        import app.pipeline_captions as pc
        self._setup(monkeypatch)
        monkeypatch.setattr(localisation_mod, "transcribe_video", _boom_transcribe)
        caption = _make_caption(
            db_session, _make_done_run(db_session, language="en"),
            lines=[{"start": 0.0, "end": 1.0, "text": "x"}],
            lines_source=pc.LINES_SOURCE_MANUAL,
        )
        pc.process_caption(caption.id, gdrive=FakeGDriveClient())
        db_session.refresh(caption)
        assert caption.audio_sha256 == self.FP
        assert caption.lines_source == pc.LINES_SOURCE_MANUAL

    @pytest.mark.parametrize(
        "donor_kwargs",
        [
            {"language": "ja"},          # other language
            {"lines": []},               # wordless transcript is not reused
            {"audio_sha256": "other-audio"},
        ],
    )
    def test_no_eligible_donor_transcribes(self, db_session, monkeypatch, donor_kwargs):
        import app.pipeline_captions as pc
        self._setup(monkeypatch)
        transcribe = _fake_transcribe(lines=[{"start": 0.0, "end": 1.0, "text": "t"}])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        donor = dict(
            status=CaptionStatus.done, language="en",
            lines=[{"start": 0.0, "end": 1.0, "text": "donor"}],
            lines_source=pc.LINES_SOURCE_TRANSCRIBE, audio_sha256=self.FP,
        )
        donor.update(donor_kwargs)
        _make_caption(db_session, _make_done_run(db_session, language="en"), **donor)

        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())
        db_session.refresh(new)
        assert len(transcribe.calls) == 1
        assert new.lines_source == pc.LINES_SOURCE_TRANSCRIBE
        assert new.lines_reused_from is None

    def test_no_fingerprint_never_reuses(self, db_session, monkeypatch):
        import app.pipeline_captions as pc
        self._setup(monkeypatch, fp=None)
        transcribe = _fake_transcribe(lines=[{"start": 0.0, "end": 1.0, "text": "t"}])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        _make_caption(
            db_session, _make_done_run(db_session, language="en"),
            status=CaptionStatus.done, language="en",
            lines=[{"start": 0.0, "end": 1.0, "text": "donor"}],
            lines_source=pc.LINES_SOURCE_TRANSCRIBE, audio_sha256=None,
        )
        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())
        db_session.refresh(new)
        assert len(transcribe.calls) == 1
        assert new.audio_sha256 is None


class TestTranscriptReuseWait:
    """A newer same-audio job waits for an older one that is transcribing."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.FP = uuid.uuid4().hex * 2
        _patch_probe(monkeypatch)
        _patch_burn_subtitles(monkeypatch)
        monkeypatch.setattr(media_mod, "audio_fingerprint", lambda path: self.FP)

    def _older_in_flight(self, db_session, fp="own"):
        from datetime import datetime, timedelta
        return _make_caption(
            db_session, _make_done_run(db_session, language="en"),
            status=CaptionStatus.transcribing, language="en",
            audio_sha256=self.FP if fp == "own" else fp,
            created_at=datetime.utcnow() - timedelta(minutes=1),
        )

    def test_waits_for_older_then_reuses(self, db_session, monkeypatch, db_engine):
        import app.pipeline_captions as pc
        from sqlalchemy.orm import Session
        monkeypatch.setattr(localisation_mod, "transcribe_video", _boom_transcribe)
        older = self._older_in_flight(db_session)
        lines = [{"start": 0.0, "end": 1.0, "text": "from older"}]

        def _fake_sleep(sec):
            # The older job's worker finishes its transcription meanwhile.
            with Session(db_engine) as other:
                row = other.get(RunCaption, older.id)
                row.lines = lines
                row.lines_source = pc.LINES_SOURCE_TRANSCRIBE
                row.status = CaptionStatus.burning
                other.commit()

        monkeypatch.setattr(pc.time, "sleep", _fake_sleep)
        monkeypatch.setattr(pc, "CAPTION_DONOR_WAIT_SEC", 300.0)
        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())

        db_session.refresh(new)
        assert new.status == CaptionStatus.done
        assert new.lines == lines
        assert new.lines_reused_from == older.id

    def test_gives_up_after_the_wait_and_transcribes(self, db_session, monkeypatch):
        import app.pipeline_captions as pc
        transcribe = _fake_transcribe(lines=[{"start": 0.0, "end": 1.0, "text": "t"}])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(pc, "CAPTION_DONOR_WAIT_SEC", 0.0)
        sleeps = []
        monkeypatch.setattr(pc.time, "sleep", sleeps.append)
        self._older_in_flight(db_session)

        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())
        db_session.refresh(new)
        assert len(transcribe.calls) == 1
        assert new.lines_source == pc.LINES_SOURCE_TRANSCRIBE
        assert sleeps == []

    def test_older_job_never_waits_for_a_newer_one(self, db_session, monkeypatch):
        import app.pipeline_captions as pc
        from datetime import datetime, timedelta
        transcribe = _fake_transcribe(lines=[{"start": 0.0, "end": 1.0, "text": "t"}])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(pc.time, "sleep", lambda s: pytest.fail("must not wait"))
        _make_caption(
            db_session, _make_done_run(db_session, language="en"),
            status=CaptionStatus.transcribing, language="en", audio_sha256=self.FP,
            created_at=datetime.utcnow() + timedelta(minutes=1),
        )
        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())
        db_session.refresh(new)
        assert len(transcribe.calls) == 1

    def test_waits_for_older_that_has_not_fingerprinted_yet(
        self, db_session, monkeypatch, db_engine
    ):
        import app.pipeline_captions as pc
        from sqlalchemy.orm import Session
        monkeypatch.setattr(localisation_mod, "transcribe_video", _boom_transcribe)
        monkeypatch.setattr(pc, "CAPTION_DONOR_WAIT_SEC", 300.0)
        monkeypatch.setattr(pc, "CAPTION_UNFINGERPRINTED_WAIT_SEC", 30.0)
        older = self._older_in_flight(db_session, fp=None)
        lines = [{"start": 0.0, "end": 1.0, "text": "raced"}]

        def _fake_sleep(sec):
            with Session(db_engine) as other:
                row = other.get(RunCaption, older.id)
                row.audio_sha256 = self.FP
                row.lines = lines
                row.lines_source = pc.LINES_SOURCE_TRANSCRIBE
                other.commit()

        monkeypatch.setattr(pc.time, "sleep", _fake_sleep)
        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())
        db_session.refresh(new)
        assert new.lines == lines
        assert new.lines_reused_from == older.id

    def test_unfingerprinted_older_is_waited_on_only_briefly(self, db_session, monkeypatch):
        import app.pipeline_captions as pc
        transcribe = _fake_transcribe(lines=[{"start": 0.0, "end": 1.0, "text": "t"}])
        monkeypatch.setattr(localisation_mod, "transcribe_video", transcribe)
        monkeypatch.setattr(pc, "CAPTION_DONOR_WAIT_SEC", 300.0)
        sleeps = []
        monkeypatch.setattr(pc.time, "sleep", sleeps.append)
        self._older_in_flight(db_session, fp=None)  # e.g. an audio-less video

        new = _make_caption(db_session, _make_done_run(db_session, language="en"))
        pc.process_caption(new.id, gdrive=FakeGDriveClient())
        db_session.refresh(new)
        assert len(transcribe.calls) == 1
        assert sleeps == []  # CAPTION_UNFINGERPRINTED_WAIT_SEC is 0 here


# ---------------------------------------------------------------------------
# AICP-368: the frozen caption placement reaches build_ass
# ---------------------------------------------------------------------------


class TestProcessCaptionPlacement:
    def _burn(self, db_session, monkeypatch, *, placement, width=1080, height=1920):
        from app.pipeline_captions import process_caption

        _patch_probe(monkeypatch, width=width, height=height)
        _patch_burn_subtitles(monkeypatch)
        monkeypatch.setattr(localisation_mod, "transcribe_video", _boom_transcribe)
        run = _make_done_run(db_session)
        caption = _make_caption(
            db_session, run, lines_source="manual", placement=placement,
            lines=[{"start": 0.5, "end": 2.0, "text": "Hola amigos"}],
        )
        process_caption(caption.id, gdrive=FakeGDriveClient())
        db_session.refresh(caption)
        assert caption.status == CaptionStatus.done
        with open(os.path.join(caption_dir(caption.id, run.project_id), "captions.ass"),
                  encoding="utf-8") as fh:
            return fh.read()

    def test_band_placement_is_burned_centred(self, db_session, monkeypatch):
        ass = self._burn(db_session, monkeypatch, placement={
            "mode": "band", "center_y": 0.6219, "frame_aspect": 0.5625})
        assert "{\\pos(540,1194)}Hola amigos" in ass
        assert ",5,65,65,0,1" in ass  # Alignment 5, margins, MarginV 0

    def test_null_placement_is_the_house_style(self, db_session, monkeypatch):
        ass = self._burn(db_session, monkeypatch, placement=None)
        assert "\\pos" not in ass and ",2,65,65,307,1" in ass

    def test_other_aspect_source_falls_back_to_house(self, db_session, monkeypatch):
        ass = self._burn(db_session, monkeypatch, width=1920, height=1080, placement={
            "mode": "band", "center_y": 0.6219, "frame_aspect": 0.5625})
        assert "\\pos" not in ass
