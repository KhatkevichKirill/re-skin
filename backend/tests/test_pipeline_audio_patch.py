"""
Tests for app/pipeline_audio_patch.py — the "Fix audio from another run" tool.

Strategy (mirrors tests/test_pipeline_tail.py)
---------------------------------------------
* Own file-based SQLite engine; pipeline_audio_patch's get_session is
  monkeypatched onto it.
* REAL ffmpeg. The target and donor are synthetic videos carrying DIFFERENT
  tones (440 Hz vs 880 Hz), which is what makes the central assertion possible:
  after a patch, the window must contain the donor's tone and everything
  outside it must still contain the target's. A stubbed encoder could not tell
  a correct splice from one that silently moved or dropped audio.
* A FakeGDriveClient records update_file_content / upload_file so the
  "same file id, content replaced" contract is asserted rather than assumed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_audiopatch_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

from sqlalchemy import create_engine, event  # noqa: E402 — must follow guard
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.media import probe  # noqa: E402
from app.models import Run, RunAudioPatch, VideoProject  # noqa: E402
from app.pipeline_audio_patch import (  # noqa: E402
    AudioPatchValidationError,
    create_audio_patch,
    process_audio_patch,
)
from app.state_machine import AudioPatchStatus, ProjectStatus, RunStatus  # noqa: E402
from app.storage import audio_patch_dir, run_results_dir  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    db_path = os.path.join(_TMP_DIR, "pipeline_audio_patch_test.db")
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

    import app.pipeline_audio_patch as mod

    monkeypatch.setattr(mod, "get_session", _test_get_session)


@pytest.fixture(autouse=True)
def _no_default_folder(monkeypatch):
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGDriveClient:
    """Records both delivery paths so a test can tell them apart.

    ``update_file_content`` (content replaced, id kept) is the normal path and
    the one the feature's whole premise rests on; ``upload_file`` is only for a
    run that never got a Drive result.
    """

    def __init__(self, fail_times: int = 0):
        self.update_calls: list[dict] = []
        self.upload_calls: list[dict] = []
        self._fail_times = fail_times

    def _maybe_fail(self):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("injected Drive failure")

    def update_file_content(self, file_id, local_path, mime_type="video/mp4"):
        self._maybe_fail()
        self.update_calls.append({"file_id": file_id, "local_path": local_path})
        return {
            "id": file_id,
            "webViewLink": f"https://drive.google.com/file/d/{file_id}/view",
        }

    def upload_file(self, local_path, folder_id=None, name=None, **kw):
        self._maybe_fail()
        self.upload_calls.append(
            {"local_path": local_path, "folder_id": folder_id, "name": name}
        )
        file_id = f"gdrive-{uuid.uuid4().hex[:8]}"
        return {
            "id": file_id,
            "webViewLink": f"https://drive.google.com/file/d/{file_id}/view",
        }


class AlwaysFailGDriveClient:
    def update_file_content(self, *a, **kw):
        raise RuntimeError("Drive is down")

    def upload_file(self, *a, **kw):
        raise RuntimeError("Drive is down")


# ---------------------------------------------------------------------------
# Synthetic media helpers
# ---------------------------------------------------------------------------


def _make_video(path: str, duration: float, *, tone_hz: int = 440,
                width: int = 320, height: int = 240, fps: int = 10) -> None:
    """A synthetic video whose audio is a single pure tone.

    The tone is the marker: a patched file is checked by measuring which tone
    is present in which second, which is a direct assertion about the audio
    actually landing where it was asked to.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cmd = [
        _FFMPEG, "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
        "-f", "lavfi",
        "-i", f"sine=frequency={tone_hz}:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not create test video: {result.stderr.decode()[-500:]}"
        )


_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?[0-9.]+) dB")


def _tone_level_db(path: str, start: float, end: float, freq: int) -> float:
    """Mean level of *path*'s audio in [start, end] after a narrow bandpass at
    *freq* — how loudly that tone is present in that window."""
    cmd = [
        _FFMPEG, "-hide_banner", "-ss", str(start), "-to", str(end),
        "-i", path,
        "-af", f"bandpass=f={freq}:width_type=h:w=60,volumedetect",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    m = _MEAN_VOLUME_RE.search(result.stderr.decode())
    if not m:
        raise AssertionError(
            f"no mean_volume in ffmpeg output: {result.stderr.decode()[-400:]}"
        )
    return float(m.group(1))


def _assert_tone(path: str, start: float, end: float, *, expected: int,
                 other: int, margin_db: float = 10.0) -> None:
    """Assert *expected* dominates *other* in [start, end] by *margin_db*."""
    exp = _tone_level_db(path, start, end, expected)
    oth = _tone_level_db(path, start, end, other)
    assert exp > oth + margin_db, (
        f"[{start}, {end}] of {os.path.basename(path)}: expected {expected}Hz to "
        f"dominate {other}Hz, got {exp:.1f} dB vs {oth:.1f} dB"
    )


# ---------------------------------------------------------------------------
# DB fixture builders
# ---------------------------------------------------------------------------


def _make_project(session) -> VideoProject:
    project = VideoProject(
        source_type="upload", source_ref="src.mp4", status=ProjectStatus.ready
    )
    session.add(project)
    session.flush()
    return project


def _make_done_run(session, project: VideoProject, *, duration: float = 8.0,
                   tone_hz: int = 440, no_result: bool = False,
                   **kwargs) -> Run:
    defaults = dict(
        project_id=project.id,
        prompt="swap the character",
        status=RunStatus.done,
        # Every done run in production has one; a test that wants the "never
        # delivered" path passes result_gdrive_file_id=None explicitly.
        result_gdrive_file_id=f"drive-{uuid.uuid4().hex[:8]}",
    )
    defaults.update(kwargs)
    run = Run(**defaults)
    session.add(run)
    session.flush()

    if not no_result:
        result_path = os.path.join(
            run_results_dir(run.id, project.id), "final.mp4"
        )
        _make_video(result_path, duration, tone_hz=tone_hz)
        run.result_local_path = result_path
        run.result_duration_sec = probe(result_path).duration_sec

    session.commit()
    return run


def _make_pair(session, **run_kwargs):
    """A target run (440 Hz) and a donor run (880 Hz) in one project.

    *run_kwargs* apply to the TARGET only — the donor is never the run under
    test, it is just a file to read audio out of.
    """
    project = _make_project(session)
    target = _make_done_run(session, project, tone_hz=440, **run_kwargs)
    donor = _make_done_run(session, project, tone_hz=880)
    return project, target, donor


def _make_patch(session, target: Run, donor: Run, **kwargs) -> RunAudioPatch:
    defaults = dict(
        run_id=target.id,
        donor_run_id=donor.id,
        target_start_sec=3.0,
        target_end_sec=4.0,
        donor_start_sec=1.0,
        crossfade_sec=0.06,
        base_result_path=target.result_local_path,
        status=AudioPatchStatus.queued,
    )
    defaults.update(kwargs)
    patch = RunAudioPatch(**defaults)
    session.add(patch)
    session.commit()
    return patch


# ---------------------------------------------------------------------------
# The core contract: the right audio lands in the right place, and only there
# ---------------------------------------------------------------------------


def test_patch_replaces_only_the_window(db_session):
    _project, target, donor = _make_pair(db_session)
    target.result_gdrive_file_id = "drive-original"
    db_session.commit()
    patch = _make_patch(db_session, target, donor)
    gdrive = FakeGDriveClient()

    process_audio_patch(patch.id, gdrive=gdrive)

    db_session.expire_all()
    patch = db_session.get(RunAudioPatch, patch.id)
    assert patch.status == AudioPatchStatus.done
    assert patch.promoted_at is not None
    out = patch.output_path
    assert os.path.exists(out)

    # The window carries the donor's tone; both sides still carry the target's.
    _assert_tone(out, 3.2, 3.8, expected=880, other=440)
    _assert_tone(out, 0.5, 2.5, expected=440, other=880)
    _assert_tone(out, 4.5, 7.0, expected=440, other=880)


def test_patch_preserves_video_duration_and_frame_count(db_session):
    _project, target, donor = _make_pair(db_session)
    before = probe(target.result_local_path)
    patch = _make_patch(db_session, target, donor)

    process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    after = probe(db_session.get(RunAudioPatch, patch.id).output_path)
    # The picture is stream-copied, so its duration must not move at all.
    assert after.duration_sec == pytest.approx(before.duration_sec, abs=0.05)
    assert (after.width, after.height) == (before.width, before.height)


def test_promotion_repoints_the_run_and_keeps_the_drive_id(db_session):
    _project, target, donor = _make_pair(db_session)
    target.result_gdrive_file_id = "drive-original"
    original_result = target.result_local_path
    db_session.commit()
    patch = _make_patch(db_session, target, donor)
    gdrive = FakeGDriveClient()

    process_audio_patch(patch.id, gdrive=gdrive)

    db_session.expire_all()
    target = db_session.get(Run, target.id)
    patch = db_session.get(RunAudioPatch, patch.id)
    # The run now serves the patched file...
    assert target.result_local_path == patch.output_path
    assert target.result_local_path != original_result
    # ...through the SAME Drive id, whose content was replaced in place. This
    # is what keeps an already-exported Final Creatives row correct.
    assert target.result_gdrive_file_id == "drive-original"
    assert gdrive.update_calls == [
        {"file_id": "drive-original", "local_path": patch.output_path}
    ]
    assert gdrive.upload_calls == []
    # The pre-patch delivery is still on disk — nothing is overwritten.
    assert os.path.exists(original_result)
    assert patch.base_result_path == original_result


def test_run_without_drive_result_gets_a_fresh_upload(db_session, monkeypatch):
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-1")
    _project, target, donor = _make_pair(db_session, result_gdrive_file_id=None)
    assert target.result_gdrive_file_id is None
    patch = _make_patch(db_session, target, donor)
    gdrive = FakeGDriveClient()

    process_audio_patch(patch.id, gdrive=gdrive)

    db_session.expire_all()
    target = db_session.get(Run, target.id)
    assert gdrive.update_calls == []
    assert len(gdrive.upload_calls) == 1
    assert gdrive.upload_calls[0]["folder_id"] == "folder-1"
    assert target.result_gdrive_file_id is not None
    assert db_session.get(RunAudioPatch, patch.id).gdrive_file_id == (
        target.result_gdrive_file_id
    )


# ---------------------------------------------------------------------------
# Failure paths — a failed patch must leave the delivered result untouched
# ---------------------------------------------------------------------------


def test_drive_failure_leaves_the_run_untouched(db_session):
    _project, target, donor = _make_pair(db_session)
    target.result_gdrive_file_id = "drive-original"
    original_result = target.result_local_path
    db_session.commit()
    patch = _make_patch(db_session, target, donor)

    with pytest.raises(Exception):
        process_audio_patch(patch.id, gdrive=AlwaysFailGDriveClient())

    db_session.expire_all()
    target = db_session.get(Run, target.id)
    patch = db_session.get(RunAudioPatch, patch.id)
    assert patch.status == AudioPatchStatus.failed
    assert patch.error_message
    assert patch.promoted_at is None
    # The run still points at what it always did.
    assert target.result_local_path == original_result
    assert target.result_gdrive_file_id == "drive-original"


def test_transient_drive_failure_is_retried(db_session):
    _project, target, donor = _make_pair(db_session)
    target.result_gdrive_file_id = "drive-original"
    db_session.commit()
    patch = _make_patch(db_session, target, donor)
    gdrive = FakeGDriveClient(fail_times=2)

    import app.pipeline_audio_patch as mod

    original_backoff = mod.AUDIO_PATCH_DELIVER_BACKOFF_SEC
    mod.AUDIO_PATCH_DELIVER_BACKOFF_SEC = 0
    try:
        process_audio_patch(patch.id, gdrive=gdrive)
    finally:
        mod.AUDIO_PATCH_DELIVER_BACKOFF_SEC = original_backoff

    db_session.expire_all()
    assert db_session.get(RunAudioPatch, patch.id).status == AudioPatchStatus.done
    assert len(gdrive.update_calls) == 1


def test_missing_donor_run_fails_the_patch(db_session):
    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor, donor_run_id=str(uuid.uuid4()))

    with pytest.raises(Exception):
        process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    patch = db_session.get(RunAudioPatch, patch.id)
    assert patch.status == AudioPatchStatus.failed
    assert "no longer exists" in patch.error_message


def test_stale_base_result_refuses_to_promote(db_session):
    """A patch whose base was re-pointed while it queued must not overwrite the
    newer result — the exact race the base_result_path check exists for."""
    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor)

    # Something else (another patch, a range attempt) promoted meanwhile.
    moved = os.path.join(
        run_results_dir(target.id, target.project_id), "final_v2.mp4"
    )
    _make_video(moved, 8.0, tone_hz=440)
    target.result_local_path = moved
    db_session.commit()

    with pytest.raises(Exception):
        process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    patch = db_session.get(RunAudioPatch, patch.id)
    target = db_session.get(Run, target.id)
    assert patch.status == AudioPatchStatus.failed
    assert target.result_local_path == moved


def test_already_done_patch_is_not_reprocessed(db_session):
    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor)
    gdrive = FakeGDriveClient()
    process_audio_patch(patch.id, gdrive=gdrive)

    assert len(gdrive.update_calls) == 1

    # A duplicate enqueue (or a recovery re-enqueue) must be a no-op, not a
    # second Drive write onto an already-promoted run.
    process_audio_patch(patch.id, gdrive=gdrive)
    assert len(gdrive.update_calls) == 1
    assert gdrive.upload_calls == []


def test_orphan_resume_from_processing(db_session):
    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor, status=AudioPatchStatus.processing)

    process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    assert db_session.get(RunAudioPatch, patch.id).status == AudioPatchStatus.done


# ---------------------------------------------------------------------------
# create_audio_patch validation
# ---------------------------------------------------------------------------


def test_create_rejects_self_as_donor(db_session):
    _project, target, _donor = _make_pair(db_session)
    with pytest.raises(AudioPatchValidationError, match="different run"):
        create_audio_patch(
            run=target, donor=target, target_start_sec=1.0, target_end_sec=2.0,
            donor_start_sec=1.0, crossfade_sec=0.06, db=db_session,
        )


def test_create_rejects_unfinished_donor(db_session):
    project = _make_project(db_session)
    target = _make_done_run(db_session, project)
    donor = _make_done_run(db_session, project, status=RunStatus.processing)
    with pytest.raises(AudioPatchValidationError, match="not done"):
        create_audio_patch(
            run=target, donor=donor, target_start_sec=1.0, target_end_sec=2.0,
            donor_start_sec=1.0, crossfade_sec=0.06, db=db_session,
        )


def test_stale_staging_debris_is_swept(db_session):
    """§1-gate follow-up finding: a SIGKILL (what a redeploy does to an
    in-flight job here) skips the `finally`, and nothing else ever cleans the
    directory — the reaper only marks the row failed."""
    import app.pipeline_audio_patch as mod

    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor)
    out_dir = audio_patch_dir(patch.id, target.id, target.project_id)
    debris = os.path.join(out_dir, f"{mod._STAGING_PREFIX}deadbeef.mp4")
    with open(debris, "wb") as fh:
        fh.write(b"half a video")
    old = time.time() - (mod._STAGING_STALE_AFTER_SEC + 60)
    os.utime(debris, (old, old))

    process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    assert not os.path.exists(debris)


def test_debris_from_an_earlier_killed_patch_is_swept(db_session):
    """§1-gate finding (5th pass): `failed` is terminal, so the operator's
    retry is a NEW patch with a NEW directory — a sweep confined to the
    current patch's own directory would never collect the killed one's
    leftovers, and they would accumulate forever."""
    import app.pipeline_audio_patch as mod

    _project, target, donor = _make_pair(db_session)
    killed = _make_patch(db_session, target, donor, status=AudioPatchStatus.failed)
    killed_dir = audio_patch_dir(killed.id, target.id, target.project_id)
    debris = os.path.join(killed_dir, f"{mod._STAGING_PREFIX}orphaned.mp4")
    with open(debris, "wb") as fh:
        fh.write(b"half a video from a SIGKILLed worker")
    old_mtime = time.time() - (mod._STAGING_STALE_AFTER_SEC + 60)
    os.utime(debris, (old_mtime, old_mtime))

    retry = _make_patch(db_session, target, donor)
    process_audio_patch(retry.id, gdrive=FakeGDriveClient())

    assert not os.path.exists(debris)


def test_a_fresh_staging_file_is_never_swept(db_session):
    """The age guard is what makes the sweep safe when two workers share this
    directory: a file young enough to belong to a live render is left alone."""
    import app.pipeline_audio_patch as mod

    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor)
    out_dir = audio_patch_dir(patch.id, target.id, target.project_id)
    live = os.path.join(out_dir, f"{mod._STAGING_PREFIX}inflight.mp4")
    with open(live, "wb") as fh:
        fh.write(b"another worker is writing this")

    process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    assert os.path.exists(live)


def test_create_rejects_a_donor_from_another_project(db_session):
    """§1-gate finding (2026-09-07): the picker offers siblings only, but the
    endpoint took a run id straight from a form — a pasted id from another
    project would splice unrelated speech into a delivered creative, behind its
    existing Final Creatives link."""
    project_a = _make_project(db_session)
    target = _make_done_run(db_session, project_a, tone_hz=440)
    project_b = _make_project(db_session)
    stranger = _make_done_run(db_session, project_b, tone_hz=880)

    with pytest.raises(AudioPatchValidationError, match="different project"):
        create_audio_patch(
            run=target, donor=stranger, target_start_sec=3.0, target_end_sec=4.0,
            donor_start_sec=1.0, crossfade_sec=0.06, db=db_session,
        )


def test_the_worker_re_checks_the_donor_project(db_session):
    """The gate is in the shared validator, so a row that got past the API
    (created before the check existed, or by a moved run) still fails safely in
    the worker instead of patching the audio."""
    project_a = _make_project(db_session)
    target = _make_done_run(db_session, project_a, tone_hz=440)
    project_b = _make_project(db_session)
    stranger = _make_done_run(db_session, project_b, tone_hz=880)
    original_result = target.result_local_path
    patch = _make_patch(db_session, target, stranger)

    with pytest.raises(Exception):
        process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    assert db_session.get(RunAudioPatch, patch.id).status == AudioPatchStatus.failed
    assert db_session.get(Run, target.id).result_local_path == original_result


def test_no_staging_file_survives_a_successful_render(db_session):
    """§1-gate finding (2026-09-07): the candidate path is derived from the
    patch id alone, so two workers on the same patch would both aim ffmpeg's
    `-y` at one file and half of it could reach Drive. The render now goes to a
    unique .part and is published with an atomic rename."""
    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor)

    process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    out = db_session.get(RunAudioPatch, patch.id).output_path
    assert os.path.exists(out)
    assert not os.path.basename(out).startswith(".staging-")
    leftovers = [
        f for f in os.listdir(os.path.dirname(out)) if f.startswith(".staging-")
    ]
    assert leftovers == [], f"staging files left behind: {leftovers}"


def test_no_staging_file_survives_a_failed_render(db_session):
    _project, target, donor = _make_pair(db_session)
    # A window the donor cannot supply: validation passes at create time only
    # because we build the row directly, so the failure happens in ffmpeg.
    patch = _make_patch(
        db_session, target, donor,
        target_start_sec=1.0, target_end_sec=2.0, donor_start_sec=99.0,
    )

    with pytest.raises(Exception):
        process_audio_patch(patch.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    patch = db_session.get(RunAudioPatch, patch.id)
    assert patch.status == AudioPatchStatus.failed
    out_dir = audio_patch_dir(patch.id, target.id, target.project_id)
    leftovers = [f for f in os.listdir(out_dir) if f.startswith(".staging-")]
    assert leftovers == [], f"staging files left behind: {leftovers}"


def test_create_rejects_window_past_the_end(db_session):
    _project, target, donor = _make_pair(db_session)
    with pytest.raises(AudioPatchValidationError, match="long"):
        create_audio_patch(
            run=target, donor=donor, target_start_sec=7.0, target_end_sec=20.0,
            donor_start_sec=1.0, crossfade_sec=0.06, db=db_session,
        )


def test_create_rejects_donor_too_short_for_the_window(db_session):
    _project, target, donor = _make_pair(db_session)
    with pytest.raises(AudioPatchValidationError, match="only"):
        create_audio_patch(
            run=target, donor=donor, target_start_sec=1.0, target_end_sec=4.0,
            donor_start_sec=7.0, crossfade_sec=0.06, db=db_session,
        )


def test_create_rejects_a_too_short_window(db_session):
    _project, target, donor = _make_pair(db_session)
    with pytest.raises(AudioPatchValidationError, match="at least"):
        create_audio_patch(
            run=target, donor=donor, target_start_sec=1.0, target_end_sec=1.01,
            donor_start_sec=1.0, crossfade_sec=0.06, db=db_session,
        )


def test_create_rejects_a_run_with_no_result_on_disk(db_session):
    project = _make_project(db_session)
    target = _make_done_run(db_session, project, no_result=True)
    donor = _make_done_run(db_session, project, tone_hz=880)
    with pytest.raises(AudioPatchValidationError, match="no result video"):
        create_audio_patch(
            run=target, donor=donor, target_start_sec=1.0, target_end_sec=2.0,
            donor_start_sec=1.0, crossfade_sec=0.06, db=db_session,
        )


def test_one_active_patch_per_run_is_enforced_by_the_database(db_session):
    """The application-level 409 is a non-locking SELECT; this index is what
    actually stops two concurrent creates."""
    _project, target, donor = _make_pair(db_session)
    _make_patch(db_session, target, donor)
    with pytest.raises(IntegrityError):
        _make_patch(db_session, target, donor)
    db_session.rollback()


def test_a_terminal_patch_does_not_block_a_new_one(db_session):
    _project, target, donor = _make_pair(db_session)
    _make_patch(db_session, target, donor, status=AudioPatchStatus.done)
    # No IntegrityError: the partial index only covers active statuses.
    second = _make_patch(db_session, target, donor)
    assert second.status == AudioPatchStatus.queued


def test_successive_patches_compose(db_session):
    """A second patch takes the first's output as its base, so corrections
    stack instead of fighting over one file."""
    project = _make_project(db_session)
    target = _make_done_run(db_session, project, tone_hz=440)
    donor_a = _make_done_run(db_session, project, tone_hz=880)
    donor_b = _make_done_run(db_session, project, tone_hz=1760)

    first = _make_patch(db_session, target, donor_a)
    process_audio_patch(first.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    target = db_session.get(Run, target.id)
    second = _make_patch(
        db_session, target, donor_b,
        target_start_sec=6.0, target_end_sec=7.0, donor_start_sec=1.0,
    )
    assert second.base_result_path == target.result_local_path

    process_audio_patch(second.id, gdrive=FakeGDriveClient())

    db_session.expire_all()
    out = db_session.get(RunAudioPatch, second.id).output_path
    # Both windows survive in the final file, plus the untouched original.
    _assert_tone(out, 3.2, 3.8, expected=880, other=440)
    _assert_tone(out, 6.2, 6.8, expected=1760, other=440)
    _assert_tone(out, 0.5, 2.5, expected=440, other=880)


def test_no_drive_target_and_no_folder_promotes_locally(db_session, monkeypatch):
    """A run that was never delivered still gets its local result repaired —
    there is nothing on Drive to diverge from, so skipping delivery is right
    (mirrors pipeline_tail's "no folder_id -> skip delivery, still done")."""
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "")
    _project, target, donor = _make_pair(db_session, result_gdrive_file_id=None)
    patch = _make_patch(db_session, target, donor)
    gdrive = FakeGDriveClient()

    process_audio_patch(patch.id, gdrive=gdrive)

    db_session.expire_all()
    patch = db_session.get(RunAudioPatch, patch.id)
    target = db_session.get(Run, target.id)
    assert patch.status == AudioPatchStatus.done
    assert patch.gdrive_file_id is None
    assert target.result_local_path == patch.output_path
    assert gdrive.update_calls == [] and gdrive.upload_calls == []
    _assert_tone(patch.output_path, 3.2, 3.8, expected=880, other=440)


def test_process_returns_the_published_final_only_for_its_own_promotion(db_session):
    """AICP-299 §4: the committed descriptor for the wrapper's knowledge link."""
    from app.knowledge.publication import PublishedFinal

    _project, target, donor = _make_pair(db_session)
    patch = _make_patch(db_session, target, donor)
    gdrive = FakeGDriveClient()
    published = process_audio_patch(patch.id, gdrive=gdrive)
    db_session.expire_all()
    run = db_session.get(Run, target.id)
    assert published == PublishedFinal(run.id, run.project_id, run.result_local_path, "audio",
                                       f"audio_patch:{patch.id}")
    assert process_audio_patch(patch.id, gdrive=gdrive) is None  # already done: nothing published
