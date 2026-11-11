"""
Tests for app/pipeline_v2.py — analyze_project and process_run.

Strategy
--------
* Own SQLite DB engine (file-based for cross-session visibility) isolated from
  other test modules. pipeline_v2's get_session() is monkeypatched to use this
  engine.
* Tiny synthetic video created via ffmpeg (no GPU required).
* face.propose_segments is monkeypatched to return a fixed 3-segment partition:
    swap  0–9s  | keep  9–15s | swap  15–25s
  over a 25-second synthetic video.
* Fake KieClient and GDriveClient — no real network calls.
"""

from __future__ import annotations

import json
import os
import shutil
import time as _time
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager

import pytest

# ---------------------------------------------------------------------------
# Prevent accidental real kie/gdrive calls; set a data dir for v2.
# ---------------------------------------------------------------------------

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_v2_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# App imports
# ---------------------------------------------------------------------------

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import ai_models
from app.db import Base
from app.models import LibraryClip, Run, RunSegment, SegmentDef, VideoProject
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus, transition
from app.kie_client import KieTaskFailed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Duration of the synthetic source video (seconds).
_VIDEO_DURATION = 25.0


@pytest.fixture(scope="session")
def synthetic_video():
    """
    Create a 25-second synthetic mp4 using ffmpeg lavfi sources.
    Returns the path; cleaned up at session end.
    """
    path = os.path.join(_TMP_DIR, "source_v2.mp4")
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"color=c=red:s=320x240:d={int(_VIDEO_DURATION)}:r=25",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={int(_VIDEO_DURATION)}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
            "-c:a", "aac",
            path,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"ffmpeg failed:\n{r.stderr}"
    yield path


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    """
    Create a file-based SQLite engine isolated from the rest of the test suite.
    File-based (not :memory:) so multiple sessions see the same data.
    """
    db_path = os.path.join(_TMP_DIR, "pipeline_v2_test.db")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
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
    """Per-test database session; does NOT auto-rollback because pipeline
    functions open their own sessions and we need data to persist."""
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture(autouse=True)
def patch_get_session(db_engine, monkeypatch):
    """
    Replace app.pipeline_v2's get_session with one that uses the test engine.
    """
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

    import app.pipeline_v2 as pipeline_v2_mod
    monkeypatch.setattr(pipeline_v2_mod, "get_session", _test_get_session)


# ---------------------------------------------------------------------------
# Fixed segment partition returned by the monkeypatched propose_segments.
# Partition: [0, 9) swap | [9, 15) keep | [15, 25) swap
# ---------------------------------------------------------------------------

from app.face import ProposedSegment as _PS


@pytest.fixture()
def tmp_path():
    """Per-test scratch dir INSIDE the storage root, shadowing pytest's own.

    Serving and kie.ai uploads refuse any stored path outside DATA_DIR
    (AICP-148), and real stored paths always live there — so fixture files
    must too.
    """
    import pathlib

    from app import storage

    root = storage.data_dir_root()
    os.makedirs(root, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix="tmp-path-", dir=root))

_FIXED_SEGMENTS = [
    _PS(start_sec=0.0, end_sec=9.0, has_face=True, action="swap"),
    _PS(start_sec=9.0, end_sec=15.0, has_face=False, action="keep"),
    _PS(start_sec=15.0, end_sec=25.0, has_face=True, action="swap"),
]


@pytest.fixture()
def patch_propose(monkeypatch):
    """Monkeypatch face.propose_segments to return the fixed partition."""
    import app.pipeline_v2 as pipeline_v2_mod
    import app.face as face_mod

    monkeypatch.setattr(face_mod, "propose_segments", lambda *a, **kw: _FIXED_SEGMENTS)
    monkeypatch.setattr(
        pipeline_v2_mod.face_mod, "propose_segments", lambda *a, **kw: _FIXED_SEGMENTS
    )
    return _FIXED_SEGMENTS


# ---------------------------------------------------------------------------
# Fake external clients
# ---------------------------------------------------------------------------

# Sentinel recorded by FakeKieClient when an optional createTask kwarg was never
# passed. Distinguishing "omitted" from "passed as None/False" matters for
# `generate_audio`: only Seedance 2.5 accepts the field, and every other model's
# payload must stay exactly as it was before 2.5 existed.
NOT_PASSED = object()


class FakeKieClient:
    """
    Fake KieClient.

    - upload_file  → returns a fake URL, records calls.
    - create_task  → returns a fake task ID; records prompt + reference_image_urls.
    - poll_task    → returns a fake result URL (raises KieTaskFailed if the
                     task_id matches fail_task_id).
    - download_result → copies the synthetic video to dst so it is a real file.
    """

    def __init__(self, synthetic_video_path: str, fail_task_id: str | None = None):
        self._src = synthetic_video_path
        self._fail_task_id = fail_task_id
        self.upload_calls: list[str] = []
        # Each entry: {"local_path", "upload_path", "file_name"} — captures the
        # remote naming so tests can assert uploads are namespaced per Run.
        self.upload_records: list[dict] = []
        self.create_task_calls: list[str] = []
        # Each entry: {"task_id", "prompt", "reference_image_urls", "model",
        #   "resolution", "aspect_ratio", "duration", "generate_audio",
        #   "extra_kwargs"}
        self.create_task_records: list[dict] = []
        # Gemini Omni task records: {"task_id", "prompt", "image_urls", "video_url",
        #   "video_start", "video_end", "resolution", "aspect_ratio", "duration",
        #   "extra_kwargs"}
        self.create_omni_records: list[dict] = []
        # GPT Image 2 (image-to-image) task records: {"task_id", "prompt",
        #   "input_urls", "aspect_ratio", "resolution"}
        self.create_image_records: list[dict] = []
        self.poll_calls: list[str] = []

    def upload_file(
        self, local_path: str, upload_path: str = "charswap",
        file_name: str | None = None, **kw,
    ) -> str:
        self.upload_calls.append(local_path)
        remote_name = file_name or os.path.basename(local_path)
        self.upload_records.append({
            "local_path": local_path,
            "upload_path": upload_path,
            "file_name": file_name,
        })
        # Mirror the real host: the URL is keyed on the remote object name, so a
        # namespaced file_name yields a collision-free URL.
        return f"https://fake-kie.example.com/{upload_path}/{remote_name}"

    def create_task(
        self,
        *,
        prompt,
        reference_image_urls,
        reference_video_urls,
        resolution,
        aspect_ratio,
        duration,
        model="bytedance/seedance-2",
        generate_audio=NOT_PASSED,
        reference_audio_urls=NOT_PASSED,
        **kw,
    ) -> str:
        """Mirror of KieClient.create_task.

        *generate_audio* defaults to :data:`NOT_PASSED` rather than ``None`` so a
        test can tell "the pipeline omitted the kwarg entirely" (required for the
        Seedance 2.0 family and Gemini Omni — those backends reject the field, so
        their payload must stay byte-identical to what it has always been) apart
        from "the pipeline explicitly asked for no audio" (``False``, which
        Seedance 2.5 does accept). ``**kw`` swallows anything else so a new
        model's kwarg surfaces as a failed assertion, not a TypeError.
        """
        kw.pop("once", None)  # AICP-351 transport flag, not part of the payload
        task_id = f"fake-task-{uuid.uuid4().hex[:8]}"
        self.create_task_calls.append(task_id)
        self.create_task_records.append({
            "task_id": task_id,
            "prompt": prompt,
            "reference_image_urls": list(reference_image_urls),
            "model": model,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            # NOT_PASSED for the same reason as generate_audio: the models
            # without the field reject the whole request, so "no voice" must be
            # an ABSENT kwarg, distinguishable from an empty list.
            "reference_audio_urls": reference_audio_urls,
            "extra_kwargs": dict(kw),
        })
        return task_id

    def create_omni_task(
        self,
        *,
        prompt,
        image_urls,
        video_url,
        video_start,
        video_end,
        resolution,
        aspect_ratio,
        duration,
        seed=None,
        **kw,
    ) -> str:
        kw.pop("once", None)  # AICP-351 transport flag, not part of the payload
        task_id = f"fake-omni-{uuid.uuid4().hex[:8]}"
        self.create_task_calls.append(task_id)
        self.create_omni_records.append({
            "task_id": task_id,
            "prompt": prompt,
            "image_urls": list(image_urls),
            "video_url": video_url,
            "video_start": video_start,
            "video_end": video_end,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            # Gemini Omni has no audio switch at all; recording the leftovers
            # lets a test prove nothing extra was smuggled into its payload.
            "extra_kwargs": dict(kw),
        })
        return task_id

    def create_image_task(
        self,
        *,
        prompt,
        input_urls,
        aspect_ratio="auto",
        resolution="1K",
        once=False,  # AICP-351: the single-send flag of the real client
    ) -> str:
        task_id = f"fake-img-{uuid.uuid4().hex[:8]}"
        self.create_task_calls.append(task_id)
        self.create_image_records.append({
            "task_id": task_id,
            "prompt": prompt,
            "input_urls": list(input_urls),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        })
        return task_id

    def poll_task(self, task_id: str, **kw) -> str:
        self.poll_calls.append(task_id)
        if self._fail_task_id and task_id == self._fail_task_id:
            raise KieTaskFailed("injected failure")
        return f"https://fake-kie.example.com/results/{task_id}.mp4"

    def get_task(self, task_id: str, **kw) -> dict:
        # The v2 poll loop uses get_task; return a terminal recordInfo 'data' dict.
        self.poll_calls.append(task_id)
        if self._fail_task_id and task_id == self._fail_task_id:
            return {"state": "fail", "failMsg": "injected failure"}
        url = f"https://fake-kie.example.com/results/{task_id}.mp4"
        return {"state": "success", "resultJson": json.dumps({"resultUrls": [url]})}

    def download_result(self, url: str, dst_path: str) -> None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(self._src, dst_path)


class FakeGDriveClient:
    """Fake GDriveClient — records calls, returns predictable data."""

    def __init__(self):
        self.upload_calls: list[dict] = []
        self.download_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.update_error: Exception | None = None

    def update_file_content(self, file_id: str, local_path: str, mime_type: str = "video/mp4") -> dict:
        self.update_calls.append({"file_id": file_id, "local_path": local_path})
        if self.update_error is not None:
            raise self.update_error
        return {"id": file_id, "webViewLink": f"https://drive.google.com/file/{file_id}"}

    def download_file(self, link_or_id: str, dst_path: str) -> str:
        self.download_calls.append({"link": link_or_id, "dst": dst_path})
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as fh:
            fh.write(b"fake-gdrive-content")
        return dst_path

    def upload_file(
        self,
        local_path: str,
        folder_id: str | None = None,
        name: str | None = None,
        **kw,
    ) -> dict:
        self.upload_calls.append(
            {"local_path": local_path, "folder_id": folder_id, "name": name}
        )
        file_id = f"gdrive-{uuid.uuid4().hex[:8]}"
        return {
            "id": file_id,
            "webViewLink": f"https://drive.google.com/file/{file_id}",
        }


def test_poll_pending_tasks_does_not_hold_session_while_sleeping(monkeypatch, tmp_path):
    """The poll wait loop must not keep a DB session/transaction open."""
    import app.pipeline_v2 as pv2

    active_sessions = {"count": 0}
    slept = {"called": False}

    class _FakeRunSegment:
        def __init__(self):
            self.status = SegmentStatus.generating
            self.seedance_result_url = None
            self.local_result_path = None
            # Real RunSegments carry this; the truncated-result guard
            # reads it to compare against the returned clip.
            self.local_clip_path = None

    fake_rs = _FakeRunSegment()

    class _FakeSession:
        def get(self, _model, _id):
            return fake_rs

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    @contextmanager
    def fake_get_session():
        active_sessions["count"] += 1
        try:
            yield _FakeSession()
        finally:
            active_sessions["count"] -= 1

    class _SlowThenDoneKie:
        def __init__(self):
            self.calls = 0

        def get_task(self, _task_id):
            self.calls += 1
            if self.calls == 1:
                return {"state": "processing"}
            return {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://fake/result.mp4"]}),
            }

        def download_result(self, _url, dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, "wb") as fh:
                fh.write(b"result")

    def fake_sleep(_seconds):
        slept["called"] = True
        assert active_sessions["count"] == 0

    monkeypatch.setattr(pv2, "get_session", fake_get_session)
    monkeypatch.setattr(pv2.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        pv2, "transition", lambda obj, status: setattr(obj, "status", status)
    )

    pending = {
        "task-1": {
            "rs_id": "rs-1",
            "index": 0,
            # Wall-clock epoch deadline (time.time()) — both _poll_pending_tasks
            # and poll_run_once_internal use time.time() for deadline comparisons
            # so the deadline survives Redis serialisation between poll rounds.
            "deadline": pv2.time.time() + 60,
        }
    }

    pv2._poll_pending_tasks(
        pending=pending,
        kie=_SlowThenDoneKie(),
        results_dir=str(tmp_path),
    )

    assert slept["called"] is True
    assert pending == {}
    assert fake_rs.status == SegmentStatus.completed
    assert fake_rs.seedance_result_url == "https://fake/result.mp4"


def test_poll_pending_tasks_captures_credits_consumed(monkeypatch, tmp_path):
    """On success, kie.ai's `creditsConsumed` is recorded as the segment cost."""
    import app.pipeline_v2 as pv2

    class _FakeRunSegment:
        def __init__(self):
            self.status = SegmentStatus.generating
            self.seedance_result_url = None
            self.local_result_path = None
            # Real RunSegments carry this; the truncated-result guard
            # reads it to compare against the returned clip.
            self.local_clip_path = None
            self.error_message = None
            self.cost_credits = None

    fake_rs = _FakeRunSegment()

    class _FakeSession:
        def get(self, _model, _id):
            return fake_rs

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    @contextmanager
    def fake_get_session():
        yield _FakeSession()

    class _Kie:
        def get_task(self, _task_id):
            return {
                "state": "success",
                "creditsConsumed": 42,
                "resultJson": json.dumps({"resultUrls": ["https://fake/result.mp4"]}),
            }

        def download_result(self, _url, dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, "wb") as fh:
                fh.write(b"result")

    monkeypatch.setattr(pv2, "get_session", fake_get_session)
    monkeypatch.setattr(pv2.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        pv2, "transition", lambda obj, status: setattr(obj, "status", status)
    )

    pending = {
        "task-1": {"rs_id": "rs-1", "index": 0, "deadline": pv2.time.time() + 60}
    }
    pv2._poll_pending_tasks(pending=pending, kie=_Kie(), results_dir=str(tmp_path))

    assert fake_rs.status == SegmentStatus.completed
    assert fake_rs.cost_credits == 42.0


def test_parse_cost_credits_handles_missing_and_bad_values():
    import app.pipeline_v2 as pv2

    assert pv2._parse_cost_credits({"creditsConsumed": 50}) == 50.0
    assert pv2._parse_cost_credits({"creditsConsumed": "12.5"}) == 12.5
    assert pv2._parse_cost_credits({}) is None
    assert pv2._parse_cost_credits({"creditsConsumed": None}) is None
    assert pv2._parse_cost_credits({"creditsConsumed": "n/a"}) is None


# ---------------------------------------------------------------------------
# Resubmit-recipe forward/backward compatibility
#
# The poll loop resubmits a failed task with `_create_swap_task(kie=kie,
# **recipe)`, where `recipe` was serialised to Redis as JSON — possibly by a
# DIFFERENT worker version during a rolling deploy or a rollback. A key this
# version does not know must not raise TypeError there: the surrounding
# `except Exception` turns that into `new_task_id = None`, which marks the
# segment permanently failed and converts a retryable transient kie.ai error
# into a lost run.
# ---------------------------------------------------------------------------


def _valid_recipe(**overrides) -> dict:
    """A resubmit recipe as _submit_swap_segment_isolated builds one."""
    recipe = {
        "run_model": "seedance",
        "effective_prompt": "Replace the character",
        "ref_urls": [],
        "clip_url": "https://kie.test/clip.mp4",
        "clip_start": 0.0,
        "clip_end": 6.0,
        "run_resolution": "480p",
        "project_aspect_ratio": "9:16",
        "project_width": 1080,
        "project_height": 1920,
        "generate_audio": None,
    }
    recipe.update(overrides)
    return recipe


class _RecordingKie:
    """Minimal kie stub for _create_swap_task (records the createTask kwargs)."""

    def __init__(self):
        self.calls: list[dict] = []

    def create_task(self, **kw) -> str:
        self.calls.append(kw)
        return f"task-{len(self.calls)}"


def test_create_swap_task_ignores_unknown_recipe_keys():
    """An unrecognised recipe key is dropped, not fatal."""
    import app.pipeline_v2 as pv2

    kie = _RecordingKie()
    task_id = pv2._create_swap_task(
        kie=kie,
        **_valid_recipe(
            # e.g. the key this very change removed, replayed by an older worker.
            run_audio_mode="original",
            some_future_field=[1, 2, 3],
        ),
    )

    assert task_id == "task-1"
    # The request is still built correctly from the keys we do understand.
    assert len(kie.calls) == 1
    assert kie.calls[0]["model"] == "bytedance/seedance-2"
    assert kie.calls[0]["duration"] == 6
    # And nothing unknown was forwarded to the client.
    assert "run_audio_mode" not in kie.calls[0]
    assert "some_future_field" not in kie.calls[0]


def test_recipe_without_generate_audio_omits_the_field():
    """A pre-`generate_audio` recipe replayed against 2.5 leaves the field off.

    That is the safe degradation: the model applies its own default (it writes
    audio) instead of being explicitly silenced by a key that was never stored.
    """
    import app.pipeline_v2 as pv2

    recipe = _valid_recipe(run_model="seedance-2-5", run_audio_mode="seedance")
    del recipe["generate_audio"]

    kie = _RecordingKie()
    pv2._create_swap_task(kie=kie, **recipe)

    assert len(kie.calls) == 1
    assert kie.calls[0]["model"] == "bytedance/seedance-2-5"
    assert "generate_audio" not in kie.calls[0]


def test_unknown_recipe_key_is_warned_about_only_once(caplog):
    """One WARNING per unknown key, not one per segment per poll round."""
    import logging

    import app.pipeline_v2 as pv2

    pv2._WARNED_UNKNOWN_RECIPE_KEYS.discard("mystery_key")
    kie = _RecordingKie()
    with caplog.at_level(logging.WARNING, logger="app.pipeline_v2"):
        for _ in range(3):
            pv2._create_swap_task(kie=kie, **_valid_recipe(mystery_key=1))

    warnings = [r for r in caplog.records if "mystery_key" in r.getMessage()]
    assert len(warnings) == 1
    assert len(kie.calls) == 3


def test_stale_recipe_retry_does_not_permanently_fail_the_segment(
    monkeypatch, tmp_path
):
    """The end-to-end rollback scenario, through the real poll loop.

    A transient task failure whose stored recipe carries an obsolete key must be
    resubmitted (segment stays in flight and then completes) rather than marked
    failed — the bug being guarded against is a schema change silently killing
    every in-flight run of the previous worker version.
    """
    import app.pipeline_v2 as pv2

    class _FakeRunSegment:
        def __init__(self):
            self.status = SegmentStatus.generating
            self.seedance_task_id = "task-old"
            self.seedance_result_url = None
            self.local_result_path = None
            # Real RunSegments carry this; the truncated-result guard
            # reads it to compare against the returned clip.
            self.local_clip_path = None
            self.error_message = None
            self.cost_credits = None

    fake_rs = _FakeRunSegment()

    class _FakeSession:
        def get(self, _model, _id):
            return fake_rs

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    @contextmanager
    def fake_get_session():
        yield _FakeSession()

    class _FailThenSucceedKie(_RecordingKie):
        def get_task(self, task_id):
            if task_id == "task-old":
                return {"state": "fail", "failMsg": "Internal Error, Please try again later."}
            return {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://fake/result.mp4"]}),
            }

        def download_result(self, _url, dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, "wb") as fh:
                fh.write(b"result")

    monkeypatch.setattr(pv2, "get_session", fake_get_session)
    monkeypatch.setattr(pv2.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        pv2, "transition", lambda obj, status: setattr(obj, "status", status)
    )
    # AICP-351: the attempt receipt needs a real DB; this test is about the
    # recipe the resubmit sends, so the protocol is stubbed to the bare send.
    monkeypatch.setattr(pv2, "_paid_swap_send", lambda *, recipe, kie, **_kw: (
        "attempt-x", pv2._create_swap_task(kie=kie, once=True, **recipe)))
    monkeypatch.setattr(pv2, "_record_swap_receipt", lambda *a, **kw: True)

    kie = _FailThenSucceedKie()
    pending = {
        "task-old": {
            "rs_id": "rs-1",
            "index": 0,
            "deadline": pv2.time.time() + 600,
            "attempt": 1,
            "recipe": _valid_recipe(run_audio_mode="original"),
        }
    }
    pv2._poll_pending_tasks(pending=pending, kie=kie, results_dir=str(tmp_path))

    # Resubmitted (not swallowed into new_task_id=None) and then completed.
    assert len(kie.calls) == 1
    assert fake_rs.status == SegmentStatus.completed
    assert fake_rs.seedance_result_url == "https://fake/result.mp4"


# ---------------------------------------------------------------------------
# Helper: create a VideoProject row and return its id.
# ---------------------------------------------------------------------------


def _create_project(
    db_session,
    synthetic_video: str,
    *,
    source_type: str = "upload",
    status: ProjectStatus = ProjectStatus.created,
    project_type: str = "face_swap",
    mute_source: bool = False,
    max_segment_sec: float | None = None,
    aspect_ratio: str | None = None,
    hook_sec: float | None = None,
) -> str:
    project = VideoProject(
        id=str(uuid.uuid4()),
        source_type=source_type,
        source_ref=(
            synthetic_video
            if source_type == "upload"
            else "https://drive.google.com/file/d/fake123/view"
        ),
        source_local_path=synthetic_video if source_type == "upload" else None,
        project_type=project_type,
        mute_source=mute_source,
        aspect_ratio=aspect_ratio,
        # NULL (the default) means "the universal cap every model can generate"
        # — the behaviour every project had before this column existed.
        max_segment_sec=max_segment_sec,
        # Localisation only; NULL → settings.LOCALISATION_DEFAULT_HOOK_SEC.
        hook_sec=hook_sec,
        status=status,
    )
    db_session.add(project)
    db_session.commit()
    return project.id


def _create_run(
    db_session,
    project_id: str,
    *,
    status: RunStatus = RunStatus.queued,
    gdrive_folder_id: str | None = None,
    reference_image_urls: list | None = None,
    prompt: str = "Replace the character",
    stitch_plan: list | None = None,
) -> str:
    run = Run(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name="Test Run",
        prompt=prompt,
        reference_image_urls=reference_image_urls or [],
        resolution="480p",
        status=status,
        gdrive_folder_id=gdrive_folder_id,
        stitch_plan=stitch_plan,
    )
    db_session.add(run)
    db_session.commit()
    return run.id


# ---------------------------------------------------------------------------
# Tests: analyze_project
# ---------------------------------------------------------------------------


class TestAnalyzeProject:
    def test_analyze_happy_path(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """analyze_project: persists probe fields + SegmentDefs; project ends ready."""
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)

        analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.status == ProjectStatus.ready
            assert project.duration_sec is not None
            assert project.duration_sec == pytest.approx(_VIDEO_DURATION, abs=1.0)
            assert project.width == 320
            assert project.height == 240
            assert project.fps is not None
            assert project.aspect_ratio == "4:3"

            seg_defs = list(project.segments)
            assert len(seg_defs) == 3

            # index 0 — swap
            assert seg_defs[0].action == "swap"
            assert seg_defs[0].has_face is True
            assert seg_defs[0].start_sec == pytest.approx(0.0)
            assert seg_defs[0].end_sec == pytest.approx(9.0)

            # index 1 — keep
            assert seg_defs[1].action == "keep"
            assert seg_defs[1].has_face is False

            # index 2 — swap
            assert seg_defs[2].action == "swap"
            assert seg_defs[2].has_face is True

    def test_analyze_transitions_to_failed_on_probe_error(
        self, db_session, synthetic_video, monkeypatch
    ):
        """If probe raises, project transitions to failed and error_message is set."""
        from app.pipeline_v2 import analyze_project
        import app.pipeline_v2 as pipeline_v2_mod

        project_id = _create_project(db_session, synthetic_video)

        def _bad_probe(path):
            raise RuntimeError("simulated probe failure")

        monkeypatch.setattr(pipeline_v2_mod.media_mod, "probe", _bad_probe)

        with pytest.raises(RuntimeError, match="simulated probe failure"):
            analyze_project(project_id)

        db_session.expire_all()
        project = db_session.get(VideoProject, project_id)
        assert project.status == ProjectStatus.failed
        assert "simulated probe failure" in project.error_message

    def test_analyze_source_not_found_raises(self, db_session, db_engine):
        """Missing local file raises FileNotFoundError; project ends failed."""
        from app.pipeline_v2 import analyze_project

        project = VideoProject(
            id=str(uuid.uuid4()),
            source_type="upload",
            source_ref="/nonexistent/video.mp4",
            source_local_path="/nonexistent/video.mp4",
            status=ProjectStatus.created,
        )
        db_session.add(project)
        db_session.commit()
        project_id = project.id

        with pytest.raises(FileNotFoundError):
            analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            p = s.get(VideoProject, project_id)
            assert p.status == ProjectStatus.failed


# ---------------------------------------------------------------------------
# Tests: hook_split segmentation (localisation — docs/localisation.md §5)
# ---------------------------------------------------------------------------


class TestHookSplitSegments:
    """Unit tests for the partition itself, with no video and no DB.

    _hook_split_segments reads a clock, not frames, so everything worth
    checking is arithmetic: the clamp at both ends, and whether the trailing
    keep segment exists.
    """

    # Smallest min_clip_sec in the model registry — the clamp floor.
    FLOOR = min(s.min_clip_sec for s in ai_models.AI_MODELS.values())

    def _split(self, **kw):
        from app.pipeline_v2 import _hook_split_segments

        kw.setdefault("max_segment_sec", 10.0)
        return _hook_split_segments(**kw)

    def test_hook_shorter_than_video_gives_swap_then_keep(self):
        segs = self._split(duration_sec=25.0, hook_sec=8.0)
        assert [(s.action, s.start_sec, s.end_sec) for s in segs] == [
            ("swap", 0.0, 8.0),
            ("keep", 8.0, 25.0),
        ]
        # Nothing was detected — the hook boundary is an editorial choice.
        assert all(s.has_face is False for s in segs)

    def test_hook_equal_to_video_omits_the_keep_segment(self):
        segs = self._split(duration_sec=9.0, hook_sec=9.0, max_segment_sec=10.0)
        assert [(s.action, s.start_sec, s.end_sec) for s in segs] == [
            ("swap", 0.0, 9.0),
        ]

    def test_hook_longer_than_video_is_clamped_to_the_video(self):
        """A 30s hook on a 9s upload is the whole clip, not a segment that runs
        off the end of it."""
        segs = self._split(duration_sec=9.0, hook_sec=30.0, max_segment_sec=30.0)
        assert [(s.action, s.start_sec, s.end_sec) for s in segs] == [
            ("swap", 0.0, 9.0),
        ]

    def test_hook_longer_than_the_cap_is_clamped_to_the_cap(self):
        """The segmentation cap decides what the models can generate; the hook
        does not get to escape it. The remainder becomes the keep segment."""
        segs = self._split(duration_sec=25.0, hook_sec=22.0, max_segment_sec=10.0)
        assert [(s.action, s.start_sec, s.end_sec) for s in segs] == [
            ("swap", 0.0, 10.0),
            ("keep", 10.0, 25.0),
        ]

    def test_a_larger_cap_lets_the_hook_through(self):
        segs = self._split(duration_sec=40.0, hook_sec=22.0, max_segment_sec=30.0)
        assert [(s.action, s.start_sec, s.end_sec) for s in segs] == [
            ("swap", 0.0, 22.0),
            ("keep", 22.0, 40.0),
        ]

    def test_hook_below_the_model_floor_is_clamped_up(self):
        """A 0.5s swap clip is rejected by the backend at submit time — clamping
        up yields a hook that can actually be generated."""
        segs = self._split(duration_sec=25.0, hook_sec=0.5)
        assert segs[0].end_sec == pytest.approx(self.FLOOR)
        assert segs[1].start_sec == pytest.approx(self.FLOOR)

    def test_floor_never_overruns_a_very_short_source(self):
        """The duration bound wins over the floor: a 1s upload yields a 1s hook,
        not a segment longer than the video."""
        segs = self._split(duration_sec=1.0, hook_sec=0.1)
        assert [(s.action, s.start_sec, s.end_sec) for s in segs] == [
            ("swap", 0.0, 1.0),
        ]

    @pytest.mark.parametrize("empty", [None, 0.0])
    def test_unset_hook_uses_the_configured_default(self, empty):
        from app.config import settings

        segs = self._split(
            duration_sec=25.0, hook_sec=empty, max_segment_sec=30.0
        )
        assert segs[0].end_sec == pytest.approx(settings.LOCALISATION_DEFAULT_HOOK_SEC)

    def test_segments_cover_the_whole_video_without_gaps(self):
        segs = self._split(duration_sec=25.0, hook_sec=7.0)
        assert segs[0].start_sec == 0.0
        assert segs[-1].end_sec == pytest.approx(25.0)
        for prev, nxt in zip(segs, segs[1:]):
            assert prev.end_sec == pytest.approx(nxt.start_sec)

    def test_default_localisation_hook_fits_seedance_fast(self):
        """Localisation defaults to seedance-fast (15s). Under the universal
        segmentation cap the hook clamp must never propose a swap longer than
        that ceiling — otherwise the New Run default would launch a guaranteed
        source-fallback segment."""
        from app.config import settings
        from app.project_types import LOCALISATION, spec_for

        default_model = spec_for(LOCALISATION).default_model
        assert default_model == "seedance-fast"
        fast_ceiling = ai_models.max_clip_sec(default_model)
        assert fast_ceiling == 15.0

        # NULL project cap → universal 10s; requested hook 15s is clamped.
        segs = self._split(
            duration_sec=40.0,
            hook_sec=15.0,
            max_segment_sec=ai_models.UNIVERSAL_MAX_SEGMENT_SEC,
        )
        swap = [s for s in segs if s.action == "swap"]
        assert len(swap) == 1
        assert swap[0].end_sec - swap[0].start_sec <= fast_ceiling + 0.05
        assert swap[0].end_sec == pytest.approx(ai_models.UNIVERSAL_MAX_SEGMENT_SEC)

        # Configured default hook (10s) is itself under Fast's ceiling.
        assert settings.LOCALISATION_DEFAULT_HOOK_SEC <= fast_ceiling


class TestHookSplitPlan:
    """The clamp as data — the thing the project page shows the operator.

    Regression: the clamp was correct but INVISIBLE. A project created with
    hook_sec=15 and the default 10s segmentation cap was cut swap[0,10] while
    the row still said 15, and nothing outside the worker log distinguished
    "your hook is 10s because you asked for 10s" from "your hook is 10s because
    your cap is 10s" — two states that want opposite fixes. hook_split_plan is
    what makes the difference reportable, so the *reason* is under test, not
    just the number.
    """

    FLOOR = min(s.min_clip_sec for s in ai_models.AI_MODELS.values())

    def _plan(self, **kw):
        from app.pipeline_v2 import hook_split_plan

        kw.setdefault("max_segment_sec", 10.0)
        kw.setdefault("duration_sec", 25.0)
        return hook_split_plan(**kw)

    def test_hook_that_fits_reports_no_clamp(self):
        from app.pipeline_v2 import HOOK_CLAMP_NONE

        plan = self._plan(hook_sec=8.0)
        assert plan["requested"] == pytest.approx(8.0)
        assert plan["effective"] == pytest.approx(8.0)
        assert plan["reason"] == HOOK_CLAMP_NONE
        assert plan["defaulted"] is False

    def test_the_segmentation_cap_is_named_as_the_reason(self):
        """The reported bug: hook 15, default cap 10 → a 10s cut."""
        from app.pipeline_v2 import HOOK_CLAMP_CAP

        plan = self._plan(hook_sec=15.0, max_segment_sec=10.0, duration_sec=25.0)
        assert plan["effective"] == pytest.approx(10.0)
        assert plan["reason"] == HOOK_CLAMP_CAP
        assert plan["cap"] == pytest.approx(10.0)

    def test_a_video_shorter_than_the_hook_is_named_as_the_reason(self):
        from app.pipeline_v2 import HOOK_CLAMP_DURATION

        plan = self._plan(hook_sec=20.0, max_segment_sec=30.0, duration_sec=3.1)
        assert plan["effective"] == pytest.approx(3.1)
        assert plan["reason"] == HOOK_CLAMP_DURATION

    def test_a_hook_below_the_model_floor_is_padded_up_and_says_so(self):
        from app.pipeline_v2 import HOOK_CLAMP_FLOOR

        plan = self._plan(hook_sec=0.5)
        assert plan["effective"] == pytest.approx(self.FLOOR)
        assert plan["reason"] == HOOK_CLAMP_FLOOR
        assert plan["floor"] == pytest.approx(self.FLOOR)

    def test_when_both_bounds_bind_the_video_is_named(self):
        """cap == duration: blaming the cap would send the operator off to raise
        a setting that would change nothing."""
        from app.pipeline_v2 import HOOK_CLAMP_DURATION

        plan = self._plan(hook_sec=30.0, max_segment_sec=10.0, duration_sec=10.0)
        assert plan["reason"] == HOOK_CLAMP_DURATION

    def test_unknown_duration_is_bounded_by_the_cap_alone(self):
        """The page renders before analysis has probed the video; the cap can
        still clamp, but the duration cannot be blamed for something unknown."""
        from app.pipeline_v2 import HOOK_CLAMP_CAP

        plan = self._plan(hook_sec=15.0, max_segment_sec=10.0, duration_sec=None)
        assert plan["duration"] is None
        assert plan["effective"] == pytest.approx(10.0)
        assert plan["reason"] == HOOK_CLAMP_CAP

    @pytest.mark.parametrize("empty", [None, 0.0])
    def test_an_unset_hook_is_flagged_as_the_default(self, empty):
        from app.config import settings

        plan = self._plan(hook_sec=empty, max_segment_sec=30.0)
        assert plan["defaulted"] is True
        assert plan["requested"] == pytest.approx(
            settings.LOCALISATION_DEFAULT_HOOK_SEC
        )

    @pytest.mark.parametrize(
        "hook,cap,duration",
        [
            (8.0, 10.0, 25.0),    # fits
            (15.0, 10.0, 25.0),   # capped
            (20.0, 30.0, 3.1),    # shorter video
            (0.5, 10.0, 25.0),    # below the floor
            (None, 10.0, 25.0),   # default
            (25.0, 30.0, 25.0),   # hook covers the whole video
        ],
    )
    def test_the_plan_is_what_the_segments_actually_are(self, hook, cap, duration):
        """The whole point of extracting the clamp: what the page says the hook
        is must be the boundary the segmentation really lays down. If these two
        ever disagree the UI is confidently lying about the rows underneath it.
        """
        from app.pipeline_v2 import _hook_split_segments, hook_split_plan

        plan = hook_split_plan(
            duration_sec=duration, hook_sec=hook, max_segment_sec=cap
        )
        segs = _hook_split_segments(
            duration_sec=duration, hook_sec=hook, max_segment_sec=cap
        )
        swap_end = max(s.end_sec for s in segs if s.action == "swap")
        assert swap_end == pytest.approx(plan["effective"])


class TestEffectiveSegmentCap:
    """The cap resolution analysis applies, exposed for the project page."""

    def test_null_resolves_to_the_universal_default(self):
        from app.pipeline_v2 import effective_segment_cap_sec

        assert effective_segment_cap_sec(None) == pytest.approx(
            ai_models.UNIVERSAL_MAX_SEGMENT_SEC
        )

    def test_a_cap_above_every_model_is_bounded(self):
        from app.pipeline_v2 import effective_segment_cap_sec

        assert effective_segment_cap_sec(9999.0) == pytest.approx(
            ai_models.ABSOLUTE_MAX_SEGMENT_SEC
        )

    def test_zero_is_a_value_not_an_unset(self):
        """`or` would redraw a broken 0.0 row as the 10s default and hide it;
        analyze_project is equally scrupulous about this."""
        from app.pipeline_v2 import effective_segment_cap_sec

        assert effective_segment_cap_sec(0.0) == pytest.approx(0.0)


class TestAnalyzeHookSplit:
    """The same partition, end to end through analyze_project."""

    def test_localisation_project_is_segmented_hook_then_keep(
        self, db_engine, db_session, synthetic_video
    ):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(
            db_session, synthetic_video, project_type="localisation", hook_sec=6.0
        )

        analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.status == ProjectStatus.ready
            seg_defs = sorted(project.segments, key=lambda sd: sd.index)
            assert [(sd.action, sd.start_sec) for sd in seg_defs] == [
                ("swap", 0.0),
                ("keep", 6.0),
            ]
            assert seg_defs[0].end_sec == pytest.approx(6.0)
            assert seg_defs[1].end_sec == pytest.approx(_VIDEO_DURATION, abs=1.0)

    def test_null_hook_sec_falls_back_to_the_default(
        self, db_engine, db_session, synthetic_video
    ):
        from app.config import settings
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(
            db_session, synthetic_video, project_type="localisation"
        )

        analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            seg_defs = sorted(
                s.get(VideoProject, project_id).segments, key=lambda sd: sd.index
            )
        assert len(seg_defs) == 2
        assert seg_defs[0].end_sec == pytest.approx(
            settings.LOCALISATION_DEFAULT_HOOK_SEC
        )

    def test_hook_covering_the_whole_video_yields_one_segment(
        self, db_engine, db_session, synthetic_video
    ):
        from app.pipeline_v2 import analyze_project

        # Cap raised above the source duration so the hook is bounded by the
        # video rather than by the segmentation cap.
        project_id = _create_project(
            db_session,
            synthetic_video,
            project_type="localisation",
            hook_sec=999.0,
            max_segment_sec=30.0,
        )

        analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            seg_defs = list(project.segments)
            assert len(seg_defs) == 1
            assert seg_defs[0].action == "swap"
            assert seg_defs[0].end_sec == pytest.approx(project.duration_sec)

    def test_no_face_detection_runs_for_a_localisation_project(
        self, db_session, synthetic_video, monkeypatch
    ):
        """hook_split must not touch InsightFace — the boundary is a number."""
        import app.pipeline_v2 as pipeline_v2_mod
        from app.pipeline_v2 import analyze_project

        def _boom(*a, **kw):
            raise AssertionError("propose_segments must not be called")

        monkeypatch.setattr(pipeline_v2_mod.face_mod, "propose_segments", _boom)

        project_id = _create_project(
            db_session, synthetic_video, project_type="localisation"
        )
        analyze_project(project_id)


# ---------------------------------------------------------------------------
# Tests: transcribe_project (localisation side channel — §8)
# ---------------------------------------------------------------------------


_TRANSCRIPT = {
    "schema_version": 1,
    "model": "gemini-2.5-pro",
    "prompt_version": "transcribe/v1",
    "created_at": "2026-08-09T12:00:00+00:00",
    "source_language": "en",
    "lines": [
        {
            "id": 1,
            "start": 0.0,
            "end": 2.4,
            "speaker": "the woman in the red jacket",
            "on_screen": True,
            "text": "Hey, what are you doing?",
        }
    ],
    "on_screen_text": [],
}


class TestTranscribeProject:
    def _patch_transcribe(self, monkeypatch, result):
        """Patch localisation.transcribe_video; *result* is a dict or an
        exception instance to raise. Returns the call log."""
        import app.pipeline_v2 as pipeline_v2_mod

        calls = []

        def _fake(video_path, *, kie=None, model=None, spend_scope=None):
            calls.append(video_path)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(
            pipeline_v2_mod.localisation_mod, "transcribe_video", _fake
        )
        return calls

    def _project(self, db_session, synthetic_video, **kw):
        return _create_project(
            db_session, synthetic_video, project_type="localisation", **kw
        )

    def test_ready_when_the_model_returns_lines(
        self, db_engine, db_session, synthetic_video, monkeypatch
    ):
        from app.pipeline_v2 import transcribe_project

        calls = self._patch_transcribe(monkeypatch, _TRANSCRIPT)
        project_id = self._project(db_session, synthetic_video)

        transcribe_project(project_id)

        assert calls == [synthetic_video]
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.transcript_status == "ready"
            assert project.transcript_error is None
            assert project.transcript["source_language"] == "en"
            assert len(project.transcript["lines"]) == 1
            # The side channel touches nothing the pipeline depends on.
            assert project.status == ProjectStatus.created
            assert project.error_message is None

    def test_empty_when_the_model_hears_no_speech(
        self, db_engine, db_session, synthetic_video, monkeypatch
    ):
        """A wordless hook is a legal outcome, not a failure."""
        from app.pipeline_v2 import transcribe_project

        silent = dict(_TRANSCRIPT, lines=[])
        self._patch_transcribe(monkeypatch, silent)
        project_id = self._project(db_session, synthetic_video)

        transcribe_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.transcript_status == "empty"
            assert project.transcript_error is None
            # Stored anyway — the detected language is still useful.
            assert project.transcript["lines"] == []

    def test_failed_records_the_message_and_does_not_raise(
        self, db_engine, db_session, synthetic_video, monkeypatch
    ):
        from app.localisation import LocalisationError
        from app.pipeline_v2 import transcribe_project

        self._patch_transcribe(
            monkeypatch, LocalisationError("transcribe: kie.ai 401 invalid key")
        )
        project_id = self._project(db_session, synthetic_video)

        transcribe_project(project_id)  # must NOT raise

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.transcript_status == "failed"
            assert "401" in project.transcript_error
            # §8: a failed transcript leaves the project fully usable.
            assert project.status == ProjectStatus.created
            assert project.error_message is None

    def test_failure_keeps_a_previously_good_transcript(
        self, db_engine, db_session, synthetic_video, monkeypatch
    ):
        from app.localisation import LocalisationError
        from app.pipeline_v2 import transcribe_project

        project_id = self._project(db_session, synthetic_video)
        self._patch_transcribe(monkeypatch, _TRANSCRIPT)
        transcribe_project(project_id)

        self._patch_transcribe(monkeypatch, LocalisationError("gemini is down"))
        transcribe_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.transcript_status == "failed"
            assert project.transcript["lines"], "the good transcript was lost"

    def test_unexpected_error_is_recorded_too(
        self, db_engine, db_session, synthetic_video, monkeypatch
    ):
        from app.pipeline_v2 import transcribe_project

        self._patch_transcribe(monkeypatch, RuntimeError("boom"))
        project_id = self._project(db_session, synthetic_video)

        transcribe_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.transcript_status == "failed"
            assert "boom" in project.transcript_error

    def test_missing_source_file_fails_without_calling_the_model(
        self, db_engine, db_session, monkeypatch
    ):
        from app.pipeline_v2 import transcribe_project

        calls = self._patch_transcribe(monkeypatch, _TRANSCRIPT)
        project = VideoProject(
            id=str(uuid.uuid4()),
            source_type="upload",
            source_ref="/nonexistent/video.mp4",
            source_local_path="/nonexistent/video.mp4",
            project_type="localisation",
            status=ProjectStatus.ready,
        )
        db_session.add(project)
        db_session.commit()

        transcribe_project(project.id)

        assert calls == []
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            assert s.get(VideoProject, project.id).transcript_status == "failed"

    def test_rerun_is_idempotent(
        self, db_engine, db_session, synthetic_video, monkeypatch
    ):
        """Re-running is how the operator retries — it must converge, not
        accumulate."""
        from app.pipeline_v2 import transcribe_project

        calls = self._patch_transcribe(monkeypatch, _TRANSCRIPT)
        project_id = self._project(db_session, synthetic_video)

        transcribe_project(project_id)
        transcribe_project(project_id)

        assert len(calls) == 2
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.transcript_status == "ready"
            assert len(project.transcript["lines"]) == 1

    def test_recovers_from_a_previous_failure(
        self, db_engine, db_session, synthetic_video, monkeypatch
    ):
        from app.localisation import LocalisationError
        from app.pipeline_v2 import transcribe_project

        project_id = self._project(db_session, synthetic_video)
        self._patch_transcribe(monkeypatch, LocalisationError("timeout"))
        transcribe_project(project_id)

        self._patch_transcribe(monkeypatch, _TRANSCRIPT)
        transcribe_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.transcript_status == "ready"
            assert project.transcript_error is None

    def test_missing_project_raises(self, db_session):
        """A scheduling bug, unlike every provider failure above."""
        from app.pipeline_v2 import transcribe_project

        with pytest.raises(ValueError, match="VideoProject not found"):
            transcribe_project(str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Tests: process_run
# ---------------------------------------------------------------------------


class TestProcessRun:
    def _setup_project_and_run(
        self,
        db_session,
        db_engine,
        synthetic_video,
        patch_propose,
        *,
        gdrive_folder_id: str | None = None,
        reference_image_urls: list | None = None,
    ):
        """
        Run analyze_project to get a ready project, then create a queued run.
        Returns (project_id, run_id).
        """
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)

        run_id = _create_run(
            db_session,
            project_id,
            gdrive_folder_id=gdrive_folder_id,
            reference_image_urls=reference_image_urls,
        )
        return project_id, run_id

    def test_process_happy_path(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """All swap RunSegments reach completed; run ends done; stitch produces a file."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup_project_and_run(
            db_session, db_engine, synthetic_video, patch_propose
        )
        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done, f"Expected done, got {run.status}"
            assert run.result_local_path is not None
            assert os.path.exists(run.result_local_path)

            run_segs = list(run.run_segments)
            assert len(run_segs) == 2  # two swap segments in our fixed partition
            assert all(rs.status == SegmentStatus.completed for rs in run_segs)
            for rs in run_segs:
                assert rs.local_result_path is not None
                assert os.path.exists(rs.local_result_path)

        # Two swap segments → two clip uploads + no ref images → 2 total upload calls
        assert len(fake_kie.upload_calls) == 2
        assert len(fake_kie.create_task_calls) == 2
        assert len(fake_kie.poll_calls) == 2

    def test_clip_uploads_are_namespaced_per_run_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """Regression: clip uploads must carry a per-run_segment remote file_name.

        Clip basenames are index-based (clip_0001.mp4) and identical across every
        Run, so an un-namespaced upload path collides when two Runs process the
        same segment index concurrently — one Run's clip overwrites the other's
        at the shared remote path and Seedance fetches the wrong input. The
        remote file_name must be prefixed with the run_segment id so it is
        globally unique. (See Run e5c5682f: result_0001 came from another Run.)
        """
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup_project_and_run(
            db_session, db_engine, synthetic_video, patch_propose
        )
        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs_ids = {rs.id for rs in s.get(Run, run_id).run_segments}

        clip_records = [
            r for r in fake_kie.upload_records
            if r["upload_path"] == "charswap/segments"
        ]
        assert clip_records, "expected clip uploads to charswap/segments"

        remote_names = []
        for rec in clip_records:
            fname = rec["file_name"]
            # Must be explicitly namespaced (not defaulting to the bare basename).
            assert fname is not None, "clip upload must pass an explicit file_name"
            assert fname != os.path.basename(rec["local_path"]), (
                f"clip upload file_name {fname!r} is the bare basename — "
                "this collides across Runs"
            )
            # The namespace must be a real run_segment id from this run.
            assert any(fname.startswith(rid) for rid in rs_ids), (
                f"clip upload file_name {fname!r} is not prefixed by a "
                f"run_segment id {rs_ids}"
            )
            remote_names.append(fname)

        # No two segments may share a remote object name.
        assert len(remote_names) == len(set(remote_names)), (
            f"clip uploads collide on remote name: {remote_names}"
        )

    def test_threaded_submit_sees_new_run_segments_without_ffmpeg(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Fresh RunSegments are committed before submit threads open sessions."""
        from app.media import MediaInfo
        import app.pipeline_v2 as pipeline_v2_mod
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")

        project_id = _create_project(
            db_session,
            str(source),
            status=ProjectStatus.ready,
        )
        segs = [
            SegmentDef(
                project_id=project_id,
                index=0,
                start_sec=0.0,
                end_sec=5.0,
                has_face=True,
                action="swap",
            ),
            SegmentDef(
                project_id=project_id,
                index=1,
                start_sec=5.0,
                end_sec=10.0,
                has_face=True,
                action="swap",
            ),
        ]
        db_session.add_all(segs)
        run_id = _create_run(db_session, project_id)

        monkeypatch.setattr(
            pipeline_v2_mod.media_mod,
            "probe",
            lambda _path: MediaInfo(
                duration_sec=10.0,
                width=320,
                height=240,
                fps=25.0,
                aspect_ratio="4:3",
                has_audio=True,
            ),
        )
        monkeypatch.setattr(
            pipeline_v2_mod.media_mod,
            "cut_clip",
            lambda _src, _start, _end, dst, **_kw: (os.makedirs(os.path.dirname(dst), exist_ok=True), open(dst, "wb").write(b"clip")),
        )
        monkeypatch.setattr(
            pipeline_v2_mod.media_mod,
            "stitch",
            lambda _clips, audio_source, dst, **_kw: (os.makedirs(os.path.dirname(dst), exist_ok=True), open(dst, "wb").write(b"final")),
        )

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            run_segments = (
                s.query(RunSegment)
                .filter(RunSegment.run_id == run_id)
                .order_by(RunSegment.index)
                .all()
            )
            assert len(run_segments) == 2
            assert all(rs.status == SegmentStatus.completed for rs in run_segments)

        assert len(fake_kie.create_task_calls) == 2

    def test_process_gdrive_upload_called_when_folder_set(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """When gdrive_folder_id is set on the run, upload is called after stitching."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup_project_and_run(
            db_session,
            db_engine,
            synthetic_video,
            patch_propose,
            gdrive_folder_id="folder-abc123",
        )
        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        assert len(fake_gdrive.upload_calls) == 1
        call = fake_gdrive.upload_calls[0]
        assert call["folder_id"] == "folder-abc123"

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.result_gdrive_file_id is not None
            assert run.status == RunStatus.done

    def test_resumability_skips_completed_run_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        Pre-mark one swap RunSegment as completed with a real result file.
        process_run must NOT call upload_file or create_task for it.
        """
        from app.pipeline_v2 import analyze_project, process_run
        from app.storage import run_results_dir

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        # Load the project's segment defs and manually create RunSegments.
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            seg_defs = list(project.segments)
            swap_defs = [sd for sd in seg_defs if sd.action == "swap"]

            # Create RunSegments for all swap defs.
            for sd in swap_defs:
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                )
                s.add(rs)
            s.commit()

        # Now pre-mark the FIRST swap RunSegment as completed with a real file.
        r_dir = run_results_dir(run_id, project_id)
        with Session() as s:
            run = s.get(Run, run_id)
            run_segs = sorted(run.run_segments, key=lambda rs: rs.index)
            first_rs = run_segs[0]

            fake_result = os.path.join(r_dir, f"result_{first_rs.index:04d}.mp4")
            shutil.copy2(synthetic_video, fake_result)

            first_rs.status = SegmentStatus.completed
            first_rs.local_result_path = fake_result
            first_rs.seedance_task_id = "pre-existing-task"
            first_rs.seedance_result_url = "https://fake/pre-existing.mp4"
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        # Only the second swap segment was actually processed.
        assert len(fake_kie.upload_calls) == 1, (
            f"Expected 1 upload call (only the non-resumed segment), "
            f"got {fake_kie.upload_calls}"
        )
        assert len(fake_kie.create_task_calls) == 1
        assert len(fake_kie.poll_calls) == 1

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done

    def test_retry_resets_failed_run_segment_and_completes(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        Regression: a RunSegment left in `failed` by a prior run must be reset
        to `pending` on retry — not crash with 'Invalid transition: failed → uploading'.
        """
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            seg_defs = list(project.segments)
            swap_defs = [sd for sd in seg_defs if sd.action == "swap"]

            # Create RunSegments and force the first one to failed.
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.failed if i == 0 else SegmentStatus.pending,
                    error_message="old failure" if i == 0 else None,
                    seedance_task_id="stale-task" if i == 0 else None,
                )
                s.add(rs)
            s.commit()

        # The stale task reports `fail` so the resume check resubmits the
        # segment (exercising the failed→pending reset) rather than recovering it
        # via the no-rebill path. Its resubmitted task (a new id) then succeeds.
        fake_kie = FakeKieClient(synthetic_video, fail_task_id="stale-task")
        fake_gdrive = FakeGDriveClient()

        # Must not raise InvalidTransition.
        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            for rs in run.run_segments:
                assert rs.status == SegmentStatus.completed
                assert rs.error_message is None

        # Both segments were processed (the failed one was reset and re-processed).
        assert len(fake_kie.create_task_calls) == 2

    def test_progress_is_committed_during_processing(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        process_run must COMMIT intermediate state so a separate reader (the API/UI)
        sees live progress — not stay 'queued / 0 done' until the whole run finishes.
        """
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        observed = {}

        class ObservingKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                # Read from a SEPARATE session/connection mid-processing.
                with sessionmaker(bind=db_engine)() as s2:
                    r = s2.get(Run, run_id)
                    observed["run_status"] = r.status
                    observed["generating"] = [
                        rs.index
                        for rs in r.run_segments
                        if rs.status == SegmentStatus.generating
                    ]
                return super().get_task(task_id, **kw)

        process_run(
            run_id, kie=ObservingKie(synthetic_video), gdrive=FakeGDriveClient()
        )

        # While the first segment was generating, a separate session saw the run
        # already in 'processing' and at least one RunSegment marked 'generating'.
        assert observed.get("run_status") == RunStatus.processing
        assert len(observed.get("generating", [])) >= 1

    def test_segment_permanent_failure_marks_run_incomplete(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        When ONE swap segment fails on every attempt (after retries), the run is
        marked `incomplete` and NOT stitched — we never mix a generated video
        with the original clip. The other segment still completes, its result is
        kept for the eventual re-stitch, and result_local_path stays None.
        """
        from app.pipeline_v2 import analyze_project, process_run
        import app.pipeline_v2 as pipeline_v2_mod

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        class FailSegmentZeroKie(FakeKieClient):
            """Every task generated from segment 0's clip (clip_0000) fails on
            every attempt; all other segments succeed."""

            def __init__(self, src):
                super().__init__(src)
                self._fail_tasks: set[str] = set()

            def create_task(self, *, reference_video_urls, **kw) -> str:
                task_id = super().create_task(
                    reference_video_urls=reference_video_urls, **kw
                )
                if reference_video_urls and "clip_0000" in reference_video_urls[0]:
                    self._fail_tasks.add(task_id)
                return task_id

            def get_task(self, task_id: str, **kw) -> dict:
                self.poll_calls.append(task_id)
                if task_id in self._fail_tasks:
                    return {"state": "fail", "failMsg": "injected failure for test"}
                url = f"https://fake/results/{task_id}.mp4"
                return {"state": "success",
                        "resultJson": json.dumps({"resultUrls": [url]})}

        fake_kie = FailSegmentZeroKie(synthetic_video)
        fake_gdrive = FakeGDriveClient()

        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.incomplete
            assert run.result_local_path is None  # NOT stitched
            assert "did not complete" in (run.error_message or "")

            failed = [rs for rs in run.run_segments if rs.status == SegmentStatus.failed]
            completed = [rs for rs in run.run_segments if rs.status == SegmentStatus.completed]
            assert len(failed) == 1
            assert len(completed) == 1
            # The completed segment's result is preserved for the re-stitch.
            assert completed[0].local_result_path is not None

        # Segment 0 was retried up to the configured attempt cap before failing.
        assert len(fake_kie._fail_tasks) == pipeline_v2_mod.RUN_TASK_MAX_ATTEMPTS

    def test_transient_failure_retried_then_run_completes(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A segment whose FIRST task fails transiently is resubmitted; when the
        retry succeeds the run completes normally (done) with a stitched video."""
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        class FailFirstTaskKie(FakeKieClient):
            """Only the very first task id fails; its resubmission (a new id)
            succeeds — exercising the retry-recovery path."""

            def __init__(self, src):
                super().__init__(src)
                self._first_task: str | None = None

            def create_task(self, **kw) -> str:
                task_id = super().create_task(**kw)
                if self._first_task is None:
                    self._first_task = task_id
                return task_id

            def get_task(self, task_id: str, **kw) -> dict:
                self.poll_calls.append(task_id)
                if task_id == self._first_task:
                    return {"state": "fail", "failMsg": "Internal Error, Please try again later."}
                url = f"https://fake/results/{task_id}.mp4"
                return {"state": "success",
                        "resultJson": json.dumps({"resultUrls": [url]})}

        fake_kie = FailFirstTaskKie(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_local_path and os.path.exists(run.result_local_path)
            assert all(
                rs.status == SegmentStatus.completed for rs in run.run_segments
            )
        # One extra create_task beyond the 2 segments = the single retry.
        assert len(fake_kie.create_task_calls) == 3

    def test_dead_resource_failure_reuploads_clip_before_retrying(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """kie.ai validates an uploaded clip_url asynchronously, so a stale URL
        (e.g. reused from a checkpoint by crash-resume or a manual Retry Run
        long after the original upload) only surfaces as a `fail` on poll — with
        a message like "The parameter `content[6].video_url` ... is not valid:
        resource not found." That failure is not transient: resubmitting the
        same recipe would repeat it identically on every remaining attempt (run
        68eb3923, 2026-08-31). The fix re-uploads the clip fresh before
        resubmitting whenever the failure names a missing resource."""
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        class DeadResourceFirstTaskKie(FakeKieClient):
            """The first task fails as if its clip_url had expired on kie.ai's
            side; the resubmission (after a fresh re-upload) succeeds."""

            def __init__(self, src):
                super().__init__(src)
                self._first_task: str | None = None

            def create_task(self, **kw) -> str:
                task_id = super().create_task(**kw)
                if self._first_task is None:
                    self._first_task = task_id
                return task_id

            def get_task(self, task_id: str, **kw) -> dict:
                self.poll_calls.append(task_id)
                if task_id == self._first_task:
                    return {
                        "state": "fail",
                        "failMsg": (
                            "The parameter `content[6].video_url` specified in "
                            "the request is not valid: resource not found."
                        ),
                    }
                url = f"https://fake/results/{task_id}.mp4"
                return {"state": "success",
                        "resultJson": json.dumps({"resultUrls": [url]})}

        fake_kie = DeadResourceFirstTaskKie(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_local_path and os.path.exists(run.result_local_path)
            assert all(
                rs.status == SegmentStatus.completed for rs in run.run_segments
            )

        # 2 segments' initial uploads, plus one more re-upload triggered by the
        # dead-resource failure before its retry (not just a bare resubmit of
        # the same stale URL).
        assert len(fake_kie.upload_calls) == 3
        assert len(fake_kie.create_task_calls) == 3
        assert "_retry2_" in fake_kie.upload_records[-1]["file_name"]

    def test_fail_with_numeric_failcode_and_no_failmsg_does_not_crash_the_poll(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """kie.ai's `fail` state carries `failMsg` OR a bare numeric `failCode`
        (see test_kie_client.py's `failCode: 4001` fixture) — `poll_run_once_internal`
        falls back to the int when `failMsg` is absent, so the reason handed to
        `_resubmit_segment_task` is not always a str. The dead-resource-URL check
        added there must not assume `reason` is a string, or a fail like this one
        crashes the whole poll round (not just this segment) with an
        AttributeError on `.lower()`."""
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        class NumericFailCodeKie(FakeKieClient):
            """Every task fails with only a numeric failCode, no failMsg —
            exercising the non-str `reason` path with no crash and normal
            attempt-exhaustion behaviour."""

            def get_task(self, task_id: str, **kw) -> dict:
                self.poll_calls.append(task_id)
                return {"state": "fail", "failCode": 4001}

        fake_kie = NumericFailCodeKie(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.incomplete
            assert all(
                rs.status == SegmentStatus.failed for rs in run.run_segments
            )

    def test_run_segments_created_idempotently(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        If RunSegments already exist (e.g. from a prior call), process_run must not
        create duplicates — the ensure-RunSegment step is idempotent.
        """
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        # Pre-create RunSegments manually (simulating a prior incomplete call).
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            for sd in project.segments:
                if sd.action == "swap":
                    rs = RunSegment(
                        run_id=run_id,
                        segment_def_id=sd.id,
                        index=sd.index,
                        status=SegmentStatus.pending,
                    )
                    s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session2 = sessionmaker(bind=db_engine)
        with Session2() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            # Should still be exactly 2 RunSegments (not 4).
            assert len(run.run_segments) == 2

    def test_process_run_missing_project_not_found(self, db_session):
        """process_run raises ValueError for a non-existent run."""
        from app.pipeline_v2 import process_run

        with pytest.raises(ValueError, match="Run not found"):
            process_run(str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Tests: TR6 per-segment override
# ---------------------------------------------------------------------------


class TestSegmentOverride:
    """
    Tests for per-segment prompt_override and reference_image_urls_override.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, prompt="run-level prompt")
        return project_id, run_id

    def test_prompt_override_used_for_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A RunSegment with prompt_override causes create_task to use that prompt."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        Session = sessionmaker(bind=db_engine)
        # Pre-create RunSegments; set prompt_override on the first one.
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = [sd for sd in project.segments if sd.action == "swap"]
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                    prompt_override="override prompt for seg 0" if i == 0 else None,
                )
                s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        from app.pipeline_v2 import process_run
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Check that two create_task calls were made and the first used the override.
        assert len(fake_kie.create_task_records) == 2
        prompts = {r["prompt"] for r in fake_kie.create_task_records}
        assert "override prompt for seg 0" in prompts
        assert "run-level prompt" in prompts

    def test_reference_override_used_for_segment(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A RunSegment with reference_image_urls_override uses those URLs for create_task."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        Session = sessionmaker(bind=db_engine)
        override_refs = ["https://override-ref.example.com/img.jpg"]
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = [sd for sd in project.segments if sd.action == "swap"]
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                    reference_image_urls_override=override_refs if i == 0 else None,
                )
                s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        ref_sets = [frozenset(r["reference_image_urls"]) for r in fake_kie.create_task_records]
        # One record should include the override ref
        assert any("override-ref.example.com" in u for refs in ref_sets for u in refs)

    def test_single_segment_rerun_scenario(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        Simulate the full per-segment re-run flow:
        1. Pre-complete BOTH swap RunSegments (run is effectively done).
        2. Reset ONE to pending with a prompt_override.
        3. Call process_run → only that one is reprocessed (1 create_task call).
        4. Run ends done.
        """
        from app.pipeline_v2 import analyze_project, process_run
        from app.storage import run_results_dir

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, prompt="original prompt")

        r_dir = run_results_dir(run_id, project_id)

        Session = sessionmaker(bind=db_engine)

        # Pre-create RunSegments and mark both as completed.
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = sorted(
                [sd for sd in project.segments if sd.action == "swap"],
                key=lambda sd: sd.index,
            )
            rs_ids = []
            for sd in swap_defs:
                fake_result = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
                shutil.copy2(synthetic_video, fake_result)
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.completed,
                    local_result_path=fake_result,
                    seedance_task_id=f"pre-task-{sd.index}",
                    seedance_result_url=f"https://fake/pre-{sd.index}.mp4",
                )
                s.add(rs)
                s.flush()
                rs_ids.append(rs.id)
            s.commit()

        # Now reset the FIRST RunSegment to pending with an override prompt.
        with Session() as s:
            rs0 = s.get(RunSegment, rs_ids[0])
            rs0.status = SegmentStatus.pending
            rs0.error_message = None
            rs0.seedance_task_id = None
            rs0.seedance_result_url = None
            rs0.local_result_path = None
            rs0.prompt_override = "per-segment override prompt"
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Only ONE create_task call (the second was already completed, skipped).
        assert len(fake_kie.create_task_calls) == 1, (
            f"Expected 1 create_task (only the reset segment), "
            f"got {len(fake_kie.create_task_calls)}"
        )
        assert fake_kie.create_task_records[0]["prompt"] == "per-segment override prompt"

        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: per-segment GPT Image 2 regeneration — the three-way ref priority
# ---------------------------------------------------------------------------


def _data_file(name: str) -> str:
    """A real file under the storage root — submit only uploads local refs
    that resolve inside DATA_DIR (AICP-148)."""
    from app import storage

    d = os.path.join(storage.data_dir_root(), "fake-refs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "wb") as fh:
        fh.write(b"\xff\xd8fake")
    return path


class TestSegmentGeneratedImagePriority:
    """generated_reference_urls > reference_image_urls_override > run-level.

    See pipeline_v2.submit_run's reference-resolution block and
    RunSegment.generated_reference_urls's docstring in models.py.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(
            db_session, project_id, prompt="run-level prompt",
            reference_image_urls=[_data_file("run_character.jpg")],
        )
        return project_id, run_id

    def test_generated_reference_urls_wins_over_everything(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A segment with generated_reference_urls sends exactly that pair,
        ignoring its own hand override AND the run-level references."""
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        Session = sessionmaker(bind=db_engine)
        generated_pair = [_data_file("character.jpg"), _data_file("generated_output.png")]
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = [sd for sd in project.segments if sd.action == "swap"]
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                    generated_reference_urls=generated_pair if i == 0 else None,
                    reference_image_urls_override=(
                        ["https://hand-override.example.com/x.jpg"] if i == 0 else None
                    ),
                )
                s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        ref_sets = [frozenset(r["reference_image_urls"]) for r in fake_kie.create_task_records]

        # The generated segment's refs are EXACTLY the two-entry pair — never
        # the hand override, never the run-level set, never a splice of them.
        generated_set = next(
            s for s in ref_sets if any("generated_output" in u for u in s)
        )
        assert len(generated_set) == 2
        assert any("character.jpg" in u for u in generated_set)
        assert not any("hand-override" in u for u in generated_set)
        assert not any("run_character" in u for u in generated_set)

        # The OTHER (non-generated) segment still gets the run-level ref.
        other_set = next(s for s in ref_sets if s != generated_set)
        assert any("run_character" in u for u in other_set)

    def test_hand_override_wins_when_no_generated_image(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = [sd for sd in project.segments if sd.action == "swap"]
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id,
                    segment_def_id=sd.id,
                    index=sd.index,
                    status=SegmentStatus.pending,
                    reference_image_urls_override=(
                        ["https://hand-override.example.com/x.jpg"] if i == 0 else None
                    ),
                )
                s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        ref_sets = [frozenset(r["reference_image_urls"]) for r in fake_kie.create_task_records]
        assert any("hand-override" in u for refs in ref_sets for u in refs)

    def test_run_level_used_when_neither_set(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        from app.pipeline_v2 import process_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        ref_sets = [frozenset(r["reference_image_urls"]) for r in fake_kie.create_task_records]
        assert all(any("run_character" in u for u in refs) for refs in ref_sets)


# ---------------------------------------------------------------------------
# Tests: model routing (Seedance vs Gemini Omni)
# ---------------------------------------------------------------------------


class TestModelRouting:
    """process_run must route to the kie method matching run.model."""

    def _setup_ready_project(self, db_session, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        return project_id

    def test_seedance_routes_to_create_task(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A default (seedance) run uses create_task, not create_omni_task."""
        from app.pipeline_v2 import process_run

        project_id = self._setup_ready_project(db_session, synthetic_video, patch_propose)
        run_id = _create_run(db_session, project_id)  # model defaults to seedance

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        assert len(fake_kie.create_omni_records) == 0

    def test_gemini_routes_to_create_omni_task(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A gemini-omni run uses create_omni_task with snapped duration + video trim."""
        from app.pipeline_v2 import process_run

        project_id = self._setup_ready_project(db_session, synthetic_video, patch_propose)

        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Gemini Run",
            prompt="Replace the character",
            reference_image_urls=[],
            model="gemini-omni",
            resolution="720p",
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run.id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Two swap segments → two omni calls, no seedance calls.
        assert len(fake_kie.create_omni_records) == 2
        assert len(fake_kie.create_task_records) == 0

        for rec in fake_kie.create_omni_records:
            assert rec["resolution"] == "720p"
            # 4:3 synthetic video (320x240) → landscape aspect for Gemini.
            assert rec["aspect_ratio"] == "16:9"
            assert rec["duration"] in (4, 6, 8, 10)
            assert rec["video_start"] == 0
            # Trim never exceeds Gemini's 10s cap.
            assert 0 < rec["video_end"] <= 10.0

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            r = s.get(Run, run.id)
            assert r.status == RunStatus.done

    @pytest.mark.parametrize(
        "run_model, expected_kie_model",
        [
            ("seedance", "bytedance/seedance-2"),
            ("seedance-fast", "bytedance/seedance-2-fast"),
            ("seedance-mini", "bytedance/seedance-2-mini"),
            ("seedance-2-5", "bytedance/seedance-2-5"),
        ],
    )
    def test_seedance_variant_passes_correct_kie_model_id(
        self, db_engine, db_session, synthetic_video, patch_propose,
        run_model, expected_kie_model,
    ):
        """Each Seedance variant maps to its own kie.ai model id in create_task,
        while still routing through create_task (not create_omni_task)."""
        from app.pipeline_v2 import process_run

        project_id = self._setup_ready_project(db_session, synthetic_video, patch_propose)

        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Variant Run",
            prompt="Replace the character",
            reference_image_urls=[],
            model=run_model,
            resolution="720p",
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run.id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_omni_records) == 0
        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["model"] == expected_kie_model

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            r = s.get(Run, run.id)
            assert r.status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: Gemini Omni video-only clips + 10s limit
# ---------------------------------------------------------------------------


class TestGeminiOmniAudioAndLimit:
    """Gemini Omni clips are uploaded video-only (it fails on an audio track),
    capped at 10s, and always stitched with the original source audio.

    Media (probe / cut_clip / stitch) is fully mocked so these run without
    ffmpeg.
    """

    @staticmethod
    def _mock_media(monkeypatch, *, duration_sec, cut_calls, stitch_calls):
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(
                duration_sec=duration_sec, width=320, height=240,
                fps=25.0, aspect_ratio="4:3", has_audio=True,
            ),
        )

        def spy_cut_clip(_src, _start, _end, dst, *, include_audio=True,
                         fast_intermediate=False):
            cut_calls.append({
                "dst": dst,
                # Recorded so the per-model clip trim (pre/post-roll pushing the
                # cut past AIModelSpec.max_clip_sec) is directly observable.
                "start": _start,
                "end": _end,
                "include_audio": include_audio,
                "fast_intermediate": fast_intermediate,
            })
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"clip")

        monkeypatch.setattr(pv2.media_mod, "cut_clip", spy_cut_clip)

        def spy_stitch(_clips, audio_source, dst, **kw):
            stitch_calls.append(kw)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"final")

        monkeypatch.setattr(pv2.media_mod, "stitch", spy_stitch)

    @staticmethod
    def _make_run(
        db_session, project_id, *, model, audio_mode="original", stitch_plan=None
    ):
        # stitch_plan=None is the default "stitch every segment in order" plan;
        # [] means "no stitch" (the per-segment clips are the deliverable).
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="t",
            prompt="Replace the character",
            reference_image_urls=[],
            model=model,
            resolution="720p" if model == "gemini-omni" else "480p",
            audio_mode=audio_mode,
            stitch_plan=stitch_plan,
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def _two_swaps(self, db_session, project_id):
        # Two adjacent swap segments, no keep gap → every cut_clip call during
        # the run is a swap-submit cut.
        db_session.add_all([
            SegmentDef(project_id=project_id, index=0, start_sec=0.0, end_sec=5.0,
                       has_face=True, action="swap"),
            SegmentDef(project_id=project_id, index=1, start_sec=5.0, end_sec=10.0,
                       has_face=True, action="swap"),
        ])
        db_session.commit()

    def test_gemini_clip_sent_without_audio(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Every clip uploaded to Gemini Omni is cut with include_audio=False."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        self._two_swaps(db_session, project_id)
        run_id = self._make_run(db_session, project_id, model="gemini-omni")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_omni_records) == 2
        assert cut_calls, "expected at least one clip cut"
        assert all(c["include_audio"] is False for c in cut_calls), (
            f"Gemini clips must be video-only: {cut_calls}"
        )
        # Gemini always overlays the original source audio.
        assert stitch_calls and stitch_calls[0].get("audio_mode") == "original"

    def test_seedance_clip_keeps_audio(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Seedance clips are still cut WITH their audio track."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        self._two_swaps(db_session, project_id)
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        assert cut_calls and all(c["include_audio"] is True for c in cut_calls)

    def test_gemini_forces_original_audio_over_requested_seedance(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A Gemini run requesting audio_mode='seedance' still stitches 'original'."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        self._two_swaps(db_session, project_id)
        run_id = self._make_run(
            db_session, project_id, model="gemini-omni", audio_mode="seedance"
        )

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert stitch_calls and stitch_calls[0].get("audio_mode") == "original"

    def test_gemini_segment_over_10s_is_skipped(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A >10s swap segment is not sent to Gemini (create_omni_task never
        called); the original footage is stitched in its place and the run
        completes. Gemini's 10s ceiling is the tightest in the registry, so this
        is the case an operator hits most often."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source), status=ProjectStatus.ready)
        db_session.add(
            SegmentDef(project_id=project_id, index=0, start_sec=0.0, end_sec=12.0,
                       has_face=True, action="swap")
        )
        db_session.commit()
        run_id = self._make_run(db_session, project_id, model="gemini-omni")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=12.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # No Gemini task created for the over-long segment — no credits spent.
        assert len(fake_kie.create_omni_records) == 0
        # It still stitched: the original footage stands in for the segment.
        assert len(stitch_calls) == 1, stitch_calls
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_local_path is not None
            assert run.source_fallback_segments == 1
            rs = (
                s.query(RunSegment)
                .filter(RunSegment.run_id == run_id)
                .one()
            )
            assert rs.status == SegmentStatus.failed
            assert rs.source_fallback is True
            assert "limit" in (rs.error_message or "")


# ---------------------------------------------------------------------------
# Tests: the per-model clip ceiling (ai_models.max_clip_sec) at submit time
# ---------------------------------------------------------------------------


class TestModelClipCeiling:
    """The over-long-clip guard is model-generic, not Gemini-specific.

    It used to be a hardcoded 10s check behind ``if is_omni``. Now that a
    project picks its own segmentation cap (VideoProject.max_segment_sec) and
    Seedance 2.5 can generate 30s clips, the same code path has to protect a
    10s, a 15s and a 30s backend, so it reads
    ``ai_models.max_clip_sec(run.model)``. Two distinct outcomes:

    * the SEGMENT itself is over the ceiling → the swap is skipped (marked
      failed) rather than generating only its first N seconds;
    * only pre/post-roll pushed the CLIP over → the clip is trimmed back to the
      ceiling and the segment is generated normally.

    Media is fully mocked — no ffmpeg needed.
    """

    _mock_media = staticmethod(TestGeminiOmniAudioAndLimit._mock_media)
    _make_run = staticmethod(TestGeminiOmniAudioAndLimit._make_run)

    @staticmethod
    def _one_swap(db_session, project_id, *, start, end, pre_roll=0.0, post_roll=0.0):
        """A single swap segment, optionally with pre/post-roll context."""
        db_session.add(
            SegmentDef(
                project_id=project_id, index=0, start_sec=start, end_sec=end,
                has_face=True, action="swap",
                pre_roll_sec=pre_roll, post_roll_sec=post_roll,
            )
        )
        db_session.commit()

    def _swap_cuts(self, cut_calls):
        """Only the clip cuts that feed the AI model (keep cuts are 'fast')."""
        return [c for c in cut_calls if not c["fast_intermediate"]]

    # -- over-long segment: skipped --------------------------------------

    @pytest.mark.parametrize(
        "model, model_limit",
        [
            ("seedance", 15.0),
            ("seedance-fast", 15.0),
            ("gemini-omni", 10.0),
        ],
    )
    def test_segment_over_the_model_ceiling_is_skipped(
        self, db_engine, db_session, tmp_path, monkeypatch, model, model_limit
    ):
        """A 22s swap segment falls back to the source on models that can't do it.

        This is the safety net for a project segmented at 30s (for Seedance 2.5)
        that is then run on a shorter-clip backend. The segment is marked failed
        with `source_fallback` set and a message naming *that* model's ceiling.

        Downstream, this failure is treated as DETERMINISTIC — no retry makes a
        22s segment generatable on a 15s model — so the completeness gate lets
        the run through and the stitch substitutes the ORIGINAL footage for this
        segment. The run reaches `done` and delivers, with the un-swapped part
        recorded on Run.source_fallback_segments. Contrast a transient failure,
        which still blocks the run (TestTransientFailureStillBlocks).
        """
        from app import ai_models
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=30.0,
        )
        self._one_swap(db_session, project_id, start=0.0, end=22.0)
        run_id = self._make_run(db_session, project_id, model=model)

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=25.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # No credits spent: nothing submitted on either payload path, and no
        # model-bound clip cut.
        assert fake_kie.create_task_records == []
        assert fake_kie.create_omni_records == []
        assert self._swap_cuts(cut_calls) == []

        # But the run DID stitch and deliver — the original footage stood in.
        assert len(stitch_calls) == 1, stitch_calls
        # The substituted clip is a plain source cut (fast_intermediate=True,
        # like a "keep" segment) spanning exactly the segment's own bounds, so
        # the timeline length is unchanged and the soundtrack stays in sync.
        fallback_cuts = [c for c in cut_calls if c["fast_intermediate"]]
        assert len(fallback_cuts) == 1, cut_calls
        assert fallback_cuts[0]["start"] == 0.0
        assert fallback_cuts[0]["end"] == 22.0

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            # Status stays `failed` — the swap genuinely did not happen — but
            # source_fallback is what tells the gate and the UI it was handled.
            assert rs.status == SegmentStatus.failed
            assert rs.source_fallback is True
            msg = rs.error_message or ""
            # The message must name THIS model's limit, not a hardcoded 10s.
            assert f"{model_limit:.0f}s" in msg, msg
            assert ai_models.spec_for(model).label in msg, msg
            assert "22.0s" in msg, msg
            # Operator-facing copy: the run UI truncates this at 80 chars, so
            # the consequence must survive truncation.
            assert len(msg) <= 80, (len(msg), msg)
            assert "original footage" in msg, msg

            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            # Denormalised count so the runs list / webhook can flag the run
            # without walking its segments.
            assert run.source_fallback_segments == 1

    # -- the asymmetry: only DETERMINISTIC failures fall back ------------

    def test_transient_task_failure_still_blocks_the_run(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A kie task failure must NOT fall back to the original footage.

        This is the load-bearing distinction. An over-long segment is skipped and
        substituted because retrying can never fix it. A task that errored *can*
        be fixed by retrying, and shipping the original footage instead would
        silently throw away the swap the operator paid for and asked for. So this
        path keeps the old behaviour: run `incomplete`, no video, no fallback.

        If someone ever "simplifies" _skip_segment by making every failure fall
        back, this test is what should stop them.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        # Well within every model's ceiling — the only failure here is the task.
        self._one_swap(db_session, project_id, start=0.0, end=6.0)
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        # Task ids are random, so fail every task rather than naming one.
        class AlwaysFailingKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                self.poll_calls.append(task_id)
                return {"state": "fail", "failMsg": "injected failure"}

        fake_kie = AlwaysFailingKie(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # It really did try (and exhaust its attempts) — otherwise this test
        # would pass for the wrong reason.
        assert fake_kie.create_task_records, "no task was ever submitted"

        assert stitch_calls == [], "a transient failure must not be stitched over"

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            assert rs.status == SegmentStatus.failed
            # The crucial bit: NOT marked as a source fallback.
            assert rs.source_fallback is False

            run = s.get(Run, run_id)
            assert run.status == RunStatus.incomplete
            assert run.result_local_path is None
            # Never finalized, so the count was never written.
            assert run.source_fallback_segments is None

    def test_fallback_and_successful_swap_coexist_in_one_stitch(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A partly-swapped delivery: one segment generated, one substituted.

        The whole point of the change — the gate now tolerates precisely this
        mix, and only this kind of mix.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=30.0,
        )
        # idx 0 fits Seedance 2.0 (15s ceiling); idx 1 does not.
        db_session.add_all([
            SegmentDef(project_id=project_id, index=0, start_sec=0.0,
                       end_sec=6.0, has_face=True, action="swap"),
            SegmentDef(project_id=project_id, index=1, start_sec=6.0,
                       end_sec=28.0, has_face=True, action="swap"),
        ])
        db_session.commit()
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=28.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Exactly one segment was billed — the one that fits.
        assert len(fake_kie.create_task_records) == 1
        assert len(stitch_calls) == 1

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            by_idx = {
                rs.index: rs
                for rs in s.query(RunSegment).filter(
                    RunSegment.run_id == run_id
                )
            }
            assert by_idx[0].status == SegmentStatus.completed
            assert by_idx[0].source_fallback is False
            assert by_idx[1].status == SegmentStatus.failed
            assert by_idx[1].source_fallback is True

            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.source_fallback_segments == 1

    def test_rerun_clears_the_fallback_verdict(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Re-running a fallback segment on a capable model really swaps it.

        source_fallback describes a PAST attempt. If a re-submit did not clear
        it, the stitch would keep substituting original footage even after the
        segment generated successfully — the operator's fix would appear to do
        nothing. Simulated here by flipping the run's model to one that can
        generate the segment, which is what "switch models and retry" does.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=30.0,
        )
        self._one_swap(db_session, project_id, start=0.0, end=22.0)
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=25.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        # First pass on a 15s model → fallback.
        process_run(run_id, kie=FakeKieClient(str(source)),
                    gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            assert rs.source_fallback is True
            # Switch to the 30s model and reset the run for a retry.
            run = s.get(Run, run_id)
            run.model = "seedance-2-5"
            run.status = RunStatus.queued
            run.result_local_path = None
            s.commit()

        # Second pass on the 30s model → real swap, flag cleared.
        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 1

        with Session() as s:
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            assert rs.status == SegmentStatus.completed
            assert rs.source_fallback is False, (
                "a successfully generated segment must not still be flagged as "
                "source-fallback, or the stitch keeps using original footage"
            )
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            # Nothing un-swapped any more — 0, not None: finalize ran and checked.
            assert run.source_fallback_segments == 0

    def test_segment_within_the_2_5_ceiling_is_submitted(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The very same 22s segment IS generated by Seedance 2.5 (30s ceiling)."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=30.0,
        )
        self._one_swap(db_session, project_id, start=0.0, end=22.0)
        run_id = self._make_run(db_session, project_id, model="seedance-2-5")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=25.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 1
        rec = fake_kie.create_task_records[0]
        assert rec["model"] == "bytedance/seedance-2-5"
        # The whole 22s is cut and uploaded — no silent truncation.
        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        assert swap_cuts[0]["start"] == pytest.approx(0.0)
        assert swap_cuts[0]["end"] == pytest.approx(22.0)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            assert rs.status == SegmentStatus.completed
            assert s.get(Run, run_id).status == RunStatus.done

    # -- requested output duration ---------------------------------------

    @pytest.mark.parametrize(
        "model, seg_end, expected_duration",
        [
            # 2.5 sends the -1 sentinel: it reads the output length off the
            # input clip, and the live API rejects any concrete value. That is
            # also more faithful than the 2.0 path, which rounds to whole
            # seconds and so drifts against the clip it was cut from.
            ("seedance-2-5", 22.0, -1),
            ("seedance-2-5", 30.0, -1),
            # The 2.0 family is unchanged below its own ceiling.
            ("seedance", 12.0, 12),
        ],
    )
    def test_requested_duration_follows_the_model_range(
        self, db_engine, db_session, tmp_path, monkeypatch,
        model, seg_end, expected_duration,
    ):
        """create_task's duration comes from ai_models, so it is per-model."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=30.0,
        )
        self._one_swap(db_session, project_id, start=0.0, end=seg_end)
        run_id = self._make_run(db_session, project_id, model=model)

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=40.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 1
        assert fake_kie.create_task_records[0]["duration"] == expected_duration

    # -- over-long clip (pre/post-roll): trimmed, not skipped -------------

    def test_pre_post_roll_over_the_2_5_ceiling_trims_the_clip(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A 28s segment + 5s post-roll is cut back to 30s rather than skipped.

        The segment fits; only the surrounding context does not. Trimming keeps
        the swap (with less context) instead of losing it entirely.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=30.0,
        )
        self._one_swap(db_session, project_id, start=0.0, end=28.0, post_roll=5.0)
        run_id = self._make_run(db_session, project_id, model="seedance-2-5")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=40.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        # [0, 33] would exceed the 30s ceiling → trimmed back to [0, 30].
        assert swap_cuts[0]["start"] == pytest.approx(0.0)
        assert swap_cuts[0]["end"] == pytest.approx(30.0)
        assert len(fake_kie.create_task_records) == 1
        # The clip was trimmed to 30s, but the REQUEST still carries -1: 2.5
        # derives the output length from the (now 30s) input video itself.
        assert fake_kie.create_task_records[0]["duration"] == -1

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            assert s.get(Run, run_id).status == RunStatus.done

    # -- follows_input_geometry: adaptive + -1, sent as a PAIR --------------

    @pytest.mark.parametrize("source_aspect", ["9:16", "16:9", "1:1", "21:9", None])
    def test_2_5_always_sends_adaptive_aspect(
        self, db_engine, db_session, tmp_path, monkeypatch, source_aspect
    ):
        """2.5 requires `adaptive` even when the source ratio is one it accepts.

        This is the prod 500 (run 1736426f): a 9:16 source produced
        `aspect_ratio="9:16"` and the API answered
          Issues: [0] `ratio` must be `adaptive`. [1] `duration` must be -1.
        Seedance classifies our submissions as video editing — we always attach a
        reference video — and then takes both from the input clip.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            aspect_ratio=source_aspect,
        )
        self._one_swap(db_session, project_id, start=0.0, end=10.0)
        run_id = self._make_run(db_session, project_id, model="seedance-2-5")

        self._mock_media(monkeypatch, duration_sec=34.0,
                         cut_calls=[], stitch_calls=[])
        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        rec = fake_kie.create_task_records[0]
        assert rec["aspect_ratio"] == "adaptive", source_aspect
        # The pair must always travel together — the API validates them jointly.
        assert rec["duration"] == -1, source_aspect

    def test_2_0_still_sends_the_concrete_source_aspect(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The 2.0 family is untouched: real ratio, real duration."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            aspect_ratio="9:16",
        )
        self._one_swap(db_session, project_id, start=0.0, end=10.0)
        run_id = self._make_run(db_session, project_id, model="seedance")

        self._mock_media(monkeypatch, duration_sec=34.0,
                         cut_calls=[], stitch_calls=[])
        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        rec = fake_kie.create_task_records[0]
        assert rec["aspect_ratio"] == "9:16"
        assert rec["duration"] == 10

    # -- min_clip_sec: 2.5 needs a 4-30s input ----------------------------

    def test_short_segment_is_padded_to_the_2_5_minimum(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A 2.5s segment must be padded to 4s, not the 2s Seedance 2.0 floor.

        2.5 in video-editing mode rejects the whole request when the input video
        is under 4s, so the old global MIN_SWAP_VIDEO_SEC=2.0 would have produced
        a second class of 500 as soon as anyone ran a short segment.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(db_session, project_id, start=1.0, end=3.5)
        run_id = self._make_run(db_session, project_id, model="seedance-2-5")

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=30.0,
                         cut_calls=cut_calls, stitch_calls=[])
        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        clip_len = swap_cuts[0]["end"] - swap_cuts[0]["start"]
        assert clip_len >= 4.0 - 0.01, swap_cuts[0]
        # The segment itself still starts where it should; padding grows forward.
        assert swap_cuts[0]["start"] == pytest.approx(1.0)
        assert len(fake_kie.create_task_records) == 1

    def test_short_segment_keeps_the_2_0_floor_on_2_0(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Unchanged for 2.0: a 2.5s segment is not inflated to 4s of input."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(db_session, project_id, start=1.0, end=3.5)
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=30.0,
                         cut_calls=cut_calls, stitch_calls=[])
        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert swap_cuts[0]["end"] - swap_cuts[0]["start"] == pytest.approx(2.5)

    def test_source_shorter_than_the_model_minimum_falls_back(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A source too short to pad to 4s is deterministic → original footage.

        Nothing can lengthen the video, so retrying is futile; this takes the
        same source-fallback path as an over-long segment rather than blocking.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(db_session, project_id, start=0.0, end=3.0)
        run_id = self._make_run(db_session, project_id, model="seedance-2-5")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        # Whole source is 3s — under 2.5's 4s input floor.
        self._mock_media(monkeypatch, duration_sec=3.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)
        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Never submitted — no credits burned on a request the API would reject.
        assert fake_kie.create_task_records == []
        assert self._swap_cuts(cut_calls) == []
        # But it still delivered, using the original footage.
        assert len(stitch_calls) == 1

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            assert rs.status == SegmentStatus.failed
            assert rs.source_fallback is True
            assert "minimum" in (rs.error_message or "")
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.source_fallback_segments == 1

    # -- trim ORDER: roll context is sacrificed, never segment footage --

    def test_trim_gives_up_post_roll_before_pre_roll(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The over-long-clip trim must not shorten the clip from the END.

        ``clip_start = sd_start_sec - pre_roll_sec``, so the naive
        ``clip_end = clip_start + max_clip`` spends the budget on pre-roll and
        drops real segment footage off the tail. Worked example (the one that
        exposed it): max_clip=10, a 9.5s segment, pre_roll=2.0, post_roll=0.5.
        The segment passes the skip check, so it IS generated — and the old trim
        cut [3.0, 13.0], losing the segment's final 1.5s while keeping 2.0s of
        pre-roll. The result is concatenated whole, so the swap covered footage
        the operator never marked and skipped footage they did.

        Correct order: post-roll first, then pre-roll, never crossing
        [sd_start_sec, sd_end_sec].
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(
            db_session, project_id,
            start=5.0, end=14.5, pre_roll=2.0, post_roll=0.5,
        )
        run_id = self._make_run(db_session, project_id, model="gemini-omni")

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=40.0,
                         cut_calls=cut_calls, stitch_calls=[])

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        start, end = swap_cuts[0]["start"], swap_cuts[0]["end"]
        # The whole segment is covered — this is the assertion that failed before.
        assert start <= 5.0 and end >= 14.5
        # Budget respected.
        assert end - start == pytest.approx(10.0)
        # Post-roll is gone entirely; the remaining slack went to pre-roll.
        assert end == pytest.approx(14.5)
        assert start == pytest.approx(4.5)

    def test_trim_keeps_pre_roll_when_dropping_post_roll_is_enough(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Pre-roll is only touched once post-roll is fully spent.

        max_clip=15, a 12s segment with pre_roll=1.0 and post_roll=5.0 → the
        naive cut is [4.0, 22.0] (18s). Trimming 3s of post-roll is sufficient,
        so the lead-in survives untouched.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=15.0,
        )
        self._one_swap(
            db_session, project_id,
            start=5.0, end=17.0, pre_roll=1.0, post_roll=5.0,
        )
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=40.0,
                         cut_calls=cut_calls, stitch_calls=[])

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        start, end = swap_cuts[0]["start"], swap_cuts[0]["end"]
        assert start == pytest.approx(4.0)   # full 1.0s pre-roll retained
        assert end == pytest.approx(19.0)    # post-roll cut from 5.0s to 2.0s
        assert end - start == pytest.approx(15.0)
        assert start <= 5.0 and end >= 17.0

    def test_trim_eats_into_pre_roll_only_after_post_roll_is_gone(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Both rolls are reduced when one is not enough — post-roll to zero first.

        max_clip=15, a 14s segment with 2.0s on each side → the naive cut is
        [3.0, 21.0] (18s). All 2.0s of post-roll goes, then 1.0s of the pre-roll.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready,
            max_segment_sec=15.0,
        )
        self._one_swap(
            db_session, project_id,
            start=5.0, end=19.0, pre_roll=2.0, post_roll=2.0,
        )
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=40.0,
                         cut_calls=cut_calls, stitch_calls=[])

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        start, end = swap_cuts[0]["start"], swap_cuts[0]["end"]
        assert end == pytest.approx(19.0)    # post-roll fully sacrificed
        assert start == pytest.approx(4.0)    # 1.0s of the 2.0s pre-roll kept
        assert end - start == pytest.approx(15.0)
        assert start <= 5.0 and end >= 19.0

    def test_segment_inside_the_skip_epsilon_loses_at_most_the_epsilon(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A segment in the 50ms tolerance band is clamped, not skipped.

        The skip check allows ``max_clip + 0.05``, so a 10.04s segment on a 10s
        model is generated. Nothing can be given back (there is no roll context
        and the segment alone is over budget), so the final clamp is the only
        option — and it can never cost more than that sub-frame epsilon.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(db_session, project_id, start=0.0, end=10.04)
        run_id = self._make_run(db_session, project_id, model="gemini-omni")

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=40.0,
                         cut_calls=cut_calls, stitch_calls=[])

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        start, end = swap_cuts[0]["start"], swap_cuts[0]["end"]
        assert end - start == pytest.approx(10.0)
        assert 10.04 - end <= 0.05

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            assert rs.status == SegmentStatus.completed

    def test_gemini_clip_trim_at_10s_is_unchanged(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Regression: making the guard generic must not move Gemini's ceiling.

        A 9s segment with 5s of post-roll still yields a 10s clip and a 10s
        omni trim window, exactly as before.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(db_session, project_id, start=0.0, end=9.0, post_roll=5.0)
        run_id = self._make_run(db_session, project_id, model="gemini-omni")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=40.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        swap_cuts = self._swap_cuts(cut_calls)
        assert len(swap_cuts) == 1
        assert swap_cuts[0]["end"] == pytest.approx(10.0)
        assert len(fake_kie.create_omni_records) == 1
        assert fake_kie.create_omni_records[0]["video_end"] == pytest.approx(10.0)
        assert fake_kie.create_omni_records[0]["duration"] == 10


# ---------------------------------------------------------------------------
# Tests: the Seedance 2.5 `generate_audio` switch
# ---------------------------------------------------------------------------


class TestGenerateAudioSwitch:
    """`generate_audio` belongs to Seedance 2.5 alone.

    Every other backend rejects the field, so it must be ABSENT from their
    createTask payload — not present-and-False.

    Its value is decided once per run in _build_submit_state, and the rule is
    "True unless we are CERTAIN the generated track is discarded":

    ==============================  ==========================  ==============
    stitch_plan                     audio_mode                  generate_audio
    ==============================  ==========================  ==============
    None / custom list (stitches)   "original"                  False
    None / custom list (stitches)   "seedance"                  True
    ``[]`` (no stitch)              either                      True
    ==============================  ==========================  ==============

    The False row is the credit saving: the stitch overlays the source
    soundtrack and throws the generated one away. The no-stitch row is the trap
    — those runs deliver the per-segment clips themselves, so a track derived
    from audio_mode (which defaults to "original") would hand the operator a set
    of silent clips.
    """

    _mock_media = staticmethod(TestGeminiOmniAudioAndLimit._mock_media)
    _make_run = staticmethod(TestGeminiOmniAudioAndLimit._make_run)
    _one_swap = staticmethod(TestModelClipCeiling._one_swap)

    def _run(
        self, db_session, tmp_path, monkeypatch, *, model, audio_mode,
        stitch_plan=None,
    ):
        """Process a one-swap-segment run and return the FakeKieClient."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(db_session, project_id, start=0.0, end=5.0)
        run_id = self._make_run(
            db_session, project_id, model=model, audio_mode=audio_mode,
            stitch_plan=stitch_plan,
        )

        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=[], stitch_calls=[])

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())
        return fake_kie

    @pytest.mark.parametrize(
        "audio_mode, expected",
        [("seedance", True), ("original", False)],
    )
    def test_seedance_2_5_sends_the_switch_derived_from_audio_mode(
        self, db_engine, db_session, tmp_path, monkeypatch, audio_mode, expected
    ):
        """On a STITCHING run the value follows the run's audio_mode.

        audio_mode="original" is the only case where the generated track is
        provably thrown away, so it is the only case that skips paying for it.
        """
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode=audio_mode,
        )

        assert len(fake_kie.create_task_records) == 1
        rec = fake_kie.create_task_records[0]
        assert rec["generate_audio"] is expected
        assert rec["model"] == "bytedance/seedance-2-5"

    def test_no_stitch_run_asks_for_audio_despite_audio_mode_original(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """THE regression this rule exists for: stitch_plan=[] + audio_mode default.

        A no-stitch run submits and bills every swap segment and produces no
        stitched video — the per-segment clips ARE the deliverable. audio_mode is
        meaningless for it (nothing is stitched, so nothing overlays the source
        soundtrack) yet it defaults to "original", so deriving the switch from
        audio_mode alone made every no-stitch Seedance 2.5 run hand back SILENT
        clips. Pure model-choice regression: 2.0 has no switch and always writes
        audio.
        """
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode="original", stitch_plan=[],
        )

        assert len(fake_kie.create_task_records) == 1
        assert fake_kie.create_task_records[0]["generate_audio"] is True

    @pytest.mark.parametrize("audio_mode", ["original", "seedance"])
    def test_no_stitch_run_asks_for_audio_for_either_audio_mode(
        self, db_engine, db_session, tmp_path, monkeypatch, audio_mode
    ):
        """audio_mode cannot suppress audio on a run that delivers raw clips."""
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode=audio_mode, stitch_plan=[],
        )

        assert len(fake_kie.create_task_records) == 1
        assert fake_kie.create_task_records[0]["generate_audio"] is True

    def test_no_stitch_run_produces_no_final_video(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Pins the premise of the rule above: stitch_plan=[] delivers the clips.

        If this ever changes (a no-stitch run starts stitching something), the
        "always ask for audio" branch needs revisiting — hence the assertion
        lives next to it rather than only in the stitch-plan tests.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        self._one_swap(db_session, project_id, start=0.0, end=5.0)
        run_id = self._make_run(
            db_session, project_id, model="seedance-2-5", stitch_plan=[]
        )

        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=[], stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert stitch_calls == []
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert not run.result_local_path
            # The segment result is the deliverable, and it exists.
            rs = s.query(RunSegment).filter(RunSegment.run_id == run_id).one()
            assert rs.status == SegmentStatus.completed

    def test_custom_stitch_plan_still_counts_as_stitching(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A non-empty custom plan stitches, so the credit saving still applies.

        Only the EMPTY plan means "no stitch". A populated one concatenates the
        clips, so audio_mode="original" really does discard the generated track
        and the switch stays False — a custom plan must not be lumped in with [].

        Uses real media (the custom-plan stitch path muxes per-clip audio with
        ffmpeg, which placeholder clip files cannot satisfy).
        """
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        db_session.expire_all()
        segs = (
            db_session.query(SegmentDef)
            .filter_by(project_id=project_id)
            .order_by(SegmentDef.index)
            .all()
        )
        plan = [{"kind": "segment", "segment_def_id": segs[0].id}]
        run_id = self._make_run(
            db_session, project_id, model="seedance-2-5",
            audio_mode="original", stitch_plan=plan,
        )

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 1
        assert fake_kie.create_task_records[0]["generate_audio"] is False

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            # It really did stitch — that is what makes the False correct.
            assert run.result_local_path

    def test_every_segment_of_a_run_makes_the_same_request(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The switch is a run-level decision, so it cannot vary per segment."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        TestGeminiOmniAudioAndLimit._two_swaps(self, db_session, project_id)
        run_id = self._make_run(
            db_session, project_id, model="seedance-2-5",
            audio_mode="original", stitch_plan=[],
        )

        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=[], stitch_calls=[])

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        assert {r["generate_audio"] for r in fake_kie.create_task_records} == {True}

    @pytest.mark.parametrize(
        "model", ["seedance", "seedance-fast", "seedance-mini"],
    )
    @pytest.mark.parametrize("stitch_plan", [None, []])
    def test_seedance_2_0_family_never_sends_the_switch(
        self, db_engine, db_session, tmp_path, monkeypatch, model, stitch_plan
    ):
        """The 2.0 payload stays byte-identical: the key is never passed at all.

        `is NOT_PASSED` (rather than `is False`) is the whole assertion — sending
        the field as False would still be a payload the 2.0 models reject.

        Both stitch plans are exercised because the run-level decision is now
        computed for EVERY run: a no-stitch run resolves it to True, and only the
        model's capability stops that from reaching a 2.0 payload (where
        KieClient.create_task would raise ValueError and fail the whole run).
        """
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            # Ask for generated audio, so only the model's capability can be
            # what suppresses the field.
            model=model, audio_mode="seedance", stitch_plan=stitch_plan,
        )

        assert len(fake_kie.create_task_records) == 1
        rec = fake_kie.create_task_records[0]
        assert rec["generate_audio"] is NOT_PASSED
        assert rec["extra_kwargs"] == {}

    @pytest.mark.parametrize("stitch_plan", [None, []])
    def test_gemini_omni_payload_carries_no_audio_switch(
        self, db_engine, db_session, tmp_path, monkeypatch, stitch_plan
    ):
        """Gemini Omni has its own payload shape with no audio switch at all."""
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="gemini-omni", audio_mode="seedance", stitch_plan=stitch_plan,
        )

        assert fake_kie.create_task_records == []
        assert len(fake_kie.create_omni_records) == 1
        assert fake_kie.create_omni_records[0]["extra_kwargs"] == {}


# ---------------------------------------------------------------------------
# Tests: project-level "remove original audio" (VideoProject.mute_source)
# ---------------------------------------------------------------------------


class TestProjectMuteSource:
    """`mute_source` uploads video-only swap clips so the AI model writes its
    own audio from the prompt instead of reacting to the source soundtrack.

    The source file and the "keep" clips are untouched, so both audio_mode
    options still work. Media is fully mocked — no ffmpeg needed.
    """

    # Same media mocks and Run factory as the Gemini class above; re-wrapped in
    # staticmethod so `self.` access doesn't bind the instance as an argument.
    _mock_media = staticmethod(TestGeminiOmniAudioAndLimit._mock_media)
    _make_run = staticmethod(TestGeminiOmniAudioAndLimit._make_run)

    @staticmethod
    def _swap_then_keep(db_session, project_id):
        """One swap segment followed by one keep segment.

        The keep segment is what proves muting is scoped to the AI upload: it is
        cut at stitch time and must still carry its audio.
        """
        db_session.add_all([
            SegmentDef(project_id=project_id, index=0, start_sec=0.0, end_sec=5.0,
                       has_face=True, action="swap"),
            SegmentDef(project_id=project_id, index=1, start_sec=5.0, end_sec=10.0,
                       has_face=False, action="keep"),
        ])
        db_session.commit()

    def test_muted_project_uploads_video_only_clips(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Seedance swap clips are cut with include_audio=False."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready, mute_source=True
        )
        self._swap_then_keep(db_session, project_id)
        run_id = self._make_run(
            db_session, project_id, model="seedance", audio_mode="seedance"
        )

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 1
        # The swap clip (the only one cut with fast_intermediate=False) is muted…
        swap_cuts = [c for c in cut_calls if not c["fast_intermediate"]]
        assert swap_cuts and all(c["include_audio"] is False for c in swap_cuts), (
            f"muted project must upload video-only swap clips: {cut_calls}"
        )
        # …while the keep clip, which only ever feeds the stitch, keeps its audio.
        keep_cuts = [c for c in cut_calls if c["fast_intermediate"]]
        assert keep_cuts and all(c["include_audio"] is True for c in keep_cuts), (
            f"keep clips must retain the original audio: {cut_calls}"
        )

    def test_unmuted_project_is_unchanged(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The default (mute_source=False) still sends audio to Seedance."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready, mute_source=False
        )
        self._swap_then_keep(db_session, project_id)
        run_id = self._make_run(db_session, project_id, model="seedance")

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert cut_calls and all(c["include_audio"] is True for c in cut_calls)

    def test_muted_project_still_stitches_requested_audio_mode(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Muting the AI input does not take away audio_mode='original'.

        The source file is never modified, so an operator can still overlay the
        untouched soundtrack over the swapped visuals.
        """
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready, mute_source=True
        )
        self._swap_then_keep(db_session, project_id)
        run_id = self._make_run(
            db_session, project_id, model="seedance", audio_mode="original"
        )

        cut_calls: list[dict] = []
        stitch_calls: list[dict] = []
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=cut_calls, stitch_calls=stitch_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert stitch_calls and stitch_calls[0].get("audio_mode") == "original"


class TestMuteSourceClipCheckpoint:
    """Toggling `mute_source` invalidates a crash-resume clip checkpoint.

    Reusing a clip cut under the previous setting would silently ignore the
    toggle on the retry an operator makes precisely because the model reacted
    to the source audio.
    """

    @staticmethod
    def _clip(tmp_path, name, *, has_audio):
        path = tmp_path / name
        path.write_bytes(b"clip")
        return str(path), has_audio

    def _patch_probe(self, monkeypatch, mapping):
        """Report has_audio per path; raise for anything unmapped."""
        from app.media import MediaInfo, MediaError
        import app.pipeline_v2 as pv2

        def fake_probe(path):
            if path not in mapping:
                raise MediaError(f"unprobeable: {path}")
            return MediaInfo(
                duration_sec=5.0, width=320, height=240, fps=25.0,
                aspect_ratio="4:3", has_audio=mapping[path],
            )

        monkeypatch.setattr(pv2.media_mod, "probe", fake_probe)

    def test_clip_with_audio_is_stale_when_muting(self, tmp_path, monkeypatch):
        from app.pipeline_v2 import _clip_audio_is_stale

        path, _ = self._clip(tmp_path, "c.mp4", has_audio=True)
        self._patch_probe(monkeypatch, {path: True})
        assert _clip_audio_is_stale(path, want_muted=True) is True
        assert _clip_audio_is_stale(path, want_muted=False) is False

    def test_muted_clip_is_stale_when_audio_wanted_again(self, tmp_path, monkeypatch):
        from app.pipeline_v2 import _clip_audio_is_stale

        path, _ = self._clip(tmp_path, "c.mp4", has_audio=False)
        self._patch_probe(monkeypatch, {path: False})
        assert _clip_audio_is_stale(path, want_muted=False) is True
        assert _clip_audio_is_stale(path, want_muted=True) is False

    def test_missing_or_unprobeable_clip_is_not_stale(self, tmp_path, monkeypatch):
        """Can't tell → reuse the checkpoint, preserving the existing behaviour."""
        from app.pipeline_v2 import _clip_audio_is_stale

        self._patch_probe(monkeypatch, {})
        assert _clip_audio_is_stale(None, want_muted=True) is False
        assert _clip_audio_is_stale(
            str(tmp_path / "gone.mp4"), want_muted=True
        ) is False
        broken, _ = self._clip(tmp_path, "broken.mp4", has_audio=True)
        assert _clip_audio_is_stale(broken, want_muted=True) is False


class TestAnalyzeSegmentCap:
    """analyze_project caps segmentation at the PROJECT's max_segment_sec.

    The cap used to be a global setting (``settings.SEGMENT_MAX_SECONDS``, since
    deleted) folded together with Gemini's clip ceiling. It is now
    ``VideoProject.max_segment_sec``, read at analyze time only, where NULL means
    ``ai_models.UNIVERSAL_MAX_SEGMENT_SEC`` — the smallest per-clip ceiling in
    the registry, i.e. exactly what every pre-existing project already had.
    """

    @staticmethod
    def _analyze_and_capture(db_session, tmp_path, monkeypatch, *, max_segment_sec):
        """Analyze a project with this cap; return propose_segments' kwargs."""
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), max_segment_sec=max_segment_sec
        )

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(duration_sec=120.0, width=320, height=240,
                                 fps=25.0, aspect_ratio="4:3", has_audio=True),
        )
        captured: dict = {}

        def spy_propose(_path, **kw):
            captured.update(kw)
            return []

        monkeypatch.setattr(pv2.face_mod, "propose_segments", spy_propose)

        analyze_project(project_id)
        return captured

    def test_analyze_null_cap_uses_universal_default(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A project that never chose a cap is segmented at 10s, as before.

        This is the compatibility guarantee for every row that predates the
        column: NULL resolves to the smallest ceiling any model has, so the
        resulting segmentation can be run on ANY backend.
        """
        from app import ai_models

        captured = self._analyze_and_capture(
            db_session, tmp_path, monkeypatch, max_segment_sec=None
        )

        assert captured["max_segment_sec"] == ai_models.UNIVERSAL_MAX_SEGMENT_SEC
        assert captured["max_segment_sec"] == 10.0

    @pytest.mark.parametrize(
        "project_cap, expected",
        [
            (30.0, 30.0),   # Seedance 2.5's ceiling — the point of the setting
            (15.0, 15.0),   # the Seedance 2.0 family's ceiling
            (10.0, 10.0),   # the universal default, chosen explicitly
            (2.5, 2.5),     # small caps are honoured verbatim, not floored
        ],
    )
    def test_analyze_uses_project_cap(
        self, db_engine, db_session, tmp_path, monkeypatch, project_cap, expected
    ):
        """An explicit cap is passed straight through to propose_segments."""
        captured = self._analyze_and_capture(
            db_session, tmp_path, monkeypatch, max_segment_sec=project_cap
        )
        assert captured["max_segment_sec"] == expected

    def test_analyze_clamps_cap_above_absolute_max(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A cap beyond the longest-clip model is clamped, not proposed.

        Nothing in the registry could generate a segment that long, so proposing
        one would only guarantee it gets skipped at submit time.
        """
        from app import ai_models

        captured = self._analyze_and_capture(
            db_session, tmp_path, monkeypatch,
            max_segment_sec=ai_models.ABSOLUTE_MAX_SEGMENT_SEC + 45.0,
        )

        assert captured["max_segment_sec"] == ai_models.ABSOLUTE_MAX_SEGMENT_SEC
        assert captured["max_segment_sec"] == 30.0


class TestAnalyzeNonFaceProjectTypes:
    """Subtitle-removal / cover-change projects skip detection entirely."""

    def _probe_30s(self, monkeypatch):
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(duration_sec=30.0, width=320, height=240,
                                 fps=25.0, aspect_ratio="4:3", has_audio=True),
        )

    # cover_change is the blank-slate type; subtitle_removal now scene-detects
    # (covered by TestAnalyzeSubtitleRemovalScenes), so it is not listed here.
    @pytest.mark.parametrize("project_type", ["cover_change"])
    def test_seeds_one_full_length_keep_segment(
        self, db_engine, db_session, tmp_path, monkeypatch, project_type
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), project_type=project_type
        )
        self._probe_30s(monkeypatch)

        def boom(*_a, **_kw):  # detection must never be reached
            raise AssertionError("propose_segments called for a non-face project")

        monkeypatch.setattr(pv2.face_mod, "propose_segments", boom)

        analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.status == ProjectStatus.ready
            assert project.duration_sec == pytest.approx(30.0)

            segs = list(project.segments)
            assert len(segs) == 1
            assert segs[0].index == 0
            assert segs[0].start_sec == pytest.approx(0.0)
            assert segs[0].end_sec == pytest.approx(30.0)
            assert segs[0].action == "keep"
            assert segs[0].has_face is False

    def test_face_swap_still_detects(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The default type keeps calling propose_segments (no regression)."""
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(db_session, str(source))
        self._probe_30s(monkeypatch)

        calls: list = []
        monkeypatch.setattr(
            pv2.face_mod, "propose_segments",
            lambda _p, **kw: calls.append(kw) or [],
        )

        analyze_project(project_id)
        assert len(calls) == 1


class TestAnalyzeSubtitleRemovalScenes:
    """Subtitle-removal projects cut on scene changes and mark scenes swap."""

    def _probe(self, monkeypatch, duration=30.0, fps=25.0):
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(duration_sec=duration, width=320, height=240,
                                 fps=fps, aspect_ratio="4:3", has_audio=True),
        )

    def test_uses_scene_cuts_marked_swap(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), project_type="subtitle_removal"
        )
        self._probe(monkeypatch)

        monkeypatch.setattr(
            pv2.face_mod, "propose_segments",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("propose_segments called")),
        )
        captured = {}

        def fake_detect(path, *, fps, min_scene_len_sec):
            captured["fps"] = fps
            captured["min_scene_len_sec"] = min_scene_len_sec
            return [(0.0, 6.0), (6.0, 16.0), (16.0, 30.0)]

        monkeypatch.setattr(pv2.scene_mod, "detect_scenes", fake_detect)

        analyze_project(project_id)

        assert captured["min_scene_len_sec"] == pytest.approx(4.0)
        assert captured["fps"] == pytest.approx(25.0)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            assert project.status == ProjectStatus.ready
            segs = list(project.segments)
            assert all(sd.action == "swap" for sd in segs)
            assert all(sd.has_face is False for sd in segs)
            assert segs[0].start_sec == pytest.approx(0.0)
            assert segs[-1].end_sec == pytest.approx(30.0)
            for sd in segs:
                assert sd.end_sec - sd.start_sec <= 10.0 + 1e-6
            assert len(segs) > 3
            for a, b in zip(segs, segs[1:]):
                assert b.start_sec == pytest.approx(a.end_sec)

    def test_single_shot_falls_back_to_one_full_swap(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), project_type="subtitle_removal"
        )
        self._probe(monkeypatch, duration=8.0)
        monkeypatch.setattr(
            pv2.scene_mod, "detect_scenes",
            lambda path, *, fps, min_scene_len_sec: [],
        )

        analyze_project(project_id)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            segs = list(s.get(VideoProject, project_id).segments)
            assert len(segs) == 1
            assert segs[0].action == "swap"
            assert segs[0].start_sec == pytest.approx(0.0)
            assert segs[0].end_sec == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Tests: TR7 audio_mode forwarding
# ---------------------------------------------------------------------------


class TestAudioModeForwarding:
    """
    Verify that process_run passes run.audio_mode to media_mod.stitch.
    Uses monkeypatch/spy on media_mod.stitch to capture kwargs without
    actually running ffmpeg for the stitch step.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose,
               audio_mode: str):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)

        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Audio Mode Test",
            prompt="Replace the character",
            reference_image_urls=[],
            resolution="480p",
            audio_mode=audio_mode,
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()
        return project_id, run.id

    def test_audio_mode_seedance_forwarded_to_stitch(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """process_run passes audio_mode='seedance' to media_mod.stitch."""
        from app.pipeline_v2 import process_run
        import app.pipeline_v2 as pipeline_v2_mod

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose, "seedance"
        )

        stitch_calls: list[dict] = []
        real_stitch = pipeline_v2_mod.media_mod.stitch

        def spy_stitch(*args, **kwargs):
            stitch_calls.append({"args": args, "kwargs": kwargs})
            return real_stitch(*args, **kwargs)

        monkeypatch.setattr(pipeline_v2_mod.media_mod, "stitch", spy_stitch)

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(stitch_calls) == 1, "stitch should be called exactly once"
        call = stitch_calls[0]
        assert call["kwargs"].get("audio_mode") == "seedance", (
            f"Expected audio_mode='seedance', got {call['kwargs'].get('audio_mode')!r}"
        )

    def test_audio_mode_original_forwarded_to_stitch(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """process_run passes audio_mode='original' to media_mod.stitch."""
        from app.pipeline_v2 import process_run
        import app.pipeline_v2 as pipeline_v2_mod

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose, "original"
        )

        stitch_calls: list[dict] = []
        real_stitch = pipeline_v2_mod.media_mod.stitch

        def spy_stitch(*args, **kwargs):
            stitch_calls.append({"args": args, "kwargs": kwargs})
            return real_stitch(*args, **kwargs)

        monkeypatch.setattr(pipeline_v2_mod.media_mod, "stitch", spy_stitch)

        fake_kie = FakeKieClient(synthetic_video)
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(stitch_calls) == 1
        call = stitch_calls[0]
        assert call["kwargs"].get("audio_mode") == "original", (
            f"Expected audio_mode='original', got {call['kwargs'].get('audio_mode')!r}"
        )


# ---------------------------------------------------------------------------
# Tests: delivery retry + delivery-only retry (skip re-stitch)
# ---------------------------------------------------------------------------


class TestDeliveryRetryAndReuse:
    def test_delivery_retries_then_succeeds(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """A transient Drive-upload failure is retried within the same run."""
        from app.pipeline_v2 import analyze_project, process_run
        import app.pipeline_v2 as pv2

        # No real backoff sleeps.
        monkeypatch.setattr(pv2.time, "sleep", lambda *a, **k: None)
        # Skip the real ffmpeg stitch for speed (we only care about delivery).
        monkeypatch.setattr(pv2.media_mod, "stitch", lambda *a, **k: None)

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, gdrive_folder_id="folder-x")

        class FlakyGDrive(FakeGDriveClient):
            def __init__(self, fail_times):
                super().__init__()
                self._left = fail_times

            def upload_file(self, *a, **k):
                if self._left > 0:
                    self._left -= 1
                    raise RuntimeError("transient upload fail")
                return super().upload_file(*a, **k)

        gd = FlakyGDrive(2)  # fail twice, succeed on the 3rd (RUN_DELIVER_ATTEMPTS=3)
        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gd)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_gdrive_file_id is not None
        assert len(gd.upload_calls) == 1  # only the successful attempt recorded

    def test_delivery_only_retry_skips_stitch(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """Retry after a delivery failure: all segments completed + final exists →
        the run re-delivers WITHOUT re-stitching."""
        from app.pipeline_v2 import analyze_project, process_run
        from app.storage import run_results_dir
        import app.pipeline_v2 as pv2

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, gdrive_folder_id="folder-x")

        Session = sessionmaker(bind=db_engine)
        r_dir = run_results_dir(run_id, project_id)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            for sd in [d for d in project.segments if d.action == "swap"]:
                res = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
                shutil.copy2(synthetic_video, res)
                s.add(RunSegment(
                    run_id=run_id, segment_def_id=sd.id, index=sd.index,
                    status=SegmentStatus.completed, local_result_path=res,
                ))
            run = s.get(Run, run_id)
            final = os.path.join(r_dir, "final.mp4")
            shutil.copy2(synthetic_video, final)
            run.result_local_path = final
            s.commit()

        stitch_calls: list = []
        monkeypatch.setattr(pv2.media_mod, "stitch", lambda *a, **k: stitch_calls.append(1))

        gd = FakeGDriveClient()
        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gd)

        assert stitch_calls == []          # re-stitch skipped
        assert len(gd.upload_calls) == 1   # but it WAS re-delivered
        with Session() as s:
            assert s.get(Run, run_id).status == RunStatus.done

    def test_stale_final_is_restitched_when_result_is_newer(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """Regression (run 7322a091): a segment rerun is already completed by
        the time finalize runs, so segment state alone looks identical to a
        delivery-only retry. When a result file is newer than the existing
        final.mp4, the final is stale and MUST be re-stitched — otherwise the
        old generation is silently re-delivered."""
        from app.pipeline_v2 import analyze_project, process_run
        from app.storage import run_results_dir
        import app.pipeline_v2 as pv2

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, gdrive_folder_id="folder-x")

        Session = sessionmaker(bind=db_engine)
        r_dir = run_results_dir(run_id, project_id)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            for sd in [d for d in project.segments if d.action == "swap"]:
                res = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
                shutil.copy2(synthetic_video, res)
                s.add(RunSegment(
                    run_id=run_id, segment_def_id=sd.id, index=sd.index,
                    status=SegmentStatus.completed, local_result_path=res,
                ))
            run = s.get(Run, run_id)
            final = os.path.join(r_dir, "final.mp4")
            shutil.copy2(synthetic_video, final)
            # Make the final OLDER than the segment results — the on-disk
            # state after a rerun downloaded a fresh result_XXXX.mp4.
            old = os.path.getmtime(final) - 100
            os.utime(final, (old, old))
            run.result_local_path = final
            s.commit()

        stitch_calls: list = []
        monkeypatch.setattr(pv2.media_mod, "stitch", lambda *a, **k: stitch_calls.append(1))

        gd = FakeGDriveClient()
        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gd)

        assert stitch_calls == [1]         # stale final re-stitched
        assert len(gd.upload_calls) == 1   # and re-delivered
        with Session() as s:
            assert s.get(Run, run_id).status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: parallel stitch cut_clip (STITCH_CUT_CONCURRENCY)
# ---------------------------------------------------------------------------


class TestStitchCutConcurrency:
    """
    Verify that the parallel cut_clip loop in the stitch phase:
    1. Produces clips in the correct segment order (not first-finished order).
    2. Is controlled by STITCH_CUT_CONCURRENCY env var.
    3. A cut_clip failure propagates and marks the run failed.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose):
        """Set up a ready project + queued run."""
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, prompt="concurrency test")
        return project_id, run_id

    def test_stitch_clip_order_preserved_with_concurrency(
        self,
        db_engine,
        db_session,
        synthetic_video,
        patch_propose,
        monkeypatch,
    ):
        """
        Even with STITCH_CUT_CONCURRENCY=2, clips must appear in ascending
        segment-index order in the stitch call — not in completion order.
        """
        import app.pipeline_v2 as pv2

        monkeypatch.setenv("STITCH_CUT_CONCURRENCY", "2")
        # Reload the constant so the monkeypatched env takes effect.
        monkeypatch.setattr(pv2, "STITCH_CUT_CONCURRENCY", 2)

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose
        )

        stitch_calls: list[list[str]] = []
        real_stitch = pv2.media_mod.stitch

        def spy_stitch(clip_paths, **kwargs):
            stitch_calls.append(list(clip_paths))
            return real_stitch(clip_paths, **kwargs)

        monkeypatch.setattr(pv2.media_mod, "stitch", spy_stitch)

        fake_kie = FakeKieClient(synthetic_video)
        from app.pipeline_v2 import process_run

        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(stitch_calls) == 1, "stitch should be called exactly once"
        clips = stitch_calls[0]

        # Verify ordering: segment indices appear in ascending order.
        # The fixed partition has 3 segments: swap(0), keep(1), swap(2).
        # The stitch list should be [result_0000, clip_0001, result_0002]
        # (or orig_0000/orig_0002 for fallbacks — doesn't matter, the point
        # is they're in index order).
        assert len(clips) == 3, f"Expected 3 clips, got {len(clips)}: {clips}"

        # Check filenames contain ascending indices. The index is always the
        # first 4-digit group right after the clip_/result_ prefix — cannot
        # anchor on the filename's END any more, since a keep clip's cache
        # path is now bounds-keyed (storage.keep_clip_path), not bare
        # clip_%04d.mp4 (see the finding-3 hardening pass).
        import re
        indices_in_filenames = [
            int(m.group(1))
            for path in clips
            if (m := re.search(r"(?:clip|result)_(\d{4})", path))
        ]
        assert indices_in_filenames == sorted(indices_in_filenames), (
            f"Clips not in ascending index order: {clips}"
        )

        with sessionmaker(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done

    def test_stitch_cut_error_propagates(
        self,
        db_engine,
        db_session,
        synthetic_video,
        patch_propose,
        monkeypatch,
    ):
        """
        If a cut_clip call raises during the stitch phase, the exception must
        propagate (run ends failed), not be silently swallowed.
        """
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(pv2, "STITCH_CUT_CONCURRENCY", 2)

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose
        )

        call_count = {"n": 0}
        real_cut_clip = pv2.media_mod.cut_clip

        def failing_cut_clip(src, start, end, dst, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated cut_clip failure in stitch")
            return real_cut_clip(src, start, end, dst, **kw)

        monkeypatch.setattr(pv2.media_mod, "cut_clip", failing_cut_clip)

        fake_kie = FakeKieClient(synthetic_video)
        from app.pipeline_v2 import process_run

        with pytest.raises(RuntimeError, match="simulated cut_clip failure"):
            process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        with sessionmaker(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.failed

    def test_stitch_concurrency_1_behaves_identically(
        self,
        db_engine,
        db_session,
        synthetic_video,
        patch_propose,
        monkeypatch,
    ):
        """
        STITCH_CUT_CONCURRENCY=1 (serial) and =2 (parallel) must both produce
        a completed run — verify the serial path still works.
        """
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(pv2, "STITCH_CUT_CONCURRENCY", 1)

        project_id, run_id = self._setup(
            db_session, db_engine, synthetic_video, patch_propose
        )

        from app.pipeline_v2 import process_run

        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=FakeGDriveClient())

        with sessionmaker(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done


# ---------------------------------------------------------------------------
# Tests: parallel stitch ordering guarantee (no ffmpeg/DB required)
# ---------------------------------------------------------------------------


class TestParallelStitchOrdering:
    """
    Pure-logic tests verifying the ThreadPoolExecutor ordering guarantee in
    _cut_or_lookup / ordered_futures.  No ffmpeg, no DB, no pipeline_v2 imports —
    tests the Python-level contract: futures collected in insertion order, results
    retrieved in that order, regardless of completion order.
    """

    def test_futures_collected_in_insertion_order(self):
        """
        When N futures are submitted to a ThreadPoolExecutor in a specific order
        and may complete in any order, collecting via ordered list (not as_completed)
        must return results in submission order.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        # Simulate segments with varying execution times to exercise out-of-order
        # completion:  seg 0 sleeps longer than seg 1.
        segments = [
            {"index": 0, "delay": 0.05, "result": "clip_0000.mp4"},
            {"index": 1, "delay": 0.01, "result": "clip_0001.mp4"},
            {"index": 2, "delay": 0.03, "result": "clip_0002.mp4"},
        ]

        def _work(seg):
            time.sleep(seg["delay"])
            return seg["result"]

        ordered_futures = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for seg in segments:
                fut = pool.submit(_work, seg)
                ordered_futures.append((seg["index"], fut))

        results = [fut.result() for _idx, fut in ordered_futures]

        # Must be in submission order, not completion order.
        assert results == ["clip_0000.mp4", "clip_0001.mp4", "clip_0002.mp4"]

    def test_future_exception_propagates_on_result(self):
        """
        If one of the futures raises, calling .result() must re-raise the exception.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _bad_work():
            raise ValueError("simulated cut error")

        def _good_work(val):
            return val

        ordered_futures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            ordered_futures.append(pool.submit(_bad_work))
            ordered_futures.append(pool.submit(_good_work, "ok"))

        with pytest.raises(ValueError, match="simulated cut error"):
            for fut in ordered_futures:
                fut.result()

    def test_concurrency_1_produces_same_order(self):
        """
        STITCH_CUT_CONCURRENCY=1 (max_workers=1) still produces results in
        insertion order — the pool serializes execution but order is preserved.
        """
        from concurrent.futures import ThreadPoolExecutor

        inputs = [f"clip_{i:04d}.mp4" for i in range(5)]
        ordered_futures = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            for item in inputs:
                fut = pool.submit(lambda x: x, item)
                ordered_futures.append(fut)

        results = [fut.result() for fut in ordered_futures]
        assert results == inputs


# ---------------------------------------------------------------------------
# Tests: perf/bottlenecks — keep-clip fast path (Fix #7, #8, #15)
# ---------------------------------------------------------------------------


class TestKeepClipPerfFixes:
    """
    Verifies the three perf fixes applied to cut_clip / _cut_or_lookup:

    Fix #7  — keep cuts use fast_intermediate=True (ultrafast/crf14).
    Fix #8  — keep cuts are checkpointed: if the dest file already exists
               with non-zero size, cut_clip is NOT called again.
    Fix #15 — swap cuts are NOT passed fast_intermediate (quality unchanged).
    """

    @staticmethod
    def _make_project_with_keep(db_session, source_path: str) -> str:
        """Project in ready state with one swap + one keep SegmentDef."""
        project_id = _create_project(
            db_session, source_path, status=ProjectStatus.ready
        )
        db_session.add_all([
            SegmentDef(
                project_id=project_id, index=0, start_sec=0.0, end_sec=5.0,
                has_face=True, action="swap",
            ),
            SegmentDef(
                project_id=project_id, index=1, start_sec=5.0, end_sec=10.0,
                has_face=False, action="keep",
            ),
        ])
        db_session.commit()
        return project_id

    @staticmethod
    def _mock_media(monkeypatch, cut_calls: list) -> None:
        """Patch probe/cut_clip/stitch so no real ffmpeg runs."""
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(
                duration_sec=10.0, width=320, height=240,
                fps=25.0, aspect_ratio="4:3", has_audio=True,
            ),
        )

        def spy_cut_clip(_src, _start, _end, dst, *,
                         include_audio=True, fast_intermediate=False):
            cut_calls.append({
                "dst": dst,
                "include_audio": include_audio,
                "fast_intermediate": fast_intermediate,
            })
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"clip")

        monkeypatch.setattr(pv2.media_mod, "cut_clip", spy_cut_clip)
        monkeypatch.setattr(
            pv2.media_mod, "stitch",
            lambda _clips, audio_source, dst, **_kw: (
                os.makedirs(os.path.dirname(dst), exist_ok=True),
                open(dst, "wb").write(b"final"),
            ),
        )

    def test_keep_cut_uses_fast_intermediate(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Keep clips must be cut with fast_intermediate=True."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = self._make_project_with_keep(db_session, str(source))
        run_id = _create_run(db_session, project_id)

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, cut_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        keep_calls = [c for c in cut_calls if "clip_0001" in c["dst"]]
        assert keep_calls, "expected a cut_clip call for the keep segment"
        assert all(c["fast_intermediate"] is True for c in keep_calls), (
            f"keep cuts must use fast_intermediate=True: {keep_calls}"
        )

    def test_swap_cut_does_not_use_fast_intermediate(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Swap clips must NOT be cut with fast_intermediate (quality unchanged)."""
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = self._make_project_with_keep(db_session, str(source))
        run_id = _create_run(db_session, project_id)

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, cut_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Swap segment (index 0) is cut for upload — must use default quality.
        swap_cuts = [c for c in cut_calls if "clip_0001" not in c["dst"]]
        assert swap_cuts, "expected at least one cut_clip call for the swap segment"
        assert all(c.get("fast_intermediate") is False for c in swap_cuts), (
            f"swap cuts must NOT use fast_intermediate: {swap_cuts}"
        )

    def test_keep_cut_reuses_existing_file(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """
        If the keep destination file already exists with non-zero size,
        cut_clip must not be called again (checkpoint reuse).
        """
        from app.pipeline_v2 import process_run
        from app.storage import keep_clip_path

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = self._make_project_with_keep(db_session, str(source))
        run_id = _create_run(db_session, project_id)

        # Pre-seed the keep clip file so it appears to have been cut already.
        # Keyed by (index, bounds) — see storage.keep_clip_path — matching
        # the keep SegmentDef _make_project_with_keep creates: index=1,
        # [5.0, 10.0).
        pre_existing = keep_clip_path(run_id, project_id, 1, 5.0, 10.0)
        with open(pre_existing, "wb") as fh:
            fh.write(b"pre-existing-clip")

        cut_calls: list[dict] = []
        self._mock_media(monkeypatch, cut_calls)

        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        keep_cuts = [c for c in cut_calls if "clip_0001" in c["dst"]]
        assert keep_cuts == [], (
            "cut_clip must NOT be called for a keep clip that already exists on disk; "
            f"got: {keep_cuts}"
        )


# ---------------------------------------------------------------------------
# Tests: staged pipeline architecture (submit_run / poll_run_once / finalize_run)
# ---------------------------------------------------------------------------


class TestStagedPipeline:
    """
    Verify that the three-stage pipeline functions (submit_run, poll_run_once,
    finalize_run) behave correctly both individually and in concert.

    These tests use the DB+media mocks from the module preamble.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)
        return project_id, run_id

    def test_submit_run_returns_pending_map(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """submit_run returns a non-empty pending map when tasks are submitted."""
        from app.pipeline_v2 import submit_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Two swap segments → two tasks submitted.
        assert pending is not None
        assert len(pending) == 2
        # Each entry has the required keys.
        for meta in pending.values():
            assert "rs_id" in meta
            assert "index" in meta
            assert "deadline" in meta
            assert "attempt" in meta
            assert "recipe" in meta
        # Two uploads and two task creates.
        assert len(fake_kie.upload_calls) == 2
        assert len(fake_kie.create_task_calls) == 2

    def test_submit_run_returns_empty_when_all_segments_completed(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """submit_run returns {} when all swap segments are already completed."""
        from app.pipeline_v2 import analyze_project, submit_run
        from app.storage import run_results_dir

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)
        r_dir = run_results_dir(run_id, project_id)

        # Pre-complete both swap segments.
        from sqlalchemy.orm import sessionmaker as _sm
        Session = _sm(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            for sd in [seg for seg in project.segments if seg.action == "swap"]:
                result = os.path.join(r_dir, f"result_{sd.index:04d}.mp4")
                shutil.copy2(synthetic_video, result)
                s.add(RunSegment(
                    run_id=run_id, segment_def_id=sd.id, index=sd.index,
                    status=SegmentStatus.completed, local_result_path=result,
                ))
            s.commit()

        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert pending == {}
        # No uploads (all segments already done).
        assert len(fake_kie.upload_calls) == 0

    def test_submit_run_returns_none_for_terminal_run(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """submit_run returns None when the run is already in a terminal state."""
        from app.pipeline_v2 import submit_run

        project_id = _create_project(db_session, synthetic_video, status=ProjectStatus.ready)
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="t", prompt="p",
            reference_image_urls=[],
            status=RunStatus.done,
        )
        db_session.add(run)
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        result = submit_run(run.id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert result is None
        assert len(fake_kie.upload_calls) == 0

    def test_poll_run_once_returns_empty_when_all_succeed(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """poll_run_once returns {} when all pending tasks succeed in one round."""
        from app.pipeline_v2 import analyze_project, submit_run, poll_run_once

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert pending

        # poll_run_once: FakeKieClient always returns success on get_task.
        remaining = poll_run_once(run_id, pending=pending, kie=fake_kie)
        assert remaining == {}

    def test_poll_run_once_leaves_pending_when_still_generating(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """poll_run_once leaves tasks in pending when state is still in-progress."""
        from app.pipeline_v2 import analyze_project, submit_run, poll_run_once

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        class AlwaysGeneratingKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                self.poll_calls.append(task_id)
                return {"state": "processing"}

        fake_kie = AlwaysGeneratingKie(synthetic_video)
        pending = submit_run(run_id, kie=FakeKieClient(synthetic_video),
                             gdrive=FakeGDriveClient())
        # Poll with a client that always says "processing".
        remaining = poll_run_once(run_id, pending=pending, kie=fake_kie)
        # All tasks should still be in remaining (not terminal yet).
        assert len(remaining) == len(pending)

    def test_finalize_run_completes_when_all_segments_done(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """finalize_run produces a stitched video when all swap segments are complete."""
        from app.pipeline_v2 import analyze_project, submit_run, poll_run_once, finalize_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())
        remaining = poll_run_once(run_id, pending=pending, kie=fake_kie)
        assert remaining == {}

        finalize_run(run_id, gdrive=FakeGDriveClient())

        from sqlalchemy.orm import sessionmaker as _sm
        with _sm(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_local_path is not None
            assert os.path.exists(run.result_local_path)

    def test_finalize_run_returns_the_published_final(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """AICP-299 §3.2: finalize_run returns the committed PublishedFinal."""
        from app.knowledge.publication import PublishedFinal
        from app.pipeline_v2 import submit_run, poll_run_once, finalize_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)
        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())
        assert poll_run_once(run_id, pending=pending, kie=fake_kie) == {}

        published = finalize_run(run_id, gdrive=FakeGDriveClient(), publication_id=f"stitch:{run_id}:job-1")
        from sqlalchemy.orm import sessionmaker as _sm
        with _sm(bind=db_engine)() as s:
            run = s.get(Run, run_id)
        assert isinstance(published, PublishedFinal)
        assert published == PublishedFinal(run_id, project_id, run.result_local_path, "stitch",
                                           f"stitch:{run_id}:job-1")

    def test_poll_run_once_reenqueues_itself_while_pending(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """
        poll_run_once returns non-empty when tasks remain; the tasks.py layer is
        responsible for re-enqueuing — verify that poll_run_once returns non-empty
        so callers know to re-enqueue.
        """
        from app.pipeline_v2 import submit_run, poll_run_once

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        first_call = {"done": False}

        class OnceGeneratingKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                self.poll_calls.append(task_id)
                if not first_call["done"]:
                    first_call["done"] = True
                    return {"state": "processing"}
                url = f"https://fake/results/{task_id}.mp4"
                return {"state": "success",
                        "resultJson": json.dumps({"resultUrls": [url]})}

        fake_kie = OnceGeneratingKie(synthetic_video)
        pending = submit_run(run_id, kie=FakeKieClient(synthetic_video),
                             gdrive=FakeGDriveClient())

        # First poll round: first task is still "processing" (returns non-empty).
        remaining1 = poll_run_once(run_id, pending=pending, kie=fake_kie)
        # At least one task should still be pending.
        assert len(remaining1) > 0, "Expected non-empty remaining after first round"

    def test_finalize_run_marks_incomplete_when_segment_failed(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """finalize_run marks run incomplete when a swap segment is not completed."""
        from app.pipeline_v2 import analyze_project, finalize_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        # Manually create RunSegments with one failed.
        from sqlalchemy.orm import sessionmaker as _sm
        Session = _sm(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            swap_defs = [sd for sd in project.segments if sd.action == "swap"]
            for i, sd in enumerate(swap_defs):
                rs = RunSegment(
                    run_id=run_id, segment_def_id=sd.id, index=sd.index,
                    status=SegmentStatus.failed if i == 0 else SegmentStatus.completed,
                )
                if i != 0:
                    result = os.path.join(
                        __import__("app.storage", fromlist=["run_results_dir"]).run_results_dir(run_id, project_id),
                        f"result_{sd.index:04d}.mp4"
                    )
                    shutil.copy2(synthetic_video, result)
                    rs.local_result_path = result
                s.add(rs)
            # Advance run to processing state.
            run = s.get(Run, run_id)
            run.status = RunStatus.processing
            s.commit()

        finalize_run(run_id, gdrive=FakeGDriveClient())

        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.incomplete
            assert run.result_local_path is None

    def test_state_survives_across_poll_invocations(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """
        The pending map returned by submit_run can be serialised, deserialised,
        and passed to a subsequent poll_run_once call — simulating the Redis
        persistence between poll rounds.
        """
        import json
        from app.pipeline_v2 import submit_run, poll_run_once

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)

        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())
        assert pending

        # Simulate Redis round-trip: JSON encode + decode.
        serialised = json.dumps(pending)
        restored = json.loads(serialised)

        # Verify the restored map still works as input to poll_run_once.
        remaining = poll_run_once(run_id, pending=restored, kie=fake_kie)
        assert remaining == {}  # all tasks completed


# ---------------------------------------------------------------------------
# Tests: Fix #9 — crash-resume skips re-cut/re-upload when clip already on kie
# ---------------------------------------------------------------------------


class TestCrashResumeClipCheckpoint:
    """
    Verify Fix #9: when a RunSegment has kie_upload_url already set (the clip
    was previously uploaded to kie.ai), submit_run must NOT call cut_clip or
    upload_file for that segment.  Only create_task is called with the stored URL.

    If only local_clip_path exists (clip cut but upload failed), submit_run must
    skip the cut but still call upload_file.
    """

    @staticmethod
    def _make_project_with_one_swap(db_session, source_path: str) -> str:
        project_id = _create_project(
            db_session, source_path, status=ProjectStatus.ready
        )
        db_session.add(
            SegmentDef(
                project_id=project_id, index=0, start_sec=0.0, end_sec=5.0,
                has_face=True, action="swap",
            )
        )
        db_session.commit()
        return project_id

    @staticmethod
    def _mock_media_no_cut(monkeypatch) -> list:
        """Patch media so cut_clip records calls but doesn't need a real source."""
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(
                duration_sec=10.0, width=320, height=240,
                fps=25.0, aspect_ratio="4:3", has_audio=True,
            ),
        )
        cut_calls: list[dict] = []

        def spy_cut_clip(_src, _start, _end, dst, **_kw):
            cut_calls.append(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(b"clip")

        monkeypatch.setattr(pv2.media_mod, "cut_clip", spy_cut_clip)
        monkeypatch.setattr(
            pv2.media_mod, "stitch",
            lambda _clips, audio_source, dst, **_kw: (
                os.makedirs(os.path.dirname(dst), exist_ok=True),
                open(dst, "wb").write(b"final"),
            ),
        )
        return cut_calls

    def test_resume_skips_cut_and_upload_when_kie_url_stored(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """
        Fix #9: If rs.kie_upload_url is already set (clip previously uploaded),
        submit_run must skip cut_clip AND upload_file, and go straight to
        create_task with the stored URL.
        """
        from app.pipeline_v2 import submit_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = self._make_project_with_one_swap(db_session, str(source))
        run_id = _create_run(db_session, project_id)

        cut_calls = self._mock_media_no_cut(monkeypatch)

        # Pre-create a RunSegment that was uploading (clip already on kie.ai).
        from sqlalchemy.orm import sessionmaker as _sm
        Session = _sm(bind=db_engine)
        stored_url = "https://fake-kie.example.com/files/pre-uploaded-clip.mp4"
        with Session() as s:
            project = s.get(VideoProject, project_id)
            sd = next(seg for seg in project.segments if seg.action == "swap")
            rs = RunSegment(
                run_id=run_id,
                segment_def_id=sd.id,
                index=sd.index,
                status=SegmentStatus.uploading,  # crashed mid-upload
                local_clip_path=str(tmp_path / "clip_0000.mp4"),
                kie_upload_url=stored_url,
            )
            s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(str(source))
        # submit_run goes straight to task creation
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # No cut_clip call — clip checkpoint reused.
        assert cut_calls == [], (
            f"cut_clip must not be called when kie_upload_url is set; got: {cut_calls}"
        )
        # No upload_file call — stored URL reused.
        assert len(fake_kie.upload_calls) == 0, (
            f"upload_file must not be called when kie_upload_url is set; "
            f"got: {fake_kie.upload_calls}"
        )
        # create_task must be called with the stored URL.
        assert len(fake_kie.create_task_calls) == 1
        assert pending is not None
        # The recipe in the pending map must use the pre-stored URL.
        task_id = list(pending.keys())[0]
        assert pending[task_id]["recipe"]["clip_url"] == stored_url

    def test_resume_skips_cut_but_reuploads_when_local_clip_exists(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """
        Fix #9: If only local_clip_path exists (clip cut but upload failed),
        submit_run must skip cut_clip but still call upload_file.
        """
        from app.pipeline_v2 import submit_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = self._make_project_with_one_swap(db_session, str(source))
        run_id = _create_run(db_session, project_id)

        cut_calls = self._mock_media_no_cut(monkeypatch)

        # Pre-create the clip file on disk (simulating a cut that completed
        # but the upload was interrupted).
        clip_path = tmp_path / "clip_0000.mp4"
        clip_path.write_bytes(b"already-cut-clip")

        from sqlalchemy.orm import sessionmaker as _sm
        Session = _sm(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            sd = next(seg for seg in project.segments if seg.action == "swap")
            rs = RunSegment(
                run_id=run_id,
                segment_def_id=sd.id,
                index=sd.index,
                status=SegmentStatus.uploading,  # crashed during upload
                local_clip_path=str(clip_path),
                kie_upload_url=None,  # upload never completed
            )
            s.add(rs)
            s.commit()

        fake_kie = FakeKieClient(str(source))
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # cut_clip must NOT be called.
        assert cut_calls == [], (
            f"cut_clip must not be called when local_clip_path exists; "
            f"got: {cut_calls}"
        )
        # upload_file MUST be called (re-uploading the existing clip).
        assert len(fake_kie.upload_calls) == 1, (
            f"upload_file must be called once to re-upload the existing clip; "
            f"got: {fake_kie.upload_calls}"
        )
        # create_task must succeed.
        assert len(fake_kie.create_task_calls) == 1
        assert pending is not None


# ---------------------------------------------------------------------------
# Tests: Fix #16 — concurrent downloads in poll_run_once
# ---------------------------------------------------------------------------


class TestConcurrentPollDownloads:
    """
    Verify Fix #16: when multiple tasks complete in the same poll round,
    download_result is called for all of them (concurrent path in
    poll_run_once_internal).
    """

    def test_all_completed_tasks_downloaded_in_one_round(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch
    ):
        """
        When all pending tasks are complete in a single poll round,
        every segment gets its result downloaded.
        """
        from app.pipeline_v2 import submit_run, poll_run_once

        from app.pipeline_v2 import analyze_project
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())
        assert len(pending) == 2, f"Expected 2 pending tasks, got {len(pending)}"

        # Poll once — FakeKieClient returns success for all tasks.
        remaining = poll_run_once(run_id, pending=pending, kie=fake_kie)
        assert remaining == {}, f"Expected empty remaining after one round, got {remaining}"

        # Both segments should be completed with result files.
        from sqlalchemy.orm import sessionmaker as _sm
        with _sm(bind=db_engine)() as s:
            run = s.get(Run, run_id)
            run_segs = list(run.run_segments)
            assert all(rs.status == SegmentStatus.completed for rs in run_segs), (
                f"All segments should be completed; statuses: "
                f"{[rs.status for rs in run_segs]}"
            )
            for rs in run_segs:
                assert rs.local_result_path is not None
                assert os.path.exists(rs.local_result_path)

    def test_concurrent_downloads_bounded_by_env_var(
        self, monkeypatch, tmp_path
    ):
        """
        POLL_DOWNLOAD_CONCURRENCY env-var controls the ThreadPoolExecutor size.
        With 3 completed tasks and concurrency=2, all 3 downloads complete and
        no task is lost.
        """
        import app.pipeline_v2 as pv2
        monkeypatch.setattr(pv2, "POLL_DOWNLOAD_CONCURRENCY", 2)

        completed_downloads = []

        class _FakeRs:
            def __init__(self):
                self.status = SegmentStatus.generating
                self.seedance_result_url = None
                self.local_result_path = None
                # Real RunSegments carry this; the truncated-result guard
                # reads it to compare against the returned clip.
                self.local_clip_path = None
                self.cost_credits = None
                self.error_message = None

        rs_store: dict[str, "_FakeRs"] = {
            f"rs-{i}": _FakeRs() for i in range(3)
        }

        class _FakeSession:
            def __init__(self, rs_id):
                self._rs_id = rs_id

            def get(self, _model, rs_id):
                return rs_store[rs_id]

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        @contextmanager
        def fake_get_session():
            # The session is used per-download-thread; we track calls by rs_id
            # via a shared closure—but we don't know which rs_id ahead of time.
            # Return a session that returns any rs from rs_store by id.
            class _S:
                def get(self, _model, rs_id):
                    return rs_store[rs_id]
                def commit(self): pass
                def rollback(self): pass
                def close(self): pass
            yield _S()

        monkeypatch.setattr(pv2, "get_session", fake_get_session)
        monkeypatch.setattr(
            pv2, "transition", lambda obj, status: setattr(obj, "status", status)
        )

        class _Kie:
            def get_task(self, task_id):
                url = f"https://fake/results/{task_id}.mp4"
                return {"state": "success",
                        "resultJson": json.dumps({"resultUrls": [url]})}

            def download_result(self, url, dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                with open(dst_path, "wb") as fh:
                    fh.write(b"result")
                completed_downloads.append(url)

        pending = {
            f"task-{i}": {
                "rs_id": f"rs-{i}",
                "index": i,
                "deadline": pv2.time.time() + 60,
            }
            for i in range(3)
        }

        remaining = pv2.poll_run_once_internal(
            pending=pending,
            kie=_Kie(),
            results_dir=str(tmp_path),
        )

        assert remaining == {}
        assert len(completed_downloads) == 3, (
            f"Expected 3 downloads, got {len(completed_downloads)}"
        )
        for rs in rs_store.values():
            assert rs.status == SegmentStatus.completed


# ---------------------------------------------------------------------------
# Tests: stitch plans (custom stitch / no stitch / library clips)
# ---------------------------------------------------------------------------


class TestStitchPlan:
    """process_run with Run.stitch_plan set.

    Fixed partition (patch_propose): swap 0-9 | keep 9-15 | swap 15-25.
    """

    def _ready_project(self, db_session, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        db_session.expire_all()
        segs = (
            db_session.query(SegmentDef)
            .filter_by(project_id=project_id)
            .order_by(SegmentDef.index)
            .all()
        )
        return project_id, segs

    def _make_clip(self, db_session, synthetic_video) -> str:
        clip = LibraryClip(
            id=str(uuid.uuid4()),
            name="Outro test",
            local_path=synthetic_video,
            duration_sec=25.0,
        )
        db_session.add(clip)
        db_session.commit()
        return clip.id

    def test_custom_plan_replaces_outro_with_library_clip(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """Outro swap segment excluded → not submitted; library clip stitched in."""
        from app.pipeline_v2 import process_run
        from sqlalchemy.orm import sessionmaker

        project_id, segs = self._ready_project(
            db_session, synthetic_video, patch_propose
        )
        clip_id = self._make_clip(db_session, synthetic_video)
        # Keep swap #0 and keep #1; replace the final swap (outro) with the clip.
        plan = [
            {"kind": "segment", "segment_def_id": segs[0].id},
            {"kind": "segment", "segment_def_id": segs[1].id},
            {"kind": "library", "clip_id": clip_id},
        ]
        run_id = _create_run(db_session, project_id, stitch_plan=plan)

        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()
        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        # Only the ONE swap segment in the plan was submitted (credits saved).
        assert len(fake_kie.create_task_calls) == 1

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_local_path is not None
            assert os.path.exists(run.result_local_path)
            # No RunSegment exists for the excluded outro swap segment.
            rs_seg_ids = {rs.segment_def_id for rs in run.run_segments}
            assert segs[0].id in rs_seg_ids
            assert segs[2].id not in rs_seg_ids

    def test_empty_plan_processes_all_swaps_but_skips_stitch(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """stitch_plan == [] → both swaps generated, no final video, no delivery."""
        from app.pipeline_v2 import process_run
        from sqlalchemy.orm import sessionmaker

        project_id, _segs = self._ready_project(
            db_session, synthetic_video, patch_propose
        )
        run_id = _create_run(
            db_session, project_id, stitch_plan=[],
            gdrive_folder_id="folder-123",
        )

        fake_kie = FakeKieClient(synthetic_video)
        fake_gdrive = FakeGDriveClient()
        process_run(run_id, kie=fake_kie, gdrive=fake_gdrive)

        # Both swap segments were still processed.
        assert len(fake_kie.create_task_calls) == 2
        # ...but nothing was stitched or delivered.
        assert fake_gdrive.upload_calls == []

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_local_path is None
            assert run.result_gdrive_file_id is None
            statuses = {rs.status for rs in run.run_segments}
            assert statuses == {SegmentStatus.completed}

    def test_custom_plan_missing_library_clip_fails_run(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A plan referencing a deleted clip fails the run with a clear error."""
        from app.pipeline_v2 import process_run
        from sqlalchemy.orm import sessionmaker

        project_id, segs = self._ready_project(
            db_session, synthetic_video, patch_propose
        )
        plan = [
            {"kind": "segment", "segment_def_id": segs[0].id},
            {"kind": "library", "clip_id": "deleted-clip-id"},
        ]
        run_id = _create_run(db_session, project_id, stitch_plan=plan)

        fake_kie = FakeKieClient(synthetic_video)
        with pytest.raises(RuntimeError, match="no longer exists"):
            process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.failed


# ---------------------------------------------------------------------------
# Tests: preprocess_reference_images (GPT Image 2 phase)
# ---------------------------------------------------------------------------


class TestPreprocessReferenceImages:
    """Tests for the optional GPT Image 2 reference preprocessing phase."""

    def _create_preprocess_run(
        self,
        db_session,
        project_id,
        *,
        refs,
        prompts,
        status=RunStatus.preprocessing,
        task_ids=None,
    ):
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Preprocess Run",
            prompt="swap",
            reference_image_urls=refs,
            resolution="480p",
            status=status,
            preprocess_enabled=True,
            preprocess_image_prompts=prompts,
            preprocess_task_ids=task_ids,
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def test_disabled_returns_empty(self, db_session, synthetic_video):
        """preprocess_enabled=False → no tasks created, empty pending."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            prompt="swap",
            reference_image_urls=["https://x.test/a.jpg"],
            resolution="480p",
            status=RunStatus.preprocessing,
            preprocess_enabled=False,
        )
        db_session.add(run)
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run.id, kie=fake_kie)
        assert pending == {}
        assert fake_kie.create_image_records == []

    def test_creates_task_per_ref_with_prompt(self, db_session, synthetic_video):
        """R1-new: one GPT Image 2 task per TRIM (index 1..N-1) with a non-empty
        prompt. Index 0 is the user ref — always pass-through. Each task's
        input_urls = [refs[0], refs[i]] (user ref + that trim)."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg", "https://x.test/c.jpg"]
        # prompts[0] = "" (user ref pass-through); trims at 1, 2.
        prompts = ["", "make a redshirt", "make a blueshirt"]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)

        # 2 trims with prompts → 2 tasks.
        assert len(pending) == 2
        # pending maps task_id → trim_index (1 and 2, NOT 0).
        assert sorted(pending.values()) == [1, 2]
        assert len(fake_kie.create_image_records) == 2
        for rec in fake_kie.create_image_records:
            assert rec["aspect_ratio"] == "auto"
            assert rec["resolution"] == "1K"
            # R1-new: input_urls = [user_ref, trim] (length 2).
            assert len(rec["input_urls"]) == 2
            assert rec["input_urls"][0] == "https://x.test/a.jpg"
        assert {r["prompt"] for r in fake_kie.create_image_records} == {
            "make a redshirt", "make a blueshirt",
        }
        # User ref (index 0) is never transformed.
        assert all("make a" not in (r.get("prompt", "")) or r["prompt"] != "" for r in fake_kie.create_image_records)

    def test_pass_through_refs_get_no_task(self, db_session, synthetic_video):
        """R1-new: trims with an empty prompt are skipped (pass-through). Index 0
        (user ref) is always pass-through."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg", "https://x.test/c.jpg", "https://x.test/d.jpg"]
        # prompts[0] = "" (user ref); trims 1, 2, 3 — only trim 1 has a prompt.
        prompts = ["", "transform b", "   ", ""]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)

        assert len(pending) == 1
        assert list(pending.values()) == [1]
        assert len(fake_kie.create_image_records) == 1
        assert fake_kie.create_image_records[0]["prompt"] == "transform b"

    def test_all_pass_through_returns_empty(self, db_session, synthetic_video):
        """R1-new: all trims pass-through (and the user ref is always pass-
        through) → empty pending (caller pauses for review)."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg"]
        prompts = ["", ""]  # user ref + one pass-through trim
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)
        assert pending == {}
        assert fake_kie.create_image_records == []

    def test_no_trims_is_noop(self, db_session, synthetic_video):
        """R1-new: a run with only the user ref (no trims) → no-op (empty
        pending)."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg"]  # only the user ref
        prompts = [""]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)
        assert pending == {}
        assert fake_kie.create_image_records == []

    def test_persists_task_ids_positional(self, db_session, db_engine, synthetic_video):
        """R1-new: preprocess_task_ids is stored parallel to reference_image_urls;
        index 0 (user ref) and pass-through trims are None."""
        from app.pipeline_v2 import preprocess_reference_images
        from sqlalchemy.orm import sessionmaker

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg"]
        prompts = ["", "transform b"]  # user ref + one trim with a prompt
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert len(run.preprocess_task_ids) == 2
            assert run.preprocess_task_ids[0] is None  # user ref pass-through
            assert run.preprocess_task_ids[1] is not None  # trim has a task
            assert run.preprocess_task_ids[1] in pending

    def test_resume_does_not_rebill(self, db_session, synthetic_video):
        """R1-new: on resume, trims with an existing task_id are not resubmitted."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg"]
        prompts = ["", "transform b"]
        existing_task = "prior-task-abc"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[None, existing_task],
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)

        # Existing task is returned for the poll phase to re-check, but no new
        # task is created (no rebill).
        assert pending == {existing_task: 1}
        assert fake_kie.create_image_records == []

    def test_ref_with_a_downloaded_result_is_neither_rebilled_nor_repolled(
        self, db_engine, db_session, synthetic_video
    ):
        """The single-image regen: the API cleared trim 2's task id and result
        and re-entered preprocessing. Trim 1 still holds its image, so it must
        not be resubmitted (rebill) NOR handed back for polling — its kie.ai
        task is old and may be long gone."""
        from app.pipeline_v2 import preprocess_reference_images
        from sqlalchemy.orm import sessionmaker

        project_id = _create_project(db_session, synthetic_video)
        refs = [
            "https://x.test/char.jpg", "https://x.test/t1.jpg",
            "https://x.test/t2.jpg",
        ]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs,
            prompts=["", "pose it", "pose it"],
            task_ids=[None, "task1", None],
        )
        run = db_session.get(Run, run_id)
        run.preprocessed_reference_urls = [refs[0], "/d/pp/ref_0001.png", None]
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)

        # Exactly one new task, for the cleared trim only.
        assert len(fake_kie.create_image_records) == 1
        assert list(pending.values()) == [2]
        # Trim 1 is absent from pending — nothing re-polls it.
        assert "task1" not in pending
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            # Its task id stays as the audit trail of what produced the image.
            assert run.preprocess_task_ids[1] == "task1"
            assert run.preprocess_task_ids[2] in pending

    def test_in_flight_task_without_a_result_is_still_repolled(
        self, db_session, synthetic_video
    ):
        """The other half of the same rule: a task that has NOT produced a file
        yet (worker restarted mid-phase) is re-polled, not recreated."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/char.jpg", "https://x.test/t1.jpg"]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=["", "pose it"],
            task_ids=[None, "task1"],
        )
        run = db_session.get(Run, run_id)
        run.preprocessed_reference_urls = [refs[0], None]
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)

        assert pending == {"task1": 1}
        assert fake_kie.create_image_records == []

    def test_prompt_length_mismatch_is_lenient(self, db_session, synthetic_video):
        """R1-new: prompts length != refs length no longer raises — the pipeline
        treats missing trim prompts as pass-through. Prompt reuse/padding
        happens at the API layer."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg"]
        # prompts has only the user-ref slot (index 0 = ""); trim 1 has no entry
        # → pass-through (no task).
        prompts = [""]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)
        # Trim 1 has no prompt → pass-through → no tasks.
        assert pending == {}
        assert fake_kie.create_image_records == []

    def test_no_refs_returns_empty(self, db_session, synthetic_video):
        """A run with no refs → empty pending (nothing to preprocess)."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=[], prompts=[]
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)
        assert pending == {}
        assert fake_kie.create_image_records == []


# ---------------------------------------------------------------------------
# Tests: poll_preprocess_once (GPT Image 2 poll phase)
# ---------------------------------------------------------------------------


class TestPollPreprocessOnce:
    """Tests for the GPT Image 2 poll phase (poll_preprocess_once)."""

    def _create_preprocess_run(
        self,
        db_session,
        project_id,
        *,
        refs,
        prompts,
        task_ids=None,
        preprocessed=None,
    ):
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="PP Poll Run",
            prompt="swap",
            reference_image_urls=refs,
            resolution="480p",
            status=RunStatus.preprocessing,
            preprocess_enabled=True,
            preprocess_image_prompts=prompts,
            preprocess_task_ids=task_ids,
            preprocessed_reference_urls=preprocessed,
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def test_success_downloads_and_records_path(
        self, db_engine, db_session, synthetic_video
    ):
        """A successful task → image downloaded into preprocessed dir, path
        recorded in preprocessed_reference_urls."""
        from app.pipeline_v2 import poll_preprocess_once
        from sqlalchemy.orm import sessionmaker

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg"]
        prompts = ["transform a", ""]  # ref 1 is pass-through
        task_a = "task-a-123"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[task_a, None],
        )

        fake_kie = FakeKieClient(synthetic_video)
        result = poll_preprocess_once(
            run_id, pending={task_a: 0}, kie=fake_kie,
        )

        assert result["pending"] == {}  # task terminal
        assert result["failed"] == []
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            pp = run.preprocessed_reference_urls
            assert len(pp) == 2
            # ref 0: downloaded preprocessed image (local path in preprocessed dir)
            assert pp[0] is not None
            assert os.path.exists(pp[0])
            assert os.path.basename(pp[0]).startswith("ref_0000.")
            # ref 1: pass-through → original ref
            assert pp[1] == "https://x.test/b.jpg"

    def test_fail_records_failure_no_fallback(
        self, db_engine, db_session, synthetic_video
    ):
        """R2: a failed task → recorded in ``failed`` list, no fallback to
        the original ref."""
        from app.pipeline_v2 import poll_preprocess_once
        from sqlalchemy.orm import sessionmaker

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg"]
        prompts = ["transform a"]
        task_a = "fail-task-1"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[task_a],
        )

        fake_kie = FakeKieClient(synthetic_video, fail_task_id=task_a)
        result = poll_preprocess_once(
            run_id, pending={task_a: 0}, kie=fake_kie,
        )

        assert result["pending"] == {}  # fail is terminal
        assert len(result["failed"]) == 1
        assert result["failed"][0]["task_id"] == task_a
        assert "failed" in result["failed"][0]["reason"]
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            # R2: no fallback — preprocessed slot stays None.
            assert run.preprocessed_reference_urls[0] is None

    def test_success_no_result_url_records_failure(
        self, db_engine, db_session, synthetic_video
    ):
        """R2: a task that succeeds but returns no resultUrl → recorded as
        failed (no fallback)."""
        from app.pipeline_v2 import poll_preprocess_once

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg"]
        prompts = ["transform a"]
        task_a = "no-url-task-1"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[task_a],
        )

        class _NoUrlKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                return {"state": "success"}  # no resultUrl

        fake_kie = _NoUrlKie(synthetic_video)
        result = poll_preprocess_once(
            run_id, pending={task_a: 0}, kie=fake_kie,
        )
        assert result["pending"] == {}
        assert len(result["failed"]) == 1
        assert "no result URL" in result["failed"][0]["reason"]

    def test_download_error_records_failure(
        self, db_engine, db_session, synthetic_video
    ):
        """R2: a download failure → recorded as failed (no fallback)."""
        from app.pipeline_v2 import poll_preprocess_once

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg"]
        prompts = ["transform a"]
        task_a = "dl-err-task-1"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[task_a],
        )

        class _DlErrKie(FakeKieClient):
            def download_result(self, url, dst, **kw):
                raise RuntimeError("disk full")

        fake_kie = _DlErrKie(synthetic_video)
        result = poll_preprocess_once(
            run_id, pending={task_a: 0}, kie=fake_kie,
        )
        assert result["pending"] == {}
        assert len(result["failed"]) == 1
        assert "Failed to download" in result["failed"][0]["reason"]

    def test_waiting_remains_pending(self, db_session, synthetic_video):
        """A still-waiting task stays in the pending map for the next round."""
        from app.pipeline_v2 import poll_preprocess_once

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg"]
        prompts = ["transform a"]
        task_a = "waiting-task-1"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[task_a],
        )

        class _WaitingKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                return {"state": "waiting"}

        fake_kie = _WaitingKie(synthetic_video)
        result = poll_preprocess_once(
            run_id, pending={task_a: 0}, kie=fake_kie,
        )
        assert result["pending"] == {task_a: 0}
        assert result["failed"] == []

    def test_get_task_error_keeps_pending(self, db_session, synthetic_video):
        """A transient get_task error keeps the task pending (retry next round)."""
        from app.pipeline_v2 import poll_preprocess_once

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg"]
        prompts = ["transform a"]
        task_a = "err-task-1"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[task_a],
        )

        class _ErrorKie(FakeKieClient):
            def get_task(self, task_id, **kw):
                raise RuntimeError("transient network error")

        fake_kie = _ErrorKie(synthetic_video)
        result = poll_preprocess_once(
            run_id, pending={task_a: 0}, kie=fake_kie,
        )
        assert result["pending"] == {task_a: 0}
        assert result["failed"] == []

    def test_pass_through_filled_with_original(
        self, db_engine, db_session, synthetic_video
    ):
        """Pass-through refs get the original ref in preprocessed_reference_urls
        even when no tasks are pending for them."""
        from app.pipeline_v2 import poll_preprocess_once
        from sqlalchemy.orm import sessionmaker

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg", "https://x.test/c.jpg"]
        prompts = ["", "", ""]  # all pass-through
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[None, None, None],
        )

        fake_kie = FakeKieClient(synthetic_video)
        # No pending tasks at all — but pass-throughs should still be filled.
        result = poll_preprocess_once(run_id, pending={}, kie=fake_kie)
        assert result["pending"] == {}
        assert result["failed"] == []

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.preprocessed_reference_urls == refs

    def test_a_slot_awaiting_its_own_task_is_not_filled_with_the_original(
        self, db_engine, db_session, synthetic_video
    ):
        """Pass-through means "this ref has no prompt", not "no task id yet".

        A single-image regen clears one slot's task id and result on purpose and
        the new task is created a moment later; filling the slot with the
        untouched original in between would quietly offer the operator an
        un-preprocessed frame to approve."""
        from app.pipeline_v2 import poll_preprocess_once
        from sqlalchemy.orm import sessionmaker

        project_id = _create_project(db_session, synthetic_video)
        refs = [
            "https://x.test/char.jpg", "https://x.test/t1.jpg",
            "https://x.test/t2.jpg",
        ]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs,
            # Trim 2 carries a prompt but has no task id and no result yet.
            prompts=["", "pose it", "pose it"],
            task_ids=[None, "task1", None],
        )
        run = db_session.get(Run, run_id)
        run.preprocessed_reference_urls = [refs[0], "/d/pp/ref_0001.png", None]
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        result = poll_preprocess_once(run_id, pending={}, kie=fake_kie)
        assert result["failed"] == []

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.preprocessed_reference_urls[2] is None

    def test_user_ref_pass_through_at_index_zero(
        self, db_engine, db_session, synthetic_video
    ):
        """R1-new: index 0 (user ref) is always pass-through — poll fills
        preprocessed_reference_urls[0] with refs[0] even when a task exists for
        a trim."""
        from app.pipeline_v2 import poll_preprocess_once
        from sqlalchemy.orm import sessionmaker

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/user.jpg", "https://x.test/trim1.jpg"]
        # prompts[0] = "" (user ref); trim 1 has a task.
        prompts = ["", "transform trim"]
        task_trim = "task-trim-1"
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
            task_ids=[None, task_trim],
        )

        fake_kie = FakeKieClient(synthetic_video)
        result = poll_preprocess_once(
            run_id, pending={task_trim: 1}, kie=fake_kie,
        )
        assert result["pending"] == {}
        assert result["failed"] == []

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            pp = run.preprocessed_reference_urls
            # Index 0 = user ref pass-through (the original ref, NOT a downloaded image).
            assert pp[0] == "https://x.test/user.jpg"
            # Index 1 = downloaded preprocessed trim (local path).
            assert pp[1] is not None
            assert os.path.exists(pp[1])
            assert os.path.basename(pp[1]).startswith("ref_0001.")


# ---------------------------------------------------------------------------
# Tests: preprocess_reference_images — all_as_refs input_urls (R1-new)
# ---------------------------------------------------------------------------


class TestPreprocessAllAsRefsInputUrls:
    """R1-new: all_as_refs builds input_urls from ALL refs (user ref + trims)."""

    def _create_preprocess_run(
        self,
        db_session,
        project_id,
        *,
        refs,
        prompts,
        mode="all_as_refs",
        task_ids=None,
    ):
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="PP all_as_refs Run",
            prompt="swap",
            reference_image_urls=refs,
            resolution="480p",
            status=RunStatus.preprocessing,
            preprocess_enabled=True,
            preprocess_mode=mode,
            preprocess_image_prompts=prompts,
            preprocess_task_ids=task_ids,
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def test_all_as_refs_input_urls_is_all_refs(
        self, db_session, synthetic_video
    ):
        """R1-new: all_as_refs sends input_urls = [refs[0], refs[1], ..., refs[N-1]]
        (user ref + every trim) as a single task."""
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        refs = ["https://x.test/a.jpg", "https://x.test/b.jpg", "https://x.test/c.jpg"]
        # all_as_refs stores a length-1 prompts list (the single trim prompt).
        prompts = ["combine all"]
        run_id = self._create_preprocess_run(
            db_session, project_id, refs=refs, prompts=prompts,
        )

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run_id, kie=fake_kie)

        assert len(pending) == 1
        assert list(pending.values()) == [0]
        assert len(fake_kie.create_image_records) == 1
        rec = fake_kie.create_image_records[0]
        # R1-new: input_urls = ALL refs (user ref + every trim).
        assert rec["input_urls"] == refs
        assert rec["prompt"] == "combine all"


# ---------------------------------------------------------------------------
# Tests: submit_run ref-source switch (preprocessed vs originals)
# ---------------------------------------------------------------------------


class TestSubmitRunRefSourceSwitch:
    """submit_run uses preprocessed_reference_urls when preprocessing is on."""

    def _ready_project(self, db_session, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        db_session.expire_all()
        return project_id

    def test_uses_preprocessed_when_enabled(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """When preprocess_enabled + preprocessed_reference_urls set, those are
        the refs Seedance receives (not the originals)."""
        from app.pipeline_v2 import submit_run
        from sqlalchemy.orm import sessionmaker

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="PP Run",
            prompt="swap",
            reference_image_urls=["https://orig.test/face.jpg"],
            resolution="480p",
            status=RunStatus.queued,
            preprocess_enabled=True,
            preprocessed_reference_urls=["https://pp.test/face.png"],
        )
        db_session.add(run)
        db_session.commit()
        run_id = run.id

        fake_kie = FakeKieClient(synthetic_video)
        submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # The fixed project has 2 swap segments → 2 Seedance tasks, each must
        # have received the preprocessed ref, not the original.
        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["reference_image_urls"] == ["https://pp.test/face.png"]

    def test_uses_originals_when_disabled(
        self, db_session, synthetic_video, patch_propose
    ):
        """preprocess_enabled=False → originals used (existing behaviour)."""
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        run_id = _create_run(
            db_session, project_id,
            reference_image_urls=["https://orig.test/face.jpg"],
        )

        fake_kie = FakeKieClient(synthetic_video)
        submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["reference_image_urls"] == ["https://orig.test/face.jpg"]

    def test_uses_originals_when_preprocessed_empty(
        self, db_session, synthetic_video, patch_propose
    ):
        """preprocess_enabled but preprocessed_reference_urls is None/empty →
        originals used (preprocessing not yet run or all pass-through)."""
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="PP Run",
            prompt="swap",
            reference_image_urls=["https://orig.test/face.jpg"],
            resolution="480p",
            status=RunStatus.queued,
            preprocess_enabled=True,
            preprocessed_reference_urls=None,
        )
        db_session.add(run)
        db_session.commit()
        run_id = run.id

        fake_kie = FakeKieClient(synthetic_video)
        submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["reference_image_urls"] == ["https://orig.test/face.jpg"]

    def test_empty_preprocessed_list_with_prompts_still_uses_originals(
        self, db_session, synthetic_video, patch_propose
    ):
        """An unwritten result list is the originals fallback, NOT a hole.

        A GPT Image 2 phase that died before writing any result leaves prompts
        set and preprocessed_reference_urls empty. Submit has always fallen back
        to the originals for that shape, so the hole pre-flight must not refuse
        it — doing so would strand a run that submits fine today (and a
        per-segment rerun of a done run that completed the same way).
        """
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="PP Run",
            prompt="swap",
            reference_image_urls=[
                "https://orig.test/face.jpg", "https://orig.test/trim.jpg",
            ],
            resolution="480p",
            status=RunStatus.queued,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            preprocess_image_prompts=["", "pose like this"],
            preprocessed_reference_urls=[],
        )
        db_session.add(run)
        db_session.commit()
        run_id = run.id

        fake_kie = FakeKieClient(synthetic_video)
        submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["reference_image_urls"] == [
                "https://orig.test/face.jpg", "https://orig.test/trim.jpg",
            ]

    def test_hole_in_preprocessed_list_is_refused_before_any_upload(
        self, db_session, synthetic_video, patch_propose
    ):
        """A real hole fails fast, with a message that names the slot.

        Regression for run ce9b5bfd: this used to reach resolve_reference_urls,
        re-upload every surviving ref to kie.ai and then die on None.strip().
        """
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="PP Run",
            prompt="swap",
            reference_image_urls=[
                "https://orig.test/face.jpg", "https://orig.test/trim.jpg",
            ],
            resolution="480p",
            status=RunStatus.queued,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            preprocess_image_prompts=["", "pose like this"],
            preprocessed_reference_urls=["https://orig.test/face.jpg", None],
        )
        db_session.add(run)
        db_session.commit()
        run_id = run.id

        fake_kie = FakeKieClient(synthetic_video)
        with pytest.raises(ValueError, match="Reference 1"):
            submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        # Nothing was uploaded or billed on the way to the refusal.
        assert fake_kie.create_task_records == []

    @pytest.mark.parametrize("junk", ["T3", "/etc/passwd", "/proc/self/environ"])
    def test_junk_reference_fails_the_run_cleanly_before_any_upload(
        self, db_session, synthetic_video, patch_propose, junk
    ):
        """AICP-148: a legacy row holding a non-URL that is not a file under
        DATA_DIR (prod has two runs with ``T3``/``T4``) fails the run with a
        message naming the slot — nothing uploaded to kie.ai, nothing billed."""
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="Junk Ref Run",
            prompt="swap",
            reference_image_urls=["https://orig.test/face.jpg", junk],
            resolution="480p",
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()
        run_id = run.id

        fake_kie = FakeKieClient(synthetic_video)
        uploads: list = []
        fake_kie.upload_file = lambda *a, **k: uploads.append(a) or "https://x/y"
        with pytest.raises(ValueError, match="Reference 1"):
            submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert uploads == []
        assert fake_kie.create_task_records == []
        db_session.expire_all()
        failed = db_session.get(Run, run_id)
        assert failed.status == RunStatus.failed
        assert "Reference 1" in (failed.error_message or "")


# ---------------------------------------------------------------------------
# Tests: the Copy-run × submit_run seam.
#
# _build_copied_run (api_v2.py) decides WHAT a copy carries; submit_run
# (pipeline_v2.py:1673) decides which of those fields it reads. A unit test on
# either function alone can't catch a mismatch between them — this class
# drives both, end to end, the way a real copy-then-submit does.
# ---------------------------------------------------------------------------


class TestCopyRunSubmitSeam:
    def _ready_project(self, db_session, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        db_session.expire_all()
        return project_id

    def _preprocess_source_run(self, db_session, project_id) -> str:
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="PP Source",
            prompt="swap",
            reference_image_urls=[
                "https://orig.test/char.jpg",
                "https://orig.test/frame1.jpg",
                "https://orig.test/frame2.jpg",
            ],
            resolution="480p",
            status=RunStatus.done,
            preprocess_enabled=True,
            preprocess_mode="per_image",
            preprocess_image_prompts=["", "pose like this", "and this"],
            preprocessed_reference_urls=[
                "https://orig.test/char.jpg",
                "https://pp.test/gen1.png",
                "https://pp.test/gen2.png",
            ],
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def test_plain_copy_submits_generated_images_not_originals(
        self, db_session, synthetic_video, patch_propose
    ):
        """The bug: a plain copy of a GPT Image 2 run used to submit the raw
        originals despite carrying the generated images. Drive the copy
        through the real submit path and prove Seedance receives the
        generated images."""
        from app.api_v2 import _build_copied_run
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        src_id = self._preprocess_source_run(db_session, project_id)
        src = db_session.get(Run, src_id)

        new_run = _build_copied_run(
            src, db_session,
            resolution=None, name=None, ref_files=[], ref_urls=[], preprocess=False,
        )
        new_id = new_run.id
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        submit_run(new_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["reference_image_urls"] == [
                "https://orig.test/char.jpg",
                "https://pp.test/gen1.png",
                "https://pp.test/gen2.png",
            ]

    def test_copy_with_new_photo_submits_new_photo_raw(
        self, db_session, synthetic_video, patch_propose
    ):
        """A copy WITH a new photo carries no generated images at all, so
        submit_run falls through to the (new) originals — the new photo is
        never inert."""
        from app.api_v2 import _build_copied_run
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        src_id = self._preprocess_source_run(db_session, project_id)
        src = db_session.get(Run, src_id)

        new_run = _build_copied_run(
            src, db_session,
            resolution=None, name=None, ref_files=[],
            ref_urls=["https://new.example/face.jpg"], preprocess=False,
        )
        new_id = new_run.id
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        submit_run(new_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["reference_image_urls"] == ["https://new.example/face.jpg"]

    def test_copy_of_a_copy_still_submits_generated_images(
        self, db_session, synthetic_video, patch_propose
    ):
        """A copy of a copy (no new photo at either step) must keep resolving
        to the generated images — guards against the one-generation-later
        regression."""
        from app.api_v2 import _build_copied_run
        from app.pipeline_v2 import submit_run

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        src_id = self._preprocess_source_run(db_session, project_id)
        src = db_session.get(Run, src_id)

        first_copy = _build_copied_run(
            src, db_session,
            resolution=None, name=None, ref_files=[], ref_urls=[], preprocess=False,
        )
        db_session.commit()
        db_session.expire_all()
        first_copy = db_session.get(Run, first_copy.id)

        second_copy = _build_copied_run(
            first_copy, db_session,
            resolution=None, name=None, ref_files=[], ref_urls=[], preprocess=False,
        )
        second_id = second_copy.id
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        submit_run(second_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert len(fake_kie.create_task_records) == 2
        for rec in fake_kie.create_task_records:
            assert rec["reference_image_urls"] == [
                "https://orig.test/char.jpg",
                "https://pp.test/gen1.png",
                "https://pp.test/gen2.png",
            ]

    def test_preprocess_copy_of_a_plain_copy_creates_real_tasks(
        self, db_session, synthetic_video, patch_propose
    ):
        """Addendum regression: a plain copy (no new photo) must carry the
        FULL recipe forward — including preprocess_image_prompts, not just
        preprocessed_reference_urls and the flag. Without the prompts, a
        further `preprocess=true` copy of it would pass the "created with
        preprocessing" guard yet have nothing for GPT Image 2 to reuse:
        every trim falls through pass-through, zero kie.ai tasks are
        created, and the raw new photo would ship ungenerated."""
        from app.api_v2 import _build_copied_run
        from app.pipeline_v2 import preprocess_reference_images

        project_id = self._ready_project(db_session, synthetic_video, patch_propose)
        src_id = self._preprocess_source_run(db_session, project_id)
        src = db_session.get(Run, src_id)

        plain_copy = _build_copied_run(
            src, db_session,
            resolution=None, name=None, ref_files=[], ref_urls=[], preprocess=False,
        )
        db_session.commit()
        db_session.expire_all()
        plain_copy = db_session.get(Run, plain_copy.id)

        # Sanity: the plain copy carries the whole recipe, not just a subset.
        assert plain_copy.preprocess_enabled is True
        assert plain_copy.preprocess_image_prompts == [
            "", "pose like this", "and this"
        ]

        preprocess_copy = _build_copied_run(
            plain_copy, db_session,
            resolution=None, name=None, ref_files=[],
            ref_urls=["https://new.example/face2.jpg"], preprocess=True,
        )
        preprocess_copy_id = preprocess_copy.id
        assert preprocess_copy.status == RunStatus.preprocessing
        db_session.commit()

        fake_kie = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(preprocess_copy_id, kie=fake_kie)

        # The bug this guards against: prompts=None meant every trim was
        # pass-through and pending came back empty ({}) with zero tasks.
        assert len(fake_kie.create_image_records) == 2
        assert pending != {}
        for rec in fake_kie.create_image_records:
            assert rec["input_urls"][0] == "https://new.example/face2.jpg"


class TestSegmentSnapshot:
    """A run finalizes against the segments it LAUNCHED with, not whatever the
    project's segments have become since.

    Regression cover for run 76734005 (2026-08-11): an operator split the
    project's second segment while that run was still mid-flight, prepping the
    next run. The new SegmentDef had no RunSegment on the older run, so the
    completeness gate scored it as a swap segment that "did not complete" and
    parked a fully generated — and fully paid for — run at `incomplete`. Had the
    gate passed, the stitch would have KeyError'd on the same phantom def.
    """

    def _setup(self, db_session, db_engine, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)
        return project_id, run_id

    @staticmethod
    def _add_segment_def(db_engine, project_id, *, index, start_sec, end_sec):
        """Insert a new swap SegmentDef into the project, as the editor would."""
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            # Push existing defs at or after `index` down one slot, exactly as a
            # split does — this is what renumbers the original def out from under
            # an in-flight run.
            for sd in s.get(VideoProject, project_id).segments:
                if sd.index >= index:
                    sd.index += 1
            s.add(SegmentDef(
                project_id=project_id, index=index,
                start_sec=start_sec, end_sec=end_sec,
                has_face=False, action="swap",
            ))
            s.commit()

    def _run_to_completed_segments(self, run_id, synthetic_video):
        """submit_run + one poll round → every segment completed, results on disk."""
        from app.pipeline_v2 import submit_run, poll_run_once
        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())
        remaining = poll_run_once(run_id, pending=pending, kie=fake_kie)
        assert remaining == {}, "fake kie should complete every task in one round"

    def test_submit_run_freezes_segment_snapshot(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """submit_run records the project's defs on the run before cutting clips."""
        from app.pipeline_v2 import submit_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)
        submit_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            snap = s.get(Run, run_id).segment_snapshot
            live = sorted(
                s.get(VideoProject, project_id).segments, key=lambda sd: sd.index
            )
            assert snap, "submit_run must freeze the segment defs onto the run"
            assert [e["id"] for e in snap] == [sd.id for sd in live]
            assert [e["index"] for e in snap] == [sd.index for sd in live]
            assert [e["action"] for e in snap] == [sd.action for sd in live]
            assert [e["start_sec"] for e in snap] == [sd.start_sec for sd in live]
            assert [e["end_sec"] for e in snap] == [sd.end_sec for sd in live]

    def test_segment_added_mid_run_does_not_strand_run(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """THE regression: a def added mid-run must not make finalize cry wolf."""
        from app.pipeline_v2 import finalize_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)
        self._run_to_completed_segments(run_id, synthetic_video)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            launched_ids = {e["id"] for e in s.get(Run, run_id).segment_snapshot}

        # Operator splits the project's segments while the run is in flight.
        self._add_segment_def(
            db_engine, project_id, index=1, start_sec=0.5, end_sec=0.8,
        )

        finalize_run(run_id, gdrive=FakeGDriveClient())

        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done, (
                f"run stranded at {run.status}: {run.error_message}"
            )
            assert run.result_local_path and os.path.exists(run.result_local_path)
            # The phantom def must not have leaked into the run.
            live_ids = {sd.id for sd in s.get(VideoProject, project_id).segments}
            assert live_ids - launched_ids, "test did not actually add a new def"
            assert {e["id"] for e in run.segment_snapshot} == launched_ids

    def test_segment_added_mid_run_strands_run_without_snapshot(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """Fallback path: a run with no snapshot keeps the old (buggy) behaviour.

        Pins the compatibility contract for runs that predate the column —
        finalize still reads the live project segments for those, so this asserts
        the OLD outcome on purpose. It is what makes the fix above meaningful.
        """
        from app.pipeline_v2 import finalize_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)
        self._run_to_completed_segments(run_id, synthetic_video)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            s.get(Run, run_id).segment_snapshot = None   # simulate a legacy run
            s.commit()

        self._add_segment_def(
            db_engine, project_id, index=1, start_sec=0.5, end_sec=0.8,
        )
        finalize_run(run_id, gdrive=FakeGDriveClient())

        with Session() as s:
            assert s.get(Run, run_id).status == RunStatus.incomplete

    def test_run_segment_defs_prefers_snapshot_over_live_edits(self):
        """Boundary rewrites after launch must not move the run's cut points."""
        from app.pipeline_v2 import run_segment_defs, build_segment_snapshot
        from types import SimpleNamespace

        launched = [
            SimpleNamespace(id="a", index=0, action="swap", start_sec=0.0, end_sec=9.0),
            SimpleNamespace(id="b", index=1, action="keep", start_sec=9.0, end_sec=38.0),
        ]
        run = SimpleNamespace(segment_snapshot=build_segment_snapshot(launched))
        # The project has since been re-cut: 'b' now starts at 24.0 and there is
        # a new def in between.
        project = SimpleNamespace(segments=[
            SimpleNamespace(id="a", index=0, action="swap", start_sec=0.0, end_sec=9.0),
            SimpleNamespace(id="c", index=1, action="swap", start_sec=9.0, end_sec=24.0),
            SimpleNamespace(id="b", index=2, action="keep", start_sec=24.0, end_sec=38.0),
        ])

        defs = run_segment_defs(run, project)
        assert [sd.id for sd in defs] == ["a", "b"]
        assert [sd.start_sec for sd in defs] == [0.0, 9.0]
        assert [sd.action for sd in defs] == ["swap", "keep"]

        # No snapshot → fall back to the project's live segments.
        assert [sd.id for sd in run_segment_defs(
            SimpleNamespace(segment_snapshot=None), project
        )] == ["a", "c", "b"]

    def test_resubmit_keeps_the_original_snapshot(
        self, db_engine, db_session, synthetic_video, patch_propose
    ):
        """A second submit_run must not re-freeze against edited defs.

        The clips already uploaded were cut to the ORIGINAL boundaries, so
        re-snapshotting would describe a run that was never submitted.
        """
        from app.pipeline_v2 import submit_run

        project_id, run_id = self._setup(db_session, db_engine, synthetic_video, patch_propose)
        submit_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=FakeGDriveClient())

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            first = list(s.get(Run, run_id).segment_snapshot)

        self._add_segment_def(
            db_engine, project_id, index=1, start_sec=0.5, end_sec=0.8,
        )
        submit_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=FakeGDriveClient())

        with Session() as s:
            assert s.get(Run, run_id).segment_snapshot == first


# ---------------------------------------------------------------------------
# Tests: the Seedance 2.5 voice reference (`reference_audio_urls`)
# ---------------------------------------------------------------------------


class TestVoiceReference:
    """A run's voice sample reaches createTask — for the models that take one.

    Same absent-vs-empty contract as `generate_audio`: every other backend
    rejects an unknown input key, so a run without a voice must produce a
    payload byte-identical to the pre-feature one.

    The interesting rule is the interaction with `generate_audio`: when a run
    stitches with audio_mode="original" every generated track is discarded at
    stitch time, so the reference could not reach the delivered video even in
    principle and is deliberately NOT sent.
    """

    _mock_media = staticmethod(TestGeminiOmniAudioAndLimit._mock_media)
    _one_swap = staticmethod(TestModelClipCeiling._one_swap)

    @staticmethod
    def _make_run(
        db_session, project_id, *, model, audio_mode="seedance",
        stitch_plan=None, voices=None,
    ):
        run = Run(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name="t",
            prompt="Replace the character",
            reference_image_urls=[],
            reference_audio_urls=voices,
            model=model,
            resolution="720p" if model == "gemini-omni" else "480p",
            audio_mode=audio_mode,
            stitch_plan=stitch_plan,
            status=RunStatus.queued,
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def _run(
        self, db_session, tmp_path, monkeypatch, *, model,
        audio_mode="seedance", stitch_plan=None, voices=None, n_swaps=1,
    ):
        from app.pipeline_v2 import process_run

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        project_id = _create_project(
            db_session, str(source), status=ProjectStatus.ready
        )
        if n_swaps == 1:
            self._one_swap(db_session, project_id, start=0.0, end=5.0)
        else:
            db_session.add_all([
                SegmentDef(project_id=project_id, index=i, start_sec=i * 5.0,
                           end_sec=i * 5.0 + 5.0, has_face=True, action="swap")
                for i in range(n_swaps)
            ])
            db_session.commit()

        # A real file on disk: the voice is uploaded like any other reference.
        voice_paths = None
        if voices:
            vdir = tmp_path / "voice"
            vdir.mkdir(exist_ok=True)
            voice_paths = []
            for i, _ in enumerate(voices):
                vp = vdir / f"voice{i}.mp3"
                vp.write_bytes(b"ID3voice")
                voice_paths.append(str(vp))

        run_id = self._make_run(
            db_session, project_id, model=model, audio_mode=audio_mode,
            stitch_plan=stitch_plan, voices=voice_paths,
        )
        self._mock_media(monkeypatch, duration_sec=10.0,
                         cut_calls=[], stitch_calls=[])
        fake_kie = FakeKieClient(str(source))
        process_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())
        return fake_kie

    def test_voice_is_sent_for_seedance_2_5(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode="seedance", voices=["v"],
        )
        rec = fake_kie.create_task_records[0]
        assert len(rec["reference_audio_urls"]) == 1
        # The local path was uploaded and the resulting URL is what is sent.
        assert rec["reference_audio_urls"][0].startswith("https://fake-kie")

    def test_key_is_absent_without_a_voice(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """No voice → the kwarg is never passed, so 2.0 payloads stay unchanged."""
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode="seedance", voices=None,
        )
        rec = fake_kie.create_task_records[0]
        assert rec["reference_audio_urls"] is NOT_PASSED

    @pytest.mark.parametrize("model", ["seedance", "seedance-fast", "seedance-mini"])
    def test_never_sent_to_a_model_without_the_field(
        self, db_engine, db_session, tmp_path, monkeypatch, model
    ):
        """Belt-and-braces: even a run carrying a voice must not send it to 2.0.

        The API rejects a voice on these models at run creation, but a run can
        also reach this state by having its model changed after the fact — and
        the request would then fail wholesale, losing every segment.
        """
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model=model, audio_mode="seedance", voices=["v"],
        )
        rec = fake_kie.create_task_records[0]
        assert rec["reference_audio_urls"] is NOT_PASSED

    def test_not_sent_when_the_generated_track_is_discarded(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """A stitching run on audio_mode="original" throws every track away.

        generate_audio is False there, so the voice cannot reach the delivered
        video; sending it would ask the API to weigh an input that cannot matter.
        """
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode="original", voices=["v"],
        )
        rec = fake_kie.create_task_records[0]
        assert rec["generate_audio"] is False
        assert rec["reference_audio_urls"] is NOT_PASSED

    def test_sent_on_a_no_stitch_run_despite_audio_mode_original(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The mirror of the generate_audio no-stitch rule.

        stitch_plan=[] makes the per-segment clips the deliverable, so their
        audio is real output regardless of audio_mode — and the voice applies.
        """
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode="original", stitch_plan=[],
            voices=["v"],
        )
        rec = fake_kie.create_task_records[0]
        assert rec["generate_audio"] is True
        assert len(rec["reference_audio_urls"]) == 1

    def test_uploaded_once_and_shared_by_every_segment(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """The whole point of a run-level voice: every segment gets the SAME url.

        Uploading per segment would both waste uploads and — since kie.ai hands
        back a distinct URL each time — give each segment a nominally different
        reference for one file.
        """
        fake_kie = self._run(
            db_session, tmp_path, monkeypatch,
            model="seedance-2-5", audio_mode="seedance", voices=["v"], n_swaps=3,
        )
        assert len(fake_kie.create_task_records) == 3
        sent = {
            tuple(r["reference_audio_urls"]) for r in fake_kie.create_task_records
        }
        assert len(sent) == 1

        voice_uploads = [
            c for c in fake_kie.upload_calls if c.endswith("voice0.mp3")
        ]
        assert len(voice_uploads) == 1


# ---------------------------------------------------------------------------
# Truncated-result guard (_result_shortfall_sec)
#
# Seedance 2.5 intermittently returns a result truncated to exactly 14.72s
# instead of the clip's length, while reporting the task as a success. Nothing
# used to check, so the stitch spliced it in and the run delivered a timeline
# missing that footage — run 4e631e56 went out at 25.1s instead of 38s.
#
# It is NOT a ceiling: 7 of 11 generations from a >15s clip came back at full
# length (up to 29.04s from a 29.30s clip), and the identical 27.5s segment
# returned 27.392s on run fd3948b2 and 14.720s on 4e631e56. So it cannot be
# rejected at submit time by max_clip_sec — it has to be measured on return and
# RETRIED, which is what TestTruncatedResultIsRetried below pins down.
# ---------------------------------------------------------------------------


def _video_of(path, seconds, fps=24):
    """Write a tiny mp4 of exactly *seconds* at *fps*."""
    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s=160x120:d={seconds}:r={fps}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "40",
            str(path),
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return str(path)


class TestResultShortfall:

    def test_matching_lengths_pass(self, tmp_path):
        from app.pipeline_v2 import _result_too_short

        clip = _video_of(tmp_path / "clip.mp4", 10)
        result = _video_of(tmp_path / "res.mp4", 10)
        assert _result_too_short(clip, result) is None

    def test_small_honest_drift_passes(self, tmp_path):
        """A generated clip is never exactly the input length.

        The worst honest shortfall measured in production is 0.27s (a 10.0s
        clip returning 9.728s). If this guard tripped on that it would reject
        every healthy generation, so the tolerance must clear it.
        """
        from app.pipeline_v2 import _result_too_short

        clip = _video_of(tmp_path / "clip.mp4", 10)
        result = _video_of(tmp_path / "res.mp4", 9.75)
        assert _result_too_short(clip, result) is None

    def test_truncated_result_is_reported(self, tmp_path):
        """The real failure: 27.5s sent, ~14.7s back."""
        from app.pipeline_v2 import _result_too_short

        clip = _video_of(tmp_path / "clip.mp4", 27.5)
        result = _video_of(tmp_path / "res.mp4", 14.7)
        too_short = _result_too_short(clip, result)
        assert too_short is not None
        # Both durations are carried, not just the delta: the operator-facing
        # message names what was sent and what came back.
        assert too_short.clip_sec == pytest.approx(27.5, abs=0.2)
        assert too_short.result_sec == pytest.approx(14.7, abs=0.2)
        assert too_short.shortfall_sec == pytest.approx(12.8, abs=0.2)

    def test_longer_result_passes(self, tmp_path):
        """Only a SHORT result desyncs the timeline; never flag a long one."""
        from app.pipeline_v2 import _result_too_short

        clip = _video_of(tmp_path / "clip.mp4", 10)
        result = _video_of(tmp_path / "res.mp4", 11)
        assert _result_too_short(clip, result) is None

    def test_unprobeable_input_accepts_rather_than_fails(self, tmp_path):
        """An unknown duration is not evidence of truncation.

        This guard was bolted onto a path that previously accepted everything.
        Failing the segment because ffprobe choked would convert a fine result
        into a lost swap, so the guard stays silent when it cannot measure.
        """
        from app.pipeline_v2 import _result_too_short

        clip = _video_of(tmp_path / "clip.mp4", 10)
        missing = str(tmp_path / "nope.mp4")
        assert _result_too_short(clip, missing) is None
        assert _result_too_short(missing, clip) is None
        # No clip checkpoint recorded at all (resumed run) — same reasoning.
        assert _result_too_short(None, clip) is None


class TestShortResultIsStitchedButFlagged:
    """Whatever the model returns gets stitched — short or long — and is flagged.

    The operator's call: refusing a short result delivers nothing at all while
    Seedance 2.5 is in one of its truncating spells, and retrying just re-bills
    for the same clip (three resubmits of run 4e631e56 with the length spelled
    out in the prompt all returned the same 14.72s).

    The cost is real, which is why the flag is not optional: a short segment
    shortens the delivered video against the source timeline, and that is
    exactly how run 4e631e56 went out at 25.1s of a 38s video with nobody
    noticing.
    """

    def _pending(self, attempt=1, recipe=True):
        return {
            "task-trunc": {
                "rs_id": "rs-1",
                "index": 0,
                "attempt": attempt,
                "recipe": {"foo": "bar"} if recipe else None,
                "deadline": _time.time() + 600,
            }
        }

    def _wire(self, monkeypatch, tmp_path, clip_sec, result_sec):
        """Stub the DB + kie so only the length logic is under test."""
        import app.pipeline_v2 as pv2

        clip = _video_of(tmp_path / "clip.mp4", clip_sec)
        result_src = _video_of(tmp_path / "src_result.mp4", result_sec)

        class _Rs:
            def __init__(self):
                self.status = SegmentStatus.generating
                self.seedance_task_id = "task-trunc"
                self.seedance_result_url = None
                self.local_result_path = None
                self.local_clip_path = clip
                self.cost_credits = None
                self.error_message = None
                self.source_fallback = False

        rs = _Rs()

        class _Session:
            def get(self, _m, _i): return rs
            def commit(self): pass
            def rollback(self): pass
            def close(self): pass

        @contextmanager
        def _get_session():
            yield _Session()

        class _Kie:
            def get_task(self, _t):
                return {
                    "state": "success",
                    "creditsConsumed": 100,
                    "resultJson": json.dumps({"resultUrls": ["https://x/r.mp4"]}),
                }

            def download_result(self, _url, dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(result_src, dst)

        monkeypatch.setattr(pv2, "get_session", _get_session)
        monkeypatch.setattr(
            pv2, "transition", lambda o, s: setattr(o, "status", s)
        )
        monkeypatch.setattr(
            pv2, "_create_swap_task",
            lambda **kw: pytest.fail("a short result must NOT be resubmitted"),
        )
        return pv2, rs, _Kie()

    def test_short_result_is_completed_and_stitched(self, monkeypatch, tmp_path):
        """27.5s in, 14.7s back → completed, with the result wired to the stitch."""
        pv2, rs, kie = self._wire(monkeypatch, tmp_path, 27.5, 14.7)
        results_dir = str(tmp_path / "res")

        pending = pv2.poll_run_once_internal(
            pending=self._pending(), kie=kie, results_dir=results_dir,
        )

        assert pending == {}, "nothing may be left in flight"
        assert rs.status == SegmentStatus.completed
        # local_result_path is what the stitch reads — it must be set.
        assert rs.local_result_path == os.path.join(results_dir, "result_0000.mp4")
        assert os.path.exists(rs.local_result_path)
        assert rs.source_fallback is False
        assert rs.cost_credits == 100

    def test_short_result_carries_a_visible_warning(self, monkeypatch, tmp_path):
        """Completed + a message = the run page's "short result" flag.

        Without this the shortened timeline is invisible, which is the whole
        defect being guarded against.
        """
        pv2, rs, kie = self._wire(monkeypatch, tmp_path, 27.5, 14.7)

        pv2.poll_run_once_internal(
            pending=self._pending(), kie=kie, results_dir=str(tmp_path / "res"),
        )

        assert rs.error_message, "a short result must not complete silently"
        assert "14.7" in rs.error_message and "27.5" in rs.error_message
        # The run UI truncates error_message at 80 chars.
        assert len(rs.error_message) <= 80

    def test_full_length_result_completes_with_no_warning(self, monkeypatch, tmp_path):
        """The normal path must stay clean, or the flag means nothing."""
        pv2, rs, kie = self._wire(monkeypatch, tmp_path, 10.0, 9.9)

        pending = pv2.poll_run_once_internal(
            pending=self._pending(), kie=kie, results_dir=str(tmp_path / "res"),
        )

        assert pending == {}
        assert rs.status == SegmentStatus.completed
        assert rs.error_message is None
        assert rs.local_result_path is not None

    def test_a_longer_result_is_never_flagged(self, monkeypatch, tmp_path):
        """Only SHORT results shorten the timeline; a long one is not a defect."""
        pv2, rs, kie = self._wire(monkeypatch, tmp_path, 10.0, 11.0)

        pv2.poll_run_once_internal(
            pending=self._pending(), kie=kie, results_dir=str(tmp_path / "res"),
        )

        assert rs.status == SegmentStatus.completed
        assert rs.error_message is None


# ---------------------------------------------------------------------------
# Daily kie.ai budget (AICP-155): an admitted run is never refused mid-flight
# ---------------------------------------------------------------------------


@pytest.mark.spend_cap
class TestDailyBudgetInFlight:
    def test_over_budget_day_does_not_stop_an_admitted_run_and_its_spend_is_measured(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch,
    ):
        """The day is over budget (someone else spent it all after this run was
        launched). The run's worker still submits every segment — refusing it
        here is what strands runs — and each created task is recorded, beyond
        the run's (here: zero) reservation, as overdraft the next admission
        sees."""
        from sqlalchemy import select

        from app import spend
        from app.config import settings
        from app.models import KieSpendDraw
        from app.pipeline_v2 import analyze_project, submit_run

        monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        # The module's DB is shared with earlier tests, so size the cap to
        # whatever today already holds, then spend the last 200 credits.
        Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
        s = Session()
        charged = spend.today_summary(s)["charged_credits"]
        monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", (charged + 200) * 0.005)
        spend.reserve(s, "run:someone-else", 200, kind="new run")
        s.commit()
        assert spend.today_summary(s)["over"]
        before = spend.today_summary(s)["charged_credits"]
        s.close()

        fake_kie = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fake_kie, gdrive=FakeGDriveClient())

        assert pending is not None and len(pending) == 2
        assert len(fake_kie.create_task_calls) == 2
        s = Session()
        draws = s.execute(select(KieSpendDraw)).scalars().all()
        assert {d.call_key for d in draws} >= set(pending)
        after = spend.today_summary(s)["charged_credits"]
        s.close()
        assert after > before
        # And a NEW action is still refused.
        s = Session()
        with pytest.raises(spend.BudgetExceeded):
            spend.reserve(s, "run:new", 1, kind="new run")
        s.rollback()
        s.close()


# ---------------------------------------------------------------------------
# AICP-342: a re-delivery replaces the content behind the existing Drive id
# ---------------------------------------------------------------------------


class TestInPlaceRedelivery:
    def _delivered_run(self, db_session, db_engine, synthetic_video, gdrive):
        """A run delivered once through the real pipeline (folder set)."""
        from app.pipeline_v2 import analyze_project, process_run

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id, gdrive_folder_id="folder-1")
        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gdrive)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            file_id = run.result_gdrive_file_id
            rs_ids = [rs.id for rs in sorted(run.run_segments, key=lambda r: r.index)]
        assert file_id and len(gdrive.upload_calls) == 1
        return project_id, run_id, file_id, rs_ids

    def _rerun_first(self, db_engine, run_id, rs_ids):
        """What the rerun endpoint does to the rows (minus the HTTP layer)."""
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            rs = s.get(RunSegment, rs_ids[0])
            rs.status = SegmentStatus.pending
            rs.seedance_task_id = None
            rs.seedance_result_url = None
            rs.local_result_path = None
            run = s.get(Run, run_id)
            run.result_local_path = None
            run.result_duration_sec = None
            transition(run, RunStatus.queued)
            s.commit()

    def test_rerun_redelivers_to_the_same_drive_id(
        self, db_engine, db_session, synthetic_video, patch_propose,
    ):
        from app.pipeline_v2 import process_run

        gdrive = FakeGDriveClient()
        _, run_id, file_id, rs_ids = self._delivered_run(
            db_session, db_engine, synthetic_video, gdrive,
        )
        self._rerun_first(db_engine, run_id, rs_ids)

        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gdrive)

        assert len(gdrive.upload_calls) == 1  # no second upload
        assert [c["file_id"] for c in gdrive.update_calls] == [file_id]
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.done
            assert run.result_gdrive_file_id == file_id
            # Uploaded from this finalize's private pin of final.mp4 (removed
            # afterwards), never from the shared name.
            pinned = gdrive.update_calls[0]["local_path"]
            assert os.path.basename(pinned).startswith(".deliver-")
            assert os.path.dirname(pinned) == os.path.dirname(run.result_local_path)
            assert not os.path.exists(pinned)
            assert run.result_duration_sec is not None

    def test_failed_in_place_update_fails_the_run_and_never_uploads_a_new_file(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch,
    ):
        from app import pipeline_v2 as pv2
        from app.pipeline_v2 import process_run

        monkeypatch.setattr(pv2, "RUN_DELIVER_BACKOFF_SEC", 0.0)
        gdrive = FakeGDriveClient()
        _, run_id, file_id, rs_ids = self._delivered_run(
            db_session, db_engine, synthetic_video, gdrive,
        )
        self._rerun_first(db_engine, run_id, rs_ids)
        gdrive.update_error = RuntimeError("HttpError 404: File not found")

        with pytest.raises(RuntimeError):
            process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gdrive)

        assert len(gdrive.upload_calls) == 1
        assert len(gdrive.update_calls) == pv2.RUN_DELIVER_ATTEMPTS
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.failed
            assert run.result_gdrive_file_id == file_id
            # The final stays on disk, so Retry re-delivers without re-stitching.
            assert run.result_local_path and os.path.exists(run.result_local_path)

    def test_a_finalize_that_lost_the_lock_never_touches_drive(
        self, db_engine, db_session, synthetic_video, patch_propose,
    ):
        from app.pipeline_v2 import finalize_run

        gdrive = FakeGDriveClient()
        _, run_id, file_id, rs_ids = self._delivered_run(
            db_session, db_engine, synthetic_video, gdrive,
        )
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            run.result_local_path = None
            transition(run, RunStatus.queued)
            transition(run, RunStatus.processing)
            s.commit()

        assert finalize_run(run_id, gdrive=gdrive, owner_check=lambda: False) is None
        assert gdrive.update_calls == [] and len(gdrive.upload_calls) == 1
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status != RunStatus.failed
            assert run.result_gdrive_file_id == file_id

    def test_custom_plan_mux_cache_follows_the_restored_clip(
        self, db_engine, db_session, synthetic_video, patch_propose, tmp_path,
    ):
        """A restored take is an OLDER file at another path: the original-audio
        mux cache must not hand back the mux of the clip stitched last."""
        from app.pipeline_v2 import analyze_project, finalize_run, process_run
        from app.storage import run_results_dir

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, project_id)
            plan = [{"kind": "segment", "segment_def_id": sd.id}
                    for sd in sorted(project.segments, key=lambda d: d.index)]
        run_id = _create_run(db_session, project_id, stitch_plan=plan)
        process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=FakeGDriveClient())
        r_dir = run_results_dir(run_id, project_id)
        before = {f for f in os.listdir(r_dir) if f.startswith("resultaud_")}
        assert before

        # "Restore" segment 0 to an older take file at another path.
        take = str(tmp_path / "take.mp4")
        shutil.copy2(synthetic_video, take)
        os.utime(take, (1_000_000, 1_000_000))
        with Session() as s:
            run = s.get(Run, run_id)
            rs0 = sorted(run.run_segments, key=lambda r: r.index)[0]
            rs0.local_result_path = take
            run.result_local_path = None
            transition(run, RunStatus.queued)
            transition(run, RunStatus.processing)
            s.commit()
        finalize_run(run_id, gdrive=FakeGDriveClient())

        after = {f for f in os.listdir(r_dir) if f.startswith("resultaud_")}
        assert len(after - before) == 1  # the take got its own muxed clip

    def test_a_final_replaced_by_another_stitch_is_never_uploaded(
        self, db_engine, db_session, synthetic_video, patch_propose, monkeypatch,
    ):
        """The upload reads this finalize's pinned bytes, and only while
        final.mp4 is still that same file."""
        from app import pipeline_v2 as pv2
        from app.pipeline_v2 import process_run

        gdrive = FakeGDriveClient()
        _, run_id, file_id, rs_ids = self._delivered_run(
            db_session, db_engine, synthetic_video, gdrive,
        )
        self._rerun_first(db_engine, run_id, rs_ids)

        real = pv2._deliver_in_place

        def racing(session, run, *, final_dst, **kw):
            # Another stitch replaces final.mp4 after ours was pinned.
            tmp = final_dst + ".other.mp4"
            shutil.copy2(final_dst, tmp)
            os.replace(tmp, final_dst)
            return real(session, run, final_dst=final_dst, **kw)

        monkeypatch.setattr(pv2, "_deliver_in_place", racing)
        with pytest.raises(RuntimeError, match="replaced by another stitch"):
            process_run(run_id, kie=FakeKieClient(synthetic_video), gdrive=gdrive)
        assert gdrive.update_calls == []
        r_dir = os.path.dirname(gdrive.upload_calls[0]["local_path"])
        assert not [f for f in os.listdir(r_dir) if f.startswith(".deliver-")]
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            run = s.get(Run, run_id)
            assert run.status == RunStatus.failed
            assert run.result_gdrive_file_id == file_id


# ---------------------------------------------------------------------------
# AICP-351 §5.1: a restart test per paid submit intent site — a worker that
# dies after the provider accepted the POST and before the receipt committed
# must leave a PARKED attempt, never a second submit on recovery.
# ---------------------------------------------------------------------------


class _Crash(BaseException):
    """Simulates the worker dying right after the POST was accepted (nothing
    after it runs: not an Exception, so no handler records an outcome)."""


class _AcceptThenDie(FakeKieClient):
    """createTask is accepted by "kie.ai" (recorded), then the worker dies."""

    def __init__(self, src, *, die_on=None):
        super().__init__(src)
        self.die_on = die_on   # None = every create; else the n-th create (1-based)
        self.n = 0

    def _maybe_die(self):
        self.n += 1
        if self.die_on is None or self.n == self.die_on:
            raise _Crash()

    def create_task(self, **kw):
        tid = super().create_task(**kw)
        self._maybe_die()
        return tid

    def create_image_task(self, **kw):
        tid = super().create_image_task(**kw)
        self._maybe_die()
        return tid


@pytest.fixture()
def no_evidence(monkeypatch):
    from app import paid_attempts

    monkeypatch.setattr(paid_attempts, "default_evidence", lambda: (lambda row: "boot changed"))


def _attempt_rows(db_engine, **where):
    from sqlalchemy.orm import sessionmaker as _sm

    from app.paid_attempts_models import PaidSubmitAttempt

    with _sm(bind=db_engine)() as s:
        q = s.query(PaidSubmitAttempt)
        for k, v in where.items():
            q = q.filter(getattr(PaidSubmitAttempt, k) == v)
        return q.order_by(PaidSubmitAttempt.started_at).all()


class TestPaidSubmitAttemptsRestart:
    def _setup(self, db_session, synthetic_video, patch_propose):
        from app.pipeline_v2 import analyze_project

        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        return project_id, _create_run(db_session, project_id)

    def test_initial_submit_crash_parks_and_recovery_never_resubmits(
        self, db_engine, db_session, synthetic_video, patch_propose, no_evidence
    ):
        from app.pipeline_v2 import finalize_run, submit_run

        project_id, run_id = self._setup(db_session, synthetic_video, patch_propose)
        dying = _AcceptThenDie(synthetic_video)
        with pytest.raises(_Crash):
            submit_run(run_id, kie=dying, gdrive=FakeGDriveClient())
        assert len(dying.create_task_calls) >= 1          # kie.ai accepted (and billed)
        rows = _attempt_rows(db_engine, run_id=run_id)
        assert rows and all(r.status == "started" and r.provider_task_id is None for r in rows)

        # Recovery re-enqueues the run (it is still `processing`).
        fresh = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fresh, gdrive=FakeGDriveClient())
        parked = {r.owner_id for r in _attempt_rows(db_engine, run_id=run_id, status="parked")}
        assert parked == {r.owner_id for r in rows}
        assert all(r.sender_released_at is not None for r in _attempt_rows(db_engine, run_id=run_id))
        assert not ({m["rs_id"] for m in (pending or {}).values()} & parked)
        # The parked segments were NOT sent again.
        sent_again = [r for r in fresh.create_task_records]
        assert len(sent_again) == 2 - len(parked)

        # Other segments keep going; the run does not finalize.
        from app.pipeline_v2 import poll_run_once

        while pending:
            pending = poll_run_once(run_id, pending=pending, kie=fresh)
        finalize_run(run_id, gdrive=FakeGDriveClient())
        db_session.expire_all()
        assert db_session.get(Run, run_id).status == RunStatus.incomplete

    def test_in_poll_resubmit_crash_parks_with_the_predecessor(
        self, db_engine, db_session, synthetic_video, patch_propose, no_evidence
    ):
        from app.pipeline_v2 import poll_run_once, submit_run

        project_id, run_id = self._setup(db_session, synthetic_video, patch_propose)
        first = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=first, gdrive=FakeGDriveClient())
        victim_tid = next(iter(pending))
        victim_rs = pending[victim_tid]["rs_id"]

        class FailThenDie(_AcceptThenDie):
            def get_task(self, task_id, **kw):
                if task_id == victim_tid:
                    return {"state": "fail", "failMsg": "Internal Error, Please try again later."}
                return {"state": "generating"}

        dying = FailThenDie(synthetic_video)
        with pytest.raises(_Crash):
            poll_run_once(run_id, pending=dict(pending), kie=dying)
        (row,) = _attempt_rows(db_engine, owner_id=victim_rs, status="started")
        assert row.predecessor_task_id == victim_tid and row.provider_task_id is None
        db_session.expire_all()
        assert db_session.get(RunSegment, victim_rs).seedance_task_id == victim_tid

        # Recovery: the predecessor even reports success (a truncated one) —
        # the segment must NOT be recovered from it, nor resubmitted.
        class PredecessorSucceeded(FakeKieClient):
            def get_task(self, task_id, **kw):
                return super().get_task(task_id)

        fresh = PredecessorSucceeded(synthetic_video)
        pending2 = submit_run(run_id, kie=fresh, gdrive=FakeGDriveClient())
        (parked,) = _attempt_rows(db_engine, owner_id=victim_rs)[-1:]
        assert parked.status == "parked" and parked.predecessor_task_id == victim_tid
        assert victim_rs not in {m["rs_id"] for m in (pending2 or {}).values()}
        assert fresh.create_task_records == []
        db_session.expire_all()
        rs = db_session.get(RunSegment, victim_rs)
        assert rs.status != SegmentStatus.completed and "parked" in (rs.error_message or "")

    def test_unknown_create_outcome_parks_instead_of_resending(
        self, db_engine, db_session, synthetic_video, patch_propose, no_evidence
    ):
        from app.kie_client import KieTaskUnknown
        from app.pipeline_v2 import submit_run

        project_id, run_id = self._setup(db_session, synthetic_video, patch_propose)

        class Unknown(FakeKieClient):
            def create_task(self, **kw):
                super().create_task(**kw)
                raise KieTaskUnknown("createTask HTTP 502")

        k = Unknown(synthetic_video)
        # Every segment parks; the run is NOT failed (it finalizes incomplete).
        assert submit_run(run_id, kie=k, gdrive=FakeGDriveClient()) == {}
        db_session.expire_all()
        assert db_session.get(Run, run_id).status == RunStatus.processing
        rows = _attempt_rows(db_engine, run_id=run_id)
        assert rows and {r.status for r in rows} == {"parked"}
        assert len(k.create_task_records) == len(rows)      # one POST each, no client retry

    def test_preprocess_per_image_crash_mid_loop(
        self, db_engine, db_session, synthetic_video, no_evidence
    ):
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        run = Run(id=str(uuid.uuid4()), project_id=project_id, prompt="swap",
                  reference_image_urls=["https://x.test/a.jpg", "https://x.test/b.jpg",
                                        "https://x.test/c.jpg"],
                  resolution="480p", status=RunStatus.preprocessing, preprocess_enabled=True,
                  preprocess_image_prompts=["", "p1", "p2"])
        db_session.add(run)
        db_session.commit()
        dying = _AcceptThenDie(synthetic_video, die_on=2)
        with pytest.raises(_Crash):
            preprocess_reference_images(run.id, kie=dying)
        db_session.expire_all()
        ids = db_session.get(Run, run.id).preprocess_task_ids
        assert ids[1] == dying.create_image_records[0]["task_id"]   # committed per image
        assert ids[2] is None

        fresh = FakeKieClient(synthetic_video)
        pending = preprocess_reference_images(run.id, kie=fresh)
        assert pending == {ids[1]: 1}                     # slot 1 re-polled, slot 2 parked
        assert fresh.create_image_records == []
        (parked,) = _attempt_rows(db_engine, owner_id=f"{run.id}:2")
        assert parked.status == "parked"

    def test_preprocess_all_as_refs_crash(
        self, db_engine, db_session, synthetic_video, no_evidence
    ):
        from app.pipeline_v2 import preprocess_reference_images

        project_id = _create_project(db_session, synthetic_video)
        run = Run(id=str(uuid.uuid4()), project_id=project_id, prompt="swap",
                  reference_image_urls=["https://x.test/a.jpg", "https://x.test/b.jpg"],
                  resolution="480p", status=RunStatus.preprocessing, preprocess_enabled=True,
                  preprocess_mode="all_as_refs", preprocess_image_prompts=["one prompt"])
        db_session.add(run)
        db_session.commit()
        with pytest.raises(_Crash):
            preprocess_reference_images(run.id, kie=_AcceptThenDie(synthetic_video))
        fresh = FakeKieClient(synthetic_video)
        assert preprocess_reference_images(run.id, kie=fresh) == {}
        assert fresh.create_image_records == []
        (parked,) = _attempt_rows(db_engine, owner_id=f"{run.id}:0")
        assert parked.status == "parked"


class TestParkedSegmentDoesNotStopItsSiblings:
    """AICP-351 §5.1 (independent review): one segment's createTask outcome is
    unknown -> that segment parks, the other segments are submitted, polled and
    completed, the run ends `incomplete` (not `failed`), the parked segment is
    never resubmitted by the pipeline, and Resubmit from the run page card
    completes it."""

    _SEGS = [
        _PS(start_sec=0.0, end_sec=6.0, has_face=True, action="swap"),
        _PS(start_sec=6.0, end_sec=8.0, has_face=False, action="keep"),
        _PS(start_sec=8.0, end_sec=14.0, has_face=True, action="swap"),
        _PS(start_sec=14.0, end_sec=16.0, has_face=False, action="keep"),
        _PS(start_sec=16.0, end_sec=25.0, has_face=True, action="swap"),
    ]

    def test_middle_segment_parks_siblings_complete_then_resubmit(
        self, db_engine, db_session, synthetic_video, monkeypatch, no_evidence
    ):
        import app.api_v2 as api_v2
        import app.face as face_mod
        import app.pipeline_v2 as pv2
        from app import paid_attempts
        from app.api_paid_attempts import relaunch
        from app.kie_client import KieTaskUnknown
        from app.pipeline_v2 import analyze_project, finalize_run, poll_run_once, submit_run

        monkeypatch.setattr(face_mod, "propose_segments", lambda *a, **kw: self._SEGS)
        monkeypatch.setattr(pv2.face_mod, "propose_segments", lambda *a, **kw: self._SEGS)
        project_id = _create_project(db_session, synthetic_video)
        analyze_project(project_id)
        run_id = _create_run(db_session, project_id)

        class MiddleUnknown(FakeKieClient):
            def create_task(self, **kw):
                tid = super().create_task(**kw)
                if any("clip_0002" in u for u in kw["reference_video_urls"]):
                    raise KieTaskUnknown("createTask HTTP 502")
                return tid

        kie = MiddleUnknown(synthetic_video)
        pending = submit_run(run_id, kie=kie, gdrive=FakeGDriveClient())
        assert len(kie.create_task_records) == 3      # one POST each
        assert len(pending) == 2                       # the siblings keep going
        db_session.expire_all()
        middle = next(rs for rs in db_session.get(Run, run_id).run_segments if rs.index == 2)
        (att,) = _attempt_rows(db_engine, owner_id=middle.id)
        assert att.status == "parked" and att.sender_released_at is not None
        assert db_session.get(Run, run_id).status == RunStatus.processing

        while pending:
            pending = poll_run_once(run_id, pending=pending, kie=kie)
        finalize_run(run_id, gdrive=FakeGDriveClient())
        db_session.expire_all()
        run = db_session.get(Run, run_id)
        assert run.status == RunStatus.incomplete
        done = {rs.index: rs.status for rs in run.run_segments}
        assert done[0] == SegmentStatus.completed and done[4] == SegmentStatus.completed
        assert done[2] != SegmentStatus.completed
        assert len(kie.create_task_records) == 3       # never resubmitted by the pipeline

        # A Retry Run (the run re-queued) does not resubmit it either.
        run.status = RunStatus.queued
        db_session.commit()
        retry_kie = FakeKieClient(synthetic_video)
        assert submit_run(run_id, kie=retry_kie, gdrive=FakeGDriveClient()) == {}
        assert retry_kie.create_task_records == []
        finalize_run(run_id, gdrive=FakeGDriveClient())
        db_session.expire_all()
        assert db_session.get(Run, run_id).status == RunStatus.incomplete

        # Resubmit from the card (the operator checked kie.ai): CAS + relaunch.
        enq = []
        monkeypatch.setattr(api_v2, "enqueue_process_run", lambda rid, **kw: enq.append(rid))
        db = db_session
        row = paid_attempts.resubmit(db, att.id, actor="op", note="nothing on kie.ai")
        relaunch(db, row)
        db.commit()
        assert enq == [run_id]
        fresh = FakeKieClient(synthetic_video)
        pending = submit_run(run_id, kie=fresh, gdrive=FakeGDriveClient())
        assert len(fresh.create_task_records) == 1 and len(pending) == 1
        while pending:
            pending = poll_run_once(run_id, pending=pending, kie=fresh)
        finalize_run(run_id, gdrive=FakeGDriveClient())
        db_session.expire_all()
        assert db_session.get(Run, run_id).status == RunStatus.done
        rows = _attempt_rows(db_engine, owner_id=middle.id)
        assert [r.status for r in rows] == ["resubmitted", "accepted"]
