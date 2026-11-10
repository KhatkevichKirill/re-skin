"""Tests for app/tasks.py enqueue helpers — ensure generous RQ job_timeouts
and correct queue routing.

Regression: RQ's default job_timeout (180s) killed long analyze/process jobs
mid-flight ("Task exceeded maximum timeout value (180 seconds)") while the
Seedance task kept running on kie.ai.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock

import pytest

import app.tasks as tasks


def _patch_queue(monkeypatch, queue_name=None):
    """Patch _get_queue so any call returns fake_queue regardless of queue name."""
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-123")
    fake_queue.enqueue_in.return_value = MagicMock(id="rq-456")
    if queue_name is None:
        monkeypatch.setattr(tasks, "_get_queue", lambda *args, **kwargs: fake_queue)
    else:
        monkeypatch.setattr(tasks, "_get_queue",
                            lambda name=tasks.QUEUE_PROCESS, **kw:
                            fake_queue if name == queue_name else MagicMock())
    return fake_queue


def test_enqueue_analyze_project_sets_generous_timeout(monkeypatch):
    """enqueue_analyze_project pushes to the 'analyze' queue with a generous timeout."""
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-123")
    # Capture which queue name was requested.
    captured_name = {}
    def _fake_get_queue(name=tasks.QUEUE_PROCESS, **kw):
        captured_name["name"] = name
        return fake_queue
    monkeypatch.setattr(tasks, "_get_queue", _fake_get_queue)

    tasks.enqueue_analyze_project("proj-abc")

    # Must enqueue to the analyze queue.
    assert captured_name.get("name") == tasks.QUEUE_ANALYZE

    args, kwargs = fake_queue.enqueue.call_args
    assert args[0] == "app.tasks.run_analyze_project"
    assert args[1] == "proj-abc"
    assert kwargs["job_timeout"] == tasks.ANALYZE_JOB_TIMEOUT
    assert tasks.ANALYZE_JOB_TIMEOUT >= 600  # well above RQ's 180s default


def test_enqueue_process_run_routes_to_process_queue(monkeypatch):
    """enqueue_process_run pushes run_submit_run to the 'process' queue."""
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-123")
    captured_name = {}
    def _fake_get_queue(name=tasks.QUEUE_PROCESS, **kw):
        captured_name["name"] = name
        return fake_queue
    monkeypatch.setattr(tasks, "_get_queue", _fake_get_queue)

    tasks.enqueue_process_run("run-xyz")

    # Must enqueue to the process queue.
    assert captured_name.get("name") == tasks.QUEUE_PROCESS

    args, kwargs = fake_queue.enqueue.call_args
    assert args[0] == "app.tasks.run_submit_run"
    assert args[1] == "run-xyz"
    assert kwargs["job_timeout"] == tasks.SUBMIT_JOB_TIMEOUT
    # Submit phase timeout must be well above RQ's 180s default.
    assert tasks.SUBMIT_JOB_TIMEOUT >= 600


def test_enqueue_process_run_sets_generous_timeout(monkeypatch):
    """PROCESS_JOB_TIMEOUT remains defined and ≥ 3600 (used for lock TTL)."""
    _patch_queue(monkeypatch)
    # PROCESS_JOB_TIMEOUT is the combined lifecycle budget; kept for lock sizing.
    assert tasks.PROCESS_JOB_TIMEOUT >= 3600


def test_enqueue_process_overlay_routes_to_overlay_queue(monkeypatch):
    """enqueue_process_overlay pushes to the 'overlay' queue."""
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-789")
    captured_name = {}
    def _fake_get_queue(name=tasks.QUEUE_PROCESS, **kw):
        captured_name["name"] = name
        return fake_queue
    monkeypatch.setattr(tasks, "_get_queue", _fake_get_queue)

    tasks.enqueue_process_overlay("overlay-job-1")

    assert captured_name.get("name") == tasks.QUEUE_OVERLAY
    args, kwargs = fake_queue.enqueue.call_args
    assert args[0] == "app.tasks.run_process_overlay"
    assert args[1] == "overlay-job-1"
    assert kwargs["job_timeout"] == tasks.OVERLAY_JOB_TIMEOUT


class _FakeFinalCreativesSheets:
    """Records rows appended to the Final Creatives tab, with a canned header."""

    HEADER = [
        "Date", "Job ID", "Idea", "Input Link", "Creative Link",
        "Pipeline Type", "Run on Meta",
    ]

    def __init__(self):
        self.appended = []

    def get_header(self, sid, tab):
        return list(self.HEADER)

    def column_values(self, sid, tab, column_index, *, last_n=None):
        return ["Job ID"]  # header only — nothing to dedupe against

    def append_row(self, sid, tab, row):
        self.appended.append(row)
        return {"updates": {"updatedRows": 1}}


class _FakeSession:
    """Minimal SessionLocal stand-in: a context manager whose get() is canned."""

    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, model, oid):
        return self._obj


def test_export_run_to_sheet_writes_idea_column(monkeypatch):
    """The Final Creatives row carries Idea = project_run_MODEL under its header."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc

    project = SimpleNamespace(
        id="proj-1234-5678",
        name="Gardener Morning Routine Hooks (Jp)",
        source_ref="https://drive/in/view",
    )
    run = SimpleNamespace(
        id="abcd1234-ef56-7890-1234-567890abcdef",
        name="Q-00873 fast",
        model="seedance-2-5",
        result_gdrive_file_id="FILE1",
        project=project,
        language=None,
        short_id="abcd1234",  # no language → bare prefix (Run.short_id)
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(run))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_run_to_sheet(run.id)

    assert len(fake.appended) == 1
    idea = fake.appended[0][_FakeFinalCreativesSheets.HEADER.index("Idea")]
    # "(Jp)" dropped whole, spaces collapsed, project clamped to 27 / run to 7.
    assert idea == "GardenerMorningRoutineHooks_Q-00873_SD25"
    assert len(idea) <= sc.IDEA_MAX_LEN


def test_export_tail_edit_to_sheet_reuses_parent_run_idea(monkeypatch):
    """A tail edit exports the SAME Idea as the run it was cut from."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id="run-1", name="Q-00873", model="seedance-fast",
        result_gdrive_file_id="RUNFILE", project=project, language=None,
    )
    edit = SimpleNamespace(
        id="edit-abcd-1234", run_id="run-1", run=run, gdrive_file_id="EDITFILE",
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(edit))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_tail_edit_to_sheet(edit.id)

    idea = fake.appended[0][_FakeFinalCreativesSheets.HEADER.index("Idea")]
    assert idea == "Gardener_Q-00873_SDf2"


# ---------------------------------------------------------------------------
# Run language → sheet Job ID
# ---------------------------------------------------------------------------
# The downstream renaming pipeline reads the creative's language back out of
# the Job ID cell (see app/languages.py), so these pin the exact strings it
# parses: "-ja" for Japanese, NOTHING for English.


def _job_id_cell(fake):
    return fake.appended[0][_FakeFinalCreativesSheets.HEADER.index("Job ID")]


def test_export_run_to_sheet_writes_language_suffixed_job_id(monkeypatch):
    """A Japanese run's Job ID is its 8-char prefix + '-ja'."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id="abcd1234-ef56-7890-1234-567890abcdef",
        name="Q-00873", model="seedance-2-5", result_gdrive_file_id="FILE1",
        project=project, language="ja", short_id="abcd1234-ja",
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(run))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_run_to_sheet(run.id)

    assert _job_id_cell(fake) == "abcd1234-ja"


def test_export_tail_edit_to_sheet_inherits_parent_run_language(monkeypatch):
    """A tail edit keeps its OWN prefix but the PARENT run's language."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id="run-1", name="Q-00873", model="seedance-fast",
        result_gdrive_file_id="RUNFILE", project=project, language="ja",
    )
    edit = SimpleNamespace(
        id="edit-abcd-1234", run_id="run-1", run=run, gdrive_file_id="EDITFILE",
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(edit))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_tail_edit_to_sheet(edit.id)

    # Own 8-char prefix (so it doesn't collide with the run's row), parent's
    # language (so it doesn't rename as English).
    assert _job_id_cell(fake) == "edit-abc-ja"


def _overlay_export_fakes(monkeypatch, *, sibling_count, language="ja"):
    """Wire run_export_overlay_to_sheet up to fakes; return (overlay, run, sheet).

    *sibling_count* is what overlay_runs.count_overlay_jobs_for_run would
    return for the parent run — the "one overlay or several?" input to the
    Job ID rule, faked here so the test doesn't need a database.
    """
    from types import SimpleNamespace

    import app.db as db_mod
    import app.models as models_mod
    import app.overlay_runs as overlay_runs
    import app.sheets_client as sc

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id="run-1234-5678", name="Q-00873", model="seedance-2-5",
        result_gdrive_file_id="RUNFILE", project=project, language=language,
        short_id=f"run-1234-{language}" if language else "run-1234",
    )
    overlay = SimpleNamespace(
        id="ovl-abcd-1234", gdrive_file_id="OVLFILE",
        source_run_id=run.id, source_tail_edit_id=None,
    )

    class _ByModelSession(_FakeSession):
        """db.get() dispatches on the model, unlike the single-object fake."""

        def get(self, model, oid):
            return run if model is models_mod.Run else overlay

    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _ByModelSession(overlay))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)
    monkeypatch.setattr(
        overlay_runs, "count_overlay_jobs_for_run", lambda run_id, db: sibling_count
    )
    return overlay, run, fake


def test_export_overlay_to_sheet_uses_parent_run_job_id(monkeypatch):
    """The run's ONLY overlay goes to the sheet under the run's own Job ID.

    Not an id of its own: the overlay is that run's creative re-dressed, and
    Pipeline Type (re-skin-overlay vs re-skin-v2) is what keeps the two rows
    apart downstream.
    """
    overlay, run, fake = _overlay_export_fakes(monkeypatch, sibling_count=1)

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert _job_id_cell(fake) == run.short_id == "run-1234-ja"
    pipeline = fake.appended[0][_FakeFinalCreativesSheets.HEADER.index("Pipeline Type")]
    assert pipeline == "re-skin-overlay"


def test_export_overlay_to_sheet_combines_ids_when_run_has_several(monkeypatch):
    """With several overlays on one run each carries run prefix + own prefix."""
    overlay, run, fake = _overlay_export_fakes(monkeypatch, sibling_count=2)

    tasks.run_export_overlay_to_sheet(overlay.id)

    # {run prefix}_{overlay prefix} + the PARENT run's language suffix.
    assert _job_id_cell(fake) == "run-1234_ovl-abcd-ja"


def test_export_overlay_to_sheet_english_run_id_is_unsuffixed(monkeypatch):
    """English (the default) writes the bare run id, as everywhere else."""
    overlay, run, fake = _overlay_export_fakes(
        monkeypatch, sibling_count=1, language=None
    )

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert _job_id_cell(fake) == "run-1234"


def test_export_overlay_to_sheet_dedupes_on_creative_link(monkeypatch):
    """A re-run of the export task doesn't append the same creative twice.

    Job ID can no longer serve as the dedupe key (it is deliberately shared
    with the parent run's row), so the check is against Creative Link.
    """
    overlay, run, fake = _overlay_export_fakes(monkeypatch, sibling_count=1)
    creative_link = "https://drive.google.com/file/d/OVLFILE/view"
    creative_idx = _FakeFinalCreativesSheets.HEADER.index("Creative Link")

    def _column_values(sid, tab, column_index, *, last_n=None):
        return [creative_link] if column_index == creative_idx else []

    fake.column_values = _column_values

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert fake.appended == []


def test_export_overlay_to_sheet_appends_despite_the_runs_own_row(monkeypatch):
    """The parent run's row sharing this Job ID must NOT block the overlay.

    The whole point of the shared id: both rows exist, told apart by Pipeline
    Type. A Job-ID dedupe would have silently dropped this append.
    """
    overlay, run, fake = _overlay_export_fakes(monkeypatch, sibling_count=1)
    job_idx = _FakeFinalCreativesSheets.HEADER.index("Job ID")

    def _column_values(sid, tab, column_index, *, last_n=None):
        # The run already exported itself under the very same Job ID.
        return [run.short_id] if column_index == job_idx else []

    fake.column_values = _column_values

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert len(fake.appended) == 1
    assert _job_id_cell(fake) == run.short_id


class _FakeRedis:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.set_calls = []
        self.eval_calls = []

    def set(self, *args, **kwargs):
        self.set_calls.append((args, kwargs))
        return self.acquired

    def eval(self, *args):
        self.eval_calls.append(args)
        return 1

    def get(self, *args, **kwargs):
        return None  # no poll state

    def setex(self, *args, **kwargs):
        pass

    def delete(self, *args, **kwargs):
        pass


def test_run_submit_run_skips_duplicate_when_lock_held(monkeypatch):
    """run_submit_run skips processing if the per-run lock is already held."""
    fake_redis = _FakeRedis(acquired=False)
    submit_calls = []

    # Patch the module-level cached connection.
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    # Clear cached queues so we don't use stale ones.
    monkeypatch.setattr(tasks, "_queues", {})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "submit_run", lambda *a, **k: submit_calls.append((a, k)))

    tasks.run_submit_run("run-locked")

    assert submit_calls == []
    assert fake_redis.eval_calls == []
    assert fake_redis.set_calls[0][1]["nx"] is True
    assert fake_redis.set_calls[0][1]["ex"] == tasks._RUN_LOCK_TTL_SEC


def test_run_submit_run_releases_lock_when_nothing_to_do(monkeypatch):
    """run_submit_run releases the lock when submit_run returns None (terminal run)."""
    fake_redis = _FakeRedis(acquired=True)

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {})

    import app.pipeline_v2 as pipeline_v2
    import app.gdrive_client as gdrive_client

    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    # submit_run returns None → run is terminal, nothing to do.
    monkeypatch.setattr(pipeline_v2, "submit_run", lambda *a, **k: None)

    tasks.run_submit_run("run-terminal")

    # Lock must have been acquired and then released.
    assert len(fake_redis.set_calls) == 1
    assert len(fake_redis.eval_calls) == 1
    script, num_keys, lock_key, token = fake_redis.eval_calls[0]
    assert "redis.call(\"GET\"" in script
    assert lock_key == "reskin:run:lock:run-terminal"


def test_run_submit_run_enqueues_poll_when_pending(monkeypatch):
    """run_submit_run enqueues run_poll_run (via enqueue_in) when tasks are pending."""
    fake_redis = _FakeRedis(acquired=True)
    fake_queue = MagicMock()
    fake_queue.enqueue_in.return_value = MagicMock(id="rq-poll")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    import app.gdrive_client as gdrive_client

    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    # submit_run returns a non-empty pending map → should schedule a poll round.
    pending = {"task-abc": {"rs_id": "rs-1", "index": 0, "deadline": 9e9, "attempt": 1}}
    monkeypatch.setattr(pipeline_v2, "submit_run", lambda *a, **k: dict(pending))

    tasks.run_submit_run("run-with-tasks")

    # enqueue_in must have been called once (to schedule the first poll).
    assert fake_queue.enqueue_in.called
    call_args = fake_queue.enqueue_in.call_args
    func_ref = call_args[0][1]  # second positional arg is the function reference
    assert "run_poll_run" in func_ref


def test_run_submit_run_enqueues_finalize_when_empty_pending(monkeypatch):
    """run_submit_run enqueues run_finalize_run (via enqueue) when pending is empty."""
    fake_redis = _FakeRedis(acquired=True)
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-fin")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    import app.gdrive_client as gdrive_client

    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    # submit_run returns empty pending → all segments already done, go finalize.
    monkeypatch.setattr(pipeline_v2, "submit_run", lambda *a, **k: {})

    tasks.run_submit_run("run-all-done")

    assert fake_queue.enqueue.called
    call_args = fake_queue.enqueue.call_args
    func_ref = call_args[0][0]
    assert "run_finalize_run" in func_ref


def test_run_poll_run_reenqueues_when_still_pending(monkeypatch):
    """run_poll_run re-enqueues itself when tasks remain pending after the round."""
    fake_redis = _FakeRedis()
    fake_redis._state = {"task-1": {"rs_id": "rs-1", "index": 0, "deadline": 9e9, "attempt": 1}}

    import json as _json
    fake_redis.get = lambda key: _json.dumps(fake_redis._state).encode()

    fake_queue = MagicMock()
    fake_queue.enqueue_in.return_value = MagicMock(id="rq-poll2")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    # poll_run_once returns non-empty → tasks still pending
    monkeypatch.setattr(pipeline_v2, "poll_run_once",
                        lambda run_id, pending, kie=None: dict(pending))

    tasks.run_poll_run("run-still-pending",
                       lock_key="reskin:run:lock:run-still-pending",
                       lock_token="tok")

    # Must re-enqueue itself via enqueue_in.
    assert fake_queue.enqueue_in.called
    call_args = fake_queue.enqueue_in.call_args
    func_ref = call_args[0][1]
    assert "run_poll_run" in func_ref


def test_run_poll_run_enqueues_finalize_when_all_done(monkeypatch):
    """run_poll_run enqueues run_finalize_run when all tasks complete."""
    fake_redis = _FakeRedis()
    import json as _json
    fake_redis.get = lambda key: _json.dumps(
        {"task-1": {"rs_id": "rs-1", "index": 0, "deadline": 9e9, "attempt": 1}}
    ).encode()

    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-fin")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    # poll_run_once returns empty → all tasks terminal
    monkeypatch.setattr(pipeline_v2, "poll_run_once",
                        lambda run_id, pending, kie=None: {})

    tasks.run_poll_run("run-all-done",
                       lock_key="reskin:run:lock:run-all-done",
                       lock_token="tok")

    assert fake_queue.enqueue.called
    call_args = fake_queue.enqueue.call_args
    func_ref = call_args[0][0]
    assert "run_finalize_run" in func_ref


def test_run_poll_run_aborts_when_lock_lost(monkeypatch):
    """run_poll_run stops its chain when the lock TTL refresh fails (lost lock)."""
    fake_redis = _FakeRedis()
    fake_redis.eval = lambda *args: 0  # refresh reports lock lost

    fake_queue = MagicMock()
    poll_calls = []

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "poll_run_once",
                        lambda run_id, pending, kie=None: poll_calls.append(run_id))

    tasks.run_poll_run("run-lock-lost",
                       lock_key="reskin:run:lock:run-lock-lost",
                       lock_token="tok")

    # No polling, no re-enqueue, no finalize — the new lock owner drives the run.
    assert poll_calls == []
    assert not fake_queue.enqueue_in.called
    assert not fake_queue.enqueue.called


def test_run_poll_run_refreshes_lock_ttl(monkeypatch):
    """Each poll round re-arms the lock TTL via the token-checked EXPIRE script."""
    fake_redis = _FakeRedis()
    import json as _json
    fake_redis.get = lambda key: _json.dumps(
        {"task-1": {"rs_id": "rs-1", "index": 0, "deadline": 9e9, "attempt": 1}}
    ).encode()

    fake_queue = MagicMock()
    fake_queue.enqueue_in.return_value = MagicMock(id="rq-poll2")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "poll_run_once",
                        lambda run_id, pending, kie=None: dict(pending))

    tasks.run_poll_run("run-refresh",
                       lock_key="reskin:run:lock:run-refresh",
                       lock_token="tok")

    refresh_calls = [c for c in fake_redis.eval_calls if "EXPIRE" in c[0]]
    assert len(refresh_calls) == 1
    assert refresh_calls[0][1:5] == (
        2, "reskin:run:lock:run-refresh", "reskin:run:lock:run-refresh:gen", "tok")


# Legacy backward-compat: run_process_run must still delegate to run_submit_run.
def test_run_process_run_delegates_to_submit(monkeypatch):
    """run_process_run (legacy) delegates to run_submit_run."""
    called = []
    monkeypatch.setattr(tasks, "run_submit_run", lambda run_id: called.append(run_id))
    tasks.run_process_run("run-legacy")
    assert called == ["run-legacy"]


def test_run_process_run_skips_duplicate_when_lock_held(monkeypatch):
    """Legacy run_process_run skips processing if the per-run lock is already held."""
    fake_redis = _FakeRedis(acquired=False)
    submit_calls = []

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "submit_run", lambda *a, **k: submit_calls.append((a, k)))

    tasks.run_process_run("run-locked")

    assert submit_calls == []


def test_worker_masks_redis_url_password():
    from worker.worker import _mask_redis_url

    assert (
        _mask_redis_url("redis://:secret-password@redis:6379/0")
        == "redis://***:***@redis:6379/0"
    )
    assert (
        _mask_redis_url("redis://user:secret-password@redis:6379/0")
        == "redis://***:***@redis:6379/0"
    )


# ---------------------------------------------------------------------------
# Preprocess (GPT Image 2) RQ wrappers
# ---------------------------------------------------------------------------


def test_enqueue_preprocess_run_routes_to_process_queue(monkeypatch):
    """enqueue_preprocess_run pushes run_preprocess_refs to the 'process' queue."""
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-pp")
    captured_name = {}

    def _fake_get_queue(name=tasks.QUEUE_PROCESS, **kw):
        captured_name["name"] = name
        return fake_queue

    monkeypatch.setattr(tasks, "_get_queue", _fake_get_queue)

    tasks.enqueue_preprocess_run("run-pp-1")

    assert captured_name.get("name") == tasks.QUEUE_PROCESS
    args, kwargs = fake_queue.enqueue.call_args
    assert args[0] == "app.tasks.run_preprocess_refs"
    assert args[1] == "run-pp-1"
    assert kwargs["job_timeout"] == tasks.SUBMIT_JOB_TIMEOUT


def test_run_preprocess_refs_skips_duplicate_when_lock_held(monkeypatch):
    """run_preprocess_refs skips when the per-run lock is already held."""
    fake_redis = _FakeRedis(acquired=False)
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {})

    import app.pipeline_v2 as pipeline_v2
    calls = []
    monkeypatch.setattr(pipeline_v2, "preprocess_reference_images",
                        lambda *a, **k: calls.append(k) or {})

    tasks.run_preprocess_refs("run-locked-pp")
    assert calls == []


def test_run_preprocess_refs_enqueues_poll_when_pending(monkeypatch):
    """When preprocess creates pending tasks, run_preprocess_refs schedules
    run_preprocess_poll (via enqueue_in)."""
    fake_redis = _FakeRedis(acquired=True)
    fake_queue = MagicMock()
    fake_queue.enqueue_in.return_value = MagicMock(id="rq-pp-poll")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    import app.gdrive_client as gdrive_client
    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    pending = {"img-task-1": 0, "img-task-2": 1}
    monkeypatch.setattr(pipeline_v2, "preprocess_reference_images",
                        lambda *a, **k: dict(pending))
    # Avoid touching the DB for the transition.
    monkeypatch.setattr(tasks, "_transition_preprocessing_to_queued", lambda *a, **k: None)

    tasks.run_preprocess_refs("run-pp-with-tasks")

    assert fake_queue.enqueue_in.called
    func_ref = fake_queue.enqueue_in.call_args[0][1]
    assert "run_preprocess_poll" in func_ref


def test_run_preprocess_refs_hands_off_to_review_when_no_pending(monkeypatch):
    """R2-new: when preprocess returns no pending tasks (all pass-through /
    disabled / no trims), run_preprocess_refs transitions to preprocess_review
    (NOT queued) and releases the lock — submit is NOT enqueued. The operator
    must approve before Seedance starts."""
    fake_redis = _FakeRedis(acquired=True)
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-submit")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    import app.gdrive_client as gdrive_client
    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    monkeypatch.setattr(pipeline_v2, "preprocess_reference_images",
                        lambda *a, **k: {})
    # Ordinary run (not a preprocess copy) — patched rather than read from the
    # DB, like the transition helpers below.
    monkeypatch.setattr(tasks, "_preprocess_auto_approve", lambda rid: False)
    transitioned = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda *a, **k: transitioned.append(a),
    )

    tasks.run_preprocess_refs("run-pp-empty")

    assert transitioned  # transitioned to preprocess_review
    # R2-new: submit is NOT enqueued — the operator must approve first.
    assert not fake_queue.enqueue.called
    assert not fake_queue.enqueue_in.called


def test_run_preprocess_poll_reenqueues_when_still_pending(monkeypatch):
    """run_preprocess_poll re-enqueues itself when tasks remain pending."""
    import json as _json
    fake_redis = _FakeRedis()
    state = {"img-task-1": 0}
    fake_redis.get = lambda key: _json.dumps(state).encode()

    fake_queue = MagicMock()
    fake_queue.enqueue_in.return_value = MagicMock(id="rq-pp-poll2")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "poll_preprocess_once",
                        lambda run_id, pending, kie=None: {"pending": dict(pending), "failed": []})

    tasks.run_preprocess_poll("run-pp-still",
                               lock_key="reskin:run:lock:run-pp-still",
                               lock_token="tok")

    assert fake_queue.enqueue_in.called
    func_ref = fake_queue.enqueue_in.call_args[0][1]
    assert "run_preprocess_poll" in func_ref


def test_run_preprocess_poll_pauses_for_review_when_all_done(monkeypatch):
    """R2-new: run_preprocess_poll transitions to preprocess_review (NOT
    queued) when all preprocess tasks are terminal (poll returns empty).
    Submit is NOT enqueued; the lock is released so the approve endpoint can
    re-acquire it."""
    import json as _json
    fake_redis = _FakeRedis()
    state = {"img-task-1": 0}
    fake_redis.get = lambda key: _json.dumps(state).encode()

    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-submit")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "poll_preprocess_once",
                        lambda run_id, pending, kie=None: {})
    # Ordinary run (not a preprocess copy) — patched rather than read from the
    # DB, like the transition helpers below.
    monkeypatch.setattr(tasks, "_preprocess_auto_approve", lambda rid: False)
    transitioned = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda *a, **k: transitioned.append(a),
    )

    tasks.run_preprocess_poll("run-pp-done",
                               lock_key="reskin:run:lock:run-pp-done",
                               lock_token="tok")

    assert transitioned  # transitioned to preprocess_review
    # R2-new: submit is NOT enqueued.
    assert not fake_queue.enqueue.called
    assert not fake_queue.enqueue_in.called


def test_run_preprocess_poll_auto_approves_and_submits(monkeypatch):
    """A copy carrying preprocess_auto_approve skips the review pause: it takes
    the preprocessing → queued edge and hands the STILL-HELD lock to submit, so
    the chain runs unbroken from preprocessing through finalize."""
    import json as _json
    fake_redis = _FakeRedis()
    fake_redis.get = lambda key: _json.dumps({"img-task-1": 1}).encode()

    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-submit")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "poll_preprocess_once", lambda *a, **k: {})
    monkeypatch.setattr(tasks, "_preprocess_auto_approve", lambda rid: True)
    queued = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_queued",
        lambda rid: queued.append(rid) or True,
    )
    reviewed = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda *a, **k: reviewed.append(a),
    )

    tasks.run_preprocess_poll("run-pp-auto",
                              lock_key="reskin:run:lock:run-pp-auto",
                              lock_token="tok")

    assert queued == ["run-pp-auto"]
    assert reviewed == []            # never paused for a human
    args, kwargs = fake_queue.enqueue.call_args
    assert args[0] == "app.tasks.run_submit_run"
    assert args[1] == "run-pp-auto"
    # The lock is handed off, not released — submit refreshes this same token.
    assert kwargs["lock_token"] == "tok"


def test_run_preprocess_refs_start_failure_parks_at_review_with_a_reference_set(
    monkeypatch,
):
    """A regen that cannot even create its task must not take the run down: the
    images already on disk stay, and the operator retries from the review card.
    Failing would strand the run — `failed → queued` is the only edge out, and
    that submits."""
    fake_redis = _FakeRedis()
    fake_queue = MagicMock()
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2

    def _boom(*a, **k):
        raise RuntimeError("kie 503")

    monkeypatch.setattr(pipeline_v2, "preprocess_reference_images", _boom)
    monkeypatch.setattr(tasks, "_should_park_failed_preprocess", lambda rid: True)
    failed = []
    monkeypatch.setattr(
        tasks, "_fail_preprocessing_run",
        lambda rid, error_message=None: failed.append(rid),
    )
    reviewed = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda rid, **kw: reviewed.append((rid, kw.get("error_message"))),
    )

    with pytest.raises(RuntimeError):
        tasks.run_preprocess_refs("run-pp-boom")

    assert failed == []
    assert reviewed and reviewed[0][0] == "run-pp-boom"
    assert "kie 503" in reviewed[0][1]


def test_run_preprocess_refs_auto_approves_when_no_pending(monkeypatch):
    """Same for the no-tasks path (every ref pass-through): an auto-approve run
    goes to submit instead of parking at preprocess_review."""
    fake_redis = _FakeRedis(acquired=True)
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-submit")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    import app.gdrive_client as gdrive_client
    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    monkeypatch.setattr(pipeline_v2, "preprocess_reference_images", lambda *a, **k: {})
    monkeypatch.setattr(tasks, "_preprocess_auto_approve", lambda rid: True)
    monkeypatch.setattr(tasks, "_transition_preprocessing_to_queued", lambda rid: True)
    reviewed = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda *a, **k: reviewed.append(a),
    )

    tasks.run_preprocess_refs("run-pp-auto-empty")

    assert reviewed == []
    assert fake_queue.enqueue.call_args[0][0] == "app.tasks.run_submit_run"


def test_auto_approve_does_not_submit_from_an_unexpected_status(monkeypatch):
    """If the run could not reach `queued`, the lock is released and submit is
    NOT enqueued — submit_run would fail the run outright from any other
    status, so silence beats a wrong-status submission."""
    import json as _json
    fake_redis = _FakeRedis()
    fake_redis.get = lambda key: _json.dumps({"img-task-1": 1}).encode()

    fake_queue = MagicMock()
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "poll_preprocess_once", lambda *a, **k: {})
    monkeypatch.setattr(tasks, "_preprocess_auto_approve", lambda rid: True)
    monkeypatch.setattr(tasks, "_transition_preprocessing_to_queued", lambda rid: False)

    tasks.run_preprocess_poll("run-pp-stuck",
                              lock_key="reskin:run:lock:run-pp-stuck",
                              lock_token="tok")

    assert not fake_queue.enqueue.called
    assert not fake_queue.enqueue_in.called
    # Lock released (a DEL eval carrying the token).
    assert any("del" in c[0].lower() for c in fake_redis.eval_calls)


def test_preprocess_failure_still_fails_an_auto_approve_run(monkeypatch):
    """Auto-approve skips the REVIEW, not the error handling: a failed GPT
    Image 2 task still fails the run and never reaches Seedance.

    The park-at-review path must NOT catch this one. By the time the check runs,
    poll has already written the character photo into
    `preprocessed_reference_urls`, so "does the run hold a reference set?" is
    true — it is the unattended half of the rule that keeps it failing. Left to
    the REAL helper (over a stand-in session) rather than a stub, because that
    is the whole invariant: a parked automation run is not terminal and would
    stall its batch and every batch after it.
    """
    import json as _json
    from app.models import Run
    from app.state_machine import RunStatus

    fake_redis = _FakeRedis()
    fake_redis.get = lambda key: _json.dumps({"img-task-1": 1}).encode()

    fake_queue = MagicMock()
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(
        pipeline_v2, "poll_preprocess_once",
        lambda *a, **k: {"pending": {}, "failed": [{"reason": "content policy"}]},
    )

    run = Run(
        id="run-pp-badgen",
        project_id="p1",
        prompt="swap",
        resolution="480p",
        status=RunStatus.preprocessing,
        preprocess_enabled=True,
        preprocess_auto_approve=True,
        reference_image_urls=["/d/char.png", "/d/f0.jpg"],
        # What poll leaves behind: the character photo passed through, the one
        # generated trim failed.
        preprocessed_reference_urls=["/d/char.png", None],
        preprocess_task_ids=[None, "img-task-1"],
    )

    class _FakeSession:
        def get(self, model, run_id):
            return run

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import app.db as db_mod
    monkeypatch.setattr(db_mod, "get_session", lambda: _FakeSession())

    failed = []
    monkeypatch.setattr(
        tasks, "_fail_preprocessing_run",
        lambda rid, error_message=None: failed.append((rid, error_message)),
    )
    reviewed = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda rid, **kw: reviewed.append(rid),
    )

    tasks.run_preprocess_poll("run-pp-badgen",
                              lock_key="reskin:run:lock:run-pp-badgen",
                              lock_token="tok")

    assert reviewed == []
    assert failed and failed[0][0] == "run-pp-badgen"
    assert "content policy" in failed[0][1]
    assert not fake_queue.enqueue.called


def test_preprocess_failure_parks_at_review_when_a_reference_set_exists(
    monkeypatch,
):
    """A failed task does NOT take the run down once the run holds a reference
    set — that is the single-image regen case, where failing would throw away
    exactly what the operator kept AND strand the run (`failed → queued` is the
    only edge out, and it submits). It parks back at review with the error
    instead, and never enqueues submit."""
    import json as _json
    fake_redis = _FakeRedis()
    fake_redis.get = lambda key: _json.dumps({"img-task-1": 2}).encode()

    fake_queue = MagicMock()
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(
        pipeline_v2, "poll_preprocess_once",
        lambda *a, **k: {"pending": {}, "failed": [{"reason": "content policy"}]},
    )
    monkeypatch.setattr(tasks, "_should_park_failed_preprocess", lambda rid: True)
    failed = []
    monkeypatch.setattr(
        tasks, "_fail_preprocessing_run",
        lambda rid, error_message=None: failed.append((rid, error_message)),
    )
    reviewed = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda rid, **kw: reviewed.append((rid, kw.get("error_message"))),
    )

    tasks.run_preprocess_poll("run-pp-partial",
                              lock_key="reskin:run:lock:run-pp-partial",
                              lock_token="tok")

    assert failed == []
    assert reviewed and reviewed[0][0] == "run-pp-partial"
    assert "content policy" in reviewed[0][1]
    assert not fake_queue.enqueue.called


def test_preprocess_failure_with_tasks_still_in_flight_fails_the_run(monkeypatch):
    """The survivor rule only applies on the LAST round. While other tasks are
    still running there is nothing stable to park on, so R2's fail-fast
    stands."""
    import json as _json
    fake_redis = _FakeRedis()
    fake_redis.get = lambda key: _json.dumps({"img-task-1": 1}).encode()

    fake_queue = MagicMock()
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(
        pipeline_v2, "poll_preprocess_once",
        lambda *a, **k: {
            "pending": {"img-task-2": 2},
            "failed": [{"reason": "content policy"}],
        },
    )
    monkeypatch.setattr(tasks, "_should_park_failed_preprocess", lambda rid: True)
    failed = []
    monkeypatch.setattr(
        tasks, "_fail_preprocessing_run",
        lambda rid, error_message=None: failed.append((rid, error_message)),
    )
    reviewed = []
    monkeypatch.setattr(
        tasks, "_transition_preprocessing_to_review",
        lambda rid, **kw: reviewed.append(rid),
    )

    tasks.run_preprocess_poll("run-pp-inflight",
                              lock_key="reskin:run:lock:run-pp-inflight",
                              lock_token="tok")

    assert reviewed == []
    assert failed and failed[0][0] == "run-pp-inflight"


def test_should_park_counts_pass_through_images_and_skips_unattended_runs(
    monkeypatch, tmp_path,
):
    """Two halves of the park rule.

    Regenerating the ONLY generated image must still park at review: the
    stricter "a generated image survives" test flips to False there, and failing
    that run strands it in a fail → retry → fail loop (retry submits, and the
    empty slot crashes resolve_reference_urls).

    An auto-approve run must NEVER park: `preprocess_review` is not in
    automation._TERMINAL_RUN_STATUSES, so one parked run would block its batch
    and every later batch of that automation until a human noticed."""
    from app.models import Run
    from app.state_machine import RunStatus

    class _FakeSession:
        def __init__(self, run):
            self._run = run

        def get(self, model, run_id):
            return self._run

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    run = Run(
        id="run-pp-lastone",
        project_id="p1",
        prompt="swap",
        resolution="480p",
        status=RunStatus.preprocessing,
        preprocess_enabled=True,
        # The character photo passed through; the one generated trim was just
        # cleared for regeneration, so no slot has BOTH a task id and a file.
        reference_image_urls=["/d/char.png", "/d/f0.jpg"],
        preprocessed_reference_urls=["/d/char.png", None],
        preprocess_task_ids=[None, None],
    )
    # The helper imports get_session from app.db inside the function.
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "get_session", lambda: _FakeSession(run))

    assert tasks._should_park_failed_preprocess("run-pp-lastone") is True

    run.preprocessed_reference_urls = None
    assert tasks._should_park_failed_preprocess("run-pp-lastone") is False

    # Unattended: fail as R2 always did, whatever the run holds.
    run.preprocessed_reference_urls = ["/d/char.png", None]
    run.preprocess_auto_approve = True
    assert tasks._should_park_failed_preprocess("run-pp-lastone") is False


def test_parking_at_review_voids_auto_approve(monkeypatch):
    """The failure paths are the ONLY way an auto-approve run reaches review.
    From there a human is in the loop and the card promises the run will not
    start until they approve — so the flag must not survive, or the next
    Regenerate would submit straight past them."""
    from app.models import Run
    from app.state_machine import RunStatus

    class _FakeSession:
        def __init__(self, run):
            self._run = run
            self.committed = False

        def get(self, model, run_id):
            return self._run

        def commit(self):
            self.committed = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    run = Run(
        id="run-pp-auto-parked",
        project_id="p1",
        prompt="swap",
        resolution="480p",
        status=RunStatus.preprocessing,
        preprocess_enabled=True,
        preprocess_auto_approve=True,
    )
    session = _FakeSession(run)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "get_session", lambda: session)

    tasks._transition_preprocessing_to_review(
        "run-pp-auto-parked", error_message="GPT Image 2 preprocessing failed: x",
    )

    assert run.status == RunStatus.preprocess_review
    assert run.preprocess_auto_approve is False
    assert run.error_message == "GPT Image 2 preprocessing failed: x"
    assert session.committed


def test_run_preprocess_poll_aborts_when_lock_lost(monkeypatch):
    """run_preprocess_poll stops its chain when the lock refresh fails."""
    fake_redis = _FakeRedis()
    # eval returns 0 → lock lost.
    fake_redis.eval = lambda *args: 0
    fake_queue = MagicMock()

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    poll_calls = []
    monkeypatch.setattr(pipeline_v2, "poll_preprocess_once",
                        lambda *a, **k: poll_calls.append(k) or {})

    tasks.run_preprocess_poll("run-pp-lost",
                               lock_key="reskin:run:lock:run-pp-lost",
                               lock_token="tok")

    assert poll_calls == []  # never polled
    assert not fake_queue.enqueue.called
    assert not fake_queue.enqueue_in.called


def test_run_submit_run_accepts_handed_off_lock(monkeypatch):
    """run_submit_run refreshes (not re-acquires) when given an existing lock."""
    fake_redis = _FakeRedis(acquired=True)
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-fin")

    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)
    monkeypatch.setattr(tasks, "_queues", {tasks.QUEUE_PROCESS: fake_queue})

    import app.pipeline_v2 as pipeline_v2
    import app.gdrive_client as gdrive_client
    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    monkeypatch.setattr(pipeline_v2, "submit_run", lambda *a, **k: {})

    tasks.run_submit_run("run-handoff",
                         lock_key="reskin:run:lock:run-handoff",
                         lock_token="existing-tok")

    # No new SET (acquire) call was made — the lock was handed off, only refreshed.
    assert fake_redis.set_calls == []
    # A refresh (eval with EXPIRE) was issued.
    refresh_calls = [c for c in fake_redis.eval_calls if "EXPIRE" in c[0]]
    assert len(refresh_calls) == 1
    assert refresh_calls[0][4] == "existing-tok"


# ---------------------------------------------------------------------------
# Localisation transcription — queue routing + post-analysis wiring
# (docs/localisation.md §8)
# ---------------------------------------------------------------------------


class _FakeProject:
    """Stand-in for a VideoProject row."""

    def __init__(self, project_type="localisation"):
        self.project_type = project_type
        self.transcript_status = None
        self.transcript_error = "an error from a previous attempt"


def _patch_project_lookup(monkeypatch, project):
    """Patch app.db.get_session so tasks' lazy `from .db import get_session`
    yields a session returning *project* (or None). Returns the commit counter.
    """
    from contextlib import contextmanager

    import app.db as db_mod

    commits = []

    class _Session:
        def get(self, _model, _pk):
            return project

        def commit(self):
            commits.append(True)

        def rollback(self):
            pass

    @contextmanager
    def _fake_get_session():
        yield _Session()

    monkeypatch.setattr(db_mod, "get_session", _fake_get_session)
    # The daily-budget admission (AICP-155) needs a real DB; these fakes have
    # none, so admit everything here (refusal is covered below).
    import app.spend as spend_mod

    monkeypatch.setattr(spend_mod, "reserve", lambda *a, **k: 0.0)
    return commits


def test_enqueue_transcribe_project_routes_to_process_queue(monkeypatch):
    """Transcription is network-bound, not CPU-bound: it belongs on the same
    tier as merges and tail edits, not on the analyze queue."""
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-tr")
    captured_name = {}

    def _fake_get_queue(name=tasks.QUEUE_PROCESS, **kw):
        captured_name["name"] = name
        return fake_queue

    monkeypatch.setattr(tasks, "_get_queue", _fake_get_queue)

    tasks.enqueue_transcribe_project("proj-loc")

    assert captured_name.get("name") == tasks.QUEUE_PROCESS
    args, kwargs = fake_queue.enqueue.call_args
    assert args[0] == "app.tasks.run_transcribe_project"
    assert args[1] == "proj-loc"
    assert kwargs["job_timeout"] == tasks.TRANSCRIBE_JOB_TIMEOUT
    # Must outlast the provider timeout the LLM call itself is bounded by.
    from app.config import settings
    assert tasks.TRANSCRIBE_JOB_TIMEOUT > settings.LOCALISATION_TIMEOUT_SEC


def _patch_durable_admission(monkeypatch):
    """AICP-365: the hand-off admits through app.localisation_jobs (covered on a
    real database in tests/test_localisation_jobs.py). With the fake sessions
    here, stand in for it at its boundary: same status writes, same reserve,
    same 429 mapping. Returns the doorbells rung."""
    from types import SimpleNamespace

    from fastapi import HTTPException

    import app.localisation_jobs as lj
    import app.spend as spend_mod

    def _admit(session, project, *, origin):
        project.transcript_status, project.transcript_error = "pending", None
        try:
            spend_mod.reserve(session, "transcribe:x", 20.0, kind="transcription")
        except spend_mod.BudgetExceeded as exc:
            raise HTTPException(status_code=429, detail=exc.message) from exc
        return SimpleNamespace(id="job-1", origin=origin, project=project)

    rang = []
    monkeypatch.setattr(lj, "admit_transcription", _admit)
    monkeypatch.setattr(lj, "enqueue", rang.append)
    return rang


def test_run_analyze_project_enqueues_transcription_for_localisation(monkeypatch):
    """A localisation project is transcribed automatically after analysis."""
    _patch_queue(monkeypatch)
    project = _FakeProject("localisation")
    commits = _patch_project_lookup(monkeypatch, project)
    rang = _patch_durable_admission(monkeypatch)

    order = []
    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(
        pipeline_v2, "analyze_project", lambda pid: order.append(("analyze", pid))
    )

    tasks.run_analyze_project("proj-loc")

    # AICP-365: a durable job admitted for the "analysis" origin, then its doorbell.
    assert [j.origin for j in rang] == ["analysis"]
    # Enqueued AFTER analysis returned, never from inside it.
    assert order == [("analyze", "proj-loc")]
    # Status is pending before the job is queued, and the stale error is cleared.
    assert project.transcript_status == "pending"
    assert project.transcript_error is None
    assert commits


def test_run_analyze_project_skips_transcription_for_other_types(monkeypatch):
    fake_queue = _patch_queue(monkeypatch)
    project = _FakeProject("face_swap")
    _patch_project_lookup(monkeypatch, project)

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "analyze_project", lambda pid: None)

    tasks.run_analyze_project("proj-face")

    assert not fake_queue.enqueue.called
    assert project.transcript_status is None


def test_failed_analysis_never_reaches_transcription(monkeypatch):
    """Nothing to transcribe if the source could not even be segmented."""
    fake_queue = _patch_queue(monkeypatch)
    _patch_project_lookup(monkeypatch, _FakeProject("localisation"))

    import app.pipeline_v2 as pipeline_v2

    def _boom(_pid):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(pipeline_v2, "analyze_project", _boom)

    with pytest.raises(RuntimeError, match="probe failed"):
        tasks.run_analyze_project("proj-loc")

    assert not fake_queue.enqueue.called


def test_transcription_enqueue_failure_does_not_fail_analysis(monkeypatch):
    """§8: a transcription outage must never break analysis. The enqueue is a
    side channel, so its own failures are swallowed."""
    _patch_queue(monkeypatch)

    import app.db as db_mod
    import app.pipeline_v2 as pipeline_v2

    def _dead_session():
        raise RuntimeError("database is down")

    monkeypatch.setattr(db_mod, "get_session", _dead_session)
    monkeypatch.setattr(pipeline_v2, "analyze_project", lambda pid: None)

    tasks.run_analyze_project("proj-loc")  # must not raise


def test_missing_project_after_analysis_is_not_fatal(monkeypatch):
    fake_queue = _patch_queue(monkeypatch)
    _patch_project_lookup(monkeypatch, None)

    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(pipeline_v2, "analyze_project", lambda pid: None)

    tasks.run_analyze_project("proj-gone")

    assert not fake_queue.enqueue.called


def test_run_transcribe_project_delegates_to_the_pipeline(monkeypatch):
    """No per-project lock: like merge/tail-edit, a duplicate wastes one LLM
    call instead of double-submitting to an AI backend."""
    calls = []
    import app.pipeline_v2 as pipeline_v2
    monkeypatch.setattr(
        pipeline_v2, "transcribe_project", lambda pid: calls.append(pid)
    )
    fake_redis = _FakeRedis()
    monkeypatch.setattr(tasks, "_redis_conn", fake_redis)

    tasks.run_transcribe_project("proj-loc")

    assert calls == ["proj-loc"]
    assert fake_redis.set_calls == []  # no lock acquired


# ---------------------------------------------------------------------------
# Sheet exports: retry the transient Sheets failures, undo the stamp on the
# permanent ones
# ---------------------------------------------------------------------------
# Regression (2026-08-18): a single Sheets 503 on the header read killed run
# 59f29a5c's Final Creatives job. The endpoint stamps
# ``final_creatives_exported_at`` BEFORE enqueuing, so the UI kept showing
# "original sent" for a row the sheet never received, and nothing retried or
# corrected it.

_EXPORT_ENQUEUERS = [
    ("enqueue_export_run_to_sheet", "app.tasks.run_export_run_to_sheet",
     "run-1", "Run", "final_creatives_exported_at"),
    ("enqueue_export_overlay_to_sheet", "app.tasks.run_export_overlay_to_sheet",
     "ov-1", "OverlayJob", "final_creatives_exported_at"),
    ("enqueue_export_tail_edit_to_sheet", "app.tasks.run_export_tail_edit_to_sheet",
     "edit-1", "RunTailEdit", "final_creatives_exported_at"),
    ("enqueue_export_caption_to_sheet", "app.tasks.run_export_caption_to_sheet",
     "cap-1", "RunCaption", "final_creatives_exported_at"),
]


@pytest.mark.parametrize(
    "helper_name,task_path,entity_id,model,stamp", _EXPORT_ENQUEUERS
)
def test_export_enqueue_sets_retry_and_stamp_repair(
    monkeypatch, helper_name, task_path, entity_id, model, stamp
):
    """Every export job carries a Retry policy and the stamp-clearing callback."""
    fake_queue = _patch_queue(monkeypatch)

    getattr(tasks, helper_name)(entity_id)

    args, kwargs = fake_queue.enqueue.call_args
    assert args[0] == task_path
    assert args[1] == entity_id
    assert kwargs["job_timeout"] == tasks._EXPORT_JOB_TIMEOUT
    retry = kwargs["retry"]
    assert retry.max == tasks._EXPORT_RETRY_MAX
    # Consumed in order by RQ, so the gap widens instead of hammering Sheets.
    assert list(retry.intervals) == tasks._EXPORT_RETRY_INTERVALS
    # rq.Callback, not a bare function: passing a function is deprecated in
    # RQ 1.15 and warns on every enqueue.
    assert kwargs["on_failure"].func is tasks.clear_export_stamp_on_final_failure
    # The field to undo travels in meta, not derived from the task name.
    assert kwargs["meta"] == {"export_model": model, "export_stamp": stamp}


def test_export_retry_intervals_match_max():
    """RQ indexes intervals by ``len(intervals) - retries_left``; a short list
    would silently reuse the first gap for the later retries."""
    assert len(tasks._EXPORT_RETRY_INTERVALS) == tasks._EXPORT_RETRY_MAX


class _FakeStampJob:
    """Stand-in for the rq Job handed to an on_failure callback."""

    def __init__(self, *, retries_left, meta, args=("run-1",)):
        self.id = "rq-fail-1"
        self.func_name = "app.tasks.run_export_run_to_sheet"
        self.retries_left = retries_left
        self.meta = meta
        self.args = args


class _FakeStampSession:
    """SessionLocal stand-in that hands back one entity and records commits."""

    def __init__(self, obj):
        self._obj = obj
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, model, oid):
        return self._obj

    def commit(self):
        self.commits += 1


def _fail_callback(monkeypatch, job, entity):
    import app.db as db_mod

    session = _FakeStampSession(entity)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: session)
    tasks.clear_export_stamp_on_final_failure(
        job, None, RuntimeError, RuntimeError("503 Service Unavailable"), None
    )
    return session


def test_stamp_survives_while_retries_remain(monkeypatch):
    """A retryable failure must NOT clear the stamp — the job will run again."""
    from datetime import datetime
    from types import SimpleNamespace

    entity = SimpleNamespace(final_creatives_exported_at=datetime(2026, 8, 18, 13, 22))
    job = _FakeStampJob(
        retries_left=2,
        meta={"export_model": "Run", "export_stamp": "final_creatives_exported_at"},
    )

    session = _fail_callback(monkeypatch, job, entity)

    assert entity.final_creatives_exported_at is not None
    assert session.commits == 0


def test_stamp_cleared_when_retries_are_exhausted(monkeypatch):
    """The last failed attempt puts the Send button back."""
    from datetime import datetime
    from types import SimpleNamespace

    entity = SimpleNamespace(final_creatives_exported_at=datetime(2026, 8, 18, 13, 22))
    job = _FakeStampJob(
        retries_left=0,
        meta={"export_model": "Run", "export_stamp": "final_creatives_exported_at"},
    )

    session = _fail_callback(monkeypatch, job, entity)

    assert entity.final_creatives_exported_at is None
    assert session.commits == 1


def test_stamp_cleared_for_a_status_string_field(monkeypatch):
    """The repair works generically: a stamp can be a status string, not just
    a timestamp — the callback only ever getattr/setattr's ``export_stamp``."""
    from types import SimpleNamespace

    entity = SimpleNamespace(some_export_status="queued")
    job = _FakeStampJob(
        retries_left=None,  # no Retry configured at all -> this was the last try
        meta={"export_model": "Run", "export_stamp": "some_export_status"},
    )

    session = _fail_callback(monkeypatch, job, entity)

    assert entity.some_export_status is None
    assert session.commits == 1


def test_stamp_repair_is_a_noop_when_already_cleared(monkeypatch):
    """A second failed job for the same entity must not commit again."""
    from types import SimpleNamespace

    entity = SimpleNamespace(final_creatives_exported_at=None)
    job = _FakeStampJob(
        retries_left=0,
        meta={"export_model": "Run", "export_stamp": "final_creatives_exported_at"},
    )

    session = _fail_callback(monkeypatch, job, entity)

    assert session.commits == 0


def test_stamp_repair_is_a_noop_when_the_entity_is_gone(monkeypatch):
    """Deleted entity: nothing to clear, and no crash."""
    job = _FakeStampJob(
        retries_left=0,
        meta={"export_model": "Run", "export_stamp": "final_creatives_exported_at"},
    )

    session = _fail_callback(monkeypatch, job, None)

    assert session.commits == 0


def test_stamp_repair_ignores_a_job_without_meta(monkeypatch):
    """Jobs enqueued before this change carry no meta — leave them alone."""
    from types import SimpleNamespace

    entity = SimpleNamespace(final_creatives_exported_at="stamped")
    job = _FakeStampJob(retries_left=0, meta={})

    session = _fail_callback(monkeypatch, job, entity)

    assert entity.final_creatives_exported_at == "stamped"
    assert session.commits == 0


def test_stamp_repair_never_raises(monkeypatch):
    """RQ re-raises a failing callback and would bury the real Sheets error."""
    import app.db as db_mod

    def _boom():
        raise RuntimeError("database is down")

    monkeypatch.setattr(db_mod, "SessionLocal", _boom)
    job = _FakeStampJob(
        retries_left=0,
        meta={"export_model": "Run", "export_stamp": "final_creatives_exported_at"},
    )

    tasks.clear_export_stamp_on_final_failure(
        job, None, RuntimeError, RuntimeError("503"), None
    )  # must not raise


class _HeaderlessSheets(_FakeFinalCreativesSheets):
    """The Final Creatives tab answers with no header row at all."""

    def get_header(self, sid, tab):
        return []


def test_export_raises_when_the_tab_has_no_header(monkeypatch):
    """A missing header used to return quietly: the job "succeeded", no row was
    written, and the stamp stayed. It has to fail so RQ retries and repairs."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id="abcd1234-ef56-7890-1234-567890abcdef", name="Q-1",
        model="seedance-2-5", result_gdrive_file_id="FILE1",
        project=project, language=None, short_id="abcd1234",
    )
    fake = _HeaderlessSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(run))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    with pytest.raises(sc.SheetsError):
        tasks.run_export_run_to_sheet(run.id)

    assert fake.appended == []


# ---------------------------------------------------------------------------
# The one gap the retry/callback pair cannot close: the enqueue itself failing
# ---------------------------------------------------------------------------
# The endpoints stamp and commit BEFORE queuing. If the enqueue throws (Redis
# down) there is no job in existence to carry clear_export_stamp_on_final_failure,
# so the stamp has to be undone right here or the UI claims a sheet row forever.


class _RollbackEntity:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class _RollbackDb:
    def __init__(self, fail_commit=False):
        self.commits = 0
        self.fail_commit = fail_commit

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("database is down too")
        self.commits += 1


def test_enqueue_export_or_rollback_passes_through_on_success():
    """The happy path must not touch the stamp or the session."""
    from datetime import datetime

    entity = _RollbackEntity(final_creatives_exported_at=datetime(2026, 8, 18))
    db = _RollbackDb()
    called = []

    tasks.enqueue_export_or_rollback(
        called.append, "run-1", entity, "final_creatives_exported_at", db
    )

    assert called == ["run-1"]
    assert entity.final_creatives_exported_at is not None
    assert db.commits == 0


def test_enqueue_export_or_rollback_undoes_the_stamp():
    """Redis down: clear the stamp, commit it, and raise ExportEnqueueError."""
    from datetime import datetime

    entity = _RollbackEntity(final_creatives_exported_at=datetime(2026, 8, 18))
    db = _RollbackDb()

    def _boom(_id):
        raise ConnectionError("Error 111 connecting to redis:6379")

    with pytest.raises(tasks.ExportEnqueueError) as excinfo:
        tasks.enqueue_export_or_rollback(
            _boom, "run-1", entity, "final_creatives_exported_at", db
        )

    assert entity.final_creatives_exported_at is None
    assert db.commits == 1
    # The message reaches the operator verbatim (the button JS renders `detail`).
    assert "try again" in str(excinfo.value).lower()
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_enqueue_export_or_rollback_undoes_a_status_string_field():
    """The rollback works generically: a stamp can be a status string, not
    just a timestamp — it only ever getattr/setattr's the given field."""
    entity = _RollbackEntity(some_export_status="queued")
    db = _RollbackDb()

    def _boom(_id):
        raise ConnectionError("redis is gone")

    with pytest.raises(tasks.ExportEnqueueError):
        tasks.enqueue_export_or_rollback(
            _boom, "ov-1", entity, "some_export_status", db
        )

    assert entity.some_export_status is None
    assert db.commits == 1


def test_enqueue_export_or_rollback_reraises_when_the_rollback_fails():
    """If the rollback can't commit, the DB error surfaces — deliberately NOT
    ExportEnqueueError, since its 'try again' message would be a second lie
    while the stamp is still stuck set. The enqueue failure is chained on."""
    from datetime import datetime

    entity = _RollbackEntity(final_creatives_exported_at=datetime(2026, 8, 18))
    db = _RollbackDb(fail_commit=True)

    def _boom(_id):
        raise ConnectionError("redis is gone")

    with pytest.raises(RuntimeError, match="database is down too") as excinfo:
        tasks.enqueue_export_or_rollback(
            _boom, "run-1", entity, "final_creatives_exported_at", db
        )

    assert not isinstance(excinfo.value, tasks.ExportEnqueueError)
    # The enqueue failure that started it all is still reachable for the logs.
    assert isinstance(excinfo.value.__context__, ConnectionError)
    assert db.commits == 0


# ---------------------------------------------------------------------------
# Italian (it) — the exact Final Creatives Job IDs, per creative kind
# ---------------------------------------------------------------------------
#
# Synthetic uuids, deliberately in the shape the sheet actually receives, so
# these strings can be handed to the downstream parser as a fixture.
# ai-fb-audit has parsed all of them since PR #178 (verified at 5cea1a0 on
# 2026-10-06; its trailing rule is -(es|pt|de|ja|sv|ar|fr|it)$). The Polish
# equivalent below is NOT parsed yet — see AICP-181 and
# wiki/components/run-language.md.

_IT_RUN_ID = "1736426f-aaaa-bbbb-cccc-ddddeeeeffff"
_IT_TAIL_ID = "9c2a41b7-aaaa-bbbb-cccc-ddddeeeeffff"
_IT_CAPTION_ID = "b2c3d4e5-aaaa-bbbb-cccc-ddddeeeeffff"
_IT_OVERLAY_ID = "a1b2c3d4-aaaa-bbbb-cccc-ddddeeeeffff"


def test_export_italian_run_job_id(monkeypatch):
    """Run 1736426f… in Italian -> '1736426f-it'."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc
    from app import languages

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id=_IT_RUN_ID, name="Q-00873", model="seedance-2-5",
        result_gdrive_file_id="FILE1", project=project, language="it",
        short_id=languages.job_id_for(_IT_RUN_ID, "it"),
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(run))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_run_to_sheet(run.id)

    assert _job_id_cell(fake) == "1736426f-it"
    pipeline = fake.appended[0][_FakeFinalCreativesSheets.HEADER.index("Pipeline Type")]
    assert pipeline == "re-skin-v2"


def test_export_polish_run_job_id(monkeypatch):
    """Run 1736426f… in Polish -> '1736426f-pl' (AICP-180).

    ai-fb-audit parses it as ("PL", "1736426f") since AICP-181 (2026-10-08).
    """
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc
    from app import languages

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id=_IT_RUN_ID, name="Q-00873", model="seedance-2-5",
        result_gdrive_file_id="FILE1", project=project, language="pl",
        short_id=languages.job_id_for(_IT_RUN_ID, "pl"),
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(run))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_run_to_sheet(run.id)

    assert _job_id_cell(fake) == "1736426f-pl"



def test_export_french_run_job_id(monkeypatch):
    """Run 1736426f… in French -> '1736426f-fr', which ai-fb-audit parses as
    ("FR", "1736426f") (verified at c680e61, 2026-10-20)."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc
    from app import languages

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id=_IT_RUN_ID, name="Q-00873", model="seedance-2-5",
        result_gdrive_file_id="FILE1", project=project, language="fr",
        short_id=languages.job_id_for(_IT_RUN_ID, "fr"),
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(run))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_run_to_sheet(run.id)

    assert _job_id_cell(fake) == "1736426f-fr"

def test_export_italian_tail_edit_job_id(monkeypatch):
    """Tail edit 9c2a41b7… under an Italian run -> '9c2a41b7-it'.

    Its OWN prefix (so it doesn't collide with the run's row) + the PARENT
    run's language (so it doesn't rename as English).
    """
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id=_IT_RUN_ID, name="Q-00873", model="seedance-fast",
        result_gdrive_file_id="RUNFILE", project=project, language="it",
    )
    edit = SimpleNamespace(
        id=_IT_TAIL_ID, run_id=_IT_RUN_ID, run=run, gdrive_file_id="EDITFILE",
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(edit))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_tail_edit_to_sheet(edit.id)

    assert _job_id_cell(fake) == "9c2a41b7-it"


def test_export_italian_caption_job_id(monkeypatch):
    """Caption b2c3d4e5… inherits the PARENT run's language -> 'b2c3d4e5-it'."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id=_IT_RUN_ID, name="Q-00873", model="seedance-2-5",
        result_gdrive_file_id="RUNFILE", project=project, language="it",
    )
    caption = SimpleNamespace(
        id=_IT_CAPTION_ID, run_id=_IT_RUN_ID, run=run, gdrive_file_id="CAPFILE",
    )
    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(caption))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_caption_to_sheet(caption.id)

    assert _job_id_cell(fake) == "b2c3d4e5-it"
    pipeline = fake.appended[0][_FakeFinalCreativesSheets.HEADER.index("Pipeline Type")]
    assert pipeline == "re-skin-captions"


def _italian_overlay_fakes(monkeypatch, *, sibling_count, source_run=True):
    """As _overlay_export_fakes, but with the task's synthetic Italian ids.

    *source_run=False* is the upload-sourced overlay: no parent to inherit a
    language from, so it keeps its own unsuffixed prefix.
    """
    from types import SimpleNamespace

    import app.db as db_mod
    import app.models as models_mod
    import app.overlay_runs as overlay_runs
    import app.sheets_client as sc
    from app import languages

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id=_IT_RUN_ID, name="Q-00873", model="seedance-2-5",
        result_gdrive_file_id="RUNFILE", project=project, language="it",
        short_id=languages.job_id_for(_IT_RUN_ID, "it"),
    )
    # An upload-sourced overlay has ALL THREE source ids null — the export
    # walks source_run_id -> source_tail_edit_id -> source_caption_id, so a
    # fake missing the third attribute AttributeErrors instead of falling
    # through to the no-parent branch.
    overlay = SimpleNamespace(
        id=_IT_OVERLAY_ID, gdrive_file_id="OVLFILE",
        source_run_id=_IT_RUN_ID if source_run else None,
        source_tail_edit_id=None,
        source_caption_id=None,
    )

    class _ByModelSession(_FakeSession):
        def get(self, model, oid):
            if model is models_mod.Run:
                return run if source_run else None
            return overlay

    fake = _FakeFinalCreativesSheets()
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _ByModelSession(overlay))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)
    monkeypatch.setattr(
        overlay_runs, "count_overlay_jobs_for_run", lambda run_id, db: sibling_count
    )
    return overlay, run, fake


def test_export_italian_single_overlay_uses_the_parent_run_job_id(monkeypatch):
    """A run's ONLY overlay exports under the RUN's id -> '1736426f-it'."""
    overlay, run, fake = _italian_overlay_fakes(monkeypatch, sibling_count=1)

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert _job_id_cell(fake) == "1736426f-it" == run.short_id
    pipeline = fake.appended[0][_FakeFinalCreativesSheets.HEADER.index("Pipeline Type")]
    assert pipeline == "re-skin-overlay"


def test_export_italian_overlay_combines_ids_when_run_has_several(monkeypatch):
    """Several overlays -> '{run8}_{overlay8}-it' = '1736426f_a1b2c3d4-it'."""
    overlay, run, fake = _italian_overlay_fakes(monkeypatch, sibling_count=2)

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert _job_id_cell(fake) == "1736426f_a1b2c3d4-it"


def test_export_upload_sourced_overlay_stays_unsuffixed(monkeypatch):
    """No parent run -> its own prefix, no language affix, exactly as before."""
    overlay, _run, fake = _italian_overlay_fakes(
        monkeypatch, sibling_count=1, source_run=False
    )

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert _job_id_cell(fake) == "a1b2c3d4"


def test_italian_overlay_dedupe_key_is_still_creative_link(monkeypatch):
    """Adding a suffix must not change WHICH key blocks a duplicate append."""
    overlay, _run, fake = _italian_overlay_fakes(monkeypatch, sibling_count=1)
    creative_link = "https://drive.google.com/file/d/OVLFILE/view"
    creative_idx = _FakeFinalCreativesSheets.HEADER.index("Creative Link")

    def _column_values(sid, tab, column_index, *, last_n=None):
        return [creative_link] if column_index == creative_idx else []

    fake.column_values = _column_values

    tasks.run_export_overlay_to_sheet(overlay.id)

    assert fake.appended == []


def test_italian_run_dedupe_matches_the_suffixed_job_id(monkeypatch):
    """A re-run of the export sees '1736426f-it' already present and skips."""
    from types import SimpleNamespace

    import app.db as db_mod
    import app.sheets_client as sc
    from app import languages

    project = SimpleNamespace(id="proj-1", name="Gardener", source_ref="in")
    run = SimpleNamespace(
        id=_IT_RUN_ID, name="Q-00873", model="seedance-2-5",
        result_gdrive_file_id="FILE1", project=project, language="it",
        short_id=languages.job_id_for(_IT_RUN_ID, "it"),
    )
    fake = _FakeFinalCreativesSheets()
    job_idx = _FakeFinalCreativesSheets.HEADER.index("Job ID")

    def _column_values(sid, tab, column_index, *, last_n=None):
        return ["1736426f-it"] if column_index == job_idx else []

    fake.column_values = _column_values
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: _FakeSession(run))
    monkeypatch.setattr(sc, "SheetsClient", lambda *a, **k: fake)

    tasks.run_export_run_to_sheet(run.id)

    assert fake.appended == []


def test_auto_transcription_over_budget_is_skipped_with_a_clear_reason(monkeypatch):
    """AICP-155: the post-analysis transcription is a new paid call. Refused by
    the daily budget, the project stays ready (analysis succeeded), nothing is
    queued, and the transcript panel says why."""
    from datetime import date

    import app.pipeline_v2 as pipeline_v2
    import app.spend as spend_mod

    fake_queue = _patch_queue(monkeypatch)
    project = _FakeProject("localisation")
    commits = _patch_project_lookup(monkeypatch, project)
    rang = _patch_durable_admission(monkeypatch)
    monkeypatch.setattr(pipeline_v2, "analyze_project", lambda pid: None)

    def _refuse(*a, **k):
        raise spend_mod.BudgetExceeded(
            day=date(2026, 10, 16), charged=50_000, needed=20, cap=50_000,
            kind="transcription",
        )

    monkeypatch.setattr(spend_mod, "reserve", _refuse)

    tasks.run_analyze_project("proj-loc")

    fake_queue.enqueue.assert_not_called()
    assert rang == []
    assert project.transcript_status == "failed"
    assert "Daily kie.ai budget reached" in project.transcript_error
    assert commits


# ---------------------------------------------------------------------------
# Disk cleanup tick
# ---------------------------------------------------------------------------


def test_disk_cleanup_tick_boot_starts_one_chain(monkeypatch):
    fake_queue = _patch_queue(monkeypatch)
    redis = MagicMock()
    redis.set.side_effect = [True, None]  # first boot wins the lease, second loses
    monkeypatch.setattr(tasks, "_get_redis", lambda: redis)

    tasks.enqueue_disk_cleanup_tick()
    tasks.enqueue_disk_cleanup_tick()

    assert fake_queue.enqueue_in.call_count == 1
    args, kwargs = fake_queue.enqueue_in.call_args
    assert args[1] == "app.tasks.run_disk_cleanup_tick"
    assert redis.set.call_args_list[0].kwargs["nx"] is True


def test_disk_cleanup_tick_reschedules_even_when_sweep_raises(monkeypatch):
    import app.disk_cleanup as disk_cleanup

    calls = []
    monkeypatch.setattr(
        tasks, "enqueue_disk_cleanup_tick",
        lambda **kw: calls.append(kw),
    )

    def _boom(session):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(disk_cleanup, "sweep", _boom)
    import app.db as db
    from contextlib import contextmanager

    @contextmanager
    def _fake_session():
        yield MagicMock()

    monkeypatch.setattr(db, "get_session", _fake_session)

    tasks.run_disk_cleanup_tick()

    assert calls == [{"is_continuation": True}]
