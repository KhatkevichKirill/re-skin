"""Tests for app/tasks.py enqueue helpers — ensure generous RQ job_timeouts.

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

import app.tasks as tasks


def _patch_queue(monkeypatch):
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = MagicMock(id="rq-123")
    monkeypatch.setattr(tasks, "_get_queue", lambda: fake_queue)
    return fake_queue


def test_enqueue_analyze_sets_generous_timeout(monkeypatch):
    q = _patch_queue(monkeypatch)
    tasks.enqueue_analyze("job-abc")
    args, kwargs = q.enqueue.call_args
    assert args[0] == "app.tasks.run_analyze"
    assert args[1] == "job-abc"
    assert kwargs["job_timeout"] == tasks.ANALYZE_JOB_TIMEOUT
    assert tasks.ANALYZE_JOB_TIMEOUT >= 600  # well above RQ's 180s default


def test_enqueue_process_sets_generous_timeout(monkeypatch):
    q = _patch_queue(monkeypatch)
    tasks.enqueue_process("job-xyz")
    args, kwargs = q.enqueue.call_args
    assert args[0] == "app.tasks.run_process"
    assert args[1] == "job-xyz"
    assert kwargs["job_timeout"] == tasks.PROCESS_JOB_TIMEOUT
    # process runs several Seedance jobs sequentially — needs a large budget
    assert tasks.PROCESS_JOB_TIMEOUT >= 3600


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


def test_run_process_run_skips_duplicate_when_lock_held(monkeypatch):
    fake_redis = _FakeRedis(acquired=False)
    process_calls = []

    monkeypatch.setattr(tasks.Redis, "from_url", lambda _url: fake_redis)

    import app.pipeline_v2 as pipeline_v2

    monkeypatch.setattr(pipeline_v2, "process_run", lambda *a, **k: process_calls.append((a, k)))

    tasks.run_process_run("run-locked")

    assert process_calls == []
    assert fake_redis.eval_calls == []
    assert fake_redis.set_calls[0][1]["nx"] is True
    assert fake_redis.set_calls[0][1]["ex"] == tasks._RUN_LOCK_TTL_SEC


def test_run_process_run_releases_lock_atomically(monkeypatch):
    fake_redis = _FakeRedis(acquired=True)
    process_calls = []

    monkeypatch.setattr(tasks.Redis, "from_url", lambda _url: fake_redis)

    import app.kie_client as kie_client
    import app.gdrive_client as gdrive_client
    import app.pipeline_v2 as pipeline_v2

    monkeypatch.setattr(kie_client, "KieClient", lambda: object())
    monkeypatch.setattr(gdrive_client, "GDriveClient", lambda: object())
    monkeypatch.setattr(pipeline_v2, "process_run", lambda *a, **k: process_calls.append((a, k)))

    tasks.run_process_run("run-ok")

    assert len(process_calls) == 1
    assert len(fake_redis.eval_calls) == 1
    script, num_keys, lock_key, token = fake_redis.eval_calls[0]
    assert "redis.call(\"GET\"" in script
    assert num_keys == 1
    assert lock_key == "reskin:run:lock:run-ok"
    assert token


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
