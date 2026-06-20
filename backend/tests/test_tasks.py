"""Tests for app/tasks.py enqueue helpers — ensure generous RQ job_timeouts.

Regression: RQ's default job_timeout (180s) killed long analyze/process jobs
mid-flight ("Task exceeded maximum timeout value (180 seconds)") while the
Seedance task kept running on kie.ai.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
