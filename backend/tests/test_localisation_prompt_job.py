"""Unit tests for app.localisation_prompt_job (AICP-338).

The API lifecycle (202, dedupe, polling, stale refusal) is pinned in
test_api_v2.TestLocalisationPromptJob against an inline queue. This file pins
the module's own pieces: the timeouts, the per-job translate-timeout override,
the job envelope, and — when fakeredis is installed — the real Redis/RQ
admission and poll logic. No network, no kie.ai.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app import localisation, localisation_prompt_job as lpj
from app.config import settings


def test_job_timeout_covers_four_transport_attempts(monkeypatch):
    monkeypatch.setattr(settings, "LOCALISATION_TRANSLATE_JOB_CALL_TIMEOUT_SEC", 420.0)
    monkeypatch.setattr(settings, "LOCALISATION_PROMPT_JOB_TIMEOUT_SEC", 60)
    b = lpj.translate_budgets()
    assert lpj.job_timeout_sec() >= 2 * b["retry_budget_sec"] + 4 * 420
    assert lpj.job_timeout_sec() >= b["shorten_admit_sec"] + 4 * b["shorten_timeout_sec"]
    monkeypatch.setattr(settings, "LOCALISATION_PROMPT_JOB_TIMEOUT_SEC", 9999)
    assert lpj.job_timeout_sec() == 9999


def test_call_timeout_never_drops_below_the_request_timeout(monkeypatch):
    monkeypatch.setattr(settings, "LOCALISATION_TRANSLATE_JOB_CALL_TIMEOUT_SEC", 10.0)
    monkeypatch.setattr(settings, "LOCALISATION_TRANSLATE_TIMEOUT_SEC", 90.0)
    assert lpj.call_timeout_sec() == 90.0


def _snapshot():
    return {"project_id": "p1", "target_language": "es"}


def test_run_raises_the_translate_timeout_only_for_the_job(monkeypatch):
    """localisation.translate_lines reads the setting at call time; the job
    swaps it for its own duration and restores it afterwards."""
    monkeypatch.setattr(settings, "LOCALISATION_TRANSLATE_TIMEOUT_SEC", 90.0)
    monkeypatch.setattr(settings, "LOCALISATION_TRANSLATE_JOB_CALL_TIMEOUT_SEC", 420.0)
    seen = {}

    def _compute(snapshot, translate_budgets=None):
        seen["timeout"] = settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC
        seen["budgets"] = translate_budgets
        from app.schemas_v2 import LocalisationPromptResponse

        return LocalisationPromptResponse(target_language="es", hook_sec=10.0, prompt="P")

    import app.api_v2 as api_v2_module

    monkeypatch.setattr(api_v2_module, "_compute_localisation_prompt", _compute)
    out = lpj.run(_snapshot())
    assert seen["timeout"] == 420.0
    assert seen["budgets"] == lpj.translate_budgets()
    assert settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC == 90.0
    assert out["ok"] is True
    assert out["result"]["prompt"] == "P"


def test_run_turns_a_model_failure_into_a_502_envelope(monkeypatch):
    import app.api_v2 as api_v2_module

    monkeypatch.setattr(settings, "LOCALISATION_TRANSLATE_TIMEOUT_SEC", 90.0)

    def _boom(snapshot, **_kw):
        raise localisation.LocalisationError("translate: model down")

    monkeypatch.setattr(api_v2_module, "_compute_localisation_prompt", _boom)
    assert lpj.run(_snapshot()) == {
        "ok": False, "status_code": 502, "detail": "translate: model down",
    }
    assert settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC == 90.0


def test_an_unexpected_error_restores_the_timeout_and_propagates(monkeypatch):
    import app.api_v2 as api_v2_module

    monkeypatch.setattr(settings, "LOCALISATION_TRANSLATE_TIMEOUT_SEC", 90.0)

    def _crash(snapshot, **_kw):
        raise RuntimeError("bug")

    monkeypatch.setattr(api_v2_module, "_compute_localisation_prompt", _crash)
    with pytest.raises(RuntimeError):
        lpj.run(_snapshot())
    assert settings.LOCALISATION_TRANSLATE_TIMEOUT_SEC == 90.0


def _project(**kw):
    base = dict(
        transcript={"lines": [{"id": 1, "text": "hi"}]}, transcript_status="ready",
        hook_sec=None, source_local_path="/x.mp4", duration_sec=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _seg(i, a, b):
    return SimpleNamespace(id=f"s{i}", start_sec=a, end_sec=b)


def test_fingerprint_moves_with_transcript_segments_and_hook():
    base = lpj.project_fingerprint(_project(), [_seg(0, 0, 10)])
    assert base == lpj.project_fingerprint(_project(), [_seg(0, 0.0, 10.0)])
    assert base != lpj.project_fingerprint(
        _project(transcript={"lines": [{"id": 1, "text": "ho"}]}), [_seg(0, 0, 10)]
    )
    assert base != lpj.project_fingerprint(_project(), [_seg(0, 0, 9)])
    assert base != lpj.project_fingerprint(_project(), [_seg(1, 0, 10)])
    assert base != lpj.project_fingerprint(_project(hook_sec=12.0), [_seg(0, 0, 10)])


def test_dedupe_key_separates_every_request_parameter():
    keys = {
        lpj.dedupe_key("p", "es", False, None, "f"),
        lpj.dedupe_key("p", "pt", False, None, "f"),
        lpj.dedupe_key("p", "es", True, None, "f"),
        lpj.dedupe_key("p", "es", False, 30.0, "f"),
        lpj.dedupe_key("p", "es", False, None, "g"),
        lpj.dedupe_key("q", "es", False, None, "f"),
    }
    assert len(keys) == 6
    # Extra optional request fields (e.g. a future per-call model) separate too,
    # and an empty dict is the same request as none.
    assert lpj.dedupe_key("p", "es", False, None, "f", extra={"model": "m"}) not in keys
    assert lpj.dedupe_key("p", "es", False, None, "f", extra={}) in keys
    assert lpj.dedupe_key("p", "es", False, None, "f").startswith("p:")


# ---------------------------------------------------------------------------
# Real Redis/RQ logic — needs fakeredis (not in requirements-dev; install it
# ad hoc to run these: pip install "fakeredis[lua]").
# ---------------------------------------------------------------------------

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture()
def conn():
    return fakeredis.FakeStrictRedis()


def _enqueue_into(conn, meta):
    from rq import Queue

    def _enqueue(job_id):
        Queue("localise", connection=conn).enqueue(
            "app.tasks.run_localisation_prompt", {"project_id": meta["project_id"]},
            job_id=job_id, meta=meta, job_timeout=60, ttl=600, result_ttl=600,
        )

    return _enqueue


def test_admit_dedupes_an_active_job_and_reserves_once(conn):
    reserves = []
    meta = {"project_id": "p1", "fingerprint": "f"}
    a, created_a = lpj.admit(
        conn, key="k", reserve=lambda: reserves.append(1), enqueue=_enqueue_into(conn, meta)
    )
    b, created_b = lpj.admit(
        conn, key="k", reserve=lambda: reserves.append(1), enqueue=_enqueue_into(conn, meta)
    )
    assert (created_a, created_b) == (True, False)
    assert a == b
    assert reserves == [1]
    assert lpj.job_state(conn, a, "p1")["status"] == "queued"
    assert lpj.job_state(conn, a, "p2") is None


def test_admit_starts_a_new_job_once_the_old_one_finished(conn):
    from rq.job import Job

    meta = {"project_id": "p1", "fingerprint": "f"}
    a, _ = lpj.admit(conn, key="k", reserve=lambda: None, enqueue=_enqueue_into(conn, meta))
    job = Job.fetch(a, connection=conn)
    job.set_status("finished")
    b, created = lpj.admit(conn, key="k", reserve=lambda: None, enqueue=_enqueue_into(conn, meta))
    assert created is True and b != a


def test_a_reserve_refusal_enqueues_nothing(conn):
    class _Refused(Exception):
        pass

    def _reserve():
        raise _Refused()

    meta = {"project_id": "p1", "fingerprint": "f"}
    with pytest.raises(_Refused):
        lpj.admit(conn, key="k", reserve=_reserve, enqueue=_enqueue_into(conn, meta))
    assert conn.get(lpj.POINTER_PREFIX + "k") is None
    # The admit lock was released: a retry is not refused with a 409.
    lpj.admit(conn, key="k", reserve=lambda: None, enqueue=_enqueue_into(conn, meta))


def test_started_job_with_a_dead_heartbeat_is_interrupted(conn):
    from rq.job import Job

    meta = {"project_id": "p1", "fingerprint": "f"}
    a, _ = lpj.admit(conn, key="k", reserve=lambda: None, enqueue=_enqueue_into(conn, meta))
    job = Job.fetch(a, connection=conn)
    job.set_status("started")
    job.heartbeat(datetime.utcnow(), 60)
    assert lpj.job_state(conn, a, "p1")["status"] == "running"

    job.heartbeat(datetime.utcnow() - timedelta(seconds=lpj._HEARTBEAT_STALE_SEC + 5), 60)
    state = lpj.job_state(conn, a, "p1")
    assert state["status"] == "failed"
    assert state["error"]["status_code"] == 503
    # ...and an interrupted job no longer blocks a fresh click.
    b, created = lpj.admit(conn, key="k", reserve=lambda: None, enqueue=_enqueue_into(conn, meta))
    assert created is True and b != a


def test_finished_job_envelopes_map_to_done_and_failed(conn):
    from rq import Queue, SimpleWorker

    meta = {"project_id": "p1", "fingerprint": "f"}
    q = Queue("localise", connection=conn)
    ok = q.enqueue(
        "tests.test_localisation_prompt_job._ok_job", job_id="localise-prompt-ok", meta=meta,
    )
    bad = q.enqueue(
        "tests.test_localisation_prompt_job._bad_job", job_id="localise-prompt-bad", meta=meta,
    )
    crash = q.enqueue(
        "tests.test_localisation_prompt_job._crash_job", job_id="localise-prompt-crash",
        meta=meta,
    )
    SimpleWorker([q], connection=conn).work(burst=True)

    done = lpj.job_state(conn, ok.id, "p1")
    assert done["status"] == "done" and done["result"] == {"prompt": "P"}
    failed = lpj.job_state(conn, bad.id, "p1")
    assert failed["status"] == "failed"
    assert failed["error"] == {"status_code": 502, "detail": "nope"}
    crashed = lpj.job_state(conn, crash.id, "p1")
    assert crashed["status"] == "failed"
    assert crashed["error"]["status_code"] == 500
    assert "Traceback" not in crashed["error"]["detail"]


def test_missing_job_is_expired(conn):
    assert lpj.job_state(conn, "localise-prompt-gone", "p1") == {"status": "expired"}
    assert lpj.job_state(conn, "not-ours", "p1") is None


def _ok_job():
    return {"ok": True, "result": {"prompt": "P"}}


def _bad_job():
    return {"ok": False, "status_code": 502, "detail": "nope"}


def _crash_job():
    raise RuntimeError("bug")


def test_a_lock_lost_during_reserve_does_not_enqueue_a_duplicate(conn):
    """The reserve waited past the admit lock's lease and another admission
    enqueued meanwhile: re-check the pointer instead of enqueueing again."""
    meta = {"project_id": "p1", "fingerprint": "f"}
    enqueued = []
    other_id = "localise-prompt-other"

    def _reserve():
        # The lease runs out and a competing request wins the key.
        conn.delete(lpj.LOCK_PREFIX + "k")
        _enqueue_into(conn, meta)(other_id)
        conn.set(lpj.POINTER_PREFIX + "k", other_id)

    def _enqueue(job_id):
        enqueued.append(job_id)

    job_id, created = lpj.admit(conn, key="k", reserve=_reserve, enqueue=_enqueue)
    assert (job_id, created) == (other_id, False)
    assert enqueued == []


def test_a_lock_lost_with_no_active_job_refuses_instead_of_enqueueing(conn):
    """Another admission may be past its own pointer check: never enqueue
    after losing the lock."""
    meta = {"project_id": "p1", "fingerprint": "f"}
    enqueued = []

    def _reserve():
        conn.delete(lpj.LOCK_PREFIX + "k")

    with pytest.raises(lpj.AdmissionError) as exc:
        lpj.admit(conn, key="k", reserve=_reserve, enqueue=enqueued.append)
    assert exc.value.status_code == 409
    assert enqueued == []


def test_a_failed_enqueue_withdraws_its_pointer(conn):
    def _enqueue(job_id):
        raise RuntimeError("redis hiccup")

    with pytest.raises(RuntimeError):
        lpj.admit(conn, key="k", reserve=lambda: None, enqueue=_enqueue)
    assert conn.get(lpj.POINTER_PREFIX + "k") is None


def test_a_fresh_pointer_without_a_job_is_being_started(conn):
    """Pointer first, then enqueue: a pointer published moments ago whose job
    has not landed yet must not let a second admission enqueue."""
    conn.set(lpj.POINTER_PREFIX + "k", "localise-prompt-pending", ex=lpj._pointer_ttl_full())
    enqueued = []
    with pytest.raises(lpj.AdmissionError) as exc:
        lpj.admit(conn, key="k", reserve=lambda: None, enqueue=enqueued.append)
    assert exc.value.status_code == 409
    assert enqueued == []


def test_an_old_pointer_without_a_job_is_stale(conn):
    conn.set(lpj.POINTER_PREFIX + "k", "localise-prompt-gone", ex=60)
    enqueued = []
    job_id, created = lpj.admit(conn, key="k", reserve=lambda: None, enqueue=enqueued.append)
    assert created is True and enqueued == [job_id]


def test_failed_enqueue_cleanup_spares_a_newer_pointer(conn):
    def _enqueue(job_id):
        conn.set(lpj.POINTER_PREFIX + "k", "localise-prompt-newer")
        raise RuntimeError("redis hiccup")

    with pytest.raises(RuntimeError):
        lpj.admit(conn, key="k", reserve=lambda: None, enqueue=_enqueue)
    assert conn.get(lpj.POINTER_PREFIX + "k") == b"localise-prompt-newer"


def test_default_budgets_fit_the_whole_video_case_and_the_default_job_timeout():
    """Operator 2026-10-30: a ~280 s call still gets the re-ask and the shorten
    pass, within the 420 s call timeout and the 2700 s job timeout."""
    b = lpj.translate_budgets()
    assert b["retry_budget_sec"] >= 280
    assert b["shorten_admit_sec"] >= 2 * 280  # a re-asked whole video still shortens
    assert b["shorten_timeout_sec"] >= 280
    assert lpj.call_timeout_sec() == 420.0
    assert lpj.job_timeout_sec() == 2700


def test_translate_lines_honours_per_call_budgets(monkeypatch):
    """The job's budgets are arguments; the request-path defaults stay.

    Each fake model call takes 200 s: past the request path's 120 s re-ask
    budget, inside the job's 300 s one."""
    now = {"t": 0.0}
    calls = []

    class _Client:
        def create_response(self, **kw):
            now["t"] += 200.0
            calls.append("call")
            return "raw"

    rejoins = {"n": 0}

    def _rejoin(raw, lines):
        rejoins["n"] += 1
        if rejoins["n"] % 2 == 1:
            raise localisation.LocalisationError("model split line 1")
        return [dict(lines[0], text="hola", source_text=lines[0]["text"])]

    shortens = []
    monkeypatch.setattr(localisation, "get_shared_client", lambda: _Client())
    monkeypatch.setattr(localisation.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(localisation.spend, "record_draw", lambda *a, **k: None)
    monkeypatch.setattr(localisation, "_rejoin_translations", _rejoin)
    monkeypatch.setattr(
        localisation, "_shorten_overruns",
        lambda translated, **kw: shortens.append(kw.get("timeout_sec")) or translated,
    )
    line = {"id": 1, "start": 0.0, "end": 2.0, "speaker": "a", "text": "hi"}

    with pytest.raises(localisation.LocalisationError):
        localisation.translate_lines([line], source_language="en", target_language="es")
    assert len(calls) == 1 and shortens == []

    calls.clear()
    now["t"] = 0.0
    rejoins["n"] = 0
    out = localisation.translate_lines(
        [line], source_language="en", target_language="es",
        retry_budget_sec=300, shorten_admit_sec=600, shorten_timeout_sec=300,
    )
    assert [ln["text"] for ln in out] == ["hola"]
    assert len(calls) == 2          # re-asked
    assert shortens == [300]        # and shortened, with the job's timeout
