"""Knowledge capture hooks — unit tests without PostgreSQL (AICP-299).

docs/knowledge-capture-hooks.md §12. The PostgreSQL half is
tests/test_knowledge_capture_pg.py.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import pytest

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_kcap_unit_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKEND = os.path.join(os.path.dirname(__file__), "..")
PID = "11111111-2222-4333-8444-555555555555"


# ---------------------------------------------------------------- scope / gate


def test_capture_scope_is_an_explicit_uuid_list():
    from app.knowledge.capture import CaptureScope

    ok = CaptureScope.parse(f"{PID}, {PID.upper()}")
    assert ok.valid and ok.allows(PID) and ok.allows(PID.upper()) and len(ok.project_ids) == 1
    assert len(ok.sha256) == 64 and ok.scope8 == ok.sha256[:8]
    assert not CaptureScope.parse("*").valid
    assert not CaptureScope.parse(f"{PID},*").valid
    assert not CaptureScope.parse("").valid
    assert not CaptureScope.parse("not-a-uuid").valid
    assert not CaptureScope.parse(None).valid
    assert not CaptureScope.parse(f"{PID}").allows("99999999-2222-4333-8444-555555555555")


def test_capture_is_off_unless_flag_postgres_and_scope(monkeypatch):
    from app.config import settings
    from app.knowledge import capture

    monkeypatch.setattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", PID, raising=False)
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://x/y", raising=False)
    assert not capture.enabled()
    monkeypatch.setattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", True, raising=False)
    assert capture.enabled() and capture.in_scope(PID)
    monkeypatch.setattr(settings, "DATABASE_URL", "sqlite:///x.db", raising=False)
    assert not capture.enabled()
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://x/y", raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", "*", raising=False)
    assert not capture.enabled() and not capture.in_scope(PID)


def test_every_hook_is_a_no_op_with_capture_off(monkeypatch):
    """Flags off: no executor, no thread, no DB, no syscall on the publication path."""
    from app.config import settings
    from app.knowledge import capture, capture_exec
    from app.knowledge.publication import PublishedFinal

    monkeypatch.setattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", False, raising=False)

    def boom(*a, **k):
        raise AssertionError("capture touched the executor while off")

    monkeypatch.setattr(capture_exec, "executor", boom)
    monkeypatch.setattr(os, "link", boom)
    assert capture.swap_intent(rs_id="x", recipe={}) is None
    assert capture.image_intent(run_id="r", rs_id=None, target_kind="t", target_id="x", attempt_kind="segment_image",
                                origin="operator", prompt="p", input_urls=[], locators=[], recipe_context={},
                                site="A6") is None
    capture.submitted("a", "t")
    capture.dispatch_error("a", RuntimeError("x"))
    capture.intent_and_submission_batch(run_id="r", items=[{"index": 1}])
    capture.range_ambiguous("a", None)
    capture.output(task_id="t", path="/x", run_id="r", rs_id=None)
    capture.terminal(task_id="t", status="failed")
    assert capture.link_publication(PublishedFinal("r", PID, "/x", "stitch", "stitch:r:1")) is None
    capture.link_preserve("r", PID, "/x")
    capture.record_link_failure(None)
    capture.begin_job()


# ---------------------------------------------------------------- links


def test_link_names_round_trip_any_publication_id():
    from app.knowledge import links

    st = os.stat(__file__)
    for pub in ("stitch:run-1:abc", "range_attempt:9f_x__y", "assistant_promotion:a b/c"):
        name = links.link_name("stitch", pub, st, "deadbeef", time_ns=123)
        parsed = links.parse_link(os.path.join("/d", name))
        assert parsed is not None and parsed.publication_id == pub and parsed.kind == "stitch"
        assert parsed.identity == f"{st.st_dev}:{st.st_ino}:{st.st_mtime_ns}" and parsed.time_ns == 123
    assert links.parse_link("/d/random.mp4") is None


@pytest.fixture()
def capture_on(monkeypatch, tmp_path):
    from app import storage
    from app.config import settings
    from app.knowledge import capture

    monkeypatch.setattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", PID, raising=False)
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://unit/test", raising=False)
    monkeypatch.setattr(storage, "data_dir_root", lambda: str(tmp_path))
    capture._scope_cache.clear()
    yield tmp_path
    capture._scope_cache.clear()


def test_link_publication_hardlinks_and_absorbs_every_failure(capture_on, monkeypatch):
    from rq.timeouts import JobTimeoutException

    from app.knowledge import capture, links
    from app.knowledge.publication import PublishedFinal

    final = capture_on / "final.mp4"
    final.write_bytes(b"A" * 100)
    pf = PublishedFinal("run1", PID, str(final), "stitch", "stitch:run1:job1")
    assert capture.link_publication(pf) is None
    (made,) = links.iter_links(PID, "run1")
    assert os.path.samefile(made.path, final) and made.publication_id == "stitch:run1:job1"
    assert made.scope8 == capture.capture_scope().scope8

    # Out of scope: no syscall at all.
    called = []
    monkeypatch.setattr(capture.os, "stat", lambda *a, **k: called.append(a))
    monkeypatch.setattr(links, "make_link", lambda *a, **k: called.append(a))
    assert capture.link_publication(PublishedFinal("run1", str(uuid.uuid4()), str(final), "stitch", "s:x")) is None
    assert called == []


def test_link_publication_failure_returns_identity_and_absorbs_rq_timeout(capture_on, monkeypatch):
    from rq.timeouts import JobTimeoutException

    from app.knowledge import capture, links
    from app.knowledge.publication import PublishedFinal

    final = capture_on / "final.mp4"
    final.write_bytes(b"B")
    pf = PublishedFinal("run2", PID, str(final), "stitch", "stitch:run2:j")
    monkeypatch.setattr(links.os, "link", lambda *a: (_ for _ in ()).throw(OSError(18, "EXDEV")))
    failed = capture.link_publication(pf)
    assert failed is not None and failed.stat is not None and failed.stat.st_ino == os.stat(final).st_ino

    monkeypatch.setattr(links.os, "link", lambda *a: (_ for _ in ()).throw(JobTimeoutException("alarm")))
    failed = capture.link_publication(pf)  # point (i) absorbs RQ's timeout
    assert isinstance(failed.exc, JobTimeoutException)

    real_stat = os.stat

    def stat_timeout(path, *a, **k):
        if str(path) == str(final):
            raise JobTimeoutException("alarm in stat")
        return real_stat(path, *a, **k)

    monkeypatch.setattr(capture.os, "stat", stat_timeout)
    failed = capture.link_publication(pf)
    assert failed.stat is None and failed.failed_at_ns > 0  # NULL identity


def test_link_preserve_reraises_rq_timeout_and_skips_linked_inodes(capture_on, monkeypatch):
    from rq.timeouts import JobTimeoutException

    from app.knowledge import capture, links

    final = capture_on / "final.mp4"
    final.write_bytes(b"C")
    capture.link_preserve("run3", PID, str(final))
    (first,) = links.iter_links(PID, "run3")
    assert first.kind == links.SUPERSEDED and os.path.samefile(first.path, final)
    capture.link_preserve("run3", PID, str(final))  # st_nlink == 2 now: nothing more
    assert len(links.iter_links(PID, "run3")) == 1
    capture.link_preserve("run3", PID, str(capture_on / "missing.mp4"))  # absent: silently nothing
    other = capture_on / "other.mp4"
    other.write_bytes(b"D")
    monkeypatch.setattr(links.os, "link", lambda *a: (_ for _ in ()).throw(JobTimeoutException("alarm")))
    with pytest.raises(JobTimeoutException):
        capture.link_preserve("run3", PID, str(other))
    monkeypatch.setattr(links.os, "link", lambda *a: (_ for _ in ()).throw(PermissionError("EACCES")))
    capture.link_preserve("run3", PID, str(other))  # ordinary errors are swallowed


def test_live_link_lookup_finds_only_publishable_links_of_the_live_inode(capture_on):
    from app.knowledge import links

    a, b = capture_on / "a.mp4", capture_on / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    links.make_link(str(a), PID, "r", "stitch", "stitch:r:1", "ab")
    links.make_link(str(b), PID, "r", links.SUPERSEDED, "superseded:r:1", "ab")
    assert links.live_link_lookup(PID, "r", str(a)).publication_id == "stitch:r:1"
    assert links.live_link_lookup(PID, "r", str(b)) is None
    assert links.live_link_lookup(PID, "r", None) is None


# ---------------------------------------------------------------- executor


def _executor(**kw):
    from app.knowledge.capture_exec import CaptureExecutor

    return CaptureExecutor(session_factory=lambda: _FakeSession(), heartbeat=False, **kw)


class _FakeSession:
    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def test_executor_bounds_the_wait_and_never_queues():
    ex = _executor(threads=1, queue_max=1)
    gate = threading.Event()
    t0 = time.monotonic()
    assert ex.call(lambda s: gate.wait(5), 0.2) is None  # abandoned, never waited again
    assert time.monotonic() - t0 < 0.5
    assert ex.call(lambda s: 1, 0.2) is None  # the one slot is queued behind the stuck job...
    t1 = time.monotonic()
    assert ex.call(lambda s: 2, 5) is None  # ...so this is skipped at once (queue full)
    assert time.monotonic() - t1 < 0.1
    gate.set()
    time.sleep(0.2)
    assert ex.call(lambda s: 3, 2) == 3
    assert ex.call(lambda s: 4, 0) is None  # fire-and-forget


def test_executor_survives_errors_and_replaces_dead_threads():
    from sqlalchemy.exc import OperationalError

    made = []

    class S(_FakeSession):
        def __init__(self):
            made.append(self)

    from app.knowledge.capture_exec import CaptureExecutor

    ex = CaptureExecutor(session_factory=S, heartbeat=False, threads=1)
    assert ex.call(lambda s: (_ for _ in ()).throw(ValueError("x")), 2) is None
    assert ex.call(lambda s: "ok", 2) == "ok" and len(made) == 1  # same session after an ordinary error
    assert ex.call(lambda s: (_ for _ in ()).throw(OperationalError("s", {}, Exception("gone"))), 2) is None
    assert ex.call(lambda s: "again", 2) == "again" and len(made) == 2  # reconnected after a connection error
    assert ex.consecutive_failures == 0 and ex.last_ok is not None

    class Dead:
        def is_alive(self):
            return False

    ex.threads[0] = Dead()
    assert ex.call(lambda s: "replaced", 2) == "replaced"
    assert all(t.daemon for t in ex.threads)


def test_a_horse_forked_while_the_parent_holds_the_capture_lock_does_not_deadlock():
    """B3 runs capture in the RQ parent; a fork mid-``_lock`` must not hang the horse."""
    from app.knowledge import capture_exec

    with capture_exec._lock:
        pid = os.fork()
        if pid == 0:  # the horse: the inherited lock is held by a thread that no longer exists
            ok = capture_exec._lock.acquire(timeout=2)  # held forever without the at-fork reset
            os._exit(0 if ok and capture_exec._state["pid"] == os.getpid() else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_health_rule():
    from app.knowledge.capture_exec import unhealthy

    assert not unhealthy(None)
    assert unhealthy({"consecutive_failures": 5})
    assert unhealthy({"last_attempt": 1000.0, "last_ok": 100.0})
    assert not unhealthy({"last_attempt": 1000.0, "last_ok": 999.0, "consecutive_failures": 1})


def test_budget_respects_the_job_deadline(monkeypatch):
    from app.knowledge import capture

    monkeypatch.setattr(capture, "_DEADLINE", time.monotonic() + 60)  # < 90 s margin
    assert capture._budget(3.0) is None
    monkeypatch.setattr(capture, "_DEADLINE", time.monotonic() + 1000)
    assert capture._budget(3.0) == 3.0
    monkeypatch.setattr(capture, "_DEADLINE", None)
    out = []
    t = threading.Thread(target=lambda: out.append(capture._budget(3.0)))
    t.start()
    t.join()
    assert out == [None]  # a pool thread with no deadline skips
    assert capture._budget(3.0) == 3.0  # main thread outside any RQ job (API, boot)


# ---------------------------------------------------------------- atomic stitch (§3.1)


def test_stitch_replaces_atomically_and_a_failure_keeps_the_previous_final(monkeypatch, tmp_path):
    from app import media

    dst = tmp_path / "final.mp4"
    dst.write_bytes(b"OLD")
    old_ino = os.stat(dst).st_ino
    holder = open(dst, "rb")

    def ok(clips, audio_source, out, *a):
        assert out != str(dst) and out.endswith(".part.mp4")
        with open(out, "wb") as fh:
            fh.write(b"NEW")

    monkeypatch.setattr(media, "_stitch_into", ok)
    media.stitch(["c"], audio_source="s", dst=str(dst), width=1, height=1, fps=1.0)
    assert dst.read_bytes() == b"NEW" and os.stat(dst).st_ino != old_ino
    assert holder.read() == b"OLD"  # an open reader keeps the old, complete inode
    holder.close()

    def bad(clips, audio_source, out, *a):
        with open(out, "wb") as fh:
            fh.write(b"PARTIAL")
        raise media.MediaError("ffmpeg died")

    monkeypatch.setattr(media, "_stitch_into", bad)
    with pytest.raises(media.MediaError):
        media.stitch(["c"], audio_source="s", dst=str(dst), width=1, height=1, fps=1.0)
    assert dst.read_bytes() == b"NEW"
    assert [p.name for p in tmp_path.iterdir()] == ["final.mp4"]  # no temp file left


# ---------------------------------------------------------------- kie payload builders (D6)


def test_create_calls_send_exactly_the_builder_payloads(monkeypatch):
    from app import kie_client
    from app.config import settings

    sent = []
    client = kie_client.KieClient(api_key="k")
    monkeypatch.setattr(client, "_create_task_with_retry", lambda url, payload: sent.append(payload) or
                        {"code": 200, "data": {"taskId": "t"}})
    client.create_task(prompt="p", reference_image_urls=["i"], reference_video_urls=["v"], resolution="720p",
                       aspect_ratio="adaptive", duration=-1, model="bytedance/seedance-2-5",
                       generate_audio=False, reference_audio_urls=["a"])
    assert sent[-1] == {"model": "bytedance/seedance-2-5", "input": {
        "prompt": "p", "reference_image_urls": ["i"], "reference_video_urls": ["v"], "resolution": "720p",
        "aspect_ratio": "adaptive", "duration": -1, "generate_audio": False, "reference_audio_urls": ["a"]}}
    client.create_task(prompt="p", reference_image_urls=[], reference_video_urls=["v"], duration=5)
    assert sent[-1] == {"model": "bytedance/seedance-2", "input": {
        "prompt": "p", "reference_image_urls": [], "reference_video_urls": ["v"], "resolution": "720p",
        "aspect_ratio": "9:16", "duration": 5}}
    client.create_omni_task(prompt="p", image_urls=["i"], video_url="v", video_start=0.4, video_end=7.6,
                            duration=8, seed=3)
    assert sent[-1] == {"model": "gemini-omni-video", "input": {
        "prompt": "p", "image_urls": ["i"], "video_list": [{"url": "v", "start": 0, "ends": 8}],
        "resolution": "720p", "aspect_ratio": "9:16", "duration": "8", "seed": 3}}
    client.create_image_task(prompt="p", input_urls=["a", "b"])
    assert sent[-1] == {"model": settings.KIE_IMAGE_MODEL, "input": {
        "prompt": "p", "input_urls": ["a", "b"], "aspect_ratio": "auto",
        "resolution": settings.KIE_IMAGE_RESOLUTION}}
    assert sent[-1] == kie_client.build_image_payload(prompt="p", input_urls=["a", "b"])


def test_swap_task_call_matches_create_swap_task():
    from app import pipeline_v2

    recipe = dict(run_model="seedance", effective_prompt="p", ref_urls=["r"], clip_url="c", clip_start=1.0,
                  clip_end=9.0, run_resolution="720p", project_aspect_ratio="9:16", project_width=1080,
                  project_height=1920, generate_audio=None, audio_ref_urls=[], stray_key=1)
    calls = []

    class K:
        def create_task(self, **kw):
            calls.append(("create_task", kw))
            return "t1"

        def create_omni_task(self, **kw):
            calls.append(("create_omni_task", kw))
            return "t2"

    assert pipeline_v2._create_swap_task(kie=K(), **recipe) == "t1"
    assert calls[-1] == pipeline_v2._swap_task_call(**recipe)
    omni = dict(recipe, run_model="gemini-omni")
    assert pipeline_v2._create_swap_task(kie=K(), **omni) == "t2"
    assert calls[-1] == pipeline_v2._swap_task_call(**omni)


def test_strip_volatile_and_locators():
    from app.knowledge.capture import normalize_locator, strip_volatile

    req = {"model": "m", "input": {"prompt": "p", "reference_image_urls": ["u1", "u2"],
                                   "video_list": [{"url": "v", "start": 0}]}}
    assert strip_volatile(req) == {"model": "m", "input": {"prompt": "p", "reference_image_urls": 2,
                                                           "video_list": [{"start": 0}]}}
    assert normalize_locator("https://drive.google.com/file/d/1AbCdEfGhIjKlMn/view") == "drive:1AbCdEfGhIjKlMn"
    assert normalize_locator("https://x.example/a.png") == "https://x.example/a.png"
    assert normalize_locator("") is None


# ---------------------------------------------------------------- wiring (static)


HOSTING_WRAPPERS = ("run_submit_run", "run_preprocess_refs", "run_preprocess_poll", "run_poll_run",
                    "run_finalize_run", "run_process_audio_patch", "run_process_segment_image",
                    "run_process_segment_image_poll", "run_process_range_attempt",
                    "run_process_range_attempt_poll")


def test_every_hosting_wrapper_records_its_deadline_first():
    tree = ast.parse(open(os.path.join(BACKEND, "app", "tasks.py")).read())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in HOSTING_WRAPPERS:
        body = [s for s in funcs[name].body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        first = body[0]
        assert isinstance(first, ast.Expr) and getattr(first.value.func, "id", None) == "_kcap_begin", name


def test_finalize_releases_the_lock_before_the_failure_row(monkeypatch):
    import app.pipeline_v2 as pv2
    import app.tasks as tasks
    from app.knowledge import capture
    from app.knowledge.publication import PublishedFinal

    order = []
    pf = PublishedFinal("r", PID, "/x/final.mp4", "stitch", "stitch:r:j")
    monkeypatch.setattr(pv2, "finalize_run", lambda run_id, **kw: order.append(("finalize", kw["publication_id"])) or pf)
    monkeypatch.setattr(capture, "link_publication", lambda p: order.append(("link", p.publication_id)) or "FAILED")
    monkeypatch.setattr(capture, "record_link_failure", lambda f: order.append(("row", f)))
    monkeypatch.setattr(tasks, "_release_lock", lambda *a: order.append(("release",)))
    monkeypatch.setattr(tasks, "_get_redis", lambda: None)
    monkeypatch.setattr(tasks, "_refresh_lock", lambda *a: True)
    monkeypatch.setattr("app.gdrive_client.GDriveClient", lambda: None)
    tasks.run_finalize_run("r", lock_key="k", lock_token="t")
    assert [o[0] for o in order] == ["finalize", "link", "release", "row"]

    order.clear()
    monkeypatch.setattr(pv2, "finalize_run", lambda run_id, **kw: None)  # incomplete / no-stitch
    tasks.run_finalize_run("r", lock_key="k", lock_token="t")
    assert order == [("release",)]


# ---------------------------------------------------------------- knowledge worker processes


def test_knowledge_worker_and_tick_path_import_no_ml():
    code = ("import sys; import worker.knowledge_worker, worker.knowledge_ticker, worker.knowledge_supervisor; "
            "import app.tasks, app.knowledge.finals_scan, app.knowledge.capture, app.knowledge.importer; "
            "bad=[m for m in ('app.face','insightface','onnxruntime','cv2') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    root = os.path.abspath(os.path.join(BACKEND, ".."))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.abspath(BACKEND), root]))
    if not os.path.isdir(os.path.join(root, "worker")):
        pytest.skip("worker/ package not present next to backend/")
    res = subprocess.run([sys.executable, "-c", code], cwd=os.path.abspath(BACKEND), env=env,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_supervisor_exits_non_zero_when_a_child_dies():
    root = os.path.abspath(os.path.join(BACKEND, ".."))
    if not os.path.isdir(os.path.join(root, "worker")):
        pytest.skip("worker/ package not present next to backend/")
    sys.path.insert(0, root)
    from worker import knowledge_supervisor

    rc = knowledge_supervisor.supervise(commands=[
        ("worker", [sys.executable, "-c", "import time; time.sleep(30)"]),
        ("ticker", [sys.executable, "-c", "import sys; sys.exit(3)"]),
    ])
    assert rc == 1


def test_ticker_admits_one_tick_per_slot(monkeypatch):
    import app.tasks as tasks
    from app.config import settings
    from app.knowledge import capture

    monkeypatch.setattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", PID, raising=False)
    capture._scope_cache.clear()

    class R:
        def __init__(self):
            self.kv = {}

        def set(self, k, v, nx=False, ex=None):
            if nx and k in self.kv:
                return False
            self.kv[k] = v
            return True

    enq = []

    class Q:
        def get_job_ids(self):
            return []

        def enqueue(self, fn, job_id, job_timeout):
            enq.append(job_id)

    r = R()
    monkeypatch.setattr(tasks, "_get_queue", lambda name: Q())
    monkeypatch.setattr(tasks, "_live_worker_current_jobs", lambda conn: set())
    assert tasks.enqueue_knowledge_tick_for_slot(r, 7, "t1")
    assert not tasks.enqueue_knowledge_tick_for_slot(r, 7, "t2")  # the second ticker loses the slot
    assert enq == ["knowledge-tick-7"]
    monkeypatch.setattr(tasks, "_live_worker_current_jobs", lambda conn: {"knowledge-tick-7"})
    assert not tasks.enqueue_knowledge_tick_for_slot(r, 8, "t1")  # a live tick is never overlapped
    monkeypatch.setattr(tasks, "_live_worker_current_jobs", lambda conn: set())
    assert tasks.enqueue_knowledge_tick_for_slot(r, 8, "t1")
    capture._scope_cache.clear()


class _LockRedis:
    def __init__(self):
        self.kv = {}

    def set(self, k, v, nx=False, ex=None):
        if nx and k in self.kv:
            return False
        self.kv[k] = v
        return True

    def get(self, k):
        return self.kv.get(k)

    def eval(self, script, nkeys, key, *args):
        if "'SET'" in script:  # compare-and-set
            holder, me, _ttl = args
            if self.kv.get(key) == holder:
                self.kv[key] = me
                return 1
            return 0
        (me,) = args  # compare-and-delete
        if self.kv.get(key) == me:
            del self.kv[key]
            return 1
        return 0


def test_tick_lock_is_taken_over_only_from_a_holder_that_is_not_running(monkeypatch):
    import app.tasks as tasks

    r = _LockRedis()
    live = {"job-a"}
    monkeypatch.setattr(tasks, "knowledge_job_is_live", lambda conn, job_id: job_id in live)
    assert tasks.acquire_knowledge_tick_lock(r, "job-a")
    assert not tasks.acquire_knowledge_tick_lock(r, "job-b")  # holder still running: skip
    tasks.release_knowledge_tick_lock(r, "job-b")  # a non-owner never releases it
    assert r.get(tasks._KNOWLEDGE_TICK_LOCK) == "job-a"
    live.clear()  # the holder's worker is gone (no heartbeat, not queued)
    assert tasks.acquire_knowledge_tick_lock(r, "job-b")
    assert r.get(tasks._KNOWLEDGE_TICK_LOCK) == "job-b"
    tasks.release_knowledge_tick_lock(r, "job-b")
    assert tasks._KNOWLEDGE_TICK_LOCK not in r.kv


def test_run_knowledge_tick_skips_while_another_tick_runs(monkeypatch):
    import app.tasks as tasks
    from app.config import settings

    monkeypatch.setattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", True, raising=False)
    monkeypatch.setattr(tasks, "_get_redis", lambda: None)
    monkeypatch.setattr(tasks, "acquire_knowledge_tick_lock", lambda conn, me: False)
    ran = []
    monkeypatch.setattr(tasks, "knowledge_import_tick", lambda: ran.append(1))
    assert tasks.run_knowledge_tick() is None and ran == []


def test_range_published_variant_returns_the_descriptor_only_when_done(monkeypatch):
    import app.pipeline_range_attempt as pra
    from app.knowledge.publication import PublishedFinal

    pf = PublishedFinal("r", PID, "/c/final.mp4", "range", "range_attempt:a1")

    def fake(attempt_id, *, kie=None, gdrive=None, _published=None):
        _published.append(pf)
        return status

    monkeypatch.setattr(pra, "poll_range_attempt_once", fake)
    status = "done"
    assert pra.poll_range_attempt_once_published("a1") == ("done", pf)
    status = "pending"
    assert pra.poll_range_attempt_once_published("a1") == ("pending", None)


def test_range_poll_wrapper_links_after_re_enqueue_decision(monkeypatch):
    import app.pipeline_range_attempt as pra
    import app.tasks as tasks
    from app.knowledge import capture
    from app.knowledge.publication import PublishedFinal

    pf = PublishedFinal("r", PID, "/c/final.mp4", "range", "range_attempt:a1")
    calls = []
    monkeypatch.setattr(pra, "poll_range_attempt_once_published", lambda a: ("done", pf))
    monkeypatch.setattr(tasks, "_enqueue_range_attempt_poll", lambda *a, **k: calls.append("requeue"))
    monkeypatch.setattr(capture, "link_publication", lambda p: calls.append(("link", p)) or None)
    monkeypatch.setattr(capture, "record_link_failure", lambda f: calls.append(("row", f)))
    tasks.run_process_range_attempt_poll("a1")
    assert calls == [("link", pf), ("row", None)]
    calls.clear()
    monkeypatch.setattr(pra, "poll_range_attempt_once_published", lambda a: ("pending", None))
    tasks.run_process_range_attempt_poll("a1")
    assert calls == ["requeue"]


def test_assistant_published_variant_returns_the_descriptor_only_for_a_fresh_promotion(monkeypatch):
    import app.assistant_run_result as arr
    from app.knowledge.publication import PublishedFinal

    pf = PublishedFinal("r", PID, "/a/result.mp4", "assistant", "assistant_promotion:i1")

    def fake(session, service, run_id, request, *, actor="operator", drive=None, _published=None):
        if fresh:
            _published.append(pf)
        return {"id": "i1"}

    monkeypatch.setattr(arr, "promote_run_result", fake)
    fresh = True
    assert arr.promote_run_result_published(None, None, "r", {}) == ({"id": "i1"}, pf)
    fresh = False
    assert arr.promote_run_result_published(None, None, "r", {}) == ({"id": "i1"}, None)


def test_a_finalize_that_lost_its_lock_does_nothing(monkeypatch):
    """AICP-342: a superseded finalize must not stitch, transition or deliver."""
    import app.pipeline_v2 as pv2
    import app.tasks as tasks

    called = []
    monkeypatch.setattr(pv2, "finalize_run", lambda run_id, **kw: called.append(run_id))
    monkeypatch.setattr(tasks, "_get_redis", lambda: None)
    monkeypatch.setattr(tasks, "_refresh_lock", lambda *a: False)
    monkeypatch.setattr(tasks, "_release_lock", lambda *a: called.append("release"))
    tasks.run_finalize_run("r", lock_key="k", lock_token="t")
    assert called == []
