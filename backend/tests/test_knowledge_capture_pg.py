"""Knowledge capture on PostgreSQL (AICP-299; docs/knowledge-capture-hooks.md §12).

Opt-in like tests/test_knowledge_pg.py: set RESKIN_TEST_PG_URL to a DISPOSABLE
PostgreSQL (this creates and drops a database). The capture executor runs on
real threads against that database; hardlinks go to a temporary DATA_DIR.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)")

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_kcap_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAR_FUTURE_NS = time.time_ns() + 10 ** 12  # every link is "settled" for the scanner


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    import app.models  # noqa: F401
    from app.db import Base

    name = "kcap_" + uuid.uuid4().hex[:12]
    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    url = make_url(_PG_URL).set(database=name).render_as_string(hide_password=False)
    eng = create_engine(url, pool_size=10, max_overflow=0)
    Base.metadata.create_all(eng)
    eng.kb_url = url
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture(scope="module")
def Session(engine):
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def world(Session, engine, monkeypatch, tmp_path):
    """A project, a done Run with one swap RunSegment, capture enabled for it."""
    from app import storage
    from app.config import settings
    from app.knowledge import capture, capture_exec
    from app.models import Run, RunSegment, VideoProject

    s = Session()
    pid = str(uuid.uuid4())
    s.add(VideoProject(id=pid, source_type="upload", source_ref="fixture", project_type="face_swap"))
    s.flush()
    run = Run(project_id=pid, model="seedance", reference_image_urls=["/data/ref0.png", "/data/ref1.png"],
              segment_snapshot=[{"id": "sd1", "index": 0, "start_sec": 1.0, "end_sec": 5.0}])
    s.add(run)
    s.flush()
    rs = RunSegment(run_id=run.id, segment_def_id="sd1", index=0)
    s.add(rs)
    s.commit()
    monkeypatch.setattr(settings, "KNOWLEDGE_CAPTURE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", pid, raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA", False, raising=False)
    monkeypatch.setattr(settings, "DATABASE_URL", engine.kb_url, raising=False)
    monkeypatch.setattr(storage, "data_dir_root", lambda: str(tmp_path))
    capture._scope_cache.clear()
    monkeypatch.setattr(capture, "_DEADLINE", None)
    ex = capture_exec.CaptureExecutor(session_factory=Session, heartbeat=False)
    monkeypatch.setitem(capture_exec._state, "pid", os.getpid())
    monkeypatch.setitem(capture_exec._state, "executor", ex)
    monkeypatch.setattr(capture_exec, "count", lambda site, outcome: None)
    yield {"s": s, "pid": pid, "run_id": run.id, "rs_id": rs.id, "dir": tmp_path}
    s.rollback()
    s.close()
    capture._scope_cache.clear()


RECIPE = dict(run_model="seedance", effective_prompt="swap the face", ref_urls=["https://kie/u1", "https://kie/u2"],
              clip_url="https://kie/clip1", clip_start=0.5, clip_end=5.5, run_resolution="720p",
              project_aspect_ratio="9:16", project_width=1080, project_height=1920, generate_audio=None,
              audio_ref_urls=[])


def _attempts(s, **where):
    from sqlalchemy import select

    from app.knowledge.models import KbAttempt

    q = select(KbAttempt)
    for k, v in where.items():
        q = q.where(getattr(KbAttempt, k) == v)
    s.expire_all()
    return s.execute(q.order_by(KbAttempt.seq)).scalars().all()


def _observations(s, attempt_id):
    from sqlalchemy import select

    from app.knowledge.models import KbAttemptObservation

    s.expire_all()
    return s.execute(select(KbAttemptObservation).where(KbAttemptObservation.attempt_id == attempt_id)
                     .order_by(KbAttemptObservation.seq)).scalars().all()


# ---------------------------------------------------------------- A / B


def test_swap_intent_submission_output_and_segment_version(world):
    from app.knowledge import capture, store

    s, run_id, rs_id = world["s"], world["run_id"], world["rs_id"]
    aid = capture.swap_intent(rs_id=rs_id, recipe=RECIPE, include_audio=True, site="A1")
    assert aid is not None
    (a,) = _attempts(s, id=aid)
    assert a.attempt_kind == "generation" and a.target_kind == "run_segment" and a.target_id == rs_id
    assert a.request["model"] == "bytedance/seedance-2" and a.request["input"]["reference_video_urls"] == ["https://kie/clip1"]
    assert a.reference_set_sha256 is None
    assert a.recipe_context["dispatch_basis"]["ref_locators"] == ["local:/data/ref0.png", "local:/data/ref1.png"]
    capture.submitted(aid, "task-1", site="A1")
    result = world["dir"] / "result_0000.mp4"
    result.write_bytes(b"generated")
    capture.output(task_id="task-1", path=str(result), run_id=run_id, rs_id=rs_id, segment_version=True, site="B1")
    capture.output(task_id="task-1", path=str(result), run_id=run_id, rs_id=rs_id, segment_version=True, site="B3")
    kinds = [(o.kind, o.status) for o in _observations(s, aid)]
    assert kinds == [("status", "submitted"), ("output", None), ("status", "succeeded")]  # B1 ∥ B3 deduped
    case = store.current_case(s, store.segment_case_key(run_id, "sd1"))
    vs = store.versions(s, case.id)
    assert [(v.version_no, v.reason, v.source_record) for v in vs] == \
        [(1, "initial_output", f"import:run_segments:{rs_id}:task-1")]
    assert (case.time_start_sec, case.time_end_sec) == (1.0, 5.0)

    # A rerun after a completed dispatch is a new attempt with the predecessor.
    aid2 = capture.swap_intent(rs_id=rs_id, recipe=dict(RECIPE, effective_prompt="again"), site="A1")
    (a2,) = _attempts(s, id=aid2)
    assert a2.attempt_kind == "rerun" and a2.predecessor_task_id == "task-1"
    capture.submitted(aid2, "task-2", site="A1")
    result.write_bytes(b"regenerated")
    capture.output(task_id="task-2", path=str(result), run_id=run_id, rs_id=rs_id, segment_version=True)
    assert [v.reason for v in store.versions(s, case.id)] == ["initial_output", "regeneration"]


def test_recovered_dispatch_with_reuploaded_refs_is_a_redispatch(world):
    from app.knowledge import capture

    s, rs_id = world["s"], world["rs_id"]
    first = capture.swap_intent(rs_id=rs_id, recipe=RECIPE, site="A1")
    # The worker died before createTask returned; recovery re-uploads (new URLs).
    again = capture.swap_intent(rs_id=rs_id, recipe=dict(RECIPE, ref_urls=["https://kie/n1", "https://kie/n2"],
                                                         clip_url="https://kie/clip-new"), site="A1")
    assert again == first
    assert [o.kind for o in _observations(s, first)] == ["redispatch"]
    other = capture.swap_intent(rs_id=rs_id, recipe=dict(RECIPE, effective_prompt="a new prompt"), site="A1")
    assert other != first  # a different recipe is a new dispatch


def test_resubmit_records_the_predecessor_failure(world):
    from app.knowledge import capture

    s, rs_id = world["s"], world["rs_id"]
    aid = capture.swap_intent(rs_id=rs_id, recipe=RECIPE, site="A1")
    capture.submitted(aid, "task-a", site="A1")
    aid2 = capture.swap_intent(rs_id=rs_id, recipe=RECIPE, resubmit_of="task-a", site="A2")
    (a2,) = _attempts(s, id=aid2)
    assert a2.attempt_kind == "resubmit" and a2.predecessor_task_id == "task-a" and a2.parent_attempt_id == aid
    capture.submitted(aid2, "task-b", site="A2", predecessor_failed={
        "task_id": "task-a", "error_class": "Internal Error", "payload": {"failMsg": "Internal Error"}})
    capture.submitted(aid2, "task-b", site="A2", predecessor_failed={
        "task_id": "task-a", "error_class": "Internal Error", "payload": {"failMsg": "Internal Error"}})
    statuses = [(o.status, o.error_class) for o in _observations(s, aid) if o.kind == "status"]
    assert statuses == [("submitted", None), ("failed", "Internal Error")]
    capture.terminal(task_id="task-b", status="unknown", error_class="pipeline_timeout")
    assert [o.status for o in _observations(s, aid2) if o.kind == "status"] == ["submitted", "unknown"]


def test_orphan_output_creates_one_ambiguous_attempt(world):
    from app.knowledge import capture

    s, run_id, rs_id = world["s"], world["run_id"], world["rs_id"]
    result = world["dir"] / "result.mp4"
    result.write_bytes(b"x")
    for _ in range(2):
        capture.output(task_id="orphan-task", path=str(result), run_id=run_id, rs_id=rs_id, site="B1")
    (a,) = _attempts(s, idempotency_key="orphan:kie:orphan-task")
    kinds = [o.kind for o in _observations(s, a.id)]
    assert kinds.count("ambiguous") == 1 and kinds.count("output") == 1 and a.request is None


def test_preprocess_batch_is_recorded_after_the_loop(world):
    from app.knowledge import capture, store

    s, run_id = world["s"], world["run_id"]
    capture.intent_and_submission_batch(run_id=run_id, items=[
        {"index": 1, "task_id": "img-1", "prompt": "p1", "input_urls": ["u0", "u1"], "locators": ["/data/a", "/data/b"]},
        {"index": 2, "task_id": "img-2", "prompt": "p2", "input_urls": ["u0", "u2"], "locators": ["/data/a", "/data/c"]},
    ])
    rows = _attempts(s, attempt_kind="preprocess_image")
    assert [r.target_id for r in rows] == [f"{run_id}:1", f"{run_id}:2"]
    assert all(r.recipe_context["intent_recorded_after_call"] for r in rows)
    run_case = store.current_case(s, store.run_case_key(run_id))
    assert {r.case_id for r in rows} == {run_case.id}
    assert all([o.status for o in _observations(s, r.id)] == ["submitted"] for r in rows)


def test_first_case_creation_skips_when_the_run_is_locked(world, Session):
    from sqlalchemy import select

    from app.knowledge import capture, store
    from app.models import Run

    run_id, rs_id = world["run_id"], world["rs_id"]
    holder = Session()
    holder.execute(select(Run.id).where(Run.id == run_id).with_for_update())
    try:
        t0 = time.monotonic()
        assert capture.swap_intent(rs_id=rs_id, recipe=RECIPE, site="A1") is None
        assert time.monotonic() - t0 < 2.5
    finally:
        holder.rollback()
        holder.close()
    s = Session()
    assert store.current_case(s, store.run_case_key(run_id)) is None  # nothing written
    s.close()
    assert capture.swap_intent(rs_id=rs_id, recipe=RECIPE, site="A1") is not None  # free again: created


# ---------------------------------------------------------------- finals: links, scan, C2, S14


def _final(world, name, data):
    from app.models import Run

    path = world["dir"] / name
    path.write_bytes(data)
    s = world["s"]
    s.expire_all()
    run = s.get(Run, world["run_id"])
    run.result_local_path = str(path)
    s.commit()
    return str(path)


def _versions(s, run_id):
    from app.knowledge import store

    s.expire_all()
    case = store.current_case(s, store.run_case_key(run_id))
    return [] if case is None else store.versions(s, case.id)


def _drain(Session, run_id):
    from app.knowledge import store

    s = Session()
    try:
        out = store.drain_run_captures(s, run_id)
        s.commit()
        return [d.outcome for d in out]
    finally:
        s.close()


def test_publication_link_is_scanned_queued_and_drained_as_a_preserved_version(world, Session):
    from app.knowledge import capture, finals_scan
    from app.knowledge.models import KbArtifact
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    path = _final(world, "final.mp4", b"FINAL-A")
    pf = PublishedFinal(run_id, pid, path, "stitch", f"stitch:{run_id}:job-1")
    assert capture.link_publication(pf) is None
    s = Session()
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 1
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 0  # covered now
    s.close()
    assert _drain(Session, run_id) == ["appended"]
    (v,) = _versions(world["s"], run_id)
    art = world["s"].get(KbArtifact, v.artifact_id)
    assert v.source_record == pf.publication_id and v.reason == "initial_output" and not v.historical
    assert art.locator_kind == "preserved_copy" and os.path.samefile(art.locator, path)


def test_finals_scan_skips_a_project_no_longer_in_scope(world, Session, monkeypatch):
    import time as _time

    from app.config import settings
    from app.knowledge import capture, finals_scan
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    path = _final(world, "final.mp4", b"FINAL-OUT")
    assert capture.link_publication(PublishedFinal(run_id, pid, path, "stitch", f"stitch:{run_id}:j")) is None
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", str(uuid.uuid4()))  # pid dropped from the scope
    total = finals_scan.run(Session, deadline=_time.monotonic() + 30)
    assert total["enqueued"] == 0 and total["runs"] == 0 and total["out_of_scope"] == 1


class _TickRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    def eval(self, script, numkeys, key, token, *args):
        if self.data.get(key) != token:
            return 0
        if "EXPIRE" not in script:
            del self.data[key]
        return 1


def test_drain_only_tick_drains_in_scope_and_leaves_a_dropped_project_pending(world, Session, monkeypatch):
    from sqlalchemy import exists, select

    from app.config import settings
    from app.knowledge import capture, finals_scan
    from app.knowledge.importer import run_tick
    from app.knowledge.models import KbCaptureDone, KbCaptureQueue
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    path = _final(world, "final.mp4", b"FINAL-DRAIN")
    assert capture.link_publication(PublishedFinal(run_id, pid, path, "stitch", f"stitch:{run_id}:d")) is None
    s = Session()
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 1
    s.close()

    def pending():
        x = Session()
        try:
            return x.execute(select(KbCaptureQueue.id).where(
                KbCaptureQueue.run_id == run_id,
                ~exists().where(KbCaptureDone.queue_id == KbCaptureQueue.id))).all()
        finally:
            x.close()

    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", str(uuid.uuid4()))  # pid dropped after queueing
    assert run_tick(Session, _TickRedis(), hasher=None, drain_only=True)["outcome"] == "finished"
    assert len(pending()) == 1 and not _versions(world["s"], run_id)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", pid)  # back in scope: the next tick drains it
    assert run_tick(Session, _TickRedis(), hasher=None, drain_only=True)["outcome"] == "finished"
    assert pending() == [] and len(_versions(world["s"], run_id)) == 1


def test_round4_b1_a_late_link_of_an_obsolete_final_is_historical(world, Session):
    """Stitch A, then a promotion B; A's link is made late. Current stays B."""
    from app.knowledge import capture, finals_scan, links
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    a = _final(world, "final.mp4", b"A")
    st_a = os.stat(a)
    tmp_a = world["dir"] / "keep-a.mp4"
    os.link(a, tmp_a)  # keep A's inode alive, as a descheduled worker would hold it
    b = _final(world, "range_final.mp4", b"B")
    capture.link_publication(PublishedFinal(run_id, pid, b, "range", "range_attempt:r1"))
    links.make_link(str(tmp_a), pid, run_id, "stitch", f"stitch:{run_id}:late", capture.capture_scope().scope8, st_a)
    s = Session()
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 2
    s.close()
    outcomes = _drain(Session, run_id)
    assert sorted(outcomes) == ["appended", "historical"]
    vs = _versions(world["s"], run_id)
    current = [v for v in vs if not v.historical]
    assert [v.source_record for v in current] == ["range_attempt:r1"]


def test_superseded_link_is_history_only_until_a_failure_row_proves_it(world, Session):
    from sqlalchemy import select

    from app.knowledge import capture, finals_scan
    from app.knowledge.models import KbArtifact, KbCaptureQueue, KbLinkFailure
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    path = _final(world, "final.mp4", b"DELIVERED")
    capture.link_preserve(run_id, pid, path)  # point (ii) before a restitch
    s = Session()
    stats = finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)
    assert stats == {"enqueued": 0, "history": 1, "pending": 1}
    assert s.execute(select(KbCaptureQueue).where(KbCaptureQueue.run_id == run_id)).first() is None
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["history"] == 0  # idempotent
    s.close()

    # Point (i) for that inode had failed: the durable row proves the publication.
    pf = PublishedFinal(run_id, pid, path, "stitch", f"stitch:{run_id}:job-x")
    failed = capture.LinkFailed(pf, OSError(18, "EXDEV"), os.stat(path), time.time_ns())
    capture.record_link_failure(failed)
    capture.record_link_failure(failed)  # idempotent
    s = Session()
    assert len(s.execute(select(KbLinkFailure).where(KbLinkFailure.run_id == run_id)).all()) == 1
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 1
    s.close()
    assert _drain(Session, run_id) == ["appended"]
    (v,) = _versions(world["s"], run_id)
    assert v.source_record == pf.publication_id
    assert world["s"].get(KbArtifact, v.artifact_id).locator_kind == "preserved_copy"


def test_null_identity_failure_pairs_with_the_next_superseded_link(world, Session):
    from app.knowledge import capture, finals_scan, links
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    path = _final(world, "final.mp4", b"X")
    pf = PublishedFinal(run_id, pid, path, "stitch", f"stitch:{run_id}:nostat")
    capture.record_link_failure(capture.LinkFailed(pf, TimeoutError(), None, time.time_ns()))
    capture.link_preserve(run_id, pid, path)
    s = Session()
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 1
    s.close()
    assert _drain(Session, run_id) == ["appended"]


def test_s14_sync_records_the_live_link_and_the_scan_adds_nothing(world, Session):
    from app.knowledge import capture, finals_scan, store
    from app.knowledge.models import KbArtifact
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    path = _final(world, "final.mp4", b"LIVE")
    pf = PublishedFinal(run_id, pid, path, "stitch", f"stitch:{run_id}:job-s14")
    capture.link_publication(pf)
    s = Session()
    store.lock_run_and_cases(s, run_id, [])
    case = store.ensure_run_case(s, run_id)
    v = store.sync_run_case_version(s, case)
    s.commit()
    assert v.source_record == pf.publication_id
    assert s.get(KbArtifact, v.artifact_id).locator_kind == "preserved_copy"
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 0
    s.close()
    assert len(_versions(world["s"], run_id)) == 1


def test_s14_a_start_before_the_link_keeps_its_version_and_its_bytes_count_as_preserved(world, Session):
    """Start records ``local_path`` before capture links the same bytes.

    The late link is a same-bytes duplicate, so feedback stays on v1, and S10
    sees v1's bytes as preserved (content-addressed) once the link exists.
    """
    from app.knowledge import capture, finals_scan, store
    from app.knowledge.models import KbArtifact
    from app.knowledge.publication import PublishedFinal

    run_id, pid = world["run_id"], world["pid"]
    path = _final(world, "final.mp4", b"LIVE-EARLY")
    s = Session()
    store.lock_run_and_cases(s, run_id, [])
    case = store.ensure_run_case(s, run_id)
    v1 = store.sync_run_case_version(s, case)  # no link yet: the worker is paused before point (i)
    s.commit()
    v1_no, art1_id = v1.version_no, v1.artifact_id
    art1 = s.get(KbArtifact, art1_id)
    assert art1.locator_kind == "local_path" and not store.bytes_preserved(s, art1)
    pf = PublishedFinal(run_id, pid, path, "stitch", f"stitch:{run_id}:late-link")
    assert capture.link_publication(pf) is None
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 1
    s.close()
    assert _drain(Session, run_id) == ["duplicate"]
    (only,) = _versions(world["s"], run_id)
    assert only.version_no == v1_no  # feedback given on v1 stays on the current version
    s = Session()
    assert store.bytes_preserved(s, s.get(KbArtifact, art1_id))
    # Once linked, neither a re-sync nor a re-scan adds anything.
    store.lock_run_and_cases(s, run_id, [case.id])
    assert store.sync_run_case_version(s, s.get(store.KbCase, case.id)) is None
    s.commit()
    assert finals_scan.scan_run(s, pid, run_id, now_ns=FAR_FUTURE_NS)["enqueued"] == 0
    s.close()
    assert len(_versions(world["s"], run_id)) == 1


def test_bytes_preserved_ignores_other_projects_and_withdrawn_copies(world, Session):
    from app.knowledge import store

    s = Session()
    sha = "ab" * 32
    local = store.get_or_create_artifact(s, kind="delivered_final", locator_kind="local_path", locator="/l",
                                         sha256=sha, project_id=world["pid"])
    other = store.get_or_create_artifact(s, kind="delivered_final", locator_kind="preserved_copy", locator="/o",
                                         sha256=sha, project_id=str(uuid.uuid4()))
    assert not store.bytes_preserved(s, local)  # same bytes, another project's copy
    mine = store.get_or_create_artifact(s, kind="delivered_final", locator_kind="preserved_copy", locator="/m",
                                        sha256=sha, project_id=world["pid"])
    assert store.bytes_preserved(s, local)
    s.commit()
    store.invalidate(s, target_kind="artifact", target_id=mine.id, reason="test")
    s.commit()
    assert not store.bytes_preserved(s, local)
    assert other.id != mine.id
    s.close()


def test_capture_scope_row_is_recorded_once(world, Session):
    from sqlalchemy import select

    from app.knowledge import capture
    from app.knowledge.models import KbCaptureScope

    s = Session()
    for _ in range(2):
        capture.record_capture_scope(s, recorded_by="test")
        s.commit()
    rows = s.execute(select(KbCaptureScope).where(
        KbCaptureScope.scope_sha256 == capture.capture_scope().sha256)).scalars().all()
    assert len(rows) == 1 and rows[0].project_ids == [world["pid"]]
    s.close()
