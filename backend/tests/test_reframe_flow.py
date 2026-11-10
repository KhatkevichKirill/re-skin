"""
Reframe flow (AICP-251, docs/reframe-design.md §12): create -> prepare -> admit
through the real provider layer (pipeline_provider.admit, transport never
reached), refusals, recovery, deletion guards, the operator-action endpoints,
job QA and the render, on SQLite with real ffmpeg media. No Redis: every
enqueue is captured.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import sessionmaker

from app import db as db_mod
from app import pipeline_provider as pp
from app import pipeline_reframe as prf
from app import recovery, spend, storage, tasks
from app.config import settings
from app.db import Base
from app.models import KieSpendReservation, LibraryClip, Run, VideoProject
from app.provider_models import ProviderTask, ProviderTaskAction
from app.reframe_models import ReframeJob, ReframeRender
from app.state_machine import ProjectStatus, RunStatus
from reframe_media import make_raw, make_source

pytestmark = [pytest.mark.spend_cap, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sqlite_engine():
    path = tempfile.mkstemp(suffix=".db")[1]
    eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        dbapi_conn.isolation_level = None
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    @event.listens_for(eng, "begin")
    def _begin(conn):
        conn.exec_driver_sql("BEGIN")

    return eng


@pytest.fixture(scope="module")
def engine():
    eng = _sqlite_engine()
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture(scope="module")
def media_dir():
    d = tempfile.mkdtemp()
    return {
        "src": make_source(os.path.join(d, "src.mp4"), w=1280, h=720, fps=30, sec=6.0),
        "silent": make_source(os.path.join(d, "silent.mp4"), w=1280, h=720, fps=30, sec=4.0,
                              audio=False),
        "cta": make_source(os.path.join(d, "cta.mp4"), w=768, h=1344, fps=30, sec=1.0),
        "cta_wide": make_source(os.path.join(d, "ctaw.mp4"), w=1344, h=768, fps=30, sec=1.0),
    }


class Env:
    pass


@pytest.fixture()
def env(engine, media_dir, monkeypatch):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_mod, "SessionLocal", factory)
    e = Env()
    e.factory = factory
    e.enq = []
    monkeypatch.setattr(tasks, "enqueue_reframe_prepare", lambda j: e.enq.append(("prepare", j)))
    monkeypatch.setattr(tasks, "enqueue_reframe_qa", lambda j: e.enq.append(("qa", j)))
    monkeypatch.setattr(tasks, "enqueue_reframe_render", lambda r: e.enq.append(("render", r)))
    monkeypatch.setattr(tasks, "enqueue_fal_job", lambda fn, t, **k: e.enq.append((fn, t)))
    monkeypatch.setattr(pp, "_enqueue", lambda fn, t, delay_sec=0: e.enq.append((fn, t)))
    monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)
    monkeypatch.setattr(settings, "FAL_ENABLED", True)
    monkeypatch.setattr(settings, "FAL_API_KEY", "test-key")
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 250.0)
    from app import api_reframe

    monkeypatch.setattr(api_reframe, "_kick_qa", lambda j: e.enq.append(("kick_qa", j)))
    e.pid = _project(e, media_dir["src"])
    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", e.pid)
    from app.main import app

    e.client = TestClient(app, raise_server_exceptions=True)
    e.media = media_dir
    yield e
    s = factory()
    s.execute(text("DROP TRIGGER IF EXISTS provider_task_actions_no_delete"))
    for model in (ReframeRender, ReframeJob, ProviderTaskAction, ProviderTask,
                  KieSpendReservation):
        s.query(model).delete()
    s.commit()
    s.close()


def _project(e, src, *, ptype="reframe", w=1280, h=720, dur=6.0) -> str:
    with db_mod.get_session() as s:
        p = VideoProject(source_type="upload", source_ref="s.mp4", project_type=ptype,
                         status=ProjectStatus.ready, width=w, height=h, duration_sec=dur,
                         fps=30.0)
        s.add(p)
        s.flush()
        pdir = os.path.join(storage.data_dir_root(), "projects", p.id)
        os.makedirs(pdir, exist_ok=True)
        dst = os.path.join(pdir, "source.mp4")
        with open(src, "rb") as a, open(dst, "wb") as b:
            b.write(a.read())
        p.source_local_path = dst
        return p.id


def _create(e, start=0.5, end=3.5, res="480p", pid=None):
    return e.client.post(f"/api/v2/projects/{pid or e.pid}/reframe-jobs",
                         data={"window_start_sec": start, "window_end_sec": end,
                               "resolution": res})


def _job(e, jid) -> ReframeJob:
    with db_mod.get_session() as s:
        j = s.get(ReframeJob, jid)
        s.expunge(j)
        return j


def _task(e, tid) -> ProviderTask:
    with db_mod.get_session() as s:
        t = s.get(ProviderTask, tid)
        s.expunge(t)
        return t


def _sql(stmt, **kw):
    with db_mod.get_session() as s:
        s.execute(text(stmt), kw)


def _admitted(e, **kw) -> ReframeJob:
    r = _create(e, **kw)
    assert r.status_code == 201, r.text
    prf.reframe_prepare(r.json()["id"])
    job = _job(e, r.json()["id"])
    assert job.status == "admitted", (job.error_code, job.error_message)
    return job


def _done(e, job, frames=None, size=(480, 832)):
    """Make the job's task `done` with a raw output at its destination."""
    t = _task(e, job.provider_task_id)
    frames = frames if frames is not None else round(job.input_duration_sec * 24)
    make_raw(t.destination_path, w=size[0], h=size[1], frames=frames)
    _sql("UPDATE provider_tasks SET status='done', reservation_held=0, result_path=:p, "
         "result_meta=:m WHERE id=:id", p=t.destination_path, id=t.id,
         m='{"seed": 7, "frame_count": %d}' % frames)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_inserts_preparing_and_enqueues(env):
    r = _create(env)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["estimated_usd_base"] == pytest.approx(0.15)
    assert body["estimated_usd_reserved"] == pytest.approx(0.30)
    job = _job(env, body["id"])
    assert (job.status, job.target_aspect, job.resolution) == ("preparing", "9:16", "480p")
    assert ("prepare", job.id) in env.enq
    assert os.path.isdir(storage.reframe_job_dir(job.id, env.pid))
    # A second click while preparing: 409, and the unique index backs it.
    assert _create(env).status_code == 409


@pytest.mark.parametrize("patch, code", [
    ({"FAL_ENABLED": False}, 403),
    ({"FAL_API_KEY": ""}, 403),
])
def test_create_fal_gate_403(env, monkeypatch, patch, code):
    for k, v in patch.items():
        monkeypatch.setattr(settings, k, v)
    assert _create(env).status_code == code
    with db_mod.get_session() as s:
        assert s.execute(select(ReframeJob)).first() is None


def test_create_reframe_project_needs_no_allowlist(env, monkeypatch):
    # A reframe project is admitted without being listed in FAL_PROJECT_IDS.
    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", "someone-else")
    assert _create(env).status_code == 201
    with db_mod.get_session() as s:
        assert s.execute(select(ReframeJob)).scalar_one().status == "preparing"


def test_create_503_without_fal_worker(env, monkeypatch):
    monkeypatch.setattr(pp, "_fal_worker_alive", lambda: False)
    r = _create(env)
    assert r.status_code == 503 and "fal-worker" in r.json()["detail"]
    assert not [x for x in env.enq if x[0] == "prepare"]


def test_create_refusals(env):
    assert _create(env, start=3, end=2).status_code == 400
    assert env.client.post(f"/api/v2/projects/{env.pid}/reframe-jobs",
                           data={"window_start_sec": 0, "window_end_sec": 3,
                                 "target_aspect": "16:9"}).status_code == 400
    other = _project(env, env.media["src"], ptype="cover_change")
    assert _create(env, pid=other).status_code == 409


def test_budget_precheck_429(env, monkeypatch):
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0001)
    assert _create(env).status_code == 429


# ---------------------------------------------------------------------------
# Prepare + admit
# ---------------------------------------------------------------------------


def test_prepare_admits_one_task_and_enqueues_submit(env):
    job = _admitted(env)
    jdir = storage.reframe_job_dir(job.id, env.pid)
    assert os.path.isfile(job.input_path) and os.path.isfile(job.audio_path)
    assert job.input_start_sec == pytest.approx(15 / 30, abs=1e-3)  # first frame >= 0.5
    assert job.input_frames == round((3.5 - job.input_start_sec) * 30)
    assert job.prep_claim_owner is None and job.prep_attempts == 1
    assert job.audio_duration_sec == pytest.approx(6.0 - job.input_start_sec, abs=0.02)
    t = _task(env, job.provider_task_id)
    assert (t.owner_kind, t.owner_id, t.owner_slot, t.owner_project_id) == (
        "reframe_job", job.id, "main", env.pid)
    assert t.operation_key == f"reframe_job:{job.id}:main:1"
    assert t.destination_path == os.path.join(jdir, "raw.mp4") and os.path.isabs(t.destination_path)
    assert t.kind == "fal_reframe" and t.status == "queued"
    assert t.request_spec["params"] == {"aspect_ratio": "9:16", "resolution": "480p"}
    assert env.enq.count(("fal_submit", t.id)) == 1
    assert not [d for d in os.listdir(jdir) if d.startswith("prep-")]


def test_create_409_while_task_active_for_each_active_status(env):
    job = _admitted(env)
    for st, held in (("queued", 1), ("submitted", 0), ("submit_unknown", 1), ("refused", 1)):
        _sql("UPDATE provider_tasks SET status=:s, reservation_held=:h WHERE id=:id",
             s=st, h=held, id=job.provider_task_id)
        assert _create(env).status_code == 409, st
    _sql("UPDATE provider_tasks SET status='refused', reservation_held=0 WHERE id=:id",
         id=job.provider_task_id)
    assert _create(env).status_code == 201


@pytest.mark.parametrize("code, error_code", [(503, "fal_worker_down"), (403, "fal_disabled"),
                                              (422, "admission_invalid")])
def test_admission_refusal_ends_prepare_failed(env, monkeypatch, code, error_code):
    r = _create(env)
    jid = r.json()["id"]
    if code == 503:
        monkeypatch.setattr(pp, "_fal_worker_alive", lambda: False)
    elif code == 403:
        monkeypatch.setattr(settings, "FAL_ENABLED", False)
    else:
        monkeypatch.setattr(settings, "FAL_MAX_INPUT_BYTES", 10**12)
        monkeypatch.setattr(settings, "FAL_OUTPUT_DURATION_TOLERANCE_SEC", -1.0)
    prf.reframe_prepare(jid)
    job = _job(env, jid)
    assert (job.status, job.error_code) == ("prepare_failed", error_code)
    assert job.provider_task_id is None
    assert not os.path.exists(os.path.join(storage.reframe_job_dir(jid, env.pid), "input.mp4"))
    with db_mod.get_session() as s:
        assert s.execute(select(ProviderTask)).first() is None
        assert s.execute(select(KieSpendReservation)).first() is None
    assert not [x for x in env.enq if x[0] == "fal_submit"]


def test_budget_exceeded_at_admission(env, monkeypatch):
    jid = _create(env).json()["id"]
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0001)
    prf.reframe_prepare(jid)
    job = _job(env, jid)
    assert (job.status, job.error_code) == ("prepare_failed", "budget")
    with db_mod.get_session() as s:
        assert s.execute(select(ProviderTask)).first() is None


def test_input_too_large(env, monkeypatch):
    jid = _create(env).json()["id"]
    monkeypatch.setattr(settings, "FAL_MAX_INPUT_BYTES", 2000)
    prf.reframe_prepare(jid)
    job = _job(env, jid)
    assert (job.status, job.error_code) == ("prepare_failed", "input_too_large")


def test_silent_source_has_no_audio_snapshot(env, monkeypatch):
    pid = _project(env, env.media["silent"], dur=4.0)
    monkeypatch.setattr(settings, "FAL_PROJECT_IDS", pid)
    r = _create(env, start=0, end=3, pid=pid)
    prf.reframe_prepare(r.json()["id"])
    job = _job(env, r.json()["id"])
    assert job.status == "admitted" and job.source_has_audio is False and job.audio_path is None


def test_lost_claim_never_admits(env):
    jid = _create(env).json()["id"]
    _sql("UPDATE reframe_jobs SET prep_claim_owner='other', prep_attempts=1, "
         "prep_claim_expires_at=:x WHERE id=:id", x=_utcnow() + timedelta(hours=1), id=jid)
    prf.reframe_prepare(jid)  # not claimable: another execution owns it
    assert _job(env, jid).status == "preparing"
    # An execution whose claim was taken over mid-way changes zero rows.
    jdir = storage.reframe_job_dir(jid, env.pid)
    staging = os.path.join(jdir, "prep-mine")
    os.mkdir(staging)
    prf._prepare_and_admit(jid, "mine", env.pid, os.path.join(
        storage.data_dir_root(), "projects", env.pid, "source.mp4"), 30.0, 0.5, 3.5,
        "9:16", "480p", jdir, staging)
    job = _job(env, jid)
    assert job.status == "preparing" and job.provider_task_id is None
    assert not os.path.exists(os.path.join(jdir, "input.mp4"))
    with db_mod.get_session() as s:
        assert s.execute(select(ProviderTask)).first() is None


def test_claim_refuses_a_fourth_attempt(env):
    jid = _create(env).json()["id"]
    _sql("UPDATE reframe_jobs SET prep_attempts=3 WHERE id=:id", id=jid)
    prf.reframe_prepare(jid)
    assert _job(env, jid).status == "preparing"


def test_deleted_job_dir_is_not_resurrected(env):
    jid = _create(env).json()["id"]
    import shutil

    shutil.rmtree(storage.reframe_job_dir(jid, env.pid))
    prf.reframe_prepare(jid)
    assert not os.path.exists(storage.reframe_job_dir(jid, env.pid))


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def _age(jid, **cols):
    sets = ", ".join(f"{k}=:{k}" for k in cols)
    _sql(f"UPDATE reframe_jobs SET updated_at=:u{', ' + sets if sets else ''} WHERE id=:id",
         u=_utcnow() - timedelta(hours=1), id=jid, **cols)


def test_reaper_reenqueues_and_never_fails_a_live_claim(env):
    jid = _create(env).json()["id"]
    env.enq.clear()
    recovery._reconcile_reframe_jobs(set())
    assert env.enq == []                                    # inside the grace window
    _age(jid)
    recovery._reconcile_reframe_jobs({jid})
    assert env.enq == []                                    # a live RQ job
    recovery._reconcile_reframe_jobs(set())
    assert env.enq == [("prepare", jid)]
    env.enq.clear()
    _age(jid, prep_claim_owner="w", prep_attempts=3,
         prep_claim_expires_at=_utcnow() + timedelta(minutes=5))
    recovery._reconcile_reframe_jobs(set())
    assert env.enq == [] and _job(env, jid).status == "preparing"   # live claim survives
    _age(jid, prep_claim_expires_at=_utcnow() - timedelta(minutes=1))
    recovery._reconcile_reframe_jobs(set())
    job = _job(env, jid)
    assert (job.status, job.error_code) == ("prepare_failed", "prepare_crashed")


def test_reaper_reenqueues_qa(env):
    job = _admitted(env)
    _done(env, job)
    env.enq.clear()
    recovery._reconcile_reframe_jobs(set())
    assert ("qa", job.id) in env.enq
    env.enq.clear()
    _sql("UPDATE reframe_jobs SET qa_status='running', qa_claim_owner='w', "
         "qa_claim_expires_at=:x WHERE id=:id", x=_utcnow() + timedelta(minutes=5), id=job.id)
    recovery._reconcile_reframe_jobs(set())
    assert env.enq == []
    _sql("UPDATE reframe_jobs SET qa_claim_expires_at=:x WHERE id=:id",
         x=_utcnow() - timedelta(minutes=1), id=job.id)
    recovery._reconcile_reframe_jobs(set())
    assert ("qa", job.id) in env.enq


def test_render_is_not_in_the_unconditional_reaper():
    assert not any(s.model is ReframeRender for s in recovery._DERIVED_CREATIVE_SPECS)


def test_orphaned_render_is_reaped_to_failed(env):
    job = _admitted(env)
    with db_mod.get_session() as s:
        r = ReframeRender(job_id=job.id, status="processing", picture_end_sec=1, fade_sec=0,
                          black_tail_sec=0, language="en",
                          updated_at=_utcnow() - timedelta(hours=1))
        s.add(r)
        s.flush()
        rid = r.id
    recovery._reconcile_reframe_jobs({rid})
    with db_mod.get_session() as s:
        assert s.get(ReframeRender, rid).status == "processing"            # live job
    recovery._reconcile_reframe_jobs(set())
    with db_mod.get_session() as s:
        assert s.get(ReframeRender, rid).status == "failed"
        assert s.get(ProviderTask, job.provider_task_id).status == "queued"  # never touched


def test_render_reaper_cas_leaves_a_render_that_moved_on(env):
    job = _admitted(env)
    with db_mod.get_session() as s:
        r = ReframeRender(job_id=job.id, status="delivering", claim_owner="w",
                          picture_end_sec=1, fade_sec=0, black_tail_sec=0, language="en")
        s.add(r)
        s.flush()
        rid = r.id
    with db_mod.get_session() as s:
        seen = s.get(ReframeRender, rid).updated_at
    # Observed `delivering` by w; it finished before the reaper's write.
    _sql("UPDATE reframe_renders SET status='done' WHERE id=:id", id=rid)
    with db_mod.get_session() as s:
        assert prf.reap_orphaned_render(s, rid, "delivering", "w", seen, _utcnow()) is False
        assert s.get(ReframeRender, rid).status == "done"
    # Retried and claimed by a new worker: the old observation changes nothing.
    _sql("UPDATE reframe_renders SET status='processing', claim_owner='w2' WHERE id=:id", id=rid)
    with db_mod.get_session() as s:
        assert prf.reap_orphaned_render(s, rid, "processing", "w", seen, _utcnow()) is False
    # Two `queued` executions (both claim NULL): a retry re-stamps updated_at.
    _sql("UPDATE reframe_renders SET status='queued', claim_owner=NULL, updated_at=:u "
         "WHERE id=:id", u=_utcnow(), id=rid)
    with db_mod.get_session() as s:
        assert prf.reap_orphaned_render(s, rid, "queued", None, seen, _utcnow()) is False
        assert s.get(ReframeRender, rid).status == "queued"


# ---------------------------------------------------------------------------
# Deletion, re-analysis, runs
# ---------------------------------------------------------------------------


def test_delete_guards(env):
    r = _create(env)
    jid = r.json()["id"]
    assert env.client.delete(f"/api/v2/projects/{env.pid}").status_code == 409   # preparing
    assert env.client.delete(f"/api/v2/reframe-jobs/{jid}").status_code == 409
    prf.reframe_prepare(jid)
    job = _job(env, jid)
    assert env.client.delete(f"/api/v2/projects/{env.pid}").status_code == 409   # task active
    assert env.client.delete(f"/api/v2/reframe-jobs/{jid}").status_code == 409
    _done(env, job)
    with db_mod.get_session() as s:
        s.add(ReframeRender(id="r-active", job_id=jid, status="queued", picture_end_sec=1,
                            fade_sec=0, black_tail_sec=0, language="en"))
    assert env.client.delete(f"/api/v2/reframe-jobs/{jid}").status_code == 409   # render active
    assert env.client.delete(f"/api/v2/projects/{env.pid}").status_code == 409
    _sql("UPDATE reframe_renders SET status='failed' WHERE id='r-active'")
    _sql("UPDATE reframe_jobs SET qa_status='running', qa_claim_owner='w', qa_claim_expires_at=:x "
         "WHERE id=:id", x=_utcnow() + timedelta(minutes=5), id=jid)
    assert env.client.delete(f"/api/v2/reframe-jobs/{jid}").status_code == 409   # QA running
    _sql("UPDATE reframe_jobs SET qa_status='passed' WHERE id=:id", id=jid)
    assert env.client.delete(f"/api/v2/reframe-jobs/{jid}").status_code == 204
    assert not os.path.exists(storage.reframe_job_dir(jid, env.pid))
    with db_mod.get_session() as s:
        assert s.get(ProviderTask, job.provider_task_id) is not None  # the billing record stays
        assert s.get(ReframeRender, "r-active") is None
    assert env.client.delete(f"/api/v2/projects/{env.pid}").status_code == 204


def test_reanalysis_and_runs_refused(env):
    assert env.client.post(f"/api/v2/projects/{env.pid}/analyze").status_code == 409
    r = env.client.post(f"/api/v2/projects/{env.pid}/runs", data={"prompt": "x"})
    assert r.status_code == 409 and "no runs" in r.json()["detail"]


def test_disk_cleanup_never_touches_reframe(env):
    from app import disk_cleanup

    job = _admitted(env)
    _done(env, job)
    before = sorted(os.listdir(storage.reframe_job_dir(job.id, env.pid)))
    with db_mod.get_session() as s:
        disk_cleanup.sweep(s, now=_utcnow().replace(tzinfo=timezone.utc) + timedelta(days=365))
    assert sorted(os.listdir(storage.reframe_job_dir(job.id, env.pid))) == before


# ---------------------------------------------------------------------------
# Operator actions on the job's task
# ---------------------------------------------------------------------------


def test_task_actions(env):
    job = _admitted(env)
    url = f"/api/v2/reframe-jobs/{job.id}/task-actions"
    assert env.client.post(f"{url}/explode", data={"note": "x"}).status_code == 404
    assert env.client.post(f"{url}/abandon", data={"note": "  "}).status_code == 400
    r = env.client.post(f"{url}/resume", data={"note": "checked"})
    assert r.status_code == 409 and "needs_reconciliation" in r.json()["detail"]
    r = env.client.post(f"{url}/adopt", data={"note": "checked", "request_id": ""})
    assert r.status_code == 409  # adopt is only for submit_unknown
    r = env.client.post(f"{url}/abandon", data={"note": "nothing at Fal"})
    assert r.status_code == 200 and r.json()["status"] == "abandoned"
    with db_mod.get_session() as s:
        a = s.execute(select(ProviderTaskAction)).scalars().one()
        assert (a.actor, a.action, a.task_id) == ("dashboard", "abandon", job.provider_task_id)
    # The task id comes from the job; a job without a task has nothing to act on.
    jid = _create(env).json()["id"]
    assert env.client.post(f"/api/v2/reframe-jobs/{jid}/task-actions/close",
                           data={"note": "x"}).status_code == 409
    assert env.client.post("/api/v2/reframe-jobs/nope/task-actions/close",
                           data={"note": "x"}).status_code == 404


def test_action_buttons_follow_the_table():
    from app.api_reframe import task_actions_for

    class T:
        def __init__(self, status, held=False):
            self.status, self.reservation_held = status, held

    assert task_actions_for(T("submit_unknown")) == ["adopt", "abandon", "close"]
    assert task_actions_for(T("refused", True)) == ["abandon", "close"]
    assert task_actions_for(T("refused", False)) == []
    assert task_actions_for(T("needs_reconciliation")) == ["resume", "close"]
    assert task_actions_for(T("submitted")) == []


# ---------------------------------------------------------------------------
# QA and render
# ---------------------------------------------------------------------------


def test_qa_passes_and_render_delivers(env, monkeypatch):
    job = _admitted(env)
    # Render refused before the task is done, then before QA passed.
    assert env.client.post(f"/api/v2/reframe-jobs/{job.id}/renders",
                           data={"picture_end_sec": 2.0}).status_code == 409
    _done(env, job)
    assert env.client.post(f"/api/v2/reframe-jobs/{job.id}/renders",
                           data={"picture_end_sec": 2.0}).status_code == 409
    # The page kicks QA for a done task with QA pending.
    page = env.client.get(f"/v2/projects/{env.pid}/reframe-fragment")
    assert page.status_code == 200 and ("kick_qa", job.id) in env.enq
    prf.reframe_qa(job.id)
    job = _job(env, job.id)
    assert job.qa_status == "passed", job.job_qa
    assert job.job_qa["J2"]["ok"] and job.job_qa["J1"]["ok"]
    assert os.path.isfile(os.path.join(storage.reframe_job_dir(job.id, env.pid), "sheet.jpg"))
    page = env.client.get(f"/v2/projects/{env.pid}/reframe-fragment")
    assert "Ending" in page.text and "Render" in page.text

    d = prf.job_times(job)
    from app import reframe as rf

    dflt = rf.default_ending(d)
    # Validation and CTA aspect.
    bad = env.client.post(f"/api/v2/reframe-jobs/{job.id}/renders",
                          data={"picture_end_sec": 99, "audio_end_sec": 99})
    assert bad.status_code == 400
    with db_mod.get_session() as s:
        wide = LibraryClip(name="wide", local_path=env.media["cta_wide"], width=1344, height=768)
        tall_path = os.path.join(tempfile.mkdtemp(), "tall.mp4")
        with open(env.media["cta"], "rb") as a, open(tall_path, "wb") as b:
            b.write(a.read())
        tall = LibraryClip(name="tall", local_path=tall_path, width=768, height=1344)
        s.add_all([wide, tall])
        s.flush()
        wide_id, tall_id = wide.id, tall.id
    data = {"picture_end_sec": dflt.picture_end, "fade_sec": 0.5,
            "audio_end_sec": dflt.audio_end, "black_tail_sec": 0.25}
    assert env.client.post(f"/api/v2/reframe-jobs/{job.id}/renders",
                           data={**data, "library_clip_id": wide_id}).status_code == 400
    r = env.client.post(f"/api/v2/reframe-jobs/{job.id}/renders",
                        data={**data, "library_clip_id": tall_id})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    os.remove(tall_path)  # the library clip goes away; the render has its snapshot

    class FakeDrive:
        def __init__(self):
            self.calls = []

        def upload_file(self, path, folder, name):
            self.calls.append((path, name))
            return {"id": "drive-1", "webViewLink": "https://drive/x"}

    drive = FakeDrive()
    prf.reframe_render(rid, gdrive=drive)
    with db_mod.get_session() as s:
        rr = s.get(ReframeRender, rid)
        assert rr.status == "done", (rr.error_message, rr.qa)
        assert rr.gdrive_file_id == "drive-1" and rr.qa["ok"]
        assert drive.calls[0][1] == f"reframe_{job.id[:8]}_{rid[:8]}.mp4"
        assert rr.qa["final"]["frames"] >= rr.qa["frames"]["NL"] + 23
        assert (rr.result_width, rr.result_height) == (480, 832)
    # A second run of the same render is a no-op (only `queued` is picked up).
    prf.reframe_render(rid, gdrive=drive)
    assert len(drive.calls) == 1
    got = env.client.get(f"/api/v2/reframe-renders/{rid}/result")
    assert got.status_code == 200 and len(got.content) > 1000
    assert rr.qa["R2_final"]["ok"]
    rdir = storage.reframe_render_dir(rid, job.id, env.pid)
    assert sorted(os.listdir(rdir)) == ["cta.mp4", os.path.basename(rr.result_local_path)]


def test_superseded_render_worker_changes_nothing(env):
    """Reaped while its worker still ran, then retried: the old execution's
    token no longer owns the row, so it can neither deliver nor fail it."""
    job = _admitted(env)
    with db_mod.get_session() as s:
        r = ReframeRender(job_id=job.id, status="processing", claim_owner="old",
                          picture_end_sec=1, fade_sec=0, black_tail_sec=0, language="en")
        s.add(r)
        s.flush()
        rid = r.id
    _sql("UPDATE reframe_renders SET status='queued', claim_owner=NULL WHERE id=:id", id=rid)
    _sql("UPDATE reframe_renders SET status='processing', claim_owner='new' WHERE id=:id", id=rid)
    assert prf._set_render(rid, "processing", "old", status="delivering") is False
    assert prf._set_render(rid, "processing", "old", status="failed") is False
    with db_mod.get_session() as s:
        assert s.get(ReframeRender, rid).status == "processing"


def test_qa_fails_on_frame_count_mismatch(env):
    job = _admitted(env)
    _done(env, job, frames=round(job.input_duration_sec * 24) - 5)
    prf.reframe_qa(job.id)
    job = _job(env, job.id)
    assert job.qa_status == "failed" and not job.job_qa["J2"]["ok"]
    r = env.client.post(f"/api/v2/reframe-jobs/{job.id}/renders", data={"picture_end_sec": 1.0})
    assert r.status_code == 409


def test_two_qa_claims_one_wins(env):
    job = _admitted(env)
    _done(env, job)
    _sql("UPDATE reframe_jobs SET qa_status='running', qa_claim_owner='other', "
         "qa_claim_expires_at=:x WHERE id=:id", x=_utcnow() + timedelta(minutes=5), id=job.id)
    prf.reframe_qa(job.id)
    job = _job(env, job.id)
    assert job.qa_status == "running" and job.qa_claim_owner == "other"


def test_render_failure_and_retry(env):
    job = _admitted(env)
    _done(env, job)
    prf.reframe_qa(job.id)
    from app import reframe as rf

    dflt = rf.default_ending(prf.job_times(_job(env, job.id)))
    rid = env.client.post(f"/api/v2/reframe-jobs/{job.id}/renders",
                          data={"picture_end_sec": dflt.picture_end,
                                "audio_end_sec": dflt.audio_end}).json()["id"]
    os.remove(_task(env, job.provider_task_id).result_path)
    prf.reframe_render(rid)
    with db_mod.get_session() as s:
        assert s.get(ReframeRender, rid).status == "failed"
    r = env.client.post(f"/api/v2/reframe-renders/{rid}/retry")
    assert r.status_code == 200 and ("render", rid) in env.enq
    assert env.client.post(f"/api/v2/reframe-renders/{rid}/retry").status_code == 409
    assert env.client.delete(f"/api/v2/reframe-renders/{rid}").status_code == 409  # queued
