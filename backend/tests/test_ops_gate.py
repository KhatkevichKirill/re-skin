"""AICP-350 deploy admission gate on SQLite: 503 while paused, the drain
status endpoint + banner, counted status for analysis admissions,
Idempotency-Key replay/conflict, the autopilot view of the gate, and the
recovery grace for just-admitted analyses.

Row-lock behaviour (begin waits for in-flight admissions, duplicate keys wait
for each other) needs real PostgreSQL: tests/test_ops_gate_pg.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

_db_fd, _db_path = tempfile.mkstemp(suffix="_ops_gate.db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select, update
from sqlalchemy.orm import sessionmaker

import app.api_v2 as api_v2_module
from app import ops_gate, ops_status
from app.db import Base, get_db
from app.models import Run, RunCaption, RunTailEdit, VideoProject
from app.ops_gate import ApiIdempotencyKey, OpsAdmissionGate
from app.state_machine import CaptionStatus, ProjectStatus, RunStatus, TailEditStatus


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="module")
def SF(engine):
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def db(SF):
    s = SF()
    yield s
    s.rollback()
    s.close()


@pytest.fixture(autouse=True)
def _gate_open(SF):
    with SF() as s:
        row = s.get(OpsAdmissionGate, 1)
        if row is None:
            s.add(OpsAdmissionGate(id=1, paused=False, generation=0))
        else:
            row.paused, row.owner, row.reason, row.paused_at = False, None, None, None
        s.query(ApiIdempotencyKey).delete()
        s.commit()
    yield


def _pause(SF, *, minutes_ago: float = 1.0, owner="deploy:test", reason="unit test"):
    with SF() as s:
        row = s.get(OpsAdmissionGate, 1)
        row.paused, row.owner, row.reason = True, owner, reason
        row.generation = (row.generation or 0) + 1
        row.paused_at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        s.commit()
        return row.generation


@pytest.fixture()
def calls(monkeypatch):
    seen: dict[str, list] = {"caption": [], "tail": [], "analyze": []}
    monkeypatch.setattr(api_v2_module, "enqueue_process_caption", lambda cid: seen["caption"].append(cid))
    monkeypatch.setattr(api_v2_module, "enqueue_process_tail_edit", lambda eid: seen["tail"].append(eid))
    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project",
                        lambda pid, **kw: seen["analyze"].append((pid, kw)))
    return seen


@pytest.fixture()
def client(SF, calls, monkeypatch):
    from app.main import app

    def override_get_db():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    @contextmanager
    def _test_get_session():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(api_v2_module, "get_session", _test_get_session)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def _done_run(db, tmp_dir) -> Run:
    p = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.ready)
    db.add(p)
    db.commit()
    from app import storage

    root = storage.data_dir_root()
    os.makedirs(root, exist_ok=True)
    result = os.path.join(tempfile.mkdtemp(dir=root), "final.mp4")
    with open(result, "wb") as fh:
        fh.write(b"\0" * 64)
    r = Run(project_id=p.id, prompt="x", resolution="720p", status=RunStatus.done,
            reference_image_urls=[], result_local_path=result, result_duration_sec=10.0)
    db.add(r)
    db.commit()
    return r


# ---------------------------------------------------------------------------
# 503 while paused
# ---------------------------------------------------------------------------


def test_paused_gate_refuses_admission_with_503_and_writes_nothing(client, db, SF, calls, tmp_path):
    run = _done_run(db, tmp_path)
    _pause(SF)
    r = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"})
    assert r.status_code == 503
    assert r.headers.get("retry-after") == str(ops_gate.RETRY_AFTER_SEC)
    assert "Deploy in progress" in r.json()["detail"]
    db.expire_all()
    assert db.execute(select(RunCaption).where(RunCaption.run_id == run.id)).first() is None
    assert calls["caption"] == []
    # Reads keep working while paused.
    assert client.get(f"/api/v2/runs/{run.id}/captions").status_code == 200


def test_open_gate_admits(client, db, calls, tmp_path):
    run = _done_run(db, tmp_path)
    r = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"})
    assert r.status_code in (200, 201), r.text
    assert calls["caption"] == [r.json()["caption_id"]]


def test_ungated_route_still_works_while_paused(client, db, SF, tmp_path):
    run = _done_run(db, tmp_path)
    _pause(SF)
    r = client.patch(f"/api/v2/runs/{run.id}", json={"name": "renamed"})
    assert r.status_code == 200, r.text


def test_project_create_paused_creates_nothing(client, db, SF, calls):
    _pause(SF)
    before = db.execute(select(VideoProject.id)).all()
    r = client.post("/api/v2/projects", data={"gdrive_link": "https://drive.google.com/file/d/abc/view"})
    assert r.status_code == 503
    db.expire_all()
    assert db.execute(select(VideoProject.id)).all() == before
    assert calls["analyze"] == []


# ---------------------------------------------------------------------------
# Counted status for analysis admissions (design §2)
# ---------------------------------------------------------------------------


def test_project_create_commits_analyzing_so_the_drain_counts_it(client, db, calls):
    r = client.post("/api/v2/projects", data={"gdrive_link": "https://drive.google.com/file/d/abc/view"})
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"]
    assert r.json()["status"] == "analyzing"
    db.expire_all()
    assert db.get(VideoProject, pid).status == ProjectStatus.analyzing
    assert calls["analyze"] == [(pid, {})]
    assert ops_status.counts(db)["active"]["by_kind"].get("projects", 0) >= 1


def test_reanalysis_commits_analyzing_and_a_second_click_409s(client, db, calls):
    p = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.ready)
    db.add(p)
    db.commit()
    r = client.post(f"/api/v2/projects/{p.id}/analyze")
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "analyzing"
    db.expire_all()
    assert db.get(VideoProject, p.id).status == ProjectStatus.analyzing
    assert client.post(f"/api/v2/projects/{p.id}/analyze").status_code == 409
    assert calls["analyze"] == [(p.id, {})]


def test_analyze_job_accepts_an_already_analyzing_project(db, monkeypatch, tmp_path):
    """The worker side of the counted status: a project the admission already
    put in `analyzing` is analyzed, not refused by analyzing -> analyzing."""
    from app import pipeline_v2

    p = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.analyzing,
                     source_local_path=str(tmp_path / "missing.mp4"))
    db.add(p)
    db.commit()

    @contextmanager
    def _gs():
        s = db.get_bind()
        from sqlalchemy.orm import Session

        sess = Session(bind=s, expire_on_commit=False)
        try:
            yield sess
            sess.commit()
        finally:
            sess.close()

    monkeypatch.setattr(pipeline_v2, "get_session", _gs)
    # The source file is missing, so the job fails AFTER taking the status: it
    # must reach `failed` (FileNotFoundError), not die on InvalidTransition.
    with pytest.raises(FileNotFoundError):
        pipeline_v2.analyze_project(p.id)
    db.expire_all()
    assert db.get(VideoProject, p.id).status == ProjectStatus.failed


# ---------------------------------------------------------------------------
# GET /api/v2/ops/admission + banner
# ---------------------------------------------------------------------------


def test_admission_endpoint_open(client, monkeypatch):
    monkeypatch.setattr(ops_status, "rq_diagnostic", lambda: {"status": "unknown"})
    body = client.get("/api/v2/ops/admission").json()
    assert body["admitting"] is True and body["banner"] is None
    assert body["marker"]["paused"] is False
    assert isinstance(body["active"]["total"], int)
    assert "parked" in body and "build" in body and body["rq"] == {"status": "unknown"}
    assert body["build"]["code_sha256"]


def test_admission_endpoint_paused_and_banner_turns_red(client, SF, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(ops_status, "rq_diagnostic", lambda: {"status": "unknown"})
    monkeypatch.setattr(settings, "OPS_GATE_WARN_AFTER_MIN", 45)
    gen = _pause(SF, minutes_ago=5)
    body = client.get("/api/v2/ops/admission").json()
    assert body["admitting"] is False
    assert body["banner"]["level"] == "info" and "command" not in body["banner"]
    html = client.get("/v2/ops-gate-banner").text
    assert "Deploy in progress since" in html and "force-end" not in html

    with SF() as s:
        s.get(OpsAdmissionGate, 1).paused_at = datetime.now(timezone.utc) - timedelta(minutes=50)
        s.commit()
    b = client.get("/api/v2/ops/admission").json()["banner"]
    assert b["level"] == "red"
    assert b["command"] == ops_gate.force_end_command(gen)
    assert f"-v gen={gen}" in b["command"] and "force-end.sql" in b["command"]
    html = client.get("/v2/ops-gate-banner").text
    assert "force-end.sql" in html and f"gen={gen}" in html


def test_banner_fragment_empty_while_admitting(client):
    assert client.get("/v2/ops-gate-banner").text == ""


def test_parked_provider_task_is_not_active(db):
    from app.provider_models import ProviderTask

    cols = {c.name for c in ProviderTask.__table__.columns}
    assert "status" in cols
    before = ops_status.counts(db)
    assert before["parked"]["total"] == 0
    # Only the SQL split is under test here: count with a raw row.
    db.execute(ProviderTask.__table__.insert().values(**_provider_row(db, "needs_reconciliation")))
    db.commit()
    after = ops_status.counts(db)
    assert after["parked"]["by_kind"] == {"provider_tasks": 1}
    assert after["active"]["by_kind"].get("provider_tasks", 0) == before["active"]["by_kind"].get("provider_tasks", 0)
    db.execute(ProviderTask.__table__.delete())
    db.commit()


def _provider_row(db, status):
    """A minimal provider_tasks row: fill every NOT NULL column without a
    default with a type-appropriate dummy."""
    import uuid as _uuid

    from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text

    from app.provider_models import ProviderTask

    out = {}
    for c in ProviderTask.__table__.columns:
        if c.name == "status":
            out[c.name] = status
            continue
        if c.nullable or c.default is not None or c.server_default is not None:
            continue
        t = c.type
        if isinstance(t, (Integer,)):
            out[c.name] = 1
        elif isinstance(t, Float):
            out[c.name] = 0.0
        elif isinstance(t, Boolean):
            out[c.name] = False
        elif isinstance(t, DateTime):
            out[c.name] = datetime.now(timezone.utc)
        elif isinstance(t, JSON):
            out[c.name] = {}
        else:
            out[c.name] = str(_uuid.uuid4())[: getattr(t, "length", None) or 36]
    return out


# ---------------------------------------------------------------------------
# Idempotency-Key (design §3)
# ---------------------------------------------------------------------------


def test_keyed_caption_create_replays_the_same_caption(client, db, calls, tmp_path):
    run = _done_run(db, tmp_path)
    h = {"Idempotency-Key": "flow:abc:captions:1"}
    r1 = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"}, headers=h)
    assert r1.status_code in (200, 201), r1.text
    # The caption moves on (a replay must not re-validate business state).
    db.expire_all()
    cap = db.get(RunCaption, r1.json()["caption_id"])
    cap.status = CaptionStatus.transcribing
    db.commit()
    r2 = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"}, headers=h)
    assert r2.status_code in (200, 201), r2.text
    assert r2.json()["caption_id"] == r1.json()["caption_id"]
    assert r2.json()["status"] == "transcribing"
    db.expire_all()
    assert len(db.execute(select(RunCaption).where(RunCaption.run_id == run.id)).all()) == 1
    assert calls["caption"] == [r1.json()["caption_id"]]  # enqueued once
    key = db.get(ApiIdempotencyKey, "flow:abc:captions:1")
    assert key.operation == {"resource_kind": "run_caption", "resource_ids": [r1.json()["caption_id"]]}
    assert key.route == "POST /api/v2/runs/{run_id}/captions"


def test_same_key_different_request_is_409(client, db, tmp_path):
    run = _done_run(db, tmp_path)
    h = {"Idempotency-Key": "k-409"}
    assert client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"}, headers=h).status_code in (200, 201)
    r = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "es"}, headers=h)
    assert r.status_code == 409
    # Same key on another route is a different request too (route in the hash).
    cap = db.execute(select(RunCaption).where(RunCaption.run_id == run.id)).scalars().first()
    cap.status = CaptionStatus.failed
    db.commit()
    assert client.post(f"/api/v2/captions/{cap.id}/retry", headers=h).status_code == 409


def test_replay_of_a_deleted_resource_is_410(client, db, tmp_path):
    run = _done_run(db, tmp_path)
    h = {"Idempotency-Key": "k-410"}
    r1 = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"}, headers=h)
    cid = r1.json()["caption_id"]
    db.execute(RunCaption.__table__.delete().where(RunCaption.id == cid))
    db.commit()
    r2 = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"}, headers=h)
    assert r2.status_code == 410
    assert r2.json()["detail"]["resource_ids"] == [cid]


def test_refused_admission_leaves_no_key_so_a_resend_executes(client, db, tmp_path, calls):
    run = _done_run(db, tmp_path)
    run.status = RunStatus.failed
    db.commit()
    h = {"Idempotency-Key": "k-rollback"}
    assert client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"}, headers=h).status_code == 409
    db.expire_all()
    assert db.get(ApiIdempotencyKey, "k-rollback") is None
    run = db.get(Run, run.id)
    run.status = RunStatus.done
    db.commit()
    r = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"}, headers=h)
    assert r.status_code in (200, 201), r.text


def test_keyed_retry_and_tail_edit_replay(client, db, calls, tmp_path):
    run = _done_run(db, tmp_path)
    cap = RunCaption(run_id=run.id, status=CaptionStatus.failed, source_type="run")
    db.add(cap)
    db.commit()
    h = {"Idempotency-Key": "k-retry"}
    r1 = client.post(f"/api/v2/captions/{cap.id}/retry", headers=h)
    assert r1.status_code == 200, r1.text
    r2 = client.post(f"/api/v2/captions/{cap.id}/retry", headers=h)  # caption is queued: replay, not 409
    assert r2.status_code == 200 and r2.json()["id"] == cap.id
    assert calls["caption"] == [cap.id]

    h = {"Idempotency-Key": "k-tail"}
    t1 = client.post(f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": "5"}, headers=h)
    assert t1.status_code == 201, t1.text
    t2 = client.post(f"/api/v2/runs/{run.id}/tail-edits", data={"cut_sec": "5"}, headers=h)
    assert t2.json()["tail_edit_id"] == t1.json()["tail_edit_id"]
    assert calls["tail"] == [t1.json()["tail_edit_id"]]


def test_key_with_file_part_is_refused(client, db, tmp_path):
    run = _done_run(db, tmp_path)
    r = client.post(f"/api/v2/runs/{run.id}/copy", headers={"Idempotency-Key": "k-file"},
                    files={"reference_files": ("a.jpg", b"\xff\xd8\xff", "image/jpeg")})
    assert r.status_code == 400 and "file parts" in r.json()["detail"]


def test_bad_key_is_400(client, db, tmp_path):
    run = _done_run(db, tmp_path)
    r = client.post(f"/api/v2/runs/{run.id}/captions", data={"language": "en"},
                    headers={"Idempotency-Key": "x" * 129})
    assert r.status_code == 400


def test_in_process_call_ignores_the_admission_marker(db, calls, tmp_path):
    """The autopilot calls endpoint functions directly: the ADMISSION default
    is a Depends marker there, never an Admission — no key, no crash."""
    run = _done_run(db, tmp_path)
    resp = api_v2_module.create_caption(run.id, language="en", source_tail_edit_id=None,
                                        placement=None, db=db)
    assert resp.caption_id and calls["caption"] == [resp.caption_id]


def test_request_fingerprint_is_canonical():
    a = ops_gate.request_fingerprint("POST /x/{id}", {"id": "1"}, [("b", "2"), ("a", "1")],
                                     {"json": {"y": 1, "x": [1, {"q": 2, "p": 1}]}})
    b = ops_gate.request_fingerprint("POST /x/{id}", {"id": "1"}, [("a", "1"), ("b", "2")],
                                     {"json": {"x": [1, {"p": 1, "q": 2}], "y": 1}})
    assert a == b
    assert a != ops_gate.request_fingerprint("POST /y/{id}", {"id": "1"}, [], None)


def test_purge_idempotency_keys_keeps_young_keys_and_live_resources(db, tmp_path):
    run = _done_run(db, tmp_path)
    live = RunCaption(run_id=run.id, status=CaptionStatus.queued, source_type="run")
    done = RunCaption(run_id=run.id, status=CaptionStatus.done, source_type="run")
    db.add_all([live, done])
    db.commit()
    old = datetime.now(timezone.utc) - timedelta(days=31)

    def key(k, op, created=old):
        db.add(ApiIdempotencyKey(key=k, route="r", request_sha256="0" * 64,
                                 operation=op, created_at=created))

    key("old-done", {"resource_kind": "run_caption", "resource_ids": [done.id]})
    key("old-gone", {"resource_kind": "run_caption", "resource_ids": ["nope"]})
    key("old-live", {"resource_kind": "run_caption", "resource_ids": [live.id]})
    key("old-unknown-kind", {"resource_kind": "mystery", "resource_ids": ["x"]})
    key("new", {"resource_kind": "run_caption", "resource_ids": [done.id]},
        created=datetime.now(timezone.utc))
    db.commit()
    key("ancient-gone", {"resource_kind": "run_caption", "resource_ids": ["nope2"]},
        created=datetime.now(timezone.utc) - timedelta(days=91))
    db.commit()
    # The done caption finished just now: retention counts from its terminal
    # change, so even its 31-day-old key stays; a deleted resource's key is
    # never purged (its replay keeps answering 410).
    assert ops_gate.purge_idempotency_keys(db) == 0
    left = {k for (k,) in db.execute(select(ApiIdempotencyKey.key))}
    assert left == {"old-done", "old-gone", "ancient-gone", "old-live", "old-unknown-kind", "new"}
    db.execute(update(RunCaption).where(RunCaption.id == done.id).values(
        updated_at=datetime.utcnow() - timedelta(days=31)))
    db.commit()
    assert ops_gate.purge_idempotency_keys(db) == 1
    left = {k for (k,) in db.execute(select(ApiIdempotencyKey.key))}
    assert left == {"old-gone", "ancient-gone", "old-live", "old-unknown-kind", "new"}


# ---------------------------------------------------------------------------
# Autopilot + recovery
# ---------------------------------------------------------------------------


def test_autopilot_gate_reads_the_global_gate_without_writing_its_own_row(SF):
    from app.autopilot import core
    from app.models_autopilot import AutopilotGate

    with SF() as s:
        g = core._gate(s, share=True)
        assert g.held is False
        s.rollback()
    _pause(SF)
    with SF() as s:
        g = core._gate(s, share=True)
        assert g.held is True
        s.commit()
    with SF() as s:
        assert s.get(AutopilotGate, 1).held is False  # never written
    assert core._gate_held(SF) is True


def test_recovery_leaves_a_just_admitted_analysis_alone(db, monkeypatch):
    from app import recovery

    p = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.analyzing)
    old = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.analyzing)
    db.add_all([p, old])
    db.commit()
    db.execute(update(VideoProject).where(VideoProject.id == old.id).values(
        updated_at=datetime.utcnow() - timedelta(seconds=recovery._DERIVED_ORPHAN_GRACE_SEC + 60)))
    db.commit()

    @contextmanager
    def _gs():
        from sqlalchemy.orm import Session

        sess = Session(bind=db.get_bind(), expire_on_commit=False)
        try:
            yield sess
            sess.commit()
        finally:
            sess.close()

    monkeypatch.setattr(recovery, "get_session", _gs)
    monkeypatch.setattr(recovery, "_live_run_ids", lambda *a, **k: set())
    monkeypatch.setattr(recovery, "_reap_orphaned_derived_creatives", lambda live: {})
    monkeypatch.setattr(recovery, "_reconcile_reframe_jobs", lambda live: {})
    recovery._reconcile_orphaned_runs_impl(redis_conn=None)
    db.expire_all()
    assert db.get(VideoProject, p.id).status == ProjectStatus.analyzing
    assert db.get(VideoProject, old.id).status == ProjectStatus.failed
    # cleanup so other tests' counts stay small
    db.execute(VideoProject.__table__.delete().where(VideoProject.id.in_([p.id, old.id])))
    db.commit()


def test_recovery_reaps_no_project_when_the_fresh_scan_fails(db, monkeypatch):
    from app import recovery

    old = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.analyzing)
    db.add(old)
    db.commit()
    db.execute(update(VideoProject).where(VideoProject.id == old.id).values(
        updated_at=datetime.utcnow() - timedelta(hours=1)))
    db.commit()

    @contextmanager
    def _gs():
        from sqlalchemy.orm import Session

        sess = Session(bind=db.get_bind(), expire_on_commit=False)
        try:
            yield sess
            sess.commit()
        finally:
            sess.close()

    calls = {"n": 0}

    def _live(*a, **k):
        calls["n"] += 1
        if k.get("strict"):
            raise RuntimeError("redis down")
        return set()

    monkeypatch.setattr(recovery, "get_session", _gs)
    monkeypatch.setattr(recovery, "_live_run_ids", _live)
    monkeypatch.setattr(recovery, "_reap_orphaned_derived_creatives", lambda live: {})
    monkeypatch.setattr(recovery, "_reconcile_reframe_jobs", lambda live: {})
    recovery._reconcile_orphaned_runs_impl(redis_conn=None)
    db.expire_all()
    assert db.get(VideoProject, old.id).status == ProjectStatus.analyzing
    db.execute(VideoProject.__table__.delete().where(VideoProject.id == old.id))
    db.commit()


def test_ops_status_matches_drain_check_sql():
    """deploy/drain-check.sql (read by scripts/deploy.sh while the API may be
    down) and app/ops_status.py (the endpoint) must count the same thing."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "drain-check.sql")
    if not os.path.exists(path):
        pytest.skip("repo root not mounted")
    body = [l.strip() for l in open(path).read().splitlines()
            if l.strip() and not l.lstrip().startswith("--") and not l.lstrip().startswith("\\")]
    rows = {}
    import re as _re

    for line in body:
        m = _re.match(r"(?:UNION ALL )?SELECT '(active|parked)'(?: AS state)?, '([a-z_]+)', count\(\*\) (.+?);?$", line)
        assert m, f"unexpected drain-check line: {line}"
        rows[(m.group(1), m.group(2))] = m.group(3)
    expected = {("active", k): ops_status.drain_check_fragment(f) for k, f in ops_status.ACTIVE}
    expected.update({("parked", k): ops_status.drain_check_fragment(f) for k, f in ops_status.PARKED})
    assert rows == expected


def test_failed_localisation_analysis_settles_its_promised_transcript(db):
    from app import pipeline_v2

    p = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.failed,
                     project_type="localisation", transcript_status="pending")
    db.add(p)
    db.commit()
    assert pipeline_v2.settle_unstarted_transcript(db, p) is True
    assert p.transcript_status == "failed" and "re-analyze" in p.transcript_error.lower()
    db.rollback()
    db.execute(VideoProject.__table__.delete().where(VideoProject.id == p.id))
    db.commit()


def test_reaper_rechecks_under_the_row_lock_and_settles_the_transcript(db, monkeypatch):
    """A stale candidate whose project was re-admitted meanwhile (fresh
    `analyzing`) is left alone; a reaped localisation project's promised
    transcript is settled so the drain does not count it forever."""
    from app import recovery

    old_ts = datetime.utcnow() - timedelta(seconds=recovery._DERIVED_ORPHAN_GRACE_SEC + 60)
    loc = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.analyzing,
                       project_type="localisation", transcript_status="pending")
    readmitted = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.analyzing)
    db.add_all([loc, readmitted])
    db.commit()
    db.execute(update(VideoProject).where(VideoProject.id.in_([loc.id, readmitted.id]))
               .values(updated_at=old_ts))
    db.commit()

    @contextmanager
    def _gs():
        from sqlalchemy.orm import Session

        sess = Session(bind=db.get_bind(), expire_on_commit=False)
        try:
            yield sess
            sess.commit()
        finally:
            sess.close()

    def _live(*a, **k):
        if k.get("strict"):
            # The fresh scan runs after the select: the analysis finished and a
            # new one was admitted for `readmitted` in between.
            with _gs() as s2:
                s2.execute(update(VideoProject).where(VideoProject.id == readmitted.id)
                           .values(updated_at=datetime.utcnow()))
        return set()

    monkeypatch.setattr(recovery, "get_session", _gs)
    monkeypatch.setattr(recovery, "_live_run_ids", _live)
    monkeypatch.setattr(recovery, "_reap_orphaned_derived_creatives", lambda live: {})
    monkeypatch.setattr(recovery, "_reconcile_reframe_jobs", lambda live: {})
    recovery._reconcile_orphaned_runs_impl(redis_conn=None)
    db.expire_all()
    assert db.get(VideoProject, readmitted.id).status == ProjectStatus.analyzing
    reaped = db.get(VideoProject, loc.id)
    assert reaped.status == ProjectStatus.failed and reaped.transcript_status == "failed"
    db.execute(VideoProject.__table__.delete().where(VideoProject.id.in_([loc.id, readmitted.id])))
    db.commit()


def test_uncounted_jobs_do_not_start_while_paused(db, SF, monkeypatch, tmp_path):
    """A Drive re-link / caption-placement scan has no running status, so it
    must not START between a deploy's begin and its up -d: paused -> deferred,
    nothing stamped, nothing done."""
    from app import caption_placement, drive_relink, tasks
    from app.models import RunDriveRelink

    @contextmanager
    def _gs():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    deferred = []
    monkeypatch.setattr(tasks, "defer_uncounted_job",
                        lambda q, func, *args, job_timeout: deferred.append((q, func, args)))
    monkeypatch.setattr(drive_relink, "get_session", _gs)
    monkeypatch.setattr("app.db.get_session", _gs)

    run = _done_run(db, tmp_path)
    row = RunDriveRelink(run_id=run.id, status="queued", old_file_id="o", target_file_id="t",
                         result_path="/x", run_status="done")
    db.add(row)
    db.commit()
    db.expire_all()
    stamp_before = db.get(RunDriveRelink, row.id).updated_at  # as read back from the DB
    _pause(SF)
    assert drive_relink.process_relink(row.id, gdrive=object()) == "deferred"
    assert caption_placement.scan_and_store(run.project_id, 1) is None
    assert [d[1] for d in deferred] == ["app.tasks.run_drive_relink", "app.tasks.run_caption_placement_scan"]
    db.expire_all()
    r = db.get(RunDriveRelink, row.id)
    assert r.status == "queued" and r.updated_at == stamp_before
    assert not db.get(VideoProject, run.project_id).caption_placement
    db.execute(RunDriveRelink.__table__.delete())
    db.commit()


def test_purge_pages_past_kept_keys(db):
    """Kept keys (deleted resources) must not starve later eligible keys out
    of a fixed-size batch."""
    old = datetime.now(timezone.utc) - timedelta(days=40)
    for i in range(3):
        db.add(ApiIdempotencyKey(key=f"tomb-{i}", route="r", request_sha256="0" * 64,
                                 operation={"resource_kind": "run_caption", "resource_ids": [f"gone-{i}"]},
                                 created_at=old - timedelta(minutes=10 - i)))
    db.add(ApiIdempotencyKey(key="eligible", route="r", request_sha256="0" * 64,
                             operation=None, created_at=old))
    db.commit()
    assert ops_gate.purge_idempotency_keys(db, limit=2) == 1
    assert db.get(ApiIdempotencyKey, "eligible") is None
    db.query(ApiIdempotencyKey).delete()
    db.commit()
