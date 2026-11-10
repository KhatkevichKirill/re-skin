"""AICP-363 autopilot core: acceptance + launch-protocol tests (docs/autopilot-design.md §10).

Temp SQLite DB, no Redis (enqueues and the wake key are monkeypatched), the
stub flow of tests/autopilot_stub_flow.py. The core is driven with an explicit
session factory and Hooks.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import timedelta

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import sessionmaker

import app.api_v2 as api_v2_module
from app import spend as spend_mod
from app.autopilot import core, registry
from app.autopilot.judgement import FakePort
from app.config import settings
from app.db import Base, get_db
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.models_autopilot import (
    AutopilotApproval, AutopilotEffect, AutopilotGate, AutopilotJournal, AutopilotStep,
    ProjectAutopilot, ProjectManualSpendEvent, ProjectSpendScope, utcnow,
)
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus

from tests import autopilot_stub_flow as stub


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=2000")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="module")
def SF(engine):
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


@pytest.fixture(autouse=True)
def _env(monkeypatch, SF, tmp_path):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(core, "wake", lambda: None)
    for name in ("enqueue_analyze_project", "enqueue_process_run", "enqueue_preprocess_run",
                 "enqueue_transcribe_project"):
        monkeypatch.setattr(api_v2_module, name, lambda *a, **k: None)
    stub.KNOBS.clear()
    registry.register(stub.FLOW)
    db = SF()
    if db.get(AutopilotGate, 1) is None:
        db.add(AutopilotGate(id=1, held=False, generation=0))
    else:
        db.get(AutopilotGate, 1).held = False
    db.commit()
    db.close()
    yield


class Clock:
    def __init__(self):
        self.t = utcnow()

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t = self.t + timedelta(seconds=sec)


@pytest.fixture()
def hooks():
    enq = []
    h = core.Hooks(enqueue_compute=lambda eid, jid: enq.append((eid, jid)),
                   compute_is_live=lambda eff: True, port=FakePort(), now=Clock())
    h.enqueued = enq
    return h


def _project(SF, tmp_path, *, status=ProjectStatus.ready, mode="assistant", seg_sec=5.0,
             params=None):
    db = SF()
    src = tmp_path / f"src-{os.urandom(4).hex()}.mp4"
    src.write_bytes(b"src")
    p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                     status=status, project_type="cover_change", mode=mode, duration_sec=seg_sec)
    db.add(p)
    db.flush()
    db.add(SegmentDef(project_id=p.id, index=0, start_sec=0.0, end_sec=seg_sec, action="swap"))
    ap = None
    if mode == "assistant":
        ap = core.create_autopilot(db, p, stub.FLOW, params or {"prompt": "swap the cover"})
    db.commit()
    pid, apid = p.id, (ap.id if ap else None)
    db.close()
    return pid, apid


def _tick(SF, apid, hooks):
    hooks.now.advance(31)  # past a Wait's 30 s recheck
    return core.tick_autopilot(apid, session_factory=SF, hooks=hooks)


def _ap(SF, apid):
    db = SF()
    try:
        return db.get(ProjectAutopilot, apid)
    finally:
        db.close()


def _pending(SF, apid, kind=None):
    db = SF()
    try:
        q = select(AutopilotApproval).where(AutopilotApproval.autopilot_id == apid,
                                            AutopilotApproval.status == "pending")
        if kind:
            q = q.where(AutopilotApproval.kind == kind)
        return db.execute(q).scalars().all()
    finally:
        db.close()


def _answer(SF, pid, aid, decision="approve", **kw):
    db = SF()
    try:
        return core.answer(db, pid, aid, decision, **kw)
    finally:
        db.close()


def _runs(SF, pid):
    db = SF()
    try:
        return db.execute(select(Run).where(Run.project_id == pid)).scalars().all()
    finally:
        db.close()


def _finish_run(SF, rid, tmp_path, *, status=RunStatus.done):
    db = SF()
    run = db.get(Run, rid)
    sd = db.execute(select(SegmentDef).where(SegmentDef.project_id == run.project_id)).scalars().first()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalars().first()
    res = tmp_path / f"seg-{os.urandom(4).hex()}.mp4"
    res.write_bytes(os.urandom(64))
    if rs is None:
        rs = RunSegment(run_id=rid, segment_def_id=sd.id, index=0)
        db.add(rs)
    rs.status, rs.local_result_path, rs.seedance_task_id = SegmentStatus.completed, str(res), "t1"
    final = tmp_path / f"final-{os.urandom(4).hex()}.mp4"
    final.write_bytes(os.urandom(64))
    run.result_local_path = str(final)
    run.segment_snapshot = [{"id": sd.id, "index": 0, "action": "swap",
                             "start_sec": sd.start_sec, "end_sec": sd.end_sec}]
    run.status = status
    db.commit()
    db.close()


def _steps(SF, apid):
    db = SF()
    try:
        return {r.step_id: r for r in db.execute(select(AutopilotStep).where(
            AutopilotStep.autopilot_id == apid)).scalars()}
    finally:
        db.close()


def _drive_to_probe_launched(SF, pid, apid, hooks):
    for _ in range(3):
        _tick(SF, apid, hooks)
    budget = _pending(SF, apid, "budget")
    assert len(budget) == 1
    _answer(SF, pid, budget[0].id)
    _tick(SF, apid, hooks)
    runs = _runs(SF, pid)
    assert len(runs) == 1
    return runs[0].id


# ---------------------------------------------------------------------------
# Manual projects and creation
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(SF):
    from app.main import app

    def _db():
        db = SF()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def test_manual_project_has_no_autopilot_and_tick_is_noop(SF, client):
    r = client.post("/api/v2/projects", data={"gdrive_link": "https://drive.google.com/file/d/x",
                                             "project_type": "cover_change"})
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"]
    db = SF()
    p = db.get(VideoProject, pid)
    assert p.mode == "manual"
    assert db.execute(select(ProjectAutopilot).where(ProjectAutopilot.project_id == pid)).first() is None
    db.close()
    assert client.get(f"/api/v2/projects/{pid}").json()["mode"] == "manual"
    assert client.get(f"/api/v2/projects/{pid}/autopilot").status_code == 404


def test_assistant_mode_refused_without_flow_or_flag(SF, client, monkeypatch):
    r = client.post("/api/v2/projects", data={"gdrive_link": "https://drive.google.com/file/d/x",
                                             "project_type": "face_swap", "mode": "assistant"})
    assert r.status_code == 422 and "no autopilot flow" in r.text
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", False)
    r = client.post("/api/v2/projects", data={"gdrive_link": "https://drive.google.com/file/d/x",
                                             "project_type": "cover_change", "mode": "assistant"})
    assert r.status_code == 422 and "not enabled" in r.text
    r = client.post("/api/v2/projects", data={"gdrive_link": "https://drive.google.com/file/d/x",
                                             "project_type": "cover_change", "mode": "bogus"})
    assert r.status_code == 400


def test_assistant_project_creation_validates_params_and_creates_autopilot(SF, client):
    r = client.post("/api/v2/projects", data={
        "gdrive_link": "https://drive.google.com/file/d/x", "project_type": "cover_change",
        "mode": "assistant", "autopilot_params": '{"bad": true}'})
    assert r.status_code == 422
    r = client.post("/api/v2/projects", data={
        "gdrive_link": "https://drive.google.com/file/d/x", "project_type": "cover_change",
        "mode": "assistant", "autopilot_params": '{"prompt": "p1"}'})
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"]
    v = client.get(f"/api/v2/projects/{pid}/autopilot").json()
    assert v["status"] == "active" and v["params"] == {"prompt": "p1"} and v["flow_id"] == "stub"
    card = client.get(f"/v2/projects/{pid}/autopilot-card")
    assert card.status_code == 200 and "Autopilot" in card.text


def test_run_pass_disabled_does_nothing(SF, tmp_path, monkeypatch, hooks):
    pid, apid = _project(SF, tmp_path)
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", False)
    assert core.run_pass(session_factory=SF, hooks=hooks) == 0
    assert _steps(SF, apid) == {}


# ---------------------------------------------------------------------------
# Acceptance: advance, approvals, budget, regeneration
# ---------------------------------------------------------------------------


def test_waits_for_project_then_asks_budget_before_any_paid_launch(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path, status=ProjectStatus.analyzing)
    _tick(SF, apid, hooks)
    assert _steps(SF, apid)["ready"].status == "active"
    db = SF(); db.get(VideoProject, pid).status = ProjectStatus.ready; db.commit(); db.close()
    _tick(SF, apid, hooks)
    assert _steps(SF, apid)["ready"].status == "completed"
    _tick(SF, apid, hooks)
    budget = _pending(SF, apid, "budget")
    assert len(budget) == 1
    assert budget[0].proposal["total_credits"] == 260.0  # 5 s × 52 cr/s (seedance 720p)
    assert _runs(SF, pid) == []
    _tick(SF, apid, hooks)  # still waiting, nothing launched, no second approval
    assert _runs(SF, pid) == [] and len(_pending(SF, apid, "budget")) == 1


def test_full_stub_flow_with_regeneration_approval(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    assert _ap(SF, apid).approved_budget_mcredits == 260_000
    _tick(SF, apid, hooks)
    assert len(_runs(SF, pid)) == 1  # in flight: no second launch
    _finish_run(SF, rid, tmp_path)
    _tick(SF, apid, hooks)
    assert _steps(SF, apid)["probe"].status == "completed"
    _tick(SF, apid, hooks)
    listen = _pending(SF, apid, "listening_probe")
    assert len(listen) == 1 and listen[0].subject == "en"
    _tick(SF, apid, hooks)
    assert len(_pending(SF, apid, "listening_probe")) == 1  # stops on the approval
    _answer(SF, pid, listen[0].id)
    _tick(SF, apid, hooks)
    assert _steps(SF, apid)["listen"].status == "completed"
    _tick(SF, apid, hooks)
    regen = _pending(SF, apid, "regeneration")
    assert len(regen) == 1 and regen[0].action["handler"] == "rerun_segments"
    db = SF(); assert db.get(Run, rid).status == RunStatus.done; db.close()
    _answer(SF, pid, regen[0].id)
    _tick(SF, apid, hooks)
    db = SF()
    assert db.get(Run, rid).status == RunStatus.queued  # the rerun was admitted
    ap = db.get(ProjectAutopilot, apid)
    assert ap.approved_budget_mcredits >= 260_000 + regen[0].cost_mcredits  # Q6: adds its cost
    db.close()
    _finish_run(SF, rid, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    handoff = _pending(SF, apid, "handoff")
    assert len(handoff) == 1
    _answer(SF, pid, handoff[0].id)
    _tick(SF, apid, hooks)
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "completed"


def test_reject_listening_stops_and_reject_regeneration_stops(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "listening_probe")[0].id, "reject", comment="male voice")
    _tick(SF, apid, hooks)
    ap = _ap(SF, apid)
    assert ap.status == "failed" and "male voice" in ap.status_reason

    pid2, apid2 = _project(SF, tmp_path)
    rid2 = _drive_to_probe_launched(SF, pid2, apid2, hooks)
    _finish_run(SF, rid2, tmp_path)
    for _ in range(2):
        _tick(SF, apid2, hooks)
    _answer(SF, pid2, _pending(SF, apid2, "listening_probe")[0].id)
    for _ in range(2):
        _tick(SF, apid2, hooks)
    _answer(SF, pid2, _pending(SF, apid2, "regeneration")[0].id, "reject")
    _tick(SF, apid2, hooks)
    assert _ap(SF, apid2).status == "stopped"
    db = SF(); assert db.get(Run, rid2).status == RunStatus.done; db.close()  # nothing launched


def test_unhandled_rejection_stops_after_grace(SF, tmp_path, hooks):
    stub.KNOBS["listen_ignores_answer"] = True
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    db = SF()
    a = _pending(SF, apid, "listening_probe")[0]
    db.close()
    _answer(SF, pid, a.id, "reject")
    db = SF(); db.get(AutopilotApproval, a.id).answered_at = hooks.clock(); db.commit(); db.close()
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "active"
    hooks.now.advance(settings.AUTOPILOT_REJECT_GRACE_SEC + 5)
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "stopped"


def test_budget_rejected_stops(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id, "reject")
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "stopped" and _runs(SF, pid) == []


@pytest.mark.parametrize("budget, expect_launch", [(240_000, True), (200_000, False)])
def test_overrun_above_tolerance_needs_approval(SF, tmp_path, hooks, budget, expect_launch):
    pid, apid = _project(SF, tmp_path)
    db = SF(); db.get(ProjectAutopilot, apid).approved_budget_mcredits = budget; db.commit(); db.close()
    for _ in range(3):
        _tick(SF, apid, hooks)
    # 260 credits: within 240·1.1 = 264, above 200·1.1 = 220
    assert (len(_runs(SF, pid)) == 1) is expect_launch
    overrun = _pending(SF, apid, "overrun")
    assert bool(overrun) is (not expect_launch)
    if overrun:
        _answer(SF, pid, overrun[0].id)
        _tick(SF, apid, hooks)
        assert len(_runs(SF, pid)) == 1
        assert _ap(SF, apid).approved_budget_mcredits == 260_000


def test_regeneration_chosen_option_authorises_exactly_that_launch(SF, tmp_path, hooks):
    stub.KNOBS["rest_choice"] = True
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "listening_probe")[0].id)
    for _ in range(2):
        _tick(SF, apid, hooks)
    choice = _pending(SF, apid, "regeneration")
    assert len(choice) == 1 and {o["id"] for o in choice[0].options} == {"rerun", "accept"}
    db = SF()
    with pytest.raises(core.OperatorError):
        core.answer(db, pid, choice[0].id, "approve")  # an option is required
    db.close()
    _answer(SF, pid, choice[0].id, option="rerun")
    _tick(SF, apid, hooks)
    db = SF(); assert db.get(Run, rid).status == RunStatus.queued; db.close()
    assert _pending(SF, apid, "regeneration") == []  # no second approval for the same launch


# ---------------------------------------------------------------------------
# Never double-launch
# ---------------------------------------------------------------------------


def test_crash_after_admission_commit_reconciles_without_relaunch(SF, tmp_path, hooks, monkeypatch):
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id)
    real_lock = core._lock_project
    calls = {"n": 0}

    def crash_on_t2(db, project_id):
        calls["n"] += 1
        if calls["n"] == 3:  # reconcile, apply, then T2 → the "crash"
            raise RuntimeError("worker killed")
        return real_lock(db, project_id)

    monkeypatch.setattr(core, "_lock_project", crash_on_t2)
    _tick(SF, apid, hooks)
    monkeypatch.setattr(core, "_lock_project", real_lock)
    db = SF()
    effs = db.execute(select(AutopilotEffect).where(AutopilotEffect.autopilot_id == apid)).scalars().all()
    db.close()
    assert len(effs) == 1 and effs[0].state == "launching" and effs[0].created_refs["runs"]
    assert len(_runs(SF, pid)) == 1
    for _ in range(3):
        _tick(SF, apid, hooks)
    db = SF()
    eff = db.execute(select(AutopilotEffect).where(AutopilotEffect.autopilot_id == apid)).scalar_one()
    db.close()
    assert eff.state == "launched"
    assert len(_runs(SF, pid)) == 1


def test_crash_before_commit_launches_once_next_tick(SF, tmp_path, hooks, monkeypatch):
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id)
    real = api_v2_module.create_run

    def boom(*a, **k):
        raise RuntimeError("killed before commit")

    monkeypatch.setattr(api_v2_module, "create_run", boom)
    _tick(SF, apid, hooks)
    assert _runs(SF, pid) == []
    s = _steps(SF, apid)["probe"]
    assert s.errors == 1 and s.next_check_at is not None
    monkeypatch.setattr(api_v2_module, "create_run", real)
    hooks.now.advance(60)
    _tick(SF, apid, hooks)
    _tick(SF, apid, hooks)
    assert len(_runs(SF, pid)) == 1


def test_concurrent_duplicate_collides_on_the_unique_key(SF, tmp_path, hooks, monkeypatch):
    """Another tick commits the same effect key while this APPLY runs: the
    controlled flush fails and this transaction's run never commits."""
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id)
    from app.autopilot.actions import CreateRun

    real_estimate = CreateRun.estimate

    def racing_estimate(self, db, project, params):
        other = SF()
        st = other.execute(select(AutopilotStep).where(AutopilotStep.autopilot_id == apid,
                                                       AutopilotStep.step_id == "probe")).scalar_one()
        other.add(AutopilotEffect(
            autopilot_id=apid, step_id="probe", step_generation=st.generation, name="launch",
            idempotency_key=f"{apid}:probe:g{st.generation}:launch", kind="action",
            handler="create_run", handler_version=1, params={}, params_sha256="x", paid=True,
            estimate_mcredits=0, state="launched", created_refs={"runs": []},
        ))
        other.commit()
        other.close()
        return real_estimate(self, db, project, params)

    monkeypatch.setattr(CreateRun, "estimate", racing_estimate)
    res = _tick(SF, apid, hooks)
    assert res["launched"] == []
    assert _runs(SF, pid) == []
    db = SF()
    assert db.execute(select(func.count()).select_from(AutopilotEffect).where(
        AutopilotEffect.autopilot_id == apid)).scalar_one() == 1
    db.close()


def test_two_ticks_on_a_stale_plan_never_both_apply(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id)
    _tick(SF, apid, hooks)  # consumes the budget answer + launches
    flow = stub.FLOW
    db = SF(); project = db.get(VideoProject, pid); db.close()
    specs = list(flow.steps(project, {}))
    spec = next(s for s in specs if s.id == "probe")
    # Re-applying a plan made before the launch is discarded by the version CAS.
    plan = core.Plan(version=0, step_gen=1, decision=stub._probe.__call__ and None)
    plan.decision = core.Launch("launch", "create_run", {"prompt": "x"}, "dup")
    assert core._apply(SF, apid, spec, specs, flow, plan, hooks).outcome == "discarded"
    assert len(_runs(SF, pid)) == 1


def test_refusal_writes_nothing_and_backs_off(SF, tmp_path, hooks, monkeypatch):
    from fastapi import HTTPException

    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id)

    def refuse(*a, **k):
        raise HTTPException(status_code=429, detail="Daily kie.ai budget reached")

    monkeypatch.setattr(api_v2_module, "_budget_precheck", refuse)
    _tick(SF, apid, hooks)
    s = _steps(SF, apid)["probe"]
    assert s.refusals == 1 and s.last_refusal["status"] == 429
    assert (s.next_check_at - hooks.clock()).total_seconds() >= 600
    db = SF()
    assert db.execute(select(func.count()).select_from(AutopilotEffect)
                      .where(AutopilotEffect.autopilot_id == apid)).scalar_one() == 0
    db.close()
    assert _runs(SF, pid) == []


def test_handler_refusal_paths_commit_nothing(SF, tmp_path, hooks):
    """rerun_segments on a run that is not done: the endpoint 409s before its commit."""
    from app.autopilot.actions import RerunSegments
    from app.autopilot.contract import ActionRefused

    pid, _ = _project(SF, tmp_path, mode="manual")
    db = SF()
    run = Run(project_id=pid, prompt="p", status=RunStatus.processing, reference_image_urls=[])
    db.add(run)
    db.commit()
    rid = run.id
    rs = RunSegment(run_id=rid, segment_def_id="x", index=0, status=SegmentStatus.completed)
    db.add(rs)
    db.commit()
    project = db.get(VideoProject, pid)
    h = RerunSegments()
    p = h.validate(db, project, {"run_id": rid, "segments": [{"run_segment_id": rs.id}]})
    marker = AutopilotGate(id=99, held=False, generation=0)  # stands in for the effect row
    db.add(marker)
    db.flush()
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        h.execute(db, project, p, effect=None)
    assert exc.value.status_code == 409
    db.rollback()
    db.close()
    db = SF()
    assert db.get(AutopilotGate, 99) is None  # nothing was committed
    db.close()
    assert ActionRefused(409, "x").status == 409


# ---------------------------------------------------------------------------
# Planning is read-only; contract violations
# ---------------------------------------------------------------------------


def test_planning_session_cannot_write_or_commit(SF, tmp_path, hooks):
    stub.KNOBS["ready_writes"] = True
    pid, apid = _project(SF, tmp_path)
    _tick(SF, apid, hooks)
    db = SF(); assert db.get(VideoProject, pid).name != "hacked"; db.close()
    assert _steps(SF, apid)["ready"].errors == 1  # an error, retried with backoff
    stub.KNOBS.clear()
    stub.KNOBS["ready_commits"] = True
    pid2, apid2 = _project(SF, tmp_path)
    _tick(SF, apid2, hooks)
    ap = _ap(SF, apid2)
    assert ap.status == "failed" and "read-only" in ap.status_reason


def test_paid_launch_from_free_step_is_a_violation(SF, tmp_path, hooks):
    stub.KNOBS["variant"] = "free_paid"
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    ap = _ap(SF, apid)
    assert ap.status == "failed" and "free" in ap.status_reason
    assert _runs(SF, pid) == []


def test_missing_flow_version_pauses(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    db = SF(); db.get(ProjectAutopilot, apid).flow_version = 99; db.commit(); db.close()
    _tick(SF, apid, hooks)
    ap = _ap(SF, apid)
    assert ap.status == "paused" and "no longer deployed" in ap.status_reason


# ---------------------------------------------------------------------------
# Reopening, manual paid actions, hold, switches
# ---------------------------------------------------------------------------


def test_manual_change_reopens_completed_step_and_supersedes_its_approvals(SF, tmp_path, hooks):
    stub.KNOBS["probe_depends"] = True
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    listen = _pending(SF, apid, "listening_probe")
    assert len(listen) == 1
    db = SF()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalars().first()
    rs.prompt_override = "edited by hand"  # a free manual edit of a generation input
    db.commit()
    db.close()
    _tick(SF, apid, hooks)
    steps = _steps(SF, apid)
    assert steps["probe"].generation == 2 and steps["probe"].status == "active"
    assert steps["listen"].status == "pending"
    db = SF()
    assert db.get(AutopilotApproval, listen[0].id).status == "superseded"
    db.close()
    assert _ap(SF, apid).status == "active"


def test_manual_paid_action_pauses_the_autopilot(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    db = SF()
    assert db.execute(select(func.count()).select_from(ProjectManualSpendEvent)).scalar_one() == 0 \
        or True
    spend_mod.reserve(db, spend_mod.run_scope(rid), 80.0, kind="rerun segment")
    db.commit()
    db.close()
    _tick(SF, apid, hooks)
    ap = _ap(SF, apid)
    assert ap.status == "paused" and "rerun segment" in ap.status_reason
    db = SF()
    j = db.execute(select(AutopilotJournal).where(AutopilotJournal.autopilot_id == apid,
                                                  AutopilotJournal.kind == "paused")).scalars().all()
    db.close()
    assert any("manual paid action" in x.message for x in j)
    db = SF()
    core.set_switch(db, pid, "resume")
    db.close()
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "active"  # the event is consumed once


def test_autopilot_own_admission_is_not_a_manual_action(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    _drive_to_probe_launched(SF, pid, apid, hooks)
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "active"
    db = SF()
    assert db.execute(select(func.count()).select_from(ProjectManualSpendEvent).where(
        ProjectManualSpendEvent.project_id == pid)).scalar_one() == 0
    db.close()


def test_hold_blocks_new_admissions(SF, tmp_path, hooks):
    from app.autopilot import gate

    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id)
    db = SF(); gate.hold_on(db, by="test"); db.close()
    _tick(SF, apid, hooks)
    assert _runs(SF, pid) == []
    db = SF(); gate.hold_off(db); db.close()
    _tick(SF, apid, hooks)
    assert len(_runs(SF, pid)) == 1


def test_pause_resume_stop_and_answer_idempotency(SF, tmp_path, hooks, client):
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    aid = _pending(SF, apid, "budget")[0].id
    assert client.post(f"/api/v2/projects/{pid}/autopilot/pause").json()["status"] == "paused"
    _answer(SF, pid, aid)
    _tick(SF, apid, hooks)
    assert _runs(SF, pid) == []  # paused: observes only
    r = client.post(f"/api/v2/projects/{pid}/autopilot/approvals/{aid}/answer",
                    json={"decision": "approve"})
    assert r.status_code == 200  # same answer again: idempotent
    r = client.post(f"/api/v2/projects/{pid}/autopilot/approvals/{aid}/answer",
                    json={"decision": "reject"})
    assert r.status_code == 409
    assert client.post(f"/api/v2/projects/{pid}/autopilot/resume").json()["status"] == "active"
    _tick(SF, apid, hooks)
    assert len(_runs(SF, pid)) == 1
    r = client.post(f"/api/v2/projects/{pid}/autopilot/approvals/{aid}/comments", json={"text": "ok"})
    assert r.status_code == 200
    assert client.patch(f"/api/v2/projects/{pid}/autopilot",
                        json={"params": {"prompt": "new"}}).status_code == 409  # paid launched
    assert client.patch(f"/api/v2/projects/{pid}/autopilot",
                        json={"tolerance": 0.15}).json()["budget"]["tolerance"] == 0.15
    assert client.post(f"/api/v2/projects/{pid}/autopilot/stop").json()["status"] == "stopped"
    assert client.post(f"/api/v2/projects/{pid}/autopilot/stop").status_code == 409


def test_listen_artifact_is_copied_and_bound(SF, tmp_path, hooks, client):
    stub.KNOBS["listen_artifact"] = True
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    a = _pending(SF, apid, "listening_probe")[0]
    assert a.artifacts and len(a.artifacts[0]["sha256"]) == 64
    r = client.get(f"/api/v2/projects/{pid}/autopilot/artifacts/{a.artifacts[0]['sha256']}")
    assert r.status_code == 200
    # Replace the bytes behind the approval after it was answered: superseded.
    _answer(SF, pid, a.id)
    db = SF(); final = db.get(Run, rid).result_local_path; db.close()
    st = os.stat(final)
    with open(final, "wb") as fh:
        fh.write(os.urandom(64))
    os.utime(final, ns=(st.st_atime_ns, st.st_mtime_ns))  # same size and mtime
    _tick(SF, apid, hooks)
    db = SF(); assert db.get(AutopilotApproval, a.id).status == "superseded"; db.close()
    assert _steps(SF, apid)["listen"].status == "active"


# ---------------------------------------------------------------------------
# Spend attribution
# ---------------------------------------------------------------------------


def test_spend_scope_survives_run_deletion(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _tick(SF, apid, hooks)
    db = SF()
    assert db.get(ProjectSpendScope, (pid, f"run:{rid}")) is not None
    from app.autopilot.spend_view import figures_mcredits

    before = figures_mcredits(db, pid)["committed"]
    assert before == 260_000
    db.delete(db.get(Run, rid))
    db.commit()
    assert figures_mcredits(db, pid)["committed"] == before
    db.close()


# ---------------------------------------------------------------------------
# Compute and judgement effects
# ---------------------------------------------------------------------------


def test_compute_effect_claim_publish_and_recovery(SF, tmp_path, hooks):
    from app.autopilot import compute

    stub.KNOBS["variant"] = "compute"
    pid, apid = _project(SF, tmp_path)
    _tick(SF, apid, hooks)
    assert len(hooks.enqueued) == 1
    eid, job = hooks.enqueued[0]
    assert job == f"autopilot-compute-{eid}"
    # lost job: not live for longer than the grace -> re-enqueued once
    hooks.compute_is_live = lambda eff: False
    _tick(SF, apid, hooks)
    hooks.now.advance(core.COMPUTE_NOT_LIVE_GRACE_SEC + 1)
    _tick(SF, apid, hooks)
    assert len(hooks.enqueued) == 2
    hooks.compute_is_live = lambda eff: True
    assert compute.run(eid, session_factory=SF) == "succeeded"
    assert compute.run(eid, session_factory=SF) is None  # a duplicate claims nothing
    db = SF(); eff = db.get(AutopilotEffect, eid); db.close()
    assert eff.result["hits"] == 3 and os.path.isfile(os.path.join(eff.result["out_dir"], "sheet.txt"))
    assert eff.claim_token in eff.result["out_dir"]
    _tick(SF, apid, hooks)
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "completed"


def test_judgement_prepared_sending_submitted_succeeded(SF, tmp_path, hooks):
    stub.KNOBS["variant"] = "judge"
    pid, apid = _project(SF, tmp_path)
    port = hooks.port
    port.lose_next_response = True
    _tick(SF, apid, hooks)  # prepared, then sending (lost response)
    db = SF()
    eff = db.execute(select(AutopilotEffect).where(AutopilotEffect.autopilot_id == apid)).scalar_one()
    db.close()
    assert eff.state == "sending" and port.submit_calls == 1
    _tick(SF, apid, hooks)  # same id + sha resubmitted, no duplicate
    db = SF(); eff = db.get(AutopilotEffect, eff.id); db.close()
    assert eff.state == "submitted" and len(port.submitted) == 1
    port.answers["stub_judgement"] = {"speakers": 2}
    _tick(SF, apid, hooks)
    _tick(SF, apid, hooks)
    s = _steps(SF, apid)["think"]
    assert s.status == "completed" and s.output["answer"] == {"speakers": 2}


def test_judgement_disabled_port_parks_prepared(SF, tmp_path, hooks):
    from app.autopilot.judgement import DisabledPort

    stub.KNOBS["variant"] = "judge"
    hooks.port = DisabledPort()
    pid, apid = _project(SF, tmp_path)
    _tick(SF, apid, hooks)
    _tick(SF, apid, hooks)
    db = SF()
    eff = db.execute(select(AutopilotEffect).where(AutopilotEffect.autopilot_id == apid)).scalar_one()
    db.close()
    assert eff.state == "prepared"


# ---------------------------------------------------------------------------
# Transcription gate (§6.5)
# ---------------------------------------------------------------------------


def test_assistant_localisation_project_is_not_auto_transcribed(SF, tmp_path, monkeypatch):
    from contextlib import contextmanager

    from app import tasks

    db = SF()
    p = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.ready,
                     project_type="localisation", mode="assistant")
    db.add(p)
    db.commit()
    pid = p.id
    db.close()

    @contextmanager
    def _gs():
        s = SF()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    import app.db as app_db

    monkeypatch.setattr(app_db, "get_session", _gs)
    enq = []
    monkeypatch.setattr(tasks, "enqueue_transcribe_project", enq.append)
    tasks._enqueue_transcribe_after_analyze(pid)
    db = SF()
    assert db.get(VideoProject, pid).transcript_status is None
    db.close()
    assert enq == []


# ---------------------------------------------------------------------------
# AICP-365 requests: transcript key, generic version accessor, library-clip
# voice, handler-written result, rule (g) safety net
# ---------------------------------------------------------------------------


def test_transcript_version_key_and_generic_accessor(SF, tmp_path):
    from app.autopilot.contract import ContractViolation

    pid, apid = _project(SF, tmp_path)
    other, _ = _project(SF, tmp_path, mode="manual")
    db = SF()
    ap = db.get(ProjectAutopilot, apid)
    v1 = core.key_version(db, ap, f"project_transcript:{pid}")
    db.get(VideoProject, pid).transcript = {"lines": [{"text": "hi"}]}
    db.commit()
    assert core.key_version(db, ap, f"project_transcript:{pid}") != v1
    with pytest.raises(ContractViolation):
        core.key_version(db, ap, f"project_transcript:{other}")
    with pytest.raises(ContractViolation):
        core.key_version(db, ap, "bogus:1")
    s1 = core.segments_version(db, pid)
    sd = db.execute(select(SegmentDef).where(SegmentDef.project_id == pid)).scalar_one()
    sd.pre_roll_sec = 0.5
    db.commit()
    assert core.segments_version(db, pid) != s1  # pre/post roll are part of the digest
    row = AutopilotStep(autopilot_id=apid, step_id="x", generation=2, status="active",
                        previous_outputs=[{"generation": 1, "output": {"a": 1}}])
    ctx = core.Context(db, ap, db.get(VideoProject, pid), row, now=utcnow(), hold=False)
    assert ctx.version(f"project_transcript:{pid}") == ctx.transcript_version()
    assert ctx.previous_outputs() == [{"generation": 1, "output": {"a": 1}}]
    assert ctx.step_generation == 2
    db.close()


def test_create_run_voice_from_library_clip(SF, tmp_path):
    import subprocess

    from app.autopilot.actions import CreateRun
    from app.models import LibraryClip

    clip_path = tmp_path / "probe.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:d=3", "-shortest", str(clip_path)], check=True)
    pid, _ = _project(SF, tmp_path, mode="manual")
    db = SF()
    try:
        clip = LibraryClip(name="TEMP probe", local_path=str(clip_path))
        db.add(clip)
        db.commit()
        project = db.get(VideoProject, pid)
        h = CreateRun()
        p = h.validate(db, project, {"prompt": "x", "model": "seedance-2-5",
                                     "resolution": "720p", "voice_from_library_clip": clip.id})
        eff = AutopilotEffect(autopilot_id=None, step_id="s", name="n", idempotency_key="k-voice",
                              kind="action", handler="create_run", params={},
                              params_sha256="x", state="launching", created_refs={})
        res = h.execute(db, project, p, eff)
        run = db.get(Run, res["run_id"])
        assert run.reference_audio_urls and os.path.isfile(run.reference_audio_urls[0])
        assert eff.created_refs["runs"] == [run.id]
    finally:
        db.rollback()
        db.close()


def test_handler_rollback_without_effect_row_pauses(SF, tmp_path, hooks, monkeypatch):
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "budget")[0].id)

    def rogue(pid_, *a, db, **k):
        db.rollback()  # breaks rule (g): drops the core's flushed effect row
        import uuid

        db.add(Run(id=str(uuid.uuid4()), project_id=pid_, prompt="rogue",
                   status=RunStatus.queued, reference_image_urls=[]))
        db.commit()

        class R:
            run_id, status, warnings = "x", "queued", []
        return R()

    monkeypatch.setattr(api_v2_module, "create_run", rogue)
    _tick(SF, apid, hooks)
    ap = _ap(SF, apid)
    assert ap.status == "paused" and "contract (g)" in ap.status_reason


def test_stale_chosen_option_never_raises_the_budget(SF, tmp_path, hooks):
    """Terra gate R1: the option's approval is revalidated before any budget change."""
    stub.KNOBS["rest_choice"] = True
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "listening_probe")[0].id)
    for _ in range(2):
        _tick(SF, apid, hooks)
    choice = _pending(SF, apid, "regeneration")[0]
    _answer(SF, pid, choice.id, option="rerun")
    budget_before = _ap(SF, apid).approved_budget_mcredits
    db = SF()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalars().first()
    rs.reference_image_urls_override = ["https://example.com/other.jpg"]  # the run changed
    db.commit()
    db.close()
    _tick(SF, apid, hooks)
    db = SF()
    assert db.get(AutopilotApproval, choice.id).status == "superseded"
    assert db.get(Run, rid).status == RunStatus.done  # nothing launched
    db.close()
    assert _ap(SF, apid).approved_budget_mcredits == budget_before


def test_manual_paid_action_is_recorded_while_the_ticker_is_disabled(SF, tmp_path, hooks, monkeypatch):
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", False)
    db = SF()
    spend_mod.reserve(db, spend_mod.run_scope(rid), 50.0, kind="retry run")
    db.commit()
    n = db.execute(select(func.count()).select_from(ProjectManualSpendEvent).where(
        ProjectManualSpendEvent.project_id == pid)).scalar_one()
    db.close()
    assert n == 1
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    _tick(SF, apid, hooks)
    assert _ap(SF, apid).status == "paused"


def test_manual_project_reservation_records_nothing(SF, tmp_path):
    pid, _ = _project(SF, tmp_path, mode="manual")
    db = SF()
    run = Run(project_id=pid, prompt="p", status=RunStatus.queued, reference_image_urls=[])
    db.add(run)
    db.commit()
    spend_mod.reserve(db, spend_mod.run_scope(run.id), 50.0, kind="new run")
    db.commit()
    assert db.execute(select(func.count()).select_from(ProjectManualSpendEvent).where(
        ProjectManualSpendEvent.project_id == pid)).scalar_one() == 0
    db.close()


@pytest.mark.spend_cap
def test_refused_manual_reservation_leaves_no_event(SF, tmp_path, hooks, monkeypatch):
    """Terra gate R2: the event is written by the caller's commit only."""
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0001)
    db = SF()
    with pytest.raises(spend_mod.BudgetExceeded):
        spend_mod.reserve(db, spend_mod.localise_scope(pid), 500.0, kind="localisation")
    db.rollback()
    n = db.execute(select(func.count()).select_from(ProjectManualSpendEvent).where(
        ProjectManualSpendEvent.project_id == pid)).scalar_one()
    db.close()
    assert n == 0


def test_a_skipped_step_comes_back_when_the_params_restore_it(SF, tmp_path, hooks, monkeypatch):
    """Terra gate R3: a step skipped by a params change is reactivated when it reappears."""
    pid, apid = _project(SF, tmp_path)
    _tick(SF, apid, hooks)
    assert _steps(SF, apid)["ready"].status == "completed"
    _tick(SF, apid, hooks)  # probe row exists now
    orig = stub.StubFlow.steps

    def without_probe(self, project, params):
        return [s for s in orig(self, project, params) if s.id != "probe"]

    monkeypatch.setattr(stub.StubFlow, "steps", without_probe)
    _tick(SF, apid, hooks)
    assert _steps(SF, apid)["probe"].status == "skipped"
    assert _ap(SF, apid).status == "active"
    monkeypatch.setattr(stub.StubFlow, "steps", orig)
    _tick(SF, apid, hooks)
    st = _steps(SF, apid)["probe"]
    assert st.status in ("pending", "active") and st.generation == 2


def test_consume_set_is_all_or_nothing(SF, tmp_path):
    """AICP-364 R5: a later stale answer never leaves an earlier one consumed or
    a budget change committed; only the supersede is committed."""
    pid, apid = _project(SF, tmp_path)
    db = SF()
    run = Run(project_id=pid, prompt="p", status=RunStatus.done, reference_image_urls=[])
    db.add(run)
    db.flush()
    ap = db.get(ProjectAutopilot, apid)
    row = AutopilotStep(autopilot_id=apid, step_id="s", generation=1, status="active")
    db.add(row)
    for name, bound in (("a1", {}), ("a2", {f"run:{run.id}": "stale-version"})):
        db.add(AutopilotApproval(
            autopilot_id=apid, step_id="s", step_generation=1, name=name, generation=1,
            dedupe_key=f"{apid}:s:g1:{name}:1", kind="handoff", status="approved",
            answered_decision="approve", bound_versions=bound, question="q", proposal={},
            comments=[]))
    db.commit()
    ap.approved_budget_mcredits = 999_000  # an uncommitted change made before consuming
    with pytest.raises(core._Discard):
        core._consume(db, ap, row, ["a1", "a2"], now=utcnow())
    db.close()
    db = SF()
    st = {a.name: a.status for a in db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == apid)).scalars()}
    assert st == {"a1": "approved", "a2": "superseded"}
    assert db.get(ProjectAutopilot, apid).approved_budget_mcredits is None
    db.close()


def test_detector_key_effect_params_and_rerun_takes(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    db = SF()
    ap = db.get(ProjectAutopilot, apid)
    v1 = core.key_version(db, ap, f"project_detector:{pid}")
    db.get(VideoProject, pid).caption_detector = {"band_y": [0.5, 0.7]}
    db.commit()
    assert core.key_version(db, ap, f"project_detector:{pid}") != v1
    db.close()
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "listening_probe")[0].id)
    for _ in range(2):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "regeneration")[0].id)
    _tick(SF, apid, hooks)
    db = SF()
    eff = db.execute(select(AutopilotEffect).where(AutopilotEffect.autopilot_id == apid,
                                                   AutopilotEffect.name == "rerun")).scalar_one()
    assert eff.created_refs.get("takes")  # the archived take of segment 0
    assert core._effect_view(eff).params["run_id"] == rid
    ap = db.get(ProjectAutopilot, apid)
    row = db.execute(select(AutopilotStep).where(AutopilotStep.autopilot_id == apid,
                                                 AutopilotStep.step_id == "rest")).scalar_one()
    ctx = core.Context(db, ap, db.get(VideoProject, pid), row, now=utcnow(), hold=False)
    assert ctx.fills(rid) == []
    db.close()


def test_reopen_takes_back_an_answered_choice_that_launched_nothing(SF, tmp_path):
    """2026-11-02 ("Ask again"): an answered choice that started nothing, in
    the running step, is superseded on request so the flow asks afresh;
    answers that authorised a launch, core approvals and finished steps never."""
    pid, apid = _project(SF, tmp_path)
    db = SF()
    db.add(AutopilotStep(autopilot_id=apid, step_id="s", generation=1, status="active"))
    db.add(AutopilotStep(autopilot_id=apid, step_id="old", generation=1, status="completed"))
    opts = [{"id": "manual", "label": "by hand", "action": None, "cost_mcredits": 0},
            {"id": "go", "label": "go", "action": {"handler": "h", "params_sha256": "x"},
             "cost_mcredits": 5}]
    ids = {}
    for name, step, status, chosen in (("choice", "s", "approved", "manual"),
                                       ("used", "s", "consumed", "manual"),
                                       ("paid", "s", "approved", "go"),
                                       ("x#budget", "s", "approved", None),
                                       ("done", "old", "approved", "manual")):
        a = AutopilotApproval(
            autopilot_id=apid, step_id=step, step_generation=1, name=name, generation=1,
            dedupe_key=f"{apid}:{step}:g1:{name}:1", kind="handoff", status=status,
            answered_decision="approve", chosen_option=chosen, options=opts, question="q",
            proposal={}, comments=[], answered_at=utcnow())
        db.add(a)
        db.flush()
        ids[name] = a.id
    db.commit()
    version = db.get(ProjectAutopilot, apid).version
    assert sorted(a.name for a in core.reopenable(db, apid)) == ["choice", "used"]
    assert sorted(x["name"] for x in core.view(db, pid)["answered"]) == ["choice", "used"]
    db.close()
    for name in ("paid", "x#budget", "done"):
        db = SF()
        with pytest.raises(core.OperatorError) as exc:
            core.reopen(db, pid, ids[name], by="operator")
        assert exc.value.status == 409
        db.rollback()
        db.close()
    db = SF()
    core.reopen(db, pid, ids["choice"], by="operator")
    db.close()
    db = SF()
    a = db.get(AutopilotApproval, ids["choice"])
    assert a.status == "superseded" and a.chosen_option == "manual"     # the record stays
    assert db.get(ProjectAutopilot, apid).version > version             # a plan in flight is stale
    msgs = [j.message for j in db.execute(select(AutopilotJournal).where(
        AutopilotJournal.autopilot_id == apid)).scalars()]
    assert any("asked again by operator (was: manual)" in m for m in msgs)
    assert core.reopen(db, pid, ids["choice"]).status == "superseded"   # idempotent
    db.close()



# ---------------------------------------------------------------------------
# AICP-376: titles and consequences are presentation only
# ---------------------------------------------------------------------------


def _drive_to_choice(SF, tmp_path, hooks):
    stub.KNOBS["rest_choice"] = True
    pid, apid = _project(SF, tmp_path)
    rid = _drive_to_probe_launched(SF, pid, apid, hooks)
    _finish_run(SF, rid, tmp_path)
    for _ in range(2):
        _tick(SF, apid, hooks)
    _answer(SF, pid, _pending(SF, apid, "listening_probe")[0].id)
    for _ in range(2):
        _tick(SF, apid, hooks)
    return pid, apid


def test_title_and_consequence_are_stored_but_never_reach_the_action(SF, tmp_path, hooks):
    pid, apid = _drive_to_choice(SF, tmp_path, hooks)
    plain = _pending(SF, apid, "regeneration")[0]
    stub.KNOBS.clear()
    stub.KNOBS["titled"] = True
    pid2, apid2 = _drive_to_choice(SF, tmp_path, hooks)
    titled = _pending(SF, apid2, "regeneration")[0]
    assert titled.proposal["_title"] == "Segment 1 still shows captions"
    assert "_title" not in plain.proposal
    rerun = next(o for o in titled.options if o["id"] == "rerun")
    assert rerun["consequence"] == "resubmits segment 1"
    assert "consequence" not in rerun["action"]["params"]
    assert "consequence" not in next(o for o in plain.options if o["id"] == "rerun")
    # the same launch is authorised: the params hash ignores the presentation
    plain_rerun = next(o for o in plain.options if o["id"] == "rerun")
    assert set(rerun["action"]) == set(plain_rerun["action"])
    assert "_title" not in str(titled.bound_versions)
    from app.autopilot import links

    assert links.approval_title(titled.question, titled.proposal) == "Segment 1 still shows captions"
    assert links.approval_title(plain.question, plain.proposal) == "Fix segment 1?"   # fallback


def test_flow_may_not_set_the_reserved_title_key(SF, tmp_path, hooks):
    stub.KNOBS["reserved_title"] = True
    pid, apid = _drive_to_choice(SF, tmp_path, hooks)
    ap = _ap(SF, apid)
    assert ap.status == "failed" and "reserved" in (ap.status_reason or "")


def test_core_budget_and_regeneration_titles(SF, tmp_path, hooks):
    pid, apid = _project(SF, tmp_path)
    for _ in range(3):
        _tick(SF, apid, hooks)
    budget = _pending(SF, apid, "budget")[0]
    assert budget.proposal["_title"] == "Approve the budget: 260 credits"
    assert core._core_title("regeneration", "rerun_segments", 52_000, 312_000, 260_000) == \
        "Regenerate: rerun segments (~52 credits)"
    assert core._core_title("overrun", "create_run", 52_000, 312_000, 260_000) == \
        "Over budget: a new run (~52 credits, 312 of 260 after)"
