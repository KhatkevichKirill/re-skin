"""AICP-363 autopilot on real PostgreSQL row locks (docs/autopilot-design.md §10).

Opt in with AUTOPILOT_TEST_POSTGRES_URL pointing at a THROWAWAY database
(never production). Each test gets its own schema with the full metadata.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

_OPT_IN = "AUTOPILOT_TEST_POSTGRES_URL"
pytestmark = pytest.mark.skipif(not os.environ.get(_OPT_IN),
                                reason=f"opt in with {_OPT_IN}; throwaway test DB only")


@pytest.fixture
def pg(monkeypatch, tmp_path):
    import app.api_v2 as api_v2_module
    from app.autopilot import core, registry
    from app.config import settings
    from app.db import Base
    from tests import autopilot_stub_flow as stub

    url = os.environ[_OPT_IN]
    assert "re-skin-db" not in url and "@db" not in url, "never the production database"
    schema = "ap_" + uuid.uuid4().hex[:12]
    admin = create_engine(url, poolclass=NullPool)
    with admin.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, poolclass=NullPool, connect_args={
        "options": f"-c search_path={schema} -c lock_timeout=10000"})
    Base.metadata.create_all(engine)
    SF = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0)
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(core, "wake", lambda: None)
    for name in ("enqueue_analyze_project", "enqueue_process_run", "enqueue_preprocess_run"):
        monkeypatch.setattr(api_v2_module, name, lambda *a, **k: None)
    stub.KNOBS.clear()
    registry.register(stub.FLOW)
    from app.models_autopilot import AutopilotGate

    with SF.begin() as db:
        db.add(AutopilotGate(id=1, held=False, generation=0))
    try:
        yield SF
    finally:
        engine.dispose()
        with admin.begin() as c:
            c.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _setup(SF, tmp_path):
    from app.autopilot import core
    from app.models import SegmentDef, VideoProject
    from app.state_machine import ProjectStatus
    from tests import autopilot_stub_flow as stub

    db = SF()
    p = VideoProject(source_type="upload", source_ref="t.mp4", status=ProjectStatus.ready,
                     project_type="cover_change", mode="assistant", duration_sec=5.0)
    db.add(p)
    db.flush()
    db.add(SegmentDef(project_id=p.id, index=0, start_sec=0.0, end_sec=5.0, action="swap"))
    ap = core.create_autopilot(db, p, stub.FLOW, {"prompt": "x"})
    db.commit()
    ids = (p.id, ap.id)
    db.close()
    return ids


def _hooks():
    from app.autopilot import core
    from app.autopilot.judgement import FakePort
    from app.models_autopilot import utcnow

    state = {"t": utcnow()}

    def now():
        return state["t"]

    h = core.Hooks(enqueue_compute=lambda *a: None, compute_is_live=lambda e: True,
                   port=FakePort(), now=now)
    h.state = state
    return h


def _advance(h, sec=31):
    from datetime import timedelta

    h.state["t"] = h.state["t"] + timedelta(seconds=sec)


def test_concurrent_applies_of_one_plan_admit_once(pg, tmp_path):
    from app.autopilot import core
    from app.models import Run, VideoProject
    from app.models_autopilot import AutopilotApproval
    from tests import autopilot_stub_flow as stub

    SF = pg
    pid, apid = _setup(SF, tmp_path)
    h = _hooks()
    for _ in range(3):
        _advance(h)
        core.tick_autopilot(apid, session_factory=SF, hooks=h)
    db = SF()
    a = db.execute(select(AutopilotApproval).where(AutopilotApproval.kind == "budget")).scalar_one()
    db.close()
    db = SF(); core.answer(db, pid, a.id, "approve"); db.close()
    _advance(h)
    core._reconcile(SF, apid, h)  # consumes the budget answer
    db = SF(); project = db.get(VideoProject, pid); db.close()
    specs = list(stub.FLOW.steps(project, {}))
    spec = next(s for s in specs if s.id == "probe")
    plan = core._plan(SF, apid, spec, specs, stub.FLOW, now=h.clock(), hold=False)
    results, barrier = [], threading.Barrier(2)

    def go():
        barrier.wait()
        results.append(core._apply(SF, apid, spec, specs, stub.FLOW, plan, h).outcome)

    ts = [threading.Thread(target=go) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    db = SF()
    n = db.execute(select(func.count()).select_from(Run).where(Run.project_id == pid)).scalar_one()
    db.close()
    assert sorted(results) == ["discarded", "launched"]
    assert n == 1


def test_unique_effect_key_rolls_back_the_losers_run(pg, tmp_path):
    """The DB property the launch protocol relies on: the second transaction
    blocks on the unique index, then fails, and its run never commits."""
    from app.models import Run
    from app.models_autopilot import AutopilotEffect

    SF = pg
    pid, apid = _setup(SF, tmp_path)
    key = f"{apid}:probe:g1:launch"

    def eff():
        return AutopilotEffect(autopilot_id=apid, step_id="probe", name="launch",
                               idempotency_key=key, kind="action", handler="create_run",
                               params={}, params_sha256="x", state="launching")

    a, b = SF(), SF()
    a.add(eff()); a.flush()
    a.add(Run(id=str(uuid.uuid4()), project_id=pid, prompt="a", reference_image_urls=[]))
    a.flush()
    errors = []

    def second():
        try:
            b.add(eff())
            b.flush()  # blocks until A ends
            b.add(Run(id=str(uuid.uuid4()), project_id=pid, prompt="b", reference_image_urls=[]))
            b.commit()
        except IntegrityError as exc:
            errors.append(exc)
            b.rollback()

    t = threading.Thread(target=second)
    t.start()
    time.sleep(0.5)
    assert t.is_alive()  # waiting on the unique index
    a.commit()
    t.join(15)
    a.close(); b.close()
    assert errors
    db = SF()
    assert [r.prompt for r in db.execute(select(Run).where(Run.project_id == pid)).scalars()] == ["a"]
    db.close()


def test_hold_on_waits_for_an_admission_in_progress(pg, tmp_path):
    from app.autopilot import core, gate

    SF = pg
    a = SF()
    core._gate(a, share=True)  # an admitting transaction holds the gate FOR SHARE
    done = threading.Event()

    def hold():
        db = SF()
        gate.hold_on(db, by="test")
        db.close()
        done.set()

    t = threading.Thread(target=hold)
    t.start()
    assert not done.wait(0.7)  # the barrier waits for the admission
    a.commit(); a.close()
    assert done.wait(10)
    db = SF()
    assert gate.status(db)["held"] is True
    db.close()


def test_planning_session_is_read_only_on_postgres(pg, tmp_path):
    from app.autopilot import core
    from app.models import VideoProject

    SF = pg
    pid, _ = _setup(SF, tmp_path)
    with pytest.raises(Exception) as exc:
        with core._ReadOnly(SF) as db:
            db.execute(text("UPDATE video_projects SET name = 'x'"))
    assert "read-only" in str(exc.value).lower()
    db = SF()
    assert db.get(VideoProject, pid).name is None
    db.close()


def test_one_live_autopilot_per_project(pg, tmp_path):
    from app.models_autopilot import ProjectAutopilot

    SF = pg
    pid, _ = _setup(SF, tmp_path)
    db = SF()
    db.add(ProjectAutopilot(project_id=pid, flow_id="stub", flow_version=1, status="active",
                            params={}, tolerance=0.1, version=0, outstanding=0))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.add(ProjectAutopilot(project_id=pid, flow_id="stub", flow_version=1, status="stopped",
                            params={}, tolerance=0.1, version=0, outstanding=0))
    db.commit()  # finished ones are history
    db.close()


def test_delete_project_racing_a_tick_does_not_deadlock(pg, tmp_path):
    import app.api_v2 as api_v2_module
    from app.autopilot import core
    from app.models import VideoProject
    from app.models_autopilot import ProjectAutopilot

    SF = pg
    pid, apid = _setup(SF, tmp_path)
    tick = SF()
    core._lock_project(tick, pid)
    core._lock_ap(tick, apid)
    done, errors = threading.Event(), []

    def delete():
        db = SF()
        try:
            api_v2_module.delete_project(pid, db=db)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            db.close()
            done.set()

    t = threading.Thread(target=delete)
    t.start()
    assert not done.wait(0.7)  # waits for the tick's project lock
    tick.commit(); tick.close()
    assert done.wait(15) and not errors
    db = SF()
    assert db.get(VideoProject, pid) is None and db.get(ProjectAutopilot, apid) is None
    db.close()


def test_compute_claim_waits_for_the_tick_and_does_not_deadlock(pg, tmp_path):
    from app.autopilot import compute, core, registry
    from app.models_autopilot import AutopilotEffect

    SF = pg
    pid, apid = _setup(SF, tmp_path)
    db = SF()
    e = AutopilotEffect(autopilot_id=apid, step_id="scan", name="scan", idempotency_key="k-c",
                        kind="compute", handler="stub_scan", handler_version=1,
                        params={"args": {"n": 1}}, params_sha256="x", state="queued")
    db.add(e); db.commit(); eid = e.id; db.close()
    tick = SF()
    core._lock_project(tick, pid)
    core._lock_ap(tick, apid)
    out = []
    t = threading.Thread(target=lambda: out.append(compute.run(eid, session_factory=SF)))
    t.start()
    time.sleep(0.5)
    assert t.is_alive()
    tick.commit(); tick.close()
    t.join(15)
    assert out == ["succeeded"]
