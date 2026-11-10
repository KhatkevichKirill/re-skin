"""AICP-369: the assistant's work shown inside its Run (app/autopilot/run_view.py).

Records are written directly (no flow is driven): the resolver is a pure read
over ``project_autopilots`` / ``autopilot_effects`` / ``autopilot_approvals``.
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
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.autopilot import core, run_view
from app.config import settings
from app.db import Base, get_db
from app.models import Run, VideoProject
from app.models_autopilot import (
    AutopilotApproval, AutopilotEffect, AutopilotJournal, AutopilotStep, ProjectAutopilot,
    utcnow,
)
from app.state_machine import ProjectStatus, RunStatus


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


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "wake", lambda: None)
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path), raising=False)


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


T0 = utcnow() - timedelta(hours=1)


class W:
    """Writes the records of one assistant project."""

    def __init__(self, SF, tmp_path, *, project_type="subtitle_removal", flow="subtitle_removal"):
        self.db = SF()
        src = tmp_path / f"src-{os.urandom(4).hex()}.mp4"
        src.write_bytes(b"src")
        p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                         status=ProjectStatus.ready, project_type=project_type, mode="assistant",
                         duration_sec=10.0)
        self.db.add(p)
        self.db.flush()
        self.pid, self.flow = p.id, flow
        self.ap = None
        self.n = 0

    def autopilot(self, *, status="active", at=None):
        ap = ProjectAutopilot(project_id=self.pid, flow_id=self.flow, flow_version=1,
                              status=status, params={}, created_at=at or T0)
        self.db.add(ap)
        self.db.flush()
        self.ap = ap
        return ap

    def step(self, step_id, *, gen=1, status="active"):
        s = AutopilotStep(autopilot_id=self.ap.id, step_id=step_id, generation=gen, status=status,
                          started_at=self._t())
        self.db.add(s)
        self.db.flush()
        return s

    def _t(self):
        self.n += 1
        return T0 + timedelta(seconds=self.n)

    def run(self, *, at=None):
        r = Run(project_id=self.pid, prompt="p", status=RunStatus.processing,
                reference_image_urls=[], created_at=at or self._t())
        self.db.add(r)
        self.db.flush()
        return r

    def launch(self, step_id, run=None, *, gen=1, handler="create_run", state="launched"):
        t = self._t()
        e = AutopilotEffect(autopilot_id=self.ap.id, step_id=step_id, step_generation=gen,
                            name=f"launch:{self.n}", idempotency_key=f"k-{os.urandom(6).hex()}",
                            kind="action", handler=handler, params={}, params_sha256="x",
                            paid=True, state=state, created_at=t,
                            created_refs={"runs": [run.id]} if run else {})
        self.db.add(e)
        self.db.flush()
        return e

    def ask(self, step_id, *, kind="regeneration", bound=None, artifacts=None, gen=1):
        a = AutopilotApproval(autopilot_id=self.ap.id, step_id=step_id, step_generation=gen,
                              name=f"a{self.n}", dedupe_key=f"d-{os.urandom(6).hex()}", kind=kind,
                              question=f"question {self.n}?", proposal={"x": 1},
                              bound_versions=bound or {}, artifacts=artifacts,
                              created_at=self._t())
        self.n += 1
        self.db.add(a)
        self.db.flush()
        return a

    def done(self):
        self.db.commit()
        self.db.close()


def _pr(SF, pid):
    db = SF()
    try:
        return run_view.project_runs(db, pid)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Which runs are the assistant's
# ---------------------------------------------------------------------------


def test_subtitle_main_run_and_manual_run(SF, tmp_path):
    w = W(SF, tmp_path)
    w.autopilot()
    w.step("launch", status="completed")
    w.step("clean")
    r1 = w.run()
    w.launch("launch", r1)
    manual = w.run()
    w.done()
    pr = _pr(SF, w.pid)
    assert set(pr.by_run) == {r1.id}
    ar = pr.by_run[r1.id]
    assert (ar.role, ar.language, ar.current) == ("main", None, True)
    assert set(ar.steps) == {"launch", "clean", "handoff"}
    assert pr.kind_of(manual) == "manual" and pr.kind_of(r1) == "assistant"


def test_localisation_probe_and_final_per_language_and_replacement(SF, tmp_path):
    w = W(SF, tmp_path, project_type="localisation", flow="localisation")
    w.autopilot()
    w.step("probe:es", status="completed")
    w.step("rest:es", gen=2)
    probe = w.run()
    w.launch("probe:es", probe, handler="localisation_create_run")
    old = w.run()
    w.launch("rest:es", old, handler="localisation_create_run", gen=1)
    new = w.run()
    w.launch("rest:es", new, handler="localisation_create_run", gen=2)
    w.done()
    pr = _pr(SF, w.pid)
    assert pr.by_run[probe.id].label == "probe ES"
    assert pr.by_run[new.id].label == "final ES" and pr.by_run[new.id].current
    assert pr.by_run[old.id].label == "final ES replaced"
    assert "qa:es" in pr.by_run[new.id].steps and "captions:es" in pr.by_run[new.id].steps


def test_earlier_attempt_runs_are_not_current(SF, tmp_path):
    w = W(SF, tmp_path)
    w.autopilot(status="stopped", at=T0)
    w.step("launch", status="completed")
    r_old = w.run()
    w.launch("launch", r_old)
    w.autopilot(at=T0 + timedelta(minutes=5))
    w.step("launch")
    w.done()
    pr = _pr(SF, w.pid)
    assert pr.by_run[r_old.id].current is False
    assert pr.latest_live is True and pr.latest_autopilot_id != pr.by_run[r_old.id].autopilot_id


def test_ambiguous_launch_marks_later_runs_unconfirmed_not_manual(SF, tmp_path):
    w = W(SF, tmp_path)
    w.autopilot()
    w.step("launch")
    before = w.run()
    w.launch("launch", None, state="unknown")
    after = w.run()
    w.done()
    pr = _pr(SF, w.pid)
    assert pr.by_run == {}
    assert pr.kind_of(before) == "manual"
    assert pr.kind_of(after) == "unconfirmed"


def test_deleted_run_is_skipped(SF, tmp_path):
    w = W(SF, tmp_path)
    w.autopilot()
    w.step("launch")
    r = w.run()
    w.launch("launch", r)
    w.db.delete(r)
    w.done()
    assert _pr(SF, w.pid).by_run == {}


# ---------------------------------------------------------------------------
# Where an approval is shown
# ---------------------------------------------------------------------------


def test_approval_placement_rules(SF, tmp_path):
    w = W(SF, tmp_path)
    w.autopilot()
    w.step("segmentation", status="completed")
    w.step("launch", status="completed")
    w.step("clean")
    r = w.run()
    w.launch("launch", r)
    budget = w.ask("launch", kind="budget", bound={f"run:{r.id}": "v"})
    pre = w.ask("segmentation", kind="handoff")
    bound = w.ask("clean", bound={f"run:{r.id}": "v1"})
    by_step = w.ask("clean", kind="handoff",
                    artifacts=[{"label": "sheet", "kind": "image", "producer": "effect:e1",
                                "sha256": "0" * 64}])
    by_art = w.ask("segmentation", kind="handoff",
                   artifacts=[{"label": "v", "kind": "video", "producer": f"run:{r.id}",
                               "sha256": "1" * 64}])
    foreign = w.ask("segmentation", bound={"run:not-the-assistants": "v"})
    w.done()
    pr = _pr(SF, w.pid)
    assert run_view.approval_runs(budget, pr)[0] is None           # budget: always project
    assert run_view.approval_runs(pre, pr)[0] is None              # pre-run step
    assert run_view.approval_runs(bound, pr)[0] == r.id
    assert run_view.approval_runs(by_step, pr)[0] == r.id          # step owner (clean)
    assert run_view.approval_runs(by_art, pr)[0] == r.id           # run-produced artifact
    assert run_view.approval_runs(foreign, pr)[0] is None          # not one of ours


def test_several_bound_runs_go_to_the_step_owner_or_the_project(SF, tmp_path):
    w = W(SF, tmp_path, project_type="localisation", flow="localisation")
    w.autopilot()
    w.step("probe:es", status="completed")
    w.step("rest:es")
    probe = w.run()
    w.launch("probe:es", probe, handler="localisation_create_run")
    final = w.run()
    w.launch("rest:es", final, handler="localisation_create_run")
    both_qa = w.ask("qa:es", bound={f"run:{probe.id}": "a", f"run:{final.id}": "b"})
    both_pre = w.ask("prompts:es", bound={f"run:{probe.id}": "a", f"run:{final.id}": "b"})
    w.done()
    pr = _pr(SF, w.pid)
    assert run_view.approval_runs(both_qa, pr)[0] == final.id
    rid, cands, _why = run_view.approval_runs(both_pre, pr)
    assert rid is None and set(cands) == {probe.id, final.id}


def test_reopened_step_without_its_new_run_goes_to_the_project(SF, tmp_path):
    w = W(SF, tmp_path, project_type="localisation", flow="localisation")
    w.autopilot()
    w.step("rest:es", gen=2)
    old = w.run()
    w.launch("rest:es", old, handler="localisation_create_run", gen=1)
    a = w.ask("rest:es", gen=2)  # gen 2 has no run yet: must not land on the old one
    w.done()
    assert run_view.approval_runs(a, _pr(SF, w.pid))[0] is None


# ---------------------------------------------------------------------------
# Pages and endpoints
# ---------------------------------------------------------------------------


def _subtitle_with_pending(SF, tmp_path):
    w = W(SF, tmp_path)
    w.autopilot()
    w.step("segmentation", status="completed")
    w.step("launch", status="completed")
    w.step("clean")
    r = w.run()
    eff = w.launch("launch", r)
    manual = w.run()
    on_run = w.ask("clean", bound={f"run:{r.id}": "v"})
    on_project = w.ask("segmentation", kind="handoff")
    w.db.add(AutopilotJournal(autopilot_id=w.ap.id, step_id="launch", kind="launch",
                              message="launching the run", effect_id=eff.id, at=w._t()))
    w.db.add(AutopilotJournal(autopilot_id=w.ap.id, step_id="segmentation", kind="decision",
                              message="segmentation note", at=w._t()))
    apid = w.ap.id
    w.done()
    return w.pid, apid, r.id, manual.id, on_run.id, on_project.id


def test_project_card_links_run_approvals_and_lists_runs(SF, tmp_path, client):
    pid, apid, rid, _m, on_run, on_project = _subtitle_with_pending(SF, tmp_path)
    html = client.get(f"/v2/projects/{pid}/autopilot-card").text
    assert "The assistant&#39;s runs" in html or "The assistant's runs" in html
    assert f'href="/v2/runs/{rid}"' in html and "Answer on the run" in html
    # the project-level approval is a full form there, the run one is not
    assert f'value="{on_project}"' in html and f'value="{on_run}"' not in html
    assert f'name="autopilot_id" value="{apid}"' in html


def test_run_card_shows_its_approval_and_journal(SF, tmp_path, client):
    pid, apid, rid, _m, on_run, on_project = _subtitle_with_pending(SF, tmp_path)
    html = client.get(f"/v2/runs/{rid}/autopilot-card").text
    assert f'value="{on_run}"' in html and f'value="{on_project}"' not in html
    # the journal is loaded when opened (AICP-377): the run's lines only
    assert f'hx-get="/v2/runs/{rid}/autopilot-journal"' in html
    jl = client.get(f"/v2/runs/{rid}/autopilot-journal?all=1").text
    assert "launching the run" in jl and "segmentation note" not in jl
    assert "1 more approval(s)" in html
    assert 'hx-post="/v2/runs/' + rid + '/autopilot-card/switch"' in html


def test_manual_run_card_and_runs_table_labels(SF, tmp_path, client):
    pid, apid, rid, manual, *_ = _subtitle_with_pending(SF, tmp_path)
    html = client.get(f"/v2/runs/{manual}/autopilot-card").text
    assert "Manual run" in html and "does not automatically use this run" in html
    db = SF()
    try:
        labels = run_view.run_labels(db, db.get(VideoProject, pid),
                                     [db.get(Run, rid), db.get(Run, manual)])
    finally:
        db.close()
    assert labels[rid]["kind"] == "assistant" and labels[rid]["label"] == "main"
    assert labels[manual]["kind"] == "manual"


def test_answer_from_the_run_page(SF, tmp_path, client):
    pid, apid, rid, _m, on_run, _p = _subtitle_with_pending(SF, tmp_path)
    r = client.post(f"/v2/runs/{rid}/autopilot-card/answer",
                    data={"approval_id": on_run, "decision": "reject"})
    assert r.status_code == 200 and 'id="autopilot-run-card"' in r.text
    db = SF()
    try:
        assert db.get(AutopilotApproval, on_run).status == "rejected"
    finally:
        db.close()


def test_manual_run_page_cannot_answer_or_switch(SF, tmp_path, client):
    pid, apid, rid, manual, on_run, _p = _subtitle_with_pending(SF, tmp_path)
    r = client.post(f"/v2/runs/{manual}/autopilot-card/switch", data={"action": "pause"})
    assert "did not create this run" in r.text
    r = client.post(f"/v2/runs/{manual}/autopilot-card/answer",
                    data={"approval_id": on_run, "decision": "approve"})
    assert "did not create this run" in r.text
    db = SF()
    try:
        assert db.get(ProjectAutopilot, apid).status == "active"
        assert db.get(AutopilotApproval, on_run).status == "pending"
    finally:
        db.close()


def test_old_attempt_run_page_has_no_controls_and_cannot_act_on_the_new_one(SF, tmp_path, client):
    w = W(SF, tmp_path)
    w.autopilot(status="stopped", at=T0)
    w.step("launch", status="completed")
    old_run = w.run()
    w.launch("launch", old_run)
    w.autopilot(at=T0 + timedelta(minutes=5))
    new_ap = w.ap.id
    w.step("launch")
    pid = w.pid
    w.done()
    html = client.get(f"/v2/runs/{old_run.id}/autopilot-card").text
    assert "earlier autopilot attempt" in html and "/autopilot-card/switch" not in html
    r = client.post(f"/v2/runs/{old_run.id}/autopilot-card/switch", data={"action": "pause"})
    assert "no longer the project" in r.text
    db = SF()
    try:
        assert db.get(ProjectAutopilot, new_ap).status == "active"
    finally:
        db.close()


def test_stopped_attempt_run_approval_stays_answerable_on_the_run(SF, tmp_path, client):
    # Stop does not clear pending approvals and the core still accepts them, so
    # the run page (the only place a run-bound one is a form) must keep the form.
    pid, apid, rid, _m, on_run, _p = _subtitle_with_pending(SF, tmp_path)
    client.post(f"/v2/projects/{pid}/autopilot-card/switch",
                data={"action": "stop", "autopilot_id": apid})
    html = client.get(f"/v2/runs/{rid}/autopilot-card").text
    assert f'value="{on_run}"' in html and "/autopilot-card/switch" not in html
    r = client.post(f"/v2/runs/{rid}/autopilot-card/answer",
                    data={"approval_id": on_run, "decision": "reject"})
    assert r.status_code == 200
    db = SF()
    try:
        assert db.get(ProjectAutopilot, apid).status == "stopped"
        assert db.get(AutopilotApproval, on_run).status == "rejected"
    finally:
        db.close()


def test_answer_refused_when_a_new_attempt_appears_before_the_lock(SF, tmp_path, monkeypatch):
    # Operator actions lock the project row before reading the latest attempt
    # (serializing with start_new); a successor committed just before that lock
    # must make the answer 409, not land on the old attempt.
    pid, apid, rid, _m, on_run, _p = _subtitle_with_pending(SF, tmp_path)
    db = SF()
    try:
        db.get(ProjectAutopilot, apid).status = "stopped"
        db.commit()
        real_lock = core._lock_project

        def lock_after_successor(d, project_id):
            other = SF()
            try:
                other.add(ProjectAutopilot(project_id=pid, flow_id="subtitle_removal",
                                           flow_version=1, status="active", params={},
                                           created_at=utcnow()))
                other.commit()
            finally:
                other.close()
            return real_lock(d, project_id)

        monkeypatch.setattr(core, "_lock_project", lock_after_successor)
        with pytest.raises(core.OperatorError) as exc:
            core.answer(db, pid, on_run, "reject", expected_id=apid)
        assert exc.value.status == 409
    finally:
        db.close()
    db = SF()
    try:
        assert db.get(AutopilotApproval, on_run).status == "pending"
    finally:
        db.close()


def test_project_card_stale_attempt_id_is_refused(SF, tmp_path, client):
    pid, apid, *_ = _subtitle_with_pending(SF, tmp_path)
    r = client.post(f"/v2/projects/{pid}/autopilot-card/switch",
                    data={"action": "pause", "autopilot_id": "an-older-attempt"})
    assert "no longer the project" in r.text
    r = client.post(f"/v2/projects/{pid}/autopilot-card/switch",
                    data={"action": "pause", "autopilot_id": apid})
    db = SF()
    try:
        assert db.get(ProjectAutopilot, apid).status == "paused"
    finally:
        db.close()


def test_run_page_loads_the_card_only_on_assistant_projects(SF, tmp_path, client):
    pid, apid, rid, *_ = _subtitle_with_pending(SF, tmp_path)
    html = client.get(f"/v2/runs/{rid}").text
    assert f'hx-get="/v2/runs/{rid}/autopilot-card"' in html
