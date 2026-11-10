"""AICP-377: status line, stepper, budget bar, paused banner (app/autopilot/status.py)
and the journal timeline (app/autopilot/timeline.py + the journal fragments)."""

from __future__ import annotations

import os
import re
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

from app.autopilot import core, status, timeline
from app.autopilot.flows.localisation import LocalisationFlow
from app.config import settings
from app.db import Base, get_db
from app.models import Run, VideoProject
from app.models_autopilot import (
    AutopilotApproval, AutopilotEffect, AutopilotJournal, AutopilotStep, ProjectAutopilot,
    utcnow,
)
from app.state_machine import ProjectStatus, RunStatus

T0 = utcnow() - timedelta(hours=2)


@pytest.fixture()
def SF(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 's.db'}", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    eng.dispose()


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


class W:
    def __init__(self, SF, tmp_path, *, flow="subtitle_removal", version=1, status="active",
                 reason=None, params=None, budget_mc=None):
        self.db, self.n = SF(), 0
        src = tmp_path / f"src-{os.urandom(4).hex()}.mp4"
        src.write_bytes(b"src")
        p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                         status=ProjectStatus.ready,
                         project_type="localisation" if flow == "localisation" else "subtitle_removal",
                         mode="assistant", duration_sec=10.0)
        self.db.add(p)
        self.db.flush()
        self.pid = p.id
        self.ap = ProjectAutopilot(project_id=p.id, flow_id=flow, flow_version=version, status=status,
                                   status_reason=reason, params=params or {}, created_at=T0,
                                   approved_budget_mcredits=budget_mc)
        self.db.add(self.ap)
        self.db.flush()

    def t(self):
        self.n += 1
        return T0 + timedelta(seconds=self.n)

    def step(self, sid, st="active", reason=None, gen=1):
        self.db.add(AutopilotStep(autopilot_id=self.ap.id, step_id=sid, status=st, reason=reason,
                                  generation=gen, started_at=self.t()))
        self.db.flush()

    def line(self, kind, msg, step_id=None, **kw):
        j = AutopilotJournal(autopilot_id=self.ap.id, kind=kind, message=msg, step_id=step_id,
                             at=self.t(), **kw)
        self.db.add(j)
        self.db.flush()
        return j.id

    def ask(self, step_id="clean", bound=None):
        a = AutopilotApproval(autopilot_id=self.ap.id, step_id=step_id, step_generation=1,
                              name=f"q{self.n}", dedupe_key=os.urandom(6).hex(), kind="handoff",
                              question="Is it clean?", proposal={}, bound_versions=bound or {},
                              created_at=self.t())
        self.db.add(a)
        self.db.flush()
        return a

    def done(self):
        self.db.commit()
        ids = (self.pid, self.ap.id)
        self.db.close()
        return ids


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("st,reason,pending,steps,tone,text", [
    ("active", None, 2, [], "you", "Waiting for you: 2 questions"),
    ("active", None, 1, [], "you", "Waiting for you: 1 question"),
    ("active", None, 0, [{"step_id": "clean", "status": "active",
                          "reason": "run d46bb1b8 is stitching"}],
     "work", "Cleaning — run d46bb1b8 is stitching"),
    ("active", None, 0, [{"step_id": "probe:es", "status": "active", "reason": "refused: 503: no worker"}],
     "work", "Probe (ES) — refused, retrying later: 503: no worker"),
    ("active", None, 0, [{"step_id": "launch", "status": "completed", "reason": "x"}],
     "work", "Working: deciding the next step"),
    ("paused", "manual paid action: rerun (r1, ~4200 credits) — Resume to continue", 1, [],
     "paused", "Paused: manual paid action: rerun (r1, ~4200 credits) — Resume to continue"),
    ("stopped", "stopped by operator", 0, [], "ended", "Stopped: stopped by operator"),
    ("failed", "step clean failed: boom", 0, [], "ended", "Failed: step clean failed: boom"),
    ("completed", None, 0, [], "done", "Completed"),
])
def test_status_line(st, reason, pending, steps, tone, text):
    assert status.status_line(st, reason, steps, pending) == {"tone": tone, "text": text}


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


def test_subtitle_stepper_follows_the_flow_order_and_marks_waiting():
    rows = status.stepper(["segmentation", "launch", "clean", "handoff"], [
        {"step_id": "launch", "status": "completed", "generation": 1, "reason": ""},
        {"step_id": "segmentation", "status": "completed", "generation": 2, "reason": ""},
        {"step_id": "clean", "status": "active", "generation": 1, "reason": "waiting for the operator"},
    ], ["clean"])
    assert len(rows) == 1 and rows[0]["language"] is None
    cells = rows[0]["cells"]
    assert [c["label"] for c in cells] == ["Segmentation", "Launch", "Cleaning", "Handoff"]
    assert [c["state"] for c in cells] == ["done", "done", "you", "pending"]
    assert cells[0]["generation"] == 2


def test_localisation_stepper_has_one_row_per_language():
    flow = LocalisationFlow()
    ids = [s.id for s in flow.steps(None, {"languages": ["es", "pt"]})]
    rows = status.stepper(ids, [{"step_id": "translate:pt", "status": "active", "generation": 1,
                                 "reason": ""}], [])
    assert [r["language"] for r in rows] == [None, "ES", "PT"]
    assert [c["label"] for c in rows[0]["cells"]] == ["Intake", "Transcribe", "Segments", "Breakdown"]
    assert [c["label"] for c in rows[1]["cells"]] == [
        "Translate", "Prompts", "Probe", "Rest", "QA", "Captions", "Tail", "Cleanup"]
    assert rows[2]["cells"][0]["state"] == "active"
    # the Run page keeps only its steps
    only = status.stepper(ids, [], [], only=["probe:es"])
    assert [(r["language"], [c["step_id"] for c in r["cells"]]) for r in only] == [("ES", ["probe:es"])]


def test_stepper_without_the_flow_uses_the_rows():
    rows = status.stepper(None, [{"step_id": "a", "status": "completed", "generation": 1, "reason": ""},
                                 {"step_id": "b", "status": "failed", "generation": 1, "reason": "x"}], [])
    assert [c["state"] for c in rows[0]["cells"]] == ["done", "failed"]


# ---------------------------------------------------------------------------
# Budget bar
# ---------------------------------------------------------------------------


def _b(approved, committed, tol=0.1):
    return {"approved": {"credits": approved} if approved is not None else None,
            "committed": {"credits": committed}, "tolerance": tol}


def test_budget_bar_widths_and_tones():
    assert status.budget_bar(_b(None, 50)) is None
    ok = status.budget_bar(_b(100, 50))
    assert ok["tone"] == "ok" and ok["ceiling_pct"] == 100.0 and ok["approved_pct"] == pytest.approx(90.9, 0.01)
    assert status.budget_bar(_b(100, 105))["tone"] == "band"
    over = status.budget_bar(_b(100, 200))
    assert over["tone"] == "over" and over["committed_pct"] == 100.0 and over["ceiling_pct"] == 55.0


# ---------------------------------------------------------------------------
# Paused banner
# ---------------------------------------------------------------------------


def test_banner_for_manual_spend_and_operator_pause(SF, tmp_path):
    db = SF()
    try:
        b = status.pause_banner(db, "x", "paused", "manual paid action: rerun (r, ~10 credits) — Resume",
                                flow_deployed=True)
        assert b["resume"] and "by hand" in b["cause"] and "re-checks the runs" in b["hint"]
        assert status.pause_banner(db, "x", "paused", "paused by operator", flow_deployed=True)["cause"] \
            == "You paused it."
        assert status.pause_banner(db, "x", "active", None, flow_deployed=True) is None
    finally:
        db.close()


@pytest.mark.parametrize("tail", ["", "\n"])
def test_banner_for_refusals_shows_the_step_lines(SF, tmp_path, tail):
    reason = "void:abc refused 8 times: 503: no fal worker" + tail   # a detail may end in whitespace
    w = W(SF, tmp_path, status="paused", reason=reason)
    w.line("refused", "void:abc refused: 503: no fal worker", step_id="clean")
    w.line("warning", "other step", step_id="launch")
    w.line("paused", reason, step_id="clean")
    # a manual paid action while already paused journals a newer pause line (no step)
    for _ in range(60):
        w.line("paused", "manual paid action: rerun (r, ~10 credits) — Resume to continue")
    _pid, apid = w.done()
    db = SF()
    try:
        b = status.pause_banner(db, apid, "paused", reason, flow_deployed=True)
        assert b["resume"] and [j["message"] for j in b["lines"]] == ["void:abc refused: 503: no fal worker"]
    finally:
        db.close()


def test_flow_not_deployed_has_no_resume(SF, tmp_path, client):
    w = W(SF, tmp_path, version=99, status="paused", reason="flow subtitle_removal v99 is no longer deployed")
    pid, _ = w.done()
    html = client.get(f"/v2/projects/{pid}/autopilot-card").text
    assert "Stop it, then Start a new autopilot" in html
    assert 'value="resume"' not in html and 'value="stop"' in html


# ---------------------------------------------------------------------------
# The cards
# ---------------------------------------------------------------------------


def test_project_card_status_stepper_bar_and_stop_copy(SF, tmp_path, client):
    w = W(SF, tmp_path, budget_mc=260_000)
    w.step("segmentation", "completed")
    w.step("launch", "completed")
    w.step("clean", "active", "waiting for the operator: eyes:1")
    w.ask("clean")
    pid, apid = w.done()
    html = client.get(f"/v2/projects/{pid}/autopilot-card").text
    assert "Waiting for you: 1 question" in html
    assert re.search(r'ap-step ap-step-you"\s+title="clean: waiting for you', html)
    assert "ap-bar-mark" in html and "+10 % tolerance, shaded" in html
    assert "Work already running (renders, fills) continues and is still billed" in html
    assert "answers given to this attempt are not carried over" in html
    # the journal is a preserved, lazily loaded box per attempt
    assert f'id="ap-journal-{apid}"' in html and 'hx-preserve="true"' in html
    assert f'/v2/projects/{pid}/autopilot-journal?autopilot_id={apid}' in html


def test_paused_card_has_banner_and_resume(SF, tmp_path, client):
    w = W(SF, tmp_path, status="paused", reason="manual paid action: rerun (r, ~10 credits) — Resume to continue")
    pid, _ = w.done()
    html = client.get(f"/v2/projects/{pid}/autopilot-card").text
    assert "ap-banner" in html and "A paid action was started by hand" in html
    assert 'value="resume"' in html


# ---------------------------------------------------------------------------
# Journal timeline
# ---------------------------------------------------------------------------


def test_important_filter_reaches_old_lines_through_pages(SF, tmp_path, client):
    w = W(SF, tmp_path)
    first = w.line("approval_requested", "the oldest important line", step_id="clean")
    for i in range(80):
        w.line("decision", f"waiting: noise {i}", step_id="clean")
    for i in range(35):
        w.line("launch", f"launch {i}", step_id="launch", cost_mcredits=1000 * i)
    pid, apid = w.done()
    db = SF()
    try:
        p1 = timeline.page(db, apid)
        assert len(p1["entries"]) == 30 and p1["next_before"]
        assert all(e["kind"] == "launch" for e in p1["entries"])
        p2 = timeline.page(db, apid, before=p1["next_before"])
        assert [e["id"] for e in p2["entries"]][-1] == first and p2["next_before"] is None
        assert not {e["id"] for e in p1["entries"]} & {e["id"] for e in p2["entries"]}
        everything = timeline.page(db, apid, important=False, limit=200)
        assert len(everything["entries"]) == 116
    finally:
        db.close()
    html = client.get(f"/v2/projects/{pid}/autopilot-journal?autopilot_id={apid}").text
    assert "launch 34" in html and "noise" not in html and "(34 credits)" in html
    m = re.search(r'hx-get="([^"]*before=\d+)"', html)
    assert m and "all=0" in m.group(1)
    older = client.get(m.group(1).replace("&amp;", "&")).text
    assert "the oldest important line" in older and "<ul" not in older and "Older" not in older
    all_html = client.get(f"/v2/projects/{pid}/autopilot-journal?autopilot_id={apid}&all=1").text
    m = re.search(r'hx-get="([^"]*all=1&before=\d+)"', all_html)
    assert m                                             # All lines pages too
    assert "noise 79" in client.get(m.group(1).replace("&amp;", "&")).text


def test_project_journal_refuses_another_projects_attempt(SF, tmp_path, client):
    a = W(SF, tmp_path)
    pid_a, _ = a.done()
    b = W(SF, tmp_path)
    _pid_b, ap_b = b.done()
    assert client.get(f"/v2/projects/{pid_a}/autopilot-journal?autopilot_id={ap_b}").status_code == 404


def test_run_journal_marks_step_only_lines_during_this_run(SF, tmp_path, client):
    w = W(SF, tmp_path)
    w.step("launch", "completed")
    w.step("clean")
    r = Run(project_id=w.pid, prompt="p", status=RunStatus.processing, reference_image_urls=[],
            created_at=w.t())
    w.db.add(r)
    w.db.flush()
    e = AutopilotEffect(autopilot_id=w.ap.id, step_id="launch", step_generation=1, name="launch:r1",
                        idempotency_key=os.urandom(6).hex(), kind="action", handler="create_run",
                        params={}, params_sha256="x", paid=True, state="launched", created_at=w.t(),
                        created_refs={"runs": [r.id]})
    w.db.add(e)
    w.db.flush()
    w.line("launch", "launched the run", step_id="launch", effect_id=e.id)
    w.line("warning", "step clean error: boom", step_id="clean")
    w.line("decision", "waiting: stitching", step_id="clean")
    rid = r.id
    w.done()
    html = client.get(f"/v2/runs/{rid}/autopilot-journal").text
    assert "launched the run" in html and "step clean error: boom" in html
    assert "waiting: stitching" not in html                     # not important
    li_warn = html[html.index("step clean error"):].split("</li>")[0]
    li_launch = html[html.index("launched the run"):].split("</li>")[0]
    assert "during this run" in li_warn and "during this run" not in li_launch
    assert "waiting: stitching" in client.get(f"/v2/runs/{rid}/autopilot-journal?all=1").text
    card = client.get(f"/v2/runs/{rid}/autopilot-card").text
    assert f'id="ap-run-journal-{rid}"' in card and "Cleaning" in card
