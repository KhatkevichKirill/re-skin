"""AICP-373: shared link / anchor / title / "answerable" helpers (app/autopilot/links.py).

Records are written directly, as in test_autopilot_run_view.py.
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

from app.autopilot import core, links, run_view
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



def _subtitle(SF, tmp_path, *, status="active", at=None):
    """A subtitle project: main run r, a run-bound pending ask and a project one."""
    w = W(SF, tmp_path)
    w.autopilot(status=status, at=at)
    w.step("segmentation", status="completed")
    w.step("launch", status="completed")
    w.step("clean")
    r = w.run()
    w.launch("launch", r)
    on_run = w.ask("clean", bound={f"run:{r.id}": "v"})
    on_project = w.ask("segmentation", kind="handoff")
    return w, r, on_run, on_project


def _place(SF, approval_id):
    db = SF()
    try:
        return links.place(db, db.get(AutopilotApproval, approval_id))
    finally:
        db.close()


def _answerable_ids(SF, project_id=None):
    db = SF()
    try:
        return [a.id for a, _ap in links.answerable(db, project_id=project_id)]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


def test_title_is_the_first_sentence_of_the_question():
    q = "Segment 1 was refused by the provider.  Fill it from the source or retry?"
    assert links.approval_title(q, {}) == "Segment 1 was refused by the provider."


def test_title_keeps_decimals_and_colons_in_one_sentence():
    q = "Cut at 1.5 s: the keep segment is short. Approve?"
    assert links.approval_title(q, None) == "Cut at 1.5 s: the keep segment is short."


def test_long_title_is_cut_at_80_on_a_word():
    q = ("Segment 3 needs the operator: caption leftovers, but no fill can be placed: the final "
         "is not verified on the source timeline (alignment)")
    t = links.approval_title(q, {})
    assert len(t) <= links.TITLE_MAX and t.endswith("…")
    assert t.startswith("Segment 3 needs the operator: caption leftovers")
    assert not t[:-1].endswith(" ")


def test_stored_title_wins_and_is_clipped():
    assert links.approval_title("Long question. More.", {"_title": "  Check the  band "}) \
        == "Check the band"
    assert len(links.approval_title("q", {"_title": "x" * 200})) == links.TITLE_MAX
    assert links.approval_title("Q one. Q two.", {"_title": "   "}) == "Q one."
    assert links.approval_title("Q one. Q two.", {"_title": 7}) == "Q one."


def test_empty_question_has_a_title():
    assert links.approval_title("", {}) == "Question"
    assert links.approval_title(None, None) == "Question"


def test_approval_dict_carries_title_and_anchor(SF, tmp_path):
    w, _r, on_run, _p = _subtitle(SF, tmp_path)
    d = core.approval_dict(on_run, w.pid)
    assert d["anchor"] == f"ap-{on_run.id}" and d["title"].startswith("question")
    w.done()


# ---------------------------------------------------------------------------
# Placement and links
# ---------------------------------------------------------------------------


def test_run_bound_approval_links_to_its_run(SF, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://reskin.example")
    w, r, on_run, on_project = _subtitle(SF, tmp_path)
    pid, rid, short = w.pid, r.id, r.short_id
    a_run, a_proj = on_run.id, on_project.id
    w.done()
    pl = _place(SF, a_run)
    assert (pl.run_id, pl.latest, pl.open) == (rid, True, True)
    assert pl.path == f"/v2/runs/{rid}#ap-{a_run}"
    assert pl.url == f"https://reskin.example/v2/runs/{rid}#ap-{a_run}"
    assert pl.where == f"run {short} (main)"
    pp = _place(SF, a_proj)
    assert pp.run_id is None and pp.path == f"/v2/projects/{pid}#ap-{a_proj}"
    assert pp.where == "project"


def test_no_public_base_means_no_absolute_url(SF, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
    w, _r, on_run, _p = _subtitle(SF, tmp_path)
    aid = on_run.id
    w.done()
    pl = _place(SF, aid)
    assert pl.url is None and pl.path.startswith("/v2/runs/")


def test_budget_approval_links_to_the_project(SF, tmp_path):
    w = W(SF, tmp_path)
    w.autopilot()
    w.step("launch")
    a = w.ask("launch", kind="budget")
    pid, aid = w.pid, a.id
    w.done()
    pl = _place(SF, aid)
    assert pl.run_id is None and pl.path == f"/v2/projects/{pid}#ap-{aid}" and pl.open


def test_deleted_run_falls_back_to_the_project(SF, tmp_path):
    w, r, on_run, _p = _subtitle(SF, tmp_path)
    pid, aid = w.pid, on_run.id
    w.db.delete(r)
    w.done()
    pl = _place(SF, aid)
    assert pl.run_id is None and pl.path == f"/v2/projects/{pid}#ap-{aid}"
    assert pl.open  # still pending on the live latest attempt: answerable on the project card
    assert aid in _answerable_ids(SF, pid)


def test_stopped_attempt_is_not_answerable_but_still_links_to_its_run(SF, tmp_path):
    w, r, on_run, on_project = _subtitle(SF, tmp_path, status="stopped")
    pid, rid, aid = w.pid, r.id, on_run.id
    w.done()
    pl = _place(SF, aid)
    assert pl.latest and not pl.open
    assert pl.path == f"/v2/runs/{rid}#ap-{aid}"   # the run card still shows it
    assert _answerable_ids(SF, pid) == []


def test_newer_attempt_replaces_the_older_ones_questions(SF, tmp_path):
    w, r, old_on_run, old_on_project = _subtitle(SF, tmp_path, status="stopped", at=T0)
    w.autopilot(at=T0 + timedelta(minutes=5))
    w.step("launch")
    fresh = w.ask("launch", kind="budget")
    pid, rid = w.pid, r.id
    ids = (old_on_run.id, old_on_project.id, fresh.id)
    w.done()
    old_run_pl = _place(SF, ids[0])
    assert old_run_pl.run_id == rid and not old_run_pl.latest and not old_run_pl.open
    # the run card answers nothing for a replaced attempt: the link goes to the project
    assert old_run_pl.path == f"/v2/projects/{pid}#ap-{ids[0]}"
    assert old_run_pl.where == "project"
    assert not _place(SF, ids[1]).open
    assert _answerable_ids(SF, pid) == [ids[2]]


def test_paused_attempt_questions_are_answerable(SF, tmp_path):
    w, _r, on_run, on_project = _subtitle(SF, tmp_path, status="paused")
    pid, ids = w.pid, [on_run.id, on_project.id]
    w.done()
    assert _answerable_ids(SF, pid) == ids          # oldest first
    db = SF()
    try:
        assert {ap.status for _a, ap in links.answerable(db, project_id=pid)} == {"paused"}
    finally:
        db.close()


def test_answered_and_superseded_approvals_are_not_answerable(SF, tmp_path):
    w, _r, on_run, on_project = _subtitle(SF, tmp_path)
    on_run.status = "approved"
    on_project.status = "superseded"
    pid, aid = w.pid, on_run.id
    w.done()
    assert _answerable_ids(SF, pid) == []
    assert not _place(SF, aid).open


def test_answerable_spans_projects(SF, tmp_path):
    w1, _r1, a1, _p1 = _subtitle(SF, tmp_path)
    w1.done()
    w2, _r2, a2, _p2 = _subtitle(SF, tmp_path, status="failed")
    w2.done()
    every = _answerable_ids(SF)
    assert a1.id in every and a2.id not in every


# ---------------------------------------------------------------------------
# Anchors on the cards
# ---------------------------------------------------------------------------


def test_cards_carry_the_anchors_and_the_fragment_script(SF, tmp_path, client):
    w, r, on_run, on_project = _subtitle(SF, tmp_path)
    pid, rid, a_run, a_proj = w.pid, r.id, on_run.id, on_project.id
    w.done()
    html = client.get(f"/v2/projects/{pid}/autopilot-card").text
    assert f'id="ap-{a_proj}"' in html and f'id="ap-{a_run}"' in html
    assert f'href="/v2/runs/{rid}#ap-{a_run}"' in html          # "Answer on the run"
    assert 'apFocusFragment(document.getElementById(\'autopilot-card\'))' in html
    html = client.get(f"/v2/runs/{rid}/autopilot-card").text
    assert f'id="ap-{a_run}"' in html and f'id="ap-{a_proj}"' not in html
    assert 'apFocusFragment(document.getElementById(\'autopilot-run-card\'))' in html


def test_old_link_to_a_closed_question_finds_no_anchor(SF, tmp_path, client):
    # The card renders no element for a closed question, so the fragment script
    # shows "no longer open" instead of scrolling.
    w, r, on_run, _p = _subtitle(SF, tmp_path)
    on_run.status = "superseded"
    rid, aid = r.id, on_run.id
    w.done()
    pl = _place(SF, aid)
    assert not pl.open and pl.path == f"/v2/runs/{rid}#ap-{aid}"
    html = client.get(f"/v2/runs/{rid}/autopilot-card").text
    assert f'id="ap-{aid}"' not in html and "no longer open" in html


def test_fragment_script_opens_the_answers_list_for_an_answered_question(SF, tmp_path, client):
    # Terra gate: an answered (reopenable) question's anchor is inside the
    # collapsed "Your answers" <details>; the script must open it, not scroll
    # to a hidden element.
    w, r, _on_run, _p = _subtitle(SF, tmp_path)
    rid = r.id
    w.done()
    html = client.get(f"/v2/runs/{rid}/autopilot-card").text
    assert "el.closest('details')" in html and "d.open = true" in html
