"""AICP-375: the "Waiting for you" inbox and header badge (app/autopilot/inbox.py)."""

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

from app.autopilot import core, inbox, links
from app.config import settings
from app.db import Base, get_db
from app.models import VideoProject
from app.models_autopilot import (
    AutopilotApproval, AutopilotNotification, ProjectAutopilot, utcnow,
)
from app.state_machine import ProjectStatus

T0 = utcnow() - timedelta(hours=3)
_TICK = iter(range(1, 10 ** 6))          # one clock for every project: creation order is global


@pytest.fixture()
def SF(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'i.db'}", connect_args={"check_same_thread": False})

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
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "", raising=False)


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


class P:
    """One assistant project and its attempts."""

    def __init__(self, SF, tmp_path, name):
        self.SF, self.n = SF, 0
        db = SF()
        src = tmp_path / f"src-{os.urandom(4).hex()}.mp4"
        src.write_bytes(b"src")
        p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                         status=ProjectStatus.ready, project_type="subtitle_removal", mode="assistant",
                         duration_sec=10.0, name=name)
        db.add(p)
        db.commit()
        self.pid = p.id
        db.close()

    def _t(self):
        return T0 + timedelta(seconds=next(_TICK))

    def attempt(self, status="active", reason=None, finished=None):
        db = self.SF()
        ap = ProjectAutopilot(project_id=self.pid, flow_id="subtitle_removal", flow_version=1,
                              status=status, status_reason=reason, params={}, created_at=self._t(),
                              updated_at=self._t(), finished_at=finished)
        db.add(ap)
        db.commit()
        db.close()
        self.ap_id = ap.id
        return ap.id

    def ask(self, question="Fill segment 3?", status="pending", cost=0, ap_id=None):
        db = self.SF()
        a = AutopilotApproval(autopilot_id=ap_id or self.ap_id, step_id="clean", step_generation=1,
                              name=f"a{self.n}", dedupe_key=f"d-{os.urandom(6).hex()}", kind="regeneration",
                              question=question, proposal={}, status=status, cost_mcredits=cost,
                              created_at=self._t())
        db.add(a)
        db.commit()
        db.close()
        return a.id


def _world(SF, tmp_path):
    """Five projects covering every case of the rules."""
    now = utcnow()
    live = P(SF, tmp_path, "Live")
    live.attempt()
    q1 = live.ask("Fill segment 3? It has caption leftovers.", cost=40000)
    live.ask("Old answered", status="consumed")

    paused = P(SF, tmp_path, "System paused")
    paused.attempt(status="paused", reason="manual paid action: fal_inpaint — Resume to continue")
    q2 = paused.ask("Accept segment 5?")

    oppaused = P(SF, tmp_path, "Operator paused")
    oppaused.attempt(status="paused", reason="paused by operator")
    q3 = oppaused.ask("Accept segment 7?")

    stopped = P(SF, tmp_path, "Stopped")
    stopped.attempt(status="stopped", reason="stopped by operator", finished=now - timedelta(hours=1))
    stopped.ask("Never answerable")

    failed = P(SF, tmp_path, "Failed")
    failed.attempt(status="failed", reason="step clean failed: boom", finished=now - timedelta(days=1))
    failed.ask("Dead question")

    replaced = P(SF, tmp_path, "Replaced")
    old = replaced.attempt(status="stopped", reason="the operator rejected the budget",
                           finished=now - timedelta(days=2))
    replaced.ask("From the old attempt", ap_id=old)
    replaced.attempt()                     # newer live attempt, nothing pending

    old_fail = P(SF, tmp_path, "Long ago")
    old_fail.attempt(status="failed", reason="ancient", finished=now - timedelta(days=9))
    return {"answerable": [q1, q2, q3], "pids": {
        "live": live.pid, "paused": paused.pid, "oppaused": oppaused.pid, "stopped": stopped.pid,
        "failed": failed.pid, "replaced": replaced.pid, "old": old_fail.pid}}


def test_badge_counts_exactly_what_the_cards_can_answer(SF, tmp_path):
    w = _world(SF, tmp_path)
    db = SF()
    try:
        b = inbox.badge(db)
        assert b.waiting == 3
        assert b.paused == 1                         # the system pause only
        # the same rule as the cards: every pending approval, judged by its latest attempt
        card_answerable = 0
        for a in db.query(AutopilotApproval).all():
            ap = db.get(ProjectAutopilot, a.autopilot_id)
            if links.is_answerable(a, links.latest_attempt(db, ap.project_id)):
                card_answerable += 1
        assert card_answerable == b.waiting
        assert sorted(a.id for a, _ in links.answerable(db)) == sorted(w["answerable"])
    finally:
        db.close()


def test_view_sections(SF, tmp_path):
    w = _world(SF, tmp_path)
    db = SF()
    try:
        v = inbox.view(db)
    finally:
        db.close()
    assert [q["id"] for q in v["to_answer"]] == w["answerable"]          # oldest first
    q = v["to_answer"][0]
    assert q["title"] == "Fill segment 3?" and q["cost"]["credits"] == 40
    assert q["path"] == f"/v2/projects/{w['pids']['live']}#ap-{q['id']}"
    assert [x["paused"] for x in v["to_answer"]] == [False, True, True]
    assert [(p["project"], p["by_operator"]) for p in v["paused"]] == [
        ("System paused", False), ("Operator paused", True)]
    # failed and system-stopped latest attempts of the week; not the operator stop,
    # not a replaced attempt, not one older than 7 days
    assert [e["project"] for e in v["ended"]] == ["Failed"]
    assert v["badge"].waiting == 3 and v["badge"].paused == 1


def test_inbox_page_and_badge_render(SF, tmp_path, client):
    w = _world(SF, tmp_path)
    r = client.get("/v2/assistant")
    assert r.status_code == 200
    html = r.text
    assert 'hx-get="/v2/assistant/badge"' in html                       # the header hook
    assert "To answer (3)" in html and "Paused (2)" in html
    assert f'href="/v2/projects/{w["pids"]["live"]}#ap-{w["answerable"][0]}"' in html
    assert "The assistant is paused — answer, then Resume." in html
    assert "You paused it" in html
    assert "Never answerable" not in html and "Dead question" not in html
    assert "Telegram notifications are off" in html
    b = client.get("/v2/assistant/badge")
    assert b.status_code == 200
    assert "Waiting for you · 3" in b.text and "· 1 paused" in b.text
    f = client.get("/v2/assistant/inbox-fragment")
    assert f.status_code == 200 and 'id="assistant-inbox"' in f.text


def test_empty_inbox(SF, client):
    r = client.get("/v2/assistant")
    assert r.status_code == 200 and "Nothing is waiting for you." in r.text
    b = client.get("/v2/assistant/badge")
    assert ">Waiting for you</a>" in b.text and "paused" not in b.text


def test_titles_are_escaped(SF, tmp_path, client):
    p = P(SF, tmp_path, "X <i>y</i>")
    p.attempt()
    p.ask("Use <script>alert(1)</script>?")
    html = client.get("/v2/assistant").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "X &lt;i&gt;y&lt;/i&gt;" in html


def test_failing_notifications_are_listed(SF, tmp_path, client):
    p = P(SF, tmp_path, "Notified")
    p.attempt()
    aid = p.ask()
    db = SF()
    db.add(AutopilotNotification(key=f"approval:{aid}", kind="approval", project_id=p.pid,
                                 approval_id=aid, chat_id="42", attempts=10,
                                 last_error="sendMessage: Bad Request: chat not found"))
    db.commit()
    db.close()
    html = client.get("/v2/assistant").text
    assert "Notification failing (1)" in html and "chat not found" in html


def test_option_cost_and_pause_time(SF, tmp_path):
    from app.models_autopilot import AutopilotJournal

    p = P(SF, tmp_path, "Loc")
    p.attempt(status="paused", reason="step x failed 3 times in a row: y")
    db = SF()
    db.add(AutopilotJournal(autopilot_id=p.ap_id, kind="paused", message="step x failed 3 times in a row: y",
                            at=utcnow() - timedelta(hours=5)))
    # a manual paid action while already paused writes another "paused" line
    db.add(AutopilotJournal(autopilot_id=p.ap_id, kind="paused", message="manual paid action: x",
                            at=utcnow() - timedelta(minutes=1)))
    a = AutopilotApproval(autopilot_id=p.ap_id, step_id="listen", step_generation=1, name="l",
                          dedupe_key="d-loc", kind="listening_final", question="Listen?", proposal={},
                          options=[{"id": "ship", "label": "Ship"},
                                   {"id": "regen", "label": "Regenerate", "cost_mcredits": 120000}])
    db.add(a)
    db.get(ProjectAutopilot, p.ap_id).updated_at = utcnow()      # an answer touched it just now
    db.commit()
    db.close()
    db = SF()
    try:
        v = inbox.view(db)
    finally:
        db.close()
    assert v["to_answer"][0]["cost"] == {"credits": 120.0, "usd": v["to_answer"][0]["cost"]["usd"], "upto": True}
    assert v["paused"][0]["ago"].startswith("5 h")


def test_cost_is_the_dearest_of_the_proposal_and_its_options(SF, tmp_path):
    p = P(SF, tmp_path, "Blocked")
    p.attempt()
    db = SF()
    db.add(AutopilotApproval(autopilot_id=p.ap_id, step_id="clean", step_generation=1, name="b",
                             dedupe_key="d-b", kind="regeneration", question="Blocked?", proposal={},
                             cost_mcredits=20000,
                             options=[{"id": "void", "label": "VOID"},
                                      {"id": "retry", "label": "Retry", "cost_mcredits": 300000}]))
    db.add(AutopilotApproval(autopilot_id=p.ap_id, step_id="clean", step_generation=1, name="c",
                             dedupe_key="d-c", kind="regeneration", question="Cheap options?", proposal={},
                             cost_mcredits=50000, options=[{"id": "x", "label": "X", "cost_mcredits": 10000}]))
    db.commit()
    db.close()
    db = SF()
    try:
        costs = {q["title"]: q["cost"] for q in inbox.view(db)["to_answer"]}
    finally:
        db.close()
    assert costs["Blocked?"]["credits"] == 300 and costs["Blocked?"]["upto"] is True
    assert costs["Cheap options?"]["credits"] == 50 and costs["Cheap options?"]["upto"] is False


def test_pause_time_survives_an_identical_later_line_and_a_long_reason(SF, tmp_path):
    from app.models_autopilot import AutopilotJournal

    reason = "manual paid action: fal_inpaint (fal:x, ~20 credits) — Resume to continue " + "z" * 4100
    p = P(SF, tmp_path, "Long")
    p.attempt(status="paused", reason=reason)
    db = SF()
    for hours in (30, 8, 6):                    # an older pause, resumed; then this one + a repeat
        db.add(AutopilotJournal(autopilot_id=p.ap_id, kind="paused", message=reason[:4000],
                                at=utcnow() - timedelta(hours=hours)))
    db.add(AutopilotJournal(autopilot_id=p.ap_id, kind="resumed", message="resumed by k",
                            at=utcnow() - timedelta(hours=20)))
    db.commit()
    db.close()
    db = SF()
    try:
        v = inbox.view(db)
    finally:
        db.close()
    assert v["paused"][0]["ago"].startswith("8 h")
