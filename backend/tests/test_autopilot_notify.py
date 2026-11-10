"""AICP-374: chat notifications of the assistant's questions (app/autopilot/notify.py).

A fake channel stands in for Telegram; every test gets its own database.
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

import httpx
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.autopilot import core, notify
from app.config import settings
from app.db import Base
from app.models import Run, VideoProject
from app.models_autopilot import (
    AutopilotApproval, AutopilotEffect, AutopilotJournal, AutopilotNotification as Note,
    AutopilotStep, ProjectAutopilot, utcnow,
)
from app.state_machine import ProjectStatus, RunStatus

BASE = "https://reskin.test"
T0 = utcnow() - timedelta(hours=1)


@pytest.fixture()
def SF(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'n.db'}", connect_args={"check_same_thread": False})

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
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", BASE, raising=False)


class FakeChannel:
    def __init__(self):
        self.calls = []          # (method, chat, message_id|None, text, reply_to)
        self.next_id = 100
        self.fail = []           # queue of ChannelError to raise on the next calls

    def _maybe_fail(self):
        if self.fail:
            raise self.fail.pop(0)

    def post(self, chat_id, text, *, reply_to=None):
        self._maybe_fail()
        self.next_id += 1
        self.calls.append(("post", chat_id, self.next_id, text, reply_to))
        return self.next_id

    def edit(self, chat_id, message_id, text):
        self._maybe_fail()
        self.calls.append(("edit", chat_id, message_id, text, None))

    def methods(self):
        return [c[0] for c in self.calls]


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t = self.t + timedelta(**kw)


@pytest.fixture()
def env(SF):
    ch = FakeChannel()
    clock = Clock(utcnow())
    n = notify.Notifier(ch, "42", session_factory=SF, sleep=lambda s: None, clock=clock, owner="test")
    return n, ch, clock


class W:
    """Writes the records of one assistant project."""

    def __init__(self, SF, tmp_path, name="Project A"):
        self.SF = SF
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
        self.n = 0

    def _t(self):
        self.n += 1
        return T0 + timedelta(seconds=self.n)

    def autopilot(self, *, status="active", at=None, reason=None):
        db = self.SF()
        ap = ProjectAutopilot(project_id=self.pid, flow_id="subtitle_removal", flow_version=1,
                              status=status, status_reason=reason, params={}, created_at=at or self._t())
        db.add(ap)
        db.commit()
        db.close()
        self.ap_id = ap.id
        return ap.id

    def ask(self, *, question="Segment 3 was refused. Fill it?", options=None, cost=0, proposal=None):
        db = self.SF()
        a = AutopilotApproval(autopilot_id=self.ap_id, step_id="clean", step_generation=1,
                              name=f"a{self.n}", dedupe_key=f"d-{os.urandom(6).hex()}", kind="regeneration",
                              question=question, proposal=proposal or {}, options=options,
                              cost_mcredits=cost, created_at=self._t())
        db.add(a)
        db.commit()
        db.close()
        return a.id

    def set_approval(self, aid, **kw):
        db = self.SF()
        a = db.get(AutopilotApproval, aid)
        for k, v in kw.items():
            setattr(a, k, v)
        db.commit()
        db.close()

    def set_ap(self, ap_id=None, *, journal_kind=None, **kw):
        db = self.SF()
        ap = db.get(ProjectAutopilot, ap_id or self.ap_id)
        for k, v in kw.items():
            setattr(ap, k, v)
        jid = None
        if journal_kind:
            j = AutopilotJournal(autopilot_id=ap.id, kind=journal_kind, message=ap.status_reason or "")
            db.add(j)
            db.flush()
            jid = j.id
        db.commit()
        db.close()
        return jid


def _rows(SF):
    db = SF()
    try:
        return {r.key: r for r in db.execute(select(Note)).scalars()}
    finally:
        db.close()


def _texts(ch, method=None):
    return [c[3] for c in ch.calls if method is None or c[0] == method]


# ---------------------------------------------------------------------------
# A question's message through its states
# ---------------------------------------------------------------------------


def test_new_question_posts_once_with_a_link_to_its_card(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask(cost=40000)
    s = n.run_pass()
    assert s.discovered == 1 and ch.methods() == ["post"]
    text = ch.calls[0][3]
    assert "Waiting for you" in text and "Segment 3 was refused." in text
    assert "Project A · project" in text
    assert f'href="{BASE}/v2/projects/{w.pid}#ap-{aid}"' in text
    assert "Cost: ~40 credits" in text
    assert ch.calls[0][1] == "42"
    n.run_pass()
    clock.advance(seconds=30)
    n.run_pass()
    assert ch.methods() == ["post"]           # nothing changed → no call


def test_answered_then_consumed_edits_the_same_message(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask(options=[{"id": "void", "label": "Fill with VOID"}, {"id": "asis", "label": "As is"}])
    n.run_pass()
    mid = ch.calls[0][2]
    w.set_approval(aid, status="approved", answered_decision="approve", chosen_option="void",
                   answered_by="operator", answered_at=utcnow())
    n.run_pass()
    assert ch.calls[-1][:3] == ("edit", "42", mid)
    assert "Answered by operator" in ch.calls[-1][3] and "approve (Fill with VOID)" in ch.calls[-1][3]
    w.set_approval(aid, status="consumed")
    n.run_pass()
    assert "✅ <b>Done</b>: approve (Fill with VOID)" in ch.calls[-1][3]
    assert _rows(SF)[f"approval:{aid}"].closed_at is not None
    # consumption reverted by reconcile (not_launched): the message follows
    w.set_approval(aid, status="approved")
    clock.advance(minutes=6)
    n.run_pass()
    assert "Answered by operator" in ch.calls[-1][3]
    assert _rows(SF)[f"approval:{aid}"].closed_at is None


def test_a_consumed_rejection_says_reject_not_approve(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    n.run_pass()
    w.set_approval(aid, status="consumed", answered_decision="reject", answered_by="operator")
    n.run_pass()
    assert "Done</b>: reject" in ch.calls[-1][3] and "approve" not in ch.calls[-1][3]


def test_superseded_after_and_before_an_answer(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    a1, a2 = w.ask(), w.ask(question="Second?")
    n.run_pass()
    w.set_approval(a1, status="superseded", answered_decision="approve", answered_by="k")
    w.set_approval(a2, status="superseded")
    n.run_pass()
    texts = _texts(ch, "edit")
    assert any("Answer not used" in t for t in texts)
    assert any("No longer needed" in t and "<s>Second?</s>" in t for t in texts)


def test_operator_stop_closes_the_question_and_sends_no_end_message(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    n.run_pass()
    clock.advance(seconds=10)
    w.set_ap(status="stopped", status_reason="stopped by operator", finished_at=clock(), journal_kind="stopped")
    n.run_pass()
    assert ch.methods() == ["post", "edit"]
    assert "No longer needed</b> — the assistant stopped" in ch.calls[-1][3]
    assert f"end:{w.ap_id}" not in _rows(SF)


def test_newer_attempt_replaces_the_question(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    old = w.autopilot()
    aid = w.ask()
    n.run_pass()
    w.set_ap(old, status="stopped", status_reason="stopped by k", finished_at=clock())
    w.autopilot(at=utcnow())
    n.run_pass()
    assert "No longer needed" in ch.calls[-1][3]


def test_answered_before_it_was_ever_posted_sends_nothing(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    db = SF()
    notify.discover(db, "42", clock())
    db.close()
    w.set_approval(aid, status="approved", answered_decision="approve", answered_by="k")
    n.run_pass()
    assert ch.calls == []
    row = _rows(SF)[f"approval:{aid}"]
    assert row.message_id is None and row.closed_at is None and row.state == "answered"


def test_titles_and_names_are_html_escaped(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path, name="A <b>&</b> B")
    w.autopilot()
    w.ask(question="Use <script>x</script> & go?")
    n.run_pass()
    t = ch.calls[0][3]
    assert "<script>" not in t and "&lt;script&gt;" in t and "A &lt;b&gt;&amp;&lt;/b&gt; B" in t


def test_stored_title_is_used(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    w.ask(question="Very long engineering sentence: alignment etc.", proposal={"_title": "Segment 3: fill?"})
    n.run_pass()
    assert "<b>Segment 3: fill?</b>" in ch.calls[0][3]


# ---------------------------------------------------------------------------
# Pauses and ends
# ---------------------------------------------------------------------------


def test_system_pause_posts_and_resume_closes_it_and_a_second_pause_is_new(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    n.run_pass()
    j1 = w.set_ap(status="paused", status_reason="step clean failed 3 times in a row: boom",
                  journal_kind="paused")
    n.run_pass()
    texts = _texts(ch)
    assert any("The assistant paused</b> — step clean failed 3 times" in t for t in texts)
    # the open question now says the assistant is paused
    assert any("answer, then Resume" in t for t in _texts(ch, "edit"))
    w.set_ap(status="active", status_reason=None, journal_kind="resumed")
    n.run_pass()
    assert any("No longer paused</b> — now running again" in t for t in _texts(ch, "edit"))
    j2 = w.set_ap(status="paused", status_reason="run 1234 has not moved for 3 h", journal_kind="paused")
    n.run_pass()
    rows = _rows(SF)
    assert f"pause:{j1}" in rows and f"pause:{j2}" in rows and j1 != j2
    assert sum(1 for c in ch.calls if c[0] == "post" and "The assistant paused" in c[3]) == 2


def test_operator_pause_sends_no_pause_message(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    w.set_ap(status="paused", status_reason="paused by operator", journal_kind="paused")
    n.run_pass()
    assert ch.calls == []


def test_manual_spend_line_while_already_paused_is_not_a_new_pause(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    w.set_ap(status="paused", status_reason="analysis failed: re-analyze the project, then Resume",
             journal_kind="paused")
    n.run_pass()
    db = SF()
    db.add(AutopilotJournal(autopilot_id=w.ap_id, kind="paused", message="manual paid action: x"))
    db.commit()
    db.close()
    n.run_pass()
    assert ch.methods() == ["post"]


def test_failure_after_switch_on_is_announced_history_is_not(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot(status="failed", reason="contract violation in step x: y")
    w.set_ap(finished_at=clock() - timedelta(hours=3))          # long before the first start
    n.run_pass()
    assert ch.calls == []
    w2 = W(SF, tmp_path, name="Project B")
    w2.autopilot()
    clock.advance(minutes=1)
    w2.set_ap(status="failed", status_reason="step clean failed: provider refused",
              finished_at=clock(), journal_kind="failed")
    n.run_pass()
    assert ch.methods() == ["post"]
    assert "The assistant ended</b> — step clean failed: provider refused" in ch.calls[0][3]
    assert f'href="{BASE}/v2/projects/{w2.pid}"' in ch.calls[0][3]
    n.run_pass()
    assert ch.methods() == ["post"]


def test_rejected_budget_stop_leaves_out_the_operator_comment(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    n.run_pass()
    aid = w.ask()
    w.set_approval(aid, status="consumed", answered_decision="reject", answer_comment="too pricey, Bob's call")
    clock.advance(minutes=1)
    w.set_ap(status="stopped", status_reason="the operator rejected the budget: too pricey, Bob's call",
             finished_at=clock())
    n.run_pass()
    end = [t for t in _texts(ch) if "The assistant ended" in t]
    assert end and "the operator rejected the budget" in end[0] and "pricey" not in end[0]


def test_completed_lists_the_runs(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    n.run_pass()
    db = SF()
    r = Run(project_id=w.pid, prompt="p", status=RunStatus.done, reference_image_urls=[])
    db.add(r)
    db.flush()
    db.add(AutopilotStep(autopilot_id=w.ap_id, step_id="launch", generation=1, status="completed"))
    db.add(AutopilotEffect(autopilot_id=w.ap_id, step_id="launch", step_generation=1, name="launch:1",
                           idempotency_key="k1", kind="action", handler="create_run", params={},
                           params_sha256="x", paid=True, state="launched", created_refs={"runs": [r.id]}))
    db.commit()
    short = r.short_id
    db.close()
    clock.advance(minutes=1)
    w.set_ap(status="completed", status_reason="all steps completed", finished_at=clock())
    n.run_pass()
    assert "The assistant finished" in ch.calls[-1][3] and short in ch.calls[-1][3]


# ---------------------------------------------------------------------------
# Delivery: claims, crashes, errors, reminders
# ---------------------------------------------------------------------------


def test_crash_between_send_and_save_gives_one_duplicate_never_a_lost_message(SF, tmp_path, env, monkeypatch):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    real = notify.Notifier._finish
    state = {"crash": True}

    def finish(self, db, rid, values, now):
        if state["crash"] and "message_id" in values:
            state["crash"] = False
            raise RuntimeError("killed after Telegram answered")
        return real(self, db, rid, values, now)

    monkeypatch.setattr(notify.Notifier, "_finish", finish)
    n.run_pass()                                  # posted; the write-back "crashed"
    assert ch.methods() == ["post"]
    db = SF()
    db.execute(Note.__table__.update().values(claim_owner="dead", claim_expires_at=clock() + notify.CLAIM_TTL,
                                              next_try_at=None))
    db.commit()
    db.close()
    clock.advance(seconds=30)
    n.run_pass()                                  # the dead sender's claim still holds
    assert ch.methods() == ["post"]
    clock.advance(seconds=notify.CLAIM_TTL.total_seconds())
    n.run_pass()                                  # claim expired → sent again: the one duplicate
    assert ch.methods() == ["post", "post"]
    row = _rows(SF)[f"approval:{aid}"]
    assert row.message_id == ch.calls[-1][2] and row.claim_owner is None
    clock.advance(seconds=30)
    n.run_pass()
    assert ch.methods() == ["post", "post"]


def test_unknown_outcome_is_retried_later_with_backoff(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    ch.fail.append(notify.ChannelError("sendMessage: ReadTimeout, outcome unknown", ambiguous=True))
    n.run_pass()
    row = _rows(SF)[f"approval:{aid}"]
    assert row.message_id is None and row.attempts == 1 and row.next_try_at > clock()
    assert row.claim_owner is None
    n.run_pass()
    assert ch.calls == []
    clock.advance(minutes=2)
    n.run_pass()
    assert ch.methods() == ["post"]
    assert _rows(SF)[f"approval:{aid}"].attempts == 0


def test_rate_limit_stops_the_pass_and_waits(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    w.ask(), w.ask(question="Two?")
    ch.fail.append(notify.ChannelError("sendMessage: Too Many Requests", retry_after=40))
    n.run_pass()
    assert ch.calls == [] and n.pause_until == clock() + timedelta(seconds=40)
    clock.advance(seconds=20)
    n.run_pass()
    assert ch.calls == []
    clock.advance(seconds=25)
    n.run_pass()
    assert ch.methods() == ["post", "post"]


def test_bad_chat_halts_sending_for_an_hour(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    w.ask()
    ch.fail.append(notify.ChannelError("sendMessage: Forbidden: bot was blocked by the user", halt=True))
    n.run_pass()
    assert n.halted and n.pause_until == clock() + timedelta(seconds=notify.HALT_SEC)


def test_deleted_message_is_reposted_only_while_it_waits(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    a1, a2 = w.ask(), w.ask(question="Two?")
    n.run_pass()
    # a1 still waits but its text changes (paused) and the message was deleted
    w.set_ap(status="paused", status_reason="paused by k")
    ch.fail += [notify.ChannelError("editMessageText: message to edit not found", gone=True)] * 2
    n.run_pass()
    rows = _rows(SF)
    assert rows[f"approval:{a1}"].message_id is None and not rows[f"approval:{a1}"].gone
    n.run_pass()
    assert ch.methods()[-2:] == ["post", "post"]
    # now a2 is answered and its message deleted: no re-post, marked gone
    w.set_approval(a2, status="approved", answered_decision="approve", answered_by="k")
    ch.fail.append(notify.ChannelError("editMessageText: message to edit not found", gone=True))
    before = len(ch.calls)
    n.run_pass()
    assert len(ch.calls) == before and _rows(SF)[f"approval:{a2}"].gone


def test_not_modified_counts_as_success(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    n.run_pass()
    w.set_approval(aid, status="approved", answered_decision="approve", answered_by="k")
    ch.fail.append(notify.ChannelError("editMessageText: message is not modified", not_modified=True))
    n.run_pass()
    row = _rows(SF)[f"approval:{aid}"]
    assert row.state == "answered" and row.attempts == 0


def test_reminders_after_2h_then_every_24h_as_replies(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    n.run_pass()
    mid = ch.calls[0][2]
    clock.advance(minutes=119)
    n.run_pass()
    assert ch.methods() == ["post"]
    clock.advance(minutes=2)
    n.run_pass()
    assert ch.calls[-1][0] == "post" and ch.calls[-1][4] == mid
    assert "Still waiting for you" in ch.calls[-1][3] and f"#ap-{aid}" in ch.calls[-1][3]
    clock.advance(hours=23)
    n.run_pass()
    assert len(ch.calls) == 2
    clock.advance(hours=1, minutes=1)
    n.run_pass()
    assert len(ch.calls) == 3 and ch.calls[-1][4] == mid
    w.set_approval(aid, status="approved", answered_decision="approve", answered_by="k")
    n.run_pass()
    clock.advance(hours=25)
    n.run_pass()
    assert [c[0] for c in ch.calls[3:]] == ["edit"]


def test_failing_rows_are_listed(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    for _ in range(notify.FAILING_AFTER):
        ch.fail.append(notify.ChannelError("sendMessage: Bad Request: something"))
        n.run_pass()
        clock.advance(hours=2)
    db = SF()
    try:
        assert [r.approval_id for r in notify.failing(db)] == [aid]
        assert notify.status(db)["failing"][0]["attempts"] == notify.FAILING_AFTER
    finally:
        db.close()


def test_chat_change_keeps_old_messages_in_their_chat(SF, tmp_path, env):
    n, ch, _ = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    n.run_pass()
    n.chat_id = "99"
    w.ask(question="New one?")
    w.set_approval(aid, status="approved", answered_decision="approve", answered_by="k")
    n.run_pass()
    chats = {c[0]: c[1] for c in ch.calls[1:]}
    assert chats == {"edit": "42", "post": "99"}


# ---------------------------------------------------------------------------
# The Telegram adapter and config
# ---------------------------------------------------------------------------


TOKEN = "123456:SECRET-token-value"


def _tg(handler):
    return notify.TelegramChannel(TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_telegram_payloads():
    seen = []

    def handler(req):
        import json

        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})

    ch = _tg(handler)
    assert ch.post("42", "<b>x</b>", reply_to=5) == 7
    ch.edit("42", 7, "y")
    (p1, b1), (p2, b2) = seen
    assert p1.endswith("/sendMessage") and b1["parse_mode"] == "HTML"
    assert b1["reply_parameters"] == {"message_id": 5, "allow_sending_without_reply": True}
    assert b1["link_preview_options"] == {"is_disabled": True}
    assert p2.endswith("/editMessageText") and b2["message_id"] == 7


@pytest.mark.parametrize("status,body,flag", [
    (429, {"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 12",
           "parameters": {"retry_after": 12}}, "retry_after"),
    (400, {"ok": False, "error_code": 400, "description": "Bad Request: message is not modified"}, "not_modified"),
    (400, {"ok": False, "error_code": 400, "description": "Bad Request: message to edit not found"}, "gone"),
    (403, {"ok": False, "error_code": 403, "description": "Forbidden: bot was blocked by the user"}, "halt"),
    (400, {"ok": False, "error_code": 400, "description": "Bad Request: group chat was upgraded",
           "parameters": {"migrate_to_chat_id": -100123}}, "halt"),
    (502, {}, "ambiguous"),
])
def test_telegram_errors_are_classified(status, body, flag):
    ch = _tg(lambda req: httpx.Response(status, json=body))
    with pytest.raises(notify.ChannelError) as ei:
        ch.edit("42", 1, "x")
    assert getattr(ei.value, flag)


def test_the_token_never_appears_in_an_error():
    ch = _tg(lambda req: httpx.Response(400, json={"ok": False, "error_code": 400,
                                                   "description": f"Bad Request: bot{TOKEN} odd"}))
    with pytest.raises(notify.ChannelError) as ei:
        ch.post("42", "x")
    assert TOKEN not in str(ei.value) and "<token>" in str(ei.value)

    def boom(req):
        raise httpx.ReadTimeout(f"timed out on {req.url}", request=req)

    with pytest.raises(notify.ChannelError) as ei:
        _tg(boom).post("42", "x")
    assert TOKEN not in str(ei.value) and ei.value.ambiguous


def test_config_problem(monkeypatch):
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "", raising=False)
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "42", raising=False)
    assert notify.config_problem().startswith("off")
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "t", raising=False)
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "", raising=False)
    assert notify.config_problem().startswith("refused")
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", BASE, raising=False)
    assert notify.config_problem() is None


def test_httpx_request_logs_are_silenced_so_the_token_url_is_not_logged():
    import logging

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING


def test_a_lost_claim_forgets_the_hash_so_the_next_pass_re_edits(SF, tmp_path, env):
    n, ch, clock = env
    w = W(SF, tmp_path)
    w.autopilot()
    aid = w.ask()
    n.run_pass()
    w.set_approval(aid, status="approved", answered_decision="approve", answered_by="k")
    real_edit = ch.edit

    def slow_edit(chat, mid, text):
        # the lease expires during the call and another sender takes the row
        db = SF()
        db.execute(Note.__table__.update().values(claim_owner="other", state_hash="newer"))
        db.commit()
        db.close()
        real_edit(chat, mid, text)

    ch.edit = slow_edit
    n.run_pass()
    row = _rows(SF)[f"approval:{aid}"]
    assert row.state_hash is None and row.claim_owner == "other"
    ch.edit = real_edit
    db = SF()
    db.execute(Note.__table__.update().values(claim_owner=None))
    db.commit()
    db.close()
    n.run_pass()
    assert ch.calls[-1][0] == "edit" and _rows(SF)[f"approval:{aid}"].state_hash is not None


def test_many_open_messages_do_not_starve_a_new_question(SF, tmp_path, env, monkeypatch):
    n, ch, clock = env
    monkeypatch.setattr(notify, "MAX_CALLS_PER_PASS", 1000)
    w = W(SF, tmp_path)
    w.autopilot()
    for _ in range(30):
        w.ask()
    n.run_pass()
    assert len(ch.calls) == 30
    newest = w.ask(question="The newest?")
    clock.advance(seconds=10)
    n.run_pass()
    assert ch.calls[-1][0] == "post" and "The newest?" in ch.calls[-1][3]
