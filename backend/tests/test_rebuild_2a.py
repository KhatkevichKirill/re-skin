"""Rebuild wave 2a (docs/rebuild-from-reference-design.md §5.2, §5.3, §8, §10)
on SQLite: the storyboard job, keyframes, their step handlers, the API, the
reapers and the rendered project page.

No provider is ever reached: kie.ai is a fake that COUNTS every POST, and the
media helpers (ffmpeg) are stubbed except in the one proxy test that runs the
real encoder. The Postgres-only parts (concurrent admissions, selection under
concurrency) are in tests/test_rebuild_pg.py.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from sqlalchemy import create_engine, event, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import rebuild_analysis, rebuild_jobs as jobs, rebuild_keyframes as kfmod  # noqa: E402
from app import rebuild_llm_steps, rebuild_paid_call as pc, rebuild_storyboard as sbmod  # noqa: E402
from app import spend  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.kie_client import SendOutcome  # noqa: E402
from app.models import KieSpendDraw, KieSpendReservation, Run, SegmentDef, VideoProject  # noqa: E402
from app.rebuild_models import (  # noqa: E402
    ProjectStoryboard,
    RebuildKeyframe,
    RebuildLlmCall,
    RebuildShot,
)
from app.state_machine import ProjectStatus, RunStatus  # noqa: E402

NOW = datetime(2026, 11, 3, 12, 0, 0)
# 7 detected shots, the 0.5 s one folds into the next: 6 anchors, 2 chunks.
RANGES = [(0, 3), (3, 6), (6, 6.5), (6.5, 10), (10, 14), (14, 18), (18, 20)]

GLOBAL = {
    "style": "3D animated cartoon",
    "characters": [{"id": "C1", "name_hint": "Hero", "look": "tall", "outfit": "suit",
                    "role": "main"},
                   {"id": "C2", "name_hint": "Friend", "look": "short", "outfit": "dress",
                    "role": "secondary"}],
    "sheet_groups": [{"key": "street", "description": "street clothes",
                      "characters": ["C1", "C2"]},
                     {"key": "wedding", "description": "wedding outfits",
                      "characters": ["C1"]}],
    "audio": {"kind": "voiceover", "lines": ["hello"]},
    "on_screen_text": ["SALE 50%"],
}


def shot_answer(**kw):
    out = {"location": "street", "framing": "close-up", "camera": "slow push-in",
           "characters": ["C1"], "action": "Hero walks.", "speaking_on_camera": None,
           "is_brand_card": False, "sheet_key": "street"}
    out.update(kw)
    return out


def accepted(obj) -> SendOutcome:
    content = obj if isinstance(obj, str) else "```json\n" + json.dumps(obj) + "\n```"
    body = {"choices": [{"message": {"content": content}}]}
    return SendOutcome("accepted", content=content, body=body, http_status=200)


def _text(messages) -> str:
    return " ".join(p.get("text", "") for p in messages[0]["content"] if p.get("type") == "text")


class FakeKie:
    """Counts every provider POST; never touches the network."""

    def __init__(self):
        self.chat: list = []
        self.create: list = []
        self.uploads: list = []
        self.get_calls: list = []
        self.download_fail = 0
        self.responder = self.default_responder
        self.task_state = "success"

    @staticmethod
    def default_responder(model, messages):
        if "shot-by-shot remake" in _text(messages):
            return accepted(GLOBAL)
        return accepted(shot_answer())

    def upload_file(self, path, upload_path="", file_name=None):
        assert os.path.exists(path)
        self.uploads.append(file_name)
        return f"https://up.example/{file_name}"

    def chat_completion_once(self, model, messages, *, timeout, response_format=None):
        self.chat.append((model, _text(messages)))
        return self.responder(model, messages)

    def create_task_once(self, payload, *, timeout):
        self.create.append(payload)
        return SendOutcome("accepted", task_id=f"T{len(self.create)}", http_status=200)

    def get_task(self, task_id):
        self.get_calls.append(task_id)
        return {"state": self.task_state, "creditsConsumed": 9,
                "resultJson": json.dumps({"resultUrls": [f"https://cdn.example/{task_id}.png"]})}

    def download_result(self, url, dst):
        if self.download_fail:
            self.download_fail -= 1
            raise RuntimeError("cdn closed the connection")
        from PIL import Image

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.new("RGB", (36, 64), (10, 20, 30)).save(dst, format="PNG")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def SF():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _fk(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    Base.metadata.drop_all(eng)
    eng.dispose()


class Harness:
    """Captures every enqueue and runs the jobs in order, as RQ would."""

    def __init__(self, SF):
        self.SF = SF
        self.steps: list = []
        self.sb_jobs: list = []
        self.ran: list = []

    def drive(self, max_iter=500, only=None):
        for _ in range(max_iter):
            if self.steps:
                table, row_id, gen, step = self.steps.pop(0)
                path = pc.STEP_HANDLERS[table]
                mod, fn = path.rsplit(".", 1)
                job_id = pc.step_job_id(table, row_id, gen, step,
                                        "s" if step in (pc.STEP_POLL, pc.STEP_FETCH) else None)
                self.ran.append((table, row_id, gen, step))
                getattr(importlib.import_module(mod), fn)(row_id, gen, step, job_id)
                continue
            if self.sb_jobs:
                sb_id, gen, token = self.sb_jobs.pop(0)
                self.ran.append(("storyboard", sb_id, gen, token))
                sbmod.run_storyboard_job(sb_id, gen, token, sf=self.SF)
                continue
            return
        raise AssertionError("the job chain did not settle")


@pytest.fixture()
def env(SF, monkeypatch):
    fake = FakeKie()
    h = Harness(SF)
    monkeypatch.setattr(settings, "REBUILD_ENABLED", True)
    monkeypatch.setattr(settings, "LOCALISATION_TRANSCRIBE_MODEL", "gemini-3.1-pro")
    monkeypatch.setattr(settings, "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS", "gemini-3-pro,flash")
    monkeypatch.setattr(jobs, "kie_client", lambda: fake)
    monkeypatch.setattr(jobs, "session_factory", lambda: SF)
    monkeypatch.setattr(rebuild_llm_steps, "_session_factory", lambda: SF)
    monkeypatch.setattr(pc, "_BOOT_ID", "boot-test")
    monkeypatch.setattr(pc, "enqueue_step",
                        lambda table, row_id, gen, step, delay_sec=0, seq=None:
                        h.steps.append((table, row_id, int(gen), step)) or "jid")
    monkeypatch.setattr(sbmod, "_enqueue",
                        lambda sb_id, gen, token, jid, delay_sec=0:
                        h.sb_jobs.append((sb_id, int(gen), token)))

    def fake_proxy(src, dst, *, vertical, start=None, end=None, max_bytes=0):
        with open(dst, "wb") as fh:
            fh.write(b"proxy")
        return dst

    def fake_frame(src, t, dst, *, width=None, quality=2):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(b"jpg")
        return dst

    monkeypatch.setattr(jobs, "build_llm_proxy", fake_proxy)
    monkeypatch.setattr(jobs, "extract_frame", fake_frame)
    h.fake = fake
    return h


@pytest.fixture()
def db(SF):
    s = SF()
    yield s
    s.rollback()
    s.close()


_SRC_DIR = tempfile.mkdtemp(prefix="rb2a-src-")


def make_project(SF, *, ranges=RANGES, status=ProjectStatus.ready, project_type="rebuild"):
    src = os.path.join(_SRC_DIR, f"{uuid.uuid4().hex}.mp4")
    with open(src, "wb") as fh:
        fh.write(b"src")
    with SF() as s:
        p = VideoProject(source_type="upload", source_ref="s.mp4", status=ProjectStatus.analyzing,
                         project_type=project_type, source_local_path=src, width=720,
                         height=1280, duration_sec=float(ranges[-1][1]), fps=30.0)
        s.add(p)
        s.flush()
        rebuild_analysis.publish_analysis(s, p, ranges)
        p.status = status
        s.commit()
        return p.id


@pytest.fixture()
def client(SF, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api_v2 as api_v2_module
    from app.db import get_db
    from app.main import app

    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project", lambda pid, **kw: None)

    def override():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app.dependency_overrides[get_db] = override
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def _calls(SF, sb_id=None, **kw):
    with SF() as s:
        q = select(RebuildLlmCall)
        if sb_id:
            q = q.where(RebuildLlmCall.storyboard_id == sb_id)
        return list(s.execute(q.order_by(RebuildLlmCall.created_at, RebuildLlmCall.id)).scalars())


def _sb(SF, sb_id) -> ProjectStoryboard:
    with SF() as s:
        return s.get(ProjectStoryboard, sb_id)


def _shots(SF, pid):
    with SF() as s:
        return rebuild_analysis.project_shots(s, pid)


def build_storyboard(client, env, pid, **data):
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data=data or {"mode": "resume"})
    assert r.status_code == 202, r.text
    env.drive()
    return r.json()["storyboard_id"]


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------


def test_storyboard_happy_path_one_post_per_row(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 202, r.text
    sb_id = r.json()["storyboard_id"]
    # Budget = (1 global + 6 shots) x 2 (one fallback each) x llm_credits.
    assert r.json()["budget_credits"] == 7 * 2 * spend.llm_credits()
    with SF() as s:
        res = s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend.storyboard_scope(sb_id))).scalar_one()
        assert res.reserved_mcredits == 7 * 2 * spend.llm_credits() * 1000
    env.drive()
    sb = _sb(SF, sb_id)
    assert sb.status == "succeeded", sb.error
    assert sb.applied_at is not None and sb.model == "gemini-3.1-pro"
    calls = _calls(SF, sb_id)
    assert [c.kind for c in calls].count("global") == 1
    assert [c.kind for c in calls].count("shot") == 6
    assert all(c.status == "succeeded" for c in calls)
    # Exactly one provider POST per row.
    assert len(env.fake.chat) == len(calls) == 7
    # Every row drew its estimate, keyed by the row.
    with SF() as s:
        keys = {d.call_key for d in s.execute(select(KieSpendDraw)).scalars()}
    assert {f"rebuild:{c.id}" for c in calls} <= keys
    shots = _shots(SF, pid)
    anchors = [x for x in shots if x.folded_into is None]
    assert all(x.action == "Hero walks." and x.sheet_key == "street" for x in anchors)
    assert all(x.action is None for x in shots if x.folded_into is not None)
    # Global prompt names the shot starts, the shot calls carry the context.
    assert "shot-by-shot remake" in env.fake.chat[0][1]
    assert all('"C1"' in t for _, t in env.fake.chat[1:])
    # Plain upload names (the pilot's dotted name was refused).
    assert all(n.replace(".mp4", "").replace(".jpg", "").isalnum() for n in env.fake.uploads)


def test_storyboard_file_unavailable_admits_the_next_model_as_a_new_row(client, env, SF):
    pid = make_project(SF)
    seen = {"global": 0, "shot1": 0}

    def responder(model, messages):
        text = _text(messages)
        if "shot-by-shot remake" in text:
            seen["global"] += 1
            if model == "gemini-3.1-pro":
                return SendOutcome("refused", http_status=200, file_unavailable=True,
                                   error="400 Failed to get the file information")
            return accepted(GLOBAL)
        if "from 3.00 s" in text:  # shot 2: an invalid first answer, then a good one
            seen["shot1"] += 1
            if seen["shot1"] == 1:
                return accepted("I cannot describe this.")
        return accepted(shot_answer())

    env.fake.responder = responder
    sb_id = build_storyboard(client, env, pid)
    sb = _sb(SF, sb_id)
    assert sb.status == "succeeded", sb.error
    globals_ = [c for c in _calls(SF, sb_id) if c.kind == "global"]
    assert [(g.model, g.status) for g in globals_] == [("gemini-3.1-pro", "failed"),
                                                       ("gemini-3-pro", "succeeded")]
    assert globals_[0].cost_credits == 0.0  # a refusal is settled to 0
    assert sb.global_call_id == globals_[1].id and sb.model == "gemini-3-pro"
    shot2 = [s for s in _shots(SF, pid) if s.index == 1][0]
    rows = [c for c in _calls(SF, sb_id) if c.shot_id == shot2.id]
    assert [r.model for r in rows] == ["gemini-3.1-pro", "gemini-3-pro"]
    assert len(env.fake.chat) == len(_calls(SF, sb_id))


def test_unknown_global_call_is_never_resent(client, env, SF):
    pid = make_project(SF)
    env.fake.responder = lambda m, msgs: SendOutcome("unknown", error="read timeout")
    sb_id = build_storyboard(client, env, pid)
    sb = _sb(SF, sb_id)
    assert sb.status == "failed" and "may have been billed" in sb.error
    calls = _calls(SF, sb_id)
    assert [c.status for c in calls] == ["unknown"] and len(env.fake.chat) == 1
    with SF() as s:
        (draw,) = s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key == f"rebuild:{calls[0].id}")).scalars()
        assert draw.mcredits == spend.llm_credits() * 1000  # stays at the estimate


def test_storyboard_stops_at_its_budget(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    with SF() as s:  # room for the global call and 2 shot calls only
        s.get(ProjectStoryboard, sb_id).budget_credits = 3 * spend.llm_credits()
        s.commit()
    env.drive()
    sb = _sb(SF, sb_id)
    calls = _calls(SF, sb_id)
    assert len(calls) == 3 and len(env.fake.chat) == 3
    assert sb.status == "failed" and "could not be described" in sb.error
    assert list(sb.result["errors"].values()).count(sbmod.BUDGET_USED_UP) == 4


def test_cancel_cancels_unsent_calls_in_the_same_transaction(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    # First advance: the global call is admitted (queued); claim it.
    env.sb_jobs and env.sb_jobs.pop(0)
    sbmod.run_storyboard_job(sb_id, 1, None, sf=SF)
    (table, gid, gen, step) = env.steps.pop(0)
    sbmod.llm_step(gid, gen, step, "j", sf=SF)  # claimed + request prepared
    assert _calls(SF, sb_id)[0].status == "claimed"
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard/cancel", data={"x": "1"})
    assert r.status_code == 200, r.text
    assert _sb(SF, sb_id).status == "interrupted"
    (row,) = _calls(SF, sb_id)
    assert (row.status, row.error, row.claim_token) == ("failed", pc.CANCELLED_MESSAGE, None)
    # The send step queued before the cancel is fenced out: no POST.
    env.drive()
    assert env.fake.chat == []
    # A cancelled call is not counted against the next storyboard's budget.
    assert sbmod.committed_credits([row]) == 0.0


def test_build_again_resumes_from_the_previous_global_call(client, env, SF):
    pid = make_project(SF)
    state = {"n": 0}

    def flaky(model, messages):
        if "shot-by-shot remake" in _text(messages):
            return accepted(GLOBAL)
        state["n"] += 1
        return SendOutcome("unknown", error="timeout") if state["n"] <= 2 else accepted(shot_answer())

    env.fake.responder = flaky
    first = build_storyboard(client, env, pid)
    assert _sb(SF, first).status == "failed"  # 2 of 6 shots unknown > 10 %
    posts = len(env.fake.chat)
    env.fake.responder = FakeKie.default_responder
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 202, r.text
    second = r.json()["storyboard_id"]
    assert r.json()["reused_global_call_id"] == _sb(SF, first).global_call_id
    assert r.json()["budget_credits"] == 2 * 2 * spend.llm_credits()  # only the 2 shots
    env.drive()
    assert _sb(SF, second).status == "succeeded"
    new = _calls(SF, second)
    assert [c.kind for c in new] == ["shot", "shot"]  # no new global, only the 2 unknowns
    assert len(env.fake.chat) == posts + 2
    # From scratch calls everything again.
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "from_scratch"})
    assert r.status_code == 202 and r.json()["reused_global_call_id"] is None
    env.drive()
    assert len(env.fake.chat) == posts + 2 + 7


def test_build_again_waits_for_a_sending_call(client, env, SF):
    pid = make_project(SF)
    sb_id = build_storyboard(client, env, pid)
    with SF() as s:
        row = s.get(RebuildLlmCall, _calls(SF, sb_id)[1].id)
        row.status = "sending"
        s.commit()
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 409 and "previous worker" in r.text
    with SF() as s:
        s.get(RebuildLlmCall, row.id).status = "unknown"
        s.commit()


def test_edited_shots_are_kept_unless_overwrite(client, env, SF):
    pid = make_project(SF)
    build_storyboard(client, env, pid)
    shot = _shots(SF, pid)[0]
    r = client.patch(f"/api/v2/rebuild/shots/{shot.id}", data={"action": "My own words."})
    assert r.status_code == 200, r.text
    build_storyboard(client, env, pid, mode="from_scratch")
    assert _shots(SF, pid)[0].action == "My own words."
    build_storyboard(client, env, pid, mode="resume", overwrite_edits="true")
    s0 = _shots(SF, pid)[0]
    assert s0.action == "Hero walks." and s0.edited_at is None


def test_speech_is_warned_not_blocked(client, env, SF):
    pid = make_project(SF)
    talky = dict(GLOBAL, audio={"kind": "on_camera_dialogue", "lines": []})
    env.fake.responder = lambda m, msgs: accepted(
        talky if "shot-by-shot remake" in _text(msgs) else shot_answer(speaking_on_camera="C1"))
    sb_id = build_storyboard(client, env, pid)
    sb = _sb(SF, sb_id)
    assert sb.status == "succeeded" and sb.result["warnings"] == [sbmod.SPEECH_WARNING]
    r = client.get(f"/api/v2/projects/{pid}/rebuild/storyboard")
    assert r.json()["warnings"] == [sbmod.SPEECH_WARNING]


def test_validation_rejects_unknown_ids_and_sheet_keys():
    with pytest.raises(sbmod.InvalidAnswer):
        sbmod.validate_shot(shot_answer(characters=["C9"]), sbmod.validate_global(GLOBAL))
    with pytest.raises(sbmod.InvalidAnswer):
        sbmod.validate_shot(shot_answer(sheet_key="beach"), sbmod.validate_global(GLOBAL))
    with pytest.raises(sbmod.InvalidAnswer):
        sbmod.validate_global(dict(GLOBAL, audio={"kind": "talking"}))
    ok = sbmod.validate_shot(shot_answer(sheet_key="Street", is_brand_card="true"),
                             sbmod.validate_global(GLOBAL))
    assert ok["sheet_key"] == "street" and ok["is_brand_card"] is True


@pytest.mark.parametrize("raw", [None, "", "null"])
def test_validation_accepts_a_null_sheet_key_when_groups_exist(raw):
    # Prod 2026-11-03 (project 16eca3ae): 8 of 51 shots (night street, car,
    # logo, no-character close-ups) answered sheet_key null on both models and
    # the whole storyboard failed. A null key must mean "no group".
    ctx = sbmod.validate_global(GLOBAL)
    assert [g["key"] for g in ctx["sheet_groups"]]
    for chars in (["C1"], []):
        ok = sbmod.validate_shot(shot_answer(sheet_key=raw, characters=chars), ctx)
        assert ok["sheet_key"] is None


def test_model_chain_and_next_model(monkeypatch):
    monkeypatch.setattr(settings, "LOCALISATION_TRANSCRIBE_MODEL", "gemini-3.1-pro")
    monkeypatch.setattr(settings, "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS",
                        " gemini-3-pro , gemini-3.1-pro,flash")
    assert sbmod.model_chain() == ["gemini-3.1-pro", "gemini-3-pro", "flash"]
    assert sbmod.next_model("gemini-3.1-pro") == "gemini-3-pro"
    assert sbmod.next_model("flash") is None


# ---------------------------------------------------------------------------
# Storyboard reaper
# ---------------------------------------------------------------------------


def test_storyboard_reaper_requeues_queued_and_interrupts_running(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    env.sb_jobs.clear()  # the doorbell was lost
    enq = []
    common = dict(session_factory=SF, is_live=lambda prefix: False,
                  enqueue_fn=lambda *a: enq.append(a))
    assert sbmod.reap(now=NOW, **common)["observed"] == 1
    assert sbmod.reap(now=NOW + timedelta(seconds=30), **common)["requeued"] == 0
    assert sbmod.reap(now=NOW + timedelta(seconds=200), **common)["requeued"] == 1
    assert enq == [(sb_id, 2, sbmod.job_id(sb_id, 2, "0"))]
    # The stale gen-1 doorbell claims nothing.
    assert sbmod.run_storyboard_job(sb_id, 1, None, sf=SF) is None
    assert sbmod.run_storyboard_job(sb_id, 2, None, sf=SF) == "running"
    # Global call admitted + claimed (prepared), its continuation lost.
    env.sb_jobs.clear()
    (_, gid, gen, step) = env.steps.pop(0)
    sbmod.llm_step(gid, gen, step, "j", sf=SF)
    env.steps.clear()  # the send job dies with the worker
    with SF() as s:
        s.get(ProjectStoryboard, sb_id).not_live_since = None
        s.commit()
    assert sbmod.reap(now=NOW + timedelta(seconds=300), **common)["observed"] == 1
    out = sbmod.reap(now=NOW + timedelta(seconds=500), **common)
    assert out["interrupted"] == 1
    sb = _sb(SF, sb_id)
    assert sb.status == "interrupted" and sb.claim_token is None
    (row,) = _calls(SF, sb_id)
    assert row.status == "failed" and row.error == pc.CANCELLED_MESSAGE
    assert env.fake.chat == []


def test_storyboard_reaper_leaves_a_live_chain_alone(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    out = sbmod.reap(session_factory=SF, is_live=lambda prefix: prefix.startswith(
        f"rb-sb:{sb_id}:1:"), enqueue_fn=lambda *a: None, now=NOW + timedelta(days=1))
    assert {k: out[k] for k in ("observed", "requeued", "interrupted")} == {
        "observed": 0, "requeued": 0, "interrupted": 0}
    env.drive()


# ---------------------------------------------------------------------------
# Single send across reaper / recovery races (design §7.2)
# ---------------------------------------------------------------------------


def _ready_with_storyboard(client, env, SF):
    pid = make_project(SF)
    build_storyboard(client, env, pid)
    env.fake.chat.clear()
    return pid


def test_keyframe_single_send_across_reaper_requeue_and_duplicate_delivery(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][0]
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes", data={"shots": shot.id})
    assert r.status_code == 202, r.text
    (kid,) = r.json()["keyframe_ids"]
    (claim1,) = env.steps
    env.steps.clear()
    kfmod.step(kid, 1, "claim", "j-claim", sf=SF)
    (send1,) = env.steps
    env.steps.clear()
    # The worker dies before the send job runs; the reaper re-drives the claimed row.
    with SF() as s:
        s.get(RebuildKeyframe, kid).not_live_since = NOW - timedelta(seconds=999)
        s.commit()
    enq = []
    pc.reap(session_factory=SF, conn=object(), is_live=lambda q, p: False,
            evidence=lambda row: None, enqueue=lambda *a: enq.append(a), now=NOW,
            models=(RebuildKeyframe,))
    assert enq == [("rebuild_keyframes", kid, 2, "claim")]
    # The old generation's send job arrives late: fenced, no POST.
    assert kfmod.step(kid, 1, "send", "j-old-send", sf=SF) is None
    assert env.fake.create == []
    kfmod.step(kid, 2, "claim", "j-claim2", sf=SF)
    assert kfmod.step(kid, 2, "send", "rb:k:2:send", sf=SF) == "submitted"
    # A duplicate delivery of the same send job: nothing.
    assert kfmod.step(kid, 2, "send", "rb:k:2:send-dup", sf=SF) is None
    assert len(env.fake.create) == 1
    env.steps.clear()
    kfmod.step(kid, 2, "poll", "p", sf=SF)
    kfmod.step(kid, 2, "fetch", "f", sf=SF)
    with SF() as s:
        kf = s.get(RebuildKeyframe, kid)
        assert kf.status == "succeeded" and kf.selected and kf.width == 36
    assert len(env.fake.create) == 1


def test_sending_row_waits_for_proof_and_is_never_resent(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][1]
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes", data={"shots": shot.id})
    (kid,) = r.json()["keyframe_ids"]
    env.steps.clear()
    kfmod.step(kid, 1, "claim", "j", sf=SF)
    env.steps.clear()
    # Crash right after the `sending` commit, before the POST.
    with SF() as s:
        row = s.get(RebuildKeyframe, kid)
        assert pc.begin_send(s, RebuildKeyframe, kid, gen=1, token=row.claim_token,
                             request=row.request, scope_key=spend.keyframes_scope(row.batch_id),
                             sender=pc.SenderIdentity("h", "boot-test", "rb:x"))
        s.commit()
        s.get(RebuildKeyframe, kid).not_live_since = NOW - timedelta(seconds=999)
        s.commit()
    common = dict(session_factory=SF, conn=object(), is_live=lambda q, p: False,
                  enqueue=lambda *a: None, now=NOW, models=(RebuildKeyframe,))
    assert pc.reap(evidence=lambda row: None, **common)["waiting"] == 1
    assert kfmod.step(kid, 1, "send", "j2", sf=SF) is None  # a re-delivered send: nothing
    assert pc.reap(evidence=lambda row: "boot changed", **common)["unknown"] == 1
    assert env.fake.create == []
    with SF() as s:
        assert s.get(RebuildKeyframe, kid).status == "unknown"


def test_llm_send_delivered_twice_posts_once(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    sbmod.run_storyboard_job(sb_id, 1, None, sf=SF)
    env.sb_jobs.clear()
    (_, gid, gen, _) = env.steps.pop(0)
    rebuild_llm_steps.handle(gid, gen, "claim", "c")
    assert rebuild_llm_steps.handle(gid, gen, "send", "s1") == "succeeded"
    assert rebuild_llm_steps.handle(gid, gen, "send", "s1") is None
    assert len(env.fake.chat) == 1
    env.steps.clear()
    env.sb_jobs.clear()
    with SF() as s:
        sbmod.cancel_storyboard(s, s.get(VideoProject, pid))
        s.commit()


def test_send_refuses_without_a_boot_id(client, env, SF, monkeypatch):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][2]
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes", data={"shots": shot.id})
    (kid,) = r.json()["keyframe_ids"]
    env.steps.clear()
    kfmod.step(kid, 1, "claim", "j", sf=SF)
    monkeypatch.setattr(pc, "_BOOT_ID", None)
    assert kfmod.step(kid, 1, "send", "j", sf=SF) is None
    with SF() as s:
        assert s.get(RebuildKeyframe, kid).status == "claimed"
    assert env.fake.create == []
    monkeypatch.setattr(pc, "_BOOT_ID", "boot-test")
    env.steps.clear()
    env.drive()


def test_unknown_llm_kind_logs_an_error_and_does_nothing(SF, env, monkeypatch):
    pid = make_project(SF)
    with SF() as s:
        sb = ProjectStoryboard(project_id=pid, shots_generation=1, status="failed")
        s.add(sb)
        s.flush()
        row = RebuildLlmCall(project_id=pid, storyboard_id=sb.id, kind="global", model="m",
                             estimate_credits=1.0)
        s.add(row)
        s.commit()
        rid = row.id
    import app.rebuild_llm_steps as steps_mod

    original = dict(steps_mod.KIND_HANDLERS)
    steps_mod.KIND_HANDLERS.pop("global")
    try:
        # A recording logger (alembic's fileConfig in another module disables
        # existing loggers, which would blind caplog).
        errors = []
        monkeypatch.setattr(steps_mod.log, "error", lambda *a, **k: errors.append(a[0] % a[1:]))
        assert rebuild_llm_steps.handle(rid, 1, "claim", "j") is None
        assert any("no handler for kind 'global'" in e for e in errors)
    finally:
        steps_mod.KIND_HANDLERS.clear()
        steps_mod.KIND_HANDLERS.update(original)
    with SF() as s:
        assert s.get(RebuildLlmCall, rid).status == "queued"
    assert env.fake.chat == []


def test_step_handlers_are_registered_by_dotted_path():
    assert pc.STEP_HANDLERS["rebuild_keyframes"] == "app.rebuild_keyframes.step"
    assert pc.STEP_HANDLERS["rebuild_llm_calls"] == "app.rebuild_llm_steps.handle"
    # Integration: all three tables driven; the LLM dispatcher routes every kind.
    assert pc.STEP_HANDLERS["rebuild_chunk_attempts"] == "app.pipeline_rebuild.chunk_attempt_step"
    assert rebuild_llm_steps.KIND_HANDLERS == {
        "global": "app.rebuild_storyboard.llm_step",
        "shot": "app.rebuild_storyboard.llm_step",
        "character_check": "app.pipeline_rebuild.character_check_step",
    }
    from app import pipeline_rebuild

    assert pc.step_job_timeout("rebuild_llm_calls", "send") >= \
        pipeline_rebuild.CHECK_HTTP_TIMEOUT_SEC + 60
    for path in (pc.STEP_HANDLERS["rebuild_keyframes"], pc.STEP_HANDLERS["rebuild_llm_calls"],
                 pc.STEP_HANDLERS["rebuild_chunk_attempts"],
                 *rebuild_llm_steps.KIND_HANDLERS.values()):
        mod, fn = path.rsplit(".", 1)
        assert callable(getattr(importlib.import_module(mod), fn))
    assert pc.step_job_timeout("rebuild_llm_calls", "send") >= \
        max(sbmod.LLM_TIMEOUT_SEC.values()) + 60
    assert pc.step_job_timeout("rebuild_keyframes", "send") == pc.STEP_JOB_TIMEOUT_SEC["send"]


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------


def test_keyframes_need_an_applied_storyboard(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes", data={"shots": "all_missing"})
    assert r.status_code == 409 and "storyboard" in r.text


def test_generate_missing_keyframes_and_sheets(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                    data={"shots": "all_missing", "sheets": "all_missing"})
    assert r.status_code == 202, r.text
    ids = r.json()["keyframe_ids"]
    assert len(ids) == 6 + 2 and r.json()["reserved_credits"] == 8 * 10.0
    with SF() as s:
        rows = [s.get(RebuildKeyframe, i) for i in ids]
        assert len({k.batch_id for k in rows}) == 1
        assert all(r.aspect_ratio == "9:16" and r.resolution == "2K" for r in rows)
        shot_rows = [r for r in rows if r.shot_id]
        sheet_rows = [r for r in rows if not r.shot_id]
        shots = {x.id: x for x in rebuild_analysis.project_shots(s, pid)}
        for r in shot_rows:  # one keyframe per shot, at its midpoint (Q3)
            sh = shots[r.shot_id]
            assert r.input_paths[0]["t"] == pytest.approx((sh.start_sec + sh.end_sec) / 2, abs=1e-3)
            assert "Remove ALL burned-in captions" in r.prompt and "3D animated cartoon" in r.prompt
        assert sorted(r.sheet_key for r in sheet_rows) == ["street", "wedding"]
        assert all(1 <= len(r.input_paths) <= 5 for r in sheet_rows)
        res = s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend.keyframes_scope(rows[0].batch_id))).scalar_one()
        assert res.reserved_mcredits == 80_000
    env.drive()
    assert len(env.fake.create) == 8
    assert all(p["model"] == "gpt-image-2-5-flare-image-to-image" for p in env.fake.create)
    assert all(p["input"]["resolution"] == "2K" and p["input"]["aspect_ratio"] == "9:16"
               for p in env.fake.create)
    with SF() as s:
        rows = [s.get(RebuildKeyframe, i) for i in ids]
        assert all(r.status == "succeeded" and r.selected for r in rows)
        draws = s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key.in_([f"rebuild:{i}" for i in ids]))).scalars().all()
        assert len(draws) == 8 and all(d.mcredits == 9000 for d in draws)  # creditsConsumed
    # Nothing missing any more: a second "all missing" has nothing to do.
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                    data={"shots": "all_missing", "sheets": "all_missing"})
    assert r.status_code == 409 and "nothing to generate" in r.text


def test_regenerate_one_and_selection_rules(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][0]
    first = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                        data={"shots": shot.id}).json()["keyframe_ids"][0]
    env.drive()
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                    data={"shots": shot.id, "prompt": "warmer light please"})
    second = r.json()["keyframe_ids"][0]
    # Regenerating while one is live is refused.
    r2 = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes", data={"shots": shot.id})
    assert r2.status_code == 409 and "already being generated" in r2.text
    env.drive()

    def selected():
        with SF() as s:
            return [k.id for k in s.execute(select(RebuildKeyframe).where(
                RebuildKeyframe.shot_id == shot.id, RebuildKeyframe.selected.is_(True))).scalars()]

    assert selected() == [second]  # the newest success is auto-selected
    with SF() as s:
        assert s.get(RebuildKeyframe, second).prompt == "warmer light please"
    r = client.patch(f"/api/v2/rebuild/keyframes/{first}", data={"selected": "true"})
    assert r.status_code == 200, r.text
    assert selected() == [first]
    # A newer attempt completing later does NOT override the operator's pick.
    third = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                        data={"shots": shot.id}).json()["keyframe_ids"][0]
    env.drive()
    assert selected() == [first]
    client.patch(f"/api/v2/rebuild/keyframes/{third}", data={"selected": "true"})
    assert selected() == [third]
    # A failed attempt cannot be selected.
    env.fake.create_task_once = lambda payload, timeout: SendOutcome("refused", error="400")
    bad = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                      data={"shots": shot.id}).json()["keyframe_ids"][0]
    env.drive()
    r = client.patch(f"/api/v2/rebuild/keyframes/{bad}", data={"selected": "true"})
    assert r.status_code == 409
    assert selected() == [third]


def test_folded_shot_has_no_keyframe(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    folded = [s for s in _shots(SF, pid) if s.folded_into is not None][0]
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes", data={"shots": folded.id})
    assert r.status_code == 400 and "unfold" in r.text


def test_failed_download_retries_the_download_only(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][3]
    env.fake.download_fail = 2
    kid = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                      data={"shots": shot.id}).json()["keyframe_ids"][0]
    env.drive()
    with SF() as s:
        kf = s.get(RebuildKeyframe, kid)
        assert kf.status == "succeeded" and kf.fetch_attempts == 2
    assert len(env.fake.create) == 1


def test_provider_fail_is_final_and_settled(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][4]
    env.fake.task_state = "fail"
    kid = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                      data={"shots": shot.id}).json()["keyframe_ids"][0]
    env.drive()
    with SF() as s:
        kf = s.get(RebuildKeyframe, kid)
        assert kf.status == "failed" and not kf.selected
        (draw,) = s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key == f"rebuild:{kid}")).scalars()
        assert draw.mcredits == 9000
    assert len(env.fake.create) == 1


@pytest.mark.spend_cap
def test_keyframe_batch_over_budget_is_429_and_admits_nothing(client, env, SF, monkeypatch):
    pid = _ready_with_storyboard(client, env, SF)
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0001)
    with SF() as s:
        before = s.execute(select(RebuildKeyframe.id)).all()
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes", data={"shots": "all_missing"})
    assert r.status_code == 429
    with SF() as s:
        assert s.execute(select(RebuildKeyframe.id)).all() == before
    assert env.steps == []


@pytest.mark.spend_cap
def test_storyboard_over_budget_is_429(client, env, SF, monkeypatch):
    pid = make_project(SF)
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0001)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 429
    with SF() as s:
        assert sbmod.live_storyboard(s, pid) is None


# ---------------------------------------------------------------------------
# Admission guards: ready + project lock + flag
# ---------------------------------------------------------------------------


def test_every_admission_requires_ready(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][0]
    kid = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                      data={"shots": shot.id}).json()["keyframe_ids"][0]
    env.drive()
    with SF() as s:
        s.get(VideoProject, pid).status = ProjectStatus.analyzing
        s.commit()
    calls = [
        ("post", f"/api/v2/projects/{pid}/rebuild/storyboard", {"mode": "resume"}),
        ("post", f"/api/v2/projects/{pid}/rebuild/keyframes", {"shots": "all_missing"}),
        ("patch", f"/api/v2/rebuild/keyframes/{kid}", {"selected": "true"}),
        ("patch", f"/api/v2/rebuild/shots/{shot.id}", {"action": "x"}),
        ("post", f"/api/v2/projects/{pid}/rebuild/keep-brand-cards", {"keep": "true"}),
    ]
    for method, url, data in calls:
        r = getattr(client, method)(url, data=data)
        assert r.status_code == 409 and "not ready" in r.text, (url, r.text)
    with SF() as s:
        s.get(VideoProject, pid).status = ProjectStatus.ready
        s.commit()


def test_flag_off_refuses_every_mutation_but_reads_work(client, env, SF, monkeypatch):
    pid = _ready_with_storyboard(client, env, SF)
    shot = _shots(SF, pid)[0]
    monkeypatch.setattr(settings, "REBUILD_ENABLED", False)
    for method, url, data in [
        ("post", f"/api/v2/projects/{pid}/rebuild/storyboard", {"mode": "resume"}),
        ("post", f"/api/v2/projects/{pid}/rebuild/storyboard/cancel", {"x": "1"}),
        ("post", f"/api/v2/projects/{pid}/rebuild/keyframes", {"shots": "all_missing"}),
        ("patch", f"/api/v2/rebuild/keyframes/{uuid.uuid4()}", {"selected": "true"}),
        ("patch", f"/api/v2/rebuild/shots/{shot.id}", {"action": "x"}),
        ("post", f"/api/v2/projects/{pid}/rebuild/keep-brand-cards", {"keep": "true"}),
    ]:
        r = getattr(client, method)(url, data=data)
        assert r.status_code == 409 and "REBUILD_ENABLED" in r.text, url
    assert client.get(f"/api/v2/projects/{pid}/rebuild/storyboard").status_code == 200
    assert client.get(f"/api/v2/projects/{pid}/rebuild/shots").status_code == 200
    assert env.steps == [] and env.sb_jobs == []


def test_non_rebuild_project_is_404(client, env, SF):
    with SF() as s:
        p = VideoProject(source_type="upload", source_ref="s.mp4", status=ProjectStatus.ready,
                         project_type="face_swap")
        s.add(p)
        s.commit()
        pid = p.id
    assert client.post(f"/api/v2/projects/{pid}/rebuild/storyboard",
                       data={"mode": "resume"}).status_code == 404
    assert client.get(f"/v2/projects/{pid}/rebuild-fragment").status_code == 404


def test_reanalysis_is_refused_while_a_storyboard_is_live(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 202
    r = client.post(f"/api/v2/projects/{pid}/analyze")
    assert r.status_code == 409 and "in flight" in r.text
    env.drive()


# ---------------------------------------------------------------------------
# Shots: edit, unfold, brand cards -> rechunk
# ---------------------------------------------------------------------------


def test_shot_edit_and_frame_time(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    shot = _shots(SF, pid)[1]
    r = client.patch(f"/api/v2/rebuild/shots/{shot.id}",
                     data={"frame_sec": "3.5", "characters": "C1, C2", "is_brand_card": "false"})
    assert r.status_code == 200, r.text
    s1 = _shots(SF, pid)[1]
    assert s1.frame_sec == 3.5 and s1.characters == ["C1", "C2"] and s1.edited_at is not None
    r = client.patch(f"/api/v2/rebuild/shots/{shot.id}", data={"frame_sec": "9.0"})
    assert r.status_code == 400
    # A new keyframe takes its frame at the edited time (its own file).
    kid = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                      data={"shots": shot.id}).json()["keyframe_ids"][0]
    with SF() as s:
        inp = s.get(RebuildKeyframe, kid).input_paths[0]
        assert inp["t"] == 3.5 and inp["path"].endswith(f"{shot.id}_3500.jpg")
    env.drive()


def test_unfold_and_brand_cards_rechunk_and_are_refused_while_a_run_is_active(client, env, SF):
    pid = make_project(SF)
    shots = _shots(SF, pid)
    folded = [s for s in shots if s.folded_into is not None][0]
    with SF() as s:
        run = Run(project_id=pid, model="kling-4-0", resolution="720p",
                  status=RunStatus.processing, generation_mode="images_only")
        s.add(run)
        s.commit()
        rid = run.id
    r = client.patch(f"/api/v2/rebuild/shots/{folded.id}", data={"force_anchor": "true"})
    assert r.status_code == 409 and "run" in r.text
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keep-brand-cards", data={"keep": "true"})
    assert r.status_code == 409
    with SF() as s:
        s.get(Run, rid).status = RunStatus.done
        s.commit()
    r = client.patch(f"/api/v2/rebuild/shots/{folded.id}", data={"force_anchor": "true"})
    assert r.status_code == 200 and r.json()["rechunked"]
    assert all(s.folded_into is None for s in _shots(SF, pid))
    # Mark the last shot a brand card, then keep it as original footage.
    last = _shots(SF, pid)[-1]
    client.patch(f"/api/v2/rebuild/shots/{last.id}", data={"is_brand_card": "true"})
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keep-brand-cards", data={"keep": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["segments"][-1]["action"] == "keep"
    with SF() as s:
        segs = s.execute(select(SegmentDef).where(SegmentDef.project_id == pid)
                         .order_by(SegmentDef.index)).scalars().all()
        assert segs[-1].action == "keep" and segs[-1].start_sec == 18
    # An unfold keeps the brand-card keep (re-chunked the same way).
    shot0 = _shots(SF, pid)[0]
    client.patch(f"/api/v2/rebuild/shots/{shot0.id}", data={"force_anchor": "true"})
    with SF() as s:
        segs = s.execute(select(SegmentDef).where(SegmentDef.project_id == pid)
                         .order_by(SegmentDef.index)).scalars().all()
        assert segs[-1].action == "keep"


# ---------------------------------------------------------------------------
# Rendered HTML of the project page (the localisation lesson)
# ---------------------------------------------------------------------------


def _frag(client, pid) -> str:
    r = client.get(f"/v2/projects/{pid}/rebuild-fragment")
    assert r.status_code == 200, r.text
    return r.text


def test_html_no_storyboard(client, env, SF):
    pid = make_project(SF)
    html = _frag(client, pid)
    assert 'id="rb-sb-none"' in html and 'id="rb-sb-build"' in html
    assert "Build storyboard" in html and "up to 280 cr" in html  # (1 + 6) x 2 x 20
    assert "build the storyboard" in html  # unmet precondition
    assert 'id="rb-launch-btn" disabled' in html
    assert "hx-trigger=\"every 4s" not in html  # nothing live, no polling
    assert "6 after folding" in html and "<strong>2</strong> chunks" in html
    # The full project page loads the card and hides the v2v form.
    page = client.get(f"/v2/projects/{pid}")
    assert page.status_code == 200
    assert 'id="rebuild-content"' in page.text and 'rebuild-fragment' in page.text
    assert 'id="new-run-details" hidden' in page.text


def test_html_storyboard_running_polls(client, env, SF):
    pid = make_project(SF)
    client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    html = _frag(client, pid)
    assert 'id="rb-sb-running"' in html and 'id="rb-sb-cancel"' in html
    assert 'hx-trigger="every 4s' in html and "closest('#rebuild-content form')" in html
    assert "the storyboard is still running" in html
    env.drive()


def test_html_done_with_speech_warning_and_keep_captions(client, env, SF):
    pid = make_project(SF)
    talky = dict(GLOBAL, audio={"kind": "mixed", "lines": []})
    env.fake.responder = lambda m, msgs: accepted(
        talky if "shot-by-shot remake" in _text(msgs) else
        shot_answer(is_brand_card="from 18.00 s" in _text(msgs)))
    build_storyboard(client, env, pid)
    client.post(f"/api/v2/projects/{pid}/rebuild/keep-brand-cards", data={"keep": "true"})
    html = _frag(client, pid)
    assert 'id="rb-speech-warning"' in html and "does not lip-sync" in html
    assert "3D animated cartoon" in html and "SALE 50%" in html
    assert "keeps source captions" in html
    assert 'id="rb-end-card"' in html  # a trailing brand-card keep
    assert "Build again (resume)" in html and 'id="rb-sb-scratch"' in html
    assert 'id="rb-generate-missing"' in html
    assert "Generate brand-card shots again" in html


def test_html_keyframes_partial_then_all_selected(client, env, SF):
    pid = _ready_with_storyboard(client, env, SF)
    anchors = [s for s in _shots(SF, pid) if s.folded_into is None]
    client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                data={"shots": ",".join(a.id for a in anchors[:3])})
    html = _frag(client, pid)
    # The launch precondition list is rebuild_launch.preconditions' (one source).
    assert "keyframe(s) / sheet(s) still being generated" in html
    assert 'hx-trigger="every 4s' in html
    env.drive()
    html = _frag(client, pid)
    assert "Generate missing keyframes (3 + 2 sheets × 10 cr = 50 cr)" in html
    assert "no selected keyframe for shot(s)" in html
    assert 'id="rb-launch-btn" disabled' in html
    client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                data={"shots": "all_missing", "sheets": "all_missing"})
    env.drive()
    html = _frag(client, pid)
    assert 'id="rb-all-selected"' in html and 'id="rb-unmet"' not in html
    assert 'id="rb-launch-btn" >' in html or 'id="rb-launch-btn">' in html
    # The launch form posts the create-run fields.
    for field in ('name="model"', 'name="resolution"', 'name="language"', 'name="prompt"',
                  'name="audio_mode"', "chunk_sheets", "stitch_plan", "kling-4-0",
                  "seedance-2-5"):
        assert field in html, field
    assert '<option value="kling-4-0" selected>' in html


def test_html_flag_off_is_read_only(client, env, SF, monkeypatch):
    pid = _ready_with_storyboard(client, env, SF)
    monkeypatch.setattr(settings, "REBUILD_ENABLED", False)
    html = _frag(client, pid)
    assert 'id="rb-flag-off"' in html
    for button in ('id="rb-sb-build"', 'id="rb-generate-missing"', "rbRegenerate(event",
                   "rbUnfold(", 'onclick="rbSelect('):
        assert button not in html, button
    assert 'id="rb-launch-btn" disabled' in html


# ---------------------------------------------------------------------------
# The real proxy encoder (ffmpeg in the image)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_llm_proxy_is_small_mono_and_plain(tmp_path):
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=s=720x1280:r=30:d=3",
                    "-f", "lavfi", "-i", "sine=f=440:d=3", "-shortest", "-c:v", "libx264",
                    "-c:a", "aac", "-ac", "2", src], check=True, capture_output=True)
    dst = str(tmp_path / "p.mp4")
    out = jobs.build_llm_proxy(src, dst, vertical=True, start=0.5, end=2.0)
    from app.media import probe

    info = probe(out)
    assert (info.width, info.height) == (480, 854)
    assert abs(info.duration_sec - 1.5) < 0.2 and info.fps <= 15.01
    assert os.path.getsize(out) <= jobs.LLM_PROXY_MAX_BYTES
    frame = str(tmp_path / "f.jpg")
    jobs.extract_frame(src, 1.0, frame)
    assert os.path.getsize(frame) > 0


# ---------------------------------------------------------------------------
# Terra 2a round 1
# ---------------------------------------------------------------------------


def _reservation(SF, sb_id):
    with SF() as s:
        return s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend.storyboard_scope(sb_id))).scalar_one()


def test_finished_and_interrupted_storyboards_release_their_unused_reservation(client, env, SF):
    pid = make_project(SF)
    sb_id = build_storyboard(client, env, pid)
    res = _reservation(SF, sb_id)
    # 7 calls drew 7 x llm; the fallback half of the reservation is released.
    assert res.consumed_mcredits == 7 * spend.llm_credits() * 1000
    assert res.reserved_mcredits == res.consumed_mcredits == res.charged_mcredits
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "from_scratch"})
    sb2 = r.json()["storyboard_id"]
    client.post(f"/api/v2/projects/{pid}/rebuild/storyboard/cancel", data={"x": "1"})
    res = _reservation(SF, sb2)
    assert res.reserved_mcredits == res.consumed_mcredits == 0
    env.drive()
    assert env.fake.chat[7:] == []


def test_resume_reuses_a_global_answer_the_interrupted_build_never_copied(client, env, SF):
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    sbmod.run_storyboard_job(sb_id, 1, None, sf=SF)
    env.sb_jobs.clear()
    (_, gid, gen, _) = env.steps.pop(0)
    rebuild_llm_steps.handle(gid, gen, "claim", "c")
    rebuild_llm_steps.handle(gid, gen, "send", "s")  # answered...
    env.steps.clear()
    client.post(f"/api/v2/projects/{pid}/rebuild/storyboard/cancel", data={"x": "1"})
    assert _sb(SF, sb_id).global_call_id is None  # ...but never copied
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.json()["reused_global_call_id"] == gid
    env.drive()
    assert sum("shot-by-shot remake" in t for _, t in env.fake.chat) == 1


def test_resume_does_not_reach_past_a_failed_from_scratch_build(client, env, SF):
    pid = make_project(SF)
    build_storyboard(client, env, pid)
    env.fake.responder = lambda m, msgs: SendOutcome("refused", error="400 bad")
    failed = build_storyboard(client, env, pid, mode="from_scratch")
    assert _sb(SF, failed).status == "failed"
    env.fake.responder = FakeKie.default_responder
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 202 and r.json()["reused_global_call_id"] is None
    env.drive()


def test_keyframe_poll_errors_still_time_out(client, env, SF, monkeypatch):
    pid = _ready_with_storyboard(client, env, SF)
    shot = [s for s in _shots(SF, pid) if s.folded_into is None][5]
    kid = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                      data={"shots": shot.id}).json()["keyframe_ids"][0]
    env.steps.clear()
    kfmod.step(kid, 1, "claim", "c", sf=SF)
    env.steps.clear()
    kfmod.step(kid, 1, "send", "s", sf=SF)
    env.steps.clear()

    def boom(task_id):
        raise RuntimeError("recordInfo 502")

    monkeypatch.setattr(env.fake, "get_task", boom)
    assert kfmod.step(kid, 1, "poll", "p", sf=SF) == "submitted"
    assert env.steps and env.steps[-1][3] == "poll"
    env.steps.clear()
    with SF() as s:
        s.get(RebuildKeyframe, kid).send_deadline = datetime.utcnow() - timedelta(hours=2)
        s.commit()
    assert kfmod.step(kid, 1, "poll", "p2", sf=SF) == "failed"
    with SF() as s:
        assert "no result after" in s.get(RebuildKeyframe, kid).error


def test_a_call_sending_at_the_cancel_releases_its_estimate_when_refused(client, env, SF):
    """Terra 2a round 2: the release at the interrupt keeps a sending call's
    estimate; its later refusal (settled to 0) must free it again."""
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    sbmod.run_storyboard_job(sb_id, 1, None, sf=SF)
    env.sb_jobs.clear()
    (_, gid, gen, _) = env.steps.pop(0)
    rebuild_llm_steps.handle(gid, gen, "claim", "c")

    def refuse_after_cancel(model, messages):
        # The operator cancels while this call is in flight.
        r2 = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard/cancel", data={"x": "1"})
        assert r2.status_code == 200, r2.text
        res = _reservation(SF, sb_id)
        assert res.reserved_mcredits == spend.llm_credits() * 1000  # the sending call
        return SendOutcome("refused", error="400 bad request")

    env.fake.responder = refuse_after_cancel
    assert rebuild_llm_steps.handle(gid, gen, "send", "s") == "failed"
    res = _reservation(SF, sb_id)
    assert res.consumed_mcredits == 0 and res.reserved_mcredits == 0 and res.charged_mcredits == 0


def test_reaper_sweep_releases_an_ended_storyboard_left_over_reserved(client, env, SF,
                                                                      monkeypatch):
    """Terra 2a round 3: the post-outcome release is its own transaction; if
    it is lost, the storyboard reaper's sweep releases on the next tick."""
    pid = make_project(SF)
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    sb_id = r.json()["storyboard_id"]
    sbmod.run_storyboard_job(sb_id, 1, None, sf=SF)
    env.sb_jobs.clear()
    (_, gid, gen, _) = env.steps.pop(0)
    rebuild_llm_steps.handle(gid, gen, "claim", "c")

    def refuse_after_cancel(model, messages):
        client.post(f"/api/v2/projects/{pid}/rebuild/storyboard/cancel", data={"x": "1"})
        return SendOutcome("refused", error="400 bad request")

    env.fake.responder = refuse_after_cancel
    monkeypatch.setattr(sbmod, "_release_if_ended", lambda row_id, sf: None)  # "worker died"
    rebuild_llm_steps.handle(gid, gen, "send", "s")
    assert _reservation(SF, sb_id).reserved_mcredits == spend.llm_credits() * 1000
    out = sbmod.reap(session_factory=SF, is_live=lambda p: True, enqueue_fn=lambda *a: None)
    assert out["released"] >= 1
    res = _reservation(SF, sb_id)
    assert res.reserved_mcredits == res.consumed_mcredits == res.charged_mcredits == 0
    assert sbmod.reap(session_factory=SF, is_live=lambda p: True,
                      enqueue_fn=lambda *a: None)["released"] == 0


def test_reaper_sweep_releases_the_scope_of_a_deleted_storyboard(SF):
    """Terra 2a round 4: an over-reserved scope whose storyboard was deleted
    (with its project) is released too."""
    ghost = str(uuid.uuid4())
    with SF() as s:
        spend.reserve(s, spend.storyboard_scope(ghost), 40.0, kind="rebuild storyboard")
        s.commit()
    assert _reservation(SF, ghost).reserved_mcredits == 40_000
    sbmod.reap(session_factory=SF, is_live=lambda p: True, enqueue_fn=lambda *a: None)
    res = _reservation(SF, ghost)
    assert res.reserved_mcredits == res.charged_mcredits == 0
