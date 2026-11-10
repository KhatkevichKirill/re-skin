"""Rebuild wave 1 (docs/rebuild-from-reference-design.md): everything below
the pure helpers, on SQLite.

* kie.ai: images-only payload builders (pilot-verified shapes), the
  single-attempt ``create_task_once`` / ``chat_completion_once`` outcome
  matrices, and the structural guarantee that no rebuild module can reach the
  retrying client;
* v2v byte-identity: every v2v model still sends exactly main's payloads;
* spend: images-only rates, the run estimator (pending chunks, live attempts,
  the character-check allowance), reserve dispatch;
* the paid-call state machine and its reaper (fencing, draws, sender proofs,
  host attestation);
* the REBUILD_ENABLED gates and the live-row guards through the real API;
* analysis of a rebuild project (shots, folding, chunk SegmentDefs, a new
  generation on re-analysis, nothing published after recovery reset it);
* the migration against the ORM, on SQLite, and the drain check.

The Postgres-only parts (partial indexes under concurrency, the scope lock,
the migration rehearsal) are in tests/test_rebuild_pg.py.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402
import pytest  # noqa: E402
import respx  # noqa: E402
from sqlalchemy import create_engine, event, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import ai_models, kie_client, rebuild_paid_call as pc, spend  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.models import (  # noqa: E402
    KieSpendDraw,
    KieSpendReservation,
    Run,
    RunSegment,
    SegmentDef,
    VideoProject,
)
from app.rebuild_models import (  # noqa: E402
    ProjectStoryboard,
    RebuildChunkAttempt,
    RebuildKeyframe,
    RebuildLlmCall,
    RebuildShot,
)
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus  # noqa: E402

BACKEND = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.normpath(os.path.join(BACKEND, ".."))
NOW = datetime(2026, 11, 3, 12, 0, 0)


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


@pytest.fixture()
def db(SF):
    s = SF()
    yield s
    s.rollback()
    s.close()


def _project(db, *, project_type="rebuild", status=ProjectStatus.ready, **kw):
    p = VideoProject(source_type="upload", source_ref="s.mp4", status=status,
                     project_type=project_type, **kw)
    db.add(p)
    db.commit()
    return p


def _run(db, project, *, mode="images_only", model="kling-4-0", resolution="720p",
         status=RunStatus.queued, **kw):
    r = Run(project_id=project.id, model=model, resolution=resolution, status=status,
            generation_mode=mode, **kw)
    db.add(r)
    db.commit()
    return r


# ---------------------------------------------------------------------------
# kie.ai: payloads (pilot-verified shapes, design §6.4)
# ---------------------------------------------------------------------------


def test_kling_payload_is_the_pilot_shape():
    p = kie_client.build_kling_payload(prompt="P", image_urls=[f"u{i}" for i in range(9)],
                                       duration=16, resolution="720p", aspect_ratio="9:16")
    assert p == {"model": "kling-4.0/reference-to-video", "input": {
        "prompt": "P", "duration": 16, "resolution": "720p", "aspect_ratio": "9:16",
        "image_urls": [f"u{i}" for i in range(9)]}}
    with pytest.raises(ValueError):
        kie_client.build_kling_payload(prompt="P", image_urls=["u"] * 11, duration=10)
    with pytest.raises(ValueError):
        kie_client.build_kling_payload(prompt="P", image_urls=["u"], duration=31)
    with pytest.raises(ValueError):
        kie_client.build_kling_payload(prompt="P", image_urls=["u"], duration=10,
                                       resolution="480p")
    with pytest.raises(ValueError):
        kie_client.build_kling_payload(prompt="P", image_urls=["u"], duration=10,
                                       aspect_ratio="adaptive")
    with pytest.raises(ValueError):
        kie_client.build_kling_payload(prompt="P", image_urls=[], duration=10)


def test_seedance_images_payload_is_the_pilot_shape():
    p = kie_client.build_seedance_images_payload(
        prompt="P", reference_image_urls=["a", "b"], duration=15, resolution="720p")
    assert p == {"model": "bytedance/seedance-2-5", "input": {
        "prompt": "P", "reference_image_urls": ["a", "b"], "reference_video_urls": [],
        "resolution": "720p", "aspect_ratio": "9:16", "duration": 15, "generate_audio": False}}
    assert isinstance(p["input"]["duration"], int)
    with pytest.raises(ValueError):
        kie_client.build_seedance_images_payload(prompt="P", reference_image_urls=["a"] * 10,
                                                 duration=8)
    with pytest.raises(ValueError):  # our ceiling is 15 (a chunk is <= 14.72 s)
        kie_client.build_seedance_images_payload(prompt="P", reference_image_urls=["a"],
                                                 duration=16)
    with pytest.raises(ValueError):  # the v2v-only 2.0 family has no images-only mode
        kie_client.build_seedance_images_payload(prompt="P", reference_image_urls=["a"],
                                                 duration=8, model="seedance")
    assert kie_client.build_images_only_payload(
        model="kling-4-0", prompt="P", image_urls=["a"], duration=5, resolution="1080p",
        aspect_ratio="16:9")["input"]["resolution"] == "1080p"
    with pytest.raises(ValueError):
        kie_client.build_images_only_payload(model="gemini-omni", prompt="P", image_urls=["a"],
                                             duration=5, resolution="720p", aspect_ratio="9:16")


# ---------------------------------------------------------------------------
# kie.ai: single-attempt calls (design §7.2, §5.2)
# ---------------------------------------------------------------------------

CREATE = "https://api.kie.ai/api/v1/jobs/createTask"
PAYLOAD = {"model": "kling-4.0/reference-to-video", "input": {"prompt": "p"}}


@pytest.fixture()
def kie():
    return kie_client.KieClient(api_key="test-key", sleep_fn=lambda s: None)


@pytest.mark.parametrize("response,kind,task_id", [
    (httpx.Response(200, json={"code": 200, "data": {"taskId": "T1"}}), "accepted", "T1"),
    (httpx.Response(422, json={"code": 422, "msg": "bad"}), "refused", None),
    (httpx.Response(401, text="no"), "refused", None),
    (httpx.Response(200, json={"code": 402, "msg": "Insufficient credits", "data": None}),
     "refused", None),
    (httpx.Response(500, text="boom"), "unknown", None),
    (httpx.Response(503, json={"code": 503}), "unknown", None),
    (httpx.Response(200, text="<html>"), "unknown", None),
    (httpx.Response(200, json={"code": 200, "data": {}}), "unknown", None),
])
def test_create_task_once_outcome_matrix(kie, response, kind, task_id):
    with respx.mock() as m:
        route = m.post(CREATE).mock(return_value=response)
        out = kie.create_task_once(PAYLOAD, timeout=5)
    assert out.kind == kind and out.task_id == task_id
    assert route.call_count == 1  # never retried inside the client
    if kind != "accepted":
        assert out.error


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.ConnectError("down"),
                                 httpx.RemoteProtocolError("cut")])
def test_create_task_once_transport_errors_are_unknown_and_sent_once(kie, exc):
    with respx.mock() as m:
        route = m.post(CREATE).mock(side_effect=exc)
        out = kie.create_task_once(PAYLOAD, timeout=5)
    assert out.unknown and route.call_count == 1


def test_create_task_once_sends_the_payload_verbatim(kie):
    with respx.mock() as m:
        route = m.post(CREATE).mock(return_value=httpx.Response(
            200, json={"code": 200, "data": {"taskId": "T"}}))
        kie.create_task_once(PAYLOAD, timeout=5)
    assert json.loads(route.calls[0].request.content) == PAYLOAD
    with pytest.raises(ValueError):
        kie.create_task_once({"input": {}}, timeout=5)


CHAT = "https://api.kie.ai/gemini-3.1-pro/v1/chat/completions"


@pytest.mark.parametrize("response,kind,file_unavailable", [
    (httpx.Response(200, json={"choices": [{"message": {"content": "{\"a\":1}"}}]}),
     "accepted", False),
    (httpx.Response(200, json={"code": 400, "msg": "Failed to get the file information",
                               "data": None}), "refused", True),
    (httpx.Response(422, json={"code": 422, "msg": "The model is not supported"}),
     "refused", False),
    (httpx.Response(200, json={"code": 500, "msg": "The server is currently being maintained"}),
     "refused", False),
    (httpx.Response(503, text="x"), "unknown", False),
    (httpx.Response(200, text="not json"), "unknown", False),
    (httpx.Response(200, json={"choices": []}), "unknown", False),
])
def test_chat_completion_once_outcome_matrix(kie, response, kind, file_unavailable):
    with respx.mock() as m:
        route = m.post(CHAT).mock(return_value=response)
        out = kie.chat_completion_once("gemini-3.1-pro", [{"role": "user", "content": "hi"}],
                                       timeout=5)
    assert (out.kind, out.file_unavailable) == (kind, file_unavailable)
    assert route.call_count == 1
    if kind == "accepted":
        assert out.content == "{\"a\":1}" and out.body["choices"]


def test_chat_completion_once_timeout_is_unknown(kie):
    with respx.mock() as m:
        route = m.post(CHAT).mock(side_effect=httpx.ReadTimeout("slow"))
        out = kie.chat_completion_once("gemini-3.1-pro", [], timeout=1)
    assert out.unknown and route.call_count == 1


_FORBIDDEN = {
    "_create_task_with_retry", "_chat_with_retry", "_call_llm_route", "create_task",
    "create_image_task", "create_omni_task", "chat_completion", "create_response",
}


def _rebuild_modules():
    app_dir = os.path.join(BACKEND, "app")
    paths = set(glob.glob(os.path.join(app_dir, "rebuild*.py")))
    paths |= set(glob.glob(os.path.join(app_dir, "*_rebuild*.py")))
    paths |= set(glob.glob(os.path.join(app_dir, "**", "*rebuild*.py"), recursive=True))
    return sorted(paths)


def test_no_rebuild_module_can_reach_the_retrying_client():
    """Design §7.2 / §5.2: every rebuild provider call is ONE request through
    create_task_once / chat_completion_once. Any reference (call, attribute or
    name) to a retrying entry point in a rebuild module fails this test — new
    wave-2 modules named rebuild_* / *_rebuild* are covered automatically."""
    mods = _rebuild_modules()
    names = {os.path.basename(p) for p in mods}
    assert {"rebuild.py", "rebuild_models.py", "rebuild_paid_call.py",
            "rebuild_analysis.py",
            # waves 2a / 2b (the integration): every module of the feature
            "rebuild_jobs.py", "rebuild_storyboard.py", "rebuild_keyframes.py",
            "rebuild_llm_steps.py", "api_rebuild.py", "rebuild_launch.py",
            "pipeline_rebuild.py", "rebuild_run_view.py", "api_rebuild_run.py"} <= names
    # ... and nothing named like the feature escapes the glob.
    every = {os.path.basename(p) for p in glob.glob(os.path.join(BACKEND, "app", "**", "*.py"),
                                                     recursive=True) if "rebuild" in p}
    assert every <= names, every - names
    offenders = []
    for path in mods:
        tree = ast.parse(open(path).read(), filename=path)
        for node in ast.walk(tree):
            ident = None
            if isinstance(node, ast.Attribute):
                ident = node.attr
            elif isinstance(node, ast.Name):
                ident = node.id
            elif isinstance(node, (ast.ImportFrom,)):
                for alias in node.names:
                    if alias.name in _FORBIDDEN:
                        offenders.append((path, alias.name))
            if ident in _FORBIDDEN:
                offenders.append((os.path.basename(path), ident, node.lineno))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value in _FORBIDDEN:
                offenders.append((os.path.basename(path), node.value, node.lineno))
    assert offenders == []


def test_single_attempt_methods_do_not_route_through_tenacity():
    import inspect
    import textwrap

    for meth in (kie_client.KieClient.create_task_once, kie_client.KieClient.chat_completion_once):
        tree = ast.parse(textwrap.dedent(inspect.getsource(meth)))
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not used & _FORBIDDEN
        assert not hasattr(meth, "retry")  # not tenacity-wrapped


# ---------------------------------------------------------------------------
# v2v byte-identity (design §1, §14)
# ---------------------------------------------------------------------------


def test_v2v_payloads_are_byte_identical_to_main():
    """Every v2v model (and every generate_audio variant) still builds exactly
    the request main a0a182f built (fixture generated from main with the same
    script), through the real _swap_task_call + KieClient path."""
    from app import pipeline_v2

    golden = json.load(open(os.path.join(BACKEND, "tests", "fixtures",
                                         "rebuild_v2v_payloads_main.json")))
    seen = set()
    for model in ai_models.V2V_MODELS:
        for ga in (None, False, True):
            key = f"{model}|{ga}"
            if key not in golden:
                continue
            recipe = dict(
                run_model=model, effective_prompt="swap the character",
                ref_urls=["https://x/r1.png", "https://x/r2.png"], clip_url="https://x/clip.mp4",
                clip_start=12.3, clip_end=19.65, run_resolution="720p",
                project_aspect_ratio="9:16", project_width=1080, project_height=1920,
                generate_audio=ga,
                audio_ref_urls=(["https://x/v.mp3"] if model == "seedance-2-5" and ga else None),
            )
            method, kwargs = pipeline_v2._swap_task_call(**recipe)
            with respx.mock() as m:
                route = m.post(CREATE).mock(return_value=httpx.Response(
                    200, json={"code": 200, "data": {"taskId": "t1"}}))
                getattr(kie_client.KieClient(api_key="k", sleep_fn=lambda s: None),
                        method)(**kwargs)
            assert json.loads(route.calls[0].request.content) == golden[key], key
            seen.add(key)
    assert seen == set(golden)
    assert set(ai_models.V2V_MODELS) == {k.split("|")[0] for k in golden}


def test_v2v_path_refuses_an_images_only_model():
    from app import pipeline_v2

    with pytest.raises(ValueError, match="images-only"):
        pipeline_v2._swap_task_call(
            run_model="kling-4-0", effective_prompt="p", ref_urls=[], clip_url="u",
            clip_start=0, clip_end=5, run_resolution="720p", project_aspect_ratio="9:16",
            project_width=1080, project_height=1920)


def test_v2v_aggregates_ignore_kling():
    assert ai_models.UNIVERSAL_MAX_SEGMENT_SEC == 10.0
    assert ai_models.ABSOLUTE_MAX_SEGMENT_SEC == 30.0
    from app import pipeline_v2

    assert pipeline_v2.whole_video_floor_sec() == 4.0
    # The v2v credit table is untouched; Kling has only images-only rates.
    assert "kling-4-0" not in spend.CREDITS_PER_CLIP_SEC
    assert spend.video_credits("seedance-2-5", "720p", 10.0) == 800.0


# ---------------------------------------------------------------------------
# Spend: images-only rates + estimator (design §7.3, §12)
# ---------------------------------------------------------------------------


def test_images_only_rates_and_estimates(monkeypatch):
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    assert spend.images_only_video_credits("kling-4-0", "720p", 16) == 456.0  # pilot: 456
    assert spend.images_only_video_credits("kling-4-0", "1080p", 15) == 637.5
    assert spend.images_only_video_credits("seedance-2-5", "720p", 15) == 960.0
    assert spend.images_only_video_credits("nope", "720p", 1) == 160.0  # most expensive
    assert spend.storyboard_credits(54) == 55 * 2 * 20
    assert spend.keyframes_credits(55) == 550.0
    assert spend.storyboard_scope("s") == "storyboard:s"
    assert spend.keyframes_scope("b") == "keyframes:b"
    assert spend.rebuild_call_key("r") == "rebuild:r"


@pytest.mark.parametrize("slots,verdict,expected", [
    ({}, False, 2), ({0: "queued"}, False, 2), ({0: "claimed"}, False, 2),
    ({0: "sending"}, False, 1), ({0: "failed"}, False, 1), ({0: "unknown"}, False, 1),
    ({0: "failed", 1: "queued"}, False, 1), ({0: "failed", 1: "sending"}, False, 0),
    ({0: "failed", 1: "failed"}, False, 0), ({0: "submitted"}, False, 1),
    # Integration gate round 1: a primary that succeeded with an unusable
    # answer still gets its fallback, so the slot stays reserved until the
    # check has a verdict (a valid answer or "unavailable").
    ({0: "succeeded"}, False, 1), ({0: "succeeded", 1: "queued"}, False, 1),
    ({0: "succeeded"}, True, 0), ({0: "failed", 1: "failed"}, True, 0), ({}, True, 0),
])
def test_character_check_allowance(monkeypatch, slots, verdict, expected):
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    assert spend.check_allowance_credits(slots, verdict=verdict) == expected * 20.0


def _chunked_run(db, n=3, *, model="kling-4-0", seg_len=10.0):
    p = _project(db)
    defs = []
    for i in range(n):
        sd = SegmentDef(project_id=p.id, index=i, start_sec=i * seg_len,
                        end_sec=(i + 1) * seg_len, action="swap")
        db.add(sd)
        defs.append(sd)
    db.flush()
    snapshot = [{"id": sd.id, "index": sd.index, "action": "swap", "start_sec": sd.start_sec,
                 "end_sec": sd.end_sec} for sd in defs]
    run = _run(db, p, model=model, segment_snapshot=snapshot, rebuild_snapshot={
        "version": 1, "chunks": [{"segment_def_id": sd.id, "request_sec": 10} for sd in defs]})
    segs = []
    for sd in defs:
        rs = RunSegment(run_id=run.id, segment_def_id=sd.id, index=sd.index,
                        status=SegmentStatus.pending)
        db.add(rs)
        segs.append(rs)
    db.commit()
    return p, run, defs, segs


def _attempt(db, run, rs, n=1, status="queued", **kw):
    a = RebuildChunkAttempt(run_id=run.id, run_segment_id=rs.id, attempt_no=n, model=run.model,
                            resolution=run.resolution, request_sec=10, chunk_frames=300,
                            status=status, estimate_credits=285.0, **kw)
    db.add(a)
    db.commit()
    return a


def test_images_only_run_estimate(db, monkeypatch):
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    p, run, defs, segs = _chunked_run(db)
    per_chunk = 285.0 + 40.0
    assert spend.estimate_images_only_run_credits(db, run) == pytest.approx(3 * per_chunk)
    # The v2v entry point dispatches on generation_mode.
    assert spend.estimate_run_credits(db, run) == pytest.approx(3 * per_chunk)
    # Chunk 0 in flight (live attempt), chunk 1 completed, chunk 2 pending.
    segs[0].status = SegmentStatus.uploading
    segs[1].status = SegmentStatus.completed
    db.commit()
    live = _attempt(db, run, segs[0], status="queued")
    done = _attempt(db, run, segs[1], status="succeeded")
    # Chunk 2 + the QUEUED attempt (admitted, undrawn: its estimate + its
    # check) + the done attempt's full check allowance (2 slots). Terra
    # round 1: a release_unused re-baseline must not release a queued
    # attempt's money.
    assert spend.estimate_images_only_run_credits(db, run) == pytest.approx(
        per_chunk + (285.0 + 40.0) + 40.0)
    live.status = "sending"  # its video draw is in; only its check is undrawn
    db.commit()
    assert spend.estimate_images_only_run_credits(db, run) == pytest.approx(
        per_chunk + 40.0 + 40.0)
    db.add(RebuildLlmCall(project_id=p.id, attempt_id=done.id, kind="character_check",
                          slot=0, model="gemini-3.1-pro", status="sending",
                          estimate_credits=20.0))
    db.commit()
    assert spend.estimate_images_only_run_credits(db, run) == pytest.approx(
        per_chunk + 40.0 + 20.0)
    # A rerun of the completed chunk (reset to pending) is counted again even
    # though its older attempt stays selected (round 2, R2-5).
    segs[1].status = SegmentStatus.pending
    db.commit()
    assert spend.estimate_images_only_run_credits(db, run) == pytest.approx(
        2 * per_chunk + 40.0 + 20.0)
    # A chunk left out of a custom stitch plan is never generated.
    run.stitch_plan = [{"kind": "segment", "segment_def_id": defs[0].id},
                       {"kind": "segment", "segment_def_id": defs[1].id}]
    db.commit()
    assert spend.estimate_images_only_run_credits(db, run) == pytest.approx(
        per_chunk + 40.0 + 20.0)
    # Failed / unknown attempts get no check: they count nothing.
    live.status = "unknown"
    db.commit()
    assert spend.estimate_images_only_run_credits(db, run) == pytest.approx(per_chunk + 20.0)


def test_reserve_run_dispatches_images_only_runs_to_reserve_computed(db, monkeypatch):
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    _p, run, _d, _s = _chunked_run(db, n=2)
    calls = []
    real = spend.reserve_computed

    def spy(*a, **kw):
        calls.append(kw.get("kind"))
        return real(*a, **kw)

    monkeypatch.setattr(spend, "reserve_computed", spy)
    spend.reserve_run(db, run, kind="new run", now=NOW.replace(tzinfo=timezone.utc))
    db.commit()
    assert calls == ["new run"]
    res = db.execute(select(KieSpendReservation).where(
        KieSpendReservation.scope_key == spend.run_scope(run.id))).scalar_one()
    assert res.reserved_mcredits == 2 * 325 * 1000


@pytest.mark.spend_cap
def test_reserve_computed_refuses_over_the_cap_and_rebaselines(db, monkeypatch):
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 1000.0)
    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    scope = f"run:{uuid.uuid4()}"
    day1 = datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 11, 2, 1, 0, tzinfo=timezone.utc)
    spend.reserve_computed(db, scope, lambda s: 500.0, kind="t", now=day1)
    db.commit()
    spend.record_draw(scope, 100.0, kind="t", call_key=f"k{uuid.uuid4()}",
                      sessions=lambda: _ctx(db), now=day1)
    # Next day, release_unused: yesterday's unconsumed 400 is released.
    spend.reserve_computed(db, scope, lambda s: 50.0, kind="t", release_unused=True, now=day2)
    db.commit()
    rows = {r.day.isoformat(): (r.reserved_mcredits, r.consumed_mcredits) for r in db.execute(
        select(KieSpendReservation).where(KieSpendReservation.scope_key == scope)).scalars()}
    assert rows == {"2026-11-01": (100000, 100000), "2026-11-02": (50000, 0)}
    with pytest.raises(spend.BudgetExceeded):
        spend.reserve_computed(db, f"run:{uuid.uuid4()}", lambda s: 10 ** 7, kind="t", now=day2)
    db.rollback()


@contextmanager
def _ctx(db):
    yield db


# ---------------------------------------------------------------------------
# Paid-call state machine (design §7.2)
# ---------------------------------------------------------------------------

SENDER = pc.SenderIdentity(host="c0ffee123456", boot="boot-1", job="rb:x:y:1:send")


def _keyframe(db, project, **kw):
    shot = RebuildShot(project_id=project.id, index=uuid.uuid4().int % 10 ** 9, generation=1,
                       start_sec=0, end_sec=1, frame_sec=0.5)
    db.add(shot)
    db.flush()
    kf = RebuildKeyframe(project_id=project.id, batch_id=str(uuid.uuid4()), shot_id=shot.id,
                         estimate_credits=10.0, **kw)
    db.add(kf)
    db.commit()
    return kf


def _reload(db, model, rid):
    db.expire_all()
    return db.get(model, rid)


def _draws(db, key):
    return db.execute(select(KieSpendDraw).where(KieSpendDraw.call_key == key)).scalars().all()


def test_claim_send_accept_poll_fetch_happy_path(db):
    p = _project(db)
    kf = _keyframe(db, p)
    scope = spend.keyframes_scope(kf.batch_id)
    assert pc.claim(db, RebuildKeyframe, kf.id, gen=2) is None  # stale generation
    token = pc.claim(db, RebuildKeyframe, kf.id, gen=1)
    db.commit()
    assert token and pc.claim(db, RebuildKeyframe, kf.id, gen=1) is None  # duplicate
    assert not pc.begin_send(db, RebuildKeyframe, kf.id, gen=1, token="other", request={},
                             sender=SENDER, scope_key=scope)
    req = {"model": "gpt-image-2", "input": {"prompt": "x"}}
    assert pc.begin_send(db, RebuildKeyframe, kf.id, gen=1, token=token, request=req,
                         sender=SENDER, scope_key=scope, now=NOW)
    db.commit()
    row = _reload(db, RebuildKeyframe, kf.id)
    assert (row.status, row.sender_host, row.sender_boot, row.request) == (
        "sending", SENDER.host, "boot-1", req)
    assert row.send_deadline == NOW + timedelta(seconds=pc.SEND_WINDOW_SEC)
    (draw,) = _draws(db, f"rebuild:{kf.id}")
    assert draw.mcredits == 10000  # the draw is committed WITH the marker
    out = kie_client.SendOutcome("accepted", task_id="T9")
    assert pc.record_send_outcome(db, RebuildKeyframe, kf.id, gen=1, token="x", outcome=out,
                                  scope_key=scope) is None  # fenced
    assert pc.record_send_outcome(db, RebuildKeyframe, kf.id, gen=1, token=token, outcome=out,
                                  scope_key=scope) == "submitted"
    db.commit()
    assert pc.record_poll(db, RebuildKeyframe, kf.id, gen=1, state="generating",
                          scope_key=scope) == "submitted"
    assert pc.record_poll(db, RebuildKeyframe, kf.id, gen=1, state="success",
                          credits_consumed=9.0, scope_key=scope) == "fetching"
    db.commit()
    assert pc.fail_fetch(db, RebuildKeyframe, kf.id, gen=1, error="502", scope_key=scope) == "retry"
    db.commit()
    assert pc.complete_fetch(db, RebuildKeyframe, kf.id, gen=1, scope_key=scope,
                             values={"image_path": "/x.png"})
    db.commit()
    row = _reload(db, RebuildKeyframe, kf.id)
    assert (row.status, row.image_path, row.fetch_attempts, row.claim_token) == (
        "succeeded", "/x.png", 1, None)
    (draw,) = _draws(db, f"rebuild:{kf.id}")
    assert draw.mcredits == 9000 and draw.settled_at is not None


@pytest.mark.parametrize("outcome,status,settled", [
    (kie_client.SendOutcome("refused", error="400"), "failed", 0),
    (kie_client.SendOutcome("unknown", error="timeout"), "unknown", 10000),
    (None, "failed", 0),  # skipped: the send window elapsed, nothing left the process
])
def test_send_outcomes_and_their_accounting(db, outcome, status, settled):
    p = _project(db)
    kf = _keyframe(db, p)
    scope = spend.keyframes_scope(kf.batch_id)
    token = pc.claim(db, RebuildKeyframe, kf.id, gen=1)
    pc.begin_send(db, RebuildKeyframe, kf.id, gen=1, token=token, request={"r": 1},
                  sender=SENDER, scope_key=scope)
    db.commit()
    assert pc.record_send_outcome(db, RebuildKeyframe, kf.id, gen=1, token=token,
                                  outcome=outcome, scope_key=scope) == status
    db.commit()
    row = _reload(db, RebuildKeyframe, kf.id)
    assert row.status == status and row.claim_token is None
    (draw,) = _draws(db, f"rebuild:{kf.id}")
    assert draw.mcredits == settled
    if status == "unknown":
        assert pc.UNKNOWN_MESSAGE in row.error


def test_llm_call_accept_goes_straight_to_succeeded(db):
    p = _project(db)
    sb = ProjectStoryboard(project_id=p.id, shots_generation=1, status="running")
    db.add(sb)
    db.flush()
    call = RebuildLlmCall(project_id=p.id, storyboard_id=sb.id, kind="global",
                          model="gemini-3.1-pro", estimate_credits=20.0)
    db.add(call)
    db.commit()
    token = pc.claim(db, RebuildLlmCall, call.id, gen=1)
    pc.begin_send(db, RebuildLlmCall, call.id, gen=1, token=token, request={"m": 1},
                  sender=SENDER, scope_key=spend.storyboard_scope(sb.id))
    db.commit()
    out = kie_client.SendOutcome("accepted", content="{}", body={"choices": [1]})
    assert pc.record_send_outcome(db, RebuildLlmCall, call.id, gen=1, token=token, outcome=out,
                                  scope_key=spend.storyboard_scope(sb.id)) == "succeeded"
    db.commit()
    assert _reload(db, RebuildLlmCall, call.id).response == {"choices": [1]}


def _sent_llm_call(db):
    p = _project(db)
    sb = ProjectStoryboard(project_id=p.id, shots_generation=1, status="running")
    db.add(sb)
    db.flush()
    call = RebuildLlmCall(project_id=p.id, storyboard_id=sb.id, kind="global",
                          model="gemini-3.1-pro", estimate_credits=20.0)
    db.add(call)
    db.commit()
    scope = spend.storyboard_scope(sb.id)
    spend.reserve_computed(db, scope, lambda _db: 40.0, kind="rebuild storyboard")
    token = pc.claim(db, RebuildLlmCall, call.id, gen=1)
    pc.begin_send(db, RebuildLlmCall, call.id, gen=1, token=token, request={"m": 1},
                  sender=SENDER, scope_key=scope)
    db.commit()
    return sb, call, token, scope


def test_llm_accept_settles_the_draw_to_credits_consumed(db):
    """AICP-382: kie.ai reports the chat call's own cost; the draw (booked at
    the 20 cr estimate at ``sending``) is settled to it in the same commit, and
    the storyboard's end-of-build release then shrinks the reservation to it."""
    from app import rebuild_storyboard as sbmod

    sb, call, token, scope = _sent_llm_call(db)
    (draw,) = _draws(db, f"rebuild:{call.id}")
    assert draw.mcredits == 20000 and draw.settled_at is None
    body = {"choices": [{"message": {"content": "{}"}}], "credits_consumed": 0.25,
            "usage": {"prompt_tokens": 1864, "completion_tokens": 91, "total_tokens": 2597}}
    out = kie_client.SendOutcome("accepted", content="{}", body=body)
    assert pc.record_send_outcome(db, RebuildLlmCall, call.id, gen=1, token=token, outcome=out,
                                  scope_key=scope) == "succeeded"
    db.commit()
    row = _reload(db, RebuildLlmCall, call.id)
    assert row.cost_credits == 0.25 and row.response == body
    (draw,) = _draws(db, f"rebuild:{call.id}")
    assert draw.mcredits == 250 and draw.estimated_mcredits == 20000
    assert draw.settled_at is not None
    res = db.execute(select(KieSpendReservation).where(
        KieSpendReservation.scope_key == scope)).scalar_one()
    assert res.consumed_mcredits == 250 and res.charged_mcredits == 40000
    sbmod.release_unused(db, db.get(ProjectStoryboard, sb.id))
    db.commit()
    db.expire_all()
    res = db.execute(select(KieSpendReservation).where(
        KieSpendReservation.scope_key == scope)).scalar_one()
    assert (res.reserved_mcredits, res.consumed_mcredits, res.charged_mcredits) == (250, 250, 250)
    # A duplicate record (fenced out: the row is terminal) changes nothing.
    assert pc.record_send_outcome(db, RebuildLlmCall, call.id, gen=1, token=token, outcome=out,
                                  scope_key=scope) is None
    db.rollback()
    (draw,) = _draws(db, f"rebuild:{call.id}")
    assert draw.mcredits == 250


@pytest.mark.parametrize("cc", [None, "n/a", -1, float("nan"), float("inf"), True, [0.3]])
def test_llm_accept_without_a_usable_cost_keeps_the_estimate(db, cc):
    _sb, call, token, scope = _sent_llm_call(db)
    body = {"choices": [{"message": {"content": "{}"}}]}
    if cc is not None:
        body["credits_consumed"] = cc
    out = kie_client.SendOutcome("accepted", content="{}", body=body)
    assert pc.record_send_outcome(db, RebuildLlmCall, call.id, gen=1, token=token, outcome=out,
                                  scope_key=scope) == "succeeded"
    db.commit()
    assert _reload(db, RebuildLlmCall, call.id).cost_credits is None
    (draw,) = _draws(db, f"rebuild:{call.id}")
    assert draw.mcredits == 20000 and draw.settled_at is None


def test_chat_credits_consumed_parser():
    f = kie_client.chat_credits_consumed
    assert f({"credits_consumed": 0.45}) == 0.45
    assert f({"credits_consumed": 2}) == 2.0
    assert f({"credits_consumed": "1.81"}) == 1.81
    assert f({"credits_consumed": 0}) == 0.0
    for bad in (None, {}, [], "x", {"credits_consumed": None}, {"credits_consumed": False},
                {"credits_consumed": -0.1}, {"credits_consumed": "abc"},
                {"credits_consumed": float("nan")}, {"credits_consumed": {"v": 1}}):
        assert f(bad) is None


def test_poll_fail_settles_and_fetch_gives_up(db):
    p = _project(db)
    kf = _keyframe(db, p, status="submitted", task_id="T")
    scope = "keyframes:x"
    assert pc.record_poll(db, RebuildKeyframe, kf.id, gen=1, state="fail", credits_consumed=0,
                          error="nsfw", scope_key=scope) == "failed"
    db.commit()
    assert _reload(db, RebuildKeyframe, kf.id).status == "failed"
    # Terra round 2: a failed task without a reported cost keeps its estimate.
    kf_b = _keyframe(db, p, status="sending")
    spend.draw_in_session(db, scope, f"rebuild:{kf_b.id}", 10.0, kind="rebuild keyframe",
                          provider="kie", scope_lock=True)
    kf_b.status, kf_b.task_id = "submitted", "TB"
    db.commit()
    assert pc.record_poll(db, RebuildKeyframe, kf_b.id, gen=1, state="fail",
                          credits_consumed=None, scope_key=scope) == "failed"
    db.commit()
    assert _reload(db, RebuildKeyframe, kf_b.id).cost_credits is None
    (draw,) = _draws(db, f"rebuild:{kf_b.id}")
    assert draw.mcredits == 10000 and draw.settled_at is None
    kf2 = _keyframe(db, p, status="fetching", task_id="T2", cost_credits=10.0)
    for i in range(pc.FETCH_MAX_ATTEMPTS - 1):
        assert pc.fail_fetch(db, RebuildKeyframe, kf2.id, gen=1, error="dl",
                             scope_key=scope) == "retry"
        db.commit()
    assert pc.fail_fetch(db, RebuildKeyframe, kf2.id, gen=1, error="dl",
                         scope_key=scope) == "failed"
    db.commit()
    row = _reload(db, RebuildKeyframe, kf2.id)
    assert row.status == "failed" and "task T2" in row.error and row.task_id == "T2"
    # final=True (e.g. "result too short") fails at once, never re-creates.
    kf3 = _keyframe(db, p, status="fetching", task_id="T3")
    assert pc.fail_fetch(db, RebuildKeyframe, kf3.id, gen=1, error="result too short",
                         final=True, scope_key=scope) == "failed"


def test_cancel_unsent_only_touches_queued_and_claimed(db):
    p = _project(db)
    sb = ProjectStoryboard(project_id=p.id, shots_generation=1, status="running")
    db.add(sb)
    db.flush()
    rows = {}
    for st in ("queued", "claimed", "sending", "succeeded"):
        c = RebuildLlmCall(project_id=p.id, storyboard_id=sb.id, kind="shot",
                           shot_id=str(uuid.uuid4()), global_call_id="g", model="m", status=st,
                           claim_token="tok")
        db.add(c)
        rows[st] = c
    db.commit()
    assert pc.cancel_unsent(db, RebuildLlmCall, RebuildLlmCall.storyboard_id == sb.id) == 2
    db.commit()
    got = {st: _reload(db, RebuildLlmCall, c.id).status for st, c in rows.items()}
    assert got == {"queued": "failed", "claimed": "failed", "sending": "sending",
                   "succeeded": "succeeded"}
    assert _reload(db, RebuildLlmCall, rows["claimed"].id).claim_token is None


def test_send_window_and_sender_identity(monkeypatch):
    assert pc.send_window_open(100.0, clock=lambda: 124.0)
    assert not pc.send_window_open(100.0, clock=lambda: 126.0)
    monkeypatch.setattr(pc, "_BOOT_ID", None)
    with pytest.raises(pc.SenderUnavailable):
        pc.sender_identity("rb:t:r:1:send")

    class FakeRedis(dict):
        def set(self, k, v):
            self[k] = v

    conn = FakeRedis()
    boot = pc.register_boot(conn, hostname="h1")
    assert conn["rebuild:boot:h1"] == boot and pc.current_boot_id() == boot
    with pytest.raises(pc.SenderUnavailable):
        pc.sender_identity(None)
    assert pc.sender_identity("j").boot == boot
    boot2 = pc.register_boot(conn, hostname="h1")
    assert boot2 != boot


def test_step_job_ids():
    jid = pc.step_job_id("rebuild_keyframes", "abc", 3, "send")
    assert jid == "rb:rebuild_keyframes:abc:3:send"
    assert jid.startswith(pc.step_job_prefix("rebuild_keyframes", "abc", 3))
    assert pc.parse_step_job_id(jid) == ("rebuild_keyframes", "abc", 3, "send")
    assert pc.parse_step_job_id(pc.step_job_id("t", "r", 1, "poll", "s1"))[3] == "poll"
    assert pc.parse_step_job_id("process-run-x") is None
    with pytest.raises(ValueError):
        pc.step_job_id("t", "r", 1, "bogus")


@pytest.mark.parametrize("kw,proven", [
    # Restart: same hostname, new boot id -> automatic.
    (dict(sender_boot="b1", registered_boot="b2", job_status=None, job_exc=None), True),
    # Frozen container: same boot, job hash says started -> stays sending.
    (dict(sender_boot="b1", registered_boot="b1", job_status="started", job_exc=None), False),
    # Recreation: the old host has no live key change (absent) -> stays sending.
    (dict(sender_boot="b1", registered_boot=None, job_status=None, job_exc=None), False),
    # Horse reaped by its own worker (rq 1.15 monitor_work_horse) -> automatic.
    (dict(sender_boot="b1", registered_boot="b1", job_status="failed",
          job_exc="Work-horse terminated unexpectedly; waitpid returned 9 (signal 9); "), True),
    # Abandoned-job cleanup only -> proves nothing.
    (dict(sender_boot="b1", registered_boot="b1", job_status="failed",
          job_exc="Moved to FailedJobRegistry, due to AbandonedJobError, at 2026-11-03"), False),
    (dict(sender_boot="b1", registered_boot="b1", job_status="finished", job_exc=None), True),
    (dict(sender_boot="b1", registered_boot="b1", job_status="failed",
          job_exc="Traceback (most recent call last):\n  JobTimeoutException"), True),
    (dict(sender_boot="b1", registered_boot="b1", job_status="failed", job_exc=None), False),
])
def test_sender_proof_matrix(kw, proven):
    assert bool(pc.sender_proof(**kw)) is proven


@pytest.mark.parametrize("status,live,since,proof,handler,kind,step", [
    ("queued", True, None, None, True, "none", None),
    ("queued", True, NOW, None, True, "clear", None),
    ("queued", False, None, None, True, "observe", None),
    ("queued", False, NOW - timedelta(seconds=60), None, True, "none", None),
    ("queued", False, NOW - timedelta(seconds=200), None, True, "requeue", "claim"),
    ("claimed", False, NOW - timedelta(seconds=200), None, True, "requeue", "claim"),
    ("submitted", False, NOW - timedelta(seconds=200), None, True, "requeue", "poll"),
    ("fetching", False, NOW - timedelta(seconds=200), None, True, "requeue", "fetch"),
    ("sending", False, NOW - timedelta(seconds=200), None, True, "wait", None),
    ("sending", False, NOW - timedelta(seconds=200), "boot changed", False, "unknown", None),
    ("queued", False, NOW - timedelta(seconds=200), None, False, "none", None),
    ("succeeded", False, NOW - timedelta(seconds=999), None, True, "none", None),
])
def test_reaper_decision(status, live, since, proof, handler, kind, step):
    a = pc.reaper_decision(status=status, step_live=live, not_live_since=since, now=NOW,
                           proof=proof, has_handler=handler)
    assert (a.kind, a.step) == (kind, step)


def _terminalise_live_rows(db):
    """The module shares one DB: park every other test's live rows."""
    from sqlalchemy import update

    from app.rebuild_models import PAID_CALL_LIVE_STATUSES, PAID_CALL_MODELS

    for m in PAID_CALL_MODELS:
        db.execute(update(m).where(m.status.in_(PAID_CALL_LIVE_STATUSES)).values(status="failed"))
    db.commit()


def test_reaper_pass_end_to_end(db, SF, monkeypatch):
    _terminalise_live_rows(db)
    p = _project(db)
    rows = {st: _keyframe(db, p, status=st, task_id="T", sender_host="h", sender_boot="b",
                          sender_job="rb:j") for st in ("queued", "claimed", "sending",
                                                        "submitted", "fetching")}
    proven_send = _keyframe(db, p, status="sending", sender_host="h", sender_boot="old",
                            sender_job="rb:j2")
    monkeypatch.setitem(pc.STEP_HANDLERS, "rebuild_keyframes", "tests.nowhere.handler")
    enq = []
    common = dict(session_factory=SF, conn=object(), is_live=lambda q, prefix: False,
                  evidence=lambda row: "boot changed" if row.sender_boot == "old" else None,
                  enqueue=lambda *a: enq.append(a), models=(RebuildKeyframe,))
    out = pc.reap(now=NOW, **common)
    assert out["observed"] == 6 and not enq
    out = pc.reap(now=NOW + timedelta(seconds=60), **common)
    assert out["requeued"] == 0
    out = pc.reap(now=NOW + timedelta(seconds=200), **common)
    assert out == {"observed": 0, "requeued": 4, "unknown": 1, "waiting": 1}
    by = {r.id: _reload(db, RebuildKeyframe, r.id) for r in [*rows.values(), proven_send]}
    assert by[rows["queued"].id].status == "queued" and by[rows["queued"].id].dispatch_gen == 2
    assert by[rows["claimed"].id].status == "queued" and by[rows["claimed"].id].claim_token is None
    assert by[rows["submitted"].id].status == "submitted"
    assert by[rows["sending"].id].status == "sending"  # no proof: waits however long
    assert by[proven_send.id].status == "unknown"
    assert "released by the reaper" in by[proven_send.id].sender_note
    assert sorted((a[0], a[2], a[3]) for a in enq) == sorted([
        ("rebuild_keyframes", 2, "claim"), ("rebuild_keyframes", 2, "claim"),
        ("rebuild_keyframes", 2, "poll"), ("rebuild_keyframes", 2, "fetch")])
    # A stale step job of the old generation is fenced out.
    assert pc.claim(db, RebuildKeyframe, rows["queued"].id, gen=1) is None
    assert pc.claim(db, RebuildKeyframe, rows["queued"].id, gen=2)


def test_reaper_without_a_deployed_handler_only_resolves_sending(db, SF, monkeypatch):
    _terminalise_live_rows(db)
    # Wave 2a deployed the keyframe handler; this pins the no-handler rule.
    monkeypatch.setitem(pc.STEP_HANDLERS, "rebuild_keyframes", None)
    p = _project(db)
    q = _keyframe(db, p, status="queued", not_live_since=NOW - timedelta(seconds=999))
    s = _keyframe(db, p, status="sending", sender_boot="old", sender_host="h",
                  not_live_since=NOW - timedelta(seconds=999))
    assert pc.STEP_HANDLERS["rebuild_keyframes"] is None
    out = pc.reap(session_factory=SF, conn=object(), is_live=lambda q, p: False,
                  evidence=lambda row: "boot changed", enqueue=lambda *a: None, now=NOW,
                  models=(RebuildKeyframe,))
    assert out["unknown"] == 1 and out["requeued"] == 0
    assert _reload(db, RebuildKeyframe, q.id).status == "queued"
    assert _reload(db, RebuildKeyframe, s.id).status == "unknown"


def test_release_sender_host_attestation(db):
    p = _project(db)
    kf = _keyframe(db, p, status="sending", sender_host="c0ffee123456")
    with pytest.raises(ValueError, match="absent|exited"):
        pc.release_sender(db, kf.id, attest="c0ffee123456:running")
    with pytest.raises(ValueError, match="not the sender host"):
        pc.release_sender(db, kf.id, attest="deadbeef:absent")
    assert pc.release_sender(db, kf.id, attest="c0ffee123456abcdef:exited") == "rebuild_keyframes"
    db.commit()
    row = _reload(db, RebuildKeyframe, kf.id)
    assert row.status == "unknown" and "host attestation" in row.sender_note
    with pytest.raises(ValueError, match="not sending"):
        pc.release_sender(db, kf.id, attest="c0ffee123456:absent")
    with pytest.raises(ValueError, match="no rebuild"):
        pc.release_sender(db, str(uuid.uuid4()), attest="c:absent")


def test_live_row_guards(db):
    p = _project(db)
    assert pc.live_rows_message(pc.project_live_rows(db, p.id)) is None
    _keyframe(db, p, status="sending")
    _keyframe(db, p, status="failed")
    counts = pc.project_live_rows(db, p.id)
    assert counts["keyframes"] == 1
    assert "1 keyframes" in pc.live_rows_message(counts)


def test_selected_keyframe_and_live_attempt_uniqueness(db):
    from sqlalchemy.exc import IntegrityError

    p = _project(db)
    kf = _keyframe(db, p, selected=True, status="succeeded")
    db.add(RebuildKeyframe(project_id=p.id, batch_id="b", shot_id=kf.shot_id, selected=True,
                           status="succeeded"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    # Two sheets of different groups may both be selected; two of one group may not.
    db.add_all([RebuildKeyframe(project_id=p.id, batch_id="b", sheet_key="street", selected=True),
                RebuildKeyframe(project_id=p.id, batch_id="b", sheet_key="wedding", selected=True)])
    db.commit()
    db.add(RebuildKeyframe(project_id=p.id, batch_id="b", sheet_key="street", selected=True))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    _p, run, _d, segs = _chunked_run(db, n=1)
    _attempt(db, run, segs[0], 1, status="sending")
    with pytest.raises(IntegrityError):
        _attempt(db, run, segs[0], 2, status="queued")
    db.rollback()
    _attempt(db, run, segs[0], 2, status="failed")  # terminal rows don't count


# ---------------------------------------------------------------------------
# API: REBUILD_ENABLED gates + live-row guards (design §7.6, §5.1.7, §7.4)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(SF, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api_v2 as api_v2_module
    from app.db import get_db

    analyzed = []
    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project",
                        lambda pid, **kw: analyzed.append((pid, kw)))
    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid: None)
    from app.main import app

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
        c.analyzed = analyzed
        yield c
    app.dependency_overrides.clear()


def test_create_project_rebuild_is_gated_by_the_flag(client, monkeypatch, SF):
    monkeypatch.setattr(settings, "REBUILD_ENABLED", False)
    files = {"video_file": ("v.mp4", b"\x00" * 64, "video/mp4")}
    r = client.post("/api/v2/projects", data={"project_type": "rebuild"}, files=files)
    assert r.status_code == 409 and "REBUILD_ENABLED" in r.text
    with SF() as s:
        before = len(s.execute(select(VideoProject.id).where(
            VideoProject.project_type == "rebuild")).all())
    monkeypatch.setattr(settings, "REBUILD_ENABLED", True)
    r = client.post("/api/v2/projects", data={"project_type": "rebuild"}, files=files)
    assert r.status_code in (201, 202), r.text
    with SF() as s:
        after = len(s.execute(select(VideoProject.id).where(
            VideoProject.project_type == "rebuild")).all())
    assert after == before + 1


def test_create_run_refusals(client, monkeypatch, SF):
    with SF() as s:
        rb = _project(s)
        fs = _project(s, project_type="face_swap")
        s.add(SegmentDef(project_id=fs.id, index=0, start_sec=0, end_sec=5, action="swap"))
        s.commit()
        rb_id, fs_id = rb.id, fs.id
    for flag in (False, True):
        monkeypatch.setattr(settings, "REBUILD_ENABLED", flag)
        r = client.post(f"/api/v2/projects/{rb_id}/runs",
                        data={"prompt": "x", "model": "kling-4-0", "resolution": "720p"})
        assert r.status_code == 409, r.text
        r = client.post(f"/api/v2/projects/{fs_id}/runs",
                        data={"prompt": "x", "model": "kling-4-0", "resolution": "720p"})
        assert r.status_code == 400 and "model must be one of" in r.text
    with SF() as s:
        assert not s.execute(select(Run).where(Run.project_id.in_([rb_id, fs_id]))).first()
    # A v2v model on a v2v project still works and is stored as v2v.
    r = client.post(f"/api/v2/projects/{fs_id}/runs",
                    data={"prompt": "x", "model": "seedance", "resolution": "720p"})
    assert r.status_code == 201, r.text
    with SF() as s:
        assert s.get(Run, r.json()["run_id"]).generation_mode == "v2v"


def test_reanalyze_rebuild_flips_status_and_refuses_live_rows(client, SF):
    with SF() as s:
        p = _project(s)
        kf = _keyframe(s, p, status="claimed")
        pid, kid = p.id, kf.id
    r = client.post(f"/api/v2/projects/{pid}/analyze")
    assert r.status_code == 409 and "in flight" in r.text
    with SF() as s:
        assert s.get(VideoProject, pid).status == ProjectStatus.ready
        s.get(RebuildKeyframe, kid).status = "failed"
        s.commit()
    r = client.post(f"/api/v2/projects/{pid}/analyze")
    assert r.status_code == 202, r.text
    with SF() as s:
        proj = s.get(VideoProject, pid)
        assert proj.status == ProjectStatus.analyzing and proj.rebuild_analysis_token
        # The admission's token travels in the job's arguments (Terra round 4).
        assert client.analyzed[-1] == (pid, {"analysis_token": proj.rebuild_analysis_token})
    # A second admission while analyzing is refused (existing rule).
    assert client.post(f"/api/v2/projects/{pid}/analyze").status_code == 409


def test_delete_project_and_run_refused_while_rows_are_live(client, SF):
    with SF() as s:
        p, run, _d, segs = _chunked_run(s, n=1)
        run.status = RunStatus.incomplete
        s.commit()
        a = _attempt(s, run, segs[0], status="claimed")
        pid, rid, aid = p.id, run.id, a.id
    assert client.delete(f"/api/v2/runs/{rid}").status_code == 409
    assert client.delete(f"/api/v2/projects/{pid}").status_code == 409
    # Terra round 3: a succeeded attempt whose character check is in flight
    # still blocks the run's deletion (the call row would cascade away).
    with SF() as s:
        s.get(RebuildChunkAttempt, aid).status = "succeeded"
        s.add(RebuildLlmCall(project_id=pid, attempt_id=aid, kind="character_check", slot=0,
                             model="gemini-3.1-pro", status="sending"))
        s.commit()
    assert client.delete(f"/api/v2/runs/{rid}").status_code == 409
    with SF() as s:
        for c in s.execute(select(RebuildLlmCall).where(RebuildLlmCall.attempt_id == aid)).scalars():
            c.status = "unknown"
        s.commit()
    assert client.delete(f"/api/v2/runs/{rid}").status_code == 204
    assert client.delete(f"/api/v2/projects/{pid}").status_code == 204


def test_forms_hide_rebuild_and_kling_unless_enabled(monkeypatch):
    from app.project_types import selectable_project_types

    assert "rebuild" not in [s.key for s in selectable_project_types(rebuild_enabled=False)]
    assert "rebuild" in [s.key for s in selectable_project_types(rebuild_enabled=True)]


# ---------------------------------------------------------------------------
# Analysis (design §5.1)
# ---------------------------------------------------------------------------


@pytest.fixture()
def analyze_env(SF, monkeypatch, tmp_path):
    from app import media, pipeline_v2, scene

    @contextmanager
    def sess():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(pipeline_v2, "get_session", sess)
    monkeypatch.setattr(media, "normalize_source", lambda p: p)

    class Info:
        duration_sec = 30.0
        width, height, fps, aspect_ratio = 1080, 1920, 30.0, "9:16"

    monkeypatch.setattr(media, "probe", lambda p: Info())
    cuts = {"ranges": [(0, 2.0), (2.0, 2.5), (2.5, 6.0), (6.0, 13.0), (13.0, 14.0),
                       (14.0, 29.97)]}
    monkeypatch.setattr(scene, "detect_scenes", lambda *a, **k: cuts["ranges"])
    return src, cuts


def test_analyze_rebuild_publishes_shots_and_chunks(SF, analyze_env):
    from app import pipeline_v2

    src, cuts = analyze_env
    with SF() as s:
        p = _project(s, status=ProjectStatus.created, source_local_path=str(src))
        pid = p.id
    pipeline_v2.analyze_project(pid)
    with SF() as s:
        p = s.get(VideoProject, pid)
        assert p.status == ProjectStatus.ready
        shots = s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)
                          .order_by(RebuildShot.index)).scalars().all()
        assert [(r.start_sec, r.end_sec) for r in shots][-1] == (14.0, 30.0)
        assert {r.generation for r in shots} == {1}
        assert shots[1].folded_into == shots[2].id  # 0.5 s shot folded forward
        assert shots[0].frame_sec == 1.0
        segs = s.execute(select(SegmentDef).where(SegmentDef.project_id == pid)
                         .order_by(SegmentDef.index)).scalars().all()
        # Groups: [0,2] [2,6](0.5 s shot folded) [6,13] [13,14] and the 16 s
        # last shot split into two 8 s pieces; chunks <= 14.72 s of whole groups.
        assert [(d.action, d.start_sec, d.end_sec) for d in segs] == [
            ("swap", 0.0, 14.0), ("swap", 14.0, 22.0), ("swap", 22.0, 30.0)]
        # Old-generation keyframes + sheets exist before re-analysis ...
        s.add(RebuildKeyframe(project_id=pid, batch_id="b", shot_id=shots[0].id,
                              status="succeeded"))
        s.add(RebuildKeyframe(project_id=pid, batch_id="b", sheet_key="street",
                              status="succeeded"))
        s.add(ProjectStoryboard(project_id=pid, shots_generation=1, status="succeeded"))
        p.status = ProjectStatus.analyzing  # what the re-analysis admission does
        p.rebuild_analysis_token = "t2"
        s.commit()
    cuts["ranges"] = [(0, 10.0), (10.0, 30.0)]
    pipeline_v2.analyze_project(pid)  # no token: a stale job, does nothing
    with SF() as s:
        assert s.get(VideoProject, pid).status == ProjectStatus.analyzing
    pipeline_v2.analyze_project(pid, analysis_token="t2")
    with SF() as s:
        shots = s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)).scalars().all()
        assert {r.generation for r in shots} == {2} and len(shots) == 2
        assert not s.execute(select(RebuildKeyframe).where(
            RebuildKeyframe.project_id == pid)).first()
        assert s.execute(select(ProjectStoryboard).where(
            ProjectStoryboard.project_id == pid)).first()  # kept for history


def test_analyze_does_not_publish_after_recovery_reset_the_project(SF, analyze_env, monkeypatch):
    from app import pipeline_v2, rebuild

    src, _ = analyze_env
    with SF() as s:
        p = _project(s, status=ProjectStatus.created, source_local_path=str(src))
        pid = p.id
    real = rebuild.detect_shots

    def detect_then_reset(*a, **kw):
        out = real(*a, **kw)
        with SF() as s2:
            s2.get(VideoProject, pid).status = ProjectStatus.failed
            s2.commit()
        return out

    monkeypatch.setattr(rebuild, "detect_shots", detect_then_reset)
    pipeline_v2.analyze_project(pid)
    with SF() as s:
        assert s.get(VideoProject, pid).status == ProjectStatus.failed
        assert not s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)).first()


@pytest.mark.parametrize("status", [ProjectStatus.ready, ProjectStatus.failed])
def test_stale_rebuild_analysis_job_does_nothing(SF, analyze_env, status):
    """Terra round 1: a stale / duplicate analysis job of a rebuild project
    must not flip a ready (or recovery-failed) project back to analyzing
    itself — only the admission does, after the live-row guard."""
    from app import pipeline_v2

    src, _ = analyze_env
    with SF() as s:
        p = _project(s, status=status, source_local_path=str(src))
        kf = _keyframe(s, p, status="claimed")  # a live job admitted meanwhile
        pid, kid = p.id, kf.id
    pipeline_v2.analyze_project(pid)
    with SF() as s:
        assert s.get(VideoProject, pid).status == status
        assert s.get(RebuildKeyframe, kid) is not None
        assert s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)).scalars() \
            .all().__len__() == 1  # only the keyframe's own shot


@pytest.mark.parametrize("raise_after", [False, True])
def test_analysis_job_is_fenced_on_the_generation_it_started_from(SF, analyze_env, monkeypatch,
                                                                  raise_after):
    """Terra round 2: two jobs start while the project is analyzing; the other
    one publishes (and work is admitted, the project ready again) while this
    one detects. This one must neither publish nor — when it then fails — mark
    the ready project failed."""
    from app import pipeline_v2, rebuild

    src, _ = analyze_env
    with SF() as s:
        p = _project(s, status=ProjectStatus.analyzing, source_local_path=str(src),
                     rebuild_analysis_token="tok")
        pid = p.id
    real = rebuild.detect_shots

    def other_job_publishes(*a, **kw):
        out = real(*a, **kw)
        with SF() as s2:
            from app import rebuild_analysis

            proj = s2.get(VideoProject, pid)
            rebuild_analysis.publish_analysis(s2, proj, [(0.0, 30.0)])
            proj.status = ProjectStatus.analyzing  # a NEW admission flipped it again
            s2.commit()
        if raise_after:
            raise RuntimeError("detector crashed")
        return out

    monkeypatch.setattr(rebuild, "detect_shots", other_job_publishes)
    if raise_after:
        with pytest.raises(RuntimeError):
            pipeline_v2.analyze_project(pid, analysis_token="tok")
    else:
        pipeline_v2.analyze_project(pid, analysis_token="tok")
    with SF() as s:
        assert s.get(VideoProject, pid).status == ProjectStatus.analyzing  # untouched
        shots = s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)).scalars().all()
        assert len(shots) == 1 and shots[0].generation == 1  # the other job's publish


@pytest.mark.parametrize("raise_after", [False, True])
def test_reaped_analysis_job_cannot_touch_the_next_admission(SF, analyze_env, monkeypatch,
                                                             raise_after):
    """Terra round 3: job A is reaped by recovery (failed), the operator admits
    B (analyzing, new token, same generation). A must neither publish nor mark
    B's analysis failed."""
    from app import pipeline_v2, rebuild

    src, _ = analyze_env
    with SF() as s:
        p = _project(s, status=ProjectStatus.analyzing, source_local_path=str(src),
                     rebuild_analysis_token="token-a")
        pid = p.id
    real = rebuild.detect_shots

    def reaped_then_readmitted(*a, **kw):
        out = real(*a, **kw)
        with SF() as s2:
            proj = s2.get(VideoProject, pid)
            proj.status = ProjectStatus.analyzing  # recovery: failed; admission B: analyzing
            proj.rebuild_analysis_token = "token-b"
            s2.commit()
        if raise_after:
            raise RuntimeError("detector crashed")
        return out

    monkeypatch.setattr(rebuild, "detect_shots", reaped_then_readmitted)
    if raise_after:
        with pytest.raises(RuntimeError):
            pipeline_v2.analyze_project(pid, analysis_token="token-a")
    else:
        pipeline_v2.analyze_project(pid, analysis_token="token-a")
    with SF() as s:
        proj = s.get(VideoProject, pid)
        assert proj.status == ProjectStatus.analyzing and proj.rebuild_analysis_token == "token-b"
        assert not s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)).first()


def test_old_queued_job_never_adopts_a_newer_admission(SF, analyze_env):
    """Terra round 4: an old job (token A, or none) starting after a newer
    admission (token B) does nothing at all."""
    from app import pipeline_v2

    src, _ = analyze_env
    with SF() as s:
        p = _project(s, status=ProjectStatus.analyzing, source_local_path=str(src),
                     rebuild_analysis_token="token-b")
        pid = p.id
    pipeline_v2.analyze_project(pid, analysis_token="token-a")
    pipeline_v2.analyze_project(pid)
    with SF() as s:
        assert s.get(VideoProject, pid).status == ProjectStatus.analyzing
        assert not s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)).first()
    pipeline_v2.analyze_project(pid, analysis_token="token-b")
    with SF() as s:
        assert s.get(VideoProject, pid).status == ProjectStatus.ready


def test_first_analysis_takes_its_own_token(SF, analyze_env):
    from app import pipeline_v2

    src, _ = analyze_env
    with SF() as s:
        p = _project(s, status=ProjectStatus.created, source_local_path=str(src))
        pid = p.id
    pipeline_v2.analyze_project(pid)
    with SF() as s:
        proj = s.get(VideoProject, pid)
        assert proj.status == ProjectStatus.ready and proj.rebuild_analysis_token


def test_rechunk_keeps_brand_cards(SF, analyze_env):
    from app import pipeline_v2, rebuild_analysis

    src, cuts = analyze_env
    cuts["ranges"] = [(0, 5.0), (5.0, 10.0), (10.0, 27.0), (27.0, 30.0)]
    with SF() as s:
        p = _project(s, status=ProjectStatus.created, source_local_path=str(src))
        pid = p.id
    pipeline_v2.analyze_project(pid)
    with SF() as s:
        shots = rebuild_analysis.project_shots(s, pid)
        shots[-1].is_brand_card = True
        shots[1].force_anchor = True
        plan = rebuild_analysis.rechunk(s, s.get(VideoProject, pid), keep_brand_cards=True)
        s.commit()
        assert plan[-1].action == "keep" and plan[-1].start == 27.0
        segs = s.execute(select(SegmentDef).where(SegmentDef.project_id == pid)
                         .order_by(SegmentDef.index)).scalars().all()
        assert [d.action for d in segs][-1] == "keep"


# ---------------------------------------------------------------------------
# Defence in depth: the v2v pipeline refuses an images-only run (§6.4)
# ---------------------------------------------------------------------------


def test_v2v_submit_and_recovery_refuse_images_only_runs(SF, monkeypatch):
    from app import pipeline_v2, recovery

    @contextmanager
    def sess():
        s = SF()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr(pipeline_v2, "get_session", sess)
    monkeypatch.setattr(recovery, "get_session", sess)
    with SF() as s:
        _p, run, _d, segs = _chunked_run(s, n=1)
        rid = run.id
    assert pipeline_v2.submit_run(rid, kie=object()) is None
    with SF() as s:
        assert s.get(Run, rid).status == RunStatus.queued  # untouched
        with pytest.raises(pipeline_v2.ImagesOnlyRunRefused):
            pipeline_v2._build_submit_state(rid, kie=object(), gdrive=None, session=s)
    with pytest.raises(pipeline_v2.ImagesOnlyRunRefused):
        pipeline_v2.poll_run_once(rid, pending={}, kie=object())
    recovery._repoll_generating_segments(rid, redis_conn=None)  # returns quietly
    assert pipeline_v2.process_run(rid, kie=object()) is None


# ---------------------------------------------------------------------------
# Migration vs ORM, on SQLite (design §4, §14)
# ---------------------------------------------------------------------------


def _migration():
    import importlib.util

    path = os.path.join(BACKEND, "alembic", "versions", "t4d9f2b7c315_rebuild_from_reference.py")
    spec = importlib.util.spec_from_file_location("mig_t4d9f2b7c315", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_matches_the_models():
    from app import rebuild_models as rm

    mig = _migration()
    assert mig.down_revision == "t4d9a2c6e815"  # main's AICP-348 head
    src = open(mig.__file__).read()
    for table in ("rebuild_shots", "project_storyboards", "rebuild_keyframes",
                  "rebuild_chunk_attempts", "rebuild_llm_calls"):
        part = src.split(f'op.create_table(\n        "{table}",', 1)[1].split("op.create_index(", 1)[0]
        cols = set(re.findall(r'sa\.Column\("([a-z_0-9]+)"', part))
        if "_paid_call_columns()" in part:
            cols |= {c.name for c in rm.RebuildKeyframe.__table__.columns
                     if c.name in {"status", "dispatch_gen", "claim_token", "rq_job_id",
                                   "not_live_since", "send_deadline", "sender_host",
                                   "sender_boot", "sender_job", "sender_note", "task_id",
                                   "request", "error", "estimate_credits", "cost_credits",
                                   "fetch_attempts", "created_at", "updated_at",
                                   "finished_at"}}
        if "_dispatch_columns()" in part:
            cols |= {"status", "dispatch_gen", "claim_token", "attempts", "rq_job_id",
                     "not_live_since", "error", "created_at", "started_at", "finished_at",
                     "updated_at"}
        assert cols == {c.name for c in Base.metadata.tables[table].columns}, table
    for name in ("PAID_CALL_STATUS_CHECK", "PAID_CALL_LIVE_WHERE", "STORYBOARD_LIVE_WHERE",
                 "LLM_SHOT_LIVE_WHERE", "LLM_GLOBAL_LIVE_WHERE", "LLM_CHECK_LIVE_WHERE",
                 "LLM_CHECK_SLOT_WHERE", "KF_SHOT_SELECTED_WHERE", "KF_SHEET_SELECTED_WHERE",
                 "ATTEMPT_SELECTED_WHERE"):
        assert getattr(mig, name) == getattr(rm, name), name
    for t in Base.metadata.tables.values():
        for idx in t.indexes:
            where = idx.dialect_options["postgresql"].get("where")
            if where is not None and t.name.startswith(("rebuild_", "project_storyboards")):
                assert f'"{idx.name}"' in src, idx.name
    body = src.split("def upgrade", 1)[1].split("def downgrade", 1)[0]
    # The revision never USES the new labels (same-transaction rule).
    assert body.count("'rebuild'") == 1 and body.count("'kling-4-0'") == 1


def test_migration_on_sqlite_and_downgrade_refuses_rows():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect, text

    eng = create_engine(f"sqlite:///{tempfile.mkstemp(suffix='.db')[1]}")
    Base.metadata.create_all(eng)
    mig = _migration()
    with eng.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mig.downgrade()
            assert "rebuild_shots" not in inspect(conn).get_table_names()
            assert "generation_mode" not in {c["name"] for c in inspect(conn).get_columns("runs")}
            mig.upgrade()
        conn.execute(text("INSERT INTO video_projects (id, project_type, source_type, source_ref, "
                          "status, mute_source) VALUES ('p1', 'face_swap', 'upload', 'x', "
                          "'ready', 0)"))
        conn.execute(text("INSERT INTO runs (id, project_id, prompt, reference_image_urls, model, "
                          "resolution, audio_mode, status, preprocess_enabled, "
                          "preprocess_auto_approve) VALUES ('r1', 'p1', '', '[]', 'seedance', "
                          "'720p', 'original', 'done', 0, 0)"))
        # The server default backfills / defaults v2v.
        assert conn.execute(text("SELECT generation_mode FROM runs")).scalar() == "v2v"
        with pytest.raises(Exception):
            conn.execute(text("UPDATE runs SET generation_mode = 'bogus'"))
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO rebuild_shots (id, project_id, \"index\", generation, "
                          "start_sec, end_sec, frame_sec, is_brand_card, force_anchor, "
                          "created_at, updated_at) VALUES ('s1', 'p1', 0, 1, 0, 1, 0.5, 0, 0, "
                          "'2026-11-03', '2026-11-03')"))
        with Operations.context(MigrationContext.configure(conn)):
            with pytest.raises(RuntimeError, match="rebuild_shots"):
                mig.downgrade()
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM rebuild_shots"))
        conn.execute(text("UPDATE runs SET generation_mode = 'images_only'"))
        with Operations.context(MigrationContext.configure(conn)):
            with pytest.raises(RuntimeError, match="images-only"):
                mig.downgrade()


def test_drain_check_counts_the_rebuild_tables():
    path = os.path.join(REPO, "deploy", "drain-check.sql")
    if not os.path.exists(path):
        pytest.skip("repo root not mounted")
    sql = open(path).read()
    live = "('queued', 'claimed', 'sending', 'submitted', 'fetching')"
    for table in ("rebuild_keyframes", "rebuild_chunk_attempts", "rebuild_llm_calls"):
        assert f"FROM {table} WHERE status IN {live}" in sql
    assert "FROM project_storyboards WHERE status IN ('queued', 'running')" in sql
    body = "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))
    assert body.rstrip().endswith(";") and body.count(";") == 1


def test_rebuild_files_are_outside_the_disk_cleanup_sweep():
    """raw.mp4 is the only copy of a paid output: no rebuild path may sit under
    a directory disk_cleanup.sweep deletes (tail_edits/<id>/body.mp4,
    runs/<id>/clips/, runs/<id>/segment_takes/)."""
    from app import storage

    paths = [
        storage.rebuild_shots_cache_path("p"), storage.rebuild_frame_path("p", "s"),
        storage.rebuild_keyframe_path("p", "k"), storage.rebuild_run_chunk_dir("r", "p", 0),
        os.path.join(storage.rebuild_attempt_dir("a", "r", "p"), "raw.mp4"),
    ]
    clips = storage.run_dir("r", "p") + os.sep + "clips"
    takes = storage.run_dir("r", "p") + os.sep + "segment_takes"
    for path in paths:
        assert not path.startswith(clips + os.sep) and not path.startswith(takes + os.sep)
        assert "tail_edits" not in path
    src = open(os.path.join(BACKEND, "app", "disk_cleanup.py")).read()
    assert "rebuild/" not in src and "rebuild_" not in src


def test_enqueue_analyze_project_on_a_real_rq_queue(monkeypatch):
    """Terra round 5 (P0): the token must reach the job as a keyword on a REAL
    rq 1.15 Queue (the API tests mock the enqueue), and the token-less enqueue
    keeps exactly its old shape."""
    import fakeredis
    from rq import Queue

    from app import tasks

    conn = fakeredis.FakeStrictRedis()
    q = Queue("analyze", connection=conn)
    monkeypatch.setattr(tasks, "_get_queue", lambda name=None: q)
    tasks.enqueue_analyze_project("p1")
    tasks.enqueue_analyze_project("p2", analysis_token="tok")
    jobs = {j.args[0]: j for j in q.get_jobs()}
    assert jobs["p1"].func_name == "app.tasks.run_analyze_project"
    assert (jobs["p1"].args, jobs["p1"].kwargs) == (("p1",), {})
    assert (jobs["p2"].args, jobs["p2"].kwargs) == (("p2",), {"analysis_token": "tok"})
    assert jobs["p2"].timeout == tasks.ANALYZE_JOB_TIMEOUT


def test_run_analyze_project_forwards_the_token(monkeypatch):
    from app import pipeline_v2, tasks

    calls = []
    monkeypatch.setattr(pipeline_v2, "analyze_project",
                        lambda pid, **kw: calls.append((pid, kw)))
    monkeypatch.setattr(tasks, "_enqueue_transcribe_after_analyze", lambda pid: None)
    tasks.run_analyze_project("p1")
    tasks.run_analyze_project("p2", analysis_token="tok")
    assert calls == [("p1", {}), ("p2", {"analysis_token": "tok"})]


def test_enqueue_step_and_liveness_on_a_real_rq_queue(monkeypatch):
    """enqueue_step / step_is_live / _job_evidence against a real rq 1.15
    Queue on fakeredis (the reaper tests inject fakes)."""
    import fakeredis
    from rq import Queue
    from rq.job import Job

    from app import tasks

    conn = fakeredis.FakeStrictRedis()
    q = Queue("process", connection=conn)
    monkeypatch.setattr(tasks, "_get_queue", lambda name=None: q)
    monkeypatch.setattr(tasks, "_live_worker_current_jobs", lambda c: set())
    prefix = pc.step_job_prefix("rebuild_keyframes", "row1", 1)
    assert not pc.step_is_live(conn, "process", prefix)
    jid = pc.enqueue_step("rebuild_keyframes", "row1", 1, "send")
    assert jid == "rb:rebuild_keyframes:row1:1:send"
    job = Job.fetch(jid, connection=conn)
    assert job.func_name == "app.rebuild_paid_call.run_step"
    assert job.args == ("rebuild_keyframes", "row1", 1, "send")
    assert job.result_ttl == pc.STEP_JOB_TTL_SEC and job.failure_ttl == pc.STEP_JOB_TTL_SEC
    assert pc.step_is_live(conn, "process", prefix)
    assert not pc.step_is_live(conn, "process", pc.step_job_prefix("rebuild_keyframes", "row1", 2))
    assert pc._job_evidence(conn, jid)[0] == "queued"
    # A delayed poll lands in the scheduled registry and counts as live.
    q.empty()
    pjid = pc.enqueue_step("rebuild_keyframes", "row2", 3, "poll", delay_sec=10)
    assert pjid.startswith("rb:rebuild_keyframes:row2:3:poll:")
    assert pc.step_is_live(conn, "process", pc.step_job_prefix("rebuild_keyframes", "row2", 3))
    # Evidence: a missing hash proves nothing.
    assert pc._job_evidence(conn, "rb:nope") == (None, None)
    conn.set("rebuild:boot:h", "b2")

    class Row:
        sender_host, sender_boot, sender_job = "h", "b1", jid

    assert pc.default_evidence(conn)(Row()) == "sender container restarted (boot b1 -> b2)"
