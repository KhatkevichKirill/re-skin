"""AICP-365 acceptance: the localisation flow on the real autopilot core.

Project 46cc6443's inputs (7 whole-video segments, its v3 transcript, the es /
pt translations) driven through every step with mocked providers: the
transcription and translation calls are faked at the localisation layer, the
run pipeline never runs (runs are finished by the test), ffmpeg is faked
(voice reference, clip probe, frame sheet). Real: the core's ticks, budget and
approvals, the durable job rows, the AICP-347 re-cut, the create_run /
rerun_segments endpoints and the AICP-335 prompt builder.

Acceptance (AICP-365): no paid call before the budget approval; the probe run
holds segment 1 only and no voice; nothing of segments 2..N is launched before
the listening approval; the second run's stitch plan is the TEMP library clip of
the accepted probe + segments 2..7 and its voice reference comes from that same
clip; the prompts match the references (tests/test_autopilot_localisation_plan.py).
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from types import SimpleNamespace

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

import app.api_v2 as api_v2_module
from app import localisation, localisation_jobs, media
from app.autopilot import compute as compute_mod
from app.autopilot import core, registry
from app.autopilot.flows import localisation as flow_mod
from app.autopilot.judgement import FakePort
from app.config import settings
from app.db import Base
from app.localisation_jobs_models import LocalisationPromptJob, ProjectTranscription
from app.models import LibraryClip, Run, RunSegment, SegmentDef, VideoProject
from app.models_autopilot import (
    AutopilotApproval, AutopilotEffect, AutopilotGate, AutopilotStep, ProjectAutopilot, utcnow,
)
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus

from tests.test_autopilot_localisation_plan import PROBE, SEG1
from tests.test_autopilot_localisation_plan import DATA, SEGMENTS, transcript_v3

DURATION = DATA["duration_sec"]


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


class Clock:
    def __init__(self):
        self.t = utcnow()

    def __call__(self):
        return self.t

    def advance(self, sec):
        from datetime import timedelta

        self.t = self.t + timedelta(seconds=sec)


@pytest.fixture()
def env(monkeypatch, SF, tmp_path):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(core, "wake", lambda: None)
    for name in ("enqueue_analyze_project", "enqueue_process_run", "enqueue_preprocess_run",
                 "enqueue_transcribe_project"):
        monkeypatch.setattr(api_v2_module, name, lambda *a, **k: None)
    rang: list = []
    monkeypatch.setattr(localisation_jobs, "enqueue", rang.append)
    monkeypatch.setattr(localisation_jobs, "_localise_worker_alive", lambda: True)
    calls = {"transcribe": 0, "translate": []}

    def fake_transcribe(path, kie=None, spend_scope=None, **kw):
        calls["transcribe"] += 1
        return transcript_v3()

    def fake_translate(window_lines, **kw):
        calls["translate"].append(kw["target_language"])
        return copy.deepcopy(DATA["translated"][kw["target_language"]])

    monkeypatch.setattr(localisation, "transcribe_video", fake_transcribe)
    monkeypatch.setattr(localisation, "translate_lines", fake_translate)

    def fake_voice(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            fout.write(fin.read())
        return 12.0

    monkeypatch.setattr(media, "make_voice_reference", fake_voice)
    monkeypatch.setattr(media, "probe", lambda path: SimpleNamespace(
        duration_sec=12.04, width=720, height=1280, fps=24.0))

    def fake_captions(args, out_dir):
        return {"frames": 158, "hits": 0, "hit_ratio": calls.get("caption_ratio", 0.0),
                "spans": [], "threshold": 0.2}

    def fake_sheet(args, out_dir):
        path = os.path.join(out_dir, "probe_sheet.jpg")
        with open(path, "wb") as fh:
            fh.write(os.urandom(32))
        calls.setdefault("sheets", []).append(args)
        return {"sheet": path, "samples": [], "source_sec": 12.13, "result_sec": 12.04,
                "length_delta_sec": -0.09, "max_diff": 7.5}

    monkeypatch.setitem(registry._compute, ("localisation_source_captions", 1), fake_captions)
    monkeypatch.setitem(registry._compute, ("localisation_probe_sheet", 1), fake_sheet)
    registry.register(flow_mod.FLOW)
    db = SF()
    if db.get(AutopilotGate, 1) is None:
        db.add(AutopilotGate(id=1, held=False, generation=0))
    else:
        db.get(AutopilotGate, 1).held = False
    db.commit()
    db.close()
    enq: list = []
    hooks = core.Hooks(enqueue_compute=lambda eid, jid: enq.append(eid),
                       compute_is_live=lambda eff: True, port=FakePort(), now=Clock())
    return SimpleNamespace(SF=SF, hooks=hooks, compute=enq, rang=rang, calls=calls,
                           tmp=tmp_path)


def make_project(env, *, languages=("es",), mute_source=False, transcript=None):
    db = env.SF()
    src = env.tmp / f"src-{os.urandom(4).hex()}.mp4"
    src.write_bytes(os.urandom(256))
    p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                     status=ProjectStatus.ready, project_type="localisation", mode="assistant",
                     localisation_scope="whole", duration_sec=DURATION, mute_source=mute_source)
    if transcript is not None:
        p.transcript, p.transcript_status = transcript, "ready"
    db.add(p)
    db.flush()
    for s in SEGMENTS:
        db.add(SegmentDef(project_id=p.id, index=s["index"], start_sec=s["start_sec"],
                          end_sec=s["end_sec"], action="swap", pre_roll_sec=0.0,
                          post_roll_sec=0.0))
    db.flush()
    # The fixture's ids are what the references are keyed by: re-key them.
    rows = db.execute(select(SegmentDef).where(SegmentDef.project_id == p.id)
                      .order_by(SegmentDef.index)).scalars().all()
    params = flow_mod.FLOW.validate_params(p, {"languages": list(languages)})
    ap = core.create_autopilot(db, p, flow_mod.FLOW, params)
    db.commit()
    out = SimpleNamespace(pid=p.id, apid=ap.id, seg_ids=[r.id for r in rows], src=str(src))
    db.close()
    return out


def tick(env, prj, n=1):
    for _ in range(n):
        env.hooks.now.advance(31)
        core.tick_autopilot(prj.apid, session_factory=env.SF, hooks=env.hooks)


def run_compute(env):
    while env.compute:
        compute_mod.run(env.compute.pop(0), session_factory=env.SF)


def run_jobs(env):
    while env.rang:
        job = env.rang.pop(0)
        if isinstance(job, ProjectTranscription):
            localisation_jobs.run_transcription_job(job.id, job.dispatch_gen, session_factory=env.SF)
        else:
            localisation_jobs.run_translate_job(job.id, job.dispatch_gen, session_factory=env.SF)


def settle(env, prj, rounds=30, *, until=None, jobs=True):
    """Tick, run queued compute and (unless *jobs* is False) jobs, until *until()*."""
    for _ in range(rounds):
        tick(env, prj)
        run_compute(env)
        if jobs:
            run_jobs(env)
        if until is not None and until():
            return
    if until is not None:
        raise AssertionError("settle: condition never met")


def q(env, model, **where):
    db = env.SF()
    try:
        stmt = select(model)
        for k, v in where.items():
            stmt = stmt.where(getattr(model, k) == v)
        return db.execute(stmt).scalars().all()
    finally:
        db.close()


def pending(env, prj, kind=None):
    rows = q(env, AutopilotApproval, autopilot_id=prj.apid, status="pending")
    return [a for a in rows if kind is None or a.kind == kind]


def answer(env, prj, a, decision="approve", **kw):
    db = env.SF()
    try:
        return core.answer(db, prj.pid, a.id, decision, **kw)
    finally:
        db.close()


def ap_row(env, prj):
    db = env.SF()
    try:
        return db.get(ProjectAutopilot, prj.apid)
    finally:
        db.close()


def steps(env, prj):
    return {s.step_id: s for s in q(env, AutopilotStep, autopilot_id=prj.apid)}


def runs(env, prj):
    db = env.SF()
    try:
        return db.execute(select(Run).where(Run.project_id == prj.pid)
                          .order_by(Run.created_at)).scalars().all()
    finally:
        db.close()


def finish_run(env, rid, *, status=RunStatus.done):
    db = env.SF()
    run = db.get(Run, rid)
    project = db.get(VideoProject, run.project_id)
    plan = run.stitch_plan or []
    active = {e["segment_def_id"] for e in plan if e.get("kind") == "segment"}
    defs = db.execute(select(SegmentDef).where(SegmentDef.project_id == project.id)
                      .order_by(SegmentDef.index)).scalars().all()
    existing = {rs.segment_def_id: rs for rs in db.execute(
        select(RunSegment).where(RunSegment.run_id == rid)).scalars()}
    for sd in defs:
        if sd.id not in active:
            continue
        rs = existing.get(sd.id) or RunSegment(run_id=rid, segment_def_id=sd.id, index=sd.index)
        db.add(rs)
        res = env.tmp / f"seg-{os.urandom(4).hex()}.mp4"
        res.write_bytes(os.urandom(64))
        rs.status, rs.local_result_path = SegmentStatus.completed, str(res)
        rs.seedance_task_id = f"t-{os.urandom(3).hex()}"
        if status != RunStatus.done and sd.index == max(d.index for d in defs if d.id in active):
            rs.status, rs.local_result_path = SegmentStatus.failed, None
    final = env.tmp / f"final-{os.urandom(4).hex()}.mp4"
    final.write_bytes(os.urandom(128))
    run.result_local_path = str(final) if status == RunStatus.done else None
    run.segment_snapshot = [{"id": d.id, "index": d.index, "action": d.action,
                             "start_sec": d.start_sec, "end_sec": d.end_sec,
                             "pre_roll_sec": 0.0, "post_roll_sec": 0.0} for d in defs]
    run.status = status
    db.commit()
    db.close()


def drive_to_budget(env, prj):
    settle(env, prj, until=lambda: bool(pending(env, prj, "budget")))
    return pending(env, prj, "budget")[0]


def drive_to_listen(env, prj):
    b = drive_to_budget(env, prj)
    answer(env, prj, b)
    settle(env, prj, until=lambda: len(runs(env, prj)) == 1)
    probe = runs(env, prj)[0]
    finish_run(env, probe.id)
    settle(env, prj, until=lambda: bool(pending(env, prj, "listening_probe")))
    return probe, pending(env, prj, "listening_probe")[0]


# ---------------------------------------------------------------------------


def test_happy_path_one_language(env):
    prj = make_project(env)
    budget = drive_to_budget(env, prj)

    # No paid call before the budget approval: no job, no run, no provider call.
    assert q(env, ProjectTranscription, project_id=prj.pid) == []
    assert runs(env, prj) == []
    assert env.calls["transcribe"] == 0
    lines = {ln["step_id"]: ln["credits"] for ln in budget.proposal["lines"] if ln["step_id"]}
    # AICP-366's qa:es carries the one budgeted ASR caption of the final.
    assert set(lines) == {"transcribe", "translate:es", "probe:es", "rest:es", "qa:es"}
    assert lines["probe:es"] == 13 * 80 and lines["rest:es"] == (14 + 10 + 14 + 9 + 9 + 13) * 80
    db = env.SF()
    assert db.get(VideoProject, prj.pid).mute_source is True  # intake turned it on
    db.close()

    probe, listen = drive_to_listen(env, prj)
    # The probe: segment 1 only, no voice, its prompt without a voice-reference claim.
    assert probe.stitch_plan == [{"kind": "segment", "segment_def_id": prj.seg_ids[0]}]
    assert list(probe.reference_audio_urls or []) == []
    assert probe.mute_source is True
    assert overrides(env, probe.id) == {prj.seg_ids[0]: probe.prompt}
    assert "reference audio" not in probe.prompt and "\nSpeakers:\n" in probe.prompt
    assert env.calls["transcribe"] == 1 and env.calls["translate"] == ["es"]
    # Nothing of segments 2..N before the listening approval.
    tick(env, prj, 3)
    assert len(runs(env, prj)) == 1
    assert listen.subject == "es"
    assert [a["kind"] for a in listen.artifacts] == ["video", "image"]

    answer(env, prj, listen)
    settle(env, prj, until=lambda: len(runs(env, prj)) == 2)
    rest = runs(env, prj)[1]
    clips = q(env, LibraryClip)
    clip = next(c for c in clips if c.name.startswith(f"TEMP autopilot {prj.pid[:8]} es"))
    assert clip.language is None
    assert rest.stitch_plan == [{"kind": "library", "clip_id": clip.id}] + [
        {"kind": "segment", "segment_def_id": sid} for sid in prj.seg_ids[1:]]
    # Voice reference = the TEMP clip's audio (same bytes as the accepted probe final).
    voice = list(rest.reference_audio_urls)
    assert len(voice) == 1
    with open(voice[0], "rb") as fh, open(probe_final(env, probe.id), "rb") as fp:
        assert fh.read() == fp.read()
    assert rest.mute_source is True and rest.language == "es"
    rs_prompts = overrides(env, rest.id)
    assert set(rs_prompts) == set(prj.seg_ids[1:])
    assert "Speakers (the reference audio contains both human voices):" in rest.prompt

    finish_run(env, rest.id)
    # The finishing steps (AICP-366) continue from here; test_autopilot_localisation_finish.py.
    settle(env, prj, until=lambda: steps(env, prj)["rest:es"].status == "completed")
    st = steps(env, prj)
    # The real AICP-347 re-cut ran and moved boundaries off spoken lines.
    assert st["segments"].output["recut"]["applied"] is True
    assert st["segments"].output["segments"][0][2] == pytest.approx(13.0)
    assert st["rest:es"].output["run_id"] == rest.id
    assert st["rest:es"].output["temp_clip_id"] == clip.id
    assert st["probe:es"].output["probe_run_id"] == probe.id
    assert "WOMAN (female, person)" in st["breakdown"].reason


def overrides(env, rid):
    db = env.SF()
    try:
        return {rs.segment_def_id: rs.prompt_override for rs in db.execute(
            select(RunSegment).where(RunSegment.run_id == rid)).scalars()}
    finally:
        db.close()


def probe_final(env, rid):
    db = env.SF()
    try:
        return db.get(Run, rid).result_local_path
    finally:
        db.close()


def _keep_the_cut(monkeypatch):
    """The references were run on the operator's hand cut of 46cc6443; keep it
    (the re-cut handler then reports "unchanged")."""
    from app import speech_recut

    monkeypatch.setattr(speech_recut, "recut_project_segments", lambda db, project, **kw:
                        SimpleNamespace(plan=SimpleNamespace(to_dict=lambda: {
                            "feasible": True, "changed": False, "boundaries": []}),
                            fingerprint="kept", applied=False, warnings=[], segments=[]))


def test_segment_prompts_reach_the_runs(env, monkeypatch):
    """create_run stores the per-segment prompts the flow built, byte for byte
    the AICP-335 references (refs/pt_run_inputs.json through the fixture)."""
    _keep_the_cut(monkeypatch)
    prj = make_project(env, languages=("pt",))
    probe, listen = drive_to_listen(env, prj)
    from tests.test_autopilot_localisation_plan import prompts_for

    want = prompts_for("pt")
    # the fixture's segment ids -> this project's ids, by index
    by_index = {i: sid for i, sid in enumerate(prj.seg_ids)}
    fixture_ids = PROBE["order"]
    assert probe.prompt == want["probe_prompt"]
    answer(env, prj, listen)
    settle(env, prj, until=lambda: len(runs(env, prj)) == 2)
    rest = runs(env, prj)[1]
    stored = overrides(env, rest.id)
    for i, fid in enumerate(fixture_ids[1:], start=1):
        assert stored[by_index[i]] == want["segment_prompts"][fid].strip()
    assert by_index[0] not in stored
    assert overrides(env, probe.id) == {by_index[0]: want["probe_prompt"].strip()}


def test_budget_rejected_stops_without_spending(env):
    prj = make_project(env)
    b = drive_to_budget(env, prj)
    answer(env, prj, b, "reject")
    tick(env, prj, 3)
    assert ap_row(env, prj).status == "stopped"
    assert q(env, ProjectTranscription, project_id=prj.pid) == []
    assert env.calls["transcribe"] == 0 and runs(env, prj) == []


def test_ready_v3_transcript_is_reused(env):
    prj = make_project(env, transcript=transcript_v3(), mute_source=True)
    b = drive_to_budget(env, prj)
    assert b.step_id == "translate:es"  # the first paid launch is the translation
    lines = {ln["step_id"]: ln["credits"] for ln in b.proposal["lines"] if ln["step_id"]}
    assert "transcribe" not in lines or lines["transcribe"] == 0
    answer(env, prj, b)
    settle(env, prj, until=lambda: len(runs(env, prj)) == 1)
    assert env.calls["transcribe"] == 0


def test_rejected_probe_regenerates_with_the_operator_cue(env):
    prj = make_project(env)
    probe, listen = drive_to_listen(env, prj)
    budget_before = ap_row(env, prj).approved_budget_mcredits
    answer(env, prj, listen, "reject", comment="the interviewer must be a WOMAN")
    settle(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    fix = pending(env, prj, "regeneration")[0]
    assert fix.name.startswith("fix:r1:")
    rerun = next(o for o in fix.options if o["id"] == "rerun")
    prompt = rerun["action"]["params"]["segments"][0]["prompt"]
    assert prompt.endswith("Corrections from the operator's listening check — follow them "
                           "exactly:\n- the interviewer must be a WOMAN")
    # The rejected listening answer was consumed by the cue (no grace stop).
    lst = next(a for a in q(env, AutopilotApproval, autopilot_id=prj.apid)
               if a.name.startswith("listen:r1:"))
    assert lst.status == "consumed"
    env.hooks.now.advance(settings.AUTOPILOT_REJECT_GRACE_SEC + 60)
    tick(env, prj)
    assert ap_row(env, prj).status == "active"

    answer(env, prj, fix, option="rerun")
    settle(env, prj, until=lambda: q(env, Run, id=probe.id)[0].status == RunStatus.queued)
    db = env.SF()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == probe.id)).scalars().one()
    assert rs.prompt_override == prompt
    db.close()
    assert ap_row(env, prj).approved_budget_mcredits > budget_before  # Q6
    assert len(runs(env, prj)) == 1  # still no remaining-segments run

    finish_run(env, probe.id)
    settle(env, prj, until=lambda: bool(pending(env, prj, "listening_probe")))
    listen2 = pending(env, prj, "listening_probe")[0]
    assert listen2.name.startswith("listen:r2:")
    answer(env, prj, listen2)
    settle(env, prj, until=lambda: len(runs(env, prj)) == 2)
    assert steps(env, prj)["probe:es"].output["round"] == 2


def test_rejected_fix_fails_the_autopilot(env):
    prj = make_project(env)
    probe, listen = drive_to_listen(env, prj)
    answer(env, prj, listen, "reject", comment="wrong voice")
    settle(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    answer(env, prj, pending(env, prj, "regeneration")[0], "reject")
    tick(env, prj, 2)
    ap = ap_row(env, prj)
    assert ap.status == "failed" and "wrong voice" in ap.status_reason
    assert len(runs(env, prj)) == 1


def test_failed_transcription_asks_a_regeneration(env, monkeypatch):
    prj = make_project(env)
    answer(env, prj, drive_to_budget(env, prj))

    def boom(*a, **k):
        raise localisation.LocalisationError("kie.ai said no")

    monkeypatch.setattr(localisation, "transcribe_video", boom)
    settle(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    retry = pending(env, prj, "regeneration")[0]
    assert retry.name == "retry:r1" and "kie.ai said no" in retry.proposal["error"]
    monkeypatch.setattr(localisation, "transcribe_video", lambda *a, **k: transcript_v3())
    answer(env, prj, retry, option="retry")
    settle(env, prj, until=lambda: steps(env, prj)["transcribe"].status == "completed")
    jobs = q(env, ProjectTranscription, project_id=prj.pid)
    assert sorted(j.status for j in jobs) == ["failed", "succeeded"]


def test_interrupted_translation_is_never_replayed(env):
    prj = make_project(env, transcript=transcript_v3(), mute_source=True)
    answer(env, prj, drive_to_budget(env, prj))
    settle(env, prj, rounds=10, until=lambda: bool(env.rang), jobs=False)
    job = env.rang.pop(0)  # the worker never runs it: claim it, then "die"
    db = env.SF()
    row = db.get(LocalisationPromptJob, job.id)
    row.status, row.claim_token = "running", "dead-worker"
    db.commit()
    db.close()
    clock = env.hooks.now
    localisation_jobs.reap(session_factory=env.SF, is_live=lambda q_, j: False,
                           enqueue_fn=env.rang.append, now=clock())
    clock.advance(localisation_jobs.NOT_LIVE_GRACE_SEC + 1)
    out = localisation_jobs.reap(session_factory=env.SF, is_live=lambda q_, j: False,
                                 enqueue_fn=env.rang.append, now=clock())
    assert out["interrupted"] == 1 and env.rang == []
    settle(env, prj, rounds=5, until=lambda: bool(pending(env, prj, "regeneration")))
    retry = pending(env, prj, "regeneration")[0]
    assert retry.proposal["status"] == "interrupted"
    assert env.calls["translate"] == []  # nothing was ever sent twice


def test_captioned_source_is_handed_to_the_operator(env):
    env.calls["caption_ratio"] = 0.6
    prj = make_project(env)
    settle(env, prj, until=lambda: bool(pending(env, prj, "handoff")))
    a = pending(env, prj, "handoff")[0]
    assert a.name == "source_captions"
    answer(env, prj, a, option="stop")
    tick(env, prj, 2)
    ap = ap_row(env, prj)
    assert ap.status == "failed" and "subtitle removal" in ap.status_reason


def test_second_language_waits_for_the_first_probe(env):
    prj = make_project(env, languages=("es", "pt"))
    answer(env, prj, drive_to_budget(env, prj))
    settle(env, prj, until=lambda: len(runs(env, prj)) == 1)
    tick(env, prj, 4)
    assert len(runs(env, prj)) == 1  # pt's probe waits for es's acceptance
    assert sorted(env.calls["translate"]) == ["es", "pt"]  # translations run early
    st = steps(env, prj)
    assert "waiting until the es probe is accepted" in (st["probe:pt"].reason or "")


def test_hook_scope_project_is_refused_at_creation(env):
    with pytest.raises(Exception, match="localisation_scope=whole"):
        flow_mod.FLOW.validate_params(
            VideoProject(project_type="localisation", localisation_scope="hook"),
            {"languages": ["es"]})


@pytest.fixture()
def client(env):
    from fastapi.testclient import TestClient

    from app.db import get_db
    from app.main import app

    def _db():
        db = env.SF()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def test_assistant_localisation_project_from_the_api(env, client):
    base = {"gdrive_link": "https://drive.google.com/file/d/x", "project_type": "localisation",
            "mode": "assistant"}
    r = client.post("/api/v2/projects", data=dict(base, localisation_scope="whole",
                                                  autopilot_params='{"languages": ["PT", "es"]}'))
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"]
    db = env.SF()
    ap = db.execute(select(ProjectAutopilot).where(ProjectAutopilot.project_id == pid)).scalar_one()
    db.close()
    assert ap.flow_id == "localisation" and ap.params == {
        "languages": ["pt", "es"], "tails": {"pt": None, "es": None},
        "model": "seedance-2-5", "resolution": "720p"}
    r = client.post("/api/v2/projects", data=dict(base, localisation_scope="whole"))
    assert r.status_code == 422 and "languages is required" in r.text
    r = client.post("/api/v2/projects", data=dict(base, autopilot_params='{"languages": ["es"]}'))
    assert r.status_code == 422 and "localisation_scope=whole" in r.text


def _move_boundary(env, prj, index, delta):
    """Shift the boundary between segments *index* and *index+1* (a free manual edit)."""
    db = env.SF()
    a = db.execute(select(SegmentDef).where(SegmentDef.id == prj.seg_ids[index])).scalar_one()
    b = db.execute(select(SegmentDef).where(SegmentDef.id == prj.seg_ids[index + 1])).scalar_one()
    a.end_sec = b.start_sec = a.end_sec + delta
    db.commit()
    db.close()


def test_reopened_mid_flight_translation_is_a_regeneration(env, monkeypatch):
    """Terra §1 R1: a step reopened while its paid job ran must not re-buy it silently."""
    _keep_the_cut(monkeypatch)
    prj = make_project(env, transcript=transcript_v3(), mute_source=True)
    answer(env, prj, drive_to_budget(env, prj))
    settle(env, prj, rounds=10, until=lambda: bool(env.rang), jobs=False)
    env.rang.clear()  # the translate job stays queued (its worker is slow)
    assert len(q(env, LocalisationPromptJob, project_id=prj.pid)) == 1
    _move_boundary(env, prj, 2, 0.2)
    settle(env, prj, rounds=10, jobs=False,
           until=lambda: any(a.name == "launch#regeneration" for a in pending(env, prj)))
    a = next(a for a in pending(env, prj) if a.name == "launch#regeneration")
    assert a.step_id == "translate:es" and a.kind == "regeneration"
    assert len(q(env, LocalisationPromptJob, project_id=prj.pid)) == 1  # nothing bought twice


def test_manual_change_of_the_probe_gets_a_fresh_sheet(env):
    """Terra §1 R1: the re-asked listening approval shows the new take's sheet."""
    prj = make_project(env)
    probe, listen = drive_to_listen(env, prj)
    assert len(env.calls["sheets"]) == 1
    db = env.SF()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == probe.id)).scalar_one()
    new = env.tmp / "manual-take.mp4"
    new.write_bytes(os.urandom(96))
    rs.local_result_path = str(new)  # e.g. a manual restore of another take
    db.commit()
    db.close()
    answer(env, prj, listen)  # answered against the old take
    settle(env, prj, until=lambda: any(
        a.kind == "listening_probe" and a.name != listen.name for a in pending(env, prj)))
    assert len(env.calls["sheets"]) == 2
    assert len(runs(env, prj)) == 1  # the old answer did not unlock segments 2..N
    listen_b = next(a for a in pending(env, prj, "listening_probe"))
    answer(env, prj, listen_b)
    settle(env, prj, until=lambda: steps(env, prj)["probe:es"].status == "completed")
    old = [a for a in q(env, AutopilotApproval, autopilot_id=prj.apid) if a.name == listen.name]
    assert all(a.status == "superseded" for a in old)


def test_manual_take_after_a_rejection_is_listened_to_first(env):
    """Terra §1 R2: a rejection (and its pending fix) about take A is never
    applied to a take B the operator made by hand; B is asked for listening."""
    prj = make_project(env)
    probe, listen = drive_to_listen(env, prj)
    answer(env, prj, listen, "reject", comment="wrong voice")
    settle(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    fix = pending(env, prj, "regeneration")[0]
    db = env.SF()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == probe.id)).scalar_one()
    new = env.tmp / "take-b.mp4"
    new.write_bytes(os.urandom(80))
    rs.local_result_path = str(new)
    db.commit()
    db.close()
    answer(env, prj, fix, option="rerun")  # answered about take A
    settle(env, prj, until=lambda: any(
        a.kind == "listening_probe" and a.name != listen.name for a in pending(env, prj)))
    assert q(env, Run, id=probe.id)[0].status == RunStatus.done  # no rerun was launched
    assert ap_row(env, prj).status == "active"
    listen_b = next(a for a in pending(env, prj, "listening_probe") if a.name != listen.name)
    answer(env, prj, listen_b)
    settle(env, prj, until=lambda: steps(env, prj)["probe:es"].status == "completed")
    fixes = [a for a in q(env, AutopilotApproval, autopilot_id=prj.apid) if a.name == fix.name]
    assert fixes and all(a.status == "superseded" for a in fixes)
    assert q(env, Run, id=probe.id)[0].status == RunStatus.done


def _replace_take(env, run_id, name):
    db = env.SF()
    for rs in db.execute(select(RunSegment).where(RunSegment.run_id == run_id)).scalars():
        new = env.tmp / f"{name}-{rs.index}.mp4"
        new.write_bytes(os.urandom(72))
        rs.local_result_path = str(new)
    db.commit()
    db.close()


def test_stale_rejection_never_stops_the_autopilot(env):
    """Terra §1 R3: a rejection of take A, then a manual take B before the flow
    recorded it — waiting on B's listening must not hit the rejection grace."""
    prj = make_project(env)
    probe, listen = drive_to_listen(env, prj)
    answer(env, prj, listen, "reject", comment="about take A")
    _replace_take(env, probe.id, "take-b")  # before the next tick consumes the answer
    settle(env, prj, until=lambda: any(
        a.kind == "listening_probe" and a.name != listen.name for a in pending(env, prj)))
    old = [a for a in q(env, AutopilotApproval, autopilot_id=prj.apid) if a.name == listen.name]
    assert all(a.status == "superseded" for a in old)
    env.hooks.now.advance(settings.AUTOPILOT_REJECT_GRACE_SEC + 120)
    tick(env, prj, 2)
    assert ap_row(env, prj).status == "active"
    assert not [a for a in pending(env, prj, "regeneration")]  # A's cue was never applied to B


def test_rest_completes_after_a_manual_fix_of_its_failure(env):
    """Terra §1 R3: a failure approval answered after the operator fixed the run
    by hand is void; the step completes on the good run."""
    prj = make_project(env)
    probe, listen = drive_to_listen(env, prj)
    answer(env, prj, listen)
    settle(env, prj, until=lambda: len(runs(env, prj)) == 2)
    rest = runs(env, prj)[1]
    finish_run(env, rest.id, status=RunStatus.incomplete)
    settle(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    fail = pending(env, prj, "regeneration")[0]
    assert fail.name.startswith("fail:r1:") and fail.step_id == "rest:es"
    finish_run(env, rest.id)  # the operator's own rerun succeeded
    answer(env, prj, fail, option="rerun")
    settle(env, prj, until=lambda: steps(env, prj)["rest:es"].status == "completed")
    assert ap_row(env, prj).status in ("active", "completed")
    assert q(env, Run, id=rest.id)[0].status == RunStatus.done
    stale = [a for a in q(env, AutopilotApproval, autopilot_id=prj.apid) if a.name == fail.name]
    assert all(a.status == "superseded" for a in stale)


def test_stale_rejection_cleared_while_the_manual_take_is_running(env):
    """Terra §1 R4: the stale answer is cleared even while take B is in flight."""
    prj = make_project(env)
    probe, listen = drive_to_listen(env, prj)
    answer(env, prj, listen, "reject", comment="about take A")
    db = env.SF()
    db.get(Run, probe.id).status = RunStatus.processing  # the operator's rerun runs
    db.commit()
    db.close()
    tick(env, prj, 3)
    old = [a for a in q(env, AutopilotApproval, autopilot_id=prj.apid) if a.name == listen.name]
    assert all(a.status == "superseded" for a in old)
    env.hooks.now.advance(settings.AUTOPILOT_REJECT_GRACE_SEC + 120)
    tick(env, prj, 2)
    assert ap_row(env, prj).status == "active"


def test_acknowledgement_lists_current_answers_too_and_consumes_none(env):
    """Terra §1 R4: a stale and a current answer at once: the stale one is
    superseded, the current one survives and is acted on."""
    prj = make_project(env)
    probe, listen_a = drive_to_listen(env, prj)
    _replace_take(env, probe.id, "take-b")
    settle(env, prj, until=lambda: any(
        a.kind == "listening_probe" and a.name != listen_a.name for a in pending(env, prj)))
    listen_b = next(a for a in pending(env, prj, "listening_probe") if a.name != listen_a.name)
    # 2026-11-03 (stale cards, docs/segment-fill-batch-design.md §5): the core
    # already superseded the stale card on its last decision, so it can no
    # longer be answered (before: it was answered and superseded at consumption).
    with pytest.raises(core.OperatorError, match="superseded"):
        answer(env, prj, listen_a)
    answer(env, prj, listen_b)
    settle(env, prj, until=lambda: steps(env, prj)["probe:es"].status == "completed")
    rows = {a.name: a.status for a in q(env, AutopilotApproval, autopilot_id=prj.apid)
            if a.kind == "listening_probe"}
    assert rows[listen_a.name] == "superseded" and rows[listen_b.name] == "consumed"
    assert ap_row(env, prj).status == "active"


def test_manual_transcription_after_our_failure_is_used(env, monkeypatch):
    """Terra §1 R5: a ready transcript made by hand after our job failed completes
    the step; no second paid transcription is asked."""
    prj = make_project(env)
    answer(env, prj, drive_to_budget(env, prj))

    def boom(*a, **k):
        raise localisation.LocalisationError("down")

    monkeypatch.setattr(localisation, "transcribe_video", boom)
    settle(env, prj, until=lambda: bool(pending(env, prj, "regeneration")))
    db = env.SF()
    p = db.get(VideoProject, prj.pid)
    p.transcript, p.transcript_status, p.transcript_error = transcript_v3(), "ready", None
    db.commit()
    db.close()
    settle(env, prj, until=lambda: steps(env, prj)["transcribe"].status == "completed")
    assert len(q(env, ProjectTranscription, project_id=prj.pid)) == 1
