"""AICP-364: the subtitle-removal flow on the REAL autopilot core.

Temp SQLite DB, the core's tick / APPLY / launch protocol, the real endpoints
behind every handler (``update_project_segments``, ``create_run``,
``create_fill_preview``, ``admit_start`` + ``pipeline_provider.admit``), the
real computes run through ``app.autopilot.compute.run``. Mocked: the caption
detector scans, ffprobe, the sheets, face samples, every enqueue and the Fal
worker liveness -- no provider call, no video decoded.
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
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

import app.api_v2 as api_v2_module
import app.tasks as tasks_mod
from app import pipeline_provider as pp
from app import qa_tools, storage
from app import subtitle_loop as sl
from app.autopilot import compute as compute_mod
from app.autopilot import core, registry
from app.autopilot.flows import subtitle_compute, subtitle_handlers
from app.autopilot.flows import subtitle_removal as flow_mod
from app.autopilot.flows import subtitle_removal_plan as P
from app.autopilot.judgement import FakePort
from app.config import settings
from app.db import Base
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.models_autopilot import (
    AutopilotApproval, AutopilotEffect, AutopilotGate, ProjectAutopilot, ProjectSpendScope,
    utcnow,
)
from app.provider_models import ProviderTask
from app.qa_tools import CaptionFrame, CaptionScan
from app.segment_fill_models import RunSegmentFillJob
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus

FPS = 10.0
DUR = 12.0
END_CARD = (0.20, 0.49, 0.47, 0.53)


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
        self.t = self.t + timedelta(seconds=sec)


def _word_box(i):
    w = 0.10 + 0.04 * (i % 3)
    return (0.5 - w, 0.60, 0.5 + w, 0.64)


def _captions(a, b):
    return {round(i / FPS, 3): _word_box(i // 5) for i in range(int(a * FPS), int(b * FPS))}


def _static(a, b):
    return {round(i / FPS, 3): END_CARD for i in range(int(a * FPS), int(b * FPS))}


def _scan(boxes):
    frames = [CaptionFrame(t=round(i / FPS, 3), score=200 if boxes.get(round(i / FPS, 3)) else 0,
                           box=boxes.get(round(i / FPS, 3))) for i in range(int(DUR * FPS))]
    hits = sum(1 for f in frames if f.box)
    return CaptionScan(path="x", fps=FPS, threshold=40, frames=len(frames), hits=hits,
                       hit_ratio=round(hits / len(frames), 4), spans=[], scores=frames)


def _touch(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


@pytest.fixture()
def world(SF, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTOPILOT_ENABLED", True)
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(core, "wake", lambda: None)
    calls = {"process": [], "prepare": [], "fal": []}
    for name in ("enqueue_analyze_project", "enqueue_preprocess_run", "enqueue_transcribe_project"):
        monkeypatch.setattr(api_v2_module, name, lambda *a, **k: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid, *a, **k:
                        calls["process"].append(rid))
    monkeypatch.setattr(tasks_mod, "enqueue_segment_fill_prepare",
                        lambda j: calls["prepare"].append(j))
    monkeypatch.setattr(tasks_mod, "enqueue_fal_job", lambda fn, t, **kw: calls["fal"].append(t))
    monkeypatch.setattr(pp.settings, "FAL_ENABLED", True)
    monkeypatch.setattr(pp.settings, "FAL_API_KEY", "test-key")
    monkeypatch.setattr(pp, "_fal_worker_alive", lambda: True)
    monkeypatch.setattr(pp, "_probe_duration", lambda path: 4.0)
    # Detector + sheets (files are written: artifacts are copied by the core)
    scans = {}

    class Info:
        def __init__(self):
            self.duration_sec, self.fps = DUR, 30.0

    monkeypatch.setattr(qa_tools, "scan_captions", lambda path, **k: scans[path])
    monkeypatch.setattr(qa_tools, "locate_captions", lambda path, **k: {
        "state": "none", "candidate_band_y": None, "positions": [], "members": [],
        "competing": [], "reasons": [], "frames": 24, "fps": 2.0, "detector": {}})
    monkeypatch.setattr(qa_tools, "_probe", lambda p: Info())
    monkeypatch.setattr(qa_tools, "contact_sheet", lambda src, out, **k: {"out": _touch(out)})
    monkeypatch.setattr(qa_tools, "comparison_sheet", lambda s, f, out, **k: {"out": _touch(out)})
    monkeypatch.setattr(qa_tools, "alignment_check", lambda s, r, **k: {
        "in_sync": True, "points": 20, "max_abs_window_median": 0.0, "frame_count_delta": 0,
        "length_ok": True, "windows": []})
    monkeypatch.setattr(sl, "_crop_sheet", lambda s, f, spans, out: {
        "out": _touch(out), "times": [t for t, _ in sl._word_moments(spans)]})
    monkeypatch.setattr(sl, "_fill_sheet", lambda s, f, times, box, out: {
        "out": _touch(out), "times": list(times)})
    monkeypatch.setattr(subtitle_compute, "_face_samples",
                        lambda src: [(t + 0.5, 1 if t < 8 else 0) for t in range(12)])
    monkeypatch.setattr(subtitle_compute, "fill_clip",
                        lambda src, fin, a, b, box, out: _touch(out))
    # Face evidence for the auto-accept (2026-11-03): none by default (every
    # composite is doubtful); a test sets w["faces"] to {index: evidence}.
    faces, screen = {}, {}
    monkeypatch.setattr(subtitle_compute, "fill_faces",
                        lambda src, segs, rows: {str(k): v for k, v in faces.items()})
    monkeypatch.setattr(subtitle_compute, "fill_screen",
                        lambda src, segs, rows: {str(k): v for k, v in screen.items()})
    monkeypatch.setattr(subtitle_compute, "SESSION_FACTORY", SF)
    registry.register(flow_mod.FLOW)

    db = SF()
    if db.get(AutopilotGate, 1) is None:
        db.add(AutopilotGate(id=1, held=False, generation=0))
    db.commit()
    p = VideoProject(source_type="upload", source_ref="s.mp4", source_local_path="/pending",
                     status=ProjectStatus.ready, project_type="subtitle_removal",
                     mode="assistant", duration_sec=DUR)
    db.add(p)
    db.flush()
    src = _touch(os.path.join(storage.project_dir(p.id), "source.mp4"))
    p.source_local_path = src
    for i in range(3):
        db.add(SegmentDef(project_id=p.id, index=i, start_sec=4.0 * i, end_sec=4.0 * (i + 1),
                          action="swap"))
    ap = core.create_autopilot(db, p, flow_mod.FLOW, {})
    db.commit()
    w = {"pid": p.id, "apid": ap.id, "src": src, "scans": scans, "calls": calls, "SF": SF,
         "hooks": None, "faces": faces, "screen": screen}
    db.close()
    boxes = _captions(0.0, 8.0)
    boxes.update(_static(8.2, DUR))                 # segment 2: an end card
    scans[src] = _scan(boxes)
    enq = []
    w["hooks"] = core.Hooks(enqueue_compute=lambda eid, jid: enq.append(eid),
                            compute_is_live=lambda eff: True, port=FakePort(), now=Clock())
    w["enq"] = enq
    return w


def _tick(w, sec=301):
    w["hooks"].now.advance(sec)
    return core.tick_autopilot(w["apid"], session_factory=w["SF"], hooks=w["hooks"])


def _run_computes(w):
    while w["enq"]:
        eid = w["enq"].pop(0)
        assert compute_mod.run(eid, session_factory=w["SF"]) == "succeeded", _effect(w, eid).error


def _effect(w, eid):
    db = w["SF"]()
    try:
        return db.get(AutopilotEffect, eid)
    finally:
        db.close()


def _effects(w, handler=None):
    db = w["SF"]()
    try:
        q = select(AutopilotEffect).where(AutopilotEffect.autopilot_id == w["apid"])
        if handler:
            q = q.where(AutopilotEffect.handler == handler)
        return db.execute(q.order_by(AutopilotEffect.created_at)).scalars().all()
    finally:
        db.close()


def _pending(w):
    db = w["SF"]()
    try:
        return db.execute(select(AutopilotApproval).where(
            AutopilotApproval.autopilot_id == w["apid"],
            AutopilotApproval.status == "pending")).scalars().all()
    finally:
        db.close()


def _answer(w, a, decision="approve", option=None):
    db = w["SF"]()
    try:
        core.answer(db, w["pid"], a.id, decision, option=option)
    finally:
        db.close()


def _ap(w):
    db = w["SF"]()
    try:
        return db.get(ProjectAutopilot, w["apid"])
    finally:
        db.close()


def _until(w, pred, n=12):
    for _ in range(n):
        _run_computes(w)
        got = pred()
        if got:
            return got
        _tick(w)
    _run_computes(w)
    got = pred()
    assert got, f"not reached; autopilot {_ap(w).status}: {_ap(w).status_reason}"
    return got


def _one_pending(w, kind=None, prefix=None):
    def pred():
        rows = [a for a in _pending(w) if (kind is None or a.kind == kind)
                and (prefix is None or a.name.startswith(prefix))]
        return rows[0] if rows else None
    return pred


def _finish_run(w, rid, *, leftover: bool):
    """The pipeline's job: segments 0-1 generated, stitched, delivered."""
    db = w["SF"]()
    run = db.get(Run, rid)
    defs = db.execute(select(SegmentDef).where(SegmentDef.project_id == w["pid"])
                      .order_by(SegmentDef.index)).scalars().all()
    rdir = storage.run_results_dir(rid, w["pid"])
    for sd in defs:
        rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid,
                                                 RunSegment.segment_def_id == sd.id)).scalar()
        if rs is None:
            rs = RunSegment(run_id=rid, segment_def_id=sd.id, index=sd.index)
            db.add(rs)
        rs.status = SegmentStatus.completed
        rs.local_result_path = _touch(os.path.join(rdir, f"result_{sd.index:04d}.mp4"),
                                      f"omni-{sd.index}".encode())
        rs.seedance_task_id = f"task{sd.index}" if sd.action == "swap" else None
        rs.cost_credits = 168.0 if sd.action == "swap" else 0.0
    final = os.path.join(storage.run_dir(rid, w["pid"]), "results", "final.mp4")
    _touch(final, b"final-0")
    run.result_local_path, run.result_gdrive_file_id = final, "DRIVE1"
    run.segment_snapshot = [{"id": sd.id, "index": sd.index, "action": sd.action,
                             "start_sec": sd.start_sec, "end_sec": sd.end_sec,
                             "pre_roll_sec": 0.0, "post_roll_sec": 0.0} for sd in defs]
    run.status = RunStatus.done
    db.commit()
    db.close()
    fin = _captions(5.0, 5.9) if leftover else {}
    fin.update(_static(8.2, DUR))
    w["scans"][final] = _scan(fin)
    return final


def _set_allowance(w, usd):
    db = w["SF"]()
    db.get(VideoProject, w["pid"]).void_auto_usd_per_run = usd
    db.commit()
    db.close()


@pytest.mark.parametrize("allowance", [None, 0.0])
def test_subtitle_flow_end_to_end_on_the_real_core(world, allowance):
    """Default allowance ($1): the VOID batch starts WITHOUT a question; an
    allowance of $0: the ONE batch question, approved, raises the budget (Q6)."""
    w = world
    SF = w["SF"]
    _set_allowance(w, allowance)

    # ---- segmentation: review compute -> eye check -> keep patch ------------
    seg_ask = _until(w, _one_pending(w, "handoff", "segmentation:"))
    assert [k["index"] for k in seg_ask.proposal["keep"]] == [2]
    assert seg_ask.artifacts and seg_ask.artifacts[0]["producer"].startswith("effect:")
    assert {o["id"] for o in seg_ask.options} == {"apply", "as_is"}
    _answer(w, seg_ask, option="apply")
    def seg2_keep():
        db = SF()
        try:
            return db.execute(select(SegmentDef.action).where(
                SegmentDef.project_id == w["pid"], SegmentDef.index == 2)).scalar() == "keep"
        finally:
            db.close()
    _until(w, seg2_keep)

    # ---- launch: the core's budget approval, then one Omni run --------------
    budget = _until(w, _one_pending(w, "budget"))
    lines = budget.proposal["lines"]
    void = [x for x in lines if x["label"].startswith("VOID")]
    if allowance is None:
        # the allowance is a real line of the budget: min($1, 2 swap x $0.10)
        assert "VOID rebuilds without a question: up to $0.20" in void[0]["label"]
        assert void[0]["credits"] == 40.0
    else:
        assert "each batch is approved on its own" in void[0]["label"]
        assert void[0]["credits"] == 0.0
    assert sum(x["credits"] for x in lines if x["step_id"] == "launch") == 2 * 4 * 42.0
    _answer(w, budget)

    def one_run():
        db = SF()
        try:
            return db.execute(select(Run).where(Run.project_id == w["pid"])).scalars().all()
        finally:
            db.close()
    runs = _until(w, one_run)
    assert len(runs) == 1 and runs[0].model == "gemini-omni" and runs[0].audio_mode == "original"
    rid = runs[0].id
    _tick(w)
    assert len(one_run()) == 1                                   # admitted exactly once

    # ---- clean: QA -> free preview -> the batch start ------------------------
    _finish_run(w, rid, leftover=True)
    jobs = _until(w, lambda: [j for j in _fill_jobs(SF, rid)])
    assert len(jobs) == 1 and jobs[0].status == "preparing" and w["calls"]["prepare"] == [jobs[0].id]
    assert jobs[0].base["bind"] == "segment"
    (prev_eff,) = _effects(w, P.H_FILL_PREVIEW)
    assert prev_eff.state == "launched" and prev_eff.created_refs["fill_jobs"] == [jobs[0].id]
    fid = jobs[0].id
    _prepared(SF, fid)
    budget_before = _ap(w).approved_budget_mcredits
    if allowance == 0.0:
        ask = _until(w, _one_pending(w, "regeneration", "vbatch:"))
        opt = ask.options[0]
        assert opt["id"] == "start" and opt["action"]["handler"] == P.H_FILL_BATCH_START
        assert opt["action"]["params"]["fill_ids"] == [fid] and opt["cost_mcredits"] == 20_000
        assert [o["id"] for o in ask.options][1:] == ["per_segment", "accept", "manual"]
        assert all(o["action"] is None for o in ask.options[1:])
        assert ask.artifacts and ask.artifacts[0]["producer"].startswith("effect:")  # the sheet
        _answer(w, ask, option="start")

    def admitted():
        db = SF()
        try:
            j = db.get(RunSegmentFillJob, fid)
            return j if j.status == "admitted" else None
        finally:
            db.close()
    job = _until(w, admitted)
    (eff,) = _effects(w, P.H_FILL_BATCH_START)
    assert eff.state == "launched" and eff.paid
    assert eff.name.startswith("autovoid:" if allowance is None else "do:vbatch:")
    assert eff.params["authorization"] == ("auto" if allowance is None else "operator")
    assert eff.created_refs == {"fill_batches": [job.batch_id], "fill_jobs": [fid],
                                "provider_tasks": [job.provider_task_id]}
    assert w["calls"]["fal"] == [job.provider_task_id]
    db = SF()
    task = db.get(ProviderTask, job.provider_task_id)
    assert task.owner_kind == "segment_fill" and task.reserved_mcredits == 20_000
    assert db.get(ProjectSpendScope, (w["pid"], f"fal:{task.id}")) is not None
    from app.spend import KieSpendReservation
    res = db.execute(select(KieSpendReservation).where(
        KieSpendReservation.scope_key == f"fal:{task.id}")).scalar_one()
    assert res.reserved_mcredits == 20_000                       # the ledger holds it
    from app.segment_fill_models import RunSegmentFillBatch
    batch = db.get(RunSegmentFillBatch, job.batch_id)
    assert batch.authorization_kind == ("auto" if allowance is None else "operator")
    assert batch.reserved_usd == 0.10
    db.close()
    if allowance is None:
        assert _ap(w).approved_budget_mcredits == budget_before          # inside the budget
        assert not [a for a in _all_approvals(w) if a.kind == "regeneration"]
    else:
        assert _ap(w).approved_budget_mcredits == budget_before + 20_000  # Q6

    # ---- the composite is applied and re-delivered: ONE eye-check card ------
    _apply_fill(w, rid, fid)
    eye = _until(w, _one_pending(w, "handoff", "eyes:"))
    assert eye.proposal["members"] == [fid]
    labels = [a["label"] for a in eye.artifacts]
    assert any("inside the repainted area" in x for x in labels) and any(
        "delivered | original" in x for x in labels)
    assert all(o["action"] is None for o in eye.options)
    assert [o["id"] for o in eye.options] == ["accept_all", "per_segment", "manual"]
    _answer(w, eye, option="accept_all")

    # ---- hand-off -----------------------------------------------------------
    hand = _until(w, _one_pending(w, "handoff", "handoff"))
    assert hand.proposal["to_project"]["form"]["project_type"] == "localisation"
    assert hand.proposal["keep_ranges"] == [{"start_sec": 8.0, "end_sec": 12.0}]
    assert [a["kind"] for a in hand.artifacts] == ["video", "image"]
    _answer(w, hand)
    _until(w, lambda: _ap(w).status == "completed")
    assert not any(e.handler == P.H_RERUN for e in _effects(w))
    assert len(one_run()) == 1


def test_a_clean_composite_without_a_face_under_its_mask_needs_no_eye_check(world):
    """Auto-accept (design §4.1): detector-clean, delivered, a face sampled far
    from the mask box -> no eye-check card; the hand-off names the segment
    and shows its fill sheet."""
    w = world
    SF = w["SF"]
    rid = _launch_run(w)
    _finish_run(w, rid, leftover=True)
    jobs = _until(w, lambda: [j for j in _fill_jobs(SF, rid)])
    fid = jobs[0].id
    _prepared(SF, fid)
    _until(w, lambda: (lambda j: j if j.status == "admitted" else None)(
        _fill_jobs(SF, rid)[0]))
    w["faces"][1] = {"samples": 8, "boxes": [[0.35, 0.10, 0.62, 0.40]], "fps": 2.0}
    w["screen"][1] = {"samples": 8, "screen_frames": 0, "max_share": 0.1}
    _apply_fill(w, rid, fid)
    hand = _until(w, _one_pending(w, "handoff", "handoff"))
    assert not [a for a in _all_approvals(w) if a.name.startswith(("eyes:", "eye:"))]
    assert "accepted it without an eye check" in hand.question
    assert any("accepted without an eye check" in a["label"] for a in hand.artifacts)


def _all_approvals(w):
    db = w["SF"]()
    try:
        return db.execute(select(AutopilotApproval).where(
            AutopilotApproval.autopilot_id == w["apid"])).scalars().all()
    finally:
        db.close()


def _fill_jobs(SF, rid):
    db = SF()
    try:
        return db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.run_id == rid)
                          ).scalars().all()
    finally:
        db.close()


def _prepared(SF, fid):
    """What segment_fill_prepare publishes (its own tests cover it on video)."""
    from app import pipeline_segment_fill as psf

    db = SF()
    job = db.get(RunSegmentFillJob, fid)
    work = os.path.join(psf._job_dir(job), "prep")
    _touch(os.path.join(work, "void_v.mp4"), b"v" * 64)
    _touch(os.path.join(work, "void_m.mp4"), b"m" * 64)
    job.status, job.prep_dir = "ready", work
    job.preview_path = _touch(os.path.join(work, "preview.jpg"))
    job.media = {"frames": 120, "duration_sec": 4.0, "frames_masked": 40, "void_frames": 45}
    job.mask_box, job.box_source, job.estimated_usd = [0.29, 0.58, 0.71, 0.66], "measured", 0.10
    db.commit()
    db.close()


def _apply_fill(w, rid, fid):
    """What promote + the re-stitch + delivery do (covered by their own tests)."""
    db = w["SF"]()
    job = db.get(RunSegmentFillJob, fid)
    rs = db.get(RunSegment, job.run_segment_id)
    comp = _touch(os.path.join(storage.run_dir(rid, w["pid"]), "segment_fills", fid,
                               "composite.mp4"), b"composite")
    job.status, job.composite_path, job.applied_at = "applied", comp, utcnow()
    rs.local_result_path, rs.seedance_task_id = comp, None
    if job.batch_id:
        from app.segment_fill_models import RunSegmentFillBatch
        db.get(RunSegmentFillBatch, job.batch_id).status = "applied"
    run = db.get(Run, rid)
    _touch(run.result_local_path, b"final-1-with-the-fill")
    db.commit()
    db.close()
    w["scans"][run.result_local_path] = _scan(_static(8.2, DUR))


# ---------------------------------------------------------------------------
# Handler conformance: refusals write nothing (the core's marker pattern)
# ---------------------------------------------------------------------------

def _marker(db):
    m = AutopilotGate(id=777, held=False, generation=0)    # stands in for the effect row
    db.add(m)
    db.flush()


def _marker_gone(SF):
    db = SF()
    try:
        return db.get(AutopilotGate, 777) is None
    finally:
        db.close()


def _run(SF, pid, status):
    db = SF()
    run = Run(project_id=pid, prompt="p", status=status, reference_image_urls=[],
              model="gemini-omni", audio_mode="original")
    db.add(run)
    db.commit()
    rid = run.id
    db.close()
    return rid


@pytest.mark.parametrize("status", [RunStatus.queued, RunStatus.processing, RunStatus.done])
def test_retry_handlers_refuse_non_terminal_or_done_runs_and_write_nothing(world, status):
    from app.autopilot.contract import ActionRefused

    SF = world["SF"]
    rid = _run(SF, world["pid"], status)
    for h in (subtitle_handlers.RetryRun(), subtitle_handlers.RetryDelivery()):
        db = SF()
        project = db.get(VideoProject, world["pid"])
        p = h.validate(db, project, {"run_id": rid})
        _marker(db)
        with pytest.raises(ActionRefused) as exc:
            h.execute(db, project, p, effect=None)
        assert exc.value.status == 409
        db.rollback()
        db.close()
        assert _marker_gone(SF)


def test_retry_delivery_refuses_a_retry_that_would_bill(world):
    from app.autopilot.contract import ActionRefused

    SF = world["SF"]
    rid = _run(SF, world["pid"], RunStatus.failed)       # no segment has a result yet
    db = SF()
    project = db.get(VideoProject, world["pid"])
    h = subtitle_handlers.RetryDelivery()
    p = h.validate(db, project, {"run_id": rid})
    _marker(db)
    with pytest.raises(ActionRefused) as exc:
        h.execute(db, project, p, effect=None)
    assert "would bill" in exc.value.detail
    db.rollback()
    db.close()
    assert _marker_gone(SF)
    assert subtitle_handlers.RetryRun.paid and subtitle_handlers.RetryRun.always_regeneration
    assert not subtitle_handlers.RetryDelivery.paid


def test_captions_outside_the_default_band_are_located_and_saved_by_the_flow(world, monkeypatch):
    """The 514067f2 case on the real core: captions at y 0.76-0.80, no band on
    the project -> the review locates them, the flow saves the band itself
    (no ask) and reviews again on it; the eye check that follows is the
    ordinary one (the end card keep)."""
    w = world
    low = {t: (b[0], 0.76, b[2], 0.80) for t, b in _captions(0.0, 8.0).items()}
    low.update(_static(8.2, DUR))
    empty = _scan({})

    def scan(path, **k):
        if path != w["src"]:
            return w["scans"][path]
        lo, hi = qa_tools.caption_config(k.get("config")).band_y
        return _scan(low) if lo <= 0.78 <= hi else empty

    loc = {"state": "located", "candidate_band_y": [0.70, 0.86], "frames": 24, "fps": 2.0,
           "members": [0], "competing": [], "reasons": [], "lines_outside_band": 0.0,
           "detector": {}, "positions": [{
               "y_range": [0.76, 0.80], "support": 0.6, "nonstatic_frames": 14,
               "static_frames": 0, "caption_like": True, "windows_hit": 4, "y0_p5": 0.76,
               "y1_p95": 0.80, "band_y": [0.70, 0.86], "sample_times": [1.0, 3.0, 5.0, 7.0]}]}
    located = []
    monkeypatch.setattr(qa_tools, "scan_captions", scan)
    monkeypatch.setattr(qa_tools, "locate_captions",
                        lambda path, **k: located.append(path) or dict(loc))
    seg_ask = _until(w, _one_pending(w, "handoff", "segmentation:"))
    db = w["SF"]()
    try:
        assert db.get(VideoProject, w["pid"]).caption_detector == {"band_y": [0.7, 0.86]}
    finally:
        db.close()
    patches = _effects(w, P.H_PATCH_DETECTOR)
    assert [e.name.split(":")[0] for e in patches] == ["locate"] and patches[0].approval_id is None
    assert located == [w["src"]]                      # the second review used the saved band
    assert [k["index"] for k in seg_ask.proposal["keep"]] == [2]
    assert not any(a.name.startswith("locate:") for a in _pending(w))


def test_patch_caption_detector_is_fenced_on_the_config_it_was_planned_on(world):
    from app.autopilot.contract import ActionRefused

    SF = world["SF"]
    db = SF()
    project = db.get(VideoProject, world["pid"])
    h = subtitle_handlers.PatchCaptionDetector()
    p = h.validate(db, project, {"caption_detector": {"band_y": [0.41, 0.67]},
                                 "expect_detector_sha": P.detector_sha({"other": 1})})
    _marker(db)
    with pytest.raises(ActionRefused):
        h.execute(db, project, p, effect=None)
    db.rollback()
    db.close()
    assert _marker_gone(SF)
    db = SF()
    project = db.get(VideoProject, world["pid"])
    p = h.validate(db, project, {"caption_detector": {"band_y": [0.41, 0.67]},
                                 "expect_detector_sha": P.detector_sha(None)})
    h.execute(db, project, p, effect=None)
    db.close()
    db = SF()
    assert db.get(VideoProject, world["pid"]).caption_detector == {"band_y": [0.41, 0.67]}
    db.close()


def test_patch_caption_detector_never_replaces_a_band_of_the_projects_own(world):
    """A located band (expect_no_band) is refused once the project has a band,
    even when the digest the flow planned on still matches."""
    from app.autopilot.contract import ActionInvalid, ActionRefused

    SF = world["SF"]
    db = SF()
    project = db.get(VideoProject, world["pid"])
    project.caption_detector = {"band_y": [0.5, 0.72]}           # explicit, = the default
    db.commit()
    h = subtitle_handlers.PatchCaptionDetector()
    with pytest.raises(ActionInvalid):
        h.validate(db, project, {"caption_detector": {"band_y": [0.7, 0.86]},
                                 "expect_detector_sha": "x", "expect_no_band": "yes"})
    p = h.validate(db, project, {"caption_detector": {"band_y": [0.7, 0.86]},
                                 "expect_detector_sha": P.detector_sha({"band_y": [0.5, 0.72]}),
                                 "expect_no_band": True})
    with pytest.raises(ActionRefused) as exc:
        h.execute(db, project, p, effect=None)
    assert "own caption band" in str(exc.value.detail)
    db.rollback()
    db.close()
    db = SF()
    assert db.get(VideoProject, world["pid"]).caption_detector == {"band_y": [0.5, 0.72]}
    db.close()


def test_fill_start_refusal_paths_write_nothing(world):
    """A stale preview (and a fill that is not ready) is refused with nothing
    written -- no canceled job, no provider task, no reservation."""
    from app.autopilot.contract import ActionRefused
    from fastapi import HTTPException

    SF = world["SF"]
    rid = _run(SF, world["pid"], RunStatus.done)
    _finish_run(world, rid, leftover=True)
    db = SF()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid,
                                             RunSegment.index == 1)).scalar()
    project = db.get(VideoProject, world["pid"])
    from app import pipeline_segment_fill as psf

    job = RunSegmentFillJob(run_id=rid, run_segment_id=rs.id, project_id=world["pid"],
                            status="preparing", model="void", prompt="",
                            base=psf.capture_base(db, db.get(Run, rid), rs, project))
    db.add(job)
    db.commit()
    fid = job.id
    db.close()
    h = subtitle_handlers.SegmentFillStart()
    for setup, expect in (("not_ready", HTTPException), ("stale", ActionRefused)):
        if setup == "stale":
            _prepared(SF, fid)
            db = SF()
            r = db.get(RunSegment, rs.id)
            _touch(r.local_result_path, b"a different omni output")   # the base moved
            db.close()
        db = SF()
        project = db.get(VideoProject, world["pid"])
        p = h.validate(db, project, {"run_id": rid, "fill_id": fid, "approved_usd": 0.1})
        _marker(db)
        with pytest.raises(expect) as exc:
            h.execute(db, project, p, effect=None)
        status = getattr(exc.value, "status", None) or exc.value.status_code
        assert status == 409
        db.rollback()
        db.close()
        assert _marker_gone(SF)
        db = SF()
        j = db.get(RunSegmentFillJob, fid)
        assert j.status in ("preparing", "ready") and j.provider_task_id is None
        assert db.execute(select(ProviderTask).where(ProviderTask.owner_id == fid)
                          ).scalars().all() == []
        db.close()


def test_preview_handler_records_its_job_in_the_admission(world):
    SF = world["SF"]
    rid = _run(SF, world["pid"], RunStatus.done)
    _finish_run(world, rid, leftover=True)
    db = SF()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid,
                                             RunSegment.index == 1)).scalar()
    project = db.get(VideoProject, world["pid"])
    h = subtitle_handlers.SegmentFillPreview()
    p = h.validate(db, project, {"run_id": rid, "run_segment_id": rs.id})

    class Eff:
        created_refs = {}

    eff = Eff()
    out = h.execute(db, project, p, eff)
    db.close()
    assert eff.created_refs == {"fill_jobs": [out["fill_id"]]}
    assert world["calls"]["prepare"] == [out["fill_id"]]
    assert h.reconcile(None, eff) == "launched"


def test_single_commit_handlers_reconcile_from_their_refs():
    class Eff:
        def __init__(self, refs):
            self.created_refs = refs

    start = subtitle_handlers.SegmentFillStart()
    assert start.paid and start.always_regeneration
    assert start.reconcile(None, Eff({"fill_jobs": ["f"], "provider_tasks": ["t"]})) == "launched"
    assert start.reconcile(None, Eff({})) == "unknown"
    assert subtitle_handlers.SegmentFillPreview().reconcile(None, Eff({})) == "unknown"
    batch = subtitle_handlers.SegmentFillBatchStart()
    assert batch.paid and not batch.always_regeneration      # the allowance decides (§3)
    assert batch.reconcile(None, Eff({"fill_batches": ["b"]})) == "launched"
    assert batch.reconcile(None, Eff({})) == "unknown"
    names = {h.name for h in subtitle_handlers.HANDLERS}
    assert names == {P.H_FILL_PREVIEW, P.H_FILL_CANCEL, P.H_FILL_START, P.H_FILL_APPLY_SOURCE,
                     P.H_RETRY_RUN, P.H_RETRY_DELIVERY, P.H_RESTITCH, P.H_PATCH_DETECTOR,
                     P.H_FILL_BATCH_START}
    apply_src = subtitle_handlers.SegmentFillApplySource()
    assert not apply_src.paid and not apply_src.always_regeneration
    assert apply_src.reconcile(None, Eff({})) == "unknown"
    assert all(registry.get_handler(n) is not None for n in names)
    assert registry.get_compute(P.C_QA) is subtitle_compute.subtitle_qa
    assert registry.flow_for_project_type("subtitle_removal") is flow_mod.FLOW


# ---------------------------------------------------------------------------
# A segment the provider refused (514067f2 / d8105ab5): the pending retry-only
# ask of an autopilot already running is REPLACED by the VOID-from-source
# proposal on the next ticks -- no restart.
# ---------------------------------------------------------------------------

SAFETY = ("failed after 3 attempt(s): Request blocked: The input content was flagged by safety "
          "filters as potentially dangerous and cannot be processed.")


def _launch_run(w):
    SF = w["SF"]
    seg_ask = _until(w, _one_pending(w, "handoff", "segmentation:"))
    _answer(w, seg_ask, option="apply")
    budget = _until(w, _one_pending(w, "budget"))
    _answer(w, budget)

    def one_run():
        db = SF()
        try:
            return db.execute(select(Run).where(Run.project_id == w["pid"])).scalars().all()
        finally:
            db.close()
    return _until(w, one_run)[0].id


def _incomplete_run(w, rid, error):
    """The pipeline's job: segment 0 generated, segment 1 failed -> incomplete."""
    db = w["SF"]()
    run = db.get(Run, rid)
    defs = db.execute(select(SegmentDef).where(SegmentDef.project_id == w["pid"])
                      .order_by(SegmentDef.index)).scalars().all()
    rdir = storage.run_results_dir(rid, w["pid"])
    for sd in defs:
        rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid,
                                                 RunSegment.segment_def_id == sd.id)).scalar()
        if rs is None:
            rs = RunSegment(run_id=rid, segment_def_id=sd.id, index=sd.index)
            db.add(rs)
        if sd.index == 1:
            rs.status, rs.error_message, rs.seedance_task_id = SegmentStatus.failed, error, "blk1"
        else:
            rs.status = SegmentStatus.completed
            rs.local_result_path = _touch(os.path.join(rdir, f"result_{sd.index:04d}.mp4"))
            rs.seedance_task_id = f"task{sd.index}" if sd.action == "swap" else None
    run.segment_snapshot = [{"id": sd.id, "index": sd.index, "action": sd.action,
                             "start_sec": sd.start_sec, "end_sec": sd.end_sec,
                             "pre_roll_sec": 0.0, "post_roll_sec": 0.0} for sd in defs]
    run.status, run.error_message = RunStatus.incomplete, "1 swap segment(s) did not complete: [1]"
    db.commit()
    db.close()


def _set_error(w, rid, error):
    db = w["SF"]()
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid,
                                             RunSegment.index == 1)).scalar()
    rs.error_message = error
    db.commit()
    db.close()


def _prepared_source(SF, fid, *, as_is=False):
    from app import pipeline_segment_fill as psf

    db = SF()
    job = db.get(RunSegmentFillJob, fid)
    work = os.path.join(psf._job_dir(job), "prep")
    if not as_is:
        _touch(os.path.join(work, "void_v.mp4"), b"v" * 64)
        _touch(os.path.join(work, "void_m.mp4"), b"m" * 64)
    _touch(os.path.join(work, "clip.mp4"), b"clip")
    job.status, job.prep_dir = "ready", work
    job.preview_path = _touch(os.path.join(work, "preview.jpg"))
    if as_is:
        job.media = {"frames": 120, "duration_sec": 4.0, "frames_masked": 0, "source_as_is": True}
        job.estimated_usd, job.box_source = 0.0, "none"
    else:
        job.media = {"frames": 120, "duration_sec": 4.0, "frames_masked": 40, "void_frames": 45}
        job.mask_box, job.box_source, job.estimated_usd = [0.29, 0.58, 0.71, 0.66], "measured", 0.10
    db.commit()
    db.close()


def test_blocked_segment_replaces_the_live_retry_ask_and_starts_a_source_void(world):
    w = world
    SF = w["SF"]
    rid = _launch_run(w)
    # Before the deploy: a transient-looking failure got the retry-only ask.
    _incomplete_run(w, rid, "failed after 3 attempt(s): Internal Error")
    old = _until(w, _one_pending(w, "regeneration", "retry:"))
    assert [o["id"] for o in old.options] == ["retry"]
    # The same run, same version: now classified as a content block.
    _set_error(w, rid, SAFETY)
    jobs = _until(w, lambda: _fill_jobs(SF, rid))
    assert len(jobs) == 1 and jobs[0].base["basis"] == "source"
    assert jobs[0].base["segment"]["block"] == "safety_filter"
    (eff,) = _effects(w, P.H_FILL_PREVIEW)
    assert eff.name.startswith("bpreview:") and eff.params["basis"] == "source"
    fid = jobs[0].id
    _prepared_source(SF, fid)
    ask = _until(w, _one_pending(w, "regeneration", "blocked:"))
    assert ask.name == f"blocked:{fid}"
    assert [o["id"] for o in ask.options] == ["void", "retry"]
    assert ask.options[0]["cost_mcredits"] == 20_000 and ask.artifacts
    assert ask.artifacts[0]["producer"] == f"run:{rid}"
    db = SF()
    assert db.get(AutopilotApproval, old.id).status == "superseded"   # replaced, no restart
    db.close()
    assert [a.name for a in _pending(w)] == [ask.name]
    budget_before = _ap(w).approved_budget_mcredits
    _answer(w, ask, option="void")

    def admitted():
        db = SF()
        try:
            j = db.get(RunSegmentFillJob, fid)
            return j if j.status == "admitted" else None
        finally:
            db.close()
    job = _until(w, admitted)
    (start,) = _effects(w, P.H_FILL_START)
    assert start.state == "launched" and start.paid
    assert w["calls"]["fal"] == [job.provider_task_id]
    assert _ap(w).approved_budget_mcredits == budget_before + 20_000
    assert not _effects(w, P.H_RETRY_RUN)


def test_blocked_source_as_is_is_applied_by_the_free_handler(world, monkeypatch):
    w = world
    SF = w["SF"]
    rid = _launch_run(w)
    _incomplete_run(w, rid, SAFETY)
    jobs = _until(w, lambda: _fill_jobs(SF, rid))
    fid = jobs[0].id
    _prepared_source(SF, fid, as_is=True)
    ask = _until(w, _one_pending(w, "regeneration", "blocked:"))
    assert [o["id"] for o in ask.options] == ["source", "retry"]
    assert ask.options[0]["cost_mcredits"] == 0
    assert sorted(a["kind"] for a in ask.artifacts) == ["image", "video"]
    from app import pipeline_segment_fill as psf
    monkeypatch.setattr(psf, "probe_video", lambda p: {"frames": 120})
    _answer(w, ask, option="source")
    (eff,) = _until(w, lambda: _effects(w, P.H_FILL_APPLY_SOURCE))
    assert eff.state == "launched" and not eff.paid and eff.created_refs == {"fill_jobs": [fid]}
    db = SF()
    job = db.get(RunSegmentFillJob, fid)
    rs = db.get(RunSegment, job.run_segment_id)
    run = db.get(Run, rid)
    assert job.status == "applied" and rs.local_result_path == job.composite_path
    assert run.status == RunStatus.queued and w["calls"]["process"][-1] == rid
    db.close()


def test_replaces_never_overrides_an_answer_that_raced_the_planning(world, monkeypatch):
    """The old ask is answered between planning and apply: the new Ask is
    discarded (the answer wins), the old approval is not superseded."""
    from app.autopilot import contract as Ct

    w = world
    SF = w["SF"]
    rid = _launch_run(w)
    _incomplete_run(w, rid, "failed after 3 attempt(s): Internal Error")
    old = _until(w, _one_pending(w, "regeneration", "retry:"))
    _answer(w, old, option="retry")
    racing = Ct.Ask("blocked:x", "regeneration", "q", {}, options=[
        Ct.Option("retry", "r", handler=P.H_RETRY_RUN, params={"run_id": rid})],
        replaces=(old.name,))
    import dataclasses
    patched = [dataclasses.replace(s, advance=(lambda ctx: racing)) if s.id == P.STEP_CLEAN
               else s for s in flow_mod.FLOW.steps(None, {})]
    monkeypatch.setattr(flow_mod.FLOW, "steps", lambda project, params: patched)
    _tick(w)
    db = SF()
    rows = db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == w["apid"])).scalars().all()
    by = {a.name: a for a in rows}
    assert "blocked:x" not in by and by[old.name].status == "approved"
    db.close()
    # Unanswered, it is superseded by the same Ask.
    db = SF()
    a = db.get(AutopilotApproval, old.id)
    a.status, a.answered_decision, a.chosen_option = "pending", None, None
    db.commit()
    db.close()
    _tick(w)
    db = SF()
    rows = {a.name: a for a in db.execute(select(AutopilotApproval).where(
        AutopilotApproval.autopilot_id == w["apid"])).scalars()}
    assert rows["blocked:x"].status == "pending" and rows[old.name].status == "superseded"
    db.close()


def test_restitch_handler_requeues_a_done_run_for_free_and_refuses_what_it_must_not(world):
    """2026-11-02: ``restitch_run`` re-stitches a done run's final with every
    clip cut to its slot -- only when nothing would bill or be lost."""
    from fastapi import HTTPException

    from app.autopilot.contract import ActionRefused

    SF, pid = world["SF"], world["pid"]
    h = subtitle_handlers.RestitchRun()
    assert not h.paid and not h.always_regeneration and h.estimate(None, None, {}) == 0.0

    def refused(rid, needle):
        db = SF()
        project = db.get(VideoProject, pid)
        p = h.validate(db, project, {"run_id": rid})
        _marker(db)
        with pytest.raises((ActionRefused, HTTPException)) as exc:
            h.execute(db, project, p, effect=None)
        status = getattr(exc.value, "status", None) or getattr(exc.value, "status_code", None)
        assert status == 409
        assert needle in str(exc.value.detail), exc.value.detail
        db.rollback()
        db.close()
        assert _marker_gone(SF)

    refused(_run(SF, pid, RunStatus.failed), "not done")
    rid = _run(SF, pid, RunStatus.done)
    _finish_run(world, rid, leftover=False)
    db = SF()
    run = db.get(Run, rid)
    run.stitch_plan = [{"kind": "segment", "segment_def_id": run.segment_snapshot[0]["id"]}]
    db.commit()
    db.close()
    refused(rid, "custom stitch plan")
    db = SF()
    run = db.get(Run, rid)
    run.stitch_plan = None
    snap = [dict(e) for e in run.segment_snapshot]
    run.segment_snapshot = [dict(snap[0], pre_roll_sec=0.5)] + snap[1:]
    db.commit()
    db.close()
    refused(rid, "pre/post-roll")                         # gate R1-2: nothing to conform
    db = SF()
    run = db.get(Run, rid)
    run.segment_snapshot = snap
    rs = db.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalars().first()
    gone = rs.local_result_path
    os.remove(gone)                                       # a segment would be generated again
    db.commit()
    db.close()
    refused(rid, "would bill")
    _touch(gone)

    calls_before = list(world["calls"]["process"])
    db = SF()
    project = db.get(VideoProject, pid)
    p = h.validate(db, project, {"run_id": rid})
    out = h.execute(db, project, p, effect=None)
    db.close()
    assert out["status"] == "queued"
    db = SF()
    run = db.get(Run, rid)
    assert run.status == RunStatus.queued and run.result_local_path is None
    assert run.result_gdrive_file_id == "DRIVE1"            # re-delivered in place
    db.close()
    assert world["calls"]["process"] == calls_before + [rid]
    assert h.reconcile(None, None) == "launched"
