"""AICP-364: pure step logic of the subtitle-removal autopilot flow.

The planners in ``app.autopilot.flows.subtitle_removal_plan`` are driven here
with a small simulation of the core (effects, approvals, consumption) and with
the real ``app.subtitle_loop`` functions running over synthetic detector scans
-- no database, no provider, no file decoded. The replay of project 328341d4 /
run 2845eef3 is the issue's acceptance: segment 8 (end card) proposed keep
before the launch; after the round-0 final segments 4 and 7 get VOID-fill
approvals; no Omni rerun is ever proposed.
"""
from __future__ import annotations

import json

import pytest

from app import qa_tools
from app import subtitle_loop as sl
from app.autopilot.flows import subtitle_removal_plan as P
from app.qa_tools import CaptionFrame, CaptionScan

FPS = 10.0
END_CARD = (0.20, 0.49, 0.47, 0.53)      # static UI text straddling the band's top edge
RV = "v" * 64


# ---------------------------------------------------------------------------
# Synthetic scans (the AICP-353 test conventions)
# ---------------------------------------------------------------------------

def _word_box(i: int):
    w = 0.10 + 0.04 * (i % 3)
    return (0.5 - w, 0.60, 0.5 + w, 0.64)


def _captions(start: float, end: float) -> dict:
    return {round(i / FPS, 3): _word_box(i // 5)
            for i in range(int(round(start * FPS)), int(round(end * FPS)))}


def _static(start: float, end: float, box=END_CARD) -> dict:
    return {round(i / FPS, 3): box for i in range(int(round(start * FPS)), int(round(end * FPS)))}


def _scan(boxes: dict, duration: float) -> CaptionScan:
    frames = [CaptionFrame(t=round(i / FPS, 3), score=200 if boxes.get(round(i / FPS, 3)) else 0,
                           box=boxes.get(round(i / FPS, 3)))
              for i in range(int(round(duration * FPS)))]
    hits = sum(1 for f in frames if f.box)
    return CaptionScan(path="x", fps=FPS, threshold=40, frames=len(frames), hits=hits,
                       hit_ratio=round(hits / len(frames), 4), spans=[], scores=frames)


def _points(duration: float, offsets=None):
    """Alignment points every 0.5 s (offset 0 = in sync; *offsets* maps a
    (start, end) final-time range to its offset in 30 fps frames)."""
    out = []
    for i in range(1, int(duration * 2)):
        t = round(i * 0.5, 2)
        off = 0
        for (a, b), o in (offsets or {}).items():
            if a <= t < b:
                off = o
        out.append({"t": t, "offset": off, "err": 1.0})
    return out


def _void_name(d) -> str:
    assert isinstance(d, P.Ask) and d.name.startswith("void:"), d
    return d.name


def _via_batch(sim, run, snap, option, clean=None):
    """The ONE batch question over the ready previews (2026-11-03): its
    contact-sheet compute, then the ``vbatch:`` ask; answered *option*."""
    clean = clean or (lambda: _clean(sim, run, snap))
    d = clean()
    if isinstance(d, P.Compute) and d.func == P.C_BATCH_SHEET:
        sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded",
                result={"sheet": "/att/b/batch_sheet.jpg", "out_dir": "/att/b"})
        d = clean()
    assert isinstance(d, P.Ask) and d.name.startswith("vbatch:"), d
    if option is not None:
        sim.answer(P.STEP_CLEAN, d.name, option=option)
    return d


def _via_eyes(sim, run, snap, option="per_segment"):
    """The ONE end-of-run eye-check card (2026-11-03), answered *option*."""
    d = _clean(sim, run, snap)
    assert isinstance(d, P.Ask) and d.name.startswith("eyes:"), d
    sim.answer(P.STEP_CLEAN, d.name, option=option)
    return d


@pytest.fixture
def env(monkeypatch):
    scans, durations = {}, {}

    class Info:
        def __init__(self, d):
            self.duration_sec = d
            self.fps = 30.0

    monkeypatch.setattr(qa_tools, "scan_captions", lambda path, **k: scans[path])
    monkeypatch.setattr(qa_tools, "_probe", lambda p: Info(durations[p]))
    monkeypatch.setattr(qa_tools, "contact_sheet", lambda *a, **k: {"out": a[1]})
    monkeypatch.setattr(qa_tools, "comparison_sheet", lambda *a, **k: {"out": a[2]})
    monkeypatch.setattr(qa_tools, "alignment_check", lambda s, r, **k: {
        "in_sync": True, "points": 20, "max_abs_window_median": 0.0,
        "frame_count_delta": 0, "length_ok": True, "windows": [{"x": 1}],
        "_points": _points(durations.get(r, 60.0))})
    monkeypatch.setattr(sl, "_crop_sheet", lambda s, f, spans, out: {
        "out": out, "times": [t for t, _ in sl._word_moments(spans)]})
    monkeypatch.setattr(sl, "_fill_sheet", lambda s, f, times, box, out: {
        "out": out, "times": list(times)})
    return {"scans": scans, "durations": durations}


# ---------------------------------------------------------------------------
# A tiny core simulation: effects, approvals and consumption per step
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self):
        self.effects: dict[str, list[P.EffectInfo]] = {}
        self.answers: dict[str, dict[str, P.Answer]] = {}
        self.seq = 0
        self.decisions: list = []

    def eff(self, step):
        return self.effects.setdefault(step, [])

    def ans(self, step):
        return self.answers.setdefault(step, {})

    def record(self, step, d):
        """Apply a decision the way the core would (minus the domain effects)."""
        self.decisions.append((step, d))
        if isinstance(d, (P.Launch, P.Complete, P.Fail)):
            for n in d.consumes:
                a = self.ans(step)[n]
                assert a.status in ("approved", "rejected"), n
                self.ans(step)[n] = P.Answer(n, "consumed", a.decision, a.chosen_option,
                                             a.answer_comment, members=a.members)
            left = P.unconsumed(self.ans(step))
            assert not left, f"{d} left answered approvals unconsumed: {left}"
        if isinstance(d, (P.Ask, P.Wait)):
            # The core supersedes what a decision retires / an Ask replaces,
            # when still pending (2026-11-03).
            for n in tuple(getattr(d, "retire", ())) + tuple(getattr(d, "replaces", ())):
                a = self.ans(step).get(n)
                if a is not None and a.status == "pending":
                    self.ans(step)[n] = P.Answer(n, "superseded")
        if isinstance(d, P.Ask):
            assert d.name not in self.ans(step) or self.ans(step)[d.name].status == "superseded"
            if d.kind == "regeneration":
                assert d.options, "regeneration asks need options"
            self.ans(step)[d.name] = P.Answer(d.name, "pending", members=tuple(
                (d.proposal or {}).get("members") or ()))
        return d

    def add(self, step, name, kind, handler, state, *, params=None, refs=None, result=None,
            error=None):
        self.seq += 1
        e = P.EffectInfo(id=f"{self.seq:08d}-{name.replace(':', '-')}", name=name,
                         kind=kind, handler=handler, state=state, params=params or {},
                         created_refs=refs or {}, result=result, error=error, seq=self.seq)
        self.eff(step).append(e)
        return e

    def set_state(self, step, name, state, **kw):
        lst = self.eff(step)
        i = next(i for i, e in enumerate(lst) if e.name == name)
        old = lst[i]
        lst[i] = P.EffectInfo(**{**old.__dict__, "state": state, **kw})
        return lst[i]

    def answer(self, step, name, decision="approve", option=None, comment=None):
        assert self.ans(step)[name].status == "pending"
        self.ans(step)[name] = P.Answer(name, "approved" if decision == "approve" else "rejected",
                                        decision, option, comment,
                                        members=self.ans(step)[name].members)


# ---------------------------------------------------------------------------
# 328341d4 / 2845eef3 geometry (scene cuts as analysed; prod DB, 2026-10-31)
# ---------------------------------------------------------------------------

BOUNDS = [0.0, 6.116666666666666, 12.233333333333333, 17.53333333333333, 22.833333333333332,
          30.833333333333332, 36.833333333333336, 42.333333333333336, 49.1, 54.52]
DURATION = 54.52


def _defs(keep=()):
    return [{"id": f"sd{i}", "index": i, "start_sec": BOUNDS[i], "end_sec": BOUNDS[i + 1],
             "action": "keep" if i in keep else "swap", "pre_roll_sec": 0.0,
             "post_roll_sec": 0.0} for i in range(9)]


def _source_scan():
    boxes = _captions(0.0, 49.1)
    boxes.update(_static(49.2, DURATION))
    return _scan(boxes, DURATION)


def _final_scan(leftover_segments=(4, 7)):
    boxes = {}
    if 4 in leftover_segments:
        boxes.update(_captions(23.5, 28.2))
    if 7 in leftover_segments:
        boxes.update(_captions(43.3, 48.0))
    boxes.update(_static(49.2, DURATION))          # keep segment 8: the source as is
    return _scan(boxes, DURATION)


def _faces():
    return [(t + 0.5, 1 if t < 49 else 0) for t in range(54)]


def _review_compute(env, defs, *, location=None, cfg=None):
    segs = sl.segments_from([{k: v for k, v in d.items() if k != "id"} for d in defs])
    review = sl.review_segmentation("src", segs, config=cfg, price_per_segment=168.0,
                                    faces=_faces(), out_dir="/att/review",
                                    admission_estimate=lambda sec: P.spend.video_credits(
                                        P.MODEL, None, sec), location=location)
    return P.review_result(review, segments=defs, detector_config=cfg)


def _run_info(version, results=None, status="done"):
    results = results or {}
    return P.RunInfo(id="run2845", status=status, version=version, segments=[
        {"id": f"rs{i}", "index": i, "status": "completed",
         "local_result_path": results.get(i, f"/runs/r/seg{i}-omni.mp4"),
         "seedance_task_id": f"task{i}"} for i in range(8)])


def _seg_inputs(fills=None):
    fills = fills or {}
    out = []
    for i in range(9):
        it = {"index": i, "start_sec": BOUNDS[i], "end_sec": BOUNDS[i + 1],
              "action": "keep" if i == 8 else "swap", "run_segment_id": f"rs{i}"}
        if i < 8:
            it["result_kind"] = "model"
            if i in fills:
                it.update(result_kind="void_fill", fill=fills[i])
        out.append(it)
    return out


def _context(jobs=None, active=None):
    jobs = jobs or {}
    return {"run_blockers": [], "active_fill": active,
            "segments": {f"rs{i}": {"blockers": [], "jobs": list(jobs.get(i, []))}
                         for i in range(8)}}


def _job(job_id, status="ready", **kw):
    out = {"job_id": job_id, "status": status, "box_source": "measured",
           "mask_box": [0.26, 0.58, 0.74, 0.66], "default_prompt": True,
           "estimated_usd": 0.10, "provider_task_id": None, "provider_status": None,
           "error_code": None, "error_message": None, "stale": None,
           "frames_masked": 140, "preview_url": f"/api/v2/runs/run2845/segments/fill/{job_id}"
                                                "/preview.jpg"}
    out.update(kw)
    return out


def _qa_compute(env, final, version, *, fills=None, jobs=None, takes=None):
    segs = sl.segments_from(_seg_inputs(fills))
    qa = sl.post_run_qa("src", final, segs, out_dir="/att/qa")
    ctx = _context(jobs)
    out = {"qa": qa,
           "fill": sl.fill_proposal("run2845", qa, segs, context=ctx, reserve_usd=0.10),
           "operator_options": {"omni_rerun": sl.rerun_proposal(
               "run2845", qa, segs, takes={k: len(v) for k, v in (takes or {}).items()},
               price_per_segment=168.0)},
           "handoff": sl.handoff_proposal("run2845", qa, segs)}
    return P.qa_result(out, segments_input=_seg_inputs(fills), run_version=version,
                       run_version_after=version, detector_config=None,
                       results={f"rs{i}": {"seedance_task_id": f"task{i}",
                                           "result_path": f"/runs/r/seg{i}.mp4"}
                                for i in range(8)},
                       sheets={})


def _snapshot(jobs=None, active=None, takes=None):
    return {"context": _context(jobs, active), "takes": takes or {}}


def _launches(sim):
    return [d for _, d in sim.decisions if isinstance(d, P.Launch)]


# ---------------------------------------------------------------------------
# The replay (acceptance)
# ---------------------------------------------------------------------------

def test_replay_2845eef3(env):
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, fin0=DURATION, fin1=DURATION, fin2=DURATION)
    sim = Sim()
    S = P.STEP_SEGMENTATION

    # ---- 1. segmentation: the end card is proposed keep before the launch --
    defs = _defs()
    seg_kw = dict(project_id="328341d4", project_status="ready", detector_config=None,
                  detector_version="dv0")
    d = sim.record(S, P.plan_segmentation(segments=defs, segments_version="sv0",
                                          effects=sim.eff(S), answers=sim.ans(S), **seg_kw))
    assert isinstance(d, P.Compute) and d.func == P.C_REVIEW and d.args["faces"] is True
    sim.add(S, d.name, "compute", d.func, "succeeded", result=_review_compute(env, defs))
    d = sim.record(S, P.plan_segmentation(segments=defs, segments_version="sv0",
                                          effects=sim.eff(S), answers=sim.ans(S), **seg_kw))
    assert isinstance(d, P.Ask) and d.kind == "handoff"
    assert [k["index"] for k in d.proposal["keep"]] == [8]
    assert "end card" in d.proposal["keep"][0]["reason"]
    assert d.proposal["band"]["verdict"] == "edge_touched_keep_only"   # no band change
    assert d.proposal["cost"]["credits_current"] == 1512.0
    assert d.proposal["cost"]["credits_proposed"] == 1344.0
    assert [o.id for o in d.options] == ["apply", "as_is"]
    assert [a.path for a in d.artifacts] == ["/att/review/review_seg8.jpg"]
    assert d.about == {"project_segments:328341d4": "sv0", "project_detector:328341d4": "dv0"}
    sim.answer(S, d.name, option="apply")
    d = sim.record(S, P.plan_segmentation(segments=defs, segments_version="sv0",
                                          effects=sim.eff(S), answers=sim.ans(S), **seg_kw))
    assert isinstance(d, P.Launch) and d.handler == P.H_PATCH_SEGMENTS
    assert d.params["updates"] == [{"id": "sd8", "action": "keep"}]
    assert d.expect == {"project_segments:328341d4": "sv0", "project_detector:328341d4": "dv0"}
    sim.add(S, d.name, "action", d.handler, "launched")
    defs = _defs(keep=(8,))                                   # the patch landed
    d = sim.record(S, P.plan_segmentation(segments=defs, segments_version="sv1",
                                          effects=sim.eff(S), answers=sim.ans(S), **seg_kw))
    assert isinstance(d, P.Complete) and d.output["keeps"] == [8]
    seg_out = d.output

    # ---- 2. launch: one Omni pass over the 8 swap segments -----------------
    L = P.STEP_LAUNCH
    params = P.launch_params(default_prompt="Remove all subtitles", params={})
    assert params["model"] == "gemini-omni" and params["audio_mode"] == "original"
    est = P.launch_estimate(defs, params["resolution"])
    # What create_run reserves: ceil(clip s) x 42 per clip; segment 4's 8 s is
    # 8.000000000000002 in floats, billed as 9 s -- 54 x 42 = 2268.
    assert est == sum(P.spend.video_credits("gemini-omni", params["resolution"],
                                            d["end_sec"] - d["start_sec"]) for d in defs[:8])
    assert est == 2268.0
    lines = P.budget_lines(segments=defs, duration_sec=DURATION, void_reserve_usd=0.10)
    assert all(x["credits"] == 0.0 for x in lines) and "VOID" in lines[0]["label"]
    launch_kw = dict(params=params, project_id="328341d4", project_status="ready",
                     segments=defs, segments_version="sv1", detector_config=None,
                     detector_version="dv0", seg_output=seg_out)
    d = sim.record(L, P.plan_launch(effects=sim.eff(L), answers=sim.ans(L), **launch_kw))
    assert isinstance(d, P.Launch) and d.handler == P.H_CREATE_RUN and not d.regeneration
    assert d.expect == {"project_segments:328341d4": "sv1", "project_detector:328341d4": "dv0"}
    sim.add(L, d.name, "action", d.handler, "launched", refs={"runs": ["run2845"]})
    d = sim.record(L, P.plan_launch(effects=sim.eff(L), answers=sim.ans(L), **launch_kw))
    assert isinstance(d, P.Complete) and d.output == {"run_id": "run2845"}

    # ---- 3. clean: QA of the round-0 final -> VOID fills for 4 and 7 -------
    C = P.STEP_CLEAN

    def clean(run, snapshot):
        return sim.record(C, P.plan_clean(run=run, detector_config=None, effects=sim.eff(C),
                                          answers=sim.ans(C), snapshot=snapshot,
                                          project_id="328341d4", detector_version="dv0",
                                          usd_per_credit=0.005))

    run = _run_info("ver0" + "0" * 60, status="processing")
    assert isinstance(clean(run, _snapshot()), P.Wait)
    run = _run_info("ver0" + "0" * 60)
    env["scans"]["fin0"] = _final_scan((4, 7))
    d = clean(run, _snapshot())
    assert isinstance(d, P.Compute) and d.func == P.C_QA and d.args["run_id"] == "run2845"
    qa0 = _qa_compute(env, "fin0", run.version)
    assert qa0["qa"]["dirty_segments"] == [4, 7]
    sim.add(C, d.name, "compute", d.func, "succeeded", result=qa0)

    # segments 4 and 7: free previews, one per tick -- they prepare in
    # parallel (each is bound to its own segment, 2026-11-03)
    d = clean(run, _snapshot())
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == "rs4" and d.params["mask_box"] is None
    assert d.expect == {"run:run2845": run.version}
    sim.add(C, d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f4"]})
    d = clean(run, _snapshot({4: [_job("f4", "preparing")]}))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == "rs7"
    sim.add(C, d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f7"]})
    assert isinstance(clean(run, _snapshot({4: [_job("f4", "preparing")],
                                            7: [_job("f7", "preparing")]})), P.Wait)
    # 4 is ready, 7 still preparing: the batch waits for it
    d = clean(run, _snapshot({4: [_job("f4")], 7: [_job("f7", "preparing")]}))
    assert isinstance(d, P.Wait) and "segment 7" in d.reason
    # both ready -> ONE question for both (no allowance here): first its sheet
    snap = _snapshot({4: [_job("f4")], 7: [_job("f7", mask_box=[0.2256, 0.581, 0.7763,
                                                                0.6627])]})
    d = clean(run, snap)
    assert isinstance(d, P.Compute) and d.func == P.C_BATCH_SHEET
    assert [i["index"] for i in d.args["items"]] == [4, 7]
    sim.add(C, d.name, "compute", d.func, "succeeded",
            result={"sheet": "/att/bsheet/batch_sheet.jpg", "out_dir": "/att/bsheet"})
    d = clean(run, snap)
    assert isinstance(d, P.Ask) and d.kind == "regeneration" and d.name.startswith("vbatch:")
    vb = d.name
    assert d.subject == "segments 4 and 7" and d.cost_credits == 40.0      # $0.20 at 0.005
    assert d.proposal["reserve_usd"] == 0.2 and d.proposal["expected_usd"] == 0.1
    assert d.proposal["fill_ids"] == ["f4", "f7"] and d.proposal["segments"] == [4, 7]
    assert "Segments 4 and 7 still show burned-in captions" in d.question
    assert "rebuild all 2 in one batch with VOID from the ORIGINAL video" in d.question
    assert "VOID allowance is $0" in d.question
    assert [o.id for o in d.options] == ["start", "per_segment", "accept", "manual"]
    assert d.options[0].handler == P.H_FILL_BATCH_START
    assert all(o.handler is None for o in d.options[1:])
    assert d.options[0].params == {"run_id": "run2845", "fill_ids": ["f4", "f7"],
                                   "approved_usd": 0.2, "discard_audio_patch": False,
                                   "authorization": "operator"}
    assert [a.path for a in d.artifacts] == ["/att/bsheet/batch_sheet.jpg"]
    assert isinstance(clean(run, snap), P.Wait)                          # pending
    sim.answer(C, vb, option="start")
    d = clean(run, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START and d.regeneration
    assert d.params == {"run_id": "run2845", "fill_ids": ["f4", "f7"], "approved_usd": 0.2,
                        "discard_audio_patch": False, "authorization": "operator"}
    assert d.consumes == (vb,)
    sim.add(C, d.name, "action", d.handler, "launched",
            refs={"fill_batches": ["b1"], "fill_jobs": ["f4", "f7"],
                  "provider_tasks": ["t4", "t7"]})
    admitted = _snapshot({4: [_job("f4", "admitted", provider_task_id="t4", batch_id="b1")],
                          7: [_job("f7", "composited", provider_task_id="t7", batch_id="b1")]},
                         active={"job_id": "f4", "run_segment_id": "rs4", "status": "admitted",
                                 "batch_id": "b1"})
    d = clean(run, admitted)
    assert isinstance(d, P.Wait) and "batch of 2 VOID fills" in d.reason
    # ONE promote -> ONE re-stitch and re-delivery
    assert isinstance(clean(_run_info("ver1" + "1" * 60, status="stitching"), _snapshot()),
                      P.Wait)

    fill4 = {"job_id": "f4", "mask_box": [0.2607, 0.5821, 0.7393, 0.6627],
             "frames_masked": 140, "delivery": "delivered", "replaced_take_id": "take4"}
    fill7 = {"job_id": "f7", "mask_box": [0.2256, 0.581, 0.7763, 0.6627],
             "frames_masked": 150, "delivery": "delivered", "replaced_take_id": "take7"}
    run = _run_info("ver1" + "1" * 60, results={4: "/fills/f4/composite.mp4",
                                                 7: "/fills/f7/composite.mp4"})
    applied = {4: [_job("f4", "applied", provider_task_id="t4", batch_id="b1")],
               7: [_job("f7", "applied", provider_task_id="t7", batch_id="b1")]}
    d = clean(run, _snapshot(applied))
    assert isinstance(d, P.Compute) and d.func == P.C_QA
    env["scans"]["fin1"] = _final_scan(())
    qa1 = _qa_compute(env, "fin1", run.version, fills={4: fill4, 7: fill7}, jobs=applied)
    assert qa1["qa"]["dirty_segments"] == [] and qa1["qa"]["clean"] is False
    sim.add(C, d.name, "compute", d.func, "succeeded", result=qa1)
    qa1_id = sim.eff(C)[-1].id

    # both detector-clean: ONE end-of-run eye check (no face evidence here)
    d = clean(run, _snapshot(applied))
    assert isinstance(d, P.Ask) and d.kind == "handoff" and d.name.startswith("eyes:")
    eyes = d.name
    assert d.proposal["members"] == ["f4", "f7"] and d.subject == "segments 4 and 7"
    assert [a.path for a in d.artifacts][:2] == ["/att/qa/qa_seg4_fill.jpg",
                                                 "/att/qa/qa_seg7_fill.jpg"]
    assert all(a.effect_id == qa1_id for a in d.artifacts)
    assert [o.id for o in d.options] == ["accept_all", "per_segment", "manual"]
    assert all(o.handler is None for o in d.options)
    assert isinstance(clean(run, _snapshot(applied)), P.Wait)
    sim.answer(C, eyes, option="accept_all")
    d = clean(run, _snapshot(applied))
    assert isinstance(d, P.Complete)
    assert d.consumes == (eyes,)
    assert d.depends_on == {"run:run2845": run.version, "project_detector:328341d4": "dv0"}
    assert d.output["handoff"]["keep_ranges"] == [{"start_sec": 49.1, "end_sec": 54.52}]
    assert sorted(d.output["accepted"]["accepted_fills"]) == ["f4", "f7"]
    clean_out = d.output

    # ---- 4. handoff ---------------------------------------------------------
    H = P.STEP_HANDOFF
    d = sim.record(H, P.plan_handoff(run=run, clean_output=clean_out, answers=sim.ans(H),
                                     final_path="/runs/r/results/final.mp4"))
    assert isinstance(d, P.Ask) and d.kind == "handoff"
    assert d.proposal["to_project"]["form"] == {"project_type": "localisation",
                                                "localisation_scope": "whole"}
    assert d.proposal["suggested_mode"] == "assistant"
    assert [a.kind for a in d.artifacts] == ["video", "image"]
    sim.answer(H, "handoff")
    d = sim.record(H, P.plan_handoff(run=run, clean_output=clean_out, answers=sim.ans(H),
                                     final_path="/runs/r/results/final.mp4"))
    assert isinstance(d, P.Complete) and d.consumes == ("handoff",)

    # ---- never an Omni rerun, never a paid launch without an approval ------
    launches = _launches(sim)
    assert not any(x.handler == P.H_RERUN for x in launches)
    assert all(not x.regeneration for x in launches if x.handler != P.H_FILL_BATCH_START)
    assert [x.params["fill_ids"] for x in launches
            if x.handler == P.H_FILL_BATCH_START] == [["f4", "f7"]]
    regen_asks = [d for _, d in sim.decisions if isinstance(d, P.Ask)
                  and d.kind == "regeneration"]
    assert [a.name for a in regen_asks] == [vb]
    assert "rerun" not in json.dumps([d.proposal for _, d in sim.decisions
                                      if isinstance(d, P.Ask)], default=str)


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def _seg(sim, defs, ver="sv0", cfg=None, status="ready", auto_band=True, prior_located=False):
    return sim.record(P.STEP_SEGMENTATION, P.plan_segmentation(
        project_id="p1", project_status=status, segments=defs, segments_version=ver,
        detector_config=cfg, detector_version=P.detector_sha(cfg),
        effects=sim.eff(P.STEP_SEGMENTATION),
        answers=sim.ans(P.STEP_SEGMENTATION), auto_band=auto_band,
        prior_located=prior_located))


def _short_defs(n=3, seg_len=4.0, keep=()):
    return [{"id": f"sd{i}", "index": i, "start_sec": i * seg_len, "end_sec": (i + 1) * seg_len,
             "action": "keep" if i in keep else "swap"} for i in range(n)]


def test_segmentation_waits_for_the_project_and_refuses_long_segments():
    sim = Sim()
    assert isinstance(_seg(sim, _short_defs(), status="analyzing"), P.Wait)
    d = _seg(sim, _short_defs(seg_len=12.0))
    assert isinstance(d, P.Fail) and "segment 0 is 12.0 s" in d.reason
    assert "split it in the segment editor" in d.reason


NO_LOCATION = {"state": "none", "candidate_band_y": None, "positions": [], "members": [],
               "competing": [], "reasons": ["no text-shaped clusters at all"], "frames": 24,
               "fps": 2.0, "detector": {"text_color": "#ffffff", "outline_style": "outline"}}


def _low_box(i: int):
    w = 0.10 + 0.04 * (i % 3)
    return (0.5 - w, 0.76, 0.5 + w, 0.80)        # the 514067f2 captions: below the band


def _low_captions(start, end):
    return {round(i / FPS, 3): _low_box(i // 5)
            for i in range(int(round(start * FPS)), int(round(end * FPS)))}


def _location(state="located", band=(0.70, 0.86), support=0.7, windows=5, reasons=()):
    return {"state": state, "candidate_band_y": list(band), "frames": 24, "fps": 2.0,
            "members": [0], "competing": [], "reasons": list(reasons),
            "lines_outside_band": 0.0, "detector": {},
            "positions": [{"y_range": [0.76, 0.80], "support": support, "nonstatic_frames": 17,
                           "static_frames": 0, "caption_like": True, "windows_hit": windows,
                           "y0_p5": 0.765, "y1_p95": 0.80, "band_y": list(band),
                           "sample_times": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]}]}


@pytest.fixture
def low_env(env, monkeypatch):
    """Captions at y 0.76-0.80: a band containing y 0.78 sees them, the default does not."""
    low = _scan(_low_captions(0.0, 12.0), 12.0)
    empty = _scan({}, 12.0)

    def scan(path, **k):
        lo, hi = qa_tools.caption_config(k.get("config")).band_y
        return low if lo <= 0.78 <= hi else empty

    monkeypatch.setattr(qa_tools, "scan_captions", scan)
    return env


def test_segmentation_fails_when_nothing_is_located_and_says_what_to_do(env):
    env["scans"]["src"] = _scan(_captions(0.0, 1.0), 12.0)        # 10 / 120 frames
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, defs, location=dict(NO_LOCATION)))
    d = _seg(sim, defs)
    assert isinstance(d, P.Fail)
    assert "does not see this video's captions" in d.reason
    assert "whole frame was searched" in d.reason and "QA caption detector" in d.reason
    assert "PATCH" not in d.reason


def test_segmentation_with_a_project_band_fails_without_searching(env):
    env["scans"]["src"] = _scan(_captions(0.0, 1.0), 12.0)
    sim = Sim()
    defs = _short_defs()
    cfg = {"band_y": [0.5, 0.72]}                                 # explicit = the operator's
    d = _seg(sim, defs, cfg=cfg)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, defs, cfg=cfg))
    d = _seg(sim, defs, cfg=cfg)
    assert isinstance(d, P.Fail) and "not searched for" in d.reason


def test_a_pre_locator_uncalibrated_review_is_redone_not_failed(env):
    env["scans"]["src"] = _scan(_captions(0.0, 1.0), 12.0)
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    res = _review_compute(env, defs)
    del res["review"]["location_checked"]                         # a result of the old code
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded", result=res)
    d2 = _seg(sim, defs)
    assert isinstance(d2, P.Compute) and d2.name != d.name and d2.func == P.C_REVIEW
    # A calibrated old review still counts (pending eye checks stay valid).
    env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    sim2 = Sim()
    d = _seg(sim2, defs)
    res = _review_compute(env, defs)
    del res["review"]["location_checked"]
    sim2.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded", result=res)
    assert isinstance(_seg(sim2, defs), P.Complete)


def test_located_band_is_saved_without_asking_then_reviewed_again(low_env):
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    res = _review_compute(low_env, defs, location=_location())
    band = res["review"]["band"]
    assert band["verdict"] == "located" and band["proposed_band_y"] == [0.70, 0.86]
    assert band["effective_band_y"] == [0.70, 0.86] and band["coverage"] == 1.0
    assert res["review"]["calibration"]["calibrated"]             # measured on the located band
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded", result=res)
    d = _seg(sim, defs)
    assert isinstance(d, P.Launch) and d.handler == P.H_PATCH_DETECTOR
    assert d.name.startswith("locate:") and d.consumes == ()
    assert d.params == {"caption_detector": {"band_y": [0.7, 0.86]},
                        "expect_detector_sha": P.detector_sha(None), "expect_no_band": True}
    assert d.expect == {"project_detector:p1": P.detector_sha(None)}
    sim.add(P.STEP_SEGMENTATION, d.name, "action", d.handler, "launching")
    assert isinstance(_seg(sim, defs), P.Wait)
    sim.set_state(P.STEP_SEGMENTATION, d.name, "launched")
    cfg = {"band_y": [0.7, 0.86]}
    d = _seg(sim, defs, ver="sv0", cfg=cfg)                       # new digest: a new review
    assert isinstance(d, P.Compute)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(low_env, defs, cfg=cfg))
    d = _seg(sim, defs, cfg=cfg)
    assert isinstance(d, P.Complete) and d.output["located_band_applied"] is True


def test_uncertain_located_band_asks_and_applies_on_approval(low_env):
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    loc = _location(state="ambiguous", support=0.25,
                    reasons=["the captions show in 25 % of the frames (needs 30 %)"])
    res = _review_compute(low_env, defs, location=loc)
    assert res["review"]["band"]["verdict"] == "located_uncertain"
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded", result=res)
    ask = _seg(sim, defs)
    assert isinstance(ask, P.Ask) and ask.name.startswith("locate:") and ask.kind == "handoff"
    assert [o.id for o in ask.options] == ["apply_band"]
    assert "0.70-0.86" in ask.question and "25 %" in ask.question
    assert ask.artifacts and ask.artifacts[0].path.endswith("review_location.jpg")
    assert ask.about == {"project_segments:p1": "sv0", "project_detector:p1": P.detector_sha(None)}
    assert isinstance(_seg(sim, defs), P.Wait)
    sim.answer(P.STEP_SEGMENTATION, ask.name, option="apply_band")
    d = _seg(sim, defs)
    assert isinstance(d, P.Launch) and d.handler == P.H_PATCH_DETECTOR
    assert d.consumes == (ask.name,) and d.params["caption_detector"] == {"band_y": [0.7, 0.86]}


def test_distant_caption_positions_offer_the_main_one_alone(low_env):
    sim = Sim()
    defs = _short_defs()
    loc = _location(state="ambiguous", band=(0.23, 0.86),
                    reasons=["caption text at distant heights"])
    loc["members"] = [0, 1]
    loc["positions"][0]["band_y"] = [0.70, 0.86]
    loc["positions"].append({"y_range": [0.27, 0.36], "support": 0.27, "caption_like": True,
                             "windows_hit": 2, "y0_p5": 0.26, "y1_p95": 0.37,
                             "band_y": [0.23, 0.40], "sample_times": [8.0],
                             "nonstatic_frames": 6, "static_frames": 1})
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(low_env, defs, location=loc))
    ask = _seg(sim, defs)
    assert [o.id for o in ask.options] == ["apply_band", "apply_main"]
    sim.answer(P.STEP_SEGMENTATION, ask.name, option="apply_main")
    d = _seg(sim, defs)
    assert d.handler == P.H_PATCH_DETECTOR
    assert d.params["caption_detector"] == {"band_y": [0.7, 0.86]}


def test_located_band_asks_when_automatic_saving_is_off(low_env):
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs, auto_band=False)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(low_env, defs, location=_location()))
    ask = _seg(sim, defs, auto_band=False)
    assert isinstance(ask, P.Ask) and [o.id for o in ask.options] == ["apply_band"]


def test_a_located_band_removed_by_the_operator_is_not_put_back_automatically(low_env):
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(low_env, defs, location=_location()))
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "action", d.handler, "launched")
    # The operator resets the detector (no band): the spent review is not reused,
    # a new one locates again -- and this time the flow asks.
    d = _seg(sim, defs)
    assert isinstance(d, P.Compute)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(low_env, defs, location=_location()))
    ask = _seg(sim, defs)
    assert isinstance(ask, P.Ask) and "removed again" in ask.question
    # The same across steps: launch re-plans segmentation with prior_located.
    sim2 = Sim()
    d = _seg(sim2, defs, prior_located=True)
    sim2.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
             result=_review_compute(low_env, defs, location=_location()))
    assert isinstance(_seg(sim2, defs, prior_located=True), P.Ask)


def test_located_text_the_detector_does_not_confirm_fails_with_the_evidence(env):
    env["scans"]["src"] = _scan({}, 12.0)                       # no band sees anything
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    res = _review_compute(env, defs, location=_location())
    assert res["review"]["band"]["verdict"] == "not_found"
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded", result=res)
    d = _seg(sim, defs)
    assert isinstance(d, P.Fail) and "detector confirms it in only 0 %" in d.reason


def test_default_band_that_calibrates_and_agrees_is_kept(env):
    env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)      # y 0.60-0.64
    loc = _location(band=(0.54, 0.70))
    loc["positions"][0].update(y0_p5=0.60, y1_p95=0.64)
    defs = _short_defs()
    res = _review_compute(env, defs, location=loc)
    band = res["review"]["band"]
    assert band["location_verdict"] == "ok" and band["effective_band_y"] == [0.5, 0.72]
    assert band["verdict"] == "ok"


def test_segmentation_without_proposals_completes_without_asking(env):
    env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, defs))
    d = _seg(sim, defs)
    assert isinstance(d, P.Complete) and d.output["option"] == "nothing_to_change"
    assert d.output["segments_sig"] == P.segments_sig(defs)


def test_segmentation_band_widen_then_keep_only(env):
    high = (0.40, 0.44, 0.60, 0.48)                              # above the default band
    boxes = {t: (high if int(round(t * FPS)) % 10 < 5 else (0.40, 0.60, 0.60, 0.64))
             for t in _captions(0, 8)}
    env["scans"]["src"] = _scan(boxes, 12.0)                     # segment 2: no captions
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, defs))
    ask = _seg(sim, defs)
    assert [o.id for o in ask.options] == ["apply", "keep_only", "as_is"]
    assert ask.proposal["band"]["proposed_band_y"] == [0.41, 0.67]
    # apply: the keep first, then the band, then complete
    sim.answer(P.STEP_SEGMENTATION, ask.name, option="apply")
    d = _seg(sim, defs)
    assert d.handler == P.H_PATCH_SEGMENTS and d.params["updates"] == [{"id": "sd2",
                                                                         "action": "keep"}]
    sim.add(P.STEP_SEGMENTATION, d.name, "action", d.handler, "launched")
    defs2 = _short_defs(keep=(2,))
    d = _seg(sim, defs2, ver="sv1")
    assert d.handler == P.H_PATCH_DETECTOR and d.consumes == ()
    assert d.params["caption_detector"]["band_y"] == [0.41, 0.67]
    assert d.params["expect_detector_sha"] == P.detector_sha(None)
    sim.add(P.STEP_SEGMENTATION, d.name, "action", d.handler, "launched")
    d = _seg(sim, defs2, ver="sv1", cfg={"band_y": [0.41, 0.67]})
    assert isinstance(d, P.Complete) and d.output["band_y"] == [0.41, 0.67]


def test_segmentation_keep_only_leaves_the_band(env):
    high = (0.40, 0.44, 0.60, 0.48)
    boxes = {t: (high if int(round(t * FPS)) % 10 < 5 else (0.40, 0.60, 0.60, 0.64))
             for t in _captions(0, 8)}
    env["scans"]["src"] = _scan(boxes, 12.0)
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, defs))
    ask = _seg(sim, defs)
    sim.answer(P.STEP_SEGMENTATION, ask.name, option="keep_only")
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "action", d.handler, "launched")
    d = _seg(sim, _short_defs(keep=(2,)), ver="sv1")
    assert isinstance(d, P.Complete) and d.output["band_y"] is None


def test_segmentation_edit_while_asking_reviews_again(env):
    env["scans"]["src"] = _scan(_captions(0.0, 8.0), 12.0)
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, defs))
    ask = _seg(sim, defs)
    assert isinstance(ask, P.Ask)
    # The operator edits the segments: the core supersedes the ask (about), and
    # the review of the old segments no longer counts.
    sim.ans(P.STEP_SEGMENTATION)[ask.name] = P.Answer(ask.name, "superseded")
    edited = _short_defs(keep=(1,))
    d = _seg(sim, edited, ver="sv9")
    assert isinstance(d, P.Compute) and d.args["segments_sig"] == P.segments_sig(edited)


def test_segmentation_reject_stops():
    sim = Sim()
    sim.ans(P.STEP_SEGMENTATION)["segmentation:x"] = P.Answer("segmentation:x", "rejected",
                                                               "reject", None, "wrong cuts")
    d = _seg(sim, _short_defs())
    assert isinstance(d, P.Fail) and "wrong cuts" in d.reason
    assert d.consumes == ("segmentation:x",)


def test_review_failures_are_bounded():
    sim = Sim()
    defs = _short_defs()
    for _ in range(P.MAX_COMPUTE_FAILURES):
        d = _seg(sim, defs)
        assert isinstance(d, P.Compute)
        sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "failed", error="boom")
    d = _seg(sim, defs)
    assert isinstance(d, P.Fail) and "boom" in d.reason


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------

def test_launch_reviews_again_when_segments_changed_after_segmentation(env):
    env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    sim = Sim()
    defs = _short_defs()
    seg_out = {"segments_sig": P.segments_sig(defs), "detector_sha": P.detector_sha(None)}
    edited = _short_defs(n=3, seg_len=4.0, keep=(0,))
    kw = dict(params=P.launch_params(default_prompt="x", params={}), project_id="p1",
              project_status="ready", segments_version="sv2", detector_config=None,
              detector_version="dv0", seg_output=seg_out)
    d = sim.record(P.STEP_LAUNCH, P.plan_launch(effects=sim.eff(P.STEP_LAUNCH),
                                                answers=sim.ans(P.STEP_LAUNCH),
                                                segments=edited, **kw))
    assert isinstance(d, P.Compute) and d.func == P.C_REVIEW       # not a paid launch
    sim.add(P.STEP_LAUNCH, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, edited))
    d = sim.record(P.STEP_LAUNCH, P.plan_launch(effects=sim.eff(P.STEP_LAUNCH),
                                                answers=sim.ans(P.STEP_LAUNCH),
                                                segments=edited, **kw))
    assert isinstance(d, P.Launch) and d.handler == P.H_CREATE_RUN
    assert d.expect == {"project_segments:p1": "sv2", "project_detector:p1": "dv0"}
    # a too-long segment after the review is refused before any spend
    long = _short_defs(n=2, seg_len=11.0)
    d = P.plan_launch(effects=[], answers={}, segments=long,
                      **{**kw, "seg_output": {"segments_sig": P.segments_sig(long),
                                              "detector_sha": P.detector_sha(None)}})
    assert isinstance(d, P.Fail)


def test_budget_lines_warn_on_a_long_video():
    lines = P.budget_lines(segments=_short_defs(n=10, seg_len=7.0), duration_sec=70.0,
                           void_reserve_usd=0.10)
    assert len(lines) == 2 and "long video" in lines[1]["label"]
    assert sum(x["credits"] for x in lines) == 0.0


def test_validate_params():
    assert P.validate_params(None) == {}
    assert P.validate_params({"language": "ES", "resolution": "1080p"}) == {
        "language": "es", "resolution": "1080p"}
    for bad in ({"model": "seedance"}, {"resolution": "8k"}, {"language": "xx"},
                {"prompt": " "}):
        with pytest.raises(P.FlowParamsError):
            P.validate_params(bad)


# ---------------------------------------------------------------------------
# clean: retries, stop rule, eye-check rejects, best take, false positives
# ---------------------------------------------------------------------------

def _clean(sim, run, snapshot=None, cfg=None):
    return sim.record(P.STEP_CLEAN, P.plan_clean(
        run=run, detector_config=cfg, effects=sim.eff(P.STEP_CLEAN),
        answers=sim.ans(P.STEP_CLEAN), snapshot=snapshot or _snapshot(), project_id="p1",
        detector_version=P.detector_sha(cfg), usd_per_credit=0.005))


def test_failed_run_with_every_segment_finished_retries_delivery_for_free():
    sim = Sim()
    run = _run_info("vf" * 32, status="failed")
    for n in range(P.MAX_REDELIVERIES):
        d = _clean(sim, run)
        assert isinstance(d, P.Launch) and d.handler == P.H_RETRY_DELIVERY and not d.regeneration
        sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched")
    d = _clean(sim, run)                                     # then the operator, not a stop
    assert isinstance(d, P.Ask) and d.name.startswith("redeliver_failed:")
    assert isinstance(_clean(sim, run), P.Wait)
    # A segment whose result file is gone is not "finished": the paid retry is asked.
    segs = [dict(s, has_result=s["index"] != 2) for s in run.segments]
    run2 = P.RunInfo(id=run.id, status="failed", version="vg" * 32, segments=segs)
    d = _clean(Sim(), run2)
    assert isinstance(d, P.Ask) and d.kind == "regeneration"


def test_incomplete_run_retry_is_a_regeneration_approval():
    sim = Sim()
    base = _run_info("vi" * 32, status="incomplete")
    segs = [dict(s) for s in base.segments]
    segs[3]["status"] = "failed"
    run = P.RunInfo(id=base.id, status="incomplete", version=base.version, segments=segs,
                    error_message="Google safety review")
    d = _clean(sim, run)
    assert isinstance(d, P.Ask) and d.kind == "regeneration" and d.options[0].handler == \
        P.H_RETRY_RUN
    assert "safety block" in d.question and d.about == {"run:run2845": run.version}
    sim.answer(P.STEP_CLEAN, d.name, option="retry")
    d = _clean(sim, run)
    assert isinstance(d, P.Launch) and d.handler == P.H_RETRY_RUN and d.regeneration
    assert d.consumes == (f"retry:{P.short(run.version, 12)}",)


def test_failed_run_waits_for_a_fill_in_flight_before_retrying():
    sim = Sim()
    run = _run_info("vf" * 32, status="failed")
    d = _clean(sim, run, _snapshot({4: [_job("f4", "compositing", provider_task_id="t")]},
                                   active={"job_id": "f4", "status": "compositing"}))
    assert isinstance(d, P.Wait)


def _qa_round(env, sim, run, final_boxes, fills=None, jobs=None, takes=None):
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, finx=DURATION)
    env["scans"]["finx"] = _scan(final_boxes, DURATION)
    d = _clean(sim, run, _snapshot(jobs))
    assert isinstance(d, P.Compute), d
    res = _qa_compute(env, "finx", run.version, fills=fills, jobs=jobs, takes=takes)
    return sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded", result=res)


def test_dirty_composite_hits_the_stop_rule_and_widen_gets_its_own_start_approval(env):
    """2026-11-03: a dirty VOID composite first gets ONE automatic retry with a
    grown mask (free preview, then the batch); only when that composite also
    leaves captions is the stop rule asked."""
    sim = Sim()
    fill4 = {"job_id": "f4", "mask_box": [0.3, 0.58, 0.7, 0.66], "delivery": "delivered",
             "replaced_take_id": "take4"}
    jobs = {4: [_job("f4", "applied", provider_task_id="t4")]}
    run = _run_info("vs" * 32, results={4: "/fills/f4/composite.mp4"})
    final = _captions(23.5, 24.5)                             # still words, wider than the mask
    final.update(_static(49.2, DURATION))
    _qa_round(env, sim, run, final, fills={4: fill4}, jobs=jobs)
    d = _clean(sim, run, _snapshot(jobs))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW and d.name.startswith(
        "grow:rs4:f4")
    box = d.params["mask_box"]
    assert box[0] < 0.3 and box[2] > 0.7 and box[1] < 0.58 and box[3] > 0.66   # grown
    assert d.params["mask_grow_px"] == P.GROW_PX and d.params["retry_of"] == "f4"
    assert not any(n.startswith("stop:") for n in sim.ans(P.STEP_CLEAN))
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f4g"]})
    assert isinstance(_clean(sim, run, _snapshot({4: [_job("f4g", "preparing", retry_of="f4"),
                                                      *jobs[4]]})), P.Wait)
    jobs_g = {4: [_job("f4g", box_source="request", mask_box=box, retry_of="f4"), *jobs[4]]}
    vb = _via_batch(sim, run, _snapshot(jobs_g), "start")
    assert vb.proposal["grown_retry"] == [4] and "a second, automatic try" in vb.question
    d = _clean(sim, run, _snapshot(jobs_g))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START and d.regeneration
    assert d.params["fill_ids"] == ["f4g"]
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched",
            refs={"fill_batches": ["b1"], "fill_jobs": ["f4g"]})

    # the grown composite still shows words -> now the stop rule (no loop)
    fill4g = {"job_id": "f4g", "mask_box": box, "delivery": "delivered",
              "replaced_take_id": "take4b", "retry_of": "f4"}
    jobs2 = {4: [_job("f4g", "applied", provider_task_id="t4g", retry_of="f4"), *jobs[4]]}
    run2 = _run_info("vt" * 32, results={4: "/fills/f4g/composite.mp4"})
    _qa_round(env, sim, run2, final, fills={4: fill4g}, jobs=jobs2)
    d = _clean(sim, run2, _snapshot(jobs2))
    assert isinstance(d, P.Ask) and d.name == "stop:f4g" and d.kind == "handoff"
    ids = [o.id for o in d.options]
    assert ids == ["accept", "widen_mask", "restore_previous_take", "manual", "omni_rerun"]
    assert all(o.handler is None for o in d.options)
    sim.answer(P.STEP_CLEAN, "stop:f4g", option="widen_mask")
    d = _clean(sim, run2, _snapshot(jobs2))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW and not d.regeneration
    assert d.params["mask_box"] and d.params["run_segment_id"] == "rs4"
    assert d.consumes == ("stop:f4g",) and d.name.startswith("do:stop:f4g")
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f4w"]})
    jobs3 = {4: [_job("f4w", box_source="request"), *jobs2[4]]}
    assert isinstance(_clean(sim, run2, _snapshot({4: [_job("f4w", "preparing"),
                                                       *jobs2[4]]})), P.Wait)
    _via_batch(sim, run2, _snapshot(jobs3), "start")
    d = _clean(sim, run2, _snapshot(jobs3))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START and d.regeneration
    assert d.params["fill_ids"] == ["f4w"]


def test_eye_check_reject_restores_the_replaced_take_for_free(env):
    sim = Sim()
    fill4 = {"job_id": "f4", "mask_box": [0.3, 0.58, 0.7, 0.66], "delivery": "delivered",
             "replaced_take_id": "take4"}
    jobs = {4: [_job("f4", "applied", provider_task_id="t4")]}
    run = _run_info("ve" * 32, results={4: "/fills/f4/composite.mp4"})
    _qa_round(env, sim, run, _static(49.2, DURATION), fills={4: fill4}, jobs=jobs)
    _via_eyes(sim, run, _snapshot(jobs), "per_segment")      # "some do not": one by one
    d = _clean(sim, run, _snapshot(jobs))
    assert d.name == "eye:f4"
    sim.answer(P.STEP_CLEAN, "eye:f4", option="restore_previous_take")
    d = _clean(sim, run, _snapshot(jobs))
    assert isinstance(d, P.Launch) and d.handler == P.H_RESTORE
    assert d.params == {"run_id": "run2845", "takes": ["take4"], "discard_audio_patch": False}
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched")
    # nothing else to do in this QA round: the run will change
    assert isinstance(_clean(sim, run, _snapshot(jobs)), P.Wait)


def test_omni_rerun_choice_is_launched_as_a_regeneration_and_best_take_is_asked(env):
    sim = Sim()
    run = _run_info("v0" * 32)
    final = _captions(23.5, 24.5)
    final.update(_static(49.2, DURATION))
    # A repeat of an already-paid fill -> the operator; the operator picks an Omni rerun.
    jobs = {4: [_job("f4", "failed", provider_task_id="t4", error_code="provider_failed")]}
    _qa_round(env, sim, run, final, jobs=jobs)
    d = _clean(sim, run, _snapshot(jobs))
    assert isinstance(d, P.Ask) and d.name.startswith("hand:rs4:")
    assert [o.id for o in d.options] == ["accept", "manual", "omni_rerun"]
    sim.answer(P.STEP_CLEAN, d.name, option="omni_rerun")
    d = _clean(sim, run, _snapshot(jobs))
    assert isinstance(d, P.Launch) and d.handler == P.H_RERUN and d.regeneration
    assert d.params["segments"][0]["run_segment_id"] == "rs4"
    assert "Change ONLY that box" in d.params["segments"][0]["prompt"]   # coordinate variant
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", params=d.params)
    # The rerun came back worse (more hit frames): the old take is offered back.
    run2 = P.RunInfo(id="run2845", status="done", version="v1" * 32, segments=[
        dict(s, seedance_task_id="task4b" if s["index"] == 4 else s["seedance_task_id"],
             local_result_path="/runs/r/seg4-rerun.mp4" if s["index"] == 4
             else s["local_result_path"]) for s in run.segments])
    worse = _captions(23.5, 27.0)
    worse.update(_static(49.2, DURATION))
    takes = {"rs4": [{"take_id": "take4", "seedance_task_id": "task4", "purged": False}]}
    _qa_round(env, sim, run2, worse, jobs=jobs, takes=takes)
    d = _clean(sim, run2, _snapshot(jobs, takes=takes))
    assert isinstance(d, P.Ask) and d.name.startswith("best:")
    assert [o.id for o in d.options] == ["restore", "keep_new"]
    assert d.proposal["segments"][0]["hits_now"] > d.proposal["segments"][0]["hits_before"]
    sim.answer(P.STEP_CLEAN, d.name, option="restore")
    d = _clean(sim, run2, _snapshot(jobs, takes=takes))
    assert isinstance(d, P.Launch) and d.handler == P.H_RESTORE and not d.regeneration
    assert d.params["takes"] == ["take4"]


def test_false_positive_needs_the_operator_then_clean(env):
    sim = Sim()
    run = _run_info("vp" * 32)
    env["scans"]["src"] = _source_scan()
    # Segment 7 swap shows the source's static text (a phone UI held still).
    src = _captions(0.0, 42.3)
    src.update(_static(42.4, 49.1, box=(0.30, 0.60, 0.70, 0.64)))
    src.update(_static(49.2, DURATION))
    env["scans"]["src"] = _scan(src, DURATION)
    env["durations"].update(src=DURATION, finx=DURATION)
    fin = _static(42.4, 49.1, box=(0.30, 0.60, 0.70, 0.64))
    fin.update(_static(49.2, DURATION))
    env["scans"]["finx"] = _scan(fin, DURATION)
    d = _clean(sim, run)
    sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded",
            result=_qa_compute(env, "finx", run.version))
    d = _clean(sim, run)
    assert isinstance(d, P.Ask) and d.name.startswith("fp:s7:")
    assert [o.id for o in d.options] == ["confirm", "as_caption"]
    sim.answer(P.STEP_CLEAN, d.name, option="confirm")
    d = _clean(sim, run)
    assert isinstance(d, P.Complete) and d.output["accepted"]["confirmed_fp"] == [7]


def test_any_rejection_stops_the_autopilot():
    sim = Sim()
    sim.ans(P.STEP_CLEAN)["eye:f4"] = P.Answer("eye:f4", "rejected", "reject", None, "ugly")
    d = _clean(sim, _run_info("vr" * 32))
    assert isinstance(d, P.Fail) and "ugly" in d.reason and d.consumes == ("eye:f4",)


def test_qa_failures_and_refusals(env):
    sim = Sim()
    run = _run_info("vq" * 32)
    for _ in range(P.MAX_COMPUTE_FAILURES):
        d = _clean(sim, run)
        sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "failed", error="ffmpeg died")
    assert isinstance(_clean(sim, run), P.Fail)
    sim2 = Sim()
    d = _clean(sim2, run)
    sim2.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded", result=P.qa_result(
        {}, segments_input=[], run_version=run.version, run_version_after=run.version,
        detector_config=None, results={}, refused="final is 60.00 s, source 54.52 s"))
    d = _clean(sim2, run)
    assert isinstance(d, P.Fail) and "60.00" in d.reason


def test_qa_that_saw_the_run_change_is_redone(env):
    sim = Sim()
    run = _run_info("vc" * 32)
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, finx=DURATION)
    env["scans"]["finx"] = _final_scan(())
    d = _clean(sim, run)
    res = _qa_compute(env, "finx", run.version)
    res["actionable"] = False
    sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded", result=res)
    d = _clean(sim, run)
    assert isinstance(d, P.Compute) and d.name.endswith(":r2")


def test_detector_change_redoes_qa(env):
    sim = Sim()
    run = _run_info("vd" * 32)
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, finx=DURATION)
    env["scans"]["finx"] = _final_scan(())
    d = _clean(sim, run)
    sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded",
            result=_qa_compute(env, "finx", run.version))
    d = _clean(sim, run, cfg={"band_y": [0.4, 0.8]})
    assert isinstance(d, P.Compute) and d.args["detector_sha"] == P.detector_sha(
        {"band_y": [0.4, 0.8]})


def test_apply_answers_recomputes_clean_like_post_run_qa(env):
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, finx=DURATION)
    env["scans"]["finx"] = _final_scan(())
    fill4 = {"job_id": "f4", "mask_box": [0.3, 0.58, 0.7, 0.66], "delivery": "delivered"}
    segs = sl.segments_from(_seg_inputs({4: fill4}))
    qa = sl.post_run_qa("src", "finx", segs)
    run = _run_info("va" * 32)
    eff, _ = P.apply_answers(qa, {}, run, "q")
    assert eff["clean"] is qa["clean"] is False
    accepted = {"eye:f4": P.Answer("eye:f4", "consumed", "approve", "accept")}
    eff, info = P.apply_answers(qa, accepted, run, "q")
    ref = sl.post_run_qa("src", "finx", segs, confirmed_fills=["f4"])
    assert eff["clean"] is ref["clean"] is True and info["accepted_fills"] == ["f4"]
    # a composite whose re-delivery is not confirmed cannot be accepted
    segs2 = sl.segments_from(_seg_inputs({4: dict(fill4, delivery="republished")}))
    qa2 = sl.post_run_qa("src", "finx", segs2)
    assert P.apply_answers(qa2, accepted, run, "q")[0]["clean"] is False


def test_names_fit_the_contract():
    import re

    rx = re.compile(r"^[a-z0-9_:.-]{1,96}$")
    rsid = "0b790c6d-a63e-4bd6-b11e-30ad9b2acdd2"
    for name in (f"preview:{rsid}:{'a' * 12}:r3", f"start:{rsid}", f"eye:{rsid}",
                 f"do:stop:{rsid}:r2", f"operator:{rsid}:{'b' * 8}", f"cancel:{rsid}:r1",
                 f"hand:{rsid}:{'b' * 8}", f"void:{rsid}:{'b' * 8}:{rsid}",
                 f"do:void:{rsid}:{'b' * 8}:{rsid}:r12", f"restitch:{'c' * 12}:r1",
                 f"restitch_failed:{'c' * 12}", f"edge:{rsid}:{'b' * 8}",
                 f"qa:{'c' * 12}:r1", f"review:{'d' * 12}:r1"):
        assert rx.match(name), name


# ---------------------------------------------------------------------------
# Terra §2 round 2 regressions
# ---------------------------------------------------------------------------

def _reviewed(env, sim, defs, boxes=None):
    env["scans"]["src"] = _scan(boxes if boxes is not None else _captions(0.0, 8.0), 12.0)
    d = _seg(sim, defs)
    assert isinstance(d, P.Compute)
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded",
            result=_review_compute(env, defs))
    return _seg(sim, defs)


def test_r2_1_segments_edited_back_after_the_keep_patch_are_reviewed_again(env):
    sim = Sim()
    defs = _short_defs()
    ask = _reviewed(env, sim, defs)                       # segment 2 has no captions
    sim.answer(P.STEP_SEGMENTATION, ask.name, option="apply")
    d = _seg(sim, defs)
    assert d.handler == P.H_PATCH_SEGMENTS
    sim.add(P.STEP_SEGMENTATION, d.name, "action", d.handler, "launched")
    # The operator puts segment 2 back to swap: the old approval does not cover it.
    d = _seg(sim, defs, ver="sv2")
    assert isinstance(d, P.Compute) and d.name.endswith(":r2")


def test_r2_2_detector_change_after_segmentation_is_reviewed_before_launch(env):
    env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)
    defs = _short_defs()
    seg_out = {"segments_sig": P.segments_sig(defs), "detector_sha": P.detector_sha(None)}
    cfg = {"band_y": [0.45, 0.75]}
    d = P.plan_launch(effects=[], answers={}, params={}, project_id="p1", project_status="ready",
                      segments=defs, segments_version="sv0", detector_config=cfg,
                      detector_version="dv1", seg_output=seg_out)
    assert isinstance(d, P.Compute) and d.func == P.C_REVIEW
    assert d.args["detector_sha"] == P.detector_sha(cfg)


def test_r2_3_answers_are_bound_to_the_result_identity_not_its_path():
    run = _run_info("vt" * 32)
    row = {"run_segment_id": "rs3", "start_sec": 17.5, "end_sec": 22.8}
    tok = P.result_token(run, row)
    segs = [dict(s, seedance_task_id="new") if s["id"] == "rs3" else s for s in run.segments]
    rerun = P.RunInfo(id=run.id, status="done", version="vu" * 32, segments=segs)
    assert P.result_token(rerun, row) != tok                 # same path, another output
    segs = [dict(s, result_identity=[1, 2]) if s["id"] == "rs3" else s for s in run.segments]
    assert P.result_token(P.RunInfo(id=run.id, status="done", version="x", segments=segs),
                          row) != tok


def test_a_drifted_final_is_restitched_for_free_then_unmeasurable_timing_is_asked(
        env, monkeypatch):
    """2026-11-02 (replaces R2-4's one-option "inconclusive" waiver): a final
    off the source timeline gets ONE free re-stitch (every clip cut to its
    slot); a timeline that still cannot be proven segment by segment is a real
    question with two options, never a fake one."""
    sim = Sim()
    run = _run_info("vn" * 32)
    monkeypatch.setattr(qa_tools, "alignment_check", lambda s, r, **k: {
        "in_sync": False, "points": 20, "max_abs_window_median": 12.0,
        "frame_count_delta": 0, "length_ok": True, "windows": [], "_points": []})
    final = _static(49.2, DURATION)
    _qa_round(env, sim, run, final)
    d = _clean(sim, run)
    assert isinstance(d, P.Launch) and d.handler == P.H_RESTITCH and not d.regeneration
    assert d.params == {"run_id": "run2845"} and d.expect == {"run:run2845": run.version}
    assert "free, nothing generated" in d.reason
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", params=d.params)
    run2 = _run_info("vm" * 32)
    _qa_round(env, sim, run2, final)
    d = _clean(sim, run2)
    assert isinstance(d, P.Ask) and d.name.startswith("timing:") and d.kind == "handoff"
    assert [o.id for o in d.options] == ["accept", "manual"]
    assert "could not prove" in d.question and "12 frames" in d.question
    sim.answer(P.STEP_CLEAN, d.name, option="accept")
    d = _clean(sim, run2)
    assert isinstance(d, P.Complete)
    assert [x.handler for _, x in sim.decisions if isinstance(x, P.Launch)] == [P.H_RESTITCH]


def test_no_question_has_a_single_choice_for_a_timing_doubt(env, monkeypatch):
    """The 514067f2 'inconclusive' card (one option, jargon) is gone: an
    uncalibrated detector is the only inconclusive question, with two
    options."""
    sim = Sim()
    run = _run_info("vk" * 32)
    final = _static(49.2, DURATION)
    _qa_round(env, sim, run, final)
    d = _clean(sim, run)
    assert not (isinstance(d, P.Ask) and d.name.startswith("inconclusive:"))
    for _, x in sim.decisions:
        if isinstance(x, P.Ask):
            assert len(x.options) >= 2 or x.name.startswith("keepcap:"), x.name


# ---------------------------------------------------------------------------
# Terra §2 round 3 regressions
# ---------------------------------------------------------------------------

def test_r3_1_a_leftover_at_the_cut_of_a_shifted_segment_waits_for_that_segment(
        env, monkeypatch):
    """A leftover near a cut where the NEXT segment's picture is shifted may be
    that segment's clip: it is not filled blindly. The shifted segment is
    rebuilt first (out of sync after the free re-stitch); when nothing else can
    move, the operator is asked in plain words with real options."""
    sim = Sim()
    run = _run_info("w1" * 32)
    late4 = {(BOUNDS[4], BOUNDS[5]): -8}
    monkeypatch.setattr(qa_tools, "alignment_check", lambda s, r, **k: {
        "in_sync": False, "points": 100, "max_abs_window_median": 8.0,
        "frame_count_delta": 0, "length_ok": True, "windows": [],
        "_points": _points(DURATION, late4)})
    # the re-stitch already ran on this run
    sim.add(P.STEP_CLEAN, "restitch:x:r1", "action", P.H_RESTITCH, "launched",
            params={"run_id": "run2845"})
    final = _captions(22.5, 22.8)                            # segment 3, 0.03 s before the cut
    final.update(_static(49.2, DURATION))
    _qa_round(env, sim, run, final)
    d = _clean(sim, run)
    # segment 4 is out of sync -> its rebuild from the original goes first
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == "rs4" and "out of sync" in d.reason
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f4"]})
    snap = _snapshot({4: [_job("f4")]})
    d = _via_batch(sim, run, snap, None)
    assert "Segment 4 is out of sync with the sound" in d.question
    assert "8 frames" in d.question and d.proposal["segments"] == [4]
    # the operator ships segment 4 as it is -> segment 3's leftover is asked
    sim.answer(P.STEP_CLEAN, d.name, option="accept")
    d = _clean(sim, run, snap)
    assert isinstance(d, P.Ask) and d.name.startswith("edge:rs3:"), d
    assert [o.id for o in d.options] == ["rebuild", "accept", "manual"]
    assert "unclear which segment" in d.question
    sim.answer(P.STEP_CLEAN, d.name, option="rebuild")
    d = _clean(sim, run, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == "rs3"
    assert not any(isinstance(x, P.Launch) and x.handler in (P.H_FILL_START,
                                                              P.H_FILL_BATCH_START)
                   for _, x in sim.decisions)


def test_r3_4_a_detector_changed_back_after_the_band_patch_is_reviewed_again(env):
    high = (0.40, 0.44, 0.60, 0.48)
    boxes = {t: (high if int(round(t * FPS)) % 10 < 5 else (0.40, 0.60, 0.60, 0.64))
             for t in _captions(0, 12)}
    sim = Sim()
    defs = _short_defs()
    ask = _reviewed(env, sim, defs, boxes)
    assert ask.proposal["band"]["proposed_band_y"]
    sim.answer(P.STEP_SEGMENTATION, ask.name, option="apply")
    d = _seg(sim, defs)
    assert d.handler == P.H_PATCH_DETECTOR
    sim.add(P.STEP_SEGMENTATION, d.name, "action", d.handler, "launched")
    d = _seg(sim, defs)                                        # PATCHed back to the default
    assert isinstance(d, P.Compute) and d.name.endswith(":r2")


def test_r3_3_best_take_uses_the_take_the_rerun_archived():
    prev = {"qa": {"segments": [{"run_segment_id": "rs4", "result_hits": 3}]},
            "results": {"rs4": {"seedance_task_id": None, "result_path": "/fills/c.mp4"}}}
    cur = {"qa": {"segments": [{"run_segment_id": "rs4", "result_hits": 40}]}}
    takes = {"rs4": [{"take_id": "comp", "result_path": "/takes/b.mp4"},
                     {"take_id": "later", "result_path": "/takes/c.mp4"}]}
    assert P.best_take_restores("R", rerun_rsids=["rs4"], prev_qa=prev, cur_qa=cur, takes=takes,
                                archived={"rs4": "comp"})["takes"] == ["comp"]
    # no provenance -> no guess
    assert P.best_take_restores("R", rerun_rsids=["rs4"], prev_qa=prev, cur_qa=cur,
                                takes=takes) is None


def test_r3_5_a_replaced_source_redoes_qa(env):
    sim = Sim()
    run = _run_info("w5" * 32)
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, finx=DURATION)
    env["scans"]["finx"] = _final_scan(())
    d = _clean(sim, run, {"context": _context(), "takes": {}, "source_identity": [1, 1]})
    res = _qa_compute(env, "finx", run.version)
    res["source_identity"] = [1, 1]
    sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded", result=res)
    d = _clean(sim, run, {"context": _context(), "takes": {}, "source_identity": [2, 2]})
    assert isinstance(d, P.Compute) and d.name.endswith(":r2")


def test_r3_6_delivery_retries_are_bounded_across_run_versions():
    sim = Sim()
    for n in range(P.MAX_REDELIVERIES):
        run = _run_info(f"z{n}" * 32, status="failed")         # each re-stitch moves the version
        d = _clean(sim, run)
        assert isinstance(d, P.Launch) and d.handler == P.H_RETRY_DELIVERY
        sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched")
    d = _clean(sim, _run_info("zz" * 32, status="failed"))
    assert isinstance(d, P.Ask) and d.name.startswith("redeliver_failed:")


def test_r3_7_a_rerun_effect_without_params_fails_closed(env):
    sim = Sim()
    run = _run_info("w7" * 32)
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, finx=DURATION)
    env["scans"]["finx"] = _final_scan(())
    d = _clean(sim, run)
    sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded",
            result=_qa_compute(env, "finx", run.version))
    sim.add(P.STEP_CLEAN, "do:operator:rs4:x:r1", "action", P.H_RERUN, "launched")
    run2 = _run_info("w8" * 32)
    d = _clean(sim, run2)
    sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded",
            result=_qa_compute(env, "finx", run2.version))
    with pytest.raises(ValueError, match="R7"):
        _clean(sim, run2)


# ---------------------------------------------------------------------------
# Terra §1 gate regressions
# ---------------------------------------------------------------------------

def test_gate_r1_1_a_segment_at_its_preview_limit_does_not_hold_up_the_next(env):
    sim = Sim()
    run = _run_info("g1" * 32)
    final = _captions(5.0, 5.9)                               # segment 0 leftover
    final.update(_captions(23.5, 24.5))                       # segment 4 leftover
    final.update(_static(49.2, DURATION))
    _qa_round(env, sim, run, final)
    rv = P.short(run.version, 12)
    for n in range(P.MAX_PREVIEWS_PER_SEGMENT):              # segment 0's previews all failed
        sim.add(P.STEP_CLEAN, f"preview:rs0:{rv}:r{n + 1}", "action", P.H_FILL_PREVIEW,
                "launched", refs={"fill_jobs": [f"x{n}"]})
    failed = [_job(f"x{n}", "failed", error_code="worker_died") for n in range(3)]
    d = _clean(sim, run, _snapshot({0: failed}))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == "rs4"


def test_gate_r1_2_a_failed_run_without_segment_rows_is_a_paid_retry():
    sim = Sim()
    segs = [{"id": None, "index": i, "status": "missing", "has_result": False}
            for i in range(3)]
    run = P.RunInfo(id="run2845", status="failed", version="g2" * 32, segments=segs)
    d = _clean(sim, run)
    assert isinstance(d, P.Ask) and d.kind == "regeneration"


def test_text_outside_a_calibrating_default_band_is_shown_on_the_eye_check(env):
    env["scans"]["src"] = _scan(_captions(0.0, 12.0), 12.0)      # y 0.60-0.64: calibrated
    loc = _location(band=(0.54, 0.70))
    loc["positions"][0].update(y0_p5=0.60, y1_p95=0.64)
    loc["positions"].append({"y_range": [0.18, 0.22], "support": 0.0, "caption_like": False,
                             "windows_hit": 0, "static_frames": 6, "nonstatic_frames": 0,
                             "y0_p5": 0.18, "y1_p95": 0.22, "band_y": [0.12, 0.28],
                             "sample_times": [2.0, 6.0]})
    loc["competing"] = [1]
    sim = Sim()
    defs = _short_defs()
    d = _seg(sim, defs)
    res = _review_compute(env, defs, location=loc)
    band = res["review"]["band"]
    assert band["verdict"] == "ok" and band["needs_visual_check"]
    assert band["sheet"].endswith("review_location.jpg")
    sim.add(P.STEP_SEGMENTATION, d.name, "compute", d.func, "succeeded", result=res)
    ask = _seg(sim, defs)
    assert isinstance(ask, P.Ask) and ask.name.startswith("segmentation:")
    assert any(a.path.endswith("review_location.jpg") for a in ask.artifacts)


# ---------------------------------------------------------------------------
# A segment the provider refused with a content filter (514067f2 / d8105ab5):
# VOID fill from the SOURCE is the flow's standard path, retry the secondary.
# ---------------------------------------------------------------------------

SAFETY = ("failed after 3 attempt(s): Request blocked: The input content was flagged by safety "
          "filters as potentially dangerous and cannot be processed.")


def _blocked_run(version, blocked=(1,), transient=(), status="incomplete"):
    segs = []
    for i in range(6):
        sg = {"id": f"rs{i}", "index": i, "status": "completed",
              "local_result_path": f"/runs/r/seg{i}-omni.mp4", "seedance_task_id": f"task{i}",
              "has_result": True, "content_block": None, "error_message": None,
              "start_sec": 2.0 * i, "end_sec": 2.0 * i + 2.0}
        if i in blocked:
            sg.update(status="failed", local_result_path=None, has_result=False,
                      content_block="safety_filter", error_message=SAFETY,
                      seedance_task_id=f"blocked{i}")
        if i in transient:
            sg.update(status="failed", local_result_path=None, has_result=False,
                      error_message="failed after 3 attempt(s): Internal Error")
        segs.append(sg)
    return P.RunInfo(id="run2845", status=status, version=version, segments=segs,
                     error_message="1 swap segment(s) did not complete")


def _sjob(job_id, status="ready", rs=1, **kw):
    out = _job(job_id, status, basis="source", block_task=f"blocked{rs}", source_as_is=False,
               preview_path=f"/runs/r/segment_fills/{job_id}/prep/preview.jpg",
               clip_path=f"/runs/r/segment_fills/{job_id}/prep/clip.mp4",
               reservation_held=None)
    out.update(kw)
    return out


def _bclean(sim, run, jobs=None, active=None, run_blockers=()):
    snap = _snapshot(jobs, active)
    snap["context"]["run_blockers"] = list(run_blockers)
    return sim.record(P.STEP_CLEAN, P.plan_clean(
        run=run, detector_config=None, effects=sim.eff(P.STEP_CLEAN),
        answers=sim.ans(P.STEP_CLEAN), snapshot=snap, project_id="p1",
        detector_version="det-v", usd_per_credit=0.005))


def test_blocked_segment_source_preview_then_void_fill_approval_replacing_the_retry():
    sim = Sim()
    run = _blocked_run("bk" * 32)
    rv = P.short(run.version, 12)
    sim.ans(P.STEP_CLEAN)[f"retry:{rv}"] = P.Answer(f"retry:{rv}", "pending")   # the live card
    d = _bclean(sim, run)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW and not d.regeneration
    assert d.params == {"run_id": "run2845", "run_segment_id": "rs1", "mask_box": None,
                        "prompt": None, "basis": "source"}
    assert d.name.startswith("bpreview:rs1:") and d.consumes == ()
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f1"]})
    assert isinstance(_bclean(sim, run, {1: [_sjob("f1", "preparing")]}), P.Wait)

    jobs = {1: [_sjob("f1")]}
    d = _bclean(sim, run, jobs)
    assert isinstance(d, P.Ask) and d.name == "blocked:f1" and d.kind == "regeneration"
    assert [(o.id, o.handler) for o in d.options] == [("void", P.H_FILL_START),
                                                       ("retry", P.H_RETRY_RUN)]
    assert d.options[0].params == {"run_id": "run2845", "fill_id": "f1", "approved_usd": 0.1,
                                   "discard_audio_patch": False}
    assert d.replaces == (f"retry:{rv}",)
    assert d.about == {"run:run2845": run.version, "project_detector:p1": "det-v"}
    assert [(a.kind, a.run_id) for a in d.artifacts] == [("image", "run2845")]
    assert d.cost_credits == 20.0 and "VOID fill from the SOURCE" in d.question
    assert d.proposal["block"] == "safety_filter" and d.proposal["provider_error"] == SAFETY
    sim.ans(P.STEP_CLEAN)[f"retry:{rv}"] = P.Answer(f"retry:{rv}", "superseded")  # core: replaces
    assert isinstance(_bclean(sim, run, jobs), P.Wait)             # pending
    sim.answer(P.STEP_CLEAN, "blocked:f1", option="void")
    d = _bclean(sim, run, jobs)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_START and d.regeneration
    assert d.params == {"run_id": "run2845", "fill_id": "f1", "approved_usd": 0.1,
                        "discard_audio_patch": False}
    assert d.consumes == ("blocked:f1",) and d.name == "do:blocked:f1:r1"
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched")
    # admitted / compositing: the generic in-flight wait
    assert isinstance(_bclean(sim, run, {1: [_sjob("f1", "admitted", provider_task_id="t1",
                                                   provider_status="submitted")]},
                              active={"job_id": "f1", "status": "admitted"}), P.Wait)


def test_blocked_segment_retry_option_is_a_paid_retry():
    sim = Sim()
    run = _blocked_run("br" * 32)
    jobs = {1: [_sjob("f1")]}
    d = _bclean(sim, run, jobs)
    sim.answer(P.STEP_CLEAN, d.name, option="retry")
    d = _bclean(sim, run, jobs)
    assert isinstance(d, P.Launch) and d.handler == P.H_RETRY_RUN and d.regeneration
    assert d.params == {"run_id": "run2845"} and d.name == "do:blocked:f1:r1"


def test_a_legacy_retry_answered_before_the_replacement_is_honoured():
    sim = Sim()
    run = _blocked_run("bl" * 32)
    rv = P.short(run.version, 12)
    sim.ans(P.STEP_CLEAN)[f"retry:{rv}"] = P.Answer(f"retry:{rv}", "approved", "approve", "retry")
    d = _bclean(sim, run)
    assert isinstance(d, P.Launch) and d.handler == P.H_RETRY_RUN and d.regeneration
    assert d.name == f"do:retry:{rv}:r1"


def test_source_as_is_is_asked_then_applied_for_free():
    sim = Sim()
    run = _blocked_run("bs" * 32)
    jobs = {1: [_sjob("f1", estimated_usd=0.0, source_as_is=True, mask_box=None,
                      source_as_is_reason="the detector found no caption in the source clip")]}
    d = _bclean(sim, run, jobs)
    assert isinstance(d, P.Ask) and d.name == "blocked:f1"
    assert [(o.id, o.handler) for o in d.options] == [("source", P.H_FILL_APPLY_SOURCE),
                                                       ("retry", P.H_RETRY_RUN)]
    assert d.cost_credits == 0.0 and "NO caption" in d.question and "miss is possible" in d.question
    assert sorted(a.kind for a in d.artifacts) == ["image", "video"]
    sim.answer(P.STEP_CLEAN, "blocked:f1", option="source")
    d = _bclean(sim, run, jobs)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_APPLY_SOURCE and not d.regeneration
    assert d.params == {"run_id": "run2845", "fill_id": "f1"} and d.consumes == ("blocked:f1",)


def test_a_fal_disabled_void_proposal_waits_but_source_as_is_does_not():
    sim = Sim()
    run = _blocked_run("bf" * 32)
    off = ("Fal is disabled (FAL_ENABLED / FAL_API_KEY): the start would be refused",)
    d = _bclean(sim, run, {1: [_sjob("f1")]}, run_blockers=off)
    assert isinstance(d, P.Wait) and "Fal is disabled" in d.reason
    d = _bclean(Sim(), run, {1: [_sjob("f2", estimated_usd=0.0, source_as_is=True)]},
                run_blockers=off)
    assert isinstance(d, P.Ask) and d.options[0].id == "source"
    # The free preview itself is never held up by Fal.
    d = _bclean(Sim(), run, None, run_blockers=off)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW


def test_a_stale_source_preview_is_canceled_and_a_previous_task_s_jobs_are_ignored():
    sim = Sim()
    run = _blocked_run("bt" * 32)
    d = _bclean(sim, run, {1: [_sjob("f1", stale="segment.error_message")]})
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_CANCEL
    assert d.params == {"run_id": "run2845", "fill_id": "f1"}
    # A ready preview of the PREVIOUS refusal (another blocked task) is not this one's.
    d = _bclean(Sim(), run, {1: [_sjob("f0", block_task="blocked-before-the-retry")]})
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW


def test_previews_that_keep_failing_ask_the_operator_never_loop():
    sim = Sim()
    run = _blocked_run("bp" * 32)
    failed = []
    for n in range(P.MAX_BLOCKED_PREVIEWS):
        d = _bclean(sim, run, {1: failed})
        assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW, d
        sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", refs={"fill_jobs": [f"x{n}"]})
        failed = [_sjob(f"x{n}", "failed", error_code="prepare_crashed")] + failed
    d = _bclean(sim, run, {1: failed})
    assert isinstance(d, P.Ask) and d.name.startswith("blocked_failed:rs1:")
    assert [(o.id, o.handler) for o in d.options] == [("retry", P.H_RETRY_RUN), ("manual", None)]
    sim.answer(P.STEP_CLEAN, d.name, option="manual")
    d = _bclean(sim, run, {1: failed})
    assert isinstance(d, P.Wait) and "by hand" in d.reason


def test_a_deterministic_preview_refusal_asks_at_once():
    sim = Sim()
    run = _blocked_run("bd" * 32)
    d = _bclean(sim, run, {1: [_sjob("x0", "failed", error_code="segment_too_short",
                                     error_message="the segment has 40 frames")]})
    assert isinstance(d, P.Ask) and d.name.startswith("blocked_failed:")
    assert "segment_too_short" in d.question


def test_a_source_fill_refused_at_fal_waits_for_the_operator_then_asks():
    sim = Sim()
    run = _blocked_run("bz" * 32)
    held = _sjob("f1", "admitted", provider_task_id="t1", provider_status="refused",
                 reservation_held=True)
    d = _bclean(sim, run, {1: [held]}, active={"job_id": "f1", "status": "admitted"})
    assert isinstance(d, P.Wait) and "close t1" in d.reason and "abandon t1" in d.reason
    for ps, word in (("submit_unknown", "adopt t1"), ("needs_reconciliation", "resume t1")):
        j = _sjob("f1", "admitted", provider_task_id="t1", provider_status=ps)
        d = _bclean(Sim(), run, {1: [j]}, active={"job_id": "f1", "status": "admitted"})
        assert isinstance(d, P.Wait) and word in d.reason, d
    # The operator closed it: the fill failed at Fal -> asked, never a second VOID by itself.
    gone = _sjob("f1", "failed", provider_task_id="t1", provider_status="refused",
                 error_code="provider_refused", error_message="content_policy_violation")
    d = _bclean(sim, run, {1: [gone]})
    assert isinstance(d, P.Ask) and d.name.startswith("blocked_failed:rs1:")
    assert "provider_refused" in d.question


def test_blocked_segments_go_first_then_the_transient_one_is_retried():
    sim = Sim()
    run = _blocked_run("bm" * 32, blocked=(1,), transient=(3,))
    d = _bclean(sim, run)
    assert isinstance(d, P.Launch) and d.params["run_segment_id"] == "rs1"
    # The blocked one got its fill (has a result now); the transient one remains.
    segs = [dict(sg) for sg in run.segments]
    segs[1].update(status="completed", has_result=True, content_block=None,
                   local_result_path="/fills/f1/composite.mp4", seedance_task_id=None)
    run2 = P.RunInfo(id=run.id, status="incomplete", version="bn" * 32, segments=segs)
    d = _bclean(sim, run2)
    assert isinstance(d, P.Ask) and d.name == f"retry:{P.short(run2.version, 12)}"
    assert d.options[0].handler == P.H_RETRY_RUN and "VOID fill from the source" in d.question


def test_an_audio_block_says_the_original_audio_ships():
    run = _blocked_run("ba" * 32)
    segs = [dict(sg) for sg in run.segments]
    segs[1].update(content_block="sensitive_audio",
                   error_message="failed after 3 attempt(s): InputAudioSensitiveContentDetected")
    run = P.RunInfo(id=run.id, status="incomplete", version=run.version, segments=segs)
    d = _bclean(Sim(), run, {1: [_sjob("f1")]})
    assert isinstance(d, P.Ask) and "original soundtrack" in d.question


def test_every_segment_finished_but_incomplete_asks_the_operator():
    run = _blocked_run("bu" * 32, blocked=())
    d = _bclean(Sim(), run)
    assert isinstance(d, P.Ask) and d.name.startswith("stuck:") and d.kind == "handoff"


def test_a_source_composite_gets_its_eye_check_without_an_omni_rerun(env):
    sim = Sim()
    fill1 = {"job_id": "f1", "mask_box": [0.3, 0.58, 0.7, 0.66], "delivery": "delivered",
             "replaced_take_id": None, "basis": "source"}
    jobs = {1: [_job("f1", "applied", provider_task_id="t1", basis="source")]}
    run = _run_info("se" * 32, results={1: "/fills/f1/composite.mp4"})
    _qa_round(env, sim, run, _static(49.2, DURATION), fills={1: fill1}, jobs=jobs)
    card = _via_eyes(sim, run, _snapshot(jobs), "per_segment")
    assert "rebuilt from the original (the provider refused it)" in card.question
    d = _clean(sim, run, _snapshot(jobs))
    assert isinstance(d, P.Ask) and d.name == "eye:f1"
    assert "rebuilt from the ORIGINAL video" in d.question and d.proposal["basis"] == "source"
    assert "omni_rerun" not in [o.id for o in d.options]
    # ... and a dirty one gets its grown retry, then (already a retry) the stop
    # rule, still without an Omni rerun.
    final = _captions(6.5, 7.5)
    final.update(_static(49.2, DURATION))
    sim1 = Sim()
    _qa_round(env, sim1, run, final, fills={1: fill1}, jobs=jobs)
    d = _clean(sim1, run, _snapshot(jobs))
    assert isinstance(d, P.Launch) and d.name.startswith("grow:rs1:f1")
    sim2 = Sim()
    _qa_round(env, sim2, run, final, fills={1: dict(fill1, retry_of="f0")}, jobs=jobs)
    d = _clean(sim2, run, _snapshot(jobs))
    assert isinstance(d, P.Ask) and d.name == "stop:f1"
    assert "omni_rerun" not in [o.id for o in d.options]


def test_a_segment_gate_that_waiting_cannot_change_asks_the_operator():
    run = _blocked_run("bg" * 32)
    snap_jobs = None
    sim = Sim()
    snap = _snapshot(snap_jobs)
    snap["context"]["segments"]["rs1"]["blockers"] = [
        "pre/post roll segments are not supported by the fill"]
    d = sim.record(P.STEP_CLEAN, P.plan_clean(
        run=run, detector_config=None, effects=sim.eff(P.STEP_CLEAN),
        answers=sim.ans(P.STEP_CLEAN), snapshot=snap, project_id="p1",
        detector_version="det-v", usd_per_credit=0.005))
    assert isinstance(d, P.Ask) and d.name.startswith("blocked_failed:rs1:")
    assert "pre/post roll" in d.question
    # A run-level conflict (an audio patch, a range attempt ...) is waited out.
    d = _bclean(Sim(), run, run_blockers=("a range regeneration is in progress",))
    assert isinstance(d, P.Wait) and "range regeneration" in d.reason


def test_gate_the_retry_option_names_every_segment_it_resubmits():
    run = _blocked_run("gr" * 32, blocked=(1,), transient=(3,))
    d = _bclean(Sim(), run, {1: [_sjob("f1")]})
    assert isinstance(d, P.Ask) and d.name == "blocked:f1"
    assert "[1, 3]" in d.options[1].label and d.options[1].handler == P.H_RETRY_RUN
    d = _bclean(Sim(), run, {1: [_sjob("x0", "failed", error_code="segment_too_short")]})
    assert d.name.startswith("blocked_failed:") and "[1, 3]" in d.question
    assert "[1, 3]" in d.options[0].label


def test_gate_r2_a_failed_run_with_a_preprocess_hole_goes_to_the_operator():
    base = _run_info("ph" * 32, status="failed")
    run = P.RunInfo(id=base.id, status="failed", version=base.version, segments=base.segments,
                    error_message="submit failed", preprocess_hole=True)
    d = _clean(Sim(), run)
    assert isinstance(d, P.Ask) and d.name.startswith("stuck:") and d.kind == "handoff"


def test_gate_r1_2_a_rolled_run_is_never_restitched_its_shifted_segment_is_rebuilt(
        env, monkeypatch):
    sim = Sim()
    run = _run_info("rr" * 32)
    monkeypatch.setattr(qa_tools, "alignment_check", lambda s, r, **k: {
        "in_sync": False, "points": 100, "max_abs_window_median": 6.0,
        "frame_count_delta": 0, "length_ok": True, "windows": [],
        "_points": _points(DURATION, {(BOUNDS[2], BOUNDS[3]): -6})})
    env["scans"]["src"] = _source_scan()
    env["durations"].update(src=DURATION, finx=DURATION)
    env["scans"]["finx"] = _scan(_static(49.2, DURATION), DURATION)
    d = _clean(sim, run)
    assert isinstance(d, P.Compute)
    res = _qa_compute(env, "finx", run.version)
    res["segments_input"][0]["pre_roll_sec"] = 0.5
    sim.add(P.STEP_CLEAN, d.name, "compute", d.func, "succeeded", result=res)
    d = _clean(sim, run)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == "rs2" and "out of sync" in d.reason


def test_gate_r2_1_a_restitch_over_an_audio_patch_is_asked_not_refused_forever(
        env, monkeypatch):
    sim = Sim()
    run = P.RunInfo(**{**_run_info("ap" * 32).__dict__, "audio_patch": True})
    monkeypatch.setattr(qa_tools, "alignment_check", lambda s, r, **k: {
        "in_sync": False, "points": 100, "max_abs_window_median": 6.0,
        "frame_count_delta": 0, "length_ok": True, "windows": [],
        "_points": _points(DURATION, {(BOUNDS[2], DURATION): -6})})
    _qa_round(env, sim, run, _static(49.2, DURATION))
    d = _clean(sim, run)
    assert isinstance(d, P.Ask) and d.name == f"restitch_patch:{'ap' * 6}"
    assert [o.id for o in d.options] == ["restitch", "accept", "manual"]
    assert "audio fix" in d.question
    sim.answer(P.STEP_CLEAN, d.name, option="restitch")
    d = _clean(sim, run)
    assert isinstance(d, P.Launch) and d.handler == P.H_RESTITCH
    assert d.params == {"run_id": "run2845", "discard_audio_patch": True}
    # "ship as is" accepts the timing instead
    sim2 = Sim()
    _qa_round(env, sim2, run, _static(49.2, DURATION))
    d = _clean(sim2, run)
    sim2.answer(P.STEP_CLEAN, d.name, option="accept")
    assert isinstance(_clean(sim2, run), P.Complete)


def test_gate_r3_1_a_void_rebuild_over_an_audio_patch_says_so_and_carries_the_discard(env):
    sim = Sim()
    run = P.RunInfo(**{**_run_info("aq" * 32).__dict__, "audio_patch": True})
    final = _captions(23.5, 28.2)
    final.update(_static(49.2, DURATION))
    _qa_round(env, sim, run, final, jobs={4: [_job("f4")]})
    # over an audio patch the batch is never started without a question
    d = _via_batch(sim, run, _snapshot({4: [_job("f4")]}), None)
    assert "audio fix" in d.question and "drops the audio fix" in d.options[0].label
    assert d.options[0].params["discard_audio_patch"] is True
    sim.answer(P.STEP_CLEAN, d.name, option="start")
    d = _clean(sim, run, _snapshot({4: [_job("f4")]}))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START
    assert d.params["discard_audio_patch"] is True           # the same action the ask showed


def test_gate_r4_over_an_audio_patch_legacy_starts_are_asked_again_and_omni_choices_discard(
        env):
    sim = Sim()
    run = P.RunInfo(**{**_run_info("ar" * 32).__dict__, "audio_patch": True})
    sim.ans(P.STEP_CLEAN)["start:f4"] = P.Answer("start:f4", "approved", "approve", "start")
    final = _captions(23.5, 28.2)
    final.update(_static(49.2, DURATION))
    _qa_round(env, sim, run, final, jobs={4: [_job("f4")]})
    d = _via_batch(sim, run, _snapshot({4: [_job("f4")]}), None)   # not the legacy start's launch
    assert d.options[0].params["discard_audio_patch"] is True
    # an Omni choice from a no-fix question carries the discard too (and the question says so)
    sim2 = Sim()
    jobs = {4: [_job("f4", "failed", provider_task_id="t4", error_code="provider_failed")]}
    _qa_round(env, sim2, run, _captions(23.5, 24.5) | _static(49.2, DURATION), jobs=jobs)
    d = _clean(sim2, run, _snapshot(jobs))
    assert d.name.startswith("hand:rs4:") and "audio fix" in d.question
    sim2.answer(P.STEP_CLEAN, d.name, option="omni_rerun")
    d = _clean(sim2, run, _snapshot(jobs))
    assert isinstance(d, P.Launch) and d.handler == P.H_RERUN
    assert d.params["discard_audio_patch"] is True


# ---------------------------------------------------------------------------
# VOID from the source as the run's method (2026-11-03, design §7.3)
# ---------------------------------------------------------------------------

def _void_run(has=(), fallback=(), failed=()):
    return P.RunInfo(id="runV", status="incomplete", version="vv" * 32, void_source=True,
                     segments=[{"id": f"rs{i}", "index": i,
                                "status": ("completed" if i in has else
                                           "failed" if i in failed else "pending"),
                                "local_result_path": None, "seedance_task_id": None,
                                "result_identity": None, "has_result": i in has,
                                "model_fallback": i in fallback, "content_block": None,
                                "error_message": "Omni failed" if i in failed else None,
                                "start_sec": 2.5 * i, "end_sec": 2.5 * (i + 1)}
                               for i in range(3)])


def _vjob(fid, rsid, status="ready", **kw):
    return dict(_job(fid, status, basis="source", run_segment_id=rsid, **kw))


def _vsnap(jobs):
    return {"context": {"run_blockers": [], "active_fill": None,
                        "segments": {f"rs{i}": {"blockers": [], "jobs": [
                            j for j in jobs if j["run_segment_id"] == f"rs{i}"]}
                            for i in range(3)}},
            "takes": {}}


def _vclean(sim, run, snap, allowance=1.0, spent=0.0):
    return sim.record(P.STEP_CLEAN, P.plan_clean(
        run=run, detector_config=None, effects=sim.eff(P.STEP_CLEAN),
        answers=sim.ans(P.STEP_CLEAN), snapshot=snap, project_id="pV", detector_version="dv",
        usd_per_credit=0.005, void_allowance_usd=allowance, void_auto_spent_usd=spent,
        auto_accept=True))


def test_void_source_run_previews_in_parallel_as_is_free_then_one_batch():
    sim = Sim()
    run = _void_run()
    jobs = []
    for i in range(3):
        d = _vclean(sim, run, _vsnap(jobs))
        assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
        assert d.params == {"run_id": "runV", "run_segment_id": f"rs{i}", "mask_box": None,
                            "prompt": None, "basis": "source"}
        sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched",
                refs={"fill_jobs": [f"f{i}"]})
        jobs.append(_vjob(f"f{i}", f"rs{i}", "preparing"))
    d = _vclean(sim, run, _vsnap(jobs))
    assert isinstance(d, P.Wait) and "being prepared" in d.reason
    ready = [_vjob("f0", "rs0"), _vjob("f1", "rs1"),
             _vjob("f2", "rs2", source_as_is=True, estimated_usd=0.0)]
    d = _vclean(sim, run, _vsnap(ready))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_APPLY_SOURCE      # free, no ask
    assert d.params == {"run_id": "runV", "fill_id": "f2"}
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched")
    run2 = _void_run(has=(2,))
    snap = _vsnap(ready[:2] + [_vjob("f2", "rs2", "applied", source_as_is=True)])
    d = _vclean(sim, run2, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START and not d.regeneration
    assert d.params == {"run_id": "runV", "fill_ids": ["f0", "f1"], "approved_usd": 0.2,
                        "discard_audio_patch": False, "authorization": "auto"}
    assert not [x for _s, x in sim.decisions if isinstance(x, P.Ask)]
    # without an allowance: ONE question, never a model call
    sim2 = Sim()
    d = _vclean(sim2, run2, snap, allowance=0.0)
    assert isinstance(d, P.Ask) and d.name.startswith("vbatch:") and d.kind == "regeneration"
    assert "nothing goes to Omni" in d.question and d.cost_credits == 40.0
    assert [o.id for o in d.options] == ["start", "manual"]
    sim2.answer(P.STEP_CLEAN, d.name, option="start")
    d = _vclean(sim2, run2, snap, allowance=0.0)
    assert isinstance(d, P.Launch) and d.regeneration and d.params["authorization"] == "operator"


def test_void_source_segment_void_cannot_take_asks_omni_never_calls_it_alone():
    sim = Sim()
    run = _void_run(has=(1, 2))
    sim.add(P.STEP_CLEAN, "vpreview:rs0:r1", "action", P.H_FILL_PREVIEW, "launched")
    snap = _vsnap([_vjob("f0", "rs0", "failed", error_code="segment_too_short",
                         error_message="the segment has 60 frames; VOID needs at least 69")])
    d = _vclean(sim, run, snap)
    assert isinstance(d, P.Ask) and d.name.startswith("vsrc_failed:rs0:")
    assert [o.id for o in d.options] == ["omni", "manual"] and "VOID needs" in d.question
    assert isinstance(_vclean(sim, run, snap), P.Wait)
    sim.answer(P.STEP_CLEAN, d.name, option="omni")
    d = _vclean(sim, run, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_RERUN and d.regeneration
    assert d.params["segments"] == [{"run_segment_id": "rs0", "prompt": None}]
    # a failed model fallback is asked again (Omni / back to VOID / by hand)
    sim3 = Sim()
    d = _vclean(sim3, _void_run(has=(1, 2), fallback=(0,), failed=(0,)), _vsnap([]))
    assert isinstance(d, P.Ask) and d.name.startswith("vfallback_failed:rs0:")
    assert [o.id for o in d.options] == ["omni", "void", "manual"]


def test_auto_accept_needs_screen_evidence():
    row = {"index": 3, "verdict": "clean", "result_hits": 0, "fill_needs_visual_check": True,
           "fill": {"job_id": "f3", "mask_box": [0.3, 0.74, 0.7, 0.82], "delivery": "delivered"}}
    faces = {"3": {"samples": 10, "boxes": [[0.3, 0.1, 0.6, 0.4]]}}
    qa = {"segments": [row]}
    assert P.auto_accepted_fills(qa, faces, None) == {}                       # unmeasured
    assert P.auto_accepted_fills(qa, faces, {"3": {"samples": 8, "screen_frames": 2}}) == {}
    assert P.auto_accepted_fills(qa, faces, {"3": {"samples": 8, "error": "x"}}) == {}
    ok = P.auto_accepted_fills(qa, faces, {"3": {"samples": 8, "screen_frames": 0,
                                                 "max_share": 0.2}})
    assert list(ok) == ["f3"]


def test_gate_r1_1_back_to_void_after_a_failed_fallback_reaches_the_batch():
    sim = Sim()
    run = _void_run(has=(1, 2), fallback=(0,), failed=(0,))
    d = _vclean(sim, run, _vsnap([]))
    assert d.name.startswith("vfallback_failed:rs0:")
    sim.answer(P.STEP_CLEAN, d.name, option="void")
    d = _vclean(sim, run, _vsnap([]))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["basis"] == "source" and d.params["run_segment_id"] == "rs0"
    sim.add(P.STEP_CLEAN, d.name, "action", d.handler, "launched", refs={"fill_jobs": ["g0"]})
    d = _vclean(sim, run, _vsnap([_vjob("g0", "rs0")]))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START
    assert d.params["fill_ids"] == ["g0"]
