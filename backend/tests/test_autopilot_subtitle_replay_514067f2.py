"""Replay of the live autopilot e6fd6ac7 (project 514067f2, run d8105ab5) on the
2026-11-02 planner, from its real data (tests/fixtures/autopilot_514067f2.json):

* the stored caption QA (effect b012d010) and the step's real answers -- the
  one-option "inconclusive" card answered accept, the eye check of segment 1
  accepted, and "manual" on the two no-fix questions of segments 3 and 4;
* the per-segment QA (this branch's ``post_run_qa``) of the delivered final
  (every segment after the first late by the frames the Omni clips ran over
  their slots: 0, 3, 2, 5.5, 8, 9) and of the same clips re-stitched with the
  slot conform (every segment in sync; leftovers in 3 and 4).

What the operator sees after the deploy, without a restart: a new QA, a FREE
re-stitch, a QA on the resynced final, then the VOID rebuild of segment 3 as
the leading option (segment 4 next in line), then segment 4's."""
from __future__ import annotations

import copy
import json
import os

from app import subtitle_loop as sl
from app.autopilot.flows import subtitle_removal_plan as P

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures",
                                  "autopilot_514067f2.json")))
STORED = FIX["qa_effect_b012d010"]
RID = "d8105ab5-b325-4995-bca0-8585a7e9f1e1"
PID = "514067f2-0af3-4396-baba-e1687de0abf9"
CFG = {"band_y": [0.73, 0.83]}
RV1 = STORED["run_version"]
RV2, RV3 = "e2" * 32, "e3" * 32
RS = {s["index"]: s["run_segment_id"] for s in STORED["segments_input"]}
C = P.STEP_CLEAN


class Sim:
    def __init__(self):
        self.effects, self.answers, self.decisions, self.seq = [], {}, [], 0

    def add(self, name, kind, handler, state, *, params=None, refs=None, result=None):
        self.seq += 1
        e = P.EffectInfo(id=f"{self.seq:08d}-{name.replace(':', '-')}"[:36], name=name,
                         kind=kind, handler=handler, state=state, params=params or {},
                         created_refs=refs or {}, result=result, seq=self.seq)
        self.effects.append(e)
        return e

    def record(self, d):
        self.decisions.append(d)
        if isinstance(d, (P.Launch, P.Complete, P.Fail)):
            for n in d.consumes:
                a = self.answers[n]
                self.answers[n] = P.Answer(n, "consumed", a.decision, a.chosen_option,
                                           members=a.members)
            assert not P.unconsumed(self.answers)
        if isinstance(d, (P.Ask, P.Wait)):
            for n in tuple(d.retire) + tuple(getattr(d, "replaces", ())):
                if n in self.answers and self.answers[n].status == "pending":
                    self.answers[n] = P.Answer(n, "superseded")
        if isinstance(d, P.Ask):
            self.answers[d.name] = P.Answer(d.name, "pending", members=tuple(
                (d.proposal or {}).get("members") or ()))
        return d

    def answer(self, name, option):
        assert self.answers[name].status == "pending"
        self.answers[name] = P.Answer(name, "approved", "approve", option,
                                      members=self.answers[name].members)


def _run(version, composite3=None):
    segs = []
    for s in STORED["segments_input"]:
        rsid = s["run_segment_id"]
        res = STORED["results"][rsid]
        path, task = res["result_path"], res["seedance_task_id"]
        ident = FIX["identities"][rsid]
        if s["index"] == 3 and composite3:
            path, task, ident = composite3, None, [64513, 1, 2, 3]
        segs.append({"id": rsid, "index": s["index"], "status": "completed",
                     "local_result_path": path, "seedance_task_id": task,
                     "result_identity": ident, "has_result": True})
    return P.RunInfo(id=RID, status="done", version=version, segments=segs)


def _job(fid, rsid, status="ready", **kw):
    out = {"job_id": fid, "status": status, "box_source": "measured", "basis": "model_result",
           "mask_box": [0.3107, 0.7394, 0.6874, 0.819], "default_prompt": True,
           "estimated_usd": 0.10, "provider_task_id": None, "error_code": None,
           "stale": None, "frames_masked": 120, "run_segment_id": rsid,
           "preview_path": f"/app/data/projects/{PID}/runs/{RID}/segment_fills/{fid}/prep/"
                           "preview.jpg",
           "preview_url": f"/api/v2/runs/{RID}/segments/fill/{fid}/preview.jpg"}
    out.update(kw)
    return out


SEG1_JOB = _job("a77c195e-77a4-4ae5-9c6a-fe1a7cb67b0b", RS[1], "applied", basis="source",
                provider_task_id="t1")


def _snapshot(jobs=()):
    by = {rsid: [] for rsid in RS.values()}
    by[RS[1]].append(SEG1_JOB)
    for j in jobs:
        by[j["run_segment_id"]].insert(0, j)
    return {"context": {"run_blockers": [], "active_fill": None,
                        "segments": {k: {"blockers": [], "jobs": v} for k, v in by.items()}},
            "takes": {}, "source_identity": STORED["source_identity"]}


def _plan(sim, run, snapshot):
    return sim.record(P.plan_clean(run=run, detector_config=CFG, effects=sim.effects,
                                   answers=sim.answers, snapshot=snapshot, project_id=PID,
                                   detector_version="dv", usd_per_credit=0.005))


def _qa_result(qa, version, segments_input, snapshot, clips=None):
    segs = sl.segments_from(segments_input)
    qa = copy.deepcopy(qa)
    qa["timing"]["clips"] = copy.deepcopy(clips or FIX["clips"])
    cli = {"qa": qa,
           "fill": sl.fill_proposal(RID, qa, segs, context=snapshot["context"]),
           "operator_options": {"omni_rerun": sl.rerun_proposal(
               RID, qa, segs, takes={}, price_per_segment=168.0)},
           "handoff": sl.handoff_proposal(RID, qa, segs)}
    out = P.qa_result(cli, segments_input=segments_input, run_version=version,
                      run_version_after=version, detector_config=CFG,
                      results=STORED["results"], sheets={},
                      source_identity=STORED["source_identity"])
    out["source_fps"] = 30.0
    out["out_dir"] = os.path.dirname(qa["comparison_sheet"])
    return out


def _live_state():
    """The clean step of e6fd6ac7 as it stands (prod, 2026-11-02 18:20)."""
    sim = Sim()
    sim.add("do:retry:0847b1bc2271:r1", "action", P.H_RETRY_RUN, "launched",
            params={"run_id": RID})
    sim.add(f"bpreview:{RS[1]}:da343958d4bd:r1", "action", P.H_FILL_PREVIEW, "launched",
            refs={"fill_jobs": [SEG1_JOB["job_id"]]})
    sim.add(f"do:blocked:{SEG1_JOB['job_id']}:r1", "action", P.H_FILL_START, "launched",
            refs={"fill_jobs": [SEG1_JOB["job_id"]], "provider_tasks": ["t1"]})
    sim.add("qa:d2ccb449f2ae:62b36e1e:r1", "compute", P.C_QA, "succeeded", result=STORED)
    for name, status, decision, option in FIX["answers"]:
        sim.answers[name] = P.Answer(name, status, decision, option)
    return sim


def test_the_detector_digest_and_the_live_tokens_match_production():
    assert P.detector_sha(CFG) == STORED["detector_sha"]
    run = _run(RV1)
    rows = {r["index"]: r for r in STORED["qa"]["segments"]}
    src = STORED["source_identity"]
    assert P.result_token(run, rows[3], src) == "ec33ffbf"
    assert P.result_token(run, rows[4], src) == "f14a1488"


def test_replay_e6fd6ac7_after_the_deploy():
    sim = _live_state()
    run = _run(RV1)
    snap = _snapshot()

    # 1. the stored QA has no per-segment timing -> a new QA (free)
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Compute) and d.func == P.C_QA
    assert d.name == "qa:d2ccb449f2ae:62b36e1e:r2"
    sim.add(d.name, "compute", P.C_QA, "succeeded",
            result=_qa_result(FIX["qa_delivered"], RV1, STORED["segments_input"], snap))

    # 2. segments 1-5 late by the frames the Omni clips ran over -> a FREE re-stitch,
    #    no question; the old answers are consumed (they stay readable)
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_RESTITCH and not d.regeneration
    assert d.params == {"run_id": RID} and d.expect == {f"run:{RID}": RV1}
    assert set(d.consumes) == {"inconclusive:b012d010",
                               "eye:a77c195e-77a4-4ae5-9c6a-fe1a7cb67b0b",
                               f"operator:{RS[3]}:ec33ffbf", f"operator:{RS[4]}:f14a1488"}
    assert "segments 1, 2, 3, 4 and 5" in d.reason and "10 frames" in d.reason
    assert "longer than their slots" in d.reason
    sim.add(d.name, "action", P.H_RESTITCH, "launched", params=d.params)

    # 3. the re-stitched final: QA again, every segment in sync, leftovers in 3 and 4
    run = _run(RV2)
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Compute) and d.name.startswith("qa:e2e2e2e2e2e2:62b36e1e:")
    sim.add(d.name, "compute", P.C_QA, "succeeded",
            result=_qa_result(FIX["qa_restitched"], RV2, STORED["segments_input"], snap))
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == RS[3] and d.params["mask_box"] is None
    sim.add(d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f3"]})

    # 4. 2026-11-03: segment 4's preview is made right away (previews prepare in
    #    parallel, each bound to its own segment); then ONE batch question
    d = _plan(sim, run, _snapshot([_job("f3", RS[3], "preparing")]))
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == RS[4]
    sim.add(d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f4"]})
    snap34 = _snapshot([_job("f3", RS[3]),
                        _job("f4", RS[4], mask_box=[0.3181, 0.7394, 0.6819, 0.82])])
    d = _plan(sim, run, snap34)
    assert isinstance(d, P.Compute) and d.func == P.C_BATCH_SHEET
    sim.add(d.name, "compute", d.func, "succeeded",
            result={"sheet": "/att/b/batch_sheet.jpg", "out_dir": "/att/b"})
    ask = _plan(sim, run, snap34)
    assert isinstance(ask, P.Ask) and ask.kind == "regeneration" and ask.name.startswith(
        "vbatch:e2e2e2e2e2e2:")
    assert [o.id for o in ask.options] == ["start", "per_segment", "accept", "manual"]
    assert ask.options[0].handler == P.H_FILL_BATCH_START and ask.cost_credits == 40.0
    assert ask.question.startswith("Segments 3 and 4 still show burned-in captions: rebuild "
                                   "all 2 in one batch with VOID from the ORIGINAL video "
                                   "($0.20 reserved, about $0.10 charged).")
    assert "Segment 3 (19.1-26.9 s): caption words in" in ask.question
    assert sorted(ask.proposal["members"]) == sorted([f"{RS[3]}:ec33ffbf", f"{RS[4]}:f14a1488"])
    for word in ("timeline", "calibrat", "alignment", "actionable"):
        assert word not in ask.question.lower()

    # "I fix them by hand" holds both segments ... and "Ask again" brings it back
    sim.answer(ask.name, "manual")
    d = _plan(sim, run, snap34)
    assert isinstance(d, P.Wait) and "fixes segments 3 and 4 by hand" in d.reason
    sim.answers[ask.name] = P.Answer(ask.name, "superseded")         # core.reopen
    d = _plan(sim, run, snap34)
    assert isinstance(d, P.Ask) and d.name == ask.name
    sim.answer(ask.name, "start")
    d = _plan(sim, run, snap34)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START and d.regeneration
    assert d.params == {"run_id": RID, "fill_ids": ["f3", "f4"], "approved_usd": 0.2,
                        "discard_audio_patch": False, "authorization": "operator"}
    sim.add(d.name, "action", d.handler, "launched",
            refs={"fill_batches": ["b1"], "fill_jobs": ["f3", "f4"]})

    # 5. both rebuilt in ONE promote and re-delivered: ONE eye check for both
    comp3 = f"/app/data/projects/{PID}/runs/{RID}/segment_fills/f3/comp-x/composite.mp4"
    run = _run(RV3, composite3=comp3)
    segs = [dict(x) for x in run.segments]
    segs[4].update(local_result_path=f"/app/data/projects/{PID}/runs/{RID}/segment_fills/f4/"
                   "comp-y/composite.mp4", seedance_task_id=None,
                   result_identity=[64513, 4, 5, 6])
    run = P.RunInfo(id=RID, status="done", version=RV3, segments=segs)
    seg_in = copy.deepcopy(STORED["segments_input"])
    qa3 = copy.deepcopy(FIX["qa_restitched"])
    for i, fid in ((3, "f3"), (4, "f4")):
        fill = {"job_id": fid, "mask_box": [0.3107, 0.7394, 0.6874, 0.819],
                "frames_masked": 120, "basis": "model_result", "source_as_is": False,
                "delivery": "delivered", "replaced_take_id": f"take{i}", "batch_id": "b1"}
        seg_in[i].update(result_kind="void_fill", fill=fill)
        row = qa3["segments"][i]
        row.update(verdict="clean", result_hits=0, leftovers=[], result_kind="void_fill",
                   fill=dict(fill, confirmed=False), fill_needs_visual_check=True,
                   needs_visual_check=True, fill_sheet=row["crop_sheet"])
    qa3["dirty_segments"] = []
    qa3["unconfirmed_fills"] = [{"index": 1, "fill_job_id": SEG1_JOB["job_id"]},
                                {"index": 3, "fill_job_id": "f3"},
                                {"index": 4, "fill_job_id": "f4"}]
    applied = [_job("f3", RS[3], "applied", provider_task_id="t3", batch_id="b1"),
               _job("f4", RS[4], "applied", provider_task_id="t4", batch_id="b1")]
    snap = _snapshot(applied)
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Compute)
    sim.add(d.name, "compute", P.C_QA, "succeeded", result=_qa_result(qa3, RV3, seg_in, snap))
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Ask) and d.name.startswith("eyes:") and d.kind == "handoff"
    assert d.proposal["members"] == ["f3", "f4"]           # segment 1 was accepted before
    sim.answer(d.name, "accept_all")
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Complete)
    assert set(d.output["accepted"]["accepted_fills"]) >= {"f3", "f4"}


def test_operator_texts_for_the_report(capsys):
    """Not an assertion of wording: prints the two questions (pytest -s)."""
    sim = _live_state()
    run = _run(RV1)
    snap = _snapshot()
    _plan(sim, run, snap)
    sim.add("qa:d2ccb449f2ae:62b36e1e:r2", "compute", P.C_QA, "succeeded",
            result=_qa_result(FIX["qa_delivered"], RV1, STORED["segments_input"], snap))
    d = _plan(sim, run, snap)
    print("RESTITCH:", d.reason)


SEG4_FILL = "3beedbec-2b08-450e-9234-52afeec388e4"
SEG5_PREVIEW = "925c657f-5f81-41f8-9eb8-a2feedada1a7"
# segment 4 is the operator's VOID composite now: exactly its slot
CLIPS_NOW = [dict(c, result_frames=c["target_frames"], excess_frames=0) if c["index"] == 4
             else c for c in FIX["clips"]]


def test_replay_e6fd6ac7_as_it_stands_after_the_operators_manual_fill():
    """The live state at 19:30: after the answers above the operator rebuilt
    segment 4 by hand (VOID, 18:45 -- a manual paid action, so the autopilot is
    PAUSED) and made a preview for segment 5 (ready). The re-stitched final
    still runs late from segment 1 on: the hand-made composite is exact, but
    the long Omni clips before it are not. After the deploy and Resume: QA,
    the free re-stitch, then segment 3's fix question; the hand-made
    composite of segment 4 gets its eye check; segment 5's preview is left
    alone (it goes stale with the re-stitch)."""
    sim = _live_state()
    seg_in = FIX["segments_input_current"]
    comp4 = seg_in[4]["fill"]
    rv_now, rv_after = "c4" * 32, "c5" * 32

    def run_now(version):
        r = _run(version)
        segs = [dict(s) for s in r.segments]
        segs[4].update(local_result_path=f"/app/data/projects/{PID}/runs/{RID}/segment_fills/"
                       f"{SEG4_FILL}/comp-755b186540324361a845e7189d77d9a8/composite.mp4",
                       seedance_task_id=None, result_identity=[64513, 9, 9, 9])
        return P.RunInfo(id=RID, status="done", version=version, segments=segs)

    jobs = [_job(SEG4_FILL, RS[4], "applied", provider_task_id="t4m",
                 mask_box=comp4["mask_box"]),
            _job(SEG5_PREVIEW, RS[5], mask_box=[0.2681, 0.7394, 0.7263, 0.82])]
    snap = _snapshot(jobs)
    run = run_now(rv_now)
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Compute) and d.name == "qa:c4c4c4c4c4c4:62b36e1e:r1"
    sim.add(d.name, "compute", P.C_QA, "succeeded",
            result=_qa_result(FIX["qa_current"], rv_now, seg_in, snap, CLIPS_NOW))
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_RESTITCH
    sim.add(d.name, "action", P.H_RESTITCH, "launched", params=d.params)
    run = run_now(rv_after)
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Compute)
    sim.add(d.name, "compute", P.C_QA, "succeeded",
            result=_qa_result(FIX["qa_current_restitched"], rv_after, seg_in, snap, CLIPS_NOW))
    d = _plan(sim, run, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW
    assert d.params["run_segment_id"] == RS[3]
    sim.add(d.name, "action", d.handler, "launched", refs={"fill_jobs": ["f3"]})
    snap3 = _snapshot(jobs + [_job("f3", RS[3])])
    d = _plan(sim, run, snap3)
    if isinstance(d, P.Compute):
        sim.add(d.name, "compute", d.func, "succeeded", result={"sheet": None})
        d = _plan(sim, run, snap3)
    assert isinstance(d, P.Ask) and d.name.startswith("vbatch:")
    assert 3 in d.proposal["segments"]
    sim.answer(d.name, "accept")                        # (to reach the eye check here)
    eye = _plan(sim, run, snap3)
    assert isinstance(eye, P.Ask) and eye.name.startswith("eyes:")
    assert eye.proposal["members"] == [SEG4_FILL]
    assert eye.question.startswith("Please check segment 4 by eye: VOID repainted")
