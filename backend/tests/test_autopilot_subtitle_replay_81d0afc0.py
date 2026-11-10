"""Replay of the live autopilot ff218831 (project 81d0afc0, run d46bb1b8) on the
2026-11-03 planner (batch VOID fills, docs/segment-fill-batch-design.md), from its
real data (tests/fixtures/autopilot_81d0afc0.json, read-only prod queries):

* the caption QA after the free re-stitch (effect c858675b): Omni left captions in
  segments 0, 2, 3, 4, 5, 7 and 10; segment 6 is a VOID composite that still shows
  words; segments 8 and 9 are detector-clean VOID composites;
* the step's answers as they stood at 13:44 -- six ``operator:`` cards, two
  ``stop:`` cards and one ``eye:`` card still PENDING from before the re-stitch
  (the operator answered them only at 19:20);
* the fill history (three applied composites, one canceled stale preview).

What the old planner did: one preview, one ``void:`` question, one fill, one
re-stitch and re-delivery per segment, then an eye check per composite -- an
afternoon of clicks. What this planner does: every preview at once (free), the
grown-mask retry of segment 6 (free preview), the stale cards retired, then ONE
batch start of all eight without a question (inside the $1.00 allowance) -- or,
with no allowance, ONE question for all of them."""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta

from app.autopilot.flows import subtitle_removal_plan as P

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures",
                                  "autopilot_81d0afc0.json")))
QA = FIX["qa_effect_c858675b"]
RID, PID = FIX["run_id"], FIX["project_id"]
RV = QA["run_version"]
RS = {s["index"]: s["run_segment_id"] for s in QA["segments_input"]}
IDX = {v: k for k, v in RS.items()}
NOW = datetime(2026, 11, 3, 13, 45)


class Sim:
    def __init__(self):
        self.effects, self.answers, self.decisions, self.seq = [], {}, [], 0
        for name, status, decision, option in FIX["answers"]:
            self.answers[name] = P.Answer(name, status, decision, option)
        # the free re-stitch of 13:38 (effect 3ada56db) ran before this QA
        self.add("restitch:f7eed00a8dd5:r1", "action", P.H_RESTITCH, "launched",
                 params={"run_id": RID})
        self.add("qa:b3e49575374d:74234e98:r1", "compute", P.C_QA, "succeeded", result=QA)

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


def _run():
    segs = []
    for s in QA["segments_input"]:
        rsid = s["run_segment_id"]
        res = QA["results"][rsid]
        segs.append({"id": rsid, "index": s["index"], "status": "completed",
                     "local_result_path": res["result_path"],
                     "seedance_task_id": res["seedance_task_id"],
                     "result_identity": FIX["identities"][rsid], "has_result": True})
    return P.RunInfo(id=RID, status="done", version=RV, segments=segs)


def _job(fid, rsid, status="ready", **kw):
    out = {"job_id": fid, "status": status, "box_source": "measured", "basis": "model_result",
           "mask_box": [0.25, 0.65, 0.75, 0.73], "default_prompt": True, "estimated_usd": 0.10,
           "provider_task_id": None, "error_code": None, "stale": None, "frames_masked": 90,
           "run_segment_id": rsid, "segment_bound": True,
           "created_at": (NOW - timedelta(minutes=2)).isoformat(),
           "preview_path": f"/app/data/projects/{PID}/runs/{RID}/segment_fills/{fid}/prep-x/"
                           "preview.jpg",
           "preview_url": f"/api/v2/runs/{RID}/segments/fill/{fid}/preview.jpg"}
    out.update(kw)
    return out


def _snapshot(extra=()):
    by = {rsid: [] for rsid in RS.values()}
    for j in FIX["jobs"]:
        by[j["run_segment_id"]].append(_job(j["job_id"], j["run_segment_id"], **{
            k: v for k, v in j.items() if k not in ("job_id", "run_segment_id")}))
    for j in extra:
        by[j["run_segment_id"]].insert(0, j)
    return {"context": {"run_blockers": [], "active_fill": None,
                        "segments": {k: {"blockers": [], "jobs": v} for k, v in by.items()}},
            "takes": {}, "source_identity": QA["source_identity"]}


def _plan(sim, snap, allowance=1.0, spent=0.0):
    return sim.record(P.plan_clean(run=_run(), detector_config=None, effects=sim.effects,
                                   answers=sim.answers, snapshot=snap, project_id=PID,
                                   detector_version="dv", usd_per_credit=0.005,
                                   void_allowance_usd=allowance, void_auto_spent_usd=spent,
                                   auto_accept=True, now=NOW))


def test_the_fixture_is_the_live_state():
    assert P.detector_sha(None) == QA["detector_sha"]
    rows = {r["index"]: r for r in QA["qa"]["segments"]}
    assert QA["qa"]["dirty_segments"] == [0, 2, 3, 4, 5, 6, 7, 10]
    assert (rows[6].get("fill") or {}).get("job_id", "").startswith("027778d1")
    # the pending legacy cards are bound to these exact results (live tokens)
    assert P.result_token(_run(), rows[0], QA["source_identity"]) == "c465c105"
    assert P.result_token(_run(), rows[10], QA["source_identity"]) == "d4b96d9a"


def _drive_to_the_batch(sim, allowance):
    """Free work first: the grown-mask retry of segment 6 and a preview for every
    dirty Omni segment, one per tick, all prepared in parallel."""
    made, launched = [], []
    while True:
        snap = _snapshot(made)
        d = _plan(sim, snap, allowance)
        if not (isinstance(d, P.Launch) and d.handler == P.H_FILL_PREVIEW):
            break
        rsid = d.params["run_segment_id"]
        fid = f"p{IDX[rsid]:02d}"
        launched.append((IDX[rsid], d))
        sim.add(d.name, "action", d.handler, "launched", params=d.params,
                refs={"fill_jobs": [fid]})
        made.append(_job(fid, rsid, "preparing", retry_of=d.params.get("retry_of"),
                         box_source="request" if d.params.get("mask_box") else "measured"))
    # while they prepare: a Wait, and the stale cards of 13:44 are retired
    assert isinstance(d, P.Wait), d
    ready = [dict(j, status="ready") for j in made]
    return launched, d, _snapshot(ready)


def test_replay_ff218831_all_at_once_without_a_question():
    sim = Sim()
    launched, wait, snap = _drive_to_the_batch(sim, allowance=1.0)
    assert sorted(i for i, _d in launched) == [0, 2, 3, 4, 5, 6, 7, 10]
    grow = dict(launched)[6]
    assert grow.name.startswith("grow:") and grow.params["retry_of"].startswith("027778d1")
    assert grow.params["mask_grow_px"] == P.GROW_PX
    old = rows_box = next(r for r in QA["qa"]["segments"] if r["index"] == 6)["fill"]["mask_box"]
    assert grow.params["mask_box"][0] < old[0] and grow.params["mask_box"][3] > rows_box[3]
    # 81d0afc0's stale cards (pending since before the re-stitch) are gone
    stale = [n for n, _s, _d, _o in FIX["answers"] if n.startswith(("operator:", "stop:",
                                                                   "eye:"))]
    assert set(stale) <= set(wait.retire)
    assert all(sim.answers[n].status == "superseded" for n in stale)

    d = _plan(sim, snap)
    assert isinstance(d, P.Launch) and d.handler == P.H_FILL_BATCH_START
    assert not d.regeneration and d.name.startswith("autovoid:")
    assert d.params["authorization"] == "auto" and d.params["approved_usd"] == 0.8
    assert d.params["fill_ids"] == sorted(f"p{i:02d}" for i in (0, 2, 3, 4, 5, 6, 7, 10))
    assert "inside the run's VOID allowance ($1.00 of $1.00 left)" in d.reason
    asks = [x for x in sim.decisions if isinstance(x, P.Ask)]
    assert asks == []                                  # not one question so far


def test_replay_ff218831_one_question_when_there_is_no_allowance(capsys):
    sim = Sim()
    _launched, _wait, snap = _drive_to_the_batch(sim, allowance=0.0)
    d = _plan(sim, snap, allowance=0.0)
    assert isinstance(d, P.Compute) and d.func == P.C_BATCH_SHEET
    assert [i["index"] for i in d.args["items"]] == [0, 2, 3, 4, 5, 6, 7, 10]
    sim.add(d.name, "compute", d.func, "succeeded",
            result={"sheet": "/att/b/batch_sheet.jpg", "out_dir": "/att/b"})
    ask = _plan(sim, snap, allowance=0.0)
    assert isinstance(ask, P.Ask) and ask.name.startswith("vbatch:") and ask.kind == "regeneration"
    assert ask.question.startswith(
        "Segments 0, 2, 3, 4, 5, 6, 7 and 10 still show burned-in captions: rebuild all 8 in "
        "one batch with VOID from the ORIGINAL video ($0.80 reserved, about $0.40 charged).")
    assert "a second, automatic try with a larger area" in ask.question
    assert [o.id for o in ask.options] == ["start", "per_segment", "accept", "manual"]
    assert ask.cost_credits == 160.0 and [a.path for a in ask.artifacts] == [
        "/att/b/batch_sheet.jpg"]
    print("BATCH QUESTION:", ask.question)
    print("OPTIONS:", [(o.id, o.label) for o in ask.options])
    sim.answer(ask.name, "start")
    d = _plan(sim, snap, allowance=0.0)
    assert isinstance(d, P.Launch) and d.regeneration and d.handler == P.H_FILL_BATCH_START
    assert d.params["authorization"] == "operator" and len(d.params["fill_ids"]) == 8


def test_replay_ff218831_over_the_allowance_asks_once():
    sim = Sim()
    _launched, _wait, snap = _drive_to_the_batch(sim, allowance=1.0)
    d = _plan(sim, snap, allowance=1.0, spent=0.5)
    if isinstance(d, P.Compute):
        sim.add(d.name, "compute", d.func, "succeeded", result={"sheet": None})
        d = _plan(sim, snap, allowance=1.0, spent=0.5)
    assert isinstance(d, P.Ask) and d.name.startswith("vbatch:")
    assert "the VOID allowance is $1.00 and $0.50 of it is already used" in d.question
    # without the sheet the members' own sheets are attached
    assert any("the area VOID repaints" in a.label for a in d.artifacts)


def test_replay_ff218831_after_the_batch_one_eye_check_card():
    """After the batch (the eight composites delivered in one promote): the
    composites the detector calls clean and whose mask touches no sampled face
    pass on their own; the rest are ONE end-of-run card."""
    qa = copy.deepcopy(QA)
    rows = {r["index"]: r for r in qa["qa"]["segments"]}
    for i in (0, 2, 3, 4, 5, 7, 10, 6):
        r = rows[i]
        r.update(verdict="clean", result_hits=0, leftovers=[], result_kind="void_fill",
                 fill={"job_id": f"p{i:02d}", "mask_box": [0.25, 0.65, 0.75, 0.73],
                       "delivery": "delivered", "basis": "model_result",
                       "replaced_take_id": f"take{i}"},
                 fill_needs_visual_check=True, needs_visual_check=True,
                 fill_sheet=f"/att/qa2/qa_seg{i}_fill.jpg")
    qa["qa"]["dirty_segments"] = []
    # faces sampled: far from the captions except in segment 3 (a hand-held phone
    # at chin height) -- 3 is doubtful, the others pass; 8 and 9 (older
    # composites, no evidence in this QA) stay doubtful as well
    far = {"samples": 18, "boxes": [[0.30, 0.15, 0.70, 0.45]]}
    qa["qa"]["fill_faces"] = {str(i): far for i in (0, 2, 4, 5, 6, 7, 10)}
    qa["qa"]["fill_faces"]["3"] = {"samples": 18, "boxes": [[0.30, 0.40, 0.70, 0.66]]}
    # no screen / app UI under any mask but segment 5's (a phone held into the band)
    screen = {str(i): {"samples": 12, "screen_frames": 0, "max_share": 0.2}
              for i in (0, 2, 3, 4, 6, 7, 10)}
    screen["5"] = {"samples": 12, "screen_frames": 4, "max_share": 0.61}
    auto = P.auto_accepted_fills(qa["qa"], qa["qa"]["fill_faces"], screen)
    assert sorted(v["index"] for v in auto.values()) == [0, 2, 4, 6, 7, 10]   # not 3, not 5
    eff = P.auto_accepted_fills(qa["qa"], None, screen)
    assert eff == {}                                         # no evidence: nothing passes
