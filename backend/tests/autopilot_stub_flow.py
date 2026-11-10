"""Test-only autopilot flow (AICP-363 acceptance, docs/autopilot-design.md §10).

ready (free: waits for the project) → probe (paid: create_run) → listen
(Ask listening_probe) → rest (paid: rerun_segments, a regeneration) → handoff
(Ask handoff). ``KNOBS`` lets a test bend one step's behaviour.
"""

from __future__ import annotations

from app.autopilot import registry
from app.autopilot.contract import (
    Ask, Complete, Compute, Fail, FlowParamsError, Judge, Launch, Option, StepSpec, Wait,
)

KNOBS: dict = {}


def _ready(ctx):
    st = ctx.project.status.value if hasattr(ctx.project.status, "value") else str(ctx.project.status)
    if st != "ready":
        return Wait("waiting for the analysis")
    if KNOBS.get("ready_writes"):
        from sqlalchemy import text

        ctx.db.execute(text("UPDATE video_projects SET name = 'hacked'"))
    if KNOBS.get("ready_commits"):
        ctx.db.commit()
    return Complete({"ok": True})


def _probe_estimate(ctx):
    return float(KNOBS.get("probe_estimate", 260.0))


def _probe(ctx):
    e = ctx.effect("launch")
    if e is None:
        return Launch("launch", "create_run", {
            "prompt": ctx.params.get("prompt", "swap"), "model": "seedance",
            "resolution": "720p",
        }, "probe segment 1")
    if e.state == "launching":
        return Wait("admission bookkeeping pending")
    rid = e.created_refs["runs"][0]
    run = ctx.run(rid)
    if run is None:
        return Fail("the probe run is gone")
    if not run.is_terminal:
        return Wait(f"probe run {rid[:8]} is {run.status}")
    if run.status != "done":
        return Fail(f"probe run {run.status}")
    # depends_on only when a test asks: the later `rest` step regenerates this
    # same run on purpose, which would (correctly) reopen `probe`.
    dep = {f"run:{rid}": ctx.run_version(rid)} if KNOBS.get("probe_depends") else {}
    return Complete({"run_id": rid}, depends_on=dep)


def _listen(ctx):
    rid = ctx.output("probe")["run_id"]
    a = ctx.approval("listen")
    if a is None:
        arts = []
        if KNOBS.get("listen_artifact"):
            run = ctx.run(rid)
            arts = [ctx.artifact(run.result_local_path, label="probe", kind="video", run_id=rid)]
        return Ask("listen", "listening_probe", "Listen to the probe", {"run_id": rid},
                   subject="en", artifacts=arts, about={f"run:{rid}": ctx.run_version(rid)})
    if a.status == "pending":
        return Wait("waiting for the operator to listen")
    if KNOBS.get("listen_ignores_answer"):
        return Wait("ignoring the answer")
    if a.decision == "reject":
        return Fail(f"probe rejected: {a.answer_comment}", consumes=["listen"])
    return Complete({"accepted": True}, consumes=["listen"])


def _rest(ctx):
    rid = ctx.output("probe")["run_id"]
    e = ctx.effect("rerun")
    if e is None:
        run = ctx.run(rid)
        seg = run.segments[0]
        if KNOBS.get("rest_choice"):
            a = ctx.approval("choice")
            params = {"run_id": rid, "segments": [{"run_segment_id": seg.id, "prompt": "again"}]}
            if a is None:
                titled = KNOBS.get("titled")
                return Ask("choice", "regeneration", "Fix segment 1? It shows captions.",
                           {"_title": "smuggled"} if KNOBS.get("reserved_title") else {},
                           title="Segment 1 still shows captions" if titled else "",
                           options=[Option("rerun", "Rerun it", handler="rerun_segments",
                                           params=params,
                                           consequence="resubmits segment 1" if titled else ""),
                                    Option("accept", "Accept")])
            if a.status == "pending":
                return Wait("waiting for the choice")
            if a.decision == "reject" or a.chosen_option == "accept":
                return Complete({"rerun": False}, consumes=["choice"])
            return Launch("rerun", "rerun_segments", params, "operator chose a rerun",
                          consumes=["choice"])
        return Launch("rerun", "rerun_segments",
                      {"run_id": rid, "segments": [{"run_segment_id": seg.id, "prompt": "again"}]},
                      "regenerate segment 1", regeneration=True)
    if e.state == "launching":
        return Wait("bookkeeping")
    run = ctx.run(rid)
    if not run.is_terminal:
        return Wait("rerun in flight")
    return Complete({"rerun": True}, depends_on={f"run:{rid}": ctx.run_version(rid)})


def _handoff(ctx):
    a = ctx.approval("handoff")
    if a is None:
        return Ask("handoff", "handoff", "Accept the result?", {"run_id": ctx.output("probe")["run_id"]})
    if a.status == "pending":
        return Wait("waiting for the hand-off")
    return Complete({"accepted": a.decision == "approve"}, consumes=["handoff"])


def _free_paid_violation(ctx):
    return Launch("launch", "create_run", {"prompt": "x"}, "paid from a free step")


def _compute_step(ctx):
    e = ctx.effect("scan")
    if e is None:
        return Compute("scan", "stub_scan", {"n": 3}, "scan the source")
    if e.state in ("queued", "running"):
        return Wait("scanning")
    return Complete({"scan": e.result})


def _judge_step(ctx):
    e = ctx.effect("think")
    if e is None:
        return Judge("think", "stub_judgement", {"question": "who speaks?"}, "source breakdown")
    if e.state in ("prepared", "sending", "submitted"):
        return Wait("thinking")
    return Complete({"answer": e.result, "state": e.state})


class StubFlow:
    id = "stub"
    version = 1
    label = "Stub flow"
    project_types = frozenset({"cover_change"})
    handlers = {"create_run": 1, "rerun_segments": 1}

    def validate_params(self, project, params):
        if params.get("bad"):
            raise FlowParamsError("bad params")
        return {"prompt": params.get("prompt", "swap the cover")}

    def steps(self, project, params):
        variant = KNOBS.get("variant")
        if variant == "free_paid":
            return [StepSpec("ready", "ready", advance=_ready),
                    StepSpec("oops", "oops", needs=("ready",), advance=_free_paid_violation)]
        if variant == "compute":
            return [StepSpec("scan", "scan", advance=_compute_step)]
        if variant == "judge":
            return [StepSpec("think", "think", advance=_judge_step)]
        return [
            StepSpec("ready", "Wait for the analysis", advance=_ready),
            StepSpec("probe", "Probe run", needs=("ready",), paid=True, estimate=_probe_estimate,
                     advance=_probe),
            StepSpec("listen", "Listen to the probe", needs=("probe",),
                     approvals=("listening_probe",), advance=_listen),
            StepSpec("rest", "Regenerate", needs=("listen",), paid=True,
                     approvals=("regeneration",), estimate=lambda ctx: 0.0, advance=_rest),
            StepSpec("handoff", "Hand off", needs=("rest",), approvals=("handoff",),
                     advance=_handoff),
        ]

    def budget_lines(self, ctx):
        return []


FLOW = StubFlow()
registry.register(FLOW)


def stub_scan(args, out_dir):
    import os

    with open(os.path.join(out_dir, "sheet.txt"), "w") as fh:
        fh.write("sheet")
    return {"hits": int(args["n"])}


registry.register_compute("stub_scan", stub_scan, version=1)

try:
    from pydantic import BaseModel

    class _JIn(BaseModel):
        question: str

    class _JOut(BaseModel):
        speakers: int

    registry.register_judgement("stub_judgement", _JIn, _JOut, version=1)
except Exception:  # pragma: no cover
    pass
