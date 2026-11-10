"""The subtitle-removal autopilot flow (AICP-364, "B") on the core (AICP-363).

``advance`` functions only gather inputs from the core's read-only
:class:`~app.autopilot.contract.StepContext` and map the plan decisions of
:mod:`subtitle_removal_plan` 1:1 onto the contract's. Design:
docs/autopilot-subtitle-flow.md; flow: docs/assistant-flow-subtitle-removal.md.

    segmentation -> launch -> clean -> handoff
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from ... import project_types
from .. import contract as C
from .. import registry
from . import subtitle_compute  # noqa: F401  (registers the computes)
from . import subtitle_handlers  # noqa: F401  (registers the handlers)
from . import subtitle_removal_plan as plan

CORE_SEP = "#"


# ---------------------------------------------------------------------------
# Inputs from the context
# ---------------------------------------------------------------------------

def _status(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def effects(ctx) -> list[plan.EffectInfo]:
    return [plan.EffectInfo(id=e.id, name=e.name, kind=e.kind, handler=e.handler, state=e.state,
                            params=dict(e.params or {}), created_refs=dict(e.created_refs or {}),
                            result=e.result, error=e.error, seq=i, created_at=_iso(e.created_at))
            for i, e in enumerate(ctx.effects(""))]


def answers(ctx) -> dict[str, plan.Answer]:
    """The current generation of every FLOW approval of this step (core-created
    ``#budget`` / ``#regeneration`` / ``#overrun`` ones are the core's)."""
    from ...models_autopilot import AutopilotApproval

    rows = ctx.db.execute(select(AutopilotApproval.name).where(
        AutopilotApproval.autopilot_id == ctx._ap.id,
        AutopilotApproval.step_id == ctx.step_id,
        AutopilotApproval.step_generation == ctx.step_generation)).scalars().all()
    out = {}
    for name in sorted(set(rows)):
        if CORE_SEP in name:
            continue
        a = ctx.approval(name)          # None when the latest generation is superseded
        members = ()
        if a is not None and name.startswith(("vbatch:", "eyes:")):
            row = ctx.db.get(AutopilotApproval, a.id)
            members = tuple(str(m) for m in ((row.proposal or {}).get("members") or ())) \
                if row is not None else ()
        out[name] = (plan.Answer(name, "superseded") if a is None else
                     plan.Answer(name, a.status, a.decision, a.chosen_option, a.answer_comment,
                                 members=members))
    return out


def _segments(ctx) -> list[dict]:
    return plan.segment_list([{
        "id": sd.id, "index": sd.index, "start_sec": sd.start_sec, "end_sec": sd.end_sec,
        "action": sd.action, "pre_roll_sec": sd.pre_roll_sec, "post_roll_sec": sd.post_roll_sec,
    } for sd in ctx.project.segments])


def _detector_version(ctx) -> str:
    return ctx.version(f"project_detector:{ctx.project.id}")


def run_info(ctx, run_id: str) -> Optional[plan.RunInfo]:
    """The run plus its ACTIVE swap segments with result identities (stat only)."""
    import os

    from ...content_block import segment_block
    from ...models import Run, RunSegment
    from ...pipeline_segment_fill import file_identity
    from ...pipeline_v2 import _active_swap_segment_ids, run_segment_defs

    run = ctx.db.get(Run, run_id)
    if run is None or run.project_id != ctx.project.id:
        return None
    version = ctx.run_version(run_id)
    defs = run_segment_defs(run, ctx.project)
    active = _active_swap_segment_ids(run.stitch_plan, defs)
    by_def = {rs.segment_def_id: rs for rs in ctx.db.execute(
        select(RunSegment).where(RunSegment.run_id == run_id)).scalars()}
    segs = []
    for sd in defs:
        rs = by_def.get(sd.id)
        if sd.id not in active:
            continue
        if rs is None:
            # Not even created (the run failed before its submit): unfinished,
            # so a retry of it bills -- never a free delivery retry (gate R1-2).
            segs.append({"id": None, "index": sd.index, "status": "missing",
                         "local_result_path": None, "seedance_task_id": None,
                         "result_identity": None, "has_result": False,
                         "content_block": None, "error_message": None,
                         "start_sec": sd.start_sec, "end_sec": sd.end_sec})
            continue
        path = rs.local_result_path
        segs.append({"id": rs.id, "index": sd.index, "status": _status(rs.status),
                     "local_result_path": path, "seedance_task_id": rs.seedance_task_id,
                     "result_identity": file_identity(path),
                     "has_result": (_status(rs.status) == "completed" and not rs.source_fallback
                                    and bool(path) and os.path.exists(path)),
                     # A provider content-filter refusal (AICP-358 follow-up):
                     # the flow fills it from the source instead of retrying.
                     "content_block": segment_block(rs), "error_message": rs.error_message,
                     "model_fallback": bool(getattr(rs, "model_fallback", False)),
                     "start_sec": sd.start_sec, "end_sec": sd.end_sec})
    from ...pipeline_v2 import preprocess_hole_index

    hole = bool(getattr(run, "preprocess_enabled", False)) and preprocess_hole_index(run) is not None
    from ...api_v2 import _current_result_is_audio_patch

    from ...project_types import is_void_source_run

    return plan.RunInfo(id=run.id, status=_status(run.status), version=version, segments=segs,
                        error_message=run.error_message, preprocess_hole=hole,
                        audio_patch=_current_result_is_audio_patch(run, ctx.db) is not None,
                        void_source=is_void_source_run(run))


def snapshot(ctx, run_id: str, src_fps: Optional[float]) -> Optional[dict]:
    from ... import subtitle_loop as sl
    from ...models import Run

    run = ctx.db.get(Run, run_id)
    if run is None:
        return None
    return sl.fill_snapshot(ctx.db, run, ctx.project, src_fps=src_fps)


# Card titles and option consequences (AICP-376, docs/autopilot-operator-ux.md
# §4.2): one table by approval-name prefix, filled in at conversion so every ask
# of the planner gets one without each site spelling it. "{s}" is the subject.
_TITLES = {
    "segmentation": "Check the keep segments and the caption band before launch",
    "locate": "Check where the captions were found before launch",
    "restitch_patch": "Lip sync is off: a re-stitch would drop the audio fix",
    "restitch_failed": "The re-stitch did not fix lip sync",
    "vbatch": "Remove the leftover captions in {s} with VOID",
    "void": "Captions are left in {s}: rebuild with VOID?",
    "inconclusive": "The caption check could not decide: please look",
    "timing": "Check the lip sync",
    "edge": "Caption words are left at the edge of {s}",
    "sync": "Lip sync is off in {s}",
    "eye": "Check the VOID result of {s}",
    "stop": "The VOID fill of {s} stopped",
    "fp": "Is the text in {s} a caption?",
    "keepcap": "Captions are kept in {s}",
    "eyes": "Check {s} by eye",
    "hand": "{s} needs your hand",
    "widen_failed": "The larger VOID area failed for {s}",
    "best": "Omni rerun: keep the new take or the previous one?",
    "redeliver_failed": "{s} could not be stitched",
    "stuck": "{s} is stuck",
    "retry": "Retry the unfinished segments of {s}?",
    "vsrc_failed": "VOID from the source failed for {s}",
    "vfallback_failed": "The Omni fallback failed for {s}",
    "blocked": "{s} was refused by the provider",
    "blocked_failed": "Fixing the refused {s} failed",
    "handoff": "Final check of {s} before the hand-off",
}
_CONSEQUENCES = {
    "manual": "the assistant stops proposing it and checks again when the video changes",
    "accept": "ships it as it is; nothing is regenerated",
    "accept_all": "ships all of them as they are",
    "per_segment": "the assistant asks segment by segment next",
    "start": "starts the VOID fill(s) shown; costs the reserve shown",
    "void": "fills the caption area from the source; costs the reserve shown",
    "source": "uses the original clip as it is; free",
    "rebuild": "makes a free preview first, then asks for its cost",
    "as_caption": "makes a free preview first, then asks for its cost",
    "omni_rerun": "resubmits the segment to Omni; its cost is asked separately",
    "omni": "resubmits the segment to Omni; its cost is asked separately",
    "retry": "resubmits the unfinished segments; may be refused again",
    "restore": "puts the previous take back; free",
    "restore_previous_take": "puts the previous take back; free",
    "keep_new": "keeps the rerun's take",
    "restitch": "re-stitches for free; the delivered audio fix is lost",
    "widen_mask": "retries VOID over a larger area; its cost is asked",
    "confirm": "ships the segment; the text stays",
    "apply": "changes the segments before the launch",
    "apply_band": "saves the band, then you review the segmentation on it",
    "apply_main": "saves only the main caption position",
    "keep_only": "applies the keep segments; the detector band stays",
    "as_is": "launches the segments as analysed",
}


def _title(d) -> str:
    if d.title:
        return d.title
    tpl = _TITLES.get(d.name.split(":", 1)[0])
    if not tpl:
        return ""
    subject = (d.subject or "").strip() or "the run"
    text = tpl.format(s=subject)
    return text[:1].upper() + text[1:]


def to_contract(ctx, d):
    """A plan decision -> the contract's decision (same field names)."""
    if isinstance(d, plan.Complete):
        return C.Complete(output=dict(d.output), depends_on=dict(d.depends_on), reason=d.reason,
                          consumes=tuple(d.consumes))
    if isinstance(d, plan.Wait):
        return C.Wait(d.reason, d.recheck_sec, retire=tuple(d.retire))
    if isinstance(d, plan.Launch):
        return C.Launch(d.name, d.handler, dict(d.params), d.reason, regeneration=d.regeneration,
                        expect=dict(d.expect), consumes=tuple(d.consumes))
    if isinstance(d, plan.Compute):
        return C.Compute(d.name, d.func, dict(d.args), d.reason, timeout_sec=d.timeout_sec)
    if isinstance(d, plan.Fail):
        return C.Fail(d.reason, consumes=tuple(d.consumes))
    if isinstance(d, plan.Ask):
        return C.Ask(
            d.name, d.kind, d.question, dict(d.proposal), subject=d.subject,
            options=[C.Option(o.id, o.label, handler=o.handler,
                              params=dict(o.params) if o.params is not None else None,
                              consequence=o.consequence or _CONSEQUENCES.get(o.id, ""))
                     for o in d.options],
            artifacts=[ctx.artifact(a.path, label=a.label, kind=a.kind, run_id=a.run_id,
                                    effect_id=a.effect_id) for a in d.artifacts],
            about=dict(d.about), cost_credits=float(d.cost_credits),
            replaces=tuple(d.replaces), retire=tuple(d.retire), title=_title(d))
    raise TypeError(f"not a plan decision: {d!r}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _seg_kwargs(ctx) -> dict:
    return dict(project_id=ctx.project.id, project_status=_status(ctx.project.status),
                segments=_segments(ctx), segments_version=ctx.segments_version(),
                detector_config=ctx.project.caption_detector,
                detector_version=_detector_version(ctx), effects=effects(ctx),
                answers=answers(ctx))


def advance_segmentation(ctx):
    return to_contract(ctx, plan.plan_segmentation(**_seg_kwargs(ctx)))


def _launch_params(ctx) -> dict:
    spec = project_types.spec_for(ctx.project.project_type)
    return plan.launch_params(default_prompt=spec.default_prompt, params=ctx.params)


def _void_source_project(project) -> bool:
    return project_types.caption_removal_method(project) == \
        project_types.CAPTION_REMOVAL_VOID_SOURCE


def estimate_launch(ctx) -> float:
    if _void_source_project(ctx.project):
        return 0.0          # nothing goes to the model (design §7.3)
    return plan.launch_estimate(_segments(ctx), plan.launch_resolution(ctx.params))


def advance_launch(ctx):
    kw = _seg_kwargs(ctx)
    return to_contract(ctx, plan.plan_launch(
        params=_launch_params(ctx), seg_output=ctx.output(plan.STEP_SEGMENTATION), **kw))


def _run_id(ctx) -> Optional[str]:
    out = ctx.output(plan.STEP_LAUNCH) or {}
    return out.get("run_id")


def advance_clean(ctx):
    rid = _run_id(ctx)
    run = run_info(ctx, rid) if rid else None
    effs = effects(ctx)
    qas = [e for e in effs if e.handler == plan.C_QA and e.state == "succeeded" and e.result]
    src_fps = (qas[-1].result or {}).get("source_fps") if qas else None
    snap = snapshot(ctx, rid, src_fps) if run is not None else None
    if snap is not None:
        # created_refs.takes (core R8) is a list of take ids; the planner wants
        # {run_segment_id: take_id} for each rerun it compares.
        take_seg = {t["take_id"]: rsid for rsid, ts in snap["takes"].items() for t in ts}
        effs = [plan.EffectInfo(**{**e.__dict__, "created_refs": {
            **e.created_refs, "takes": {take_seg[t]: t for t in e.created_refs.get("takes") or []
                                        if t in take_seg}}})
                if e.handler == plan.H_RERUN and isinstance(e.created_refs.get("takes"), list)
                else e for e in effs]
    from ... import pipeline_segment_fill as psf
    from ...config import settings

    return to_contract(ctx, plan.plan_clean(
        run=run, detector_config=ctx.project.caption_detector, effects=effs,
        answers=answers(ctx), snapshot=snap, project_id=ctx.project.id,
        detector_version=_detector_version(ctx),
        # Batch VOID fills (2026-11-03): the run's allowance and what was
        # already started without a question (both read from the database;
        # the admission re-checks them under the run lock).
        void_allowance_usd=psf.void_allowance_usd(ctx.project),
        void_auto_spent_usd=psf.auto_spent_usd(ctx.db, rid) if rid else 0.0,
        auto_accept=bool(getattr(settings, "AUTOPILOT_AUTO_ACCEPT_FILLS", True)),
        now=ctx.now))


def advance_handoff(ctx):
    rid = _run_id(ctx)
    run = run_info(ctx, rid) if rid else None
    view = ctx.run(rid) if rid else None
    return to_contract(ctx, plan.plan_handoff(
        run=run, clean_output=ctx.output(plan.STEP_CLEAN), answers=answers(ctx),
        final_path=view.result_local_path if view is not None else None))


class SubtitleRemovalFlow:
    id = plan.FLOW_ID
    version = plan.FLOW_VERSION
    label = "Subtitle removal (Gemini Omni + VOID fill)"
    project_types = frozenset({project_types.SUBTITLE_REMOVAL})
    handlers = {k: v for k, v in plan.HANDLERS.items()
                if k not in (plan.C_REVIEW, plan.C_QA, plan.C_BATCH_SHEET)}

    def validate_params(self, project, params: dict) -> dict:
        try:
            return plan.validate_params(params)
        except plan.FlowParamsError as exc:
            raise C.FlowParamsError(str(exc)) from None

    def steps(self, project, params: dict):
        return [
            C.StepSpec(plan.STEP_SEGMENTATION, "Segmentation review (calibration, keeps, band)",
                       advance=advance_segmentation, approvals=("handoff",)),
            C.StepSpec(plan.STEP_LAUNCH, "One Gemini Omni pass", advance=advance_launch,
                       needs=(plan.STEP_SEGMENTATION,), paid=True, approvals=("handoff",),
                       estimate=estimate_launch),
            C.StepSpec(plan.STEP_CLEAN, "Caption QA, VOID fills, eye checks",
                       advance=advance_clean, needs=(plan.STEP_LAUNCH,), paid=True,
                       approvals=("regeneration", "handoff"), estimate=lambda ctx: 0.0),
            C.StepSpec(plan.STEP_HANDOFF, "Final + hand-off to localisation",
                       advance=advance_handoff, needs=(plan.STEP_CLEAN,), approvals=("handoff",)),
        ]

    def budget_lines(self, ctx) -> list:
        from ... import pipeline_segment_fill as psf

        lines = plan.budget_lines(segments=_segments(ctx),
                                  duration_sec=getattr(ctx.project, "duration_sec", None),
                                  void_allowance_usd=psf.void_allowance_usd(ctx.project))
        return [C.BudgetLine(x["label"], float(x["credits"]), x.get("step_id")) for x in lines]


FLOW = SubtitleRemovalFlow()
registry.register(FLOW)
