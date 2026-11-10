"""Pure step logic of the subtitle-removal autopilot flow (AICP-364, "B").

The flow (docs/assistant-flow-subtitle-removal.md; design in
docs/autopilot-subtitle-flow.md) runs on the autopilot core (AICP-363,
docs/autopilot-design.md). This module holds every decision the flow makes and
nothing else: it reads no database, writes no file, calls no provider and
imports nothing from ``app.autopilot`` itself, so it is testable on its own and
the flow's ``advance`` functions (phase 2) only gather inputs and map the
returned plan decisions 1:1 onto ``app.autopilot.contract``.

Steps (one planner each)::

    segmentation -> launch -> clean -> handoff

* :func:`plan_segmentation` -- waits for the analysed project, refuses
  segments Omni cannot take and an uncalibrated detector, runs the review
  compute (``subtitle_loop.review_segmentation``), asks the operator to look at
  the keep candidates (end card, caption-free swap segments) and the detector
  band, then patches the segments / the project's caption detector.
* :func:`plan_launch` -- the one Omni pass (``create_run``) inside the budget;
  :func:`launch_estimate` is what ``create_run`` reserves and
  :func:`budget_lines` the flow's zero-credit notes for the budget approval.
* :func:`plan_clean` -- post-run QA (``subtitle_loop.post_run_qa`` as a
  compute), the operator's durable answers applied (:func:`apply_answers`),
  the VOID-fill loop (``subtitle_loop.fill_proposal``: free preview, then a
  REGENERATION approval per fill start -- operator decision Q1), eye checks of
  every VOID composite as ``handoff`` approvals with the fill sheets as
  artifacts (Q2/Q8: no automatic visual verdict), the stop rule
  (:func:`stop_rule_options`), the automatic best-take restore after an
  operator-chosen Omni rerun (:func:`best_take_restores`), retries of a failed
  run.
* :func:`plan_handoff` -- the final as an artifact and the
  ``POST /runs/{rid}/to-project`` request for the localisation project.

Every planner returns exactly one decision. Reject on any flow approval stops
the autopilot (``Fail``); the options choose what happens otherwise. Every
``Launch`` / ``Complete`` / ``Fail`` lists every answered, unconsumed approval
of its step in ``consumes`` (the core refuses a decision that ignores one);
answers are read from their ``decision`` / ``chosen_option``, which survive
consumption.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from app import ai_models, spend
from app import subtitle_loop as sl

FLOW_ID = "subtitle_removal"
FLOW_VERSION = 1

MODEL = sl.REMOVAL_MODEL            # "gemini-omni": one Omni pass (rules 1-2)
AUDIO_MODE = "original"             # the source soundtrack; VOID fills need it
# A source longer than this, or with more swap segments, gets the long-video
# warning of rule 3 on the budget approval.
LONG_VIDEO_SEC = 60.0
LONG_VIDEO_SEGMENTS = 8
# Bounded retries of free work before the operator is asked / the step fails.
MAX_COMPUTE_FAILURES = 3
MAX_PREVIEWS_PER_SEGMENT = 3
MAX_REDELIVERIES = 3
# A free re-stitch that puts every segment back in its slot (2026-11-02): once
# per run in the step; a drift it did not remove goes to the operator.
MAX_RESTITCHES_PER_RUN = 1
# Stitch drift is "explained" by the clips' extra frames when every shifted
# segment sits within this many (30 fps) frames of the predicted shift.
DRIFT_EXPLAINED_FRAMES = 1.5
# Free source previews per (blocked segment, blocked provider task) before the
# operator is asked (a provider-refused segment, AICP-358 follow-up).
MAX_BLOCKED_PREVIEWS = 3
# Fill-prepare refusals of a source preview that a new preview repeats.
BLOCKED_DETERMINISTIC_ERRORS = frozenset({
    "segment_too_short", "unsupported_media", "not_swap", "bad_box", "bad_basis"})
# Provider states of an ACTIVE fill that only the operator resolves (the Fal
# runbook: docs/fal-provider-design.md); never closed / abandoned / re-POSTed here.
PROVIDER_HOLD_STATES = ("submit_unknown", "needs_reconciliation")

STEP_SEGMENTATION = "segmentation"
STEP_LAUNCH = "launch"
STEP_CLEAN = "clean"
STEP_HANDOFF = "handoff"

# Handler / compute names (versions in HANDLERS are what the flow needs).
H_CREATE_RUN = "create_run"
H_PATCH_SEGMENTS = "patch_segments"
H_PATCH_DETECTOR = "patch_caption_detector"
H_RESTORE = "restore_takes"
H_RERUN = "rerun_segments"
H_FILL_PREVIEW = "segment_fill_preview"
H_FILL_CANCEL = "segment_fill_cancel"
H_FILL_START = "segment_fill_start"
H_FILL_BATCH_START = "segment_fill_batch_start"
H_FILL_APPLY_SOURCE = "segment_fill_apply_source"
H_RETRY_RUN = "retry_run"
H_RETRY_DELIVERY = "retry_delivery"
H_RESTITCH = "restitch_run"
C_REVIEW = "subtitle_review"
C_QA = "subtitle_qa"
HANDLERS = {H_CREATE_RUN: 1, H_PATCH_SEGMENTS: 1, H_PATCH_DETECTOR: 1, H_RESTORE: 1,
            H_RERUN: 1, H_FILL_PREVIEW: 1, H_FILL_CANCEL: 1, H_FILL_START: 1,
            H_FILL_BATCH_START: 1,
            H_FILL_APPLY_SOURCE: 1, H_RETRY_RUN: 1, H_RETRY_DELIVERY: 1, H_RESTITCH: 1,
            C_REVIEW: 1, C_QA: 1, "subtitle_batch_sheet": 1}

# Paid handlers whose launch is a regeneration (Q1: every VOID fill).
REGENERATION_HANDLERS = frozenset({H_FILL_START, H_RETRY_RUN, H_RERUN})

_PARAM_KEYS = {"resolution", "language", "prompt"}


class FlowParamsError(ValueError):
    """``autopilot_params`` are invalid (the core maps it to a 422)."""


# ---------------------------------------------------------------------------
# Plan decisions -- the contract's field names (app.autopilot.contract)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Option:
    id: str
    label: str
    handler: Optional[str] = None
    params: Optional[Mapping[str, Any]] = None
    consequence: str = ""           # what choosing it does (card only, AICP-376)


@dataclass(frozen=True)
class Artifact:
    path: str
    label: str
    kind: str                       # video | audio | image
    run_id: Optional[str] = None
    effect_id: Optional[str] = None


@dataclass(frozen=True)
class Complete:
    output: Mapping[str, Any] = field(default_factory=dict)
    depends_on: Mapping[str, str] = field(default_factory=dict)
    reason: str = ""
    consumes: tuple = ()


@dataclass(frozen=True)
class Wait:
    reason: str
    recheck_sec: int = 30
    retire: tuple = ()              # obsolete pending approvals (core supersedes them)


@dataclass(frozen=True)
class Launch:
    name: str
    handler: str
    params: Mapping[str, Any]
    reason: str
    regeneration: bool = False
    expect: Mapping[str, str] = field(default_factory=dict)
    consumes: tuple = ()


@dataclass(frozen=True)
class Compute:
    name: str
    func: str
    args: Mapping[str, Any]
    reason: str
    timeout_sec: int = 1800


@dataclass(frozen=True)
class Ask:
    name: str
    kind: str
    question: str
    proposal: Mapping[str, Any]
    subject: str = ""
    options: tuple = ()
    artifacts: tuple = ()
    about: Mapping[str, str] = field(default_factory=dict)
    cost_credits: float = 0.0
    replaces: tuple = ()            # pending approvals this one supersedes (core)
    retire: tuple = ()              # obsolete pending approvals (core supersedes them)
    title: str = ""                 # short card title (AICP-376)


@dataclass(frozen=True)
class Fail:
    reason: str
    consumes: tuple = ()


# ---------------------------------------------------------------------------
# Inputs (built by the phase-2 adapter from the core's views)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectInfo:
    """One effect of the step (contract ``EffectView`` + creation order)."""

    id: str
    name: str
    kind: str                       # action | compute
    handler: str
    state: str                      # launching/launched/not_launched/unknown | queued/running/succeeded/failed
    params: Mapping[str, Any] = field(default_factory=dict)
    created_refs: Mapping[str, Any] = field(default_factory=dict)
    result: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None
    seq: int = 0                    # creation order
    created_at: Optional[str] = None  # ISO timestamp (compared with take rows)


@dataclass(frozen=True)
class Answer:
    """The current generation of one flow approval (contract ``ApprovalView``)."""

    name: str
    status: str                     # pending | approved | rejected | consumed | superseded
    decision: Optional[str] = None  # approve | reject (kept after consumption)
    chosen_option: Optional[str] = None
    answer_comment: Optional[str] = None
    # What a batch question was about (its proposal's "members"): result
    # tokens "<rsid>:<tok>" for ``vbatch:``, fill ids for ``eyes:`` (2026-11-03).
    members: tuple = ()


@dataclass(frozen=True)
class RunInfo:
    id: str
    status: str
    version: str
    # Active swap RunSegments: {id, index, status, local_result_path,
    # seedance_task_id, result_identity ([size, mtime_ns]), has_result (completed,
    # not a source fallback, the file exists -- the free retry's own predicate),
    # content_block (the provider's content-filter refusal category of a failed
    # segment, app.content_block, or None), error_message}
    segments: Sequence[Mapping[str, Any]] = ()
    error_message: Optional[str] = None
    # A GPT Image preprocessing hole (pipeline_v2.preprocess_hole_index): a
    # requeue would fail in submit or bill preprocessing, never a free delivery.
    preprocess_hole: bool = False
    # The run's current result is a promoted audio patch (RunAudioPatch): any
    # re-stitch drops it unless the operator says so (gate R2-1).
    audio_patch: bool = False
    # runs.generation_mode == "void_source" (2026-11-03): every swap segment is
    # a VOID fill from the source; segments carry "model_fallback".
    void_source: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in ("done", "failed", "incomplete")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def short(value: str, n: int = 8) -> str:
    return (value or "")[:n]


def next_round(prefix: str, used: Sequence[str]) -> str:
    """First unused ``{prefix}:rN`` (the core's ``ctx.next_round``)."""
    used = set(used)
    n = 1
    while f"{prefix}:r{n}" in used:
        n += 1
    return f"{prefix}:r{n}"


def _with_prefix(effects: Sequence[EffectInfo], prefix: str) -> list[EffectInfo]:
    return sorted((e for e in effects if e.name.startswith(prefix + ":r")), key=lambda e: e.seq)


def _answered(a: Optional[Answer]) -> bool:
    return a is not None and a.status != "superseded" and a.decision is not None


def _approved(a: Optional[Answer], *options: str) -> bool:
    return _answered(a) and a.decision == "approve" and (
        not options or a.chosen_option in options)


def unconsumed(answers: Mapping[str, Answer]) -> tuple:
    """Names of the answered, not yet consumed approvals of a step -- every
    Launch / Complete / Fail must list them."""
    return tuple(sorted(n for n, a in answers.items()
                        if a.status in ("approved", "rejected") and a.decision is not None))


def _rejection(answers: Mapping[str, Answer]) -> Optional[Fail]:
    for name in sorted(answers):
        a = answers[name]
        if a.status == "rejected" and a.decision == "reject":
            why = f": {a.answer_comment}" if a.answer_comment else ""
            return Fail(f"the operator rejected {name}{why}", consumes=unconsumed(answers))
    return None


def _needs_ask(a: Optional[Answer]) -> bool:
    return a is None or a.status == "superseded"


def _launch(answers, **kw) -> Launch:
    return Launch(consumes=unconsumed(answers), **kw)


def _complete(answers, **kw) -> Complete:
    return Complete(consumes=unconsumed(answers), **kw)


def usd_to_credits(usd: float, usd_per_credit: Optional[float] = None) -> float:
    """What the spend ledger reserves for a Fal task of *usd* (credits)."""
    if usd_per_credit is None:
        from app.config import settings

        usd_per_credit = settings.KIE_USD_PER_CREDIT
    return round(float(usd) / float(usd_per_credit), 4) if usd_per_credit else 0.0


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

def validate_params(params: Optional[Mapping[str, Any]]) -> dict:
    """``autopilot_params`` of a subtitle-removal project: all optional.

    * ``resolution`` -- an Omni resolution (default: the model's default);
    * ``language`` -- the run's language code (default EN);
    * ``prompt`` -- overrides the project type's default removal prompt.
    """
    from app import languages

    params = dict(params or {})
    unknown = set(params) - _PARAM_KEYS
    if unknown:
        raise FlowParamsError(f"unknown params {sorted(unknown)}")
    out: dict = {}
    res = params.get("resolution")
    if res is not None:
        spec = ai_models.spec_for(MODEL)
        if res not in spec.resolutions:
            raise FlowParamsError(f"resolution {res!r}: {MODEL} takes {list(spec.resolutions)}")
        out["resolution"] = res
    lang = params.get("language")
    if lang is not None:
        code = languages.normalize(str(lang))
        if code is None or code not in languages.LANGUAGES:
            raise FlowParamsError(f"unknown language {lang!r}")
        out["language"] = code
    prompt = params.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise FlowParamsError("prompt must be a non-empty string")
        out["prompt"] = prompt.strip()[:4000]
    return out


# ---------------------------------------------------------------------------
# Compute results -- the schema of the two computes (phase 2 registers them)
# ---------------------------------------------------------------------------

def review_result(review: Mapping[str, Any], *, segments, detector_config,
                  segments_after=None, detector_after=None) -> dict:
    """``subtitle_review`` result: the review plus the inputs it was made on.
    ``*_after`` are the same inputs re-read after the scan; a change marks the
    result ``stale`` (it is redone)."""
    sig, dsha = segments_sig(segments), detector_sha(detector_config)
    stale = (segments_after is not None and segments_sig(segments_after) != sig) or (
        detector_after is not None and detector_sha(detector_after) != dsha)
    return {"review": dict(review), "segments": segment_list(segments), "segments_sig": sig,
            "detector_config": dict(detector_config) if detector_config else None,
            "detector_sha": dsha, "stale": bool(stale)}


def qa_result(cli_output: Mapping[str, Any], *, segments_input, run_version: str,
              run_version_after: Optional[str], detector_config, results,
              sheets: Optional[Mapping[str, str]] = None,
              refused: Optional[str] = None, source_identity: Any = None) -> dict:
    """``subtitle_qa`` result: what ``python -m app.subtitle_loop qa --run``
    prints, the QA's segment inputs, each run segment's result path and kie.ai
    task id, the extra sheets (``fill_clip:<index>``), and ``actionable`` =
    the run version did not change while QA ran (fill rows changing does not
    matter: the planner re-reads them every tick)."""
    out = {"run_version": run_version, "detector_sha": detector_sha(detector_config),
           "actionable": run_version_after is None or run_version_after == run_version,
           "refused": refused, "segments_input": [dict(s) for s in segments_input],
           "results": {k: dict(v) for k, v in (results or {}).items()},
           "sheets": dict(sheets or {}), "source_identity": source_identity}
    if refused is None:
        qa = copy.deepcopy(dict(cli_output["qa"]))
        if qa.get("alignment"):
            qa["alignment"] = {k: v for k, v in qa["alignment"].items() if k != "windows"}
        out.update(qa=qa, fill=dict(cli_output["fill"]),
                   operator_options=dict(cli_output.get("operator_options") or {}),
                   handoff=dict(cli_output.get("handoff") or {}))
    return out


# ---------------------------------------------------------------------------
# 1. segmentation
# ---------------------------------------------------------------------------

def segment_list(segments: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Normalised SegmentDefs (``id, index, start_sec, end_sec, action, rolls``)."""
    out = []
    for s in segments:
        out.append({"id": str(s["id"]), "index": int(s["index"]),
                    "start_sec": float(s["start_sec"]), "end_sec": float(s["end_sec"]),
                    "action": str(s.get("action") or "swap"),
                    "pre_roll_sec": float(s.get("pre_roll_sec") or 0.0),
                    "post_roll_sec": float(s.get("post_roll_sec") or 0.0)})
    return sorted(out, key=lambda s: s["index"])


def segments_sig(segments: Sequence[Mapping[str, Any]]) -> str:
    """Structural digest of the live SegmentDefs (what a review is valid for)."""
    return _sha([[s["id"], s["index"], round(s["start_sec"], 6), round(s["end_sec"], 6),
                  s["action"], round(s["pre_roll_sec"], 6), round(s["post_roll_sec"], 6)]
                 for s in segment_list(segments)])


def detector_sha(config: Optional[Mapping[str, Any]]) -> str:
    return _sha(dict(config) if config else None)


def clip_sec(seg: Mapping[str, Any]) -> float:
    return (float(seg["end_sec"]) - float(seg["start_sec"]) + float(seg.get("pre_roll_sec") or 0)
            + float(seg.get("post_roll_sec") or 0))


def too_long_segments(segments, max_clip_sec: Optional[float] = None) -> list[dict]:
    """Swap segments whose billed clip exceeds Omni's limit (the pipeline would
    skip them, captions included)."""
    limit = float(max_clip_sec if max_clip_sec is not None
                  else ai_models.spec_for(MODEL).max_clip_sec)
    return [{"index": s["index"], "clip_sec": round(clip_sec(s), 2), "max_clip_sec": limit}
            for s in segment_list(segments) if s["action"] == "swap"
            and clip_sec(s) > limit + 1e-6]


def review_keeps(review: Mapping[str, Any]) -> list[dict]:
    """Swap segments the review proposes to keep, with their reasons."""
    return [{"index": r["index"], "start_sec": r["start_sec"], "end_sec": r["end_sec"],
             "reason": r.get("reason"), "contact_sheet": r.get("contact_sheet")}
            for r in review.get("segments", [])
            if r.get("action") == "swap" and r.get("proposed_action") == "keep"]


def review_band(review: Mapping[str, Any]) -> Optional[list[float]]:
    band = review.get("band") or {}
    if band.get("verdict") == "widen" and band.get("proposed_band_y"):
        return [float(v) for v in band["proposed_band_y"]]
    return None


def apply_keeps(segments, keep_indices) -> list[dict]:
    keep = set(int(i) for i in keep_indices)
    return [dict(s, action="keep") if s["index"] in keep else dict(s)
            for s in segment_list(segments)]


def keep_patch_params(segments, keep_indices) -> dict:
    """``patch_segments`` (``update_project_segments``) body: the keeps only."""
    by_index = {s["index"]: s for s in segment_list(segments)}
    return {"updates": [{"id": by_index[i]["id"], "action": "keep"}
                        for i in sorted(set(int(i) for i in keep_indices)) if i in by_index],
            "deletes": [], "creates": []}


def detector_with_band(config: Optional[Mapping[str, Any]], band_y) -> dict:
    """The project's detector config with *band_y*, normalised exactly as
    ``PATCH /projects`` stores it (non-default fields only), so the digest the
    planner expects after the patch is the digest the row will have."""
    from app.qa_tools import caption_config

    out = dict(config or {})
    out["band_y"] = [round(float(band_y[0]), 3), round(float(band_y[1]), 3)]
    return caption_config(out).stored() or {}


def _review_effects(effects: Sequence[EffectInfo]) -> list[EffectInfo]:
    return sorted((e for e in effects if e.kind == "compute" and e.handler == C_REVIEW),
                  key=lambda e: e.seq)


def plan_segmentation(*, project_id: str, project_status: str, segments, segments_version: str,
                      detector_config, detector_version: str, effects: Sequence[EffectInfo],
                      answers: Mapping[str, Answer], max_clip_sec: Optional[float] = None,
                      auto_band: Optional[bool] = None, prior_located: bool = False):
    """Calibration, keep proposals and detector band before the launch (§1.1).
    *detector_version* is the core's ``project_detector:<pid>`` key (R4).
    *auto_band* (default ``settings.AUTOPILOT_AUTO_CAPTION_BAND``): save a
    confirmed located caption band without an eye check. *prior_located*: a
    located band was saved by an earlier step of this autopilot (``launch``
    re-runs this planner): never re-applied automatically after the operator
    removed it."""
    if project_status != "ready":
        return Wait(f"waiting for the project analysis (project is {project_status})", 60)
    rejected = _rejection(answers)
    if rejected:
        return rejected
    long = too_long_segments(segments, max_clip_sec)
    if long:
        return Fail("; ".join(
            "segment %d is %.1f s; Omni takes at most %g s" % (x["index"], x["clip_sec"],
                                                               x["max_clip_sec"])
            for x in long) + ": split it in the segment editor, then start the autopilot again",
            consumes=unconsumed(answers))
    sig = segments_sig(segments)
    dsha = detector_sha(detector_config)
    reviews = _review_effects(effects)
    dkey = {f"project_detector:{project_id}": detector_version}
    located_before = prior_located or any(
        e.state == "launched" and e.name.startswith("locate:") for e in effects)
    # A review whose keep patch was admitted is spent: the segments it saw are
    # gone, and a later edit back to them is a new state to review (R2-1).
    used = {short(r.id) for r in reviews
            if any(e.state == "launched" for e in _with_prefix(effects, f"keep:{short(r.id)}")
                   + _with_prefix(effects, f"band:{short(r.id)}")
                   + _with_prefix(effects, f"locate:{short(r.id)}"))}
    band_used = {short(r.id) for r in reviews
                 if any(e.state == "launched" for e in _with_prefix(effects, f"band:{short(r.id)}"))}

    # An approved review being applied (or already applied): finish it.
    for rev in reversed([r for r in reviews if r.state == "succeeded" and r.result]):
        a = answers.get(f"segmentation:{short(rev.id)}")
        if not _approved(a, "apply", "keep_only", "as_is"):
            continue
        res = rev.result
        keeps = [k["index"] for k in review_keeps(res["review"])] \
            if a.chosen_option in ("apply", "keep_only") else []
        band = review_band(res["review"]) if a.chosen_option == "apply" else None
        pre_sig, post_sig = res["segments_sig"], segments_sig(apply_keeps(res["segments"], keeps))
        pre_d = res["detector_sha"]
        post_d = detector_sha(detector_with_band(res.get("detector_config"), band)) \
            if band else pre_d
        seg_ok = sig == post_sig or (keeps and sig == pre_sig)
        det_ok = dsha == post_d or (band is not None and dsha == pre_d)
        if not (seg_ok and det_ok):
            break                   # edited since: review again (below)
        tag = short(rev.id)
        if keeps and sig != post_sig:
            mine = _with_prefix(effects, f"keep:{tag}")
            if mine and mine[-1].state == "launching":
                return Wait("applying the keep segments")
            if tag in used:
                break               # landed, then edited back: review again
            return _launch(answers, name=next_round(f"keep:{tag}", [e.name for e in effects]),
                           handler=H_PATCH_SEGMENTS, params=keep_patch_params(res["segments"], keeps),
                           reason="mark segments %s keep (approved)" % keeps,
                           expect={f"project_segments:{project_id}": segments_version, **dkey})
        if band is not None and dsha != post_d:
            mine = _with_prefix(effects, f"band:{tag}")
            if mine and mine[-1].state == "launching":
                return Wait("saving the detector band")
            if tag in band_used:
                break               # landed, then the detector was changed back: review again
            if not (mine and mine[-1].state == "launched"):
                return _launch(answers, name=next_round(f"band:{tag}", [e.name for e in effects]),
                               handler=H_PATCH_DETECTOR,
                               params={"caption_detector": detector_with_band(
                                   res.get("detector_config"), band), "expect_detector_sha": dsha},
                               reason="save the measured caption band y %.2f-%.2f" % tuple(band),
                               expect=dict(dkey))
        return _complete(answers, output=_seg_output(rev, a.chosen_option, keeps, band, sig,
                                                     dsha, located=located_before),
                         reason="segmentation review applied (%s)" % a.chosen_option)

    rtag = f"review:{short(sig, 12)}:{short(dsha)}"
    current = [r for r in reviews if r.name.startswith(rtag + ":")]
    live = [r for r in current if r.state in ("queued", "running")]
    if live:
        return Wait("reviewing the source (caption scan, faces)")
    done = [r for r in current if r.state == "succeeded" and r.result
            and r.result.get("segments_sig") == sig and r.result.get("detector_sha") == dsha
            and not r.result.get("stale") and short(r.id) not in used
            and not _pre_locator_uncalibrated(r.result, detector_config)]
    if not done:
        failed = [r for r in current if r.state == "failed"
                  or (r.state == "succeeded" and r.result and r.result.get("stale"))]
        if len([r for r in failed if r.state == "failed"]) >= MAX_COMPUTE_FAILURES:
            return Fail("the source review failed %d times: %s" % (
                MAX_COMPUTE_FAILURES, failed[-1].error), consumes=unconsumed(answers))
        return Compute(next_round(rtag, [e.name for e in effects]), C_REVIEW,
                       {"project_id": project_id, "segments_sig": sig, "detector_sha": dsha,
                        "faces": True},
                       "review the segmentation on the source (calibration, end card, "
                       "caption-free segments, caption geometry)")
    rev = done[-1]
    res = rev.result
    review = res["review"]
    cal = review.get("calibration") or {}
    band_info = review.get("band") or {}
    if band_info.get("verdict") in ("located", "located_uncertain"):
        return _located_band(project_id, segments_version, rev, effects, answers, dkey, dsha,
                             auto_band, located_before)
    if not cal.get("calibrated"):
        return Fail(not_found_message(review, detector_config), consumes=unconsumed(answers))
    keeps = review_keeps(review)
    band = review_band(review)
    if not keeps and band is None and not band_info.get("needs_visual_check"):
        return _complete(answers, output=_seg_output(rev, "nothing_to_change", [], None, sig,
                                                     dsha, located=located_before),
                         reason="no keep candidates, captions inside the detector band")
    name = f"segmentation:{short(rev.id)}"
    a = answers.get(name)
    if _needs_ask(a):
        return _segmentation_ask(name, project_id, segments_version, rev, keeps, band, band_info,
                                 dkey)
    if a.status == "pending":
        return Wait("waiting for the operator: segmentation eye check")
    # Answered with an option the loop above did not apply (it would have):
    # the inputs changed between the answer and now -- review again next tick.
    return Wait("the segmentation changed since the eye check; reviewing again")


def _pre_locator_uncalibrated(result: Mapping[str, Any], detector_config) -> bool:
    """A review made before the caption locator existed (no
    ``location_checked``) that found the detector uncalibrated on a project
    without a band of its own: not a verdict any more -- review again with the
    locator instead of failing on it. Calibrated old reviews still count."""
    from app.qa_tools import has_band

    review = result.get("review") or {}
    return ("location_checked" not in review
            and not (review.get("calibration") or {}).get("calibrated")
            and not has_band(detector_config))


UI_DETECTOR_HINT = ("open the project page, set Band y (and Text / Outline if the captions "
                    "are not white with a dark outline) under QA caption detector, Save, "
                    "then start the autopilot again")


def not_found_message(review: Mapping[str, Any], detector_config) -> str:
    """Why the flow cannot measure this video's captions, with what was found
    and the action the operator can take on the project page."""
    from app.qa_tools import has_band

    cal = review.get("calibration") or {}
    band = review.get("band") or {}
    head = ("the caption detector does not see this video's captions in its band y "
            "%.2f-%.2f (caption-like hits in %.1f %% of the frames, it needs %.0f %%)" % (
                *(band.get("configured_band_y") or band.get("effective_band_y") or (0, 0)),
                100 * float(cal.get("hit_ratio") or 0.0), 100 * _calibration_min()))
    loc = band.get("location")
    if has_band(detector_config) or not loc:
        found = ("; the band was set on the project, so it was not searched for"
                 if has_band(detector_config) else "")
    else:
        pos = sorted((p for p in loc.get("positions") or [] if p.get("support")),
                     key=lambda p: -float(p["support"]))
        if loc.get("candidate_band_y"):
            ccal = band.get("candidate_calibration") or {}
            found = ("; centred text was found at y %.2f-%.2f in %.0f %% of the sampled "
                     "frames, but the detector confirms it in only %.0f %% (%s)" % (
                         *loc["candidate_band_y"],
                         100 * float(pos[0]["support"]) if pos else 0.0,
                         100 * float(ccal.get("hit_ratio") or 0.0),
                         "; ".join(loc.get("reasons") or []) or "too few frames"))
        elif pos:
            found = ("; the whole frame was searched: the strongest centred text (y "
                     "%.2f-%.2f) shows in only %.0f %% of the frames" % (
                         pos[0]["y_range"][0], pos[0]["y_range"][1],
                         100 * float(pos[0]["support"])))
        elif loc.get("positions") and loc.get("reasons"):
            found = "; the whole frame was searched: " + "; ".join(loc["reasons"])
        else:
            found = ("; the whole frame was searched and no text of the detector's style "
                     "(%s text, %s outline) was found" % (
                         (loc.get("detector") or {}).get("text_color", "#ffffff"),
                         (loc.get("detector") or {}).get("outline_style", "outline")))
        found += " -- the captions may be another colour, outline or size"
    return f"{head}{found}: {UI_DETECTOR_HINT}"


def _located_band(project_id, segments_version, rev, effects, answers, dkey, dsha,
                  auto_band: Optional[bool], located_before: bool):
    """The review located the captions outside the configured (default) band:
    save the located band -- automatically when it is confirmed (verdict
    ``located``) and allowed, after an eye check otherwise -- and review again
    on the saved band (the digest change starts a new review round)."""
    from app.config import settings

    review = rev.result["review"]
    band_info = review.get("band") or {}
    proposed = [float(v) for v in band_info["proposed_band_y"]]
    tag = short(rev.id)
    mine = _with_prefix(effects, f"locate:{tag}")
    if mine and mine[-1].state == "launching":
        return Wait("saving the located caption band")
    def params_for(band_y):
        return {"caption_detector": detector_with_band(rev.result.get("detector_config"), band_y),
                "expect_detector_sha": dsha, "expect_no_band": True}

    params = params_for(proposed)
    reason = "save the located caption band y %.2f-%.2f" % tuple(proposed)
    # Several caption positions (the candidate is their union): the strongest
    # one alone is the other choice -- a union can take in a phone screen
    # between them.
    loc = band_info.get("location") or {}
    positions, members = loc.get("positions") or [], loc.get("members") or []
    main = None
    if len(members) > 1 and members[0] < len(positions) and positions[members[0]].get("band_y"):
        main = [float(v) for v in positions[members[0]]["band_y"]]
        if main == proposed:
            main = None
    name = f"locate:{tag}"
    a = answers.get(name)
    if _approved(a, "apply_band") or (_approved(a) and not a.chosen_option):
        return _launch(answers, name=next_round(name, [e.name for e in effects]),
                       handler=H_PATCH_DETECTOR, params=params,
                       reason=reason + " (approved)", expect=dict(dkey))
    if main is not None and _approved(a, "apply_main"):
        return _launch(answers, name=next_round(name, [e.name for e in effects]),
                       handler=H_PATCH_DETECTOR, params=params_for(main),
                       reason="save the main located caption band y %.2f-%.2f (approved)"
                       % tuple(main), expect=dict(dkey))
    auto = settings.AUTOPILOT_AUTO_CAPTION_BAND if auto_band is None else bool(auto_band)
    # A located band saved earlier (this step, or segmentation when ``launch``
    # re-plans) and since removed again by the operator: never put back on our
    # own -- and at most one automatic save per review.
    reverted = located_before
    if (band_info.get("verdict") == "located" and auto and not reverted and not mine
            and _needs_ask(a)):
        return _launch(answers, name=next_round(name, [e.name for e in effects]),
                       handler=H_PATCH_DETECTOR, params=params,
                       reason=reason + " (located and confirmed by the detector)",
                       expect=dict(dkey))
    if _needs_ask(a):
        why = ("; ".join(loc.get("reasons") or []) or band_info.get("note")
               or "the located band needs a look")
        if reverted:
            why = "a located band was saved before and removed again; " + why
        sheet = band_info.get("sheet")
        artifacts = (Artifact(sheet, "located caption band (green) vs the default band",
                              "image", effect_id=rev.id),) if sheet else ()
        return Ask(name=name, kind="handoff",
                   question=("The captions were found at y %.2f-%.2f of the frame, not where "
                             "the detector's default band looks (%s). Check on the sheet that the green "
                             "band holds the burned captions (not a phone screen, a sign or "
                             "an end card): apply it, or reject to stop and set the band on "
                             "the project page." % (proposed[0], proposed[1], why)),
                   proposal={"band": {k: band_info.get(k) for k in (
                       "verdict", "configured_band_y", "proposed_band_y", "coverage",
                       "candidate_calibration", "location", "note")}},
                   options=(Option("apply_band", "Save band y %.2f-%.2f and review the "
                                                 "segmentation on it" % tuple(proposed)),)
                   + ((Option("apply_main", "Save only the main caption position y %.2f-%.2f"
                                            % tuple(main)),) if main is not None else ()),
                   artifacts=artifacts,
                   about={f"project_segments:{project_id}": segments_version, **dkey})
    if a.status == "pending":
        return Wait("waiting for the operator: located caption band")
    return Wait("the detector changed since the located band was asked; reviewing again")


def _calibration_min() -> float:
    from app.qa_tools import CALIBRATION_MIN_RATIO

    return float(CALIBRATION_MIN_RATIO)


def _seg_output(rev, option, keeps, band, sig, dsha, note: Optional[str] = None,
                located: bool = False) -> dict:
    res = rev.result
    review = res["review"]
    cost = review.get("cost") or {}
    return {"review_effect": rev.id, "option": option, "segments_sig": sig,
            "detector_sha": dsha,
            "keeps": list(keeps), "band_y": band,
            "calibration": {k: (review.get("calibration") or {}).get(k)
                            for k in ("hit_ratio", "caption_like_hits", "calibrated")},
            "credits_current": cost.get("credits_current"),
            "credits_proposed": cost.get("credits_proposed"), "note": note,
            "located_band_applied": bool(located),
            "band_verdict": (review.get("band") or {}).get("verdict")}


def _segmentation_ask(name, project_id, segments_version, rev, keeps, band, band_info,
                      dkey) -> Ask:
    review = rev.result["review"]
    cost = review.get("cost") or {}
    parts, options = [], []
    if keeps:
        parts.append("keep segment%s %s" % ("s" if len(keeps) > 1 else "",
                                             ", ".join(str(k["index"]) for k in keeps)))
    if band is not None:
        parts.append("detector band y %.2f-%.2f" % tuple(band))
    if parts:
        options.append(Option("apply", "Apply: " + "; ".join(parts)))
    if keeps and band is not None:
        options.append(Option("keep_only", "Apply the keeps, leave the detector band"))
    options.append(Option("as_is", "Launch the segments as analysed (no changes)"))
    artifacts = tuple(Artifact(k["contact_sheet"], f"segment {k['index']} contact sheet",
                               "image", effect_id=rev.id)
                      for k in keeps if k.get("contact_sheet"))
    if band_info.get("sheet"):
        # Centred text the locator saw outside the band (green = the band).
        artifacts += (Artifact(band_info["sheet"], "caption band (green) and other centred "
                               "text (amber)", "image", effect_id=rev.id),)
    question = ("Before the Omni launch: %s? Confirm on the contact sheets that there is no "
                "person and no caption in a keep candidate, and that the boxes outside the "
                "band are captions, not phone UI." % ("; ".join(parts) or "check the band edge"))
    return Ask(name=name, kind="handoff", question=question,
               proposal={"keep": keeps, "band": {k: band_info.get(k) for k in (
                   "verdict", "configured_band_y", "proposed_band_y", "caption_geometry",
                   "note")},
                   "cost": {k: cost.get(k) for k in ("credits_per_segment", "credits_current",
                                                     "credits_proposed", "credits_saved",
                                                     "budget_check_credits_proposed")}},
               options=tuple(options), artifacts=artifacts,
               about={f"project_segments:{project_id}": segments_version, **dkey})


# ---------------------------------------------------------------------------
# 2. launch
# ---------------------------------------------------------------------------

def launch_resolution(params: Mapping[str, Any]) -> str:
    return ai_models.resolution_or_default(MODEL, params.get("resolution"))


def launch_params(*, default_prompt: str, params: Mapping[str, Any]) -> dict:
    """``create_run`` params of the one Omni pass (project-type defaults)."""
    return {"prompt": params.get("prompt") or default_prompt, "name": "autopilot",
            "model": MODEL, "resolution": launch_resolution(params), "audio_mode": AUDIO_MODE,
            "language": params.get("language"), "reference_urls": [], "segment_prompts": {},
            "stitch_plan": None, "voice_clip_id": None}


def launch_estimate(segments, resolution: Optional[str]) -> float:
    """What ``create_run`` reserves (the spend tariff per clip second)."""
    return float(sum(spend.video_credits(MODEL, resolution, clip_sec(s))
                     for s in segment_list(segments) if s["action"] == "swap"))


def budget_lines(*, segments, duration_sec: Optional[float],
                 void_expected_usd: float = sl.VOID_EXPECTED_USD,
                 void_reserve_usd: Optional[float] = None,
                 void_allowance_usd: float = 0.0,
                 usd_per_credit: Optional[float] = None) -> list[dict]:
    """The core's budget approval lines of the flow (2026-11-03): the VOID
    rebuilds the autopilot starts WITHOUT a question -- up to the run's
    allowance, never more than one reservation per swap segment -- as a real
    line (design §3), and the long-video warning (rule 3)."""
    swaps = [s for s in segment_list(segments) if s["action"] == "swap"]
    reserve = void_reserve_usd if void_reserve_usd is not None else sl.void_reserve_usd()
    allowance = max(0.0, float(void_allowance_usd or 0.0))
    predicted = min(allowance, reserve * len(swaps))
    if predicted > 0:
        lines = [{"label": ("VOID rebuilds without a question: up to $%.2f on this run "
                            "($%.2f reserved, ~$%.2f charged per segment); above that each "
                            "batch is asked" % (predicted, reserve, void_expected_usd)),
                  "credits": usd_to_credits(predicted, usd_per_credit), "step_id": STEP_CLEAN}]
    else:
        lines = [{"label": ("VOID rebuilds are not in this budget: each batch is approved on "
                            "its own (~$%.2f charged, $%.2f reserved per segment)" % (
                                void_expected_usd, reserve)),
                  "credits": 0.0, "step_id": STEP_CLEAN}]
    if (duration_sec or 0) > LONG_VIDEO_SEC or len(swaps) > LONG_VIDEO_SEGMENTS:
        lines.append({"label": ("long video (%.0f s, %d swap segments): some segments will very "
                                "likely need a VOID fill" % (duration_sec or 0, len(swaps))),
                      "credits": 0.0, "step_id": STEP_CLEAN})
    return lines


def plan_launch(*, effects: Sequence[EffectInfo], params: Mapping[str, Any],
                answers: Mapping[str, Answer], project_id: str, project_status: str, segments,
                segments_version: str, detector_config, detector_version: str,
                seg_output: Optional[Mapping[str, Any]], max_clip_sec: Optional[float] = None):
    """The one Omni pass. The launch is fenced on the segments the review saw:
    if they changed after ``segmentation`` completed (e.g. while the budget
    approval was open) the review runs again INSIDE this step before anything
    is admitted, and the launch ``expect``s the exact segment version it was
    planned on. ``segmentation`` itself does not depend on the segments, so a
    later edit never reopens this step into a second paid run."""
    mine = _with_prefix(effects, "launch")
    last = mine[-1] if mine else None
    if last is not None and last.state == "launching":
        return Wait("admitting the Omni run")
    if last is not None and last.state == "launched":
        runs = list((last.created_refs or {}).get("runs") or [])
        if not runs:
            return Fail("the launch admitted no run", consumes=unconsumed(answers))
        return _complete(answers, output={"run_id": runs[0]}, reason="Omni run admitted")
    if last is not None and last.state == "unknown":
        return Wait("ambiguous launch: the core paused the autopilot", 300)
    if seg_output is None:
        return Wait("the segmentation review is not complete")
    if segments_sig(segments) != seg_output.get("segments_sig") \
            or detector_sha(detector_config) != seg_output.get("detector_sha"):
        d = plan_segmentation(project_id=project_id, project_status=project_status,
                              segments=segments, segments_version=segments_version,
                              detector_config=detector_config, detector_version=detector_version,
                              effects=effects, answers=answers, max_clip_sec=max_clip_sec,
                              prior_located=bool(seg_output.get("located_band_applied")))
        if not isinstance(d, Complete):
            return d
    elif too_long_segments(segments, max_clip_sec):
        return Fail("a swap segment is longer than Omni takes", consumes=unconsumed(answers))
    if project_status != "ready":
        return Wait(f"project is {project_status}")
    return _launch(answers, name=next_round("launch", [e.name for e in effects]),
                   handler=H_CREATE_RUN, params=dict(params),
                   reason="one Gemini Omni pass over every swap segment (rule 1)",
                   expect={f"project_segments:{project_id}": segments_version,
                           f"project_detector:{project_id}": detector_version})


# ---------------------------------------------------------------------------
# 3. clean -- QA, fills, eye checks, stop rule
# ---------------------------------------------------------------------------

_ROW_OK = ("clean", "false_positive", "keep", "accepted")


def result_token(run: RunInfo, row: Mapping[str, Any], source: Any = None) -> str:
    """What an answer about one segment's CURRENT result is bound to: its
    path, kie.ai task, file identity (a rerun may rewrite the same path), the
    VOID fill behind it and its run-local bounds (R2-3)."""
    rsid = row.get("run_segment_id")
    seg = next((s for s in run.segments if s.get("id") == rsid), {}) if rsid else {}
    return short(_sha([seg.get("local_result_path"), seg.get("seedance_task_id"),
                       seg.get("result_identity"), _fill_id(row), source,
                       round(float(row.get("start_sec") or 0), 3),
                       round(float(row.get("end_sec") or 0), 3)]))


def _fill_id(row) -> Optional[str]:
    return (row.get("fill") or {}).get("job_id")


def void_ask_name(rsid: str, tok: str, fid: str) -> str:
    """The fix question of one segment's current result (2026-11-02): the VOID
    rebuild of preview *fid* and the alternatives. Bound to the segment's
    result token, so an ``accept`` / ``manual`` answer holds for that result
    whichever preview asked it."""
    return f"void:{rsid}:{tok}:{fid}"


def segment_choice(answers: Mapping[str, Answer], rsid: Optional[str], tok: str
                   ) -> Optional[str]:
    """The operator's standing choice for a segment's current result, from
    its fix questions (``void:`` / ``sync:`` / ``edge:``): ``accept`` /
    ``manual`` / ``omni_rerun`` / ``rebuild`` -- or None. Legacy
    ``operator:<rsid>:<tok>`` answers count only for ``accept`` (a ``manual``
    there was the only way out of a question without a fix option, 514067f2:
    it is not a hold)."""
    if not rsid:
        return None
    best = None
    for name, a in answers.items():
        if not _approved(a):
            continue
        kind, _, rest = name.partition(":")
        if kind == "void" and rest.startswith(f"{rsid}:{tok}:") and a.chosen_option != "start":
            best = a.chosen_option
        elif kind in ("sync", "edge", "hand") and rest == f"{rsid}:{tok}":
            best = a.chosen_option
        elif kind == "operator" and rest == f"{rsid}:{tok}" and a.chosen_option == "accept" \
                and best is None:
            best = "accept"
        elif kind == "vbatch" and a.chosen_option in ("accept", "manual") \
                and f"{rsid}:{tok}" in a.members and best is None:
            # The batch question's "ship them as they are" / "by hand" holds
            # for each member's result it was asked about (2026-11-03).
            best = a.chosen_option
    return best


def _per_segment_members(answers: Mapping[str, Answer]) -> set:
    """Results ("<rsid>:<tok>") the operator wants asked one by one: the
    batch question's ``per_segment`` choice."""
    out = set()
    for name, a in answers.items():
        if name.startswith("vbatch:") and _approved(a, "per_segment"):
            out |= set(a.members)
    return out


def _in_members(members: set, item: Mapping[str, Any], tokens: Mapping[int, str]) -> bool:
    return f"{item.get('run_segment_id')}:{tokens.get(item['index'], '')}" in members


# Auto-accept of a VOID composite (design §4.1): the mask box at most this
# tall, and a sampled face (padded by this share of its own size) must not
# touch the mask box (padded like the eye-check crop).
AUTO_ACCEPT_MAX_BOX_H = 0.30
AUTO_ACCEPT_FACE_PAD = 0.25
AUTO_ACCEPT_BOX_PAD = 0.04


def _grow(box, px, py):
    return [box[0] - px, box[1] - py, box[2] + px, box[3] + py]


def _touches(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def auto_accepted_fills(qa: Mapping[str, Any], fill_faces: Optional[Mapping[str, Any]],
                        fill_screen: Optional[Mapping[str, Any]] = None) -> dict:
    """The delivered VOID composites the flow accepts WITHOUT an eye check
    (operator decision 2026-11-03, design §4.1 / §7.4): detector-clean
    (verdict clean, 0 result hits), re-delivery confirmed, a VOID repaint (not
    "the source as is"), a mask box at most AUTO_ACCEPT_MAX_BOX_H tall, face
    samples over the segment (``fill_faces[index]``: >= 2 samples, no
    sampling error) none of which touches the mask box, and screen evidence
    (``fill_screen[index]``: measured, no frame over the screen threshold).
    Anything uncertain is doubtful (the end-of-run eye check). {fill_id:
    evidence}."""
    out = {}
    for row in qa.get("segments") or ():
        fill = row.get("fill") or {}
        fid = fill.get("job_id")
        if not fid or not row.get("fill_needs_visual_check") or row.get("verdict") != "clean":
            continue
        if int(row.get("result_hits") or 0) != 0 or row.get("fill_delivery_unconfirmed") \
                or fill.get("source_as_is") or fill.get("delivery") not in (None, "delivered"):
            continue
        box = fill.get("mask_box") or row.get("mask_box")
        if not box or float(box[3]) - float(box[1]) > AUTO_ACCEPT_MAX_BOX_H:
            continue
        faces = (fill_faces or {}).get(str(row["index"]))
        if not faces or faces.get("error") or int(faces.get("samples") or 0) < 2:
            continue
        mb = _grow([float(v) for v in box], AUTO_ACCEPT_BOX_PAD, AUTO_ACCEPT_BOX_PAD)
        hit = False
        for f in faces.get("boxes") or ():
            fb = [float(v) for v in f]
            w, h = fb[2] - fb[0], fb[3] - fb[1]
            if _touches(_grow(fb, AUTO_ACCEPT_FACE_PAD * w, AUTO_ACCEPT_FACE_PAD * h), mb):
                hit = True
                break
        if hit:
            continue
        # No screen / app UI under the mask (design §7.4): measured, no error,
        # and not one sampled frame over the screen threshold.
        scr = (fill_screen or {}).get(str(row["index"]))
        if not scr or scr.get("error") or int(scr.get("samples") or 0) < 1 \
                or int(scr.get("screen_frames") or 0) > 0:
            continue
        out[fid] = {"index": row["index"], "mask_box": box, "face_samples": faces["samples"],
                    "screen_max_share": scr.get("max_share"),
                    "faces_seen": len(faces.get("boxes") or ()),
                    "fill_sheet": row.get("fill_sheet")}
    return out


def _eyes_accepted(answers: Mapping[str, Answer]) -> set:
    """Fill ids an answered end-of-run eye check accepted (``eyes:`` accept_all,
    bound to the fill ids in its proposal), in any QA round."""
    out = set()
    for name, a in answers.items():
        if name.startswith("eyes:") and _approved(a, "accept_all"):
            out |= set(a.members)
    return out


def _eyes_per_segment(answers: Mapping[str, Answer]) -> set:
    out = set()
    for name, a in answers.items():
        if name.startswith("eyes:") and _approved(a, "per_segment"):
            out |= set(a.members)
    return out


def apply_answers(qa: Mapping[str, Any], answers: Mapping[str, Answer], run: RunInfo,
                  qa_tag: str, source: Any = None, auto_fills=()) -> tuple[dict, dict]:
    """The QA with the operator's durable answers applied, and ``clean``
    recomputed with ``post_run_qa``'s own formula.

    * ``eye:<fid>`` / ``stop:<fid>`` accept -> that VOID composite is
      confirmed (only while its re-delivery is confirmed); a dirty one is
      accepted as it is;
    * ``fp:s<N>:<token>`` confirm -> the false positive is confirmed;
      ``as_caption`` -> its hits are leftovers (fill them);
    * ``keepcap:s<N>:<token>`` / ``operator:<rsid>:<token>`` accept -> the
      segment ships as it is;
    * ``void:<rsid>:<tok>:<fid>`` / ``sync:<rsid>:<tok>`` / ``edge:...`` accept
      -> the segment's current result ships as it is (its leftovers and / or
      its sync offset);
    * ``inconclusive:<qa>`` accept -> an uncalibrated detector is waived for
      ``clean``; every other blocker (leftovers, sync, eye checks, false
      positives, keep captions) still needs its own answer (R3-2).

    ``clean`` also needs every swap segment in sync with the source
    (``qa["timing"]``, 2026-11-02) unless accepted; a QA without timing keeps
    the run-wide ``timeline_verified`` condition.
    """
    out = copy.deepcopy(dict(qa))
    info = {"accepted_fills": [], "accepted_segments": [], "confirmed_fp": [],
            "fp_as_caption": [], "inconclusive_accepted": False, "accepted_sync": [],
            "auto_accepted_fills": []}
    eyes_ok = _eyes_accepted(answers)
    auto_fills = set(auto_fills or ())
    for row in out["segments"]:
        idx, rsid = row["index"], row.get("run_segment_id")
        fid = _fill_id(row)
        tok = result_token(run, row, source)
        by_hand = [answers.get(f"{k}:{rsid}:{tok}") for k in ("operator", "hand",
                                                               "widen_failed")]
        if fid and row.get("fill_needs_visual_check"):
            delivered = not row.get("fill_delivery_unconfirmed")
            # The question asked about THIS verdict: a dirty composite is the
            # stop rule, a detector-clean one the eye check (R2-8).
            a = answers.get(f"stop:{fid}" if row["verdict"] == "leftover" else f"eye:{fid}")
            if not _approved(a, "accept"):
                a = next((x for x in by_hand if _approved(x, "accept")), a)
            clean_ok = row["verdict"] != "leftover" and (fid in eyes_ok or fid in auto_fills)
            if delivered and clean_ok and not _approved(a, "accept"):
                # The end-of-run eye check accepted it, or the flow did (no
                # caption, no face under the mask: design §4.1).
                row["fill"]["confirmed"] = True
                row["fill_needs_visual_check"] = False
                row["needs_visual_check"] = False
                info["accepted_fills"].append(fid)
                if fid in auto_fills and fid not in eyes_ok:
                    info["auto_accepted_fills"].append(fid)
            elif delivered and _approved(a, "accept"):
                row["fill"]["confirmed"] = True
                row["fill_needs_visual_check"] = False
                row["needs_visual_check"] = False
                info["accepted_fills"].append(fid)
                if row["verdict"] == "leftover":
                    row["verdict"] = "accepted"
                    info["accepted_segments"].append(idx)
        if row["verdict"] == "false_positive" and not row.get("confirmed"):
            a = answers.get(f"fp:s{idx}:{tok}")
            if _approved(a, "confirm"):
                row["confirmed"] = True
                row["needs_visual_check"] = False
                info["confirmed_fp"].append(idx)
            elif _approved(a, "as_caption"):
                row["verdict"] = "leftover"
                row["leftovers"] = list(row.get("false_positives") or [])
                info["fp_as_caption"].append(idx)
        if row["verdict"] == "keep_captions" and _approved(answers.get(f"keepcap:s{idx}:{tok}"),
                                                          "accept"):
            row["verdict"] = "accepted"
            info["accepted_segments"].append(idx)
        choice = segment_choice(answers, rsid, tok)
        if any(_approved(answers.get(f"{k}:{short(run.version, 12)}"), "accept")
               for k in ("restitch_failed", "restitch_patch")):
            choice = choice or ("accept" if (row.get("timing") or {}).get("status") == "off"
                                and row["verdict"] != "leftover" else None)
        if row["verdict"] == "leftover" and rsid and (
                any(_approved(x, "accept") for x in by_hand) or choice == "accept"):
            row["verdict"] = "accepted"
            info["accepted_segments"].append(idx)
        if (row.get("timing") or {}).get("status") == "off" and choice == "accept":
            row["timing"] = dict(row["timing"], status="accepted")
            info["accepted_sync"].append(idx)
    cal_ok = bool((out.get("calibration") or {}).get("calibrated"))
    timing = out.get("timing")
    if timing is not None:
        off = [r["index"] for r in out["segments"] if r.get("action") == "swap"
               and (r.get("timing") or {}).get("status") == "off"]
        out["sync_off_segments"] = off
        tl_ok = sl.timing_verified(timing, out["segments"], bool(out.get("timeline_verified"))) \
            or _approved(answers.get(f"timing:{qa_tag}"), "accept")
        out["timing"] = dict(timing, verified=tl_ok)
    else:
        tl_ok = bool(out.get("timeline_verified"))
    if not cal_ok and _approved(answers.get(f"inconclusive:{qa_tag}"), "accept"):
        info["inconclusive_accepted"] = True
    rows = out["segments"]
    out["dirty_segments"] = [r["index"] for r in rows if r["verdict"] == "leftover"]
    out["unconfirmed_false_positives"] = [r["index"] for r in rows
                                          if r["verdict"] == "false_positive"
                                          and not r.get("confirmed")]
    out["unconfirmed_fills"] = [{"index": r["index"], "fill_job_id": _fill_id(r)}
                                for r in rows if r.get("fill_needs_visual_check")]
    out["keep_with_captions"] = [r["index"] for r in rows if r["verdict"] == "keep_captions"]
    waived = info["inconclusive_accepted"]
    ok_verdicts = _ROW_OK + (("clean_uncalibrated",) if waived else ())
    out["clean"] = (not out["dirty_segments"] and not out["unconfirmed_false_positives"]
                    and not out["unconfirmed_fills"] and (cal_ok or waived) and tl_ok
                    and all(r["verdict"] in ok_verdicts for r in rows))
    return out, info


def _omni_rerun_params(run_id: str, seg: sl.Seg, row: Mapping[str, Any]) -> dict:
    """One segment's Omni rerun (operator option only): the coordinate <->
    default prompt alternation of ``rerun_proposal``; a segment without
    leftovers (a smear) keeps its prompt."""
    leftovers = row.get("leftovers") or []
    if not leftovers:
        prompt = None
    elif sl.is_coordinate_prompt(seg.prompt_override):
        prompt = ""
    else:
        prompt = sl.coordinate_prompt(sl.coordinate_box(leftovers))
    return {"run_id": run_id, "segments": [{"run_segment_id": seg.run_segment_id,
                                            "prompt": prompt}],
            "discard_audio_patch": False}


def stop_rule_options(run_id: str, seg: sl.Seg, row: Mapping[str, Any], *, reserve: float,
                      takes: int, accept_label: str = "Accept: it ships as it is"
                      ) -> tuple[tuple, dict]:
    """The flow doc's stop rule as approval CHOICES (from
    ``subtitle_loop._operator_options``) and the action each one leads to:
    accept, a widened mask (free preview, then its own paid start approval),
    restore the replaced take (free), fix it by hand (LTX-2.3 / delogo), or --
    only as the operator's choice -- an Omni rerun of the segment (paid: the
    core asks its own regeneration approval with the exact cost when the flow
    launches it).

    The options carry no action on purpose: several of them are answered side
    by side (§1.3.1) and the first launch consumes every answer of the step, so
    an answer must never be what authorises a paid launch -- it is a durable
    choice; the cost approval of a paid choice is its own."""
    raw = sl._operator_options(run_id, seg, dict(row), reserve)
    opts = [Option("accept", accept_label)]
    actions: dict = {}
    for o in raw:
        if o["option"] == "widen_mask" and o.get("mask_box"):
            opts.append(Option("widen_mask", "Rebuild again with VOID over a larger area "
                               "(a second paid rebuild: free preview first, then its own "
                               "cost approval)"))
            actions["widen_mask"] = (H_FILL_PREVIEW, {
                "run_id": run_id, "run_segment_id": seg.run_segment_id,
                "mask_box": list(o["mask_box"]), "prompt": None})
        elif o["option"] == "restore_previous_take":
            opts.append(Option("restore_previous_take", "Put back the video the rebuild "
                               "replaced (free; its captions come back)"))
            actions["restore_previous_take"] = (H_RESTORE, {
                "run_id": run_id, "takes": list(o["request"]["body"]["takes"]),
                "discard_audio_patch": False})
    opts.append(Option("manual", "I fix it by hand (e.g. LTX-2.3 inpaint over the same area); "
                       "asked again when its video changes"))
    # Not for a composite made from the source: the provider refused this
    # segment's frames, a rerun repeats the block.
    if seg.run_segment_id and takes + 1 < sl.STOP_AFTER_ATTEMPTS \
            and not _provider_refused_composite(row):
        opts.append(Option("omni_rerun", "Ask Omni again for this segment (paid: its cost is "
                           "asked before it starts; on 2845eef3 a second pass came out worse)"))
        actions["omni_rerun"] = (H_RERUN, _omni_rerun_params(run_id, seg, row))
    return tuple(opts), actions


def _replaced_take(takes: Sequence[Mapping[str, Any]], prev: Mapping[str, Any],
                   archived: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    """The take that holds the output an Omni rerun replaced: the take id the
    rerun's admission archived (``created_refs.takes``, request R8), else by
    its kie.ai task id, else -- a VOID composite has no task id (promotion
    clears it) -- by its path when it already was a take (a restore re-uses
    that row). Never inferred from timestamps (R3-3)."""
    live = [t for t in takes if not t.get("purged")]
    if archived:
        return next((t for t in live if t.get("take_id") == archived), None)
    task = prev.get("seedance_task_id")
    if task:
        hit = next((t for t in live if t.get("seedance_task_id") == task), None)
        if hit is not None:
            return hit
    path = prev.get("result_path")
    hit = next((t for t in live if path and t.get("result_path") == path), None)
    if hit is not None:
        return hit
    return None


def best_take_restores(run_id: str, *, rerun_rsids: Sequence[str],
                       prev_qa: Mapping[str, Any], cur_qa: Mapping[str, Any],
                       takes: Mapping[str, Sequence[Mapping[str, Any]]],
                       archived: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """After an Omni rerun: the ``restore_takes`` params that put back the
    replaced output where the new take is worse (more hit frames), or None."""
    prev_rows = {r.get("run_segment_id"): r for r in prev_qa["qa"]["segments"]}
    cur_rows = {r.get("run_segment_id"): r for r in cur_qa["qa"]["segments"]}
    candidates = {}
    for rsid in rerun_rsids:
        pr, cr = prev_rows.get(rsid), cur_rows.get(rsid)
        if pr is None or cr is None:
            continue
        take = _replaced_take(takes.get(rsid, ()), (prev_qa.get("results") or {}).get(rsid) or {},
                              (archived or {}).get(rsid))
        if take is None:
            continue
        candidates[rsid] = [{"take_id": None, "hits": int(cr.get("result_hits") or 0),
                             "created_at": "2"},
                            {"take_id": take["take_id"], "hits": int(pr.get("result_hits") or 0),
                             "created_at": "1"}]
    chosen = sl.choose_best_takes(run_id, candidates)
    if not chosen["request"]:
        return None
    return {"run_id": run_id, "takes": list(chosen["request"]["body"]["takes"]),
            "discard_audio_patch": False}


def stitch_drift(qa: Mapping[str, Any], source_fps: Optional[float]) -> dict:
    """Which segments the QA measured out of sync, and whether the clips'
    extra frames explain it.

    A stitch that keeps every clip at its own length moves each later segment
    by the frames the earlier clips ran over their slots (514067f2 / d8105ab5:
    Omni clips 3, 3, 2, 2 frames long -> segments 0, 3, 3, 6, 8, 10 frames
    late). ``expected`` is that shift per segment (30 fps frames, negative =
    late); ``explained`` = every shifted segment sits within
    ``DRIFT_EXPLAINED_FRAMES`` of it (a re-stitch that cuts each clip to its
    slot then fixes all of them for free). ``off`` lists the swap segments
    measured off (not accepted), ``late`` their measured frames."""
    timing = qa.get("timing") or {}
    rows = sorted(qa.get("segments") or [], key=lambda r: float(r["start_sec"]))
    clips = {int(c["index"]): c for c in timing.get("clips") or []}
    # clip excess is counted at the STITCH rate (the source's, capped):
    # convert it to the QA's timing frames (merge gate R3).
    from app.media import stitch_fps
    ratio = ((float(timing.get("fps") or 30.0) / float(stitch_fps(float(source_fps))))
             if source_fps else None)
    expected, cum, known = {}, 0.0, ratio is not None
    for r in rows:
        expected[r["index"]] = -cum
        if r.get("action") == "swap":
            c = clips.get(int(r["index"]))
            if c is None or c.get("excess_frames") is None:
                known = False
            elif ratio is not None:
                cum += float(c["excess_frames"]) * ratio
    off = [r for r in rows if r.get("action") == "swap"
           and (r.get("timing") or {}).get("status") == "off"]
    shifted = [r for r in rows if (r.get("timing") or {}).get("status") == "off"]
    explained = bool(shifted) and known and max((-e for e in expected.values()),
                                                default=0) >= 2
    # Every MEASURED segment must match the prediction, the in-sync ones too: a
    # final already stitched to the slots has its early segments at 0 where an
    # unconformed one would have them late (a coincidental match is not proof).
    measured = [r for r in rows if (r.get("timing") or {}).get("status") in
                ("in_sync", "off", "accepted")]
    for r in measured:
        t = r["timing"]
        if not explained:
            break
        if t.get("offset_frames") is None:
            explained = False
        elif t.get("beyond_search"):
            # At the search limit the real shift is at least that large.
            explained = float(t["offset_frames"]) < 0 and expected[r["index"]] \
                <= float(t["offset_frames"]) + DRIFT_EXPLAINED_FRAMES
        else:
            explained = abs(float(t["offset_frames"]) - expected[r["index"]]) \
                <= DRIFT_EXPLAINED_FRAMES
    return {"off": [r["index"] for r in off],
            "late": {r["index"]: (r["timing"] or {}).get("offset_frames") for r in off},
            "explained": explained,
            "expected": {k: round(v, 1) for k, v in expected.items()}}


def _frames_phrase(frames) -> str:
    if frames is None:
        return "an unknown number of frames"
    f = abs(float(frames))
    return "%g frame%s (%.2f s)" % (round(f, 1), "" if round(f, 1) == 1 else "s", f / 30.0)


def _segments_phrase(indices) -> str:
    idx = [str(i) for i in sorted(indices)]
    if not idx:
        return ""
    if len(idx) == 1:
        return "segment " + idx[0]
    return "segments " + ", ".join(idx[:-1]) + " and " + idx[-1]


def _qa_effects(effects: Sequence[EffectInfo]) -> list[EffectInfo]:
    return sorted((e for e in effects if e.kind == "compute" and e.handler == C_QA),
                  key=lambda e: e.seq)


def _action_effects(effects: Sequence[EffectInfo]) -> list[EffectInfo]:
    return sorted((e for e in effects if e.kind == "action"), key=lambda e: e.seq)


def plan_clean(*, run: Optional[RunInfo], detector_config, effects: Sequence[EffectInfo],
               answers: Mapping[str, Answer], snapshot: Optional[Mapping[str, Any]],
               project_id: str, detector_version: str,
               usd_per_credit: Optional[float] = None,
               void_allowance_usd: float = 0.0, void_auto_spent_usd: float = 0.0,
               auto_accept: bool = False, now: Any = None):
    """QA -> VOID fills -> eye checks -> clean (docs/autopilot-subtitle-flow.md
    §1.3). *snapshot* is the fresh fill snapshot read in planning:
    ``{"context": <subtitle_loop._fill_context()["context"]>, "takes": {rsid:
    [{take_id, seedance_task_id, purged}]}}``. *detector_version* (the core's
    ``project_detector:<pid>`` key, request R4) fences ``Complete``.

    Batch VOID fills (2026-11-03, docs/segment-fill-batch-design.md §2-§5):
    *void_allowance_usd* is the run's allowance of VOID reservations started
    without a question, *void_auto_spent_usd* what was already started that
    way; *auto_accept* lets a detector-clean composite whose mask touches no
    sampled face pass without an eye check; *now* (a datetime) bounds how long
    a preview still preparing holds the ready ones. A Wait / Ask carries in
    ``retire`` the pending asks of this step whose result moved on."""
    state: dict = {}
    d = _plan_clean(run=run, detector_config=detector_config, effects=effects, answers=answers,
                    snapshot=snapshot, project_id=project_id, detector_version=detector_version,
                    usd_per_credit=usd_per_credit, void_allowance_usd=void_allowance_usd,
                    void_auto_spent_usd=void_auto_spent_usd, auto_accept=auto_accept, now=now,
                    state=state)
    if isinstance(d, (Wait, Ask)) and state.get("live") is not None:
        keep = {d.name} if isinstance(d, Ask) else set()
        retire = tuple(sorted(n for n in obsolete_asks(answers, state["live"])
                              if n not in keep and n not in (getattr(d, "replaces", ()) or ())))
        if retire:
            d = dataclasses.replace(d, retire=retire)
    return d


# The per-result / per-QA questions of the operator batch (_operator_batch):
# asked only when nothing is progressing, as one set per QA round.
_BATCH_ASK_KINDS = ("eye", "stop", "operator", "hand", "fp", "keepcap", "sync", "edge",
                    "widen_failed", "eyes", "inconclusive", "timing")


def obsolete_asks(answers: Mapping[str, Answer], live: Mapping[str, Any]) -> list[str]:
    """Pending flow asks of the clean step that can no longer be acted on
    (design §5): a question of the operator batch the planner does not ask in
    this QA round (its result changed, it came from an older QA or a legacy
    name -- e.g. 81d0afc0's ``operator:`` / ``stop:`` cards from before the
    re-stitch), a per-segment ``void:`` / ``start:`` question other than the
    current one, a batch question whose members changed, a per-version ask of
    an older run version, a retry ask on a run that is done. *live*: "rv",
    "rv8", "asked" (names of this round's operator batch, or None when the
    planner did not get that far -- then those kinds are left alone), "void"
    (the current per-segment question), "vbatch" (the current batch question)."""
    out = []
    asked = live.get("asked")
    for name, a in answers.items():
        if a.status != "pending":
            continue
        kind, _, rest = name.partition(":")
        parts = rest.split(":")
        dead = False
        if kind in _BATCH_ASK_KINDS:
            dead = asked is not None and name not in asked
        elif kind in ("void", "start"):
            dead = live.get("decided") and name != live.get("void")
        elif kind == "vbatch":
            dead = live.get("decided") and name != live.get("vbatch")
        elif kind in ("restitch_patch", "restitch_failed"):
            dead = rest != live["rv"]
        elif kind in ("redeliver_failed", "stuck", "retry", "blocked", "blocked_failed"):
            dead = True                 # the run is done: nothing to retry / finish by hand
        elif kind == "best":
            dead = len(parts) != 2 or parts[1] != live["rv8"]
        if dead:
            out.append(name)
    return out


def _plan_clean(*, run, detector_config, effects, answers, snapshot, project_id,
                detector_version, usd_per_credit, void_allowance_usd, void_auto_spent_usd,
                auto_accept, now, state):
    if run is None:
        return Fail("the run is gone", consumes=unconsumed(answers))
    rejected = _rejection(answers)
    if rejected:
        return rejected
    names = [e.name for e in effects]
    acts = _action_effects(effects)
    if any(e.state == "launching" for e in acts):
        return Wait("an admission is being reconciled")
    if not run.terminal:
        return Wait(f"run {short(run.id)} is {run.status}", 30)
    ctx = dict((snapshot or {}).get("context") or {})
    takes = dict((snapshot or {}).get("takes") or {})
    # In flight first -- also before a retry of a failed run: a preview
    # preparing or a fill admitted / compositing settles before anything else.
    jobs = {j["job_id"]: dict(j, run_segment_id=rsid)
            for rsid, sc in (ctx.get("segments") or {}).items() for j in sc.get("jobs") or []}
    held = _provider_hold(jobs)
    if held:
        # Before the generic wait (design R1-3): a fill whose Fal task only the
        # operator can resolve says exactly what to do, and nothing else moves.
        return Wait(held, 300)
    active = [j for j in jobs.values() if j.get("status") in ("admitted", "compositing",
                                                                  "composited")]
    if ctx.get("active_fill") or active:
        batch = next((j.get("batch_id") for j in active if j.get("batch_id")), None)
        return Wait(("a batch of %d VOID fills is in progress" % sum(
            1 for j in active if j.get("batch_id") == batch)) if batch else
            "a VOID fill is in progress", 30)
    preparing = [j for j in jobs.values() if j.get("status") == "preparing"]
    rv = short(run.version, 12)
    if preparing and run.status != "done" and not run.void_source:
        # A failed / incomplete run waits for every preview (the source fills
        # go one at a time); a done run keeps planning while previews prepare
        # in parallel (batch fills, 2026-11-03).
        return Wait("a VOID fill preview is being prepared", 30)
    if run.status in ("failed", "incomplete"):
        return _plan_retry(run, rv, names, acts, answers, _qa_effects(effects), ctx=ctx,
                           jobs=jobs, project_id=project_id, detector_version=detector_version,
                           usd_per_credit=usd_per_credit, void_allowance=void_allowance_usd,
                           void_spent=void_auto_spent_usd, now=now, effects=effects)

    # -- QA for the current run version + detector ---------------------------
    dsha = detector_sha(detector_config)
    qtag_in = f"qa:{rv}:{short(dsha)}"
    qas = [e for e in _qa_effects(effects) if e.name.startswith(qtag_in + ":")]
    if any(e.state in ("queued", "running") for e in qas):
        return Wait("caption QA of the final is running")
    src_now = (snapshot or {}).get("source_identity")
    good = [e for e in qas if e.state == "succeeded" and e.result
            and e.result.get("run_version") == run.version
            and (src_now is None or e.result.get("source_identity") == src_now)
            and e.result.get("detector_sha") == dsha and e.result.get("actionable", True)
            and not e.result.get("refused")
            # A QA written before the per-segment timing (2026-11-02) cannot
            # say which segment is out of sync: it is done again (free).
            and (e.result.get("qa") or {}).get("timing") is not None]
    if not good:
        refused = next((e for e in reversed(qas) if e.state == "succeeded" and e.result
                        and e.result.get("refused")), None)
        if refused is not None:
            return Fail("caption QA refused the final: %s" % refused.result["refused"],
                        consumes=unconsumed(answers))
        if sum(1 for e in qas if e.state == "failed") >= MAX_COMPUTE_FAILURES:
            return Fail("caption QA failed %d times: %s" % (
                MAX_COMPUTE_FAILURES, [e for e in qas if e.state == "failed"][-1].error),
                consumes=unconsumed(answers))
        return Compute(next_round(qtag_in, names), C_QA,
                       {"run_id": run.id, "run_version": run.version, "detector_sha": dsha},
                       "caption QA of the final against the source (per segment)")
    qeff = good[-1]
    res = qeff.result
    qtag = short(qeff.id)
    source = res.get("source_identity")
    auto = auto_accepted_fills(res["qa"], (res.get("qa") or {}).get("fill_faces"),
                               (res.get("qa") or {}).get("fill_screen")) \
        if auto_accept else {}
    eff_qa, info = apply_answers(res["qa"], answers, run, qtag, source, auto_fills=auto)
    segs = sl.segments_from(res["segments_input"])
    by_index = {s.index: s for s in segs}
    reserve = float((res.get("fill") or {}).get("cost", {}).get("reserve_usd_per_segment")
                    or sl.void_reserve_usd())

    # -- timing: a free re-stitch first, then per-segment rebuilds -----------
    drift = stitch_drift(eff_qa, res.get("source_fps"))
    restitched = [e for e in acts if e.handler == H_RESTITCH
                  and (e.params or {}).get("run_id") == run.id
                  and e.state in ("launched", "launching", "unknown")]
    tokens = {r["index"]: result_token(run, r, source) for r in eff_qa["segments"]}
    timing_now = eff_qa.get("timing") or {}
    windows_off = float((eff_qa.get("alignment") or {}).get("max_abs_window_median") or 0) > \
        sl.SYNC_TOLERANCE_FRAMES
    # First a free re-stitch (once per run): a final stitched before the clips
    # were cut to their slots is late by the frames the earlier clips ran over.
    # The conform applies only without rolls (pipeline_v2.stitch_frame_counts):
    # a rolled run is never re-stitched for timing, its shifted segments go the
    # rebuild way at once (gate R1-2).
    rolled = any((s.get("pre_roll_sec") or 0) > 0 or (s.get("post_roll_sec") or 0) > 0
                 for s in res["segments_input"])
    want_restitch = not rolled and bool(
        drift["off"] or (not timing_now.get("verified") and windows_off))
    rebuild = []
    if drift["off"] and rolled:
        rebuild = list(drift["off"])
    elif drift["off"] and restitched and not drift["explained"]:
        # Still off after the re-stitch, and not by the clips' extra frames:
        # the model's picture itself is shifted -> rebuild from the original.
        rebuild = list(drift["off"])
    for name, a in answers.items():
        # The operator chose to rebuild a segment whose leftovers sat at a cut.
        if name.startswith("edge:") and _approved(a, "rebuild"):
            rsid, _, tok = name[len("edge:"):].partition(":")
            row = next((r for r in eff_qa["segments"] if r.get("run_segment_id") == rsid), None)
            if row is not None and tokens.get(row["index"]) == tok:
                rebuild.append(row["index"])
    fill = sl.fill_proposal(run.id, eff_qa, segs, context=ctx,
                            actionable=bool(res.get("actionable", True)), reserve_usd=reserve,
                            rebuild=sorted(set(rebuild)))
    items = {i["index"]: i for i in fill["segments"]}

    def held(item) -> Optional[str]:
        """The operator's standing answer that keeps this segment out of the
        automatic fixes (ship as is / by hand / Omni again), or None."""
        choice = segment_choice(answers, item.get("run_segment_id"),
                                tokens.get(item["index"], ""))
        return choice if choice in ("accept", "manual", "omni_rerun") else None

    # The live subjects of this QA round: what a pending ask must still be
    # about (stale cards are retired, design §5).
    state["live"] = {"rv": rv, "rv8": short(run.version), "asked": None, "decided": False,
                     "void": None, "vbatch": None}
    # A preview still preparing is not "in flight" any more (batch fills): the
    # previews of several segments prepare in parallel; the batch decision
    # below waits for them (bounded).
    preparing_items = {i["index"] for i in items.values() if i["state"] == "in_progress"}

    # -- best take after an operator-chosen Omni rerun (asked, not automatic) -
    best = _best_take_step(run, res, qeff, effects, takes, answers, names)
    if best is not None:
        return best

    # -- the free re-stitch: every clip cut to its own slot -------------------
    approved_start = any(
        _approved(a, "start") and (n.startswith("start:") or n.startswith("void:"))
        and a.status == "approved" for n, a in answers.items())
    if want_restitch and not restitched and not approved_start and run.audio_patch:
        # A re-stitch would drop the audio fix behind the run's Drive link: the
        # operator decides (gate R2-1).
        name = f"restitch_patch:{rv}"
        a = answers.get(name)
        if _needs_ask(a):
            return Ask(
                name=name, kind="handoff", subject="lip sync",
                question=("The picture is out of sync with the sound in %s (up to %s). The fix "
                          "is free: re-stitch the video with each segment cut to its own slot. "
                          "But the delivered video carries an audio fix (audio patch), and a "
                          "re-stitch takes the original sound again, so that audio fix would "
                          "be lost and has to be redone." % (
                              _segments_phrase(drift["off"]) or "places",
                              _frames_phrase(max((abs(v) for v in drift["late"].values()
                                                  if v is not None), default=None)))),
                proposal={"segments": drift["off"], "late_frames": drift["late"]},
                options=(Option("restitch", "Re-stitch anyway (free; the audio fix is "
                                            "dropped)"),
                         Option("accept", "Ship it as it is (lip sync stays off)"),
                         Option("manual", "I fix it by hand; checked again when the run "
                                          "changes")),
                artifacts=tuple(Artifact(p, "original vs delivered, every segment", "image",
                                         effect_id=qeff.id)
                                for p in (eff_qa.get("comparison_sheet"),) if p),
                about={f"run:{run.id}": run.version})
        if not _answered(a):
            return Wait("waiting for the operator: re-stitch over the audio fix?")
        if _approved(a, "restitch"):
            return _launch(answers, name=next_round(f"restitch:{rv}", names),
                           handler=H_RESTITCH, params={"run_id": run.id,
                                                       "discard_audio_patch": True},
                           reason="the operator chose to re-stitch over the audio fix (free)",
                           expect={f"run:{run.id}": run.version})
    elif want_restitch and not restitched and not approved_start:
        late = drift["late"]
        worst = max((abs(v) for v in late.values() if v is not None), default=None)
        return _launch(answers, name=next_round(f"restitch:{rv}", names), handler=H_RESTITCH,
                       params={"run_id": run.id},
                       reason=("the picture is out of sync with the sound in %s (up to %s)%s. "
                               "Re-stitching the video with each segment cut to its own slot "
                               "(free, nothing generated), then checking again" % (
                                   _segments_phrase(drift["off"]) or "places",
                                   _frames_phrase(worst if worst is not None else float(
                                       (eff_qa.get("alignment") or {}).get(
                                           "max_abs_window_median") or 0)),
                                   ": the model returned some segments a few frames longer "
                                   "than their slots and every later segment moved"
                                   if drift["explained"] else "")),
                       expect={f"run:{run.id}": run.version})
    if drift["explained"] and restitched:
        name = f"restitch_failed:{rv}"
        if _needs_ask(answers.get(name)):
            return Ask(
                name=name, kind="handoff", subject="timing",
                question=("The video was re-stitched with every segment cut to its slot, but "
                          "%s still run late against the sound by the same frames as before "
                          "(up to %s), so the fix did not take. Lip sync is off there. "
                          "Choose how to finish." % (
                              _segments_phrase(drift["off"]),
                              _frames_phrase(max((abs(v) for v in drift["late"].values()
                                                  if v is not None), default=None)))),
                proposal={"segments": drift["off"], "late_frames": drift["late"],
                          "expected_frames": drift["expected"]},
                options=(Option("accept", "Ship it as it is (lip sync stays off there)"),
                         Option("manual", "I fix the timing by hand; re-check when the run "
                                          "changes")),
                artifacts=tuple(Artifact(p, "original vs delivered, every segment", "image",
                                         effect_id=qeff.id)
                                for p in (eff_qa.get("comparison_sheet"),) if p),
                about={f"run:{run.id}": run.version})
        if not _answered(answers.get(name)):
            return Wait("waiting for the operator: the re-stitch did not fix the timing")

    # -- our own previews made for an operator-chosen widened mask -----------
    widen_fids = {fid for e in acts if e.handler == H_FILL_PREVIEW and e.state == "launched"
                  and e.name.startswith("do:")
                  for fid in (e.created_refs or {}).get("fill_jobs", [])}
    fills_ok = bool(fill.get("actionable"))
    # -- the stop rule's ONE automatic retry with a grown mask (design §4.3) --
    grow = _grow_plan(run, eff_qa, items, jobs, acts, names, rv, answers, fills_ok, held)
    per_seg = _per_segment_members(answers)

    # -- operator batch (computed first: acting waits for the whole batch) ---
    live_items = {k: i for k, i in items.items() if held(i) is None}
    progressing = fills_ok and (any(
        i["state"] in ("propose_preview", "propose_start", "stale_preview")
        and not _preview_limit(acts, i, rv) for i in live_items.values())
        or bool(preparing_items) or bool(grow["busy_fids"]))
    batch = [] if progressing else _operator_batch(run, res, eff_qa, fill, items, by_index,
                                                    takes, reserve, acts, rv, widen_fids, qeff,
                                                    qtag, info, answers, source,
                                                    auto=auto, grow_busy=grow["busy_fids"])
    batch_pending = any(not _answered(answers.get(a.name)) for a in batch)
    # This round's operator questions (none while the fixes progress): a
    # pending one of an older round / result is retired (design §5).
    state["live"]["asked"] = {a.name for a in batch}

    # -- act on answers -------------------------------------------------------
    if not batch_pending:
        act = _answer_action(run, answers, eff_qa, items, by_index, jobs, acts, names, rv,
                             include_batch=not progressing, fills_ok=fills_ok, source=source)
        if act is not None:
            return act

    # -- free fill progress: cancel a stale preview / make previews ----------
    # subtitle_loop's own order (a stale preview first, then every segment to
    # preview, one per tick: they prepare in parallel -- a preview is bound to
    # its own segment, so another fill does not make it stale), minus segments
    # whose previews hit the limit (gate R1-1) and segments the operator
    # answered (accept / by hand / Omni).
    nxt = None
    if fills_ok:
        eligible = [i for _, i in sorted(items.items())
                    if not (i["state"] == "propose_preview" and _preview_limit(acts, i, rv))]
        nxt = next((i for i in eligible if i["state"] == "stale_preview"), None)
        free = [i for i in eligible if held(i) is None]
        if nxt is None and grow["launch"] is None:
            nxt = next((i for i in free if i["state"] == "propose_preview"), None)
        nxt = {"index": nxt["index"], "state": nxt["state"]} if nxt is not None else None
    if nxt is not None:
        item = items[nxt["index"]]
        if nxt["state"] == "stale_preview":
            fid = item["fill_job_id"]
            return _launch(answers, name=next_round(f"cancel:{fid}", names), handler=H_FILL_CANCEL,
                           params={"run_id": run.id, "fill_id": fid},
                           reason=f"segment {item['index']}: {item.get('reason')}",
                           expect={f"run:{run.id}": run.version})
        if nxt["state"] == "propose_preview" and not _preview_limit(acts, item, rv):
            what = ("still shows captions" if item.get("why") == "leftover"
                    else "is out of sync with the sound")
            return _launch(answers,
                           name=next_round(f"preview:{item['run_segment_id']}:{rv}", names),
                           handler=H_FILL_PREVIEW,
                           params={"run_id": run.id, "run_segment_id": item["run_segment_id"],
                                   "mask_box": None, "prompt": None},
                           reason="segment %d %s: free VOID preview of a rebuild from the "
                                  "original (the platform measures the caption area)" % (
                                      item["index"], what),
                           expect={f"run:{run.id}": run.version})
    if grow["launch"] is not None:
        return dataclasses.replace(grow["launch"], consumes=unconsumed(answers))

    # -- the paid start: ONE batch over every ready preview (2026-11-03) -----
    state["live"]["decided"] = True
    if fills_ok:
        cands = _batch_candidates(items, jobs, widen_fids, grow, reserve, eff_qa, held, per_seg,
                                  tokens)
        if cands:
            young = _young_preparing(jobs, now, {r.get("run_segment_id"): r["index"]
                                                 for r in eff_qa["segments"]})
            if young:
                return Wait("VOID previews of %s are being prepared; the batch waits for them "
                            "(at most %d min)" % (_segments_phrase(young),
                                                  PREVIEW_WAIT_SEC // 60), 30)
            return _batch_step(run, cands, answers, acts, names, rv, qeff=qeff, eff_qa=eff_qa,
                               tokens=tokens, effects=effects, state=state,
                               allowance=void_allowance_usd, spent=void_auto_spent_usd,
                               usd_per_credit=usd_per_credit)

    # -- a segment the operator wants asked on its own ("per_segment") --------
    start = _start_candidate(items, jobs, widen_fids, reserve, eff_qa,
                             skip=lambda i: held(i) is not None or not _in_members(
                                 per_seg, i, tokens)) \
        if fills_ok else None
    if start is not None:
        item, job = start
        row = next((r for r in eff_qa["segments"] if r["index"] == item["index"]), {})
        tok = tokens.get(item["index"], "")
        name = void_ask_name(item["run_segment_id"], tok, job["job_id"])
        state["live"]["void"] = name
        legacy = answers.get(f"start:{job['job_id']}")
        a = answers.get(name)
        if _needs_ask(a) and (not _answered(legacy) or run.audio_patch):
            # A pending pre-2026-11-02 start:<fid> card is replaced by this one.
            queue = [i["index"] for _, i in sorted(items.items())
                     if i["index"] != item["index"] and held(i) is None
                     and _in_members(per_seg, i, tokens)
                     and i["state"] in ("propose_preview", "propose_start", "stale_preview")]
            return _void_ask(run, item, job, name, usd_per_credit, row=row, qeff=qeff,
                             by_index=by_index, takes=takes, queue=queue,
                             replaces=(f"start:{job['job_id']}",)
                             if legacy is not None and legacy.status == "pending" else ())
        if a is not None and a.status == "pending":
            return Wait(f"waiting for the operator: segment {item['index']}")

    # -- the operator batch ---------------------------------------------------
    for ask in batch:
        if _needs_ask(answers.get(ask.name)):
            return ask
    if batch_pending:
        return Wait("waiting for the operator: %s" % ", ".join(
            a.name for a in batch if not _answered(answers.get(a.name))))

    # -- clean ----------------------------------------------------------------
    if eff_qa["clean"]:
        handoff = sl.handoff_proposal(run.id, eff_qa, segs)
        deps = {f"run:{run.id}": run.version,
                f"project_detector:{project_id}": detector_version}
        return _complete(answers, output={
            "run_id": run.id, "qa_effect": qeff.id, "handoff": handoff,
            "comparison_sheet": eff_qa.get("comparison_sheet"),
            # The composites accepted without an eye check, with their evidence:
            # the hand-off card names them and shows their fill sheets (R1-7).
            "auto_accepted": [auto[f] for f in info.get("auto_accepted_fills", []) if f in auto],
            "accepted": info, "warnings": eff_qa.get("warnings", [])},
            depends_on=deps,
            reason="QA clean (%s)" % _summary(info))
    return Wait(_waiting_reason(fill, items, answers, eff_qa, tokens), 120)


# The stop rule's automatic retry (design §4.3): the old box united with the
# leftovers inside the segment, padded by subtitle_loop's widen pads plus
# these, and the glyph mask dilated by GROW_PX more; at most this many grown
# previews per composite (one paid retry).
GROW_PAD_X, GROW_PAD_Y = 0.04, 0.03
GROW_PX = 8
MAX_GROW_PREVIEWS = 2
# A preview still preparing holds the ready ones of a batch this long at most.
PREVIEW_WAIT_SEC = 900
C_BATCH_SHEET = "subtitle_batch_sheet"
MAX_SHEET_FAILURES = 2


def grown_box(fill_box, leftovers, start: float, end: float,
              centred: bool = False) -> Optional[list]:
    """The grown retry's box, or None when no leftover lies inside the segment
    (a leftover in the boundary slack is a boundary decision, not a bigger box).
    Mirrored about the frame's centre when the source captions are *centred*
    (QA row ``captions_centred``)."""
    from app.pipeline_segment_fill import FillError, mirror_x, validate_box

    boxes = [sp["box"] for sp in leftovers or () if sp.get("box")
             and float(start) - 1e-6 <= float(sp["start_sec"]) < float(end) - 1e-6]
    if not boxes:
        return None
    if fill_box:
        boxes.append(fill_box)
    px, py = sl.WIDEN_PAD_X + GROW_PAD_X, sl.WIDEN_PAD_Y + GROW_PAD_Y
    x0, y0, x1, y1 = mirror_x([min(float(b[0]) for b in boxes) - px,
                               min(float(b[1]) for b in boxes) - py,
                               max(float(b[2]) for b in boxes) + px,
                               max(float(b[3]) for b in boxes) + py], centred)
    try:
        return validate_box([max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)])
    except FillError:
        return None


def _grow_plan(run, eff_qa, items, jobs, acts, names, rv, answers, fills_ok, held) -> dict:
    """Per dirty VOID composite (fill_proposal ``stop``): its ONE automatic
    retry with a grown mask, from the durable lineage ``retry_of`` (R1-8).
    ``launch`` = the next free action (a grown preview, or the cancel of a
    stale one), ``candidates`` = ready grown previews for the batch,
    ``busy_fids`` = composites whose retry is still under way (no stop
    question yet). A composite that IS a retry, whose retry was paid and
    failed, whose previews failed twice, or that has no box to grow goes to
    the stop question (no loop)."""
    out = {"launch": None, "candidates": [], "busy_fids": set()}
    if not fills_ok:
        return out
    rows = {r["index"]: r for r in eff_qa["segments"]}
    for idx in sorted(items):
        item = items[idx]
        if item["state"] != "stop" or held(item) is not None:
            continue
        row = rows.get(idx) or {}
        finfo = row.get("fill") or {}
        fid = item.get("fill_job_id") or finfo.get("job_id")
        if not fid or finfo.get("retry_of"):
            continue                    # itself the automatic retry: the operator decides
        if _answered(answers.get(f"stop:{fid}")):
            continue                    # the operator already decided (pre-batch question)
        rsid = item["run_segment_id"]
        own = sorted((j for j in jobs.values() if j.get("retry_of") == fid),
                     key=lambda j: j.get("created_at") or "")
        latest = own[-1] if own else None
        prefix = f"grow:{rsid}:{short(fid)}"
        tries = _with_prefix(acts, prefix)
        st = (latest or {}).get("status")
        if st in ("preparing", "admitted", "compositing", "composited"):
            out["busy_fids"].add(fid)
            continue
        if st == "ready" and latest.get("stale"):
            out["busy_fids"].add(fid)
            if out["launch"] is None:
                out["launch"] = Launch(
                    name=next_round(f"cancel:{latest['job_id']}", names), handler=H_FILL_CANCEL,
                    params={"run_id": run.id, "fill_id": latest["job_id"]},
                    reason=f"segment {idx}: the grown-mask preview is stale ({latest['stale']})",
                    expect={f"run:{run.id}": run.version})
            continue
        if st == "ready":
            out["busy_fids"].add(fid)
            out["candidates"].append((dict(item, grow=True), latest))
            continue
        if any(j.get("provider_task_id") for j in own):
            continue                    # the paid retry ended without a result: ask
        if tries and tries[-1].state in ("launching", "unknown"):
            out["busy_fids"].add(fid)
            continue
        if len(tries) >= MAX_GROW_PREVIEWS or (latest or {}).get("error_code") in \
                sl.FILL_DETERMINISTIC_ERRORS:
            continue
        box = grown_box(finfo.get("mask_box") or row.get("mask_box"), row.get("leftovers"),
                        row.get("start_sec", item.get("start_sec")),
                        row.get("end_sec", item.get("end_sec")),
                        centred=bool(row.get("captions_centred")))
        if box is None:
            continue
        out["busy_fids"].add(fid)
        if out["launch"] is None:
            out["launch"] = Launch(
                name=next_round(prefix, names), handler=H_FILL_PREVIEW,
                params={"run_id": run.id, "run_segment_id": rsid, "mask_box": box,
                        "prompt": None, "mask_grow_px": GROW_PX, "retry_of": fid},
                reason=("segment %d: its VOID rebuild still shows caption words (%s); one "
                        "automatic retry with a larger area (free preview first)" % (
                            idx, _spans_phrase(row.get("leftovers")))),
                expect={f"run:{run.id}": run.version})
    return out


def _batch_candidates(items, jobs, widen_fids, grow, reserve, eff_qa, held, per_seg, tokens):
    """The ready previews ONE batch start takes (design §2): every
    ``propose_start`` segment the operator has not answered or asked to
    decide one by one, the grown retries, the operator-chosen widened masks;
    one per segment, segment order, at most SEGMENT_FILL_BATCH_MAX."""
    from app.config import settings

    out = {}
    for idx in sorted(items):
        item = items[idx]
        if item["state"] == "propose_start" and held(item) is None \
                and not _in_members(per_seg, item, tokens):
            out[idx] = (item, jobs[item["fill_job_id"]])
    for item, job in grow["candidates"]:
        out.setdefault(item["index"], (item, job))
    rows = {r.get("run_segment_id"): r for r in eff_qa["segments"]}
    for fid in sorted(widen_fids):
        job = jobs.get(fid)
        row = rows.get((job or {}).get("run_segment_id"))
        if job is None or row is None or job.get("status") != "ready" or job.get("stale") \
                or job.get("estimated_usd") is None \
                or float(job["estimated_usd"]) > reserve + 1e-9:
            continue
        item = items.get(row["index"]) or {
            "index": row["index"], "run_segment_id": row.get("run_segment_id"),
            "start_sec": row["start_sec"], "end_sec": row["end_sec"],
            "leftovers": row.get("leftovers"), "expected_mask_box": row.get("mask_box")}
        if held(item) is None:
            out.setdefault(row["index"], (item, job))
    cap = int(getattr(settings, "SEGMENT_FILL_BATCH_MAX", 12))
    return [out[k] for k in sorted(out)][:cap]


def _young_preparing(jobs, now, idx_of) -> list:
    """Segments whose preview is still preparing and young enough to wait for
    (a stuck one never holds the ready ones longer than PREVIEW_WAIT_SEC)."""
    from datetime import datetime

    out = []
    for j in jobs.values():
        if j.get("status") != "preparing":
            continue
        if now is not None and j.get("created_at"):
            try:
                age = (now - datetime.fromisoformat(j["created_at"])).total_seconds()
            except (TypeError, ValueError):
                age = 0
            if age > PREVIEW_WAIT_SEC:
                continue
        idx = idx_of.get(j.get("run_segment_id"))
        if idx is not None:
            out.append(idx)
    return sorted(set(out))


def _batch_step(run, cands, answers, acts, names, rv, *, qeff, eff_qa, tokens, effects,
                state, allowance, spent, usd_per_credit):
    """ONE decision for every ready preview (2026-11-03): inside the run's
    VOID allowance the batch starts WITHOUT a question (``autovoid:``); else
    ONE ``vbatch:`` question with one combined sheet (design §2, §3)."""
    fids = sorted(job["job_id"] for _item, job in cands)
    total = round(sum(float(job["estimated_usd"]) for _item, job in cands), 4)
    idxs = sorted(item["index"] for item, _job in cands)
    digest = short(_sha(fids), 10)
    name = f"vbatch:{rv}:{digest}"
    state["live"]["vbatch"] = name
    members = sorted(f"{item['run_segment_id']}:{tokens.get(item['index'], '')}"
                     for item, _job in cands)
    params = {"run_id": run.id, "fill_ids": fids, "approved_usd": total,
              "discard_audio_patch": bool(run.audio_patch)}
    a = answers.get(name)
    if _approved(a, "start"):
        mine = _with_prefix(acts, f"do:{name}")
        if mine and mine[-1].state in ("launched", "launching", "unknown"):
            return Wait("the approved VOID batch is being admitted")
        return _launch(answers, name=next_round(f"do:{name}", names), handler=H_FILL_BATCH_START,
                       params=dict(params, authorization="operator"), regeneration=True,
                       reason="the operator approved the VOID rebuild of %s" %
                              _segments_phrase(idxs),
                       expect={f"run:{run.id}": run.version})
    left = round(float(allowance) - float(spent), 6)
    if allowance > 0 and not run.audio_patch and total <= left + 1e-9:
        return _launch(answers, name=next_round(f"autovoid:{rv}", names),
                       handler=H_FILL_BATCH_START,
                       params=dict(params, authorization="auto", discard_audio_patch=False),
                       reason=("%s still show captions: VOID rebuild of all %d in one batch "
                               "without a question -- $%.2f reserved (about $%.2f charged) "
                               "inside the run's VOID allowance ($%.2f of $%.2f left)" % (
                                   _segments_phrase(idxs).capitalize(), len(idxs), total,
                                   sl.VOID_EXPECTED_USD * len(idxs), left, allowance)),
                       expect={f"run:{run.id}": run.version})
    if not _needs_ask(a):
        return Wait("waiting for the operator: VOID rebuild of %s" % _segments_phrase(idxs))
    # One combined sheet: per segment, delivered | original at its leftover
    # frames and the area VOID repaints (a free compute; asked without it
    # after MAX_SHEET_FAILURES).
    rows = {r["index"]: r for r in eff_qa["segments"]}
    sheets = [e for e in effects if e.kind == "compute" and e.handler == C_BATCH_SHEET
              and e.name.startswith(f"bsheet:{digest}:")]
    done = next((e for e in reversed(sheets) if e.state == "succeeded" and e.result), None)
    if done is None:
        if any(e.state in ("queued", "running") for e in sheets):
            return Wait("making the contact sheet of the VOID batch")
        if sum(1 for e in sheets if e.state == "failed") < MAX_SHEET_FAILURES:
            return Compute(next_round(f"bsheet:{digest}", names), C_BATCH_SHEET, {
                "run_id": run.id, "items": [
                    {"index": item["index"], "crop_sheet": rows.get(item["index"], {}).get(
                        "crop_sheet"), "preview_path": job.get("preview_path"),
                     "grow": bool(item.get("grow"))} for item, job in cands]},
                "contact sheet of the VOID batch (delivered | original, the repainted area)",
                timeout_sec=600)
    grown = [item["index"] for item, _job in cands if item.get("grow")]
    if run.audio_patch:
        why = ("the delivered video carries an audio fix (audio patch): the re-stitch takes "
               "the original sound again, so that fix is dropped and has to be redone")
    elif allowance <= 0:
        why = "this project starts no VOID rebuild without a question (its VOID allowance is $0)"
    else:
        why = ("it is above what the assistant may start on its own on this run: the VOID "
               "allowance is $%.2f and $%.2f of it is already used" % (allowance, spent))
    question = _batch_question(cands, rows, total, grown) + " Asked because %s." % why
    opts = (Option("start", "Rebuild all %d with VOID ($%.2f reserved, ~$%.2f)%s" % (
                len(idxs), total, sl.VOID_EXPECTED_USD * len(idxs),
                "; drops the audio fix" if run.audio_patch else ""),
                H_FILL_BATCH_START, dict(params, authorization="operator")),
            Option("per_segment", "Let me decide segment by segment (one question each, with "
                                  "Omni again / ship / by hand as alternatives)"),
            Option("accept", "Ship them as they are (the captions stay)"),
            Option("manual", "I fix them by hand; the assistant stops proposing for these "
                             "segments until their videos change"))
    if done is not None and done.result.get("sheet"):
        arts = (Artifact(done.result["sheet"], "%s: delivered | original where captions were "
                         "found, and the area VOID repaints" % _segments_phrase(idxs), "image",
                         effect_id=done.id),)
    else:
        arts = tuple(Artifact(rows[i]["crop_sheet"], f"segment {i}: delivered (left) vs "
                              "original (right)", "image", effect_id=qeff.id)
                     for i in idxs if rows.get(i, {}).get("crop_sheet"))
        arts += tuple(Artifact(job["preview_path"], f"segment {item['index']}: the area VOID "
                               "repaints", "image", run_id=run.id)
                      for item, job in cands if job.get("preview_path"))
    replaces = tuple(sorted(n for n, x in answers.items() if x.status == "pending" and (
        (n.startswith("vbatch:") and n != name)
        or (n.startswith("start:") and n[len("start:"):] in fids)
        or (n.startswith("void:") and n.rsplit(":", 1)[-1] in fids))))
    return Ask(name=name, kind="regeneration", subject=_segments_phrase(idxs),
               question=question,
               proposal={"segments": idxs, "fill_ids": fids, "members": members,
                         "reserve_usd": total, "expected_usd": round(
                             sl.VOID_EXPECTED_USD * len(idxs), 4),
                         "allowance_usd": allowance, "allowance_used_usd": spent,
                         "grown_retry": grown, "kie_credits": 0,
                         "credits_reserved": usd_to_credits(total, usd_per_credit)},
               options=opts, artifacts=arts, about={f"run:{run.id}": run.version},
               cost_credits=usd_to_credits(total, usd_per_credit), replaces=replaces)


def _batch_question(cands, rows, total: float, grown) -> str:
    """The batch question in plain words: what is wrong per segment, the one
    proposal (VOID from the original for all of them), the cost, what
    happens next (2026-11-03)."""
    idxs = sorted(item["index"] for item, _job in cands)
    one = len(idxs) == 1
    why = {item["index"]: (item.get("why") or "leftover") for item, _job in cands}
    if all(w == "leftover" for w in why.values()):
        what = "still shows burned-in captions" if one else "still show burned-in captions"
    elif all(w == "sync" for w in why.values()):
        what = "is out of sync with the sound" if one else "are out of sync with the sound"
    else:
        what = "still show captions or are out of sync with the sound"
    head = "%s %s: rebuild %s with VOID from the ORIGINAL video ($%.2f reserved, about $%.2f " \
           "charged)." % (_segments_phrase(idxs).capitalize(), what,
                          "it" if one else "all %d in one batch" % len(idxs),
                          total, sl.VOID_EXPECTED_USD * len(idxs))
    parts = []
    for item, _job in sorted(cands, key=lambda c: c[0]["index"]):
        i = item["index"]
        row = rows.get(i) or {}
        span = "%.1f-%.1f s" % (float(item.get("start_sec", row.get("start_sec", 0)) or 0),
                                float(item.get("end_sec", row.get("end_sec", 0)) or 0))
        t = row.get("timing") or item.get("timing") or {}
        if why[i] == "sync" and not row.get("leftovers"):
            parts.append("segment %d (%s): its picture runs %s %s the original sound" % (
                i, span, _frames_phrase(t.get("offset_frames")),
                "behind" if float(t.get("offset_frames") or 0) < 0 else "ahead of"))
        else:
            n = sum(int(sp.get("frames") or 0) for sp in row.get("leftovers") or ())
            parts.append("segment %d (%s): caption words in %d frames (%s)%s" % (
                i, span, n, _spans_phrase(row.get("leftovers")),
                " -- a second, automatic try with a larger area" if i in grown else ""))
    return (head + " " + "; ".join(parts).capitalize() + ". VOID repaints only the caption "
            "area of each segment (the sheet shows, per segment, the delivered frames next to "
            "the original and the area VOID repaints: check it covers the captions and not a "
            "face or the phone screen); the rest is the original picture, so the segments are "
            "also exactly in sync with the sound. The fills run at the same time; then the "
            "video is re-stitched once, re-delivered to the same Drive link and checked again.")


def _summary(info) -> str:
    parts = []
    if info["accepted_fills"]:
        parts.append("VOID composites confirmed: %d" % len(info["accepted_fills"]))
    if info["accepted_segments"]:
        parts.append("accepted as is: %s" % info["accepted_segments"])
    if info["confirmed_fp"]:
        parts.append("false positives confirmed: %s" % info["confirmed_fp"])
    return "; ".join(parts) or "detector clean"


def _waiting_reason(fill, items, answers, eff_qa, tokens=None) -> str:
    blocked = [f"segment {i['index']}: {'; '.join(i['blockers'])}" for i in items.values()
               if i["state"] == "blocked"]
    by_hand = sorted({i["index"] for i in items.values()
                      if segment_choice(answers, i.get("run_segment_id"),
                                        (tokens or {}).get(i["index"], "")) == "manual"})
    parts = blocked + (["the operator fixes %s by hand (asked again when that video changes, "
                        "or with 'Ask again' on the card)" % _segments_phrase(by_hand)]
                       if by_hand else [])
    if not fill.get("actionable") and fill.get("not_actionable_reason"):
        parts.append(fill["not_actionable_reason"])
    return "; ".join(parts) or "QA not clean: captions left in %s" % _segments_phrase(
        eff_qa.get("dirty_segments") or []) or "QA not clean"


def _preview_limit(acts, item, rv) -> bool:
    rsid = item.get("run_segment_id")
    return rsid is not None and len(_with_prefix(acts, f"preview:{rsid}:{rv}")) \
        >= MAX_PREVIEWS_PER_SEGMENT


def _start_candidate(items, jobs, widen_fids, reserve, eff_qa, skip=lambda item: False):
    for idx in sorted(items):
        item = items[idx]
        if item["state"] == "propose_start" and not skip(item):
            return item, jobs[item["fill_job_id"]]
    # A ready preview the flow made for the operator's widened mask: its start
    # is still a fill to approve, although subtitle_loop leaves it to the
    # operator (a custom box, a repeat of a paid fill, a composite segment).
    rows = {r.get("run_segment_id"): r for r in eff_qa["segments"]}
    for fid in sorted(widen_fids):
        job = jobs.get(fid)
        row = rows.get((job or {}).get("run_segment_id"))
        if job is None or row is None or job.get("status") != "ready" or job.get("stale") \
                or job.get("estimated_usd") is None \
                or float(job["estimated_usd"]) > reserve + 1e-9:
            continue
        item = items.get(row["index"]) or {
            "index": row["index"], "run_segment_id": row.get("run_segment_id"),
            "start_sec": row["start_sec"], "end_sec": row["end_sec"],
            "leftovers": row.get("leftovers"), "expected_mask_box": row.get("mask_box")}
        return item, job
    return None


def _spans_phrase(spans) -> str:
    parts = ["%.1f-%.1f s" % (float(sp["start_sec"]), float(sp["end_sec"])) for sp in spans or ()]
    return ", ".join(parts[:4]) + (" ..." if len(parts) > 4 else "")


def _provider_refused_composite(row) -> bool:
    """A composite made from the source because the provider REFUSED the
    segment (its fill froze the refusal): an Omni rerun repeats the block. A
    VOID-from-the-source run's composite has none: Omni is its fallback."""
    f = row.get("fill") or {}
    if f.get("basis") != "source":
        return False
    return f.get("block", "unknown") is not None


def _omni_allowed(seg, row, ntakes: int) -> bool:
    return bool(seg is not None and seg.run_segment_id
                and ntakes + 1 < sl.STOP_AFTER_ATTEMPTS
                and not _provider_refused_composite(row))


def _omni_credits(res) -> Optional[float]:
    cost = ((res.get("operator_options") or {}).get("omni_rerun") or {}).get("cost") or {}
    per = cost.get("credits_per_segment")
    return float(per) if per else None


_AUDIO_PATCH_NOTE = (" Note: the delivered video carries an audio fix (audio patch); any option "
                     "that rebuilds, restores or re-stitches the video drops it, and it has to "
                     "be redone.")


def _void_start_params(run, job) -> dict:
    """The fill start a ``void:`` question authorises: with a promoted audio
    patch on the run the question says the rebuild drops it, so its approval
    carries the discard (the start refuses otherwise, forever -- gate R3-1)."""
    return {"run_id": run.id, "fill_id": job["job_id"],
            "approved_usd": round(float(job["estimated_usd"]), 4),
            "discard_audio_patch": bool(run.audio_patch)}


def _void_ask(run, item, job, name, usd_per_credit, *, row, qeff, by_index, takes, queue,
              replaces=()) -> Ask:
    """The fix question of one segment (2026-11-02): what is wrong, the VOID
    rebuild from the original as the proposal (cost, the free preview sheet),
    then the alternatives -- Omni again (its cost asked separately), ship as it
    is, fix it by hand. ``start`` is the only option that authorises a launch
    (the fill start of this preview); the others are durable choices bound to
    the segment's current result (:func:`segment_choice`)."""
    idx = item["index"]
    seg = by_index.get(idx)
    est = float(job["estimated_usd"])
    params = _void_start_params(run, job)
    credits = usd_to_credits(est, usd_per_credit)
    timing = row.get("timing") or item.get("timing") or {}
    span = "%.1f-%.1f s" % (float(item["start_sec"]), float(item["end_sec"]))
    if item.get("why") == "sync" and not item.get("leftovers"):
        problem = ("Segment %d (%s) is out of sync: its picture runs %s %s the original sound "
                   "(lip sync is off), even after every segment was cut to its slot. The "
                   "caption detector finds no leftover words in it." % (
                       idx, span, _frames_phrase(timing.get("offset_frames")),
                       "behind" if float(timing.get("offset_frames") or 0) < 0 else "ahead of"))
    else:
        frames = sum(int(sp.get("frames") or 0) for sp in item.get("leftovers") or ())
        problem = ("Segment %d (%s) still shows burned-in captions: the detector finds caption "
                   "words in %d frames of the delivered video (%s) -- the model did not remove "
                   "them. The first image shows the delivered frames next to the original at "
                   "those moments." % (idx, span, frames, _spans_phrase(item.get("leftovers"))))
        if (timing.get("status") == "off"):
            problem += (" Its picture also runs %s off the original sound." %
                        _frames_phrase(timing.get("offset_frames")))
    proposal_text = (
        " Proposal: rebuild segment %d from the ORIGINAL video with a VOID fill -- VOID repaints "
        "only the caption area (shown on the preview image: check it covers the captions and "
        "not a face or the phone screen); everything else is the original picture, so the "
        "segment is also exactly in sync with the sound. Cost: $%.2f reserved, about $%.2f "
        "charged. After it, the video is re-stitched, re-delivered to the same Drive link and "
        "checked again." % (idx, est, sl.VOID_EXPECTED_USD))
    nxt = (" Next in line: %s (asked after this one: one rebuild at a time)."
           % _segments_phrase(queue)) if queue else ""
    if run.audio_patch:
        # The rebuild re-stitches with the original sound (gate R3-1).
        nxt += (" Note: the delivered video carries an audio fix (audio patch); the rebuild "
                "re-stitches with the original sound, so that audio fix is dropped and has to "
                "be redone.")
    opts = [Option("start", "Rebuild segment %d with VOID ($%.2f reserved, ~$%.2f)%s" % (
        idx, est, sl.VOID_EXPECTED_USD, "; drops the audio fix" if run.audio_patch else ""),
        H_FILL_START, params)]
    if _omni_allowed(seg, row, len(takes.get(item["run_segment_id"]) or ())):
        per = _omni_credits(qeff.result or {})
        opts.append(Option("omni_rerun", "Ask Omni again for segment %d instead (%s; its cost is "
                           "asked before it starts; on 2845eef3 a second pass came out worse)" % (
                               idx, ("about %.0f credits" % per) if per else "paid")))
    opts.append(Option("accept", "Ship segment %d as it is (%s)" % (
        idx, "the captions stay" if item.get("leftovers") else "the lip-sync offset stays")))
    opts.append(Option("manual", "I fix segment %d by hand; the assistant stops proposing for it "
                       "until its video changes ('Ask again' on the card brings this back)" % idx))
    arts = []
    if row.get("crop_sheet") and item.get("leftovers"):
        arts.append(Artifact(row["crop_sheet"], "segment %d: delivered (left) vs original "
                             "(right) where captions were found" % idx, "image",
                             effect_id=qeff.id))
    elif (qeff.result or {}).get("qa", {}).get("comparison_sheet"):
        arts.append(Artifact(qeff.result["qa"]["comparison_sheet"], "original vs delivered, "
                             "every segment", "image", effect_id=qeff.id))
    if job.get("preview_path"):
        arts.append(Artifact(job["preview_path"], "segment %d: the area VOID repaints, on the "
                             "original frames (free preview)" % idx, "image", run_id=run.id))
    return Ask(
        name=name, kind="regeneration", subject=f"segment {idx}",
        question=problem + proposal_text + nxt,
        proposal={"segment": idx, "run_segment_id": item["run_segment_id"],
                  "why": item.get("why") or "leftover",
                  "fill_id": job["job_id"], "mask_box": job.get("mask_box"),
                  "expected_mask_box": item.get("expected_mask_box"),
                  "frames_masked": job.get("frames_masked"),
                  "preview_url": job.get("preview_url"), "leftovers": item.get("leftovers"),
                  "timing": timing or None, "next_in_line": list(queue),
                  "reserve_usd": est, "expected_usd": sl.VOID_EXPECTED_USD,
                  "kie_credits": 0, "credits_reserved": credits},
        options=tuple(opts), artifacts=tuple(arts),
        about={f"run:{run.id}": run.version}, cost_credits=credits,
        replaces=tuple(replaces))


def _operator_batch(run, res, eff_qa, fill, items, by_index, takes, reserve, acts, rv,
                    widen_fids, qeff, qtag, info, answers, source, *, auto=None,
                    grow_busy=()) -> list[Ask]:
    """Every operator question of this QA round, in segment order. Only real
    decisions are asked (2026-11-02): a timing doubt is no question any more
    (the flow re-stitches or rebuilds the segment), and every question says
    what is wrong, what the assistant proposes, what it costs and what each
    option does next."""
    out: list[Ask] = []
    rows = {r["index"]: r for r in eff_qa["segments"]}
    sheets = res.get("sheets") or {}
    compare = eff_qa.get("comparison_sheet")
    # Doubtful composites are asked together, at the end (design §4.2):
    # one ``eyes:`` card, unless the operator chose to decide one by one.
    eyes_one_by_one = _eyes_per_segment(answers)
    doubtful: list = []

    def arts(*pairs):
        return tuple(Artifact(p, label, kind, effect_id=qeff.id) for p, label, kind in pairs if p)

    def ntakes(rsid):
        return len(takes.get(rsid) or ())

    def choices(seg, row, label):
        opts = stop_rule_options(run.id, seg, row, reserve=reserve, takes=ntakes(
            row.get("run_segment_id")), accept_label=label)[0]
        # No widened fill on a QA that cannot place one.
        return tuple(o for o in opts if fill.get("actionable") or o.id != "widen_mask")

    cal = eff_qa.get("calibration") or {}
    if not cal.get("calibrated"):
        # A real doubt: the detector does not see this video's captions well,
        # so "no captions left" is not proven (timing is not asked here).
        out.append(Ask(
            name=f"inconclusive:{qtag}", kind="handoff", subject="caption check",
            question=("The caption detector does not reliably see this video's captions: it "
                      "finds caption-like text in only %.0f %% of the original's frames (it "
                      "needs %.0f %%), so 'no captions left' in the delivered video is not "
                      "proven. Look at the comparison sheet (original vs delivered, every "
                      "segment)." % (100 * float(cal.get("hit_ratio") or 0),
                                     100 * sl.CALIBRATION_MIN_RATIO)),
            proposal={"calibration": cal, "warnings": eff_qa.get("warnings")},
            options=(Option("accept", "I looked: no captions are left -- continue"),
                     Option("manual", "Not clean: I set the caption band on the project page "
                                      "or fix the video by hand; the check runs again when "
                                      "the run or the detector changes")),
            artifacts=arts((compare, "original vs delivered, every segment", "image"))))
        if not _answered(answers.get(f"inconclusive:{qtag}")):
            return out
    tm = eff_qa.get("timing")
    if tm is not None and not tm.get("verified") and not eff_qa.get("sync_off_segments"):
        # Nothing measured off, but the timeline is not proven either (segments
        # too short or too still to match, or the length differs): the eye
        # decides; the assistant has no fix to propose.
        al = eff_qa.get("alignment") or {}
        out.append(Ask(
            name=f"timing:{qtag}", kind="handoff", subject="lip sync",
            question=("The assistant could not prove that the delivered picture is in sync "
                      "with the original sound: %s. Watch the video for lip sync (the "
                      "comparison sheet shows original vs delivered for every segment)." % (
                          "the delivered video is %+d frames longer than the original"
                          % int(al.get("frame_count_delta") or 0)
                          if not tm.get("length_ok", True) else
                          "the segments are too short or too still to match one by one, and "
                          "over 10-second stretches the picture is up to %s off" %
                          _frames_phrase(al.get("max_abs_window_median")))),
            proposal={"alignment": al, "timing": tm.get("segments")},
            options=(Option("accept", "Lip sync looks right: continue"),
                     Option("manual", "It is off: I fix it by hand; checked again when the run "
                                      "changes")),
            artifacts=arts((compare, "original vs delivered, every segment", "image"))))
    for idx in sorted(rows):
        row = rows[idx]
        seg = by_index.get(idx)
        item = items.get(idx)
        rsid = row.get("run_segment_id")
        fid = _fill_id(row)
        tok = result_token(run, row, source)
        choice = segment_choice(answers, rsid, tok)
        if seg is not None and seg.action == "swap" and row["verdict"] == "leftover" \
                and not fill.get("actionable") and (item is None or item["state"] != "stop"):
            out.append(_operator_ask(run, seg, row, rsid, tok, reserve, ntakes(rsid), arts,
                                     "the caption check of this video is out of date (%s), "
                                     "so no rebuild is proposed from it" % (
                                         fill.get("not_actionable_reason") or "it changed")))
            continue
        if item is not None and item["state"] == "ambiguous" and choice is None:
            amb = item.get("ambiguous") or []
            nb = sorted({a["neighbour"] for a in amb if a.get("neighbour") is not None})
            out.append(Ask(
                name=f"edge:{rsid}:{tok}", kind="handoff", subject=f"segment {idx}",
                question=("Caption words at %s sit right at the cut between segment %d and %s, "
                          "and the picture there is shifted by %s, so it is unclear which "
                          "segment they belong to. Proposal: rebuild segment %d from the "
                          "original video with VOID (free preview first, then its own approval "
                          "with the cost, $%.2f reserved); if the words belong to the "
                          "neighbour, the next check shows them there." % (
                              _spans_phrase([a["span"] for a in amb]), idx,
                              _segments_phrase(nb), _frames_phrase(
                                  (row.get("timing") or {}).get("offset_frames")), idx,
                              reserve)),
                proposal={"ambiguous": amb, "timing": row.get("timing")},
                options=(Option("rebuild", "Rebuild segment %d with VOID (preview, then its "
                                           "cost approval)" % idx),
                         Option("accept", "Ship it as it is (the words stay)"),
                         Option("manual", "I fix it by hand; asked again when this segment's "
                                          "video changes")),
                artifacts=arts((row.get("crop_sheet"), f"segment {idx}: delivered (left) vs "
                                "original (right) where captions were found", "image"),
                               (compare, "original vs delivered, every segment", "image"))))
            continue
        if item is not None and item["state"] == "operator" and item.get("why") == "sync" \
                and choice is None:
            t = row.get("timing") or {}
            opts = [Option("accept", "Ship segment %d as it is (lip sync stays off)" % idx),
                    Option("manual", "I fix segment %d by hand; asked again when its video "
                                     "changes" % idx)]
            if _omni_allowed(seg, row, ntakes(rsid)):
                per = _omni_credits(res)
                opts.insert(0, Option("omni_rerun", "Ask Omni again for segment %d (%s; its "
                                                    "cost is asked before it starts)" % (
                                          idx, ("about %.0f credits" % per) if per else "paid")))
            out.append(Ask(
                name=f"sync:{rsid}:{tok}", kind="handoff", subject=f"segment {idx}",
                question=("Segment %d's picture runs %s %s the original sound (lip sync is "
                          "off). A VOID rebuild from the original cannot help here: %s." % (
                              idx, _frames_phrase(t.get("offset_frames")),
                              "behind" if float(t.get("offset_frames") or 0) < 0
                              else "ahead of", item.get("reason"))),
                proposal={"timing": t, "reason": item.get("reason")},
                options=tuple(opts),
                artifacts=arts((compare, "original vs delivered, every segment", "image"))))
            continue
        if seg is not None and seg.action == "swap" and row.get("fill_needs_visual_check") \
                and row["verdict"] != "leftover":
            if fid and not row.get("fill_delivery_unconfirmed") and fid not in eyes_one_by_one \
                    and not _answered(answers.get(f"eye:{fid}")):
                doubtful.append((idx, seg, row, fid))
                continue
            if fid and not row.get("fill_delivery_unconfirmed"):
                finfo = row.get("fill") or {}
                look = ("The detector sees no caption, but it misses grey ghosts and smears: "
                        "look at the images. A few visibly patched frames are fine; many, or a "
                        "smear over a face or the phone screen, are not.")
                if finfo.get("source_as_is"):
                    what = ("Please check segment %d by eye: the provider refused it, so its "
                            "ORIGINAL clip was shipped as is (the detector saw no caption "
                            "there). The detector can miss captions: look for any in the "
                            "segment." % idx)
                elif finfo.get("basis") == "source":
                    what = ("Please check segment %d by eye: it was rebuilt from the ORIGINAL "
                            "video with VOID (the provider refused it). %s" % (idx, look))
                else:
                    what = ("Please check segment %d by eye: VOID repainted its captions. %s"
                            % (idx, look))
                out.append(Ask(
                    name=f"eye:{fid}", kind="handoff", subject=f"segment {idx}",
                    question=what,
                    proposal={"fill_id": fid, "mask_box": finfo.get("mask_box"),
                              "basis": finfo.get("basis") or "model_result",
                              "source_as_is": bool(finfo.get("source_as_is")),
                              "frames_masked": finfo.get("frames_masked"),
                              "result_hits": row.get("result_hits"),
                              "sheet_times": row.get("fill_sheet_times")},
                    options=choices(seg, row, "It looks fine: ship it"),
                    artifacts=arts((row.get("fill_sheet"), f"segment {idx}: delivered vs "
                                    "original inside the repainted area", "image"),
                                   (sheets.get(f"fill_clip:{idx}"), f"segment {idx}: the "
                                    "repainted area, delivered | original", "video"),
                                   (compare, "original vs delivered, every segment", "image"))))
            else:
                out.append(_operator_ask(run, seg, row, rsid, tok, reserve, ntakes(rsid), arts,
                                         ("the re-delivery of its VOID rebuild is not "
                                          "confirmed (%s), so the eye check would not be of "
                                          "this video" % row.get("fill_delivery_unconfirmed"))
                                         if fid else
                                         "its video is a VOID rebuild without its record, so "
                                         "it cannot be confirmed by id"))
            continue
        if item is not None and item["state"] == "stop" \
                and item.get("fill_job_id") in set(grow_busy or ()):
            continue                    # its automatic grown retry is under way (§4.3)
        if item is not None and item["state"] == "stop":
            out.append(Ask(
                name=f"stop:{item.get('fill_job_id') or tok}", kind="handoff",
                subject=f"segment {idx}",
                question=("Segment %d was already rebuilt with VOID and still shows caption "
                          "words (%s). The assistant does not pay for the same fix twice on "
                          "its own: choose how to finish it." % (
                              idx, _spans_phrase(row.get("leftovers")))),
                proposal={"leftovers": row.get("leftovers"), "fill_id": item.get("fill_job_id"),
                          "reason": item.get("reason")},
                options=choices(seg, row, "Ship it as it is (the words stay)"),
                artifacts=arts((row.get("crop_sheet"), f"segment {idx}: delivered (left) vs "
                                "original (right) where captions were found", "image"),
                               (row.get("fill_sheet"), f"segment {idx}: the repainted area",
                                "image"))))
            continue
        if item is not None and choice not in ("accept", "manual", "omni_rerun") and (
                item["state"] == "operator" and not any(
                j["job_id"] in widen_fids and j.get("status") == "ready"
                for j in item.get("fill_history") or [])
                or (item["state"] == "blocked" and _deterministic(item))
                or (item["state"] == "propose_preview" and _preview_limit(acts, item, rv))):
            why = item.get("reason") or "; ".join(item.get("blockers") or []) \
                or f"{MAX_PREVIEWS_PER_SEGMENT} VOID previews did not become ready"
            out.append(_operator_ask(run, seg, row, rsid, tok, reserve, ntakes(rsid), arts, why))
            continue
        if row["verdict"] == "false_positive" and not row.get("confirmed"):
            out.append(Ask(
                name=f"fp:s{idx}:{tok}", kind="handoff", subject=f"segment {idx}",
                question=("Segment %d: the detector found text in the delivered video that is "
                          "also in the original and holds still -- usually app UI or an end "
                          "card, but a caption line held on screen looks the same. Which is "
                          "it?" % idx),
                proposal={"false_positives": row.get("false_positives")},
                options=(Option("confirm", "UI / end-card text: not a caption, ship it"),
                         Option("as_caption", "A caption: rebuild the segment with VOID (free "
                                              "preview, then its cost approval)")),
                artifacts=arts((row.get("crop_sheet"), f"segment {idx}: delivered (left) vs "
                                "original (right) at that text", "image"))))
            continue
        if row["verdict"] == "keep_captions":
            out.append(Ask(
                name=f"keepcap:s{idx}:{tok}", kind="handoff", subject=f"segment {idx}",
                question=("Segment %d is kept as original footage and its original shows "
                          "captions, so they will be in the delivered video. They go away only "
                          "if the segment is made 'swap' in a new run." % idx),
                proposal={"source_caption_like_hits": row.get("source_caption_like_hits")},
                options=(Option("accept", "Ship those captions"),),
                artifacts=arts((row.get("crop_sheet"), f"segment {idx}", "image"),
                               (compare, "original vs delivered, every segment", "image"))))
    if doubtful:
        out.append(_eyes_ask(run, doubtful, qtag, qeff, sheets, compare, answers, auto or {}))
    if run.audio_patch:
        # Every action these questions lead to re-stitches the video with the
        # original sound: it drops the promoted audio fix, and says so (R4-2).
        out = [dataclasses.replace(a, question=a.question + _AUDIO_PATCH_NOTE) for a in out]
    return out


def _eyes_ask(run, doubtful, qtag, qeff, sheets, compare, answers, auto) -> Ask:
    """ONE end-of-run eye check over every doubtful VOID composite (design
    §4.2): its fill sheet and mask-box clip per segment. ``accept_all``
    confirms exactly these fill ids (``members``); ``per_segment`` asks each
    one on its own (today's ``eye:<fid>`` questions, with the stop-rule
    options); ``manual`` keeps them for the operator."""
    idxs = [d[0] for d in doubtful]
    fids = sorted(d[3] for d in doubtful)
    arts = []
    for idx, _seg, row, _fid in doubtful:
        if row.get("fill_sheet"):
            arts.append(Artifact(row["fill_sheet"], f"segment {idx}: delivered vs original "
                                 "inside the repainted area", "image", effect_id=qeff.id))
        if sheets.get(f"fill_clip:{idx}"):
            arts.append(Artifact(sheets[f"fill_clip:{idx}"], f"segment {idx}: the repainted "
                                 "area, delivered | original", "video", effect_id=qeff.id))
    if compare:
        arts.append(Artifact(compare, "original vs delivered, every segment", "image",
                             effect_id=qeff.id))
    why = []
    for idx, _seg, row, fid in doubtful:
        f = row.get("fill") or {}
        if f.get("source_as_is"):
            why.append("segment %d: the provider refused it and its ORIGINAL clip was shipped "
                       "as is (a caption the detector misses would still be there)" % idx)
        elif f.get("basis") == "source":
            why.append("segment %d: rebuilt from the original (the provider refused it)" % idx)
    auto_idx = sorted(v["index"] for v in auto.values())
    question = (
        "Please check %s by eye: VOID repainted their captions and the detector sees no "
        "caption left, but it misses grey ghosts and smears, and the assistant could not rule "
        "out that the repainted area touches a face (or it is large). A few visibly patched "
        "frames are fine; many, or a smear over a face or the phone screen, are not.%s%s" % (
            _segments_phrase(idxs), (" " + "; ".join(why) + ".") if why else "",
            (" Accepted without a question (no caption, no face under the mask): %s." %
             _segments_phrase(auto_idx)) if auto_idx else ""))
    replaces = tuple(sorted(f"eye:{fid}" for fid in fids
                            if (answers.get(f"eye:{fid}") or Answer("", "")).status == "pending"))
    return Ask(name=f"eyes:{qtag}", kind="handoff", subject=_segments_phrase(idxs),
               question=question,
               proposal={"segments": idxs, "members": fids, "fill_ids": fids,
                         "auto_accepted": auto_idx},
               options=(Option("accept_all", "They all look fine: ship them"),
                        Option("per_segment", "Some do not: let me decide segment by segment "
                                              "(restore the previous take, a larger area, by "
                                              "hand ...)"),
                        Option("manual", "I fix them by hand; checked again when their videos "
                                         "change")),
               artifacts=tuple(arts), replaces=replaces)


def _deterministic(item) -> bool:
    return any("failed:" in b and any(code in b for code in sl.FILL_DETERMINISTIC_ERRORS)
               for b in item.get("blockers") or [])


def _operator_ask(run, seg, row, rsid, tok, reserve, ntakes, arts, why) -> Ask:
    """A segment the assistant cannot fix on its own (*why*, plain words)."""
    idx = row["index"]
    opts = [Option("accept", "Ship segment %d as it is" % idx),
            Option("manual", "I fix segment %d by hand; asked again when its video changes"
                   % idx)]
    if _omni_allowed(seg, row, ntakes):
        opts.append(Option("omni_rerun", "Ask Omni again for segment %d (paid: its cost is asked "
                           "before it starts)" % idx))
    what = ("still shows caption words (%s)" % _spans_phrase(row.get("leftovers"))
            if row.get("leftovers") else "needs a decision")
    # "hand:" (2026-11-02), not the legacy "operator:" name: an old "manual" answer
    # on the same result (the only way out of 514067f2's question) must not
    # silence this one.
    return Ask(name=f"hand:{rsid}:{tok}", kind="handoff", subject=f"segment {idx}",
               question=(f"Segment {idx} {what}, and the assistant cannot fix it on its own: "
                         f"{why}."),
               proposal={"verdict": row["verdict"], "leftovers": row.get("leftovers"),
                         "reason": why},
               options=tuple(opts),
               artifacts=arts((row.get("crop_sheet"), f"segment {idx}: delivered (left) vs "
                               "original (right)", "image"),
                              (row.get("fill_sheet"), f"segment {idx}: the repainted area",
                               "image")))


def _answer_action(run, answers, eff_qa, items, by_index, jobs, acts, names, rv, *,
                   include_batch: bool, fills_ok: bool, source=None):
    """The first answered option that asks for an action not yet taken and is
    still about the segment's current result."""
    rows = {r.get("run_segment_id"): r for r in eff_qa["segments"]}
    fill_rows = {_fill_id(r): r for r in eff_qa["segments"] if _fill_id(r)}
    idx_of = {r.get("run_segment_id"): r["index"] for r in eff_qa["segments"]}
    todo = []
    for name in sorted(answers):
        a = answers[name]
        if not _approved(a):
            continue
        kind, _, rest = name.partition(":")
        if kind == "void" and a.chosen_option == "omni_rerun":
            # Not a batch question: the operator's choice acts at once (the core
            # asks the rerun's own cost approval at launch).
            rsid, tok = name.split(":")[1:3]
            row = rows.get(rsid)
            if row is not None and result_token(run, row, source) == tok:
                todo.append((row["index"], name, H_RERUN, dict(
                    _omni_rerun_params(run.id, by_index[row["index"]], row),
                    discard_audio_patch=bool(run.audio_patch)), True))
            continue
        if kind == "void":
            rest = rest.rsplit(":", 1)[-1]          # void:<rsid>:<tok>:<fid> -> the fill id
        if kind in ("start", "void") and a.chosen_option == "start":
            if not fills_ok:
                continue            # never a fill on a QA that cannot place it (R3-1)
            job = jobs.get(rest)
            if job is None or job.get("status") != "ready" or job.get("stale"):
                continue            # moved on (stale / canceled): nothing to start
            params = _void_start_params(run, job) if kind == "void" else {
                "run_id": run.id, "fill_id": rest, "approved_usd": round(float(
                    job["estimated_usd"]), 4), "discard_audio_patch": False}
            todo.append((idx_of.get(job["run_segment_id"], -1), name, H_FILL_START, params,
                         True))
        elif not include_batch:
            continue
        elif kind in ("eye", "stop") and a.chosen_option in ("widen_mask",
                                                             "restore_previous_take",
                                                             "omni_rerun"):
            row = fill_rows.get(rest)
            if row is None or (row["verdict"] == "leftover") != (kind == "stop"):
                continue            # not the segment's result / verdict any more (R2-8)
            seg = by_index[row["index"]]
            act = stop_rule_options(run.id, seg, row, reserve=0.0, takes=0)[1].get(
                a.chosen_option)
            if act is None or (act[0] == H_FILL_PREVIEW and not fills_ok):
                continue
            todo.append((row["index"], name, act[0], dict(act[1]),
                         act[0] in REGENERATION_HANDLERS))
        elif kind in ("operator", "hand", "sync", "edge") and a.chosen_option == "omni_rerun":
            rsid, _, tok = rest.partition(":")
            row = rows.get(rsid)
            if row is None or result_token(run, row, source) != tok:
                continue
            todo.append((row["index"], name, H_RERUN,
                         _omni_rerun_params(run.id, by_index[row["index"]], row), True))
    for idx, name, handler, params, regen in sorted(todo, key=lambda t: (t[0], t[1])):
        if run.audio_patch and name.startswith("start:"):
            continue                # asked again as void: (it discloses the discard, R4-1)
        if "discard_audio_patch" in params:
            params = dict(params, discard_audio_patch=bool(run.audio_patch))
        prefix = f"do:{name}"
        mine = _with_prefix(acts, prefix)
        if handler == H_FILL_PREVIEW and mine and mine[-1].state == "launched":
            # The widened preview's own lifecycle (R2-7): a stale ready preview
            # is canceled, a canceled / failed one is made again (bounded).
            wfid = next(iter((mine[-1].created_refs or {}).get("fill_jobs") or []), None)
            job = jobs.get(wfid) if wfid else None
            if job is not None and job.get("status") == "ready" and job.get("stale"):
                return _launch(answers, name=next_round(f"cancel:{wfid}", names),
                               handler=H_FILL_CANCEL, params={"run_id": run.id, "fill_id": wfid},
                               reason=f"segment {idx}: the widened preview is stale",
                               expect={f"run:{run.id}": run.version})
            if job is None or job.get("status") not in ("canceled", "failed"):
                continue            # preparing / ready / admitted / applied: in hand
            if len(mine) >= MAX_PREVIEWS_PER_SEGMENT or job.get("error_code") in \
                    sl.FILL_DETERMINISTIC_ERRORS:
                row = rows.get(job.get("run_segment_id")) or {}
                ask = f"widen_failed:{row.get('run_segment_id')}:{result_token(run, row, source)}"
                if row and _needs_ask(answers.get(ask)):
                    return Ask(name=ask, kind="handoff", subject=f"segment {idx}",
                               question=("The widened VOID preview of segment %d did not "
                                         "become ready (%s): finish the segment by hand or "
                                         "accept it." % (idx, job.get("error_code")
                                                         or job.get("status"))),
                               proposal={"fill_id": wfid, "error": job.get("error_message")},
                               options=(Option("accept", "Accept: ship the segment as it is"),
                                        Option("manual", "I fix it by hand; re-check when the "
                                               "run changes")))
                continue
        elif mine and mine[-1].state in ("launched", "launching", "unknown"):
            continue
        return _launch(answers, name=next_round(prefix, names), handler=handler, params=params,
                       reason=f"segment {idx}: the operator chose {answers[name].chosen_option}",
                       regeneration=regen, expect={f"run:{run.id}": run.version})
    return None


def _best_take_step(run, res, qeff, effects, takes, answers, names):
    """After an Omni rerun the operator chose: where the new take has more hit
    frames than the output it replaced, ASK whether to put the old one back
    (hits do not measure smears or picture changes, so this is the operator's
    call, with both crop sheets; restoring is free)."""
    reruns = [e for e in _action_effects(effects)
              if e.handler == H_RERUN and e.state == "launched"]
    if not reruns:
        return None
    rerun = reruns[-1]
    prev = [e for e in _qa_effects(effects) if e.state == "succeeded" and e.result
            and e.seq < rerun.seq and e.result.get("actionable", True) and e.result.get("qa")]
    if not prev or qeff.seq < rerun.seq:
        return None
    if not rerun.params.get("segments"):
        # Fail closed: without the effect's params (core view, request R7) the
        # rerun segments are unknown and the best-take question would be skipped.
        raise ValueError(f"rerun effect {rerun.name} carries no params (core request R7)")
    rsids = [s["run_segment_id"] for s in rerun.params.get("segments", [])]
    params = best_take_restores(run.id, rerun_rsids=rsids, prev_qa=prev[-1].result,
                                cur_qa=res, takes=takes,
                                archived=(rerun.created_refs or {}).get("takes"))
    # Bound to this exact result: a later change to the run is a new question (R2-6).
    name = f"best:{short(rerun.id)}:{short(run.version)}"
    a = answers.get(name)
    prefix = f"do:{name}"
    if params is None:
        return None
    if _approved(a, "restore"):
        mine = _with_prefix(effects, prefix)
        if mine and mine[-1].state in ("launched", "launching", "unknown"):
            return None
        return _launch(answers, name=next_round(prefix, names), handler=H_RESTORE, params=params,
                       reason="the operator restores the take the Omni rerun replaced (free)",
                       expect={f"run:{run.id}": run.version})
    if _approved(a, "keep_new"):
        return None
    if _needs_ask(a):
        prev_rows = {r.get("run_segment_id"): r for r in prev[-1].result["qa"]["segments"]}
        cur_rows = {r.get("run_segment_id"): r for r in res["qa"]["segments"]}
        cmp_rows = [{"index": cur_rows[r]["index"], "hits_before": prev_rows[r].get("result_hits"),
                     "hits_now": cur_rows[r].get("result_hits")}
                    for r in rsids if r in prev_rows and r in cur_rows]
        arts = []
        for r in rsids:
            for eff, row, when in ((prev[-1], prev_rows.get(r), "before"),
                                   (qeff, cur_rows.get(r), "now")):
                if row and row.get("crop_sheet"):
                    arts.append(Artifact(row["crop_sheet"], "segment %d %s the rerun" % (
                        row["index"], when), "image", effect_id=eff.id))
        return Ask(name=name, kind="handoff", subject="Omni rerun",
                   question=("The Omni rerun has more caption hit frames than the take it "
                             "replaced (%s). Put the old take back (free)?" % ", ".join(
                                 "segment %d: %s -> %s" % (c["index"], c["hits_before"],
                                                           c["hits_now"]) for c in cmp_rows)),
                   proposal={"segments": cmp_rows, "restore": params},
                   options=(Option("restore", "Put the previous take back"),
                            Option("keep_new", "Keep the rerun's take")),
                   artifacts=tuple(arts), about={f"run:{run.id}": run.version})
    if a.status == "pending":
        return Wait("waiting for the operator: keep or restore the Omni rerun's take")
    return None


def _provider_hold(jobs: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """The reason to wait when an ACTIVE fill's Fal task is in a state only the
    operator resolves: refused while still holding its reservation (Fal may
    have billed it), ``submit_unknown`` or ``needs_reconciliation``. Never
    resolved here: ``close`` keeps the held reservation as a charge,
    ``abandon`` releases it -- the operator's audited call."""
    for fid, j in sorted(jobs.items()):
        if j.get("status") not in ("admitted", "compositing", "composited"):
            continue
        ps = j.get("provider_status")
        held = ps == "refused" and j.get("reservation_held")
        if not held and ps not in PROVIDER_HOLD_STATES:
            continue
        tid = j.get("provider_task_id") or "?"
        cmd = "docker exec re-skin-api python -m app.pipeline_provider"
        if held:
            what = ("refused by Fal and still holds its reservation (Fal may have billed it): "
                    "`%s close %s --actor <you> --note <what you saw>` when you cannot confirm "
                    "the charge (the held reservation becomes the charge), or `... abandon %s` "
                    "only once the Fal dashboard shows nothing was billed (it releases the "
                    "reservation)" % (cmd, tid, tid))
        elif ps == "submit_unknown":
            what = ("submit_unknown (the POST may or may not have reached Fal): `%s adopt %s "
                    "<request id> --actor .. --note ..` if the request is on the Fal dashboard; "
                    "`... abandon %s` only once you confirmed no request exists; `... close %s` "
                    "if you cannot tell" % (cmd, tid, tid, tid))
        else:
            what = ("needs_reconciliation: `%s resume %s --actor .. --note ..` (a fresh cancel), "
                    "or `... close %s` when you cannot confirm the outcome" % (cmd, tid, tid))
        return ("the VOID fill %s of run segment %s waits on its Fal task %s, which is %s "
                "(`%s list`; docs/fal-provider-design.md). The autopilot never resolves it "
                "itself; once the task ends, the fill ends failed and the autopilot asks how to "
                "finish the segment" % (fid, j.get("run_segment_id"), tid, what, cmd))
    return None


_RETRY_ASKS = ("retry:", "blocked:", "blocked_failed:")


def _retry_choice(answers, acts):
    """The operator's approved ``retry`` choice not yet launched: the plain
    retry ask, or the retry option of a blocked-segment ask."""
    for name in sorted(answers):
        a = answers[name]
        # Unconsumed only: a consumed answer authorised (or was spent by) an
        # earlier decision and never authorises a new paid launch.
        if not name.startswith(_RETRY_ASKS) or a.status != "approved" \
                or not _approved(a, "retry"):
            continue
        mine = _with_prefix(acts, f"do:{name}")
        if mine and mine[-1].state in ("launched", "launching", "unknown"):
            continue
        return name
    return None


def _pending_retry_asks(answers, keep: str) -> tuple:
    """Pending retry / blocked asks of the step a new one replaces (core
    ``Ask.replaces``): the live case's retry-only ask is superseded by the
    VOID-from-source proposal without restarting the autopilot."""
    return tuple(sorted(n for n, a in answers.items()
                        if n != keep and n.startswith(_RETRY_ASKS) and a.status == "pending"))


def _plan_retry(run, rv, names, acts, answers, qas_all=(), *, ctx=None, jobs=None,
                project_id: Optional[str] = None, detector_version: Optional[str] = None,
                usd_per_credit: Optional[float] = None, void_allowance: float = 0.0,
                void_spent: float = 0.0, now=None, effects=()):
    unfinished = [s["index"] for s in run.segments
                  if not s.get("has_result", s.get("status") == "completed")]
    if run.status == "failed" and not unfinished and not run.preprocess_hole:
        # Counted per delivery episode -- every free retry since the last QA of a
        # done run -- not per run version: a re-stitch that rewrites the final
        # before failing again changes the version each time (R3-6).
        last_done = max((e.seq for e in qas_all), default=-1)
        mine = [e for e in acts if e.handler == H_RETRY_DELIVERY and e.seq > last_done]
        if len(mine) >= MAX_REDELIVERIES:
            ask = f"redeliver_failed:{rv}"
            a = answers.get(ask)
            if _needs_ask(a):
                return Ask(name=ask, kind="handoff", subject=f"run {short(run.id)}",
                           question=("The stitch / delivery of run %s failed %d times (%s): fix "
                                     "it by hand (flow-doc step 7: manual assembly, then "
                                     "Retry); the autopilot re-checks when the run changes." % (
                                         short(run.id), MAX_REDELIVERIES, run.error_message)),
                           proposal={"run_id": run.id, "error": run.error_message},
                           options=(Option("manual", "I fix it by hand"),),
                           about={f"run:{run.id}": run.version})
            return Wait("waiting for the operator: the run's delivery failed %d times"
                        % MAX_REDELIVERIES, 300)
        return _launch(answers, name=next_round(f"redeliver:{rv}", names),
                       handler=H_RETRY_DELIVERY, params={"run_id": run.id},
                       reason="every segment is finished: retry the stitch and delivery (free)",
                       expect={f"run:{run.id}": run.version})

    # 0. VOID from the source (2026-11-03): an unfinished segment of such a
    #    run is filled from the source, never retried with the model unless
    #    the operator chose that segment's fallback.
    if run.void_source and unfinished and ctx is not None:
        return _plan_void_source(run, rv, names, acts, answers, ctx=ctx, jobs=jobs or {},
                                 usd_per_credit=usd_per_credit, allowance=void_allowance,
                                 spent=void_spent, now=now, effects=effects)

    # 1. The operator's explicit retry choice wins (also on a legacy retry-only
    #    ask answered before the blocked route replaced it).
    chosen = _retry_choice(answers, acts)
    if chosen is not None:
        return _launch(answers, name=next_round(f"do:{chosen}", names), handler=H_RETRY_RUN,
                       params={"run_id": run.id}, regeneration=True,
                       reason="the operator approved the retry of segments %s" % unfinished,
                       expect={f"run:{run.id}": run.version})

    # 2. Segments the provider refused with a content filter: produced by a VOID
    #    fill from the SOURCE (operator decision 2026-11-02), first and one at a
    #    time -- a source fill never re-stitches while anything is unfinished,
    #    so nothing else is resubmitted or billed meanwhile.
    blocked = [sg for sg in sorted(run.segments, key=lambda x: x["index"])
               if sg.get("content_block") and sg.get("id") and not sg.get("has_result")]
    if blocked and ctx is not None:
        return _plan_blocked(run, blocked[0], names, acts, answers, ctx=ctx, jobs=jobs or {},
                             project_id=project_id, detector_version=detector_version,
                             usd_per_credit=usd_per_credit, unfinished=unfinished)

    if not unfinished:
        # Every segment has a result but the run cannot be finished for free:
        # the source promotion refused a requeue that would bill (R2-6/15), or
        # a preprocessing hole would fail / bill the requeue (gate R2).
        ask = f"stuck:{rv}"
        a = answers.get(ask)
        if _needs_ask(a):
            return Ask(name=ask, kind="handoff", subject=f"run {short(run.id)}",
                       question=("Run %s is %s although every segment has a result: %s. Finish "
                                 "it by hand (Retry Run bills what it shows); the autopilot "
                                 "re-checks when the run changes." % (
                                     short(run.id), run.status, run.error_message)),
                       proposal={"run_id": run.id, "error": run.error_message},
                       options=(Option("manual", "I finish it by hand"),),
                       about={f"run:{run.id}": run.version})
        return Wait(f"waiting for the operator: run {short(run.id)} needs a hand", 300)

    # 3. Everything else (transient failures, missing rows): the paid retry.
    name = f"retry:{rv}"
    a = answers.get(name)
    if _approved(a, "manual"):
        return Wait("the operator fixes run %s by hand; re-checked when the run changes"
                    % short(run.id), 300)
    if _needs_ask(a):
        return Ask(name=name, kind="regeneration", subject=f"run {short(run.id)}",
                   question=("Run %s is %s: segments %s have no result (%s). Retry them (paid)? "
                             "A segment the provider refuses with a content filter (a Google "
                             "safety block, sensitive content) goes to a VOID fill from the "
                             "source instead: retrying the same frames usually repeats the "
                             "block." % (short(run.id), run.status, unfinished,
                                         run.error_message or "no error message")),
                   proposal={"run_id": run.id, "unfinished_segments": unfinished,
                             "error": run.error_message},
                   options=(Option("retry", "Retry the unfinished segments (paid)", H_RETRY_RUN,
                                   {"run_id": run.id}),),
                   about={f"run:{run.id}": run.version},
                   replaces=_pending_retry_asks(answers, name))
    return Wait(f"waiting for the operator: retry of run {short(run.id)}")


MAX_SOURCE_PREVIEWS = 3


def _plan_void_source(run, rv, names, acts, answers, *, ctx, jobs, usd_per_credit, allowance,
                      spent, now, effects):
    """A VOID-from-the-source run that still has segments without a result
    (docs/segment-fill-batch-design.md §7.3), one decision per tick:

    1. a ready source preview that went stale -> cancel it (free);
    2. a segment without a live preview -> its free source preview (one per
       tick: they prepare in parallel; at most MAX_SOURCE_PREVIEWS);
    3. a ready preview that found no caption -> applied as is (free, no ask);
    4. every other ready preview -> ONE batch (inside the run's VOID allowance
       without a question, else one ``vbatch:`` question);
    5. a segment VOID cannot do (too short with captions, unsupported media,
       a paid fill that ended without a result, previews that keep failing) or
       whose model fallback failed -> one question each: Omni for that segment
       (paid, its cost asked), or by hand. Never a model call on its own."""
    segs = {s["id"]: s for s in run.segments if s.get("id")}
    unfinished = [s for s in sorted(run.segments, key=lambda x: x["index"])
                  if s.get("id") and not s.get("has_result")]
    own = {rsid: sorted((j for j in jobs.values() if j.get("run_segment_id") == rsid
                         and j.get("basis") == "source"),
                        key=lambda j: j.get("created_at") or "")
           for rsid in segs}
    cands, asks, young_idx = [], [], []
    stale_launch = preview_launch = as_is_launch = None
    run_blockers = [b for b in ctx.get("run_blockers") or []
                    if not b.startswith("Fal is disabled") and "one batch at a time" not in b
                    and "one fill at a time" not in b]
    fal_off = any(b.startswith("Fal is disabled") for b in ctx.get("run_blockers") or [])
    for seg in unfinished:
        rsid, idx = seg["id"], seg["index"]
        if seg.get("model_fallback"):
            # The operator sent it to the model: a failure is asked, never
            # silently refilled or retried. "Back to VOID" makes it a source
            # segment again (the source promotion clears the flag; gate R1-1).
            if seg.get("status") != "failed":
                continue
            fb = _void_fallback_failed_ask(run, seg, answers)
            if not _approved(answers.get(fb.name), "void"):
                asks.append(fb)
                continue
        mine = own.get(rsid) or []
        latest = mine[-1] if mine else None
        st = (latest or {}).get("status")
        tries = _with_prefix(acts, f"vpreview:{rsid}")
        if st == "preparing":
            young_idx.append(idx)
            continue
        if st == "ready" and latest.get("stale"):
            stale_launch = stale_launch or _launch(
                answers, name=next_round(f"cancel:{latest['job_id']}", names),
                handler=H_FILL_CANCEL, params={"run_id": run.id, "fill_id": latest["job_id"]},
                reason=f"segment {idx}: the source preview is stale ({latest['stale']})",
                expect={f"run:{run.id}": run.version})
            continue
        if st == "ready" and latest.get("source_as_is"):
            as_is_launch = as_is_launch or _launch(
                answers, name=next_round(f"asis:{latest['job_id']}", names),
                handler=H_FILL_APPLY_SOURCE,
                params={"run_id": run.id, "fill_id": latest["job_id"]},
                reason=("segment %d: the detector finds no caption in its original clip -- "
                        "the clip goes in as it is (free; the eye check after the delivery "
                        "covers a detector miss)" % idx),
                expect={f"run:{run.id}": run.version})
            continue
        if st == "ready":
            if latest.get("estimated_usd") is None:
                continue
            cands.append(({"index": idx, "run_segment_id": rsid, "why": "source",
                           "start_sec": seg.get("start_sec"), "end_sec": seg.get("end_sec")},
                          latest))
            continue
        if st in ("admitted", "compositing", "composited"):
            continue
        paid_failed = any(j.get("provider_task_id") and j.get("status") == "failed" for j in mine)
        determ = (latest or {}).get("status") == "failed" and (latest or {}).get(
            "error_code") in BLOCKED_DETERMINISTIC_ERRORS
        if paid_failed or determ or len(tries) >= MAX_SOURCE_PREVIEWS:
            asks.append(_void_cannot_ask(run, seg, latest, answers, paid_failed=paid_failed,
                                         usd_per_credit=usd_per_credit))
            continue
        if tries and tries[-1].state in ("launching", "unknown"):
            young_idx.append(idx)
            continue
        if run_blockers:
            continue
        preview_launch = preview_launch or _launch(
            answers, name=next_round(f"vpreview:{rsid}", names), handler=H_FILL_PREVIEW,
            params={"run_id": run.id, "run_segment_id": rsid, "mask_box": None,
                    "prompt": None, "basis": "source"},
            reason=("segment %d: free preview of its VOID fill from the source (the platform "
                    "measures the caption area on the original clip)" % idx),
            expect={f"run:{run.id}": run.version})
    for d in (stale_launch, preview_launch, as_is_launch):
        if d is not None:
            return d
    if young_idx:
        return Wait("VOID previews of %s are being prepared" % _segments_phrase(young_idx), 30)
    if cands:
        if fal_off:
            return Wait("Fal is disabled (FAL_ENABLED / FAL_API_KEY): the VOID fills of %s wait"
                        % _segments_phrase([c[0]["index"] for c in cands]), 300)
        from app.config import settings

        cap = int(getattr(settings, "SEGMENT_FILL_BATCH_MAX", 12))
        return _source_batch_step(run, cands[:cap], answers, acts, names, rv,
                                  usd_per_credit=usd_per_credit, allowance=allowance,
                                  spent=spent, effects=effects)
    for a in asks:
        if a is not None and _needs_ask(answers.get(a.name)):
            return a
    for a in asks:
        if a is None:
            continue
        ans = answers.get(a.name)
        if _approved(ans, "omni") and ans.status == "approved":
            seg = next(s for s in unfinished if a.name.split(":")[1] == s["id"])
            return _launch(answers, name=next_round(f"do:{a.name}", names), handler=H_RERUN,
                           params={"run_id": run.id, "segments": [{
                               "run_segment_id": seg["id"], "prompt": None}],
                               "discard_audio_patch": False},
                           regeneration=True,
                           reason=f"segment {seg['index']}: the operator chose Omni for it",
                           expect={f"run:{run.id}": run.version})
    if run_blockers:
        return Wait("the VOID fills from the source wait: %s" % "; ".join(run_blockers), 120)
    return Wait("waiting for the operator: %s" % ", ".join(
        a.name for a in asks if a is not None) if asks else "VOID from the source in progress",
        300 if asks else 30)


def _void_cannot_ask(run, seg, latest, answers, *, paid_failed, usd_per_credit) -> Ask:
    idx = seg["index"]
    tag = short(_sha([(latest or {}).get("job_id"), seg.get("id")]), 10)
    name = f"vsrc_failed:{seg['id']}:{tag}"
    why = ("its VOID fill ended at Fal without a result (%s)" % ((latest or {}).get("error_code")
                                                                or "failed")
           if paid_failed else
           "VOID cannot take it: %s" % ((latest or {}).get("error_message")
                                        or (latest or {}).get("error_code")
                                        or "%d previews did not become ready" % MAX_SOURCE_PREVIEWS))
    return Ask(name=name, kind="regeneration", subject=f"segment {idx}",
               question=("Segment %d (%.2f-%.2f s) has no result yet and %s. Choose how to "
                         "finish it: Omni for this segment (paid: its cost is asked before it "
                         "starts), or by hand (the run stitches once every segment has a "
                         "result)." % (idx, float(seg.get("start_sec") or 0),
                                       float(seg.get("end_sec") or 0), why)),
               proposal={"segment": idx, "run_segment_id": seg["id"],
                         "error": (latest or {}).get("error_message")},
               options=(Option("omni", "Omni for segment %d (paid; cost asked)" % idx),
                        Option("manual", "I finish it by hand")),
               about={f"run:{run.id}": run.version})


def _void_fallback_failed_ask(run, seg, answers) -> Ask:
    idx = seg["index"]
    name = f"vfallback_failed:{seg['id']}:{short(_sha(seg.get('error_message') or ''), 8)}"
    return Ask(name=name, kind="regeneration", subject=f"segment {idx}",
               question=("Segment %d was sent to Omni as a fallback and failed (%s). Try Omni "
                         "again (paid: its cost is asked), go back to VOID from the source "
                         "(free preview first), or finish it by hand." % (
                             idx, (seg.get("error_message") or "no message")[:200])),
               proposal={"segment": idx, "run_segment_id": seg["id"],
                         "error": seg.get("error_message")},
               options=(Option("omni", "Omni again for segment %d (paid; cost asked)" % idx),
                        Option("void", "Back to VOID from the source (free preview first)"),
                        Option("manual", "I finish it by hand")),
               about={f"run:{run.id}": run.version})


def _source_batch_step(run, cands, answers, acts, names, rv, *, usd_per_credit, allowance,
                       spent, effects):
    """ONE batch over the ready source previews of a VOID-from-the-source run:
    inside the run's VOID allowance without a question, else ONE question
    (with every member's preview sheet)."""
    fids = sorted(job["job_id"] for _item, job in cands)
    total = round(sum(float(job["estimated_usd"]) for _item, job in cands), 4)
    idxs = sorted(item["index"] for item, _job in cands)
    name = f"vbatch:{rv}:{short(_sha(fids), 10)}"
    params = {"run_id": run.id, "fill_ids": fids, "approved_usd": total,
              "discard_audio_patch": False}
    a = answers.get(name)
    if _approved(a, "start"):
        mine = _with_prefix(acts, f"do:{name}")
        if mine and mine[-1].state in ("launched", "launching", "unknown"):
            return Wait("the approved VOID batch is being admitted")
        return _launch(answers, name=next_round(f"do:{name}", names), handler=H_FILL_BATCH_START,
                       params=dict(params, authorization="operator"), regeneration=True,
                       reason="the operator approved the VOID fills of %s" % _segments_phrase(idxs),
                       expect={f"run:{run.id}": run.version})
    left = round(float(allowance) - float(spent), 6)
    if allowance > 0 and total <= left + 1e-9:
        return _launch(answers, name=next_round(f"autovoid:{rv}", names),
                       handler=H_FILL_BATCH_START, params=dict(params, authorization="auto"),
                       reason=("VOID from the source for %s in one batch without a question -- "
                               "$%.2f reserved (about $%.2f charged) inside the run's VOID "
                               "allowance ($%.2f of $%.2f left)" % (
                                   _segments_phrase(idxs), total,
                                   sl.VOID_EXPECTED_USD * len(idxs), left, allowance)),
                       expect={f"run:{run.id}": run.version})
    if not _needs_ask(a):
        return Wait("waiting for the operator: VOID fills of %s" % _segments_phrase(idxs))
    why = ("this project starts no VOID fill without a question (its VOID allowance is $0)"
           if allowance <= 0 else
           "it is above what the assistant may start on its own on this run: the VOID "
           "allowance is $%.2f and $%.2f of it is already used" % (allowance, spent))
    arts = tuple(Artifact(job["preview_path"], "segment %d: the caption area VOID repaints, on "
                          "the original frames" % item["index"], "image", run_id=run.id)
                 for item, job in cands if job.get("preview_path"))
    return Ask(
        name=name, kind="regeneration", subject=_segments_phrase(idxs),
        question=("Remove the captions of %s with VOID from the ORIGINAL video, in one batch "
                  "($%.2f reserved, about $%.2f charged): VOID repaints only the caption area "
                  "(the preview sheets show it on the original frames); the rest of each "
                  "segment is the original picture, frame-exact with the sound, and nothing "
                  "goes to Omni. The fills run at the same time; then the video is stitched, "
                  "delivered and checked. Asked because %s." % (
                      _segments_phrase(idxs), total, sl.VOID_EXPECTED_USD * len(idxs), why)),
        proposal={"segments": idxs, "fill_ids": fids, "reserve_usd": total,
                  "expected_usd": round(sl.VOID_EXPECTED_USD * len(idxs), 4),
                  "allowance_usd": allowance, "allowance_used_usd": spent, "basis": "source",
                  "credits_reserved": usd_to_credits(total, usd_per_credit)},
        options=(Option("start", "Start all %d VOID fills ($%.2f reserved, ~$%.2f)" % (
                     len(idxs), total, sl.VOID_EXPECTED_USD * len(idxs)),
                     H_FILL_BATCH_START, dict(params, authorization="operator")),
                 Option("manual", "Not now: I decide by hand")),
        artifacts=arts, about={f"run:{run.id}": run.version},
        cost_credits=usd_to_credits(total, usd_per_credit),
        replaces=tuple(sorted(n for n, x in answers.items() if x.status == "pending"
                              and n.startswith("vbatch:") and n != name)))


def _blocked_about(run, project_id, detector_version) -> dict:
    about = {f"run:{run.id}": run.version}
    if project_id and detector_version is not None:
        about[f"project_detector:{project_id}"] = detector_version
    return about


def _retry_option(run, unfinished) -> Option:
    """``retry_run`` resubmits EVERY unfinished active swap segment (gate R1):
    the option says which, and its cost (the core's estimate) covers them all."""
    return Option("retry", "Retry the run with the model instead (paid; resubmits every "
                  "unfinished segment: %s; the refused frames usually get the same block)"
                  % list(unfinished), H_RETRY_RUN, {"run_id": run.id})


def _plan_blocked(run, seg, names, acts, answers, *, ctx, jobs, project_id, detector_version,
                  usd_per_credit, unfinished=()):
    """One provider-refused segment: free source preview -> ``blocked:<fid>``
    (VOID fill from the source, or the source as is when its clip shows no
    caption; retry with the model as the secondary option) -> the fill. Bounded:
    a preview that keeps failing, a deterministic refusal or a source fill that
    ended at Fal without a result asks the operator, never loops."""
    rsid, idx = seg["id"], seg["index"]
    # The blocked attempt's provider task: a retry makes a new one, so the
    # preview bound is per (segment, refusal), not per run version (R1-9).
    tag = short(_sha(seg.get("seedance_task_id") or "none"), 12)
    sc = (ctx.get("segments") or {}).get(rsid) or {}
    own = [dict(j, run_segment_id=rsid) for j in sc.get("jobs") or []
           if j.get("basis") == "source"
           and (j.get("block_task") or None) == (seg.get("seedance_task_id") or None)]
    block = seg.get("content_block")
    why = "%s (%s)" % (block, seg.get("error_message") or "no message")
    if block == "sensitive_audio":
        # A fill rebuilds the picture only: the run ships the ORIGINAL soundtrack
        # (audio_mode original), i.e. the audio the provider refused (R2-10).
        why += ("; note: the provider refused the AUDIO, and the delivered run keeps the "
                "original soundtrack")
    about = _blocked_about(run, project_id, detector_version)

    # A source fill that ended at Fal without a result (refused / failed /
    # abandoned, after the operator resolved a held task): never a second paid
    # attempt on its own.
    hard = next((j for j in own if j.get("status") == "failed"
                 and (j.get("provider_task_id")
                      or j.get("error_code") in BLOCKED_DETERMINISTIC_ERRORS)), None)
    ready = next((j for j in own if j.get("status") == "ready"), None)
    previews = _with_prefix(acts, f"bpreview:{rsid}:{tag}")

    if ready is not None and ready.get("stale"):
        return _launch(answers, name=next_round(f"cancel:{ready['job_id']}", names),
                       handler=H_FILL_CANCEL,
                       params={"run_id": run.id, "fill_id": ready["job_id"]},
                       reason=f"segment {idx}: the source preview is stale ({ready['stale']})",
                       expect={f"run:{run.id}": run.version})
    if ready is not None and (hard is None or own.index(ready) < own.index(hard)):
        # (a preview made after the failure -- e.g. by the operator with a
        # wider box -- is this segment's current proposal)
        return _blocked_ask(run, seg, ready, answers, ctx=ctx, about=about, why=why,
                            usd_per_credit=usd_per_credit, names=names, unfinished=unfinished)
    if hard is not None or len(previews) >= MAX_BLOCKED_PREVIEWS:
        last = hard or next((j for j in own if j.get("status") in ("failed", "canceled")), {})
        reason = ("the VOID fill from the source ended without a result at Fal (%s: %s)" % (
            last.get("error_code") or last.get("provider_status"),
            last.get("error_message") or "") if last.get("provider_task_id") else
            "the free source preview did not become ready (%s: %s)" % (
                last.get("error_code") or last.get("status") or "%d previews" % len(previews),
                last.get("error_message") or ""))
        return _blocked_failed_ask(run, seg, tag, answers, reason=reason, why=why,
                                   unfinished=unfinished)
    if previews and previews[-1].state in ("launching", "unknown"):
        return Wait(f"segment {idx}: the source preview admission is being reconciled")
    blockers = list(sc.get("blockers") or [])
    if blockers:
        # The segment's own gates for a source fill (audio mode, an inactive
        # segment, pre/post roll ...) do not change by waiting: the operator.
        return _blocked_failed_ask(run, seg, tag, answers, why=why, unfinished=unfinished,
                                   reason="no fill from the source can be made: %s"
                                   % "; ".join(blockers))
    run_blockers = [b for b in ctx.get("run_blockers") or [] if not b.startswith("Fal is disabled")
                    and "one fill at a time" not in b and "one batch at a time" not in b]
    if run_blockers:
        return Wait("segment %d (refused by the provider: %s) cannot get a source preview yet: %s"
                    % (idx, block, "; ".join(run_blockers)), 120)
    return _launch(answers, name=next_round(f"bpreview:{rsid}:{tag}", names),
                   handler=H_FILL_PREVIEW,
                   params={"run_id": run.id, "run_segment_id": rsid, "mask_box": None,
                           "prompt": None, "basis": "source"},
                   reason=("segment %d was refused by the provider (%s): free preview of a VOID "
                           "fill from the source (the platform measures the caption box)"
                           % (idx, block)),
                   expect={f"run:{run.id}": run.version})


def _blocked_ask(run, seg, job, answers, *, ctx, about, why, usd_per_credit, names,
                 unfinished=()):
    idx, fid = seg["index"], job["job_id"]
    name = f"blocked:{fid}"
    as_is = bool(job.get("source_as_is"))
    fal_off = any(b.startswith("Fal is disabled") for b in ctx.get("run_blockers") or [])
    a = answers.get(name)
    if _approved(a, "void", "source"):
        if a.chosen_option == "source":
            if not as_is:
                return Wait(f"segment {idx}: the answer does not match the preview any more")
            return _launch(answers, name=next_round(f"do:{name}", names),
                           handler=H_FILL_APPLY_SOURCE, params={"run_id": run.id, "fill_id": fid},
                           reason=f"segment {idx}: the operator ships the source clip as is "
                                  "(free; the detector saw no caption)",
                           expect={f"run:{run.id}": run.version})
        if as_is:
            return Wait(f"segment {idx}: the answer does not match the preview any more")
        if fal_off:
            return Wait("Fal is disabled (FAL_ENABLED / FAL_API_KEY): the approved VOID fill of "
                        f"segment {idx} waits", 300)
        return _launch(answers, name=next_round(f"do:{name}", names), handler=H_FILL_START,
                       params=_start_params(run, job), regeneration=True,
                       reason=f"segment {idx}: the operator approved the VOID fill from the source",
                       expect={f"run:{run.id}": run.version})
    if _approved(a, "manual"):
        return Wait(f"the operator finishes segment {idx} by hand; re-checked when the run "
                    "changes", 300)
    if not _needs_ask(a):
        return Wait(f"waiting for the operator: segment {idx} was refused by the provider")
    if fal_off and not as_is:
        return Wait("Fal is disabled (FAL_ENABLED / FAL_API_KEY): segment %d (refused by the "
                    "provider: %s) needs a VOID fill from the source" % (idx, seg.get(
                        "content_block")), 300)
    est = float(job.get("estimated_usd") or 0.0)
    credits = usd_to_credits(est, usd_per_credit) if not as_is else 0.0
    retry = _retry_option(run, unfinished or [idx])
    if as_is:
        primary = Option("source", "Ship the source clip of segment %d as is (free)" % idx,
                         H_FILL_APPLY_SOURCE, {"run_id": run.id, "fill_id": fid})
        question = (
            "Segment %d (%.2f-%.2f s) was refused by the provider: %s. The caption detector "
            "finds NO caption in its source clip (%s), so the proposal is to ship the source "
            "clip as is -- nothing generated, free. A detector miss is possible (other styles, "
            "faint text): watch the clip and the frame sheet (detector band outlined). After "
            "delivery the segment gets its eye check like every VOID composite." % (
                idx, seg_start(job, seg), seg_end(job, seg), why,
                job.get("source_as_is_reason") or "no caption"))
    else:
        primary = Option("void", "VOID fill from the source ($%.2f reserved)" % est,
                         H_FILL_START, _start_params(run, job))
        question = (
            "Segment %d (%.2f-%.2f s) was refused by the provider: %s. Retrying the same frames "
            "usually repeats the block. Proposal: a VOID fill from the SOURCE -- VOID inpaints "
            "only the caption mask of the source clip, the rest of the frame stays original "
            "(the model is not involved, so the refused content is left as it was). Reserves "
            "$%.2f, charged ~$%.2f. The preview sheet shows the mask: check it covers the "
            "caption and not a face or the phone screen." % (
                idx, seg_start(job, seg), seg_end(job, seg), why, est, sl.VOID_EXPECTED_USD))
    arts = []
    if job.get("preview_path"):
        arts.append(Artifact(job["preview_path"], "segment %d: %s" % (
            idx, "source frames, detector band" if as_is else "the caption mask on the source"),
            "image", run_id=run.id))
    if as_is and job.get("clip_path"):
        arts.append(Artifact(job["clip_path"], f"segment {idx}: the source clip (what ships)",
                             "video", run_id=run.id))
    return Ask(
        name=name, kind="regeneration", subject=f"segment {idx}", question=question,
        proposal={"segment": idx, "run_segment_id": seg["id"], "fill_id": fid,
                  "basis": "source", "source_as_is": as_is,
                  "source_as_is_reason": job.get("source_as_is_reason"),
                  "block": seg.get("content_block"), "provider_error": seg.get("error_message"),
                  "mask_box": job.get("mask_box"), "frames_masked": job.get("frames_masked"),
                  "preview_url": job.get("preview_url"), "reserve_usd": est,
                  "expected_usd": 0.0 if as_is else sl.VOID_EXPECTED_USD,
                  "kie_credits": 0, "credits_reserved": credits},
        options=(primary, retry), artifacts=tuple(arts), about=about, cost_credits=credits,
        replaces=_pending_retry_asks(answers, name))


def seg_start(job, seg) -> float:
    return float(seg.get("start_sec") or 0.0)


def seg_end(job, seg) -> float:
    return float(seg.get("end_sec") or 0.0)


def _start_params(run, job) -> dict:
    return {"run_id": run.id, "fill_id": job["job_id"],
            "approved_usd": round(float(job.get("estimated_usd") or 0.0), 4),
            "discard_audio_patch": False}


def _blocked_failed_ask(run, seg, tag, answers, *, reason, why, unfinished=()):
    idx = seg["index"]
    name = f"blocked_failed:{seg['id']}:{tag}"
    a = answers.get(name)
    if _approved(a, "manual"):
        return Wait(f"the operator finishes segment {idx} by hand; re-checked when the run "
                    "changes", 300)
    if not _needs_ask(a):
        return Wait(f"waiting for the operator: segment {idx} (refused by the provider)")
    return Ask(name=name, kind="regeneration", subject=f"segment {idx}",
               question=("Segment %d was refused by the provider (%s) and %s. Retry the run "
                         "with the model (paid; it resubmits every unfinished segment: %s; the "
                         "refused frames usually repeat the block), or finish it "
                         "by hand (e.g. a fill from the source on the run page, or move the "
                         "segment boundary past the refused frames); the autopilot re-checks "
                         "when the run changes." % (idx, why, reason,
                                                    list(unfinished or [idx]))),
               proposal={"segment": idx, "run_segment_id": seg["id"],
                         "block": seg.get("content_block"),
                         "provider_error": seg.get("error_message"), "reason": reason},
               options=(_retry_option(run, unfinished or [idx]),
                        Option("manual", "I finish it by hand; re-check when the run changes")),
               about={f"run:{run.id}": run.version}, replaces=_pending_retry_asks(answers, name))


# ---------------------------------------------------------------------------
# 4. handoff
# ---------------------------------------------------------------------------

def plan_handoff(*, run: Optional[RunInfo], clean_output: Optional[Mapping[str, Any]],
                 answers: Mapping[str, Answer], final_path: Optional[str]):
    """The operator's acceptance (flow-doc step 10) is also the picture review
    the detector cannot do: invented words in another style, unrequested
    changes (a border removed, outpainting), the phone screen changed. The
    comparison sheet of the last QA is shown next to the final."""
    """The final for the operator's acceptance + the localisation hand-off."""
    if run is None:
        return Fail("the run is gone", consumes=unconsumed(answers))
    rejected = _rejection(answers)
    if rejected:
        return rejected
    if not run.terminal:
        return Wait(f"run {short(run.id)} is {run.status}")
    handoff = dict((clean_output or {}).get("handoff") or {})
    a = answers.get("handoff")
    auto = list((clean_output or {}).get("auto_accepted") or [])
    auto_idx = sorted(int(x["index"]) for x in auto)
    if _needs_ask(a):
        return Ask(
            name="handoff", kind="handoff", subject=f"run {short(run.id)}",
            question=("The caption detector calls the final clean. Watch it and the comparison "
                      "sheet: no invented words, no unrequested change (border, outpaint, "
                      "phone screen), lip sync.%s On approval create the localisation project "
                      "from this run (Run page -> To project, mode 'by Assistant' offered) "
                      "and mark the keep ranges after its analysis. Reject stops the "
                      "autopilot with your comment." % (
                          (" VOID rebuilt %s and the assistant accepted %s without an eye "
                           "check (no caption left, no face sampled under the repainted area; "
                           "there is no phone-screen detector): their fill sheets are attached "
                           "-- look at them too." % (
                               _segments_phrase(auto_idx),
                               "it" if len(auto_idx) == 1 else "them"))
                          if auto_idx else "")),
            proposal={"to_project": handoff.get("request"),
                      "keep_ranges": handoff.get("keep_ranges", []),
                      "then": handoff.get("then"), "suggested_mode": "assistant",
                      "warnings": handoff.get("warnings", [])},
            artifacts=tuple(
                [Artifact(final_path, "final", "video", run_id=run.id)] if final_path else [])
            + tuple([Artifact(clean_output["comparison_sheet"], "comparison sheet (result vs "
                              "source per swap segment)", "image",
                              effect_id=clean_output.get("qa_effect"))]
                    if (clean_output or {}).get("comparison_sheet") else [])
            + tuple(Artifact(x["fill_sheet"], "segment %d (accepted without an eye check): "
                             "delivered vs original inside the repainted area" % int(x["index"]),
                             "image", effect_id=clean_output.get("qa_effect"))
                    for x in auto if x.get("fill_sheet")),
            about={f"run:{run.id}": run.version})
    if a.status == "pending":
        return Wait("waiting for the operator: watch the final")
    return _complete(answers, output={"run_id": run.id, "keep_ranges": handoff.get(
        "keep_ranges", []), "to_project": handoff.get("request")},
        depends_on={f"run:{run.id}": run.version}, reason="the operator accepted the final")
