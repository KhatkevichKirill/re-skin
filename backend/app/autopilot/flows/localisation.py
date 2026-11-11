"""The localisation autopilot flow, generation part (AICP-365, "C1").

docs/autopilot-localisation-flow.md (steps, handlers, review) on the core of
docs/autopilot-design.md. Steps, per target language L in params order::

    intake -> transcribe -> segments -> breakdown -> translate:L -> prompts:L
           -> probe:L (segment 1 alone + listening approval) -> rest:L

The finishing steps (AICP-366, ``localisation_finish.py``) continue from
``rest:L``'s run: ``qa:L -> captions:L -> tail:L -> cleanup:L``. All decisions
are made from the database state the core hands to ``advance`` (read-only);
the pure computations live in :mod:`.localisation_plan`, the handlers and
compute functions in :mod:`.localisation_handlers`.

Approval / effect names per step (rounds are ``rN``, deterministic):

* intake: ``mute``, ``captions`` (compute), ``source_captions`` (handoff)
* transcribe / translate:L: ``launch`` then ``launch:rN`` after ``retry:rN``
  (regeneration with options) when the job failed
* segments: ``recut``
* probe:L: ``launch``; per round n: ``fail:rN`` (run failed), ``sheet:rN``
  (compute), ``listen:rN`` (listening_probe), ``cue:rN`` (records a rejection),
  ``fix:rN`` (regeneration with the corrected prompt), ``rerun:rN``
* rest:L: ``temp_clip`` (``temp_clip:rN`` if the clip vanished), ``launch``,
  ``fail:rN``, ``rerun:rN``
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy import select

from ... import ai_models
from ...localisation_jobs_models import LIVE_JOB_STATUSES, LocalisationPromptJob, ProjectTranscription
from ...models import LibraryClip, SegmentDef
from .. import registry
from ..contract import (
    Ask, Complete, Compute, Fail, FlowParamsError, Launch, Option, StepSpec, Wait,
)
from . import localisation_handlers  # noqa: F401 — registers handlers + compute
from . import localisation_plan as lp

log = logging.getLogger(__name__)

FLOW_VERSION = 2  # v2: AICP-390 (v1 stays registered: LocalisationFlow(1))
CREATE_RUN = "localisation_create_run"
_RUN_FIELDS = set(localisation_handlers.LocalisationCreateRun._FIELDS)


# ---------------------------------------------------------------------------
# Context helpers (read-only)
# ---------------------------------------------------------------------------


def _st(value) -> str:
    return value.value if hasattr(value, "value") else str(value or "")


def _segments(ctx) -> list[dict]:
    rows = ctx.db.execute(select(SegmentDef).where(
        SegmentDef.project_id == ctx.project.id).order_by(SegmentDef.index)).scalars().all()
    return lp.segment_dicts(rows)


def _max_clip_sec(params) -> float:
    return float(ai_models.spec_for(params.get("model") or lp.DEFAULT_MODEL).max_clip_sec)


def _keys(ctx) -> dict:
    """The project-level version keys every prompt-dependent decision binds."""
    return {
        f"project_segments:{ctx.project.id}": ctx.segments_version(),
        f"project_transcript:{ctx.project.id}": ctx.transcript_version(),
    }


def _answered(ctx, *names) -> list[str]:
    """The names among *names* whose current approval is answered, unconsumed."""
    out = []
    for n in names:
        a = ctx.approval(n)
        if a is not None and a.status in ("approved", "rejected"):
            out.append(n)
    return out


def _sorted_rounds(effects, prefix: str) -> list:
    """Effects named ``<prefix>:rN`` (or ``<prefix>:rN:<tag>``) ordered by N."""
    def n(e):
        try:
            return int(e.name[len(prefix) + 2:].split(":", 1)[0])
        except (IndexError, ValueError):
            return 0
    return sorted((e for e in effects if e.name.startswith(prefix + ":r")), key=n)


def _launches(ctx) -> list:
    first = ctx.effect("launch")
    return ([first] if first else []) + _sorted_rounds(ctx.effects("launch:"), "launch")


def _bought_before(ctx) -> bool:
    """Did an EARLIER generation of this step already admit paid work?

    A reopened step's next paid launch is then a replacement and must be a
    regeneration (approval + budget raise, Q6). ``previous_outputs()`` is not
    enough: the core keeps an output only for a step that had COMPLETED, while
    a step reopened mid-flight (a translation running when the segments are
    edited) may already have bought its work (Terra §1 gate R1)."""
    if ctx.step_generation <= 1:
        return False
    from ...models_autopilot import AutopilotEffect, ProjectAutopilot

    hit = ctx.db.execute(
        select(AutopilotEffect.id).join(
            ProjectAutopilot, ProjectAutopilot.id == AutopilotEffect.autopilot_id).where(
            ProjectAutopilot.project_id == ctx.project.id,
            ProjectAutopilot.status.in_(("active", "paused")),
            AutopilotEffect.step_id == ctx.step_id,
            AutopilotEffect.step_generation < ctx.step_generation,
            AutopilotEffect.kind == "action",
            AutopilotEffect.paid.is_(True),
            AutopilotEffect.state.in_(("launching", "launched", "unknown")),
        ).limit(1)
    ).first()
    return hit is not None or bool(ctx.previous_outputs())


def _open_answers(ctx) -> list[str]:
    """Every answered, unconsumed flow approval of this step generation.

    Listed in the consumes of the probe step's decisions: an answer given to a
    take that has since changed (a manual rerun) is then superseded by the
    core's revalidation instead of tripping its unconsumed-answer check, and
    the step re-asks about the current take (Terra §1 gate R2)."""
    from ...models_autopilot import AutopilotApproval, ProjectAutopilot

    rows = ctx.db.execute(
        select(AutopilotApproval.name).join(
            ProjectAutopilot, ProjectAutopilot.id == AutopilotApproval.autopilot_id).where(
            ProjectAutopilot.project_id == ctx.project.id,
            ProjectAutopilot.status.in_(("active", "paused")),
            AutopilotApproval.step_id == ctx.step_id,
            AutopilotApproval.step_generation == ctx.step_generation,
            AutopilotApproval.status.in_(("approved", "rejected")),
        )
    ).all()
    return sorted({n for (n,) in rows if "#" not in n})


def _clear_stale(ctx):
    """A consuming no-op for answers whose bound rows changed, or None.

    An answer about a take that has since changed (a manual rerun, restore) is
    superseded by the core when a decision consumes it; until then a step that
    waits on the CURRENT take would sit with a stale rejection (the core stops
    the autopilot after its grace) or could not complete. This launch lists
    them; the core supersedes them and discards it (Terra §1 gate R3)."""
    from ...models_autopilot import AutopilotApproval, ProjectAutopilot

    rows = ctx.db.execute(
        select(AutopilotApproval).join(
            ProjectAutopilot, ProjectAutopilot.id == AutopilotApproval.autopilot_id).where(
            ProjectAutopilot.project_id == ctx.project.id,
            ProjectAutopilot.status.in_(("active", "paused")),
            AutopilotApproval.step_id == ctx.step_id,
            AutopilotApproval.step_generation == ctx.step_generation,
            AutopilotApproval.status.in_(("approved", "rejected")),
        )
    ).scalars().all()
    stale = []
    for a in rows:
        if "#" in a.name:
            continue
        for key, v in (a.bound_versions or {}).items():
            try:
                changed = ctx.version(key) != v
            except Exception:  # a key of a row that is gone
                changed = True
            if changed:
                stale.append(a.name)
                break
    if not stale:
        return None
    # Every open answer is listed (the core requires a decision to account for
    # all of them); at least one is stale, so its revalidation supersedes it
    # and discards this launch before anything — a current answer included —
    # is consumed (Terra §1 gate R4).
    names = sorted({a.name for a in rows if "#" not in a.name})
    return Launch(ctx.next_round("ack"), "localisation_acknowledge", {"names": names},
                  "answers about a take that has changed since are void", consumes=names)


def _with_open(ctx, *names) -> list[str]:
    return sorted(set(names) | set(_open_answers(ctx)))


def _run_params(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k in _RUN_FIELDS and v is not None}


def _languages(ctx) -> list[str]:
    return list(ctx.params.get("languages") or [])


def _retry_options(handler: str, params: dict, label: str) -> list:
    return [Option("retry", label, handler=handler, params=params,
                   consequence="runs the paid job again; costs the estimate shown"),
            Option("stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent")]


def _job_retry(ctx, *, what: str, handler: str, params: dict, job_status: str, error: str,
               subject: str = ""):
    """The regeneration round after a failed/interrupted paid job (transcribe, translate).

    ``retry:rK`` is asked after the K-th launch; approve -> ``launch:r(K+1)``."""
    k = len(_launches(ctx))
    name = f"retry:r{k}"
    a = ctx.approval(name)
    if a is None:
        return Ask(
            name, "regeneration",
            f"The {what} {job_status}: run it again?",
            {"error": (error or "")[:1000], "status": job_status,
             "note": "an interrupted call may already have been billed"},
            subject=subject, title=f"The {what} {job_status}: run it again?",
            options=_retry_options(handler, params, f"Run the {what} again"),
        )
    if a.status == "pending":
        return Wait(f"waiting for the operator: retry the {what}?")
    if a.decision == "reject" or a.chosen_option == "stop":
        return Fail(f"the {what} {job_status} and the operator stopped: {a.answer_comment or ''}".strip(),
                    consumes=[name])
    return Launch(f"launch:r{k + 1}", handler, params, f"retry the {what} (operator approved)",
                  regeneration=True, consumes=[name])


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------


def _intake(ctx):
    p = ctx.project
    if _st(p.status) != "ready":
        return Wait("waiting for the analysis")
    rep = lp.intake(p)
    if rep.blockers:
        return Fail("; ".join(rep.blockers))
    if rep.needs_mute_source:
        mutes = ([ctx.effect("mute")] if ctx.effect("mute") else []) + _sorted_rounds(
            ctx.effects("mute:"), "mute")
        if mutes and mutes[-1].state == "launching":
            return Wait("bookkeeping of the mute_source change")
        name = "mute" if not mutes else ctx.next_round("mute")
        if len(mutes) >= 3:
            return Fail("mute_source keeps being turned off: localisation needs it on")
        return Launch(name, "set_mute_source", {},
                      "localisation re-speaks every clip: remove the original audio")
    scan = ctx.effect("captions")
    if scan is None:
        return Compute("captions", "localisation_source_captions",
                       {"source": p.source_local_path, "fps": 2.0},
                       "check the source for burned-in captions", timeout_sec=1800)
    if scan.state in ("queued", "running"):
        return Wait("scanning the source for burned-in captions")
    warnings = list(rep.warnings)
    consumes: list[str] = []
    result = dict(scan.result or {})
    if scan.state != "succeeded":
        warnings.append(f"caption scan of the source failed ({scan.error}); not checked")
    else:
        ratio = float(result.get("hit_ratio") or 0.0)
        threshold = float(result.get("threshold") or 0.2)
        if ratio >= threshold:
            a = ctx.approval("source_captions")
            if a is None:
                return Ask(
                    "source_captions", "handoff",
                    f"The source looks captioned ({ratio:.0%} of frames): is it clean?",
                    {"hit_ratio": ratio, "spans": result.get("spans") or [],
                     "hint": "a captioned source goes through the subtitle-removal flow first"},
                    title="Is the source free of burned captions?",
                    options=[Option("continue", "The source is clean: continue",
                                    consequence="the localisation goes on with this source"),
                             Option("stop", "Stop: remove the subtitles first")],
                )
            if a.status == "pending":
                return Wait("waiting for the operator: is the source clean?")
            if a.decision == "reject" or a.chosen_option == "stop":
                return Fail("the source carries burned-in captions: run subtitle removal first",
                            consumes=["source_captions"])
            consumes = ["source_captions"]
            warnings.append("the operator confirmed the source is clean despite the caption scan")
    return Complete(
        {"warnings": warnings, "duration_sec": p.duration_sec,
         "caption_hit_ratio": result.get("hit_ratio")},
        reason="; ".join(warnings) or "source ready", consumes=consumes,
    )


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


def _transcribe_estimate(ctx):
    st = lp.transcript_state(ctx.project.transcript_status, ctx.project.transcript)
    return 0.0 if st == "ready" else lp.estimate_transcription()


def _transcribe(ctx):
    p = ctx.project
    launches = _launches(ctx)
    latest = launches[-1] if launches else None
    if latest is not None and latest.state == "launching":
        return Wait("bookkeeping of the transcription launch")
    job = None
    if latest is not None:
        jid = (latest.created_refs or {}).get("transcription")
        job = ctx.db.get(ProjectTranscription, jid) if jid else None
        if job is not None and job.status in LIVE_JOB_STATUSES:
            return Wait(f"transcription {job.status}")
    st = lp.transcript_state(p.transcript_status, p.transcript)
    if st == "pending":
        return Wait("a transcription started outside the autopilot is running")
    open_names = _answered(ctx, *[f"retry:r{i}" for i in range(1, len(launches) + 1)])
    # A ready v3 transcript completes the step whoever produced it — also after
    # our own job failed and the operator transcribed by hand (gate R5).
    if st == "ready":
        problems = lp.check_languages_against_source(
            ctx.params, (p.transcript or {}).get("source_language"))
        if problems:
            return Fail("; ".join(problems), consumes=open_names)
        tr = p.transcript or {}
        return Complete(
            {"transcript_digest": lp.transcript_digest(tr),
             "source_language": tr.get("source_language") or "",
             "schema_version": tr.get("schema_version"), "lines": len(tr.get("lines") or [])},
            depends_on={f"project_transcript:{p.id}": ctx.transcript_version()},
            reason=f"transcript v{tr.get('schema_version')} ready, {len(tr.get('lines') or [])} lines",
            consumes=open_names,
        )
    if st == "empty":
        return Fail("the transcript has no speech: nothing to localise", consumes=open_names)
    if job is not None and job.status in ("failed", "interrupted"):
        return _job_retry(ctx, what="transcription", handler="transcribe_project", params={},
                          job_status=job.status, error=job.error or p.transcript_error or "")
    if latest is None:
        why = {"stale": "the transcript predates transcribe/v3 (no speakers/genders/actions)",
               "failed": "the last transcription failed", "missing": "no transcript yet"}.get(st, st)
        return Launch("launch", "transcribe_project", {}, f"transcribe the source: {why}",
                      regeneration=_bought_before(ctx))
    # Our job succeeded but the transcript is not a ready v3 one (edited by hand
    # meanwhile): the operator's change wins, and asks to transcribe again.
    return _job_retry(ctx, what="transcription", handler="transcribe_project", params={},
                      job_status=f"left the transcript {st}", error=p.transcript_error or "")


# ---------------------------------------------------------------------------
# segments + breakdown
# ---------------------------------------------------------------------------


def _segments_step_v2(ctx):
    return _segments_step(ctx, end_card=True)


def _segments_step(ctx, end_card: bool = False):
    p = ctx.project
    recut = ctx.effect("recut")
    first_generation = not ctx.previous_outputs()
    if end_card:
        # v2 waits on the operator inside this step, so it can be reopened
        # before it ever completed (no previous output): only generation 1
        # re-cuts, never over the operator's edits (Terra §1 gate R1).
        first_generation = ctx.step_generation <= 1
    if recut is None and first_generation:
        return Launch("recut", "recut_at_pauses", {},
                      "move segment boundaries off spoken lines (AICP-347)")
    if recut is not None and recut.state == "launching":
        return Wait("bookkeeping of the re-cut")
    segs = _segments(ctx)
    target = lp.whole_video_target_sec(p.max_segment_sec)
    chk = lp.check_partition(segs, duration_sec=float(p.duration_sec or 0), target_sec=target)
    if not chk.ok:
        return Fail("the segments are not a usable whole-video cut: " + "; ".join(chk.problems)
                    + " — fix them in the Segment Editor and start a new autopilot",
                    consumes=_open_answers(ctx) if end_card else ())
    res = dict((recut.result if recut else None) or {})
    out = {
        "segments": [[s["id"], round(s["start_sec"], 3), round(s["end_sec"], 3), s["action"]]
                     for s in segs],
        "recut": {k: res.get(k) for k in ("applied", "feasible", "changed", "warnings",
                                          "boundaries")} if res else None,
        "target_sec": target,
    }
    if not first_generation:
        out["note"] = "segments edited by the operator: kept as they are (no re-cut)"
    moved = len(res.get("boundaries") or [])
    reason = f"{len(segs)} segments, {moved} boundary(ies) moved off spoken lines"
    if not end_card:  # flow v1: unchanged
        return Complete(
            out, depends_on={f"project_segments:{p.id}": ctx.segments_version()}, reason=reason)
    decided = _end_card(ctx, segs)
    if not isinstance(decided, dict):
        return decided  # a Decision: ask, wait, cut
    out["end_card"] = decided["end_card"]
    ec = decided["end_card"] or {}
    if ec.get("choice") == "replace":
        reason += f"; the source's end card from {ec['start_sec']:.2f} s is cut"
    elif ec.get("choice") == "generate":
        reason += "; the source's end card is localised like the rest (operator)"
    return Complete(out, depends_on=_keys(ctx), reason=reason,
                    consumes=decided.get("consumes") or [])


# ---------------------------------------------------------------------------
# The source's own CTA end card (AICP-390, flow v2)
# ---------------------------------------------------------------------------


def _scene_cuts(ctx):
    from ...config import settings
    from ...scene import read_scene_cuts
    from ...storage import scene_cuts_cache_path

    p = ctx.project
    return read_scene_cuts(scene_cuts_cache_path(p.id), source_path=p.source_local_path,
                           min_scene_len_sec=settings.SCENE_MIN_LEN_SECONDS)


def _pending_names(ctx, prefix: str) -> list[str]:
    from ...models_autopilot import AutopilotApproval, ProjectAutopilot

    rows = ctx.db.execute(
        select(AutopilotApproval.name).join(
            ProjectAutopilot, ProjectAutopilot.id == AutopilotApproval.autopilot_id).where(
            ProjectAutopilot.project_id == ctx.project.id,
            ProjectAutopilot.status.in_(("active", "paused")),
            AutopilotApproval.step_id == ctx.step_id,
            AutopilotApproval.step_generation == ctx.step_generation,
            AutopilotApproval.status == "pending",
        )
    ).all()
    return sorted({n for (n,) in rows if n.startswith(prefix)})


def _tail_names(ctx) -> dict:
    out = {}
    for lang, cid in (ctx.params.get("tails") or {}).items():
        clip = ctx.db.get(LibraryClip, cid) if cid else None
        out[lang] = (clip.name if clip is not None else cid) if cid else None
    return out


def _end_card(ctx, segs):
    """The end-card decision of this generation: ``{"end_card": ..., "consumes": [...]}``
    or a Decision (ask / wait / the cut).

    ``end_card`` is None (no candidate), ``{"choice": "replace", "start_sec",
    "segment_ids", ...}`` (cut: the languages with a tail leave these segments
    out of their run, the others keep them as source footage) or ``{"choice":
    "generate", ...}`` (the operator said it is not an end card). A reject or
    the ``stop`` option fails the attempt, as every localisation hand-off does
    (the approval card words its "no" control that way).

    The question is tagged with the segments version, so an edit made while it
    is open (the operator moving the boundary in the Segment Editor) retires it
    and asks again about the new state. A cut is never assumed from the
    segments alone: an existing trailing keep is asked about like any other
    candidate (Terra §2: it may be source footage the operator meant to keep).
    """
    p = ctx.project
    tr = p.transcript or {}
    tver = ctx.transcript_version()

    # 1. this generation's cut, while the segments still show it
    cuts = ctx.effects("cut:")
    if cuts and cuts[-1].state == "launching":
        return Wait("bookkeeping of the end-card cut")
    launched = [e for e in cuts if e.state == "launched"]
    if launched:
        e = launched[-1]
        start = (e.result or {}).get("start_sec", (e.params or {}).get("start_sec"))
        ids = lp.end_card_segment_ids(segs, start) if start is not None else None
        if ids:
            return {"end_card": {"choice": "replace", "start_sec": float(start), "segment_ids": ids,
                                 "transcript_version": tver, "effect": e.name}}
    # 2. a reopened step: an earlier generation's cut, if the segments and the
    # transcript still show it exactly (anything else is decided again)
    for prev in reversed(ctx.previous_outputs()):
        ec = (prev.get("output") or {}).get("end_card") or {}
        if ec.get("choice") == "replace" and ec.get("transcript_version") == tver:
            ids = lp.end_card_segment_ids(segs, ec.get("start_sec"))
            if ids:
                return {"end_card": dict(ec, segment_ids=ids,
                                         note=f"kept from generation {prev.get('generation')}")}
        break

    det = lp.detect_end_card(tr, scene_cuts=_scene_cuts(ctx), duration_sec=p.duration_sec)
    if det is None:
        return {"end_card": None}
    plan = lp.end_card_plan(segs, det, lines=tr.get("lines") or [],
                            duration_sec=float(p.duration_sec or 0),
                            target_sec=lp.whole_video_target_sec(p.max_segment_sec))
    tag = ctx.segments_version()[:12]
    name = f"endcard:{tag}"
    obsolete = [n for n in _pending_names(ctx, "endcard:") if n != name]
    clear = _clear_stale(ctx)  # an answer about segments that have changed since
    if clear is not None:
        return clear
    summary = {k: det[k] for k in ("start_sec", "end_sec", "first_line_sec", "prev_line_end_sec",
                                   "scene_cut", "speakers", "on_screen_text", "notes")}
    a = ctx.approval(name)
    if a is None:
        tails = _tail_names(ctx)
        per_language = {lang: (f"cut; {tails[lang]} is appended in its place" if tails.get(lang)
                               else "no CTA tail: the source's end card stays as original footage")
                        for lang in _languages(ctx)}
        cut_params = {"start_sec": det["start_sec"], "patch": plan["patch"], "result": plan["result"]}
        options = []
        if plan["feasible"]:
            options.append(Option(
                "replace", f"Cut it from {det['start_sec']:.2f} s", handler="localisation_cut_end_card",
                params=cut_params,
                consequence="it is never generated; each language with a CTA tail ends with that tail instead"))
        already_keep = any(seg["action"] == "keep" and seg["end_sec"] > det["start_sec"] + 1e-3
                           for seg in segs)
        options.append(Option(
            "generate", "It is not an end card: leave the segments as they are",
            consequence=("its keep segments stay source footage; swap segments are generated (paid)"
                         if already_keep else "it is generated like the rest of the video (paid)")))
        options.append(Option(
            "stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent"))
        why = ("" if plan["feasible"] else
               " It cannot be cut at this boundary: " + "; ".join(plan["problems"])
               + ". Move the boundary in the Segment Editor to be asked again.")
        return Ask(
            name, "handoff",
            f"The source seems to end with its own CTA end card from {det['start_sec']:.2f} s "
            f"(\"{det['lines'][0]['text'][:80]}\"): cut it?" + why,
            {**summary, "lines": det["lines"], "per_language": per_language,
             "segments_after": plan["result"], "problems": plan["problems"],
             "hint": "the end card is never sent to the model; a wrong boundary can be fixed "
                     "in the Segment Editor while this question is open"},
            title="Cut the source's own end card?", options=options, about=_keys(ctx),
            retire=obsolete,
        )
    if a.status == "pending":
        return Wait("waiting for the operator: is this the source's own end card?", retire=obsolete)
    consumes = _with_open(ctx, name)
    if a.decision == "reject" or a.chosen_option == "stop":
        return Fail(f"stopped by the operator at the end-card question: {a.answer_comment or ''}".strip(),
                    consumes=consumes)
    if a.decision == "approve" and a.chosen_option == "replace" and plan["feasible"]:
        if not plan["changed"]:  # the segments already are the cut
            ids = lp.end_card_segment_ids(segs, det["start_sec"]) or []
            return {"end_card": {"choice": "replace", "start_sec": det["start_sec"],
                                 "segment_ids": ids, "transcript_version": tver, **summary},
                    "consumes": consumes}
        return Launch(f"cut:{tag}", "localisation_cut_end_card",
                      {"start_sec": det["start_sec"], "patch": plan["patch"], "result": plan["result"]},
                      f"cut the source's end card from {det['start_sec']:.2f} s (operator approved)",
                      expect=_keys(ctx), consumes=consumes)
    return {"end_card": {"choice": "generate", **summary,
                         "comment": (a.answer_comment or "")[:500]}, "consumes": consumes}


def _compact(breakdown: dict) -> dict:
    """Drop per-line rows when the breakdown would not fit a step output."""
    if len(json.dumps(breakdown, default=str).encode()) < 56 * 1024:
        return breakdown
    out = dict(breakdown)
    out["segments"] = [
        {k: v for k, v in s.items() if k != "lines"} | {"line_ids": [r["id"] for r in s.get("lines", [])]}
        for s in breakdown.get("segments") or []
    ]
    out["compacted"] = True
    return out


def _breakdown(ctx):
    p = ctx.project
    tr = p.transcript or {}
    segs = _segments(ctx)
    b = lp.source_breakdown(tr, segs, duration_sec=p.duration_sec,
                            max_clip_sec=_max_clip_sec(ctx.params))
    probe = lp.pick_probe_segment(segs, tr.get("lines") or [])
    if probe is None:
        return Fail(f"no segment has {lp.PROBE_MIN_SPEECH_SEC:.0f} s of speech to probe the voices with")
    speakers = ", ".join(f"{s['tag']} ({s['gender']}, {s['kind']})" for s in b["speakers"])
    reason = f"speakers: {speakers}; probe = segment {probe['index'] + 1}"
    if b["warnings"]:
        reason += "; " + "; ".join(b["warnings"][:5])
    return Complete({"breakdown": _compact(b), "probe_segment_id": probe["id"],
                     "probe_index": probe["index"]},
                    depends_on=_keys(ctx), reason=reason[:2000])


# ---------------------------------------------------------------------------
# translate:L + prompts:L
# ---------------------------------------------------------------------------


def _translate_estimate(ctx):
    return 3.0 * lp.spend.llm_credits()


def _translate(lang):
    def advance(ctx):
        p = ctx.project
        params = {"language": lang, "model": ctx.params.get("model") or lp.DEFAULT_MODEL}
        launches = _launches(ctx)
        latest = launches[-1] if launches else None
        if latest is None:
            return Launch("launch", "localisation_prompt", params,
                          f"translate the whole video into {lang}",
                          regeneration=_bought_before(ctx), expect=_keys(ctx))
        if latest.state == "launching":
            return Wait("bookkeeping of the translate launch")
        jid = (latest.created_refs or {}).get("translate_job")
        job = ctx.db.get(LocalisationPromptJob, jid) if jid else None
        if job is None:
            return Fail(f"the {lang} translate job record is gone")
        if job.status in LIVE_JOB_STATUSES:
            return Wait(f"{lang} translation {job.status}")
        open_names = _answered(ctx, *[f"retry:r{i}" for i in range(1, len(launches) + 1)])
        if job.status == "succeeded":
            from ... import localisation_prompt_job as lpj
            from ...api_v2 import _project_swap_segments

            current = lpj.project_fingerprint(p, _project_swap_segments(p.id, ctx.db))
            if current == job.fingerprint:
                warnings = list((job.result or {}).get("warnings") or [])
                return Complete(
                    {"job_id": job.id, "warnings": warnings,
                     "lines": len((job.result or {}).get("lines") or [])},
                    depends_on=_keys(ctx), consumes=open_names,
                    reason=f"{lang} translated" + (f"; {len(warnings)} warning(s)" if warnings else ""),
                )
            status, error = "is stale (the project changed after it was translated)", ""
        else:
            status, error = job.status, job.error or ""
        return _job_retry(ctx, what=f"{lang} translation", handler="localisation_prompt",
                          params=params, job_status=status, error=error, subject=lang)
    return advance


def _prompts_for(ctx, lang) -> tuple[dict, dict]:
    """(prompts, translate job snapshot) — rebuilt from the job's FROZEN inputs."""
    tout = ctx.output(f"translate:{lang}") or {}
    bout = ctx.output("breakdown") or {}
    job = ctx.db.get(LocalisationPromptJob, tout.get("job_id"))
    if job is None or not job.result:
        raise RuntimeError(f"translate job of {lang} missing")
    snap = dict(job.snapshot or {})
    prompts = lp.build_language_prompts(
        translated_lines=job.result.get("lines") or [],
        transcript=snap.get("transcript") or {},
        segments=_segments(ctx),
        probe_segment_id=bout["probe_segment_id"],
        source_language=snap.get("source_language") or "",
        target_language=lang,
        duration_sec=snap.get("duration_sec"),
        max_clip_sec=(snap.get("prompt_model") or {}).get("max_clip_sec"),
    )
    return prompts, snap


def _prompts(lang):
    def advance(ctx):
        prompts, _snap = _prompts_for(ctx, lang)
        warnings = list(prompts["warnings"]) + list(
            (ctx.output(f"translate:{lang}") or {}).get("warnings") or [])
        return Complete(
            {"sha256": lp.json_digest(prompts), "probe_segment_id": prompts["probe_segment_id"],
             "segments": len(prompts["order"]), "warnings": warnings},
            depends_on=_keys(ctx),
            reason=f"{len(prompts['order'])} {lang} segment prompts built"
            + (f"; {len(warnings)} warning(s)" if warnings else ""),
        )
    return advance


def _checked_prompts(ctx, lang):
    prompts, snap = _prompts_for(ctx, lang)
    want = (ctx.output(f"prompts:{lang}") or {}).get("sha256")
    if want != lp.json_digest(prompts):
        return None, snap
    return prompts, snap


# ---------------------------------------------------------------------------
# probe:L
# ---------------------------------------------------------------------------


def _probe_estimate(lang):
    def estimate(ctx):
        bout = ctx.output("breakdown") or {}
        est = lp.estimate_language(_segments(ctx), probe_segment_id=bout.get("probe_segment_id"),
                                   model=ctx.params.get("model") or lp.DEFAULT_MODEL,
                                   resolution=ctx.params.get("resolution") or lp.DEFAULT_RESOLUTION)
        return est["probe"]
    return estimate


def _rest_estimate(lang):
    def estimate(ctx):
        bout = ctx.output("breakdown") or {}
        est = lp.estimate_language(_segments(ctx), probe_segment_id=bout.get("probe_segment_id"),
                                   model=ctx.params.get("model") or lp.DEFAULT_MODEL,
                                   resolution=ctx.params.get("resolution") or lp.DEFAULT_RESOLUTION)
        return est["rest"]
    return estimate


def _current_probe_prompt(ctx, base: str) -> str:
    cues = [e for e in ctx.effects("cue:") if e.state == "launched"]  # created_at order
    if cues:
        return (cues[-1].result or {}).get("prompt") or (cues[-1].params or {}).get("prompt") or base
    return base


def _rerun_params(run_id: str, run_segment_ids: list, prompt: Optional[str]) -> dict:
    return {"run_id": run_id,
            "segments": [{"run_segment_id": sid, "prompt": prompt} for sid in run_segment_ids]}


def _failed_run_round(ctx, run, n: int, seg_ids: list, prompt: Optional[str], lang: str, what: str):
    # Tagged with the run version: a failure of a later (manual) take is a new
    # question, never an old pending one (Terra §1 gate R4).
    name = f"fail:r{n}:{ctx.run_version(run.id)[:16]}"
    params = _rerun_params(run.id, seg_ids, prompt)
    a = ctx.approval(name)
    if a is None:
        return Ask(
            name, "regeneration",
            f"The {lang} {what} run ended {run.status}: regenerate its unfinished segment(s)?",
            {"run_id": run.id, "status": run.status, "segments": seg_ids},
            subject=lang, title=f"The {lang} {what} run ended {run.status}",
            options=[Option("rerun", "Regenerate them", handler="rerun_segments", params=params,
                            consequence="resubmits the unfinished segments; costs the estimate shown"),
                     Option("stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent")],
            about={f"run:{run.id}": ctx.run_version(run.id)},
        )
    if a.status == "pending":
        return Wait(f"waiting for the operator: the {lang} {what} run {run.status}")
    if a.decision == "reject" or a.chosen_option == "stop":
        return Fail(f"the {lang} {what} run {run.status}; the operator stopped",
                    consumes=_with_open(ctx, name))
    return Launch(f"rerun:r{n}", "rerun_segments", params,
                  f"regenerate the {lang} {what} segment(s) (operator approved)",
                  regeneration=True, consumes=_with_open(ctx, name))


def _probe(lang, first_lang):
    def advance(ctx):
        p = ctx.project
        if lang != first_lang and ctx.output(f"probe:{first_lang}") is None:
            return Wait(f"waiting until the {first_lang} probe is accepted (it validates the "
                        "prompt format before more video is paid for)")
        prompts, snap = _checked_prompts(ctx, lang)
        if prompts is None:
            return Fail(f"the {lang} prompts no longer match the ones built in prompts:{lang}")
        seg_id = prompts["probe_segment_id"]
        launch = ctx.effect("launch")
        if launch is None:
            spec = lp.probe_run_spec(project_id=p.id, language=lang, prompts=prompts,
                                     model=ctx.params.get("model") or lp.DEFAULT_MODEL,
                                     resolution=ctx.params.get("resolution") or lp.DEFAULT_RESOLUTION)
            return Launch("launch", CREATE_RUN, _run_params(spec),
                          f"probe: generate segment {seg_id[:8]} alone in {lang}, no voice reference",
                          regeneration=_bought_before(ctx), expect=_keys(ctx))
        if launch.state == "launching":
            return Wait("bookkeeping of the probe launch")
        rid = (launch.created_refs or {}).get("runs", [None])[0]
        run = ctx.run(rid) if rid else None
        if run is None:
            return Fail(f"the {lang} probe run is gone")
        clear = _clear_stale(ctx)  # before any Wait (Terra §1 gate R4)
        if clear is not None:
            return clear
        reruns = _sorted_rounds(ctx.effects("rerun:"), "rerun")
        if reruns and reruns[-1].state == "launching":
            return Wait("bookkeeping of the probe regeneration")
        n = len(reruns) + 1
        if not run.is_terminal:
            return Wait(f"{lang} probe run {run.short_id} is {run.status}")
        prompt = _current_probe_prompt(ctx, prompts["probe_prompt"])
        seg_rows = [s.id for s in run.segments]
        if run.status != "done":
            return _failed_run_round(ctx, run, n, seg_rows, prompt, lang, "probe")
        # --- listening round n ---------------------------------------------
        # The listening round is keyed by the probe run's version too: after a
        # manual rerun (which pauses the autopilot, Q5) the new take gets its
        # own sheet and its own listening approval, and an answer about the old
        # take is never applied to it (Terra §1 gate R1, R2).
        tag = f"r{n}:{ctx.run_version(run.id)[:16]}"
        sheet_name = f"sheet:{tag}"
        sheet = ctx.effect(sheet_name)
        if sheet is None:
            seg = next(s for s in _segments(ctx) if s["id"] == seg_id)
            w0, w1 = lp._clip_window(seg, p.duration_sec, _max_clip_sec(ctx.params))
            return Compute(sheet_name, "localisation_probe_sheet",
                           {"source": p.source_local_path, "result": run.result_local_path,
                            "source_start": w0, "source_end": w1},
                           f"frames of the {lang} probe against the source")
        if sheet.state in ("queued", "running"):
            return Wait(f"building the {lang} probe frame sheet")
        listen_name = f"listen:{tag}"
        a = ctx.approval(listen_name)
        if a is None:
            sres = dict(sheet.result or {})
            arts = [ctx.artifact(run.result_local_path, label=f"{lang} probe", kind="video",
                                 run_id=run.id)]
            if sheet.state == "succeeded" and sres.get("sheet"):
                arts.append(ctx.artifact(sres["sheet"], label="probe vs source (1 fps)",
                                         kind="image", effect_id=sheet.id))
            return Ask(
                listen_name, "listening_probe",
                f"Listen to the {lang} probe: voices, gender, line order, pronunciation",
                {"run_id": run.id, "run": run.short_id, "language": lang, "round": n,
                 "prompt": prompt, "length_delta_sec": sres.get("length_delta_sec"),
                 "max_frame_diff": sres.get("max_diff"),
                 "sheet": "failed: " + (sheet.error or "") if sheet.state != "succeeded" else "ok",
                 "on_reject": "write what is wrong: it is added to the prompt of the regeneration"},
                subject=lang, artifacts=arts, title=f"Listen to the {lang} probe",
                about={f"run:{run.id}": ctx.run_version(run.id)},
            )
        if a.status == "pending":
            return Wait(f"waiting for the operator to listen to the {lang} probe")
        if a.decision == "approve":
            return Complete(
                {"probe_run_id": run.id, "probe_segment_id": seg_id, "round": n, "prompt": prompt},
                depends_on={f"run:{run.id}": ctx.run_version(run.id)},
                consumes=_with_open(ctx, listen_name),
                reason=f"{lang} probe accepted (round {n})",
            )
        # rejected: record the cue (consumes the answer), then ask the regeneration
        cue_name = f"cue:{tag}"
        cue = ctx.effect(cue_name)
        if cue is None:
            comment = (a.answer_comment or "").strip()
            corrected = lp.apply_operator_cue(prompt, comment) if comment else prompt
            return Launch(cue_name, "localisation_probe_feedback",
                          {"comment": comment, "prompt": corrected, "round": n},
                          "record the operator's correction", consumes=_with_open(ctx, listen_name))
        if cue.state == "launching":
            return Wait("bookkeeping of the operator's correction")
        corrected = (cue.result or {}).get("prompt") or (cue.params or {}).get("prompt")
        comment = (cue.params or {}).get("comment") or ""
        fix_name = f"fix:{tag}"
        params = _rerun_params(run.id, seg_rows, corrected)
        f = ctx.approval(fix_name)
        if f is None:
            return Ask(
                fix_name, "regeneration",
                f"Regenerate the {lang} probe with your correction?",
                {"run_id": run.id, "comment": comment, "prompt": corrected, "round": n},
                subject=lang, title=f"Regenerate the {lang} probe with your correction",
                options=[Option("rerun", "Regenerate the probe segment", handler="rerun_segments",
                                params=params, consequence="resubmits those segments with the correction; costs the estimate shown"),
                         Option("stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent")],
                about={f"run:{run.id}": ctx.run_version(run.id)},
            )
        if f.status == "pending":
            return Wait(f"waiting for the operator: regenerate the {lang} probe?")
        if f.decision == "reject" or f.chosen_option == "stop":
            return Fail(f"{lang} probe rejected: {comment or 'no comment'}",
                        consumes=_with_open(ctx, fix_name))
        return Launch(f"rerun:r{n}", "rerun_segments", params,
                      f"regenerate the {lang} probe with the operator's correction",
                      regeneration=True, consumes=_with_open(ctx, fix_name))
    return advance


# ---------------------------------------------------------------------------
# rest:L
# ---------------------------------------------------------------------------


def _end_card_excluded(ctx, lang) -> list[str]:
    """The source end card's segments when it was cut AND *lang* has a CTA
    tail (AICP-390): its final ends before the card and ``tail:L`` appends the
    CTA in its place. Without a tail they stay in as keep (original footage).
    Flow v1 outputs have no ``end_card``: nothing is excluded."""
    ec = (ctx.output("segments") or {}).get("end_card") or {}
    if ec.get("choice") != "replace" or not (ctx.params.get("tails") or {}).get(lang):
        return []
    return list(ec.get("segment_ids") or [])


def _rest(lang):
    def advance(ctx):
        p = ctx.project
        pout = ctx.output(f"probe:{lang}") or {}
        prid = pout.get("probe_run_id")
        prompts, _snap = _checked_prompts(ctx, lang)
        if prompts is None:
            return Fail(f"the {lang} prompts no longer match the ones built in prompts:{lang}")
        launch = ctx.effect("launch")
        clips = ([ctx.effect("temp_clip")] if ctx.effect("temp_clip") else []) + _sorted_rounds(
            ctx.effects("temp_clip:"), "temp_clip")
        if launch is None:
            clip = clips[-1] if clips else None
            if clip is not None and clip.state == "launching":
                return Wait("bookkeeping of the TEMP clip")
            clip_id = (clip.created_refs or {}).get("library_clip") if clip else None
            if clip_id is None or ctx.db.get(LibraryClip, clip_id) is None:
                if len(clips) >= 3:
                    return Fail("the TEMP library clip of the probe keeps disappearing")
                name = "temp_clip" if not clips else ctx.next_round("temp_clip")
                spec = lp.temp_clip_spec(project_id=p.id, language=lang, probe_run_id=prid)
                return Launch(name, "create_temp_library_clip",
                              {"run_id": prid, "name": spec["name"]},
                              f"reuse the accepted {lang} probe through a TEMP library clip")
            spec = lp.rest_run_spec(
                project_id=p.id, language=lang, prompts=prompts, segments=_segments(ctx),
                probe_run_id=prid, temp_clip_id=clip_id,
                model=ctx.params.get("model") or lp.DEFAULT_MODEL,
                resolution=ctx.params.get("resolution") or lp.DEFAULT_RESOLUTION,
                exclude_segment_ids=_end_card_excluded(ctx, lang))
            return Launch("launch", CREATE_RUN, _run_params(spec),
                          f"generate the remaining {lang} segments with the probe's voice",
                          regeneration=_bought_before(ctx), expect=_keys(ctx))
        if launch.state == "launching":
            return Wait("bookkeeping of the run launch")
        rid = (launch.created_refs or {}).get("runs", [None])[0]
        run = ctx.run(rid) if rid else None
        if run is None:
            return Fail(f"the {lang} run is gone")
        clear = _clear_stale(ctx)  # before any Wait (Terra §1 gate R4)
        if clear is not None:
            return clear
        reruns = _sorted_rounds(ctx.effects("rerun:"), "rerun")
        if reruns and reruns[-1].state == "launching":
            return Wait("bookkeeping of the regeneration")
        if not run.is_terminal:
            return Wait(f"{lang} run {run.short_id} is {run.status}")
        if run.status != "done":
            unfinished = [s.id for s in run.segments if s.status != "completed"]
            return _failed_run_round(ctx, run, len(reruns) + 1, unfinished or [s.id for s in run.segments],
                                     None, lang, "remaining-segments")
        clip_id = None
        for c in reversed(clips):
            clip_id = (c.created_refs or {}).get("library_clip") or clip_id
            if clip_id:
                break
        # No depends_on on this run: the finishing flow (AICP-366) regenerates
        # its segments on purpose, which must not reopen this step.
        return Complete({"run_id": run.id, "temp_clip_id": clip_id, "probe_run_id": prid,
                         "language": lang},
                        reason=f"{lang} stitched: run {run.short_id}",
                        consumes=_open_answers(ctx))
    return advance


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def _clip_lookup(clip_id: str):
    from ...db import SessionLocal

    db = SessionLocal()
    try:
        clip = db.get(LibraryClip, clip_id)
        return None if clip is None else {"id": clip.id, "language": clip.language}
    finally:
        db.close()


class LocalisationFlow:
    """Flow v1 (AICP-365/366) and v2 (AICP-390: the source's own end card is
    proposed at ``segments`` and, when cut, left out of every language that has
    a CTA tail). v1 stays registered for the attempts created under it
    (docs/autopilot-design.md §4.7); new attempts take the newest version."""

    id = "localisation"
    label = "Localisation"
    project_types = frozenset({"localisation"})

    def __init__(self, version: int = FLOW_VERSION):
        self.version = int(version)
        self.end_card = self.version >= 2
    handlers = {
        CREATE_RUN: 1, "rerun_segments": 1, "transcribe_project": 1, "localisation_prompt": 1,
        "recut_at_pauses": 1, "set_mute_source": 1, "create_temp_library_clip": 1,
        "localisation_probe_feedback": 1, "localisation_acknowledge": 1,
        # AICP-366 (finishing, localisation_finish.py)
        "restore_takes": 1, "localisation_create_caption": 1, "localisation_caption_lines": 1,
        "localisation_script_caption": 1, "localisation_create_tail": 1,
        "localisation_delete_temp_clip": 1, "localisation_listen_feedback": 1,
        "localisation_rederive_placement": 1,
        # AICP-389 (brand trailing-sibilant fix, localisation_brandfix.py)
        "localisation_apply_brand_audio_patch": 1,
        # AICP-390 (flow v2: the source's own end card)
        "localisation_cut_end_card": 1,
    }

    def validate_params(self, project, params):
        scope = getattr(project, "localisation_scope", None)
        if scope is not None and scope != "whole":
            raise FlowParamsError(
                "the localisation autopilot re-speaks the whole video: create the project "
                "with localisation_scope=whole")
        try:
            return lp.validate_params(params, project_type=getattr(project, "project_type", None),
                                      clip_lookup=_clip_lookup)
        except lp.FlowParamsError as exc:
            raise FlowParamsError(str(exc)) from None

    def steps(self, project, params):
        langs = list((params or {}).get("languages") or [])
        steps = [
            StepSpec("intake", "Intake: source, mute_source, duration", advance=_intake,
                     approvals=("handoff",)),
            StepSpec("transcribe", "Transcript (transcribe/v3)", advance=_transcribe,
                     needs=("intake",), paid=True, approvals=("regeneration",),
                     estimate=_transcribe_estimate),
            StepSpec("segments", "Segments re-cut at speech pauses", advance=_segments_step,
                     needs=("transcribe",)) if not self.end_card else
            StepSpec("segments", "Segments re-cut at speech pauses; the source's own end card",
                     advance=_segments_step_v2, needs=("transcribe",), approvals=("handoff",)),
            StepSpec("breakdown", "Source breakdown per segment", advance=_breakdown,
                     needs=("segments",)),
        ]
        first = langs[0] if langs else None
        for lang in langs:
            steps.append(StepSpec(f"translate:{lang}", f"Translate into {lang}",
                                  advance=_translate(lang), needs=("breakdown",), paid=True,
                                  approvals=("regeneration",), estimate=_translate_estimate))
        for lang in langs:
            steps.append(StepSpec(f"prompts:{lang}", f"Segment prompts ({lang})",
                                  advance=_prompts(lang), needs=(f"translate:{lang}",)))
        for lang in langs:
            steps.append(StepSpec(f"probe:{lang}", f"Probe: segment 1 alone ({lang})",
                                  advance=_probe(lang, first), needs=(f"prompts:{lang}",),
                                  paid=True, approvals=("listening_probe", "regeneration"),
                                  estimate=_probe_estimate(lang)))
        for lang in langs:
            steps.append(StepSpec(f"rest:{lang}", f"Remaining segments ({lang})",
                                  advance=_rest(lang), needs=(f"probe:{lang}",), paid=True,
                                  approvals=("regeneration",), estimate=_rest_estimate(lang)))
        # AICP-366: QA, regeneration proposals, final listening, captions, CTA
        # tail and cleanup, continuing from each rest:L run.
        from . import localisation_finish

        steps += localisation_finish.steps(langs)
        return steps

    def budget_lines(self, ctx):
        return []


FLOW_V1 = LocalisationFlow(1)
FLOW = LocalisationFlow(FLOW_VERSION)
registry.register(FLOW_V1)
registry.register(FLOW)
