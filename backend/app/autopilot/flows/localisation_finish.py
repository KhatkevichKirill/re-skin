"""The localisation autopilot flow, finishing part (AICP-366, "C2").

docs/autopilot-localisation-finishing.md (rev 3: §1 steps, §5/§5b the
resolutions of the §2 review). Continues C1 (``localisation.py``) from the
stitched final of ``rest:L``. Steps per target language L::

    qa:L        segment QA + ASR QA -> regeneration proposals (approval each)
                -> best take -> listening_final (reject = next regeneration)
    captions:L  fix wrong words / captions from the script (free burn-only)
    tail:L      CTA tail chained off the caption, deliverable <= 99 MB
    cleanup:L   delete the TEMP probe clip, journal the language's spend

Names inside ``qa:L`` (a round is keyed by the rest run's version, like C1's
probe listening round): ``segqa:<tag>`` (compute), ``best:<rerun name>``
(restore_takes), ``asr:<tag>`` (the caption job = ASR), ``regen:<tag>``
(regeneration with options), ``rerun:rK``, ``listen:<tag>``
(listening_final), ``cue:<tag>`` (records a rejection), ``fix:<tag>``
(regeneration), ``fail:rK:<ver>`` (C1's failed-run round), ``stuck:<ver>``
(handoff: a run that failed without an unfinished segment).

Every decision is read-only planning over the database (and file stats, as
the core's own ``run_version``); media work is the compute
``localisation_segment_qa`` (localisation_qa.py), paid or state-changing work
the handlers of localisation_finish_handlers.py and the core's
``rerun_segments`` / ``restore_takes``.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import func, select

from ... import localisation as loc
from ... import spend as spend_mod
from ...models_autopilot import ProjectAutopilot
from ...models import (
    KieSpendReservation, LibraryClip, Run, RunCaption, RunSegment, RunSegmentTake, RunTailEdit,
)
from ..contract import Ask, Complete, Compute, Fail, Launch, Option, StepSpec, Wait
from . import localisation as c1
from . import localisation_brandfix as bf
from . import localisation_finish_handlers as fh
from . import localisation_plan as lp
from . import localisation_qa as lqa

log = logging.getLogger(__name__)

MAX_QA_ROUNDS = 6
_ACTIVE_CAPTION = ("queued", "transcribing", "burning", "delivering")
_ACTIVE_TAIL = ("queued", "processing", "delivering")

HANDLERS = {
    "rerun_segments": 1, "restore_takes": 1, "localisation_acknowledge": 1,
    "localisation_create_caption": 1, "localisation_caption_lines": 1,
    "localisation_script_caption": 1, "localisation_create_tail": 1,
    "localisation_delete_temp_clip": 1, "localisation_listen_feedback": 1,
    "localisation_rederive_placement": 1, "localisation_apply_brand_audio_patch": 1,
}


def _st(value) -> str:
    return value.value if hasattr(value, "value") else str(value or "")


# ---------------------------------------------------------------------------
# Reading the stitched final (read-only)
# ---------------------------------------------------------------------------


def _probe_window(ctx, seg: dict) -> list:
    w0, w1 = loc.clip_window(seg["start_sec"], seg["end_sec"], seg.get("pre_roll_sec") or 0.0,
                             seg.get("post_roll_sec") or 0.0, ctx.project.duration_sec,
                             c1._max_clip_sec(ctx.params))
    return [round(w0, 3), round(w1, 3)]


def _pieces(ctx, lang: str, run_id: str) -> list[dict]:
    """The final's pieces (localisation_qa.build_pieces) from the run's own rows."""
    db = ctx.db
    run = db.get(Run, run_id)
    snapshot = list(run.segment_snapshot or []) or c1._segments(ctx)
    rows = db.execute(select(RunSegment).where(RunSegment.run_id == run_id)).scalars().all()
    rs = {r.segment_def_id: {"id": r.id, "prompt_override": r.prompt_override,
                             "local_result_path": r.local_result_path,
                             "seedance_task_id": r.seedance_task_id, "status": _st(r.status)}
          for r in rows}
    rest = ctx.output(f"rest:{lang}") or {}
    probe = ctx.output(f"probe:{lang}") or {}
    numbers = {str(s["id"]): i + 1 for i, s in enumerate(sorted(snapshot, key=lambda s: s["index"]))}
    libs = {}
    for e in run.stitch_plan or []:
        if not isinstance(e, dict) or e.get("kind") != "library":
            continue
        cid = str(e.get("clip_id"))
        clip = db.get(LibraryClip, cid)
        entry = {"path": clip.local_path if clip is not None else None}
        if cid == rest.get("temp_clip_id") and probe.get("probe_run_id"):
            seg_id = probe.get("probe_segment_id")
            prs = db.execute(select(RunSegment).where(
                RunSegment.run_id == probe["probe_run_id"],
                RunSegment.segment_def_id == seg_id)).scalars().first()
            prun = db.get(Run, probe["probe_run_id"])
            entry["prompt"] = (prs.prompt_override if prs is not None else None) or (
                prun.prompt if prun is not None else "")
            seg = next((s for s in snapshot if str(s["id"]) == str(seg_id)), None)
            if seg is not None:
                entry.update(window=_probe_window(ctx, seg), seg_number=numbers.get(str(seg_id)),
                             segment_def_id=seg_id)
        libs[cid] = entry
    return lqa.build_pieces(
        stitch_plan=run.stitch_plan, snapshot=snapshot, run_segments=rs, run_prompt=run.prompt,
        library_clips=libs, duration_sec=ctx.project.duration_sec,
        max_clip_sec=c1._max_clip_sec(ctx.params),
        genders=lqa.speaker_genders(ctx.project.transcript))


def _qa_args(ctx, run, pieces: list[dict]) -> dict:
    return {"source": ctx.project.source_local_path, "final": run.result_local_path,
            "language": run_language(ctx, run), "caption_config": ctx.project.caption_detector,
            "pieces": pieces}


def run_language(ctx, run) -> Optional[str]:
    r = ctx.db.get(Run, run.id)
    return r.language if r is not None else None


def _files_current(segres: dict) -> bool:
    """The compute's inputs are still the files it read (design §5b.3)."""
    if not segres.get("stable", False):
        return False
    return all(lqa.file_sig(path) == sig for path, sig in (segres.get("files") or {}).items())


def _raw_asr(cap: Optional[RunCaption], segres: Optional[dict]) -> tuple[bool, str]:
    """(is this caption raw ASR of exactly this final?, why not)."""
    if cap is None:
        return False, "no ASR caption"
    if _st(cap.status) != "done":
        return False, f"the ASR caption {_st(cap.status)}: {(cap.error_message or '')[:200]}"
    if cap.lines_source != "transcribe":
        return False, f"the caption's lines are {cap.lines_source or 'unknown'}, not a transcription"
    want = (segres or {}).get("final_audio_sha256")
    if not want or cap.audio_sha256 != want:
        return False, "the caption did not transcribe this final's audio"
    return True, ""


def _evaluate(ctx, lang: str, pieces: list[dict], segres: Optional[dict],
              cap: Optional[RunCaption]) -> dict:
    """Issues per piece (segment QA + raw-ASR QA), warnings, and the ASR summary."""
    issues: dict[int, list] = {int(p["pos"]): [] for p in pieces}
    warnings: list[dict] = []
    if segres is None:
        warnings.append({"kind": "segment_qa_missing", "detail": "segment QA failed or read changed files"})
    for row in (segres or {}).get("pieces") or []:
        pos = int(row["pos"])
        issues.setdefault(pos, []).extend(row.get("issues") or [])
        warnings += [dict(w, pos=pos, segment=row.get("seg_number")) for w in row.get("warnings") or []]
    raw, why = _raw_asr(cap, segres)
    asr = {"raw": raw, "why": why, "coverage": None, "garbled": False}
    if raw:
        secs = lqa.piece_secs(segres, pieces)
        chk = lqa.asr_check(lqa.final_script(pieces, secs), cap.lines or [],
                            brands=(ctx.project.transcript or {}).get("brands") or [], language=lang)
        asr.update(coverage=chk["coverage"], garbled=chk["garbled"])
        for i in chk["issues"]:
            issues.setdefault(int(i["pos"]), []).append(i)
        warnings += chk["warnings"]
        if chk["garbled"]:
            warnings.append({"kind": "asr_garbled", "coverage": chk["coverage"],
                             "detail": "captions will come from the script; listen carefully"})
    else:
        warnings.append({"kind": "no_asr_evidence", "detail": why})
    regen, other = [], []
    for p in pieces:
        found = issues.get(int(p["pos"])) or []
        if not found:
            continue
        if p["kind"] == "swap" and p.get("run_segment_id"):
            regen.append(int(p["pos"]))
        else:
            other.append(int(p["pos"]))
            warnings += [dict(i, pos=p["pos"], segment=p.get("seg_number"),
                              note="the accepted probe segment is not regenerated here") for i in found]
    return {"issues": issues, "warnings": warnings, "asr": asr, "regenerable": regen}


def _by_pos(pieces):
    return {int(p["pos"]): p for p in pieces}


def _rerun_params(run_id: str, segs: list[tuple[str, Optional[str]]], *, discard: bool = False) -> dict:
    out = {"run_id": run_id, "segments": [{"run_segment_id": s, "prompt": pr} for s, pr in segs]}
    if discard:
        # AICP-389 §7: set only on an operator choice whose card said the fix
        # is dropped (the regen / fix asks below), never automatically.
        out["discard_audio_patch"] = True
    return out


def _qa_prompt(piece: dict, found: list) -> Optional[str]:
    cues = lqa.qa_cues(found)
    if not cues:
        return None  # a plain rerun (length): the saved prompt is kept
    return lqa.apply_cues(piece.get("prompt") or "", cues)


def _compact_issue(i: dict) -> dict:
    keep = ("kind", "tag", "expected", "measured", "hz", "baseline_hz", "shift_octaves", "text",
            "name", "delta_sec", "hits", "coverage")
    return {k: i[k] for k in keep if k in i and i[k] is not None}


def _sheets(ctx, seg, segres, positions=None) -> list:
    out = []
    if seg is None or segres is None:
        return out
    for row in segres.get("pieces") or []:
        if row.get("sheet") and (positions is None or int(row["pos"]) in positions):
            out.append(ctx.artifact(row["sheet"], label=f"segment {row.get('seg_number')} vs source",
                                    kind="image", effect_id=seg.id))
    return out


def _has_lineage(ctx) -> bool:
    """Is there a subtitle-removal lineage to re-derive the placement from? (read-only)"""
    from ... import caption_placement as cp

    try:
        return bool(cp.removal_chain(ctx.db, ctx.project))
    except Exception:
        return False


def _last_round_end(ctx):
    ends = [e for e in ctx.effects() if e.name.startswith(("rerun:", "best:"))]
    return max(ends, key=lambda e: e.created_at) if ends else None


def _asr_budgeted(ctx) -> bool:
    """The first ASR caption of generation 1 is in the budget; any other is a regeneration."""
    if ctx.step_generation > 1 or c1._bought_before(ctx):
        return False
    return not [e for e in ctx.effects("asr:") if e.state in ("launching", "launched", "unknown")]


def _round_scores(ctx, lang, seg_eff, asr_eff) -> dict[str, dict]:
    """``run_segment_id -> {clip, seedance_task_id, score, asr_clean}`` of one QA round."""
    if seg_eff is None or seg_eff.state != "succeeded":
        return {}
    segres = dict(seg_eff.result or {})
    pieces = list(((seg_eff.params or {}).get("args") or {}).get("pieces") or [])
    cap = None
    if asr_eff is not None and asr_eff.state == "launched":
        cap = ctx.db.get(RunCaption, (asr_eff.created_refs or {}).get("caption"))
    raw, _why = _raw_asr(cap, segres)
    asr_issues: dict[int, int] = {}
    if raw:
        chk = lqa.asr_check(lqa.final_script(pieces, lqa.piece_secs(segres, pieces)), cap.lines or [],
                            brands=(ctx.project.transcript or {}).get("brands") or [], language=lang)
        if chk["garbled"]:
            raw = False  # garbled ASR is no evidence that a take said its lines (gate R1)
        else:
            for i in chk["issues"]:
                asr_issues[int(i["pos"])] = asr_issues.get(int(i["pos"]), 0) + 1
    out = {}
    for row in segres.get("pieces") or []:
        if row.get("kind") != "swap" or not row.get("run_segment_id"):
            continue
        pos = int(row["pos"])
        out[row["run_segment_id"]] = {
            "clip": row.get("clip"), "seedance_task_id": row.get("seedance_task_id"),
            "score": len(row.get("issues") or []) if row.get("complete") else None,
            "asr_clean": bool(raw) and asr_issues.get(pos, 0) == 0,
            "scoring_id": segres.get("scoring_id"),
        }
    return out


def _best_take_restores(ctx, lang, run, rerun_eff, seg_eff) -> list[str]:
    """Take ids to restore after *rerun_eff* (design §5.10 / §5b.9), or []."""
    segres = dict(seg_eff.result or {})
    sid = segres.get("scoring_id")
    current = _round_scores(ctx, lang, seg_eff, None)
    segments = [s["run_segment_id"] for s in (rerun_eff.params or {}).get("segments") or []]
    history = []
    for e in ctx.effects("segqa:"):
        if e.id == seg_eff.id or e.state != "succeeded" or e.created_at > rerun_eff.created_at:
            continue
        tag = e.name[len("segqa:"):]
        rnd = _round_scores(ctx, lang, e, ctx.effect(f"asr:{tag}"))
        history.append({k: (v if v["scoring_id"] == sid and v["asr_clean"] else dict(v, score=None))
                        for k, v in rnd.items()})
    picks = lqa.best_takes(history, current, segments)
    takes = []
    for rsid, best in picks.items():
        rows = ctx.db.execute(select(RunSegmentTake).where(
            RunSegmentTake.run_id == run.id, RunSegmentTake.run_segment_id == rsid)
            .order_by(RunSegmentTake.created_at.desc())).scalars().all()
        hit = next((t for t in rows if best.get("seedance_task_id")
                    and t.seedance_task_id == best["seedance_task_id"]), None) or next(
            (t for t in rows if t.result_path == best.get("clip")), None)
        if hit is not None and hit.file_purged_at is None:
            takes.append(hit.id)
    return sorted(takes)


# ---------------------------------------------------------------------------
# qa:L
# ---------------------------------------------------------------------------


def _qa(lang):
    def advance(ctx):
        rest = ctx.output(f"rest:{lang}") or {}
        rid = rest.get("run_id")
        run = ctx.run(rid) if rid else None
        if run is None:
            return Fail(f"the {lang} run is gone", consumes=c1._open_answers(ctx))
        clear = c1._clear_stale(ctx)  # before any Wait (C1 gate R4)
        if clear is not None:
            return clear
        busy = [e for e in ctx.effects() if e.kind == "action" and e.state == "launching"]
        if busy:
            return Wait(f"bookkeeping of {busy[-1].name}")
        if not run.is_terminal:
            return Wait(f"{lang} run {run.short_id} is {run.status}")
        reruns = c1._sorted_rounds(ctx.effects("rerun:"), "rerun")
        n = 1 + len(reruns) + len(ctx.effects("best:"))
        if run.status != "done":
            return _failed(ctx, run, lang, len(reruns) + 1)
        ver = ctx.run_version(run.id)
        tag = f"r{n}:{ver[:16]}"
        about = {f"run:{run.id}": ver}
        pieces = _pieces(ctx, lang, run.id)
        by_pos = _by_pos(pieces)

        # 1. segment QA (free compute, run workers)
        seg = ctx.effect(f"segqa:{tag}")
        if seg is None:
            return Compute(f"segqa:{tag}", "localisation_segment_qa", _qa_args(ctx, run, pieces),
                           f"segment QA of the {lang} final (round {n})", timeout_sec=3600)
        if seg.state in ("queued", "running"):
            return Wait(f"checking the {lang} segments against the source (round {n})")
        segres = dict(seg.result or {}) if seg.state == "succeeded" else None
        if segres is not None and not _files_current(segres):
            segres = None  # it read files that changed since: no evidence

        # 2. best take: the first round after our latest rerun, before its ASR
        last = _last_round_end(ctx)
        asr = ctx.effect(f"asr:{tag}")
        if (segres is not None and asr is None and last is not None
                and last.name.startswith("rerun:") and last.state == "launched"
                and ctx.effect(f"best:{last.name}") is None
                # AICP-389 §7: a restore would drop an audio fix of the current
                # result; it is never acknowledged automatically.
                and bf.current_patch(ctx.db, run.id) is None):
            takes = _best_take_restores(ctx, lang, run, last, seg)
            if takes:
                return Launch(f"best:{last.name}", "restore_takes", {"run_id": run.id, "takes": takes},
                              f"an earlier take of {len(takes)} segment(s) measured better than the "
                              "regenerated one: restore it", consumes=c1._open_answers(ctx))

        # 2b. AICP-389: the free brand check (trailing "-s") and its fix ask,
        # before the paid ASR caption, so a round that fixes the word does not
        # pay for a caption of the broken audio.
        bdec, bnote = bf.brand_round(ctx, lang, run, tag, n, pieces, segres,
                                     lambda *names: c1._with_open(ctx, *names),
                                     caption_paid=(asr is not None or not _asr_budgeted(ctx)))
        if bdec is not None:
            return bdec

        # 3. ASR = the caption job of the final (paid; the first one is budgeted)
        placement = fh.caption_placement_state(ctx.project, ctx.now)
        if asr is None:
            rederives = _round_launches(ctx, ("placement",))
            if rederives and rederives[-1].state == "launching":
                return Wait("bookkeeping of the caption placement re-derivation")
            if (placement in ("failed", "stale") and len(rederives) < 2
                    and _has_lineage(ctx)):
                # A caption freezes the placement at creation: re-derive a failed
                # or lost scan first (at most twice), then caption with what is
                # there (also when the lineage is gone: 368's frozen fallback).
                return Launch("placement" if not rederives else "placement:r2",
                              "localisation_rederive_placement",
                              {"expect_placement": fh.placement_sig(ctx.project)},
                              f"the caption placement scan is {placement}: derive it again (AICP-368)",
                              consumes=c1._open_answers(ctx))
            if placement == "pending":
                return Wait("waiting for the project's caption placement scan (AICP-368)")
            return Launch(f"asr:{tag}", "localisation_create_caption",
                          {"run_id": run.id, "language": lang},
                          f"ASR of the generated {lang} audio + captions (round {n})",
                          regeneration=not _asr_budgeted(ctx), consumes=c1._open_answers(ctx))
        cap = None
        if asr.state == "launched":
            cap = ctx.db.get(RunCaption, (asr.created_refs or {}).get("caption"))
            if cap is not None and _st(cap.status) in _ACTIVE_CAPTION:
                return Wait(f"ASR caption of the {lang} final is {_st(cap.status)}")

        ev = _evaluate(ctx, lang, pieces, segres, cap)
        summary = {
            "run": run.short_id, "language": lang, "round": n, "asr": ev["asr"],
            "placement": placement,
            "issues": {str(by_pos[p].get("seg_number")): [_compact_issue(i) for i in ev["issues"][p]]
                       for p in ev["regenerable"]},
            "warnings": [_compact_issue(w) | {"segment": w.get("segment")} for w in ev["warnings"]][:40],
        }
        if bnote:
            summary["brand_check"] = bnote
        drops = bf.rerun_warning(ctx.db, run.id)  # AICP-389 §7

        # 4. regeneration proposal for the segments with issues
        if ev["regenerable"] and n > MAX_QA_ROUNDS:
            summary["note"] = f"{MAX_QA_ROUNDS} QA rounds reached: no more automatic proposals"
        elif ev["regenerable"]:
            name = f"regen:{tag}"
            segs = [(by_pos[p]["run_segment_id"], _qa_prompt(by_pos[p], ev["issues"][p]))
                    for p in ev["regenerable"]]
            params = _rerun_params(run.id, segs, discard=bool(drops))
            a = ctx.approval(name)
            if a is None:
                numbers = ", ".join(str(by_pos[p].get("seg_number")) for p in ev["regenerable"])
                proposal = dict(summary, cues={
                    str(by_pos[p].get("seg_number")): lqa.qa_cues(ev["issues"][p])
                    for p in ev["regenerable"]})
                arts = [ctx.artifact(run.result_local_path, label=f"{lang} final", kind="video",
                                     run_id=run.id)] + _sheets(ctx, seg, segres, set(ev["regenerable"]))
                if drops:
                    proposal["drops_audio_fix"] = drops
                return Ask(name, "regeneration",
                           f"QA of the {lang} final found issues in segment(s) {numbers}: regenerate them?"
                           + (f" (regenerating {drops})" if drops else ""),
                           proposal, subject=lang, artifacts=arts, about=about,
                           title=f"QA found issues in {lang} segment(s) {numbers}",
                           options=[Option("rerun", f"Regenerate segment(s) {numbers}",
                                           handler="rerun_segments", params=params,
                                           consequence=("resubmits them with the QA cues; costs the estimate shown"
                                                        if not drops else
                                                        f"resubmits them; costs the estimate shown; {drops}")),
                                    Option("accept", "Keep them: go on to the listening check",
                                           consequence="nothing is regenerated; you listen to the final next"),
                                    Option("stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent")])
            if a.status == "pending":
                return Wait(f"waiting for the operator: regenerate {lang} segments?")
            if a.decision == "reject" or a.chosen_option == "stop":
                return Fail(f"the operator stopped at the {lang} QA proposal: {a.answer_comment or ''}".strip(),
                            consumes=c1._with_open(ctx, name))
            if a.chosen_option == "rerun":
                return Launch(f"rerun:r{len(reruns) + 1}", "rerun_segments", params,
                              f"regenerate {lang} segment(s) with the QA cues (operator approved)",
                              regeneration=True, consumes=c1._with_open(ctx, name))
            summary["accepted_issues"] = True  # "accept": consumed by the next decision

        # 5. the final listening approval
        listen = f"listen:{tag}"
        a = ctx.approval(listen)
        if a is None:
            arts = [ctx.artifact(run.result_local_path, label=f"{lang} final", kind="video",
                                 run_id=run.id)] + _sheets(ctx, seg, segres)
            return Ask(listen, "listening_final",
                       f"Listen to the {lang} final: voices, gender, line order, pronunciation, accent",
                       dict(summary, on_reject="name the segment(s) and what is wrong: it becomes the "
                                               "prompt cue of the next regeneration"),
                       subject=lang, artifacts=arts, about=about, title=f"Listen to the {lang} final")
        if a.status == "pending":
            return Wait(f"waiting for the operator to listen to the {lang} final")
        if a.decision == "approve":
            secs = lqa.piece_secs(segres, pieces)
            return Complete(
                {"run_id": run.id, "tag": tag, "round": n, "caption_id": cap.id if cap else None,
                 "asr_raw": ev["asr"]["raw"], "garbled": ev["asr"]["garbled"],
                 "coverage": ev["asr"]["coverage"], "secs": {str(k): v for k, v in secs.items()},
                 "placement": placement, "warnings": len(ev["warnings"])},
                depends_on=about, consumes=c1._with_open(ctx, listen),
                reason=f"{lang} final accepted (round {n})")
        return _rejected(ctx, lang, run, tag, n, a, pieces, ev, about, reruns)
    return advance


def _rejected(ctx, lang, run, tag, n, a, pieces, ev, about, reruns):
    """A rejected final: record the cue (consumes the answers), then ask the fix."""
    cue_name = f"cue:{tag}"
    cue = ctx.effect(cue_name)
    if cue is None:
        comment = " ".join(str(a.answer_comment or "").split())
        named = lqa.segments_named(comment)
        swaps = [p for p in pieces if p["kind"] == "swap" and p.get("run_segment_id")]
        note = ""
        if named:
            targets = [p for p in swaps if p.get("seg_number") in named]
            if any(p["kind"] == "library" and p.get("seg_number") in named for p in pieces):
                note = "the accepted probe segment is reused from the probe and not regenerated here"
        else:
            targets = [p for p in swaps if int(p["pos"]) in ev["regenerable"]] or swaps
        if not targets:
            return Fail(f"{lang} final rejected ({comment or 'no comment'}); no segment of this run "
                        "can be regenerated for it" + (f": {note}" if note else ""),
                        consumes=c1._with_open(ctx, a.name))
        brands = (ctx.project.transcript or {}).get("brands") or []
        segs = []
        for p in targets:
            prompt = p.get("prompt") or ""
            if comment:
                prompt = lp.apply_operator_cue(prompt, comment)
            prompt = lqa.apply_cues(prompt, lqa.comment_cues(comment, p, brands)
                                    + lqa.qa_cues(ev["issues"].get(int(p["pos"])) or []))
            segs.append({"run_segment_id": p["run_segment_id"], "prompt": prompt})
        return Launch(cue_name, "localisation_listen_feedback",
                      {"comment": comment, "round": n, "segments": segs, "named": named, "note": note},
                      "record the operator's correction", consumes=c1._with_open(ctx, a.name))
    if cue.state == "launching":
        return Wait("bookkeeping of the operator's correction")
    res = dict(cue.result or cue.params or {})
    drops = bf.rerun_warning(ctx.db, run.id)  # AICP-389 §7
    params = _rerun_params(run.id, [(s["run_segment_id"], s["prompt"]) for s in res.get("segments") or []],
                           discard=bool(drops))
    fix = f"fix:{tag}"
    f = ctx.approval(fix)
    if f is None:
        nums = [p.get("seg_number") for p in pieces if p.get("run_segment_id") in
                {s["run_segment_id"] for s in res.get("segments") or []}]
        return Ask(fix, "regeneration",
                   f"Regenerate {lang} segment(s) {', '.join(map(str, nums))} with your correction?"
                   + (f" (regenerating {drops})" if drops else ""),
                   {"run": run.short_id, "comment": res.get("comment"), "segments": nums,
                    "note": res.get("note") or None, "round": n,
                    **({"drops_audio_fix": drops} if drops else {})},
                   subject=lang, about=about,
                   title=f"Regenerate {lang} segment(s) {', '.join(map(str, nums))} with your correction",
                   options=[Option("rerun", "Regenerate them", handler="rerun_segments", params=params,
                                   consequence=("resubmits those segments with the correction; costs the estimate shown"
                                                if not drops else
                                                f"resubmits them; costs the estimate shown; {drops}")),
                            Option("stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent")])
    if f.status == "pending":
        return Wait(f"waiting for the operator: regenerate the {lang} segments?")
    if f.decision == "reject" or f.chosen_option == "stop":
        return Fail(f"{lang} final rejected: {res.get('comment') or 'no comment'}",
                    consumes=c1._with_open(ctx, fix))
    return Launch(f"rerun:r{len(reruns) + 1}", "rerun_segments", params,
                  f"regenerate {lang} segment(s) with the operator's correction",
                  regeneration=True, consumes=c1._with_open(ctx, fix))


def _failed(ctx, run, lang, k):
    """The run ended failed/incomplete (after a rerun or a restore)."""
    unfinished = [s.id for s in run.segments if s.status != "completed"]
    if unfinished:
        return c1._failed_run_round(ctx, run, k, unfinished, None, lang, "finishing")
    ver = ctx.run_version(run.id)
    name = f"stuck:{ver[:16]}"
    a = ctx.approval(name)
    if a is None:
        return Ask(name, "handoff",
                   f"The {lang} run {run.short_id} ended {run.status} with every segment finished "
                   "(stitch or delivery): retry it from the run page, then continue",
                   {"run": run.short_id, "status": run.status}, subject=lang,
                   about={f"run:{run.id}": ver}, title=f"The {lang} run {run.short_id} could not be finished",
                   options=[Option("continue", "I retried it",
                                   consequence="the assistant checks the run again and goes on"), Option("stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent")])
    if a.status == "pending":
        return Wait(f"waiting for the operator: the {lang} run is {run.status}")
    if a.decision == "reject" or a.chosen_option == "stop":
        return Fail(f"the {lang} run {run.status}; the operator stopped", consumes=c1._with_open(ctx, name))
    return Wait(f"waiting for the {lang} run to be retried (it is {run.status})", recheck_sec=60)


# ---------------------------------------------------------------------------
# captions:L
# ---------------------------------------------------------------------------


def _round_launches(ctx, bases):
    return sorted((e for e in ctx.effects() if e.name.split(":", 1)[0] in bases),
                  key=lambda e: e.created_at)


def _captions(lang):
    def advance(ctx):
        qa = ctx.output(f"qa:{lang}") or {}
        rid = qa.get("run_id")
        run = ctx.run(rid) if rid else None
        if run is None:
            return Fail(f"the {lang} run is gone", consumes=c1._open_answers(ctx))
        pieces = _pieces(ctx, lang, rid)
        secs = {int(k): float(v) for k, v in (qa.get("secs") or {}).items()}
        launches = _round_launches(ctx, ("fix", "script"))
        if launches:
            last = launches[-1]
            if last.state == "launching":
                return Wait(f"bookkeeping of {last.name}")
            cap = ctx.db.get(RunCaption, (last.created_refs or {}).get("caption"))
            if cap is None:
                return Fail(f"the {lang} caption job is gone", consumes=c1._open_answers(ctx))
            st = _st(cap.status)
            if st in _ACTIVE_CAPTION:
                return Wait(f"{lang} captions are {st}")
            if st == "done":
                res = dict(last.result or {})
                return Complete({"caption_id": cap.id, "mode": res.get("mode"),
                                 "changes": res.get("changes"), "lines": len(cap.lines or [])},
                                consumes=c1._open_answers(ctx),
                                reason=f"{lang} captions burned ({res.get('mode')}, "
                                       f"{res.get('changes') or 0} word(s) fixed)")
            if len(launches) >= 2:
                return Fail(f"the {lang} caption burn failed twice: {(cap.error_message or '')[:300]}",
                            consumes=c1._open_answers(ctx))
            # Retry the burn of EXACTLY the lines decided before (gate R1): a
            # replan would see a failed row and fall back to the script.
            base = last.name.split(":", 1)[0]
            lp_ = dict(last.params or {})
            return Launch(f"{base}:r2", "localisation_caption_lines",
                          {"caption_id": cap.id, "lines": lp_.get("lines") or cap.lines or [],
                           "expect_caption": fh.caption_sig(cap),
                           "mode": (last.result or {}).get("mode") or lp_.get("mode") or "script",
                           "changes": int(lp_.get("changes") or 0)},
                          f"retry the {lang} caption burn (free)", consumes=c1._open_answers(ctx))
        cap = ctx.db.get(RunCaption, qa.get("caption_id")) if qa.get("caption_id") else None
        return _caption_launch(ctx, lang, rid, cap, pieces, secs, qa, None)
    return advance


def _caption_launch(ctx, lang, rid, cap, pieces, secs, qa, name):
    st = _st(cap.status) if cap is not None else None
    if st in _ACTIVE_CAPTION:
        return Wait(f"the {lang} caption job is {st}")
    usable = cap is not None and st == "done" and bool(cap.lines) and bool(cap.output_path)
    if usable and qa.get("asr_raw") and not qa.get("garbled"):
        fixed, changes = lqa.fix_caption_words(cap.lines, lqa.final_script(pieces, secs), language=lang)
        if not changes:
            return Complete({"caption_id": cap.id, "mode": "asr", "changes": 0,
                             "lines": len(cap.lines or [])},
                            consumes=c1._open_answers(ctx), reason=f"{lang} ASR captions are right as heard")
        return Launch(name or "fix", "localisation_caption_lines",
                      {"caption_id": cap.id, "lines": fixed, "expect_caption": fh.caption_sig(cap),
                       "mode": "asr_fix", "changes": changes},
                      f"fix {changes} misheard word(s) in the {lang} captions (burn only, free)",
                      consumes=c1._open_answers(ctx))
    if usable and cap.lines_source in ("reused", "manual") and not qa.get("garbled"):
        return Complete({"caption_id": cap.id, "mode": cap.lines_source, "changes": 0,
                         "lines": len(cap.lines or [])}, consumes=c1._open_answers(ctx),
                        reason=f"{lang} captions kept as they are ({cap.lines_source} lines)")
    lines = lqa.script_captions(pieces, secs)
    if not lines:
        if usable:
            return Complete({"caption_id": cap.id, "mode": "unchecked", "changes": 0},
                            consumes=c1._open_answers(ctx), reason=f"{lang}: no script lines to caption from")
        return Fail(f"no {lang} captions: no usable ASR and no script lines", consumes=c1._open_answers(ctx))
    if cap is not None and st in ("done", "failed"):
        if usable and [(l["start"], l["end"], l["text"]) for l in cap.lines] == [
                (l["start"], l["end"], l["text"]) for l in lines]:
            return Complete({"caption_id": cap.id, "mode": "script", "changes": 0},
                            consumes=c1._open_answers(ctx), reason=f"{lang} captions are the script")
        return Launch(name or "fix", "localisation_caption_lines",
                      {"caption_id": cap.id, "lines": lines, "expect_caption": fh.caption_sig(cap),
                       "mode": "script", "changes": len(lines)},
                      f"{lang} captions from the script ({'garbled ASR' if qa.get('garbled') else 'no ASR'})",
                      consumes=c1._open_answers(ctx))
    return Launch(name or "script", "localisation_script_caption",
                  {"run_id": rid, "language": lang, "lines": lines},
                  f"{lang} captions from the script (no caption job to fix)",
                  consumes=c1._open_answers(ctx))


# ---------------------------------------------------------------------------
# tail:L
# ---------------------------------------------------------------------------


def _tail(lang):
    def advance(ctx):
        qa = ctx.output(f"qa:{lang}") or {}
        cout = ctx.output(f"captions:{lang}") or {}
        rid = qa.get("run_id")
        cap = ctx.db.get(RunCaption, cout.get("caption_id")) if cout.get("caption_id") else None
        if cap is None:
            return Fail(f"the {lang} caption job is gone", consumes=c1._open_answers(ctx))
        st = _st(cap.status)
        if st in _ACTIVE_CAPTION:
            return Wait(f"the {lang} caption job is {st} (retried by hand?)")
        if st != "done" or not cap.output_path:
            return Fail(f"the {lang} caption job is {st}", consumes=c1._open_answers(ctx))
        clip_id = (ctx.params.get("tails") or {}).get(lang)
        edit, last = None, None
        if clip_id:
            launches = _round_launches(ctx, ("tail",))
            last = launches[-1] if launches else None
            if last is not None and last.state == "launching":
                return Wait("bookkeeping of the tail edit")
            if last is not None:
                edit = ctx.db.get(RunTailEdit, (last.created_refs or {}).get("tail_edit"))
                est = _st(edit.status) if edit is not None else "gone"
                if est in _ACTIVE_TAIL:
                    return Wait(f"the {lang} CTA tail is {est}")
            if last is None or edit is None or _st(edit.status) == "failed":
                if len(launches) >= 2:
                    why = (edit.error_message or "") if edit is not None else "gone"
                    return Fail(f"the {lang} CTA tail failed twice: {why[:300]}",
                                consumes=c1._open_answers(ctx))
                if ctx.db.get(LibraryClip, clip_id) is None:
                    return Fail(f"the {lang} CTA clip {clip_id[:8]} was deleted", consumes=c1._open_answers(ctx))
                return Launch("tail" if last is None else "tail:r2", "localisation_create_tail",
                              {"run_id": rid, "caption_id": cap.id, "library_clip_id": clip_id,
                               "expect_caption": fh.caption_sig(cap)},
                              f"append the {lang} CTA tail to the captioned video (free)",
                              consumes=c1._open_answers(ctx))
        target = edit.output_path if edit is not None else cap.output_path
        name = f"size:{(edit.id if edit is not None else cap.id)[:8]}"
        chk = ctx.effect(name)
        if chk is None:
            return Compute(name, "localisation_deliverable_check",
                           {"path": target, "max_bytes": lqa.DELIVERABLE_MAX_BYTES,
                            "also": cap.output_path if edit is not None else None},
                           f"size and length of the {lang} deliverable")
        if chk.state in ("queued", "running"):
            return Wait(f"checking the {lang} deliverable")
        if chk.state != "succeeded":
            return Fail(f"could not check the {lang} deliverable: {chk.error}", consumes=c1._open_answers(ctx))
        res = dict(chk.result or {})
        out = {"caption_id": cap.id, "tail_edit_id": edit.id if edit is not None else None,
               "cut_sec": (last.result or {}).get("cut_sec") if edit is not None else None,
               "size_bytes": res.get("size_bytes"), "duration_sec": res.get("duration_sec"),
               "sha256": res.get("sha256"), "caption_sha256": res.get("also_sha256")}
        if res.get("ok"):
            return Complete(out, consumes=c1._open_answers(ctx),
                            reason=f"{lang} deliverable {res.get('size_bytes', 0) / 1e6:.1f} MB"
                                   + (" with the CTA tail" if edit is not None else " (no CTA tail)"))
        a = ctx.approval("size")
        if a is None:
            return Ask("size", "handoff",
                       f"The {lang} deliverable is {res.get('size_bytes', 0) / 1e6:.1f} MB (over 99 MB)",
                       {"size_bytes": res.get("size_bytes"), "max_bytes": res.get("max_bytes"),
                        "path": target}, subject=lang, title=f"The {lang} deliverable is over 99 MB",
                       options=[Option("deliver", "Deliver it as is",
                                       consequence="uploads the file over 99 MB as it is"), Option("stop", "Stop the autopilot", consequence="ends this attempt (failed); nothing more is spent")])
        if a.status == "pending":
            return Wait(f"waiting for the operator: the {lang} deliverable is over 99 MB")
        if a.decision == "reject" or a.chosen_option == "stop":
            return Fail(f"the {lang} deliverable is over 99 MB; the operator stopped",
                        consumes=c1._with_open(ctx, "size"))
        return Complete(dict(out, oversize_accepted=True), consumes=c1._with_open(ctx, "size"),
                        reason=f"{lang} deliverable over 99 MB, delivered as is (operator)")
    return advance


# ---------------------------------------------------------------------------
# cleanup:L
# ---------------------------------------------------------------------------


def language_spend(ctx, lang) -> dict:
    """Committed credits of the language's own scopes, plus the project-level shared ones."""
    db = ctx.db
    runs = [r for r in ((ctx.output(f"probe:{lang}") or {}).get("probe_run_id"),
                        (ctx.output(f"rest:{lang}") or {}).get("run_id")) if r]
    caps = set(db.execute(select(RunCaption.id).where(RunCaption.run_id.in_(runs))).scalars()) if runs else set()
    # + every caption this autopilot created for the language, even if deleted since (gate R1)
    from ...models_autopilot import AutopilotEffect

    for refs, in db.execute(select(AutopilotEffect.created_refs).where(
            AutopilotEffect.autopilot_id.in_(select(ProjectAutopilot.id).where(
                ProjectAutopilot.project_id == ctx.project.id)),
            AutopilotEffect.step_id.in_((f"qa:{lang}", f"captions:{lang}")))):
        if (refs or {}).get("caption"):
            caps.add(refs["caption"])
    # + caption reservations of these runs whose rows are gone (created and
    # deleted by hand): a caption reservation carries its run (gate R3)
    if runs:
        caps |= {k[len("caption:"):] for (k,) in db.execute(
            select(KieSpendReservation.scope_key).where(
                KieSpendReservation.run_id.in_(runs),
                KieSpendReservation.scope_key.like("caption:%")))}
    caps = sorted(caps)

    def total(scopes):
        if not scopes:
            return 0.0
        mc = db.execute(select(func.coalesce(func.sum(KieSpendReservation.charged_mcredits), 0))
                        .where(KieSpendReservation.scope_key.in_(list(scopes)))).scalar_one()
        return round(int(mc or 0) / 1000.0, 1)

    own = {"runs": total([spend_mod.run_scope(r) for r in runs]),
           "captions": total([spend_mod.caption_scope(c) for c in caps])}
    shared = total([spend_mod.transcribe_scope(ctx.project.id), spend_mod.localise_scope(ctx.project.id)])
    credits = round(own["runs"] + own["captions"], 1)
    return {"language": lang, "credits": credits, "usd": round(spend_mod.usd(credits), 2),
            "breakdown": own, "shared_project_credits": shared, "runs": runs}


def _cleanup(lang):
    def advance(ctx):
        rest = ctx.output(f"rest:{lang}") or {}
        clip_id = rest.get("temp_clip_id")
        dels = _round_launches(ctx, ("delete_temp",))
        last = dels[-1] if dels else None
        if last is not None and last.state == "launching":
            return Wait("bookkeeping of the TEMP clip deletion")
        note = ""
        if clip_id:
            clip = ctx.db.get(LibraryClip, clip_id)
            if clip is not None and not fh.is_temp_clip(clip, ctx.project.id):
                note = f"library clip {clip_id[:8]} is no longer a TEMP clip of this autopilot: kept"
            elif clip is not None and last is None:
                return Launch("delete_temp", "localisation_delete_temp_clip", {"clip_id": clip_id},
                              f"delete the {lang} TEMP probe clip (step 17)",
                              consumes=c1._open_answers(ctx))
            elif last is not None and not (last.result or {}).get("dir_removed"):
                if len(dels) >= 2:
                    note = f"the TEMP clip {clip_id[:8]} could not be removed completely"
                else:
                    return Launch("delete_temp:r2", "localisation_delete_temp_clip",
                                  {"clip_id": clip_id}, "finish removing the TEMP clip's files",
                                  consumes=c1._open_answers(ctx))
        spend = language_spend(ctx, lang)
        reason = (f"{lang} spend: {spend['credits']:.0f} credits (${spend['usd']:.2f}) — runs "
                  f"{spend['breakdown']['runs']:.0f}, captions {spend['breakdown']['captions']:.0f}; "
                  f"shared transcription/translation {spend['shared_project_credits']:.0f}")
        return Complete({"spend": spend, "temp_clip_id": clip_id, "note": note or None},
                        consumes=c1._open_answers(ctx), reason=reason + (f"; {note}" if note else ""))
    return advance


# ---------------------------------------------------------------------------
# Step list (appended to C1's by LocalisationFlow.steps)
# ---------------------------------------------------------------------------


def _qa_estimate(ctx):
    return float(spend_mod.llm_credits())


def steps(langs) -> list:
    out = []
    for lang in langs:
        out.append(StepSpec(f"qa:{lang}", f"QA, regenerations and the final listening ({lang})",
                            advance=_qa(lang), needs=(f"rest:{lang}",), paid=True,
                            approvals=("regeneration", "listening_final", "handoff"),
                            estimate=_qa_estimate))
    for lang in langs:
        out.append(StepSpec(f"captions:{lang}", f"Captions: fix words / from the script ({lang})",
                            advance=_captions(lang), needs=(f"qa:{lang}",)))
    for lang in langs:
        out.append(StepSpec(f"tail:{lang}", f"CTA tail from the caption ({lang})",
                            advance=_tail(lang), needs=(f"captions:{lang}",), approvals=("handoff",)))
    for lang in langs:
        out.append(StepSpec(f"cleanup:{lang}", f"Cleanup: TEMP clip, spend ({lang})",
                            advance=_cleanup(lang), needs=(f"tail:{lang}",)))
    return out
