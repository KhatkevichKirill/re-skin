"""Brand trailing-sibilant check and its free in-place fix, inside ``qa:L`` (AICP-389).

docs/brand-trailing-sibilant.md. Every QA round of the localisation finishing
flow, after segment QA and the best-take decision and BEFORE the paid ASR
caption, the free compute :func:`brand_check` (queue ``brand_asr``, the
brand-asr container) locates every scripted brand occurrence in the stitched
final with a local word ASR and measures its ending (``app.brand_check``).
For each occurrence that ends in a separate trailing sibilant followed by a
pause it renders, in its attempt directory, the full candidate (the window
low-passed, picture stream-copied) and two context previews.

The step then asks ``brandfix:<base sha12>:<occurrence>`` — one occurrence at
a time, before any ``regen:`` / ``listen:`` ask of the round (which it
replaces while still pending) — with two options: "apply this fix (free)"
(the handler :class:`LocalisationApplyBrandAudioPatch`, bound to the base and
candidate SHA-256) and "keep the current audio". The handler admits a
``RunAudioPatch`` of kind ``same_run_lowpass`` under the run-row lock; the
patch job promotes exactly the approved bytes (Drive content replaced under
the same id). The changed result supersedes the round; the next round checks
the patched final again (a window already patched is never offered again,
and the patches per run are bounded).

Names inside ``qa:L``: ``brand:<tag>`` (+ ``:r2``/``:r3``, the compute),
``brandfix:<sha12>:<occ>`` (the ask), ``brandapply:<sha12>:<occ>`` (+ ``:rK``,
the launch), ``brandretry:<patch8>`` / ``brandcheck:<tag>`` /
``brandcap:<sha12>`` (handoffs).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Optional

from sqlalchemy import select

from ... import brand_check as bc
from ...config import settings
from ...models import Run, RunAudioPatch
from .. import registry
from ..contract import ActionHandler, ActionInvalid, ActionRefused, Ask, Compute, Fail, Launch, Option, Wait
from . import localisation_qa as lqa

log = logging.getLogger(__name__)

C_BRAND = "localisation_brand_check"
H_APPLY = "localisation_apply_brand_audio_patch"
QUEUE = "brand_asr"

MAX_CHECK_ATTEMPTS = 3      # a failed / unavailable check is retried, then a handoff
MAX_BRAND_PATCHES = 3       # filter patches per run before a handoff (no progressive muffling)
MAX_APPLY_ATTEMPTS = 3      # apply launches per occurrence (a failed patch is an operator retry)
MAX_CANDIDATES = 4          # candidates rendered per check
PREVIEW_PAD_SEC = 2.0
OUTSIDE_LEVEL_MAX_DB = 0.5  # outside the window the level stays within the audio-patch tolerance
ASR_MANIFEST = os.getenv("BRAND_ASR_MANIFEST", "/opt/asr-small-manifest.json")
ASR_PACKAGES = os.getenv("BRAND_ASR_PACKAGES", "/opt/asr")
NEXT_CAPTION_CREDITS = 20   # shown upfront: the next round's ASR caption is a paid regeneration

ACTIVE_PATCH = ("queued", "processing", "delivering")


def _st(value) -> str:
    return value.value if hasattr(value, "value") else str(value or "")


# ---------------------------------------------------------------------------
# The compute (brand-asr container)
# ---------------------------------------------------------------------------


def _local_asr():
    from ...assistant_asr import LocalASR

    return LocalASR(Path(sys.executable), Path(settings.BRAND_ASR_MODEL_DIR), Path(ASR_MANIFEST),
                    Path(ASR_PACKAGES))


def _overlaps(a, b) -> bool:
    return min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0])) > 0.0


def _audio_duration(path: str) -> float:
    from ...assistant_audio import probe_trusted_audio

    return float(probe_trusted_audio(path, timeout_sec=60).duration_sec)


def brand_check(args: dict, out_dir: str, *, asr=None, asr_probe=None) -> dict:
    """Locate, measure and (for each finding) render the fix of one final.

    Raises on anything that makes the check unavailable (model, decode, ASR,
    render): the effect fails and the step reports "check unavailable" — it is
    never read as "nothing found"."""
    t0 = time.monotonic()
    final = str(args["final"])
    language = args.get("language") or None
    sig0 = lqa.file_sig(final)
    sha0 = bc.sha256_file(final)
    samples = bc.decode_mono(final)
    duration = len(samples) / bc.SAMPLE_RATE
    if not 0 < duration <= bc.MAX_FINAL_SEC:
        raise ValueError(f"final is {duration:.1f} s; the brand check takes up to {bc.MAX_FINAL_SEC:.0f} s")
    asr = asr or _local_asr()
    # Chunk on the audio stream's own duration (what the ASR checks its chunk
    # against), which can be a few ms shorter than the decoded sample count.
    span = min(duration, _audio_duration(final) if asr_probe is None else asr_probe(final))
    per_chunk, rss, t_asr = [], 0, time.monotonic()
    for c0, c1 in bc.chunks(span):
        tr = asr.transcribe(Path(final), start_sec=c0, end_sec=c1,
                            language=language if language and language.isalpha() else None,
                            timeout_sec=300)
        rss = max(rss, int(getattr(tr, "max_rss_kib", 0) or 0))
        per_chunk.append((c0, c1, [{"text": w.text, "start": w.start_sec, "end": w.end_sec,
                                    "probability": w.probability} for w in tr.words]))
    asr_sec = time.monotonic() - t_asr
    words = bc.merge_words(per_chunk)
    res = bc.check(samples, words, list(args.get("expected") or []),
                   list(args.get("brands") or []), language)
    skip = [list(w) for w in args.get("skip_windows") or []]
    candidates: dict[str, dict] = {}
    base_streams = None
    for row in res["occurrences"]:
        if row.get("verdict") != "found":
            continue
        if any(_overlaps(row["window"], w) for w in skip):
            row.update(verdict="patched", reason="this window was already fixed on this run")
            continue
        if len(candidates) >= MAX_CANDIDATES:
            row["note"] = "not rendered in this check (too many findings); checked again after a fix"
            continue
        occ = row["occ_id"]
        params = bc.render_params(row["window"])
        cand = os.path.join(out_dir, f"candidate_{occ}.mp4")
        bc.render_candidate(final, cand, params)
        base_streams = base_streams or bc.probe_streams(final)
        cs = bc.probe_streams(cand)
        cand_samples = bc.decode_mono(cand)
        level_delta = bc.outside_level_delta_db(samples, cand_samples, row["window"])
        checks = {"video_packets_base": base_streams["video_packets"],
                  "video_packets_candidate": cs["video_packets"],
                  "duration_base": round(base_streams["duration_sec"], 3),
                  "duration_candidate": round(cs["duration_sec"], 3),
                  "outside_level_delta_db": level_delta,
                  "outside_residual_db": bc.outside_residual_db(samples, cand_samples, row["window"])}
        del cand_samples
        if (cs["video_packets"] != base_streams["video_packets"]
                or abs(cs["duration_sec"] - base_streams["duration_sec"]) > 0.05
                or level_delta > OUTSIDE_LEVEL_MAX_DB):
            row.update(verdict="unclear", reason="the rendered fix failed its own checks", render_checks=checks)
            continue
        p0 = max(0.0, float(row["window"][0]) - PREVIEW_PAD_SEC)
        p1 = min(duration, float(row["window"][1]) + PREVIEW_PAD_SEC)
        prev_o = os.path.join(out_dir, f"preview_{occ}_original.mp4")
        prev_f = os.path.join(out_dir, f"preview_{occ}_fixed.mp4")
        bc.render_preview(final, prev_o, p0, p1)
        bc.render_preview(cand, prev_f, p0, p1)
        candidates[occ] = {
            "occ_id": occ, "window": row["window"], "measurements": row.get("measurements"),
            "render_params": params, "path": cand, "sha256": bc.sha256_file(cand),
            "size": os.path.getsize(cand), "checks": checks,
            "preview_window": [round(p0, 3), round(p1, 3)],
            "preview_original": prev_o, "preview_fixed": prev_f,
            "preview_original_sha256": bc.sha256_file(prev_o),
            "preview_fixed_sha256": bc.sha256_file(prev_f),
        }
    stable = bc.sha256_file(final) == sha0 and lqa.file_sig(final) == sig0
    return {
        "status": "ok", "stable": stable, "final": final, "base_sha256": sha0, "base_sig": sig0,
        "run_id": args.get("run_id"), "run_version": args.get("run_version"),
        "transcript_version": args.get("transcript_version"),
        "detector_version": bc.DETECTOR_VERSION, "render_version": bc.RENDER_VERSION,
        "duration_sec": round(duration, 3),
        "occurrences": res["occurrences"], "found": [r["occ_id"] for r in res["occurrences"]
                                                     if r.get("verdict") == "found"],
        "candidates": candidates, "reference_level_db": res["reference_level_db"],
        "asr": {"chunks": len(per_chunk), "words": len(words), "max_rss_kib": rss,
                "seconds": round(asr_sec, 1)},
        "queue_delay_sec": _delay(args.get("requested_at")),
        "seconds": round(time.monotonic() - t0, 1),
    }


def _delay(requested_at) -> Optional[float]:
    from datetime import datetime, timezone

    try:
        t = datetime.fromisoformat(str(requested_at))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - t).total_seconds(), 1)


# ---------------------------------------------------------------------------
# The apply handler (autopilot ticker; free)
# ---------------------------------------------------------------------------


_APPLY_FIELDS = {"run_id", "occ_id", "window", "filter_params", "base_path", "base_sha256",
                 "candidate_path", "candidate_sha256", "effect_id"}


def _sha(v, field) -> str:
    if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise ActionInvalid(f"{H_APPLY}: {field} must be a sha256 hex digest")
    return v


class LocalisationApplyBrandAudioPatch(ActionHandler):
    """Admit the approved same-run low-pass fix as a ``RunAudioPatch`` (free)."""

    name = H_APPLY
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - _APPLY_FIELDS
        if unknown or _APPLY_FIELDS - set(params):
            raise ActionInvalid(f"{self.name}: params must be exactly {sorted(_APPLY_FIELDS)}")
        run = db.get(Run, params["run_id"]) if isinstance(params["run_id"], str) else None
        if run is None or run.project_id != project.id:
            raise ActionInvalid(f"{self.name}: run is not a run of this project")
        w = params["window"]
        if (not isinstance(w, (list, tuple)) or len(w) != 2
                or not all(isinstance(x, (int, float)) for x in w) or not 0 <= w[0] < w[1]):
            raise ActionInvalid(f"{self.name}: window must be [start, end]")
        fp = params["filter_params"]
        if (not isinstance(fp, Mapping) or fp.get("kind") != "same_run_lowpass"
                or not isinstance(fp.get("render_version"), int)
                or abs(float(fp.get("start_sec", -1)) - float(w[0])) > 1e-6
                or abs(float(fp.get("end_sec", -1)) - float(w[1])) > 1e-6):
            raise ActionInvalid(f"{self.name}: filter_params do not match the window")
        from ..compute import attempt_dir

        root = os.path.realpath(os.path.dirname(attempt_dir(project.id, str(params["effect_id"]), "x")))
        cand = os.path.realpath(str(params["candidate_path"]))
        if not cand.startswith(root + os.sep):
            raise ActionInvalid(f"{self.name}: the candidate is not inside its compute's directory")
        return {"run_id": run.id, "occ_id": str(params["occ_id"]),
                "window": [float(w[0]), float(w[1])], "filter_params": dict(fp),
                "base_path": str(params["base_path"]), "base_sha256": _sha(params["base_sha256"], "base_sha256"),
                "candidate_path": str(params["candidate_path"]),
                "candidate_sha256": _sha(params["candidate_sha256"], "candidate_sha256"),
                "effect_id": str(params["effect_id"])}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import api_v2
        from ...pipeline_audio_patch import AudioPatchValidationError, create_lowpass_patch
        from ...tasks import enqueue_process_audio_patch

        if not settings.AUTOPILOT_BRAND_FIX_ENABLED:
            raise ActionRefused(409, "the autopilot brand fix is disabled (AUTOPILOT_BRAND_FIX_ENABLED)")
        run = db.get(Run, params["run_id"])
        api_v2._lock_run_for_result_mutation(run, db)
        active = api_v2._active_audio_patch(run.id, db)
        if active is not None:
            raise ActionRefused(409, f"audio patch {active.id[:8]} is already running on this run")
        attempt = api_v2._active_range_attempt(run.id, db)
        if attempt is not None:
            raise ActionRefused(409, f"range attempt {attempt.id[:8]} is running on this run")
        try:
            patch = create_lowpass_patch(
                run=run, window=tuple(params["window"]), filter_params=params["filter_params"],
                base_path=params["base_path"], base_sha256=params["base_sha256"],
                candidate_path=params["candidate_path"], candidate_sha256=params["candidate_sha256"],
                db=db)
        except AudioPatchValidationError as exc:
            raise ActionRefused(409, f"stale brand fix: {exc}") from None
        effect.created_refs = {"audio_patch": patch.id}
        effect.result = {"audio_patch": patch.id, "occ_id": params["occ_id"],
                         "window": params["window"], "candidate_sha256": params["candidate_sha256"]}
        db.commit()
        # A crash right here leaves the row queued with no job: the recovery
        # reaper fails it after its grace and the step offers a retry (a NEW
        # row) — never a duplicate patch of the same admission.
        try:
            enqueue_process_audio_patch(patch.id)
        except Exception:  # the row is committed: reaped, then retried by the operator
            log.exception("brand fix: enqueue of audio patch %s failed", patch.id)
        return dict(effect.result)

    def reconcile(self, db, effect):
        pid = (effect.created_refs or {}).get("audio_patch")
        return "launched" if pid and db.get(RunAudioPatch, pid) is not None else "unknown"


registry.register_handler(LocalisationApplyBrandAudioPatch())
registry.register_compute(C_BRAND, brand_check, version=1, queue=QUEUE)


# ---------------------------------------------------------------------------
# Planning (read-only), called from localisation_finish._qa
# ---------------------------------------------------------------------------


def run_patches(db, run_id: str) -> list:
    return db.execute(select(RunAudioPatch).where(RunAudioPatch.run_id == run_id)
                      .order_by(RunAudioPatch.created_at)).scalars().all()


def current_patch(db, run_id: str):
    """The promoted audio patch (any kind) whose output IS the run's result."""
    run = db.get(Run, run_id)
    if run is None or not run.result_local_path:
        return None
    for p in run_patches(db, run_id):
        if _st(p.status) == "done" and p.output_path == run.result_local_path:
            return p
    return None


def rerun_warning(db, run_id: str) -> Optional[str]:
    """What a rerun / restore offered now would drop, or None (AICP-389 §7)."""
    p = current_patch(db, run_id)
    if p is None:
        return None
    what = ("the brand fix" if (getattr(p, "kind", None) or "donor") == "same_run_lowpass"
            else "the audio fix")
    return (f"drops {what} at {p.target_start_sec:.2f}-{p.target_end_sec:.2f} s "
            "(the re-stitched final replaces it)")


def _pending_names(ctx, prefix: str) -> list[str]:
    from ...models_autopilot import AutopilotApproval, ProjectAutopilot

    return sorted(set(ctx.db.execute(
        select(AutopilotApproval.name).join(
            ProjectAutopilot, ProjectAutopilot.id == AutopilotApproval.autopilot_id).where(
            ProjectAutopilot.project_id == ctx.project.id,
            ProjectAutopilot.status.in_(("active", "paused")),
            AutopilotApproval.step_id == ctx.step_id,
            AutopilotApproval.step_generation == ctx.step_generation,
            AutopilotApproval.status == "pending",
            AutopilotApproval.name.like(prefix + "%"))).scalars()))


def _answered(a) -> bool:
    return a is not None and a.decision is not None


def _check_effects(ctx, tag):
    """Every check attempt of this round, oldest first (``brand:<tag>``, ``:rK``)."""
    own = [e for e in ctx.effects(f"brand:{tag}")
           if e.name == f"brand:{tag}" or e.name.startswith(f"brand:{tag}:r")]
    return sorted(own, key=lambda e: e.created_at)


def _result_usable(res: Mapping, run, transcript_version: str) -> bool:
    # The transcript (brands, script) can change without the run's version:
    # a check of the old script is no evidence about the new one (gate R2).
    return (res.get("status") == "ok" and res.get("stable") is True
            and res.get("final") == run.result_local_path
            and res.get("base_sig") == lqa.file_sig(run.result_local_path)
            and res.get("detector_version") == bc.DETECTOR_VERSION
            and res.get("transcript_version") == transcript_version)


def lineage_patches(db, run_id: str) -> list:
    """The ``done`` patches whose outputs led to the run's CURRENT result.

    Walks back from ``result_local_path`` through each patch's base. A rerun
    re-stitches the final and drops every earlier patch, so only this chain
    describes the current bytes (gate R3): its filter windows are the ones
    already fixed, and its length bounds progressive muffling."""
    run = db.get(Run, run_id)
    if run is None or not run.result_local_path:
        return []
    done = {p.output_path: p for p in run_patches(db, run_id)
            if _st(p.status) == "done" and p.output_path}
    out, path, seen = [], run.result_local_path, set()
    while path in done and path not in seen:
        seen.add(path)
        p = done[path]
        out.append(p)
        path = p.base_result_path
    return out


def brand_round(ctx, lang, run, tag, n, pieces, segres, with_open, *, caption_paid=True):
    """``(decision, note)``: a decision to return now, or ``None`` to go on
    (the ASR caption, regeneration and listening of the round), plus a short
    note for the round summary. Read-only."""
    if not settings.AUTOPILOT_BRAND_FIX_ENABLED:
        pending = _pending_names(ctx, "brand")
        if pending:
            return Wait("the brand fix is disabled: closing its open questions", retire=pending), None
        return None, None
    if _answered(ctx.approval(f"regen:{tag}")) or _answered(ctx.approval(f"listen:{tag}")):
        return None, {"status": "skipped", "why": "this round's regeneration/listening was already answered"}
    brands = (ctx.project.transcript or {}).get("brands") or []
    expected = lqa.final_script(pieces, lqa.piece_secs(segres, pieces))
    occs = bc.brand_occurrences(expected, brands, lang)
    if not occs:
        return None, None
    not_checked = [{"occ": o.occ_id, "brand": o.brand, "why": o.reason} for o in occs if o.status != "checkable"]
    if not any(o.status == "checkable" for o in occs):
        return None, {"status": "not_checked", "occurrences": not_checked}
    patches = run_patches(ctx.db, run.id)
    if any(_st(p.status) in ACTIVE_PATCH for p in patches):
        return Wait(f"an audio patch is running on the {lang} run"), None
    lowpass = [p for p in lineage_patches(ctx.db, run.id)
               if (getattr(p, "kind", None) or "donor") == "same_run_lowpass"]
    skip = [[p.target_start_sec, p.target_end_sec] for p in lowpass]

    # 1. the compute (bounded retries, then a handoff)
    tv = ctx.transcript_version()
    args = {"final": run.result_local_path, "run_id": run.id, "language": lang,
            "run_version": ctx.run_version(run.id), "transcript_version": tv,
            "expected": expected, "brands": brands, "skip_windows": skip}
    effs = _check_effects(ctx, tag)
    if not effs:
        return Compute(f"brand:{tag}", C_BRAND, dict(args, requested_at=ctx.now.isoformat()),
                       f"brand pronunciation check of the {lang} final (free, round {n})",
                       timeout_sec=1800), None
    eff = effs[-1]
    if eff.state in ("queued", "running"):
        return Wait(f"checking the brand words of the {lang} final (free; {eff.state})"), None
    res = dict(eff.result or {}) if eff.state == "succeeded" else {}
    if not _result_usable(res, run, tv):
        script_moved = bool(res) and res.get("status") == "ok" \
            and res.get("transcript_version") != tv
        failures = sum(1 for e in effs if not (
            e.state == "succeeded" and (e.result or {}).get("status") == "ok"
            and (e.result or {}).get("transcript_version") != tv))
        why = eff.error or ("the script changed while it was checked" if script_moved
                            else "the final changed while it was checked" if res else "no result")
        # A total cap over every attempt, and a smaller one over real failures.
        if len(effs) < MAX_CHECK_ATTEMPTS + 3 and (script_moved or failures < MAX_CHECK_ATTEMPTS):
            return Compute(f"brand:{tag}:r{len(effs) + 1}", C_BRAND,
                           dict(args, requested_at=ctx.now.isoformat()),
                           f"brand check again ({why[:120]})", timeout_sec=1800), None
        name = f"brandcheck:{tag}"
        a = ctx.approval(name)
        if a is None:
            return Ask(name, "handoff",
                       f"The brand pronunciation check of the {lang} final is unavailable "
                       f"({why[:200]}). Listen for the brand yourself, then continue",
                       {"run": run.short_id, "language": lang, "round": n, "error": why[:500],
                        "attempts": len(effs)},
                       subject=lang, about={f"run:{run.id}": ctx.run_version(run.id)},
                       title=f"Brand check unavailable ({lang})",
                       options=[Option("continue", "Go on without the check",
                                       consequence="nothing is fixed; you listen to the final later"),
                                Option("stop", "Stop the autopilot",
                                       consequence="ends this attempt (failed); nothing more is spent")]), None
        if a.status == "pending":
            return Wait(f"waiting for the operator: the {lang} brand check is unavailable"), None
        if a.decision == "reject" or a.chosen_option == "stop":
            return Fail(f"the {lang} brand check was unavailable; the operator stopped",
                        consumes=with_open(name)), None
        return None, {"status": "unavailable", "error": why[:200]}

    cands = dict(res.get("candidates") or {})
    note = {"status": "checked", "found": list(res.get("found") or []),
            "occurrences": [{"occ": r["occ_id"], "verdict": r.get("verdict"),
                             "why": r.get("reason"), "window": r.get("window")}
                            for r in res.get("occurrences") or []][:12]}
    if not cands:
        return None, note
    base12 = str(res["base_sha256"])[:12]
    done_lowpass = lowpass
    if len(done_lowpass) >= MAX_BRAND_PATCHES:
        name = f"brandcap:{base12}"
        a = ctx.approval(name)
        if a is None:
            return Ask(name, "handoff",
                       f"The {lang} final still has a trailing sibilant after "
                       f"{len(done_lowpass)} brand fixes: listen and decide",
                       {"run": run.short_id, "language": lang, "patches": len(done_lowpass),
                        "found": sorted(cands)}, subject=lang,
                       about={f"run:{run.id}": ctx.run_version(run.id)},
                       title=f"Brand fix limit reached ({lang})",
                       options=[Option("continue", "Go on to the listening check",
                                       consequence="nothing more is fixed automatically"),
                                Option("stop", "Stop the autopilot",
                                       consequence="ends this attempt (failed); nothing more is spent")]), None
        if a.status == "pending":
            return Wait(f"waiting for the operator: the {lang} brand fix limit"), None
        if a.decision == "reject" or a.chosen_option == "stop":
            return Fail(f"{lang}: brand fix limit reached; the operator stopped",
                        consumes=with_open(name)), None
        return None, dict(note, capped=True)

    # 2. one occurrence at a time
    for occ in sorted(cands, key=lambda k: float(cands[k]["window"][0])):
        c = cands[occ]
        name = f"brandfix:{base12}:{occ}"
        a = ctx.approval(name)
        params = {"run_id": run.id, "occ_id": occ, "window": list(c["window"]),
                  "filter_params": dict(c["render_params"]), "base_path": res["final"],
                  "base_sha256": res["base_sha256"], "candidate_path": c["path"],
                  "candidate_sha256": c["sha256"], "effect_id": eff.id}
        if a is None:
            w0, w1 = c["window"]
            m = c.get("measurements") or {}
            arts = [ctx.artifact(c["preview_original"], label=f"before ({c['preview_window'][0]:.1f}-"
                                 f"{c['preview_window'][1]:.1f} s)", kind="video", effect_id=eff.id),
                    ctx.artifact(c["preview_fixed"], label="after: the fix", kind="video", effect_id=eff.id)]
            return Ask(
                name, "handoff",
                f"The brand in the {lang} final seems to end with an extra \"-s\" at "
                f"{w0:.2f}-{w1:.2f} s. Listen to both previews: apply the free fix?",
                {"run": run.short_id, "language": lang, "round": n, "occurrence": occ,
                 "window": [w0, w1], "measurements": m, "render": c["render_params"],
                 "candidate_sha256": c["sha256"], "base_sha256": res["base_sha256"],
                 "detector_version": res.get("detector_version"),
                 "cost": (f"the fix is free; the next QA round's ASR caption is a paid "
                          f"regeneration (~{NEXT_CAPTION_CREDITS} credits), as after any change"
                          if caption_paid else
                          f"the fix is free; this round's ASR caption (~{NEXT_CAPTION_CREDITS} "
                          "credits, in the approved budget) then runs on the fixed final"),
                 "others": [k for k in sorted(cands) if k != occ]},
                subject=lang, artifacts=arts,
                about={f"run:{run.id}": ctx.run_version(run.id),
                       f"project_transcript:{ctx.project.id}": ctx.transcript_version(),
                       f"effect:{eff.id}": ctx.version(f"effect:{eff.id}")},
                title=f"Extra \"-s\" on the brand? ({lang}, {w0:.1f} s)",
                replaces=(f"regen:{tag}", f"listen:{tag}"),
                options=[Option("apply", "Apply this fix (free)", handler=H_APPLY, params=params,
                                consequence=("low-passes only this window, same Drive link; "
                                             + (f"the next caption is a ~{NEXT_CAPTION_CREDITS}-credit "
                                                "regeneration" if caption_paid else
                                                "then the budgeted caption runs"))),
                         Option("keep", "Keep the current audio",
                                consequence="nothing changes; this ending is not asked again")]), None
        if a.status == "pending":
            return Wait(f"waiting for the operator: fix the brand ending in the {lang} final?"), None
        if a.decision == "reject":
            return Fail(f"the operator stopped at the {lang} brand fix", consumes=with_open(name)), None
        if a.chosen_option == "keep":
            continue
        # "apply": launch (once per attempt), then follow the patch to its end
        d = _apply(ctx, lang, run, base12, occ, params, name, with_open)
        if d is not None:
            return d, None
        # the fix failed and the operator kept the audio: the next occurrence
    return None, dict(note, kept=sorted(cands))


def _apply(ctx, lang, run, base12, occ, params, ask_name, with_open):
    names = [f"brandapply:{base12}:{occ}"] + [f"brandapply:{base12}:{occ}:r{k}"
                                              for k in range(2, MAX_APPLY_ATTEMPTS + 1)]
    effs = [(nm, ctx.effect(nm)) for nm in names]
    done = [(nm, e) for nm, e in effs if e is not None]
    if not done:
        return Launch(names[0], H_APPLY, params,
                      f"apply the free brand fix to the {lang} final (operator approved)",
                      consumes=with_open(ask_name))
    nm, e = done[-1]
    if e.state == "launching":
        return Wait("bookkeeping of the brand fix")
    pid = (e.created_refs or {}).get("audio_patch")
    patch = ctx.db.get(RunAudioPatch, pid) if pid else None
    st = _st(patch.status) if patch is not None else "gone"
    if st in ACTIVE_PATCH:
        return Wait(f"applying the brand fix to the {lang} final ({st})")
    if st == "done":
        # The run's result moved: the next tick plans a new round on it.
        return Wait(f"the {lang} brand fix is applied; checking the new final", recheck_sec=5)
    why = (patch.error_message if patch is not None else "the patch row is gone") or st
    retry = f"brandretry:{(pid or nm)[:8]}"
    a = ctx.approval(retry)
    can_retry = len(done) < MAX_APPLY_ATTEMPTS
    if a is None:
        opts = ([Option("retry", "Try the fix again", handler=H_APPLY, params=params,
                        consequence="admits the same approved fix again (free)")] if can_retry else [])
        opts += [Option("keep", "Keep the current audio", consequence="nothing changes"),
                 Option("stop", "Stop the autopilot",
                        consequence="ends this attempt (failed); nothing more is spent")]
        return Ask(retry, "handoff",
                   f"The {lang} brand fix did not apply ({why[:200]}). The delivered final is unchanged",
                   {"run": run.short_id, "language": lang, "occurrence": occ, "error": why[:500],
                    "attempt": len(done)}, subject=lang,
                   about={f"run:{run.id}": ctx.run_version(run.id)},
                   title=f"The {lang} brand fix failed", options=opts)
    if a.status == "pending":
        return Wait(f"waiting for the operator: the {lang} brand fix failed")
    if a.decision == "reject" or a.chosen_option == "stop":
        return Fail(f"the {lang} brand fix failed ({why[:120]}); the operator stopped",
                    consumes=with_open(retry))
    if a.chosen_option == "retry" and can_retry:
        return Launch(names[len(done)], H_APPLY, params,
                      f"apply the brand fix to the {lang} final again (operator)",
                      consumes=with_open(retry))
    return None  # keep: the round goes on (the caller treats None as "continue")
