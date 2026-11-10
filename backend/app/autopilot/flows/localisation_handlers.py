"""Action handlers and compute functions of the localisation flow (AICP-365).

docs/autopilot-localisation-flow.md §3. Every handler follows the contract of
``app.autopilot.contract.ActionHandler``: one transaction, one commit, no paid
call before it, ids in ``effect.created_refs``, refusals as HTTPException 4xx /
``ActionRefused``, fresh paths for files, and never a rollback inside
``execute`` (rule (g)). Handlers that must ring a worker's doorbell commit
themselves and enqueue after the commit (rule (c)); the others leave the
commit to the core.

Registered on import of ``app.autopilot.flows.localisation``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from typing import Any, Mapping

from fastapi import HTTPException

from ... import ai_models, languages, localisation_jobs
from ... import spend as spend_mod
from ...localisation_jobs_models import LocalisationPromptJob, ProjectTranscription
from ...models import LibraryClip, Run
from .. import registry
from ..actions import CreateRun
from ..contract import ActionHandler, ActionInvalid, ActionRefused

log = logging.getLogger(__name__)


def _no_params(name: str, params: Mapping[str, Any]) -> dict:
    if params:
        raise ActionInvalid(f"{name}: takes no params, got {sorted(params)}")
    return {}


# ---------------------------------------------------------------------------
# Paid: transcription and translation (durable jobs, app/localisation_jobs.py)
# ---------------------------------------------------------------------------


class TranscribeProject(ActionHandler):
    """One durable transcription of the project's source (paid, one LLM call)."""

    name = "transcribe_project"
    version = 1
    paid = True

    def validate(self, db, project, params):
        return _no_params(self.name, params)

    def estimate(self, db, project, params):
        return float(spend_mod.llm_credits())

    def spend_scopes(self, params, targets):
        return []  # transcribe:{pid} is a project scope the core scans anyway

    def execute(self, db, project, params, effect):
        job = localisation_jobs.admit_transcription(db, project, origin="autopilot")
        effect.created_refs = {"transcription": job.id}
        db.commit()  # the single admission commit (effect row + job + reservation)
        localisation_jobs.enqueue(job)
        return {"transcription": job.id}

    def reconcile(self, db, effect):
        jid = (effect.created_refs or {}).get("transcription")
        return "launched" if jid and db.get(ProjectTranscription, jid) is not None else "unknown"


class LocalisationPrompt(ActionHandler):
    """One durable translate job for one language (paid, up to three LLM calls)."""

    name = "localisation_prompt"
    version = 1
    paid = True

    def validate(self, db, project, params):
        unknown = set(params) - {"language", "model"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        code = languages.normalize(params.get("language"))
        if code is None:
            raise ActionInvalid(f"{self.name}: a known target language is required")
        model = params.get("model") or None
        if model is not None and model not in ai_models.AI_MODELS:
            raise ActionInvalid(f"{self.name}: unknown model {model!r}")
        return {"language": code, "model": model}

    def estimate(self, db, project, params):
        return 3.0 * float(spend_mod.llm_credits())

    def spend_scopes(self, params, targets):
        return []  # localise:{pid} is a project scope

    def execute(self, db, project, params, effect):
        job = localisation_jobs.admit_translate(
            db, project, language=params["language"], swap_character=False,
            model=params.get("model"),
        )
        effect.created_refs = {"translate_job": job.id}
        db.commit()
        localisation_jobs.enqueue(job)
        return {"translate_job": job.id, "fingerprint": job.fingerprint}

    def reconcile(self, db, effect):
        jid = (effect.created_refs or {}).get("translate_job")
        return "launched" if jid and db.get(LocalisationPromptJob, jid) is not None else "unknown"


class LocalisationCreateRun(CreateRun):
    """``create_run`` with "remove original audio" frozen ON for the new run.

    The pipeline reads ``Run.mute_source`` before the project's live setting
    (pipeline_v2.effective_mute_source), so an operator toggling the project
    after the budget was approved cannot change what this run uploads
    (docs/autopilot-localisation-flow.md §4.7)."""

    name = "localisation_create_run"
    version = 1
    paid = True

    def execute(self, db, project, params, effect):
        from sqlalchemy import event

        def _freeze(session, _ctx, _instances):
            for obj in session.new:
                if isinstance(obj, Run) and obj.project_id == project.id and obj.mute_source is None:
                    obj.mute_source = True

        event.listen(db, "before_flush", _freeze)
        try:
            return super().execute(db, project, params, effect)
        finally:
            event.remove(db, "before_flush", _freeze)


# ---------------------------------------------------------------------------
# Free handlers
# ---------------------------------------------------------------------------


class RecutAtPauses(ActionHandler):
    """AICP-347's re-cut at speech pauses, through its transaction-neutral service."""

    name = "recut_at_pauses"
    version = 1
    paid = False

    def validate(self, db, project, params):
        return _no_params(self.name, params)

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ...speech_recut import RecutRefused, recut_project_segments

        try:
            outcome = recut_project_segments(db, project)
        except RecutRefused as exc:
            raise ActionRefused(409, str(exc)) from None
        plan = outcome.plan.to_dict()
        result = {
            "applied": bool(outcome.applied),
            "feasible": bool(plan.get("feasible")),
            "changed": bool(plan.get("changed")),
            "fingerprint": outcome.fingerprint,
            "warnings": list(outcome.warnings or []),
            "boundaries": [b for b in plan.get("boundaries") or [] if b.get("moved")],
        }
        effect.result = result
        return result


class SetMuteSource(ActionHandler):
    """``mute_source = True`` on the project (the localisation default)."""

    name = "set_mute_source"
    version = 1
    paid = False

    def validate(self, db, project, params):
        return _no_params(self.name, params)

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        project.mute_source = True
        return {"mute_source": True}


class CreateTempLibraryClip(ActionHandler):
    """The accepted probe's final, copied into an untagged stitch-library clip.

    The copy is the single source of both the picture reused in the remaining
    run's stitch plan and its voice reference (create_run's
    ``voice_from_library_clip``), so the two can never come from different
    takes. Language stays NULL so the New Run form never picks it as a tail.
    """

    name = "create_temp_library_clip"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"run_id", "name"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        rid = params.get("run_id")
        run = db.get(Run, rid) if isinstance(rid, str) else None
        if run is None or run.project_id != project.id:
            raise ActionInvalid(f"{self.name}: run_id must be a run of this project")
        name = str(params.get("name") or "").strip()[:255] or f"TEMP autopilot {project.id[:8]}"
        return {"run_id": rid, "name": name}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import media as media_mod
        from ...storage import library_clip_dir

        run = db.get(Run, params["run_id"])  # locked by the core (a target)
        status = run.status.value if hasattr(run.status, "value") else str(run.status)
        src = run.result_local_path
        if status != "done" or not src or not os.path.isfile(src):
            raise ActionRefused(409, f"run {run.id[:8]} has no delivered final to reuse")
        clip_id = str(uuid.uuid4())
        dest_dir = library_clip_dir(clip_id)  # fresh, unique (rule (f))
        dest = os.path.join(dest_dir, "clip.mp4")
        digest = hashlib.sha256()
        try:
            with open(src, "rb") as fin, open(dest, "wb") as fout:
                for chunk in iter(lambda: fin.read(1 << 20), b""):
                    digest.update(chunk)
                    fout.write(chunk)
            info = media_mod.probe(dest)
        except media_mod.MediaError as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise ActionRefused(409, f"the probe final is not a playable video: {exc}") from None
        except OSError:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        db.add(LibraryClip(
            id=clip_id, name=params["name"], local_path=dest, duration_sec=info.duration_sec,
            width=info.width, height=info.height, fps=info.fps, language=None,
        ))
        sha = digest.hexdigest()
        effect.created_refs = {"library_clip": clip_id, "source_run": run.id,
                               "source_sha256": sha}
        return {"library_clip": clip_id, "source_sha256": sha,
                "duration_sec": float(info.duration_sec or 0.0)}

    def reconcile(self, db, effect):
        cid = (effect.created_refs or {}).get("library_clip")
        return "launched" if cid and db.get(LibraryClip, cid) is not None else "unknown"


class ProbeFeedback(ActionHandler):
    """Record a rejected probe's comment and the corrected prompt (no domain write).

    It exists so the step can CONSUME the rejected listening answer before it
    asks the regeneration that the comment proposes: the core stops an
    autopilot whose step keeps waiting with a rejected, unconsumed answer."""

    name = "localisation_probe_feedback"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"comment", "prompt", "round"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        prompt = params.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ActionInvalid(f"{self.name}: prompt is required")
        return {"comment": str(params.get("comment") or ""), "prompt": prompt,
                "round": int(params.get("round") or 1)}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        effect.result = dict(params)
        return dict(params)


class AcknowledgeStale(ActionHandler):
    """Consume answers whose subject changed since (a manual rerun): no write.

    The core revalidates every consumed approval before a decision applies; a
    stale one is superseded and the decision discarded, so this launch
    normally never admits. It exists because the core stops an autopilot whose
    step keeps waiting with a rejected, unconsumed answer, and because a
    completion must account for every answered approval (Terra §1 gate R3)."""

    name = "localisation_acknowledge"
    version = 1
    paid = False

    def validate(self, db, project, params):
        names = params.get("names")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ActionInvalid(f"{self.name}: names must be a list of approval names")
        return {"names": sorted(names)}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        effect.result = dict(params)
        return dict(params)


# ---------------------------------------------------------------------------
# Compute functions (free, run workers, frames streamed)
# ---------------------------------------------------------------------------


def source_captions(args: dict, out_dir: str) -> dict:
    """Burned-caption scan of the source at 2 fps (qa_tools, streamed)."""
    from ... import qa_tools

    scan = qa_tools.scan_captions(args["source"], fps=float(args.get("fps") or 2.0))
    spans = [{"start": s.start, "end": s.end, "frames": s.frames} for s in scan.spans[:20]]
    return {"frames": scan.frames, "hits": scan.hits, "hit_ratio": scan.hit_ratio,
            "spans": spans, "threshold": qa_tools.CALIBRATION_MIN_RATIO}


def probe_sheet(args: dict, out_dir: str) -> dict:
    """Source window vs probe result, one frame pair per second, plus lengths."""
    from ... import media as media_mod
    from ... import qa_tools

    source, result = args["source"], args["result"]
    s0, s1 = float(args["source_start"]), float(args["source_end"])
    sheet = os.path.join(out_dir, "probe_sheet.jpg")
    out = qa_tools.comparison_sheet(
        source, result, sheet,
        windows=[{"source_start": s0, "source_end": s1, "result": result,
                  "result_start": 0.0, "label": "probe"}],
        fractions=None, every_sec=1.0, time_map="scale",
    )
    result_sec = float(media_mod.probe(result).duration_sec or 0.0)
    diffs = [s["diff"] for s in out["samples"] if s.get("diff") is not None]
    return {
        "sheet": sheet, "samples": out["samples"][:40],
        "source_sec": round(s1 - s0, 3), "result_sec": round(result_sec, 3),
        "length_delta_sec": round(result_sec - (s1 - s0), 3),
        "max_diff": max(diffs) if diffs else None,
    }


HANDLERS = (TranscribeProject(), LocalisationPrompt(), LocalisationCreateRun(), RecutAtPauses(),
            SetMuteSource(), CreateTempLibraryClip(), ProbeFeedback(), AcknowledgeStale())
for _h in HANDLERS:
    registry.register_handler(_h)
registry.register_compute("localisation_source_captions", source_captions, version=1)
registry.register_compute("localisation_probe_sheet", probe_sheet, version=1)

__all__ = ["HANDLERS", "HTTPException"]
