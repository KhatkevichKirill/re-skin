"""Action handlers and compute functions of the localisation finishing flow (AICP-366).

docs/autopilot-localisation-finishing.md §3 and §5. Every handler follows the
core contract (``app.autopilot.contract.ActionHandler``): one transaction and
one commit, no paid call before it, created ids in ``effect.created_refs``,
refusals as HTTPException 4xx / ``ActionRefused``, never a rollback inside
``execute``. Handlers that ring a worker's doorbell commit themselves and
enqueue after the commit.

AICP-368 (caption placement over the original caption area): every caption job
this flow creates freezes the project's placement — :func:`caption_placement_kwargs`
(``create_caption``'s ``placement`` field) and :func:`frozen_caption_placement`
(``caption_placement.snapshot_for_caption`` for a row created here); the flow
waits for a pending scan and re-derives a failed or lost one
(:func:`caption_placement_state`, :class:`LocalisationRederivePlacement`).

Registered on import of ``app.autopilot.flows.localisation``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from typing import Any, Mapping, Optional

from sqlalchemy import event

from ... import spend as spend_mod
from ...models import LibraryClip, Run, RunCaption, RunTailEdit
from .. import registry
from ..contract import ActionHandler, ActionInvalid, ActionRefused
from . import localisation_qa as lqa

log = logging.getLogger(__name__)

TEMP_CLIP_PREFIX = "TEMP autopilot "
# The caption placement choice the autopilot burns with (AICP-368's
# create_caption field): the project's own placement, inherited from a
# subtitle-removal parent. None of this flow's code decides placement.
CAPTION_PLACEMENT_CHOICE = "project"


def _st(value) -> str:
    return value.value if hasattr(value, "value") else str(value or "")


# ---------------------------------------------------------------------------
# AICP-368 hook
# ---------------------------------------------------------------------------


def caption_placement_kwargs() -> dict:
    """Extra keyword arguments for ``api_v2.create_caption`` (AICP-368).

    ``placement="project"`` freezes the project's caption placement onto the
    job exactly as an operator's click with the default choice. Passed
    explicitly: an omitted FastAPI ``Form`` default arrives as a ``FieldInfo``."""
    return {"placement": CAPTION_PLACEMENT_CHOICE}


def frozen_caption_placement(project) -> dict:
    """Column values for a RunCaption this module creates without the endpoint:
    AICP-368's ``snapshot_for_caption`` (else the caption burns at the house position)."""
    from ... import caption_placement as cp

    return {"placement": cp.snapshot_for_caption(project, CAPTION_PLACEMENT_CHOICE)}


# A pending scan older than this is treated as lost (its enqueue failed or the
# worker died) and re-derived.
PLACEMENT_PENDING_STALE_SEC = 1800


def caption_placement_state(project, now=None) -> str:
    """``none`` | ``pending`` | ``stale`` | ``failed`` | ``settled`` of the AICP-368 placement.

    ``pending`` = the derivation scan has not finished: a caption created now
    would freeze the provisional value (a burn-only retry keeps it), so the
    flow waits; ``stale`` = pending for longer than
    :data:`PLACEMENT_PENDING_STALE_SEC`; ``failed`` is retryable (re-derive)."""
    from datetime import datetime, timezone

    value = getattr(project, "caption_placement", None)
    if not isinstance(value, dict) or not value:
        return "none"
    src = value.get("source") or {}
    scans = {src.get("scan"), src.get("origin_scan")}
    if "pending" in scans:
        at = src.get("derived_at") or src.get("inherited_at")
        if now is not None and at:
            try:
                t = datetime.fromisoformat(str(at).replace("Z", "+00:00"))
                n = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if (n - t).total_seconds() > PLACEMENT_PENDING_STALE_SEC:
                    return "stale"
            except ValueError:
                pass
        return "pending"
    if "failed" in scans:
        return "failed"
    return "settled"


def placement_sig(project) -> str:
    body = json.dumps(getattr(project, "caption_placement", None), sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


class LocalisationRederivePlacement(ActionHandler):
    """Re-derive the project's caption placement (AICP-368) after a failed or lost scan (free).

    ``caption_placement.request_derivation`` locks the project row, writes the
    provisional value and commits once (the admission commit); the scan is
    enqueued after it. Its only rollback is the no-lineage refusal, checked
    read-only first, so rule (g) holds."""

    name = "localisation_rederive_placement"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"expect_placement"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        return {"expect_placement": str(params.get("expect_placement") or "")}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import caption_placement as cp

        # The state the flow planned on must still hold under the project row
        # lock: a scan that settled or an operator's manual placement since is
        # never overwritten by a new provisional derivation (gate R2).
        db.refresh(project, with_for_update=True)
        if placement_sig(project) != params["expect_placement"]:
            raise ActionRefused(409, "the caption placement changed since it was planned")
        if not cp.removal_chain(db, project):
            raise ActionRefused(409, "the project has no subtitle-removal lineage to derive from")
        effect.result = {"requested": True}
        rev = cp.request_derivation(db, project)  # one commit
        queued = cp.enqueue_scan(project.id, rev)
        return {"rev": rev, "queued": bool(queued)}


# ---------------------------------------------------------------------------
# Caption fence
# ---------------------------------------------------------------------------


def caption_sig(caption: Optional[RunCaption]) -> str:
    """Digest of what a caption decision is about: status, lines, their source, output."""
    if caption is None:
        return "gone"
    body = json.dumps([_st(caption.status), caption.lines, caption.lines_source,
                       caption.output_path], sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def _lock_caption_checked(db, caption_id: str, expect: Optional[str]) -> RunCaption:
    from ... import api_v2

    caption = api_v2._lock_caption_or_404(caption_id, db)
    if expect is not None and caption_sig(caption) != expect:
        raise ActionRefused(409, f"caption {caption_id[:8]} changed since it was planned")
    return caption


def _lines_param(name: str, lines) -> list[dict]:
    from ... import api_v2

    if not isinstance(lines, list) or not lines:
        raise ActionInvalid(f"{name}: a non-empty lines list is required")
    try:
        return api_v2._validate_caption_lines_payload({"lines": lines})
    except Exception as exc:  # HTTPException 400 from the shared validator
        raise ActionInvalid(f"{name}: {getattr(exc, 'detail', exc)}") from None


def _run_of_project(db, project, rid, name: str) -> Run:
    run = db.get(Run, rid) if isinstance(rid, str) else None
    if run is None or run.project_id != project.id:
        raise ActionInvalid(f"{name}: run_id must be a run of this project")
    return run


# ---------------------------------------------------------------------------
# Paid: the ASR caption
# ---------------------------------------------------------------------------


class LocalisationCreateCaption(ActionHandler):
    """The caption job of the stitched final: ASR of the generated audio + burn (paid, one LLM call)."""

    name = "localisation_create_caption"
    version = 1
    paid = True

    def validate(self, db, project, params):
        unknown = set(params) - {"run_id", "language"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        _run_of_project(db, project, params.get("run_id"), self.name)
        return {"run_id": params["run_id"], "language": params.get("language") or None}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def estimate(self, db, project, params):
        return float(spend_mod.llm_credits())

    def spend_scopes(self, params, targets):
        return []  # caption:<id> is registered in the admission flush (below)

    def execute(self, db, project, params, effect):
        from ... import api_v2
        from ..spend_view import register_scopes

        def _capture(session, _ctx, _instances):
            new = [o for o in session.new if isinstance(o, RunCaption) and o.run_id == params["run_id"]]
            for o in new:
                if o.id is None:
                    o.id = str(uuid.uuid4())
            if new:
                refs = dict(effect.created_refs or {})
                refs["caption"] = new[0].id
                effect.created_refs = refs
                register_scopes(session, project.id, [spend_mod.caption_scope(o.id) for o in new],
                                autopilot_id=effect.autopilot_id, source="admission")

        event.listen(db, "before_flush", _capture)
        try:
            resp = api_v2.create_caption(params["run_id"], language=params.get("language"),
                                         source_tail_edit_id=None, db=db,
                                         **caption_placement_kwargs())
        finally:
            event.remove(db, "before_flush", _capture)
        return {"caption": resp.caption_id, "status": resp.status}

    def reconcile(self, db, effect):
        cid = (effect.created_refs or {}).get("caption")
        return "launched" if cid and db.get(RunCaption, cid) is not None else "unknown"


# ---------------------------------------------------------------------------
# Free: caption lines, script caption, tail, TEMP clip deletion, feedback
# ---------------------------------------------------------------------------


def _requeue_burn(db, caption: RunCaption, lines: list[dict]) -> None:
    from ...state_machine import CaptionStatus, transition

    if caption.status not in (CaptionStatus.done, CaptionStatus.failed):
        raise ActionRefused(409, f"caption {caption.id[:8]} is {_st(caption.status)}")
    caption.lines = lines
    caption.lines_source = "manual"  # the worker burns only: no transcription, no spend
    caption.lines_reused_from = None
    caption.error_message = None
    transition(caption, CaptionStatus.queued)


class LocalisationCaptionLines(ActionHandler):
    """New caption lines on an existing caption job + a burn-only re-queue (free).

    ``PATCH /captions/{id}/lines`` + ``POST /captions/{id}/retry`` in ONE
    commit and WITHOUT the retry's LLM reservation: with manual, non-empty
    lines the worker skips the transcription (pipeline_captions), so nothing
    is billed. Fenced by ``expect_caption`` (design §5.5)."""

    name = "localisation_caption_lines"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"caption_id", "lines", "expect_caption", "mode", "changes"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        cid = params.get("caption_id")
        cap = db.get(RunCaption, cid) if isinstance(cid, str) else None
        run = db.get(Run, cap.run_id) if cap is not None else None
        if cap is None or run is None or run.project_id != project.id:
            raise ActionInvalid(f"{self.name}: caption_id must be a caption of this project")
        return {"caption_id": cid, "lines": _lines_param(self.name, params.get("lines")),
                "expect_caption": str(params.get("expect_caption") or ""),
                "mode": str(params.get("mode") or "asr_fix"),
                "changes": int(params.get("changes") or 0)}

    def targets(self, db, project, params):
        cap = db.get(RunCaption, params["caption_id"])
        return {"runs": [cap.run_id]} if cap is not None else {}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import api_v2

        cap = db.get(RunCaption, params["caption_id"])
        if cap is None:
            raise ActionRefused(409, "the caption is gone")
        if cap.source_type == "run":
            api_v2._admit_retained_run_source(db, [cap.run_id])  # run row, then the caption
        cap = _lock_caption_checked(db, params["caption_id"], params["expect_caption"] or None)
        _requeue_burn(db, cap, params["lines"])
        effect.created_refs = {"caption": cap.id}
        effect.result = {"caption": cap.id, "mode": params["mode"], "changes": params["changes"],
                         "lines": len(params["lines"])}
        db.commit()  # the single admission commit
        api_v2.enqueue_process_caption(cap.id)
        return dict(effect.result)

    def reconcile(self, db, effect):
        return "launched"


class LocalisationScriptCaption(ActionHandler):
    """A new caption job whose lines come from the script (garbled / missing ASR; free)."""

    name = "localisation_script_caption"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"run_id", "language", "lines"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        _run_of_project(db, project, params.get("run_id"), self.name)
        return {"run_id": params["run_id"], "language": params.get("language") or None,
                "lines": _lines_param(self.name, params.get("lines"))}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import api_v2
        from ...state_machine import CaptionStatus, RunStatus

        run = db.get(Run, params["run_id"])
        api_v2._lock_run_for_result_mutation(run, db)
        if run.status != RunStatus.done or not run.result_local_path or not os.path.exists(
                run.result_local_path):
            raise ActionRefused(409, f"run {run.id[:8]} has no delivered final to caption")
        language = api_v2._validate_language(params.get("language"))
        cap = RunCaption(id=str(uuid.uuid4()), run_id=run.id, language=language, source_type="run",
                         status=CaptionStatus.queued, lines=params["lines"], lines_source="manual",
                         **frozen_caption_placement(project))
        db.add(cap)
        effect.created_refs = {"caption": cap.id}
        effect.result = {"caption": cap.id, "mode": "script", "lines": len(params["lines"])}
        db.commit()
        api_v2.enqueue_process_caption(cap.id)
        return dict(effect.result)

    def reconcile(self, db, effect):
        cid = (effect.created_refs or {}).get("caption")
        return "launched" if cid and db.get(RunCaption, cid) is not None else "unknown"


class LocalisationCreateTail(ActionHandler):
    """The CTA tail chained off the caption: ``cut_sec = floor2(duration − 0.01)`` (free)."""

    name = "localisation_create_tail"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"run_id", "caption_id", "library_clip_id", "expect_caption"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        _run_of_project(db, project, params.get("run_id"), self.name)
        cap = db.get(RunCaption, params.get("caption_id")) if isinstance(params.get("caption_id"), str) else None
        if cap is None or cap.run_id != params["run_id"]:
            raise ActionInvalid(f"{self.name}: caption_id must be a caption of run_id")
        clip = params.get("library_clip_id")
        if not isinstance(clip, str) or not clip:
            raise ActionInvalid(f"{self.name}: library_clip_id is required")
        return {"run_id": params["run_id"], "caption_id": params["caption_id"],
                "library_clip_id": clip, "expect_caption": str(params.get("expect_caption") or "")}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import api_v2
        from ... import media as media_mod
        from ...state_machine import CaptionStatus

        cap = _lock_caption_checked(db, params["caption_id"], params["expect_caption"] or None)
        if cap.status != CaptionStatus.done or not cap.output_path or not os.path.exists(cap.output_path):
            raise ActionRefused(409, f"caption {cap.id[:8]} has no captioned video to extend")
        try:
            duration = float(media_mod.probe(cap.output_path).duration_sec or 0.0)
        except media_mod.MediaError as exc:
            raise ActionRefused(409, f"cannot probe the captioned video: {exc}") from None
        cut = lqa.tail_cut_sec(duration)

        def _capture(session, _ctx, _instances):
            new = [o for o in session.new if isinstance(o, RunTailEdit) and o.run_id == params["run_id"]]
            for o in new:
                if o.id is None:
                    o.id = str(uuid.uuid4())
            if new:
                effect.created_refs = {"tail_edit": new[0].id}
                effect.result = {"tail_edit": new[0].id, "cut_sec": cut, "source_sec": duration}

        event.listen(db, "before_flush", _capture)
        try:
            resp = api_v2.create_tail_edit(params["run_id"], cut_sec=cut,
                                           library_clip_id=params["library_clip_id"],
                                           source_caption_id=params["caption_id"], db=db)
        finally:
            event.remove(db, "before_flush", _capture)
        return {"tail_edit": resp.tail_edit_id, "cut_sec": cut, "source_sec": duration}

    def reconcile(self, db, effect):
        tid = (effect.created_refs or {}).get("tail_edit")
        return "launched" if tid and db.get(RunTailEdit, tid) is not None else "unknown"


def temp_clip_path(clip_id: str) -> str:
    """A library clip's directory WITHOUT creating it (storage.library_clip_dir makes it)."""
    from ... import storage

    return os.path.join(storage._BASE, "library", clip_id)


def is_temp_clip(clip: Optional[LibraryClip], project_id: str) -> bool:
    """A TEMP probe clip of this project's autopilot (C1's temp_clip_spec name, untagged)."""
    return bool(clip is not None and clip.language is None
                and (clip.name or "").startswith(f"{TEMP_CLIP_PREFIX}{project_id[:8]} "))


class LocalisationDeleteTempClip(ActionHandler):
    """Delete the language's TEMP probe clip (step 17; free). Idempotent across the
    endpoint's commit-then-rmtree boundary: with the row gone it removes the dir."""

    name = "localisation_delete_temp_clip"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"clip_id"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        cid = params.get("clip_id")
        if not isinstance(cid, str) or not cid:
            raise ActionInvalid(f"{self.name}: clip_id is required")
        return {"clip_id": cid}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from sqlalchemy import Text, select

        from ... import api_v2

        cid = params["clip_id"]
        clip_dir = temp_clip_path(cid)
        clip = db.get(LibraryClip, cid)
        if clip is None:  # a retry after the endpoint's commit: the directory only
            shutil.rmtree(clip_dir, ignore_errors=True)
            if os.path.exists(clip_dir):
                raise ActionRefused(409, f"could not remove the TEMP clip directory {clip_dir}")
            out = {"deleted": False, "dir_removed": True}
            effect.result = out
            return out
        if not is_temp_clip(clip, project.id):
            raise ActionRefused(409, f"library clip {cid[:8]} is not this autopilot's TEMP clip")
        # Serialise with every re-stitch admission of a run that names the clip
        # (api_v2._require_stitch_library_clips runs under the same run lock):
        # lock them all, in id order, BEFORE the endpoint's active-run check.
        rids = sorted(rid for rid, plan in db.execute(
            select(Run.id, Run.stitch_plan).where(
                Run.stitch_plan.is_not(None), Run.stitch_plan.cast(Text).like(f"%{cid}%"))).all()
            if any(isinstance(e, dict) and e.get("kind") == "library" and e.get("clip_id") == cid
                   for e in (plan or [])))
        if rids:
            db.execute(select(Run.id).where(Run.id.in_(rids)).order_by(Run.id).with_for_update()).all()
        effect.created_refs = {"clip": cid}
        effect.result = {"deleting": cid, "runs": rids}
        api_v2.delete_library_clip(cid, db=db)  # one commit, then rmtree
        return {"deleted": True, "dir_removed": not os.path.exists(clip_dir), "runs": rids}

    def reconcile(self, db, effect):
        return "launched"


class LocalisationListenFeedback(ActionHandler):
    """Record a rejected final's comment and the corrected prompts (no domain write).

    It consumes the rejected listening answer at once (the core stops an
    autopilot that waits on a rejected answer); the regeneration it proposes
    is asked afterwards (C1's probe feedback, for several segments)."""

    name = "localisation_listen_feedback"
    version = 1
    paid = False

    def validate(self, db, project, params):
        unknown = set(params) - {"comment", "round", "segments", "named", "note"}
        if unknown:
            raise ActionInvalid(f"{self.name}: unknown params {sorted(unknown)}")
        segs = params.get("segments")
        if not isinstance(segs, list) or not segs:
            raise ActionInvalid(f"{self.name}: segments are required")
        out = []
        for s in segs:
            if not isinstance(s, Mapping) or not isinstance(s.get("run_segment_id"), str) \
                    or not isinstance(s.get("prompt"), str):
                raise ActionInvalid(f"{self.name}: each segment needs run_segment_id and prompt")
            out.append({"run_segment_id": s["run_segment_id"], "prompt": s["prompt"]})
        return {"comment": str(params.get("comment") or ""), "round": int(params.get("round") or 1),
                "segments": out, "named": [int(n) for n in params.get("named") or []],
                "note": str(params.get("note") or "")}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        effect.result = dict(params)
        return dict(params)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

HANDLERS = (LocalisationRederivePlacement(), LocalisationCreateCaption(), LocalisationCaptionLines(),
            LocalisationScriptCaption(), LocalisationCreateTail(), LocalisationDeleteTempClip(),
            LocalisationListenFeedback())
for _h in HANDLERS:
    registry.register_handler(_h)
registry.register_compute("localisation_segment_qa", lqa.segment_qa, version=1)
registry.register_compute("localisation_deliverable_check", lqa.deliverable_check, version=1)

__all__ = ["HANDLERS", "caption_placement_kwargs", "frozen_caption_placement", "caption_sig",
           "is_temp_clip", "TEMP_CLIP_PREFIX", "caption_placement_state", "Any"]
