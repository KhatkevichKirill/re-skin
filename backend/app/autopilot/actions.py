"""Action handlers shipped by the core (docs/autopilot-design.md §4.4).

Each one is a thin wrapper around the existing endpoint function, called with
EVERY parameter explicitly (an omitted FastAPI ``Form(...)`` default would
arrive as a ``FieldInfo``). The endpoint's single commit is the admission
commit: it persists the core's pending effect row together with the domain
writes (§5.4). The core maps an ``HTTPException`` 4xx to ``ActionRefused``.

Not shipped here, with the reason (§4.4): ``retry_run`` (accepts non-terminal
runs), ``segment_fill_start`` (commits before a 409), ``transcribe_project``
and ``localisation_prompt`` (no durable dispatch), ``to_project``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from sqlalchemy import event, select

from .. import spend as spend_mod
from ..models import Run, RunSegment, VideoProject
from . import registry
from .contract import ActionHandler, ActionInvalid, ActionRefused


def _str_list(value, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ActionInvalid(f"{field} must be a list of strings")
    return list(value)


class CreateRun(ActionHandler):
    """``api_v2.create_run`` — a new run of the project (paid)."""

    name = "create_run"
    version = 1
    paid = True

    _FIELDS = {"prompt", "name", "model", "resolution", "audio_mode", "language",
               "reference_urls", "segment_prompts", "stitch_plan", "voice_clip_id",
               "voice_from_library_clip"}

    def validate(self, db, project, params):
        unknown = set(params) - self._FIELDS
        if unknown:
            raise ActionInvalid(f"create_run: unknown params {sorted(unknown)}")
        prompt = params.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ActionInvalid("create_run: prompt is required")
        out = {
            "prompt": prompt,
            "name": params.get("name"),
            "model": params.get("model") or "seedance",
            "resolution": params.get("resolution") or "720p",
            "audio_mode": params.get("audio_mode") or "original",
            "language": params.get("language"),
            "reference_urls": _str_list(params.get("reference_urls"), "reference_urls"),
            "segment_prompts": dict(params.get("segment_prompts") or {}),
            "stitch_plan": params.get("stitch_plan"),
            "voice_clip_id": params.get("voice_clip_id"),
            # AICP-365: a stitch-library clip (e.g. the accepted probe's TEMP
            # clip) whose audio becomes the run's voice reference — uploaded
            # to create_run exactly like the form's voice_file.
            "voice_from_library_clip": params.get("voice_from_library_clip"),
        }
        if out["stitch_plan"] is not None and not isinstance(out["stitch_plan"], list):
            raise ActionInvalid("create_run: stitch_plan must be a list or null")
        if out["voice_clip_id"] and out["voice_from_library_clip"]:
            raise ActionInvalid("create_run: voice_clip_id and voice_from_library_clip exclude each other")
        return out

    def estimate(self, db, project, params):
        from .. import project_types
        from ..pipeline_v2 import _active_swap_segment_ids

        if project.project_type == project_types.SUBTITLE_REMOVAL and \
                project_types.caption_removal_method(project) == \
                project_types.CAPTION_REMOVAL_VOID_SOURCE:
            # A VOID-from-the-source run reserves nothing at creation: its
            # fills reserve on their own provider tasks (design §7.2, R2-9).
            return 0.0
        defs = list(project.segments)
        active = _active_swap_segment_ids(params.get("stitch_plan"), defs)
        return float(sum(
            spend_mod.video_credits(params["model"], params["resolution"], spend_mod._clip_sec(sd))
            for sd in defs if sd.action == "swap" and sd.id in active
        ))

    def spend_scopes(self, params, targets):
        return []  # the run id is only known inside execute (created_refs)

    def execute(self, db, project, params, effect):
        from .. import api_v2

        def _capture(session, _ctx, _instances):
            ids = sorted(
                o.id for o in session.new
                if isinstance(o, Run) and o.project_id == project.id
            )
            if ids:
                if any(i is None for i in ids):
                    raise RuntimeError("create_run: a new Run has no id at flush time")
                refs = dict(effect.created_refs or {})
                refs["runs"] = sorted(set(refs.get("runs", [])) | set(ids))
                effect.created_refs = refs
                from .spend_view import register_scopes

                register_scopes(session, project.id, [f"run:{i}" for i in ids],
                                autopilot_id=effect.autopilot_id, source="admission")

        voice_file, voice_fh = None, None
        clip_id = params.get("voice_from_library_clip")
        if clip_id:
            import os

            from starlette.datastructures import UploadFile

            from ..models import LibraryClip

            clip = db.get(LibraryClip, clip_id)
            if clip is None or not clip.local_path or not os.path.isfile(clip.local_path):
                raise ActionRefused(409, f"library clip {clip_id} is missing its file")
            voice_fh = open(clip.local_path, "rb")
            voice_file = UploadFile(file=voice_fh, filename=os.path.basename(clip.local_path))

        event.listen(db, "before_flush", _capture)
        try:
            resp = api_v2.create_run(
                project.id,
                prompt=params["prompt"],
                name=params.get("name"),
                model=params["model"],
                resolution=params["resolution"],
                audio_mode=params["audio_mode"],
                language=params.get("language"),
                gdrive_folder_id=None,
                reference_files=[],
                reference_urls=",".join(params.get("reference_urls") or []) or None,
                segment_prompts=(json.dumps(params["segment_prompts"])
                                 if params.get("segment_prompts") else None),
                stitch_plan=(json.dumps(params["stitch_plan"])
                             if params.get("stitch_plan") is not None else None),
                preprocess_enabled=False,
                preprocess_mode="per_image",
                preprocess_image_prompts=None,
                preprocess_reference_file=None,
                preprocess_reference_url=None,
                voice_clip_id=params.get("voice_clip_id"),
                voice_file=voice_file,
                db=db,
            )
        finally:
            event.remove(db, "before_flush", _capture)
            if voice_fh is not None:
                voice_fh.close()
        return {"run_id": resp.run_id, "status": resp.status, "warnings": list(resp.warnings or [])}

    def reconcile(self, db, effect):
        return "launched" if (effect.created_refs or {}).get("runs") else "unknown"


class RerunSegments(ActionHandler):
    """``api_v2.rerun_segments`` — paid, always a regeneration."""

    name = "rerun_segments"
    version = 1
    paid = True
    always_regeneration = True

    def validate(self, db, project, params):
        run_id = params.get("run_id")
        segs = params.get("segments")
        if not isinstance(run_id, str) or not isinstance(segs, list) or not segs:
            raise ActionInvalid("rerun_segments: run_id and a non-empty segments list required")
        out_segs = []
        for item in segs:
            if not isinstance(item, Mapping) or not isinstance(item.get("run_segment_id"), str):
                raise ActionInvalid("rerun_segments: each segment needs run_segment_id")
            p = item.get("prompt")
            if p is not None and not isinstance(p, str):
                raise ActionInvalid("rerun_segments: prompt must be a string or null")
            out_segs.append({"run_segment_id": item["run_segment_id"], "prompt": p})
        run = db.get(Run, run_id)
        if run is None or run.project_id != project.id:
            raise ActionInvalid("rerun_segments: run is not a run of this project")
        return {"run_id": run_id, "segments": out_segs,
                "discard_audio_patch": bool(params.get("discard_audio_patch", False))}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]],
                "run_segments": [s["run_segment_id"] for s in params["segments"]]}

    def estimate(self, db, project, params):
        from ..api_v2 import _segment_rerun_need

        run = db.get(Run, params["run_id"])
        if run is None:
            return 0.0
        ids = [s["run_segment_id"] for s in params["segments"]]
        segs = [rs for rs in db.execute(
            select(RunSegment).where(RunSegment.id.in_(ids))).scalars()]
        return float(_segment_rerun_need(db, run, segs))

    def execute(self, db, project, params, effect):
        from .. import api_v2
        from ..schemas_v2 import SegmentRerunItem, SegmentsRerunRequest

        from ..models import RunSegmentTake

        body = SegmentsRerunRequest(
            segments=[SegmentRerunItem(**s) for s in params["segments"]],
            discard_audio_patch=params["discard_audio_patch"],
        )

        def _capture(session, _ctx, _instances):  # AICP-364 R8: the takes it archived
            import uuid as _uuid

            takes = [o for o in session.new if isinstance(o, RunSegmentTake)
                     and o.run_id == params["run_id"]]
            for o in takes:
                if o.id is None:
                    o.id = str(_uuid.uuid4())  # the column default, assigned early
            ids = sorted(o.id for o in takes)
            if ids:
                refs = dict(effect.created_refs or {})
                refs["takes"] = sorted(set(refs.get("takes", [])) | set(ids))
                effect.created_refs = refs

        # A segment whose current result IS an existing take (restored earlier)
        # is not archived again: record that take too.
        seg_ids = [s["run_segment_id"] for s in params["segments"]]
        current = {rs.id: (rs.local_result_path, rs.seedance_task_id) for rs in db.execute(
            select(RunSegment).where(RunSegment.id.in_(seg_ids))).scalars()}

        def _is_current(t):
            path, task = current.get(t.run_segment_id, (None, None))
            return bool((t.result_path and t.result_path == path)
                        or (t.seedance_task_id and t.seedance_task_id == task))

        existing = sorted(t.id for t in db.execute(select(RunSegmentTake).where(
            RunSegmentTake.run_segment_id.in_(seg_ids))).scalars() if _is_current(t))
        if existing:
            refs = dict(effect.created_refs or {})
            refs["takes"] = sorted(set(refs.get("takes", [])) | set(existing))
            effect.created_refs = refs

        event.listen(db, "before_flush", _capture)
        try:
            resp = api_v2.rerun_segments(params["run_id"], body, db=db)
        finally:
            event.remove(db, "before_flush", _capture)
        return {"run_id": resp.id, "status": str(resp.status)}


class RestoreTakes(ActionHandler):
    """``api_v2.restore_segment_takes`` — free (the endpoint refuses when it would bill)."""

    name = "restore_takes"
    version = 1
    paid = False

    def validate(self, db, project, params):
        run_id = params.get("run_id")
        takes = _str_list(params.get("takes"), "takes")
        if not isinstance(run_id, str) or not takes:
            raise ActionInvalid("restore_takes: run_id and takes required")
        run = db.get(Run, run_id)
        if run is None or run.project_id != project.id:
            raise ActionInvalid("restore_takes: run is not a run of this project")
        return {"run_id": run_id, "takes": takes,
                "discard_audio_patch": bool(params.get("discard_audio_patch", False))}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def execute(self, db, project, params, effect):
        from .. import api_v2
        from ..schemas_v2 import SegmentTakesRestoreRequest

        body = SegmentTakesRestoreRequest(takes=params["takes"],
                                          discard_audio_patch=params["discard_audio_patch"])
        resp = api_v2.restore_segment_takes(params["run_id"], body, db=db)
        return {"run_id": resp.id, "status": str(resp.status)}


class PatchSegments(ActionHandler):
    """``api_v2.update_project_segments`` — free (its own active-run 409)."""

    name = "patch_segments"
    version = 1
    paid = False

    def validate(self, db, project, params):
        from ..schemas_v2 import SegmentsUpdateRequest

        try:
            body = SegmentsUpdateRequest.model_validate(dict(params))
        except Exception as exc:
            raise ActionInvalid(f"patch_segments: {exc}") from None
        return json.loads(body.model_dump_json())

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from .. import api_v2
        from ..schemas_v2 import SegmentsUpdateRequest

        out = api_v2.update_project_segments(
            project.id, SegmentsUpdateRequest.model_validate(params), db=db)
        return {"segments": len(out)}


for _h in (CreateRun(), RerunSegments(), RestoreTakes(), PatchSegments()):
    registry.register_handler(_h)
