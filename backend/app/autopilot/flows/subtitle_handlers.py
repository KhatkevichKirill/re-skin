"""Action handlers of the subtitle-removal flow (AICP-364, docs/autopilot-subtitle-flow.md §3).

Each follows the core's ``execute`` contract (contract.ActionHandler): every
domain write in the admission transaction, one commit after every check,
nothing written on a refusal path, created ids in ``effect.created_refs``
inside that transaction, never ``rollback()`` (rule g), enqueue only after the
commit. Refusals are the endpoints' own ``HTTPException`` 4xx (the core maps
them) or :class:`ActionRefused`.

* ``segment_fill_preview`` / ``segment_fill_cancel`` -- free, the AICP-358
  endpoints as they are (one commit, refusals before it).
* ``segment_fill_start`` -- paid, ALWAYS a regeneration (operator decision
  Q1): ``pipeline_segment_fill.admit_start``, the single-commit service split
  out of ``start_fill``.
* ``segment_fill_apply_source`` -- free: the source clip as is for a
  provider-refused segment whose source shows no caption.
* ``retry_run`` (paid, regeneration) / ``retry_delivery`` (free) --
  ``api_v2.retry_run`` on a terminal run only, re-checked under the run lock
  the core holds; the delivery retry refuses whatever would bill.
* ``restitch_run`` -- free: re-stitch + in-place re-delivery of a done run
  (every clip cut to its slot), refused whenever it would bill or drop work.
* ``patch_caption_detector`` -- free, ``api_v2.update_project`` with the
  caption detector only, fenced on the config the flow saw.
"""

from __future__ import annotations

import math
from typing import Mapping

from fastapi import HTTPException
from sqlalchemy import event, select

from ... import spend as spend_mod
from ...models import Run, RunSegment, VideoProject
from .. import registry
from ..contract import ActionHandler, ActionInvalid, ActionRefused
from . import subtitle_removal_plan as plan


def _run_of(db, project, run_id) -> Run:
    if not isinstance(run_id, str):
        raise ActionInvalid("run_id is required")
    run = db.get(Run, run_id)
    if run is None or run.project_id != project.id:
        raise ActionInvalid("the run is not a run of this project")
    return run


def _status(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


class SegmentFillPreview(ActionHandler):
    """``api_segment_fill.create_fill_preview`` -- free: freeze the base, queue
    the prepare (mask + VOID inputs + preview sheet)."""

    name = plan.H_FILL_PREVIEW
    version = 1
    paid = False

    def validate(self, db, project, params):
        run = _run_of(db, project, params.get("run_id"))
        rsid = params.get("run_segment_id")
        rs = db.get(RunSegment, rsid) if isinstance(rsid, str) else None
        if rs is None or rs.run_id != run.id:
            raise ActionInvalid("segment_fill_preview: run_segment_id is not a segment of the run")
        box = params.get("mask_box")
        if box is not None:
            from ...pipeline_segment_fill import FillError, validate_box

            try:
                box = validate_box(box)
            except FillError as exc:
                raise ActionInvalid(f"segment_fill_preview: {exc.message}") from None
        prompt = params.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ActionInvalid("segment_fill_preview: prompt must be a string or null")
        out = {"run_id": run.id, "run_segment_id": rs.id, "mask_box": box, "prompt": prompt}
        # The stop rule's grown retry (2026-11-03): only present when set, so
        # every older preview's params hash stays what it was.
        grow = params.get("mask_grow_px")
        if grow is not None:
            from ...pipeline_segment_fill import MAX_GROW_PX

            if not isinstance(grow, int) or isinstance(grow, bool) \
                    or not 0 <= grow <= MAX_GROW_PX:
                raise ActionInvalid(f"segment_fill_preview: mask_grow_px must be 0..{MAX_GROW_PX}")
            out["mask_grow_px"] = grow
        if params.get("retry_of") is not None:
            if not isinstance(params["retry_of"], str):
                raise ActionInvalid("segment_fill_preview: retry_of must be a fill id")
            out["retry_of"] = params["retry_of"]
        basis = params.get("basis")
        if basis not in (None, "model_result", "source"):
            raise ActionInvalid("segment_fill_preview: basis must be model_result or source")
        if basis == "source":
            # Only then: every params hash of a model-result preview stays what
            # it was before the basis existed.
            out["basis"] = "source"
        return out

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]], "run_segments": [params["run_segment_id"]]}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        import uuid

        from ... import api_segment_fill as asf
        from ...segment_fill_models import RunSegmentFillJob

        def _capture(session, _ctx, _instances):
            jobs = [o for o in session.new if isinstance(o, RunSegmentFillJob)]
            for o in jobs:
                if o.id is None:
                    o.id = str(uuid.uuid4())
            if jobs:
                refs = dict(effect.created_refs or {})
                refs["fill_jobs"] = sorted(set(refs.get("fill_jobs", [])) | {o.id for o in jobs})
                effect.created_refs = refs

        event.listen(db, "before_flush", _capture)
        try:
            out = asf.create_fill_preview(
                params["run_id"], params["run_segment_id"],
                asf.FillPreviewRequest(mask_box=params.get("mask_box"),
                                       prompt=params.get("prompt"),
                                       basis=params.get("basis") or "model_result",
                                       mask_grow_px=params.get("mask_grow_px"),
                                       retry_of=params.get("retry_of")), db=db)
        finally:
            event.remove(db, "before_flush", _capture)
        return {"fill_id": out["id"], "status": out["status"]}

    def reconcile(self, db, effect):
        return "launched" if (effect.created_refs or {}).get("fill_jobs") else "unknown"


class SegmentFillCancel(ActionHandler):
    """``api_segment_fill.cancel_fill`` -- free: drop a (stale) ready preview."""

    name = plan.H_FILL_CANCEL
    version = 1
    paid = False

    def validate(self, db, project, params):
        run = _run_of(db, project, params.get("run_id"))
        if not isinstance(params.get("fill_id"), str):
            raise ActionInvalid("segment_fill_cancel: fill_id is required")
        return {"run_id": run.id, "fill_id": params["fill_id"]}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]], "fill_jobs": [params["fill_id"]]}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import api_segment_fill as asf

        out = asf.cancel_fill(params["run_id"], params["fill_id"], db=db)
        return {"fill_id": out["id"], "status": out["status"]}


class SegmentFillStart(ActionHandler):
    """The paid VOID start of a ready preview (``psf.admit_start``)."""

    name = plan.H_FILL_START
    version = 1
    paid = True
    always_regeneration = True          # Q1: every VOID fill is approved on its own

    def validate(self, db, project, params):
        from ...segment_fill_models import RunSegmentFillJob

        run = _run_of(db, project, params.get("run_id"))
        fid = params.get("fill_id")
        job = db.get(RunSegmentFillJob, fid) if isinstance(fid, str) else None
        if job is None or job.run_id != run.id:
            raise ActionInvalid("segment_fill_start: fill_id is not a fill of the run")
        usd = params.get("approved_usd")
        if not isinstance(usd, (int, float)) or isinstance(usd, bool) \
                or not math.isfinite(float(usd)) or float(usd) < 0:
            raise ActionInvalid("segment_fill_start: approved_usd must be a finite amount >= 0")
        return {"run_id": run.id, "fill_id": job.id, "approved_usd": round(float(usd), 4),
                "discard_audio_patch": bool(params.get("discard_audio_patch", False))}

    def targets(self, db, project, params):
        from ...segment_fill_models import RunSegmentFillJob

        job = db.get(RunSegmentFillJob, params["fill_id"])
        return {"runs": [params["run_id"]], "run_segments": [job.run_segment_id],
                "fill_jobs": [job.id]}

    def estimate(self, db, project, params):
        """What ``pipeline_provider.admit`` reserves for this request: the VOID
        registry worst case over the job's clip, in credits."""
        from ... import fal_models
        from ...config import settings
        from ...pipeline_segment_fill import VOID_RESOLUTION
        from ...segment_fill_models import RunSegmentFillJob

        job = db.get(RunSegmentFillJob, params["fill_id"])
        dur = float((job.media or {}).get("duration_sec") or 0.0) if job is not None else 0.0
        if settings.KIE_USD_PER_CREDIT <= 0:
            return 0.0
        usd = fal_models.endpoint(fal_models.VOID_ENDPOINT).worst_case_usd(VOID_RESOLUTION,
                                                                          max(dur, 0.1))
        if job is not None and job.estimated_usd is not None:
            usd = max(usd, float(job.estimated_usd))
        return float(usd) / settings.KIE_USD_PER_CREDIT

    def spend_scopes(self, params, targets):
        return []                       # the fal:<task> scope is known only in execute

    def execute(self, db, project, params, effect):
        from ... import pipeline_segment_fill as psf
        from ..spend_view import register_scopes

        run = db.get(Run, params["run_id"])
        try:
            job, task, created = psf.admit_start(
                db, run, params["fill_id"], approved_usd=params["approved_usd"],
                discard_audio_patch=params["discard_audio_patch"])
        except psf.StalePreview as exc:
            # Nothing was written. The flow cancels the stale preview with the
            # free handler next tick and makes a new one.
            raise ActionRefused(409, str(exc)) from None
        except HTTPException as exc:
            if exc.status_code >= 500:
                # No live fal worker (503): nothing admitted -- a refusal with
                # backoff, not an error.
                raise ActionRefused(exc.status_code, str(exc.detail)) from None
            raise
        effect.created_refs = {"fill_jobs": [job.id], "provider_tasks": [task.id]}
        register_scopes(db, project.id, [spend_mod.provider_task_scope(task.id)],
                        autopilot_id=getattr(effect, "autopilot_id", None), source="admission")
        db.commit()
        if created:
            from ...tasks import enqueue_fal_job

            try:
                enqueue_fal_job("fal_submit", task.id)
            except Exception:  # the Fal ticker recovers a queued task
                import logging

                logging.getLogger(__name__).exception(
                    "segment_fill_start: enqueue of fal_submit for %s failed", task.id)
        return {"fill_id": job.id, "provider_task_id": task.id, "created": bool(created)}

    def reconcile(self, db, effect):
        return "launched" if (effect.created_refs or {}).get("provider_tasks") else "unknown"


class SegmentFillBatchStart(ActionHandler):
    """ONE paid start over several ready previews of a run (2026-11-03,
    docs/segment-fill-batch-design.md): ``psf.admit_batch_start``. NOT
    ``always_regeneration``: inside the run's VOID allowance the flow launches
    it with ``regeneration=False`` and authorization ``auto`` (checked again
    in the admission under the run lock, R1-5); above it the flow asks its
    ``vbatch:`` question and launches it with ``regeneration=True``, the
    approval's ``start`` option authorising exactly these params. The core's
    budget rule applies either way (``overrun`` past the approved budget)."""

    name = plan.H_FILL_BATCH_START
    version = 1
    paid = True
    always_regeneration = False

    def validate(self, db, project, params):
        from ...segment_fill_models import RunSegmentFillJob

        run = _run_of(db, project, params.get("run_id"))
        fids = params.get("fill_ids")
        if not isinstance(fids, list) or not fids or not all(isinstance(f, str) for f in fids):
            raise ActionInvalid("segment_fill_batch_start: fill_ids must be a non-empty list")
        if len(set(fids)) != len(fids):
            raise ActionInvalid("segment_fill_batch_start: fill_ids repeat")
        for fid in fids:
            job = db.get(RunSegmentFillJob, fid)
            if job is None or job.run_id != run.id:
                raise ActionInvalid(f"segment_fill_batch_start: {fid} is not a fill of the run")
        usd = params.get("approved_usd")
        if not isinstance(usd, (int, float)) or isinstance(usd, bool) \
                or not math.isfinite(float(usd)) or float(usd) < 0:
            raise ActionInvalid("segment_fill_batch_start: approved_usd must be a finite "
                                "amount >= 0")
        auth = params.get("authorization", "operator")
        if auth not in ("operator", "auto"):
            raise ActionInvalid("segment_fill_batch_start: authorization is operator or auto")
        return {"run_id": run.id, "fill_ids": sorted(fids),
                "approved_usd": round(float(usd), 4),
                "discard_audio_patch": bool(params.get("discard_audio_patch", False)),
                "authorization": auth}

    def targets(self, db, project, params):
        from ...segment_fill_models import RunSegmentFillJob

        rsids = []
        for fid in params["fill_ids"]:
            job = db.get(RunSegmentFillJob, fid)
            if job is not None:
                rsids.append(job.run_segment_id)
        return {"runs": [params["run_id"]], "run_segments": sorted(rsids),
                "fill_jobs": list(params["fill_ids"])}

    def estimate(self, db, project, params):
        """What the admission reserves: every member's own estimate (the
        registry worst case, flat for VOID), in credits (R1-6)."""
        total = sum(SegmentFillStart().estimate(db, project, {"fill_id": fid})
                    for fid in params["fill_ids"])
        return float(total)

    def spend_scopes(self, params, targets):
        return []                       # one fal:<task> scope per member, known in execute

    def execute(self, db, project, params, effect):
        from ... import pipeline_segment_fill as psf
        from ..spend_view import register_scopes

        run = db.get(Run, params["run_id"])
        try:
            batch, members = psf.admit_batch_start(
                db, run, params["fill_ids"], approved_usd=params["approved_usd"],
                discard_audio_patch=params["discard_audio_patch"],
                authorization=params["authorization"])
        except psf.StalePreview as exc:
            raise ActionRefused(409, str(exc)) from None
        except psf.AllowanceExceeded as exc:
            raise ActionRefused(409, str(exc)) from None
        except HTTPException as exc:
            if exc.status_code >= 500:
                raise ActionRefused(exc.status_code, str(exc.detail)) from None
            raise
        effect.created_refs = {"fill_batches": [batch.id],
                               "fill_jobs": [j.id for j, _t, _c in members],
                               "provider_tasks": [t.id for _j, t, _c in members]}
        register_scopes(db, project.id, [spend_mod.provider_task_scope(t.id)
                                         for _j, t, _c in members],
                        autopilot_id=getattr(effect, "autopilot_id", None), source="admission")
        db.commit()
        from ...tasks import enqueue_fal_job

        for _j, task, created in members:
            if not created:
                continue
            try:
                enqueue_fal_job("fal_submit", task.id)
            except Exception:  # the Fal ticker recovers a queued task
                import logging

                logging.getLogger(__name__).exception(
                    "segment_fill_batch_start: enqueue of fal_submit for %s failed", task.id)
        return {"batch_id": batch.id, "fill_ids": [j.id for j, _t, _c in members],
                "reserved_usd": batch.reserved_usd}

    def reconcile(self, db, effect):
        return "launched" if (effect.created_refs or {}).get("fill_batches") else "unknown"


class SegmentFillApplySource(ActionHandler):
    """FREE: ship the source clip as is for a provider-refused segment whose
    source preview found no caption (``psf.apply_source_as_is``). Asked first
    (``blocked:<fid>``, option ``source``), then the QA eye check after the
    delivery. The fill id is written to ``created_refs`` BEFORE the service's
    single commit, so a crash after it reconciles as launched."""

    name = plan.H_FILL_APPLY_SOURCE
    version = 1
    paid = False

    def validate(self, db, project, params):
        from ...segment_fill_models import RunSegmentFillJob

        run = _run_of(db, project, params.get("run_id"))
        fid = params.get("fill_id")
        job = db.get(RunSegmentFillJob, fid) if isinstance(fid, str) else None
        if job is None or job.run_id != run.id:
            raise ActionInvalid("segment_fill_apply_source: fill_id is not a fill of the run")
        return {"run_id": run.id, "fill_id": job.id}

    def targets(self, db, project, params):
        from ...segment_fill_models import RunSegmentFillJob

        job = db.get(RunSegmentFillJob, params["fill_id"])
        return {"runs": [params["run_id"]], "run_segments": [job.run_segment_id],
                "fill_jobs": [job.id]}

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import pipeline_segment_fill as psf

        run = db.get(Run, params["run_id"])
        if effect is not None:
            effect.created_refs = {"fill_jobs": [params["fill_id"]]}
        try:
            job = psf.apply_source_as_is(db, run, params["fill_id"])
        except psf.StalePreview as exc:
            raise ActionRefused(409, str(exc)) from None
        return {"fill_id": job.id, "status": job.status}

    def reconcile(self, db, effect):
        from ...segment_fill_models import RunSegmentFillJob

        fids = (effect.created_refs or {}).get("fill_jobs") or []
        job = db.get(RunSegmentFillJob, fids[0]) if fids else None
        return "launched" if job is not None and job.status == "applied" else "unknown"


_TERMINAL_RETRYABLE = ("failed", "incomplete")


def _retry_guard(db, run: Run) -> None:
    """Under the run lock the core holds: a terminal run only (never an orphan
    in an active status: the reaper owns those), and never the preprocess
    branch (it reserves GPT Image tasks this handler's estimate leaves out)."""
    from ...pipeline_v2 import preprocess_hole_index

    db.refresh(run)
    st = _status(run.status)
    if st not in _TERMINAL_RETRYABLE:
        raise ActionRefused(409, f"retry: the run is {st}, not failed/incomplete")
    if st == "failed" and preprocess_hole_index(run) is not None \
            and not any(rs.seedance_task_id for rs in run.run_segments):
        raise ActionRefused(409, "retry: the run would re-enter preprocessing (GPT Image): "
                                 "not an autopilot retry")


class RetryRun(ActionHandler):
    """``api_v2.retry_run`` of a failed / incomplete run -- paid regeneration."""

    name = plan.H_RETRY_RUN
    version = 1
    paid = True
    always_regeneration = True

    def validate(self, db, project, params):
        return {"run_id": _run_of(db, project, params.get("run_id")).id}

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def estimate(self, db, project, params):
        run = db.get(Run, params["run_id"])
        return float(spend_mod.estimate_run_credits(db, run)) if run is not None else 0.0

    def execute(self, db, project, params, effect):
        from ... import api_v2

        run = db.get(Run, params["run_id"])
        _retry_guard(db, run)
        resp = api_v2.retry_run(params["run_id"], db=db)
        return {"run_id": resp.id, "status": _status(resp.status)}


class RetryDelivery(RetryRun):
    """The same endpoint as a FREE stitch + delivery retry: refused whenever
    the retry would bill (a segment without a finished result)."""

    name = plan.H_RETRY_DELIVERY
    version = 1
    paid = False
    always_regeneration = False

    def estimate(self, db, project, params):
        return 0.0

    def execute(self, db, project, params, effect):
        from ... import api_v2

        run = db.get(Run, params["run_id"])
        _retry_guard(db, run)
        if _status(run.status) != "failed":
            raise ActionRefused(409, "retry_delivery: only a failed run is retried for free")
        need = float(spend_mod.estimate_run_credits(db, run))
        if need > 0:
            raise ActionRefused(409, f"retry_delivery: the retry would bill ~{need:.0f} credits")
        from ...pipeline_v2 import preprocess_hole_index

        if getattr(run, "preprocess_enabled", False) and preprocess_hole_index(run) is not None:
            # submit cannot resolve a reference set with a hole (gate R2): never
            # a free delivery retry, whatever task ids the segments carry.
            raise ActionRefused(409, "retry_delivery: the run has a GPT Image preprocessing hole")
        resp = api_v2.retry_run(params["run_id"], db=db)
        return {"run_id": resp.id, "status": _status(resp.status)}


class RestitchRun(ActionHandler):
    """A FREE re-stitch + in-place re-delivery of a ``done`` run (2026-11-02):
    the stitch now cuts each clip to its segment's slot
    (``pipeline_v2.stitch_frame_counts``), so a final stitched before that --
    every model clip a few frames long, every later segment late against the
    source sound -- comes back on the timeline without generating anything.

    Under the run lock the core holds (and ``_admit_segment_rerun`` takes
    again): a ``done`` run on the default timeline with the source soundtrack,
    every active swap segment finished, nothing billable (the estimate is 0,
    no preprocessing hole), no fill / range attempt / audio patch / GPT Image
    job in flight, and no promoted audio patch as the current result (the
    re-stitch would drop it). Then the run's final is invalidated and the run
    requeued (``_invalidate_final_and_requeue``: one commit, enqueue after it;
    the stitch replaces the bytes behind the run's Drive id)."""

    name = plan.H_RESTITCH
    version = 1
    paid = False
    always_regeneration = False

    def validate(self, db, project, params):
        out = {"run_id": _run_of(db, project, params.get("run_id")).id}
        if params.get("discard_audio_patch"):
            out["discard_audio_patch"] = True       # the operator's explicit choice
        return out

    def targets(self, db, project, params):
        return {"runs": [params["run_id"]]}

    def estimate(self, db, project, params):
        return 0.0

    def execute(self, db, project, params, effect):
        from ... import api_v2
        from ...pipeline_v2 import preprocess_hole_index

        run = db.get(Run, params["run_id"])
        api_v2._admit_segment_rerun(run, db)          # run lock + reconstruction / library
        st = _status(run.status)
        if st != "done":
            raise ActionRefused(409, f"restitch: the run is {st}, not done")
        if run.stitch_plan is not None:
            raise ActionRefused(409, "restitch: the run has a custom stitch plan")
        if (run.audio_mode or "original") != "original":
            raise ActionRefused(409, "restitch: the run does not use the source soundtrack")
        from ...pipeline_v2 import run_segment_defs, stitch_frame_counts

        from ...media import stitch_fps

        fps = stitch_fps(float(getattr(run.project, "fps", None) or 30.0))
        if stitch_frame_counts(run_segment_defs(run, run.project), fps, plan=None,
                               audio_mode="original") is None:
            # Rolled segments are stitched as they are: a re-stitch would
            # re-deliver the same timeline (gate R1-2).
            raise ActionRefused(409, "restitch: a segment has pre/post-roll; the stitch "
                                     "cannot cut its clips to their slots")
        api_v2._check_segment_rerun_conflicts(run.id, db)
        api_v2._require_audio_patch_discard_ack(run, db,
                                                bool(params.get("discard_audio_patch")))
        if getattr(run, "preprocess_enabled", False) and preprocess_hole_index(run) is not None:
            raise ActionRefused(409, "restitch: the run has a GPT Image preprocessing hole")
        need = float(spend_mod.estimate_run_credits(db, run, include_preprocess=False))
        if need > 0:
            raise ActionRefused(409, f"restitch: it would bill ~{need:.0f} credits "
                                     "(a segment without a finished result)")
        api_v2._invalidate_final_and_requeue(run, db, kind="restitch")
        return {"run_id": run.id, "status": _status(run.status)}

    def reconcile(self, db, effect):
        # One commit: the effect row exists only when the requeue landed.
        return "launched"


class PatchCaptionDetector(ActionHandler):
    """``api_v2.update_project`` with ``caption_detector`` only (free)."""

    name = plan.H_PATCH_DETECTOR
    version = 1
    paid = False

    def validate(self, db, project, params):
        from ...qa_tools import QAToolError, caption_config

        cfg = params.get("caption_detector")
        if not isinstance(cfg, Mapping):
            raise ActionInvalid("patch_caption_detector: caption_detector must be an object")
        try:
            caption_config(dict(cfg))
        except QAToolError as exc:
            raise ActionInvalid(f"patch_caption_detector: {exc}") from None
        sha = params.get("expect_detector_sha")
        if not isinstance(sha, str):
            raise ActionInvalid("patch_caption_detector: expect_detector_sha is required")
        out = {"caption_detector": dict(cfg), "expect_detector_sha": sha}
        if params.get("expect_no_band") is not None:
            if not isinstance(params["expect_no_band"], bool):
                raise ActionInvalid("patch_caption_detector: expect_no_band must be a boolean")
            out["expect_no_band"] = params["expect_no_band"]
        return out

    def spend_scopes(self, params, targets):
        return []

    def execute(self, db, project, params, effect):
        from ... import api_v2
        from ...schemas_v2 import ProjectUpdate

        # The core holds the project row (lock 1); update_project takes it too.
        current = db.execute(select(VideoProject.caption_detector).where(
            VideoProject.id == project.id)).scalar_one_or_none()
        if plan.detector_sha(current) != params["expect_detector_sha"]:
            raise ActionRefused(409, "patch_caption_detector: the detector changed since the "
                                     "review")
        if params.get("expect_no_band"):
            from ...qa_tools import has_band

            # A located band never replaces a band of the project's own.
            if has_band(current):
                raise ActionRefused(409, "patch_caption_detector: the project has its own "
                                         "caption band now")
        api_v2.update_project(project.id, ProjectUpdate(
            caption_detector=dict(params["caption_detector"])), db=db)
        return {"caption_detector": dict(params["caption_detector"])}


HANDLERS = (SegmentFillPreview(), SegmentFillCancel(), SegmentFillStart(),
            SegmentFillBatchStart(), SegmentFillApplySource(), RetryRun(), RetryDelivery(), RestitchRun(),
            PatchCaptionDetector())
for _h in HANDLERS:
    registry.register_handler(_h)
