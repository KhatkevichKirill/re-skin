"""
pipeline_segment_image.py — per-segment GPT Image 2 regeneration.

One :class:`~app.models.RunSegmentImageJob` is one attempt at replacing the
generated reference image ONE swap segment sends to Seedance/Gemini Omni,
before the operator re-runs that segment. It mirrors
:mod:`app.overlay_pipeline` / :mod:`app.pipeline_tail` in shape (one entity,
a worker entry point, a strict try/except that records a failure) but the
create-task/poll-download cycle instead mirrors
``pipeline_v2.preprocess_reference_images`` / ``poll_preprocess_once``, since
this is the same kie.ai GPT Image 2 endpoint applied to a single ref pair.

Two-phase, like the preprocess chain, so the worker never sleeps:

    submit_segment_image_job(job_id)      creates (or resumes) the kie.ai task
    poll_segment_image_job_once(job_id)   one round of status-check; the
                                           caller (app/tasks.py) re-enqueues
                                           itself while the result is pending

Deliberately NOT under the run's Redis processing lock (``tasks.run_lock_key``):
serialisation is per-SEGMENT, enforced by ``api_v2``'s 409 gate (only one
active job per RunSegment at a time) rather than by taking the run-wide lock.
Taking that lock here would be worse than the race it prevents: a concurrent
``run_submit_run`` would find it held and silently skip (see
``tasks.py``'s lock-acquire logging), parking the run in ``queued`` with no
worker — see docs/gpt-image-2-preprocess.md's "per-segment regeneration"
section.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

from . import spend as spend_mod
from .knowledge import capture as kcap
from .db import get_session
from .gdrive_client import GDriveClient
from .kie_client import KieClient, get_shared_client
from .models import Run, RunSegment, RunSegmentImageJob, VideoProject
from .pipeline_common import extract_project_frame, resolve_reference_urls
from .pipeline_v2 import _image_ext_from_url, _parse_cost_credits, _parse_result_url
from .state_machine import SegmentImageStatus, transition
from .storage import segment_image_job_dir

log = logging.getLogger(__name__)

# Cap how much of an exception message is persisted to error_message — same
# convention as RunTailEdit's (pipeline_tail._ERROR_MESSAGE_MAX_LEN).
_ERROR_MESSAGE_MAX_LEN = 2000


def _truncate_error(exc: Exception) -> str:
    """Render *exc* as a bounded-length string safe to store in error_message."""
    msg = str(exc)
    if len(msg) > _ERROR_MESSAGE_MAX_LEN:
        msg = msg[:_ERROR_MESSAGE_MAX_LEN] + "... (truncated)"
    return msg


def _default_gdrive() -> GDriveClient:
    return GDriveClient()


def submit_segment_image_job(
    job_id: str,
    *,
    kie: Optional[KieClient] = None,
    gdrive: Optional[GDriveClient] = None,
) -> Optional[str]:
    """Create (or resume) this attempt's kie.ai GPT Image 2 task.

    Resume-aware exactly like ``pipeline_v2.preprocess_reference_images``: if
    ``job.kie_task_id`` is already set — a prior invocation got this far
    before its worker died, or this call is a duplicate — the existing task
    is returned for polling rather than recreated. A retry must never rebill
    kie.ai for the same attempt (the rule the preprocess phase follows for
    run-level tasks; NOT ``OverlayJob``'s "just run it again").

    Returns the kie.ai task id to poll, or ``None`` when the job is already
    terminal (``done``/``failed``) — nothing left to do.
    """
    if kie is None:
        kie = get_shared_client()

    with get_session() as session:
        job: Optional[RunSegmentImageJob] = session.get(RunSegmentImageJob, job_id)
        if job is None:
            raise ValueError(f"RunSegmentImageJob not found: {job_id}")

        if job.status in (SegmentImageStatus.done, SegmentImageStatus.failed):
            log.info(
                "submit_segment_image_job: job %s already terminal (status=%s) "
                "— nothing to do", job_id, job.status,
            )
            return None

        if job.kie_task_id:
            log.info(
                "submit_segment_image_job: job %s has existing task_id=%s — "
                "resuming without recreating", job_id, job.kie_task_id,
            )
            if job.status == SegmentImageStatus.queued:
                transition(job, SegmentImageStatus.generating)
                session.commit()
            return job.kie_task_id

        # Everything below can fail (missing rows, ffmpeg, network, kie.ai).
        # Unlike the poll phase's task-level failures (handled explicitly
        # below), any exception here must still land the job on `failed` with
        # a reason — otherwise it stays `queued` forever with no error
        # message, and the 409 gates in api_v2 (patch_run_segment,
        # rerun_segment, regenerate_segment_image, retry_run) stay closed
        # until the recovery reaper's 300s grace window expires. Mirrors
        # overlay_pipeline.process_overlay_job / pipeline_tail.process_tail_edit.
        try:
            rs: Optional[RunSegment] = session.get(RunSegment, job.run_segment_id)
            run: Optional[Run] = session.get(Run, job.run_id)
            if rs is None or run is None:
                raise ValueError(
                    f"submit_segment_image_job: RunSegment/Run missing for job {job_id}"
                )
            project: Optional[VideoProject] = session.get(VideoProject, run.project_id)
            if project is None:
                raise ValueError(
                    f"submit_segment_image_job: project missing for run {run.id}"
                )

            character = list(job.input_paths or [])[:1]
            if not character or not character[0]:
                raise ValueError(f"job {job_id} has no character photo recorded")

            # Extract the source frame at the recorded timestamp and copy it
            # into THIS job's own directory — never left pointing at the
            # shared project frame cache, which a later "Add frame" or
            # project re-analysis can evict/replace out from under a retry.
            cache_path, t_norm = extract_project_frame(project, job.frame_sec)
            job_dir = segment_image_job_dir(job.id, rs.id, run.id, run.project_id)
            ext = os.path.splitext(cache_path)[1] or ".jpg"
            frame_path = os.path.join(job_dir, f"frame{ext}")
            shutil.copy2(cache_path, frame_path)

            job.input_paths = character + [frame_path]
            job.frame_sec = t_norm
            session.commit()

            input_urls = resolve_reference_urls(list(job.input_paths), kie, gdrive=gdrive)
            # Knowledge capture A6 (AICP-299): single call; never raises.
            kb_attempt = kcap.image_intent(
                run_id=job.run_id, rs_id=job.run_segment_id, target_kind="segment_image_job", target_id=job.id,
                attempt_kind="segment_image", origin="operator", prompt=job.prompt, input_urls=input_urls,
                locators=list(job.input_paths), recipe_context={"frame_sec": job.frame_sec}, site="A6",
            )
            try:
                task_id = kie.create_image_task(prompt=job.prompt, input_urls=input_urls)
            except Exception as exc:
                kcap.dispatch_error(kb_attempt, exc, site="A6")
                raise
            # Measured before anything else can fail (AICP-155).
            spend_mod.record_draw(
                spend_mod.segment_image_scope(job_id), spend_mod.image_credits(),
                kind="segment image", call_key=task_id, sessions=get_session,
            )
            job.kie_task_id = task_id
            transition(job, SegmentImageStatus.generating)
            session.commit()
            kcap.submitted(kb_attempt, task_id, site="A6")  # after the commit (P1)
            log.info(
                "submit_segment_image_job: job %s (segment %s) → task %s",
                job_id, job.run_segment_id, task_id,
            )
            return task_id
        except Exception as exc:
            log.exception("submit_segment_image_job failed: job_id=%s", job_id)
            job.error_message = _truncate_error(exc)
            if job.status not in (SegmentImageStatus.done, SegmentImageStatus.failed):
                try:
                    transition(job, SegmentImageStatus.failed)
                except Exception:
                    job.status = SegmentImageStatus.failed
            try:
                session.commit()
            except Exception:
                pass
            raise


def poll_segment_image_job_once(
    job_id: str, *, kie: Optional[KieClient] = None,
) -> str:
    """One round of status-check for a single job's kie.ai task.

    On success: downloads the output image into the job's own directory,
    composes ``RunSegment.generated_reference_urls = [character photo,
    output]`` and stamps ``generated_image_job_id`` — this attempt is
    auto-applied the moment it succeeds, exactly what a later manual re-Apply
    of an older attempt also does — and marks the job ``done``. On failure
    (task failed, no result url, or a download error) the job is marked
    ``failed`` with a readable ``error_message``; ``RunSegment`` is left
    untouched, so a segment that already had a generated image keeps it.
    Otherwise the job stays ``generating`` and the caller re-enqueues this
    function.

    Returns one of ``"pending"``, ``"done"``, ``"failed"``.
    """
    if kie is None:
        kie = get_shared_client()

    with get_session() as session:
        job: Optional[RunSegmentImageJob] = session.get(RunSegmentImageJob, job_id)
        if job is None:
            raise ValueError(f"RunSegmentImageJob not found: {job_id}")

        if job.status in (SegmentImageStatus.done, SegmentImageStatus.failed):
            return job.status.value

        if not job.kie_task_id:
            raise ValueError(f"job {job_id} has no kie_task_id to poll")

        try:
            data = kie.get_task(job.kie_task_id)
        except Exception as exc:
            log.warning(
                "poll_segment_image_job_once: job %s get_task failed: %s — "
                "will retry next round", job_id, exc,
            )
            return "pending"

        state = (data.get("state") or "").lower()
        if state in ("success", "fail"):
            spend_mod.settle_draw(
                job.kie_task_id, _parse_cost_credits(data),
                scope_key=spend_mod.segment_image_scope(job_id),
                kind="segment image", sessions=get_session,
            )

        if state == "fail":
            reason = data.get("failMsg") or "unknown error"
            job.error_message = f"GPT Image task failed: {reason}"
            transition(job, SegmentImageStatus.failed)
            session.commit()
            kcap.terminal(task_id=job.kie_task_id, status="failed", error_class=str(reason),
                          payload={"failMsg": str(reason)[:500]}, site="T6")
            log.warning(
                "poll_segment_image_job_once: job %s task %s FAILED (failMsg=%s)",
                job_id, job.kie_task_id, reason,
            )
            return "failed"

        if state != "success":
            # waiting / queuing / generating / unknown — keep polling.
            return "pending"

        url = _parse_result_url(data)
        if not url:
            job.error_message = "GPT Image task succeeded but returned no result URL"
            transition(job, SegmentImageStatus.failed)
            session.commit()
            kcap.terminal(task_id=job.kie_task_id, status="failed", error_class="no_result_url", site="T6")
            log.warning(
                "poll_segment_image_job_once: job %s success but no resultUrl "
                "— marking failed", job_id,
            )
            return "failed"

        rs: Optional[RunSegment] = session.get(RunSegment, job.run_segment_id)
        run: Optional[Run] = session.get(Run, job.run_id)
        if rs is None or run is None:
            job.error_message = "RunSegment/Run was deleted while this job was running"
            transition(job, SegmentImageStatus.failed)
            session.commit()
            return "failed"

        job_dir = segment_image_job_dir(job.id, rs.id, run.id, run.project_id)
        ext = _image_ext_from_url(url)
        dst = os.path.join(job_dir, f"result.{ext}")
        try:
            kie.download_result(url, dst)
        except Exception as exc:
            job.error_message = f"Failed to download GPT Image result: {exc}"
            transition(job, SegmentImageStatus.failed)
            session.commit()
            kcap.terminal(task_id=job.kie_task_id, status="failed", error_class="download_failed",
                          payload={"stage": "download"}, site="T6")
            log.warning(
                "poll_segment_image_job_once: job %s download failed: %s",
                job_id, exc,
            )
            return "failed"

        job.output_path = dst
        job.cost_credits = _parse_cost_credits(data)
        job.error_message = None
        transition(job, SegmentImageStatus.done)

        # Auto-apply: the whole point of regenerating is that the NEXT Re-run
        # of this segment uses the new image, without a separate click. A
        # later manual Apply of a different (older) attempt overwrites this.
        # AICP-363 §4.3: a generation input of the run changes only under its
        # row lock (fences the autopilot's `run:` version keys).
        _run_row = session.get(Run, rs.run_id)
        if _run_row is not None:
            session.refresh(_run_row, with_for_update=True)
        character_photo = (job.input_paths or [None])[0]
        rs.generated_reference_urls = [character_photo, dst]
        rs.generated_image_job_id = job.id

        session.commit()
        # Knowledge capture B5 (AICP-299), after the commit; never raises.
        kcap.output(task_id=job.kie_task_id, path=dst, run_id=run.id, rs_id=rs.id, artifact_kind="reference_image",
                    attempt_kind="segment_image", origin="operator", target_kind="segment_image_job",
                    target_id=job.id, site="B5")
        log.info(
            "poll_segment_image_job_once: job %s done → %s (auto-applied to "
            "segment %s)", job_id, dst, job.run_segment_id,
        )
        return "done"
