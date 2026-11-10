"""
RQ job bodies of the reframe workflow (AICP-251, docs/reframe-design.md §5-§7).

reframe_prepare(job_id)
    Claim the job (CAS, lease = RQ timeout + 300 s, at most 3 attempts), cut
    the window from the project source in ONE ffmpeg demux into claim-scoped
    staging (the paid input + the pinned audio snapshot), then in one
    transaction under the project lock: CAS the job ``preparing`` ->
    ``admitted`` on this claim, publish the staged files, and admit the
    provider task (``pipeline_provider.admit``, in this transaction). After
    the commit, ``fal_submit`` is enqueued only when the task was created.
    A refused admission or an exceeded budget rolls everything back and ends
    the job ``prepare_failed`` in a second, claim-checked transaction.

reframe_qa(job_id)
    The job-level checks J1-J4 on the provider's raw output, a claimed step
    taken under the project lock; renders require ``qa_status = passed``.

reframe_render(render_id)
    A free ffmpeg finish of the raw output (ending + optional CTA), checks
    R1-R4, then Drive delivery. ``queued`` -> ``processing`` -> ``delivering``
    -> ``done`` (or ``failed``); only a ``queued`` row is ever picked up.

Lock order everywhere: video_projects -> reframe_jobs -> reframe_renders ->
the layer's task row -> the spend day row.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import or_, select, update

from . import media as media_mod
from . import pipeline_provider as pp
from . import reframe as rf
from . import spend, storage
from .config import settings
from .db import get_session
from .models import VideoProject
from .provider_models import ProviderTask
from .reframe_models import ReframeJob, ReframeRender
from .state_machine import ProviderTaskStatus

log = logging.getLogger(__name__)

# Patched by tests (same seam as pipeline_provider._sessions).
_sessions = get_session

MAX_PREP_ATTEMPTS = 3
_ERROR_MAX = 2000

DELIVER_ATTEMPTS = int(os.getenv("REFRAME_DELIVER_ATTEMPTS", "3"))
DELIVER_BACKOFF_SEC = float(os.getenv("REFRAME_DELIVER_BACKOFF_SEC", "10"))

# AdmissionRefused status -> prepare_failed error_code (design §5).
_REFUSAL_CODES = {403: "fal_disabled", 503: "fal_worker_down", 422: "admission_invalid",
                  404: "admission_conflict", 409: "admission_conflict"}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def prep_lease_sec() -> int:
    return settings.REFRAME_PREP_JOB_TIMEOUT + 300


def qa_lease_sec() -> int:
    return settings.REFRAME_QA_JOB_TIMEOUT + 300


def _trunc(msg: str) -> str:
    return msg if len(msg) <= _ERROR_MAX else msg[:_ERROR_MAX] + "... (truncated)"


def _remove(path: Optional[str]) -> None:
    if path:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("reframe: could not remove %s: %s", path, exc)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lock_project(s, project_id: str) -> Optional[VideoProject]:
    return s.execute(
        select(VideoProject).where(VideoProject.id == project_id).with_for_update()
    ).scalar_one_or_none()


def enqueue_fal_submit(task_id: str) -> None:
    """After the admission commit, only for a created task. A failure here
    loses nothing: the layer's recovery resubmits a `queued` row after
    FAL_QUEUED_GRACE_SEC."""
    from .tasks import enqueue_fal_job

    try:
        enqueue_fal_job("fal_submit", task_id)
    except Exception:
        log.exception("reframe: enqueue of fal_submit for task %s failed; the layer's "
                      "recovery resubmits it", task_id)


# ---------------------------------------------------------------------------
# Prepare + admit
# ---------------------------------------------------------------------------


def _fail_prepare(job_id: str, token: str, code: str, message: str,
                  published: tuple[Optional[str], ...] = ()) -> bool:
    """Claim-checked end of a preparation, under the project lock. Deletes
    *published* files only when this execution still owned the job."""
    with _sessions() as s:
        job = s.get(ReframeJob, job_id)
        if job is None:
            return False
        _lock_project(s, job.project_id)
        n = s.execute(
            update(ReframeJob)
            .where(ReframeJob.id == job_id, ReframeJob.status == "preparing",
                   ReframeJob.prep_claim_owner == token)
            .values(status="prepare_failed", error_code=code, error_message=_trunc(message),
                    prep_claim_owner=None, prep_claim_expires_at=None, updated_at=_now())
        ).rowcount
        s.commit()
    if n == 1:
        for p in published:
            _remove(p)
    log.warning("reframe: job %s prepare_failed(%s): %s (applied=%s)", job_id, code, message, n == 1)
    return n == 1


def reframe_prepare(job_id: str) -> None:
    token = uuid.uuid4().hex
    now = _now()
    with _sessions() as s:
        claimed = s.execute(
            update(ReframeJob)
            .where(ReframeJob.id == job_id, ReframeJob.status == "preparing",
                   or_(ReframeJob.prep_claim_owner.is_(None),
                       ReframeJob.prep_claim_expires_at < now),
                   ReframeJob.prep_attempts < MAX_PREP_ATTEMPTS)
            .values(prep_claim_owner=token,
                    prep_claim_expires_at=now + timedelta(seconds=prep_lease_sec()),
                    prep_attempts=ReframeJob.prep_attempts + 1, updated_at=now)
        ).rowcount
        s.commit()
        if claimed != 1:
            log.info("reframe_prepare: job %s not claimable (owned, finished or gone)", job_id)
            return
        job = s.get(ReframeJob, job_id)
        project = s.get(VideoProject, job.project_id)
        pid = job.project_id
        src = project.source_local_path if project else None
        src_fps = (project.fps if project else None) or 30.0
        w_start, w_end = job.window_start_sec, job.window_end_sec
        aspect, resolution = job.target_aspect, job.resolution

    jdir = storage.reframe_job_dir(job_id, pid)
    staging = os.path.join(jdir, f"prep-{token}")
    try:
        os.mkdir(staging)  # never makedirs: a deleted job stays deleted
    except FileNotFoundError:
        log.info("reframe_prepare: job dir of %s is gone; exiting", job_id)
        return
    try:
        _prepare_and_admit(job_id, token, pid, src, src_fps, w_start, w_end,
                           aspect, resolution, jdir, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _prepare_and_admit(job_id, token, pid, src, src_fps, w_start, w_end,
                       aspect, resolution, jdir, staging) -> None:
    if not src or not os.path.isfile(src):
        _fail_prepare(job_id, token, "source_missing", "the project's source file is missing")
        return
    try:
        info = media_mod.probe(src)
        input_start = rf.first_frame_at_or_after(src, w_start)
        if input_start >= w_end - rf.MIN_WINDOW_SEC / 2:
            raise rf.ReframeError("no video frame early enough inside the window")
        video_sec = w_end - input_start
        audio_end = min(info.duration_sec, w_end + settings.REFRAME_AUDIO_RUNON_MAX_SEC)
        audio_sec = max(0.0, audio_end - input_start)
        cap = settings.FAL_MAX_INPUT_BYTES
        staged_in = os.path.join(staging, "input.mp4")
        staged_audio = os.path.join(staging, "audio.flac") if info.has_audio else None
        out_fps = min(src_fps or 30.0, 30.0)

        r = rf.input_bitrate(video_sec, cap)
        media_mod._run(rf.prepare_argv(
            src=src, input_start=input_start, video_sec=video_sec, audio_sec=audio_sec,
            fps=out_fps, video_dst=staged_in, audio_dst=staged_audio, bitrate=r,
            has_audio=info.has_audio))
        first_size = os.path.getsize(staged_in)
        if first_size > cap:
            media_mod._run(rf.prepare_argv(
                src=src, input_start=input_start, video_sec=video_sec, audio_sec=None,
                fps=out_fps, video_dst=staged_in, audio_dst=None,
                bitrate=int(r * 0.6), short_side=768, has_audio=info.has_audio))
            if os.path.getsize(staged_in) > cap:
                _fail_prepare(job_id, token, "input_too_large",
                              f"the prepared input is {first_size} bytes, then "
                              f"{os.path.getsize(staged_in)} after the fallback; the "
                              f"limit is {cap}")
                return
        in_info = media_mod.probe(staged_in)
        frames = rf.count_video_frames(staged_in)
        if abs(in_info.duration_sec - video_sec) > 1.0 / out_fps + 1e-3:
            _fail_prepare(job_id, token, "input_duration",
                          f"the prepared input is {in_info.duration_sec:.3f} s, "
                          f"expected {video_sec:.3f} s")
            return
        audio_dur = None
        if staged_audio:
            _, audio_dur = media_mod.probe_audio(staged_audio)
        in_sha, in_bytes = _sha256(staged_in), os.path.getsize(staged_in)
    except (media_mod.MediaError, rf.ReframeError) as exc:
        _fail_prepare(job_id, token, "prepare_error", str(exc))
        return

    input_path = os.path.join(jdir, "input.mp4")
    audio_path = os.path.join(jdir, "audio.flac") if staged_audio else None
    dest = os.path.join(jdir, "raw.mp4")
    base_usd, _ = rf.estimate_usd(resolution, in_info.duration_sec)

    task_id: Optional[str] = None
    created = False
    refusal: Optional[tuple[str, str]] = None
    with _sessions() as s:
        if _lock_project(s, pid) is None:
            s.rollback()
            return
        n = s.execute(
            update(ReframeJob)
            .where(ReframeJob.id == job_id, ReframeJob.status == "preparing",
                   ReframeJob.prep_claim_owner == token,
                   ReframeJob.prep_claim_expires_at > _now())
            .values(status="admitted", input_start_sec=input_start, input_frames=frames,
                    input_duration_sec=in_info.duration_sec, input_path=input_path,
                    input_sha256=in_sha, input_bytes=in_bytes, audio_path=audio_path,
                    audio_start_sec=input_start if audio_path else None,
                    audio_duration_sec=audio_dur, source_has_audio=bool(audio_path),
                    estimated_usd_base=base_usd,
                    prep_claim_owner=None, prep_claim_expires_at=None, updated_at=_now())
        ).rowcount
        if n != 1:
            s.rollback()
            log.info("reframe_prepare: job %s lost its claim before admission", job_id)
            return
        s.flush()
        # Publish under the lock: only the claim holder ever writes these paths.
        os.replace(staged_in, input_path)
        if staged_audio:
            os.replace(staged_audio, audio_path)
        try:
            task, created = pp.admit(
                s, endpoint_id=rf.ENDPOINT_ID,
                params={"aspect_ratio": aspect, "resolution": resolution},
                inputs=[pp.TaskInput("video_url", input_path, "video/mp4")],
                output_sec=in_info.duration_sec, resolution=resolution,
                owner_kind="reframe_job", owner_id=job_id, owner_slot="main",
                owner_project_id=pid, destination_path=dest,
                operation_key=f"reframe_job:{job_id}:main:1",
            )
        except pp.AdmissionRefused as exc:
            s.rollback()
            refusal = (_REFUSAL_CODES.get(exc.status_code, "admission_conflict"),
                       f"Fal admission refused ({exc.status_code}): {exc.detail}")
        except spend.BudgetExceeded as exc:
            s.rollback()
            refusal = ("budget", f"daily budget: {exc}")
        else:
            task_id = task.id
            s.execute(update(ReframeJob).where(ReframeJob.id == job_id)
                      .values(provider_task_id=task_id))
            s.commit()
    if refusal is not None:
        _fail_prepare(job_id, token, refusal[0], refusal[1], (input_path, audio_path))
        return
    log.info("reframe_prepare: job %s admitted as task %s (created=%s)", job_id, task_id, created)
    if created and task_id:
        enqueue_fal_submit(task_id)


# ---------------------------------------------------------------------------
# Job QA (J1-J4)
# ---------------------------------------------------------------------------


def reframe_qa(job_id: str) -> None:
    token = uuid.uuid4().hex
    with _sessions() as s:
        job = s.get(ReframeJob, job_id)
        if job is None or job.status != "admitted" or not job.provider_task_id:
            return
        pid = job.project_id
        _lock_project(s, pid)
        task = s.get(ProviderTask, job.provider_task_id)
        if task is None or task.status != ProviderTaskStatus.done.value:
            s.rollback()
            return
        now = _now()
        n = s.execute(
            update(ReframeJob)
            .where(ReframeJob.id == job_id, ReframeJob.status == "admitted",
                   or_(ReframeJob.qa_status == "pending",
                       (ReframeJob.qa_status == "running")
                       & (ReframeJob.qa_claim_expires_at < now)))
            .values(qa_status="running", qa_claim_owner=token,
                    qa_claim_expires_at=now + timedelta(seconds=qa_lease_sec()),
                    updated_at=now)
        ).rowcount
        s.commit()
        if n != 1:
            return
        s.refresh(job)
        raw = task.result_path or task.destination_path
        meta = dict(task.result_meta or {})
        inp, in_dur = job.input_path, job.input_duration_sec or 0.0
        aspect, res = job.target_aspect, job.resolution

    jdir = storage.reframe_job_dir(job_id, pid)
    tmp_sheet = os.path.join(jdir, f"sheet-{token}.jpg")
    qa: dict = {"checked_at": _now().isoformat()}
    passed = True
    try:
        if not raw or not os.path.isfile(raw):
            raise media_mod.MediaError("the raw output is missing on disk")
        info = media_mod.probe(raw)
        raw_frames = rf.count_video_frames(raw)
        qa["raw_frames"] = raw_frames
        qa["raw_duration"] = info.duration_sec
        exp_w, exp_h = rf.output_size(aspect, res)
        qa["J1"] = {"width": info.width, "height": info.height, "expected": [exp_w, exp_h],
                    "ok": (info.width, info.height) == (exp_w, exp_h), "outcome": "warn"}
        expected_frames = round(in_dur * rf.OUTPUT_FPS)
        j2_ok = abs(raw_frames - expected_frames) <= 1
        qa["J2"] = {"raw_frames": raw_frames, "expected": expected_frames, "ok": j2_ok,
                    "outcome": "fail"}
        passed = passed and j2_ok
        rect = rf.parse_placement(meta)
        if rect is None:
            qa["J3"] = {"skipped": "Fal's result carried no usable placement", "ok": None}
        else:
            values = []
            for frac in (0.25, 0.5, 0.75):
                v = rf.ssim_at(raw, inp, info.duration_sec * frac, rect)
                if v is not None:
                    values.append(round(v, 4))
            qa["J3"] = {"placement": list(rect), "ssim": values,
                        "ok": bool(values) and min(values) >= rf.J3_SSIM_MIN, "outcome": "warn"}
        if not os.path.isdir(jdir):
            return
        rf.contact_sheet(raw, raw_frames, tmp_sheet)
        qa["J4"] = {"ok": True}
        for k in ("seed", "frame_count", "frames_per_second"):
            if k in meta:
                qa.setdefault("fal", {})[k] = meta[k]
    except media_mod.MediaError as exc:
        passed = False
        qa["error"] = _trunc(str(exc))
    final = "passed" if passed else "failed"
    with _sessions() as s:
        n = s.execute(
            update(ReframeJob)
            .where(ReframeJob.id == job_id, ReframeJob.qa_status == "running",
                   ReframeJob.qa_claim_owner == token, ReframeJob.qa_claim_expires_at > _now())
            .values(qa_status=final, job_qa=qa, qa_claim_owner=None,
                    qa_claim_expires_at=None, updated_at=_now())
        ).rowcount
        s.commit()
    if n == 1 and os.path.exists(tmp_sheet):
        os.replace(tmp_sheet, os.path.join(jdir, "sheet.jpg"))
    else:
        _remove(tmp_sheet)
    log.info("reframe_qa: job %s -> %s (applied=%s)", job_id, final, n == 1)


# ---------------------------------------------------------------------------
# Render (R1-R4 + delivery)
# ---------------------------------------------------------------------------


def job_times(job: ReframeJob) -> rf.JobTimes:
    has_audio = bool(job.source_has_audio and job.audio_path)
    return rf.JobTimes(
        input_start=job.input_start_sec or 0.0, window_end=job.window_end_sec,
        raw_frames=int((job.job_qa or {}).get("raw_frames") or 0), has_audio=has_audio,
        audio_end_max=(job.audio_start_sec or 0.0) + (job.audio_duration_sec or 0.0),
    )


def output_filename(job_id: str, render_id: str) -> str:
    return f"reframe_{job_id[:8]}_{render_id[:8]}.mp4"


def _set_render(render_id: str, expect: str, token: Optional[str], **values) -> bool:
    """CAS on (status, claim_owner): only the execution that owns the render
    moves it on (a superseded worker changes zero rows)."""
    with _sessions() as s:
        n = s.execute(update(ReframeRender)
                      .where(ReframeRender.id == render_id, ReframeRender.status == expect,
                             ReframeRender.claim_owner == token)
                      .values(updated_at=_now(), **values)).rowcount
        s.commit()
    return n == 1


def _deliver_with_retry(gdrive, path: str, name: str, render_id: str) -> dict:
    last: Optional[Exception] = None
    for attempt in range(1, DELIVER_ATTEMPTS + 1):
        try:
            return gdrive.upload_file(path, settings.GDRIVE_DEFAULT_FOLDER_ID or None, name)
        except Exception as exc:
            last = exc
            log.warning("reframe render %s: Drive upload %d/%d failed: %s",
                        render_id, attempt, DELIVER_ATTEMPTS, exc)
            if attempt < DELIVER_ATTEMPTS:
                time.sleep(DELIVER_BACKOFF_SEC * attempt)
    raise last  # type: ignore[misc]


def reframe_render(render_id: str, *, gdrive=None) -> None:
    token = uuid.uuid4().hex
    with _sessions() as s:
        n = s.execute(update(ReframeRender)
                      .where(ReframeRender.id == render_id, ReframeRender.status == "queued")
                      .values(status="processing", claim_owner=token, error_message=None,
                              updated_at=_now())).rowcount
        s.commit()
    if n != 1:
        log.info("reframe_render: render %s is not queued; nothing to do", render_id)
        return
    tmp_final: Optional[str] = None
    try:
        with _sessions() as s:
            r = s.get(ReframeRender, render_id)
            job = s.get(ReframeJob, r.job_id)
            task = s.get(ProviderTask, job.provider_task_id) if job.provider_task_id else None
            if task is None or task.status != ProviderTaskStatus.done.value:
                raise rf.ReframeError("the generation is not done")
            raw = task.result_path or task.destination_path
            times = job_times(job)
            ending = rf.Ending(r.picture_end_sec, r.fade_sec, r.audio_end_sec, r.black_tail_sec)
            frames = rf.validate_ending(times, ending)
            rdir = storage.reframe_render_dir(r.id, job.id, job.project_id)
            audio = job.audio_path if times.has_audio else None
            cta = r.cta_path
            w, h = rf.output_size(job.target_aspect, job.resolution)
            job_id = job.id
        qa = _render_files(raw, audio, cta, frames, rdir, w, h, token)
        tmp_final = qa.pop("_final")
    except (rf.ReframeError, media_mod.MediaError, OSError) as exc:
        _set_render(render_id, "processing", token, status="failed",
                    error_message=_trunc(str(exc)))
        log.warning("reframe_render: render %s failed: %s", render_id, exc)
        return
    if not qa["ok"]:
        _remove(tmp_final)
        _set_render(render_id, "processing", token, status="failed", qa=qa,
                    error_message=_trunc("QA failed: " + ", ".join(qa["failed"])))
        return
    # Each execution keeps its own token-named final; the CAS records the
    # winner's path, so a superseded worker can never overwrite the file the
    # row points at (it can at most leave its own file behind in the dir).
    final = tmp_final
    if not _set_render(render_id, "processing", token, status="delivering", qa=qa,
                       result_local_path=final, result_bytes=os.path.getsize(final),
                       result_sha256=_sha256(final),
                       result_duration_sec=qa["final"]["duration"],
                       result_width=w, result_height=h):
        _remove(final)
        log.info("reframe_render: render %s was superseded; output discarded", render_id)
        return
    try:
        if gdrive is None:
            from .gdrive_client import GDriveClient

            gdrive = GDriveClient()
        res = _deliver_with_retry(gdrive, final, output_filename(job_id, render_id), render_id)
    except Exception as exc:
        _set_render(render_id, "delivering", token, status="failed",
                    error_message=_trunc(f"Drive delivery failed: {exc}"))
        return
    _set_render(render_id, "delivering", token, status="done", gdrive_file_id=res.get("id"),
                gdrive_link=res.get("webViewLink"))
    log.info("reframe_render: render %s done (%s)", render_id, res.get("id"))


def _render_files(raw: str, audio: Optional[str], cta: Optional[str], f: rf.Frames,
                  rdir: str, w: int, h: int, token: str) -> dict:
    """Body, then the final (body + CTA), into token-scoped files; R1-R4 on
    both (design §6). Returns the QA dict with the final's path in
    ``_final``; the caller records it only after winning its CAS."""
    if not os.path.isdir(rdir):
        raise rf.ReframeError("the render directory is gone (deleted)")
    if not raw or not os.path.isfile(raw):
        raise rf.ReframeError("the raw output is missing on disk")
    body = os.path.join(rdir, f"body-{token}.mp4")
    final = os.path.join(rdir, f"final-{token}.mp4")
    try:
        media_mod._run(rf.render_body_argv(raw=raw, audio=audio, f=f, dst=body,
                                           width=w, height=h))
        failed: list[str] = []
        qa: dict = {"frames": {"NP": f.np, "NF": f.nf, "NA": f.na, "NT": f.nt, "NL": f.nl}}
        bf = rf.media_facts(body)
        qa["body"] = bf
        if not (bf["frames"] == f.nl and (bf["width"], bf["height"]) == (w, h)):
            failed.append("R1 body")
        r2 = rf.audio_check(body, audio, f.na, f.nl)
        qa["R2"] = r2
        if not r2["ok"]:
            failed.append("R2 body")
        tol = 1.0 / rf.OUTPUT_FPS + 1e-3
        if abs(bf["audio_duration"] - bf["frames"] / rf.OUTPUT_FPS) > tol:
            failed.append("R3 body")
        if cta:
            if not os.path.isfile(cta):
                raise rf.ReframeError("the CTA snapshot is missing")
            cta_info = media_mod.probe(cta)
            media_mod.stitch([body, cta], audio_source=body, dst=final, width=w, height=h,
                             fps=float(rf.OUTPUT_FPS), audio_mode="seedance")
            _remove(body)
            expected = f.nl + round(cta_info.duration_sec * rf.OUTPUT_FPS)
        else:
            os.replace(body, final)
            expected = f.nl
        try:
            media_mod.enforce_size_limit(final)
        except media_mod.MediaError as exc:
            log.warning("reframe render: could not shrink %s: %s", final, exc)
        ff = rf.media_facts(final)
        qa["final"] = ff
        if abs(ff["frames"] - expected) > 1 or (ff["width"], ff["height"]) != (w, h):
            failed.append("R1 final")
        r2f = rf.audio_check(final, audio, f.na, f.nl, final=bool(cta))
        qa["R2_final"] = r2f
        if not r2f["ok"]:
            failed.append("R2 final")
        if abs(ff["audio_duration"] - ff["frames"] / rf.OUTPUT_FPS) > tol:
            failed.append("R3 final")
        size = os.path.getsize(final)
        qa["R4"] = {"bytes": size, "ok": size <= media_mod._DELIVERY_MAX_BYTES}
        if not qa["R4"]["ok"]:
            failed.append("R4")
    except BaseException:
        _remove(body)
        _remove(final)
        raise
    finally:
        _remove(body)
    qa["failed"] = failed
    qa["ok"] = not failed
    qa["_final"] = final
    return qa


# ---------------------------------------------------------------------------
# Deletion guards (design §7) — called under the project lock
# ---------------------------------------------------------------------------


def job_busy(s, job: ReframeJob) -> Optional[str]:
    """Why *job* cannot be deleted now, or None. The caller holds the project
    lock, so a prepare admission or a QA claim cannot start meanwhile."""
    if job.status == "preparing":
        return "a generation is still being prepared"
    if job.provider_task_id:
        task = s.execute(
            select(ProviderTask).where(ProviderTask.id == job.provider_task_id,
                                       pp._ACTIVE)
        ).scalar_one_or_none()
        if task is not None:
            return ("its Fal task is still active (it may have been billed; close or "
                    "abandon it first)")
    if job.qa_status == "running" and job.qa_claim_expires_at and job.qa_claim_expires_at > _now():
        return "its QA is running"
    active = s.execute(
        select(ReframeRender.id).where(
            ReframeRender.job_id == job.id,
            ReframeRender.status.in_(("queued", "processing", "delivering")))
    ).first()
    if active is not None:
        return "one of its renders is still running"
    return None


def reframe_project_busy(s, project_id: str) -> Optional[str]:
    """Why a project with reframe jobs cannot be deleted now, or None."""
    for job in s.execute(
        select(ReframeJob).where(ReframeJob.project_id == project_id)
    ).scalars():
        why = job_busy(s, job)
        if why:
            return f"reframe job {job.id[:8]}: {why}"
    return None


def fail_crashed_prepare(s, job_id: str, project_id: str, now: datetime) -> bool:
    """The recovery reaper's only failure (design §7): under the project lock,
    re-decided against the row, so a live claim always survives and an
    admission that committed first wins (zero rows here). Commits."""
    _lock_project(s, project_id)
    n = s.execute(
        update(ReframeJob)
        .where(ReframeJob.id == job_id, ReframeJob.status == "preparing",
               ReframeJob.prep_attempts >= MAX_PREP_ATTEMPTS,
               or_(ReframeJob.prep_claim_owner.is_(None),
                   ReframeJob.prep_claim_expires_at < now))
        .values(status="prepare_failed", error_code="prepare_crashed",
                error_message=f"preparation crashed on every attempt ({MAX_PREP_ATTEMPTS}); "
                              "start a new job",
                prep_claim_owner=None, prep_claim_expires_at=None, updated_at=now)
    ).rowcount
    s.commit()
    return n == 1


def reap_orphaned_render(s, render_id: str, status: str, claim_owner: Optional[str],
                         updated_at: datetime, now: datetime) -> bool:
    """Recovery's failure of an orphaned render: a CAS on the status, claim
    and updated_at it observed (every status change, and Retry, sets
    updated_at), so a render that moved on meanwhile — finished, retried, or
    retried and claimed — is left alone. Commits."""
    n = s.execute(
        update(ReframeRender)
        .where(ReframeRender.id == render_id, ReframeRender.status == status,
               ReframeRender.updated_at == updated_at,
               (ReframeRender.claim_owner == claim_owner) if claim_owner is not None
               else ReframeRender.claim_owner.is_(None))
        .values(status="failed", claim_owner=None, updated_at=now,
                error_message="Interrupted by a worker restart before it finished; nothing "
                              "new was delivered by this attempt. Press Retry.")
    ).rowcount
    s.commit()
    return n == 1
