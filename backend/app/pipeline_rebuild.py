"""The images-only (rebuild) run pipeline — design §6.2–6.6, §7.2–7.3.

An images-only run never enters the v2v submit/poll code. Its parts:

* :func:`submit_images_only` — the first entry of ``tasks.run_submit_run``
  for an images-only run (a launch, a Retry and a recovery hand-off alike):
  ``queued -> processing`` and one ``rebuild_chunk_attempts`` row per active
  chunk segment that is ``pending`` with no live attempt.
* :func:`observe_run` — the run's driver while attempts are in flight
  (``tasks.run_rebuild_observe``): pure observation of attempts; it hands
  over to the existing finalize once every active chunk is ``completed`` and
  marks the run ``incomplete`` once a chunk failed and nothing is live.
* :func:`chunk_attempt_step` — the ``rebuild_chunk_attempts`` step handler
  (``rebuild_paid_call.STEP_HANDLERS``): claim -> send (ONE
  ``create_task_once``) -> poll -> fetch (download + :func:`fit_attempt`).
  Never a second create for a row; ``failed`` / ``unknown`` are terminal and
  only an operator's rerun / Retry admits a NEW row.
* QA (§6.5, flags only, never blocking): :func:`run_attempt_qa` (cut timing,
  duration, burned text, contact sheet — free) and the automatic character
  check, a ``RebuildLlmCall(kind="character_check")`` driven by
  :func:`character_check_step` (registered in
  ``app/rebuild_llm_steps.py: KIND_HANDLERS``).
* Repair: :func:`reset_for_retry`, :func:`reset_for_rerun`,
  :func:`use_attempt` (called by the API in the reservation transaction).
* The frame-exact keep cut the finalize uses for an images-only run
  (:func:`cut_keep_frames`).

Every provider call goes out once through ``KieClient.create_task_once`` /
``chat_completion_once``; polling (``get_task``), uploads and downloads are
free and idempotent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from . import rebuild as rb
from . import rebuild_jobs as jobs
from . import rebuild_paid_call as pc
from . import spend as spend_mod
from .db import get_session
from .rebuild_models import (
    PAID_CALL_LIVE_STATUSES,
    PC_CLAIMED,
    PC_FAILED,
    PC_FETCHING,
    PC_QUEUED,
    PC_SENDING,
    PC_SUBMITTED,
    PC_SUCCEEDED,
    PC_UNKNOWN,
    RebuildChunkAttempt,
    RebuildLlmCall,
)

log = logging.getLogger(__name__)

TABLE = RebuildChunkAttempt.__tablename__
LLM_TABLE = RebuildLlmCall.__tablename__

# Give up polling a task this long after it was sent (the provider never
# answered). The draw stays at the estimate: the task may have been billed.
POLL_GIVE_UP_SEC = 6 * 3600
# Re-schedule of a fetch whose download failed (never a new create).
FETCH_RETRY_DELAY_SEC = 30
# QA job (free part) on the autopilot_compute queue, served by the run workers.
QA_JOB_TIMEOUT_SEC = 900
QA_JOB_PREFIX = "rbqa:"
# A succeeded attempt whose free QA is still missing after this long is
# re-enqueued by :func:`reap_qa` (the job was lost).
QA_REAP_GRACE_SEC = 600
# ... and only attempts that succeeded this recently are looked at.
QA_REAP_WINDOW_DAYS = 7
# Free QA runs before a missing contact sheet makes the check "unavailable".
FREE_QA_MAX_RUNS = 3
# Character check (design §6.5, Q9).
CHECK_HTTP_TIMEOUT_SEC = 180
CHECK_SLOTS = spend_mod.CHECK_AUTO_SLOTS
CHECK_UNAVAILABLE = "unavailable"
# Contact sheet cell (one row per planned shot: generated | keyframe).
SHEET_CELL_LONG = 360


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _kie():
    """The kie.ai client (module-level so tests patch one place)."""
    from .kie_client import get_shared_client

    return get_shared_client()


def _status(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def snapshot_chunks(run) -> list[dict]:
    return list((run.rebuild_snapshot or {}).get("chunks") or [])


def snapshot_chunk(run, segment_def_id: str) -> Optional[dict]:
    for ch in snapshot_chunks(run):
        if ch.get("segment_def_id") == segment_def_id:
            return ch
    return None


def active_chunk_ids(run, project) -> set:
    """Segment-def ids of the run's chunks the stitch plan actually uses."""
    from .pipeline_v2 import _active_swap_segment_ids, run_segment_defs

    snap_ids = {ch.get("segment_def_id") for ch in snapshot_chunks(run)}
    return snap_ids & _active_swap_segment_ids(run.stitch_plan, run_segment_defs(run, project))


def live_attempt_count(db, run_id: str) -> int:
    """Live chunk attempts of the run (character checks excluded)."""
    return int(db.execute(select(func.count()).select_from(RebuildChunkAttempt).where(
        RebuildChunkAttempt.run_id == run_id,
        RebuildChunkAttempt.status.in_(PAID_CALL_LIVE_STATUSES))).scalar() or 0)


def _fps_out(run) -> float:
    return float((run.rebuild_snapshot or {}).get("fps_out") or 30.0)


# ---------------------------------------------------------------------------
# Submit (first entry, also after every recovery) — design §6.4
# ---------------------------------------------------------------------------


def submit_images_only(run_id: str) -> bool:
    """``queued -> processing``, then admit one attempt per active chunk
    segment that is ``pending`` with no live attempt. Idempotent: the live
    index ``uq_rebuild_attempt_live`` makes a racing duplicate admission fail
    harmlessly. Returns False when the run is not drivable (missing,
    terminal, not images-only) — the caller releases the lock."""
    from .models import Run, RunSegment, VideoProject
    from .state_machine import RunStatus, SegmentStatus, transition

    admitted: list[str] = []
    with get_session() as s:
        run = s.get(Run, run_id, with_for_update=True)
        if run is None or run.generation_mode != "images_only":
            log.error("submit_images_only: run %s missing or not images-only", run_id)
            return False
        if run.status == RunStatus.queued:
            transition(run, RunStatus.processing)
            run.error_message = None
        elif run.status != RunStatus.processing:
            log.info("submit_images_only: run %s is %s — nothing to do", run_id, run.status)
            return False
        project = s.get(VideoProject, run.project_id)
        active = active_chunk_ids(run, project)
        rs_by_def = {rs.segment_def_id: rs for rs in s.execute(
            select(RunSegment).where(RunSegment.run_id == run.id)).scalars()}
        for ch in snapshot_chunks(run):
            sd_id = ch.get("segment_def_id")
            if sd_id in active and sd_id not in rs_by_def:
                # Repair only: launch commits the rows with the reservation.
                rs = RunSegment(run_id=run.id, segment_def_id=sd_id, index=int(ch["index"]),
                                status=SegmentStatus.pending)
                s.add(rs)
                rs_by_def[sd_id] = rs
        s.flush()
        live_segs = set(s.execute(select(RebuildChunkAttempt.run_segment_id).where(
            RebuildChunkAttempt.run_id == run.id,
            RebuildChunkAttempt.status.in_(PAID_CALL_LIVE_STATUSES))).scalars())
        for ch in snapshot_chunks(run):
            rs = rs_by_def.get(ch.get("segment_def_id"))
            if rs is None or ch.get("segment_def_id") not in active:
                continue
            if _status(rs.status) != SegmentStatus.pending.value or rs.id in live_segs:
                continue
            n = int(s.execute(select(func.max(RebuildChunkAttempt.attempt_no)).where(
                RebuildChunkAttempt.run_segment_id == rs.id)).scalar() or 0) + 1
            attempt = RebuildChunkAttempt(
                run_id=run.id, run_segment_id=rs.id, attempt_no=n,
                model=run.model, resolution=run.resolution,
                request_sec=int(ch["request_sec"]), chunk_frames=int(ch["chunk_frames"]),
                status=PC_QUEUED,
                estimate_credits=spend_mod.images_only_video_credits(
                    run.model, run.resolution, int(ch["request_sec"])),
            )
            sp = s.begin_nested()
            try:
                s.add(attempt)
                s.flush()
                sp.commit()
            except IntegrityError:
                sp.rollback()
                log.info("submit_images_only: chunk %s of run %s already has a live attempt",
                         ch.get("index"), run_id)
                continue
            rs.status = SegmentStatus.uploading
            rs.error_message = None
            admitted.append(attempt.id)
        s.commit()
    for aid in admitted:
        try:
            pc.enqueue_step(TABLE, aid, 1, pc.STEP_CLAIM)
        except Exception:  # the reaper re-sends a lost claim
            log.exception("submit_images_only: enqueue of attempt %s failed", aid)
    log.info("submit_images_only: run %s processing, %d attempt(s) admitted", run_id,
             len(admitted))
    return True


# ---------------------------------------------------------------------------
# Observer — design §6.4 ("run_rebuild_observe")
# ---------------------------------------------------------------------------

OBSERVE_FINALIZE = "finalize"
OBSERVE_INCOMPLETE = "incomplete"
OBSERVE_WAIT = "wait"
OBSERVE_STOP = "stop"


def observe_run(run_id: str) -> str:
    """One observation of a ``processing`` images-only run.

    Never admits an attempt and never touches a live one. Returns
    :data:`OBSERVE_FINALIZE` (every active chunk ``completed`` with its file),
    :data:`OBSERVE_INCOMPLETE` (the run was moved to ``incomplete``: a chunk
    failed and no attempt is live), :data:`OBSERVE_WAIT` or
    :data:`OBSERVE_STOP` (the run is not ``processing``).

    The one write besides ``incomplete``: a chunk whose attempt ended by the
    REAPER (``sending -> unknown`` on a sender proof, or a host attestation)
    never got its segment updated by a step, so a non-completed segment with
    no live attempt is marked ``failed`` here from its last attempt.
    """
    from .models import Run, RunSegment, VideoProject
    from .state_machine import RunStatus, SegmentStatus, transition

    with get_session() as s:
        run = s.get(Run, run_id, with_for_update=True)
        if run is None or run.status != RunStatus.processing:
            return OBSERVE_STOP
        project = s.get(VideoProject, run.project_id)
        active = active_chunk_ids(run, project)
        # Attempts FIRST, segments after (integration gate round 1): a fetch
        # commits its attempt `succeeded` and its segment `completed` in ONE
        # transaction, and each statement here reads its own snapshot. Read
        # the other way round, a fetch committing in between left an old
        # (non-completed) segment next to a terminal attempt, and the observer
        # overwrote the paid, usable result with `failed`. In this order a
        # terminal attempt implies the segment read after it is current.
        attempts: dict[str, list] = {}
        for a in s.execute(select(RebuildChunkAttempt).where(
                RebuildChunkAttempt.run_id == run.id)
                .order_by(RebuildChunkAttempt.attempt_no)).scalars():
            attempts.setdefault(a.run_segment_id, []).append(a)
        rows = {rs.segment_def_id: rs for rs in s.execute(
            select(RunSegment).where(RunSegment.run_id == run.id)
            .execution_options(populate_existing=True)).scalars()}
        done = live = 0
        failed: list[int] = []
        for ch in snapshot_chunks(run):
            if ch.get("segment_def_id") not in active:
                continue
            rs = rows.get(ch["segment_def_id"])
            if rs is None:
                failed.append(int(ch["index"]))
                continue
            mine = attempts.get(rs.id, [])
            if any(a.status in PAID_CALL_LIVE_STATUSES for a in mine):
                live += 1
                continue
            if (_status(rs.status) == SegmentStatus.completed.value and rs.local_result_path
                    and os.path.exists(rs.local_result_path)):
                done += 1
                continue
            if _status(rs.status) != SegmentStatus.failed.value:
                last = mine[-1] if mine else None
                was_completed = _status(rs.status) == SegmentStatus.completed.value
                rs.status = SegmentStatus.failed
                rs.error_message = (
                    (last.error if last is not None and last.error and not was_completed
                     else None)
                    or ("its result file is missing" if was_completed
                        else "no live attempt — re-run this chunk")
                )[:2000]
            failed.append(int(rs.index))
        total = done + live + len(failed)
        if live:
            s.commit()
            return OBSERVE_WAIT
        if not failed and done == total:
            s.commit()
            return OBSERVE_FINALIZE
        transition(run, RunStatus.incomplete)
        run.error_message = (
            f"{len(failed)} chunk(s) did not complete: {sorted(failed)}. Re-run them (or "
            "use an earlier attempt) — the video is stitched once every chunk succeeds."
        )
        s.commit()
        log.warning("observe_run: run %s incomplete — chunks %s", run_id, sorted(failed))
        return OBSERVE_INCOMPLETE


# ---------------------------------------------------------------------------
# Attempt steps — design §6.4, §7.2
# ---------------------------------------------------------------------------


def chunk_attempt_step(row_id: str, gen: int, step: str, job_id: Optional[str]):
    """``rebuild_paid_call.STEP_HANDLERS["rebuild_chunk_attempts"]``."""
    handler = {pc.STEP_CLAIM: _step_claim, pc.STEP_SEND: _step_send,
               pc.STEP_POLL: _step_poll, pc.STEP_FETCH: _step_fetch}.get(step)
    if handler is None:
        log.error("chunk_attempt_step: unknown step %r for %s", step, row_id)
        return None
    return handler(row_id, int(gen), job_id)


def _load(s, row_id: str):
    """``(attempt, run_segment, run)`` with fresh state, or Nones."""
    from .models import Run, RunSegment

    a = pc._get(s, RebuildChunkAttempt, row_id)
    if a is None:
        return None, None, None
    rs = s.get(RunSegment, a.run_segment_id, populate_existing=True)
    run = s.get(Run, a.run_id, populate_existing=True)
    return a, rs, run


def _segment_to(rs, status, *, error: Optional[str] = None) -> None:
    from .state_machine import SegmentStatus

    if rs is None:
        return
    rs.status = SegmentStatus(status) if isinstance(status, str) else status
    if error is not None:
        rs.error_message = error[:2000]


def _fail_unsent(s, a, rs, *, gen: int, token: Optional[str], error: str) -> bool:
    """``claimed -> failed`` for a deterministic input error found before the
    send (nothing left the process: no draw exists, nothing to settle)."""
    ok = jobs.fail_unsent(s, RebuildChunkAttempt, a.id, gen=gen, token=token, error=error)
    if ok:
        _segment_to(rs, "failed", error=error)
    return ok


def chunk_inputs(run, rs) -> tuple[dict, list[str], str]:
    """``(snapshot chunk, image paths, prompt)`` of a chunk's request: the
    run-owned keyframe copies then the sheet, the frozen prompt or the
    segment's ``prompt_override`` (a rerun with an edited prompt). Raises
    ``ValueError`` when the snapshot or a file is missing / altered."""
    from .rebuild_launch import file_sha256

    ch = snapshot_chunk(run, rs.segment_def_id)
    if ch is None:
        raise ValueError(f"chunk {rs.index} is not in the run's rebuild snapshot")
    images = list(ch.get("images") or [])
    if not images:
        raise ValueError(f"chunk {rs.index} has no images in the snapshot")
    expected = {sh["keyframe_path"]: sh.get("sha256") for sh in ch.get("shots") or []
                if sh.get("keyframe_path")}
    if ch.get("sheet_path"):
        expected[ch["sheet_path"]] = ch.get("sheet_sha256")
    for path in images:
        if not os.path.exists(path):
            raise ValueError(f"run-owned keyframe {path} is missing")
        want = expected.get(path)
        if want and file_sha256(path) != want:
            raise ValueError(f"run-owned keyframe {path} changed since launch")
    prompt = (rs.prompt_override or "").strip() or ch.get("prompt") or ""
    if not prompt:
        raise ValueError(f"chunk {rs.index} has no prompt")
    return ch, images, prompt


def _step_claim(row_id: str, gen: int, job_id: Optional[str]):
    with get_session() as s:
        token = pc.claim(s, RebuildChunkAttempt, row_id, gen)
        if token is None:
            s.rollback()
            return None
        a, rs, run = _load(s, row_id)
        try:
            chunk_inputs(run, rs)
        except ValueError as exc:
            _fail_unsent(s, a, rs, gen=gen, token=token, error=str(exc))
            s.commit()
            return None
        s.commit()
    pc.enqueue_step(TABLE, row_id, gen, pc.STEP_SEND)
    return None


def build_request(run, rs, attempt, image_urls: list[str], prompt: str) -> dict:
    """The exact createTask body of one attempt (pilot shapes, design §6.4)."""
    from .kie_client import build_images_only_payload

    snap = run.rebuild_snapshot or {}
    return build_images_only_payload(
        model=attempt.model, prompt=prompt, image_urls=list(image_urls),
        duration=int(attempt.request_sec), resolution=attempt.resolution,
        aspect_ratio=snap.get("aspect_ratio") or "9:16",
    )


def _upload_all(kie, paths: list[str], attempt_id: str) -> list[str]:
    """Upload the chunk's images under names unique to THIS attempt: the
    run-owned copies are ``<chunk>/1.png`` … ``sheet.png`` in every chunk, and
    the upload host overwrites a same-named file — two chunks sending at once
    would otherwise both be sent the images uploaded last."""
    return [jobs.upload(p, jobs.upload_name("ca", attempt_id, p, i), client=kie)
            for i, p in enumerate(paths)]


def _step_send(row_id: str, gen: int, job_id: Optional[str]):
    kie = _kie()
    with get_session() as s:
        a, rs, run = _load(s, row_id)
        if a is None or a.status != PC_CLAIMED or int(a.dispatch_gen) != gen:
            return None
        token = a.claim_token
        try:
            _ch, images, prompt = chunk_inputs(run, rs)
        except ValueError as exc:
            _fail_unsent(s, a, rs, gen=gen, token=token, error=str(exc))
            s.commit()
            return None
        # Uploads are free and idempotent (a transport error raises: the row
        # stays `claimed` and the reaper re-drives it).
        urls = _upload_all(kie, images, a.id)
        try:
            request = build_request(run, rs, a, urls, prompt)
        except ValueError as exc:
            _fail_unsent(s, a, rs, gen=gen, token=token, error=f"invalid request: {exc}")
            s.commit()
            return None
        try:
            sender = pc.sender_identity(job_id)
        except pc.SenderUnavailable as exc:
            log.error("attempt %s: not sending (%s)", row_id, exc)
            return None
        scope = spend_mod.run_scope(run.id)
        if not pc.begin_send(s, RebuildChunkAttempt, row_id, gen=gen, token=token,
                             request=request, sender=sender, scope_key=scope):
            s.rollback()
            return None
        s.commit()
        t0 = time.monotonic()
        outcome = None
        if pc.send_window_open(t0):
            outcome = kie.create_task_once(request, timeout=pc.SEND_HTTP_TIMEOUT_SEC)
        new = pc.record_send_outcome(s, RebuildChunkAttempt, row_id, gen=gen, token=token,
                                     outcome=outcome, scope_key=scope)
        a, rs, _run = _load(s, row_id)
        if new == PC_SUBMITTED:
            _segment_to(rs, "submitted")
        elif new in (PC_FAILED, PC_UNKNOWN):
            _segment_to(rs, "failed", error=a.error or new)
        s.commit()
    if new == PC_SUBMITTED:
        pc.enqueue_step(TABLE, row_id, gen, pc.STEP_POLL, delay_sec=pc.POLL_DELAY_VIDEO_SEC)
    return new


def _parse_result_url(data: dict) -> Optional[str]:
    raw = data.get("resultJson") or "{}"
    try:
        urls = json.loads(raw).get("resultUrls") or []
    except (ValueError, TypeError, AttributeError):
        urls = []
    return urls[0] if urls else None


def _parse_cost(data: dict) -> Optional[float]:
    raw = data.get("creditsConsumed")
    try:
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return None


def _sent_at(a) -> Optional[datetime]:
    if a.send_deadline is None:
        return None
    return a.send_deadline - timedelta(seconds=pc.SEND_WINDOW_SEC)


def _step_poll(row_id: str, gen: int, job_id: Optional[str]):
    kie = _kie()
    with get_session() as s:
        a, rs, run = _load(s, row_id)
        if a is None or a.status != PC_SUBMITTED or int(a.dispatch_gen) != gen:
            return None
        task_id = a.task_id
        sent = _sent_at(a)
    scope = spend_mod.run_scope(a.run_id)
    try:
        data = kie.get_task(task_id) or {}
        state = str(data.get("state") or "")
        err = data.get("failMsg") or data.get("failCode")
    except Exception as exc:  # noqa: BLE001 - polling is idempotent: try again
        log.warning("attempt %s: recordInfo failed (%s) — polling again", row_id, exc)
        data, state, err = {}, "", None
    overdue = sent is not None and (_utcnow() - sent).total_seconds() > POLL_GIVE_UP_SEC
    if state not in ("success", "fail") and overdue:
        state, err = "fail", (f"the provider did not finish task {task_id} within "
                              f"{POLL_GIVE_UP_SEC // 3600} h; it may still have billed")
    with get_session() as s:
        new = pc.record_poll(s, RebuildChunkAttempt, row_id, gen=gen, state=state,
                             scope_key=scope, credits_consumed=_parse_cost(data),
                             error=str(err) if err else None)
        a, rs, _run = _load(s, row_id)
        if new == PC_FETCHING:
            _segment_to(rs, "generating")
        elif new == PC_FAILED:
            _segment_to(rs, "failed", error=a.error or "provider task failed")
        s.commit()
    if new == PC_FETCHING:
        pc.enqueue_step(TABLE, row_id, gen, pc.STEP_FETCH)
    elif new == PC_SUBMITTED:
        pc.enqueue_step(TABLE, row_id, gen, pc.STEP_POLL, delay_sec=pc.POLL_DELAY_VIDEO_SEC)
    return new


def _ffprobe_json(path: str, *args: str) -> dict:
    from .media import harden_media_argv

    cmd = harden_media_argv(["ffprobe", "-v", "error", *args, "-of", "json", path])
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr[-500:]}")
    return json.loads(out.stdout or "{}")


def count_frames(path: str) -> int:
    """Decoded video frame count of *path* (exact; reads the whole stream)."""
    data = _ffprobe_json(path, "-select_streams", "v:0", "-count_frames",
                         "-show_entries", "stream=nb_read_frames")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"{path} has no video stream")
    return int(streams[0].get("nb_read_frames") or 0)


def _run_ffmpeg(cmd: list[str]) -> None:
    from .media import _run

    _run(cmd)


def fit_attempt(raw: str, fitted: str, chunk_frames: int, fps: float) -> dict:
    """The fit (design §6.4): resample *raw* to the output grid and keep its
    first *chunk_frames* frames (holding the last frame for up to
    ``FIT_SHORT_TOLERANCE_SEC``). Writes *fitted* atomically. Returns the
    duration QA values; raises :class:`FitRefused` when the result is too
    short (a final failure — never an automatic resubmit)."""
    from .media import probe

    info = probe(raw)
    if not info.duration_sec or info.duration_sec <= 0:
        raise RuntimeError(f"{raw} does not decode (no duration)")
    raw_frames = int(float(info.duration_sec) * float(fps) + 1e-6)
    plan = rb.fit_plan(raw_frames, chunk_frames, fps)
    if not plan.ok:
        raise FitRefused(plan.reason)
    part = f"{fitted}.{uuid.uuid4().hex[:8]}.part.mp4"
    try:
        _run_ffmpeg(rb.fit_ffmpeg_args(raw, part, plan, fps))
        got = count_frames(part)
        if got != chunk_frames:
            raise RuntimeError(f"fit produced {got} frames, expected {chunk_frames}")
        os.replace(part, fitted)
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
    return {"raw_duration_sec": round(float(info.duration_sec), 3),
            "raw_frames": raw_frames, "chunk_frames": int(chunk_frames),
            "padded_frames": int(plan.padded_frames), "fps_out": float(fps)}


class FitRefused(RuntimeError):
    """The raw result is too short for its chunk (a final attempt failure)."""


def _step_fetch(row_id: str, gen: int, job_id: Optional[str]):
    from .storage import rebuild_attempt_dir

    kie = _kie()
    with get_session() as s:
        a, rs, run = _load(s, row_id)
        if a is None or a.status != PC_FETCHING or int(a.dispatch_gen) != gen:
            return None
        task_id, chunk_frames_ = a.task_id, int(a.chunk_frames)
        fps = _fps_out(run)
        adir = rebuild_attempt_dir(a.id, run.id, run.project_id)
        run_id = run.id
    scope = spend_mod.run_scope(run_id)
    # Paths unique to THIS fetch execution (Terra 2b round 1): a stale fetch
    # overlapping a reaper re-drive must never write the files the winning
    # generation selected. The winner's paths go on the row with the fenced
    # `succeeded` CAS; a fenced-out execution removes its own files.
    tag = f"g{gen}-{uuid.uuid4().hex[:8]}"
    raw = os.path.join(adir, f"raw-{tag}.mp4")
    fitted = os.path.join(adir, f"fitted-{tag}.mp4")
    _sweep_stale_fetch_files(adir, gen)
    final_error = None
    retry_error = None
    duration_qa = None
    try:
        data = kie.get_task(task_id) or {}
        url = _parse_result_url(data)
        if not url:
            raise RuntimeError("recordInfo has no result URL")
        os.makedirs(adir, exist_ok=True)
        kie.download_result(url, raw)
        duration_qa = fit_attempt(raw, fitted, chunk_frames_, fps)
    except FitRefused as exc:
        final_error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a failed download retries the download only
        log.warning("attempt %s: fetch failed: %s", row_id, exc)
        retry_error = f"{type(exc).__name__}: {exc}"
    with get_session() as s:
        if duration_qa is not None:
            a, rs, _run = _load(s, row_id)
            if a is None or a.status != PC_FETCHING or int(a.dispatch_gen) != gen:
                _discard(raw, fitted)
                return None
            s.execute(update(RebuildChunkAttempt).where(
                RebuildChunkAttempt.run_segment_id == a.run_segment_id,
                RebuildChunkAttempt.id != a.id,
                RebuildChunkAttempt.selected.is_(True),
            ).values(selected=False).execution_options(synchronize_session=False))
            qa = dict(a.qa or {})
            qa["duration"] = dict(duration_qa, request_sec=int(a.request_sec))
            qa["flags"] = qa_flags(qa)
            if not pc.complete_fetch(s, RebuildChunkAttempt, row_id, gen=gen, scope_key=scope,
                                     values={"raw_path": raw, "fitted_path": fitted,
                                             "selected": True, "qa": qa}):
                s.rollback()
                _discard(raw, fitted)
                return None
            a, rs, _run = _load(s, row_id)
            _segment_to(rs, "completed")
            rs.local_result_path = fitted
            rs.error_message = None
            rs.cost_credits = a.cost_credits
            s.commit()
            outcome = PC_SUCCEEDED
        else:
            _discard(fitted)
            if final_error is not None and os.path.exists(raw):
                # A too-short take was still paid for: keep it on the row.
                pc.cas(s, RebuildChunkAttempt, row_id, expect_status=PC_FETCHING,
                       expect_gen=gen, values={"raw_path": raw})
            else:
                _discard(raw)
            res = pc.fail_fetch(s, RebuildChunkAttempt, row_id, gen=gen,
                                error=final_error or retry_error or "fetch failed",
                                scope_key=scope, final=final_error is not None)
            a, rs, _run = _load(s, row_id)
            if res == "failed":
                _segment_to(rs, "failed", error=a.error or "fetch failed")
            elif res is None:
                s.rollback()
                _discard(raw)
                return None
            s.commit()
            outcome = res
    if outcome == PC_SUCCEEDED:
        enqueue_attempt_qa(row_id)
    elif outcome == "retry":
        pc.enqueue_step(TABLE, row_id, gen, pc.STEP_FETCH, delay_sec=FETCH_RETRY_DELAY_SEC)
    return outcome


_FETCH_FILE_RE = re.compile(r"^(?:raw|fitted)-g(\d+)-[0-9a-f]{8}\.mp4(?:\..*)?$")


def _sweep_stale_fetch_files(adir: str, gen: int) -> None:
    """Remove the files of fetch executions of OLDER generations (Terra 2b
    round 2): every one of them is fenced out, and a worker killed after its
    download would otherwise leave a whole take behind on every re-drive."""
    try:
        names = os.listdir(adir)
    except OSError:
        return
    for name in names:
        m = _FETCH_FILE_RE.match(name)
        if m and int(m.group(1)) < int(gen):
            _discard(os.path.join(adir, name))


def _discard(*paths: str) -> None:
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            log.warning("could not remove %s", path)


# ---------------------------------------------------------------------------
# Keep segments of an images-only run (finalize) — design §6.4
# ---------------------------------------------------------------------------


def cut_keep_frames(source: str, dst: str, start_sec: float, end_sec: float,
                    fps: float) -> str:
    """Frame-exact keep cut ``[F(start), F(end))`` on the output grid (video
    only; the stitch muxes the soundtrack). Reuses an existing *dst*."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    f0, f1 = rb.frame_index(start_sec, fps), rb.frame_index(end_sec, fps)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = f"{dst}.{uuid.uuid4().hex[:8]}.part.mp4"
    try:
        _run_ffmpeg(rb.keep_cut_ffmpeg_args(source, part, f0, f1, fps))
        os.replace(part, dst)
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
    return dst


def keep_clip_name(index: int, start_sec: float, end_sec: float, fps: float) -> str:
    f0, f1 = rb.frame_index(start_sec, fps), rb.frame_index(end_sec, fps)
    return f"rbkeep_{int(index):04d}_{f0}_{f1}.mp4"


# ---------------------------------------------------------------------------
# QA (flags only, never blocks) — design §6.5
# ---------------------------------------------------------------------------


def qa_flags(qa: dict) -> list[str]:
    """Human-readable flags of an attempt's QA record."""
    flags: list[str] = []
    ct = qa.get("cut_timing") or {}
    if ct.get("flagged"):
        bits = []
        if ct.get("max_abs_delta"):
            bits.append(f"max |Δ| {ct['max_abs_delta']:.2f}s")
        if ct.get("missing"):
            bits.append(f"{len(ct['missing'])} missing")
        if ct.get("extra"):
            bits.append(f"{len(ct['extra'])} extra")
        flags.append("cut timing (" + ", ".join(bits) + ")" if bits else "cut timing")
    tb = qa.get("text_band") or {}
    if tb.get("candidate_band_y"):
        flags.append(f"burned text ({tb.get('state')})")
    dur = qa.get("duration") or {}
    if dur.get("padded_frames"):
        flags.append(f"padded {dur['padded_frames']} frame(s)")
    cc = qa.get("character_check")
    if isinstance(cc, dict):
        for shot in cc.get("shots") or []:
            if shot.get("missing"):
                flags.append(f"shot {shot.get('shot')}: missing {', '.join(map(str, shot['missing']))}")
            for d in shot.get("identity_drift") or []:
                if isinstance(d, dict):
                    flags.append(f"shot {shot.get('shot')}: {d.get('character')} looks like "
                                 f"{d.get('looks_like')}")
                else:
                    flags.append(f"shot {shot.get('shot')}: identity drift ({d})")
    return flags


def _write_qa(attempt_id: str, updates: dict) -> Optional[dict]:
    """Merge *updates* into the attempt's ``qa`` (recomputing ``flags``) under
    a row lock; QA never changes any state. Returns the new record."""
    with get_session() as s:
        a = s.execute(select(RebuildChunkAttempt).where(RebuildChunkAttempt.id == attempt_id)
                      .with_for_update().execution_options(populate_existing=True)
                      ).scalar_one_or_none()
        if a is None:
            return None
        qa = dict(a.qa or {})
        qa.update(updates)
        qa["flags"] = qa_flags(qa)
        a.qa = qa
        s.commit()
        return qa


def detected_cuts(path: str, fps: float) -> list[float]:
    from . import scene as scene_mod

    ranges = scene_mod.detect_scenes(path, fps=fps, min_scene_len_sec=rb.REBUILD_SHOT_MIN_SEC)
    return [float(start) for start, _end in list(ranges)[1:]]


def contact_sheet(fitted: str, shots: list[dict], dst: str, *, vertical: bool) -> str:
    """One row per planned shot: its midpoint frame of the fitted clip next to
    its keyframe (the operator's eye check, and the character check's input)."""
    cw, ch = (SHEET_CELL_LONG * 9 // 16, SHEET_CELL_LONG) if vertical else \
        (SHEET_CELL_LONG, SHEET_CELL_LONG * 9 // 16)
    cw -= cw % 2
    cmd = ["ffmpeg", "-v", "error", "-y"]
    parts = []
    cell = (f"scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    for i, sh in enumerate(shots):
        mid = max(0.0, (float(sh["rel_start"]) + float(sh["rel_end"])) / 2.0)
        cmd += ["-ss", f"{mid:.3f}", "-i", fitted, "-i", sh["keyframe_path"]]
        parts.append(f"[{2 * i}:v]{cell},trim=end_frame=1[g{i}]")
        parts.append(f"[{2 * i + 1}:v]{cell}[k{i}]")
        parts.append(f"[g{i}][k{i}]hstack=inputs=2[r{i}]")
    n = len(shots)
    if n == 1:
        out = "[r0]"
    else:
        parts.append("".join(f"[r{i}]" for i in range(n)) + f"vstack=inputs={n}[out]")
        out = "[out]"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = f"{dst}.{uuid.uuid4().hex[:8]}.part.jpg"
    cmd += ["-filter_complex", ";".join(parts), "-map", out, "-frames:v", "1",
            "-q:v", "4", part]
    try:
        _run_ffmpeg(cmd)
        os.replace(part, dst)
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
    return dst


def free_qa(attempt_id: str) -> Optional[dict]:
    """The free QA values (cut timing, burned text, contact sheet), each
    independent: one failing records its error and the others still run."""
    from .models import Run, RunSegment
    from .storage import rebuild_attempt_dir

    with get_session() as s:
        a = s.get(RebuildChunkAttempt, attempt_id)
        if a is None or a.status != PC_SUCCEEDED or not a.fitted_path:
            return None
        run = s.get(Run, a.run_id)
        rs = s.get(RunSegment, a.run_segment_id)
        ch = snapshot_chunk(run, rs.segment_def_id) if run is not None and rs else None
        fitted, fps = a.fitted_path, _fps_out(run)
        free_runs = int((a.qa or {}).get("free_runs") or 0) + 1
        adir = rebuild_attempt_dir(a.id, run.id, run.project_id)
        vertical = ((run.rebuild_snapshot or {}).get("aspect_ratio") or "9:16") == "9:16"
    shots = list((ch or {}).get("shots") or [])
    # Count this run BEFORE the work (gate round 2): a run that crashes or
    # times out still counts toward FREE_QA_MAX_RUNS in reap_qa.
    _write_qa(attempt_id, {"free_runs": free_runs})
    updates: dict = {"free_done_at": _utcnow().isoformat(timespec="seconds"),
                     "free_runs": free_runs}
    errors: dict = {}
    try:
        pieces = [rb.ChunkPiece(sh["shot_id"], float(sh["rel_start"]), float(sh["rel_end"]))
                  for sh in shots]
        updates["cut_timing"] = rb.cut_timing_report(rb.planned_cuts(pieces),
                                                     detected_cuts(fitted, fps))
    except Exception as exc:  # noqa: BLE001
        errors["cut_timing"] = f"{type(exc).__name__}: {exc}"[:500]
    try:
        from . import qa_tools

        found = qa_tools.locate_captions(fitted)
        updates["text_band"] = {"state": found.get("state"),
                                "candidate_band_y": found.get("candidate_band_y")}
    except Exception as exc:  # noqa: BLE001
        errors["text_band"] = f"{type(exc).__name__}: {exc}"[:500]
    try:
        if shots:
            updates["contact_sheet"] = contact_sheet(
                fitted, shots, os.path.join(adir, "sheet.jpg"), vertical=vertical)
    except Exception as exc:  # noqa: BLE001
        errors["contact_sheet"] = f"{type(exc).__name__}: {exc}"[:500]
    if errors:
        updates["errors"] = errors
    return _write_qa(attempt_id, updates)


def _give_up_check(attempt_id: str) -> bool:
    """Write the ``unavailable`` check verdict for an attempt whose free QA
    never produced a contact sheet — re-checked UNDER the attempt row lock
    (gate round 3): the reaper's snapshot may be stale, and a QA run finishing
    meanwhile may have written the sheet and admitted the check. Only when the
    sheet is still missing, the runs are used up, there is no verdict and no
    check call exists for the attempt."""
    from sqlalchemy import func as sa_func

    with get_session() as s:
        a = s.execute(select(RebuildChunkAttempt).where(RebuildChunkAttempt.id == attempt_id)
                      .with_for_update().execution_options(populate_existing=True)
                      ).scalar_one_or_none()
        if a is None or a.status != PC_SUCCEEDED:
            return False
        qa = dict(a.qa or {})
        if "character_check" in qa or (qa.get("free_done_at") and qa.get("contact_sheet")):
            return False
        if int(qa.get("free_runs") or 0) < FREE_QA_MAX_RUNS:
            return False
        calls = s.execute(select(sa_func.count()).select_from(RebuildLlmCall).where(
            RebuildLlmCall.attempt_id == attempt_id,
            RebuildLlmCall.kind == "character_check")).scalar()
        if calls:
            return False
        qa["character_check"] = CHECK_UNAVAILABLE
        qa["flags"] = qa_flags(qa)
        a.qa = qa
        s.commit()
        return True


def run_attempt_qa(attempt_id: str) -> None:
    """RQ entry (``autopilot_compute`` queue): the free QA of a succeeded
    attempt, then the automatic character check (Q9)."""
    qa = free_qa(attempt_id)
    if qa is None:
        return
    admit_auto_check(attempt_id)


def qa_job_id(attempt_id: str) -> str:
    return f"{QA_JOB_PREFIX}{attempt_id}"


def enqueue_attempt_qa(attempt_id: str) -> None:
    """Best effort (a lost QA job is re-enqueued by :func:`reap_qa`)."""
    try:
        from .tasks import QUEUE_AUTOPILOT_COMPUTE, _enqueue_fixed_id

        _enqueue_fixed_id(QUEUE_AUTOPILOT_COMPUTE, "app.pipeline_rebuild.run_attempt_qa",
                          (attempt_id,), qa_job_id(attempt_id), QA_JOB_TIMEOUT_SEC)
    except Exception:
        log.exception("enqueue of QA for attempt %s failed — the QA reaper retries", attempt_id)


def reap_qa(*, conn=None, now: Optional[datetime] = None, is_live=None,
            enqueue=None) -> dict:
    """From ``run_reconcile_tick`` (attempts that succeeded in the last
    :data:`QA_REAP_WINDOW_DAYS` days):

    * re-enqueue the FREE QA of an attempt whose QA job was lost;
    * admit the automatic character check when it was never admitted;
    * fold a check whose last call ended WITHOUT its handler (the paid-call
      reaper released a ``sending`` row to ``unknown``) into the attempt's QA
      — the fallback slot once, else ``unavailable``.

    Never re-sends a call: a check row is driven by its own protocol."""
    now = now or _utcnow()
    out = {"qa": 0, "checks": 0, "folded": 0}
    cutoff = now - timedelta(seconds=QA_REAP_GRACE_SEC)
    if is_live is None:
        from .tasks import QUEUE_AUTOPILOT_COMPUTE, _get_redis, rq_job_is_live

        conn = conn or _get_redis()
        is_live = lambda jid: rq_job_is_live(conn, QUEUE_AUTOPILOT_COMPUTE, jid)  # noqa: E731
    enqueue = enqueue or enqueue_attempt_qa
    with get_session() as s:
        rows = s.execute(select(RebuildChunkAttempt.id, RebuildChunkAttempt.qa).where(
            RebuildChunkAttempt.status == PC_SUCCEEDED,
            RebuildChunkAttempt.finished_at < cutoff,
            RebuildChunkAttempt.finished_at > now - timedelta(days=QA_REAP_WINDOW_DAYS),
        )).all()
        calls: dict[str, list] = {}
        if rows:
            for cid, aid, slot, status in s.execute(select(
                    RebuildLlmCall.id, RebuildLlmCall.attempt_id, RebuildLlmCall.slot,
                    RebuildLlmCall.status).where(
                    RebuildLlmCall.kind == "character_check",
                    RebuildLlmCall.attempt_id.in_([r[0] for r in rows]))).all():
                calls.setdefault(aid, []).append((int(slot or 0), status, cid))
    for aid, qa in rows:
        qa = qa or {}
        verdict = "character_check" in qa
        if not qa.get("free_done_at") or (not verdict and not qa.get("contact_sheet")):
            # The free QA job was lost, crashed, or ran but made no contact
            # sheet (a transient ffmpeg / disk error): the check cannot be
            # admitted without one. Re-run the free part a bounded number of
            # times (free_runs is counted when a run STARTS), then give the
            # check its verdict instead of retrying silently forever and
            # keeping its allowance reserved (integration gate rounds 1-2).
            if is_live(qa_job_id(aid)):
                continue  # a run is in progress: let it finish
            if int(qa.get("free_runs") or 0) < FREE_QA_MAX_RUNS:
                enqueue(aid)
                out["qa"] += 1
            elif not verdict and _give_up_check(aid):
                out["folded"] += 1
            continue
        if verdict:
            continue
        mine = sorted(calls.get(aid, []))
        if not mine:
            if admit_auto_check(aid):
                out["checks"] += 1
        elif all(st not in PAID_CALL_LIVE_STATUSES for _slot, st, _cid in mine):
            _check_terminal(mine[-1][2])
            out["folded"] += 1
    return out


# ---------------------------------------------------------------------------
# Character check (the one paid QA step) — design §6.5, Q9
# ---------------------------------------------------------------------------


def check_models() -> list[str]:
    """Primary + its one fallback (the storyboard's model chain)."""
    from .config import settings

    chain = [settings.LOCALISATION_TRANSCRIBE_MODEL] + [
        m.strip() for m in (settings.LOCALISATION_TRANSCRIBE_FALLBACK_MODELS or "").split(",")
        if m.strip()]
    out: list[str] = []
    for m in chain:
        if m and m not in out:
            out.append(m)
    return (out + out[:1])[:2] if out else ["gemini-3.1-pro", "gemini-3-pro"]


def admit_check(attempt_id: str, slot: int) -> Optional[str]:
    """Admit the character-check call of *slot* (0 = primary, 1 = fallback)
    for a succeeded attempt with a contact sheet; covered by the run's
    reservation (the estimator counts both slots). Returns the row id, or
    None (no sheet, a live check, the slot already used)."""
    from .models import Run

    with get_session() as s:
        a = s.get(RebuildChunkAttempt, attempt_id)
        if a is None:
            return None
        # The Run row lock first (integration gate round 1): delete_run takes
        # it before counting live checks, so an admission either lands before
        # (the delete then 409s on the queued check) or finds the run gone —
        # never a check sent for a run whose delete already passed its guard.
        run = s.execute(select(Run).where(Run.id == a.run_id).with_for_update()
                        .execution_options(populate_existing=True)).scalar_one_or_none()
        if run is None:
            return None
        # Then the attempt row (lock order runs -> attempts): serialises with
        # _give_up_check, which writes the "unavailable" verdict under this
        # lock only while no check call exists. An automatic slot is never
        # admitted once a verdict exists (its allowance is then released by
        # the estimator); a manual check (slot >= 2) is reserved on its own.
        a = s.execute(select(RebuildChunkAttempt).where(RebuildChunkAttempt.id == attempt_id)
                      .with_for_update().execution_options(populate_existing=True)
                      ).scalar_one_or_none()
        if a is None or a.status != PC_SUCCEEDED:
            return None
        if not (a.qa or {}).get("contact_sheet"):
            return None
        if int(slot) in CHECK_SLOTS and "character_check" in (a.qa or {}):
            return None
        call = RebuildLlmCall(project_id=run.project_id, attempt_id=a.id,
                              kind="character_check", slot=int(slot),
                              model=check_models()[min(int(slot), 1)], status=PC_QUEUED,
                              estimate_credits=spend_mod.llm_credits())
        try:
            s.add(call)
            s.commit()
        except IntegrityError:
            s.rollback()
            return None
        cid = call.id
    try:
        pc.enqueue_step(LLM_TABLE, cid, 1, pc.STEP_CLAIM)
    except Exception:
        log.exception("enqueue of character check %s failed — the reaper re-sends it", cid)
    return cid


def admit_auto_check(attempt_id: str) -> Optional[str]:
    """The automatic check (Q9): slot 0 once, never again — and never once
    the check already has a verdict (the QA reaper may have given up)."""
    with get_session() as s:
        a = s.get(RebuildChunkAttempt, attempt_id)
        if a is None or "character_check" in (a.qa or {}):
            return None
    return admit_check(attempt_id, 0)


def character_check_prompt(chunk: dict) -> str:
    chars = chunk.get("characters") or []
    lines = []
    for c in chars:
        bits = "; ".join(x for x in (c.get("look"), c.get("outfit")) if x)
        lines.append(f"- {c.get('id')}: {c.get('name_hint') or c.get('id')}"
                     + (f" ({bits})" if bits else ""))
    shot_lines = []
    for i, sh in enumerate(chunk.get("shots") or [], start=1):
        expected = ", ".join(map(str, sh.get("characters") or [])) or "nobody in particular"
        shot_lines.append(f"Shot {i}: expected {expected}")
    n = len(chunk.get("shots") or [])
    return (
        f"The image is a contact sheet of {n} row(s), one per shot, in order. In each row the "
        "LEFT image is a frame of a generated video and the RIGHT image is the reference "
        "keyframe it was generated from.\n"
        "Characters:\n" + ("\n".join(lines) or "- (none described)") + "\n"
        "Expected characters per shot:\n" + "\n".join(shot_lines) + "\n"
        "For every shot, compare the LEFT frame with the RIGHT keyframe and the character "
        "descriptions. Report expected characters that are missing from the LEFT frame, and "
        "identity drift (a character in the LEFT frame looks like a DIFFERENT character).\n"
        'Answer with ONE JSON object only: {"shots": [{"shot": 1, "present": ["id"], '
        '"missing": ["id"], "identity_drift": [{"character": "id", "looks_like": "id"}], '
        '"notes": "short"}]}'
    )


def _parse_json_object(text: Optional[str]) -> Optional[dict]:
    raw = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    if m:
        raw = m.group(1)
    for cand in (raw, raw[raw.find("{"): raw.rfind("}") + 1] if "{" in raw else ""):
        if not cand:
            continue
        try:
            val = json.loads(cand)
        except (TypeError, ValueError):
            continue
        if isinstance(val, dict):
            return val
    return None


def _chat_content(body) -> Optional[str]:
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def summarize_check(answer: Optional[dict], chunk: dict) -> dict:
    """The stored character-check result: per shot ``present / missing /
    identity_drift / notes`` (shots numbered 1..n), plus ``flagged``."""
    shots_out = []
    n = len(chunk.get("shots") or [])
    for item in (answer or {}).get("shots") or []:
        if not isinstance(item, dict):
            continue
        try:
            k = int(item.get("shot"))
        except (TypeError, ValueError):
            continue
        if not 1 <= k <= max(n, 1):
            continue
        drift = [d for d in (item.get("identity_drift") or []) if d]
        shots_out.append({
            "shot": k,
            "present": [str(x) for x in item.get("present") or []],
            "missing": [str(x) for x in item.get("missing") or []],
            "identity_drift": drift,
            "notes": str(item.get("notes") or "")[:300],
        })
    flagged = any(sh["missing"] or sh["identity_drift"] for sh in shots_out)
    return {"shots": shots_out, "flagged": flagged}


def character_check_step(row_id: str, gen: int, step: str, job_id: Optional[str]):
    """The ``character_check`` kind of the ``rebuild_llm_calls`` dispatcher
    (``app/rebuild_llm_steps.py``). An LLM call has no poll / fetch:

    * claim: CAS, then the free preparation (upload the contact sheet under a
      name unique to this call, build the request) stored on the row
      (``rebuild_jobs.store_prepared_request``); a missing input ends the row
      ``failed`` "not sent" (``rebuild_jobs.fail_unsent``, nothing drawn);
    * send: ``rebuild_jobs.send_claimed_row`` — ONE ``chat_completion_once``
      inside the send window, drawn in the run scope (the same send step as
      the storyboard calls).

    On a non-success the fallback slot is admitted once; after it the check is
    ``unavailable`` (:func:`_check_terminal`)."""
    gen = int(gen)
    sf = jobs.session_factory()
    if step == pc.STEP_CLAIM:
        return _check_claim(row_id, gen, sf)
    if step != pc.STEP_SEND:
        log.error("character_check_step: unexpected step %r for %s", step, row_id)
        return None
    kie = _kie()

    def call(request: dict):
        return kie.chat_completion_once(request["model"], request["messages"],
                                        timeout=float(request.get("timeout")
                                                      or CHECK_HTTP_TIMEOUT_SEC))

    with get_session() as s:
        row = s.get(RebuildLlmCall, row_id)
        attempt = s.get(RebuildChunkAttempt, row.attempt_id) if row is not None else None
        run_id = attempt.run_id if attempt is not None else None
    if run_id is None:
        return None
    new = jobs.send_claimed_row(RebuildLlmCall, row_id, gen, job_id, sf=sf, call=call,
                                scope_key_fn=lambda _row: spend_mod.run_scope(run_id))
    if new is not None:
        _check_terminal(row_id)
    return new


def _check_claim(row_id: str, gen: int, sf) -> Optional[str]:
    from .models import Run, RunSegment

    db = sf()
    try:
        token = pc.claim(db, RebuildLlmCall, row_id, gen)
        if token is None:
            db.rollback()
            return None
        call = db.get(RebuildLlmCall, row_id)
        a = db.get(RebuildChunkAttempt, call.attempt_id)
        run = db.get(Run, a.run_id) if a is not None else None
        rs = db.get(RunSegment, a.run_segment_id) if a is not None else None
        chunk = snapshot_chunk(run, rs.segment_def_id) if run is not None and rs else None
        sheet = (a.qa or {}).get("contact_sheet") if a is not None else None
        model = call.model
        db.commit()
    finally:
        db.close()

    error = None
    request = None
    if chunk is None or not sheet or not os.path.exists(sheet):
        error = "no contact sheet to check"
    else:
        try:
            url = jobs.upload(sheet, jobs.upload_name("cc", row_id, sheet), client=_kie())
            request = {"model": model, "timeout": CHECK_HTTP_TIMEOUT_SEC, "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": character_check_prompt(chunk)},
                    {"type": "image_url", "image_url": {"url": url}},
                ]}]}
        except Exception as exc:  # noqa: BLE001 - free work failed: nothing sent
            log.exception("character check %s: input preparation failed", row_id)
            error = f"{type(exc).__name__}: {exc}"
    db = sf()
    try:
        if error is not None:
            ok = jobs.fail_unsent(db, RebuildLlmCall, row_id, gen=gen, token=token, error=error)
            db.commit()
            if ok:
                _check_terminal(row_id)
            return PC_FAILED if ok else None
        if not jobs.store_prepared_request(db, RebuildLlmCall, row_id, gen=gen, token=token,
                                           request=request):
            db.rollback()
            return None
        db.commit()
    finally:
        db.close()
    pc.enqueue_step(LLM_TABLE, row_id, gen, pc.STEP_SEND)
    return PC_CLAIMED


def _check_terminal(call_id: str) -> None:
    """Fold a terminal check call into its attempt's QA: a success writes the
    per-shot result; a failure admits the fallback slot once, then marks the
    check ``unavailable``."""
    from .models import Run, RunSegment

    with get_session() as s:
        call = s.get(RebuildLlmCall, call_id)
        if call is None or call.status in PAID_CALL_LIVE_STATUSES:
            return
        a = s.get(RebuildChunkAttempt, call.attempt_id)
        if a is None:
            return
        run = s.get(Run, a.run_id)
        rs = s.get(RunSegment, a.run_segment_id)
        chunk = snapshot_chunk(run, rs.segment_def_id) if run is not None and rs else {}
        status, slot, body, attempt_id = call.status, int(call.slot or 0), call.response, a.id
    answer = _parse_json_object(_chat_content(body)) if status == PC_SUCCEEDED else None
    if answer is not None:
        result = summarize_check(answer, chunk or {})
        result["call_id"] = call_id
        _write_qa(attempt_id, {"character_check": result})
        return
    # A failed / unknown call, or an answer with no JSON object in it (Terra
    # 2b round 1: never shown as a clean check): the fallback slot once.
    if slot < max(CHECK_SLOTS):
        if admit_check(attempt_id, slot + 1):
            return
        # Lost the admission to a concurrent fold (the handler and the QA
        # reaper both folding this call — Terra 2b round 2): the later slot's
        # own fold decides, never this one.
        with get_session() as s:
            later = s.execute(select(func.count()).select_from(RebuildLlmCall).where(
                RebuildLlmCall.attempt_id == attempt_id,
                RebuildLlmCall.kind == "character_check",
                RebuildLlmCall.slot > slot)).scalar()
        if later:
            return
    _write_qa(attempt_id, {"character_check": CHECK_UNAVAILABLE})


# ---------------------------------------------------------------------------
# Repair (API, in the reservation transaction) — design §6.6, §7.3
# ---------------------------------------------------------------------------


class RepairRefused(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _require_no_live_attempts(db, run) -> None:
    n = live_attempt_count(db, run.id)
    if n:
        raise RepairRefused(409, f"{n} chunk attempt(s) of this run are still in flight; "
                                 "wait for them to finish")


def reset_for_retry(db, run) -> list:
    """Retry Run (failed/incomplete): every active chunk segment that is not
    completed with its file goes back to ``pending`` (its attempt failed or
    ended unknown — the new attempt is a NEW row). Caller reserves next."""
    from .models import RunSegment, VideoProject
    from .state_machine import SegmentStatus

    _require_no_live_attempts(db, run)
    project = db.get(VideoProject, run.project_id)
    active = active_chunk_ids(run, project)
    reset = []
    for rs in db.execute(select(RunSegment).where(RunSegment.run_id == run.id)).scalars():
        if rs.segment_def_id not in active:
            continue
        ok = (_status(rs.status) == SegmentStatus.completed.value and rs.local_result_path
              and os.path.exists(rs.local_result_path))
        if ok:
            continue
        rs.status = SegmentStatus.pending
        rs.error_message = None
        rs.local_result_path = None
        reset.append(rs)
    db.flush()
    return reset


def reset_for_rerun(db, run, segments, prompts: Optional[dict] = None) -> None:
    """Segment rerun ("re-roll"): each listed chunk back to ``pending`` for a
    NEW attempt from the frozen snapshot (an optional prompt override is
    stored on the RunSegment and used by the next request). Its current
    attempt is unselected — "use attempt N" brings any older one back."""
    from .models import VideoProject
    from .state_machine import SegmentStatus

    _require_no_live_attempts(db, run)
    project = db.get(VideoProject, run.project_id)
    active = active_chunk_ids(run, project)
    for rs in segments:
        if rs.segment_def_id not in active:
            raise RepairRefused(400, f"segment {rs.id} is not an active chunk of this run")
    for rs in segments:
        if prompts and rs.id in prompts:
            text = prompts[rs.id]
            if text is not None:
                rs.prompt_override = text.strip() or None
        db.execute(update(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_segment_id == rs.id,
            RebuildChunkAttempt.selected.is_(True)).values(selected=False)
            .execution_options(synchronize_session=False))
        rs.status = SegmentStatus.pending
        rs.error_message = None
        rs.local_result_path = None
        rs.source_fallback = False
    db.flush()


def use_attempt(db, run, rs, attempt) -> None:
    """"Use attempt N": re-select an older succeeded attempt of a chunk (its
    fitted file must still exist); the caller re-stitches."""
    from .state_machine import SegmentStatus

    _require_no_live_attempts(db, run)
    if attempt.run_segment_id != rs.id or attempt.run_id != run.id:
        raise RepairRefused(404, "attempt not found on this chunk")
    if attempt.status != PC_SUCCEEDED or not attempt.fitted_path \
            or not os.path.exists(attempt.fitted_path):
        raise RepairRefused(409, "only a succeeded attempt whose fitted file still exists "
                                 "can be used")
    db.execute(update(RebuildChunkAttempt).where(
        RebuildChunkAttempt.run_segment_id == rs.id,
        RebuildChunkAttempt.id != attempt.id,
        RebuildChunkAttempt.selected.is_(True)).values(selected=False)
        .execution_options(synchronize_session=False))
    db.flush()
    attempt.selected = True
    rs.status = SegmentStatus.completed
    rs.local_result_path = attempt.fitted_path
    rs.error_message = None
    rs.cost_credits = attempt.cost_credits
    db.flush()


__all__ = [
    "submit_images_only", "observe_run", "chunk_attempt_step", "character_check_step",
    "run_attempt_qa", "reap_qa", "cut_keep_frames", "reset_for_retry", "reset_for_rerun",
    "use_attempt", "fit_attempt", "PC_SENDING",
]
