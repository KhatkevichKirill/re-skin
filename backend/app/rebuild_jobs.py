"""Shared plumbing of the rebuild project-phase jobs (wave 2a).

docs/rebuild-from-reference-design.md §5.2-§5.3, §7.2. The storyboard
(``app/rebuild_storyboard.py``) and the keyframes (``app/rebuild_keyframes.py``)
drive their paid-call rows through the wave 1 protocol
(``app/rebuild_paid_call.py``); what they share lives here:

* the session factory and the kie.ai client getter (both patched in tests);
* :func:`send_claimed_row` — the ``send`` step of §2.1 of the implementation
  notes: fenced ``claimed -> sending`` + draw, COMMIT, ONE provider call
  (``create_task_once`` / ``chat_completion_once`` — never a retrying method),
  record the outcome;
* :func:`fail_unsent` — a ``claimed`` row whose free input preparation failed
  ends ``failed`` with nothing sent and nothing drawn;
* free media helpers: a full-resolution source frame, the low-res LLM proxy;
* the project's aspect ratio for GPT Image.

Nothing here makes a provider request except through the callable a step
passes to :func:`send_claimed_row`.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from . import rebuild_paid_call as pc
from .rebuild_models import PC_CLAIMED, PC_FAILED, PC_SUBMITTED

log = logging.getLogger(__name__)

# Uploaded inputs go under this kie.ai temp-host directory.
UPLOAD_PATH = "rebuild"
# The LLM proxy must stay small: gemini-3.1-pro refused a 4 MB upload with
# "Failed to get the file information" and worked under 1 MB (design §13).
LLM_PROXY_MAX_BYTES = 1_000_000
# (long side px, crf, fps, mono AAC kbit/s) — the first rung is the spec
# (scale -2:854, 15 fps, crf 28, mono aac); later rungs only when the clip is
# still over the ceiling (a 95 s source's audio alone is ~570 KB at 48k).
LLM_PROXY_LADDER = ((854, 28, 15, 48), (854, 34, 15, 32), (640, 38, 12, 24),
                    (480, 42, 10, 16))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def session_factory():
    from .db import SessionLocal

    return SessionLocal


def kie_client():
    """The kie.ai client the step jobs use (tests patch this)."""
    from .kie_client import get_shared_client

    return get_shared_client()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def fail_unsent(db, model, row_id: str, *, gen: int, token: str, error: str) -> bool:
    """``claimed -> failed`` for a row whose free preparation failed. Nothing
    was sent and nothing was drawn (the draw is recorded only at ``sending``),
    so ``cost_credits`` is 0. Fenced by token + gen. In the caller's
    transaction."""
    now = _utcnow()
    return pc.cas(db, model, row_id, expect_status=PC_CLAIMED, expect_gen=gen,
                  expect_token=token, now=now, values={
                      "status": PC_FAILED, "error": f"not sent: {error}"[:2000],
                      "claim_token": None, "finished_at": now, "cost_credits": 0.0,
                  })


def store_prepared_request(db, model, row_id: str, *, gen: int, token: str,
                           request: dict) -> bool:
    """Write the fully built provider request onto a ``claimed`` row (fenced).
    The send step sends exactly this body; ``begin_send`` records it again
    with the ``sending`` marker."""
    return pc.cas(db, model, row_id, expect_status=PC_CLAIMED, expect_gen=gen,
                  expect_token=token, values={"request": request})


def send_claimed_row(
    model,
    row_id: str,
    gen: int,
    job_id: Optional[str],
    *,
    scope_key_fn: Callable[[object], str],
    call: Callable[[dict], object],
    sf: Optional[Callable] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Optional[str]:
    """The ``send`` step (implementation notes §2.1 step 4).

    Reads the row; stops unless it is ``claimed`` at *gen* with a prepared
    request; records the sender identity (refuses to send without one);
    ``begin_send`` and COMMIT; then — only while the send window is open —
    exactly ONE provider call ``call(request)`` returning a
    ``kie_client.SendOutcome``; records the outcome and commits. Returns the
    row's new status, or None when nothing was done.
    """
    sf = sf or session_factory()
    db = sf()
    try:
        row = db.get(model, row_id)
        if row is None or row.status != PC_CLAIMED or int(row.dispatch_gen) != int(gen):
            db.rollback()
            return None
        token = row.claim_token
        request = row.request
        scope_key = scope_key_fn(row)
        if not token or not isinstance(request, dict) or not request:
            log.error("%s %s: send step without a prepared request", model.__tablename__, row_id)
            db.rollback()
            return None
        try:
            sender = pc.sender_identity(job_id)
        except pc.SenderUnavailable:
            log.error("%s %s: no sender identity in this process — not sending",
                      model.__tablename__, row_id)
            db.rollback()
            return None
        if not pc.begin_send(db, model, row_id, gen=gen, token=token, request=request,
                             sender=sender, scope_key=scope_key):
            db.rollback()
            return None
        db.commit()
    finally:
        db.close()

    t0 = clock()
    outcome = None
    if pc.send_window_open(t0, clock=clock):
        outcome = call(request)
    else:  # pragma: no cover - needs a stalled process between commit and here
        log.warning("%s %s: send window elapsed — not sending", model.__tablename__, row_id)

    db = sf()
    try:
        status = pc.record_send_outcome(db, model, row_id, gen=gen, token=token,
                                        outcome=outcome, scope_key=scope_key)
        if status is None:
            # Only the reaper can move a `sending` row, and only on a proof
            # that this process is gone — so this is unexpected.
            log.error("%s %s: outcome %s could not be recorded (fenced out)",
                      model.__tablename__, row_id, getattr(outcome, "kind", None))
            db.rollback()
            return None
        db.commit()
        return status
    finally:
        db.close()


def is_submitted(status: Optional[str]) -> bool:
    return status == PC_SUBMITTED


# ---------------------------------------------------------------------------
# Free media helpers
# ---------------------------------------------------------------------------


def aspect_ratio_for(width: Optional[int], height: Optional[int]) -> str:
    """The GPT Image aspect of the project (design §5.3: 9:16 / 16:9)."""
    w, h = int(width or 0), int(height or 0)
    if w and h and w == h:
        return "1:1"
    if w and h and w > h:
        return "16:9"
    return "9:16"


def frame_path(project_id: str, shot_id: str, t: float) -> str:
    """A full-resolution source frame of *shot_id* at *t* seconds.

    The time is part of the name (``frames/<shot id>_<ms>.jpg``): an operator
    may move a shot's ``frame_sec`` while an older keyframe row still names
    the old frame, so every frame a row was fed stays on disk unchanged.
    """
    from .storage import rebuild_project_dir

    ms = int(round(float(t) * 1000))
    return os.path.join(rebuild_project_dir(project_id), "frames", f"{shot_id}_{ms}.jpg")


def _replace_from_tmp(dst: str, write: Callable[[str], None], suffix: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rb.", suffix=suffix, dir=os.path.dirname(dst))
    os.close(fd)
    try:
        write(tmp)
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            from .media import MediaError

            raise MediaError(f"ffmpeg wrote nothing for {dst}")
        os.replace(tmp, dst)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    return dst


def extract_frame(src: str, t: float, dst: str, *, width: Optional[int] = None,
                  quality: int = 2) -> str:
    """One JPEG still of *src* at *t* seconds (idempotent: an existing *dst*
    is kept; written via a temp file + ``os.replace``)."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    from .media import FFMPEG, _run

    def write(tmp: str) -> None:
        cmd = [FFMPEG, "-y", "-ss", f"{max(0.0, float(t)):.3f}", "-i", src, "-frames:v", "1"]
        if width:
            cmd += ["-vf", f"scale={int(width)}:-2"]
        cmd += ["-q:v", str(int(quality)), "-f", "image2", tmp]
        _run(cmd)

    return _replace_from_tmp(dst, write, ".jpg")


def build_llm_proxy(src: str, dst: str, *, vertical: bool, start: Optional[float] = None,
                    end: Optional[float] = None, max_bytes: int = LLM_PROXY_MAX_BYTES) -> str:
    """A low-res proxy of *src* (or of ``[start, end]``) for Gemini to read.

    First rung: long side 854 (``scale=-2:854`` for a vertical source), 15 fps,
    CRF 28, mono AAC — then smaller rungs while the file is over *max_bytes*
    (the last rung is kept whatever its size). Name the upload plainly: a
    dotted/underscored name was refused in the pilot.
    """
    from .media import FFMPEG, _FFMPEG_THREADS, _run

    last = None
    for long_side, crf, fps, kbps in LLM_PROXY_LADDER:
        scale = f"scale=-2:{long_side}" if vertical else f"scale={long_side}:-2"

        def write(tmp: str, scale=scale, crf=crf, fps=fps, kbps=kbps) -> None:
            cmd = [FFMPEG, "-y", "-threads", _FFMPEG_THREADS]
            if start is not None:
                cmd += ["-ss", f"{max(0.0, float(start)):.3f}"]
            cmd += ["-i", src]
            if start is not None and end is not None:
                cmd += ["-t", f"{max(0.05, float(end) - float(start)):.3f}"]
            cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-vf", f"{scale},fps={fps}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-b:a", f"{kbps}k",
                    "-movflags", "+faststart", "-f", "mp4", tmp]
            _run(cmd)

        if os.path.exists(dst):
            os.unlink(dst)
        last = _replace_from_tmp(dst, write, ".mp4")
        if os.path.getsize(last) <= max_bytes:
            return last
    return last


def upload(path: str, name: str, *, client=None) -> str:
    """Upload a local input to the kie.ai temp host (free; idempotent from the
    protocol's point of view — a re-upload just yields another URL).

    *name* MUST be unique to the row (callers derive it from the row id): the
    host OVERWRITES a same-named file in a folder, so two rows uploading
    ``1.png`` at once would both be sent whichever landed last (the AICP
    captions "wrong words" bug of 2026-09-25). *client*: the caller's kie
    client (default :func:`kie_client`)."""
    return (client or kie_client()).upload_file(path, upload_path=UPLOAD_PATH, file_name=name)


def upload_name(prefix: str, row_id: str, path: str, n: Optional[int] = None) -> str:
    """A row-unique, alphanumeric upload name: ``<prefix><row id hex>[n<n>]<ext>``."""
    ext = os.path.splitext(path)[1].lower() or ".bin"
    return f"{prefix}{row_id.replace('-', '')}" + (f"n{n}" if n is not None else "") + ext
