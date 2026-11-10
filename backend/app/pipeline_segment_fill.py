"""Segment caption fill (AICP-358, docs/segment-caption-fill-design.md).

Rebuilds ONE swap segment of a done ``subtitle_removal`` run as its source
frames + a Fal VOID inpaint pasted back only inside the caption mask:

* ``segment_fill_prepare`` (free, run worker): cut the segment clip exactly as
  the pipeline cuts it, measure an inclusive caption box with the platform
  detector, build the per-frame mask, the VOID inputs and a preview sheet.
* :func:`start_fill` (paid, called by the API under project → run → fill row
  locks): the operator's approval becomes a provider task.
* ``segment_fill_composite`` (free, run worker): paste the VOID frames back
  inside the mask over the clip, then promote the composite as the segment's
  result and requeue the run for the normal stitch + in-place delivery.
* :func:`reconcile_fill_jobs` (recovery tick): the only trigger for the
  composite, plus crash recovery and terminal cleanup.

A fill has a *basis* (frozen in ``base["basis"]``): ``model_result`` (the
AICP-358 case above: a done run's model output is replaced) or ``source``
(a swap segment the video provider refused with a content filter -- see
``app.content_block`` -- on an ``incomplete`` run: there is no model result,
the composite IS the segment's first result). The media path is the same
(the composite is always built on the SOURCE clip); a source fill whose clip
shows no caption is ``source_as_is`` and is applied for free by
:func:`apply_source_as_is` (no Fal call). A source promotion requeues the run
for its stitch + first delivery only once every active swap segment has a
finished result and the requeue would bill nothing (:func:`_apply_source_result`).

Every work file lives in an immutable claim directory
``runs/<rid>/segment_fills/<job>/<phase>-<token>/``; the winner's paths are
written to the row under the row lock and a claim check, never renamed.
Frames are streamed one at a time (whole-video numpy hung the VPS twice on
2026-10-30).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from typing import Iterator, Optional

import cv2
import numpy as np
from sqlalchemy import select, update

from . import fal_models, storage
from .config import settings
from .db import get_session
from .media import FFMPEG, FFPROBE
from .models import Run, RunSegment, VideoProject
from .provider_models import ProviderTask
from .segment_fill_models import (BATCH_LIVE, FILL_ACTIVE, BatchStatus, FillStatus,
                                  RunSegmentFillBatch, RunSegmentFillJob)
from .state_machine import ProviderTaskStatus

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
VOID_MIN_FRAMES = 69
VOID_MAX_FRAMES = 197
VOID_W, VOID_H = 720, 1280
VOID_RESOLUTION = "720p"
MIN_FREE_BYTES = 3 * 1024 ** 3
READY_TTL = timedelta(days=7)
BOX_PAD_X, BOX_PAD_Y = 0.03, 0.02
MIN_BOX_SIDE = 0.02
BASIS_MODEL = "model_result"
BASIS_SOURCE = "source"
BASES = (BASIS_MODEL, BASIS_SOURCE)
SOURCE_SHEET_FRAMES = 10
DEFAULT_PROMPT = ("Clean footage with no text and no captions. Keep the person, hands, "
                  "phone and background unchanged.")
# A preview frozen with this bind compares its own SEGMENT's inputs, not the
# delivered final (batch fills, docs/segment-fill-batch-design.md §1.2).
BIND_SEGMENT = "segment"
_SEGMENT_BOUND_SKIP = {("run", "final_revision"), ("run", "status")}
MAX_GROW_PX = 24


class FillError(Exception):
    """A deterministic refusal: the job ends ``failed`` with this code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def lease_sec() -> int:
    return int(getattr(settings, "SEGMENT_FILL_LEASE_SEC", 1800))


def fill_root(run_id: str, project_id: str) -> str:
    return os.path.join(storage.run_dir(run_id, project_id), "segment_fills")


def _job_dir(job: RunSegmentFillJob) -> str:
    return os.path.join(fill_root(job.run_id, job.project_id), job.id)


# ---------------------------------------------------------------------------
# The frozen base (design §4)
# ---------------------------------------------------------------------------


def file_identity(path: Optional[str]) -> Optional[list]:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def snapshot_entry(run: Run, rs: RunSegment) -> Optional[dict]:
    for sd in run.segment_snapshot or []:
        if sd.get("id") == rs.segment_def_id:
            return sd
    return None


def fill_basis(job_or_base) -> str:
    """The basis of a fill row (or of its frozen base): a row written before
    the basis existed is ``model_result``; an unknown value fails closed."""
    base = job_or_base if isinstance(job_or_base, dict) else (job_or_base.base or {})
    basis = base.get("basis") or BASIS_MODEL
    if basis not in BASES:
        raise FillError("bad_basis", f"unknown fill basis {basis!r}")
    return basis


def capture_base(db, run: Run, rs: RunSegment, project: VideoProject,
                 basis: str = BASIS_MODEL, bind: Optional[str] = None) -> dict:
    """The frozen base of a fill. A ``model_result`` base is exactly what
    AICP-358 froze (rows in flight at a deploy compare equal); a ``source``
    base also freezes the provider's refusal (message + category) and the
    caption detector the mask / the no-caption verdict was measured with."""
    from . import content_block
    from .api_v2 import _current_result_is_audio_patch
    from .assistant_run_result import result_revision

    if basis not in BASES:
        raise FillError("bad_basis", f"unknown fill basis {basis!r}")
    sd = snapshot_entry(run, rs) or {}
    patch = _current_result_is_audio_patch(run, db)
    status = rs.status.value if hasattr(rs.status, "value") else str(rs.status)
    run_status = run.status.value if hasattr(run.status, "value") else str(run.status)
    out = {
        "basis": basis,
        "segment": {"id": rs.id, "local_result_path": rs.local_result_path,
                    "identity": file_identity(rs.local_result_path),
                    "seedance_task_id": rs.seedance_task_id, "status": status,
                    "source_fallback": bool(rs.source_fallback),
                    "start_sec": sd.get("start_sec"), "end_sec": sd.get("end_sec")},
        "run": {"status": run_status, "stitch_plan_sha": _sha(run.stitch_plan),
                "segment_snapshot_sha": _sha(run.segment_snapshot),
                "audio_mode": run.audio_mode, "final_revision": result_revision(run)},
        "source": {"path": project.source_local_path,
                   "identity": file_identity(project.source_local_path)},
        "audio_patch_id": patch.id if patch is not None else None,
    }
    if basis == BASIS_SOURCE:
        out["segment"]["error_message"] = rs.error_message
        out["segment"]["block"] = content_block.segment_block(rs)
        out["source"]["detector_sha"] = _sha(project.caption_detector)
    if bind == BIND_SEGMENT:
        # Bound to the segment's own inputs (+ the Drive id the in-place
        # delivery keeps): another segment's fill re-stitching the final does
        # not make this preview stale (design §1.2). A base without "bind"
        # is exactly what AICP-358 froze.
        out["bind"] = BIND_SEGMENT
        out["run"]["drive_file_id"] = run.result_gdrive_file_id
    return out


def base_bind(job_or_base) -> Optional[str]:
    base = job_or_base if isinstance(job_or_base, dict) else (job_or_base.base or {})
    return base.get("bind")


def live_base(db, run: Run, rs: RunSegment, project: VideoProject, job) -> dict:
    """The live base to compare *job*'s frozen one with (same basis, same bind)."""
    return capture_base(db, run, rs, project, fill_basis(job), bind=base_bind(job))


def base_mismatch(frozen: dict, live: dict) -> Optional[str]:
    """The first part of the base that changed, or None. A segment-bound base
    ignores the delivered final's revision and the run status (the gates
    check the status on their own)."""
    skip = _SEGMENT_BOUND_SKIP if frozen.get("bind") == BIND_SEGMENT else set()
    if frozen.get("bind") != live.get("bind"):
        return "bind"
    for part in ("segment", "run", "source"):
        a, b = frozen.get(part) or {}, live.get(part) or {}
        for k in sorted(set(a) | set(b)):
            if (part, k) in skip:
                continue
            if a.get(k) != b.get(k):
                return f"{part}.{k}"
    if frozen.get("audio_patch_id") != live.get("audio_patch_id"):
        return "audio_patch_id"
    return None


# ---------------------------------------------------------------------------
# Media helpers (streamed)
# ---------------------------------------------------------------------------


def probe_video(path: str) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,sample_aspect_ratio,nb_read_packets"
         ":stream_side_data=rotation:stream_tags=rotate",
         "-of", "json", path],
        capture_output=True, text=True, timeout=300, check=False,
    )
    try:
        st = json.loads(out.stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        raise FillError("media_unreadable", f"ffprobe could not read {os.path.basename(path)}")
    rot = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if (st.get("tags") or {}).get("rotate"):
        rot = int(float(st["tags"]["rotate"]))
    return {"w": int(st["width"]), "h": int(st["height"]), "r_fps": st.get("r_frame_rate"),
            "avg_fps": st.get("avg_frame_rate"), "sar": st.get("sample_aspect_ratio") or "1:1",
            "rotation": rot, "frames": int(st.get("nb_read_packets") or 0)}


def check_supported(info: dict) -> None:
    w, h = info["w"], info["h"]
    if info["rotation"] % 360 != 0:
        raise FillError("unsupported_media", "the source is rotated; not supported yet")
    if info["sar"] not in ("1:1", "0:1", "N/A"):
        raise FillError("unsupported_media", f"non-square pixels ({info['sar']}) not supported")
    if w * 16 != h * 9:
        raise FillError("unsupported_media", f"only 9:16 sources are supported ({w}x{h})")
    try:
        r, a = Fraction(info["r_fps"]), Fraction(info["avg_fps"])
    except (TypeError, ValueError, ZeroDivisionError):
        raise FillError("unsupported_media", "the frame rate is unreadable")
    if r <= 0 or abs(float(r) - float(a)) > 0.01:
        raise FillError("unsupported_media", "variable frame rate is not supported")


def iter_frames(path: str, w: int, h: int) -> Iterator[np.ndarray]:
    """Yield BGR frames of *path* at its own rate, scaled to w×h, one at a time."""
    cmd = [FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        if p.poll() is None:
            p.kill()
        rc = p.wait()
    if rc not in (0, -9):
        raise FillError("media_unreadable", f"ffmpeg could not decode {os.path.basename(path)}")


class Encoder:
    def __init__(self, path: str, w: int, h: int, fps: str, crf: str = "14"):
        self.path, self.w, self.h = path, w, h
        self.count = 0
        self.p = subprocess.Popen(
            [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", crf,
             "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def write(self, im: np.ndarray) -> None:
        if im.shape[1] != self.w or im.shape[0] != self.h:
            im = cv2.resize(im, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.p.stdin.write(np.ascontiguousarray(im).tobytes())
        self.count += 1

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise FillError("encode_failed", f"ffmpeg could not encode {os.path.basename(self.path)}")


# ---------------------------------------------------------------------------
# Caption mask (port of /root/omni11-eval/inpaint/caption_mask.py, box as input)
# ---------------------------------------------------------------------------

_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
_NEAR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
_GROW = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def text_mask(frame_bgr: np.ndarray, box) -> np.ndarray:
    """uint8 0/1 mask over the full frame: white glyphs at least 55 % ringed by
    the black stroke inside *box*, closed between letters, plus the stroke and
    an anti-aliasing margin. Scales the kernels for frames other than 1080 wide."""
    h, w = frame_bgr.shape[:2]
    bx0, by0 = int(box[0] * w), int(box[1] * h)
    bx1, by1 = int(math.ceil(box[2] * w)), int(math.ceil(box[3] * h))
    sub = frame_bgr[by0:by1, bx0:bx1].astype(np.int16)
    full = np.zeros((h, w), np.uint8)
    if sub.size == 0:
        return full
    mn, mx = sub.min(axis=2), sub.max(axis=2)
    white = ((mn > 195) & (mx - mn < 45)).astype(np.uint8)
    black = (mx < 75).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    keep = np.zeros_like(white)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < 40 or ch > 0.6 * sub.shape[0]:
            continue
        x0, y0 = max(x - 6, 0), max(y - 6, 0)
        x1, y1 = min(x + cw + 6, sub.shape[1]), min(y + ch + 6, sub.shape[0])
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(comp, _K7) & (1 - comp)
        if ring.sum() == 0:
            continue
        if (ring & black[y0:y1, x0:x1]).sum() / ring.sum() > 0.55:
            keep[y0:y1, x0:x1] |= comp
    if keep.any():
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, _CLOSE)
        stroke = (mx < 110).astype(np.uint8) & cv2.dilate(keep, _NEAR)
        full[by0:by1, bx0:bx1] = cv2.dilate(cv2.dilate(keep, kd) | stroke, _GROW)
    return full


def validate_box(box) -> list[float]:
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        raise FillError("bad_box", "mask_box must be four numbers") from None
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        raise FillError("bad_box", "mask_box must be four finite numbers")
    x0, y0, x1, y1 = vals
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise FillError("bad_box", "mask_box must be ordered fractions of the frame")
    if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
        raise FillError("bad_box", "mask_box is too small")
    return [round(v, 4) for v in vals]


# Centred captions (run d46bb1b8 seg 6): the detector's per-frame box is ONE
# cluster -- a word ("KNOWING" of "KNOWING NOBODY.") -- so the union of a
# segment's boxes can miss the line's last word and the fill leaves it ("Y."
# at x 0.76-0.82 outside a box ending at 0.76). The box is therefore built
# from each hit frame's caption LINE (qa_tools.CaptionFrame.line) and, when
# the captions are horizontally centred, mirrored about the frame's centre
# (x0' = min(x0, 1-x1), x1' = max(x1, 1-x0)): a word the detector never saw
# on one side has its counterpart on the other. "Centred" is measured, never
# assumed: at least CENTRED_SHARE of at least CENTRED_MIN_LINES hit lines have
# their centre within CENTRE_TOL of 0.5. Left-aligned or side captions share
# an edge, not a centre, so their line centres scatter with the line length
# and their box is not widened. The VOID mask inside the box stays the glyph
# mask (text_mask), so a wider box repaints no non-caption pixels.
CENTRE_TOL = 0.03
CENTRED_SHARE = 0.8
CENTRED_MIN_LINES = 5


def _frame_line(f):
    """A scan frame's caption line, else its box (frames built without lines)."""
    return getattr(f, "line", None) or getattr(f, "box", None)


def captions_centred(frames) -> bool:
    """Are the hit lines of *frames* (qa_tools.CaptionFrame) horizontally
    centred on the frame? See ``CENTRE_TOL``."""
    lines = [ln for ln in (_frame_line(f) for f in frames) if ln is not None]
    if len(lines) < CENTRED_MIN_LINES:
        return False
    near = sum(1 for ln in lines if abs((float(ln[0]) + float(ln[2])) / 2 - 0.5) <= CENTRE_TOL)
    return near >= CENTRED_SHARE * len(lines)


def mirror_x(box, centred: bool) -> list[float]:
    """*box* made symmetric about the frame's vertical centre line when
    *centred* (never narrower), else unchanged."""
    x0, y0, x1, y1 = (float(v) for v in box)
    if centred:
        x0, x1 = min(x0, 1.0 - x1), max(x1, 1.0 - x0)
    return [x0, y0, x1, y1]


def caption_box(frames, pad_x: float = BOX_PAD_X, pad_y: float = BOX_PAD_Y
                ) -> Optional[list[float]]:
    """Inclusive fill box from scan *frames*: the UNION of the hit frames'
    caption lines (not percentiles: those cut off wide words), padded,
    mirrored when the captions are centred, validated (FillError). None
    without a hit."""
    lines = [ln for ln in (_frame_line(f) for f in frames) if ln is not None]
    if not lines:
        return None
    a = np.array(lines, dtype=float)
    box = [float(a[:, 0].min()) - pad_x, float(a[:, 1].min()) - pad_y,
           float(a[:, 2].max()) + pad_x, float(a[:, 3].max()) + pad_y]
    x0, y0, x1, y1 = mirror_x(box, captions_centred(frames))
    return validate_box([max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)])


def measure_box(clip: str, fps: float, detector_config) -> Optional[list[float]]:
    """Inclusive box over every frame of the clip (:func:`caption_box`)."""
    from . import qa_tools

    scan = qa_tools.scan_captions(clip, fps=fps, config=detector_config)
    return caption_box(scan.scores)


def void_sampling(n: int) -> list[list[int]]:
    """Source frame interval [a, b] for each VOID input frame: the largest valid
    count 69 + 8k ≤ min(n, 197), spread evenly over the clip."""
    if n < VOID_MIN_FRAMES:
        raise FillError("segment_too_short",
                        f"the segment has {n} frames; VOID needs at least {VOID_MIN_FRAMES}")
    m = VOID_MIN_FRAMES + 8 * ((min(n, VOID_MAX_FRAMES) - VOID_MIN_FRAMES) // 8)
    out = []
    for k in range(m):
        a = min(int(math.floor(k * n / m)), n - 1)
        b = max(a, min(int(math.floor((k + 1) * n / m)) - 1, n - 1))
        out.append([a, b])
    return out


def frame_to_void(n: int, sampling: list[list[int]]) -> list[int]:
    """For every clip frame, the VOID frame whose interval holds it."""
    idx = [0] * n
    for k, (a, b) in enumerate(sampling):
        for i in range(a, b + 1):
            idx[i] = k
    return idx


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def _claim(job_id: str, phase_status: str, *, from_status: Optional[str] = None
           ) -> Optional[tuple[str, RunSegmentFillJob]]:
    """CAS the row into *phase_status* with a fresh token when it is unclaimed,
    its claim expired, or (composite) it is still ``admitted``."""
    token = uuid.uuid4().hex
    now = _now()
    with get_session() as s:
        job = s.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == job_id)
                        .with_for_update()).scalar_one_or_none()
        if job is None:
            return None
        ok_status = job.status == phase_status or (from_status and job.status == from_status)
        if not ok_status:
            return None
        if job.status == phase_status and job.claim_token and job.claim_expires_at \
                and job.claim_expires_at > now:
            return None
        if job.status == from_status:
            job.attempts = 0
        if job.attempts >= MAX_ATTEMPTS:
            return None
        job.status = phase_status
        job.claim_token = token
        job.claim_expires_at = now + timedelta(seconds=lease_sec())
        job.attempts = job.attempts + 1
        s.commit()
        s.refresh(job)
        s.expunge(job)
        return token, job


def _still_ours(s, job_id: str, status: str, token: str) -> Optional[RunSegmentFillJob]:
    job = s.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == job_id)
                    .with_for_update()).scalar_one_or_none()
    if job is None or job.status != status or job.claim_token != token:
        return None
    if job.claim_expires_at is None or job.claim_expires_at <= _now():
        return None
    return job


def _fail(job_id: str, status: str, token: Optional[str], code: str, message: str) -> None:
    with get_session() as s:
        q = update(RunSegmentFillJob).where(RunSegmentFillJob.id == job_id,
                                            RunSegmentFillJob.status == status)
        if token is not None:
            q = q.where(RunSegmentFillJob.claim_token == token)
        n = s.execute(q.values(status=FillStatus.failed.value, error_code=code,
                               error_message=message[:500], claim_token=None,
                               claim_expires_at=None)).rowcount
        s.commit()
    if n:
        log.warning("segment_fill %s: failed (%s) %s", job_id, code, message)


# ---------------------------------------------------------------------------
# Prepare (free)
# ---------------------------------------------------------------------------


def segment_fill_prepare(job_id: str) -> None:
    claimed = _claim(job_id, FillStatus.preparing.value)
    if claimed is None:
        log.info("segment_fill_prepare %s: not claimable — nothing to do", job_id)
        return
    token, job = claimed
    with get_session() as s:
        run = s.get(Run, job.run_id)
        rs = s.get(RunSegment, job.run_segment_id)
        project = s.get(VideoProject, job.project_id)
        if run is None or rs is None or project is None:
            log.info("segment_fill_prepare %s: run/segment/project gone — canceled", job_id)
            return
        sd = snapshot_entry(run, rs)
        source = project.source_local_path
        detector = project.caption_detector
    root = _job_dir(job)
    if not os.path.isdir(storage.run_dir(job.run_id, job.project_id)):
        return  # the run was deleted: never recreate its directory
    work = os.path.join(root, f"prep-{token}")
    try:
        os.makedirs(work, exist_ok=False)
        if _drop_if_run_deleted(job):
            return
        _prepare_into(job, work, sd, source, detector, token)
    except FillError as exc:
        shutil.rmtree(work, ignore_errors=True)
        _fail(job_id, FillStatus.preparing.value, token, exc.code, exc.message)
    except Exception as exc:  # unexpected: leave the claim to expire → recovery retries
        shutil.rmtree(work, ignore_errors=True)
        log.exception("segment_fill_prepare %s: crashed: %s", job_id, exc)
        if job.attempts >= MAX_ATTEMPTS:
            _fail(job_id, FillStatus.preparing.value, token, "prepare_crashed",
                  f"{type(exc).__name__}: {exc}")


def _drop_if_run_deleted(job: RunSegmentFillJob) -> bool:
    """A run deleted while we created files under it: remove what we recreated
    (the run's whole directory is already meant to be gone)."""
    with get_session() as s:
        alive = s.get(Run, job.run_id) is not None
        project_alive = s.get(VideoProject, job.project_id) is not None
    if not alive:
        target = (storage.run_dir(job.run_id, job.project_id) if project_alive
                  else os.path.join(storage.data_dir_root(), "projects", job.project_id))
        shutil.rmtree(target, ignore_errors=True)
        log.info("segment_fill %s: run %s was deleted meanwhile — files removed",
                 job.id, job.run_id)
    return not alive


def _prepare_into(job: RunSegmentFillJob, work: str, sd: Optional[dict], source: Optional[str],
                  detector, token: str) -> None:
    from . import media as media_mod

    basis = fill_basis(job)
    if basis == BASIS_SOURCE and _sha(detector) != (job.base.get("source") or {}).get(
            "detector_sha"):
        # The mask / the no-caption verdict must come from the detector the
        # preview froze (design R2-5): a change in between is a new preview.
        raise FillError("base_changed", "the caption detector changed since the preview")
    if not sd or sd.get("action") != "swap":
        raise FillError("not_swap", "the segment is not a swap segment of the run's snapshot")
    if (sd.get("pre_roll_sec") or 0) != 0 or (sd.get("post_roll_sec") or 0) != 0:
        raise FillError("unsupported_media", "segments with pre/post roll are not supported")
    if not source or file_identity(source) != (job.base.get("source") or {}).get("identity"):
        raise FillError("base_changed", "the project source changed since the preview")
    if shutil.disk_usage(work).free < MIN_FREE_BYTES:
        raise FillError("insufficient_disk", "less than 3 GiB free on the data volume")
    src_info = probe_video(source)
    check_supported(src_info)

    # The pipeline's own cut, so the composite sits on the frame grid an Omni
    # result is stitched on (pipeline_v2: cut_clip(source, start, end, dst, include_audio=False)).
    clip = os.path.join(work, "clip.mp4")
    media_mod.cut_clip(source, sd["start_sec"], sd["end_sec"], clip, include_audio=False)
    info = probe_video(clip)
    n, w, h, fps_str = info["frames"], info["w"], info["h"], info["r_fps"]
    fps = float(Fraction(fps_str))
    if basis == BASIS_SOURCE:
        # Measured FIRST: a source clip without captions needs no VOID call at
        # all, so the VOID frame minimum must not refuse it (design R1-2).
        if job.mask_box:
            box, box_source = validate_box(job.mask_box), "request"
        else:
            box, box_source = measure_box(clip, fps, detector), "measured"
        if box is None:
            return _publish_source_as_is(job, work, token, clip, info, detector, None, "none",
                                         "the detector found no caption in the source clip")
    sampling = void_sampling(n)
    m = len(sampling)

    if basis == BASIS_SOURCE:
        pass
    elif job.mask_box:
        box, box_source = validate_box(job.mask_box), "request"
    else:
        box, box_source = measure_box(clip, fps, detector), "measured"
        if box is None:
            raise FillError("no_captions_detected",
                            "the detector found no captions in this segment; send a mask_box")

    # Pass 1: raw per-frame masks to a memmap; pass 2: OR with t±1.
    raw = np.lib.format.open_memmap(os.path.join(work, "masks_raw.npy"), "w+", np.uint8, (n, h, w))
    k = -1
    grow = int(job.mask_grow_px or 0)
    k_grow = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
              if grow > 0 else None)
    for k, f in enumerate(iter_frames(clip, w, h)):
        if k >= n:
            break
        mk = text_mask(f, box)
        if k_grow is not None and mk.any():
            # The stop rule's grown retry (design §4.3): the glyph mask is
            # dilated further, still clipped to the (grown) box.
            bx0, by0 = int(box[0] * w), int(box[1] * h)
            bx1, by1 = int(math.ceil(box[2] * w)), int(math.ceil(box[3] * h))
            g = cv2.dilate(mk, k_grow)
            clipped = np.zeros_like(g)
            clipped[by0:by1, bx0:bx1] = g[by0:by1, bx0:bx1]
            mk = clipped
        raw[k] = mk
    if k + 1 != n:
        raise FillError("media_unreadable", f"decoded {k + 1} frames, expected {n}")
    masks = np.lib.format.open_memmap(os.path.join(work, "masks.npy"), "w+", np.uint8, (n, h, w))
    area = []
    for i in range(n):
        mm = np.array(raw[i])
        if i > 0:
            mm |= raw[i - 1]
        if i + 1 < n:
            mm |= raw[i + 1]
        masks[i] = mm
        area.append(int(mm.sum()))
    masks.flush()
    del raw
    os.remove(os.path.join(work, "masks_raw.npy"))
    masked = sum(1 for a in area if a > 0)
    if masked == 0:
        if basis == BASIS_SOURCE:
            del masks
            os.remove(os.path.join(work, "masks.npy"))
            return _publish_source_as_is(
                job, work, token, clip, info, detector, box, box_source,
                "the detector saw caption-like text, but no caption glyph is inside its box")
        raise FillError("no_captions_detected", "the mask is empty inside the box")

    # VOID inputs: m sampled frames + the inverted union mask of each interval.
    dur = n / fps
    void_fps = f"{m}/{dur:.6f}"
    vv = Encoder(os.path.join(work, "void_v.mp4"), VOID_W, VOID_H, void_fps, crf="12")
    vm = Encoder(os.path.join(work, "void_m.mp4"), VOID_W, VOID_H, void_fps, crf="12")
    starts = {a: q for q, (a, _b) in enumerate(sampling)}
    for i, f in enumerate(iter_frames(clip, w, h)):
        if i >= n:
            break
        q = starts.get(i)
        if q is None:
            continue
        a, b = sampling[q]
        u = np.zeros((h, w), np.uint8)
        for j in range(a, b + 1):
            u |= masks[j]
        small = cv2.resize(u.astype(np.float32), (VOID_W, VOID_H), interpolation=cv2.INTER_AREA) > 0
        keep = np.where(small, 0, 255).astype(np.uint8)  # VOID: 0 = remove, 255 = keep
        vv.write(f)
        vm.write(cv2.cvtColor(keep, cv2.COLOR_GRAY2BGR))
    vv.close()
    vm.close()
    if vv.count != m or vm.count != m:
        raise FillError("encode_failed", f"wrote {vv.count}/{vm.count} VOID frames, expected {m}")

    preview = os.path.join(work, "preview.jpg")
    _preview_sheet(clip, w, h, box, masks, area, fps, preview)
    est = fal_models.endpoint(fal_models.VOID_ENDPOINT).worst_case_usd(VOID_RESOLUTION, dur)
    media = {"frames": n, "fps": fps_str, "w": w, "h": h, "duration_sec": round(dur, 4),
             "frames_masked": masked, "void_frames": m, "void_fps": void_fps,
             "sampling": sampling}

    with get_session() as s:
        row = _still_ours(s, job.id, FillStatus.preparing.value, token)
        if row is None:
            s.rollback()
            shutil.rmtree(work, ignore_errors=True)
            _drop_if_run_deleted(job)
            log.info("segment_fill_prepare %s: lost the claim before publishing", job.id)
            return
        row.status = FillStatus.ready.value
        row.prep_dir, row.preview_path, row.media = work, preview, media
        row.mask_box, row.box_source, row.estimated_usd = box, box_source, est
        row.claim_token = row.claim_expires_at = None
        row.attempts = 0
        s.commit()
    log.info("segment_fill_prepare %s: ready (%d/%d frames masked, VOID %d frames, box %s)",
             job.id, masked, n, m, box)


def _publish_source_as_is(job: RunSegmentFillJob, work: str, token: str, clip: str,
                          info: dict, detector, box, box_source: str, reason: str) -> None:
    """A source fill whose clip shows no caption to remove: ``ready`` with
    ``media.source_as_is`` and no VOID inputs; applying it (free, the
    operator's approval) puts the source clip in as the segment's result."""
    from . import qa_tools

    n, w, h, fps_str = info["frames"], info["w"], info["h"], info["r_fps"]
    fps = float(Fraction(fps_str))
    try:
        band = list(qa_tools.caption_config(detector).band_y)
    except Exception:  # the detector config was validated when it was set
        band = None
    preview = os.path.join(work, "preview.jpg")
    _source_sheet(clip, w, h, n, fps, band, box, preview)
    media = {"frames": n, "fps": fps_str, "w": w, "h": h, "duration_sec": round(n / fps, 4),
             "frames_masked": 0, "source_as_is": True, "source_as_is_reason": reason,
             "band_y": band}
    with get_session() as s:
        row = _still_ours(s, job.id, FillStatus.preparing.value, token)
        if row is None:
            s.rollback()
            shutil.rmtree(work, ignore_errors=True)
            _drop_if_run_deleted(job)
            log.info("segment_fill_prepare %s: lost the claim before publishing", job.id)
            return
        row.status = FillStatus.ready.value
        row.prep_dir, row.preview_path, row.media = work, preview, media
        row.mask_box, row.box_source, row.estimated_usd = box, box_source, 0.0
        row.claim_token = row.claim_expires_at = None
        row.attempts = 0
        s.commit()
    log.info("segment_fill_prepare %s: ready as the source clip as is (%s)", job.id, reason)


def _source_sheet(clip, w, h, n, fps, band, box, out) -> None:
    """Evenly spaced frames of the source clip (streamed), the detector band
    (and a measured box, if any) outlined: what "the source as is" ships."""
    k = min(SOURCE_SHEET_FRAMES, n)
    want = sorted({min(n - 1, int(round(i * (n - 1) / max(1, k - 1)))) for i in range(k)})
    cells, top = [], max(want)
    pick = set(want)
    for i, f in enumerate(iter_frames(clip, w, h)):
        if i in pick:
            im = f.copy()
            if band:
                cv2.rectangle(im, (0, int(band[0] * h)), (w - 1, int(band[1] * h)),
                              (255, 160, 0), 3)
            if box:
                cv2.rectangle(im, (int(box[0] * w), int(box[1] * h)),
                              (int(box[2] * w), int(box[3] * h)), (0, 255, 255), 3)
            cv2.putText(im, f"t={i / fps:.2f}s", (20, 60), 0, 2, (0, 255, 255), 4)
            cells.append(cv2.resize(im, (w // 4, h // 4), interpolation=cv2.INTER_AREA))
        if i >= top:
            break
    if not cells:
        raise FillError("media_unreadable", "no frame of the clip could be decoded")
    while len(cells) % 5:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[r:r + 5]) for r in range(0, len(cells), 5)])
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _preview_sheet(clip, w, h, box, masks, area, fps, out) -> None:
    n = len(area)
    nz = [i for i, a in enumerate(area) if a > 0]
    picks = sorted(set([nz[0], nz[-1]] + sorted(nz, key=lambda i: -area[i])[:8]))[:10]
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = int(box[2] * w), int(box[3] * h)
    cells = {}
    want = set(picks)
    for i, f in enumerate(iter_frames(clip, w, h)):
        if i in want:
            im = f.copy()
            m = np.array(masks[i]) > 0
            im[m] = (0.55 * im[m] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 255), 3)
            cv2.putText(im, f"t={i / fps:.2f}s", (20, 60), 0, 2, (0, 255, 255), 4)
            cells[i] = cv2.resize(im, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        if i >= max(want):
            break
    row = [cells[i] for i in picks if i in cells]
    while len(row) % 5:
        row.append(np.zeros_like(row[0]))
    grid = np.vstack([np.hstack(row[r:r + 5]) for r in range(0, len(row), 5)])
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------------------
# Start (paid) — the caller holds project → run → fill row locks
# ---------------------------------------------------------------------------


def admit_provider_task(db, job: RunSegmentFillJob, run: Run, *, reserve: bool = True):
    """Admit the VOID task for a ``ready`` job inside the caller's transaction
    (*reserve=False*: the batch reserves every member at once)."""
    from . import pipeline_provider as pp

    media = job.media or {}
    dest_dir = _job_dir(job)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"raw-{uuid.uuid4().hex}.mp4")
    params = {"prompt": job.prompt or DEFAULT_PROMPT, "num_frames": int(media["void_frames"]),
              "enable_pass2_refinement": False}
    return pp.admit(
        db, endpoint_id=fal_models.VOID_ENDPOINT, params=params,
        inputs=[pp.TaskInput("video_url", os.path.join(job.prep_dir, "void_v.mp4"), "video/mp4"),
                pp.TaskInput("quad_mask_video_url", os.path.join(job.prep_dir, "void_m.mp4"),
                             "video/mp4")],
        output_sec=float(media["duration_sec"]), resolution=VOID_RESOLUTION,
        owner_kind="segment_fill", owner_id=job.id, owner_project_id=job.project_id,
        owner_run_id=run.id, destination_path=dest, operation_key=f"segment_fill:{job.id}",
        output_frames=int(media["void_frames"]), reserve=reserve,
    )


def _basis_or_409(job: RunSegmentFillJob) -> str:
    from fastapi import HTTPException

    try:
        return fill_basis(job)
    except FillError as exc:
        raise HTTPException(status_code=409, detail=exc.message) from None


class StalePreview(Exception):
    """``admit_start``: the ready preview no longer matches the run (*part*
    changed). Nothing was written; the endpoint cancels the job itself.
    *fill_id* names the stale member of a batch start."""

    def __init__(self, part: str, fill_id: Optional[str] = None):
        self.part = part
        self.fill_id = fill_id
        what = f"fill {fill_id}: " if fill_id else ""
        super().__init__(f"{what}{part} changed since the preview; make a new preview")


def admit_start(db, run: Run, fill_id: str, *, approved_usd: float,
                discard_audio_patch: bool = False):
    """Every check and write of a paid fill start in the CALLER's transaction
    (AICP-364): never commits, never rolls back (an autopilot admission holds
    its own pending effect row in the same transaction). Refusals raise
    ``HTTPException`` (409/422/403/503) or :class:`StalePreview` before
    anything is written, except an admission refusal of the provider task or
    the daily budget, whose partial writes the CALLER rolls back (the endpoint
    does; the autopilot core rolls back the whole admission). Returns
    ``(job, task, created)``; the caller commits, then enqueues ``fal_submit``
    when *created*.

    Lock order project -> run -> fill row (``pipeline_provider._lock_owner``'s).
    """
    from fastapi import HTTPException
    from sqlalchemy.exc import IntegrityError

    from . import pipeline_provider as pp
    from . import spend
    from .api_segment_fill import _require_fillable
    from .api_v2 import (_admit_segment_rerun, _check_segment_rerun_conflicts,
                         _require_audio_patch_discard_ack)

    if not math.isfinite(approved_usd) or approved_usd < 0:
        raise HTTPException(status_code=422, detail="approved_usd must be a finite amount ≥ 0")
    peek = db.get(RunSegmentFillJob, fill_id)
    if peek is not None and peek.run_id == run.id and _basis_or_409(peek) == BASIS_MODEL:
        # 2026-11-03: a model-result start is a batch of one (one promote path
        # for every new fill). A source fill keeps the legacy single path
        # below (design R1-1): its promotion requeues only a complete run.
        batch, members = admit_batch_start(
            db, run, [fill_id], approved_usd=approved_usd,
            discard_audio_patch=discard_audio_patch)
        job, task, created = members[0]
        return job, task, created
    project = db.execute(select(VideoProject).where(VideoProject.id == run.project_id)
                         .with_for_update()).scalar_one_or_none()
    # Our own row is not active yet, so the Run lock's active-fill check only
    # sees ANOTHER fill (a legacy one or a batch member).
    _admit_segment_rerun(run, db)
    job = db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == fill_id)
                     .with_for_update()).scalar_one_or_none()
    if job is None or job.run_id != run.id:
        raise HTTPException(status_code=404, detail="fill job not found in this run")
    if job.status != FillStatus.ready.value:
        raise HTTPException(status_code=409, detail=f"the fill is {job.status}, not ready")
    basis = _basis_or_409(job)
    if (job.media or {}).get("source_as_is"):
        raise HTTPException(status_code=409, detail=(
            "this preview found no caption to remove: apply the source clip as is "
            "(apply-source, free) instead of starting a VOID fill"))
    rs = db.get(RunSegment, job.run_segment_id)
    _require_fillable(run, project, rs, basis)
    changed = base_mismatch(job.base, live_base(db, run, rs, project, job))
    if changed:
        raise StalePreview(changed)
    _check_segment_rerun_conflicts(run.id, db)
    _require_audio_patch_discard_ack(run, db, discard_audio_patch)
    # A source fill never re-stitches while another segment is unfinished
    # (_apply_source_result requeues only a complete, free run), so only the
    # model-result fill needs this check.
    missing = (paid_restitch_segments(db, run, exclude_rs_id=rs.id)
               if basis == BASIS_MODEL else [])
    if missing:
        raise HTTPException(status_code=409, detail=(
            f"segment(s) {missing} have no finished result: the re-stitch after the fill "
            "would generate them (paid). Finish or restore them first."))
    if job.estimated_usd is None or approved_usd + 1e-9 < job.estimated_usd:
        raise HTTPException(status_code=409, detail=(
            f"the reservation for this fill is ${job.estimated_usd}; approve at least that"))
    try:
        task, created = admit_provider_task(db, job, run)
    except pp.AdmissionRefused as exc:
        raise HTTPException(status_code=exc.status_code,
                            detail=f"Fal admission refused: {exc.detail}") from None
    except spend.BudgetExceeded as exc:
        raise HTTPException(status_code=409, detail=f"daily budget: {exc}") from None
    job.status = FillStatus.admitted.value
    job.provider_task_id = task.id
    job.approved_usd = float(approved_usd)
    job.discard_audio_patch = bool(discard_audio_patch)
    # The partial unique index "one active fill per run" fails HERE, inside a
    # savepoint, as a refusal -- not as a commit-time IntegrityError.
    sp = db.begin_nested()
    try:
        db.flush()
    except IntegrityError:
        sp.rollback()
        raise HTTPException(status_code=409,
                            detail="another fill is already active on this run") from None
    sp.commit()
    return job, task, created


class AllowanceExceeded(Exception):
    """An ``auto`` batch would go over the run's VOID allowance."""


def void_allowance_usd(project: VideoProject) -> float:
    """The run allowance of VOID reservations the autopilot may start without
    asking (design §3): the project's own value, else the setting."""
    v = getattr(project, "void_auto_usd_per_run", None)
    if v is None:
        v = getattr(settings, "AUTOPILOT_VOID_AUTO_USD", 1.0)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def auto_spent_usd(db, run_id: str) -> float:
    """Reservations of every ``auto`` batch ever admitted on this run, in USD
    (no refund for a member whose reservation was released: the allowance is
    a bound on what was STARTED without a question)."""
    from sqlalchemy import func

    v = db.execute(select(func.coalesce(func.sum(RunSegmentFillBatch.reserved_usd), 0.0))
                   .where(RunSegmentFillBatch.run_id == run_id,
                          RunSegmentFillBatch.authorization_kind == "auto")).scalar_one()
    return float(v or 0.0)


def live_batch(db, run_id: str) -> Optional[RunSegmentFillBatch]:
    return db.execute(select(RunSegmentFillBatch).where(
        RunSegmentFillBatch.run_id == run_id, RunSegmentFillBatch.status.in_(BATCH_LIVE)
    ).limit(1)).scalar_one_or_none()


def admit_batch_start(db, run: Run, fill_ids, *, approved_usd: float,
                      discard_audio_patch: bool = False, authorization: str = "operator"):
    """ONE paid start over several ready, model-result previews of *run*
    (docs/segment-fill-batch-design.md §1.3), in the CALLER's transaction:
    never commits, never rolls back (the ``admit_start`` contract). Refusals
    raise ``HTTPException`` / :class:`StalePreview` (``part`` names the fill)
    before anything is written, except the provider / daily-budget refusals,
    whose partial writes the caller rolls back.

    Lock order project -> run -> fill rows (id order). Every member's provider
    task is admitted WITHOUT its ledger reservation, then all of them are
    reserved with ONE ``spend.reserve_many`` (one day lock, the sum checked
    against the daily cap) as the last write (design R1-2). An ``auto``
    batch must fit the run's VOID allowance, checked here under the locks
    (R1-5). Returns ``(batch, [(job, task, created), ...])``; the caller
    commits, then enqueues ``fal_submit`` for every created task."""
    from fastapi import HTTPException
    from sqlalchemy.exc import IntegrityError

    from . import pipeline_provider as pp
    from . import spend
    from .api_segment_fill import _require_fillable
    from .api_v2 import (_admit_segment_rerun, _check_segment_rerun_conflicts,
                         _require_audio_patch_discard_ack)
    from .assistant_run_result import result_revision

    if authorization not in ("operator", "auto"):
        raise HTTPException(status_code=422, detail="authorization must be operator or auto")
    try:
        approved = float(approved_usd)
    except (TypeError, ValueError):
        approved = float("nan")
    if not math.isfinite(approved) or approved < 0:
        raise HTTPException(status_code=422, detail="approved_usd must be a finite amount ≥ 0")
    ids = list(fill_ids or [])
    cap = int(getattr(settings, "SEGMENT_FILL_BATCH_MAX", 12))
    if not ids or len(ids) > cap or len(set(ids)) != len(ids) \
            or not all(isinstance(i, str) for i in ids):
        raise HTTPException(status_code=422, detail=(
            f"a batch takes 1..{cap} distinct fill ids"))
    project = db.execute(select(VideoProject).where(VideoProject.id == run.project_id)
                         .with_for_update()).scalar_one_or_none()
    # The Run lock + its active-fill check: no other active fill / batch member.
    _admit_segment_rerun(run, db)
    if live_batch(db, run.id) is not None:
        raise HTTPException(status_code=409, detail="another fill batch is active on this run")
    jobs = []
    for fid in sorted(ids):
        job = db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == fid)
                         .with_for_update()).scalar_one_or_none()
        if job is None or job.run_id != run.id:
            raise HTTPException(status_code=404, detail=f"fill job {fid} not found in this run")
        jobs.append(job)
    if len({j.run_segment_id for j in jobs}) != len(jobs):
        raise HTTPException(status_code=409, detail="a batch takes one fill per segment")
    from .project_types import is_void_source_run

    bases = {_basis_or_409(j) for j in jobs}
    if len(bases) != 1:
        raise HTTPException(status_code=409, detail="a batch takes fills of one basis")
    basis = bases.pop()
    if basis == BASIS_SOURCE and not is_void_source_run(run):
        # A provider-refused segment of a model run keeps its own single start.
        raise HTTPException(status_code=409, detail=(
            "a fill from the source of a model run is started on its own"))
    for job in jobs:
        if job.status != FillStatus.ready.value:
            raise HTTPException(status_code=409,
                                detail=f"fill {job.id} is {job.status}, not ready")
        if (job.media or {}).get("source_as_is"):
            raise HTTPException(status_code=409, detail=(
                f"fill {job.id} found no caption: apply the source clip as is (free)"))
        rs = db.get(RunSegment, job.run_segment_id)
        _require_fillable(run, project, rs, basis)
        changed = base_mismatch(job.base, live_base(db, run, rs, project, job))
        if changed:
            raise StalePreview(changed, fill_id=job.id)
        if job.estimated_usd is None:
            raise HTTPException(status_code=409, detail=f"fill {job.id} has no estimate")
    _check_segment_rerun_conflicts(run.id, db)
    _require_audio_patch_discard_ack(run, db, discard_audio_patch)
    members_rs = {j.run_segment_id for j in jobs}
    # A source batch never re-stitches an unfinished run (its promote requeues
    # only a complete, free one), so only a model-result batch needs this.
    missing = (paid_restitch_segments(db, run, exclude_rs_ids=members_rs)
               if basis == BASIS_MODEL else [])
    if missing:
        raise HTTPException(status_code=409, detail=(
            f"segment(s) {missing} have no finished result: the re-stitch after the fills "
            "would generate them (paid). Finish or restore them first."))
    est_mc = sum(_usd_mc(j.estimated_usd) for j in jobs)
    if _usd_mc(approved) < est_mc:
        raise HTTPException(status_code=409, detail=(
            "the reservation for these %d fill(s) is $%.2f; approve at least that"
            % (len(jobs), est_mc / 1e5)))
    if authorization == "auto":
        allowance = void_allowance_usd(project)
        used = auto_spent_usd(db, run.id)
        if used + est_mc / 1e5 > allowance + 1e-9:
            raise AllowanceExceeded(
                "the run's VOID allowance is $%.2f; $%.2f already started without a question, "
                "this batch needs $%.2f" % (allowance, used, est_mc / 1e5))
    out = []
    for job in jobs:
        try:
            task, created = admit_provider_task(db, job, run, reserve=False)
        except pp.AdmissionRefused as exc:
            raise HTTPException(status_code=exc.status_code,
                                detail=f"Fal admission refused: {exc.detail}") from None
        if not created:
            raise HTTPException(status_code=409, detail=(
                f"fill {job.id} already has a provider task ({task.id})"))
        if task.estimated_usd is None or _usd_mc(task.estimated_usd) > _usd_mc(job.estimated_usd):
            # The amount the operator / the allowance saw must be what the
            # ledger reserves (R1-6): a priced-up task is a new preview.
            raise HTTPException(status_code=409, detail=(
                f"fill {job.id}: Fal prices it at ${task.estimated_usd}, the preview at "
                f"${job.estimated_usd}; make a new preview"))
        out.append((job, task, created))
    reserved_mc = sum(int(t.reserved_mcredits or 0) for _j, t, _c in out)
    reserved_usd = round(sum(float(t.estimated_usd or 0.0) for _j, t, _c in out), 6)
    batch = RunSegmentFillBatch(
        id=str(uuid.uuid4()), run_id=run.id, project_id=run.project_id,
        status=BatchStatus.active.value, basis=basis, authorization_kind=authorization,
        approved_usd=round(approved, 6), reserved_mcredits=reserved_mc,
        reserved_usd=reserved_usd, discard_audio_patch=bool(discard_audio_patch),
        base={"run": {"status": _status(run.status), "final_revision": result_revision(run),
                      "drive_file_id": run.result_gdrive_file_id,
                      "stitch_plan_sha": _sha(run.stitch_plan),
                      "segment_snapshot_sha": _sha(run.segment_snapshot)},
              "members": sorted(j.id for j in jobs)},
        attempts=0,
    )
    db.add(batch)
    for job, task, _created in out:
        job.status = FillStatus.admitted.value
        job.provider_task_id = task.id
        job.approved_usd = float(job.estimated_usd)
        job.discard_audio_patch = bool(discard_audio_patch)
        job.batch_id = batch.id
    # The one-live-batch-per-run index fails HERE, inside a savepoint, as a
    # refusal -- not as a commit-time IntegrityError.
    sp = db.begin_nested()
    try:
        db.flush()
    except IntegrityError:
        sp.rollback()
        raise HTTPException(status_code=409,
                            detail="another fill batch is already active on this run") from None
    sp.commit()
    try:
        spend.reserve_many(db, [(spend.provider_task_scope(t.id), t.reserved_mcredits / 1000.0,
                                 t.kind) for _j, t, _c in out], provider=pp.PROVIDER)
    except spend.BudgetExceeded as exc:
        raise HTTPException(status_code=409, detail=f"daily budget: {exc}") from None
    return batch, out


def _usd_mc(usd) -> int:
    """USD as an integer of 1e-5 dollars (sums without float drift)."""
    return int(round(float(usd or 0.0) * 1e5))


def _status(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def apply_source_as_is(db, run: Run, fill_id: str):
    """The operator's FREE "deliver the source clip as is" for a ``ready``
    source preview that found no caption (``media.source_as_is``): no Fal
    call, no worker -- the prepared source clip is copied into its own claim
    directory and promoted in this transaction (design R1-1/2).

    Lock order project -> run -> fill row. Refusals raise ``HTTPException`` or
    :class:`StalePreview` before anything is written to the database; the
    promotion COMMITS (:func:`_apply_source_result`), so a caller that must
    record something atomically with it (the autopilot effect's created_refs)
    writes it before calling. Returns the job."""
    from fastapi import HTTPException

    from .api_segment_fill import _require_fillable
    from .api_v2 import (_admit_segment_rerun, _check_segment_rerun_conflicts,
                         _require_audio_patch_discard_ack)

    project = db.execute(select(VideoProject).where(VideoProject.id == run.project_id)
                         .with_for_update()).scalar_one_or_none()
    _admit_segment_rerun(run, db)
    job = db.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.id == fill_id)
                     .with_for_update()).scalar_one_or_none()
    if job is None or job.run_id != run.id:
        raise HTTPException(status_code=404, detail="fill job not found in this run")
    if job.status != FillStatus.ready.value:
        raise HTTPException(status_code=409, detail=f"the fill is {job.status}, not ready")
    if _basis_or_409(job) != BASIS_SOURCE or not (job.media or {}).get("source_as_is"):
        raise HTTPException(status_code=409, detail=(
            "only a source preview that found no caption can be applied as is; "
            "start the VOID fill instead"))
    rs = db.get(RunSegment, job.run_segment_id)
    _require_fillable(run, project, rs, BASIS_SOURCE)
    changed = base_mismatch(job.base, live_base(db, run, rs, project, job))
    if changed:
        raise StalePreview(changed)
    _check_segment_rerun_conflicts(run.id, db)
    _require_audio_patch_discard_ack(run, db, False)
    clip = os.path.join(job.prep_dir or "", "clip.mp4")
    media = job.media or {}
    if not job.prep_dir or not os.path.isfile(clip):
        raise HTTPException(status_code=409, detail="the prepared clip is gone; make a new preview")
    work = os.path.join(_job_dir(job), f"apply-{uuid.uuid4().hex}")
    os.makedirs(work, exist_ok=False)
    out = os.path.join(work, "composite.mp4")
    try:
        shutil.copyfile(clip, out)
        if probe_video(out)["frames"] != int(media.get("frames") or -1):
            raise FillError("encode_failed", "the copied clip does not have the prepared frames")
    except (OSError, FillError) as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=409, detail=f"could not copy the source clip: {exc}")
    job.approved_usd = 0.0
    try:
        _apply_source_result(db, run, rs, job, out)
    except Exception:
        # Refused before its commit (or the enqueue after it failed): the copy
        # is removed unless the promotion committed (then it IS the segment's
        # result and the reaper re-dispatches the queued run). A process
        # crash leaves it for this job's terminal cleanup (cancel / expiry).
        if not _committed_applied(job.id, out):
            shutil.rmtree(work, ignore_errors=True)
        raise
    log.info("segment_fill %s: the source clip applied as is to run %s segment %s",
             job.id, run.id, rs.id)
    return job


def _committed_applied(job_id: str, composite: str) -> bool:
    try:
        with get_session() as s2:
            row = s2.get(RunSegmentFillJob, job_id)
            return row is not None and row.status == FillStatus.applied.value \
                and row.composite_path == composite
    except Exception:  # unknown: keep the file (an orphan beats a dangling result)
        return True


# ---------------------------------------------------------------------------
# Composite + promote (free)
# ---------------------------------------------------------------------------


def segment_fill_composite(job_id: str) -> None:
    claimed = _claim(job_id, FillStatus.compositing.value, from_status=FillStatus.admitted.value)
    if claimed is None:
        log.info("segment_fill_composite %s: not claimable — nothing to do", job_id)
        return
    token, job = claimed
    with get_session() as s:
        task = s.get(ProviderTask, job.provider_task_id) if job.provider_task_id else None
        task_ok = (task is not None and task.status == ProviderTaskStatus.done.value
                   and task.result_path and task.result_path == task.destination_path)
        raw = task.result_path if task_ok else None
    if not task_ok:
        # Fence: an abandoned/closed/refused task never composites. Give the row
        # back to `admitted` so the reconciler decides on the task's state.
        with get_session() as s:
            s.execute(update(RunSegmentFillJob).where(
                RunSegmentFillJob.id == job_id, RunSegmentFillJob.claim_token == token,
                RunSegmentFillJob.status == FillStatus.compositing.value,
            ).values(status=FillStatus.admitted.value, claim_token=None, claim_expires_at=None))
            s.commit()
        return
    work = os.path.join(_job_dir(job), f"comp-{token}")
    try:
        os.makedirs(work, exist_ok=False)
        out = _composite_into(job, raw, work)
    except FillError as exc:
        shutil.rmtree(work, ignore_errors=True)
        _fail(job_id, FillStatus.compositing.value, token, exc.code, exc.message)
        return
    except Exception as exc:
        shutil.rmtree(work, ignore_errors=True)
        log.exception("segment_fill_composite %s: crashed: %s", job_id, exc)
        if job.attempts >= MAX_ATTEMPTS:
            _fail(job_id, FillStatus.compositing.value, token, "composite_crashed",
                  f"{type(exc).__name__}: {exc}")
        return
    if job.batch_id:
        # A batch member waits for the batch's ONE promote (design §1.4).
        if _mark_composited(job_id, token, out, work):
            promote_batch(job.batch_id)
        return
    promote(job_id, token, out, work)


def _mark_composited(job_id: str, token: str, composite: str, work: str) -> bool:
    """compositing -> composited under the row lock + claim check; the
    composite stays in its claim dir until the batch promote applies it."""
    try:
        size, sha = os.path.getsize(composite), _file_sha256(composite)
        with get_session() as s:
            row = _still_ours(s, job_id, FillStatus.compositing.value, token)
            if row is None:
                s.rollback()
                shutil.rmtree(work, ignore_errors=True)
                log.info("segment_fill %s: lost the claim before marking composited", job_id)
                return False
            row.status = FillStatus.composited.value
            row.composite_path = composite
            row.media = {**(row.media or {}), "composite_size": size, "composite_sha256": sha}
            row.claim_token = row.claim_expires_at = None
            s.commit()
    except Exception:
        # Ambiguous: keep the directory if the row did reach `composited`
        # with it (R1-12); otherwise the expired claim is retried.
        with get_session() as s2:
            row = s2.get(RunSegmentFillJob, job_id)
            if row is None or row.composite_path != composite:
                shutil.rmtree(work, ignore_errors=True)
        raise
    log.info("segment_fill %s: composited, waiting for its batch's promote", job_id)
    return True


def _claim_batch(batch_id: str) -> Optional[str]:
    """CAS the batch into ``promoting`` with a fresh token, under its row lock:
    an ``active`` batch whose members all settled (none ``admitted`` /
    ``compositing``), or a ``promoting`` one whose claim expired. Bounded:
    a batch whose promote crashed MAX_ATTEMPTS times ends ``failed``."""
    token = uuid.uuid4().hex
    now = _now()
    with get_session() as s:
        b = s.execute(select(RunSegmentFillBatch).where(RunSegmentFillBatch.id == batch_id)
                      .with_for_update()).scalar_one_or_none()
        if b is None or b.status not in BATCH_LIVE:
            return None
        if b.status == BatchStatus.promoting.value and b.claim_expires_at \
                and b.claim_expires_at > now:
            return None
        unsettled = s.execute(select(RunSegmentFillJob.id).where(
            RunSegmentFillJob.batch_id == batch_id,
            RunSegmentFillJob.status.in_((FillStatus.admitted.value,
                                          FillStatus.compositing.value))).limit(1)).first()
        if unsettled is not None:
            return None
        if b.attempts >= MAX_ATTEMPTS:
            _fail_batch_rows(s, b, "promote_crashed",
                             f"the batch promote did not finish {b.attempts} times")
            s.commit()
            return None
        b.status = BatchStatus.promoting.value
        b.claim_token = token
        b.claim_expires_at = now + timedelta(seconds=lease_sec())
        b.attempts = (b.attempts or 0) + 1
        s.commit()
    return token


def _fail_batch_rows(s, b: RunSegmentFillBatch, code: str, message: str,
                     summary: Optional[dict] = None) -> None:
    """The batch and every member still waiting for it (``composited``) end
    ``failed``: nothing is applied, the segments keep their results; the
    composites are removed by the terminal cleanup (composite_path cleared)."""
    rows = s.execute(select(RunSegmentFillJob).where(
        RunSegmentFillJob.batch_id == b.id,
        RunSegmentFillJob.status == FillStatus.composited.value)
        .with_for_update()).scalars().all()
    for r in rows:
        r.status, r.error_code, r.error_message = FillStatus.failed.value, code, message[:500]
        r.composite_path = None
    b.status, b.error_code, b.error_message = BatchStatus.failed.value, code, message[:500]
    b.claim_token = b.claim_expires_at = None
    b.summary = summary or {r.id: f"failed {code}" for r in rows}
    log.warning("segment_fill batch %s: failed (%s) %s", b.id, code, message)


def promote_batch(batch_id: str) -> Optional[str]:
    """The batch's ONE promote (design §1.5): every ``composited`` member whose
    segment-bound base still matches is put in as its segment's result, the
    others end ``failed``; then ONE ``_invalidate_final_and_requeue`` (one
    stitch, one in-place re-delivery). Lock order project -> run -> batch ->
    member rows. Returns the batch's final status (or None when not claimed)."""
    from fastapi import HTTPException

    from . import api_v2
    from .api_segment_fill import _require_fillable
    from .state_machine import SegmentStatus

    token = _claim_batch(batch_id)
    if token is None:
        return None
    committed = False
    try:
        with get_session() as db:
            b0 = db.get(RunSegmentFillBatch, batch_id)
            if b0 is None:
                return None
            project = db.execute(select(VideoProject).where(VideoProject.id == b0.project_id)
                                 .with_for_update()).scalar_one_or_none()
            run = db.get(Run, b0.run_id)
            if project is None or run is None:
                return None
            try:
                api_v2._lock_run_for_result_mutation(run, db, own_batch_id=batch_id)
                b = db.execute(select(RunSegmentFillBatch).where(
                    RunSegmentFillBatch.id == batch_id).with_for_update()).scalar_one()
                if b.status != BatchStatus.promoting.value or b.claim_token != token \
                        or b.claim_expires_at is None or b.claim_expires_at <= _now():
                    db.rollback()
                    log.info("segment_fill batch %s: lost the claim before promoting", batch_id)
                    return None
                members = db.execute(select(RunSegmentFillJob).where(
                    RunSegmentFillJob.batch_id == batch_id).order_by(RunSegmentFillJob.id)
                    .with_for_update()).scalars().all()
                api_v2._require_segment_reconstruction_allowed(run.id, db)
                basis = b.basis or BASIS_MODEL
                if basis == BASIS_SOURCE:
                    if _status(run.status) not in SOURCE_RUN_STATUSES:
                        raise FillError("base_changed", f"the run is {_status(run.status)}, "
                                                        "not incomplete/failed")
                elif _status(run.status) != "done":
                    raise FillError("base_changed", f"the run is {_status(run.status)}, not done")
                api_v2._check_segment_rerun_conflicts(run.id, db)
                api_v2._require_audio_patch_discard_ack(run, db, b.discard_audio_patch)
            except (FillError, HTTPException) as exc:
                code = exc.code if isinstance(exc, FillError) else f"gate_{exc.status_code}"
                msg = exc.message if isinstance(exc, FillError) else str(exc.detail)
                db.rollback()
                b = db.execute(select(RunSegmentFillBatch).where(
                    RunSegmentFillBatch.id == batch_id).with_for_update()).scalar_one()
                if b.status == BatchStatus.promoting.value and b.claim_token == token:
                    _fail_batch_rows(db, b, code, msg)
                    db.commit()
                return BatchStatus.failed.value
            summary, apply = {}, []
            for m in members:
                if m.status != FillStatus.composited.value:
                    summary[m.id] = m.status if m.status != "failed" else \
                        f"failed {m.error_code or ''}".strip()
                    continue
                rs = db.get(RunSegment, m.run_segment_id)
                why = None
                if rs is None or not m.composite_path or not os.path.isfile(m.composite_path):
                    why = ("composite_missing", "the composite or its segment is gone")
                else:
                    try:
                        _require_fillable(run, project, rs, basis)
                        changed = base_mismatch(m.base, live_base(db, run, rs, project, m))
                        if changed:
                            why = ("base_changed", f"{changed} changed since the preview")
                    except HTTPException as exc:
                        why = (f"gate_{exc.status_code}", str(exc.detail))
                if why:
                    m.status, m.error_code, m.error_message = FillStatus.failed.value, why[0], \
                        why[1][:500]
                    m.composite_path = None
                    summary[m.id] = f"failed {why[0]}"
                    continue
                apply.append((m, rs))
            missing = (paid_restitch_segments(db, run, exclude_rs_ids={rs.id for _m, rs in apply})
                       if basis == BASIS_MODEL else [])
            if not apply or missing:
                code = "restitch_would_bill" if missing else "nothing_to_apply"
                msg = (f"segment(s) {missing} have no finished result" if missing else
                       "no member of the batch could be applied")
                _fail_batch_rows(db, b, code, msg, summary={**summary, **{
                    m.id: f"failed {code}" for m, _rs in apply}})
                db.commit()
                return BatchStatus.failed.value
            now = _now()
            if basis == BASIS_SOURCE:
                # Each member is its segment's FIRST result; then ONE check:
                # requeued for the stitch + first delivery only when the run
                # is complete and the requeue bills nothing (design §7.2).
                for m, rs in apply:
                    _put_source_result(rs, m, m.composite_path)
                    summary[m.id] = "applied"
                b.status = BatchStatus.applied.value
                b.applied_at = now
                b.summary = summary
                b.claim_token = b.claim_expires_at = None
                committed = True
                _requeue_if_complete(db, run, f"batch {batch_id}")
                log.info("segment_fill batch %s: applied %d source fill(s) to run %s",
                         batch_id, len(apply), b0.run_id)
                return BatchStatus.applied.value
            for m, rs in apply:
                take = api_v2._archive_current_take(run, rs, db)
                rs.local_result_path = m.composite_path
                rs.seedance_task_id = None
                rs.seedance_result_url = None
                rs.error_message = None
                rs.source_fallback = False
                rs.status = SegmentStatus.completed
                m.status = FillStatus.applied.value
                m.replaced_take_id = take.id if take is not None else None
                m.applied_at = now
                m.claim_token = m.claim_expires_at = None
                summary[m.id] = "applied"
            b.status = BatchStatus.applied.value
            b.applied_at = now
            b.summary = summary
            b.claim_token = b.claim_expires_at = None
            # One commit for the batch, its members, the segments and the run's
            # invalidation; one process_run after it (R1-12).
            committed = True
            api_v2._invalidate_final_and_requeue(run, db, kind="segment fill batch")
        log.info("segment_fill batch %s: applied %d member(s) to run %s; run requeued",
                 batch_id, sum(1 for v in summary.values() if v == "applied"), b0.run_id)
        return BatchStatus.applied.value
    except Exception:
        if committed and _batch_committed(batch_id):
            # Applied; only the enqueue after the commit failed: the run reaper
            # re-dispatches the queued run. Nothing is cleaned.
            log.exception("segment_fill batch %s: applied; the process_run enqueue failed "
                          "(the run reaper re-dispatches it)", batch_id)
            return BatchStatus.applied.value
        raise


def _batch_committed(batch_id: str) -> bool:
    try:
        with get_session() as s:
            b = s.get(RunSegmentFillBatch, batch_id)
            return b is not None and b.status == BatchStatus.applied.value
    except Exception:  # unknown: treat as applied (an orphan beats a dangling result)
        return True


def _composite_into(job: RunSegmentFillJob, raw: str, work: str) -> str:
    media = job.media or {}
    n, w, h, fps = int(media["frames"]), int(media["w"]), int(media["h"]), media["fps"]
    sampling = media["sampling"]
    m = len(sampling)
    got = probe_video(raw)["frames"]
    if got != m:
        raise FillError("result_frames", f"VOID returned {got} frames, expected {m}")
    clip = os.path.join(job.prep_dir, "clip.mp4")
    masks = np.load(os.path.join(job.prep_dir, "masks.npy"), mmap_mode="r")
    if masks.shape != (n, h, w):
        raise FillError("prep_missing", "the prepared masks do not match the clip")
    to_void = frame_to_void(n, sampling)
    out = os.path.join(work, "composite.mp4")
    enc = Encoder(out, w, h, fps)
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    void_iter = iter_frames(raw, w, h)
    cur_q, cur = -1, None
    for i, f in enumerate(iter_frames(clip, w, h)):
        if i >= n:
            break
        q = to_void[i]
        while cur_q < q:
            cur = next(void_iter, None)
            if cur is None:
                raise FillError("result_frames", "the VOID output ended early")
            cur_q += 1
        mk = masks[i]
        if mk.any():
            a = cv2.GaussianBlur(cv2.dilate(np.asarray(mk), k_d).astype(np.float32),
                                 (9, 9), 2.0)[..., None]
            f = (f * (1 - a) + cur * a).astype(np.uint8)
        enc.write(f)
    void_iter.close()
    enc.close()
    if enc.count != n or probe_video(out)["frames"] != n:
        raise FillError("encode_failed", f"composite has {enc.count} frames, expected {n}")
    return out


def promote(job_id: str, token: str, composite: str, work: str) -> None:
    """Put the composite in as the segment's result (design §3.5.5). Lock order
    project → run → fill row, like Fal admission."""
    from fastapi import HTTPException

    from . import api_v2
    from .state_machine import SegmentStatus

    with get_session() as db:
        job = db.get(RunSegmentFillJob, job_id)
        if job is None:
            shutil.rmtree(work, ignore_errors=True)
            return
        if job.batch_id:
            # Never a single promote of a batch member: the batch applies it.
            log.error("segment_fill %s: batch member reached the single promote", job_id)
            return
        project = db.execute(select(VideoProject).where(VideoProject.id == job.project_id)
                             .with_for_update()).scalar_one_or_none()
        run = db.get(Run, job.run_id)
        rs = db.get(RunSegment, job.run_segment_id)
        if project is None or run is None or rs is None:
            shutil.rmtree(work, ignore_errors=True)
            return
        try:
            api_v2._lock_run_for_result_mutation(run, db, own_fill_id=job_id)
            api_v2._require_segment_reconstruction_allowed(run.id, db)
            row = _still_ours(db, job_id, FillStatus.compositing.value, token)
            if row is None:
                db.rollback()
                shutil.rmtree(work, ignore_errors=True)
                log.info("segment_fill %s: lost the claim before promoting", job_id)
                return
            basis = fill_basis(row)
            run_status = run.status.value if hasattr(run.status, "value") else str(run.status)
            if basis == BASIS_SOURCE:
                if run_status not in SOURCE_RUN_STATUSES:
                    raise FillError("base_changed",
                                    f"the run is {run_status}, not incomplete/failed")
            elif run_status != "done":
                raise FillError("base_changed", f"the run is {run_status}, not done")
            changed = base_mismatch(row.base, live_base(db, run, rs, project, row))
            if changed:
                raise FillError("base_changed", f"{changed} changed since the preview")
            api_v2._check_segment_rerun_conflicts(run.id, db)
            api_v2._require_audio_patch_discard_ack(run, db, row.discard_audio_patch)
            if basis == BASIS_SOURCE:
                from .api_segment_fill import _require_fillable

                _require_fillable(run, project, rs, BASIS_SOURCE)
                _apply_source_result(db, run, rs, row, composite)
                log.info("segment_fill %s: applied from the source to run %s segment %s",
                         job_id, job.run_id, job.run_segment_id)
                return
            missing = paid_restitch_segments(db, run, exclude_rs_id=rs.id)
            if missing:
                raise FillError("restitch_would_bill",
                                f"segment(s) {missing} have no finished result")
            take = api_v2._archive_current_take(run, rs, db)
            rs.local_result_path = composite
            rs.seedance_task_id = None
            rs.seedance_result_url = None
            rs.error_message = None
            rs.source_fallback = False
            rs.status = SegmentStatus.completed
            row.status = FillStatus.applied.value
            row.composite_path = composite
            row.media = {**(row.media or {}), "composite_size": os.path.getsize(composite),
                         "composite_sha256": _file_sha256(composite)}
            row.replaced_take_id = take.id if take is not None else None
            row.applied_at = _now()
            row.claim_token = row.claim_expires_at = None
            api_v2._invalidate_final_and_requeue(run, db, kind="segment fill")
        except FillError as exc:
            db.rollback()
            shutil.rmtree(work, ignore_errors=True)
            _fail(job_id, FillStatus.compositing.value, token, exc.code, exc.message)
            return
        except HTTPException as exc:
            db.rollback()
            shutil.rmtree(work, ignore_errors=True)
            _fail(job_id, FillStatus.compositing.value, token, f"gate_{exc.status_code}",
                  str(exc.detail))
            return
    log.info("segment_fill %s: applied to run %s segment %s; run requeued for the stitch",
             job_id, job.run_id, job.run_segment_id)


SOURCE_RUN_STATUSES = ("incomplete", "failed")


def _apply_source_result(db, run: Run, rs: RunSegment, job: RunSegmentFillJob,
                         composite: str) -> None:
    """Put a source-basis composite (or the source clip as is) in as the
    blocked segment's FIRST result, inside the caller's locks (project ->
    run -> fill row), and COMMIT. The run is requeued for its stitch + first
    delivery only when every active swap segment now has a finished result and
    the submit phase would bill nothing (checked here, under the run lock --
    the stitch's completeness gate is not a spending guard, design R1-6/7);
    otherwise it stays ``incomplete`` (another blocked segment gets its own
    fill; a transient one waits for the operator's Retry, which never
    resubmits this now-completed segment)."""
    _put_source_result(rs, job, composite)
    _requeue_if_complete(db, run, job.id)


def _put_source_result(rs: RunSegment, job: RunSegmentFillJob, composite: str) -> None:
    """A source-basis composite becomes the segment's FIRST result (no take:
    there was none); the job ``applied``. No commit."""
    from .state_machine import SegmentStatus

    rs.local_result_path = composite
    rs.seedance_task_id = None
    rs.seedance_result_url = None
    rs.error_message = None
    rs.source_fallback = False
    rs.model_fallback = False
    rs.status = SegmentStatus.completed
    job.status = FillStatus.applied.value
    job.composite_path = composite
    job.media = {**(job.media or {}), "composite_size": os.path.getsize(composite),
                 "composite_sha256": _file_sha256(composite)}
    job.replaced_take_id = None
    job.applied_at = _now()
    job.claim_token = job.claim_expires_at = None


def _requeue_if_complete(db, run: Run, label: str) -> None:
    """Under the run lock: requeue the run for its stitch + delivery only when
    every active swap segment has a finished result and the requeue bills
    nothing; otherwise leave it parked with its error recomputed. COMMITS."""
    from . import api_v2, spend
    from .pipeline_v2 import preprocess_hole_index

    job_id = label
    db.flush()
    missing = paid_restitch_segments(db, run)
    need = spend.estimate_run_credits(db, run, include_preprocess=True) if not missing else None
    hole = preprocess_hole_index(run) if run.preprocess_enabled else None
    if not missing and need == 0 and hole is None:
        api_v2._invalidate_final_and_requeue(run, db, kind="segment fill")
        return
    if not missing:
        log.error("segment_fill %s: every segment is finished but a requeue of run %s would "
                  "bill (%.1f credits, preprocess hole %s) -- left %s for the operator",
                  job_id, run.id, need or 0.0, hole, run.status)
    from .project_types import is_void_source_run

    run.error_message = (
        (f"VOID from the source: {len(missing)} swap segment(s) still need their fill: "
         f"{missing}. The video is stitched once every segment has one."
         if is_void_source_run(run) else
         f"{len(missing)} swap segment(s) did not complete: {missing}. Re-run them — the full "
         "video is stitched only once every segment succeeds.") if missing else
        "every segment has a result, but the re-stitch would bill (generation or GPT Image "
        "preprocessing): finish the run by hand")
    db.commit()


def paid_restitch_segments(db, run: Run, *, exclude_rs_id: Optional[str] = None,
                           exclude_rs_ids=()) -> list[int]:
    """Active swap segments a re-stitch would (re)generate — paid. Mirrors the
    completeness check of restore_segment_takes; a source-fallback segment is
    counted too (submit would try it again)."""
    from .api_v2 import _run_segment_defs_for
    from .pipeline_v2 import _active_swap_segment_ids
    from .state_machine import SegmentStatus

    defs = _run_segment_defs_for(run)
    active = _active_swap_segment_ids(run.stitch_plan, defs)
    by_def = {rs.segment_def_id: rs for rs in run.run_segments}
    skip = set(exclude_rs_ids or ()) | ({exclude_rs_id} if exclude_rs_id else set())
    bad = []
    for sd in defs:
        if sd.id not in active:
            continue
        rs = by_def.get(sd.id)
        if rs is not None and rs.id in skip:
            continue
        if (rs is None or rs.status != SegmentStatus.completed or rs.source_fallback
                or not rs.local_result_path or not os.path.exists(rs.local_result_path)):
            bad.append(sd.index)
    return bad


def active_fill(db, run_id: str, *, exclude_batch_id: Optional[str] = None
                ) -> Optional[RunSegmentFillJob]:
    """An active fill of the run: a legacy single one or a batch member
    (admitted / compositing / composited); *exclude_batch_id*: not one of
    that batch's members (its own promote)."""
    q = select(RunSegmentFillJob).where(RunSegmentFillJob.run_id == run_id,
                                        RunSegmentFillJob.status.in_(FILL_ACTIVE))
    if exclude_batch_id is not None:
        q = q.where((RunSegmentFillJob.batch_id.is_(None))
                    | (RunSegmentFillJob.batch_id != exclude_batch_id))
    return db.execute(q.order_by(RunSegmentFillJob.id).limit(1)).scalar_one_or_none()


def delivery_state(job: RunSegmentFillJob, run: Optional[Run],
                   rs: Optional[RunSegment] = None, db=None) -> Optional[str]:
    """For an applied fill: ``superseded`` once the segment no longer uses its
    composite; ``delivered`` when it does, the run is done and its final differs
    from the one the fill replaced (a batch member: the final at the batch's
    ADMISSION -- a segment-bound preview's own revision may be older); else
    the run's status."""
    if job.status != FillStatus.applied.value or run is None:
        return None
    from .assistant_run_result import result_revision

    if rs is None or not _is_composite(job, rs.local_result_path):
        return "superseded"
    baseline = (job.base.get("run") or {}).get("final_revision")
    if job.batch_id and db is not None:
        b = db.get(RunSegmentFillBatch, job.batch_id)
        if b is not None:
            baseline = ((b.base or {}).get("run") or {}).get("final_revision")
    st = run.status.value if hasattr(run.status, "value") else str(run.status)
    if st == "done" and result_revision(run) != baseline:
        # Only the pipeline's own stitch is known to contain the segment; an
        # assistant publication or an audio patch re-points the final elsewhere.
        final = os.path.join(storage.run_dir(run.id, run.project_id), "results", "final.mp4")
        return "delivered" if run.result_local_path == final else "republished"
    return st


def _file_sha256(path: str) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _is_composite(job: RunSegmentFillJob, path: Optional[str]) -> bool:
    """The composite itself, or a take copy of it put back by a restore."""
    if not path:
        return False
    if path == job.composite_path:
        return True
    media = job.media or {}
    size, sha = media.get("composite_size"), media.get("composite_sha256")
    if not size or not sha:
        return False
    try:
        if os.path.getsize(path) != size:
            return False
    except OSError:
        return False
    return _file_sha256(path) == sha


# ---------------------------------------------------------------------------
# Reconcile (recovery tick) + terminal cleanup
# ---------------------------------------------------------------------------


def reconcile_fill_jobs(live_ids: set[str], *, grace_sec: int = 120) -> dict:
    from sqlalchemy import text as sa_text

    from .state_machine import PROVIDER_TASK_ACTIVE_WHERE
    from .tasks import (enqueue_segment_fill_batch_promote, enqueue_segment_fill_composite,
                        enqueue_segment_fill_prepare)

    now = _now()
    cutoff = now - timedelta(seconds=grace_sec)
    to_prepare, to_composite, failed, canceled, cleaned = [], [], [], [], 0
    with get_session() as s:
        rows = s.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.status.in_(
            (FillStatus.preparing.value, FillStatus.compositing.value)))).scalars().all()
        for job in rows:
            if job.id in live_ids or (job.updated_at and job.updated_at > cutoff):
                continue
            if job.claim_token and job.claim_expires_at and job.claim_expires_at > now:
                continue
            if job.attempts >= MAX_ATTEMPTS:
                code = "prepare_crashed" if job.status == "preparing" else "composite_crashed"
                n = s.execute(update(RunSegmentFillJob).where(
                    RunSegmentFillJob.id == job.id, RunSegmentFillJob.status == job.status,
                    RunSegmentFillJob.attempts == job.attempts,
                    RunSegmentFillJob.updated_at == job.updated_at,
                ).values(status=FillStatus.failed.value, error_code=code,
                         error_message=f"{job.attempts} attempts crashed",
                         claim_token=None, claim_expires_at=None)).rowcount
                if n:
                    failed.append(job.id)
            elif job.status == FillStatus.preparing.value:
                to_prepare.append(job.id)
            else:
                to_composite.append(job.id)

        admitted = s.execute(select(RunSegmentFillJob, ProviderTask).join(
            ProviderTask, ProviderTask.id == RunSegmentFillJob.provider_task_id
        ).where(RunSegmentFillJob.status == FillStatus.admitted.value)).all()
        for job, task in admitted:
            if job.id in live_ids:
                continue
            if task.status == ProviderTaskStatus.done.value:
                to_composite.append(job.id)
                continue
            still_active = s.execute(select(ProviderTask.id).where(
                ProviderTask.id == task.id, sa_text(PROVIDER_TASK_ACTIVE_WHERE))).first()
            if still_active is None:
                n = s.execute(update(RunSegmentFillJob).where(
                    RunSegmentFillJob.id == job.id,
                    RunSegmentFillJob.status == FillStatus.admitted.value,
                ).values(status=FillStatus.failed.value, error_code=f"provider_{task.status}",
                         error_message=(task.error_message or "")[:500])).rowcount
                if n:
                    failed.append(job.id)

        to_promote, detached = _reconcile_batches(s, live_ids, now, cutoff)

        stale = s.execute(update(RunSegmentFillJob).where(
            RunSegmentFillJob.status == FillStatus.ready.value,
            RunSegmentFillJob.updated_at < now - READY_TTL,
        ).values(status=FillStatus.canceled.value, error_code="preview_expired")).rowcount
        if stale:
            canceled.append(stale)
        s.commit()

        terminal = s.execute(select(RunSegmentFillJob).where(
            RunSegmentFillJob.status.in_(("applied", "failed", "canceled")),
            RunSegmentFillJob.files_cleaned_at.is_(None))).scalars().all()
        for job in terminal:
            if job.provider_task_id:
                busy = s.execute(select(ProviderTask.id).where(
                    ProviderTask.id == job.provider_task_id,
                    sa_text(PROVIDER_TASK_ACTIVE_WHERE))).first()
                if busy is not None:
                    continue
            _clean_job_files(job)
            job.files_cleaned_at = now
            cleaned += 1
        s.commit()

    for jid in to_prepare:
        try:
            enqueue_segment_fill_prepare(jid)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("reconcile_fill: re-enqueue of prepare %s failed: %s", jid, exc)
    for jid in to_composite:
        try:
            enqueue_segment_fill_composite(jid)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("reconcile_fill: enqueue of composite %s failed: %s", jid, exc)
    for bid in to_promote:
        try:
            enqueue_segment_fill_batch_promote(bid)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("reconcile_fill: enqueue of batch promote %s failed: %s", bid, exc)
    if to_prepare or to_composite or failed or canceled or cleaned or to_promote or detached:
        log.info("reconcile_fill: prepare %s composite %s failed %s canceled %s cleaned %d "
                 "promote %s detached %s", to_prepare, to_composite, failed, canceled, cleaned,
                 to_promote, detached)
    return {"prepare": len(to_prepare), "composite": len(to_composite),
            "failed": len(failed), "cleaned": cleaned, "promote": len(to_promote),
            "detached": len(detached)}


HELD_PROVIDER_STATES = ("submit_unknown", "needs_reconciliation")


def _held(task) -> bool:
    """A provider task only the operator resolves (the Fal runbook)."""
    if task is None:
        return False
    return (task.status == "refused" and bool(task.reservation_held)) \
        or task.status in HELD_PROVIDER_STATES


def _reconcile_batches(s, live_ids: set, now: datetime, cutoff: datetime):
    """Live batches (design §1.4-1.5, R1-3, R1-9): a ``promoting`` batch whose
    claim expired, or an ``active`` one whose members all settled (past the
    grace, no live promote job), gets its promote enqueued. A member whose Fal
    task only the operator can resolve is DETACHED (``failed held_detached``:
    its task, reservation and charge stay with the provider task; a late
    result is never applied) once every other member settled at least
    ``SEGMENT_FILL_BATCH_HOLD_SEC`` ago AND one of them is ``composited`` --
    useful work is then not held up by an uncertain charge. Without a
    composited member the batch keeps holding, like a single fill."""
    hold = timedelta(seconds=int(getattr(settings, "SEGMENT_FILL_BATCH_HOLD_SEC", 1800)))
    to_promote, detached = [], []
    batches = s.execute(select(RunSegmentFillBatch).where(
        RunSegmentFillBatch.status.in_(BATCH_LIVE))).scalars().all()
    for b in batches:
        if b.id in live_ids:
            continue
        if b.status == BatchStatus.promoting.value:
            if b.claim_expires_at is None or b.claim_expires_at <= now:
                to_promote.append(b.id)
            continue
        members = s.execute(select(RunSegmentFillJob).where(
            RunSegmentFillJob.batch_id == b.id)).scalars().all()
        unsettled = [m for m in members if m.status in (FillStatus.admitted.value,
                                                        FillStatus.compositing.value)]
        if not unsettled:
            if all(m.updated_at is None or m.updated_at <= cutoff for m in members):
                to_promote.append(b.id)
            continue
        if any(m.status != FillStatus.admitted.value for m in unsettled):
            continue
        tasks = {m.id: (s.get(ProviderTask, m.provider_task_id) if m.provider_task_id else None)
                 for m in unsettled}
        if not all(_held(t) for t in tasks.values()):
            continue
        others = [m for m in members if m not in unsettled]
        if not any(m.status == FillStatus.composited.value for m in others):
            continue
        last = max([m.updated_at for m in others if m.updated_at] + [b.created_at])
        if now - last < hold:
            continue
        # Re-read every held task under its row lock (gate R1-2): a task that
        # completed (or moved) since the read above is not detached -- its
        # result composites normally on the next pass.
        still = []
        for m in unsettled:
            t = s.execute(select(ProviderTask).where(ProviderTask.id == m.provider_task_id)
                          .with_for_update().execution_options(populate_existing=True)
                          ).scalar_one_or_none()
            if not _held(t):
                still = None
                break
            still.append((m, t))
        if still is None:
            continue
        for m, t in still:
            n = s.execute(update(RunSegmentFillJob).where(
                RunSegmentFillJob.id == m.id, RunSegmentFillJob.status == FillStatus.admitted.value,
            ).values(status=FillStatus.failed.value, error_code="held_detached",
                     error_message=(
                         f"its Fal task {t.id} is {t.status} (only the operator resolves it: "
                         "python -m app.pipeline_provider list); detached so the batch's other "
                         "members promote -- a late result is not applied"))).rowcount
            if n:
                detached.append(m.id)
        to_promote.append(b.id)
    return to_promote, detached


def _clean_job_files(job: RunSegmentFillJob) -> None:
    """Keep only the composite (when applied) and the preview JPEG."""
    root = _job_dir(job)
    if not os.path.isdir(root):
        return
    keep_dirs = {os.path.dirname(p) for p in (job.composite_path,) if p}
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if path == job.prep_dir and job.preview_path:
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if fp != job.preview_path:
                    try:
                        os.remove(fp) if os.path.isfile(fp) else shutil.rmtree(fp, True)
                    except OSError:
                        pass
            continue
        if path in keep_dirs:
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if fp != job.composite_path:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
            continue
        try:
            os.remove(path) if os.path.isfile(path) else shutil.rmtree(path, True)
        except OSError:
            pass
