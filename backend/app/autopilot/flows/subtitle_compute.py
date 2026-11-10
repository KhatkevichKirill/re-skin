"""Free computes of the subtitle-removal flow (AICP-364, docs/autopilot-subtitle-flow.md §4).

Run on the run workers' ``autopilot_compute`` queue (core §5.5) as
``func(args, out_dir) -> dict``. Each opens its own read-only database
session, never writes the database, writes files only into its attempt dir
(``out_dir``) and streams video: the caption detector reads frames one at a
time (AICP-343) and the side-by-side clips are made by ffmpeg, never decoded
into Python memory (VPS crashes of 2026-10-30).

* ``subtitle_review`` -- ``subtitle_loop.review_segmentation`` on the
  project's source (with InsightFace face samples at 1 fps), shaped by
  ``subtitle_removal_plan.review_result``.
* ``subtitle_qa`` -- what ``python -m app.subtitle_loop qa --run`` prints,
  shaped by ``subtitle_removal_plan.qa_result``, plus a mask-box clip
  (result | source) per VOID composite for the eye check.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Callable, Optional

from ...rebuild import frame_index
from .. import registry
from . import subtitle_removal_plan as plan

log = logging.getLogger(__name__)

# Tests point this at their own engine; production uses app.db.SessionLocal.
SESSION_FACTORY: Optional[Callable] = None
CLIP_HEIGHT = 240
CLIP_TIMEOUT_SEC = 300


def _session():
    if SESSION_FACTORY is not None:
        return SESSION_FACTORY()
    from ...db import SessionLocal

    return SessionLocal()


def _read_only(session, *, repeatable: bool = False) -> None:
    from sqlalchemy import text

    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                             if repeatable else "SET TRANSACTION READ ONLY"))


def _project_segments(project) -> list[dict]:
    return plan.segment_list([{
        "id": sd.id, "index": sd.index, "start_sec": sd.start_sec, "end_sec": sd.end_sec,
        "action": sd.action, "pre_roll_sec": sd.pre_roll_sec, "post_roll_sec": sd.post_roll_sec,
    } for sd in project.segments])


def _face_samples(source: str):
    from ... import subtitle_loop as sl

    return sl._face_samples(source)


def subtitle_review(args: dict, out_dir: str) -> dict:
    from ... import qa_tools, spend
    from ... import subtitle_loop as sl
    from ...models import VideoProject

    pid = args["project_id"]
    with _session() as s:
        _read_only(s, repeatable=True)
        project = s.get(VideoProject, pid)
        if project is None:
            raise ValueError(f"project {pid} not found")
        segments = _project_segments(project)
        config = project.caption_detector
        source = project.source_local_path
        price = sl.price_per_segment(s, plan.MODEL, None)
        s.rollback()
    segs = sl.segments_from([{k: v for k, v in d.items() if k != "id"} for d in segments])
    faces = _face_samples(source) if args.get("faces") else None
    # No band of the project's own: find where the captions sit (full frame,
    # streamed, 2 fps) and let the review pick the band it measures with.
    location = None if qa_tools.has_band(config) else qa_tools.locate_captions(
        source, config=config)
    review = sl.review_segmentation(
        source, segs, config=config, price_per_segment=price, faces=faces, out_dir=out_dir,
        admission_estimate=lambda sec: spend.video_credits(plan.MODEL, None, sec),
        location=location)
    with _session() as s:                      # a FRESH transaction: what changed meanwhile
        _read_only(s)
        project = s.get(VideoProject, pid)
        after_segs = _project_segments(project) if project is not None else []
        after_cfg = project.caption_detector if project is not None else {"gone": True}
        s.rollback()
    return plan.review_result(review, segments=segments, detector_config=config,
                              segments_after=after_segs, detector_after=after_cfg)


def _crop_expr(box, pad: float = 0.04) -> str:
    x0, y0 = max(0.0, box[0] - pad), max(0.0, box[1] - pad)
    x1, y1 = min(1.0, box[2] + pad), min(1.0, box[3] + pad)
    return (f"crop=trunc(iw*{x1 - x0:.4f}/2)*2:trunc(ih*{y1 - y0:.4f}/2)*2:"
            f"trunc(iw*{x0:.4f}):trunc(ih*{y0:.4f}),scale=-2:{CLIP_HEIGHT}")


def fill_clip(source: str, final: str, start: float, end: float, box, out: str) -> Optional[str]:
    """Result | source, cropped to the fill's mask box over one segment, by
    ffmpeg (streamed; no frame ever reaches Python). None when it fails -- the
    fill sheet still carries the eye check."""
    from ...media import FFMPEG

    dur = max(0.1, float(end) - float(start))
    crop = _crop_expr(box)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", final,
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", source,
           "-filter_complex", f"[0:v]{crop}[a];[1:v]{crop}[b];[a][b]hstack=inputs=2[v]",
           "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-threads", "1", out]
    try:
        subprocess.run(cmd, check=True, timeout=CLIP_TIMEOUT_SEC, capture_output=True)
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("subtitle_qa: mask-box clip %s failed: %s", out, exc)
        return None
    return out if os.path.isfile(out) else None


def clip_lengths(segments, results, fps) -> list[dict]:
    """Each swap segment's current result length against its segment, in
    frames at the stitch rate (*fps*, the source's capped by media.stitch_fps): what an unconformed
    stitch adds to the timeline after it (``excess_frames`` > 0 = longer).
    ffprobe only (container duration); None where it cannot be read."""
    from ... import qa_tools

    out = []
    if not fps or fps <= 0:
        return out
    for seg in segments:
        if seg.action != "swap" or not seg.run_segment_id:
            continue
        target = frame_index(seg.end, fps) - frame_index(seg.start, fps)
        path = (results.get(seg.run_segment_id) or {}).get("result_path")
        frames = None
        if path and os.path.isfile(path):
            try:
                # half up, like ffmpeg's fps filter (round=near)
                frames = frame_index(float(qa_tools._probe(path).duration_sec), fps)
            except Exception:  # noqa: BLE001 -- a hint for the planner only
                frames = None
        out.append({"index": seg.index, "target_frames": target, "result_frames": frames,
                    "excess_frames": None if frames is None else frames - target})
    return out


def subtitle_qa(args: dict, out_dir: str) -> dict:
    from ... import pipeline_segment_fill as psf
    from ... import qa_tools
    from ... import subtitle_loop as sl
    from ...models import Run, RunSegment
    from ...qa_tools import QAToolError
    from ..core import run_version as core_run_version

    rid = args["run_id"]
    with _session() as s:
        _read_only(s, repeatable=True)
        version = core_run_version(s, rid)
        inputs = sl._run_inputs(s, rid)
        run = inputs["run"]
        price = sl.price_per_segment(s, run.model, run.resolution)
        results = {rs.id: {"seedance_task_id": rs.seedance_task_id,
                           "result_path": rs.local_result_path}
                   for rs in s.query(RunSegment).filter(RunSegment.run_id == rid)}
        config = inputs.get("config")
        s.rollback()
    source, final = inputs["source"], inputs.get("final")
    common = dict(segments_input=inputs["segments"], run_version=version,
                  detector_config=config, results=results,
                  source_identity=psf.file_identity(source))
    refused = None
    if not final:
        refused = "the run has no final (result_local_path)"
    elif inputs.get("custom_plan"):
        refused = "the run has a custom stitch plan: its final is not on the source timeline"
    if refused is not None:
        return plan.qa_result({}, run_version_after=version, refused=refused, **common)
    segments = sl.segments_from(inputs["segments"])
    try:
        qa = sl.post_run_qa(source, final, segments, config=config, out_dir=out_dir)
    except QAToolError as exc:
        return plan.qa_result({}, run_version_after=version, refused=str(exc), **common)
    if not qa_tools.has_band(config):
        # Warning only: caption-like source text outside the band QA used.
        try:
            chk = sl.out_of_band_check(qa_tools.locate_captions(source, config=config), config)
        except QAToolError as exc:
            chk = {"warning": None, "error": str(exc)[:300]}
        qa["band_location"] = chk
        if chk.get("warning"):
            qa["warnings"].append(chk["warning"])
    fill = sl.fill_proposal(rid, qa, segments, context=inputs.get("fill_context"))
    cli = {"qa": qa, "fill": fill,
           "operator_options": {"omni_rerun": sl.rerun_proposal(
               rid, qa, segments, takes=inputs.get("takes", {}), price_per_segment=price,
               unfinished=inputs.get("unfinished", []))},
           "handoff": sl.handoff_proposal(rid, qa, segments)}
    sheets = {}
    by_index = {sg.index: sg for sg in segments}
    for row in qa["segments"]:
        seg = by_index.get(row["index"])
        box = ((row.get("fill") or {}).get("mask_box") or row.get("mask_box")
               or row.get("fill_sheet_box"))
        if seg is not None and seg.result_kind == "void_fill" and box:
            clip = fill_clip(source, final, seg.start, seg.end, box,
                             os.path.join(out_dir, f"qa_seg{seg.index}_fill.mp4"))
            if clip:
                sheets[f"fill_clip:{seg.index}"] = clip
    # Auto-accept evidence (2026-11-03): faces under the mask of every
    # detector-clean VOID composite, sampled from the source.
    qa["fill_faces"] = fill_faces(source, segments, qa["segments"])
    qa["fill_screen"] = fill_screen(source, segments, qa["segments"])
    try:
        common["source_fps"] = float(qa_tools._probe(source).fps)
    except Exception:  # only a planner hint for the VOID frame minimum
        common["source_fps"] = None
    if qa.get("timing") is not None:
        # At the STITCH rate: a 60 fps source is stitched at 30 (merge gate).
        from ... import media as media_mod
        qa["timing"]["clips"] = clip_lengths(
            segments, results,
            media_mod.stitch_fps(common["source_fps"]) if common["source_fps"] else None)
    with _session() as s:                      # a FRESH transaction (Terra R1-8)
        _read_only(s)
        after = core_run_version(s, rid) if s.get(Run, rid) is not None else "gone"
        s.rollback()
    source_fps = common.pop("source_fps")
    out = plan.qa_result(cli, run_version_after=after, sheets=sheets, **common)
    out["source_fps"] = source_fps
    return out


FACE_SAMPLE_FPS = 2.0
SHEET_WIDTH = 1080


def fill_faces(source: str, segments, rows, fps: float = FACE_SAMPLE_FPS) -> dict:
    """Face boxes (normalised) sampled from the SOURCE at *fps* inside every
    segment whose current result is a detector-clean VOID composite: the
    evidence the planner's auto-accept needs (design §4.1). One frame in
    memory at a time (seek + read); a failure is recorded per segment, never
    raised (the segment is then doubtful)."""
    import cv2

    from ... import face

    want = [(sg, r) for sg, r in ((next((x for x in segments if x.index == r["index"]), None),
                                   r) for r in rows) if sg is not None
            and r.get("result_kind") == "void_fill" and r.get("verdict") == "clean"]
    out: dict = {}
    if not want:
        return out
    try:
        detector = face._build_detector()
    except Exception as exc:  # noqa: BLE001 -- no InsightFace: every one is doubtful
        return {str(r["index"]): {"error": f"face detector: {type(exc).__name__}"}
                for _sg, r in want}
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return {str(r["index"]): {"error": "the source cannot be opened"} for _sg, r in want}
        for sg, r in want:
            boxes, n, err = [], 0, None
            t = float(sg.start) + 0.25 / fps
            while t < float(sg.end) - 1e-3:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    err = "a frame could not be read"
                    break
                h, w = bgr.shape[:2]
                try:
                    for f in detector.get(bgr) or ():
                        x1, y1, x2, y2 = (float(v) for v in f.bbox[:4])
                        boxes.append([round(max(0.0, x1 / w), 4), round(max(0.0, y1 / h), 4),
                                      round(min(1.0, x2 / w), 4), round(min(1.0, y2 / h), 4)])
                except Exception as exc:  # noqa: BLE001
                    err = f"face detection failed: {type(exc).__name__}"
                    break
                n += 1
                t += 1.0 / fps
            out[str(r["index"])] = {"samples": n, "boxes": boxes[:200], "fps": fps,
                                    **({"error": err} if err else {})}
    finally:
        cap.release()
    return out


# Screen / app-UI evidence under a VOID mask (design §7.4). A frame counts as
# "screen" when at least SCREEN_FRAME_SHARE of the padded mask box (caption
# glyphs excluded) is lit-screen background: bright (V > 140), neutral (S < 45)
# or cool blue-grey (H 90-130, S < 120), not skin (warm hue with S >= 35).
# Calibrated on the trial 514067f2 / d8105ab5 with this function (2 fps, frames
# with captions): per-segment maxima 0.39, 0.48, 0.25, 0.72, 0.88, 0.47 for
# segments 0-5 (frames over the threshold 2/8, 2/8, 0/9, 2/16, 8/16, 6/6) --
# 0, 1, 3, 4, 5 (iPad / phone screens under the captions) flagged, 2 (hand,
# sleeve, book) not. Six segments are not a validation set: a flagged or
# unmeasured segment only goes to the eye check, it is never auto-accepted.
SCREEN_FRAME_SHARE = 0.34
SCREEN_SAMPLE_FPS = 2.0


def screen_share(frame_bgr, box) -> Optional[float]:
    """Share of lit-screen background pixels inside the padded *box* of one
    source frame, the caption glyphs (the fill's own ``text_mask``, dilated)
    excluded; None when the frame shows no caption glyph there."""
    import cv2
    import numpy as np

    from ...pipeline_segment_fill import text_mask

    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, int((box[0] - 0.04) * w)), max(0, int((box[1] - 0.04) * h))
    x1, y1 = min(w, int((box[2] + 0.04) * w)), min(h, int((box[3] + 0.04) * h))
    glyph = text_mask(frame_bgr, box)
    if not glyph.any():
        return None
    glyph = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    keep = glyph[y0:y1, x0:x1] == 0
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i].astype(np.int16) for i in range(3))
    skin = ((H <= 25) | (H >= 165)) & (S >= 35)
    bg = (V > 140) & ((S < 45) | ((H >= 90) & (H <= 130) & (S < 120))) & ~skin & keep
    n = int(keep.sum())
    return float(bg.sum()) / n if n else None


def fill_screen(source: str, segments, rows, fps: float = SCREEN_SAMPLE_FPS) -> dict:
    """Per detector-clean VOID composite: how many sampled source frames (with
    captions under the mask) look like a phone / tablet screen or app UI
    there. Seek + read one frame at a time; errors recorded, never raised."""
    import cv2

    out: dict = {}
    want = []
    for r in rows:
        sg = next((x for x in segments if x.index == r["index"]), None)
        box = (r.get("fill") or {}).get("mask_box")
        if sg is not None and r.get("result_kind") == "void_fill" \
                and r.get("verdict") == "clean" and box:
            want.append((sg, r, [float(v) for v in box]))
    if not want:
        return out
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return {str(r["index"]): {"error": "the source cannot be opened"}
                    for _sg, r, _b in want}
        for sg, r, box in want:
            n, flagged, worst, err = 0, 0, 0.0, None
            t = float(sg.start) + 0.25 / fps
            while t < float(sg.end) - 1e-3:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, bgr = cap.read()
                t += 1.0 / fps
                if not ok or bgr is None:
                    err = "a frame could not be read"
                    break
                share = screen_share(bgr, box)
                if share is None:
                    continue
                n += 1
                worst = max(worst, share)
                if share >= SCREEN_FRAME_SHARE:
                    flagged += 1
            out[str(r["index"])] = {"samples": n, "screen_frames": flagged,
                                    "max_share": round(worst, 3),
                                    "threshold": SCREEN_FRAME_SHARE,
                                    **({"error": err} if err else {})}
    finally:
        cap.release()
    return out


def subtitle_batch_sheet(args: dict, out_dir: str) -> dict:
    """ONE contact sheet for a VOID batch question (design §2): per member,
    its QA crop sheet (delivered | original where captions were found) above
    its preview sheet (the area VOID repaints), labelled. Reads JPEGs only
    (never a video), writes one JPEG into the attempt dir."""
    import cv2
    import numpy as np

    blocks = []
    for it in args.get("items") or ():
        parts = []
        for key in ("crop_sheet", "preview_path"):
            p = it.get(key)
            if not p or not os.path.isfile(p):
                continue
            im = cv2.imread(p)
            if im is None:
                continue
            scale = SHEET_WIDTH / float(im.shape[1])
            parts.append(cv2.resize(im, (SHEET_WIDTH, max(1, int(im.shape[0] * scale))),
                                    interpolation=cv2.INTER_AREA))
        if not parts:
            continue
        label = np.zeros((56, SHEET_WIDTH, 3), np.uint8)
        cv2.putText(label, "segment %s%s: delivered | original, then the area VOID repaints" % (
            it.get("index"), " (grown retry)" if it.get("grow") else ""), (12, 38), 0, 0.9,
            (0, 255, 255), 2)
        blocks.append(np.vstack([label] + parts))
    if not blocks:
        # Nothing readable: the question attaches each member's own sheets.
        return {"sheet": None, "members": 0}
    out = os.path.join(out_dir, "batch_sheet.jpg")
    cv2.imwrite(out, np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return {"sheet": out, "members": len(blocks)}


registry.register_compute(plan.C_REVIEW, subtitle_review, version=1)
registry.register_compute(plan.C_BATCH_SHEET, subtitle_batch_sheet, version=1)
registry.register_compute(plan.C_QA, subtitle_qa, version=1)
