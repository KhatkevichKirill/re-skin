"""Subtitle-removal loop: pre-launch review, post-run QA, fill proposal (AICP-353/359).

The free, deterministic half of the loop the assistant runs for a
``subtitle_removal`` project (docs/assistant-flow-subtitle-removal.md). Built
on the AICP-343 caption detector (:mod:`app.qa_tools`). Nothing here writes to
the database, calls a model or spends credits; the only files written are the
JPEG sheets in the caller's ``out_dir``. Every paid step comes out as a
*proposal* -- the exact request for an existing endpoint, with its cost -- that
the operator approves before anyone sends it. Verdicts are evidence for that
approval, not a substitute for the visual check the sheets are for.

* :func:`review_segmentation` -- before launch. Scans the SOURCE once and
  proposes ``keep`` for every swap segment with no captions in the caption band
  and for an end card (a text box that stays put for seconds with no person on
  screen; caption words change every 0.3-1.5 s). Checks the caption geometry
  against the configured band and prices the launch, current vs proposed.
* :func:`post_run_qa` -- after a run. Scans the final, calibrated on the
  source, maps every hit to its segment and calls each segment ``clean``,
  ``leftover``, ``false_positive`` (static UI / end-card text that is also in
  the source) or ``keep``. Writes a crop sheet per dirty segment (result next
  to source at the hit times) so a vision model or the operator can read the
  words -- there is no OCR in the image. A segment whose result is a VOID
  caption fill (AICP-358) keeps its detector verdict and always needs an eye
  check (``fill_needs_visual_check``): the detector misses ghosts and smears.
* :func:`fill_proposal` -- the flow's answer to a dirty segment (rule 1): the
  AICP-358 VOID fill, i.e. the free ``fill/preview`` request with the
  measured mask box and the USD reservation + expected charge of the paid
  ``start`` that follows it; one fill at a time. A VOID composite that is
  still dirty goes to the operator (the doc's stop rule).
* :func:`rerun_proposal` -- an Omni rerun of the dirty segments. Only an
  explicit OPERATOR option now (``operator_options.omni_rerun`` in the CLI),
  never the proposal for a dirty segment.
* :func:`choose_best_takes` -- after a re-check of several takes, the take to
  keep per segment and the ``POST /runs/{rid}/segments/restore`` body.
* :func:`handoff_proposal` -- when the final is clean, the
  ``POST /runs/{rid}/to-project`` request that starts the localisation.

CLI (JSON on stdout; read-only DB access), inside the api container::

    python -m app.subtitle_loop review --project PID [--out-dir DIR] [--faces]
    python -m app.subtitle_loop qa --run RID [--out-dir DIR] [--confirm-fill FILL_IDS]

Both take ``--source`` / ``--final`` / ``--segments JSON`` to run on files
outside the database (fixtures, a take being judged); requests built on such
inputs are evidence only (``fill.actionable`` is false).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from app import qa_tools
from app.qa_tools import CALIBRATION_MIN_RATIO, QAToolError, caption_config

# Gemini Omni is the only caption-removal model (flow doc, rules 1-2: one Omni
# pass, never Seedance), and a VOID fill rebuilds what it leaves (rule 1).
REMOVAL_MODEL = "gemini-omni"
FILL_MODEL = "void"
# Observed Fal charge of one VOID fill request (Fal logs, run 2845eef3 segments
# 4 and 7, 2026-10-31). An observation, not a tariff: the RESERVATION the spend
# cap holds (and approved_usd must cover) comes from fal_models (flat_usd).
VOID_EXPECTED_USD = 0.05
# Fill-prepare refusals that a new preview of the same segment repeats.
FILL_DETERMINISTIC_ERRORS = frozenset({
    "no_captions_detected", "unsupported_media", "segment_too_short", "not_swap",
    "bad_box"})
# Widened-mask option: grow the union of the old mask and the leftovers by this.
WIDEN_PAD_X, WIDEN_PAD_Y = 0.03, 0.02

# A text box that stays within IoU 0.9 of itself for this long is not a burned
# caption: on 328341d4 caption words held 0.5-1.5 s, the end card 4.3 s.
STATIC_IOU = 0.9
STATIC_RUN_SEC = 2.0
# Share of a segment's source hit frames inside static runs above which the
# segment's text is UI / end card rather than captions (328341d4 seg 8: 0.86).
STATIC_SEGMENT_SHARE = 0.6
# A face in at most this share of the sampled frames = "no person on screen".
NO_FACE_RATIO = 0.2
# The review scans a band this much taller on each side than the configured
# one, so captions that sit across its edge are seen at all.
BAND_WIDEN = 0.10
BAND_MARGIN = 0.03
EDGE_EPS = 0.005
# Final vs source duration: further apart than this and the final is not on
# the source timeline (a tail, a stitch plan, a truncated clip).
TIMELINE_TOLERANCE_SEC = 0.3
# A result text box counts as the source's (UI / end card) when half of it
# lies inside a source box grown by this much (frame fractions).
SOURCE_MATCH_PAD = 0.03
# A keep segment with this much caption-like (non-static) source text ships
# captions (3 frames at 10 fps; a single-frame hit is detector noise).
KEEP_CAPTION_MIN_SEC = 0.3
FP_MATCH_SHARE = 0.9
# Mapping hits to segments by time tolerates this much picture drift (30 fps
# frames, 0.3 s; 2845eef3 drifts 3); the flow doc's +-1 frame sync is a warning.
MAPPING_TOLERANCE_FRAMES = 9
# Per-segment timing (2026-11-02, after d8105ab5): the alignment search reaches
# this far (30 fps frames) so a segment 9-12 frames late is measured, not cut
# off at the search limit; a segment is measured on the points at least
# TIMING_EDGE_SEC inside its bounds, needs TIMING_MIN_POINTS of them, and is in
# sync within SYNC_TOLERANCE_FRAMES (the flow doc's +-1 frame at final QA).
ALIGN_SEARCH_FRAMES = 15
TIMING_EDGE_SEC = 0.4
TIMING_MIN_POINTS = 3
SYNC_TOLERANCE_FRAMES = 1
# A leftover is attributed to its segment only clear of the cut by the picture
# shift there plus this much; nearer, it may be the neighbour's clip.
ATTRIBUTION_PAD_SEC = 0.1
# Omni rerun (operator option only): after this many preserved outputs of one
# segment the option is withheld. Preserved takes include restored composites
# and hand-placed clips, so this is a lower bound on Omni attempts, nothing more.
STOP_AFTER_ATTEMPTS = 3
MAX_CROPS_PER_SEGMENT = 10
# Consecutive hit boxes below this IoU belong to different words.
WORD_BOX_IOU = 0.97

COORDINATE_PROMPT_MARKER = "Change ONLY that box"


@dataclass
class Seg:
    """One segment on the source timeline (a SegmentDef or a run's snapshot)."""

    index: int
    start: float
    end: float
    action: str = "swap"
    run_segment_id: Optional[str] = None
    prompt_override: Optional[str] = None
    status: Optional[str] = None
    pre_roll: float = 0.0
    post_roll: float = 0.0
    # What the segment's CURRENT result is: "model" (an Omni output),
    # "void_fill" (an applied AICP-358 composite, see *fill*) or "unknown"
    # (e.g. a hand-placed clip, or a composite whose delivery is not
    # confirmed); None = not known (files outside the database).
    result_kind: Optional[str] = None
    # The applied fill behind a "void_fill" result: {job_id, mask_box,
    # frames_masked, applied_at, delivery}.
    fill: Optional[dict] = None

    @property
    def clip_sec(self) -> float:
        """What a generation of this segment is billed on (spend._clip_sec)."""
        return self.end - self.start + self.pre_roll + self.post_roll


def segments_from(items: Sequence[dict]) -> list[Seg]:
    """Validate ``[{index, start_sec, end_sec, action?, run_segment_id?, ...}]``."""
    out = []
    for it in items:
        if not isinstance(it, dict):
            raise QAToolError("each segment must be an object")
        try:
            seg = Seg(index=int(it["index"]), start=float(it["start_sec"]),
                      end=float(it["end_sec"]), action=str(it.get("action") or "swap"),
                      run_segment_id=it.get("run_segment_id"),
                      prompt_override=it.get("prompt_override"),
                      status=it.get("status"),
                      pre_roll=float(it.get("pre_roll_sec") or 0.0),
                      post_roll=float(it.get("post_roll_sec") or 0.0),
                      result_kind=it.get("result_kind"),
                      fill=it.get("fill"))
        except (KeyError, TypeError, ValueError) as exc:
            raise QAToolError(f"bad segment {it!r}: {exc}") from None
        if not (math.isfinite(seg.start) and math.isfinite(seg.end) and seg.end > seg.start >= 0):
            raise QAToolError(f"segment {seg.index} has bad bounds {seg.start}-{seg.end}")
        if seg.action not in ("swap", "keep"):
            raise QAToolError(f"segment {seg.index} has unknown action {seg.action!r}")
        if seg.result_kind not in (None, "model", "void_fill", "unknown"):
            raise QAToolError(f"segment {seg.index} has unknown result_kind {seg.result_kind!r}")
        if seg.fill is not None and (not isinstance(seg.fill, dict)
                                     or not seg.fill.get("job_id")):
            raise QAToolError(f"segment {seg.index}: fill must be an object with a job_id")
        if seg.fill is not None and seg.result_kind is None:
            seg.result_kind = "void_fill"
        out.append(seg)
    out.sort(key=lambda s: s.start)
    if not out:
        raise QAToolError("no segments")
    return out


# ---------------------------------------------------------------------------
# Frame-level helpers
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _in_band(box, band_y) -> bool:
    return band_y[0] <= (box[1] + box[3]) / 2 <= band_y[1]


def _frames_in(scan, seg: Seg) -> list:
    return [f for f in scan.scores if seg.start - 1e-6 <= f.t < seg.end - 1e-6]


def _static_flags(frames, fps: float) -> list[bool]:
    """Per frame: is it part of a run of a stable hit box >= STATIC_RUN_SEC?"""
    flags = [False] * len(frames)
    need = max(2, int(round(STATIC_RUN_SEC * fps)))
    i = 0
    while i < len(frames):
        if frames[i].box is None:
            i += 1
            continue
        j = i
        while (j + 1 < len(frames) and frames[j + 1].box is not None
               and _iou(frames[j + 1].box, frames[j].box) >= STATIC_IOU):
            j += 1
        if j - i + 1 >= need:
            for k in range(i, j + 1):
                flags[k] = True
        i = j + 1
    return flags


def _longest_static_sec(frames, fps: float) -> float:
    best = cur = 0
    prev = None
    for f in frames:
        if f.box is not None and prev is not None and _iou(f.box, prev) >= STATIC_IOU:
            cur += 1
        else:
            cur = 1 if f.box is not None else 0
        best = max(best, cur)
        prev = f.box
    return round(best / fps, 2)


def _segment_text(frames, fps: float, band_y) -> dict:
    """What the source shows in one segment: in-band hits, static share."""
    hits = [f for f in frames if f.box is not None and _in_band(f.box, band_y)]
    flags = _static_flags(frames, fps)
    static_hits = sum(1 for f, s in zip(frames, flags) if s and f.box is not None)
    all_hits = sum(1 for f in frames if f.box is not None)
    share = static_hits / all_hits if all_hits else 0.0
    return {
        "frames": len(frames),
        "band_hits": len(hits),
        "longest_static_sec": _longest_static_sec(frames, fps),
        "static_share": round(share, 2),
        "static_text": (_longest_static_sec(frames, fps) >= STATIC_RUN_SEC
                        and share >= STATIC_SEGMENT_SHARE),
    }


def _face_ratio(faces, seg: Seg) -> Optional[float]:
    """Share of sampled frames in *seg* with a face; None without samples."""
    if faces is None:
        return None
    inside = [n for t, n in faces if seg.start <= t < seg.end]
    if not inside:
        return None
    return round(sum(1 for n in inside if n > 0) / len(inside), 2)


def _percentiles(boxes) -> Optional[dict]:
    if not boxes:
        return None
    a = np.array(boxes)
    p = lambda col, q: round(float(np.percentile(a[:, col], q)), 3)  # noqa: E731
    return {"n": len(boxes), "x0_p5": p(0, 5), "x1_p95": p(2, 95),
            "y0_p5": p(1, 5), "y1_p95": p(3, 95)}


def _sheet_times(seg: Seg, n: int = 6) -> list[float]:
    length = seg.end - seg.start
    return [round(seg.start + length * (i + 0.5) / n, 2) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Pre-launch segmentation review
# ---------------------------------------------------------------------------

def review_segmentation(source: str, segments: Sequence[Seg], *, config=None,
                        price_per_segment: float, faces=None, fps: float = 10.0,
                        out_dir: Optional[str] = None, admission_estimate=None,
                        location: Optional[dict] = None) -> dict:
    """Propose ``keep`` for swap segments that need no caption removal.

    Every proposed swap -> keep is a CANDIDATE (``needs_visual_check``): the
    detector misses caption styles it was not configured for, a caption held
    still over a faceless shot looks like an end card, and the face sample can
    miss a person. The contact sheet in *out_dir* is what confirms it.

    Two scans of the source: the configured band decides hits (one band, so a
    stronger cluster outside it cannot mask an in-band caption); a band
    ``BAND_WIDEN`` taller on each side only measures where the captions sit.
    *faces* is an optional ``[(t, n_faces), ...]`` sample (CLI ``--faces``).
    *price_per_segment* is the expected bill per clip (observed); the optional
    *admission_estimate(seconds) -> credits* is what the spend cap reserves.
    Zero hits count as "no captions" only when the detector sees captions on
    this source at all (calibrated).

    *location* is a :func:`app.qa_tools.locate_captions` result for the
    source, given when the project names no band of its own
    (``qa_tools.has_band``). The located candidate band is then re-measured
    with the normal detector, and the band the whole review uses (rows,
    keeps, calibration) is chosen by :func:`effective_band`; ``band`` reports
    ``verdict`` ``located`` / ``located_uncertain`` / ``not_found`` with the
    evidence (``band.location``) when it is not the configured band.
    """
    cfg = caption_config(config)
    scan = qa_tools.scan_captions(source, fps=fps, config=cfg)
    calibration = _caption_calibration(scan, fps)
    located = None
    if location is not None:
        located = effective_band(source, cfg, location, scan, calibration, fps=fps,
                                 out_dir=out_dir)
        if located["use_candidate"]:
            cfg, scan, calibration = located.pop("config"), located.pop("scan"), \
                located.pop("calibration")
        located.pop("config", None), located.pop("scan", None), located.pop("calibration", None)
    band = tuple(cfg.band_y)
    wide_cfg = cfg.model_copy(update={"band_y": (max(0.0, band[0] - BAND_WIDEN),
                                                 min(1.0, band[1] + BAND_WIDEN))})
    wide = qa_tools.scan_captions(source, fps=fps, config=wide_cfg)
    calibrated = calibration["calibrated"]

    rows, caption_boxes = [], []
    for seg in segments:
        frames = _frames_in(scan, seg)
        wframes = _frames_in(wide, seg)
        text = _segment_text(frames, fps, band)
        wtext = _segment_text(wframes, fps, band)
        # Static (UI / end-card) text on either scan: the configured band may
        # cut an end card's text box in half and break its stability.
        static = text["static_text"] or wtext["static_text"]
        text.update(longest_static_sec=max(text["longest_static_sec"],
                                           wtext["longest_static_sec"]),
                    static_share=max(text["static_share"], wtext["static_share"]),
                    static_text=static)
        face_ratio = _face_ratio(faces, seg)
        row = {"index": seg.index, "start_sec": round(seg.start, 3),
               "end_sec": round(seg.end, 3), "clip_sec": seg.clip_sec,
               "action": seg.action, **text,
               "face_ratio": face_ratio, "proposed_action": seg.action,
               "reason": "", "needs_visual_check": False}
        if seg.action == "keep":
            row["reason"] = "already keep"
        elif not frames:
            row["reason"] = "no frames decoded in this window; left as is"
        elif text["band_hits"] == 0 and calibrated:
            row.update(proposed_action="keep", needs_visual_check=True,
                       reason="no caption hits in the caption band; confirm on the "
                              "contact sheet (a caption style the detector is not "
                              "configured for also gives 0 hits)")
        elif text["band_hits"] == 0:
            row["reason"] = ("no hits, but the detector does not see this video's "
                             "captions (uncalibrated): not evidence, left as swap")
        elif static and (face_ratio is None or face_ratio <= NO_FACE_RATIO):
            row.update(
                proposed_action="keep", needs_visual_check=True,
                reason=("end card: text held still for %.1f s, %s; confirm on the "
                        "contact sheet that there is no person and no caption" % (
                            text["longest_static_sec"],
                            "faces not sampled" if face_ratio is None
                            else "face in %d%% of samples" % round(face_ratio * 100))))
        elif static:
            row["reason"] = ("static text with a person on screen (phone UI?): "
                             "swap, keep 'do not change the phone screen' in the prompt")
        else:
            row["reason"] = "burned captions"
        if row["proposed_action"] == "swap":
            flags = _static_flags(wframes, fps)
            caption_boxes += [f.box for f, st in zip(wframes, flags)
                              if f.box is not None and not st]
        if out_dir and row["proposed_action"] == "keep" and seg.action == "swap":
            path = os.path.join(out_dir, f"review_seg{seg.index}.jpg")
            qa_tools.contact_sheet(source, path, times=_sheet_times(seg))
            row["contact_sheet"] = path
        rows.append(row)

    band_out = _band_check(wide, rows, segments, band, caption_boxes)
    band_out["effective_band_y"] = list(band)
    if located is not None:
        band_out = _with_location(band_out, located)
    return {
        "source": source,
        "calibration": calibration,
        "segments": rows,
        "band": band_out,
        "location_checked": location is not None,
        "cost": _launch_cost(rows, price_per_segment, admission_estimate),
        "approval_required": True,
    }


# A located band counts as confirmed when the normal detector, scanning it,
# hits within this much of at least LOCATED_COVERAGE of the locator's sampled
# caption moments.
LOCATED_COVERAGE = 0.7
_COVERAGE_WINDOW_SEC = 0.25


def _inside(position: dict, band) -> bool:
    return (position.get("y0_p5", 1.0) >= band[0] - EDGE_EPS
            and position.get("y1_p95", 0.0) <= band[1] + EDGE_EPS)


def _coverage(scan, times: Sequence[float]) -> Optional[float]:
    if not times:
        return None
    hit_times = [f.t for f in scan.scores if f.box is not None]
    covered = sum(1 for t in times
                  if any(abs(h - t) <= _COVERAGE_WINDOW_SEC + 1e-6 for h in hit_times))
    return round(covered / len(times), 3)


def effective_band(source: str, cfg, location: dict, scan, calibration: dict, *,
                   fps: float = 10.0, out_dir: Optional[str] = None) -> dict:
    """Which band a review of a project WITHOUT its own band uses.

    The configured (default) band stays when it calibrates and every
    caption-like position the locator found lies inside it. Otherwise the
    located candidate band is scanned with the normal detector; it is used
    when THAT calibrates. ``verdict``:

    * ``ok`` -- the configured band (nothing to change);
    * ``located`` -- the candidate, confirmed: locator ``located``, the
      normal scan calibrates on it and hits at >= ``LOCATED_COVERAGE`` of the
      locator's caption moments (eligible for the autopilot's automatic save);
    * ``located_uncertain`` -- a calibrating candidate that misses one of
      those (an eye check decides);
    * ``not_found`` -- no candidate, or one the normal detector does not see:
      the detector's pixel rule probably does not match this caption style.

    Returns the evidence plus ``use_candidate`` and, when used, the
    candidate's ``config`` / ``scan`` / ``calibration`` (popped by the caller).
    """
    configured = list(cfg.band_y)
    cand = location.get("candidate_band_y")
    members = [location["positions"][i] for i in location.get("members") or []]
    outside = [p for p in members if not _inside(p, configured)]
    others = [location["positions"][i] for i in location.get("competing") or []]
    out = {"location": _location_summary(location), "configured_band_y": configured,
           "candidate_band_y": cand, "default_calibration": calibration,
           "use_candidate": False,
           "outside_members": [p.get("band_y") for p in outside],
           "outside_text": [p.get("band_y") for p in others if not _inside(p, configured)]}
    if calibration["calibrated"] and not outside:
        out["verdict"] = "ok"
        _outside_sheet(source, out, [p for p in others if not _inside(p, configured)],
                       configured, out_dir)
        return out
    if not cand:
        out["verdict"] = "not_found"
        return out
    ccfg = cfg.model_copy(update={"band_y": (float(cand[0]), float(cand[1]))})
    cscan = qa_tools.scan_captions(source, fps=fps, config=ccfg)
    ccal = _caption_calibration(cscan, fps)
    times = [t for p in members for t in (p.get("sample_times") or [])]
    coverage = _coverage(cscan, times)
    out.update(candidate_calibration=ccal, coverage=coverage)
    if not ccal["calibrated"]:
        out["verdict"] = "not_found"
        if calibration["calibrated"]:
            _outside_sheet(source, out, outside + [p for p in others
                                                   if not _inside(p, configured)],
                           configured, out_dir)
        out["note"] = ("the located text at y %.2f-%.2f does not calibrate with the "
                       "normal detector (%.0f %% of the frames)" % (
                           cand[0], cand[1], 100 * ccal["hit_ratio"]))
        return out
    confirmed = (location.get("state") == qa_tools.LOCATED
                 and coverage is not None and coverage >= LOCATED_COVERAGE)
    out["verdict"] = "located" if confirmed else "located_uncertain"
    out.update(use_candidate=True, config=ccfg, scan=cscan, calibration=ccal)
    if out_dir:
        try:
            sheet = qa_tools.location_sheet(source, os.path.join(out_dir, "review_location.jpg"),
                                            location, other_bands=[configured])
            out["sheet"] = (sheet or {}).get("out")
        except QAToolError as exc:  # the sheet is evidence, not the verdict
            out["sheet_error"] = str(exc)[:300]
    return out


def out_of_band_check(location: dict, config=None) -> dict:
    """Post-run QA aid for a project WITHOUT its own band: where the locator
    finds caption-like text on the SOURCE vs the band the QA measured with.
    A position outside that band means the QA cannot see captions (or
    leftovers) there -- a warning, never a verdict change."""
    band = list(caption_config(config).band_y)
    idx = list(location.get("members") or []) + list(location.get("competing") or [])
    plausible = [location["positions"][i] for i in idx if i < len(location["positions"])]
    outside = [p for p in plausible if not _inside(p, band)]
    out = {"location": _location_summary(location), "band_y": band,
           "outside": [p.get("band_y") for p in outside], "warning": None}
    if outside:
        out["warning"] = (
            "caption-like text on the source sits outside the QA band y %.2f-%.2f (at %s): "
            "captions or leftovers there are not measured -- set Band y on the project page "
            "(QA caption detector) to %s and run the QA again" % (
                band[0], band[1], ", ".join("y %.2f-%.2f" % tuple(b) for b in
                                            [p.get("band_y") for p in outside]),
                "y %.2f-%.2f" % tuple(location["candidate_band_y"])
                if location.get("candidate_band_y") and all(
                    _inside(p, location["candidate_band_y"]) for p in outside)
                else "a band covering the captions' height"))
    return out


def _outside_sheet(source: str, out: dict, positions: list, configured, out_dir) -> None:
    """Evidence for the segmentation eye check when the configured band is
    kept but the locator saw centred text outside it: frames at that text's
    own moments, the configured band in green, the outside text in amber."""
    if not out_dir or not positions:
        return
    times = sorted({t for p in positions for t in (p.get("sample_times") or [])})
    if not times:
        return
    try:
        sheet = qa_tools.contact_sheet(
            source, os.path.join(out_dir, "review_location.jpg"),
            times=qa_tools._spread(times, 6), columns=min(6, len(times)),
            bands=[configured] + [p["band_y"] for p in positions if p.get("band_y")])
        out["sheet"] = (sheet or {}).get("out")
    except QAToolError as exc:  # evidence, not the verdict
        out["sheet_error"] = str(exc)[:300]


def _location_summary(location: dict) -> dict:
    keep = ("y_range", "support", "nonstatic_frames", "static_frames", "caption_like",
            "windows_hit", "y0_p5", "y1_p95", "band_y")
    return {"state": location.get("state"), "candidate_band_y": location.get("candidate_band_y"),
            "frames": location.get("frames"), "fps": location.get("fps"),
            "members": list(location.get("members") or []),
            "competing": list(location.get("competing") or []),
            "lines_outside_band": location.get("lines_outside_band"),
            "positions": [{k: p.get(k) for k in keep if k in p}
                          for p in (location.get("positions") or [])[:12]],
            "reasons": list(location.get("reasons") or []),
            "detector": location.get("detector")}


def _with_location(band_out: dict, located: dict) -> dict:
    """The band check of the effective band, plus the location verdict."""
    band_out = dict(band_out)
    band_out["location"] = located["location"]
    band_out["location_verdict"] = located["verdict"]
    band_out["configured_band_y"] = located["configured_band_y"]
    for k in ("coverage", "candidate_calibration", "sheet", "default_calibration"):
        if k in located:
            band_out[k] = located[k]
    verdict = located["verdict"]
    if located.get("outside_text"):
        band_out["outside_text"] = located["outside_text"]
    if verdict == "ok":
        if located.get("outside_text"):
            # Captions may also sit elsewhere: the segmentation eye check shows it.
            band_out["needs_visual_check"] = True
            band_out["note"] = (band_out.get("note") or "") + (
                "; other centred text outside the band at %s: check the sheets that it is "
                "not captions" % ", ".join("y %.2f-%.2f" % tuple(b)
                                          for b in located["outside_text"]))
        return band_out
    if verdict == "not_found":
        note = located.get("note") or "no caption band located"
        # The default band calibrates, but the locator saw caption-like text
        # outside it that the detector could not confirm: an eye check, not a
        # silent pass.
        unconfirmed = bool((located.get("default_calibration") or {}).get("calibrated")
                           and (located.get("outside_members") or located.get("outside_text")))
        if unconfirmed:
            note += ("; the default band calibrates, but centred text outside it (y %s) was "
                     "not confirmed: check the sheets that it is not captions" % ", ".join(
                         "%.2f-%.2f" % tuple(b) for b in (located.get("outside_members") or [])
                         + (located.get("outside_text") or [])))
        band_out.update(verdict="not_found", proposed_band_y=None,
                        needs_visual_check=unconfirmed or bool(band_out.get("needs_visual_check")),
                        note=note)
        return band_out
    # located / located_uncertain: the candidate is the effective band; a
    # widen of it (captions crossing its edges) is folded in and makes it
    # uncertain.
    proposed = list(located["candidate_band_y"])
    if band_out.get("verdict") == "widen" and band_out.get("proposed_band_y"):
        w = band_out["proposed_band_y"]
        proposed = [round(min(proposed[0], w[0]), 2), round(max(proposed[1], w[1]), 2)]
        verdict = "located_uncertain"
    band_out.update(
        verdict=verdict, proposed_band_y=proposed,
        needs_visual_check=verdict != "located",
        note=("captions located at y %.2f-%.2f (%s)" % (proposed[0], proposed[1],
              "confirmed by the detector" if verdict == "located"
              else "; ".join(located["location"]["reasons"]) or "check the sheet")))
    return band_out


def _band_check(scan, rows, segments, band, caption_boxes) -> dict:
    proposed = {r["index"]: r["proposed_action"] for r in rows}
    edge = {}
    for seg in segments:
        for f in _frames_in(scan, seg):
            b = f.box
            if b is None:
                continue
            # Straddles (or sits on) the configured band's top or bottom edge.
            if (b[1] <= band[0] + EDGE_EPS <= b[3]) or (b[1] <= band[1] - EDGE_EPS <= b[3]):
                edge[seg.index] = edge.get(seg.index, 0) + 1
    edge_rows = [{"index": i, "frames": n, "proposed_action": proposed.get(i)}
                 for i, n in sorted(edge.items())]
    geometry = _percentiles(caption_boxes)
    out = {"configured_band_y": list(band), "caption_geometry": geometry,
           "edge_hits": edge_rows, "proposed_band_y": None,
           "needs_visual_check": False}
    if geometry is None:
        out["verdict"] = "no_captions_seen"
        out["note"] = "no caption boxes in the swap segments"
        return out
    if geometry["y0_p5"] < band[0] or geometry["y1_p95"] > band[1]:
        out["verdict"] = "widen"
        out["proposed_band_y"] = [round(max(0.0, geometry["y0_p5"] - BAND_MARGIN), 2),
                                  round(min(1.0, geometry["y1_p95"] + BAND_MARGIN), 2)]
        out["needs_visual_check"] = True
        out["note"] = ("text boxes in the swap segments cross the configured band: "
                       "if the contact sheets show they are captions (not phone UI), "
                       "PATCH the project's caption_detector.band_y before trusting "
                       "a clean verdict")
    elif any(r["proposed_action"] == "swap" for r in edge_rows):
        out["verdict"] = "edge_touched"
        out["needs_visual_check"] = True
        out["note"] = ("hits touch the band edge inside swap segments: check the "
                       "sheets; the captions themselves sit inside the band")
    elif edge_rows:
        out["verdict"] = "edge_touched_keep_only"
        out["note"] = ("hits touch the band edge only in keep segments (end card / "
                       "UI text); the captions sit inside the band (y %.2f-%.2f): "
                       "no band change" % (geometry["y0_p5"], geometry["y1_p95"]))
    else:
        out["verdict"] = "ok"
        out["note"] = "captions sit inside the band (y %.2f-%.2f)" % (
            geometry["y0_p5"], geometry["y1_p95"])
    return out


def _launch_cost(rows, price, admission_estimate=None) -> dict:
    current = [r for r in rows if r["action"] == "swap"]
    proposed = [r for r in rows if r["proposed_action"] == "swap"]
    out = {"model": REMOVAL_MODEL, "credits_per_segment": price,
           "swap_segments_current": len(current), "swap_segments_proposed": len(proposed),
           "credits_current": round(price * len(current), 2),
           "credits_proposed": round(price * len(proposed), 2),
           "credits_saved": round(price * (len(current) - len(proposed)), 2)}
    if admission_estimate is not None:
        # What the spend cap reserves at launch (per-second tariff): the
        # budget check, not the bill.
        out["budget_check_credits_proposed"] = round(sum(
            admission_estimate(r["clip_sec"]) for r in proposed), 2)
    return out


# ---------------------------------------------------------------------------
# 2. Post-run QA
# ---------------------------------------------------------------------------

def _spans(frames, fps: float, merge_gap_sec: float = 0.5) -> list[dict]:
    spans: list[dict] = []
    for f in frames:
        if f.box is None:
            continue
        if spans and f.t - spans[-1]["end_sec"] <= merge_gap_sec + 1e-6:
            sp = spans[-1]
            sp["end_sec"] = f.t
            sp["frames"] += 1
            sp["max_score"] = max(sp["max_score"], f.score)
            sp["_boxes"].append(f.box)
            sp["_times"].append(f.t)
        else:
            spans.append({"start_sec": f.t, "end_sec": f.t, "frames": 1,
                          "max_score": f.score, "_boxes": [f.box], "_times": [f.t]})
    for sp in spans:
        a = np.array(sp["_boxes"])
        sp["box"] = [round(float(a[:, 0].min()), 3), round(float(a[:, 1].min()), 3),
                     round(float(a[:, 2].max()), 3), round(float(a[:, 3].max()), 3)]
    return spans


def _crop(img, box, pad: float = 0.08):
    if img is None:
        return None
    h, w = img.shape[:2]
    x0 = int(max(0.0, box[0] - pad) * w)
    x1 = int(min(1.0, box[2] + pad) * w)
    y0 = int(max(0.0, box[1] - pad) * h)
    y1 = int(min(1.0, box[3] + pad) * h)
    return img[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]


def _word_moments(spans: list[dict]) -> list[tuple[float, list]]:
    """One (time, box) per distinct caption box in *spans*: the detector box
    stays put while one word is on screen and jumps when the next appears, so
    each word gets its own crop (its middle hit)."""
    out = []
    for sp in spans:
        boxes, times = sp["_boxes"], sp["_times"]
        start = 0
        for i in range(1, len(boxes) + 1):
            if i == len(boxes) or _iou(boxes[i], boxes[i - 1]) < WORD_BOX_IOU:
                mid = (start + i - 1) // 2
                out.append((times[mid], boxes[mid]))
                start = i
    return out


def _crop_sheet(source: str, final: str, spans: list[dict], out_path: str) -> dict:
    """Result crop next to the source crop at each word's middle hit."""
    cells, times = [], []
    cw, ch = 360, 160
    for t, box in _word_moments(spans)[:MAX_CROPS_PER_SEGMENT]:
        cells.append(qa_tools._cell(_crop(qa_tools._grab_frame(final, t), box),
                                    f"R {t:.2f}s", cw, ch))
        cells.append(qa_tools._cell(_crop(qa_tools._grab_frame(source, t), box),
                                    f"S {t:.2f}s", cw, ch))
        times.append(t)
    qa_tools._write_grid(cells, 2, out_path)
    return {"out": out_path, "times": times}


def mask_box_from(frames) -> Optional[list[float]]:
    """The fill's inclusive caption box from detector hit frames: the union
    of their caption lines, padded, mirrored when centred and validated
    exactly like ``pipeline_segment_fill.measure_box`` (which runs it over
    every frame of the cut clip; QA has the 10 fps source scan, so this is an
    estimate of the box the preview will measure)."""
    from app.pipeline_segment_fill import FillError, caption_box

    try:
        return caption_box(frames)
    except FillError:
        return None


def _centred(frames) -> bool:
    from app.pipeline_segment_fill import captions_centred

    return captions_centred(frames)


def _fill_sheet(source: str, final: str, times: Sequence[float], box, out_path: str) -> dict:
    """Result next to source, cropped to the fill's mask box, at *times*: the
    eye check for ghosts and smears the detector cannot see."""
    cells = []
    cw, ch = 360, 160
    for t in times:
        cells.append(qa_tools._cell(_crop(qa_tools._grab_frame(final, t), box, pad=0.04),
                                    f"R {t:.2f}s", cw, ch))
        cells.append(qa_tools._cell(_crop(qa_tools._grab_frame(source, t), box, pad=0.04),
                                    f"S {t:.2f}s", cw, ch))
    qa_tools._write_grid(cells, 2, out_path)
    return {"out": out_path, "times": list(times)}


def _fill_sheet_times(seg: Seg, sframes, fps: float) -> list[float]:
    """Where the source shows caption words (one moment per word), else spread
    over the segment. A sample: it cannot count the patched frames."""
    moments = [t for t, _ in _word_moments(_spans(sframes, fps))]
    if len(moments) > MAX_CROPS_PER_SEGMENT:
        step = len(moments) / MAX_CROPS_PER_SEGMENT
        moments = [moments[int(i * step)] for i in range(MAX_CROPS_PER_SEGMENT)]
    return moments or _sheet_times(seg)


def _covered(box, by, pad: float = SOURCE_MATCH_PAD) -> float:
    """Share of *box* inside *by* grown by *pad* on every side."""
    grown = (by[0] - pad, by[1] - pad, by[2] + pad, by[3] + pad)
    ix = max(0.0, min(box[2], grown[2]) - max(box[0], grown[0]))
    iy = max(0.0, min(box[3], grown[3]) - max(box[1], grown[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 0.0


def _matches_source(sp: dict, sframes, window_sec: float = 1.0) -> bool:
    """At least ``FP_MATCH_SHARE`` of *sp*'s hits lie (mostly) on a SOURCE hit
    box within +-window_sec. Padded and not all-or-nothing: Omni re-renders an
    end card with its text a few pixels off, and one frame of 52 even further
    (328341d4 seg 8), while a caption sits well clear of it."""
    matched = sum(
        1 for t, box in zip(sp["_times"], sp["_boxes"])
        if any(f.box is not None and abs(f.t - t) <= window_sec
               and _covered(box, f.box) >= 0.5 for f in sframes))
    return matched >= FP_MATCH_SHARE * len(sp["_times"])


def _caption_calibration(scan, fps: float) -> dict:
    """Calibration on CAPTION-LIKE source hits only: hits inside static runs
    (UI, an end card) prove nothing about whether the detector sees this
    video's captions."""
    flags = _static_flags(scan.scores, fps)
    caption_like = sum(1 for f, st in zip(scan.scores, flags) if f.box is not None and not st)
    ratio = caption_like / scan.frames if scan.frames else 0.0
    return {"frames": scan.frames, "hits": scan.hits, "caption_like_hits": caption_like,
            "hit_ratio": round(ratio, 4), "calibrated": ratio >= CALIBRATION_MIN_RATIO}


def _qa_window(segments: Sequence[Seg], i: int, slack: float) -> Seg:
    """Segment *i*'s window for result hits: a swap segment reaches *slack*
    into an adjacent keep segment (picture drift may push its leftover across
    the boundary, where a keep verdict would swallow it)."""
    seg = segments[i]
    if seg.action != "swap" or slack <= 0:
        return seg
    start, end = seg.start, seg.end
    if i > 0 and segments[i - 1].action == "keep":
        start = max(segments[i - 1].start, start - slack)
    if i + 1 < len(segments) and segments[i + 1].action == "keep":
        end = min(segments[i + 1].end, end + slack)
    return Seg(index=seg.index, start=start, end=end, action=seg.action)


def segment_timing(points: Sequence[dict], segments: Sequence[Seg], *,
                   search: int = ALIGN_SEARCH_FRAMES) -> list[dict]:
    """Each segment's picture offset against the source, from the alignment
    points (``qa_tools.alignment_check``'s ``_points``: final time ``t`` and
    the best ``offset`` in 30 fps frames; negative = the final shows an
    EARLIER source frame, i.e. the picture is late against the sound).

    Measured on the points at least ``TIMING_EDGE_SEC`` inside the segment
    (the cut itself may sit a few frames off). ``status``:

    * ``in_sync`` -- the median is within +-``SYNC_TOLERANCE_FRAMES``;
    * ``off`` -- measured further off (``beyond_search`` when most points sat
      at the search limit: the real offset is larger than ``offset_frames``);
    * ``unknown`` -- fewer than ``TIMING_MIN_POINTS`` points (a short segment):
      nothing is claimed either way.
    """
    out = []
    for seg in sorted(segments, key=lambda sg: sg.start):
        lo, hi = seg.start + TIMING_EDGE_SEC, seg.end - TIMING_EDGE_SEC
        offs = [int(p["offset"]) for p in points if lo - 1e-6 <= float(p["t"]) <= hi + 1e-6]
        row = {"index": seg.index, "points": len(offs), "offset_frames": None,
               "beyond_search": False, "status": "unknown"}
        if len(offs) >= TIMING_MIN_POINTS:
            med = float(np.median(offs))
            censored = sum(1 for o in offs if search and abs(o) >= search)
            row["offset_frames"] = med
            row["beyond_search"] = censored * 2 >= len(offs)
            row["status"] = ("off" if row["beyond_search"] or abs(med) > SYNC_TOLERANCE_FRAMES
                             else "in_sync")
        out.append(row)
    return out


def timing_verified(timing: dict, rows: Sequence[dict], timeline_verified: bool) -> bool:
    """The final is on the source timeline: as long as the source, no swap
    segment measured off (an ``accepted`` one counts as decided), and -- where
    a segment is too short to measure -- the run-wide 10 s windows agree."""
    if not timing.get("length_ok", False):
        return False
    swaps = [r for r in rows if r.get("action") == "swap"]
    states = [(r.get("timing") or {}).get("status") for r in swaps]
    if any(st == "off" for st in states):
        return False
    return bool(timeline_verified) or all(st in ("in_sync", "accepted") for st in states)


def _shift_sec(timing_row: Optional[dict], qa: Optional[dict] = None) -> Optional[float]:
    """How far (s) a segment's clip may sit off its nominal bounds in the
    final, or None when nothing bounds it: a shift at the search limit is at
    least that large, and an unmeasured segment (too short) is bounded only by
    the run-wide 10 s windows when those verified the timeline."""
    if timing_row and timing_row.get("offset_frames") is not None \
            and not timing_row.get("beyond_search"):
        return abs(float(timing_row["offset_frames"])) / qa_tools.ALIGN_FPS
    if timing_row and timing_row.get("beyond_search"):
        return None
    al = (qa or {}).get("alignment") or {}
    if (qa or {}).get("timeline_verified") and al.get("max_abs_window_median") is not None:
        return float(al["max_abs_window_median"]) / qa_tools.ALIGN_FPS
    return None


def attribution(qa: dict, row: dict) -> dict:
    """Which of *row*'s leftover spans certainly belong to this segment.

    A span is the segment's own unless it lies within the picture shift (+
    ``ATTRIBUTION_PAD_SEC``) of a cut to a neighbour: at its start the shift of
    THIS segment's clip decides where the cut sits in the final, at its end the
    next segment's. A shift nothing bounds (:func:`_shift_sec` None: at the
    search limit, or unmeasured on an unverified timeline) makes every span of
    the segment doubtful (the cut may sit anywhere in it, gate R2-2). ``{"own": [...], "ambiguous": [{span, edge, neighbour}]}``.
    A QA without per-segment timing (written before 2026-11-02) claims nothing:
    every span is own when its run-wide timeline is verified, else ambiguous.
    """
    spans = list(row.get("leftovers") or [])
    timing = qa.get("timing")
    if not timing:
        if qa.get("timeline_verified", True):
            return {"own": spans, "ambiguous": []}
        return {"own": [], "ambiguous": [{"span": sp, "edge": None, "neighbour": None}
                                         for sp in spans]}
    rows = sorted(qa["segments"], key=lambda r: r["start_sec"])
    by_idx = {t["index"]: t for t in timing.get("segments") or []}
    by_idx.update({r["index"]: r["timing"] for r in rows if r.get("timing")})
    pos = next(i for i, r in enumerate(rows) if r["index"] == row["index"])
    prev = rows[pos - 1] if pos > 0 else None
    nxt = rows[pos + 1] if pos + 1 < len(rows) else None
    inf = float("inf")                      # an unbounded shift: the whole edge is doubtful
    s_shift = _shift_sec(by_idx.get(row["index"]), qa) if prev else 0.0
    e_shift = _shift_sec(by_idx.get(nxt["index"]), qa) if nxt else 0.0
    start_zone = (inf if s_shift is None else s_shift + ATTRIBUTION_PAD_SEC) if prev else 0.0
    end_zone = (inf if e_shift is None else e_shift + ATTRIBUTION_PAD_SEC) if nxt else 0.0
    own, amb = [], []
    for sp in spans:
        a, b = float(sp["start_sec"]), float(sp["end_sec"])
        if prev is not None and a < float(row["start_sec"]) + start_zone:
            amb.append({"span": sp, "edge": "start", "neighbour": prev["index"]})
        elif nxt is not None and b > float(row["end_sec"]) - end_zone:
            amb.append({"span": sp, "edge": "end", "neighbour": nxt["index"]})
        else:
            own.append(sp)
    return {"own": own, "ambiguous": amb}


def post_run_qa(source: str, final: str, segments: Sequence[Seg], *, config=None,
                fps: float = 10.0, out_dir: Optional[str] = None,
                check_alignment: bool = True,
                confirmed_false_positives: Sequence[int] = (),
                confirmed_fills: Sequence[str] = ()) -> dict:
    """Per-segment caption verdicts for a final on the source timeline.

    Mapping a hit to a segment by its time is only valid when the final is on
    the source timeline: a duration gate refuses a final of another length,
    and the picture alignment check (``qa_tools.alignment_check``, searched
    +-``ALIGN_SEARCH_FRAMES``) measures every segment's offset
    (:func:`segment_timing`, ``timing``): a swap segment more than +-1 frame
    off is a lip-sync defect (``sync_off_segments``) and the run is clean only
    when ``timing["verified"]`` (2026-11-02). ``timeline_verified`` keeps the
    run-wide 10 s window rule; a swap segment's window reaches that drift
    (+0.1 s, at most ``MAPPING_TOLERANCE_FRAMES``) into an adjacent keep
    segment, and :func:`attribution` says which leftovers may be a
    neighbour's.

    ``false_positive`` = the segment's source text is static (see
    :func:`_segment_text`) and >= ``FP_MATCH_SHARE`` of the hits lie on a
    source text box. A caption line held still for 2 s looks exactly like
    that, so a false positive blocks ``clean`` until a look at its crop sheet
    confirms it (*confirmed_false_positives*, segment indices). A keep segment
    whose source shows captions (not static text) is ``keep_captions`` and
    blocks ``clean`` too: those captions ship. A calibrated 0-hit segment is
    ``clean`` for the detector only: invented text in another style and
    picture changes need the comparison sheet (``qa_compare.jpg``).

    A segment whose result is a VOID fill (``Seg.result_kind == "void_fill"``)
    keeps the detector's verdict, plus ``fill_needs_visual_check`` until its
    fill job id is in *confirmed_fills* (the eye check on its ``fill_sheet``
    was done): the white-text detector is blind to grey ghosts and smears
    (flow doc step 6), so an unconfirmed fill blocks ``clean``. A fill
    confirmation never clears a false-positive confirmation or vice versa.
    Every swap row carries ``mask_box``: the source caption box a fill of that
    segment would mask (see :func:`mask_box_from`).
    """
    src_info = qa_tools._probe(source)
    fin_info = qa_tools._probe(final)
    drift = abs(fin_info.duration_sec - src_info.duration_sec)
    if drift > TIMELINE_TOLERANCE_SEC:
        raise QAToolError(
            "final is %.2f s, source %.2f s: the final is not on the source timeline "
            "(tail, stitch plan or truncated clip), so hits cannot be mapped to segments"
            % (fin_info.duration_sec, src_info.duration_sec))
    cfg = caption_config(config)
    band = tuple(cfg.band_y)
    src = qa_tools.scan_captions(source, fps=fps, config=cfg)
    fin = qa_tools.scan_captions(final, fps=fps, config=cfg)
    calibration = _caption_calibration(src, fps)
    calibrated = calibration["calibrated"]
    alignment = None
    timeline_verified = False
    slack = 0.0
    timing = None
    if check_alignment:
        al = qa_tools.alignment_check(source, final, max_offset=ALIGN_SEARCH_FRAMES)
        alignment = {k: al.get(k) for k in ("in_sync", "points", "max_abs_window_median",
                                            "frame_count_delta", "length_ok", "windows")}
        timeline_verified = bool(al["points"]) and al["length_ok"] and (
            al["max_abs_window_median"] < MAPPING_TOLERANCE_FRAMES)
        slack = min(float(al["max_abs_window_median"]), float(MAPPING_TOLERANCE_FRAMES)) \
            / qa_tools.ALIGN_FPS + 0.1
        timing = {"fps": qa_tools.ALIGN_FPS, "search_frames": ALIGN_SEARCH_FRAMES,
                  "tolerance_frames": SYNC_TOLERANCE_FRAMES,
                  "length_ok": bool(al["length_ok"]),
                  "segments": segment_timing(al.get("_points") or [], segments)}
    confirmed = set(int(i) for i in confirmed_false_positives)
    fills_ok = set(str(i) for i in confirmed_fills)

    segments = sorted(segments, key=lambda sg: sg.start)
    rows = []
    for i, seg in enumerate(segments):
        sframes = _frames_in(src, seg)
        fframes = _frames_in(fin, seg)
        window = _qa_window(segments, i, slack)
        if (window.start, window.end) != (seg.start, seg.end):
            # Slack frames from the keep side count only where the source shows
            # no text then: a keep segment's own captions are its verdict.
            # Slack frames from the keep side count unless the source shows the
            # same text box then: a keep segment's own captions/UI are its own
            # verdict, a caption that drifted across the boundary is not.
            extra = [f for f in _frames_in(fin, window)
                     if not (seg.start - 1e-6 <= f.t < seg.end - 1e-6)
                     and f.box is not None and not any(
                         g.box is not None and abs(g.t - f.t) < 0.5 / fps
                         and not (seg.start - 1e-6 <= g.t < seg.end - 1e-6)
                         and _covered(f.box, g.box) >= 0.5 for g in src.scores)]
            fframes = sorted(fframes + extra, key=lambda f: f.t)
        stext = _segment_text(sframes, fps, band)
        spans = _spans(fframes, fps)
        real, fp = [], []
        for sp in spans:
            (fp if stext["static_text"] and _matches_source(sp, sframes) else real).append(sp)
        row = {"index": seg.index, "run_segment_id": seg.run_segment_id,
               "start_sec": round(seg.start, 3), "end_sec": round(seg.end, 3),
               "action": seg.action, "source_hits": stext["band_hits"],
               "result_hits": sum(sp["frames"] for sp in spans),
               "needs_visual_check": False}
        if seg.action == "keep":
            sflags = _static_flags(sframes, fps)
            caption_like = sum(1 for f, st in zip(sframes, sflags)
                               if f.box is not None and not st and _in_band(f.box, band))
            row["source_caption_like_hits"] = caption_like
            # Any caption-like source text (not part of a static run), even
            # next to an end card: those frames ship as they are.
            if caption_like >= max(1, math.ceil(KEEP_CAPTION_MIN_SEC * fps - 1e-9)):
                row["verdict"] = "keep_captions"
                row["needs_visual_check"] = True
                row["note"] = ("original footage with captions on the source: they ship "
                               "unless this segment is made swap")
            else:
                row["verdict"] = "keep"
                row["note"] = "original footage (no captions, or static UI / end card)"
        elif real:
            row["verdict"] = "leftover"
        elif fp:
            row["verdict"] = "false_positive"
            row["confirmed"] = seg.index in confirmed
            row["needs_visual_check"] = not row["confirmed"]
            row["note"] = ("static UI / end-card text that is also in the source; "
                           "a caption line held still looks the same: confirm on the "
                           "crop sheet")
        elif calibrated:
            row["verdict"] = "clean"
        else:
            row["verdict"] = "clean_uncalibrated"
            row["note"] = "0 hits, but the detector does not see this source's captions"
        row["leftovers"] = [_public_span(sp) for sp in real]
        row["false_positives"] = [_public_span(sp) for sp in fp]
        if seg.action == "swap":
            row["result_kind"] = seg.result_kind
            row["mask_box"] = mask_box_from(sframes)
            # Centred source captions: a widened / grown box is mirrored about
            # the frame's centre too (pipeline_segment_fill.captions_centred).
            row["captions_centred"] = _centred(sframes)
            if seg.result_kind == "void_fill" and seg.fill:
                fill = dict(seg.fill)
                # A composite the final is not known to contain (re-delivery
                # pending, or the final was republished from elsewhere) cannot
                # be confirmed: the eye check would not be of this final.
                delivered = fill.get("delivery") in (None, "delivered")
                fill["confirmed"] = delivered and str(fill.get("job_id")) in fills_ok
                row["fill"] = fill
                if not delivered:
                    row["fill_delivery_unconfirmed"] = fill.get("delivery")
                row["fill_needs_visual_check"] = not fill["confirmed"]
                row["fill_note"] = (
                    "VOID composite: the detector result above is necessary, not "
                    "sufficient (it misses grey ghosts and smears). Look at the "
                    "fill sheet: a few visibly patched frames pass; many, or a "
                    "smear on a face / the phone screen, go to the operator; "
                    "then confirm with --confirm-fill %s" % fill.get("job_id"))
                if out_dir:
                    # A source fill applied as is has no mask box (the detector
                    # saw no caption): the eye check still gets its sheet, over
                    # the detector band.
                    box = fill.get("mask_box") or row["mask_box"] or [0.0, band[0], 1.0, band[1]]
                    row["fill_sheet_box"] = box
                    if box:
                        sheet = _fill_sheet(
                            source, final, _fill_sheet_times(seg, sframes, fps), box,
                            os.path.join(out_dir, f"qa_seg{seg.index}_fill.jpg"))
                        row["fill_sheet"] = sheet["out"]
                        row["fill_sheet_times"] = sheet["times"]
            elif seg.result_kind == "void_fill":
                # A composite without its fill record: nothing to confirm by id.
                row["fill"] = None
                row["fill_needs_visual_check"] = True
            if row.get("fill_needs_visual_check"):
                row["needs_visual_check"] = True
        if out_dir and (real or fp) and row["verdict"] != "keep":
            sheet = _crop_sheet(source, final, real or fp,
                                os.path.join(out_dir, f"qa_seg{seg.index}.jpg"))
            row["crop_sheet"] = sheet["out"]
            row["crop_times"] = sheet["times"]
            row["words"] = None  # read from the crop sheet (no OCR in the image)
        rows.append(row)

    compare = None
    swaps = [sg for sg in segments if sg.action == "swap"]
    if out_dir and swaps:
        compare = qa_tools.comparison_sheet(
            source, final, os.path.join(out_dir, "qa_compare.jpg"),
            windows=[{"source_start": sg.start, "source_end": sg.end,
                      "label": f"seg {sg.index}"} for sg in swaps],
            fractions=(0.2, 0.5, 0.8))["out"]

    dirty = [r["index"] for r in rows if r["verdict"] == "leftover"]
    unconfirmed = [r["index"] for r in rows
                   if r["verdict"] == "false_positive" and not r["confirmed"]]
    unconfirmed_fills = [{"index": r["index"],
                          "fill_job_id": (r.get("fill") or {}).get("job_id")}
                         for r in rows if r.get("fill_needs_visual_check")]
    warnings = []
    if alignment is not None and not alignment["in_sync"]:
        warnings.append(
            "picture is up to %g frame(s) off the source in places (the flow doc "
            "wants +-1 at final QA): look at the comparison sheet / lip sync"
            % alignment["max_abs_window_median"])
    timing_rows = {t["index"]: t for t in (timing or {}).get("segments") or []}
    for r in rows:
        if r["index"] in timing_rows:
            r["timing"] = timing_rows[r["index"]]
    sync_off = [r["index"] for r in rows if r["action"] == "swap"
                and (r.get("timing") or {}).get("status") == "off"]
    if timing is not None:
        timing["verified"] = timing_verified(timing, rows, timeline_verified)
    return {
        "source": source, "final": final,
        "calibration": calibration,
        "duration_drift_sec": round(drift, 3),
        "alignment": alignment,
        "timeline_verified": timeline_verified,
        "timing": timing,
        "sync_off_segments": sync_off,
        "segments": rows,
        "dirty_segments": dirty,
        "unconfirmed_false_positives": unconfirmed,
        "unconfirmed_fills": unconfirmed_fills,
        "keep_with_captions": [r["index"] for r in rows if r["verdict"] == "keep_captions"],
        "comparison_sheet": compare,
        "warnings": warnings,
        # On the source timeline: every swap segment measured in sync (or too
        # short to measure) and the picture as long as the source -- a segment
        # off by more than +-1 frame is a lip-sync defect under the original
        # soundtrack, not a warning (2026-11-02).
        "clean": not dirty and not unconfirmed and not unconfirmed_fills
        and calibrated and timing is not None and timing["verified"]
        and all(
            r["verdict"] in ("clean", "false_positive", "keep") for r in rows),
    }


def _public_span(sp: dict) -> dict:
    return {k: sp[k] for k in ("start_sec", "end_sec", "frames", "max_score", "box")}


# ---------------------------------------------------------------------------
# 3. Rerun proposal / best take / hand-off
# ---------------------------------------------------------------------------

def is_coordinate_prompt(prompt: Optional[str]) -> bool:
    return bool(prompt) and COORDINATE_PROMPT_MARKER in prompt


def coordinate_box(spans: Sequence[dict]) -> list[float]:
    """Union of the leftover boxes, padded, in frame fractions."""
    a = np.array([sp["box"] for sp in spans])
    return [round(max(0.0, float(a[:, 0].min()) - 0.05), 2),
            round(max(0.0, float(a[:, 1].min()) - 0.04), 2),
            round(min(1.0, float(a[:, 2].max()) + 0.05), 2),
            round(min(1.0, float(a[:, 3].max()) + 0.04), 2)]


def coordinate_prompt(box: Sequence[float]) -> str:
    x0, y0, x1, y1 = box
    return (
        "Remove the burned-in caption: white text with a thin black outline, in the "
        f"box x {x0:.2f}–{x1:.2f}, y {y0:.2f}–{y1:.2f} of the frame. "
        f"{COORDINATE_PROMPT_MARKER}: reconstruct what is behind the text. Do not "
        "touch anything outside the box, do not change the phone screen, the "
        "character, the face, the lips, the background, lighting, framing or "
        "motion. Do not add any text. Do not add any audio.")


def rerun_proposal(run_id: str, qa: dict, segments: Sequence[Seg], *,
                   takes: dict, price_per_segment: float,
                   unfinished: Sequence[str] = (), admission_estimate=None) -> dict:
    """OPERATOR OPTION ONLY: rerun the dirty segments on Omni with the other
    prompt variant.

    The flow never proposes this for a dirty segment (rule 1 and the stop
    rule: one Omni pass, then a VOID fill -- see :func:`fill_proposal`); on
    2845eef3 the coordinate-prompt rerun made both segments worse. It is kept
    for an operator who explicitly asks for an Omni retake, and the CLI puts it
    under ``operator_options`` with ``executable: false`` so an autopilot never
    sends it on its own.

    *takes* maps run_segment_id -> number of preserved takes (AICP-342 rows:
    completed outputs a rerun or restore replaced; failed generations are not
    in it). ``outputs`` = takes + the current output, a LOWER bound on the
    attempts; at ``STOP_AFTER_ATTEMPTS`` the segment stops for the operator
    instead of another paid take. *unfinished* lists the run_segment_ids of
    active swap segments without a finished result: ``submit_run`` resubmits
    those too, so the ones not already listed are billed with this rerun. *admission_estimate(segments) ->
    credits*, when given, is ``_segment_rerun_need`` -- what the spend cap
    checks, shown next to the expected bill.

    Prompt alternation: a segment on the coordinate prompt goes back to the
    run prompt (``"prompt": ""`` clears the override); any other segment gets
    a coordinate prompt boxed on its leftovers, REPLACING a custom override
    if it has one (``replaces_override``). Both rerun and restore re-stitch
    and replace the content behind the run's Drive link; when the current
    result is an audio patch the endpoint 409s until the operator decides
    ``discard_audio_patch``.
    """
    by_index = {s.index: s for s in segments}
    reruns, stops = [], []
    for row in qa["segments"]:
        if row["verdict"] != "leftover":
            continue
        seg = by_index[row["index"]]
        outputs = int(takes.get(seg.run_segment_id, 0)) + 1
        if outputs >= STOP_AFTER_ATTEMPTS:
            stops.append({
                "index": seg.index, "run_segment_id": seg.run_segment_id,
                "outputs": outputs,
                "options": ["VOID fill of the segment (see the fill proposal)",
                            "delogo the leftover box over its time window (free)",
                            "accept the best take"],
                "leftovers": row["leftovers"]})
            continue
        if is_coordinate_prompt(seg.prompt_override):
            prompt, variant = "", "default"
        else:
            prompt = coordinate_prompt(coordinate_box(row["leftovers"]))
            variant = "coordinate"
        reruns.append({"run_segment_id": seg.run_segment_id, "index": seg.index,
                       "outputs_so_far": outputs, "variant": variant, "prompt": prompt,
                       "current_prompt_override": seg.prompt_override,
                       "replaces_override": bool(seg.prompt_override)})
    option = {"operator_option": True, "executable": False,
              "why_not_proposed": ("flow doc rule 1 / stop rule: a dirty segment gets a "
                                   "VOID fill, not another Omni take; send only on the "
                                   "operator's explicit request")}
    if not reruns:
        return {"request": None, "stop": stops, "approval_required": bool(stops), **option}
    body = {"segments": [{"run_segment_id": r["run_segment_id"], "prompt": r["prompt"]}
                         for r in reruns],
            "discard_audio_patch": False}
    listed = {r["run_segment_id"] for r in reruns}
    other_unfinished = len(set(unfinished) - listed)
    billed = len(reruns) + other_unfinished
    cost = {"model": REMOVAL_MODEL, "credits_per_segment": price_per_segment,
            "segments_billed": billed, "other_unfinished": other_unfinished,
            "credits": round(price_per_segment * billed, 2)}
    if admission_estimate is not None:
        cost["budget_check_credits"] = round(
            float(admission_estimate([r["run_segment_id"] for r in reruns])), 2)
    return {
        "request": {"method": "POST", "path": f"/api/v2/runs/{run_id}/segments/rerun",
                    "body": body},
        "segments": reruns,
        "stop": stops,
        "cost": cost,
        "note": "re-stitches the run and replaces the file behind its Drive link; "
                "previous outputs are kept as takes (free restore)",
        "approval_required": True,
        **option,
    }


FILL_LIVE = ("preparing", "ready", "admitted", "compositing", "composited")


def void_reserve_usd(duration_sec: float = 8.0) -> float:
    """What the spend cap reserves for one VOID request (and what the start's
    ``approved_usd`` must cover): the registry's worst case, flat for VOID."""
    from app import fal_models
    from app.pipeline_segment_fill import VOID_RESOLUTION

    return float(fal_models.endpoint(fal_models.VOID_ENDPOINT).worst_case_usd(
        VOID_RESOLUTION, duration_sec))


def _fill_paths(run_id: str, rsid: Optional[str]) -> dict:
    return {
        "preview": f"/api/v2/runs/{run_id}/segments/{rsid}/fill/preview",
        "start": f"/api/v2/runs/{run_id}/segments/fill/{{fill_id}}/start",
        "cancel": f"/api/v2/runs/{run_id}/segments/fill/{{fill_id}}/cancel",
        "list": f"/api/v2/runs/{run_id}/segments/fill",
    }


def _paid_cost(reserve: float) -> dict:
    return {"currency": "USD", "reserve_usd": round(reserve, 4),
            "expected_usd": VOID_EXPECTED_USD, "kie_credits": 0}


def _preview_steps(run_id: str, seg: Seg, reserve: float, body: dict) -> tuple[dict, dict]:
    paths = _fill_paths(run_id, seg.run_segment_id)
    request = {"method": "POST", "path": paths["preview"], "body": body, "paid": False}
    then = {
        "step": "start", "executable": False, "paid": True,
        "method": "POST", "path": paths["start"],
        "body": {"approved_usd": round(reserve, 4), "discard_audio_patch": False},
        "fill_id_from": "the preview response's id",
        "requires": [
            f"poll GET {paths['list']} until this job is 'ready' (or 'failed')",
            "inspect the preview: mask_box covers expected_mask_box, preview sheet "
            "shows the caption masked and nothing else (face, phone screen)",
            "estimated_usd <= the approved reservation (else ask again)",
            "run qa again: it proposes this start for the ready job "
            "(state propose_start) only while the preview still matches the run",
        ],
    }
    return request, then


def _widen_box(fill_box, leftovers, seg: Seg, centred: bool = False) -> Optional[list[float]]:
    """Union of the old mask box and the leftovers that lie inside the segment
    (a leftover in the boundary slack needs a boundary decision, not a wider
    box), padded, mirrored about the frame's centre when the source captions
    are *centred*, validated like a request box."""
    from app.pipeline_segment_fill import FillError, mirror_x, validate_box

    boxes = [sp["box"] for sp in leftovers
             if sp.get("box") and seg.start - 1e-6 <= sp["start_sec"] < seg.end - 1e-6]
    if not boxes:
        return None
    if fill_box:
        boxes.append(fill_box)
    a = np.array(boxes, dtype=float)
    x0, y0, x1, y1 = mirror_x([float(a[:, 0].min()) - WIDEN_PAD_X,
                               float(a[:, 1].min()) - WIDEN_PAD_Y,
                               float(a[:, 2].max()) + WIDEN_PAD_X,
                               float(a[:, 3].max()) + WIDEN_PAD_Y], centred)
    try:
        return validate_box([max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)])
    except FillError:
        return None


def _operator_options(run_id: str, seg: Seg, row: dict, reserve: float) -> list[dict]:
    """The doc's stop rule for a VOID composite that is still dirty, or whose
    eye check failed. None of these is executable without the operator."""
    fill = seg.fill or {}
    leftovers = row.get("leftovers") or []
    opts = []
    wide = _widen_box(fill.get("mask_box"), leftovers, seg,
                      centred=bool(row.get("captions_centred")))
    if wide is not None:
        request, then = _preview_steps(run_id, seg, reserve, {"mask_box": wide})
        opts.append({"option": "widen_mask", "executable": False, "paid": True,
                     "second_paid_attempt": True, "mask_box": wide,
                     "request": request, "then": then, "cost": _paid_cost(reserve),
                     "note": "a wider box does not make the glyph mask catch a ghost; "
                             "only for edge letters outside the old box"})
    elif leftovers:
        opts.append({"option": "widen_mask", "available": False,
                     "note": "the leftovers lie outside the segment (boundary slack): "
                             "move the boundary instead of widening the box"})
    opts.append({"option": "ltx_inpaint", "available": False,
                 "endpoint": "fal-ai/ltx-2.3-quality/inpaint", "usd_estimate": [0.41, 0.54],
                 "note": "same mask, judged slightly cleaner by eye (AICP-354); not a "
                         "platform endpoint yet: by hand"})
    if leftovers:
        a = np.array([sp["box"] for sp in leftovers], dtype=float)
        opts.append({"option": "delogo", "executable": False, "paid": False, "manual": True,
                     "box": [round(max(0.0, float(a[:, 0].min()) - 0.01), 3),
                             round(max(0.0, float(a[:, 1].min()) - 0.01), 3),
                             round(min(1.0, float(a[:, 2].max()) + 0.01), 3),
                             round(min(1.0, float(a[:, 3].max()) + 0.01), 3)],
                     "windows": [[sp["start_sec"], sp["end_sec"]] for sp in leftovers],
                     "note": "ffmpeg delogo over that box and time window, for a small "
                             "leftover on a plain background (flow step 6); no endpoint"})
    if fill.get("replaced_take_id"):
        opts.append({"option": "restore_previous_take", "executable": False, "paid": False,
                     "request": {"method": "POST",
                                 "path": f"/api/v2/runs/{run_id}/segments/restore",
                                 "body": {"takes": [fill["replaced_take_id"]],
                                          "discard_audio_patch": False}},
                     "note": "puts back the output the fill replaced (its captions too)"})
    opts.append({"option": "accept", "executable": False, "paid": False})
    opts.append({"option": "omni_rerun", "executable": False,
                 "see": "operator_options.omni_rerun"})
    return opts


def _repeat_reason(paid_before: list) -> str:
    return ("a VOID fill of this segment was already admitted (%s): a repeat is a "
            "second paid attempt, the operator decides" % ", ".join(
                "%s %s/%s" % (j["job_id"], j["status"], j.get("provider_status"))
                for j in paid_before))


def fill_proposal(run_id: str, qa: dict, segments: Sequence[Seg], *,
                  context: Optional[dict] = None, actionable: bool = True,
                  not_actionable_reason: Optional[str] = None,
                  reserve_usd: Optional[float] = None,
                  rebuild: Sequence[int] = ()) -> dict:
    """The flow's proposal for the dirty segments: a VOID caption fill each.

    *rebuild*: indices of swap segments to rebuild from the source although
    the detector found no leftover in them (the autopilot passes segments whose
    picture is out of sync with the source soundtrack after a conformed
    stitch): the fill's composite is the source clip with the captions
    inpainted, so it is exactly on the timeline. A segment whose source shows
    no caption has nothing to mask: ``operator``. Items carry ``why``:
    ``leftover`` or ``sync``.

    A QA with per-segment timing (``qa["timing"]``) attributes each leftover
    span to its segment (:func:`attribution`): a segment whose every span sits
    within the picture shift of a cut is ``ambiguous`` (not proposed; the other
    segments go first, their rebuild puts them back on the timeline). A QA
    without it keeps the run-wide rule: an unverified timeline proposes
    nothing.

    One item per dirty (``leftover``) swap segment, in segment order, each in
    one ``state``:

    * ``propose_preview`` -- ``request`` is the free
      ``POST .../segments/{rsid}/fill/preview`` with an EMPTY body: prepare
      then measures the inclusive box on every frame of the cut clip (QA's
      10 fps box, ``expected_mask_box``, would be a subset of it). ``then`` is
      the paid start as a TEMPLATE (``executable: false``): its fill id exists
      only after the preview, and its approval is of that preview.
    * ``propose_start`` -- a ``ready`` preview of this segment that still
      matches the run (frozen base), default prompt, measured box, estimate
      within the reservation: ``request`` is its start, ``approved_usd`` = its
      own ``estimated_usd``. Paid: needs the operator's approval of the
      preview sheet + amount.
    * ``stale_preview`` -- that ready preview no longer matches the run:
      ``request`` cancels it (free); then run qa again.
    * ``in_progress`` -- a fill of this segment is preparing / admitted /
      compositing: wait, then run qa again.
    * ``blocked`` -- a known gate would refuse (``blockers``).
    * ``operator`` -- not automatic (``reason``): ambiguous previews, a custom
      ready preview, a repeat of an already-paid fill, a result that is not a
      known Omni output.
    * ``stop`` -- the result already IS a VOID composite and still shows
      caption words (doc stop rule): ``options`` for the operator.

    ``next`` is at most ONE request: fills go one at a time (one active fill
    per run, and every preview is bound to the final it was made on, so
    another fill's re-delivery makes it stale). After a fill is delivered, run
    qa again instead of continuing an older proposal. Nothing is ``next``
    when not *actionable* (inputs not tied to the live run) or while any fill
    of the run is in flight.

    *context* comes from the database (:func:`_fill_context`): per-segment
    blockers + fill history and run-wide blockers. Without it (files outside
    the database) the requests are shown with ``prechecked: false``.

    Cost: the reservation (registry worst case, what ``approved_usd`` must
    cover; flat $0.10 for VOID) and the expected charge ($0.05 observed) are
    two numbers for the same call, never a sum. No kie.ai credits; the
    re-stitch is free (start refuses when it would generate another segment).
    """
    reserve = void_reserve_usd() if reserve_usd is None else float(reserve_usd)
    rebuild = {int(i) for i in rebuild}
    if actionable and not qa.get("timing") and qa.get("timeline_verified", True) is False:
        # Hits are mapped to segments by time: off the source timeline, a
        # leftover may belong to the neighbour, and a fill there is wasted.
        actionable = False
        not_actionable_reason = ("the final is not verified on the source timeline "
                                 "(alignment): leftovers may sit in another segment")
    by_index = {s.index: s for s in segments}
    run_blockers = list((context or {}).get("run_blockers") or [])
    seg_ctx = (context or {}).get("segments") or {}
    items, on_reject = [], []
    for row in qa["segments"]:
        seg = by_index.get(row["index"])
        if seg is None or seg.action != "swap":
            continue
        sync = seg.index in rebuild and row["verdict"] in ("clean", "clean_uncalibrated",
                                                            "leftover")
        if row["verdict"] != "leftover" and not sync:
            if row.get("fill_needs_visual_check"):
                on_reject.append({"index": seg.index, "run_segment_id": seg.run_segment_id,
                                  "fill_job_id": (seg.fill or {}).get("job_id"),
                                  "options": _operator_options(run_id, seg, row, reserve)})
            continue
        item = {"index": seg.index, "run_segment_id": seg.run_segment_id,
                "start_sec": row["start_sec"], "end_sec": row["end_sec"],
                "leftovers": row["leftovers"], "expected_mask_box": row.get("mask_box"),
                "result_kind": seg.result_kind, "request": None, "blockers": [],
                "prechecked": context is not None,
                "why": "leftover" if row["verdict"] == "leftover" else "sync",
                "timing": row.get("timing")}
        ctx = seg_ctx.get(seg.run_segment_id) if context is not None else None
        jobs = (ctx or {}).get("jobs") or []
        item["fill_history"] = jobs
        if row["verdict"] == "leftover" and qa.get("timing"):
            att = attribution(qa, row)
            item["ambiguous"] = att["ambiguous"]
            if not att["own"] and not sync:
                item.update(state="ambiguous", reason=(
                    "every leftover sits at a cut where the picture is shifted, so it may be "
                    "segment %s's clip" % " or ".join(sorted(
                        {str(a["neighbour"]) for a in att["ambiguous"]}))))
                items.append(item)
                continue
        if sync and item["why"] == "sync" and not row.get("source_hits"):
            item.update(state="operator",
                        reason="the picture is out of sync, but the source shows no caption "
                               "in this segment: a VOID rebuild has nothing to mask")
            items.append(item)
            continue
        if seg.result_kind == "void_fill" and item["why"] == "sync":
            item.update(state="operator", fill_job_id=(seg.fill or {}).get("job_id"),
                        reason="the result is already a VOID composite built on the source "
                               "and still measures out of sync: decide by hand")
            items.append(item)
            continue
        if seg.result_kind == "void_fill":
            item.update(state="stop", fill_job_id=(seg.fill or {}).get("job_id"),
                        reason="the result is already a VOID composite and the detector "
                               "still sees caption words: no automatic second fill",
                        options=_operator_options(run_id, seg, row, reserve))
            items.append(item)
            continue
        if seg.result_kind == "unknown":
            item.update(state="operator",
                        reason="the current result is not a known Omni output (a "
                               "hand-placed clip, or a composite whose delivery is not "
                               "confirmed): decide by hand")
            items.append(item)
            continue
        if not seg.run_segment_id:
            item.update(state="blocked", blockers=["no run segment for this segment"])
            items.append(item)
            continue
        live = [j for j in jobs if j.get("status") in FILL_LIVE]
        # Any admitted fill of this segment that is not live any more (failed
        # after admission, or applied then replaced): a new start would be a
        # second paid attempt, whichever preview it starts.
        paid_before = [j for j in jobs if j.get("provider_task_id")
                       and j.get("status") not in FILL_LIVE]
        blockers = run_blockers + list((ctx or {}).get("blockers") or [])
        paths = _fill_paths(run_id, seg.run_segment_id)
        if len(live) > 1:
            item.update(state="operator",
                        reason="%d unfinished fills of this segment (%s): ambiguous, "
                               "cancel the extra previews first" % (
                                   len(live), ", ".join(j["job_id"] for j in live)))
        elif live and live[0]["status"] != "ready":
            item.update(state="in_progress", fill_job_id=live[0]["job_id"],
                        reason="fill %s is %s: wait, then run qa again" % (
                            live[0]["job_id"], live[0]["status"]))
        elif live:
            job = live[0]
            item["fill_job_id"] = job["job_id"]
            est = job.get("estimated_usd")
            if job.get("stale"):
                item.update(state="stale_preview",
                            reason=f"preview {job['job_id']} is stale ({job['stale']} "
                                   "changed): cancel it, then run qa again",
                            request={"method": "POST",
                                     "path": paths["cancel"].replace("{fill_id}", job["job_id"]),
                                     "body": {}, "paid": False})
            elif paid_before:
                item.update(state="operator",
                            reason=_repeat_reason(paid_before))
            elif not job.get("default_prompt") or job.get("box_source") != "measured":
                item.update(state="operator",
                            reason="the ready preview has a custom prompt or requested box: "
                                   "the operator approves it on its own preview sheet")
            elif est is None or est > reserve + 1e-9:
                item.update(state="operator",
                            reason=f"the preview reserves ${est}, above this proposal's "
                                   f"${reserve}: ask again")
            elif blockers:
                item.update(state="blocked", blockers=blockers)
            else:
                item.update(
                    state="propose_start", mask_box=job.get("mask_box"),
                    frames_masked=job.get("frames_masked"),
                    preview_url=job.get("preview_url"), cost=_paid_cost(est),
                    needs_visual_check=True,
                    request={"method": "POST",
                             "path": paths["start"].replace("{fill_id}", job["job_id"]),
                             "body": {"approved_usd": round(float(est), 4),
                                      "discard_audio_patch": False},
                             "paid": True})
        else:
            last = jobs[0] if jobs else None
            if paid_before:
                item.update(state="operator", reason=_repeat_reason(paid_before))
            elif last and last.get("status") == "failed" \
                    and last.get("error_code") in FILL_DETERMINISTIC_ERRORS:
                item.update(state="blocked", blockers=run_blockers + [
                    f"the last preview {last['job_id']} failed: {last['error_code']} "
                    f"({last.get('error_message')})"])
            elif blockers:
                item.update(state="blocked", blockers=blockers)
            else:
                request, then = _preview_steps(run_id, seg, reserve, {})
                item.update(state="propose_preview", request=request, then=then,
                            cost=_paid_cost(reserve))
        items.append(item)

    in_flight = (context or {}).get("active_fill") is not None or any(
        i["state"] == "in_progress" for i in items)
    nxt = None
    if actionable and not in_flight:
        first = (next((i for i in items if i["state"] in ("propose_start", "stale_preview")),
                      None)
                 or next((i for i in items if i["state"] == "propose_preview"), None))
        if first is not None:
            nxt = {"index": first["index"], "run_segment_id": first["run_segment_id"],
                   "state": first["state"], "request": first["request"]}
    paid = [i for i in items if i["state"] in ("propose_preview", "propose_start")]
    return {
        "model": FILL_MODEL,
        "endpoint": "fal-ai/void-video-inpainting",
        "actionable": bool(actionable),
        "not_actionable_reason": None if actionable else not_actionable_reason,
        "segments": items,
        "order": [i["index"] for i in paid],
        "sequential": True,
        "next": nxt,
        "after_next": ("one fill at a time: after this request, poll "
                       f"GET {_fill_paths(run_id, None)['list']} and run qa again; "
                       "after a fill is delivered, run qa again rather than "
                       "continuing this proposal"),
        "run_blockers": run_blockers,
        "active_fill": (context or {}).get("active_fill"),
        "stop": [i for i in items if i["state"] == "stop"],
        "on_visual_reject": on_reject,
        "cost": {"currency": "USD", "segments": len(paid),
                 "reserve_usd_per_segment": round(reserve, 4),
                 "expected_usd_per_segment": VOID_EXPECTED_USD,
                 "reserve_usd_total": round(reserve * len(paid), 4),
                 "expected_usd_total": round(VOID_EXPECTED_USD * len(paid), 4),
                 "kie_credits": 0,
                 "note": "reserve = what the spend cap holds and approved_usd must cover; "
                         "expected = the observed Fal charge; the same call, not a sum"},
        "approval_required": bool(items),
    }


def choose_best_takes(run_id: str, candidates: dict) -> dict:
    """Pick the output to keep per segment after re-checking them.

    *candidates* maps run_segment_id -> ``[{"take_id": id or None (the current
    output), "hits": n, "created_at": iso}]``, each one scanned on the same
    timeline. Fewest hits wins, the newest on a tie -- hits only: a take can
    win by carrying a caption style the detector misses, so the winner still
    goes through the crop / comparison sheets. Restoring is free; only
    segments whose best output is NOT the current one go into the body. A
    restore also brings back that take's prompt/reference overrides.
    """
    keep, restore = {}, []
    for rsid, items in candidates.items():
        if not items:
            continue
        # Newest first, so min() (which keeps the first of equals) picks the
        # newest of the outputs tied on hits.
        newest_first = sorted(items, key=lambda c: c.get("created_at") or "", reverse=True)
        best = min(newest_first, key=lambda c: c["hits"])
        keep[rsid] = best
        if best.get("take_id"):
            restore.append(best["take_id"])
    request = None
    if restore:
        request = {"method": "POST", "path": f"/api/v2/runs/{run_id}/segments/restore",
                   "body": {"takes": restore, "discard_audio_patch": False}}
    return {"best": keep, "request": request}


def handoff_proposal(run_id: str, qa: dict, segments: Sequence[Seg],
                     review: Optional[dict] = None) -> dict:
    """``POST /runs/{rid}/to-project`` -> localisation, whole scope, once the
    final is clean and on the source timeline.

    The request cannot carry segments: the new project's analysis makes its
    own. The keep ranges (end card / no captions) are a separate, required
    edit of the new project's segments after its analysis, before its run.
    """
    if not qa.get("clean"):
        if qa.get("dirty_segments"):
            reason = "the final is not clean yet"
        elif qa.get("unconfirmed_fills"):
            reason = ("VOID composites need the eye check first (--confirm-fill): %s"
                      % ", ".join(str(f.get("fill_job_id")) for f in qa["unconfirmed_fills"]))
        elif qa.get("unconfirmed_false_positives"):
            reason = "false positives need confirming on their crop sheets (--confirm-fp)"
        else:
            reason = "QA is not conclusive (calibration or timeline)"
        return {"request": None, "reason": reason,
                "dirty_segments": qa.get("dirty_segments", [])}
    ranges = {(round(s.start, 3), round(s.end, 3)) for s in segments if s.action == "keep"}
    ranges |= {(r["start_sec"], r["end_sec"]) for r in qa["segments"]
               if r["verdict"] == "false_positive"}
    if review:
        ranges |= {(r["start_sec"], r["end_sec"]) for r in review["segments"]
                   if r["proposed_action"] == "keep"}
    return {
        "request": {"method": "POST", "path": f"/api/v2/runs/{run_id}/to-project",
                    "form": {"project_type": "localisation",
                             "localisation_scope": "whole"}},
        "keep_ranges": [{"start_sec": a, "end_sec": b} for a, b in sorted(ranges)],
        "then": "after the new project's analysis, mark these ranges keep "
                "(PATCH its segments) before launching the localisation run",
        "warnings": qa.get("warnings", []),
        "cost": 0,
        "approval_required": True,
    }


# ---------------------------------------------------------------------------
# Database glue (read-only) + CLI
# ---------------------------------------------------------------------------

def price_per_segment(session, model: str, resolution: Optional[str]) -> float:
    """Median billed credits of recent completed segments of *model* at
    *resolution* (Omni bills a flat 168 per clip, which the per-second spend
    estimate overstates); the spend estimate for a max-length clip without
    history."""
    from sqlalchemy import select

    from app import spend
    from app.models import Run, RunSegment

    stmt = (select(RunSegment.cost_credits).join(Run, Run.id == RunSegment.run_id)
            .where(Run.model == model, RunSegment.cost_credits.is_not(None),
                   RunSegment.cost_credits > 0)
            .order_by(RunSegment.updated_at.desc()).limit(200))
    if resolution:
        stmt = stmt.where(Run.resolution == resolution)
    costs = [float(c) for c in session.execute(stmt).scalars()]
    if costs:
        return float(statistics.median(costs))
    from app import ai_models
    return float(spend.video_credits(model, resolution, ai_models.max_clip_sec(model)))


def _face_samples(source: str, fps: float = 1.0):
    from app import face

    return [(fr.t_sec, len(fr.faces)) for fr in face.detect_timeline(source, sample_fps=fps)]


def _load_json_arg(value: str):
    text = value
    if not value.lstrip().startswith(("[", "{")):
        with open(value, encoding="utf-8") as fh:
            text = fh.read()
    return json.loads(text)


def _project_inputs(session, project_id: str) -> dict:
    from app.models import VideoProject

    project = session.get(VideoProject, project_id)
    if project is None:
        raise QAToolError(f"project {project_id!r} not found")
    segs = [{"index": sd.index, "start_sec": sd.start_sec, "end_sec": sd.end_sec,
             "action": sd.action, "pre_roll_sec": sd.pre_roll_sec,
             "post_roll_sec": sd.post_roll_sec} for sd in project.segments]
    return {"source": project.source_local_path, "config": project.caption_detector,
            "segments": segs}


def _run_inputs(session, run_id: str) -> dict:
    from sqlalchemy import func, select

    from app import spend
    from app.models import Run, RunSegment, RunSegmentTake
    from app.pipeline_v2 import _active_swap_segment_ids, run_segment_defs

    run = session.get(Run, run_id)
    if run is None:
        raise QAToolError(f"run {run_id!r} not found")
    project = run.project
    # First: everything below is checked against this at the end of the QA
    # (the CLI also reads it all in one REPEATABLE READ snapshot).
    evidence = run_evidence(session, run, project)
    defs = run_segment_defs(run, project)
    rs_by_def = {rs.segment_def_id: rs for rs in session.execute(
        select(RunSegment).where(RunSegment.run_id == run.id)).scalars()}
    active = _active_swap_segment_ids(run.stitch_plan, defs)
    segs, unfinished, rerun_extra = [], [], {}
    for sd in defs:
        rs = rs_by_def.get(sd.id)
        segs.append({"index": sd.index, "start_sec": sd.start_sec, "end_sec": sd.end_sec,
                     "action": sd.action if sd.id in active else "keep",
                     "pre_roll_sec": sd.pre_roll_sec, "post_roll_sec": sd.post_roll_sec,
                     "run_segment_id": rs.id if rs else None,
                     "prompt_override": rs.prompt_override if rs else None,
                     "status": getattr(rs.status, "value", rs.status) if rs else None})
        if sd.id not in active:
            continue
        # The same rule as the spend estimate / _segment_rerun_need: a segment
        # submit would send to kie.ai again is unfinished (billed with any
        # rerun); a paid one adds its own clip once a rerun resets it.
        if rs is None or not spend.segment_is_paid(rs):
            unfinished.append(rs.id if rs else sd.id)
        else:
            rerun_extra[rs.id] = spend.video_credits(run.model, run.resolution,
                                                     spend._clip_sec(sd))
    fills = _fill_context(session, run, project, defs, rs_by_def, active)
    for seg in segs:
        info = fills["result"].get(seg["run_segment_id"]) if seg["run_segment_id"] else None
        if info and seg["action"] == "swap":
            seg["result_kind"] = info["result_kind"]
            if info.get("fill"):
                seg["fill"] = info["fill"]
    base = spend.estimate_run_credits(session, run, include_preprocess=False)
    takes = dict(session.execute(
        select(RunSegmentTake.run_segment_id, func.count())
        .where(RunSegmentTake.run_id == run.id)
        .group_by(RunSegmentTake.run_segment_id)).all())
    return {"run": run, "source": project.source_local_path,
            "final": run.result_local_path, "config": project.caption_detector,
            "segments": segs, "takes": takes, "unfinished": unfinished,
            "custom_plan": bool(run.stitch_plan),
            "fill_context": fills["context"],
            "evidence": evidence,
            # _segment_rerun_need for a list of run_segment_ids.
            "admission": lambda ids: base + sum(rerun_extra.get(i, 0.0) for i in ids)}


def _own_file(job, path: Optional[str]) -> Optional[str]:
    from app import pipeline_segment_fill as psf

    if not path or not os.path.isfile(path):
        return None
    root = os.path.realpath(psf._job_dir(job))
    return path if os.path.realpath(path).startswith(root + os.sep) else None


def _job_summary(session, job, run, rs, project) -> dict:
    from app import pipeline_segment_fill as psf
    from app.provider_models import ProviderTask

    task = session.get(ProviderTask, job.provider_task_id) if job.provider_task_id else None
    media = job.media or {}
    base = job.base or {}
    out = {"job_id": job.id, "status": job.status,
           # AICP-358 follow-up: what the fill rebuilds (a row without a basis
           # predates it: a model result) and, for a source fill, the blocked
           # provider task it answers and whether its clip needs no VOID at all.
           "basis": base.get("basis") or psf.BASIS_MODEL,
           "block_task": (base.get("segment") or {}).get("seedance_task_id"),
           "source_as_is": bool(media.get("source_as_is")),
           "source_as_is_reason": media.get("source_as_is_reason"),
           # Files the autopilot may attach as run-produced artifacts: only
           # when they lie under this fill's own directory (design R1-13).
           "preview_path": _own_file(job, job.preview_path),
           "clip_path": _own_file(job, os.path.join(job.prep_dir, "clip.mp4")
                                  if job.prep_dir else None),
           "reservation_held": (bool(task.reservation_held) if task is not None else None),
           "created_at": job.created_at.isoformat() if job.created_at else None,
           "box_source": job.box_source, "mask_box": job.mask_box,
           "default_prompt": not job.prompt or job.prompt == psf.DEFAULT_PROMPT,
           "estimated_usd": job.estimated_usd, "approved_usd": job.approved_usd,
           "frames_masked": media.get("frames_masked"),
           "preview_url": (f"/api/v2/runs/{job.run_id}/segments/fill/{job.id}/preview.jpg"
                           if job.preview_path else None),
           "provider_task_id": job.provider_task_id,
           "provider_status": task.status if task is not None else None,
           "error_code": job.error_code, "error_message": job.error_message,
           "stale": None}
    out.update({"batch_id": job.batch_id, "retry_of": job.retry_of,
                "mask_grow_px": job.mask_grow_px,
                "segment_bound": psf.base_bind(job) == psf.BIND_SEGMENT})
    if job.status == "ready" and rs is not None:
        try:
            live = psf.live_base(session, run, rs, project, job)
            out["stale"] = psf.base_mismatch(job.base or {}, live)
        except psf.FillError as exc:
            out["stale"] = exc.code
    return out


def _fill_context(session, run, project, defs, rs_by_def, active, *,
                  src_fps: Optional[float] = None, probe: bool = True,
                  classify: bool = True) -> dict:
    """Read-only facts for :func:`fill_proposal`: the real API gates (each
    raises an HTTPException we only collect), the fill history per segment,
    and what each segment's current result is. Authoritative checks stay with
    the endpoints (Fal worker, daily budget, concurrent changes)."""
    from fastapi import HTTPException
    from sqlalchemy import desc, select

    from app import api_segment_fill as asf
    from app import pipeline_segment_fill as psf
    from app.api_v2 import (_check_segment_rerun_conflicts, _require_audio_patch_discard_ack,
                            _require_segment_reconstruction_allowed)
    from app.config import settings
    from app.segment_fill_models import RunSegmentFillJob

    def gate(out: list, fn, *args) -> None:
        try:
            fn(*args)
        except HTTPException as exc:
            out.append(str(exc.detail))

    run_blockers: list[str] = []
    gate(run_blockers, _require_segment_reconstruction_allowed, run.id, session)
    gate(run_blockers, _check_segment_rerun_conflicts, run.id, session)
    gate(run_blockers, _require_audio_patch_discard_ack, run, session, False)
    if not (settings.FAL_ENABLED and settings.FAL_API_KEY):
        run_blockers.append("Fal is disabled (FAL_ENABLED / FAL_API_KEY): the start would "
                            "be refused")
    act = psf.active_fill(session, run.id)
    active_fill = None
    if act is not None:
        active_fill = {"job_id": act.id, "run_segment_id": act.run_segment_id,
                       "status": act.status, "batch_id": act.batch_id}
        run_blockers.append(
            (f"fill batch {act.batch_id} is active (member {act.id} {act.status}): one batch "
             "at a time" if act.batch_id else
             f"fill {act.id} is {act.status} on run segment {act.run_segment_id}: one fill "
             "at a time"))
    jobs = session.execute(select(RunSegmentFillJob).where(RunSegmentFillJob.run_id == run.id)
                           .order_by(desc(RunSegmentFillJob.created_at))).scalars().all()
    if src_fps is None and probe:
        try:
            src_fps = float(qa_tools._probe(project.source_local_path).fps)
        except Exception:  # an unreadable source: prepare refuses it with its own code
            pass
    from app import content_block

    segments, result = {}, {}
    for sd in defs:
        rs = rs_by_def.get(sd.id)
        if rs is None or sd.id not in active:
            continue
        # A segment the provider refused with a content filter is filled from
        # the SOURCE (no model result to rebuild); every other one as AICP-358.
        block = content_block.segment_block(rs)
        basis = psf.BASIS_SOURCE if block else psf.BASIS_MODEL
        blockers: list[str] = []
        gate(blockers, asf._require_fillable, run, project, rs, basis)
        if (sd.pre_roll_sec or 0) or (sd.post_roll_sec or 0):
            blockers.append("pre/post roll segments are not supported by the fill")
        # (A source fill of a short clip without captions needs no VOID call:
        # its prepare decides, segment_too_short only when there are captions.)
        if basis == psf.BASIS_MODEL and src_fps \
                and (sd.end_sec - sd.start_sec) * src_fps < psf.VOID_MIN_FRAMES - 0.5:
            blockers.append("about %d frames; VOID needs at least %d" % (
                round((sd.end_sec - sd.start_sec) * src_fps), psf.VOID_MIN_FRAMES))
        missing = (psf.paid_restitch_segments(session, run, exclude_rs_id=rs.id)
                   if basis == psf.BASIS_MODEL else [])
        if missing:
            blockers.append(f"segment(s) {missing} have no finished result: the re-stitch "
                            "after the fill would generate them (paid)")
        own = [j for j in jobs if j.run_segment_id == rs.id]
        segments[rs.id] = {"blockers": blockers, "basis": basis, "block": block,
                           "jobs": [_job_summary(session, j, run, rs, project) for j in own]}
        if not classify:
            continue
        current = next((j for j in own if j.status == "applied"
                        and psf._is_composite(j, rs.local_result_path)), None)
        if current is not None:
            media = current.media or {}
            result[rs.id] = {"result_kind": "void_fill", "fill": {
                "job_id": current.id, "mask_box": current.mask_box,
                "retry_of": current.retry_of, "batch_id": current.batch_id,
                # The provider's content refusal a source fill answered (None
                # for a VOID-from-the-source run's fill): an Omni rerun is an
                # option only without one.
                "block": ((current.base or {}).get("segment") or {}).get("block"),
                "frames_masked": media.get("frames_masked"),
                "basis": (current.base or {}).get("basis") or psf.BASIS_MODEL,
                "source_as_is": bool(media.get("source_as_is")),
                "applied_at": current.applied_at.isoformat() if current.applied_at else None,
                "replaced_take_id": current.replaced_take_id,
                "delivery": psf.delivery_state(current, run, rs, db=session)}}
        elif rs.seedance_task_id and not rs.source_fallback and run.model == REMOVAL_MODEL:
            # An Omni output (flow rule 2: no other model removes captions; a
            # run on another model is the operator's call, not a fill proposal).
            result[rs.id] = {"result_kind": "model"}
        else:
            result[rs.id] = {"result_kind": "unknown"}
    return {"context": {"run_blockers": run_blockers, "active_fill": active_fill,
                        "segments": segments},
            "result": result}


def fill_snapshot(session, run, project, *, src_fps: Optional[float] = None) -> dict:
    """The fill facts an autopilot planner reads every tick (AICP-364), safe
    for its read-only session: the API gates and the fill history only -- no
    ffprobe (*src_fps* comes from the last QA) and no composite identification
    (``_is_composite`` hashes a whole restored composite; the QA compute
    classified the results). Plus the run's preserved takes and the source's
    file identity."""
    from sqlalchemy import select

    from app import pipeline_segment_fill as psf
    from app.models import RunSegment, RunSegmentTake
    from app.pipeline_v2 import _active_swap_segment_ids, run_segment_defs

    defs = run_segment_defs(run, project)
    rs_by_def = {rs.segment_def_id: rs for rs in session.execute(
        select(RunSegment).where(RunSegment.run_id == run.id)).scalars()}
    active = _active_swap_segment_ids(run.stitch_plan, defs)
    ctx = _fill_context(session, run, project, defs, rs_by_def, active, src_fps=src_fps,
                        probe=False, classify=False)["context"]
    takes: dict = {}
    for t in session.execute(select(RunSegmentTake).where(RunSegmentTake.run_id == run.id)
                             .order_by(RunSegmentTake.created_at)).scalars():
        takes.setdefault(t.run_segment_id, []).append({
            "take_id": t.id, "seedance_task_id": t.seedance_task_id,
            "result_path": t.result_path, "purged": t.file_purged_at is not None,
            "created_at": t.created_at.isoformat() if t.created_at else None})
    return {"context": ctx, "takes": takes,
            "source_identity": psf.file_identity(project.source_local_path)}


def run_evidence(session, run, project) -> dict:
    """What a QA result is bound to: the final, the source, the snapshot / plan
    and every segment's current result. Compared before and after the scans."""
    from sqlalchemy import select

    from app import pipeline_segment_fill as psf
    from app.assistant_run_result import result_revision
    from app.models import RunSegment
    from app.segment_fill_models import RunSegmentFillJob

    rows = session.execute(select(RunSegment).where(RunSegment.run_id == run.id)).scalars()
    fills = session.execute(select(RunSegmentFillJob.id, RunSegmentFillJob.status,
                                   RunSegmentFillJob.provider_task_id)
                            .where(RunSegmentFillJob.run_id == run.id)).all()
    return {
        "run_id": run.id,
        "run_status": getattr(run.status, "value", run.status),
        "final_revision": result_revision(run),
        "segment_snapshot_sha": psf._sha(run.segment_snapshot),
        "stitch_plan_sha": psf._sha(run.stitch_plan),
        "source": [project.source_local_path, psf.file_identity(project.source_local_path)],
        "caption_detector_sha": psf._sha(project.caption_detector),
        # A run without a snapshot uses the project's live segments.
        "project_segments_sha": psf._sha([
            [sd.id, sd.index, sd.start_sec, sd.end_sec, sd.action, sd.pre_roll_sec,
             sd.post_roll_sec] for sd in sorted(project.segments, key=lambda d: d.index)]),
        "segments": {rs.id: [rs.local_result_path, psf.file_identity(rs.local_result_path),
                             rs.seedance_task_id, getattr(rs.status, "value", rs.status)]
                     for rs in sorted(rows, key=lambda r: r.id)},
        "fills": sorted([j_id, st, task] for j_id, st, task in fills),
    }


def _read_evidence(run_id: str) -> Optional[dict]:
    from sqlalchemy import text

    from app.db import SessionLocal
    from app.models import Run

    with SessionLocal() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        run = session.get(Run, run_id)
        out = run_evidence(session, run, run.project) if run is not None else None
        session.rollback()
    return out


def _same_file(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a) and bool(b) and os.path.realpath(a) == os.path.realpath(b)


def _dump(obj) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=1, default=str)
    sys.stdout.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.subtitle_loop",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("review", "qa"):
        p = sub.add_parser(name)
        if name == "review":
            p.add_argument("--project", metavar="PID")
            p.add_argument("--faces", action="store_true",
                           help="sample faces at 1 fps (InsightFace) for the end-card rule")

        else:
            p.add_argument("--run", metavar="RID")
            p.add_argument("--final", help="judge this file instead of the run's final")
            p.add_argument("--confirm-fp", metavar="INDICES", default="",
                           help="segment indices whose false positives were confirmed "
                                "on their crop sheets, e.g. 8 or 7,8")
            p.add_argument("--confirm-fill", metavar="FILL_IDS", default="",
                           help="fill job ids whose VOID composites passed the eye "
                                "check on their fill sheets (comma-separated)")
        p.add_argument("--no-locate", action="store_true",
                       help="do not locate the captions when the config names no band_y")
        p.add_argument("--source")
        p.add_argument("--segments", metavar="JSON", help="segments list (inline or a file)")
        p.add_argument("--config", metavar="JSON", help="caption detector config")
        p.add_argument("--price", type=float, help="credits per segment (else from the DB)")
        p.add_argument("--out-dir")
        p.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args(argv)

    try:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
        inputs: dict = {}
        price = args.price
        db_id = args.project if args.cmd == "review" else args.run
        if db_id:
            from sqlalchemy import text

            from app.db import SessionLocal

            with SessionLocal() as session:
                # Read-only by construction: the spend estimate flushes the
                # session, so the transaction itself refuses writes, and it is
                # rolled back, never committed.
                # One snapshot for the evidence, the inputs and the fill
                # context (PostgreSQL; the evidence is re-read after the scans).
                session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, "
                                     "READ ONLY"))
                if args.cmd == "review":
                    inputs = _project_inputs(session, db_id)
                    model, resolution = REMOVAL_MODEL, None
                else:
                    inputs = _run_inputs(session, db_id)
                    model, resolution = inputs["run"].model, inputs["run"].resolution
                if price is None:
                    price = price_per_segment(session, model, resolution)
                session.rollback()
        if args.source:
            inputs["source"] = args.source
        if args.segments:
            inputs["segments"] = _load_json_arg(args.segments)
        if args.config:
            inputs["config"] = _load_json_arg(args.config)
        if not inputs.get("source") or not inputs.get("segments"):
            raise QAToolError("need --project/--run or --source and --segments")
        if price is None:
            raise QAToolError("need --price without a database id")
        segments = segments_from(inputs["segments"])

        if args.cmd == "review":
            from app import spend

            faces = _face_samples(inputs["source"]) if args.faces else None
            location = None
            if not args.no_locate and not qa_tools.has_band(inputs.get("config")):
                location = qa_tools.locate_captions(inputs["source"],
                                                    config=inputs.get("config"))
            _dump(review_segmentation(
                inputs["source"], segments, config=inputs.get("config"),
                price_per_segment=price, faces=faces, fps=args.fps, out_dir=args.out_dir,
                admission_estimate=lambda sec: spend.video_credits(REMOVAL_MODEL, None, sec),
                location=location))
            return 0

        final = args.final or inputs.get("final")
        if not final:
            raise QAToolError("the run has no final (result_local_path); pass --final")
        own_final = bool(db_id) and _same_file(final, inputs.get("final"))
        if inputs.get("custom_plan") and (not args.final or own_final):
            raise QAToolError("the run has a custom stitch plan: its final is not on "
                              "the source timeline")
        # Requests are actionable only when the judged inputs ARE the live run:
        # its own final, source, segments and detector config.
        overrides = [name for name, val in (("--source", args.source),
                                            ("--segments", args.segments),
                                            ("--config", args.config)) if val]
        reason = None
        if not db_id:
            reason = "no database run: files only"
        elif not own_final:
            reason = "--final is not the run's current final"
        elif overrides:
            reason = "judged with %s instead of the run's own" % ", ".join(overrides)
        if reason and db_id:
            # The segment provenance and the fill history describe the LIVE
            # run, not these files: judge them as plain model outputs.
            segments = segments_from([{k: v for k, v in sg.items()
                                       if k not in ("result_kind", "fill")}
                                      for sg in inputs["segments"]])
        confirmed = [int(x) for x in args.confirm_fp.split(",") if x.strip()]
        confirmed_fills = [x.strip() for x in args.confirm_fill.split(",") if x.strip()]
        qa = post_run_qa(inputs["source"], final, segments, config=inputs.get("config"),
                         fps=args.fps, out_dir=args.out_dir,
                         confirmed_false_positives=confirmed,
                         confirmed_fills=confirmed_fills)
        if not args.no_locate and not qa_tools.has_band(inputs.get("config")):
            chk = out_of_band_check(qa_tools.locate_captions(
                inputs["source"], config=inputs.get("config")), inputs.get("config"))
            qa["band_location"] = chk
            if chk["warning"]:
                qa["warnings"].append(chk["warning"])
        if not reason and inputs.get("evidence") is not None \
                and _read_evidence(db_id) != inputs["evidence"]:
            reason = ("the run changed while QA ran (final, source, segments or fills): "
                      "run qa again")
        run_id = db_id or "RUN_ID"
        fill = fill_proposal(run_id, qa, segments,
                             context=None if reason else inputs.get("fill_context"),
                             actionable=reason is None, not_actionable_reason=reason)
        fill["evidence"] = inputs.get("evidence")
        handoff = handoff_proposal(run_id, qa, segments)
        if reason and handoff.get("request"):
            handoff = {"request": None, "reason": reason, "would_request": handoff["request"]}
        _dump({
            "qa": qa,
            "fill": fill,
            "operator_options": {
                "omni_rerun": rerun_proposal(run_id, qa, segments,
                                             takes=inputs.get("takes", {}),
                                             price_per_segment=price,
                                             unfinished=inputs.get("unfinished", []),
                                             admission_estimate=inputs.get("admission")),
            },
            "handoff": handoff,
        })
    except (QAToolError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
