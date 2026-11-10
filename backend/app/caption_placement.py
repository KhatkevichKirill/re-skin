"""caption_placement.py — put generated captions over the ORIGINAL caption area.

AICP-368. A subtitle-removal run leaves faint patches where the burned-in
captions used to be: the Fal VOID fills of leftover segments
(``run_segment_fill_jobs``) and the Omni reconstruction of the band itself.
Our own burned captions sit bottom-centred (:data:`app.captions.MARGIN_V_RATIO`,
baseline at y ~= 0.84), i.e. below that band, so the patches stay uncovered.
A *placement* moves them: every caption block is centred on the band's centre,
so the new captions cover the patched area.

Where the band comes from — BOTH sources, unioned
-------------------------------------------------
* **VOID fill boxes** of the source run (``mask_box``, the padded union of the
  detector's hit boxes over the filled segment, or the operator's box). They
  are exactly where the most visible patches are, but exist only for the
  leftover segments that needed a fill. Only fills known to be IN the copied
  final count (``pipeline_segment_fill.delivery_state == "delivered"``): not
  a superseded fill, not one still waiting for its re-stitch, not a
  ``republished`` final (an assistant publication may be another picture —
  an audio-patched run therefore falls back to the detector band). Once
  read, the fill evidence is FROZEN in the value (``source.fill_band``) and
  reused by the scan, by a child promoted while the scan is pending and by an
  explicit re-derive; without frozen evidence (a project made before
  AICP-368) only fills applied by the copy time count.
* **The detector's MEDIAN caption band** on the parent subtitle-removal
  project's SOURCE (``qa_tools.calibrate_captions`` with that project's
  ``caption_detector`` config): ``[x0_p5, y0_p50, x1_p95, y1_p50]``. It says
  where the captions sat across the whole video — i.e. where the Omni
  reconstruction is — but only as a statistic. The median rows, not
  ``y0_p5``/``y1_p95``: two-line captions and stray hits drag those far up
  (on 328341d4 two-line captions start at y ~= 0.43, which would centre the
  band at ~0.55 and miss the one-line band and both fills).

Several removal passes (a subtitle-removal project made from another one's
run) all left patches: the evidence is the whole chain
(:func:`removal_chain`) — every pass's fills and a scan of every pass's own
source — never just the nearest pass, and never the parent's stored value.

The union of the two is the band; its centre is ``center_y``. On the real case
(project 328341d4, run 2845eef3) both agree: fills y 0.5821-0.6627 (centre
0.622), detector median 0.605-0.643 (centre 0.624). The individual boxes and
the detector statistics are kept in ``source`` as evidence, with warnings when
the band is broad or the two sources disagree.

The detector scan streams frames (one at a time, 2 fps, 540x960) in a worker
job on the ``analyze`` queue; it never decodes a whole video into memory.

Fitting rule (``app.captions``): the block is centred on ``center_y`` and grows
symmetrically; rows wrap to ``max_width`` (default: the house width) and up
to :data:`app.captions.PLACEMENT_MAX_ROWS` (3) rows per caption, longer text is
split in time as today. Text too short to cover the band still covers its
centre.

Life cycle and ownership (``source.kind``)
------------------------------------------
* ``derived`` — computed by this module. Written at project-from-run when the
  parent is a subtitle-removal project: first a provisional value from the
  fill boxes (``scan: pending``), then the scan job replaces it with the union
  (``scan: done|failed|uncalibrated``). The scan reuses the fill evidence
  frozen in the provisional value (``source.fill_band``), never the run's
  later fills: those may belong to a re-delivery the copied video does not
  contain. Re-derivable.
* ``inherited`` — copied from the parent project at project-from-run (a
  subtitle-removal -> face swap -> localisation chain keeps the band) when the
  parent's value is settled. A parent still waiting for its scan is NOT copied
  (that would freeze the provisional value): the child derives its own from
  the lineage and gets its own scan. Never refreshed from the parent
  afterwards.
* ``manual`` — set by the operator (``PATCH /projects/{pid}``). A PATCH of
  ``null`` stores the sticky ``{"mode": "bottom"}``: "house position, and do
  not re-derive". Only an explicit derive request replaces a manual value.

Every write bumps ``source.rev``; the scan job carries the rev it was enqueued
for and writes only if the row (locked) still has that rev and kind
``derived``, so an operator edit or a newer derivation is never overwritten by
a late or out-of-order scan. A scan killed by a deploy leaves the provisional
value with ``scan: pending``; re-run it with
``POST /api/v2/projects/{pid}/caption-placement/derive`` or
``python -m app.caption_placement derive <pid>``.

A caption job copies the project's placement when it is CREATED
(``run_captions.placement``), so retries burn the same; NULL there is the house
style (every caption made before AICP-368).

Autopilot hand-off (AICP-364/366): :func:`promote_run_to_project` does all of
this automatically; an autopilot that builds a derived project another way
calls :func:`request_derivation` (writes the provisional value, enqueues the
scan) or, inside a worker, :func:`derive_now`. The captions step needs nothing
special: ``create_caption`` snapshots the project's placement by default
(:func:`snapshot_for_caption`, which any other ``RunCaption`` creator must use
too).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field as PydField, ValidationError, model_validator

from . import captions as captions_mod

log = logging.getLogger(__name__)

MODE_BAND = "band"
MODE_BOTTOM = "bottom"
KIND_DERIVED = "derived"
KIND_INHERITED = "inherited"
KIND_MANUAL = "manual"

SCAN_PENDING = "pending"
SCAN_DONE = "done"
SCAN_FAILED = "failed"
SCAN_UNCALIBRATED = "uncalibrated"

#: Detector sampling rate for the band scan. The median rows are robust at
#: 2 fps (a 60 s source is 120 frames) and it keeps the job to seconds of CPU.
SCAN_FPS = 2.0
#: How many project-from-run hops to walk back collecting subtitle-removal
#: passes (a guard against corrupt cycles, not a product limit: real chains
#: are two or three hops).
MAX_LINEAGE_DEPTH = 20
#: center_y bounds: a three-row block (~0.14 of the frame) centred here stays
#: inside the frame with margin.
CENTER_Y_MIN, CENTER_Y_MAX = 0.12, 0.88
#: max_width bounds (fraction of frame width); the top is the house width.
MAX_WIDTH_MIN = 0.30
MAX_WIDTH_MAX = round(1.0 - 2 * captions_mod.MARGIN_H_RATIO, 4)
#: Warnings thresholds (fractions of frame height).
BROAD_BAND = 0.20
SOURCES_DISAGREE = 0.05
#: A burn source whose aspect differs from the measured frame by more than this
#: (relative) is not the same picture geometry: placement is skipped.
ASPECT_TOLERANCE = 0.02

_SOURCE_MAX_CHARS = 8000
#: Evidence lists kept in ``source`` (the band itself uses every box).
_EVIDENCE_MAX_ITEMS = 20


class PlacementError(Exception):
    """Invalid placement value or an impossible derivation request."""


def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("must be finite")
    return v


class CaptionPlacement(BaseModel):
    """A validated project/caption placement (the stored JSON)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["band", "bottom"] = MODE_BAND
    center_y: Optional[float] = PydField(None, ge=CENTER_Y_MIN, le=CENTER_Y_MAX,
                                         allow_inf_nan=False)
    max_width: Optional[float] = PydField(None, ge=MAX_WIDTH_MIN, le=MAX_WIDTH_MAX,
                                          allow_inf_nan=False)
    max_rows: int = PydField(captions_mod.PLACEMENT_MAX_ROWS, ge=1,
                             le=captions_mod.PLACEMENT_MAX_ROWS)
    # [x0, y0, x1, y1] fractions: what the placement was derived to cover.
    band: Optional[tuple[float, float, float, float]] = None
    # width / height of the frame the band was measured on.
    frame_aspect: Optional[float] = PydField(None, gt=0.1, lt=10.0, allow_inf_nan=False)
    # Provenance (server-written): kind, rev, lineage, evidence, scan state.
    source: Optional[dict] = None

    @model_validator(mode="after")
    def _check(self):
        if self.mode == MODE_BAND and self.center_y is None:
            raise ValueError("center_y is required when mode is 'band'")
        if self.band is not None:
            x0, y0, x1, y1 = (_finite(float(v)) for v in self.band)
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError("band must be ordered fractions [x0, y0, x1, y1]")
        if self.source is not None and len(json.dumps(self.source)) > _SOURCE_MAX_CHARS:
            raise ValueError("source is too large")
        return self


def validate(value) -> Optional[CaptionPlacement]:
    """Coerce None / dict / model into a :class:`CaptionPlacement` or raise
    :class:`PlacementError` with field paths only (never the input)."""
    if value is None:
        return None
    if isinstance(value, CaptionPlacement):
        return value
    if not isinstance(value, dict):
        raise PlacementError("caption placement must be a JSON object")
    try:
        return CaptionPlacement.model_validate(value)
    except ValidationError as exc:
        parts = ["%s: %s" % (".".join(str(x) for x in e["loc"]) or "placement", e["msg"])
                 for e in exc.errors(include_url=False, include_input=False,
                                     include_context=False)]
        raise PlacementError("invalid caption placement: " + "; ".join(parts)) from None


def _dump(p: CaptionPlacement) -> dict:
    return p.model_dump(mode="json", exclude_none=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rev_of(value) -> int:
    try:
        return int(((value or {}).get("source") or {}).get("rev") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def kind_of(value) -> Optional[str]:
    try:
        return ((value or {}).get("source") or {}).get("kind")
    except AttributeError:
        return None


def bottom(kind: str = KIND_MANUAL, *, rev: int = 0, **extra) -> dict:
    """The explicit house-position value (renders exactly like NULL)."""
    return {"mode": MODE_BOTTOM, "max_rows": captions_mod.PLACEMENT_MAX_ROWS,
            "source": {"kind": kind, "rev": rev, **extra}}


def manual(value, *, current=None) -> dict:
    """What a ``PATCH /projects/{pid}`` with *value* stores: ``null`` -> the
    sticky house position; an object -> validated, ``source`` replaced by the
    server's (kind ``manual``, next rev)."""
    rev = rev_of(current) + 1
    # Carry the frozen fill evidence (what this project's video contains) so
    # a later explicit re-derive can still use it.
    carried = {}
    csrc = (current or {}).get("source") if isinstance(current, dict) else None
    if isinstance(csrc, dict) and "fill_band" in csrc and csrc.get("run_id"):
        carried = {"run_id": csrc["run_id"], **{k: csrc.get(k) for k in _FROZEN_KEYS}}
    # ...and the whole measured band (fills AND detector), so a later removal
    # pass still has it if the earlier pass is deleted (removal_chain).
    if isinstance(csrc, dict):
        if (csrc.get("kind") in (KIND_DERIVED, KIND_INHERITED)
                and current.get("mode") == MODE_BAND and current.get("band")):
            carried["evidence_band"] = list(current["band"])
        elif csrc.get("evidence_band"):
            carried["evidence_band"] = csrc["evidence_band"]
    if value is None:
        return _dump(validate(bottom(KIND_MANUAL, rev=rev, set_at=_now_iso(), **carried)))
    if not isinstance(value, dict):
        raise PlacementError("caption placement must be a JSON object or null")
    body = dict(value)
    body["source"] = {"kind": KIND_MANUAL, "rev": rev, "set_at": _now_iso(), **carried}
    return _dump(validate(body))


CAPTION_CHOICES = ("project", "bottom")


def snapshot_for_caption(project, choice: Optional[str] = "project") -> Optional[dict]:
    """The value a NEW caption job freezes into ``run_captions.placement``.

    ``project`` (default; None/blank too): the project's current
    ``caption_placement`` as stored (None = house position). ``bottom``: the
    explicit house position whatever the project says. Raises
    :class:`PlacementError` on any other choice. ``create_caption`` uses this;
    so must any other code that creates a ``RunCaption`` (e.g. an autopilot
    captions step), or its caption burns at the house position.
    """
    choice = (choice or "project").strip() or "project"
    if choice not in CAPTION_CHOICES:
        raise PlacementError("placement must be 'project' or 'bottom'")
    if choice == "bottom":
        return bottom(KIND_MANUAL)
    value = getattr(project, "caption_placement", None) if project is not None else None
    # A deep copy: the frozen value must never share nested lists/dicts with
    # the project's JSON.
    return copy.deepcopy(value) if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Derivation (pure)
# ---------------------------------------------------------------------------


def _clean_box(box) -> Optional[list[float]]:
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return [round(v, 4) for v in vals]


def detector_band(calibration: Optional[dict]) -> Optional[list[float]]:
    """The detector's median caption band ``[x0_p5, y0_p50, x1_p95, y1_p50]``
    from a :func:`app.qa_tools.calibrate_captions` result, or None when the
    calibration failed (hit ratio below the floor) or has no geometry."""
    if not calibration or not calibration.get("calibrated"):
        return None
    g = calibration.get("geometry") or {}
    try:
        return _clean_box([g["x0_p5"], g["y0_p50"], g["x1_p95"], g["y1_p50"]])
    except KeyError:
        return None


def derive_placement(*, fill_boxes: list, calibration: Optional[dict] = None,
                     frame_aspect: Optional[float] = None,
                     source: Optional[dict] = None) -> dict:
    """The placement for a subtitle-removal lineage (see the module docstring).

    *fill_boxes*: normalised ``[x0, y0, x1, y1]`` of the delivered VOID fills.
    *calibration*: a ``calibrate_captions`` result (None = not scanned), or a
    list of them — one per subtitle-removal pass of the chain, nearest first.
    Returns a validated dict: ``mode: band`` centred on the union of the fill
    boxes and the detector's median band, or ``mode: bottom`` (house position)
    when neither source has anything. *source* is merged into the provenance.
    """
    boxes = [b for b in (_clean_box(b) for b in fill_boxes or []) if b]
    cals = (list(calibration) if isinstance(calibration, list)
            else [] if calibration is None else [calibration])
    cals = [c for c in cals if isinstance(c, dict)]
    dets = [d for d in (detector_band(c) for c in cals) if d]
    det = ([min(d[0] for d in dets), min(d[1] for d in dets),
            max(d[2] for d in dets), max(d[3] for d in dets)] if dets else None)
    parts = boxes + ([det] if det else [])
    warnings: list[str] = []
    evidence = dict(source or {})
    evidence["fills"] = len(boxes)
    evidence["fill_boxes"] = boxes[:_EVIDENCE_MAX_ITEMS]
    # The union of EVERY fill box: the frozen fill evidence the scan reuses
    # (see scan_and_store), so it never needs the list to be complete.
    evidence["fill_band"] = ([min(b[0] for b in boxes), min(b[1] for b in boxes),
                              max(b[2] for b in boxes), max(b[3] for b in boxes)]
                             if boxes else None)
    if isinstance(evidence.get("fill_ids"), list):
        evidence["fill_ids"] = evidence["fill_ids"][:_EVIDENCE_MAX_ITEMS]
    summaries = []
    for cal in cals:
        summaries.append({
            "calibrated": bool(cal.get("calibrated")),
            "hit_ratio": cal.get("hit_ratio"),
            "geometry": cal.get("geometry"),
        })
        if not cal.get("calibrated") and "detector_uncalibrated" not in warnings:
            warnings.append("detector_uncalibrated")
        g = cal.get("geometry") or {}
        try:
            if (float(g["y1_p95"]) - float(g["y0_p5"]) > BROAD_BAND
                    and "detector_spread" not in warnings):
                warnings.append("detector_spread")  # multi-line captions / stray hits
        except (KeyError, TypeError, ValueError):
            pass
    if summaries:
        evidence["detector"] = summaries[0]
        if len(summaries) > 1:
            evidence["detectors"] = summaries
    if not parts:
        evidence["warnings"] = warnings + ["no_evidence"]
        return _dump(validate(bottom(evidence.pop("kind", KIND_DERIVED),
                                     rev=int(evidence.pop("rev", 0)), **evidence)))
    x0 = min(b[0] for b in parts)
    y0 = min(b[1] for b in parts)
    x1 = max(b[2] for b in parts)
    y1 = max(b[3] for b in parts)
    if y1 - y0 > BROAD_BAND:
        warnings.append("broad_band")
    if boxes and det:
        fill_c = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2.0
        det_c = (det[1] + det[3]) / 2.0
        if abs(fill_c - det_c) > SOURCES_DISAGREE:
            warnings.append("fills_detector_disagree")
    center = min(CENTER_Y_MAX, max(CENTER_Y_MIN, (y0 + y1) / 2.0))
    evidence["warnings"] = warnings
    evidence.setdefault("kind", KIND_DERIVED)
    out = {
        "mode": MODE_BAND,
        "center_y": round(center, 4),
        "max_rows": captions_mod.PLACEMENT_MAX_ROWS,
        "band": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
        "source": evidence,
    }
    if frame_aspect and math.isfinite(frame_aspect) and 0.1 < frame_aspect < 10.0:
        out["frame_aspect"] = round(float(frame_aspect), 4)
    return _dump(validate(out))


def burn_placement(value, *, width: int, height: int) -> Optional[captions_mod.Placement]:
    """What :func:`app.captions.build_ass` gets for a caption's stored
    placement: None (house style) for NULL / ``mode: bottom`` / an invalid
    value / a frame whose aspect is not the one the band was measured on."""
    if value is None:
        return None
    try:
        p = validate(value)
    except PlacementError as exc:
        log.warning("caption placement ignored (%s) — burning at the house position", exc)
        return None
    if p is None or p.mode != MODE_BAND or p.center_y is None:
        return None
    if p.frame_aspect and width > 0 and height > 0:
        aspect = width / float(height)
        if abs(aspect - p.frame_aspect) / p.frame_aspect > ASPECT_TOLERANCE:
            log.warning(
                "caption placement ignored: frame %dx%d (aspect %.4f) is not the "
                "measured %.4f — burning at the house position", width, height,
                aspect, p.frame_aspect)
            return None
    return captions_mod.Placement(center_y=p.center_y, max_width=p.max_width,
                                  max_rows=p.max_rows)


# ---------------------------------------------------------------------------
# Lineage + evidence (DB)
# ---------------------------------------------------------------------------


def removal_chain(db, project) -> list:
    """Every subtitle-removal ancestor reached through project-from-run
    backlinks, nearest first, as ``(run, subtitle_removal_project,
    first_child)``: the run whose final the chain copied, its project, and the
    project that copied it. More than one entry = several removal passes;
    every pass's patches are still in the picture."""
    from .models import Run, VideoProject
    from .project_types import SUBTITLE_REMOVAL

    out = []
    seen = {project.id}
    current = project
    for _ in range(MAX_LINEAGE_DEPTH):
        rid = current.source_run_id
        if not rid:
            break
        run = db.get(Run, rid)
        if run is None:
            break
        parent = db.get(VideoProject, run.project_id)
        if parent is None or parent.id in seen:
            break
        if parent.project_type == SUBTITLE_REMOVAL:
            out.append((run, parent, current))
        seen.add(parent.id)
        current = parent
    return out


def find_lineage(db, project):
    """The nearest entry of :func:`removal_chain`, or None."""
    chain = removal_chain(db, project)
    return chain[0] if chain else None


def delivered_fill_boxes(db, run, *, applied_before=None) -> list[dict]:
    """The applied VOID fills of *run* known to be IN its current final
    (``delivery_state == "delivered"``: the segment still uses the composite
    and the pipeline's own stitch re-delivered it), oldest first, as
    ``{"id", "box"}``. Not ``republished`` (an assistant publication may be
    another picture) and not a fill still waiting for its re-stitch (it is not
    in the file a promote copies). *applied_before* (naive UTC) keeps only
    fills applied by then — the copy time of a project made earlier."""
    from sqlalchemy import select

    from .models import RunSegment
    from .pipeline_segment_fill import delivery_state
    from .segment_fill_models import FillStatus, RunSegmentFillJob

    jobs = db.execute(
        select(RunSegmentFillJob)
        .where(RunSegmentFillJob.run_id == run.id,
               RunSegmentFillJob.status == FillStatus.applied.value)
        .order_by(RunSegmentFillJob.created_at)
    ).scalars().all()
    stitched = _final_stitched_from_segments(db, run) if jobs else False
    out = []
    for job in jobs:
        rs = db.get(RunSegment, job.run_segment_id)
        if applied_before is not None and (job.applied_at is None
                                           or job.applied_at > applied_before):
            continue
        state = delivery_state(job, run, rs, db=db)
        # "republished" covers finals the platform still stitched from the
        # segment results (a promoted range attempt), optionally behind audio
        # patches (-c:v copy): the fill — its segment still uses the composite
        # — is in that picture too. Not an assistant publication.
        if state != "delivered" and not (state == "republished" and stitched):
            continue
        box = _clean_box(job.mask_box or [])
        if box:
            out.append({"id": job.id, "box": box})
    return out


def _final_stitched_from_segments(db, run) -> bool:
    """True when *run*'s final picture is a platform stitch of its segment
    results: the pipeline's ``results/final.mp4`` or a promoted range
    attempt's final, possibly behind promoted audio patches (which copy the
    picture). False for anything else (an assistant publication)."""
    import os

    from sqlalchemy import select

    from . import storage
    from .models import RunAudioPatch, RunSegmentRangeAttempt

    patches = db.execute(select(RunAudioPatch).where(
        RunAudioPatch.run_id == run.id, RunAudioPatch.promoted_at.isnot(None))).scalars().all()
    by_output = {p.output_path: p for p in patches if p.output_path}
    path, seen = run.result_local_path, set()
    while path and path in by_output and path not in seen:
        seen.add(path)
        path = by_output[path].base_result_path
    if not path:
        return False
    if path == os.path.join(storage.run_dir(run.id, run.project_id), "results", "final.mp4"):
        return True
    return db.execute(select(RunSegmentRangeAttempt.id).where(
        RunSegmentRangeAttempt.run_id == run.id,
        RunSegmentRangeAttempt.promoted_at.isnot(None),
        RunSegmentRangeAttempt.final_local_path == path).limit(1)).first() is not None


def _aspect(project) -> Optional[float]:
    w, h = getattr(project, "width", None), getattr(project, "height", None)
    if w and h and w > 0 and h > 0:
        return w / float(h)
    return None


_FROZEN_KEYS = ("fills", "fill_boxes", "fill_ids", "fill_band")


def _frozen(value, run_id: str) -> Optional[dict]:
    """The fill evidence frozen in a derived value for *run_id*'s lineage
    (read when that video was copied), or None."""
    src = (value or {}).get("source") if isinstance(value, dict) else None
    if not isinstance(src, dict) or src.get("kind") not in (KIND_DERIVED, KIND_INHERITED,
                                                            KIND_MANUAL):
        return None
    if src.get("run_id") != run_id or "fill_band" not in src:
        return None
    return {k: src.get(k) for k in _FROZEN_KEYS}


def _derive_with(frozen: dict, *, calibration, frame_aspect, source: dict) -> dict:
    """:func:`derive_placement` from FROZEN fill evidence (its one union band)
    with the frozen per-box evidence kept in the provenance."""
    band = _clean_box(frozen.get("fill_band") or []) if frozen.get("fill_band") else None
    value = derive_placement(fill_boxes=[band] if band else [], calibration=calibration,
                             frame_aspect=frame_aspect, source=source)
    value["source"]["fills"] = int(frozen.get("fills") or 0)
    value["source"]["fill_boxes"] = list(frozen.get("fill_boxes") or [])
    value["source"]["fill_ids"] = list(frozen.get("fill_ids") or [])
    return _dump(validate(value))


def _provisional(db, chain: list, *, rev: int, scan: str,
                 frozen: Optional[dict] = None) -> dict:
    run, ancestor, _ = chain[0]
    source = {"kind": KIND_DERIVED, "rev": rev, "run_id": run.id,
              "project_id": ancestor.id, "scan": scan, "derived_at": _now_iso()}
    if len(chain) > 1:
        source["chain"] = [{"run_id": r.id, "project_id": a.id} for r, a, _ in chain]
    if frozen is not None:
        return _derive_with(frozen, calibration=None, frame_aspect=_aspect(ancestor),
                            source=source)
    # Every pass's fills that were in its run's final when the chain copied
    # it: now, for a project being created; by the copier's creation time for
    # an older one.
    # An earlier pass's copier froze what IT copied: prefer that evidence
    # (its band may also include passes further up) over the run's current
    # state, which can have superseded a fill that is still in the picture.
    fills, boxes = [], []
    for _, a, _ in chain:
        # A removal-pass project's own stored evidence describes the passes
        # before it — still in its source, so in every descendant — even when
        # the link to them has been deleted since: its whole derived/inherited
        # band (fills AND earlier detector scans), or a manual value's frozen
        # fill band.
        value = a.caption_placement if isinstance(a.caption_placement, dict) else {}
        own = value.get("source") if isinstance(value.get("source"), dict) else {}
        band = _clean_box(own["evidence_band"]) if own.get("evidence_band") else None
        if (band is None and own.get("kind") in (KIND_DERIVED, KIND_INHERITED)
                and value.get("mode") == MODE_BAND and value.get("band")):
            band = _clean_box(value["band"])
        if band is None and own.get("fill_band"):
            band = _clean_box(own["fill_band"])
        if band:
            boxes.append(band)
    for i, (r, _, first_child) in enumerate(chain):
        fz = (_frozen(getattr(first_child, "caption_placement", None), r.id)
              if i > 0 else None)
        band = _clean_box(fz.get("fill_band") or []) if fz and fz.get("fill_band") else None
        if fz is not None:
            if band:
                boxes.append(band)
            continue
        got = delivered_fill_boxes(db, r, applied_before=getattr(first_child, "created_at",
                                                                 None))
        fills += got
        boxes += [f["box"] for f in got]
    source["fill_ids"] = [f["id"] for f in fills]
    return derive_placement(fill_boxes=boxes, calibration=None,
                            frame_aspect=_aspect(ancestor), source=source)


def _pending(value) -> bool:
    """Not settled: the scan is still to run, or failed (retryable) — a copy
    would freeze an incomplete value into the child."""
    src = (value or {}).get("source") or {}
    return bool({SCAN_PENDING, SCAN_FAILED} & {src.get("scan"), src.get("origin_scan")})


def inherit_on_promote(db, new_project, run, parent) -> bool:
    """Set *new_project*'s placement at project-from-run (before its commit).

    The parent's own SETTLED placement is copied (kind ``inherited``).
    Otherwise — no parent placement, or one still waiting for its scan, which
    a copy would freeze — a project with a subtitle-removal lineage (the
    parent itself, or further up) gets its own provisional derived value from
    that lineage's fills. Returns True when a detector scan should be
    enqueued after the commit (:func:`enqueue_scan`)."""
    from .project_types import SUBTITLE_REMOVAL

    if parent is None:
        return False
    removal_parent = parent.project_type == SUBTITLE_REMOVAL
    chain = None
    if parent.caption_placement and not removal_parent and _pending(parent.caption_placement):
        chain = removal_chain(db, new_project)
    # Inherit a settled value; an unsettled one only when no chain is left to
    # derive from (best evidence available).
    if (parent.caption_placement and not removal_parent
            and (not _pending(parent.caption_placement) or not chain)):
        try:
            p = validate(parent.caption_placement)
        except PlacementError:
            log.warning("project %s: parent %s placement invalid — not inherited",
                        new_project.id, parent.id)
            return False
        body = _dump(p)
        psrc = p.source or {}
        body["source"] = {"kind": KIND_INHERITED, "rev": 1, "from_project_id": parent.id,
                          "from_run_id": run.id, "origin": psrc.get("kind"),
                          "origin_scan": psrc.get("scan") or psrc.get("origin_scan"),
                          "inherited_at": _now_iso()}
        # Keep the lineage run and its frozen fill evidence, so an explicit
        # re-derive of this child reuses what its video contains.
        if "fill_band" in psrc and psrc.get("run_id"):
            body["source"]["run_id"] = psrc["run_id"]
            body["source"].update({k: psrc.get(k) for k in _FROZEN_KEYS})
        # The measured band travels too: the parent's own derived/inherited
        # band, or the one a manual edit preserved.
        if psrc.get("evidence_band"):
            body["source"]["evidence_band"] = psrc["evidence_band"]
        elif psrc.get("kind") in (KIND_DERIVED, KIND_INHERITED) and p.mode == MODE_BAND and p.band:
            body["source"]["evidence_band"] = list(p.band)
        new_project.caption_placement = _dump(validate(body))
        return False
    if chain is None:
        chain = removal_chain(db, new_project)
    if not chain:
        return False
    # A subtitle-removal parent is a NEW removal pass: the evidence is the
    # whole chain (every pass's fills and source scan), never the parent's
    # value, which describes only the earlier passes. A pending parent's
    # frozen fills (same nearest run) describe the video this child copies.
    new_project.caption_placement = _provisional(
        db, chain, rev=1, scan=SCAN_PENDING,
        frozen=_frozen(parent.caption_placement, chain[0][0].id),
    )
    return True


def request_derivation(db, project) -> int:
    """(Re)derive *project*'s placement from its subtitle-removal lineage:
    lock the row, write the provisional fill-box value (replacing whatever is
    there — this is an explicit request), commit, and return the rev the scan
    must be enqueued for. Raises :class:`PlacementError` without a lineage."""
    db.refresh(project, with_for_update=True)
    chain = removal_chain(db, project)
    if not chain:
        db.rollback()
        raise PlacementError("project has no subtitle-removal lineage to derive from")
    rev = rev_of(project.caption_placement) + 1
    # Frozen fill evidence (from when this project's video was copied) wins
    # over the run's fills now; without it, fills applied by the copy time.
    project.caption_placement = _provisional(
        db, chain, rev=rev, scan=SCAN_PENDING,
        frozen=_frozen(project.caption_placement, chain[0][0].id),
    )
    db.commit()
    return rev


def enqueue_scan(project_id: str, rev: int) -> bool:
    """Enqueue the detector scan; False (logged) when Redis refuses. The
    provisional value stays, marked ``scan: pending``."""
    try:
        from .tasks import enqueue_caption_placement_scan

        enqueue_caption_placement_scan(project_id, rev)
        return True
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        log.error("caption placement scan for project %s not enqueued: %s", project_id, exc)
        return False


def _scan(source_path: Optional[str], detector_config) -> tuple[str, Optional[dict], Optional[str]]:
    """Run the streamed detector calibration: (scan state, calibration, error)."""
    import os

    from . import qa_tools

    if not source_path or not os.path.exists(source_path):
        return SCAN_FAILED, None, "subtitle-removal source missing on disk"
    try:
        cal = qa_tools.calibrate_captions(source_path, fps=SCAN_FPS, config=detector_config)
    except Exception as exc:
        return SCAN_FAILED, None, f"{type(exc).__name__}: {str(exc)[:300]}"
    cal.pop("source", None)  # a host path: not provenance anyone needs
    return (SCAN_DONE if cal.get("calibrated") else SCAN_UNCALIBRATED), cal, None


def scan_and_store(project_id: str, rev: int) -> Optional[dict]:
    """Worker job: detector scan of the lineage's source, then a
    compare-and-set write (row lock; still kind ``derived`` at *rev*).
    Returns the stored placement, or None when the job was stale."""
    from .db import get_session
    from .models import VideoProject

    # AICP-350: a running scan has no status the deploy drain could count, so
    # it holds the admission gate's share lock for its whole run (a deploy's
    # begin.sql waits for it) and does not start while the gate is paused.
    from . import ops_gate

    with ops_gate.job_barrier(get_session) as admitted:
        if not admitted:
            from .tasks import (CAPTION_PLACEMENT_SCAN_TIMEOUT, QUEUE_ANALYZE,
                                defer_uncounted_job)

            defer_uncounted_job(QUEUE_ANALYZE, "app.tasks.run_caption_placement_scan",
                                project_id, int(rev), job_timeout=CAPTION_PLACEMENT_SCAN_TIMEOUT)
            return None
        return _scan_and_store(project_id, rev)


def _scan_and_store(project_id: str, rev: int) -> Optional[dict]:
    from .db import get_session
    from .models import VideoProject

    with get_session() as db:
        project = db.get(VideoProject, project_id)
        if project is None:
            return None
        current = project.caption_placement
        if kind_of(current) != KIND_DERIVED or rev_of(current) != rev:
            log.info("caption placement scan %s rev %s: stale (now %s rev %s) — skipped",
                     project_id, rev, kind_of(current), rev_of(current))
            return None
        chain = removal_chain(db, project)
        if not chain:
            # The parent project or run was deleted (the copy outlives them):
            # settle as failed on the frozen evidence instead of staying pending.
            src0 = (current or {}).get("source") or {}
            run_id, ancestor_id = src0.get("run_id"), src0.get("project_id")
            aspect = (current or {}).get("frame_aspect")
            chain_meta = src0.get("chain")
            sources = []
        else:
            run, ancestor, _ = chain[0]
            run_id, ancestor_id, aspect = run.id, ancestor.id, _aspect(ancestor)
            chain_meta = ([{"run_id": r.id, "project_id": a.id} for r, a, _ in chain]
                          if len(chain) > 1 else None)
            # One streamed scan per removal pass's source (its own config).
            sources = [(a.source_local_path, a.caption_detector) for _, a, _ in chain]
    # No session (and no transaction) is held across the scans.
    if not sources:
        scan, cal, error = SCAN_FAILED, None, "subtitle-removal lineage deleted"
    else:
        results = [_scan(path, config) for path, config in sources]
        cal = [c for _, c, _ in results if c is not None] or None
        states = [st for st, _, _ in results]
        # Any failed pass leaves the whole value retryable ("failed"): its
        # band may be missing, so descendants re-derive instead of copying.
        scan = (SCAN_FAILED if SCAN_FAILED in states
                else SCAN_DONE if SCAN_DONE in states else SCAN_UNCALIBRATED)
        error = "; ".join(e for _, _, e in results if e) or None
    with get_session() as db:
        project = db.get(VideoProject, project_id)
        if project is None:
            return None
        db.refresh(project, with_for_update=True)
        current = project.caption_placement
        if kind_of(current) != KIND_DERIVED or rev_of(current) != rev:
            log.info("caption placement scan %s rev %s: superseded while scanning", project_id, rev)
            return None
        # The fills are the ones FROZEN in the provisional value, read when
        # the project's video was copied (or the derivation requested) — not
        # the run's current fills, which may describe a later re-delivery.
        frozen = _frozen(current, run_id) or {k: None for k in _FROZEN_KEYS}
        src = {"kind": KIND_DERIVED, "rev": rev + 1, "run_id": run_id,
               "project_id": ancestor_id, "scan": scan, "derived_at": _now_iso()}
        if chain_meta:
            src["chain"] = chain_meta
        if error:
            src["scan_error"] = error[:600]
        value = _derive_with(frozen, calibration=cal, frame_aspect=aspect, source=src)
        project.caption_placement = value
        db.commit()
    log.info("caption placement for project %s: %s (scan=%s)", project_id,
             {k: value.get(k) for k in ("mode", "center_y", "band")}, scan)
    return value


def derive_now(project_id: str) -> dict:
    """Synchronous derivation (worker / CLI only — it scans video): request,
    then scan inline. Raises :class:`PlacementError` without a lineage."""
    from .db import get_session
    from .models import VideoProject

    with get_session() as db:
        project = db.get(VideoProject, project_id)
        if project is None:
            raise PlacementError(f"project {project_id} not found")
        rev = request_derivation(db, project)
    return scan_and_store(project_id, rev) or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.caption_placement",
                                 description="AICP-368 caption placement")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="print a project's caption placement")
    s.add_argument("project_id")
    d = sub.add_parser("derive", help="(re)derive from the subtitle-removal lineage; "
                       "scans the source inline (streamed) unless --enqueue")
    d.add_argument("project_id")
    d.add_argument("--enqueue", action="store_true",
                   help="write the provisional value and enqueue the scan instead")
    args = ap.parse_args(argv)

    from .db import get_session
    from .models import VideoProject

    try:
        if args.cmd == "show":
            with get_session() as db:
                project = db.get(VideoProject, args.project_id)
                if project is None:
                    raise PlacementError(f"project {args.project_id} not found")
                out = project.caption_placement
        elif args.enqueue:
            with get_session() as db:
                project = db.get(VideoProject, args.project_id)
                if project is None:
                    raise PlacementError(f"project {args.project_id} not found")
                rev = request_derivation(db, project)
            if not enqueue_scan(args.project_id, rev):
                raise PlacementError("scan not enqueued (Redis)")
            out = {"enqueued": True, "rev": rev}
        else:
            out = derive_now(args.project_id)
    except PlacementError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
