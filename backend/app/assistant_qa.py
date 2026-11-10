"""The AICP-343 QA tools behind the assistant's closed tool boundary (AICP-345).

:mod:`app.qa_tools` is the algorithm; this module only swaps its media
backend for the mandatory confined launcher and turns its results into
bounded, path-free evidence. No model, provider, database write or network.

* Every decode goes through :func:`assistant_media._stream` (one continuous
  ffmpeg filter, so frame sampling is the CLI's exactly) and every probe
  through :func:`assistant_media.probe_trusted_media` (the strict normalized
  profile; inputs outside it are refused, never re-encoded).
* Single-frame grabs decode to raw BGR instead of PNG, so no compressed image
  is decoded in the supervisor process.
* Results are reduced to the AICP-345 evidence models: verdicts, counts,
  bounded spans/outliers/samples, and a sha256 of the full detail so a later
  audit can prove which numbers the summary came from.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import assistant_media as media
from . import qa_tools
from .media import MediaInfo

MAX_DURATION_SEC = 120
MAX_LONG_EDGE = 1920
MAX_PIXELS = 2_073_600
# Records per decode: the 120 s ceiling at the highest sampling rate the tools
# use (30 fps alignment), plus slack for the fps filter's rounding at the end.
MAX_STREAM_RECORDS = MAX_DURATION_SEC * 30 + 64
MAX_SPANS = 24
MAX_OUTLIERS = 24
MAX_ALIGN_WINDOWS = 16
MAX_MATCH_SAMPLES = 40
MAX_COMPARE_SAMPLES = 48


class QAInputUnavailable(media.AssistantMediaError):
    """The input is outside the QA tools' resource profile; safe to report."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class ConfinedQAMedia:
    """:class:`qa_tools.LocalMedia`'s interface over the confined launcher.

    *remaining* returns the seconds left for the current operation and
    *check* raises on cancellation/deadline; both come from the agent media
    service so the QA work shares its per-operation and per-turn budgets.
    """

    def __init__(self, *, remaining, check):
        self.remaining, self.check = remaining, check
        self._probes: dict[str, MediaInfo] = {}

    def _budget(self):
        return media._budget(self.remaining(), self.check)

    def probe(self, path: str) -> MediaInfo:
        if path not in self._probes:
            info = media.probe_trusted_media(path, timeout_sec=self.remaining(), cancelled=self.check)
            if (info.video_duration_sec > MAX_DURATION_SEC or max(info.width, info.height) > MAX_LONG_EDGE
                    or info.width * info.height > MAX_PIXELS):
                raise QAInputUnavailable("QA input exceeds the inspection resource limits")
            # app.media.probe reports the CONTAINER duration; so does this.
            self._probes[path] = MediaInfo(
                duration_sec=info.duration_sec, width=info.width, height=info.height,
                fps=float(Fraction(info.fps)), aspect_ratio="", has_audio=info.has_audio)
        return self._probes[path]

    def container_duration(self, path: str) -> float:
        """Container duration only (what app.media.probe reports), through the
        confined launcher but WITHOUT the strict editing profile: script
        captions need the length of hand-assembled or 24 fps pieces too."""
        raw = media._run([media.FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         self._budget(), inputs=(Path(path),))
        try:
            value = float(json.loads(raw)["format"]["duration"])
        except (KeyError, TypeError, ValueError):
            raise QAInputUnavailable("a run file has no readable duration") from None
        if not 0 < value <= 600:
            raise QAInputUnavailable("a run file has no usable duration")
        return value

    def frames(self, path, *, fps, width, height, pix_fmt="rgb24", start=None, duration=None):
        self.probe(path)
        channels = qa_tools._CHANNELS[pix_fmt]
        shape = (height, width) if channels == 1 else (height, width, channels)
        argv = qa_tools._decode_argv(path, fps=fps, width=width, height=height, pix_fmt=pix_fmt,
                                     start=start, duration=duration)
        for record in media._stream(argv, self._budget(), inputs=(Path(path),),
                                    chunk_bytes=width * height * channels, max_chunks=MAX_STREAM_RECORDS):
            yield np.frombuffer(record, np.uint8).reshape(shape)

    def grab(self, path, t):
        info = self.probe(path)
        argv = [media.FFMPEG, "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", path,
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        records = list(media._stream(argv, self._budget(), inputs=(Path(path),),
                                     chunk_bytes=info.width * info.height * 3, max_chunks=1))
        if not records:
            return None
        return np.frombuffer(records[0], np.uint8).reshape((info.height, info.width, 3))


def _r(value, places=3):
    return round(float(value), places)


def caption_evidence(scan: qa_tools.CaptionScan, *, target_ref, calibration_ref, start_sec, duration_sec,
                     calibration_start_sec, calibration_duration_sec, project_config: bool) -> dict:
    calibration = scan.calibration
    geometry = calibration.get("geometry")
    return {
        "tool": "detect_captions", "target": target_ref,
        "start_sec": start_sec, "duration_sec": duration_sec,
        "config_sha256": digest(scan.config), "project_config": project_config,
        "fps": float(scan.fps), "threshold": scan.threshold, "frames": scan.frames,
        "hits": scan.hits, "hit_ratio": float(scan.hit_ratio), "verdict": scan.verdict,
        "spans": [{"start_sec": _r(sp.start), "end_sec": _r(sp.end), "frames": sp.frames,
                   "max_score": sp.max_score} for sp in scan.spans[:MAX_SPANS]],
        "spans_total": len(scan.spans),
        "scores_sha256": digest([[f.t, f.score] for f in scan.scores]),
        "calibration": {
            "source": calibration_ref, "start_sec": calibration_start_sec,
            "duration_sec": calibration_duration_sec, "frames": calibration["frames"],
            "hits": calibration["hits"], "hit_ratio": float(calibration["hit_ratio"]),
            "calibrated": bool(calibration["calibrated"]),
            "geometry": None if geometry is None else {k: float(v) for k, v in geometry.items()},
        },
    }


def alignment_evidence(result: dict, *, source_ref, result_ref) -> dict:
    points = result["_points"]
    return {
        "tool": "check_alignment", "mode": "timeline", "source": source_ref, "result": result_ref,
        "source_frames": result["source_frames"], "result_frames": result["result_frames"],
        "points": result["points"], "nonzero_points": len(result["nonzero"]),
        "outliers": [{"t_sec": float(p["t"]), "offset_frames": p["offset"], "err": float(p["err"])}
                     for p in result["outliers"][:MAX_OUTLIERS]],
        "outliers_total": len(result["outliers"]),
        "at_search_limit": len(result["at_search_limit"]),
        "windows": [{"start_sec": float(w["start"]), "points": w["points"],
                     "median_offset": float(w["median_offset"])} for w in result["windows"][:MAX_ALIGN_WINDOWS]],
        "max_abs_window_median": float(result["max_abs_window_median"]),
        "length_ok": bool(result["length_ok"]), "in_sync": bool(result["in_sync"]),
        "tolerance_frames": result["tolerance_frames"],
        "detail_sha256": digest(points),
    }


def match_evidence(result: dict, *, source_ref, result_ref) -> dict:
    samples = result["samples"]
    if len(samples) > MAX_MATCH_SAMPLES:
        raise QAInputUnavailable("window alignment result is too long; use a clip of at most 20 s")
    return {
        "tool": "check_alignment", "mode": "window", "source": source_ref, "result": result_ref,
        "source_start_sec": float(result["source_start"]), "window_duration_sec": float(result["duration"]),
        "result_duration_sec": float(result["result_duration"]),
        "samples": [{"result_t_sec": float(s["result_t"]), "source_t_sec": float(s["source_t"]),
                     "drift_sec": float(s["drift"]), "err": float(s["err"])} for s in samples],
        "median_drift_sec": None if result["median_drift"] is None else float(result["median_drift"]),
        "max_abs_drift_sec": None if result["max_abs_drift"] is None else float(result["max_abs_drift"]),
        "detail_sha256": digest(samples),
    }


def compare_evidence(result: dict, *, source_ref, result_ref, time_map) -> dict:
    return {
        "tool": "compare_frames", "source": source_ref, "result": result_ref, "time_map": time_map,
        "samples": [{"window": s["window"], "source_t_sec": float(s["source_t"]),
                     "result_t_sec": float(s["result_t"]),
                     "diff": None if s["diff"] is None else float(s["diff"])} for s in result["samples"]],
    }
