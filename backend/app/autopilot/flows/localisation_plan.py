"""Pure pieces of the localisation autopilot flow (AICP-365, "C1").

The localisation flow (docs/assistant-flow-localisation.md steps 0-9, design in
docs/autopilot-localisation-flow.md) runs on the autopilot core (AICP-363,
docs/autopilot-design.md). This module holds everything the flow decides that
does NOT need the core: it reads no database, writes no file, calls no
provider, and imports nothing from ``app.autopilot`` itself, so it can be
tested on its own and the flow's ``advance`` functions stay thin.

* :func:`validate_params` -- the ``autopilot_params`` given at project
  creation (languages in order, a CTA tail per language, model, resolution).
* :func:`check_partition` / :func:`plan_recut` -- the whole-video segment
  partition (AICP-340) and its re-cut at speech pauses (AICP-347; the pure
  function lives in ``app.speech_recut`` and is called through one boundary
  here, stubbed while that module is absent).
* :func:`pick_probe_segment` -- which segment is generated alone first.
* :func:`source_breakdown` -- per segment, from the ``transcribe/v3``
  transcript: who speaks, gender, on/off face, body actions, brands,
  overlapping device voices (operator decision Q2: the transcript is the
  source; no judgement call).
* :func:`build_language_prompts` -- the AICP-335 prompts for one language from
  the translated lines: the probe prompt (no voice reference) and the prompts of
  every other segment (voice reference = the probe's dialogue).
* :func:`estimate_language` / :func:`estimate_transcription` -- credits per
  language and for the transcription, from ``app.spend``'s rates.
* :func:`probe_run_spec` / :func:`rest_run_spec` -- the ``create_run``
  parameters of the probe run (probe segment only, no voice reference) and of
  the run for the remaining segments (stitch plan = TEMP library clip of the
  accepted probe in its place + every other segment; voice reference = the
  probe's own audio).
* :func:`apply_operator_cue` -- a rejected probe's comment turned into the
  corrected prompt of the regeneration it proposes.

Everything returned is plain JSON (step outputs and effect params are JSON).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from app import ai_models, languages, localisation, spend

FLOW_ID = "localisation"

# The model the flow generates with. Seedance 2.5 is the only registered model
# that takes a voice reference (``max_reference_audios``), which segments 2..N
# need; 720p is what project 46cc6443 was localised at (4 languages, accepted).
DEFAULT_MODEL = "seedance-2-5"
DEFAULT_RESOLUTION = "720p"
# Each generated clip carries its own speech (``Run.audio_mode="seedance"``).
AUDIO_MODE = "seedance"

# Intake warns above this: a long video rarely comes out clean first time, so
# the operator should expect regenerations (operator rule, flow journal step 3).
LONG_VIDEO_SEC = 60.0

# A probe segment needs at least this much speech, or the voice reference cut
# from it is too short for Seedance (``ai_models.VOICE_REF_MIN_SEC`` = 2 s) and
# the probe proves nothing about voices.
PROBE_MIN_SPEECH_SEC = max(2.0, float(ai_models.VOICE_REF_MIN_SEC))

# The localisation-prompt admission reserves three LLM calls (a translation, a
# 1:1 re-ask, a shorten pass: api_v2.build_localisation_prompt); estimate the
# same so the budget line matches what the ledger will be charged.
TRANSLATE_LLM_CALLS = 3

# Partition checks tolerate float noise of this size.
_EPS = 1e-3

_PARAM_KEYS = {"languages", "tails", "model", "resolution"}


try:  # the core's error type when the core is present (it maps it to a 422)
    from app.autopilot.contract import FlowParamsError
except ImportError:  # pragma: no cover — the pure module stays importable alone
    class FlowParamsError(ValueError):
        """``autopilot_params`` are invalid (the core maps it to a 422)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _get(obj, name: str, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _f(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def segment_dicts(segments: Iterable) -> list[dict]:
    """SegmentDef rows (or dicts) as plain dicts, ordered by index."""
    out = []
    for seg in segments or []:
        out.append({
            "id": str(_get(seg, "id")),
            "index": int(_get(seg, "index", 0) or 0),
            "start_sec": _f(_get(seg, "start_sec")),
            "end_sec": _f(_get(seg, "end_sec")),
            "action": str(_get(seg, "action", "swap") or "swap"),
            "pre_roll_sec": _f(_get(seg, "pre_roll_sec")),
            "post_roll_sec": _f(_get(seg, "post_roll_sec")),
        })
    out.sort(key=lambda s: (s["index"], s["start_sec"]))
    return out


def clip_sec(seg: Mapping) -> float:
    """Length of the clip a swap segment uploads (mirrors spend._clip_sec)."""
    return (
        seg["end_sec"] - seg["start_sec"]
        + max(0.0, seg.get("pre_roll_sec") or 0.0)
        + max(0.0, seg.get("post_roll_sec") or 0.0)
    )


def json_digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. Params (Q4: languages and CTA tails at creation)
# ---------------------------------------------------------------------------


def validate_params(
    raw: Any,
    *,
    project_type: Optional[str] = "localisation",
    clip_lookup: Optional[Callable[[str], Any]] = None,
) -> dict:
    """Validate and normalise the flow's ``autopilot_params``.

    Shape::

        {"languages": ["es", "pt"],          # order = order of generation
         "tails": {"es": "<library clip id>", "pt": null},   # CTA tail or none
         "model": "seedance-2-5",            # optional
         "resolution": "720p"}               # optional

    *clip_lookup* resolves a library clip id to its row (``None`` = unknown);
    without it the tail ids are only checked for shape. A tail tagged with a
    language must match the language it is the tail of; an untagged clip is a
    language-agnostic tail and is accepted. Tails are consumed by the
    finishing flow (AICP-366); they are validated here because they are given
    at creation.

    The source language is not known before transcription, so "target equals
    source" is checked later (:func:`check_languages_against_source`).

    Raises :class:`FlowParamsError` with an operator-readable message.
    """
    if project_type is not None and project_type != "localisation":
        raise FlowParamsError(
            f"the localisation flow runs on localisation projects, not {project_type!r}"
        )
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise FlowParamsError("autopilot_params must be a JSON object")
    unknown = sorted(set(raw) - _PARAM_KEYS)
    if unknown:
        raise FlowParamsError(
            f"unknown autopilot_params field(s): {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(_PARAM_KEYS))})"
        )

    langs_raw = raw.get("languages")
    if isinstance(langs_raw, str):
        langs_raw = [part for part in langs_raw.split(",")]
    if not isinstance(langs_raw, (list, tuple)) or not langs_raw:
        raise FlowParamsError(
            "languages is required: a non-empty list of target language codes, "
            f"in the order to generate them (one of {sorted(languages.LANGUAGES)})"
        )
    langs: list[str] = []
    for item in langs_raw:
        code = languages.normalize(item if isinstance(item, str) else None)
        if code is None:
            raise FlowParamsError(
                f"unknown language {item!r}: must be one of {sorted(languages.LANGUAGES)}"
            )
        if code in langs:
            raise FlowParamsError(f"language {code!r} is listed twice")
        langs.append(code)

    tails_raw = raw.get("tails") or {}
    if not isinstance(tails_raw, Mapping):
        raise FlowParamsError("tails must be an object {language: library clip id or null}")
    tails: dict[str, Optional[str]] = {code: None for code in langs}
    for key, value in tails_raw.items():
        code = languages.normalize(key if isinstance(key, str) else None)
        if code is None or code not in langs:
            raise FlowParamsError(
                f"tails has an entry for {key!r}, which is not one of the languages {langs}"
            )
        if value is None or (isinstance(value, str) and not value.strip()):
            tails[code] = None
            continue
        if not isinstance(value, str):
            raise FlowParamsError(f"tails[{code!r}] must be a library clip id or null")
        clip_id = value.strip()
        if clip_lookup is not None:
            clip = clip_lookup(clip_id)
            if clip is None:
                raise FlowParamsError(f"tails[{code!r}]: unknown library clip {clip_id!r}")
            clip_language = languages.normalize(_get(clip, "language"))
            if clip_language and clip_language != code:
                raise FlowParamsError(
                    f"tails[{code!r}]: library clip {clip_id!r} is tagged "
                    f"{clip_language!r}, not {code!r}"
                )
        tails[code] = clip_id

    model = raw.get("model") or DEFAULT_MODEL
    if model not in ai_models.V2V_MODELS:
        raise FlowParamsError(
            f"unknown model {model!r}: must be one of {sorted(ai_models.V2V_MODELS)}"
        )
    spec = ai_models.spec_for(model)
    if not spec.produces_audio:
        raise FlowParamsError(f"model {model!r} produces no audio and cannot localise")
    if not ai_models.supports_reference_audio(model):
        raise FlowParamsError(
            f"model {model!r} takes no voice reference, which the flow needs for "
            "the segments after the probe; use one of "
            f"{sorted(ai_models.models_supporting_reference_audio())}"
        )
    resolution = raw.get("resolution") or DEFAULT_RESOLUTION
    allowed = ai_models.resolutions_for(model)
    if resolution not in allowed:
        raise FlowParamsError(
            f"resolution {resolution!r} not allowed for {model!r}; one of {sorted(allowed)}"
        )
    return {"languages": langs, "tails": tails, "model": model, "resolution": resolution}


def check_languages_against_source(params: Mapping, source_language: Optional[str]) -> list[str]:
    """Problems with the target languages once the source language is known.

    Returns operator-readable messages (empty = fine). A target equal to the
    source would spend a generation re-speaking the original.
    """
    source = languages.normalize(source_language) if source_language else None
    if not source:
        return []
    return [
        f"target language {code!r} is the source language of the video"
        for code in params.get("languages") or []
        if code == source
    ]


# ---------------------------------------------------------------------------
# 2. Intake
# ---------------------------------------------------------------------------


@dataclass
class IntakeReport:
    """What intake found; ``blockers`` stop the flow, ``warnings`` are journaled."""

    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    needs_mute_source: bool = False
    transcript: str = "missing"   # missing | pending | failed | stale | ready

    def to_dict(self) -> dict:
        return {
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "needs_mute_source": self.needs_mute_source,
            "transcript": self.transcript,
        }


def transcript_state(transcript_status: Optional[str], transcript: Optional[dict]) -> str:
    """``ready`` only for a ready ``transcribe/v3`` (schema 3) transcript.

    A ready v1/v2 transcript is ``stale``: it lacks the speakers, genders,
    per-line face visibility and actions the prompts need (Q2), so the flow
    re-transcribes it.
    """
    status = (transcript_status or "").strip()
    if status in ("pending", "running"):
        return "pending"
    if status == "failed":
        return "failed"
    if status == "empty":
        return "empty"
    if status != "ready" or not isinstance(transcript, dict):
        return "missing"
    try:
        version = int(transcript.get("schema_version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version < localisation.TRANSCRIPT_SCHEMA_VERSION:
        return "stale"
    return "ready"


def intake(project) -> IntakeReport:
    """Check a localisation project before anything is spent.

    *project* is a VideoProject (or an object with the same attributes).
    Whole-video scope is required (AICP-340): the flow re-speaks every segment.
    """
    report = IntakeReport()
    if _get(project, "project_type") != "localisation":
        report.blockers.append("not a localisation project")
    if _get(project, "localisation_scope") != "whole":
        report.blockers.append(
            "the project is analysed in hook scope; the autopilot localises the "
            "whole video: re-analyze it with localisation_scope=whole"
        )
    if not _get(project, "mute_source", False):
        report.needs_mute_source = True
    duration = _f(_get(project, "duration_sec"))
    if duration > LONG_VIDEO_SEC:
        report.warnings.append(
            f"the video is {duration:.0f} s long (> {LONG_VIDEO_SEC:.0f} s): "
            "expect some segments to need a regeneration"
        )
    report.transcript = transcript_state(
        _get(project, "transcript_status"), _get(project, "transcript")
    )
    if report.transcript == "empty":
        report.blockers.append("the transcript has no speech: nothing to localise")
    return report


# ---------------------------------------------------------------------------
# 3. Segments: partition check + re-cut boundary (AICP-340 / AICP-347)
# ---------------------------------------------------------------------------


def whole_video_target_sec(max_segment_sec: Optional[float]) -> float:
    from app.pipeline_v2 import whole_video_target_sec as _target

    return float(_target(max_segment_sec))


def whole_video_floor_sec() -> float:
    """Shortest swap clip any audio model accepts (AICP-347's floor)."""
    try:
        from app.pipeline_v2 import whole_video_floor_sec as _floor  # AICP-347
    except ImportError:
        floors = [
            spec.min_clip_sec for spec in ai_models.V2V_MODELS.values() if spec.produces_audio
        ]
        return float(max(floors or [0.0]))
    return float(_floor())


@dataclass
class PartitionCheck:
    ok: bool
    problems: list = field(default_factory=list)
    over_target: list = field(default_factory=list)   # segment ids
    keep_segments: list = field(default_factory=list)  # segment ids

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "over_target": list(self.over_target),
            "keep_segments": list(self.keep_segments),
        }


def check_partition(
    segments: Sequence, *, duration_sec: float, target_sec: float
) -> PartitionCheck:
    """Is *segments* a usable whole-video partition?

    Contiguous from 0 to the probed duration, the first segment a swap, every
    swap clip (with its rolls) within *target_sec* (14.72 s: Seedance 2.5's
    intermittent truncation length). Keep segments are allowed (their source
    plays as is) and listed, since their lines are not re-spoken.
    """
    segs = segment_dicts(segments)
    problems: list[str] = []
    over: list[str] = []
    keeps: list[str] = []
    if not segs:
        return PartitionCheck(ok=False, problems=["the project has no segments"])
    if abs(segs[0]["start_sec"]) > _EPS:
        problems.append(f"the first segment starts at {segs[0]['start_sec']:.2f} s, not 0")
    for prev, cur in zip(segs, segs[1:]):
        if abs(cur["start_sec"] - prev["end_sec"]) > _EPS:
            problems.append(
                f"segments {prev['index'] + 1} and {cur['index'] + 1} are not contiguous "
                f"({prev['end_sec']:.2f} s vs {cur['start_sec']:.2f} s)"
            )
    if duration_sec and abs(segs[-1]["end_sec"] - float(duration_sec)) > 0.05:
        problems.append(
            f"the last segment ends at {segs[-1]['end_sec']:.2f} s, the video at "
            f"{float(duration_sec):.2f} s"
        )
    for seg in segs:
        if seg["action"] != "swap":
            keeps.append(seg["id"])
            continue
        if clip_sec(seg) > target_sec + _EPS:
            over.append(seg["id"])
            problems.append(
                f"segment {seg['index'] + 1} is {clip_sec(seg):.2f} s, over the "
                f"{target_sec:.2f} s whole-video target"
            )
    if segs[0]["action"] != "swap":
        problems.append("the first segment is keep: there is nothing to probe")
    return PartitionCheck(ok=not problems, problems=problems, over_target=over, keep_segments=keeps)


@dataclass
class RecutResult:
    """The re-cut plan, or the stub's "nothing moved" when AICP-347 is absent."""

    available: bool
    feasible: bool
    changed: bool
    ranges: list
    warnings: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "feasible": self.feasible,
            "changed": self.changed,
            "ranges": [[_round(s), _round(e)] for s, e in self.ranges],
            "warnings": list(self.warnings),
            "boundaries": list(self.boundaries),
        }


def _speech_recut():
    """``app.speech_recut.recut_at_speech_pauses`` (AICP-347) or None."""
    try:
        from app.speech_recut import recut_at_speech_pauses
    except ImportError:
        return None
    return recut_at_speech_pauses


def plan_recut(
    segments: Sequence,
    lines: Iterable,
    *,
    duration_sec: float,
    target_sec: float,
    floor_sec: Optional[float] = None,
    scene_cuts: Optional[Iterable[float]] = None,
    recut: Optional[Callable[..., Any]] = None,
) -> RecutResult:
    """The re-cut of *segments* at speech pauses, through one call boundary.

    *recut* defaults to AICP-347's pure ``recut_at_speech_pauses``; while that
    module is absent the stub keeps every boundary and says so. The flow only
    uses this to decide whether a re-cut would change anything and to journal
    it; the APPLY goes through the ``recut_at_pauses`` action handler, which
    recomputes the plan under the project lock (``expected_fingerprint``).
    """
    segs = segment_dicts(segments)
    fn = recut or _speech_recut()
    ranges = [(s["start_sec"], s["end_sec"]) for s in segs]
    if fn is None:
        return RecutResult(
            available=False, feasible=True, changed=False, ranges=ranges,
            warnings=[
                "re-cut at speech pauses (AICP-347) is not available in this "
                "build: segment boundaries stay where analysis put them"
            ],
        )
    plan = fn(
        segs, list(lines or []), duration_sec=float(duration_sec),
        target_sec=float(target_sec),
        floor_sec=float(whole_video_floor_sec() if floor_sec is None else floor_sec),
        scene_cuts=list(scene_cuts) if scene_cuts else None,
    )
    as_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
    return RecutResult(
        available=True,
        feasible=bool(as_dict.get("feasible", True)),
        changed=bool(as_dict.get("changed", False)),
        ranges=[(float(s), float(e)) for s, e in as_dict.get("ranges") or ranges],
        warnings=list(as_dict.get("warnings") or []),
        boundaries=list(as_dict.get("boundaries") or []),
    )


# ---------------------------------------------------------------------------
# 4. Line assignment + source breakdown (Q2: from transcribe/v3)
# ---------------------------------------------------------------------------


def _line_midpoint(line: Mapping) -> float:
    start, end = _f(line.get("start")), _f(line.get("end"))
    return start if end <= start else (start + end) / 2.0


def assign_lines(lines: Iterable, segments: Sequence) -> tuple[dict[str, list], list]:
    """Every line to exactly one segment by its midpoint; ``(by_segment, dropped)``.

    The rule of ``api_v2._assign_lines_to_segments`` (the translate job's
    per-segment split), restated here so the flow stays importable without the
    API module; a test pins the two together. Only SWAP segments receive lines
    (a keep segment is never generated), so a line in a keep segment is dropped.
    """
    segs = [s for s in segment_dicts(segments) if s["action"] == "swap"]
    by_segment: dict[str, list] = {s["id"]: [] for s in segs}
    dropped: list = []
    for line in lines or []:
        if not isinstance(line, Mapping):
            continue
        mid = _line_midpoint(line)
        for seg in segs:
            if seg["start_sec"] <= mid < seg["end_sec"]:
                by_segment[seg["id"]].append(line)
                break
        else:
            dropped.append(line)
    return by_segment, dropped


def _clip_window(seg: Mapping, duration_sec: Optional[float], max_clip_sec: Optional[float]):
    return localisation.clip_window(
        seg["start_sec"], seg["end_sec"], seg.get("pre_roll_sec") or 0.0,
        seg.get("post_roll_sec") or 0.0, duration_sec, max_clip_sec,
    )


def _speech_sec(lines: Iterable[Mapping]) -> float:
    """Seconds covered by the union of the lines' spans."""
    spans = sorted(
        (max(0.0, _f(l.get("start"))), max(0.0, _f(l.get("end")))) for l in lines
    )
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def transcript_digest(transcript: Optional[dict]) -> str:
    """Hash of every transcript field the prompts read."""
    tr = transcript if isinstance(transcript, dict) else {}
    return json_digest({
        k: tr.get(k)
        for k in ("schema_version", "source_language", "lines", "speakers",
                  "actions", "brands", "on_screen_text")
    })


def source_breakdown(
    transcript: dict,
    segments: Sequence,
    *,
    duration_sec: Optional[float] = None,
    max_clip_sec: Optional[float] = None,
) -> dict:
    """Per-segment breakdown of the source from the ``transcribe/v3`` transcript.

    For each swap segment: its clip window, its lines (speaker tag, kind,
    gender, on/off face and what the frame shows instead), the persons present
    and how many lines each secondary person says, the body actions inside the
    window (relative seconds), the brands spoken, and the device lines that
    play UNDER other speakers (kept in the prompt as BACKGROUND, operator
    decision of 2026-10-29). Globally: the speaker profiles (as the prompt
    builder derives them), lines no segment says, lines straddling a cut.

    Speakers, genders, face visibility and actions come from the transcript
    only (operator decision Q2); nothing here looks at frames.
    """
    tr = transcript if isinstance(transcript, dict) else {}
    lines = [ln for ln in tr.get("lines") or [] if isinstance(ln, dict)]
    segs = segment_dicts(segments)
    meta_lines = localisation._with_line_meta(lines, tr)
    profiles = localisation._speaker_profiles(meta_lines, tr) if meta_lines else {}
    background_ids = {
        ln.get("id") for ln in localisation.background_lines(meta_lines, tr)
    }
    by_segment, dropped = assign_lines(meta_lines, segs)
    actions = localisation.clean_actions(tr.get("actions"))
    brands = [b["name"] for b in localisation.clean_brands(tr.get("brands"))]

    def tag_of(line):
        prof = profiles.get(localisation._speaker_key(line))
        return prof["tag"] if prof else "SPEAKER"

    out_segments = []
    straddling = []
    for seg in segs:
        if seg["action"] != "swap":
            out_segments.append({
                "id": seg["id"], "index": seg["index"], "action": seg["action"],
                "start_sec": _round(seg["start_sec"]), "end_sec": _round(seg["end_sec"]),
                "lines": [], "note": "keep segment: plays the source, nothing is re-spoken",
            })
            continue
        w0, w1 = _clip_window(seg, duration_sec, max_clip_sec)
        mine = by_segment.get(seg["id"], [])
        rows = []
        for line in mine:
            prof = profiles.get(localisation._speaker_key(line), {})
            start, end = _f(line.get("start")), _f(line.get("end"))
            visible = line.get("face_visible")
            if not isinstance(visible, bool):
                visible = None
            # The prompt builder's rule (localisation._off_face_cue): a PERSON
            # seen on camera elsewhere whose face is out of frame on this line.
            # An off-screen speaker (the interviewer) is never "off face".
            off_face = (
                prof.get("kind") == "person" and bool(prof.get("on_camera"))
                and not (visible if visible is not None else bool(line.get("on_screen")))
            )
            rows.append({
                "id": line.get("id"),
                "start_sec": _round(start, 2), "end_sec": _round(end, 2),
                "rel_start_sec": _round(max(0.0, start - w0), 2),
                "rel_end_sec": _round(min(w1 - w0, max(0.0, end - w0)), 2),
                "speaker": tag_of(line),
                "kind": prof.get("kind", "person"),
                "gender": prof.get("gender", "unknown"),
                "on_screen": bool(line.get("on_screen")),
                "face_visible": visible,
                "off_face": off_face,
                "frame": (line.get("frame") or "").strip() if off_face else "",
                "background": line.get("id") in background_ids,
            })
            if start < seg["start_sec"] - 0.15 or end > seg["end_sec"] + 0.15:
                straddling.append(line.get("id"))
        counts = Counter(r["speaker"] for r in rows if not r["background"])
        persons = [
            {"tag": p["tag"], "gender": p.get("gender", "unknown"),
             "primary": bool(p.get("primary")), "lines": counts.get(p["tag"], 0)}
            for p in sorted(
                (p for p in profiles.values() if p["kind"] == "person"),
                key=lambda p: (not p.get("primary"), p["order"]),
            )
        ]
        seg_actions = []
        for act in actions:
            a0, a1 = act["start"] - w0, act["end"] - w0
            if a1 < 0 or a0 >= (w1 - w0):
                continue
            seg_actions.append({
                "rel_start_sec": _round(max(0.0, a0), 2),
                "rel_end_sec": _round(min(w1 - w0, a1), 2),
                "speaker": act["speaker"], "action": act["action"],
                "do_not": act["do_not"],
            })
        out_segments.append({
            "id": seg["id"], "index": seg["index"], "action": "swap",
            "start_sec": _round(seg["start_sec"]), "end_sec": _round(seg["end_sec"]),
            "clip_window": [_round(w0), _round(w1)],
            "speech_sec": _round(_speech_sec(mine), 2),
            "lines": rows,
            "persons": persons,
            "off_face_line_ids": [r["id"] for r in rows if r["off_face"]],
            "background_line_ids": [r["id"] for r in rows if r["background"]],
            "actions": seg_actions,
            "brands": [b for b in brands if localisation._mentions(b, mine)],
        })

    warnings = []
    version = tr.get("schema_version")
    if not isinstance(version, int) or version < localisation.TRANSCRIPT_SCHEMA_VERSION:
        warnings.append(
            f"transcript schema {version!r} is older than v{localisation.TRANSCRIPT_SCHEMA_VERSION}: "
            "no speaker genders, face visibility or actions"
        )
    for line in dropped:
        warnings.append(
            f"line {line.get('id')} ({_f(line.get('start')):.1f}-{_f(line.get('end')):.1f} s) "
            "is said by no swap segment and will not be re-spoken"
        )
    for lid in straddling:
        warnings.append(f"line {lid} crosses a segment boundary and will be cut at the seam")
    for lid in sorted(background_ids, key=lambda v: (str(type(v)), v)):
        warnings.append(
            f"line {lid} is a device voice playing under other speakers: kept in the "
            "prompt as BACKGROUND (the model may not voice it)"
        )
    speakers = [
        {"label": p["label"], "tag": p["tag"], "kind": p["kind"],
         "gender": p.get("gender", "unknown"), "primary": bool(p.get("primary")),
         "on_camera": bool(p.get("on_camera"))}
        for p in sorted(profiles.values(), key=lambda p: p["order"])
    ]
    return {
        "transcript_digest": transcript_digest(tr),
        "schema_version": version,
        "source_language": tr.get("source_language") or "",
        "speakers": speakers,
        "segments": out_segments,
        "dropped_line_ids": [ln.get("id") for ln in dropped],
        "straddling_line_ids": straddling,
        "background_line_ids": sorted(background_ids, key=lambda v: (str(type(v)), v)),
        "warnings": warnings,
    }


def pick_probe_segment(segments: Sequence, lines: Iterable) -> Optional[dict]:
    """The segment generated alone first: the first swap segment with speech.

    "Segment 1" in the flow doc. A segment whose lines cover less than
    :data:`PROBE_MIN_SPEECH_SEC` is skipped (a silent intro proves nothing
    about voices and yields no usable voice reference). None when no segment
    qualifies.
    """
    segs = segment_dicts(segments)
    by_segment, _ = assign_lines(lines, segs)
    for seg in segs:
        if seg["action"] != "swap":
            continue
        if _speech_sec(by_segment.get(seg["id"], [])) >= PROBE_MIN_SPEECH_SEC - _EPS:
            return seg
    return None


# ---------------------------------------------------------------------------
# 5. Prompts (AICP-335) for one language
# ---------------------------------------------------------------------------


def build_language_prompts(
    *,
    translated_lines: Sequence[dict],
    transcript: dict,
    segments: Sequence,
    probe_segment_id: str,
    source_language: str,
    target_language: str,
    duration_sec: Optional[float],
    max_clip_sec: Optional[float] = None,
) -> dict:
    """The run prompts of one language, built from the translation's lines.

    Same construction as the translate job (``api_v2._compute_localisation_prompt``):
    lines partitioned by midpoint, each segment timed against its own clip
    window, ``all_lines`` = every translated line, the speech-only template
    (``swap_character=False``). Two differences, both from the flow:

    * segments other than the probe say "the reference audio contains ..." in
      their Speakers header: their run carries the probe's dialogue as a voice
      reference; the probe has none, so its header does not claim one;
    * device lines under other speakers stay in, marked BACKGROUND (the
      prompt builder's rule since 2026-10-29).

    Returns ``{"probe_prompt", "segment_prompts": {seg_id: prompt},
    "run_prompt", "order", "dropped_line_ids", "warnings"}`` where
    ``segment_prompts`` holds every swap segment EXCEPT the probe.
    """
    segs = segment_dicts(segments)
    swaps = [s for s in segs if s["action"] == "swap"]
    if probe_segment_id not in {s["id"] for s in swaps}:
        raise ValueError(f"probe segment {probe_segment_id!r} is not a swap segment")
    lines = [dict(ln) for ln in translated_lines or [] if isinstance(ln, dict)]
    by_segment, dropped = assign_lines(lines, swaps)

    def build(seg, reference_audio: bool) -> str:
        return localisation.build_prompt(
            lines=by_segment[seg["id"]],
            source_language=source_language,
            target_language=target_language,
            swap_character=False,
            window=_clip_window(seg, duration_sec, max_clip_sec),
            transcript=transcript,
            all_lines=lines,
            reference_audio=reference_audio,
        )

    probe = next(s for s in swaps if s["id"] == probe_segment_id)
    segment_prompts = {s["id"]: build(s, True) for s in swaps if s["id"] != probe_segment_id}
    end = float(duration_sec) if duration_sec else (segs[-1]["end_sec"] if segs else 0.0)
    run_prompt = localisation.build_prompt(
        lines=lines, source_language=source_language, target_language=target_language,
        swap_character=False, window=(0.0, end), transcript=transcript,
        all_lines=lines, reference_audio=True,
    )
    warnings = [
        f"translated line {ln.get('id')} falls in no swap segment and is not said"
        for ln in dropped
    ]
    for seg in swaps:
        if not by_segment[seg["id"]]:
            warnings.append(f"segment {seg['index'] + 1} has no dialogue: it is generated silent")
    return {
        "probe_segment_id": probe_segment_id,
        "probe_prompt": build(probe, False),
        "segment_prompts": segment_prompts,
        "run_prompt": run_prompt,
        "order": [s["id"] for s in swaps],
        "dropped_line_ids": [ln.get("id") for ln in dropped],
        "warnings": warnings,
    }


_CUE_HEADER = "Corrections from the operator's listening check — follow them exactly:"


def apply_operator_cue(prompt: str, comment: str) -> str:
    """The probe prompt with the operator's rejection comment as a cue.

    A rejected probe (listening approval) comes back with a comment ("the
    interviewer must be a woman", "she says 'Listenify', not 'Listenifyed'").
    The flow cannot rewrite the prompt by judgement (the judgement port is
    disabled, Q2), so the comment is appended verbatim under one fixed header;
    a second rejection appends a second bullet. The regeneration approval
    shows the resulting prompt before anything is billed.
    """
    text = " ".join(str(comment or "").split())
    if not text:
        raise ValueError("an operator cue needs a comment")
    body = prompt.rstrip()
    bullet = f"- {text}"
    if _CUE_HEADER in body:
        return body + "\n" + bullet
    return body + "\n\n" + _CUE_HEADER + "\n" + bullet


# ---------------------------------------------------------------------------
# 6. Estimates (credits, app.spend rates)
# ---------------------------------------------------------------------------


def estimate_transcription() -> float:
    """One transcription call (what POST /transcribe reserves)."""
    return float(spend.llm_credits())


def estimate_language(
    segments: Sequence,
    *,
    probe_segment_id: Optional[str],
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
) -> dict:
    """Credits one language should cost if no segment is regenerated.

    ``translate`` (the reservation of the translate admission), ``probe``
    (the probe segment's clip) and ``rest`` (every other swap segment), each at
    ``spend.video_credits`` -- the rate the run admission reserves, so the
    budget line equals the reservation. 46cc6443 (79 s, 7 segments, 720p):
    probe 1 040 + rest 5 520; real spend was 912 + 4 940..5 016.
    """
    segs = [s for s in segment_dicts(segments) if s["action"] == "swap"]
    probe = 0.0
    rest = 0.0
    for seg in segs:
        credits = spend.video_credits(model, resolution, clip_sec(seg))
        if probe_segment_id is not None and seg["id"] == probe_segment_id:
            probe += credits
        else:
            rest += credits
    if probe_segment_id is None and segs:
        # Before the transcript exists the probe is assumed to be segment 1.
        first = spend.video_credits(model, resolution, clip_sec(segs[0]))
        probe, rest = first, rest - first
    translate = TRANSLATE_LLM_CALLS * float(spend.llm_credits())
    return {
        "translate": translate,
        "probe": probe,
        "rest": rest,
        "total": translate + probe + rest,
    }


def estimate_probe_regeneration(
    segments: Sequence, *, probe_segment_id: str, model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
) -> float:
    seg = next(s for s in segment_dicts(segments) if s["id"] == probe_segment_id)
    return spend.video_credits(model, resolution, clip_sec(seg))


# ---------------------------------------------------------------------------
# 7. Run specs (create_run parameters)
# ---------------------------------------------------------------------------


def _short(project_id: str) -> str:
    return str(project_id)[:8]


def probe_run_spec(
    *, project_id: str, language: str, prompts: Mapping, model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
) -> dict:
    """``create_run`` parameters of the probe: the probe segment ONLY, no voice.

    The stitch plan names only the probe segment, so no other swap segment is
    submitted (``pipeline_v2._active_swap_segment_ids``). The probe segment's
    own prompt is both the run prompt and its override (the run prompt is what
    the operator sees on the run page).
    """
    seg_id = prompts["probe_segment_id"]
    return {
        "name": f"Autopilot {language} probe",
        "prompt": prompts["probe_prompt"],
        "model": model,
        "resolution": resolution,
        "audio_mode": AUDIO_MODE,
        "language": language,
        "segment_prompts": {seg_id: prompts["probe_prompt"]},
        "stitch_plan": [{"kind": "segment", "segment_def_id": seg_id}],
        "voice_from_library_clip": None,
    }


def temp_clip_spec(*, project_id: str, language: str, probe_run_id: str) -> dict:
    """Parameters of the TEMP library clip made from the accepted probe.

    Untagged (``language=None``) so the New Run form never auto-picks it as a
    tail; the name says TEMP, the project and the language. Its source is the
    probe run's delivered final (the file the operator listened to), resolved
    by the handler under the probe run's lock.
    """
    return {
        "name": f"TEMP autopilot {_short(project_id)} {language} probe",
        "language": None,
        "from_run": probe_run_id,
        "file": "final",
    }


def rest_run_spec(
    *, project_id: str, language: str, prompts: Mapping, segments: Sequence,
    probe_run_id: str, temp_clip_id: str, model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
) -> dict:
    """``create_run`` parameters of the run that generates every other segment.

    Stitch plan, in segment order: the TEMP library clip of the accepted probe
    in the probe segment's place, then every other segment (keep segments
    included as their source). Voice reference: the audio of that same TEMP
    clip -- the accepted probe's full dialogue, both speakers ("voice B" of
    the 46cc6443 journal). Taking it from the clip, an immutable copy made
    under the probe run's lock, rather than from the probe run's mutable
    final means the reused picture and the voice can never come from two
    different takes (Terra §2 round 1, finding 12). The probe segment is not
    in the plan as a segment, so it is never regenerated.
    """
    probe_id = prompts["probe_segment_id"]
    plan = []
    for seg in segment_dicts(segments):
        if seg["id"] == probe_id:
            plan.append({"kind": "library", "clip_id": temp_clip_id})
        else:
            plan.append({"kind": "segment", "segment_def_id": seg["id"]})
    return {
        "name": f"Autopilot {language}",
        "prompt": prompts["run_prompt"],
        "model": model,
        "resolution": resolution,
        "audio_mode": AUDIO_MODE,
        "language": language,
        "segment_prompts": dict(prompts["segment_prompts"]),
        "stitch_plan": plan,
        "voice_from_library_clip": temp_clip_id,
        "probe_run_id": probe_run_id,
    }
