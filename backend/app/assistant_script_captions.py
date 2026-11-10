"""Captions from the script for the assistant (AICP-345, flow 2 step 14).

A trusted, run-scoped resolver: the run is the pinned context's run, never a
model-named one, and every caption line comes from prompts stored in the
database (``RunSegment.prompt_override`` or ``Run.prompt``), never from model
text. The model may only name, for a library entry of the stitch plan, which
RunSegment of the same project it stands for -- and that binding is checked
against the run's frozen segment snapshot and, while the library clip still
exists, its bytes.

What this proves and what it does not (see docs/assistant-qa-tools.md):

* the assistant's selected version is byte-identical to the run's current
  result, and every piece is a completed, generated clip of a stitched,
  ``seedance``-audio run whose summed length matches the result;
* it does NOT prove that the stored prompt is the one the clip was generated
  from (prompts stay editable after generation), nor that the words were
  spoken -- the lines are script-derived and need the operator's review.

Refused (fail closed): no snapshot, no-stitch runs, ``original`` audio mode,
keep pieces, source-fallback pieces, a library entry without a valid donor,
a library clip whose bytes differ from its donor, and a length mismatch.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from . import qa_tools
from .assistant_qa import QAInputUnavailable, digest

MAX_LINES = 60
MAX_LINE_CHARS = 240
MAX_PIECES = 12
MAX_CAPTIONS = 8
WINDOW_EPSILON = 1e-4
LENGTH_TOLERANCE_SEC = 0.1
APPLY_STATUSES = ("done", "failed")


class ScriptCaptionsRefused(QAInputUnavailable):
    """The run's composition cannot be established; safe to report."""


def _status(value) -> str:
    return getattr(value, "value", value)


def hash_file(path: str, *, data_root: Path, check) -> tuple[str, tuple]:
    """sha256 of a regular file inside the data root, plus its signature."""
    resolved = Path(path).resolve(strict=False)
    try:
        resolved.relative_to(data_root)
    except ValueError:
        raise ScriptCaptionsRefused("a run file is outside the data root") from None
    try:
        fd = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise ScriptCaptionsRefused("a run file is missing") from None
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ScriptCaptionsRefused("a run file is not a regular file")
        sha = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            check()
            sha.update(chunk)
    return sha.hexdigest(), _signature(before)


def _signature(info) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _unchanged(path: str, signature: tuple) -> None:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        raise ScriptCaptionsRefused("a run file changed while it was measured") from None
    if _signature(info) != signature:
        raise ScriptCaptionsRefused("a run file changed while it was measured")


def resolve(session, *, run, library_donors, data_root: Path, probe, check) -> dict:
    """Build the script-caption evidence for *run* (a locked-free read).

    *probe(path)* returns container duration (the confined QA probe);
    *library_donors* maps stitch-plan index -> RunSegment id.
    """
    from .models import LibraryClip, Run, RunCaption, RunSegment

    if _status(run.status) != "done" or not run.result_local_path:
        raise ScriptCaptionsRefused("the run has no delivered result")
    if not run.segment_snapshot:
        raise ScriptCaptionsRefused("the run predates frozen segment snapshots")
    if run.stitch_plan == []:
        raise ScriptCaptionsRefused("a no-stitch run has no single captionable result")
    if run.audio_mode != "seedance":
        raise ScriptCaptionsRefused("the result carries the original soundtrack, not the generated dialogue")
    snapshot = {entry["id"]: entry for entry in run.segment_snapshot}
    if run.stitch_plan is None:
        plan = [{"kind": "segment", "segment_def_id": entry["id"]}
                for entry in sorted(run.segment_snapshot, key=lambda e: e["index"])]
    else:
        plan = list(run.stitch_plan)
    if len(plan) > MAX_PIECES:
        raise ScriptCaptionsRefused("the run has too many pieces for script captions")
    planned_defs = {entry.get("segment_def_id") for entry in plan if entry.get("kind") == "segment"}
    unknown = set(library_donors) - {i for i, e in enumerate(plan) if e.get("kind") == "library"}
    if unknown:
        raise ScriptCaptionsRefused("a library donor names a plan position that is not a library entry")
    if len(set(library_donors.values())) != len(library_donors):
        raise ScriptCaptionsRefused("a donor segment can stand for one library entry only")
    own_segments = {rs.segment_def_id: rs for rs in session.query(RunSegment).filter(RunSegment.run_id == run.id)}

    pieces, spec, warnings, used_defs = [], [], [], set()
    lineage = "run_segments"
    for index, entry in enumerate(plan):
        check()
        if entry.get("kind") == "segment":
            definition = snapshot.get(entry.get("segment_def_id"))
            if definition is None:
                raise ScriptCaptionsRefused("the stitch plan names a segment outside the run's snapshot")
            if definition["action"] != "swap":
                raise ScriptCaptionsRefused("keep pieces are not supported by script captions yet")
            segment = own_segments.get(definition["id"])
            if segment is None:
                raise ScriptCaptionsRefused("a planned swap segment has no generated result")
            window = definition
            kind, clip_path, piece_run = "segment", segment.local_result_path, run
        elif entry.get("kind") == "library":
            donor_id = library_donors.get(index)
            if donor_id is None:
                raise ScriptCaptionsRefused("a library entry needs the run segment it stands for")
            segment = session.get(RunSegment, donor_id)
            piece_run = session.get(Run, segment.run_id) if segment is not None else None
            if segment is None or piece_run is None or piece_run.project_id != run.project_id:
                raise ScriptCaptionsRefused("the library donor is not a segment of this project")
            window = snapshot.get(segment.segment_def_id)
            donor_window = next((e for e in (piece_run.segment_snapshot or [])
                                 if e["id"] == segment.segment_def_id), None)
            if (window is None or window["action"] != "swap" or window["id"] in planned_defs
                    or window["id"] in used_defs or donor_window is None
                    or abs(donor_window["start_sec"] - window["start_sec"]) > WINDOW_EPSILON
                    or abs(donor_window["end_sec"] - window["end_sec"]) > WINDOW_EPSILON):
                raise ScriptCaptionsRefused(
                    "the library donor must be the replaced swap segment's window of this run")
            used_defs.add(window["id"])
            kind, clip_path = "library", segment.local_result_path
            lineage = "library_donor_attestation_required"
        else:
            raise ScriptCaptionsRefused("unknown stitch plan entry")
        if _status(segment.status) != "completed" or segment.source_fallback or not clip_path:
            raise ScriptCaptionsRefused("a piece is not a completed generated clip")
        clip_sha, clip_signature = hash_file(clip_path, data_root=data_root, check=check)
        if kind == "library":
            library = session.get(LibraryClip, entry.get("clip_id")) if entry.get("clip_id") else None
            if library is None:
                warnings.append("library_clip_missing")
            elif hash_file(library.local_path, data_root=data_root, check=check)[0] != clip_sha:
                raise ScriptCaptionsRefused("the library clip's bytes differ from its donor segment")
        duration = probe(clip_path).duration_sec
        _unchanged(clip_path, clip_signature)
        prompt = segment.prompt_override or piece_run.prompt or ""
        script = qa_tools.parse_dialogue(prompt)
        spec.append({"source_start": window["start_sec"], "source_end": window["end_sec"],
                     "clip_duration": duration, "prompt": prompt})
        pieces.append({"kind": kind, "segment_def_id": window["id"], "run_segment_id": segment.id,
                       "run_id": piece_run.id, "source_start_sec": float(window["start_sec"]),
                       "source_end_sec": float(window["end_sec"]), "clip_duration_sec": float(duration),
                       "clip_sha256": clip_sha, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "script_lines": len(script)})

    final_sha, final_signature = hash_file(run.result_local_path, data_root=data_root, check=check)
    final_duration = probe(run.result_local_path).duration_sec
    _unchanged(run.result_local_path, final_signature)
    total = sum(piece["clip_duration_sec"] for piece in pieces)
    if abs(final_duration - total) > LENGTH_TOLERANCE_SEC:
        raise ScriptCaptionsRefused("the result's length does not match its pieces")
    offset = 0.0
    for piece in pieces:
        piece["offset_sec"] = round(offset, 3)
        offset += piece["clip_duration_sec"]

    lines = qa_tools.captions_from_script(spec)
    if not lines:
        raise ScriptCaptionsRefused("the prompts contain no timed dialogue")
    if len(lines) > MAX_LINES or any(len(line["text"]) > MAX_LINE_CHARS for line in lines):
        raise ScriptCaptionsRefused("the script is too long for one caption proposal")

    captions = session.query(RunCaption).filter(RunCaption.run_id == run.id).order_by(
        RunCaption.created_at.desc()).all()
    eligible = [{"caption_id": c.id, "status": _status(c.status), "source_type": c.source_type,
                 "language": c.language, "lines_count": len(c.lines or []),
                 "lines_sha256": digest(c.lines) if c.lines else None,
                 "apply_allowed": _status(c.status) in APPLY_STATUSES and c.source_type == "run"}
                for c in captions[:MAX_CAPTIONS]]
    return {
        "tool": "script_captions", "lineage": lineage,
        "final_sha256": final_sha, "final_duration_sec": float(final_duration),
        "pieces_duration_sec": round(float(total), 6), "pieces": pieces,
        "lines": [{"start": float(line["start"]), "end": float(line["end"]), "text": line["text"]} for line in lines],
        "lines_sha256": digest([{"start": float(l["start"]), "end": float(l["end"]), "text": l["text"]} for l in lines]),
        "captions": eligible, "captions_total": len(captions),
        "warnings": sorted(set(warnings)),
    }
