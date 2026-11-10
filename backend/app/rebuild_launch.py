"""Launch of an images-only (rebuild) run — design §6.1.

``api_v2.create_run`` calls :func:`launch_run` for a project whose type
generates images-only runs. Everything happens in the caller's transaction,
under the project row lock, in this order (design §6.1 / §7.1):

1. lock the project row, require ``ready`` (every rebuild admission does,
   §5.1.8), no live storyboard / keyframe row;
2. the preconditions (409 with the list of what is missing): a storyboard
   applied to the current shot generation, a selected succeeded keyframe for
   every unfolded shot of every ACTIVE swap chunk, a character sheet per
   chunk, the chunk within the model's image count and duration ceiling;
3. the default stitch plan with the trailing end card replaced by a
   language-matched library clip (Q2, :func:`default_end_card_plan`);
4. the Run row (``generation_mode='images_only'``, ``audio_mode='original'``,
   no preprocessing / references / voice), ``segment_snapshot`` written NOW;
5. run-owned keyframe copies (hard link, else copy, sha256 recorded) under
   ``storage.rebuild_run_chunk_dir`` and ``runs.rebuild_snapshot``;
6. one ``pending`` RunSegment per active swap chunk;
7. ``spend.reserve_images_only_run`` LAST (the estimate sees the segments).

Launching the run is the operator's approval of the keyframes (Q6). Nothing
here calls a provider.

:func:`load_inputs` + :func:`preconditions` are the ONE implementation of the
launch preconditions: the server enforces them here, and the project page
(``api_rebuild.rebuild_context`` -> ``unmet``) shows the same list, per model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

from sqlalchemy import select

from . import ai_models
from . import rebuild as rb
from . import rebuild_analysis
from . import rebuild_keyframes as kfmod
from . import rebuild_storyboard as sbmod
from . import spend as spend_mod
from .project_types import REBUILD_DEFAULT_RESOLUTION
from .rebuild_models import (
    PAID_CALL_LIVE_STATUSES,
    ProjectStoryboard,
    RebuildKeyframe,
    RebuildLlmCall,
)

log = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1


class LaunchRefused(Exception):
    """The launch cannot be admitted; ``status_code`` is 400 (bad request)
    or 409 (a precondition is missing — ``missing`` lists what)."""

    def __init__(self, status_code: int, detail: str, missing: Optional[list] = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.missing = list(missing or [])


@dataclass
class LaunchRequest:
    """The New Run form fields an images-only launch accepts (design §6.1).

    *stitch_plan*: the parsed plan (``api_v2._parse_stitch_plan``), or None
    for the default (every segment in order, the trailing end card replaced
    by a language-matched library clip unless *keep_source_end_card*).
    *sheet_overrides*: ``{segment_def_id: sheet_key}`` — the operator picked
    another outfit/setting group's sheet for that chunk.
    """

    model: str
    resolution: Optional[str] = None
    language: Optional[str] = None
    name: Optional[str] = None
    gdrive_folder_id: Optional[str] = None
    stitch_plan: Optional[list] = None
    stitch_plan_given: bool = False
    keep_source_end_card: bool = False
    sheet_overrides: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def aspect_for(width: Optional[int], height: Optional[int]) -> str:
    """The concrete images-only aspect ratio of a project (9:16 / 16:9 / 1:1)."""
    w, h = int(width or 0), int(height or 0)
    if w and h:
        if h > w:
            return "9:16"
        if w > h:
            return "16:9"
        return "1:1"
    return "9:16"


def stitch_target(project) -> tuple[int, int, float]:
    """``(width, height, fps_out)`` the stitch will use for *project* —
    ``media.get_default_target`` over a probe of the source, falling back to
    the probe stored on the project. The fps is the output frame grid every
    chunk is quantised on (design §6.4)."""
    from . import media as media_mod

    info = None
    src = getattr(project, "source_local_path", None)
    if src and os.path.exists(src):
        try:
            info = media_mod.probe(src)
        except Exception:  # noqa: BLE001 - fall back to the stored probe
            log.warning("rebuild launch: could not probe %s", src, exc_info=True)
    if info is None:
        info = SimpleNamespace(width=project.width, height=project.height, fps=project.fps)
    width, height, fps = media_mod.get_default_target(info)
    if not fps or fps <= 0:
        raise LaunchRefused(409, "the project's source has no usable frame rate — re-analyze it")
    return int(width or 0), int(height or 0), float(fps)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# Storyboard LLM call kinds (a run's character checks are not project work).
_STORYBOARD_CALL_KINDS = ("global", "shot")


def live_rows(db, project_id: str) -> dict:
    """The live rows that block a launch (design §6.1: "no keyframe or
    storyboard row live"): a ``queued``/``running`` storyboard, a live project
    keyframe or character-sheet row, a live storyboard LLM call (one still
    ``sending`` after its storyboard ended). Another run's chunk attempts and
    character checks are that run's work and do not block a launch."""
    from sqlalchemy import func

    def count(stmt) -> int:
        return int(db.execute(stmt).scalar() or 0)

    return {
        "storyboards": count(select(func.count()).select_from(ProjectStoryboard).where(
            ProjectStoryboard.project_id == project_id,
            ProjectStoryboard.status.in_(("queued", "running")))),
        "keyframes": count(select(func.count()).select_from(RebuildKeyframe).where(
            RebuildKeyframe.project_id == project_id,
            RebuildKeyframe.owner_run_id.is_(None),
            RebuildKeyframe.status.in_(PAID_CALL_LIVE_STATUSES))),
        "storyboard_calls": count(select(func.count()).select_from(RebuildLlmCall).where(
            RebuildLlmCall.project_id == project_id,
            RebuildLlmCall.kind.in_(_STORYBOARD_CALL_KINDS),
            RebuildLlmCall.status.in_(PAID_CALL_LIVE_STATUSES))),
    }


def live_messages(live: dict) -> list[str]:
    out = []
    if live.get("storyboards"):
        out.append("rebuild work is in flight: the storyboard is still running")
    if live.get("keyframes"):
        out.append(f"rebuild work is in flight: {live['keyframes']} keyframe(s) / "
                   "sheet(s) still being generated")
    if live.get("storyboard_calls"):
        out.append(f"rebuild work is in flight: {live['storyboard_calls']} storyboard "
                   "call(s) still running (a 'sending' call waits for its worker)")
    return out


def selected_keyframes(db, project_id: str) -> tuple[dict, dict]:
    """``({shot id: kf}, {sheet_key: kf})`` of the project's selected,
    succeeded keyframes (``rebuild_keyframes.selected_by_slot``) whose file is
    on disk — a launch copies it."""
    shots, sheets = kfmod.selected_by_slot(kfmod.project_keyframes(db, project_id))
    on_disk = lambda kf: bool(kf.image_path) and os.path.exists(kf.image_path)  # noqa: E731
    return ({k: v for k, v in shots.items() if on_disk(v)},
            {k: v for k, v in sheets.items() if on_disk(v)})


@dataclass
class LaunchInputs:
    """The rows a launch (and the project page's precondition list) reads."""

    project: object
    board: Optional[ProjectStoryboard]
    result: dict
    shots: list
    seg_defs: list
    kf_by_shot: dict
    sheets: dict
    group_keys: list
    live: dict


def load_inputs(db, project) -> LaunchInputs:
    """Read everything :func:`preconditions` needs. The storyboard is the one
    applied to the CURRENT shot generation
    (``rebuild_storyboard.applied_storyboard`` over
    ``rebuild_analysis.current_generation``)."""
    from .models import SegmentDef

    board = sbmod.applied_storyboard(db, project.id)
    result = dict(board.result or {}) if board is not None else {}
    kf_by_shot, sheets = selected_keyframes(db, project.id)
    return LaunchInputs(
        project=project, board=board, result=result,
        shots=rebuild_analysis.project_shots(db, project.id),
        seg_defs=list(db.execute(select(SegmentDef).where(SegmentDef.project_id == project.id)
                                 .order_by(SegmentDef.index)).scalars()),
        kf_by_shot=kf_by_shot, sheets=sheets,
        group_keys=[g.get("key") for g in result.get("sheet_groups") or [] if g.get("key")],
        live=live_rows(db, project.id),
    )


def basic_missing(inputs: LaunchInputs) -> list[str]:
    """The model- and chunk-independent preconditions: ``ready``, nothing
    live, an applied storyboard, shots and segments."""
    from .state_machine import ProjectStatus

    project = inputs.project
    missing: list[str] = []
    status = getattr(project.status, "value", project.status)
    if status != ProjectStatus.ready.value:
        missing.append(f"the project is {status!r}, not 'ready'")
    missing += live_messages(inputs.live)
    if inputs.board is None:
        missing.append("build the storyboard (it must be applied to the current shots)")
    if not inputs.shots:
        missing.append("the project has no shots — analyze it")
    if not inputs.seg_defs:
        missing.append("the project has no segments")
    return missing


def preconditions(inputs: LaunchInputs, *, model: str, active_ids: Optional[set] = None,
                  sheet_overrides: Optional[dict] = None, fps_out: Optional[float] = None,
                  aspect_ratio: Optional[str] = None) -> tuple[list[dict], list[str]]:
    """``(chunks, missing)`` — the §6.1 launch preconditions for *model*.

    *active_ids*: the swap chunks the stitch plan uses (None = every swap
    chunk, which is what the project page's form launches). *fps_out* /
    *aspect_ratio* only shape the returned chunk plan (frames, prompt); the
    page passes neither. Empty *missing* = the launch is admissible.
    """
    project = inputs.project
    missing = basic_missing(inputs)
    overrides = dict(sheet_overrides or {})
    for sd_id, key in overrides.items():
        if key not in inputs.sheets:
            missing.append(f"sheet override for segment {sd_id}: no selected sheet {key!r}")
    if active_ids is None:
        active_ids = {sd.id for sd in inputs.seg_defs if sd.action == "swap"}
    chunks, chunk_missing = plan_chunks(
        model=model, seg_defs=inputs.seg_defs, active_ids=active_ids, shots=inputs.shots,
        kf_by_shot=inputs.kf_by_shot, sheets=inputs.sheets, sheet_overrides=overrides,
        group_keys=inputs.group_keys, storyboard_result=inputs.result,
        aspect_ratio=aspect_ratio or aspect_for(project.width, project.height),
        fps_out=float(fps_out or project.fps or 30.0),
    )
    return chunks, missing + chunk_missing


def shots_in(shots, start: float, end: float) -> list:
    """Shots overlapping ``[start, end]`` by more than a hair."""
    return [s for s in shots
            if min(float(s.end_sec), end) - max(float(s.start_sec), start) > 1e-6]


def trailing_end_card_ids(seg_defs, shots) -> list[str]:
    """Ids of the trailing end card (§6.1): the maximal run of ``keep``
    segments at the end whose overlapping (anchor) shots are all brand cards.
    The project page shows the same ids (``api_rebuild.rebuild_context``)."""
    anchors = [s for s in shots if getattr(s, "folded_into", None) is None]
    entries = []
    for sd in sorted(seg_defs, key=lambda s: s.index):
        # Anchor shots only: a folded shot belongs to its anchor, and a group
        # is a brand card when its ANCHOR is (wave 1 brand-card rule).
        inside = shots_in(anchors, float(sd.start_sec), float(sd.end_sec))
        entries.append((sd.id, sd.action,
                        bool(inside) and all(bool(s.is_brand_card) for s in inside)))
    return rb.trailing_end_card(entries)


def language_end_card_clip(db, language: Optional[str]):
    """The newest library END CARD (``kind == "end_card"``, AICP-381) whose
    language is the run's (``en`` when the run has none) and whose file is on
    disk. Never an ``insert``: "newest clip in the language" picked the 27.8 s
    demo ``ip_reader`` on 813b251f. None if there is none (the launch then
    keeps the source card and warns)."""
    from .models import LIBRARY_CLIP_KIND_END_CARD, LibraryClip

    code = (language or "en").lower()
    for clip in db.execute(select(LibraryClip).where(
            LibraryClip.language == code,
            LibraryClip.kind == LIBRARY_CLIP_KIND_END_CARD,
    ).order_by(LibraryClip.created_at.desc(), LibraryClip.id)).scalars():
        if clip.local_path and os.path.exists(clip.local_path):
            return clip
    return None


def end_card_clips(db) -> list:
    """Every library end card, newest first (the New Run form's picker)."""
    from .models import LIBRARY_CLIP_KIND_END_CARD, LibraryClip

    return list(db.execute(select(LibraryClip).where(
        LibraryClip.kind == LIBRARY_CLIP_KIND_END_CARD,
    ).order_by(LibraryClip.created_at.desc(), LibraryClip.id).limit(200)).scalars())


def auto_end_card_by_language(db, codes) -> dict:
    """``{language code: {"id", "name", "duration_sec"} or None}`` — what the
    default plan would put in place of the source end card for each language
    (:func:`language_end_card_clip`), so the form can show it before launch."""
    out: dict = {}
    for code in codes:
        clip = language_end_card_clip(db, code)
        out[code] = None if clip is None else {
            "id": clip.id, "name": clip.name,
            "duration_sec": float(clip.duration_sec) if clip.duration_sec else None,
        }
    return out


def default_end_card_plan(db, seg_defs, shots, language: Optional[str]):
    """``(plan, end_card_ids, clip)``: the default plan with the trailing end
    card replaced by the language-matched library clip (Q2), or ``(None, ids,
    None)`` when there is no end card or no matching clip."""
    end_ids = trailing_end_card_ids(seg_defs, shots)
    if not end_ids:
        return None, [], None
    clip = language_end_card_clip(db, language)
    ordered = [sd.id for sd in sorted(seg_defs, key=lambda s: s.index)]
    plan = rb.default_stitch_plan_with_end_card(ordered, end_ids, clip.id if clip else None)
    return plan, end_ids, clip


# ---------------------------------------------------------------------------
# Chunk planning (pure over the loaded rows)
# ---------------------------------------------------------------------------


def chunk_label(chunk_no: int, start: float, end: float) -> str:
    """How a chunk is named in a precondition (the project page's numbering:
    the 1-based position among the swap segments)."""
    return f"chunk {chunk_no} ({start:.2f}–{end:.2f} s)"


def resolve_sheet(order: list, by_id: dict, sheets: dict, group_keys: list,
                  override: Optional[str]) -> tuple[Optional[str], object]:
    """``(sheet_key, selected sheet keyframe or None)`` of one chunk (§6.1,
    Q3): the operator's override, else the key held by most of its shots
    (ties -> earliest); a chunk where no shot names a group uses the first
    group (storyboard order) that has a selected sheet."""
    key = override or rb.sheet_for_chunk(
        [by_id[sid].sheet_key if sid in by_id else None for sid in order])
    if key is None:
        for k in list(group_keys) + sorted(k for k in sheets if k not in group_keys):
            if k in sheets:
                return k, sheets[k]
        return None, None
    return key, sheets.get(key)


def plan_chunks(*, model: str, seg_defs, active_ids: set, shots, kf_by_shot: dict,
                sheets: dict, sheet_overrides: dict, storyboard_result: dict,
                aspect_ratio: str, fps_out: float,
                group_keys: Optional[list] = None) -> tuple[list[dict], list[str]]:
    """The per-chunk launch plan and the list of missing preconditions.

    Each chunk dict: ``segment_def_id, index, chunk_no, start_sec, end_sec,
    request_sec, chunk_frames, shots, keyframe_order, keyframes [kf],
    sheet_key, sheet_kf, prompt``.
    """
    by_id = {s.id: s for s in shots}
    spans = [rb.ShotSpan(s.id, float(s.start_sec), float(s.end_sec), s.folded_into)
             for s in shots]
    if group_keys is None:
        group_keys = [g.get("key") for g in (storyboard_result or {}).get("sheet_groups") or []
                      if g.get("key")]
    chunks: list[dict] = []
    missing: list[str] = []
    chunk_no = 0
    for sd in sorted(seg_defs, key=lambda s: s.index):
        if sd.action != "swap":
            continue
        chunk_no += 1
        if sd.id not in active_ids:
            continue
        label = chunk_label(chunk_no, float(sd.start_sec), float(sd.end_sec))
        before = len(missing)
        try:
            pieces = rb.chunk_shot_pieces(float(sd.start_sec), float(sd.end_sec), spans)
        except ValueError as exc:
            missing.append(f"{label}: {exc}")
            continue
        try:
            rb.check_chunk_image_count(model, pieces)
        except ValueError as exc:
            missing.append(f"{label}: {exc}")
            continue
        try:
            request_sec = rb.request_sec_for_chunk(model, float(sd.end_sec) - float(sd.start_sec))
        except ValueError as exc:
            missing.append(f"{label}: {exc} — split this chunk")
            continue
        order = rb.chunk_keyframe_ids(pieces)
        kfs = [kf_by_shot.get(sid) for sid in order]
        no_kf = [str(by_id[sid].index + 1) if sid in by_id else "?"
                 for sid, kf in zip(order, kfs) if kf is None]
        if no_kf:
            missing.append(f"{label}: no selected keyframe for shot(s) {', '.join(no_kf)}")
        key, sheet = resolve_sheet(order, by_id, sheets, group_keys,
                                   sheet_overrides.get(sd.id))
        if sheet is None:
            if key is not None:
                missing.append(f"{label}: no selected character sheet for group {key!r}")
            elif not group_keys:
                missing.append(f"{label}: no selected character sheet (the storyboard names "
                               "no outfit/setting group to make one for)")
            else:
                missing.append(f"{label}: no selected character sheet")
        chunk_shots = []
        for p in pieces:
            shot = by_id.get(p.shot_id)
            chunk_shots.append({
                "shot_id": p.shot_id, "rel_start": p.rel_start, "rel_end": p.rel_end,
                "framing": getattr(shot, "framing", None), "camera": getattr(shot, "camera", None),
                "action": getattr(shot, "action", None),
                "characters": list(getattr(shot, "characters", None) or []),
            })
        prompt = None
        if len(missing) == before:
            prompt = rb.render_chunk_prompt({"request_sec": request_sec, "shots": chunk_shots},
                                            storyboard_result or {}, model,
                                            aspect_ratio=aspect_ratio)
        chunks.append({
            "segment_def_id": sd.id, "index": int(sd.index), "chunk_no": chunk_no,
            "start_sec": float(sd.start_sec), "end_sec": float(sd.end_sec),
            "request_sec": int(request_sec),
            "chunk_frames": rb.chunk_frames(float(sd.start_sec), float(sd.end_sec), fps_out),
            "shots": chunk_shots, "keyframe_order": order, "keyframes": kfs,
            "sheet_key": key, "sheet_kf": sheet, "prompt": prompt,
        })
    if not chunks and not missing:
        missing.append("no chunk to generate (every segment is kept as original footage)")
    return chunks, missing


def _character_lines(storyboard_result: dict, chunk_shots: list) -> list[dict]:
    present: list = []
    for sh in chunk_shots:
        for cid in sh.get("characters") or []:
            if cid not in present:
                present.append(cid)
    out = []
    for c in storyboard_result.get("characters") or []:
        if c.get("id") in present:
            out.append({k: c.get(k) for k in ("id", "name_hint", "look", "outfit")})
    return out


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------


def launch_run(db, project, req: LaunchRequest) -> tuple:
    """Create an images-only run in the caller's transaction (no commit).

    Returns ``(run, warnings)``. Raises :class:`LaunchRefused`, or
    ``spend.BudgetExceeded`` from the reservation (the caller maps it to 429).
    Files written for a launch that then fails are removed (best effort).
    """
    from .models import Run, RunSegment
    from .pipeline_v2 import _active_swap_segment_ids, build_segment_snapshot
    from .state_machine import ProjectStatus, RunStatus, SegmentStatus, transition
    from .storage import rebuild_run_chunk_dir, run_dir

    # 1. Lock + ready (design §5.1.8).
    db.refresh(project, with_for_update=True)
    if project.status != ProjectStatus.ready:
        raise LaunchRefused(409, f"Cannot create run: project status is {project.status!r}, "
                                 "expected 'ready'")
    model = req.model
    if not ai_models.supports_images_only(model):
        raise LaunchRefused(400, f"model must be one of {sorted(ai_models.IMAGES_ONLY_MODELS)} "
                                 "on a rebuild project")
    spec = ai_models.AI_MODELS[model]
    resolution = req.resolution or REBUILD_DEFAULT_RESOLUTION
    if resolution not in spec.resolutions:
        raise LaunchRefused(400, f"resolution {resolution!r} not allowed for model {model!r}; "
                                 f"choose one of {sorted(spec.resolutions)}")
    if (model not in spend_mod.IMAGES_ONLY_CREDITS_PER_SEC
            or resolution not in spend_mod.IMAGES_ONLY_CREDITS_PER_SEC[model]):
        raise LaunchRefused(400, f"no images-only rate for {model} at {resolution}")

    # 2. Preconditions — the same list the project page shows (preconditions()).
    inputs = load_inputs(db, project)
    if basic_missing(inputs):
        # No plan can be made yet: refuse with the FULL list, exactly what the
        # project page shows for the form's launch (every swap chunk).
        _chunks, missing = preconditions(inputs, model=model,
                                         sheet_overrides=req.sheet_overrides or {})
        raise LaunchRefused(409, "Cannot launch the rebuild: " + "; ".join(missing), missing)
    board, result, shots, seg_defs = inputs.board, inputs.result, inputs.shots, inputs.seg_defs

    warnings: list[str] = []
    plan = req.stitch_plan
    end_ids: list = []
    if not req.stitch_plan_given and not req.keep_source_end_card:
        plan, end_ids, clip = default_end_card_plan(db, seg_defs, shots, req.language)
        if end_ids and clip is not None:
            dur = f", {float(clip.duration_sec):.1f} s" if clip.duration_sec else ""
            warnings.append(f"the source end card is replaced by the library end card "
                            f"{clip.name!r} ({(req.language or 'en')}{dur})")
        elif end_ids:
            warnings.append("the source end card is kept: the library has no end card in "
                            "the run language — it still carries the source's burned captions")
    if result.get("on_screen_text"):
        kept = [sd for sd in seg_defs if sd.action == "keep"
                and (plan is None or any(isinstance(e, dict) and e.get("segment_def_id") == sd.id
                                         for e in plan))]
        for sd in kept:
            warnings.append(f"segment #{sd.index} is source footage and keeps its burned text")

    active_ids = _active_swap_segment_ids(plan, seg_defs)
    width, height, fps_out = stitch_target(project)
    aspect = aspect_for(width or project.width, height or project.height)
    chunks, missing = preconditions(
        inputs, model=model, active_ids=active_ids, sheet_overrides=req.sheet_overrides or {},
        fps_out=fps_out, aspect_ratio=aspect,
    )
    if missing:
        raise LaunchRefused(409, "Cannot launch the rebuild: " + "; ".join(missing), missing)

    # 4. The run.
    run_id = str(uuid.uuid4())
    from .config import settings

    run = Run(
        id=run_id, project_id=project.id, name=req.name, prompt="",
        model=model, resolution=resolution, audio_mode="original",
        language=req.language,
        gdrive_folder_id=req.gdrive_folder_id or settings.GDRIVE_DEFAULT_FOLDER_ID or None,
        stitch_plan=plan, status=RunStatus.created, reference_image_urls=[],
        reference_audio_urls=None, preprocess_enabled=False,
        generation_mode="images_only",
        segment_snapshot=build_segment_snapshot(seg_defs),
    )
    db.add(run)
    db.flush()

    # 5. Run-owned keyframe copies + the snapshot.
    written_dir = run_dir(run_id, project.id)
    try:
        snap_chunks = []
        for ch in chunks:
            cdir = rebuild_run_chunk_dir(run_id, project.id, ch["index"])
            copies = {}
            images = []
            for n, (sid, kf) in enumerate(zip(ch["keyframe_order"], ch["keyframes"]), start=1):
                dst = os.path.join(cdir, f"{n}.png")
                _link_or_copy(kf.image_path, dst)
                copies[sid] = {"keyframe_id": kf.id, "keyframe_path": dst,
                               "sha256": file_sha256(dst)}
                images.append(dst)
            sheet_dst = os.path.join(cdir, "sheet.png")
            _link_or_copy(ch["sheet_kf"].image_path, sheet_dst)
            images.append(sheet_dst)
            snap_chunks.append({
                "segment_def_id": ch["segment_def_id"], "index": ch["index"],
                "start_sec": ch["start_sec"], "end_sec": ch["end_sec"],
                "request_sec": ch["request_sec"], "chunk_frames": ch["chunk_frames"],
                "prompt": ch["prompt"],
                "shots": [dict(sh, **copies[sh["shot_id"]]) for sh in ch["shots"]],
                "images": images,
                "sheet_key": ch["sheet_key"], "sheet_keyframe_id": ch["sheet_kf"].id,
                "sheet_path": sheet_dst, "sheet_sha256": file_sha256(sheet_dst),
                "characters": _character_lines(result, ch["shots"]),
            })
        run.rebuild_snapshot = {
            "version": SNAPSHOT_VERSION, "model": model, "resolution": resolution,
            "aspect_ratio": aspect, "fps_out": fps_out,
            "storyboard_id": board.id, "shots_generation": int(board.shots_generation),
            "style": result.get("style"), "end_card_ids": end_ids,
            "chunks": snap_chunks,
        }
        # 6. Pending RunSegments (they must exist before the reservation).
        for ch in chunks:
            db.add(RunSegment(run_id=run_id, segment_def_id=ch["segment_def_id"],
                              index=ch["index"], status=SegmentStatus.pending))
        transition(run, RunStatus.queued)
        db.flush()
        # 7. Reserve LAST (design §6.1.4, §7.1).
        spend_mod.reserve_images_only_run(db, run, kind="new run")
    except BaseException:
        shutil.rmtree(written_dir, ignore_errors=True)
        raise
    log.info("rebuild launch: run %s on project %s, %d chunk(s), model %s %s",
             run_id, project.id, len(chunks), model, resolution)
    return run, warnings


def discard_run_files(run) -> None:
    """Remove a failed launch's files (the commit did not land)."""
    from .storage import run_dir

    shutil.rmtree(run_dir(run.id, run.project_id), ignore_errors=True)
