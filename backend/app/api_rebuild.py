"""HTTP surface of the rebuild project phase (design §8, §10) — wave 2a.

JSON/action endpoints under ``/api/v2`` (:data:`router`) and the project-page
fragment under ``/v2`` (:data:`web_router`):

* storyboard: build (Resume / From scratch), cancel, status;
* shots: list, edit storyboard fields / ``frame_sec`` / unfold;
* keyframes: admit a batch (missing shots, explicit shots = regenerate, sheets),
  select an attempt, images;
* re-chunking: keep brand-card shots as original footage, unfold — both
  through ``rebuild_analysis.rechunk``.

Every mutating route requires ``REBUILD_ENABLED`` (409 otherwise), a rebuild
project, the project row lock and ``status == ready`` (design §5.1.8); reads
stay available with the flag off so existing rows remain visible (§7.6).
The run launch is the EXISTING ``POST /api/v2/projects/{pid}/runs`` (wave 2b
implements its images-only branch); the fragment only renders its form.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import ai_models, languages
from . import rebuild as rb
from . import rebuild_analysis
from . import rebuild_jobs as jobs
from . import rebuild_keyframes as kfmod
from . import rebuild_launch
from . import rebuild_paid_call as pc
from . import rebuild_storyboard as sbmod
from . import spend as spend_mod
from .config import settings
from .db import get_db
from .ops_gate import GATED
from .models import Run, SegmentDef, VideoProject
from .project_types import REBUILD_DEFAULT_RESOLUTION, is_rebuild, spec_for
from .rebuild_models import (
    PAID_CALL_LIVE_STATUSES,
    PC_SUCCEEDED,
    PC_UNKNOWN,
    RebuildKeyframe,
    RebuildShot,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["rebuild"])
web_router = APIRouter(tags=["rebuild-web"])

FLAG_OFF = "Rebuild from reference is not enabled on this server (REBUILD_ENABLED)"
_SHOT_TEXT_FIELDS = ("location", "framing", "camera", "action")
# The placeholder prompt the launch form posts: create_run requires a prompt
# field, an images-only run renders its prompts per chunk from the snapshot.
LAUNCH_PROMPT_PLACEHOLDER = "Rebuild from reference (prompts are rendered per chunk)"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _require_enabled() -> None:
    if not settings.REBUILD_ENABLED:
        raise HTTPException(status_code=409, detail=FLAG_OFF)


def _project_or_404(db: Session, pid: str) -> VideoProject:
    project = db.get(VideoProject, pid)
    if project is None or not is_rebuild(project.project_type):
        raise HTTPException(status_code=404, detail="rebuild project not found")
    return project


def _shot_or_404(db: Session, sid: str) -> RebuildShot:
    shot = db.get(RebuildShot, sid)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    return shot


def _locked_ready(db: Session, project: VideoProject) -> VideoProject:
    locked = sbmod._lock_project(db, project.id)
    if locked is None:
        raise HTTPException(status_code=404, detail="project not found")
    sbmod.require_ready(locked)
    return locked


def _require_no_active_run(db: Session, project: VideoProject, action: str) -> None:
    from .api_v2 import _ACTIVE_RUN_STATUSES

    if db.execute(select(Run.id).where(Run.project_id == project.id,
                                       Run.status.in_(_ACTIVE_RUN_STATUSES))).first():
        raise HTTPException(status_code=409, detail=(
            f"Cannot {action} while one of this project's runs is active — wait for it"))
    if pc.project_live_rows(db, project.id)["attempts"]:
        raise HTTPException(status_code=409, detail=(
            f"Cannot {action} while a chunk attempt is in flight"))
    if sbmod.live_storyboard(db, project.id) is not None:
        raise HTTPException(status_code=409, detail=(
            f"Cannot {action} while the storyboard is running"))


def _keeping_brand_cards(db: Session, project_id: str) -> bool:
    return db.execute(select(SegmentDef.id).where(
        SegmentDef.project_id == project_id, SegmentDef.action == "keep")).first() is not None


def _csv(value: Optional[str]) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _bool(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "on", "yes")


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------


def _storyboard_json(db: Session, sb) -> Optional[dict]:
    if sb is None:
        return None
    calls = sbmod.storyboard_calls(db, sb.id)
    counts: dict = {}
    for c in calls:
        counts[c.status] = counts.get(c.status, 0) + 1
    return {
        "id": sb.id, "status": sb.status, "build_mode": sb.build_mode, "model": sb.model,
        "shots_generation": sb.shots_generation, "global_call_id": sb.global_call_id,
        "budget_credits": sb.budget_credits, "error": sb.error,
        "applied_at": sb.applied_at.isoformat() if sb.applied_at else None,
        "created_at": sb.created_at.isoformat() if sb.created_at else None,
        "calls": counts, "result": sb.result,
    }


@router.post("/projects/{pid}/rebuild/storyboard", status_code=status.HTTP_202_ACCEPTED, dependencies=[GATED])
def start_storyboard(
    pid: str,
    mode: str = Form("resume"),
    overwrite_edits: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    """Admit a storyboard build (design §5.2). 409 while one is live / a
    previous call is still sending; 429 on budget."""
    _require_enabled()
    project = _project_or_404(db, pid)
    sb = sbmod.admit_storyboard(db, project, mode=mode, overwrite_edits=_bool(overwrite_edits))
    db.commit()
    sbmod.enqueue_first(sb)
    return {"storyboard_id": sb.id, "status": sb.status, "budget_credits": sb.budget_credits,
            "reused_global_call_id": sb.global_call_id}


@router.post("/projects/{pid}/rebuild/storyboard/cancel")
def cancel_storyboard(pid: str, db: Session = Depends(get_db)) -> dict:
    """Interrupt the running storyboard; its unsent calls are cancelled in the
    same transaction (calls already sending finish on their own)."""
    _require_enabled()
    project = _project_or_404(db, pid)
    sb = sbmod.cancel_storyboard(db, project)
    db.commit()
    return {"storyboard_id": sb.id, "status": sb.status}


@router.get("/projects/{pid}/rebuild/storyboard")
def get_storyboard(pid: str, db: Session = Depends(get_db)) -> dict:
    project = _project_or_404(db, pid)
    latest = sbmod.latest_storyboard(db, project.id)
    applied = sbmod.applied_storyboard(db, project.id)
    shots = rebuild_analysis.project_shots(db, project.id)
    warn = sbmod.speech_warning(applied.result if applied else None, shots)
    return {"latest": _storyboard_json(db, latest),
            "applied_storyboard_id": applied.id if applied else None,
            "warnings": [warn] if warn else [],
            "estimate_credits": spend_mod.storyboard_credits(
                sum(1 for s in shots if s.folded_into is None))}


# ---------------------------------------------------------------------------
# Shots
# ---------------------------------------------------------------------------


def _shot_json(shot: RebuildShot, selected: Optional[RebuildKeyframe]) -> dict:
    return {
        "id": shot.id, "index": shot.index, "generation": shot.generation,
        "start_sec": shot.start_sec, "end_sec": shot.end_sec, "frame_sec": shot.frame_sec,
        "folded_into": shot.folded_into, "force_anchor": shot.force_anchor,
        "location": shot.location, "framing": shot.framing, "camera": shot.camera,
        "characters": shot.characters, "action": shot.action,
        "speaking_on_camera": shot.speaking_on_camera, "is_brand_card": shot.is_brand_card,
        "sheet_key": shot.sheet_key, "edited": shot.edited_at is not None,
        "selected_keyframe_id": selected.id if selected else None,
    }


@router.get("/projects/{pid}/rebuild/shots")
def list_shots(pid: str, db: Session = Depends(get_db)) -> list[dict]:
    project = _project_or_404(db, pid)
    sel, _ = kfmod.selected_by_slot(kfmod.project_keyframes(db, project.id))
    return [_shot_json(s, sel.get(s.id)) for s in rebuild_analysis.project_shots(db, project.id)]


@router.patch("/rebuild/shots/{sid}")
def edit_shot(
    sid: str,
    location: Optional[str] = Form(None),
    framing: Optional[str] = Form(None),
    camera: Optional[str] = Form(None),
    action: Optional[str] = Form(None),
    characters: Optional[str] = Form(None),
    speaking_on_camera: Optional[str] = Form(None),
    is_brand_card: Optional[str] = Form(None),
    sheet_key: Optional[str] = Form(None),
    frame_sec: Optional[float] = Form(None),
    force_anchor: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    """Edit a shot's storyboard fields (sets ``edited_at``), its keyframe
    frame time, or unfold it (``force_anchor`` — re-chunks, refused while a
    run is active). Locks the project row; ``ready`` required."""
    _require_enabled()
    shot = _shot_or_404(db, sid)
    project = _project_or_404(db, shot.project_id)
    project = _locked_ready(db, project)
    shot = db.execute(select(RebuildShot).where(RebuildShot.id == sid).with_for_update()
                      .execution_options(populate_existing=True)).scalar_one_or_none()
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    unfold = force_anchor is not None and _bool(force_anchor) != bool(shot.force_anchor)
    if unfold:
        _require_no_active_run(db, project, "re-chunk the project")
    now = jobs._utcnow()
    edited = False
    for name, value in (("location", location), ("framing", framing), ("camera", camera),
                        ("action", action)):
        if value is not None:
            setattr(shot, name, " ".join(value.split()) or None)
            edited = True
    if characters is not None:
        shot.characters = _csv(characters)
        edited = True
    if speaking_on_camera is not None:
        shot.speaking_on_camera = speaking_on_camera.strip()[:64] or None
        edited = True
    if is_brand_card is not None:
        shot.is_brand_card = _bool(is_brand_card)
        edited = True
    if sheet_key is not None:
        shot.sheet_key = sbmod._key(sheet_key) or None
        edited = True
    if frame_sec is not None:
        if not (float(shot.start_sec) <= float(frame_sec) < float(shot.end_sec)):
            raise HTTPException(status_code=400, detail=(
                f"frame_sec must lie inside the shot ({shot.start_sec:.2f}–{shot.end_sec:.2f} s)"))
        shot.frame_sec = round(float(frame_sec), 3)
    if edited:
        shot.edited_at = now
    shot.updated_at = now
    rechunked = False
    if unfold:
        shot.force_anchor = _bool(force_anchor)
        db.flush()
        rebuild_analysis.rechunk(db, project,
                                 keep_brand_cards=_keeping_brand_cards(db, project.id))
        rechunked = True
    db.commit()
    db.refresh(shot)
    return {"shot": _shot_json(shot, None), "rechunked": rechunked}


@router.post("/projects/{pid}/rebuild/keep-brand-cards")
def keep_brand_cards(pid: str, keep: str = Form("true"),
                     db: Session = Depends(get_db)) -> dict:
    """Re-cut the chunks so brand-card shots become ``keep`` (original
    footage) segments — or back (``keep=false``). Free; refused while a run
    is active, like every segmentation change (design §5.1.5)."""
    _require_enabled()
    project = _locked_ready(db, _project_or_404(db, pid))
    _require_no_active_run(db, project, "re-chunk the project")
    try:
        plan = rebuild_analysis.rechunk(db, project, keep_brand_cards=_bool(keep))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.commit()
    return {"segments": [{"start_sec": p.start, "end_sec": p.end, "action": p.action}
                         for p in plan]}


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------


@router.post("/projects/{pid}/rebuild/keyframes", status_code=status.HTTP_202_ACCEPTED, dependencies=[GATED])
def generate_keyframes(
    pid: str,
    shots: Optional[str] = Form(None),
    sheets: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    sheet_prompt: Optional[str] = Form(None),
    sheet_inputs: Optional[str] = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    """Admit a keyframe batch (design §5.3): ``shots`` = ``all_missing`` or a
    comma list of shot ids (one id + ``prompt`` = regenerate one);
    ``sheets`` = ``all_missing`` or a comma list of sheet keys;
    ``sheet_inputs`` = JSON ``{sheet_key: [shot ids]}`` (the input-frame pick)."""
    _require_enabled()
    project = _project_or_404(db, pid)
    shot_ids = [] if (shots or "").strip() == "all_missing" else _csv(shots)
    sheet_keys = [] if (sheets or "").strip() == "all_missing" else _csv(sheets)
    inputs = None
    if sheet_inputs:
        try:
            inputs = json.loads(sheet_inputs)
            if not isinstance(inputs, dict):
                raise ValueError
            inputs = {str(k): [str(x) for x in v] for k, v in inputs.items()}
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(status_code=400,
                                detail="sheet_inputs must be a JSON object") from exc
    batch_id, rows = kfmod.admit_keyframes(
        db, project, shot_ids=shot_ids, all_missing=(shots or "").strip() == "all_missing",
        sheet_keys=sheet_keys, all_missing_sheets=(sheets or "").strip() == "all_missing",
        prompt=prompt, sheet_prompt_override=sheet_prompt, sheet_inputs=inputs)
    db.commit()
    kfmod.enqueue_batch(rows)
    return {"batch_id": batch_id, "keyframe_ids": [r.id for r in rows],
            "reserved_credits": spend_mod.keyframes_credits(len(rows))}


@router.patch("/rebuild/keyframes/{kid}")
def select_keyframe(kid: str, selected: str = Form("true"),
                    db: Session = Depends(get_db)) -> dict:
    """``{selected: true}`` — pick this attempt for its shot / sheet (free)."""
    _require_enabled()
    if not _bool(selected):
        raise HTTPException(status_code=400, detail="only selected=true is supported")
    kf = db.get(RebuildKeyframe, kid)
    if kf is None:
        raise HTTPException(status_code=404, detail="keyframe not found")
    project = _project_or_404(db, kf.project_id)
    kf = kfmod.select_keyframe(db, project, kid)
    db.commit()
    return {"keyframe_id": kf.id, "selected": True}


def _png(path: Optional[str]) -> FileResponse:
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="image not on disk")
    return FileResponse(path)


@router.get("/rebuild/keyframes/{kid}/image")
def keyframe_image(kid: str, db: Session = Depends(get_db)) -> FileResponse:
    kf = db.get(RebuildKeyframe, kid)
    if kf is None:
        raise HTTPException(status_code=404, detail="keyframe not found")
    return _png(kf.image_path)


@router.get("/rebuild/shots/{sid}/frame")
def shot_frame(sid: str, db: Session = Depends(get_db)) -> FileResponse:
    """The source frame at the shot's ``frame_sec`` (extracted once, cached
    under the name the keyframe job uses). Free."""
    shot = _shot_or_404(db, sid)
    project = db.get(VideoProject, shot.project_id)
    src = getattr(project, "source_local_path", None)
    if not src or not os.path.exists(src):
        raise HTTPException(status_code=404, detail="source not on disk")
    path = jobs.frame_path(shot.project_id, shot.id, float(shot.frame_sec))
    try:
        jobs.extract_frame(src, float(shot.frame_sec), path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"frame not available: {exc}") from exc
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Project-page fragment (design §10)
# ---------------------------------------------------------------------------


def _run_estimates(chunks: list[dict]) -> dict:
    """``{model: {resolution: credits}}`` of a launch over the swap chunks:
    the video task per chunk + its character-check allowance (2 calls)."""
    out: dict = {}
    for key, spec in ai_models.IMAGES_ONLY_MODELS.items():
        rates = spend_mod.IMAGES_ONLY_CREDITS_PER_SEC.get(key, {})
        out[key] = {}
        for res in spec.resolutions:
            if res not in rates:
                continue
            total = 0.0
            for c in chunks:
                if c["action"] != "swap":
                    continue
                sec = rb.request_sec_for_chunk(key, c["end"] - c["start"])
                total += spend_mod.images_only_video_credits(key, res, sec) \
                    + 2 * spend_mod.llm_credits()
            out[key][res] = round(total, 1)
    return out


def rebuild_context(db: Session, project: VideoProject) -> dict:
    """Everything the rebuild project card renders (read-only)."""
    enabled = bool(settings.REBUILD_ENABLED)
    shots = rebuild_analysis.project_shots(db, project.id)
    by_id = {s.id: s for s in shots}
    segs = list(db.execute(select(SegmentDef).where(SegmentDef.project_id == project.id)
                           .order_by(SegmentDef.index)).scalars())
    kfs = kfmod.project_keyframes(db, project.id)
    sel_shots, sel_sheets = kfmod.selected_by_slot(kfs)
    live_kf_shots, live_kf_sheets = kfmod.live_slots(kfs)
    attempts: dict = {}
    sheet_attempts: dict = {}
    for r in reversed(kfs):  # newest first
        if r.shot_id:
            attempts.setdefault(r.shot_id, []).append(r)
        else:
            sheet_attempts.setdefault(r.sheet_key, []).append(r)

    latest = sbmod.latest_storyboard(db, project.id)
    applied = sbmod.applied_storyboard(db, project.id)
    sb_live = latest is not None and latest.status in ("queued", "running")
    calls = sbmod.storyboard_calls(db, latest.id) if latest is not None else []
    progress = {"total": sum(1 for c in calls), "in_flight": sum(
        1 for c in calls if c.status in PAID_CALL_LIVE_STATUSES),
        "unknown": sum(1 for c in calls if c.status == PC_UNKNOWN)}
    shot_call_state: dict = {}
    for c in calls:
        if c.kind == "shot":
            shot_call_state[c.shot_id] = c
    result = (applied.result if applied else None) or (latest.result if latest else None) or {}
    groups = result.get("sheet_groups") or []
    group_keys = [g["key"] for g in groups]

    anchors = [s for s in shots if s.folded_into is None]
    spans = [rb.ShotSpan(s.id, float(s.start_sec), float(s.end_sec), s.folded_into)
             for s in shots]
    # The launch preconditions: ONE implementation (rebuild_launch), shown here
    # per images-only model exactly as create_run enforces them for the form's
    # launch (every swap chunk, the auto-picked sheets).
    project_spec = spec_for(project.project_type)
    inputs = rebuild_launch.load_inputs(db, project)
    unmet_by_model: dict = {}
    plan_by_model: dict = {}
    for key in ai_models.IMAGES_ONLY_MODELS:
        planned, missing = rebuild_launch.preconditions(inputs, model=key)
        unmet_by_model[key] = missing
        plan_by_model[key] = {c["segment_def_id"]: c for c in planned}
    default_model = project_spec.default_model
    if default_model not in unmet_by_model:
        default_model = next(iter(unmet_by_model))
    unmet = unmet_by_model[default_model]
    planned_default = plan_by_model[default_model]

    chunks = []
    swap_index = 0
    for seg in segs:
        start, end = float(seg.start_sec), float(seg.end_sec)
        inside = [s for s in shots if float(s.end_sec) > start + 1e-6
                  and float(s.start_sec) < end - 1e-6]
        entry = {"seg": seg, "action": seg.action, "start": start, "end": end,
                 "shots": inside, "keyframe_ids": [], "sheet_key": None, "chunk_no": None,
                 "too_many": {}}
        if seg.action == "swap":
            swap_index += 1
            entry["chunk_no"] = swap_index
            try:
                pieces = rb.chunk_shot_pieces(start, end, spans)
                ids = rb.chunk_keyframe_ids(pieces)
            except ValueError:
                pieces, ids = [], []
            entry["keyframe_ids"] = ids
            planned = planned_default.get(seg.id) or next(
                (pm[seg.id] for pm in plan_by_model.values() if seg.id in pm), None)
            entry["sheet_key"] = planned["sheet_key"] if planned else rb.sheet_for_chunk(
                [by_id[i].sheet_key for i in ids if i in by_id])
            for key, spec in ai_models.IMAGES_ONLY_MODELS.items():
                try:
                    rb.check_chunk_image_count(key, pieces)
                except ValueError as exc:
                    entry["too_many"][key] = str(exc)
        chunks.append(entry)

    missing_kf = [s for s in anchors if s.id not in sel_shots and s.id not in live_kf_shots]
    missing_sheets = [k for k in group_keys if k not in sel_sheets and k not in live_kf_sheets]
    end_card = rebuild_launch.trailing_end_card_ids(segs, shots)
    keep_with_text = bool(result.get("on_screen_text")) and any(
        c["action"] == "keep" for c in chunks)
    # AICP-381: the End card picker lists library END CARDS only, and the form
    # shows what "Automatic" resolves to for each language before launch.
    clips = rebuild_launch.end_card_clips(db) if end_card else []
    auto_end_card = (rebuild_launch.auto_end_card_by_language(db, languages.LANGUAGES)
                     if end_card else {})
    warn = sbmod.speech_warning(applied.result if applied else None, shots)
    llm = spend_mod.llm_credits()
    img = spend_mod.image_credits(kfmod.KEYFRAME_RESOLUTION)
    return {
        "project": project, "rebuild_enabled": enabled, "shots": shots, "by_id": by_id,
        "anchors": anchors, "chunks": chunks, "segments": segs,
        "n_swap_chunks": sum(1 for c in chunks if c["action"] == "swap"),
        "duration": float(project.duration_sec or (segs[-1].end_sec if segs else 0) or 0),
        "storyboard": latest, "applied": applied, "storyboard_live": sb_live,
        "storyboard_progress": progress, "shot_call_state": shot_call_state,
        "result": result, "sheet_groups": groups,
        "selected_shots": sel_shots, "selected_sheets": sel_sheets,
        "attempts": attempts, "sheet_attempts": sheet_attempts,
        "live_kf_shots": live_kf_shots, "live_kf_sheets": live_kf_sheets,
        "missing_keyframes": missing_kf, "missing_sheets": missing_sheets,
        "sheet_auto_inputs": {g["key"]: [s.id for s in kfmod.sheet_input_shots(
            shots, g["key"], g.get("characters") or [])] for g in groups},
        "storyboard_estimate": spend_mod.storyboard_credits(len(anchors)),
        "llm_credits": llm, "image_credits": img,
        "keyframe_estimate": (len(missing_kf) + len(missing_sheets)) * img,
        "run_estimates": _run_estimates(chunks),
        # Resolution options per model IN REGISTRY ORDER (a JSON object's keys
        # come back sorted, which listed 1080p before 480p).
        "run_resolutions": {k: [r for r in spec.resolutions
                                if r in spend_mod.IMAGES_ONLY_CREDITS_PER_SEC.get(k, {})]
                            for k, spec in ai_models.IMAGES_ONLY_MODELS.items()},
        "images_only_models": ai_models.IMAGES_ONLY_MODELS,
        "default_model": default_model,
        "default_resolution": REBUILD_DEFAULT_RESOLUTION,
        "languages": languages.LANGUAGES, "default_language": languages.DEFAULT_LANGUAGE,
        "library_clips": clips, "end_card_ids": end_card,
        "auto_end_card": auto_end_card,
        "segment_ids": [str(s.id) for s in segs],
        "speech_warning": warn, "keep_with_text": keep_with_text,
        "keeping_brand_cards": any(c["action"] == "keep" for c in chunks),
        "brand_card_shots": sum(1 for s in anchors if s.is_brand_card),
        "unmet": unmet, "launch_ready": enabled and not unmet,
        "unmet_by_model": unmet_by_model,
        "launch_ready_by_model": {k: enabled and not v for k, v in unmet_by_model.items()},
        # Poll while anything the launch waits for is live (a storyboard
        # call still `sending` after its storyboard ended included).
        "live": sb_live or bool(live_kf_shots or live_kf_sheets)
        or bool(inputs.live.get("storyboard_calls")),
        "launch_prompt": LAUNCH_PROMPT_PLACEHOLDER,
        "succeeded": PC_SUCCEEDED,
    }


@web_router.get("/projects/{pid}/rebuild-fragment", response_class=HTMLResponse)
def rebuild_fragment(pid: str, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    from .web_v2 import templates

    project = _project_or_404(db, pid)
    ctx = rebuild_context(db, project)
    ctx["request"] = request
    return templates.TemplateResponse(request, "partials/project_rebuild_card.html", ctx)
