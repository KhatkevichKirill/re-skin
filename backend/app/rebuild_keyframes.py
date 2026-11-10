"""Keyframes and character sheets of a rebuild project (design §5.3).

One ``rebuild_keyframes`` row per GPT Image attempt
(``gpt-image-2-5-flare-image-to-image``, 2K, the project's 9:16 / 16:9):

* a **shot keyframe** — one per unfolded shot, from the source frame at the
  shot's ``frame_sec`` (default the midpoint, Q3/Q4), redrawn clean (no
  captions);
* a **character sheet** — one per storyboard ``sheet_groups`` key (one per
  outfit/setting, Q4), from up to five frames of that group's shots.

Every row is ONE paid call driven by the wave 1 protocol (§7.2, notes §2.1),
``STEP_HANDLERS["rebuild_keyframes"] = "app.rebuild_keyframes.step"``:
``claim`` (frames + uploads, free) -> ``send`` (one ``create_task_once``)
-> ``poll`` -> ``fetch`` (download + verify; a failed download retries the
DOWNLOAD only). ``failed`` / ``unknown`` are final: a regeneration is a new
row with its own reservation, never an automatic re-buy.

**Selection.** At most one selected keyframe per shot and one selected sheet
per (project, sheet key) — the partial unique indexes of §4. Selecting is
"unselect the slot, then select the row" under the project row lock
(:func:`select_keyframe`). The newest successful attempt is selected
automatically when it completes, unless the operator picked another attempt
(no column records a manual pick: a pick stamps ``updated_at`` after the
row's ``finished_at``, while an automatic selection writes both with the same
instant — see :func:`_auto_select`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import timedelta
from typing import Callable, Optional, Sequence

from fastapi import HTTPException
from sqlalchemy import select, update

from . import rebuild_jobs as jobs
from . import rebuild_paid_call as pc
from . import rebuild_storyboard as sbmod
from . import spend as spend_mod
from .models import VideoProject
from .rebuild_models import (
    PAID_CALL_LIVE_STATUSES,
    PC_FETCHING,
    PC_SUBMITTED,
    PC_SUCCEEDED,
    RebuildKeyframe,
    RebuildShot,
)

log = logging.getLogger(__name__)

TABLE = RebuildKeyframe.__tablename__
# Pinned (design §5.3): the 2.5 variants take the 9:16 / 16:9 aspect at 2K.
KEYFRAME_MODEL = "gpt-image-2-5-flare-image-to-image"
KEYFRAME_RESOLUTION = spend_mod.REBUILD_KEYFRAME_RESOLUTION  # "2K"
SHEET_MAX_INPUTS = 5
# A submitted task that has not finished after this long is given up (the
# draw stays at the estimate; the task id stays on the row).
POLL_MAX_SEC = 3600
FETCH_RETRY_DELAY_SEC = 15

DEFAULT_SHOT_PROMPT = (
    "Recreate this exact frame of a {style} film as a clean, sharp, high-quality render. "
    "Remove ALL burned-in captions, subtitles and overlaid text completely; reconstruct "
    "what is behind them. Keep everything else exactly the same: the same characters with "
    "the same faces, hair, skin tone, body and outfits, the same pose, expression, framing, "
    "camera angle, background, lighting and colours. Do not add any text, logos or "
    "watermarks."
)
_FRAMING_RANK = {f: i for i, f in enumerate(sbmod.FRAMINGS)}


# ---------------------------------------------------------------------------
# Prompts + inputs
# ---------------------------------------------------------------------------


def shot_prompt(storyboard_result: Optional[dict]) -> str:
    style = sbmod._s((storyboard_result or {}).get("style")).rstrip(".") or "animated"
    return DEFAULT_SHOT_PROMPT.format(style=style)


def sheet_prompt(storyboard_result: Optional[dict], key: str) -> str:
    res = storyboard_result or {}
    group = next((g for g in res.get("sheet_groups") or [] if g.get("key") == key), None) or {}
    members = group.get("characters") or []
    clauses = []
    for c in res.get("characters") or []:
        if c.get("id") in members:
            details = "; ".join(x for x in (c.get("look"), c.get("outfit")) if x)
            clauses.append(f"{c.get('name_hint') or c.get('id')} ({details})" if details
                           else str(c.get("name_hint") or c.get("id")))
    style = sbmod._s(res.get("style")).rstrip(".") or "animated"
    who = ", ".join(clauses) if clauses else "the characters in the input images"
    setting = f" in their {group.get('description')}" if group.get("description") else ""
    return (
        f"A clean character reference sheet of a {style} film: {who}{setting}, standing "
        "side by side, full body, front view, on a plain light background. Exactly the same "
        "faces, hair, skin tone, body and outfits as in the input images. No text, no "
        "labels, no captions, no logos, no watermarks."
    )


def sheet_input_shots(shots: Sequence[RebuildShot], key: str, members: Sequence[str],
                      limit: int = SHEET_MAX_INPUTS) -> list[RebuildShot]:
    """Up to *limit* anchor shots of the group *key*, chosen so each member
    character appears where it is largest (tightest framing first)."""
    anchors = [s for s in shots if s.folded_into is None]
    pool = [s for s in anchors if s.sheet_key == key] or \
        [s for s in anchors if set(s.characters or []) & set(members)]

    def rank(s):
        return (_FRAMING_RANK.get((s.framing or "").lower(), len(_FRAMING_RANK)), s.index)

    out: list[RebuildShot] = []
    for cid in members:
        for s in sorted((s for s in pool if cid in (s.characters or [])), key=rank):
            if s not in out:
                out.append(s)
                break
        if len(out) >= limit:
            return out
    for s in sorted(pool, key=rank):
        if len(out) >= limit:
            break
        if s not in out:
            out.append(s)
    return out


def _inputs(project_id: str, shots: Sequence[RebuildShot]) -> list[dict]:
    return [{"shot_id": s.id, "t": round(float(s.frame_sec), 3),
             "path": jobs.frame_path(project_id, s.id, float(s.frame_sec))} for s in shots]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def project_keyframes(db, project_id: str) -> list[RebuildKeyframe]:
    return list(db.execute(select(RebuildKeyframe).where(
        RebuildKeyframe.project_id == project_id, RebuildKeyframe.owner_run_id.is_(None),
    ).order_by(RebuildKeyframe.created_at, RebuildKeyframe.id)).scalars())


def selected_by_slot(rows: Sequence[RebuildKeyframe]) -> tuple[dict, dict]:
    """``({shot_id: kf}, {sheet_key: kf})`` of the selected, succeeded rows."""
    shots, sheets = {}, {}
    for r in rows:
        if not r.selected or r.status != PC_SUCCEEDED:
            continue
        if r.shot_id:
            shots[r.shot_id] = r
        else:
            sheets[r.sheet_key] = r
    return shots, sheets


def live_slots(rows: Sequence[RebuildKeyframe]) -> tuple[set, set]:
    shots = {r.shot_id for r in rows if r.shot_id and r.status in PAID_CALL_LIVE_STATUSES}
    sheets = {r.sheet_key for r in rows if not r.shot_id and r.status in PAID_CALL_LIVE_STATUSES}
    return shots, sheets


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def admit_keyframes(
    db,
    project: VideoProject,
    *,
    shot_ids: Optional[Sequence[str]] = None,
    all_missing: bool = False,
    sheet_keys: Optional[Sequence[str]] = None,
    all_missing_sheets: bool = False,
    prompt: Optional[str] = None,
    sheet_prompt_override: Optional[str] = None,
    sheet_inputs: Optional[dict] = None,
) -> tuple[str, list[RebuildKeyframe]]:
    """Admit one keyframe batch in the caller's transaction (no commit; the
    caller commits, then :func:`enqueue_batch`). Locks the project row and
    requires ``ready``; requires a storyboard applied to the current shot
    generation and none running; reserves ``N x 10`` credits under
    ``keyframes_scope(batch)`` (429 when today's budget cannot cover it)."""
    locked = sbmod._lock_project(db, project.id)
    if locked is None:
        raise HTTPException(status_code=404, detail="project not found")
    project = locked
    sbmod.require_ready(project)
    if not project.source_local_path or not os.path.exists(project.source_local_path):
        raise HTTPException(status_code=409, detail="this project has no source video on disk")
    if sbmod.live_storyboard(db, project.id) is not None:
        raise HTTPException(status_code=409, detail="a storyboard is running — wait for it")
    board = sbmod.applied_storyboard(db, project.id)
    if board is None:
        raise HTTPException(status_code=409, detail=(
            "build the storyboard first: keyframes need a storyboard applied to the "
            "current shots"))
    result = board.result or {}
    from . import rebuild_analysis

    shots = rebuild_analysis.project_shots(db, project.id)
    by_id = {s.id: s for s in shots}
    rows = project_keyframes(db, project.id)
    sel_shots, sel_sheets = selected_by_slot(rows)
    live_shots, live_sheets = live_slots(rows)

    want_shots: list[RebuildShot] = []
    if all_missing:
        want_shots = [s for s in shots if s.folded_into is None and s.id not in sel_shots
                      and s.id not in live_shots]
    for sid in shot_ids or []:
        s = by_id.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail=f"shot {sid} is not a shot of this project")
        if s.folded_into is not None:
            raise HTTPException(status_code=400, detail=(
                f"shot {s.index + 1} is folded into another shot and has no keyframe of its "
                "own — unfold it first"))
        if s.id in live_shots:
            raise HTTPException(status_code=409, detail=(
                f"a keyframe of shot {s.index + 1} is already being generated"))
        if s not in want_shots:
            want_shots.append(s)

    groups = {g["key"]: g for g in result.get("sheet_groups") or []}
    want_sheets: list[str] = []
    if all_missing_sheets:
        want_sheets = [k for k in groups if k not in sel_sheets and k not in live_sheets]
    for key in sheet_keys or []:
        if key not in groups:
            raise HTTPException(status_code=404, detail=f"no sheet group {key!r} in the storyboard")
        if key in live_sheets:
            raise HTTPException(status_code=409, detail=f"the {key!r} sheet is already being generated")
        if key not in want_sheets:
            want_sheets.append(key)
    if not want_shots and not want_sheets:
        raise HTTPException(status_code=409, detail="nothing to generate")

    aspect = jobs.aspect_ratio_for(project.width, project.height)
    batch_id = str(uuid.uuid4())
    est = spend_mod.image_credits(KEYFRAME_RESOLUTION)
    shot_text = (prompt or "").strip() or shot_prompt(result)
    created: list[RebuildKeyframe] = []
    for s in want_shots:
        created.append(RebuildKeyframe(
            id=str(uuid.uuid4()), project_id=project.id, batch_id=batch_id, shot_id=s.id,
            prompt=shot_text, input_paths=_inputs(project.id, [s]),
            resolution=KEYFRAME_RESOLUTION, aspect_ratio=aspect, estimate_credits=est,
            status="queued", dispatch_gen=1,
        ))
    for key in want_sheets:
        members = groups[key].get("characters") or []
        picked = None
        if sheet_inputs and sheet_inputs.get(key):
            picked = []
            for sid in sheet_inputs[key][:SHEET_MAX_INPUTS]:
                s = by_id.get(sid)
                if s is None:
                    raise HTTPException(status_code=404, detail=f"shot {sid} is not a shot of this project")
                picked.append(s)
        else:
            picked = sheet_input_shots(shots, key, members)
        if not picked:
            raise HTTPException(status_code=409, detail=(
                f"no shot of the {key!r} group to take the sheet's frames from — pick them"))
        created.append(RebuildKeyframe(
            id=str(uuid.uuid4()), project_id=project.id, batch_id=batch_id, shot_id=None,
            sheet_key=key,
            prompt=(sheet_prompt_override or "").strip() or sheet_prompt(result, key),
            input_paths=_inputs(project.id, picked), resolution=KEYFRAME_RESOLUTION,
            aspect_ratio=aspect, estimate_credits=est, status="queued", dispatch_gen=1,
        ))
    for row in created:
        db.add(row)
    db.flush()
    try:
        spend_mod.reserve(db, spend_mod.keyframes_scope(batch_id),
                          spend_mod.keyframes_credits(len(created)), kind="rebuild keyframes")
    except spend_mod.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=exc.message) from exc
    return batch_id, created


def enqueue_batch(rows: Sequence[RebuildKeyframe]) -> None:
    """Claim doorbells after the admission commit (best effort: the reaper
    re-sends a lost one)."""
    for r in rows:
        try:
            pc.enqueue_step(TABLE, r.id, int(r.dispatch_gen or 1), pc.STEP_CLAIM)
        except Exception:
            log.exception("keyframe %s: claim enqueue failed — the reaper re-sends it", r.id)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _unselect_slot(db, kf: RebuildKeyframe, now) -> None:
    conds = [RebuildKeyframe.project_id == kf.project_id, RebuildKeyframe.owner_run_id.is_(None),
             RebuildKeyframe.selected.is_(True), RebuildKeyframe.id != kf.id]
    if kf.shot_id:
        conds.append(RebuildKeyframe.shot_id == kf.shot_id)
    else:
        conds += [RebuildKeyframe.shot_id.is_(None), RebuildKeyframe.sheet_key == kf.sheet_key]
    db.execute(update(RebuildKeyframe).where(*conds).values(selected=False, updated_at=now)
               .execution_options(synchronize_session=False))


def select_keyframe(db, project: VideoProject, kf_id: str) -> RebuildKeyframe:
    """Operator pick (free): under the project row lock, ``ready`` required,
    unselect the slot then select the row. Caller commits."""
    locked = sbmod._lock_project(db, project.id)
    if locked is None:
        raise HTTPException(status_code=404, detail="project not found")
    sbmod.require_ready(locked)
    kf = db.execute(select(RebuildKeyframe).where(RebuildKeyframe.id == kf_id)
                    .with_for_update().execution_options(populate_existing=True)
                    ).scalar_one_or_none()
    if kf is None or kf.project_id != project.id or kf.owner_run_id is not None:
        raise HTTPException(status_code=404, detail="keyframe not found")
    if kf.status != PC_SUCCEEDED or not kf.image_path:
        raise HTTPException(status_code=409, detail=f"this attempt is {kf.status}, not succeeded")
    now = jobs._utcnow()
    _unselect_slot(db, kf, now)
    db.flush()
    kf.selected, kf.updated_at = True, now
    db.flush()
    return kf


def _auto_select(db, kf: RebuildKeyframe, now) -> bool:
    """Select a just-completed attempt unless the operator picked another one.

    The slot's current selection S is kept when it was a manual pick
    (``S.updated_at > S.finished_at``: a pick stamps ``updated_at`` later than
    the row finished; an automatic selection writes both at one instant) or
    when S is a NEWER attempt than *kf*.
    """
    conds = [RebuildKeyframe.project_id == kf.project_id, RebuildKeyframe.owner_run_id.is_(None),
             RebuildKeyframe.selected.is_(True), RebuildKeyframe.id != kf.id]
    if kf.shot_id:
        conds.append(RebuildKeyframe.shot_id == kf.shot_id)
    else:
        conds += [RebuildKeyframe.shot_id.is_(None), RebuildKeyframe.sheet_key == kf.sheet_key]
    cur = db.execute(select(RebuildKeyframe).where(*conds)
                     .execution_options(populate_existing=True)).scalars().first()
    if cur is not None:
        manual = cur.finished_at is not None and cur.updated_at > cur.finished_at
        if manual or (cur.created_at, cur.id) > (kf.created_at, kf.id):
            return False
    _unselect_slot(db, kf, now)
    db.flush()
    db.execute(update(RebuildKeyframe).where(RebuildKeyframe.id == kf.id)
               .values(selected=True, updated_at=now, finished_at=now)
               .execution_options(synchronize_session=False))
    return True


# ---------------------------------------------------------------------------
# Step handler (rebuild_paid_call.STEP_HANDLERS["rebuild_keyframes"])
# ---------------------------------------------------------------------------


def _scope(row) -> str:
    return spend_mod.keyframes_scope(row.batch_id)


def step(row_id: str, gen: int, step_name: str, job_id: Optional[str], *,
         sf: Optional[Callable] = None) -> Optional[str]:
    sf = sf or jobs.session_factory()
    if step_name == pc.STEP_CLAIM:
        return _claim(row_id, gen, sf)
    if step_name == pc.STEP_SEND:
        status = jobs.send_claimed_row(
            RebuildKeyframe, row_id, gen, job_id, sf=sf, scope_key_fn=_scope,
            call=lambda req: jobs.kie_client().create_task_once(
                req, timeout=pc.SEND_HTTP_TIMEOUT_SEC))
        if status == PC_SUBMITTED:
            pc.enqueue_step(TABLE, row_id, gen, pc.STEP_POLL, delay_sec=pc.POLL_DELAY_IMAGE_SEC)
        return status
    if step_name == pc.STEP_POLL:
        return _poll(row_id, gen, sf)
    if step_name == pc.STEP_FETCH:
        return _fetch(row_id, gen, sf)
    log.error("keyframe %s: unknown step %s", row_id, step_name)
    return None


def _claim(row_id: str, gen: int, sf) -> Optional[str]:
    db = sf()
    try:
        token = pc.claim(db, RebuildKeyframe, row_id, gen)
        if token is None:
            db.rollback()
            return None
        row = db.get(RebuildKeyframe, row_id)
        project = db.get(VideoProject, row.project_id)
        src = project.source_local_path if project is not None else None
        inputs = list(row.input_paths or [])
        prompt, aspect, res = row.prompt, row.aspect_ratio or "9:16", row.resolution
        db.commit()
    finally:
        db.close()

    try:
        if not src or not os.path.exists(src):
            raise RuntimeError(f"source file not found at {src!r}")
        if not inputs:
            raise RuntimeError("no input frame")
        urls = []
        plain = row_id.replace("-", "")
        for i, item in enumerate(inputs):
            path = item["path"] if isinstance(item, dict) else str(item)
            if isinstance(item, dict):
                jobs.extract_frame(src, float(item["t"]), path)
            urls.append(jobs.upload(path, f"kf{plain}n{i}.jpg"))
        from .kie_client import build_image_payload

        request = build_image_payload(prompt=prompt, input_urls=urls, aspect_ratio=aspect,
                                      resolution=res, model=KEYFRAME_MODEL)
    except Exception as exc:  # noqa: BLE001 — free work failed: nothing sent
        log.exception("keyframe %s: input preparation failed", row_id)
        db = sf()
        try:
            jobs.fail_unsent(db, RebuildKeyframe, row_id, gen=gen, token=token,
                             error=f"{type(exc).__name__}: {exc}")
            db.commit()
        finally:
            db.close()
        return "failed"

    db = sf()
    try:
        if not jobs.store_prepared_request(db, RebuildKeyframe, row_id, gen=gen, token=token,
                                           request=request):
            db.rollback()
            return None
        db.commit()
    finally:
        db.close()
    pc.enqueue_step(TABLE, row_id, gen, pc.STEP_SEND)
    return "claimed"


def _credits(data: dict) -> Optional[float]:
    raw = data.get("creditsConsumed")
    try:
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return None


def _result_url(data: dict) -> Optional[str]:
    try:
        urls = json.loads(data.get("resultJson") or "{}").get("resultUrls") or []
    except (TypeError, ValueError, AttributeError):
        urls = []
    return urls[0] if urls else None


def _poll(row_id: str, gen: int, sf) -> Optional[str]:
    db = sf()
    try:
        row = db.get(RebuildKeyframe, row_id)
        if row is None or row.status != PC_SUBMITTED or int(row.dispatch_gen) != int(gen):
            return None
        task_id, scope = row.task_id, _scope(row)
        sent_at = (row.send_deadline - timedelta(seconds=pc.SEND_WINDOW_SEC)
                   if row.send_deadline else row.updated_at)
    finally:
        db.close()
    try:
        data = jobs.kie_client().get_task(task_id) or {}
    except Exception:  # noqa: BLE001 — polling is idempotent: poll again (bounded below)
        log.warning("keyframe %s: recordInfo failed", row_id, exc_info=True)
        data = {}
    state = (data.get("state") or "").lower()
    if state not in ("success", "fail") and \
            (jobs._utcnow() - sent_at).total_seconds() > POLL_MAX_SEC:
        state, err = "fail", f"no result after {POLL_MAX_SEC // 60} min; task {task_id}"
    else:
        err = str(data.get("failMsg") or data.get("failCode") or "provider task failed")
    db = sf()
    try:
        status = pc.record_poll(db, RebuildKeyframe, row_id, gen=gen, state=state,
                                scope_key=scope, credits_consumed=_credits(data), error=err)
        db.commit()
    finally:
        db.close()
    if status == PC_FETCHING:
        pc.enqueue_step(TABLE, row_id, gen, pc.STEP_FETCH)
    elif status == PC_SUBMITTED:
        pc.enqueue_step(TABLE, row_id, gen, pc.STEP_POLL, delay_sec=pc.POLL_DELAY_IMAGE_SEC)
    return status


def _verify_image(path: str) -> tuple[str, int, int]:
    from PIL import Image

    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        width, height = im.size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), int(width), int(height)


def _fetch(row_id: str, gen: int, sf) -> Optional[str]:
    from .storage import rebuild_keyframe_path

    db = sf()
    try:
        row = db.get(RebuildKeyframe, row_id)
        if row is None or row.status != PC_FETCHING or int(row.dispatch_gen) != int(gen):
            return None
        task_id, pid, scope = row.task_id, row.project_id, _scope(row)
    finally:
        db.close()
    dst = rebuild_keyframe_path(pid, row_id)
    try:
        url = _result_url(jobs.kie_client().get_task(task_id) or {})
        if not url:
            raise RuntimeError("the task has no result URL")
        jobs.kie_client().download_result(url, dst)  # .part + os.replace inside
        sha, width, height = _verify_image(dst)
    except Exception as exc:  # noqa: BLE001 — retry the download only, never the create
        log.warning("keyframe %s: fetch failed: %s", row_id, exc)
        db = sf()
        try:
            out = pc.fail_fetch(db, RebuildKeyframe, row_id, gen=gen, scope_key=scope,
                                error=f"{type(exc).__name__}: {exc}")
            db.commit()
        finally:
            db.close()
        if out == "retry":
            pc.enqueue_step(TABLE, row_id, gen, pc.STEP_FETCH, delay_sec=FETCH_RETRY_DELAY_SEC)
        return out

    db = sf()
    try:
        sbmod._lock_project(db, pid)  # selection is serialised on the project row
        now = jobs._utcnow()
        if not pc.complete_fetch(db, RebuildKeyframe, row_id, gen=gen, scope_key=scope, now=now,
                                 values={"image_path": dst, "image_sha256": sha,
                                         "width": width, "height": height}):
            db.rollback()
            return None
        kf = db.execute(select(RebuildKeyframe).where(RebuildKeyframe.id == row_id)
                        .execution_options(populate_existing=True)).scalar_one()
        _auto_select(db, kf, now)
        db.commit()
        return PC_SUCCEEDED
    finally:
        db.close()
