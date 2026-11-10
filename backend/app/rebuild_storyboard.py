"""Storyboard of a rebuild project (design §5.2) — Gemini on kie.ai's chat route.

A storyboard is one ``project_storyboards`` row (the AICP-365 dispatch
protocol: ``queued -> running -> succeeded | failed | interrupted``) driven by
a chain of short **advance** jobs on the ``process`` queue. Every Gemini call
is its own ``rebuild_llm_calls`` row under the paid-call protocol (§7.2),
driven by its own step jobs (``app/rebuild_llm_steps.py`` dispatches kinds
``global`` / ``shot`` to :func:`llm_step`). The advance job never calls a
provider; it only admits call rows, observes them and finally applies the
result:

1. **Global call** (one): a low-res proxy of the whole source -> ``style``,
   ``characters``, ``sheet_groups`` (one per outfit/setting), ``audio``,
   ``on_screen_text``.
2. **Per-shot calls** (one per unfolded shot, at most :data:`SLICE_LIVE_MAX`
   live at a time): the shot's own sub-clip + its frame at ``frame_sec`` +
   the global result as context. Aligned with the detected boundaries by
   construction (the pilot's whole-video and per-chunk storyboards drifted by
   ±1 shot).

**Fallback.** A call may have at most :data:`MAX_ROWS_PER_CALL` rows per
storyboard: the primary model of the chain (``LOCALISATION_TRANSCRIBE_MODEL``
then ``..._FALLBACK_MODELS``) and ONE fallback with the next model, admitted as
a NEW row when the first was refused (``file_unavailable`` — gemini-3.1-pro's
"Failed to get the file information" — or any other refusal) or answered
something invalid. An ``unknown`` call (may have been billed) is never
re-called automatically; the operator's next Build calls it again.

**Budget.** Admission reserves ``(calls needed) x 2 x llm_credits()`` under
``spend.storyboard_scope(<id>)`` (stored as ``budget_credits``); the advance
job admits a call row only while the committed total (every row of this
storyboard that drew or will draw) plus one more call stays within it.

**Interruption** (operator cancel, or the reaper finding the advance chain
dead) moves the storyboard to ``interrupted`` and, in the SAME transaction,
cancels every call row of it still ``queued``/``claimed``
(``rebuild_paid_call.cancel_unsent``). Rows already ``sending`` keep their
§7.2 life; their answers are reused by the next Build (Resume).

**Resume / From scratch** (Build again) — see :func:`admit_storyboard`.

**Applying.** On success the shot rows' storyboard fields are filled, except
shots the operator edited (``edited_at``) unless ``overwrite_edits``.
``audio.kind`` = ``on_camera_dialogue``/``mixed`` (or any shot speaking on
camera) adds the on-camera-speech warning (Q7: warn, never block).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import timedelta
from typing import Callable, Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from . import rebuild_analysis
from . import rebuild_jobs as jobs
from . import rebuild_paid_call as pc
from . import spend as spend_mod
from .config import settings
from .localisation_jobs_models import LIVE_JOB_STATUSES
from .models import VideoProject
from .rebuild_models import (
    PAID_CALL_LIVE_STATUSES,
    PC_FAILED,
    PC_SUCCEEDED,
    PC_UNKNOWN,
    ProjectStoryboard,
    RebuildLlmCall,
    RebuildShot,
)
from .state_machine import ProjectStatus

log = logging.getLogger(__name__)

TABLE = RebuildLlmCall.__tablename__
KIND_GLOBAL = "global"
KIND_SHOT = "shot"

# At most this many shot calls of one storyboard are live at once (one
# "slice" of the design: up to 8 shot calls per advance).
SLICE_LIVE_MAX = 8
# Rows per call per storyboard: the primary model and one fallback.
MAX_ROWS_PER_CALL = 2
# The advance chain re-schedules itself this often while calls are in flight.
CONTINUE_DELAY_SEC = 5
ADVANCE_JOB_TIMEOUT_SEC = 300
JOB_PREFIX = "rb-sb"
NOT_LIVE_GRACE_SEC = 120
# The storyboard fails if more than this share of its shots stays undescribed.
MAX_FAILED_SHOT_SHARE = 0.10
# One chat/completions call with a video proxy attached.
LLM_TIMEOUT_SEC = {KIND_GLOBAL: 300.0, KIND_SHOT: 240.0}

AUDIO_KINDS = ("voiceover", "song", "on_camera_dialogue", "mixed")
SPEECH_AUDIO_KINDS = ("on_camera_dialogue", "mixed")
SPEECH_WARNING = (
    "This video has on-camera speech; phase 1 keeps the original audio and does "
    "not lip-sync (you may still proceed)."
)
SHOTS_CHANGED = "the shot list changed while the storyboard ran; build it again"
BUDGET_USED_UP = "budget for this storyboard used up; build again to continue"
UNKNOWN_CALL = ("the call may have been billed (no answer recorded); Build again "
                "calls it again")
INTERRUPTED = ("interrupted: the storyboard job stopped (e.g. a deploy). Calls already "
               "answered are kept — Build again (Resume) continues from them.")
CANCELLED = "cancelled by the operator"


# ---------------------------------------------------------------------------
# Model chain
# ---------------------------------------------------------------------------


def model_chain() -> list[str]:
    """``LOCALISATION_TRANSCRIBE_MODEL`` then the configured fallbacks."""
    out = [settings.LOCALISATION_TRANSCRIBE_MODEL]
    for m in (settings.LOCALISATION_TRANSCRIBE_FALLBACK_MODELS or "").split(","):
        m = m.strip()
        if m and m not in out:
            out.append(m)
    return [m for m in out if m]


def next_model(model: Optional[str]) -> Optional[str]:
    chain = model_chain()
    if model not in chain:
        return chain[0] if chain else None
    i = chain.index(model)
    return chain[i + 1] if i + 1 < len(chain) else None


# ---------------------------------------------------------------------------
# Prompts + tolerant parsing / validation
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json|jsonc)?\s*\n(.*?)\n?\s*```\s*$", re.S | re.I)
_KEY_RE = re.compile(r"[^a-z0-9_-]+")
FRAMINGS = ("extreme close-up", "close-up", "medium close-up", "medium shot",
            "medium wide", "wide shot", "extreme wide")


class InvalidAnswer(ValueError):
    """The model answered, but not a usable storyboard object."""


def global_prompt(shot_starts: list[float], duration: float) -> str:
    starts = ", ".join(f"{t:.2f}" for t in shot_starts)
    return (
        "You are preparing a shot-by-shot remake of this short video. Watch the whole "
        "video and listen to its soundtrack. Answer with ONE JSON object and nothing else, "
        "with exactly these keys:\n"
        '- "style": one sentence on the visual style (e.g. "3D animated cartoon, soft '
        'lighting, saturated colours").\n'
        '- "characters": the recurring characters, each {"id": "C1", "name_hint": a short '
        'label, "look": face, hair, skin tone, body, "outfit": what they wear when first '
        'seen, "role": "main" | "secondary" | "background"}. Use ids C1, C2, ... in order of '
        "appearance; [] when nobody appears.\n"
        '- "sheet_groups": one entry per distinct outfit/setting combination the characters '
        'appear in, each {"key": a short lowercase slug such as "street" or "wedding", '
        '"description": one sentence, "characters": [ids]}. At least one group when there '
        "are characters.\n"
        '- "audio": {"kind": "voiceover" | "song" | "on_camera_dialogue" | "mixed", '
        '"lines": [short notes of what is said or sung]}. Use "on_camera_dialogue" when a '
        'visible character\'s lips speak the words, "mixed" when both happen.\n'
        '- "on_screen_text": every burned-in caption, title or logo text you can read, as '
        "strings.\n"
        f"The video is {duration:.2f} s long and has {len(shot_starts)} shots between hard "
        f"cuts, starting at these seconds: {starts}."
    )


def shot_prompt(shot: RebuildShot, context: dict) -> str:
    ctx = {k: context.get(k) for k in ("style", "characters", "sheet_groups")}
    keys = [g["key"] for g in context.get("sheet_groups") or []]
    return (
        f"This is ONE shot of a video: its own clip, and a still of it at "
        f"{float(shot.frame_sec):.2f} s. In the full video it runs from "
        f"{float(shot.start_sec):.2f} s to {float(shot.end_sec):.2f} s.\n"
        f"Context for the whole video (JSON): {json.dumps(ctx, ensure_ascii=False)}\n"
        "Describe ONLY this shot. Answer with ONE JSON object and nothing else:\n"
        '{"location": where it happens, '
        f'"framing": one of {json.dumps(list(FRAMINGS))}, '
        '"camera": the camera MOVEMENT in words (e.g. "slow push-in", "pan left", '
        '"handheld follow", "tilt up"); say "static" only when the camera is truly locked '
        'off, '
        '"characters": [ids from the context who are visible in this shot], '
        '"action": one or two sentences of what happens, '
        '"speaking_on_camera": the id of a character whose lips visibly speak, else null, '
        '"is_brand_card": true when this is an end card, logo, app screen or title card '
        "with no story action, "
        f'"sheet_key": the key of the outfit/setting group this shot shows, one of '
        f"{json.dumps(keys)} (null when none of them fits this shot, when no character "
        "is visible, or when there are none)}}"
    )


def _parse_object(raw: Optional[str]) -> dict:
    text = (raw or "").strip()
    if not text:
        raise InvalidAnswer("empty answer")
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    candidates = [text]
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        candidates.append(text[a:b + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    raise InvalidAnswer(f"no JSON object in the answer: {text[:200]!r}")


def _s(value) -> str:
    return " ".join(str(value or "").split())


def _key(value) -> str:
    return _KEY_RE.sub("-", _s(value).lower()).strip("-")[:64]


def validate_global(obj: dict) -> dict:
    """The global answer, normalised; raises :class:`InvalidAnswer`."""
    if not isinstance(obj, dict):
        raise InvalidAnswer("global answer is not an object")
    style = _s(obj.get("style"))
    if not style:
        raise InvalidAnswer("global answer has no style")
    chars, ids = [], []
    for c in obj.get("characters") or []:
        if not isinstance(c, dict):
            raise InvalidAnswer("a character is not an object")
        cid = _s(c.get("id"))
        if not cid or cid in ids:
            raise InvalidAnswer(f"character id {cid!r} missing or repeated")
        ids.append(cid)
        chars.append({"id": cid, "name_hint": _s(c.get("name_hint")) or cid,
                      "look": _s(c.get("look")), "outfit": _s(c.get("outfit")),
                      "role": _s(c.get("role"))})
    groups, keys = [], []
    for g in obj.get("sheet_groups") or []:
        if not isinstance(g, dict):
            raise InvalidAnswer("a sheet group is not an object")
        key = _key(g.get("key"))
        if not key or key in keys:
            raise InvalidAnswer(f"sheet group key {g.get('key')!r} missing or repeated")
        members = [_s(x) for x in g.get("characters") or []]
        bad = [m for m in members if m not in ids]
        if bad:
            raise InvalidAnswer(f"sheet group {key!r} names unknown characters {bad}")
        keys.append(key)
        groups.append({"key": key, "description": _s(g.get("description")),
                       "characters": members})
    if chars and not groups:
        raise InvalidAnswer("characters but no sheet group")
    audio = obj.get("audio") if isinstance(obj.get("audio"), dict) else {}
    kind = _s(audio.get("kind")).lower()
    if kind not in AUDIO_KINDS:
        raise InvalidAnswer(f"audio.kind {audio.get('kind')!r} is not one of {AUDIO_KINDS}")
    lines = [_s(x) for x in audio.get("lines") or [] if _s(x)]
    texts = [_s(x) for x in obj.get("on_screen_text") or [] if _s(x)]
    return {"style": style, "characters": chars, "sheet_groups": groups,
            "audio": {"kind": kind, "lines": lines}, "on_screen_text": texts}


def validate_shot(obj: dict, context: dict) -> dict:
    """One shot answer, checked against the global result (character ids and
    ``sheet_key`` must exist in it); raises :class:`InvalidAnswer`."""
    if not isinstance(obj, dict):
        raise InvalidAnswer("shot answer is not an object")
    ids = [c["id"] for c in context.get("characters") or []]
    keys = [g["key"] for g in context.get("sheet_groups") or []]
    chars = [_s(x) for x in obj.get("characters") or []]
    bad = [c for c in chars if c not in ids]
    if bad:
        raise InvalidAnswer(f"unknown character ids {bad}")
    speaking = obj.get("speaking_on_camera")
    speaking = _s(speaking) or None
    if speaking in ("null", "none", "None"):
        speaking = None
    if speaking is not None and speaking not in ids:
        raise InvalidAnswer(f"speaking_on_camera {speaking!r} is not a character id")
    sheet = obj.get("sheet_key")
    sheet = _key(sheet) if sheet not in (None, "", "null") else None
    if not keys:
        sheet = None
    elif sheet is not None and sheet not in keys:
        # A null key is a valid answer: a shot in a setting no group covers
        # (a night street, a car, a logo) or with no character at all. The
        # chunk's sheet then comes from its other shots or the first group
        # with a selected sheet (rebuild_launch.resolve_sheet). An invented
        # key is still refused, so the fallback model gets a try.
        raise InvalidAnswer(f"sheet_key {obj.get('sheet_key')!r} is not one of {keys}")
    action = _s(obj.get("action"))
    if not action:
        raise InvalidAnswer("shot answer has no action")
    brand = obj.get("is_brand_card")
    if isinstance(brand, str):
        brand = brand.strip().lower() in ("true", "yes", "1")
    return {"location": _s(obj.get("location")), "framing": _s(obj.get("framing")),
            "camera": _s(obj.get("camera")), "characters": chars, "action": action,
            "speaking_on_camera": speaking, "is_brand_card": bool(brand),
            "sheet_key": sheet}


def _content(row: RebuildLlmCall) -> Optional[str]:
    body = row.response or {}
    try:
        msg = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(msg, list):  # content parts
        msg = "".join(p.get("text", "") for p in msg if isinstance(p, dict))
    return msg if isinstance(msg, str) else None


def parse_global_row(row: Optional[RebuildLlmCall]) -> tuple[Optional[dict], Optional[str]]:
    """``(result, None)`` for a succeeded, valid global call, else ``(None, why)``."""
    if row is None:
        return None, "no global call"
    if row.status != PC_SUCCEEDED:
        return None, row.error or row.status
    try:
        return validate_global(_parse_object(_content(row))), None
    except InvalidAnswer as exc:
        return None, f"invalid answer: {exc}"


def parse_shot_row(row: RebuildLlmCall, context: dict) -> tuple[Optional[dict], Optional[str]]:
    if row.status != PC_SUCCEEDED:
        return None, row.error or row.status
    try:
        return validate_shot(_parse_object(_content(row)), context), None
    except InvalidAnswer as exc:
        return None, f"invalid answer: {exc}"


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def latest_storyboard(db, project_id: str) -> Optional[ProjectStoryboard]:
    return db.execute(select(ProjectStoryboard).where(
        ProjectStoryboard.project_id == project_id,
    ).order_by(ProjectStoryboard.created_at.desc(), ProjectStoryboard.id)).scalars().first()


def live_storyboard(db, project_id: str) -> Optional[ProjectStoryboard]:
    return db.execute(select(ProjectStoryboard).where(
        ProjectStoryboard.project_id == project_id,
        ProjectStoryboard.status.in_(LIVE_JOB_STATUSES),
    )).scalars().first()


def applied_storyboard(db, project_id: str) -> Optional[ProjectStoryboard]:
    """The newest storyboard that succeeded and was applied to the project's
    CURRENT shot generation (the precondition of keyframes and of a launch)."""
    gen = rebuild_analysis.current_generation(db, project_id)
    return db.execute(select(ProjectStoryboard).where(
        ProjectStoryboard.project_id == project_id,
        ProjectStoryboard.status == "succeeded",
        ProjectStoryboard.applied_at.is_not(None),
        ProjectStoryboard.shots_generation == gen,
    ).order_by(ProjectStoryboard.applied_at.desc())).scalars().first()


def anchor_shots(db, project_id: str) -> list[RebuildShot]:
    return [s for s in rebuild_analysis.project_shots(db, project_id) if s.folded_into is None]


def storyboard_calls(db, storyboard_id: str) -> list[RebuildLlmCall]:
    return list(db.execute(select(RebuildLlmCall).where(
        RebuildLlmCall.storyboard_id == storyboard_id,
    ).order_by(RebuildLlmCall.created_at, RebuildLlmCall.id)).scalars())


def _shot_calls_under(db, global_call_id: str) -> list[RebuildLlmCall]:
    return list(db.execute(select(RebuildLlmCall).where(
        RebuildLlmCall.kind == KIND_SHOT,
        RebuildLlmCall.global_call_id == global_call_id,
    ).order_by(RebuildLlmCall.created_at, RebuildLlmCall.id)).scalars())


def storyboard_global(db, sb: ProjectStoryboard) -> tuple[Optional[RebuildLlmCall], Optional[dict]]:
    """The valid, succeeded global call a storyboard was built under: its
    ``global_call_id`` — or, when an interruption came before an advance
    copied it, the newest valid global answer among the storyboard's own
    rows (that answer was paid for; Resume reuses it)."""
    if sb.global_call_id:
        row = db.get(RebuildLlmCall, sb.global_call_id)
        ctx, _ = parse_global_row(row)
        return (row, ctx) if ctx is not None else (None, None)
    for row in reversed(storyboard_calls(db, sb.id)):
        if row.kind == KIND_GLOBAL and row.status == PC_SUCCEEDED:
            ctx, _ = parse_global_row(row)
            if ctx is not None:
                return row, ctx
    return None, None


def release_unused(db, sb: ProjectStoryboard) -> None:
    """Shrink the storyboard's reservation to what its calls drew (design §7.2
    accounting; Terra 2a round 1): called when it ends (finished or
    interrupted). Every call row of it is then terminal or already drew at
    ``sending`` (unsent ones were just cancelled), so nothing undrawn remains
    to cover; a still-sending call's draw is already in the ledger. Lock
    order: the scope lock and day locks come after the project / storyboard
    / call-row locks the caller holds, as on the draw path."""
    spend_mod.reserve_computed(db, spend_mod.storyboard_scope(sb.id), lambda _db: 0.0,
                               kind="rebuild storyboard", release_unused=True)


def described_shots(db, global_call_id: Optional[str], context: Optional[dict]) -> dict:
    """``{shot_id: result}`` of every valid shot answer made under
    *global_call_id* (any storyboard) — the results a Resume reuses."""
    if not global_call_id or not context:
        return {}
    out: dict = {}
    for row in reversed(_shot_calls_under(db, global_call_id)):  # newest first
        if row.shot_id in out or row.status != PC_SUCCEEDED:
            continue
        res, _ = parse_shot_row(row, context)
        if res is not None:
            out[row.shot_id] = res
    return out


def sending_calls(db, project_id: str) -> list[RebuildLlmCall]:
    """Storyboard calls of the project that are still live (``sending``
    included): Build again waits for them (design §5.2)."""
    return list(db.execute(select(RebuildLlmCall).where(
        RebuildLlmCall.project_id == project_id,
        RebuildLlmCall.storyboard_id.is_not(None),
        RebuildLlmCall.status.in_(PAID_CALL_LIVE_STATUSES),
    )).scalars())


def committed_credits(rows) -> float:
    """What this storyboard's call rows drew or will draw: every row except
    one that ended with nothing sent / a definite refusal (settled to 0) or was
    cancelled before send."""
    total = 0.0
    for r in rows:
        if r.status == PC_FAILED and (r.cost_credits == 0.0 or r.error == pc.CANCELLED_MESSAGE):
            continue
        total += float(r.estimate_credits or 0.0)
    return total


def speech_warning(result: Optional[dict], shots=()) -> Optional[str]:
    """The Q7 warning text when the storyboard found on-camera speech."""
    if not result:
        return None
    kind = ((result.get("audio") or {}).get("kind") or "").lower()
    if kind in SPEECH_AUDIO_KINDS:
        return SPEECH_WARNING
    if any(getattr(s, "speaking_on_camera", None) for s in shots):
        return SPEECH_WARNING
    if any((v or {}).get("speaking_on_camera") for v in (result.get("shots") or {}).values()):
        return SPEECH_WARNING
    return None


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def _lock_project(db, project_id: str) -> Optional[VideoProject]:
    """The project row, locked, with its attributes RE-READ after the lock:
    without ``populate_existing`` an object already in the session keeps the
    values loaded before the lock wait (e.g. ``ready`` while a re-analysis
    that held the lock has just committed ``analyzing``)."""
    return db.execute(select(VideoProject).where(VideoProject.id == project_id)
                      .with_for_update().execution_options(populate_existing=True)
                      ).scalar_one_or_none()


def require_ready(project: VideoProject) -> None:
    st = project.status.value if hasattr(project.status, "value") else str(project.status)
    if st != ProjectStatus.ready.value:
        raise HTTPException(status_code=409,
                            detail=f"the project is {st}, not ready — wait for the analysis")


def admit_storyboard(db, project: VideoProject, *, mode: str = "resume",
                     overwrite_edits: bool = False) -> ProjectStoryboard:
    """Admit one storyboard build in the caller's transaction (no commit; the
    caller commits, then :func:`enqueue_first`).

    Locks the project row and requires ``ready`` (design §5.1.8). 409 while a
    storyboard is live, while any previous storyboard call is still live
    ("waiting for the previous worker"), without shots or a source; 429 on
    budget.

    * ``resume`` (default): when the newest storyboard of this shot generation
      has a valid, succeeded global call, the new row copies its
      ``global_call_id`` and reuses every valid shot answer made under it;
      only the other shots are called. Otherwise the same as from scratch.
    * ``from_scratch``: a new global call; nothing reused.
    """
    if mode not in ("resume", "from_scratch"):
        raise HTTPException(status_code=400, detail="mode must be 'resume' or 'from_scratch'")
    locked = _lock_project(db, project.id)
    if locked is None:
        raise HTTPException(status_code=404, detail="project not found")
    project = locked
    require_ready(project)
    if not project.source_local_path or not os.path.exists(project.source_local_path):
        raise HTTPException(status_code=409, detail="this project has no source video on disk")
    live = live_storyboard(db, project.id)
    if live is not None:
        raise HTTPException(status_code=409, detail=f"a storyboard is already {live.status}")
    if sending_calls(db, project.id):
        raise HTTPException(status_code=409, detail=(
            "waiting for the previous worker to finish: a storyboard call is still in "
            "flight (a row that is 'sending' waits for its worker)"))
    gen = rebuild_analysis.current_generation(db, project.id)
    anchors = anchor_shots(db, project.id)
    if not anchors:
        raise HTTPException(status_code=409, detail="this project has no shots — analyze it first")

    reuse_global = None
    if mode == "resume":
        # Only the NEWEST storyboard of this shot generation defines Resume
        # (design §5.2): a failed From-scratch build must not let an older
        # storyboard's global context come back.
        prev = db.execute(select(ProjectStoryboard).where(
            ProjectStoryboard.project_id == project.id,
            ProjectStoryboard.shots_generation == gen,
        ).order_by(ProjectStoryboard.created_at.desc(), ProjectStoryboard.id)).scalars().first()
        if prev is not None:
            g, ctx = storyboard_global(db, prev)
            if ctx is not None:
                reuse_global = (g, ctx)
    described = described_shots(db, reuse_global[0].id, reuse_global[1]) if reuse_global else {}
    n_needed = sum(1 for s in anchors if s.id not in described)
    budget = ((0 if reuse_global else 1) + n_needed) * MAX_ROWS_PER_CALL \
        * spend_mod.llm_credits()

    sb = ProjectStoryboard(
        id=str(uuid.uuid4()), project_id=project.id, shots_generation=gen,
        build_mode=mode, status="queued", dispatch_gen=1, attempts=0,
        budget_credits=float(budget), overwrite_edits=bool(overwrite_edits),
    )
    if reuse_global:
        sb.global_call_id = reuse_global[0].id
        sb.model = reuse_global[0].model
    sb.rq_job_id = job_id(sb.id, 1, "0")
    db.add(sb)
    try:
        db.flush()
    except IntegrityError as exc:  # uq_project_storyboards_live
        raise HTTPException(status_code=409, detail="a storyboard is already live") from exc
    try:
        spend_mod.reserve_computed(db, spend_mod.storyboard_scope(sb.id), lambda _db: budget,
                                   kind="rebuild storyboard")
    except spend_mod.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=exc.message) from exc
    return sb


def cancel_storyboard(db, project: VideoProject, *, reason: str = CANCELLED) -> ProjectStoryboard:
    """Interrupt the project's live storyboard (operator cancel) and cancel
    its unsent calls in the SAME transaction. Caller commits."""
    locked = _lock_project(db, project.id)
    if locked is None:
        raise HTTPException(status_code=404, detail="project not found")
    sb = live_storyboard(db, project.id)
    if sb is None:
        raise HTTPException(status_code=409, detail="no storyboard is running")
    sb = db.execute(select(ProjectStoryboard).where(ProjectStoryboard.id == sb.id)
                    .with_for_update().execution_options(populate_existing=True)).scalar_one()
    if sb.status not in LIVE_JOB_STATUSES:
        raise HTTPException(status_code=409, detail=f"the storyboard already ended ({sb.status})")
    _interrupt(db, sb, reason)
    return sb


def _interrupt(db, sb: ProjectStoryboard, reason: str) -> int:
    now = jobs._utcnow()
    sb.status, sb.claim_token, sb.error = "interrupted", None, reason
    sb.finished_at, sb.updated_at, sb.not_live_since = now, now, None
    n = pc.cancel_unsent(db, RebuildLlmCall, RebuildLlmCall.storyboard_id == sb.id, now=now)
    db.flush()
    release_unused(db, sb)
    return n


# ---------------------------------------------------------------------------
# The advance chain (RQ)
# ---------------------------------------------------------------------------


def job_id(storyboard_id: str, gen: int, seq: str) -> str:
    return f"{JOB_PREFIX}:{storyboard_id}:{int(gen)}:{seq}"


def job_prefix(storyboard_id: str, gen: int) -> str:
    return f"{JOB_PREFIX}:{storyboard_id}:{int(gen)}:"


def _enqueue(storyboard_id: str, gen: int, token: Optional[str], jid: str,
             delay_sec: float = 0) -> None:
    from .tasks import QUEUE_PROCESS, _get_queue

    q = _get_queue(QUEUE_PROCESS)
    kw = dict(job_id=jid, job_timeout=ADVANCE_JOB_TIMEOUT_SEC)
    if delay_sec > 0:
        q.enqueue_in(timedelta(seconds=delay_sec), "app.rebuild_storyboard.run_storyboard_job",
                     storyboard_id, int(gen), token, **kw)
    else:
        q.enqueue("app.rebuild_storyboard.run_storyboard_job", storyboard_id, int(gen), token, **kw)


def enqueue_first(sb: ProjectStoryboard) -> None:
    """The doorbell after the admission commit. Best effort: a lost enqueue is
    re-sent by :func:`reap`."""
    try:
        _enqueue(sb.id, int(sb.dispatch_gen), None, sb.rq_job_id)
    except Exception:
        log.exception("storyboard %s: enqueue failed — the reaper re-sends it", sb.id)


def _enqueue_claims(rows: list[tuple[str, int]]) -> None:
    for row_id, gen in rows:
        try:
            pc.enqueue_step(TABLE, row_id, gen, pc.STEP_CLAIM)
        except Exception:
            log.exception("storyboard call %s: claim enqueue failed — the reaper re-sends it",
                          row_id)


def run_storyboard_job(storyboard_id: str, gen: int, token: Optional[str] = None, *,
                       sf: Optional[Callable] = None) -> Optional[str]:
    """RQ entry point of one advance. The first job (``token`` None) claims a
    ``queued`` storyboard at *gen*; a continuation is fenced by
    (``running``, *gen*, *token*). Returns the storyboard status, or None when
    fenced out."""
    sf = sf or jobs.session_factory()
    db = sf()
    claims: list[tuple[str, int]] = []
    cont = None
    try:
        sb0 = db.get(ProjectStoryboard, storyboard_id)
        if sb0 is None:
            return None
        project = _lock_project(db, sb0.project_id)
        sb = db.execute(select(ProjectStoryboard).where(ProjectStoryboard.id == storyboard_id)
                        .with_for_update().execution_options(populate_existing=True)
                        ).scalar_one_or_none()
        if sb is None or int(sb.dispatch_gen) != int(gen):
            db.rollback()
            return None
        now = jobs._utcnow()
        if token is None:
            if sb.status != "queued":
                db.rollback()
                return None
            token = uuid.uuid4().hex
            sb.status, sb.claim_token = "running", token
            sb.attempts = int(sb.attempts or 0) + 1
            sb.started_at = sb.started_at or now
        elif sb.status != "running" or sb.claim_token != token:
            db.rollback()
            return None
        sb.not_live_since, sb.updated_at = None, now
        if project is None:
            sb.status, sb.error, sb.finished_at, sb.claim_token = "failed", "project deleted", now, None
            db.commit()
            return "failed"
        claims, done = advance(db, project, sb)
        if not done:
            cont = job_id(sb.id, gen, uuid.uuid4().hex[:8])
            sb.rq_job_id = cont
        status = sb.status
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    _enqueue_claims(claims)
    if cont is not None:
        try:
            _enqueue(storyboard_id, gen, token, cont, delay_sec=CONTINUE_DELAY_SEC)
        except Exception:
            log.exception("storyboard %s: continuation enqueue failed — the reaper will "
                          "interrupt it", storyboard_id)
    return status


def _admit_call(db, sb: ProjectStoryboard, *, kind: str, model: str,
                shot: Optional[RebuildShot] = None) -> Optional[RebuildLlmCall]:
    row = RebuildLlmCall(
        id=str(uuid.uuid4()), project_id=sb.project_id, storyboard_id=sb.id, kind=kind,
        shot_id=shot.id if shot is not None else None, model=model,
        global_call_id=sb.global_call_id if kind == KIND_SHOT else None,
        status="queued", dispatch_gen=1, estimate_credits=spend_mod.llm_credits(),
    )
    sp = db.begin_nested()
    try:
        db.add(row)
        db.flush()
    except IntegrityError:  # the live-call index: already live
        sp.rollback()
        log.info("storyboard %s: %s call for %s already live", sb.id, kind,
                 shot.id if shot is not None else "global")
        return None
    sp.commit()
    return row


def advance(db, project: VideoProject, sb: ProjectStoryboard) -> tuple[list, bool]:
    """One observation + admission pass under the project and storyboard row
    locks. Returns ``(claims to enqueue after the commit, finished)``."""
    claims: list[tuple[str, int]] = []
    if int(sb.shots_generation) != rebuild_analysis.current_generation(db, project.id):
        _finish(db, project, sb, None, {}, {}, apply=False, error=SHOTS_CHANGED)
        return claims, True
    own = storyboard_calls(db, sb.id)
    budget = float(sb.budget_credits or 0.0)
    committed = committed_credits(own)
    llm = spend_mod.llm_credits()

    def affordable() -> bool:
        return committed + llm <= budget + 1e-6

    # --- the global call -------------------------------------------------
    context = None
    if sb.global_call_id:
        context, why = parse_global_row(db.get(RebuildLlmCall, sb.global_call_id))
        if context is None:
            _finish(db, project, sb, None, {}, {}, apply=False,
                    error=f"the global call cannot be used: {why}")
            return claims, True
    else:
        mine = [r for r in own if r.kind == KIND_GLOBAL]
        if any(r.status in PAID_CALL_LIVE_STATUSES for r in mine):
            return claims, False
        last = mine[-1] if mine else None
        err = None
        if last is not None and last.status == PC_SUCCEEDED:
            context, err = parse_global_row(last)
            if context is not None:
                sb.global_call_id, sb.model = last.id, last.model
        elif last is not None and last.status == PC_UNKNOWN:
            _finish(db, project, sb, None, {}, {}, apply=False,
                    error=f"global call: {UNKNOWN_CALL}")
            return claims, True
        elif last is not None:
            err = last.error or last.status
        if context is None:
            model = model_chain()[0] if last is None else next_model(last.model)
            if len(mine) >= MAX_ROWS_PER_CALL or model is None:
                _finish(db, project, sb, None, {}, {}, apply=False,
                        error=f"the global call failed: {err}")
                return claims, True
            if not affordable():
                _finish(db, project, sb, None, {}, {}, apply=False, error=BUDGET_USED_UP)
                return claims, True
            row = _admit_call(db, sb, kind=KIND_GLOBAL, model=model)
            if row is not None:
                claims.append((row.id, 1))
            return claims, False

    # --- the shot calls --------------------------------------------------
    anchors = anchor_shots(db, project.id)
    described = described_shots(db, sb.global_call_id, context)
    by_shot: dict[str, list[RebuildLlmCall]] = {}
    for r in _shot_calls_under(db, sb.global_call_id):
        by_shot.setdefault(r.shot_id, []).append(r)
    errors: dict[str, str] = {}
    inflight = 0
    pending: list[tuple[RebuildShot, str]] = []
    for shot in anchors:
        if shot.id in described:
            continue
        rows = by_shot.get(shot.id, [])
        if any(r.status in PAID_CALL_LIVE_STATUSES for r in rows):
            inflight += 1
            continue
        mine = [r for r in rows if r.storyboard_id == sb.id]
        last = mine[-1] if mine else None
        if last is not None and last.status == PC_UNKNOWN:
            errors[shot.id] = UNKNOWN_CALL
            continue
        why = None
        if last is not None:
            _, why = parse_shot_row(last, context)
        model = model_chain()[0] if last is None else next_model(last.model)
        if len(mine) >= MAX_ROWS_PER_CALL or model is None:
            errors[shot.id] = f"not described: {why or 'no model left'}"
            continue
        pending.append((shot, model))
    waiting = False
    for i, (shot, model) in enumerate(pending):
        if inflight >= SLICE_LIVE_MAX:
            waiting = True
            break
        if not affordable():
            for s, _m in pending[i:]:
                errors[s.id] = BUDGET_USED_UP
            break
        row = _admit_call(db, sb, kind=KIND_SHOT, model=model, shot=shot)
        if row is None:
            inflight += 1
            continue
        committed += llm
        inflight += 1
        claims.append((row.id, 1))
    if inflight or waiting:
        return claims, False
    _finish(db, project, sb, context, described, errors, apply=True)
    return claims, True


def _finish(db, project, sb: ProjectStoryboard, context: Optional[dict], described: dict,
            errors: dict, *, apply: bool, error: Optional[str] = None) -> None:
    now = jobs._utcnow()
    result = dict(context or {})
    result["shots"] = described
    result["errors"] = errors
    anchors = anchor_shots(db, project.id) if apply else []
    n = len(anchors)
    failed_share = (len(errors) / n) if n else 0.0
    ok = error is None and context is not None and failed_share <= MAX_FAILED_SHOT_SHARE + 1e-9
    if error is None and not ok:
        error = (f"{len(errors)} of {n} shots could not be described "
                 f"(more than {int(MAX_FAILED_SHOT_SHARE * 100)} %); build again to continue")
    warning = speech_warning(result) if context else None
    result["warnings"] = [warning] if warning else []
    sb.result = result
    sb.status = "succeeded" if ok else "failed"
    sb.error = None if ok else error
    sb.claim_token, sb.finished_at, sb.updated_at = None, now, now
    # Nothing more is admitted or used: a row of this storyboard that has not
    # been sent yet (possible only on the early-exit branches) never will be.
    pc.cancel_unsent(db, RebuildLlmCall, RebuildLlmCall.storyboard_id == sb.id, now=now)
    if ok:
        for shot in anchors:
            res = described.get(shot.id)
            if res is None:
                continue
            if shot.edited_at is not None and not sb.overwrite_edits:
                continue
            for k in ("location", "framing", "camera", "characters", "action",
                      "speaking_on_camera", "is_brand_card", "sheet_key"):
                setattr(shot, k, res[k])
            shot.storyboard_id = sb.id
            if sb.overwrite_edits:
                shot.edited_at = None
            shot.updated_at = now
        sb.applied_at = now
    db.flush()
    release_unused(db, sb)


# ---------------------------------------------------------------------------
# The per-call steps (kinds global / shot), dispatched by rebuild_llm_steps
# ---------------------------------------------------------------------------


def llm_step(row_id: str, gen: int, step: str, job_id_: Optional[str], *,
             sf: Optional[Callable] = None) -> Optional[str]:
    """Step handler of a storyboard call row (design §7.2; notes §2.1)."""
    sf = sf or jobs.session_factory()
    if step == pc.STEP_CLAIM:
        return _claim_step(row_id, gen, sf)
    if step == pc.STEP_SEND:
        def call(request: dict):
            return jobs.kie_client().chat_completion_once(
                request["model"], request["messages"], timeout=float(request["timeout"]))

        status = jobs.send_claimed_row(
            RebuildLlmCall, row_id, gen, job_id_, sf=sf, call=call,
            scope_key_fn=lambda r: spend_mod.storyboard_scope(r.storyboard_id))
        if status is not None:
            _release_if_ended(row_id, sf)
        return status
    # A chat call has no task: it never reaches submitted/fetching.
    log.error("storyboard call %s: unexpected step %s", row_id, step)
    return None


def _release_if_ended(row_id: str, sf) -> None:
    """A call that ended AFTER its storyboard ended (it was ``sending`` at an
    interruption): re-release the storyboard's reservation, so a draw settled
    below its estimate (a refusal settles to 0) frees the difference (Terra 2a
    round 2). Lock order as everywhere: project, storyboard, then the scope /
    day locks inside ``release_unused``. Best effort: a failure leaves the
    estimate reserved for the day, never a missing draw."""
    db = sf()
    try:
        row = db.get(RebuildLlmCall, row_id)
        if row is None or row.storyboard_id is None:
            return
        _lock_project(db, row.project_id)
        sb = db.execute(select(ProjectStoryboard).where(ProjectStoryboard.id == row.storyboard_id)
                        .with_for_update().execution_options(populate_existing=True)
                        ).scalar_one_or_none()
        if sb is None or sb.status in LIVE_JOB_STATUSES:
            db.rollback()
            return
        release_unused(db, sb)
        db.commit()
    except Exception:
        db.rollback()
        log.exception("storyboard call %s: reservation release failed", row_id)
    finally:
        db.close()


def _claim_step(row_id: str, gen: int, sf) -> Optional[str]:
    db = sf()
    try:
        token = pc.claim(db, RebuildLlmCall, row_id, gen)
        if token is None:
            db.rollback()
            return None
        row = db.get(RebuildLlmCall, row_id)
        project = db.get(VideoProject, row.project_id)
        shot = db.get(RebuildShot, row.shot_id) if row.shot_id else None
        context = None
        if row.kind == KIND_SHOT:
            context, _ = parse_global_row(db.get(RebuildLlmCall, row.global_call_id))
        shot_starts = [float(s.start_sec) for s in anchor_shots(db, row.project_id)]
        kind, model = row.kind, row.model
        src = project.source_local_path if project is not None else None
        vertical = jobs.aspect_ratio_for(getattr(project, "width", None),
                                         getattr(project, "height", None)) != "16:9"
        duration = float(getattr(project, "duration_sec", None) or 0.0)
        pid = row.project_id
        db.commit()
    finally:
        db.close()

    try:
        if not src or not os.path.exists(src):
            raise RuntimeError(f"source file not found at {src!r}")
        if kind == KIND_SHOT and (shot is None or context is None):
            raise RuntimeError("the shot or its global context is gone")
        request = _build_request(pid, row_id, kind, model, src, vertical, duration,
                                 shot_starts, shot, context)
    except Exception as exc:  # noqa: BLE001 — free work failed: nothing sent
        log.exception("storyboard call %s: input preparation failed", row_id)
        db = sf()
        try:
            jobs.fail_unsent(db, RebuildLlmCall, row_id, gen=gen, token=token,
                             error=f"{type(exc).__name__}: {exc}")
            db.commit()
        finally:
            db.close()
        return PC_FAILED

    db = sf()
    try:
        if not jobs.store_prepared_request(db, RebuildLlmCall, row_id, gen=gen, token=token,
                                           request=request):
            db.rollback()
            return None
        db.commit()
    finally:
        db.close()
    pc.enqueue_step(TABLE, row_id, gen, pc.STEP_SEND)
    return "claimed"


def _build_request(pid, row_id, kind, model, src, vertical, duration, shot_starts, shot,
                   context) -> dict:
    from .storage import rebuild_project_dir

    work = os.path.join(rebuild_project_dir(pid), "llm")
    os.makedirs(work, exist_ok=True)
    plain = row_id.replace("-", "")
    proxy = os.path.join(work, f"{plain}.mp4")
    parts: list[dict] = []
    try:
        if kind == KIND_GLOBAL:
            jobs.build_llm_proxy(src, proxy, vertical=vertical)
            parts.append({"type": "image_url",
                          "image_url": {"url": jobs.upload(proxy, f"sb{plain}.mp4")}})
            text = global_prompt(shot_starts, duration)
        else:
            jobs.build_llm_proxy(src, proxy, vertical=vertical, start=float(shot.start_sec),
                                 end=float(shot.end_sec))
            frame = os.path.join(work, f"{plain}.jpg")
            jobs.extract_frame(src, float(shot.frame_sec), frame, width=540, quality=4)
            parts.append({"type": "image_url",
                          "image_url": {"url": jobs.upload(proxy, f"sh{plain}.mp4")}})
            parts.append({"type": "image_url",
                          "image_url": {"url": jobs.upload(frame, f"fr{plain}.jpg")}})
            text = shot_prompt(shot, context)
    finally:
        for p in (proxy, os.path.join(work, f"{plain}.jpg")):
            try:
                os.unlink(p)
            except OSError:
                pass
    parts.append({"type": "text", "text": text})
    return {"model": model, "messages": [{"role": "user", "content": parts}],
            "stream": False, "timeout": LLM_TIMEOUT_SEC[kind]}


# ---------------------------------------------------------------------------
# Reaper (from tasks.run_reconcile_tick)
# ---------------------------------------------------------------------------


def reap(*, session_factory: Optional[Callable] = None,
         is_live: Optional[Callable[[str], bool]] = None,
         enqueue_fn: Optional[Callable] = None, now=None) -> dict:
    """One pass over live storyboards (the AICP-365 rule):

    * ``queued`` whose first job has not been live for the grace -> new
      ``dispatch_gen``, re-enqueue (nothing ran);
    * ``running`` whose advance chain has not been live for the grace ->
      ``interrupted`` and, in the same transaction, every call row of it
      still ``queued``/``claimed`` cancelled. Calls already ``sending`` keep
      their own life (the paid-call reaper).
    """
    sf = session_factory or jobs.session_factory()
    if is_live is None:
        from .tasks import QUEUE_PROCESS, _get_redis

        conn = _get_redis()

        def is_live(prefix: str) -> bool:
            return pc.step_is_live(conn, QUEUE_PROCESS, prefix)
    enqueue_fn = enqueue_fn or (lambda sb_id, gen, jid: _enqueue(sb_id, gen, None, jid))
    now = now or jobs._utcnow()
    grace = timedelta(seconds=NOT_LIVE_GRACE_SEC)
    out = {"observed": 0, "requeued": 0, "interrupted": 0}
    db = sf()
    try:
        ids = [(i, p) for (i, p) in db.execute(select(
            ProjectStoryboard.id, ProjectStoryboard.project_id).where(
            ProjectStoryboard.status.in_(LIVE_JOB_STATUSES)))]
    finally:
        db.close()
    for sb_id, pid in ids:
        db = sf()
        requeue = None
        try:
            _lock_project(db, pid)  # lock order: project, then storyboard
            sb = db.execute(select(ProjectStoryboard).where(ProjectStoryboard.id == sb_id)
                            .with_for_update(skip_locked=True)
                            .execution_options(populate_existing=True)).scalar_one_or_none()
            if sb is None or sb.status not in LIVE_JOB_STATUSES:
                db.rollback()
                continue
            if is_live(job_prefix(sb.id, sb.dispatch_gen)):
                if sb.not_live_since is not None:
                    sb.not_live_since = None
                    db.commit()
                else:
                    db.rollback()
                continue
            if sb.not_live_since is None:
                sb.not_live_since = now
                db.commit()
                out["observed"] += 1
                continue
            if now - sb.not_live_since < grace:
                db.rollback()
                continue
            if sb.status == "queued":
                sb.dispatch_gen = int(sb.dispatch_gen) + 1
                sb.rq_job_id = job_id(sb.id, sb.dispatch_gen, "0")
                sb.not_live_since, sb.updated_at = None, now
                requeue = (sb.id, int(sb.dispatch_gen), sb.rq_job_id)
                db.commit()
                out["requeued"] += 1
            else:
                n = _interrupt(db, sb, INTERRUPTED)
                db.commit()
                out["interrupted"] += 1
                log.warning("storyboard %s interrupted (advance chain gone); %d unsent calls "
                            "cancelled", sb_id, n)
        except Exception:
            db.rollback()
            log.exception("storyboard reaper: %s failed — continuing", sb_id)
        finally:
            db.close()
        if requeue is not None:
            try:
                enqueue_fn(*requeue)
            except Exception:
                log.exception("storyboard reaper: enqueue of %s failed", sb_id)
    out["released"] = sweep_unreleased(sf)
    return out


def sweep_unreleased(sf) -> int:
    """Re-release every ENDED storyboard whose reservation still exceeds its
    draws (Terra 2a round 3): the release after a late call outcome
    (:func:`_release_if_ended`) is its own transaction, and a worker dying
    between the two commits would otherwise leave that estimate charged for
    the rest of the day. Idempotent; a no-op when nothing is over-reserved."""
    from .models import KieSpendReservation

    prefix = spend_mod.storyboard_scope("")
    db = sf()
    try:
        keys = sorted({k for (k,) in db.execute(select(KieSpendReservation.scope_key).where(
            KieSpendReservation.scope_key.like(prefix + "%"),
            KieSpendReservation.reserved_mcredits > KieSpendReservation.consumed_mcredits))})
        candidates = []
        for key in keys:
            sb_id = key[len(prefix):]
            sb = db.get(ProjectStoryboard, sb_id)
            if sb is None:
                # Deleted with its project (Terra 2a round 4): deletion is
                # refused while any call is live, so nothing can draw on it.
                candidates.append((sb_id, None))
            elif sb.status not in LIVE_JOB_STATUSES:
                candidates.append((sb.id, sb.project_id))
    finally:
        db.close()
    n = 0
    for sb_id, pid in candidates:
        db = sf()
        try:
            if pid is None:
                if db.get(ProjectStoryboard, sb_id) is None:
                    spend_mod.reserve_computed(db, spend_mod.storyboard_scope(sb_id),
                                               lambda _db: 0.0, kind="rebuild storyboard",
                                               release_unused=True)
                    db.commit()
                    n += 1
                else:
                    db.rollback()
                continue
            _lock_project(db, pid)
            sb = db.execute(select(ProjectStoryboard).where(ProjectStoryboard.id == sb_id)
                            .with_for_update().execution_options(populate_existing=True)
                            ).scalar_one_or_none()
            if sb is None or sb.status in LIVE_JOB_STATUSES:
                db.rollback()
                continue
            release_unused(db, sb)
            db.commit()
            n += 1
        except Exception:
            db.rollback()
            log.exception("storyboard reaper: release of %s failed", sb_id)
        finally:
            db.close()
    return n
