"""Deploy admission gate (AICP-350; docs/assistant-drain-check-design.md §2-§3).

One singleton row, ``ops_admission_gate`` (id = 1), says whether NEW work may be
admitted. It is written only by the plain SQL files in ``deploy/ops-gate/``
(``begin`` / ``end`` / ``force-end``), so pausing works while the API is down
or on an old image; this module only reads it.

The barrier
-----------
Every top-level admission reads the row ``FOR SHARE`` as the FIRST lock of its
transaction (:func:`require_admitting`, a route dependency, or
:func:`lock_admitting` for the automation tick / the autopilot) and keeps that
share lock until its commit. ``begin.sql`` is a short ``UPDATE`` of that row, so
on PostgreSQL it waits for every admission in flight, and every later admission
reads ``paused``: when ``begin`` returns, no admission is half-done and none can
start. Paused -> 503 + ``Retry-After``, nothing written.

Only admissions are gated, never work downstream of them (worker reserves,
polls, stitches, deliveries, Fal ``admit()``): admitted work must be able to
finish so ``active`` (``app/ops_status.py``) reaches 0.

The marker never expires (fail closed, design §2 Q3): ``scripts/deploy.sh``
ends it from an exit trap; a forgotten one turns the dashboard banner red after
``OPS_GATE_WARN_AFTER_MIN`` minutes, with the ``force-end`` command.

Route inventory
---------------
Every mutating FastAPI route is either marked gated (``dependencies=[GATED]``
on its decorator) or listed in :data:`UNGATED` with the reason. A test
(``tests/test_ops_gate_routes.py``) enumerates the app's routes and fails on a
route that is neither, or on a stale UNGATED entry.

Idempotency (design §3)
-----------------------
``api_idempotency_keys``: an admission route that accepts an
``Idempotency-Key`` header claims the key INSIDE its admission transaction,
right after the gate: ``INSERT ... ON CONFLICT DO NOTHING RETURNING``. A
concurrent duplicate waits on the first transaction's uncommitted row: it
inserts if that one rolled back, gets nothing if it committed -> read the key:
different request -> 409, same request -> replay the recorded resource
(without re-running mutable business-state validation). A rollback or a kill
removes the key with everything else, so a resend executes afresh.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy import BigInteger, Boolean, DateTime, Integer, JSON, String, Text, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from .config import settings
from .db import Base, get_db

log = logging.getLogger(__name__)

GATE_ID = 1
RETRY_AFTER_SEC = 60
PAUSED_DETAIL = (
    "Deploy in progress: new work is paused until it finishes. Nothing was "
    "started; try again in a few minutes."
)
IDEMPOTENCY_HEADER = "Idempotency-Key"
IDEMPOTENCY_KEY_MAX = 128
IDEMPOTENCY_RETENTION_DAYS = 30


class OpsAdmissionGate(Base):
    """Singleton (id = 1), inserted by migration w2c6e8a0b4d7.

    Written only by ``deploy/ops-gate/*.sql``; the app never writes it.
    """

    __tablename__ = "ops_admission_gate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    owner: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generation: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    paused_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)


class ApiIdempotencyKey(Base):
    """One claimed ``Idempotency-Key`` (design §3). Global keys; the route is
    part of the request hash, so the same key on another route is a 409."""

    __tablename__ = "api_idempotency_keys"

    key: Mapped[str] = mapped_column(String(IDEMPOTENCY_KEY_MAX), primary_key=True)
    route: Mapped[str] = mapped_column(String(255), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    operation: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc), index=True,
    )


# ---------------------------------------------------------------------------
# Gate reads
# ---------------------------------------------------------------------------


class AdmissionPaused(Exception):
    """The gate is paused: admit nothing."""

    def __init__(self, gate: "GateState"):
        super().__init__(PAUSED_DETAIL)
        self.gate = gate


@dataclass(frozen=True)
class GateState:
    paused: bool
    generation: int = 0
    owner: Optional[str] = None
    reason: Optional[str] = None
    paused_at: Optional[datetime] = None

    @property
    def admitting(self) -> bool:
        return not self.paused


_OPEN = GateState(paused=False)


def _state(row: Optional[OpsAdmissionGate]) -> GateState:
    if row is None:
        # The migration inserts the row; a SQLite test schema may not have
        # it. A missing row cannot have been paused by begin.sql (it upserts
        # the row before pausing), so it reads as open.
        return _OPEN
    return GateState(
        paused=bool(row.paused), generation=int(row.generation or 0),
        owner=row.owner, reason=row.reason, paused_at=row.paused_at,
    )


def read_gate(db: Session, *, share: bool = False) -> GateState:
    """The gate row; ``share=True`` takes it ``FOR SHARE`` (held until the
    caller's commit/rollback — the admission barrier)."""
    q = select(OpsAdmissionGate).where(OpsAdmissionGate.id == GATE_ID)
    if share:
        q = q.with_for_update(read=True)
    row = db.execute(q.execution_options(populate_existing=True)).scalar_one_or_none()
    return _state(row)


def lock_admitting(db: Session) -> GateState:
    """Take the gate share lock in *db*'s transaction; raise
    :class:`AdmissionPaused` when paused. Must be the first lock of the
    admission transaction and the caller must commit (or roll back) the
    admission in that same transaction."""
    gate = read_gate(db, share=True)
    if gate.paused:
        raise AdmissionPaused(gate)
    return gate


@contextmanager
def job_barrier(session_cm) -> Iterator[bool]:
    """For a worker job whose RUNNING state has no status the drain could count
    (Drive re-link, caption-placement scan). Opens its own session from
    *session_cm* (a ``get_session``-style context manager) and takes the gate
    share lock; yields False when paused (the caller defers the job and starts
    nothing), else yields True and HOLDS the share lock until the job is done.

    So a deploy's ``begin.sql`` waits for every such job in flight, and none
    can start after it: these jobs need no drain row and no time window.
    Lock order is unchanged — the gate row is the first lock the job holds,
    exactly as for an admission; the holding session takes no other lock."""
    with session_cm() as gate_db:
        try:
            lock_admitting(gate_db)
        except AdmissionPaused:
            gate_db.rollback()
            yield False
            return
        yield True


def paused_http_exception() -> HTTPException:
    return HTTPException(
        status_code=503, detail=PAUSED_DETAIL,
        headers={"Retry-After": str(RETRY_AFTER_SEC)},
    )


def require_admitting(db: Session = Depends(get_db)) -> GateState:
    """Route dependency (``dependencies=[GATED]``): the gate share lock is the
    first lock of the request's transaction (FastAPI resolves route-level
    dependencies before the endpoint, and ``get_db`` is cached per request so
    the endpoint gets the SAME session). Paused -> 503 + Retry-After."""
    try:
        return lock_admitting(db)
    except AdmissionPaused:
        db.rollback()
        raise paused_http_exception() from None


GATED = Depends(require_admitting)


def is_gated_dependant(dependant) -> bool:
    """True when *dependant* (a FastAPI route's dependency tree) contains
    :func:`require_admitting`."""
    stack = [dependant]
    while stack:
        d = stack.pop()
        if getattr(d, "call", None) is require_admitting:
            return True
        stack.extend(getattr(d, "dependencies", []) or [])
    return False


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def force_end_command(generation: int) -> str:
    return (
        "docker exec -i re-skin-db psql -U reskin -d reskin -v ON_ERROR_STOP=1 "
        f"-v gen={int(generation)} < deploy/ops-gate/force-end.sql"
    )


def banner(gate: GateState, *, now: Optional[datetime] = None) -> Optional[dict]:
    """What the dashboard shows while paused (None when admitting)."""
    if not gate.paused:
        return None
    now = now or datetime.now(timezone.utc)
    since = gate.paused_at
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    minutes = int((now - since).total_seconds() // 60) if since else None
    warn_after = max(int(getattr(settings, "OPS_GATE_WARN_AFTER_MIN", 45) or 45), 1)
    overdue = minutes is not None and minutes >= warn_after
    text = "Deploy in progress since {} UTC".format(since.strftime("%H:%M") if since else "?")
    if gate.owner or gate.reason:
        text += " — {}: {}".format(gate.owner or "?", gate.reason or "")
    out = {
        "level": "red" if overdue else "info",
        "text": text.rstrip(": "),
        "minutes": minutes,
        "warn_after_min": warn_after,
        "generation": gate.generation,
    }
    if overdue:
        out["command"] = force_end_command(gate.generation)
        out["command_note"] = (
            "Run it only after confirming the deploy that paused admissions is "
            "gone (no docker compose / build process on the host, its session "
            "ended)."
        )
    return out


# ---------------------------------------------------------------------------
# Idempotency-Key (design §3)
# ---------------------------------------------------------------------------


@dataclass
class IdemRequest:
    """The parsed header + canonical request fingerprint (no DB access)."""

    key: str
    route: str
    request_sha256: str


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def request_fingerprint(route: str, path_params: dict, query: list, body: Any) -> str:
    """SHA-256 of canonical JSON ``{route template, path params, sorted query,
    body with sorted keys}`` (design §3)."""
    doc = {
        "route": route,
        "path": _canonical(dict(path_params or {})),
        "query": sorted([list(map(str, kv)) for kv in (query or [])]),
        "body": _canonical(body),
    }
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _valid_key(key: str) -> bool:
    return 0 < len(key) <= IDEMPOTENCY_KEY_MAX and key.isprintable() and key == key.strip()


async def idempotency_request(request: Request) -> Optional[IdemRequest]:
    """Dependency: ``None`` without the header (behaviour unchanged), else the
    key and the request fingerprint. Refuses file parts (design §3: routes that
    take file parts stay unavailable to keyed callers)."""
    raw = request.headers.get(IDEMPOTENCY_HEADER)
    if raw is None:
        return None
    if not _valid_key(raw):
        raise HTTPException(status_code=400, detail=(
            f"{IDEMPOTENCY_HEADER} must be 1-{IDEMPOTENCY_KEY_MAX} printable "
            "characters without surrounding spaces"))
    route_obj = request.scope.get("route")
    template = getattr(route_obj, "path", None) or request.url.path
    # An included router's route carries its path WITHOUT the mount prefix;
    # recover the prefix from the concrete URL so the template is the full one.
    full = request.url.path
    try:
        concrete = template.format(**request.path_params)
    except (KeyError, IndexError, ValueError):
        concrete = None
    if concrete and full.endswith(concrete):
        template = full[: len(full) - len(concrete)] + template
    route = f"{request.method} {template}"
    ctype = (request.headers.get("content-type") or "").lower()
    body: Any = None
    if ctype.startswith("multipart/form-data") or ctype.startswith("application/x-www-form-urlencoded"):
        form = await request.form()
        items = []
        for k, v in form.multi_items():
            if not isinstance(v, str):
                if getattr(v, "filename", None):
                    raise HTTPException(status_code=400, detail=(
                        f"{IDEMPOTENCY_HEADER} is not accepted with file parts"))
                continue
            items.append([k, v])
        body = {"form": sorted(items)}
    else:
        data = await request.body()
        if data:
            try:
                body = {"json": json.loads(data)}
            except ValueError:
                body = {"raw_sha256": hashlib.sha256(data).hexdigest()}
    return IdemRequest(
        key=raw, route=route,
        request_sha256=request_fingerprint(
            route, request.path_params, list(request.query_params.multi_items()), body),
    )


class IdempotencyConflict(HTTPException):
    pass


@dataclass
class Admission:
    """What a keyed admission route gets after the gate and the key claim.

    ``replay`` is the recorded ``operation`` when this key already committed
    for the same request (the route returns the recorded resource and does
    nothing else); ``None`` means execute normally and call :meth:`record`
    before the commit.
    """

    db: Session
    idem: Optional[IdemRequest] = None
    replay: Optional[dict] = None
    _row: Optional[ApiIdempotencyKey] = field(default=None, repr=False)

    def record(self, resource_kind: str, resource_ids: list[str], **fields: Any) -> None:
        """Store the operation on the claimed key (same transaction; the key
        row is this transaction's own insert, so this never waits on a lock)."""
        if self.idem is None or self._row is None:
            return
        op = {"resource_kind": resource_kind, "resource_ids": [str(i) for i in resource_ids]}
        op.update({k: v for k, v in fields.items() if v is not None})
        self._row.operation = op
        self.db.flush()


def _insert_key(db: Session, idem: IdemRequest) -> bool:
    """``INSERT ... ON CONFLICT (key) DO NOTHING RETURNING key``; True when
    this transaction now owns the key."""
    dialect = db.get_bind().dialect.name
    values = dict(key=idem.key, route=idem.route, request_sha256=idem.request_sha256,
                  operation=None, created_at=datetime.now(timezone.utc))
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    stmt = insert(ApiIdempotencyKey).values(**values).on_conflict_do_nothing(
        index_elements=["key"]).returning(ApiIdempotencyKey.key)
    return db.execute(stmt).scalar_one_or_none() is not None


def claim_idempotency_key(db: Session, idem: Optional[IdemRequest]) -> Admission:
    """Claim *idem* in *db*'s transaction (after the gate lock). Returns an
    :class:`Admission`; raises 409 for a different request under the key."""
    if idem is None:
        return Admission(db=db)
    if _insert_key(db, idem):
        row = db.get(ApiIdempotencyKey, idem.key, populate_existing=True)
        return Admission(db=db, idem=idem, _row=row)
    row = db.get(ApiIdempotencyKey, idem.key, populate_existing=True)
    if row is None:  # pragma: no cover - deleted between the insert and the read
        raise HTTPException(status_code=409, detail=(
            f"{IDEMPOTENCY_HEADER} {idem.key!r} is being reused concurrently; retry"))
    if row.request_sha256 != idem.request_sha256:
        raise IdempotencyConflict(status_code=409, detail=(
            f"{IDEMPOTENCY_HEADER} {idem.key!r} was already used for a different "
            "request"))
    if not row.operation:
        # Committed without a recorded operation: the route did not record one
        # (a bug) — never execute twice under one key.
        raise HTTPException(status_code=409, detail=(
            f"{IDEMPOTENCY_HEADER} {idem.key!r} was used, but its result was not "
            "recorded; check the resource before sending a new key"))
    return Admission(db=db, idem=idem, replay=dict(row.operation), _row=row)


def admission(
    idem: Optional[IdemRequest] = Depends(idempotency_request),
    db: Session = Depends(get_db),
    _gate: GateState = Depends(require_admitting),
) -> Admission:
    """Route dependency for keyed admission routes: gate share lock FIRST, then
    the key claim (design §3 order). Implies GATED."""
    return claim_idempotency_key(db, idem)


ADMISSION = Depends(admission)


def keyed(adm: Any) -> Optional[Admission]:
    """The :class:`Admission` of a request that sent an ``Idempotency-Key``,
    else None — also for an in-process call (the autopilot's action handlers
    call endpoint functions directly, where the parameter's default is
    FastAPI's ``Depends`` marker, not an Admission: such a caller holds the gate
    itself and has its own exactly-once effect row)."""
    if isinstance(adm, Admission) and adm.idem is not None:
        return adm
    return None


def replay_gone(op: dict) -> HTTPException:
    """410 for a replay whose recorded resource no longer exists."""
    return HTTPException(status_code=410, detail={
        "message": "The resource this Idempotency-Key created no longer exists",
        "resource_kind": op.get("resource_kind"),
        "resource_ids": op.get("resource_ids") or [],
    })


def replay_one(adm: Admission, model, render: Callable[[Any], Any]):
    """Common replay: load the single recorded resource of *model* and render
    it, or 410."""
    op = adm.replay or {}
    ids = op.get("resource_ids") or []
    obj = adm.db.get(model, ids[0]) if ids else None
    if obj is None:
        raise replay_gone(op)
    return render(obj)


# resource_kind -> (model import path, terminal status values). A key is kept
# while its resource can still change; see purge_idempotency_keys.
_KEY_RESOURCES = {
    "run_caption": ("app.models:RunCaption", ("done", "failed")),
    "run_tail_edit": ("app.models:RunTailEdit", ("done", "failed")),
    "run": ("app.models:Run", ("done", "failed")),
}


def _terminal_or_gone(db: Session, op: Optional[dict], cutoff: datetime,
                      created_at: Optional[datetime] = None) -> bool:
    """True when every recorded resource is terminal AND unchanged since
    *cutoff* (retention counts from the resource's last change, i.e. its
    terminal transition, not from the key's creation). A deleted resource's key
    is kept."""
    if not op:
        return True
    spec = _KEY_RESOURCES.get(op.get("resource_kind"))
    if spec is None:
        return False  # unknown kind: keep (fail safe)
    import importlib

    mod, name = spec[0].split(":")
    model = getattr(importlib.import_module(mod), name)
    for rid in op.get("resource_ids") or []:
        obj = db.get(model, rid)
        if obj is None:
            # Deleted: when it finished is unknown, so the key is never purged
            # (a replay keeps answering 410 instead of executing again).
            return False
        status = getattr(obj.status, "value", obj.status)
        if status not in spec[1]:
            return False
        changed = getattr(obj, "updated_at", None)
        if changed is not None:
            if changed.tzinfo is None:
                changed = changed.replace(tzinfo=timezone.utc)
            if changed >= cutoff:
                return False
    return True


def purge_idempotency_keys(db: Session, *, now: Optional[datetime] = None,
                           days: int = IDEMPOTENCY_RETENTION_DAYS, limit: int = 1000) -> int:
    """Delete keys older than *days* whose recorded resource is terminal and
    unchanged for *days* (design §3 retention: "as long as its
    flow is not terminal, then 30 days" — stage 1 has no flow journal, so the
    resource stands in for it; the assistant journal can tighten this when it
    lands). Returns the count."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    gone = 0
    after: Optional[tuple] = None
    while True:
        # Keyset pagination over ALL expired keys: kept ones (live or deleted
        # resources) never starve later eligible keys out of a fixed batch.
        q = select(ApiIdempotencyKey).where(ApiIdempotencyKey.created_at < cutoff)
        if after is not None:
            q = q.where((ApiIdempotencyKey.created_at > after[0])
                        | ((ApiIdempotencyKey.created_at == after[0])
                           & (ApiIdempotencyKey.key > after[1])))
        rows = db.execute(q.order_by(ApiIdempotencyKey.created_at, ApiIdempotencyKey.key)
                          .limit(limit)).scalars().all()
        if not rows:
            break
        for row in rows:
            created = row.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if _terminal_or_gone(db, row.operation, cutoff, created):
                db.delete(row)
                gone += 1
        after = (rows[-1].created_at, rows[-1].key)
        db.flush()
        if len(rows) < limit:
            break
    db.commit()
    return gone


# ---------------------------------------------------------------------------
# Ungated mutating routes, with the reason (design §2: a rule, not a hand
# list — every route that creates/reactivates counted work or enqueues a job
# is gated; everything below does neither, or must stay usable during a drain).
# Keyed by "<module>.<function>" of the endpoint. tests/test_ops_gate_routes.py
# fails on an unmarked route and on a stale entry here.
# ---------------------------------------------------------------------------

_EDIT = "edits settings/rows; creates no counted work and enqueues no job"
_DELETE = "deletion of an inactive resource; no new work"
_LIBRARY = "library management (synchronous, no job, no paid call)"
_DRAFT = "edits a draft; the job is admitted by its gated submit/render route"
_KB = "knowledge-base bookkeeping (no job, no paid call)"
_AUTOPILOT = ("autopilot settings/answers; the autopilot's launches are admitted by its "
              "ticker, which takes the gate (autopilot.core._gate)")
_RESOLVE = "resolves or cancels admitted work; must stay usable while draining"

UNGATED: dict[str, str] = {
    # api_v2
    "api_v2.update_project": _EDIT,
    "api_v2.delete_project": _DELETE,
    "api_v2.update_project_segments": _EDIT,
    "api_v2.recut_project_at_pauses": "re-cuts segment defs synchronously from the saved transcript; no job",
    "api_v2.stage_project_frame": "extracts one frame synchronously into the project's staging dir; no job",
    "api_v2.update_project_transcript": _EDIT,
    "api_v2.create_library_clip": _LIBRARY,
    "api_v2.update_library_clip": _LIBRARY,
    "api_v2.delete_library_clip": _LIBRARY,
    "api_v2.create_voice_clip": _LIBRARY,
    "api_v2.update_voice_clip": _LIBRARY,
    "api_v2.delete_voice_clip": _LIBRARY,
    "api_v2.update_run": _EDIT,
    "api_v2.delete_run": _DELETE,
    "api_v2.delete_preprocess_ref": _EDIT,
    "api_v2.patch_run_segment": _EDIT,
    "api_v2.apply_segment_image_job": "applies a finished image job to the segment row; no job",
    "api_v2.clear_segment_generated_image": _EDIT,
    "api_v2.delete_tail_edit": _DELETE,
    "api_v2.update_caption_lines": _EDIT,
    "api_v2.delete_caption": _DELETE,
    "api_v2.put_run_automation": "automation settings; its launches are gated in the automation tick",
    # api_assistant (AICP-261 run chat)
    "api_assistant.cancel_turn": _RESOLVE,
    "api_assistant.choose_version": _EDIT,
    "api_assistant.import_run": "imports an existing run into its chat (rows + local copies); no job",
    "api_assistant.import_sources": "imports existing done runs as chat sources; no job",
    "api_assistant.import_caption": "imports an existing caption as a chat source; no job",
    "api_assistant.publish_run_result": ("promotes an already rendered version as the run result; "
                                          "reconcile/retry of the same saved request must work while draining"),
    # knowledge
    "api.create_rule": _KB,
    "api.add_version": _KB,
    "api.review": _KB,
    "api.post_feedback": _KB,
    "api.post_correction": _KB,
    "api.post_retract": _KB,
    "api.post_start": _KB,
    "api.post_external_repair": _KB,
    "web.knowledge_start": _KB,
    "web.knowledge_feedback": _KB,
    "web.knowledge_correction": _KB,
    "web.knowledge_retract": _KB,
    "web.knowledge_external_repair": _KB,
    # reframe
    "api_reframe.reframe_task_action": ("operator close/resume/adopt of an existing Fal task "
                                         "(parked-work resolution); must stay usable while draining"),
    "api_reframe.delete_reframe_job": _DELETE,
    "api_reframe.delete_reframe_render": _DELETE,
    # rebuild
    "api_rebuild.cancel_storyboard": _RESOLVE,
    "api_rebuild.edit_shot": _EDIT,
    "api_rebuild.keep_brand_cards": _EDIT,
    "api_rebuild.select_keyframe": _EDIT,
    # segment fill
    "api_segment_fill.cancel_fill": _RESOLVE,
    # autopilot
    "api.patch_autopilot": _AUTOPILOT,
    "api.switch_autopilot": _AUTOPILOT,
    "api.answer_approval": _AUTOPILOT,
    "api.reopen_approval": _AUTOPILOT,
    "api.comment_approval": _AUTOPILOT,
    "api.autopilot_card_answer": _AUTOPILOT,
    "api.autopilot_card_reopen": _AUTOPILOT,
    "api.autopilot_card_switch": _AUTOPILOT,
    "api.autopilot_run_card_answer": _AUTOPILOT,
    "api.autopilot_run_card_reopen": _AUTOPILOT,
    "api.autopilot_run_card_switch": _AUTOPILOT,
    # overlays
    "api_overlay.create_library_asset": _LIBRARY,
    "api_overlay.delete_library_asset": _LIBRARY,
    "api_overlay.create_overlay_job": _DRAFT,
    "api_overlay.upload_source_video": _DRAFT,
    "api_overlay.upload_asset": _DRAFT,
    "api_overlay.add_asset_from_drive": _DRAFT,
    "api_overlay.attach_library_asset": _DRAFT,
    "api_overlay.replace_overlays": _DRAFT,
    "api_overlay.delete_overlay_job": _DELETE,
}


def endpoint_name(fn) -> str:
    return f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}"


def build_started_at() -> datetime:
    return _STARTED_AT


_STARTED_AT = datetime.now(timezone.utc)
_CODE_SHA: Optional[str] = None


def build_info() -> dict:
    """``{revision, started_at, code_sha256}``: revision from
    ``RESKIN_BUILD_REVISION`` when the deploy exported it, plus a fingerprint
    of the baked app code (comparable with a checkout via
    ``python -m app.ops_status fingerprint``)."""
    global _CODE_SHA
    if _CODE_SHA is None:
        _CODE_SHA = code_fingerprint()
    return {
        "revision": os.environ.get("RESKIN_BUILD_REVISION") or None,
        "started_at": _STARTED_AT.isoformat(),
        "code_sha256": _CODE_SHA,
    }


def code_fingerprint(root: Optional[str] = None) -> str:
    root = root or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            h.update(os.path.relpath(path, root).encode())
            try:
                with open(path, "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
            except OSError:
                continue
    return h.hexdigest()[:16]
