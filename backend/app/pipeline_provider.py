"""Second-provider task layer — Fal (AICP-312 Stage 1, AICP-330).

The contract is docs/fal-provider-design.md (revision 11, frozen) as amended by
the AICP-330 checklist; where the two differ the checklist wins:

* B1 — every Fal job runs on its own ``fal`` queue, served by the dedicated
  ``fal-worker`` container (never by the run workers); the fetch deadline
  counts from the FIRST started attempt; output media are retained for
  ``FAL_OUTPUT_LIFECYCLE_SEC`` (24 h by default).
* B2 — ``provider`` on the spend tables (``spend.py``, ``models.py``).
* B3 — EVERY submit 4xx keeps its reservation; only the operator's audited
  ``abandon`` releases it (``close`` converts it into the draw).
* M1 — the draw backfill touches only ``draw_pending`` rows (and recovery of
  ``accepted_unrecorded``), never a released one.
* M2 — ``api_v2.delete_project`` / ``delete_run`` 409 while an active task
  belongs to them (:func:`active_task_for_project` / :func:`active_task_for_run`).
* M3 — a ``cancelling`` row whose cancel was never DELIVERED gets ``fal_cancel``
  re-enqueued by recovery whatever else is live.
* M4 — every path that writes a draw for an existing row UPDATEs the task row
  first (row lock), and only then takes the spend day lock.
* M5 — an admission unique conflict is re-read by ``(provider, operation_key)``;
  absent means the owner slot is taken → 409.

Ownership changes are ALWAYS one guarded UPDATE (:func:`_cas`) that must change
exactly one row and bumps ``version_id``. Every paid POST is preceded by a
committed ``submit_started`` marker; nothing here ever POSTs a row that is not
``queued`` under our live claim, and nothing moves a row back to ``queued``.

Session usage: jobs open their own short sessions via :data:`_sessions`
(patchable in tests); operator actions and :func:`admit` work in the caller's
session and leave the commit to the caller unless stated otherwise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

from sqlalchemy import and_, exists, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import fal_models, spend, storage
from .config import settings
from .fal_client import (
    STATUS_COMPLETED,
    STATUS_TERMINAL_FAILURE,
    FalClient,
    FalDownloadError,
    FalUploadError,
    request_urls,
)
from .provider_models import ProviderTask, ProviderTaskAction
from .state_machine import (
    PROVIDER_TASK_ACTIVE_WHERE,
    PROVIDER_TASK_DRAIN_STATUSES,
    PROVIDER_TASK_NON_TERMINAL_STATUSES,
    ProviderTaskStatus as S,
    can_transition,
)

log = logging.getLogger(__name__)

PROVIDER = "fal"
# Result downloads are bounded too (a reframe of <= 300 s at 768p).
MAX_RESULT_BYTES = 1024 * 1024 * 1024
OPERATOR_ACTIONS = ("adopt", "abandon", "close", "resume")


# ---------------------------------------------------------------------------
# Injection points (tests patch these)
# ---------------------------------------------------------------------------


def _default_sessions():
    from .db import get_session

    return get_session()


_sessions: Callable = _default_sessions


def _client() -> FalClient:
    return FalClient(settings.FAL_API_KEY)


def _enqueue(fn: str, task_id: str, *, delay_sec: int = 0) -> None:
    from . import tasks

    tasks.enqueue_fal_job(fn, task_id, delay_sec=delay_sec)


# An IDLE RQ worker heartbeats only after each blocking dequeue, which waits up
# to its TTL (RQ 1.15: DEFAULT_WORKER_TTL 420 s, dequeue timeout 405 s), so a
# healthy idle worker can show a heartbeat ~7 min old. Liveness is judged on
# that TTL plus a margin, not on recovery's 150 s busy-worker cutoff.
_FAL_WORKER_FRESH_SEC = 420 + 60


def _live_fal_workers(conn) -> list:
    from rq import Worker

    from .tasks import QUEUE_FAL

    cutoff = datetime.utcnow() - timedelta(seconds=_FAL_WORKER_FRESH_SEC)
    return [w for w in Worker.all(connection=conn)
            if QUEUE_FAL in w.queue_names() and w.last_heartbeat and w.last_heartbeat >= cutoff]


def _fal_worker_alive() -> bool:
    """True when some RQ worker serving the "fal" queue is alive. The
    fal-worker container is behind a compose profile; admitting a task with
    nobody to submit, poll and fetch it would strand paid work."""
    from .tasks import _get_redis

    try:
        return bool(_live_fal_workers(_get_redis()))
    except Exception:
        log.exception("fal: could not check for a live fal worker")
        return False


def fal_worker_alive() -> bool:
    """Public form of the admission's worker check, for a consumer that asks
    before doing local work admission would then throw away (AICP-251)."""
    return _fal_worker_alive()


def _now() -> datetime:
    """Naive UTC, like every timestamp column of these tables."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base of the small typed set callers can tell apart (design §3)."""


class AdmissionRefused(ProviderError):
    """Admission said no; nothing was inserted or reserved. ``status_code`` is
    what an API layer should answer (403 disabled, 404 owner gone, 409 slot
    taken, 422 invalid spec)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class ActionRefused(ProviderError):
    """An operator action was refused; nothing changed (unless stated)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class ProviderRefused(ProviderError):
    pass


class ProviderSubmitUnknown(ProviderError):
    pass


class ProviderTaskFailed(ProviderError):
    pass


class ProviderResultGone(ProviderError):
    pass


# ---------------------------------------------------------------------------
# Settings validation (fal-worker start-up)
# ---------------------------------------------------------------------------


# The longest RQ job timeout in tasks.py (PROCESS_JOB_TIMEOUT, 3 h): a floor for
# how long a result may sit before anyone reads it, NOT a proof (the residual is
# stated in docs/fal-provider-design.md and watched by watchdog()).
_RUN_WORKER_LONGEST_JOB_SEC = 10800


def validate_settings(s=settings) -> list[str]:
    """Return every violated bound (empty = valid)."""
    errors = []
    if s.FAL_CLAIM_LEASE_SEC < s.FAL_JOB_TIMEOUT + 60:
        errors.append("FAL_CLAIM_LEASE_SEC must exceed FAL_JOB_TIMEOUT by at least 60 s")
    if s.FAL_FETCH_WINDOW_SEC < 2 * s.FAL_JOB_TIMEOUT:
        errors.append("FAL_FETCH_WINDOW_SEC must be at least 2 x FAL_JOB_TIMEOUT")
    if s.FAL_OUTPUT_LIFECYCLE_SEC < s.FAL_FETCH_WINDOW_SEC + _RUN_WORKER_LONGEST_JOB_SEC:
        errors.append("FAL_OUTPUT_LIFECYCLE_SEC must cover FAL_FETCH_WINDOW_SEC + 3 h")
    if s.FAL_INPUT_LIFECYCLE_SEC < (
        s.FAL_JOB_TIMEOUT + s.FAL_POLL_DEADLINE_SEC + s.FAL_CANCEL_CONFIRM_SEC
    ):
        errors.append(
            "FAL_INPUT_LIFECYCLE_SEC must cover FAL_JOB_TIMEOUT + FAL_POLL_DEADLINE_SEC "
            "+ FAL_CANCEL_CONFIRM_SEC"
        )
    if s.FAL_FETCH_MAX_ATTEMPTS < 1:
        errors.append("FAL_FETCH_MAX_ATTEMPTS must be >= 1")
    if s.FAL_MAX_INPUT_BYTES <= 0:
        errors.append("FAL_MAX_INPUT_BYTES must be > 0")
    return errors


def allowed_project_ids() -> set[str]:
    return {p.strip() for p in (settings.FAL_PROJECT_IDS or "").split(",") if p.strip()}


def project_allowed(project_id: str, project_type: Optional[str],
                    family: Optional[str] = None) -> bool:
    """The per-project data-handling gate: the explicit ``FAL_PROJECT_IDS``
    allowlist, or any ``reframe`` project — that type exists only to send its
    source to Fal, and every generation is confirmed with its cost estimate.
    A ``subtitle_removal`` project is admitted for the ``inpaint`` family only
    (AICP-358 caption fills, operator decision 2026-10-31); *family* must come
    from the endpoint registry, never from a request.
    ``FAL_ENABLED`` stays the master switch (checked separately)."""
    from .project_types import REFRAME, SUBTITLE_REMOVAL

    if project_id in allowed_project_ids() or project_type == REFRAME:
        return True
    return project_type == SUBTITLE_REMOVAL and family == "inpaint"


# ---------------------------------------------------------------------------
# The one CAS primitive
# ---------------------------------------------------------------------------


def _claim_free(now: datetime):
    return or_(ProviderTask.claim_owner.is_(None), ProviderTask.claim_expires_at < now)


def _cas(
    db: Session,
    task_id: str,
    *,
    expected: Iterable[S] | S,
    to: Optional[S] = None,
    token: Optional[str] = None,
    claim: str = "release",
    require: str = "free",
    extra_where: Iterable = (),
    values: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> bool:
    """Guarded UPDATE of one task row; True iff exactly one row changed.

    ``require``: ``"free"`` — no live claim of someone else (free, expired, or
    ours when *token* is given); ``"ours_live"`` — our token with an unexpired
    lease (the marker and every pre-side-effect re-check).
    ``claim``: ``"take"`` (set *token* + a fresh lease), ``"release"`` (clear
    it), ``"keep"`` (leave it).
    """
    now = now or _now()
    expected = {expected} if isinstance(expected, S) else set(expected)
    if to is not None:
        for e in expected:
            if not can_transition(e, to):
                raise ValueError(f"illegal provider task transition {e.value} -> {to.value}")
    conds = [ProviderTask.id == task_id,
             ProviderTask.status.in_([e.value for e in expected])]
    if require == "ours_live":
        conds += [ProviderTask.claim_owner == token, ProviderTask.claim_expires_at > now]
    else:
        free = _claim_free(now)
        conds.append(or_(free, ProviderTask.claim_owner == token) if token else free)
    conds.extend(extra_where)
    vals = dict(values or {})
    if to is not None:
        vals["status"] = to.value
    if claim == "take":
        vals["claim_owner"] = token
        vals["claim_expires_at"] = now + timedelta(seconds=settings.FAL_CLAIM_LEASE_SEC)
    elif claim == "release":
        vals["claim_owner"] = None
        vals["claim_expires_at"] = None
    vals["version_id"] = ProviderTask.version_id + 1
    vals["updated_at"] = now
    res = db.execute(
        update(ProviderTask).where(and_(*conds)).values(**vals)
        .execution_options(synchronize_session=False)
    )
    return res.rowcount == 1


def _load(db: Session, task_id: str) -> Optional[ProviderTask]:
    return db.execute(
        select(ProviderTask).where(ProviderTask.id == task_id)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


def _still_ours(db: Session, task_id: str, status: S, token: str) -> bool:
    """The three-part re-check before any side effect: status, token, lease."""
    return db.execute(
        select(ProviderTask.id).where(
            ProviderTask.id == task_id,
            ProviderTask.status == status.value,
            ProviderTask.claim_owner == token,
            ProviderTask.claim_expires_at > _now(),
        )
    ).first() is not None


def _token() -> str:
    return uuid.uuid4().hex


def _credits(task: ProviderTask) -> float:
    return (task.reserved_mcredits or 0) / 1000.0


def _release(db: Session, task: ProviderTask) -> None:
    """Release the task's unconsumed reservation in the caller's transaction."""
    spend.reserve(db, spend.provider_task_scope(task.id), 0.0, kind=task.kind,
                  release_unused=True, provider=PROVIDER)


def _draw(db: Session, task: ProviderTask) -> None:
    """Strict draw at the reserved amount (raises on failure). The caller has
    already UPDATEd the task row (M4: task row lock before the day lock)."""
    spend.draw_in_session(
        db, spend.provider_task_scope(task.id), spend.provider_task_call_key(task.id),
        _credits(task), kind=task.kind, provider=PROVIDER,
    )


# ---------------------------------------------------------------------------
# Active-task checks (consumer contract, design §6; M2)
# ---------------------------------------------------------------------------


_ACTIVE = text(PROVIDER_TASK_ACTIVE_WHERE)


def _under(path: Optional[str], root: str) -> bool:
    if not path:
        return False
    real, real_root = os.path.realpath(path), os.path.realpath(root)
    try:
        return real != real_root and os.path.commonpath([real, real_root]) == real_root
    except ValueError:
        return False


def active_task_for_project(db: Session, project_id: str) -> Optional[ProviderTask]:
    return db.execute(
        select(ProviderTask).where(ProviderTask.owner_project_id == project_id, _ACTIVE)
        .limit(1)
    ).scalar_one_or_none()


def active_task_for_run(db: Session, run_id: str, project_id: str) -> Optional[ProviderTask]:
    """An active task owned by the run, or writing under its directory."""
    rows = db.execute(
        select(ProviderTask).where(ProviderTask.owner_project_id == project_id, _ACTIVE)
    ).scalars().all()
    rdir = storage.run_dir(run_id, project_id)
    for t in rows:
        if t.owner_run_id == run_id or t.owner_id == run_id or _under(t.destination_path, rdir):
            return t
    return None


# ---------------------------------------------------------------------------
# Admission (rule 4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInput:
    param: str          # request parameter that carries the CDN URL
    path: str           # local file under DATA_DIR
    content_type: str


def _probe_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _hash_file(path: str, cap: int) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > cap:
                raise AdmissionRefused(
                    422, f"input {os.path.basename(path)} is larger than "
                    f"FAL_MAX_INPUT_BYTES ({cap} bytes); offer a smaller re-encode"
                )
            h.update(chunk)
    return size, h.hexdigest()


def _check_destination(dest: str, project_id: str, run_id: Optional[str]) -> str:
    try:
        real = storage.ensure_under_data_dir(dest)
    except storage.PathOutsideDataDir:
        raise AdmissionRefused(422, "destination is not under the data directory") from None
    proj = os.path.join(storage.data_dir_root(), "projects", project_id)
    runs = os.path.join(proj, "runs")
    if run_id is not None:
        rdir = os.path.join(runs, run_id)
        if not _under(real, rdir):
            raise AdmissionRefused(422, "destination must lie under the owner run's directory")
        if _under(real, os.path.join(rdir, "clips")):
            raise AdmissionRefused(422, "destination must not lie under clips/ (disk cleanup)")
    else:
        if not _under(real, proj) or _under(real, runs) or real == runs:
            raise AdmissionRefused(
                422, "destination must lie under the owner project's directory, outside runs/"
            )
    if _under(real, os.path.join(proj, "tail_edits")):
        raise AdmissionRefused(422, "destination must not lie under tail_edits/ (disk cleanup)")
    if os.path.lexists(real):
        raise AdmissionRefused(409, "destination already exists; every task needs a new path")
    if not os.path.isdir(os.path.dirname(real)):
        raise AdmissionRefused(422, "destination directory does not exist")
    return real


def _lock_owner(db: Session, project_id: str, run_id: Optional[str]) -> None:
    """Lock order: the project row, then the run row (design §6)."""
    from .models import Run, VideoProject

    project = db.execute(
        select(VideoProject).where(VideoProject.id == project_id).with_for_update()
    ).scalar_one_or_none()
    if project is None:
        raise AdmissionRefused(404, "owner project not found")
    if run_id is not None:
        run = db.execute(
            select(Run).where(Run.id == run_id).with_for_update()
        ).scalar_one_or_none()
        if run is None or run.project_id != project_id:
            raise AdmissionRefused(404, "owner run not found in that project")


def admit(
    db: Session,
    *,
    endpoint_id: str,
    params: dict,
    inputs: list[TaskInput],
    output_sec: float,
    resolution: str,
    owner_kind: str,
    owner_id: str,
    owner_project_id: str,
    destination_path: str,
    operation_key: str,
    owner_slot: str = "main",
    owner_run_id: Optional[str] = None,
    output_frames: Optional[int] = None,
    reserve: bool = True,
) -> tuple[ProviderTask, bool]:
    """Freeze, validate, insert ``queued`` and reserve — in the CALLER's
    transaction (the caller commits, then calls ``tasks.enqueue_fal_job(
    "fal_submit", task.id)`` only when *created* is True).

    Returns ``(task, created)``. A repeat of the same ``operation_key`` returns
    the existing row with ``created=False`` and reserves nothing. Raises
    :class:`AdmissionRefused` (nothing written) or ``spend.BudgetExceeded``
    (the caller must roll back; no row, no request).

    *reserve=False* (batch VOID fills): the task row holds its
    ``reserved_mcredits`` but the ledger reservation is the CALLER's, made
    once for every task of the batch with ``spend.reserve_many`` after all of
    them are admitted (one day lock, design R1-2). A caller that passes it
    and does not reserve before its commit breaks the spend accounting.
    """
    if not (settings.FAL_ENABLED and settings.FAL_API_KEY):
        raise AdmissionRefused(403, "Fal is disabled (FAL_ENABLED / FAL_API_KEY)")
    try:
        spec = fal_models.endpoint(endpoint_id)
        spec.usd_per_sec(resolution)
    except (fal_models.UnknownEndpoint, fal_models.UntariffedResolution) as exc:
        raise AdmissionRefused(422, str(exc)) from None
    if owner_project_id not in allowed_project_ids():
        from .models import VideoProject

        project = db.get(VideoProject, owner_project_id)
        if not project_allowed(owner_project_id, project.project_type if project else None,
                               spec.family):
            raise AdmissionRefused(
                403, "this project is not in FAL_PROJECT_IDS (and is not a reframe project, "
                "nor a subtitle_removal project asking for an inpaint endpoint)"
            )
    if not _fal_worker_alive():
        raise AdmissionRefused(
            503, "no fal-worker is running (start it with `docker compose --profile fal up -d`)"
        )
    if spec.media_check == "frames":
        if not isinstance(output_frames, int) or isinstance(output_frames, bool) \
                or output_frames <= 0:
            raise AdmissionRefused(422, f"{endpoint_id} needs a positive integer output_frames")
    elif spec.media_check != "duration":
        raise AdmissionRefused(422, f"{endpoint_id}: unknown media check {spec.media_check!r}")
    elif output_frames is not None:
        raise AdmissionRefused(422, f"{endpoint_id} does not take output_frames")
    params_in = [inp.param for inp in inputs]
    if len(set(params_in)) != len(params_in):
        raise AdmissionRefused(422, "an input parameter is listed twice")
    if spec.required_inputs and set(params_in) != spec.required_inputs:
        raise AdmissionRefused(
            422, f"{endpoint_id} needs exactly the inputs {sorted(spec.required_inputs)}"
        )
    if set(params_in) & set(params):
        raise AdmissionRefused(422, "an input parameter is also passed as a plain parameter")
    if params.get("resolution") not in (None, resolution):
        raise AdmissionRefused(422, "params.resolution disagrees with resolution")
    unknown = set(params) - spec.allowed_params
    if unknown:
        raise AdmissionRefused(422, f"parameters not allowed for {endpoint_id}: {sorted(unknown)}")
    if "aspect_ratio" in params and params["aspect_ratio"] not in spec.aspect_ratios:
        raise AdmissionRefused(422, f"aspect ratio {params['aspect_ratio']!r} not supported")
    if not spec.min_input_sec <= float(output_sec) <= spec.max_input_sec:
        raise AdmissionRefused(422, f"duration {output_sec} s outside the endpoint's limits")
    if not inputs:
        raise AdmissionRefused(422, "a task needs at least one input")
    if settings.KIE_USD_PER_CREDIT <= 0:
        raise AdmissionRefused(422, "cannot price the task")

    _lock_owner(db, owner_project_id, owner_run_id)

    # Same operation already admitted? (Cheap first read; the savepoint below
    # is what makes a concurrent double start safe.)
    existing = _by_operation(db, operation_key)
    if existing is not None:
        return existing, False

    dest = _check_destination(destination_path, owner_project_id, owner_run_id)
    refs = []
    priced_sec = float(output_sec)
    for inp in inputs:
        try:
            path = storage.ensure_under_data_dir(inp.path)
        except storage.PathOutsideDataDir:
            raise AdmissionRefused(422, "an input is not under the data directory") from None
        if not os.path.isfile(path):
            raise AdmissionRefused(422, f"input {os.path.basename(path)} does not exist")
        size, digest = _hash_file(path, settings.FAL_MAX_INPUT_BYTES)
        if inp.content_type.startswith("video/"):
            dur = _probe_duration(path)
            if dur is None or not spec.min_input_sec <= dur <= spec.max_input_sec:
                raise AdmissionRefused(
                    422, f"input {os.path.basename(path)} duration {dur} s is outside "
                    f"{spec.min_input_sec}-{spec.max_input_sec} s"
                )
            # The reservation and the result check both use output_sec; for a
            # video-to-video endpoint the output is as long as the input.
            if abs(dur - float(output_sec)) > settings.FAL_OUTPUT_DURATION_TOLERANCE_SEC:
                raise AdmissionRefused(
                    422, f"output_sec {output_sec} s does not match the input's "
                    f"{dur:.2f} s; the reservation would not cover the request"
                )
            priced_sec = max(priced_sec, dur)
        refs.append({"param": inp.param, "path": path, "content_type": inp.content_type,
                     "size": size, "sha256": digest})
    # Priced on the LONGEST of output_sec and every probed video input, so a
    # short output_sec within the tolerance can never under-reserve.
    usd = spec.worst_case_usd(resolution, priced_sec)

    request_spec = {"endpoint": endpoint_id, "params": dict(params), "resolution": resolution,
                    "output_sec": float(output_sec),
                    "inputs": [{k: r[k] for k in ("param", "content_type", "size", "sha256")}
                               for r in refs]}
    # The result check is frozen with the request (AICP-358): a task admitted
    # before this field existed has none and keeps the duration check.
    if spec.media_check == "frames":
        request_spec["verify"] = {"kind": "frames", "frames": int(output_frames)}
    if not spec.send_resolution:
        request_spec["send_resolution"] = False
    canonical = json.dumps(request_spec, sort_keys=True, separators=(",", ":"))
    credits = usd / settings.KIE_USD_PER_CREDIT
    task = ProviderTask(
        id=str(uuid.uuid4()), provider=PROVIDER, endpoint=endpoint_id,
        kind=f"fal_{spec.family}", owner_kind=owner_kind, owner_id=owner_id,
        owner_slot=owner_slot, owner_project_id=owner_project_id, owner_run_id=owner_run_id,
        destination_path=dest, operation_key=operation_key, status=S.queued.value,
        reservation_held=True, draw_pending=False, reserved_mcredits=spend._mc(credits),
        request_spec=request_spec, input_refs=refs,
        request_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        output_sec=float(output_sec), resolution=resolution, estimated_usd=usd,
        version_id=0,
    )
    sp = db.begin_nested()
    try:
        db.add(task)
        db.flush()
    except IntegrityError:
        sp.rollback()
        winner = _by_operation(db, operation_key)
        if winner is not None:
            return winner, False
        raise AdmissionRefused(
            409, "another task is still active for this owner slot (or uses this "
            "destination); close or abandon it first"
        ) from None
    sp.commit()
    if reserve:
        spend.reserve(db, spend.provider_task_scope(task.id), credits, kind=task.kind,
                      provider=PROVIDER)
    log.info("fal: admitted task %s (%s, %s, worst case %.2f USD) for %s:%s",
             task.id, endpoint_id, resolution, usd, owner_kind, owner_id)
    return task, True


def _by_operation(db: Session, operation_key: str) -> Optional[ProviderTask]:
    # M5: by (provider, operation_key) — never another provider's row.
    return db.execute(
        select(ProviderTask).where(ProviderTask.provider == PROVIDER,
                                   ProviderTask.operation_key == operation_key)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


# ---------------------------------------------------------------------------
# fal_submit (rules 4.2, 4.3, 4.6, 4.7)
# ---------------------------------------------------------------------------


def _read_bounded(path: str, cap: int) -> bytes:
    with open(path, "rb") as fh:
        data = fh.read(cap + 1)
    if len(data) > cap:
        raise ValueError("too_large")
    return data


def _local_failure(task_id: str, token: str, code: str, message: str) -> None:
    """queued -> failed before the marker: nothing was sent, so the
    reservation is released in the same transaction."""
    with _sessions() as db:
        task = _load(db, task_id)
        if task is None:
            return
        if _cas(db, task_id, expected=S.queued, to=S.failed, token=token,
                values={"error_code": code, "error_message": message[:500],
                        "reservation_held": False}):
            _release(db, task)
            db.commit()
            log.warning("fal: task %s failed locally before submit (%s)", task_id, code)


def fal_submit(task_id: str) -> None:
    token = _token()
    with _sessions() as db:
        if not _cas(db, task_id, expected=S.queued, token=token, claim="take"):
            log.info("fal_submit %s: not queued or claimed elsewhere — nothing to do", task_id)
            return
        db.commit()
        task = _load(db, task_id)
    spec = fal_models.endpoint(task.endpoint)

    client = _client()
    try:
        body = dict(task.request_spec["params"])
        if task.request_spec.get("send_resolution", True):
            body["resolution"] = task.resolution
        for ref in task.input_refs:
            try:
                path = storage.ensure_under_data_dir(ref["path"])
            except storage.PathOutsideDataDir:
                return _local_failure(task_id, token, "input_path", "input left the data directory")
            try:
                data = _read_bounded(path, settings.FAL_MAX_INPUT_BYTES)
            except FileNotFoundError:
                return _local_failure(task_id, token, "input_missing", f"{ref['param']} is gone")
            except ValueError:
                return _local_failure(task_id, token, "input_changed", f"{ref['param']} grew past the cap")
            except OSError as exc:
                return _local_failure(task_id, token, "input_missing", f"{type(exc).__name__}")
            if hashlib.sha256(data).hexdigest() != ref["sha256"]:
                return _local_failure(task_id, token, "input_changed",
                                      f"{ref['param']} changed since admission")
            with _sessions() as db:
                if not _still_ours(db, task_id, S.queued, token):
                    log.info("fal_submit %s: lost the claim during upload — exiting", task_id)
                    return
            try:
                body[ref["param"]] = client.upload(
                    data, content_type=ref["content_type"],
                    file_name=f"{task_id}-{ref['param']}",
                    lifecycle_sec=settings.FAL_INPUT_LIFECYCLE_SEC,
                )
            except FalUploadError as exc:
                return _local_failure(task_id, token, "upload_failed", str(exc))
            del data

        # The marker: from here on the request may exist at Fal.
        with _sessions() as db:
            if not _cas(db, task_id, expected=S.queued, to=S.submit_started, token=token,
                        require="ours_live", claim="keep",
                        values={"submit_started_at": _now()}):
                log.info("fal_submit %s: marker refused (lease lost or abandoned) — no POST", task_id)
                return
            db.commit()
        with _sessions() as db:
            if not _still_ours(db, task_id, S.submit_started, token):
                log.warning("fal_submit %s: ownership lost after the marker — no POST", task_id)
                return

        outcome = client.submit(task.endpoint, body, queue_app=spec.queue_app,
                                output_lifecycle_sec=settings.FAL_OUTPUT_LIFECYCLE_SEC)
    finally:
        client.close()

    if outcome.kind == "accepted":
        _accept(task, token, outcome.request_id, outcome.urls)
    elif outcome.kind == "refused":
        # B3: every submit 4xx keeps the reservation until the operator acts.
        with _sessions() as db:
            _cas(db, task_id, expected=S.submit_started, to=S.refused, token=token,
                 require="ours_live",
                 values={"error_code": f"submit_http_{outcome.http_status}",
                         "error_message": outcome.error})
            db.commit()
        log.warning("fal: task %s refused at submit (HTTP %s); reservation held for the "
                    "operator's close/abandon", task_id, outcome.http_status)
    else:
        with _sessions() as db:
            _cas(db, task_id, expected=S.submit_started, to=S.submit_unknown, token=token,
                 require="ours_live",
                 values={"error_code": "submit_unknown",
                         "error_message": f"{outcome.error}; a billed generation may exist"})
            db.commit()
        log.error("fal: task %s submit outcome UNKNOWN (%s); parked for the operator "
                  "(adopt / abandon / close)", task_id, outcome.error)


def _accept(task: ProviderTask, token: str, request_id: str, urls: dict) -> None:
    """The acceptance commit (rule 4.7): submitted + draw in one commit,
    else accepted_unrecorded, else an ERROR hint (recovery -> submit_unknown)."""
    accepted = {"request_id": request_id, "accepted_at": _now(), **urls}
    for attempt in range(3):
        try:
            with _sessions() as db:
                if not _cas(db, task.id, expected=S.submit_started, to=S.submitted,
                            token=token, require="ours_live",
                            values={**accepted, "reservation_held": False}):
                    log.error("fal: task %s accepted as %s but our claim is gone",
                              task.id, request_id)
                    return
                _draw(db, task)
                db.commit()
            _enqueue("fal_poll", task.id, delay_sec=settings.FAL_POLL_INTERVAL_SEC)
            log.info("fal: task %s accepted, request %s", task.id, request_id)
            return
        except Exception:
            log.exception("fal: acceptance commit %d failed for task %s", attempt + 1, task.id)
    for attempt in range(3):
        try:
            with _sessions() as db:
                if _cas(db, task.id, expected=S.submit_started, to=S.accepted_unrecorded,
                        token=token, require="ours_live", values=accepted):
                    db.commit()
                    log.error("fal: task %s accepted as %s, draw NOT recorded; recovery "
                              "writes it", task.id, request_id)
                return
        except Exception:
            log.exception("fal: accepted_unrecorded commit %d failed for task %s",
                          attempt + 1, task.id)
    log.error("fal: OPERATOR HINT — task %s was ACCEPTED by Fal as request_id=%s but no "
              "local commit succeeded; it will become submit_unknown — adopt it with this id",
              task.id, request_id)


# ---------------------------------------------------------------------------
# fal_poll (rules 4.8, 4.9)
# ---------------------------------------------------------------------------

_POLLED = (S.submitted, S.cancelling, S.needs_reconciliation)


def _poll_delay(task: ProviderTask) -> int:
    if task.status == S.needs_reconciliation.value:
        return settings.FAL_SLOW_POLL_SEC
    base = settings.FAL_POLL_INTERVAL_SEC
    return min(base * (2 ** min(task.poll_failures or 0, 5)), settings.FAL_SLOW_POLL_SEC)


def fal_poll(task_id: str) -> None:
    with _sessions() as db:
        task = _load(db, task_id)
    if task is None or task.status not in {s.value for s in _POLLED}:
        return
    status = S(task.status)
    now = _now()
    if not settings.FAL_API_KEY:
        log.error("fal_poll %s: FAL_API_KEY missing — holding", task_id)
        return _enqueue("fal_poll", task_id, delay_sec=settings.FAL_SLOW_POLL_SEC)

    if status == S.cancelling:
        if task.cancel_requested_at and now >= task.cancel_requested_at + timedelta(
                seconds=settings.FAL_CANCEL_CONFIRM_SEC):
            with _sessions() as db:
                if _cas(db, task_id, expected=S.cancelling, to=S.needs_reconciliation,
                        values={"error_code": "cancel_unconfirmed"}):
                    db.commit()
                    log.error("fal: task %s cancel unconfirmed after %ss — needs reconciliation",
                              task_id, settings.FAL_CANCEL_CONFIRM_SEC)
            return _enqueue("fal_poll", task_id, delay_sec=settings.FAL_SLOW_POLL_SEC)
        if task.cancel_delivered_at is None and (
                task.cancel_attempted_at is None
                or now >= task.cancel_attempted_at + timedelta(seconds=4 * settings.FAL_POLL_INTERVAL_SEC)):
            _enqueue("fal_cancel", task_id)

    client = _client()
    try:
        out = client.status(task.status_url)
    finally:
        client.close()

    with _sessions() as db:
        if out.kind == "ok" and out.fal_status == STATUS_COMPLETED:
            if _cas(db, task_id, expected=status, to=S.fetching,
                    values={"fetch_attempts": 0, "fetch_deadline_at": None,
                            "completed_seen_at": now, "poll_failures": 0,
                            "error_code": None}):
                db.commit()
                _enqueue("fal_fetch", task_id)
            else:
                _enqueue("fal_poll", task_id, delay_sec=settings.FAL_POLL_INTERVAL_SEC)
            return
        if out.kind == "ok" and (out.fal_status or "").upper() in STATUS_TERMINAL_FAILURE:
            if _cas(db, task_id, expected=status, to=S.failed,
                    values={"error_code": f"fal_{out.fal_status.lower()}",
                            "error_message": "Fal reported the request ended without a result"}):
                db.commit()
                return
        elif out.kind in ("ok", "transient"):
            # Optimistic on the version this tick READ: if any other job (a
            # concurrent poll that recorded a 401, an operator action) wrote
            # the row since, this tick's view is stale — it neither clears a
            # hold nor applies the deadline, and simply polls again.
            seen = [ProviderTask.version_id == task.version_id]
            if out.kind == "ok":
                held = False
                fresh = _cas(db, task_id, expected=status, claim="keep", extra_where=seen,
                             values={"poll_failures": 0, "error_code": None}
                             if task.error_code == "auth_rejected" else {"poll_failures": 0})
            else:
                held = task.error_code == "auth_rejected"
                fresh = _cas(db, task_id, expected=status, claim="keep", extra_where=seen,
                             values={"poll_failures": (task.poll_failures or 0) + 1})
            db.commit()
            held = held or not fresh
            # The poll deadline is applied only AFTER this tick's GET, and never
            # under an auth hold: a rejected key says nothing about the request,
            # and a cancel sent then could kill a paid job the restored key
            # would have collected.
            if status == S.submitted and not held and task.accepted_at and \
                    now >= task.accepted_at + timedelta(seconds=settings.FAL_POLL_DEADLINE_SEC):
                # Guarded in the CAS too: a concurrent poll may have just
                # recorded a hold this tick's read did not see.
                no_hold = or_(ProviderTask.error_code.is_(None),
                              ProviderTask.error_code != "auth_rejected")
                if _cas(db, task_id, expected=S.submitted, to=S.cancelling,
                        extra_where=[no_hold, ProviderTask.version_id == task.version_id + 1],
                        values={"cancel_requested_at": now, "cancel_delivered_at": None}):
                    db.commit()
                    log.warning("fal: task %s passed its poll deadline — cancelling", task_id)
                    _enqueue("fal_cancel", task_id)
        elif out.kind == "auth":
            _cas(db, task_id, expected=status, claim="keep",
                 values={"error_code": "auth_rejected", "error_message": out.error})
            db.commit()
            log.error("fal: status GET for task %s answered %s — key rejected; holding "
                      "(no cancel, no status change)", task_id, out.http_status)
        else:  # gone: 404 / other 4xx
            if status != S.needs_reconciliation and _cas(
                    db, task_id, expected=status, to=S.needs_reconciliation,
                    values={"error_code": f"status_http_{out.http_status}",
                            "error_message": out.error}):
                log.error("fal: task %s status GET answered %s — needs reconciliation",
                          task_id, out.http_status)
            db.commit()
        task = _load(db, task_id)
    if task is not None and task.status in {s.value for s in _POLLED}:
        _enqueue("fal_poll", task_id, delay_sec=_poll_delay(task))


# ---------------------------------------------------------------------------
# fal_cancel (rule 4.9; M3)
# ---------------------------------------------------------------------------


def fal_cancel(task_id: str) -> None:
    token = _token()
    with _sessions() as db:
        if not _cas(db, task_id, expected=S.cancelling, token=token, claim="take",
                    extra_where=[ProviderTask.cancel_delivered_at.is_(None)]):
            return
        db.commit()
        task = _load(db, task_id)
    client = _client()
    try:
        code = client.cancel(task.cancel_url)
    finally:
        client.close()
    delivered = code is not None and (200 <= code < 300 or code == 400)  # 400 = already completed
    with _sessions() as db:
        _cas(db, task_id, expected=S.cancelling, token=token, require="ours_live",
             values={"cancel_attempted_at": _now(),
                     **({"cancel_delivered_at": _now()} if delivered else {})})
        db.commit()
    log.log(logging.INFO if delivered else logging.WARNING,
            "fal: cancel for task %s answered %s (%s)", task_id, code,
            "delivered" if delivered else "will retry")


# ---------------------------------------------------------------------------
# fal_fetch (rule 4.10; B1)
# ---------------------------------------------------------------------------


def _staging_path(task: ProviderTask, token: str) -> str:
    d, base = os.path.split(task.destination_path)
    return os.path.join(d, f".{base}.fal-{task.id}-{token}.part")


def _probe_frames(path: str) -> Optional[int]:
    """Decoded video frame count (counts packets: no full decode)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=120, check=False,
        )
        return int(out.stdout.strip().split(",")[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _verify_media(path: str, output_sec: float, verify: Optional[dict] = None) -> Optional[str]:
    """*verify* is the contract frozen in ``request_spec["verify"]`` at admission.
    Absent (every task admitted before AICP-358, i.e. reframe) = the duration
    check; present but unknown = refuse, never a silent fallback."""
    try:
        if os.path.getsize(path) <= 0:
            return "empty file"
    except OSError:
        return "missing file"
    if verify is not None:
        if not isinstance(verify, dict) or verify.get("kind") != "frames" \
                or not isinstance(verify.get("frames"), int):
            return f"unknown result check {verify!r}"
        got = _probe_frames(path)
        if got is None:
            return "ffprobe could not count the frames"
        if got != verify["frames"]:
            return f"{got} frames, expected {verify['frames']}"
        return None
    dur = _probe_duration(path)
    if dur is None:
        return "ffprobe could not read the file"
    if abs(dur - float(output_sec)) > settings.FAL_OUTPUT_DURATION_TOLERANCE_SEC:
        return f"duration {dur:.2f} s, expected {output_sec:.2f} s"
    return None


_RESULT_META_KEYS = ("placement", "seed", "frame_count", "frames_per_second", "timings")
_RESULT_VIDEO_META_KEYS = ("content_type", "file_size", "width", "height", "duration", "fps")


def _result_meta(body: dict) -> dict:
    """The QA-relevant, URL-free part of a result body (whitelist: Fal adds
    fields over time, and media URLs must never be stored)."""
    def clean(v):
        if isinstance(v, str) and "://" in v:
            return None
        if isinstance(v, dict):
            return {k: c for k, x in v.items() if (c := clean(x)) is not None}
        if isinstance(v, list):
            return [c for x in v if (c := clean(x)) is not None]
        return v

    meta = {k: clean(body[k]) for k in _RESULT_META_KEYS if k in body}
    video = body.get("video") if isinstance(body.get("video"), dict) else {}
    vmeta = {k: clean(video[k]) for k in _RESULT_VIDEO_META_KEYS if k in video}
    if vmeta:
        meta["video"] = vmeta
    return meta


def _fetch_end(task_id: str, token: str, to: S, code: Optional[str], message: str = "",
               result_path: Optional[str] = None, result_meta: Optional[dict] = None) -> None:
    with _sessions() as db:
        values = {"error_code": code, "error_message": message[:500] or None}
        if result_path:
            values["result_path"] = result_path
        if result_meta is not None:
            values["result_meta"] = result_meta
        _cas(db, task_id, expected=S.fetching, to=to, token=token, require="ours_live",
             values=values)
        db.commit()


def _fetch_retry(task: ProviderTask, token: str, message: str) -> None:
    attempts = task.fetch_attempts or 1
    with _sessions() as db:
        _cas(db, task.id, expected=S.fetching, token=token, require="ours_live",
             values={"error_message": f"fetch attempt {attempts}: {message}"[:500]})
        db.commit()
    log.warning("fal: fetch attempt %d of task %s failed: %s", attempts, task.id, message)
    _enqueue("fal_fetch", task.id, delay_sec=60 * (2 ** (attempts - 1)))


def _fetch_auth_hold(task: ProviderTask, token: str, message: str) -> None:
    """A rejected key says nothing about the result: give the attempt back,
    restart the window, and retry slowly (the output stays retained at Fal for
    FAL_OUTPUT_LIFECYCLE_SEC; the watchdog alerts half-way)."""
    with _sessions() as db:
        _cas(db, task.id, expected=S.fetching, token=token, require="ours_live",
             values={"fetch_attempts": ProviderTask.fetch_attempts - 1,
                     "fetch_deadline_at": None, "error_code": "auth_rejected",
                     "error_message": message[:500]})
        db.commit()
    log.error("fal: result GET for task %s rejected the key — holding (attempt not counted)",
              task.id)
    _enqueue("fal_fetch", task.id, delay_sec=settings.FAL_SLOW_POLL_SEC)


def fal_fetch(task_id: str) -> None:
    if not settings.FAL_API_KEY:
        # No attempt counted, and the window restarts once the key is back.
        with _sessions() as db:
            _cas(db, task_id, expected=S.fetching, claim="keep",
                 values={"fetch_deadline_at": None, "error_code": "auth_rejected",
                         "error_message": "FAL_API_KEY missing"})
            db.commit()
        log.error("fal_fetch %s: FAL_API_KEY missing — holding (no attempt counted)", task_id)
        return _enqueue("fal_fetch", task_id, delay_sec=settings.FAL_SLOW_POLL_SEC)
    token = _token()
    now = _now()
    window = timedelta(seconds=settings.FAL_FETCH_WINDOW_SEC)
    with _sessions() as db:
        # The claim counts the attempt and starts the window (B1, crash-safe):
        # an attempt that dies anywhere after this commit has been counted.
        if not _cas(db, task_id, expected=S.fetching, token=token, claim="take", now=now,
                    values={"fetch_attempts": ProviderTask.fetch_attempts + 1,
                            "fetch_deadline_at": _coalesce_deadline(now + window)}):
            return
        db.commit()
        task = _load(db, task_id)

    # A result already installed (a crash between os.replace and `done`) wins
    # over the exhaustion rule: destinations are unique per task.
    if os.path.exists(task.destination_path) and \
            _verify_media(task.destination_path, task.output_sec,
                          (task.request_spec or {}).get("verify")) is None:
        # Re-read the (idempotent) result for its metadata; the file itself is
        # already ours, so a failed read still ends `done`, with the gap noted.
        client = _client()
        try:
            res = client.result(task.response_url)
        finally:
            client.close()
        if res.kind == "ok":
            return _fetch_end(task_id, token, S.done, None, result_path=task.destination_path,
                              result_meta=_result_meta(res.body))
        return _fetch_end(task_id, token, S.done, None,
                          f"installed result recovered; result metadata unavailable "
                          f"(result GET {res.http_status})",
                          result_path=task.destination_path)
    if (task.fetch_attempts or 0) > settings.FAL_FETCH_MAX_ATTEMPTS or (
            task.fetch_attempts > 1 and task.fetch_deadline_at and now > task.fetch_deadline_at):
        log.error("fal: task %s result is gone (%d attempts, window ended %s); draw kept",
                  task_id, task.fetch_attempts - 1, task.fetch_deadline_at)
        return _fetch_end(task_id, token, S.failed, "result_gone",
                          "the result could not be obtained; Fal may have billed it")

    client = _client()
    staging = _staging_path(task, token)
    try:
        res = client.result(task.response_url)
        if res.kind == "refused":
            return _fetch_end(task_id, token, S.refused, res.category or "refused", res.error)
        if res.http_status in (401, 403):
            return _fetch_auth_hold(task, token, f"result GET {res.http_status}: {res.error}")
        if res.kind != "ok":
            return _fetch_retry(task, token, f"result GET {res.http_status}: {res.error}")
        video = res.body.get("video") if isinstance(res.body.get("video"), dict) else {}
        url = video.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            return _fetch_retry(task, token, "result has no video url")
        if not os.path.isdir(os.path.dirname(task.destination_path)):
            return _fetch_end(task_id, token, S.failed, "owner_gone",
                              "the destination directory is gone; the draw stays")
        try:
            client.download(url, staging, max_bytes=MAX_RESULT_BYTES)
        except (FalDownloadError, OSError) as exc:
            return _fetch_retry(task, token, f"download: {exc}")
        problem = _verify_media(staging, task.output_sec,
                                (task.request_spec or {}).get("verify"))
        if problem:
            return _fetch_retry(task, token, f"downloaded file rejected: {problem}")
        with _sessions() as db:
            if not _still_ours(db, task_id, S.fetching, token):
                log.warning("fal: task %s lost its fetch claim before install", task_id)
                return
        try:
            os.replace(staging, task.destination_path)
        except OSError as exc:
            return _fetch_end(task_id, token, S.failed, "owner_gone", f"install failed: {exc}")
        _fetch_end(task_id, token, S.done, None, result_path=task.destination_path,
                   result_meta=_result_meta(res.body))
        log.info("fal: task %s done -> %s", task_id, task.destination_path)
    finally:
        client.close()
        try:
            os.unlink(staging)
        except OSError:
            pass


def _coalesce_deadline(value: datetime):
    from sqlalchemy import func

    return func.coalesce(ProviderTask.fetch_deadline_at, value)


# ---------------------------------------------------------------------------
# fal_close (the close's steps 2-3; recovery finishes it too) — M4
# ---------------------------------------------------------------------------


def fal_close(task_id: str) -> None:
    token = _token()
    with _sessions() as db:
        if not _cas(db, task_id, expected=S.closing, token=token, claim="take"):
            return
        db.commit()
        task = _load(db, task_id)
    if task.request_id and task.cancel_url and settings.FAL_API_KEY:
        client = _client()
        try:
            code = client.cancel(task.cancel_url)
        finally:
            client.close()
        log.info("fal: close of task %s sent its final cancel (HTTP %s)", task_id, code)
    with _sessions() as db:
        # M4: the status UPDATE (task row lock) first, then the draw.
        if not _cas(db, task_id, expected=S.closing, to=S.failed, token=token,
                    require="ours_live",
                    values={"error_code": "operator_closed",
                            "error_message": "closed by the operator; Fal may still "
                                             "complete and bill this request"}):
            return
        sp = db.begin_nested()
        try:
            _draw(db, task)
            sp.commit()
            pending = False
        except Exception:
            sp.rollback()
            pending = True
            log.exception("fal: close of task %s could not write its draw; draw_pending set "
                          "(the held reservation carries the charge)", task_id)
        db.execute(
            update(ProviderTask).where(ProviderTask.id == task_id)
            .values(draw_pending=pending, reservation_held=pending,
                    version_id=ProviderTask.version_id + 1)
            .execution_options(synchronize_session=False)
        )
        db.commit()


# ---------------------------------------------------------------------------
# Operator actions (rules 4.5, 4.6) — each writes provider_task_actions in the
# transaction of its CAS, and commits.
# ---------------------------------------------------------------------------


def _action(db: Session, task: ProviderTask, action: str, actor: str, to: str,
            note: Optional[str], request_id: Optional[str] = None,
            details: Optional[dict] = None) -> None:
    db.add(ProviderTaskAction(task_id=task.id, action=action, actor=actor,
                              from_status=task.status, to_status=to, note=note,
                              request_id=request_id, details=details))
    db.flush()


def _require(db: Session, task_id: str) -> ProviderTask:
    task = _load(db, task_id)
    if task is None:
        raise ActionRefused(404, "task not found")
    return task


def abandon(db: Session, task_id: str, *, actor: str, note: str) -> ProviderTask:
    """The operator confirmed NOTHING exists at Fal (or the task never left
    ``queued``): release the reservation. Legal from ``queued``,
    ``submit_unknown`` and a ``refused`` row still holding its reservation."""
    task = _require(db, task_id)
    st = S(task.status)
    if st == S.refused and not task.reservation_held:
        raise ActionRefused(409, "this refused task holds no reservation; nothing to abandon")
    if st not in (S.queued, S.submit_unknown, S.refused):
        raise ActionRefused(409, f"abandon is not allowed from {st.value}; use close")
    extra = [ProviderTask.reservation_held.is_(True)] if st == S.refused else []
    if not _cas(db, task_id, expected=st, to=S.abandoned, extra_where=extra,
                values={"reservation_held": False}):
        db.rollback()
        raise ActionRefused(409, "the task is being worked on or changed; try again")
    _release(db, task)
    _action(db, task, "abandon", actor, S.abandoned.value, note)
    db.commit()
    return _load(db, task_id)


def close(db: Session, task_id: str, *, actor: str, note: str) -> ProviderTask:
    """The operator cannot confirm either way. Never releases anything."""
    task = _require(db, task_id)
    st = S(task.status)
    if st == S.refused:
        if not task.reservation_held:
            raise ActionRefused(409, "this refused task holds no reservation; nothing to close")
        # Strict: the held reservation becomes the draw, or nothing changes.
        if not _cas(db, task_id, expected=S.refused, to=S.refused,
                    extra_where=[ProviderTask.reservation_held.is_(True)],
                    values={"reservation_held": False}):
            db.rollback()
            raise ActionRefused(409, "the task is being worked on or changed; try again")
        try:
            _draw(db, task)
        except Exception:
            db.rollback()
            raise ActionRefused(503, "the draw could not be written; nothing changed, "
                                     "the reservation still carries the charge") from None
        _action(db, task, "close", actor, S.refused.value, note)
        db.commit()
        return _load(db, task_id)
    if st in (S.queued, S.closing) or st.value not in {s.value for s in PROVIDER_TASK_NON_TERMINAL_STATUSES}:
        raise ActionRefused(409, f"close is not allowed from {st.value}"
                                 + ("; a queued task is cancelled with abandon" if st == S.queued else ""))
    if not _cas(db, task_id, expected=st, to=S.closing):
        db.rollback()
        raise ActionRefused(409, "another job holds this task right now; try again shortly")
    _action(db, task, "close", actor, S.closing.value, note)
    db.commit()
    _enqueue("fal_close", task_id)
    return _load(db, task_id)


def resume(db: Session, task_id: str, *, actor: str, note: str) -> ProviderTask:
    """A fresh cancel for a ``needs_reconciliation`` task."""
    task = _require(db, task_id)
    if task.status != S.needs_reconciliation.value:
        raise ActionRefused(409, "resume is only for needs_reconciliation")
    if not _cas(db, task_id, expected=S.needs_reconciliation, to=S.cancelling,
                values={"cancel_requested_at": _now(), "cancel_delivered_at": None,
                        "cancel_attempted_at": None}):
        db.rollback()
        raise ActionRefused(409, "the task is being worked on; try again")
    _action(db, task, "resume", actor, S.cancelling.value, note)
    db.commit()
    _enqueue("fal_cancel", task_id)
    _enqueue("fal_poll", task_id, delay_sec=settings.FAL_POLL_INTERVAL_SEC)
    return _load(db, task_id)


def adopt(db: Session, task_id: str, request_id: str, *, actor: str, note: str,
          client: Optional[FalClient] = None) -> ProviderTask:
    """The request exists at Fal: verify the id with ONE status GET, then
    record it (and its draw) and start polling."""
    task = _require(db, task_id)
    if task.status != S.submit_unknown.value:
        raise ActionRefused(409, "adopt is only for submit_unknown")
    request_id = (request_id or "").strip()
    if not request_id or "/" in request_id:
        raise ActionRefused(422, "a request id is required")
    clash = db.execute(select(ProviderTask.id).where(
        ProviderTask.provider == PROVIDER, ProviderTask.request_id == request_id,
        ProviderTask.id != task_id)).first()
    if clash is not None:
        raise ActionRefused(409, f"request {request_id} is already recorded on task {clash[0]}")
    urls = request_urls(fal_models.endpoint(task.endpoint).queue_app, request_id)
    own = client is None
    client = client or _client()
    try:
        out = client.status(urls["status_url"])
    finally:
        if own:
            client.close()
    if out.kind == "gone" and out.http_status == 404:
        _cas(db, task_id, expected=S.submit_unknown, claim="keep",
             values={"error_message": f"adopt of {request_id} refused: Fal answered 404"})
        _action(db, task, "adopt", actor, task.status, note, request_id=request_id,
                details={"refused": "status GET 404"})
        db.commit()
        raise ActionRefused(422, f"Fal does not know request {request_id} (404)")
    if out.kind != "ok":
        raise ActionRefused(503, f"Fal answered {out.http_status or out.error}; try again")
    try:
        if not _cas(db, task_id, expected=S.submit_unknown, to=S.submitted,
                    values={"request_id": request_id, "accepted_at": _now(),
                            "reservation_held": False, "error_code": None, **urls}):
            db.rollback()
            raise ActionRefused(409, "the task changed or is held; try again")
        _draw(db, task)
        _action(db, task, "adopt", actor, S.submitted.value, note, request_id=request_id)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ActionRefused(409, f"request {request_id} is already recorded on another task") from None
    except ActionRefused:
        raise
    except Exception:
        db.rollback()
        raise ActionRefused(503, "the draw could not be written; adopt again (or close)") from None
    _enqueue("fal_poll", task_id)
    return _load(db, task_id)


# ---------------------------------------------------------------------------
# Recovery (design §7) and watchdog — run by the fal-worker's ticker
# ---------------------------------------------------------------------------


def _backfill(task_id: str) -> None:
    """M1/M4: draw for a draw_pending row only; row lock first."""
    with _sessions() as db:
        res = db.execute(
            update(ProviderTask).where(ProviderTask.id == task_id,
                                       ProviderTask.draw_pending.is_(True))
            .values(version_id=ProviderTask.version_id + 1)
            .execution_options(synchronize_session=False))
        if res.rowcount != 1:
            return
        task = _load(db, task_id)
        try:
            _draw(db, task)
        except Exception:
            db.rollback()
            log.exception("fal: backfill draw for task %s failed again", task_id)
            return
        db.execute(update(ProviderTask).where(ProviderTask.id == task_id)
                   .values(draw_pending=False, reservation_held=False,
                           version_id=ProviderTask.version_id + 1)
                   .execution_options(synchronize_session=False))
        db.commit()
        log.info("fal: backfilled the draw of task %s", task_id)


def _record_unrecorded(task_id: str) -> bool:
    with _sessions() as db:
        task = _load(db, task_id)
        if task is None or not _cas(db, task_id, expected=S.accepted_unrecorded,
                                    to=S.submitted, values={"reservation_held": False}):
            return False
        try:
            _draw(db, task)
        except Exception:
            db.rollback()
            log.exception("fal: recovery could not write the draw of accepted task %s", task_id)
            return False
        db.commit()
    return True


def reconcile_provider_tasks(redis_conn, *, now: Optional[datetime] = None) -> dict:
    """One recovery pass. NEVER sends a POST; every job it enqueues is
    claim-guarded and idempotent, so a duplicate enqueue is harmless."""
    from .recovery import _live_run_ids
    from .tasks import QUEUE_FAL

    now = now or _now()
    live = _live_run_ids(redis_conn, queue_names=[QUEUE_FAL])
    report: dict[str, int] = {}

    def bump(key):
        report[key] = report.get(key, 0) + 1

    with _sessions() as db:
        rows = db.execute(select(ProviderTask).where(or_(
            ProviderTask.status.in_([s.value for s in PROVIDER_TASK_DRAIN_STATUSES]),
            ProviderTask.draw_pending.is_(True),
        ))).scalars().all()
        db.expunge_all()

    def claim_live(t):
        return bool(t.claim_owner and t.claim_expires_at and t.claim_expires_at > now)

    for t in rows:
        st = S(t.status)
        if t.draw_pending:
            _backfill(t.id)
            bump("backfilled")
        if st == S.queued:
            if not claim_live(t) and t.id not in live and \
                    t.updated_at <= now - timedelta(seconds=settings.FAL_QUEUED_GRACE_SEC):
                _enqueue("fal_submit", t.id)
                bump("resubmit_queued")
        elif st == S.submit_started:
            if t.request_id is None and not claim_live(t):
                with _sessions() as db:
                    if _cas(db, t.id, expected=S.submit_started, to=S.submit_unknown, now=now,
                            extra_where=[ProviderTask.request_id.is_(None)],
                            values={"error_code": "submit_unknown",
                                    "error_message": "the submit worker died after the marker; "
                                                     "a billed generation may exist"}):
                        db.commit()
                        bump("submit_unknown")
                        log.error("fal: task %s is submit_unknown (stale marker); the "
                                  "operator must adopt, abandon or close it", t.id)
        elif st == S.accepted_unrecorded:
            if _record_unrecorded(t.id):
                _enqueue("fal_poll", t.id)
                bump("recorded")
        elif st in _POLLED:
            if st == S.cancelling and t.cancel_delivered_at is None:
                _enqueue("fal_cancel", t.id)  # M3: regardless of poll liveness
                bump("recancel")
            if t.id not in live:
                _enqueue("fal_poll", t.id)
                bump("repoll")
        elif st == S.fetching:
            if not claim_live(t) and t.id not in live:
                _enqueue("fal_fetch", t.id)
                bump("refetch")
        elif st == S.closing:
            if not claim_live(t) and t.id not in live:
                _enqueue("fal_close", t.id)
                bump("reclose")
    return report


def watchdog(redis_conn, *, now: Optional[datetime] = None) -> list[str]:
    """Alerts the fal-worker's ticker logs at ERROR (independent of the job
    backlog): a result unread half-way to its retention, no live Fal worker,
    and an old pending Fal job."""
    from rq import Queue

    from .tasks import QUEUE_FAL

    now = now or _now()
    alerts = []
    half = timedelta(seconds=settings.FAL_OUTPUT_LIFECYCLE_SEC / 2)
    with _sessions() as db:
        stale = db.execute(select(ProviderTask.id, ProviderTask.completed_seen_at).where(
            ProviderTask.status == S.fetching.value,
            ProviderTask.completed_seen_at < now - half)).all()
        active = db.execute(select(ProviderTask.id).where(
            ProviderTask.status.in_([s.value for s in PROVIDER_TASK_DRAIN_STATUSES]))).first()
    for tid, seen in stale:
        alerts.append(f"task {tid} completed at Fal ~{seen} and is still unread: its output "
                      f"expires {settings.FAL_OUTPUT_LIFECYCLE_SEC}s after completion")
    if active is not None:
        try:
            if not _live_fal_workers(redis_conn):
                alerts.append("provider tasks are active but no fal-queue worker has a fresh heartbeat")
            q = Queue(QUEUE_FAL, connection=redis_conn)
            ids = q.get_job_ids(0, 0)
            if ids:
                from rq.job import Job

                job = Job.fetch(ids[0], connection=redis_conn)
                if job.enqueued_at and job.enqueued_at < datetime.utcnow() - timedelta(minutes=15):
                    alerts.append(f"the oldest pending fal job has waited since {job.enqueued_at}")
        except Exception as exc:  # pragma: no cover - Redis trouble is itself the alert
            alerts.append(f"watchdog could not read the fal queue: {type(exc).__name__}")
    for a in alerts:
        log.error("fal watchdog: %s", a)
    return alerts


# ---------------------------------------------------------------------------
# Operator CLI: python -m app.pipeline_provider <action> <task id> ...
# ---------------------------------------------------------------------------


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m app.pipeline_provider",
                                description="Fal provider task operator actions")
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="list active provider tasks")
    ls.add_argument("--all", action="store_true")
    for name in OPERATOR_ACTIONS:
        sp = sub.add_parser(name)
        sp.add_argument("task_id")
        if name == "adopt":
            sp.add_argument("request_id")
        sp.add_argument("--actor", required=True)
        sp.add_argument("--note", required=True, help="what you saw (e.g. on the Fal dashboard)")
    args = p.parse_args(argv)
    from .db import get_session

    with get_session() as db:
        if args.cmd == "list":
            q = select(ProviderTask).order_by(ProviderTask.created_at)
            if not args.all:
                q = q.where(_ACTIVE)
            for t in db.execute(q).scalars():
                print(f"{t.id} {t.status:20} held={t.reservation_held!s:5} "
                      f"pending={t.draw_pending!s:5} req={t.request_id or '-'} "
                      f"{t.owner_kind}:{t.owner_id}:{t.owner_slot} err={t.error_code or '-'}")
            return 0
        try:
            if args.cmd == "adopt":
                t = adopt(db, args.task_id, args.request_id, actor=args.actor, note=args.note)
            else:
                t = globals()[args.cmd](db, args.task_id, actor=args.actor, note=args.note)
        except ActionRefused as exc:
            print(f"refused ({exc.status_code}): {exc.detail}")
            return 1
        print(f"{t.id} is now {t.status}")
        if args.cmd == "close":
            print("Fal may still complete and bill this request; the charge stays on the day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
