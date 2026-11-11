"""
Daily kie.ai spend cap (AICP-155).

Two halves, deliberately asymmetric:

* **Admission** (:func:`reserve`) — every NEW credit-spending action (run
  create / copy / copy-batch / automation launch, Retry Run, segment rerun,
  range rerun, GPT Image regenerations, transcription, localisation, captions)
  reserves its *estimated* cost against the current UTC day, inside the caller's
  own transaction. If the day's total would exceed ``KIE_DAILY_BUDGET_USD`` the
  action is refused with :class:`BudgetExceeded` (the API maps it to HTTP 429)
  and nothing is written.
* **Measurement** (:func:`record_draw`, :func:`settle_draw`) — every paid
  kie.ai call the workers make (video task, image task, LLM call) is recorded
  in ``kie_spend_draws`` right after kie.ai accepted it, keyed by its taskId so
  a replayed submit or a double poll cannot count it twice, at the same
  estimate admission used; when the task's terminal state reports
  ``creditsConsumed`` the estimate is replaced by the real figure. A draw NEVER
  refuses and never raises: the work it measures was already admitted, and
  refusing it mid-pipeline is exactly how runs get stranded (see the
  segment-snapshot and segment-edit-guard incidents). A draw beyond what was
  reserved (poll-loop resubmits — up to RUN_TASK_MAX_ATTEMPTS per segment —,
  the reconcile tick re-billing a parked run, a call nobody reserved for) is
  counted as overdraft, so a day's total is always >= what was submitted and
  later admissions see it. The cap is therefore hard for ADMISSIONS; already
  admitted work can overshoot it by its own retries, never by new work.

Accounting model
----------------
``kie_spend_reservations`` holds one row per (scope_key, UTC day): ``reserved``
R, ``consumed`` C (sum of its draws) and ``charged`` = max(R, C).
``kie_spend_days`` holds the sum of ``charged`` per day. All amounts are integer
milli-credits. A reservation is attributed to the day it was made on, and its
later draws stay on that day — so "spend per day" means "cost of the work
admitted that UTC day" (a run launched at 23:50 that bills after midnight
counts against the earlier day).

``release_unused`` (Retry Run on a failed/incomplete run, segment rerun,
caption retry — actions on an IDLE scope) re-baselines R to C + estimate and
releases earlier days' unconsumed reservations for the scope, so segments that
never got submitted are not counted twice. Everything else is additive.

Concurrency
-----------
Every mutation of a day's numbers happens under that day's row lock, taken by an
``UPDATE kie_spend_days ... WHERE day = :d`` — a row lock on Postgres held to
commit, and the database-wide write lock on SQLite. Reservers and drawers
therefore serialize per day and the check+reserve is atomic. Lock-order rules:
:func:`reserve` flushes the caller's pending changes BEFORE taking the day lock
and the caller takes no other lock between it and its commit, so a reserver
never waits on anything while holding a day lock; several day locks are always
taken in ascending date order; a draw runs in its own short session and only
locks spend rows.

Cap semantics: ``KIE_DAILY_BUDGET_USD`` unset -> 250 USD; ``<= 0`` disables the
refusal but keeps the accounting (and the header readout). Accounting errors in
a draw are logged as ``ACCOUNTING FAILURE`` and leave the day understated.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from contextlib import AbstractContextManager
from datetime import date, datetime, timezone
from typing import Callable, Iterable, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from . import ai_models
from .config import settings
from .models import KieSpendDay, KieSpendReservation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost table
#
# kie.ai returns the real cost (`creditsConsumed`) only once a task finishes and
# publishes no rate card for these models, so admission has to estimate. Video
# rates are credits per second of the SUBMITTED CLIP (they already include
# output seconds — Seedance 2.5 bills ~38 credits/s of input + 38/s of output),
# calibrated from production RunSegment.cost_credits / clip length on
# 2026-10-16 and rounded up:
#
#   model          res    n    mean   max
#   seedance-2-5   480p   46   33.2   39.7
#   seedance-2-5   720p  303   74.1   88.7
#   seedance       720p   37   48.6   50.0
#   seedance       1080p   4  120.5  124.0
#   seedance-fast  720p   20   30.7   45.0
#   gemini-omni    720p    9   31.5   42.0
#
# Rows with no production data (2.5 at 1080p, mini, omni 4k) are scaled from
# their neighbours and erred high. A rate that is too high only makes the cap
# a little stricter; draws use the same table, so the books stay consistent.
# ---------------------------------------------------------------------------

CREDITS_PER_CLIP_SEC: dict[str, dict[str, float]] = {
    "seedance": {"480p": 25.0, "720p": 52.0, "1080p": 125.0},
    "seedance-fast": {"480p": 22.0, "720p": 45.0},
    "seedance-mini": {"480p": 18.0, "720p": 36.0},
    "seedance-2-5": {"480p": 40.0, "720p": 80.0, "1080p": 190.0},
    "gemini-omni": {"720p": 42.0, "1080p": 42.0, "4k": 84.0},
}
# Unknown model/resolution: the most expensive known rate.
_FALLBACK_CREDITS_PER_SEC = max(
    r for rates in CREDITS_PER_CLIP_SEC.values() for r in rates.values()
)

# GPT Image (image-to-image), measured live 2026-09-12: 6 credits at 1K, 10 at
# 2K for every variant (gpt-image-2, 2.5 flare, 2.5 sunburst).
IMAGE_CREDITS: dict[str, float] = {"1K": 6.0, "2K": 10.0, "4K": 16.0}

# Images-only (rebuild) video rates, credits per REQUESTED OUTPUT second (there
# is no input clip). docs/rebuild-from-reference-design.md §12:
#   kling-4-0     720p 28.5 / 1080p 42.5 — list price, matched exactly in 7/7
#                 tasks (2026-11-02 pilot + AICP-256 eval);
#   seedance-2-5  480p ~40 (AICP-243), 720p 63 measured -> 64, 1080p 158
#                 measured (AICP-256) -> 160. Erring high, as above.
# Kept apart from CREDITS_PER_CLIP_SEC on purpose: the v2v table and every v2v
# estimate stay byte-for-byte what they were.
IMAGES_ONLY_CREDITS_PER_SEC: dict[str, dict[str, float]] = {
    "kling-4-0": {"720p": 28.5, "1080p": 42.5},
    "seedance-2-5": {"480p": 40.0, "720p": 64.0, "1080p": 160.0},
}
_IMAGES_ONLY_FALLBACK_PER_SEC = max(
    r for rates in IMAGES_ONLY_CREDITS_PER_SEC.values() for r in rates.values()
)
# Rebuild keyframes and character sheets are always generated at 2K.
REBUILD_KEYFRAME_RESOLUTION = "2K"


def video_credits(model: Optional[str], resolution: Optional[str], clip_sec: float) -> float:
    """Estimated credits for one video generation from a clip of *clip_sec*."""
    spec = ai_models.spec_for(model)
    res = ai_models.resolution_or_default(model, resolution)
    rate = CREDITS_PER_CLIP_SEC.get(spec.key, {}).get(res, _FALLBACK_CREDITS_PER_SEC)
    sec = max(float(clip_sec or 0.0), spec.min_clip_sec, ai_models.MIN_SWAP_VIDEO_SEC)
    sec = min(sec, spec.max_clip_sec)
    return math.ceil(sec) * rate


def image_credits(resolution: Optional[str] = None) -> float:
    res = (resolution or settings.KIE_IMAGE_RESOLUTION or "1K").upper()
    return IMAGE_CREDITS.get(res, max(IMAGE_CREDITS.values()))


def llm_credits() -> float:
    return float(settings.KIE_LLM_CALL_EST_CREDITS)


def images_only_video_credits(
    model: Optional[str], resolution: Optional[str], request_sec: float,
) -> float:
    """Estimated credits for ONE images-only video task asking for
    *request_sec* seconds of output (an unknown model/resolution pays the most
    expensive known images-only rate)."""
    rate = IMAGES_ONLY_CREDITS_PER_SEC.get(model or "", {}).get(
        resolution or "", _IMAGES_ONLY_FALLBACK_PER_SEC
    )
    return float(math.ceil(float(request_sec or 0.0))) * rate


def storyboard_credits(n_shot_calls: int) -> float:
    """Reservation of one storyboard build (design §5.2): one global call plus
    one call per shot, each allowed one fallback."""
    return (1 + max(0, int(n_shot_calls))) * 2 * llm_credits()


def keyframes_credits(n_images: int) -> float:
    """Reservation of one keyframe batch: *n_images* GPT Image 2K tasks."""
    return max(0, int(n_images)) * image_credits(REBUILD_KEYFRAME_RESOLUTION)


# ---------------------------------------------------------------------------
# Units / cap
# ---------------------------------------------------------------------------


def usd(credits: float) -> float:
    return float(credits or 0.0) * settings.KIE_USD_PER_CREDIT


def cap_usd() -> Optional[float]:
    """The daily cap in USD, or None when enforcement is disabled (<= 0)."""
    value = float(settings.KIE_DAILY_BUDGET_USD or 0.0)
    return value if value > 0 else None


def cap_credits() -> Optional[float]:
    value = cap_usd()
    if value is None or settings.KIE_USD_PER_CREDIT <= 0:
        return None
    return value / settings.KIE_USD_PER_CREDIT


def utc_day(now: Optional[datetime] = None) -> date:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc)
    return now.date()


class BudgetExceeded(Exception):
    """A new credit-spending action does not fit in today's kie.ai budget."""

    def __init__(self, *, day: date, charged: float, needed: float, cap: float, kind: str):
        self.day = day
        self.charged = charged
        self.needed = needed
        self.cap = cap
        self.kind = kind
        super().__init__(self.message)

    @property
    def message(self) -> str:
        need = (
            f"this action ({self.kind}) needs about ${usd(self.needed):,.2f} more"
            if self.needed > 0 else f"nothing more can be started ({self.kind})"
        )
        return (
            f"Daily kie.ai budget reached: ${usd(self.charged):,.2f} of "
            f"${usd(self.cap):,.2f} already committed today "
            f"({self.day.isoformat()} UTC), and {need}. Nothing was started; "
            f"runs already in progress keep going. The budget resets at 00:00 "
            f"UTC — to raise it, change KIE_DAILY_BUDGET_USD."
        )


# ---------------------------------------------------------------------------
# Day lock
# ---------------------------------------------------------------------------


def _mc(credits: float) -> int:
    """Credits -> integer milli-credits (storage unit)."""
    return int(round(max(0.0, float(credits or 0.0)) * 1000))


def _cr(mcredits: int) -> float:
    return (mcredits or 0) / 1000.0


def _dialect_insert(db: Session):
    name = db.get_bind().dialect.name
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert


def _lock_day(db: Session, day: date, now: datetime) -> KieSpendDay:
    """Ensure *day*'s row exists, take its lock, and return it fresh."""
    insert = _dialect_insert(db)
    db.execute(
        insert(KieSpendDay.__table__)
        .values(day=day, charged_mcredits=0, updated_at=now)
        .on_conflict_do_nothing(index_elements=["day"])
    )
    # The UPDATE is the lock (row lock on Postgres, write lock on SQLite); the
    # SELECT after it then reads the committed state of every earlier holder
    # (READ COMMITTED takes a fresh snapshot per statement).
    db.execute(
        update(KieSpendDay)
        .where(KieSpendDay.day == day)
        .values(updated_at=now)
        .execution_options(synchronize_session=False)
    )
    return db.execute(
        select(KieSpendDay).where(KieSpendDay.day == day)
        .execution_options(populate_existing=True)
    ).scalar_one()


def _reservation(db: Session, scope_key: str, day: date) -> Optional[KieSpendReservation]:
    return db.execute(
        select(KieSpendReservation).where(
            KieSpendReservation.scope_key == scope_key,
            KieSpendReservation.day == day,
        ).execution_options(populate_existing=True)
    ).scalar_one_or_none()


def _new_reservation(db: Session, scope_key: str, day: date, kind: str,
                     run_id: Optional[str], now: datetime,
                     provider: str = "kie") -> KieSpendReservation:
    res = KieSpendReservation(
        scope_key=scope_key, day=day, kind=kind, run_id=run_id, provider=provider,
        reserved_mcredits=0, consumed_mcredits=0, charged_mcredits=0,
        created_at=now, updated_at=now,
    )
    db.add(res)
    db.flush()
    return res


def _recharge(day_row: KieSpendDay, res: KieSpendReservation, now: datetime) -> int:
    """Recompute ``res.charged`` and move the difference onto the day."""
    old = res.charged_mcredits or 0
    res.charged_mcredits = max(res.reserved_mcredits or 0, res.consumed_mcredits or 0)
    res.updated_at = now
    delta = res.charged_mcredits - old
    day_row.charged_mcredits = max(0, (day_row.charged_mcredits or 0) + delta)
    return delta


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def reserve(
    db: Session,
    scope_key: str,
    credits: float,
    *,
    kind: str,
    release_unused: bool = False,
    run_id: Optional[str] = None,
    now: Optional[datetime] = None,
    provider: str = "kie",
) -> float:
    """Atomically check and reserve *credits* for *scope_key* today.

    Runs in the CALLER's transaction (a refusal or a later rollback leaves no
    trace). The caller's pending ORM changes are flushed FIRST, so the commit
    that follows has nothing left to write but spend rows — the day lock is
    therefore never held while waiting on a Run/segment row lock. The caller
    must take no further lock between this call and its commit.

    Returns the change to today's total, in credits. Raises
    :class:`BudgetExceeded` when the cap is enabled and the day's total would
    go over it. An action whose reservation does not RAISE today's total
    (estimate 0, e.g. a delivery-only Retry Run) is always admitted.
    """
    db.flush()
    # AICP-363 §5.7: a manual paid admission on an assistant project pauses its
    # autopilot. Before the day lock; savepoint-guarded; never raises.
    if credits and credits > 0:
        from .autopilot.spend_view import note_reservation

        note_reservation(db, scope_key, credits, kind)
    now = now or datetime.now(timezone.utc)
    day = utc_day(now)
    want = _mc(credits)

    older: list = []
    if release_unused:
        # The scope's reservations from EARLIER days are dead too (the run is
        # idle): release their unconsumed part so a next-day Retry does not
        # count those segments twice. Past days' totals have no enforcement
        # meaning any more, so lowering them is safe. Day locks are always
        # taken in ascending date order, so two reservers cannot deadlock.
        older_days = sorted(set(db.execute(
            select(KieSpendReservation.day).where(
                KieSpendReservation.scope_key == scope_key,
                KieSpendReservation.day < day,
                KieSpendReservation.reserved_mcredits > KieSpendReservation.consumed_mcredits,
            )
        ).scalars()))
        for older_day in older_days:
            older_row = _lock_day(db, older_day, now)
            old_res = _reservation(db, scope_key, older_day)
            if old_res is not None and old_res.reserved_mcredits > old_res.consumed_mcredits:
                older.append((older_row, old_res))

    day_row = _lock_day(db, day, now)
    res = _reservation(db, scope_key, day)
    old_r = res.reserved_mcredits if res else 0
    old_c = res.consumed_mcredits if res else 0
    old_charged = res.charged_mcredits if res else 0

    base = old_c if release_unused else max(old_r, old_c)
    new_r = base + want
    delta = max(new_r, old_c) - old_charged

    cap = cap_credits()
    if cap is not None and delta > 0 and (day_row.charged_mcredits or 0) + delta > _mc(cap):
        log.warning(
            "kie spend: REFUSED %s (%s) — needs %.0f credits, day %s at %.0f/%.0f",
            kind, scope_key, _cr(delta), day, _cr(day_row.charged_mcredits), cap,
        )
        raise BudgetExceeded(
            day=day, charged=_cr(day_row.charged_mcredits), needed=_cr(delta),
            cap=cap, kind=kind,
        )

    for older_row, old_res in older:
        old_res.reserved_mcredits = old_res.consumed_mcredits
        _recharge(older_row, old_res, now)
    if res is None:
        res = _new_reservation(db, scope_key, day, kind, run_id, now, provider)
    res.reserved_mcredits = new_r
    if run_id and not res.run_id:
        res.run_id = run_id
    _recharge(day_row, res, now)
    db.flush()
    log.info(
        "kie spend: reserved %.0f credits for %s (%s); day %s now %.0f credits "
        "($%.2f of %s)",
        _cr(want), kind, scope_key, day, _cr(day_row.charged_mcredits),
        usd(_cr(day_row.charged_mcredits)),
        f"${cap_usd():.2f}" if cap_usd() else "no cap",
    )
    return _cr(delta)


def reserve_many(
    db: Session,
    items: list,
    *,
    provider: str = "kie",
    now: Optional[datetime] = None,
) -> float:
    """:func:`reserve` for SEVERAL new scopes in one go (batch VOID fills,
    docs/segment-fill-batch-design.md R1-2): *items* are ``(scope_key,
    credits, kind)``. The caller's writes are flushed and every manual-spend
    note added BEFORE the day lock, the day row is locked ONCE, the SUM is
    checked against the cap (one refusal refuses all), then every reservation
    row is written. The caller must take no further lock before its commit.
    Returns the change to today's total, in credits."""
    db.flush()
    from .autopilot.spend_view import note_reservation

    items = [(k, float(c), kind) for k, c, kind in items]
    for key, credits, kind in items:
        if credits > 0:
            note_reservation(db, key, credits, kind)
    db.flush()
    now = now or datetime.now(timezone.utc)
    day = utc_day(now)
    day_row = _lock_day(db, day, now)
    plans, total = [], 0
    for key, credits, kind in items:
        res = _reservation(db, key, day)
        old_r = res.reserved_mcredits if res else 0
        old_c = res.consumed_mcredits if res else 0
        old_charged = res.charged_mcredits if res else 0
        new_r = max(old_r, old_c) + _mc(credits)
        delta = max(new_r, old_c) - old_charged
        plans.append((key, kind, res, new_r))
        total += max(0, delta)
    cap = cap_credits()
    if cap is not None and total > 0 and (day_row.charged_mcredits or 0) + total > _mc(cap):
        log.warning("kie spend: REFUSED batch of %d (%s) — needs %.0f credits, day %s at "
                    "%.0f/%.0f", len(items), ", ".join(k for k, _c, _k in items), _cr(total),
                    day, _cr(day_row.charged_mcredits), cap)
        raise BudgetExceeded(day=day, charged=_cr(day_row.charged_mcredits),
                             needed=_cr(total), cap=cap, kind=items[0][2] if items else "batch")
    before = day_row.charged_mcredits or 0
    for key, kind, res, new_r in plans:
        if res is None:
            res = _new_reservation(db, key, day, kind, None, now, provider)
        res.reserved_mcredits = new_r
        _recharge(day_row, res, now)
    db.flush()
    log.info("kie spend: reserved a batch of %d scopes; day %s now %.0f credits", len(items),
             day, _cr(day_row.charged_mcredits))
    return _cr((day_row.charged_mcredits or 0) - before)


def reserve_committed(
    sessions: Callable[[], AbstractContextManager],
    scope_key: str,
    credits: float,
    *,
    kind: str,
    run_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> float:
    """:func:`reserve` in its own short, immediately committed transaction —
    for callers that are about to make a SLOW paid call themselves (the
    synchronous translation endpoint, the post-analysis transcription), which
    must not hold the day lock across it."""
    with sessions() as db:
        delta = reserve(db, scope_key, credits, kind=kind, run_id=run_id, now=now)
        db.commit()
        return delta


def reserve_computed(
    db: Session,
    scope_key: str,
    estimate_fn: Callable[[Session], float],
    *,
    kind: str,
    release_unused: bool = False,
    run_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> float:
    """:func:`reserve` with the estimate computed UNDER the locks (design §7.3).

    Order: flush the caller's changes -> the scope lock (:func:`_lock_scope`)
    -> every day lock this call will touch, ascending (the earlier days
    *release_unused* re-baselines for this scope, then today) ->
    ``estimate_fn(db)`` -> re-baseline + reserve. So a draw that committed
    before the locks is visible to the estimate, and one still waiting (on the
    scope lock) lands after our commit and chooses its day from the
    re-baselined reservations. *estimate_fn* must read with plain SELECTs (no
    row locks): the draw path takes "paid-call row -> scope lock -> day lock",
    and a row lock taken here after the scope lock would close a cycle.

    Used by the rebuild feature only (images-only runs, storyboards, keyframe
    batches). Unlike :func:`reserve` it does not ping the autopilot (a rebuild
    project has no autopilot, design §1 non-goals). Returns the change to
    today's total; raises :class:`BudgetExceeded` like :func:`reserve`.
    """
    db.flush()
    _lock_scope(db, scope_key)
    now = now or datetime.now(timezone.utc)
    day = utc_day(now)

    older: list = []
    if release_unused:
        older_days = sorted(set(db.execute(
            select(KieSpendReservation.day).where(
                KieSpendReservation.scope_key == scope_key,
                KieSpendReservation.day < day,
                KieSpendReservation.reserved_mcredits > KieSpendReservation.consumed_mcredits,
            )
        ).scalars()))
        for older_day in older_days:
            older_row = _lock_day(db, older_day, now)
            old_res = _reservation(db, scope_key, older_day)
            if old_res is not None and old_res.reserved_mcredits > old_res.consumed_mcredits:
                older.append((older_row, old_res))

    day_row = _lock_day(db, day, now)
    credits = float(estimate_fn(db) or 0.0)
    want = _mc(credits)
    res = _reservation(db, scope_key, day)
    old_r = res.reserved_mcredits if res else 0
    old_c = res.consumed_mcredits if res else 0
    old_charged = res.charged_mcredits if res else 0

    base = old_c if release_unused else max(old_r, old_c)
    new_r = base + want
    delta = max(new_r, old_c) - old_charged

    cap = cap_credits()
    if cap is not None and delta > 0 and (day_row.charged_mcredits or 0) + delta > _mc(cap):
        log.warning(
            "kie spend: REFUSED %s (%s) — needs %.0f credits, day %s at %.0f/%.0f",
            kind, scope_key, _cr(delta), day, _cr(day_row.charged_mcredits), cap,
        )
        raise BudgetExceeded(
            day=day, charged=_cr(day_row.charged_mcredits), needed=_cr(delta),
            cap=cap, kind=kind,
        )

    for older_row, old_res in older:
        old_res.reserved_mcredits = old_res.consumed_mcredits
        _recharge(older_row, old_res, now)
    if res is None:
        res = _new_reservation(db, scope_key, day, kind, run_id, now, "kie")
    res.reserved_mcredits = new_r
    if run_id and not res.run_id:
        res.run_id = run_id
    _recharge(day_row, res, now)
    db.flush()
    log.info(
        "kie spend: reserved %.0f credits (computed under lock) for %s (%s); day %s "
        "now %.0f credits",
        credits, kind, scope_key, day, _cr(day_row.charged_mcredits),
    )
    return _cr(delta)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def scope_lock_key(scope_key: str) -> int:
    """Stable signed 64-bit key of *scope_key* for ``pg_advisory_xact_lock``."""
    import hashlib

    return int.from_bytes(
        hashlib.sha256(scope_key.encode("utf-8")).digest()[:8], "big", signed=True
    )


def _lock_scope(db: Session, scope_key: str) -> None:
    """Take the per-scope lock (design §7.3 "Scope lock") until commit.

    PostgreSQL: a transaction-level advisory lock on a stable hash of
    *scope_key*. SQLite: nothing — the database-wide write lock the caller's
    flushed writes already hold serialises every writer. Used only by the
    rebuild feature's scopes (images-only runs, storyboards, keyframe batches);
    no v2v path takes it.
    """
    if db.get_bind().dialect.name != "postgresql":
        return
    from sqlalchemy import text as _text

    db.execute(_text("SELECT pg_advisory_xact_lock(:k)"), {"k": scope_lock_key(scope_key)})


def _apply_draw(
    db: Session, scope_key: str, call_key: str, mcredits: int, kind: str,
    now: datetime, *, settle: bool, provider: str = "kie", scope_lock: bool = False,
) -> None:
    from .models import KieSpendDraw

    if scope_lock:
        # Scope lock BEFORE the target day is chosen: a concurrent
        # reserve_computed re-baselining this scope finishes first, so the day
        # picked below reflects its reservations (design §7.3, round 9).
        _lock_scope(db, scope_key)
    existing = db.execute(
        select(KieSpendDraw).where(KieSpendDraw.call_key == call_key)
    ).scalar_one_or_none()
    if existing is not None:
        res_day = db.execute(
            select(KieSpendReservation.day).where(
                KieSpendReservation.id == existing.reservation_id
            )
        ).scalar_one()
    else:
        res_day = _draw_target_day(db, scope_key, mcredits) or utc_day(now)

    # Day lock FIRST (same order as reserve), then re-read under it.
    day_row = _lock_day(db, res_day, now)
    existing = db.execute(
        select(KieSpendDraw).where(KieSpendDraw.call_key == call_key)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()

    if existing is not None:
        if not settle or existing.settled_at is not None and existing.mcredits == mcredits:
            return  # duplicate observation of the same call — idempotent
        res = db.get(KieSpendReservation, existing.reservation_id, populate_existing=True)
        res.consumed_mcredits = max(0, res.consumed_mcredits - existing.mcredits + mcredits)
        existing.mcredits = mcredits
        existing.settled_at = now
        _recharge(day_row, res, now)
        return

    res = _reservation(db, scope_key, res_day)
    if res is None:
        # Spend nobody admitted (a call site with no reservation, a task
        # submitted before this feature deployed): still real, so it lands as
        # pure overdraft on today.
        log.warning(
            "kie spend: %s call %s has no reservation for %s — recording as "
            "overdraft", kind, call_key, scope_key,
        )
        res = _new_reservation(db, scope_key, res_day, kind, None, now, provider)
    db.add(KieSpendDraw(
        call_key=call_key, reservation_id=res.id, kind=kind, provider=provider,
        estimated_mcredits=mcredits, mcredits=mcredits, created_at=now,
        settled_at=now if settle else None,
    ))
    res.consumed_mcredits = (res.consumed_mcredits or 0) + mcredits
    delta = _recharge(day_row, res, now)
    if delta > 0 and res.reserved_mcredits:
        log.warning(
            "kie spend: %s (%s) drew %.0f credits beyond its reservation; day %s "
            "now %.0f credits",
            kind, scope_key, _cr(delta), res_day, _cr(day_row.charged_mcredits),
        )


def _draw_target_day(db: Session, scope_key: str, mcredits: int) -> Optional[date]:
    """Which of the scope's reservations a new draw is charged to.

    Newest first, the first with room for the whole draw; else the newest with
    any room; else the newest. So a video segment of a run launched yesterday
    lands on yesterday's reservation (which reserved it) even after today's
    small GPT Image regen opened a second one, while today's image draw lands
    on today's.
    """
    rows = db.execute(
        select(
            KieSpendReservation.day,
            KieSpendReservation.reserved_mcredits,
            KieSpendReservation.consumed_mcredits,
        )
        .where(KieSpendReservation.scope_key == scope_key)
        .order_by(KieSpendReservation.day.desc())
    ).all()
    if not rows:
        return None
    for day, r, c in rows:
        if (r or 0) - (c or 0) >= mcredits:
            return day
    for day, r, c in rows:
        if (r or 0) - (c or 0) > 0:
            return day
    return rows[0][0]


def _run_draw(scope_key, call_key, credits, kind, sessions, now, settle) -> None:
    try:
        if sessions is None:
            from .db import get_session as sessions  # noqa: N806
        now = now or datetime.now(timezone.utc)
        with sessions() as db:
            _apply_draw(
                db, scope_key or "unscoped", str(call_key), _mc(credits), kind,
                now, settle=settle, provider="kie",
            )
            db.commit()
    except Exception:  # accounting must never fail a paid call that happened
        log.exception(
            "kie spend: ACCOUNTING FAILURE — could not record %.0f credits for "
            "%s call %s (%s); the call itself went through and today's total "
            "is now understated", credits or 0.0, kind, call_key, scope_key,
        )


def record_draw(
    scope_key: Optional[str],
    credits: float,
    *,
    kind: str,
    call_key: Optional[str] = None,
    sessions: Optional[Callable[[], AbstractContextManager]] = None,
    now: Optional[datetime] = None,
) -> None:
    """Record a paid kie.ai call that has ALREADY been made. Never raises.

    *call_key* identifies the call (the kie.ai taskId); recording the same key
    twice is a no-op, so a replayed submit or a double poll cannot double-count.
    LLM calls have no task id and pass none (a fresh id is generated).
    *sessions* is the caller module's ``get_session`` (so tests that patch a
    pipeline module's session factory see the draw in their DB). The draw
    commits in its own short transaction, independent of the caller's.
    """
    if not call_key:
        call_key = f"{kind}:{uuid.uuid4().hex}"
    _run_draw(scope_key, call_key, credits, kind, sessions, now, settle=False)


def draw_in_session(
    db: Session,
    scope_key: str,
    call_key: str,
    credits: float,
    *,
    kind: str,
    provider: str,
    now: Optional[datetime] = None,
    scope_lock: bool = False,
) -> None:
    """Record a paid call in the CALLER's transaction, and RAISE on failure.

    The strict sibling of :func:`record_draw` for callers whose own commit must
    carry the draw (the Fal provider layer: a task is ``submitted`` only if its
    draw exists, docs/fal-provider-design.md §5). The caller's pending changes
    are flushed first, so any row lock the caller's UPDATEs took is held BEFORE
    the day lock — the order :func:`reserve` uses. Idempotent by *call_key*.
    Never commits.
    """
    db.flush()
    _apply_draw(
        db, scope_key, str(call_key), _mc(credits), kind,
        now or datetime.now(timezone.utc), settle=False, provider=provider,
        scope_lock=scope_lock,
    )
    db.flush()


def settle_in_session(
    db: Session,
    scope_key: str,
    call_key: str,
    credits: float,
    *,
    kind: str,
    provider: str = "kie",
    now: Optional[datetime] = None,
    scope_lock: bool = True,
) -> None:
    """Settle a paid call's draw to *credits* in the CALLER's transaction and
    RAISE on failure — the strict sibling of :func:`settle_draw`, for the
    rebuild paid-call rows whose terminal state and settled draw must commit
    together (design §7.2 "Accounting"). Idempotent (sets, never adds); a call
    with no draw yet is recorded at *credits*. Takes the scope lock by default
    (every rebuild scope is scope-locked). Never commits."""
    db.flush()
    _apply_draw(
        db, scope_key, str(call_key), _mc(credits), kind,
        now or datetime.now(timezone.utc), settle=True, provider=provider,
        scope_lock=scope_lock,
    )
    db.flush()


def has_draw(db: Session, call_key: str) -> bool:
    """Whether a draw for *call_key* exists (the ledger is the source of truth)."""
    from .models import KieSpendDraw

    return db.execute(
        select(KieSpendDraw.id).where(KieSpendDraw.call_key == call_key)
    ).first() is not None


def settle_draw(
    call_key: Optional[str],
    actual_credits: Optional[float],
    *,
    scope_key: Optional[str],
    kind: str,
    sessions: Optional[Callable[[], AbstractContextManager]] = None,
    now: Optional[datetime] = None,
) -> None:
    """Replace a call's estimate with kie.ai's reported ``creditsConsumed``.

    Idempotent (sets, never adds). Never raises. A task with no draw yet
    (submitted before this feature deployed) is recorded at its actual cost.
    """
    if not call_key or actual_credits is None:
        return
    _run_draw(scope_key, call_key, actual_credits, kind, sessions, now, settle=True)


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------


def today_summary(db: Session, now: Optional[datetime] = None) -> dict:
    day = utc_day(now)
    charged = _cr(db.execute(
        select(KieSpendDay.charged_mcredits).where(KieSpendDay.day == day)
    ).scalar_one_or_none() or 0)
    cap = cap_credits()
    return {
        "day": day,
        "charged_credits": charged,
        "charged_usd": usd(charged),
        "cap_credits": cap,
        "cap_usd": cap_usd(),
        "remaining_credits": None if cap is None else max(0.0, cap - charged),
        "over": cap is not None and charged >= cap,
    }


def headroom_credits(db: Session, now: Optional[datetime] = None) -> Optional[float]:
    """Unlocked read of today's remaining budget (None = no cap). Advisory
    only — :func:`reserve` is the authoritative check."""
    return today_summary(db, now)["remaining_credits"]


# ---------------------------------------------------------------------------
# Estimates for whole actions
# ---------------------------------------------------------------------------


def _clip_sec(sd) -> float:
    return (
        float(sd.end_sec) - float(sd.start_sec)
        + float(getattr(sd, "pre_roll_sec", 0.0) or 0.0)
        + float(getattr(sd, "post_roll_sec", 0.0) or 0.0)
    )


def preprocess_image_count(run, *, fresh: bool = False) -> int:
    """GPT Image tasks the run's preprocessing phase would still create
    (*fresh*: as if nothing had been generated yet — a copy of *run*)."""
    if not getattr(run, "preprocess_enabled", False):
        return 0
    refs = list(run.reference_image_urls or [])
    prompts = list(run.preprocess_image_prompts or [])
    done = [] if fresh else list(run.preprocessed_reference_urls or [])
    task_ids = [] if fresh else list(run.preprocess_task_ids or [])
    if (run.preprocess_mode or "per_image") == "all_as_refs":
        if not refs or not (prompts[0] if prompts else "").strip():
            return 0
        if (done and done[0]) or (task_ids and task_ids[0]):
            return 0
        return 1
    count = 0
    for i in range(1, len(refs)):
        prompt = (prompts[i] if i < len(prompts) else "") or ""
        if not prompt.strip():
            continue
        if i < len(done) and done[i]:
            continue
        if i < len(task_ids) and task_ids[i]:
            continue
        count += 1
    return count


def segment_is_paid(rs) -> bool:
    """True when submit would NOT create a new kie.ai task for *rs*: it has a
    result on disk, or a kie.ai task that exists. A submitted/generating row
    WITHOUT a task id never reached kie.ai (its createTask failed), and a
    completed row whose result file is gone gets regenerated — both bill."""
    from .state_machine import SegmentStatus

    if rs.status == SegmentStatus.completed:
        return bool(rs.local_result_path and os.path.exists(rs.local_result_path))
    if rs.status in (SegmentStatus.submitted, SegmentStatus.generating):
        return bool(rs.seedance_task_id)
    return False


def estimate_run_credits(
    db: Session, run, *, include_preprocess: Optional[bool] = None,
    fresh: bool = False,
) -> float:
    """Credits the run would still spend from where it stands now.

    Counts every active swap segment that has no finished result and no live
    kie.ai task (exactly what the submit phase would create a task for), plus
    the GPT Image tasks preprocessing would still create when the run is about
    to (re)enter that phase. *fresh* ignores the run's own progress — the cost
    of a brand-new copy of it (the automation tick's pre-trim).
    """
    from .models import RunSegment, VideoProject
    from .pipeline_v2 import _active_swap_segment_ids, run_segment_defs
    from .state_machine import RunStatus, SegmentStatus

    if is_images_only_run(run):
        # Images-only runs have their own estimator (design §7.3); reached
        # here only from a caller that did not branch itself.
        return estimate_images_only_run_credits(db, run, fresh=fresh)
    db.flush()  # sessions here run autoflush=False; see just-added RunSegments
    project = db.get(VideoProject, run.project_id)
    if project is None:
        return 0.0
    seg_defs = run_segment_defs(run, project)
    active = _active_swap_segment_ids(run.stitch_plan, seg_defs)
    rs_by_def = {} if fresh else {
        rs.segment_def_id: rs
        for rs in db.execute(
            select(RunSegment).where(RunSegment.run_id == run.id)
        ).scalars()
    }
    from .project_types import is_void_source_run

    void_source = is_void_source_run(run)
    total = 0.0
    for sd in seg_defs:
        if sd.action != "swap" or sd.id not in active:
            continue
        rs = rs_by_def.get(sd.id)
        if rs is not None and segment_is_paid(rs):
            continue
        if void_source and (fresh or rs is None or not rs.model_fallback):
            # A VOID-from-the-source run submits only its model-fallback
            # segments (docs/segment-fill-batch-design.md §7); VOID fills
            # reserve on their own provider tasks.
            continue
        total += video_credits(run.model, run.resolution, _clip_sec(sd))
    if include_preprocess is None:
        include_preprocess = run.status in (RunStatus.created, RunStatus.preprocessing)
    if include_preprocess:
        total += preprocess_image_count(run, fresh=fresh) * image_credits()
    return total


def reserve_run(
    db: Session,
    run,
    *,
    kind: str,
    release_unused: bool = False,
    include_preprocess: Optional[bool] = None,
    extra_credits: float = 0.0,
    now: Optional[datetime] = None,
) -> float:
    """Reserve what *run* would still spend (see :func:`estimate_run_credits`).

    An images-only run is reserved by :func:`reserve_images_only_run` (the
    estimate computed under the scope + day locks); every v2v run takes the
    unchanged path below."""
    if is_images_only_run(run):
        return reserve_images_only_run(
            db, run, kind=kind, release_unused=release_unused,
            extra_credits=extra_credits, now=now,
        )
    est = estimate_run_credits(db, run, include_preprocess=include_preprocess)
    return reserve(
        db,
        run_scope(run.id),
        est + extra_credits,
        kind=kind,
        release_unused=release_unused,
        run_id=run.id,
        now=now,
    )


# ---------------------------------------------------------------------------
# Images-only (rebuild) runs — design §7.3
# ---------------------------------------------------------------------------

# Automatic character-check call slots of one succeeded attempt (design §6.5):
# 0 = the primary model, 1 = its one fallback. An operator's later "Run
# character check" is a separately reserved call (slot >= 2) and not part of
# this allowance.
CHECK_AUTO_SLOTS = (0, 1)


def is_images_only_run(run) -> bool:
    return getattr(run, "generation_mode", None) == "images_only"


def images_only_chunk_request_sec(run, segment_def_id: str, chunk_sec: float) -> int:
    """The frozen ``request_sec`` of a chunk (rebuild_snapshot), else derived."""
    snap = getattr(run, "rebuild_snapshot", None) or {}
    for chunk in snap.get("chunks") or []:
        if chunk.get("segment_def_id") == segment_def_id and chunk.get("request_sec"):
            return int(chunk["request_sec"])
    try:
        return ai_models.images_only_duration(run.model, chunk_sec)
    except ValueError:
        spec = ai_models.spec_for(run.model)
        return int(spec.images_only_max_sec or math.ceil(chunk_sec))


def check_allowance_credits(statuses_by_slot: dict, *, verdict: bool = False) -> float:
    """Undrawn remainder of one succeeded attempt's automatic character-check
    allowance (design §7.3). *statuses_by_slot* maps slot -> status of the
    call row in that slot (absent = no row yet).

    A slot counts ``llm_credits()`` while its call is still undrawn (no row,
    or the row is ``queued``/``claimed``); a row that reached ``sending`` has
    drawn its estimate and counts 0, as does a terminal one. The fallback slot
    counts only while it may still be needed: until the check has a VERDICT
    (*verdict*: the attempt's ``qa`` holds ``character_check`` — a valid answer
    or ``unavailable``). A primary that "succeeded" at the protocol level with
    an unusable answer still gets its fallback (integration gate round 1: the
    old "not once the primary succeeded" rule let a re-baseline release that
    fallback's headroom).
    """
    unsent = ("queued", "claimed")
    total = 0.0
    primary = statuses_by_slot.get(0)
    if not verdict and (primary is None or primary in unsent):
        total += llm_credits()
    fallback = statuses_by_slot.get(1)
    if not verdict and (fallback is None or fallback in unsent):
        total += llm_credits()
    return total


def estimate_images_only_run_credits(db: Session, run, *, fresh: bool = False) -> float:
    """Credits an images-only run would still spend (design §7.3).

    Everything still UNDRAWN (a paid call draws its estimate at ``sending``):

    * every ACTIVE swap chunk whose RunSegment is ``pending`` (or missing) and
      has no live attempt: ``IMAGES_ONLY_CREDITS_PER_SEC x request_sec`` plus
      ``2 x llm_credits()`` (the attempt's character check and its fallback);
    * every live attempt still ``queued``/``claimed`` (admitted, not yet
      sent): its ``estimate_credits`` plus ``2 x llm_credits()``;
    * every live attempt already ``sending``/``submitted``/``fetching`` (its
      video draw is in): ``2 x llm_credits()`` for the check still to come;
    * the undrawn character-check allowance of every succeeded attempt
      (:func:`check_allowance_credits`). Failed / unknown attempts get no
      check and count 0.

    So a Retry / rerun's ``release_unused`` re-baseline never releases the
    allowance of an attempt that is about to draw. Plain reads only (no row
    locks): it runs inside :func:`reserve_computed` after the scope and day
    locks. *fresh*: as if nothing had run.
    """
    from .models import RunSegment, VideoProject
    from .pipeline_v2 import _active_swap_segment_ids, run_segment_defs
    from .rebuild_models import (
        PAID_CALL_LIVE_STATUSES,
        RebuildChunkAttempt,
        RebuildLlmCall,
    )
    from .state_machine import SegmentStatus

    project = db.get(VideoProject, run.project_id)
    if project is None:
        return 0.0
    seg_defs = run_segment_defs(run, project)
    active = _active_swap_segment_ids(run.stitch_plan, seg_defs)
    rs_by_def = {} if fresh else {
        rs.segment_def_id: rs
        for rs in db.execute(select(RunSegment).where(RunSegment.run_id == run.id)).scalars()
    }
    live = [] if fresh else db.execute(
        select(RebuildChunkAttempt.run_segment_id, RebuildChunkAttempt.status,
               RebuildChunkAttempt.estimate_credits).where(
            RebuildChunkAttempt.run_id == run.id,
            RebuildChunkAttempt.status.in_(PAID_CALL_LIVE_STATUSES),
        )
    ).all()
    live_segments = {seg_id for seg_id, _st, _est in live}
    total = 0.0
    for _seg_id, status, est in live:
        if status in ("queued", "claimed"):
            total += float(est or 0.0)
        total += 2 * llm_credits()
    for sd in seg_defs:
        if sd.action != "swap" or sd.id not in active:
            continue
        rs = rs_by_def.get(sd.id)
        if rs is not None:
            status = rs.status.value if hasattr(rs.status, "value") else rs.status
            if status != SegmentStatus.pending.value or rs.id in live_segments:
                continue
        request_sec = images_only_chunk_request_sec(
            run, sd.id, float(sd.end_sec) - float(sd.start_sec)
        )
        total += images_only_video_credits(run.model, run.resolution, request_sec)
        total += 2 * llm_credits()
    if fresh:
        return total
    succeeded_rows = db.execute(
        select(RebuildChunkAttempt.id, RebuildChunkAttempt.qa).where(
            RebuildChunkAttempt.run_id == run.id,
            RebuildChunkAttempt.status == "succeeded",
        )
    ).all()
    succeeded = [aid for aid, _qa in succeeded_rows]
    verdicts = {aid: "character_check" in (qa or {}) for aid, qa in succeeded_rows}
    if succeeded:
        slots: dict = {aid: {} for aid in succeeded}
        for aid, slot, status in db.execute(
            select(RebuildLlmCall.attempt_id, RebuildLlmCall.slot, RebuildLlmCall.status).where(
                RebuildLlmCall.attempt_id.in_(succeeded),
                RebuildLlmCall.kind == "character_check",
                RebuildLlmCall.slot.in_(CHECK_AUTO_SLOTS),
            )
        ).all():
            slots[aid][int(slot)] = status
        for aid in succeeded:
            total += check_allowance_credits(slots[aid], verdict=verdicts[aid])
    return total


def reserve_images_only_run(
    db: Session,
    run,
    *,
    kind: str,
    release_unused: bool = False,
    extra_credits: float = 0.0,
    now: Optional[datetime] = None,
) -> float:
    """Reserve an images-only run's remaining cost, estimated under the locks
    (:func:`reserve_computed`). The SAME estimator serves launch, Retry Run and
    segment rerun; the caller resets segments to ``pending`` (and creates the
    RunSegments at launch) BEFORE calling, in the same transaction."""
    return reserve_computed(
        db,
        run_scope(run.id),
        lambda s: estimate_images_only_run_credits(s, run) + float(extra_credits or 0.0),
        kind=kind,
        release_unused=release_unused,
        run_id=run.id,
        now=now,
    )


def reserve_new_runs(db: Session, runs: Iterable, *, kind: str, now: Optional[datetime] = None) -> None:
    for run in runs:
        reserve_run(db, run, kind=kind, include_preprocess=bool(run.preprocess_enabled), now=now)


# Scope keys — one place so admission and draws can never disagree.
def run_scope(run_id: str) -> str:
    return f"run:{run_id}"


def range_scope(attempt_id: str) -> str:
    return f"range:{attempt_id}"


def segment_image_scope(job_id: str) -> str:
    return f"segimg:{job_id}"


def caption_scope(caption_id: str) -> str:
    return f"caption:{caption_id}"


def transcribe_scope(project_id: str) -> str:
    return f"transcribe:{project_id}"


def localise_scope(project_id: str) -> str:
    return f"localise:{project_id}"


def transcribe_job_scope(job_id: str) -> str:
    """One durable transcription job (AICP-351 §5.2: its 3-attempt worst case
    is reserved and released per job, never shared with another job)."""
    return f"transcribe-job:{job_id}"


def localise_job_scope(job_id: str) -> str:
    """One durable translate job (``localisation_prompt_jobs``, AICP-351)."""
    return f"localise-job:{job_id}"


def storyboard_scope(storyboard_id: str) -> str:
    """Scope of one storyboard build (all its LLM call rows)."""
    return f"storyboard:{storyboard_id}"


def keyframes_scope(batch_id: str) -> str:
    """Scope of one keyframe batch (all its GPT Image rows)."""
    return f"keyframes:{batch_id}"


def rebuild_call_key(row_id: str) -> str:
    """Draw key of one rebuild paid-call row: one row, one request, one draw."""
    return f"rebuild:{row_id}"


def provider_task_scope(task_id: str) -> str:
    return f"fal:{task_id}"


def provider_task_call_key(task_id: str) -> str:
    """One task, one POST, one draw: the key is known before Fal answers."""
    return f"fal:task:{task_id}"
