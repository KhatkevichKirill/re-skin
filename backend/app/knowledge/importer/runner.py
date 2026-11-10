"""Dry-run / apply driver, cursors, leases and the RQ tick (design §6.3, §11).

* **Dry run**: one transaction over every selected source, the full report,
  then ROLLBACK — nothing is written (the batch row itself included). It takes
  no Run row locks and no cursor leases, so it never blocks the pipeline.
* **Apply**: one ``kb_import_batches`` row; each source row is processed in a
  savepoint and committed together with its cursor advance, so the Run lock
  (§6.2) is held for one row at a time. A failing row is rolled back to its
  savepoint, reported, and the cursor moves past it (``from_start`` retries).
* Cursors: per source on ``(created_at, id)``; ``kb_import_cursors`` lease
  columns stop two importers working the same source.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

from sqlalchemy import select, text, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

from .. import store
from ..models import KbImportCursor
from . import reconcile
from .context import MANIFEST_PATH, MANIFEST_SHA256, ImportContext, LeaseLost, MediaChanged
from .seed import SEED_DIR, import_seed
from .sources import COMMIT_LAG, SOURCES, SOURCE_NAMES, _utc, horizon, write_export

log = logging.getLogger(__name__)

TICK_LIMIT = 200
LEASE_TTL_SEC = 900
TICK_MAX_SEC = 600  # stop taking new rows well before the leases could lapse
REDIS_LEASE_KEY = "reskin:knowledge:import:lease"
STEPS = SOURCE_NAMES + ("seed", "reconcile")
_RELEASE_LUA = 'if redis.call("GET", KEYS[1]) == ARGV[1] then return redis.call("DEL", KEYS[1]) else return 0 end'
_REFRESH_LUA = ('if redis.call("GET", KEYS[1]) == ARGV[1] then return redis.call("EXPIRE", KEYS[1], ARGV[2]) '
                'else return 0 end')


def code_revision() -> str:
    return os.getenv("RESKIN_CODE_REVISION", "unknown")


# ---------------------------------------------------------------- cursors / leases


def read_cursor(session, source: str) -> Optional[dict]:
    row = session.get(KbImportCursor, source)
    return row.cursor if row is not None else None


def write_cursor(session, source: str, cursor: Optional[dict], owner: str) -> None:
    stmt = pg_insert(KbImportCursor).values(source=source, cursor=cursor, updated_at=datetime.now(timezone.utc))
    session.execute(stmt.on_conflict_do_update(
        index_elements=["source"], set_={"cursor": stmt.excluded.cursor, "updated_at": stmt.excluded.updated_at},
        where=KbImportCursor.lease_owner == owner))


def acquire_lease(session_factory, source: str, owner: str, ttl_sec: int = LEASE_TTL_SEC) -> bool:
    with session_factory() as s:
        s.execute(pg_insert(KbImportCursor).values(source=source).on_conflict_do_nothing(index_elements=["source"]))
        got = s.execute(text(
            "UPDATE kb_import_cursors SET lease_owner = :o, lease_expires_at = now() + make_interval(secs => :t), "
            "updated_at = now() WHERE source = :s AND (lease_owner IS NULL OR lease_expires_at < now() "
            "OR lease_owner = :o) RETURNING source"), {"o": owner, "s": source, "t": ttl_sec}).first()
        s.commit()
        return got is not None


def release_lease(session_factory, source: str, owner: str) -> None:
    with session_factory() as s:
        s.execute(text("UPDATE kb_import_cursors SET lease_owner = NULL, lease_expires_at = NULL, updated_at = now() "
                       "WHERE source = :s AND lease_owner = :o"), {"s": source, "o": owner})
        s.commit()


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------- sources


PENDING_CAP = 2000
EXPORT_CAP = 2000  # queued export events per source cursor; full = the source stops before the next one
EXPORT_TRIES = 3
REVISIT_SHARE = 4  # revisits take at most budget // 4 per pass, so new rows and draw coverage keep moving
PENDING_MAX_AGE = timedelta(days=7)  # runs only; one that becomes done later is caught by reconcile_finals


def _revisit_count(budget: Optional[int], pending: int, turn: str) -> int:
    if budget is None:
        return pending
    if budget == 1:
        return min(pending, 1) if turn == "revisit" else 0  # alternate passes
    return min(pending, max(1, budget // REVISIT_SHARE), budget - 1)  # >= 1 left for new rows


def run_source(ctx: ImportContext, src, budget: Optional[int], owner: str, from_start: bool,
               until: Optional[datetime] = None) -> tuple[int, bool]:
    """Revisit rows imported while non-terminal, then scan new rows below the commit
    horizon after the cursor. Returns (rows, exhausted).

    A row that is not terminal yet is imported as it stands and kept in the
    cursor's ``pending`` list; passes re-process it (idempotently, a bounded
    share of the budget, rotating through the list) until it is terminal or
    gone; only Runs age out (``PENDING_MAX_AGE``), since final reconciliation
    recovers a Run that becomes done later. A non-terminal row the full list
    cannot track is not imported and the cursor stops before it. A Run row
    locked by the pipeline for more than 200 ms is skipped without advancing.
    """
    session = ctx.session
    model = src.model()
    stored = None if from_start else read_cursor(session, src.name)
    cursor = {k: stored[k] for k in ("ts", "id")} if stored and stored.get("ts") else None
    pending = list((stored or {}).get("pending") or [])
    # Export events a committed row queued: durable in the same commit as the
    # cursor advance, written afterwards (key → Run → case), removed once written.
    exports = [list(e) for e in (stored or {}).get("exports") or []]
    turn = (stored or {}).get("turn", "revisit")
    tally = dict((stored or {}).get("tally") or {})  # e.g. spend_draws' unattributable draws
    ctx.cursor_tally = tally
    stats = ctx.report.source(src.name)
    stats.setdefault("cursor_from", cursor)
    left = [budget]

    def take() -> bool:
        if ctx.out_of_time() or (left[0] is not None and left[0] <= 0):
            return False
        if left[0] is not None:
            left[0] -= 1
        return True

    def process(row_id: str, row, *, trackable: bool = True) -> str:
        """One row in a savepoint: ``terminal``, ``pending`` (keep revisiting),
        ``untracked`` or ``locked`` (both rolled back, nothing imported)."""
        ctx.ensure_batch()
        try:
            ctx.row_cache = {}  # filled by the media prefetch, reused under the Run lock
            media = src.media(ctx, row)
            ctx.redeclare = lambda: src.media(ctx, row)  # re-derived from the row as re-read under the lock
            try:
                ctx.begin_row(media)  # file I/O happens here, before any Run lock
            except OSError as exc:  # unreadable file: import unhashed rather than lose the row
                ctx.report.note("missing_metadata", what="media probe failed; imported unhashed", source=src.name,
                                id=row_id, error=str(exc)[:200])
                ctx.begin_row([(path, False) for path, _ in media])
        except Exception as exc:
            stats["failed"] += 1
            ctx.report.note("failures", source=src.name, id=row_id, error=f"{type(exc).__name__}: {exc}"[:500])
            return "terminal"
        sp = session.begin_nested()
        ctx.defer_exports, ctx.deferred_exports, committed = True, [], False
        tally_before = dict(tally)
        try:
            if not ctx.dry_run:
                session.execute(text("SET LOCAL lock_timeout = '200ms'"))
            src.process(ctx, row)
            terminal = bool(src.ready(row, ctx.now))
            if (not terminal and not trackable) or len(exports) + len(ctx.deferred_exports) > EXPORT_CAP:
                sp.rollback()
                return "untracked"
            sp.commit()
            committed = True
            exports.extend([case_id, table, export_row_id, stamp.isoformat(), 0]
                           for case_id, table, export_row_id, stamp in ctx.deferred_exports)
        except (LeaseLost, store.BatchWithdrawn):
            raise  # stop the pass without advancing; BatchWithdrawn aborts this batch
        except MediaChanged:
            sp.rollback()
            stats["media_changed"] = stats.get("media_changed", 0) + 1
            return "locked"  # retried next pass, like a held Run
        except OperationalError as exc:
            sp.rollback()
            if "lock" not in str(exc).lower():
                raise
            stats["locked"] = stats.get("locked", 0) + 1
            return "locked"
        except Exception as exc:  # malformed/conflicting row: reported, never fatal
            sp.rollback()
            stats["failed"] += 1
            kind = "conflicts" if isinstance(exc, store.KnowledgeConflict) else "failures"
            ctx.report.note(kind, source=src.name, id=row_id, error=f"{type(exc).__name__}: {exc}"[:500])
            return "terminal"
        finally:
            ctx.defer_exports = False
            if not committed:
                ctx.deferred_exports = []  # events of a rolled-back row are never written
                tally.clear()
                tally.update(tally_before)  # nor are its tally entries
        stats["rows"] += 1
        return "terminal" if terminal else "pending"

    def save() -> None:
        if not ctx.dry_run:
            ctx.heartbeat(src.name)  # raises LeaseLost before anything is committed
            write_cursor(session, src.name, {**(cursor or {}), "pending": pending, "turn": turn, "exports": exports,
                                             **({"tally": tally} if tally else {})}, owner)
            session.commit()

    def flush_exports() -> None:
        """Write queued export events, each after a deadline check and heartbeat, in
        its own transaction; a locked one waits for a later pass, a failing one is
        dropped (reported) after ``EXPORT_TRIES``. Nothing to do in a dry run."""
        if ctx.dry_run:
            return
        i = 0
        while i < len(exports):
            if ctx.out_of_time():
                return  # still durable in the cursor for the next tick
            ctx.heartbeat(src.name)
            case_id, table, export_row_id, stamp, tries = exports[i]
            outcome = reconcile._row(ctx, lambda: write_export(ctx, case_id, table, export_row_id, _ts(stamp)))
            if outcome == "ok":
                exports.pop(i)
            elif outcome == "failed" and tries + 1 >= EXPORT_TRIES:
                ctx.report.note("failures", source=src.name, what="export event abandoned", id=export_row_id,
                                case_id=case_id)
                exports.pop(i)
            else:
                exports[i][4] = tries + (outcome == "failed")
                i += 1
            save()

    flush_exports()  # what an earlier pass queued but could not write
    for _ in range(_revisit_count(budget, len(pending), turn)):
        if not take():
            break
        row_id = pending.pop(0)
        row = session.get(model, row_id)
        outcome = "terminal" if row is None else process(row_id, row)
        if outcome == "terminal":
            stats["revisit_done"] = stats.get("revisit_done", 0) + 1
        elif src.ages_out and (_utc(row.created_at) or ctx.now) <= ctx.now - PENDING_MAX_AGE:
            ctx.report.note("skipped", what="non-terminal row aged out of the revisit list", source=src.name,
                            id=row_id)
        else:
            pending.append(row_id)  # rotate (a locked row too): the rest of the list goes first next pass
        save()
        flush_exports()
        if outcome == "locked":
            break
    turn = "new" if turn == "revisit" else "revisit"
    col = getattr(model, src.cursor_col)
    edge = horizon(ctx, col)
    q = select(model).where(col <= edge)
    if until is not None:
        q = q.where(col < until)  # strict: a draw at the bound's own timestamp waits
    if cursor:
        q = q.where(tuple_(col, model.id) > tuple_(_ts(cursor["ts"]), cursor["id"]))
    q = q.order_by(col, model.id)
    if left[0] is not None:
        q = q.limit(left[0] + 1)
    rows = session.execute(q).scalars().all()
    keys = [(getattr(r, src.cursor_col), r.id) for r in rows]
    exhausted = True
    for row, (created_at, row_id) in zip(rows, keys):
        if not take():
            exhausted = False
            break
        outcome = process(row_id, row, trackable=len(pending) < PENDING_CAP)
        if outcome in ("untracked", "locked"):
            if outcome == "untracked":
                ctx.report.note("skipped", what="revisit or export list full; source stopped before this row",
                                source=src.name, id=row_id)
            exhausted = False
            break
        if outcome == "pending" and row_id not in pending:
            pending.append(row_id)
        cursor = {"ts": created_at.isoformat(), "id": row_id}
        save()  # the row, its cursor advance and its queued exports in one commit
        flush_exports()
    save()  # persist the alternation turn even when nothing moved
    stats["caught_up"] = exhausted
    return _done(stats, cursor, pending, budget, left)


def _done(stats: dict, cursor, pending: list, budget, left) -> tuple[int, bool]:
    stats["cursor_to"] = cursor
    stats["pending"] = len(pending)
    return (budget - left[0]) if budget is not None else stats["rows"] + stats["failed"], stats["caught_up"]


def _draw_until(ctx: ImportContext, from_start: bool) -> Optional[datetime]:
    """Draws are matched to attempts only up to where every task-bearing source has
    scanned: a draw is created after its segment/run/range/image row, so below that
    bound the row that names its task has already been imported (or is pending)."""
    bounds = []
    for src in SOURCES:
        if not src.task_bearing:
            continue
        stats = ctx.report.sources.get(src.name) or {}
        if stats.get("caught_up"):
            bounds.append(horizon(ctx, getattr(src.model(), src.cursor_col)))
            continue
        cur = stats.get("cursor_to") or (None if from_start else read_cursor(ctx.session, src.name))
        if cur and cur.get("ts"):
            # A task row's own update (its task id committed) lands within one job of
            # the draw, so draws are held back a commit lag behind the slowest source.
            bounds.append(_ts(cur["ts"]) - COMMIT_LAG)
            continue
        model = src.model()
        col = getattr(model, src.cursor_col)
        if ctx.session.execute(select(model.id).where(col <= horizon(ctx, col))
                               .limit(1)).first() is not None:
            return None  # rows this source has not scanned yet may name a draw's task
    draws = next(s for s in SOURCES if s.name == "spend_draws").model()
    return min(bounds) if bounds else horizon(ctx, draws.created_at)


def _schedule(ctx: ImportContext, sources: list, limit: Optional[int], owner: str, from_start: bool) -> None:
    """Fair per-tick budget: each source gets an equal share first, starting at a
    rotating source, then any leftover goes round the sources that still have work.
    Draw coverage runs every pass, bounded by the task-bearing sources' progress."""
    def run(src, budget) -> tuple[int, bool]:
        if src.name == "spend_draws":
            until = _draw_until(ctx, from_start)
            if until is None:
                ctx.report.note("skipped", what="draw coverage waits until every task source has a position")
                return 0, False
            return run_source(ctx, src, budget, owner, from_start, until=until)
        return run_source(ctx, src, budget, owner, from_start)

    if not sources:
        return
    if limit is None:
        for src in sources:
            run(src, None)
        return
    start = int(ctx.now.timestamp() // 60) % len(sources)
    order = sources[start:] + sources[:start]
    share, left, busy = max(1, limit // len(order)), limit, []
    for src in order:
        if left <= 0 or ctx.out_of_time():
            break
        n, exhausted = run(src, min(share, left))
        left -= n
        if not exhausted:
            busy.append(src)
    for src in busy:
        if left <= 0 or ctx.out_of_time():
            break
        n, _ = run(src, left)
        left -= n


def _step(ctx: ImportContext, name: str, fn: Callable[[], object]):
    if ctx.out_of_time():
        ctx.report.note("skipped", what="tick deadline reached", step=name)
        return None
    ctx.ensure_batch()
    sp = ctx.session.begin_nested()
    try:
        out = fn()
        sp.commit()
    except (LeaseLost, store.BatchWithdrawn):
        raise
    except Exception as exc:
        sp.rollback()
        ctx.report.note("failures", source=f"reconcile:{name}", error=f"{type(exc).__name__}: {exc}"[:500])
        return None
    if not ctx.dry_run:
        ctx.heartbeat("reconcile")
        ctx.session.commit()
    return out


def run_import(session_factory, *, apply: bool, sources: Optional[Iterable[str]] = None,
               limit: Optional[int] = None, hasher: Optional[Callable[[str], str]] = None,
               exists: Callable[[str], bool] = os.path.isfile, stat: Optional[Callable[[str], object]] = None,
               from_start: bool = False,
               revision: Optional[str] = None, seed_dir: Optional[str] | bool = None, owner: Optional[str] = None,
               now: Optional[datetime] = None, manifest_path: str = MANIFEST_PATH,
               manifest_sha256: str = MANIFEST_SHA256, keepalive: Optional[Callable[[], bool]] = None,
               deadline: Optional[datetime] = None, lease_ttl_sec: int = LEASE_TTL_SEC,
               reconcile_keys: Optional[Iterable[str]] = None,
               drain_project_ids: Optional[Iterable[str]] = None) -> dict:
    """One import pass; returns the JSON-ready report.

    ``reconcile_keys`` limits the reconcile step to these sub-steps
    (``settle``, ``intents``, ``exports``, ``finals``, ``drain``); None runs all.
    """
    selected = [s for s in STEPS if sources is None or s in set(sources)]
    unknown = sorted(set(sources or ()) - set(STEPS))
    if unknown:
        raise ValueError(f"unknown import sources: {unknown}")
    owner = owner or f"import:{uuid.uuid4()}"
    leased: list[str] = []
    if apply:
        for name in selected:
            if acquire_lease(session_factory, name, owner, lease_ttl_sec):
                leased.append(name)
        if not leased:
            return {"dry_run": False, "skipped": "every selected source is leased by another importer"}
    else:
        leased = list(selected)
    session = session_factory()
    ctx = ImportContext(session=session, dry_run=not apply, code_revision=revision or code_revision(),
                        source_label="history:" + ",".join(leased), hasher=hasher, exists=exists, stat=stat,
                        now=now or datetime.now(timezone.utc), owner=owner, lease_ttl_sec=lease_ttl_sec,
                        keepalive=keepalive, deadline=deadline)
    report = ctx.report
    report.extra["leased_elsewhere"] = sorted(set(selected) - set(leased))
    try:
        ctx.load_manifest(manifest_path, manifest_sha256)
        ctx.ensure_batch()
        if apply:
            session.commit()
        _schedule(ctx, [src for src in SOURCES if src.name in leased], limit, owner, from_start)
        if "seed" in leased and seed_dir is not False:
            ctx.ensure_batch()
            import_seed(ctx, seed_dir or SEED_DIR)
        if "reconcile" in leased:
            step_budget = limit if limit is not None else 10**9
            stored = read_cursor(session, "reconcile") or {}
            # --from-start rescans positions but keeps "already recorded missing"
            # bookkeeping, so a rescan never records the same miss twice.
            state = dict(stored) if not from_start else {
                "finals": {"missing": ((stored.get("finals") or {}).get("missing") or {})}}

            def keep(key: str, fn):
                moved = _step(ctx, key, lambda: {"cursor": fn(state.get(key))})
                if moved is not None:
                    state[key] = moved["cursor"]
                    if apply:
                        ctx.heartbeat("reconcile")
                        write_cursor(session, "reconcile", state, owner)
                        session.commit()

            wanted = set(reconcile_keys) if reconcile_keys is not None else None

            def on(key: str) -> bool:
                return wanted is None or key in wanted

            if on("settle"):
                keep("settle", lambda c: reconcile.settle_draws(ctx, step_budget, c))
            if on("intents"):
                keep("intents", lambda c: reconcile.mark_stale_intents(ctx, step_budget, c))
            def saver(key: str):
                def save(cursor) -> None:  # per-row commit: the row's Run lock is released here
                    state[key] = cursor
                    if apply:
                        ctx.heartbeat("reconcile")
                        write_cursor(session, "reconcile", state, owner)
                        session.commit()
                return save

            for key, fn in (("exports", reconcile.record_exports), ("finals", reconcile.reconcile_finals)):
                if not on(key):
                    continue
                try:
                    fn(ctx, step_budget, state.get(key), saver(key))
                except (LeaseLost, store.BatchWithdrawn):
                    raise
                except Exception as exc:
                    if apply:
                        session.rollback()
                    report.note("failures", source=f"reconcile:{key}", error=f"{type(exc).__name__}: {exc}"[:500])
            if on("drain") and not ctx.out_of_time():  # commits per run itself, so not inside a step savepoint
                reconcile.drain_captures(ctx, step_budget, state.get("drain"), saver("drain"),
                                         project_ids=(frozenset(drain_project_ids)
                                                      if drain_project_ids is not None else None))
        out = _finish(ctx, "finished")
        if apply:
            session.commit()
        else:
            session.rollback()
        return out
    except (LeaseLost, store.BatchWithdrawn) as exc:
        session.rollback()  # the row in flight and its cursor advance are discarded
        report.note("failures", source="import", error=f"{type(exc).__name__}: {exc}")
        out = _finish(ctx, "aborted")
        session.commit()
        return out
    except Exception as exc:
        session.rollback()
        report.note("failures", source="import", error=f"{type(exc).__name__}: {exc}"[:500])
        if apply and ctx.batch_id is not None and session.execute(
                text("SELECT 1 FROM kb_import_batches WHERE id = :i"), {"i": ctx.batch_id}).first():
            out = _finish(ctx, "failed")
            session.commit()
            return out
        return {**report.as_dict(), "dry_run": not apply, "outcome": "failed"}
    finally:
        session.close()
        if apply:
            for name in leased:
                release_lease(session_factory, name, owner)


def _finish(ctx: ImportContext, outcome: str) -> dict:
    report = {**ctx.report.as_dict(), "batch_id": ctx.batch_id, "dry_run": ctx.dry_run, "outcome": outcome,
              "provenance": {"origin": "import", "import_batch_id": ctx.batch_id,
                             "code_revision": ctx.code_revision, "source": ctx.source_label,
                             "hashing": ctx.hasher is not None, "started_at": ctx.now.isoformat()}}
    counts = {"created": report["created"], "reused": report["reused"],
              **{k: report[k]["total"] for k in ("duplicates", "missing_media", "missing_metadata", "conflicts",
                                                 "failures", "excluded_cases", "refused", "skipped")}}
    store.finish_import_batch(ctx.session, ctx.batch_id, kind=outcome, counts=counts, report=report)
    return report


# ---------------------------------------------------------------- RQ tick


def run_tick(session_factory, redis, *, limit: int = TICK_LIMIT,
             hasher: Optional[Callable[[str], str]] = store.sha256_file,
             lease_ttl_sec: int = LEASE_TTL_SEC, drain_only: bool = False) -> dict:
    """Bounded incremental apply under a Redis lease (plus the cursor leases).

    Both leases are refreshed and re-verified before every commit; the pass
    stops taking rows at a deadline well inside the lease TTL.
    """
    token = f"tick:{uuid.uuid4()}"
    if not redis.set(REDIS_LEASE_KEY, token, nx=True, ex=lease_ttl_sec):
        return {"skipped": "another knowledge import tick holds the lease"}
    def keepalive() -> bool:
        return redis.eval(_REFRESH_LUA, 1, REDIS_LEASE_KEY, token, lease_ttl_sec) == 1

    try:
        # drain_only: just the C2 capture queue, whose items only in-scope capture
        # creates; every historical source and the other reconcile steps read all
        # projects (AICP-298 opt-in).
        scope_ids = None
        if drain_only:
            from .. import capture

            scope = capture.capture_scope()
            scope_ids = frozenset(scope.project_ids) if capture.enabled() and scope.valid else frozenset()
        return run_import(session_factory, apply=True, limit=limit, hasher=hasher, owner=token,
                          sources=["reconcile"] if drain_only else [s for s in STEPS if s != "seed"],
                          reconcile_keys=("drain",) if drain_only else None, drain_project_ids=scope_ids,
                          keepalive=keepalive,
                          lease_ttl_sec=lease_ttl_sec,
                          deadline=datetime.now(timezone.utc) + timedelta(seconds=min(TICK_MAX_SEC, lease_ttl_sec // 2)))
    finally:
        try:
            redis.eval(_RELEASE_LUA, 1, REDIS_LEASE_KEY, token)
        except Exception:  # the lease expires on its own
            log.warning("knowledge import tick: lease release failed", exc_info=True)


__all__ = ["run_import", "run_tick", "acquire_lease", "release_lease", "read_cursor", "STEPS", "TICK_LIMIT"]
