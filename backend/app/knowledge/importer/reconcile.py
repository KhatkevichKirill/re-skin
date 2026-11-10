"""Coverage reconciliation (design §6.1, last paragraphs; §6.3).

* settled draws whose attempt lacks a ``cost_settled`` observation get one
  (draws have no ``updated_at``; ``mcredits`` is replaced on settlement);
* a kie intent with neither a submission nor a terminal status after
  2 × the job timeout gets ``intent_without_submission``;
* a Run whose live final hashes differently from its latest Run case version,
  with no pending capture item, is queued as a ``reconcile:`` publication;
* Final Creatives stamps become ``export_observed`` events — never acceptance.

Each step is bounded and idempotent; unmatched draws are handled by the
``spend_draws`` source (ambiguous attempts).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Callable, Optional

from sqlalchemy import exists, func, select, text, true, tuple_
from sqlalchemy.exc import OperationalError

from .. import store
from ..models import KbArtifact, KbArtifactCheck, KbAttempt, KbAttemptSubmission, KbCaptureDone, KbCaptureQueue
from .context import ImportContext, LeaseLost, status_value
from .sources import _job_timeout_sec, _utc, horizon, record_export, run_case


def _cursor_after(column, id_column, cursor: Optional[dict]):
    return tuple_(column, id_column) > tuple_(_ts(cursor["ts"]), cursor["id"]) if cursor else true()


def settle_draws(ctx: ImportContext, budget: int, cursor: Optional[dict]) -> Optional[dict]:
    """Settled draws in ``(settled_at, id)`` order below the commit horizon, ``budget`` per pass.

    A settlement re-stamps ``settled_at`` (and may revise ``mcredits``), so a
    re-settled draw is seen again and a changed amount is appended as a newer
    ``cost_settled`` (latest by ``seq`` wins). A draw settled before its attempt
    existed gets its settled cost when the attempt is imported (``draw_costs``).
    """
    from ...models import KieSpendDraw

    rows = ctx.session.execute(select(KieSpendDraw).where(
        KieSpendDraw.settled_at.isnot(None), KieSpendDraw.settled_at <= horizon(ctx, KieSpendDraw.settled_at),
        _cursor_after(KieSpendDraw.settled_at, KieSpendDraw.id, cursor))
        .order_by(KieSpendDraw.settled_at, KieSpendDraw.id).limit(budget)).scalars().all()
    done = 0
    for draw in rows:
        if ctx.out_of_time():
            break  # the cursor stays at the last settled draw handled
        attempt = ctx.attempt_for_call_key(draw.call_key)
        if attempt is not None and ctx.draw_costs(attempt, draw):
            done += 1
        cursor = {"ts": draw.settled_at.isoformat(), "id": draw.id}
    ctx.report.bump(ctx.report.extra.setdefault("reconcile", {}), "settlements_recorded", done)
    return cursor


def mark_stale_intents(ctx: ImportContext, budget: int, cursor: Optional[int]) -> Optional[int]:
    """kie attempts without a submission, in ``seq`` order, stopping at the first
    one younger than the timeout; the ``seq`` cursor never revisits a decided one."""
    cutoff = ctx.now - timedelta(seconds=2 * _job_timeout_sec())
    q = select(KbAttempt).where(KbAttempt.provider == "kie",
                                ~exists().where(KbAttemptSubmission.attempt_id == KbAttempt.id))
    if cursor is not None:
        q = q.where(KbAttempt.seq > cursor)
    invalid = ctx.invalid()
    done = 0
    for attempt in ctx.session.execute(q.order_by(KbAttempt.seq).limit(budget)).scalars().all():
        if _utc(attempt.created_at) >= cutoff or ctx.out_of_time():
            break
        cursor = attempt.seq
        if invalid.attempt_invalid(attempt) or any(
                o.kind == "ambiguous" or (o.kind == "status" and o.status in store.TERMINAL_STATUSES + ("submitted",))
                for o in ctx.observations(attempt.id)):
            continue
        ctx.observe(attempt, kind="status", status="intent_without_submission",
                    payload={"timeout_sec": 2 * _job_timeout_sec(), "possibly_charged": True})
        done += 1
    ctx.report.bump(ctx.report.extra.setdefault("reconcile", {}), "intents_without_submission", done)
    return cursor


FINAL_RETRIES = 3
MISSING_IDENTITIES_KEPT = 8  # per run: every (path, size, mtime) recorded missing is recorded once
DRAIN_MAX_ITEMS = 20


class _Changed(Exception):
    """The final moved while it was being hashed: retry next pass, not a failure."""


def _row(ctx: ImportContext, fn) -> str:
    """One row in a savepoint; in apply mode Run locks wait at most 200 ms.

    Returns ``ok``, ``locked`` / ``changed`` (retried next pass) or ``failed``.
    """
    ctx.begin_row()
    sp = ctx.session.begin_nested()
    try:
        if not ctx.dry_run:
            ctx.session.execute(text("SET LOCAL lock_timeout = '200ms'"))
        fn()
        sp.commit()
        return "ok"
    except (LeaseLost, store.BatchWithdrawn):
        raise  # stop the pass without advancing any cursor; the batch is aborted
    except _Changed:
        sp.rollback()
        return "changed"
    except _Missing:
        sp.rollback()
        return "missing"
    except OperationalError as exc:
        sp.rollback()
        if "lock" in str(exc).lower():
            return "locked"
        raise
    except Exception as exc:
        sp.rollback()
        ctx.report.note("failures", source="reconcile", error=f"{type(exc).__name__}: {exc}"[:500])
        return "failed"


def reconcile_finals(ctx: ImportContext, budget: int, cursor: Optional[dict],
                     save: Callable[[Optional[dict]], None]) -> Optional[dict]:
    """Queue a ``reconcile:<run>:<hash>:<version_no>`` publication for a changed final.

    A Run case with no version yet (imported before it had a final) is queued
    as ``version_no`` 0. Needs hashing; a latest version recorded without a
    hash at the same path cannot be compared and is reported, not queued.
    One run per transaction (``save`` heartbeats, advances and commits), the
    deadline checked before every hash. A locked Run, a final that moved while
    hashing, or a failure stops the pass at that run without advancing; after
    ``FINAL_RETRIES`` failures the run is passed with a durable missing record.
    Returns the rotating cursor (with its per-run failure counts).
    """
    from ...models import Run

    stats = ctx.report.extra.setdefault("reconcile", {})
    if ctx.hasher is None:
        stats["finals"] = "skipped: hashing disabled"
        return cursor
    position = {k: cursor[k] for k in ("ts", "id")} if cursor and cursor.get("ts") else None
    failures = {k: v for k, v in ((cursor or {}).get("failures") or {}).items() if isinstance(v, int)}
    missing = dict((cursor or {}).get("missing") or {})  # run -> [[path, identity], ...] recorded missing
    runs = ctx.session.execute(select(Run).where(Run.result_local_path.isnot(None), Run.status == "done",
                                                 _cursor_after(Run.created_at, Run.id, position))
                               .order_by(Run.created_at, Run.id).limit(budget)).scalars().all()
    keys = [(r.created_at, r.id) for r in runs]

    def state(pos):
        return {**(pos or {}), "failures": failures, "missing": missing} if pos or failures or missing else None

    for run, (created_at, run_id) in zip(runs, keys):
        if ctx.out_of_time():
            ctx.report.note("skipped", what="tick deadline reached", step="finals")
            return state(position)
        if [run.result_local_path, _jsonable(ctx.identity(run.result_local_path))] in missing.get(run_id, []):
            # Already recorded missing for this exact file: nothing new until it changes.
            position = {"ts": created_at.isoformat(), "id": run_id}
            save(state(position))
            continue
        outcome = _row(ctx, lambda: _reconcile_final(ctx, run, stats))
        if outcome in ("locked", "changed"):
            ctx.report.bump(stats, f"finals_runs_{outcome}_retry")
            return state(position)
        if outcome == "failed":
            failures[run_id] = failures.get(run_id, 0) + 1
            if failures[run_id] < FINAL_RETRIES:
                save(state(position))
                return state(position)
        if outcome in ("failed", "missing"):
            if _row(ctx, lambda: _record_unreconciled(ctx, run)) != "ok":
                save(state(position))  # the durable missing record is the condition for moving on
                return state(position)
            ctx.report.note("missing_media", run_id=run_id, path=run.result_local_path,
                            what="final file missing" if outcome == "missing"
                            else f"final not reconciled after {FINAL_RETRIES} attempts")
            given_up = missing.setdefault(run_id, [])
            given_up.append([run.result_local_path, _jsonable(ctx.identity(run.result_local_path))])
            del given_up[:-MISSING_IDENTITIES_KEPT]
        failures.pop(run_id, None)
        if outcome == "ok":
            # Readable again: a later disappearance is a new fact to record.
            missing.pop(run_id, None)
        position = {"ts": created_at.isoformat(), "id": run_id}
        save(state(position))
    if len(runs) < budget:
        position = None  # wrap around next time
        save(state(position))
    return state(position)


class _Missing(Exception):
    """The done Run's final file is not on disk: record its absence once."""


def _reconcile_final(ctx: ImportContext, run, stats: dict) -> None:
    path = run.result_local_path
    if not ctx.exists(path):
        raise _Missing()
    identity = ctx.identity(path)
    sha = ctx.hasher(path)  # before the Run lock: hashing must never hold a production row
    # Under the Run lock, re-read the Run and re-verify the file, so a promotion or
    # restitch that landed meanwhile can never be overtaken by this stale hash.
    ctx.lock_run(run.id)
    ctx.session.refresh(run)
    if status_value(run.status) != "done" or run.result_local_path != path or ctx.identity(path) != identity:
        raise _Changed()
    # S9a: a done run whose case or first version is missing (its capture lost)
    # is queued too; the case is ensured under the Run lock (§6.2).
    case = store.current_case(ctx.session, store.run_case_key(run.id), ctx.invalid()) or run_case(ctx, run)
    latest = store.latest_version(ctx.session, case.id, ctx.invalid())
    if latest is not None:
        art = ctx.session.get(KbArtifact, latest.artifact_id)
        if art.sha256 == sha:
            return
        if art.sha256 is None and art.locator == path:
            ctx.report.note("missing_metadata", what="latest run version unhashed at the same path; "
                            "not comparable", run_id=run.id)
            return
    pending = ctx.session.execute(select(KbCaptureQueue.id).where(
        KbCaptureQueue.run_id == run.id,
        ~exists().where(KbCaptureDone.queue_id == KbCaptureQueue.id))).first()
    if pending:
        return
    final = ctx.artifact(kind="delivered_final" if run.result_gdrive_file_id else "stitched_final",
                         locator_kind="local_path", locator=path, project_id=run.project_id,
                         sha256=sha, present=True, verified=True, duration_sec=run.result_duration_sec)
    live_identity = store.file_identity(path)  # under the Run lock; C2 re-checks it before "current"
    if live_identity is None:
        raise _Changed()
    store.enqueue_final_capture(ctx.session, run_id=run.id, artifact_id=final.id, reason="import",
                                publication_id=f"reconcile:{run.id}:{sha}:{latest.version_no if latest else 0}",
                                file_identity=live_identity)
    ctx.report.bump(stats, "finals_queued")


def _jsonable(identity):
    return list(identity) if isinstance(identity, (tuple, list)) else identity


def _record_unreconciled(ctx: ImportContext, run) -> None:
    """§6.1: a final that could not be captured is recorded unhashed with its availability.

    A deleted file is idempotent across cursor resets (--from-start): the existing
    unhashed record of the path is reused, and a check is added only when presence
    changed since the last one.
    """
    path, present = run.result_local_path, bool(ctx.exists(run.result_local_path))
    if present:
        # Unreadable but present: each file identity is its own record (the
        # cursor's missing list dedupes identities already recorded).
        ctx.artifact(kind="delivered_final" if run.result_gdrive_file_id else "stitched_final",
                     locator_kind="local_path", locator=path, project_id=run.project_id,
                     sha256=None, present=True)
        return
    invalid = ctx.invalid()
    existing = next((a for a in ctx.session.execute(select(KbArtifact).where(
        KbArtifact.locator_kind == "local_path", KbArtifact.locator == path, KbArtifact.sha256.is_(None),
        KbArtifact.project_id == run.project_id).order_by(KbArtifact.seq.desc())).scalars()
        if not invalid.has("artifact", a.id)), None)
    if existing is None:
        ctx.artifact(kind="delivered_final" if run.result_gdrive_file_id else "stitched_final",
                     locator_kind="local_path", locator=path, project_id=run.project_id,
                     sha256=None, present=present)
        return
    # Presence last observed for this path through any record of it (a hashed
    # record made while it was readable counts), so gone→back→gone is two misses.
    last = ctx.session.execute(select(KbArtifactCheck).join(KbArtifact, KbArtifact.id == KbArtifactCheck.artifact_id)
                               .where(KbArtifact.locator_kind == "local_path", KbArtifact.locator == path,
                                      KbArtifact.project_id == run.project_id)
                               .order_by(KbArtifactCheck.seq.desc()).limit(1)).scalar_one_or_none()
    if last is None or last.present != present:
        store.record_artifact_check(ctx.session, existing.id, present=present, sha256_matches=None)


def drain_captures(ctx: ImportContext, budget: int, cursor: Optional[int],
                   save: Callable[[Optional[int]], None],
                   project_ids: Optional[frozenset] = None) -> Optional[int]:
    """C2 (§6.1): allocate pending capture items per Run under its lock (apply only).

    Runs are taken in order of their oldest pending item and rotated with a
    cursor on that ``c1_seq``, so a backlog larger than one pass cannot hold the
    same Runs every tick; at most ``DRAIN_MAX_ITEMS`` per Run, deadline and
    heartbeat checked per Run, a Run locked for 200 ms skipped until next time.
    ``project_ids`` (the drain-only tick's current capture scope) leaves items of
    any other project pending: nothing of theirs is read or written.
    """
    # Bounded scan in c1_seq order (unique index) from the cursor, never a global
    # GROUP BY over the whole backlog; the first pending item per Run picks the Run.
    q = select(KbCaptureQueue.run_id, KbCaptureQueue.c1_seq).where(
        ~exists().where(KbCaptureDone.queue_id == KbCaptureQueue.id))
    if project_ids is not None:
        from ...models import Run

        q = q.where(exists().where(Run.id == KbCaptureQueue.run_id, Run.project_id.in_(sorted(project_ids))))
    if cursor is not None:
        q = q.where(KbCaptureQueue.c1_seq > cursor)
    if not ctx.dry_run:
        ctx.session.execute(text("SET LOCAL statement_timeout = '5s'"))
    seen, rows, scanned, last_scanned = set(), [], 0, None
    scan_limit = budget * DRAIN_MAX_ITEMS
    for run_id, seq in ctx.session.execute(q.order_by(KbCaptureQueue.c1_seq).limit(scan_limit)):
        scanned, last_scanned = scanned + 1, seq
        if run_id not in seen:
            seen.add(run_id)
            rows.append((run_id, seq))
            if len(rows) >= budget:
                break
    if not ctx.dry_run:
        ctx.session.execute(text("SET LOCAL statement_timeout = 0"))
    stats = ctx.report.extra.setdefault("reconcile", {})
    if ctx.dry_run:
        stats["capture_runs_pending"] = len(rows)
        return cursor
    for run_id, oldest in rows:
        if ctx.out_of_time():
            ctx.report.note("skipped", what="tick deadline reached", step="drain")
            return cursor
        sp = ctx.session.begin_nested()
        try:
            # S9d: a run held by its submit phase is skipped until the next tick, not waited on.
            ctx.session.execute(text("SET LOCAL lock_timeout = '200ms'"))
            done = store.drain_run_captures(ctx.session, run_id, max_items=DRAIN_MAX_ITEMS)
            sp.commit()
        except OperationalError as exc:
            sp.rollback()
            if "lock" not in str(exc).lower():
                raise
            ctx.report.bump(stats, "capture_runs_locked_skipped")
            done = []
        for row in done:
            ctx.report.bump(stats, f"capture_{row.outcome}")
        cursor = oldest
        save(cursor)  # heartbeat + commit per Run: its lock is released here
    if scanned >= scan_limit and last_scanned is not None:
        # The scan window was full (e.g. one Run's backlog filled it): continue past
        # it next pass so later Runs are reached; this Run's rest drains on the wrap.
        cursor = max(cursor or 0, last_scanned)
        save(cursor)
    elif len(rows) < budget:
        cursor = None  # wrap around next time
        save(cursor)
    return cursor


def _export_tables():
    from ...models import ManualCreative, OverlayJob, Run, RunCaption, RunTailEdit

    return (("runs", Run, store.run_case_key), ("run_tail_edits", RunTailEdit, None),
            ("run_captions", RunCaption, None), ("overlay_jobs", OverlayJob, None),
            ("manual_creatives", ManualCreative, store.manual_case_key))


def record_exports(ctx: ImportContext, budget: int, cursors: Optional[dict],
                   save: Callable[[dict], None]) -> dict:
    """Stamps set after a row was imported, per table on ``(stamp, id)`` below the
    commit horizon. A stamped row whose case does not exist yet gets its event
    when its adapter imports it (``record_export``).

    The event writer takes the Run lock, so each stamp is its own transaction
    (``save`` commits) with a 200 ms lock timeout; a locked Run stops that
    table's scan at the stamp, which is retried next pass.
    """
    cursors = dict(cursors or {})
    stats = ctx.report.extra.setdefault("reconcile", {})
    for table, model, keyfn in _export_tables():
        col = model.final_creatives_exported_at
        rows = ctx.session.execute(select(model).where(col.isnot(None), col <= horizon(ctx, col),
                                                       _cursor_after(col, model.id, cursors.get(table)))
                                   .order_by(col, model.id).limit(budget)).scalars().all()
        keys = [(r.final_creatives_exported_at, r.id) for r in rows]
        for row, (stamp, row_id) in zip(rows, keys):
            if ctx.out_of_time():
                return cursors
            case_key = keyfn(row_id) if keyfn else store.derived_case_key(table, row_id)

            def one(row=row, case_key=case_key) -> None:
                case = store.current_case(ctx.session, case_key, ctx.invalid())
                if case is not None:
                    record_export(ctx, case, table, row)

            if _row(ctx, one) == "locked":
                ctx.report.bump(stats, "exports_locked_skipped")
                break
            cursors[table] = {"ts": stamp.isoformat(), "id": row_id}
            save(cursors)
    return cursors


def _ts(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)


__all__ = ["settle_draws", "mark_stale_intents", "reconcile_finals", "drain_captures", "record_exports"]
