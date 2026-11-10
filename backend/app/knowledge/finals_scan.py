"""The finals scanner (AICP-299 §3.3): publication links → capture queue.

Runs on the knowledge worker. Per Run with pending links:

- Phase 1 (read-only, bounded): list the Run's link directory and select at
  most ``K`` candidates — the publishable link whose inode is the live final
  first, then other publishable links oldest first, then history-only links —
  and sha256 only those (their inodes are immutable). Nothing is written.
- Phase 2 (one short transaction, statement/lock timeouts 1 s, no Run lock):
  artifacts, and ``enqueue_final_capture`` with the link's inode identity for
  publishable links (C2 later decides current vs historical under the Run
  lock); an artifact plus a check for history-only links.

A link is *pending* exactly when ``store.publication_covered`` is false (for
publishable links) or no valid artifact names its locator (history only).
A point (ii) ``superseded`` link is publishable only when a ``kb_link_failures``
row proves its inode was published (identity match, or the NULL-identity rule).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

from . import links
from .publication import REASON_FOR_KIND

log = logging.getLogger(__name__)

K = 6
SETTLE_SEC = 30


def _failures(session, run_id: str) -> list:
    from sqlalchemy import select

    from .models import KbLinkFailure

    return session.execute(select(KbLinkFailure).where(KbLinkFailure.run_id == run_id)
                           .order_by(KbLinkFailure.failed_at_ns)).scalars().all()


def _pair(link: links.LinkName, failures: list, all_links: list):
    """The ``kb_link_failures`` row that proves a ``superseded`` link, or None."""
    for f in failures:
        if f.st_ino is not None and (f.st_dev, f.st_ino, f.st_mtime_ns) == \
                (link.st_dev, link.st_ino, link.st_mtime_ns):
            return f
    for f in failures:  # NULL identity: the first (ii) after the failure, before the next (i)
        if f.st_ino is not None or f.kind != "stitch" or os.path.basename(f.path) != "final.mp4":
            continue
        next_pub = min((x.time_ns for x in all_links if x.publishable and x.time_ns > f.failed_at_ns),
                       default=None)
        first = next((x for x in all_links if x.kind == links.SUPERSEDED and x.time_ns > f.failed_at_ns
                      and (next_pub is None or x.time_ns < next_pub)), None)
        if first is not None and first.path == link.path:
            return f
    return None


def _artifact_exists(session, locator: str) -> bool:
    from sqlalchemy import select

    from . import store
    from .models import KbArtifact

    return session.execute(select(KbArtifact.id).where(
        KbArtifact.locator_kind == "preserved_copy", KbArtifact.locator == locator,
        ~store.withdrawn_clause("artifact", KbArtifact.id)).limit(1)).first() is not None


def _live_identity(session, run_id: str) -> Optional[tuple]:
    from sqlalchemy import select

    from ..models import Run

    path = session.execute(select(Run.result_local_path).where(Run.id == run_id)).scalar()
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def scan_run(session, project_id: str, run_id: str, *, hasher: Optional[Callable[[str], str]] = None,
             now_ns: Optional[int] = None, k: int = K) -> dict:
    """One bounded pass over one Run's links. Commits its own Phase-2 transaction."""
    from sqlalchemy import text

    from . import store

    hasher = hasher or store.sha256_file
    now_ns = now_ns if now_ns is not None else time.time_ns()
    stats = {"enqueued": 0, "history": 0, "pending": 0}
    all_links = links.iter_links(project_id, run_id)
    ready = [x for x in all_links if now_ns - x.time_ns >= SETTLE_SEC * 1_000_000_000]
    if not ready:
        return stats
    failures = _failures(session, run_id)
    publishable, history = [], []
    for x in ready:
        if x.publishable:
            pub, kind = x.publication_id, x.kind
        else:
            f = _pair(x, failures, all_links)
            if f is not None:
                pub, kind = f.publication_id, f.kind
            else:
                if not _artifact_exists(session, x.path):
                    history.append(x)
                continue
        if not store.publication_covered(session, run_id, pub):
            publishable.append((x, pub, kind))
    stats["pending"] = len(publishable) + len(history)
    if not publishable and not history:
        session.rollback()
        return stats
    # Phase 1: select ≤ k (live first), hash only those.
    live = _live_identity(session, run_id)
    session.rollback()  # end the read transaction before hashing
    publishable.sort(key=lambda t: (0 if live and (t[0].st_dev, t[0].st_ino) == live else 1, t[0].time_ns))
    chosen = publishable[:k]
    chosen_hist = history[:max(0, k - len(chosen))]
    hashed = []
    for x, pub, kind in chosen:
        try:
            hashed.append((x, pub, kind, hasher(x.path), os.path.getsize(x.path)))
        except OSError:
            continue
    hashed_hist = []
    for x in chosen_hist:
        try:
            hashed_hist.append((x, hasher(x.path), os.path.getsize(x.path)))
        except OSError:
            continue
    # Phase 2: one short transaction, no Run lock.
    try:
        session.execute(text("SET LOCAL statement_timeout = '1s'"))
        session.execute(text("SET LOCAL lock_timeout = '1s'"))
        for x, pub, kind, sha, size in hashed:
            art = store.get_or_create_artifact(session, kind="delivered_final", locator_kind="preserved_copy",
                                               locator=x.path, sha256=sha, project_id=project_id, bytes=size)
            store.enqueue_final_capture(session, run_id=run_id, artifact_id=art.id,
                                        reason=REASON_FOR_KIND.get(kind, "restitch"), publication_id=pub,
                                        file_identity=x.identity)
            stats["enqueued"] += 1
        for x, sha, size in hashed_hist:
            art = store.get_or_create_artifact(session, kind="stitched_final", locator_kind="preserved_copy",
                                               locator=x.path, sha256=sha, project_id=project_id, bytes=size)
            store.record_artifact_check(session, art.id, present=True, sha256_matches=True)
            stats["history"] += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    return stats


def _is_rq_timeout(exc) -> bool:
    try:
        from rq.timeouts import BaseTimeoutException
    except Exception:  # pragma: no cover
        return False
    return isinstance(exc, BaseTimeoutException)


def run(session_factory, *, deadline: float, k: int = K) -> dict:
    """Every Run directory under ``knowledge/finals``, until the monotonic deadline."""
    total = {"runs": 0, "enqueued": 0, "history": 0, "errors": 0}
    from . import capture

    for project_id, run_id in links.iter_run_dirs():
        if time.monotonic() > deadline:
            break
        if not capture.in_scope(project_id):
            total["out_of_scope"] = total.get("out_of_scope", 0) + 1
            continue  # links stay on disk; nothing is read or written for this project
        session = session_factory()
        try:
            stats = scan_run(session, project_id, run_id, k=k)
            total["runs"] += 1
            total["enqueued"] += stats["enqueued"]
            total["history"] += stats["history"]
        except Exception as exc:  # one Run's problem never stops the scan
            if _is_rq_timeout(exc):
                raise  # RQ's job timeout is never swallowed
            total["errors"] += 1
            log.warning("finals_scan: run %s failed: %s: %s", run_id, type(exc).__name__, exc)
        finally:
            session.close()
    return total


__all__ = ["K", "SETTLE_SEC", "run", "scan_run"]
