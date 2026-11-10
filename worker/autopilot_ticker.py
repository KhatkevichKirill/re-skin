"""The autopilot ticker (AICP-363, docs/autopilot-design.md §5.1).

PID 1 of the re-skin-autopilot-worker container: a plain loop, not an RQ
worker. Every AUTOPILOT_TICK_INTERVAL_SEC (or within a second of an operator
answer / resume, via the Redis wake key) it runs one pass over the autopilots
that need ticking. It runs no run recovery, imports no InsightFace and never
does heavy work: compute effects go to the run workers' autopilot_compute
queue. With AUTOPILOT_ENABLED false it only sleeps.

A Redis pass lease keeps a second container (if ever scaled) from running the
same pass concurrently. Correctness never depends on it: every admission is
fenced in the database (unique effect key, version compare-and-set, gate row).
"""
from __future__ import annotations

import logging
import os
import socket
import time
import uuid

from redis import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("autopilot_ticker")

LEASE_KEY = "reskin:autopilot-tick:lease"
WAKE_KEY = "reskin:autopilot:wake"

_RELEASE = """\
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def pass_once(conn, ticker_id: str) -> int:
    from app.autopilot.core import run_pass
    from app.config import settings

    if not settings.AUTOPILOT_ENABLED:
        return 0
    if not conn.set(LEASE_KEY, ticker_id, nx=True, ex=max(settings.AUTOPILOT_TICK_INTERVAL_SEC * 10, 300)):
        return 0
    try:
        return run_pass()
    finally:
        try:
            conn.eval(_RELEASE, 1, LEASE_KEY, ticker_id)
        except Exception:
            logger.warning("autopilot ticker: lease release failed (it expires)", exc_info=True)


def _sleep(conn, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if conn.getdel(WAKE_KEY):
                return
        except Exception:
            pass
        time.sleep(1)


def main() -> None:
    from app.autopilot import actions  # noqa: F401 — registers the handlers
    from app.autopilot import registry
    from app.config import settings

    registry.load_flows()
    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    ticker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    logger.info("autopilot ticker %s started (enabled=%s, interval=%ss)", ticker_id,
                settings.AUTOPILOT_ENABLED, settings.AUTOPILOT_TICK_INTERVAL_SEC)
    from app.autopilot import notify

    while True:
        try:
            # AICP-374: chat notifications run in their own thread, independent of
            # AUTOPILOT_ENABLED and the deploy hold; restarted here if it died.
            notify.ensure_thread()
        except Exception:
            logger.exception("autopilot ticker: cannot start the notifier")
        try:
            n = pass_once(conn, ticker_id)
            if n:
                logger.info("autopilot pass: %d autopilot(s) ticked", n)
        except Exception:
            logger.exception("autopilot ticker: pass failed (next loop retries)")
        _sleep(conn, settings.AUTOPILOT_TICK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
