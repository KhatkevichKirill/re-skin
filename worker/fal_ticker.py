"""The Fal ticker (AICP-330): recovery pass + watchdog, every FAL_TICK_INTERVAL_SEC.

Started by ``worker.fal_supervisor`` as its own process. It runs the provider
recovery (``pipeline_provider.reconcile_provider_tasks``) and the watchdog
DIRECTLY, not as RQ jobs, so neither waits behind the fal queue it watches. A
Redis lease keeps a second container (if ever scaled) from running the same
pass at the same time; every action of the pass is claim-guarded anyway.
"""
from __future__ import annotations

import logging
import os
import socket
import time
import uuid

from redis import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("fal_ticker")

LEASE_KEY = "reskin:fal-tick:lease"


def tick_once(conn, ticker_id: str) -> bool:
    from app.config import settings
    from app.pipeline_provider import reconcile_provider_tasks, watchdog

    if not conn.set(LEASE_KEY, ticker_id, nx=True, ex=max(settings.FAL_TICK_INTERVAL_SEC * 3, 60)):
        return False
    try:
        report = reconcile_provider_tasks(conn)
        if report:
            logger.info("fal recovery: %s", report)
        watchdog(conn)
    finally:
        if conn.get(LEASE_KEY) == ticker_id.encode():
            conn.delete(LEASE_KEY)
    return True


def main() -> None:
    from app.config import settings

    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    ticker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    logger.info("fal ticker %s started", ticker_id)
    while True:
        try:
            tick_once(conn, ticker_id)
        except Exception:
            logger.exception("fal ticker: pass failed (next loop retries)")
        time.sleep(settings.FAL_TICK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
