"""The knowledge ticker (AICP-299 §6.2): enqueues one tick per slot.

Started by ``worker.knowledge_supervisor`` as its own process with its own
Redis connection; nothing forks from it. Every 30 s (and immediately at boot)
it computes the current slot and calls ``tasks.enqueue_knowledge_tick_for_slot``,
whose only deciding write is an idempotent ``SET slot NX`` — never deleted
early — so overlapping tickers enqueue at most one tick per slot.
"""
from __future__ import annotations

import logging
import os
import socket
import time
import uuid

from redis import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("knowledge_ticker")

LOOP_SEC = 30


def tick_once(conn, ticker_id: str, now: float | None = None) -> bool:
    from app import tasks

    slot = int(now if now is not None else time.time()) // tasks.KNOWLEDGE_TICK_INTERVAL_SEC
    return tasks.enqueue_knowledge_tick_for_slot(conn, slot, ticker_id)


def main() -> None:
    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    ticker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    logger.info("knowledge ticker %s started", ticker_id)
    while True:
        try:
            if tick_once(conn, ticker_id):
                logger.info("knowledge tick enqueued")
        except Exception:
            logger.exception("knowledge ticker: enqueue failed (next loop retries)")
        time.sleep(LOOP_SEC)


if __name__ == "__main__":
    main()
