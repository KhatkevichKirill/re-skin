"""Slim RQ worker for the "brand_asr" queue only (AICP-389).

PID 1 of the brand-asr container. It serves the localisation autopilot's free
brand check (``app.autopilot.flows.localisation_brandfix.brand_check``): local
word ASR (faster-whisper small, pinned, offline, in a seccomp'd child — the
assistant's ``LocalASR``) over bounded chunks of a stitched final, the
trailing-sibilant rule and the candidate renders.

One worker, so one ASR job at a time on this queue. No InsightFace, no run
reconciliation, no scheduler. A job killed by a restart is re-enqueued by the
autopilot's compute recovery (the effect is a compute; a duplicate attempt is
waste, never harm). The image is the application image plus the hashed ASR
wheels under /opt/asr (backend/Dockerfile, target ``brand-asr``); the run
workers and the assistant ASR sidecar are unchanged.
"""
from __future__ import annotations

import logging
import os
import sys

from redis import Redis
from rq import Queue, Worker
from rq.logutils import setup_loghandlers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("brand_asr_worker")


def main() -> None:
    setup_loghandlers("INFO")
    from app.config import settings
    from app.tasks import QUEUE_BRAND_ASR

    if not os.path.isdir("/opt/asr/faster_whisper"):
        logger.error("brand-asr worker: /opt/asr has no faster_whisper — wrong image target")
        sys.exit(2)
    model = settings.BRAND_ASR_MODEL_DIR
    if not os.path.isfile(os.path.join(model, "model.bin")):
        logger.error("brand-asr worker: no model at %s (mount the asr-small dir read-only)", model)
        sys.exit(2)
    conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
    logger.info("brand-asr worker: listening on %r (no scheduler), model %s", QUEUE_BRAND_ASR, model)
    Worker([Queue(QUEUE_BRAND_ASR, connection=conn)], connection=conn).work()


if __name__ == "__main__":
    main()
