"""PID 1 of the fal-worker container (AICP-330).

The knowledge supervisor's loop (``worker.knowledge_supervisor.supervise``)
with the Fal children: if the RQ worker or the ticker exits, the other is
terminated and the container exits non-zero, so ``restart: unless-stopped``
replaces both — a dead ticker can never leave Fal recovery silently stopped.
"""
from __future__ import annotations

import sys

from worker.knowledge_supervisor import supervise

CHILDREN = (("worker", "worker.fal_worker"), ("ticker", "worker.fal_ticker"))


if __name__ == "__main__":
    sys.exit(supervise([(name, [sys.executable, "-m", mod]) for name, mod in CHILDREN]))
