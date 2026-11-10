"""PID 1 of the knowledge-worker container (AICP-299 §6.2).

A plain loop with no Redis and no threads. It starts the RQ worker
(``worker.knowledge_worker``) and the ticker (``worker.knowledge_ticker``) as
child processes and waits. If either child exits, it terminates the other and
exits non-zero, so ``restart: unless-stopped`` restarts the whole container —
a dead ticker or worker is replaced in seconds, even while a tick runs.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

CHILDREN = (("worker", "worker.knowledge_worker"), ("ticker", "worker.knowledge_ticker"))
TERM_GRACE_SEC = 30


def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [knowledge_supervisor] {msg}", flush=True)


def supervise(commands=None, popen=subprocess.Popen, wait=os.wait) -> int:
    """Run the children until one exits; returns the supervisor's exit code."""
    commands = commands or [(name, [sys.executable, "-m", mod]) for name, mod in CHILDREN]
    procs = {popen(argv).pid: name for name, argv in commands}
    stopping = {"flag": False}

    def _term(signum, _frame):
        stopping["flag"] = True
        for pid in list(procs):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    pid, status = wait()
    name = procs.pop(pid, "?")
    if not stopping["flag"]:
        _log(f"knowledge_container_restarted: child {name} (pid {pid}) exited with status {status}")
    for other in list(procs):
        try:
            os.kill(other, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + TERM_GRACE_SEC
    while procs and time.monotonic() < deadline:
        try:
            done, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if done:
            procs.pop(done, None)
        else:
            time.sleep(0.2)
    for other in list(procs):
        try:
            os.kill(other, signal.SIGKILL)
        except OSError:
            pass
    return 0 if stopping["flag"] else 1


if __name__ == "__main__":
    sys.exit(supervise())
