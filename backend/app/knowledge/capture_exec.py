"""The capture executor (AICP-299 §1.3): bounded, never blocking, self-healing.

Every DB-touching capture call runs here, never on a pipeline thread. The
caller waits at most a budget on an ``Event`` and then abandons the call; a
full queue means the call is skipped. Threads are daemon threads (the stdlib
``ThreadPoolExecutor`` cannot make them and joins at exit), so abandoned work
dies with the RQ work-horse's ``os._exit`` and PostgreSQL rolls its open
transaction back on disconnect. Each thread owns one connection with
connect/statement/lock timeouts; a dead thread is replaced on the next call,
and a connection error drops the session so the next job reconnects.

A daemon heartbeat thread publishes the executor's health to Redis every
15 s (TTL 60 s), independent of capture calls.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import socket
import threading
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)

QUEUE_MAX = 8
THREADS = 2
HEARTBEAT_SEC = 15
HEARTBEAT_TTL_SEC = 60
HEALTH_KEY_PREFIX = "reskin:knowledge:capture:health:"
COUNTER_KEY_PREFIX = "reskin:knowledge:capture:"

_lock = threading.Lock()
_state: dict = {"pid": None, "engine": None, "executor": None, "redis": None}


# ---------------------------------------------------------------- engine / redis


def capture_engine():
    """A small engine of capture's own (never the pipeline's pool), per PID."""
    from sqlalchemy import create_engine

    from ..config import settings

    _fork_check()
    if _state["engine"] is None:
        with _lock:
            if _state["engine"] is None:
                _state["engine"] = create_engine(
                    settings.DATABASE_URL, pool_size=THREADS, max_overflow=0, pool_timeout=0.5,
                    pool_pre_ping=True,
                    connect_args={"connect_timeout": 2, "keepalives": 1, "keepalives_idle": 5,
                                  "keepalives_interval": 2, "keepalives_count": 2, "tcp_user_timeout": 5000,
                                  "options": "-c statement_timeout=1000 -c lock_timeout=1000"},
                )
    return _state["engine"]


def new_session():
    from sqlalchemy.orm import Session

    return Session(bind=capture_engine(), autoflush=False, expire_on_commit=False)


def redis_client():
    from redis import Redis

    from ..config import settings

    _fork_check()
    if _state["redis"] is None:
        _state["redis"] = Redis.from_url(settings.REDIS_URL, socket_timeout=0.2, socket_connect_timeout=0.2)
    return _state["redis"]


def count(site: str, outcome: str) -> None:
    """Best-effort Redis counter (not evidence); never raises."""
    try:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        r = redis_client()
        key = f"{COUNTER_KEY_PREFIX}{day}"
        r.hincrby(key, f"{site}:{outcome}", 1)
        r.expire(key, 35 * 86400)
    except Exception:
        pass


def _fork_check() -> None:
    pid = os.getpid()
    if _state["pid"] != pid:
        with _lock:
            if _state["pid"] != pid:
                # A forked child must never reuse the parent's connections or threads.
                _state.update(pid=pid, engine=None, executor=None, redis=None)


def _after_fork_in_child() -> None:
    # The parent may fork while one of its capture threads holds ``_lock``
    # (B3 runs capture in the RQ parent at boot); the child inherits it held,
    # with no thread left to release it. Replace it before any hook runs.
    global _lock
    _lock = threading.Lock()
    _state.update(pid=os.getpid(), engine=None, executor=None, redis=None)


os.register_at_fork(after_in_child=_after_fork_in_child)


# ---------------------------------------------------------------- executor


class _Job:
    __slots__ = ("fn", "done", "result", "error")

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.result = None
        self.error = None


class CaptureExecutor:
    def __init__(self, session_factory: Callable = new_session, threads: int = THREADS,
                 queue_max: int = QUEUE_MAX, heartbeat: bool = True):
        self.q: "queue.Queue[_Job]" = queue.Queue(maxsize=queue_max)
        self.session_factory = session_factory
        self.threads: list = [None] * threads
        self.heartbeat_thread = None
        self.heartbeat = heartbeat
        self.last_attempt: Optional[float] = None
        self.last_ok: Optional[float] = None
        self.consecutive_failures = 0
        self._guard = threading.Lock()
        self._ensure_threads()

    # supervision: called on every call()
    def _ensure_threads(self) -> None:
        with self._guard:
            for i, t in enumerate(self.threads):
                if t is None or not t.is_alive():
                    if t is not None:
                        count("executor", "thread_replaced")
                    self.threads[i] = threading.Thread(target=self._loop, name=f"kb-capture-{i}", daemon=True)
                    self.threads[i].start()
            if self.heartbeat and (self.heartbeat_thread is None or not self.heartbeat_thread.is_alive()):
                self.heartbeat_thread = threading.Thread(target=self._beat, name="kb-capture-heartbeat",
                                                         daemon=True)
                self.heartbeat_thread.start()

    def _loop(self) -> None:
        session = None
        while True:
            job = self.q.get()
            self.last_attempt = time.time()
            try:
                if session is None:
                    session = self.session_factory()  # lazy, per thread
                job.result = job.fn(session)
                session.commit()
                self.last_ok = time.time()
                self.consecutive_failures = 0
            except BaseException as exc:  # noqa: BLE001 — a capture failure must never kill the thread
                job.error = exc
                self.consecutive_failures += 1
                try:
                    if session is not None:
                        session.rollback()
                except Exception:
                    pass
                if session is not None and _is_connection_error(exc):
                    try:
                        session.close()
                    except Exception:
                        pass
                    session = None  # reconnect on the next job
            finally:
                job.done.set()

    def call(self, fn: Callable, budget_sec: float, *, site: str = "capture"):
        """Run ``fn(session)`` on a capture thread; wait at most ``budget_sec``.

        Returns the result, or None when skipped (queue full), abandoned (over
        budget), fire-and-forget (budget 0) or failed. Never raises.
        """
        try:
            self._ensure_threads()
            job = _Job(fn)
            try:
                self.q.put_nowait(job)
            except queue.Full:
                count(site, "executor_full")
                return None
            if budget_sec <= 0:
                return None
            if not job.done.wait(budget_sec):
                count(site, "budget_exceeded")
                return None
            if job.error is not None:
                count(site, "error")
                log.warning("KNOWLEDGE_CAPTURE site=%s outcome=error err=%s: %s", site,
                            type(job.error).__name__, job.error)
                return None
            count(site, "ok")
            return job.result
        except Exception:  # pragma: no cover — defensive
            return None

    def health(self) -> dict:
        return {"last_attempt": self.last_attempt, "last_ok": self.last_ok,
                "consecutive_failures": self.consecutive_failures,
                "threads_alive": sum(1 for t in self.threads if t is not None and t.is_alive()),
                "queue_depth": self.q.qsize(), "pid": os.getpid()}

    def _beat(self) -> None:
        key = f"{HEALTH_KEY_PREFIX}{socket.gethostname()}:{os.getpid()}"
        while True:
            try:
                redis_client().set(key, json.dumps(self.health()), ex=HEARTBEAT_TTL_SEC)
            except Exception:
                pass
            time.sleep(HEARTBEAT_SEC)


def _is_connection_error(exc: BaseException) -> bool:
    try:
        from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
    except Exception:  # pragma: no cover
        return False
    if isinstance(exc, (OperationalError, InterfaceError)):
        return True
    return isinstance(exc, DBAPIError) and bool(getattr(exc, "connection_invalidated", False))


def executor() -> CaptureExecutor:
    _fork_check()
    if _state["executor"] is None:
        with _lock:
            if _state["executor"] is None:
                _state["executor"] = CaptureExecutor()
    return _state["executor"]


def read_health_keys(redis) -> dict:
    """All live capture health keys (missing = no process, not stale)."""
    out = {}
    for key in redis.scan_iter(match=f"{HEALTH_KEY_PREFIX}*", count=200):
        try:
            k = key.decode() if isinstance(key, bytes) else key
            out[k[len(HEALTH_KEY_PREFIX):]] = json.loads(redis.get(key) or b"null")
        except Exception:
            continue
    return out


def unhealthy(health: Optional[dict]) -> bool:
    """§1.3: an existing key with ≥ 5 consecutive failures or no success for 600 s after an attempt."""
    if not health:
        return False
    if (health.get("consecutive_failures") or 0) >= 5:
        return True
    la, lo = health.get("last_attempt"), health.get("last_ok")
    return la is not None and (la - (lo or 0)) > 600


__all__ = ["CaptureExecutor", "capture_engine", "count", "executor", "new_session", "read_health_keys",
           "redis_client", "unhealthy"]
