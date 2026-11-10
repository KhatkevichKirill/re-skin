"""Serialized native planner supervisor. No database, media, or application secrets.

Only the deployment invokes main. Importing this module starts nothing. The
SQLite ledger prevents replay after a lost response or supervisor restart.
"""

from __future__ import annotations

import argparse
from collections import deque
import fcntl
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
from pathlib import Path
import re
import signal
import socket
from socketserver import ThreadingMixIn
import sqlite3
import stat
import struct
import threading
import time
from uuid import UUID

from .assistant_broker_client import BrokerError, BrokerHealth, BrokerResult, Digest, _decode, context_fingerprint
from .assistant_contracts import ContractModel
from .assistant_agent_contracts import AgentStepContext, MAX_AGENT_INPUT_BYTES
from .assistant_planner import _agent_reply
from .assistant_planner import (
    MAX_INPUT_BYTES, MAX_STDOUT_BYTES, ClaudePlanner, ClaudePlannerConfig, PlannerCancelled,
    PlannerContext, PlannerError, PlannerTimeout, _configuration, _reply, planner_failure_details,
)

MODEL = "claude-sonnet-5-5"
ACTIVE = ("queued", "running", "cancel_requested")


def _log_planner_failure(turn_id, error):
    """Only a canonical invocation UUID and fixed labels may reach the log."""
    try:
        safe_id = str(UUID(turn_id)) if type(turn_id) is str else 'invalid_invocation'
        if safe_id != turn_id:
            safe_id = 'invalid_invocation'
    except (ValueError, TypeError, AttributeError):
        safe_id = 'invalid_invocation'
    code, cause = planner_failure_details(error)
    try:
        logging.getLogger(__name__).warning('Assistant planner failure invocation=%s code=%s cause=%s',
            safe_id, code, cause)
    except Exception:
        pass  # Diagnostics must not strand the broker's durable terminal update.


class _Submit(ContractModel):
    context: PlannerContext
    agent_step: AgentStepContext | None = None


class _Fence(ContractModel):
    context_sha256: Digest


class Broker:
    def __init__(self, ledger_path: Path, planner, *, queue_limit=4, queue_timeout=120,
                 retention_sec=3600, max_results=128, max_records=100000, max_pages=4096):
        if not 1 <= queue_limit <= 4 or not 0 < queue_timeout <= 120 or not 0 < retention_sec <= 86400:
            raise ValueError("invalid supervisor bounds")
        parent = ledger_path.parent
        info = parent.stat()
        if parent.is_symlink() or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("ledger directory must be private and runner-owned")
        self._lock_fd = os.open(str(ledger_path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if ledger_path.is_symlink():
                raise ValueError("ledger cannot be a symlink")
            self.db = sqlite3.connect(ledger_path, check_same_thread=False, timeout=1)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            actual_limit = self.db.execute("PRAGMA max_page_count=" + str(int(max_pages))).fetchone()[0]
            if actual_limit != max_pages or self.db.execute("PRAGMA page_size").fetchone()[0] != 4096:
                raise ValueError("ledger exceeds its storage bound")
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("invalid ledger")
            self.db.execute("""CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, state TEXT NOT NULL,
                reply TEXT, error TEXT, acknowledged INTEGER NOT NULL DEFAULT 0,
                updated REAL NOT NULL)""")
            self.db.execute("UPDATE turns SET state='interrupted',error='interrupted',updated=? "
                            "WHERE state IN ('queued','running','cancel_requested')", (time.time(),))
            self.db.commit()
            for row in self.db.execute("SELECT * FROM turns"):
                self._result(row)
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            os.close(self._lock_fd)
            raise BrokerError("ledger_unavailable") from None
        self.planner = planner
        self.queue_limit, self.queue_timeout, self.retention_sec = queue_limit, queue_timeout, retention_sec
        self.max_results, self.max_records = max_results, max_records
        self.condition = threading.Condition(threading.RLock())
        self.pending = deque()
        self.events = {}
        self.active_id = None
        self.stopping = False
        self.poisoned = False
        self.worker = threading.Thread(target=self._work, name="planner", daemon=True)
        self.worker.start()

    @staticmethod
    def _result(row) -> BrokerResult:
        if row is None:
            raise BrokerError("not_found", 404)
        stored = None if row["reply"] is None else _decode(row["reply"].encode())
        is_action = isinstance(stored, dict) and set(stored) == {"action"}
        return BrokerResult.model_validate({
            "turn_id": row["id"], "context_sha256": row["fingerprint"], "state": row["state"],
            "reply": None if is_action else stored,
            **({"action": stored["action"]} if is_action else {}),
            "error_code": row["error"], "acknowledged": bool(row["acknowledged"]),
        })

    def _ready(self):
        if self.poisoned or self.stopping:
            raise BrokerError("ledger_unavailable")

    def _poison(self):
        self.poisoned = True
        for event in self.events.values():
            event.set()
        self.condition.notify_all()

    def _sql(self, query, values=()):
        try:
            return self.db.execute(query, values)
        except sqlite3.Error:
            self._poison()
            raise BrokerError("ledger_unavailable") from None

    def _commit(self):
        try:
            self.db.commit()
        except sqlite3.Error:
            self._poison()
            raise BrokerError("ledger_unavailable") from None

    def _get(self, turn_id):
        try:
            return self._result(self._sql("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone())
        except BrokerError:
            raise  # An absent ID is expected, not evidence of ledger corruption.
        except (ValueError, TypeError, KeyError, RecursionError):
            self._poison()
            raise BrokerError("ledger_unavailable") from None

    def _expire(self):
        self._sql("UPDATE turns SET reply=NULL,state='failed',error='expired' WHERE reply IS NOT NULL "
                  "AND acknowledged=0 AND updated<?", (time.time() - self.retention_sec,))
        self._commit()

    def submit(self, turn_id: str, context: PlannerContext, *, agent_step: AgentStepContext | None = None) -> BrokerResult:
        from .assistant_broker_client import BrokerClient
        turn_id = BrokerClient._id(turn_id)
        fingerprint = context_fingerprint(context, agent_step)
        with self.condition:
            self._ready()
            self._expire()
            existing = self._sql("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
            if existing is not None:
                if existing["fingerprint"] != fingerprint:
                    raise BrokerError("conflict", 409)
                return self._get(turn_id)
            if len(self.pending) >= self.queue_limit:
                raise BrokerError("backpressure", 429)
            count = self._sql("SELECT count(*) FROM turns").fetchone()[0]
            results = self._sql("SELECT count(*) FROM turns WHERE acknowledged=0 AND state NOT IN "
                                "('queued','running','cancel_requested')").fetchone()[0]
            if count >= self.max_records or results >= self.max_results:
                raise BrokerError("backpressure", 429)
            self._sql("INSERT INTO turns(id,fingerprint,state,updated) VALUES (?,?,'queued',?)",
                      (turn_id, fingerprint, time.time()))
            self._commit()  # Durable intent precedes native execution.
            self.events[turn_id] = threading.Event()
            self.pending.append((turn_id, context, agent_step, time.monotonic()))
            result = self._get(turn_id)
            self.condition.notify_all()
            return result

    def poll(self, turn_id):
        with self.condition:
            self._ready()
            self._expire()
            return self._get(turn_id)

    def cancel(self, turn_id):
        with self.condition:
            self._ready()
            result = self._get(turn_id)
            if result.state == "queued":
                self.pending = deque(item for item in self.pending if item[0] != turn_id)
                self.events.pop(turn_id).set()
                self._terminal(turn_id, "cancelled", error="cancelled")
            elif result.state in ("running", "cancel_requested"):
                self.events[turn_id].set()
                self._sql("UPDATE turns SET state='cancel_requested',updated=? WHERE id=?", (time.time(), turn_id))
                self._commit()
            return self._get(turn_id)

    def acknowledge(self, turn_id):
        with self.condition:
            self._ready()
            if self._get(turn_id).state in ACTIVE:
                raise BrokerError("conflict", 409)
            self._sql("UPDATE turns SET acknowledged=1,reply=NULL WHERE id=?", (turn_id,))
            self._commit()
            return self._get(turn_id)

    def fence(self, turn_id, context_sha256):
        """Cancel an existing intent or durably prevent a delayed first submit.

        Absent IDs have no execution or payload to acknowledge. Their compact,
        acknowledged tombstone consumes the same bounded durable ID capacity as
        every completed turn, without occupying an unacknowledged result slot.
        """
        from .assistant_broker_client import BrokerClient
        turn_id = BrokerClient._id(turn_id)
        fingerprint = _Fence(context_sha256=context_sha256).context_sha256
        with self.condition:
            self._ready()
            existing = self._sql("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
            if existing is not None:
                result = self._get(turn_id)
                if result.context_sha256 != fingerprint:
                    raise BrokerError("conflict", 409)
                return self.cancel(turn_id)
            if self._sql("SELECT count(*) FROM turns").fetchone()[0] >= self.max_records:
                raise BrokerError("backpressure", 429)
            self._sql("INSERT INTO turns(id,fingerprint,state,error,acknowledged,updated) "
                      "VALUES (?,?,'cancelled','cancelled',1,?)", (turn_id, fingerprint, time.time()))
            self._commit()
            return self._get(turn_id)

    def health(self):
        with self.condition:
            return BrokerHealth(ready=not self.poisoned and not self.stopping,
                                active=int(self.active_id is not None), queued=len(self.pending), model=MODEL)

    def _terminal(self, turn_id, state, *, reply=None, error=None):
        self._sql("UPDATE turns SET state=?,reply=?,error=?,updated=? WHERE id=?",
                  (state, reply, error, time.time(), turn_id))
        self._commit()

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.stopping or self.poisoned)
                if self.stopping or self.poisoned:
                    return
                turn_id, context, agent_step, queued_at = self.pending.popleft()
                event = self.events[turn_id]
                self.active_id = turn_id
                try:
                    if time.monotonic() - queued_at > self.queue_timeout:
                        self._terminal(turn_id, "failed", error="timeout")
                        self.events.pop(turn_id)
                        self.active_id = None
                        continue
                    self._sql("UPDATE turns SET state='running',updated=? WHERE id=?", (time.time(), turn_id))
                    self._commit()
                except BrokerError:
                    return
            state, result, error = "failed", None, "planner_failed"
            try:
                if agent_step is None:
                    reply = self.planner.plan(context, cancelled=event.is_set)
                    # Even injected implementations cannot bypass pinned/domain checks.
                    reply = _reply(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                                              "structured_output": {"reply": reply.model_dump(mode="json")}}).encode(), context)
                    result = reply.model_dump_json()
                else:
                    action = self.planner.plan_step(agent_step, cancelled=event.is_set)
                    action = _agent_reply(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                        "structured_output": {"action": action.model_dump(mode="json")}}).encode(), agent_step)
                    result = json.dumps({"action": action.model_dump(mode="json")}, separators=(",", ":"))
                if len(result.encode()) > MAX_STDOUT_BYTES - 2048:
                    raise PlannerError("oversized reply")
                state, error = "succeeded", None
            except PlannerCancelled as failure:
                state, error = "cancelled", "cancelled"
                _log_planner_failure(turn_id, failure)
            except PlannerTimeout as failure:
                error = "timeout"
                _log_planner_failure(turn_id, failure)
            except Exception as failure:
                _log_planner_failure(turn_id, failure)
            with self.condition:
                try:
                    if event.is_set():
                        state, result, error = "cancelled", None, "cancelled"
                    self._terminal(turn_id, state, reply=result, error=error)
                except BrokerError:
                    return
                finally:
                    self.events.pop(turn_id, None)
                    self.active_id = None

    def close(self):
        with self.condition:
            self.stopping = True
            for event in self.events.values():
                event.set()
            self.condition.notify_all()
        self.worker.join(timeout=125)
        if self.worker.is_alive():
            raise BrokerError("unavailable")  # External init/cgroup owns final cleanup.
        self.db.close()
        os.close(self._lock_fd)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _respond(self, status, value):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle(self):
        try:
            pid, uid, gid = struct.unpack("3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid not in self.server.allowed_uids:
                raise BrokerError("unauthorized", 401)
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") is not None or len(lengths) > 1:
                raise BrokerError("invalid_request", 400)
            length = lengths[0] if lengths else "0"
            if not length.isdecimal() or len(length) > 8 or int(length) > MAX_AGENT_INPUT_BYTES:
                raise BrokerError("invalid_request", 400)
            size = int(length)
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise BrokerError("invalid_request", 400)
            if self.path == "/health" and self.command == "GET" and not raw:
                return self._respond(200, self.server.broker.health().model_dump(mode="json"))
            match = re.fullmatch(r"/v1/turns/([0-9a-f-]{36})(/cancel|/acknowledge|/fence)?", self.path)
            if match is None:
                raise BrokerError("not_found", 404)
            turn_id, action = match.groups()
            from .assistant_broker_client import BrokerClient
            BrokerClient._id(turn_id)
            if self.command == "PUT" and action is None:
                submission = _Submit.model_validate(_decode(raw))
                result = self.server.broker.submit(turn_id, submission.context, agent_step=submission.agent_step)
            elif self.command == "POST" and action == "/fence":
                result = self.server.broker.fence(turn_id, _Fence.model_validate(_decode(raw)).context_sha256)
            elif raw:
                raise BrokerError("invalid_request", 400)
            elif self.command == "GET" and action is None:
                result = self.server.broker.poll(turn_id)
            elif self.command == "POST" and action == "/cancel":
                result = self.server.broker.cancel(turn_id)
            elif self.command == "POST" and action == "/acknowledge":
                result = self.server.broker.acknowledge(turn_id)
            else:
                raise BrokerError("not_found", 404)
            self._respond(202 if result.state in ACTIVE else 200, result.model_dump(mode="json"))
        except BrokerError as error:
            self._respond(error.status, {"error_code": error.code})
        except (ValueError, KeyError, TypeError, RecursionError):
            self._respond(400, {"error_code": "invalid_request"})
        except Exception:
            self._respond(503, {"error_code": "unavailable"})

    do_GET = do_POST = do_PUT = _handle


class BrokerServer(ThreadingMixIn, HTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, path: Path, broker: Broker, *, allowed_uids=(10001,)):
        if not allowed_uids or any(type(uid) is not int or uid < 1 for uid in allowed_uids):
            raise ValueError("explicit unprivileged peer UIDs required")
        self.broker, self.allowed_uids = broker, frozenset(allowed_uids)
        self.slots = threading.BoundedSemaphore(8)
        # Do not let HTTPServer inspect TCP hostnames for this Unix transport.
        super().__init__(str(path), _Handler, bind_and_activate=False)
        self.socket.bind(str(path))
        os.chmod(path, 0o666)  # Kernel peer credentials, not pathname secrecy, authenticate callers.
        self.server_activate()
        self.server_name, self.server_port = "localhost", 0

    def process_request(self, request, client_address):
        request.settimeout(5)
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        pass


def main():
    from .assistant_egress import ProxyRelay
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-reference", required=True)
    parser.add_argument("--socket", type=Path, default=Path("/socket/broker.sock"))
    parser.add_argument("--ledger", type=Path, default=Path("/state/ledger.sqlite"))
    parser.add_argument("--peer-uid", type=int, action="append", default=None)
    args = parser.parse_args()
    os.umask(0o077)
    config = ClaudePlannerConfig(binary=Path("/opt/reskin-assistant/bin/claude"), model=MODEL,
                                 config_dir=Path("/config"), cwd=Path("/work"), safe_path=(Path("/usr/bin"),),
                                 enabled=True, isolation_qualified=True,
                                 qualification_reference=args.qualification_reference, timeout_sec=120,
                                 https_proxy="http://127.0.0.1:18080")
    _configuration(config)
    # One native config can be owned by only one supervisor, including across containers.
    config_lock = os.open(config.config_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fcntl.flock(config_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    relay = ProxyRelay(Path("/egress/proxy.sock"))
    relay.start()
    broker = Broker(args.ledger, ClaudePlanner(config))
    if args.socket.exists():
        if not stat.S_ISSOCK(args.socket.lstat().st_mode):
            raise ValueError("unexpected socket entry")
        args.socket.unlink()
    server = BrokerServer(args.socket, broker, allowed_uids=tuple(args.peer_uid or [10001]))
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        broker.close()
        relay.close()
        args.socket.unlink(missing_ok=True)
        os.close(config_lock)


if __name__ == "__main__":
    main()
