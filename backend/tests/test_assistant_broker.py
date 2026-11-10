"""Offline broker tests use injected planners and local sockets only."""

from dataclasses import replace
import json
import os
import socket
import threading
import time

import pytest

from app.assistant_broker import Broker, BrokerServer
from app.assistant_broker_client import BrokerClient, BrokerError, context_fingerprint
from app.assistant_planner import MAX_INPUT_BYTES, ClarifyReply, PlannerCancelled, PlannerError, PlannerTimeout
from tests.test_assistant_planner import context, reply, fake

TURN = "66666666-6666-4666-8666-666666666666"
OTHER_TURN = "77777777-7777-4777-8777-777777777777"


class FakePlanner:
    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.error = None

    def plan(self, context, *, cancelled):
        self.calls += 1
        self.started.set()
        while self.block and not self.release.wait(.01):
            if cancelled():
                raise PlannerCancelled("synthetic private message")
        if self.error:
            raise self.error
        return ClarifyReply.model_validate(reply())


@pytest.fixture
def setup_broker(tmp_path):
    instances = []
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    path = root / "ledger.sqlite"
    def make(**kwargs):
        planner = FakePlanner()
        broker = Broker(path, planner, **kwargs)
        instances.append(broker)
        return broker, planner
    yield make, path
    for broker in instances:
        if not broker.stopping:
            broker.close()


def terminal(broker, turn_id=TURN):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = broker.poll(turn_id)
        if result.state not in ("queued", "running", "cancel_requested"):
            return result
        time.sleep(.01)
    pytest.fail("broker did not terminate")


def near_limit_context(extra_bytes=0):
    dialogue = [{"role": "operator", "text": "x" * 2000} for _ in range(31)]
    dialogue.append({"role": "operator", "text": "x" * 1000})
    candidate = context(dialogue=dialogue)
    raw = json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")).encode()
    padding = MAX_INPUT_BYTES - 64 - len(raw) + extra_bytes
    dialogue[-1]["text"] += "x" * padding
    return context(dialogue=dialogue)


def test_context_fingerprint_reserves_transport_envelope():
    candidate = near_limit_context()
    assert len(json.dumps(candidate.model_dump(mode="json"), separators=(",", ":")).encode()) == MAX_INPUT_BYTES - 64
    assert len(context_fingerprint(candidate)) == 64
    with pytest.raises(BrokerError, match="invalid_request"):
        context_fingerprint(near_limit_context(1))


def test_idempotency_hash_scope_and_ack_keep_tombstone(setup_broker):
    make, _ = setup_broker
    broker, planner = make()
    assert broker.submit(TURN, context()).context_sha256 == context_fingerprint(context())
    result = terminal(broker)
    assert result.state == "succeeded" and result.reply.kind == "clarify"
    assert broker.submit(TURN, context()).state == "succeeded" and planner.calls == 1
    with pytest.raises(BrokerError, match="conflict"):
        broker.submit(TURN, context(dialogue=[{"role": "operator", "text": "changed"}]))
    ack = broker.acknowledge(TURN)
    assert ack.acknowledged and ack.reply is None and ack.state == "succeeded"
    assert broker.submit(TURN, context()).acknowledged and planner.calls == 1


def test_missing_intent_fence_survives_restart_and_blocks_delayed_inference(setup_broker):
    make, _ = setup_broker
    broker, planner = make(max_results=1, max_records=2)
    broker.submit(TURN, context())
    assert terminal(broker).state == "succeeded"  # Fill the unacknowledged cap.
    fingerprint = context_fingerprint(context())
    with pytest.raises(BrokerError, match="not_found"):
        broker.poll(OTHER_TURN)
    assert broker.health().ready
    result = broker.fence(OTHER_TURN, fingerprint)
    assert result.state == "cancelled" and result.acknowledged and result.reply is None
    with pytest.raises(BrokerError, match="conflict"):
        broker.fence(OTHER_TURN, "a" * 64)
    with pytest.raises(BrokerError, match="backpressure"):
        broker.fence("88888888-8888-4888-8888-888888888888", fingerprint)
    assert broker.submit(OTHER_TURN, context()).acknowledged and planner.calls == 1
    broker.close()
    broker2, planner2 = make()
    assert broker2.submit(OTHER_TURN, context()).state == "cancelled"
    assert broker2.poll(OTHER_TURN).acknowledged and planner2.calls == 0


def test_existing_intent_fence_checks_hash_and_waits_for_process_exit(setup_broker):
    make, _ = setup_broker
    broker, planner = make()
    planner.block = True
    broker.submit(TURN, context())
    assert planner.started.wait(1)
    with pytest.raises(BrokerError, match="conflict"):
        broker.fence(TURN, "a" * 64)
    assert broker.poll(TURN).state == "running"
    result = broker.fence(TURN, context_fingerprint(context()))
    assert result.state == "cancel_requested" and not result.acknowledged
    assert terminal(broker).state == "cancelled"
    assert broker.acknowledge(TURN).acknowledged


def test_restart_never_replays_queued_or_active_intents(setup_broker):
    make, _ = setup_broker
    broker, planner = make()
    planner.block = True
    broker.submit(TURN, context())
    assert planner.started.wait(1)
    broker.submit(OTHER_TURN, context())
    broker.close()
    broker2, planner2 = make()
    assert broker2.poll(OTHER_TURN).state == "interrupted"
    assert broker2.submit(OTHER_TURN, context()).state == "interrupted"
    assert planner2.calls == 0


def test_serialized_queue_backpressure_and_queued_cancel(setup_broker):
    make, _ = setup_broker
    broker, planner = make(queue_limit=1)
    planner.block = True
    broker.submit(TURN, context())
    assert planner.started.wait(1)
    broker.submit(OTHER_TURN, context())
    with pytest.raises(BrokerError, match="backpressure"):
        broker.submit("88888888-8888-4888-8888-888888888888", context())
    assert broker.health().active == 1 and broker.health().queued == 1
    assert broker.cancel(OTHER_TURN).state == "cancelled"
    assert planner.calls == 1
    assert broker.cancel(TURN).state == "cancel_requested"
    assert terminal(broker).state == "cancelled"


def test_cancel_after_planner_return_suppresses_success(setup_broker):
    make, _ = setup_broker
    broker, planner = make()
    planner.block = True
    broker.submit(TURN, context())
    assert planner.started.wait(1)
    broker.cancel(TURN)
    planner.release.set()
    assert terminal(broker).state == "cancelled"


@pytest.mark.parametrize("error,code", [(PlannerError("private-token"), "planner_failed"),
                                       (RuntimeError("private-token"), "planner_failed"),
                                       (PlannerTimeout("private-token"), "timeout")])
def test_failure_is_fixed_code_without_native_diagnostics(setup_broker, error, code):
    make, _ = setup_broker
    broker, planner = make()
    planner.error = error
    broker.submit(TURN, context())
    result = terminal(broker)
    assert result.error_code == code and result.reply is None
    assert "private-token" not in result.model_dump_json()


def test_queue_deadline_and_storage_limits_fail_closed(setup_broker):
    make, _ = setup_broker
    broker, planner = make(queue_timeout=.02, max_records=2)
    planner.block = True
    broker.submit(TURN, context())
    assert planner.started.wait(1)
    broker.submit(OTHER_TURN, context())
    time.sleep(.03)
    planner.release.set()
    assert terminal(broker, OTHER_TURN).error_code == "timeout"
    assert planner.calls == 1
    with pytest.raises(BrokerError, match="backpressure"):
        broker.submit("88888888-8888-4888-8888-888888888888", context())


def test_reply_retention_drops_payload_without_replay(setup_broker):
    make, _ = setup_broker
    broker, planner = make(retention_sec=.03)
    broker.submit(TURN, context())
    assert terminal(broker).reply is not None
    time.sleep(.04)
    assert broker.poll(TURN).error_code == "expired"
    assert broker.submit(TURN, context()).reply is None and planner.calls == 1


def test_lock_and_corrupt_ledger_refuse_startup(setup_broker):
    make, path = setup_broker
    broker, _ = make()
    with pytest.raises(BrokerError, match="ledger_unavailable"):
        Broker(path, FakePlanner())
    broker.close()
    path.write_bytes(b"not a database")
    with pytest.raises(BrokerError, match="ledger_unavailable"):
        Broker(path, FakePlanner())


def test_runtime_ledger_corruption_marks_unready(setup_broker):
    make, _ = setup_broker
    broker, planner = make()
    broker.submit(TURN, context())
    terminal(broker)
    with broker.condition:
        broker.db.execute("UPDATE turns SET state='invalid'")
        broker.db.commit()
    with pytest.raises(BrokerError, match="ledger_unavailable"):
        broker.poll(TURN)
    assert not broker.health().ready


def test_actual_sqlite_storage_exhaustion_stops_admission(setup_broker):
    make, _ = setup_broker
    broker, _ = make(max_pages=4)
    broker.submit(TURN, context())
    terminal(broker)
    with broker.condition:
        with pytest.raises(BrokerError, match="ledger_unavailable"):
            broker._sql("UPDATE turns SET reply=? WHERE id=?", ("x" * 65536, TURN))
    assert not broker.health().ready
    with pytest.raises(BrokerError, match="ledger_unavailable"):
        broker.submit(OTHER_TURN, context())


def test_unix_transport_auth_roundtrip_and_client_bounds(setup_broker, tmp_path):
    make, _ = setup_broker
    broker, planner = make()
    path = tmp_path / "broker.sock"
    server = BrokerServer(path, broker, allowed_uids=(10001,))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
    thread.start()
    client = BrokerClient(path)
    try:
        if os.getuid() != 10001:
            with pytest.raises(BrokerError, match="unauthorized"):
                client.health()
        # Tests alone permit their current process UID (including root). Native
        # deployment refuses a root peer allowlist and runs exclusively uid999.
        server.allowed_uids = frozenset({os.getuid()})
        assert client.health().ready
        assert client.submit(TURN, context()).context_sha256 == context_fingerprint(context())
        terminal(broker)
        assert client.poll(TURN).reply.kind == "clarify"
        assert client.acknowledge(TURN).acknowledged
        candidate = near_limit_context()
        assert client.submit(OTHER_TURN, candidate).context_sha256 == context_fingerprint(candidate)
        assert terminal(broker, OTHER_TURN).state == "succeeded"
        fenced_id = "88888888-8888-4888-8888-888888888888"
        assert client.fence(fenced_id, context_fingerprint(context())).acknowledged
        assert client.submit(fenced_id, context()).state == "cancelled"
        with pytest.raises(BrokerError, match="invalid_request"):
            client.poll("../../file")
        with pytest.raises(BrokerError, match="invalid_request"):
            client.submit(OTHER_TURN, context(dialogue=[{"role": "operator", "text": "x" * 4000}] * 32))
        with socket.socket(socket.AF_UNIX) as raw:
            raw.connect(str(path))
            raw.sendall(b"PUT /v1/turns/" + OTHER_TURN.encode() + b" HTTP/1.0\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}")
            assert b"400" in raw.recv(1024)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_trusted_proxy_environment_and_validation(fake, tmp_path):
    from app import assistant_planner as planner
    record = tmp_path / "env.json"
    body = ("sys.stdin.buffer.read()\n" + f"open({str(record)!r},'w').write(json.dumps(dict(os.environ)))\n"
            + f"sys.stdout.write({json.dumps({'type':'result','subtype':'success','is_error':False,'structured_output':{'reply':reply()}})!r})\n")
    config = replace(fake(body), https_proxy="http://127.0.0.1:18080")
    planner.ClaudePlanner(config).plan(context())
    observed = json.loads(record.read_text())
    assert observed["HTTPS_PROXY"] == observed["HTTP_PROXY"] == config.https_proxy
    assert observed["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert observed["CLAUDE_CODE_DISABLE_ARTIFACT"] == "1"
    assert observed["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert "ANTHROPIC_API_KEY" not in observed and "NO_PROXY" not in observed
    with pytest.raises(planner.PlannerError, match="proxy"):
        planner.ClaudePlanner(replace(config, https_proxy="http://private.invalid:80")).plan(context())
