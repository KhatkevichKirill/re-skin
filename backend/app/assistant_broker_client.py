"""Bounded Unix-socket client; never contains native runtime credentials."""

from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import socket
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StringConstraints, model_validator

from .assistant_contracts import ContractModel, EntityId
from .assistant_agent_contracts import AgentAction, AgentStepContext, MAX_AGENT_INPUT_BYTES
from .assistant_planner import MAX_INPUT_BYTES, MAX_STDOUT_BYTES, PlannerContext, PlannerReply, _pairs, _invalid_constant

BrokerState = Literal["queued", "running", "cancel_requested", "succeeded", "failed", "cancelled", "interrupted"]
ErrorCode = Literal["unavailable", "invalid_request", "unauthorized", "not_found", "conflict", "backpressure",
                    "ledger_unavailable", "planner_failed", "timeout", "cancelled", "interrupted", "expired"]
Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]


class BrokerError(ValueError):
    def __init__(self, code: str, status: int = 503):
        self.code, self.status = code, status
        super().__init__("assistant broker: " + code)


class BrokerResult(ContractModel):
    turn_id: EntityId
    context_sha256: Digest
    state: BrokerState
    reply: PlannerReply | None = None
    action: AgentAction | None = Field(default=None, exclude_if=lambda value: value is None)
    error_code: ErrorCode | None = None
    acknowledged: StrictBool = False

    @model_validator(mode="after")
    def consistent(self):
        if self.state == "succeeded":
            if self.error_code is not None or (self.reply is None and self.action is None and not self.acknowledged):
                raise ValueError("incomplete successful result")
        elif self.reply is not None or self.action is not None:
            raise ValueError("reply before successful completion")
        if self.state in ("queued", "running", "cancel_requested") and (self.error_code is not None or self.acknowledged):
            raise ValueError("invalid active result")
        if self.reply is not None and self.action is not None:
            raise ValueError("ambiguous result")
        return self


class BrokerHealth(ContractModel):
    ready: StrictBool
    active: Annotated[int, Field(strict=True, ge=0, le=1)]
    queued: Annotated[int, Field(strict=True, ge=0, le=4)]
    model: Literal["claude-sonnet-5-5"]
    protocol_version: Literal[1] = 1


def context_fingerprint(context: PlannerContext, agent_step: AgentStepContext | None = None) -> str:
    context = PlannerContext.model_validate(context.model_dump(mode="python"))
    value = context.model_dump(mode="json")
    limit = MAX_INPUT_BYTES
    if agent_step is not None:
        agent_step = AgentStepContext.model_validate(agent_step.model_dump(mode="python"))
        if agent_step.context != context:
            raise BrokerError("invalid_request", 400)
        value = {"context": value, "agent_step": agent_step.model_dump(mode="json")}
        limit = MAX_AGENT_INPUT_BYTES
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    # Reserve the strict transport object's framing, not just context bytes.
    if len(raw) > limit - 64:
        raise BrokerError("invalid_request", 400)
    return hashlib.sha256(raw).hexdigest()


def _decode(raw: bytes):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid_constant)


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: Path, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.path))


class BrokerClient:
    def __init__(self, socket_path: Path, timeout_sec: float = 5):
        if not isinstance(socket_path, Path) or not socket_path.is_absolute() or not 0 < timeout_sec <= 30:
            raise ValueError("invalid trusted broker configuration")
        self.socket_path, self.timeout_sec = socket_path, timeout_sec

    def _request(self, method: str, path: str, payload=None):
        connection = _UnixConnection(self.socket_path, self.timeout_sec)
        try:
            raw = b"" if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                                        separators=(",", ":")).encode("utf-8")
            if len(raw) > MAX_AGENT_INPUT_BYTES:
                raise BrokerError("invalid_request", 400)
            connection.request(method, path, raw, {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read(MAX_STDOUT_BYTES + 1)
            if len(body) > MAX_STDOUT_BYTES:
                raise ValueError("response too large")
            value = _decode(body)
            if response.status not in (200, 202):
                codes = {400: "invalid_request", 401: "unauthorized", 404: "not_found", 409: "conflict",
                         429: "backpressure", 503: "ledger_unavailable"}
                raise BrokerError(codes.get(response.status, "unavailable"), response.status)
            return value
        except BrokerError:
            raise
        except (OSError, ValueError, TypeError, http.client.HTTPException, RecursionError):
            raise BrokerError("unavailable") from None
        finally:
            connection.close()

    @staticmethod
    def _id(turn_id: str) -> str:
        import re
        if not isinstance(turn_id, str) or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", turn_id) is None:
            raise BrokerError("invalid_request", 400)
        return turn_id

    def _result(self, method, turn_id, action="", payload=None):
        turn_id = self._id(turn_id)
        try:
            result = BrokerResult.model_validate(self._request(method, "/v1/turns/" + turn_id + action, payload))
            if result.turn_id != turn_id:
                raise ValueError("changed turn")
            return result
        except BrokerError:
            raise
        except ValueError:
            raise BrokerError("unavailable") from None

    def submit(self, turn_id: str, context: PlannerContext, *, agent_step: AgentStepContext | None = None) -> BrokerResult:
        fingerprint = context_fingerprint(context, agent_step)
        payload = {"context": context.model_dump(mode="json")}
        if agent_step is not None:
            payload["agent_step"] = agent_step.model_dump(mode="json")
        result = self._result("PUT", turn_id, payload=payload)
        if result.context_sha256 != fingerprint:
            raise BrokerError("unavailable")
        return result

    def poll(self, turn_id: str) -> BrokerResult:
        return self._result("GET", turn_id)

    def cancel(self, turn_id: str) -> BrokerResult:
        return self._result("POST", turn_id, "/cancel")

    def fence(self, turn_id: str, context_sha256: str) -> BrokerResult:
        import re
        if not isinstance(context_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", context_sha256) is None:
            raise BrokerError("invalid_request", 400)
        result = self._result("POST", turn_id, "/fence", {"context_sha256": context_sha256})
        if result.context_sha256 != context_sha256:
            raise BrokerError("unavailable")
        return result

    def acknowledge(self, turn_id: str) -> BrokerResult:
        return self._result("POST", turn_id, "/acknowledge")

    def health(self) -> BrokerHealth:
        try:
            return BrokerHealth.model_validate(self._request("GET", "/health"))
        except BrokerError:
            raise
        except ValueError:
            raise BrokerError("unavailable") from None
