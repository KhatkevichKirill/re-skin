"""Judgement port (docs/autopilot-design.md §8).

The port is defined and the core handles judgement effects durably
(``prepared`` → ``sending`` → ``submitted`` → ``succeeded`` / ``failed`` /
``unknown``), but production ships :class:`DisabledPort` (operator decision Q2,
2026-10-31): no broker change in this wave and no billed fallback. Tests use
:class:`FakePort`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class JudgementState:
    state: str  # unknown_to_runtime | running | succeeded | failed | fenced
    result: Optional[dict] = None
    error: Optional[str] = None


class JudgementPort(Protocol):
    def available(self) -> bool: ...
    def submit(self, invocation_id: str, kind: str, payload: dict,
               payload_sha256: str) -> JudgementState: ...
    def poll(self, invocation_id: str) -> JudgementState: ...
    def fence(self, invocation_id: str, payload_sha256: str) -> JudgementState: ...


class DisabledPort:
    """Production port in this wave: never available."""

    def available(self) -> bool:
        return False

    def submit(self, invocation_id, kind, payload, payload_sha256):
        raise RuntimeError("the autopilot judgement runtime is disabled")

    def poll(self, invocation_id):
        return JudgementState("unknown_to_runtime")

    def fence(self, invocation_id, payload_sha256):
        return JudgementState("fenced")


@dataclass
class FakePort:
    """Idempotent by (invocation_id, sha) like the broker's PUT /v1/turns/{id}."""

    up: bool = True
    answers: dict = field(default_factory=dict)  # kind -> result dict
    submitted: dict = field(default_factory=dict)  # id -> (kind, sha)
    submit_calls: int = 0
    lose_next_response: bool = False

    def available(self) -> bool:
        return self.up

    def submit(self, invocation_id, kind, payload, payload_sha256):
        self.submit_calls += 1
        prev = self.submitted.get(invocation_id)
        if prev is not None and prev[1] != payload_sha256:
            raise RuntimeError("conflict: same invocation id, different payload")
        self.submitted[invocation_id] = (kind, payload_sha256)
        if self.lose_next_response:
            self.lose_next_response = False
            raise ConnectionError("lost response")
        return JudgementState("running")

    def poll(self, invocation_id):
        if invocation_id not in self.submitted:
            return JudgementState("unknown_to_runtime")
        kind, _ = self.submitted[invocation_id]
        if kind in self.answers:
            return JudgementState("succeeded", result=self.answers[kind])
        return JudgementState("running")

    def fence(self, invocation_id, payload_sha256):
        return JudgementState("fenced")


_port: JudgementPort = DisabledPort()


def get_port() -> JudgementPort:
    return _port


def set_port(port: JudgementPort) -> None:
    """Tests only."""
    global _port
    _port = port
