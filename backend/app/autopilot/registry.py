"""Registries of flows, action handlers, compute functions and judgement kinds.

Production flows live in ``app.autopilot.flows`` and register themselves on
import (:func:`load_flows`). At the merge of the core none is registered, so
no assistant project can be created (docs/autopilot-design.md §2.1).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_flows: dict[tuple[str, int], Any] = {}
_handlers: dict[tuple[str, int], Any] = {}
_compute: dict[tuple[str, int], Callable] = {}
# AICP-389: the RQ queue a compute function runs on. Absent = the run
# workers' ``autopilot_compute`` queue; the brand check runs on ``brand_asr``
# (the brand-asr container, the only one with the local word-ASR stack).
_compute_queues: dict[tuple[str, int], str] = {}
DEFAULT_COMPUTE_QUEUE = "autopilot_compute"
_judgements: dict[tuple[str, int], "JudgementKind"] = {}
_flows_loaded = False


@dataclass(frozen=True)
class JudgementKind:
    kind: str
    version: int
    input_model: Any
    output_model: Any


def register(flow) -> None:
    """Register a flow version (``Flow`` protocol, contract.py)."""
    for attr in ("id", "version", "label", "project_types", "steps", "validate_params"):
        if not hasattr(flow, attr):
            raise TypeError(f"flow {flow!r} lacks {attr}")
    key = (flow.id, int(flow.version))
    _flows[key] = flow
    log.info("autopilot: registered flow %s v%d", *key)


def unregister(flow_id: str, version: int) -> None:
    _flows.pop((flow_id, int(version)), None)


def get_flow(flow_id: str, version: int):
    load_flows()
    return _flows.get((flow_id, int(version)))


def flow_for_project_type(project_type: str):
    """The newest registered flow version that runs *project_type*, or None."""
    load_flows()
    candidates = [f for f in _flows.values() if project_type in f.project_types]
    if not candidates:
        return None
    return max(candidates, key=lambda f: (f.id, int(f.version)))


def project_types_with_flow() -> set[str]:
    load_flows()
    return {t for f in _flows.values() for t in f.project_types}


def register_handler(handler) -> None:
    _handlers[(handler.name, int(handler.version))] = handler


def get_handler(name: str, version: Optional[int] = None):
    load_flows()  # flows register their handlers/compute/judgement kinds on import
    if version is None:
        versions = [v for (n, v) in _handlers if n == name]
        if not versions:
            return None
        version = max(versions)
    return _handlers.get((name, int(version)))


def register_compute(name: str, func: Callable, version: int = 1,
                     queue: Optional[str] = None) -> None:
    """*func(args: dict, out_dir: str) -> dict* — free, pure, streams frames."""
    _compute[(name, int(version))] = func
    if queue:
        _compute_queues[(name, int(version))] = queue
    else:
        _compute_queues.pop((name, int(version)), None)


def compute_queue(name: str, version: Optional[int] = None) -> str:
    """The RQ queue of a registered compute function (default ``autopilot_compute``)."""
    load_flows()
    if version is None:
        version = compute_version(name)
    if version is None:
        return DEFAULT_COMPUTE_QUEUE
    return _compute_queues.get((name, int(version)), DEFAULT_COMPUTE_QUEUE)


def get_compute(name: str, version: Optional[int] = None) -> Optional[Callable]:
    load_flows()  # flows register their handlers/compute/judgement kinds on import
    if version is None:
        versions = [v for (n, v) in _compute if n == name]
        if not versions:
            return None
        version = max(versions)
    return _compute.get((name, int(version)))


def compute_version(name: str) -> Optional[int]:
    load_flows()  # flows register their handlers/compute/judgement kinds on import
    versions = [v for (n, v) in _compute if n == name]
    return max(versions) if versions else None


def register_judgement(kind: str, input_model, output_model, version: int = 1) -> None:
    _judgements[(kind, int(version))] = JudgementKind(kind, int(version), input_model, output_model)


def get_judgement(kind: str, version: Optional[int] = None) -> Optional[JudgementKind]:
    load_flows()  # flows register their handlers/compute/judgement kinds on import
    if version is None:
        versions = [v for (k, v) in _judgements if k == kind]
        if not versions:
            return None
        version = max(versions)
    return _judgements.get((kind, int(version)))


def load_flows() -> None:
    """Import the production flow package once (flows register on import)."""
    global _flows_loaded
    if _flows_loaded:
        return
    _flows_loaded = True
    try:
        importlib.import_module("app.autopilot.flows")
    except Exception:  # a broken flow must not take the API down
        log.exception("autopilot: loading app.autopilot.flows failed")
