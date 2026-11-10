"""AICP-350 route inventory (docs/assistant-drain-check-design.md §2): every
mutating FastAPI route is either gated (``dependencies=[GATED]`` or the keyed
``ADMISSION`` dependency) or listed in ``ops_gate.UNGATED`` with a reason, so a
new launch route cannot slip past the deploy admission gate."""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("DATABASE_URL", "sqlite:///" + tempfile.mkstemp(suffix="_ops_routes.db")[1])
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.routing import APIRoute

from app import ops_gate
from app.db import get_db

_READ_ONLY = {"GET", "HEAD", "OPTIONS"}


def _walk(routes, prefix=""):
    for r in routes:
        if isinstance(r, APIRoute):
            yield prefix + r.path, r
        elif hasattr(r, "original_router"):  # FastAPI >= 0.13x lazy include
            yield from _walk(r.original_router.routes, prefix + r.include_context.prefix)
        elif getattr(r, "routes", None) and not getattr(r, "app", None):
            yield from _walk(r.routes, prefix + getattr(r, "path", ""))


def _mutating():
    from app.main import app

    out = [(p, r) for p, r in _walk(app.routes) if set(r.methods) - _READ_ONLY]
    assert len(out) > 100, "route walk found too few routes — FastAPI internals changed?"
    return out


def test_every_mutating_route_is_gated_or_ungated_with_a_reason():
    unmarked, both = [], []
    for path, route in _mutating():
        name = ops_gate.endpoint_name(route.endpoint)
        gated = ops_gate.is_gated_dependant(route.dependant)
        if gated and name in ops_gate.UNGATED:
            both.append(f"{sorted(route.methods)} {path} ({name})")
        if not gated and name not in ops_gate.UNGATED:
            unmarked.append(f"{sorted(route.methods)} {path} ({name})")
    assert not unmarked, (
        "mutating routes neither gated nor ungated — add dependencies=[GATED] "
        "(it admits new work) or an ops_gate.UNGATED entry with the reason:\n  "
        + "\n  ".join(unmarked))
    assert not both, "gated AND listed as ungated:\n  " + "\n  ".join(both)


def test_ungated_entries_are_not_stale_and_carry_a_reason():
    names = {ops_gate.endpoint_name(r.endpoint) for _, r in _mutating()}
    stale = sorted(set(ops_gate.UNGATED) - names)
    assert not stale, f"UNGATED lists endpoints that are no longer mutating routes: {stale}"
    assert all(isinstance(v, str) and len(v.strip()) > 10 for v in ops_gate.UNGATED.values())


def _calls(dependant):
    stack, out = [dependant], []
    while stack:
        d = stack.pop()
        if getattr(d, "call", None) is not None:
            out.append(d.call)
        stack.extend(d.dependencies or [])
    return out


def test_gated_routes_lock_the_gate_in_the_request_session():
    """The gate share lock only fences the admission if it is taken on the SAME
    session the endpoint commits: every gated endpoint's own `db` must come
    from app.db.get_db (FastAPI caches it per request, and require_admitting
    depends on it)."""
    bad = []
    for path, route in _mutating():
        if not ops_gate.is_gated_dependant(route.dependant):
            continue
        db_params = [d for d in route.dependant.dependencies if d.name == "db"]
        if not db_params or db_params[0].call is not get_db:
            bad.append(f"{path} ({ops_gate.endpoint_name(route.endpoint)})")
    assert not bad, "gated routes whose `db` is not app.db.get_db:\n  " + "\n  ".join(bad)


def test_known_admission_routes_are_gated():
    """The design's named members (§2) — a regression guard on the inventory."""
    must = {
        ("POST", "/api/v2/projects"), ("POST", "/api/v2/projects/{pid}/analyze"),
        ("POST", "/api/v2/projects/{pid}/transcribe"),
        ("POST", "/api/v2/projects/{pid}/localisation-prompt"),
        ("POST", "/api/v2/projects/{pid}/runs"), ("POST", "/api/v2/runs/{rid}/copy"),
        ("POST", "/api/v2/runs/{rid}/copy-batch"), ("POST", "/api/v2/runs/{rid}/retry"),
        ("POST", "/api/v2/runs/{rid}/approve-preprocess"),
        ("POST", "/api/v2/runs/{rid}/regenerate-preprocess"),
        ("POST", "/api/v2/runs/{rid}/regenerate-preprocess-ref/{index}"),
        ("POST", "/api/v2/runs/{rid}/segments/{rsid}/rerun"),
        ("POST", "/api/v2/runs/{rid}/segments/rerun"),
        ("POST", "/api/v2/runs/{rid}/segments/{rsid}/range-rerun"),
        ("POST", "/api/v2/runs/{rid}/segments/{rsid}/regenerate-image"),
        ("POST", "/api/v2/runs/{run_id}/captions"), ("POST", "/api/v2/captions/{caption_id}/retry"),
        ("POST", "/api/v2/runs/{run_id}/tail-edits"), ("POST", "/api/v2/tail-edits/{edit_id}/retry"),
        ("POST", "/api/v2/overlays/{job_id}/submit"), ("POST", "/api/v2/runs/{run_id}/merge"),
        ("POST", "/api/v2/runs/{rid}/audio-patches"),
        ("POST", "/api/v2/projects/{pid}/reframe-jobs"), ("POST", "/api/v2/reframe-jobs/{jid}/renders"),
        ("POST", "/api/v2/reframe-renders/{rid}/retry"),
        ("POST", "/api/v2/assistant/runs/{run_id}/turns"),
        ("POST", "/api/v2/runs/{rid}/segments/fill/{fid}/start"),
        ("POST", "/api/v2/runs/{rid}/segments/fill/batch/start"),
    }
    gated = {(m, p) for p, r in _mutating() for m in r.methods
             if ops_gate.is_gated_dependant(r.dependant)}
    assert not must - gated, sorted(must - gated)
