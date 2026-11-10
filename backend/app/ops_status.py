"""Drain status: what a restart would kill (``active``) vs what waits durably
for the operator (``parked``) — AICP-350, docs/assistant-drain-check-design.md §2.

``active`` must reach 0 before a deploy; ``parked`` is reported, never waited
on. The same split is in ``deploy/drain-check.sql`` (the deploy script reads
that file, so it works while the API is down or on an old image); keep the two
in step — ``tests/test_ops_status.py`` compares them when the repo root is
mounted.

RQ queue/started counts are a diagnostic only (restart-sensitive job types,
never the self-rescheduling ticks), not part of the zero rule; when Redis
cannot be read they say "unknown", never 0.

CLI (inside the api container): ``python -m app.ops_status`` prints the JSON
of ``GET /api/v2/ops/admission``; ``python -m app.ops_status fingerprint``
prints the code fingerprint of the checkout/image it runs from.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from . import ops_gate

log = logging.getLogger(__name__)

# (kind, "FROM ... WHERE ...") — copied verbatim into deploy/drain-check.sql as
# "SELECT '<state>', '<kind>', count(*) <fragment>".
ACTIVE: tuple[tuple[str, str], ...] = (
    ("projects", "FROM video_projects WHERE status = 'analyzing' OR transcript_status IN ('pending', 'running')"),
    ("runs", "FROM runs WHERE status IN ('queued', 'processing', 'stitching', 'delivering', 'preprocessing')"),
    ("range_attempts", "FROM run_segment_range_attempts WHERE status IN ('queued', 'generating', 'stitching', 'delivering')"),
    ("segment_images", "FROM run_segment_image_jobs WHERE status IN ('queued', 'generating')"),
    ("tail_edits", "FROM run_tail_edits WHERE status IN ('queued', 'processing', 'delivering')"),
    ("captions", "FROM run_captions WHERE status IN ('queued', 'transcribing', 'burning', 'delivering')"),
    ("overlays", "FROM overlay_jobs WHERE status IN ('queued', 'processing', 'delivering')"),
    ("insert_merges", "FROM insert_merges WHERE status IN ('queued', 'compositing', 'delivering')"),
    ("audio_patches", "FROM run_audio_patches WHERE status IN ('queued', 'processing', 'delivering')"),
    ("reframe_jobs", "FROM reframe_jobs WHERE status = 'preparing' OR qa_status = 'running'"),
    ("reframe_renders", "FROM reframe_renders WHERE status IN ('queued', 'processing', 'delivering')"),
    ("segment_fills", "FROM run_segment_fill_jobs WHERE status IN ('preparing', 'compositing', 'composited')"),
    ("segment_fill_batches", "FROM run_segment_fill_batches WHERE status IN ('active', 'promoting')"),
    ("provider_tasks", "FROM provider_tasks WHERE status IN ('accepted_unrecorded', 'cancelling', 'closing', 'fetching', 'queued', 'submit_started', 'submitted')"),
    ("autopilot_launching", "FROM autopilot_effects WHERE kind = 'action' AND state = 'launching'"),
    ("translate_jobs", "FROM localisation_prompt_jobs WHERE status IN ('queued', 'running')"),
    ("rebuild_storyboards", "FROM project_storyboards WHERE status IN ('queued', 'running')"),
    ("rebuild_keyframes", "FROM rebuild_keyframes WHERE status IN ('queued', 'claimed', 'sending', 'submitted', 'fetching')"),
    ("rebuild_chunk_attempts", "FROM rebuild_chunk_attempts WHERE status IN ('queued', 'claimed', 'sending', 'submitted', 'fetching')"),
    ("rebuild_llm_calls", "FROM rebuild_llm_calls WHERE status IN ('queued', 'claimed', 'sending', 'submitted', 'fetching')"),
    # Drive re-link and caption-placement scan jobs have no running status:
    # they hold the gate share lock for their whole run instead
    # (ops_gate.job_barrier), so begin.sql itself waits for them.
    # AICP-261 run chat. An `unknown` paid intent would be `parked` only once its
    # sender is proven stopped (design §2 supervisor acknowledgement, §9 open
    # item 1) — not implemented, so it stays active: resolve it before a deploy.
    ("assistant_turns", "FROM assistant_turns WHERE status IN ('queued', 'running', 'cancel_requested')"),
    ("assistant_jobs", "FROM assistant_jobs WHERE status IN ('queued', 'running', 'cancel_requested')"),
    ("assistant_paid_intents", "FROM assistant_paid_call_intents WHERE status IN ('submitting', 'unknown')"),
)

PARKED: tuple[tuple[str, str], ...] = (
    ("provider_tasks", "FROM provider_tasks WHERE status IN ('needs_reconciliation', 'submit_unknown')"),
)

# Restart-sensitive RQ queues (diagnostic). The ticks reschedule themselves and
# are never counted.
RQ_QUEUES = ("analyze", "process", "overlay", "localise", "autopilot_compute", "brand_asr", "fal")
_TICK_MARKERS = ("_tick", "knowledge", "fal_ticker")


def drain_check_fragment(fragment: str) -> str:
    """The deploy/drain-check.sql spelling of an ACTIVE/PARKED fragment
    (identical today; tests/test_ops_gate.py compares them through this)."""
    return fragment


def _count(db: Session, fragment: str) -> Optional[int]:
    try:
        with db.begin_nested():
            return int(db.execute(text(f"SELECT count(*) {fragment}")).scalar_one())
    except Exception as exc:  # a table missing on an older schema
        log.warning("ops_status: count failed (%s): %s", fragment.split(" WHERE")[0], exc)
        return None


def counts(db: Session) -> dict:
    active = {kind: _count(db, frag) for kind, frag in ACTIVE}
    parked = {kind: _count(db, frag) for kind, frag in PARKED}
    known = [v for v in active.values() if v is not None]
    total = sum(known) if len(known) == len(active) else None
    return {
        "active": {"total": total, "by_kind": {k: v for k, v in active.items() if v}},
        "active_unknown": sorted(k for k, v in active.items() if v is None),
        "parked": {"total": sum(v or 0 for v in parked.values()),
                   "by_kind": {k: v for k, v in parked.items() if v}},
    }


def rq_diagnostic() -> dict:
    """Queued + started job counts per restart-sensitive queue, ticks excluded.
    ``{"status": "unknown"}`` when Redis cannot be read."""
    try:
        from rq.job import Job
        from rq.registry import StartedJobRegistry

        from .tasks import _get_queue, _get_redis

        conn = _get_redis()
        out: dict = {}
        for name in RQ_QUEUES:
            q = _get_queue(name)
            ids = list(q.get_job_ids()) + list(StartedJobRegistry(name, connection=conn).get_job_ids())
            n = 0
            for jid in ids:
                job = Job.fetch(jid, connection=conn) if jid else None
                func = (getattr(job, "func_name", "") or "") if job else ""
                if any(m in func for m in _TICK_MARKERS):
                    continue
                n += 1
            if n:
                out[name] = n
        return {"status": "ok", "by_queue": out}
    except Exception as exc:
        log.warning("ops_status: RQ diagnostic unavailable: %s", exc)
        return {"status": "unknown"}


def admission_status(db: Session, *, include_rq: bool = True) -> dict:
    """The body of ``GET /api/v2/ops/admission`` (read-only)."""
    gate = ops_gate.read_gate(db)
    c = counts(db)
    body = {
        "admitting": gate.admitting,
        "marker": {
            "paused": gate.paused,
            "generation": gate.generation,
            "owner": gate.owner,
            "reason": gate.reason,
            "paused_at": gate.paused_at.isoformat() if gate.paused_at else None,
        },
        "banner": ops_gate.banner(gate),
        "active": c["active"],
        "parked": c["parked"],
        "drained": c["active"]["total"] == 0,
        "build": ops_gate.build_info(),
    }
    if c["active_unknown"]:
        body["active_unknown"] = c["active_unknown"]
    if include_rq:
        body["rq"] = rq_diagnostic()
    return body


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - CLI
    import json
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["fingerprint"]:
        print(ops_gate.code_fingerprint())
        return 0
    from .db import SessionLocal

    db = SessionLocal()
    try:
        print(json.dumps(admission_status(db), indent=2, default=str))
    finally:
        db.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
