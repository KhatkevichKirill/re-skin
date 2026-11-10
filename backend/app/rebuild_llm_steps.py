"""Step handler of ``rebuild_llm_calls`` — a dispatcher by ``kind``.

``rebuild_paid_call.STEP_HANDLERS["rebuild_llm_calls"]`` points here
(``app.rebuild_llm_steps.handle``). Every Gemini call of the rebuild feature
is a row of that one table (storyboard ``global`` / ``shot`` calls, the
per-attempt ``character_check``), but its inputs and what its answer feeds
differ by kind, so each kind registers its own handler below — a dotted path
``module.func`` called as ``func(row_id, gen, step, job_id)``, resolved in
whatever process runs the step (the RQ work-horse, or the reaper's enqueue
target), exactly like ``STEP_HANDLERS`` itself.

An unknown kind is logged at ERROR and left alone: nothing is sent, and the
paid-call reaper keeps re-driving it.
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

from .rebuild_models import RebuildLlmCall

log = logging.getLogger(__name__)

# kind -> dotted path of ``handler(row_id, gen, step, job_id)``.
KIND_HANDLERS: dict[str, str] = {
    # Storyboard calls (design §5.2): claim (proxy + frame + uploads, free) ->
    # send (ONE chat_completion_once).
    "global": "app.rebuild_storyboard.llm_step",
    "shot": "app.rebuild_storyboard.llm_step",
    # The automatic per-shot character check of a succeeded chunk attempt
    # (design §6.5, Q9): claim -> send (one chat_completion_once); the
    # fallback slot and the "unavailable" verdict are handled by the handler.
    "character_check": "app.pipeline_rebuild.character_check_step",
}


def _session_factory():
    from .db import SessionLocal

    return SessionLocal


def handle(row_id: str, gen: int, step: str, job_id: Optional[str]):
    """Dispatch one step of an LLM call row to its kind's handler."""
    sf = _session_factory()
    db = sf()
    try:
        row = db.get(RebuildLlmCall, row_id)
        kind = row.kind if row is not None else None
    finally:
        db.close()
    if kind is None:
        log.warning("rebuild llm step %s for %s g%s: no such row", step, row_id, gen)
        return None
    path = KIND_HANDLERS.get(kind)
    if not path:
        log.error("rebuild llm step %s for %s g%s: no handler for kind %r — nothing done",
                  step, row_id, gen, kind)
        return None
    mod_name, func_name = path.rsplit(".", 1)
    func = getattr(importlib.import_module(mod_name), func_name)
    return func(row_id, int(gen), step, job_id)
