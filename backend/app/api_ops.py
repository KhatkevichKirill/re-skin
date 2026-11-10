"""Ops endpoints for the deploy admission gate (AICP-350).

* ``GET /api/v2/ops/admission`` — read-only drain status
  (``app/ops_status.py``): ``{admitting, marker, banner, active, parked,
  drained, build, rq}``. Not under the nginx ``kie_spend`` limiter.
* ``GET /v2/ops-gate-banner`` — the HTMX fragment the shared site header polls:
  empty while admitting; "Deploy in progress since …" while paused, red after
  ``OPS_GATE_WARN_AFTER_MIN`` with the ready-to-run ``force-end`` command.
"""

from __future__ import annotations

from html import escape

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from . import ops_gate, ops_status
from .db import get_db

router = APIRouter(tags=["ops"])
web_router = APIRouter(include_in_schema=False)


@router.get("/ops/admission")
def get_admission(db: Session = Depends(get_db)) -> dict:
    return ops_status.admission_status(db)


def render_banner(b: dict | None) -> str:
    if not b:
        return ""
    red = b.get("level") == "red"
    style = ("background:#fde2e1;border:1px solid #d93025;color:#8a1c14" if red
             else "background:#fff4ce;border:1px solid #e0b300;color:#5c4400")
    parts = [
        f'<div class="ops-gate-banner" role="status" style="{style};'
        'padding:6px 12px;margin:0;font-size:13px">',
        f"<strong>{escape(b.get('text') or 'Deploy in progress')}</strong>",
        " &mdash; new work is paused; running work continues.",
    ]
    if red:
        mins = b.get("minutes")
        parts.append(
            f" Paused for {escape(str(mins))} min (over {escape(str(b.get('warn_after_min')))}"
            " min). If the deploy that paused it is gone, end the pause from /root/re-skin:"
            f"<br><code style=\"user-select:all\">{escape(b.get('command') or '')}</code>"
            f"<br><em>{escape(b.get('command_note') or '')}</em>"
        )
    parts.append("</div>")
    return "".join(parts)


@web_router.get("/ops-gate-banner", response_class=HTMLResponse)
def ops_gate_banner(db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        gate = ops_gate.read_gate(db)
    except Exception:  # table not migrated yet / DB hiccup: show nothing
        return HTMLResponse("")
    return HTMLResponse(render_banner(ops_gate.banner(gate)),
                        headers={"Cache-Control": "no-store"})
