"""How the approval card v2 presents one approval (AICP-376).

Design: ``docs/autopilot-operator-ux.md`` §4. Presentation only: nothing here
changes what an answer does. The card posts the same ``decision`` / ``option``
the v1 form posted; this module only decides how each choice is labelled and
which "no" control matches what the code does for the approval's ORIGIN (§4.3):

* core budget (``…#budget``): reject → consumed in reconcile → attempt stopped;
* core regeneration / overrun (``…#regeneration`` / ``…#overrun``): reject →
  consumed on the launch path → stopped;
* subtitle flow, any ask: reject → ``_rejection`` → Fail → attempt failed, the
  comment in the reason;
* localisation listening probe / final: reject records the cue and asks a
  corrective regeneration next (the final fails if nothing can be regenerated);
* localisation regen / fix / retry asks: reject, or the ``stop`` option, → Fail.
"""

from __future__ import annotations

from typing import Optional

from ..models_autopilot import utcnow
from . import links

CORE_SEP = "#"
STOP_OPTION = "stop"

KIND_HEADER = {
    "budget": "Budget",
    "regeneration": "Costs credits",
    "overrun": "Costs credits",
    "handoff": "Check this",
    "listening_probe": "Listen",
    "listening_final": "Listen",
}


def origin(name: str, kind: str, flow_id: Optional[str]) -> str:
    """``core`` | ``subtitle`` | ``localisation`` | ``flow`` (any other flow)."""
    if CORE_SEP in (name or ""):
        return "core"
    if flow_id == "subtitle_removal":
        return "subtitle"
    if flow_id == "localisation":
        return "localisation"
    return "flow"


def no_control(name: str, kind: str, flow_id: Optional[str], option_ids) -> dict:
    """The quieter "no" control of §4.3: what it posts and how it is worded.

    ``post`` is ``{"decision": "reject"}`` or ``{"decision": "approve",
    "option": "stop"}`` (the localisation stop option, which the flow fails on
    exactly like a reject)."""
    o = origin(name, kind, flow_id)
    reject = {"decision": "reject"}
    if o == "core" and kind == "budget":
        return {"label": "Stop the assistant…", "post": reject,
                "confirm": "Stop the assistant? This attempt ends and nothing more is spent.",
                "comment_prompt": "Why? (optional — goes into the stop reason)", "note": ""}
    if o == "core":
        return {"label": "Don't spend this — stop the assistant…", "post": reject,
                "confirm": "Stop the assistant instead of spending this? This attempt ends.",
                "comment_prompt": "Why? (optional — goes into the stop reason)", "note": ""}
    if o == "localisation" and kind in ("listening_probe", "listening_final"):
        note = "You approve the corrected regeneration and its cost next."
        if kind == "listening_final":
            note += " If nothing can be regenerated, the attempt ends."
        return {"label": "Request changes", "post": reject, "confirm": "",
                "comment_prompt": "What is wrong? (it goes into the prompt; empty = the same prompt)",
                "note": note}
    if o == "localisation" and STOP_OPTION in option_ids:
        return {"label": "End this attempt…", "post": {"decision": "approve", "option": STOP_OPTION},
                "confirm": "End this attempt? The assistant stops here (failed).",
                "comment_prompt": "Why? (optional)", "note": ""}
    if o in ("subtitle", "localisation"):
        return {"label": "End this attempt…", "post": reject,
                "confirm": "End this attempt? The assistant stops here (failed) with your comment as the reason.",
                "comment_prompt": "Why? (optional — becomes the reason)", "note": ""}
    return {"label": "Reject…", "post": reject, "confirm": "Reject this?",
            "comment_prompt": "Why? (optional)", "note": ""}


def _approve_label(kind: str, cost_credits: float) -> str:
    if kind == "budget":
        return f"Approve {cost_credits:.0f} credits"
    if kind in ("regeneration", "overrun"):
        return f"Spend ~{cost_credits:.0f} credits" if cost_credits else "Approve"
    if kind.startswith("listening"):
        return "Sounds right: approve"
    return "Looks right: approve"


def _waited(seconds: float) -> str:
    mins = max(0, int(seconds // 60))
    if mins < 60:
        return f"{mins} min"
    hours, mins = divmod(mins, 60)
    return f"{hours} h {mins:02d} min" if hours < 48 else f"{hours // 24} days"


def present(d: dict, *, flow_id: Optional[str], now=None) -> dict:
    """The card view of an ``approval_dict``: ``d`` plus ``card`` (a dict)."""
    from .. import spend

    now = now or utcnow()
    kind, name = d.get("kind") or "", d.get("name") or ""
    raw_opts = [o for o in (d.get("options") or []) if isinstance(o, dict)]
    ids = {o.get("id") for o in raw_opts}
    no = no_control(name, kind, flow_id, ids)
    stop_is_no = no["post"].get("option") == STOP_OPTION
    options = []
    for o in raw_opts:
        if stop_is_no and o.get("id") == STOP_OPTION:
            continue
        credits = (o.get("cost_mcredits") or 0) / 1000.0
        options.append({"id": o.get("id"), "label": o.get("label") or o.get("id"),
                        "consequence": o.get("consequence") or "",
                        "credits": credits, "usd": round(spend.usd(credits), 2) if credits else 0.0})
    cost = d.get("cost") or {}
    proposal = {k: v for k, v in (d.get("proposal") or {}).items() if k != links.TITLE_KEY}
    after = proposal.get("budget_after_credits")
    budget = proposal.get("approved_budget_credits")
    media = {"videos": [], "images": [], "audios": []}
    for art in d.get("artifacts") or []:
        media.setdefault({"video": "videos", "image": "images", "audio": "audios"}.get(
            art.get("kind"), "images"), []).append(art)
    preview = proposal.get("preview_url")
    created = d.get("created_at")
    try:
        from datetime import datetime

        waited = _waited((now - datetime.fromisoformat(created)).total_seconds()) if created else ""
    except (TypeError, ValueError):
        waited = ""
    card = {
        "origin": origin(name, kind, flow_id),
        "header": KIND_HEADER.get(kind, kind or "Question"),
        "waited": waited,
        "options": options,
        "approve_label": None if options else _approve_label(kind, float(cost.get("credits") or 0)),
        "no": no,
        "cost": cost if cost.get("credits") else None,
        "after": ({"credits": after, "budget": budget} if after is not None and budget is not None
                  else None),
        "budget": ({"lines": proposal.get("lines") or [], "total_credits": proposal.get("total_credits"),
                    "total_usd": proposal.get("total_usd"), "tolerance": proposal.get("tolerance")}
                   if kind == "budget" else None),
        "media": media,
        # Two videos side by side with one scrub bar: the result and what it is
        # compared with (§4.1); a single video plays alone.
        "pair": media["videos"][:2] if len(media["videos"]) >= 2 else None,
        "preview": preview if isinstance(preview, str) and preview.startswith("/api/v2/") else None,
        "details": {k: v for k, v in proposal.items()
                    if k not in ("preview_url", "lines") and not (kind == "budget" and k.startswith("total"))},
    }
    return dict(d, card=card)
