"""AICP-376: the approval card v2 (app/autopilot/card.py + the approval_form macro)."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import timedelta

os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mkstemp(suffix='.db')[1]}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.autopilot import card
from app.models_autopilot import utcnow


def _render(a, url="/v2/projects/p/autopilot-card/answer", target="#autopilot-card"):
    from app.web_v2 import templates

    m = templates.env.get_template("partials/autopilot_macros.html").module
    return str(m.approval_form(a, url, target, "ap1"))


def _dict(**kw):
    d = {"id": "a1", "anchor": "ap-a1", "autopilot_id": "ap1", "name": "q", "kind": "handoff",
         "step_id": "clean", "step_generation": 1, "subject": "", "question": "Is it clean? Look.",
         "title": "Is it clean?", "proposal": {}, "options": None, "artifacts": [],
         "cost": {"credits": 0.0, "usd": 0.0}, "comments": [],
         "created_at": (utcnow() - timedelta(minutes=75)).isoformat()}
    d.update(kw)
    return d


def _posts(html):
    return [json.loads(v.replace("&#34;", '"')) for v in re.findall(r"hx-vals='([^']*)'", html)]


# ---------------------------------------------------------------------------
# The "no" control by origin (§4.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kind,flow,ids,label,post,confirm", [
    ("launch#budget", "budget", "subtitle_removal", set(), "Stop the assistant…", {"decision": "reject"}, True),
    ("rerun#regeneration", "regeneration", "localisation", set(),
     "Don't spend this — stop the assistant…", {"decision": "reject"}, True),
    ("launch#overrun", "overrun", "subtitle_removal", set(),
     "Don't spend this — stop the assistant…", {"decision": "reject"}, True),
    ("blocked:f1", "regeneration", "subtitle_removal", {"void", "retry"}, "End this attempt…",
     {"decision": "reject"}, True),
    ("segmentation:ab", "handoff", "subtitle_removal", {"apply", "as_is"}, "End this attempt…",
     {"decision": "reject"}, True),
    ("listen:en", "listening_probe", "localisation", set(), "Request changes", {"decision": "reject"}, False),
    ("listen:en:final", "listening_final", "localisation", set(), "Request changes",
     {"decision": "reject"}, False),
    ("fix:en", "regeneration", "localisation", {"rerun", "stop"}, "End this attempt…",
     {"decision": "approve", "option": "stop"}, True),
    ("size", "handoff", "localisation", {"deliver"}, "End this attempt…", {"decision": "reject"}, True),
    ("x", "handoff", "stub", set(), "Reject…", {"decision": "reject"}, True),
])
def test_no_control_matches_what_the_code_does(name, kind, flow, ids, label, post, confirm):
    no = card.no_control(name, kind, flow, ids)
    assert no["label"] == label and no["post"] == post and bool(no["confirm"]) is confirm


def test_listening_final_says_the_attempt_may_end():
    probe = card.no_control("listen:en", "listening_probe", "localisation", set())
    final = card.no_control("listen:en", "listening_final", "localisation", set())
    assert "regeneration and its cost next" in probe["note"] and "attempt ends" not in probe["note"]
    assert "If nothing can be regenerated, the attempt ends." in final["note"]


def test_localisation_stop_option_moves_to_the_no_control():
    d = card.present(_dict(name="fix:en", kind="regeneration", options=[
        {"id": "rerun", "label": "Regenerate them", "cost_mcredits": 120000,
         "consequence": "resubmits those segments"},
        {"id": "stop", "label": "Stop the autopilot", "cost_mcredits": 0}]), flow_id="localisation")
    assert [o["id"] for o in d["card"]["options"]] == ["rerun"]
    html = _render(d)
    posts = _posts(html)
    assert {"decision": "approve", "option": "rerun"} in posts
    assert {"decision": "approve", "option": "stop"} in posts          # the no control
    assert "End this attempt…" in html and "Stop the autopilot" not in html
    assert "resubmits those segments" in html and "~120 credits" in html
    assert "estimated when asked" in html


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_an_old_approval_renders_with_the_fallback_title_and_bare_options():
    # created before AICP-376: no _title, options without consequence
    d = card.present(_dict(name="blocked:f1", kind="regeneration", title="Segment 1 was refused.",
                           question="Segment 1 was refused. VOID or retry?",
                           options=[{"id": "void", "label": "VOID fill", "cost_mcredits": 0},
                                    {"id": "retry", "label": "Retry", "cost_mcredits": 0}]),
                     flow_id="subtitle_removal")
    html = _render(d)
    assert 'id="ap-a1"' in html and "Segment 1 was refused." in html
    assert "Costs credits" in html and "waiting 1 h 15 min" in html
    assert {"decision": "approve", "option": "void"} in _posts(html)
    assert {"decision": "reject"} in _posts(html) and "hx-confirm=" in html
    assert "<details" in html
    # the full question stays visible: warnings in it must not hide under Details
    q = html.index("Segment 1 was refused. VOID or retry?")
    assert q < html.index("<details")


def test_budget_card_shows_the_table_and_approve_n_credits():
    d = card.present(_dict(name="launch#budget", kind="budget", cost={"credits": 260.0, "usd": 2.6},
                           proposal={"lines": [{"label": "already committed", "credits": 10},
                                               {"label": "probe", "credits": 250}],
                                     "total_credits": 260.0, "total_usd": 2.6, "tolerance": 0.1,
                                     "_title": "Approve the budget: 260 credits"}),
                     flow_id="subtitle_removal")
    html = _render(d)
    assert "Approve 260 credits" in html and "Stop the assistant…" in html
    assert "already committed" in html and "+10 % tolerance" in html
    assert "_title" not in html                                    # reserved key not shown


def test_core_regeneration_shows_budget_after_as_an_estimate():
    d = card.present(_dict(name="rest#regeneration", kind="regeneration", cost={"credits": 52.0, "usd": 0.52},
                           proposal={"budget_after_credits": 312.0, "approved_budget_credits": 260.0}),
                     flow_id="stub")
    html = _render(d)
    assert "Spend ~52 credits" in html and "after this: 312 of 260 credits" in html
    assert "estimated when asked" in html


def test_two_videos_share_one_scrub_bar_and_text_is_escaped():
    d = card.present(_dict(title="<b>x</b>", artifacts=[
        {"label": "result", "kind": "video", "url": "/api/v2/a/1"},
        {"label": "source", "kind": "video", "url": "/api/v2/a/2"},
        {"label": "sheet", "kind": "image", "url": "/api/v2/a/3"}]), flow_id="subtitle_removal")
    html = _render(d)
    assert "data-ap-pair" in html and html.count("<video") == 2 and "data-ap-scrub" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html and "<b>x</b>" not in html
    assert 'src="/api/v2/a/3"' in html


def test_card_scripts_are_on_both_cards():
    from app.web_v2 import templates

    for name in ("partials/autopilot_card.html", "partials/autopilot_run_card.html"):
        src = templates.env.loader.get_source(templates.env, name)[0]
        assert "m.media_script(" in src and "m.fragment_script(" in src


def test_answer_buttons_share_one_fieldset_that_disables_while_posting():
    d = card.present(_dict(name="blocked:f1", kind="regeneration", options=[
        {"id": "void", "label": "VOID", "cost_mcredits": 0}, {"id": "retry", "label": "Retry", "cost_mcredits": 0}]),
        flow_id="subtitle_removal")
    html = _render(d)
    fs = html[html.index("<fieldset"):html.index("</fieldset>")]
    assert fs.count("hx-post=") == 3 and fs.count('hx-disabled-elt="closest fieldset"') == 3
    assert "<textarea" not in fs                       # the comment is still sent
