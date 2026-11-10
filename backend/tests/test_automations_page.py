"""Smoke test for the global Automations overview page (GET /v2/automations)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import AutomationLaunch

# Reuse the fixtures + helpers from the main automation test module.
from tests.test_automation import (  # noqa: F401
    SessionFactory,
    client,
    db,
    engine,
    _make_automation,
    _make_ready_project_with_run,
)


def test_automations_page_groups_active_and_previously(db, client):
    # 1) Active automation (enabled), no launches yet.
    active_run = _make_ready_project_with_run(db)
    active_run.name = "ACTIVE-TEMPLATE"
    db.commit()
    _make_automation(db, active_run.id, enabled=True, filter_persona="Student")

    # 2) Previously active: disabled, but has a launch in its ledger.
    prev_run = _make_ready_project_with_run(db)
    prev_run.name = "PREV-TEMPLATE"
    db.commit()
    prev_auto = _make_automation(db, prev_run.id, enabled=False)
    db.add(
        AutomationLaunch(
            automation_id=prev_auto.id,
            batch_id="b1",
            queue_id="Q-1",
            ref_url="http://x/1",
            launched_at=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc).replace(
                tzinfo=None
            ),
        )
    )
    db.commit()

    # 3) Disabled + never launched → omitted from both groups.
    idle_run = _make_ready_project_with_run(db)
    idle_run.name = "IDLE-TEMPLATE"
    db.commit()
    _make_automation(db, idle_run.id, enabled=False)

    resp = client.get("/v2/automations")
    assert resp.status_code == 200
    body = resp.text

    assert "ACTIVE-TEMPLATE" in body      # active section
    assert "PREV-TEMPLATE" in body        # previously section
    assert "IDLE-TEMPLATE" not in body    # omitted
    assert "Active automations" in body
    assert "Previously active" in body
