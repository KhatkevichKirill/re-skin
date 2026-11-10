"""
Tests for run automation (app/automation.py + the /runs/{rid}/automation API).

Strategy mirrors test_api_v2.py: temp SQLite DB, no Redis — run_tick returns
the run ids to enqueue instead of touching RQ, and the SheetsClient / webhook
POST are replaced with in-memory fakes.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

# Must set DATABASE_URL and DATA_DIR before any app import
_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

import app.automation as automation_mod
from app.automation import (
    QueueRow,
    match_rows,
    parse_queue_rows,
    run_tick,
)
from app.db import Base, get_db
from app.models import (
    AutomationLaunch,
    Run,
    RunAutomation,
    RunSegment,
    RunSegmentRangeAttempt,
    SegmentDef,
    VideoProject,
)
from app.state_machine import ProjectStatus, RangeAttemptStatus, RunStatus, SegmentStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        f"sqlite:///{_db_path}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(eng, "connect")
    def set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="module")
def SessionFactory(engine):
    return sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )


@pytest.fixture()
def db(SessionFactory):
    session = SessionFactory()
    yield session
    # Hard-clean automation tables between tests (module-scoped schema).
    session.rollback()
    for table in (
        AutomationLaunch, RunAutomation, RunSegmentRangeAttempt,
        RunSegment, SegmentDef, Run, VideoProject,
    ):
        for row in session.execute(select(table)).scalars().all():
            session.delete(row)
    session.commit()
    session.close()


def _make_ready_project_with_run(db) -> Run:
    project = VideoProject(
        source_type="upload", source_ref="test.mp4", status=ProjectStatus.ready
    )
    db.add(project)
    db.commit()
    run = Run(
        project_id=project.id,
        name="template",
        prompt="swap character",
        resolution="720p",
        status=RunStatus.done,
        reference_image_urls=[],
    )
    db.add(run)
    db.commit()
    return run


def _make_automation(db, run_id: str, **kwargs) -> RunAutomation:
    defaults = dict(run_id=run_id, enabled=True, plan="close-up", daily_count=5)
    defaults.update(kwargs)
    automation = RunAutomation(**defaults)
    db.add(automation)
    db.commit()
    return automation


def _sheet_row(i: int, persona="Student", age="20–22", gender="M", eth="East Asian"):
    """Raw sheet row with Queue ID Q-<i>, close-up in R (17), midplan in Y (24)."""
    row = [""] * 25
    row[0] = f"Q-{i:05d}"
    row[2] = persona
    row[3] = age
    row[4] = gender
    row[5] = eth
    row[17] = f"https://drive.google.com/uc?id=closeup{i}"
    row[24] = f"https://drive.google.com/uc?id=midplan{i}"
    return row


class FakeSheets:
    def __init__(self, rows):
        self.rows = rows
        self.read_calls = 0

    def read_rows(self, spreadsheet_id, tab, last_column="Z"):
        self.read_calls += 1
        return self.rows


class FailingSheets:
    """Always raises SheetsError; counts read attempts."""

    def __init__(self):
        self.read_calls = 0

    def read_rows(self, spreadsheet_id, tab, last_column="Z"):
        self.read_calls += 1
        from app.sheets_client import SheetsError

        raise SheetsError("boom")


@pytest.fixture(autouse=True)
def _webhook_policy(monkeypatch):
    """Tests use https://hook.test/...; allow it (AICP-156: no code default)."""
    from app.config import settings

    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_ALLOWED_HOSTS", "hook.test")
    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_URL", "https://hook.test/default")


@pytest.fixture()
def webhook_spy(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return True

    monkeypatch.setattr(automation_mod, "_post_webhook", fake_post)
    return calls


NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Parsing + filtering
# ---------------------------------------------------------------------------


def test_parse_queue_rows_skips_blank_ids():
    rows = parse_queue_rows([_sheet_row(1), [""], ["  "], _sheet_row(2)])
    assert [r.queue_id for r in rows] == ["Q-00001", "Q-00002"]
    assert rows[0].closeup_url.endswith("closeup1")
    assert rows[0].midplan_url.endswith("midplan1")


def test_parse_queue_rows_tolerates_short_rows():
    rows = parse_queue_rows([["Q-1", "x", "Student"]])
    assert rows[0].persona == "Student"
    assert rows[0].closeup_url == ""
    assert rows[0].midplan_url == ""


def test_match_rows_filters_case_insensitively(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(
        db, run.id,
        filter_persona="student", filter_gender="m", filter_ethnicity="east asian",
    )
    rows = parse_queue_rows(
        [_sheet_row(1), _sheet_row(2, persona="Parent"), _sheet_row(3, gender="F")]
    )
    matched = match_rows(rows, automation)
    assert [r.queue_id for r in matched] == ["Q-00001"]


def test_match_rows_empty_filter_means_any(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(db, run.id, filter_age="  ")
    rows = parse_queue_rows([_sheet_row(1), _sheet_row(2, age="30-35")])
    assert len(match_rows(rows, automation)) == 2


def test_match_rows_comma_separated_values_match_any(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(
        db, run.id, filter_ethnicity="East Asian, latino ,South Asian"
    )
    rows = parse_queue_rows([
        _sheet_row(1, eth="East Asian"),
        _sheet_row(2, eth="Latino"),
        _sheet_row(3, eth="Black"),
        _sheet_row(4, eth="South Asian"),
    ])
    matched = {r.queue_id for r in match_rows(rows, automation)}
    assert matched == {"Q-00001", "Q-00002", "Q-00004"}


def test_match_rows_exclude_inverts_filter(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(
        db, run.id, filter_ethnicity="East Asian,Latino", exclude_ethnicity=True
    )
    rows = parse_queue_rows([
        _sheet_row(1, eth="East Asian"),
        _sheet_row(2, eth="Latino"),
        _sheet_row(3, eth="Black"),
    ])
    matched = {r.queue_id for r in match_rows(rows, automation)}
    assert matched == {"Q-00003"}


def test_match_rows_exclude_with_empty_filter_matches_all(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(db, run.id, filter_persona="", exclude_persona=True)
    rows = parse_queue_rows([_sheet_row(1), _sheet_row(2, persona="Parent")])
    assert len(match_rows(rows, automation)) == 2


def test_match_rows_exclude_and_include_combine(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(
        db, run.id,
        filter_gender="M",
        filter_persona="Student,Parent", exclude_persona=True,
    )
    rows = parse_queue_rows([
        _sheet_row(1, persona="Student", gender="M"),
        _sheet_row(2, persona="Gamer", gender="M"),
        _sheet_row(3, persona="Gamer", gender="F"),
    ])
    matched = {r.queue_id for r in match_rows(rows, automation)}
    assert matched == {"Q-00002"}


def test_age_filter_single_number_matches_containing_ranges(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(db, run.id, filter_age="18")
    rows = parse_queue_rows([
        _sheet_row(1, age="18–22"),
        _sheet_row(2, age="20–26"),
        _sheet_row(3, age="18–30"),
        _sheet_row(4, age="26–30 (mature student)"),
        _sheet_row(5, age="18"),
    ])
    matched = {r.queue_id for r in match_rows(rows, automation)}
    assert matched == {"Q-00001", "Q-00003", "Q-00005"}


def test_age_filter_range_matches_regardless_of_dash_style(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(db, run.id, filter_age="18-22")  # plain hyphen
    rows = parse_queue_rows([
        _sheet_row(1, age="18–22"),  # en-dash in the sheet
        _sheet_row(2, age="18–30"),
    ])
    matched = {r.queue_id for r in match_rows(rows, automation)}
    assert matched == {"Q-00001"}


def test_age_filter_non_numeric_falls_back_to_exact_match(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(db, run.id, filter_age="teen")
    rows = parse_queue_rows([_sheet_row(1, age="Teen"), _sheet_row(2, age="18–22")])
    matched = {r.queue_id for r in match_rows(rows, automation)}
    assert matched == {"Q-00001"}


def test_match_rows_requires_photo_url_for_plan(db):
    run = _make_ready_project_with_run(db)
    automation = _make_automation(db, run.id, plan="midplan")
    row = _sheet_row(1)
    row[24] = "Link to GDrive"  # placeholder text, not a URL
    assert match_rows(parse_queue_rows([row]), automation) == []
    # close-up plan still matches (column R holds a real URL)
    automation.plan = "close-up"
    assert len(match_rows(parse_queue_rows([row]), automation)) == 1


# ---------------------------------------------------------------------------
# Launch phase
# ---------------------------------------------------------------------------


def test_tick_launches_up_to_daily_count(db):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=3)
    sheets = FakeSheets([_sheet_row(i) for i in range(1, 10)])

    created = run_tick(db, sheets=sheets, now=NOW)

    assert len(created) == 3
    launches = db.execute(select(AutomationLaunch)).scalars().all()
    assert len(launches) == 3
    assert {l.queue_id for l in launches} == {"Q-00001", "Q-00002", "Q-00003"}
    assert len({l.batch_id for l in launches}) == 1
    # New runs are named after the sheet's Queue ID and carry its close-up URL.
    new_run = db.get(Run, created[0])
    assert new_run.name == "Q-00001"
    assert new_run.reference_image_urls == ["https://drive.google.com/uc?id=closeup1"]
    assert new_run.prompt == template.prompt
    assert new_run.resolution == template.resolution


def _make_preprocess_template(db) -> Run:
    """A template run that was built with GPT Image 2: char photo + 2 frames."""
    template = _make_ready_project_with_run(db)
    template.reference_image_urls = [
        "https://old.example/char.jpg",
        "https://old.example/frame1.jpg",
        "https://old.example/frame2.jpg",
    ]
    template.preprocess_enabled = True
    template.preprocess_mode = "per_image"
    template.preprocess_image_prompts = ["", "put her in this shot", "and this one"]
    template.preprocessed_reference_urls = ["a.png", "b.png", "c.png"]
    db.commit()
    return template


def test_tick_preprocess_refs_reruns_gpt_image_2_per_photo(db):
    """With preprocess_refs, each sheet photo becomes the character reference of
    a launch that re-runs the template's recipe — same frames, same prompts —
    and lands in `preprocessing`, not `queued`."""
    template = _make_preprocess_template(db)
    _make_automation(db, template.id, daily_count=2, preprocess_refs=True)

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1), _sheet_row(2)]), now=NOW)

    assert len(created) == 2
    for rid, i in zip(created, (1, 2)):
        run = db.get(Run, rid)
        assert run.reference_image_urls == [
            f"https://drive.google.com/uc?id=closeup{i}",
            "https://old.example/frame1.jpg",
            "https://old.example/frame2.jpg",
        ]
        assert run.preprocess_image_prompts == [
            "", "put her in this shot", "and this one",
        ]
        assert run.preprocess_enabled is True
        assert run.preprocess_auto_approve is True   # nobody is awake at 3am
        assert run.status == RunStatus.preprocessing
        # The template's generated images are its own face — not inherited.
        assert not run.preprocessed_reference_urls


def test_tick_without_preprocess_refs_is_unchanged(db):
    """Default automation still hands the raw sheet photo to Seedance."""
    template = _make_preprocess_template(db)
    _make_automation(db, template.id, daily_count=1)

    created = run_tick(db, sheets=FakeSheets([_sheet_row(3)]), now=NOW)

    run = db.get(Run, created[0])
    assert run.reference_image_urls == ["https://drive.google.com/uc?id=closeup3"]
    assert run.preprocess_enabled is False
    assert run.status == RunStatus.queued


def test_tick_preprocess_refs_on_a_plain_template_falls_back(db):
    """A template with no recipe to re-run launches raw instead of failing every
    row — the flag is refused at the API, so this only covers a template that
    changed underneath a saved automation."""
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=1, preprocess_refs=True)

    created = run_tick(db, sheets=FakeSheets([_sheet_row(4)]), now=NOW)

    assert len(created) == 1
    run = db.get(Run, created[0])
    assert run.preprocess_enabled is False
    assert run.status == RunStatus.queued


def test_tick_midplan_uses_column_y(db):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, plan="midplan", daily_count=1)
    created = run_tick(db, sheets=FakeSheets([_sheet_row(7)]), now=NOW)
    new_run = db.get(Run, created[0])
    assert new_run.reference_image_urls == ["https://drive.google.com/uc?id=midplan7"]


def test_tick_respects_batch_cap(db, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "AUTOMATION_MAX_PER_BATCH", 2)
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=50)
    created = run_tick(db, sheets=FakeSheets([_sheet_row(i) for i in range(1, 8)]), now=NOW)
    assert len(created) == 2


def test_tick_never_reuses_a_queue_row(db, webhook_spy):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=2)
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2), _sheet_row(3)])

    first = run_tick(db, sheets=sheets, now=NOW)
    assert len(first) == 2
    # Finish the first batch so the next tick may launch again.
    for rid in first:
        db.get(Run, rid).status = RunStatus.done
    db.commit()

    second = run_tick(db, sheets=sheets, now=NOW + timedelta(days=1))
    assert len(second) == 1  # only Q-00003 left
    used = {l.queue_id for l in db.execute(select(AutomationLaunch)).scalars().all()}
    assert used == {"Q-00001", "Q-00002", "Q-00003"}


def test_tick_skips_queue_ids_already_used_as_run_names_in_project(db):
    """Project-level dedup: a Queue ID matching ANY run name in the project is
    skipped — covers manual Copy runs and sibling automations."""
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=5)
    # A manually copied run already named after Q-00001 (case/space-insensitive).
    manual = Run(
        project_id=template.project_id,
        name="  q-00001 ",
        prompt="swap character",
        resolution="720p",
        status=RunStatus.done,
        reference_image_urls=[],
    )
    db.add(manual)
    db.commit()

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1), _sheet_row(2)]), now=NOW)

    launched = {l.queue_id for l in db.execute(select(AutomationLaunch)).scalars().all()}
    assert launched == {"Q-00002"}
    assert len(created) == 1


def test_tick_dedup_is_scoped_to_the_project(db):
    """A same-named run in a DIFFERENT project does not block the Queue ID."""
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=5)
    other_project_run = _make_ready_project_with_run(db)  # separate project
    other_project_run.name = "Q-00001"
    db.commit()

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW)

    launched = {l.queue_id for l in db.execute(select(AutomationLaunch)).scalars().all()}
    assert launched == {"Q-00001"}
    assert len(created) == 1


def test_tick_skips_when_quota_exhausted_today(db, webhook_spy):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=1)
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])

    first = run_tick(db, sheets=sheets, now=NOW)
    assert len(first) == 1
    db.get(Run, first[0]).status = RunStatus.done
    db.commit()

    # Same UTC day → quota spent, nothing new even though the batch finished.
    assert run_tick(db, sheets=sheets, now=NOW + timedelta(hours=2)) == []
    # Next UTC day → quota resets.
    assert len(run_tick(db, sheets=sheets, now=NOW + timedelta(days=1))) == 1


def test_tick_blocks_next_batch_while_in_flight(db, webhook_spy):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=10)
    sheets = FakeSheets([_sheet_row(i) for i in range(1, 30)])

    first = run_tick(db, sheets=sheets, now=NOW)
    assert len(first) == 10
    # Batch still processing → no new launches on the next tick.
    assert run_tick(db, sheets=sheets, now=NOW + timedelta(minutes=10)) == []


def test_tick_reads_sheet_once_regardless_of_automation_count(db):
    """B6: the Daily Queue tab is the same spreadsheet/tab for every enabled
    automation in a tick — it must be read at most once per tick, not once
    per automation."""
    template1 = _make_ready_project_with_run(db)
    template2 = _make_ready_project_with_run(db)
    _make_automation(db, template1.id, daily_count=5)
    _make_automation(db, template2.id, daily_count=5)
    sheets = FakeSheets([_sheet_row(i) for i in range(1, 5)])

    created = run_tick(db, sheets=sheets, now=NOW)

    assert len(created) == 8  # 4 rows x 2 automations, distinct projects
    assert sheets.read_calls == 1


def test_tick_skips_sheet_read_when_no_automation_needs_it(db):
    """Lazy read: an automation whose quota is already exhausted (or whose
    template/project isn't ready) must not trigger the sheet read at all."""
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, enabled=False)
    sheets = FakeSheets([_sheet_row(1)])

    assert run_tick(db, sheets=sheets, now=NOW) == []
    assert sheets.read_calls == 0


def test_tick_caches_sheet_read_failure_across_automations_in_one_tick(db):
    """A Sheets outage should cost one network round-trip per tick, not one
    per enabled automation — but each blocked automation still logs its own
    'sheet read failed' warning (observable behavior unchanged)."""
    template1 = _make_ready_project_with_run(db)
    template2 = _make_ready_project_with_run(db)
    _make_automation(db, template1.id, daily_count=5)
    _make_automation(db, template2.id, daily_count=5)
    sheets = FailingSheets()

    created = run_tick(db, sheets=sheets, now=NOW)

    assert created == []
    assert sheets.read_calls == 1  # cached failure, not retried per automation


def test_tick_skips_disabled_and_not_ready(db):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, enabled=False)
    assert run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW) == []

    project = db.get(VideoProject, template.project_id)
    project.status = ProjectStatus.analyzing
    automation = db.execute(select(RunAutomation)).scalar_one()
    automation.enabled = True
    db.commit()
    assert run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW) == []


def test_tick_skips_a_speech_only_localisation_template(db, caplog):
    """Automation is Copy run on a daily quota, so it inherits Copy run's one
    refusal: a speech-only localisation template keeps the person on screen, so
    the sheet's photo is inert and every launch would repeat the same clip.

    The API refuses to *enable* such an automation; this is the half that binds
    on one enabled before that rule existed, or whose template was edited after
    the fact (docs/localisation.md §7).
    """
    from app import project_types

    template = _make_ready_project_with_run(db)
    project = db.get(VideoProject, template.project_id)
    project.project_type = "localisation"
    template.prompt = project_types._LOCALISATION_KEEP_PROMPT.format(
        source_language="english", target_language="japanese",
        dialogue="**Woman:**     はい",
    )
    db.commit()
    _make_automation(db, template.id, enabled=True)

    assert run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW) == []
    assert db.execute(select(AutomationLaunch)).scalars().all() == []

    # Same automation, same sheet — the speech + character template launches.
    template.prompt = project_types._LOCALISATION_SWAP_PROMPT.format(
        source_language="english", target_language="japanese",
        dialogue="**Woman:**     はい",
    )
    db.commit()
    assert len(run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW)) == 1


# ---------------------------------------------------------------------------
# Reconcile phase — webhook per finished batch
# ---------------------------------------------------------------------------


def test_batch_webhook_sent_once_when_all_runs_terminal(db, webhook_spy):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=2, webhook_url="https://hook.test/x")
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])

    created = run_tick(db, sheets=sheets, now=NOW)
    assert webhook_spy == []  # runs still queued

    r1, r2 = (db.get(Run, rid) for rid in created)
    r1.status = RunStatus.done
    r1.result_gdrive_file_id = "FILE1"
    r2.status = RunStatus.failed
    r2.error_message = "boom"
    db.commit()

    run_tick(db, sheets=sheets, now=NOW + timedelta(hours=1))
    assert len(webhook_spy) == 1
    url, payload = webhook_spy[0]
    assert url == "https://hook.test/x"
    assert payload["event"] == "automation.batch.completed"
    by_queue = {item["queue_id"]: item for item in payload["runs"]}
    assert by_queue["Q-00001"]["status"] == "done"
    assert by_queue["Q-00001"]["drive_link"] == (
        "https://drive.google.com/file/d/FILE1/view"
    )
    assert by_queue["Q-00002"]["status"] == "failed"
    assert by_queue["Q-00002"]["error"] == "boom"

    # Already notified → no duplicate webhook on the next tick.
    quota_spent = run_tick(db, sheets=sheets, now=NOW + timedelta(hours=2))
    assert quota_spent == []
    assert len(webhook_spy) == 1
    launches = db.execute(select(AutomationLaunch)).scalars().all()
    assert all(l.notified_at is not None for l in launches)


def test_failed_webhook_is_retried_and_blocks_next_batch(db, monkeypatch):
    attempts: list[str] = []

    def failing_post(url, payload):
        attempts.append(url)
        return False

    monkeypatch.setattr(automation_mod, "_post_webhook", failing_post)

    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=5, webhook_url="https://hook.test/x")
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])

    created = run_tick(db, sheets=sheets, now=NOW)
    for rid in created:
        db.get(Run, rid).status = RunStatus.done
    db.commit()

    # Webhook fails → batch stays un-notified, nothing new launches.
    assert run_tick(db, sheets=sheets, now=NOW + timedelta(hours=1)) == []
    assert run_tick(db, sheets=sheets, now=NOW + timedelta(hours=2)) == []
    assert len(attempts) == 2
    launches = db.execute(select(AutomationLaunch)).scalars().all()
    assert all(l.notified_at is None for l in launches)


def test_deleted_run_does_not_block_batch(db, webhook_spy):
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=2)
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])

    created = run_tick(db, sheets=sheets, now=NOW)
    done_run = db.get(Run, created[0])
    done_run.status = RunStatus.done
    db.delete(db.get(Run, created[1]))  # operator deleted one copy
    db.commit()

    run_tick(db, sheets=sheets, now=NOW + timedelta(hours=1))
    assert len(webhook_spy) == 1
    statuses = {i["queue_id"]: i["status"] for i in webhook_spy[0][1]["runs"]}
    assert statuses["Q-00001"] == "done"
    assert statuses["Q-00002"] == "deleted"


# ---------------------------------------------------------------------------
# API — GET/PUT /runs/{rid}/automation
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(engine, SessionFactory):
    from app.main import app

    def override_get_db():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def test_automation_api_defaults_then_upsert(db, client):
    run = _make_ready_project_with_run(db)

    resp = client.get(f"/api/v2/runs/{run.id}/automation")
    assert resp.status_code == 200
    assert resp.json()["configured"] is False
    assert resp.json()["enabled"] is False

    body = {
        "enabled": True,
        "filter_persona": " Student ",
        "filter_age": "",
        "filter_gender": "M,F",
        "filter_ethnicity": None,
        "exclude_gender": True,
        "plan": "midplan",
        "daily_count": 7,
        "webhook_url": "https://hook.test/y",
    }
    resp = client.put(f"/api/v2/runs/{run.id}/automation", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True
    assert data["enabled"] is True
    assert data["filter_persona"] == "Student"  # trimmed
    assert data["filter_age"] is None  # blank → any
    assert data["filter_gender"] == "M,F"
    assert data["exclude_gender"] is True
    assert data["exclude_persona"] is False
    assert data["plan"] == "midplan"
    assert data["daily_count"] == 7

    # Update in place (no duplicate row)
    body["enabled"] = False
    body["daily_count"] = 2
    resp = client.put(f"/api/v2/runs/{run.id}/automation", json=body)
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    assert len(db.execute(select(RunAutomation)).scalars().all()) == 1


def test_automation_api_validation(db, client):
    run = _make_ready_project_with_run(db)
    base = {"enabled": True, "plan": "close-up", "daily_count": 5}

    resp = client.put(
        f"/api/v2/runs/{run.id}/automation", json={**base, "plan": "wide"}
    )
    assert resp.status_code == 400

    resp = client.put(
        f"/api/v2/runs/{run.id}/automation", json={**base, "daily_count": 0}
    )
    assert resp.status_code == 400

    resp = client.put(
        f"/api/v2/runs/{run.id}/automation",
        json={**base, "webhook_url": "ftp://nope"},
    )
    assert resp.status_code == 400

    resp = client.get("/api/v2/runs/does-not-exist/automation")
    assert resp.status_code == 404


def test_automation_api_preprocess_refs_needs_a_preprocess_template(db, client):
    """Ticking "re-run GPT Image 2 per launch" on a template that never used it
    is refused, not silently ignored — the operator would otherwise only find
    out from the delivered videos."""
    run = _make_ready_project_with_run(db)
    base = {"enabled": True, "plan": "close-up", "daily_count": 5}

    resp = client.put(
        f"/api/v2/runs/{run.id}/automation", json={**base, "preprocess_refs": True}
    )
    assert resp.status_code == 409
    assert "not created with preprocessing" in resp.json()["detail"].lower()

    # And the panel knows not to offer it.
    assert client.get(f"/api/v2/runs/{run.id}/automation").json() \
        ["preprocess_available"] is False


def test_automation_api_preprocess_refs_round_trips(db, client):
    run = _make_preprocess_template(db)
    body = {
        "enabled": True, "plan": "close-up", "daily_count": 5,
        "preprocess_refs": True,
    }

    resp = client.put(f"/api/v2/runs/{run.id}/automation", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["preprocess_refs"] is True
    assert resp.json()["preprocess_available"] is True

    got = client.get(f"/api/v2/runs/{run.id}/automation").json()
    assert got["preprocess_refs"] is True

    # Untickable again.
    resp = client.put(
        f"/api/v2/runs/{run.id}/automation", json={**body, "preprocess_refs": False}
    )
    assert resp.json()["preprocess_refs"] is False


def test_automation_api_counters(db, client, webhook_spy):
    run = _make_ready_project_with_run(db)
    _make_automation(db, run.id, daily_count=2)
    run_tick(db, sheets=FakeSheets([_sheet_row(1), _sheet_row(2)]), now=datetime.now(timezone.utc))

    resp = client.get(f"/api/v2/runs/{run.id}/automation")
    data = resp.json()
    assert data["launched_today"] == 2
    assert data["in_flight"] == 2
    assert data["used_total"] == 2


def test_batch_webhook_reports_partly_swapped_runs(db, webhook_spy):
    """`unswapped_segments` distinguishes a fully-swapped `done` from a partial one.

    A run whose project was segmented longer than its model can generate still
    reaches `done` and delivers — with the ORIGINAL footage in place of the
    segments the model couldn't do (see RunSegment.source_fallback). Without this
    field a batch of partly-swapped videos reports as clean success downstream,
    which is exactly the failure the flag exists to make visible.
    """
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=2, webhook_url="https://hook.test/x")
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])

    created = run_tick(db, sheets=sheets, now=NOW)
    r1, r2 = (db.get(Run, rid) for rid in created)
    # Both delivered; only the second had segments substituted.
    r1.status = RunStatus.done
    r1.source_fallback_segments = 0
    r2.status = RunStatus.done
    r2.source_fallback_segments = 2
    db.commit()

    run_tick(db, sheets=sheets, now=NOW + timedelta(hours=1))
    _url, payload = webhook_spy[0]
    by_queue = {item["queue_id"]: item for item in payload["runs"]}

    # Same status, different swap coverage — the point of the field.
    assert by_queue["Q-00001"]["status"] == "done"
    assert by_queue["Q-00002"]["status"] == "done"
    assert by_queue["Q-00001"]["unswapped_segments"] == 0
    assert by_queue["Q-00002"]["unswapped_segments"] == 2
    # And it is not smuggled through `error`, which stays a failure channel.
    assert by_queue["Q-00002"]["error"] is None


# ---------------------------------------------------------------------------
# Run-local segment range regeneration inheritance
# ---------------------------------------------------------------------------


def _make_ready_project_with_snapshot_run(db) -> tuple:
    """A done template Run over keep[0,5) | swap[5,10) | keep[10,20), with
    Run.segment_snapshot frozen exactly like a real first submit would."""
    project = VideoProject(
        source_type="upload", source_ref="test.mp4", status=ProjectStatus.ready,
        duration_sec=20.0,
    )
    db.add(project)
    db.commit()
    sd_keep0 = SegmentDef(project_id=project.id, index=0, action="keep", start_sec=0.0, end_sec=5.0)
    sd_swap1 = SegmentDef(project_id=project.id, index=1, action="swap", start_sec=5.0, end_sec=10.0)
    sd_keep2 = SegmentDef(project_id=project.id, index=2, action="keep", start_sec=10.0, end_sec=20.0)
    db.add_all([sd_keep0, sd_swap1, sd_keep2])
    db.commit()
    run = Run(
        project_id=project.id, name="template", prompt="swap character",
        resolution="720p", status=RunStatus.done, reference_image_urls=[],
        segment_snapshot=[
            {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 5.0},
            {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 5.0, "end_sec": 10.0},
            {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
        ],
    )
    db.add(run)
    db.commit()
    rs = RunSegment(
        run_id=run.id, segment_def_id=sd_swap1.id, index=1,
        status=SegmentStatus.completed, prompt_override="a specific look",
    )
    db.add(rs)
    db.commit()
    return run, sd_keep0, sd_swap1, sd_keep2, rs


def test_tick_after_promoted_range_launches_with_new_range_without_resave(db):
    """Enable automation, promote a range correction on the template
    (simulated the same way pipeline_range_attempt._promote writes it — a
    deep-copied, updated Run.segment_snapshot, no other Run field touched),
    then tick: the new child Run must carry the PROMOTED range, not the
    original one, without the operator re-saving the automation config."""
    run, sd_keep0, sd_swap1, sd_keep2, rs = _make_ready_project_with_snapshot_run(db)
    _make_automation(db, run.id, daily_count=1)

    promoted_snapshot = [
        {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 4.6},
        {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 4.6, "end_sec": 10.0},
        {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
    ]
    run.segment_snapshot = promoted_snapshot
    db.commit()

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW)
    assert len(created) == 1
    new_run = db.get(Run, created[0])
    assert new_run.segment_snapshot == promoted_snapshot
    # Per-segment prompt override retained; Daily Queue photo is run-level
    # only, so the per-segment override is untouched by it.
    new_rs = [x for x in new_run.run_segments if x.segment_def_id == sd_swap1.id]
    assert len(new_rs) == 1
    assert new_rs[0].prompt_override == "a specific look"


def test_tick_child_does_not_use_current_project_timing(db):
    """Even if the Project's live SegmentDefs are edited after promotion (an
    operator edit unrelated to this run), the automation-launched copy must
    still use the RUN's frozen (promoted) timeline, not the live one."""
    run, sd_keep0, sd_swap1, sd_keep2, rs = _make_ready_project_with_snapshot_run(db)
    _make_automation(db, run.id, daily_count=1)

    promoted_snapshot = [
        {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 4.6},
        {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 4.6, "end_sec": 10.0},
        {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
    ]
    run.segment_snapshot = promoted_snapshot
    # Live project segments now disagree with the frozen snapshot (simulates
    # an unrelated later project edit).
    sd_swap1.start_sec = 6.0
    sd_swap1.end_sec = 11.0
    db.commit()

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW)
    new_run = db.get(Run, created[0])
    assert new_run.segment_snapshot == promoted_snapshot


def test_tick_still_launches_after_the_template_run_segments_live_def_is_deleted(db):
    """Regression for the independent audit's blocking finding #1: a live
    SegmentDef can be deleted (Project re-analysis, or a bulk segment edit)
    after the template Run has already promoted a range correction.
    RunSegment.segment_def_id is not an FK to segment_defs any more (see its
    docstring in models.py), so the template's own RunSegment survives that
    delete — and _build_copied_run must still be able to insert a fresh
    RunSegment for the child naming that same (now-live-deleted)
    segment_def_id without hitting an FK error."""
    run, sd_keep0, sd_swap1, sd_keep2, rs = _make_ready_project_with_snapshot_run(db)
    _make_automation(db, run.id, daily_count=1)

    promoted_snapshot = [
        {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 4.6},
        {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 4.6, "end_sec": 10.0},
        {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
    ]
    run.segment_snapshot = promoted_snapshot
    db.commit()

    # Delete the live SegmentDef the template's RunSegment/snapshot name —
    # the same delete pipeline_v2.analyze_project performs wholesale.
    db.delete(sd_swap1)
    db.commit()
    assert db.get(SegmentDef, sd_swap1.id) is None

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW)
    assert len(created) == 1
    new_run = db.get(Run, created[0])
    assert new_run.segment_snapshot == promoted_snapshot
    new_rs = [x for x in new_run.run_segments if x.segment_def_id == sd_swap1.id]
    assert len(new_rs) == 1
    assert new_rs[0].prompt_override == "a specific look"


def test_tick_skips_template_with_in_flight_range_attempt(db):
    """An active range attempt on the template must skip this tick entirely —
    no quota consumed, no sheet row reserved/marked, no Run enqueued."""
    run, sd_keep0, sd_swap1, sd_keep2, rs = _make_ready_project_with_snapshot_run(db)
    _make_automation(db, run.id, daily_count=5)

    attempt = RunSegmentRangeAttempt(
        run_id=run.id, run_segment_id=rs.id,
        requested_start_sec=4.6, requested_end_sec=10.0,
        base_segment_snapshot=run.segment_snapshot,
        candidate_segment_snapshot=run.segment_snapshot,
        status=RangeAttemptStatus.generating,
    )
    db.add(attempt)
    db.commit()

    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])
    created = run_tick(db, sheets=sheets, now=NOW)

    assert created == []
    assert sheets.read_calls == 0, "must not even read the sheet — no row should be at risk of being marked used"
    assert db.execute(select(AutomationLaunch)).scalars().all() == []


def test_tick_launches_normally_once_attempt_is_terminal(db):
    """A FAILED (or done) attempt must not block future ticks — only an
    ACTIVE one does."""
    run, sd_keep0, sd_swap1, sd_keep2, rs = _make_ready_project_with_snapshot_run(db)
    _make_automation(db, run.id, daily_count=5)

    attempt = RunSegmentRangeAttempt(
        run_id=run.id, run_segment_id=rs.id,
        requested_start_sec=4.6, requested_end_sec=10.0,
        base_segment_snapshot=run.segment_snapshot,
        candidate_segment_snapshot=run.segment_snapshot,
        status=RangeAttemptStatus.failed,
        error_message="injected failure",
    )
    db.add(attempt)
    db.commit()

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW)
    assert len(created) == 1
    # A failed attempt is never inherited — the child gets the template's
    # ORIGINAL (still-active) range, not the rejected candidate's.
    new_run = db.get(Run, created[0])
    assert new_run.segment_snapshot == run.segment_snapshot


def test_tick_after_failed_candidate_uses_prior_active_range_not_the_rejected_one(db):
    """Stronger version of the terminal-attempt case above: the failed
    attempt's candidate range is DELIBERATELY DIFFERENT from the template's
    active snapshot, proving the rejected candidate cannot leak into a launch
    even though it is the most RECENT attempt row on the template."""
    run, sd_keep0, sd_swap1, sd_keep2, rs = _make_ready_project_with_snapshot_run(db)
    _make_automation(db, run.id, daily_count=5)
    original_snapshot = list(run.segment_snapshot)

    rejected_candidate = [
        {"id": sd_keep0.id, "index": 0, "action": "keep", "start_sec": 0.0, "end_sec": 4.6},
        {"id": sd_swap1.id, "index": 1, "action": "swap", "start_sec": 4.6, "end_sec": 10.0},
        {"id": sd_keep2.id, "index": 2, "action": "keep", "start_sec": 10.0, "end_sec": 20.0},
    ]
    attempt = RunSegmentRangeAttempt(
        run_id=run.id, run_segment_id=rs.id,
        requested_start_sec=4.6, requested_end_sec=10.0,
        base_segment_snapshot=original_snapshot,
        candidate_segment_snapshot=rejected_candidate,
        status=RangeAttemptStatus.failed,
        error_message="injected failure — never promoted",
    )
    db.add(attempt)
    db.commit()

    created = run_tick(db, sheets=FakeSheets([_sheet_row(1)]), now=NOW)
    assert len(created) == 1
    new_run = db.get(Run, created[0])
    assert new_run.segment_snapshot == original_snapshot
    by_id = {e["id"]: e for e in new_run.segment_snapshot}
    # The rejected candidate's 4.6 boundary must NOT have leaked in.
    assert by_id[sd_swap1.id]["start_sec"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Italian (it) inheritance through the automation copy path
# ---------------------------------------------------------------------------


def test_tick_inherits_italian_from_the_template_run(db):
    """Automation reuses _build_copied_run, so an Italian template can only
    spawn Italian batches — never English ones.

    Sheets is the in-memory FakeSheets and run_tick returns ids instead of
    enqueueing, so nothing here touches Redis, RQ or a live spreadsheet.
    """
    template = _make_ready_project_with_run(db)
    template.language = "it"
    db.commit()
    _make_automation(db, template.id, daily_count=3)
    sheets = FakeSheets([_sheet_row(i) for i in range(1, 10)])

    created = run_tick(db, sheets=sheets, now=NOW)

    assert len(created) == 3
    for run_id in created:
        run = db.get(Run, run_id)
        assert run.language == "it", run_id
        # The id the sheet would receive for each child.
        assert run.short_id == f"{run_id[:8]}-it"


def test_tick_leaves_an_unset_template_language_unset(db):
    """Regression: adding a sixth language must not start defaulting NULL."""
    template = _make_ready_project_with_run(db)
    assert template.language is None
    _make_automation(db, template.id, daily_count=2)
    sheets = FakeSheets([_sheet_row(i) for i in range(1, 10)])

    created = run_tick(db, sheets=sheets, now=NOW)

    assert len(created) == 2
    for run_id in created:
        run = db.get(Run, run_id)
        assert run.language is None, run_id
        assert run.short_id == run_id[:8]


# ---------------------------------------------------------------------------
# Daily kie.ai budget (AICP-155)
# ---------------------------------------------------------------------------


def _budget_template(db, seconds=5.0) -> Run:
    """A template whose copies cost 5s x 52 credits (seedance, 720p) = 260."""
    template = _make_ready_project_with_run(db)
    template.model = "seedance"
    db.add(SegmentDef(
        project_id=template.project_id, index=0, start_sec=0.0, end_sec=seconds,
        has_face=True, action="swap",
    ))
    db.commit()
    return template


@pytest.fixture()
def budget(db, monkeypatch):
    """Cap in credits for NOW's UTC day, with that day's spend rows wiped."""
    from app.config import settings
    from app.models import KieSpendDay, KieSpendDraw, KieSpendReservation

    for model in (KieSpendDraw, KieSpendReservation, KieSpendDay):
        db.query(model).delete()
    db.commit()
    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)

    def _cap(credits):
        monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", credits * 0.005)

    yield _cap
    for model in (KieSpendDraw, KieSpendReservation, KieSpendDay):
        db.query(model).delete()
    db.commit()


@pytest.mark.spend_cap
def test_tick_launches_only_what_todays_budget_covers(db, budget):
    from app import spend

    template = _budget_template(db)
    _make_automation(db, template.id, daily_count=5)
    budget(2 * 260 + 100)  # room for two copies

    created = run_tick(db, sheets=FakeSheets([_sheet_row(i) for i in range(1, 10)]), now=NOW)

    assert len(created) == 2
    launches = db.execute(select(AutomationLaunch)).scalars().all()
    # Only the launched rows are consumed; Q-00003.. stay available.
    assert {l.queue_id for l in launches} == {"Q-00001", "Q-00002"}
    assert spend.today_summary(db, NOW)["charged_credits"] == pytest.approx(520)


@pytest.mark.spend_cap
def test_tick_over_budget_launches_nothing_and_consumes_no_quota(db, budget):
    template = _budget_template(db)
    automation = _make_automation(db, template.id, daily_count=5)
    budget(100)  # less than one copy

    created = run_tick(db, sheets=FakeSheets([_sheet_row(i) for i in range(1, 4)]), now=NOW)

    assert created == []
    assert db.execute(select(AutomationLaunch)).scalars().all() == []
    assert len(db.execute(select(Run)).scalars().all()) == 1  # just the template
    assert automation_mod.launched_today(db, automation.id, NOW) == 0

    # Next UTC day the budget is fresh and the same rows launch.
    budget(10_000)
    created = run_tick(
        db, sheets=FakeSheets([_sheet_row(i) for i in range(1, 4)]),
        now=NOW + timedelta(days=1),
    )
    assert len(created) == 3


@pytest.mark.spend_cap
def test_tick_that_loses_the_budget_race_rolls_the_whole_batch_back(db, budget, monkeypatch):
    """Pre-trim saw room, but another spender took it before the reserve: the
    batch is rolled back whole — no runs, no launches — not half-launched."""
    from app import spend

    template = _budget_template(db)
    _make_automation(db, template.id, daily_count=5)
    budget(10_000)
    # Pre-trim reads a stale, generous headroom...
    monkeypatch.setattr(spend, "headroom_credits", lambda _db, _now=None: 10_000.0)
    # ...while the day is in fact nearly spent.
    spend.reserve(db, "run:someone-else", 9_900, kind="new run", now=NOW)
    db.commit()

    created = run_tick(db, sheets=FakeSheets([_sheet_row(i) for i in range(1, 4)]), now=NOW)

    assert created == []
    assert db.execute(select(AutomationLaunch)).scalars().all() == []
    assert len(db.execute(select(Run)).scalars().all()) == 1
    assert spend.today_summary(db, NOW)["charged_credits"] == pytest.approx(9_900)

# AICP-156: webhook lock-down
# ---------------------------------------------------------------------------


def test_unset_webhook_env_disables_webhook_and_run_is_unaffected(db, monkeypatch, caplog):
    from app.config import settings

    monkeypatch.setattr(settings, "AUTOMATION_WEBHOOK_URL", "")
    monkeypatch.setattr(automation_mod, "_warned_disabled", False)

    def boom(url, payload):
        raise AssertionError("no webhook may be attempted")

    monkeypatch.setattr(automation_mod, "_post_webhook", boom)
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=2)
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])
    created = run_tick(db, sheets=sheets, now=NOW)
    assert created
    for rid in created:
        db.get(Run, rid).status = RunStatus.done
    db.commit()
    with caplog.at_level("WARNING"):
        run_tick(db, sheets=sheets, now=NOW + timedelta(hours=1))
        run_tick(db, sheets=sheets, now=NOW + timedelta(hours=2))
    launches = db.execute(select(AutomationLaunch)).scalars().all()
    assert all(l.notified_at is not None for l in launches)  # not blocked
    assert all(db.get(Run, rid).status == RunStatus.done for rid in created)
    assert sum("webhook disabled" in r.getMessage() for r in caplog.records) == 1


def test_policy_refused_webhook_does_not_block_automation(db, monkeypatch):
    from app.webhook_safety import WebhookRefused

    def refuse(url, payload):
        raise WebhookRefused("nope")

    monkeypatch.setattr(automation_mod, "_post_webhook", refuse)
    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=2, webhook_url="https://hook.test/x")
    sheets = FakeSheets([_sheet_row(1), _sheet_row(2)])
    for rid in run_tick(db, sheets=sheets, now=NOW):
        db.get(Run, rid).status = RunStatus.done
    db.commit()
    run_tick(db, sheets=sheets, now=NOW + timedelta(hours=1))
    launches = db.execute(select(AutomationLaunch)).scalars().all()
    assert all(l.notified_at is not None for l in launches)


@pytest.mark.parametrize("url", [
    "http://hook.test/x", "https://evil.example/x", "https://127.0.0.1/x",
    "https://169.254.169.254/x", "https://[::1]/x",
])
def test_automation_api_rejects_unsafe_webhook_url(db, client, url):
    run = _make_ready_project_with_run(db)
    resp = client.put(
        f"/api/v2/runs/{run.id}/automation",
        json={"enabled": True, "plan": "close-up", "daily_count": 5, "webhook_url": url},
    )
    assert resp.status_code == 400
    assert db.execute(select(RunAutomation)).scalars().all() == []


# ---------------------------------------------------------------------------
# AICP-350: the deploy admission gate skips the tick
# ---------------------------------------------------------------------------


def test_tick_skips_launch_while_the_deploy_gate_is_paused(db):
    from app.ops_gate import OpsAdmissionGate

    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=3)
    sheets = FakeSheets([_sheet_row(i) for i in range(1, 10)])
    gate = db.get(OpsAdmissionGate, 1) or OpsAdmissionGate(id=1, generation=0)
    gate.paused, gate.owner, gate.paused_at = True, "deploy:test", NOW
    db.add(gate)
    db.commit()
    try:
        assert run_tick(db, sheets=sheets, now=NOW) == []
        assert db.execute(select(AutomationLaunch)).scalars().all() == []
        assert sheets.read_calls == 0  # nothing consumed, not even a sheet read
    finally:
        gate.paused, gate.owner, gate.paused_at = False, None, None
        db.commit()
    assert len(run_tick(db, sheets=sheets, now=NOW)) == 3


def test_tick_rechecks_the_gate_under_its_share_lock_before_the_batch(db, monkeypatch):
    """Paused between the cheap read and the batch: the share-locked re-check
    refuses, the batch is rolled back, nothing is consumed."""
    from app import ops_gate

    template = _make_ready_project_with_run(db)
    _make_automation(db, template.id, daily_count=3)
    sheets = FakeSheets([_sheet_row(i) for i in range(1, 10)])

    def _paused(session):
        raise ops_gate.AdmissionPaused(ops_gate.GateState(paused=True))

    monkeypatch.setattr(ops_gate, "lock_admitting", _paused)
    assert run_tick(db, sheets=sheets, now=NOW) == []
    assert db.execute(select(AutomationLaunch)).scalars().all() == []
