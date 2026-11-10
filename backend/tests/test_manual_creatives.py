"""Tests for app/manual_creatives.py — script-made creatives (AICP-239).

Temp SQLite DB via create_all, a fake Sheets client. Covers registration
(validation, snapshot, the double-registration/double-export guards), the Job
ID rule and the sheet export (row contents, dedupe, prefix collision, the
stamp only after success).
"""

from __future__ import annotations

import os
import sys
import tempfile

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import manual_creatives as mcm
from app.db import Base
from app.models import ManualCreative, OverlayAsset, OverlayJob
from app.state_machine import OverlayStatus

HEADER = ["Date", "Job ID", "Input Link", "Creative Link", "Pipeline Type",
          "Run on Meta", "Idea"]
SOURCE = "https://drive.google.com/file/d/SRC123/view"


@pytest.fixture()
def db():
    eng = create_engine(f"sqlite:///{tempfile.mkstemp(suffix='.db')[1]}")
    Base.metadata.create_all(eng)
    s = sessionmaker(bind=eng, expire_on_commit=False)()
    yield s
    s.close()
    eng.dispose()


def _overlay(db, *, gdrive="OUTFILE", status=OverlayStatus.done, exported=False):
    job = OverlayJob(
        source_type="upload", status=status, gdrive_file_id=gdrive,
        overlays=[{"asset_path": "assets/card_01.png", "nx": 0.0, "ny": 0.36875,
                   "sw": 1.0, "start_sec": 0.0, "end_sec": 60.0, "fade_sec": 0.0}],
        src_width=1080, src_height=1920, src_duration_sec=11.47,
        final_creatives_exported_at=datetime.now(timezone.utc) if exported else None,
    )
    db.add(job)
    db.commit()
    return job


def _asset(db):
    a = OverlayAsset(name="TenBooksList_DE", local_path="/x/card.png")
    db.add(a)
    db.commit()
    return a


def _register(db, job, **kw):
    args = dict(overlay_job_id=job.id, idea="jjuliaggr-Sport-DE", language="de",
                source_link=SOURCE, linear_issue="AICP-238")
    args.update(kw)
    return mcm.register_overlay(db, **args)


class FakeSheets:
    def __init__(self, rows=()):
        self.rows = [list(r) for r in rows]
        self.appended = []

    def get_header(self, sid, tab):
        return HEADER

    def column_values(self, sid, tab, idx, *, last_n=None):
        return [r[idx] for r in self.rows]

    def append_row(self, sid, tab, row):
        self.appended.append(row)
        self.rows.append(row)
        return {}


def _cell(row, name):
    return row[HEADER.index(name)]


# --- registration -----------------------------------------------------------

def test_register_snapshots_the_trail(db):
    job = _overlay(db)
    asset = _asset(db)

    mc = _register(db, job, asset_id=asset.id)

    assert mc.pipeline_type == "re-skin-manual-script"
    assert (mc.idea, mc.language, mc.source_link) == ("jjuliaggr-Sport-DE", "de", SOURCE)
    assert (mc.gdrive_file_id, mc.overlay_job_id) == ("OUTFILE", job.id)
    assert (mc.overlay_asset_id, mc.overlay_asset_name) == (asset.id, "TenBooksList_DE")
    assert mc.details["overlays"][0]["ny"] == 0.36875
    assert mc.details["src_duration_sec"] == 11.47
    assert mc.linear_issue == "AICP-238"


def test_job_id_is_own_prefix_plus_language_suffix(db):
    mc = _register(db, _overlay(db))
    assert mcm.sheet_job_id(mc) == f"{mc.id[:8]}-de"


def test_english_is_stored_null_and_unsuffixed(db):
    mc = _register(db, _overlay(db), language="EN")
    assert mc.language is None
    assert mcm.sheet_job_id(mc) == mc.id[:8]


@pytest.mark.parametrize("idea", ["", "has space", "-lead", "trail-", "a--b",
                                  "ünicode", "x" * 41])
def test_bad_idea_is_rejected(db, idea):
    with pytest.raises(mcm.ManualCreativeError):
        _register(db, _overlay(db), idea=idea)


def test_unknown_language_is_rejected(db):
    with pytest.raises(mcm.ManualCreativeError, match="unknown language"):
        _register(db, _overlay(db), language="nl")


def test_undelivered_overlay_is_rejected(db):
    with pytest.raises(mcm.ManualCreativeError, match="done, delivered"):
        _register(db, _overlay(db, status=OverlayStatus.processing))
    with pytest.raises(mcm.ManualCreativeError, match="done, delivered"):
        _register(db, _overlay(db, gdrive=None))


def test_overlay_already_exported_as_overlay_is_rejected(db):
    with pytest.raises(mcm.ManualCreativeError, match="already exported"):
        _register(db, _overlay(db, exported=True))


def test_registering_twice_is_rejected(db):
    job = _overlay(db)
    _register(db, job)
    with pytest.raises(mcm.ManualCreativeError, match="already registered"):
        _register(db, job)


def test_unknown_asset_is_rejected(db):
    with pytest.raises(mcm.ManualCreativeError, match="asset"):
        _register(db, _overlay(db), asset_id="nope")


def test_find_by_job_id_and_prefix(db):
    mc = _register(db, _overlay(db))
    assert mcm.find(db, mcm.sheet_job_id(mc)).id == mc.id
    assert mcm.find(db, mc.id).id == mc.id
    with pytest.raises(mcm.ManualCreativeError):
        mcm.find(db, "zzzzzzzz")


# --- export -----------------------------------------------------------------

def test_export_appends_the_row_and_stamps(db, monkeypatch):
    mc = _register(db, _overlay(db))
    sheets = FakeSheets()

    assert mcm.export_to_sheet(db, mc, client=sheets) == "added"

    row = sheets.appended[0]
    assert _cell(row, "Job ID") == f"{mc.id[:8]}-de"
    assert _cell(row, "Pipeline Type") == "re-skin-manual-script"
    assert _cell(row, "Idea") == "jjuliaggr-Sport-DE"
    assert _cell(row, "Input Link") == SOURCE
    assert _cell(row, "Creative Link") == "https://drive.google.com/file/d/OUTFILE/view"
    assert _cell(row, "Run on Meta") == "need_to_rename"
    assert db.get(ManualCreative, mc.id).final_creatives_exported_at is not None


def test_export_is_idempotent(db):
    mc = _register(db, _overlay(db))
    sheets = FakeSheets()
    mcm.export_to_sheet(db, mc, client=sheets)
    assert mcm.export_to_sheet(db, mc, client=sheets) == "already"
    assert len(sheets.appended) == 1


def test_export_dedupes_on_creative_link(db):
    """A row already in the sheet (stamp lost) is adopted, not duplicated."""
    mc = _register(db, _overlay(db))
    existing = [""] * len(HEADER)
    existing[HEADER.index("Creative Link")] = mcm.creative_link(mc)
    sheets = FakeSheets([existing])

    assert mcm.export_to_sheet(db, mc, client=sheets) == "exists"
    assert sheets.appended == []
    assert mc.final_creatives_exported_at is not None


def test_export_refuses_a_job_id_held_by_another_creative(db):
    mc = _register(db, _overlay(db))
    other = [""] * len(HEADER)
    other[HEADER.index("Job ID")] = mcm.sheet_job_id(mc)
    other[HEADER.index("Creative Link")] = "https://drive.google.com/file/d/OTHER/view"
    sheets = FakeSheets([other])

    with pytest.raises(mcm.ManualCreativeError, match="another creative"):
        mcm.export_to_sheet(db, mc, client=sheets)
    assert sheets.appended == []
    assert mc.final_creatives_exported_at is None


def test_failed_append_leaves_it_unstamped(db):
    mc = _register(db, _overlay(db))
    sheets = FakeSheets()

    def boom(*a, **k):
        raise RuntimeError("Sheets 503")

    sheets.append_row = boom
    with pytest.raises(RuntimeError):
        mcm.export_to_sheet(db, mc, client=sheets)
    db.refresh(mc)
    assert mc.final_creatives_exported_at is None
