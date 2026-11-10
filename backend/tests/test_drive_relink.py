"""Tests for app/drive_relink.py — re-link a run whose Drive file was deleted
(AICP-348): the worker's re-checks, the idempotent tagged upload, the
re-point, the audit row and the sheet scan. No network: fake Drive/Sheets."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_drive_relink_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import app.drive_relink as dr
from app.db import Base
from app.gdrive_client import GDriveError
from app.models import (
    Run,
    RunCaption,
    RunDriveRelink,
    RunSegment,
    RunSegmentRangeAttempt,
    RunTailEdit,
    VideoProject,
)
from app.state_machine import (
    CaptionStatus,
    RangeAttemptStatus,
    RunStatus,
    SegmentStatus,
    TailEditStatus,
)


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        f"sqlite:///{os.path.join(_TMP_DIR, 'relink.db')}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(eng, "connect")
    def _fk(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def Session(engine, monkeypatch):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    @contextmanager
    def _get_session():
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr(dr, "get_session", _get_session)
    monkeypatch.setattr(dr.settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-x", raising=False)
    monkeypatch.setattr(dr.settings, "FINAL_CREATIVES_SPREADSHEET_ID", "sheet-x", raising=False)
    return factory


class FakeDrive:
    """Drive by id: ``state`` answers for the OLD file; any other id is
    present iff it was created. Creating an existing id fails (Drive 409)."""

    def __init__(self, *, state="missing"):
        self.state = state
        self.files: dict[str, dict] = {}   # id -> {"md5", "props"}
        self.uploads: list[dict] = []
        self.updates: list[tuple] = []
        self.upload_error: Exception | None = None
        self.lose_response = False
        self.state_error_for_new: Exception | None = None
        self.folders = {"folder-run": "present", "folder-x": "present"}

    def file_state(self, file_id):
        if file_id in self.folders:
            return self.folders[file_id]
        if file_id == "old-id":
            if isinstance(self.state, Exception):
                raise self.state
            return self.state
        if self.state_error_for_new is not None:
            raise self.state_error_for_new
        return "present" if file_id in self.files else "missing"

    def upload_file(self, path, folder_id=None, name=None, mime_type="video/mp4",
                    app_properties=None, file_id=None):
        self.uploads.append({"path": path, "folder": folder_id, "name": name,
                             "props": app_properties, "id": file_id})
        if self.upload_error is not None:
            raise self.upload_error
        if file_id in self.files:
            raise RuntimeError("HttpError 409 A file already exists with the provided ID")
        self.files[file_id] = {"md5": _md5(path), "props": dict(app_properties or {})}
        if self.lose_response:
            self.lose_response = False
            raise RuntimeError("connection reset after the last chunk")
        return {"id": file_id}

    def file_md5(self, file_id):
        return self.files[file_id]["md5"]

    def update_file_content(self, file_id, path, mime_type="video/mp4"):
        self.updates.append((file_id, path))
        self.files[file_id]["md5"] = _md5(path)
        return {"id": file_id}


class FakeSheets:
    def __init__(self, rows):
        self.rows = rows  # list of dicts, header implied by keys

    def get_header(self, sid, tab):
        return ["Date", "Job ID", "Input Link", "Creative Link", "Pipeline Type"]

    def column_values(self, sid, tab, idx):
        header = self.get_header(sid, tab)
        return [header[idx]] + [r.get(header[idx], "") for r in self.rows]


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _world(Session, *, status=RunStatus.done, exported=True):
    s = Session()
    project = VideoProject(source_type="upload", source_ref="src.mp4",
                           source_local_path="/nonexistent/src.mp4")
    s.add(project)
    s.flush()
    final_dir = tempfile.mkdtemp(dir=_TMP_DIR)
    final = os.path.join(final_dir, "final.mp4")
    with open(final, "wb") as fh:
        fh.write(os.urandom(4096))
    run = Run(project_id=project.id, prompt="p", model="seedance", resolution="480p",
              status=status, result_local_path=final, result_gdrive_file_id="old-id",
              gdrive_folder_id="folder-run", language="es",
              final_creatives_exported_at=datetime.now(timezone.utc) if exported else None)
    s.add(run)
    s.commit()
    rid = run.id
    s.close()
    return rid, final


def _queue(Session, rid, final, *, old="old-id", run_status="done"):
    s = Session()
    row = RunDriveRelink(run_id=rid, status="queued", old_file_id=old,
                         target_file_id="target-1", result_path=final,
                         run_status=run_status)
    s.add(row)
    s.commit()
    rl = row.id
    s.close()
    return rl


def _get(Session, model, pk):
    s = Session()
    obj = s.get(model, pk)
    s.expunge(obj)
    s.close()
    return obj


def test_relink_uploads_tagged_file_and_repoints_the_run(Session):
    rid, final = _world(Session, status=RunStatus.failed)
    rl = _queue(Session, rid, final, run_status="failed")
    drive = FakeDrive()
    old_link = "https://drive.google.com/file/d/old-id/view"
    sheets = FakeSheets([
        {"Job ID": rid[:8] + "-es", "Creative Link": old_link, "Pipeline Type": "re-skin-v2"},
        {"Job ID": "other", "Creative Link": "https://drive.google.com/file/d/zzz/view"},
        {"Job ID": "tail1-es", "Input Link": old_link, "Pipeline Type": "re-skin-tail-edit"},
    ])

    assert dr.process_relink(rl, gdrive=drive, sheets=sheets, sleep=lambda s: None) == "done"

    run = _get(Session, Run, rid)
    row = _get(Session, RunDriveRelink, rl)
    assert run.result_gdrive_file_id == "target-1" == row.new_file_id
    assert run.status == RunStatus.failed, "a re-link never changes the run's status"
    assert run.short_id == rid[:8] + "-es"
    assert row.status == "done" and row.error_message is None
    assert len(drive.uploads) == 1
    up = drive.uploads[0]
    assert up["props"] == {dr.APP_PROPERTY: rl}
    assert up["id"] == "target-1", "created with the id reserved at admission"
    assert up["folder"] == "folder-run"
    assert up["name"] == f"reskin_run_{rid}.mp4"
    assert os.path.basename(up["path"]).startswith(".deliver-")
    assert not os.path.exists(up["path"]), "the private pin is removed"
    assert drive.files["target-1"]["md5"] == _md5(final)
    assert row.exported_rows == [{
        "kind": "original", "id": rid, "job_id": rid[:8] + "-es",
        "column": "Creative Link", "pipeline_type": "re-skin-v2",
    }]
    assert [(r["row"], r["column"]) for r in row.sheet_scan["rows"]] == [
        (2, "Creative Link"), (4, "Input Link"),
    ]
    assert row.sheet_scan["error"] is None


@pytest.mark.parametrize("state", ["present", "trashed"])
def test_refused_unless_drive_reports_missing(Session, state):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive(state=state)
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "refused"
    assert drive.uploads == []
    assert _get(Session, Run, rid).result_gdrive_file_id == "old-id"
    assert state in _get(Session, RunDriveRelink, rl).error_message


def test_refused_when_drive_cannot_answer(Session):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive(state=GDriveError("503"))
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "refused"
    assert drive.uploads == []


def test_refused_when_the_run_moved_on(Session):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    s = Session()
    s.get(Run, rid).status = RunStatus.queued  # a retry was admitted meanwhile
    s.commit()
    s.close()
    drive = FakeDrive()
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "refused"
    assert drive.uploads == []
    assert "status" in _get(Session, RunDriveRelink, rl).error_message


def test_refused_when_the_drive_id_changed(Session):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final, old="some-other-id")
    drive = FakeDrive()
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "refused"
    assert drive.uploads == []


def test_refused_while_a_range_attempt_is_active(Session):
    rid, final = _world(Session)
    s = Session()
    rs = RunSegment(run_id=rid, segment_def_id="sd-x", index=0, status=SegmentStatus.completed)
    s.add(rs)
    s.flush()
    s.add(RunSegmentRangeAttempt(
        run_id=rid, run_segment_id=rs.id, requested_start_sec=0.0, requested_end_sec=1.0,
        status=RangeAttemptStatus.generating,
        base_segment_snapshot=[], candidate_segment_snapshot=[],
    ))
    s.commit()
    s.close()
    rl = _queue(Session, rid, final)
    drive = FakeDrive()
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "refused"
    assert drive.uploads == []
    assert "range regeneration" in _get(Session, RunDriveRelink, rl).error_message


def test_lost_upload_response_is_adopted_not_duplicated(Session):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive()
    drive.lose_response = True
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "done"
    assert len(drive.uploads) == 1, "files.get found the reserved id instead of uploading again"
    assert len(drive.files) == 1
    assert _get(Session, Run, rid).result_gdrive_file_id == "target-1"


def test_resumed_relink_adopts_and_fixes_an_earlier_upload(Session):
    """A worker killed after the upload: the resumed job finds the tagged file
    and, since the final changed meanwhile, replaces its content in place."""
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive()
    drive.files["target-1"] = {"md5": "stale", "props": {dr.APP_PROPERTY: rl}}
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "done"
    assert drive.uploads == []
    assert [u[0] for u in drive.updates] == ["target-1"]
    assert _get(Session, Run, rid).result_gdrive_file_id == "target-1"


def test_failed_upload_keeps_the_row_and_its_reserved_id(Session):
    """Even when Drive says the reserved id does not exist (a request it is
    still processing could create it later), the row stays queued: the next
    press resumes it with the SAME id, never a second one."""
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive()
    drive.upload_error = RuntimeError("quota")
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "queued"
    assert len(drive.uploads) == dr.UPLOAD_ATTEMPTS
    assert {u["id"] for u in drive.uploads} == {"target-1"}
    assert _get(Session, Run, rid).result_gdrive_file_id == "old-id"
    row = _get(Session, RunDriveRelink, rl)
    assert row.status == "queued" and "does not exist yet" in row.error_message

    drive.upload_error = None
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "done"
    assert _get(Session, Run, rid).result_gdrive_file_id == "target-1"


def test_unreadable_folder_means_access_problem_not_deletion(Session):
    """A 404 for the file AND the folder: likely a lost share (e.g. a rotated
    service account) — re-linking would replace a link that still works."""
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive()
    drive.folders["folder-run"] = "missing"
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "refused"
    assert drive.uploads == []
    assert "access problem" in _get(Session, RunDriveRelink, rl).error_message


def test_refused_resume_names_the_file_an_earlier_attempt_created(Session):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive(state="present")  # the old file came back meanwhile
    drive.files["target-1"] = {"md5": "x", "props": {}}
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "refused"
    msg = _get(Session, RunDriveRelink, rl).error_message
    assert "target-1" in msg and "NOT linked" in msg
    assert _get(Session, Run, rid).result_gdrive_file_id == "old-id"


def test_unknown_upload_outcome_stays_queued_for_a_resume(Session):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    drive = FakeDrive()
    drive.upload_error = RuntimeError("socket timeout")
    drive.state_error_for_new = GDriveError("Drive 503")
    assert dr.process_relink(rl, gdrive=drive, sleep=lambda s: None) == "queued"
    row = _get(Session, RunDriveRelink, rl)
    assert row.status == "queued" and "may exist" in row.error_message and "target-1" in row.error_message
    assert _get(Session, Run, rid).result_gdrive_file_id == "old-id"


def test_terminal_row_is_not_reprocessed(Session):
    rid, final = _world(Session)
    rl = _queue(Session, rid, final)
    s = Session()
    s.get(RunDriveRelink, rl).status = "done"
    s.commit()
    s.close()
    drive = FakeDrive()
    assert dr.process_relink(rl, gdrive=drive) == "done"
    assert drive.uploads == []


def test_expected_export_rows_lists_run_row_and_direct_derivatives(Session):
    rid, final = _world(Session)
    s = Session()
    now = datetime.now(timezone.utc)
    s.add(RunTailEdit(run_id=rid, cut_sec=1.0, source_type="run",
                      status=TailEditStatus.done, final_creatives_exported_at=now))
    s.add(RunCaption(run_id=rid, source_type="run", status=CaptionStatus.done,
                     final_creatives_exported_at=now))
    s.add(RunTailEdit(run_id=rid, cut_sec=2.0, source_type="run",
                      status=TailEditStatus.done))  # never exported: not listed
    s.commit()
    run = s.get(Run, rid)
    rows = dr.expected_export_rows(s, run)
    s.close()
    kinds = [(r["kind"], r["column"]) for r in rows]
    assert kinds == [("original", "Creative Link"), ("tail", "Input Link"), ("captioned", "Input Link")]
    assert rows[1]["job_id"].endswith("-es")


def test_scan_sheet_reports_read_errors_instead_of_raising(Session):
    class Broken:
        def get_header(self, *a):
            raise RuntimeError("Sheets 500")

    out = dr.scan_sheet("old-id", client=Broken())
    assert out["rows"] == [] and "Sheets 500" in out["error"]
