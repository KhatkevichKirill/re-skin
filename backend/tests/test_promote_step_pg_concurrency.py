"""
Real-PostgreSQL proof that "Move to new project" from a run's step (AICP-331)
copies a file nothing can change or delete underneath it.

The step's row is locked (after the run's) for the done-check AND the copy;
a retry (done -> queued, after which a worker rewrites the output) or a Delete
of that step must wait until the promote has committed. SQLite has no row
locks, so this only runs against a disposable PostgreSQL:

    RESKIN_TEST_PG_URL=postgresql://reskin:t@127.0.0.1:15433/t331

Never point it at the production database — it creates and drops tables.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    not _PG_URL, reason="RESKIN_TEST_PG_URL not set — needs a disposable PostgreSQL"
)

_TMP_DIR = tempfile.mkdtemp(prefix="reskin_test_promote_pg_")
os.environ.setdefault("DATA_DIR", os.path.join(_TMP_DIR, "data"))
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if _PG_URL:
    from fastapi import HTTPException
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.api_v2 as api_v2_module
    from app import storage
    from app.db import Base
    from app.models import Run, RunCaption, RunTailEdit, VideoProject
    from app.state_machine import CaptionStatus, ProjectStatus, RunStatus, TailEditStatus


@pytest.fixture(scope="module")
def Session():
    engine = create_engine(_PG_URL)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def held_copy(monkeypatch):
    """Make the promote's copy pause (after the locks) until released."""
    copying, release = threading.Event(), threading.Event()
    real_copy = api_v2_module.shutil.copy2

    def slow_copy(src, dst, *a, **k):
        copying.set()
        assert release.wait(20), "test never released the copy"
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(api_v2_module.shutil, "copy2", slow_copy)
    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project", lambda pid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_tail_edit", lambda eid: None)
    return copying, release


def _fixture_rows(Session, payload: bytes):
    root = storage.data_dir_root()
    os.makedirs(root, exist_ok=True)
    src = tempfile.NamedTemporaryFile(dir=root, suffix=".mp4", delete=False)
    src.write(payload)
    src.close()
    with Session() as s:
        p = VideoProject(source_type="upload", source_ref="x", status=ProjectStatus.ready)
        s.add(p)
        s.flush()
        run = Run(project_id=p.id, prompt="p", resolution="480p", status=RunStatus.done,
                  reference_image_urls=[], result_local_path=src.name, name="R")
        s.add(run)
        s.flush()
        cap = RunCaption(run_id=run.id, status=CaptionStatus.done, output_path=src.name)
        s.add(cap)
        s.flush()
        # caption-sourced: its retry does not lock the Run, so only the
        # artifact's own row lock can make it wait.
        edit = RunTailEdit(run_id=run.id, cut_sec=1.0, status=TailEditStatus.done,
                           source_type="caption", source_caption_id=cap.id,
                           output_path=src.name)
        s.add(edit)
        s.commit()
        return run.id, cap.id, edit.id, src.name


def _promote(Session, rid, kind, sid, out):
    with Session() as s:
        try:
            out["resp"] = api_v2_module.promote_run_to_project(
                rid, project_type="cover_change", name=None, localisation_scope=None,
                mode="manual", autopilot_params=None, source_kind=kind, source_id=sid, db=s)
        except BaseException as exc:  # surfaced by the test
            out["exc"] = exc


def _run_concurrently(Session, held_copy, rid, kind, sid, contender):
    copying, release = held_copy
    a_out, b_out = {}, {}
    a = threading.Thread(target=_promote, args=(Session, rid, kind, sid, a_out))
    a.start()
    assert copying.wait(10), a_out

    def b_body():
        with Session() as s:
            try:
                b_out["result"] = contender(s)
                s.commit()
            except BaseException as exc:
                b_out["exc"] = exc
            b_out["done_at"] = time.monotonic()

    b = threading.Thread(target=b_body)
    b.start()
    time.sleep(1.5)
    assert "done_at" not in b_out, "the contender did not wait for the promote's row lock"
    released_at = time.monotonic()
    release.set()
    a.join(20)
    b.join(20)
    assert "exc" not in a_out, a_out.get("exc")
    assert b_out["done_at"] >= released_at
    return a_out, b_out


def test_tail_edit_retry_waits_for_the_promote(Session, held_copy):
    rid, _, edit_id, src = _fixture_rows(Session, b"tail-bytes")
    a_out, b_out = _run_concurrently(
        Session, held_copy, rid, "tail_edit", edit_id,
        lambda s: api_v2_module.retry_tail_edit(edit_id, db=s))
    assert "exc" not in b_out, b_out.get("exc")
    with Session() as s:
        p = s.get(VideoProject, a_out["resp"].project_id)
        assert (p.source_kind, p.source_artifact_id) == ("tail_edit", edit_id)
        assert s.get(RunTailEdit, edit_id).status == TailEditStatus.queued
    with open(p.source_local_path, "rb") as fh:
        assert fh.read() == b"tail-bytes"


def test_caption_delete_waits_for_the_promote(Session, held_copy, monkeypatch):
    rid, cap_id, _, _ = _fixture_rows(Session, b"caption-bytes")
    removed = []
    monkeypatch.setattr(api_v2_module.shutil, "rmtree",
                        lambda path, ignore_errors=False: removed.append(path))
    a_out, b_out = _run_concurrently(
        Session, held_copy, rid, "caption", cap_id,
        lambda s: api_v2_module.delete_caption(cap_id, db=s))
    assert "exc" not in b_out, b_out.get("exc")
    with Session() as s:
        p = s.get(VideoProject, a_out["resp"].project_id)
        assert s.get(RunCaption, cap_id) is None
    with open(p.source_local_path, "rb") as fh:
        assert fh.read() == b"caption-bytes"


def test_promote_after_a_committed_retry_is_409(Session, monkeypatch):
    monkeypatch.setattr(api_v2_module, "enqueue_process_tail_edit", lambda eid: None)
    rid, _, edit_id, _ = _fixture_rows(Session, b"x")
    with Session() as s:
        api_v2_module.retry_tail_edit(edit_id, db=s)
        s.commit()
    out = {}
    _promote(Session, rid, "tail_edit", edit_id, out)
    assert isinstance(out.get("exc"), HTTPException) and out["exc"].status_code == 409
