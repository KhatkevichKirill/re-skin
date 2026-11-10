"""Opt-in real PostgreSQL races; only the disposable assistant_test DB."""
import threading
import os
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.assistant_access import ProjectAllowlist
from app.assistant_models import AssistantRunResultIntent, AssistantRunResultPromotion
from app.assistant_run_result import promote_run_result, run_result_context, ensure_no_unresolved_intent
from app.assistant_store import AssistantConflict, import_run_version, publish_version, reserve_version
from app.assistant_turns import AssistantTurnService
from app.models import Run
from tests.test_assistant_postgres import pg_namespace, pg_store, _baseline, _plan, _output
from tests.test_assistant_run_result import Drive

pytestmark = pytest.mark.skipif(not os.environ.get("ASSISTANT_TEST_PG_URL"), reason="isolated PostgreSQL opt-in only")


@pytest.fixture
def publication(pg_store, monkeypatch):
    sessions, storage, data, engine = pg_store
    run_id, project_id, original, base_id, conversation_id = _baseline(pg_store)
    service = AssistantTurnService(storage=storage, policy=ProjectAllowlist(project_id), enabled=True)
    with sessions.begin() as session:
        version = reserve_version(session, _plan(run_id, base_id), idempotency_key="pg-edit", actor="operator")
        output = _output(data, "pg-main")
        publish_version(session, version.id, full=output, preview=output, storage=storage)
        version_id = version.id
    monkeypatch.setattr("app.assistant_run_result._probe", lambda path, budget: 10.0 if Path(path).read_bytes().startswith(b"isolated-edited") else 20.0)
    with sessions() as session:
        revision = run_result_context(session, service, run_id)["revision"]
    request = {"version_id": version_id, "expected_selection_epoch": 0,
        "expected_result_revision": revision, "idempotency_key": "pg-publish"}
    return sessions, storage, service, run_id, request


def test_same_key_concurrent_promotion_uploads_once_and_replays_one_receipt(publication):
    sessions, storage, service, run_id, request = publication
    drive = Drive(sessions)
    barrier = threading.Barrier(2)
    result, errors = [], []
    def execute():
        try:
            with sessions() as session:
                barrier.wait(timeout=5)
                result.append(promote_run_result(session, service, run_id, request, drive=drive))
        except BaseException as exc:
            errors.append(repr(exc))
    threads = [threading.Thread(target=execute, daemon=True) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert errors == []
    assert len(result) == 2 and result[0]["id"] == result[1]["id"]
    assert drive.begins == drive.finishes == 1
    with sessions() as session:
        assert len(session.scalars(select(AssistantRunResultPromotion)).all()) == 1


def test_unresolved_intent_blocks_legacy_writer_under_real_run_lock(publication):
    sessions, storage, service, run_id, request = publication
    drive = Drive(sessions)
    drive.fail = True
    from app.gdrive_client import GDriveUploadError
    with sessions() as session:
        with pytest.raises(GDriveUploadError):
            promote_run_result(session, service, run_id, request, drive=drive)
    with sessions.begin() as session:
        run = session.get(Run, run_id)
        session.refresh(run, with_for_update=True)
        with pytest.raises(AssistantConflict):
            ensure_no_unresolved_intent(session, run_id)
        assert session.scalar(select(AssistantRunResultIntent)).completed_at is None
    drive.fail = False
    with sessions() as session:
        receipt = promote_run_result(session, service, run_id, request, drive=drive)
        assert receipt["delivery"] == "drive_updated"
    assert drive.begins == 1


@pytest.mark.parametrize("kind", ["audio", "range"])
def test_legacy_result_job_admitted_during_preparation_prevents_publication(publication, monkeypatch, kind):
    from app import assistant_run_result
    from app.models import RunAudioPatch, RunSegmentRangeAttempt, RunSegment
    from app.api_v2 import _lock_runs_for_result_admission
    import uuid
    sessions, storage, service, run_id, request = publication
    prepared, release = threading.Event(), threading.Event()
    original_copy = assistant_run_result._owned_copy
    def prepare(*args, **kwargs):
        value = original_copy(*args, **kwargs)
        prepared.set()
        assert release.wait(timeout=10)
        return value
    monkeypatch.setattr(assistant_run_result, "_owned_copy", prepare)
    drive, outcomes = Drive(sessions), []
    def promote():
        with sessions() as session:
            try:
                promote_run_result(session, service, run_id, request, drive=drive)
                outcomes.append("unexpected success")
            except AssistantConflict:
                outcomes.append("conflict")
    thread = threading.Thread(target=promote, daemon=True)
    thread.start()
    assert prepared.wait(timeout=10)
    try:
        with sessions.begin() as session:
            run = _lock_runs_for_result_admission(session, [run_id])[run_id]
            if kind == "audio":
                session.add(RunAudioPatch(run_id=run_id, donor_run_id=run_id, target_start_sec=0,
                    target_end_sec=1, donor_start_sec=0, crossfade_sec=0.01,
                    base_result_path=run.result_local_path, status="queued"))
            else:
                segment = RunSegment(run_id=run_id, segment_def_id=str(uuid.uuid4()), index=0, status="completed")
                session.add(segment)
                session.flush()
                session.add(RunSegmentRangeAttempt(run_id=run_id, run_segment_id=segment.id,
                    requested_start_sec=0, requested_end_sec=1, base_segment_snapshot=[],
                    candidate_segment_snapshot=[], status="generating"))
    finally:
        release.set()
    thread.join(timeout=20)
    assert not thread.is_alive() and outcomes == ["conflict"]
    assert drive.begins == drive.finishes == 0
    with sessions() as session:
        assert session.scalar(select(AssistantRunResultIntent)) is None


def test_unresolved_intent_blocks_actual_admission_and_run_project_deletion(publication):
    from fastapi import HTTPException
    from app.api_v2 import _lock_runs_for_result_admission, delete_run, delete_project
    from app.gdrive_client import GDriveUploadError
    sessions, storage, service, run_id, request = publication
    drive = Drive(sessions)
    drive.fail = True
    with sessions() as session:
        with pytest.raises(GDriveUploadError):
            promote_run_result(session, service, run_id, request, drive=drive)
    for action in (lambda session: _lock_runs_for_result_admission(session, [run_id]),
                   lambda session: delete_run(run_id, db=session),
                   lambda session: delete_project(session.get(Run, run_id).project_id, db=session)):
        with sessions() as session:
            with pytest.raises(HTTPException) as error:
                action(session)
            assert error.value.status_code == 409
            assert session.get(Run, run_id) is not None
