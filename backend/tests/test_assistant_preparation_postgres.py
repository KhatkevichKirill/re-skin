"""Real PostgreSQL preparation races in the explicitly disposable test DB.

Only native staging is stubbed with an owned typed media result. Admission,
transactions, ownership, selection, request receipts and byte pinning are real.
Native preservation has its own independent acceptance tests.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading

import pytest
from sqlalchemy import select

from app import assistant_media as media
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_models import (
    AssistantMessage, AssistantRunResultIntent, AssistantSelection, AssistantVersion,
)
from app.assistant_preparation import prepare_editable_version, prepared_child
from app.assistant_store import (
    ArtifactInput, AssistantConflict, get_version_artifact, import_run_version,
    publish_version, reserve_version, select_version,
)
from app.assistant_turns import AssistantTurnService, TurnConflict
from app.models import Run, VideoProject
from tests.test_assistant_postgres import pg_namespace, pg_store, _legacy, _plan, _output

pytestmark = pytest.mark.skipif(not os.environ.get("ASSISTANT_TEST_PG_URL"), reason="disposable PostgreSQL opt-in only")


@pytest.fixture
def preparation(pg_store):
    sessions, storage, data, engine = pg_store
    run_id, project_id, source = _legacy(pg_store)
    service = AssistantTurnService(storage=storage, policy=ProjectAllowlist(project_id), enabled=True)
    with sessions.begin() as session:
        version = import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage,
            full=ArtifactInput(source, metadata={"duration_sec": 20., "fps": "30", "has_audio": True,
                "subtitle_status": "unknown", "needs_editing_preparation": True,
                "editing_preparation": {"kind": "audio_tail", "added_frames": 3, "added_duration_sec": .1,
                    "target_duration_sec": 20.1, "audio_copied": True, "source_video_frames": 600,
                    "audio_duration_sec": 20.08, "audio_duration_ts": 885528, "audio_time_base": "1/44100",
                    "audio_overhang_sec": .08}}))
        version_id, conversation_id = version.id, version.conversation_id
    return dict(sessions=sessions, storage=storage, data=data, run=run_id, project=project_id,
                source=source, version=version_id, conversation=conversation_id, service=service)


def staged_result(source_path, destination_path, **kwargs):
    source, destination = Path(source_path), Path(destination_path)
    destination.write_bytes(b"owned-prepared-stage")
    info = media.MediaMetadata(20.1, 20.1, 160, 120, "30", "1:1", True)
    return media.MediaEditResult("audio_tail_preparation_v1",
        (media.SourceProvenance(str(source), hashlib.sha256(source.read_bytes()).hexdigest()),),
        str(destination), hashlib.sha256(destination.read_bytes()).hexdigest(), info, 20.1,
        (media.TimelineSpan(0, 20, 0),), "stream_copy_all_aac", "Unknown subtitle provenance retained.")


def invoke(c, *, key="pg-prepare", epoch=0):
    with c["sessions"]() as session:
        return prepare_editable_version(session, c["service"], c["run"], c["version"],
            expected_selection_epoch=epoch, idempotency_key=key)


def other_version(c):
    with c["sessions"].begin() as session:
        version = reserve_version(session, _plan(c["run"], c["version"]), idempotency_key="owned-other-edit", actor="operator")
        item = _output(c["data"], "owned-selection-change")
        publish_version(session, version.id, full=item, preview=item, storage=c["storage"])
        return version.id


def threaded_preparation(c, monkeypatch, *, epoch=0):
    staged, release = threading.Event(), threading.Event()

    def controlled(*args, **kwargs):
        result = staged_result(*args, **kwargs)
        staged.set()
        assert release.wait(timeout=10), "test never released native staging"
        return result

    monkeypatch.setattr("app.assistant_preparation.prepare_audio_tail", controlled)
    result, errors = [], []

    def execute():
        try:
            result.append(invoke(c, epoch=epoch).id)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert staged.wait(timeout=10), "preparation did not reach native staging"
    return thread, release, result, errors


def finish(thread, release):
    release.set()
    thread.join(timeout=20)
    assert not thread.is_alive(), "preparation transaction did not terminate"


@pytest.mark.parametrize("same_key", [True, False])
def test_concurrent_preparations_create_one_owned_child_with_real_row_locks(preparation, monkeypatch, same_key):
    c = preparation
    barrier = threading.Barrier(2)

    def controlled(*args, **kwargs):
        result = staged_result(*args, **kwargs)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr("app.assistant_preparation.prepare_audio_tail", controlled)
    result, errors = [], []

    def execute(index):
        try:
            result.append(invoke(c, key="pg-prepare" if same_key else f"pg-prepare-{index}").id)
        except BaseException as exc:
            errors.append(repr(exc))

    threads = [threading.Thread(target=execute, args=(i,), daemon=True) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=25)
        assert not thread.is_alive()
    assert errors == []
    assert len(result) == 2 and result[0] == result[1]
    with c["sessions"]() as session:
        assert len(session.scalars(select(AssistantVersion)).all()) == 2
        child = prepared_child(session, c["version"])
        assert child.id == result[0]
        assert (child.parent_version_id, child.base_version_id) == (c["version"], c["version"])
        assert get_version_artifact(session, child.id).owner_version_id == child.id
        selected = session.get(AssistantSelection, c["conversation"])
        assert (selected.selected_version_id, selected.epoch) == (child.id, 0)
        requests = session.scalars(select(AssistantMessage).where(AssistantMessage.idempotency_key.like("prepare-request:%"))).all()
        assert len(requests) == (1 if same_key else 2)
        assert session.get(Run, c["run"]).result_local_path == str(c["source"])


@pytest.mark.parametrize("change", ["away_and_back", "automatic_same_epoch"])
def test_selection_changes_during_staging_cannot_publish_stale_preparation(preparation, monkeypatch, change):
    c = preparation
    epoch = 0
    if change == "away_and_back":
        other = other_version(c)
        with c["sessions"].begin() as session:
            epoch = select_version(session, c["conversation"], c["version"], expected_epoch=0).epoch
    thread, release, result, errors = threaded_preparation(c, monkeypatch, epoch=epoch)
    try:
        if change == "away_and_back":
            with c["sessions"].begin() as session:
                selected = select_version(session, c["conversation"], other, expected_epoch=epoch)
                selected = select_version(session, c["conversation"], c["version"], expected_epoch=selected.epoch)
                expected = (c["version"], selected.epoch)
        else:
            other = other_version(c)
            expected = (other, 0)
    finally:
        finish(thread, release)
    assert result == [] and len(errors) == 1 and isinstance(errors[0], AssistantConflict)
    with c["sessions"]() as session:
        assert prepared_child(session, c["version"]) is None
        selected = session.get(AssistantSelection, c["conversation"])
        assert (selected.selected_version_id, selected.epoch) == expected


def test_durable_publication_intent_reserved_during_staging_blocks_finalizer(preparation, monkeypatch):
    from app.api_v2 import _lock_runs_for_result_admission
    c = preparation
    thread, release, result, errors = threaded_preparation(c, monkeypatch)
    try:
        with c["sessions"].begin() as session:
            _lock_runs_for_result_admission(session, [c["run"]])
            session.add(AssistantRunResultIntent(run_id=c["run"], conversation_id=c["conversation"],
                version_id=c["version"], artifact_id=get_version_artifact(session, c["version"]).id,
                idempotency_key="owned-concurrent-publication", request_sha256="a" * 64,
                actor="operator", request_json={}, before_json={}, after_json={}))
    finally:
        finish(thread, release)
    assert result == [] and len(errors) == 1 and isinstance(errors[0], AssistantConflict)
    with c["sessions"]() as session:
        assert prepared_child(session, c["version"]) is None
        assert session.scalar(select(AssistantRunResultIntent)).completed_at is None
        assert session.get(AssistantSelection, c["conversation"]).selected_version_id == c["version"]


@pytest.mark.parametrize("change", ["delete", "owner_change", "scope_revoked"])
def test_live_owner_and_scope_rechecked_after_unlocked_staging(preparation, monkeypatch, change):
    c = preparation
    if change == "owner_change":
        with c["sessions"].begin() as session:
            project = VideoProject(source_type="upload", source_ref="owned-other-project", status="ready")
            session.add(project)
            session.flush()
            other_project = project.id
        c["service"].policy = ProjectAllowlist([c["project"], other_project])
    thread, release, result, errors = threaded_preparation(c, monkeypatch)
    try:
        if change == "delete":
            from app.api_v2 import delete_run
            with c["sessions"]() as session:
                delete_run(c["run"], db=session)
        elif change == "owner_change":
            with c["sessions"].begin() as session:
                run = session.get(Run, c["run"])
                session.refresh(run, with_for_update=True)
                run.project_id = other_project
        else:
            c["service"].policy = ProjectAllowlist()
    finally:
        finish(thread, release)
    assert result == [] and len(errors) == 1
    assert isinstance(errors[0], (AssistantConflict, AssistantAccessDenied, TurnConflict))
    with c["sessions"]() as session:
        assert prepared_child(session, c["version"]) is None
        assert session.get(AssistantSelection, c["conversation"]).selected_version_id == c["version"]
