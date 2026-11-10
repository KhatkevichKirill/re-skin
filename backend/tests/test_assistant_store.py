"""Isolated assistant persistence, immutable pinning and selection race tests."""

from __future__ import annotations

import hashlib
import importlib.util
import errno
import os
import shutil
import stat
import threading
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, event, inspect, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Run, VideoProject
from app.assistant_contracts import EditPlan
from app.assistant_models import (
    AssistantArtifact, AssistantConversation, AssistantMessage, AssistantSelection,
    AssistantVersion, AssistantVersionArtifact,
)
from app.assistant_store import (
    ArtifactInput, ArtifactIntegrityError, AssistantArtifactStorage, AssistantConflict,
    AssistantStoreError, append_message, get_or_create_conversation,
    get_version_artifact, import_run_version, list_messages, list_versions,
    mark_version_terminal, publish_version, reserve_version, select_version,
    verify_artifact,
)


@pytest.fixture
def setup(tmp_path):
    # Explicit SQLite file and media root: no configured production engine/files.
    data = tmp_path / "data"
    data.mkdir()
    engine = create_engine(f"sqlite:///{tmp_path / 'isolated.db'}", connect_args={"timeout": 5})

    @event.listens_for(engine, "connect")
    def pragmas(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")

    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    storage = AssistantArtifactStorage(data)
    yield sessions, storage, data
    engine.dispose()


def legacy(sessions, data, *, content=b"original-video", duration=20):
    with sessions.begin() as session:
        project = VideoProject(source_type="upload", source_ref="fixture.mp4", status="ready")
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status="done", prompt="original prompt",
                  result_duration_sec=duration, result_gdrive_file_id="untouched-drive-id",
                  segment_snapshot=[{"id": str(uuid.uuid4()), "start_sec": 0, "end_sec": duration}])
        session.add(run)
        session.flush()
        directory = data / "projects" / project.id / "runs" / run.id
        directory.mkdir(parents=True)
        source = directory / "final.mp4"
        source.write_bytes(content)
        run.result_local_path = str(source)
        return run.id, project.id, source


def imported(setup, *, content=b"original-video"):
    sessions, storage, data = setup
    run_id, project_id, source = legacy(sessions, data, content=content)
    with sessions.begin() as session:
        version = import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage)
        return run_id, project_id, source, version.id, version.conversation_id


def plan(run_id, base_id, *, start=0, end=10):
    return EditPlan.model_validate({"run_id": run_id, "base_version_id": base_id,
                                   "operations": [{"kind": "trim", "start_sec": start, "end_sec": end}]})


def result(data, name, content=b"edited-video", duration=10):
    path = data / name
    path.write_bytes(content)
    return ArtifactInput(path, metadata={"duration_sec": duration})


def publish(session, version_id, storage, item):
    return publish_version(session, version_id, full=item, preview=item, storage=storage)


def test_import_pins_bytes_settings_and_provenance_without_touching_run(setup):
    sessions, storage, data = setup
    run_id, _, source, version_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        version = session.get(AssistantVersion, version_id)
        artifact = get_version_artifact(session, version_id)
        run = session.get(Run, run_id)
        assert version.status == "succeeded" and version.sequence == 1
        assert version.parent_version_id is None and version.base_version_id is None
        assert version.recipe["settings"]["prompt"] == "original prompt"
        assert version.recipe["legacy_drive_file_id"] == "untouched-drive-id"
        pinned = verify_artifact(artifact, storage=storage)
        assert pinned.read_bytes() == source.read_bytes()
        assert pinned.stat().st_ino != source.stat().st_ino
        assert artifact.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert source.read_bytes() == b"original-video"
        assert run.result_local_path == str(source)
        assert run.result_gdrive_file_id == "untouched-drive-id"
        selection = session.get(AssistantSelection, conversation_id)
        assert selection.selected_version_id == selection.latest_version_id == version_id
        assert selection.epoch == 0


def test_import_retry_keeps_original_snapshot_after_mutable_legacy_changes(setup):
    sessions, storage, data = setup
    run_id, _, source, version_id, _ = imported(setup)
    source.write_bytes(b"legacy-in-place-overwrite")
    with sessions.begin() as session:
        run = session.get(Run, run_id)
        run.prompt = "changed later"
    with sessions.begin() as session:
        version = import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage)
        assert version.id == version_id
        assert version.recipe["settings"]["prompt"] == "original prompt"
        assert verify_artifact(get_version_artifact(session, version_id), storage=storage).read_bytes() == b"original-video"


def test_legacy_deletion_cannot_cascade_assistant_history_or_pinned_media(setup):
    sessions, storage, data = setup
    run_id, project_id, source, version_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        session.delete(session.get(VideoProject, project_id))
    shutil.rmtree(data / "projects" / project_id)
    with sessions.begin() as session:
        assert session.get(Run, run_id) is None
        assert session.get(AssistantConversation, conversation_id).project_id == project_id
        assert verify_artifact(get_version_artifact(session, version_id), storage=storage).read_bytes() == b"original-video"


@pytest.mark.parametrize("status", ["created", "failed", "incomplete", "processing"])
def test_only_successful_run_imports(setup, status):
    sessions, storage, data = setup
    run_id, _, source = legacy(sessions, data)
    with sessions.begin() as session:
        session.get(Run, run_id).status = status
    with pytest.raises(AssistantStoreError), sessions.begin() as session:
        import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage)
    assert source.read_bytes() == b"original-video"
    with sessions() as session:
        assert session.query(AssistantVersion).count() == 0


def test_import_and_publish_honor_caller_rollback(setup):
    sessions, storage, data = setup
    run_id, _, source = legacy(sessions, data)
    with sessions() as session:
        version = import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage)
        version_id = version.id
        session.rollback()
    with sessions() as session:
        assert session.get(AssistantVersion, version_id) is None
        assert session.query(AssistantConversation).count() == 0
    with sessions.begin() as session:
        base = import_run_version(session, session.get(Run, run_id), actor="operator", storage=storage)
        candidate = reserve_version(session, plan(run_id, base.id), idempotency_key="rollback-candidate", actor="operator")
        candidate_id = candidate.id
    item = result(data, "candidate.mp4")
    with sessions() as session:
        publish(session, candidate_id, storage, item)
        session.rollback()
    with sessions() as session:
        assert session.get(AssistantVersion, candidate_id).status == "pending"
        assert session.query(AssistantVersionArtifact).filter_by(version_id=candidate_id, role="full").count() == 0
        assert session.get(AssistantSelection, base.conversation_id).latest_version_id == base.id
    assert source.read_bytes() == b"original-video"


def test_reserve_idempotency_frozen_plan_and_durable_branch_parent(setup):
    sessions, storage, data = setup
    run_id, _, _, base_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        accepted = plan(run_id, base_id)
        version = reserve_version(session, accepted, idempotency_key="request-1", actor="operator")
        again = reserve_version(session, accepted, idempotency_key="request-1", actor="operator")
        assert again.id == version.id
        assert version.parent_version_id == version.base_version_id == base_id
        assert version.sequence == 2
        assert session.get(AssistantConversation, conversation_id).next_version_sequence == 3
    with pytest.raises(AssistantConflict), sessions.begin() as session:
        reserve_version(session, plan(run_id, base_id, end=9), idempotency_key="request-1", actor="operator")
    with sessions.begin() as session:
        child = reserve_version(session, plan(run_id, base_id), idempotency_key="branch-older", actor="operator")
        assert child.sequence == 3 and child.parent_version_id == base_id
        assert [v.sequence for v in list_versions(session, conversation_id)] == [1, 2, 3]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_automatic_selection_reaches_highest_successful_sequence_in_both_orders(setup, order):
    sessions, storage, data = setup
    run_id, _, _, base_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        versions = [reserve_version(session, plan(run_id, base_id), idempotency_key=f"job-{i}", actor="operator").id
                    for i in range(2)]
    for index in order:
        item = result(data, f"output-{index}.mp4", content=f"edit-{index}".encode())
        with sessions.begin() as session:
            publish(session, versions[index], storage, item)
    with sessions() as session:
        selection = session.get(AssistantSelection, conversation_id)
        assert selection.latest_version_id == selection.selected_version_id == versions[1]
        assert selection.epoch == 0


def test_manual_selection_blocks_old_jobs_but_allows_new_branch_completion(setup):
    sessions, storage, data = setup
    run_id, _, _, base_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        old = reserve_version(session, plan(run_id, base_id), idempotency_key="old-job", actor="operator")
        old_id = old.id
        select_version(session, conversation_id, base_id, expected_epoch=0)
    with sessions.begin() as session:
        publish(session, old_id, storage, result(data, "old.mp4"))
        selection = session.get(AssistantSelection, conversation_id)
        assert selection.latest_version_id == old_id
        assert selection.selected_version_id == base_id and selection.epoch == 1
        new = reserve_version(session, plan(run_id, base_id), idempotency_key="new-branch", actor="operator")
        new_id = new.id
    with sessions.begin() as session:
        publish(session, new_id, storage, result(data, "new.mp4", b"new-result"))
        selection = session.get(AssistantSelection, conversation_id)
        assert selection.selected_version_id == selection.latest_version_id == new_id
        assert selection.epoch == 1


def test_stale_manual_selection_and_wrong_run_base_are_rejected(setup):
    sessions, storage, data = setup
    run_id, _, _, base_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        select_version(session, conversation_id, base_id, expected_epoch=0)
    with pytest.raises(AssistantConflict, match="selection changed"), sessions.begin() as session:
        select_version(session, conversation_id, base_id, expected_epoch=0)
    with pytest.raises(AssistantConflict), sessions.begin() as session:
        reserve_version(session, plan(str(uuid.uuid4()), base_id), idempotency_key="bad-run", actor="operator")


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_cancelled_versions_never_advance_and_cannot_publish(setup, status):
    sessions, storage, data = setup
    run_id, _, _, base_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        candidate = reserve_version(session, plan(run_id, base_id), idempotency_key="terminal", actor="operator")
        candidate_id = candidate.id
        mark_version_terminal(session, candidate_id, status=status)
        mark_version_terminal(session, candidate_id, status=status)
    with pytest.raises(AssistantConflict), sessions.begin() as session:
        publish(session, candidate_id, storage, result(data, "never-published.mp4"))
    with sessions() as session:
        assert session.get(AssistantSelection, conversation_id).latest_version_id == base_id


def test_repeated_publish_is_idempotent_but_different_bytes_or_metadata_conflict(setup):
    sessions, storage, data = setup
    run_id, _, source, base_id, _ = imported(setup)
    with sessions.begin() as session:
        candidate = reserve_version(session, plan(run_id, base_id), idempotency_key="publish", actor="operator")
        version_id = candidate.id
    item = result(data, "result.mp4")
    with sessions.begin() as session:
        publish(session, version_id, storage, item)
        first_ids = [get_version_artifact(session, version_id, role=role).id for role in ("full", "preview")]
    with sessions.begin() as session:
        publish(session, version_id, storage, item)
        assert [get_version_artifact(session, version_id, role=role).id for role in ("full", "preview")] == first_ids
    with pytest.raises(AssistantConflict), sessions.begin() as session:
        publish(session, version_id, storage, ArtifactInput(item.source_path, metadata={"duration_sec": 9}))
    item.source_path.write_bytes(b"different-result")
    with pytest.raises(AssistantConflict), sessions.begin() as session:
        publish(session, version_id, storage, item)
    assert source.read_bytes() == b"original-video"
    with sessions() as session:
        assert verify_artifact(get_version_artifact(session, version_id), storage=storage).read_bytes() == b"edited-video"


def test_thumbnail_cannot_substitute_for_completed_playable_preview(setup):
    sessions, storage, data = setup
    run_id, _, _, base_id, _ = imported(setup)
    with sessions.begin() as session:
        version = reserve_version(session, plan(run_id, base_id), idempotency_key="preview", actor="operator")
        version_id = version.id
    item = result(data, "full.mp4")
    thumbnail = ArtifactInput(item.source_path, media_kind="image", metadata={})
    with pytest.raises(AssistantStoreError, match="playable video"), sessions.begin() as session:
        publish_version(session, version_id, full=item, preview=thumbnail, storage=storage)
    with sessions() as session:
        assert session.get(AssistantVersion, version_id).status == "pending"


def test_message_order_idempotency_context_and_selected_base_survive_reopen(setup):
    sessions, storage, data = setup
    _, _, _, base_id, conversation_id = imported(setup)
    context = {"settings": {"language": "en"}, "assets": ["authorized-id"]}
    with sessions.begin() as session:
        first = append_message(session, conversation_id, idempotency_key="message-1", actor="operator",
                               role="operator", content="Trim the ending", base_version_id=base_id,
                               context_snapshot=context)
        assert append_message(session, conversation_id, idempotency_key="message-1", actor="operator",
                              role="operator", content="Trim the ending", base_version_id=base_id,
                              context_snapshot=context).id == first.id
        append_message(session, conversation_id, idempotency_key="message-2", actor="assistant",
                       role="assistant", content="Plan is ready", base_version_id=base_id)
    context["settings"]["language"] = "changed"
    with sessions() as session:
        messages = list_messages(session, conversation_id)
        assert [m.sequence for m in messages] == [1, 2]
        assert messages[0].context_snapshot["settings"]["language"] == "en"
        assert messages[0].base_version_id == base_id
    with pytest.raises(AssistantConflict), sessions.begin() as session:
        append_message(session, conversation_id, idempotency_key="message-1", actor="operator",
                       role="operator", content="Different message", base_version_id=base_id)


def test_cross_project_donor_binding_retains_source_and_requires_exact_version(setup):
    sessions, storage, data = setup
    run_id, _, _, base_id, _ = imported(setup)
    _, donor_project, _, donor_id, _ = imported(setup, content=b"donor-video")
    with sessions.begin() as session:
        donor_asset = get_version_artifact(session, donor_id)
        asset_id = donor_asset.id
        donor_plan = EditPlan.model_validate({"run_id": run_id, "base_version_id": base_id,
            "operations": [{"kind": "audio_splice", "start_sec": 1, "end_sec": 2,
                            "source": {"asset_id": asset_id, "version_id": donor_id}}]})
        candidate = reserve_version(session, donor_plan, idempotency_key="donor", actor="operator")
        assert session.query(AssistantVersionArtifact).filter_by(version_id=candidate.id, artifact_id=asset_id, role="input").count() == 1
    with sessions.begin() as session:
        session.delete(session.get(VideoProject, donor_project))
    with pytest.raises(IntegrityError), sessions.begin() as session:
        session.execute(delete(AssistantArtifact).where(AssistantArtifact.id == asset_id))
    with pytest.raises(AssistantConflict), sessions.begin() as session:
        wrong = donor_plan.model_dump()
        wrong["operations"][0]["source"]["version_id"] = base_id
        reserve_version(session, EditPlan.model_validate(wrong), idempotency_key="wrong-donor", actor="operator")


def test_semantic_records_are_immutable_for_orm_and_direct_sql(setup):
    sessions, storage, data = setup
    _, _, _, base_id, conversation_id = imported(setup)
    with pytest.raises(ValueError, match="immutable"), sessions.begin() as session:
        session.get(AssistantVersion, base_id).recipe = {"tampered": True}
    with pytest.raises(IntegrityError, match="immutable"), sessions.begin() as session:
        session.execute(update(AssistantVersion).where(AssistantVersion.id == base_id).values(recipe={"tampered": True}))
    with pytest.raises(IntegrityError, match="immutable"), sessions.begin() as session:
        session.execute(update(AssistantVersion).where(AssistantVersion.id == base_id).values(status="pending"))
    with sessions() as session:
        artifact_id = get_version_artifact(session, base_id).id
    with pytest.raises(IntegrityError, match="immutable"), sessions.begin() as session:
        session.execute(update(AssistantArtifact).where(AssistantArtifact.id == artifact_id).values(storage_key="elsewhere"))
    with pytest.raises(IntegrityError, match="immutable"), sessions.begin() as session:
        session.execute(update(AssistantConversation).where(AssistantConversation.id == conversation_id)
                        .values(project_id=str(uuid.uuid4())))


def test_sql_constraints_prevent_cross_conversation_selection_and_invalid_parent(setup):
    sessions, storage, data = setup
    _, _, _, base_id, conversation_id = imported(setup)
    _, _, _, other_id, _ = imported(setup)
    with pytest.raises(IntegrityError), sessions.begin() as session:
        session.execute(update(AssistantSelection).where(AssistantSelection.conversation_id == conversation_id)
                        .values(selected_version_id=other_id))
    with pytest.raises(IntegrityError), sessions.begin() as session:
        session.add(AssistantVersion(id=str(uuid.uuid4()), conversation_id=conversation_id, sequence=2,
            parent_version_id=other_id, base_version_id=other_id, selection_epoch=0,
            idempotency_key="cross-parent", request_sha256="a"*64, actor="operator", recipe={}, status="pending"))


def test_pin_no_clobber_corruption_and_unsafe_sources(setup, tmp_path):
    _, storage, data = setup
    source = data / "source.mp4"
    source.write_bytes(b"source-bytes")
    pinned = storage.pin(source)
    path = storage.verify(pinned)
    assert storage.pin(source) == pinned
    path.chmod(0o600)
    path.write_bytes(b"corrupt-file")
    with pytest.raises(ArtifactIntegrityError):
        storage.pin(source)
    assert path.read_bytes() == b"corrupt-file"  # Never overwrite an existing object.
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private")
    with pytest.raises(AssistantStoreError):
        storage.pin(outside)
    link = data / "outside-link"
    link.symlink_to(outside)
    with pytest.raises(AssistantStoreError):
        storage.pin(link)
    assert list((storage.root / "staging").iterdir()) == []


def test_pin_rejects_size_limit_empty_and_symlink_objects(setup):
    _, storage, data = setup
    source = data / "source.mp4"
    source.write_bytes(b"three")
    with pytest.raises(AssistantStoreError):
        AssistantArtifactStorage(data, max_artifact_bytes=2).pin(source)
    source.write_bytes(b"")
    with pytest.raises(AssistantStoreError):
        storage.pin(source)
    source.write_bytes(b"bytes")
    pinned = storage.pin(source)
    path = storage.verify(pinned)
    path.unlink()
    path.symlink_to(source)
    with pytest.raises(ArtifactIntegrityError):
        storage.pin(source)
    with pytest.raises(ArtifactIntegrityError):
        storage.path_for("../../secrets")


def _fsync_observer(monkeypatch, events, *, fail_directory=None):
    original = os.fsync

    def sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            events.append(("directory_sync", directory))
            if directory == fail_directory:
                raise OSError(errno.EIO, "simulated directory fsync failure")
        else:
            events.append(("file_sync", None))
        return original(descriptor)

    monkeypatch.setattr(os, "fsync", sync)


@pytest.mark.parametrize("preexisting", [False, True])
def test_storage_directory_entries_are_durable_even_after_competing_creation(tmp_path, monkeypatch, preexisting):
    data = tmp_path / "data"
    data.mkdir()
    if preexisting:
        AssistantArtifactStorage(data)
    events = []
    _fsync_observer(monkeypatch, events)
    storage = AssistantArtifactStorage(data)
    # Persist assistant_artifacts in DATA_DIR, then objects and staging in the
    # assistant root. FileExists cannot stand in for the other creator's fsync.
    assert events == [("directory_sync", data), ("directory_sync", storage.root),
                      ("directory_sync", storage.root)]


@pytest.mark.parametrize("collision", ["new", "existing", "visible_before_creator_sync"])
def test_pin_orders_file_parent_and_object_directory_sync_before_return(setup, monkeypatch, collision):
    _, storage, data = setup
    source = data / "durability.mp4"
    source.write_bytes(b"durable-source")
    if collision == "existing":
        storage.pin(source)
    events = []
    _fsync_observer(monkeypatch, events)
    original_link, original_verify = os.link, storage.verify

    def link(source_name, target_name, **kwargs):
        events.append(("link", Path(target_name)))
        if collision == "visible_before_creator_sync":
            # Deterministically model another pinner making its already-fsynced
            # inode visible, then pausing before syncing the object directory.
            original_link(source_name, target_name, **kwargs)
            raise FileExistsError(errno.EEXIST, "concurrent link is visible")
        return original_link(source_name, target_name, **kwargs)

    def verify(pinned):
        events.append(("verify_existing", None))
        return original_verify(pinned)

    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(storage, "verify", verify)
    pinned = storage.pin(source)
    path = storage.path_for(pinned.storage_key)
    expected = [("file_sync", None), ("directory_sync", storage.root / "objects"), ("link", path)]
    if collision != "new":
        expected.append(("verify_existing", None))
    expected.append(("directory_sync", path.parent))
    assert events == expected
    assert list((storage.root / "staging").iterdir()) == []


def test_storage_creation_propagates_parent_sync_failure(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    events = []
    _fsync_observer(monkeypatch, events, fail_directory=data)
    with pytest.raises(OSError, match="directory fsync failure"):
        AssistantArtifactStorage(data)
    assert events == [("directory_sync", data)]
    assert not (data / "assistant_artifacts" / "objects").exists()


@pytest.mark.parametrize("phase", ["prefix_parent", "object_entry", "existing_object_entry"])
def test_directory_sync_error_cannot_publish_a_successful_version(setup, monkeypatch, phase):
    sessions, storage, data = setup
    run_id, _, original, base_id, conversation_id = imported(setup)
    with sessions.begin() as session:
        candidate = reserve_version(session, plan(run_id, base_id), idempotency_key="fsync-error", actor="operator")
        version_id = candidate.id
    item = result(data, "durability-result.mp4", b"new-durable-result")
    digest = hashlib.sha256(item.source_path.read_bytes()).hexdigest()
    folder = storage.root / "objects" / digest[:2]
    if phase == "existing_object_entry":
        storage.pin(item.source_path)
    failure_directory = storage.root / "objects" if phase == "prefix_parent" else folder
    events = []
    with monkeypatch.context() as patch:
        _fsync_observer(patch, events, fail_directory=failure_directory)
        with pytest.raises(OSError, match="directory fsync failure"), sessions.begin() as session:
            publish(session, version_id, storage, item)
    assert ("directory_sync", failure_directory) in events
    with sessions() as session:
        assert session.get(AssistantVersion, version_id).status == "pending"
        assert session.get(AssistantSelection, conversation_id).latest_version_id == base_id
        assert session.query(AssistantVersionArtifact).filter_by(version_id=version_id, role="full").count() == 0
    assert original.read_bytes() == b"original-video"
    assert list((storage.root / "staging").iterdir()) == []
    # A retry can reuse an object left by the failed sync, but must establish
    # directory durability itself before publishing the same pending version.
    with sessions.begin() as session:
        publish(session, version_id, storage, item)
        assert session.get(AssistantVersion, version_id).status == "succeeded"


def test_source_change_during_pin_is_detected_and_never_published(setup, monkeypatch):
    from app import assistant_store
    _, storage, data = setup
    source = data / "changing.mp4"
    source.write_bytes(b"before")
    original_fstat = assistant_store.os.fstat
    calls = 0

    def change_after_copy(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            source.write_bytes(b"changed-during-pin")
        return original_fstat(descriptor)

    monkeypatch.setattr(assistant_store.os, "fstat", change_after_copy)
    with pytest.raises(AssistantConflict, match="source changed"):
        storage.pin(source)
    assert list((storage.root / "staging").iterdir()) == []
    assert not list((storage.root / "objects").rglob("?" * 64))


def test_missing_object_and_legacy_tree_symlink_root_fail_closed(setup):
    _, storage, data = setup
    source = data / "missing-later.mp4"
    source.write_bytes(b"retained")
    pinned = storage.pin(source)
    path = storage.verify(pinned)
    shutil.rmtree(path.parent)
    with pytest.raises(ArtifactIntegrityError, match="unavailable"):
        storage.verify(pinned)
    shutil.rmtree(storage.root)
    legacy_tree = data / "projects"
    legacy_tree.mkdir(exist_ok=True)
    storage.root.symlink_to(legacy_tree, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        AssistantArtifactStorage(data)


def test_concurrent_reservations_allocate_unique_sequences_and_deduplicate_keys(setup):
    sessions, storage, data = setup
    run_id, _, _, base_id, conversation_id = imported(setup)
    barrier = threading.Barrier(2)
    versions, failures = [], []

    def reserve(key):
        try:
            with sessions.begin() as session:
                barrier.wait(timeout=3)
                version = reserve_version(session, plan(run_id, base_id), idempotency_key=key, actor="operator")
                versions.append(version.id)
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=reserve, args=("same-request",)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
        assert not thread.is_alive()
    assert not failures
    assert len(set(versions)) == 1
    with sessions() as session:
        assert [v.sequence for v in list_versions(session, conversation_id)] == [1, 2]


def test_additive_migration_matches_registered_schema_on_isolated_sqlite(tmp_path, monkeypatch):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = Path(__file__).parents[1] / "alembic/versions/c4a9e2f7b601_add_assistant_history.py"
    spec = importlib.util.spec_from_file_location("assistant_history_migration_test", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine(f"sqlite:///{tmp_path / 'migration-only.db'}")
    with engine.begin() as connection:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        inspector = inspect(connection)
        names = {"assistant_conversations", "assistant_versions", "assistant_selections",
                 "assistant_artifacts", "assistant_version_artifacts", "assistant_messages"}
        assert set(inspector.get_table_names()) == names
        for name in names:
            actual = {c["name"]: c for c in inspector.get_columns(name)}
            expected = Base.metadata.tables[name]
            assert set(actual) == set(expected.columns.keys())
            assert all(actual[c.name]["nullable"] == c.nullable for c in expected.columns)
            assert {c["name"] for c in inspector.get_unique_constraints(name)} == {
                c.name for c in expected.constraints if c.__class__.__name__ == "UniqueConstraint"
            }
            assert {c["name"] for c in inspector.get_check_constraints(name)} == {
                c.name for c in expected.constraints if c.__class__.__name__ == "CheckConstraint"
            }
            assert {i["name"] for i in inspector.get_indexes(name)} == {i.name for i in expected.indexes}
        conversation_id, version_id = str(uuid.uuid4()), str(uuid.uuid4())
        connection.execute(text("INSERT INTO assistant_conversations (id,run_id,project_id) VALUES (:id,:run,:project)"),
                           {"id": conversation_id, "run": str(uuid.uuid4()), "project": str(uuid.uuid4())})
        connection.execute(text("INSERT INTO assistant_versions "
                                "(id,conversation_id,sequence,selection_epoch,idempotency_key,request_sha256,actor,recipe) "
                                "VALUES (:id,:conversation,1,0,'migration-test',:digest,'operator','{}')"),
                           {"id": version_id, "conversation": conversation_id, "digest": "a" * 64})
        # An unchanged value is allowed; a semantic change is rejected.
        connection.execute(text("UPDATE assistant_versions SET recipe='{}' WHERE id=:id"), {"id": version_id})
        with pytest.raises(IntegrityError, match="immutable"):
            connection.execute(text("UPDATE assistant_versions SET recipe=:recipe WHERE id=:id"),
                               {"id": version_id, "recipe": '{"changed":true}'})
        connection.execute(text("DELETE FROM assistant_versions WHERE id=:id"), {"id": version_id})
        connection.execute(text("DELETE FROM assistant_conversations WHERE id=:id"), {"id": conversation_id})
        migration.downgrade()
        assert inspect(connection).get_table_names() == []
    engine.dispose()
