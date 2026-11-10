"""Main-result publication with isolated SQLite/files and a stateful fake Drive."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from app.assistant_access import ProjectAllowlist
from app.assistant_models import (AssistantRunResultIntent, AssistantRunResultPromotion,
    AssistantSelection, AssistantVersion, AssistantVersionArtifact)
from app.assistant_store import (AssistantConflict, ArtifactInput, get_version_artifact,
    import_run_version, publish_version, reserve_version, select_version)
from app.assistant_turns import AssistantTurnService
from app.assistant_run_result import (promote_run_result, run_result_context,
    ensure_no_unresolved_intent, ensure_segment_reconstruction_allowed, history_media)
from app.db import get_db
from app.models import Run, OverlayJob
from app.gdrive_client import GDriveUploadError
from tests.test_assistant_store import setup, legacy, plan


class Drive:
    def __init__(self, sessions):
        self.sessions = sessions
        self.begins = 0
        self.finishes = 0
        self.fail = False
        self.after_transfer = None
        self.remote = None

    def begin_result_update(self, file_id, path, *, deadline):
        self.begins += 1
        with self.sessions() as session:
            assert session.scalar(select(AssistantRunResultIntent)).resumable_uri is None
        return "https://www.googleapis.com/upload/drive/v3/files/fixture?upload_id=private-do-not-expose"

    def finish_result_update(self, uri, file_id, path, **kwargs):
        self.finishes += 1
        with self.sessions() as session:
            assert session.scalar(select(AssistantRunResultIntent)).resumable_uri == uri
            assert session.scalar(select(AssistantRunResultIntent)).completed_at is None
        self.remote = Path(path).read_bytes()
        if self.fail:
            raise GDriveUploadError("private-do-not-expose")
        if self.after_transfer:
            callback, self.after_transfer = self.after_transfer, None
            callback()
        return {"id": file_id}


@pytest.fixture
def case(setup, monkeypatch):
    sessions, storage, data = setup
    run_id, project_id, original = legacy(sessions, data)
    svc = AssistantTurnService(storage=storage, policy=ProjectAllowlist(project_id), enabled=True)
    with sessions.begin() as session:
        run = session.get(Run, run_id)
        base = import_run_version(session, run, actor="operator", storage=storage)
        base_id = base.id
        child = reserve_version(session, plan(run_id, base.id), idempotency_key="edit", actor="operator")
        edited = data / "edited.mp4"
        edited.write_bytes(b"shorter-edited-full-video")
        preview = data / "preview.mp4"
        preview.write_bytes(b"different-preview-only")
        publish_version(session, child.id, full=ArtifactInput(edited, metadata={"duration_sec": 10}),
            preview=ArtifactInput(preview, metadata={"duration_sec": 2}), storage=storage)
        version_id = child.id
    from app import assistant_run_result as module
    monkeypatch.setattr(module, "_probe", lambda path, budget: 10.0 if Path(path).read_bytes() == edited.read_bytes() else 20.0)
    drive = Drive(sessions)
    with sessions() as session:
        context = run_result_context(session, svc, run_id)
        selection = session.get(AssistantSelection, base.conversation_id)
        request = {"version_id": version_id, "expected_selection_epoch": selection.epoch,
            "expected_result_revision": context["revision"], "idempotency_key": "publish-1"}
    return SimpleNamespace(sessions=sessions, storage=storage, data=data, run_id=run_id,
        project_id=project_id, original=original, base_id=base_id, version_id=version_id,
        conversation_id=base.conversation_id, service=svc, drive=drive, request=request)


def publish(c, request=None):
    with c.sessions() as session:
        return promote_run_result(session, c.service, c.run_id, request or c.request, drive=c.drive)


def test_full_result_history_integrity_and_idempotent_stale_replay(case):
    c = case
    original = c.original.read_bytes()
    with c.sessions() as session:
        before = session.get(Run, c.run_id)
        preserved = (before.id, before.prompt, before.segment_snapshot, before.result_gdrive_file_id, before.final_creatives_exported_at)
    receipt = publish(c)
    assert receipt["delivery"] == "drive_updated" and receipt["is_current"]
    assert c.drive.remote == b"shorter-edited-full-video"
    assert c.drive.begins == 1
    with c.sessions() as session:
        run = session.get(Run, c.run_id)
        assert (run.id, run.prompt, run.segment_snapshot, run.result_gdrive_file_id, run.final_creatives_exported_at) == preserved
        assert run.result_duration_sec == 10 and Path(run.result_local_path).read_bytes() == c.drive.remote
        assert os.stat(run.result_local_path).st_mode & 0o444 == 0o444
        assert c.storage.root not in Path(run.result_local_path).parents
        assert history_media(session, c.service, c.run_id, receipt["id"], "before").read_bytes() == original
        assert history_media(session, c.service, c.run_id, receipt["id"], "after").read_bytes() == c.drive.remote
        assert c.original.read_bytes() == original
        assert run_result_context(session, c.service, c.run_id)["version_id"] == c.version_id
        with pytest.raises(AssistantConflict):
            ensure_segment_reconstruction_allowed(session, c.run_id)
    # Selection changes cannot defeat idempotent receipt lookup.
    with c.sessions.begin() as session:
        select_version(session, c.conversation_id, c.base_id, expected_epoch=c.request["expected_selection_epoch"])
    assert publish(c)["id"] == receipt["id"]
    assert c.drive.finishes == 1
    with pytest.raises(AssistantConflict):
        publish(c, {**c.request, "version_id": c.base_id})


def test_local_only_does_not_create_drive_file(case):
    c = case
    with c.sessions.begin() as session:
        session.get(Run, c.run_id).result_gdrive_file_id = None
    with c.sessions() as session:
        c.request["expected_result_revision"] = run_result_context(session, c.service, c.run_id)["revision"]
    receipt = publish(c)
    assert receipt["delivery"] == "local_only"
    assert c.drive.begins == c.drive.finishes == 0


def test_unknown_drive_outcome_keeps_fence_and_exact_session_recovers_after_selection_change(case):
    c = case
    c.drive.fail = True
    with pytest.raises(GDriveUploadError):
        publish(c)
    with c.sessions.begin() as session:
        run = session.get(Run, c.run_id)
        assert run.result_local_path == str(c.original) and run.result_duration_sec == 20
        pending = run_result_context(session, c.service, c.run_id)["pending"]
        assert pending["idempotency_key"] == c.request["idempotency_key"]
        assert "private-do-not-expose" not in json.dumps(pending)
        with pytest.raises(AssistantConflict):
            ensure_no_unresolved_intent(session, c.run_id)
        select_version(session, c.conversation_id, c.base_id, expected_epoch=c.request["expected_selection_epoch"])
    c.drive.fail = False
    receipt = publish(c)
    assert receipt["version_id"] == c.version_id and c.drive.begins == 1 and c.drive.finishes == 2
    with c.sessions() as session:
        assert run_result_context(session, c.service, c.run_id)["pending"] is None


@pytest.mark.parametrize("change", ["selection", "result", "overlay", "bad_full"])
def test_refuses_stale_or_conflicting_work_before_drive(case, change):
    c = case
    with c.sessions.begin() as session:
        if change == "selection":
            select_version(session, c.conversation_id, c.base_id, expected_epoch=c.request["expected_selection_epoch"])
        elif change == "result":
            session.get(Run, c.run_id).result_gdrive_file_id = "moved-drive"
        elif change == "overlay":
            session.add(OverlayJob(source_type="run", source_run_id=c.run_id, status="created"))
        else:
            artifact = get_version_artifact(session, c.version_id)
            path = c.storage.path_for(artifact.storage_key)
            path.chmod(0o600)
            path.write_bytes(b"tampered")
    with pytest.raises((AssistantConflict, ValueError)):
        publish(c)
    assert c.drive.begins == c.drive.finishes == 0
    with c.sessions() as session:
        assert session.scalar(select(AssistantRunResultIntent)) is None


def test_reused_same_run_full_artifact_is_valid(case):
    c = case
    with c.sessions.begin() as session:
        previous = session.get(AssistantVersion, c.version_id)
        reused = reserve_version(session, plan(c.run_id, c.version_id), idempotency_key="reuse", actor="operator")
        full = get_version_artifact(session, previous.id)
        session.add(AssistantVersionArtifact(version_id=reused.id, artifact_id=full.id, role="full", slot=0))
        session.add(AssistantVersionArtifact(version_id=reused.id, artifact_id=full.id, role="preview", slot=0))
        reused.status = "succeeded"
        selection = session.get(AssistantSelection, c.conversation_id)
        selection.selected_version_id = reused.id
        c.request["version_id"] = reused.id
    assert publish(c)["version_id"] == c.request["version_id"]


def test_intent_receipt_sql_guards_block_mutation_and_deletion(case):
    c = case
    receipt = publish(c)
    with c.sessions() as session:
        with pytest.raises(IntegrityError):
            session.execute(update(AssistantRunResultPromotion).values(delivery="local_only"))
    with c.sessions() as session:
        with pytest.raises(IntegrityError):
            session.execute(update(AssistantRunResultIntent).values(resumable_uri="different"))


def test_api_receipt_history_private_and_reauthorized(case, monkeypatch):
    c = case
    from app.api_assistant import router, get_assistant_service
    from app import assistant_run_result
    monkeypatch.setattr("app.gdrive_client.GDriveClient", lambda: c.drive)
    app = FastAPI()
    app.include_router(router, prefix="/api/v2")
    def db():
        with c.sessions() as session:
            yield session
            session.commit()
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_assistant_service] = lambda: c.service
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            root = f"/api/v2/assistant/runs/{c.run_id}"
            response = await client.post(root + "/result", json=c.request)
            assert response.status_code == 200, response.text
            payload = response.json()
            assert str(c.data) not in response.text and "private-do-not-expose" not in response.text
            assert (await client.get(payload["before_url"])).content == c.original.read_bytes()
            assert (await client.get(payload["full_url"])).content == c.drive.remote
            c.service.policy = ProjectAllowlist()
            assert (await client.get(payload["before_url"])).status_code == 404
            assert (await client.post(root + "/result", json=c.request)).status_code == 404
    asyncio.run(exercise())


def test_receipt_cannot_forge_intent_version_or_request_binding(case):
    c = case
    c.drive.fail = True
    with pytest.raises(GDriveUploadError):
        publish(c)
    with c.sessions() as session:
        intent = session.scalar(select(AssistantRunResultIntent))
        forged = dict(id=intent.id, conversation_id=intent.conversation_id, run_id=intent.run_id,
            version_id=c.base_id, artifact_id=intent.artifact_id, idempotency_key=intent.idempotency_key,
            request_sha256=intent.request_sha256, actor=intent.actor, before_json=intent.before_json,
            after_json=intent.after_json, delivery="drive_updated", noop=False)
        with pytest.raises(IntegrityError):
            session.execute(insert(AssistantRunResultPromotion).values(**forged))


def test_additive_migration_matches_model_columns_constraints_and_downgrades_empty(setup):
    import importlib.util
    from sqlalchemy import inspect
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    sessions, storage, data = setup
    engine = sessions.kw["bind"]
    AssistantRunResultPromotion.__table__.drop(engine)
    AssistantRunResultIntent.__table__.drop(engine)
    path = Path(__file__).parents[1] / "alembic/versions/j0e5f2a7b914_assistant_run_result_publication.py"
    spec = importlib.util.spec_from_file_location("publication_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        inspector = inspect(connection)
        for model in (AssistantRunResultIntent, AssistantRunResultPromotion):
            table = model.__table__
            assert {column["name"] for column in inspector.get_columns(table.name)} == set(table.columns.keys())
            assert {item["name"] for item in inspector.get_unique_constraints(table.name)} == {
                item.name for item in table.constraints if item.__class__.__name__ == "UniqueConstraint"}
            assert {item["name"] for item in inspector.get_check_constraints(table.name)} == {
                item.name for item in table.constraints if item.__class__.__name__ == "CheckConstraint"}
            assert {item["name"] for item in inspector.get_foreign_keys(table.name)} == {
                item.name for item in table.constraints if item.__class__.__name__ == "ForeignKeyConstraint"}
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()


def test_confirmed_drive_then_database_failure_reconciles_same_session(case):
    c = case
    with c.sessions() as session:
        original_commit = session.commit
        def fail_final_commit():
            if any(isinstance(row, AssistantRunResultPromotion) for row in session.new):
                raise RuntimeError("isolated final DB failure")
            return original_commit()
        session.commit = fail_final_commit
        with pytest.raises(RuntimeError, match="isolated final DB failure"):
            promote_run_result(session, c.service, c.run_id, c.request, drive=c.drive)
    with c.sessions() as session:
        assert session.get(Run, c.run_id).result_local_path == str(c.original)
        intent = session.scalar(select(AssistantRunResultIntent))
        assert intent.resumable_uri and intent.completed_at is None
        assert session.scalar(select(AssistantRunResultPromotion)) is None
    assert publish(c)["delivery"] == "drive_updated"
    assert c.drive.begins == 1


def test_automatic_new_selection_same_epoch_still_rejects_stale_version(case):
    c = case
    with c.sessions.begin() as session:
        version = reserve_version(session, plan(c.run_id, c.version_id), idempotency_key="newer", actor="operator")
        item = ArtifactInput(c.original, metadata={"duration_sec": 20})
        publish_version(session, version.id, full=item, preview=item, storage=c.storage)
        selection = session.get(AssistantSelection, c.conversation_id)
        assert selection.epoch == c.request["expected_selection_epoch"]
        assert selection.selected_version_id != c.request["version_id"]
    with pytest.raises(AssistantConflict):
        publish(c)
    assert c.drive.begins == 0


@pytest.mark.parametrize("edited_duration", [1.0, 4.0])
def test_native_shorter_and_longer_result_duration_and_download_bytes(setup, edited_duration):
    import subprocess
    sessions, storage, data = setup
    original_media, edited_media = data / "native-before.mp4", data / "native-after.mp4"
    for path, duration, color in ((original_media, 2, "red"), (edited_media, edited_duration, "blue")):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=160x120:r=30",
            "-t", str(duration), "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)
    run_id, project_id, original = legacy(sessions, data, content=original_media.read_bytes(), duration=2)
    svc = AssistantTurnService(storage=storage, policy=ProjectAllowlist(project_id), enabled=True)
    with sessions.begin() as session:
        run = session.get(Run, run_id)
        run.result_gdrive_file_id = None
        session.flush()
        base = import_run_version(session, run, actor="operator", storage=storage)
        child = reserve_version(session, plan(run_id, base.id, end=1), idempotency_key="native", actor="operator")
        item = ArtifactInput(edited_media, metadata={"duration_sec": edited_duration})
        publish_version(session, child.id, full=item, preview=item, storage=storage)
        version_id = child.id
    with sessions() as session:
        request = {"version_id": version_id, "expected_selection_epoch": 0,
            "expected_result_revision": run_result_context(session, svc, run_id)["revision"], "idempotency_key": "native-publication"}
        receipt = promote_run_result(session, svc, run_id, request, drive=Drive(sessions))
        run = session.get(Run, run_id)
        assert abs(run.result_duration_sec - edited_duration) < 0.05
        assert Path(run.result_local_path).read_bytes() == edited_media.read_bytes()
        assert history_media(session, svc, run_id, receipt["id"], "before").read_bytes() == original_media.read_bytes()
        assert history_media(session, svc, run_id, receipt["id"], "after").read_bytes() == edited_media.read_bytes()


def test_reservation_commit_ack_loss_preserves_bound_file_for_exact_retry(case):
    c = case
    bound_paths = []
    with c.sessions() as session:
        original_commit = session.commit
        def commit_then_lose_ack():
            intents = [row for row in session.new if isinstance(row, AssistantRunResultIntent)]
            if intents:
                bound_paths.append(Path(intents[0].after_json["path"]))
                original_commit()
                raise RuntimeError("isolated reservation commit acknowledgement lost")
            return original_commit()
        session.commit = commit_then_lose_ack
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            promote_run_result(session, c.service, c.run_id, c.request, drive=c.drive)
    assert len(bound_paths) == 1 and bound_paths[0].read_bytes() == b"shorter-edited-full-video"
    assert c.drive.begins == c.drive.finishes == 0
    with c.sessions() as session:
        intent = session.scalar(select(AssistantRunResultIntent))
        assert intent is not None and intent.after_json["path"] == str(bound_paths[0])
        assert intent.completed_at is None and intent.resumable_uri is None
        assert session.get(Run, c.run_id).result_local_path == str(c.original)
    receipt = publish(c)
    assert receipt["is_current"] and c.drive.begins == c.drive.finishes == 1
