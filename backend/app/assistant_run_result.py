"""Explicit, recoverable publication of immutable assistant video to a Run.

The durable intent is a writer fence, never a lease. A possibly committed Drive
transfer is reconciled through the same private resumable session, never replaced.
This module makes no model calls and does not enqueue work.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time
import uuid

from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from .assistant_access import AssistantAccessDenied, canonical_uuid
from .assistant_models import (AssistantArtifact, AssistantRunResultIntent,
    AssistantRunResultPromotion, AssistantSelection, AssistantVersion)
from .assistant_store import (AssistantConflict, AssistantStoreError, PinnedFile,
    _lock, get_version_artifact, verify_artifact)
from .segment_fill_models import RunSegmentFillJob
from .models import (InsertMerge, OverlayJob, Run, RunAudioPatch, RunCaption,
    RunSegmentRangeAttempt, RunTailEdit, VideoProject)

MAX_BYTES = 256 * 1024 * 1024
MAX_DURATION_SEC = 120
REQUEST_TIMEOUT_SEC = 90


class PublicationUnavailable(RuntimeError):
    """Fixed public error; provider/session details must stay private."""


class _Budget:
    def __init__(self):
        self.deadline = time.monotonic() + REQUEST_TIMEOUT_SEC

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PublicationUnavailable("Publication deadline reached; retry the same saved request")
        return remaining

    def cancelled(self):
        self.remaining()
        return False


def ensure_no_unresolved_intent(session: Session, run_id: str) -> None:
    if session.scalar(select(AssistantRunResultIntent.id).where(
            AssistantRunResultIntent.run_id == run_id,
            AssistantRunResultIntent.completed_at.is_(None)).limit(1)):
        raise AssistantConflict("A main-result publication needs exact-request reconciliation; retry it in edit chat first")


def ensure_segment_reconstruction_allowed(session: Session, run_id: str) -> None:
    ensure_no_unresolved_intent(session, run_id)
    if session.scalar(select(AssistantRunResultPromotion.id).where(
            AssistantRunResultPromotion.run_id == run_id).limit(1)):
        raise AssistantConflict("This Run has published assistant edits; copy to a new Run for segment regeneration")


def ensure_retained_run_source_retry_allowed(session: Session, run_id: str) -> None:
    ensure_no_unresolved_intent(session, run_id)
    if session.scalar(select(AssistantRunResultPromotion.id).where(
            AssistantRunResultPromotion.run_id == run_id).limit(1)):
        raise AssistantConflict("The Run result has been published from edit chat; create a fresh job against its current result")


def _identity(path):
    try:
        info = Path(path).lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
            raise AssistantConflict("A main result is unavailable or exceeds the publication size limit")
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    except (OSError, TypeError):
        raise AssistantConflict("A main result is unavailable") from None


def result_revision(run: Run) -> str:
    try:
        identity = _identity(run.result_local_path) if run.result_local_path else None
    except AssistantConflict:
        identity = None
    return hashlib.sha256(json.dumps([run.result_local_path, run.result_gdrive_file_id, identity],
        separators=(",", ":")).encode()).hexdigest()


def _digest(path, budget):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    before = _identity(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            budget.remaining()
            sha.update(chunk)
            md5.update(chunk)
        after = os.fstat(source.fileno())
    if before != _identity(path) or before != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
        raise AssistantConflict("Result bytes changed while publication was being prepared")
    return sha.hexdigest(), md5.hexdigest(), before


def _verify_snapshot(snapshot, storage, budget, *, local=False):
    budget.remaining()
    storage.verify(PinnedFile(snapshot["storage_key"], snapshot["sha256"], snapshot["size_bytes"]), cancelled=budget.cancelled)
    if local:
        sha, md5, identity = _digest(snapshot["path"], budget)
        if (sha, md5, identity) != (snapshot["sha256"], snapshot["md5"], snapshot["file_identity"]):
            raise AssistantConflict("Publication result bytes changed; exact reconciliation required")


def _probe(path, budget):
    from .assistant_media import probe_trusted_media
    from .assistant_media import AssistantMediaError, EditingPreparationRequired
    try:
        info = probe_trusted_media(path, timeout_sec=min(30, budget.remaining()), cancelled=budget.cancelled)
    except EditingPreparationRequired:
        raise
    except AssistantMediaError:
        raise AssistantConflict("The result could not be safely probed; publication was not started") from None
    duration = info.video_duration_sec
    if type(duration) not in (float, int) or not math.isfinite(duration) or not 0 < duration <= MAX_DURATION_SEC:
        raise AssistantConflict("Publication supports videos up to 120 seconds")
    return float(duration)


def _owned_copy(source, storage, project_id, run_id, intent_id, expected_sha, budget):
    directory = storage.data_root
    for component in ("projects", project_id, "runs", run_id, "results", "assistant", intent_id):
        directory = directory / component
        try:
            directory.mkdir(mode=0o755)
            directory.chmod(0o755)
            storage._sync_directory(directory.parent)
        except FileExistsError:
            if component == intent_id:
                raise AssistantConflict("Publication output already exists") from None
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise AssistantConflict("Publication storage is unsafe")
    destination = directory / "result.mp4"
    staging = directory / ".preparing.mp4"
    sha = hashlib.sha256()
    try:
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as original, open(staging, "xb") as output:
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                budget.remaining()
                output.write(chunk)
                sha.update(chunk)
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            os.fsync(output.fileno())
        if sha.hexdigest() != expected_sha:
            raise AssistantConflict("Selected version bytes changed while copying")
        os.link(staging, destination, follow_symlinks=False)
        staging.unlink()
        storage._sync_directory(directory)
        return destination
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _locks(session, service, run_id, conversation_id, budget):
    if session.get_bind().dialect.name == "postgresql":
        milliseconds = max(1, min(5000, int(budget.remaining() * 1000)))
        session.execute(text(f"SET LOCAL lock_timeout = '{milliseconds}ms'"))
    _lock(session, conversation_id)
    run = service.require_run(session, run_id)
    session.refresh(run, with_for_update=True)
    service.policy.require_project(run.project_id)
    if getattr(run.status, "value", run.status) != "done":
        raise AssistantConflict("Only a completed Run can receive an assistant result")
    return run


def _active_work(session, run, conversation_id):
    from .assistant_turns import AssistantTurn
    checks = (
        select(AssistantVersion.id).where(AssistantVersion.conversation_id == conversation_id, AssistantVersion.status == "pending"),
        select(AssistantTurn.id).where(AssistantTurn.conversation_id == conversation_id, AssistantTurn.status.in_(("queued", "running", "cancel_requested"))),
        select(RunAudioPatch.id).where(RunAudioPatch.run_id == run.id, RunAudioPatch.status.in_(("queued", "processing", "delivering"))),
        select(RunSegmentRangeAttempt.id).where(RunSegmentRangeAttempt.run_id == run.id, RunSegmentRangeAttempt.status.in_(("queued", "generating", "stitching", "delivering"))),
        select(RunTailEdit.id).where(RunTailEdit.run_id == run.id, RunTailEdit.source_type == "run", RunTailEdit.status.in_(("queued", "processing", "delivering"))),
        select(RunCaption.id).where(RunCaption.run_id == run.id, RunCaption.source_type == "run", RunCaption.status.in_(("queued", "transcribing", "burning", "delivering"))),
        select(OverlayJob.id).where(OverlayJob.source_run_id == run.id, OverlayJob.source_type == "run", OverlayJob.deleted_at.is_(None), OverlayJob.status.in_(("created", "queued", "processing", "delivering"))),
        select(RunSegmentFillJob.id).where(RunSegmentFillJob.run_id == run.id, RunSegmentFillJob.status.in_(("admitted", "compositing", "composited"))),
        select(InsertMerge.id).where(or_(InsertMerge.child_run_id == run.id, InsertMerge.target_project_id.in_(select(VideoProject.id).where(VideoProject.source_run_id == run.id))), InsertMerge.status.in_(("queued", "compositing", "delivering"))),
    )
    if any(session.scalar(statement.limit(1)) for statement in checks):
        raise AssistantConflict("Finish active edits and direct Run-source jobs, including overlay drafts, before publishing")


def _version_full(session, service, conversation_id, version_id):
    version = session.get(AssistantVersion, version_id, populate_existing=True)
    if version is None or version.conversation_id != conversation_id or version.status != "succeeded":
        raise AssistantAccessDenied("assistant access denied")
    service.catalog(session).require_target(version_id)
    artifact = get_version_artifact(session, version_id, role="full")
    from .assistant_preparation import require_editable_artifact
    require_editable_artifact(session, artifact)
    owner = session.get(AssistantVersion, artifact.owner_version_id, populate_existing=True)
    if (owner is None or owner.conversation_id != conversation_id or owner.status != "succeeded"
            or artifact.media_kind != "video" or artifact.size_bytes > MAX_BYTES):
        raise AssistantConflict("A successful full video with exact same-Run ownership is required")
    service.catalog(session).resolve_reference(target_version_id=version_id, asset_id=artifact.id,
        version_id=artifact.owner_version_id)
    return artifact


def _receipt(receipt, run):
    root = f"/api/v2/assistant/runs/{run.id}/result-history/{receipt.id}"
    current = (run.result_local_path == receipt.after_json["path"] and result_revision(run) == receipt.after_json["revision"])
    return {"id": receipt.id, "idempotency_key": receipt.idempotency_key, "version_id": receipt.version_id,
        "created_at": receipt.created_at.isoformat(), "before_url": root + "/before", "full_url": root + "/after",
        "delivery": receipt.delivery, "duration_sec": receipt.after_json["duration_sec"], "noop": receipt.noop,
        "is_current": current}


def run_result_context(session, service, run_id):
    conversation = service.conversation(session, run_id)
    run = service.require_run(session, run_id)
    history = session.scalars(select(AssistantRunResultPromotion).where(
        AssistantRunResultPromotion.conversation_id == conversation.id).order_by(AssistantRunResultPromotion.created_at.desc(), AssistantRunResultPromotion.id.desc()).limit(100)).all()
    receipts = [_receipt(item, run) for item in history]
    pending = session.scalar(select(AssistantRunResultIntent).where(
        AssistantRunResultIntent.run_id == run.id, AssistantRunResultIntent.completed_at.is_(None)))
    return {"revision": result_revision(run), "version_id": next((item["version_id"] for item in receipts if item["is_current"]), None),
        "drive_available": bool(run.result_gdrive_file_id), "history": receipts,
        "pending": ({"id": pending.id, "version_id": pending.version_id, "created_at": pending.created_at.isoformat(),
            **pending.request_json} if pending else None)}


def history_media(session, service, run_id, receipt_id, which):
    conversation = service.conversation(session, run_id)
    receipt = session.scalar(select(AssistantRunResultPromotion).where(AssistantRunResultPromotion.id == canonical_uuid(receipt_id),
        AssistantRunResultPromotion.conversation_id == conversation.id))
    if receipt is None or which not in ("before", "after"):
        raise AssistantAccessDenied("assistant access denied")
    snapshot = receipt.before_json if which == "before" else receipt.after_json
    return service.storage.verify(PinnedFile(snapshot["storage_key"], snapshot["sha256"], snapshot["size_bytes"]))


def promote_run_result_published(session, service, run_id, request, *, actor="operator", drive=None):
    """:func:`promote_run_result` plus the committed publication descriptor
    (AICP-299 §4) when THIS call promoted; ``None`` on a replay."""
    box: list = []
    receipt = promote_run_result(session, service, run_id, request, actor=actor, drive=drive, _published=box)
    return receipt, (box[0] if box else None)


def promote_run_result(session, service, run_id, request, *, actor="operator", drive=None, _published=None):
    """Owns reservation/session/final commits; never wrap in a caller transaction."""
    from .gdrive_client import GDriveClient
    from .assistant_models import _now
    budget = _Budget()
    request = dict(request)
    version_id = canonical_uuid(request["version_id"])
    key = request["idempotency_key"]
    if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", key) is None:
        raise AssistantStoreError("Invalid publication key")
    if type(request["expected_selection_epoch"]) is not int or request["expected_selection_epoch"] < 0:
        raise AssistantStoreError("Invalid selection epoch")
    if not isinstance(request["expected_result_revision"], str) or re.fullmatch(r"[a-f0-9]{64}", request["expected_result_revision"]) is None:
        raise AssistantStoreError("Invalid result revision")
    digest = hashlib.sha256(json.dumps({"request": request, "actor": actor}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    conversation = service.conversation(session, run_id)
    conversation_id = conversation.id
    run = service.require_run(session, run_id)  # Authorization always precedes replay.
    intent = session.scalar(select(AssistantRunResultIntent).where(
        AssistantRunResultIntent.conversation_id == conversation_id, AssistantRunResultIntent.idempotency_key == key))
    if intent and intent.request_sha256 != digest:
        raise AssistantConflict("Publication key was already used for another request")
    if intent:
        receipt = session.get(AssistantRunResultPromotion, intent.id)
        if receipt:
            return _receipt(receipt, run)
    staged = None
    reserved = intent is not None
    reservation_commit_attempted = False
    try:
        if not reserved:
            artifact = _version_full(session, service, conversation_id, version_id)
            source = verify_artifact(artifact, storage=service.storage, cancelled=budget.cancelled)
            source_sha, source_md5, _ = _digest(source, budget)
            if source_sha != artifact.sha256:
                raise AssistantConflict("Selected full result integrity check failed")
            duration = _probe(source, budget)
            revision = result_revision(run)
            if revision != request["expected_result_revision"]:
                raise AssistantConflict("Run result changed; reload before publishing")
            current_path = run.result_local_path
            current_identity = _identity(current_path)
            before_pin = service.storage.pin(current_path, cancelled=budget.cancelled)
            before_path = service.storage.verify(before_pin, cancelled=budget.cancelled)
            try:
                from .assistant_media import EditingPreparationRequired
                before_duration = _probe(before_path, budget)
                archive_profile = None
            except EditingPreparationRequired as error:
                # Archive-only exception, bound to the exact imported original
                # in this Run. This never validates a selected output for use.
                from .assistant_models import AssistantVersionArtifact
                original = session.scalar(select(AssistantArtifact).join(AssistantVersionArtifact,
                    AssistantVersionArtifact.artifact_id == AssistantArtifact.id).join(AssistantVersion,
                    AssistantVersion.id == AssistantVersionArtifact.version_id).where(
                        AssistantVersion.conversation_id == conversation_id, AssistantVersion.sequence == 1,
                        AssistantVersion.status == "succeeded", AssistantVersionArtifact.role == "full",
                        AssistantArtifact.owner_version_id == AssistantVersion.id,
                        AssistantVersion.recipe["kind"].as_string() == "legacy_import",
                        AssistantArtifact.sha256 == before_pin.sha256, AssistantArtifact.size_bytes == before_pin.size_bytes))
                from .assistant_preparation import needs_preparation
                if original is None or not needs_preparation(original):
                    raise AssistantConflict("Archive profile is not bound to this Run's imported original")
                before_duration = error.info.video_duration_sec
                archive_profile = {"kind": error.preparation["kind"] + "_archive_only",
                    "audio_duration_sec": error.preparation["audio_duration_sec"],
                    **({"source_final_pts_sec": error.preparation["source_final_pts_sec"],
                        "source_declared_duration_sec": error.preparation["source_declared_duration_sec"]}
                        if error.preparation["kind"] == "audio_tail_grid_gap" else {})}
            before_sha, before_md5, final_before_identity = _digest(current_path, budget)
            if (before_sha, final_before_identity) != (before_pin.sha256, current_identity):
                raise AssistantConflict("Current result changed while preparing publication")
            intent_id = str(uuid.uuid4())
            staged = _owned_copy(source, service.storage, run.project_id, run.id, intent_id, artifact.sha256, budget)
            after = {"path": str(staged), "storage_key": artifact.storage_key, "sha256": artifact.sha256,
                "md5": source_md5, "size_bytes": artifact.size_bytes, "duration_sec": duration,
                "drive_file_id": run.result_gdrive_file_id, "file_identity": _identity(staged)}
            before = {"path": current_path, "storage_key": before_pin.storage_key, "sha256": before_pin.sha256,
                "md5": before_md5, "size_bytes": before_pin.size_bytes, "duration_sec": before_duration,
                "drive_file_id": run.result_gdrive_file_id, "file_identity": current_identity, "revision": revision,
                **({"archive_profile": archive_profile} if archive_profile else {})}
            artifact_id = artifact.id
            session.commit()  # Release preparation's read transaction before row locks.
            run = _locks(session, service, run_id, conversation_id, budget)
            existing = session.scalar(select(AssistantRunResultIntent).where(
                AssistantRunResultIntent.conversation_id == conversation_id, AssistantRunResultIntent.idempotency_key == key))
            if existing:
                if existing.request_sha256 != digest:
                    raise AssistantConflict("Publication key was already used for another request")
                shutil.rmtree(staged.parent, ignore_errors=True)
                staged = None
                intent = existing
                reserved = True
            else:
                ensure_no_unresolved_intent(session, run.id)
                selection = session.get(AssistantSelection, conversation_id, populate_existing=True)
                if not selection or selection.selected_version_id != version_id or selection.epoch != request["expected_selection_epoch"]:
                    raise AssistantConflict("Selected version changed; reload before publishing")
                if result_revision(run) != request["expected_result_revision"] or _identity(run.result_local_path) != before["file_identity"]:
                    raise AssistantConflict("Run result changed; reload before publishing")
                _active_work(session, run, conversation_id)
                artifact = _version_full(session, service, conversation_id, version_id)
                _verify_snapshot(after, service.storage, budget, local=True)
                _verify_snapshot(before, service.storage, budget, local=True)
                intent = AssistantRunResultIntent(id=intent_id, conversation_id=conversation_id, run_id=run.id,
                    version_id=version_id, artifact_id=artifact_id, idempotency_key=key, request_sha256=digest,
                    actor=actor, request_json=request, before_json=before, after_json=after)
                session.add(intent)
                # COMMIT may succeed while its acknowledgement is lost. From this
                # point cleanup cannot know the directory is unreferenced.
                reservation_commit_attempted = True
                session.commit()  # Durable writer exclusion BEFORE external writes.
                reserved = True
        # Every stage of recovery takes the same lock order and fresh scope check.
        intent_id = intent.id
        session.commit()
        run = _locks(session, service, run_id, conversation_id, budget)
        intent = session.get(AssistantRunResultIntent, intent_id, populate_existing=True)
        receipt = session.get(AssistantRunResultPromotion, intent_id)
        if receipt:
            return _receipt(receipt, run)
        _version_full(session, service, conversation_id, version_id)
        if result_revision(run) != intent.before_json["revision"]:
            raise AssistantConflict("Reserved publication base changed; manual reconciliation is required")
        _verify_snapshot(intent.before_json, service.storage, budget, local=True)
        _verify_snapshot(intent.after_json, service.storage, budget, local=True)
        drive_id = intent.before_json["drive_file_id"]
        if drive_id:
            drive = drive or GDriveClient()
            if intent.resumable_uri is None:
                # Initiation sends metadata only. A lost response here sent NO media.
                uri = drive.begin_result_update(drive_id, intent.after_json["path"], deadline=budget.deadline)
                intent.resumable_uri = uri
                session.commit()  # Persist the private URI BEFORE any media bytes.
                run = _locks(session, service, run_id, conversation_id, budget)
                intent = session.get(AssistantRunResultIntent, intent_id, populate_existing=True)
                receipt = session.get(AssistantRunResultPromotion, intent_id)
                if receipt:
                    return _receipt(receipt, run)
                if result_revision(run) != intent.before_json["revision"]:
                    raise AssistantConflict("Reserved publication base changed; manual reconciliation is required")
                _verify_snapshot(intent.after_json, service.storage, budget, local=True)
            drive.finish_result_update(intent.resumable_uri, drive_id, intent.after_json["path"],
                size_bytes=intent.after_json["size_bytes"], sha256=intent.after_json["sha256"],
                md5=intent.after_json["md5"], deadline=budget.deadline)
        budget.remaining()
        run.result_local_path = intent.after_json["path"]
        run.result_duration_sec = intent.after_json["duration_sec"]
        after = {**intent.after_json, "revision": result_revision(run)}
        receipt = AssistantRunResultPromotion(id=intent.id, conversation_id=conversation_id, run_id=run.id,
            version_id=intent.version_id, artifact_id=intent.artifact_id, idempotency_key=key, request_sha256=digest, actor=actor,
            before_json=intent.before_json, after_json=after, delivery="drive_updated" if drive_id else "local_only", noop=False)
        session.add(receipt)
        intent.completed_at = _now()
        session.commit()
        if _published is not None:
            from .knowledge.publication import PublishedFinal

            _published.append(PublishedFinal(run_id=run.id, project_id=run.project_id,
                                             path=intent.after_json["path"], kind="assistant",
                                             publication_id=f"assistant_promotion:{intent.id}"))
        return _receipt(receipt, run)
    except BaseException:
        session.rollback()
        if staged is not None and not reserved and not reservation_commit_attempted:
            shutil.rmtree(staged.parent, ignore_errors=True)
        raise
