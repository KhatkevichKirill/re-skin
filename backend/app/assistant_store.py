"""Transactional assistant history and no-clobber artifact pinning.

Caller owns commit/rollback AND authorization (including target/donor allowlists).
No function calls a provider, enqueues work, touches Drive, or changes a Run.
Job publication must call publish_version inside its fenced, uncancelled SQL
transaction. Files left after rollback are unreferenced retention/GC candidates;
this module never removes an object that a committed version may reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .assistant_contracts import EditPlan
from .assistant_models import (
    AssistantArtifact, AssistantConversation, AssistantMessage, AssistantSelection,
    AssistantVersion, AssistantVersionArtifact,
)

MAX_ARTIFACT_BYTES = 2 * 1024**3
MAX_JSON_BYTES = 4 * 1024**2
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class AssistantStoreError(ValueError):
    pass


class AssistantConflict(AssistantStoreError):
    pass


class AssistantNotFound(AssistantStoreError):
    pass


class ArtifactIntegrityError(AssistantStoreError):
    pass


def _id(value: str) -> str:
    if not isinstance(value, str) or _UUID.fullmatch(value) is None:
        raise AssistantStoreError("expected a canonical UUID")
    return value


def _key(value: str) -> str:
    if not isinstance(value, str) or _KEY.fullmatch(value) is None:
        raise AssistantStoreError("idempotency key must be 1-128 ASCII identifier characters")
    return value


def _actor(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise AssistantStoreError("actor must be a nonempty bounded identifier")
    return value


def _snapshot(value) -> tuple[object, str]:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AssistantStoreError("snapshot must be finite JSON data") from exc
    raw = encoded.encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise AssistantStoreError("snapshot exceeds the JSON size limit")
    return json.loads(encoded), hashlib.sha256(raw).hexdigest()


def _duration(metadata: Mapping) -> float:
    value = metadata.get("duration_sec")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 86_400:
        raise AssistantStoreError("full output requires finite positive duration_sec metadata <= 24 hours")
    return float(value)


@dataclass(frozen=True)
class ArtifactInput:
    """Trusted renderer/resolver input, never an assistant-tool request schema."""

    source_path: str | Path
    media_kind: str = "video"
    metadata: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class PinnedFile:
    storage_key: str
    sha256: str
    size_bytes: int


class AssistantArtifactStorage:
    """Content-addressed copies outside projects/, published by exclusive link.

    Sources must resolve inside data_root. Directories/files under this private
    root must be owned by the backend; it is not an untrusted shared workspace.
    No hard link to the mutable source is retained. No existing object is replaced.
    """

    def __init__(self, data_root: str | Path, *, max_artifact_bytes: int = MAX_ARTIFACT_BYTES):
        self.data_root = Path(data_root).resolve()
        if not self.data_root.is_dir():
            raise AssistantStoreError("data root must already exist")
        if isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int) or max_artifact_bytes <= 0:
            raise AssistantStoreError("artifact size limit must be a positive integer")
        self.max_artifact_bytes = max_artifact_bytes
        self.root = self.data_root / "assistant_artifacts"
        self._directory(self.root)
        self._directory(self.root / "objects")
        self._directory(self.root / "staging")

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _directory(cls, path: Path) -> None:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ArtifactIntegrityError("artifact directory must be a real directory, not a symlink")
        # Persist the directory's entry before relying on any files under it.
        # An existing entry may belong to a concurrent creator that has not yet
        # synced its parent; each successful caller must establish durability.
        cls._sync_directory(path.parent)

    def _source(self, path: str | Path) -> Path:
        try:
            resolved = Path(path).resolve(strict=True)
            resolved.relative_to(self.data_root)
        except (OSError, ValueError) as exc:
            raise AssistantStoreError("source must exist inside the data root") from exc
        return resolved

    def path_for(self, storage_key: str) -> Path:
        if not isinstance(storage_key, str) or re.fullmatch(r"objects/[0-9a-f]{2}/[0-9a-f]{64}", storage_key) is None:
            raise ArtifactIntegrityError("invalid artifact storage key")
        digest = storage_key.rsplit("/", 1)[1]
        if storage_key.split("/")[1] != digest[:2]:
            raise ArtifactIntegrityError("artifact prefix does not match hash")
        path = self.root / storage_key
        try:
            for directory in (self.root, self.root / "objects", path.parent):
                if not stat.S_ISDIR(directory.lstat().st_mode):
                    raise ArtifactIntegrityError("artifact directory is not a real directory")
        except OSError as exc:
            raise ArtifactIntegrityError("artifact directory is unavailable") from exc
        return path

    @staticmethod
    def _signature(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def verify(self, pinned: PinnedFile, *, cancelled=None, verification_cache: dict | None = None) -> Path:
        verification_started_ns = time.time_ns()
        if _DIGEST.fullmatch(pinned.sha256) is None or pinned.size_bytes <= 0:
            raise ArtifactIntegrityError("invalid artifact identity")
        path = self.path_for(pinned.storage_key)
        try:
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != pinned.size_bytes:
                    raise ArtifactIntegrityError("artifact size or file type changed")
                # A caller may reuse successful byte verification only within
                # one bounded history inspection. Reopen O_NOFOLLOW and check
                # the actual descriptor on every use; this caches no ownership
                # or policy decision and never survives an API request.
                key = (self.root, pinned.storage_key, pinned.sha256, pinned.size_bytes)
                signature = self._signature(before)
                # Kernel/filesystem timestamps can remain identical during
                # rapid writes in one clock tick, even with *_ns fields. Fresh
                # or future-ctime objects always take the full hash path.
                cacheable = before.st_ctime_ns <= verification_started_ns - 1_000_000_000
                if verification_cache is not None and cacheable and verification_cache.get(key) == signature:
                    if cancelled is not None and cancelled():
                        from .assistant_media import AssistantMediaCancelled
                        raise AssistantMediaCancelled("artifact verification cancelled")
                    if signature != self._signature(os.fstat(handle.fileno())):
                        raise ArtifactIntegrityError("artifact content changed")
                    return path
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    if cancelled is not None and cancelled():
                        from .assistant_media import AssistantMediaCancelled
                        raise AssistantMediaCancelled("artifact verification cancelled")
                    digest.update(chunk)
                if cancelled is not None and cancelled():
                    from .assistant_media import AssistantMediaCancelled
                    raise AssistantMediaCancelled("artifact verification cancelled")
                if self._signature(before) != self._signature(os.fstat(handle.fileno())) or digest.hexdigest() != pinned.sha256:
                    raise ArtifactIntegrityError("artifact content changed")
                if verification_cache is not None:
                    if cacheable:
                        verification_cache[key] = signature
                    else:
                        verification_cache.pop(key, None)
        except OSError as exc:
            raise ArtifactIntegrityError("artifact is unavailable or unsafe") from exc
        return path

    def pin(self, source_path: str | Path, *, cancelled=None) -> PinnedFile:
        def check():
            if cancelled is not None and cancelled():
                from .assistant_media import AssistantMediaCancelled
                raise AssistantMediaCancelled("artifact pinning cancelled")
        check()
        source = self._source(source_path)
        descriptor, temporary = tempfile.mkstemp(prefix="pin-", dir=self.root / "staging")
        try:
            digest, size = hashlib.sha256(), 0
            with os.fdopen(descriptor, "wb") as destination:
                with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as original:
                    before = os.fstat(original.fileno())
                    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= self.max_artifact_bytes:
                        raise AssistantStoreError("source must be a nonempty regular file within the artifact size limit")
                    for chunk in iter(lambda: original.read(1024 * 1024), b""):
                        check()
                        size += len(chunk)
                        if size > self.max_artifact_bytes:
                            raise AssistantStoreError("source exceeds the artifact size limit")
                        digest.update(chunk)
                        destination.write(chunk)
                    if self._signature(before) != self._signature(os.fstat(original.fileno())) or size != before.st_size:
                        raise AssistantConflict("source changed while being pinned")
                destination.flush()
                os.fchmod(destination.fileno(), 0o400)
                os.fsync(destination.fileno())
            sha256 = digest.hexdigest()
            check()
            folder = self.root / "objects" / sha256[:2]
            self._directory(folder)
            pinned = PinnedFile(f"objects/{sha256[:2]}/{sha256}", sha256, size)
            path = self.path_for(pinned.storage_key)
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                if cancelled is None:
                    self.verify(pinned)
                else:
                    self.verify(pinned, cancelled=cancelled)
            # This must also run after FileExistsError: another pinner can make
            # the link visible before it fsyncs this directory. We cannot return
            # a commit-ready reference until our own directory sync succeeds.
            self._sync_directory(folder)
            return pinned
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _storage(storage: AssistantArtifactStorage | None) -> AssistantArtifactStorage:
    if storage is not None:
        return storage
    from .storage import data_dir_root
    return AssistantArtifactStorage(data_dir_root())


def _lock(session: Session, conversation_id: str) -> AssistantConversation:
    # A no-op UPDATE is an actual row lock on Postgres, a writer lock on SQLite.
    # Every selection/counter/version change takes this lock first.
    result = session.execute(update(AssistantConversation).where(
        AssistantConversation.id == _id(conversation_id),
    ).values(next_version_sequence=AssistantConversation.next_version_sequence).execution_options(synchronize_session=False))
    if result.rowcount != 1:
        raise AssistantNotFound("conversation not found")
    return session.get(AssistantConversation, conversation_id, populate_existing=True)


def get_or_create_conversation(session: Session, *, run_id: str, project_id: str) -> AssistantConversation:
    """IDs are trusted provenance; caller must check live Run/project access."""
    run_id, project_id = _id(run_id), _id(project_id)
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        raise AssistantStoreError("assistant store supports PostgreSQL and SQLite only")
    session.execute(insert(AssistantConversation).values(
        id=str(uuid.uuid4()), run_id=run_id, project_id=project_id,
        next_version_sequence=1, next_message_sequence=1,
    ).on_conflict_do_nothing(index_elements=["run_id"]))
    conversation = session.execute(select(AssistantConversation).where(
        AssistantConversation.run_id == run_id,
    ).execution_options(populate_existing=True)).scalar_one()
    if conversation.project_id != project_id:
        raise AssistantConflict("Run is already bound to another project")
    return conversation


def _selection(session: Session, conversation_id: str) -> AssistantSelection:
    selection = session.get(AssistantSelection, conversation_id, populate_existing=True)
    if selection is None:
        selection = AssistantSelection(conversation_id=conversation_id, epoch=0)
        session.add(selection)
        session.flush()
    return selection


def _version(session: Session, version_id: str) -> AssistantVersion:
    version = session.get(AssistantVersion, _id(version_id), populate_existing=True)
    if version is None:
        raise AssistantNotFound("version not found")
    return version


def get_version_artifact(session: Session, version_id: str, *, role: str = "full") -> AssistantArtifact:
    _id(version_id)
    if role not in {"full", "preview", "clean", "captions"}:
        raise AssistantStoreError("unsupported output role")
    artifact = session.execute(select(AssistantArtifact).join(
        AssistantVersionArtifact, AssistantVersionArtifact.artifact_id == AssistantArtifact.id,
    ).where(AssistantVersionArtifact.version_id == version_id,
            AssistantVersionArtifact.role == role, AssistantVersionArtifact.slot == 0)).scalar_one_or_none()
    if artifact is None:
        raise AssistantNotFound("version output not found")
    return artifact


def verify_artifact(artifact: AssistantArtifact, *, storage: AssistantArtifactStorage | None = None,
                    cancelled=None, verification_cache: dict | None = None) -> Path:
    options = {"cancelled":cancelled} if cancelled is not None else {}
    if verification_cache is not None:
        options["verification_cache"] = verification_cache
    return _storage(storage).verify(PinnedFile(artifact.storage_key, artifact.sha256, artifact.size_bytes), **options)


def _add_artifact(session: Session, version: AssistantVersion, item: ArtifactInput,
                  storage: AssistantArtifactStorage) -> AssistantArtifact:
    if item.media_kind not in {"video", "audio", "image", "subtitle"}:
        raise AssistantStoreError("unsupported artifact media kind")
    metadata, _ = _snapshot(dict(item.metadata))
    pinned = storage.pin(item.source_path)
    artifact = AssistantArtifact(
        id=str(uuid.uuid4()), owner_version_id=version.id, media_kind=item.media_kind,
        storage_key=pinned.storage_key, sha256=pinned.sha256, size_bytes=pinned.size_bytes,
        metadata_json=metadata,
    )
    session.add(artifact)
    session.flush()
    return artifact


def _reference(session: Session, version_id: str, artifact_id: str, role: str, slot: int = 0):
    session.add(AssistantVersionArtifact(
        version_id=version_id, artifact_id=artifact_id, role=role, slot=slot,
    ))


def _advance_selection(session: Session, version: AssistantVersion):
    selection = _selection(session, version.conversation_id)
    latest = _version(session, selection.latest_version_id) if selection.latest_version_id else None
    selected = _version(session, selection.selected_version_id) if selection.selected_version_id else None
    if latest is None or version.sequence > latest.sequence:
        selection.latest_version_id = version.id
    if selection.epoch == version.selection_epoch and (selected is None or version.sequence > selected.sequence):
        selection.selected_version_id = version.id


def import_run_version(
    session: Session, run, *, actor: str, idempotency_key: str = "legacy-import-v1",
    storage: AssistantArtifactStorage | None = None,
    full: ArtifactInput | None = None, clean: ArtifactInput | None = None,
    captions: ArtifactInput | None = None,
) -> AssistantVersion:
    """Pin the currently successful Run once, without changing any legacy row.

    Retry of the same import returns V1 even if legacy pointers/settings changed
    later. Import is a snapshot, not a synchronization mechanism or historical
    reconstruction. Caller must supply a persisted, freshly authorized Run.
    """
    from .models import Run
    if not isinstance(run, Run) or run.id is None or run.project_id is None:
        raise AssistantStoreError("import requires a persisted Run")
    actor, idempotency_key = _actor(actor), _key(idempotency_key)
    conversation = get_or_create_conversation(session, run_id=run.id, project_id=run.project_id)
    conversation = _lock(session, conversation.id)
    existing = session.execute(select(AssistantVersion).where(
        AssistantVersion.conversation_id == conversation.id,
        AssistantVersion.idempotency_key == idempotency_key,
    )).scalar_one_or_none()
    if existing:
        if existing.recipe.get("kind") != "legacy_import":
            raise AssistantConflict("idempotency key was used for another operation")
        return existing
    if conversation.next_version_sequence != 1:
        raise AssistantConflict("conversation already has an imported base version")
    # Re-read before pinning; caller authorization must cover this exact Run.
    session.refresh(run, with_for_update=True)
    if getattr(run.status, "value", run.status) != "done" or not run.result_local_path:
        raise AssistantStoreError("only a successful Run with a local full result can be imported")
    if full is None:
        _duration({"duration_sec": run.result_duration_sec})
    settings_fields = (
        "name", "prompt", "model", "resolution", "audio_mode", "language",
        "reference_image_urls", "reference_audio_urls", "stitch_plan", "segment_snapshot",
        "preprocess_enabled", "preprocess_mode", "preprocess_image_prompts",
        "preprocessed_reference_urls", "generate_audio",
    )
    settings, _ = _snapshot({name: getattr(run, name, None) for name in settings_fields})
    recipe, request_hash = _snapshot({
        "kind": "legacy_import", "schema_version": 1, "run_id": run.id,
        "settings": settings, "legacy_drive_file_id": run.result_gdrive_file_id,
        "provenance": "current successful Run snapshot; earlier in-place history unavailable",
    })
    selection = _selection(session, conversation.id)
    version = AssistantVersion(
        id=str(uuid.uuid4()), conversation_id=conversation.id, sequence=1,
        selection_epoch=selection.epoch, idempotency_key=idempotency_key,
        actor=actor, request_sha256=request_hash, recipe=recipe, status="pending",
    )
    session.add(version)
    session.flush()
    full = full or ArtifactInput(
        run.result_local_path, "video", {"duration_sec": run.result_duration_sec,
                                       "preview_mode": "full_as_preview", "subtitle_status": "unknown"},
    )
    if full.media_kind != "video":
        raise AssistantStoreError("imported full must be video")
    _duration(full.metadata)
    if (clean is None) != (captions is None):
        raise AssistantStoreError("editable import requires clean picture and subtitle track")
    if captions is not None:
        from .assistant_subtitles import SubtitleTrack, validate_duration
        if captions.media_kind != "subtitle" or clean.media_kind != "video":
            raise AssistantStoreError("invalid editable import artifact types")
        validate_duration(SubtitleTrack.model_validate_json(Path(captions.source_path).read_bytes()), _duration(full.metadata))
    artifact = _add_artifact(session, version, full, _storage(storage))
    _reference(session, version.id, artifact.id, "full")
    _reference(session, version.id, artifact.id, "preview")
    for role, item in (("clean", clean), ("captions", captions)):
        if item is not None:
            child = _add_artifact(session, version, item, _storage(storage))
            _reference(session, version.id, child.id, role)
    conversation.next_version_sequence = 2
    version.status, version.completed_at = "succeeded", datetime.now(timezone.utc)
    session.flush()
    _advance_selection(session, version)
    session.flush()
    return version


def import_media_version(session, run, *, full: ArtifactInput, clean: ArtifactInput,
                         captions: ArtifactInput, actor: str, storage,
                         idempotency_key: str = "legacy-import-v1"):
    """Trusted adapter entrypoint for a proven full/clean/editable-track snapshot.

    Caller verifies legacy lineage and exact rendered cue/style provenance. This
    function only freezes bytes; it grants no clean-source or access authority.
    """
    return import_run_version(session, run, actor=actor, storage=storage,
        idempotency_key=idempotency_key, full=full, clean=clean, captions=captions)


def import_derived_media_version(session, run, *, full: ArtifactInput, clean: ArtifactInput,
                                 captions: ArtifactInput, actor: str, storage,
                                 idempotency_key: str, provenance: dict):
    """Append a trusted, proven legacy creative snapshot without changing V1.

    Caller performs source/render proof and authorization. Idempotency binds
    exact bytes, metadata and provenance, including after selection changes.
    """
    from .models import Run
    from .assistant_subtitles import SubtitleTrack, validate_duration
    if not isinstance(run,Run) or run.id is None:
        raise AssistantStoreError("derived import requires persisted Run")
    actor,key=_actor(actor),_key(idempotency_key)
    conversation=get_or_create_conversation(session,run_id=run.id,project_id=run.project_id)
    conversation=_lock(session,conversation.id)
    session.refresh(run,with_for_update=True)
    if getattr(run.status,"value",run.status)!="done":
        raise AssistantStoreError("derived import requires successful Run")
    if full.media_kind!="video" or clean.media_kind!="video" or captions.media_kind!="subtitle":
        raise AssistantStoreError("invalid derived import artifact kinds")
    duration=_duration(full.metadata)
    if abs(_duration(clean.metadata)-duration)>0.000001:
        raise AssistantStoreError("derived clean picture duration differs")
    validate_duration(SubtitleTrack.model_validate_json(Path(captions.source_path).read_bytes()),duration)
    storage=_storage(storage)
    identity={}
    for role,item in (("full",full),("clean",clean),("captions",captions)):
        pinned=storage.pin(item.source_path)
        identity[role]={"sha256":pinned.sha256,"size_bytes":pinned.size_bytes,
                        "media_kind":item.media_kind,"metadata":dict(item.metadata)}
    recipe,digest=_snapshot({"kind":"legacy_derived_import","schema_version":1,
        "run_id":run.id,"provenance":provenance,"artifacts":identity})
    existing=session.scalar(select(AssistantVersion).where(
        AssistantVersion.conversation_id==conversation.id,AssistantVersion.idempotency_key==key))
    if existing:
        if existing.actor!=actor or existing.request_sha256!=digest:
            raise AssistantConflict("derived import idempotency payload differs")
        return existing
    selection=_selection(session,conversation.id)
    if selection.selected_version_id is None:
        raise AssistantConflict("derived import requires an existing successful base")
    base=_version(session,selection.selected_version_id)
    if base.status!="succeeded":
        raise AssistantConflict("derived import selected base is not successful")
    version=AssistantVersion(id=str(uuid.uuid4()),conversation_id=conversation.id,
        sequence=conversation.next_version_sequence,parent_version_id=base.id,base_version_id=base.id,
        selection_epoch=selection.epoch,idempotency_key=key,actor=actor,request_sha256=digest,
        recipe=recipe,status="pending")
    session.add(version)
    session.flush()
    conversation.next_version_sequence+=1
    for role,item in (("full",full),("clean",clean),("captions",captions)):
        artifact=_add_artifact(session,version,item,storage)
        if (artifact.sha256,artifact.size_bytes)!=(identity[role]["sha256"],identity[role]["size_bytes"]):
            raise AssistantConflict("derived source changed during import")
        _reference(session,version.id,artifact.id,role)
        if role=="full":
            _reference(session,version.id,artifact.id,"preview")
    version.status,version.completed_at="succeeded",datetime.now(timezone.utc)
    session.flush()
    _advance_selection(session,version)
    session.flush()
    return version


def reserve_version(session: Session, plan: EditPlan, *, idempotency_key: str, actor: str) -> AssistantVersion:
    """Freeze an accepted plan and its input references; caller authorizes all IDs."""
    # Revalidate JSON to avoid trusting model_construct/model_copy bypasses.
    plan = EditPlan.model_validate(plan.model_dump(mode="json"))
    idempotency_key, actor = _key(idempotency_key), _actor(actor)
    base = _version(session, plan.base_version_id)
    conversation = _lock(session, base.conversation_id)
    base = _version(session, plan.base_version_id)
    if conversation.run_id != plan.run_id or base.status != "succeeded":
        raise AssistantConflict("plan requires a successful base in its Run")
    recipe, request_hash = _snapshot(plan.model_dump(mode="json"))
    existing = session.execute(select(AssistantVersion).where(
        AssistantVersion.conversation_id == conversation.id,
        AssistantVersion.idempotency_key == idempotency_key,
    )).scalar_one_or_none()
    if existing is not None:
        if existing.request_sha256 != request_hash or existing.actor != actor:
            raise AssistantConflict("idempotency key payload does not match")
        return existing
    inputs = {get_version_artifact(session, base.id).id}
    for role in ("clean", "captions"):
        try:
            inputs.add(get_version_artifact(session, base.id, role=role).id)
        except AssistantNotFound:
            pass
    for operation in plan.operations:
        source = getattr(operation, "source", None)
        if source is None:
            continue
        artifact = session.get(AssistantArtifact, source.asset_id)
        if artifact is None or artifact.owner_version_id != source.version_id:
            raise AssistantConflict("donor asset/version binding does not match")
        if _version(session, source.version_id).status != "succeeded":
            raise AssistantConflict("donor version must be successful")
        inputs.add(artifact.id)
    selection = _selection(session, conversation.id)
    version = AssistantVersion(
        id=str(uuid.uuid4()), conversation_id=conversation.id,
        sequence=conversation.next_version_sequence, parent_version_id=base.id, base_version_id=base.id,
        selection_epoch=selection.epoch, idempotency_key=idempotency_key,
        request_sha256=request_hash, actor=actor, recipe=recipe, status="pending",
    )
    conversation.next_version_sequence += 1
    session.add(version)
    session.flush()
    for slot, artifact_id in enumerate(sorted(inputs)):
        _reference(session, version.id, artifact_id, "input", slot)
    session.flush()
    return version


def publish_version(
    session: Session, version_id: str, *, full: ArtifactInput, preview: ArtifactInput,
    storage: AssistantArtifactStorage | None = None,
    clean: ArtifactInput | None = None, captions: ArtifactInput | None = None,
) -> AssistantVersion:
    """Publish only inside the job's lease/cancel guard transaction; no commits.

    This store does not prove media semantics. Caller supplies verified renderer
    metadata and enforces supported capabilities before reaching publication.
    """
    version = _version(session, version_id)
    _lock(session, version.conversation_id)
    version = _version(session, version_id)
    if version.status not in {"pending", "succeeded"}:
        raise AssistantConflict("failed/cancelled version cannot publish")
    if full.media_kind != "video" or preview.media_kind != "video":
        raise AssistantStoreError("full output and completed-version preview must be playable video")
    _duration(full.metadata)
    if (clean is None) != (captions is None):
        raise AssistantStoreError("editable captions require both clean picture and track")
    if clean is not None:
        from .assistant_subtitles import SubtitleTrack
        if clean.media_kind != "video" or captions.media_kind != "subtitle":
            raise AssistantStoreError("invalid clean picture or caption artifact kind")
        # Track bytes are strict backend JSON, not arbitrary renderer code.
        SubtitleTrack.model_validate_json(Path(captions.source_path).read_bytes())
    outputs = [("full", full), ("preview", preview)]
    if clean is not None:
        outputs += [("clean", clean), ("captions", captions)]
    storage = _storage(storage)
    if version.status == "succeeded":
        for role in ("clean", "captions"):
            try:
                get_version_artifact(session, version.id, role=role)
                present = True
            except AssistantNotFound:
                present = False
            if present != (clean is not None):
                raise AssistantConflict("successful subtitle lineage is immutable")
        for role, item in outputs:
            existing = get_version_artifact(session, version.id, role=role)
            pinned = storage.pin(item.source_path)
            metadata, _ = _snapshot(dict(item.metadata))
            if (existing.sha256, existing.size_bytes, existing.media_kind, existing.metadata_json) != (
                pinned.sha256, pinned.size_bytes, item.media_kind, metadata,
            ):
                raise AssistantConflict("successful output is immutable; repeated publish differs")
        return version
    for role, item in outputs:
        artifact = _add_artifact(session, version, item, storage)
        _reference(session, version.id, artifact.id, role)
    version.status, version.completed_at = "succeeded", datetime.now(timezone.utc)
    session.flush()
    _advance_selection(session, version)
    session.flush()
    return version


def mark_version_terminal(session: Session, version_id: str, *, status: str) -> AssistantVersion:
    if status not in {"failed", "cancelled"}:
        raise AssistantStoreError("terminal failure status must be failed or cancelled")
    version = _version(session, version_id)
    _lock(session, version.conversation_id)
    version = _version(session, version_id)
    if version.status == status:
        return version
    if version.status != "pending":
        raise AssistantConflict("terminal version state cannot change")
    version.status, version.completed_at = status, datetime.now(timezone.utc)
    session.flush()
    return version


def select_version(session: Session, conversation_id: str, version_id: str, *, expected_epoch: int) -> AssistantSelection:
    _lock(session, conversation_id)
    if isinstance(expected_epoch, bool) or not isinstance(expected_epoch, int) or expected_epoch < 0:
        raise AssistantStoreError("expected_epoch must be a nonnegative integer")
    selection = _selection(session, conversation_id)
    version = _version(session, version_id)
    if version.conversation_id != conversation_id or version.status != "succeeded":
        raise AssistantConflict("selection requires a successful version of this conversation")
    if selection.epoch != expected_epoch:
        raise AssistantConflict("selection changed; reload before selecting")
    selection.selected_version_id, selection.epoch = version.id, selection.epoch + 1
    session.flush()
    return selection


def append_message(
    session: Session, conversation_id: str, *, idempotency_key: str, actor: str,
    role: str, content: str, base_version_id: str | None, context_snapshot: Mapping | None = None,
) -> AssistantMessage:
    conversation = _lock(session, conversation_id)
    idempotency_key, actor = _key(idempotency_key), _actor(actor)
    if role not in {"operator", "assistant", "tool", "system"}:
        raise AssistantStoreError("unsupported message role")
    if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > 256 * 1024:
        raise AssistantStoreError("message must be nonempty and at most 256 KiB")
    if base_version_id is not None:
        base = _version(session, base_version_id)
        if base.conversation_id != conversation.id or base.status != "succeeded":
            raise AssistantConflict("message base must be a successful version of this Run")
    context, _ = _snapshot(dict(context_snapshot or {}))
    _, request_hash = _snapshot({"role": role, "actor": actor, "content": content,
                                 "base_version_id": base_version_id, "context_snapshot": context})
    existing = session.execute(select(AssistantMessage).where(
        AssistantMessage.conversation_id == conversation.id,
        AssistantMessage.idempotency_key == idempotency_key,
    )).scalar_one_or_none()
    if existing is not None:
        if existing.request_sha256 != request_hash:
            raise AssistantConflict("message idempotency key payload does not match")
        return existing
    message = AssistantMessage(
        conversation_id=conversation.id, sequence=conversation.next_message_sequence,
        idempotency_key=idempotency_key, request_sha256=request_hash, actor=actor,
        role=role, content=content, base_version_id=base_version_id, context_snapshot=context,
    )
    conversation.next_message_sequence += 1
    session.add(message)
    session.flush()
    return message


def list_versions(session: Session, conversation_id: str) -> list[AssistantVersion]:
    return list(session.execute(select(AssistantVersion).where(
        AssistantVersion.conversation_id == _id(conversation_id),
    ).order_by(AssistantVersion.sequence)).scalars())


def list_messages(session: Session, conversation_id: str) -> list[AssistantMessage]:
    return list(session.execute(select(AssistantMessage).where(
        AssistantMessage.conversation_id == _id(conversation_id),
    ).order_by(AssistantMessage.sequence)).scalars())
