"""Pin the selected Run's real project input for agent inspection.

This adds an immutable catalog artifact, never a version or selection. Caller
owns the transaction. Model input cannot choose a filesystem path or project.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from sqlalchemy import select

from .assistant_access import AssistantAccessDenied, AuthorizedArtifactCatalog, canonical_uuid
from .assistant_models import AssistantArtifact, AssistantVersion
from .assistant_planner import PlannerSource
from .assistant_store import AssistantStoreError, verify_artifact
from . import assistant_media as media
from .models import VideoProject


class SourceCapabilityUnavailable(ValueError):
    """The real input exists but is outside the supported inspection profile."""


def import_original_source(session, *, run_id, base_version_id, storage, policy,
                           cancelled=None, timeout_sec=60) -> PlannerSource | None:
    budget = media._budget(timeout_sec, cancelled)
    catalog = AuthorizedArtifactCatalog(session, policy=policy,
        is_available=lambda artifact: bool(verify_artifact(artifact, storage=storage, cancelled=cancelled)))
    target = catalog.require_editable_target(base_version_id)
    if target.run_id != canonical_uuid(run_id):
        raise AssistantAccessDenied("assistant access denied")
    # Serialize import for this owner; the input snapshot stays stable even if
    # a legacy source file is later replaced. Source bytes are never overwritten.
    version = session.scalar(select(AssistantVersion).where(
        AssistantVersion.id == target.version_id).with_for_update())
    existing = session.scalars(select(AssistantArtifact).where(
        AssistantArtifact.owner_version_id == version.id,
        AssistantArtifact.media_kind == "video").order_by(AssistantArtifact.created_at)).all()
    for item in existing:
        if item.metadata_json.get("source_role") == "original_project_source":
            verify_artifact(item, storage=storage, cancelled=cancelled)
            budget.check()
            return _descriptor(item, target)
    project = session.get(VideoProject, target.project_id, populate_existing=True)
    if project is None or not project.source_local_path:
        return None
    # A missing input may be reported as unavailable only after confinement;
    # out-of-root paths and permission/integrity failures remain fatal.
    try:
        Path(project.source_local_path).resolve(strict=False).relative_to(storage.data_root)
    except (OSError, ValueError):
        raise AssistantStoreError("original source is outside the data root") from None
    try:
        path = storage._source(project.source_local_path)
    except AssistantStoreError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            raise SourceCapabilityUnavailable("original source file is unavailable") from None
        raise
    if not 0 < path.stat().st_size <= min(storage.max_artifact_bytes, 256 * 1024 * 1024):
        raise SourceCapabilityUnavailable("original source exceeds agent import size limit")
    budget.check()
    pinned = storage.pin(path, cancelled=budget.check)
    budget.check()
    # Probe the pinned snapshot, never metadata supplied by the model or the
    # mutable legacy row. Unsupported source profiles fail explicitly.
    pinned_path = storage.verify(pinned, cancelled=cancelled)
    try:
        info = media.probe_trusted_media(pinned_path,
            timeout_sec=max(.001, budget.deadline - time.monotonic()), cancelled=cancelled)
    except (media.AssistantMediaCancelled, media.AssistantMediaTimeout):
        raise
    except media.AssistantMediaError:
        raise SourceCapabilityUnavailable("original source profile cannot be inspected") from None
    if info.video_duration_sec > 120:
        raise SourceCapabilityUnavailable("original source exceeds agent duration limit")
    budget.check()
    artifact = AssistantArtifact(id=str(uuid.uuid4()), owner_version_id=version.id,
        media_kind="video", storage_key=pinned.storage_key, sha256=pinned.sha256,
        size_bytes=pinned.size_bytes, metadata_json={
            "source_role": "original_project_source", "duration_sec": info.video_duration_sec,
            "fps": info.fps, "has_audio": info.has_audio})
    session.add(artifact)
    session.flush()
    catalog.require_editable_target(base_version_id)
    return _descriptor(artifact, target)


def _descriptor(artifact, target):
    metadata = artifact.metadata_json
    return PlannerSource(asset_id=artifact.id, version_id=target.version_id,
        run_id=target.run_id, project_id=target.project_id, media_kind="video",
        duration_sec=metadata["duration_sec"], has_audio=metadata["has_audio"],
        fps=metadata["fps"], label="Original project source video")
