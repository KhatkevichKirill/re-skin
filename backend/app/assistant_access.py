"""Private-operator assistant scope; no provider, endpoint, or file access.

The server injects explicit project scope and a trusted artifact availability
check. Empty scope denies everything unless trusted all-project mode is enabled;
malformed scope always denies everything. Catalog queries
resolve ownership from persisted records, including the live Run/Project, and
return scalar metadata without storage keys, local paths, URLs, or credentials.
This module is not named-user authorization for the shared dashboard login.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select, true
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from .assistant_models import AssistantArtifact

_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_CATALOG_PAGE = 100


class AssistantAccessDenied(ValueError):
    """Generic denial; callers must not expose whether an unscoped ID exists."""


def canonical_uuid(value: object) -> str:
    """Normalize a hyphenated UUID; reject paths, wildcards and compact aliases."""
    if isinstance(value, UUID):
        return str(value)
    if not isinstance(value, str) or _UUID.fullmatch(value) is None:
        raise ValueError("expected a hyphenated UUID")
    return str(UUID(value))


def _duration(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


@dataclass(frozen=True, init=False)
class ProjectAllowlist:
    """Immutable server-supplied scope; parse a CSV string or explicit collection.

    Parsing is atomic: one invalid entry denies the whole scope, rather than
    retaining a possibly unintended valid prefix. Only the trusted literal
    all_projects=True switch permits existing and future projects. References
    are normalized the same way as configured project IDs.
    """

    project_ids: frozenset[str]
    all_projects: bool

    def __init__(self, configured_project_ids: object = None, *, all_projects: object = False):
        valid_configuration = True
        try:
            if configured_project_ids is None:
                entries = ()
            elif isinstance(configured_project_ids, str):
                entries = (
                    tuple(entry.strip() for entry in configured_project_ids.split(","))
                    if configured_project_ids.strip() else ()
                )
            elif isinstance(configured_project_ids, (list, tuple, set, frozenset)):
                entries = configured_project_ids
            else:
                raise ValueError("unsupported project scope configuration")
            projects = frozenset(canonical_uuid(entry) for entry in entries)
        except (TypeError, ValueError):
            projects = frozenset()
            valid_configuration = False
        object.__setattr__(self, "project_ids", projects)
        object.__setattr__(self, "all_projects", all_projects is True and valid_configuration)

    @classmethod
    def from_settings(cls, settings):
        return cls(getattr(settings, "ASSISTANT_PROJECT_IDS", ""),
            all_projects=getattr(settings, "ASSISTANT_ALL_PROJECTS", False))

    @property
    def has_scope(self) -> bool:
        return self.all_projects or bool(self.project_ids)

    def allows_project(self, project_id: object) -> bool:
        try:
            self.require_project(project_id)
        except AssistantAccessDenied:
            return False
        return True

    def sql_filter(self, project_id_column):
        return true() if self.all_projects else project_id_column.in_(self.project_ids)

    def require_project(self, project_id: object) -> str:
        try:
            project_id = canonical_uuid(project_id)
        except ValueError:
            raise AssistantAccessDenied("assistant access denied") from None
        if not self.all_projects and project_id not in self.project_ids:
            raise AssistantAccessDenied("assistant access denied")
        return project_id

    def require_target_and_donors(
        self, target_project_id: object, donor_project_ids: Iterable[object] = (),
    ) -> tuple[str, tuple[str, ...]]:
        target = self.require_project(target_project_id)
        if isinstance(donor_project_ids, (str, bytes)):
            raise AssistantAccessDenied("assistant access denied")
        try:
            donors = tuple(self.require_project(project) for project in donor_project_ids)
        except TypeError:
            raise AssistantAccessDenied("assistant access denied") from None
        return target, donors


@dataclass(frozen=True)
class CatalogVersion:
    version_id: str
    run_id: str
    project_id: str


@dataclass(frozen=True)
class CatalogArtifact:
    asset_id: str
    version_id: str
    run_id: str
    project_id: str
    media_kind: str
    sha256: str
    size_bytes: int
    duration_sec: float | None
    has_audio: bool | None


@dataclass(frozen=True)
class CatalogPage:
    artifacts: tuple[CatalogArtifact, ...]
    next_after_asset_id: str | None


class AuthorizedArtifactCatalog:
    """Read-only metadata queries over authorized immutable artifact ownership.

    ``is_available`` is trusted backend code, never planner data. It should
    verify the pinned artifact's existence/integrity through the storage layer.
    Failure, exception, or a non-boolean result denies that artifact. Availability
    and policy must be checked again when accepting/executing a job; a returned
    descriptor is not a durable capability or a substitute for byte pinning.
    The supplied session must have no unrelated pending application changes.
    """

    def __init__(
        self,
        session: Session,
        *,
        policy: ProjectAllowlist,
        is_available: Callable[[AssistantArtifact], bool],
    ):
        # Lazy imports keep policy use independent of application model setup.
        from .assistant_models import AssistantArtifact, AssistantConversation, AssistantVersion
        from .models import Run, VideoProject

        self._session = session
        self._policy = policy
        self._is_available = is_available
        self._artifact = AssistantArtifact
        self._conversation = AssistantConversation
        self._version = AssistantVersion
        self._run = Run
        self._project = VideoProject

    @staticmethod
    def _id(value: object) -> str:
        try:
            return canonical_uuid(value)
        except ValueError:
            raise AssistantAccessDenied("assistant access denied") from None

    def _scope(self, statement):
        version, conversation = self._version, self._conversation
        run, project = self._run, self._project
        return (
            statement.join(conversation, version.conversation_id == conversation.id)
            .join(run, (run.id == conversation.run_id) & (run.project_id == conversation.project_id))
            .join(project, project.id == conversation.project_id)
            .where(
                version.status == "succeeded",
                self._policy.sql_filter(conversation.project_id),
            )
            .execution_options(populate_existing=True, autoflush=False)
        )

    def require_target(self, version_id: object) -> CatalogVersion:
        version_id = self._id(version_id)
        version, conversation = self._version, self._conversation
        statement = self._scope(select(
            version.id, conversation.run_id, conversation.project_id,
        ).select_from(version)).where(version.id == version_id)
        row = self._session.execute(statement).one_or_none()
        if row is None:
            raise AssistantAccessDenied("assistant access denied")
        return CatalogVersion(self._id(row[0]), self._id(row[1]), self._policy.require_project(row[2]))

    def require_editable_target(self, version_id):
        target = self.require_target(version_id)
        from .assistant_store import get_version_artifact
        from .assistant_preparation import require_editable_artifact
        require_editable_artifact(self._session, get_version_artifact(self._session, target.version_id))
        return target

    def resolve_editable_reference(self, *, target_version_id, asset_id, version_id):
        self.require_editable_target(target_version_id)
        descriptor = self.resolve_reference(target_version_id=target_version_id, asset_id=asset_id, version_id=version_id)
        from .assistant_preparation import require_editable_artifact
        require_editable_artifact(self._session, self._session.get(self._artifact, descriptor.asset_id))
        return descriptor

    def list_editable_artifacts(self, **kwargs):
        self.require_editable_target(kwargs["target_version_id"])
        page = self.list_artifacts(**kwargs)
        from .assistant_preparation import require_editable_artifact, PreparationCapabilityError
        editable = []
        for item in page.artifacts:
            try:
                require_editable_artifact(self._session, self._session.get(self._artifact, item.asset_id))
            except PreparationCapabilityError:
                continue
            editable.append(item)
        return CatalogPage(tuple(editable), page.next_after_asset_id)

    def _artifacts(self):
        artifact, version, conversation = self._artifact, self._version, self._conversation
        return self._scope(select(
            artifact, conversation.run_id, conversation.project_id,
        ).select_from(artifact).join(version, artifact.owner_version_id == version.id))

    def _describe(self, row) -> CatalogArtifact:
        artifact, run_id, project_id = row
        # Availability is checked on every read, with no cross-scope cache.
        try:
            available = self._is_available(artifact) is True
        except Exception:
            available = False
        if not available:
            raise AssistantAccessDenied("assistant access denied")
        if (artifact.media_kind not in ("video", "audio", "image")
                or not isinstance(artifact.sha256, str)
                or _SHA256.fullmatch(artifact.sha256) is None
                or type(artifact.size_bytes) is not int or artifact.size_bytes <= 0):
            raise AssistantAccessDenied("assistant access denied")
        metadata = artifact.metadata_json if isinstance(artifact.metadata_json, dict) else {}
        duration = _duration(metadata.get("duration_sec"))
        audio = metadata.get("has_audio")
        return CatalogArtifact(
            asset_id=self._id(artifact.id), version_id=self._id(artifact.owner_version_id),
            run_id=self._id(run_id), project_id=self._policy.require_project(project_id),
            media_kind=artifact.media_kind, sha256=artifact.sha256,
            size_bytes=artifact.size_bytes, duration_sec=duration,
            has_audio=audio if type(audio) is bool else None,
        )

    def resolve_reference(
        self, *, target_version_id: object, asset_id: object, version_id: object,
    ) -> CatalogArtifact:
        """Require target and donor access, plus exact original owner binding.

        An input/output link from another version does not transfer ownership.
        Missing, denied, unsuccessful, deleted and unavailable references share
        one denial result, without returning donor metadata or a storage key.
        """
        self.require_target(target_version_id)
        asset_id, version_id = self._id(asset_id), self._id(version_id)
        statement = self._artifacts().where(
            self._artifact.id == asset_id,
            self._artifact.owner_version_id == version_id,
        )
        row = self._session.execute(statement).one_or_none()
        if row is None:
            raise AssistantAccessDenied("assistant access denied")
        return self._describe(row)

    def list_artifacts(
        self, *, target_version_id: object, project_id: object = None,
        after_asset_id: object = None, limit: int = 50,
    ) -> CatalogPage:
        """List allowlisted donor metadata with bounded availability checks.

        ``limit`` bounds candidate rows/probes, not the number of available
        results. Continue using the cursor even if a page is empty. Requests
        cannot expand scope; an explicit denied project fails rather than being
        ignored. No user-controlled SQL, filenames, URLs or metadata are queried.
        """
        self.require_target(target_version_id)
        if type(limit) is not int or not 1 <= limit <= MAX_CATALOG_PAGE:
            raise ValueError("catalog limit is outside the supported range")
        statement = self._artifacts()
        if project_id is not None:
            project_id = self._policy.require_project(project_id)
            statement = statement.where(self._conversation.project_id == project_id)
        if after_asset_id is not None:
            statement = statement.where(self._artifact.id > self._id(after_asset_id))
        rows = self._session.execute(statement.order_by(self._artifact.id).limit(limit)).all()
        available = []
        for row in rows:
            try:
                available.append(self._describe(row))
            except AssistantAccessDenied:
                continue
        cursor = self._id(rows[-1][0].id) if len(rows) == limit else None
        return CatalogPage(tuple(available), cursor)
