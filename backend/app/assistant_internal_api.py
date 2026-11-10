"""Finite read-only internal metadata tools for the durable assistant loop.

No HTTP transport, arbitrary query, media import/probe, or application write.
Inventory handles are provenance, never authority to inspect or edit media.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING
from sqlalchemy import select
from sqlalchemy.orm import Session

from .assistant_access import AssistantAccessDenied, AuthorizedArtifactCatalog
from .assistant_agent_contracts import (
    AssetsReadPage, InternalAssetRow, InternalProjectRow, InternalRunRow,
    InternalVersionRow, ProjectsReadPage, RunsReadPage, VersionsReadPage,
    InternalReadPage, ListProjectsAction, ListRunsAction, ListVersionsAction, ListAssetsAction,
    internal_read_cursor, internal_read_dependencies,
)
from .assistant_models import AssistantArtifact, AssistantConversation, AssistantVersion
from .assistant_planner import InternalReadDependency, PlannerContext
from .assistant_store import verify_artifact
from .models import Run, VideoProject

PAGE_SIZE = 20
if TYPE_CHECKING:
    from .assistant_turns import AssistantTurnService

InternalReadAction = ListProjectsAction | ListRunsAction | ListVersionsAction | ListAssetsAction


def extend_dependencies(context: PlannerContext, dependencies: tuple[InternalReadDependency, ...]) -> PlannerContext:
    values = tuple(dict.fromkeys((*context.internal_read_dependencies, *dependencies)))
    payload = context.model_dump(mode='python')
    payload['internal_read_dependencies'] = values
    return PlannerContext.model_validate(payload)


def reauthorize_dependencies(session: Session, *, service: AssistantTurnService,
        dependencies: tuple[InternalReadDependency, ...], target_version_id: str, verify_bytes: bool = False, cancelled=None) -> None:
    """Fresh metadata ownership checks; inventory grants no media authority.

    Metadata history must not repeatedly hash every listed asset. The initial
    list_assets read verifies bytes with cancellation; later media use resolves
    and verifies its own sources independently.
    """
    def available(artifact):
        if verify_bytes:
            verify_artifact(artifact, storage=service.storage, cancelled=cancelled)
        return True
    catalog = AuthorizedArtifactCatalog(session, policy=service.policy, is_available=available)
    catalog.require_target(target_version_id)
    for dependency in dependencies:
        if cancelled:
            cancelled()
        project_id = service.policy.require_project(dependency.project_id)
        if session.scalar(select(VideoProject.id).where(VideoProject.id == project_id)) is None:
            raise AssistantAccessDenied('assistant access denied')
        if dependency.run_id is not None:
            run = service.require_run(session, dependency.run_id)
            if run.project_id != project_id:
                raise AssistantAccessDenied('assistant access denied')
        if dependency.version_id is not None:
            owner = catalog.require_target(dependency.version_id)
            if (owner.run_id, owner.project_id) != (dependency.run_id, project_id):
                raise AssistantAccessDenied('assistant access denied')
        if dependency.asset_id is not None:
            owner = catalog.resolve_reference(target_version_id=target_version_id,
                asset_id=dependency.asset_id, version_id=dependency.version_id)
            if (owner.run_id, owner.project_id) != (dependency.run_id, project_id):
                raise AssistantAccessDenied('assistant access denied')
        if cancelled:
            cancelled()


def _page(statement, identifier, cursor, session):
    if cursor is not None:
        statement = statement.where(identifier > cursor)
    rows = session.execute(statement.order_by(identifier).limit(PAGE_SIZE).execution_options(autoflush=False)).all()
    return rows, rows[-1][0] if len(rows) == PAGE_SIZE else None


def read_internal_metadata(session: Session, *, service: AssistantTurnService, context: PlannerContext,
        action: InternalReadAction, cancelled=None) -> tuple[InternalReadPage, PlannerContext]:
    """Authorize pinned context and exact requested owner on every call."""
    service.reauthorize_context(session, context)
    if (action.run_id, action.base_version_id) != (context.run_id, context.base_version_id):
        raise AssistantAccessDenied('assistant access denied')
    # Hash checks obey the existing turn cancellation/deadline between chunks.
    catalog = AuthorizedArtifactCatalog(session, policy=service.policy,
        is_available=lambda artifact: bool(verify_artifact(artifact, storage=service.storage, cancelled=cancelled)))
    target = catalog.require_target(context.base_version_id)
    if target.run_id != context.run_id:
        raise AssistantAccessDenied('assistant access denied')
    cursor = internal_read_cursor(action)
    if cancelled:
        cancelled()
    if action.kind == 'list_projects':
        owner = InternalReadDependency(kind='project', project_id=target.project_id)
        rows, following = _page(select(VideoProject.id, VideoProject.project_type, VideoProject.status).where(
            service.policy.sql_filter(VideoProject.id)), VideoProject.id, cursor, session)
        page = ProjectsReadPage(kind=action.kind, owner=owner, next_after_id=following,
            records=tuple(InternalProjectRow(project_id=row[0], project_type=row[1], status=row[2].value) for row in rows))
    elif action.kind == 'list_runs':
        owner = InternalReadDependency(kind='project', project_id=service.policy.require_project(action.project_id))
        reauthorize_dependencies(session, service=service, dependencies=(owner,), target_version_id=context.base_version_id)
        rows, following = _page(select(Run.id, Run.project_id, Run.status, Run.language).join(
            VideoProject, VideoProject.id == Run.project_id).where(Run.project_id == owner.project_id), Run.id, cursor, session)
        page = RunsReadPage(kind=action.kind, owner=owner, next_after_id=following,
            records=tuple(InternalRunRow(run_id=row[0], project_id=row[1], status=row[2].value,
                language=row[3] if row[3] in ('en', 'es', 'pt', 'ja', 'de', 'it', 'pl', 'fr') else None) for row in rows))
    elif action.kind == 'list_versions':
        run = service.require_run(session, action.source_run_id)
        owner = InternalReadDependency(kind='run', project_id=run.project_id, run_id=run.id)
        rows, following = _page(select(AssistantVersion.id, AssistantVersion.sequence, AssistantVersion.parent_version_id).join(
            AssistantConversation, AssistantConversation.id == AssistantVersion.conversation_id).where(
                AssistantConversation.run_id == run.id, AssistantConversation.project_id == run.project_id,
                AssistantVersion.status == 'succeeded'), AssistantVersion.id, cursor, session)
        page = VersionsReadPage(kind=action.kind, owner=owner, next_after_id=following,
            records=tuple(InternalVersionRow(version_id=row[0], run_id=run.id, project_id=run.project_id,
                sequence=row[1], parent_version_id=row[2]) for row in rows))
    elif action.kind == 'list_assets':
        version = catalog.require_target(action.source_version_id)
        owner = InternalReadDependency(kind='version', project_id=version.project_id, run_id=version.run_id, version_id=version.version_id)
        # Constrain rows to this exact owner before touching artifact bytes.
        rows, following = _page(select(AssistantArtifact.id).where(
            AssistantArtifact.owner_version_id == version.version_id,
            AssistantArtifact.media_kind.in_(('video', 'audio', 'image'))), AssistantArtifact.id, cursor, session)
        records = []
        for row in rows:
            if cancelled:
                cancelled()
            try:
                descriptor = catalog.resolve_reference(target_version_id=context.base_version_id,
                    asset_id=row[0], version_id=version.version_id)
            except AssistantAccessDenied:
                continue
            records.append(InternalAssetRow(**asdict(descriptor)))
        page = AssetsReadPage(kind=action.kind, owner=owner, next_after_id=following, records=tuple(records))
    else:
        raise ValueError('unsupported internal metadata action')
    dependencies = internal_read_dependencies(page)
    reauthorize_dependencies(session, service=service, dependencies=dependencies,
        target_version_id=context.base_version_id, cancelled=cancelled)
    service.reauthorize_context(session, context)
    return page, extend_dependencies(context, dependencies)
