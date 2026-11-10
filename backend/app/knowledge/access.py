"""Knowledge visibility (design §6.5).

``KnowledgeScope`` = the assistant's trusted ``ProjectAllowlist`` narrowed by
``KNOWLEDGE_PROJECT_IDS``; knowledge can never see a project the assistant
boundary cannot. Scope is always resolved from the owning project of a stored
row, never from a body id.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import false, true

from ..assistant_access import AssistantAccessDenied, ProjectAllowlist, canonical_uuid


class KnowledgeAccessDenied(Exception):
    """Rendered as 404 so out-of-scope ids are indistinguishable from missing ones."""


@dataclass(frozen=True)
class KnowledgeScope:
    assistant: ProjectAllowlist
    project_ids: frozenset
    whole_assistant_scope: bool

    @classmethod
    def from_settings(cls, settings):
        return cls.build(ProjectAllowlist.from_settings(settings),
                         getattr(settings, "KNOWLEDGE_PROJECT_IDS", ""))

    @classmethod
    def build(cls, assistant: ProjectAllowlist, configured: object):
        configured = configured if isinstance(configured, str) else ""
        if configured.strip() == "*":
            return cls(assistant, frozenset(), True)
        try:
            ids = frozenset(canonical_uuid(p.strip()) for p in configured.split(",") if p.strip())
        except (TypeError, ValueError):
            ids = frozenset()  # one bad entry denies the whole list, like ProjectAllowlist
        return cls(assistant, ids, False)

    @property
    def everything(self) -> bool:
        """True only when every current and future project is visible."""
        return self.whole_assistant_scope and self.assistant.all_projects

    def allows_project(self, project_id) -> bool:
        if project_id is None:
            return self.everything
        try:
            pid = canonical_uuid(project_id)
        except (TypeError, ValueError):
            return False
        if not self.assistant.allows_project(pid):
            return False
        return self.whole_assistant_scope or pid in self.project_ids

    def require_project(self, project_id) -> None:
        if not self.allows_project(project_id):
            raise KnowledgeAccessDenied("knowledge resource unavailable")

    def sql_filter(self, project_id_column):
        if self.everything:
            return true()
        if self.whole_assistant_scope:
            return self.assistant.sql_filter(project_id_column)
        ids = [p for p in self.project_ids if self.assistant.allows_project(p)]
        return project_id_column.in_(ids) if ids else false()

    def rule_visibility(self, *, shareable: bool, scope_project_ids, evidence_project_ids) -> str | None:
        """'full', 'shareable' (out-of-scope citations become a count) or None.

        A non-shareable version is suppressed entirely unless every scoped and
        every cited project is visible; its text may embed project detail.
        """
        projects = set(scope_project_ids or ()) | set(evidence_project_ids)
        if all(self.allows_project(p) for p in projects):
            return "full"
        return "shareable" if shareable else None


__all__ = ["KnowledgeAccessDenied", "KnowledgeScope", "AssistantAccessDenied"]
